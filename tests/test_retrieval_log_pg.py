"""Postgres integration tests of the retrieval log and the conversation
retrieval endpoint (6.5).

Run through the fresh-database runner (MIRIX_PG_URI must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_retrieval_log_pg.py

What only Postgres can show (the SQLite path is in tests/test_retrieval_log.py
and tests/test_retrieve_conversation_api.py):

* the row's JSON columns take the full per-type trace, with non-finite scores
  stored as null (Postgres rejects NaN in JSON), and its TIMESTAMPTZ drives
  the newest-first listing and the retention purge;
* the foreign keys hold: a row needs its organization and user;
* a query carrying a NUL (U+0000), which a Postgres text column rejects, is
  stored with U+FFFD instead of losing the row;
* ``POST /memory/retrieve/conversation`` over the real Postgres managers — the
  skill ranking on ``retrieval_tsv`` and pgvector, the gate — writes one row
  whose trace names the Postgres backend, and erasing the user's memories
  deletes it.

The topic-extraction model is the offline fake server; nothing reaches a
model provider.
"""

import datetime as dt
import json
import math
import uuid
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]

SCOPE = "test"
DIM = 4096
T0 = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)


def vec(cosine):
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))] + [0.0] * (DIM - 2)


async def _execute(sql: str, **params):
    from mirix.server.server import engine

    async with engine.begin() as conn:
        return await conn.execute(text(sql), params)


async def _scalar(sql: str, **params):
    from mirix.server.server import engine

    async with engine.connect() as conn:
        return (await conn.execute(text(sql), params)).scalar()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def schema():
    from mirix.orm.base import Base
    from mirix.server.server import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def t(schema):
    org = f"org-{uuid.uuid4().hex[:8]}"
    users = [f"user-{uuid.uuid4().hex[:8]}" for _ in range(2)]
    await _execute("INSERT INTO organizations (id, name) VALUES (:o, :o)", o=org)
    for uid in users:
        await _execute(
            "INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
            "VALUES (:u, :u, :o, 'active', 'UTC', FALSE)",
            u=uid,
            o=org,
        )
    return SimpleNamespace(org=org, user=users[0], other_user=users[1])


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now


def _kwargs(t, **overrides):
    values = dict(
        organization_id=t.org,
        user_id=t.user,
        client_id="client-pg",
        session_id="task-pg",
        memory_types=["procedural"],
        query_text="USER: q",
        extracted_topics="q",
        extraction_status="ok",
        embedding_status="ok",
        per_type={"procedural": {"candidates": [{"id": "a", "cosine": 0.5}]}},
        llm_usage={"topic_extraction": {"prompt_tokens": 1}},
        latency_ms=3,
        retry_events=[],
        embedding_chunked_calls=0,
    )
    values.update(overrides)
    return values


async def test_rows_round_trip_with_null_for_non_finite_scores(t):
    from mirix.services.retrieval_log_manager import RetrievalLogManager

    manager = RetrievalLogManager(clock=Clock())
    per_type = {
        "procedural": {
            "candidates": [
                {
                    "id": f"s{i}",
                    "cosine": math.nan if i == 0 else i / 10,
                    "bm25": math.inf,
                }
                for i in range(200)
            ]
        }
    }
    log_id = await manager.create_log(
        **_kwargs(t, per_type=per_type, embedding_chunked_calls=4)
    )
    stored = await _scalar(
        "SELECT per_type::text FROM retrieval_log WHERE id = :i", i=log_id
    )
    candidates = json.loads(stored)["procedural"]["candidates"]
    assert len(candidates) == 200
    assert candidates[0]["cosine"] is None and candidates[1]["cosine"] == 0.1
    assert all(c["bm25"] is None for c in candidates)
    [row] = await manager.list_logs(organization_id=t.org, client_id="client-pg")
    assert row.id == log_id and row.embedding_chunked_calls == 4
    assert row.created_at.astimezone(dt.timezone.utc) == T0


async def test_a_row_needs_its_user(t):
    """No row for a user that does not exist or was deleted: the
    retrieval that raced the user's deletion writes nothing (no FK error)."""
    from mirix.services.retrieval_log_manager import RetrievalLogManager

    manager = RetrievalLogManager()
    assert await manager.create_log(**_kwargs(t, user_id="user-missing")) is None
    assert (
        await _scalar(
            "SELECT count(*) FROM retrieval_log WHERE user_id = 'user-missing'"
        )
        == 0
    )
    # The foreign key to users still holds.
    assert (
        await _scalar(
            "SELECT count(*) FROM information_schema.referential_constraints rc "
            "JOIN information_schema.key_column_usage k "
            "ON k.constraint_name = rc.constraint_name "
            "WHERE k.table_name = 'retrieval_log' AND k.column_name = 'user_id'"
        )
        >= 1
    )
    await _execute("UPDATE users SET is_deleted = TRUE WHERE id = :u", u=t.other_user)
    assert await manager.create_log(**_kwargs(t, user_id=t.other_user)) is None


async def test_a_query_with_nul_is_stored_with_u_fffd(t):
    from mirix.services.retrieval_log_manager import RetrievalLogManager

    log_id = await RetrievalLogManager().create_log(
        **_kwargs(
            t,
            client_id="client-pg-nul",
            query_text="TOOL: PK\x03\x04\x00",
            extracted_topics="zip\x00",
        )
    )
    assert log_id is not None
    stored = await _scalar(
        "SELECT query_text || '|' || extracted_topics FROM retrieval_log "
        "WHERE id = :i",
        i=log_id,
    )
    assert stored == "TOOL: PK\x03\x04\ufffd|zip\ufffd"
    usage = await _scalar(
        "SELECT llm_usage::text FROM retrieval_log WHERE id = :i", i=log_id
    )
    assert json.loads(usage)["nul_chars_replaced"] == 2


async def test_listing_order_and_the_retention_purge(t):
    from mirix.services.retrieval_log_manager import RetrievalLogManager

    clock = Clock()
    manager = RetrievalLogManager(clock=clock)
    client = f"client-{uuid.uuid4().hex[:6]}"
    ids = {}
    for age in (45, 31, 10, 0):
        clock.now = T0 - dt.timedelta(days=age)
        ids[age] = await manager.create_log(**_kwargs(t, client_id=client))
    clock.now = T0
    listed = await manager.list_logs(organization_id=t.org, client_id=client)
    assert [log.id for log in listed] == [ids[0], ids[10], ids[31], ids[45]]
    assert await manager.purge_expired(retention_days=30) >= 2
    left = await manager.list_logs(organization_id=t.org, client_id=client)
    assert [log.id for log in left] == [ids[0], ids[10]]


async def _endpoint_server(t, monkeypatch, fake_llm_server, **managers):
    """The retrieval endpoints over the real Postgres managers (procedural memory,
    users, the retrieval log, and ``managers``), the topic-extraction model on the
    offline fake server and a fake query vector: (client, headers, add a skill)."""
    import mirix.embeddings as embeddings_module
    import mirix.services.procedural_memory_manager as pmm
    from mirix.schemas.agent import AgentState, AgentType
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.server import rest_api
    from mirix.services.procedural_memory_manager import ProceduralMemoryManager
    from mirix.services.retrieval_log_manager import RetrievalLogManager
    from mirix.services.user_manager import UserManager

    client = PydanticClient(
        id=f"client-{uuid.uuid4().hex[:8]}",
        organization_id=t.org,
        name="c",
        write_scope=SCOPE,
        read_scopes=[SCOPE],
    )
    agent_state = AgentState(
        id="agent-pg",
        name="meta",
        system="s",
        agent_type=AgentType.meta_memory_agent,
        llm_config=LLMConfig(
            model="fake-model",
            model_endpoint_type="openai",
            model_endpoint=fake_llm_server.base_url,
            context_window=200_000,
            max_output_tokens=16_000,
            api_key="fake-key",
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=2,
        ),
        tools=[],
    )

    class Embedder:
        def __init__(self):
            self.calls = []

        async def get_text_embedding(self, text_value):
            self.calls.append(text_value)
            return [1.0, 0.0]

    embedder = Embedder()

    async def fake_embedding_model(config, *args, **kwargs):
        return embedder

    async def no_sleep(_s):
        return None

    class Clients:
        async def get_client_by_id(self, cid):
            return client if cid == client.id else None

    class Agents:
        async def list_agents(self, actor, limit=None):
            return [agent_state]

    server = SimpleNamespace(
        client_manager=Clients(),
        agent_manager=Agents(),
        user_manager=UserManager(),
        procedural_memory_manager=ProceduralMemoryManager(sleep=no_sleep),
        retrieval_log_manager=RetrievalLogManager(),
        **managers,
    )
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(embeddings_module, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(rest_api, "get_server", lambda: server)

    async def add(skill_id, name, cosine, triggers):
        await _execute(
            "INSERT INTO procedural_memory (id, organization_id, user_id, name, entry_type, "
            "description, instructions, triggers, examples, version, filter_tags, "
            "last_modify, is_deleted, retrieval_embedding) VALUES (:id, :o, :u, :n, "
            "'workflow', '', 'Do it.', CAST(:tr AS json), '[]', '0.1.0', "
            "CAST(:ft AS json), '{}', FALSE, CAST(CAST(:v AS text) AS vector))",
            id=skill_id,
            o=t.org,
            u=t.user,
            n=name,
            tr=json.dumps(triggers),
            ft=json.dumps({"scope": SCOPE}),
            v=json.dumps(vec(cosine)),
        )

    headers = {"x-client-id": client.id, "x-org-id": t.org}
    return SimpleNamespace(
        client=client, headers=headers, add=add, embedder=embedder, app=rest_api.app
    )


async def test_the_endpoint_writes_one_row_and_erasure_deletes_it(
    t, monkeypatch, fake_llm_server
):
    from mirix.services import topic_extraction as te
    from mirix.services.user_manager import UserManager

    api = await _endpoint_server(t, monkeypatch, fake_llm_server)
    add = api.add
    good = f"proc-{uuid.uuid4().hex[:8]}"
    await add(good, "export report", 0.92, ["export a report as csv"])
    await add(f"proc-{uuid.uuid4().hex[:8]}", "cook pasta", 0.4, ["dinner"])
    fake_llm_server.queue_tool_call(
        te.EXTRACTION_FUNCTION_NAME, {"topic": "export report; csv"}
    )

    transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
    headers = api.headers
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/memory/retrieve/conversation",
            json={
                "user_id": t.user,
                "messages": [{"role": "user", "content": "Export the report as CSV"}],
                "memory_types": ["procedural"],
                "session_id": "task-pg-1",
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]
        listed = (
            await http.get(
                "/memory/retrieval_logs",
                params={"session_id": "task-pg-1"},
                headers=headers,
            )
        ).json()["retrieval_logs"]

    [row] = listed
    assert row["id"] == body["retrieval_id"]
    trace = row["per_type"]["procedural"]
    assert trace["backend"] == "postgresql" and trace["returned_ids"] == [good]
    assert trace["bm25"]["query_terms"]  # lexemes of to_tsvector('english', topics)
    assert trace["gate"]["applied"] is True and trace["gate"]["passed"] == 1
    assert row["extraction_status"] == "ok" and row["embedding_status"] == "ok"
    assert row["extracted_topics"] == "export report; csv"

    await UserManager().delete_memories_by_user_id(t.user)
    assert (
        await _scalar("SELECT count(*) FROM retrieval_log WHERE user_id = :u", u=t.user)
        == 0
    )


async def test_a_nul_in_the_query_reaches_no_search_on_postgres(
    t, monkeypatch, fake_llm_server
):
    """When topic extraction fails, the conversation's raw text is the query.
    A NUL in it (a tool that printed ``\\x00``) failed every search on Postgres
    (``procedural: DBAPIError (Error): invalid byte sequence for encoding "UTF8":
    0x00``, ``success: false``), the same way on every retry. The query is searched
    with U+FFFD, as MIRIX stores text: by the procedural ranking (``to_tsquery``,
    pgvector) and by the keyword types; the topic endpoint and ``/memory/search`` too."""
    from mirix.services.episodic_memory_manager import EpisodicMemoryManager
    from mirix.services.semantic_memory_manager import SemanticMemoryManager

    api = await _endpoint_server(
        t,
        monkeypatch,
        fake_llm_server,
        semantic_memory_manager=SemanticMemoryManager(),
        episodic_memory_manager=EpisodicMemoryManager(),
    )
    good = f"proc-{uuid.uuid4().hex[:8]}"
    await api.add(good, "export report", 0.92, ["export a report as csv"])
    fake_llm_server.queue_text("I cannot call tools today.")  # no usable topic
    keyword_types = ["procedural", "semantic", "episodic"]
    transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            "/memory/retrieve/conversation",
            json={
                "user_id": t.user,
                "messages": [
                    {"role": "user", "content": "Export the report\x00 as CSV"}
                ],
                "memory_types": keyword_types,
                "session_id": "task-pg-nul",
                "use_cache": False,
            },
            headers=api.headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["extraction_status"] == "fallback"
        assert body["success"] is True and body["errors"] == {}, body
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]
        assert api.embedder.calls == ["Export the report\ufffd as CSV"]

        topic = await http.get(
            "/memory/retrieve/topic",
            params={"user_id": t.user, "topic": "export\x00report", "use_cache": False},
            headers=api.headers,
        )
        assert topic.status_code == 200, topic.text
        # Only the three managers this server has can answer; none failed on the NUL.
        assert not set(topic.json()["errors"]) & set(keyword_types), topic.json()

        search = await http.get(
            "/memory/search",
            params={
                "user_id": t.user,
                "query": "export\x00report",
                "memory_type": "procedural",
            },
            headers=api.headers,
        )
        assert search.status_code == 200, search.text
        found = search.json()
        assert found["success"] is True, found
        assert [r["id"] for r in found["results"]] == [good]

    logged = await _scalar(
        "SELECT query_text FROM retrieval_log WHERE id = :i", i=body["retrieval_id"]
    )
    assert "\x00" not in logged and "\ufffd" in logged
