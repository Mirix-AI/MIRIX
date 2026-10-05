"""POST /memory/retrieve/conversation, GET /memory/retrieve/topic and
GET /memory/retrieval_logs.

The FastAPI app runs in-process over httpx's ASGI transport against a
throwaway SQLite database. The topic-extraction model is the offline fake
model server (``fake_llm_server``) reached through MIRIX's real OpenAI client;
the query vector comes from an in-process fake (a skill's cosine to the query
is the number the test names). Procedural memory, users, files and the
retrieval log are the real managers; the other memory managers are spies, so
a test can prove which of them a retrieval touched. Nothing reaches a model
provider.
"""

import base64
import io
import math
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.embeddings as embeddings_module
import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix import constants
from mirix.client import MirixClient
from mirix.client.render import render_retrieved_memories
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.embedding_guard import record_chunked_embedding_call
from mirix.orm.base import Base
from mirix.orm.file import FileMetadata as FileORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.server import rest_api
from mirix.services import topic_extraction as te
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.retrieval_log_manager import RetrievalLogManager
from mirix.services.user_manager import UserManager

pytestmark = pytest.mark.asyncio(loop_scope="module")

SCOPE = "test"
ALL_TYPES = [
    "episodic",
    "semantic",
    "resource",
    "procedural",
    "knowledge_vault",
    "core",
]


def vec(cosine):
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))] + [0.0] * (
        MAX_EMBEDDING_DIM - 2
    )


def _png(size=(20, 10), color="green") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("retrieve_conversation") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _ctx():
        async with local() as session:
            try:
                yield session
            finally:
                await session.close()

    yield _ctx
    await engine.dispose()


class _QueryEmbedder:
    """Every query embeds to (1, 0, ...). ``fail`` raises; ``chunks`` makes
    each call count as one chunked call (as GuardedEmbedding does)."""

    def __init__(self):
        self.calls = []
        self.fail = False
        self.chunks = 0

    async def get_text_embedding(self, text_value):
        self.calls.append(text_value)
        if self.chunks:
            record_chunked_embedding_call(self.chunks)
        if self.fail:
            raise ConnectionError("embedding endpoint down")
        return [1.0, 0.0]


class _Spy:
    """A memory manager that records every call (and may refuse them)."""

    def __init__(self, name, *, forbid=False, fail=False):
        self.name = name
        self.calls = []
        self.forbid = forbid
        self.fail = fail

    def __getattr__(self, method):
        if method.startswith("_"):
            raise AttributeError(method)

        async def call(*args, **kwargs):
            self.calls.append((method, kwargs))
            if self.forbid:
                raise AssertionError(f"{self.name}.{method} must not be called")
            if self.fail:
                raise RuntimeError(f"{self.name} store is down")
            if method == "get_total_number_of_items":
                return 0
            return []

        return call


@pytest_asyncio.fixture(loop_scope="module")
async def api(sm, monkeypatch, tmp_path, fake_llm_server):
    import mirix.server.server as server_module
    from mirix.settings import settings

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    monkeypatch.setattr(constants, "RETRIEVAL_LOG_ENABLED", True)

    org_id = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    other_user_id = f"user-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        for uid in (user_id, other_user_id):
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org_id,
                    status="active",
                    timezone="UTC",
                )
            )
        await db.commit()

    def pclient(cid):
        return PydanticClient(
            id=cid,
            organization_id=org_id,
            name=cid,
            write_scope=SCOPE,
            read_scopes=[SCOPE],
        )

    client = pclient(f"client-{uuid.uuid4().hex[:8]}")
    other_client = pclient(f"client-{uuid.uuid4().hex[:8]}")
    agent_state = AgentState(
        id="agent-retrieve",
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
    embedder = _QueryEmbedder()

    async def fake_embedding_model(config, *args, **kwargs):
        return embedder

    async def no_sleep(_s):
        return None

    class Clients:
        async def get_client_by_id(self, cid):
            return {client.id: client, other_client.id: other_client}.get(cid)

    class Agents:
        agents = [agent_state]

        async def list_agents(self, actor, limit=None):
            return list(self.agents)

    spies = {
        "episodic": _Spy("episodic"),
        "semantic": _Spy("semantic"),
        "resource": _Spy("resource"),
        "knowledge_vault": _Spy("knowledge_vault"),
        "core": _Spy("core"),
    }
    procedural = ProceduralMemoryManager(session_maker=sm, sleep=no_sleep)
    server = SimpleNamespace(
        client_manager=Clients(),
        agent_manager=Agents(),
        user_manager=UserManager(),
        procedural_memory_manager=procedural,
        episodic_memory_manager=spies["episodic"],
        semantic_memory_manager=spies["semantic"],
        resource_memory_manager=spies["resource"],
        knowledge_vault_manager=spies["knowledge_vault"],
        block_manager=spies["core"],
        retrieval_log_manager=RetrievalLogManager(session_maker=sm),
    )
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(embeddings_module, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(rest_api, "get_server", lambda: server)

    async def add_skill(
        name, *, cosine=None, triggers=(), description="", user=user_id
    ):
        skill_id = f"proc-{uuid.uuid4().hex[:8]}"
        async with sm() as db:
            db.add(
                ProceduralORM(
                    id=skill_id,
                    organization_id=org_id,
                    user_id=user,
                    name=name,
                    entry_type="workflow",
                    description=description,
                    instructions=f"Steps for {name}.",
                    triggers=list(triggers),
                    examples=[],
                    filter_tags={"scope": SCOPE},
                    retrieval_embedding=vec(cosine) if cosine is not None else None,
                )
            )
            await db.commit()
        return skill_id

    transport = httpx.ASGITransport(app=rest_api.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:

        def headers(as_client=None):
            c = as_client or client
            return {"x-client-id": c.id, "x-org-id": org_id}

        async def retrieve(as_client=None, **body):
            body.setdefault("user_id", user_id)
            return await http.post(
                "/memory/retrieve/conversation", json=body, headers=headers(as_client)
            )

        async def logs(as_client=None, **params):
            return await http.get(
                "/memory/retrieval_logs", params=params, headers=headers(as_client)
            )

        yield SimpleNamespace(
            http=http,
            retrieve=retrieve,
            logs=logs,
            headers=headers,
            add_skill=add_skill,
            embedder=embedder,
            spies=spies,
            server=server,
            fake=fake_llm_server,
            sm=sm,
            org=org_id,
            user=user_id,
            other_user=other_user_id,
            client=client,
            other_client=other_client,
            agents=Agents,
            images=tmp_path / "images",
        )


def _topics(api, topic="export report csv", temporal="", usage=None):
    api.fake.queue_tool_call(
        te.EXTRACTION_FUNCTION_NAME,
        {"topic": topic, "temporal_expression": temporal},
        usage=usage
        or {"prompt_tokens": 200, "completion_tokens": 9, "total_tokens": 209},
    )


async def _log_row(api, log_id):
    async with api.sm() as db:
        return (
            await db.execute(
                select(RetrievalLogORM).where(RetrievalLogORM.id == log_id)
            )
        ).scalar_one()


async def _log_count(api):
    async with api.sm() as db:
        return (await db.execute(select(func.count(RetrievalLogORM.id)))).scalar_one()


def _user_text(request):
    content = request.json["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p["type"] == "text")


# What a conversation retrieval for procedural + semantic answers when the
# client's organization has no such user.
_NOTHING_FOUND = {
    "success": True,
    "retrieval_id": None,
    "topics": None,
    "procedural_query": None,
    "temporal_expression": None,
    "extraction_status": "skipped",
    "date_range": None,
    "memories": {
        "procedural": {"total_count": 0, "items": []},
        "semantic": {"total_count": 0, "items": []},
    },
    "errors": {},
}


async def _foreign_user(api) -> str:
    """A user of another organization (with its own organization row)."""
    foreign_org = f"org-{uuid.uuid4().hex[:8]}"
    foreign_user = f"user-{uuid.uuid4().hex[:8]}"
    async with api.sm() as db:
        db.add(OrganizationORM(id=foreign_org, name=foreign_org))
        db.add(
            UserORM(
                id=foreign_user,
                name=foreign_user,
                organization_id=foreign_org,
                status="active",
                timezone="UTC",
            )
        )
        await db.commit()
    return foreign_user


# --------------------------------------------------------------------------- #
# Procedural retrieval through the conversation endpoint                       #
# --------------------------------------------------------------------------- #


class TestProceduralRetrieval:
    async def test_only_the_procedural_manager_is_called(self, api):
        """memory_types=['procedural']: no other memory manager is touched."""
        for spy in api.spies.values():
            spy.forbid = True
        good = await api.add_skill(
            "export-report", cosine=0.9, triggers=["export a report"]
        )
        await api.add_skill("near-miss", cosine=0.7)  # below 0.9 x best
        await api.add_skill("no-vector")  # no retrieval vector yet
        _topics(api)
        response = await api.retrieve(
            messages=[
                {"role": "user", "content": "Please export the Q3 report as CSV"}
            ],
            memory_types=["procedural"],
            session_id="task-001",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["memories"]) == {"procedural"}
        [item] = body["memories"]["procedural"]["items"]
        assert item["id"] == good
        scores = item["retrieval_scores"]
        assert set(scores) == {"bm25", "cosine", "rrf"}
        assert scores["cosine"] == pytest.approx(0.9) and scores["rrf"] > 0
        assert body["retrieval_id"].startswith("rlog-")
        assert body["topics"] == "export report csv"
        assert body["procedural_query"] is None  # a legacy one-string answer
        assert body["extraction_status"] == "ok" and body["errors"] == {}
        assert all(spy.calls == [] for spy in api.spies.values())
        # The extracted topics — not the raw text — were the query.
        assert api.embedder.calls == ["export report csv"]

    async def test_skills_rank_the_task_level_query(self, api):
        """A structured extraction answer sends the kind of task (goal
        and operations) to the skills, and the full topics (the key objects
        too) to the keyword-searched types."""
        good = await api.add_skill(
            "export-report", cosine=0.9, triggers=["export a report"]
        )
        api.fake.queue_tool_call(
            te.EXTRACTION_FUNCTION_NAME,
            {
                "goal": "export a report as a file",
                "operations": "open the report; export it",
                "key_objects": "Q3 report; CSV",
                "temporal_expression": "",
            },
        )
        response = await api.retrieve(
            messages=[{"role": "user", "content": "Export the Q3 report as CSV"}],
            memory_types=["procedural", "semantic"],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        task_query = "export a report as a file; open the report; export it"
        topics = "export a report as a file; Q3 report; CSV; open the report; export it"
        assert body["topics"] == topics
        assert body["procedural_query"] == task_query
        assert api.embedder.calls == [task_query]
        [(_method, kwargs)] = api.spies["semantic"].calls[:1]
        assert kwargs["query"] == topics
        [item] = body["memories"]["procedural"]["items"]
        assert item["id"] == good
        row = await _log_row(api, body["retrieval_id"])
        assert row.extracted_topics == topics
        assert row.llm_usage["topic_extraction"]["procedural_query"] == task_query
        assert row.per_type["procedural"]["bm25"]["query_terms"] == sorted(
            {"export", "file", "open", "report"}
        )

    async def test_the_gate_can_return_no_skill(self, api):
        await api.add_skill("export-report", cosine=0.3, triggers=["export a report"])
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "export the report"}],
                memory_types=["procedural"],
            )
        ).json()
        assert body["memories"]["procedural"]["items"] == []
        assert render_retrieved_memories(body) == ""
        row = await _log_row(api, body["retrieval_id"])
        gate = row.per_type["procedural"]["gate"]
        assert gate["applied"] is True and gate["passed"] == 0
        # The agent's embedding model ("x") has no calibrated gate: the
        # log row says the defaults applied uncalibrated.
        assert gate["calibration"] == "uncalibrated"
        assert (gate["min_similarity"], gate["relative_ratio"]) == (0.62, 0.9)

    @pytest.mark.parametrize(
        "content",
        [
            "Export the Q3 report as CSV",
            [
                {"type": "text", "text": "Export the Q3 report"},
                {"type": "text", "text": "as CSV"},
            ],
        ],
        ids=["string", "list-of-parts"],
    )
    async def test_both_message_formats_reach_the_extractor(self, api, content):
        """String content counted as 'no text' and a list of parts raised
        TypeError — extraction never ran. Both now reach the model."""
        _topics(api, topic="q3 report export")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": content}],
                memory_types=["procedural"],
            )
        ).json()
        assert (
            body["topics"] == "q3 report export" and body["extraction_status"] == "ok"
        )
        text = _user_text(api.fake.last_chat_request())
        assert "Export the Q3 report" in text and "as CSV" in text

    async def test_images_go_to_the_extractor_and_are_never_stored(self, api):
        """The image is sent with the text; the retrieval stores no file."""
        _topics(api, topic="read chart")
        raw = _png()
        body = (
            await api.retrieve(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this chart?"},
                            {"type": "image_url", "image_url": {"url": _data_url(raw)}},
                        ],
                    }
                ],
                memory_types=["procedural"],
            )
        ).json()
        assert body["extraction_status"] == "ok"
        content = api.fake.last_chat_request().json["messages"][-1]["content"]
        [image] = [p for p in content if p["type"] == "image_url"]
        assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == raw
        async with api.sm() as db:
            files = (await db.execute(select(func.count(FileORM.id)))).scalar_one()
        assert files == 0 and not api.images.exists()
        row = await _log_row(api, body["retrieval_id"])
        extraction = row.llm_usage["topic_extraction"]
        assert (
            extraction["input"]["images"] == 1
            and extraction["input"]["image_tokens"] > 0
        )

    async def test_extraction_failure_falls_back_to_the_raw_text(self, api):
        """No usable answer -> the conversation's own text is the query."""
        good = await api.add_skill("export-report", cosine=0.9)
        api.fake.queue_text("I cannot call tools today.")
        body = (
            await api.retrieve(
                messages=[
                    {"role": "user", "content": "Export   the report"},
                    {"role": "assistant", "content": "Which format?"},
                ],
                memory_types=["procedural"],
            )
        ).json()
        assert body["extraction_status"] == "fallback" and body["topics"] is None
        # The query is the messages' text without the role labels:
        # "user" / "assistant" are not what was asked about.
        assert api.embedder.calls == ["Export the report\nWhich format?"]
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]
        row = await _log_row(api, body["retrieval_id"])
        assert row.extraction_status == "fallback" and row.extracted_topics is None
        # The log keeps the conversation as it was (with its roles).
        assert row.query_text == "USER: Export the report\nASSISTANT: Which format?"
        assert "no usable topic" in row.llm_usage["topic_extraction"]["error"]

    async def test_role_labels_are_not_fallback_search_terms(self, api):
        """A skill about users and assistants is not matched by every
        failed extraction's labels."""
        labels = await api.add_skill(
            "user-assistant-handoff", triggers=["user assistant handoff"]
        )
        api.embedder.fail = True  # BM25 only
        api.fake.queue_text("no tool call")
        body = (
            await api.retrieve(
                messages=[
                    {"role": "user", "content": "quarterly invoice totals"},
                    {"role": "assistant", "content": "Which quarter?"},
                ],
                memory_types=["procedural"],
            )
        ).json()
        assert body["extraction_status"] == "fallback"
        ids = [i["id"] for i in body["memories"]["procedural"]["items"]]
        assert labels not in ids

    async def test_keyword_types_get_a_short_fallback_query(self, api, monkeypatch):
        """The other types search the fallback by keywords: a few hundred
        characters, not the whole conversation."""
        monkeypatch.setattr(constants, "RETRIEVAL_FALLBACK_KEYWORD_QUERY_MAX_CHARS", 30)
        api.fake.queue_text("no tool call")
        text = "START " + " ".join(f"word{i}" for i in range(400)) + " END"
        await api.retrieve(
            messages=[{"role": "user", "content": text}],
            memory_types=["semantic", "procedural"],
        )
        [(method, kwargs)] = api.spies["semantic"].calls[:1]
        assert method == "list_semantic_items"
        assert kwargs["query"].startswith("START") and kwargs["query"].endswith(" END")
        assert len(kwargs["query"]) == 31
        [skill_query] = api.embedder.calls
        assert skill_query == text  # skills rank the whole text

    async def test_a_nul_in_a_fallback_query_never_reaches_a_search(self, api):
        """The conversation's raw text is the query when the extraction
        fails; a NUL in it (a tool that printed ``\\x00``) is searched as U+FFFD, as
        MIRIX stores text: Postgres rejects U+0000 in any text parameter
        (tests/test_retrieval_log_pg.py runs this on Postgres)."""
        good = await api.add_skill("export-report", cosine=0.9)
        api.fake.queue_text("no tool call")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "Export the\x00 report"}],
                memory_types=["semantic", "episodic", "procedural"],
            )
        ).json()
        assert body["extraction_status"] == "fallback"
        assert body["success"] is True and body["errors"] == {}
        assert api.embedder.calls == ["Export the\ufffd report"]
        queries = [
            kwargs["query"]
            for name in ("semantic", "episodic")
            for _method, kwargs in api.spies[name].calls
            if "query" in kwargs
        ]
        assert queries == ["Export the\ufffd report"] * 2
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]
        row = await _log_row(api, body["retrieval_id"])
        assert row.query_text == "USER: Export the\ufffd report"

    async def test_a_nul_in_the_extracted_topics_never_reaches_a_search(self, api):
        _topics(api, topic="export\x00report")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "export the report"}],
                memory_types=["semantic", "procedural"],
            )
        ).json()
        assert body["extraction_status"] == "ok" and body["success"] is True
        assert api.embedder.calls == ["export\ufffdreport"]
        [(_method, kwargs)] = api.spies["semantic"].calls[:1]
        assert kwargs["query"] == "export\ufffdreport"

    async def test_a_long_fallback_query_keeps_head_and_tail(self, api, monkeypatch):
        monkeypatch.setattr(constants, "RETRIEVAL_FALLBACK_QUERY_MAX_CHARS", 40)
        api.fake.queue_text("no tool call")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "START " + "x" * 200 + " END"}],
                memory_types=["procedural"],
            )
        ).json()
        [query] = api.embedder.calls
        assert query.startswith("START") and query.endswith(" END")
        assert len(query) == 41  # 40 characters and the joining newline
        row = await _log_row(api, body["retrieval_id"])
        assert row.llm_usage["topic_extraction"]["fallback_query_truncated_chars"] > 0
        assert len(row.query_text) > 200  # the log keeps the whole raw text

    async def test_embedding_failure_falls_back_to_bm25_without_the_gate(self, api):
        match = await api.add_skill("csv-export", cosine=0.1, triggers=["export csv"])
        api.embedder.fail = True
        _topics(api, topic="export csv")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "export csv"}],
                memory_types=["procedural"],
            )
        ).json()
        [item] = body["memories"]["procedural"]["items"]
        assert item["id"] == match and item["retrieval_scores"]["cosine"] is None
        row = await _log_row(api, body["retrieval_id"])
        assert row.embedding_status == "failed"
        assert row.per_type["procedural"]["gate"]["applied"] is False

    async def test_the_default_retrieves_every_memory_type(self, api):
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "export the report"}]
            )
        ).json()
        assert set(body["memories"]) == set(ALL_TYPES)
        for name, spy in api.spies.items():
            assert spy.calls, name
        row = await _log_row(api, body["retrieval_id"])
        assert row.memory_types == ALL_TYPES and set(row.per_type) == set(ALL_TYPES)
        assert row.per_type["semantic"] == {
            "method": "bm25",
            "search_field": "details",
            "limit": 10,
            "returned_ids": [],
        }

    async def test_a_failing_memory_type_is_reported_not_hidden(self, api):
        api.spies["semantic"].fail = True
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}],
                memory_types=["semantic", "procedural"],
            )
        ).json()
        assert body["success"] is True
        assert body["errors"] == {"semantic": "RuntimeError: semantic store is down"}
        assert body["memories"]["semantic"] == {"total_count": 0, "items": []}
        row = await _log_row(api, body["retrieval_id"])
        assert row.per_type["semantic"] == {
            "error": "RuntimeError: semantic store is down"
        }

    async def test_every_requested_type_failing_is_not_success(self, api):
        """Empty results because every requested type FAILED are an
        error, not "nothing matched" (the harness checks ``success``)."""
        api.spies["semantic"].fail = True
        api.spies["episodic"].fail = True
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}],
                memory_types=["semantic", "episodic"],
            )
        ).json()
        assert body["success"] is False
        assert body["error"] == (
            "every requested memory type failed: "
            "semantic: RuntimeError: semantic store is down; "
            "episodic: RuntimeError: episodic store is down"
        )
        assert body["retrieval_id"]  # the failed retrieval is still logged
        assert set(body["errors"]) == {"semantic", "episodic"}

    async def test_a_database_error_is_reported_without_its_sql(self, api):
        """A SQLAlchemy error's text (the statement and its bound
        parameters: the query, ids) never reaches the response or the log."""
        from sqlalchemy.exc import OperationalError

        secret = "my private query about salaries"

        class Broken(_Spy):
            def __getattr__(self, method):
                async def call(*args, **kwargs):
                    raise OperationalError(
                        "SELECT * FROM semantic_memory WHERE details @@ :q",
                        {"q": secret},
                        Exception("database is locked\nDETAIL: user u-1"),
                    )

                return call

        api.server.semantic_memory_manager = Broken("semantic")
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}],
                memory_types=["semantic", "procedural"],
            )
        ).json()
        error = body["errors"]["semantic"]
        assert error == "OperationalError (Exception): database is locked"
        row = await _log_row(api, body["retrieval_id"])
        assert row.per_type["semantic"] == {"error": error}
        assert secret not in str(body) and "SELECT" not in str(body)

    async def test_the_meta_agent_configs_are_used(self, api, monkeypatch):
        """The retrieval uses the meta memory agent's models (the skills
        were embedded with its embedding config), whatever the agent order."""
        chat_agent = AgentState(
            id="agent-chat",
            name="chat",
            system="s",
            agent_type=AgentType.chat_agent,
            llm_config=LLMConfig(
                model="unreachable",
                model_endpoint_type="openai",
                model_endpoint="http://127.0.0.1:9/v1",
                context_window=8000,
                api_key="k",
            ),
            embedding_config=EmbeddingConfig(
                embedding_endpoint_type="openai",
                embedding_endpoint="http://127.0.0.1:9/v1",
                embedding_model="chat-embedder",
                embedding_dim=2,
            ),
            tools=[],
        )
        monkeypatch.setattr(api.agents, "agents", [chat_agent, *api.agents.agents])
        configs = []

        async def recording_model(config, *args, **kwargs):
            configs.append(config.embedding_model)
            return api.embedder

        monkeypatch.setattr(pmm, "embedding_model", recording_model)
        good = await api.add_skill("export-report", cosine=0.9)
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "export the report"}],
                memory_types=["procedural"],
            )
        ).json()
        assert body["extraction_status"] == "ok"
        assert len(api.fake.chat_requests()) == 1
        assert configs == ["x"]  # the meta agent's, not "chat-embedder"
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]

    async def test_a_vault_entry_of_an_unreadable_scope_is_never_returned(
        self, api, monkeypatch
    ):
        """Knowledge vault retrieval is scoped like every other type."""
        from mirix.orm.knowledge_vault import KnowledgeVaultItem as VaultORM
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager

        ids = {}
        async with api.sm() as db:
            for scope in (SCOPE, "other-client-scope"):
                ids[scope] = f"kv-{uuid.uuid4().hex[:8]}"
                db.add(
                    VaultORM(
                        id=ids[scope],
                        organization_id=api.org,
                        user_id=api.user,
                        entry_type="credential",
                        source="test",
                        sensitivity="low",
                        secret_value="s3cret",
                        caption="warehouse portal login",
                        filter_tags={"scope": scope},
                    )
                )
            await db.commit()
        monkeypatch.setattr(
            api.server, "knowledge_vault_manager", KnowledgeVaultManager()
        )
        _topics(api, topic="warehouse portal login")
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "log into the portal"}],
                memory_types=["knowledge_vault"],
                use_cache=False,
            )
        ).json()
        assert body["success"] is True, body
        returned = [i["id"] for i in body["memories"]["knowledge_vault"]["items"]]
        assert returned == [ids[SCOPE]]

    async def test_a_local_model_gets_a_bounded_transcript(self, api, monkeypatch):
        """The local model's input is the transcript's head and
        tail, never the whole conversation (its window is unknown)."""
        monkeypatch.setattr(constants, "RETRIEVAL_LOCAL_EXTRACTION_MAX_CHARS", 40)
        sent = []

        async def local_model(messages, model_name, conversation=None):
            sent.append((model_name, conversation))
            return "export report"

        monkeypatch.setattr(rest_api, "extract_topics_with_local_model", local_model)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "START " + "x" * 500 + " END"}],
                memory_types=["procedural"],
                local_model_for_retrieval="llama-local",
            )
        ).json()
        [(model_name, conversation)] = sent
        assert model_name == "llama-local"
        assert conversation.startswith("USER: START") and conversation.endswith(" END")
        assert len(conversation) == 41
        assert body["extraction_status"] == "ok"
        assert body["topics"] == "export report"
        assert api.fake.chat_requests() == []
        row = await _log_row(api, body["retrieval_id"])
        entry = row.llm_usage["topic_extraction"]
        assert entry["local"] is True and entry["input"]["text_chars_cut"] > 400

    async def test_a_local_model_is_skipped_for_images(self, api, monkeypatch):
        """A text-only local model would not see the images: the
        configured model, which does, extracts instead."""
        called = []

        async def local_model(*args, **kwargs):
            called.append(1)
            return "should not be used"

        monkeypatch.setattr(rest_api, "extract_topics_with_local_model", local_model)
        _topics(api, topic="green chart")
        body = (
            await api.retrieve(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url(_png())},
                            },
                        ],
                    }
                ],
                memory_types=["procedural"],
                local_model_for_retrieval="llama-local",
            )
        ).json()
        assert called == []
        assert body["extraction_status"] == "ok" and body["topics"] == "green chart"
        assert len(api.fake.chat_requests()) == 1
        row = await _log_row(api, body["retrieval_id"])
        extraction_input = row.llm_usage["topic_extraction"]["input"]
        assert extraction_input["local_model"] == "llama-local"
        assert extraction_input["local_model_skipped"] == "images"
        assert extraction_input["images"] == 1

    async def test_core_only_skips_extraction(self, api):
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "hello"}], memory_types=["core"]
            )
        ).json()
        assert body["extraction_status"] == "skipped" and body["topics"] is None
        assert api.fake.chat_requests() == []
        row = await _log_row(api, body["retrieval_id"])
        assert row.extraction_status == "skipped" and row.embedding_status == "skipped"

    async def test_the_harness_call_and_render(self, api):
        """evals/skill_bench/methods/mirix.py: retrieve_with_conversation(
        messages=first_messages, memory_types=['procedural'],
        session_id=task_uid) with local_model_for_retrieval=None, then
        render_retrieved_memories(response)."""
        await api.add_skill(
            "open-drawer",
            cosine=0.95,
            triggers=["need an item from a drawer"],
            description="Open a drawer before taking from it.",
        )
        _topics(api, topic="take item; open drawer")
        response = await api.retrieve(
            messages=[
                {"role": "user", "content": "You are in a room. Your task is to ..."}
            ],
            limit=10,
            local_model_for_retrieval=None,
            memory_types=["procedural"],
            session_id="t-0123456789abcdef",
        )
        assert response.status_code == 200, response.text
        body = response.json()
        text = render_retrieved_memories(body)
        assert text == (
            "## Relevant skills from memory\n"
            "### open-drawer\n"
            "Open a drawer before taking from it.\n"
            "When to use: need an item from a drawer\n"
            "Instructions:\n"
            "Steps for open-drawer."
        )
        assert body["retrieval_id"] and body["topics"] == "take item; open drawer"


class TestSessionHeader:
    async def test_the_retrieval_serves_its_session_on_the_model_calls(
        self, api, monkeypatch
    ):
        """With MIRIX_LLM_SESSION_HEADER set, the topic extraction and the query
        embedding of a retrieval carry its session_id (a proxy's per-session cost);
        without a session_id, or with the setting off (the default), nothing is sent."""
        from mirix.observability.context import outbound_session_headers
        from mirix.settings import settings

        header = "x-skill-bench-item"
        seen = []
        original = api.embedder.get_text_embedding

        async def embed(text_value):
            seen.append(outbound_session_headers())
            return await original(text_value)

        monkeypatch.setattr(api.embedder, "get_text_embedding", embed)
        await api.add_skill("open-drawer", cosine=0.95)
        body = {
            "messages": [{"role": "user", "content": "Your task is to open a drawer"}],
            "memory_types": ["procedural"],
            "local_model_for_retrieval": None,
        }
        before = len(api.fake.chat_requests())
        monkeypatch.setattr(settings, "llm_session_header", header)
        for session_id in ("t0123456789abcdef0123", None):
            _topics(api, topic="open drawer")
            response = await api.retrieve(
                **body, **({"session_id": session_id} if session_id else {})
            )
            assert response.status_code == 200, response.text
        monkeypatch.setattr(settings, "llm_session_header", None)
        _topics(api, topic="open drawer")
        response = await api.retrieve(**body, session_id="t0123456789abcdef0123")
        assert response.status_code == 200, response.text
        sent = [r.headers.get(header) for r in api.fake.chat_requests()[before:]]
        assert sent == ["t0123456789abcdef0123", None, None]
        assert seen == [{header: "t0123456789abcdef0123"}, {}, {}]


# --------------------------------------------------------------------------- #
# Request validation                                                           #
# --------------------------------------------------------------------------- #


class TestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {"messages": []},
            {"messages": [{"role": "user", "content": ""}]},
            {"messages": [{"role": "user", "content": None}]},
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": " "}]}
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "file_uri", "file_uri": "/etc/hosts"}],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,bm90IGFuIGltYWdl"
                                },
                            }
                        ],
                    }
                ]
            },
            {"messages": [{"role": "user", "content": "q"}], "memory_types": []},
            {
                "messages": [{"role": "user", "content": "q"}],
                "memory_types": ["skills"],
            },
            {"messages": [{"role": "user", "content": "q"}], "session_id": "bad id"},
            {"messages": [{"role": "user", "content": "q"}], "session_id": ""},
            {"messages": [{"role": "user", "content": "q"}], "limit": -1},
            {"messages": "q"},
        ],
        ids=[
            "empty-list",
            "empty-string",
            "null",
            "blank-part",
            "server-file",
            "bad-image",
            "no-types",
            "unknown-type",
            "bad-session",
            "empty-session",
            "negative-limit",
            "not-a-list",
        ],
    )
    async def test_unusable_requests_are_422_before_any_model_call(self, api, body):
        response = await api.retrieve(**body)
        assert response.status_code == 422, response.text
        assert api.fake.chat_requests() == []
        assert api.embedder.calls == []

    async def test_a_foreign_stored_image_is_422(self, api):
        file_id = f"file-{uuid.uuid4()}"
        async with api.sm() as db:
            db.add(
                FileORM(
                    id=file_id,
                    organization_id=api.org,
                    user_id=api.other_user,
                    file_name="x.png",
                    file_path="/nonexistent/x.png",
                )
            )
            await db.commit()
        response = await api.retrieve(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "q"},
                        {"type": "database_image_id", "image_id": file_id},
                    ],
                }
            ],
            memory_types=["procedural"],
        )
        assert response.status_code == 422 and "not accessible" in response.text
        assert api.fake.chat_requests() == []

    async def test_duplicate_memory_types_are_merged(self, api):
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}],
                memory_types=["procedural", "procedural"],
            )
        ).json()
        row = await _log_row(api, body["retrieval_id"])
        assert row.memory_types == ["procedural"]

    async def test_a_user_of_another_organization_finds_nothing(self, api):
        """Tenant guard: no cross-organization retrieval, no log row, no
        extraction call — and exactly the answer of a user that does not exist,
        so a read is no user-id oracle."""
        foreign_user = await _foreign_user(api)
        await api.add_skill("export-report", cosine=0.9, user=foreign_user)
        before = await _log_count(api)
        foreign = await api.retrieve(
            user_id=foreign_user,
            messages=[{"role": "user", "content": "export the report"}],
            memory_types=["procedural", "semantic"],
        )
        missing = await api.retrieve(
            user_id="user-nobody",
            messages=[{"role": "user", "content": "export the report"}],
            memory_types=["procedural", "semantic"],
        )
        assert foreign.status_code == missing.status_code == 200
        assert foreign.json() == missing.json() == _NOTHING_FOUND
        for user_id in (foreign_user, "user-nobody"):
            topic = await api.http.get(
                "/memory/retrieve/topic",
                params={"user_id": user_id, "topic": "export report"},
                headers=api.headers(),
            )
            assert topic.status_code == 200
            assert topic.json() == {
                "success": True,
                "retrieval_id": None,
                "topic": "export report",
                "memories": {
                    "episodic": {"total_count": 0, "recent": [], "relevant": []},
                    "semantic": {"total_count": 0, "items": []},
                    "procedural": {"total_count": 0, "items": []},
                    "resource": {"total_count": 0, "items": []},
                    "knowledge_vault": {"total_count": 0, "items": []},
                    "core": {"total_count": 0, "scopes": {}},
                },
                "errors": {},
            }
        assert api.fake.chat_requests() == [] and await _log_count(api) == before
        assert all(not spy.calls for spy in api.spies.values())

    async def test_an_unknown_client_is_404(self, api):
        response = await api.http.post(
            "/memory/retrieve/conversation",
            json={"user_id": api.user, "messages": [{"role": "user", "content": "q"}]},
            headers={"x-client-id": "client-nobody", "x-org-id": api.org},
        )
        assert response.status_code == 404

    async def test_an_unknown_user_finds_nothing(self, api):
        """A first retrieval before the first add (which creates the user)
        is an empty success, as before the tenant guard."""
        before = await _log_count(api)
        response = await api.retrieve(
            user_id="user-nobody",
            messages=[{"role": "user", "content": "q"}],
            memory_types=["procedural", "semantic"],
        )
        assert response.status_code == 200
        assert response.json() == _NOTHING_FOUND
        assert api.fake.chat_requests() == [] and await _log_count(api) == before

    async def test_no_agents_keeps_the_old_answer(self, api, monkeypatch):
        monkeypatch.setattr(api.agents, "agents", [])
        body = (await api.retrieve(messages=[{"role": "user", "content": "q"}])).json()
        assert body == {
            "success": False,
            "error": "No agents found for this user",
            "topics": None,
            "memories": {},
        }


# --------------------------------------------------------------------------- #
# The retrieval log                                                            #
# --------------------------------------------------------------------------- #


class TestRetrievalLog:
    async def test_the_row_has_every_field(self, api):
        good = await api.add_skill(
            "export-report", cosine=0.9, triggers=["export a report"]
        )
        await api.add_skill("no-vector")
        _topics(
            api,
            usage={"prompt_tokens": 321, "completion_tokens": 12, "total_tokens": 333},
        )
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "Export the report"}],
                memory_types=["procedural"],
                session_id="task-log-1",
            )
        ).json()
        row = await _log_row(api, body["retrieval_id"])
        assert row.organization_id == api.org and row.user_id == api.user
        assert row.client_id == api.client.id and row.session_id == "task-log-1"
        assert row.memory_types == ["procedural"]
        assert row.query_text == "USER: Export the report"
        assert row.extracted_topics == "export report csv"
        assert row.extraction_status == "ok" and row.embedding_status == "ok"
        trace = row.per_type["procedural"]
        assert trace["returned_ids"] == [good]
        assert trace["missing_vector_count"] == 1 and trace["candidate_count"] == 2
        assert {c["id"] for c in trace["candidates"]} >= {good}
        for key in ("bm25", "bm25_rank", "cosine", "cosine_rank", "rrf", "rrf_rank"):
            assert key in trace["candidates"][0]
        assert trace["gate"]["applied"] is True
        assert (
            trace["gate"]["min_similarity"] == constants.SKILL_RETRIEVAL_MIN_SIMILARITY
        )
        usage = row.llm_usage["topic_extraction"]
        assert usage["prompt_tokens"] == 321 and usage["completion_tokens"] == 12
        assert usage["model"] == "fake-model" and usage["status"] == "ok"
        assert usage["input"]["truncated"] is False
        assert isinstance(row.latency_ms, int) and row.latency_ms >= 0
        assert row.retry_events == [] and row.embedding_chunked_calls == 0

    async def test_chunked_embedding_calls_are_counted(self, api):
        api.embedder.chunks = 3
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}], memory_types=["procedural"]
            )
        ).json()
        row = await _log_row(api, body["retrieval_id"])
        assert row.embedding_chunked_calls == 1

    async def test_retries_are_recorded(self, api):
        """The SDK's transport retry of a 503 lands in retry_events."""
        from fakes.fake_llm_server import Fault

        api.fake.inject_fault(Fault.http_error(503, "overloaded"))
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}], memory_types=["procedural"]
            )
        ).json()
        assert body["extraction_status"] == "ok"
        row = await _log_row(api, body["retrieval_id"])
        assert len(row.retry_events) == 1
        [event] = row.retry_events
        # The SDK's own retry (recorded by the transport hook), inside the
        # retrieval's recording block.
        assert event["kind"] == "infra" and event["stage"] == "transport"
        assert event["attempt"] == 1 and event["waited_s"] > 0

    async def test_the_switch_turns_the_log_off(self, api, monkeypatch):
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_ENABLED", False)
        before = await _log_count(api)
        _topics(api)
        body = (
            await api.retrieve(
                messages=[{"role": "user", "content": "q"}], memory_types=["procedural"]
            )
        ).json()
        assert body["success"] is True and body["retrieval_id"] is None
        assert await _log_count(api) == before

    async def test_a_failed_log_write_does_not_fail_the_retrieval(
        self, api, monkeypatch
    ):
        async def broken(**kwargs):
            raise RuntimeError("log table is gone")

        monkeypatch.setattr(api.server.retrieval_log_manager, "create_log", broken)
        _topics(api)
        response = await api.retrieve(
            messages=[{"role": "user", "content": "q"}], memory_types=["procedural"]
        )
        assert response.status_code == 200
        assert response.json()["retrieval_id"] is None

    async def test_listing_is_scoped_to_the_client(self, api):
        ids = {}
        for session, as_client in (
            ("sess-a", None),
            ("sess-b", None),
            ("sess-a", api.other_client),
        ):
            _topics(api)
            body = (
                await api.retrieve(
                    as_client=as_client,
                    messages=[{"role": "user", "content": "q"}],
                    memory_types=["procedural"],
                    session_id=session,
                )
            ).json()
            ids[(session, as_client is None)] = body["retrieval_id"]

        response = await api.logs(session_id="sess-a")
        assert response.status_code == 200
        [row] = response.json()["retrieval_logs"]
        assert row["id"] == ids[("sess-a", True)] and row["session_id"] == "sess-a"
        assert row["per_type"]["procedural"]["gate"]["applied"] is True
        listed = [r["id"] for r in (await api.logs()).json()["retrieval_logs"]]
        assert ids[("sess-a", True)] in listed and ids[("sess-b", True)] in listed
        assert ids[("sess-a", False)] not in listed
        other = (await api.logs(as_client=api.other_client)).json()["retrieval_logs"]
        assert [r["id"] for r in other] == [ids[("sess-a", False)]]
        assert len((await api.logs(limit=1)).json()["retrieval_logs"]) == 1
        by_user = (await api.logs(user_id=api.other_user)).json()["retrieval_logs"]
        assert by_user == []

    @pytest.mark.parametrize(
        "params", [{"session_id": "bad id"}, {"limit": 0}, {"limit": 501}]
    )
    async def test_listing_validates(self, api, params):
        assert (await api.logs(**params)).status_code == 422

    async def test_listing_requires_authentication(self, api):
        response = await api.http.get("/memory/retrieval_logs")
        assert response.status_code == 401


class TestTopicEndpoint:
    async def test_topic_retrieval_is_logged_and_gated(self, api):
        good = await api.add_skill(
            "export-report", cosine=0.9, triggers=["export report"]
        )
        await api.add_skill("weak", cosine=0.2, triggers=["export report"])
        response = await api.http.get(
            "/memory/retrieve/topic",
            params={"user_id": api.user, "topic": "export report"},
            headers=api.headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [i["id"] for i in body["memories"]["procedural"]["items"]] == [good]
        assert api.fake.chat_requests() == []  # the topic IS the query
        row = await _log_row(api, body["retrieval_id"])
        assert row.query_text == "export report" and row.extraction_status == "skipped"
        assert row.memory_types == ALL_TYPES and row.session_id is None

    async def test_a_nul_in_the_topic_never_reaches_a_search(self, api):
        """The topic is the query; its NUL is searched as U+FFFD."""
        response = await api.http.get(
            "/memory/retrieve/topic",
            params={"user_id": api.user, "topic": "export\x00report"},
            headers=api.headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True and body["errors"] == {}
        assert api.embedder.calls == ["export\ufffdreport"]
        [(_method, kwargs)] = api.spies["semantic"].calls[:1]
        assert kwargs["query"] == "export\ufffdreport"
        row = await _log_row(api, body["retrieval_id"])
        assert row.query_text == "export\ufffdreport"

    async def test_topic_for_an_unknown_user_finds_nothing(self, api):
        before = await _log_count(api)
        response = await api.http.get(
            "/memory/retrieve/topic",
            params={"user_id": "user-nobody", "topic": "x"},
            headers=api.headers(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True and body["retrieval_id"] is None
        assert all(
            not v.get("items") and not v.get("relevant")
            for v in body["memories"].values()
        )
        assert await _log_count(api) == before

    async def test_every_type_failing_is_not_success(self, api):
        """A topic retrieval all of whose types failed says so."""
        for spy in api.spies.values():
            spy.fail = True
        api.embedder.fail = True

        async def broken(*args, **kwargs):
            raise RuntimeError("skill store is down")

        api.server.procedural_memory_manager = SimpleNamespace(retrieve_skills=broken)
        response = await api.http.get(
            "/memory/retrieve/topic",
            params={"user_id": api.user, "topic": "export"},
            headers=api.headers(),
        )
        body = response.json()
        assert body["success"] is False
        assert body["error"].startswith("every requested memory type failed")
        assert set(body["errors"]) == set(ALL_TYPES)


# --------------------------------------------------------------------------- #
# The client                                                                   #
# --------------------------------------------------------------------------- #


def _recording_client(monkeypatch, response):
    client = MirixClient(api_key="test-key", base_url="http://test")
    calls = []

    async def fake_request(method, endpoint, json=None, params=None, headers=None):
        calls.append(
            {"method": method, "endpoint": endpoint, "json": json, "params": params}
        )
        return response

    async def no_user(user_id=None, headers=None):
        return None

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "_ensure_user_exists", no_user)
    client._meta_agent = SimpleNamespace(id="agent-1")
    return client, calls


class TestClient:
    async def test_retrieve_with_conversation_sends_the_new_fields(self, monkeypatch):
        client, calls = _recording_client(monkeypatch, {"success": True})
        try:
            await client.retrieve_with_conversation(
                "u1",
                [{"role": "user", "content": "q"}],
                memory_types=["procedural"],
                session_id="task-1",
            )
            await client.retrieve_with_conversation(
                "u1", [{"role": "user", "content": "q"}]
            )
            with pytest.raises(ValueError):
                await client.retrieve_with_conversation(
                    "u1", [{"role": "user", "content": "q"}], session_id="bad id"
                )
        finally:
            await client.close()
        first, second = calls
        assert first["endpoint"] == "/memory/retrieve/conversation"
        assert first["json"]["memory_types"] == ["procedural"]
        assert first["json"]["session_id"] == "task-1"
        assert (
            "memory_types" not in second["json"] and "session_id" not in second["json"]
        )

    async def test_list_retrieval_logs(self, monkeypatch):
        client, calls = _recording_client(
            monkeypatch, {"retrieval_logs": [{"id": "rlog-1"}]}
        )
        try:
            assert await client.list_retrieval_logs(session_id="s1", limit=5) == [
                {"id": "rlog-1"}
            ]
            await client.list_retrieval_logs(user_id="u2")
            with pytest.raises(ValueError):
                await client.list_retrieval_logs(session_id="bad id")
        finally:
            await client.close()
        assert calls[0] == {
            "method": "GET",
            "endpoint": "/memory/retrieval_logs",
            "json": None,
            "params": {"limit": 5, "session_id": "s1"},
        }
        assert calls[1]["params"] == {"limit": 50, "user_id": "u2"}

    async def test_the_endpoints_and_fields_the_harness_checks_exist(self):
        """methods/mirix.py's missing_features(): the retrieve request schema
        has memory_types and session_id."""
        schema = rest_api.app.openapi()
        paths = schema["paths"]
        assert "post" in paths["/memory/retrieve/conversation"]
        assert "get" in paths["/memory/retrieval_logs"]
        ref = paths["/memory/retrieve/conversation"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]["$ref"]
        fields = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]["properties"]
        assert {"memory_types", "session_id"} <= set(fields)
