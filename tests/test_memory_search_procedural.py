"""GET /memory/search for procedural memory.

``/memory/search`` ranks skills with the M7.3 pipeline — true BM25 over the
retrieval text, cosine, RRF — but WITHOUT the retrieval gate: it returns the
top ``limit`` whatever their similarity, each with ``retrieval_scores``
``{bm25, cosine, rrf}``. Field-specific keyword searches (``instructions``),
string/fuzzy matching and the empty query keep the per-field listing
(``retrieval_scores`` is null there).

The FastAPI app is driven in-process through httpx's ASGI transport; auth and
the server singleton are fakes, the procedural manager is real on a throwaway
SQLite file, and the query vector comes from an in-process fake (cosine of a
skill to it = the number the test names).
"""

import json
import math
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.embeddings as embeddings_module
import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.base import Base
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.server import rest_api
from mirix.services.procedural_memory_manager import ProceduralMemoryManager

pytestmark = pytest.mark.asyncio(loop_scope="module")

SCOPE = "test"


def vec(cosine):
    return [cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))] + [0.0] * (
        MAX_EMBEDDING_DIM - 2
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("memory_search_procedural") / "test.db"
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
    """Every query embeds to (1, 0, ...); counts the calls. ``fail`` raises."""

    def __init__(self):
        self.calls = []
        self.fail = False

    async def get_text_embedding(self, text_value):
        self.calls.append(text_value)
        if self.fail:
            raise ConnectionError("embedding endpoint down")
        return [1.0, 0.0]


@pytest_asyncio.fixture(loop_scope="module")
async def api(sm, monkeypatch):
    org_id = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        await db.commit()
    client = PydanticClient(
        id=f"client-{uuid.uuid4().hex[:8]}",
        organization_id=org_id,
        name="c",
        write_scope=SCOPE,
        read_scopes=[SCOPE],
    )
    user = PydanticUser(
        id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
    )
    agent_state = AgentState(
        id="agent-search",
        name="meta",
        system="s",
        agent_type=AgentType.meta_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
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

    class Agents:
        async def list_agents(self, actor, limit=None):
            return [agent_state]

    class Users:
        async def get_user_by_id(self, uid=None):
            return user

    async def fake_auth(authorization=None, http_request=None):
        return client, "api_key"

    server = SimpleNamespace(
        agent_manager=Agents(),
        user_manager=Users(),
        procedural_memory_manager=ProceduralMemoryManager(
            session_maker=sm, sleep=no_sleep
        ),
    )
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(embeddings_module, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", fake_auth)
    monkeypatch.setattr(rest_api, "get_server", lambda: server)

    async def add(skill_id, *, name, cosine=None, scope=SCOPE, instructions="Do it."):
        async with sm() as db:
            db.add(
                ProceduralORM(
                    id=skill_id,
                    organization_id=org_id,
                    user_id=user_id,
                    name=name,
                    entry_type="workflow",
                    description="",
                    instructions=instructions,
                    triggers=[],
                    examples=[],
                    filter_tags={"scope": scope},
                    retrieval_embedding=vec(cosine) if cosine is not None else None,
                )
            )
            await db.commit()

    transport = httpx.ASGITransport(app=rest_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:

        async def search(**params):
            params.setdefault("user_id", user_id)
            response = await http.get(
                "/memory/search",
                params=params,
                headers={"Authorization": "Bearer fake"},
            )
            assert response.status_code == 200, response.text
            return response.json()

        async def get(path, **params):
            response = await http.get(
                path, params=params, headers={"Authorization": "Bearer fake"}
            )
            assert response.status_code == 200, response.text
            return response.json()

        yield SimpleNamespace(
            add=add,
            search=search,
            get=get,
            embedder=embedder,
            user=user,
            server=server,
            client=client,
        )


def _procedural(body):
    return [r for r in body["results"] if r["memory_type"] == "procedural"]


async def test_procedural_search_is_the_ranking_without_the_gate(api):
    # Every cosine is below the 0.60 gate: /memory/search returns them anyway.
    await api.add("s-a", name="rotate logs", cosine=0.55)
    await api.add("s-b", name="archive logs", cosine=0.3)
    await api.add("s-c", name="water plants", cosine=0.1)
    body = await api.search(query="rotate logs", memory_type="procedural", limit=2)
    assert body["search_method"] == "hybrid"
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["s-a", "s-b"]
    assert body["count"] == 2 and body["total_count"] == 3
    scores = rows[0]["retrieval_scores"]
    assert set(scores) == {"bm25", "cosine", "rrf"}
    assert scores["cosine"] == pytest.approx(0.55, abs=1e-6)
    assert scores["bm25"] > 0 and scores["rrf"] == pytest.approx(2 / 61)
    # The same order the manager ranks without the gate.
    direct = await api.server.procedural_memory_manager.retrieve_skills(
        "rotate logs",
        api.user,
        [SCOPE],
        2,
        apply_gate=False,
        query_embedding=[1.0, 0.0],
    )
    assert [r.skill.id for r in direct.skills] == ["s-a", "s-b"]
    gated = await api.server.procedural_memory_manager.retrieve_skills(
        "rotate logs", api.user, [SCOPE], 2, apply_gate=True, query_embedding=[1.0, 0.0]
    )
    assert gated.skills == []  # what /memory/retrieve/conversation would inject


async def test_the_all_sweep_ranks_procedural_memory_with_one_embedding(api):
    await api.add("all-a", name="brew coffee", cosine=0.9)
    body = await api.search(query="brew coffee")  # memory_type=all, default methods
    assert body["search_method"] == "embedding"  # the other types' method
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["all-a"]
    assert rows[0]["retrieval_scores"]["cosine"] == pytest.approx(0.9, abs=1e-6)
    assert rows[0]["retrieval_scores"]["bm25"] > 0
    # The sweep's one precomputed query vector is reused for skills.
    assert api.embedder.calls == ["brew coffee"]


async def test_the_all_sweep_ranks_skills_on_their_default_field(api):
    # A field named for the sweep is ignored by it ("all" searches every
    # type's default field): skills are still ranked over the retrieval text.
    await api.add("sw-a", name="tune guitar", cosine=0.9, instructions="nothing")
    body = await api.search(query="tune guitar", search_field="instructions")
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["sw-a"]
    scores = rows[0]["retrieval_scores"]
    # The default hybrid ran (not the instructions keyword search).
    assert scores["bm25"] > 0 and scores["cosine"] == pytest.approx(0.9, abs=1e-6)


async def test_an_explicit_bm25_search_ranks_by_bm25_alone(api):
    await api.add("bm-a", name="file taxes", cosine=0.9)
    await api.add("bm-b", name="unrelated", cosine=0.99)
    body = await api.search(
        query="file taxes", memory_type="procedural", search_method="bm25"
    )
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["bm-a"]
    assert rows[0]["retrieval_scores"]["cosine"] is None
    assert api.embedder.calls == []


async def test_an_embedding_search_honors_similarity_threshold(api):
    await api.add("th-near", name="alpha", cosine=0.8)
    await api.add("th-far", name="beta", cosine=0.2)
    body = await api.search(
        query="gamma",
        memory_type="procedural",
        search_method="embedding",
        similarity_threshold=0.5,  # cosine distance < 0.5
    )
    assert [r["id"] for r in _procedural(body)] == ["th-near"]
    unfiltered = await api.search(
        query="gamma", memory_type="procedural", search_method="embedding"
    )
    assert [r["id"] for r in _procedural(unfiltered)] == ["th-near", "th-far"]


async def test_an_instructions_search_is_keyword_search_without_scores(api):
    await api.add(
        "in-a", name="n1", instructions="run kubectl rollout restart", cosine=0.9
    )
    await api.add("in-b", name="kubectl", instructions="nothing here", cosine=0.9)
    body = await api.search(
        query="kubectl", memory_type="procedural", search_field="instructions"
    )
    assert body["search_method"] == "bm25"  # Instructions carry no vector
    rows = _procedural(body)
    assert rows[0]["id"] == "in-a"
    assert all(r["retrieval_scores"] is None for r in rows)
    assert api.embedder.calls == []


async def test_an_empty_query_lists_the_skills_unranked(api):
    await api.add("em-a", name="first", cosine=0.9)
    body = await api.search(query="", memory_type="procedural")
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["em-a"]
    assert rows[0]["retrieval_scores"] is None
    assert api.embedder.calls == []


async def test_a_failed_query_vector_still_answers_by_bm25(api):
    await api.add("fb-hit", name="restart pods", cosine=0.1)
    await api.add("fb-miss", name="bake bread", cosine=0.99)
    api.embedder.fail = True
    body = await api.search(query="restart pods", memory_type="procedural")
    assert body["success"] is True
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["fb-hit"]
    assert rows[0]["retrieval_scores"]["cosine"] is None
    assert len(api.embedder.calls) > 1  # retried before falling back


async def test_only_the_clients_read_scopes_are_searched(api):
    await api.add("sc-mine", name="walk dog", cosine=0.9)
    await api.add("sc-other", name="walk dog o", cosine=0.9, scope="other")
    body = await api.search(query="walk dog", memory_type="procedural")
    assert [r["id"] for r in _procedural(body)] == ["sc-mine"]


async def test_an_embedding_outage_searches_every_type_by_bm25(api):
    """No query vector is no 500 — the sweep (and an explicit embedding
    search) answers by keywords, and says so."""
    await api.add("out-hit", name="restart pods", cosine=0.1)
    await api.add("out-miss", name="bake bread", cosine=0.99)
    api.embedder.fail = True
    body = await api.search(query="restart pods")  # "all": embedding by default
    assert body["success"] is True
    assert body["search_method"] == "bm25"
    [warning] = body["warnings"]
    assert "query embedding failed" in warning and "ConnectionError" in warning
    rows = _procedural(body)
    assert [r["id"] for r in rows] == ["out-hit"]
    assert rows[0]["retrieval_scores"]["cosine"] is None

    explicit = await api.search(
        query="restart pods", memory_type="procedural", search_method="embedding"
    )
    assert explicit["success"] is True and explicit["search_method"] == "bm25"
    assert [r["id"] for r in _procedural(explicit)] == ["out-hit"]


async def test_a_working_embedding_has_no_warnings(api):
    await api.add("ok-a", name="brew tea", cosine=0.9)
    body = await api.search(query="brew tea")
    assert "warnings" not in body and body["search_method"] == "embedding"


@pytest.mark.parametrize("who", ["foreign", "missing"])
async def test_a_user_outside_the_organization_finds_nothing(api, monkeypatch, who):
    """/memory/search is tenant-guarded like the retrieval endpoints: a
    user of another organization reads nothing — the same answer as a user
    that does not exist (no user-id oracle)."""
    from mirix.orm.errors import NoResultFound

    await api.add(f"tg-{who}", name="walk dog", cosine=0.9)

    class Users:
        async def get_user_by_id(self, uid=None):
            if who == "missing":
                raise NoResultFound(f"no user {uid}")
            return PydanticUser(
                id=uid, name=uid, organization_id="org-someone-else", timezone="UTC"
            )

    monkeypatch.setattr(api.server, "user_manager", Users())
    for memory_type in ("procedural", "all"):
        body = await api.search(query="walk dog", memory_type=memory_type)
        assert body["success"] is True
        assert body["results"] == [] and body["count"] == 0
    assert api.embedder.calls == []  # nothing was searched


async def test_the_cross_user_search_survives_an_embedding_outage(api, monkeypatch):
    """/memory/search_all_users falls back to keywords the same way."""
    await api.add("xu-hit", name="rotate secrets", cosine=0.1)
    client = api.client

    class Clients:
        async def get_client_by_id(self, cid):
            assert cid == client.id
            return client

    monkeypatch.setattr(api.server, "client_manager", Clients(), raising=False)
    api.embedder.fail = True
    body = await api.get(
        "/memory/search_all_users",
        query="rotate secrets",
        memory_type="procedural",
        search_method="embedding",
        client_id=client.id,
    )
    assert body["success"] is True and body["search_method"] == "bm25"
    [warning] = body["warnings"]
    assert "query embedding failed" in warning
    # Exactly what an explicit keyword search answers.
    explicit = await api.get(
        "/memory/search_all_users",
        query="rotate secrets",
        memory_type="procedural",
        search_method="bm25",
        client_id=client.id,
    )
    assert "warnings" not in explicit
    assert body["results"] == explicit["results"]


async def test_a_nul_in_the_query_is_searched_as_u_fffd(api, monkeypatch):
    """Postgres rejects U+0000 in any text parameter, so /memory/search
    and /memory/search_all_users search (and echo) the query with U+FFFD, as MIRIX
    stores text (tests/test_retrieval_log_pg.py runs /memory/search on Postgres)."""
    await api.add("nul-hit", name="rotate logs", cosine=0.9)
    manager = api.server.procedural_memory_manager
    ranked, by_org = manager.retrieve_skills, manager.list_procedures_by_org
    queries: dict[str, list] = {"ranked": [], "by_org": []}

    async def ranking(query, *args, **kwargs):
        queries["ranked"].append(query)
        return await ranked(query, *args, **kwargs)

    async def listing(*args, **kwargs):
        queries["by_org"].append(kwargs.get("query"))
        return await by_org(*args, **kwargs)

    monkeypatch.setattr(manager, "retrieve_skills", ranking)
    monkeypatch.setattr(manager, "list_procedures_by_org", listing)
    body = await api.search(query="rotate\x00logs", memory_type="procedural")
    assert body["success"] is True and body["query"] == "rotate\ufffdlogs"
    assert [r["id"] for r in _procedural(body)] == ["nul-hit"]
    assert api.embedder.calls == ["rotate\ufffdlogs"]

    client = api.client

    class Clients:
        async def get_client_by_id(self, cid):
            assert cid == client.id
            return client

    monkeypatch.setattr(api.server, "client_manager", Clients(), raising=False)
    everyone = await api.get(
        "/memory/search_all_users",
        query="rotate\x00logs",
        memory_type="procedural",
        search_method="bm25",
        client_id=client.id,
    )
    assert everyone["success"] is True
    assert everyone["query"] == "rotate\ufffdlogs"
    for searched in queries.values():  # /memory/search ranks, the sweep lists by org
        assert searched and set(searched) == {"rotate\ufffdlogs"}
    assert "\x00" not in json.dumps(everyone)
