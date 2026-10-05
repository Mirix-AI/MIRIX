"""Hermetic tests for procedural memory's retrieval fields.

A skill now has ONE vector, `retrieval_embedding` (the per-field description /
instructions vectors are gone). These tests pin the behavior that change must
keep coherent, without Postgres, Redis or any model provider:

* an "embedding" search on `instructions` runs as full-text keyword
  search on `instructions` (and never embeds the query);
* the write paths compute exactly one vector, into `retrieval_embedding`, of
  the skill's retrieval text (name + triggers + description), and
  recompute it when any of those change;
* procedural memory never consults a cache: reads come from the
  database only.

The SQLite fallback branch is forced like tests/test_scope_read_consistency.py.
"""

import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register all ORM classes before Base is used
import mirix.services.procedural_memory_manager as pmm
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem as ProceduralORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate
from mirix.schemas.user import User as PydanticUser
from mirix.services.procedural_memory_manager import (
    ProceduralMemoryManager,
    effective_procedural_search_method,
)
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

# An unreachable local endpoint: if any code path escaped the monkeypatched
# embedding factory, it fails fast on localhost instead of reaching a provider.
_OFFLINE_EMBEDDING = {
    "embedding_endpoint_type": "openai",
    "embedding_endpoint": "http://127.0.0.1:9/v1",
    "embedding_model": "x",
    "embedding_dim": 8,
}


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_maker(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("retrieval_fields") / "test.db"
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


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def seed(session_maker):
    org_id = f"rf-org-{uuid.uuid4().hex[:8]}"
    user_id = f"rf-user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    async with session_maker() as session:
        session.add(OrganizationORM(id=org_id, name=org_id))
        session.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        session.add(AgentORM(id=agent_id, organization_id=org_id))
        session.add(
            ClientORM(
                id=client_id,
                organization_id=org_id,
                name="rf-client",
                status="active",
                write_scope="test",
                read_scopes=["test"],
            )
        )
        # Inserted in this order on purpose: the description-only match comes
        # FIRST, so only an instructions-text ranking puts rollout-restart on top.
        # Fillers keep the BM25 IDF of "kubectl" positive (1 hit in 4 docs).
        skills = [
            (
                "mug-wash",
                "kubectl is mentioned only in this description",
                "Rinse the mug in the sink",
            ),
            ("filler-a", "Water the plants", "Fill the can and water"),
            ("filler-b", "Sort the mail", "Open the box and sort letters"),
            (
                "rollout-restart",
                "Restart a deployment safely",
                "Run kubectl rollout restart and watch the pods",
            ),
        ]
        for name, description, instructions in skills:
            session.add(
                ProceduralORM(
                    id=f"proc-{uuid.uuid4()}",
                    organization_id=org_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    name=name,
                    entry_type="workflow",
                    description=description,
                    instructions=instructions,
                )
            )
        await session.commit()
    return {
        "org_id": org_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "client_id": client_id,
    }


@pytest.fixture(scope="module")
def user(seed):
    return PydanticUser(
        id=seed["user_id"], name="u", organization_id=seed["org_id"], timezone="UTC"
    )


@pytest.fixture(scope="module")
def actor(seed):
    return PydanticClient(
        id=seed["client_id"],
        organization_id=seed["org_id"],
        name="rf-client",
        write_scope="test",
        read_scopes=["test"],
    )


@pytest.fixture(scope="module")
def agent_state():
    return AgentState(
        id="agent-rf",
        name="rf-agent",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(**_OFFLINE_EMBEDDING),
        tools=[],
    )


class _RecordingEmbedModel:
    def __init__(self, calls):
        self.calls = calls

    async def get_text_embedding(self, text):
        self.calls.append(text)
        return [0.25] * 8


@pytest.fixture
def embed_calls(monkeypatch):
    calls = []

    async def fake_embedding_model(config, *args, **kwargs):
        return _RecordingEmbedModel(calls)

    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    return calls


@pytest.fixture
def no_embedding(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("an instructions search must not embed the query")

    monkeypatch.setattr(pmm, "embedding_model", forbidden)
    monkeypatch.setattr(pmm, "build_query", forbidden)


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method, field, expected",
    [
        ("embedding", "instructions", "bm25"),
        ("embedding", "description", "embedding"),
        ("embedding", "", "embedding"),
        ("hybrid", "instructions", "bm25"),  # hybrid needs a vector too
        ("hybrid", "description", "hybrid"),
        ("bm25", "instructions", "bm25"),
        ("string_match", "instructions", "string_match"),
    ],
)
def test_effective_search_method(method, field, expected):
    assert effective_procedural_search_method(method, field) == expected


async def test_embedding_search_on_instructions_runs_as_keyword_search(
    session_maker, seed, user, agent_state, no_embedding
):
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker
    results = await mgr.list_procedures(
        agent_state=agent_state,
        user=user,
        query="kubectl",
        search_field="instructions",
        search_method="embedding",
        use_cache=False,
    )
    # The skill whose INSTRUCTIONS mention kubectl ranks first, ahead of the
    # description-only mention: the search ran over instructions text, not over
    # a (non-existent) instructions vector. (The SQLite BM25 fallback returns
    # zero-score rows after the hits; that is pre-existing behavior.)
    assert results[0].name == "rollout-restart"


async def test_org_wide_embedding_search_on_instructions_builds_a_fulltext_query(
    no_embedding,
):
    """The org-wide path reroutes too: its SQL ranks instructions by full text."""
    statements = []

    class _Result:
        def scalars(self):
            return self

        def all(self):
            return []

    class _FakeSession:
        async def execute(self, statement):
            statements.append(
                str(statement.compile(compile_kwargs={"literal_binds": False}))
            )
            return _Result()

    @asynccontextmanager
    async def fake_session_maker():
        yield _FakeSession()

    mgr = ProceduralMemoryManager()
    mgr.session_maker = fake_session_maker
    agent_state = AgentState(
        id="agent-rf2",
        name="rf-agent",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(**_OFFLINE_EMBEDDING),
        tools=[],
    )
    await mgr.list_procedures_by_org(
        agent_state=agent_state,
        organization_id="org-x",
        query="kubectl",
        search_field="instructions",
        search_method="embedding",
        use_cache=False,
    )
    assert len(statements) == 1
    sql = statements[0]
    assert "to_tsvector" in sql and "instructions" in sql
    assert "retrieval_embedding" not in sql.split("FROM", 1)[1]


# --------------------------------------------------------------------------- #
# Write paths compute exactly one vector                                      #
# --------------------------------------------------------------------------- #


async def test_insert_procedure_embeds_only_the_retrieval_text(
    session_maker, seed, agent_state, actor, embed_calls
):
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker
    created = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=seed["agent_id"],
        name="new-skill",
        description="describe it",
        instructions="long instructions that must not be embedded",
        entry_type="workflow",
        triggers=["when asked", "on demand"],
        actor=actor,
        organization_id=seed["org_id"],
        user_id=seed["user_id"],
        use_cache=False,
    )
    # Name, each trigger on its own line, description — never instructions.
    assert embed_calls == ["new-skill\nwhen asked\non demand\ndescribe it"]
    assert len(created.retrieval_embedding) == MAX_EMBEDDING_DIM
    assert created.retrieval_embedding[:8] == [0.25] * 8
    assert all(v == 0.0 for v in created.retrieval_embedding[8:])


async def test_update_recomputes_retrieval_embedding_when_the_retrieval_text_changes(
    session_maker, seed, user, agent_state, actor, embed_calls
):
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker
    created = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=seed["agent_id"],
        name="edit-me",
        description="v1",
        instructions="i1",
        entry_type="workflow",
        actor=actor,
        organization_id=seed["org_id"],
        user_id=seed["user_id"],
        use_cache=False,
    )
    embed_calls.clear()

    updated = await mgr.update_item(
        ProceduralMemoryItemUpdate(id=created.id, instructions="i2"),
        user=user,
        actor=actor,
        agent_state=agent_state,
    )
    assert embed_calls == []  # instructions carry no vector
    assert updated.instructions == "i2"

    updated = await mgr.update_item(
        ProceduralMemoryItemUpdate(id=created.id, description="v2"),
        user=user,
        actor=actor,
        agent_state=agent_state,
    )
    assert embed_calls == ["edit-me\nv2"]
    assert len(updated.retrieval_embedding) == MAX_EMBEDDING_DIM

    # Triggers and name are part of the retrieval text too.
    embed_calls.clear()
    await mgr.update_item(
        ProceduralMemoryItemUpdate(id=created.id, triggers=["on release"]),
        user=user,
        actor=actor,
        agent_state=agent_state,
    )
    assert embed_calls == ["edit-me\non release\nv2"]
    embed_calls.clear()
    await mgr.update_item(
        ProceduralMemoryItemUpdate(id=created.id, name="edited"),
        user=user,
        actor=actor,
        agent_state=agent_state,
    )
    assert embed_calls == ["edited\non release\nv2"]


# --------------------------------------------------------------------------- #
# No cache                                                                    #
# --------------------------------------------------------------------------- #


def _forbid_cache(monkeypatch):
    import mirix.database.cache_provider as cache_provider
    import mirix.database.redis_client as redis_client

    def forbidden(*args, **kwargs):
        raise AssertionError("procedural memory must not touch the cache")

    monkeypatch.setattr(cache_provider, "get_cache_provider", forbidden)
    monkeypatch.setattr(redis_client, "get_redis_client", forbidden)


async def test_get_item_by_id_reads_the_database_only(
    monkeypatch, session_maker, seed, user, actor, agent_state, embed_calls
):
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker
    created = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=seed["agent_id"],
        name="db-only",
        description="d",
        instructions="i",
        entry_type="workflow",
        actor=actor,
        organization_id=seed["org_id"],
        user_id=seed["user_id"],
    )
    _forbid_cache(monkeypatch)
    item = await mgr.get_item_by_id(
        item_id=created.id, user=user, timezone_str="UTC", actor=actor
    )
    assert item.id == created.id


async def test_list_procedures_with_use_cache_still_reads_the_database_only(
    monkeypatch, session_maker, seed, user, agent_state
):
    _forbid_cache(monkeypatch)
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker
    for method in ("bm25", "string_match", "fuzzy_match"):
        results = await mgr.list_procedures(
            agent_state=agent_state,
            user=user,
            query="kubectl",
            search_field="instructions",
            search_method=method,
            use_cache=True,
        )
        assert any(r.name == "rollout-restart" for r in results), method
    recent = await mgr.list_procedures(
        agent_state=agent_state, user=user, query="", use_cache=True
    )
    assert recent


async def test_the_orm_cache_hook_never_caches_a_skill(monkeypatch):
    """No cache at the ORM layer: even a stray create_with_redis/update_with_redis on
    a skill row writes nothing to the cache (procedural_memory is not a cached
    table), while the other memory tables still are."""
    import mirix.database.cache_provider as cache_provider
    import mirix.database.redis_client as redis_client
    from mirix.orm.episodic_memory import EpisodicEvent as EpisodicORM

    writes = []

    class _Cache:
        PROCEDURAL_PREFIX = "procedural:"
        EPISODIC_PREFIX = "episodic:"
        SEMANTIC_PREFIX = "semantic:"
        RESOURCE_PREFIX = "resource:"
        KNOWLEDGE_PREFIX = "knowledge:"
        RAW_MEMORY_PREFIX = "raw_memory:"
        BLOCK_PREFIX = MESSAGE_PREFIX = ORGANIZATION_PREFIX = USER_PREFIX = "x:"
        AGENT_PREFIX = TOOL_PREFIX = "x:"

        async def set_json(self, key, data, ttl=None):
            writes.append(key)

        async def delete(self, key):
            writes.append(("delete", key))

    monkeypatch.setattr(cache_provider, "get_cache_provider", lambda: _Cache())
    monkeypatch.setattr(redis_client, "get_redis_client", lambda: None)

    skill = ProceduralORM(
        id="proc-nocache",
        organization_id="o",
        user_id="u",
        name="n",
        entry_type="workflow",
        description="d",
        instructions="i",
        triggers=[],
        examples=[],
    )
    for operation in ("create", "update", "delete"):
        await skill._update_redis_cache(operation=operation)
    assert writes == []

    # The hook itself still works for a cached table (the test is not vacuous).
    event = EpisodicORM(id="ep-1")
    monkeypatch.setattr(EpisodicORM, "to_pydantic", lambda self: _Dumpable())
    await event._update_redis_cache(operation="update")
    assert writes == ["episodic:ep-1"]


class _Dumpable:
    def model_dump(self, mode="json"):
        return {"id": "ep-1"}
