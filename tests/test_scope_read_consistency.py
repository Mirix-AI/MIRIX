"""Hermetic-SQLite regression tests for scope-consistent memory reads.

These lock in the read-side half of the procedural-scope fix: the SQLite
in-memory ``bm25`` fallback and the ``fuzzy_match`` path in every document-memory
manager must apply the SAME scope filtering as that manager's scoped base_query.
Before the fix these candidate loads selected on ``user_id`` only, silently
leaking NULL-scope rows to scoped readers (and masking the Postgres bug on dev
machines).

The suite forces the SQLite branch by patching ``settings.mirix_pg_uri_no_default``
to ``None`` on the shared settings singleton's class (the property every manager
consults via ``from mirix.settings import settings``), so the tests are
deterministic regardless of the runner's PG env.

Template: the hermetic throwaway-SQLite pattern from tests/test_skill_experience.py
(module-scoped sqlite+aiosqlite engine, Base.metadata.create_all, an
``@asynccontextmanager`` session_maker matching ``db_context``'s shape, per-manager
``session_maker`` override, and ``@pytest.mark.asyncio(loop_scope="module")``).
"""

import datetime as dt
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register all ORM classes before Base is used
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.knowledge_vault import KnowledgeVaultItem
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.orm.resource_memory import ResourceMemoryItem
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.settings import settings

SCOPE = "test"


# --------------------------------------------------------------------------- #
# Force the SQLite in-memory bm25/fuzzy branch for the whole module.
# Every manager reads ``settings.mirix_pg_uri_no_default`` on the shared singleton
# to choose PG-native fulltext vs the SQLite fallback; returning None routes them
# all through the fallback (the exact path this fix hardens).
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings),
        "mirix_pg_uri_no_default",
        property(lambda self: None),
    )


# --------------------------------------------------------------------------- #
# Hermetic sqlite engine + org/user/agent seed (module scoped).
# --------------------------------------------------------------------------- #
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def session_maker(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("scope_read") / "test.db"
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
    org_id = f"scope-org-{uuid.uuid4().hex[:8]}"
    user_id = f"scope-user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
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
        await session.commit()
    return {"org_id": org_id, "user_id": user_id, "agent_id": agent_id}


@pytest.fixture(scope="module")
def user(seed):
    return PydanticUser(
        id=seed["user_id"],
        name="scope-user",
        organization_id=seed["org_id"],
        timezone="UTC",
    )


@pytest.fixture(scope="module")
def actor(seed):
    return PydanticClient(
        id=f"client-{uuid.uuid4().hex[:8]}",
        organization_id=seed["org_id"],
        name="scope-client",
        write_scope=SCOPE,
        read_scopes=[SCOPE],
    )


@pytest.fixture(scope="module")
def agent_state():
    return AgentState(
        id="agent-scope",
        name="scope-agent",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai", embedding_model="x", embedding_dim=8
        ),
        tools=[],
    )


# --------------------------------------------------------------------------- #
# Per-manager specs: how to build the two rows and how to call the list fn.
# Each spec seeds one row with filter_tags={"scope": SCOPE} and one with
# filter_tags=None, using a shared search token so bm25/fuzzy match both.
# --------------------------------------------------------------------------- #
_TOKEN = "alphaunique"


def _make_procedural_rows(seed, scoped_id, unscoped_id):
    def _row(row_id, filter_tags, suffix):
        return ProceduralMemoryItem(
            id=row_id,
            organization_id=seed["org_id"],
            user_id=seed["user_id"],
            agent_id=seed["agent_id"],
            # Unique name per row: (org, user, name) is a unique constraint, so
            # the bm25 and fuzzy specs must not collide on reused suffixes.
            name=f"skill-{row_id}",
            entry_type="workflow",
            description=f"{_TOKEN} deploy workflow {suffix}",
            instructions="do the thing",
            filter_tags=filter_tags,
        )

    return [
        _row(scoped_id, {"scope": SCOPE}, "scoped"),
        _row(unscoped_id, None, "unscoped"),
    ]


def _make_episodic_rows(seed, scoped_id, unscoped_id):
    def _row(row_id, filter_tags, suffix):
        return EpisodicEvent(
            id=row_id,
            organization_id=seed["org_id"],
            user_id=seed["user_id"],
            agent_id=seed["agent_id"],
            occurred_at=datetime.now(dt.timezone.utc),
            actor="user",
            event_type="note",
            summary=f"{_TOKEN} event {suffix}",
            details="details here",
            filter_tags=filter_tags,
        )

    return [
        _row(scoped_id, {"scope": SCOPE}, "scoped"),
        _row(unscoped_id, None, "unscoped"),
    ]


def _make_semantic_rows(seed, scoped_id, unscoped_id):
    def _row(row_id, filter_tags, suffix):
        return SemanticMemoryItem(
            id=row_id,
            organization_id=seed["org_id"],
            user_id=seed["user_id"],
            agent_id=seed["agent_id"],
            name=f"{_TOKEN} concept {suffix}",
            summary="a summary",
            details="the details",
            source="manual",
            filter_tags=filter_tags,
        )

    return [
        _row(scoped_id, {"scope": SCOPE}, "scoped"),
        _row(unscoped_id, None, "unscoped"),
    ]


def _make_resource_rows(seed, scoped_id, unscoped_id):
    def _row(row_id, filter_tags, suffix):
        return ResourceMemoryItem(
            id=row_id,
            organization_id=seed["org_id"],
            user_id=seed["user_id"],
            agent_id=seed["agent_id"],
            title=f"resource {suffix}",
            summary="a summary",
            content=f"{_TOKEN} resource body {suffix}",
            resource_type="doc",
            filter_tags=filter_tags,
        )

    return [
        _row(scoped_id, {"scope": SCOPE}, "scoped"),
        _row(unscoped_id, None, "unscoped"),
    ]


def _make_knowledge_rows(seed, scoped_id, unscoped_id):
    def _row(row_id, filter_tags, suffix):
        return KnowledgeVaultItem(
            id=row_id,
            organization_id=seed["org_id"],
            user_id=seed["user_id"],
            agent_id=seed["agent_id"],
            entry_type="credential",
            source="manual",
            sensitivity="low",
            secret_value="x",
            caption=f"{_TOKEN} secret note {suffix}",
            filter_tags=filter_tags,
        )

    return [
        _row(scoped_id, {"scope": SCOPE}, "scoped"),
        _row(unscoped_id, None, "unscoped"),
    ]


def _procedural_call(session_maker):
    mgr = ProceduralMemoryManager()
    mgr.session_maker = session_maker

    async def _call(agent_state, user, method, scopes):
        return await mgr.list_procedures(
            agent_state=agent_state,
            user=user,
            query=_TOKEN,
            search_field="description",
            search_method=method,
            scopes=scopes,
            use_cache=False,
        )

    return _call


def _episodic_call(session_maker):
    mgr = EpisodicMemoryManager()
    mgr.session_maker = session_maker

    async def _call(agent_state, user, method, scopes):
        return await mgr.list_episodic_memory(
            agent_state=agent_state,
            user=user,
            query=_TOKEN,
            search_field="summary",
            search_method=method,
            scopes=scopes,
            use_cache=False,
        )

    return _call


def _semantic_call(session_maker):
    mgr = SemanticMemoryManager()
    mgr.session_maker = session_maker

    async def _call(agent_state, user, method, scopes):
        return await mgr.list_semantic_items(
            agent_state=agent_state,
            user=user,
            query=_TOKEN,
            search_field="name",
            search_method=method,
            scopes=scopes,
            use_cache=False,
        )

    return _call


def _resource_call(session_maker):
    mgr = ResourceMemoryManager()
    mgr.session_maker = session_maker

    async def _call(agent_state, user, method, scopes):
        return await mgr.list_resources(
            agent_state=agent_state,
            user=user,
            query=_TOKEN,
            search_field="content",
            search_method=method,
            scopes=scopes,
            use_cache=False,
        )

    return _call


def _knowledge_call(session_maker):
    mgr = KnowledgeVaultManager()
    mgr.session_maker = session_maker

    async def _call(agent_state, user, method, scopes):
        return await mgr.list_knowledge(
            agent_state=agent_state,
            user=user,
            query=_TOKEN,
            search_field="caption",
            search_method=method,
            scopes=scopes,
            use_cache=False,
        )

    return _call


# (id, make_rows, make_call, supports_fuzzy)
_SPECS = [
    ("procedural", _make_procedural_rows, _procedural_call, True),
    ("episodic", _make_episodic_rows, _episodic_call, True),
    ("semantic", _make_semantic_rows, _semantic_call, True),
    ("resource", _make_resource_rows, _resource_call, False),
    ("knowledge_vault", _make_knowledge_rows, _knowledge_call, True),
]


async def _seed_rows(session_maker, rows):
    async with session_maker() as session:
        for row in rows:
            session.add(row)
        await session.commit()


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "manager_id,make_rows,make_call,supports_fuzzy",
    _SPECS,
    ids=[s[0] for s in _SPECS],
)
async def test_bm25_scope_filtering(
    session_maker,
    seed,
    user,
    agent_state,
    manager_id,
    make_rows,
    make_call,
    supports_fuzzy,
):
    scoped_id = f"{manager_id}-scoped-{uuid.uuid4().hex[:6]}"
    unscoped_id = f"{manager_id}-unscoped-{uuid.uuid4().hex[:6]}"
    await _seed_rows(session_maker, make_rows(seed, scoped_id, unscoped_id))
    call = make_call(session_maker)

    # scopes=[SCOPE] -> only the scoped row is visible (NULL scope hidden).
    got = await call(agent_state, user, "bm25", [SCOPE])
    ids = {r.id for r in got}
    assert scoped_id in ids
    assert unscoped_id not in ids, (
        f"{manager_id} bm25 SQLite fallback leaked a NULL-scope row to a scoped reader"
    )

    # scopes=None -> unscoped read returns both rows.
    got = await call(agent_state, user, "bm25", None)
    ids = {r.id for r in got}
    assert {scoped_id, unscoped_id} <= ids

    # scopes=[] -> empty-scope read returns nothing (WHERE 1=0).
    got = await call(agent_state, user, "bm25", [])
    assert got == []


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize(
    "manager_id,make_rows,make_call,supports_fuzzy",
    [s for s in _SPECS if s[3]],
    ids=[s[0] for s in _SPECS if s[3]],
)
async def test_fuzzy_scope_filtering(
    session_maker,
    seed,
    user,
    agent_state,
    manager_id,
    make_rows,
    make_call,
    supports_fuzzy,
):
    scoped_id = f"{manager_id}-fz-scoped-{uuid.uuid4().hex[:6]}"
    unscoped_id = f"{manager_id}-fz-unscoped-{uuid.uuid4().hex[:6]}"
    await _seed_rows(session_maker, make_rows(seed, scoped_id, unscoped_id))
    call = make_call(session_maker)

    got = await call(agent_state, user, "fuzzy_match", [SCOPE])
    ids = {r.id for r in got}
    assert scoped_id in ids
    assert unscoped_id not in ids, (
        f"{manager_id} fuzzy_match candidate load leaked a NULL-scope row"
    )

    got = await call(agent_state, user, "fuzzy_match", None)
    ids = {r.id for r in got}
    assert {scoped_id, unscoped_id} <= ids

    got = await call(agent_state, user, "fuzzy_match", [])
    assert got == []


# --------------------------------------------------------------------------- #
# Write-side round trip: a stamped skill is retrievable by a scoped reader.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio(loop_scope="module")
async def test_procedural_roundtrip_scoped_read(
    session_maker, seed, user, actor, agent_state, monkeypatch
):
    import mirix.services.procedural_memory_manager as pmm
    from mirix.orm.client import Client as ClientORM

    # Avoid real embedding calls in the hermetic test.
    monkeypatch.setattr(pmm, "BUILD_EMBEDDINGS_FOR_MEMORY", False)

    # insert_procedure stamps client_id + audit FKs (_created_by_id) from the
    # actor, so the owning clients row must exist for the FK constraint.
    async with session_maker() as session:
        session.add(
            ClientORM(
                id=actor.id,
                organization_id=seed["org_id"],
                name="scope-client",
                status="active",
                write_scope=SCOPE,
                read_scopes=[SCOPE],
            )
        )
        await session.commit()

    mgr = pmm.ProceduralMemoryManager()
    mgr.session_maker = session_maker

    inserted = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=seed["agent_id"],
        name=f"roundtrip-{uuid.uuid4().hex[:6]}",
        description=f"{_TOKEN} roundtrip skill",
        instructions="steps",
        entry_type="workflow",
        actor=actor,
        organization_id=seed["org_id"],
        filter_tags={"scope": SCOPE},
        use_cache=False,
        user_id=user.id,
    )
    assert inserted.filter_tags == {"scope": SCOPE}

    got = await mgr.list_procedures(
        agent_state=agent_state,
        user=user,
        query=_TOKEN,
        search_field="description",
        search_method="bm25",
        scopes=[SCOPE],
        use_cache=False,
    )
    assert inserted.id in {r.id for r in got}, (
        "a scope-stamped skill must be retrievable by a scoped reader"
    )


# --------------------------------------------------------------------------- #
# Unit: scope_filter_tags(actor)
# --------------------------------------------------------------------------- #
class TestScopeFilterTags:
    def test_with_write_scope(self):
        from mirix.services.skill_experience_curator import scope_filter_tags

        actor = type("A", (), {"write_scope": "admin"})()
        assert scope_filter_tags(actor) == {"scope": "admin"}

    def test_without_write_scope(self):
        from mirix.services.skill_experience_curator import scope_filter_tags

        actor = type("A", (), {"write_scope": None})()
        assert scope_filter_tags(actor) is None

    def test_missing_attr(self):
        from mirix.services.skill_experience_curator import scope_filter_tags

        actor = type("A", (), {})()
        assert scope_filter_tags(actor) is None


# --------------------------------------------------------------------------- #
# Migration text-assertion (TestMigrationSql convention).
# --------------------------------------------------------------------------- #
class TestBackfillMigrationSql:
    from pathlib import Path

    SQL_PATH = Path("scripts/migrate_backfill_procedural_scope.sql")

    def _sql(self) -> str:
        return self.SQL_PATH.read_text()

    def test_contains_required_constructs(self):
        sql = self._sql()
        for token in (
            "BEGIN;",
            "COMMIT;",
            "jsonb_set",
            # Non-object filter_tags (SQL NULL / JSON scalar 'null') must be
            # normalized before jsonb_set, which errors on scalars.
            "jsonb_typeof",
            "FROM clients",
            "write_scope IS NOT NULL",
            "jsonb_exists",
            "'{scope}'",
            "RAISE NOTICE",
        ):
            assert token in sql, f"backfill migration missing {token!r}"


# --------------------------------------------------------------------------- #
# Production wiring: run_experience_evolution must construct the procedural
# agent WITH the actor-derived scope stamp. The round-trip test above supplies
# filter_tags by hand, so without this spy the stamping line in
# skill_experience_curator could be deleted and the suite would stay green.
# --------------------------------------------------------------------------- #
class _StampSentinel(Exception):
    """Raised by the spy to stop the run right after agent construction."""


@pytest.mark.asyncio
async def test_run_experience_evolution_stamps_actor_scope(
    monkeypatch, actor, user, agent_state
):
    import mirix.agent as agent_pkg
    import mirix.server.server as server_module
    import mirix.services.skill_experience_curator as curator

    captured: dict = {}

    class _SpyProceduralAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _StampSentinel

    class _StubAgentManager:
        async def update_agent_tools_and_system_prompts(self, *a, **k):
            raise RuntimeError("refresh skipped in spy test")

    class _StubServer:
        agent_manager = _StubAgentManager()

        def default_interface_factory(self):
            return None

    async def _fake_resolve(server, resolve_actor, meta_agent_state):
        return agent_state

    monkeypatch.setattr(agent_pkg, "ProceduralMemoryAgent", _SpyProceduralAgent)
    monkeypatch.setattr(server_module, "get_server", lambda: _StubServer())
    monkeypatch.setattr(curator, "_resolve_procedural_agent_state", _fake_resolve)

    with pytest.raises(_StampSentinel):
        await curator.run_experience_evolution(
            user=user, actor=actor, meta_agent_state=agent_state
        )

    assert captured["filter_tags"] == {"scope": SCOPE}
    assert captured["actor"] is actor
