"""Every write to procedural memory goes through ``apply_skill_change``.

All paths that change a skill go through it, so every change is versioned,
locked and atomic. Three layers of proof:

1. a SPY on :meth:`ProceduralMemoryManager.apply_skill_change` (delegating to
   the real one): each public write method of the manager, the three curator
   skill tools and the REST restore path is called for real, and must call it
   with the right change type and provenance — and leave a version behind;
2. a CLASSIFICATION of every public method of the manager as a read, a
   versioned write or an erasure: a new method that is not classified fails,
   so it cannot slip past (1);
3. a SOURCE GUARD (AST): inside the manager only the transaction steps of
   ``apply_skill_change`` and the erasure paths construct, update or delete
   skill rows, and nothing else in ``mirix/`` writes ``procedural_memory`` —
   no ORM construction, no ``update()/delete()/insert()`` statement, no raw
   SQL, no ``*_with_redis`` / ``hard_delete`` helper.

Hermetic: SQLite, a fake embedding model, no Redis, no model provider.
"""

import ast
import inspect
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix.functions.function_sets import memory_tools as mt
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.skill_version import SkillVersion as SkillVersionORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.procedural_memory import (
    ProceduralMemoryItem as PydanticProceduralMemoryItem,
)
from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate
from mirix.schemas.user import User as PydanticUser
from mirix.services.procedural_memory_manager import (
    ACTOR_AGENT,
    ACTOR_API,
    ACTOR_CURATOR,
    ACTOR_SYSTEM,
    ProceduralMemoryManager,
)
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

REPO = Path(__file__).resolve().parent.parent
MANAGER_FILE = REPO / "mirix" / "services" / "procedural_memory_manager.py"


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_write_paths") / "test.db"
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


class _FakeEmbedModel:
    async def get_text_embedding(self, text):
        return [0.1] * 8


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = ProceduralMemoryManager.apply_skill_change

    async def recording(self, change_type, **kwargs):
        calls.append(SimpleNamespace(change_type=change_type, **kwargs))
        return await real(self, change_type, **kwargs)

    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel()

    monkeypatch.setattr(ProceduralMemoryManager, "apply_skill_change", recording)
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    return calls


@pytest_asyncio.fixture(loop_scope="module")
async def env(sm, spy):
    org = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.flush()
        db.add(
            UserORM(
                id=user_id,
                name="u",
                organization_id=org,
                status="active",
                timezone="UTC",
            )
        )
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org,
                name="c",
                status="active",
                write_scope="test",
                read_scopes=["test"],
            )
        )
        db.add(AgentORM(id=agent_id, organization_id=org))
        await db.commit()
    actor = PydanticClient(
        id=client_id,
        organization_id=org,
        name="c",
        write_scope="test",
        read_scopes=["test"],
    )
    user = PydanticUser(id=user_id, name="u", organization_id=org, timezone="UTC")
    agent_state = AgentState(
        id=agent_id,
        name="proc",
        system="s",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=8,
        ),
        tools=[],
    )
    mgr = ProceduralMemoryManager(session_maker=sm)
    return SimpleNamespace(
        sm=sm,
        mgr=mgr,
        actor=actor,
        user=user,
        agent_state=agent_state,
        spy=spy,
        org=org,
    )


async def _versions(env, skill_id):
    async with env.sm() as db:
        rows = (
            (
                await db.execute(
                    select(SkillVersionORM)
                    .where(SkillVersionORM.skill_id == skill_id)
                    .order_by(SkillVersionORM.version_no)
                )
            )
            .scalars()
            .all()
        )
    return [(r.version_no, r.change_type, r.actor_kind, r.run_id) for r in rows]


def _item(env, name):
    return PydanticProceduralMemoryItem(
        name=name,
        entry_type="workflow",
        description=f"d {name}",
        instructions=f"i {name}",
        user_id=env.user.id,
        organization_id=env.org,
        filter_tags={"scope": "test"},
    )


# --------------------------------------------------------------------------- #
# 1. The spy: every write method goes through apply_skill_change               #
# --------------------------------------------------------------------------- #


class TestManagerWriteMethods:
    async def test_create_item(self, env):
        created = await env.mgr.create_item(
            _item(env, "ci"), env.actor, user_id=env.user.id
        )
        assert [c.change_type for c in env.spy] == ["create"]
        assert await _versions(env, created.id) == [(1, "create", ACTOR_API, None)]

    async def test_insert_procedure(self, env):
        created = await env.mgr.insert_procedure(
            agent_state=env.agent_state,
            agent_id=env.agent_state.id,
            name="ip",
            description="d",
            instructions="i",
            entry_type="workflow",
            actor=env.actor,
            organization_id=env.org,
            user_id=env.user.id,
            run_id="run-1:b0",
            actor_kind=ACTOR_CURATOR,
        )
        assert [c.change_type for c in env.spy] == ["create"]
        assert await _versions(env, created.id) == [
            (1, "create", ACTOR_CURATOR, "run-1:b0")
        ]

    async def test_create_many_items(self, env):
        created = await env.mgr.create_many_items(
            [_item(env, "m1"), _item(env, "m2")], env.user, actor=env.actor
        )
        assert [c.change_type for c in env.spy] == ["create", "create"]
        for skill in created:
            assert await _versions(env, skill.id) == [(1, "create", ACTOR_API, None)]

    async def test_update_item(self, env):
        created = await env.mgr.create_item(
            _item(env, "ui"), env.actor, user_id=env.user.id
        )
        env.spy.clear()
        await env.mgr.update_item(
            ProceduralMemoryItemUpdate(id=created.id, instructions="new"),
            user=env.user,
            actor=env.actor,
            run_id="run-2:b0",
            actor_kind=ACTOR_CURATOR,
        )
        assert [c.change_type for c in env.spy] == ["edit"]
        assert (await _versions(env, created.id))[-1] == (
            2,
            "edit",
            ACTOR_CURATOR,
            "run-2:b0",
        )

    async def test_soft_delete_and_restore(self, env):
        created = await env.mgr.create_item(
            _item(env, "sd"), env.actor, user_id=env.user.id
        )
        env.spy.clear()
        await env.mgr.soft_delete_procedure_by_id(
            created.id, actor=env.actor, user=env.user, reason="wrong"
        )
        await env.mgr.restore_skill(created.id, 1, actor=env.actor, user_id=env.user.id)
        assert [c.change_type for c in env.spy] == ["delete", "restore"]
        assert [v[1] for v in await _versions(env, created.id)] == [
            "create",
            "delete",
            "restore",
        ]

    async def test_bulk_soft_deletes(self, env):
        a = await env.mgr.create_item(_item(env, "bu"), env.actor, user_id=env.user.id)
        b = await env.mgr.create_item(_item(env, "bc"), env.actor, user_id=env.user.id)
        await env.mgr.soft_delete_procedure_by_id(b.id, actor=env.actor, user=env.user)
        await env.mgr.restore_skill(b.id, 1, actor=env.actor, user_id=env.user.id)
        env.spy.clear()
        assert await env.mgr.soft_delete_by_user_id(env.user.id) == 2
        assert [(c.change_type, c.actor_kind) for c in env.spy] == [
            ("delete", ACTOR_SYSTEM),
            ("delete", ACTOR_SYSTEM),
        ]
        await env.mgr.restore_skill(a.id, 1, actor=env.actor, user_id=env.user.id)
        env.spy.clear()
        assert await env.mgr.soft_delete_by_client_id(env.actor) == 1
        assert [c.change_type for c in env.spy] == ["delete"]

    async def test_reembed_skill(self, env):
        # Filling a missing retrieval vector is an edit through the one
        # write path, guarded by the retrieval text it embedded.
        created = await env.mgr.create_item(
            _item(env, "re"), env.actor, user_id=env.user.id
        )
        assert created.retrieval_embedding is None  # create_item got no agent
        async with env.sm() as db:
            row = (
                await db.execute(
                    select(pmm.ProceduralMemoryItem).where(
                        pmm.ProceduralMemoryItem.id == created.id
                    )
                )
            ).scalar_one()
            row.embedding_config = env.agent_state.embedding_config
            await db.commit()
        env.spy.clear()
        assert await env.mgr.reembed_skill(created.id) == pmm.REEMBED_FILLED
        [call] = env.spy
        assert call.change_type == "edit" and call.actor_kind == ACTOR_SYSTEM
        assert set(call.expected_fields) == {"name", "triggers", "description"}
        assert set(call.fields) == {"retrieval_embedding"}
        # No version: the vector is not part of the history.
        assert await _versions(env, created.id) == [(1, "create", ACTOR_API, None)]


class TestNulCharacters:
    """A NUL character (U+0000) in any stored field of a
    create or an edit is refused by the one write path with a clear
    ``SkillFieldInvalid`` — before the skill is read, embedded or written —
    whichever manager method or caller made it. A soft delete or a restore
    of an existing skill is not affected."""

    @pytest.mark.parametrize(
        "field, value, named",
        [
            ("name", "nul\x00name", "'name'"),
            ("description", "d\x00", "'description'"),
            ("instructions", "a\x00b", "'instructions'"),
            ("triggers", ["ok", "bad\x00"], "'triggers[1]'"),
            ("examples", [{"in\x00put": "x"}], "'examples[0]'"),
        ],
    )
    async def test_a_create_or_edit_holding_one_is_refused_first(
        self, env, monkeypatch, field, value, named
    ):
        embedded = []

        class Counting:
            async def get_text_embedding(self, text):
                embedded.append(text)
                return [0.1] * 8

        async def counting_model(config, *args, **kwargs):
            return Counting()

        monkeypatch.setattr(pmm, "embedding_model", counting_model)
        fields = dict(
            name=f"nul-{field}",
            entry_type="workflow",
            description="d",
            instructions="i",
            triggers=["t"],
            examples=[],
            organization_id=env.org,
        )
        fields[field] = value
        with pytest.raises(pmm.SkillFieldInvalid, match="U\\+0000") as raised:
            await env.mgr.apply_skill_change(
                "create",
                actor=env.actor,
                user_id=env.user.id,
                fields=fields,
                agent_state=env.agent_state,
            )
        assert named in str(raised.value)
        assert isinstance(raised.value, ValueError)

        existing = await env.mgr.create_item(
            _item(env, f"nul-edit-{field}"), env.actor, user_id=env.user.id
        )
        embedded.clear()
        with pytest.raises(pmm.SkillFieldInvalid, match=re.escape(named)):
            await env.mgr.apply_skill_change(
                "edit",
                actor=env.actor,
                user_id=env.user.id,
                skill_id=existing.id,
                fields={field: value},
                agent_state=env.agent_state,
            )
        assert embedded == []
        assert await _versions(env, existing.id) == [(1, "create", ACTOR_API, None)]
        # Through the schema-typed method too (ProceduralMemoryItemUpdate).
        if field in ("description", "instructions", "triggers"):
            with pytest.raises(pmm.SkillFieldInvalid):
                await env.mgr.update_item(
                    ProceduralMemoryItemUpdate(id=existing.id, **{field: value}),
                    user=env.user,
                    actor=env.actor,
                )
        # A delete and a restore of the skill still work.
        await env.mgr.soft_delete_procedure_by_id(
            existing.id, actor=env.actor, user=env.user, reason="r"
        )
        await env.mgr.restore_skill(
            existing.id, 1, actor=env.actor, user_id=env.user.id
        )


class TestCuratorTools:
    def _agent(self, env, *, run_id=None):
        agent = SimpleNamespace(
            procedural_memory_manager=env.mgr,
            agent_state=env.agent_state,
            user=env.user,
            actor=env.actor,
            user_id=env.user.id,
            filter_tags={"scope": "test"},
            use_cache=True,
        )
        if run_id is not None:
            agent.skill_run_id = run_id
        return agent

    async def test_create_edit_delete_are_versioned_curator_changes(self, env):
        agent = self._agent(env, run_id="run-7:b1")
        created = await mt.skill_create(
            agent,
            name="tool-skill",
            # Real content: the skill tools reject placeholders and schema
            # echoes ("what it does", a trigger "when"), in or out of a run.
            description="Clears the review queue from the dashboard",
            instructions="Open the dashboard, check the queue, then do the thing.",
            entry_type="workflow",
            triggers=["queue backlog"],
        )
        assert "created successfully" in created
        skill_id = created.split("ID: ")[1].split(",")[0]
        edited = await mt.skill_edit(
            agent, skill_id, "instructions", old_text="the thing", new_text="the task"
        )
        assert "updated" in edited
        deleted = await mt.skill_delete(agent, skill_id, reason="merged into another skill")
        assert "soft-deleted" in deleted
        assert env.spy[-1].reason == "merged into another skill"
        assert [c.change_type for c in env.spy] == ["create", "edit", "delete"]
        assert await _versions(env, skill_id) == [
            (1, "create", ACTOR_CURATOR, "run-7:b1"),
            (2, "edit", ACTOR_CURATOR, "run-7:b1"),
            (3, "delete", ACTOR_CURATOR, "run-7:b1"),
        ]

    async def test_outside_a_learning_run_the_tools_are_agent_changes(self, env):
        agent = self._agent(env)
        created = await mt.skill_create(
            agent,
            name="in-flow",
            description="Opens an item and checks it",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
        skill_id = created.split("ID: ")[1].split(",")[0]
        assert await _versions(env, skill_id) == [(1, "create", ACTOR_AGENT, None)]


# --------------------------------------------------------------------------- #
# 2. Every public manager method is classified                                 #
# --------------------------------------------------------------------------- #

READS = {
    "get_item_by_id",
    "get_live_skill_by_name",
    "get_skill_labels",
    "list_skills_page",
    "get_most_recently_updated_item",
    "get_total_number_of_items",
    "list_procedures",
    "list_procedures_by_org",
    "get_skill_history",
    "get_skill_version",
    "list_skill_ids_missing_retrieval_embedding",
    "retrieve_skills",  # the skill ranking (tests/test_skill_retrieval.py)
    "get_served_skill_texts",  # The served versions' texts, rebuilt
    "list_live_skill_names",  # The names a curator batch may mention
}
# Covered by TestManagerWriteMethods / TestCuratorTools above.
VERSIONED_WRITES = {
    "apply_skill_change",
    "create_item",
    "insert_procedure",
    "create_many_items",
    "update_item",
    "soft_delete_procedure_by_id",
    "restore_skill",
    "soft_delete_by_user_id",
    "soft_delete_by_client_id",
    "ensure_baseline_version",  # writes only a baseline version, never the skill
    "reembed_skill",  # the retrieval vector, through apply_skill_change
}
# Erasure removes the skill with its whole history (tests/test_skill_history.py).
ERASURES = {
    "delete_procedure_by_id",
    "delete_by_user_id",
    "delete_by_client_id",
    "purge_organization_history",
}


def test_every_public_manager_method_is_classified():
    public = {
        name
        for name, member in inspect.getmembers(ProceduralMemoryManager)
        if not name.startswith("_") and callable(member)
    }
    unclassified = public - READS - VERSIONED_WRITES - ERASURES
    assert not unclassified, (
        f"classify {sorted(unclassified)} as a read, a versioned write (and "
        "cover it in TestManagerWriteMethods) or an erasure"
    )
    assert not (READS | VERSIONED_WRITES | ERASURES) - public


# --------------------------------------------------------------------------- #
# 3. Source guard                                                              #
# --------------------------------------------------------------------------- #

_SKILL_MODEL_NAMES = {"ProceduralMemoryItem", "ProceduralMemoryModel"}
_ROW_FIELDS = {
    "is_deleted",
    "name",
    "entry_type",
    "description",
    "instructions",
    "triggers",
    "examples",
    "filter_tags",
    "retrieval_embedding",
    "last_modify",
    "version",
    "user_id",
    "organization_id",
}
_FORBIDDEN_HELPERS = {
    "create_with_redis",
    "update_with_redis",
    "delete_with_redis",
    "hard_delete",
}
# The only places the manager may write skill rows: the transaction steps of
# apply_skill_change, and the erasure paths.
_ALLOWED_WRITERS = {"_commit_create", "_commit_existing", "_purge_skills"}


def _skill_writes(tree, *, row_names=("row",)):
    """(line, what) for every skill-row write under ``tree``."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _SKILL_MODEL_NAMES:
                found.append((node.lineno, f"{func.id}(...) constructed"))
            if (
                isinstance(func, ast.Name)
                and func.id in {"update", "delete", "insert"}
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in _SKILL_MODEL_NAMES
            ):
                found.append((node.lineno, f"{func.id}({node.args[0].id})"))
            if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_HELPERS:
                found.append((node.lineno, f".{func.attr}()"))
            if isinstance(func, ast.Name) and func.id == "setattr" and node.args:
                target = node.args[0]
                if isinstance(target, ast.Name) and target.id in row_names:
                    found.append((node.lineno, "setattr(row, ...)"))
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in row_names
                    and target.attr in _ROW_FIELDS
                ):
                    found.append((node.lineno, f"{target.value.id}.{target.attr} ="))
    return found


def test_the_manager_writes_skill_rows_only_inside_apply_skill_change_and_erasure():
    tree = ast.parse(MANAGER_FILE.read_text())
    manager = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ProceduralMemoryManager"
    )
    offenders = []
    for method in manager.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        writes = _skill_writes(method, row_names=("row", "item", "procedure", "skill"))
        if writes and method.name not in _ALLOWED_WRITERS:
            offenders.append((method.name, writes))
    assert not offenders, offenders


def test_nothing_else_in_mirix_writes_procedural_memory():
    raw_sql = re.compile(
        r"(UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+procedural_memory\b", re.IGNORECASE
    )
    allowed = {MANAGER_FILE.resolve()}
    offenders = []
    for path in (REPO / "mirix").rglob("*.py"):
        if path.resolve() in allowed:
            continue
        source = path.read_text()
        if raw_sql.search(source):
            offenders.append((str(path), "raw SQL write"))
        if "ProceduralMemoryItem" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {
                    "update",
                    "delete",
                    "insert",
                }:
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Name)
                        and (node.args[0].id in _SKILL_MODEL_NAMES)
                    ):
                        offenders.append((str(path), node.lineno))
                if (
                    isinstance(func, ast.Name)
                    and func.id in _SKILL_MODEL_NAMES
                    and "orm" not in path.parts
                ):
                    offenders.append((str(path), node.lineno))
    assert not offenders, offenders


def test_skill_history_module_never_writes_skill_rows():
    tree = ast.parse((REPO / "mirix" / "services" / "skill_history.py").read_text())
    assert _skill_writes(tree) == []
