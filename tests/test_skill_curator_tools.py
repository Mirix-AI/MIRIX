"""The skill tools inside and outside a curator batch.

Against a hermetic SQLite database with the real ProceduralMemoryManager,
SkillExperienceManager and SkillLearningRunManager (a fake embedding model; no
model provider):

* edit budget = the number of experiences of the batch: exactly that many
  creates/edits go through, the next is refused and writes nothing;
* every validation — the duplicate-name check included — runs before the
  write, so a rejected call costs nothing;
* at most one soft delete per learning run, across batches; a delete needs a
  reason and does not use the edit budget; no size gate: a whole-document
  rewrite passes; SKILL_MAX_INSTRUCTIONS_CHARS still asks to condense;
* scope checks by id and by name;
* ``skill_list`` pages through every skill with the total and the next
  offset; ``skill_read`` is capped head + tail with a counted metric;
* the curator's write guard runs inside the write's transaction: a run that is
  no longer active cannot change a skill, and the refusal ends the step (it is
  not a tool result); every write stamps the batch lineage.
"""

import ast
import json
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
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.orm.skill_version import SkillVersion as SkillVersionORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.skill_curator_state import (
    SKILL_LIST_PAGE_SIZE,
    CuratorBatchState,
    CuratorRunLimits,
    head_tail_chars,
    skill_output_char_cap,
)
from mirix.services.skill_experience_manager import SkillExperienceManager
from mirix.services.skill_learning_run_manager import (
    LearningRunNotActive,
    SkillLearningRunManager,
)
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

REPO = Path(__file__).resolve().parent.parent
SCOPE = "tools-scope"


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_curator_tools") / "test.db"
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


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel()

    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)


@pytest_asyncio.fixture(loop_scope="module")
async def env(sm, monkeypatch):
    import mirix.server.server as server_module

    # SkillLearningRunManager / SkillExperienceManager default to db_context.
    monkeypatch.setattr(server_module, "db_context", sm)
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
                write_scope=SCOPE,
                read_scopes=[SCOPE],
            )
        )
        db.add(AgentORM(id=agent_id, organization_id=org))
        await db.commit()
    actor = PydanticClient(
        id=client_id,
        organization_id=org,
        name="c",
        write_scope=SCOPE,
        read_scopes=[SCOPE],
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
    runs = SkillLearningRunManager(session_maker=sm)
    exps = SkillExperienceManager()
    exps.session_maker = sm
    return SimpleNamespace(
        sm=sm,
        mgr=mgr,
        runs=runs,
        exps=exps,
        actor=actor,
        user=user,
        agent_state=agent_state,
        org=org,
        agent_id=agent_id,
    )


def _agent(env, *, batch=None, scope=SCOPE):
    agent = SimpleNamespace(
        procedural_memory_manager=env.mgr,
        agent_state=env.agent_state,
        user=env.user,
        actor=env.actor,
        user_id=env.user.id,
        filter_tags={"scope": scope} if scope else None,
    )
    if batch is not None:
        agent.curator_batch = batch
        agent.skill_run_id = batch.batch_id
        agent.learning_run_id = batch.run_id
    return agent


def _batch(
    budget, *, run_limits=None, batch_id="run-t:b0", guard=None, ids=(), cap=60_000
):
    limits = run_limits or CuratorRunLimits(run_id=batch_id.split(":")[0])
    return CuratorBatchState(
        run_id=limits.run_id,
        batch_id=batch_id,
        experience_ids=tuple(ids),
        budget=budget,
        run_limits=limits,
        write_guard=guard,
        output_char_cap=cap,
    )


def _id(message: str) -> str:
    return message.split("ID: ")[1].split(",")[0].split(")")[0]


def _cite(*ids, quote=""):
    """``experience_ids`` citing ``ids``, each with ``quote``."""
    return [{"id": i, "quote": quote} for i in ids]


async def _create(agent, name, **overrides):
    fields = dict(
        name=name,
        description=f"what {name} does",
        instructions=f"Steps for {name}: open it, check it, finish it.",
        entry_type="workflow",
    )
    fields.update(overrides)
    return await mt.skill_create(agent, **fields)


async def _count_versions(env) -> int:
    async with env.sm() as db:
        return len((await db.execute(select(SkillVersionORM.id))).all())


async def _skill_names(env, *, include_deleted=False):
    async with env.sm() as db:
        stmt = select(ProceduralMemoryItem.name).where(
            ProceduralMemoryItem.user_id == env.user.id
        )
        if not include_deleted:
            stmt = stmt.where(~ProceduralMemoryItem.is_deleted)
        return sorted(r[0] for r in (await db.execute(stmt)).all())


# --------------------------------------------------------------------------- #
# Edit budget                                                                  #
# --------------------------------------------------------------------------- #


class TestEditBudget:
    async def test_the_budget_is_exactly_the_number_of_experiences(self, env):
        batch = _batch(3)
        agent = _agent(env, batch=batch)
        first = await _create(agent, "budget-a")
        assert "created successfully" in first
        second = await _create(agent, "budget-b")
        third = await mt.skill_edit(
            agent, _id(first), "instructions", old_text="check it", new_text="verify it"
        )
        assert "created successfully" in second and "updated" in third
        assert batch.spent == 3 and batch.remaining == 0
        versions_before = await _count_versions(env)

        fourth = await _create(agent, "budget-c")
        assert "budget exhausted" in fourth.lower()
        # The next step is the triage of what no change cites, then the finish
        # (a finish with unaddressed experiences is refused once).
        assert "experience_triage" in fourth and "finish_memory_update()" in fourth
        edit = await mt.skill_edit(
            agent, _id(second), "description", value="a different description"
        )
        assert "budget exhausted" in edit.lower()
        # Nothing was written by the refused calls.
        assert await _count_versions(env) == versions_before
        assert "budget-c" not in await _skill_names(env)
        assert batch.spent == 3
        assert batch.changes() == {
            "created": [_id(first), _id(second)],
            "edited": [],
            "deleted": [],
        }

    async def test_rejected_writes_cost_nothing(self, env):
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        existing = await mt.skill_create(
            _agent(env),
            name="taken-name",
            description="a skill that already has this name",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
        versions_before = await _count_versions(env)
        rejected = [
            # The duplicate check runs before anything is charged.
            await _create(agent, "taken-name"),
            await _create(agent, "bad-type", entry_type="recipe"),
            await _create(agent, "  "),
            await _create(agent, "bad-triggers", triggers=[1, 2]),
            await _create(agent, "too-long", instructions="x " * 6_001),
            await mt.skill_edit(agent, "proc-missing", "instructions", value="new"),
            await mt.skill_edit(agent, _id(existing), "bogus", value="x"),
            await mt.skill_edit(
                agent, _id(existing), "instructions", old_text="nope", new_text="x"
            ),
            # Whitespace anchors nothing (it would insert at the start).
            await mt.skill_edit(
                agent, _id(existing), "instructions", old_text="  \n ", new_text="x"
            ),
            await mt.skill_edit(
                agent,
                _id(existing),
                "instructions",
                value="Open the item, check it, then close it.",
            ),  # no change
            await mt.skill_edit(agent, _id(existing), "triggers", value="not json"),
            await mt.skill_edit(
                agent, _id(existing), "examples", value='["not an object"]'
            ),
            await mt.skill_edit(agent, _id(existing), "entry_type", value="recipe"),
            await mt.skill_edit(
                agent, _id(existing), "name", value="taken-name"
            ),  # no change
            await mt.skill_edit(
                agent,
                _id(existing),
                "instructions",
                old_text="i",
                new_text="x",
                value="y",
            ),
        ]
        assert all(
            "created successfully" not in r and "updated" not in r for r in rejected
        )
        assert "condense" in rejected[4]
        assert batch.spent == 0 and len(batch.rejected) == len(rejected)
        assert await _count_versions(env) == versions_before
        kinds = batch.rejected_by_kind()
        assert kinds["duplicate_name"] == 1 and kinds["too_long"] == 1
        assert kinds["not_found"] == 1 and kinds["no_change"] == 2
        # The whole budget is still there for a valid write.
        assert "created successfully" in await _create(agent, "after-rejections")
        assert batch.spent == 1

    async def test_schema_limits_are_rejections_before_any_embedding(
        self, env, monkeypatch
    ):
        """A trigger past SKILL_MAX_TRIGGER_LEN or too many examples is a
        rejection the model can act on, found before the retrieval vector is
        computed — not a tool error raised by the write's own validation after
        an embedding call (found by the offline end-to-end test)."""
        from mirix.schemas.procedural_memory import (
            SKILL_MAX_EXAMPLES_COUNT,
            SKILL_MAX_TRIGGER_LEN,
        )

        embedded = []

        class Counting:
            async def get_text_embedding(self, text):
                embedded.append(text)
                return [0.1] * 8

        async def counting_model(config, *args, **kwargs):
            return Counting()

        monkeypatch.setattr(pmm, "embedding_model", counting_model)
        existing = await _create(_agent(env), "limits-base")
        embedded.clear()
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        versions_before = await _count_versions(env)
        long_trigger = "t" * (SKILL_MAX_TRIGGER_LEN + 1)
        too_many = [
            {"input": f"order {i} arrived late", "output": f"refund order {i}"}
            for i in range(SKILL_MAX_EXAMPLES_COUNT + 1)
        ]
        rejected = [
            await _create(agent, "limits-a", triggers=["ok", long_trigger]),
            await _create(agent, "limits-b", examples=too_many),
            await mt.skill_edit(
                agent, _id(existing), "triggers", value=f'["ok", "{long_trigger}"]'
            ),
            await mt.skill_edit(
                agent, _id(existing), "examples", value=json.dumps(too_many)
            ),
        ]
        assert "triggers[1]" in rejected[0] and "triggers[1]" in rejected[2]
        assert str(SKILL_MAX_EXAMPLES_COUNT) in rejected[1]
        assert str(SKILL_MAX_EXAMPLES_COUNT) in rejected[3]
        assert batch.rejected_by_kind() == {"invalid_arguments": 4}
        assert batch.spent == 0 and embedded == []
        assert await _count_versions(env) == versions_before
        # At the limit it goes through.
        ok = await _create(
            agent,
            "limits-c",
            triggers=["t" * SKILL_MAX_TRIGGER_LEN],
            examples=too_many[:-1],
        )
        assert "created successfully" in ok and batch.spent == 1

    async def test_a_nul_character_is_a_rejection_before_any_embedding(
        self, env, monkeypatch
    ):
        """Postgres cannot store U+0000 (the keyword
        index decodes the triggers through JSONB: SQLSTATE 22P05), so a NUL
        in any stored field is a rejection the model can act on, found before
        the retrieval vector is paid for — for a create, a whole-field edit, a
        patch and a JSON field edit alike."""
        embedded = []

        class Counting:
            async def get_text_embedding(self, text):
                embedded.append(text)
                return [0.1] * 8

        async def counting_model(config, *args, **kwargs):
            return Counting()

        monkeypatch.setattr(pmm, "embedding_model", counting_model)
        existing = await _create(_agent(env), "nul-base")
        embedded.clear()
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        versions_before = await _count_versions(env)
        rejected = [
            await _create(agent, "nul-a", triggers=["fine", "bad\x00trigger"]),
            await _create(agent, "nul\x00b"),
            await _create(agent, "nul-c", description="a\x00b"),
            await _create(agent, "nul-d", instructions="step\x00two"),
            await _create(agent, "nul-e", examples=[{"input": "x\x00"}]),
            await mt.skill_edit(
                agent, _id(existing), "triggers", value='["ok", "bad\\u0000"]'
            ),
            await mt.skill_edit(agent, _id(existing), "description", value="d\x00"),
            await mt.skill_edit(
                agent,
                _id(existing),
                "instructions",
                old_text="open it",
                new_text="open\x00it",
            ),
        ]
        for message in rejected:
            assert "NUL character (U+0000)" in message, message
        assert "triggers[1]" in rejected[0] and "triggers[1]" in rejected[5]
        assert "'name'" in rejected[1] and "examples[0]" in rejected[4]
        assert batch.rejected_by_kind() == {"invalid_arguments": len(rejected)}
        assert batch.spent == 0 and embedded == []
        assert await _count_versions(env) == versions_before

    async def test_a_name_held_by_another_scope_is_refused_without_leaking_its_id(
        self, env
    ):
        other = await mt.skill_create(
            _agent(env, scope="other-scope"),
            name="shared-name",
            description="a skill of another scope",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
        batch = _batch(1)
        answer = await _create(_agent(env, batch=batch), "shared-name")
        assert "cannot access" in answer and _id(other) not in answer
        assert batch.spent == 0

    async def test_outside_a_learning_run_there_is_no_budget(self, env):
        agent = _agent(env)
        for i in range(4):
            assert "created successfully" in await _create(agent, f"inflow-{i}")


class TestNoSizeGate:
    async def test_a_whole_document_rewrite_passes(self, env):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        created = await _create(agent, "rewrite-me", instructions="old text " * 200)
        skill_id = _id(created)
        new_doc = "Completely new procedure. " * 300  # ~7,800 chars, all different
        answer = await mt.skill_edit(agent, skill_id, "instructions", value=new_doc)
        assert "updated" in answer and batch.spent == 2
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.instructions == new_doc

    async def test_a_patch_replacing_the_whole_text_passes_too(self, env):
        agent = _agent(env, batch=_batch(2))
        old = "step one. step two. step three."
        skill_id = _id(await _create(agent, "patch-all", instructions=old))
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text=old,
            new_text="replace the old step " * 450,
        )
        assert "updated" in answer

    async def test_replacement_text_is_literal(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "literal", instructions="use PATH here"))
        await mt.skill_edit(
            agent, skill_id, "instructions", old_text="PATH", new_text=r"C:\new\1 dir"
        )
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.instructions == r"use C:\new\1 dir here"

    async def test_a_patch_matches_across_any_whitespace(self, env):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(agent, "spacing", instructions="open the\n  file, then save")
        )
        # Two spaces where the field has a newline and indentation, and vice versa.
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="the  file,\nthen",
            new_text="the file and then",
        )
        assert "updated" in answer, answer
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.instructions == "open the file and then save"
        assert batch.spent == 2

    async def test_the_instructions_ceiling_asks_to_condense(self, env):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(agent, "ceiling", instructions="a short set of steps")
        )
        answer = await mt.skill_edit(
            agent, skill_id, "instructions", value="y " * 6_001
        )
        assert "condense" in answer.lower() and batch.spent == 1


# --------------------------------------------------------------------------- #
# Concurrent writers and ambiguous commits                                     #
# --------------------------------------------------------------------------- #


def _race_before_the_first_write(env, monkeypatch, skill_id, **concurrent):
    """The first apply_skill_change after this runs another writer's edit of
    ``skill_id`` first — a change committed between the tool's read and its
    write."""
    real = env.mgr.apply_skill_change
    state = {"raced": False}

    async def racing(change_type, **kwargs):
        if not state["raced"]:
            state["raced"] = True
            await real(
                "edit",
                actor=env.actor,
                user_id=env.user.id,
                skill_id=skill_id,
                fields=concurrent,
            )
        return await real(change_type, **kwargs)

    monkeypatch.setattr(env.mgr, "apply_skill_change", racing)


class _Busy(Exception):
    sqlite_errorcode = 5  # SQLITE_BUSY: a transient database failure


def _commit_then_fail_once(monkeypatch):
    from sqlalchemy.exc import OperationalError

    real = ProceduralMemoryManager._commit_change
    state = {"failed": False}

    async def commit_then_fail(self, plan, **kwargs):
        result = await real(self, plan, **kwargs)
        if not state["failed"]:
            state["failed"] = True  # committed, but the client saw an error
            raise OperationalError("COMMIT", {}, _Busy("database is locked"))
        return result

    monkeypatch.setattr(ProceduralMemoryManager, "_commit_change", commit_then_fail)


class TestConcurrentWriters:
    async def test_a_patch_lands_on_the_value_written_meanwhile(self, env, monkeypatch):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(
                agent,
                "raced-patch",
                instructions="open the file\ncheck the totals\nsave the report",
            )
        )
        _race_before_the_first_write(
            env,
            monkeypatch,
            skill_id,
            instructions="open the file\ncheck the totals\nsave the report\nmail it",
            version="0.1.1",
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="check the totals",
            new_text="recount the totals",
        )
        assert "updated" in answer, answer
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        # The concurrent "mail it" survives; the patch applied once, charged once.
        assert skill.instructions == (
            "open the file\nrecount the totals\nsave the report\nmail it"
        )
        assert skill.version == "0.1.2"  # bumped from the locked row's 0.1.1
        assert batch.spent == 2

    async def test_a_rewrite_over_a_change_made_meanwhile_is_refused_free(
        self, env, monkeypatch
    ):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(agent, "raced-rewrite", description="the first description")
        )
        _race_before_the_first_write(env, monkeypatch, skill_id, description="theirs")
        answer = await mt.skill_edit(
            agent, skill_id, "description", value="my own new description"
        )
        assert "changed while this edit was being made" in answer
        assert batch.rejected_by_kind() == {"changed_meanwhile": 1}
        assert batch.spent == 1  # only the create
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.description == "theirs"

    async def test_a_change_to_another_field_does_not_block_an_edit(
        self, env, monkeypatch
    ):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "raced-other-field"))
        _race_before_the_first_write(env, monkeypatch, skill_id, description="theirs")
        answer = await mt.skill_edit(agent, skill_id, "entry_type", value="guide")
        assert "updated" in answer, answer
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert (skill.description, skill.entry_type) == ("theirs", "guide")

    async def test_json_and_name_fields_edit_without_a_false_conflict(self, env):
        # The optimistic check compares canonical bytes: a JSON field read
        # back from the item (key order, list types) matches the locked row.
        batch = _batch(4)
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(
                agent,
                "json-fields",
                triggers=["on error"],
                examples=[{"output": "save the file", "input": "open the file"}],
            )
        )
        for field, value in (
            ("triggers", '["on error", "nightly"]'),
            ("examples", '[{"output": "fill every field", "input": "open the form"}]'),
            ("name", "json-fields-renamed"),
        ):
            answer = await mt.skill_edit(agent, skill_id, field, value=value)
            assert "updated" in answer, (field, answer)
        skill = await env.mgr.get_item_by_id(
            item_id=skill_id, user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.triggers == ["on error", "nightly"]
        assert skill.examples == [
            {"output": "fill every field", "input": "open the form"}
        ]
        assert skill.name == "json-fields-renamed" and skill.version == "0.1.3"
        assert batch.spent == 4 and batch.rejected == []

    async def test_an_ambiguous_commit_of_a_create_is_charged_once(
        self, env, monkeypatch
    ):
        batch = _batch(2)
        _commit_then_fail_once(monkeypatch)
        answer = await _create(_agent(env, batch=batch), "ambiguous-create")
        assert "created successfully" in answer, answer
        assert batch.spent == 1 and batch.rejected == []
        assert (await _skill_names(env)).count("ambiguous-create") == 1

    async def test_an_ambiguous_commit_of_a_delete_uses_the_runs_delete(
        self, env, monkeypatch
    ):
        setup = _agent(env)
        first = _id(await _create(setup, "ambiguous-del-a"))
        second = _id(await _create(setup, "ambiguous-del-b"))
        limits = CuratorRunLimits(run_id="run-amb")
        batch = _batch(1, run_limits=limits, batch_id="run-amb:b0")
        _commit_then_fail_once(monkeypatch)
        assert "soft-deleted" in await mt.skill_delete(
            _agent(env, batch=batch), first, reason="wrong"
        )
        assert limits.deletes_used == 1
        refused = await mt.skill_delete(_agent(env, batch=batch), second, reason="x")
        assert "at most 1" in refused
        names = await _skill_names(env)
        assert "ambiguous-del-a" not in names and "ambiguous-del-b" in names


# --------------------------------------------------------------------------- #
# Deletes                                                                      #
# --------------------------------------------------------------------------- #


class TestDeletes:
    async def test_one_delete_per_run_across_batches(self, env):
        setup = _agent(env)
        first = _id(await _create(setup, "del-a"))
        second = _id(await _create(setup, "del-b"))
        limits = CuratorRunLimits(run_id="run-del")
        b0 = _batch(1, run_limits=limits, batch_id="run-del:b0")
        b1 = _batch(1, run_limits=limits, batch_id="run-del:b1")
        ok = await mt.skill_delete(_agent(env, batch=b0), first, reason="wrong advice")
        assert "soft-deleted" in ok
        # Deleting does not use the edit budget.
        assert b0.spent == 0 and b0.remaining == 1
        refused = await mt.skill_delete(_agent(env, batch=b1), second, reason="dup")
        assert "at most 1" in refused and "del-b" in await _skill_names(env)
        assert b1.rejected_by_kind() == {"delete_limit": 1}
        assert limits.deletes_used == 1
        # The recorded delete carries its reason.
        async with env.sm() as db:
            version = (
                (
                    await db.execute(
                        select(SkillVersionORM)
                        .where(SkillVersionORM.skill_id == first)
                        .order_by(SkillVersionORM.version_no.desc())
                    )
                )
                .scalars()
                .first()
            )
        assert (version.change_type, version.reason, version.run_id) == (
            "delete",
            "wrong advice",
            "run-del:b0",
        )

    async def test_a_delete_needs_a_reason(self, env):
        skill_id = _id(await _create(_agent(env), "del-reason"))
        batch = _batch(1)
        for reason in ("", "   ", None):
            answer = await mt.skill_delete(
                _agent(env, batch=batch), skill_id, reason=reason
            )
            assert "reason" in answer and "soft-deleted" not in answer
        assert batch.run_limits.deletes_used == 0
        assert "del-reason" in await _skill_names(env)

    async def test_a_refused_delete_does_not_use_the_runs_delete(self, env):
        limits = CuratorRunLimits(run_id="run-d2")
        batch = _batch(1, run_limits=limits, batch_id="run-d2:b0")
        missing = await mt.skill_delete(
            _agent(env, batch=batch), "proc-nope", reason="x"
        )
        assert "not found" in missing and limits.deletes_left == 1

    async def test_outside_a_run_deletes_are_not_limited(self, env):
        agent = _agent(env)
        ids = [_id(await _create(agent, f"free-del-{i}")) for i in range(2)]
        for skill_id in ids:
            assert "soft-deleted" in await mt.skill_delete(
                agent, skill_id, reason="obsolete"
            )


# --------------------------------------------------------------------------- #
# Scope                                                                        #
# --------------------------------------------------------------------------- #


class TestScope:
    async def test_cross_scope_access_by_id_and_name_is_denied(self, env):
        foreign = _agent(env, scope="foreign-scope")
        foreign_id = _id(await _create(foreign, "foreign-skill"))
        batch = _batch(2)
        mine = _agent(env, batch=batch)
        assert "not found" in await mt.skill_read(mine, foreign_id)
        assert "not found" in await mt.skill_read(mine, "foreign-skill")
        assert "not found" in await mt.skill_edit(
            mine, foreign_id, "instructions", value="x"
        )
        assert "not found" in await mt.skill_delete(mine, foreign_id, reason="x")
        listing = await mt.skill_list(mine, query="foreign-skill")
        assert "No skills found" in listing
        assert batch.spent == 0 and batch.run_limits.deletes_used == 0
        # Untouched, and fully visible to its own scope.
        own = await mt.skill_read(foreign, foreign_id)
        assert "Steps for foreign-skill" in own

    async def test_the_manager_refuses_a_write_outside_the_given_scopes(self, env):
        from mirix.orm.errors import NoResultFound
        from mirix.services import skill_history as history

        foreign_id = _id(await _create(_agent(env, scope="scope-z"), "scope-z-skill"))
        with pytest.raises(NoResultFound):
            await env.mgr.apply_skill_change(
                history.CHANGE_EDIT,
                actor=env.actor,
                user_id=env.user.id,
                skill_id=foreign_id,
                fields={"instructions": "hijacked"},
                scopes=[SCOPE],
            )
        with pytest.raises(NoResultFound):
            await env.mgr.get_item_by_id(
                item_id=foreign_id,
                user=env.user,
                timezone_str="UTC",
                actor=env.actor,
                scopes=[SCOPE],
            )

    async def test_an_agent_without_a_scope_is_not_restricted(self, env):
        scoped_id = _id(await _create(_agent(env, scope="scope-q"), "scope-q-skill"))
        unscoped = _agent(env, scope=None)
        assert "scope-q-skill" in await mt.skill_read(unscoped, scoped_id)


# --------------------------------------------------------------------------- #
# skill_list paging and skill_read cap                                         #
# --------------------------------------------------------------------------- #


class TestListAndRead:
    async def test_more_than_fifty_skills_are_all_listed(self, env):
        scope = f"list-{uuid.uuid4().hex[:6]}"
        agent = _agent(env, scope=scope)
        names = [f"lst-{i:03d}" for i in range(120)]
        for name in names:
            await _create(agent, name)
        seen = []
        offset = 0
        pages = 0
        while True:
            page = await mt.skill_list(agent, offset=offset)
            pages += 1
            lines = [line for line in page.splitlines() if line.startswith("[ID: ")]
            seen += [line.split("] ", 1)[1].split(" (v", 1)[0] for line in lines]
            assert "of 120" in page.splitlines()[0]
            if "Next page:" not in page:
                assert page.rstrip().endswith("End of list.")
                break
            offset = int(page.split("offset=")[-1].split(")")[0])
        assert seen == sorted(names) and pages == 3
        first = await mt.skill_list(agent)
        assert (
            len([x for x in first.splitlines() if x.startswith("[ID: ")])
            == SKILL_LIST_PAGE_SIZE
        )

    async def test_a_query_filters_on_every_term(self, env):
        scope = f"q-{uuid.uuid4().hex[:6]}"
        agent = _agent(env, scope=scope)
        await _create(agent, "export-report", description="export the monthly report")
        await _create(agent, "export-table", description="export a table")
        await _create(
            agent,
            "import-data",
            description="load data from a file",
            triggers=["monthly import"],
        )
        page = await mt.skill_list(agent, query="export monthly")
        assert "export-report" in page and "export-table" not in page
        assert "of 1 matching" in page
        page = await mt.skill_list(agent, query="MONTHLY")
        assert "export-report" in page and "import-data" in page
        assert "No skills found" in await mt.skill_list(
            agent, query="nothing-like-this"
        )

    async def test_paging_edges(self, env):
        agent = _agent(env, scope=f"edge-{uuid.uuid4().hex[:6]}")
        await _create(agent, "only-one")
        assert "No skills at offset 5" in await mt.skill_list(agent, offset=5)
        batch = _batch(1)
        refused = await mt.skill_list(_agent(env, batch=batch), offset=-1)
        assert "offset" in refused and batch.rejected_by_kind() == {
            "invalid_arguments": 1
        }

    async def test_skill_read_is_capped_head_and_tail_and_counted(self, env):
        batch = _batch(1, cap=2_000)
        agent = _agent(env, batch=batch)
        instructions = "HEAD-MARK " + words_block(11_000) + " TAIL-MARK"
        skill_id = _id(
            await _create(_agent(env), "long-read", instructions=instructions)
        )
        text = await mt.skill_read(agent, skill_id)
        assert len(text) <= 2_000
        assert text.startswith(f"ID: {skill_id}") and "TAIL-MARK" in text
        assert "characters omitted" in text
        assert batch.outputs_capped == 1 and batch.output_chars_omitted > 9_000
        # Outside a run the cap follows the model's window (>= 60,000 chars).
        full = await mt.skill_read(_agent(env), "long-read")
        assert "HEAD-MARK" in full and "characters omitted" not in full

    def test_the_output_cap_formula(self):
        assert skill_output_char_cap(SimpleNamespace(context_window=8_192)) == 60_000
        # floor(1,050,000 * 0.8) * 4 * 0.1
        assert (
            skill_output_char_cap(SimpleNamespace(context_window=1_050_000)) == 336_000
        )
        text, omitted = head_tail_chars("a" * 50 + "b" * 50, 40)
        assert len(text) == 40 and text.startswith("a") and text.endswith("b")
        marker = f"\n…[{omitted} characters omitted]…\n"
        assert marker in text
        assert omitted == 100 - (len(text) - len(marker))
        assert head_tail_chars("short", 40) == ("short", 0)

    async def test_the_agent_does_not_cut_skill_tool_output_head_only(self):
        source = (REPO / "mirix" / "agent" / "agent.py").read_text()
        block = source.split('"archival_memory_search",', 1)[1].split("]:", 1)[0]
        assert '"skill_list"' in block and '"skill_read"' in block


def words_block(n: int) -> str:
    base = "alpha beta gamma delta epsilon "
    return (base * (n // len(base) + 1))[:n]


# --------------------------------------------------------------------------- #
# The write guard and the batch lineage                                        #
# --------------------------------------------------------------------------- #


class TestWriteGuard:
    async def _run(self, env):
        run = await env.runs.create_standalone_run(
            organization_id=env.org,
            user_id=env.user.id,
            agent_id=env.agent_id,
            mode="procedural",
        )
        assert await env.runs.mark_running(run.id)
        return run.id

    async def test_a_run_that_is_no_longer_active_cannot_write(self, env):
        run_id = await self._run(env)

        async def guard(db):
            await env.runs.lock_active_run(db, run_id)

        batch = _batch(3, batch_id=f"{run_id}:b0", guard=guard)
        agent = _agent(env, batch=batch)
        assert "created successfully" in await _create(agent, "guarded-ok")
        # The lease is lost (or the user erased, or the run cancelled).
        assert await env.runs.fail(run_id, "lease_expired")
        versions_before = await _count_versions(env)
        with pytest.raises(LearningRunNotActive):
            await _create(agent, "guarded-late")
        with pytest.raises(LearningRunNotActive):
            await mt.skill_edit(
                agent,
                _id(await _create(_agent(env), "guarded-target")),
                "description",
                value="a new description of it",
            )
        assert "guarded-late" not in await _skill_names(env)
        assert (
            await _count_versions(env) == versions_before + 1
        )  # only the setup create
        assert batch.spent == 1

    async def test_the_guard_error_ends_the_agent_step(self):
        from mirix.errors import AbortAgentStep

        assert issubclass(LearningRunNotActive, AbortAgentStep)
        source = (REPO / "mirix" / "agent" / "agent.py").read_text()
        # The tool executor and the response handler re-raise it.
        assert (
            source.count("except AbortAgentStep:")
            + source.count("except (AbortAgentStep, ToolCallRefused):")
            >= 3
        )

    async def test_every_write_stamps_the_batch_lineage(self, env):
        from mirix.orm.skill_experience import SkillExperience as ExperienceORM

        run_id = await self._run(env)
        mine = await env.exps.create_experience(
            agent_id=env.agent_id,
            user_id=env.user.id,
            organization_id=env.org,
            session_id="s-lineage",
            experience_type="worth_learning",
            title="lineage",
        )
        batch = _batch(2, batch_id=f"{run_id}:b0", ids=[mine.id, "sexp-of-nobody"])
        agent = _agent(env, batch=batch)
        created = _id(await _create(agent, "lineage-skill"))
        await mt.skill_edit(agent, created, "description", value="a better description")
        async with env.sm() as db:
            row = await db.get(ExperienceORM, mine.id)
        assert row.influenced_skill_ids == [created]
        assert row.status == "pending"  # consumed only when the batch settles


# --------------------------------------------------------------------------- #
# Retired machinery                                                            #
# --------------------------------------------------------------------------- #


class TestRetired:
    def test_the_old_budget_and_gates_are_gone(self):
        import mirix.constants as constants
        import mirix.services.skill_experience_curator as curator

        for name in (
            "compute_edit_budget",
            "_edit_exceeds_size_gate",
            "_record_authorizes_delete",
            "_consume_edit_budget",
        ):
            assert not hasattr(mt, name), name
        for name in (
            "SKILL_EVOLVE_MAX_CHAINING_STEPS",
            "SKILL_EDIT_BUDGET_B0",
            "SKILL_EDIT_BUDGET_ALPHA_FAIL",
            "SKILL_EDIT_BUDGET_ALPHA_SUCC",
            "SKILL_EDIT_BUDGET_MIN",
            "SKILL_EDIT_BUDGET_MAX",
            "SKILL_USE_AUTONOMOUS_BUDGET",
            "SKILL_MAX_EDIT_CHAR_DELTA",
            "SKILL_EDIT_MAJOR_RATIO",
            "SKILL_DELETE_BUDGET_MAX",
            "SKILL_EVOLVE_RECORDS_WINDOW",
        ):
            assert not hasattr(constants, name), name
        assert constants.SKILL_MAX_INSTRUCTIONS_CHARS == 12_000  # Kept
        assert constants.SKILL_CURATOR_MAX_STEPS == 100
        assert constants.SKILL_CURATOR_BATCH_FRACTION == 0.5
        assert not hasattr(curator, "_MAX_EXPERIENCES_PER_RUN")

    def test_no_curator_code_marks_experiences_superseded(self):
        tree = ast.parse(
            (REPO / "mirix" / "services" / "skill_experience_curator.py").read_text()
        )
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "mark_superseded" not in names

    def test_the_constants_parse_their_environment(self, monkeypatch):
        import mirix.constants as constants

        assert constants._fraction_setting("X", "0.25", 0.5) == 0.25
        for bad in ("0", "1.5", "-1", "abc"):
            assert constants._fraction_setting("X", bad, 0.5) == 0.5
        assert constants._fraction_setting("X", None, 0.5) == 0.5


# --------------------------------------------------------------------------- #
# The experience pool's tools: citations and triage                            #
# --------------------------------------------------------------------------- #


async def _experience(env, title, *, user_id=None, agent_id=None):
    return await env.exps.create_experience(
        agent_id=agent_id or env.agent_id,
        user_id=user_id or env.user.id,
        organization_id=env.org,
        session_id=f"s-{uuid.uuid4().hex[:6]}",
        experience_type="worth_learning",
        title=title,
    )


async def _experience_rows(env, ids):
    from mirix.orm.skill_experience import SkillExperience as ExperienceORM

    async with env.sm() as db:
        rows = (
            (
                await db.execute(
                    select(ExperienceORM).where(ExperienceORM.id.in_(list(ids)))
                )
            )
            .scalars()
            .all()
        )
        return {row.id: row for row in rows}


def _pool_batch(budget, ids, history=(), **kwargs):
    batch = _batch(budget, ids=ids, **kwargs)
    batch.history_ids = tuple(history)
    return batch


class TestCitations:
    async def test_each_write_stamps_only_the_ids_it_cites(self, env):
        e1, e2, e3 = [await _experience(env, f"cite {i}") for i in range(3)]
        lesson = await _experience(env, "an earlier lesson")  # a pool experience
        batch = _pool_batch(3, [e1.id, e2.id, e3.id], history=[lesson.id])
        agent = _agent(env, batch=batch)
        created = await _create(
            agent,
            "cited-skill",
            experience_ids=_cite(e1.id, lesson.id, quote="Steps for cited-skill"),
        )
        assert "created successfully" in created
        assert f"Cited: {e1.id}, {lesson.id}." in created
        skill_id = _id(created)
        edited = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="check it",
            new_text="verify it twice before moving on",
            value="",
            experience_ids=_cite(e2.id, quote="verify it twice"),
        )
        assert "updated" in edited and f"Cited: {e2.id}." in edited
        other = _id(await _create(_agent(env), "to-be-deleted"))
        deleted = await mt.skill_delete(
            agent, other, "merged elsewhere", experience_ids=[]
        )
        assert "soft-deleted" in deleted and "It cites no experience" in deleted
        rows = await _experience_rows(env, [e1.id, e2.id, e3.id, lesson.id])
        assert rows[e1.id].cited_skill_ids == [skill_id]
        assert rows[lesson.id].cited_skill_ids == [skill_id]
        assert rows[e2.id].cited_skill_ids == [skill_id]
        assert rows[e3.id].cited_skill_ids is None
        # The batch lineage still names every batch experience, never the lesson.
        for experience in (e1, e2, e3):
            assert rows[experience.id].influenced_skill_ids == [skill_id, other]
        assert rows[lesson.id].influenced_skill_ids is None
        assert batch.citations == {
            e1.id: [skill_id],
            lesson.id: [skill_id],
            e2.id: [skill_id],
        }
        # A soft delete costs no budget and is no write: an uncited one is
        # counted on its own, never as an uncited write.
        assert batch.uncited_writes == 0 and batch.uncited_deletes == 1
        assert batch.spent == 2
        assert batch.untriaged() == [e3.id]

    async def test_a_cited_id_the_triage_also_lists_is_counted_once(self, env):
        """A citation wins over the triage (the id counts as used), so
        the triage counts leave a cited id out instead of counting it twice."""
        e1, e2 = [await _experience(env, f"both {i}") for i in range(2)]
        batch = _pool_batch(2, [e1.id, e2.id])
        agent = _agent(env, batch=batch)
        await _create(
            agent,
            "cited-and-triaged",
            experience_ids=_cite(e1.id, quote="Steps for cited-and-triaged"),
        )
        answer = await mt.experience_triage(
            agent, covered_ids=[e1.id, e2.id], not_worth_ids=[], deferred_ids=[]
        )
        assert "count as used whatever list they are in" in answer
        assert batch.triage == {e1.id: "covered", e2.id: "covered"}
        assert batch.triage_counts() == {"covered": 1, "not_worth": 0, "deferred": 0}

    async def test_a_bad_citation_is_rejected_free_before_any_embedding(
        self, env, monkeypatch
    ):
        embedded = []

        class Counting:
            async def get_text_embedding(self, text):
                embedded.append(text)
                return [0.1] * 8

        async def counting_model(config, *args, **kwargs):
            return Counting()

        mine = await _experience(env, "mine")
        pooled_not_shown = await _experience(env, "in the pool but not shown")
        stranger = f"user-{uuid.uuid4().hex[:8]}"
        async with env.sm() as db:
            db.add(
                UserORM(
                    id=stranger,
                    name="s",
                    organization_id=env.org,
                    status="active",
                    timezone="UTC",
                )
            )
            await db.commit()
        foreign = await _experience(env, "someone else's", user_id=stranger)
        target = _id(await _create(_agent(env), "citation-target"))
        monkeypatch.setattr(pmm, "embedding_model", counting_model)
        batch = _pool_batch(2, [mine.id])
        agent = _agent(env, batch=batch)
        versions_before = await _count_versions(env)
        rejected = [
            await _create(agent, "bad-cite-a", experience_ids=["sexp-unknown"]),
            await _create(agent, "bad-cite-b", experience_ids=[foreign.id]),
            await _create(
                agent, "bad-cite-c", experience_ids=[mine.id, pooled_not_shown.id]
            ),
            await _create(agent, "bad-cite-d", experience_ids=[1, 2]),
            await _create(agent, "bad-cite-e", experience_ids={"id": mine.id}),
            await mt.skill_edit(
                agent,
                target,
                "description",
                old_text="",
                new_text="",
                value="new description",
                experience_ids=["sexp-unknown"],
            ),
            await mt.skill_delete(agent, target, "wrong", experience_ids=[foreign.id]),
        ]
        for message in rejected[:3] + rejected[5:]:
            assert "which this batch did not show" in message, message
            assert "cost no budget" in message
        assert (
            """Rejected: 'experience_ids' must be a list of objects with an "id" and a "quote"."""
            in rejected[3]
        )
        assert 'must be a list of objects with an "id" and a "quote"' in rejected[4]
        assert batch.rejected_by_kind() == {"invalid_arguments": len(rejected)}
        assert batch.spent == 0 and batch.run_limits.deletes_used == 0
        assert embedded == []
        assert await _count_versions(env) == versions_before
        rows = await _experience_rows(env, [mine.id, foreign.id, pooled_not_shown.id])
        assert all(row.cited_skill_ids is None for row in rows.values())

    async def test_a_json_list_string_a_bare_id_and_empty_fillers_are_accepted(
        self, env
    ):
        e1, e2 = [await _experience(env, f"loose {i}") for i in range(2)]
        batch = _pool_batch(3, [e1.id, e2.id])
        agent = _agent(env, batch=batch)
        as_json = await _create(
            agent,
            "loose-json",
            experience_ids=json.dumps(
                _cite(e1.id, quote="Steps for loose-json") + [{"id": "", "quote": ""}]
            ),
        )
        assert f"Cited: {e1.id}." in as_json
        # A bare id (the format before quotes) is parsed, but it has no quote.
        bare = await _create(agent, "loose-bare", experience_ids=f"  {e2.id} ")
        assert f"{e2.id} (no quote given)" in bare
        assert "NOT counted as applied" in bare
        filler = await _create(
            agent, "loose-filler", experience_ids=["", " ", {"id": " ", "quote": ""}]
        )
        assert "It cites no experience" in filler
        # The unquoted bare citation counts as an uncited write too.
        # The filler write cites nothing; the bare-id write's citation is
        # counted as unverified only (not also as an uncited write).
        assert batch.uncited_writes == 1 and batch.spent == 3
        assert batch.citations_unverified == 1
        assert batch.unverified_citations == {e2.id: [_id(bare)]}

    async def test_outside_a_batch_experience_ids_are_ignored(self, env):
        mine = await _experience(env, "in-flow")
        agent = _agent(env)
        noted = await _create(agent, "inflow-cited", experience_ids=[mine.id])
        assert "created successfully" in noted
        assert "(experience_ids are ignored outside a skill curation batch.)" in noted
        plain = await _create(agent, "inflow-plain", experience_ids=[])
        assert "ignored" not in plain and "cites no experience" not in plain
        malformed = await _create(agent, "inflow-bad", experience_ids=[1])
        assert "created successfully" in malformed
        rows = await _experience_rows(env, [mine.id])
        assert rows[mine.id].cited_skill_ids is None


class TestTriage:
    async def _triage(self, agent, covered=(), not_worth=(), deferred=()):
        return await mt.experience_triage(
            agent,
            covered_ids=list(covered),
            not_worth_ids=list(not_worth),
            deferred_ids=list(deferred),
        )

    async def test_it_records_and_a_later_call_overrides(self, env):
        batch = _pool_batch(3, ["e1", "e2", "e3"], history=["p1"])
        agent = _agent(env, batch=batch)
        first = await self._triage(
            agent, covered=["e1"], not_worth=["p1", ""], deferred=["e2"]
        )
        assert (
            "Triage recorded: covered 1 (e1); not worth a skill 1 (p1); deferred 1 (e2)."
            in first
        )
        assert "Neither cited nor triaged yet (they count as deferred): e3." in first
        assert batch.triage == {"e1": "covered", "p1": "not_worth", "e2": "deferred"}
        second = await self._triage(agent, not_worth=["e1"], deferred=["e3"])
        assert batch.triage == {
            "e1": "not_worth",
            "p1": "not_worth",
            "e2": "deferred",
            "e3": "deferred",
        }
        assert (
            "Every experience and earlier lesson of this batch is cited or triaged."
            in second
        )
        assert batch.triage_calls == 2
        assert batch.triage_counts() == {"covered": 0, "not_worth": 2, "deferred": 2}
        # All-empty lists record nothing and are no triage call.
        assert "Nothing recorded" in await self._triage(agent)
        assert batch.triage_calls == 2 and len(batch.triage) == 4

    async def test_bad_triage_is_rejected_free_and_records_nothing(self, env):
        batch = _pool_batch(1, ["e1", "e2"])
        agent = _agent(env, batch=batch)
        twice = await self._triage(agent, covered=["e1"], deferred=["e1"])
        assert "more than one list" in twice and "e1" in twice
        unknown = await self._triage(agent, covered=["e1", "sexp-nope"])
        assert "sexp-nope" in unknown and "was not shown in this batch" in unknown
        malformed = await mt.experience_triage(
            agent, covered_ids=[{"id": "e1"}], not_worth_ids=[], deferred_ids=[]
        )
        assert "Rejected: 'covered_ids' must be a list of experience ids." in malformed
        assert batch.triage == {} and batch.triage_calls == 0
        assert batch.rejected_by_kind() == {"invalid_arguments": 3}
        # Outside a batch there is nothing to triage.
        outside = await self._triage(_agent(env), covered=["e1"])
        assert "works only inside a skill curation batch" in outside

    async def test_cited_ids_count_as_used_and_the_untriaged_list_is_bounded(self, env):
        ids = [f"e{i:03d}" for i in range(60)]
        batch = _pool_batch(1, ids)
        batch.record_citations(["e000"], "proc-x")
        agent = _agent(env, batch=batch)
        answer = await self._triage(agent, covered=["e000"])
        assert "e000 are cited by a change of this batch" in answer
        assert "… and 9 more" in answer  # 59 untriaged, 50 named
        assert batch.untriaged() == ids[1:]

    async def test_an_earlier_attempts_citation_is_not_untriaged(self, env):
        batch = _pool_batch(1, ["e1", "e2"])
        batch.precited_ids = ("e1",)
        assert batch.untriaged() == ["e2"]
        assert batch.is_cited("e1") and not batch.is_cited("e2")


# --------------------------------------------------------------------------- #
# skill_edit with strict-mode fillers                                         #
# --------------------------------------------------------------------------- #


class TestStrictModeFillers:
    async def test_a_patch_with_an_empty_value_goes_through(self, env):
        agent = _agent(env, batch=_batch(2))
        skill = _id(await _create(agent, "filler-patch"))
        answer = await mt.skill_edit(
            agent,
            skill,
            "instructions",
            old_text="check it",
            new_text="verify it",
            value="",
        )
        assert "updated" in answer, answer
        rows = await _skill_rows(env, skill)
        assert "verify it" in rows.instructions

    async def test_a_rewrite_with_empty_old_and_new_text_goes_through(self, env):
        agent = _agent(env, batch=_batch(2))
        skill = _id(await _create(agent, "filler-rewrite"))
        answer = await mt.skill_edit(
            agent,
            skill,
            "instructions",
            old_text="",
            new_text="",
            value="A whole new text.",
        )
        assert "updated" in answer, answer
        assert (await _skill_rows(env, skill)).instructions == "A whole new text."

    async def test_an_empty_new_text_deletes_the_found_text(self, env):
        agent = _agent(env, batch=_batch(2))
        skill = _id(await _create(agent, "filler-delete"))
        answer = await mt.skill_edit(
            agent, skill, "instructions", old_text="check it, ", new_text="", value=""
        )
        assert "updated" in answer, answer
        instructions = (await _skill_rows(env, skill)).instructions
        assert "check it" not in instructions
        assert instructions.startswith("Steps for filler-delete: open it,")
        assert instructions.endswith("finish it.")

    async def test_both_or_neither_is_refused_free(self, env):
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        skill = _id(await _create(_agent(env), "filler-refused"))
        both = await mt.skill_edit(
            agent, skill, "instructions", old_text="check it", new_text="x", value="y"
        )
        assert (
            "Give either a patch (old_text and new_text, with value empty) or a whole new field (value, with old_text and new_text empty), not both."
            in both
        )
        neither = await mt.skill_edit(
            agent, skill, "instructions", old_text="", new_text="Y", value=""
        )
        assert (
            "give old_text (the text to find) and new_text, with value empty" in neither
        )
        assert batch.spent == 0 and batch.rejected_by_kind() == {"invalid_arguments": 2}

    async def test_json_and_entry_type_fields_ignore_the_text_fillers(self, env):
        agent = _agent(env, batch=_batch(3))
        skill = _id(await _create(agent, "filler-json"))
        triggers = await mt.skill_edit(
            agent, skill, "triggers", old_text="", new_text="", value='["on a form"]'
        )
        assert "updated" in triggers, triggers
        entry = await mt.skill_edit(
            agent, skill, "entry_type", old_text="x", new_text="y", value="guide"
        )
        assert "updated" in entry, entry
        missing = await mt.skill_edit(
            agent, skill, "examples", old_text="", new_text="", value=""
        )
        assert "value parameter is required" in missing
        row = await _skill_rows(env, skill)
        assert row.triggers == ["on a form"] and row.entry_type == "guide"


async def _skill_rows(env, skill_id):
    async with env.sm() as db:
        return await db.get(ProceduralMemoryItem, skill_id)


_ABSENT = object()


def _combination_args(old, new, value):
    args = {}
    for key, item in (("old_text", old), ("new_text", new), ("value", value)):
        if item is not _ABSENT:
            args[key] = item
    return args


_TEXT_COMBINATIONS = [
    (old, new, value)
    for old in (_ABSENT, None, "", "check it")
    for new in (_ABSENT, "", "verify it")
    for value in (_ABSENT, "", "A whole new text.")
]
_JSON_COMBINATIONS = [
    (old, new, value)
    for old in (_ABSENT, "", "x")
    for new in (_ABSENT, "", "y")
    for value in (_ABSENT, "", '["when a form fails"]')
]


class TestValidatorAgreesWithTheTool:
    """The agent's validator (a failed tool call) and the tool (a recorded
    rejection) give the same verdict on every combination of the three
    parameters, missing / empty / given."""

    async def _verdicts(self, env, field, old, new, value):
        from mirix.agent.tool_validators import validate_tool_args

        skill = _id(await _create(_agent(env), f"agree-{uuid.uuid4().hex[:10]}"))
        args = {"skill_id": skill, "field": field, **_combination_args(old, new, value)}
        validator_ok = validate_tool_args("skill_edit", dict(args)) is None
        answer = await mt.skill_edit(_agent(env, batch=_batch(1)), **args)
        return validator_ok, "updated" in answer, answer

    @pytest.mark.parametrize("old, new, value", _TEXT_COMBINATIONS)
    async def test_text_field(self, env, old, new, value):
        validator_ok, tool_ok, answer = await self._verdicts(
            env, "instructions", old, new, value
        )
        old_given = old not in (_ABSENT, None, "")
        value_given = value not in (_ABSENT, None, "")
        expected = (old_given and not value_given and new is not _ABSENT) or (
            value_given and not old_given
        )
        assert tool_ok == expected, answer
        assert validator_ok == tool_ok, answer

    @pytest.mark.parametrize("old, new, value", _JSON_COMBINATIONS)
    async def test_json_field(self, env, old, new, value):
        validator_ok, tool_ok, answer = await self._verdicts(
            env, "triggers", old, new, value
        )
        assert tool_ok == (value not in (_ABSENT, "")), answer
        assert validator_ok == tool_ok, answer


# --------------------------------------------------------------------------- #
# Tool schemas and wiring                                                      #
# --------------------------------------------------------------------------- #


class TestToolSchemas:
    def _schemas(self):
        from mirix.functions.functions import load_function_set

        return {
            name: spec["json_schema"] for name, spec in load_function_set(mt).items()
        }

    def test_the_strict_schemas_require_the_new_arrays(self):
        from mirix.llm_api.helpers import convert_to_structured_output

        schemas = self._schemas()
        triage = convert_to_structured_output(schemas["experience_triage"])
        assert triage["parameters"]["required"] == [
            "covered_ids",
            "not_worth_ids",
            "deferred_ids",
        ]
        for name in ("covered_ids", "not_worth_ids", "deferred_ids"):
            assert triage["parameters"]["properties"][name]["type"] == "array"
            assert triage["parameters"]["properties"][name]["items"] == {
                "type": "string"
            }
        edit = convert_to_structured_output(schemas["skill_edit"])
        assert edit["parameters"]["required"][-1] == "experience_ids"
        delete = convert_to_structured_output(schemas["skill_delete"])
        assert delete["parameters"]["required"] == [
            "skill_id",
            "reason",
            "experience_ids",
        ]

    def test_skill_create_is_strict_with_typed_examples_and_an_entry_type_enum(self):
        """An example is an object with an input and an output string, and
        entry_type is one of three values, so skill_create converts to a strict
        schema (it did not while examples were free-form objects). Strict mode
        makes every parameter required: the model sends [] for the lists it
        does not use, as for skill_edit."""
        from mirix.llm_api.helpers import convert_to_structured_output

        schema = self._schemas()["skill_create"]
        properties = schema["parameters"]["properties"]
        assert properties["examples"]["items"]["properties"].keys() == {
            "input",
            "output",
        }
        assert properties["examples"]["items"]["required"] == ["input", "output"]
        assert properties["entry_type"]["enum"] == ["workflow", "guide", "script"]
        strict = convert_to_structured_output(schema)
        assert strict["strict"] is True
        assert strict["parameters"]["required"] == [
            "name",
            "description",
            "instructions",
            "entry_type",
            "triggers",
            "examples",
            "experience_ids",
        ]
        item = strict["parameters"]["properties"]["examples"]["items"]
        assert item["additionalProperties"] is False
        assert item["required"] == ["input", "output"]
        assert strict["parameters"]["properties"]["entry_type"]["enum"] == [
            "workflow",
            "guide",
            "script",
        ]

    def test_no_skill_tool_schema_or_prompt_names_a_concrete_example_skill(self):
        """A weak model echoed the docstrings' example name
        ("deploy-production", with description "deploy" and instructions "do")
        into a real skill. The docs describe the format instead."""
        schemas = self._schemas()
        texts = [json.dumps(schemas[name]) for name in schemas if "skill" in name]
        texts.append(json.dumps(schemas["experience_triage"]))
        texts.append(
            (
                REPO
                / "mirix"
                / "prompts"
                / "system"
                / "base"
                / "procedural_memory_agent.txt"
            ).read_text()
        )
        for source in ("schemas/procedural_memory.py", "orm/procedural_memory.py"):
            texts.append((REPO / "mirix" / source).read_text())
        for text in texts:
            assert "deploy-production" not in text
            assert "e.g." not in text or "deploy" not in text

    def test_a_literal_of_strings_is_an_enum(self):
        from typing import Literal, Optional

        from mirix.functions.schema_generator import type_to_json_schema_type

        assert type_to_json_schema_type(Literal["a", "b"]) == {
            "type": "string",
            "enum": ["a", "b"],
        }
        assert type_to_json_schema_type(Optional[Literal["a"]]) == {
            "type": "string",
            "enum": ["a"],
        }
        with pytest.raises(ValueError):
            type_to_json_schema_type(Literal[1, 2])

    def test_examples_must_be_input_output_strings(self):
        problem = mt._validate_skill_fields(
            examples=[{"key": "value"}], check=("examples",)
        )
        assert problem and '"input" and "output"' in problem
        assert (
            mt._validate_skill_fields(
                examples=[{"input": "open file", "output": "file opened"}],
                check=("examples",),
            )
            is None
        )
        assert mt._validate_skill_fields(
            examples=[{"input": "open file", "output": {"a": 1}}], check=("examples",)
        )

    def test_finish_memory_update_is_unchanged(self):
        schema = self._schemas()["finish_memory_update"]
        assert schema["parameters"]["properties"] == {}

    def test_only_the_procedural_agent_gets_experience_triage(self):
        import inspect as _inspect

        from mirix import constants
        from mirix.services import agent_manager, tool_manager

        source = _inspect.getsource(agent_manager)
        assert (
            source.count("SKILL_TOOLS + SKILL_CURATION_TOOLS + UNIVERSAL_MEMORY_TOOLS")
            == 2
        )
        assert "SKILL_CURATION_TOOLS" not in constants.AUTO_DREAM_TOOLS
        assert "experience_triage" not in constants.AUTO_DREAM_TOOLS
        assert "SKILL_CURATION_TOOLS" in _inspect.getsource(tool_manager)

    async def test_a_non_strict_tool_is_noted_once_per_process(self, monkeypatch):
        import logging

        from mirix.llm_api import openai_client
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import Message
        from mirix.schemas.mirix_message_content import TextContent

        monkeypatch.setattr(openai_client, "_NON_STRICT_TOOLS_LOGGED", set())
        logged = []
        monkeypatch.setattr(
            openai_client.logger,
            "log",
            lambda level, msg, *args: logged.append((level, msg % args)),
        )
        warned = []
        monkeypatch.setattr(
            openai_client.logger, "warning", lambda *a, **k: warned.append(a)
        )
        schemas = self._schemas()
        # A tool with a free-form object parameter cannot be made strict.
        free_form = {
            "name": "free_form_tool",
            "description": "A tool whose argument is any object.",
            "parameters": {
                "type": "object",
                "properties": {"blob": {"type": "object", "description": "Anything."}},
                "required": ["blob"],
            },
        }
        tools = [
            schemas[n] for n in ("skill_create", "skill_edit", "experience_triage")
        ] + [free_form]
        client = openai_client.OpenAIClient(
            LLMConfig(
                model="fake-model",
                model_endpoint_type="openai",
                model_endpoint="http://127.0.0.1:9/v1",
                context_window=200_000,
                api_key="fake-key",
            )
        )
        messages = [
            Message(role=MessageRole.system, content=[TextContent(text="s")]),
            Message(role=MessageRole.user, content=[TextContent(text="u")]),
        ]
        for _ in range(3):
            data = await client.build_request_data(
                messages, client.llm_config, tools=tools
            )
        sent = {
            t["function"]["name"]: t["function"]["parameters"] for t in data["tools"]
        }
        # Converted: every parameter required, nothing else allowed.
        assert sent["skill_edit"]["additionalProperties"] is False
        assert "experience_ids" in sent["skill_edit"]["required"]
        assert sent["experience_triage"]["additionalProperties"] is False
        # skill_create converts too now: its lists are required.
        assert sent["skill_create"]["additionalProperties"] is False
        assert sent["skill_create"]["required"][-3:] == [
            "triggers",
            "examples",
            "experience_ids",
        ]
        # Sent as before, unconverted.
        assert sent["free_form_tool"]["required"] == ["blob"]
        assert warned == []
        assert [level for level, _ in logged] == [
            logging.INFO,
            logging.DEBUG,
            logging.DEBUG,
        ]
        assert all("free_form_tool" in text for _, text in logged)
        assert "cannot be made strict" in logged[0][1]
        assert "skill_create" not in openai_client._NON_STRICT_TOOLS_LOGGED


# --------------------------------------------------------------------------- #
# Content checks: a placeholder or a schema echo is rejected                   #
# --------------------------------------------------------------------------- #


class TestContentChecks:
    async def test_placeholders_and_schema_echoes_are_rejected_free(
        self, env, monkeypatch
    ):
        """Placeholder and schema-echo values a weak model may send, and
        their generic kin: each is a free invalid_arguments
        rejection, before any embedding, and writes nothing."""
        embedded = []

        class Counting:
            async def get_text_embedding(self, text):
                embedded.append(text)
                return [0.1] * 8

        async def counting_model(config, *args, **kwargs):
            return Counting()

        monkeypatch.setattr(pmm, "embedding_model", counting_model)
        e1 = await _experience(env, "placeholder source")
        batch = _pool_batch(3, [e1.id])
        agent = _agent(env, batch=batch)
        versions_before = await _count_versions(env)
        cases = {
            # The docstring's example name with one-word content.
            "example-name": dict(
                name="deploy-production", description="deploy", instructions="do"
            ),
            "dots": dict(description="..."),
            "text": dict(description="text"),
            "string": dict(description="string"),
            "one-word": dict(description="deploy"),
            "do": dict(instructions="do"),
            "instructions-text": dict(instructions="text"),
            "echo-instructions": dict(instructions="detailed plain text"),
            "echo-name": dict(name="kebab-case name"),
            "echo-description": dict(description="one/two sentences"),
            "stub": dict(
                instructions="This placeholder skill was created in error. Please delete it."
            ),
            "trigger-letter": dict(triggers=["a"]),
            "trigger-echo": dict(triggers=["when relevant"]),
            "example-shape": dict(examples=[{"key": "value"}]),
            "example-placeholder": dict(examples=[{"input": "...", "output": "..."}]),
        }
        answers = {}
        for label, fields in cases.items():
            answers[label] = await _create(
                agent,
                fields.pop("name", f"content-{label}"),
                experience_ids=[e1.id],
                **fields,
            )
        for label, answer in answers.items():
            assert (
                answer.startswith("Rejected: ") and "Nothing was created" in answer
            ), (
                label,
                answer,
            )
        assert "has 1 word(s) of content" in answers["example-name"]
        assert "repeats the words of the tool's description" in answers["echo-name"]
        assert "says it is a placeholder" in answers["stub"]
        assert 'exactly two strings, "input" and "output"' in answers["example-shape"]
        assert batch.rejected_by_kind() == {"invalid_arguments": len(cases)}
        assert batch.spent == 0 and batch.citations == {} and embedded == []
        assert await _count_versions(env) == versions_before
        rows = await _experience_rows(env, [e1.id])
        assert rows[e1.id].cited_skill_ids is None

    async def test_edits_to_placeholders_are_rejected_too(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "content-edit"))
        for field, value in (
            ("description", "..."),
            ("instructions", "detailed plain text"),
            ("name", "skill name"),
            ("triggers", '["a"]'),
            ("examples", '[{"key": "value"}]'),
        ):
            answer = await mt.skill_edit(agent, skill_id, field, value=value)
            assert "Nothing was changed" in answer or answer.startswith("Field "), (
                field,
                answer,
            )
            assert "updated" not in answer, (field, answer)
        patched = await mt.skill_edit(
            agent,
            skill_id,
            "description",
            old_text="what content-edit does",
            new_text="TBD",
        )
        assert "Nothing was changed" in patched

    async def test_real_short_content_passes(self, env):
        agent = _agent(env, batch=_batch(4))
        ok = [
            await _create(
                agent,
                "json-output",
                description="Retry failed uploads",
                instructions="Retry the upload twice.",
                triggers=["json output", "upload failed"],
                examples=[{"input": "open file", "output": "file opened"}],
            ),
            await _create(
                agent,
                "cjk-skill",
                description="部署到生产环境",
                instructions="先检查配置，再部署。",
                triggers=["部署时"],
            ),
            await _create(
                agent,
                "template-fill",
                description="Fill a template before sending it",
                instructions="Replace the placeholder text with the customer name.",
            ),
        ]
        assert all("created successfully" in a for a in ok), ok

    async def test_strict_mode_fillers_in_a_creates_lists_are_dropped(self, env):
        """skill_create is strict, so a model must send triggers and
        examples even when it has none; a blank entry ([""], an example of two
        empty strings) is that filler, like experience_ids' [""]: dropped, not
        stored and not a rejection. A blank entry next to real ones is dropped
        too; a real entry is still checked."""
        agent = _agent(env, batch=_batch(3))
        created = await _create(
            agent,
            "filler-lists",
            triggers=["", "  "],
            examples=[{"input": "", "output": " "}],
        )
        assert "created successfully" in created, created
        skill = await env.mgr.get_item_by_id(
            item_id=_id(created), user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.triggers == [] and skill.examples == []
        mixed = await _create(
            agent,
            "filler-mixed",
            triggers=["", "upload failed"],
            examples=[
                {"input": "", "output": ""},
                {"input": "open file", "output": "file opened"},
            ],
        )
        assert "created successfully" in mixed, mixed
        skill = await env.mgr.get_item_by_id(
            item_id=_id(mixed), user=env.user, timezone_str="UTC", actor=env.actor
        )
        assert skill.triggers == ["upload failed"]
        assert skill.examples == [{"input": "open file", "output": "file opened"}]
        # Half an example is no filler: it is checked (and rejected).
        half = await _create(
            agent, "filler-half", examples=[{"input": "open file", "output": ""}]
        )
        assert half.startswith("Rejected: ") and "Nothing was created" in half


class TestCitationsNeedAddedText:
    async def _setup(self, env, instructions):
        e1 = await _experience(env, "an edit's source")
        batch = _pool_batch(2, [e1.id])
        agent = _agent(env, batch=batch)
        skill_id = _id(
            await _create(
                _agent(env), f"cite-{uuid.uuid4().hex[:8]}", instructions=instructions
            )
        )
        return e1, batch, agent, skill_id

    async def _assert_not_recorded(self, env, e1, batch, answer):
        assert "updated" in answer and "were NOT recorded" in answer, answer
        assert e1.id not in batch.citations and batch.citations_dropped == 1
        assert batch.unaddressed_experiences() == [e1.id]
        rows = await _experience_rows(env, [e1.id])
        assert rows[e1.id].cited_skill_ids is None
        [logged] = batch.write_log
        assert logged["added_words"] == 0 and logged["citations_recorded"] is False

    async def test_a_deletion_only_patch_records_no_citation(self, env):
        e1, batch, agent, skill_id = await self._setup(
            env,
            "1. Read every edition.\n4. Merging across editions:\nPrefer the latest figure.",
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="4. Merging across editions:",
            new_text="",
            experience_ids=[e1.id],
        )
        await self._assert_not_recorded(env, e1, batch, answer)
        assert batch.spent == 1  # the edit itself went through and was charged

    async def test_a_renumbering_records_no_citation(self, env):
        e1, batch, agent, skill_id = await self._setup(
            env, "4. Open the file.\n5. Check the totals."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="5.",
            new_text="6.",
            experience_ids=[e1.id],
        )
        await self._assert_not_recorded(env, e1, batch, answer)

    async def test_a_whitespace_only_rewrite_records_no_citation(self, env):
        e1, batch, agent, skill_id = await self._setup(
            env, "Open the file. Check the totals."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            value="Open   the file.\n\nCheck the totals.",
            experience_ids=[e1.id],
        )
        await self._assert_not_recorded(env, e1, batch, answer)

    async def test_a_rename_records_no_citation_and_says_why(self, env):
        """A rename or an entry_type change carries no lesson: its citations
        are not recorded, and the answer says so for what it is (not "it only
        deletes, renumbers or reflows")."""
        e1, batch, agent, skill_id = await self._setup(
            env, "Open the file. Check the totals."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "name",
            value="check-report-totals",
            experience_ids=[e1.id],
        )
        assert "updated" in answer and "were NOT recorded" in answer, answer
        assert "a rename carries no lesson" in answer
        assert "renumbers" not in answer
        assert e1.id not in batch.citations and batch.citations_dropped == 1
        typed = await mt.skill_edit(
            agent, skill_id, "entry_type", value="guide", experience_ids=[e1.id]
        )
        assert "an entry_type change carries no lesson" in typed, typed
        assert batch.citations_dropped == 2

    async def test_a_patch_that_adds_a_word_is_cited(self, env):
        e1, batch, agent, skill_id = await self._setup(
            env, "Open the file. Trust the success message."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="Trust the success message.",
            new_text="Never trust the success message.",
            experience_ids=_cite(e1.id, quote="Never trust the success message"),
        )
        assert f"Cited: {e1.id}." in answer
        assert batch.citations == {e1.id: [skill_id]}
        assert batch.citations_dropped == 0 and batch.unaddressed_experiences() == []
        [logged] = batch.write_log
        assert logged["added_words"] == 1 and logged["citations_recorded"] is True
        rows = await _experience_rows(env, [e1.id])
        assert rows[e1.id].cited_skill_ids == [skill_id]

    async def test_a_numeric_patch_is_cited(self, env):
        """A changed number is a lesson ("3 times" -> "5 times"): its citations
        are recorded, and a dropped-citation answer never claims a numeric
        change only deleted, renumbered or reflowed."""
        e1, batch, agent, skill_id = await self._setup(
            env, "1. Retry a failed upload 3 times, waiting 30 seconds."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="3 times, waiting 30 seconds",
            new_text="5 times, waiting 120 seconds",
            experience_ids=_cite(e1.id, quote="5 times, waiting 120 seconds"),
        )
        assert f"Cited: {e1.id}." in answer and "NOT recorded" not in answer, answer
        assert batch.citations == {e1.id: [skill_id]} and batch.citations_dropped == 0
        [logged] = batch.write_log
        assert logged["added_words"] == 2 and logged["removed_words"] == 2
        rows = await _experience_rows(env, [e1.id])
        assert rows[e1.id].cited_skill_ids == [skill_id]

    async def test_the_dropped_citation_reason_names_what_was_measured(self, env):
        e1, batch, agent, skill_id = await self._setup(
            env, "Open the file. Check the totals. Stop."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text=" Stop.",
            new_text="",
            experience_ids=[e1.id],
        )
        await self._assert_not_recorded(env, e1, batch, answer)
        assert "this change adds no new words" in answer
        assert "it only deletes" not in answer

    async def test_an_edit_outside_a_batch_does_not_diff_the_text(
        self, env, monkeypatch
    ):
        """In-flow edits never read the word counts, so the (CPU) diff is not
        run on the event loop for them."""
        import mirix.services.skill_text_checks as checks

        skill_id = _id(await _create(_agent(env), f"inflow-{uuid.uuid4().hex[:6]}"))

        def _no_diff(*args, **kwargs):
            raise AssertionError("added_words ran outside a batch")

        monkeypatch.setattr(checks, "added_words", _no_diff)
        answer = await mt.skill_edit(
            _agent(env),
            skill_id,
            "instructions",
            old_text="check it",
            new_text="verify it",
        )
        assert "updated" in answer, answer


class TestEditFeedback:
    async def test_a_rewrite_names_the_sentences_it_dropped(self, env):
        """A whole-field rewrite reports every old sentence it no longer
        has, not even reworded (advice only: the rewrite goes through)."""
        agent = _agent(env, batch=_batch(3))
        old = (
            "Look in every drawer before you give up on an object. "
            "Do not treat a cup as a mug when the task names a mug. "
            "Trust the observed object state over an action's success message. "
            "Put the object down only after you checked the target receptacle. "
            "Heat food in the microwave rather than on the stove top. "
            "Report the final location of every object you moved."
        )
        skill_id = _id(await _create(agent, "rewrite-feedback", instructions=old))
        new = (
            "Look in every drawer before you give up on an object. "
            "Heat food in the microwave rather than on the stove top. "
            "Report the final location of each object that you moved. "
            "A clear successful action result is evidence that the action completed."
        )
        answer = await mt.skill_edit(agent, skill_id, "instructions", value=new)
        assert "updated" in answer
        assert "This rewrite dropped 3 of 6 sentences of the old text" in answer
        for dropped in (
            "Do not treat a cup as a mug",
            "Trust the observed object state",
            "Put the object down only after",
        ):
            assert dropped in answer, dropped
        assert "Look in every drawer" not in answer.split("dropped")[1]
        assert "Report the final location" not in answer.split("dropped")[1]
        # A rewrite that keeps everything says nothing about dropped text.
        same = await mt.skill_edit(
            agent, skill_id, "instructions", value=new + " Then finish."
        )
        assert "updated" in same and "dropped" not in same

    async def test_broken_numbering_is_reported(self, env):
        agent = _agent(env, batch=_batch(2))
        created = await _create(
            agent,
            "numbering-feedback",
            instructions="1. Open the report.\n2. Check the totals.\n3. Save it.",
        )
        assert "numbered steps" not in created
        skill_id = _id(created)
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="3. Save it.",
            new_text="3. Match the time window.\n3. Resolve the footnotes.\n4. Save it.",
        )
        assert "updated" in answer
        assert "Warning: the numbered steps are out of order (3 repeated)" in answer


class TestFinishGuard:
    async def test_an_empty_triage_is_no_triage(self, env):
        """Three empty lists used to count as a triage call."""
        batch = _pool_batch(1, ["e1", "e2"])
        agent = _agent(env, batch=batch)
        rejected = await mt.experience_triage(
            agent, covered_ids=[], not_worth_ids=[""], deferred_ids=[]
        )
        assert "all three lists are empty" in rejected and "e1, e2" in rejected
        assert batch.triage_calls == 0 and batch.rejected_by_kind() == {"no_change": 1}
        batch.record_citations(["e1"], "proc-x")
        await mt.experience_triage(
            agent, covered_ids=["e2"], not_worth_ids=[], deferred_ids=[]
        )
        assert batch.triage_calls == 1
        accepted = await mt.experience_triage(
            agent, covered_ids=[], not_worth_ids=[], deferred_ids=[]
        )
        assert accepted.startswith("Nothing recorded") and batch.triage_calls == 1
        assert batch.rejected_by_kind() == {"no_change": 1}

    async def test_finish_refuses_once_while_an_experience_is_unaddressed(self, env):
        from mirix.errors import ToolCallRefused

        batch = _pool_batch(1, ["e1", "e2"], history=["p1"])
        agent = _agent(env, batch=batch)
        with pytest.raises(ToolCallRefused) as refused:
            await mt.finish_memory_update(agent)
        assert "Not finished: 2 experience(s)" in refused.value.message
        assert "e1, e2" in refused.value.message and "p1" not in refused.value.message
        assert batch.finish_refusals == 1
        assert batch.rejected_by_kind() == {"unaddressed_experiences": 1}
        # The second call goes through (they default to deferred).
        assert await mt.finish_memory_update(agent) == ""

    async def test_unaddressed_earlier_lessons_do_not_block_the_finish(self, env):
        batch = _pool_batch(1, ["e1"], history=["p1", "p2"])
        batch.record_triage(["e1"], [], [])
        assert await mt.finish_memory_update(_agent(env, batch=batch)) == ""
        assert batch.finish_refusals == 0
        # An earlier attempt's citation addresses an experience too.
        batch = _pool_batch(1, ["e1"])
        batch.precited_ids = ("e1",)
        assert await mt.finish_memory_update(_agent(env, batch=batch)) == ""

    async def test_outside_a_batch_finish_is_a_no_op(self, env):
        assert await mt.finish_memory_update(_agent(env)) == ""


# --------------------------------------------------------------------------- #
# A skill's size target; what retrieval matches                               #
# --------------------------------------------------------------------------- #

_LONG_RULE = "Check the totals twice before you report them to anyone. "


class TestSkillSizeTarget:
    """A retrieved skill is injected whole, so the tools say when a
    skill's instructions pass their soft target (advice, never a rejection),
    and skill_read / skill_list show every skill's size."""

    @pytest.fixture(autouse=True)
    def _small_target(self, monkeypatch):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "SKILL_TARGET_INSTRUCTIONS_CHARS", 200)

    async def test_an_edit_over_the_target_goes_through_and_says_so(self, env):
        batch = _batch(2)
        agent = _agent(env, batch=batch)
        skill_id = _id(await _create(_agent(env), "size-edit"))
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="finish it.",
            new_text="finish it. " + _LONG_RULE * 4,
        )
        assert answer.startswith("Skill 'size-edit' updated (field: instructions")
        assert batch.spent == 1
        length = len(
            (
                await env.mgr.get_live_skill_by_name(user=env.user, name="size-edit")
            ).instructions
        )
        assert length > 200
        assert f"Its instructions are now {length} characters, over the " in answer
        assert "200-character target of a skill" in answer
        assert "condense them" in answer and "into its own skill" in answer
        assert answer.index("over the 200-character target") < answer.index(
            "Edit budget left"
        )

    async def test_an_edit_within_the_target_has_no_size_note(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(_agent(env), "size-small"))
        answer = await mt.skill_edit(
            agent, skill_id, "instructions", old_text="check it", new_text="verify it"
        )
        assert "updated" in answer and "character target" not in answer

    async def test_a_description_edit_has_no_size_note(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(
            await _create(
                _agent(env), "size-desc", instructions="Do it. " + _LONG_RULE * 4
            )
        )
        answer = await mt.skill_edit(
            agent, skill_id, "description", value="what size-desc does, in short"
        )
        assert "updated" in answer and "character target" not in answer

    async def test_a_create_over_the_target_says_so(self, env):
        instructions = "Open the item first. " + _LONG_RULE * 4
        answer = await _create(
            _agent(env, batch=_batch(1)), "size-create", instructions=instructions
        )
        assert "created successfully" in answer
        assert (
            f"Its instructions are now {len(instructions)} characters, over the "
            "200-character target" in answer
        )
        small = await _create(_agent(env, batch=_batch(1)), "size-create-small")
        assert "character target" not in small

    async def test_skill_read_shows_the_size_and_the_target(self, env):
        big = "Open the item first. " + _LONG_RULE * 4
        big_id = _id(await _create(_agent(env), "size-read-big", instructions=big))
        small_id = _id(await _create(_agent(env), "size-read-small"))
        text = await mt.skill_read(_agent(env), big_id)
        assert f"Instructions ({len(big)} characters; target 200):\n{big}" in text
        assert f"Its instructions are now {len(big)} characters" in text
        small = await mt.skill_read(_agent(env), small_id)
        assert "characters; target 200):" in small
        assert "Its instructions are now" not in small

    async def test_skill_list_marks_a_skill_over_the_target(self, env):
        scope = f"size-{uuid.uuid4().hex[:6]}"
        agent = _agent(env, scope=scope)
        big = "Open the item first. " + _LONG_RULE * 4
        await _create(agent, "size-list-big", instructions=big)
        await _create(agent, "size-list-small")
        lines = {
            line.split("] ", 1)[1].split(" (v", 1)[0]: line
            for line in (await mt.skill_list(agent)).splitlines()
            if line.startswith("[ID: ")
        }
        assert lines["size-list-big"].endswith(
            f" [instructions {len(big)} chars: over target]"
        )
        assert "over target" not in lines["size-list-small"]

    def test_the_target_setting(self):
        import mirix.constants as constants

        setting = constants._skill_target_instructions_chars
        assert setting(None, 12_000) == 6_000  # half the hard cap
        assert setting("4000", 12_000) == 4_000
        assert setting("50000", 12_000) == 12_000  # never above the cap
        for bad in ("0", "-5", "many"):
            assert setting(bad, 12_000) == 6_000


class TestRetrievalTextContract:
    """Retrieval matches a skill by its name, description and triggers
    only; the tools say so where an instructions change can leave them stale,
    and a follow-up change of them costs no edit budget."""

    RETRIEVAL_NOTE = (
        "Retrieval finds a skill only by its name, description and triggers"
    )

    async def test_an_instructions_edit_names_what_retrieval_matches(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(_agent(env), "rt-edit"))
        answer = await mt.skill_edit(
            agent, skill_id, "instructions", old_text="check it", new_text="verify it"
        )
        assert "updated" in answer and self.RETRIEVAL_NOTE in answer
        assert "update the description or triggers to match" in answer
        # In a batch: the follow-up costs nothing.
        assert "costs no edit budget" in answer
        outside = await mt.skill_edit(
            _agent(env),
            skill_id,
            "instructions",
            old_text="open it",
            new_text="open the file",
        )
        assert self.RETRIEVAL_NOTE in outside and "costs no edit budget" not in outside

    async def test_other_fields_and_rejections_have_no_retrieval_note(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(_agent(env), "rt-other"))
        described = await mt.skill_edit(
            agent, skill_id, "description", value="what rt-other does, in short"
        )
        assert "updated" in described and self.RETRIEVAL_NOTE not in described
        rejected = await mt.skill_edit(
            agent, skill_id, "instructions", old_text="not there", new_text="x"
        )
        assert "updated" not in rejected and self.RETRIEVAL_NOTE not in rejected

    async def test_skill_read_names_the_retrieval_fields(self, env):
        skill_id = _id(await _create(_agent(env), "rt-read"))
        text = await mt.skill_read(_agent(env), skill_id)
        assert "Retrieval matches only: name, description, triggers." in text

    async def test_a_follow_up_retrieval_text_edit_is_free_once_per_field(self, env):
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        skill_id = _id(await _create(_agent(env), "rt-free"))
        other_id = _id(await _create(_agent(env), "rt-free-other"))
        edited = await mt.skill_edit(
            agent, skill_id, "instructions", old_text="check it", new_text="verify it"
        )
        assert "updated" in edited and batch.remaining == 0
        # The retrieval text of that skill, once per field: no budget.
        described = await mt.skill_edit(
            agent, skill_id, "description", value="what rt-free does, now wider"
        )
        assert "updated" in described and "cost no edit budget" in described
        triggers = await mt.skill_edit(
            agent, skill_id, "triggers", value='["a file must be verified"]'
        )
        assert "updated" in triggers
        assert batch.spent == 1 and batch.retrieval_followups == 2
        # A second change of the same field, another field or another skill
        # is an ordinary edit again.
        for field, value, target in (
            ("description", "what rt-free does, wider still", skill_id),
            ("instructions", "Open it, verify it twice, finish it.", skill_id),
            ("description", "what rt-free-other does, wider", other_id),
            ("entry_type", "guide", skill_id),
        ):
            refused = await mt.skill_edit(agent, target, field, value=value)
            assert "budget exhausted" in refused.lower(), (field, refused)
        assert batch.spent == 1 and batch.retrieval_followups == 2

    async def test_a_free_follow_up_needs_no_citation_and_records_none(self, env):
        """The free follow-up was told to cite experiences and
        counted as an uncited write; when it did cite, it recorded them, so
        up to three content-carrying writes per budgeted edit marked
        experiences 'used' at no budget cost."""
        exp_a = await _experience(env, "lesson a")
        exp_b = await _experience(env, "lesson b")
        exp_c = await _experience(env, "lesson c")
        batch = _pool_batch(1, [exp_a.id, exp_b.id, exp_c.id])
        agent = _agent(env, batch=batch)
        skill_id = _id(await _create(_agent(env), f"rt-cite-{uuid.uuid4().hex[:6]}"))
        edited = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="check it",
            new_text="verify it twice before closing",
            experience_ids=_cite(exp_a.id, quote="verify it twice"),
        )
        assert f"Cited: {exp_a.id}." in edited and batch.remaining == 0
        uncited = await mt.skill_edit(
            agent, skill_id, "description", value="what it does, verified twice"
        )
        assert "cost no edit budget and needs no citation" in uncited
        assert "It cites no experience" not in uncited
        assert batch.uncited_writes == 0
        cited = await mt.skill_edit(
            agent,
            skill_id,
            "triggers",
            value='["a result must be verified"]',
            experience_ids=[exp_a.id, exp_c.id],
        )
        assert "updated" in cited, cited
        assert f"Its citations ({exp_c.id}) were NOT recorded" in cited
        assert exp_a.id not in cited.split("were NOT recorded")[0].split("(")[-1]
        assert batch.spent == 1 and batch.retrieval_followups == 2
        assert set(batch.citations) == {exp_a.id}
        assert batch.unaddressed_experiences() == [exp_b.id, exp_c.id]
        rows = await _experience_rows(env, [exp_a.id, exp_c.id])
        assert rows[exp_a.id].cited_skill_ids == [skill_id]
        assert rows[exp_c.id].cited_skill_ids is None
        assert batch.citations_dropped == 1

    async def test_an_earlier_attempts_instructions_edit_makes_the_follow_up_free(
        self, env
    ):
        """A returned batch is retried with a fresh state; the
        instructions edit its earlier attempt made (a curator version) makes
        the retrieval-text follow-up free there too, once per field."""
        name = f"rt-prior-{uuid.uuid4().hex[:6]}"
        skill_id = _id(await _create(_agent(env), name))
        described = _id(await _create(_agent(env), f"{name}-desc"))
        # Attempt 1 (a curator batch) changes the instructions of the first
        # and only the description of the second.
        first = _agent(env, batch=_batch(2, batch_id="run-old:b0"))
        await mt.skill_edit(
            first, skill_id, "instructions", old_text="check it", new_text="verify it"
        )
        await mt.skill_edit(
            first, described, "description", value="what it does, now wider"
        )
        # An in-flow edit (no batch) is no curator change.
        await mt.skill_edit(
            _agent(env),
            described,
            "instructions",
            old_text="check it",
            new_text="test it",
        )
        labels = await env.mgr.get_skill_labels(
            user=env.user, skill_ids=[skill_id, described]
        )
        assert labels[skill_id]["instructions_changed_at"] is not None
        # Created in-flow, then only its description changed by a curator.
        assert labels[described]["instructions_changed_at"] is None
        from mirix.services.skill_curator_state import CuratorBatchState

        # Attempt 2: a fresh state that knows the earlier instructions edit.
        batch = _batch(1, batch_id="run-new:b0")
        batch.prior_instructions_edits = (skill_id,)
        agent = _agent(env, batch=batch)
        created = await _create(agent, f"{name}-spend")
        assert "created successfully" in created and batch.remaining == 0
        free = await mt.skill_edit(
            agent, skill_id, "description", value="what it does, verified"
        )
        assert "cost no edit budget" in free and batch.spent == 1
        again = await mt.skill_edit(
            agent, skill_id, "description", value="what it does, again"
        )
        assert "budget exhausted" in again.lower()
        assert isinstance(batch, CuratorBatchState)

    async def test_skill_labels_name_only_curator_instructions_changes(self, env):
        from mirix.services import skill_history as history

        skill_id = _id(await _create(_agent(env), f"rt-labels-{uuid.uuid4().hex[:6]}"))
        labels = await env.mgr.get_skill_labels(user=env.user, skill_ids=[skill_id])
        assert labels[skill_id]["instructions_changed_at"] is None  # in-flow create
        agent = _agent(env, batch=_batch(3, batch_id="run-lab:b0"))
        await mt.skill_edit(agent, skill_id, "description", value="what it does, wider")
        await mt.skill_edit(
            agent, skill_id, "triggers", value='["a file must be checked"]'
        )
        labels = await env.mgr.get_skill_labels(user=env.user, skill_ids=[skill_id])
        assert labels[skill_id]["instructions_changed_at"] is None
        await mt.skill_edit(
            agent, skill_id, "instructions", old_text="check it", new_text="verify it"
        )
        labels = await env.mgr.get_skill_labels(user=env.user, skill_ids=[skill_id])
        assert labels[skill_id]["instructions_changed_at"] is not None
        assert history.CHANGE_EDIT == "edit"

    async def test_without_an_instructions_edit_metadata_costs_budget(self, env):
        batch = _batch(1)
        agent = _agent(env, batch=batch)
        created = await _create(agent, "rt-created")  # a create is no follow-up
        assert batch.remaining == 0
        refused = await mt.skill_edit(
            agent, _id(created), "description", value="what rt-created does, wider"
        )
        assert "budget exhausted" in refused.lower()
        assert batch.retrieval_followups == 0


async def test_a_rejected_call_is_counted_on_the_agent(env):
    """The agent holds an escalated finish while a call of the same turn was
    rejected (tool_call_plan.held_terminal_text): every rejection counts on
    the agent, in a batch or not; a write that goes through does not."""
    agent = _agent(env, batch=_batch(2))
    rejected = await _create(agent, "x", description="...")
    assert rejected.startswith("Rejected: ") and agent.declined_tool_calls == 1
    missing = await mt.skill_edit(_agent(env), "proc_missing", "name", value="y-z")
    assert "not found" in missing
    created = await _create(agent, f"counted-{uuid.uuid4().hex[:6]}")
    assert "created successfully" in created and agent.declined_tool_calls == 1


def test_the_triage_tool_states_the_retirement_rule_the_pool_applies():
    """The deferred_ids parameter description matches the code."""
    doc = mt.experience_triage.__doc__
    assert "ignored in three batches without coming up again leaves the pool" in doc


# --------------------------------------------------------------------------- #
# A citation names the text that applies it; a patch of the whole field is    #
# a rewrite; the finish refusal names no bypass                               #
# --------------------------------------------------------------------------- #


class TestCitationQuotes:
    """An experience a write cites counts as used only when the write's
    quote for it is text the write added; the write itself always goes
    through."""

    async def _setup(self, env, *, budget=3, n=2, instructions=None):
        exps = [await _experience(env, f"quoted {i}") for i in range(n)]
        batch = _pool_batch(budget, [e.id for e in exps])
        agent = _agent(env, batch=batch)
        skill_id = None
        if instructions is not None:
            skill_id = _id(
                await _create(
                    _agent(env),
                    f"quoted-{uuid.uuid4().hex[:8]}",
                    instructions=instructions,
                )
            )
        return exps, batch, agent, skill_id

    async def test_a_create_counts_only_the_citations_its_text_shows(self, env):
        (e1, e2), batch, agent, _ = await self._setup(env)
        answer = await _create(
            agent,
            "quoted-create",
            instructions="Open the box. Weigh every parcel before labelling it.",
            experience_ids=[
                {"id": e1.id, "quote": "weigh every parcel before labelling"},
                {"id": e2.id, "quote": "ship it overnight to the customer"},
            ],
        )
        skill_id = _id(answer)
        assert "created successfully" in answer  # the write is never rejected
        assert f"Cited: {e1.id}." in answer
        assert "NOT counted as applied" in answer
        assert f"{e2.id} (quote not found in the text this change adds)" in answer
        assert batch.citations == {e1.id: [skill_id]}
        assert batch.unverified_citations == {e2.id: [skill_id]}
        assert batch.citations_unverified == 1 and batch.spent == 1
        assert batch.unaddressed_experiences() == [e2.id]
        rows = await _experience_rows(env, [e1.id, e2.id])
        assert rows[e1.id].cited_skill_ids == [skill_id]
        assert rows[e2.id].cited_skill_ids is None  # never stamped "used"
        [logged] = batch.write_log
        assert logged["citations_unverified"] == {
            e2.id: "quote not found in the text this change adds"
        }

    async def test_a_quote_of_text_the_skill_already_had_does_not_count(self, env):
        (e1, e2), batch, agent, skill_id = await self._setup(
            env, instructions="Open the box. Check the label twice before sealing."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="Open the box.",
            new_text="Open the box on a flat table and keep the lid upright.",
            value="",
            experience_ids=[
                {"id": e1.id, "quote": "keep the lid upright"},
                {"id": e2.id, "quote": "Check the label twice"},
            ],
        )
        assert "updated" in answer and f"Cited: {e1.id}." in answer
        assert f"{e2.id} (quote not found in the text this change adds)" in answer
        assert batch.citations == {e1.id: [skill_id]}
        # A whole new value: a quote of a sentence it kept is text the skill had.
        rewrite = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            value=(
                "Open the box on a flat table and keep the lid upright. "
                "Photograph the contents before you repack them."
            ),
            experience_ids=[
                {"id": e2.id, "quote": "keep the lid upright"},
            ],
        )
        assert f"{e2.id} (quote is text the skill already had)" in rewrite
        fixed = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="before you repack them.",
            new_text="before you repack them, and log the parcel weight.",
            experience_ids=[{"id": e2.id, "quote": "log the parcel weight"}],
        )
        assert f"Cited: {e2.id}." in fixed
        assert batch.unaddressed_experiences() == []

    async def test_a_quote_of_a_new_skills_name_shows_nothing(self, env):
        (e1,), batch, agent, _ = await self._setup(env, n=1)
        answer = await _create(
            agent,
            "weigh-every-parcel-first",
            description="How to send a parcel.",
            instructions="Open the box. Seal it again.",
            experience_ids=[{"id": e1.id, "quote": "weigh-every-parcel-first"}],
        )
        assert f"{e1.id} (quote not found in the text this change adds)" in answer
        assert batch.citations == {} and batch.unaddressed_experiences() == [e1.id]

    async def test_a_patch_quote_of_text_the_field_already_had_shows_nothing(self, env):
        (e1, e2), batch, agent, skill_id = await self._setup(
            env, instructions="Open the box. Check the label twice before sealing."
        )
        # The patch repeats a sentence the field already had elsewhere (and
        # adds new words too): a quote of the repeated sentence shows nothing.
        moved = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="Open the box.",
            new_text=(
                "Check the label twice before sealing. Open the box on a flat "
                "table with gloves."
            ),
            value="",
            experience_ids=[
                {"id": e1.id, "quote": "Check the label twice before sealing"},
                {"id": e2.id, "quote": "on a flat table with gloves"},
            ],
        )
        assert f"{e1.id} (quote is text the skill already had)" in moved, moved
        assert f"Cited: {e2.id}." in moved
        assert batch.citations == {e2.id: [skill_id]}

    async def test_the_answer_says_the_quote_minimum_and_that_they_stay_pending(
        self, env
    ):
        (e1,), batch, agent, _ = await self._setup(env, n=1)
        answer = await _create(
            agent,
            "quoted-short-two",
            instructions="Check the lid. Seal it.",
            experience_ids=[{"id": e1.id, "quote": "check the lid"}],
        )
        assert f"{e1.id} (quote too short to show where)" in answer
        assert "They stay pending" in answer and "at least 12 letters" in answer

    async def test_a_list_field_quote_comes_from_its_new_entries(self, env):
        (e1,), batch, agent, skill_id = await self._setup(
            env, n=1, instructions="Open the box. Check the label twice."
        )
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "triggers",
            value='["a parcel arrives without a label"]',
            experience_ids=[{"id": e1.id, "quote": "arrives without a label"}],
        )
        assert f"Cited: {e1.id}." in answer, answer

    async def test_a_short_or_missing_quote_does_not_count(self, env):
        (e1, e2), batch, agent, _ = await self._setup(env)
        answer = await _create(
            agent,
            "quoted-short",
            instructions="Open the box. Weigh every parcel before labelling it.",
            experience_ids=[
                {"id": e1.id, "quote": "the box"},
                {"id": e2.id, "quote": ""},
            ],
        )
        assert f"{e1.id} (quote too short to show where)" in answer
        assert f"{e2.id} (no quote given)" in answer
        # Counted once, as unverified citations, not also as uncited.
        assert batch.citations == {} and batch.uncited_writes == 0
        assert batch.citations_unverified == 2
        assert "It cites no experience" not in answer

    async def test_an_elided_quote_counts_when_every_part_is_there(self, env):
        (e1,), batch, agent, _ = await self._setup(env, n=1)
        answer = await _create(
            agent,
            "quoted-elided",
            instructions="Weigh every parcel. Then, before labelling, record the weight.",
            experience_ids=[
                {"id": e1.id, "quote": "Weigh every parcel ... record the weight"}
            ],
        )
        assert f"Cited: {e1.id}." in answer, answer

    async def test_the_finish_refusal_names_an_unquoted_citation(self, env):
        from mirix.errors import ToolCallRefused

        (e1,), batch, agent, _ = await self._setup(env, n=1)
        await _create(agent, "quoted-refusal", experience_ids=[e1.id])
        with pytest.raises(ToolCallRefused) as refused:
            await mt.finish_memory_update(agent)
        message = refused.value.message
        assert f"neither cited by a change nor triaged: {e1.id}" in message
        assert f"so not counted: {e1.id}" in message
        assert "with a quote of the words that apply it" in message

    async def test_a_deletion_needs_no_quote(self, env):
        (e1,), batch, agent, skill_id = await self._setup(
            env, n=1, instructions="Open the box. Check the label twice."
        )
        answer = await mt.skill_delete(
            agent, skill_id, "it is wrong", experience_ids=[{"id": e1.id, "quote": ""}]
        )
        assert "soft-deleted" in answer and f"Cited: {e1.id}." in answer
        assert batch.citations == {e1.id: [skill_id]}

    def test_the_citation_items_are_strict_objects(self):
        from mirix.functions.functions import load_function_set
        from mirix.llm_api.helpers import convert_to_structured_output

        schemas = {
            name: spec["json_schema"] for name, spec in load_function_set(mt).items()
        }
        for tool in ("skill_create", "skill_edit", "skill_delete"):
            strict = convert_to_structured_output(schemas[tool])
            items = strict["parameters"]["properties"]["experience_ids"]["items"]
            assert items["type"] == "object", tool
            assert items["required"] == ["id", "quote"], tool
            assert items["additionalProperties"] is False, tool


class TestPatchRewriteReport:
    """A patch whose old_text is (most of) the whole field reports the
    sentences it dropped like a rewrite with value, also when the model sent
    value="" (normalised to None)."""

    _OLD = (
        "Look in every drawer before you give up on an object. "
        "Trust the observed object state over an action's success message. "
        "Report the final location of every object you moved."
    )

    async def test_a_whole_field_patch_reports_what_it_dropped(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "patch-rewrite", instructions=self._OLD))
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text=self._OLD,
            new_text="Look in every drawer before you give up on an object.",
            value="",
        )
        assert "updated" in answer
        assert "This rewrite dropped 2 of 3 sentences of the old text" in answer
        assert "Trust the observed object state" in answer

    async def test_a_patch_of_most_of_the_field_reports_too(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "patch-most", instructions=self._OLD))
        most = self._OLD.split(". ", 1)[1]  # all but the first sentence
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text=most,
            new_text="Report where each object ended up.",
            value="",
        )
        assert "This rewrite dropped" in answer, answer

    async def test_a_small_patch_does_not(self, env):
        agent = _agent(env, batch=_batch(2))
        skill_id = _id(await _create(agent, "patch-small", instructions=self._OLD))
        answer = await mt.skill_edit(
            agent,
            skill_id,
            "instructions",
            old_text="Report the final location of every object you moved.",
            new_text="Report where each object ended up.",
            value="",
        )
        assert "updated" in answer and "This rewrite dropped" not in answer

    def test_the_share_rule(self):
        assert mt._patch_rewrites("a b c d", "a b c d")
        assert not mt._patch_rewrites("aaaa bbbb", "aaaa")  # 4 of 9 chars
        assert mt._patch_rewrites("aaaa bbbb", "aaaa b")  # 6 of 9 chars
        assert not mt._patch_rewrites("aaaa bbbb cccc", "aaaa")
        assert not mt._patch_rewrites("anything", None)


class TestFinishRefusalNamesNoBypass:
    """The refusal, the tool's own description and the batch header say
    what to do, never that another finish call ends the batch anyway."""

    async def test_the_refusal_text(self, env):
        from mirix.errors import ToolCallRefused

        batch = _pool_batch(1, ["e1"])
        with pytest.raises(ToolCallRefused) as refused:
            await mt.finish_memory_update(_agent(env, batch=batch))
        message = refused.value.message
        for bypass in ("again", "ends the batch", "leaves them deferred", "once"):
            assert bypass not in message, bypass

    def test_the_tool_descriptions_and_the_prompt(self):
        from mirix.functions.functions import load_function_set

        schemas = {
            name: spec["json_schema"] for name, spec in load_function_set(mt).items()
        }
        for tool in ("finish_memory_update", "experience_triage"):
            text = json.dumps(schemas[tool])
            assert "refuses once" not in text and "next call ends" not in text, tool
        prompt = (
            REPO
            / "mirix"
            / "prompts"
            / "system"
            / "base"
            / "procedural_memory_agent.txt"
        ).read_text()
        assert "calling it again ends the batch" not in prompt
        assert "the first `finish_memory_update()` does not end" not in prompt
