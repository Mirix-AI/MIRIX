"""Postgres integration: the curator stage of a learning run.

Run through the fresh-database runner (MIRIX_PG_URI is read at import time):

    scripts/run_pg_integration_tests.sh tests/test_curator_pg_integration.py

A real meta agent with its procedural child on Postgres (real tools, real
system prompt, the production OpenAI-compatible client), every model call
answered by the offline fake model server with scripted tool calls — no paid
call, nothing leaves localhost. Covered on the real row locks, JSON columns
and filter-tag SQL:

* a two-batch run through ``Agent.step``: the batches run in order, the second
  sees (reads, rewrites, deletes with a reason) the skill the first created;
  every experience is consumed by its own batch id, every skill version names
  its batch; the run's metrics and usage are recorded; the model
  override reaches every call and is not persisted;
* the write guard really waits on the run's row lock (``FOR UPDATE``): a skill
  write while another transaction holds the run waits, and when that
  transaction fails the run the write is refused and nothing is written;
* a lost lease mid-batch stops every write; the batch entry stays ``running``;
* an unfinished batch returned three times is closed out (unaddressed);
* scope: another scope's skill is not listed, read, edited or deleted;
  the ``skill_list`` query and paging run on Postgres SQL.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import select, update

from curator_harness import (
    add_experience,
    experiences,
    make_world,
    procedural_agent,
    skills,
    start_run,
    tool_results,
    user_texts,
    versions,
    words,
)
from db_isolation import block_non_loopback_network, isolated_database
from fakes.fake_llm_server import ChatResponse, FakeLLMServer
from mirix.functions.function_sets import memory_tools as mt
from mirix.schemas.usage import MirixUsageStatistics
from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def offline():
    with pytest.MonkeyPatch.context() as monkeypatch:
        block_non_loopback_network(monkeypatch)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        yield


@pytest.fixture(scope="module")
def fake(offline):
    with FakeLLMServer() as server:
        yield server


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def database(tmp_path_factory, offline):
    from mirix import constants
    from mirix.server import server
    from mirix.services import context_budget

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "images_dir", tmp_path_factory.mktemp("images"))
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        context_budget._reset_tokenizer_state_for_tests()
        context_budget._tokenizer_state.failed_at = time.monotonic() + 10**9
        await server.engine.dispose(close=False)
        async with isolated_database(tmp_path_factory.mktemp("curator-pg")) as engine:
            yield engine
        await server.engine.dispose()
        context_budget._reset_tokenizer_state_for_tests()


@pytest.fixture(autouse=True)
def _fresh_fake(fake):
    fake.reset()
    fake.set_default_responder(None)
    yield
    fake.reset()


def _call(name, args=None):
    return ChatResponse(tool_calls=[{"name": name, "arguments": args or {}}])


def _id(answer: str) -> str:
    return answer.split("ID: ")[1].split(",")[0].split(")")[0]


async def _curate(world, run_id, **kwargs):
    from mirix.services.skill_experience_curator import curate_pending_experiences

    return await curate_pending_experiences(
        run_id=run_id,
        user=world.user,
        actor=world.actor,
        meta_agent_state=world.meta,
        **kwargs,
    )


async def _run_record(run_id):
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    return await SkillLearningRunManager().get_run(run_id)


def _big(i):
    # ~20,000 characters: two never share a batch of the 32k window.
    return f"LESSON-{i} " + words(20_000, seed=i)


async def test_a_two_batch_run_through_agent_step(fake):
    from mirix.services.skill_experience_curator import run_curator_batch_step

    world = await make_world(fake)
    exps = [
        await add_experience(world, title=f"pg {i}", content=_big(i)) for i in range(2)
    ]
    run_id = await start_run(world)
    fake.queue_response(
        _call("skill_list"),
        _call(
            "skill_create",
            {
                "name": "pg-skill",
                # Real content (placeholders are rejected), and the batch's
                # experience cited before the finish (an unaddressed one would
                # make the first finish list it instead of ending the batch).
                "description": "The first version of the checking skill",
                "instructions": "Open it. Check it.",
                "entry_type": "workflow",
                "triggers": ["when checking"],
                "experience_ids": [{"id": exps[0].id, "quote": "Open it. Check it."}],
            },
        ),
        _call("finish_memory_update"),
        _call("skill_list", {"query": "checking"}),
        _call("skill_read", {"name_or_id": "pg-skill"}),
    )
    created = {}

    def batch_1_writes(body):
        step = created["step"] = created.get("step", 0) + 1
        if step == 1:
            return _call(
                "skill_edit",
                {
                    "skill_id": created["id"],
                    "field": "instructions",
                    "value": "A full rewrite.",
                    "experience_ids": [{"id": exps[1].id, "quote": "A full rewrite."}],
                },
            )
        if step == 2:
            return _call(
                "skill_delete",
                {"skill_id": created["id"], "reason": "merged elsewhere"},
            )
        if step == 3:
            return _call("finish_memory_update")
        raise AssertionError("unexpected extra request")

    async def run_batch(agent, payload, *, actor, user):
        if agent.skill_run_id == f"{run_id}:b1":
            created["id"] = (await skills(world))["pg-skill"].id
            fake.set_default_responder(batch_1_writes)
        return await run_curator_batch_step(agent, payload, actor=actor, user=user)

    outcome = await _curate(
        world, run_id, run_batch=run_batch, model_override="override-model"
    )

    assert fake.unscripted_requests == 0 and fake.pending_responses == 0
    requests = fake.chat_requests()
    assert len(requests) == 8
    assert {r.json["model"] for r in requests} == {"override-model"}
    assert "LESSON-0 " in user_texts(requests[0].json)[-1]
    assert "LESSON-1 " in user_texts(requests[3].json)[-1]
    assert "LESSON-0 " not in "".join(user_texts(requests[3].json))
    # The query (a JSON trigger match, Postgres SQL) found batch 0's skill.
    assert "pg-skill" in tool_results(requests[4].json)[-1]
    assert [b["stop_reason"] for b in outcome.batches] == ["finish", "finish"]
    assert [b["steps"] for b in outcome.batches] == [3, 5]
    assert [b["writes"] for b in outcome.batches] == [1, 1]
    assert [b["deletes"] for b in outcome.batches] == [0, 1]

    rows = await experiences(world)
    assert [rows[e.id].consumed_by for e in exps] == [f"{run_id}:b0", f"{run_id}:b1"]
    skill_id = created["id"]
    # Lineage (a JSON column) stamped in each write's transaction.
    assert rows[exps[0].id].influenced_skill_ids == [skill_id]
    assert rows[exps[1].id].influenced_skill_ids == [skill_id]
    assert [(v[1], v[3], v[4]) for v in await versions(skill_id)] == [
        ("create", f"{run_id}:b0", None),
        ("edit", f"{run_id}:b1", None),
        ("delete", f"{run_id}:b1", "merged elsewhere"),
    ]
    assert outcome.changes == {
        "created": [skill_id],
        "edited": [skill_id],
        "deleted": [skill_id],
    }
    run = await _run_record(run_id)
    assert [(e["batch_id"], e["status"]) for e in run.curator_metrics] == [
        (f"{run_id}:b0", "settled"),
        (f"{run_id}:b1", "settled"),
    ]
    assert run.llm_usage["by_stage"]["curate"]["calls"] == 8
    stored = await world.server.agent_manager.get_agent_by_id(
        agent_id=world.procedural.id, actor=world.actor
    )
    assert stored.llm_config.model == "fake-model"


async def test_the_write_guard_waits_on_the_run_lock_then_refuses(fake):
    from mirix.orm.skill_learning_run import SkillLearningRun as RunORM
    from mirix.server.server import db_context
    from mirix.services.skill_learning_run_manager import LearningRunNotActive

    world = await make_world(fake)
    exp = await add_experience(world, title="locked")
    run_id = await start_run(world)
    observed = {}

    async def runner(agent, payload, *, actor, user):
        async with db_context() as holder:
            await holder.execute(
                select(RunORM.id).where(RunORM.id == run_id).with_for_update()
            )
            write = asyncio.create_task(
                mt.skill_create(
                    agent,
                    name="raced",
                    description="Opens an item and checks it",
                    instructions="Open the item, check it, then close it.",
                    entry_type="guide",
                )
            )
            await asyncio.sleep(1.0)
            observed["waiting"] = not write.done()
            await holder.execute(
                update(RunORM).where(RunORM.id == run_id).values(status="failed")
            )
            await holder.commit()
        await write
        raise AssertionError("the write should have been refused")

    with pytest.raises(LearningRunNotActive):
        await _curate(world, run_id, run_batch=runner)
    assert observed["waiting"], "the skill write did not wait for the run's row lock"
    assert "raced" not in await skills(world, include_deleted=True)
    row = (await experiences(world))[exp.id]
    assert row.status == "pending" and not row.influenced_skill_ids
    [entry] = (await _run_record(run_id)).curator_metrics
    assert entry["status"] == "running"


async def test_a_lost_lease_mid_batch_stops_every_write(fake):
    """Through the real ``Agent.step``: the lease is lost between two skill
    writes of a batch; the second write is refused, the refusal ends the step
    (it is not a tool result the model could react to) and the run."""
    from mirix.services.skill_learning_run_manager import (
        LearningRunNotActive,
        SkillLearningRunManager,
    )

    world = await make_world(fake)
    # Two experiences: an edit budget of 2, so the second create reaches the
    # write (and its guard) instead of being refused for the budget.
    exps = [await add_experience(world, title=f"lease {i}") for i in range(2)]
    run_id = await start_run(world)
    asked, expired = threading.Event(), threading.Event()

    def second_call(body):
        # Answered on the fake's thread once the lease is gone.
        asked.set()
        assert expired.wait(30), "the lease was never expired"
        return _call(
            "skill_create",
            {
                "name": "after",
                "description": "Opens an item and checks it",
                "instructions": "Open the item, check it, then close it.",
                "entry_type": "guide",
            },
        )

    fake.queue_response(
        _call(
            "skill_create",
            {
                "name": "before",
                "description": "Opens an item and checks it",
                "instructions": "Open the item, check it, then close it.",
                "entry_type": "guide",
            },
        )
    )
    fake.set_default_responder(second_call)

    async def expire_when_asked():
        while not asked.is_set():
            await asyncio.sleep(0.02)
        assert await SkillLearningRunManager().fail(run_id, "lease_expired")
        expired.set()

    expiry = asyncio.create_task(expire_when_asked())
    try:
        with pytest.raises(LearningRunNotActive):
            await _curate(world, run_id)
    finally:
        await asyncio.wait_for(expiry, 30)
    assert len(fake.chat_requests()) == 2  # no call after the refused write
    names = set(await skills(world, include_deleted=True))
    assert "before" in names and "after" not in names
    rows = await experiences(world)
    before_id = (await skills(world))["before"].id
    for exp in exps:
        row = rows[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0
        assert row.influenced_skill_ids == [before_id]  # for the next attempt
    [entry] = (await _run_record(run_id)).curator_metrics
    assert entry["status"] == "running"


async def test_an_unfinished_batch_is_closed_out_at_the_third_attempt(fake):
    world = await make_world(fake)
    exp = await add_experience(world, title="unfinished")

    async def runner(agent, payload, *, actor, user):
        return MirixUsageStatistics(step_count=1, stop_reason="step_cap")

    for attempt in (1, 2, 3):
        run_id = await start_run(world)
        await _curate(world, run_id, run_batch=runner)
        row = (await experiences(world))[exp.id]
        assert row.curation_attempts == attempt
    assert row.status == "consumed" and row.consumed_by == f"{run_id}:b0:unaddressed"
    [entry] = (await _run_record(run_id)).curator_metrics
    assert entry["unaddressed_ids"] == [exp.id]


async def test_another_scope_is_invisible_to_the_curator(fake):
    world = await make_world(fake)
    foreign = procedural_agent(world, scope="foreign-scope")
    foreign_id = _id(
        await mt.skill_create(
            foreign,
            name="foreign-skill",
            description="Handles the foreign scope's task",
            instructions="Steps for foreign-skill",
            entry_type="guide",
            triggers=["foreign trigger"],
        )
    )
    mine = procedural_agent(world)
    for i in range(55):
        await mt.skill_create(
            mine,
            name=f"own-{i:02d}",
            description="Opens an item and checks it",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
    await add_experience(world, title="scope")
    run_id = await start_run(world)
    answers = {}

    async def runner(agent, payload, *, actor, user):
        answers["page_1"] = await mt.skill_list(agent)
        answers["page_2"] = await mt.skill_list(agent, offset=50)
        answers["query"] = await mt.skill_list(agent, query="foreign trigger")
        answers["read"] = await mt.skill_read(agent, foreign_id)
        answers["edit"] = await mt.skill_edit(
            agent, foreign_id, "instructions", value="x"
        )
        answers["delete"] = await mt.skill_delete(agent, foreign_id, reason="x")
        return MirixUsageStatistics(step_count=1, stop_reason="finish")

    outcome = await _curate(world, run_id, run_batch=runner)
    assert "own-00" in answers["page_1"] and "own-49" in answers["page_1"]
    assert "own-50" in answers["page_2"] and "own-54" in answers["page_2"]
    assert "foreign-skill" not in answers["page_1"] + answers["page_2"]
    assert "No skills found" in answers["query"]
    for key in ("read", "edit", "delete"):
        assert "not found" in answers[key], (key, answers[key])
    [entry] = outcome.batches
    assert entry["writes"] == 0 and entry["deletes"] == 0
    assert entry["rejected_by_kind"] == {"not_found": 3}
    foreign_row = (await skills(world))["foreign-skill"]
    assert foreign_row.instructions == "Steps for foreign-skill"
    assert foreign_row.filter_tags == {"scope": "foreign-scope"}


# --------------------------------------------------------------------------- #
# The experience pool on Postgres                                              #
# --------------------------------------------------------------------------- #


def _pool_vec(cos_to_e0: float, axis: int = 1, dim: int = 3072) -> list:
    """A 3,072-wide vector (gemini-embedding-001's width) at ``cos_to_e0``."""
    v = [0.0] * dim
    v[0] = cos_to_e0
    v[axis] = (max(0.0, 1.0 - cos_to_e0 * cos_to_e0)) ** 0.5
    return v


def _pool_embedder(vectors, calls):
    async def embed_one(text):
        title = text.split("\n\n", 1)[0]
        calls.append(title)
        return vectors[title]

    return embed_one


async def test_the_pool_round_trips_pgvector_and_settles_under_row_locks(fake):
    """A deferred lesson's padded vector is stored in pgvector and read back
    (the pool scan undefers it), a similar new experience brings it back as an
    earlier lesson, and the consume settle (FOR UPDATE on the batch and the
    pool rows) records the citation, the recurrence and the retirement."""
    from sqlalchemy import text

    from mirix.constants import MAX_EMBEDDING_DIM
    from mirix.server.server import db_context
    from mirix.services.skill_experience_manager import SkillExperienceManager

    world = await make_world(fake)
    calls = []
    embed = _pool_embedder(
        {"pg lesson A": _pool_vec(1.0), "pg lesson B": _pool_vec(0.9)}, calls
    )
    a = await add_experience(world, title="pg lesson A")

    async def finish(agent, payload, *, actor, user):
        return MirixUsageStatistics(step_count=1, stop_reason="finish")

    await _curate(world, await start_run(world), run_batch=finish, embed_one=embed)
    assert calls == []  # an empty pool embeds nothing
    b = await add_experience(world, title="pg lesson B")
    seen = {}

    async def cite(agent, payload, *, actor, user):
        seen["payload"] = payload
        answer = await mt.skill_create(
            agent,
            name="pg-general-skill",
            description="Opens an item and checks it",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
            experience_ids=[
                {"id": i, "quote": "check it, then close it"} for i in (a.id, b.id)
            ],
        )
        seen["skill"] = _id(answer)
        await mt.experience_triage(
            agent, covered_ids=[], not_worth_ids=[], deferred_ids=[]
        )
        return MirixUsageStatistics(step_count=1, stop_reason="finish")

    run_id = await start_run(world)
    await _curate(world, run_id, run_batch=cite, embed_one=embed)
    assert sorted(calls) == ["pg lesson A", "pg lesson B"]
    assert f'<earlier_lesson id="{a.id}">' in seen["payload"]
    async with db_context() as db:
        [(dims, first, typ)] = (
            await db.execute(
                text(
                    "SELECT vector_dims(pool_embedding), "
                    "(pool_embedding::real[])[1], "
                    "format_type(a.atttypid, a.atttypmod) "
                    "FROM skill_experience, pg_attribute a "
                    "WHERE skill_experience.id = :id "
                    "AND a.attrelid = 'skill_experience'::regclass "
                    "AND a.attname = 'pool_embedding'"
                ),
                {"id": a.id},
            )
        ).all()
    assert dims == MAX_EMBEDDING_DIM and first == 1.0 and typ == "vector(4096)"
    rows = await experiences(world)
    assert rows[a.id].pool_retire_reason == "used"
    assert rows[a.id].recurrence_count == 2 and rows[a.id].pushed_count == 1
    assert rows[a.id].cited_skill_ids == [seen["skill"]]
    assert rows[b.id].disposition == "used"
    [entry] = (await _run_record(run_id)).curator_metrics
    assert entry["history_ids"] == [a.id] and entry["recurrences"] == 1
    assert entry["pool_active_after"] == 0
    assert entry["history"]["max_cosines"][0]["max_cosine"] == 0.9

    # The pool scan on Postgres: deferred lessons with their vectors, in scope.
    later = _pool_embedder(
        {"pg lesson C": _pool_vec(0.1, axis=2), "pg lesson D": _pool_vec(0.1, axis=3)},
        calls,
    )
    c = await add_experience(world, title="pg lesson C")
    await _curate(world, await start_run(world), run_batch=finish, embed_one=later)
    d = await add_experience(world, title="pg lesson D")
    await _curate(world, await start_run(world), run_batch=finish, embed_one=later)
    candidates, truncated = await SkillExperienceManager().list_active_pool(
        agent_id=world.meta.id,
        user_id=world.user.id,
        organization_id=world.org,
        scope=world.scope,
        limit=10,
    )
    assert [cand.id for cand in candidates] == [c.id, d.id] and not truncated
    for cand in candidates:
        # pgvector's text form, parsed by the column type's own processor.
        assert len(cand.vector) == MAX_EMBEDDING_DIM
        assert cand.vector.dtype == np.float32
        assert float(cand.vector[0]) == pytest.approx(0.1, abs=1e-6)
        assert cand.vector_model == "text-embedding-3-small@1536"
    # C and D are unrelated (cosine 0.01 apart from e0): C was not shown to D.
    assert candidates[0].pushed_count == 0
    assert (
        await SkillExperienceManager().list_active_pool(
            agent_id=world.meta.id,
            user_id=world.user.id,
            organization_id=world.org,
            scope="another-scope",
            limit=10,
        )
    ) == ([], False)
