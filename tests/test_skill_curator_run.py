"""The curator stage of a learning run.

Hermetic and offline: a fresh database initialized like server startup
(``db_isolation.isolated_database``), a real meta agent with its procedural
child (real tools, real system prompt), and every model call answered by the
local fake LLM server. Two kinds of tests:

* the bookkeeping, with an injected batch runner that calls the real skill
  tools on the real agent instance: finish consumes the batch; context_limit /
  step_cap return it with ``curation_attempts + 1`` and close it out
  (unaddressed) at the 3rd attempt; an infrastructure failure keeps it pending
  uncounted and stops the remaining batches; a non-infrastructure failure
  counts; one delete per run across batches; a lost lease stops all writing;
  an interrupted batch leaves its lineage for the next attempt; the per-batch
  metrics; the run's changes merge every batch's;
* the real ``Agent.step`` with scripted tool calls: two batches in order (the
  second sees the first's skill), the model override, no auto-summary,
  the three stop reasons, and invalid tool arguments costing nothing.
"""

import asyncio
import json
import re
import time
from pathlib import Path

import pytest
import pytest_asyncio

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

pytestmark = [pytest.mark.asyncio(loop_scope="module")]

REPO = Path(__file__).resolve().parent.parent


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
    from mirix.services import context_budget
    from mirix.settings import settings

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "images_dir", tmp_path_factory.mktemp("images"))
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
        # Deterministic counts: the 3-characters-per-token estimate, with no
        # vocabulary load attempted during the tests.
        context_budget._reset_tokenizer_state_for_tests()
        context_budget._tokenizer_state.failed_at = time.monotonic() + 10**9
        async with isolated_database(tmp_path_factory.mktemp("curator-db")) as engine:
            yield engine
        context_budget._reset_tokenizer_state_for_tests()


@pytest.fixture(autouse=True)
def _fresh_fake(fake):
    fake.reset()
    fake.set_default_responder(None)
    yield
    fake.reset()


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


def _usage(stop_reason, steps=1):
    return MirixUsageStatistics(step_count=steps, stop_reason=stop_reason)


def _pydantic_validation_error():
    """A real pydantic ``ValidationError`` (a ``ValueError`` subclass)."""
    from pydantic import TypeAdapter, ValidationError

    try:
        TypeAdapter(int).validate_python("not a number")
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def _big_content(i):
    # ~17,000 characters: two such experiences never share a batch of the
    # default 32k window (capacity ~ 7,000 tokens at 3 characters per token,
    # less the system prompt and the tool schemas), and one fits whole (19,000
    # stopped fitting once the curator prompt grew).
    return f"LESSON-{i} " + words(17_000, seed=i)


# --------------------------------------------------------------------------- #
# Bookkeeping, with an injected batch runner                                   #
# --------------------------------------------------------------------------- #


class TestSettling:
    async def test_a_citation_without_a_quote_leaves_its_experience_pending(self, fake):
        """End to end: the write goes through, the experience it cited
        without a verifiable quote is not consumed by the finished batch."""
        world = await make_world(fake)
        quoted, unquoted, other = [
            await add_experience(world, title=f"quote {i}") for i in range(3)
        ]
        run_id = await start_run(world)

        async def runner(agent, payload, *, actor, user):
            created = await mt.skill_create(
                agent,
                name="quoted-skill",
                description="Opens an item and checks it",
                instructions="Open the item. Weigh it before labelling it.",
                entry_type="workflow",
                experience_ids=[
                    {"id": quoted.id, "quote": "weigh it before labelling"},
                    {"id": unquoted.id, "quote": ""},
                ],
            )
            assert "They stay pending" in created, created
            return MirixUsageStatistics(step_count=2, stop_reason="finish")

        await _curate(world, run_id, run_batch=runner)
        rows = await experiences(world)
        assert rows[quoted.id].disposition == "used"
        assert rows[unquoted.id].status == "pending"
        assert rows[unquoted.id].curation_attempts == 1
        assert rows[other.id].disposition == "deferred"  # neither cited: as before
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["consumed"] == 2 and entry["returned"] == 1
        assert entry["returned_ids"] == [unquoted.id]
        assert entry["unverified_citations"] == {
            unquoted.id: [rows[quoted.id].cited_skill_ids[0]]
        }

    async def test_finish_consumes_the_whole_batch_with_its_metrics(self, fake):
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"finish {i}") for i in range(3)]
        run_id = await start_run(world)
        seen = {}

        async def runner(agent, payload, *, actor, user):
            seen.update(
                learning_run_id=agent.learning_run_id,
                skill_run_id=agent.skill_run_id,
                budget=agent.curator_batch.budget,
                payload=payload,
            )
            created = await mt.skill_create(
                agent,
                name="finished-skill",
                description="Opens an item and checks it",
                instructions="Open the item, then do it.",
                entry_type="workflow",
            )
            assert "created successfully" in created
            rejected = await mt.skill_create(
                agent,
                name="finished-skill",
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            assert "already exists" in rejected
            return MirixUsageStatistics(
                step_count=4,
                stop_reason="finish",
                context_peak_tokens=1234,
                failed_tool_calls=1,
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
            )

        outcome = await _curate(world, run_id, run_batch=runner)
        batch_id = f"{run_id}:b0"
        assert seen["learning_run_id"] == run_id and seen["skill_run_id"] == batch_id
        assert seen["budget"] == 3
        for exp in exps:  # every experience rendered in the one payload
            assert exp.id in seen["payload"]
        rows = await experiences(world)
        assert {r.status for r in rows.values()} == {"consumed"}
        assert {r.consumed_by for r in rows.values()} == {batch_id}
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["batch_id"] == batch_id and entry["status"] == "settled"
        assert entry["experiences"] == entry["budget"] == 3
        assert entry["writes"] == 1 and entry["creates"] == 1 and entry["deletes"] == 0
        assert entry["steps"] == 4 and entry["stop_reason"] == "finish"
        assert entry["context_peak_tokens"] == 1234
        assert entry["rejected_tool_calls"] == 2  # 1 by the tool, 1 by the agent
        assert entry["rejected_by_kind"] == {"duplicate_name": 1}
        assert entry["returned"] == 0 and entry["unaddressed"] == 0
        assert entry["consumed"] == 3
        run = await _run_record(run_id)
        assert run.llm_usage["by_stage"]["curate"]["calls"] == 4
        assert outcome.skills_changed == 1
        skill = (await skills(world))["finished-skill"]
        assert await versions(skill.id) == [(1, "create", "curator", batch_id, None)]

    @pytest.mark.parametrize("stop", ["context_limit", "step_cap"])
    async def test_an_unfinished_batch_returns_and_is_closed_out_at_the_third_attempt(
        self, fake, stop
    ):
        world = await make_world(fake)
        exp = await add_experience(world, title=f"unfinished {stop}")

        async def runner(agent, payload, *, actor, user):
            return _usage(stop, steps=2)

        run_ids = []
        for attempt in (1, 2, 3):
            run_id = await start_run(world)
            run_ids.append(run_id)
            await _curate(world, run_id, run_batch=runner)
            row = (await experiences(world))[exp.id]
            assert row.curation_attempts == attempt
            [entry] = (await _run_record(run_id)).curator_metrics
            assert entry["stop_reason"] == stop
            if attempt < 3:
                assert row.status == "pending" and entry["returned"] == 1
                assert entry["unaddressed"] == 0
            else:
                assert row.status == "consumed"
                assert row.consumed_by == f"{run_id}:b0:unaddressed"
                assert entry["unaddressed"] == 1 and entry["unaddressed_ids"] == [
                    exp.id
                ]
        # Closed out: a 4th run has nothing to curate.
        outcome = await _curate(world, await start_run(world), run_batch=runner)
        assert outcome.skipped == "no_pending_experiences"

    async def test_an_infrastructure_failure_keeps_everything_pending_uncounted(
        self, fake
    ):
        from mirix.errors import LLMServerError

        world = await make_world(fake)
        exps = [
            await add_experience(world, title=f"infra {i}", content=_big_content(i))
            for i in range(2)
        ]
        run_id = await start_run(world)
        calls = []

        async def runner(agent, payload, *, actor, user):
            calls.append(agent.skill_run_id)
            raise LLMServerError("provider down")

        outcome = await _curate(world, run_id, run_batch=runner)
        # The first batch failed; the second was not even tried.
        assert calls == [f"{run_id}:b0"]
        [entry] = outcome.batches
        assert entry["error_kind"] == "infra" and entry["outcome"] == "keep"
        rows = await experiences(world)
        assert all(rows[e.id].status == "pending" for e in exps)
        assert all(rows[e.id].curation_attempts == 0 for e in exps)

    async def test_a_database_failure_is_recorded_then_fails_the_run(self, fake):
        from sqlalchemy.exc import OperationalError

        world = await make_world(fake)
        exps = [
            await add_experience(world, title=f"db {i}", content=_big_content(30 + i))
            for i in range(2)
        ]
        run_id = await start_run(world)
        calls = []

        async def runner(agent, payload, *, actor, user):
            calls.append(agent.skill_run_id)
            raise OperationalError("UPDATE agents", {}, Exception("database is locked"))

        with pytest.raises(OperationalError):
            await _curate(world, run_id, run_batch=runner)
        assert calls == [f"{run_id}:b0"]  # the second batch never ran
        rows = await experiences(world)
        assert all(rows[e.id].status == "pending" for e in exps)
        assert all(rows[e.id].curation_attempts == 0 for e in exps)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["status"] == "settled" and entry["error_kind"] == "db"
        assert entry["outcome"] == "keep"

    @pytest.mark.parametrize(
        "error, kind",
        [
            ("truncated", "truncated"),
            ("unusable", "parse"),
            ("refused", "other"),
            ("over_budget", "budget"),
            ("timed_out", "timeout"),
        ],
    )
    async def test_a_failed_model_exchange_counts_an_attempt(self, fake, error, kind):
        from mirix import errors

        failures = {
            "truncated": errors.LLMOutputTruncatedError(),
            "unusable": errors.LLMInvalidResponseError("empty message, 3 times"),
            "refused": errors.LLMBadRequestError("400: invalid tool schema"),
            "over_budget": errors.ContextBudgetExceeded(),
            "timed_out": errors.LLMTimeoutError("600 s deadline"),
        }
        world = await make_world(fake)
        exp = await add_experience(world, title=f"exchange {error}")
        run_id = await start_run(world)

        async def runner(agent, payload, *, actor, user):
            raise failures[error]

        outcome = await _curate(world, run_id, run_batch=runner)
        [entry] = outcome.batches
        assert entry["stop_reason"] == "error" and entry["error_kind"] == kind
        assert entry["outcome"] == "return"
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 1

    @pytest.mark.parametrize(
        "bug",
        [
            ValueError("stored agent state no longer validates"),
            KeyError("tool_rules"),
            TypeError("unexpected keyword"),
        ],
        ids=["ValueError", "KeyError", "TypeError"],
    )
    async def test_a_bug_keeps_the_batch_pending_never_counts_and_fails_the_run(
        self, fake, bug
    ):
        # An exception during the run leaves the batch pending. A
        # deterministic bug must not close every experience out as
        # "unaddressed" after three runs: it is never counted. It is MIRIX's
        # own failure, so the run fails with it (recorded) instead of
        # "succeeding" without curating anything, and the later batches stay
        # pending for a fixed MIRIX.
        world = await make_world(fake)
        exps = [
            await add_experience(world, title=f"bug {i}", content=_big_content(50 + i))
            for i in range(2)
        ]
        calls = []

        async def runner(agent, payload, *, actor, user):
            calls.append(agent.skill_run_id)
            raise bug

        for _ in range(4):
            run_id = await start_run(world)
            with pytest.raises(type(bug)):
                await _curate(world, run_id, run_batch=runner)
            [entry] = (await _run_record(run_id)).curator_metrics
            assert entry["outcome"] == "keep" and entry["error_kind"] == "other"
            assert entry["status"] == "settled"
        # The first batch ran each time; the second never did ...
        assert len(calls) == 4 and all(c.endswith(":b0") for c in calls)
        rows = await experiences(world)
        # ... and nothing was counted or closed out.
        assert all(rows[e.id].status == "pending" for e in exps)
        assert all(rows[e.id].curation_attempts == 0 for e in exps)

    async def test_the_settle_is_safe_to_repeat(self, fake):
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        exp = await add_experience(world, title="settle twice")
        run_id = await start_run(world)
        runs = SkillLearningRunManager()
        for _ in range(2):
            settled = await runs.settle_curator_batch(
                run_id,
                organization_id=world.org,
                user_id=world.user.id,
                agent_id=world.meta.id,
                batch_id=f"{run_id}:b0",
                experience_ids=[exp.id],
                outcome="return",
                entry={"batch_id": f"{run_id}:b0"},
            )
            assert settled["returned"] == [exp.id]
        assert (await experiences(world))[exp.id].curation_attempts == 1
        assert len((await _run_record(run_id)).curator_metrics) == 1

    async def test_the_settle_is_safe_to_repeat_with_the_pool(self, fake):
        """A repeated settle (a database retry) of a batch already
        settled returns its recorded pool summary and applies nothing twice:
        no second recurrence, decay or retirement."""
        from mirix.services.skill_experience_manager import PoolSettlement
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        world = await make_world(fake)
        embed = _Embedder({"shown": _vec(1.0), "worthless": _vec(0.2, 2)})
        shown = await add_experience(world, title="shown")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        new = await add_experience(world, title="settle with the pool")
        worthless = await add_experience(world, title="worthless")
        run_id = await start_run(world)
        runs = SkillLearningRunManager()
        pool = PoolSettlement(
            scope=world.scope,
            pushed_ids=(shown.id,),
            recurred_ids=(shown.id,),
            triage={worthless.id: "not_worth"},
        )
        results = []
        for _ in range(3):
            results.append(
                await runs.settle_curator_batch(
                    run_id,
                    organization_id=world.org,
                    user_id=world.user.id,
                    agent_id=world.meta.id,
                    batch_id=f"{run_id}:b0",
                    experience_ids=[new.id, worthless.id],
                    outcome="consume",
                    entry={"batch_id": f"{run_id}:b0"},
                    pool=pool,
                )
            )
        first = results[0]
        assert first["consumed"] == [new.id, worthless.id]
        assert first["pool"]["recurrences"] == 1
        assert first["pool"]["history_ignored"] == 1
        assert first["pool"]["dispositions"]["not_worth"] == 1
        for again in results[1:]:
            assert again == first  # the recorded summary, nothing applied twice
        row = await _pool_row_state(shown.id)
        assert (row.pushed_count, row.recurrence_count, row.ignored_pushes) == (1, 2, 1)
        assert (await _pool_row_state(worthless.id)).ignored_pushes == 1
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["recurrences"] == 1 and entry["pool_active_after"] == 3

    async def test_the_stage_loads_the_vocabulary_before_planning(
        self, fake, monkeypatch
    ):
        # A failed startup preload is retried before the batches are
        # planned (the default counter only; an injected counter is exact).
        from mirix.services import context_budget

        world = await make_world(fake)
        await add_experience(world, title="vocab")
        calls = []

        async def spy():
            calls.append(len(fake.chat_requests()))
            return False

        async def runner(agent, payload, *, actor, user):
            return _usage("finish")

        monkeypatch.setattr(context_budget, "ensure_tokenizer", spy)
        await _curate(world, await start_run(world), run_batch=runner)
        assert calls == [0]  # once, before any batch
        await add_experience(world, title="vocab-2")
        await _curate(
            world,
            await start_run(world),
            run_batch=runner,
            token_counter=context_budget.TokenCounter(),
        )
        assert calls == [0]

    async def test_a_scope_less_owner_is_skipped_and_touches_no_skill(self, fake):
        # Fails closed: without a write scope the skill tools would reach
        # the user's skills of every scope.
        from mirix.services.skill_experience_curator import curate_pending_experiences

        world = await make_world(fake)
        other = procedural_agent(world, scope="another-scope")
        await mt.skill_create(
            other,
            name="another-scopes-skill",
            description="Opens an item and checks it",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
        exp = await add_experience(world, title="scope-less")
        run_id = await start_run(world)
        ran = []

        async def runner(agent, payload, *, actor, user):
            ran.append(agent)
            return _usage("finish")

        outcome = await curate_pending_experiences(
            run_id=run_id,
            user=world.user,
            actor=world.actor.model_copy(update={"write_scope": None}),
            meta_agent_state=world.meta,
            run_batch=runner,
        )
        assert outcome.skipped == "no_write_scope" and outcome.batches == []
        assert ran == [] and fake.chat_requests() == []
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0
        assert not (await _run_record(run_id)).curator_metrics

    async def test_the_stages_retries_land_in_the_runs_retry_events(
        self, fake, monkeypatch
    ):
        # Every retry is recorded — here a skill write's transaction
        # retry (a transient database failure) inside a batch.
        from sqlalchemy.exc import OperationalError

        from mirix.services.procedural_memory_manager import ProceduralMemoryManager

        world = await make_world(fake)
        await add_experience(world, title="retried write")
        run_id = await start_run(world)
        original = ProceduralMemoryManager._commit_change
        failed = []

        async def flaky_commit(self, plan, **kwargs):
            if not failed:
                failed.append(plan.skill_id)
                raise OperationalError(
                    "INSERT INTO procedural_memory", {}, Exception("database is locked")
                )
            return await original(self, plan, **kwargs)

        monkeypatch.setattr(ProceduralMemoryManager, "_commit_change", flaky_commit)

        async def runner(agent, payload, *, actor, user):
            created = await mt.skill_create(
                agent,
                name="written-after-a-retry",
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            assert "created successfully" in created
            return _usage("finish")

        await _curate(world, run_id, run_batch=runner)
        assert failed and "written-after-a-retry" in await skills(world)
        events = (await _run_record(run_id)).retry_events
        [event] = [e for e in events if e.get("stage") == "skill_write"]
        assert event["kind"] == "db" and event["attempt"] == 1
        assert "database is locked" in event["error"]

    async def test_nothing_pending_or_no_procedural_agent_is_a_clean_skip(
        self, fake, monkeypatch
    ):
        import mirix.services.skill_experience_curator as curator

        world = await make_world(fake)
        run_id = await start_run(world)
        assert (await _curate(world, run_id)).skipped == "no_pending_experiences"
        exp = await add_experience(world, title="no agent")

        async def no_agent(*args, **kwargs):
            return None

        monkeypatch.setattr(curator, "_resolve_procedural_agent_state", no_agent)
        assert (await _curate(world, run_id)).skipped == "no_procedural_agent"
        assert (await experiences(world))[exp.id].status == "pending"


class TestBatchesInOneRun:
    async def test_batches_run_in_order_and_later_ones_see_earlier_edits(self, fake):
        world = await make_world(fake)
        exps = [
            await add_experience(world, title=f"big {i}", content=_big_content(i))
            for i in range(3)
        ]
        run_id = await start_run(world)
        seen_lists = []

        async def runner(agent, payload, *, actor, user):
            index = int(agent.skill_run_id.rsplit(":b", 1)[1])
            assert f"LESSON-{index} " in payload  # its own experience, in order
            seen_lists.append(await mt.skill_list(agent))
            await mt.skill_create(
                agent,
                name=f"skill-of-batch-{index}",
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            return _usage("finish")

        outcome = await _curate(world, run_id, run_batch=runner)
        assert [b["batch_id"] for b in outcome.batches] == [
            f"{run_id}:b{i}" for i in range(3)
        ]
        assert "No skills found" in seen_lists[0]
        assert "skill-of-batch-0" in seen_lists[1]
        assert "skill-of-batch-1" in seen_lists[2]
        rows = await experiences(world)
        assert [rows[e.id].consumed_by for e in exps] == [
            f"{run_id}:b{i}" for i in range(3)
        ]
        # Every batch's changes are in the outcome, not only the first's.
        assert len(outcome.changes["created"]) == 3 and outcome.skills_changed == 3
        for entry in outcome.batches:
            assert entry["payload_tokens"] <= entry["capacity_tokens"]

    async def test_one_delete_per_run_across_batches(self, fake):
        world = await make_world(fake)
        setup_agent = procedural_agent(world)
        ids = []
        for name in ("doomed-a", "doomed-b"):
            answer = await mt.skill_create(
                setup_agent,
                name=name,
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            ids.append(answer.split("ID: ")[1].split(",")[0])
        for i in range(2):
            await add_experience(world, title=f"del {i}", content=_big_content(10 + i))
        run_id = await start_run(world)
        answers = []

        async def runner(agent, payload, *, actor, user):
            index = int(agent.skill_run_id.rsplit(":b", 1)[1])
            answers.append(await mt.skill_delete(agent, ids[index], reason="wrong"))
            return _usage("finish")

        outcome = await _curate(world, run_id, run_batch=runner)
        assert len(outcome.batches) == 2
        assert "soft-deleted" in answers[0] and "at most 1" in answers[1]
        assert set(await skills(world)) == {"doomed-b"}
        assert [b["deletes"] for b in outcome.batches] == [1, 0]
        assert outcome.batches[1]["rejected_by_kind"] == {"delete_limit": 1}


class TestSafety:
    async def test_a_lost_lease_stops_every_write(self, fake):
        from mirix.services.skill_learning_run_manager import (
            LearningRunNotActive,
            SkillLearningRunManager,
        )

        world = await make_world(fake)
        exp = await add_experience(world, title="lease")
        run_id = await start_run(world)

        async def runner(agent, payload, *, actor, user):
            # The lease expires (another process reclaims the run) mid-batch.
            assert await SkillLearningRunManager().fail(run_id, "lease_expired")
            await mt.skill_create(
                agent,
                name="too-late",
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            raise AssertionError("the write should have ended the batch")

        with pytest.raises(LearningRunNotActive):
            await _curate(world, run_id, run_batch=runner)
        assert "too-late" not in await skills(world, include_deleted=True)
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["status"] == "running"  # never settled

    async def test_an_interrupted_batch_leaves_its_lineage_for_the_next_attempt(
        self, fake
    ):
        world = await make_world(fake)
        exp = await add_experience(world, title="interrupted")
        first_run = await start_run(world)
        created = {}

        async def interrupted(agent, payload, *, actor, user):
            answer = await mt.skill_create(
                agent,
                name="half-done",
                description="Opens an item and checks it",
                instructions="Open the item, check it, then close it.",
                entry_type="guide",
            )
            created["id"] = answer.split("ID: ")[1].split(",")[0]
            raise asyncio.CancelledError()  # the process is stopped mid-batch

        with pytest.raises(asyncio.CancelledError):
            await _curate(world, first_run, run_batch=interrupted)
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.influenced_skill_ids == [created["id"]]
        [entry] = (await _run_record(first_run)).curator_metrics
        assert entry["status"] == "running"

        second_run = await start_run(world)
        payloads = []

        async def resumed(agent, payload, *, actor, user):
            payloads.append(payload)
            return _usage("finish")

        await _curate(world, second_run, run_batch=resumed)
        assert f'{created["id"]} "half-done"' in payloads[0]
        assert "already changed by an earlier attempt" in payloads[0]
        assert (await experiences(world))[exp.id].consumed_by == f"{second_run}:b0"

    async def test_lineage_never_names_a_skill_of_another_scope(self, fake):
        from sqlalchemy import update

        from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
        from mirix.server.server import db_context

        world = await make_world(fake)
        foreign = procedural_agent(world, scope="foreign-scope")
        answer = await mt.skill_create(
            foreign,
            name="secret-foreign-name",
            description="Opens an item and checks it",
            instructions="Open the item, check it, then close it.",
            entry_type="guide",
        )
        foreign_id = answer.split("ID: ")[1].split(",")[0]
        exp = await add_experience(world, title="lineage scope")
        async with db_context() as db:
            await db.execute(
                update(SkillExperienceORM)
                .where(SkillExperienceORM.id == exp.id)
                .values(influenced_skill_ids=[foreign_id], curation_attempts=1)
            )
            await db.commit()
        payloads = []

        async def capture(agent, payload, *, actor, user):
            payloads.append(payload)
            return _usage("finish")

        await _curate(world, await start_run(world), run_batch=capture)
        assert "secret-foreign-name" not in payloads[0]
        assert f"{foreign_id} (no longer exists or not accessible)" in payloads[0]


# --------------------------------------------------------------------------- #
# The real Agent.step against the fake model server                           #
# --------------------------------------------------------------------------- #

_HUGE_USAGE = {
    "prompt_tokens": 900_000,
    "completion_tokens": 10,
    "total_tokens": 900_010,
}


def _call(name, args=None, usage=None):
    return ChatResponse(
        tool_calls=[{"name": name, "arguments": args or {}}],
        usage=usage,
    )


class TestRealAgentStep:
    async def test_two_batches_scripted_end_to_end(self, fake, monkeypatch):
        from mirix.agent.agent import Agent
        from mirix.services.skill_experience_curator import run_curator_batch_step

        async def no_summaries(self, *args, **kwargs):
            raise AssertionError("the curator must never summarize")

        monkeypatch.setattr(Agent, "summarize_messages_inplace", no_summaries)
        world = await make_world(fake)
        exps = [
            await add_experience(world, title=f"e2e {i}", content=_big_content(20 + i))
            for i in range(2)
        ]
        run_id = await start_run(world)
        # Batch 0: survey, create, finish. Its reported usage is far above the
        # window: an auto-summarizing agent would summarize right here.
        fake.queue_response(
            _call("skill_list", {}, usage=_HUGE_USAGE),
            _call(
                "skill_create",
                {
                    "name": "e2e-skill",
                    "description": "the first version of it",
                    "instructions": "Open it. Check it.",
                    "entry_type": "workflow",
                    "triggers": ["when checking"],
                    # Every experience is cited (with a quote of the text
                    # that applies it) or triaged before the finish.
                    "experience_ids": [
                        {"id": exps[0].id, "quote": "Open it. Check it."}
                    ],
                },
                usage=_HUGE_USAGE,
            ),
            _call("finish_memory_update"),
            # Batch 1: survey (it must see batch 0's skill) and read it ...
            _call("skill_list", {}),
            _call("skill_read", {"name_or_id": "e2e-skill"}),
        )
        created = {}

        def batch_1_writes(body):
            # ... then rewrite the whole field, delete with a reason, finish
            # (scripted here: the calls need the id batch 0 created).
            step = created["step"] = created.get("step", 0) + 1
            if step == 1:
                return _call(
                    "skill_edit",
                    {
                        "skill_id": created["id"],
                        "field": "instructions",
                        "value": "A full rewrite.",
                        "experience_ids": [
                            {"id": exps[1].id, "quote": "A full rewrite."}
                        ],
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
                created["id"] = (await skills(world))["e2e-skill"].id
                fake.set_default_responder(batch_1_writes)
            return await run_curator_batch_step(agent, payload, actor=actor, user=user)

        outcome = await _curate(
            world, run_id, run_batch=run_batch, model_override="override-model"
        )
        assert fake.unscripted_requests == 0 and fake.pending_responses == 0
        requests = fake.chat_requests()
        assert len(requests) == 8
        # Every curator call ran on the run's model override ...
        assert {r.json["model"] for r in requests} == {"override-model"}
        # ... and the stored agent keeps its own model.
        stored = await world.server.agent_manager.get_agent_by_id(
            agent_id=world.procedural.id, actor=world.actor
        )
        assert stored.llm_config.model == "fake-model"
        # Batch 0's payload carried experience 0 in full; batch 1 started from a
        # fresh context with experience 1 only.
        assert _big_content(20) in user_texts(requests[0].json)[-1]
        batch_1_first = "".join(user_texts(requests[3].json))
        assert _big_content(21) in batch_1_first and "LESSON-20 " not in batch_1_first
        assert "batch 2 of 2" in batch_1_first
        # Batch 1 saw batch 0's skill when it surveyed.
        assert "e2e-skill" in tool_results(requests[4].json)[-1]
        assert "Open it. Check it." in tool_results(requests[5].json)[-1]
        assert [b["stop_reason"] for b in outcome.batches] == ["finish", "finish"]
        assert [b["steps"] for b in outcome.batches] == [3, 5]
        assert [b["writes"] for b in outcome.batches] == [1, 1]
        assert [b["deletes"] for b in outcome.batches] == [0, 1]
        assert all(b["context_peak_tokens"] > 0 for b in outcome.batches)
        assert all(b["rejected_tool_calls"] == 0 for b in outcome.batches)
        rows = await experiences(world)
        assert [rows[e.id].consumed_by for e in exps] == [
            f"{run_id}:b0",
            f"{run_id}:b1",
        ]
        skill_id = created["id"]
        assert [(v[1], v[3], v[4]) for v in await versions(skill_id)] == [
            ("create", f"{run_id}:b0", None),
            ("edit", f"{run_id}:b1", None),
            ("delete", f"{run_id}:b1", "merged elsewhere"),
        ]
        # The outcome merges both batches' changes.
        assert outcome.changes == {
            "created": [skill_id],
            "edited": [skill_id],
            "deleted": [skill_id],
        }
        run = await _run_record(run_id)
        assert [e["batch_id"] for e in run.curator_metrics] == [
            f"{run_id}:b0",
            f"{run_id}:b1",
        ]
        assert run.llm_usage["by_stage"]["curate"]["calls"] == 8

    @pytest.mark.parametrize("fault", ["503", "429", "timeout"])
    async def test_a_provider_failure_mid_batch_never_consumes_it(
        self, fake, monkeypatch, fault
    ):
        # The agent used to retry a failing call, then resend ONLY the last
        # message (a heartbeat: no system prompt, no batch payload). A finish
        # answered to that context-less call consumed the whole batch although
        # the model never saw it. The curator step is strict now: the provider
        # error (after the transport layer's retries — none here) ends the
        # batch and nothing is consumed.
        from mirix import constants
        from fakes.fake_llm_server import Fault

        world = await make_world(fake)
        exp = await add_experience(world, title=f"provider {fault}")
        run_id = await start_run(world)
        failure = {
            "503": Fault.http_error(503),
            "429": Fault.rate_limit(retry_after=0),
            "timeout": Fault.timeout(3),
        }[fault]
        if fault == "timeout":
            monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 0.3)
        # The provider recovers after three failures and would answer finish.
        fake.queue_response(
            _call("skill_list"),
            failure,
            failure,
            failure,
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 2  # one good call, one failed call — no more
        for request in requests:  # never a request without the system prompt
            assert request.json["messages"][0]["role"] == "system"
            assert any(f"provider {fault}" in t for t in user_texts(request.json))
        assert fake.pending_responses == 3  # the finish was never requested
        [entry] = outcome.batches
        assert entry["stop_reason"] == "error"
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.consumed_by is None
        if fault == "timeout":
            # A timed-out call is not an infrastructure failure; it
            # counts one curation attempt (the batch is still not consumed).
            assert entry["error_kind"] == "timeout" and entry["outcome"] == "return"
            assert row.curation_attempts == 1
        else:
            assert entry["error_kind"] == "infra" and entry["outcome"] == "keep"
            assert row.curation_attempts == 0

    @pytest.mark.parametrize(
        "make_bug",
        [
            lambda: KeyError("tool_rules"),
            lambda: AssertionError("stored message no longer converts"),
            lambda: ValueError("bad stored state"),
            lambda: _pydantic_validation_error(),
            # Causes whose own kind is PARSE / TIMEOUT / INFRA — raised
            # while MIRIX built the request, they are none of those.
            lambda: json.JSONDecodeError("stored tool arguments", "{", 1),
            lambda: TimeoutError("reading a stored attachment"),
            lambda: FileNotFoundError(2, "No such file", "/images/gone.png"),
        ],
        ids=[
            "KeyError",
            "AssertionError",
            "ValueError",
            "ValidationError",
            "JSONDecodeError",
            "TimeoutError",
            "FileNotFoundError",
        ],
    )
    async def test_a_request_mirix_cannot_build_is_never_counted_and_fails_the_run(
        self, fake, monkeypatch, make_bug
    ):
        # The strict step used to turn ANY ValueError / KeyError /
        # AssertionError of its LLM call into "the model's answer is
        # unparseable" (PARSE, counted) — MIRIX's own bug in build_request_data
        # included, before anything was sent. Three runs later every
        # experience was closed out as unaddressed. Now nothing is counted,
        # nothing reaches the model, and each run fails with the error.
        from mirix.errors import LLMRequestBuildError
        from mirix.llm_api.openai_client import OpenAIClient
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager

        async def broken_build(self, *args, **kwargs):
            raise make_bug()

        monkeypatch.setattr(OpenAIClient, "build_request_data", broken_build)
        world = await make_world(fake)
        exps = [
            await add_experience(
                world, title=f"build {i}", content=_big_content(60 + i)
            )
            for i in range(2)
        ]
        runs = SkillLearningRunManager()
        for _ in range(4):  # one more than the 3 attempts that close one out
            run_id = await start_run(world)
            with pytest.raises(LLMRequestBuildError) as raised:
                await runs.execute(run_id, lambda run_id=run_id: _curate(world, run_id))
            assert type(raised.value.__cause__) is type(make_bug())
            run = await _run_record(run_id)
            assert run.status == "failed"
            assert run.error.startswith("LLMRequestBuildError")
            [entry] = run.curator_metrics  # the second batch never started
            assert entry["error_kind"] == "internal" and entry["outcome"] == "keep"
        assert fake.chat_requests() == []  # nothing was ever sent
        rows = await experiences(world)
        assert all(rows[e.id].status == "pending" for e in exps)
        assert all(rows[e.id].curation_attempts == 0 for e in exps)
        assert all(rows[e.id].consumed_by is None for e in exps)

    async def test_an_sdk_call_mirix_gets_wrong_is_never_counted(
        self, fake, monkeypatch
    ):
        # The failure mode of an unpinned SDK: it refuses how MIRIX calls it (a TypeError
        # for an argument it does not take) before sending. That is MIRIX's
        # bug, not an LLMError the curator would count.
        from openai.resources.chat import AsyncCompletions

        from mirix.errors import LLMRequestBuildError

        async def refuse(self, *args, **kwargs):
            raise TypeError("create() got an unexpected keyword argument 'temperature'")

        monkeypatch.setattr(AsyncCompletions, "create", refuse)
        world = await make_world(fake)
        exp = await add_experience(world, title="sdk call")
        run_id = await start_run(world)
        with pytest.raises(LLMRequestBuildError) as raised:
            await _curate(world, run_id)
        assert raised.value.stage == "sdk_call"
        assert isinstance(raised.value.__cause__, TypeError)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["error_kind"] == "internal" and entry["outcome"] == "keep"
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0

    async def test_an_answer_that_stays_empty_is_still_a_counted_parse_failure(
        self, fake, monkeypatch
    ):
        # The other side: an answer the model DID send that stays
        # unusable after the agent's retries is the model's failure — PARSE,
        # one curation attempt — and the run itself goes on.
        from mirix.settings import settings

        monkeypatch.setattr(settings, "llm_retry_backoff_factor", 0.0)
        world = await make_world(fake)
        exp = await add_experience(world, title="empty answers")
        run_id = await start_run(world)
        fake.queue_response(*[ChatResponse(content="")] * settings.llm_retry_limit)
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == settings.llm_retry_limit
        [entry] = outcome.batches
        assert entry["error_kind"] == "parse" and entry["outcome"] == "return"
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 1

    async def test_transport_retries_are_the_only_retries_and_are_recorded(
        self, fake, monkeypatch
    ):
        # MIRIX_LLM_MAX_RETRIES bounds the retries of a curator
        # call — the SDK's transport retries, each one recorded on the run —
        # and the agent adds none on top of them.
        from mirix import constants
        from fakes.fake_llm_server import Fault

        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 2)
        world = await make_world(fake)
        exp = await add_experience(world, title="transport")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_list"),
            *[Fault.http_error(503)] * 6,
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 1 + 3  # 1 + MIRIX_LLM_MAX_RETRIES
        [entry] = outcome.batches
        assert entry["error_kind"] == "infra" and entry["outcome"] == "keep"
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0
        events = [
            e
            for e in (await _run_record(run_id)).retry_events
            if e.get("stage") == "transport"
        ]
        assert [(e["kind"], e["attempt"]) for e in events] == [
            ("infra", 1),
            ("infra", 2),
        ]

    async def test_step_cap(self, fake, monkeypatch):
        from mirix import constants

        monkeypatch.setattr(constants, "SKILL_CURATOR_MAX_STEPS", 2)
        world = await make_world(fake)
        exp = await add_experience(world, title="cap")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_list"), _call("skill_list"), _call("skill_list")
        )
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 2  # no grace call after the cap
        assert all(
            "maximum chaining steps" not in t
            for r in fake.chat_requests()
            for t in user_texts(r.json)
        )
        [entry] = outcome.batches
        assert entry["stop_reason"] == "step_cap" and entry["steps"] == 2
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 1

    async def test_context_limit_before_the_first_call(self, fake):
        from mirix.services.skill_experience_curator import run_curator_batch_step

        # floor(32000 * 0.8) - min(3200, 3200) = 22,400 tokens. The planned
        # payload always fits; one inflated past the limit must stop the step
        # before any call is made.
        world = await make_world(fake)
        exp = await add_experience(world, title="no room")
        run_id = await start_run(world)

        async def inflated(agent, payload, *, actor, user):
            return await run_curator_batch_step(
                agent, payload + "\n" + words(90_000, seed=3), actor=actor, user=user
            )

        outcome = await _curate(world, run_id, run_batch=inflated)
        assert fake.chat_requests() == []
        [entry] = outcome.batches
        assert entry["stop_reason"] == "context_limit" and entry["steps"] == 0
        assert entry["context_peak_tokens"] > 22_400
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 1

    async def test_a_window_too_small_for_the_system_prompt_is_a_config_skip(
        self, fake
    ):
        # floor(6000 * 0.8) - 600 = 4,200 tokens: the system prompt and tools
        # alone are more. Nothing is sent and nothing is counted.
        world = await make_world(fake, context_window=6_000, max_output_tokens=2_000)
        exp = await add_experience(world, title="window")
        run_id = await start_run(world)
        outcome = await _curate(world, run_id)
        assert outcome.skipped == "context_window_too_small" and outcome.batches == []
        assert fake.chat_requests() == []
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0
        assert not (await _run_record(run_id)).curator_metrics

    async def test_context_limit_in_the_middle_of_a_batch(self, fake):
        from mirix.services.context_budget import ModelLimits

        world = await make_world(fake, context_window=200_000, max_output_tokens=4_000)
        setup = procedural_agent(world)
        await mt.skill_create(
            setup,
            name="huge-skill",
            description="Opens an item and checks it",
            instructions=words(11_900, seed=7),
            entry_type="guide",
        )
        # Probe: the estimated input of a batch's first call.
        await add_experience(world, title="probe", content="short lesson")
        # The first finish lists the unaddressed experience; the second ends it.
        fake.queue_response(
            _call("finish_memory_update"), _call("finish_memory_update")
        )
        probe = await _curate(world, await start_run(world))
        first_call = probe.batches[0]["context_peak_tokens"]
        assert first_call > 0

        # Now a window whose limit is 1,500 tokens above that: the first call
        # fits, the call after reading the 11,900-character skill does not.
        limit = first_call + 1_500
        window = limit * 10 // 7  # limit = floor(0.8 w) - floor(0.1 w) below 32k
        limits = ModelLimits(context_window=window, max_output_tokens=4_000)
        assert limits.usable_window - limits.planning_output_reserve in range(
            limit - 2, limit + 2
        )
        await world.server.agent_manager.update_llm_config(
            agent_id=world.procedural.id,
            llm_config=world.llm_config.model_copy(update={"context_window": window}),
            actor=world.actor,
        )
        exp = await add_experience(world, title="probe", content="short lesson")
        run_id = await start_run(world)
        fake.reset()
        fake.queue_response(
            _call("skill_read", {"name_or_id": "huge-skill"}), _call("skill_list")
        )
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 1
        [entry] = outcome.batches
        assert entry["stop_reason"] == "context_limit" and entry["steps"] == 1
        assert entry["context_peak_tokens"] > limit - 2
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 1

    async def test_invalid_tool_arguments_cost_nothing(self, fake):
        world = await make_world(fake)
        exp = await add_experience(world, title="invalid args")
        run_id = await start_run(world)
        fake.queue_response(
            # Not JSON at all -> an error tool result from the agent.
            ChatResponse(
                tool_calls=[{"name": "skill_create", "arguments": "{not json"}]
            ),
            # Fails the validator (empty name).
            _call(
                "skill_create",
                {
                    "name": "",
                    "description": "Opens an item and checks it",
                    "instructions": "Open the item, check it, then close it.",
                    "entry_type": "guide",
                },
            ),
            # Missing the required reason.
            _call("skill_delete", {"skill_id": "proc-x"}),
            # A valid create still has the whole budget (1).
            _call(
                "skill_create",
                {
                    "name": "valid-one",
                    "description": "Opens an item and checks it",
                    "instructions": "Open the item, check it, then close it.",
                    "entry_type": "guide",
                    "experience_ids": [
                        {"id": exp.id, "quote": "Open the item, check it"}
                    ],
                },
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["writes"] == 1
        assert entry["failed_tool_calls"] == 3 and entry["rejected_tool_calls"] == 3
        # Why each call failed: all of them the model's own arguments.
        assert [
            (f["tool"], f["stage"], f["exception"], f["count"])
            for f in entry["tool_failures"]
        ] == [
            ("skill_create", "arguments", "ValueError", 1),
            ("skill_create", "validation", None, 1),
            ("skill_delete", "validation", None, 1),
        ]
        assert "'reason' is required" in entry["tool_failures"][2]["error"]
        assert entry["tool_failures_omitted"] == 0
        assert "valid-one" in await skills(world)
        results = [tool_results(r.json)[-1] for r in fake.chat_requests()[1:4]]
        assert all('"status": "Failed"' in r for r in results), results

    async def test_a_batch_serves_its_session_only_when_it_has_just_one(
        self, fake, monkeypatch
    ):
        """With MIRIX_LLM_SESSION_HEADER set, a curator batch whose experiences all
        come from one session sends that session on its model calls; a batch mixing
        sessions sends none (nothing is ever attributed to the wrong session)."""
        from mirix.settings import settings

        header = "x-skill-bench-item"
        monkeypatch.setattr(settings, "llm_session_header", header)
        world = await make_world(fake)
        single = [
            await add_experience(world, title=title, session_id="t-single")
            for title in ("single a", "single b")
        ]
        run_id = await start_run(world)
        before = len(fake.chat_requests())
        fake.queue_response(
            _call("skill_list", {}),
            _call(
                "experience_triage",
                {
                    "covered_ids": [],
                    "not_worth_ids": [],
                    "deferred_ids": [e.id for e in single],
                },
            ),
            _call("finish_memory_update"),
        )
        [entry] = (await _curate(world, run_id)).batches
        assert entry["stop_reason"] == "finish" and entry["experiences"] == 2
        calls = fake.chat_requests()[before:]
        assert len(calls) == 3
        assert [c.headers.get(header) for c in calls] == ["t-single"] * 3

        await add_experience(world, title="mixed a", session_id="t-first")
        await add_experience(world, title="mixed b", session_id="t-second")
        run_id = await start_run(world)
        before = len(fake.chat_requests())
        # The first finish lists the unaddressed experiences; the second ends it.
        fake.queue_response(
            _call("finish_memory_update"), _call("finish_memory_update")
        )
        [entry] = (await _curate(world, run_id)).batches
        assert entry["experiences"] == 2
        calls = fake.chat_requests()[before:]
        assert calls and all(header not in c.headers for c in calls)

    async def test_an_exception_inside_a_tool_is_recorded_by_class(
        self, fake, monkeypatch
    ):
        """A database error inside a tool reaches the model as a failed tool result
        like a bad argument does; the batch's metrics name its class (``execution``),
        apart from a tool that answers with an error text (``error_result``)."""
        from sqlalchemy.exc import DBAPIError

        async def broken_read(self, name_or_id):
            raise DBAPIError(
                "SELECT skill",
                {},
                Exception("can't subtract offset-naive and offset-aware datetimes"),
            )

        async def refusing_list(self, query="", offset=0):
            return "Error: the skill index is being rebuilt"

        world = await make_world(fake)
        # After the agent's tools are registered from the real functions: the agent
        # looks a tool's function up by name when it runs it.
        monkeypatch.setattr(mt, "skill_read", broken_read)
        monkeypatch.setattr(mt, "skill_list", refusing_list)
        await add_experience(world, title="tool exception")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_read", {"name_or_id": "some-skill"}),
            _call("skill_read", {"name_or_id": "other-skill"}),
            _call("skill_list", {}),
            _call("finish_memory_update"),  # lists the unaddressed experience
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["failed_tool_calls"] == 3
        failures = entry["tool_failures"]
        assert [
            (f["tool"], f["stage"], f["exception"], f["count"]) for f in failures
        ] == [
            ("skill_list", "error_result", None, 1),
            ("skill_read", "execution", "DBAPIError", 2),
        ]
        assert failures[0]["error"] == "Error: the skill index is being rebuilt"
        assert failures[1]["error"].startswith("DBAPIError: ")
        assert "offset-naive" in failures[1]["error"]
        assert len(failures[1]["error"]) <= 300
        # The run record keeps the entry, so the harness can read it.
        [stored] = (await _run_record(run_id)).curator_metrics
        assert stored["tool_failures"] == failures

    async def test_an_error_text_never_inherits_an_earlier_calls_exception(
        self, fake, monkeypatch
    ):
        """A tool call answered with an error text by a path that runs no tool (an
        agent that answers some calls itself) is ``error_result``, even right after
        a call whose tool raised: the exception belongs to that earlier call only."""
        from sqlalchemy.exc import DBAPIError

        from mirix.agent.agent import Agent

        async def broken_read(self, name_or_id):
            raise DBAPIError("SELECT skill", {}, Exception("connection reset"))

        run_tool = Agent.execute_tool_and_persist_state

        async def answering_list_itself(self, function_name, *args, **kwargs):
            if function_name == "skill_list":
                return "Error: skill_list is not available to this agent"
            return await run_tool(self, function_name, *args, **kwargs)

        world = await make_world(fake)
        monkeypatch.setattr(mt, "skill_read", broken_read)
        monkeypatch.setattr(
            Agent, "execute_tool_and_persist_state", answering_list_itself
        )
        await add_experience(world, title="stale exception")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_read", {"name_or_id": "some-skill"}),
            _call("skill_list", {}),
            _call("finish_memory_update"),  # lists the unaddressed experience
            _call("finish_memory_update"),
        )
        [entry] = (await _curate(world, run_id)).batches
        assert entry["failed_tool_calls"] == 2
        assert [
            (f["tool"], f["stage"], f["exception"]) for f in entry["tool_failures"]
        ] == [
            ("skill_list", "error_result", None),
            ("skill_read", "execution", "DBAPIError"),
        ]

    async def test_a_text_reply_does_not_end_the_batch(self, fake):
        world = await make_world(fake)
        exp = await add_experience(world, title="text reply")
        run_id = await start_run(world)
        fake.queue_response(
            ChatResponse(content="Let me look at the skills first."),
            _call(
                "experience_triage",
                {"covered_ids": [exp.id], "not_worth_ids": [], "deferred_ids": []},
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 3
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["steps"] == 3
        assert (await experiences(world))[exp.id].consumed_by == f"{run_id}:b0"

    async def test_a_turn_ended_without_finishing_is_reminded_not_ended(
        self, fake, monkeypatch
    ):
        from mirix.helpers import ToolRulesSolver

        # A tool rule that makes skill_list terminal: the chain ends after it
        # although the curator never called finish_memory_update.
        original = ToolRulesSolver.is_terminal_tool

        def terminal(self, name):
            return name == "skill_list" or original(self, name)

        monkeypatch.setattr(ToolRulesSolver, "is_terminal_tool", terminal)
        world = await make_world(fake)
        exp = await add_experience(world, title="reminded")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_list"),
            _call(
                "experience_triage",
                {"covered_ids": [exp.id], "not_worth_ids": [], "deferred_ids": []},
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 3
        assert (
            "without calling finish_memory_update()" in user_texts(requests[1].json)[-1]
        )
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish"
        assert (await experiences(world))[exp.id].consumed_by == f"{run_id}:b0"


# --------------------------------------------------------------------------- #
# The curator's guards against a weak model                                    #
# --------------------------------------------------------------------------- #


def _burst(*calls):
    """One answer with several tool calls ``(name, arguments)``."""
    return ChatResponse(
        tool_calls=[{"name": name, "arguments": args} for name, args in calls]
    )


def _triage_call(covered=(), not_worth=(), deferred=()):
    return _call(
        "experience_triage",
        {
            "covered_ids": list(covered),
            "not_worth_ids": list(not_worth),
            "deferred_ids": list(deferred),
        },
    )


def _tool_call_ids(request_json):
    return [
        m.get("tool_call_id")
        for m in request_json.get("messages", [])
        if m.get("role") == "tool"
    ]


class TestCuratorGuards:
    async def test_a_burst_runs_one_call_and_writes_nothing(self, fake):
        """One answer calling every
        tool once with the tool schemas' words as arguments. It used to run all
        nine (a placeholder create, a delete attempt, an empty triage, a finish
        voided by its failing siblings). Now its first call runs (a tool the
        curation step does not offer: refused), the second, another tool
        it does not offer, is answered the same way at once (never "call it
        again"), and the seven others are answered "Not run";
        nothing is written and the batch goes on."""
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"burst {i}") for i in range(2)]
        run_id = await start_run(world)
        placeholder = (
            '"episodic", "resource", "procedural", "knowledge_vault", "semantic", "all"'
        )
        fake.queue_response(
            _burst(
                (
                    "list_memory_within_timerange",
                    {
                        "memory_type": placeholder,
                        "start_time": '"%Y-%m-%d %H:%M:%S"',
                        "end_time": '"%Y-%m-%d %H:%M:%S"',
                    },
                ),
                (
                    "search_in_memory",
                    {
                        "memory_type": placeholder,
                        "query": "",
                        "search_field": "",
                        "search_method": "'bm25', 'embedding', 'hybrid'",
                    },
                ),
                (
                    "experience_triage",
                    {"covered_ids": [], "not_worth_ids": [], "deferred_ids": []},
                ),
                (
                    "skill_create",
                    {
                        "name": "kebab-case name",
                        "description": "one/two sentences",
                        "instructions": "detailed plain text",
                        "entry_type": "workflow/guide/script",
                        "triggers": [],
                        "examples": [],
                        "experience_ids": [],
                    },
                ),
                ("skill_read", {"name_or_id": "skill name or ID"}),
                ("finish_memory_update", {}),
                (
                    "skill_delete",
                    {
                        "skill_id": "skill ID",
                        "reason": "why delete",
                        "experience_ids": [],
                    },
                ),
                (
                    "skill_edit",
                    {
                        "skill_id": "skill ID",
                        "field": "name/description/instructions/entry_type/triggers/examples",
                        "old_text": "text to replace",
                        "new_text": "replacement text",
                        "value": "full new value",
                        "experience_ids": [],
                    },
                ),
                ("skill_list", {"query": "search terms", "offset": 0}),
            ),
            _triage_call(deferred=[e.id for e in exps]),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 3  # the burst did not end the batch
        # Every call of the burst was answered, in the answer's order.
        second = requests[1].json
        assistant = [m for m in second["messages"] if m.get("tool_calls")][-1]
        assert _tool_call_ids(second)[-9:] == [c["id"] for c in assistant["tool_calls"]]
        results = tool_results(second)[-9:]
        assert "not available in this step" in results[0]
        assert "not available in this step" in results[1]
        assert "Not run: " not in results[1]
        assert all(r.count("Not run: ") == 1 for r in results[2:]), results
        assert "call finish_memory_update() alone" in results[5]
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish"
        assert entry["skipped_tool_calls"] == 7 and entry["multi_call_turns"] == 1
        assert entry["failed_tool_calls"] == 2
        assert entry["writes"] == 0 and entry["deletes"] == 0
        assert entry["triage_calls"] == 1
        assert await skills(world) == {}
        assert outcome.changes == {"created": [], "edited": [], "deleted": []}

    async def test_a_valid_create_in_a_burst_does_not_run_blind(self, fake):
        """A burst of six calls, a valid create among them: only the first
        call runs; the next request answers all six tool_call_ids in order."""
        world = await make_world(fake)
        exp = await add_experience(world, title="blind burst")
        run_id = await start_run(world)
        create = {
            "name": "blind-create",
            "description": "Opens an item and checks it",
            "instructions": "Open the item, check it, then close it.",
            "entry_type": "guide",
            "triggers": [],
            "examples": [],
            "experience_ids": [{"id": exp.id, "quote": "Open the item, check it"}],
        }
        fake.queue_response(
            _burst(
                ("skill_list", {}),
                ("skill_read", {"name_or_id": "blind-create"}),
                ("skill_create", create),
                (
                    "skill_edit",
                    {
                        "skill_id": "proc-x",
                        "field": "description",
                        "value": "A new one.",
                    },
                ),
                (
                    "experience_triage",
                    {"covered_ids": [exp.id], "not_worth_ids": [], "deferred_ids": []},
                ),
                ("finish_memory_update", {}),
            ),
            _call("skill_create", create),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 3
        second = requests[1].json
        assistant = [m for m in second["messages"] if m.get("tool_calls")][-1]
        asked = [c["id"] for c in assistant["tool_calls"]]
        assert _tool_call_ids(second)[-6:] == asked
        results = tool_results(second)[-6:]
        assert "No skills found" in results[0]
        assert all("Not run: " in r for r in results[1:])
        [entry] = outcome.batches
        assert entry["skipped_tool_calls"] == 5 and entry["writes"] == 1
        # The create ran once, in its own turn.
        assert list(await skills(world)) == ["blind-create"]
        assert (await experiences(world))[exp.id].disposition == "used"

    async def test_every_request_offers_the_curator_tools_only(self, fake):
        """The curator used to offer the procedural agent's whole tool set
        (search_in_memory with the knowledge vault's secret_value field,
        list_memory_within_timerange). Every request of the batch offers
        exactly the curator's tools, also after the agent state is reloaded
        from the database (after every call)."""
        from mirix.constants import SKILL_CURATOR_TOOLS

        world = await make_world(fake)
        exp = await add_experience(world, title="tool set")
        run_id = await start_run(world)
        fake.queue_response(
            _call("skill_list", {}),
            _triage_call(covered=[exp.id]),
            _call("finish_memory_update"),
        )
        await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 3
        for request in requests:
            offered = {t["function"]["name"] for t in request.json["tools"]}
            assert offered == set(SKILL_CURATOR_TOOLS)

    async def test_a_tool_outside_the_curator_set_never_runs(self, fake, monkeypatch):
        from mirix.functions.function_sets import base

        async def must_not_run(*args, **kwargs):
            raise AssertionError("search_in_memory ran in a curation batch")

        monkeypatch.setattr(base, "search_in_memory", must_not_run)
        world = await make_world(fake)
        exp = await add_experience(world, title="no search")
        run_id = await start_run(world)
        fake.queue_response(
            _call(
                "search_in_memory",
                {
                    "memory_type": "knowledge_vault",
                    "query": "access credentials",
                    "search_field": "secret_value",
                    "search_method": "bm25",
                },
            ),
            _triage_call(not_worth=[exp.id]),
            _call("finish_memory_update"),
        )
        [entry] = (await _curate(world, run_id)).batches
        result = tool_results(fake.chat_requests()[1].json)[-1]
        assert "Tool search_in_memory is not available in this step" in result
        assert entry["failed_tool_calls"] == 1 and entry["stop_reason"] == "finish"
        assert [(f["tool"], f["stage"]) for f in entry["tool_failures"]] == [
            ("search_in_memory", "unknown_tool")
        ]

    async def test_a_procedural_agent_without_a_curator_tool_is_a_config_skip(
        self, fake, monkeypatch
    ):
        """A procedural agent that lacks experience_triage (even after the
        refresh) cannot curate: a configuration failure, nothing is counted."""
        world = await make_world(fake)
        exp = await add_experience(world, title="missing tool")
        [triage] = [t for t in world.procedural.tools if t.name == "experience_triage"]
        await world.server.agent_manager.detach_tool(
            agent_id=world.procedural.id, tool_id=triage.id, actor=world.actor
        )

        async def no_refresh(*args, **kwargs):
            return None

        monkeypatch.setattr(
            world.server.agent_manager,
            "update_agent_tools_and_system_prompts",
            no_refresh,
        )
        run_id = await start_run(world)
        outcome = await _curate(world, run_id)
        assert outcome.skipped == "curator_tools_missing" and outcome.batches == []
        assert fake.chat_requests() == []
        row = (await experiences(world))[exp.id]
        assert row.status == "pending" and row.curation_attempts == 0

    async def test_finish_first_lists_the_unaddressed_experiences(self, fake):
        """A finish with every experience neither cited nor triaged used to
        end the batch and default them all to deferred. The first finish is
        refused with the list; the model triages, and the next finish ends it."""
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"unaddressed {i}") for i in range(2)]
        run_id = await start_run(world)
        fake.queue_response(
            _call("finish_memory_update"),
            _triage_call(covered=[exps[0].id], deferred=[exps[1].id]),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        requests = fake.chat_requests()
        assert len(requests) == 3
        refusal = tool_results(requests[1].json)[-1]
        assert "Not finished" in refusal
        assert exps[0].id in refusal and exps[1].id in refusal
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["finish_refusals"] == 1
        assert entry["rejected_by_kind"] == {"unaddressed_experiences": 1}
        assert entry["dispositions"]["defaulted"] == 0
        rows = await experiences(world)
        assert rows[exps[0].id].disposition == "covered"
        assert rows[exps[1].id].disposition == "deferred"

    async def test_a_second_finish_ends_the_batch(self, fake):
        """The reminder is given once; a model that finishes again is not
        held (it never spins until the step cap)."""
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"twice {i}") for i in range(2)]
        run_id = await start_run(world)
        fake.queue_response(
            _call("finish_memory_update"), _call("finish_memory_update")
        )
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 2 and fake.pending_responses == 0
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["finish_refusals"] == 1
        assert entry["dispositions"]["defaulted"] == 2
        rows = await experiences(world)
        assert {rows[e.id].disposition for e in exps} == {"deferred"}

    async def test_a_retrieval_text_follow_up_is_free_and_counted(self, fake):
        """After an instructions edit, keeping the skill's description in
        step costs no edit budget (a batch of one experience makes both)."""
        world = await make_world(fake)
        target = await mt.skill_create(
            procedural_agent(world),
            name="merge-editions",
            description="Merge figures reported across editions",
            instructions="Read every edition, then prefer the latest revised figure.",
            entry_type="guide",
        )
        skill_id = target.split("ID: ")[1].split(",")[0]
        exp = await add_experience(world, title="widen")
        run_id = await start_run(world)
        fake.queue_response(
            _call(
                "skill_edit",
                {
                    "skill_id": skill_id,
                    "field": "instructions",
                    "old_text": "prefer the latest revised figure.",
                    "new_text": "prefer the latest revised figure, and note "
                    "which edition each figure comes from.",
                    "value": "",
                    "experience_ids": [
                        {
                            "id": exp.id,
                            "quote": "note which edition each figure comes from",
                        }
                    ],
                },
            ),
            _call(
                "skill_edit",
                {
                    "skill_id": skill_id,
                    "field": "description",
                    "old_text": "",
                    "new_text": "",
                    "value": "Merge figures reported across editions, naming "
                    "the edition of each",
                    "experience_ids": [],
                },
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        results = tool_results(fake.chat_requests()[1].json)
        assert "Retrieval finds a skill only by its name" in results[-1]
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish"
        # "writes" is the budget spent; the free follow-up is counted apart.
        assert entry["writes"] == 1 and entry["retrieval_followups"] == 1
        assert [w["field"] for w in entry["write_log"]] == [
            "instructions",
            "description",
        ]
        assert entry["rejected_by_kind"] == {}
        skill = (await skills(world))["merge-editions"]
        assert skill.description.endswith("naming the edition of each")

    async def test_a_deletion_only_edit_does_not_retire_what_it_cites(self, fake):
        """An edit that deletes a heading (new_text "") and cites two
        experiences used to count them as used although no lesson reached
        a skill. The edit goes through; its citations are not recorded, so the
        experience stays to be triaged (and defaults to deferred)."""
        world = await make_world(fake)
        target = await mt.skill_create(
            procedural_agent(world),
            name="merge-editions",
            description="Merge figures reported across editions",
            instructions=(
                "1. Read every edition.\n4. Merging across editions:\n"
                "Prefer the latest revised figure."
            ),
            entry_type="guide",
        )
        skill_id = target.split("ID: ")[1].split(",")[0]
        exp = await add_experience(world, title="heading")
        run_id = await start_run(world)
        fake.queue_response(
            _call(
                "skill_edit",
                {
                    "skill_id": skill_id,
                    "field": "instructions",
                    "old_text": "4. Merging across editions:",
                    "new_text": "",
                    "value": "",
                    "experience_ids": [exp.id],
                },
            ),
            _call("finish_memory_update"),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        answer = tool_results(fake.chat_requests()[1].json)[-1]
        assert "updated" in answer and "were NOT recorded" in answer
        [entry] = outcome.batches
        assert entry["writes"] == 1 and entry["citations_dropped"] == 1
        assert entry["citations"] == {}
        assert entry["write_log"] == [
            {
                "tool": "skill_edit",
                "skill_id": skill_id,
                "field": "instructions",
                "added_words": 0,
                "removed_words": 3,  # "Merging across editions:" ("4." is no word)
                "cited": [exp.id],
                "citations_recorded": False,
                "citations_unverified": {},
            }
        ]
        row = (await experiences(world))[exp.id]
        assert row.cited_skill_ids is None and row.disposition == "deferred"
        assert (
            "Merging across editions"
            not in (await skills(world))["merge-editions"].instructions
        )


# --------------------------------------------------------------------------- #
# The experience pool                                                          #
# --------------------------------------------------------------------------- #

_DIM = 16


def _vec(cos_to_e0: float, other_axis: int = 1) -> list:
    """A unit vector at cosine ``cos_to_e0`` to e0, the rest on ``other_axis``."""
    v = [0.0] * _DIM
    v[0] = cos_to_e0
    v[other_axis] = (max(0.0, 1.0 - cos_to_e0 * cos_to_e0)) ** 0.5
    return v


class _Embedder:
    """``embed_one`` for the pool: a vector per title (the first line of the
    pool text), counting the calls; ``fail`` titles raise."""

    def __init__(self, vectors, fail=()):
        self.vectors = dict(vectors)
        self.fail = set(fail)
        self.titles = []
        self.on_call = None

    async def __call__(self, text):
        title = text.split("\n\n", 1)[0]
        self.titles.append(title)
        if self.on_call is not None:
            await self.on_call(title)
        if title in self.fail:
            raise ValueError(f"the embedding provider refused {title!r}")
        return self.vectors[title]


async def _finish(agent, payload, *, actor, user):
    return _usage("finish")


async def _pool_row_state(experience_id):
    from sqlalchemy import select
    from sqlalchemy.orm import undefer

    from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            await db.execute(
                select(SkillExperienceORM)
                .options(undefer(SkillExperienceORM.pool_embedding))
                .where(SkillExperienceORM.id == experience_id)
            )
        ).scalar_one()


#: The key of the harness's embedding configuration (model and dimension).
POOL_MODEL = "text-embedding-3-small@1536"

POOL_ENTRY_KEYS = {
    "history_ids",
    "history_pushed",
    "history_tokens",
    "history_skipped",
    "history",
    "citations",
    "uncited_writes",
    "uncited_deletes",
    "triage_calls",
    "triaged",
    "dispositions",
    "history_cited",
    "history_covered",
    "history_not_worth",
    "history_ignored",
    "recurrences",
    "embedding_failures",
    "retired",
    "retired_ignored",
    "pool_active_after",
}


class TestExperiencePool:
    async def test_a_deferred_lesson_comes_back_recurs_and_is_used(self, fake):
        world = await make_world(fake)
        embed = _Embedder({"label field A": _vec(1.0), "label field B": _vec(0.9)})
        a = await add_experience(world, title="label field A")
        first = await start_run(world)
        await _curate(world, first, run_batch=_finish, embed_one=embed)
        row = (await experiences(world))[a.id]
        assert row.status == "consumed" and row.disposition == "deferred"
        assert row.pool_scope == world.scope
        [entry] = (await _run_record(first)).curator_metrics
        assert POOL_ENTRY_KEYS <= set(entry)
        assert entry["history_skipped"] == "empty_pool" and entry["history_ids"] == []
        assert embed.titles == []  # an empty pool costs no embedding call
        assert entry["dispositions"]["deferred"] == 1
        assert entry["dispositions"]["defaulted"] == 1
        assert entry["pool_active_after"] == 1

        b = await add_experience(world, title="label field B")
        second = await start_run(world)
        seen = {}

        async def cite_both(agent, payload, *, actor, user):
            seen["payload"] = payload
            answer = await mt.skill_create(
                agent,
                name="read-labelled-fields",
                description="Read the value next to a label, verbatim.",
                instructions="Find the label, then copy the value next to it verbatim.",
                entry_type="guide",
                experience_ids=[
                    {"id": i, "quote": "copy the value next to it verbatim"}
                    for i in (a.id, b.id)
                ],
            )
            assert "created successfully" in answer, answer
            seen["skill"] = answer.split("ID: ")[1].split(",")[0]
            return _usage("finish")

        await _curate(world, second, run_batch=cite_both, embed_one=embed)
        payload = seen["payload"]
        assert (
            "## Earlier lessons not yet turned into skills (1, for reference)"
            in payload
        )
        assert f'<earlier_lesson id="{a.id}">' in payload
        # A recurred in this batch (cosine 0.9): seen twice already.
        assert "- seen 2 time(s); shown in 0 earlier batch(es)" in payload
        assert payload.index(f'<experience id="{b.id}">') < payload.index(
            "<earlier_lesson"
        )
        assert "the user confirmed it" not in payload.split("<earlier_lesson", 1)[1]
        assert sorted(embed.titles) == ["label field A", "label field B"]
        skill = seen["skill"]
        rows = await experiences(world)
        assert rows[b.id].disposition == "used" and rows[b.id].cited_skill_ids == [
            skill
        ]
        assert rows[a.id].cited_skill_ids == [skill]
        assert rows[a.id].pool_retire_reason == "used"
        assert rows[a.id].recurrence_count == 2 and rows[a.id].pushed_count == 1
        assert rows[a.id].pool_embedding_model == POOL_MODEL
        [entry] = (await _run_record(second)).curator_metrics
        assert POOL_ENTRY_KEYS <= set(entry)
        assert entry["history_ids"] == [a.id] and entry["history_pushed"] == 1
        assert entry["history_skipped"] is None and entry["history_tokens"] > 0
        assert entry["history"]["recurred_ids"] == [a.id]
        assert entry["history"]["max_cosines"][0]["id"] == a.id
        assert entry["history"]["max_cosines"][0]["max_cosine"] == 0.9
        assert entry["citations"] == {a.id: [skill], b.id: [skill]}
        assert entry["uncited_writes"] == 0
        assert entry["recurrences"] == 1 and entry["history_cited"] == 1
        assert entry["retired"] == {
            "used": 1,
            "covered": 0,
            "ignored": 0,
            "merged": 0,
            "unembeddable": 0,
        }
        assert entry["dispositions"]["used"] == 1
        assert entry["pool_active_after"] == 0
        assert (
            entry["payload_tokens"] + entry["history_tokens"]
            <= entry["capacity_tokens"]
        )

    async def test_the_pool_is_isolated_by_user_agent_organization_and_scope(
        self, fake
    ):
        from mirix.orm.agent import Agent as AgentORM
        from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM
        from mirix.orm.user import User as UserORM
        from mirix.server.server import db_context

        world = await make_world(fake)
        other_world = await make_world(fake)
        embed = _Embedder({"mine": _vec(1.0), "new": _vec(1.0)})
        stranger = f"pool-stranger-{world.org}"
        other_agent = f"agent-pool-other-{world.org}"
        async with db_context() as db:
            db.add(
                UserORM(
                    id=stranger,
                    name="s",
                    organization_id=world.org,
                    status="active",
                    timezone="UTC",
                )
            )
            db.add(AgentORM(id=other_agent, organization_id=world.org))
            await db.commit()
        model = POOL_MODEL  # comparable vectors: only the scope keeps them out
        foreign_ids = []
        async with db_context() as db:
            for agent_id, user_id, org_id, scope in (
                (world.meta.id, stranger, world.org, world.scope),  # another user
                (
                    other_agent,
                    world.user.id,
                    world.org,
                    world.scope,
                ),  # another meta agent
                (
                    world.meta.id,
                    world.user.id,
                    world.org,
                    "another-scope",
                ),  # another scope
                (
                    other_world.meta.id,
                    other_world.user.id,
                    other_world.org,
                    world.scope,
                ),
            ):
                row = SkillExperienceORM(
                    agent_id=agent_id,
                    user_id=user_id,
                    organization_id=org_id,
                    session_id="s",
                    experience_type="worth_learning",
                    title="foreign",
                    content="c",
                    status="consumed",
                    consumed_by="r:b0",
                    disposition="deferred",
                    pool_scope=scope,
                    pool_embedding=_vec(1.0),
                    pool_embedding_model=model,
                )
                db.add(row)
                await db.flush()
                foreign_ids.append(row.id)
            await db.commit()
        mine = await add_experience(world, title="mine")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        new = await add_experience(world, title="new")
        seen = {}

        async def try_foreign(agent, payload, *, actor, user):
            seen["payload"] = payload
            for foreign in foreign_ids:
                answer = await mt.skill_create(
                    agent,
                    name=f"steal-{foreign[-6:]}",
                    description="Opens an item and checks it",
                    instructions="Open the item, check it, then close it.",
                    entry_type="guide",
                    experience_ids=[foreign],
                )
                assert "which this batch did not show" in answer
            answer = await mt.experience_triage(
                agent, covered_ids=[foreign_ids[0]], not_worth_ids=[], deferred_ids=[]
            )
            assert "was not shown in this batch" in answer
            return _usage("finish")

        run_id = await start_run(world)
        await _curate(world, run_id, run_batch=try_foreign, embed_one=embed)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["history_ids"] == [mine.id]
        for foreign in foreign_ids:
            assert foreign not in seen["payload"]
            state = await _pool_row_state(foreign)
            assert state.pushed_count == 0 and state.recurrence_count == 1
            assert state.cited_skill_ids is None and state.pool_retired_at is None
        assert entry["rejected_by_kind"] == {"invalid_arguments": len(foreign_ids) + 1}
        assert (await experiences(world))[new.id].disposition == "deferred"

    async def test_an_embedding_failure_shows_nothing_and_the_batch_runs(self, fake):
        world = await make_world(fake)
        embed = _Embedder({"old": _vec(1.0), "fresh": _vec(1.0)}, fail={"fresh"})
        old = await add_experience(world, title="old")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        fresh = await add_experience(world, title="fresh")
        run_id = await start_run(world)
        outcome = await _curate(world, run_id, run_batch=_finish, embed_one=embed)
        [entry] = outcome.batches
        assert entry["history_skipped"] == "embedding_failed"
        assert entry["history"]["embedding_error_kind"] == "other"
        assert (
            "the embedding provider refused 'fresh'"
            in entry["history"]["embedding_error"]
        )
        assert entry["history_ids"] == [] and entry["recurrences"] == 0
        rows = await experiences(world)
        assert rows[fresh.id].status == "consumed"  # the batch ran as usual
        assert rows[old.id].pushed_count == 0 and rows[old.id].recurrence_count == 1
        assert rows[old.id].ignored_pushes == 0
        # The vector that did succeed is kept for the next batch.
        assert rows[old.id].pool_embedding_model == POOL_MODEL
        assert rows[fresh.id].pool_embedding_model is None

    async def test_a_returned_batch_leaves_the_pool_and_a_later_finish_applies_it_once(
        self, fake
    ):
        world = await make_world(fake)
        embed = _Embedder({"kept": _vec(1.0), "retry": _vec(0.85)})
        kept = await add_experience(world, title="kept")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        retry = await add_experience(world, title="retry")

        async def step_cap(agent, payload, *, actor, user):
            return _usage("step_cap")

        returned_run = await start_run(world)
        await _curate(world, returned_run, run_batch=step_cap, embed_one=embed)
        [entry] = (await _run_record(returned_run)).curator_metrics
        assert entry["outcome"] == "return" and entry["history_ids"] == [kept.id]
        assert entry["recurrences"] == 0
        row = (await experiences(world))[kept.id]
        assert (row.pushed_count, row.recurrence_count, row.ignored_pushes) == (0, 1, 0)
        calls = len(embed.titles)
        finished_run = await start_run(world)
        await _curate(world, finished_run, run_batch=_finish, embed_one=embed)
        assert len(embed.titles) == calls  # both vectors were stored the first time
        [entry] = (await _run_record(finished_run)).curator_metrics
        assert entry["history_ids"] == [kept.id]  # the same plan again
        row = (await experiences(world))[kept.id]
        assert (row.pushed_count, row.recurrence_count, row.ignored_pushes) == (1, 2, 1)
        assert (await experiences(world))[retry.id].disposition == "deferred"

    async def test_a_closed_out_experience_never_comes_back(self, fake):
        world = await make_world(fake)
        embed = _Embedder({"hopeless": _vec(1.0), "similar": _vec(0.95)})
        hopeless = await add_experience(world, title="hopeless")

        async def context_limit(agent, payload, *, actor, user):
            return _usage("context_limit")

        for _ in range(3):
            await _curate(
                world, await start_run(world), run_batch=context_limit, embed_one=embed
            )
        row = (await experiences(world))[hopeless.id]
        assert row.disposition == "unaddressed" and row.pool_scope is None
        await add_experience(world, title="similar")
        run_id = await start_run(world)
        await _curate(world, run_id, run_batch=_finish, embed_one=embed)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["history_skipped"] == "empty_pool"

    async def test_a_full_batch_has_no_room_for_earlier_lessons(self, fake):
        from mirix.services.context_budget import get_token_counter

        world = await make_world(fake)
        embed = _Embedder({"small": _vec(1.0), "LESSON-7 big": _vec(1.0)})
        await add_experience(world, title="small")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        # Larger than a batch: cut to fill it (oversize), leaving no room.
        await add_experience(world, title="LESSON-7 big", content=words(30_000, seed=7))
        payloads = []

        async def capture(agent, payload, *, actor, user):
            payloads.append(payload)
            return _usage("finish")

        run_id = await start_run(world)
        await _curate(world, run_id, run_batch=capture, embed_one=embed)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["oversize"]
        assert entry["history_skipped"] == "no_room"
        assert entry["history"]["dropped_for_room"] == 1
        assert entry["history"]["recurred_ids"] and entry["recurrences"] == 1
        assert "<earlier_lesson" not in payloads[0]
        assert get_token_counter().count_text(payloads[0]) <= entry["capacity_tokens"]

    async def test_a_lesson_ignored_three_times_retires(self, fake):
        world = await make_world(fake)
        vectors = {"base": _vec(1.0)}
        for i in range(2, 6):
            vectors[f"near {i}"] = _vec(0.7, other_axis=i)  # 0.49 to each other
        embed = _Embedder(vectors)
        base = await add_experience(world, title="base")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        entries = []
        for i in range(2, 6):
            await add_experience(world, title=f"near {i}")
            run_id = await start_run(world)
            await _curate(world, run_id, run_batch=_finish, embed_one=embed)
            [entry] = (await _run_record(run_id)).curator_metrics
            entries.append(entry)
        assert [e["history_ids"] for e in entries[:3]] == [[base.id]] * 3
        assert [e["history_ignored"] for e in entries[:3]] == [1, 1, 1]
        assert entries[2]["retired_ignored"] == 1
        row = (await experiences(world))[base.id]
        assert row.pool_retire_reason == "ignored" and row.ignored_pushes == 3
        assert row.pushed_count == 3 and row.recurrence_count == 1
        # Retired: the fourth similar batch no longer sees it.
        assert entries[3]["history_ids"] == []
        assert entries[3]["history_skipped"] == "no_candidates"

    async def test_triage_decides_the_dispositions(self, fake):
        world = await make_world(fake)
        embed = _Embedder({})
        exps = [await add_experience(world, title=f"triaged {i}") for i in range(4)]
        run_id = await start_run(world)

        async def triage(agent, payload, *, actor, user):
            answer = await mt.experience_triage(
                agent,
                covered_ids=[exps[0].id],
                not_worth_ids=[exps[1].id],
                deferred_ids=[exps[2].id],
            )
            assert exps[3].id in answer  # still untriaged
            return _usage("finish")

        await _curate(world, run_id, run_batch=triage, embed_one=embed)
        rows = await experiences(world)
        assert [rows[e.id].disposition for e in exps] == [
            "covered",
            "not_worth",
            "deferred",
            "deferred",
        ]
        assert rows[exps[1].id].ignored_pushes == 1
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["triage_calls"] == 1
        assert entry["triaged"] == {"covered": 1, "not_worth": 1, "deferred": 1}
        assert entry["dispositions"] == {
            "used": 0,
            "covered": 1,
            "not_worth": 1,
            "deferred": 2,
            "defaulted": 1,
            "unaddressed": 0,
        }
        assert entry["pool_active_after"] == 3

    async def test_switched_off_the_pool_shows_nothing_but_dispositions_are_kept(
        self, fake, monkeypatch
    ):
        from mirix import constants

        world = await make_world(fake)
        embed = _Embedder({"one": _vec(1.0), "two": _vec(1.0)})
        await add_experience(world, title="one")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        monkeypatch.setattr(constants, "SKILL_EXPERIENCE_POOL_ENABLED", False)
        two = await add_experience(world, title="two")
        run_id = await start_run(world)
        await _curate(world, run_id, run_batch=_finish, embed_one=embed)
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["history_skipped"] == "disabled" and embed.titles == []
        assert entry["dispositions"]["deferred"] == 1
        assert (await experiences(world))[two.id].disposition == "deferred"

    async def test_a_lost_lease_while_storing_vectors_writes_nothing(self, fake):
        from mirix.services.skill_learning_run_manager import (
            LearningRunNotActive,
            SkillLearningRunManager,
        )

        world = await make_world(fake)
        embed = _Embedder({"stored": _vec(1.0), "late": _vec(1.0)})
        stored = await add_experience(world, title="stored")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        late = await add_experience(world, title="late")
        run_id = await start_run(world)

        async def lose_lease(title):
            await SkillLearningRunManager().fail(run_id, "lease_expired")

        embed.on_call = lose_lease
        with pytest.raises(LearningRunNotActive):
            await _curate(world, run_id, run_batch=_finish, embed_one=embed)
        rows = await experiences(world)
        assert rows[stored.id].pool_embedding_model is None
        assert rows[late.id].pool_embedding_model is None
        assert rows[late.id].status == "pending"
        assert (await _run_record(run_id)).curator_metrics in (None, [])

    async def test_a_lesson_the_embedding_model_keeps_refusing_costs_only_itself(
        self, fake
    ):
        """A pool experience whose embedding always fails used to turn
        off every earlier lesson of its owner, batch after batch. Now it is left
        out of each batch's ranking while the rest are shown, and it retires
        (unembeddable) after three finished batches."""
        world = await make_world(fake)
        vectors = {"good": _vec(1.0), "poison": _vec(1.0)}
        for i in range(4):
            vectors[f"later {i}"] = _vec(0.9, other_axis=2 + i)
        embed = _Embedder(vectors)
        good = await add_experience(world, title="good")
        poison = await add_experience(world, title="poison")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        embed.fail = {"poison"}
        entries = []
        for i in range(4):
            await add_experience(world, title=f"later {i}")
            run_id = await start_run(world)
            await _curate(world, run_id, run_batch=_finish, embed_one=embed)
            [entry] = (await _run_record(run_id)).curator_metrics
            entries.append(entry)
        for entry in entries[:3]:
            assert entry["history_skipped"] is None
            assert (
                good.id in entry["history_ids"]
                and poison.id not in entry["history_ids"]
            )
            assert entry["history"]["embedding_failed_ids"] == [poison.id]
            assert entry["history"]["embedding_error_kind"] == "other"
            assert entry["embedding_failures"] == 1
        assert entries[2]["retired"]["unembeddable"] == 1
        row = await _pool_row_state(poison.id)
        assert row.pool_retire_reason == "unembeddable"
        assert row.pool_embedding_failures == 3
        # Retired: the fourth batch no longer tries it.
        assert entries[3]["history"]["embedding_failed_ids"] == []
        assert entries[3]["history"]["embedding_error"] is None
        # Its own batch had an empty pool (no embedding call): three tries, then
        # retired.
        assert embed.titles.count("poison") == 3

    async def test_a_recurring_lesson_stays_one_pool_row(self, fake):
        """A new experience that recurred a lesson still in the pool
        is merged into it, so a third sighting shows one lesson seen 3 times,
        not two rows splitting the count."""
        world = await make_world(fake)
        embed = _Embedder(
            {"lesson A": _vec(1.0), "lesson A2": _vec(0.95), "lesson A3": _vec(0.93)}
        )
        a = await add_experience(world, title="lesson A")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        a2 = await add_experience(world, title="lesson A2")
        second = await start_run(world)
        await _curate(world, second, run_batch=_finish, embed_one=embed)
        [entry] = (await _run_record(second)).curator_metrics
        assert entry["history_ids"] == [a.id] and entry["recurrences"] == 1
        assert entry["retired"]["merged"] == 1 and entry["pool_active_after"] == 1
        rows = await experiences(world)
        assert rows[a2.id].disposition == "deferred"
        assert rows[a2.id].pool_retire_reason == "merged"
        await add_experience(world, title="lesson A3")
        payloads = []

        async def capture(agent, payload, *, actor, user):
            payloads.append(payload)
            return _usage("finish")

        third = await start_run(world)
        await _curate(world, third, run_batch=capture, embed_one=embed)
        [entry] = (await _run_record(third)).curator_metrics
        assert entry["history_ids"] == [a.id]
        assert payloads[0].count("<earlier_lesson id=") == 1
        assert "- seen 3 time(s); shown in 1 earlier batch(es)" in payloads[0]
        assert (await experiences(world))[a.id].recurrence_count == 3

    async def test_a_database_failure_while_planning_is_recorded_then_fails_the_run(
        self, fake, monkeypatch
    ):
        """The batch whose earlier lessons could not be planned has
        its curator_metrics entry (kept, error_kind db, error_stage history)
        before the database failure fails the run; nothing ran or changed."""
        from sqlalchemy.exc import OperationalError

        from mirix.services.skill_experience_manager import SkillExperienceManager

        world = await make_world(fake)
        embed = _Embedder({"kept": _vec(1.0), "next": _vec(1.0)})
        await add_experience(world, title="kept")
        await _curate(world, await start_run(world), run_batch=_finish, embed_one=embed)
        nxt = await add_experience(world, title="next")

        async def broken(self, **kwargs):
            raise OperationalError(
                "SELECT", {}, Exception("server closed the connection")
            )

        monkeypatch.setattr(SkillExperienceManager, "list_active_pool", broken)
        ran = []

        async def must_not_run(agent, payload, *, actor, user):
            ran.append(payload)
            return _usage("finish")

        run_id = await start_run(world)
        with pytest.raises(OperationalError):
            await _curate(world, run_id, run_batch=must_not_run, embed_one=embed)
        assert ran == []
        [entry] = (await _run_record(run_id)).curator_metrics
        assert entry["status"] == "settled" and entry["outcome"] == "keep"
        assert entry["error_kind"] == "db" and entry["error_stage"] == "history"
        assert "server closed the connection" in entry["error"]
        assert entry["experience_ids"] == [nxt.id]
        row = (await experiences(world))[nxt.id]
        assert row.status == "pending" and row.curation_attempts == 0

    async def test_the_procedural_agent_has_the_triage_tool(self, fake):
        world = await make_world(fake)
        names = {tool.name for tool in world.procedural.tools}
        assert {"experience_triage", "skill_edit", "finish_memory_update"} <= names
        meta_names = {tool.name for tool in world.meta.tools or []}
        assert "experience_triage" not in meta_names


class TestPoolThroughAgentStep:
    async def test_triage_and_finish_in_one_turn(self, fake):
        """A curation turn runs one call. The triage runs; the finish of the
        same turn is answered "Not run" (the model has not read the triage's
        answer yet), and the finish of the next turn ends the batch."""
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"one turn {i}") for i in range(2)]
        run_id = await start_run(world)
        fake.queue_response(
            ChatResponse(
                tool_calls=[
                    {
                        "name": "experience_triage",
                        "arguments": {
                            "covered_ids": [exps[0].id],
                            "not_worth_ids": [exps[1].id],
                            "deferred_ids": [],
                        },
                    },
                    {"name": "finish_memory_update", "arguments": {}},
                ]
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        assert fake.pending_responses == 0 and len(fake.chat_requests()) == 2
        results = tool_results(fake.chat_requests()[1].json)
        assert "Triage recorded" in results[-2]
        assert "Not run: call finish_memory_update() alone" in results[-1]
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["triage_calls"] == 1
        assert entry["skipped_tool_calls"] == 1 and entry["multi_call_turns"] == 1
        assert entry["finish_refusals"] == 0
        rows = await experiences(world)
        assert rows[exps[0].id].disposition == "covered"
        assert rows[exps[1].id].disposition == "not_worth"
        request = fake.chat_requests()[0].json
        tool_names = {t["function"]["name"] for t in request["tools"]}
        assert "experience_triage" in tool_names

    async def test_a_finish_paired_again_is_not_held_until_the_step_cap(self, fake):
        """A model that pairs its finish with another call again, after being
        told once to call it alone (e.g. triage + finish in one turn), would
        never get its finish run: the batch would spin to the
        step cap and be returned. The second paired turn runs the other call,
        then the finish."""
        world = await make_world(fake)
        exps = [await add_experience(world, title=f"paired {i}") for i in range(2)]
        run_id = await start_run(world)

        def paired(covered, not_worth):
            return _burst(
                (
                    "experience_triage",
                    {
                        "covered_ids": covered,
                        "not_worth_ids": not_worth,
                        "deferred_ids": [],
                    },
                ),
                ("finish_memory_update", {}),
            )

        fake.queue_response(paired([exps[0].id], []), paired([], [exps[1].id]))
        outcome = await _curate(world, run_id)
        assert len(fake.chat_requests()) == 2 and fake.pending_responses == 0
        [entry] = outcome.batches
        assert entry["stop_reason"] == "finish" and entry["triage_calls"] == 2
        assert entry["skipped_tool_calls"] == 1 and entry["multi_call_turns"] == 2
        rows = await experiences(world)
        assert rows[exps[0].id].disposition == "covered"
        assert rows[exps[1].id].disposition == "not_worth"

    async def test_a_skill_edit_with_empty_string_fillers_goes_through(self, fake):
        world = await make_world(fake)
        target = await mt.skill_create(
            procedural_agent(world),
            name="filled-in",
            description="Opens an item and checks it",
            instructions="Open it. Check it.",
            entry_type="guide",
        )
        skill_id = target.split("ID: ")[1].split(",")[0]
        exp = await add_experience(world, title="patch it")
        run_id = await start_run(world)
        fake.queue_response(
            _call(
                "skill_edit",
                {
                    "skill_id": skill_id,
                    "field": "instructions",
                    "old_text": "Check it.",
                    "new_text": "Check it twice.",
                    "value": "",
                    "experience_ids": [{"id": exp.id, "quote": "Check it twice"}],
                },
            ),
            _call(
                "skill_edit",
                {
                    "skill_id": skill_id,
                    # Not the retrieval text: a follow-up change of that after
                    # an instructions edit would cost no budget.
                    "field": "entry_type",
                    "old_text": "",
                    "new_text": "",
                    "value": "workflow",
                    "experience_ids": [""],
                },
            ),
            _call("finish_memory_update"),
        )
        outcome = await _curate(world, run_id)
        [entry] = outcome.batches
        assert entry["writes"] == 1 and entry["rejected_tool_calls"] == 1
        assert entry["rejected_by_kind"] == {"budget_exhausted": 1}  # budget 1
        results = [tool_results(r.json)[-1] for r in fake.chat_requests()[1:3]]
        assert "updated" in results[0] and f"Cited: {exp.id}." in results[0]
        assert "Edit budget exhausted" in results[1]
        skill = (await skills(world))["filled-in"]
        assert skill.instructions == "Open it. Check it twice."
        row = (await experiences(world))[exp.id]
        assert row.disposition == "used" and row.cited_skill_ids == [skill_id]


class TestPrompt:
    PROMPT = (
        REPO / "mirix" / "prompts" / "system" / "base" / "procedural_memory_agent.txt"
    )

    def test_it_explains_budget_deletes_and_the_three_stop_conditions(self):
        text = self.PROMPT.read_text()
        assert "as many create/edit operations as it has experiences" in text
        assert "At most one skill may be deleted per learning run" in text
        assert "every deletion must state its reason" in text
        assert "skill_delete(skill_id, reason, experience_ids)" in text
        assert "finish_memory_update()" in text
        assert "The context limit" in text and "The step limit" in text
        assert "skill_list(query?, offset?)" in text

    def test_it_explains_citations_triage_and_earlier_lessons(self):
        text = self.PROMPT.read_text()
        assert "experience_ids" in text
        assert "experience_triage(covered_ids, not_worth_ids, deferred_ids)" in text
        assert "Earlier lessons not yet turned into skills" in text
        assert '"Seen N times"' in text
        assert "earlier lessons do not add to it" in text
        assert "An experience you cite counts as used." in text
        # The prompt never says how to get past finish's refusal.
        assert "counts as deferred" not in text
        assert "refuses (once)" not in text and "calling it again" not in text
        # The quote's minimum and what an unquoted citation leaves.
        assert "a phrase of at least 12 letters or digits" in text
        assert "stays pending" in text
        assert "pass an empty string or an empty list for one you do not use" in text
        assert "give `old_text` and `new_text` and leave `value` empty" in text

    def test_the_old_rules_are_gone(self):
        text = self.PROMPT.read_text()
        for gone in (
            "Keep edits small",
            "Delta, never wholesale",
            "No deletes in this mode",
            "names the skill actively harmful",
            "too large",
            "Records-Based",
            # A lesson that keeps coming up is shown again.
            "a lesson ignored three times is not shown again",
        ):
            assert gone not in text, gone

    def test_it_states_the_pool_retirement_rule_the_code_applies(self):
        """The ignore count starts afresh when a lesson comes up again."""
        text = self.PROMPT.read_text()
        assert (
            "a lesson ignored three times without coming up again leaves the pool"
            in text
        )
        assert "starts its count afresh and may be shown again" in text
        assert "(as not worth, twice)" in text

    def test_it_asks_for_compact_general_skills(self):
        """Generalize, keep skills compact, condense or split."""
        text = self.PROMPT.read_text()
        assert "Write the general rule a case teaches, not the case" in text
        assert "never write that an answer was correct unless" in text
        assert "Keep skills compact" in text
        assert "over its target size, condense it before adding to it" in text
        assert "split out a separate kind of task" in text
        # Nothing else asks for long or all-covering skills.
        assert "Comprehensive" not in text
        assert "Complete but compact plain text" in text
        assert "one general skill" not in text
        assert "kept within that skill's size target" in text
        from mirix.services.skill_experience_curator import build_history_section

        header = build_history_section(["<earlier_lesson/>"])
        assert "one general skill" not in header
        assert "kept within that skill's size target" in header

    def test_a_general_numeric_rule_is_kept(self):
        """The generalize rule drops one task's values, not a number that is
        part of the rule itself (a limit, a unit, a setting)."""
        text = self.PROMPT.read_text()
        rule = text.split("- Write the general rule a case teaches", 1)[1].split("\n")[
            0
        ]
        assert "of one particular task" in rule
        assert "keep a value that is part of the general rule itself" in rule

    def test_it_says_a_changed_number_is_a_new_word(self):
        text = self.PROMPT.read_text()
        cite = text.split("**Cite your sources.**", 1)[1].split("\n")[0]
        assert "adds no new words" in cite
        assert "a changed number counts as a new word" in cite

    def test_it_states_what_retrieval_matches(self):
        """The retrieval text is the whole index of a skill."""
        text = self.PROMPT.read_text()
        assert (
            "Retrieval matches a skill only by its name, description and triggers "
            "(never its instructions)"
        ) in text
        assert "keep them in step with the instructions" in text
        assert "costs no budget" in text  # the follow-up of an instructions edit

    def test_it_asks_for_self_contained_skills(self):
        """A retrieved skill may be shown without the skill it names."""
        text = self.PROMPT.read_text()
        assert "Make each skill self-contained: do not refer to another skill" in text

    def test_it_has_no_benchmark_specific_wording(self):
        text = self.PROMPT.read_text().lower()
        for word in (
            "alfworld",
            "officeqa",
            "docvqa",
            "benchmark",
            "webshop",
            "locomo",
            "household",
            "treasury",
            "skillopt",
            "gpt-",
            "glm",
        ):
            assert word not in text, word
        # "eval" as a word (not inside "retrieval").
        assert not re.search(r"\beval(uation|s)?\b", text)
