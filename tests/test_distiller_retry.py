"""Distiller failures and retries.

One case per distiller row of the retry table, end to end: a claimed learning
run executed by ``AutoDreamManager`` on a hermetic SQLite database, with the
production distiller and the production OpenAI-compatible client talking to
the offline fake model server (``tests/fakes``). Nothing is paid, nothing
leaves localhost.

    infrastructure (5xx, 429, dropped connection) -> backoff, never counted
    finish_reason=length                         -> halve + re-chunk (<= 3), then 1 failure
    ContextBudgetExceeded                        -> the same
    unparseable output                           -> repair prompt in the dialogue (<= 2), then 1 failure
    timeout                                      -> one retry, then 1 failure
    database                                     -> transaction retried (<= 5), then the run fails

plus: 3 counted failures retire the session for good; a backed-off session is
not counted; an explicit ``[]`` is "nothing to learn" but garbage is not; every
retry lands in the run's ``retry_events``; the per-session metrics and the
run's ``llm_usage``; a session reopened DURING the distill call keeps its
post-watermark rows undistilled.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import threading
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from fakes.fake_llm_server import ChatResponse, Fault
from mirix import constants
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.conversation_message import ConversationMessage as MessageORM
from mirix.orm.conversation_session import ConversationSession as SessionORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.skill_experience import SkillExperience as ExperienceORM
from mirix.orm.user import User as UserORM
from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.services import context_budget, retry_policy, skill_learning_trigger
from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
from mirix.services.context_budget import ModelLimits
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_session_manager import (
    ConversationSessionManager,
    as_utc,
)
from mirix.services.retry_policy import ErrorKind, RetryPolicy
from mirix.services.session_experience_distiller import SessionExperienceDistiller
from mirix.services.skill_learning_run_manager import SkillLearningRunManager

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _experience(title, kind="worth_learning"):
    return {
        "experience_type": kind,
        "title": title,
        "content": f"do {title}",
        "importance": 0.5,
        "credibility": 0.5,
        "evidence": {"quote": "ok", "signal_type": "inferred"},
    }


def _user_text(request_json):
    """The (first) user message of a recorded chat request, as text."""
    for message in request_json["messages"]:
        if message["role"] == "user":
            content = message["content"]
            if isinstance(content, str):
                return content
            return "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
    return ""


def _chunk_label(request_json):
    text = _user_text(request_json)
    if "covering rounds" not in text:
        return "whole"
    return text.split("\nchunk ", 1)[1].split(" covering", 1)[0]


def per_chunk_answer(request_json):
    """A valid answer naming the chunk it was given."""
    return json.dumps([_experience(f"lesson {_chunk_label(request_json)}")])


# --------------------------------------------------------------------------- #
# Hermetic database, scopes, and the run environment                           #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("distiller_retry") / "test.db"
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
async def org_id(sm):
    org = f"org-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.commit()
    return org


@pytest.fixture
def llm_config(fake_llm_server):
    return LLMConfig(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=fake_llm_server.base_url,
        context_window=200_000,
        max_output_tokens=16_000,
        api_key="fake-key",
    )


@pytest_asyncio.fixture(loop_scope="module")
async def scope(sm, org_id, llm_config):
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org_id,
                name=client_id,
                status="active",
                write_scope="test",
                read_scopes=["test"],
            )
        )
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        agent = AgentORM(
            id=agent_id,
            organization_id=org_id,
            agent_type="meta_memory_agent",
            name="meta",
        )
        agent._created_by_id = client_id
        db.add(agent)
        await db.commit()
    return SimpleNamespace(
        org=org_id,
        agent=agent_id,
        user=PydanticUser(
            id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
        ),
        actor=PydanticClient(
            id=client_id,
            organization_id=org_id,
            name="test client",
            write_scope="test",
            read_scopes=["test"],
        ),
        # The run uses the meta agent's LLM config: the fake server.
        meta=SimpleNamespace(id=agent_id, llm_config=llm_config),
    )


@pytest.fixture
def env(sm, monkeypatch):
    """Route AutoDreamManager's managers to the hermetic database; stub only
    what is not the distiller (the curator, the leftover drain, the dream
    checkpoint, background scheduling). The distiller and the LLM client are
    the production ones."""
    import mirix.server.server as server_module
    from mirix.services.auto_dream_manager import AutoDreamManager

    monkeypatch.setattr(server_module, "db_context", sm)
    # Talk to the fake directly (a system proxy would turn drops into 502s).
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    # One transport attempt per call unless a test says otherwise, so each
    # injected fault reaches the application layer.
    monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
    # Database retries without real waiting.
    monkeypatch.setitem(
        retry_policy.RETRY_POLICIES,
        ErrorKind.DB,
        RetryPolicy(
            ErrorKind.DB, retry_policy.STRATEGY_RETRY_TRANSACTION, 5, False, False
        ),
    )
    context_budget._reset_tokenizer_state_for_tests()  # deterministic counts

    state = SimpleNamespace(distiller_kwargs={})

    async def no_curation(self, **kwargs):
        from mirix.services.skill_experience_curator import CuratorOutcome

        return CuratorOutcome()

    async def no_checkpoint(self, *args, **kwargs):
        return None

    def make_distiller(self, *, llm_config, conversation_manager):
        return SessionExperienceDistiller(
            llm_config=llm_config,
            conversation_manager=conversation_manager,
            **state.distiller_kwargs,
        )

    monkeypatch.setattr(AutoDreamManager, "_make_distiller", make_distiller)
    monkeypatch.setattr(AutoDreamManager, "_curate", no_curation)
    monkeypatch.setattr(AutoDreamManager, "write_checkpoint", no_checkpoint)
    monkeypatch.setattr(
        skill_learning_trigger, "schedule_learning_run", lambda *a, **k: None
    )
    yield state
    context_budget._reset_tokenizer_state_for_tests()


class Clock:
    def __init__(self):
        self.now = dt.datetime.now(dt.timezone.utc)

    def __call__(self):
        return self.now


def managers(sm, clock=None):
    return SimpleNamespace(
        messages=ConversationMessageManager(session_maker=sm, clock=clock),
        sessions=ConversationSessionManager(session_maker=sm, clock=clock),
        trigger=AgentTriggerStateManager(session_maker=sm, clock=clock),
        runs=SkillLearningRunManager(session_maker=sm, clock=clock),
    )


async def write_session(m, scope, sid, *, rounds=3, obs="observation", size=40):
    turns = []
    for i in range(rounds):
        turns.append({"role": "user", "content": f"{obs} {i} " + "o" * size})
        turns.append({"role": "assistant", "content": f"action {i}"})
    await m.messages.record_turns(
        session_id=sid,
        user_id=scope.user.id,
        organization_id=scope.org,
        turns=turns,
        actor=scope.actor,
        session_tag="task",
        agent_id=scope.agent,
    )
    await m.sessions.close_session(
        organization_id=scope.org, user_id=scope.user.id, session_id=sid
    )


async def claim(m, scope, **kwargs):
    kwargs.setdefault("threshold", 5)
    kwargs.setdefault("force", True)
    return await m.trigger.check_and_claim_fire(
        agent_id=scope.agent, user_id=scope.user.id, organization_id=scope.org, **kwargs
    )


async def execute(scope, run_id):
    from mirix.services.auto_dream_manager import AutoDreamManager

    return await AutoDreamManager().run(
        request=AutoDreamRequest(mode="procedural"),
        user=scope.user,
        actor=scope.actor,
        meta_agent_state=scope.meta,
        run_id=run_id,
    )


async def learn(m, scope, sid):
    """Claim ``sid`` (forced) and execute the run; return the finished run."""
    c = await claim(m, scope)
    assert sid in c.session_ids
    await execute(scope, c.run_id)
    return await m.runs.get_run(c.run_id)


async def session_row(sm, scope, sid):
    async with sm() as db:
        return (
            await db.execute(
                select(SessionORM).where(
                    SessionORM.organization_id == scope.org,
                    SessionORM.user_id == scope.user.id,
                    SessionORM.session_id == sid,
                )
            )
        ).scalar_one()


async def rows(sm, scope, sid):
    async with sm() as db:
        return (
            (
                await db.execute(
                    select(MessageORM)
                    .where(
                        MessageORM.user_id == scope.user.id,
                        MessageORM.session_id == sid,
                    )
                    .order_by(MessageORM.created_at)
                )
            )
            .scalars()
            .all()
        )


async def experiences(sm, scope, sid):
    async with sm() as db:
        return (
            (
                await db.execute(
                    select(ExperienceORM).where(
                        ExperienceORM.user_id == scope.user.id,
                        ExperienceORM.session_id == sid,
                    )
                )
            )
            .scalars()
            .all()
        )


def metric(run, sid):
    [entry] = [e for e in run.distill_metrics if e["session_id"] == sid]
    return entry


def events(run, kind=None):
    return [e for e in run.retry_events if kind is None or e["kind"] == kind]


async def assert_counted_failure(sm, scope, sid, kind, run, failures=1):
    entry = metric(run, sid)
    assert entry["status"] in ("failed", "failed_permanently")
    assert entry["failure_kind"] == kind and entry["counted_failure"] is True
    assert entry["distill_failures"] == failures
    session = await session_row(sm, scope, sid)
    assert session.distill_failures == failures
    assert session.distill_retry_after is None
    assert all(r.distilled_at is None for r in await rows(sm, scope, sid))
    assert await experiences(sm, scope, sid) == []
    assert run.status == "succeeded"  # a session failure never fails the run


# --------------------------------------------------------------------------- #
# Infrastructure errors: backoff, never counted                                #
# --------------------------------------------------------------------------- #


class TestInfrastructureErrors:
    @pytest.mark.parametrize(
        "fault",
        [
            Fault.http_error(503),
            Fault.http_error(500),
            Fault.rate_limit(retry_after=0),
            Fault.drop_connection(),
        ],
        ids=["503", "500", "429", "dropped"],
    )
    async def test_the_session_backs_off_and_is_not_counted(
        self, sm, env, scope, fake_llm_server, fault
    ):
        m = managers(sm)
        await write_session(m, scope, "infra")
        fake_llm_server.inject_fault(fault)
        before = dt.datetime.now(dt.timezone.utc)
        run = await learn(m, scope, "infra")

        entry = metric(run, "infra")
        assert entry["status"] == "deferred" and entry["failure_kind"] == "infra"
        assert entry["failure_kinds"] == ["infra"]
        assert entry["counted_failure"] is False
        session = await session_row(sm, scope, "infra")
        assert session.distill_failures == 0 and session.distill_failed_at is None
        assert session.distill_infra_failures == 1
        retry_after = as_utc(session.distill_retry_after)
        assert before + dt.timedelta(seconds=55) <= retry_after
        assert retry_after <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(
            seconds=65
        )
        assert entry["distill_retry_after"] == retry_after.isoformat()
        assert all(r.distilled_at is None for r in await rows(sm, scope, "infra"))
        assert run.status == "succeeded"

    async def test_a_backed_off_session_is_not_counted_until_it_is_due(
        self, sm, env, scope, fake_llm_server
    ):
        m = managers(sm)
        await write_session(m, scope, "later")
        fake_llm_server.inject_fault(Fault.http_error(502))
        await learn(m, scope, "later")
        # Not counted, not claimable while backing off: N = 1 does not fire.
        assert (await claim(m, scope, threshold=1, force=False)).fired is False
        # Once due it counts again.
        clock = Clock()
        clock.now += dt.timedelta(minutes=2)
        due = await claim(managers(sm, clock), scope, threshold=1, force=False)
        assert due.fired and due.session_ids == ["later"]

    async def test_consecutive_infrastructure_failures_double_the_backoff(
        self, sm, env, scope, fake_llm_server
    ):
        m = managers(sm)
        await write_session(m, scope, "twice")
        fake_llm_server.inject_fault(Fault.http_error(503))
        await learn(m, scope, "twice")
        clock = Clock()
        clock.now += dt.timedelta(minutes=2)
        later = managers(sm, clock)
        fake_llm_server.inject_fault(Fault.http_error(503))
        c = await claim(later, scope)
        await execute(scope, c.run_id)
        session = await session_row(sm, scope, "twice")
        assert session.distill_infra_failures == 2 and session.distill_failures == 0
        delay = as_utc(session.distill_retry_after) - dt.datetime.now(dt.timezone.utc)
        assert dt.timedelta(seconds=110) <= delay <= dt.timedelta(seconds=125)

    async def test_a_window_too_small_for_the_distiller_prompt_is_a_config_backoff(
        self, sm, env, scope, fake_llm_server
    ):
        # The distiller's system prompt and the output reserve fill the
        # usable window: a configuration error, not the session's fault. It
        # backs off (like an outage) and is never counted — three runs would
        # otherwise retire every session.
        env.distiller_kwargs = {
            "model_limits": ModelLimits(context_window=1_000, max_output_tokens=500)
        }
        m = managers(sm)
        await write_session(m, scope, "tiny-window", rounds=6, size=300)
        run = await learn(m, scope, "tiny-window")
        assert fake_llm_server.chat_requests() == []
        entry = metric(run, "tiny-window")
        assert entry["status"] == "deferred" and entry["failure_kind"] == "config"
        assert entry["counted_failure"] is False and entry["halvings"] == 0
        # The same configuration defers it on every run: its metrics say which.
        assert entry["config_deferral"] == "input_budget_too_small"
        session = await session_row(sm, scope, "tiny-window")
        assert session.distill_failures == 0 and session.distill_failed_at is None
        assert session.distill_retry_after is not None

    async def test_a_transport_retry_that_succeeds_needs_no_backoff(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 2)
        m = managers(sm)
        await write_session(m, scope, "transient")
        fake_llm_server.inject_fault(Fault.rate_limit(retry_after=0))
        fake_llm_server.queue_text(json.dumps([_experience("after a 429")]))
        run = await learn(m, scope, "transient")
        assert metric(run, "transient")["status"] == "ok"
        assert [e.title for e in await experiences(sm, scope, "transient")] == [
            "after a 429"
        ]
        session = await session_row(sm, scope, "transient")
        assert (
            session.distill_retry_after is None and session.distill_infra_failures == 0
        )
        # The SDK's transport retry is recorded on the run too, with
        # its session.
        [event] = [e for e in events(run) if e.get("stage") == "transport"]
        assert event["kind"] == "infra" and event["attempt"] == 1
        assert event["session_id"] == "transient"

    async def test_a_run_loads_the_vocabulary_a_failed_preload_missed(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        # A failed startup preload (offline) is retried by the next
        # learning run, before it plans its chunks, once the cooldown passed.
        import tiktoken

        loads = []

        def offline(name):
            loads.append(name)
            raise OSError("offline")

        monkeypatch.setattr(tiktoken, "get_encoding", offline)
        assert await context_budget.preload_tokenizer() is False
        monkeypatch.setattr(context_budget, "TOKENIZER_RETRY_SECONDS", 0.0)

        class Words:
            def encode(self, text, disallowed_special=()):
                return text.split()

        def online(name):
            loads.append(name)
            return Words()

        monkeypatch.setattr(tiktoken, "get_encoding", online)
        import mirix.services.session_experience_distiller as distiller_module

        awaited = []
        real_ensure = distiller_module.ensure_tokenizer

        async def ensure_then_record():
            loaded = await real_ensure()
            awaited.append((loaded, len(fake_llm_server.chat_requests())))
            return loaded

        monkeypatch.setattr(distiller_module, "ensure_tokenizer", ensure_then_record)
        m = managers(sm)
        await write_session(m, scope, "vocab")
        fake_llm_server.queue_text(json.dumps([_experience("counted exactly")]))
        run = await learn(m, scope, "vocab")
        assert metric(run, "vocab")["status"] == "ok"
        # Loaded (exact counts) before the chunks were planned and sent.
        assert awaited == [(True, 0)]
        assert loads == ["o200k_base", "o200k_base"]
        assert context_budget.get_token_counter().is_exact


# --------------------------------------------------------------------------- #
# Truncated output / over-budget input: halve the chunk budget                 #
# --------------------------------------------------------------------------- #


class TestHalving:
    async def test_a_truncated_answer_is_discarded_and_the_session_rechunked(
        self, sm, env, scope, fake_llm_server
    ):
        def responder(request_json):
            if _chunk_label(request_json) == "whole":
                # Parseable, but cut off by the output cap: must never be used.
                return ChatResponse(
                    content=json.dumps([_experience("FROM A TRUNCATED ANSWER")]),
                    finish_reason="length",
                )
            return ChatResponse(content=per_chunk_answer(request_json))

        fake_llm_server.set_default_responder(responder)
        m = managers(sm)
        await write_session(m, scope, "long", rounds=12, size=600)
        run = await learn(m, scope, "long")

        entry = metric(run, "long")
        assert entry["status"] == "ok" and entry["halvings"] == 1
        assert entry["failure_kinds"] == ["truncated"]
        assert entry["chunks"] >= 2
        labels = [_chunk_label(r.json) for r in fake_llm_server.chat_requests()]
        assert labels[0] == "whole"
        n = entry["chunks"]
        assert sorted(labels[1:]) == sorted(f"{i}/{n}" for i in range(1, n + 1))
        titles = sorted(e.title for e in await experiences(sm, scope, "long"))
        assert titles == sorted(f"lesson {i}/{n}" for i in range(1, n + 1))
        assert "FROM A TRUNCATED ANSWER" not in titles
        assert [
            (e["kind"], e["attempt"], e["stage"]) for e in events(run, "truncated")
        ] == [("truncated", 1, "distill")]
        assert all(r.distilled_at is not None for r in await rows(sm, scope, "long"))

    async def test_still_truncated_after_three_halvings_is_one_counted_failure(
        self, sm, env, scope, fake_llm_server
    ):
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(content="[", finish_reason="length")
        )
        m = managers(sm)
        await write_session(m, scope, "hopeless", rounds=40, size=900)
        run = await learn(m, scope, "hopeless")
        entry = metric(run, "hopeless")
        assert entry["halvings"] == 3
        # One entry per failed call: at least one per attempt (sibling chunks
        # that finished truncated before being cancelled add theirs).
        assert set(entry["failure_kinds"]) == {"truncated"}
        assert len(entry["failure_kinds"]) >= 4
        assert len(fake_llm_server.chat_requests()) >= 4
        await assert_counted_failure(sm, scope, "hopeless", "truncated", run)
        assert [e["attempt"] for e in events(run, "truncated")] == [1, 2, 3]

    async def test_context_budget_exceeded_halves_before_anything_is_sent(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        # The distiller plans with a big window, the client enforces a small
        # one: the whole session and the first halving are refused BEFORE
        # sending (ContextBudgetExceeded), the second halving fits.
        env.distiller_kwargs = {
            "model_limits": ModelLimits(
                context_window=1_000_000, max_output_tokens=100_000
            )
        }
        scope.meta.llm_config = scope.meta.llm_config.model_copy(
            update={"context_window": 12_000, "max_output_tokens": 2_000}
        )
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(content=per_chunk_answer(request_json))
        )
        m = managers(sm)
        await write_session(m, scope, "wide", rounds=40, size=900)
        run = await learn(m, scope, "wide")

        entry = metric(run, "wide")
        assert entry["status"] == "ok"
        assert entry["halvings"] >= 1
        assert set(entry["failure_kinds"]) == {"budget"}
        requests = fake_llm_server.chat_requests()
        # Only the fitting chunks ever reached the server (the refused calls
        # were stopped before sending and are not calls), each with its cap.
        assert len(requests) == entry["chunks"] == entry["llm_calls"] >= 2
        assert all(_chunk_label(r.json) != "whole" for r in requests)
        for request in requests:
            assert request.json["max_completion_tokens"] >= 1024
        assert len(await experiences(sm, scope, "wide")) == entry["chunks"]
        assert [e["kind"] for e in events(run)] == ["budget"] * entry["halvings"]


# --------------------------------------------------------------------------- #
# Unparseable output: repair prompts in the same dialogue                      #
# --------------------------------------------------------------------------- #


class TestRepairPrompts:
    async def test_a_repair_prompt_is_appended_to_the_same_dialogue(
        self, sm, env, scope, fake_llm_server
    ):
        fake_llm_server.queue_text("Sorry, I can only describe it in prose.")
        fake_llm_server.queue_text(json.dumps([_experience("repaired")]))
        m = managers(sm)
        await write_session(m, scope, "prose")
        run = await learn(m, scope, "prose")

        first, second = (r.json["messages"] for r in fake_llm_server.chat_requests())
        assert [msg["role"] for msg in first] == ["system", "user"]
        assert [msg["role"] for msg in second] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert second[:2] == first  # the same dialogue, continued
        assert second[2]["content"] == "Sorry, I can only describe it in prose."
        assert 'not one JSON object with an "experiences" array' in second[3]["content"]
        assert [e.title for e in await experiences(sm, scope, "prose")] == ["repaired"]
        entry = metric(run, "prose")
        assert entry["status"] == "ok" and entry["repair_prompts"] == 1
        assert entry["failure_kinds"] == ["parse"]
        assert [(e["kind"], e["attempt"]) for e in events(run, "parse")] == [
            ("parse", 1)
        ]

    async def test_an_array_without_a_valid_experience_is_repaired_too(
        self, sm, env, scope, fake_llm_server
    ):
        fake_llm_server.queue_text(
            json.dumps([{"kind": "lesson", "text": "wrong keys"}])
        )
        fake_llm_server.queue_text(json.dumps([_experience("right keys")]))
        m = managers(sm)
        await write_session(m, scope, "keys")
        await learn(m, scope, "keys")
        repair = fake_llm_server.chat_requests()[1].json["messages"][-1]["content"]
        assert "none of its elements is a valid experience" in repair
        assert [e.title for e in await experiences(sm, scope, "keys")] == ["right keys"]

    async def test_unparseable_after_two_repairs_is_one_counted_failure(
        self, sm, env, scope, fake_llm_server
    ):
        for _ in range(3):
            fake_llm_server.queue_text("no json here")
        m = managers(sm)
        await write_session(m, scope, "garbage")
        run = await learn(m, scope, "garbage")
        requests = fake_llm_server.chat_requests()
        assert [len(r.json["messages"]) for r in requests] == [2, 4, 6]
        # Unparseable output is NOT "no experiences": nothing is marked distilled.
        await assert_counted_failure(sm, scope, "garbage", "parse", run)
        assert metric(run, "garbage")["repair_prompts"] == 2
        assert metric(run, "garbage")["failure_kinds"] == ["parse"] * 3
        assert [e["attempt"] for e in events(run, "parse")] == [1, 2]

    async def test_an_explicit_empty_array_is_nothing_to_learn(
        self, sm, env, scope, fake_llm_server
    ):
        fake_llm_server.queue_text("[]")
        m = managers(sm)
        await write_session(m, scope, "empty")
        run = await learn(m, scope, "empty")
        entry = metric(run, "empty")
        assert entry["status"] == "ok" and entry["experiences"] == 0
        assert all(r.distilled_at is not None for r in await rows(sm, scope, "empty"))
        assert (await session_row(sm, scope, "empty")).distill_failures == 0


# --------------------------------------------------------------------------- #
# Timeouts: one retry, then counted                                            #
# --------------------------------------------------------------------------- #


class TestTimeouts:
    async def test_a_timed_out_call_is_retried_once(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 0.5)
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(
                content=json.dumps([_experience("in time")])
            )
        )
        fake_llm_server.inject_fault(Fault.slow(2.0))
        m = managers(sm)
        await write_session(m, scope, "slow")
        run = await learn(m, scope, "slow")
        entry = metric(run, "slow")
        assert entry["status"] == "ok" and entry["timeouts"] == 1
        assert entry["llm_calls"] == 2 and entry["failure_kinds"] == ["timeout"]
        assert [e.title for e in await experiences(sm, scope, "slow")] == ["in time"]
        assert [(e["kind"], e["attempt"]) for e in events(run, "timeout")] == [
            ("timeout", 1)
        ]

    async def test_timing_out_again_is_one_counted_failure(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        monkeypatch.setattr(constants, "LLM_TIMEOUT_SECONDS", 0.5)
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(content="[]")
        )
        fake_llm_server.inject_fault(Fault.slow(2.0), times=2)
        m = managers(sm)
        await write_session(m, scope, "slower")
        run = await learn(m, scope, "slower")
        assert metric(run, "slower")["timeouts"] == 2
        assert metric(run, "slower")["failure_kinds"] == ["timeout", "timeout"]
        await assert_counted_failure(sm, scope, "slower", "timeout", run)

    # --- the eval gateway enforces the 10-minute deadline itself ---

    async def test_a_gateways_model_timeout_is_the_calls_timeout(
        self, sm, env, scope, fake_llm_server
    ):
        """The gateway's 504 saying the model's own attempt used the whole
        deadline (``blame: model``) is this call timing out: retried once, then
        one counted failure — never an infrastructure back-off."""
        from fakes.gateway_errors import GATEWAY_MODEL_TIMEOUT

        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(content="[]")
        )
        fake_llm_server.inject_fault(
            Fault(status=504, body=GATEWAY_MODEL_TIMEOUT), times=2
        )
        m = managers(sm)
        await write_session(m, scope, "gw-slow")
        run = await learn(m, scope, "gw-slow")
        entry = metric(run, "gw-slow")
        assert entry["timeouts"] == 2 and entry["llm_calls"] == 2
        assert entry["failure_kinds"] == ["timeout", "timeout"]
        await assert_counted_failure(sm, scope, "gw-slow", "timeout", run)
        session = await session_row(sm, scope, "gw-slow")
        assert session.distill_infra_failures == 0

    async def test_a_deadline_the_gateway_lost_stays_infrastructure(
        self, sm, env, scope, fake_llm_server
    ):
        from fakes.gateway_errors import GATEWAY_LOST_DEADLINE

        fake_llm_server.inject_fault(Fault(status=504, body=GATEWAY_LOST_DEADLINE))
        m = managers(sm)
        await write_session(m, scope, "gw-lost")
        run = await learn(m, scope, "gw-lost")
        entry = metric(run, "gw-lost")
        assert entry["status"] == "deferred" and entry["failure_kind"] == "infra"
        assert entry["counted_failure"] is False

    async def test_a_moderation_refusal_is_counted_not_deferred(
        self, sm, env, scope, fake_llm_server
    ):
        """OpenRouter's moderation 403 refuses this transcript: it would be
        refused again, so it is one counted failure (not a configuration
        back-off that defers the session across runs, run after run)."""
        from fakes.gateway_errors import GATEWAY_MODERATION_403

        fake_llm_server.inject_fault(Fault(status=403, body=GATEWAY_MODERATION_403))
        m = managers(sm)
        await write_session(m, scope, "flagged")
        run = await learn(m, scope, "flagged")
        assert metric(run, "flagged")["llm_calls"] == 1  # never retried
        await assert_counted_failure(sm, scope, "flagged", "refused", run)


# --------------------------------------------------------------------------- #
# Database errors: the transaction is retried, then the run fails              #
# --------------------------------------------------------------------------- #


def _locked():
    from sqlalchemy.exc import OperationalError

    return OperationalError("COMMIT", {}, Exception("database is locked"))


class TestDatabaseErrors:
    async def test_a_transient_commit_failure_is_retried(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        real_commit = SkillLearningRunManager.commit_distilled_session
        failures = {"left": 2}

        async def flaky(self, *args, **kwargs):
            if failures["left"]:
                failures["left"] -= 1
                raise _locked()
            return await real_commit(self, *args, **kwargs)

        monkeypatch.setattr(SkillLearningRunManager, "commit_distilled_session", flaky)
        fake_llm_server.queue_text(json.dumps([_experience("committed once")]))
        m = managers(sm)
        await write_session(m, scope, "flaky-db")
        run = await learn(m, scope, "flaky-db")
        assert [e.title for e in await experiences(sm, scope, "flaky-db")] == [
            "committed once"
        ]
        assert [(e["kind"], e["attempt"], e["stage"]) for e in events(run, "db")] == [
            ("db", 1, "distill_commit"),
            ("db", 2, "distill_commit"),
        ]

    async def test_a_repeated_commit_never_duplicates_experiences(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        """The connection drops AFTER the commit went through: the retried
        commit finds nothing left to mark and inserts nothing again."""
        real_commit = SkillLearningRunManager.commit_distilled_session
        state = {"dropped": False}

        async def commit_then_drop(self, *args, **kwargs):
            result = await real_commit(self, *args, **kwargs)
            if not state["dropped"]:
                state["dropped"] = True
                raise _locked()
            return result

        monkeypatch.setattr(
            SkillLearningRunManager, "commit_distilled_session", commit_then_drop
        )
        fake_llm_server.queue_text(json.dumps([_experience("exactly once")]))
        m = managers(sm)
        await write_session(m, scope, "dup")
        await learn(m, scope, "dup")
        assert [e.title for e in await experiences(sm, scope, "dup")] == [
            "exactly once"
        ]

    @pytest.mark.parametrize("settle", ["counted", "deferred"])
    async def test_a_repeated_failure_settle_counts_once(
        self, sm, env, scope, fake_llm_server, monkeypatch, settle
    ):
        """The connection drops AFTER a failure settle committed. The
        retried settle finds its own marker on the run and changes nothing:
        one failure is one count (or one backoff step), never two."""
        method = (
            "record_distill_failure"
            if settle == "counted"
            else "defer_distilled_session"
        )
        real = getattr(SkillLearningRunManager, method)
        state = {"dropped": 0}

        async def settle_then_drop(self, *args, **kwargs):
            result = await real(self, *args, **kwargs)
            if not state["dropped"]:
                state["dropped"] += 1
                raise _locked()
            return result

        monkeypatch.setattr(SkillLearningRunManager, method, settle_then_drop)
        if settle == "counted":
            fake_llm_server.set_default_responder(lambda request_json: "never json")
        else:
            fake_llm_server.inject_fault(Fault.http_error(503))
        m = managers(sm)
        sid = f"settle-once-{settle}"
        await write_session(m, scope, sid)
        run = await learn(m, scope, sid)
        assert state["dropped"] == 1  # the settle really was repeated
        assert [e["stage"] for e in events(run, "db")] == ["distill_settle"]
        session = await session_row(sm, scope, sid)
        entry = metric(run, sid)
        if settle == "counted":
            assert session.distill_failures == 1 and session.distill_failed_at is None
            assert entry["distill_failures"] == 1
            assert entry["settled"]["action"] == "counted"
        else:
            assert session.distill_infra_failures == 1
            assert session.distill_failures == 0
            # The first backoff step (1 minute), not the second (2 minutes).
            waited = as_utc(session.distill_retry_after) - as_utc(session.updated_at)
            assert waited == dt.timedelta(
                seconds=constants.DISTILL_RETRY_BACKOFF_MIN_SECONDS
            )
            assert entry["settled"]["action"] == "deferred"

    async def test_a_persistent_database_failure_fails_the_run_and_releases(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        async def broken(self, *args, **kwargs):
            raise _locked()

        monkeypatch.setattr(SkillLearningRunManager, "commit_distilled_session", broken)
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(content=json.dumps([_experience("x")]))
        )
        m = managers(sm)
        await write_session(m, scope, "db-down")
        c = await claim(m, scope)
        with pytest.raises(Exception, match="database is locked"):
            await execute(scope, c.run_id)
        run = await m.runs.get_run(c.run_id)
        assert run.status == "failed" and "database is locked" in run.error
        assert len(events(run, "db")) == 5
        # Not counted against the session, which is claimable again.
        session = await session_row(sm, scope, "db-down")
        assert session.distill_failures == 0 and session.distill_retry_after is None
        assert (await claim(m, scope)).session_ids == ["db-down"]


# --------------------------------------------------------------------------- #
# MIRIX's own failure: neither counted nor deferred, the run fails             #
# --------------------------------------------------------------------------- #


def _pydantic_validation_error():
    from pydantic import TypeAdapter, ValidationError

    try:
        TypeAdapter(int).validate_python("not a number")
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


class TestInternalErrors:
    @pytest.mark.parametrize(
        "make_bug",
        [
            lambda: KeyError("tool_rules"),
            lambda: _pydantic_validation_error(),
            lambda: FileNotFoundError(2, "No such file", "/images/page-1.png"),
            lambda: json.JSONDecodeError("stored arguments", "{", 1),
        ],
        ids=["KeyError", "ValidationError", "FileNotFoundError", "JSONDecodeError"],
    )
    async def test_a_request_mirix_cannot_build_is_never_counted_or_deferred(
        self, sm, env, scope, fake_llm_server, monkeypatch, make_bug
    ):
        # A KeyError / ValidationError used to count (3 runs retired the
        # transcript: a silent drop caused by a MIRIX bug); an OSError used to
        # defer it forever (INFRA, backoff capped at an hour, never counted).
        # Now the session stays pending with no count and no backoff, the
        # run's other sessions are learned, and the run fails loudly.
        from mirix.errors import LearningRunInternalError
        from mirix.llm_api.openai_client import OpenAIClient

        original = OpenAIClient.build_request_data

        def mentions(messages, needle):
            return any(
                needle in (getattr(part, "text", None) or "")
                for message in messages
                for part in (getattr(message, "content", None) or [])
            )

        async def broken(self, messages, *args, **kwargs):
            if mentions(messages, "session_id: broken"):
                raise make_bug()
            return await original(self, messages, *args, **kwargs)

        monkeypatch.setattr(OpenAIClient, "build_request_data", broken)
        fake_llm_server.set_default_responder(
            lambda request_json: ChatResponse(
                content=json.dumps([_experience("learned")])
            )
        )
        m = managers(sm)
        await write_session(m, scope, "broken")
        await write_session(m, scope, "healthy")
        for attempt in range(4):  # one more than the 3 counted failures
            c = await claim(m, scope)
            assert "broken" in c.session_ids
            with pytest.raises(LearningRunInternalError) as raised:
                await execute(scope, c.run_id)
            assert raised.value.session_ids == ["broken"]
            run = await m.runs.get_run(c.run_id)
            assert run.status == "failed"
            assert run.error.startswith("LearningRunInternalError")
            entry = metric(run, "broken")
            assert entry["status"] == "internal_error"
            assert entry["failure_kind"] == "internal"
            assert entry["counted_failure"] is False
            # No request was sent, so no LLM call is counted.
            assert entry["llm_calls"] == 0
            session = await session_row(sm, scope, "broken")
            assert session.distill_failures == 0 and session.distill_failed_at is None
            assert session.distill_retry_after is None
            assert session.distill_infra_failures == 0
            assert all(r.distilled_at is None for r in await rows(sm, scope, "broken"))
            if attempt == 0:  # the other session of the run is learned
                assert metric(run, "healthy")["status"] == "ok"
                assert metric(run, "healthy")["llm_calls"] == 1
        assert [e.title for e in await experiences(sm, scope, "healthy")] == ["learned"]
        assert await experiences(sm, scope, "broken") == []
        # Nothing of the broken session ever reached the model.
        assert not any(
            "session_id: broken" in _user_text(r.json)
            for r in fake_llm_server.chat_requests()
        )


# --------------------------------------------------------------------------- #
# Three strikes; metrics; reopen during the distill call                       #
# --------------------------------------------------------------------------- #


class TestFailureLimit:
    async def test_three_counted_failures_retire_the_session(
        self, sm, env, scope, fake_llm_server
    ):
        fake_llm_server.set_default_responder(lambda request_json: "still not json")
        m = managers(sm)
        await write_session(m, scope, "doomed")
        for attempt in (1, 2, 3):
            run = await learn(m, scope, "doomed")
            await assert_counted_failure(
                sm, scope, "doomed", "parse", run, failures=attempt
            )
        assert metric(run, "doomed")["status"] == "failed_permanently"
        assert metric(run, "doomed")["failed_permanently"] is True
        session = await session_row(sm, scope, "doomed")
        assert session.distill_failed_at is not None
        # Never selected again, and not counted toward N.
        assert "doomed" not in await m.sessions.list_pending_distill_sessions(
            organization_id=scope.org, user_id=scope.user.id, agent_id=scope.agent
        )
        assert (await claim(m, scope, threshold=1, force=False)).fired is False
        # A new write reopens it and resets the count.
        await write_session(m, scope, "doomed")
        session = await session_row(sm, scope, "doomed")
        assert session.distill_failures == 0 and session.distill_failed_at is None

    async def test_infrastructure_failures_never_add_up_to_the_limit(
        self, sm, env, scope, fake_llm_server
    ):
        m = managers(sm)
        await write_session(m, scope, "patient")
        clock = Clock()
        for _ in range(4):
            fake_llm_server.inject_fault(Fault.http_error(503))
            clock.now += dt.timedelta(hours=2)  # past any backoff
            c = await claim(managers(sm, clock), scope)
            await execute(scope, c.run_id)
        session = await session_row(sm, scope, "patient")
        assert session.distill_failures == 0 and session.distill_failed_at is None
        assert session.distill_infra_failures == 4


class TestMetricsAndUsage:
    async def test_every_m4_4_field_and_the_runs_llm_usage(
        self, sm, env, scope, fake_llm_server
    ):
        def responder(request_json):
            if "session_id: measured" in _user_text(request_json):
                return ChatResponse(
                    content=json.dumps(
                        [
                            _experience("one"),
                            _experience("two", "worth_avoiding"),
                            {"bad": 1},
                        ]
                    ),
                    usage={
                        "prompt_tokens": 1234,
                        "completion_tokens": 56,
                        "total_tokens": 1290,
                    },
                )
            return ChatResponse(
                content="[]",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "total_tokens": 102,
                },
            )

        fake_llm_server.set_default_responder(responder)
        m = managers(sm)
        await write_session(m, scope, "measured", rounds=4)
        await write_session(m, scope, "idle-empty", rounds=1)
        c = await claim(m, scope)
        await execute(scope, c.run_id)
        run = await m.runs.get_run(c.run_id)
        entry = metric(run, "measured")
        assert entry["messages"] == 8 and entry["rounds"] == 4
        assert entry["raw_tokens"] > 0 and entry["images"] == 0
        assert entry["chunks"] == 1 and entry["chunk_budget_tokens"] is None
        assert entry["truncated_messages"] == 0 and entry["truncated_tokens"] == 0
        assert entry["llm_calls"] == 1 and entry["failure_kinds"] == []
        assert entry["experiences"] == 2 and entry["invalid_items"] == 1
        usages = sorted(
            metric(run, sid)["llm_usage"]["prompt_tokens"]
            for sid in ("measured", "idle-empty")
        )
        assert usages == [100, 1234]
        assert run.llm_usage == {
            "calls": 2,
            "prompt_tokens": 1334,
            "completion_tokens": 58,
            "total_tokens": 1392,
            "by_stage": {
                "distill": {
                    "calls": 2,
                    "prompt_tokens": 1334,
                    "completion_tokens": 58,
                    "total_tokens": 1392,
                }
            },
        }


class TestSessionHeader:
    async def test_each_session_is_served_on_its_own_distiller_calls(
        self, sm, env, scope, fake_llm_server, monkeypatch
    ):
        """With MIRIX_LLM_SESSION_HEADER set, every distiller call carries the id of
        the session it distills, while three sessions are distilled at once (each
        session's task keeps its own); with the setting off (the default), none."""
        from mirix.settings import settings

        header = "x-skill-bench-item"
        monkeypatch.setattr(settings, "llm_session_header", header)
        together = threading.Barrier(3, timeout=10)

        def responder(request_json):
            together.wait()  # all three sessions' calls are in flight at once
            return ChatResponse(content="[]")

        fake_llm_server.set_default_responder(responder)
        m = managers(sm)
        sids = [f"t{uuid.uuid4().hex[:20]}" for _ in range(3)]
        for sid in sids:
            await write_session(m, scope, sid, rounds=2)
        before = len(fake_llm_server.chat_requests())
        c = await claim(m, scope)
        assert set(sids) <= set(c.session_ids)
        await execute(scope, c.run_id)
        run = await m.runs.get_run(c.run_id)
        assert all(metric(run, sid)["status"] == "ok" for sid in sids)
        calls = fake_llm_server.chat_requests()[before:]
        assert len(calls) == 3
        served = {
            _user_text(call.json)
            .split("session_id: ", 1)[1]
            .split()[0]: call.headers.get(header)
            for call in calls
        }
        assert served == {sid: sid for sid in sids}

        monkeypatch.setattr(settings, "llm_session_header", None)
        fake_llm_server.set_default_responder(lambda request_json: "[]")
        await write_session(m, scope, "unserved", rounds=1)
        before = len(fake_llm_server.chat_requests())
        c = await claim(m, scope)
        await execute(scope, c.run_id)
        calls = fake_llm_server.chat_requests()[before:]
        assert calls and all(header not in call.headers for call in calls)


class TestReopenDuringDistill:
    async def test_rows_written_while_the_model_runs_stay_undistilled(
        self, sm, env, scope, fake_llm_server
    ):
        """The claim-time watermark, with the write landing DURING the distill call."""
        entered = threading.Event()
        release = threading.Event()

        def responder(request_json):
            entered.set()
            release.wait(10)
            return ChatResponse(content=json.dumps([_experience("before reopen")]))

        fake_llm_server.set_default_responder(responder)
        m = managers(sm)
        await write_session(m, scope, "reopened", rounds=2)
        c = await claim(m, scope)
        task = asyncio.create_task(execute(scope, c.run_id))
        assert await asyncio.to_thread(entered.wait, 10)
        # The user comes back while the model is still reading the session.
        out = await m.messages.record_turns(
            session_id="reopened",
            user_id=scope.user.id,
            organization_id=scope.org,
            turns=[{"role": "user", "content": "written during the distill call"}],
            actor=scope.actor,
            session_tag="task",
            agent_id=scope.agent,
        )
        assert out.reopened is True
        release.set()
        await task

        marks = [r.distilled_at is not None for r in await rows(sm, scope, "reopened")]
        assert marks == [True, True, True, True, False]
        [request] = fake_llm_server.chat_requests()
        assert "written during the distill call" not in _user_text(request.json)
        assert [e.title for e in await experiences(sm, scope, "reopened")] == [
            "before reopen"
        ]
