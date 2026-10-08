"""Session lifecycle, trigger claims, learning runs and the sweeper.

Hermetic: a throwaway SQLite file per module (sqlite+aiosqlite, create_all),
managers pointed at it through their ``session_maker`` / ``clock`` hooks, and an
injectable clock so every state transition is checked without sleeping. No
server, no network, no LLM (the distiller gets a scripted in-process client).

Postgres-only behavior (row locks, cross-process claim serialization, the
set-based idle UPDATE) is covered by tests/test_agent_trigger_state_integration.py.
"""

import asyncio
import datetime as dt
import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.conversation_message import ConversationMessage as MessageORM
from mirix.orm.conversation_session import ConversationSession as SessionORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.skill_learning_run import SkillLearningRun as RunORM
from mirix.orm.user import User as UserORM
from mirix.schemas.auto_dream import AutoDreamRequest
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.services import skill_learning_trigger
from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_session_manager import (
    ConversationSessionManager,
    as_utc,
)
from mirix.services.skill_learning_run_manager import (
    LearningRunLeaseLost,
    SkillLearningRunManager,
    background_learning_tasks,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

IDLE = 172800  # the default MIRIX_SESSION_IDLE_TIMEOUT_SECONDS


class FakeClock:
    def __init__(self, start=None):
        self.now = start or dt.datetime(2030, 1, 1, 12, 0, tzinfo=dt.timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + dt.timedelta(**kwargs)
        return self.now


# --------------------------------------------------------------------------- #
# Hermetic database and per-test scopes                                        #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_learning") / "test.db"
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


@pytest_asyncio.fixture(loop_scope="module")
async def fresh_sm(tmp_path):
    """A private database, for global operations (the sweeper) whose effects
    must not be mixed with other tests' rows."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}")
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


async def _new_scope(sm, org_id, *, client_id=None):
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    new_client = client_id is None
    client_id = client_id or f"client-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        if new_client:
            # The meta agent's owning client: a run executes as it.
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
    actor = PydanticClient(
        id=client_id,
        organization_id=org_id,
        name="test client",
        write_scope="test",
        read_scopes=["test"],
    )
    user = PydanticUser(
        id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
    )
    meta = SimpleNamespace(id=agent_id, llm_config=SimpleNamespace(model="base-model"))
    return SimpleNamespace(
        org=org_id, user=user, actor=actor, agent=agent_id, meta=meta
    )


@pytest_asyncio.fixture(loop_scope="module")
async def scope(sm, org_id):
    return await _new_scope(sm, org_id)


@pytest.fixture
def clock():
    return FakeClock()


def _managers(sm, clock):
    return SimpleNamespace(
        messages=ConversationMessageManager(session_maker=sm, clock=clock),
        sessions=ConversationSessionManager(session_maker=sm, clock=clock),
        trigger=AgentTriggerStateManager(session_maker=sm, clock=clock),
        runs=SkillLearningRunManager(
            session_maker=sm, clock=clock, heartbeat_seconds=0.01
        ),
    )


@pytest.fixture
def m(sm, clock):
    return _managers(sm, clock)


async def _write(m, scope, sid, *, tag="task", n=2, text="turn"):
    return await m.messages.record_turns(
        session_id=sid,
        user_id=scope.user.id,
        organization_id=scope.org,
        turns=[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"{text} {i}"}
            for i in range(n)
        ],
        actor=scope.actor,
        session_tag=tag,
        agent_id=scope.agent,
    )


async def _ended_sessions(m, scope, count, *, prefix="s", tag="task"):
    sids = []
    for i in range(count):
        sid = f"{prefix}{i}-{uuid.uuid4().hex[:6]}"
        await _write(m, scope, sid, tag=tag)
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id=sid
        )
        m.sessions.clock.advance(seconds=1)
        sids.append(sid)
    return sids


async def _check(m, scope, **kwargs):
    kwargs.setdefault("threshold", 5)
    return await m.trigger.check_and_claim_fire(
        agent_id=scope.agent,
        user_id=scope.user.id,
        organization_id=scope.org,
        **kwargs,
    )


async def _pending_experiences(sm, scope, count, *, status="pending"):
    """Experiences a distill committed for the scope, waiting for the curator."""
    from mirix.orm.skill_experience import SkillExperience as ExperienceORM

    async with sm() as db:
        for i in range(count):
            db.add(
                ExperienceORM(
                    id=f"exp-{uuid.uuid4().hex[:10]}",
                    organization_id=scope.org,
                    user_id=scope.user.id,
                    agent_id=scope.agent,
                    session_id=f"done-{i}",
                    experience_type="worth_learning",
                    title=f"left over {i}",
                    content="check the totals first",
                    importance=0.5,
                    credibility=0.5,
                    evidence="{}",
                    status=status,
                )
            )
        await db.commit()


async def _session_row(sm, scope, sid):
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


async def _rows(sm, scope, sid):
    async with sm() as db:
        return (
            (
                await db.execute(
                    select(MessageORM)
                    .where(
                        MessageORM.organization_id == scope.org,
                        MessageORM.user_id == scope.user.id,
                        MessageORM.session_id == sid,
                    )
                    .order_by(MessageORM.created_at)
                )
            )
            .scalars()
            .all()
        )


# --------------------------------------------------------------------------- #
# 2.1  Session writes, tags, close, idle, reopen                               #
# --------------------------------------------------------------------------- #


class TestSessionWrites:
    async def test_first_write_creates_the_session_row(self, m, scope, clock):
        out = await _write(m, scope, "first-write", n=3)
        assert out.created is True and out.reopened is False
        assert out.effective_tag == "task" and out.warnings == []
        s = out.session
        assert s.agent_id == scope.agent
        assert s.message_count == 3
        assert s.first_message_at == out[0].created_at
        assert s.last_message_at == out[-1].created_at
        assert as_utc(s.first_message_at) == clock.now
        assert s.closed_at is None and s.reopen_count == 0

    async def test_turns_append_in_order_across_writes(self, m, scope, clock):
        first = await _write(m, scope, "ordering", n=2)
        second = await _write(m, scope, "ordering", n=2)  # same clock tick
        assert second[0].created_at > first[-1].created_at
        assert second.session.message_count == 4
        assert second.session.last_message_at == second[-1].created_at

    async def test_untagged_first_write_takes_the_default_tag(self, m, scope):
        out = await _write(m, scope, "untagged", tag=None)
        assert out.effective_tag == "conversation"
        assert out[0].session_tag == "conversation"

    async def test_q2_first_tag_wins_and_mismatch_is_a_warning(self, m, scope):
        await _write(m, scope, "q2", tag="task")
        again = await _write(m, scope, "q2", tag="conversation")
        assert again.effective_tag == "task"
        assert [t.session_tag for t in again] == ["task", "task"]
        assert len(again.warnings) == 1
        assert again.warnings[0].startswith("session_tag mismatch")
        assert "'task'" in again.warnings[0]
        # Not stating a tag keeps the stored one silently.
        silent = await _write(m, scope, "q2", tag=None)
        assert silent.effective_tag == "task" and silent.warnings == []
        assert (
            await _session_row(m.sessions.session_maker, scope, "q2")
        ).session_tag == "task"

    async def test_legacy_untagged_session_adopts_the_first_named_tag(
        self, sm, m, scope
    ):
        await _write(m, scope, "legacy", tag="task")
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(
                    SessionORM.session_id == "legacy",
                    SessionORM.user_id == scope.user.id,
                )
                .values(session_tag=None)
            )
            await db.commit()
        out = await _write(m, scope, "legacy", tag="conversation")
        assert out.effective_tag == "conversation" and out.warnings == []

    async def test_legacy_row_without_agent_adopts_the_writers_agent(
        self, sm, m, scope
    ):
        await _write(m, scope, "no-agent")
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(
                    SessionORM.session_id == "no-agent",
                    SessionORM.user_id == scope.user.id,
                )
                .values(agent_id=None)
            )
            await db.commit()
        out = await _write(m, scope, "no-agent")
        assert out.session.agent_id == scope.agent

    async def test_empty_batch_writes_nothing(self, sm, m, scope):
        out = await m.messages.record_turns(
            session_id="empty-batch",
            user_id=scope.user.id,
            organization_id=scope.org,
            turns=[],
            actor=scope.actor,
            agent_id=scope.agent,
        )
        assert list(out) == [] and out.session is None
        async with sm() as db:
            found = (
                await db.execute(
                    select(SessionORM.id).where(SessionORM.session_id == "empty-batch")
                )
            ).first()
        assert found is None

    async def test_invalid_turn_writes_neither_rows_nor_session(self, sm, m, scope):
        with pytest.raises(Exception):
            await m.messages.record_turns(
                session_id="bad-role",
                user_id=scope.user.id,
                organization_id=scope.org,
                turns=[{"role": "system", "content": "x"}],
                actor=scope.actor,
                agent_id=scope.agent,
            )
        async with sm() as db:
            found = (
                await db.execute(
                    select(SessionORM.id).where(SessionORM.session_id == "bad-role")
                )
            ).first()
        assert found is None


class TestCloseIdleReopen:
    async def test_explicit_close_is_conditional_and_idempotent(self, m, scope, clock):
        await _write(m, scope, "close-me")
        clock.advance(minutes=5)
        first = await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="close-me"
        )
        assert first.already_closed is False
        assert as_utc(first.session.closed_at) == clock.now
        assert first.session.close_reason == "explicit"
        clock.advance(minutes=5)
        second = await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="close-me"
        )
        assert second.already_closed is True
        assert second.session.closed_at == first.session.closed_at

    async def test_close_unknown_session_returns_none(self, m, scope):
        assert (
            await m.sessions.close_session(
                organization_id=scope.org, user_id=scope.user.id, session_id="nope"
            )
            is None
        )

    async def test_close_is_scoped_to_the_user(self, sm, m, scope, org_id):
        other = await _new_scope(sm, org_id)
        await _write(m, scope, "mine")
        assert (
            await m.sessions.close_session(
                organization_id=scope.org, user_id=other.user.id, session_id="mine"
            )
            is None
        )

    async def test_idle_session_ends_exactly_after_the_timeout(
        self, sm, m, scope, clock
    ):
        out = await _write(m, scope, "idle")
        last = as_utc(out.session.last_message_at)
        clock.now = last + dt.timedelta(seconds=IDLE)  # at the cutoff: still open
        row = await _session_row(sm, scope, "idle")
        assert m.sessions.is_ended(row) is False
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 0
        clock.advance(microseconds=1)
        row = await _session_row(sm, scope, "idle")
        assert m.sessions.is_ended(row) is True
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 1
        row = await _session_row(sm, scope, "idle")
        assert row.close_reason == "idle"
        assert as_utc(row.closed_at) == last + dt.timedelta(seconds=IDLE)
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 0

    async def test_close_after_idle_reports_already_closed(self, m, scope, clock):
        await _write(m, scope, "idle-then-close")
        clock.advance(seconds=IDLE + 1)
        outcome = await m.sessions.close_session(
            organization_id=scope.org,
            user_id=scope.user.id,
            session_id="idle-then-close",
        )
        assert outcome.already_closed is True
        assert outcome.session.close_reason == "idle"

    async def test_write_to_a_closed_session_reopens_it(self, m, scope, clock):
        await _write(m, scope, "reopen")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="reopen"
        )
        clock.advance(hours=1)
        out = await _write(m, scope, "reopen")
        assert out.reopened is True
        assert out.session.closed_at is None and out.session.close_reason is None
        assert out.session.reopen_count == 1
        assert out.session.message_count == 4

    async def test_write_to_an_idle_unmaterialized_session_reopens_it(
        self, m, scope, clock
    ):
        await _write(m, scope, "idle-reopen")
        clock.advance(seconds=IDLE + 60)
        out = await _write(m, scope, "idle-reopen")
        assert out.reopened is True and out.session.reopen_count == 1
        # The session is live again: nothing to materialize.
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 0

    async def test_write_before_the_timeout_keeps_it_open(self, m, scope, clock):
        await _write(m, scope, "busy")
        clock.advance(seconds=IDLE - 10)
        out = await _write(m, scope, "busy")
        assert out.reopened is False and out.session.reopen_count == 0
        clock.advance(seconds=20)  # idle counts from the NEW last message
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 0

    async def test_close_open_sessions_closes_only_open_learnable_ones(self, m, scope):
        await _write(m, scope, "open-task")
        await _write(m, scope, "open-chat", tag="conversation")
        await _write(m, scope, "closed-task")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="closed-task"
        )
        closed = await m.sessions.close_open_sessions(
            organization_id=scope.org, user_id=scope.user.id, agent_id=scope.agent
        )
        assert closed == ["open-task"]


# --------------------------------------------------------------------------- #
# 2.1  Pending selection                                                       #
# --------------------------------------------------------------------------- #


class TestPendingSelection:
    async def _pending(self, m, scope, **kw):
        return await m.sessions.list_pending_distill_sessions(
            organization_id=scope.org, user_id=scope.user.id, agent_id=scope.agent, **kw
        )

    async def test_only_ended_task_sessions_with_undistilled_rows(
        self, sm, m, scope, clock
    ):
        ended = await _ended_sessions(m, scope, 2, prefix="ended")
        await _write(m, scope, "still-open")
        await _ended_sessions(m, scope, 1, prefix="chat", tag="conversation")
        assert await self._pending(m, scope) == ended  # oldest first

        # Fully distilled sessions drop out; one new row brings a session back.
        marked = await m.messages.mark_sessions_distilled(
            session_watermarks={ended[0]: clock.now},
            user_id=scope.user.id,
            organization_id=scope.org,
            actor=scope.actor,
        )
        assert marked == 2
        assert await self._pending(m, scope) == [ended[1]]

    async def test_idle_sessions_are_pending_before_materialization(
        self, m, scope, clock
    ):
        await _write(m, scope, "idle-pending")
        assert await self._pending(m, scope) == []
        clock.advance(seconds=IDLE + 1)
        assert await self._pending(m, scope) == ["idle-pending"]

    async def test_failed_and_backed_off_sessions_are_not_pending(
        self, sm, m, scope, clock
    ):
        a, b = await _ended_sessions(m, scope, 2, prefix="bk")
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(SessionORM.session_id == a, SessionORM.user_id == scope.user.id)
                .values(distill_failed_at=clock.now)
            )
            await db.execute(
                update(SessionORM)
                .where(SessionORM.session_id == b, SessionORM.user_id == scope.user.id)
                .values(distill_retry_after=clock.now + dt.timedelta(minutes=5))
            )
            await db.commit()
        assert await self._pending(m, scope) == []
        clock.advance(minutes=5)
        assert await self._pending(m, scope) == [b]

    async def test_legacy_untagged_sessions_are_pending(self, sm, m, scope):
        [sid] = await _ended_sessions(m, scope, 1, prefix="legacy")
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(
                    SessionORM.session_id == sid, SessionORM.user_id == scope.user.id
                )
                .values(session_tag=None)
            )
            await db.commit()
        assert await self._pending(m, scope) == [sid]

    async def test_soft_deleted_rows_do_not_make_a_session_pending(self, sm, m, scope):
        [sid] = await _ended_sessions(m, scope, 1, prefix="deleted")
        async with sm() as db:
            await db.execute(
                update(MessageORM)
                .where(
                    MessageORM.session_id == sid, MessageORM.user_id == scope.user.id
                )
                .values(is_deleted=True)
            )
            await db.commit()
        assert await self._pending(m, scope) == []

    async def test_scopes_are_isolated(self, sm, m, scope, org_id):
        other = await _new_scope(sm, org_id)
        await _ended_sessions(m, scope, 1, prefix="mine")
        assert (
            await m.sessions.list_pending_distill_sessions(
                organization_id=other.org, user_id=other.user.id, agent_id=other.agent
            )
            == []
        )
        # Same user, another meta agent: not this scope.
        assert (
            await m.sessions.list_pending_distill_sessions(
                organization_id=scope.org, user_id=scope.user.id, agent_id=other.agent
            )
            == []
        )

    async def test_list_pending_scopes(self, sm, m, scope, org_id, clock):
        await _ended_sessions(m, scope, 1, prefix="scoped")
        scopes = await m.sessions.list_pending_scopes()
        assert (scope.org, scope.user.id, scope.agent) in scopes


# --------------------------------------------------------------------------- #
# 2.2  Trigger claim                                                           #
# --------------------------------------------------------------------------- #


class TestTriggerClaim:
    async def test_below_threshold_never_fires_however_long_it_waits(
        self, m, scope, clock
    ):
        await _ended_sessions(m, scope, 4)
        for days in (0, 3, 30, 365):
            clock.advance(days=days)
            claim = await _check(m, scope)
            assert claim.fired is False and claim.run_id is None
            assert claim.pending_count == 4

    async def test_the_first_check_of_a_scope_can_fire(self, sm, m, scope):
        sids = await _ended_sessions(m, scope, 5)
        claim = await _check(m, scope)  # no cursor existed before this call
        assert claim.fired is True
        assert claim.session_ids == sids
        assert claim.pending_count == 5
        async with sm() as db:
            run = (
                await db.execute(select(RunORM).where(RunORM.id == claim.run_id))
            ).scalar_one()
        assert run.status == "queued" and run.trigger == "threshold"
        assert run.session_ids == sids
        for sid in sids:
            rows = await _rows(sm, scope, sid)
            assert dt.datetime.fromisoformat(run.session_watermarks[sid]) == as_utc(
                rows[-1].created_at
            )
        assert claim.state.last_fired_tied_session_ids == sids

    async def test_claimed_sessions_are_not_counted_again(self, m, scope):
        await _ended_sessions(m, scope, 5)
        assert (await _check(m, scope)).fired is True
        again = await _check(m, scope)
        assert again.fired is False and again.pending_count == 0

    async def test_two_live_runs_never_share_a_session(self, sm, m, scope):
        first = await _ended_sessions(m, scope, 5, prefix="a")
        run_a = await _check(m, scope)
        second = await _ended_sessions(m, scope, 5, prefix="b")
        run_b = await _check(m, scope)
        assert run_a.fired and run_b.fired
        assert run_a.session_ids == first and run_b.session_ids == second
        # A flush now finds nothing claimable: both live runs hold their claims.
        flush = await _check(m, scope, force=True, trigger="flush")
        assert flush.fired is False
        async with sm() as db:
            runs = (
                (
                    await db.execute(
                        select(RunORM).where(RunORM.user_id == scope.user.id)
                    )
                )
                .scalars()
                .all()
            )
        claimed = [sid for run in runs for sid in run.session_ids]
        assert len(claimed) == len(set(claimed)) == 10

    async def test_concurrent_checks_fire_exactly_once(self, m, scope):
        await _ended_sessions(m, scope, 5)
        claims = await asyncio.gather(*[_check(m, scope) for _ in range(8)])
        fired = [c for c in claims if c.fired]
        assert len(fired) == 1
        assert len(fired[0].session_ids) == 5

    async def test_conversation_sessions_never_count(self, m, scope):
        await _ended_sessions(m, scope, 6, tag="conversation")
        claim = await _check(m, scope)
        assert claim.fired is False and claim.pending_count == 0
        assert (await _check(m, scope, force=True)).fired is False

    async def test_idle_sessions_are_materialized_and_counted(
        self, sm, m, scope, clock
    ):
        for i in range(5):
            await _write(m, scope, f"idle-{i}")
        assert (await _check(m, scope)).fired is False  # all still open
        clock.advance(seconds=IDLE + 1)
        claim = await _check(m, scope)
        assert claim.fired is True and len(claim.session_ids) == 5
        row = await _session_row(sm, scope, "idle-0")
        assert row.close_reason == "idle"

    async def test_flush_fires_with_fewer_than_the_threshold(self, m, scope):
        sids = await _ended_sessions(m, scope, 3)
        claim = await _check(m, scope, force=True, trigger="flush")
        assert claim.fired is True and claim.session_ids == sids

    async def test_flush_with_nothing_pending_creates_no_run(self, m, scope):
        assert (await _check(m, scope, force=True, trigger="flush")).fired is False

    async def test_flush_curates_experiences_left_without_a_pending_session(
        self, sm, m, scope
    ):
        # A run that crashed after its distill commit (or whose curator
        # stopped on an infrastructure error, or that failed) leaves
        # experiences pending while every session is distilled. A flush is a
        # trigger (the curator's leftovers are retried at the next one):
        # it starts a run without sessions for them.
        await _pending_experiences(sm, scope, 2)
        claim = await _check(m, scope, force=True, trigger="flush")
        assert claim.fired is True and claim.session_ids == []
        assert claim.pending_count == 0 and claim.pending_experiences == 2
        run = await m.runs.get_run(claim.run_id)
        assert run.trigger == "flush" and run.status == "queued"
        assert run.session_ids == []
        # That run curates them: a second flush meanwhile starts none.
        again = await _check(m, scope, force=True, trigger="flush")
        assert again.fired is False and again.pending_experiences == 2
        # Experiences never count toward the threshold of the automatic trigger.
        await m.runs.succeed(claim.run_id)
        assert (await _check(m, scope, threshold=1)).fired is False

    async def test_flush_ignores_consumed_or_foreign_experiences(
        self, sm, m, scope, org_id
    ):
        await _pending_experiences(sm, scope, 1, status="consumed")
        other = await _new_scope(sm, org_id)
        await _pending_experiences(sm, other, 1)
        claim = await _check(m, scope, force=True, trigger="flush")
        assert claim.fired is False and claim.pending_experiences == 0

    async def test_manual_run_is_recorded_even_when_empty(self, m, scope):
        claim = await _check(m, scope, force=True, allow_empty=True, trigger="manual")
        assert claim.fired is True and claim.session_ids == []

    async def test_limit_claims_the_oldest(self, m, scope):
        sids = await _ended_sessions(m, scope, 4)
        claim = await _check(m, scope, force=True, limit=2)
        assert claim.session_ids == sids[:2]

    async def test_threshold_is_configurable(self, m, scope):
        await _ended_sessions(m, scope, 2)
        assert (await _check(m, scope, threshold=3)).fired is False
        await _ended_sessions(m, scope, 1, prefix="third")
        assert (await _check(m, scope, threshold=3)).fired is True

    async def test_invalid_threshold_rejected(self, m, scope):
        with pytest.raises(ValueError):
            await _check(m, scope, threshold=0)

    async def test_default_threshold_comes_from_the_setting(
        self, m, scope, monkeypatch
    ):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "SKILL_TRIGGER_SESSION_THRESHOLD", 2)
        await _ended_sessions(m, scope, 2)
        claim = await m.trigger.check_and_claim_fire(
            agent_id=scope.agent, user_id=scope.user.id, organization_id=scope.org
        )
        assert claim.fired is True


# --------------------------------------------------------------------------- #
# 2.3  Learning runs, lease, heartbeat                                         #
# --------------------------------------------------------------------------- #


class TestLearningRuns:
    async def _claim(self, m, scope, n=5):
        await _ended_sessions(m, scope, n)
        claim = await _check(m, scope, force=True)
        assert claim.fired
        return claim

    async def test_status_transitions_are_conditional(self, m, scope):
        claim = await self._claim(m, scope)
        assert await m.runs.mark_running(claim.run_id) is True
        assert await m.runs.mark_running(claim.run_id) is False  # not queued any more
        assert await m.runs.succeed(claim.run_id) is True
        assert await m.runs.fail(claim.run_id, "late") is False  # terminal stays
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded" and run.error is None
        assert run.started_at is not None and run.finished_at is not None

    async def test_lease_expiry_fails_the_run_and_releases_its_sessions(
        self, m, scope, clock
    ):
        claim = await self._claim(m, scope)
        assert (await _check(m, scope, force=True)).fired is False  # all claimed
        # Recovery is global (other tests' runs share this database), so only
        # this run's membership is asserted.
        clock.advance(seconds=299)
        assert claim.run_id not in await m.runs.recover_expired_leases()
        clock.advance(seconds=2)
        assert claim.run_id in await m.runs.recover_expired_leases()
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and run.error == "lease_expired"
        again = await _check(m, scope)
        assert again.fired is True and again.session_ids == claim.session_ids

    async def test_heartbeat_keeps_a_running_run_alive(self, m, scope, clock):
        claim = await self._claim(m, scope)
        await m.runs.mark_running(claim.run_id)
        clock.advance(seconds=250)
        assert await m.runs.heartbeat(claim.run_id) is True
        clock.advance(seconds=250)
        assert claim.run_id not in await m.runs.recover_expired_leases()
        assert (await m.runs.get_run(claim.run_id)).status == "running"

    async def test_execute_heartbeats_and_records_success(self, m, scope, clock):
        claim = await self._claim(m, scope)
        created = (await m.runs.get_run(claim.run_id)).heartbeat_at

        async def work():
            assert await m.runs.mark_running(claim.run_id)
            clock.advance(seconds=30)
            await asyncio.sleep(0.08)  # several 0.01 s heartbeats
            return "done"

        assert await m.runs.execute(claim.run_id, work) == "done"
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        assert as_utc(run.heartbeat_at) > as_utc(created)

    async def test_execute_records_the_error_and_reraises(self, m, scope):
        claim = await self._claim(m, scope)

        async def work():
            await m.runs.mark_running(claim.run_id)
            raise RuntimeError("curator exploded")

        with pytest.raises(RuntimeError):
            await m.runs.execute(claim.run_id, work)
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed"
        assert "curator exploded" in run.error
        # A failed run releases its claims.
        assert (await _check(m, scope, force=True)).fired is True

    async def test_execute_stops_work_when_the_lease_is_lost(self, m, scope, clock):
        claim = await self._claim(m, scope)
        stopped = asyncio.Event()

        async def work():
            await m.runs.mark_running(claim.run_id)
            clock.advance(seconds=400)
            await m.runs.recover_expired_leases()  # "another process" reclaims it
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                stopped.set()
                raise

        with pytest.raises(LearningRunLeaseLost):
            await m.runs.execute(claim.run_id, work)
        assert stopped.is_set()
        assert (await m.runs.get_run(claim.run_id)).error == "lease_expired"

    @pytest.mark.parametrize("outcome", ["succeeds", "fails"])
    async def test_execute_counts_the_works_chunked_embedding_calls(
        self, m, scope, outcome
    ):
        # Every embedding call of the run whose input had to be
        # chunked lands in embedding_chunked_calls — the work's own calls and
        # those of tasks it starts, whatever the run's outcome.
        from mirix.embedding_guard import GuardedEmbedding

        class Inner:
            async def get_text_embedding(self, text):
                return [1.0, float(len(text))]

        class Words:
            def count_text(self, text):
                return len(text.split())

            async def acount_text(self, text):
                return len(text.split())

        model = GuardedEmbedding(
            Inner(), model_name="m", token_limits={"m": 100}, token_counter=Words()
        )
        long_text = " ".join(["word"] * 400)
        claim = await self._claim(m, scope)

        async def work():
            await m.runs.mark_running(claim.run_id)
            await model.get_text_embedding(long_text)  # chunked
            await model.get_text_embedding("short")  # not chunked
            await asyncio.gather(
                model.get_text_embedding(long_text),
                model.get_text_embedding(long_text),
            )
            if outcome == "fails":
                raise RuntimeError("after the embeddings")
            return "ok"

        if outcome == "fails":
            with pytest.raises(RuntimeError):
                await m.runs.execute(claim.run_id, work)
        else:
            assert await m.runs.execute(claim.run_id, work) == "ok"
        run = await m.runs.get_run(claim.run_id)
        assert run.embedding_chunked_calls == 3
        assert run.status == ("succeeded" if outcome == "succeeds" else "failed")
        # Outside the run nothing is counted.
        await model.get_text_embedding(long_text)
        assert (await m.runs.get_run(claim.run_id)).embedding_chunked_calls == 3

    async def test_metrics_append(self, m, scope):
        claim = await self._claim(m, scope)
        await m.runs.append_distill_metrics(claim.run_id, [{"session_id": "a"}])
        await m.runs.append_distill_metrics(claim.run_id, [{"session_id": "b"}])
        await m.runs.append_curator_metrics(claim.run_id, [{"batch_id": "x:b0"}])
        await m.runs.add_llm_usage(
            claim.run_id, {"prompt_tokens": 10, "by_model": {"m": 1}}
        )
        await m.runs.add_llm_usage(
            claim.run_id, {"prompt_tokens": 5, "by_model": {"m": 2}}
        )
        await m.runs.add_embedding_chunked_calls(claim.run_id, 3)
        run = await m.runs.get_run(claim.run_id)
        assert [e["session_id"] for e in run.distill_metrics] == ["a", "b"]
        assert run.curator_metrics == [{"batch_id": "x:b0"}]
        assert run.llm_usage == {"prompt_tokens": 15, "by_model": {"m": 3}}
        assert run.embedding_chunked_calls == 3

    async def test_list_runs_filters(self, m, scope):
        claim = await self._claim(m, scope)
        active = await m.runs.list_runs(
            organization_id=scope.org, user_id=scope.user.id, status="active"
        )
        assert [r.id for r in active] == [claim.run_id]
        assert await m.runs.list_runs(organization_id=scope.org, agent_ids=[]) == []
        assert (
            await m.runs.list_runs(
                organization_id=scope.org, user_id=scope.user.id, status="succeeded"
            )
            == []
        )

    async def test_scheduled_task_is_retained_until_it_finishes(self):
        gate = asyncio.Event()

        async def job():
            await gate.wait()

        task = SkillLearningRunManager.schedule(job(), name="retained")
        assert task in background_learning_tasks()
        gate.set()
        await task
        await asyncio.sleep(0)
        assert task not in background_learning_tasks()


# --------------------------------------------------------------------------- #
# Procedural auto-dream: executing a claimed run                               #
# --------------------------------------------------------------------------- #


class _FakeLLM:
    """Scripted distiller completion: one experience per call."""

    def __init__(self):
        self.calls = []

    async def send_llm_request(self, messages):
        self.calls.append(messages[-1].content[0].text)
        content = json.dumps(
            [
                {
                    "experience_type": "worth_learning",
                    "title": f"lesson {len(self.calls)}",
                    "content": "do the thing",
                    "importance": 0.5,
                    "credibility": 0.5,
                    "evidence": {"quote": "ok", "signal_type": "inferred"},
                }
            ]
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


@pytest.fixture
def dream_env(sm, monkeypatch):
    """Route every manager AutoDreamManager builds to the hermetic database and
    replace the model-facing / server-facing steps with recorders."""
    import mirix.server.server as server_module
    from mirix.services.auto_dream_manager import AutoDreamManager
    from mirix.services.session_experience_distiller import SessionExperienceDistiller

    monkeypatch.setattr(server_module, "db_context", sm)
    env = SimpleNamespace(
        llm=_FakeLLM(),
        curate_calls=[],
        scheduled=[],
        # The changes the stubbed curator reports, one dict per batch (the
        # run's response must merge all of them).
        batch_changes=[{"created": ["skill-b0"], "edited": ["skill-shared"]}],
    )

    def make_distiller(self, *, llm_config, conversation_manager):
        env.llm_config = llm_config
        return SessionExperienceDistiller(
            llm_client=env.llm, conversation_manager=conversation_manager
        )

    async def curate(self, *, run_id, user, actor, meta_agent_state, model=None):
        from mirix.services.skill_experience_curator import CuratorOutcome
        from mirix.services.skill_experience_manager import SkillExperienceManager

        pending = await SkillExperienceManager().list_pending_for_curation(
            agent_id=meta_agent_state.id, user_id=user.id
        )
        env.curate_calls.append(
            {"run_id": run_id, "pending": [e.id for e in pending], "model": model}
        )
        outcome = CuratorOutcome(experiences=len(pending))
        for changes in env.batch_changes:
            outcome.merge(changes)
        return outcome

    async def no_checkpoint(self, user, actor, meta_agent_state, dream_time):
        return None

    monkeypatch.setattr(AutoDreamManager, "_make_distiller", make_distiller)
    monkeypatch.setattr(AutoDreamManager, "_curate", curate)
    monkeypatch.setattr(AutoDreamManager, "write_checkpoint", no_checkpoint)

    def fake_schedule(run_id, **context):
        env.scheduled.append(run_id)
        env.scheduled_context = context
        return None

    monkeypatch.setattr(skill_learning_trigger, "schedule_learning_run", fake_schedule)
    return env


@pytest.fixture
def real_clock_managers(sm):
    """Managers on the real clock (AutoDreamManager builds its own)."""
    return _managers(sm, None)


async def _skill_without_vector(sm, scope, name):
    """A skill of the scope's user stored without a retrieval vector (an
    embedding outage when the curator wrote it)."""
    from mirix.orm.procedural_memory import ProceduralMemoryItem as SkillORM
    from mirix.schemas.embedding_config import EmbeddingConfig

    skill_id = f"proc-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(
            SkillORM(
                id=skill_id,
                organization_id=scope.org,
                user_id=scope.user.id,
                agent_id=scope.agent,
                name=name,
                entry_type="workflow",
                description=f"How to {name}",
                instructions="Do it.",
                triggers=[f"when {name}"],
                examples=[],
                filter_tags={"scope": "test"},
                version="0.1.0",
                embedding_config=EmbeddingConfig(
                    embedding_endpoint_type="openai",
                    embedding_endpoint="http://127.0.0.1:9/v1",
                    embedding_model="fake-embed",
                    embedding_dim=8,
                ),
            )
        )
        await db.commit()
    return skill_id


async def _vector_of(sm, skill_id):
    from mirix.orm.procedural_memory import ProceduralMemoryItem as SkillORM

    async with sm() as db:
        return (
            await db.execute(
                select(SkillORM.retrieval_embedding).where(SkillORM.id == skill_id)
            )
        ).scalar_one()


class TestProceduralRun:
    async def _run(self, scope, run_id=None, **request_kwargs):
        from mirix.services.auto_dream_manager import AutoDreamManager

        return await AutoDreamManager().run(
            request=AutoDreamRequest(mode="procedural", **request_kwargs),
            user=scope.user,
            actor=scope.actor,
            meta_agent_state=scope.meta,
            run_id=run_id,
        )

    async def test_run_processes_exactly_its_claimed_sessions(
        self, sm, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        claimed = []
        for i in range(3):
            sid = f"claimed-{i}"
            await _write(m, scope, sid)
            await m.sessions.close_session(
                organization_id=scope.org, user_id=scope.user.id, session_id=sid
            )
            claimed.append(sid)
        claim = await _check(m, scope, force=True)
        # Ended AFTER the claim: pending, but not this run's.
        await _write(m, scope, "late")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="late"
        )

        response = await self._run(scope, run_id=claim.run_id)
        assert response.run_ids == [claim.run_id]
        assert response.processed["procedural"].total == 3
        assert len(dream_env.llm.calls) == 3
        for sid in claimed:
            assert all(r.distilled_at is not None for r in await _rows(sm, scope, sid))
        assert all(r.distilled_at is None for r in await _rows(sm, scope, "late"))

        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        assert sorted(e["session_id"] for e in run.distill_metrics) == sorted(claimed)
        assert all(
            e["status"] == "ok" and e["messages"] == 2 for e in run.distill_metrics
        )
        # The curator stage ran for this run over the 3 fresh experiences.
        [curated] = dream_env.curate_calls
        assert curated["run_id"] == claim.run_id
        assert len(curated["pending"]) == 3
        # The late session is claimable by the next run.
        assert (await _check(m, scope, force=True)).session_ids == ["late"]

    async def test_reopen_during_a_run_keeps_rows_after_the_watermark_pending(
        self, sm, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "reopened", n=2)
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="reopened"
        )
        claim = await _check(m, scope, force=True)
        # The user comes back before the run executes: the session reopens.
        again = await _write(m, scope, "reopened", n=1, text="after claim")
        assert again.reopened is True

        await self._run(scope, run_id=claim.run_id)
        rows = await _rows(sm, scope, "reopened")
        assert [r.distilled_at is not None for r in rows] == [True, True, False]
        # The distiller never saw the post-claim row.
        assert "after claim" not in dream_env.llm.calls[0]
        # Still open → not pending; once it ends again only the new row is
        # pending. (The stub curator left the run's experiences pending: a
        # flush meanwhile claims no session, only curates them.)
        still_open = await _check(m, scope, force=True)
        assert still_open.session_ids == [] and still_open.pending_count == 0
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="reopened"
        )
        next_claim = await _check(m, scope, force=True)
        assert next_claim.session_ids == ["reopened"]
        dream_env.llm.calls.clear()
        await self._run(scope, run_id=next_claim.run_id)
        assert "after claim" in dream_env.llm.calls[0]
        assert "turn 0" not in dream_env.llm.calls[0]
        assert all(
            r.distilled_at is not None for r in await _rows(sm, scope, "reopened")
        )

    async def test_manual_run_claims_everything_pending_and_waits(
        self, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        for i in range(2):
            await _write(m, scope, f"manual-{i}")
            await m.sessions.close_session(
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id=f"manual-{i}",
            )
        response = await self._run(scope)
        [run_id] = response.run_ids
        run = await m.runs.get_run(run_id)
        assert run.trigger == "manual" and run.status == "succeeded"
        assert sorted(run.session_ids) == ["manual-0", "manual-1"]
        assert dream_env.scheduled == []

    async def test_manual_run_without_wait_returns_the_run_id_at_once(
        self, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "nowait")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="nowait"
        )
        response = await self._run(scope, wait=False)
        [run_id] = response.run_ids
        assert dream_env.scheduled == [run_id]
        assert dream_env.llm.calls == []
        run = await m.runs.get_run(run_id)
        assert run.status == "queued" and run.session_ids == ["nowait"]

    async def test_manual_run_with_nothing_pending_still_drains_leftovers(
        self, sm, dream_env, real_clock_managers, scope
    ):
        from mirix.services.skill_experience_manager import SkillExperienceManager

        mgr = SkillExperienceManager()
        mgr.session_maker = sm
        old = await mgr.create_experience(
            agent_id=scope.meta.id,
            user_id=scope.user.id,
            organization_id=scope.org,
            session_id="an-earlier-run",
            experience_type="worth_learning",
            title="left pending by an earlier run",
        )
        response = await self._run(scope)
        [run_id] = response.run_ids
        # No session to distill, yet the curator stage runs over everything
        # still pending in the scope (all pending, not just this run's).
        assert dream_env.curate_calls == [
            {"run_id": run_id, "pending": [old.id], "model": None}
        ]
        run = await real_clock_managers.runs.get_run(run_id)
        assert run.session_ids == [] and run.status == "succeeded"

    async def test_dry_run_counts_without_side_effects(
        self, sm, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "dry")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="dry"
        )
        response = await self._run(scope, dry_run=True)
        assert response.dry_run is True and response.run_ids == []
        assert "1 pending session" in response.message
        assert (
            await m.runs.list_runs(organization_id=scope.org, user_id=scope.user.id)
            == []
        )

    async def test_k15_recovery_changes_are_kept(
        self, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "k15")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="k15"
        )
        dream_env.batch_changes = [
            {"created": ["skill-b0"], "edited": ["skill-shared"]},
            {"created": ["skill-b1"], "edited": ["skill-shared"], "deleted": ["skill-x"]},
        ]
        response = await self._run(scope)
        assert len(dream_env.curate_calls) == 1
        # Every batch's changes are in the response, not only the first's.
        assert response.changes["created"] == ["skill-b0", "skill-b1"]
        assert response.changes["edited"] == ["skill-shared"]
        assert response.changes["deleted"] == ["skill-x"]
        assert response.skills_changed == 4

    async def test_k16_model_override_reaches_distiller_and_curator(
        self, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "k16")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="k16"
        )
        await self._run(scope, model="override-model")
        assert dream_env.llm_config.model == "override-model"
        assert scope.meta.llm_config.model == "base-model"  # not mutated
        assert dream_env.curate_calls[0]["model"] == "override-model"

    def _embeddings(self, monkeypatch, *, down: bool):
        """The fake provider behind every embedding model the pass builds;
        the embedding retry backoff without waiting."""
        import dataclasses

        import mirix.services.procedural_memory_manager as pmm
        from mirix.services import retry_policy

        calls = []

        class Model:
            async def get_text_embedding(self, text):
                calls.append(text)
                if down:
                    raise ConnectionError("embedding endpoint down")
                return [0.5] * 8

        async def embedding_model(config, *args, **kwargs):
            return Model()

        monkeypatch.setattr(pmm, "embedding_model", embedding_model)
        policy = retry_policy.RETRY_POLICIES[retry_policy.ErrorKind.EMBEDDING]
        monkeypatch.setitem(
            retry_policy.RETRY_POLICIES,
            retry_policy.ErrorKind.EMBEDDING,
            dataclasses.replace(policy, base_delay=0.0, max_delay=0.0),
        )
        return calls

    async def test_a_run_ends_by_re_embedding_what_its_curator_left_without_a_vector(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        """The whole run — distill, curate, then the
        end-of-run re-embedding pass — gives a skill the curator
        stored without a vector its vector, and records the pass on the run
        as its ``reembed`` curator entry."""
        from mirix.services.auto_dream_manager import AutoDreamManager

        m = real_clock_managers
        calls = self._embeddings(monkeypatch, down=False)
        stubbed_curate = AutoDreamManager._curate
        written = []

        async def curate_then_store_without_vector(self, **kwargs):
            outcome = await stubbed_curate(self, **kwargs)
            written.append(
                await _skill_without_vector(sm, scope, "stored-in-an-outage")
            )
            return outcome

        monkeypatch.setattr(
            AutoDreamManager, "_curate", curate_then_store_without_vector
        )
        await _write(m, scope, "n3")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="n3"
        )
        claim = await _check(m, scope, force=True)
        await self._run(scope, run_id=claim.run_id)

        [skill_id] = written
        assert await _vector_of(sm, skill_id) is not None
        assert len(calls) == 1
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        [entry] = [e for e in run.curator_metrics if e.get("stage") == "reembed"]
        assert entry["batch_id"] == f"{claim.run_id}:reembed"
        assert entry["filled"] == 1 and entry["still_missing"] == 0
        assert entry["stopped_early"] is False and "error_kind" not in entry

    async def test_a_configuration_skip_of_the_curator_is_recorded_on_the_run(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        """The procedural agent lacks a curator tool: the curator logs an
        error and returns skipped='curator_tools_missing'. The run used to
        read as a normal run that changed nothing, and the experiences waited
        forever unseen; the run now records the skip (error_kind config)."""
        from mirix.services.auto_dream_manager import AutoDreamManager
        from mirix.services.skill_experience_curator import CuratorOutcome

        async def tools_missing(self, **kwargs):
            return CuratorOutcome(skipped="curator_tools_missing", experiences=3)

        monkeypatch.setattr(AutoDreamManager, "_curate", tools_missing)
        m = real_clock_managers
        await _write(m, scope, "skip1")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="skip1"
        )
        claim = await _check(m, scope, force=True)
        await self._run(scope, run_id=claim.run_id)
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        [entry] = [e for e in run.curator_metrics if e.get("stage") == "curate"]
        assert entry["batch_id"] == f"{claim.run_id}:curate"
        assert entry["status"] == "skipped"
        assert entry["skipped"] == "curator_tools_missing"
        assert entry["error_kind"] == "config" and entry["experiences"] == 3
        assert "3 pending experience(s) stay pending" in entry["error"]

    async def test_nothing_pending_is_no_skip_record(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        from mirix.services.auto_dream_manager import AutoDreamManager
        from mirix.services.skill_experience_curator import (
            CONFIGURATION_SKIPS,
            CuratorOutcome,
            curation_skip_entry,
        )

        async def nothing(self, **kwargs):
            return CuratorOutcome(skipped="no_pending_experiences")

        monkeypatch.setattr(AutoDreamManager, "_curate", nothing)
        m = real_clock_managers
        await _write(m, scope, "skip2")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="skip2"
        )
        claim = await _check(m, scope, force=True)
        await self._run(scope, run_id=claim.run_id)
        run = await m.runs.get_run(claim.run_id)
        assert not [e for e in run.curator_metrics or [] if e.get("stage") == "curate"]
        assert "no_pending_experiences" not in CONFIGURATION_SKIPS
        for reason in CONFIGURATION_SKIPS:
            entry = curation_skip_entry(
                "r", CuratorOutcome(skipped=reason, experiences=1), started_at="t"
            )
            assert entry["error_kind"] == "config" and entry["skipped"] == reason

    async def test_an_outage_stops_the_end_of_run_pass_and_is_recorded(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        """During a sustained embedding outage the
        end-of-run pass stops after MIRIX_SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES
        skills in a row failed; the run still succeeds and its reembed entry
        says what was left (error_kind embedding: an infrastructure
        problem the harness does not ignore)."""
        from mirix import constants

        m = real_clock_managers
        calls = self._embeddings(monkeypatch, down=True)
        monkeypatch.setattr(constants, "SKILL_REEMBED_MAX_CONSECUTIVE_FAILURES", 2)
        monkeypatch.setattr(constants, "SKILL_REEMBED_CONCURRENCY", 1)
        ids = [await _skill_without_vector(sm, scope, f"outage-{i}") for i in range(6)]
        await _write(m, scope, "n4")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="n4"
        )
        claim = await _check(m, scope, force=True)
        await self._run(scope, run_id=claim.run_id)

        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        [entry] = [e for e in run.curator_metrics if e.get("stage") == "reembed"]
        assert entry["error_kind"] == "embedding"
        assert entry["stopped_early"] is True
        assert entry["outcomes"] == {"failed": 2, "skipped": 4}
        assert entry["still_missing"] == 6
        assert "did not try 4 skill(s)" in entry["error"]
        # 2 skills x (1 + 3 retries), not 6 x 4.
        assert len(calls) == 2 * 4
        assert [e["kind"] for e in run.retry_events].count("embedding") == 2 * 3
        for skill_id in ids:
            assert await _vector_of(sm, skill_id) is None

    async def test_a_failing_run_is_recorded_and_releases_its_sessions(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        from mirix.services.auto_dream_manager import AutoDreamManager

        m = real_clock_managers
        await _write(m, scope, "boom")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="boom"
        )
        claim = await _check(m, scope, force=True)

        async def broken_checkpoint(self, user, actor, meta_agent_state, dream_time):
            raise RuntimeError("checkpoint store down")

        monkeypatch.setattr(AutoDreamManager, "write_checkpoint", broken_checkpoint)
        with pytest.raises(RuntimeError):
            await self._run(scope, run_id=claim.run_id)
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and "checkpoint store down" in run.error

    async def test_a_run_of_another_scope_is_refused(
        self, sm, dream_env, org_id, scope
    ):
        other = await _new_scope(sm, org_id)
        m = _managers(sm, None)
        await _write(m, other, "theirs")
        await m.sessions.close_session(
            organization_id=other.org, user_id=other.user.id, session_id="theirs"
        )
        claim = await _check(m, other, force=True)
        with pytest.raises(ValueError):
            await self._run(scope, run_id=claim.run_id)
        assert (await m.runs.get_run(claim.run_id)).status == "queued"

    async def test_executing_a_reclaimed_run_does_nothing(
        self, dream_env, real_clock_managers, scope
    ):
        from mirix.services.skill_learning_run_manager import LearningRunNotActive

        m = real_clock_managers
        await _write(m, scope, "reclaimed")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="reclaimed"
        )
        claim = await _check(m, scope, force=True)
        await m.runs.fail(claim.run_id, "lease_expired")
        with pytest.raises(LearningRunNotActive):
            await self._run(scope, run_id=claim.run_id)
        assert dream_env.llm.calls == []


# --------------------------------------------------------------------------- #
# Orchestration: trigger + schedule, flush, background execution, sweeper      #
# --------------------------------------------------------------------------- #


class TestOrchestration:
    async def test_trigger_schedules_the_fired_run(self, m, scope, monkeypatch):
        scheduled = []
        monkeypatch.setattr(
            skill_learning_trigger,
            "schedule_learning_run",
            lambda run_id, **ctx: scheduled.append((run_id, ctx)),
        )
        await _ended_sessions(m, scope, 5)
        claim = await skill_learning_trigger.trigger_skill_learning(
            organization_id=scope.org,
            user_id=scope.user.id,
            agent_id=scope.agent,
            actor=scope.actor,
            trigger_manager=m.trigger,
        )
        assert claim.fired
        assert scheduled == [
            (
                claim.run_id,
                {"actor": scope.actor, "user": None, "meta_agent_state": None},
            )
        ]

    async def test_trigger_without_a_scope_does_nothing(self):
        claim = await skill_learning_trigger.trigger_skill_learning(
            organization_id="org", user_id="user", agent_id=None
        )
        assert claim.fired is False

    async def test_flush_closes_open_sessions_first(self, m, scope, monkeypatch):
        monkeypatch.setattr(
            skill_learning_trigger, "schedule_learning_run", lambda run_id, **ctx: None
        )
        ended = await _ended_sessions(m, scope, 2)
        await _write(m, scope, "still-open")
        result = await skill_learning_trigger.flush_skills(
            organization_id=scope.org,
            user_id=scope.user.id,
            agent_id=scope.agent,
            close_open_sessions=True,
            session_manager=m.sessions,
            trigger_manager=m.trigger,
        )
        assert result.closed_session_ids == ["still-open"]
        assert result.session_count == 3 and result.learning_run_id
        empty = await skill_learning_trigger.flush_skills(
            organization_id=scope.org,
            user_id=scope.user.id,
            agent_id=scope.agent,
            session_manager=m.sessions,
            trigger_manager=m.trigger,
        )
        assert empty.learning_run_id is None and empty.session_count == 0
        assert set(ended) < set(
            (await m.runs.get_run(result.learning_run_id)).session_ids
        )

    async def test_background_run_is_executed_and_errors_are_contained(
        self, monkeypatch
    ):
        seen = []

        async def fake_execute(run_id, **ctx):
            seen.append(run_id)
            raise RuntimeError("model down")

        monkeypatch.setattr(
            skill_learning_trigger, "execute_learning_run", fake_execute
        )
        task = skill_learning_trigger.schedule_learning_run("slrun-x")
        assert task in background_learning_tasks()
        await task  # the error is logged, not raised
        assert seen == ["slrun-x"]

    async def test_cancel_background_runs(self, monkeypatch):
        started = asyncio.Event()

        async def slow_execute(run_id, **ctx):
            started.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(
            skill_learning_trigger, "execute_learning_run", slow_execute
        )
        task = skill_learning_trigger.schedule_learning_run("slrun-slow")
        await started.wait()
        assert (
            await skill_learning_trigger.cancel_background_learning_runs(timeout=1) >= 1
        )
        assert task.cancelled()


class TestSweeper:
    async def test_one_pass_reclaims_materializes_and_fires(
        self, fresh_sm, clock, monkeypatch
    ):
        sm = fresh_sm
        org_id = f"org-{uuid.uuid4().hex[:8]}"
        async with sm() as db:
            db.add(OrganizationORM(id=org_id, name=org_id))
            await db.commit()
        m = _managers(sm, clock)
        scope = await _new_scope(sm, org_id)
        scheduled = []
        monkeypatch.setattr(
            skill_learning_trigger,
            "schedule_learning_run",
            lambda run_id, **ctx: scheduled.append(run_id),
        )
        # A stale run from a "crashed process" holds 5 sessions.
        stale = await _ended_sessions(m, scope, 5, prefix="stale")
        stale_run = await _check(m, scope)
        # Another scope: 5 sessions that ended by idling; one scope below N.
        idle_scope = await _new_scope(sm, org_id)
        for i in range(5):
            await _write(m, idle_scope, f"idle-{i}")
        small = await _new_scope(sm, org_id)
        await _ended_sessions(m, small, 2, prefix="small")
        clock.advance(seconds=IDLE + 1)

        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=60, session_maker=sm, clock=clock
        )
        result = await sweeper.run_once()
        assert result.reclaimed_run_ids == [stale_run.run_id]
        assert result.materialized_sessions == 5
        assert result.scopes_checked == 3
        fired = {(await m.runs.get_run(r)).user_id: r for r in result.fired_run_ids}
        assert set(fired) == {scope.user.id, idle_scope.user.id}
        assert (await m.runs.get_run(fired[scope.user.id])).session_ids == stale
        assert (await m.runs.get_run(fired[scope.user.id])).trigger == "sweep"
        assert sorted(scheduled) == sorted(result.fired_run_ids)
        # Below threshold stays pending.
        assert (await _check(m, small)).fired is False

    async def test_a_failing_scope_does_not_stop_the_sweep(
        self, sm, m, scope, org_id, clock
    ):
        other = await _new_scope(sm, org_id)
        await _ended_sessions(m, scope, 1, prefix="a")
        await _ended_sessions(m, other, 1, prefix="b")
        seen = []

        async def flaky(**kwargs):
            seen.append(kwargs["user_id"])
            if kwargs["user_id"] == scope.user.id:
                raise RuntimeError("scope broke")
            return skill_learning_trigger.ClaimFireResult(fired=False)

        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=60, session_maker=sm, clock=clock, trigger_fn=flaky
        )
        await sweeper.run_once()
        assert scope.user.id in seen and other.user.id in seen

    async def test_loop_runs_passes_and_stops_cleanly(self, sm, clock):
        naps = []

        async def fake_sleep(seconds):
            naps.append(seconds)
            await asyncio.sleep(0)

        async def no_fire(**kwargs):
            return skill_learning_trigger.ClaimFireResult(fired=False)

        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=3600,
            lease_interval_seconds=150,
            session_maker=sm,
            clock=clock,
            sleep=fake_sleep,
            trigger_fn=no_fire,
        )
        task = sweeper.start()
        assert sweeper.start() is task  # idempotent
        while sweeper.passes < 3 or sweeper.lease_passes < 3:
            await asyncio.sleep(0.01)
        await sweeper.stop()
        assert task.done() and not sweeper.running
        # Two cadences: the sweep, and lease recovery every half lease.
        assert naps and set(naps) == {3600, 150}
        await sweeper.stop()  # stopping twice is fine

    async def test_disabled_sweep_still_recovers_leases(self, fresh_sm, clock):
        """MIRIX_SESSION_SWEEP_INTERVAL_SECONDS <= 0 turns the sweep off, but
        lease recovery still runs at startup and then every half lease: a
        crashed run must not stay 'running' forever."""
        sm = fresh_sm
        org_id = f"org-{uuid.uuid4().hex[:8]}"
        async with sm() as db:
            db.add(OrganizationORM(id=org_id, name=org_id))
            await db.commit()
        m = _managers(sm, clock)
        scope = await _new_scope(sm, org_id)
        await _ended_sessions(m, scope, 2, prefix="crash")
        crashed = await _check(m, scope, force=True)  # its process then dies
        naps = []
        gate = asyncio.Event()

        async def fake_sleep(seconds):
            naps.append(seconds)
            await gate.wait()  # one pass per test step
            gate.clear()

        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=0,
            lease_interval_seconds=150,
            session_maker=sm,
            clock=clock,
            sleep=fake_sleep,
        )
        assert sweeper.start() is None and not sweeper.sweeping
        assert sweeper.running  # the lease loop
        while sweeper.lease_passes < 1:
            await asyncio.sleep(0.01)
        # Startup pass: the heartbeat is fresh, nothing is reclaimed yet.
        assert (await m.runs.get_run(crashed.run_id)).status == "queued"
        clock.advance(seconds=301)
        gate.set()
        while sweeper.lease_passes < 2:
            await asyncio.sleep(0.01)
        run = await m.runs.get_run(crashed.run_id)
        assert run.status == "failed" and run.error == "lease_expired"
        assert naps and set(naps) == {150}
        await sweeper.stop()
        assert not sweeper.running
        # Its sessions are claimable again.
        assert (await _check(m, scope, force=True)).session_ids == crashed.session_ids

    async def test_a_failing_pass_does_not_kill_the_loop(self, sm, clock, monkeypatch):
        calls = []

        async def fake_sleep(seconds):
            await asyncio.sleep(0)

        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=1, session_maker=sm, clock=clock, sleep=fake_sleep
        )

        async def flaky_run_once():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("db hiccup")
            sweeper.passes += 1

        monkeypatch.setattr(sweeper, "run_once", flaky_run_once)
        sweeper.start()
        while len(calls) < 3:
            await asyncio.sleep(0.01)
        await sweeper.stop()
        assert len(calls) >= 3

    async def test_disabled_sweep_does_not_start_the_sweep_loop(self, sm):
        sweeper = skill_learning_trigger.SessionSweeper(
            interval_seconds=0, lease_interval_seconds=0, session_maker=sm
        )
        assert sweeper.start() is None and not sweeper.running


class TestDeletedMetaAgent:
    async def test_scopes_of_deleted_agents_are_not_swept(self, sm, m, scope):
        await _ended_sessions(m, scope, 1, prefix="orphan")
        assert (
            scope.org,
            scope.user.id,
            scope.agent,
        ) in await m.sessions.list_pending_scopes()
        async with sm() as db:
            await db.execute(
                update(AgentORM)
                .where(AgentORM.id == scope.agent)
                .values(is_deleted=True)
            )
            await db.commit()
        assert (scope.org, scope.user.id, scope.agent) not in (
            await m.sessions.list_pending_scopes()
        )


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


async def _new_org(sm):
    org = f"org-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.commit()
    return org


async def _add_client(sm, org_id, *, deleted=False):
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
                is_deleted=deleted,
            )
        )
        await db.commit()
    return PydanticClient(
        id=client_id,
        organization_id=org_id,
        name=client_id,
        write_scope="test",
        read_scopes=["test"],
    )


async def _unattribute(sm, scope, sid, *, writer="keep"):
    """Make ``sid`` look like a backfilled legacy session (agent_id NULL);
    ``writer`` optionally rewrites who wrote its rows."""
    async with sm() as db:
        await db.execute(
            update(SessionORM)
            .where(SessionORM.user_id == scope.user.id, SessionORM.session_id == sid)
            .values(agent_id=None)
        )
        if writer != "keep":
            await db.execute(
                update(MessageORM)
                .where(
                    MessageORM.user_id == scope.user.id, MessageORM.session_id == sid
                )
                .values({MessageORM._created_by_id: writer})
            )
        await db.commit()


async def _experiences(sm, scope, sid=None):
    from mirix.orm.skill_experience import SkillExperience as ExperienceORM

    async with sm() as db:
        stmt = select(ExperienceORM).where(ExperienceORM.user_id == scope.user.id)
        if sid is not None:
            stmt = stmt.where(ExperienceORM.session_id == sid)
        return list((await db.execute(stmt)).scalars().all())


class TestLegacyAttribution:
    """Sessions backfilled without a meta agent become learnable."""

    async def test_a_claim_adopts_a_session_its_client_wrote(self, sm, m, scope):
        [sid] = await _ended_sessions(m, scope, 1, prefix="legacy")
        await _unattribute(sm, scope, sid)
        assert (await _session_row(sm, scope, sid)).agent_id is None
        claim = await _check(m, scope, force=True)
        assert claim.fired and claim.session_ids == [sid]
        assert (await _session_row(sm, scope, sid)).agent_id == scope.agent

    async def test_flush_closes_and_claims_an_open_legacy_session(
        self, sm, m, scope, monkeypatch
    ):
        monkeypatch.setattr(
            skill_learning_trigger, "schedule_learning_run", lambda run_id, **ctx: None
        )
        await _write(m, scope, "legacy-open")
        await _unattribute(sm, scope, "legacy-open")
        result = await skill_learning_trigger.flush_skills(
            organization_id=scope.org,
            user_id=scope.user.id,
            agent_id=scope.agent,
            close_open_sessions=True,
            session_manager=m.sessions,
            trigger_manager=m.trigger,
        )
        assert result.closed_session_ids == ["legacy-open"]
        assert result.session_count == 1

    async def test_another_clients_session_is_not_adopted(
        self, sm, m, scope, org_id
    ):
        other = await _new_scope(sm, org_id)
        [sid] = await _ended_sessions(m, scope, 1, prefix="theirs")
        await _unattribute(sm, scope, sid, writer=other.actor.id)
        assert (await _check(m, scope, force=True)).fired is False
        assert (await _session_row(sm, scope, sid)).agent_id is None
        # The global pass attributes it to the writer's own meta agent.
        assert await m.sessions.adopt_unattributed_sessions(user_id=scope.user.id) == 1
        assert (await _session_row(sm, scope, sid)).agent_id == other.agent

    async def test_ambiguous_sessions_stay_unattributed(self, sm, m, scope, org_id):
        other = await _add_client(sm, org_id)
        [sid] = await _ended_sessions(m, scope, 1, prefix="mixed")
        await _unattribute(sm, scope, sid)
        # A second writer (with no meta agent at all) in the same session.
        async with sm() as db:
            [row] = [
                r
                for r in (
                    await db.execute(
                        select(MessageORM).where(
                            MessageORM.user_id == scope.user.id,
                            MessageORM.session_id == sid,
                        )
                    )
                )
                .scalars()
                .all()
            ][:1]
            row._created_by_id = other.id
            await db.commit()
        assert await m.sessions.adopt_unattributed_sessions(user_id=scope.user.id) == 0
        assert (await _check(m, scope, force=True)).fired is False

    async def test_rows_without_a_writer_go_to_the_orgs_only_meta_agent(self, sm):
        clock = FakeClock()
        m = _managers(sm, clock)
        org = await _new_org(sm)
        scope = await _new_scope(sm, org)
        [sid] = await _ended_sessions(m, scope, 1, prefix="nowriter")
        await _unattribute(sm, scope, sid, writer=None)
        claim = await _check(m, scope, force=True)
        assert claim.session_ids == [sid]
        # A second meta agent in the org makes the fallback ambiguous.
        second = await _new_scope(sm, org)
        [sid2] = await _ended_sessions(m, second, 1, prefix="nowriter2")
        await _unattribute(sm, second, sid2, writer=None)
        assert await m.sessions.adopt_unattributed_sessions(organization_id=org) == 0

    async def test_the_sweeper_attributes_legacy_sessions(
        self, fresh_sm, clock, monkeypatch
    ):
        sm = fresh_sm
        org = await _new_org(sm)
        m = _managers(sm, clock)
        scope = await _new_scope(sm, org)
        sids = await _ended_sessions(m, scope, 5, prefix="swept")
        for sid in sids:
            await _unattribute(sm, scope, sid)
        monkeypatch.setattr(
            skill_learning_trigger, "schedule_learning_run", lambda run_id, **ctx: None
        )
        result = await skill_learning_trigger.SessionSweeper(
            interval_seconds=60, session_maker=sm, clock=clock
        ).run_once()
        assert result.attributed_sessions == 5
        [run_id] = result.fired_run_ids
        assert sorted((await m.runs.get_run(run_id)).session_ids) == sorted(sids)


class TestSessionOwnership:
    """A session_id is one meta agent's; another cannot write into it."""

    async def test_another_meta_agent_cannot_write_into_the_session(
        self, sm, m, scope, org_id
    ):
        from mirix.services.conversation_session_manager import (
            SessionOwnershipConflict,
        )

        other = await _new_scope(sm, org_id)
        await _write(m, scope, "owned")
        with pytest.raises(SessionOwnershipConflict):
            await m.messages.record_turns(
                session_id="owned",
                user_id=scope.user.id,
                organization_id=scope.org,
                turns=[{"role": "user", "content": "intruder"}],
                actor=other.actor,
                agent_id=other.agent,
            )
        rows = await _rows(sm, scope, "owned")
        assert [r.content for r in rows] == ["turn 0", "turn 1"]
        row = await _session_row(sm, scope, "owned")
        assert row.agent_id == scope.agent and row.message_count == 2

    async def test_a_legacy_session_written_by_another_client_is_not_adopted(
        self, sm, m, scope, org_id
    ):
        from mirix.services.conversation_session_manager import (
            SessionOwnershipConflict,
        )

        other = await _new_scope(sm, org_id)
        await _write(m, other, "legacy-foreign")
        # Same user, so the session key collides; move it to scope's user.
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(SessionORM.session_id == "legacy-foreign")
                .values(user_id=scope.user.id, agent_id=None)
            )
            await db.execute(
                update(MessageORM)
                .where(MessageORM.session_id == "legacy-foreign")
                .values(user_id=scope.user.id)
            )
            await db.commit()
        with pytest.raises(SessionOwnershipConflict):
            await _write(m, scope, "legacy-foreign", n=1)
        assert len(await _rows(sm, scope, "legacy-foreign")) == 2
        assert (await _session_row(sm, scope, "legacy-foreign")).agent_id is None


class TestReviewSessionLifecycle:
    async def test_close_of_an_idle_session_counts_as_newly_ended(
        self, m, scope, clock
    ):
        """The explicit close that records an idle end triggers."""
        await _write(m, scope, "idle-close")
        clock.advance(seconds=IDLE + 5)
        first = await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="idle-close"
        )
        assert first.already_closed is True and first.newly_ended is True
        again = await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="idle-close"
        )
        assert again.already_closed is True and again.newly_ended is False
        await _write(m, scope, "open-close")
        explicit = await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="open-close"
        )
        assert explicit.already_closed is False and explicit.newly_ended is True

    async def test_reopen_resets_the_failure_backoff(self, sm, m, scope, clock):
        """A new segment must not inherit the old segment's backoff."""
        [sid] = await _ended_sessions(m, scope, 1, prefix="backoff")
        async with sm() as db:
            await db.execute(
                update(SessionORM)
                .where(SessionORM.user_id == scope.user.id, SessionORM.session_id == sid)
                .values(
                    distill_failures=3,
                    distill_failed_at=clock.now,
                    distill_retry_after=clock.now + dt.timedelta(days=30),
                )
            )
            await db.commit()
        out = await _write(m, scope, sid, n=1, text="new segment")
        assert out.reopened is True
        row = await _session_row(sm, scope, sid)
        assert row.distill_failures == 0
        assert row.distill_failed_at is None and row.distill_retry_after is None
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id=sid
        )
        assert (await _check(m, scope, force=True)).session_ids == [sid]

    async def test_deleted_users_and_owner_clients_are_not_swept(
        self, sm, m, org_id
    ):
        """Nothing could run their learning, so the sweep skips them."""
        deleted_user = await _new_scope(sm, org_id)
        await _ended_sessions(m, deleted_user, 1, prefix="gone-user")
        deleted_client = await _new_scope(sm, org_id)
        await _ended_sessions(m, deleted_client, 1, prefix="gone-client")
        missing_client = await _new_scope(sm, org_id)
        await _ended_sessions(m, missing_client, 1, prefix="no-client")
        live = await _new_scope(sm, org_id)
        await _ended_sessions(m, live, 1, prefix="live")
        async with sm() as db:
            await db.execute(
                update(UserORM)
                .where(UserORM.id == deleted_user.user.id)
                .values(is_deleted=True)
            )
            await db.execute(
                update(ClientORM)
                .where(ClientORM.id == deleted_client.actor.id)
                .values(is_deleted=True)
            )
            agent = await db.get(AgentORM, missing_client.agent)
            agent._created_by_id = "client-that-never-existed"
            await db.commit()
        scopes = await m.sessions.list_pending_scopes()
        users = {user for _, user, _ in scopes}
        assert live.user.id in users
        assert deleted_user.user.id not in users
        assert deleted_client.user.id not in users
        assert missing_client.user.id not in users


class TestClaimSerialization:
    async def test_a_write_during_a_sqlite_claim_waits_for_it(
        self, sm, m, scope, clock, monkeypatch
    ):
        """On SQLite the claim holds the write lock from its first statement,
        so a write that reopens a claimed session lands after the claim and its
        row stays above the watermark."""
        [sid] = await _ended_sessions(m, scope, 1, prefix="race")
        real = ConversationMessageManager.session_watermarks
        observed = {}

        async def watermarks_with_a_concurrent_write(self, db, **kwargs):
            clock.advance(seconds=5)
            writer = asyncio.create_task(
                _write(m, scope, sid, n=1, text="during claim")
            )
            await asyncio.sleep(0.3)
            observed["write_done_during_claim"] = writer.done()
            observed["writer"] = writer
            return await real(self, db, **kwargs)

        monkeypatch.setattr(
            ConversationMessageManager,
            "session_watermarks",
            watermarks_with_a_concurrent_write,
        )
        claim = await _check(m, scope, force=True)
        out = await observed["writer"]
        assert observed["write_done_during_claim"] is False
        assert claim.fired and claim.session_ids == [sid]
        assert out.reopened is True
        run = await m.runs.get_run(claim.run_id)
        watermark = dt.datetime.fromisoformat(run.session_watermarks[sid])
        late = [r for r in await _rows(sm, scope, sid) if r.content == "during claim 0"]
        assert as_utc(late[0].created_at) > as_utc(watermark)

    async def test_a_claim_reclaims_its_users_expired_runs(
        self, sm, m, scope, org_id, clock
    ):
        """With the sweep disabled, a claim still frees sessions of a run
        that died (its lease expired) — only its own user's runs."""
        other = await _new_scope(sm, org_id)
        await _ended_sessions(m, other, 1, prefix="other-dead")
        other_dead = await _check(m, other, force=True)
        sids = await _ended_sessions(m, scope, 2, prefix="dead")
        dead = await _check(m, scope, force=True)
        clock.advance(seconds=301)
        again = await _check(m, scope, force=True)
        assert again.fired and sorted(again.session_ids) == sorted(sids)
        run = await m.runs.get_run(dead.run_id)
        assert run.status == "failed" and run.error == "lease_expired"
        assert (await m.runs.get_run(other_dead.run_id)).status == "queued"


class TestSettingsBounds:
    """Settings outside their bounds fall back to safe values."""

    async def test_positive_int_setting(self):
        from mirix.constants import _positive_int_setting

        assert _positive_int_setting("X", "120", 7) == 120
        assert _positive_int_setting("X", None, 7) == 7
        assert _positive_int_setting("X", " ", 7) == 7
        for bad in ("0", "-5", "abc"):
            assert _positive_int_setting("X", bad, 7) == 7

    async def test_lease_is_at_least_three_heartbeats(self):
        from mirix.constants import MIN_LEASE_HEARTBEATS, _lease_seconds_setting

        assert MIN_LEASE_HEARTBEATS == 3
        assert _lease_seconds_setting("600", 30) == 600
        assert _lease_seconds_setting("20", 30) == 90
        assert _lease_seconds_setting(None, 30) == 300
        assert _lease_seconds_setting("-1", 30) == 300


class _GatedLLM(_FakeLLM):
    """Blocks every call whose transcript contains ``marker`` until released."""

    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def send_llm_request(self, messages):
        if self.marker in messages[-1].content[0].text:
            self.blocked.set()
            await self.release.wait()
        return await super().send_llm_request(messages)


class TestAtomicDistillCommit:
    """A session's experiences and its distilled marks commit together."""

    async def test_commit_is_refused_once_the_run_is_not_active(
        self, sm, m, scope
    ):
        from mirix.services.skill_learning_run_manager import LearningRunNotActive

        [sid] = await _ended_sessions(m, scope, 1, prefix="refused")
        claim = await _check(m, scope, force=True)
        await m.runs.fail(claim.run_id, "lease_expired")
        payload = {
            "agent_id": scope.agent,
            "user_id": scope.user.id,
            "organization_id": scope.org,
            "session_id": sid,
            "experience_type": "worth_learning",
            "title": "t",
            "content": "c",
            "importance": 0.5,
            "credibility": 0.5,
            "evidence": "",
            "status": "pending",
        }
        run = await m.runs.get_run(claim.run_id)
        with pytest.raises(LearningRunNotActive):
            await m.runs.commit_distilled_session(
                claim.run_id,
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id=sid,
                watermark=dt.datetime.fromisoformat(run.session_watermarks[sid]),
                experiences=[payload],
            )
        assert await _experiences(sm, scope, sid) == []
        assert all(r.distilled_at is None for r in await _rows(sm, scope, sid))

    async def test_cancelling_a_run_mid_distill_never_duplicates(
        self, sm, dream_env, real_clock_managers, scope
    ):
        from mirix.services.auto_dream_manager import AutoDreamManager

        m = real_clock_managers
        for sid, text in (("fast", "quick"), ("slow", "blockme")):
            await _write(m, scope, sid, text=text)
            await m.sessions.close_session(
                organization_id=scope.org, user_id=scope.user.id, session_id=sid
            )
        claim = await _check(m, scope, force=True)
        assert sorted(claim.session_ids) == ["fast", "slow"]
        dream_env.llm = _GatedLLM("blockme")
        task = asyncio.create_task(
            AutoDreamManager().run(
                request=AutoDreamRequest(mode="procedural"),
                user=scope.user,
                actor=scope.actor,
                meta_agent_state=scope.meta,
                run_id=claim.run_id,
            )
        )
        await asyncio.wait_for(dream_env.llm.blocked.wait(), 5)
        for _ in range(200):  # the fast session commits on its own
            if await _experiences(sm, scope, "fast"):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and run.error == "cancelled"
        assert len(await _experiences(sm, scope, "fast")) == 1
        assert all(r.distilled_at is not None for r in await _rows(sm, scope, "fast"))
        assert await _experiences(sm, scope, "slow") == []
        assert all(r.distilled_at is None for r in await _rows(sm, scope, "slow"))

        # The next run takes only the unfinished session: one experience each.
        dream_env.llm = _FakeLLM()
        next_claim = await _check(m, scope, force=True)
        assert next_claim.session_ids == ["slow"]
        await AutoDreamManager().run(
            request=AutoDreamRequest(mode="procedural"),
            user=scope.user,
            actor=scope.actor,
            meta_agent_state=scope.meta,
            run_id=next_claim.run_id,
        )
        assert len(await _experiences(sm, scope, "fast")) == 1
        assert len(await _experiences(sm, scope, "slow")) == 1

    async def test_a_failure_before_committing_releases_the_sessions(
        self, sm, dream_env, real_clock_managers, scope, monkeypatch
    ):
        """A run that fails before any session committed is failed and
        its sessions are claimable again, rows still pending."""
        from mirix.services.auto_dream_manager import AutoDreamManager

        m = real_clock_managers
        await _write(m, scope, "early-fail")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="early-fail"
        )
        claim = await _check(m, scope, force=True)

        class BrokenDistiller:
            async def distill_sessions(self, **kwargs):
                raise RuntimeError("distiller store down")

        monkeypatch.setattr(
            AutoDreamManager, "_make_distiller", lambda self, **kw: BrokenDistiller()
        )
        with pytest.raises(RuntimeError):
            await AutoDreamManager().run(
                request=AutoDreamRequest(mode="procedural"),
                user=scope.user,
                actor=scope.actor,
                meta_agent_state=scope.meta,
                run_id=claim.run_id,
            )
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and "distiller store down" in run.error
        assert all(
            r.distilled_at is None for r in await _rows(sm, scope, "early-fail")
        )
        again = await _check(m, scope, force=True)
        assert again.session_ids == claim.session_ids


class TestErasureQuiesce:
    """Erasure stops in-flight learning before it purges."""

    async def test_a_blocked_run_is_stopped_and_writes_nothing(
        self, sm, dream_env, real_clock_managers, scope
    ):
        from mirix.services.auto_dream_manager import (
            AutoDreamManager,
            _procedural_dream_guard,
        )
        from mirix.services.skill_learning_run_manager import LearningRunAborted

        m = real_clock_managers
        await _write(m, scope, "erase-me", text="blockme")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="erase-me"
        )
        claim = await _check(m, scope, force=True)
        dream_env.llm = _GatedLLM("blockme")
        task = asyncio.create_task(
            AutoDreamManager().run(
                request=AutoDreamRequest(mode="procedural"),
                user=scope.user,
                actor=scope.actor,
                meta_agent_state=scope.meta,
                run_id=claim.run_id,
            )
        )
        await asyncio.wait_for(dream_env.llm.blocked.wait(), 5)

        async with skill_learning_trigger.quiesce_learning_for_erasure(
            user_id=scope.user.id, extra_owners=[(scope.user.id, scope.org)]
        ) as stopped:
            assert stopped == [claim.run_id]
            assert task.done()
            with pytest.raises(LearningRunAborted):
                task.result()
            run = await m.runs.get_run(claim.run_id)
            assert run.status == "failed" and run.error == "erased"
            # The owner's guard is held for the purge: a new run would wait.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    _procedural_dream_guard(scope.user.id, scope.org).__aenter__(),
                    0.1,
                )
        # The model answers late: nothing may be written any more.
        dream_env.llm.release.set()
        await asyncio.sleep(0.05)
        assert await _experiences(sm, scope, "erase-me") == []
        assert dream_env.curate_calls == []
        async with _procedural_dream_guard(scope.user.id, scope.org):
            pass  # released after the purge

    async def test_user_erasure_purges_inside_the_quiesce(
        self, sm, dream_env, m, scope, monkeypatch
    ):
        from mirix.services.auto_dream_manager import _procedural_dream_guard
        from mirix.services.user_manager import UserManager

        await _ended_sessions(m, scope, 1, prefix="purge")
        claim = await _check(m, scope, force=True)
        seen = {}

        async def fake_purge(self, user_id):
            seen["status"] = (await m.runs.get_run(claim.run_id)).status
            try:
                await asyncio.wait_for(
                    _procedural_dream_guard(user_id, scope.org).__aenter__(), 0.1
                )
                seen["guard_free"] = True
            except asyncio.TimeoutError:
                seen["guard_free"] = False

        monkeypatch.setattr(UserManager, "_purge_user_memories", fake_purge)
        await UserManager().delete_memories_by_user_id(scope.user.id)
        assert seen == {"status": "failed", "guard_free": False}

    async def test_client_erasure_fails_its_agents_runs_first(
        self, sm, dream_env, m, scope, monkeypatch
    ):
        from mirix.services.client_manager import ClientManager

        await _ended_sessions(m, scope, 1, prefix="purge-client")
        claim = await _check(m, scope, force=True)
        seen = {}

        async def fake_purge(self, client_id):
            seen["run"] = await m.runs.get_run(claim.run_id)

        monkeypatch.setattr(ClientManager, "_purge_client_memories", fake_purge)
        await ClientManager().delete_memories_by_client_id(scope.actor.id)
        assert seen["run"].status == "failed" and seen["run"].error == "erased"


class TestClientErasureSessions:
    async def test_a_session_shared_with_another_client_is_recounted(
        self, sm, m, scope, org_id, clock
    ):
        """Erasing one client keeps a session another client wrote to."""
        other = await _add_client(sm, org_id)
        await _write(m, scope, "shared", n=2)
        clock.advance(seconds=10)
        await m.messages.record_turns(
            session_id="shared",
            user_id=scope.user.id,
            organization_id=scope.org,
            turns=[{"role": "user", "content": "from the other client"}],
            actor=other,
            agent_id=scope.agent,
        )
        await _write(m, scope, "solo", n=2)
        await m.messages.delete_by_client_id(scope.actor)

        shared = await _session_row(sm, scope, "shared")
        assert shared.message_count == 1 and shared.session_tag == "task"
        [remaining] = await _rows(sm, scope, "shared")
        assert remaining.content == "from the other client"
        assert as_utc(shared.first_message_at) == as_utc(remaining.created_at)
        assert as_utc(shared.last_message_at) == as_utc(remaining.created_at)
        async with sm() as db:
            solo = (
                await db.execute(
                    select(SessionORM.id).where(
                        SessionORM.user_id == scope.user.id,
                        SessionORM.session_id == "solo",
                    )
                )
            ).first()
        assert solo is None


class TestBackgroundFailures:
    async def test_a_failure_before_execution_is_recorded(
        self, sm, m, scope, monkeypatch
    ):
        """e.g. a scope mismatch raised before execute() fails the run
        instead of leaving it queued (holding its sessions) until the lease."""
        import mirix.server.server as server_module

        monkeypatch.setattr(server_module, "db_context", sm)
        await _ended_sessions(m, scope, 1, prefix="pre-exec")
        claim = await _check(m, scope, force=True)

        async def mismatch(run_id, **ctx):
            raise ValueError("belongs to another scope")

        monkeypatch.setattr(skill_learning_trigger, "execute_learning_run", mismatch)
        await skill_learning_trigger.schedule_learning_run(claim.run_id)
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and "belongs to another scope" in run.error
        assert (await _check(m, scope, force=True)).session_ids == claim.session_ids


class TestRecordedGenericRuns:
    """Every auto_dream run is recorded; wait=false works
    for every mode and carries the request to the background."""

    @pytest.fixture
    def generic(self, dream_env, monkeypatch):
        from mirix.schemas.auto_dream import AutoDreamResponse
        from mirix.services.auto_dream_manager import AutoDreamManager

        calls = []

        async def fake_generic(self, *, request, user, actor, meta_agent_state, now):
            calls.append(request)
            return AutoDreamResponse(
                start_date=None,
                end_date=None,
                processed={},
                last_dream_at=now,
                dry_run=request.dry_run,
                message="generic done",
            )

        monkeypatch.setattr(AutoDreamManager, "_run_generic", fake_generic)
        dream_env.generic_calls = calls
        return dream_env

    async def _run(self, scope, run_id=None, **kwargs):
        from mirix.services.auto_dream_manager import AutoDreamManager

        return await AutoDreamManager().run(
            request=AutoDreamRequest(**kwargs),
            user=scope.user,
            actor=scope.actor,
            meta_agent_state=scope.meta,
            run_id=run_id,
        )

    async def test_a_waited_run_is_recorded(self, generic, real_clock_managers, scope):
        response = await self._run(scope, mode="episodic")
        [run_id] = response.run_ids
        run = await real_clock_managers.runs.get_run(run_id)
        assert run.mode == "episodic" and run.trigger == "manual"
        assert run.status == "succeeded" and run.session_ids == []
        assert len(generic.generic_calls) == 1

    async def test_a_dry_run_is_not_recorded(
        self, generic, real_clock_managers, scope
    ):
        response = await self._run(scope, mode="core", dry_run=True)
        assert response.run_ids == []
        assert (
            await real_clock_managers.runs.list_runs(
                organization_id=scope.org, user_id=scope.user.id
            )
            == []
        )

    async def test_wait_false_queues_and_the_background_run_executes(
        self, generic, real_clock_managers, scope, monkeypatch
    ):
        response = await self._run(scope, mode="semantic", wait=False, model="m-x")
        [run_id] = response.run_ids
        assert generic.scheduled == [run_id] and generic.generic_calls == []
        request = generic.scheduled_context["request"]
        assert request.mode == "semantic" and request.model == "m-x"
        assert (await real_clock_managers.runs.get_run(run_id)).status == "queued"

        async def resolved(run, **ctx):
            return scope.user, scope.actor, scope.meta

        monkeypatch.setattr(skill_learning_trigger, "resolve_run_context", resolved)
        # Also without the request: the run's own mode is used.
        await skill_learning_trigger.execute_learning_run(run_id)
        run = await real_clock_managers.runs.get_run(run_id)
        assert run.status == "succeeded"
        assert generic.generic_calls[0].mode == "semantic"

    async def test_a_run_id_of_another_mode_is_refused(
        self, generic, real_clock_managers, scope
    ):
        run = await real_clock_managers.runs.create_standalone_run(
            organization_id=scope.org,
            user_id=scope.user.id,
            agent_id=scope.agent,
            mode="episodic",
        )
        with pytest.raises(ValueError):
            await self._run(scope, run_id=run.id, mode="core")
        assert (await real_clock_managers.runs.get_run(run.id)).status == "queued"

    async def test_procedural_wait_false_carries_the_model(
        self, dream_env, real_clock_managers, scope
    ):
        m = real_clock_managers
        await _write(m, scope, "async-model")
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="async-model"
        )
        response = await self._run(
            scope, mode="procedural", wait=False, model="override-model"
        )
        assert dream_env.scheduled == response.run_ids
        assert dream_env.scheduled_context["request"].model == "override-model"


class _OnePendingExperience:
    """An experience manager with one pending experience (curator unit tests)."""

    async def list_pending_for_curation(self, **kwargs):
        return [
            SimpleNamespace(
                id="e1",
                experience_type="worth_learning",
                title="t",
                content="c",
                evidence="",
                importance=0.5,
                credibility=0.5,
                curation_attempts=0,
                influenced_skill_ids=None,
            )
        ]


class TestCuratorModelOverride:
    async def test_the_curator_agent_runs_on_the_override(self, monkeypatch):
        """The curator builds the procedural agent on the override
        model without mutating the stored agent state."""
        import mirix.agent as agent_pkg
        import mirix.server.server as server_module
        import mirix.services.skill_experience_curator as curator

        stored = SimpleNamespace(
            id="proc-agent", llm_config=SimpleNamespace(model="stored-model")
        )
        captured = {}

        class Stop(Exception):
            pass

        class SpyAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                raise Stop

        class Agents:
            async def update_agent_tools_and_system_prompts(self, *a, **k):
                return None

            async def get_agent_by_id(self, agent_id, actor):
                return stored

        server = SimpleNamespace(
            agent_manager=Agents(), default_interface_factory=lambda: None
        )

        async def resolve(server_, actor, meta_agent_state):
            return stored

        monkeypatch.setattr(agent_pkg, "ProceduralMemoryAgent", SpyAgent)
        monkeypatch.setattr(server_module, "get_server", lambda: server)
        monkeypatch.setattr(curator, "_resolve_procedural_agent_state", resolve)
        actor = PydanticClient(
            id="client-x",
            organization_id="org-x",
            name="c",
            write_scope="test",
            read_scopes=["test"],
        )
        user = PydanticUser(id="user-x", name="u", organization_id="org-x", timezone="UTC")
        with pytest.raises(Stop):
            await curator.curate_pending_experiences(
                run_id="run-x",
                user=user,
                actor=actor,
                meta_agent_state=SimpleNamespace(id="meta-x"),
                model_override="override-model",
                experience_manager=_OnePendingExperience(),
            )
        assert captured["agent_state"].llm_config.model == "override-model"
        assert stored.llm_config.model == "stored-model"

        captured.clear()
        with pytest.raises(Stop):
            await curator.curate_pending_experiences(
                run_id="run-x",
                user=user,
                actor=actor,
                meta_agent_state=SimpleNamespace(id="meta-x"),
                experience_manager=_OnePendingExperience(),
            )
        assert captured["agent_state"] is stored


class _GuardConnection:
    def __init__(self):
        self.invalidated = False

    async def invalidate(self):
        self.invalidated = True


class _GuardSession:
    """A Postgres-looking session recording the advisory-lock statements."""

    def __init__(self, *, lock_gate=None, unlock_gate=None, unlock_fails=False):
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        self.statements = []
        self.conn = _GuardConnection()
        self.lock_gate = lock_gate
        self.unlock_gate = unlock_gate
        self.unlock_fails = unlock_fails
        self.exited = False

    async def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        if "pg_advisory_lock" in sql and self.lock_gate is not None:
            await self.lock_gate.wait()
        if "pg_advisory_unlock" in sql:
            if self.unlock_gate is not None:
                await self.unlock_gate.wait()
            if self.unlock_fails:
                raise RuntimeError("connection reset")

    async def connection(self):
        return self.conn


class _GuardSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc):
        self.session.exited = True
        return False


class TestDreamGuardCancellation:
    """The advisory lock can never outlive an interrupted holder."""

    @pytest.fixture
    def guard_env(self, monkeypatch):
        import mirix.server.server as server_module

        env = SimpleNamespace(sessions=[])

        def use(session):
            env.sessions.append(session)
            monkeypatch.setattr(
                server_module, "db_context", lambda: _GuardSessionContext(session)
            )
            return session

        env.use = use
        return env

    def _guard(self):
        from mirix.services.auto_dream_manager import _procedural_dream_guard

        return _procedural_dream_guard(f"user-{uuid.uuid4().hex[:6]}", "org-g")

    async def test_a_normal_run_locks_and_unlocks(self, guard_env):
        session = guard_env.use(_GuardSession())
        async with self._guard():
            assert any("pg_advisory_lock" in s for s in session.statements)
        assert any("pg_advisory_unlock" in s for s in session.statements)
        assert session.exited and not session.conn.invalidated

    async def test_cancel_while_waiting_for_the_lock_drops_the_connection(
        self, guard_env
    ):
        session = guard_env.use(_GuardSession(lock_gate=asyncio.Event()))
        guard = self._guard()
        entered = asyncio.Event()

        async def holder():
            async with guard:
                entered.set()

        task = asyncio.create_task(holder())
        while not session.statements:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not entered.is_set()
        assert session.conn.invalidated and session.exited
        assert not any("pg_advisory_unlock" in s for s in session.statements)

    async def test_a_failed_unlock_drops_the_connection(self, guard_env):
        session = guard_env.use(_GuardSession(unlock_fails=True))
        async with self._guard():
            pass
        assert session.conn.invalidated and session.exited

    async def test_cancel_during_the_unlock_drops_the_connection(self, guard_env):
        session = guard_env.use(_GuardSession(unlock_gate=asyncio.Event()))

        async def holder():
            async with self._guard():
                pass

        task = asyncio.create_task(holder())
        while not any("pg_advisory_unlock" in s for s in session.statements):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.conn.invalidated and session.exited

    async def test_the_process_lock_is_free_again_after_a_cancel(self, guard_env):
        from mirix.services.auto_dream_manager import _procedural_dream_guard

        guard_env.use(_GuardSession(lock_gate=asyncio.Event()))
        task = asyncio.create_task(
            _procedural_dream_guard("user-cancel", "org-g").__aenter__()
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        guard_env.use(_GuardSession())
        async with _procedural_dream_guard("user-cancel", "org-g"):
            pass


# --------------------------------------------------------------------------- #
# cross-run backoff and retry events                                           #
# --------------------------------------------------------------------------- #


class TestInfraBackoff:
    """An infrastructure failure defers the session across runs: 1 min,
    doubling, capped at 1 h; it is neither counted nor claimed meanwhile, and
    never counts toward the 3 distill failures."""

    async def test_backoff_doubles_up_to_the_cap(self, sm, m, scope, clock):
        [sid] = await _ended_sessions(m, scope, 1, prefix="backoff")
        waits = []
        for _ in range(9):
            deferred = await m.sessions.defer_distill(
                organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
            )
            waits.append((as_utc(deferred[sid]) - clock.now).total_seconds())
        assert waits == [60, 120, 240, 480, 960, 1920, 3600, 3600, 3600]
        row = await _session_row(sm, scope, sid)
        assert row.distill_infra_failures == 9
        assert row.distill_failures == 0 and row.distill_failed_at is None

    async def test_a_deferred_session_is_not_pending_until_it_is_due(
        self, m, scope, clock
    ):
        [sid] = await _ended_sessions(m, scope, 1, prefix="due")

        async def pending():
            return await m.sessions.list_pending_distill_sessions(
                organization_id=scope.org, user_id=scope.user.id, agent_id=scope.agent
            )

        assert sid in await pending()
        await m.sessions.defer_distill(
            organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
        )
        assert sid not in await pending()
        clock.advance(seconds=59)
        assert sid not in await pending()
        clock.advance(seconds=1)
        assert sid in await pending()

    async def test_the_trigger_does_not_count_a_deferred_session(self, m, scope, clock):
        sids = await _ended_sessions(m, scope, 5, prefix="count")
        await m.sessions.defer_distill(
            organization_id=scope.org, user_id=scope.user.id, session_ids=[sids[0]]
        )
        assert not (await _check(m, scope)).fired  # 4 of 5
        clock.advance(minutes=1)
        claim = await _check(m, scope)
        assert claim.fired and sorted(claim.session_ids) == sorted(sids)

    async def test_a_successful_distill_ends_the_streak(self, sm, m, scope, clock):
        [sid] = await _ended_sessions(m, scope, 1, prefix="success")
        for _ in range(3):
            await m.sessions.defer_distill(
                organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
            )
        clock.advance(hours=1)
        claim = await _check(m, scope, force=True)
        assert claim.session_ids == [sid]
        run = await m.runs.get_run(claim.run_id)
        await m.runs.commit_distilled_session(
            claim.run_id,
            organization_id=scope.org,
            user_id=scope.user.id,
            session_id=sid,
            watermark=dt.datetime.fromisoformat(run.session_watermarks[sid]),
            experiences=[],
        )
        row = await _session_row(sm, scope, sid)
        assert row.distill_infra_failures == 0 and row.distill_retry_after is None

    async def test_a_reopen_ends_the_streak(self, sm, m, scope):
        [sid] = await _ended_sessions(m, scope, 1, prefix="reopen")
        await m.sessions.defer_distill(
            organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
        )
        await _write(m, scope, sid)  # a new segment: the session reopens
        row = await _session_row(sm, scope, sid)
        assert row.reopen_count == 1
        assert row.distill_infra_failures == 0 and row.distill_retry_after is None

    async def test_clear_and_empty_calls(self, sm, m, scope):
        [sid] = await _ended_sessions(m, scope, 1, prefix="clear")
        assert (
            await m.sessions.defer_distill(
                organization_id=scope.org, user_id=scope.user.id, session_ids=[]
            )
            == {}
        )
        await m.sessions.defer_distill(
            organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
        )
        cleared = await m.sessions.clear_distill_backoff(
            organization_id=scope.org, user_id=scope.user.id, session_ids=[sid]
        )
        assert cleared == 1
        row = await _session_row(sm, scope, sid)
        assert row.distill_infra_failures == 0 and row.distill_retry_after is None
        # Other users' sessions with the same id are never touched.
        assert (
            await m.sessions.defer_distill(
                organization_id=scope.org, user_id="someone-else", session_ids=[sid]
            )
            == {}
        )


class TestRetryEvents:
    """Every retry is recorded — kind, attempt, time waited."""

    async def test_events_append_to_the_learning_run(self, m, scope):
        from mirix.services.retry_policy import ErrorKind, RetryEventRecorder

        await _ended_sessions(m, scope, 1, prefix="events")
        claim = await _check(m, scope, force=True)
        recorder = RetryEventRecorder.for_learning_run(claim.run_id, run_manager=m.runs)
        await recorder.record(
            ErrorKind.TIMEOUT, 1, 0.0, stage="distill", session_id="s-1"
        )
        await recorder.record(
            "db", 2, 0.2, stage="curate", error=RuntimeError("deadlock detected")
        )
        events = (await m.runs.get_run(claim.run_id)).retry_events
        assert [(e["kind"], e["attempt"], e["waited_s"]) for e in events] == [
            ("timeout", 1, 0.0),
            ("db", 2, 0.2),
        ]
        assert events[0]["stage"] == "distill" and events[0]["session_id"] == "s-1"
        assert events[1]["error"] == "RuntimeError: deadlock detected"
        assert all("at" in e for e in events)
        assert recorder.unsent == []

    async def test_a_missing_run_keeps_the_events_pending(self, m):
        from mirix.services.retry_policy import RetryEventRecorder

        recorder = RetryEventRecorder.for_learning_run("slrun-gone", run_manager=m.runs)
        await recorder.record("infra", 1, 1.5)
        assert len(recorder.unsent) == 1 and len(recorder.events) == 1

    async def test_events_append_to_a_retrieval_log(self, sm, scope):
        from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogORM
        from mirix.services.retry_policy import (
            RetryEventRecorder,
            append_retrieval_retry_events,
        )

        async with sm() as db:
            db.add(
                RetrievalLogORM(
                    id="rlog-events",
                    organization_id=scope.org,
                    user_id=scope.user.id,
                )
            )
            await db.commit()
        recorder = RetryEventRecorder.for_retrieval_log("rlog-events", session_maker=sm)
        await recorder.record("embedding", 1, 1.0, stage="embed")
        await recorder.record("embedding", 2, 2.0, stage="embed")
        async with sm() as db:
            row = (
                await db.execute(
                    select(RetrievalLogORM).where(RetrievalLogORM.id == "rlog-events")
                )
            ).scalar_one()
        assert [(e["kind"], e["attempt"]) for e in row.retry_events] == [
            ("embedding", 1),
            ("embedding", 2),
        ]
        assert (
            await append_retrieval_retry_events(
                "rlog-missing", [{"kind": "infra"}], session_maker=sm
            )
            is False
        )
