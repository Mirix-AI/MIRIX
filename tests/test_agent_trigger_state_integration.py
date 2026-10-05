"""Postgres integration tests for session lifecycle, trigger claims and runs.

These need a real Postgres: SQLite ignores ``SELECT ... FOR UPDATE``, has no
row-level locking and takes the per-row idle-materialization path, so the
concurrency guarantees below would be false positives there. Run through the
fresh-database runner (MIRIX_PG_URI must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_agent_trigger_state_integration.py

Sessions are written the way production writes them — through
``ConversationMessageManager.record_turns`` into ``conversation_message`` and
``conversation_session`` (the old version of this file wrote the
agent-loop ``messages`` table, which the trigger never read).

Covered: threshold / below-threshold / first-check fire, cross-coroutine claim
serialization by the DATABASE lock alone (the process-local lock is disabled),
disjoint claims of two live runs, lease expiry, scope isolation, the tag gate,
soft-deleted rows, a write racing idle materialization, a write racing a claim,
the set-based idle UPDATE, and a full learning run whose session is reopened
after the claim.
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

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]

IDLE = 172800


class FakeClock:
    def __init__(self):
        self.now = dt.datetime.now(dt.timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + dt.timedelta(**kwargs)


class NoProcessLock:
    """Disables the process-local scope lock so only Postgres serializes."""

    @asynccontextmanager
    async def acquire(self, key):
        yield


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tables():
    """Schema, plus an engine pool bound to THIS module's event loop.

    The server's global async engine pools asyncpg connections, which belong to
    the loop that opened them; another module (another loop) reusing them fails.
    So drop whatever an earlier module left behind, and dispose ours at the end.
    """
    from mirix.server import server

    await server.engine.dispose(close=False)
    await server.ensure_tables_created()
    yield
    await server.engine.dispose()


async def _new_scope(tables=None):
    from mirix.orm.agent import Agent as AgentORM
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.user import User as PydanticUser
    from mirix.server.server import db_context

    org_id = f"it-org-{uuid.uuid4().hex[:8]}"
    user_id = f"it-user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    client_id = f"it-client-{uuid.uuid4().hex[:8]}"
    async with db_context() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        await db.commit()
    async with db_context() as db:
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
        user=PydanticUser(id=user_id, name=user_id, organization_id=org_id, timezone="UTC"),
        actor=PydanticClient(
            id=client_id,
            organization_id=org_id,
            name=client_id,
            write_scope="it",
            read_scopes=["it"],
        ),
        meta=SimpleNamespace(id=agent_id, llm_config=SimpleNamespace(model="m")),
    )


@pytest_asyncio.fixture(loop_scope="module")
async def scope(tables):
    return await _new_scope()


@pytest.fixture
def clock():
    return FakeClock()


def _managers(clock=None, *, process_lock=True):
    from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
    from mirix.services.conversation_message_manager import ConversationMessageManager
    from mirix.services.conversation_session_manager import ConversationSessionManager
    from mirix.services.skill_learning_run_manager import SkillLearningRunManager

    return SimpleNamespace(
        messages=ConversationMessageManager(clock=clock),
        sessions=ConversationSessionManager(clock=clock),
        trigger=AgentTriggerStateManager(
            clock=clock, scope_locks=None if process_lock else NoProcessLock()
        ),
        runs=SkillLearningRunManager(clock=clock, heartbeat_seconds=0.05),
    )


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


async def _ended(m, scope, count, *, prefix="s", tag="task"):
    sids = []
    for i in range(count):
        sid = f"{prefix}{i}-{uuid.uuid4().hex[:6]}"
        await _write(m, scope, sid, tag=tag)
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id=sid
        )
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


async def _rows(scope, sid):
    from mirix.orm.conversation_message import ConversationMessage as MessageORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            (
                await db.execute(
                    select(MessageORM)
                    .where(MessageORM.user_id == scope.user.id, MessageORM.session_id == sid)
                    .order_by(MessageORM.created_at)
                )
            )
            .scalars()
            .all()
        )


class TestThreshold:
    async def test_first_check_fires_at_the_threshold(self, scope, clock):
        m = _managers(clock)
        sids = await _ended(m, scope, 5)
        claim = await _check(m, scope)
        assert claim.fired and claim.session_ids == sids
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "queued" and run.trigger == "threshold"
        for sid in sids:
            rows = await _rows(scope, sid)
            assert dt.datetime.fromisoformat(run.session_watermarks[sid]) == rows[-1].created_at

    async def test_below_threshold_never_fires(self, scope, clock):
        m = _managers(clock)
        await _ended(m, scope, 4)
        for days in (0, 2, 30, 400):
            clock.advance(days=days)
            assert (await _check(m, scope)).fired is False

    async def test_conversation_sessions_never_count(self, scope, clock):
        m = _managers(clock)
        await _ended(m, scope, 5, tag="conversation")
        assert (await _check(m, scope)).fired is False
        assert (await _check(m, scope, force=True)).fired is False

    async def test_soft_deleted_rows_do_not_count(self, scope, clock):
        from mirix.orm.conversation_message import ConversationMessage as MessageORM
        from mirix.server.server import db_context

        m = _managers(clock)
        sids = await _ended(m, scope, 5)
        async with db_context() as db:
            await db.execute(
                update(MessageORM)
                .where(MessageORM.user_id == scope.user.id, MessageORM.session_id == sids[0])
                .values(is_deleted=True)
            )
            await db.commit()
        claim = await _check(m, scope)
        assert claim.fired is False and claim.pending_count == 4


class TestConcurrency:
    async def test_concurrent_checks_fire_once_by_the_database_lock_alone(
        self, scope, clock
    ):
        writer = _managers(clock)
        await _ended(writer, scope, 5)
        # Independent managers, NO process-local lock: only SELECT ... FOR UPDATE
        # on the scope's cursor row serializes these.
        claims = await asyncio.gather(
            *[_check(_managers(clock, process_lock=False), scope) for _ in range(8)]
        )
        fired = [c for c in claims if c.fired]
        assert len(fired) == 1 and len(fired[0].session_ids) == 5

    async def test_concurrent_flushes_never_share_a_session(self, scope, clock):
        writer = _managers(clock)
        await _ended(writer, scope, 6)
        claims = await asyncio.gather(
            *[
                _check(_managers(clock, process_lock=False), scope, force=True, trigger="flush")
                for _ in range(6)
            ]
        )
        claimed = [sid for c in claims if c.fired for sid in c.session_ids]
        assert len(claimed) == len(set(claimed)) == 6

    async def test_two_live_runs_claim_disjoint_sessions(self, scope, clock):
        m = _managers(clock)
        a = await _ended(m, scope, 5, prefix="a")
        run_a = await _check(m, scope)
        b = await _ended(m, scope, 5, prefix="b")
        run_b = await _check(m, scope)
        assert run_a.session_ids == a and run_b.session_ids == b
        assert (await _check(m, scope, force=True)).fired is False

    async def test_a_write_racing_idle_materialization_wins(self, scope, clock):
        from mirix.orm.conversation_message import ConversationMessage as MessageORM
        from mirix.server.server import db_context

        m = _managers(clock)
        await _write(m, scope, "race-idle")
        clock.advance(seconds=IDLE + 5)
        async with db_context() as db:
            row, state = await m.sessions.lock_for_write(
                db,
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id="race-idle",
                agent_id=scope.agent,
                requested_tag="task",
                actor_id=scope.actor.id,
                now=clock.now,
            )
            assert state.reopened is True  # idle-ended, not yet materialized
            materialize = asyncio.create_task(
                m.sessions.materialize_idle(user_id=scope.user.id, session_id="race-idle")
            )
            await asyncio.sleep(0.3)
            assert not materialize.done()  # blocked on the session row lock
            db.add(
                MessageORM(
                    session_id="race-idle",
                    user_id=scope.user.id,
                    organization_id=scope.org,
                    role="user",
                    content="back again",
                    session_tag="task",
                    created_at=clock.now,
                )
            )
            m.sessions.apply_write(
                row,
                first_message_at=clock.now,
                last_message_at=clock.now,
                message_count=1,
                created=False,
                actor_id=scope.actor.id,
                now=clock.now,
            )
            await db.commit()
        assert await materialize == 0  # re-checked after the lock: no longer idle
        session = await m.sessions.get_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="race-idle"
        )
        assert session.closed_at is None and session.reopen_count == 1

    async def test_a_write_racing_a_claim_keeps_its_session_out(self, scope, clock):
        from mirix.orm.conversation_message import ConversationMessage as MessageORM
        from mirix.server.server import db_context

        m = _managers(clock)
        sids = await _ended(m, scope, 5)
        async with db_context() as db:
            row, state = await m.sessions.lock_for_write(
                db,
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id=sids[0],
                agent_id=scope.agent,
                requested_tag="task",
                actor_id=scope.actor.id,
                now=clock.now,
            )
            assert state.reopened is True
            claim_task = asyncio.create_task(_check(m, scope, force=True))
            await asyncio.sleep(0.3)
            assert not claim_task.done()  # waits for the session row
            db.add(
                MessageORM(
                    session_id=sids[0],
                    user_id=scope.user.id,
                    organization_id=scope.org,
                    role="user",
                    content="reopened",
                    session_tag="task",
                    created_at=clock.now + dt.timedelta(seconds=1),
                )
            )
            m.sessions.apply_write(
                row,
                first_message_at=clock.now,
                last_message_at=clock.now + dt.timedelta(seconds=1),
                message_count=1,
                created=False,
                actor_id=scope.actor.id,
                now=clock.now,
            )
            await db.commit()
        claim = await claim_task
        assert claim.fired and claim.session_ids == sids[1:]


class TestLeaseAndIsolation:
    async def test_lease_expiry_releases_the_claim(self, scope, clock):
        m = _managers(clock)
        await _ended(m, scope, 5)
        claim = await _check(m, scope)
        clock.advance(seconds=301)
        assert claim.run_id in await m.runs.recover_expired_leases()
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "failed" and run.error == "lease_expired"
        again = await _check(m, scope)
        assert again.fired and again.session_ids == claim.session_ids

    async def test_users_and_agents_are_isolated(self, scope, clock):
        m = _managers(clock)
        other = await _new_scope()
        await _ended(m, other, 5)
        assert (await _check(m, scope)).fired is False
        # Same user as `other`, but through another meta agent's scope.
        crossed = SimpleNamespace(**{**vars(other), "agent": scope.agent})
        assert (await _check(m, crossed, force=True)).fired is False
        assert (await _check(m, other)).fired is True

    async def test_set_based_idle_materialization(self, scope, clock):
        m = _managers(clock)
        written = [await _write(m, scope, f"pg-idle-{i}") for i in range(3)]
        clock.advance(seconds=IDLE + 1)
        assert await m.sessions.materialize_idle(user_id=scope.user.id) == 3
        for out in written:
            session = await m.sessions.get_session(
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id=out.session.session_id,
            )
            assert session.close_reason == "idle"
            assert session.closed_at == out.session.last_message_at + dt.timedelta(
                seconds=IDLE
            )


class TestLearningRunOnPostgres:
    async def test_reopened_session_keeps_post_claim_rows_pending(self, scope, monkeypatch):
        from mirix.schemas.auto_dream import AutoDreamRequest
        from mirix.services.auto_dream_manager import AutoDreamManager
        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )

        seen = []

        class FakeLLM:
            async def send_llm_request(self, messages):
                seen.append(messages[-1].content[0].text)
                content = json.dumps(
                    [
                        {
                            "experience_type": "worth_learning",
                            "title": "lesson",
                            "content": "c",
                            "importance": 0.5,
                            "credibility": 0.5,
                        }
                    ]
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        def make_distiller(self, *, llm_config, conversation_manager):
            return SessionExperienceDistiller(
                llm_client=FakeLLM(), conversation_manager=conversation_manager
            )

        async def no_curation(self, **kwargs):
            from mirix.services.skill_experience_curator import CuratorOutcome

            return CuratorOutcome()

        async def no_checkpoint(*args, **kwargs):
            return None

        monkeypatch.setattr(AutoDreamManager, "_make_distiller", make_distiller)
        monkeypatch.setattr(AutoDreamManager, "_curate", no_curation)
        monkeypatch.setattr(AutoDreamManager, "write_checkpoint", no_checkpoint)

        m = _managers()
        await _write(m, scope, "pg-reopen", n=2)
        await m.sessions.close_session(
            organization_id=scope.org, user_id=scope.user.id, session_id="pg-reopen"
        )
        claim = await _check(m, scope, force=True)
        await _write(m, scope, "pg-reopen", n=1, text="after claim")

        response = await AutoDreamManager().run(
            request=AutoDreamRequest(mode="procedural"),
            user=scope.user,
            actor=scope.actor,
            meta_agent_state=scope.meta,
            run_id=claim.run_id,
        )
        assert response.run_ids == [claim.run_id]
        rows = await _rows(scope, "pg-reopen")
        assert [r.distilled_at is not None for r in rows] == [True, True, False]
        assert "after claim" not in seen[0]
        run = await m.runs.get_run(claim.run_id)
        assert run.status == "succeeded"
        assert run.distill_metrics[0]["messages"] == 2


# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


def _asyncpg_dsn() -> str:
    from mirix.settings import settings as _s

    uri = _s.mirix_pg_uri_no_default
    for prefix in ("postgresql+pg8000://", "postgresql+asyncpg://"):
        if uri.startswith(prefix):
            return "postgresql://" + uri[len(prefix) :]
    return uri


async def _db_now():
    from sqlalchemy import func, select

    from mirix.server.server import db_context

    async with db_context() as db:
        return (await db.execute(select(func.clock_timestamp()))).scalar_one()


class TestReviewOnPostgres:
    async def test_lease_timing_uses_the_database_clock(self, scope):
        """Hosts whose clocks are 10 minutes off neither reclaim a live
        run nor stamp a heartbeat that looks stale."""
        from mirix.orm.skill_learning_run import SkillLearningRun as RunORM
        from mirix.server.server import db_context
        from mirix.services.skill_learning_run_manager import SkillLearningRunManager
        from sqlalchemy import update as sa_update

        m = _managers()  # no injected clock: the production configuration
        await _ended(m, scope, 1, prefix="skew")
        claim = await _check(m, scope, force=True)
        real = dt.datetime.now

        ahead = SkillLearningRunManager()
        ahead.clock = lambda: real(dt.timezone.utc) + dt.timedelta(minutes=10)
        assert await ahead.recover_expired_leases(user_id=scope.user.id) == []

        behind = SkillLearningRunManager()
        behind.clock = lambda: real(dt.timezone.utc) - dt.timedelta(minutes=10)
        assert await behind.heartbeat(claim.run_id) is True
        run = await m.runs.get_run(claim.run_id)
        db_now = await _db_now()
        assert abs((db_now - run.heartbeat_at).total_seconds()) < 5

        # A heartbeat genuinely older than the lease (by the DB clock) is
        # reclaimed even by the host that is 10 minutes behind.
        async with db_context() as db:
            await db.execute(
                sa_update(RunORM)
                .where(RunORM.id == claim.run_id)
                .values(
                    heartbeat_at=db_now
                    - dt.timedelta(seconds=behind.lease_seconds + 5)
                )
            )
            await db.commit()
        assert await behind.recover_expired_leases(user_id=scope.user.id) == [
            claim.run_id
        ]

    async def test_a_cancelled_guard_never_keeps_the_advisory_lock(self, tables):
        """Cancelled while waiting for the advisory lock (held by "another
        process"), the guard's connection is dropped, so the lock is never
        granted to a pooled connection afterwards."""
        import asyncpg

        from mirix.services.auto_dream_manager import (
            _advisory_lock_key,
            _procedural_dream_guard,
        )

        user, org = f"it-guard-{uuid.uuid4().hex[:6]}", "it-guard-org"
        key = _advisory_lock_key(user, org)
        other_process = await asyncpg.connect(_asyncpg_dsn())
        try:
            await other_process.execute("SELECT pg_advisory_lock($1)", key)
            entered = asyncio.Event()

            async def waiter():
                async with _procedural_dream_guard(user, org):
                    entered.set()

            task = asyncio.create_task(waiter())
            for _ in range(100):  # until it waits on the advisory lock
                waiting = await other_process.fetchval(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                    "AND NOT granted"
                )
                if waiting:
                    break
                await asyncio.sleep(0.02)
            assert waiting
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not entered.is_set()
            await other_process.execute("SELECT pg_advisory_unlock($1)", key)
            await asyncio.sleep(0.2)
            # Nobody holds it: a third party gets it at once.
            assert await other_process.fetchval("SELECT pg_try_advisory_lock($1)", key)
            await other_process.execute("SELECT pg_advisory_unlock($1)", key)
        finally:
            await other_process.close()
        async with _procedural_dream_guard(user, org):
            pass

    async def test_an_erased_run_cannot_commit_a_session(self, scope):
        """On Postgres the commit re-checks the run under a row lock."""
        from mirix.services.skill_learning_run_manager import (
            LearningRunNotActive,
            SkillLearningRunManager,
        )

        m = _managers()
        [sid] = await _ended(m, scope, 1, prefix="erased")
        claim = await _check(m, scope, force=True)
        failed = await SkillLearningRunManager().fail_active_runs(
            user_id=scope.user.id
        )
        assert [r.id for r in failed] == [claim.run_id]
        run = await m.runs.get_run(claim.run_id)
        with pytest.raises(LearningRunNotActive):
            await m.runs.commit_distilled_session(
                claim.run_id,
                organization_id=scope.org,
                user_id=scope.user.id,
                session_id=sid,
                watermark=dt.datetime.fromisoformat(run.session_watermarks[sid]),
                experiences=[],
            )
        assert all(r.distilled_at is None for r in await _rows(scope, sid))

    async def test_client_erasure_recounts_a_shared_session(self, scope):
        """Client erasure recounts a shared session on Postgres (row-value IN +
        FOR UPDATE)."""
        from mirix.orm.conversation_session import ConversationSession as SessionORM
        from mirix.schemas.client import Client as PydanticClient
        from mirix.server.server import db_context
        from sqlalchemy import select

        m = _managers()
        other = PydanticClient(
            id=f"it-client-{uuid.uuid4().hex[:8]}",
            organization_id=scope.org,
            name="other",
            write_scope="it",
            read_scopes=["it"],
        )
        await _write(m, scope, "pg-shared", n=2)
        await m.messages.record_turns(
            session_id="pg-shared",
            user_id=scope.user.id,
            organization_id=scope.org,
            turns=[{"role": "user", "content": "theirs"}],
            actor=other,
            agent_id=scope.agent,
        )
        await _write(m, scope, "pg-solo", n=1)
        await m.messages.delete_by_client_id(scope.actor)
        async with db_context() as db:
            rows = {
                r.session_id: r
                for r in (
                    await db.execute(
                        select(SessionORM).where(SessionORM.user_id == scope.user.id)
                    )
                )
                .scalars()
                .all()
            }
        assert set(rows) == {"pg-shared"}
        assert rows["pg-shared"].message_count == 1
        [remaining] = await _rows(scope, "pg-shared")
        assert remaining.content == "theirs"
