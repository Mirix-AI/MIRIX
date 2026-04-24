"""DB-backed integration tests for the session-based procedural trigger.

These tests require a real Postgres. `SELECT ... FOR UPDATE` is a no-op on
SQLite, so running these there would produce false-positive green on the
concurrency tests. The whole file is marked `integration` and skipped
unless MIRIX_PG_URI (or pg_user/pg_password/...) resolves to Postgres.

Run with:
    docker-compose up -d postgres
    pytest tests/test_agent_trigger_state_integration.py -v -m integration

The tests insert rows directly via SQLAlchemy ORM so `created_at` can be
controlled to the microsecond. This is essential for exercising the
tie-at-watermark path and the MIN-based "first-appearance" semantics.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest
import pytest_asyncio

from mirix.schemas.agent_trigger_state import TRIGGER_TYPE_PROCEDURAL_SKILL
from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
from mirix.settings import settings


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason=(
            "needs Postgres; SELECT FOR UPDATE is a no-op on SQLite so the "
            "concurrency test would false-positive"
        ),
    ),
]


# ----------------------------- infrastructure ----------------------------


@pytest.fixture(scope="module")
def server():
    """One AsyncServer instance per test module; owns the DB connection pool."""
    from mirix.server.server import AsyncServer

    return AsyncServer()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def org(server):
    """Dedicated org so we don't collide with other test suites."""
    from mirix.schemas.organization import Organization

    org_id = f"ats-org-{uuid.uuid4().hex[:8]}"
    return await server.organization_manager.create_organization(
        pydantic_org=Organization(id=org_id, name=org_id)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def user_a(server, org):
    """Primary test user."""
    from mirix.schemas.user import User

    uid = f"ats-user-a-{uuid.uuid4().hex[:8]}"
    return await server.user_manager.create_user(
        pydantic_user=User(id=uid, name=uid, organization_id=org.id, timezone="UTC")
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def user_b(server, org):
    """Second user for cross-user isolation tests."""
    from mirix.schemas.user import User

    uid = f"ats-user-b-{uuid.uuid4().hex[:8]}"
    return await server.user_manager.create_user(
        pydantic_user=User(id=uid, name=uid, organization_id=org.id, timezone="UTC")
    )


async def _insert_minimal_agent(server, org_id: str) -> str:
    """Insert a bare-bones agent row. All descriptive fields on the Agent
    ORM are nullable, so a fresh UUID and the org_id satisfy the schema.
    We bypass AgentManager to keep this test suite free of LLMConfig /
    embedding-model requirements that don't matter for the trigger path.
    """
    from mirix.orm.agent import Agent as AgentORM

    agent_id = f"agent-{uuid.uuid4()}"
    async with server.db_context() as session:
        session.add(AgentORM(id=agent_id, organization_id=org_id))
        await session.commit()
    return agent_id


@pytest_asyncio.fixture(loop_scope="module")
async def agent_id(server, org):
    """Each test gets a fresh agent so the (agent_id, user_id, trigger_type)
    unique key isolates trigger-state rows across tests.
    """
    return await _insert_minimal_agent(server, org.id)


@pytest_asyncio.fixture(loop_scope="module")
async def other_agent_id(server, org):
    """Second agent for cross-agent isolation tests."""
    return await _insert_minimal_agent(server, org.id)


async def _insert_message(
    server,
    *,
    agent_id: str,
    user_id: str,
    org_id: str,
    session_id: str,
    created_at: datetime,
    is_deleted: bool = False,
) -> str:
    """Insert a message row with a caller-specified `created_at`.

    Going through MessageManager.create_message would stamp created_at via
    the server default. For these tests we need microsecond-level control
    so the tie-at-watermark and MIN-semantics cases can be set up
    deterministically.
    """
    from mirix.orm.message import Message as MessageORM

    msg_id = f"message-{uuid.uuid4()}"
    async with server.db_context() as session:
        msg = MessageORM(
            id=msg_id,
            agent_id=agent_id,
            user_id=user_id,
            organization_id=org_id,
            role="user",
            text=f"stub-{msg_id}",
            tool_calls=[],
            tool_returns=[],
            session_id=session_id,
            created_at=created_at,
            is_deleted=is_deleted,
        )
        session.add(msg)
        await session.commit()
    return msg_id


# ----------------------------- core behaviors ----------------------------


class TestFirstInstall:
    async def test_install_when_no_cursor(self, server, agent_id, user_a):
        """First ever call must install the cursor at `now` and NOT fire,
        even if there are pre-existing messages. Otherwise enabling the
        feature would sweep up all legacy sessions on the first tick.
        """
        mgr = AgentTriggerStateManager()

        # Seed some legacy messages BEFORE first install — these must be
        # ignored on install.
        legacy_ts = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"legacy-{i}",
                created_at=legacy_ts + timedelta(seconds=i),
            )

        claim = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
            current_session_id="bootstrap",
        )

        assert claim.just_installed is True
        assert claim.fired is False
        assert claim.sessions_since == 0
        assert claim.state.last_fired_at is not None


class TestBasicFire:
    async def test_fires_when_threshold_met(self, server, agent_id, user_a):
        mgr = AgentTriggerStateManager()
        # Install cursor first.
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
            current_session_id="seed",
        )
        # Insert 5 distinct NEW sessions strictly after the install cursor.
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"sess-{i}",
                created_at=base + timedelta(seconds=i),
            )

        claim = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
            current_session_id="sess-4",
        )
        assert claim.fired is True
        assert claim.sessions_since == 5
        # Cursor advances to the observed watermark, not wall clock.
        assert claim.state.last_fired_at == base + timedelta(seconds=4)

    async def test_below_threshold_does_not_fire(self, server, agent_id, user_a):
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(4):  # one short of threshold
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"sess-{i}",
                created_at=base + timedelta(seconds=i),
            )

        claim = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim.fired is False
        assert claim.sessions_since == 4


class TestNoDoubleCount:
    """The whole point of switching to MIN semantics. Without it, a session
    that was counted in window 1 and later sends more messages would get
    re-counted in window 2 — permanent over-fire.
    """

    async def test_old_session_continuing_does_not_recount(
        self, server, agent_id, user_a
    ):
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )

        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"sess-{i}",
                created_at=base + timedelta(seconds=i),
            )

        # First fire.
        claim1 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim1.fired is True

        # Now add more messages to the SAME sessions at a later timestamp.
        # Under MAX semantics these would re-qualify; under MIN they don't.
        later = base + timedelta(minutes=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"sess-{i}",  # same session_ids
                created_at=later + timedelta(seconds=i),
            )

        claim2 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim2.fired is False, (
            "continuing old sessions must NOT re-fire — each session "
            "contributes to exactly one window under MIN semantics"
        )
        assert claim2.sessions_since == 0

    async def test_new_sessions_after_fire_do_refire(
        self, server, agent_id, user_a
    ):
        """Sanity: genuinely new sessions after a fire DO re-qualify."""
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"round1-{i}",
                created_at=base + timedelta(seconds=i),
            )
        claim1 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim1.fired is True

        # Five brand-new session_ids at a later time.
        later = base + timedelta(minutes=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"round2-{i}",
                created_at=later + timedelta(seconds=i),
            )
        claim2 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim2.fired is True
        assert claim2.sessions_since == 5


class TestSoftDeleteExclusion:
    async def test_deleted_sessions_do_not_count(self, server, agent_id, user_a):
        """A user wiping history should not leave the procedural trigger
        armed by the deleted sessions."""
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        # 3 live sessions...
        for i in range(3):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"live-{i}",
                created_at=base + timedelta(seconds=i),
            )
        # ...plus 3 sessions whose only messages are soft-deleted.
        for i in range(3):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"ghost-{i}",
                created_at=base + timedelta(seconds=10 + i),
                is_deleted=True,
            )

        claim = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim.fired is False
        assert claim.sessions_since == 3, (
            "only the 3 live sessions should count; deleted sessions are "
            "invisible to the trigger"
        )


class TestUserIsolation:
    async def test_one_users_sessions_do_not_fire_another(
        self, server, agent_id, user_a, user_b
    ):
        """Cursor is keyed per (agent, user, trigger_type). Five sessions
        for user A must not trip user B's threshold."""
        mgr = AgentTriggerStateManager()
        # Install cursors for both users.
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_b.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_b.organization_id,
        )

        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        # Five new sessions for user A ONLY.
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"a-sess-{i}",
                created_at=base + timedelta(seconds=i),
            )

        claim_a = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        claim_b = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_b.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_b.organization_id,
        )
        assert claim_a.fired is True
        assert claim_b.fired is False
        assert claim_b.sessions_since == 0


class TestTiedAtWatermark:
    async def test_tied_sessions_recorded_and_not_double_counted(
        self, server, agent_id, user_a
    ):
        """When multiple sessions share the exact watermark timestamp,
        they should all be recorded in last_fired_tied_session_ids so the
        next window does not re-count them via the `MIN == cursor` branch
        of the HAVING filter.
        """
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )

        # 3 sessions with distinct timestamps, then 2 sharing the exact
        # same watermark microsecond. The aggregate should see all 5, and
        # the tied set should capture the 2 sharing the max.
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(3):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"early-{i}",
                created_at=base + timedelta(seconds=i),
            )
        watermark = base + timedelta(seconds=10)
        for sid in ("tied-a", "tied-b"):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=sid,
                created_at=watermark,
            )

        claim = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim.fired is True
        assert claim.sessions_since == 5
        assert claim.state.last_fired_at == watermark
        assert set(claim.state.last_fired_tied_session_ids or []) == {"tied-a", "tied-b"}

        # Running again with NO new messages must not fire and must not
        # re-count either tied-a or tied-b.
        claim2 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim2.fired is False
        assert claim2.sessions_since == 0

    async def test_late_arriving_session_at_watermark_picked_up_next_window(
        self, server, agent_id, user_a
    ):
        """If a session's first message commits at the exact watermark but
        was invisible to our SELECT, the tie-breaker must allow it into
        the next window — i.e. `MIN == cursor AND session_id NOT IN tied`.
        We simulate the invisibility by inserting AFTER the first fire.
        """
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )

        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        watermark = base + timedelta(seconds=10)
        # 4 "seen" sessions at strictly increasing ts, plus 1 at watermark.
        for i in range(4):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"seen-{i}",
                created_at=base + timedelta(seconds=i),
            )
        await _insert_message(
            server,
            agent_id=agent_id,
            user_id=user_a.id,
            org_id=user_a.organization_id,
            session_id="seen-at-watermark",
            created_at=watermark,
        )

        claim1 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim1.fired is True

        # Now insert a session whose first message is AT the watermark but
        # wasn't visible during claim1 — a delayed commit. Under a naive
        # `MIN > cursor` filter this would be lost forever. The tie-breaker
        # (`MIN = cursor AND session_id NOT IN tied_ids`) must rescue it.
        await _insert_message(
            server,
            agent_id=agent_id,
            user_id=user_a.id,
            org_id=user_a.organization_id,
            session_id="late-commit",
            created_at=watermark,
        )
        # Four more new sessions strictly after watermark — together with
        # the rescued late-commit we have 5 → fire.
        for i in range(4):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"after-{i}",
                created_at=watermark + timedelta(seconds=1 + i),
            )

        claim2 = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert claim2.fired is True, "late-commit session must be rescued"
        assert claim2.sessions_since == 5
        # Sanity: the late-commit session is one of the counted, proving
        # the tie-breaker really did include the `MIN == cursor` branch.
        # (We can't assert the ids directly from the result, but the count
        # of 5 only checks out if it was included.)


class TestConcurrency:
    async def test_concurrent_fire_serialized_to_one_winner(
        self, server, agent_id, user_a
    ):
        """Two concurrent check_and_claim_fire coroutines against the same
        (agent, user, trigger_type) must serialize on SELECT FOR UPDATE.
        Exactly one gets fired=True; the other sees the advanced cursor
        and fired=False.
        """
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"conc-{i}",
                created_at=base + timedelta(seconds=i),
            )

        async def claim():
            return await mgr.check_and_claim_fire(
                agent_id=agent_id,
                user_id=user_a.id,
                trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
                threshold=5,
                organization_id=user_a.organization_id,
            )

        # Fire two coroutines at the same time. asyncio.gather schedules
        # them concurrently; the real serialization happens at the
        # Postgres row lock.
        result_1, result_2 = await asyncio.gather(claim(), claim())
        fired_flags = sorted([result_1.fired, result_2.fired])
        assert fired_flags == [False, True], (
            "exactly one concurrent call must win the fire; got "
            f"{fired_flags}"
        )


class TestAgentIsolation:
    async def test_sessions_on_one_agent_do_not_count_for_another(
        self, server, agent_id, other_agent_id, user_a
    ):
        """Cursor is per-agent. Sessions on agent X must not contribute to
        the threshold on agent Y even for the same user.
        """
        mgr = AgentTriggerStateManager()
        await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        await mgr.check_and_claim_fire(
            agent_id=other_agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        base = datetime.now(timezone.utc) + timedelta(seconds=1)
        for i in range(5):
            await _insert_message(
                server,
                agent_id=agent_id,
                user_id=user_a.id,
                org_id=user_a.organization_id,
                session_id=f"x-sess-{i}",
                created_at=base + timedelta(seconds=i),
            )

        fire_x = await mgr.check_and_claim_fire(
            agent_id=agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        fire_y = await mgr.check_and_claim_fire(
            agent_id=other_agent_id,
            user_id=user_a.id,
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            threshold=5,
            organization_id=user_a.organization_id,
        )
        assert fire_x.fired is True
        assert fire_y.fired is False
        assert fire_y.sessions_since == 0
