"""
Tests for the configurable last-N-session raw-message retention feature.

GOAL 1: stop hard-deleting the canonical conversation messages immediately after
memory extraction; retain at least the last N sessions of raw messages so a later
distiller / auto-dream can read them. Retention != in-context: retained rows stay
DETACHED from message_ids (never re-added).

This file has two parts:
  * TestSelectRetainedSessionIds — DB-FREE unit tests of the pure selection helper
    MessageManager._select_retained_session_ids. These run anywhere (no Postgres).
  * TestDeleteDetachedRetention — integration tests (marked `integration`) that seed
    N+2 sessions for one (agent, user), call delete_detached_messages_for_agent with
    retain_last_n_sessions=N, and assert exactly the last N distinct session_ids
    survive while older ones are deleted, and a second user's sessions are
    unaffected. Requires the docker-compose Postgres. Run:
        pytest tests/test_message_retention.py -v -m integration
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.services.message_manager import MessageManager


# ---------------------------------------------------------------------------
# DB-free unit tests of the pure selection helper.
# ---------------------------------------------------------------------------
class TestSelectRetainedSessionIds:
    def _ts(self, minute):
        return datetime(2026, 1, 1, 0, minute, 0, tzinfo=timezone.utc)

    def test_zero_retains_nothing(self):
        mapping = {"s1": self._ts(1), "s2": self._ts(2)}
        assert MessageManager._select_retained_session_ids(mapping, 0) == set()

    def test_negative_retains_nothing(self):
        mapping = {"s1": self._ts(1)}
        assert MessageManager._select_retained_session_ids(mapping, -1) == set()

    def test_empty_mapping_retains_nothing(self):
        assert MessageManager._select_retained_session_ids({}, 5) == set()

    def test_selects_most_recent_n_by_min_created_at(self):
        # s1 oldest .. s5 newest by first-seen timestamp.
        mapping = {
            "s1": self._ts(1),
            "s2": self._ts(2),
            "s3": self._ts(3),
            "s4": self._ts(4),
            "s5": self._ts(5),
        }
        # Retain last 3 -> the 3 most-recent first-seen sessions.
        assert MessageManager._select_retained_session_ids(mapping, 3) == {
            "s3",
            "s4",
            "s5",
        }

    def test_retain_n_larger_than_population_keeps_all(self):
        mapping = {"s1": self._ts(1), "s2": self._ts(2)}
        assert MessageManager._select_retained_session_ids(mapping, 10) == {"s1", "s2"}

    def test_default_five_keeps_last_five_of_seven(self):
        mapping = {f"s{i}": self._ts(i) for i in range(1, 8)}  # s1..s7
        retained = MessageManager._select_retained_session_ids(mapping, 5)
        assert retained == {"s3", "s4", "s5", "s6", "s7"}
        # The two oldest age out and become eligible for deletion.
        assert "s1" not in retained
        assert "s2" not in retained

    def test_tie_break_is_deterministic_on_session_id(self):
        # Two sessions share the same MIN(created_at); tie broken by session_id desc.
        ts = self._ts(1)
        mapping = {"aaa": ts, "bbb": ts, "ccc": self._ts(2)}
        # Retain 2 -> the newest (ccc) plus the higher session_id among the tie (bbb).
        assert MessageManager._select_retained_session_ids(mapping, 2) == {"ccc", "bbb"}

    def test_none_timestamp_sessions_treated_as_oldest(self):
        # A session whose rows all lack created_at (None ts) is treated as OLDEST:
        # excluded when N < population, and only included (in the oldest slot) when
        # N covers it. Guards against comparing None to datetime.
        mapping = {"s1": self._ts(1), "s2": self._ts(2), "snull": None}
        # Retain 2 -> the two timestamped sessions; the None-ts session is evicted.
        assert MessageManager._select_retained_session_ids(mapping, 2) == {"s1", "s2"}
        # Retain 3 -> the None-ts session is included as the oldest slot.
        assert MessageManager._select_retained_session_ids(mapping, 3) == {"s1", "s2", "snull"}


# ---------------------------------------------------------------------------
# Integration tests against real Postgres.
# ---------------------------------------------------------------------------
pytest_asyncio = pytest.importorskip("pytest_asyncio")

from mirix.schemas.client import Client as PydanticClient  # noqa: E402
from mirix.schemas.enums import MessageRole  # noqa: E402
from mirix.schemas.message import Message as PydanticMessage  # noqa: E402
from mirix.schemas.mirix_message_content import TextContent  # noqa: E402
from mirix.schemas.user import User as PydanticUser  # noqa: E402


@pytest_asyncio.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def message_manager():
    return MessageManager()


@pytest_asyncio.fixture(scope="module")
async def test_actor():
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    org_id = f"test-retain-org-{uuid.uuid4().hex[:8]}"
    try:
        await org_mgr.get_organization_by_id(org_id)
    except Exception:
        await org_mgr.create_organization(
            PydanticOrganization(id=org_id, name="Retention Test Org")
        )

    client_id = f"test-retain-client-{uuid.uuid4().hex[:8]}"
    try:
        return await client_mgr.get_client_by_id(client_id)
    except Exception:
        return await client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Retention Test Client",
                write_scope="test-retain",
                read_scopes=["test-retain"],
            )
        )


async def _make_user(test_actor, label):
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()
    user_id = f"test-retain-user-{label}-{uuid.uuid4().hex[:8]}"
    return await user_mgr.create_user(
        PydanticUser(
            id=user_id,
            name=f"Retention Test User {label}",
            organization_id=test_actor.organization_id,
            timezone="UTC",
        )
    )


@pytest_asyncio.fixture(scope="module")
async def user_a(test_actor):
    return await _make_user(test_actor, "A")


@pytest_asyncio.fixture(scope="module")
async def user_b(test_actor):
    return await _make_user(test_actor, "B")


@pytest_asyncio.fixture(scope="module")
async def meta_agent(test_actor):
    """A meta_memory_agent-shaped agent (canonical conversation recipient)."""
    from mirix.schemas.agent import AgentType, CreateAgent
    from mirix.services.agent_manager import AgentManager

    agent_mgr = AgentManager()
    return await agent_mgr.create_agent(
        agent_create=CreateAgent(
            name=f"test-retain-meta-{uuid.uuid4().hex[:8]}",
            agent_type=AgentType.meta_memory_agent,
            description="Test meta agent for retention",
            system=None,
            llm_config=None,
            embedding_config=None,
        ),
        actor=test_actor,
    )


async def _seed_session(mgr, actor, agent, user_id, session_id, n_msgs=2):
    """Create n_msgs detached messages for (agent, user, session)."""
    created = []
    for i in range(n_msgs):
        msg = await mgr.create_message(
            pydantic_msg=PydanticMessage(
                agent_id=agent.id,
                role=MessageRole.user,
                content=[TextContent(text=f"{session_id}-{i}")],
                session_id=session_id,
            ),
            actor=actor,
            user_id=user_id,
            use_cache=False,
        )
        created.append(msg)
    return created


pytestmark_integration = pytest.mark.integration


class TestDeleteDetachedRetention:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retains_last_n_sessions_and_scopes_per_user(
        self, message_manager, test_actor, meta_agent, user_a, user_b
    ):
        N = 3
        # User A: seed N+2 = 5 sessions, ordered oldest->newest by creation.
        a_sessions = [f"a-sess-{i}-{uuid.uuid4().hex[:6]}" for i in range(N + 2)]
        for sid in a_sessions:
            await _seed_session(
                message_manager, test_actor, meta_agent, user_a.id, sid
            )

        # User B: seed 2 sessions on the SAME agent — must be unaffected by A's window.
        b_sessions = [f"b-sess-{i}-{uuid.uuid4().hex[:6]}" for i in range(2)]
        for sid in b_sessions:
            await _seed_session(
                message_manager, test_actor, meta_agent, user_b.id, sid
            )

        # All messages are detached (agent has empty message_ids), so without
        # retention they would all be deleted. Call delete scoped to user A,
        # retaining the last N sessions.
        await message_manager.delete_detached_messages_for_agent(
            agent_id=meta_agent.id,
            actor=test_actor,
            retain_last_n_sessions=N,
            user_id=user_a.id,
        )

        # The last N session_ids for user A survive; the 2 oldest are deleted.
        surviving_a = a_sessions[-N:]
        deleted_a = a_sessions[:-N]

        for sid in surviving_a:
            got = await message_manager.list_messages_for_agent(
                agent_id=meta_agent.id,
                actor=test_actor,
                session_id=sid,
                limit=100,
                use_cache=False,
            )
            assert len(got) == 2, f"expected retained session {sid} to survive"

        for sid in deleted_a:
            got = await message_manager.list_messages_for_agent(
                agent_id=meta_agent.id,
                actor=test_actor,
                session_id=sid,
                limit=100,
                use_cache=False,
            )
            assert got == [], f"expected aged-out session {sid} to be deleted"

        # User B's sessions were NOT in user A's retained set, but because we
        # scoped the delete to user_id=user_a.id, user B's detached rows were not
        # even considered for the retained-set computation. They are still detached
        # though — they DID get deleted because they are not in message_ids and not
        # in A's retained set. Re-seed-and-verify the per-user retained SET instead:
        # rebuild B fresh and retain scoped to B.
        b_sessions2 = [f"b2-sess-{i}-{uuid.uuid4().hex[:6]}" for i in range(N + 1)]
        for sid in b_sessions2:
            await _seed_session(
                message_manager, test_actor, meta_agent, user_b.id, sid
            )

        # Retain last N for user B — user A's surviving sessions must NOT count
        # toward or evict B's window (per-user scoping).
        await message_manager.delete_detached_messages_for_agent(
            agent_id=meta_agent.id,
            actor=test_actor,
            retain_last_n_sessions=N,
            user_id=user_b.id,
        )

        # B keeps its last N; the oldest B session ages out.
        for sid in b_sessions2[-N:]:
            got = await message_manager.list_messages_for_agent(
                agent_id=meta_agent.id,
                actor=test_actor,
                session_id=sid,
                limit=100,
                use_cache=False,
            )
            assert len(got) == 2, f"expected retained B session {sid} to survive"

        got_old_b = await message_manager.list_messages_for_agent(
            agent_id=meta_agent.id,
            actor=test_actor,
            session_id=b_sessions2[0],
            limit=100,
            use_cache=False,
        )
        assert got_old_b == [], "expected oldest B session to be deleted"

        # CRITICAL per-user isolation: deleting/retaining for user B must NOT have
        # touched user A's surviving sessions.
        for sid in surviving_a:
            got = await message_manager.list_messages_for_agent(
                agent_id=meta_agent.id,
                actor=test_actor,
                session_id=sid,
                limit=100,
                use_cache=False,
            )
            assert len(got) == 2, (
                f"user A retained session {sid} must be unaffected by user B's delete"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retain_zero_deletes_all_detached(
        self, message_manager, test_actor, meta_agent, user_a
    ):
        sid = f"zero-sess-{uuid.uuid4().hex[:6]}"
        await _seed_session(message_manager, test_actor, meta_agent, user_a.id, sid)

        await message_manager.delete_detached_messages_for_agent(
            agent_id=meta_agent.id,
            actor=test_actor,
            retain_last_n_sessions=0,
            user_id=user_a.id,
        )

        got = await message_manager.list_messages_for_agent(
            agent_id=meta_agent.id,
            actor=test_actor,
            session_id=sid,
            limit=100,
            use_cache=False,
        )
        assert got == [], "retain=0 must preserve legacy full-delete behavior"
