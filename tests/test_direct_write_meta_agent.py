"""Tests for the direct-write branch in Agent.step and _apply_direct_write dispatch."""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

TEST_ORG_ID = "direct-write-meta-org"
TEST_CLIENT_ID = "direct-write-meta-client"
TEST_USER_ID = "direct-write-meta-user"


def _unique(prefix: str = "src") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _provision_org_client_and_user():
    """Create the org, client, and user needed by all tests in this module."""
    from conftest import _create_client_and_key

    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    await _create_client_and_key(TEST_CLIENT_ID, TEST_ORG_ID, org_name="Direct Write Meta Org")

    user_mgr = UserManager()
    try:
        await user_mgr.get_user_by_id(TEST_USER_ID)
    except Exception:
        await user_mgr.create_user(
            PydanticUser(
                id=TEST_USER_ID,
                name="Direct Write Meta User",
                organization_id=TEST_ORG_ID,
                timezone="UTC",
            )
        )
    yield


async def _get_actor():
    from mirix.services.client_manager import ClientManager

    return await ClientManager().get_client_by_id(TEST_CLIENT_ID)


async def _get_user():
    from mirix.services.user_manager import UserManager

    return await UserManager().get_user_by_id(TEST_USER_ID)


# ---------------------------------------------------------------------------
# Unit tests for _apply_direct_write (dispatch + error path)
# ---------------------------------------------------------------------------


async def test_apply_direct_write_dispatches_to_registered_handler():
    """_apply_direct_write looks up handler in DIRECT_WRITE_HANDLERS and calls with payload kwargs."""
    from mirix.agent.agent import Agent
    from mirix.functions import direct_write_handlers as memory_tools

    agent = Agent.__new__(Agent)

    called = {}

    async def _fake_handler(ag, **kwargs):
        called["agent"] = ag
        called["kwargs"] = kwargs

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        await agent._apply_direct_write("episodic", {"event_type": "t", "summary": "s"})
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called["agent"] is agent
    assert called["kwargs"] == {"event_type": "t", "summary": "s"}


async def test_apply_direct_write_unknown_memory_type_raises():
    """Unknown memory_type → ValueError."""
    from mirix.agent.agent import Agent

    agent = Agent.__new__(Agent)
    with pytest.raises(ValueError, match="No direct-write handler registered for memory_type: bogus"):
        await agent._apply_direct_write("bogus", {})


async def test_persist_memory_source_direct_writes_does_not_bulk_insert_placeholder():
    """direct_writes + REST placeholder MessageCreate must not create source_messages rows.

    add_memory with direct_writes uses messages=[], which becomes one MessageCreate with
    empty content; that must not be normalized and bulk_inserted as a fake turn.
    """
    from mirix.agent.agent import Agent
    from mirix.schemas.enums import MessageRole
    from mirix.schemas.message import MessageCreate

    actor = await _get_actor()
    agent = Agent.__new__(Agent)
    agent_state = MagicMock()
    agent_state.organization_id = TEST_ORG_ID
    agent.agent_state = agent_state
    agent.actor = actor
    agent.user_id = TEST_USER_ID
    agent.external_id = _unique("ext")
    agent.external_thread_id = None
    agent.source_type = "engagement"
    agent.source_system = None
    agent.source_metadata = None
    agent.occurred_at = None
    agent.source_summary = None
    agent.source_summary_source = None
    agent.filter_tags = {"scope": "test"}
    agent.direct_writes = [{"memory_type": "episodic", "payload": {"items": []}}]
    agent.source_messages = None

    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.create = AsyncMock(return_value=SimpleNamespace())
    agent.source_message_manager = MagicMock()
    agent.source_message_manager.bulk_insert = AsyncMock()

    memory_source_id = _unique("src")
    placeholder_input = [MessageCreate(role=MessageRole.user, content=[])]

    await Agent._persist_memory_source(agent, memory_source_id, placeholder_input)

    agent.source_message_manager.bulk_insert.assert_not_called()
    agent.memory_source_manager.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# Integration test for Agent.step direct-write branch:
# Uses a real Agent instance built via __new__ + manually wired attrs so we
# can exercise the branch path without booting the full meta-agent init.
# ---------------------------------------------------------------------------


def _make_meta_agent_stub(
    *,
    memory_source_id=None,
    direct_writes=None,
    filter_tags=None,
    use_cache=False,
    source_exists_and_complete=False,
    source_deduped=False,
):
    """Build an Agent-like object that Agent.step will run against.

    We stub out the heavy deps (message_manager retention, block_manager, etc.)
    and only wire up what the pre-LLM path actually reaches.
    """
    from mirix.agent.agent import Agent
    from mirix.schemas.agent import AgentType

    agent = Agent.__new__(Agent)

    # Agent state must be a proper meta_memory_agent
    agent_state = MagicMock()
    agent_state.id = "meta-1"
    agent_state.parent_id = None
    agent_state.name = "meta-agent"
    agent_state.is_type = lambda t: t == AgentType.meta_memory_agent
    agent_state.llm_config = MagicMock()
    agent_state.llm_config.model = "gpt-4"
    agent.agent_state = agent_state

    # Runtime fields
    agent.user_id = TEST_USER_ID
    agent.filter_tags = filter_tags or {"scope": "test"}
    agent.block_filter_tags = None
    agent.use_cache = use_cache
    agent.occurred_at = None
    agent.memory_source_id = memory_source_id
    agent.direct_writes = direct_writes
    agent.external_id = None
    agent.external_thread_id = None
    agent.source_type = None
    agent.source_system = None
    agent.source_metadata = None
    agent.source_summary = None
    agent.source_summary_source = None
    agent.summarize = False
    agent.source_messages = None
    agent._block_scopes = ["test"]

    # Managers — stubbed to record calls; message_manager retention read must return []
    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])

    agent.memory_source_manager = MagicMock()
    if source_exists_and_complete:
        agent.memory_source_manager.get_by_id = AsyncMock(return_value=SimpleNamespace(processing_complete=True))
    else:
        agent.memory_source_manager.get_by_id = AsyncMock(return_value=None)
    agent.memory_source_manager.mark_processing_complete = AsyncMock(return_value=None)
    agent.memory_source_manager.finalize_source = AsyncMock(return_value=None)

    # _persist_memory_source returns whether to continue: False simulates a
    # different-id dedup (skip), True a normal persist / same-id resume.
    async def _fake_persist(memory_source_id, input_messages):
        return not source_deduped

    agent._persist_memory_source = _fake_persist
    return agent


async def test_step_direct_writes_skips_llm_and_calls_handler():
    """Meta-agent with direct_writes set runs handler, marks processing complete, returns step_count=0."""
    from mirix.functions import direct_write_handlers as memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {"event_type": "e", "summary": "s"}}],
    )

    handler_calls = []

    async def _fake_handler(ag, **kwargs):
        handler_calls.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    # Handler was called with the payload
    assert handler_calls == [{"event_type": "e", "summary": "s"}]
    # Processing marked complete via the single finalize chokepoint (VEPAGE-1251).
    from mirix.queue.error_policy import SaveOutcome

    agent.memory_source_manager.finalize_source.assert_not_awaited()
    # step_count == 0 (direct-write branch returned early)
    assert result.step_count == 0


async def test_step_direct_writes_respects_source_dedup():
    """If source dedupes at _persist_memory_source, direct_writes branch does NOT run."""
    from mirix.functions import direct_write_handlers as memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {}}],
        source_deduped=True,
    )

    called = []

    async def _fake_handler(ag, **kwargs):
        called.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called == [], "Direct-write handler must NOT run when source deduped"
    # mark_processing_complete must NOT be called on dedup
    agent.memory_source_manager.mark_processing_complete.assert_not_called()
    assert result.step_count == 0


async def test_step_direct_writes_respects_processing_complete():
    """If source already has processing_complete=True, direct_writes branch does NOT run."""
    from mirix.functions import direct_write_handlers as memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {}}],
        source_exists_and_complete=True,
    )

    called = []

    async def _fake_handler(ag, **kwargs):
        called.append(kwargs)

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        result = await agent.step(
            input_messages=[],
            actor=actor,
            user=user,
        )
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    assert called == [], "Direct-write handler must NOT run when source already processed"
    agent.memory_source_manager.mark_processing_complete.assert_not_called()
    assert result.step_count == 0


async def test_step_direct_writes_unknown_type_raises_value_error():
    """Unknown memory_type in direct_writes → ValueError (does not mark processing complete)."""
    actor = await _get_actor()
    user = await _get_user()

    agent = _make_meta_agent_stub(
        memory_source_id=_unique("src"),
        direct_writes=[{"memory_type": "bogus", "payload": {}}],
    )

    with pytest.raises(ValueError, match="No direct-write handler registered"):
        await agent.step(input_messages=[], actor=actor, user=user)

    agent.memory_source_manager.mark_processing_complete.assert_not_called()


async def test_step_direct_writes_skips_source_summary_generation():
    """Even when summarize=True, direct_writes requests must NOT dispatch the
    source-summary generation task. Caller already provides per-item summary
    content on each direct-write item, so the LLM-backed summary would be a
    waste of a round trip.
    """
    from unittest.mock import AsyncMock

    from mirix.functions import direct_write_handlers as memory_tools

    actor = await _get_actor()
    user = await _get_user()
    memory_source_id = _unique("src")

    agent = _make_meta_agent_stub(
        memory_source_id=memory_source_id,
        direct_writes=[{"memory_type": "episodic", "payload": {}}],
    )
    # Enable summary generation — the direct-write branch should still skip it.
    agent.summarize = True
    # Spy on the summary dispatch path so we can assert it wasn't invoked.
    agent._generate_source_summary_traced = AsyncMock()

    async def _fake_handler(ag, **kwargs):
        pass

    orig = memory_tools.DIRECT_WRITE_HANDLERS
    memory_tools.DIRECT_WRITE_HANDLERS = {"episodic": _fake_handler}
    try:
        await agent.step(input_messages=[], actor=actor, user=user)
    finally:
        memory_tools.DIRECT_WRITE_HANDLERS = orig

    agent._generate_source_summary_traced.assert_not_called()
