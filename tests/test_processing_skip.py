"""Unit tests for processing-level idempotency skip.

Verifies that the meta_memory_agent's step() returns early when a memory
source has already been fully processed (processing_complete=True).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.usage import MirixUsageStatistics


def _make_agent_state(agent_type_name="meta_memory_agent"):
    """Create a minimal AgentState-like object for testing."""
    from mirix.schemas.agent import AgentType

    state = MagicMock()
    state.id = "agent-123"
    state.name = agent_type_name
    state.created_by_id = "client-1"
    state.parent_id = None
    state.agent_type = getattr(AgentType, agent_type_name)
    state.is_type = lambda t: t == getattr(AgentType, agent_type_name)
    state.llm_config = MagicMock()
    state.llm_config.model = "gpt-4o-mini"
    state.llm_config.model_endpoint_type = "openai"
    state.llm_config.context_window = 128000
    state.tools = []
    state.tool_rules = []
    state.system = "test system prompt"
    state.embedding_config = MagicMock()
    return state


def _make_actor():
    """Create a minimal Client-like object for testing."""
    actor = MagicMock()
    actor.id = "client-1"
    actor.organization_id = "org-1"
    actor.message_set_retention_count = 0  # No retention — avoids DB reads
    return actor


def _make_user():
    """Create a minimal User-like object for testing."""
    user = MagicMock()
    user.id = "user-1"
    user.name = "test-user"
    return user


def _make_pydantic_source(processing_complete=False):
    """Create a minimal PydanticMemorySource-like object."""
    source = MagicMock()
    source.processing_complete = processing_complete
    return source


def _setup_agent(memory_source_id, source_processing_complete=None):
    """Create an Agent instance with common mocks for testing step()."""
    from mirix.agent.agent import Agent

    agent_state = _make_agent_state("meta_memory_agent")
    user = _make_user()
    actor = _make_actor()

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.actor = actor
    agent.user_id = user.id
    agent.memory_source_id = memory_source_id
    agent.direct_writes = None
    agent.memory_source_manager = MagicMock()
    if source_processing_complete is not None:
        agent.memory_source_manager.get_by_id = AsyncMock(
            return_value=_make_pydantic_source(processing_complete=source_processing_complete)
        )
    else:
        agent.memory_source_manager.get_by_id = AsyncMock(return_value=None)
    agent.memory_source_manager.mark_processing_complete = AsyncMock()
    agent.memory_source_manager.finalize_source = AsyncMock()
    agent._persist_memory_source = AsyncMock()
    agent.interface = MagicMock()
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent.use_cache = True
    agent.client_id = "client-1"
    agent.logger = MagicMock()
    agent.model = "gpt-4o-mini"
    agent.summarize = False
    agent.source_summary = None
    agent.source_summary_source = None

    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent = AsyncMock(return_value=[])
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])
    agent.message_manager.create_many_messages = AsyncMock()
    agent.message_manager.hard_delete_user_messages_for_agent = AsyncMock()
    agent.source_message_manager = MagicMock()

    return agent, actor, user


def _make_step_response():
    """Create a mock AgentStepResponse for a successful single-step execution."""
    from mirix.schemas.openai.chat_completion_response import UsageStatistics

    resp = MagicMock()
    resp.continue_chaining = False
    resp.function_failed = False
    resp.usage = UsageStatistics(completion_tokens=10, prompt_tokens=20, total_tokens=30)
    resp.messages = []
    return resp


class TestProcessingSkip:
    """Test that step() skips processing when source is already complete."""

    @pytest.mark.asyncio
    async def test_skip_when_processing_complete(self):
        """Already-processed source (processing_complete=true) is skipped on redelivery."""
        agent, actor, user = _setup_agent("src-abc123", source_processing_complete=True)

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient") as mock_llm:
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        assert isinstance(result, MirixUsageStatistics)
        assert result.step_count == 0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_redelivery_of_complete_source_skips_before_persist(self):
        """An already-complete source (redelivery) must short-circuit BEFORE
        _persist_memory_source runs.

        This is the ordering that lets fault directives resolve before persist
        (so a tool=source_messages fault can fire on the dedup write) without
        ever injecting on a redelivery of a completed source: the
        processing-complete check sits before persist, so a completed
        redelivery returns before any source_messages write happens.
        """
        agent, actor, user = _setup_agent("src-abc123", source_processing_complete=True)

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        assert result.step_count == 0
        # The whole point: no persist (hence no source_messages write) on a
        # completed-source redelivery.
        agent._persist_memory_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_skip_when_processing_incomplete(self):
        """Source with processing_complete=false proceeds normally."""
        agent, actor, user = _setup_agent("src-abc123", source_processing_complete=False)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        assert isinstance(result, MirixUsageStatistics)
        agent.inner_step.assert_called_once()
        from mirix.queue.error_policy import SaveOutcome

        agent.memory_source_manager.finalize_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_skip_check_without_memory_source_id(self):
        """Without memory_source_id, no processing skip check occurs."""
        agent, actor, user = _setup_agent(None)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        agent._persist_memory_source.assert_not_called()
        agent.memory_source_manager.get_by_id.assert_not_called()
        agent.inner_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_processing_complete_stays_false_on_agent_failure(self):
        """Partial agent failure leaves processing_complete=false."""
        agent, actor, user = _setup_agent("src-abc123", source_processing_complete=False)

        agent.inner_step = AsyncMock(side_effect=RuntimeError("Agent crashed"))
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            with pytest.raises(RuntimeError, match="Agent crashed"):
                await agent.step(
                    input_messages=[input_msg],
                    chaining=False,
                    max_chaining_steps=1,
                    stream=False,
                    skip_verify=True,
                    actor=actor,
                    user=user,
                )

        agent.memory_source_manager.mark_processing_complete.assert_not_called()
        agent.memory_source_manager.finalize_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_when_source_not_found(self):
        """If get_by_id returns None after persist, processing continues (defensive)."""
        agent, actor, user = _setup_agent("src-abc123", source_processing_complete=None)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=None)

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            result = await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        agent.inner_step.assert_called_once()
