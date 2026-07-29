"""Unit tests for thread-level message dedup (ECMS-513).

Tests the filter_new_messages pure function and the get_seen_keys_for_thread
lookup, plus the Agent.step integration that wires them together.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.usage import MirixUsageStatistics
from mirix.services.source_message_manager import (
    compute_content_hash,
    filter_new_messages,
)


# ---------------------------------------------------------------------------
# filter_new_messages (pure function)
# ---------------------------------------------------------------------------


class TestFilterNewMessages:
    def test_all_new(self):
        msgs = [
            {"role": "user", "content": "hello", "external_message_id": "m1"},
            {"role": "assistant", "content": "hi", "external_message_id": "m2"},
        ]
        result = filter_new_messages(msgs, set(), set())
        assert result == msgs

    def test_filters_seen_by_ext_id(self):
        msgs = [
            {"role": "user", "content": "hello", "external_message_id": "m1"},
            {"role": "assistant", "content": "hi", "external_message_id": "m2"},
            {"role": "user", "content": "new msg", "external_message_id": "m3"},
        ]
        result = filter_new_messages(msgs, seen_ext_ids={"m1", "m2"}, seen_hashes=set())
        assert len(result) == 1
        assert result[0]["external_message_id"] == "m3"

    def test_falls_back_to_content_hash(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "new"},
        ]
        # normalize_message converts "hello" -> {"text": "hello"} before hashing
        seen_hash = compute_content_hash("user", {"text": "hello"})
        result = filter_new_messages(msgs, seen_ext_ids=set(), seen_hashes={seen_hash})
        assert len(result) == 1
        assert result[0]["content"] == "new"

    def test_dedupes_within_batch_by_ext_id(self):
        msgs = [
            {"role": "user", "content": "hello", "external_message_id": "m1"},
            {"role": "user", "content": "hello again", "external_message_id": "m1"},
        ]
        result = filter_new_messages(msgs, set(), set())
        assert len(result) == 1

    def test_dedupes_within_batch_by_hash(self):
        msgs = [
            {"role": "user", "content": "yes"},
            {"role": "user", "content": "yes"},
        ]
        result = filter_new_messages(msgs, set(), set())
        assert len(result) == 1

    def test_all_seen_returns_empty(self):
        msgs = [
            {"role": "user", "content": "hello", "external_message_id": "m1"},
            {"role": "assistant", "content": "hi", "external_message_id": "m2"},
        ]
        result = filter_new_messages(msgs, seen_ext_ids={"m1", "m2"}, seen_hashes=set())
        assert result == []

    def test_mixed_ext_id_and_hash_dedup(self):
        """Messages with ext_id use ext_id; those without use hash."""
        seen_hash = compute_content_hash("user", {"text": "old"})
        msgs = [
            {"role": "user", "content": "old", "external_message_id": "m1"},
            {"role": "user", "content": "old"},  # no ext_id, falls back to hash
            {"role": "user", "content": "new", "external_message_id": "m3"},
        ]
        result = filter_new_messages(msgs, seen_ext_ids={"m1"}, seen_hashes={seen_hash})
        assert len(result) == 1
        assert result[0]["external_message_id"] == "m3"

    def test_preserves_order(self):
        msgs = [
            {"role": "user", "content": "a", "external_message_id": "m1"},
            {"role": "user", "content": "b", "external_message_id": "m2"},
            {"role": "user", "content": "c", "external_message_id": "m3"},
        ]
        result = filter_new_messages(msgs, seen_ext_ids={"m2"}, seen_hashes=set())
        assert [m["external_message_id"] for m in result] == ["m1", "m3"]

    def test_empty_input(self):
        assert filter_new_messages([], set(), set()) == []


# ---------------------------------------------------------------------------
# Agent.step integration — thread message dedup
# ---------------------------------------------------------------------------


def _make_agent_state():
    from mirix.schemas.agent import AgentType

    state = MagicMock()
    state.id = "agent-123"
    state.name = "meta_memory_agent"
    state.created_by_id = "client-1"
    state.parent_id = None
    state.agent_type = AgentType.meta_memory_agent
    state.is_type = lambda t: t == AgentType.meta_memory_agent
    state.llm_config = MagicMock()
    state.llm_config.model = "gpt-4o-mini"
    state.llm_config.model_endpoint_type = "openai"
    state.llm_config.context_window = 128000
    state.tools = []
    state.tool_rules = []
    state.system = "test system prompt"
    state.embedding_config = MagicMock()
    return state


def _setup_agent(
    memory_source_id="src-123",
    external_thread_id="thread-1",
    external_id=None,
    source_messages=None,
):
    from mirix.agent.agent import Agent

    agent_state = _make_agent_state()
    actor = MagicMock()
    actor.id = "client-1"
    actor.organization_id = "org-1"
    actor.message_set_retention_count = 0
    user = MagicMock()
    user.id = "user-1"
    user.name = "test-user"

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.actor = actor
    agent.user_id = user.id
    agent.memory_source_id = memory_source_id
    agent.direct_writes = None
    agent.external_id = external_id
    agent.external_thread_id = external_thread_id
    agent.source_type = "conversation"
    agent.source_system = None
    agent.source_metadata = None
    agent.source_summary = None
    agent.source_summary_source = None
    agent.summarize = False
    agent.source_messages = source_messages
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent.use_cache = True
    agent.client_id = "client-1"
    agent.logger = MagicMock()
    agent.model = "gpt-4o-mini"
    agent.blocks_in_memory = None
    agent.interface = MagicMock()
    agent._block_scopes = None

    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.get_by_id = AsyncMock(return_value=None)
    agent.source_message_manager = MagicMock()
    agent.source_message_manager.get_seen_keys_for_thread = AsyncMock(
        return_value=(set(), set())
    )

    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent = AsyncMock(return_value=[])
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])
    agent.message_manager.create_many_messages = AsyncMock()
    agent.message_manager.hard_delete_user_messages_for_agent = AsyncMock()

    agent._persist_memory_source = AsyncMock(return_value=True)

    return agent, actor, user


class TestThreadMessageDedupInStep:
    @pytest.mark.asyncio
    async def test_overlapping_send_filters_seen_messages(self):
        """Send #2 with overlap: only new messages reach persist + LLM."""
        source_messages = [
            {"role": "user", "content": "old msg", "external_message_id": "m4"},
            {"role": "assistant", "content": "old reply", "external_message_id": "m5"},
            {"role": "user", "content": "new msg", "external_message_id": "m6"},
        ]
        agent, actor, user = _setup_agent(source_messages=source_messages)
        agent.source_message_manager.get_seen_keys_for_thread = AsyncMock(
            return_value=({"m4", "m5"}, set())
        )

        resp = MagicMock()
        resp.continue_chaining = False
        resp.function_failed = False
        resp.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        resp.messages = []
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="placeholder")

        with patch("mirix.agent.agent.LLMClient"):
            await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        # source_messages stays intact for full provenance storage
        assert len(agent.source_messages) == 3
        agent._persist_memory_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_seen_skips_entirely(self):
        """When every message is already seen, step() short-circuits."""
        source_messages = [
            {"role": "user", "content": "old", "external_message_id": "m1"},
            {"role": "assistant", "content": "old", "external_message_id": "m2"},
        ]
        agent, actor, user = _setup_agent(source_messages=source_messages)
        agent.source_message_manager.get_seen_keys_for_thread = AsyncMock(
            return_value=({"m1", "m2"}, set())
        )

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="placeholder")

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
        assert result.step_count == 0
        agent._persist_memory_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_external_id_bypasses_thread_dedup(self):
        """When external_id is set, thread-level dedup is skipped entirely."""
        source_messages = [
            {"role": "user", "content": "msg", "external_message_id": "m1"},
        ]
        agent, actor, user = _setup_agent(
            source_messages=source_messages,
            external_id="call-123",
        )

        resp = MagicMock()
        resp.continue_chaining = False
        resp.function_failed = False
        resp.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        resp.messages = []
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="placeholder")

        with patch("mirix.agent.agent.LLMClient"):
            await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        agent.source_message_manager.get_seen_keys_for_thread.assert_not_called()
        agent._persist_memory_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_skips_thread_dedup(self):
        """On Kafka retry (source exists, processing_complete=False), thread
        dedup is skipped so _persist_memory_source handles idempotency."""
        source_messages = [
            {"role": "user", "content": "old msg", "external_message_id": "m4"},
            {"role": "assistant", "content": "old reply", "external_message_id": "m5"},
            {"role": "user", "content": "new msg", "external_message_id": "m6"},
        ]
        agent, actor, user = _setup_agent(source_messages=source_messages)
        # Simulate retry: source exists but not complete
        existing_source = MagicMock()
        existing_source.processing_complete = False
        agent.memory_source_manager.get_by_id = AsyncMock(return_value=existing_source)

        resp = MagicMock()
        resp.continue_chaining = False
        resp.function_failed = False
        resp.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        resp.messages = []
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="placeholder")

        with patch("mirix.agent.agent.LLMClient"):
            await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        agent.source_message_manager.get_seen_keys_for_thread.assert_not_called()
        assert len(agent.source_messages) == 3
        agent._persist_memory_source.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_thread_id_bypasses_thread_dedup(self):
        """Without external_thread_id, thread-level dedup is skipped."""
        source_messages = [
            {"role": "user", "content": "msg", "external_message_id": "m1"},
        ]
        agent, actor, user = _setup_agent(
            source_messages=source_messages,
            external_thread_id=None,
        )

        resp = MagicMock()
        resp.continue_chaining = False
        resp.function_failed = False
        resp.usage = MagicMock(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        resp.messages = []
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="placeholder")

        with patch("mirix.agent.agent.LLMClient"):
            await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

        agent.source_message_manager.get_seen_keys_for_thread.assert_not_called()
        agent._persist_memory_source.assert_called_once()
