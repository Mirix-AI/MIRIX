"""
Tests for message handling after the message_ids refactor.

Tests cover:
1. get_messages_for_agent_user returns messages in chronological order
2. hard_delete_user_messages_for_agent deletes correct rows and keeps newest N
3. Retention=0 path: no DB persistence after step
4. Retention=N path: persists input messages and prunes to N newest
5. Context overflow summarization recovery
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mirix.agent.agent import Agent
from mirix.errors import ContextWindowExceededError
from mirix.schemas.agent import AgentState, AgentStepResponse, AgentType
from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import MessageRole
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.user import User
from mirix.services.message_manager import MessageManager


def make_client(id="client-1", org_id="org-1", retention=0):
    """Create a real Client object for tests."""
    return Client(
        id=id,
        organization_id=org_id,
        name="Test Client",
        status="active",
        write_scope="test",
        read_scopes=["test"],
        message_set_retention_count=retention,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def make_user(id="user-1", org_id="org-1"):
    """Create a real User object for tests."""
    return User(
        id=id,
        organization_id=org_id,
        name="Test User",
        status="active",
        timezone="UTC",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def make_pydantic_message(id: str, role: str = "user", user_id: str = "user-1") -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.id = id
    msg.role = role
    msg.user_id = user_id
    return msg


class TestGetMessagesForAgentUser:
    """Tests for MessageManager.get_messages_for_agent_user()"""

    def test_returns_messages_in_chronological_order(self):
        """DB returns newest-first; method should reverse to chronological."""
        manager = MessageManager()

        # Simulate DB returning newest-first (DESC order)
        msg_old = MagicMock()
        msg_old.to_pydantic.return_value = make_pydantic_message("msg-1")
        msg_new = MagicMock()
        msg_new.to_pydantic.return_value = make_pydantic_message("msg-2")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg_new, msg_old]  # newest first from DB

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.get_messages_for_agent_user(
                    agent_id="agent-1", user_id="user-1", actor=actor, limit=10
                )

        result = asyncio.run(run())

        # Should be reversed to chronological order
        assert len(result) == 2
        assert result[0].id == "msg-1"  # oldest first
        assert result[1].id == "msg-2"

    def test_returns_empty_when_no_messages(self):
        """Returns empty list when no messages exist."""
        manager = MessageManager()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.get_messages_for_agent_user(
                    agent_id="agent-1", user_id="user-1", actor=actor, limit=10
                )

        result = asyncio.run(run())
        assert result == []


class TestHardDeleteUserMessagesForAgent:
    """Tests for MessageManager.hard_delete_user_messages_for_agent()"""

    def test_deletes_all_when_keep_newest_n_is_zero(self):
        """keep_newest_n=0 means delete everything."""
        manager = MessageManager()

        delete_ids_result = MagicMock()
        delete_ids_result.all.return_value = [("msg-1",), ("msg-2",), ("msg-3",)]

        execute_results = [
            delete_ids_result,  # select IDs to delete
            MagicMock(),  # DELETE statement
        ]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=execute_results)
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                with patch("mirix.database.redis_client.get_redis_client", return_value=None):
                    actor = make_client()
                    return await manager.hard_delete_user_messages_for_agent(
                        agent_id="agent-1",
                        user_id="user-1",
                        actor=actor,
                        keep_newest_n=0,
                    )

        count = asyncio.run(run())
        assert count == 3

    def test_returns_zero_when_no_messages_exist(self):
        """Returns 0 when there are no messages to delete."""
        manager = MessageManager()

        delete_ids_result = MagicMock()
        delete_ids_result.all.return_value = []  # nothing to delete

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=delete_ids_result)
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.hard_delete_user_messages_for_agent(
                    agent_id="agent-1",
                    user_id="user-1",
                    actor=actor,
                    keep_newest_n=0,
                )

        count = asyncio.run(run())
        assert count == 0


class TestRetentionBehavior:
    """Tests that retention=0 vs retention>0 produces correct persistence behavior."""

    def test_client_default_retention_is_zero(self):
        """Clients default to message_set_retention_count=0."""
        client = make_client()
        assert (client.message_set_retention_count or 0) == 0

    def test_client_with_retention_has_correct_value(self):
        """Clients configured with retention=5 expose that value."""
        client = make_client(retention=5)
        assert client.message_set_retention_count == 5


def make_agent_state(
    agent_id: str,
    agent_type: AgentType,
    parent_id: str | None = None,
) -> AgentState:
    """Create a minimal AgentState for unit-testing Agent.step."""
    return AgentState(
        id=agent_id,
        name=agent_type.value,
        system="System prompt",
        agent_type=agent_type,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
        parent_id=parent_id,
    )


def make_runtime_message(agent_id: str, text: str) -> Message:
    """Create a runtime Message object used by Agent.step."""
    return Message.dict_to_message(
        agent_id=agent_id,
        model="gpt-4o-mini",
        openai_message_dict={"role": "user", "content": text},
    )


def build_step_test_agent(agent_state: AgentState, user: User) -> Agent:
    """Build an Agent instance with only fields required by step()."""
    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.user_id = user.id
    agent.client_id = "client-1"
    agent.model = "gpt-4o-mini"
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent._block_scopes = None
    agent.blocks_in_memory = None
    agent.last_function_response = None
    agent.block_manager = SimpleNamespace(get_blocks=AsyncMock(return_value=[]))
    agent.message_manager = SimpleNamespace(
        get_messages_for_agent_user=AsyncMock(return_value=[]),
        create_many_messages=AsyncMock(return_value=[]),
        hard_delete_user_messages_for_agent=AsyncMock(return_value=0),
    )
    agent._extract_topics_from_messages = AsyncMock(return_value="topic-a;topic-b")
    agent.inner_step = AsyncMock(
        return_value=AgentStepResponse(
            messages=[],
            continue_chaining=False,
            function_failed=False,
            usage=UsageStatistics(),
            traj={},
        )
    )
    agent.interface = SimpleNamespace(step_complete=lambda: None)
    return agent


class TestAgentStepRetentionAndTopics:
    @pytest.mark.asyncio
    async def test_step_reads_retention_from_parent_scope_for_sub_agent(self):
        user = make_user()
        client = make_client(retention=2)
        agent_state = make_agent_state(
            agent_id="agent-child",
            agent_type=AgentType.episodic_memory_agent,
            parent_id="agent-meta",
        )
        agent = build_step_test_agent(agent_state, user)
        agent.message_manager.get_messages_for_agent_user = AsyncMock(
            return_value=[make_runtime_message("agent-meta", "r1")]
        )

        with patch("mirix.agent.agent.LLMClient.create", return_value=object()):
            await agent.step(
                input_messages=make_runtime_message("agent-child", "current"),
                chaining=False,
                actor=client,
                user=user,
            )

        agent.message_manager.get_messages_for_agent_user.assert_awaited_once()
        read_kwargs = agent.message_manager.get_messages_for_agent_user.await_args.kwargs
        assert read_kwargs["agent_id"] == "agent-meta"
        assert read_kwargs["limit"] == 2
        agent.message_manager.create_many_messages.assert_not_awaited()
        agent.message_manager.hard_delete_user_messages_for_agent.assert_not_awaited()
        agent._extract_topics_from_messages.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_step_meta_persists_only_original_input_and_prunes(self):
        user = make_user()
        client = make_client(retention=2)
        agent_state = make_agent_state(
            agent_id="agent-meta",
            agent_type=AgentType.meta_memory_agent,
        )
        agent = build_step_test_agent(agent_state, user)
        original_input = make_runtime_message("agent-meta", "persist-me")
        heartbeat_like_message = make_runtime_message("agent-meta", "heartbeat-ish follow-up")
        agent.inner_step = AsyncMock(
            return_value=AgentStepResponse(
                messages=[heartbeat_like_message],
                continue_chaining=False,
                function_failed=False,
                usage=UsageStatistics(),
                traj={},
            )
        )

        with patch("mirix.agent.agent.LLMClient.create", return_value=object()):
            await agent.step(
                input_messages=original_input,
                chaining=False,
                actor=client,
                user=user,
            )

        persisted_messages = agent.message_manager.create_many_messages.await_args.args[0]
        assert len(persisted_messages) == 1
        assert persisted_messages[0].id == original_input.id

        prune_kwargs = agent.message_manager.hard_delete_user_messages_for_agent.await_args.kwargs
        assert prune_kwargs["agent_id"] == "agent-meta"
        assert prune_kwargs["keep_newest_n"] == 2

    @pytest.mark.asyncio
    async def test_step_meta_extracts_topics_from_retained_plus_current(self):
        user = make_user()
        client = make_client(retention=2)
        agent_state = make_agent_state(
            agent_id="agent-meta",
            agent_type=AgentType.meta_memory_agent,
        )
        agent = build_step_test_agent(agent_state, user)
        retained_1 = make_runtime_message("agent-meta", "retained-one")
        retained_2 = make_runtime_message("agent-meta", "retained-two")
        current = make_runtime_message("agent-meta", "current-input")
        agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[retained_1, retained_2])

        with patch("mirix.agent.agent.LLMClient.create", return_value=object()):
            await agent.step(
                input_messages=current,
                chaining=False,
                actor=client,
                user=user,
            )

        extract_arg = agent._extract_topics_from_messages.await_args.args[0]
        assert [m.id for m in extract_arg] == [retained_1.id, retained_2.id, current.id]

    @pytest.mark.asyncio
    async def test_step_retention_zero_skips_read_write_persistence(self):
        user = make_user()
        client = make_client(retention=0)
        agent_state = make_agent_state(
            agent_id="agent-meta",
            agent_type=AgentType.meta_memory_agent,
        )
        agent = build_step_test_agent(agent_state, user)

        with patch("mirix.agent.agent.LLMClient.create", return_value=object()):
            await agent.step(
                input_messages=make_runtime_message("agent-meta", "current-input"),
                chaining=False,
                actor=client,
                user=user,
            )

        agent.message_manager.get_messages_for_agent_user.assert_not_awaited()
        agent.message_manager.create_many_messages.assert_not_awaited()
        agent.message_manager.hard_delete_user_messages_for_agent.assert_not_awaited()


def _make_context_overflow_error():
    """Create an httpx error that is_context_overflow_error() recognises."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(
        400,
        json={
            "error": {
                "message": "This model's maximum context length is 8192 tokens",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
        request=request,
    )
    return httpx.HTTPStatusError(
        message="maximum context length",
        request=request,
        response=response,
    )


def build_inner_step_test_agent(agent_state: AgentState, user: User) -> Agent:
    """Build an Agent with mocks suitable for testing inner_step directly."""
    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.user_id = user.id
    agent.client_id = "client-1"
    agent.model = "gpt-4o-mini"
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent._block_scopes = None
    agent.blocks_in_memory = None
    agent.last_function_response = None
    agent.logger = MagicMock()
    agent.block_manager = SimpleNamespace(get_blocks=AsyncMock(return_value=[]))
    agent.message_manager = SimpleNamespace(
        get_messages_for_agent_user=AsyncMock(return_value=[]),
        create_many_messages=AsyncMock(return_value=[]),
        hard_delete_user_messages_for_agent=AsyncMock(return_value=0),
        create_message=AsyncMock(side_effect=lambda msg, **kw: msg),
        delete_message_by_id=AsyncMock(),
    )
    agent.step_manager = SimpleNamespace(
        log_step=AsyncMock(return_value=SimpleNamespace(id="step-1")),
    )
    agent.interface = SimpleNamespace(step_complete=lambda: None)
    agent.actor = make_client(retention=3)
    return agent


class TestSummarizeAndReplaceRetainedMessages:
    """Tests for Agent.summarize_and_replace_retained_messages()"""

    @pytest.mark.asyncio
    async def test_calls_summarize_and_persists_summary(self):
        """Verifies the method calls summarize_messages, persists the result,
        and deletes the original retained messages."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        retained = [
            make_runtime_message("agent-1", "old-msg-1"),
            make_runtime_message("agent-1", "old-msg-2"),
        ]

        with patch("mirix.agent.agent.summarize_messages", new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = "Summary of old messages"
            result = await agent.summarize_and_replace_retained_messages(retained)

        mock_summarize.assert_awaited_once()
        assert result.message_type == "summary"
        assert result.role == MessageRole.user
        assert result.content[0].text == "Summary of old messages"

        agent.message_manager.create_message.assert_awaited_once()
        assert agent.message_manager.delete_message_by_id.await_count == 2

    @pytest.mark.asyncio
    async def test_summary_message_has_correct_agent_and_user(self):
        """Summary message should be scoped to the correct agent and user."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        retained = [make_runtime_message("agent-1", "old-msg")]

        with patch("mirix.agent.agent.summarize_messages", new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = "Summary"
            result = await agent.summarize_and_replace_retained_messages(retained)

        assert result.agent_id == "agent-1"
        assert result.user_id == "user-1"

    @pytest.mark.asyncio
    async def test_summary_scoped_to_meta_agent_when_called_from_sub_agent(self):
        """When a sub-agent summarizes, the summary message's agent_id should
        be the parent (meta) agent, not the sub-agent itself."""
        user = make_user()
        agent_state = make_agent_state(
            "agent-child",
            AgentType.episodic_memory_agent,
            parent_id="agent-meta",
        )
        agent = build_inner_step_test_agent(agent_state, user)

        retained = [make_runtime_message("agent-meta", "old-msg")]

        with patch("mirix.agent.agent.summarize_messages", new_callable=AsyncMock) as mock_summarize:
            mock_summarize.return_value = "Summary"
            result = await agent.summarize_and_replace_retained_messages(retained)

        assert result.agent_id == "agent-meta"
        assert result.user_id == "user-1"


class TestContextOverflowSummarizationRecovery:
    """Tests for the summarization recovery path in inner_step."""

    @pytest.mark.asyncio
    async def test_summarization_triggered_on_overflow_with_retained_messages(self):
        """When context overflows and retained messages exist, summarization
        should be attempted and inner_step retried."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        retained_msg = make_runtime_message("agent-1", "retained")
        current_msg = make_runtime_message("agent-1", "current")
        summary_msg = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            content=[TextContent(text="Summary")],
            message_type="summary",
        )

        call_count = 0

        async def mock_get_ai_reply(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_context_overflow_error()
            # Second call succeeds
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok", tool_calls=None))],
                usage=UsageStatistics(),
                id="resp-1",
            )

        agent._get_ai_reply = mock_get_ai_reply
        agent.build_system_prompt_with_memories = AsyncMock(return_value=("system prompt", {}))
        agent.summarize_and_replace_retained_messages = AsyncMock(return_value=summary_msg)
        agent._handle_ai_response = AsyncMock(return_value=([], False, False))

        result = await agent.inner_step(
            messages=[current_msg],
            accumulated=[retained_msg],
            retained_count=1,
            chaining=False,
        )

        agent.summarize_and_replace_retained_messages.assert_awaited_once_with([retained_msg], None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_hard_fail_when_no_retained_messages(self):
        """When context overflows but there are no retained messages,
        ContextWindowExceededError should be raised immediately."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        current_msg = make_runtime_message("agent-1", "current")

        agent._get_ai_reply = AsyncMock(side_effect=_make_context_overflow_error())
        agent.build_system_prompt_with_memories = AsyncMock(return_value=("system prompt", {}))

        with pytest.raises(ContextWindowExceededError):
            await agent.inner_step(
                messages=[current_msg],
                accumulated=[],
                retained_count=0,
                chaining=False,
            )

    @pytest.mark.asyncio
    async def test_hard_fail_after_summarization_already_attempted(self):
        """If summarization was already attempted and context still overflows,
        raise ContextWindowExceededError."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        summary_msg = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            content=[TextContent(text="Summary")],
            message_type="summary",
        )
        current_msg = make_runtime_message("agent-1", "current")

        agent._get_ai_reply = AsyncMock(side_effect=_make_context_overflow_error())
        agent.build_system_prompt_with_memories = AsyncMock(return_value=("system prompt", {}))

        with pytest.raises(ContextWindowExceededError):
            await agent.inner_step(
                messages=[current_msg],
                accumulated=[summary_msg],
                retained_count=1,
                _summarization_attempted=True,
                chaining=False,
            )

    @pytest.mark.asyncio
    async def test_hard_fail_when_summarization_itself_fails(self):
        """If summarize_and_replace_retained_messages raises, the original
        context overflow should still surface as ContextWindowExceededError."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        retained_msg = make_runtime_message("agent-1", "retained")
        current_msg = make_runtime_message("agent-1", "current")

        agent._get_ai_reply = AsyncMock(side_effect=_make_context_overflow_error())
        agent.build_system_prompt_with_memories = AsyncMock(return_value=("system prompt", {}))
        agent.summarize_and_replace_retained_messages = AsyncMock(side_effect=RuntimeError("LLM summarization failed"))

        with pytest.raises(ContextWindowExceededError, match="summarization recovery failed"):
            await agent.inner_step(
                messages=[current_msg],
                accumulated=[retained_msg],
                retained_count=1,
                chaining=False,
            )

    @pytest.mark.asyncio
    async def test_chaining_outputs_preserved_after_summarization(self):
        """When summarization fires, chaining outputs (accumulated beyond
        retained_count) should be preserved in the retry."""
        user = make_user()
        agent_state = make_agent_state("agent-1", AgentType.meta_memory_agent)
        agent = build_inner_step_test_agent(agent_state, user)

        retained_msg = make_runtime_message("agent-1", "retained")
        chaining_msg = make_runtime_message("agent-1", "heartbeat")
        current_msg = make_runtime_message("agent-1", "current")
        summary_msg = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            content=[TextContent(text="Summary")],
            message_type="summary",
        )

        call_count = 0

        async def mock_get_ai_reply(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_context_overflow_error()
            return MagicMock(
                choices=[MagicMock(message=MagicMock(content="ok", tool_calls=None))],
                usage=UsageStatistics(),
                id="resp-1",
            )

        agent._get_ai_reply = mock_get_ai_reply
        agent.build_system_prompt_with_memories = AsyncMock(return_value=("system prompt", {}))
        agent.summarize_and_replace_retained_messages = AsyncMock(return_value=summary_msg)
        agent._handle_ai_response = AsyncMock(return_value=([], False, False))

        result = await agent.inner_step(
            messages=[current_msg],
            accumulated=[retained_msg, chaining_msg],
            retained_count=1,
            chaining=False,
        )

        assert result is not None
        # The retry should have been called with accumulated=[summary_msg, chaining_msg]
        # and retained_count=1 (for the single summary message)
        assert call_count == 2


class TestMessageTypeField:
    """Tests for the message_type field on Message schema."""

    def test_default_message_type_is_original(self):
        msg = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            content=[TextContent(text="hello")],
        )
        assert msg.message_type == "original"

    def test_summary_message_type(self):
        msg = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            content=[TextContent(text="summary text")],
            message_type="summary",
        )
        assert msg.message_type == "summary"
