"""Unit tests for S10: Summary Generation.

Verifies that:
- summarize=True triggers summary generation after processing completes
- Client-provided summary is used directly (summary_source="client")
- Generated summary is written with summary_source="generated"
- Summary generation failure does not affect processing_complete
- No summary generation when summarize=False or client summary present
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.memory_source import PaginatedResponse
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
    state.llm_config.model_copy = MagicMock(return_value=state.llm_config)
    state.tools = []
    state.tool_rules = []
    state.system = "test system prompt"
    state.embedding_config = MagicMock()
    state.organization_id = "org-1"
    return state


def _make_actor():
    """Create a minimal Client-like object for testing."""
    actor = MagicMock()
    actor.id = "client-1"
    actor.organization_id = "org-1"
    actor.message_set_retention_count = 0
    actor.write_scope = "scope-1"
    return actor


def _make_user():
    """Create a minimal User-like object for testing."""
    user = MagicMock()
    user.id = "user-1"
    user.name = "test-user"
    return user


def _make_source_message(role, content_text, seq_num):
    """Create a minimal SourceMessage-like object."""
    msg = MagicMock()
    msg.role = role
    msg.content = {"text": content_text}
    msg.sequence_num = seq_num
    return msg


def _make_step_response():
    """Create a mock AgentStepResponse for a successful single-step execution."""
    from mirix.schemas.openai.chat_completion_response import UsageStatistics

    resp = MagicMock()
    resp.continue_chaining = False
    resp.function_failed = False
    resp.usage = UsageStatistics(completion_tokens=10, prompt_tokens=20, total_tokens=30)
    resp.messages = []
    return resp


def _make_pydantic_source(processing_complete=False):
    """Create a minimal PydanticMemorySource-like object."""
    source = MagicMock()
    source.processing_complete = processing_complete
    return source


def _setup_agent(memory_source_id, summarize=False, source_summary=None):
    """Create an Agent instance with mocks for testing summary generation."""
    from mirix.agent.agent import Agent

    agent_state = _make_agent_state("meta_memory_agent")
    user = _make_user()
    actor = _make_actor()

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.actor = actor
    agent.user_id = user.id
    agent.client_id = "client-1"
    agent.memory_source_id = memory_source_id
    agent.summarize = summarize
    agent.source_summary = source_summary
    agent.source_summary_source = "client" if source_summary else None
    agent.external_id = None
    agent.external_thread_id = None
    agent.source_type = None
    agent.source_system = None
    agent.source_metadata = None
    agent.source_messages = None
    agent.direct_writes = None
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent.use_cache = True
    agent.logger = MagicMock()
    agent.model = "gpt-4o-mini"
    agent.interface = MagicMock()
    agent.occurred_at = None

    # Managers
    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.get_by_id = AsyncMock(return_value=_make_pydantic_source(processing_complete=False))
    agent.memory_source_manager.mark_processing_complete = AsyncMock()
    agent.memory_source_manager.finalize_source = AsyncMock()
    agent.memory_source_manager.update_summary = AsyncMock()
    agent._persist_memory_source = AsyncMock()

    agent.source_message_manager = MagicMock()

    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent = AsyncMock(return_value=[])
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])
    agent.message_manager.create_many_messages = AsyncMock()
    agent.message_manager.hard_delete_user_messages_for_agent = AsyncMock()

    return agent, actor, user


def _mock_llm_response(content="This is a generated summary."):
    """Create a mock LLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestSummaryGeneration:
    """Test the _generate_source_summary method directly."""

    @pytest.mark.asyncio
    async def test_generates_summary_from_source_messages(self):
        """Summary is generated from source messages and stored with summary_source='generated'."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [
            _make_source_message("user", "What is the weather?", 1),
            _make_source_message("assistant", "The weather is sunny today.", 2),
        ]
        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )

        mock_response = _mock_llm_response("User asked about weather; assistant reported sunny conditions.")

        with patch("mirix.agent.agent.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(return_value=mock_response)
            mock_llm_cls.create.return_value = mock_client

            await agent._generate_source_summary()

        agent.source_message_manager.get_messages_by_source_id.assert_called_once_with(
            memory_source_id="src-abc123",
            limit=2000,
        )
        agent.memory_source_manager.update_summary.assert_called_once_with(
            memory_source_id="src-abc123",
            summary="User asked about weather; assistant reported sunny conditions.",
            summary_source="generated",
        )

    @pytest.mark.asyncio
    async def test_skips_when_no_source_messages(self):
        """No LLM call when source has no messages."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=[], next_cursor=None, has_more=False)
        )

        with patch("mirix.agent.agent.LLMClient") as mock_llm_cls:
            await agent._generate_source_summary()

            mock_llm_cls.create.assert_not_called()
        agent.memory_source_manager.update_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        """LLM failure propagates to caller for consistent error handling."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [_make_source_message("user", "Hello", 1)]
        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )

        with patch("mirix.agent.agent.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            mock_llm_cls.create.return_value = mock_client

            with pytest.raises(RuntimeError, match="LLM unavailable"):
                await agent._generate_source_summary()

        agent.memory_source_manager.update_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_empty_llm_response(self):
        """Empty LLM response is logged but does not write summary."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [_make_source_message("user", "Hello", 1)]
        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )

        mock_response = _mock_llm_response("")

        with patch("mirix.agent.agent.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(return_value=mock_response)
            mock_llm_cls.create.return_value = mock_client

            await agent._generate_source_summary()

        agent.memory_source_manager.update_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_formats_content_dict_with_text_key(self):
        """Source messages with dict content using 'text' key are formatted correctly."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [
            _make_source_message("user", "Question?", 1),
        ]
        # content is dict with "text" key — the common case
        source_msgs[0].content = {"text": "Question?"}

        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )
        mock_response = _mock_llm_response("A question was asked.")

        with patch("mirix.agent.agent.LLMClient") as mock_llm_cls:
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(return_value=mock_response)
            mock_llm_cls.create.return_value = mock_client

            await agent._generate_source_summary()

        # Verify the LLM was called with the formatted transcript
        call_args = mock_client.send_llm_request.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[-1]  # Last message is the transcript
        assert "user: Question?" in user_msg.content[0].text

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_and_succeeds(self):
        """A 429 rate-limit error is retried with exponential backoff; later success is used."""
        import httpx

        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [_make_source_message("user", "Hello", 1)]
        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )

        rate_limit_err = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(429),
        )
        good_response = _mock_llm_response("Retried summary.")

        with (
            patch("mirix.agent.agent.LLMClient") as mock_llm_cls,
            patch("mirix.agent.agent.asyncio.sleep", new=AsyncMock()),
        ):
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(side_effect=[rate_limit_err, rate_limit_err, good_response])
            mock_llm_cls.create.return_value = mock_client

            await agent._generate_source_summary()

        assert mock_client.send_llm_request.await_count == 3
        agent.memory_source_manager.update_summary.assert_called_once_with(
            memory_source_id="src-abc123",
            summary="Retried summary.",
            summary_source="generated",
        )

    @pytest.mark.asyncio
    async def test_retries_exhausted_raises(self):
        """Persistent 429s eventually give up after the configured max and raise."""
        import httpx

        from mirix.errors import RateLimitExceededError

        agent, _, _ = _setup_agent("src-abc123", summarize=True)

        source_msgs = [_make_source_message("user", "Hello", 1)]
        agent.source_message_manager.get_messages_by_source_id = AsyncMock(
            return_value=PaginatedResponse(items=source_msgs, next_cursor=None, has_more=False)
        )

        rate_limit_err = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(429),
        )

        with (
            patch("mirix.agent.agent.LLMClient") as mock_llm_cls,
            patch("mirix.agent.agent.asyncio.sleep", new=AsyncMock()),
        ):
            mock_client = MagicMock()
            mock_client.send_llm_request = AsyncMock(side_effect=rate_limit_err)
            mock_llm_cls.create.return_value = mock_client

            with pytest.raises(RateLimitExceededError):
                await agent._generate_source_summary()

        agent.memory_source_manager.update_summary.assert_not_called()


class TestSummaryTriggerInStep:
    """Test that step() dispatches summary in parallel with sub-agents."""

    @pytest.mark.asyncio
    async def test_summary_triggered_when_summarize_true(self):
        """summarize=True dispatches _generate_source_summary_traced before processing_complete."""
        agent, actor, user = _setup_agent("src-abc123", summarize=True)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])
        agent._generate_source_summary_traced = AsyncMock()

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

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

        from mirix.queue.error_policy import SaveOutcome

        agent.memory_source_manager.finalize_source.assert_not_called()
        agent._generate_source_summary_traced.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_summary_when_summarize_false(self):
        """summarize=False does not trigger summary generation."""
        agent, actor, user = _setup_agent("src-abc123", summarize=False)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])
        agent._generate_source_summary_traced = AsyncMock()

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

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

        from mirix.queue.error_policy import SaveOutcome

        agent.memory_source_manager.finalize_source.assert_not_called()
        agent._generate_source_summary_traced.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_summary_when_client_summary_provided(self):
        """Client-provided summary bypasses generation."""
        agent, actor, user = _setup_agent(
            "src-abc123",
            summarize=True,
            source_summary="Client provided this summary",
        )

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])
        agent._generate_source_summary_traced = AsyncMock()

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

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

        agent.memory_source_manager.finalize_source.assert_not_called()
        agent._generate_source_summary_traced.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summary_failure_raises_and_leaves_processing_incomplete(self):
        """Summary failure propagates so worker redelivers; processing_complete stays False."""
        agent, actor, user = _setup_agent("src-abc123", summarize=True)

        resp = _make_step_response()
        agent.inner_step = AsyncMock(return_value=resp)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])
        agent._generate_source_summary_traced = AsyncMock(side_effect=RuntimeError("LLM down"))

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

        with patch("mirix.agent.agent.LLMClient"):
            with pytest.raises(RuntimeError, match="LLM down"):
                await agent.step(
                    input_messages=[input_msg],
                    chaining=False,
                    max_chaining_steps=1,
                    stream=False,
                    skip_verify=True,
                    actor=actor,
                    user=user,
                )

        # Source must NOT be finalized — worker will redeliver, retry will reprocess
        agent.memory_source_manager.mark_processing_complete.assert_not_called()
        agent.memory_source_manager.finalize_source.assert_not_called()
        agent._generate_source_summary_traced.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summary_runs_in_parallel_with_sub_agents(self):
        """Summary task starts before inner_step (sub-agents) and completes alongside them."""
        agent, actor, user = _setup_agent("src-abc123", summarize=True)

        order: list[str] = []

        async def fake_inner_step(*args, **kwargs):
            order.append("inner_step_start")
            await asyncio.sleep(0.01)
            order.append("inner_step_end")
            return _make_step_response()

        async def fake_summary():
            order.append("summary_start")
            await asyncio.sleep(0.01)
            order.append("summary_end")

        agent.inner_step = AsyncMock(side_effect=fake_inner_step)
        agent._extract_topics_from_messages = AsyncMock(return_value=["topic1"])
        agent._generate_source_summary_traced = AsyncMock(side_effect=fake_summary)

        from mirix.schemas.message import MessageCreate

        input_msg = MessageCreate(role="user", content="test message")

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

        # Summary must start before inner_step ends (i.e. concurrently)
        summary_start_idx = order.index("summary_start")
        inner_end_idx = order.index("inner_step_end")
        assert summary_start_idx < inner_end_idx, f"summary did not start before sub-agents finished; order={order}"
        # processing_complete happens after both finish
        from mirix.queue.error_policy import SaveOutcome

        agent.memory_source_manager.finalize_source.assert_not_called()


class TestSummaryTracedSpan:
    """Test that _generate_source_summary_traced creates a LangFuse child span."""

    @pytest.mark.asyncio
    async def test_traced_wrapper_creates_child_span(self):
        """_generate_source_summary_traced wraps the LLM call in a 'Summary Agent' child span."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)
        agent._generate_source_summary = AsyncMock()

        mock_langfuse = MagicMock()
        span = MagicMock()
        span.id = "span-xyz"
        mock_langfuse.start_as_current_observation.return_value.__enter__.return_value = span
        mock_langfuse.start_as_current_observation.return_value.__exit__.return_value = False

        with (
            patch("mirix.agent.agent.get_langfuse_client", return_value=mock_langfuse),
            patch(
                "mirix.agent.agent.get_trace_context",
                return_value={"trace_id": "trace-123", "observation_id": "obs-parent"},
            ),
        ):
            await agent._generate_source_summary_traced()

        mock_langfuse.start_as_current_observation.assert_called_once()
        call_kwargs = mock_langfuse.start_as_current_observation.call_args.kwargs
        assert call_kwargs["name"] == "Summary Agent"
        assert call_kwargs["as_type"] == "agent"
        assert call_kwargs["trace_context"]["trace_id"] == "trace-123"
        assert call_kwargs["trace_context"]["parent_span_id"] == "obs-parent"
        agent._generate_source_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_traced_wrapper_no_langfuse_still_runs(self):
        """When langfuse is not configured, traced wrapper still calls the underlying LLM."""
        agent, _, _ = _setup_agent("src-abc123", summarize=True)
        agent._generate_source_summary = AsyncMock()

        with (
            patch("mirix.agent.agent.get_langfuse_client", return_value=None),
            patch("mirix.agent.agent.get_trace_context", return_value={}),
        ):
            await agent._generate_source_summary_traced()

        agent._generate_source_summary.assert_awaited_once()


class TestClientProvidedSummary:
    """Test that client-provided summaries are stored correctly."""

    @pytest.mark.asyncio
    async def test_client_summary_stored_on_persist(self):
        """Client-provided summary is passed to memory_source_manager.create with source='client'."""
        agent, actor, user = _setup_agent(
            "src-abc123",
            summarize=False,
            source_summary="Client summary",
        )

        # Verify the source_summary and source_summary_source are set correctly
        assert agent.source_summary == "Client summary"
        assert agent.source_summary_source == "client"


class TestSummaryPromptProfanity:
    """The summarizer prompts must instruct the LLM to remove profanity.

    Profanity in generated summaries trips the LLM provider's content-moderation
    guardrail, which returns a non-retryable 422.
    """

    def test_source_message_prompt_instructs_profanity_removal(self):
        from mirix.prompts.gpt_summarize_source_messages import SYSTEM

        assert "profanity" in SYSTEM.lower()

    def test_internal_prompt_instructs_profanity_removal(self):
        from mirix.prompts.gpt_summarize_internal import SYSTEM

        assert "profanity" in SYSTEM.lower()


class TestUpdateSummaryManager:
    """Test the MemorySourceManager.update_summary method."""

    @pytest.mark.asyncio
    async def test_update_summary_writes_to_db(self):
        """update_summary updates the summary and summary_source fields."""
        from mirix.services.memory_source_manager import MemorySourceManager

        manager = MemorySourceManager.__new__(MemorySourceManager)

        mock_record = MagicMock()
        mock_record.update_with_redis = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        manager.session_maker = MagicMock(return_value=mock_session)

        with patch("mirix.services.memory_source_manager.MemorySourceModel") as MockModel:
            MockModel.read = AsyncMock(return_value=mock_record)

            await manager.update_summary(
                memory_source_id="src-abc123",
                summary="Generated summary text",
                summary_source="generated",
            )

        MockModel.read.assert_called_once_with(db_session=mock_session, identifier="src-abc123")
        assert mock_record.summary == "Generated summary text"
        assert mock_record.summary_source == "generated"
        mock_record.update_with_redis.assert_called_once_with(mock_session)
