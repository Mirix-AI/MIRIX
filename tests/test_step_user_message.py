"""The string-message entrypoint runs the real inner step without a server."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mirix.agent.agent import Agent
from mirix.schemas.agent import AgentStepResponse, AgentType
from mirix.schemas.message import Message
from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.openai.chat_completion_response import UsageStatistics


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_type", [AgentType.chat_agent, AgentType.episodic_memory_agent]
)
@pytest.mark.parametrize("fail", [False, True])
async def test_step_user_message_runs_inner_step_and_restores_session(agent_type, fail):
    agent = object.__new__(Agent)
    agent.agent_state = SimpleNamespace(
        id="agent-test",
        name="test",
        created_by_id="client-test",
        is_type=lambda kind: kind == agent_type,
        llm_config=SimpleNamespace(
            context_window=200_000, model="test", model_endpoint_type="openai"
        ),
    )
    agent.model = "test"
    agent.actor = SimpleNamespace(id="client-test")
    agent.user = SimpleNamespace(id="user-test")
    agent.user_id = "user-test"
    agent._current_step_session_id = "previous-session"
    agent._auto_summarize = False
    system = Message(role="system", content=[TextContent(text="You are an agent.")])
    answer = Message(role="assistant", content=[TextContent(text="Hello!")])
    agent.agent_manager = SimpleNamespace(
        get_in_context_messages=AsyncMock(return_value=[system]),
        append_to_in_context_messages=AsyncMock(return_value=agent.agent_state),
    )
    agent.step_manager = SimpleNamespace(
        log_step=AsyncMock(return_value=SimpleNamespace(id="step-test"))
    )
    agent.build_system_prompt_with_memories = AsyncMock(
        return_value=("You are an agent.", {})
    )
    agent._check_context_limit = AsyncMock()

    async def reply(**kwargs):
        assert agent._current_step_session_id == (
            "session-new" if agent_type == AgentType.chat_agent else None
        )
        assert kwargs["stream"] is False
        if fail:
            raise RuntimeError("model unavailable")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello!", tool_calls=None)
                )
            ],
            usage=UsageStatistics(
                prompt_tokens=12, completion_tokens=7, total_tokens=19
            ),
        )

    agent._get_ai_reply = AsyncMock(side_effect=reply)
    agent._handle_ai_response = AsyncMock(return_value=([answer], False, False))
    if fail:
        with pytest.raises(RuntimeError, match="model unavailable"):
            await agent.step_user_message(
                "hello", session_id="session-new", stream=False
            )
        agent.agent_manager.append_to_in_context_messages.assert_not_awaited()
    else:
        result = await agent.step_user_message(
            "hello", session_id="session-new", stream=False
        )
        assert isinstance(result, AgentStepResponse)
        [user_message, assistant_message] = result.messages
        assert json.loads(user_message.content[0].text)["message"] == "hello"
        assert user_message.session_id == (
            "session-new" if agent_type == AgentType.chat_agent else None
        )
        assert assistant_message is answer
        assert agent._handle_ai_response.await_args.args[0] is user_message
        assert (
            agent.agent_manager.append_to_in_context_messages.await_args.args[0]
            == result.messages
        )
        assert result.usage.total_tokens == 19
    assert agent._current_step_session_id == "previous-session"
