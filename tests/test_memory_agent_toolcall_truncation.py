from datetime import datetime
from unittest.mock import MagicMock

from mirix.agent.agent import Agent
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client
from mirix.schemas.enums import ToolType
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import (
    FunctionCall,
)
from mirix.schemas.openai.chat_completion_response import Message as ChatCompletionMessage
from mirix.schemas.openai.chat_completion_response import (
    ToolCall,
)
from mirix.schemas.tool import Tool


def make_client(id="client-1", org_id="org-1"):
    return Client(
        id=id,
        organization_id=org_id,
        name="Test Client",
        status="active",
        scope="test",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def test_memory_agent_truncates_extra_tool_calls_and_executes_only_first():
    """
    Regression test for memory-agent multi-tool-call bug:
    If the LLM returns multiple tool calls in one response, memory agents must
    truncate to only the first tool call to avoid duplicate state mutations.
    """

    # Create a minimal Agent instance without running __init__
    agent = Agent.__new__(Agent)
    agent.logger = MagicMock()
    agent.interface = MagicMock()
    agent.model = "test-model"
    agent.last_function_response = None

    # ToolRulesSolver is referenced later in the agent loop; keep it simple here.
    agent.tool_rules_solver = MagicMock()
    agent.tool_rules_solver.update_tool_usage = MagicMock()
    agent.tool_rules_solver.has_children_tools = MagicMock(return_value=False)
    agent.tool_rules_solver.is_terminal_tool = MagicMock(return_value=False)

    # Provide an AgentState-like object
    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "agent-123"
    agent_state.name = "semantic_memory_agent"
    agent_state.agent_type = AgentType.semantic_memory_agent

    # Provide tools that match the function names in tool calls
    tool_1 = Tool(
        tool_type=ToolType.MIRIX_MEMORY_CORE,
        name="semantic_memory_update",
        json_schema={"name": "semantic_memory_update", "description": "test", "parameters": {}},
        return_char_limit=10000,
    )
    tool_2 = Tool(
        tool_type=ToolType.MIRIX_MEMORY_CORE,
        name="semantic_memory_insert",
        json_schema={"name": "semantic_memory_insert", "description": "test", "parameters": {}},
        return_char_limit=10000,
    )
    agent_state.tools = [tool_1, tool_2]
    agent.agent_state = agent_state

    executed = []

    def _fake_exec(function_name, function_args, *args, **kwargs):
        executed.append(function_name)
        return "ok"

    agent.execute_tool_and_persist_state = _fake_exec

    # Build input/response messages
    input_message = Message.dict_to_message(
        id="message-1",
        agent_id=agent_state.id,
        model=agent.model,
        openai_message_dict={"role": "user", "content": "hi"},
    )

    response_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="semantic_memory_update", arguments="{}"),
            ),
            ToolCall(
                id="call-2",
                function=FunctionCall(name="semantic_memory_insert", arguments="{}"),
            ),
        ],
    )

    agent._handle_ai_response(
        input_message=input_message,
        response_message=response_message,
        existing_file_uris=[],
        response_message_id="message-2",
        retrieved_memories=None,
        chaining=False,
    )

    assert executed == ["semantic_memory_update"]
