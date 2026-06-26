"""Tests for the swallow fix on the save path.

When a tool body raises a non-correctable exception, the exception must
propagate out of `_handle_ai_response` (and therefore out of `step()`) so the
worker / `process_with_policy` machinery sees the real exception type and can
classify it.

LLM-correctable errors (bad tool args, validation against the schema, malformed
JSON) stay contained: friendly-stringified, flagged `overall_function_failed=True`,
and fed back to the LLM for a bounded re-prompt.

This is sub-piece S1 of VEPAGE-1251 / the save-path error-handling design
(docs/superpowers/specs/2026-06-06-mirix-save-path-error-handling-design.md).

Acts as the VEPAGE-1228 regression anchor: an AttributeError from inside a
tool body must escape, not turn into "Error executing function ..." then
processing_complete=True.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from mirix.agent.agent import Agent
from mirix.errors import CorrectableToolError
from mirix.schemas.agent import AgentState, AgentType
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


def _make_meta_agent() -> Agent:
    """Build a meta-memory-agent Agent stub suitable for driving
    `_handle_ai_response` without a real DB."""
    agent = Agent.__new__(Agent)
    agent.logger = MagicMock()
    agent.interface = MagicMock()
    agent.model = "test-model"
    agent.last_function_response = None

    agent.tool_rules_solver = MagicMock()
    agent.tool_rules_solver.update_tool_usage = MagicMock()
    agent.tool_rules_solver.has_children_tools = MagicMock(return_value=False)
    agent.tool_rules_solver.is_terminal_tool = MagicMock(return_value=False)

    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "agent-meta"
    agent_state.name = "meta_memory_agent"
    agent_state.agent_type = AgentType.meta_memory_agent

    tool = Tool(
        tool_type=ToolType.MIRIX_MEMORY_CORE,
        name="trigger_memory_update",
        json_schema={
            "name": "trigger_memory_update",
            "description": "test",
            "parameters": {},
        },
        return_char_limit=10000,
    )
    agent_state.tools = [tool]
    agent.agent_state = agent_state
    return agent


def _make_messages(function_name: str = "trigger_memory_update"):
    input_message = Message.dict_to_message(
        id="message-12345678",
        agent_id="agent-meta",
        model="test-model",
        openai_message_dict={"role": "user", "content": "hi"},
    )
    response_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name=function_name, arguments="{}"),
            ),
        ],
    )
    return input_message, response_message


@pytest.mark.asyncio
async def test_non_correctable_tool_exception_propagates_out_of_handle_ai_response():
    """A tool body raising AttributeError (the VEPAGE-1228 shape) must NOT
    be swallowed into a friendly string. The original exception type must
    escape so `process_with_policy` can classify it."""
    agent = _make_meta_agent()

    async def _exec_that_raises_in_body(*args, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'foo'")

    agent.execute_tool_and_persist_state = _exec_that_raises_in_body

    input_message, response_message = _make_messages()

    with pytest.raises(AttributeError, match="NoneType"):
        await agent._handle_ai_response(
            input_message=input_message,
            response_message=response_message,
            existing_file_uris=[],
            response_message_id="message-87654321",
            retrieved_memories=None,
            chaining=False,
        )


@pytest.mark.asyncio
async def test_correctable_tool_error_is_contained_and_flags_function_failed():
    """CorrectableToolError stays contained: the LLM sees a friendly error
    message and `function_failed=True` is flagged so the bounded re-prompt
    can run. The exception itself does NOT escape."""
    agent = _make_meta_agent()

    async def _exec_that_raises_correctable(*args, **kwargs):
        raise CorrectableToolError("trigger_memory_update missing required arg 'memory_types'")

    agent.execute_tool_and_persist_state = _exec_that_raises_correctable

    input_message, response_message = _make_messages()

    messages, _, function_failed = await agent._handle_ai_response(
        input_message=input_message,
        response_message=response_message,
        existing_file_uris=[],
        response_message_id="message-87654321",
        retrieved_memories=None,
        chaining=False,
    )

    assert function_failed is True, (
        "Correctable tool errors must flag function_failed so the bounded " "re-prompt fires."
    )
    # The friendly error message should be appended as the tool response
    # so the LLM can read it on the next turn.
    tool_msgs = [m for m in messages if getattr(m, "role", None) == "tool"]
    assert tool_msgs, "expected a tool-role message capturing the error"


@pytest.mark.asyncio
async def test_db_operational_error_propagates():
    """SQLAlchemy OperationalError (DB blip from inside a tool) must escape
    so the policy retries it instead of marking the source complete."""
    from sqlalchemy.exc import OperationalError

    agent = _make_meta_agent()

    async def _exec_that_raises_db(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection reset"))

    agent.execute_tool_and_persist_state = _exec_that_raises_db

    input_message, response_message = _make_messages()

    with pytest.raises(OperationalError):
        await agent._handle_ai_response(
            input_message=input_message,
            response_message=response_message,
            existing_file_uris=[],
            response_message_id="message-87654321",
            retrieved_memories=None,
            chaining=False,
        )


# ---------- Synthetic pre-execution CorrectableToolError paths ----------
#
# `_handle_ai_response` raises CorrectableToolError BEFORE the tool body
# runs in three cases:
#   1) tool name not in agent_state.tools
#   2) tool arguments fail JSON parsing
#   3) tool arguments fail validate_tool_args
# All three should be caught by the single CorrectableToolError handler and
# converted into a tool-role message with function_failed=True. The original
# exception must NOT escape, and execute_tool_and_persist_state must NOT be
# called.


@pytest.mark.asyncio
async def test_unknown_tool_name_raises_correctable_error_contained():
    """LLM hallucinates a tool name → CorrectableToolError, contained."""
    agent = _make_meta_agent()

    execute_called = {"flag": False}

    async def _exec_should_not_run(*args, **kwargs):
        execute_called["flag"] = True
        return "should not be called"

    agent.execute_tool_and_persist_state = _exec_should_not_run

    input_message, _ = _make_messages()
    response_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="not_a_real_tool_name_xyz", arguments="{}"),
            ),
        ],
    )

    messages, _, function_failed = await agent._handle_ai_response(
        input_message=input_message,
        response_message=response_message,
        existing_file_uris=[],
        response_message_id="message-87654321",
        retrieved_memories=None,
        chaining=False,
    )

    assert function_failed is True
    assert execute_called["flag"] is False, (
        "execute_tool_and_persist_state must NOT be called for an unknown "
        "tool name — the CorrectableToolError raises pre-execution."
    )
    tool_msgs = [m for m in messages if getattr(m, "role", None) == "tool"]
    assert tool_msgs, "expected a tool-role message capturing the friendly error"


@pytest.mark.asyncio
async def test_bad_json_args_raises_correctable_error_contained():
    """LLM emits unparseable JSON for tool args → CorrectableToolError.
    Empty string is the canonical "parse_json raises" input."""
    agent = _make_meta_agent()

    execute_called = {"flag": False}

    async def _exec_should_not_run(*args, **kwargs):
        execute_called["flag"] = True
        return "should not be called"

    agent.execute_tool_and_persist_state = _exec_should_not_run

    input_message, _ = _make_messages()
    response_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(
                    name="trigger_memory_update",
                    arguments="",
                ),
            ),
        ],
    )

    messages, _, function_failed = await agent._handle_ai_response(
        input_message=input_message,
        response_message=response_message,
        existing_file_uris=[],
        response_message_id="message-87654321",
        retrieved_memories=None,
        chaining=False,
    )

    assert function_failed is True
    assert execute_called["flag"] is False, (
        "execute_tool_and_persist_state must NOT be called for unparseable "
        "JSON args — the CorrectableToolError raises pre-execution."
    )
    tool_msgs = [m for m in messages if getattr(m, "role", None) == "tool"]
    assert tool_msgs, "expected a tool-role message capturing the friendly error"


@pytest.mark.asyncio
async def test_args_coerced_to_wrong_shape_raises_correctable_error():
    """parse_json's json_repair fallback may coerce malformed input to a
    list. That's not a usable shape — CorrectableToolError it so the LLM
    can re-emit, not so it crashes downstream in _filter_function_args."""
    agent = _make_meta_agent()

    execute_called = {"flag": False}

    async def _exec_should_not_run(*args, **kwargs):
        execute_called["flag"] = True
        return "should not be called"

    agent.execute_tool_and_persist_state = _exec_should_not_run

    input_message, _ = _make_messages()
    response_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(
                    name="trigger_memory_update",
                    arguments="{invalid",  # json_repair coerces this to a list
                ),
            ),
        ],
    )

    messages, _, function_failed = await agent._handle_ai_response(
        input_message=input_message,
        response_message=response_message,
        existing_file_uris=[],
        response_message_id="message-87654321",
        retrieved_memories=None,
        chaining=False,
    )

    assert function_failed is True
    assert execute_called["flag"] is False
    tool_msgs = [m for m in messages if getattr(m, "role", None) == "tool"]
    assert tool_msgs
