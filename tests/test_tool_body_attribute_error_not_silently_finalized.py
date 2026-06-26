"""Regression anchor: a tool-body AttributeError must not be laundered into a
"complete" save.

The bug this guards against was silent data loss: a tool body raised an
`AttributeError` (in `Resolve Child Agents`), the save-path swallow turned it
into a friendly string, `_handle_ai_response` detected "function_failed" via the
string-prefix check, the bounded re-prompt ran one extra "please finish"
round-trip, then the loop broke and `mark_processing_complete` ran
unconditionally. The source was marked complete with zero memories extracted.

This test reproduces the exact failure mode end-to-end (the swallow chain, not
the LLM driver itself) and asserts the fixed behavior:

  Old (buggy):  AttributeError -> processing_complete=True (no memories),
                no exception propagated.

  Fixed:        AttributeError propagates out of step() so process_with_policy
                sees the typed exception AND `finalize_source` is NOT called.

Together with the origin-split (S4), classify() additionally maps a
pure-Python AttributeError with no provider frame to PERMANENT — so
the source is marked PERMANENT_FAILURE, not retried for hours.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirix.agent.agent import Agent
from mirix.queue.error_policy import Bucket, classify
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


def _meta_agent_stub() -> Agent:
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

    agent.memory_source_id = "src-1228"
    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.mark_processing_complete = AsyncMock()
    agent.memory_source_manager.finalize_source = AsyncMock()
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
async def test_attribute_error_propagates_not_silently_finalized():
    """The headline bug, end-to-end at the layer it was swallowed. Fixed
    behavior:

    * AttributeError is NOT caught by _handle_ai_response — it escapes.
    * mark_processing_complete is NOT called.
    * finalize_source is NOT called.
    * The propagated exception (when reached by process_with_policy)
      classifies PERMANENT via the origin-split, NOT TRANSIENT (which
      would burn redeliveries).
    """
    agent = _meta_agent_stub()

    async def _resolve_child_agents_attribute_error(*args, **kwargs):
        # Mirrors the actual failure shape — a pure-Python AttributeError
        # raised inside the tool body, no provider frame.
        raise AttributeError("'NoneType' object has no attribute 'is_type'")

    agent.execute_tool_and_persist_state = _resolve_child_agents_attribute_error

    input_message, response_message = _make_messages()

    # 1) The exception escapes _handle_ai_response (no silent swallow).
    with pytest.raises(AttributeError) as excinfo:
        await agent._handle_ai_response(
            input_message=input_message,
            response_message=response_message,
            existing_file_uris=[],
            response_message_id="message-87654321",
            retrieved_memories=None,
            chaining=False,
        )

    # 2) Neither mark_processing_complete nor the new finalize chokepoint
    #    were called. The source is left NOT-complete; the worker /
    #    policy will see the typed exception and decide.
    agent.memory_source_manager.mark_processing_complete.assert_not_awaited()
    agent.memory_source_manager.finalize_source.assert_not_awaited()

    # 3) The classifier identifies this shape as PERMANENT, NOT TRANSIENT.
    #    Crucial — before S4, AttributeError fell through to the
    #    Transient default and got 3x whole-step retries + numaflow
    #    redeliveries for an exception that never succeeds on retry.
    assert classify(excinfo.value) is Bucket.PERMANENT, (
        "this shape (AttributeError, no provider frame in traceback) must "
        "classify PERMANENT so the policy fails fast — retrying a deterministic "
        "code bug burns cost and hides the bug under noise."
    )
