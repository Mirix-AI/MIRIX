"""Span-tree correctness: work executed inside a tool span must parent to that
tool span, not to the agent-level span above it.

Spans in this codebase are parented via the ``current_observation_id`` ContextVar
(``mirix/observability/context.py``); each span-creation site reads its parent from
``get_trace_context()["observation_id"]``. ``execute_tool_and_persist_state`` opens the
``tool: <name>`` span but must also publish that span's id into the ContextVar while the
tool body runs, so any child observation opened during tool execution (e.g.
``Resolve Child Agents`` or the memory sub-agent spans spawned by
``trigger_memory_update``) nests under the tool span. After the tool returns, the prior
observation id must be restored so the next sibling tool span (e.g.
``tool: finish_memory_update``) still parents to the agent span.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mirix.agent.agent import Agent
from mirix.observability import context as obs_context
from mirix.schemas.enums import ToolType


def _make_fake_self():
    """Minimal stand-in for an Agent instance for execute_tool_and_persist_state."""
    fake_self = MagicMock()
    fake_self.agent_state = SimpleNamespace(name="meta_memory_agent", id="meta-1")
    fake_self.user = SimpleNamespace(timezone="UTC")
    fake_self._block_scopes = None

    async def _get_blocks(*args, **kwargs):
        return []

    fake_self.block_manager.get_blocks = _get_blocks
    return fake_self


def _make_langfuse_with_span(span_id):
    """Fake Langfuse whose tool span exposes ``.id == span_id`` and acts as a
    context manager (mirrors the real start_as_current_observation usage)."""
    span = MagicMock()
    span.id = span_id
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    langfuse = MagicMock()
    langfuse.start_as_current_observation.return_value = cm
    return langfuse, span


@pytest.mark.asyncio
async def test_tool_body_runs_under_tool_span_as_parent():
    """While the tool body executes, the current observation id is the tool span's
    id (so child spans opened during execution parent to the tool span)."""
    fake_self = _make_fake_self()
    tool = SimpleNamespace(tool_type=ToolType.MIRIX_EXTRA)

    captured = {}

    async def probe(self):  # the tool callable; records the parent a child would see
        captured["observation_id"] = obs_context.get_trace_context().get("observation_id")
        return "ok"

    obs_context.set_trace_context(trace_id="trace-1", observation_id="meta-agent-obs")
    langfuse, _span = _make_langfuse_with_span("tool-span-obs")

    with (
        patch("mirix.agent.agent.get_langfuse_client", return_value=langfuse),
        patch("mirix.agent.agent.get_function_from_module", return_value=probe),
        patch("mirix.agent.agent.mark_observation_as_child"),
    ):
        await Agent.execute_tool_and_persist_state(
            fake_self,
            function_name="trigger_memory_update",
            function_args={},
            target_mirix_tool=tool,
        )

    assert captured["observation_id"] == "tool-span-obs"


@pytest.mark.asyncio
async def test_observation_id_restored_after_tool_returns():
    """After the tool span closes, the prior observation id is restored so the next
    sibling tool span still parents to the agent-level span."""
    fake_self = _make_fake_self()
    tool = SimpleNamespace(tool_type=ToolType.MIRIX_EXTRA)

    async def probe(self):
        return "ok"

    obs_context.set_trace_context(trace_id="trace-1", observation_id="meta-agent-obs")
    langfuse, _span = _make_langfuse_with_span("tool-span-obs")

    with (
        patch("mirix.agent.agent.get_langfuse_client", return_value=langfuse),
        patch("mirix.agent.agent.get_function_from_module", return_value=probe),
        patch("mirix.agent.agent.mark_observation_as_child"),
    ):
        await Agent.execute_tool_and_persist_state(
            fake_self,
            function_name="trigger_memory_update",
            function_args={},
            target_mirix_tool=tool,
        )

    assert obs_context.get_trace_context().get("observation_id") == "meta-agent-obs"


@pytest.mark.asyncio
async def test_observation_id_restored_to_none_when_tool_span_had_no_parent():
    """When the tool span has no parent observation (it sits directly under the
    trace root), the observation id must be cleared back to None afterward — not
    left pointing at the closed tool span — so later siblings don't nest under it."""
    fake_self = _make_fake_self()
    tool = SimpleNamespace(tool_type=ToolType.MIRIX_EXTRA)

    async def probe(self):
        return "ok"

    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1")  # trace_id but no observation_id
    langfuse, _span = _make_langfuse_with_span("tool-span-obs")

    with (
        patch("mirix.agent.agent.get_langfuse_client", return_value=langfuse),
        patch("mirix.agent.agent.get_function_from_module", return_value=probe),
        patch("mirix.agent.agent.mark_observation_as_child"),
    ):
        await Agent.execute_tool_and_persist_state(
            fake_self,
            function_name="trigger_memory_update",
            function_args={},
            target_mirix_tool=tool,
        )

    assert obs_context.get_trace_context().get("observation_id") is None
