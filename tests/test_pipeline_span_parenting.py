"""Span-tree correctness for the agent-loop pipeline spans (VEPAGE-1245).

Spans in this codebase resolve their parent from the ``current_observation_id``
ContextVar (``mirix/observability/context.py``); each span-creation site reads
``get_trace_context()["observation_id"]`` as its parent. The agent loop adds two
pipeline spans:

- ``Agent Step`` wraps the whole ``Agent.step`` invocation (pre-loop setup plus
  every loop iteration). It sits under the worker's ``Meta Agent`` span and every
  pre-loop ``timed_span`` (Persist Memory Source, Check Source Processing State,
  Load Retained History) must nest under it.
- ``Inner Step`` wraps each ``await self.inner_step(...)`` call. The NAME is the
  stable literal "Inner Step" for EVERY iteration (low cardinality); the iteration
  index lives in metadata ``step_count`` so Langfuse can aggregate across traces.

These tests assert the parenting/naming contract at the ``timed_span`` level
(matching ``test_timed_span_parenting.py`` / ``test_tool_span_parenting.py``),
without standing up real tracing or DB infra.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability import context as obs_context
from mirix.observability.timed import timedspan as timed_span


def _make_langfuse():
    """Fake Langfuse whose spans expose ``.id`` matching the requested name, and
    act as sync context managers (mirrors real start_as_current_observation)."""

    def _start(name, **kwargs):
        span = MagicMock()
        # Give each span a deterministic id derived from name + call count so
        # nested-parent assertions can identify which span is the parent.
        span.id = f"{name}-obs-{_start.calls}"
        _start.calls += 1
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=span)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    _start.calls = 0
    langfuse = MagicMock()
    langfuse.start_as_current_observation.side_effect = _start
    return langfuse


@pytest.mark.asyncio
async def test_pre_loop_span_parents_under_agent_step():
    """A pre-loop timed_span (e.g. Persist Memory Source) opened inside the
    Agent Step block parents to Agent Step, not to the Meta Agent span above."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="meta-agent-obs")
    langfuse = _make_langfuse()

    captured = {}
    with (
        patch("mirix.observability.timed.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed.mark_observation_as_child"),
    ):
        async with timed_span("Agent Step", metadata={"agent_type": "meta_memory_agent"}):
            agent_step_id = obs_context.get_trace_context().get("observation_id")
            async with timed_span("Persist Memory Source", metadata={"memory_source_id": "ms-1"}):
                pass
            # The inner span's parent is the id published by Agent Step.
            inner_call = langfuse.start_as_current_observation.call_args_list[1]
            captured["inner_parent"] = inner_call.kwargs["trace_context"].get("parent_span_id")

    assert captured["inner_parent"] == agent_step_id


@pytest.mark.asyncio
async def test_agent_step_parents_under_meta_agent():
    """Agent Step itself parents to the Meta Agent span published by the worker."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="meta-agent-obs")
    langfuse = _make_langfuse()

    with (
        patch("mirix.observability.timed.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed.mark_observation_as_child"),
    ):
        async with timed_span("Agent Step", metadata={"agent_type": "meta_memory_agent"}):
            pass

    agent_step_call = langfuse.start_as_current_observation.call_args_list[0]
    assert agent_step_call.kwargs["trace_context"].get("parent_span_id") == "meta-agent-obs"
    assert agent_step_call.kwargs["name"] == "Agent Step"


@pytest.mark.asyncio
async def test_inner_step_uses_stable_name_with_step_count_metadata():
    """Each loop iteration's Inner Step span uses the SAME literal name and carries
    the iteration index in metadata['step_count'] (low-cardinality contract)."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="agent-step-obs")
    langfuse = _make_langfuse()

    with (
        patch("mirix.observability.timed.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed.mark_observation_as_child"),
    ):
        for step_count in range(2):
            async with timed_span("Inner Step", metadata={"step_count": step_count}):
                pass

    calls = langfuse.start_as_current_observation.call_args_list
    assert [c.kwargs["name"] for c in calls] == ["Inner Step", "Inner Step"]
    assert [c.kwargs["metadata"]["step_count"] for c in calls] == [0, 1]


@pytest.mark.asyncio
async def test_inner_step_iterations_are_siblings_under_agent_step():
    """Consecutive Inner Step spans each parent back to Agent Step — the prior
    observation id is restored between iterations so they are siblings, not nested."""
    obs_context.clear_trace_context()
    obs_context.set_trace_context(trace_id="trace-1", observation_id="agent-step-obs")
    langfuse = _make_langfuse()

    with (
        patch("mirix.observability.timed.get_langfuse_client", return_value=langfuse),
        patch("mirix.observability.timed.mark_observation_as_child"),
    ):
        for step_count in range(2):
            async with timed_span("Inner Step", metadata={"step_count": step_count}):
                pass
            # After each iteration the context restores to Agent Step.
            assert obs_context.get_trace_context().get("observation_id") == "agent-step-obs"

    parents = [
        c.kwargs["trace_context"].get("parent_span_id") for c in langfuse.start_as_current_observation.call_args_list
    ]
    assert parents == ["agent-step-obs", "agent-step-obs"]
