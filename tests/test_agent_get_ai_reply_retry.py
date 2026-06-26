"""Tests for the refactored Agent._get_ai_reply retry behavior.

Covers:
- LLMUnprocessableEntityError (422) propagates on the first attempt; no retry,
  no sleep.
- LLMRateLimitError (429) and other Transient errors retry up to
  settings.llm_inline_retry_max_attempts, then propagate.
- Empty-response ValueErrors classify as Transient and are retried.
- The removed parameters (empty_response_retry_limit, backoff_factor,
  max_delay, second_try) are no longer part of the function signature.

The tests avoid constructing a real Agent (which requires DB, llm_config,
tools, etc.) and instead drive _get_ai_reply via a SimpleNamespace stub
that satisfies the attributes the method touches.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirix.agent.agent import Agent
from mirix.errors import (
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)

# ---------- signature contract ----------


def test_get_ai_reply_no_longer_exposes_old_retry_params():
    """The 4 deprecated retry parameters must not be in the signature anymore.

    If a caller still passes them, that's a bug we want to surface loudly
    rather than silently absorb.
    """
    sig = inspect.signature(Agent._get_ai_reply)
    for removed in (
        "empty_response_retry_limit",
        "backoff_factor",
        "max_delay",
        "second_try",
    ):
        assert removed not in sig.parameters, f"{removed} must be removed from _get_ai_reply"


# ---------- stub Agent for direct loop testing ----------


def _build_stub_agent(send_side_effect):
    """Construct the minimum object surface that _get_ai_reply touches.

    Returns an object usable as `self` for `Agent._get_ai_reply.__func__(self, ...)`.
    send_side_effect drives what the mocked LLM client raises / returns.
    """
    llm_client = MagicMock()
    llm_client.send_llm_request = AsyncMock(side_effect=send_side_effect)

    # tool_rules_solver: a no-op solver — no tools forced, no failed-tool history
    tool_rules_solver = MagicMock()
    tool_rules_solver.get_allowed_tool_names = MagicMock(return_value=[])
    tool_rules_solver.tool_call_history = []
    tool_rules_solver.init_tool_rules = []

    agent_state = SimpleNamespace(
        name="test-agent",
        tools=[],
        llm_config=SimpleNamespace(),
        created_by_id="user-test",
    )

    stub = SimpleNamespace(
        agent_state=agent_state,
        tool_rules_solver=tool_rules_solver,
        last_function_response=None,
        supports_structured_output=True,
        logger=MagicMock(),
        interface=MagicMock(),
    )
    return stub, llm_client


# ---------- the 422 fix ----------


@pytest.mark.asyncio
async def test_get_ai_reply_422_propagates_on_first_attempt(monkeypatch):
    """LLMUnprocessableEntityError must raise immediately — no retry, no sleep."""
    # Sleep should never be called on the Permanent path.
    sleep_mock = AsyncMock()
    monkeypatch.setattr("mirix.agent.agent.asyncio.sleep", sleep_mock)

    stub, llm_client = _build_stub_agent(send_side_effect=LLMUnprocessableEntityError("content rejected"))

    with pytest.raises(LLMUnprocessableEntityError):
        await Agent._get_ai_reply(
            stub,
            message_sequence=[],
            llm_client=llm_client,
        )

    assert (
        llm_client.send_llm_request.await_count == 1
    ), "_get_ai_reply must call the LLM exactly once on a Permanent error"
    sleep_mock.assert_not_called()


# ---------- transient retry budget ----------


@pytest.mark.asyncio
async def test_get_ai_reply_transient_retries_to_budget_then_propagates(monkeypatch):
    """LLMRateLimitError should retry max_attempts times then re-raise.

    After VEPAGE-1251 S3 the propagated exception is tagged
    `__mirix_inner_exhausted__` so process_with_policy does NOT add another
    whole-step cycle on top (kills the 3 x 3 = 9 multiplication)."""
    monkeypatch.setattr("mirix.agent.agent.asyncio.sleep", AsyncMock())
    from mirix.queue.error_policy import is_inner_exhausted
    from mirix.settings import settings

    expected_total = settings.llm_inline_retry_max_attempts + 1  # initial + retries

    stub, llm_client = _build_stub_agent(send_side_effect=LLMRateLimitError("429 still"))

    with pytest.raises(LLMRateLimitError) as excinfo:
        await Agent._get_ai_reply(
            stub,
            message_sequence=[],
            llm_client=llm_client,
        )

    assert llm_client.send_llm_request.await_count == expected_total
    assert is_inner_exhausted(excinfo.value) is True, (
        "exhausted LLM transient must be tagged so the whole-step policy "
        "doesn't trigger another retry cycle (kills the 3x3 multiplication)"
    )


@pytest.mark.asyncio
async def test_get_ai_reply_transient_then_success(monkeypatch):
    """A 5xx that clears on retry yields the eventual successful response."""
    monkeypatch.setattr("mirix.agent.agent.asyncio.sleep", AsyncMock())

    # First call raises, second returns a valid response object.
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=[]),
                finish_reason="stop",
            )
        ]
    )
    calls = {"n": 0}

    async def side_effect(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMServerError("503 try again")
        return success_response

    stub, llm_client = _build_stub_agent(send_side_effect=side_effect)

    out = await Agent._get_ai_reply(
        stub,
        message_sequence=[],
        llm_client=llm_client,
    )

    assert out is success_response
    assert calls["n"] == 2


# ---------- empty-response validation classifies as Transient ----------


@pytest.mark.asyncio
async def test_get_ai_reply_empty_response_retries(monkeypatch):
    """An empty-choices response raises LLMBadResponseShapeError → Transient → retry.

    Guards the Gemini-quirk path: providers occasionally return empty content,
    and this is treated as a transient provider quirk that retrying may fix.
    """
    monkeypatch.setattr("mirix.agent.agent.asyncio.sleep", AsyncMock())

    empty_response = SimpleNamespace(choices=[])
    success_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="recovered", tool_calls=[]),
                finish_reason="stop",
            )
        ]
    )
    responses = [empty_response, success_response]

    async def side_effect(**_kwargs):
        return responses.pop(0)

    stub, llm_client = _build_stub_agent(send_side_effect=side_effect)

    out = await Agent._get_ai_reply(
        stub,
        message_sequence=[],
        llm_client=llm_client,
    )

    assert out is success_response
    assert llm_client.send_llm_request.await_count == 2


@pytest.mark.asyncio
async def test_get_ai_reply_bad_finish_reason_retries(monkeypatch):
    """A finish_reason that isn't stop/function_call/tool_calls raises
    LLMBadResponseShapeError, which classifies TRANSIENT and gets retried."""
    from mirix.errors import LLMBadResponseShapeError

    monkeypatch.setattr("mirix.agent.agent.asyncio.sleep", AsyncMock())

    bad = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="x", tool_calls=[]),
                finish_reason="content_filter",
            )
        ]
    )

    stub, llm_client = _build_stub_agent(send_side_effect=lambda **_: bad)
    # Convert the lambda to a coroutine return path: use AsyncMock side_effect
    llm_client.send_llm_request = AsyncMock(return_value=bad)

    with pytest.raises(LLMBadResponseShapeError):
        await Agent._get_ai_reply(
            stub,
            message_sequence=[],
            llm_client=llm_client,
        )
    # Initial + retries — bad finish_reason persists, budget exhausts.
    from mirix.settings import settings

    expected_total = settings.llm_inline_retry_max_attempts + 1
    assert llm_client.send_llm_request.await_count == expected_total
