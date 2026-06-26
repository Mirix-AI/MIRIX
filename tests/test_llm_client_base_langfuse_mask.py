"""LLMClientBase pre-masks PII before it becomes a LangFuse
span attribute.

The value handed to ``start_as_current_observation(input=...)`` and to
``generation.update(output=...)`` must already be redacted (so the SDK's
``mask=`` callback is a cheap no-op). The static ``tools`` schemas carry no
user PII and must NOT be masked (wasted ispy-pii load on the path we're
relieving).
"""

import datetime
from contextlib import contextmanager
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
)
from mirix.schemas.openai.chat_completion_response import Message as ChoiceMessage
from mirix.schemas.openai.chat_completion_response import (
    UsageStatistics,
)

pytestmark = pytest.mark.asyncio


async def _fake_mask_structure(data):
    """Stand-in masker: uppercases every leaf string so a test can assert
    that a value passed through masking. Mirrors mask_structure's shape walk."""
    if isinstance(data, str):
        return data.upper()
    if isinstance(data, dict):
        return {k: await _fake_mask_structure(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([await _fake_mask_structure(i) for i in data])
    return data


class _FakeMessage:
    """Minimal stand-in for mirix.schemas.message.Message for tracing."""

    def __init__(self, role: str, text: str):
        self.role = role
        self.content = [type("Part", (), {"text": text})()]
        self.tool_calls = None


class _StubClient(LLMClientBase):
    """Concrete LLMClientBase with the abstract LLM I/O stubbed out so we
    can exercise _execute_with_langfuse without a real provider."""

    def __init__(self, llm_config, response: ChatCompletionResponse):
        super().__init__(llm_config)
        self._response = response

    async def build_request_data(self, *a, **k) -> dict:
        return {"req": "data"}

    async def request(self, request_data: dict) -> dict:
        return {"raw": "response"}

    def convert_response_to_chat_completion(self, response_data, input_messages):
        return self._response

    async def handle_llm_error(self, e: Exception) -> Exception:
        return e


def _capturing_langfuse(captured: dict):
    """Build a fake Langfuse client whose start_as_current_observation
    records the input= kwarg and whose generation records output=."""
    generation = MagicMock()

    def _update(**kwargs):
        if "output" in kwargs:
            captured["output"] = kwargs["output"]

    generation.update.side_effect = _update

    @contextmanager
    def _cm():
        yield generation

    langfuse = MagicMock()

    def _start(**kwargs):
        captured["input"] = kwargs.get("input")
        return _cm()

    langfuse.start_as_current_observation.side_effect = _start
    return langfuse


def _make_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="cc-1",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChoiceMessage(role="assistant", content="reply to bob@acme.com"),
            )
        ],
        created=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        model="gpt-4o-mini",
        usage=UsageStatistics(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


async def _run(captured: dict, tools: Optional[List[dict]]):
    cfg = LLMConfig.default_config("gpt-4o-mini")
    client = _StubClient(cfg, _make_response())
    messages = [_FakeMessage("user", "my ssn is 123-45-6789")]
    langfuse = _capturing_langfuse(captured)
    with (
        patch("mirix.llm_api.llm_client_base.get_langfuse_client", return_value=langfuse),
        patch(
            "mirix.llm_api.llm_client_base.get_trace_context",
            return_value={"trace_id": "t-1", "observation_id": "o-1"},
        ),
        patch("mirix.llm_api.llm_client_base.mark_observation_as_child"),
        patch(
            "mirix.llm_api.llm_client_base.mask_structure",
            side_effect=_fake_mask_structure,
        ),
    ):
        await client.send_llm_request(messages, tools=tools)


async def test_span_input_messages_are_masked():
    captured: dict = {}
    await _run(captured, tools=None)
    # The user message content reached the span already masked (uppercased).
    msg = captured["input"]["messages"][0]
    assert msg["content"] == "MY SSN IS 123-45-6789"


async def test_span_output_is_masked():
    captured: dict = {}
    await _run(captured, tools=None)
    assert captured["output"]["content"] == "REPLY TO BOB@ACME.COM"


async def test_tools_are_not_masked():
    captured: dict = {}
    tools = [{"type": "function", "function": {"name": "search", "description": "find stuff"}}]
    await _run(captured, tools=tools)
    # tools passed through verbatim — NOT uppercased by the fake masker.
    assert captured["input"]["tools"] == tools
