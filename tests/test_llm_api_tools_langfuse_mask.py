"""llm_api_tools.create pre-masks PII before it becomes a
LangFuse span attribute.

The value handed to ``start_as_current_observation(input=...)`` and the
``output`` passed to ``generation.update(output=...)`` must already be
redacted. Static ``tools``/``functions`` schemas carry no PII and must NOT
be masked.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.llm_api import llm_api_tools
from mirix.schemas.llm_config import LLMConfig

pytestmark = pytest.mark.asyncio


async def _fake_mask_structure(data):
    """Uppercases leaf strings so a test can assert a value was masked."""
    if isinstance(data, str):
        return data.upper()
    if isinstance(data, dict):
        return {k: await _fake_mask_structure(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([await _fake_mask_structure(i) for i in data])
    return data


class _FakeRole:
    value = "user"


class _FakeMessage:
    def __init__(self, text: str):
        self.role = _FakeRole()
        self.text = text

    def to_openai_dict(self):
        return {"role": "user", "content": self.text}


def _capturing_langfuse(captured: dict):
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


def _openai_response():
    """Minimal object graph that the OpenAI branch's output builder reads."""
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = "reply to bob@acme.com"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return response


async def _run(captured: dict, functions):
    cfg = LLMConfig.default_config("gpt-4o-mini")
    cfg.model_endpoint_type = "openai"
    cfg.model_endpoint = "https://api.openai.com/v1"
    messages = [_FakeMessage("my ssn is 123-45-6789")]
    langfuse = _capturing_langfuse(captured)

    provider = MagicMock()
    provider.get_openai_override_key = AsyncMock(return_value="sk-test")

    with (
        patch("mirix.llm_api.llm_api_tools.get_langfuse_client", return_value=langfuse),
        patch(
            "mirix.llm_api.llm_api_tools.get_trace_context",
            return_value={"trace_id": "t-1", "observation_id": "o-1"},
        ),
        patch("mirix.llm_api.llm_api_tools.mark_observation_as_child"),
        patch(
            "mirix.llm_api.llm_api_tools.mask_structure",
            side_effect=_fake_mask_structure,
        ),
        patch("mirix.llm_api.llm_api_tools.num_tokens_from_messages", return_value=1),
        patch("mirix.llm_api.llm_api_tools.num_tokens_from_functions", return_value=1),
        patch("mirix.services.provider_manager.ProviderManager", return_value=provider),
        patch("mirix.llm_api.llm_api_tools.build_openai_chat_completions_request", return_value={}),
        patch(
            "mirix.llm_api.llm_api_tools.openai_chat_completions_request",
            new=AsyncMock(return_value=_openai_response()),
        ),
    ):
        await llm_api_tools.create(cfg, messages, functions=functions)


async def test_span_input_messages_are_masked():
    captured: dict = {}
    await _run(captured, functions=None)
    assert captured["input"]["messages"][0]["content"] == "MY SSN IS 123-45-6789"


async def test_span_output_is_masked():
    captured: dict = {}
    await _run(captured, functions=None)
    assert captured["output"]["content"] == "REPLY TO BOB@ACME.COM"


async def test_tools_are_not_masked():
    captured: dict = {}
    functions = [{"name": "search", "description": "find stuff"}]
    await _run(captured, functions=functions)
    assert captured["input"]["tools"] == functions
