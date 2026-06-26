"""embeddings tracing pre-masks the text input before it
becomes a LangFuse span attribute.

``traced_embedding_with_retry`` sends the embedded text as the span
``input``; it must be redacted upstream. The output dict is
``{"embedding_dim": N}`` (no PII) and must NOT be masked.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix import embeddings

pytestmark = pytest.mark.asyncio


async def _fake_mask_structure(data):
    if isinstance(data, str):
        return data.upper()
    if isinstance(data, dict):
        return {k: await _fake_mask_structure(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([await _fake_mask_structure(i) for i in data])
    return data


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


async def _run(captured: dict):
    langfuse = _capturing_langfuse(captured)
    with (
        patch("mirix.embeddings.get_langfuse_client", return_value=langfuse),
        patch(
            "mirix.embeddings.get_trace_context",
            return_value={"trace_id": "t-1", "observation_id": "o-1"},
        ),
        patch("mirix.embeddings.mark_observation_as_child"),
        patch("mirix.embeddings.mask_structure", side_effect=_fake_mask_structure),
    ):
        await embeddings.traced_embedding_with_retry(
            model="m",
            provider="openai",
            text="my ssn is 123-45-6789",
            embedding_func=AsyncMock(return_value=[0.1, 0.2, 0.3]),
        )


async def test_embedding_span_input_text_is_masked():
    captured: dict = {}
    await _run(captured)
    assert captured["input"]["text"] == "MY SSN IS 123-45-6789"


async def test_embedding_output_dim_is_not_masked():
    captured: dict = {}
    await _run(captured)
    # Output stays the structural {"embedding_dim": N}; the int is untouched.
    assert captured["output"] == {"embedding_dim": 3}
