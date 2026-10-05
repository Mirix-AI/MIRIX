"""Token estimation used for F9: text, fallback, event-loop safety."""

from __future__ import annotations

import asyncio
import math

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.infra import tokens


@pytest.fixture
def fresh_encoding():
    tokens.reset_encoding_for_tests()
    yield
    tokens.reset_encoding_for_tests()


def test_text_uses_o200k(fresh_encoding):
    import tiktoken

    text = "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸"
    assert tokens.count_text(text) == len(
        tiktoken.get_encoding("o200k_base").encode(text)
    )
    assert tokens.encoding_name() == protocol.TOKEN_ENCODING
    assert (
        tokens.count_text("<|endoftext|>") > 1
    )  # special tokens are counted as plain text, never rejected


def test_fallback_when_tokenizer_unavailable(fresh_encoding, monkeypatch):
    import tiktoken

    def broken(name: str):
        raise ConnectionError("offline first use")

    monkeypatch.setattr(tiktoken, "get_encoding", broken)
    assert tokens.count_text("abcdefg") == math.ceil(7 / 3)
    assert tokens.encoding_name() == tokens.FALLBACK_ENCODING
    assert asyncio.run(tokens.preload_encoding()) == tokens.FALLBACK_ENCODING


def test_chat_estimate_counts_every_input_part():
    base = {"model": "m", "messages": [{"role": "user", "content": "find the total"}]}
    small = tokens.estimate_chat_input(base).total
    with_tools = {
        **base,
        "tools": [
            {
                "type": "function",
                "function": {"name": "grep", "parameters": {"type": "object"}},
            }
        ],
    }
    assert tokens.estimate_chat_input(with_tools).total > small
    with_calls = {
        **base,
        "messages": [
            *base["messages"],
            {
                "role": "assistant",
                "content": None,
                "reasoning": "I should search.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "grep", "arguments": '{"p":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "line 3: total 12"},
        ],
    }
    assert tokens.estimate_chat_input(with_calls).total > small + 20
    parts = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "find the total"}]}
        ],
    }
    assert tokens.estimate_chat_input(parts).total == small
    with pytest.raises(ValueError):
        tokens.estimate_chat_input({"messages": "not a list"})


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "file", "file": {"filename": "a.pdf", "file_data": "data:,"}},
    ],
)
def test_image_and_file_parts_are_refused(part):
    """No environment sends them; their base64 text would wreck the F9 cap."""
    body = {"messages": [{"role": "user", "content": [part]}]}
    with pytest.raises(tokens.UnsupportedContentError):
        tokens.estimate_chat_input(body)


def test_embedding_estimate():
    assert tokens.estimate_embedding_input(
        {"input": ["alpha", "beta"]}
    ).total == tokens.count_text("alpha") + tokens.count_text("beta")
    assert tokens.estimate_embedding_input(
        {"input": "alpha"}
    ).total == tokens.count_text("alpha")


@pytest.mark.asyncio
async def test_large_payloads_are_counted_off_the_event_loop(monkeypatch):
    offloaded = []
    real_to_thread = asyncio.to_thread

    async def spy(func, *args, **kwargs):
        offloaded.append(func.__name__)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(tokens.asyncio, "to_thread", spy)
    small = {"messages": [{"role": "user", "content": "hi"}]}
    await tokens.estimate_chat_input_async(small, payload_chars=10)
    assert offloaded == []

    big_text = "lorem ipsum dolor sit amet " * 120_000  # ~3.2M chars
    big = {"messages": [{"role": "user", "content": big_text}]}
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.001)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    estimate = await tokens.estimate_chat_input_async(big, payload_chars=len(big_text))
    task.cancel()
    assert offloaded == ["estimate_chat_input"]
    assert estimate.total > 500_000
    assert ticks >= 3  # the loop kept running while the text was counted
