"""Input-token estimation for the F9 output cap (applied by the gateway, H3).

* Text: ``o200k_base`` for every model (an approximation for a model with another
  tokenizer).
* ``image_url`` and ``file`` parts (PDF ...) are refused
  (:class:`UnsupportedContentError`): no environment sends them, and counting their
  base64 text would wreck the F9 cap.
* The encoding is loaded once (it may download on first use); call
  :func:`preload_encoding` at startup. If it cannot load, counting falls back
  to ``ceil(chars / 3)`` and :func:`encoding_name` reports the fallback.
* The counters are pure CPU functions. Async callers run large payloads in a
  worker thread (:func:`estimate_chat_input_async`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from evals.skill_bench.protocol import (
    LARGE_PAYLOAD_CHARS,
    TOKEN_ENCODING,
    TOKEN_ESTIMATE,
)

log = logging.getLogger(__name__)

# The numbers are protocol constants (protocol.TOKEN_ESTIMATE, fingerprinted): they decide
# the max_tokens the gateway writes into every call.
FALLBACK_ENCODING = "fallback_chars_div_3"
FALLBACK_CHARS_PER_TOKEN = TOKEN_ESTIMATE["fallback_chars_per_token"]
MESSAGE_OVERHEAD_TOKENS = TOKEN_ESTIMATE["message_overhead_tokens"]
REPLY_PRIMING_TOKENS = TOKEN_ESTIMATE["reply_priming_tokens"]
TOOL_CALL_OVERHEAD_TOKENS = TOKEN_ESTIMATE["tool_call_overhead_tokens"]

#: Content-part types the estimate refuses: an image or file (PDF ...) part would be
#: counted as its base64 text, which overstates it by orders of magnitude and starves the
#: output cap.
UNSUPPORTED_PART_TYPES = frozenset({"file", "image_url"})


class UnsupportedContentError(ValueError):
    """A content part the F9 estimate cannot count honestly (the gateway refuses the call)."""


_lock = threading.Lock()
_encoding: Any = None
_encoding_state = "unloaded"  # unloaded | loaded | fallback


def load_encoding() -> Any | None:
    """Load the tokenizer once; return None (and remember it) if it is unavailable."""
    global _encoding, _encoding_state
    with _lock:
        if _encoding_state == "unloaded":
            try:
                import tiktoken

                _encoding = tiktoken.get_encoding(TOKEN_ENCODING)
                _encoding_state = "loaded"
            except (
                Exception
            ) as exc:  # offline first use, missing package, corrupt cache
                log.warning(
                    "tiktoken %s unavailable (%s); using chars/3 estimate",
                    TOKEN_ENCODING,
                    exc,
                )
                _encoding_state = "fallback"
        return _encoding


async def preload_encoding() -> str:
    """Load the tokenizer in a worker thread (the first load may block on a download)."""
    await asyncio.to_thread(load_encoding)
    return encoding_name()


def encoding_name() -> str:
    return TOKEN_ENCODING if _encoding_state == "loaded" else FALLBACK_ENCODING


def reset_encoding_for_tests() -> None:
    global _encoding, _encoding_state
    with _lock:
        _encoding, _encoding_state = None, "unloaded"


def count_text(text: str) -> int:
    if not text:
        return 0
    enc = load_encoding()
    if enc is None:
        return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class TokenEstimate:
    text_tokens: int
    encoding: str

    @property
    def total(self) -> int:
        return self.text_tokens

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text_tokens, "encoding": self.encoding}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class _Counter:
    def __init__(self) -> None:
        self.text = 0

    def add_text(self, value: Any) -> None:
        if value is None:
            return
        self.text += count_text(value if isinstance(value, str) else _json_text(value))

    def add_content(self, content: Any) -> None:
        if content is None or isinstance(content, str):
            self.add_text(content)
            return
        if not isinstance(content, list):
            self.add_text(content)
            return
        for part in content:
            if not isinstance(part, Mapping):
                self.add_text(part)
                continue
            kind = part.get("type")
            if kind == "text":
                self.add_text(part.get("text"))
            elif kind in UNSUPPORTED_PART_TYPES:
                raise UnsupportedContentError(
                    f"content parts of type {kind!r} are not supported: the F9 input "
                    "estimate cannot count them (send text parts)"
                )
            else:  # input_audio, ...: count their serialized form (conservative)
                self.add_text(part)

    def add_message(self, message: Mapping[str, Any]) -> None:
        self.text += MESSAGE_OVERHEAD_TOKENS
        self.add_text(message.get("role"))
        self.add_text(message.get("name"))
        self.add_content(message.get("content"))
        for key in ("reasoning", "refusal", "tool_call_id"):
            self.add_text(message.get(key))
        if message.get("reasoning_details"):
            self.add_text(message["reasoning_details"])
        for call in message.get("tool_calls") or []:
            self.text += TOOL_CALL_OVERHEAD_TOKENS
            function = call.get("function") if isinstance(call, Mapping) else None
            if isinstance(function, Mapping):
                self.add_text(function.get("name"))
                self.add_text(function.get("arguments"))
            else:
                self.add_text(call)

    def result(self) -> TokenEstimate:
        return TokenEstimate(self.text, encoding_name())


def estimate_chat_input(body: Mapping[str, Any]) -> TokenEstimate:
    """Estimated input tokens of a chat-completions request: messages, tools and response format."""
    counter = _Counter()
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be an object")
        counter.add_message(message)
    counter.text += REPLY_PRIMING_TOKENS
    for key in ("tools", "functions", "response_format"):
        if body.get(key):
            counter.add_text(body[key])
    return counter.result()


def estimate_embedding_input(body: Mapping[str, Any]) -> TokenEstimate:
    counter = _Counter()
    value = body.get("input")
    items: Iterable[Any] = value if isinstance(value, list) else [value]
    for item in items:
        counter.add_content(item) if isinstance(item, list) else counter.add_text(item)
    return counter.result()


async def estimate_chat_input_async(
    body: Mapping[str, Any], payload_chars: int
) -> TokenEstimate:
    """:func:`estimate_chat_input`, in a worker thread when the payload is large."""
    if payload_chars > LARGE_PAYLOAD_CHARS:
        return await asyncio.to_thread(estimate_chat_input, body)
    return estimate_chat_input(body)
