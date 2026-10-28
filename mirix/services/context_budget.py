"""Unified context budget.

Every large model input in MIRIX is sized against ONE rule: a call's input
plus its output must fit in 80% of the model's context window, and its output
may use at most 80% of the model's own output limit. This module holds the
pieces every caller shares:

* :class:`TokenCounter` — approximate token counts with ``o200k_base`` for
  EVERY model (``mirix.utils.count_tokens`` raises KeyError for non-OpenAI
  model names). It counts text, message content-part lists (OpenAI, Anthropic,
  Google and conversation-store ``content_parts`` shapes), tool calls, tool
  schemas and image references.
* :func:`image_tokens` — the OpenAI high-detail image formula (fit in
  2048x2048, shortest side to 768, 512x512 tiles, ``85 + 170 * tiles``) times
  :data:`IMAGE_TOKEN_MULTIPLIER` (7.5), rounded up. Other providers are
  estimated with the same rule; it is an estimate, not their billing rule.
  The factor is a conservative margin: a provider may bill an image at
  several times the formula, and an underestimate would let a request
  overflow the context window. The harness gateway uses the same value
  (``evals/skill_bench/protocol.py`` ``IMAGE_TOKEN_MULTIPLIER``, whose tests
  assert the two agree, and whose settings probe refuses to start a MIRIX
  cell when they differ). An image
  whose size is unknown (a remote URL, an undecodable header) counts as the
  largest estimate (10,838).
* :class:`ModelLimits` — ``context_window`` and ``max_output_tokens`` of an
  ``LLMConfig`` (falling back to ``max_tokens``), with the three formulas::

      input_budget(system, reserve) = floor(window * 0.8) - system - reserve
      output_cap(input)             = min(floor(max_out * 0.8),
                                          floor(window * 0.8) - input)
      planning_output_reserve       = min(floor(max_out * 0.8),
                                          floor(window * 0.1))

  All arithmetic is integer (``x * 4 // 5`` is exactly ``floor(x * 0.8)``).
* :func:`output_cap_for_request` — what the LLM clients call: the
  ``max_tokens`` for one request from its estimated input, or
  :class:`~mirix.errors.ContextBudgetExceeded` (below 1,024) BEFORE sending.

Async rules (the counting itself is pure CPU, one of the allowed sync helpers):

* tiktoken downloads its vocabulary synchronously on first use, so the FastAPI
  lifespan preloads it with :func:`preload_tokenizer` (``asyncio.to_thread``),
  caching the file under the MIRIX directory (``TIKTOKEN_CACHE_DIR`` =
  ``<MIRIX_DIR>/tiktoken`` unless the operator set it). Counting never loads
  the vocabulary on the event loop: until it is loaded — or if loading failed
  (offline) — counts fall back to 3 characters per token, with one warning.
  A failed (or never attempted) load is retried at most every 5 minutes: in
  the background by the async counting entry points
  (:func:`ensure_tokenizer_soon`), and awaited by a learning run before it
  plans chunks and batches (:func:`ensure_tokenizer`).
* :meth:`TokenCounter.acount_text` / :meth:`TokenCounter.acount_request` move
  inputs larger than 50,000 characters to a worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from mirix.errors import ContextBudgetExceeded
from mirix.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "BUDGET_NUMERATOR",
    "ContextBudgetExceeded",
    "ENCODING_NAME",
    "FALLBACK_CHARS_PER_TOKEN",
    "IMAGE_TOKEN_MULTIPLIER",
    "LARGE_TEXT_CHARS",
    "MIN_OUTPUT_TOKENS",
    "ModelLimits",
    "TokenCounter",
    "UNKNOWN_IMAGE_TOKENS",
    "ensure_tokenizer",
    "ensure_tokenizer_soon",
    "get_token_counter",
    "image_dimensions",
    "image_formula_tokens",
    "image_tokens",
    "output_cap_for_request",
    "preload_tokenizer",
    "tiktoken_cache_dir",
]

ENCODING_NAME = "o200k_base"
# Texts longer than this are counted in a worker thread (async rule).
LARGE_TEXT_CHARS = 50_000
# Estimate used while the vocabulary is unavailable (offline, not loaded yet).
FALLBACK_CHARS_PER_TOKEN = 3
# Below this many output tokens a request is refused before sending.
MIN_OUTPUT_TOKENS = 1024
# Input + output <= window * 4/5; output <= max_output * 4/5.
BUDGET_NUMERATOR, BUDGET_DENOMINATOR = 4, 5
# Planning reserve: at most a tenth of the window.
PLANNING_WINDOW_DIVISOR = 10
# Chat formatting overhead per message (role markers, separators).
MESSAGE_OVERHEAD_TOKENS = 4
# A failed vocabulary load is retried at most this often (seconds).
TOKENIZER_RETRY_SECONDS = 300.0
# How long server startup waits for the vocabulary before falling back.
TOKENIZER_PRELOAD_TIMEOUT_SECONDS = 30.0

# OpenAI high-detail image formula.
IMAGE_BASE_TOKENS = 85
IMAGE_TILE_TOKENS = 170
IMAGE_TILE_SIZE = 512
IMAGE_FIT_SIZE = 2048
IMAGE_SHORT_SIDE = 768
# The estimate is the formula times this factor, rounded up: a conservative margin,
# since a provider may bill image input at several times the formula.
# MUST equal evals/skill_bench/protocol.py IMAGE_TOKEN_MULTIPLIER (the harness
# gateway's estimate): its tests (evals/skill_bench/tests/test_tokens.py) assert it,
# and the harness's settings probe (infra/mirix_server.check_llm_contract) refuses a
# MIRIX cell whose value differs.
IMAGE_TOKEN_MULTIPLIER = Fraction(15, 2)
# 2048 x 768 after scaling -> 4 x 2 tiles: the formula's maximum.
_MAX_FORMULA_TOKENS = IMAGE_BASE_TOKENS + IMAGE_TILE_TOKENS * 8


def image_formula_tokens(width: Any, height: Any) -> int:
    """The OpenAI high-detail formula alone for a ``width`` x ``height`` image.

    1. scale down to fit within 2048x2048 (aspect kept);
    2. scale down so the shortest side is 768 (never scaled up);
    3. ``85 + 170 * ceil(w / 512) * ceil(h / 512)``.

    Unknown or invalid sizes give the formula's maximum (1,445).
    """
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return _MAX_FORMULA_TOKENS
    if not (w > 0 and h > 0) or math.isinf(w) or math.isinf(h):
        return _MAX_FORMULA_TOKENS
    longest = max(w, h)
    if longest > IMAGE_FIT_SIZE:
        scale = IMAGE_FIT_SIZE / longest
        w, h = w * scale, h * scale
    shortest = min(w, h)
    if shortest > IMAGE_SHORT_SIDE:
        scale = IMAGE_SHORT_SIDE / shortest
        w, h = w * scale, h * scale
    tiles = math.ceil(w / IMAGE_TILE_SIZE - 1e-9) * math.ceil(
        h / IMAGE_TILE_SIZE - 1e-9
    )
    return IMAGE_BASE_TOKENS + IMAGE_TILE_TOKENS * max(1, tiles)


def image_tokens(width: Any, height: Any) -> int:
    """Token estimate for a ``width`` x ``height`` image: the high-detail formula
    (:func:`image_formula_tokens`) times :data:`IMAGE_TOKEN_MULTIPLIER`, rounded up.

    Unknown or invalid sizes give :data:`UNKNOWN_IMAGE_TOKENS`, the largest
    estimate, so an estimate never undercounts an image.
    """
    return math.ceil(image_formula_tokens(width, height) * IMAGE_TOKEN_MULTIPLIER)


# No image can cost more (the formula's maximum times the factor).
UNKNOWN_IMAGE_TOKENS = math.ceil(_MAX_FORMULA_TOKENS * IMAGE_TOKEN_MULTIPLIER)


# --------------------------------------------------------------------------- #
# Image dimensions from the first bytes of an encoded image                    #
# --------------------------------------------------------------------------- #

# Enough base64 to cover the header (and typical EXIF) of an encoded image.
_IMAGE_HEADER_B64_CHARS = 128 * 1024
_JPEG_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}  # fmt: skip


def image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    """``(width, height)`` from the header of a PNG/GIF/JPEG/WEBP/BMP image.

    Reads only headers (no decoding); ``None`` when the format is unknown or
    the header is incomplete.
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return int(width), int(height)
        if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return int(width), int(height)
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp_dimensions(data)
        if data[:2] == b"\xff\xd8":
            return _jpeg_dimensions(data)
        if data[:2] == b"BM" and len(data) >= 26:
            width, height = struct.unpack("<ii", data[18:26])
            return abs(int(width)), abs(int(height))
    except struct.error:
        return None
    return None


def _webp_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30:
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None


def _jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    index = 2
    size = len(data)
    while index + 4 <= size:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xFF:  # fill byte
            index += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no length field
            index += 2
            continue
        (length,) = struct.unpack(">H", data[index + 2 : index + 4])
        if marker in _JPEG_SOF_MARKERS:
            if index + 9 > size:
                return None
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        if length < 2:
            return None
        index += 2 + length
    return None


def _base64_image_dimensions(payload: str) -> Optional[Tuple[int, int]]:
    """Dimensions from the leading part of a base64 image payload."""
    head = "".join(payload[:_IMAGE_HEADER_B64_CHARS].split())
    head = head[: len(head) - len(head) % 4]
    if not head:
        return None
    try:
        raw = base64.b64decode(head, validate=False)
    except (binascii.Error, ValueError):
        return None
    return image_dimensions(raw)


def _data_url_dimensions(url: str) -> Optional[Tuple[int, int]]:
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        return None
    return _base64_image_dimensions(payload)


# --------------------------------------------------------------------------- #
# The shared o200k_base vocabulary                                             #
# --------------------------------------------------------------------------- #


def tiktoken_cache_dir() -> Path:
    """Where the vocabulary is cached: ``TIKTOKEN_CACHE_DIR`` if the operator
    set it, else ``<MIRIX_DIR>/tiktoken``."""
    configured = os.environ.get("TIKTOKEN_CACHE_DIR")
    if configured:
        return Path(configured)
    from mirix.settings import settings

    base = Path(settings.mirix_dir) if settings.mirix_dir else Path.home() / ".mirix"
    return base / "tiktoken"


class _TokenizerState:
    def __init__(self) -> None:
        self.encoding: Any = None
        self.failed_at: Optional[float] = None
        self.error: Optional[str] = None
        self.fallback_warned = False
        self.lock = threading.Lock()
        # A load is in flight (a preload, or a background load started by a
        # count): nobody starts a second one meanwhile.
        self.loading = False
        # The background load task, referenced so it is never garbage
        # collected mid-flight.
        self.background: Optional[asyncio.Task] = None


_tokenizer_state = _TokenizerState()


def _load_encoding_blocking() -> Any:
    """Load (downloading on first use) the vocabulary. Runs in a worker thread.

    A load that outlives a preload timeout still completes here and publishes
    the vocabulary for later counts.
    """
    state = _tokenizer_state
    with state.lock:
        if state.encoding is not None:
            return state.encoding
        cache_dir = tiktoken_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # tiktoken reads the cache location from the environment at load time.
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_dir))
        import tiktoken

        encoding = tiktoken.get_encoding(ENCODING_NAME)
        state.encoding = encoding
        state.failed_at = None
        state.error = None
        return encoding


async def preload_tokenizer(timeout: Optional[float] = None) -> bool:
    """Load the ``o200k_base`` vocabulary off the event loop (FastAPI lifespan).

    Returns True when it is loaded. On failure (offline, timeout) it logs a
    warning and returns False; counting then uses the 3-chars-per-token
    fallback, and :func:`ensure_tokenizer` retries later. Never raises.
    """
    state = _tokenizer_state
    if state.encoding is not None:
        state.loading = False  # a background load that found it loaded
        return True
    state.loading = True
    try:
        if timeout is None:
            await asyncio.to_thread(_load_encoding_blocking)
        else:
            await asyncio.wait_for(asyncio.to_thread(_load_encoding_blocking), timeout)
    except Exception as exc:  # noqa: BLE001 — offline must never break startup
        state.failed_at = time.monotonic()
        state.error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Could not load the %s tokenizer (%s); token counts fall back to "
            "%d characters per token until it loads",
            ENCODING_NAME,
            state.error,
            FALLBACK_CHARS_PER_TOKEN,
        )
        return False
    finally:
        state.loading = False
    return True


def _load_is_due() -> bool:
    """No vocabulary, no load in flight, and no failed load within the last
    :data:`TOKENIZER_RETRY_SECONDS`."""
    state = _tokenizer_state
    if state.encoding is not None or state.loading:
        return False
    return (
        state.failed_at is None
        or time.monotonic() - state.failed_at >= TOKENIZER_RETRY_SECONDS
    )


async def ensure_tokenizer() -> bool:
    """Load the vocabulary if needed (off the loop) and wait for it. A failed
    load is retried at most every :data:`TOKENIZER_RETRY_SECONDS`, so an
    offline host does not pay a load attempt on every call; while another load
    is in flight this returns at once (False). A learning run calls it before
    planning its chunks and batches, so a failed startup preload — or a
    process without the REST lifespan (``LocalClient``, the SDK) — still gets
    exact counts once the vocabulary can be loaded."""
    if _tokenizer_state.encoding is not None:
        return True
    if not _load_is_due():
        return False
    return await preload_tokenizer(timeout=TOKENIZER_PRELOAD_TIMEOUT_SECONDS)


def ensure_tokenizer_soon() -> None:
    """Start loading the vocabulary in the background when a load is due
    (see :func:`ensure_tokenizer`) and return at once — for the counting entry
    points, which must never wait for a download. Outside a running event
    loop it does nothing."""
    if not _load_is_due():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    state = _tokenizer_state
    state.loading = True  # before the task runs: no second task meanwhile
    try:
        state.background = loop.create_task(
            preload_tokenizer(timeout=TOKENIZER_PRELOAD_TIMEOUT_SECONDS)
        )
    except Exception:  # noqa: BLE001 — a count must never fail over this
        state.loading = False


def _reset_tokenizer_state_for_tests() -> None:
    global _tokenizer_state
    _tokenizer_state = _TokenizerState()


# --------------------------------------------------------------------------- #
# Counting                                                                     #
# --------------------------------------------------------------------------- #

_IMAGE_PART_TYPES = {"image_url", "image", "image_ref", "input_image"}
_REFERENCE_PART_TYPES = {"file_uri", "google_cloud_file_uri", "cloud_file_uri"}


class TokenCounter:
    """Approximate ``o200k_base`` token counts (see the module docstring).

    ``encoding`` is injectable (anything with ``encode(text) -> list``); by
    default the shared, preloaded vocabulary is used, or the 3-chars-per-token
    fallback while it is unavailable. The sync ``count_*`` methods are pure
    CPU; async code should use :meth:`acount_text` / :meth:`acount_request`,
    which move large inputs off the event loop.
    """

    def __init__(self, encoding: Any = None):
        self._encoding = encoding

    @property
    def encoding(self) -> Any:
        return (
            self._encoding if self._encoding is not None else _tokenizer_state.encoding
        )

    @property
    def is_exact(self) -> bool:
        """False while counting with the characters-per-token fallback."""
        return self.encoding is not None

    # ----------------------------- text ----------------------------------- #

    def count_text(self, text: Any) -> int:
        if text is None:
            return 0
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        encoding = self.encoding
        if encoding is None:
            if not _tokenizer_state.fallback_warned:
                _tokenizer_state.fallback_warned = True
                logger.warning(
                    "Token counts use the %d-characters-per-token estimate: the "
                    "%s tokenizer is not loaded%s",
                    FALLBACK_CHARS_PER_TOKEN,
                    ENCODING_NAME,
                    f" ({_tokenizer_state.error})" if _tokenizer_state.error else "",
                )
            return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except TypeError:  # an injected encoding without that keyword
            return len(encoding.encode(text))

    async def acount_text(self, text: Any) -> int:
        if self._encoding is None:
            ensure_tokenizer_soon()
        if isinstance(text, str) and len(text) > LARGE_TEXT_CHARS:
            return await asyncio.to_thread(self.count_text, text)
        return self.count_text(text)

    # ---------------------------- content --------------------------------- #

    def count_image(self, part: Any) -> int:
        """Tokens of one image reference (any supported shape)."""
        dims = _image_part_dimensions(part)
        return image_tokens(*dims) if dims else UNKNOWN_IMAGE_TOKENS

    def count_content(self, value: Any) -> int:
        """Tokens of message content: a string, a list of parts, a part dict
        (text / image / tool call / tool result, in OpenAI, Anthropic, Google or
        ``content_parts`` shape), or a pydantic content object."""
        if value is None:
            return 0
        if isinstance(value, str):
            return self.count_text(value)
        if isinstance(value, (bytes, bytearray)):
            return 0  # raw binary payloads are images/files, counted by their part
        if isinstance(value, (int, float, bool)):
            return self.count_text(json.dumps(value))
        if isinstance(value, (list, tuple)):
            return sum(self.count_content(item) for item in value)
        if hasattr(value, "model_dump") and not isinstance(value, dict):
            value = value.model_dump()
        if not isinstance(value, dict):
            return self.count_text(str(value))
        if _is_image_part(value):
            return self.count_image(value)
        part_type = value.get("type")
        if part_type in _REFERENCE_PART_TYPES:
            return UNKNOWN_IMAGE_TOKENS
        if part_type == "text" and isinstance(value.get("text"), str):
            return self.count_text(value["text"])
        if isinstance(value.get("function"), dict):  # OpenAI tool call
            function = value["function"]
            return self.count_text(function.get("name")) + self.count_arguments(
                function.get("arguments")
            )
        if part_type == "tool_use":  # Anthropic
            return self.count_text(value.get("name")) + self.count_arguments(
                value.get("input")
            )
        if "functionCall" in value and isinstance(value["functionCall"], dict):
            call = value["functionCall"]
            return self.count_text(call.get("name")) + self.count_arguments(
                call.get("args")
            )
        # Anything else (tool results, Google parts, unknown shapes): its
        # values, recursively — keys are structural and cost little.
        return sum(
            self.count_content(item)
            for key, item in value.items()
            if key not in _STRUCTURAL_KEYS
        )

    def count_arguments(self, arguments: Any) -> int:
        if arguments is None:
            return 0
        if isinstance(arguments, str):
            return self.count_text(arguments)
        return self.count_text(json.dumps(arguments, ensure_ascii=False, default=str))

    def count_tool_calls(self, tool_calls: Optional[Iterable[Any]]) -> int:
        return sum(self.count_content(call) for call in (tool_calls or []))

    def count_tools(self, tools: Optional[Iterable[Any]]) -> int:
        """Tool / function schemas, as the JSON the provider receives."""
        total = 0
        for tool in tools or []:
            if hasattr(tool, "model_dump"):
                tool = tool.model_dump(exclude_none=True)
            total += self.count_text(json.dumps(tool, ensure_ascii=False, default=str))
        return total

    def count_message(self, message: Any) -> int:
        """One chat message (dict or pydantic) incl. tool calls and overhead."""
        if hasattr(message, "model_dump") and not isinstance(message, dict):
            message = message.model_dump(exclude_none=True)
        if not isinstance(message, dict):
            return MESSAGE_OVERHEAD_TOKENS + self.count_content(message)
        total = MESSAGE_OVERHEAD_TOKENS
        for key in ("content", "parts", "text"):
            if key in message:
                total += self.count_content(message[key])
        total += self.count_tool_calls(message.get("tool_calls"))
        if isinstance(message.get("function_call"), dict):
            total += self.count_content({"function": message["function_call"]})
        for key in ("name", "tool_call_id"):
            if isinstance(message.get(key), str):
                total += self.count_text(message[key])
        return total

    def count_messages(self, messages: Optional[Iterable[Any]]) -> int:
        return sum(self.count_message(m) for m in (messages or []))

    def count_request(self, request: dict) -> int:
        """Estimated input tokens of a provider request body (OpenAI chat,
        Anthropic messages or Google generateContent shape)."""
        total = 0
        total += self.count_messages(request.get("messages"))
        total += self.count_messages(request.get("contents"))
        for key in ("system", "system_instruction", "systemInstruction"):
            if key in request:
                total += self.count_content(request[key])
        for key in ("tools", "functions"):
            total += self.count_tools(request.get(key))
        return total

    async def acount_request(self, request: dict) -> int:
        """:meth:`count_request`, in a worker thread when the request carries
        more than :data:`LARGE_TEXT_CHARS` characters of text."""
        if self._encoding is None:
            ensure_tokenizer_soon()
        if _text_size(request, LARGE_TEXT_CHARS) > LARGE_TEXT_CHARS:
            return await asyncio.to_thread(self.count_request, request)
        return self.count_request(request)


# Dict keys that carry structure, not model-visible text.
_STRUCTURAL_KEYS = {
    "type",
    "role",
    "id",
    "tool_call_id",
    "tool_use_id",
    "cache_control",
    "detail",
    "mime_type",
    "mimeType",
    "media_type",
    "is_error",
}


def _is_image_part(part: dict) -> bool:
    """An image reference in any supported shape (typed parts, Google parts)."""
    part_type = part.get("type")
    if part_type in _IMAGE_PART_TYPES:
        return True
    if part_type is not None:
        return False
    if "inline_data" in part or "inlineData" in part or "image_id" in part:
        return True
    file_data = part.get("file_data") or part.get("fileData")
    if isinstance(file_data, dict):
        mime = file_data.get("mime_type") or file_data.get("mimeType") or ""
        return str(mime).startswith("image/")
    return False


def _image_part_dimensions(part: Any) -> Optional[Tuple[int, int]]:
    """Width/height of an image part when it can be known without I/O."""
    if hasattr(part, "model_dump") and not isinstance(part, dict):
        part = part.model_dump()
    if not isinstance(part, dict):
        return None
    width, height = part.get("width"), part.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width, height
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if isinstance(image_url, str):
        return _data_url_dimensions(image_url)
    source = part.get("source")  # Anthropic
    if isinstance(source, dict) and source.get("type") == "base64":
        data = source.get("data")
        return _base64_image_dimensions(data) if isinstance(data, str) else None
    inline = part.get("inline_data") or part.get("inlineData")  # Google
    if isinstance(inline, dict) and isinstance(inline.get("data"), str):
        return _base64_image_dimensions(inline["data"])
    return None


def _text_size(value: Any, stop_at: int) -> int:
    """Characters of text in ``value`` (image payloads excluded), counting
    stops early once ``stop_at`` is exceeded."""
    total = 0
    stack = [value]
    while stack and total <= stop_at:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, dict):
            if _is_image_part(item):
                continue
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return total


_DEFAULT_COUNTER = TokenCounter()


def get_token_counter() -> TokenCounter:
    """The shared counter (uses the preloaded vocabulary)."""
    return _DEFAULT_COUNTER


# --------------------------------------------------------------------------- #
# Limits and budgets                                                           #
# --------------------------------------------------------------------------- #


def _fraction(value: int) -> int:
    return value * BUDGET_NUMERATOR // BUDGET_DENOMINATOR


@dataclass(frozen=True)
class ModelLimits:
    """A model's context window and output limit, with the budget formulas."""

    context_window: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if int(self.context_window) <= 0:
            raise ValueError(
                f"context_window must be positive, got {self.context_window}"
            )
        if int(self.max_output_tokens) <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {self.max_output_tokens}"
            )

    @classmethod
    def from_llm_config(cls, llm_config: Any) -> "ModelLimits":
        """``context_window`` and ``max_output_tokens`` of an LLMConfig;
        without ``max_output_tokens`` the configured ``max_tokens`` is the
        output limit, and without either the whole window."""
        window = int(llm_config.context_window)
        max_output = (
            getattr(llm_config, "max_output_tokens", None)
            or getattr(llm_config, "max_tokens", None)
            or window
        )
        return cls(context_window=window, max_output_tokens=int(max_output))

    @property
    def usable_window(self) -> int:
        """``floor(window * 0.8)``: input + output of one call must fit here."""
        return _fraction(self.context_window)

    @property
    def output_ceiling(self) -> int:
        """``floor(max_output * 0.8)``: the most output one call may request."""
        return _fraction(self.max_output_tokens)

    @property
    def planning_output_reserve(self) -> int:
        """``min(floor(max_output * 0.8), floor(window * 0.1))``: the output room
        planners keep free when sizing inputs (chunks, batches)."""
        return min(self.output_ceiling, self.context_window // PLANNING_WINDOW_DIVISOR)

    def input_budget(
        self, system_tokens: int = 0, output_reserve: Optional[int] = None
    ) -> int:
        """``floor(window * 0.8) - system - reserve`` (reserve defaults to the
        planning reserve). May be negative when the system prompt alone is
        too large."""
        reserve = (
            self.planning_output_reserve if output_reserve is None else output_reserve
        )
        return self.usable_window - int(system_tokens) - int(reserve)

    def output_cap(self, input_tokens: int) -> int:
        """``min(floor(max_out * 0.8), floor(window * 0.8) - input)``: the
        ``max_tokens`` for a call with this input. Never an upper bound hit by
        normal output: planners keep inputs within :meth:`input_budget`."""
        return min(self.output_ceiling, self.usable_window - int(input_tokens))

    def checked_output_cap(
        self, input_tokens: int, *, minimum: int = MIN_OUTPUT_TOKENS
    ) -> int:
        """:meth:`output_cap`, or :class:`ContextBudgetExceeded` below ``minimum``."""
        cap = self.output_cap(input_tokens)
        if cap < minimum:
            raise ContextBudgetExceeded(
                estimated_input_tokens=int(input_tokens),
                output_cap=cap,
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
                minimum_output_tokens=minimum,
            )
        return cap


async def output_cap_for_request(
    request: dict, llm_config: Any, *, counter: Optional[TokenCounter] = None
) -> int:
    """The ``max_tokens`` / ``max_completion_tokens`` for one request.

    Estimates the request's input (messages, system, tool schemas, images) and
    returns ``output_cap`` of it; raises :class:`ContextBudgetExceeded` before
    anything is sent when fewer than :data:`MIN_OUTPUT_TOKENS` would remain.
    """
    counter = counter or get_token_counter()
    limits = ModelLimits.from_llm_config(llm_config)
    estimated = await counter.acount_request(request)
    cap = limits.checked_output_cap(estimated)
    logger.debug(
        "Output cap %d for an estimated %d input tokens (window %d, max output %d)",
        cap,
        estimated,
        limits.context_window,
        limits.max_output_tokens,
    )
    return cap
