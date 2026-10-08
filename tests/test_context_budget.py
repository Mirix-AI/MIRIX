"""Unified context budget.

* the budget table, number by number, for two model configurations;
* the OpenAI high-detail image formula times the calibration factor 7.5 (the
  estimate the harness gateway also uses), and image-size detection from headers;
* TokenCounter over text, content-part lists in every provider shape, tool
  calls and tool schemas;
* the tokenizer: preloaded off the event loop into <MIRIX_DIR>/tiktoken, the
  offline fallback (3 characters per token + one warning), large counts in a
  worker thread — the loop keeps running throughout;
* server startup (``rest_api.initialize``) preloads the tokenizer.

No network: tiktoken's loader is replaced wherever a download could happen.
"""

import asyncio
import base64
import hashlib
import io
import math
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mirix.errors import ContextBudgetExceeded
from mirix.schemas.llm_config import LLMConfig
from mirix.services import context_budget
from mirix.services.context_budget import (
    FALLBACK_CHARS_PER_TOKEN,
    IMAGE_TOKEN_MULTIPLIER,
    LARGE_TEXT_CHARS,
    MIN_OUTPUT_TOKENS,
    UNKNOWN_IMAGE_TOKENS,
    ModelLimits,
    TokenCounter,
    image_dimensions,
    image_formula_tokens,
    image_tokens,
    output_cap_for_request,
)

# Two model configurations: a smaller output limit, and a large one.
LUNA = dict(context_window=1_050_000, max_output_tokens=128_000)
BIG_OUTPUT = dict(context_window=1_310_720, max_output_tokens=943_717)


@pytest.fixture(autouse=True)
def fresh_tokenizer_state():
    context_budget._reset_tokenizer_state_for_tests()
    yield
    context_budget._reset_tokenizer_state_for_tests()


class WordEncoding:
    """A deterministic stand-in for o200k_base: one token per word."""

    def __init__(self, record_threads=None, delay=0.0):
        self.record_threads = record_threads
        self.delay = delay

    def encode(self, text, disallowed_special=()):
        if self.record_threads is not None:
            self.record_threads.append(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)  # a slow encode, in whatever thread runs it
        return text.split()


# --------------------------------------------------------------------------- #
# The budget table                                                             #
# --------------------------------------------------------------------------- #


class TestM31Table:
    @pytest.mark.parametrize(
        "limits, usable, ceiling, reserve, chunk",
        [
            # window*0.8 | max_out*0.8 | planning reserve | distill chunk input
            (LUNA, 840_000, 102_400, 102_400, 737_600),
            (BIG_OUTPUT, 1_048_576, 754_973, 131_072, 917_504),
        ],
        ids=["luna", "big-output"],
    )
    def test_the_table(self, limits, usable, ceiling, reserve, chunk):
        model = ModelLimits(**limits)
        assert model.usable_window == usable
        assert model.output_ceiling == ceiling
        assert model.planning_output_reserve == reserve
        # A distiller chunk's input limit (before the system prompt) =
        # input_budget(0, planning reserve)
        assert model.input_budget(0, model.planning_output_reserve) == chunk
        assert model.input_budget() == chunk  # the reserve defaults to planning

    @pytest.mark.parametrize("limits", [LUNA, BIG_OUTPUT], ids=["luna", "big-output"])
    def test_from_llm_config(self, limits):
        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=limits["context_window"],
            max_output_tokens=limits["max_output_tokens"],
        )
        assert ModelLimits.from_llm_config(config) == ModelLimits(**limits)

    def test_input_budget_subtracts_system_and_reserve(self):
        luna = ModelLimits(**LUNA)
        assert luna.input_budget(5_000, 102_400) == 840_000 - 5_000 - 102_400
        assert luna.input_budget(5_000, 0) == 835_000

    def test_output_cap(self):
        luna = ModelLimits(**LUNA)
        big = ModelLimits(**BIG_OUTPUT)
        # Small inputs: the model's output ceiling binds.
        assert luna.output_cap(0) == 102_400
        assert luna.output_cap(737_600) == 102_400  # a full planned chunk
        assert big.output_cap(917_504) == 131_072  # window-limited already
        assert big.output_cap(100_000) == 754_973
        # Large inputs: what is left of 80% of the window binds.
        assert luna.output_cap(800_000) == 40_000
        assert big.output_cap(1_000_000) == 48_576

    def test_a_planned_chunk_always_leaves_the_planning_reserve(self):
        for limits in (LUNA, BIG_OUTPUT):
            model = ModelLimits(**limits)
            assert (
                model.output_cap(model.input_budget(0)) >= model.planning_output_reserve
            )

    def test_checked_cap_refuses_below_1024(self):
        luna = ModelLimits(**LUNA)
        assert luna.checked_output_cap(840_000 - MIN_OUTPUT_TOKENS) == 1024
        with pytest.raises(ContextBudgetExceeded) as exc:
            luna.checked_output_cap(840_000 - MIN_OUTPUT_TOKENS + 1)
        assert exc.value.output_cap == 1023
        assert exc.value.estimated_input_tokens == 838_977
        assert exc.value.context_window == 1_050_000

    def test_max_tokens_is_the_fallback_output_limit(self):
        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=100_000,
            max_tokens=4096,
        )
        assert ModelLimits.from_llm_config(config).max_output_tokens == 4096
        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=100_000,
            max_tokens=None,
        )
        assert ModelLimits.from_llm_config(config).max_output_tokens == 100_000

    def test_integer_arithmetic(self):
        # floor(x * 0.8) with no float rounding at any size.
        for window in (1, 4, 5, 9, 1_310_720, 10**12 + 3):
            assert ModelLimits(window, window).usable_window == window * 4 // 5

    @pytest.mark.parametrize("window, out", [(0, 10), (10, 0), (-5, 10)])
    def test_invalid_limits(self, window, out):
        with pytest.raises(ValueError):
            ModelLimits(window, out)

    def test_max_output_tokens_must_be_positive(self):
        with pytest.raises(ValueError):
            LLMConfig(
                model="m",
                model_endpoint_type="openai",
                context_window=1,
                max_output_tokens=0,
            )


# --------------------------------------------------------------------------- #
# Images                                                                       #
# --------------------------------------------------------------------------- #


def _encoded(size, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "blue").save(buf, fmt)
    return buf.getvalue()


class TestImageTokens:
    @pytest.mark.parametrize(
        "size, formula, tokens",
        [
            ((1024, 1024), 765, 5738),  # OpenAI's documented example
            ((2048, 4096), 1105, 8288),  # OpenAI's documented example
            ((4096, 8192), 1105, 8288),  # fit to 2048 first
            ((768, 768), 765, 5738),
            ((512, 512), 255, 1913),
            ((100, 100), 255, 1913),  # never scaled up
            ((1, 1), 255, 1913),
            ((1700, 2200), 765, 5738),  # a DocVQA-like scan
            ((2048, 768), 1445, 10838),  # the maximum
        ],
    )
    def test_high_detail_formula_times_the_factor(self, size, formula, tokens):
        assert image_formula_tokens(*size) == formula
        assert image_tokens(*size) == tokens

    def test_the_factor_is_seven_and_a_half_rounded_up(self):
        """The factor is applied and rounded up; the harness's
        protocol.IMAGE_TOKEN_MULTIPLIER must be the same (its tests assert it)."""
        assert IMAGE_TOKEN_MULTIPLIER == Fraction(15, 2)
        assert image_tokens(1024, 1024) == math.ceil(765 * 7.5)  # 5737.5 -> 5738
        assert all(
            image_tokens(w, h) >= 7.5 * image_formula_tokens(w, h)
            for w in range(64, 4097, 211)
            for h in range(64, 4097, 257)
        )

    @pytest.mark.parametrize("size", [(0, 10), (None, 5), ("x", 5), (-1, -1)])
    def test_unknown_sizes_count_as_the_maximum(self, size):
        assert image_formula_tokens(*size) == 1445
        assert image_tokens(*size) == UNKNOWN_IMAGE_TOKENS == 10838

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "WEBP", "BMP"])
    def test_dimensions_from_headers(self, fmt):
        assert image_dimensions(_encoded((37, 21), fmt)) == (37, 21)

    def test_undecodable_headers(self):
        assert image_dimensions(b"not an image") is None
        assert image_dimensions(b"\x89PNG\r\n\x1a\n") is None  # truncated


class TestContentShapes:
    def setup_method(self):
        self.counter = TokenCounter(encoding=WordEncoding())

    def _b64(self, size, fmt="PNG"):
        return base64.b64encode(_encoded(size, fmt)).decode()

    def test_openai_data_url_image(self):
        part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{self._b64((1024, 1024))}"},
        }
        assert self.counter.count_content(part) == 5738

    def test_remote_and_unresolved_images_count_as_the_maximum(self):
        assert (
            self.counter.count_content(
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
            )
            == 10838
        )
        # MIRIX's pre-fill form (a stored file id) and cloud references.
        assert (
            self.counter.count_content({"type": "image_url", "image_id": "f"}) == 10838
        )
        assert self.counter.count_content({"type": "file_uri", "file_id": "f"}) == 10838

    def test_anthropic_and_google_images(self):
        data = self._b64((100, 100), "JPEG")
        anthropic_part = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        }
        google_part = {"inline_data": {"mime_type": "image/jpeg", "data": data}}
        assert self.counter.count_content(anthropic_part) == 1913
        assert self.counter.count_content(google_part) == 1913

    def test_conversation_store_image_ref(self):
        ref = {"type": "image_ref", "file_id": "file-1", "width": 2048, "height": 4096}
        assert self.counter.count_content(ref) == 8288

    def test_text_parts_and_strings(self):
        parts = [
            {"type": "text", "text": "one two three"},
            {"type": "image_ref", "width": 512, "height": 512},
            "four five",
        ]
        assert self.counter.count_content(parts) == 3 + 1913 + 2

    def test_tool_calls_in_every_shape(self):
        openai_call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "go", "arguments": '{"to": "the sink"}'},
        }
        anthropic_call = {
            "type": "tool_use",
            "id": "t",
            "name": "go",
            "input": {"to": "sink"},
        }
        google_call = {"functionCall": {"name": "go", "args": {"to": "sink"}}}
        assert self.counter.count_content(openai_call) == 1 + 3
        assert self.counter.count_content(anthropic_call) == 1 + 2
        assert self.counter.count_content(google_call) == 1 + 2

    def test_messages_include_tool_calls_and_overhead(self):
        message = {
            "role": "assistant",
            "content": "let me look",
            "tool_calls": [
                {"type": "function", "function": {"name": "open", "arguments": "{}"}}
            ],
        }
        overhead = context_budget.MESSAGE_OVERHEAD_TOKENS
        assert self.counter.count_message(message) == overhead + 3 + 1 + 1

    def test_tool_schemas_count_as_their_json(self):
        tools = [{"type": "function", "function": {"name": "go", "parameters": {}}}]
        expected = len(
            '{"type": "function", "function": {"name": "go", "parameters": {}}}'.split()
        )
        assert self.counter.count_tools(tools) == expected

    def test_request_shapes(self):
        openai_request = {
            "model": "m",
            "messages": [{"role": "user", "content": "a b c"}],
            "tools": [{"name": "t"}],
        }
        anthropic_request = {
            "system": "be brief",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "a b"}]}
            ],
        }
        google_request = {
            "contents": [{"role": "user", "parts": [{"text": "a b c d"}]}],
            "generation_config": {"max_output_tokens": 10},
        }
        overhead = context_budget.MESSAGE_OVERHEAD_TOKENS
        assert self.counter.count_request(openai_request) == overhead + 3 + 2
        assert self.counter.count_request(anthropic_request) == 2 + overhead + 2
        assert self.counter.count_request(google_request) == overhead + 4

    def test_image_payloads_are_never_counted_as_text(self):
        """A 1 MB base64 image is ~1,913 tokens (one tile), not ~350k."""
        big = base64.b64encode(_encoded((200, 200), "BMP") * 30).decode()
        part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/bmp;base64,{big}"},
        }
        assert len(big) > LARGE_TEXT_CHARS
        assert self.counter.count_content(part) == 1913


# --------------------------------------------------------------------------- #
# Tokenizer loading, fallback and the event loop                               #
# --------------------------------------------------------------------------- #


def _capture_warnings(monkeypatch):
    """The module logger does not propagate to the root (caplog cannot see it)."""
    captured = []

    def warning(message, *args, **kwargs):
        captured.append(message % args if args else message)

    monkeypatch.setattr(context_budget.logger, "warning", warning)
    return captured


def _patch_tiktoken(monkeypatch, loader):
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", loader)


@pytest.mark.asyncio
class TestTokenizer:
    async def test_offline_falls_back_to_three_chars_per_token(self, monkeypatch):
        def offline(name):
            raise OSError("network is unreachable")

        _patch_tiktoken(monkeypatch, offline)
        warnings = _capture_warnings(monkeypatch)
        assert await context_budget.preload_tokenizer() is False
        assert "fall back to 3 characters per token" in warnings[0]
        assert "network is unreachable" in warnings[0]
        counter = TokenCounter()
        assert not counter.is_exact
        assert counter.count_text("abcdefg") == 3  # ceil(7 / 3)
        assert counter.count_text("abcdef") == 2
        assert FALLBACK_CHARS_PER_TOKEN == 3
        # One more warning the first time a count uses the estimate, not per count.
        counter.count_text("more text")
        assert len(warnings) == 2 and "estimate" in warnings[1]

    async def test_a_failed_load_is_retried_only_after_the_cooldown(self, monkeypatch):
        calls = []

        def offline(name):
            calls.append(name)
            raise OSError("offline")

        _patch_tiktoken(monkeypatch, offline)
        assert await context_budget.ensure_tokenizer() is False
        assert await context_budget.ensure_tokenizer() is False
        assert calls == ["o200k_base"]
        monkeypatch.setattr(context_budget, "TOKENIZER_RETRY_SECONDS", 0.0)
        assert await context_budget.ensure_tokenizer() is False
        assert len(calls) == 2

    async def test_a_count_loads_the_vocabulary_in_the_background(self, monkeypatch):
        # After a failed preload, a count (outside any learning run,
        # e.g. the SDK or a LocalClient) starts a background load once the
        # cooldown passed; it never waits for it and never starts two.
        calls = []

        def offline(name):
            calls.append(name)
            raise OSError("offline")

        _patch_tiktoken(monkeypatch, offline)
        assert await context_budget.preload_tokenizer() is False
        counter = TokenCounter()
        assert await counter.acount_text("a b c") == 2  # within the cooldown
        assert context_budget._tokenizer_state.background is None
        monkeypatch.setattr(context_budget, "TOKENIZER_RETRY_SECONDS", 0.0)

        def online(name):
            calls.append(name)
            return WordEncoding()

        _patch_tiktoken(monkeypatch, online)
        assert await counter.acount_text("a b c") == 2  # the estimate, at once
        assert await counter.acount_text("a b c") == 2  # one load in flight
        await context_budget._tokenizer_state.background
        assert calls == ["o200k_base", "o200k_base"]
        assert counter.is_exact and await counter.acount_text("a b c") == 3

    async def test_loaded_under_the_mirix_dir(self, monkeypatch, tmp_path):
        from mirix.settings import settings

        monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
        monkeypatch.setattr(settings, "mirix_dir", tmp_path / "mirix")
        seen = {}

        def loader(name):
            import os

            seen["name"] = name
            seen["cache"] = os.environ.get("TIKTOKEN_CACHE_DIR")
            return WordEncoding()

        _patch_tiktoken(monkeypatch, loader)
        assert await context_budget.preload_tokenizer() is True
        assert seen == {
            "name": "o200k_base",
            "cache": str(tmp_path / "mirix" / "tiktoken"),
        }
        assert (tmp_path / "mirix" / "tiktoken").is_dir()
        assert TokenCounter().is_exact
        assert TokenCounter().count_text("a b c") == 3

    async def test_an_operator_cache_dir_is_respected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "ops"))
        assert context_budget.tiktoken_cache_dir() == tmp_path / "ops"

    async def test_loading_never_blocks_the_event_loop(self, monkeypatch):
        loader_threads = []

        def slow_loader(name):
            loader_threads.append(threading.get_ident())
            time.sleep(0.3)  # a download
            return WordEncoding()

        _patch_tiktoken(monkeypatch, slow_loader)
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            assert await context_budget.preload_tokenizer() is True
        finally:
            task.cancel()
        assert loader_threads and loader_threads[0] != threading.get_ident()
        assert ticks >= 10

    async def test_a_preload_timeout_falls_back(self, monkeypatch):
        release = threading.Event()

        def hanging(name):
            release.wait(5)
            return WordEncoding()

        _patch_tiktoken(monkeypatch, hanging)
        try:
            assert await context_budget.preload_tokenizer(timeout=0.05) is False
            assert not TokenCounter().is_exact
        finally:
            release.set()

    async def test_large_texts_are_counted_in_a_worker_thread(self):
        threads = []
        counter = TokenCounter(encoding=WordEncoding(record_threads=threads, delay=0.2))
        main = threading.get_ident()
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            big = "word " * (LARGE_TEXT_CHARS // 5 + 1)
            assert await counter.acount_text(big) == LARGE_TEXT_CHARS // 5 + 1
            request = {"messages": [{"role": "user", "content": big}]}
            await counter.acount_request(request)
        finally:
            task.cancel()
        assert threads and all(t != main for t in threads)
        assert ticks >= 10  # 0.4 s of encoding, the loop kept running

    async def test_small_texts_stay_inline(self):
        threads = []
        counter = TokenCounter(encoding=WordEncoding(record_threads=threads))
        assert await counter.acount_text("a b") == 2
        assert threads == [threading.get_ident()]

    async def test_the_real_o200k_vocabulary_when_cached(self, monkeypatch):
        """Exact counts with tiktoken's own o200k_base — only from a local
        cache (never downloaded by the test suite)."""
        blob = (
            "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
        )
        cache = Path(tempfile.gettempdir()) / "data-gym-cache"
        if not (cache / hashlib.sha1(blob.encode()).hexdigest()).exists():
            pytest.skip("o200k_base is not in the local tiktoken cache")
        import tiktoken.load

        def no_download(path):
            raise AssertionError(f"the test tried to download {path}")

        monkeypatch.setattr(tiktoken.load, "read_file", no_download)
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
        assert await context_budget.preload_tokenizer() is True
        counter = TokenCounter()
        assert counter.is_exact
        assert counter.count_text("hello world") == 2
        assert counter.count_text("<|endoftext|>") > 1  # special text is just text


# --------------------------------------------------------------------------- #
# output_cap_for_request                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestOutputCapForRequest:
    def _config(self, window, max_out):
        return LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=window,
            max_output_tokens=max_out,
        )

    async def test_cap_from_the_estimated_input(self):
        counter = TokenCounter(encoding=WordEncoding())
        request = {"messages": [{"role": "user", "content": "w " * 1000}]}
        estimated = counter.count_request(request)
        cap = await output_cap_for_request(
            request, self._config(10_000, 9_000), counter=counter
        )
        assert cap == min(9_000 * 4 // 5, 10_000 * 4 // 5 - estimated)

    async def test_refused_before_sending_when_below_1024(self):
        counter = TokenCounter(encoding=WordEncoding())
        request = {"messages": [{"role": "user", "content": "w " * 3000}]}
        with pytest.raises(ContextBudgetExceeded):
            await output_cap_for_request(
                request, self._config(4_000, 4_000), counter=counter
            )


# --------------------------------------------------------------------------- #
# Server startup                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_startup_preloads_the_tokenizer(monkeypatch):
    """rest_api.initialize (run by the FastAPI lifespan) preloads o200k_base."""
    from mirix import observability
    from mirix.server import rest_api

    calls = []

    async def fake_preload(timeout=None):
        calls.append(timeout)
        return True

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(context_budget, "preload_tokenizer", fake_preload)
    monkeypatch.setattr(rest_api, "ensure_tables_created", noop)
    monkeypatch.setattr(rest_api, "initialize_queue", noop)
    monkeypatch.setattr(observability, "initialize_langfuse", noop)
    monkeypatch.setattr(
        rest_api, "get_server", lambda: SimpleNamespace(ensure_defaults=noop)
    )
    await rest_api.initialize()
    assert calls == [context_budget.TOKENIZER_PRELOAD_TIMEOUT_SECONDS]
