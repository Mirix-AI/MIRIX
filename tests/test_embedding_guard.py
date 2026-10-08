"""The embedding length guard.

Hermetic: token counts come from injected counters (or the deterministic
3-characters-per-token fallback), provider models are in-process fakes or the
offline fake OpenAI-compatible server. No provider is ever called.

* the limit table (``MIRIX_EMBEDDING_TOKEN_LIMITS``), the unknown-model default
  and its warning, the chunk cap (2,048 -> 1,800);
* splitting: paragraph > sentence > hard token boundary, every chunk within
  the cap, nothing lost;
* merging: token-weighted mean -> L2 normalization -> padding to 4096 LAST;
* the guard around every model of the ``embedding_model()`` factory, and the
  count of chunked calls;
* OpenRouter (and the other raw-HTTP embedders) raise on a non-200
  answer instead of parsing the error body.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

import mirix.embedding_guard as guard
from fakes.fake_llm_server import Fault
from mirix import constants
from mirix.constants import MAX_EMBEDDING_DIM, _embedding_token_limits_setting
from mirix.embedding_guard import (
    EmbeddingChunkCounter,
    GuardedEmbedding,
    counting_embedding_chunks,
    embedding_chunk_cap,
    embedding_token_limit,
    merge_chunk_embeddings,
    pad_embedding,
    record_chunked_embedding_call,
    split_text_for_embedding,
)
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.services import context_budget
from mirix.settings import model_settings

REPO = Path(__file__).resolve().parent.parent


def words(text: str) -> int:
    """A deterministic counter: one token per whitespace-separated word."""
    return len(text.split())


def chars(text: str) -> int:
    """A deterministic counter: one token per character."""
    return len(text)


class WordCounter:
    """TokenCounter-shaped: ``count_text`` / ``acount_text`` by words."""

    def count_text(self, text):
        return words(text)

    async def acount_text(self, text):
        return words(text)


class FakeModel:
    """A provider model: a deterministic vector per text, inputs recorded."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.inputs = []
        self.label = "fake-inner"

    def vector(self, text: str):
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        return [rng.uniform(-1, 1) for _ in range(self.dim)]

    async def get_text_embedding(self, text):
        self.inputs.append(text)
        return self.vector(text)


@pytest.fixture(autouse=True)
def direct_to_the_fake(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def fallback_counter():
    """The shared counter, on its 3-characters-per-token fallback (no
    vocabulary load is attempted)."""
    context_budget._reset_tokenizer_state_for_tests()
    context_budget._tokenizer_state.failed_at = time.monotonic() + 10**9
    yield context_budget.get_token_counter()
    context_budget._reset_tokenizer_state_for_tests()


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #


class TestLimits:
    def test_the_default_table(self):
        assert constants.DEFAULT_EMBEDDING_TOKEN_LIMITS == {
            "gemini-embedding-001": 2048
        }
        assert _embedding_token_limits_setting(None) == {"gemini-embedding-001": 2048}
        assert _embedding_token_limits_setting("  ") == {"gemini-embedding-001": 2048}

    def test_the_setting_replaces_the_table(self):
        assert _embedding_token_limits_setting(
            '{"text-embedding-3-small": 8191, "gemini-embedding-001": 2048}'
        ) == {"text-embedding-3-small": 8191, "gemini-embedding-001": 2048}

    def test_invalid_settings_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _embedding_token_limits_setting("{not json") == {
                "gemini-embedding-001": 2048
            }
            assert _embedding_token_limits_setting("[2048]") == {
                "gemini-embedding-001": 2048
            }
            assert _embedding_token_limits_setting(
                '{"a": 0, "b": -1, "c": "2048", "d": true, "e": 1.5, "ok": 512}'
            ) == {"ok": 512}
        assert (
            sum("MIRIX_EMBEDDING_TOKEN_LIMITS" in r.message for r in caplog.records)
            == 7
        )

    def test_lookup_ignores_a_provider_prefix(self):
        table = {"gemini-embedding-001": 2048, "x": 99}
        assert embedding_token_limit("gemini-embedding-001", table) == 2048
        assert embedding_token_limit("google/gemini-embedding-001", table) == 2048
        assert embedding_token_limit("models/gemini-embedding-001", table) == 2048
        assert embedding_token_limit("x", table) == 99

    def test_an_unknown_model_is_2048_with_one_warning(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            guard.logger, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        guard._warned_unknown_models.discard("never-heard-of-it")
        assert embedding_token_limit("never-heard-of-it", {}) == 2048
        assert embedding_token_limit("never-heard-of-it", {}) == 2048
        assert len([w for w in warnings if "never-heard-of-it" in w]) == 1

    @pytest.mark.parametrize(
        "limit, cap",
        [(2048, 1800), (8191, 7200), (512, 400), (100, 88), (50, 44), (1, 1)],
    )
    def test_the_chunk_cap(self, limit, cap):
        assert embedding_chunk_cap(limit) == cap

    @pytest.mark.parametrize("raw, expected", [(None, 16), ("7", 7), ("0", 16)])
    def test_the_chunk_limit_setting(self, raw, expected):
        import os
        import subprocess
        import sys

        env = {k: v for k, v in os.environ.items() if k != "MIRIX_EMBEDDING_MAX_CHUNKS"}
        if raw is not None:
            env["MIRIX_EMBEDDING_MAX_CHUNKS"] = raw
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mirix.constants as c; print(c.EMBEDDING_MAX_CHUNKS)",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip().splitlines()[-1] == str(expected)

    def test_gemini_through_the_factory_config_is_1800(self):
        model = GuardedEmbedding(FakeModel(), model_name="google/gemini-embedding-001")
        assert (model.token_limit, model.chunk_cap) == (2048, 1800)


# --------------------------------------------------------------------------- #
# Splitting                                                                    #
# --------------------------------------------------------------------------- #


def _paragraph(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(
        rng.choice(["open", "check", "queue", "merge", "retry"]) for _ in range(n_words)
    )


def _check_chunks(text, chunks, cap, count):
    assert all(count(chunk) <= cap for chunk, _ in chunks), [
        count(c) for c, _ in chunks
    ]
    assert all(tokens == max(1, count(chunk)) for chunk, tokens in chunks)
    assert "".join(chunk for chunk, _ in chunks) == text  # nothing lost


class TestSplitting:
    def test_a_text_within_the_cap_is_one_chunk_itself(self):
        text = "one two three"
        assert split_text_for_embedding(text, 3, words) == [(text, 3)]

    def test_whole_paragraphs_are_packed_together(self):
        paragraphs = [_paragraph(40, i) for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = split_text_for_embedding(text, 100, words)
        _check_chunks(text, chunks, 100, words)
        # 40 + 40 fit, a third paragraph does not: 2 + 2 + 1 paragraphs.
        assert [c.count("\n\n") for c, _ in chunks] == [2, 2, 0]
        assert [words(c) for c, _ in chunks] == [80, 80, 40]

    def test_a_paragraph_too_long_is_cut_between_sentences(self):
        sentences = [f"{_paragraph(14, i).capitalize()} end{i}." for i in range(10)]
        long_paragraph = " ".join(sentences)
        text = f"{_paragraph(10, 99)}\n\n{long_paragraph}\n\n{_paragraph(10, 98)}"
        chunks = split_text_for_embedding(text, 50, words)
        _check_chunks(text, chunks, 50, words)
        inner = [c for c, _ in chunks[1:-1]]
        # Every chunk inside the long paragraph ends at a sentence end.
        assert all(c.rstrip().endswith(".") for c in inner)
        # ... holding whole sentences (15 words each): 3 per chunk.
        assert [words(c) for c in inner] == [45, 45, 45, 15]
        # The short paragraphs around it stay whole chunks of their own.
        assert chunks[0][0].strip() == _paragraph(10, 99)
        assert chunks[-1][0] == _paragraph(10, 98)

    def test_lines_are_sentence_boundaries(self):
        lines = [_paragraph(20, i) for i in range(6)]
        text = "\n".join(lines)
        chunks = split_text_for_embedding(text, 45, words)
        _check_chunks(text, chunks, 45, words)
        assert [words(c) for c, _ in chunks] == [40, 40, 40]

    def test_cjk_sentences(self):
        sentence = "今天天气很好我们去公园散步。"
        text = sentence * 10
        chunks = split_text_for_embedding(text, 30, chars)
        _check_chunks(text, chunks, 30, chars)
        assert all(c.endswith("。") for c, _ in chunks)
        assert [len(c) for c, _ in chunks] == [28, 28, 28, 28, 28]

    def test_a_sentence_too_long_is_cut_at_a_hard_boundary_on_whitespace(self):
        text = _paragraph(300, 7)  # no punctuation, no newline
        chunks = split_text_for_embedding(text, 50, words)
        _check_chunks(text, chunks, 50, words)
        assert len(chunks) == 6
        # Cut after whitespace, never inside a word.
        assert all(c.endswith(" ") for c, _ in chunks[:-1])

    def test_text_without_any_boundary(self):
        text = "x" * 1000
        chunks = split_text_for_embedding(text, 64, chars)
        _check_chunks(text, chunks, 64, chars)
        assert [len(c) for c, _ in chunks] == [64] * 15 + [40]

    def test_the_fallback_counter_splits_within_the_cap(self, fallback_counter):
        text = "\n\n".join(_paragraph(900, i) for i in range(4))  # ~ 7,000 tokens
        cap = embedding_chunk_cap(2048)
        chunks = split_text_for_embedding(text, cap, fallback_counter.count_text)
        _check_chunks(text, chunks, cap, fallback_counter.count_text)
        assert len(chunks) > 1

    def test_whitespace_only_chunks_are_left_out(self):
        text = "a b c d\n\n   \n\n\n\ne f g h"
        chunks = split_text_for_embedding(text, 4, words)
        assert [c.strip() for c, _ in chunks] == ["a b c d", "e f g h"]

    @pytest.mark.parametrize("seed", range(12))
    def test_random_documents(self, seed):
        rng = random.Random(seed)
        parts = []
        for _ in range(rng.randint(1, 12)):
            kind = rng.random()
            if kind < 0.4:
                parts.append(_paragraph(rng.randint(1, 120), rng.random()))
            elif kind < 0.8:
                parts.append(
                    " ".join(
                        f"{_paragraph(rng.randint(1, 30), rng.random())}."
                        for _ in range(rng.randint(1, 8))
                    )
                )
            else:
                parts.append(
                    "\n".join(
                        _paragraph(rng.randint(1, 25), rng.random()) for _ in range(4)
                    )
                )
        text = "\n\n".join(parts)
        cap = rng.choice([5, 17, 40, 64])
        for count in (words, chars):
            chunks = split_text_for_embedding(text, cap, count)
            assert all(count(c) <= cap for c, _ in chunks)
            joined = "".join(c for c, _ in chunks)
            # Only whitespace-only chunks may be dropped.
            assert joined.split() == text.split() if count is words else True
            assert "".join(joined.split()) == "".join(text.split())

    def test_a_zero_cap_is_refused(self):
        with pytest.raises(ValueError):
            split_text_for_embedding("text", 0, words)


# --------------------------------------------------------------------------- #
# Merging                                                                      #
# --------------------------------------------------------------------------- #


class TestMerging:
    def test_mean_then_normalize_then_pad(self):
        # Chunks of different norms: normalizing each chunk first would give
        # [0.707, 0.707]; the weighted mean first gives [1, 0.5] / |[1, 0.5]|.
        merged = merge_chunk_embeddings([[2.0, 0.0], [0.0, 1.0]], [1, 1])
        assert len(merged) == MAX_EMBEDDING_DIM
        assert merged[:2] == pytest.approx([2 / math.sqrt(5), 1 / math.sqrt(5)])
        assert merged[2:] == [0.0] * (MAX_EMBEDDING_DIM - 2)
        assert np.linalg.norm(merged) == pytest.approx(1.0)

    def test_weights_are_token_counts(self):
        merged = merge_chunk_embeddings([[1.0, 0.0], [0.0, 1.0]], [300, 100])
        assert merged[:2] == pytest.approx([3 / math.sqrt(10), 1 / math.sqrt(10)])

    def test_padding_is_last_so_different_sizes_are_refused(self):
        # Padding first would silently average a 3-dim and a 2-dim vector.
        with pytest.raises(ValueError, match="different dimensions"):
            merge_chunk_embeddings([[1.0, 0.0, 0.0], [0.0, 1.0]], [1, 1])

    def test_dims_and_norm_of_a_realistic_merge(self):
        rng = np.random.default_rng(3)
        vectors = [rng.normal(size=3072).tolist() for _ in range(4)]
        merged = merge_chunk_embeddings(vectors, [1800, 1800, 1800, 250])
        assert len(merged) == MAX_EMBEDDING_DIM
        assert np.linalg.norm(merged) == pytest.approx(1.0)
        assert not any(merged[3072:])
        expected = np.average(
            np.array(vectors), axis=0, weights=[1800, 1800, 1800, 250]
        )
        expected /= np.linalg.norm(expected)
        assert merged[:3072] == pytest.approx(expected.tolist())

    def test_a_zero_mean_stays_zero(self):
        merged = merge_chunk_embeddings([[1.0, -1.0], [-1.0, 1.0]], [1, 1])
        assert not any(merged) and len(merged) == MAX_EMBEDDING_DIM

    @pytest.mark.parametrize(
        "vectors, weights",
        [([], []), ([[1.0]], [1, 2]), ([[1.0], [2.0]], [1, 0]), ([[1.0]], [-1])],
    )
    def test_bad_input(self, vectors, weights):
        with pytest.raises(ValueError):
            merge_chunk_embeddings(vectors, weights)

    def test_pad(self):
        assert pad_embedding([1, 2]) == [1.0, 2.0] + [0.0] * (MAX_EMBEDDING_DIM - 2)
        assert len(pad_embedding([0.5] * MAX_EMBEDDING_DIM)) == MAX_EMBEDDING_DIM
        with pytest.raises(ValueError, match="more than"):
            pad_embedding([0.0] * (MAX_EMBEDDING_DIM + 1))


# --------------------------------------------------------------------------- #
# The guard                                                                    #
# --------------------------------------------------------------------------- #


def _guard(inner, limit=100):
    return GuardedEmbedding(
        inner, model_name="m", token_limits={"m": limit}, token_counter=WordCounter()
    )


@pytest.mark.asyncio
class TestGuardedEmbedding:
    async def test_a_text_within_the_cap_is_embedded_as_is_and_padded(self):
        inner = FakeModel()
        model = _guard(inner)
        text = _paragraph(88, 1)  # exactly the cap of a 100-token model
        vector = await model.get_text_embedding(text)
        assert inner.inputs == [text]
        # The provider's vector, untouched (not renormalized), padded.
        assert vector == pad_embedding(inner.vector(text))

    async def test_a_long_text_is_embedded_in_chunks_and_merged(self):
        inner = FakeModel()
        model = _guard(inner)
        text = "\n\n".join(_paragraph(60, i) for i in range(5))
        with counting_embedding_chunks() as counter:
            vector = await model.get_text_embedding(text)
        assert len(inner.inputs) == 5
        assert all(words(chunk) <= model.chunk_cap for chunk in inner.inputs)
        assert "".join(inner.inputs) == text
        expected = merge_chunk_embeddings(
            [inner.vector(c) for c in inner.inputs], [words(c) for c in inner.inputs]
        )
        assert vector == pytest.approx(expected)
        assert len(vector) == MAX_EMBEDDING_DIM
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert counter.as_metrics() == {
            "embedding_chunked_calls": 1,
            "embedding_chunks": 5,
        }

    async def test_only_chunked_calls_are_counted_in_every_enclosing_block(self):
        model = _guard(FakeModel())
        long_text = _paragraph(300, 2)
        with counting_embedding_chunks() as outer:
            await model.get_text_embedding("short")
            with counting_embedding_chunks() as inner_block:
                await model.get_text_embedding(long_text)
            # A task started inside the block counts too.
            await asyncio.ensure_future(model.get_text_embedding(long_text))
        assert (outer.chunked_calls, inner_block.chunked_calls) == (2, 1)
        await model.get_text_embedding(long_text)  # outside any block: not counted
        assert outer.chunked_calls == 2
        record_chunked_embedding_call(3)  # a no-op outside a block

    async def test_a_large_text_is_split_off_the_event_loop(self, monkeypatch):
        calls = []
        real = asyncio.to_thread

        async def spy(func, *args, **kwargs):
            calls.append(getattr(func, "__name__", func))
            return await real(func, *args, **kwargs)

        monkeypatch.setattr(guard.asyncio, "to_thread", spy)
        model = _guard(FakeModel())
        await model.get_text_embedding(_paragraph(400, 5))
        assert calls == ["split_text_for_embedding"]
        calls.clear()
        await model.get_text_embedding("short enough")
        assert calls == []

    async def test_attributes_are_the_wrapped_models(self):
        inner = FakeModel()
        model = _guard(inner)
        assert model.label == "fake-inner" and model.inner is inner
        with pytest.raises(AttributeError):
            model.no_such_attribute  # noqa: B018

    async def test_a_failed_chunked_call_is_counted_only_when_it_succeeds(self):
        """A chunk failing mid-way records nothing; the successful retry of
        the whole call is one chunked call, not two."""

        class FailsOnce(FakeModel):
            failed = False

            async def get_text_embedding(self, text):
                if len(self.inputs) == 1 and not self.failed:
                    self.failed = True
                    raise RuntimeError("provider hiccup")
                return await super().get_text_embedding(text)

        inner = FailsOnce()
        model = _guard(inner)
        text = "\n\n".join(_paragraph(60, i) for i in range(3))
        with counting_embedding_chunks() as counter:
            with pytest.raises(RuntimeError, match="hiccup"):
                await model.get_text_embedding(text)
            assert counter.chunked_calls == 0 and counter.chunks == 0
            await model.get_text_embedding(text)
        assert counter.as_metrics() == {
            "embedding_chunked_calls": 1,
            "embedding_chunks": 3,
        }

    async def test_an_over_cap_text_with_one_real_chunk_sends_that_chunk(self):
        """The whitespace chunks of an over-cap text are dropped; the
        remaining chunk is sent, never the whole text."""

        class CharCounter:
            def count_text(self, text):
                return chars(text)

            async def acount_text(self, text):
                return chars(text)

        inner = FakeModel()
        model = GuardedEmbedding(
            inner, model_name="m", token_limits={"m": 100}, token_counter=CharCounter()
        )
        text = "deploy the queue" + " " * 400
        assert chars(text) > model.chunk_cap
        with counting_embedding_chunks() as counter:
            vector = await model.get_text_embedding(text)
        assert len(inner.inputs) == 1
        (sent,) = inner.inputs
        assert sent != text and chars(sent) <= model.chunk_cap
        assert sent.startswith("deploy the queue")
        assert vector == pad_embedding(inner.vector(sent))
        assert counter.as_metrics() == {
            "embedding_chunked_calls": 1,
            "embedding_chunks": 1,
        }

    @pytest.mark.parametrize(
        "max_chunks, head, tail", [(4, 2, 2), (5, 3, 2), (1, 1, 0)]
    )
    async def test_a_huge_text_is_its_head_and_tail_chunks(
        self, monkeypatch, max_chunks, head, tail
    ):
        """At most MIRIX_EMBEDDING_MAX_CHUNKS provider calls per text."""
        monkeypatch.setattr(constants, "EMBEDDING_MAX_CHUNKS", max_chunks)
        warnings = []
        monkeypatch.setattr(
            guard.logger, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        inner = FakeModel()
        model = _guard(inner)
        paragraphs = [_paragraph(60, i) for i in range(10)]
        text = "\n\n".join(paragraphs)
        planned = await model.plan_chunks(text)
        assert len(planned) == 10
        kept = planned[:head] + planned[len(planned) - tail :]
        with counting_embedding_chunks() as counter:
            vector = await model.get_text_embedding(text)
        assert inner.inputs == [chunk for chunk, _ in kept]
        assert vector == pytest.approx(
            merge_chunk_embeddings(
                [inner.vector(c) for c, _ in kept], [t for _, t in kept]
            )
        )
        assert counter.as_metrics() == {
            "embedding_chunked_calls": 1,
            "embedding_chunks": max_chunks,
        }
        assert len(warnings) == 1
        assert "needs 10 chunks" in warnings[0]
        assert f"MIRIX_EMBEDDING_MAX_CHUNKS={max_chunks}" in warnings[0]

    async def test_a_provider_vector_too_wide_is_an_error(self):
        model = _guard(FakeModel(dim=MAX_EMBEDDING_DIM + 8))
        with pytest.raises(ValueError, match="more than"):
            await model.get_text_embedding("short")


class TestCounter:
    def test_metrics(self):
        counter = EmbeddingChunkCounter()
        counter.record(3)
        counter.record(2)
        assert counter.as_metrics() == {
            "embedding_chunked_calls": 2,
            "embedding_chunks": 5,
        }


# --------------------------------------------------------------------------- #
# The factory                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestFactory:
    @pytest.mark.parametrize(
        "config",
        [
            dict(
                embedding_endpoint_type="openrouter",
                embedding_endpoint="http://127.0.0.1:9/v1",
                embedding_model="google/gemini-embedding-001",
                embedding_dim=3072,
                api_key="sk-or-fake",
            ),
            dict(
                embedding_endpoint_type="hugging-face",
                embedding_endpoint="http://127.0.0.1:9/v1",
                embedding_model="BAAI/bge-large-en-v1.5",
                embedding_dim=1024,
            ),
            dict(
                embedding_endpoint_type="ollama",
                embedding_endpoint="http://127.0.0.1:9",
                embedding_model="mxbai-embed-large",
                embedding_dim=1024,
            ),
            dict(
                embedding_endpoint_type="openai",
                embedding_endpoint="http://127.0.0.1:9/v1",
                embedding_model="text-embedding-3-small",
                embedding_dim=1536,
                api_key="fake",
            ),
        ],
        ids=lambda c: c["embedding_endpoint_type"],
    )
    async def test_every_model_is_guarded(self, config):
        from mirix.embeddings import embedding_model

        model = await embedding_model(EmbeddingConfig(**config))
        assert isinstance(model, GuardedEmbedding)
        assert model.model_name == config["embedding_model"]
        assert not isinstance(model.inner, GuardedEmbedding)

    async def test_end_to_end_against_the_fake_server(
        self, fake_llm_server, fallback_counter, monkeypatch
    ):
        # gemini-embedding-001 through OpenRouter (what the benchmark uses):
        # a ~4,000-token text reaches the provider as chunks of <= 1,800.
        from mirix.embeddings import embedding_model

        monkeypatch.setattr(model_settings, "openrouter_api_key", None)
        model = await embedding_model(
            EmbeddingConfig(
                embedding_endpoint_type="openrouter",
                embedding_endpoint=fake_llm_server.base_url,
                embedding_model="google/gemini-embedding-001",
                embedding_dim=3072,
                api_key="sk-or-fake",
            )
        )
        text = "\n\n".join(_paragraph(700, i) for i in range(4))
        vector = await model.get_text_embedding(text)
        inputs = [r.json["input"] for r in fake_llm_server.embedding_requests()]
        assert len(inputs) >= 3
        assert all(fallback_counter.count_text(i) <= 1800 for i in inputs)
        assert "".join(inputs) == text
        expected = merge_chunk_embeddings(
            [fake_llm_server.embed(i) for i in inputs],
            [fallback_counter.count_text(i) for i in inputs],
        )
        assert vector == pytest.approx(expected, abs=1e-6)
        assert len(vector) == MAX_EMBEDDING_DIM
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        # A short text: one request, the provider's vector, padded.
        vector = await model.get_text_embedding("put a mug in the sink")
        assert vector[:3072] == pytest.approx(
            fake_llm_server.embed("put a mug in the sink")
        )
        assert not any(vector[3072:])

    def test_the_three_dead_helpers_are_gone(self):
        import mirix.embeddings as embeddings

        for name in ("parse_and_chunk_text", "truncate_text", "check_and_split_text"):
            assert not hasattr(embeddings, name)
            for path in (REPO / "mirix").rglob("*.py"):
                assert f"{name}(" not in path.read_text(), path


# --------------------------------------------------------------------------- #
# HTTP status                                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def quick_retries(monkeypatch):
    """The embedding retry policy (3 backoff retries) with near-zero waits:
    every embedding call's single retry layer."""
    import dataclasses

    from mirix.services import retry_policy

    policy = retry_policy.RETRY_POLICIES[retry_policy.ErrorKind.EMBEDDING]
    monkeypatch.setitem(
        retry_policy.RETRY_POLICIES,
        retry_policy.ErrorKind.EMBEDDING,
        dataclasses.replace(policy, base_delay=0.001, max_delay=0.001),
    )


def _openrouter_config(fake_llm_server) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_endpoint_type="openrouter",
        embedding_endpoint=fake_llm_server.base_url,
        embedding_model="google/gemini-embedding-001",
        embedding_dim=3072,
        api_key="sk-or-secret-key",
    )


async def _openrouter(fake_llm_server):
    from mirix.embeddings import embedding_model

    return await embedding_model(_openrouter_config(fake_llm_server))


@pytest.mark.asyncio
class TestHTTPStatus:
    @pytest.mark.parametrize(
        "status, kind", [(400, "other"), (401, "config"), (404, "config")]
    )
    async def test_a_non_200_answer_raises(
        self, fake_llm_server, quick_retries, status, kind
    ):
        from mirix.services.retry_policy import classify_error

        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(
            Fault.http_error(status, "model says no"), route="embeddings"
        )
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await model.get_text_embedding("hello")
        assert raised.value.response.status_code == status
        assert f"HTTP {status}" in str(raised.value)
        assert "model says no" in str(raised.value)
        assert "sk-or-secret-key" not in str(raised.value)
        assert len(fake_llm_server.embedding_requests()) == 1  # not retried
        assert classify_error(raised.value).value == kind

    async def test_a_transient_error_is_retried(self, fake_llm_server, quick_retries):
        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(Fault.http_error(503), route="embeddings")
        vector = await model.get_text_embedding("hello")
        assert vector[:3072] == pytest.approx(fake_llm_server.embed("hello"))
        assert len(fake_llm_server.embedding_requests()) == 2

    async def test_a_dropped_connection_is_retried_too(
        self, fake_llm_server, quick_retries
    ):
        # The one retry layer retries a network failure, not only 429/5xx.
        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(Fault.drop_connection(), route="embeddings")
        vector = await model.get_text_embedding("hello")
        assert vector[:3072] == pytest.approx(fake_llm_server.embed("hello"))
        assert len(fake_llm_server.embedding_requests()) == 2

    async def test_a_persistent_outage_is_infrastructure(
        self, fake_llm_server, quick_retries
    ):
        from mirix.services.retry_policy import ErrorKind, classify_error

        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(
            Fault.http_error(503), route="embeddings", times=10
        )
        with pytest.raises(Exception) as raised:
            await model.get_text_embedding("hello")
        assert classify_error(raised.value) is ErrorKind.INFRA
        assert (
            classify_error(raised.value, operation="embedding") is ErrorKind.EMBEDDING
        )

    async def test_the_other_raw_http_embedders_check_it_too(
        self, fake_llm_server, quick_retries
    ):
        from mirix.embeddings import EmbeddingEndpoint, OllamaEmbeddings

        endpoint = EmbeddingEndpoint(
            model="bge", base_url=fake_llm_server.base_url, user="u"
        )
        fake_llm_server.inject_fault(
            Fault.http_error(422, "bad input"), route="embeddings"
        )
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 422"):
            await endpoint.get_text_embedding("hello")

        ollama = OllamaEmbeddings(
            model="mxbai", base_url=fake_llm_server.url, ollama_additional_kwargs={}
        )
        # The fake serves no /api/embeddings: its 404 error body used to be
        # read as the answer (KeyError on "embedding").
        with pytest.raises(httpx.HTTPStatusError, match="HTTP 404"):
            await ollama.get_text_embedding("hello")


@pytest.mark.asyncio
class TestOneRetryLayer:
    """Under ``retry_call(operation="embedding")`` the provider model makes
    one call per attempt — its legacy retry loop does not multiply the policy's."""

    async def test_a_persistent_429_costs_one_call_per_attempt(
        self, fake_llm_server, quick_retries, monkeypatch
    ):
        import mirix.services.procedural_memory_manager as pmm
        from mirix.services.retry_policy import RETRY_POLICIES, ErrorKind

        monkeypatch.setattr(pmm, "BUILD_EMBEDDINGS_FOR_MEMORY", True)
        waits = []

        async def _no_wait(delay):
            waits.append(delay)

        manager = pmm.ProceduralMemoryManager(session_maker=object(), sleep=_no_wait)
        fake_llm_server.inject_fault(
            Fault.http_error(429, "slow down"), route="embeddings", times=100
        )
        with pytest.raises(httpx.HTTPStatusError):
            await manager._embed_query("hello", _openrouter_config(fake_llm_server))
        attempts = RETRY_POLICIES[ErrorKind.EMBEDDING].in_run_retries + 1
        assert attempts == 4
        # Not 4 x (1 + llm_retry_limit): the policy's backoff is the only retry.
        assert len(fake_llm_server.embedding_requests()) == attempts
        assert len(waits) == attempts - 1

    async def test_a_direct_call_gets_the_same_single_policy(
        self, fake_llm_server, quick_retries
    ):
        from mirix.services.retry_policy import embedding_retries_owned, retry_call

        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(
            Fault.http_error(503), route="embeddings", times=100
        )
        with pytest.raises(httpx.HTTPStatusError):
            await model.get_text_embedding("hello")
        # Outside retry_call the model call IS the retry layer: 1 + 3 attempts.
        assert len(fake_llm_server.embedding_requests()) == 4

        seen = []

        async def _probe():
            seen.append(embedding_retries_owned())
            return 1

        assert await retry_call(_probe, operation="embedding") == 1
        assert await retry_call(_probe, stage="skill_write") == 1
        assert seen == [True, False]
        assert embedding_retries_owned() is False  # reset after the call


def _sdk_config(endpoint_type: str, fake_llm_server, **extra) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_endpoint_type=endpoint_type,
        embedding_endpoint=fake_llm_server.base_url,
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        api_key="sk-fake",
        **extra,
    )


class _GenAIEmbedStub:
    """Stands in for ``google.genai``'s ``aio.models.embed_content``: raises
    the SDK's own 429 error ``failures`` times, then answers."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    async def __call__(self, *, model, contents, config=None):
        from types import SimpleNamespace

        from google.genai import errors

        self.calls += 1
        if self.calls <= self.failures:
            raise errors.ClientError(
                429,
                {
                    "error": {
                        "code": 429,
                        "message": "quota",
                        "status": "RESOURCE_EXHAUSTED",
                    }
                },
                None,
            )
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.6, 0.8]) for _ in contents]
        )


async def _google_model(monkeypatch, stub):
    from mirix.embeddings import embedding_model

    model = await embedding_model(
        EmbeddingConfig(
            embedding_endpoint_type="google_ai",
            embedding_endpoint=None,
            embedding_model="gemini-embedding-001",
            embedding_dim=3072,
            api_key="fake-gemini-key",
        )
    )
    llama_model = model.inner._model
    monkeypatch.setattr(llama_model._client.aio.models, "embed_content", stub)
    return model


@pytest.mark.asyncio
class TestNoSDKRetries:
    """The SDK-backed embedding types — ``openai`` (llama_index
    ``OpenAIEmbedding``: a tenacity loop of 10 attempts within 60 s over an
    ``AsyncOpenAI`` with 10 retries of its own), ``google_ai`` (llama_index
    ``GoogleGenAIEmbedding``: 3 attempts), ``openai`` with an auth provider
    and ``azure`` (``AsyncOpenAI`` / ``AsyncAzureOpenAI``: 2 retries each) —
    make ONE provider call per attempt, so the embedding retry policy (1 + 3 attempts)
    is the only retry, inside ``retry_call`` or not."""

    async def _manager(self, monkeypatch):
        import mirix.services.procedural_memory_manager as pmm

        monkeypatch.setattr(pmm, "BUILD_EMBEDDINGS_FOR_MEMORY", True)
        waits = []

        async def _no_wait(delay):
            waits.append(delay)

        return pmm.ProceduralMemoryManager(
            session_maker=object(), sleep=_no_wait
        ), waits

    async def test_openai_under_a_persistent_429_costs_four_calls(
        self, fake_llm_server, quick_retries, monkeypatch
    ):
        import openai

        from mirix.embeddings import embedding_model

        fake_llm_server.inject_fault(
            Fault.http_error(429, "slow down"), route="embeddings", times=1000
        )
        manager, waits = await self._manager(monkeypatch)
        config = _sdk_config("openai", fake_llm_server)
        with pytest.raises(openai.RateLimitError):
            await manager._embed_query("hello", config)
        # Not 4 x tenacity 10 x SDK 11: the SDK and tenacity retries are off.
        assert len(fake_llm_server.embedding_requests()) == 4
        assert len(waits) == 3

        # A direct call (episodic/semantic memory, /memory/search's query
        # vector): the same single policy, 4 calls.
        model = await embedding_model(config)
        with pytest.raises(openai.RateLimitError):
            await model.get_text_embedding("hello")
        assert len(fake_llm_server.embedding_requests()) == 8

    async def test_openai_retries_a_transient_failure_once_per_attempt(
        self, fake_llm_server, quick_retries
    ):
        from mirix.embeddings import embedding_model

        fake_llm_server.inject_fault(Fault.http_error(503), route="embeddings")
        model = await embedding_model(_sdk_config("openai", fake_llm_server))
        vector = await model.get_text_embedding("hello")
        assert len(vector) == MAX_EMBEDDING_DIM and any(vector)
        assert len(fake_llm_server.embedding_requests()) == 2

    async def test_openai_is_built_without_retries(self, fake_llm_server):
        from mirix.embeddings import embedding_model

        model = await embedding_model(_sdk_config("openai", fake_llm_server))
        llama_model = model.inner._model
        assert llama_model.max_retries == 0
        assert llama_model._get_aclient().max_retries == 0

    async def test_google_under_a_persistent_429_costs_four_calls(
        self, quick_retries, monkeypatch
    ):
        from google.genai import errors

        stub = _GenAIEmbedStub(failures=1000)
        model = await _google_model(monkeypatch, stub)
        assert model.inner._model.retries == 1
        manager, waits = await self._manager(monkeypatch)
        # _embed_query builds its own model: hand it this one.
        import mirix.services.procedural_memory_manager as pmm

        async def _this_model(config, user_id=None):
            return model

        monkeypatch.setattr(pmm, "embedding_model", _this_model)
        with pytest.raises(errors.ClientError):
            await manager._embed_query("hello", object())
        # Was 12 (4 x llama_index's 3 attempts).
        assert stub.calls == 4
        assert len(waits) == 3

        stub.calls = 0
        with pytest.raises(errors.ClientError):
            await model.get_text_embedding("hello")
        assert stub.calls == 4

    async def test_google_recovers_on_the_next_attempt(
        self, quick_retries, monkeypatch
    ):
        stub = _GenAIEmbedStub(failures=1)
        model = await _google_model(monkeypatch, stub)
        vector = await model.get_text_embedding("hello")
        assert stub.calls == 2
        assert vector[:2] == pytest.approx([0.6, 0.8])
        assert not any(vector[2:])

    async def test_custom_auth_openai_makes_one_call_per_attempt(
        self, fake_llm_server, quick_retries, monkeypatch
    ):
        import openai

        from mirix.embeddings import embedding_model
        from mirix.llm_api.auth_provider import (
            AuthProvider,
            register_auth_provider,
            unregister_auth_provider,
        )

        class _Auth(AuthProvider):
            def get_auth_headers(self):
                return {"Authorization": "Bearer from-the-provider"}

        register_auth_provider("r23-test-auth", _Auth())
        try:
            fake_llm_server.inject_fault(
                Fault.http_error(429, "slow down"), route="embeddings", times=1000
            )
            model = await embedding_model(
                _sdk_config("openai", fake_llm_server, auth_provider="r23-test-auth")
            )
            with pytest.raises(openai.RateLimitError):
                await model.get_text_embedding("hello")
            # Was 12 (4 x the SDK's 3) inside retry_call, 3 outside it.
            requests = fake_llm_server.embedding_requests()
            assert len(requests) == 4
            assert all(
                r.headers.get("authorization") == "Bearer from-the-provider"
                for r in requests
            )
        finally:
            unregister_auth_provider("r23-test-auth")

    async def test_azure_is_built_without_sdk_retries(self, monkeypatch):
        from mirix.embeddings import AzureOpenAIEmbedding

        model = AzureOpenAIEmbedding(
            api_endpoint="http://127.0.0.1:9",
            api_key="fake",
            api_version="2024-02-01",
            model="text-embedding-3-small",
        )
        assert model.client.max_retries == 0

    async def test_no_legacy_retry_loop_is_left(self):
        import mirix.embeddings as embeddings

        source = Path(embeddings.__file__).read_text()
        assert "retry_with_exponential_backoff" not in source
        assert "llm_retry_limit" not in source


@pytest.mark.asyncio
class TestTimeout:
    """Every embedding type waits ``MIRIX_EMBEDDING_TIMEOUT_SECONDS``
    per attempt (default 60 s, the connect at most 5 s). The eval sets it above
    its gateway's 600 s deadline, so MIRIX receives the gateway's answer instead
    of abandoning a held call and sending it again."""

    async def test_the_setting_and_its_default(self, monkeypatch):
        from mirix import constants
        from mirix.embeddings import embedding_http_timeout, embedding_timeout_seconds
        from mirix.settings import settings

        assert constants._positive_float_setting("X", None, 60.0) == 60.0
        monkeypatch.setattr(constants, "EMBEDDING_TIMEOUT_SECONDS", 630.0)
        assert embedding_timeout_seconds() == 630.0
        assert settings.embedding_timeout_seconds == 630.0
        timeout = embedding_http_timeout()
        assert (timeout.read, timeout.write, timeout.pool) == (630.0, 630.0, 630.0)
        assert timeout.connect == 5.0
        assert embedding_http_timeout(2.0).connect == 2.0

    async def test_an_attempt_past_the_timeout_is_abandoned_and_retried(
        self, fake_llm_server, quick_retries, monkeypatch
    ):
        from mirix import constants

        monkeypatch.setattr(constants, "EMBEDDING_TIMEOUT_SECONDS", 0.3)
        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(Fault.slow(2.0), route="embeddings")
        vector = await model.get_text_embedding("hello")
        assert vector[:3072] == pytest.approx(fake_llm_server.embed("hello"))
        # The held attempt was given up after 0.3 s; the retry answered.
        assert [r.outcome for r in fake_llm_server.embedding_requests()] == [
            "fault:slow:2.0",
            "embedding",
        ]

    async def test_a_slow_answer_within_the_timeout_is_waited_for(
        self, fake_llm_server, quick_retries, monkeypatch
    ):
        from mirix import constants

        monkeypatch.setattr(constants, "EMBEDDING_TIMEOUT_SECONDS", 5.0)
        model = await _openrouter(fake_llm_server)
        fake_llm_server.inject_fault(Fault.slow(0.6), route="embeddings")
        await model.get_text_embedding("hello")
        # One exchange (the fake logs its delay and then its answer), no retry.
        assert [r.outcome for r in fake_llm_server.embedding_requests()] == [
            "fault:slow:0.6",
            "embedding",
        ]

    async def test_every_type_is_built_with_it(self, fake_llm_server, monkeypatch):
        from mirix import constants
        from mirix.embeddings import (
            AzureOpenAIEmbedding,
            EmbeddingEndpoint,
            embedding_model,
        )

        monkeypatch.setattr(constants, "EMBEDDING_TIMEOUT_SECONDS", 630.0)
        openai_model = await embedding_model(_sdk_config("openai", fake_llm_server))
        assert openai_model.inner._model.timeout == 630.0
        google = await embedding_model(
            EmbeddingConfig(
                embedding_endpoint_type="google_ai",
                embedding_endpoint=None,
                embedding_model="gemini-embedding-001",
                embedding_dim=3072,
                api_key="fake-gemini-key",
            )
        )
        options = google.inner._model._client._api_client._http_options
        assert options.timeout == 630_000  # milliseconds, on the genai client
        azure = AzureOpenAIEmbedding(
            api_endpoint="http://127.0.0.1:9",
            api_key="fake",
            api_version="2024-02-01",
            model="text-embedding-3-small",
        )
        assert azure.client.timeout.read == 630.0
        assert azure.client.timeout.connect == 5.0
        # The raw-HTTP types read it at call time (None: the setting).
        assert (
            EmbeddingEndpoint(model="bge", base_url="http://x", user="u")._timeout
            is None
        )
