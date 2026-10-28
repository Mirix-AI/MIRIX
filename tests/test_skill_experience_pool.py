"""The experience pool's pure decisions and embedding calls.

No database and no model provider: ``decide_history`` is pure CPU, and
``embed_pool_texts`` runs with an injected ``embed_one``.

* the threshold keeps a pool experience only when its max cosine to a new
  experience reaches the minimum; the score decays by 0.5 per ignored push;
  a recurrence (cosine >= the recurrence threshold) resets the decay;
* the top-K cap and the (score, created_at, id) order are deterministic
  under any input order; zero, NaN and oversize vectors are skipped and
  counted; padding (3,072 vs 4,096 dimensions) keeps the cosine;
* the cosine log is bounded and sorted, with quantiles;
* the text of a pool vector is the title and content, never the evidence;
* ``embed_pool_texts`` returns what succeeded plus the first failure (it
  never raises for an embedding failure), respects its concurrency bound,
  retries a transient failure, and lets cancellation through.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.services.skill_experience_pool import (
    PoolCandidate,
    decide_history,
    embed_pool_texts,
    embedding_model_key,
    experience_pool_text,
)

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _unit(dim: int, index: int) -> list:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _at_cosine(cos: float, dim: int = 8) -> list:
    """A unit vector whose cosine to e0 is ``cos``."""
    v = [0.0] * dim
    v[0] = cos
    v[1] = math.sqrt(max(0.0, 1.0 - cos * cos))
    return v


def _candidate(cid, vector, *, ignored=0, seconds=0, **overrides) -> PoolCandidate:
    fields = dict(
        id=cid,
        experience_type="worth_learning",
        title=f"title {cid}",
        content=f"content {cid}",
        importance=0.5,
        credibility=0.5,
        recurrence_count=1,
        pushed_count=0,
        ignored_pushes=ignored,
        created_at=T0 + dt.timedelta(seconds=seconds),
        vector=vector,
        vector_model="m",
    )
    fields.update(overrides)
    return PoolCandidate(**fields)


def _decide(new, candidates, **overrides):
    kwargs = dict(
        min_similarity=0.60,
        recurrence_similarity=0.80,
        top_k=10,
        decay=0.5,
        log_limit=50,
    )
    kwargs.update(overrides)
    return decide_history(new, candidates, **kwargs)


class TestDecideHistory:
    def test_the_threshold_keeps_only_similar_enough_lessons(self):
        new = [_unit(8, 0)]
        decision = _decide(
            new,
            [
                _candidate("a", _at_cosine(0.59)),
                _candidate("b", _at_cosine(0.60)),
                _candidate("c", _at_cosine(0.70)),
            ],
        )
        assert [r.candidate.id for r in decision.ranked] == ["c", "b"]
        assert decision.candidates == 3 and decision.above_min == 2
        assert decision.recurred_ids == []

    def test_the_score_decays_by_half_per_ignored_push(self):
        new = [_unit(8, 0)]
        decision = _decide(
            new,
            [
                # 0.79 twice ignored -> 0.1975; 0.65 fresh -> 0.65.
                _candidate("ignored-twice", _at_cosine(0.79), ignored=2),
                _candidate("fresh", _at_cosine(0.65)),
            ],
        )
        assert [r.candidate.id for r in decision.ranked] == ["fresh", "ignored-twice"]
        scores = {r.candidate.id: r.score for r in decision.ranked}
        assert scores["ignored-twice"] == pytest.approx(0.79 * 0.25)
        assert scores["fresh"] == pytest.approx(0.65)

    def test_a_recurrence_resets_the_decay_so_the_revived_lesson_ranks_first(self):
        new = [_unit(8, 0)]
        decision = _decide(
            new,
            [
                _candidate("revived", _at_cosine(0.85), ignored=2),
                _candidate("fresh", _at_cosine(0.79)),
            ],
        )
        assert [r.candidate.id for r in decision.ranked] == ["revived", "fresh"]
        revived = decision.ranked[0]
        assert revived.recurred and revived.score == pytest.approx(0.85)
        assert decision.recurred_ids == ["revived"]

    def test_the_max_cosine_over_the_new_experiences_counts(self):
        new = [_unit(8, 3), _unit(8, 0)]
        decision = _decide(new, [_candidate("a", _at_cosine(0.9))])
        assert decision.ranked[0].max_cosine == pytest.approx(0.9)

    def test_top_k_and_a_deterministic_order_under_any_input_order(self):
        new = [_unit(8, 0)]
        # Ties in score are broken by (created_at, id).
        candidates = [
            _candidate(f"id-{i:02d}", _at_cosine(0.7), seconds=i % 3) for i in range(12)
        ]
        first = _decide(new, candidates, top_k=5)
        assert len(first.ranked) == 5
        expected = [
            c.id for c in sorted(candidates, key=lambda c: (c.created_at, c.id))
        ][:5]
        assert [r.candidate.id for r in first.ranked] == expected
        rng = random.Random(7)
        for _ in range(5):
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            again = _decide(new, shuffled, top_k=5)
            assert [r.candidate.id for r in again.ranked] == expected
            assert again.cosine_log == first.cosine_log
            assert again.recurred_ids == first.recurred_ids

    def test_unusable_vectors_are_skipped_and_counted(self):
        new = [_unit(8, 0)]
        decision = _decide(
            new,
            [
                _candidate("zero", [0.0] * 8),
                _candidate("nan", [float("nan")] + [0.0] * 7),
                _candidate("inf", [float("inf")] + [0.0] * 7),
                _candidate("oversize", [1.0] * (MAX_EMBEDDING_DIM + 1)),
                _candidate("missing", None),
                _candidate("empty", []),
                _candidate("ok", _at_cosine(0.9)),
            ],
        )
        assert decision.unusable_vectors == 6
        assert decision.candidates == 1
        assert [r.candidate.id for r in decision.ranked] == ["ok"]

    def test_an_unusable_new_vector_is_ignored(self):
        decision = _decide([[0.0] * 8, _unit(8, 0)], [_candidate("a", _at_cosine(0.9))])
        assert [r.candidate.id for r in decision.ranked] == ["a"]

    @pytest.mark.parametrize(
        "new, candidates",
        [
            ([], [_candidate("a", _at_cosine(0.9))]),
            ([[0.0] * 8], [_candidate("a", _at_cosine(0.9))]),
            ([_unit(8, 0)], []),
        ],
    )
    def test_no_usable_new_vector_or_no_candidate_is_an_empty_decision(
        self, new, candidates
    ):
        decision = _decide(new, candidates)
        assert decision.ranked == [] and decision.recurred_ids == []
        assert decision.cosine_log == [] and decision.quantiles == {}
        assert decision.candidates == 0

    def test_the_cosine_log_is_bounded_sorted_and_has_quantiles(self):
        new = [_unit(8, 0)]
        cosines = [0.1 * i for i in range(10)]
        candidates = [
            _candidate(f"c{i}", _at_cosine(c), seconds=i) for i, c in enumerate(cosines)
        ]
        decision = _decide(new, candidates, log_limit=4)
        assert len(decision.cosine_log) == 4
        logged = [e["max_cosine"] for e in decision.cosine_log]
        assert logged == sorted(logged, reverse=True)
        assert logged[0] == pytest.approx(0.9)
        assert set(decision.cosine_log[0]) == {
            "id",
            "max_cosine",
            "score",
            "ignored_pushes",
            "recurred",
            "eligible",
        }
        assert decision.cosine_log[0]["recurred"] is True
        assert decision.cosine_log[0]["eligible"] is True
        assert decision.quantiles["max"] == pytest.approx(0.9)
        assert decision.quantiles["p50"] == pytest.approx(
            float(np.quantile(cosines, 0.5)), abs=1e-4
        )
        assert decision.quantiles["p90"] == pytest.approx(
            float(np.quantile(cosines, 0.9)), abs=1e-4
        )

    def test_padding_does_not_change_the_cosine(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=3072)
        b = rng.normal(size=3072)
        expected = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        padded = np.pad(b, (0, MAX_EMBEDDING_DIM - 3072)).tolist()
        for candidate_vector in (b.tolist(), padded):
            decision = _decide(
                [a.tolist()],
                [_candidate("x", candidate_vector)],
                min_similarity=-1.0,
            )
            assert decision.ranked[0].max_cosine == pytest.approx(expected, abs=1e-9)

    def test_each_new_experience_names_the_pool_experiences_it_recurred(self):
        new = [_unit(8, 0), _unit(8, 2), [0.0] * 8]
        decision = _decide(
            new,
            [
                _candidate("late", _at_cosine(0.9), seconds=2),  # recurs n1
                _candidate("early", _at_cosine(0.85), seconds=1),  # recurs n1
                _candidate("near", _at_cosine(0.7)),  # shown, never recurred
                _candidate("other", _unit(8, 2), seconds=3),  # recurs n2
            ],
            new_ids=["n1", "n2", "n3"],
        )
        # In (created_at, id) order; n3's vector is unusable, n2 recurs "other".
        assert decision.recurred_by == {"n1": ["early", "late"], "n2": ["other"]}
        assert decision.recurred_ids == ["early", "late", "other"]
        # Without ids the mapping is not asked for.
        assert _decide(new, [_candidate("late", _at_cosine(0.9))]).recurred_by == {}
        with pytest.raises(ValueError):
            _decide(new, [], new_ids=["n1"])

    def test_a_naive_created_at_sorts_like_utc(self):
        """SQLite returns naive datetimes: they order like UTC ones."""
        new = [_unit(8, 0)]
        candidates = [
            _candidate(
                "b", _at_cosine(0.7), created_at=dt.datetime(2026, 1, 1, 0, 0, 2)
            ),
            _candidate(
                "a", _at_cosine(0.7), created_at=dt.datetime(2026, 1, 1, 0, 0, 1)
            ),
        ]
        decision = _decide(new, candidates)
        assert [r.candidate.id for r in decision.ranked] == ["a", "b"]


class TestText:
    def test_the_pool_text_is_the_title_and_content_never_the_evidence(self):
        experience = SimpleNamespace(
            title="  Read   the\nlabelled field ",
            content="  Copy the value verbatim.  ",
            evidence='{"quote": "SECRET SESSION QUOTE"}',
        )
        text = experience_pool_text(experience)
        assert text == "Read the labelled field\n\nCopy the value verbatim."
        assert "SECRET" not in text
        assert experience_pool_text({"title": "t", "content": ""}) == "t"

    def test_the_embedding_model_key(self):
        assert embedding_model_key(SimpleNamespace(embedding_model=" m-1 ")) == "m-1"
        assert embedding_model_key(SimpleNamespace(embedding_model="")) is None
        assert embedding_model_key(None) is None

    def test_the_key_tells_one_model_at_two_widths_apart(self):
        """The same model at another dimension gives vectors whose zero-padded
        cosine means nothing: they must be embedded again, never compared."""
        narrow = SimpleNamespace(
            embedding_model="text-embedding-3-large", embedding_dim=1024
        )
        wide = SimpleNamespace(
            embedding_model="text-embedding-3-large", embedding_dim=3072
        )
        assert embedding_model_key(narrow) == "text-embedding-3-large@1024"
        assert embedding_model_key(wide) == "text-embedding-3-large@3072"
        assert embedding_model_key(narrow) != embedding_model_key(wide)
        for dim in (None, 0, "x", True):
            assert (
                embedding_model_key(
                    SimpleNamespace(embedding_model="m", embedding_dim=dim)
                )
                == "m"
            )


@pytest.mark.asyncio
class TestEmbedPoolTexts:
    async def test_every_text_is_embedded_and_padded(self):
        async def embed_one(text):
            return [float(len(text)), 1.0, 0.0]

        vectors, errors = await embed_pool_texts(
            {"a": "xx", "b": "yyy"}, None, concurrency=2, embed_one=embed_one
        )
        assert errors == {}
        assert set(vectors) == {"a", "b"}
        assert len(vectors["a"]) == MAX_EMBEDDING_DIM
        assert vectors["b"][:3] == [3.0, 1.0, 0.0]

    async def test_a_failure_returns_the_rest_and_each_error_by_id(self):
        async def embed_one(text):
            if text.startswith("bad"):
                raise ValueError(f"refused {text}")
            return [1.0, 0.0]

        vectors, errors = await embed_pool_texts(
            {"a": "ok-a", "b": "bad-b", "c": "bad-c", "d": "ok-d"},
            None,
            concurrency=3,
            embed_one=embed_one,
        )
        assert set(vectors) == {"a", "d"}
        assert list(errors) == ["b", "c"]  # in the texts' order
        assert isinstance(errors["b"], ValueError) and "bad-b" in str(errors["b"])
        assert "bad-c" in str(errors["c"])

    async def test_no_vector_counts_as_a_failure(self):
        async def embed_one(text):
            return None

        vectors, errors = await embed_pool_texts(
            {"a": "x"}, None, concurrency=1, embed_one=embed_one
        )
        assert vectors == {} and "no vector" in str(errors["a"])

    async def test_a_vector_wider_than_the_column_is_a_failure(self):
        async def embed_one(text):
            return [1.0] * (MAX_EMBEDDING_DIM + 1)

        vectors, errors = await embed_pool_texts(
            {"a": "x"}, None, concurrency=1, embed_one=embed_one
        )
        assert vectors == {} and set(errors) == {"a"}

    async def test_a_transient_failure_is_retried_by_the_m35_layer(self):
        import httpx

        calls = []

        async def embed_one(text):
            calls.append(text)
            if len(calls) == 1:
                raise httpx.ConnectError("connection refused")
            return [1.0]

        async def no_sleep(_):
            return None

        vectors, errors = await embed_pool_texts(
            {"a": "x"}, None, concurrency=1, embed_one=embed_one, sleep=no_sleep
        )
        assert errors == {} and set(vectors) == {"a"} and len(calls) == 2

    async def test_the_concurrency_bound_is_respected(self):
        in_flight = 0
        peak = 0

        async def embed_one(text):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return [1.0]

        vectors, errors = await embed_pool_texts(
            {str(i): str(i) for i in range(12)},
            None,
            concurrency=3,
            embed_one=embed_one,
        )
        assert errors == {} and len(vectors) == 12
        assert peak == 3

    async def test_cancellation_propagates(self):
        started = asyncio.Event()

        async def embed_one(text):
            started.set()
            await asyncio.sleep(60)
            return [1.0]

        task = asyncio.create_task(
            embed_pool_texts({"a": "x"}, None, concurrency=1, embed_one=embed_one)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_no_text_is_no_call(self):
        async def embed_one(text):
            raise AssertionError("no call expected")

        assert await embed_pool_texts({}, None, concurrency=1, embed_one=embed_one) == (
            {},
            {},
        )
