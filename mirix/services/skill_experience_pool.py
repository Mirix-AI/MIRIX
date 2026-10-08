"""The experience pool of the skill curator: its decisions.

A curator batch that finishes keeps the experiences no skill used and that it
found deferred or not worth a skill: they form the pool of their
(organization, user, meta agent) and write scope. When a later batch is
assembled, the pool experiences most similar to the batch's new experiences
are shown again as "earlier lessons", and a pool experience that is very
similar to a new one counts as having recurred.

This module holds the pure decisions — which pool experiences a batch shows
and which of them each new experience recurred (:func:`decide_history`), what
a pool vector embeds (:func:`experience_pool_text`) and which model it comes
from (:func:`embedding_model_key`) — and the embedding calls
(:func:`embed_pool_texts`). It renders nothing and opens no transaction: the
curator (``skill_experience_curator``) renders the lessons, and
``SkillExperienceManager`` reads and writes the pool.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from mirix.constants import MAX_EMBEDDING_DIM

EmbedOne = Callable[[str], Awaitable[Sequence[float]]]

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def experience_pool_text(experience) -> str:
    """What an experience's pool vector embeds: its title and content (never
    the evidence, which quotes one session)."""
    title = " ".join(str(_get(experience, "title") or "").split())
    content = str(_get(experience, "content") or "").strip()
    return f"{title}\n\n{content}".strip()


def embedding_model_key(config) -> Optional[str]:
    """The model a pool vector is stored with, and its dimension:
    ``model@dim`` (``model`` when the configuration names no dimension).
    Vectors of another key are embedded again, never compared: the same model
    at another width (``text-embedding-3-large`` at 1,024 and at 3,072
    dimensions) gives vectors whose zero-padded cosine means nothing. None
    without a model name."""
    model = getattr(config, "embedding_model", None) if config is not None else None
    model = str(model).strip() if model is not None else ""
    if not model:
        return None
    dim = getattr(config, "embedding_dim", None)
    try:
        dim = int(dim) if dim is not None and not isinstance(dim, bool) else None
    except (TypeError, ValueError):
        dim = None
    return f"{model}@{dim}" if dim and dim > 0 else model


@dataclass(frozen=True)
class PoolCandidate:
    """An active pool experience as a batch's history planning sees it."""

    id: str
    experience_type: str
    title: str
    content: str
    importance: float
    credibility: float
    recurrence_count: int
    pushed_count: int
    ignored_pushes: int
    created_at: Optional[dt.datetime]
    vector: Optional[Any] = None  # a sequence of floats or a numpy array
    vector_model: Optional[str] = None


@dataclass(frozen=True)
class RankedLesson:
    """A pool experience a batch may show, with its ranking."""

    candidate: PoolCandidate
    max_cosine: float
    score: float
    recurred: bool


@dataclass
class HistoryDecision:
    """Which pool experiences a batch shows, and what it saw doing so."""

    # eligible, top K, in push order
    ranked: List[RankedLesson] = field(default_factory=list)
    recurred_ids: List[str] = field(default_factory=list)  # in (created_at, id) order
    # new experience id -> the pool experiences it recurred, in (created_at, id)
    # order (only new experiences that recurred one; ``new_ids`` given).
    recurred_by: Dict[str, List[str]] = field(default_factory=dict)
    candidates: int = 0  # candidates with a usable vector
    above_min: int = 0
    unusable_vectors: int = 0  # zero / non-finite / wrong size: skipped
    cosine_log: List[Dict[str, Any]] = field(default_factory=list)
    quantiles: Dict[str, float] = field(default_factory=dict)  # p50, p90, max


def _timestamp(value: Optional[dt.datetime]) -> float:
    """A sortable instant: a naive datetime is UTC (SQLite drops the zone)."""
    if value is None:
        return _EPOCH.timestamp()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.timestamp()


def _unit_vector(vector) -> Optional[np.ndarray]:
    """``vector`` zero-padded to MAX_EMBEDDING_DIM (the rule of every stored
    vector: padding preserves cosine) and unit-normalized, or None when it
    cannot be compared: missing, empty, wider than MAX_EMBEDDING_DIM, not
    finite, or all zeros."""
    if vector is None:
        return None
    try:
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or arr.size > MAX_EMBEDDING_DIM:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm == 0.0:
        return None
    if arr.size < MAX_EMBEDDING_DIM:
        arr = np.pad(arr, (0, MAX_EMBEDDING_DIM - arr.size), mode="constant")
    return arr / norm


def _order_key(candidate: PoolCandidate) -> Tuple[float, str]:
    return (_timestamp(candidate.created_at), str(candidate.id))


def decide_history(
    new_vectors: Sequence[Any],
    candidates: Sequence[PoolCandidate],
    *,
    min_similarity: float,
    recurrence_similarity: float,
    top_k: int,
    decay: float,
    log_limit: int,
    new_ids: Optional[Sequence[str]] = None,
) -> HistoryDecision:
    """Which pool experiences a batch shows. Pure CPU.

    ``sim(p)`` = the max cosine of p's vector to the new vectors (both
    unit-normalized, zero-padded to MAX_EMBEDDING_DIM: padding preserves
    cosine). ``recurred`` = sim >= recurrence_similarity; the effective
    ignored count is 0 when recurred, else ``p.ignored_pushes``; ``score`` =
    sim * decay ** effective; ``eligible`` = sim >= min_similarity; ``ranked``
    = the eligible ones sorted by (-round(score, 6), created_at, id), the
    first ``top_k``. ``cosine_log``: every candidate sorted by
    (-round(sim, 6), created_at, id), the first ``log_limit``, as
    {id, max_cosine (4 dp), score (4 dp), ignored_pushes, recurred, eligible}.
    ``new_ids`` (parallel to ``new_vectors``) also gives ``recurred_by``:
    which pool experiences each new experience recurred. A vector that cannot
    be compared is skipped and counted (``unusable_vectors``). No usable new
    vector or no candidate: an empty decision.
    """
    decision = HistoryDecision()
    if new_ids is not None and len(new_ids) != len(new_vectors):
        raise ValueError("new_ids and new_vectors differ in length")
    new_units: List[np.ndarray] = []
    unit_ids: List[Optional[str]] = []
    for index, vector in enumerate(new_vectors):
        unit = _unit_vector(vector)
        if unit is not None:
            new_units.append(unit)
            unit_ids.append(str(new_ids[index]) if new_ids is not None else None)
    usable: List[Tuple[PoolCandidate, np.ndarray]] = []
    for candidate in candidates:
        unit = _unit_vector(candidate.vector)
        if unit is None:
            decision.unusable_vectors += 1
            continue
        usable.append((candidate, unit))
    if not new_units or not usable:
        return decision

    matrix = np.vstack([unit for _, unit in usable])
    pairwise = matrix @ np.vstack(new_units).T  # candidates x new experiences
    sims = pairwise.max(axis=1)
    if new_ids is not None:
        order = sorted(range(len(usable)), key=lambda i: _order_key(usable[i][0]))
        for column, new_id in enumerate(unit_ids):
            recurred = [
                usable[i][0].id
                for i in order
                if float(pairwise[i, column]) >= recurrence_similarity
            ]
            if recurred and new_id is not None:
                decision.recurred_by[new_id] = recurred
    rows = []
    for (candidate, _), sim in zip(usable, sims):
        sim = float(min(1.0, max(-1.0, sim)))
        recurred = sim >= recurrence_similarity
        effective = 0 if recurred else max(0, int(candidate.ignored_pushes or 0))
        score = sim * (decay**effective)
        rows.append(
            RankedLesson(
                candidate=candidate, max_cosine=sim, score=score, recurred=recurred
            )
        )
    decision.candidates = len(rows)
    eligible = [r for r in rows if r.max_cosine >= min_similarity]
    decision.above_min = len(eligible)
    eligible.sort(key=lambda r: (-round(r.score, 6), *_order_key(r.candidate)))
    decision.ranked = eligible[: max(0, int(top_k))]
    decision.recurred_ids = [
        r.candidate.id
        for r in sorted(
            (r for r in rows if r.recurred), key=lambda r: _order_key(r.candidate)
        )
    ]
    by_sim = sorted(
        rows, key=lambda r: (-round(r.max_cosine, 6), *_order_key(r.candidate))
    )
    decision.cosine_log = [
        {
            "id": r.candidate.id,
            "max_cosine": round(r.max_cosine, 4),
            "score": round(r.score, 4),
            "ignored_pushes": int(r.candidate.ignored_pushes or 0),
            "recurred": r.recurred,
            "eligible": r.max_cosine >= min_similarity,
        }
        for r in by_sim[: max(0, int(log_limit))]
    ]
    values = np.array([r.max_cosine for r in rows], dtype=np.float64)
    decision.quantiles = {
        "p50": round(float(np.quantile(values, 0.5)), 4),
        "p90": round(float(np.quantile(values, 0.9)), 4),
        "max": round(float(values.max()), 4),
    }
    return decision


async def embed_pool_texts(
    texts: Mapping[str, str],
    config,
    *,
    concurrency: int,
    embed_one: Optional[EmbedOne] = None,
    sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, BaseException]]:
    """Embed ``texts`` (id -> text) with at most ``concurrency`` calls in
    flight, each under the single embedding retry layer
    (``retry_call(operation="embedding", stage="experience_pool_embedding")``)
    around ``(await embedding_model(config)).get_text_embedding(text)`` — the
    length guard chunks a long text and pads the vector. ``embed_one(text)``
    replaces that call in tests (still under the retry layer).

    Returns the vectors that succeeded, zero-padded to MAX_EMBEDDING_DIM (the
    ``_pad_to_max`` rule), and each failure by id, in ``texts`` order (empty
    when every call succeeded; a vector wider than MAX_EMBEDDING_DIM is a
    failure too). An embedding failure never raises here; cancellation
    propagates.
    """
    from mirix.services.procedural_memory_manager import _pad_to_max
    from mirix.services.retry_policy import retry_call

    if not texts:
        return {}, {}

    async def _embed_with_config(text: str) -> Sequence[float]:
        from mirix.embeddings import embedding_model

        model = await embedding_model(config)
        return await model.get_text_embedding(text)

    call: EmbedOne = embed_one or _embed_with_config
    gate = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(text: str) -> Tuple[Optional[List[float]], Optional[BaseException]]:
        async with gate:
            try:
                vector = await retry_call(
                    lambda: call(text),
                    operation="embedding",
                    stage="experience_pool_embedding",
                    sleep=sleep,
                )
                if vector is None:
                    raise ValueError("the embedding model returned no vector")
                return _pad_to_max(list(vector)), None
            except Exception as exc:  # noqa: BLE001 — returned, never raised
                return None, exc

    keys = list(texts.keys())
    results = await asyncio.gather(*(_one(texts[key]) for key in keys))
    vectors: Dict[str, List[float]] = {}
    errors: Dict[str, BaseException] = {}
    for key, (vector, error) in zip(keys, results):
        if error is not None or vector is None:
            errors[key] = error or ValueError("the embedding model returned no vector")
            continue
        vectors[key] = vector
    return vectors, errors
