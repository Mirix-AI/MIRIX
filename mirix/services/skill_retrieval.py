"""Procedural (skill) retrieval: the ranking pipeline.

:func:`retrieve_skills` ranks the live skills of one user, in the caller's
scopes, for a query (``ProceduralMemoryManager.retrieve_skills`` is the entry
point the server uses). Six steps:

1. **Candidates** — every non-deleted skill of the user in ``scopes`` (and
   ``filter_tags``), read from the database, never from a cache.
2. **BM25** — Okapi BM25, k1 = 1.2, b = 0.75,
   IDF = ln(1 + (N - df + 0.5) / (df + 0.5)), with N, df and the average
   document length always taken over *all* candidates.

   * The query's terms are its first ``MIRIX_SKILL_BM25_MAX_QUERY_TERMS``
     (64) distinct terms in the order they appear (the raw conversation used
     after a failed extraction has thousands; each costs a scan of every
     candidate's lexemes). The terms left out are counted in the diagnostics.
   * Postgres: the query's terms are the lexemes of
     ``to_tsvector('english', query)``. A document's term frequencies and
     length come from the positions of its ``retrieval_tsv`` (name and
     triggers at weight A, description at weight B): a position of weight A
     counts 1.0, one of weight B 0.4, and the length is the sum over all
     positions. Above ``MIRIX_SKILL_BM25_MAX_DOCS`` candidates only the
     documents containing at least one query term are scored (a GIN-indexed
     ``@@`` filter), while N, df and the length sum are still aggregated in
     SQL over every candidate — the prefilter never changes a score.
   * SQLite (no tsvector): a simple Python tokenization (lower case, letters
     and digits, English stop words dropped) of the same three fields with the
     same weights, scored with ``rank_bm25``.

3. **Cosine** — each candidate's ``retrieval_embedding`` against the query
   vector; a candidate without a vector gets no cosine and is counted.
4. **Fusion** — Reciprocal Rank Fusion of the lanes' ranks,
   ``1 / (k + rank)`` summed with k = 60. A lane ranks only the candidates it
   scored: BM25 those with a positive score, cosine those with a vector.
5. **Gate** (``apply_gate`` only) — a skill stays only when its cosine
   is at least ``MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY`` AND at least
   ``MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO`` x the best cosine of all
   candidates. A skill without a vector never passes. The two values were
   calibrated for one embedding model: :func:`skill_retrieval_gate`
   says which values hold for the query's model and whether they were
   calibrated for it, and the gate trace records that (``calibration``).
6. **Result** — the first ``limit`` skills in fused order; possibly none.
   Each is read together with the newest version of its history in one
   statement (``RetrievedSkill.version_no``; the log's ``returned_versions``):
   the version the agent was served, which the session's distiller shows
   later.

When the query vector cannot be computed, the result is the BM25 lane alone,
ungated, and ``embedding_status`` is ``failed`` (``skipped`` when there is
nothing to embed with: embeddings off, no embedding config, an empty query or
the ``bm25`` method). Every returned skill carries its
``retrieval_scores`` ``{bm25, cosine, rrf}``; :class:`SkillRetrievalDiagnostics`
holds what the retrieval log records for the procedural type.

Ties are broken deterministically: a lane orders by score, then skill id; the
fused order by RRF score, then cosine, then BM25 score, then skill id.

The candidates are read in ONE snapshot; the scoring and ranking run in a
worker thread after that session closed (no pooled connection is held while
Python scores), and the returned skills are then read by id — one deleted
meanwhile is left out.

Nothing here writes a skill row.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import (
    Float,
    Text,
    any_,
    bindparam,
    case,
    cast,
    func,
    literal,
    null,
    select,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSQUERY, aggregate_order_by
from sqlalchemy.orm import lazyload

from mirix import constants
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.database.filter_tags_query import apply_filter_tags_sqlalchemy
from mirix.log import get_logger
from mirix.orm.procedural_memory import ProceduralMemoryItem
from mirix.services.retry_policy import retry_call

logger = get_logger(__name__)

# retrieve_skills(method=...): the lanes a ranking fuses.
METHOD_HYBRID = "hybrid"  # BM25 and cosine (the default)
METHOD_BM25 = "bm25"  # BM25 alone; the query is not embedded
METHOD_EMBEDDING = "embedding"  # cosine alone (BM25 alone if the query vector fails)
RETRIEVAL_METHODS = (METHOD_HYBRID, METHOD_BM25, METHOD_EMBEDDING)

# SkillRetrievalDiagnostics.embedding_status (the retrieval log's column).
EMBEDDING_OK = "ok"
EMBEDDING_FAILED = "failed"
EMBEDDING_SKIPPED = "skipped"

BACKEND_POSTGRES = "postgresql"
BACKEND_SQLITE = "sqlite"

# Lane names in the diagnostics.
LANE_BM25 = "bm25"
LANE_COSINE = "cosine"

# The SQLite tokenizer drops the stop words Postgres' 'english' configuration
# drops (its Snowball stop list), so both backends ignore the same filler.
_ENGLISH_STOP_WORDS = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he
    him his himself she her hers herself it its itself they them their theirs
    themselves what which who whom this that these those am is are was were be
    been being have has had having do does did doing a an the and but if or
    because as until while of at by for with about against between into through
    during before after above below to from up down in out on off over under
    again further then once here there when where why how all any both each few
    more most other some such no nor not only own same so than too very s t can
    will just don should now
    """.split()
)
_TOKEN_PATTERN = re.compile(r"[^\W_]+")


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkillRetrievalScores:
    """A returned skill's ``retrieval_scores``: its BM25 score (0.0 when it
    matched no query term), its cosine similarity (None without a query vector
    or a skill vector) and its fused RRF score."""

    bm25: float
    cosine: Optional[float]
    rrf: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"bm25": self.bm25, "cosine": self.cosine, "rrf": self.rrf}


@dataclass
class RetrievedSkill:
    """One ranked skill, its scores and the version it was served at: the
    newest ``skill_version.version_no`` of the skill when its row was read
    (None for a skill with no history yet)."""

    skill: Any  # mirix.schemas.procedural_memory.ProceduralMemoryItem
    scores: SkillRetrievalScores
    version_no: Optional[int] = None


@dataclass
class CandidateTrace:
    """One candidate's scores and ranks (1-based; None when the lane did not
    rank it). ``gate_passed`` is None when no gate was applied."""

    id: str
    bm25: float
    bm25_rank: Optional[int]
    cosine: Optional[float]
    cosine_rank: Optional[int]
    rrf: float
    rrf_rank: Optional[int]
    gate_passed: Optional[bool]
    returned: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bm25": self.bm25,
            "bm25_rank": self.bm25_rank,
            "cosine": self.cosine,
            "cosine_rank": self.cosine_rank,
            "rrf": self.rrf,
            "rrf_rank": self.rrf_rank,
            "gate_passed": self.gate_passed,
            "returned": self.returned,
        }


@dataclass
class GateTrace:
    """The gate's parameters and outcome. ``threshold`` is the cosine a skill
    needed: max(min_similarity, max_cosine x relative_ratio)."""

    applied: bool
    min_similarity: float
    relative_ratio: float
    max_cosine: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[int] = None
    #: Why a requested gate was not applied (no query vector).
    skipped_reason: Optional[str] = None
    #: Where min_similarity / relative_ratio come from (GateSettings.source):
    #: the calibrated model's name, "uncalibrated" or "env"; None when the
    #: caller resolved none (no embedding model known, or no gate asked for).
    calibration: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "min_similarity": self.min_similarity,
            "relative_ratio": self.relative_ratio,
            "calibration": self.calibration,
            "max_cosine": self.max_cosine,
            "threshold": self.threshold,
            "passed": self.passed,
            "skipped_reason": self.skipped_reason,
        }


# --------------------------------------------------------------------------- #
# The gate's settings per embedding model                                      #
# --------------------------------------------------------------------------- #

# GateSettings.source besides a calibrated model's name.
GATE_CALIBRATION_ENV = "env"
GATE_CALIBRATION_UNCALIBRATED = "uncalibrated"


@dataclass(frozen=True)
class GateSettings:
    """The similarity gate's values for one embedding model, and where they come
    from (``source``): ``env`` (an operator's MIRIX_SKILL_RETRIEVAL_* setting,
    for every model), the name of the calibrated model
    (``SKILL_RETRIEVAL_GATE_CALIBRATIONS``), or ``uncalibrated`` (the
    defaults, calibrated for another model)."""

    min_similarity: float
    relative_ratio: float
    source: str


_warned_uncalibrated_models: set = set()


def skill_retrieval_gate(embedding_model: Optional[str]) -> GateSettings:
    """The gate for skills and queries embedded with ``embedding_model``.

    An environment setting wins for every model. Otherwise the model is
    looked up in ``SKILL_RETRIEVAL_GATE_CALIBRATIONS`` as given, then without
    a provider prefix (the part after the last ``/``, as
    :func:`mirix.embedding_guard.embedding_token_limit` does). A model not in
    the table gets the defaults as ``uncalibrated``, with one warning per
    model name: its cosines may sit on another scale, so the floor may let
    everything or nothing through.
    """
    if constants.SKILL_RETRIEVAL_GATE_FROM_ENV:
        return GateSettings(
            float(constants.SKILL_RETRIEVAL_MIN_SIMILARITY),
            float(constants.SKILL_RETRIEVAL_RELATIVE_RATIO),
            GATE_CALIBRATION_ENV,
        )
    name = str(embedding_model or "")
    table = constants.SKILL_RETRIEVAL_GATE_CALIBRATIONS
    for candidate in (name, name.rsplit("/", 1)[-1]):
        if candidate in table:
            min_similarity, relative_ratio = table[candidate]
            return GateSettings(float(min_similarity), float(relative_ratio), candidate)
    if name not in _warned_uncalibrated_models:
        _warned_uncalibrated_models.add(name)
        logger.warning(
            "Skill retrieval gate: embedding model %r has no calibration (calibrated: "
            "%s); using min_similarity %s and relative_ratio %s, which may let too "
            "many or too few skills through. Calibrate it and set "
            "MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY / MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO",
            name,
            ", ".join(sorted(table)) or "none",
            constants.SKILL_RETRIEVAL_MIN_SIMILARITY,
            constants.SKILL_RETRIEVAL_RELATIVE_RATIO,
        )
    return GateSettings(
        float(constants.SKILL_RETRIEVAL_MIN_SIMILARITY),
        float(constants.SKILL_RETRIEVAL_RELATIVE_RATIO),
        GATE_CALIBRATION_UNCALIBRATED,
    )


@dataclass
class BM25Trace:
    """The BM25 lane's corpus statistics (over all candidates)."""

    query_terms: List[str]
    n_docs: int
    avgdl: float
    doc_freq: Dict[str, int]
    k1: float
    b: float
    weights: Dict[str, float]
    #: Only the candidates containing a query term were scored (N above max_docs).
    prefiltered: bool
    max_docs: int
    scored_docs: int
    #: Distinct query terms beyond MIRIX_SKILL_BM25_MAX_QUERY_TERMS, not scored.
    query_terms_dropped: int = 0
    max_query_terms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_terms": list(self.query_terms),
            "query_terms_dropped": self.query_terms_dropped,
            "max_query_terms": self.max_query_terms,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_freq": dict(self.doc_freq),
            "k1": self.k1,
            "b": self.b,
            "weights": dict(self.weights),
            "prefiltered": self.prefiltered,
            "max_docs": self.max_docs,
            "scored_docs": self.scored_docs,
        }


@dataclass
class SkillRetrievalDiagnostics:
    """What one ranking did — the procedural entry of the retrieval log's
    ``per_type``. :meth:`to_dict` is JSON-serializable.

    ``method`` is what ran (``bm25`` when the query vector was unavailable),
    ``requested_method`` what was asked for; ``lanes`` the lanes fused.
    ``candidates`` lists every candidate: first the ranked ones in fused order,
    then the unranked ones by id.
    """

    requested_method: str
    method: str
    lanes: List[str]
    backend: str
    embedding_status: str
    embedding_error: Optional[str]
    candidate_count: int
    missing_vector_count: int
    bm25: BM25Trace
    rrf_k: int
    gate: GateTrace
    limit: int
    max_cosine_distance: Optional[float]
    candidates: List[CandidateTrace]
    returned_ids: List[str]
    #: The version each returned skill was served at (``skill_id ->
    #: version_no``, None for a skill without history): what the session's
    #: distiller later shows as "served".
    returned_versions: Dict[str, Optional[int]] = field(default_factory=dict)

    def bounded_candidates(self, max_candidates: int) -> List[CandidateTrace]:
        """Every returned candidate plus the first ``max_candidates`` other
        RANKED ones, in fused order (what a retrieval log keeps)."""
        kept: List[CandidateTrace] = []
        others = 0
        for trace in self.candidates:
            if trace.returned:
                kept.append(trace)
            elif trace.rrf_rank is not None and others < max_candidates:
                kept.append(trace)
                others += 1
        return kept

    def to_dict(self, max_candidates: Optional[int] = None) -> Dict[str, Any]:
        """The diagnostics as JSON data. ``max_candidates`` bounds the
        candidate list (:meth:`bounded_candidates`) and adds
        ``candidates_logged`` / ``candidates_omitted``: a retrieval log row
        stays small whatever the number of skills."""
        candidates = self.candidates
        bounded: Dict[str, Any] = {}
        if max_candidates is not None:
            candidates = self.bounded_candidates(max(0, int(max_candidates)))
            bounded = {
                "candidates_logged": len(candidates),
                "candidates_omitted": len(self.candidates) - len(candidates),
            }
        return {
            "requested_method": self.requested_method,
            "method": self.method,
            "lanes": list(self.lanes),
            "backend": self.backend,
            "embedding_status": self.embedding_status,
            "embedding_error": self.embedding_error,
            "candidate_count": self.candidate_count,
            "missing_vector_count": self.missing_vector_count,
            "bm25": self.bm25.to_dict(),
            "rrf_k": self.rrf_k,
            "gate": self.gate.to_dict(),
            "limit": self.limit,
            "max_cosine_distance": self.max_cosine_distance,
            "candidates": [c.to_dict() for c in candidates],
            **bounded,
            "returned_ids": list(self.returned_ids),
            # Only when the ranking recorded them (every retrieval does; a
            # trace built without them keeps its earlier shape).
            **(
                {
                    "returned_versions": {
                        skill_id: self.returned_versions.get(skill_id)
                        for skill_id in self.returned_ids
                    }
                }
                if self.returned_versions
                else {}
            ),
        }


@dataclass
class SkillRetrievalResult:
    """The ranked skills (in order) and the diagnostics of the ranking."""

    skills: List[RetrievedSkill]
    diagnostics: SkillRetrievalDiagnostics

    @property
    def items(self) -> List[Any]:
        """The ranked skills without their scores."""
        return [r.skill for r in self.skills]


# --------------------------------------------------------------------------- #
# BM25                                                                         #
# --------------------------------------------------------------------------- #


def tokenize(value: Optional[str]) -> List[str]:
    """The SQLite path's tokenizer: lower case, runs of letters and digits,
    English stop words dropped. No stemming."""
    if not value:
        return []
    return [
        token
        for token in _TOKEN_PATTERN.findall(str(value).lower())
        if token not in _ENGLISH_STOP_WORDS
    ]


def query_terms_python(query: str) -> List[str]:
    """The query's distinct terms on the SQLite path, sorted."""
    return sorted(set(tokenize(query)))


def cap_query_terms(
    ordered_terms: Iterable[str], max_terms: Optional[int] = None
) -> Tuple[List[str], int]:
    """The first ``max_terms`` (``MIRIX_SKILL_BM25_MAX_QUERY_TERMS``) distinct
    terms of ``ordered_terms`` (in query order), sorted; and how many distinct
    terms were left out."""
    cap = constants.SKILL_BM25_MAX_QUERY_TERMS if max_terms is None else max_terms
    distinct = list(dict.fromkeys(ordered_terms))
    kept = distinct[: max(0, int(cap))]
    return sorted(kept), len(distinct) - len(kept)


def weighted_term_frequencies(
    weighted_texts: Iterable[Tuple[Optional[str], float]],
) -> Tuple[Dict[str, float], float]:
    """(term -> weighted frequency, document length) of texts whose every token
    position counts its text's weight — the Python twin of the Postgres
    positions arithmetic."""
    frequencies: Dict[str, float] = {}
    length = 0.0
    for value, weight in weighted_texts:
        for token in tokenize(value):
            frequencies[token] = frequencies.get(token, 0.0) + weight
            length += weight
    return frequencies, length


def skill_weighted_texts(
    name: Optional[str], triggers: Any, description: Optional[str]
) -> List[Tuple[Optional[str], float]]:
    """A skill's retrieval-text fields with their BM25 weights, like the
    ``retrieval_tsv`` expression: name and triggers A, description B."""
    weight_a = constants.SKILL_BM25_WEIGHT_A
    weight_b = constants.SKILL_BM25_WEIGHT_B
    if isinstance(triggers, str):
        triggers = [triggers]
    texts: List[Tuple[Optional[str], float]] = [(name, weight_a)]
    texts.extend((str(t), weight_a) for t in (triggers or []) if t)
    texts.append((description, weight_b))
    return texts


def bm25_idf(n_docs: int, doc_freq: int) -> float:
    """IDF = ln(1 + (N - df + 0.5) / (df + 0.5)); always positive."""
    doc_freq = min(max(int(doc_freq), 0), int(n_docs))
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


@dataclass(frozen=True)
class BM25CorpusStats:
    """N, the query terms' document frequencies and the average document
    length, over all candidates."""

    n_docs: int
    doc_freq: Dict[str, int]
    avgdl: float


def corpus_stats(
    term_freqs: Sequence[Mapping[str, float]],
    doc_lens: Sequence[float],
    query_terms: Sequence[str],
) -> BM25CorpusStats:
    """The statistics of a whole corpus (every candidate)."""
    n_docs = len(term_freqs)
    doc_freq = {
        term: sum(1 for tf in term_freqs if tf.get(term, 0.0) > 0)
        for term in query_terms
    }
    avgdl = (float(sum(doc_lens)) / n_docs) if n_docs else 0.0
    return BM25CorpusStats(n_docs=n_docs, doc_freq=doc_freq, avgdl=avgdl)


class SkillBM25(BM25Okapi):
    """``rank_bm25``'s Okapi BM25 with the skill retrieval parameters.

    ``rank_bm25`` derives its statistics from token lists; here the term
    frequencies are weighted (A = 1.0, B = 0.4) and N, df and avgdl may cover
    more documents than are scored (the prefilter above
    ``MIRIX_SKILL_BM25_MAX_DOCS``), so they are given, not counted. Scoring is
    ``rank_bm25``'s own ``get_scores``; the IDF is
    ln(1 + (N - df + 0.5) / (df + 0.5)), which is never negative (no epsilon
    floor).
    """

    def __init__(
        self,
        term_freqs: Sequence[Mapping[str, float]],
        doc_lens: Sequence[float],
        *,
        n_docs: int,
        doc_freq: Mapping[str, int],
        avgdl: float,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ):
        # Deliberately not BM25.__init__: it would count raw tokens itself.
        if len(term_freqs) != len(doc_lens):
            raise ValueError("term_freqs and doc_lens must have the same length")
        self.k1 = constants.SKILL_BM25_K1 if k1 is None else float(k1)
        self.b = constants.SKILL_BM25_B if b is None else float(b)
        self.epsilon = 0.0
        self.tokenizer = None
        self.doc_freqs = [dict(tf) for tf in term_freqs]
        self.doc_len = [float(length) for length in doc_lens]
        self.corpus_size = len(self.doc_freqs)
        self.n_docs = int(n_docs)
        self.avgdl = float(avgdl)
        self.idf: Dict[str, float] = {}
        self._calc_idf(dict(doc_freq))

    @classmethod
    def from_documents(
        cls,
        term_freqs: Sequence[Mapping[str, float]],
        doc_lens: Sequence[float],
        query_terms: Sequence[str],
    ) -> "SkillBM25":
        """A scorer whose statistics are those of the scored documents."""
        stats = corpus_stats(term_freqs, doc_lens, query_terms)
        return cls(
            term_freqs,
            doc_lens,
            n_docs=stats.n_docs,
            doc_freq=stats.doc_freq,
            avgdl=stats.avgdl,
        )

    def _calc_idf(self, nd):  # noqa: D401 — rank_bm25's hook
        for term, df in nd.items():
            self.idf[term] = bm25_idf(self.n_docs, df)
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0

    def score(self, query_terms: Sequence[str]) -> List[float]:
        """The BM25 score of every scored document, in order."""
        if not self.corpus_size:
            return []
        if self.avgdl <= 0 or not query_terms:
            return [0.0] * self.corpus_size
        return [float(s) for s in self.get_scores(list(query_terms))]


# --------------------------------------------------------------------------- #
# Ranking (pure)                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class _Candidate:
    id: str
    term_freqs: Dict[str, float] = field(default_factory=dict)
    doc_len: float = 0.0
    #: Scored by BM25 (False only for the documents a prefilter skipped).
    bm25_scored: bool = True
    has_vector: bool = False
    cosine: Optional[float] = None


@dataclass
class _Corpus:
    backend: str
    query_terms: List[str]
    candidates: List[_Candidate]  # every candidate
    stats: BM25CorpusStats
    prefiltered: bool = False
    query_terms_dropped: int = 0


def _bm25_scores(corpus: _Corpus) -> Dict[str, float]:
    """BM25 score per candidate id (0.0 for the unscored ones)."""
    scores = {c.id: 0.0 for c in corpus.candidates}
    scored = [c for c in corpus.candidates if c.bm25_scored]
    if not scored or not corpus.query_terms:
        return scores
    scorer = SkillBM25(
        [c.term_freqs for c in scored],
        [c.doc_len for c in scored],
        n_docs=corpus.stats.n_docs,
        doc_freq=corpus.stats.doc_freq,
        avgdl=corpus.stats.avgdl,
    )
    for candidate, value in zip(scored, scorer.score(corpus.query_terms)):
        scores[candidate.id] = value if math.isfinite(value) else 0.0
    return scores


def _lane_ranks(scored: Mapping[str, float]) -> Dict[str, int]:
    """1-based ranks by score (descending), ties by id."""
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    return {
        candidate_id: rank for rank, (candidate_id, _) in enumerate(ordered, start=1)
    }


@dataclass
class _Ranking:
    traces: List[CandidateTrace]
    returned: List[CandidateTrace]
    gate: GateTrace
    lanes: List[str]


def rank_candidates(
    candidates: Sequence[_Candidate],
    bm25_scores: Mapping[str, float],
    *,
    lanes: Sequence[str],
    apply_gate: bool,
    limit: int,
    rrf_k: int,
    min_similarity: float,
    relative_ratio: float,
    max_cosine_distance: Optional[float] = None,
    gate_calibration: Optional[str] = None,
) -> _Ranking:
    """Steps 4-6: fuse the lanes' ranks, gate, cut to ``limit``.

    ``lanes`` names the lanes fused (``bm25`` and/or ``cosine``). The gate is
    applied only with ``apply_gate`` and a cosine lane. ``max_cosine_distance``
    (``/memory/search``'s ``similarity_threshold``) keeps the candidates whose
    cosine distance is not below it out of the cosine lane; it never affects
    the gate's best cosine. ``gate_calibration`` is recorded in the gate
    trace (where the two gate values come from).
    """
    use_bm25 = LANE_BM25 in lanes
    use_cosine = LANE_COSINE in lanes
    bm25_lane = (
        {
            c.id: bm25_scores.get(c.id, 0.0)
            for c in candidates
            if bm25_scores.get(c.id, 0.0) > 0
        }
        if use_bm25
        else {}
    )
    cosine_lane = (
        {
            c.id: c.cosine
            for c in candidates
            if c.cosine is not None
            and (max_cosine_distance is None or 1.0 - c.cosine < max_cosine_distance)
        }
        if use_cosine
        else {}
    )
    bm25_ranks = _lane_ranks(bm25_lane)
    cosine_ranks = _lane_ranks(cosine_lane)

    cosines = (
        [c.cosine for c in candidates if c.cosine is not None] if use_cosine else []
    )
    max_cosine = max(cosines) if cosines else None
    gate = GateTrace(
        applied=bool(apply_gate and use_cosine),
        min_similarity=float(min_similarity),
        relative_ratio=float(relative_ratio),
        calibration=gate_calibration,
    )
    if apply_gate and not use_cosine:
        gate.skipped_reason = "no query vector"
    if gate.applied:
        gate.max_cosine = max_cosine
        gate.threshold = (
            max(min_similarity, max_cosine * relative_ratio)
            if max_cosine is not None
            else None
        )

    def passes(candidate: _Candidate) -> bool:
        return (
            candidate.cosine is not None
            and max_cosine is not None
            and candidate.cosine >= min_similarity
            and candidate.cosine >= max_cosine * relative_ratio
        )

    traces: List[CandidateTrace] = []
    for candidate in candidates:
        rrf = 0.0
        bm25_rank = bm25_ranks.get(candidate.id)
        cosine_rank = cosine_ranks.get(candidate.id)
        if bm25_rank is not None:
            rrf += 1.0 / (rrf_k + bm25_rank)
        if cosine_rank is not None:
            rrf += 1.0 / (rrf_k + cosine_rank)
        traces.append(
            CandidateTrace(
                id=candidate.id,
                bm25=float(bm25_scores.get(candidate.id, 0.0)),
                bm25_rank=bm25_rank,
                cosine=candidate.cosine if use_cosine else None,
                cosine_rank=cosine_rank,
                rrf=rrf,
                rrf_rank=None,
                gate_passed=passes(candidate) if gate.applied else None,
                returned=False,
            )
        )

    def fused_key(trace: CandidateTrace):
        cosine = trace.cosine if trace.cosine is not None else -math.inf
        return (-trace.rrf, -cosine, -trace.bm25, trace.id)

    ranked = sorted((t for t in traces if t.rrf > 0), key=fused_key)
    for rank, trace in enumerate(ranked, start=1):
        trace.rrf_rank = rank
    unranked = sorted((t for t in traces if t.rrf <= 0), key=lambda t: t.id)
    eligible = [t for t in ranked if not gate.applied or t.gate_passed]
    if gate.applied:
        gate.passed = len(eligible)
    returned = eligible[: max(0, limit)]
    for trace in returned:
        trace.returned = True
    used_lanes = [lane for lane in (LANE_BM25, LANE_COSINE) if lane in lanes]
    return _Ranking(
        traces=ranked + unranked, returned=returned, gate=gate, lanes=used_lanes
    )


# --------------------------------------------------------------------------- #
# Reading the candidates                                                       #
# --------------------------------------------------------------------------- #


def _candidate_filter(
    stmt,
    *,
    user_id: str,
    organization_id: str,
    scopes: Optional[Sequence[str]],
    filter_tags: Optional[Mapping[str, Any]],
):
    """Step 1's WHERE: the user's live skills in the scopes and filter tags."""
    stmt = stmt.where(
        ProceduralMemoryItem.user_id == user_id,
        ProceduralMemoryItem.organization_id == organization_id,
        ~ProceduralMemoryItem.is_deleted,
    )
    return apply_filter_tags_sqlalchemy(
        stmt,
        ProceduralMemoryItem,
        dict(filter_tags) if filter_tags else None,
        scopes=list(scopes) if scopes is not None else None,
    )


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _tsquery_literal(lexeme: str) -> str:
    """A lexeme as tsquery input, verbatim (no normalization): quoted, with
    quotes doubled and backslashes escaped."""
    return "'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'"


def _positions_weight_sql(weights):
    """Postgres: the weighted count of a lexeme's positions — each position of
    weight A counts SKILL_BM25_WEIGHT_A, every other SKILL_BM25_WEIGHT_B
    (``retrieval_tsv`` has only A and B)."""
    weight_a = constants.SKILL_BM25_WEIGHT_A
    weight_b = constants.SKILL_BM25_WEIGHT_B
    return literal(weight_b, Float) * func.cardinality(weights) + literal(
        weight_a - weight_b, Float
    ) * func.cardinality(func.array_positions(weights, literal("A", Text)))


class _PostgresExpressions:
    """Correlated expressions over a skill row's ``retrieval_tsv``."""

    def __init__(self, query_terms: Sequence[str]):
        tsv = ProceduralMemoryItem.retrieval_tsv
        lexemes = (
            func.unnest(tsv).table_valued("lexeme", "positions", "weights").alias("tl")
        )
        self.lexemes = lexemes
        weighted = _positions_weight_sql(lexemes.c.weights)
        self.terms = bindparam(
            "query_terms", value=list(query_terms), type_=ARRAY(Text)
        )
        self.doc_len = (
            select(func.coalesce(func.sum(weighted), literal(0.0, Float)))
            .select_from(lexemes)
            .scalar_subquery()
        )
        matching = lexemes.c.lexeme == any_(self.terms)
        tf_terms = (
            select(
                func.array_agg(aggregate_order_by(lexemes.c.lexeme, lexemes.c.lexeme))
            )
            .select_from(lexemes)
            .where(matching)
            .scalar_subquery()
        )
        tf_values = (
            select(func.array_agg(aggregate_order_by(weighted, lexemes.c.lexeme)))
            .select_from(lexemes)
            .where(matching)
            .scalar_subquery()
        )
        self.any_term = cast(
            bindparam(
                "any_query_term",
                value=" | ".join(_tsquery_literal(t) for t in query_terms),
                type_=Text,
            ),
            TSQUERY,
        )
        if query_terms:
            # The per-lexeme scan only for a document that holds a query term
            # (the GIN-able @@ is a binary search per term): most candidates
            # of a specific query hold none.
            holds_a_term = tsv.op("@@")(self.any_term)
            self.tf_terms = case((holds_a_term, tf_terms), else_=null())
            self.tf_values = case((holds_a_term, tf_values), else_=null())
        else:
            self.tf_terms = null()
            self.tf_values = null()


def _term_freqs(lexemes: Optional[Sequence[str]], values: Optional[Sequence[Any]]):
    if not lexemes:
        return {}
    return {
        str(lex): float(v) for lex, v in zip(lexemes, values or []) if v is not None
    }


async def _read_postgres(
    session,
    query: str,
    query_vector: Optional[List[float]],
    *,
    filters: Callable,
    max_docs: int,
) -> _Corpus:
    # One snapshot for every statement below: N, df and avgdl describe exactly
    # the candidates that are scored and returned.
    await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    query_terms: List[str] = []
    dropped = 0
    if query:
        # In the order they first appear in the query (positions[1]), capped.
        rows = await session.execute(
            text(
                "SELECT lexeme FROM unnest(to_tsvector('english', :query)) "
                "ORDER BY coalesce(positions[1], 0), lexeme"
            ),
            {"query": query},
        )
        query_terms, dropped = cap_query_terms(str(r[0]) for r in rows)
    pm = ProceduralMemoryItem
    expr = _PostgresExpressions(query_terms)
    n_docs = int(
        (
            await session.execute(filters(select(func.count()).select_from(pm)))
        ).scalar_one()
    )
    if n_docs == 0:
        return _Corpus(
            BACKEND_POSTGRES,
            query_terms,
            [],
            BM25CorpusStats(0, {}, 0.0),
            query_terms_dropped=dropped,
        )

    vector_columns = []
    if query_vector is not None:
        vector_columns.append(
            (1 - pm.retrieval_embedding.cosine_distance(query_vector)).label("cosine")
        )
    has_vector = pm.retrieval_embedding.is_not(None).label("has_vector")
    prefiltered = bool(query_terms) and n_docs > max_docs

    if not prefiltered:
        stmt = filters(
            select(
                pm.id,
                expr.doc_len.label("doc_len"),
                expr.tf_terms.label("tf_terms"),
                expr.tf_values.label("tf_values"),
                has_vector,
                *vector_columns,
            )
        ).order_by(pm.id)
        rows = (await session.execute(stmt)).mappings().all()
        corpus = await asyncio.to_thread(_postgres_corpus, rows, query_terms)
        corpus.query_terms_dropped = dropped
        return corpus

    # Above max_docs: N, df and the length sum over EVERY candidate, in SQL...
    totals = (
        await session.execute(
            filters(
                select(
                    func.count(),
                    func.coalesce(func.sum(expr.doc_len), literal(0.0, Float)),
                ).select_from(pm)
            )
        )
    ).one()
    n_all, length_sum = int(totals[0]), float(totals[1] or 0.0)
    df_rows = await session.execute(
        filters(
            select(expr.lexemes.c.lexeme, func.count())
            .select_from(pm)
            .join(expr.lexemes, true())
            .where(expr.lexemes.c.lexeme == any_(expr.terms))
            .group_by(expr.lexemes.c.lexeme)
        )
    )
    doc_freq = {term: 0 for term in query_terms}
    doc_freq.update({str(lexeme): int(count) for lexeme, count in df_rows})
    stats = BM25CorpusStats(
        n_docs=n_all, doc_freq=doc_freq, avgdl=(length_sum / n_all) if n_all else 0.0
    )
    # ...but only the candidates holding a query term are scored (GIN)...
    scored_rows = await session.execute(
        filters(
            select(
                pm.id,
                expr.doc_len.label("doc_len"),
                expr.tf_terms.label("tf_terms"),
                expr.tf_values.label("tf_values"),
            ).where(pm.retrieval_tsv.op("@@")(expr.any_term))
        )
    )
    scored = scored_rows.mappings().all()
    # ...and every candidate still gets its cosine.
    lane_rows = await session.execute(
        filters(select(pm.id, has_vector, *vector_columns)).order_by(pm.id)
    )
    lanes = lane_rows.mappings().all()
    corpus = await asyncio.to_thread(
        _prefiltered_postgres_corpus, lanes, scored, query_terms, stats
    )
    corpus.query_terms_dropped = dropped
    return corpus


def _postgres_candidate(row: Mapping[str, Any], hit: Optional[Mapping[str, Any]]):
    return _Candidate(
        id=str(row["id"]),
        term_freqs=_term_freqs(hit["tf_terms"], hit["tf_values"]) if hit else {},
        doc_len=float(hit["doc_len"] or 0.0) if hit else 0.0,
        bm25_scored=hit is not None,
        has_vector=bool(row["has_vector"]),
        cosine=_finite_or_none(row.get("cosine")),
    )


def _postgres_corpus(
    rows: Sequence[Mapping[str, Any]], query_terms: List[str]
) -> _Corpus:
    """Every candidate scored: the statistics are those of the rows (runs off
    the event loop)."""
    candidates = [_postgres_candidate(row, row) for row in rows]
    stats = corpus_stats(
        [c.term_freqs for c in candidates], [c.doc_len for c in candidates], query_terms
    )
    return _Corpus(BACKEND_POSTGRES, query_terms, candidates, stats)


def _prefiltered_postgres_corpus(
    lanes: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    query_terms: List[str],
    stats: BM25CorpusStats,
) -> _Corpus:
    """Every candidate, BM25-scored only when it holds a query term; the
    statistics are the SQL aggregates over all (runs off the event loop)."""
    hits = {str(row["id"]): row for row in scored}
    candidates = [_postgres_candidate(row, hits.get(str(row["id"]))) for row in lanes]
    return _Corpus(BACKEND_POSTGRES, query_terms, candidates, stats, prefiltered=True)


def _cosine(query: np.ndarray, query_norm: float, vector: Any) -> Optional[float]:
    if vector is None:
        return None
    values = np.asarray(vector, dtype=np.float64).ravel()
    if values.shape[0] != query.shape[0]:
        size = max(values.shape[0], query.shape[0])
        values = np.pad(values, (0, size - values.shape[0]))
        query = np.pad(query, (0, size - query.shape[0]))
    norm = float(np.linalg.norm(values))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    return _finite_or_none(float(np.dot(query, values)) / (query_norm * norm))


def _sqlite_corpus(
    rows: Sequence[Any], query: str, query_vector: Optional[List[float]]
) -> _Corpus:
    """The SQLite corpus: Python tokens and cosines (runs off the event loop)."""
    query_terms, dropped = cap_query_terms(tokenize(query))
    q = np.asarray(query_vector, dtype=np.float64) if query_vector is not None else None
    q_norm = float(np.linalg.norm(q)) if q is not None else 0.0
    candidates = []
    for row in rows:
        term_freqs, doc_len = weighted_term_frequencies(
            skill_weighted_texts(row.name, row.triggers, row.description)
        )
        vector = row.retrieval_embedding
        candidates.append(
            _Candidate(
                id=str(row.id),
                term_freqs=term_freqs,
                doc_len=doc_len,
                has_vector=vector is not None,
                cosine=_cosine(q, q_norm, vector) if q is not None else None,
            )
        )
    stats = corpus_stats(
        [c.term_freqs for c in candidates], [c.doc_len for c in candidates], query_terms
    )
    return _Corpus(
        BACKEND_SQLITE, query_terms, candidates, stats, query_terms_dropped=dropped
    )


async def _read_sqlite(
    session, query: str, query_vector: Optional[List[float]], *, filters: Callable
) -> _Corpus:
    pm = ProceduralMemoryItem
    stmt = filters(
        select(pm.id, pm.name, pm.triggers, pm.description, pm.retrieval_embedding)
    ).order_by(pm.id)
    rows = (await session.execute(stmt)).all()
    return await asyncio.to_thread(_sqlite_corpus, rows, query, query_vector)


def _dialect_name(session) -> str:
    bind = getattr(session, "bind", None)
    if bind is None:
        try:
            bind = session.get_bind()
        except Exception:  # noqa: BLE001 — unknown binds fall back to SQLite
            return ""
    return getattr(getattr(bind, "dialect", None), "name", "") or ""


def _uses_retrieval_tsv(session) -> bool:
    """Postgres with the generated ``retrieval_tsv`` column mapped (the ORM
    maps it only when MIRIX_PG_URI is set); every other store ranks with the
    Python tokenizer."""
    return _dialect_name(session) == BACKEND_POSTGRES and hasattr(
        ProceduralMemoryItem, "retrieval_tsv"
    )


# --------------------------------------------------------------------------- #
# The query vector                                                             #
# --------------------------------------------------------------------------- #


def _usable_query_vector(vector: Sequence[float]) -> List[float]:
    """The vector padded to MAX_EMBEDDING_DIM; ValueError when it cannot rank
    anything (too long, empty, zero or non-finite)."""
    values = np.asarray(list(vector), dtype=np.float64).ravel()
    if values.shape[0] == 0 or values.shape[0] > MAX_EMBEDDING_DIM:
        raise ValueError(f"query vector has {values.shape[0]} dimensions")
    norm = float(np.linalg.norm(values))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("query vector is zero or not finite")
    if values.shape[0] < MAX_EMBEDDING_DIM:
        values = np.pad(values, (0, MAX_EMBEDDING_DIM - values.shape[0]))
    return values.tolist()


async def _query_vector(
    query: str,
    *,
    wanted: bool,
    query_embedding: Optional[Sequence[float]],
    embed_query: Optional[Callable[[str], Awaitable[Optional[Sequence[float]]]]],
) -> Tuple[Optional[List[float]], str, Optional[str]]:
    """(padded query vector or None, embedding_status, error)."""
    if not wanted:
        return None, EMBEDDING_SKIPPED, None
    if not query:
        return None, EMBEDDING_SKIPPED, "empty query"
    try:
        if query_embedding is not None:
            raw: Optional[Sequence[float]] = query_embedding
        elif embed_query is None:
            return None, EMBEDDING_SKIPPED, "no embedding configured"
        else:
            raw = await embed_query(query)
            if raw is None:
                return None, EMBEDDING_SKIPPED, "embeddings are off or not configured"
        return _usable_query_vector(raw), EMBEDDING_OK, None
    except Exception as exc:  # noqa: BLE001 — M7.3: fall back to BM25 alone
        logger.warning(
            "Skill retrieval: the query vector failed (%s: %s); ranking by BM25 alone",
            type(exc).__name__,
            exc,
        )
        return None, EMBEDDING_FAILED, f"{type(exc).__name__}: {exc}"[:500]


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


async def retrieve_skills(
    query: Optional[str],
    user: Any,
    scopes: Optional[Sequence[str]],
    limit: Optional[int] = None,
    apply_gate: bool = False,
    *,
    session_maker: Callable[[], Any],
    embed_query: Optional[Callable[[str], Awaitable[Optional[Sequence[float]]]]] = None,
    query_embedding: Optional[Sequence[float]] = None,
    filter_tags: Optional[Mapping[str, Any]] = None,
    method: str = METHOD_HYBRID,
    max_cosine_distance: Optional[float] = None,
    min_similarity: Optional[float] = None,
    relative_ratio: Optional[float] = None,
    gate_calibration: Optional[str] = None,
) -> SkillRetrievalResult:
    """Rank ``user``'s live skills in ``scopes`` for ``query``.

    Args:
        query: the query text; an empty one ranks nothing.
        user: the owner (``.id``, ``.organization_id``).
        scopes: the ``filter_tags.scope`` values the caller may read; None =
            no restriction, ``[]`` = nothing.
        limit: at most this many skills (default
            ``SKILL_RETRIEVAL_DEFAULT_LIMIT``, 10); 0 returns none.
        apply_gate: apply the similarity gate (only where skills are injected into a
            context: ``/memory/retrieve/conversation``; never
            ``/memory/search``).
        session_maker: async context manager factory yielding a session.
        embed_query: embeds the query (raising on failure; None when
            embeddings are off). Not called when ``query_embedding`` is given.
        query_embedding: a precomputed query vector (padded here).
        filter_tags: further ``filter_tags`` conditions on the candidates.
        method: ``hybrid`` (default), ``bm25`` or ``embedding`` — the lanes
            fused. Without a query vector every method ranks by BM25 alone.
        max_cosine_distance: ``/memory/search``'s ``similarity_threshold`` —
            candidates whose cosine distance is not below it get no cosine
            rank.
        min_similarity, relative_ratio: the gate's values (default:
            ``MIRIX_SKILL_RETRIEVAL_MIN_SIMILARITY`` /
            ``MIRIX_SKILL_RETRIEVAL_RELATIVE_RATIO``); a caller that knows the
            query's embedding model passes :func:`skill_retrieval_gate`'s.
        gate_calibration: where those values come from
            (``GateSettings.source``), recorded in the gate trace.

    Returns:
        :class:`SkillRetrievalResult` — the ranked skills with their
        ``retrieval_scores`` and the diagnostics for the retrieval log.
    """
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"method must be one of {RETRIEVAL_METHODS}, not {method!r}")
    limit = constants.SKILL_RETRIEVAL_DEFAULT_LIMIT if limit is None else int(limit)
    if limit < 0:
        raise ValueError("limit must not be negative")
    query = (query or "").strip()
    owner_org: str = user.organization_id
    if min_similarity is None:
        min_similarity = constants.SKILL_RETRIEVAL_MIN_SIMILARITY
    if relative_ratio is None:
        relative_ratio = constants.SKILL_RETRIEVAL_RELATIVE_RATIO
    max_docs = constants.SKILL_BM25_MAX_DOCS
    rrf_k = constants.SKILL_HYBRID_RRF_K

    # The query vector first, outside any database session: an embedding call
    # (with its retries) never holds a pooled connection.
    vector, embedding_status, embedding_error = await _query_vector(
        query,
        wanted=method in (METHOD_HYBRID, METHOD_EMBEDDING),
        query_embedding=query_embedding,
        embed_query=embed_query,
    )
    if vector is None:
        lanes = [LANE_BM25]
    elif method == METHOD_HYBRID:
        lanes = [LANE_BM25, LANE_COSINE]
    elif method == METHOD_EMBEDDING:
        lanes = [LANE_COSINE]
    else:
        lanes = [LANE_BM25]

    def filters(stmt):
        return _candidate_filter(
            stmt,
            user_id=user.id,
            organization_id=owner_org,
            scopes=scopes,
            filter_tags=filter_tags,
        )

    async def _read_corpus() -> _Corpus:
        # Every statement of the corpus reads ONE snapshot (Postgres).
        async with session_maker() as session:
            if not query:
                return _Corpus(
                    _dialect_name(session) or BACKEND_SQLITE,
                    [],
                    [],
                    BM25CorpusStats(0, {}, 0.0),
                )
            if _uses_retrieval_tsv(session):
                return await _read_postgres(
                    session, query, vector, filters=filters, max_docs=max_docs
                )
            return await _read_sqlite(session, query, vector, filters=filters)

    corpus = await retry_call(_read_corpus, stage="skill_retrieval")

    def _score_and_rank() -> _Ranking:
        return rank_candidates(
            corpus.candidates,
            _bm25_scores(corpus),
            lanes=lanes,
            apply_gate=apply_gate,
            limit=limit,
            rrf_k=rrf_k,
            min_similarity=min_similarity,
            relative_ratio=relative_ratio,
            max_cosine_distance=max_cosine_distance,
            gate_calibration=gate_calibration,
        )

    # Scored off the event loop and with no pooled connection held.
    ranking = await asyncio.to_thread(_score_and_rank)
    returned_ids = [t.id for t in ranking.returned]

    async def _read_rows() -> Dict[str, Tuple[Any, Optional[int]]]:
        if not returned_ids:
            return {}
        from mirix.orm.skill_version import SkillVersion as SkillVersionModel

        # The version each skill is served at, in the SAME statement as its
        # row: a write changes the row and adds its version in one
        # transaction, so one statement's snapshot sees both or neither.
        newest = (
            select(func.max(SkillVersionModel.version_no))
            .where(SkillVersionModel.skill_id == ProceduralMemoryItem.id)
            .correlate(ProceduralMemoryItem)
            .scalar_subquery()
        )
        async with session_maker() as session:
            result = await session.execute(
                filters(select(ProceduralMemoryItem, newest))
                .where(ProceduralMemoryItem.id.in_(returned_ids))
                .options(lazyload("*"))
            )
            return {
                str(row.id): (row.to_pydantic(), version)
                for row, version in result.all()
            }

    rows = await retry_call(_read_rows, stage="skill_retrieval")

    skills: List[RetrievedSkill] = []
    for trace in ranking.returned:
        skill, served_version = rows.get(trace.id, (None, None))
        if skill is None:
            # Deleted (or moved out of the caller's scope) since the corpus
            # was read: never return it.
            logger.debug("Skill %s left the candidates after ranking", trace.id)
            trace.returned = False
            continue
        skills.append(
            RetrievedSkill(
                skill=skill,
                scores=SkillRetrievalScores(
                    bm25=trace.bm25, cosine=trace.cosine, rrf=trace.rrf
                ),
                version_no=int(served_version) if served_version is not None else None,
            )
        )

    if vector is None:
        effective_method = METHOD_BM25
    else:
        effective_method = method
    diagnostics = SkillRetrievalDiagnostics(
        requested_method=method,
        method=effective_method,
        lanes=ranking.lanes,
        backend=corpus.backend,
        embedding_status=embedding_status,
        embedding_error=embedding_error,
        candidate_count=len(corpus.candidates),
        missing_vector_count=sum(1 for c in corpus.candidates if not c.has_vector),
        bm25=BM25Trace(
            query_terms=list(corpus.query_terms),
            n_docs=corpus.stats.n_docs,
            avgdl=corpus.stats.avgdl,
            doc_freq=dict(corpus.stats.doc_freq),
            k1=constants.SKILL_BM25_K1,
            b=constants.SKILL_BM25_B,
            weights={
                "A": constants.SKILL_BM25_WEIGHT_A,
                "B": constants.SKILL_BM25_WEIGHT_B,
            },
            prefiltered=corpus.prefiltered,
            max_docs=max_docs,
            scored_docs=sum(1 for c in corpus.candidates if c.bm25_scored),
            query_terms_dropped=corpus.query_terms_dropped,
            max_query_terms=constants.SKILL_BM25_MAX_QUERY_TERMS,
        ),
        rrf_k=rrf_k,
        gate=ranking.gate,
        limit=limit,
        max_cosine_distance=max_cosine_distance,
        candidates=ranking.traces,
        returned_ids=[r.skill.id for r in skills],
        returned_versions={r.skill.id: r.version_no for r in skills},
    )
    return SkillRetrievalResult(skills=skills, diagnostics=diagnostics)
