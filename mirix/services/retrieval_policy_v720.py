"""Deterministic precision policy for the v7.20 graph retriever.

v7.19's AutoDream policy is intentionally untouched.  v7.20 changes only how
the already-built graph is entered and how its PG-backed evidence is ordered:

* exact query/anchor and query/predicate matches receive a small boost;
* multi-token, low-degree anchors beat generic high-degree hubs;
* person hubs receive the strongest degree penalty;
* PG rows get a lexical evidence pass after vector recall.

The vector search remains candidate generation.  These helpers only reorder a
bounded candidate set and are kept pure so the policy can be regression-tested
without Neo4j or PostgreSQL.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence, TypeVar


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "did", "do", "does",
    "for", "from", "has", "have", "her", "his", "how", "in", "is",
    "it", "make", "made", "of", "on", "she", "the", "their", "they",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with",
}


def content_tokens(text: str) -> set[str]:
    """Lower-cased content tokens used by the bounded lexical boosts."""

    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def anchor_policy_score(
    *,
    query: str,
    name: str,
    anchor_type: str,
    cosine: float,
    degree: int,
    predicates: Iterable[str] = (),
) -> float:
    """Score one vector-recalled anchor.

    Bonuses are deliberately smaller than the vector score.  They break the
    common ``Caroline``/``art`` hub tie without letting an unrelated literal
    token overwhelm semantic similarity.
    """

    q_norm = " ".join(_TOKEN_RE.findall((query or "").lower()))
    n_norm = " ".join(_TOKEN_RE.findall((name or "").lower()))
    q_tokens = content_tokens(query)
    n_tokens = content_tokens(name)
    overlap = len(q_tokens & n_tokens) / max(len(n_tokens), 1)

    score = float(cosine)
    if n_norm and n_norm in q_norm:
        score += 0.10
    score += 0.065 * overlap

    predicate_tokens = content_tokens(" ".join(str(p) for p in predicates or ()))
    if q_tokens and predicate_tokens:
        score += 0.045 * (len(q_tokens & predicate_tokens) / len(q_tokens))

    # A two/three-token occurrence anchor is normally more discriminating than
    # a one-token topic.  Cap the boost so long noisy names do not benefit.
    score += min(max(len(n_tokens) - 1, 0), 3) * 0.012

    # Degree is useful for recall but harmful as a final entry point: a person
    # node can fan out to hundreds of unrelated frames.  Penalise logarithmically
    # so moderate reuse remains fine and only true hubs move down materially.
    log_degree = math.log1p(max(int(degree or 0), 0))
    if (anchor_type or "").strip().lower() == "person":
        score -= min(0.24, 0.040 * log_degree)
    else:
        score -= min(0.10, 0.016 * log_degree)
    return score


T = TypeVar("T")


def rank_anchor_hits(query: str, hits: Sequence[T], limit: int) -> list[T]:
    """Return the best policy-scored hits, preserving vector order on ties."""

    scored = []
    for idx, hit in enumerate(hits):
        score = anchor_policy_score(
            query=query,
            name=getattr(hit, "name", ""),
            anchor_type=getattr(hit, "anchor_type", ""),
            cosine=getattr(hit, "cosine", 0.0),
            degree=getattr(hit, "degree", 0),
            predicates=getattr(hit, "predicates", ()),
        )
        setattr(hit, "policy_score", score)
        scored.append((score, -idx, hit))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[: max(0, limit)]]


def evidence_lexical_score(query: str, summary: str, details: str = "", name: str = "") -> float:
    """Lexical precision signal for a row already recalled through the graph."""

    q_tokens = content_tokens(query)
    if not q_tokens:
        return 0.0
    head_tokens = content_tokens(" ".join((name or "", summary or "")))
    all_tokens = head_tokens | content_tokens(details)
    coverage = len(q_tokens & all_tokens) / len(q_tokens)
    head_coverage = len(q_tokens & head_tokens) / len(q_tokens)
    phrase = " ".join(_TOKEN_RE.findall((query or "").lower()))
    body = " ".join(_TOKEN_RE.findall(" ".join((name, summary, details)).lower()))
    exact = 1.0 if phrase and phrase in body else 0.0
    return 0.58 * coverage + 0.32 * head_coverage + 0.10 * exact


def rank_memory_rows(query: str, rows: Sequence[T], limit: int) -> list[T]:
    """Blend vector order with lexical evidence specificity and deduplicate IDs."""

    if not rows or limit <= 0:
        return []
    total = max(len(rows), 1)
    scored = []
    seen_ids: set[str] = set()
    for idx, row in enumerate(rows):
        row_id = str(getattr(row, "id", "") or "")
        if row_id and row_id in seen_ids:
            continue
        if row_id:
            seen_ids.add(row_id)
        extra = getattr(row, "extra", {}) or {}
        lexical = evidence_lexical_score(
            query,
            getattr(row, "summary", ""),
            getattr(row, "details", ""),
            str(extra.get("name") or ""),
        )
        # Vector rank stays the majority signal.  Lexical matching selects the
        # concrete assertion among semantically similar memories.
        # SQL has already limited the rows by vector distance.  Treat position
        # inside that shortlist as a shallow prior rather than allowing row 1 to
        # beat an exact assertion merely because row 2 had a near-identical vector.
        vector_rank = 0.85 + 0.15 * (1.0 - (idx / total))
        score = 0.64 * vector_rank + 0.36 * lexical
        extra["v720_evidence_score"] = round(score, 6)
        setattr(row, "extra", extra)
        scored.append((score, -idx, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]
