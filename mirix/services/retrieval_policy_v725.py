"""v7.25 — one fused ordering key instead of two independently-ranked lanes.

WHY, measured. Bucket C of the error partition — 90 of 169 wrong answers — is
"the supporting text was returned and the answer is still wrong". Splitting it by
WHERE the supporting row sat, with correct answers as a control:

    support at rank 1     wrong 30.1%   correct 69.7%
    support in top-3      wrong 53.4%   correct 79.6%

Presence is not the binding property; position is. And the obvious first lever is
already spent: re-ranking the same rows on details_embedding instead of
summary_embedding, or on the max or mean of the two, moves rank-1 by at most 3.3pp
(44.3% -> 47.6%). Text similarity is exhausted.

What is NOT spent is the structural score the graph already computes and then declines
to use as an ordering key. rank_relation_facts scores each fact by role alignment and
temporal state; v7.22-v7.24 then deliver those facts as a separate LANE, merged with
the text-ranked rows by fixed quota. Two consequences follow:

  * the lanes are never scored against each other, so a weakly-matched structural row
    takes a guaranteed slot ahead of a strongly-matched textual one, and vice versa;
  * the structural lane only exists when the query parses into a relation, and 41.4%
    of questions produce no relation span at all.

v7.25 scores every row ONCE, on a key that carries both terms, and sorts.

MEASURED RESULT: NULL. Do not ship this expecting an effect.

    version   support found   rank-1   top-3
    v7.23        73/159       32.9%    52.1%
    v7.24        73/159       31.5%    54.8%
    v7.25        73/159       32.9%    54.8%

Indistinguishable, and the reason invalidates the premise above. The rows that receive
the structural bonus ARE the relation lane, and v7.22's quota already places that lane
first. The score was never "discarded before ordering" — it was applied coarsely, by
reserving slots, instead of smoothly, by contributing to a key. Replacing coarse with
smooth re-derives the same order.

What this rules out is worth more than what it delivers: a smoother application of the
EXISTING structural signal has no headroom. Moving bucket C needs a signal the quota
does not already encode — one that discriminates among the rows a matching fact cites,
or one that exists for the 38% of wrong questions that do not parse as relation queries
at all and therefore have no structural lane to promote.

The code is kept because the fused key is the right shape for such a signal to enter,
and because the delegation and version-resolution fixes made along the way are real.

DESIGN CONSTRAINT that shapes the weighting: 83.8% of questions currently succeed, and
the structural term must not disturb them. So the term is strictly ADDITIVE and zero in
the absence of structural evidence — a row no fact cites scores exactly what it scores
today, which means the 41.4% of questions with no relation span reduce to the v7.24
ordering exactly rather than approximately.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from mirix.services import retrieval_policy_v722 as _v722
from mirix.services import retrieval_policy_v724 as _v724
from mirix.services.retrieval_policy_v724 import (  # noqa: F401  (re-exported)
    QueryPlan,
    canonical_predicate_tokens,
    plan_query,
    predicate_hints,
    query_mentions_anchor,
    rank_anchor_hits,
    rank_relation_facts,
    role_alignment_score,
)

# Weight on the structural term relative to the text score, which _rank_rows already
# produces on a roughly [0, 1] scale. This is the one free parameter and it is meant to
# be swept with evals/rank_audit.py — minutes per setting, no ingest, no answerer — not
# guessed once and shipped. 0.30 is the starting point: large enough that a strong
# structural match can overtake a mid-ranked textual one, small enough that it cannot
# by itself lift a row with no textual support at all.
STRUCTURAL_WEIGHT = 0.30

# Fact policy scores are unbounded in principle; in practice rank_relation_facts
# produces values around [-1, 1.5]. Squash rather than clip so that the ordering among
# strong matches is preserved instead of being flattened at the ceiling.
def _squash(value: float) -> float:
    """Monotone map into [0, 1]; 0 stays 0 so 'no evidence' is exactly neutral."""
    if value <= 0.0:
        return 0.0
    return value / (1.0 + value)


def fuse_scores(
    query: str,
    rows: Sequence[Any],
    fact_scores: Mapping[str, float] | None,
) -> list[Any]:
    """Order rows by text score plus a structural bonus, in one key.

    ``fact_scores`` maps a PG memory id to the best policy_score among the facts that
    cite it. A row absent from the mapping receives exactly zero structural bonus, so
    its position is decided by the same text score v7.24 used.
    """
    ranked = _v722._rank_rows(query, rows)          # sets extra["v722_evidence_score"]
    scored: list[tuple[float, int, Any]] = []
    for idx, row in enumerate(ranked):
        extra = getattr(row, "extra", {}) or {}
        text = float(extra.get("v722_evidence_score") or 0.0)
        row_id = str(getattr(row, "id", "") or "")
        raw = float((fact_scores or {}).get(row_id, 0.0) or 0.0)
        structural = _squash(raw)
        fused = text + STRUCTURAL_WEIGHT * structural
        extra["v725_structural"] = round(structural, 6)
        extra["v725_fused_score"] = round(fused, 6)
        setattr(row, "extra", extra)
        scored.append((fused, -idx, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def merge_memory_rows(
    query: str,
    base_rows: Sequence[Any],
    relation_rows: Sequence[Any],
    limit: int,
    *,
    fact_scores: Mapping[str, float] | None = None,
) -> list[Any]:
    """One ranked list, not two lanes stitched by quota.

    Falls back to v7.24's quota merge when there is no structural signal at all, so a
    query the planner cannot parse into a relation behaves exactly as it does today
    rather than merely similarly. That matters: the point of the change is to move the
    questions that HAVE structural evidence without touching the ones that do not.
    """
    if limit <= 0:
        return []
    if not fact_scores:
        return _v724.merge_memory_rows(query, base_rows, relation_rows, limit)

    # The base quota is a RECALL GUARANTEE, not an ordering preference: ranking both
    # lanes purely by fused score would let the structural lane take every slot and
    # squeeze out ordinary traversal recall, which is what v7.22's own comment says the
    # quota is there to protect. So the fusion decides ORDER and a floor still decides
    # how many base-lane rows survive.
    #
    # (The 73/159 -> 31/159 collapse first seen on this version was NOT this — it was
    # a missing re-export crashing the graph pass. See __getattr__ below. The floor is
    # kept because the hazard it guards against is real, not because it fixed that.)
    plan = plan_query(query)
    base_floor = min(limit, plan.base_quota)

    base_ordered = fuse_scores(query, list(base_rows), fact_scores)
    rel_ordered = fuse_scores(query, list(relation_rows), fact_scores)
    merged = sorted(
        base_ordered + rel_ordered,
        key=lambda r: float((getattr(r, "extra", {}) or {}).get("v725_fused_score") or 0.0),
        reverse=True)
    base_ids = {str(getattr(r, "id", "") or "") for r in base_ordered}

    selected: list[Any] = []
    seen: set[str] = set()
    from_base = 0

    def admit(row) -> bool:
        row_id = str(getattr(row, "id", "") or "")
        if row_id and row_id in seen:
            return False
        if any(_v724._near_duplicate(row, prior) for prior in selected):
            return False
        if row_id:
            seen.add(row_id)
        selected.append(row)
        return True

    # Pass 1: fused order, but stop admitting non-base rows once doing so would make
    # the base floor unreachable in the slots that remain.
    for row in merged:
        if len(selected) >= limit:
            break
        is_base = str(getattr(row, "id", "") or "") in base_ids
        remaining = limit - len(selected)
        if not is_base and (base_floor - from_base) >= remaining:
            continue
        if admit(row) and is_base:
            from_base += 1
    # Pass 2: fill any slack left by dedup, base first to honour the floor.
    for row in base_ordered + merged:
        if len(selected) >= limit:
            break
        admit(row)
    return selected[:limit]


def fact_scores_from_rows(relation_rows: Sequence[Any]) -> dict[str, float]:
    """memory id -> best citing fact's policy score.

    The synthetic ``graph_fact`` rows carry both the score and the ids they cite, so
    the mapping is recoverable without re-running the fact ranking.
    """
    out: dict[str, float] = {}
    for row in relation_rows or ():
        extra = getattr(row, "extra", {}) or {}
        score = extra.get("v722_fact_score")
        if score is None:
            score = extra.get("v721_fact_score")
        if score is None:
            continue
        for mid in extra.get("citation_memory_ids") or ():
            key = str(mid)
            if out.get(key, float("-inf")) < float(score):
                out[key] = float(score)
    return out


def __getattr__(name: str):
    """Anything v7.25 does not redefine comes from v7.24.

    The hand-written re-export list this replaces was incomplete, and the failure was
    silent: the retriever pulls several helpers off the resolved policy module, one of
    them (`exact_anchor_terms`) was missing, and the AttributeError was swallowed by the
    try/except around the search graph pass. Relation queries then returned no graph
    rows at all — measured as the share of wrong answers whose supporting row came back
    dropping from 73/159 to 31/159, which looked like a ranking regression and was
    actually a crash. Delegation means a future policy version only has to define what
    it actually changes.
    """
    try:
        return getattr(_v724, name)
    except AttributeError as exc:  # noqa: BLE001
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}, and neither does "
            f"{_v724.__name__!r}") from exc


__all__ = [
    "QueryPlan",
    "STRUCTURAL_WEIGHT",
    "canonical_predicate_tokens",
    "fact_scores_from_rows",
    "fuse_scores",
    "merge_memory_rows",
    "plan_query",
    "predicate_hints",
    "query_mentions_anchor",
    "rank_anchor_hits",
    "rank_relation_facts",
    "role_alignment_score",
]
