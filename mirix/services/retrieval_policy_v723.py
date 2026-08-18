"""Adaptive evidence policy for graph v7.23.

v7.23 keeps v7.22's bounded graph-owned candidate generation and ranking.  It
adds a small, benchmark-independent answer verification contract used by clients
to decide whether a single targeted graph retry is warranted.  The verifier never
opens a flat-index path and never uses reference answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Iterable

from mirix.services.retrieval_policy_v722 import (
    QueryPlan,
    canonical_predicate_tokens,
    exact_anchor_terms,
    query_mentions_anchor,
)
from mirix.services import retrieval_policy_v722 as _v722


_WORD_RE = re.compile(r"[a-z0-9]+")
_LIST_ANSWER_TYPES = {
    "artist", "artists", "band", "bands", "name", "names", "symbol",
    "symbols", "trait", "traits",
}
_SYMBOL_RELATIONS = {
    "mean", "meaning", "meaningful", "represent", "signify", "symbolize", "value",
}
_MUSIC_EXPOSURE_RELATIONS = {
    "attend", "feature", "hear", "listen", "perform", "play", "watch",
}
_MUSIC_OBJECT_WORDS = {
    "artist", "artists", "band", "bands", "concert", "concerts", "music",
    "musical", "musician", "musicians", "performance", "performances", "show", "shows",
}
_POSSESSION_FORMS = {
    "consist", "consisted", "consists", "contain", "contained", "contains",
    "had", "has", "have", "include", "included", "includes", "own", "owned",
    "owns", "possess",
}


def plan_query(query: str) -> QueryPlan:
    """Extend v7.22 only for high-confidence, ordinary QA constructions."""

    base = _v722.plan_query(query)
    tokens = set(_WORD_RE.findall((query or "").lower()))
    relations = set(base.relations)
    forms = set(base.predicate_forms)
    answer_types = set(base.answer_types)

    # In "What subject have A and B both painted?", ``have`` is an auxiliary,
    # not a requested possession relation. Keeping both relation families lets a
    # two-person have(friendship) fact outrank the two one-person paint(subject)
    # facts that actually establish the intersection. Whenever a more specific
    # lexical relation already opened the lane, possession is the weak auxiliary.
    if "possess" in relations and len(relations) > 1:
        relations.discard("possess")
        forms.difference_update(_POSSESSION_FORMS)

    if tokens & {"symbol", "symbols", "symbolize", "symbolized", "represent", "represents"}:
        relations.update(_SYMBOL_RELATIONS)
        forms.update(_SYMBOL_RELATIONS)
    if tokens & {"see", "seen", "saw"} and tokens & _MUSIC_OBJECT_WORDS:
        relations.update(_MUSIC_EXPOSURE_RELATIONS)
        forms.update(_MUSIC_EXPOSURE_RELATIONS)
        forms.update({"see", "seen", "saw"})

    answer_types.update(tokens & _LIST_ANSWER_TYPES)
    operator = base.operator
    if operator == "single" and answer_types & _LIST_ANSWER_TYPES:
        operator = "list_union"

    budgets = {
        "single": (8, 10, 5),
        "temporal": (12, 9, 6),
        "intersection": (16, 9, 7),
        "list_union": (20, 10, 10),
        "count": (24, 10, 14),
    }
    fact_limit, base_quota, relation_quota = budgets[operator]
    constraints = set(base.constraint_tokens)
    constraints.difference_update(forms)
    constraints.difference_update(answer_types)
    return replace(
        base,
        operator=operator,
        relations=tuple(sorted(relations)),
        predicate_forms=tuple(sorted(forms)),
        constraint_tokens=tuple(sorted(constraints)),
        answer_types=tuple(sorted(answer_types)),
        fact_limit=fact_limit,
        base_quota=base_quota,
        relation_quota=relation_quota,
    )


def predicate_hints(query: str) -> list[str]:
    return list(plan_query(query).predicate_forms)


def predicate_match_score(query: str, predicate: str) -> float:
    return _v722.predicate_match_score(query, predicate, planner=plan_query)


def rank_anchor_hits(query, hits, limit):
    return _v722.rank_anchor_hits(query, hits, limit, planner=plan_query)


def rank_relation_facts(query, facts, *, exact_anchor_ids, limit=None):
    plan = plan_query(query)
    ranked = _v722.rank_relation_facts(
        query,
        facts,
        exact_anchor_ids=exact_anchor_ids,
        # Keep the bounded Neo4j candidate set long enough for the v7.23
        # intersection pass below; v7.22's generic "painting" overlap otherwise
        # consumes the small output quota before the concrete shared subject.
        limit=(len(facts) if plan.operator == "intersection" else limit),
        planner=plan_query,
    )
    if plan.operator != "intersection":
        return ranked

    exact_ids = {str(value) for value in exact_anchor_ids}
    generic = {
        "art", "create", "creation", "item", "memory", "memories", "paint",
        "painting", "paintings", "picture", "piece", "project", "subject",
        "thing", "work",
    }
    object_roles = {"content", "item", "object", "patient", "subject", "theme"}
    query_tokens = set(_WORD_RE.findall((query or "").lower()))
    strict_paint = bool(query_tokens & {"paint", "painted", "painting", "paints"})
    token_owners: dict[str, set[str]] = {}
    for fact in facts:
        predicate_tokens = set(_WORD_RE.findall(str(getattr(fact, "predicate", "")).lower()))
        if strict_paint and "paint" not in predicate_tokens:
            continue
        args = list(getattr(fact, "args", ()) or ())
        owners = {
            str(arg.get("id")) for arg in args if str(arg.get("id")) in exact_ids
        }
        for arg in args:
            if str(arg.get("id")) in exact_ids:
                continue
            if str(arg.get("role") or "").lower() not in object_roles:
                continue
            for token in _v722.content_tokens(str(arg.get("name") or "")):
                if token not in generic:
                    token_owners.setdefault(token, set()).update(owners)
    shared = {token for token, owners in token_owners.items() if len(owners) >= 2}

    def intersection_score(fact) -> tuple[float, float]:
        object_tokens: set[str] = set()
        predicate_tokens = set(
            _WORD_RE.findall(str(getattr(fact, "predicate", "")).lower())
        )
        for arg in list(getattr(fact, "args", ()) or ()):
            if (
                str(arg.get("id")) not in exact_ids
                and str(arg.get("role") or "").lower() in object_roles
            ):
                object_tokens.update(_v722.content_tokens(str(arg.get("name") or "")))
        strict_relation = not strict_paint or "paint" in predicate_tokens
        concrete_bonus = 0.50 if strict_relation and object_tokens & shared else 0.0
        # Expose the constraint group so later policies can add soft role/time
        # scores without accidentally flattening away the hard intersection.
        setattr(fact, "v723_intersection_bonus", concrete_bonus)
        return concrete_bonus + float(getattr(fact, "policy_score", 0.0) or 0.0), concrete_bonus

    ranked.sort(key=intersection_score, reverse=True)
    chosen_limit = plan.fact_limit if limit is None else max(0, limit)
    return ranked[:chosen_limit]


def merge_memory_rows(query, base_rows, relation_rows, limit):
    return _v722.merge_memory_rows(
        query, base_rows, relation_rows, limit, planner=plan_query
    )


_NUMBER_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|\d+(?:\.\d+)?)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|today|yesterday|tomorrow|last|next|"
    r"before|after|ago|week|weekend|month|year)\b",
    re.I,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(?:not mentioned|not specified|no (?:specific )?(?:information|record|"
    r"evidence)|cannot determine|can't determine|unclear|unknown|does not say|"
    r"doesn't say|do not know|don't know)\b",
    re.I,
)
_NUMBERED_LINE_RE = re.compile(r"(?m)^\s*(\d+)[.)]\s+(.+?)\s*$")


@dataclass(frozen=True)
class VerificationDecision:
    retry: bool
    reasons: tuple[str, ...]
    query: str


def is_v723(version: str | None) -> bool:
    return (version or "").strip().lower() == "v7.23"


def verification_query(question: str, plan: QueryPlan | None = None) -> str:
    """Build one conservative retry query without guessing an answer."""

    plan = plan or plan_query(question)
    additions: list[str] = []
    additions.extend(plan.relations)
    additions.extend(plan.answer_types)
    additions.extend(plan.constraint_tokens)
    seen = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    suffix: list[str] = []
    for token in additions:
        clean = str(token).strip().lower()
        if clean and clean not in seen and clean not in suffix:
            suffix.append(clean)
    return " ".join(filter(None, ((question or "").strip(), " ".join(suffix[:8]))))


def _ledger_text(evidence: Iterable[dict[str, Any]] | None) -> str:
    chunks: list[str] = []
    for item in evidence or ():
        if not isinstance(item, dict):
            continue
        chunks.extend(
            str(item.get(key) or "")
            for key in ("name", "predicate", "summary", "details", "timestamp")
        )
    return " ".join(chunks)


def verification_decision(
    question: str,
    answer: str,
    evidence: Iterable[dict[str, Any]] | None = None,
) -> VerificationDecision:
    """Return whether a single graph retry is justified by an observable defect.

    The checks are deliberately high precision.  They detect explicit uncertainty,
    missing count/time answer shapes, and mechanical count contradictions.  They do
    not reject a plausible inference merely because its wording is absent verbatim
    from evidence.
    """

    plan = plan_query(question)
    text = str(answer or "").strip()
    reasons: list[str] = []
    if not text:
        reasons.append("empty_answer")
    elif _UNCERTAINTY_RE.search(text):
        reasons.append("explicit_uncertainty")

    if plan.operator == "count":
        if not _NUMBER_RE.search(text):
            reasons.append("count_without_number")
        numbered = list(_NUMBERED_LINE_RE.finditer(text))
        if numbered:
            stated = _NUMBER_RE.search(text)
            if stated and stated.group(0).isdigit():
                first = int(stated.group(0))
                # Ignore the first match when it is simply item "1."; only flag a
                # declared count that contradicts a completed numbered ledger.
                declared = re.search(
                    r"\b(\d+)\s+(?:times?|occurrences?|events?|visits?|trips?|"
                    r"items?|children|people|places|books|activities|instruments)\b",
                    text,
                    re.I,
                )
                if declared and int(declared.group(1)) != len(numbered):
                    reasons.append("count_ledger_mismatch")

    if plan.operator == "temporal" and not _TEMPORAL_RE.search(text):
        reasons.append("temporal_answer_without_time")

    if plan.operator == "intersection":
        reasons.append("intersection_constraint_review")

    # If the graph returned nothing at all, a confident answer is still unsupported.
    if evidence is not None and not _ledger_text(evidence).strip():
        reasons.append("empty_evidence_ledger")

    unique = tuple(dict.fromkeys(reasons))
    return VerificationDecision(
        retry=bool(unique),
        reasons=unique,
        query=verification_query(question, plan),
    )


__all__ = [
    "QueryPlan",
    "VerificationDecision",
    "canonical_predicate_tokens",
    "exact_anchor_terms",
    "is_v723",
    "merge_memory_rows",
    "plan_query",
    "predicate_hints",
    "predicate_match_score",
    "query_mentions_anchor",
    "rank_anchor_hits",
    "rank_relation_facts",
    "verification_decision",
    "verification_query",
]
