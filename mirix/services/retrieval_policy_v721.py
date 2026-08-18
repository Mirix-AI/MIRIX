"""Bounded, relation-aware retrieval policy for graph version v7.21.

v7.21 keeps v7.20 ingest and v7.19 AutoDream unchanged.  It adds a second,
high-precision entry lane for questions that explicitly name an anchor and a
relation:

    exact query anchor -> predicate-filtered V7Fact -> cited PG memories

The existing vector-anchor/frame traversal remains the recall fallback.  All
candidate generation therefore still starts in Neo4j when graph memory owns a
memory kind; this module never performs an unbounded PG search.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence, TypeVar

from mirix.services.retrieval_policy_v720 import (
    anchor_policy_score,
    content_tokens,
    evidence_lexical_score,
)


_WORD_RE = re.compile(r"[a-z0-9]+")
_QUERY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "both", "by",
    "current", "date", "did", "do", "does", "for", "from", "had", "has",
    "have", "her", "his", "how", "in", "is", "it", "many", "much", "of",
    "on", "or", "she", "the", "their", "them", "they", "to", "total",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}

# Small linguistic relation families, not benchmark entities or answers.  They
# let a surface question such as "what did X buy" match canonical graph
# predicates such as purchase/own, while leaving entity and object selection to
# the graph itself.
_RELATION_FAMILIES = (
    {"buy", "purchase", "acquire", "get", "obtain"},
    {"make", "create", "paint", "draw", "build", "write", "produce"},
    {"read", "finish", "study"},
    {"recommend", "suggest", "advise"},
    {"plan", "intend", "research", "investigate", "apply", "prepare", "look"},
    {"attend", "visit", "go", "join", "participate"},
    {"live", "locate", "move", "stay", "reside"},
    {"work", "employ", "job"},
    {"like", "love", "prefer", "favorite", "enjoy"},
    {"have", "own", "include", "contain", "consist"},
    {"meet", "contact", "talk", "speak", "message"},
    {"learn", "teach", "inspire", "influence"},
    {"play", "practice", "perform"},
)

_IRREGULAR = {
    "bought": "buy", "built": "build", "did": "do", "drew": "draw",
    "felt": "feel", "found": "find", "gone": "go", "got": "get", "had": "have",
    "has": "have", "made": "make", "met": "meet", "read": "read",
    "recommendation": "recommend", "recommendations": "recommend",
    "said": "say", "suggestion": "suggest", "suggestions": "suggest",
    "sent": "send", "spoke": "speak", "taught": "teach", "thought": "think",
    "told": "tell", "went": "go", "wrote": "write",
}


def is_v721(version: str | None) -> bool:
    return (version or "").strip().lower() == "v7.21"


def _lemma(token: str) -> str:
    token = (token or "").lower()
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        if stem.endswith("i"):
            return stem[:-1] + "y"
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _lemmas(text: str) -> set[str]:
    return {_lemma(t) for t in _WORD_RE.findall((text or "").lower())}


def exact_anchor_terms(query: str, *, max_terms: int = 96) -> list[str]:
    """Generate bounded exact-name candidates without scanning user anchors.

    Neo4j has a composite ``(user_id, name_lower)`` uniqueness constraint, so
    UNWINDing these terms produces indexed lookups.  Terms are contiguous after
    dropping question glue; longer phrases are tried before single tokens.
    """

    raw = [t for t in _WORD_RE.findall((query or "").lower()) if t != "s"]
    words = [t for t in raw if t not in _QUERY_STOP]
    if not words:
        return []
    candidates: set[str] = set()
    for width in range(min(5, len(words)), 0, -1):
        for start in range(0, len(words) - width + 1):
            phrase = " ".join(words[start:start + width]).strip()
            if len(phrase) >= 2:
                candidates.add(phrase)
    return sorted(candidates, key=lambda value: (-len(value.split()), -len(value), value))[:max_terms]


def predicate_hints(query: str) -> list[str]:
    """Return a bounded relation vocabulary only when the query expresses one."""

    q = _lemmas(query)
    raw_tokens = set(_WORD_RE.findall((query or "").lower()))
    matched: list[set[str]] = []
    for family in _RELATION_FAMILIES:
        triggers = family
        if "make" in family:
            # make/create are too generic to justify opening a person hub by
            # themselves; specific creation verbs still expand to the family.
            triggers = {"paint", "draw", "build", "write", "produce"}
        elif "buy" in family:
            triggers = {"buy", "purchase", "acquire", "obtain"}
        elif "plan" in family:
            triggers = {"plan", "intend", "research", "investigate", "apply", "prepare"}
        elif "have" in family:
            # Inflected has/had are overwhelmingly auxiliaries or attribution
            # ("has gone", "traits X has"), not graph predicates.  Keep the
            # explicit base form for genuine possession questions such as
            # "What pets does X have?"; own/include/contain/consist remain
            # unambiguous triggers.
            triggers = {"have", "own", "include", "contain", "consist"} & raw_tokens
        if q & triggers:
            matched.append(family)
    # "have" is often an auxiliary ("have both painted", "has bought").
    # Retain it for genuine possession questions, but not when a more specific
    # lexical relation is present.
    if len(matched) > 1:
        matched = [family for family in matched if "have" not in family]
    hints: set[str] = set()
    for family in matched:
        hints.update(family)
    return sorted(hints)


def predicate_match_score(query: str, predicate: str) -> float:
    q = _lemmas(query)
    p = _lemmas(predicate)
    if not q or not p:
        return 0.0
    overlap = len(q & p) / len(p)
    score = 0.72 * overlap
    for family in _RELATION_FAMILIES:
        if q & family and p & family:
            score = max(score, 0.48)
    return min(score, 1.0)


def query_mentions_anchor(query: str, name: str) -> bool:
    q = " ".join(_WORD_RE.findall((query or "").lower()))
    n = " ".join(_WORD_RE.findall((name or "").lower()))
    return bool(n and re.search(r"(?:^| )" + re.escape(n) + r"(?: |$)", q))


T = TypeVar("T")


def rank_anchor_hits(query: str, hits: Sequence[T], limit: int) -> list[T]:
    """v7.20 scoring, except explicit query entities are protected as entry points."""

    scored = []
    relation_query = bool(predicate_hints(query))
    for idx, hit in enumerate(hits):
        anchor_type = str(getattr(hit, "anchor_type", "") or "").lower()
        degree = int(getattr(hit, "degree", 0) or 0)
        score = anchor_policy_score(
            query=query,
            name=getattr(hit, "name", ""),
            anchor_type=anchor_type,
            cosine=getattr(hit, "cosine", 0.0),
            degree=degree,
            predicates=getattr(hit, "predicates", ()),
        )
        if relation_query and query_mentions_anchor(query, getattr(hit, "name", "")):
            score += 0.14
            if anchor_type == "person":
                # Restore the logarithmic v7.20 person-hub penalty only for an
                # entity explicitly named by the question, then give it a small
                # deterministic entry-point bonus.
                score += min(0.24, 0.040 * math.log1p(max(degree, 0))) + 0.08
        setattr(hit, "policy_score", score)
        scored.append((score, -idx, hit))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max(0, limit)]]


def rank_relation_facts(
    query: str,
    facts: Sequence[T],
    *,
    exact_anchor_ids: Iterable[str],
    limit: int = 8,
) -> list[T]:
    """Rank already-bounded adjacent facts by relation and entity-role coverage."""

    exact_ids = set(exact_anchor_ids)
    shared_tokens: set[str] = set()
    intersection = bool(re.search(r"\b(?:both|in common|common to|shared by)\b", query, re.I))
    if intersection:
        token_owners: dict[str, set[str]] = {}
        for fact in facts:
            args = list(getattr(fact, "args", ()) or ())
            covered = {str(arg.get("id")) for arg in args if str(arg.get("id")) in exact_ids}
            for arg in args:
                if str(arg.get("id")) in exact_ids:
                    continue
                if str(arg.get("role") or "").lower() not in {
                    "patient", "theme", "content", "object", "subject", "item",
                }:
                    continue
                for token in content_tokens(str(arg.get("name") or "")):
                    if token not in {"art", "painting", "picture", "work", "thing"}:
                        token_owners.setdefault(token, set()).update(covered)
        shared_tokens = {
            token for token, owners in token_owners.items() if len(owners) >= 2
        }

    scored = []
    for idx, fact in enumerate(facts):
        pred_score = predicate_match_score(query, getattr(fact, "predicate", ""))
        if pred_score <= 0:
            continue
        args = list(getattr(fact, "args", ()) or ())
        covered = {str(arg.get("id")) for arg in args if str(arg.get("id")) in exact_ids}
        entity_bonus = 0.08 + min(0.18, 0.06 * len(covered))
        specificity = min(0.06, 0.012 * max(len(args) - 1, 0))
        object_tokens = set()
        for arg in args:
            if (
                str(arg.get("id")) not in exact_ids
                and str(arg.get("role") or "").lower() in {
                    "patient", "theme", "content", "object", "subject", "item",
                }
            ):
                object_tokens.update(content_tokens(str(arg.get("name") or "")))
        intersection_bonus = 0.26 if object_tokens & shared_tokens else 0.0
        score = pred_score + entity_bonus + specificity + intersection_bonus
        setattr(fact, "policy_score", score)
        scored.append((score, len(covered), -idx, fact))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    if limit <= 0:
        return []
    # Reserve up to two slots for a distinct synonymous predicate.  This keeps
    # exact surface predicates from crowding every related candidate out (e.g.
    # plan(...) hiding research(...)), without globally expanding the result set.
    primary_count = max(1, limit - 2)
    selected = list(scored[:primary_count])
    selected_ids = {item[3].id for item in selected}
    selected_predicates = {str(item[3].predicate).lower() for item in selected}
    for item in scored[primary_count:]:
        predicate = str(item[3].predicate).lower()
        if predicate not in selected_predicates:
            selected.append(item)
            selected_ids.add(item[3].id)
            selected_predicates.add(predicate)
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        for item in scored:
            if item[3].id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item[3].id)
            if len(selected) >= limit:
                break
    return [item[3] for item in selected]


def rank_memory_rows(
    query: str,
    rows: Sequence[T],
    limit: int,
    *,
    priority_ids: Iterable[str] = (),
) -> list[T]:
    """v7.20 lexical rerank with a bounded boost for relation-cited memories."""

    if not rows or limit <= 0:
        return []
    priority = {str(value) for value in priority_ids if value}
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
        vector_rank = 0.85 + 0.15 * (1.0 - (idx / total))
        score = 0.64 * vector_rank + 0.36 * lexical
        if row_id in priority:
            score += 0.16
            extra["v721_relation_cited"] = True
        extra["v721_evidence_score"] = round(score, 6)
        setattr(row, "extra", extra)
        scored.append((score, -idx, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]
