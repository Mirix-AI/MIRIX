"""QA adapter for the general graph v7.24 evidence policy."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Optional

from v720_policy import result_evidence_key
from v721_policy import normalize_count_answer
from mirix.services.retrieval_policy_v724 import plan_query, verification_decision


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "had", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "they", "this", "to", "was", "were", "with",
}


def is_v724(version: Optional[str]) -> bool:
    return (version or "").strip().lower() == "v7.24"


def question_operator(question: str) -> str:
    return plan_query(question).operator


def aggregation_instruction(operator: str) -> str:
    if operator == "count":
        return (
            "COUNT operator: count distinct completed real-world occurrences unless "
            "the question explicitly asks about plans. Use event_time + predicate + "
            "participants as identity, obey any as-of cutoff, and treat semantic rows "
            "as corroboration rather than another occurrence."
        )
    if operator == "list_union":
        return (
            "LIST_UNION operator: union concrete cited items, obey role and as-of "
            "constraints, and canonicalize duplicate descriptions without merging "
            "separately timed occurrences."
        )
    if operator == "temporal":
        return (
            "TEMPORAL operator: use event_time rather than mentioned_at, preserve "
            "planned versus completed status, and apply before/after/as-of constraints."
        )
    if operator == "intersection":
        return (
            "INTERSECTION operator: require the same canonical item to be supported "
            "for every named participant with matching argument roles."
        )
    return "Select the most concrete cited assertion whose argument roles match the question."


def _tokens(item: dict[str, Any]) -> set[str]:
    text = " ".join((
        str(item.get("summary") or ""), str(item.get("details") or "")
    ))
    return {token for token in _WORD_RE.findall(text.lower()) if token not in _STOP}


def _date(item: dict[str, Any]) -> str:
    return str(item.get("timestamp") or item.get("occurred_at") or "")[:10]


def _near_duplicate_episode(
    item: dict[str, Any], occurrence_ledger: Iterable[tuple[str, frozenset[str]]]
) -> bool:
    if str(item.get("memory_type") or "") != "episodic":
        return False
    date = _date(item)
    tokens = _tokens(item)
    if not date or len(tokens) < 5:
        return False
    for prior_date, prior_tokens in occurrence_ledger:
        if date != prior_date or len(prior_tokens) < 5:
            continue
        overlap = len(tokens & set(prior_tokens))
        if overlap / len(tokens | set(prior_tokens)) >= 0.78:
            return True
        if overlap / min(len(tokens), len(prior_tokens)) >= 0.90:
            return True
    return False


def deduplicate_results(
    results: Iterable[dict[str, Any]],
    seen: Optional[set[str]] = None,
    occurrence_ledger: Optional[list[tuple[str, frozenset[str]]]] = None,
) -> tuple[list[dict[str, Any]], set[str], list[tuple[str, frozenset[str]]]]:
    """Conservative event-level dedupe across bounded graph searches."""

    seen = seen if seen is not None else set()
    occurrence_ledger = occurrence_ledger if occurrence_ledger is not None else []
    unique: list[dict[str, Any]] = []
    for raw in results:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        evidence_key = result_evidence_key(item)
        if evidence_key in seen or _near_duplicate_episode(item, occurrence_ledger):
            continue
        seen.add(evidence_key)
        item["evidence_id"] = evidence_key
        if str(item.get("memory_type") or "") == "episodic":
            date = _date(item)
            tokens = frozenset(_tokens(item))
            if date and tokens:
                occurrence_ledger.append((date, tokens))
                digest = hashlib.sha1(
                    (date + "|" + " ".join(sorted(tokens))).encode("utf-8")
                ).hexdigest()[:16]
                item["occurrence_key"] = f"{date} | evt_{digest}"
        elif str(item.get("memory_type") or "") == "semantic":
            item["occurrence_role"] = "corroboration"
        unique.append(item)
    return unique, seen, occurrence_ledger


__all__ = [
    "aggregation_instruction",
    "deduplicate_results",
    "is_v724",
    "normalize_count_answer",
    "question_operator",
    "verification_decision",
]
