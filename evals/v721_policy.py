"""QA operator policy for graph version v7.21."""

from __future__ import annotations

import re
from typing import Optional

from v720_policy import (  # re-export the unchanged v7.20 evidence helpers
    aggregation_instruction as _v720_instruction,
    deduplicate_results,
    result_evidence_key,
)


_NUMBERED_LINE_RE = re.compile(r"(?m)^[ \t]*(\d+)[.)][ \t]+(.+?)[ \t]*$")
_COUNT_BEFORE_NOUN_RE = re.compile(
    r"\b\d+\b(?=\s+(?:times?|occurrences?|events?|visits?|trips?|items?|"
    r"children|people|places|books|activities|instruments)\b)",
    re.I,
)
_NON_OCCURRENCE_RE = re.compile(
    r"(?:\bgeneral frequency\b|\bhabitual (?:statement|pattern)\b|"
    r"^another mention\b|^duplicate\b|\bsame as (?:the )?(?:above|previous|"
    r"item \d+)\b|\bnot an additional (?:event|trip|visit|occurrence)\b)",
    re.I,
)


def is_v721(version: Optional[str]) -> bool:
    return (version or "").strip().lower() == "v7.21"


def question_operator(question: str) -> str:
    low = (question or "").lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bhow much\b|\btotal\b", low):
        return "count"
    if re.search(r"\b(?:both|in common|common to|shared by)\b", low):
        return "intersection"
    if re.search(
        r"\bwhat (?:items|books|activities|things|instruments|events|kinds|types|"
        r"paintings|subjects|places|plans)\b",
        low,
    ):
        return "list_union"
    if re.search(r"\b(?:which .* first|before|after|latest|most recent|when)\b", low):
        return "temporal"
    return "single"


def aggregation_instruction(operator: str) -> str:
    if operator == "count":
        return (
            "COUNT operator: build a provisional list of real-world occurrences, then "
            "merge repeated descriptions of the same event before numbering it. A general "
            "frequency statement is context, never another occurrence. Do not put a row "
            "containing 'same event', 'duplicate', or 'another mention' in the final numbered "
            "list. Renumber only the surviving unique events; the answer must equal that "
            "final list length."
        )
    if operator == "intersection":
        return (
            "Intersect concrete graph facts by the named participants and relation; "
            "do not choose a subject supported for only one participant."
        )
    if operator == "temporal":
        return (
            "Order cited events by event time, preserve separate updates, and answer "
            "relative to the date stated in the question."
        )
    return _v720_instruction(operator)


def normalize_count_answer(answer: str) -> str:
    """Make an answer's explicit count agree with its own deduplicated list.

    This deliberately does not infer events from prose or consult a reference
    answer. It only repairs a mechanically contradictory response after the model
    has already supplied a numbered occurrence ledger. Lines that explicitly call
    themselves a frequency statement or duplicate are not occurrences.
    """

    text = str(answer or "")
    matches = list(_NUMBERED_LINE_RE.finditer(text))
    if not matches:
        return text

    kept: list[str] = []
    for match in matches:
        item = match.group(2).strip()
        if _NON_OCCURRENCE_RE.search(item):
            continue
        kept.append(item)
    if not kept:
        return text

    item_index = 0

    def replace_item(match: re.Match[str]) -> str:
        nonlocal item_index
        item = match.group(2).strip()
        if _NON_OCCURRENCE_RE.search(item):
            return ""
        item_index += 1
        return f"{item_index}. {item}"

    normalized = _NUMBERED_LINE_RE.sub(replace_item, text)
    normalized = _COUNT_BEFORE_NOUN_RE.sub(str(len(kept)), normalized)
    if len(kept) == 1:
        normalized = re.sub(r"\b1 times\b", "1 time", normalized, flags=re.I)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


__all__ = [
    "aggregation_instruction",
    "deduplicate_results",
    "is_v721",
    "normalize_count_answer",
    "question_operator",
    "result_evidence_key",
]
