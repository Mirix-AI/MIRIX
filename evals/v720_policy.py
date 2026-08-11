"""LoCoMo ingest and QA evidence policy for graph version v7.20."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional


_RELATIVE_PATTERNS = (
    re.compile(r"\b(?P<n>one|two|three|four|five|six|seven|\d+) days? ago\b", re.I),
    re.compile(r"\b(?P<n>one|two|three|four|five|six|seven|\d+) weeks? ago\b", re.I),
    re.compile(r"\b(?:last|the previous) (?:year|week|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I),
    re.compile(r"\b(?:yesterday|a few weeks ago|past weekend|this past weekend)\b", re.I),
)
_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_TEXT_IMAGE_HINTS = {
    "book", "book cover", "card", "document", "label", "letter", "menu",
    "poster", "screen", "sign", "text", "writing",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def is_v720(version: Optional[str]) -> bool:
    # Later retrieval/evidence revisions deliberately inherit v7.20's multimodal,
    # temporal, and evidence-ledger input adapter.
    return (version or "").strip().lower() in {
        "v7.20", "v7.21", "v7.22", "v7.23", "v7.24"
    }


def parse_session_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value.strip())
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"):
        try:
            return datetime.strptime(cleaned.upper(), fmt)
        except ValueError:
            continue
    return None


def mentioned_at_iso(value: Optional[str]) -> Optional[str]:
    parsed = parse_session_datetime(value)
    return parsed.isoformat() if parsed else None


def _previous_weekend(reference: datetime) -> tuple[datetime, datetime]:
    days_since_sunday = (reference.weekday() - 6) % 7
    if days_since_sunday == 0:
        days_since_sunday = 7
    sunday = reference - timedelta(days=days_since_sunday)
    saturday = sunday - timedelta(days=1)
    return saturday, sunday


def _resolve_phrase(phrase: str, reference: datetime) -> str:
    low = phrase.lower()
    if low == "yesterday":
        return (reference - timedelta(days=1)).date().isoformat()
    if "few weeks ago" in low:
        # The language is intentionally imprecise; preserve that uncertainty.
        start = (reference - timedelta(weeks=4)).date().isoformat()
        end = (reference - timedelta(weeks=2)).date().isoformat()
        return f"approximately {start}..{end}"
    if "weekend" in low:
        start, end = _previous_weekend(reference)
        return f"{start.date().isoformat()}..{end.date().isoformat()}"
    if low.endswith("year"):
        return str(reference.year - 1)
    if low.endswith("week"):
        end = reference.date() - timedelta(days=1)
        start = end - timedelta(days=6)
        return f"{start.isoformat()}..{end.isoformat()}"
    for day, weekday in _WEEKDAYS.items():
        if low.endswith(day):
            delta = (reference.weekday() - weekday) % 7
            if delta == 0:
                delta = 7
            return (reference - timedelta(days=delta)).date().isoformat()
    match = re.search(r"(one|two|three|four|five|six|seven|\d+) (days?|weeks?) ago", low)
    if match:
        count = _NUMBERS.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 0)
        unit_days = 7 if match.group(2).startswith("week") else 1
        return (reference - timedelta(days=count * unit_days)).date().isoformat()
    return "unresolved"


def relative_time_annotations(text: str, session_datetime: Optional[str]) -> list[dict[str, str]]:
    """Preserve each relative phrase and add a deterministic event-time hint."""

    reference = parse_session_datetime(session_datetime)
    if reference is None:
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in _RELATIVE_PATTERNS:
        for match in pattern.finditer(text or ""):
            phrase = match.group(0)
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"phrase": phrase, "event_time_hint": _resolve_phrase(phrase, reference)})
    return found


def needs_selective_ocr(turn: dict[str, Any]) -> bool:
    if not turn.get("img_url"):
        return False
    haystack = " ".join((str(turn.get("blip_caption") or ""), str(turn.get("query") or ""))).lower()
    return any(
        re.search(r"\b" + re.escape(hint).replace(r"\ ", r"\s+") + r"\b", haystack)
        for hint in _TEXT_IMAGE_HINTS
    )


def format_visual_evidence(turn: dict[str, Any], vision_text: Optional[str] = None) -> list[str]:
    """Render model-observed visual evidence separately from retrieval metadata."""

    dia_id = str(turn.get("dia_id") or "unknown")
    lines: list[str] = []
    caption = str(turn.get("blip_caption") or "").strip()
    if caption:
        lines.append(f"[Visual evidence {dia_id}; BLIP caption]: {caption}")
    if vision_text and vision_text.strip():
        lines.append(f"[Visual evidence {dia_id}; selective OCR/vision]: {vision_text.strip()}")
    query = str(turn.get("query") or "").strip()
    if query:
        lines.append(
            f"[Image retrieval metadata {dia_id}; use only as a search hint, not direct evidence]: {query}"
        )
    return lines


def question_operator(question: str) -> str:
    low = (question or "").lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bhow much\b|\btotal\b", low):
        return "count"
    if re.search(r"\bwhat (?:items|books|activities|things|instruments|events|kinds|types)\b", low):
        return "list_union"
    return "single"


def _normalise_text(value: str) -> str:
    return " ".join(_WORD_RE.findall((value or "").lower()))


def result_evidence_key(result: dict[str, Any]) -> str:
    """Stable cross-search key; prefer PG identity, then conservative content identity."""

    identity = result.get("id") or result.get("_evidence_id")
    if identity:
        return str(identity)
    kind = str(result.get("memory_type") or "")
    timestamp = str(result.get("timestamp") or result.get("occurred_at") or "")[:10]
    text = _normalise_text(" ".join((
        str(result.get("name") or ""),
        str(result.get("summary") or ""),
        str(result.get("details") or ""),
    )))
    digest = hashlib.sha1(f"{kind}|{timestamp}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"ev_{digest}"


def deduplicate_results(
    results: Iterable[dict[str, Any]], seen: Optional[set[str]] = None
) -> tuple[list[dict[str, Any]], set[str]]:
    """Deduplicate repeated tool hits without collapsing distinct occurrences."""

    seen = seen if seen is not None else set()
    unique: list[dict[str, Any]] = []
    for raw in results:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        key = result_evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        item["evidence_id"] = key
        if str(item.get("memory_type") or "") == "episodic":
            item["occurrence_key"] = " | ".join(filter(None, (
                str(item.get("timestamp") or item.get("occurred_at") or "")[:10],
                _normalise_text(str(item.get("summary") or ""))[:100],
            )))
        unique.append(item)
    return unique, seen


def aggregation_instruction(operator: str) -> str:
    if operator == "count":
        return (
            "COUNT operator: count distinct real-world occurrences, not memory rows. "
            "Use episodic event_time + event + participants as the identity; semantic rows "
            "only corroborate an occurrence. Repeated descriptions count once."
        )
    if operator == "list_union":
        return (
            "LIST_UNION operator: take the union of concrete answer items across all "
            "searches, canonicalize paraphrases, and emit every distinct supported item once."
        )
    return "Select the most concrete assertion that directly answers the question."
