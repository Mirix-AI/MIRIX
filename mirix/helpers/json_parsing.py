"""Generic, agent-agnostic JSON-extraction helpers for LLM replies.

These are pure, dependency-free utilities for robustly recovering a JSON object
or array from a model reply that may wrap it in a ```json fenced block, plain
fences, or surrounding prose.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple


def _try_load_object(text: str) -> Optional[Dict]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _try_load_array(text: str) -> Optional[List[Dict]]:
    """Parse ``text`` as a JSON array, returning only its dict elements.

    Returns ``None`` when ``text`` is not a JSON array (so callers can fall
    through to the next strategy), but ``[]`` for a genuinely empty array.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, list):
        return None
    return [el for el in obj if isinstance(el, dict)]


#: The balanced-span scan reads at most this many characters per character of
#: the text (plus :data:`SPAN_SCAN_MIN_CHARS`); past it, the starts not reached
#: yet yield no span. Each start is resolved once (a scan resolves every start
#: nested in what it reads), so a reply only exceeds it when it has many
#: brackets inside string literals.
SPAN_SCAN_FACTOR = 8
SPAN_SCAN_MIN_CHARS = 65_536


def _balanced_spans(text: str, open_ch: str, close_ch: str) -> List[Tuple[int, int]]:
    """``(start, end)`` (inclusive) of every balanced ``open_ch … close_ch``
    substring of ``text``, in start order: from each ``open_ch``, the shortest
    substring that closes it, with brackets inside JSON strings ignored (the
    string state is the scan's own, starting outside a string).

    Linear, not quadratic: a scan from one start also resolves every start it
    meets outside a string (its future is the same scan), so a later scan skips
    a start's known span, or stops at a start known never to close; only the
    starts it met inside a string are scanned afresh. Bounded by
    :data:`SPAN_SCAN_FACTOR` in any case."""
    n = len(text)
    ends: Dict[int, int] = {}  # start -> end, or -1: it never closes
    budget = SPAN_SCAN_MIN_CHARS + SPAN_SCAN_FACTOR * n
    spent = 0
    i = text.find(open_ch)
    while i != -1 and spent <= budget:
        if i not in ends:
            stack: List[int] = []
            in_str = escaped = False
            j = i
            while j < n:
                ch = text[j]
                if in_str:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == open_ch:
                    known = ends.get(j) if j != i else None
                    if known is not None:
                        if known < 0:
                            break  # it never closes: nothing around it does
                        j = known + 1  # its span, as scanned before
                        continue
                    stack.append(j)
                elif ch == close_ch:
                    ends[stack.pop()] = j
                    if not stack:
                        break
                j += 1
            spent += j - i + 1
            for start in stack:
                ends[start] = -1
        i = text.find(open_ch, i + 1)
    return [(start, end) for start, end in sorted(ends.items()) if end >= 0]


def _balanced_objects(text: str):
    """Yield each balanced ``{...}`` substring (respecting strings), in the
    order of its opening brace (nested objects included)."""
    for start, end in _balanced_spans(text, "{", "}"):
        yield text[start : end + 1]


def _balanced_arrays(text: str):
    """Yield each balanced ``[...]`` substring (respecting strings), in the
    order of its opening bracket (nested arrays included)."""
    for start, end in _balanced_spans(text, "[", "]"):
        yield text[start : end + 1]


def parse_distiller_json(raw: Optional[str]) -> Optional[Dict]:
    """Robustly parse an LLM reply into a single JSON object (dict), or None.

    Tolerates: a bare JSON object, a ```json fenced block, a plain ``` fenced
    block, or a single object embedded in surrounding prose.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        parsed = _try_load_object(fence.group(1).strip())
        if parsed is not None:
            return parsed

    parsed = _try_load_object(text)
    if parsed is not None:
        return parsed

    for span in _balanced_objects(text):
        parsed = _try_load_object(span)
        if parsed is not None:
            return parsed

    return None


def parse_json_object_with_key(raw: Optional[str], key: str) -> Optional[Dict]:
    """The first JSON object of an LLM reply that has ``key``, or None.

    Tolerates what :func:`parse_distiller_json` does (a bare object, a fenced
    block, an object embedded in prose), and an object inside an enclosing
    array. An object without ``key`` is skipped, so the first element of a
    bare array of other objects is never taken for it.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    candidates = []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text)
    for candidate in candidates:
        parsed = _try_load_object(candidate)
        if parsed is not None and key in parsed:
            return parsed

    for span in _balanced_objects(text):
        parsed = _try_load_object(span)
        if parsed is not None and key in parsed:
            return parsed

    return None


def _load_array_strict(text: str) -> Optional[List[Dict]]:
    """Like :func:`_try_load_array`, but a non-empty array without a single
    object element (e.g. ``[1, 2]``) is not an array of objects: None."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, list):
        return None
    items = [el for el in obj if isinstance(el, dict)]
    if obj and not items:
        return None
    return items


def parse_json_array_or_none(raw: Optional[str]) -> Optional[List[Dict]]:
    """Parse an LLM reply expected to be a JSON ARRAY of objects, or None.

    Tolerates a bare array, a ```json fenced block, a plain fenced block, or an
    array embedded in prose, and wraps a single object (a model that emitted one
    item without the enclosing array). Returns the dict elements (other
    elements dropped); ``[]`` ONLY for an explicit empty array. Anything else —
    no reply, prose, broken JSON, an array without objects — is None, so a
    caller can tell "nothing worth remembering" from an unusable answer.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        items = _load_array_strict(fence.group(1).strip())
        if items is not None:
            return items

    items = _load_array_strict(text)
    if items is not None:
        return items

    for span in _balanced_arrays(text):
        items = _load_array_strict(span)
        if items is not None:
            return items

    obj = parse_distiller_json(text)
    if obj is not None:
        return [obj]

    return None


def parse_distiller_json_array(raw: Optional[str]) -> List[Dict]:
    """:func:`parse_json_array_or_none`, with ``[]`` for an unusable reply.

    Lenient: an unparseable reply and an explicit empty array both give
    ``[]``. Callers that must not mistake the one for the other (the session
    distiller) use :func:`parse_json_array_or_none`.
    """
    return parse_json_array_or_none(raw) or []
