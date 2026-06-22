"""Generic, agent-agnostic JSON-extraction helpers for LLM replies.

These are pure, dependency-free utilities for robustly recovering a JSON object
or array from a model reply that may wrap it in a ```json fenced block, plain
fences, or surrounding prose. They carry NO benchmark/MetaClaw vocabulary so the
general session-experience distiller (Goal 2) can depend on them without coupling
to the MetaClaw eval distiller module. This is the canonical home for these
helpers; ``skill_session_distiller`` keeps its own legacy copies for the eval path.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


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


def _balanced_objects(text: str):
    """Yield each top-level balanced ``{...}`` substring (respecting strings)."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
        i += 1


def _balanced_arrays(text: str):
    """Yield each top-level balanced ``[...]`` substring (respecting strings)."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
        i += 1


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


def parse_distiller_json_array(raw: Optional[str]) -> List[Dict]:
    """Robustly parse an LLM reply expected to be a JSON ARRAY of objects.

    Tolerates a bare array, a ```json fenced block, a plain fenced block, or an
    array embedded in prose. Returns the list of dict elements (non-dict elements
    dropped). Returns ``[]`` when nothing parseable is found OR when the model
    legitimately emitted an empty array (the "nothing worth remembering" case).
    Falls back to wrapping a single object so a model that emitted one item
    without the enclosing array is not silently dropped.
    """
    if not raw or not isinstance(raw, str):
        return []
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        items = _try_load_array(fence.group(1).strip())
        if items is not None:
            return items

    items = _try_load_array(text)
    if items is not None:
        return items

    for span in _balanced_arrays(text):
        items = _try_load_array(span)
        if items is not None:
            return items

    obj = parse_distiller_json(text)
    if obj is not None:
        return [obj]

    return []
