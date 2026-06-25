"""Action parsing helpers for ALFWorld model responses.

SkillOpt expects ``<think>...</think><action>...</action>`` responses. The
runner follows that format and records parser validity separately from the
fallback action used to keep an episode moving.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_FALLBACK_ACTION = "look"

_ACTION_RE = re.compile(r"<action\b[^>]*>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ParsedAction:
    """Parsed action plus response metadata useful for eval transcripts."""

    action: str
    raw_text: str
    thought: str | None = None
    source: str = "fallback"
    used_fallback: bool = True
    format_valid: bool = False
    has_think: bool = False
    contains_chinese: bool = False


def parse_action(
    text: str | None,
    *,
    fallback: str = DEFAULT_FALLBACK_ACTION,
    allow_json: bool = False,
    require_think: bool = True,
    lowercase_action: bool = True,
) -> ParsedAction:
    """Parse an ALFWorld action from model output.

    Priority:
    1. First non-empty ``<action>...</action>`` block.
    2. JSON object with an ``action`` string when ``allow_json`` is true.
    3. ``fallback`` (``look`` by default).

    ``format_valid`` mirrors SkillOpt's projection rule: XML action present,
    ``<think>`` present when required, and no Chinese characters. JSON is kept
    as an opt-in compatibility extension, not the default experiment format.
    """

    raw_text = "" if text is None else str(text)
    thought = extract_thought(raw_text)
    has_think = thought is not None
    contains_chinese = bool(_CHINESE_RE.search(raw_text))

    action = _extract_tagged_action(raw_text)
    if action:
        action = _normalize_action(action, lowercase=lowercase_action)
        return ParsedAction(
            action=action,
            raw_text=raw_text,
            thought=thought,
            source="xml",
            used_fallback=False,
            format_valid=(not require_think or has_think) and not contains_chinese,
            has_think=has_think,
            contains_chinese=contains_chinese,
        )

    if allow_json:
        action = _extract_json_action(raw_text)
        if action:
            action = _normalize_action(action, lowercase=lowercase_action)
            return ParsedAction(
                action=action,
                raw_text=raw_text,
                thought=thought,
                source="json",
                used_fallback=False,
                format_valid=False,
                has_think=has_think,
                contains_chinese=contains_chinese,
            )

    return ParsedAction(
        action=_normalize_action(fallback, lowercase=lowercase_action),
        raw_text=raw_text,
        thought=thought,
        source="fallback",
        used_fallback=True,
        format_valid=False,
        has_think=has_think,
        contains_chinese=contains_chinese,
    )


def parse_model_response(text: str | None) -> dict[str, str]:
    """Return a simple dict view for older callers.

    New harness code should use ``parse_action`` so missing actions fall back to
    ``look``.  This wrapper keeps legacy free-text behavior for existing tests
    and exploratory scripts.
    """

    raw_text = "" if text is None else str(text)
    json_thought = _extract_json_thought(raw_text)
    parsed = parse_action(raw_text, allow_json=True)
    action = parsed.action
    legacy_thought = None
    if parsed.used_fallback:
        action = _legacy_free_text_action(raw_text) or parsed.action
        legacy_thought = _legacy_free_text_thought(raw_text)
    return {
        "thought": parsed.thought
        if parsed.thought is not None
        else json_thought or legacy_thought or "",
        "action": action,
    }


def extract_action(
    text: str | None,
    *,
    fallback: str = DEFAULT_FALLBACK_ACTION,
    allow_json: bool = False,
) -> str:
    """Return only the executable ALFWorld action string."""

    return parse_action(text, fallback=fallback, allow_json=allow_json).action


def extract_thought(text: str | None) -> str | None:
    """Return preserved ``<think>`` block text, joined in response order."""

    raw_text = "" if text is None else str(text)
    thoughts = [match.group(1).strip() for match in _THINK_RE.finditer(raw_text)]
    thoughts = [thought for thought in thoughts if thought]
    if not thoughts:
        return None
    return "\n\n".join(thoughts)


def _extract_tagged_action(text: str) -> str | None:
    for match in _ACTION_RE.finditer(text):
        action = match.group(1).strip()
        if action:
            return action
    return None


def _extract_json_action(text: str) -> str | None:
    payload = _load_json_object(text)
    if not isinstance(payload, dict):
        return None

    action = payload.get("action")
    if not isinstance(action, str):
        return None

    action = action.strip()
    return action or None


def _extract_json_thought(text: str) -> str | None:
    payload = _load_json_object(text)
    if not isinstance(payload, dict):
        return None

    thought = payload.get("thought")
    if not isinstance(thought, str):
        return None

    thought = thought.strip()
    return thought or None


def _legacy_free_text_action(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[-1]


def _legacy_free_text_thought(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return None
    return "\n".join(lines[:-1])


def _normalize_action(action: str, *, lowercase: bool) -> str:
    action = " ".join(action.strip().split())
    return action.lower() if lowercase else action


def _load_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None
