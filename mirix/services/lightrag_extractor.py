"""
LightRAG-style entity & relation extractor (W2 of the write path).

Adapted from LightRAG operate.py:extract_entities. One LLM call per event;
output is delimiter-separated tuples that are parsed into structured dicts.
Optional gleaning pass (default off) re-prompts the LLM to catch misses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import (
    COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    TUPLE_DELIMITER,
    render_extraction_system_prompt,
    render_extraction_user_prompt,
)

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str


@dataclass
class ExtractedRelation:
    src: str
    tgt: str
    keywords: str
    description: str
    weight: float


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
        return s[1:-1].strip()
    return s


def _coerce_weight(raw: str) -> float:
    """Parse the trailing relationship_strength field. Defaults to 0.5 on bad input."""
    try:
        v = float(_strip_quotes(raw))
        if 0.0 <= v <= 1.0:
            return v
        # Some models emit 0..10 or 0..100. Normalize.
        if 1.0 < v <= 10.0:
            return v / 10.0
        if 10.0 < v <= 100.0:
            return v / 100.0
    except (ValueError, TypeError):
        pass
    return 0.5


def parse_extraction_output(raw: str) -> ExtractionResult:
    """
    Parse LightRAG-style delimiter output into structured entities & relations.

    Each line should look like:
        entity<|#|>NAME<|#|>TYPE<|#|>DESCRIPTION
        relation<|#|>SRC<|#|>TGT<|#|>KEYWORDS<|#|>DESCRIPTION<|#|>STRENGTH
    Lines that do not parse cleanly are logged and skipped.
    """
    result = ExtractionResult()
    if not raw:
        return result

    # Stop at the completion delimiter if the model emitted it
    cut = raw.find(COMPLETION_DELIMITER)
    if cut >= 0:
        raw = raw[:cut]

    seen_entity_names: set[str] = set()
    seen_relation_keys: set[tuple[str, str]] = set()

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or TUPLE_DELIMITER not in line:
            continue
        parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
        kind = parts[0].lower().strip("()`* ")
        if kind == "entity" and len(parts) >= 4:
            name = _strip_quotes(parts[1])
            entity_type = _strip_quotes(parts[2]) or "Other"
            description = _strip_quotes(parts[3])
            if not name or name in seen_entity_names:
                continue
            seen_entity_names.add(name)
            result.entities.append(
                ExtractedEntity(name=name, entity_type=entity_type, description=description)
            )
        elif kind == "relation" and len(parts) >= 5:
            src = _strip_quotes(parts[1])
            tgt = _strip_quotes(parts[2])
            keywords = _strip_quotes(parts[3])
            description = _strip_quotes(parts[4])
            weight = _coerce_weight(parts[5]) if len(parts) >= 6 else 0.5
            if not src or not tgt or src == tgt:
                continue
            # Treat undirected; dedup on sorted endpoints
            key = tuple(sorted([src.lower(), tgt.lower()]))
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            result.relations.append(
                ExtractedRelation(
                    src=src,
                    tgt=tgt,
                    keywords=keywords,
                    description=description,
                    weight=weight,
                )
            )
        else:
            # Unknown leading token — skip silently to avoid log spam on
            # benign formatting variations.
            continue

    return result


async def call_openai_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Bare-metal OpenAI chat completion. Mirrors v2 graph_memory_manager."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    endpoint = f"{api_base.rstrip('/')}/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Record token usage if a phase is active (no-op outside instrumented evals)
    try:
        from mirix.database.token_tracker import record as _record_tokens
        usage = (data.get("usage") or {})
        _record_tokens(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
        )
    except Exception:
        pass

    return data["choices"][0]["message"]["content"]


async def extract_entities_and_relations(
    text: str,
    *,
    llm_model: str = "gpt-4.1-mini",
    entity_types: Optional[list[str]] = None,
    language: str = "English",
    max_input_chars: int = 12000,
) -> ExtractionResult:
    """
    Run a single LLM extraction pass over ``text`` and parse the result.

    Returns an empty ``ExtractionResult`` on error so the caller can carry on.
    """
    if not text or not text.strip():
        return ExtractionResult()

    types = entity_types or DEFAULT_ENTITY_TYPES
    system_prompt = render_extraction_system_prompt(entity_types=types, language=language)
    user_prompt = render_extraction_user_prompt(
        input_text=text[:max_input_chars],
        entity_types=types,
        language=language,
    )

    try:
        raw = await call_openai_chat(system_prompt, user_prompt, model=llm_model)
    except Exception as e:
        logger.warning("LightRAG extraction LLM call failed: %s", e)
        return ExtractionResult()

    parsed = parse_extraction_output(raw)
    logger.info(
        "LightRAG extraction: %d entities, %d relations from %d chars",
        len(parsed.entities),
        len(parsed.relations),
        len(text),
    )
    return parsed
