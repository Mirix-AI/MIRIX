"""
LightRAG-style query keyword extractor (high-level / low-level split).

One LLM call per unique query, cached in Redis (or skipped if Redis is not
available — the system still works, just pays the extraction cost each time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import render_keywords_extraction_prompt
from mirix.services.lightrag_extractor import call_openai_chat

logger = get_logger(__name__)


# Match LightRAG defaults (24h is long enough for typical chat sessions).
KEYWORD_CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class Keywords:
    high_level: list[str]
    low_level: list[str]


def _cache_key(user_id: str, query: str, language: str) -> str:
    h = hashlib.sha1(f"{language}|{query}".encode("utf-8")).hexdigest()[:24]
    return f"mirix:lightrag:kw:{user_id}:{h}"


def _parse_json_loose(raw: str) -> Optional[dict]:
    """Try strict JSON first; if that fails, strip code fences and retry."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip ``` fences
    if "```" in raw:
        try:
            body = raw.split("```json")[-1] if "```json" in raw else raw.split("```")[1]
            body = body.split("```")[0]
            return json.loads(body.strip())
        except (json.JSONDecodeError, IndexError):
            pass
    return None


async def _cache_get(key: str) -> Optional[Keywords]:
    try:
        from mirix.database.cache_provider import get_cache_provider

        provider = get_cache_provider()
        if provider is None:
            return None
        data = await provider.get_json(key)
        if not data:
            return None
        return Keywords(
            high_level=list(data.get("high_level", []) or []),
            low_level=list(data.get("low_level", []) or []),
        )
    except Exception as e:
        logger.debug("Keyword cache get failed: %s", e)
        return None


async def _cache_set(key: str, kw: Keywords) -> None:
    try:
        from mirix.database.cache_provider import get_cache_provider

        provider = get_cache_provider()
        if provider is None:
            return
        await provider.set_json(
            key,
            {"high_level": kw.high_level, "low_level": kw.low_level},
            ttl=KEYWORD_CACHE_TTL_SECONDS,
        )
    except Exception as e:
        logger.debug("Keyword cache set failed: %s", e)


def _fallback_keywords(query: str) -> Keywords:
    """When the LLM returns nothing useful, treat the query itself as ll keyword.

    Mirrors LightRAG operate.py:get_keywords_from_query short-query fallback.
    """
    q = (query or "").strip()
    if not q:
        return Keywords(high_level=[], low_level=[])
    if len(q) < 50:
        return Keywords(high_level=[], low_level=[q])
    # Long but empty parse: keep first few content words as best-effort.
    words = [w for w in q.split() if len(w) > 3][:6]
    return Keywords(high_level=[], low_level=words or [q[:80]])


async def extract_keywords(
    query: str,
    *,
    user_id: str,
    llm_model: str = "gpt-4.1-mini",
    language: str = "English",
    use_cache: bool = True,
) -> Keywords:
    """
    Return (high_level, low_level) keyword lists for ``query``.

    On any failure or empty model output, falls back to using the query itself
    as a single low-level keyword (short queries) or splitting into content
    words (long queries). Never raises.
    """
    if not query or not query.strip():
        return Keywords(high_level=[], low_level=[])

    cache_key = _cache_key(user_id, query, language)
    if use_cache:
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

    prompt = render_keywords_extraction_prompt(query=query, language=language)

    try:
        raw = await call_openai_chat(
            system_prompt="You are a precise keyword extractor. Output JSON only.",
            user_prompt=prompt,
            model=llm_model,
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        logger.warning("Keyword extraction LLM call failed: %s", e)
        return _fallback_keywords(query)

    parsed = _parse_json_loose(raw)
    if not parsed:
        logger.warning("Keyword extraction returned unparsable output: %s", (raw or "")[:120])
        return _fallback_keywords(query)

    kw = Keywords(
        high_level=[k.strip() for k in parsed.get("high_level_keywords", []) if k and k.strip()],
        low_level=[k.strip() for k in parsed.get("low_level_keywords", []) if k and k.strip()],
    )
    if not kw.high_level and not kw.low_level:
        kw = _fallback_keywords(query)

    if use_cache:
        await _cache_set(cache_key, kw)
    return kw
