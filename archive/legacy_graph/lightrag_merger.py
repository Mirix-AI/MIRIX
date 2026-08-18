"""
Description merging for entities and relations (W3/W4 helper).

Adapted from LightRAG operate.py:_handle_entity_relation_summary. The strategy:

1. If the descriptions, joined, fit within ``summary_context_size`` tokens AND
   there are fewer than ``force_llm_summary_on_merge`` of them → just join with
   a separator. No LLM call.
2. If the joined text fits within ``summary_max_tokens`` → ask the LLM for a
   single summary. 1 LLM call.
3. Otherwise → split into chunks, summarize each, recurse on the summaries.

Token counts are estimated with tiktoken (cl100k_base) for cheap accuracy.
"""

from __future__ import annotations

from typing import Optional

import tiktoken

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import render_summarize_descriptions_prompt
from mirix.services.lightrag_extractor import call_openai_chat

logger = get_logger(__name__)


# Defaults align with LightRAG's recommended values. Tuned smaller to keep
# write-path cost low (MIRIX writes much more often than LightRAG ingests docs).
DEFAULT_SUMMARY_CONTEXT_SIZE = 1000  # tokens — when joined desc still fits, no summary
DEFAULT_SUMMARY_MAX_TOKENS = 500     # tokens — target output length
DEFAULT_FORCE_LLM_MERGE_AT = 6       # description count threshold
DEFAULT_SEPARATOR = " | "

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


async def merge_descriptions(
    description_type: str,
    name: str,
    descriptions: list[str],
    *,
    llm_model: str = "gpt-4.1-mini",
    summary_context_size: int = DEFAULT_SUMMARY_CONTEXT_SIZE,
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    force_llm_merge_at: int = DEFAULT_FORCE_LLM_MERGE_AT,
    separator: str = DEFAULT_SEPARATOR,
    max_recursion: int = 4,
) -> tuple[str, bool]:
    """
    Merge a list of descriptions for a single entity or relation.

    Returns ``(merged_text, llm_used)``. ``llm_used`` lets the caller decide
    whether to bump cache invalidation timestamps.
    """
    descs = [d.strip() for d in descriptions if d and d.strip()]
    if not descs:
        return "", False
    if len(descs) == 1:
        return descs[0], False

    # Phase 1: cheap path — no LLM if small enough and few enough.
    joined = separator.join(descs)
    total_tokens = _count_tokens(joined)
    if total_tokens <= summary_context_size and len(descs) < force_llm_merge_at:
        return joined, False

    # Phase 2: single LLM summary if it all fits as a prompt.
    if total_tokens <= summary_max_tokens * 4:  # rough budget for prompt+output
        summary = await _summarize_via_llm(
            description_type=description_type,
            name=name,
            descriptions=descs,
            llm_model=llm_model,
            summary_max_tokens=summary_max_tokens,
        )
        return summary or joined[: summary_max_tokens * 4], True

    # Phase 3: map-reduce. Chunk descs into groups whose joined size fits, then
    # summarize each chunk, then recurse on the chunk summaries.
    if max_recursion <= 0:
        # Hard stop: just truncate the joined text. Avoids unbounded recursion
        # on pathological input.
        return joined[: summary_max_tokens * 4], False

    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for d in descs:
        d_tokens = _count_tokens(d)
        if current and current_tokens + d_tokens > summary_context_size:
            chunks.append(current)
            current, current_tokens = [d], d_tokens
        else:
            current.append(d)
            current_tokens += d_tokens
    if current:
        chunks.append(current)

    chunk_summaries: list[str] = []
    llm_used = False
    for ch in chunks:
        if len(ch) == 1:
            chunk_summaries.append(ch[0])
            continue
        s = await _summarize_via_llm(
            description_type=description_type,
            name=name,
            descriptions=ch,
            llm_model=llm_model,
            summary_max_tokens=summary_max_tokens,
        )
        if s:
            chunk_summaries.append(s)
            llm_used = True
        else:
            # Fallback: keep raw join of this chunk
            chunk_summaries.append(separator.join(ch))

    # Recurse on the chunk summaries (now fewer items, each smaller).
    final, recurse_used = await merge_descriptions(
        description_type=description_type,
        name=name,
        descriptions=chunk_summaries,
        llm_model=llm_model,
        summary_context_size=summary_context_size,
        summary_max_tokens=summary_max_tokens,
        force_llm_merge_at=force_llm_merge_at,
        separator=separator,
        max_recursion=max_recursion - 1,
    )
    return final, llm_used or recurse_used


async def _summarize_via_llm(
    description_type: str,
    name: str,
    descriptions: list[str],
    llm_model: str,
    summary_max_tokens: int,
) -> Optional[str]:
    """One LLM call to merge ``descriptions`` into a single paragraph."""
    prompt = render_summarize_descriptions_prompt(
        description_type=description_type,
        description_name=name,
        description_list=descriptions,
        summary_length=summary_max_tokens,
    )
    try:
        # Use a tiny system prompt; the user prompt carries the full template.
        return (
            await call_openai_chat(
                system_prompt="You are a precise summarizer.",
                user_prompt=prompt,
                model=llm_model,
                max_tokens=summary_max_tokens + 200,
            )
        ).strip()
    except Exception as e:
        logger.warning("Description merge LLM call failed for %s '%s': %s", description_type, name, e)
        return None
