"""v7.3 atomic-proposition extraction.

Replaces multi-round LightRAG "gleaning" summaries with a SINGLE structured LLM
call that emits every atomic, self-contained fact. Motivation: gleaning-style
entity extraction captures salient entities but drops "cold" peripheral facts
(a county's population, a film's executive producer, a person's death year),
which are exactly the second-hop bridging facts multi-hop QA needs. Empirically
(gpt-4.1-mini): single-call proposition extraction captured cold facts LightRAG
dropped, at ~3-8x lower latency (one call vs N gleaning rounds).

Each returned proposition is stored as a fine-grained semantic memory; the
existing V7 graph build (anchor -> DESCRIBED_BY -> ConceptRef) indexes it
unchanged, so a proposition needs no new node type.
"""
from __future__ import annotations

import json
from typing import List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

PROPOSITION_PROMPT = """You extract memory facts from a conversation or document chunk.
Output EVERY atomic fact as a self-contained proposition:
- ONE fact per proposition; each must be understandable ALONE (resolve pronouns to full
  names; carry the subject and any date into every proposition).
- Include ALL numbers, dates, names, amounts, roles, populations, and preferences VERBATIM —
  these peripheral details are the point, do not summarize them away.
- For dialogue, capture what the user did / said / prefers, with the date if present.
Return JSON: {"propositions": ["...", ...]}"""


async def extract_propositions(
    text: str,
    *,
    api_key: str,
    llm_model: str = "gpt-4.1-mini",
    endpoint: str = "https://api.openai.com/v1",
    max_chars: int = 16000,
) -> List[str]:
    """Single-call atomic-proposition extraction. Returns a list of proposition
    strings (empty on failure — caller decides how to handle)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
    try:
        resp = await client.chat.completions.create(
            model=llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PROPOSITION_PROMPT},
                {"role": "user", "content": text[:max_chars]},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        props = data.get("propositions", [])
        return [p.strip() for p in props if isinstance(p, str) and p.strip()]
    except Exception as e:  # noqa: BLE001
        logger.warning("proposition extraction failed: %s", e)
        return []
