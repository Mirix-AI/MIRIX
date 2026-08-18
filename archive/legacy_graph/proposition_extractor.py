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

The same call also emits each proposition's entities. V7's graph build consumes
only `name` + `entity_type` from an extraction (relations and descriptions are
discarded), so handing those two fields to `V7GraphManager.process_memory(
entities=...)` removes the per-proposition LightRAG call entirely: one LLM call
per chunk instead of one proposition call plus one LightRAG call per proposition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from mirix.log import get_logger

logger = get_logger(__name__)

# Must stay aligned with V7GraphManager._specificity_score, which scores
# person/location/organization/event highest, then content/object, then
# concept/method. An unrecognised type scores as "other", and a single-word
# "other" anchor is dropped outright -- so prompt drift here silently shrinks
# the graph rather than failing loudly.
ENTITY_TYPES = (
    "person",
    "location",
    "organization",
    "event",
    "content",
    "object",
    "concept",
    "method",
)

PROPOSITION_PROMPT = """You extract memory facts from a conversation or document chunk.
Output EVERY atomic fact as a self-contained proposition:
- ONE fact per proposition; each must be understandable ALONE (resolve pronouns to full
  names; carry the subject and any date into every proposition).
- Include ALL numbers, dates, names, amounts, roles, populations, and preferences VERBATIM —
  these peripheral details are the point, do not summarize them away.
- For dialogue, capture what the user did / said / prefers, with the date if present.

For each proposition also list its entities. An entity is a specific, named thing the
proposition is about — a person, place, organization, event, work, or object. Do NOT list
generic words ("meeting", "food"), pronouns, or the proposition's verb.
Each entity needs a "type" from EXACTLY this list:
person, location, organization, event, content, object, concept, method

Return JSON:
{"propositions": [{"text": "...", "entities": [{"name": "...", "type": "person"}]}]}"""


@dataclass
class PropositionEntity:
    name: str
    type: str


@dataclass
class Proposition:
    text: str
    entities: List[PropositionEntity] = field(default_factory=list)


def _coerce(data: dict) -> List[Proposition]:
    """Parse the model's JSON into Propositions, tolerating a bare-string
    proposition (no entities) so a partial response still yields memories."""
    out: List[Proposition] = []
    for raw in data.get("propositions", []) or []:
        if isinstance(raw, str):
            text, ents = raw.strip(), []
        elif isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
            ents = raw.get("entities") or []
        else:
            continue
        if not text:
            continue

        entities: List[PropositionEntity] = []
        for e in ents:
            if isinstance(e, str):
                name, etype = e.strip(), "other"
            elif isinstance(e, dict):
                name = str(e.get("name") or "").strip()
                etype = str(e.get("type") or "other").strip().lower()
            else:
                continue
            if not name:
                continue
            if etype not in ENTITY_TYPES:
                logger.debug("proposition entity type %r not in vocabulary", etype)
                etype = "other"
            entities.append(PropositionEntity(name=name, type=etype))
        out.append(Proposition(text=text, entities=entities))
    return out


async def extract_propositions_with_entities(
    text: str,
    *,
    api_key: str,
    llm_model: str = "gpt-4.1-mini",
    endpoint: str = "https://api.openai.com/v1",
    max_chars: int = 16000,
) -> List[Proposition]:
    """Single-call extraction of atomic propositions AND their typed entities.
    Returns [] on failure — the caller decides how to handle."""
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
        return _coerce(json.loads(resp.choices[0].message.content))
    except Exception as e:  # noqa: BLE001
        logger.warning("proposition extraction failed: %s", e)
        return []


async def extract_propositions(
    text: str,
    *,
    api_key: str,
    llm_model: str = "gpt-4.1-mini",
    endpoint: str = "https://api.openai.com/v1",
    max_chars: int = 16000,
) -> List[str]:
    """Proposition text only, for callers that build the graph separately."""
    props = await extract_propositions_with_entities(
        text, api_key=api_key, llm_model=llm_model, endpoint=endpoint, max_chars=max_chars
    )
    return [p.text for p in props]
