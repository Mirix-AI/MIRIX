"""v7.4 GLiNER entity extraction — replaces the per-memory LightRAG LLM call.

Why: LightRAG extraction was profiled at ~20.8 s/memory (~97% of graph-build
time) and v7 only ever consumes an entity's name + type (relations/descriptions
are discarded). GLiNER is a single local encoder forward pass (~0.3 s, no API,
no GPU required) that emits exactly (span, type) — so it attacks the dominant
cost (gap G6) with no behavioural change to the downstream anchor pipeline, which
already accepts a list of ``ExtractedEntity`` via ``process_memory(entities=...)``.

It also closes part of gap G4: dialogue-role tokens ("user", "assistant") are
what LightRAG turns into the degree-42 / degree-24 noise mega-hubs. GLiNER tags
them as ``person`` too, so we drop them here — they never become anchors.

Deterministic (no sampling), so unlike the LightRAG baseline a v7.4 build is
reproducible run-to-run.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import List

from mirix.log import get_logger
from mirix.services.lightrag_extractor import ExtractedEntity

logger = get_logger(__name__)

GLINER_MODEL = os.environ.get("MIRIX_GLINER_MODEL", "urchade/gliner_medium-v2.1")
GLINER_THRESHOLD = float(os.environ.get("MIRIX_GLINER_THRESHOLD", "0.5"))

# GLiNER labels -> the entity_type vocabulary that _specificity_score scores.
# person/location/organization/event are the high-value types (+4); product/
# creative-work map to object/content (+3); concept/date are kept as-is.
_LABELS = ["person", "organization", "location", "event", "product",
           "creative work", "concept", "date"]
_LABEL_TO_TYPE = {
    "person": "person", "organization": "organization", "location": "location",
    "event": "event", "product": "object", "creative work": "content",
    "concept": "concept", "date": "date",
}
# Dialogue roles / bare pronouns GLiNER tags as person but which carry no entity
# identity — these are exactly the v7 noise hubs. Never anchor them.
_NOISE = {
    "user", "assistant", "you", "i", "me", "we", "us", "they", "them",
    "he", "she", "it", "my", "your", "our", "their",
    # Plurals slipped this filter (a "Users" anchor reached degree 1016), and this
    # set never had the singular article forms triple_extractor always had.
    "users", "assistants", "the users", "the assistants",
    "the user", "the assistant",
}


@lru_cache(maxsize=1)
def _model():
    """Load once per process (~19 s first call; cached thereafter)."""
    from gliner import GLiNER

    logger.info("loading GLiNER model %s", GLINER_MODEL)
    return GLiNER.from_pretrained(GLINER_MODEL)


def _extract_sync(text: str, threshold: float, max_chars: int) -> List[ExtractedEntity]:
    model = _model()
    spans = model.predict_entities(text[:max_chars], _LABELS, threshold=threshold)
    out: List[ExtractedEntity] = []
    seen = set()
    for sp in spans:
        name = (sp.get("text") or "").strip()
        nl = name.lower()
        if not name or len(nl) < 2 or nl in _NOISE:
            continue
        key = (nl, sp.get("label"))
        if key in seen:
            continue
        seen.add(key)
        out.append(ExtractedEntity(
            name=name,
            entity_type=_LABEL_TO_TYPE.get(sp.get("label", ""), "other"),
            description="",
        ))
    return out


async def extract_entities_gliner(
    text: str,
    *,
    threshold: float = GLINER_THRESHOLD,
    max_chars: int = 12000,
) -> List[ExtractedEntity]:
    """Single-pass local NER. Returns deduped ``ExtractedEntity`` (name + type;
    description empty). Runs the sync forward pass off the event loop."""
    if not text or not text.strip():
        return []
    try:
        return await asyncio.to_thread(_extract_sync, text, threshold, max_chars)
    except Exception as e:  # noqa: BLE001
        logger.warning("GLiNER extraction failed: %s", e)
        return []
