"""Direction D extractor — one LLM call emits knowledge-graph triples.

Replaces the v7.4 GLiNER extractor as the main path. GLiNER (a span tagger)
cannot abstract — it drops LightRAG's concept anchors (networking, team
collaboration…) and can't produce relations. An LLM, by contrast, both abstracts
("socializing with colleagues while remote" -> concept "Workplace Socializing")
and yields the relation (predicate), which v7 extracted but discarded. Every
recent top-venue KG method (HippoRAG 2, KGGen, AutoSchemaKG, EDC) uses LLM triple
extraction for exactly these reasons; see docs/graph_memory_v7/extractor_direction_D.md.

Output feeds the existing anchor pipeline: unique entities -> `_select_anchors`
(unchanged), and relations -> new anchor->anchor `V7_RELATION` edges (which v7
lacked, and which PPR retrieval needs).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mirix.log import get_logger
from mirix.services.lightrag_extractor import ExtractedEntity, call_openai_chat

logger = get_logger(__name__)

# Same type vocabulary _specificity_score scores; the LLM maps to these.
ENTITY_TYPES = (
    "person", "organization", "location", "event",
    "content", "object", "concept", "method", "date",
)

# Dialogue roles carry no entity identity — as anchors they are the v7 noise
# mega-hubs (deg 42/24, connected to everything). Dropped here; the user/assistant
# provenance is captured properly by the role attribute (from episodic.actor),
# not a "User" anchor. A relation whose endpoint is one of these simply won't form
# an anchor->anchor edge (the endpoint isn't an anchor), which is correct.
_NOISE = {
    "user", "assistant", "you", "i", "me", "we", "us", "they", "them",
    "he", "she", "it", "my", "your", "our", "their", "the user", "the assistant",
}

TRIPLE_PROMPT = """Extract knowledge-graph triples from the text: every meaningful (subject, relation, object) fact.
Subjects/objects are ENTITIES: named things (people/places/orgs/products), OR the underlying CONCEPT/THEME.
For concepts, output a CONCISE CANONICAL name (2-4 words, the general theme), NOT a copied phrase:
  "misses watercooler chats with colleagues while remote" -> concept "Remote Team Networking"
  "socializing with colleagues while working from home"   -> concept "Workplace Socializing"
Each entity has name + type from EXACTLY this list:
person, organization, location, event, concept, method, content, object, date
Keep the relation a SHORT verb phrase. Return JSON:
{"triples":[{"s":{"name":"...","type":"..."},"r":"short relation","o":{"name":"...","type":"..."}}]}"""


@dataclass
class TripleResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    # (subject_name, relation, object_name) in the entities' surface names
    relations: List[Tuple[str, str, str]] = field(default_factory=list)


def _norm_type(t: object) -> str:
    t = str(t or "").strip().lower()
    return t if t in ENTITY_TYPES else "other"


def _parse(raw: str) -> TripleResult:
    """Tolerant parse: JSON object, or the first {...} block if the model wrapped it."""
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return TripleResult()
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return TripleResult()

    ents: dict[str, ExtractedEntity] = {}
    rels: List[Tuple[str, str, str]] = []
    for tr in data.get("triples", []) or []:
        if not isinstance(tr, dict):
            continue
        s, o = tr.get("s") or {}, tr.get("o") or {}
        sn = str((s or {}).get("name") or "").strip()
        on = str((o or {}).get("name") or "").strip()
        rel = str(tr.get("r") or "").strip()
        for nm, spec in ((sn, s), (on, o)):
            if nm and nm.lower() not in _NOISE and nm.lower() not in ents:
                ents[nm.lower()] = ExtractedEntity(
                    name=nm, entity_type=_norm_type((spec or {}).get("type")), description="")
        # Keep the relation even if one endpoint is a dialogue role — _link_relation_edges
        # matches endpoints against existing anchors, so a User/Assistant endpoint just
        # yields no edge (its concept object is still anchored via the entity list).
        if sn and on and sn.lower() != on.lower():
            rels.append((sn, rel, on))
    return TripleResult(entities=list(ents.values()), relations=rels)


async def extract_triples(
    text: str,
    *,
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    max_chars: int = 12000,
) -> TripleResult:
    """Single LLM call -> typed entities + relations. Empty result on failure."""
    if not text or not text.strip():
        return TripleResult()
    try:
        raw = await call_openai_chat(
            TRIPLE_PROMPT, text[:max_chars], model, temperature=0.0, api_key=api_key)
    except Exception as e:  # noqa: BLE001
        logger.warning("triple extraction failed: %s", e)
        return TripleResult()
    return _parse(raw)
