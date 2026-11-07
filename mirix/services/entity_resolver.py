"""v7.8 entity resolution / canonicalization at extraction time.

Closes a long-standing gap (present since v7, amplified by v7.6's LLM triple
extraction): entities are named inconsistently across independently-extracted
memories ("Fitness Class" vs "Workout Session"; "Outward Hound Brick Puzzle" vs
"Outward Hound's Brick Puzzle"), so near-duplicate anchors fragment the graph
(81% singletons under v7.6). Surface `anchor_canonical_key` can't fix this (it's
semantic, not string), and embedding-cosine merge can't DECIDE identity (Monday
~ Tuesday, "42 campsites" ~ "46 campsites" are close but distinct).

This does what the research canonicalization layer (EDC / KARMA) does: embedding
retrieves CANDIDATE existing anchors, then an LLM DECIDES same-or-new and renames
each new entity to an existing canonical name when they denote the same entity.
Registry-guided — candidates are the anchors already in the graph — so naming
stays consistent as the graph grows. Because the LLM sees the names (and could see
context), this is also the only mechanism able to keep distinct same-string
entities apart (G1) rather than only merging (G2).
"""
from __future__ import annotations

import json
from typing import List

from mirix.log import get_logger
from mirix.services._graph_common import embed_batch
from mirix.services.lightrag_extractor import ExtractedEntity, call_openai_chat
from mirix.settings import settings

logger = get_logger(__name__)

_CAND_COS = 0.86   # only entities with a candidate at least this close go to the LLM
_TOPK = 4

RESOLVE_PROMPT = """You canonicalize entity names for a knowledge graph. For each NEW entity you are given candidate EXISTING entity names. For each new entity, decide whether it denotes the SAME real-world entity/concept as one of its candidates.
- If yes, output that existing candidate's exact name (reuse it).
- If there is no true match, output the new entity's own name unchanged.
Be CONSERVATIVE — only merge when they are truly the same entity. Do NOT merge different quantities ("42 campsites" vs "46 campsites"), different days/months, different form/model codes ("I-601" vs "I-601A"), or merely related-but-distinct concepts ("Comedy" vs "Comedians", "Monday" vs "Tuesday").
Return a JSON object mapping every new entity name to its chosen canonical name: {"new name": "canonical name", ...}"""


async def _candidates(driver, user_id: str, embs, names) -> dict:
    """{new_name: [existing anchor names]} for entities with a close existing match."""
    out: dict = {}
    async with driver.session(database=settings.neo4j_database) as session:
        for nm, emb in zip(names, embs):
            if emb is None:
                continue
            res = await session.run(
                """
                CALL db.index.vector.queryNodes('v7_anchor_name_emb', $k, $emb)
                YIELD node AS a, score AS sc
                WHERE a.user_id = $u AND sc >= $th AND a.name <> $nm
                RETURN a.name AS n ORDER BY sc DESC
                """,
                k=_TOPK, emb=emb, u=user_id, th=_CAND_COS, nm=nm)
            cands = [r["n"] async for r in res]
            if cands:
                out[nm] = cands
    return out


async def canonicalize_entities(
    entities: List[ExtractedEntity], *, driver, user_id: str, agent_state
) -> dict:
    """Rename entities in place to existing canonical anchors when they match.
    Returns the {original_name: canonical_name} map (for renaming relation endpoints)."""
    if not entities:
        return {}
    names = [e.name for e in entities]
    embs = await embed_batch(names, agent_state)
    cand = await _candidates(driver, user_id, embs, names)
    if not cand:
        return {}
    payload = [{"new": nm, "candidates": cs} for nm, cs in cand.items()]
    try:
        raw = await call_openai_chat(
            RESOLVE_PROMPT, json.dumps(payload, ensure_ascii=False), "gpt-4.1-mini", temperature=0.0)
        blob = raw if raw.strip().startswith("{") else raw[raw.find("{"): raw.rfind("}") + 1]
        mapping = json.loads(blob)
    except Exception as e:  # noqa: BLE001
        logger.warning("entity resolve failed: %s", e)
        return {}

    applied: dict = {}
    valid_cands = {c for cs in cand.values() for c in cs}
    for e in entities:
        canon = mapping.get(e.name)
        # only accept a rename to an actual candidate (guard against LLM inventing names)
        if isinstance(canon, str) and canon.strip() and canon != e.name and canon in valid_cands:
            applied[e.name] = canon.strip()
            e.name = canon.strip()
    return applied
