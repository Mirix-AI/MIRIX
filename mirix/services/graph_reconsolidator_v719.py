"""v7.19 bounded dirty-frontier AutoDream.

The ingest and retrieval arms are exactly v7.12.  Dream is deliberately small:

* only semantic-backed, identity-stable Anchor types are merge candidates;
* both endpoints must have the same eligible type;
* intermediate cycles use the current semantic delta, one round, max 80 pairs;
* the final cycle uses the current delta plus eligible Anchors never used as a
  source by an earlier v7.19 Dream, max 120 pairs;
* one optional repair round starts only from representatives touched by a
  confirmed first-round merge, max 60 pairs;
* explicit rejections are cached under the v7.19 policy and Anchor type;
* online Dream never cleans, merges, reports, or prunes Facts.

Episodic citations remain attached through the existing v7.12 merge primitive,
but an episodic-only Anchor can never enter the candidate set.  No PG memory is
written and no graph schema element is added.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mirix.log import get_logger
from mirix.services._graph_common import llm_model_from_agent
from mirix.services.graph_reconsolidator import MERGE_PROMPT, _BATCH, _PAIR_COS
from mirix.services.graph_reconsolidator_v717 import _merge_confirmed
from mirix.settings import settings

logger = get_logger(__name__)

_POLICY_VERSION = "v7.19"
_NEIGHBOURS_PER_SOURCE = 4
_INTERMEDIATE_MAX_VERIFY = 80
_FINAL_FIRST_MAX_VERIFY = 120
_FINAL_REPAIR_MAX_VERIFY = 60

# These types describe stable identity.  Event/date/content/method/other are
# intentionally absent: merging them can collapse distinct occurrences or
# temporal context even when their surface names have high cosine similarity.
_MERGE_SAFE_TYPES = {"person", "organization", "location", "object", "concept"}

Candidate = tuple[str, str, float, str]


def _pair_key(a: str, b: str, anchor_type: str) -> str:
    """Stable unordered rejection key scoped to type and merge policy."""
    left, right = sorted(
        ((a or "").strip().casefold(), (b or "").strip().casefold())
    )
    kind = (anchor_type or "").strip().casefold()
    payload = f"{_POLICY_VERSION}\n{kind}\n{left}\n{right}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _rejected_pair_keys(driver, user_id: str) -> set[str]:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """OPTIONAL MATCH (m:V7Meta {user_id:$u})
               RETURN coalesce(m.v719_rejected_pair_keys, []) AS keys""",
            u=user_id,
        )
        row = await result.single()
        return set(row["keys"] if row else [])


async def _cache_rejected_pair_keys(
    driver, user_id: str, rejected_keys: set[str]
) -> None:
    if not rejected_keys:
        return
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """MERGE (m:V7Meta {user_id:$u})
               SET m.v719_rejected_pair_keys =
                   coalesce(m.v719_rejected_pair_keys, [])
                   + [key IN $keys
                      WHERE NOT key IN coalesce(m.v719_rejected_pair_keys, [])],
                   m.v719_last_decision_at = datetime()""",
            u=user_id,
            keys=sorted(rejected_keys),
        )


async def _mark_dreamed_anchor_ids(
    driver, user_id: str, anchor_ids: list[str]
) -> None:
    if not anchor_ids:
        return
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """MERGE (m:V7Meta {user_id:$u})
               SET m.v719_dreamed_safe_anchor_ids =
                   coalesce(m.v719_dreamed_safe_anchor_ids, [])
                   + [id IN $ids
                      WHERE NOT id IN coalesce(m.v719_dreamed_safe_anchor_ids, [])],
                   m.v719_last_frontier_at = datetime()""",
            u=user_id,
            ids=sorted(set(anchor_ids)),
        )


async def _safe_delta_anchor_ids(
    driver, user_id: str, semantic_memory_ids: set[str]
) -> list[str]:
    if not semantic_memory_ids:
        return []
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE any(mid IN coalesce(a.semantic_ids, []) WHERE mid IN $batch)
                 AND toLower(coalesce(a.anchor_type, '')) IN $safe_types
               RETURN DISTINCT a.id AS id""",
            u=user_id,
            batch=sorted(semantic_memory_ids),
            safe_types=sorted(_MERGE_SAFE_TYPES),
        )
        return [row["id"] async for row in result if row["id"]]


async def _undreamed_safe_anchor_ids(driver, user_id: str) -> list[str]:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """OPTIONAL MATCH (m:V7Meta {user_id:$u})
               WITH coalesce(m.v719_dreamed_safe_anchor_ids, []) AS dreamed
               MATCH (a:V7Anchor {user_id:$u})
               WHERE size(coalesce(a.semantic_ids, [])) > 0
                 AND toLower(coalesce(a.anchor_type, '')) IN $safe_types
                 AND NOT a.id IN dreamed
               RETURN a.id AS id""",
            u=user_id,
            safe_types=sorted(_MERGE_SAFE_TYPES),
        )
        return [row["id"] async for row in result if row["id"]]


async def _safe_semantic_candidate_pairs(
    driver, user_id: str, source_anchor_ids: list[str]
) -> list[Candidate]:
    """Top-4 ANN with semantic backing and same safe type on both endpoints."""
    if not source_anchor_ids:
        return []
    pairs: dict[tuple[str, str, str], float] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $source_ids AS source_id
               MATCH (a:V7Anchor {id:source_id, user_id:$u})
               WHERE a.name_embedding IS NOT NULL
                 AND size(coalesce(a.semantic_ids, [])) > 0
                 AND toLower(coalesce(a.anchor_type, '')) IN $safe_types
               CALL db.index.vector.queryNodes(
                   'v7_anchor_name_emb', $neighbours, a.name_embedding)
               YIELD node AS b, score AS sc
               WHERE b.user_id = $u AND b.id <> a.id
                 AND size(coalesce(b.semantic_ids, [])) > 0
                 AND toLower(coalesce(b.anchor_type, '')) =
                     toLower(coalesce(a.anchor_type, ''))
                 AND sc >= $threshold
               RETURN a.name AS a, b.name AS b, sc,
                      toLower(a.anchor_type) AS anchor_type""",
            source_ids=source_anchor_ids,
            u=user_id,
            neighbours=_NEIGHBOURS_PER_SOURCE,
            threshold=_PAIR_COS,
            safe_types=sorted(_MERGE_SAFE_TYPES),
        )
        async for row in result:
            left, right = sorted((row["a"], row["b"]))
            key = (left, right, row["anchor_type"])
            pairs[key] = max(pairs.get(key, 0.0), float(row["sc"]))
    return [
        (a, b, score, anchor_type)
        for (a, b, anchor_type), score in sorted(
            pairs.items(), key=lambda item: -item[1]
        )
    ]


async def _verify_with_rejections(
    pairs: list[Candidate], agent_state: Any
) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Verify candidates and cache only explicit false decisions."""
    if not pairs:
        return [], set()
    from mirix.services.lightrag_extractor import call_openai_chat

    model = llm_model_from_agent(agent_state, default="gpt-4.1-mini")
    confirmed: list[tuple[str, str, str]] = []
    rejected: set[str] = set()
    for start in range(0, len(pairs), _BATCH):
        batch = pairs[start:start + _BATCH]
        allowed = {
            tuple(sorted((a, b))): (a, b, anchor_type)
            for a, b, _score, anchor_type in batch
        }
        payload = json.dumps(
            [{"a": a, "b": b} for a, b, _score, _type in batch],
            ensure_ascii=False,
        )
        try:
            raw = await call_openai_chat(
                MERGE_PROMPT, payload, model, temperature=0.0
            )
            blob = (
                raw
                if raw.strip().startswith("{")
                else raw[raw.find("{"): raw.rfind("}") + 1]
            )
            for item in (json.loads(blob) or {}).get("results", []):
                a, b = item.get("a"), item.get("b")
                if not isinstance(a, str) or not isinstance(b, str):
                    continue
                original = allowed.get(tuple(sorted((a, b))))
                if original is None:
                    continue
                oa, ob, anchor_type = original
                if item.get("same") is False:
                    rejected.add(_pair_key(oa, ob, anchor_type))
                    continue
                if item.get("same") is not True:
                    continue
                canonical = item.get("canonical") or ob
                if canonical not in (oa, ob):
                    continue
                confirmed.append(
                    (oa if canonical == ob else ob, canonical, canonical)
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("v7.19 merge verification batch failed (%s)", exc)
    return confirmed, rejected


async def _representative_anchor_ids(
    driver, user_id: str, confirmed: list[tuple[str, str, str]]
) -> list[str]:
    names = sorted({keep for _drop, keep, _canonical in confirmed})
    if not names:
        return []
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE a.name IN $names
                 AND size(coalesce(a.semantic_ids, [])) > 0
                 AND toLower(coalesce(a.anchor_type, '')) IN $safe_types
               RETURN DISTINCT a.id AS id""",
            u=user_id,
            names=names,
            safe_types=sorted(_MERGE_SAFE_TYPES),
        )
        return [row["id"] async for row in result if row["id"]]


async def _run_candidate_round(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    source_anchor_ids: list[str],
    rejected_keys: set[str],
    max_verify: int,
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    pairs = await _safe_semantic_candidate_pairs(
        driver, user_id, source_anchor_ids
    )
    unseen = [
        pair
        for pair in pairs
        if _pair_key(pair[0], pair[1], pair[3]) not in rejected_keys
    ]
    submitted = unseen[:max_verify]
    confirmed, rejected = await _verify_with_rejections(submitted, agent_state)
    await _cache_rejected_pair_keys(driver, user_id, rejected)
    rejected_keys.update(rejected)
    merged = await _merge_confirmed(driver, user_id, confirmed)
    return (
        {
            "candidate_pairs": len(pairs),
            "unseen_candidate_pairs": len(unseen),
            "pairs_submitted": len(submitted),
            "explicit_rejections_cached": len(rejected),
            "llm_confirmed_merges": len(confirmed),
            "anchors_merged": merged,
        },
        confirmed,
    )


async def reconsolidate_bounded_frontier(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
    final_full_graph: bool,
) -> dict[str, Any]:
    """Run one delta round or a bounded final frontier plus one repair round."""
    if driver is None:
        return {"skipped": "no_driver"}
    semantic_memory_ids = {mid for mid in semantic_memory_ids if mid}
    rejected_keys = await _rejected_pair_keys(driver, user_id)
    delta_ids = await _safe_delta_anchor_ids(
        driver, user_id, semantic_memory_ids
    )

    if final_full_graph:
        undreamed_ids = await _undreamed_safe_anchor_ids(driver, user_id)
        first_sources = sorted(set(delta_ids) | set(undreamed_ids))
        first_limit = _FINAL_FIRST_MAX_VERIFY
    else:
        undreamed_ids = []
        first_sources = delta_ids
        first_limit = _INTERMEDIATE_MAX_VERIFY

    first, confirmed = await _run_candidate_round(
        driver,
        user_id=user_id,
        agent_state=agent_state,
        source_anchor_ids=first_sources,
        rejected_keys=rejected_keys,
        max_verify=first_limit,
    )
    first.update(
        {
            "round": 1,
            "frontier": "final_delta_plus_undreamed" if final_full_graph else "delta",
            "source_semantic_anchors": len(first_sources),
            "max_verify": first_limit,
        }
    )
    rounds = [first]
    await _mark_dreamed_anchor_ids(driver, user_id, first_sources)

    if final_full_graph and first["anchors_merged"] > 0:
        repair_sources = await _representative_anchor_ids(
            driver, user_id, confirmed
        )
        if repair_sources:
            repair, _repair_confirmed = await _run_candidate_round(
                driver,
                user_id=user_id,
                agent_state=agent_state,
                source_anchor_ids=repair_sources,
                rejected_keys=rejected_keys,
                max_verify=_FINAL_REPAIR_MAX_VERIFY,
            )
            repair.update(
                {
                    "round": 2,
                    "frontier": "dirty_representatives",
                    "source_semantic_anchors": len(repair_sources),
                    "max_verify": _FINAL_REPAIR_MAX_VERIFY,
                }
            )
            rounds.append(repair)
            await _mark_dreamed_anchor_ids(driver, user_id, repair_sources)

    return {
        "final_bounded_frontier": final_full_graph,
        "batch_semantic_memories": len(semantic_memory_ids),
        "batch_safe_semantic_anchors": len(delta_ids),
        "undreamed_safe_semantic_anchors": len(undreamed_ids),
        "rounds": rounds,
        "round_count": len(rounds),
        "candidate_pairs": sum(r["candidate_pairs"] for r in rounds),
        "pairs_submitted": sum(r["pairs_submitted"] for r in rounds),
        "explicit_rejections_cached": sum(
            r["explicit_rejections_cached"] for r in rounds
        ),
        "llm_confirmed_merges": sum(
            r["llm_confirmed_merges"] for r in rounds
        ),
        "anchors_merged": sum(r["anchors_merged"] for r in rounds),
        "merge_safe_types": sorted(_MERGE_SAFE_TYPES),
        "episodic_only_candidates": 0,
        "online_fact_cleanup": "disabled",
        "semantic_facts_in_scope": 0,
        "mixed_facts_in_scope": 0,
        "episodic_only_facts_in_scope": 0,
        "conflict_candidates": 0,
        "conflicts_reported": 0,
        "conflict_samples": [],
    }
