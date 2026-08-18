"""v7.17 hybrid semantic-only AutoDream.

Intermediate cycles merge only from the current semantic batch and deliberately
defer every graph-wide cleanup.  The final cycle scans all semantic-backed anchors
and then performs one full semantic Fact/conflict/maintenance pass.  Episodic-only
anchors and Facts are never candidates or cleanup targets; mixed anchors remain
eligible because their semantic identity is backed by at least one semantic row,
and the v7.12 merge primitive preserves their episodic citations.
"""

from __future__ import annotations

from typing import Any

from mirix.services.graph_reconsolidator import (
    _MAX_VERIFY,
    _PAIR_COS,
    _confirm_conflicts,
    _merge_anchor,
    _verify,
)
from mirix.services.graph_reconsolidator_v714 import (
    _collapse_affected_frames,
    _nary_conflict_candidates,
    _prune_touched_dead_semantic_anchors,
    _remove_affected_degenerate_frames,
)
from mirix.services.graph_reconsolidator_v716 import _semantic_delta_anchor_ids
from mirix.settings import settings

_NEIGHBOURS_PER_SOURCE = 4  # exactly the v7.12 ANN fan-out


async def _all_semantic_anchor_ids(driver, user_id: str) -> list[str]:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE size(coalesce(a.semantic_ids, [])) > 0
               RETURN a.id AS id""",
            u=user_id,
        )
        return [row["id"] async for row in result if row["id"]]


async def _semantic_candidate_pairs(
    driver, user_id: str, source_anchor_ids: list[str]
) -> list[tuple[str, str, float]]:
    """v7.12 top-4 ANN, with semantic backing required on both endpoints."""
    if not source_anchor_ids:
        return []
    pairs: dict[tuple[str, str], float] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $source_ids AS source_id
               MATCH (a:V7Anchor {id:source_id, user_id:$u})
               WHERE a.name_embedding IS NOT NULL
                 AND size(coalesce(a.semantic_ids, [])) > 0
               CALL db.index.vector.queryNodes(
                   'v7_anchor_name_emb', $neighbours, a.name_embedding)
               YIELD node AS b, score AS sc
               WHERE b.user_id = $u AND b.id <> a.id
                 AND size(coalesce(b.semantic_ids, [])) > 0
                 AND sc >= $threshold
               RETURN a.name AS a, b.name AS b, sc""",
            source_ids=source_anchor_ids,
            u=user_id,
            neighbours=_NEIGHBOURS_PER_SOURCE,
            threshold=_PAIR_COS,
        )
        async for row in result:
            key = tuple(sorted((row["a"], row["b"])))
            pairs[key] = max(pairs.get(key, 0.0), float(row["sc"]))
    return [
        (a, b, score)
        for (a, b), score in sorted(pairs.items(), key=lambda item: -item[1])
    ]


async def _merge_confirmed(
    driver, user_id: str, confirmed: list[tuple[str, str, str]]
) -> int:
    if not confirmed:
        return 0
    merged = 0
    done: set[str] = set()
    async with driver.session(database=settings.neo4j_database) as session:
        for drop, keep, _ in confirmed:
            if drop in done or keep in done:
                continue
            await _merge_anchor(session, user_id, drop, keep)
            done.add(drop)
            merged += 1
    return merged


async def _all_semantic_scope(
    driver, user_id: str
) -> tuple[list[str], list[str]]:
    """All semantic-backed Facts and Anchors after final Anchor merging."""
    async with driver.session(database=settings.neo4j_database) as session:
        facts_result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})
               WHERE any(mid IN coalesce(f.memory_ids, [])
                         WHERE mid STARTS WITH 'sem_')
               RETURN f.id AS id""",
            u=user_id,
        )
        fact_ids = [row["id"] async for row in facts_result if row["id"]]
        anchors_result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE size(coalesce(a.semantic_ids, [])) > 0
               RETURN a.id AS id""",
            u=user_id,
        )
        anchor_ids = [row["id"] async for row in anchors_result if row["id"]]
    return fact_ids, anchor_ids


async def reconsolidate_hybrid_semantic(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
    final_full_graph: bool,
) -> dict[str, Any]:
    """Run one local cycle or the one final full-semantic cycle."""
    semantic_memory_ids = {mid for mid in semantic_memory_ids if mid}
    if final_full_graph:
        source_anchor_ids = await _all_semantic_anchor_ids(driver, user_id)
    else:
        source_anchor_ids = await _semantic_delta_anchor_ids(
            driver, user_id, semantic_memory_ids
        )

    pairs = await _semantic_candidate_pairs(driver, user_id, source_anchor_ids)
    confirmed = await _verify(pairs[:_MAX_VERIFY], agent_state) if pairs else []
    merged = await _merge_confirmed(driver, user_id, confirmed)

    stats: dict[str, Any] = {
        "final_full_semantic_sweep": final_full_graph,
        "batch_semantic_memories": len(semantic_memory_ids),
        "source_semantic_anchors": len(source_anchor_ids),
        "candidate_pairs": len(pairs),
        "llm_confirmed_merges": len(confirmed),
        "anchors_merged": merged,
        "episodic_only_candidates": 0,
    }
    if not final_full_graph:
        stats.update(
            {
                "global_cleanup": "deferred_to_final_dream",
                "semantic_facts_in_scope": 0,
                "duplicate_semantic_frames_merged": 0,
                "degenerate_semantic_frames_removed": 0,
                "dead_semantic_anchors_pruned": 0,
                "conflict_candidates": 0,
                "conflicts_reported": 0,
                "conflict_samples": [],
            }
        )
        return stats

    fact_ids, anchor_ids = await _all_semantic_scope(driver, user_id)
    duplicates = await _collapse_affected_frames(driver, user_id, fact_ids)
    degenerate = await _remove_affected_degenerate_frames(
        driver, user_id, fact_ids
    )
    dead_anchors = await _prune_touched_dead_semantic_anchors(
        driver, user_id, anchor_ids
    )
    conflict_candidates = await _nary_conflict_candidates(
        driver, user_id, fact_ids
    )
    conflicts = await _confirm_conflicts(conflict_candidates, agent_state)
    stats.update(
        {
            "global_cleanup": "completed_semantic_only",
            "semantic_facts_in_scope": len(fact_ids),
            "duplicate_semantic_frames_merged": duplicates,
            "degenerate_semantic_frames_removed": degenerate,
            "dead_semantic_anchors_pruned": dead_anchors,
            "conflict_candidates": len(conflict_candidates),
            "conflicts_reported": len(conflicts),
            "conflict_samples": conflicts[:6],
        }
    )
    return stats
