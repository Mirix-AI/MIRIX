"""v7.18 iterative semantic-only AutoDream.

This revision keeps v7.12 ingest and retrieval unchanged.  Dream owns only the
semantic identity layer:

* intermediate cycles start from the current semantic batch;
* the final cycle starts from every semantic-backed Anchor;
* both candidate endpoints must be semantic-backed;
* candidates are verified 120 at a time, merged, and recomputed against the
  updated graph;
* explicit LLM rejections are cached, so a later round or Dream never pays to
  judge the same unchanged name pair again;
* only semantic-only Facts are cleaned at the end.  A Fact carrying any
  episodic citation is an occurrence-bearing assertion and is immutable here.

No PG memory is written and no occurrence node or relationship type is added.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from mirix.log import get_logger
from mirix.services._graph_common import llm_model_from_agent
from mirix.services.graph_reconsolidator import (
    MERGE_PROMPT,
    _BATCH,
    _MAX_VERIFY,
    _confirm_conflicts,
)
from mirix.services.graph_reconsolidator_v716 import _semantic_delta_anchor_ids
from mirix.services.graph_reconsolidator_v717 import (
    _all_semantic_anchor_ids,
    _merge_confirmed,
    _semantic_candidate_pairs,
)
from mirix.settings import settings

logger = get_logger(__name__)

_SUBJECT_ROLES = {
    "agent", "subject", "experiencer", "owner", "applicant", "buyer",
    "creator", "artist", "giver", "recipient",
}


def _pair_key(a: str, b: str) -> str:
    """Stable unordered key for an explicit LLM rejection."""
    left, right = sorted(((a or "").strip().casefold(), (b or "").strip().casefold()))
    return hashlib.sha256(f"{left}\n{right}".encode("utf-8")).hexdigest()


async def _rejected_pair_keys(driver, user_id: str) -> set[str]:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """OPTIONAL MATCH (m:V7Meta {user_id:$u})
               RETURN coalesce(m.v718_rejected_pair_keys, []) AS keys""",
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
               SET m.v718_rejected_pair_keys =
                   coalesce(m.v718_rejected_pair_keys, [])
                   + [key IN $keys
                      WHERE NOT key IN coalesce(m.v718_rejected_pair_keys, [])],
                   m.v718_last_decision_at = datetime()""",
            u=user_id,
            keys=sorted(rejected_keys),
        )


async def _verify_with_rejections(
    pairs: list[tuple[str, str, float]], agent_state: Any
) -> tuple[list[tuple[str, str, str]], set[str]]:
    """Return confirmed merges and only *explicit* false decisions.

    Missing/malformed results and failed LLM batches are deliberately not cached:
    absence of an answer is not a rejection and must remain retryable.
    """
    if not pairs:
        return [], set()
    from mirix.services.lightrag_extractor import call_openai_chat

    model = llm_model_from_agent(agent_state, default="gpt-4.1-mini")
    confirmed: list[tuple[str, str, str]] = []
    rejected: set[str] = set()
    for start in range(0, len(pairs), _BATCH):
        batch = pairs[start:start + _BATCH]
        allowed = {
            tuple(sorted((a, b))): (a, b)
            for a, b, _score in batch
        }
        payload = json.dumps(
            [{"a": a, "b": b} for a, b, _score in batch],
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
                oa, ob = original
                if item.get("same") is False:
                    rejected.add(_pair_key(oa, ob))
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
            logger.warning("v7.18 merge verification batch failed (%s)", exc)
    return confirmed, rejected


def _is_semantic_only(memory_ids: list[str]) -> bool:
    return bool(memory_ids) and all(str(mid).startswith("sem_") for mid in memory_ids)


async def _semantic_only_fact_rows(driver, user_id: str) -> list[dict[str, Any]]:
    """Materialize semantic-only Fact signatures after Anchor merges."""
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})-[r:V7_FACT_ARG]->(a:V7Anchor)
               WHERE size(coalesce(f.memory_ids, [])) > 0
                 AND all(mid IN f.memory_ids WHERE mid STARTS WITH 'sem_')
               RETURN f.id AS id, f.predicate AS predicate,
                      coalesce(f.lit_keys, []) AS lit_keys,
                      coalesce(f.lit_vals, []) AS lit_vals,
                      toString(f.timestamp) AS timestamp,
                      coalesce(f.memory_ids, []) AS memory_ids,
                      coalesce(f.memory_roles, []) AS memory_roles,
                      collect({role:r.role, anchor_id:a.id}) AS args""",
            u=user_id,
        )
        return [dict(row) async for row in result]


def _fact_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    args = tuple(sorted((arg.get("role") or "", arg.get("anchor_id") or "")
                        for arg in row.get("args", [])))
    literals = tuple(sorted(
        (str(key), str(value))
        for key, value in zip(row.get("lit_keys", []), row.get("lit_vals", []))
    ))
    return (
        row.get("predicate") or "",
        args,
        literals,
        row.get("timestamp") or "",
    )


async def _merge_exact_semantic_only_facts(
    driver, user_id: str, rows: list[dict[str, Any]]
) -> int:
    """Collapse only byte-equivalent semantic assertions, preserving citations."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _is_semantic_only(row.get("memory_ids", [])):
            groups[_fact_signature(row)].append(row)

    merged = 0
    async with driver.session(database=settings.neo4j_database) as session:
        for facts in groups.values():
            if len(facts) < 2:
                continue
            keep = sorted(facts, key=lambda item: item["id"])[0]
            for duplicate in facts:
                if duplicate["id"] == keep["id"]:
                    continue
                await session.run(
                    """MATCH (k:V7Fact {id:$keep, user_id:$u})
                       MATCH (d:V7Fact {id:$drop, user_id:$u})
                       WITH k, d,
                            [i IN range(0, size(coalesce(d.memory_ids, [])) - 1)
                             WHERE NOT d.memory_ids[i] IN coalesce(k.memory_ids, [])] AS add
                       SET k.memory_ids = coalesce(k.memory_ids, [])
                                          + [i IN add | d.memory_ids[i]],
                           k.memory_roles = coalesce(k.memory_roles, [])
                                          + [i IN add | coalesce(d.memory_roles, [])[i]]
                       DETACH DELETE d""",
                    u=user_id,
                    keep=keep["id"],
                    drop=duplicate["id"],
                )
                merged += 1
    return merged


async def _remove_semantic_only_degenerate_facts(driver, user_id: str) -> int:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})
               WHERE size(coalesce(f.memory_ids, [])) > 0
                 AND all(mid IN f.memory_ids WHERE mid STARTS WITH 'sem_')
               OPTIONAL MATCH (f)-[:V7_FACT_ARG]->(a:V7Anchor)
               WITH f, count(DISTINCT a) AS arity
               WHERE arity < 2
               DETACH DELETE f
               RETURN count(*) AS n""",
            u=user_id,
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def _prune_dead_semantic_only_anchors(driver, user_id: str) -> int:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE size(coalesce(a.semantic_ids, [])) > 0
                 AND size(coalesce(a.episodic_ids, [])) = 0
                 AND NOT (a)<-[:V7_FACT_ARG]-(:V7Fact)
                 AND size(a.semantic_ids) <= 1
               DETACH DELETE a
               RETURN count(*) AS n""",
            u=user_id,
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def _semantic_only_conflict_candidates(
    driver, user_id: str
) -> list[dict[str, Any]]:
    """Report conflicts only among semantic-only assertions."""
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})-[r:V7_FACT_ARG]->(a:V7Anchor)
               WHERE size(coalesce(f.memory_ids, [])) > 0
                 AND all(mid IN f.memory_ids WHERE mid STARTS WITH 'sem_')
               RETURN f.id AS id, f.predicate AS predicate,
                      collect({role:r.role, name:a.name}) AS args,
                      coalesce(f.lit_keys, []) AS lit_keys,
                      coalesce(f.lit_vals, []) AS lit_vals""",
            u=user_id,
        )
        rows = [dict(row) async for row in result]

    groups: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    for row in rows:
        subjects = tuple(sorted(
            f"{arg['role']}:{arg['name']}"
            for arg in row["args"]
            if arg["role"] in _SUBJECT_ROLES
        ))
        if not subjects:
            continue
        values = {
            f"{arg['role']}:{arg['name']}"
            for arg in row["args"]
            if arg["role"] not in _SUBJECT_ROLES
        }
        values.update(
            f"{key}:{value}"
            for key, value in zip(row["lit_keys"], row["lit_vals"])
        )
        groups[(row.get("predicate") or "", subjects)].update(values)

    return [
        {
            "subject": "; ".join(subjects),
            "predicate": predicate,
            "objects": sorted(values),
        }
        for (predicate, subjects), values in groups.items()
        if 1 < len(values) <= 3
    ]


async def _run_candidate_round(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    source_anchor_ids: list[str],
    rejected_keys: set[str],
) -> dict[str, Any]:
    pairs = await _semantic_candidate_pairs(driver, user_id, source_anchor_ids)
    unseen = [pair for pair in pairs if _pair_key(pair[0], pair[1]) not in rejected_keys]
    submitted = unseen[:_MAX_VERIFY]
    confirmed, rejected = await _verify_with_rejections(submitted, agent_state)
    await _cache_rejected_pair_keys(driver, user_id, rejected)
    rejected_keys.update(rejected)
    merged = await _merge_confirmed(driver, user_id, confirmed)
    return {
        "candidate_pairs": len(pairs),
        "unseen_candidate_pairs": len(unseen),
        "pairs_submitted": len(submitted),
        "explicit_rejections_cached": len(rejected),
        "llm_confirmed_merges": len(confirmed),
        "anchors_merged": merged,
    }


async def reconsolidate_iterative_semantic(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
    final_full_graph: bool,
) -> dict[str, Any]:
    """Run one delta round or final iterative full-semantic consolidation."""
    if driver is None:
        return {"skipped": "no_driver"}
    semantic_memory_ids = {mid for mid in semantic_memory_ids if mid}
    rejected_keys = await _rejected_pair_keys(driver, user_id)

    rounds: list[dict[str, Any]] = []
    if final_full_graph:
        # Scheme B: merge each 120-pair batch before recomputing the candidate
        # frontier. Stop when the current graph has no unseen pair, or when a full
        # round produces no structural progress.
        while True:
            source_anchor_ids = await _all_semantic_anchor_ids(driver, user_id)
            result = await _run_candidate_round(
                driver,
                user_id=user_id,
                agent_state=agent_state,
                source_anchor_ids=source_anchor_ids,
                rejected_keys=rejected_keys,
            )
            result["source_semantic_anchors"] = len(source_anchor_ids)
            result["round"] = len(rounds) + 1
            rounds.append(result)
            if result["unseen_candidate_pairs"] == 0:
                break
            if result["anchors_merged"] == 0:
                break
    else:
        source_anchor_ids = await _semantic_delta_anchor_ids(
            driver, user_id, semantic_memory_ids
        )
        result = await _run_candidate_round(
            driver,
            user_id=user_id,
            agent_state=agent_state,
            source_anchor_ids=source_anchor_ids,
            rejected_keys=rejected_keys,
        )
        result["source_semantic_anchors"] = len(source_anchor_ids)
        result["round"] = 1
        rounds.append(result)

    stats: dict[str, Any] = {
        "final_full_semantic_sweep": final_full_graph,
        "batch_semantic_memories": len(semantic_memory_ids),
        "rounds": rounds,
        "round_count": len(rounds),
        "candidate_pairs": sum(r["candidate_pairs"] for r in rounds),
        "pairs_submitted": sum(r["pairs_submitted"] for r in rounds),
        "explicit_rejections_cached": sum(
            r["explicit_rejections_cached"] for r in rounds
        ),
        "llm_confirmed_merges": sum(r["llm_confirmed_merges"] for r in rounds),
        "anchors_merged": sum(r["anchors_merged"] for r in rounds),
        "episodic_only_candidates": 0,
    }
    if not final_full_graph:
        stats.update(
            {
                "global_cleanup": "deferred_to_final_dream",
                "semantic_only_facts_in_scope": 0,
                "mixed_facts_in_scope": 0,
                "episodic_only_facts_in_scope": 0,
                "duplicate_semantic_only_facts_merged": 0,
                "degenerate_semantic_only_facts_removed": 0,
                "dead_semantic_only_anchors_pruned": 0,
                "conflict_candidates": 0,
                "conflicts_reported": 0,
                "conflict_samples": [],
            }
        )
        return stats

    fact_rows = await _semantic_only_fact_rows(driver, user_id)
    duplicates = await _merge_exact_semantic_only_facts(driver, user_id, fact_rows)
    degenerate = await _remove_semantic_only_degenerate_facts(driver, user_id)
    dead_anchors = await _prune_dead_semantic_only_anchors(driver, user_id)
    conflict_candidates = await _semantic_only_conflict_candidates(driver, user_id)
    conflicts = await _confirm_conflicts(conflict_candidates, agent_state)
    stats.update(
        {
            "global_cleanup": "completed_semantic_only_strict",
            "semantic_only_facts_in_scope": len(fact_rows),
            "mixed_facts_in_scope": 0,
            "episodic_only_facts_in_scope": 0,
            "duplicate_semantic_only_facts_merged": duplicates,
            "degenerate_semantic_only_facts_removed": degenerate,
            "dead_semantic_only_anchors_pruned": dead_anchors,
            "conflict_candidates": len(conflict_candidates),
            "conflicts_reported": len(conflicts),
            "conflict_samples": conflicts[:6],
        }
    )
    return stats
