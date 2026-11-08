"""v7.15 bounded, semantic-delta AutoDream.

Normal ingest is unchanged: PostgreSQL episodic/semantic memories and the graph
are both written exactly as in v7.12.  This module is used only by graph-only
AutoDream.  It may read PG-derived semantic ids supplied by the manager, but all
mutations below are confined to Neo4j.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from mirix.services._graph_common import llm_model_from_agent
from mirix.services.graph_reconsolidator import _PAIR_COS, _merge_anchor
from mirix.services.graph_reconsolidator_v714 import (
    _batch_digest,
    _delta_anchors,
    _prune_touched_dead_semantic_anchors,
    _remove_affected_degenerate_frames,
)
from mirix.settings import settings

_ANN_FETCH_K = 64
_PER_DELTA_TOP_K = 4
_MAX_VERIFY = 60
_VERIFY_BATCH = 20

MERGE_PROMPT = """You canonicalize entity names for a knowledge graph. Decide whether each pair denotes the SAME real-world entity, event, or concept.
Be conservative. Related concepts are not the same. Different dates, quantities, event occurrences, or states are not the same. Use the bounded representative Fact context when present.
Return JSON only: {"results":[{"pair_id":"...","same":true/false,"canonical_id":"one of the supplied anchor ids"}]}"""


async def _already_processed(driver, user_id: str, digest: str) -> bool:
    async with driver.session(database=settings.neo4j_database) as session:
        rec = await (
            await session.run(
                """OPTIONAL MATCH (m:V7Meta {user_id:$u})
                   RETURN $digest IN coalesce(m.processed_v715_batch_digests, []) AS done""",
                u=user_id,
                digest=digest,
            )
        ).single()
    return bool(rec and rec["done"])


async def _mark_processed(driver, user_id: str, digest: str) -> None:
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """MERGE (m:V7Meta {user_id:$u})
               SET m.processed_v715_batch_digests = CASE
                     WHEN $digest IN coalesce(m.processed_v715_batch_digests, [])
                     THEN m.processed_v715_batch_digests
                     ELSE coalesce(m.processed_v715_batch_digests, []) + $digest END,
                   m.last_v715_dream_at = datetime()""",
            u=user_id,
            digest=digest,
        )


async def _candidate_pairs(
    driver, user_id: str, delta_anchor_ids: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """Return ID-deduplicated pairs after same-user semantic filtering.

    Neo4j vector indexes cannot pre-filter by tenant.  Fetch 64 globally for each
    delta anchor, then retain at most four same-user semantic-backed neighbours.
    """
    if not delta_anchor_ids:
        return [], 0

    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    returned_rows = 0
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $delta_ids AS delta_id
               MATCH (a:V7Anchor {id:delta_id, user_id:$u})
               WHERE a.name_embedding IS NOT NULL
                 AND size(coalesce(a.semantic_ids, [])) > 0
               CALL (a) {
                   WITH a
                   CALL db.index.vector.queryNodes(
                       'v7_anchor_name_emb', $fetch_k, a.name_embedding)
                   YIELD node AS b, score AS sc
                   WHERE b.user_id = $u AND b.id <> a.id
                     AND size(coalesce(b.semantic_ids, [])) > 0
                     AND sc >= $threshold
                   RETURN b, sc ORDER BY sc DESC LIMIT $per_delta_k
               }
               RETURN a.id AS a_id, a.name AS a_name,
                      b.id AS b_id, b.name AS b_name, sc""",
            delta_ids=delta_anchor_ids,
            u=user_id,
            fetch_k=_ANN_FETCH_K,
            per_delta_k=_PER_DELTA_TOP_K,
            threshold=_PAIR_COS,
        )
        async for row in result:
            returned_rows += 1
            a_id, b_id = row["a_id"], row["b_id"]
            key = tuple(sorted((a_id, b_id)))
            pair = {
                "a_id": a_id,
                "a_name": row["a_name"],
                "b_id": b_id,
                "b_name": row["b_name"],
                "score": float(row["sc"]),
                "delta_ids": {a_id},
            }
            existing = by_pair.get(key)
            if existing is None:
                by_pair[key] = pair
            else:
                existing["delta_ids"].add(a_id)
                if pair["score"] > existing["score"]:
                    pair["delta_ids"] = existing["delta_ids"]
                    by_pair[key] = pair

    pairs = sorted(by_pair.values(), key=lambda item: -item["score"])
    return pairs, returned_rows


async def _anchor_context(
    driver, user_id: str, anchor_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Fetch one bounded representative semantic Fact per shortlisted Anchor."""
    if not anchor_ids:
        return {}
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $ids AS anchor_id
               MATCH (a:V7Anchor {id:anchor_id, user_id:$u})
               OPTIONAL MATCH (f:V7Fact {user_id:$u})-[:V7_FACT_ARG]->(a)
               WITH a, CASE WHEN any(mid IN coalesce(f.memory_ids, [])
                                      WHERE mid STARTS WITH 'sem_')
                            THEN f ELSE null END AS sf
               ORDER BY coalesce(toString(sf.timestamp), '') DESC, sf.id
               WITH a, head([f IN collect(sf) WHERE f IS NOT NULL]) AS f
               OPTIONAL MATCH (f)-[r:V7_FACT_ARG]->(arg:V7Anchor)
               RETURN a.id AS id, a.name AS name, a.anchor_type AS anchor_type,
                      f.predicate AS predicate, toString(f.timestamp) AS timestamp,
                      coalesce(f.lit_keys, []) AS lit_keys,
                      coalesce(f.lit_vals, []) AS lit_vals,
                      collect(CASE WHEN arg IS NULL THEN null
                                   ELSE {role:r.role, name:arg.name} END) AS args""",
            ids=sorted(set(anchor_ids)),
            u=user_id,
        )
        out: dict[str, dict[str, Any]] = {}
        async for row in result:
            args = [arg for arg in row["args"] if arg]
            fact = None
            if row["predicate"]:
                fact = {
                    "predicate": row["predicate"],
                    "args": args[:6],
                    "literals": list(zip(row["lit_keys"], row["lit_vals"]))[:6],
                    "timestamp": row["timestamp"],
                }
            out[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "type": row["anchor_type"] or "Other",
                "representative_fact": fact,
            }
        return out


async def _verify(
    pairs: list[dict[str, Any]], contexts: dict[str, dict[str, Any]], agent_state: Any
) -> list[dict[str, Any]]:
    from mirix.services.lightrag_extractor import call_openai_chat

    model = llm_model_from_agent(agent_state, default="gpt-4.1-mini")
    by_pair_id: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        pair_id = f"p{index}"
        by_pair_id[pair_id] = pair
        payloads.append(
            {
                "pair_id": pair_id,
                "similarity": round(pair["score"], 6),
                "a": contexts.get(
                    pair["a_id"], {"id": pair["a_id"], "name": pair["a_name"]}
                ),
                "b": contexts.get(
                    pair["b_id"], {"id": pair["b_id"], "name": pair["b_name"]}
                ),
            }
        )

    confirmed: list[dict[str, Any]] = []
    for offset in range(0, len(payloads), _VERIFY_BATCH):
        chunk = payloads[offset : offset + _VERIFY_BATCH]
        try:
            raw = await call_openai_chat(
                MERGE_PROMPT,
                json.dumps(chunk, ensure_ascii=False, default=str),
                model,
                temperature=0.0,
            )
            blob = raw if raw.strip().startswith("{") else raw[raw.find("{") : raw.rfind("}") + 1]
            for item in (json.loads(blob) or {}).get("results", []):
                pair = by_pair_id.get(item.get("pair_id"))
                if not pair or not item.get("same"):
                    continue
                canonical_id = item.get("canonical_id")
                if canonical_id not in (pair["a_id"], pair["b_id"]):
                    continue
                keep_a = canonical_id == pair["a_id"]
                confirmed.append(
                    {
                        **pair,
                        "keep_id": pair["a_id"] if keep_a else pair["b_id"],
                        "keep_name": pair["a_name"] if keep_a else pair["b_name"],
                        "drop_id": pair["b_id"] if keep_a else pair["a_id"],
                        "drop_name": pair["b_name"] if keep_a else pair["a_name"],
                    }
                )
        except Exception:
            # Verification fails closed: no merge is safer than an invented identity.
            continue
    return confirmed


async def _affected_facts(
    driver,
    user_id: str,
    semantic_memory_ids: set[str],
    confirmed_anchor_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Batch Facts plus Facts incident only to actually confirmed merge endpoints."""
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               OPTIONAL MATCH (f)-[:V7_FACT_ARG]->(a:V7Anchor)
               WITH f, collect(DISTINCT a) AS anchors
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid IN $batch)
                  OR any(a IN anchors WHERE a.id IN $confirmed_anchor_ids)
               RETURN f.id AS id, [a IN anchors | a.id] AS anchor_ids""",
            u=user_id,
            batch=sorted(semantic_memory_ids),
            confirmed_anchor_ids=sorted(confirmed_anchor_ids),
        )
        fact_ids: list[str] = []
        anchor_ids: set[str] = set()
        async for row in result:
            fact_ids.append(row["id"])
            anchor_ids.update(x for x in row["anchor_ids"] if x)
        return fact_ids, sorted(anchor_ids)


def _fact_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    arms = tuple(sorted((str(arg["role"]), str(arg["anchor_id"])) for arg in row["args"]))
    literals = tuple(
        sorted((str(key), str(value)) for key, value in zip(row["lit_keys"], row["lit_vals"]))
    )
    return (row.get("predicate") or "", arms, literals, row.get("timestamp") or "")


async def _semantic_fact_rows(driver, user_id: str) -> list[dict[str, Any]]:
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               OPTIONAL MATCH (f)-[r:V7_FACT_ARG]->(a:V7Anchor)
               RETURN f.id AS id, f.predicate AS predicate,
                      toString(f.timestamp) AS timestamp,
                      coalesce(f.lit_keys, []) AS lit_keys,
                      coalesce(f.lit_vals, []) AS lit_vals,
                      coalesce(f.memory_ids, []) AS memory_ids,
                      coalesce(f.memory_roles, []) AS memory_roles,
                      [x IN collect(CASE WHEN a IS NULL THEN null
                            ELSE {role:r.role, anchor_id:a.id} END)
                       WHERE x IS NOT NULL] AS args""",
            u=user_id,
        )
        return [dict(row) async for row in result]


async def _collapse_affected_frames(
    driver, user_id: str, affected_fact_ids: list[str]
) -> int:
    """Collapse exact semantic Fact duplicates triggered by the local delta."""
    if not affected_fact_ids:
        return 0
    affected = set(affected_fact_ids)
    rows = await _semantic_fact_rows(driver, user_id)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_fact_signature(row)].append(row)

    updates: list[dict[str, Any]] = []
    deleted = 0
    for facts in groups.values():
        if len(facts) < 2 or not any(fact["id"] in affected for fact in facts):
            continue
        facts.sort(key=lambda fact: fact["id"])
        keep, duplicates = facts[0], facts[1:]
        memory_ids: list[str] = []
        memory_roles: list[Any] = []
        for fact in facts:
            roles = fact.get("memory_roles") or []
            for index, memory_id in enumerate(fact.get("memory_ids") or []):
                if memory_id in memory_ids:
                    continue
                memory_ids.append(memory_id)
                memory_roles.append(roles[index] if index < len(roles) else None)
        updates.append(
            {
                "keep": keep["id"],
                "duplicates": [fact["id"] for fact in duplicates],
                "memory_ids": memory_ids,
                "memory_roles": memory_roles,
            }
        )
        deleted += len(duplicates)

    if not updates:
        return 0
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """UNWIND $updates AS row
               MATCH (keep:V7Fact {id:row.keep, user_id:$u})
               SET keep.memory_ids = row.memory_ids,
                   keep.memory_roles = row.memory_roles
               WITH row
               MATCH (duplicate:V7Fact {user_id:$u})
               WHERE duplicate.id IN row.duplicates
               DETACH DELETE duplicate""",
            updates=updates,
            u=user_id,
        )
    return deleted


async def reconsolidate_semantic_delta(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
) -> dict[str, Any]:
    if driver is None:
        return {"skipped": "no_driver"}
    semantic_memory_ids = {memory_id for memory_id in semantic_memory_ids if memory_id}
    if not semantic_memory_ids:
        return {"skipped": "empty_semantic_delta"}

    digest = _batch_digest(semantic_memory_ids)
    if await _already_processed(driver, user_id, digest):
        return {
            "skipped": "batch_already_processed",
            "batch_semantic_memories": len(semantic_memory_ids),
        }

    delta = await _delta_anchors(driver, user_id, semantic_memory_ids)
    pairs, per_delta_rows = await _candidate_pairs(
        driver, user_id, [anchor["id"] for anchor in delta]
    )
    selected = pairs[:_MAX_VERIFY]
    selected_ids = [
        anchor_id
        for pair in selected
        for anchor_id in (pair["a_id"], pair["b_id"])
    ]
    contexts = await _anchor_context(driver, user_id, selected_ids)
    confirmed = await _verify(selected, contexts, agent_state) if selected else []

    confirmed_anchor_ids = {
        anchor_id
        for pair in confirmed
        for anchor_id in (pair["drop_id"], pair["keep_id"])
    }
    affected_fact_ids, touched_anchor_ids = await _affected_facts(
        driver, user_id, semantic_memory_ids, confirmed_anchor_ids
    )

    merged = 0
    merged_ids: set[str] = set()
    if confirmed:
        async with driver.session(database=settings.neo4j_database) as session:
            for pair in confirmed:
                if pair["drop_id"] in merged_ids or pair["keep_id"] in merged_ids:
                    continue
                await _merge_anchor(
                    session, user_id, pair["drop_name"], pair["keep_name"]
                )
                merged_ids.add(pair["drop_id"])
                merged += 1

    duplicates = await _collapse_affected_frames(driver, user_id, affected_fact_ids)
    degenerate = await _remove_affected_degenerate_frames(
        driver, user_id, affected_fact_ids
    )
    dead_anchors = await _prune_touched_dead_semantic_anchors(
        driver, user_id, touched_anchor_ids
    )
    await _mark_processed(driver, user_id, digest)

    return {
        "batch_semantic_memories": len(semantic_memory_ids),
        "delta_semantic_anchors": len(delta),
        "ann_fetch_k": _ANN_FETCH_K,
        "per_delta_top_k": _PER_DELTA_TOP_K,
        "per_delta_top4_rows": per_delta_rows,
        "unique_candidate_pairs": len(pairs),
        "selected_for_verification": len(selected),
        "deferred_pairs": max(0, len(pairs) - len(selected)),
        "verification_llm_calls_max": (len(selected) + _VERIFY_BATCH - 1) // _VERIFY_BATCH,
        "llm_confirmed_merges": len(confirmed),
        "anchors_merged": merged,
        "semantic_facts_in_scope": len(affected_fact_ids),
        "duplicate_semantic_frames_merged": duplicates,
        "degenerate_semantic_frames_removed": degenerate,
        "dead_semantic_anchors_pruned": dead_anchors,
        "episodic_only_candidates": 0,
        "conflict_audit": "deferred_offline",
        "conflict_llm_calls": 0,
        "batch_digest": digest,
    }
