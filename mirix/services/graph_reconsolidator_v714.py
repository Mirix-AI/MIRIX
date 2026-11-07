"""v7.14 incremental, semantic-only AutoDream.

The current ingest remains deliberately dual-source: episodic and semantic rows may
both create anchors and n-ary facts.  Dream scope is narrower:

* the *source* side is only anchors touched by the current semantic-memory batch;
* the comparison side is every semantic-backed anchor owned by the user;
* episodic-only anchors and facts never become consolidation candidates;
* deterministic Fact cleanup is limited to the batch and anchors actually merged.

This makes each five-chunk dream proportional to the new semantic delta rather than
re-running an increasingly expensive old-old comparison over the whole graph.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from mirix.services.graph_reconsolidator import (
    _MAX_VERIFY,
    _PAIR_COS,
    _confirm_conflicts,
    _merge_anchor,
    _verify,
)
from mirix.settings import settings

_NEIGHBOURS_PER_DELTA_ANCHOR = 64
_SUBJECT_ROLES = {
    "agent", "subject", "experiencer", "owner", "applicant", "buyer",
    "creator", "artist", "giver", "recipient",
}


def _batch_digest(semantic_memory_ids: set[str]) -> str:
    payload = "\n".join(sorted(semantic_memory_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _already_processed(driver, user_id: str, digest: str) -> bool:
    async with driver.session(database=settings.neo4j_database) as session:
        rec = await (
            await session.run(
                """OPTIONAL MATCH (m:V7Meta {user_id:$u})
                   RETURN m.last_v714_batch_digest AS digest""",
                u=user_id,
            )
        ).single()
    return bool(rec and rec["digest"] == digest)


async def _mark_processed(
    driver, user_id: str, digest: str, semantic_memory_ids: set[str]
) -> None:
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """MERGE (m:V7Meta {user_id:$u})
               SET m.last_v714_batch_digest = $digest,
                   m.last_v714_batch_size = $size,
                   m.last_v714_dream_at = datetime()""",
            u=user_id,
            digest=digest,
            size=len(semantic_memory_ids),
        )


async def _delta_anchors(
    driver, user_id: str, semantic_memory_ids: set[str]
) -> list[dict[str, str]]:
    """Semantic-backed anchors touched by at least one memory in this batch."""
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (a:V7Anchor {user_id:$u})
               WHERE any(mid IN coalesce(a.semantic_ids, []) WHERE mid IN $batch)
               RETURN a.id AS id, a.name AS name""",
            u=user_id,
            batch=sorted(semantic_memory_ids),
        )
        return [dict(row) async for row in result]


async def _candidate_pairs(
    driver, user_id: str, delta_anchor_ids: list[str]
) -> list[tuple[str, str, float]]:
    """Delta semantic anchors versus the full semantic-backed anchor registry.

    Starting the query from ``delta_anchor_ids`` proves that every returned pair has
    at least one current-batch endpoint. The target-side semantic_ids predicate is
    the strict episodic-isolation rule: an episodic-only anchor is not even proposed
    to the LLM.
    """
    if not delta_anchor_ids:
        return []
    pairs: dict[tuple[str, str], float] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $delta_ids AS delta_id
               MATCH (a:V7Anchor {id:delta_id, user_id:$u})
               WHERE a.name_embedding IS NOT NULL
                 AND size(coalesce(a.semantic_ids, [])) > 0
               CALL db.index.vector.queryNodes(
                   'v7_anchor_name_emb', $neighbours, a.name_embedding)
               YIELD node AS b, score AS sc
               WHERE b.user_id = $u AND b.id <> a.id
                 AND size(coalesce(b.semantic_ids, [])) > 0
                 AND sc >= $threshold
               RETURN a.name AS a, b.name AS b, sc""",
            delta_ids=delta_anchor_ids,
            u=user_id,
            neighbours=_NEIGHBOURS_PER_DELTA_ANCHOR,
            threshold=_PAIR_COS,
        )
        async for row in result:
            key = tuple(sorted((row["a"], row["b"])))
            pairs[key] = max(pairs.get(key, 0.0), float(row["sc"]))
    return [
        (a, b, score)
        for (a, b), score in sorted(pairs.items(), key=lambda item: -item[1])
    ]


async def _affected_facts(
    driver,
    user_id: str,
    semantic_memory_ids: set[str],
    anchor_names: set[str],
) -> tuple[list[str], list[str]]:
    """Return semantic-backed Fact ids and Anchor ids in the touched subgraph."""
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               OPTIONAL MATCH (f)-[:V7_FACT_ARG]->(a:V7Anchor)
               WITH f, collect(DISTINCT a) AS anchors
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid IN $batch)
                  OR any(a IN anchors WHERE a.name IN $anchor_names)
               RETURN f.id AS id, [a IN anchors | a.id] AS anchor_ids""",
            u=user_id,
            batch=sorted(semantic_memory_ids),
            anchor_names=sorted(anchor_names),
        )
        fact_ids: list[str] = []
        anchor_ids: set[str] = set()
        async for row in result:
            fact_ids.append(row["id"])
            anchor_ids.update(x for x in row["anchor_ids"] if x)
        return fact_ids, sorted(anchor_ids)


async def _collapse_affected_frames(
    driver, user_id: str, affected_fact_ids: list[str]
) -> int:
    """Merge exact semantic Fact duplicates when one member is in this batch.

    Timestamp and literal values are part of the v7.14 signature. Earlier cleanup
    ignored them and could collapse two otherwise-identical events that happened on
    different dates. Episodic-only Facts are excluded from both the affected and the
    counterpart sides.
    """
    if not affected_fact_ids:
        return 0
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $affected AS affected_id
               MATCH (af:V7Fact {id:affected_id, user_id:$u})-[ar:V7_FACT_ARG]->(aa:V7Anchor)
               WHERE any(mid IN coalesce(af.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               WITH af, ar.role + ':' + toLower(aa.name) AS arm
               ORDER BY af.id, arm
               WITH af, collect(arm) AS arms
               WITH collect(DISTINCT coalesce(af.predicate,'') + '#'
                    + reduce(acc='', x IN arms | acc + '|' + x)
                    + '#lit=' + CASE
                        WHEN size(coalesce(af.lit_keys, [])) = 0 THEN ''
                        ELSE reduce(acc='', i IN range(0, size(af.lit_keys) - 1) |
                             acc + '|' + af.lit_keys[i] + ':'
                             + coalesce(toString(af.lit_vals[i]), '')) END
                    + '#ts=' + coalesce(toString(af.timestamp), '')) AS affected_keys
               MATCH (f:V7Fact {user_id:$u})-[r:V7_FACT_ARG]->(a:V7Anchor)
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               WITH affected_keys, f, r.role + ':' + toLower(a.name) AS arm
               ORDER BY f.id, arm
               WITH affected_keys, f, collect(arm) AS arms
               WITH affected_keys,
                    coalesce(f.predicate,'') + '#'
                    + reduce(acc='', x IN arms | acc + '|' + x)
                    + '#lit=' + CASE
                        WHEN size(coalesce(f.lit_keys, [])) = 0 THEN ''
                        ELSE reduce(acc='', i IN range(0, size(f.lit_keys) - 1) |
                             acc + '|' + f.lit_keys[i] + ':'
                             + coalesce(toString(f.lit_vals[i]), '')) END
                    + '#ts=' + coalesce(toString(f.timestamp), '') AS fact_key,
                    f
               WHERE fact_key IN affected_keys
               WITH fact_key, collect(f) AS facts
               WHERE size(facts) > 1
               WITH head(facts) AS keep, tail(facts) AS duplicates
               UNWIND duplicates AS duplicate
               WITH keep, duplicate,
                    [i IN range(0, size(coalesce(duplicate.memory_ids, [])) - 1)
                     WHERE NOT duplicate.memory_ids[i] IN coalesce(keep.memory_ids, [])] AS add
               SET keep.memory_ids = coalesce(keep.memory_ids, [])
                                     + [i IN add | duplicate.memory_ids[i]],
                   keep.memory_roles = coalesce(keep.memory_roles, [])
                                     + [i IN add | coalesce(duplicate.memory_roles, [])[i]]
               WITH DISTINCT duplicate
               DETACH DELETE duplicate
               RETURN count(*) AS n""",
            u=user_id,
            affected=affected_fact_ids,
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def _remove_affected_degenerate_frames(
    driver, user_id: str, affected_fact_ids: list[str]
) -> int:
    if not affected_fact_ids:
        return 0
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $affected AS affected_id
               MATCH (f:V7Fact {id:affected_id, user_id:$u})-[:V7_FACT_ARG]->(a:V7Anchor)
               WHERE any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               WITH f, count(DISTINCT a) AS arity
               WHERE arity < 2
               DETACH DELETE f
               RETURN count(*) AS n""",
            u=user_id,
            affected=affected_fact_ids,
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def _prune_touched_dead_semantic_anchors(
    driver, user_id: str, touched_anchor_ids: list[str]
) -> int:
    if not touched_anchor_ids:
        return 0
    async with driver.session(database=settings.neo4j_database) as session:
        result = await session.run(
            """UNWIND $anchor_ids AS anchor_id
               MATCH (a:V7Anchor {id:anchor_id, user_id:$u})
               WHERE size(coalesce(a.semantic_ids, [])) > 0
                 AND NOT (a)<-[:V7_FACT_ARG]-(:V7Fact)
                 AND size(coalesce(a.semantic_ids, []))
                     + size(coalesce(a.episodic_ids, [])) <= 1
               DETACH DELETE a
               RETURN count(*) AS n""",
            u=user_id,
            anchor_ids=touched_anchor_ids,
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def _nary_conflict_candidates(
    driver, user_id: str, affected_fact_ids: list[str]
) -> list[dict[str, Any]]:
    """Build report-only conflict candidates around affected n-ary semantic Facts."""
    if not affected_fact_ids:
        return []
    async with driver.session(database=settings.neo4j_database) as session:
        predicates_result = await session.run(
            """UNWIND $ids AS id
               MATCH (f:V7Fact {id:id, user_id:$u})
               RETURN collect(DISTINCT f.predicate) AS predicates""",
            ids=affected_fact_ids,
            u=user_id,
        )
        row = await predicates_result.single()
        predicates = [p for p in (row["predicates"] if row else []) if p]
        if not predicates:
            return []
        result = await session.run(
            """MATCH (f:V7Fact {user_id:$u})-[r:V7_FACT_ARG]->(a:V7Anchor)
               WHERE f.predicate IN $predicates
                 AND any(mid IN coalesce(f.memory_ids, []) WHERE mid STARTS WITH 'sem_')
               RETURN f.id AS id, f.predicate AS predicate,
                      collect({role:r.role, name:a.name}) AS args,
                      coalesce(f.lit_keys, []) AS lit_keys,
                      coalesce(f.lit_vals, []) AS lit_vals""",
            u=user_id,
            predicates=predicates,
        )
        rows = [dict(item) async for item in result]

    groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = defaultdict(
        lambda: {"values": set(), "affected": False}
    )
    affected = set(affected_fact_ids)
    for row in rows:
        subjects = sorted(
            f"{arg['role']}:{arg['name']}"
            for arg in row["args"]
            if arg["role"] in _SUBJECT_ROLES
        )
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
        key = (row["predicate"] or "", tuple(subjects))
        groups[key]["values"].update(values)
        groups[key]["affected"] |= row["id"] in affected

    candidates: list[dict[str, Any]] = []
    for (predicate, subjects), group in groups.items():
        values = sorted(group["values"])
        if group["affected"] and 1 < len(values) <= 3:
            candidates.append(
                {
                    "subject": "; ".join(subjects),
                    "predicate": predicate,
                    "objects": values,
                }
            )
    return candidates


async def reconsolidate_semantic_delta(
    driver,
    *,
    user_id: str,
    agent_state: Any,
    semantic_memory_ids: set[str],
) -> dict[str, Any]:
    """Run one restart-safe v7.14 semantic-delta dream cycle."""
    if driver is None:
        return {"skipped": "no_driver"}
    semantic_memory_ids = {mid for mid in semantic_memory_ids if mid}
    if not semantic_memory_ids:
        return {"skipped": "empty_semantic_delta"}

    digest = _batch_digest(semantic_memory_ids)
    if await _already_processed(driver, user_id, digest):
        return {"skipped": "batch_already_processed", "batch_semantic_memories": len(semantic_memory_ids)}

    delta = await _delta_anchors(driver, user_id, semantic_memory_ids)
    delta_ids = [anchor["id"] for anchor in delta]
    pairs = await _candidate_pairs(driver, user_id, delta_ids)
    confirmed = await _verify(pairs[:_MAX_VERIFY], agent_state) if pairs else []

    # Facts around a target-side old anchor become in-scope only when the LLM
    # actually approved that merge. A merely similar-but-rejected old anchor must
    # not cause unrelated old Facts to enter this incremental cycle.
    confirmed_names = {name for merge in confirmed for name in merge[:2]}
    delta_names = {anchor["name"] for anchor in delta}
    affected_fact_ids, touched_anchor_ids = await _affected_facts(
        driver,
        user_id,
        semantic_memory_ids,
        delta_names | confirmed_names,
    )

    merged = 0
    done: set[str] = set()
    if confirmed:
        async with driver.session(database=settings.neo4j_database) as session:
            for drop, keep, _ in confirmed:
                if drop in done or keep in done:
                    continue
                await _merge_anchor(session, user_id, drop, keep)
                done.add(drop)
                merged += 1

    duplicates = await _collapse_affected_frames(driver, user_id, affected_fact_ids)
    degenerate = await _remove_affected_degenerate_frames(
        driver, user_id, affected_fact_ids
    )
    dead_anchors = await _prune_touched_dead_semantic_anchors(
        driver, user_id, touched_anchor_ids
    )

    conflict_candidates = await _nary_conflict_candidates(
        driver, user_id, affected_fact_ids
    )
    conflicts = await _confirm_conflicts(conflict_candidates, agent_state)

    await _mark_processed(driver, user_id, digest, semantic_memory_ids)
    return {
        "batch_semantic_memories": len(semantic_memory_ids),
        "delta_semantic_anchors": len(delta),
        "candidate_pairs": len(pairs),
        "llm_confirmed_merges": len(confirmed),
        "anchors_merged": merged,
        "semantic_facts_in_scope": len(affected_fact_ids),
        "duplicate_semantic_frames_merged": duplicates,
        "degenerate_semantic_frames_removed": degenerate,
        "dead_semantic_anchors_pruned": dead_anchors,
        "conflict_candidates": len(conflict_candidates),
        "conflicts_reported": len(conflicts),
        "conflict_samples": conflicts[:6],
        "episodic_only_candidates": 0,
        "batch_digest": digest,
    }
