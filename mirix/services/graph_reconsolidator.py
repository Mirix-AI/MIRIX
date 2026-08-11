"""Periodic SEMANTIC reconsolidation of the v7 graph.

``maintain_graph`` handles the *structural* redundancy — byte-identical triples,
tautologies, dead weight. This module handles what it cannot see:

* **Pass A — cluster near-duplicates.** Anchors that mean the same thing but were
  written differently ("nasal irrigation" / "nasal saline irrigation",
  "Duracell" / "Duracell batteries"). Ingest-time resolution
  (:mod:`entity_resolver`) only compares against whatever registry existed at the
  time, so late-arriving variants are never reconciled. Embedding similarity only
  proposes candidates — an LLM decides, because cosine cannot: in this store
  ``"10-20% of Overall Marketing Budget"`` and ``"5-10% of Overall Marketing
  Budget"`` sit at cos 0.975 and must NOT be merged.

* **Pass B — flag conflicts, do not resolve them.** Same subject + predicate with
  different objects looks like a contradiction but usually is not: ``include`` /
  ``offers`` / ``enhanced with`` are legitimately multi-valued, and "resolving"
  them would delete real list structure (697 such groups here, nearly all
  legitimate). Genuine contradictions live on *functional* predicates — a count, a
  price, a speed. Those are flagged for review only; nothing is deleted, because
  the single- vs multi-valued classifier has not been validated yet.
"""
from __future__ import annotations

import json

from typing import Any, Optional

from mirix.log import get_logger
from mirix.services._graph_common import llm_model_from_agent
from mirix.settings import settings

logger = get_logger(__name__)

# Cosine only proposes; the LLM disposes. Set well above the ingest-time gate so a
# periodic sweep re-examines only the genuinely close pairs.
_PAIR_COS = 0.93
_MAX_VERIFY = 120          # cap LLM calls per cycle
_BATCH = 20

MERGE_PROMPT = """You canonicalize entity names for a knowledge graph. For each PAIR below, decide whether the two names denote the SAME real-world entity/concept.
Be CONSERVATIVE — only say true when they are truly the same thing. Say false for different quantities or ranges ("5-10% of budget" vs "10-20% of budget", "42 campsites" vs "46 campsites"), different days/months, different form/model codes ("I-601" vs "I-601A"), and merely related-but-distinct concepts ("Comedy" vs "Comedians").
Prefer the more complete/specific name as the canonical one when they DO match.
Return JSON: {"results": [{"a": "...", "b": "...", "same": true/false, "canonical": "..."}, ...]}"""

# Whether a predicate is single-valued is NOT expressible as a regex — measured on this
# store, a loose pattern flagged ~5x too much ("charge fees for [Checked Bags, carry-on]",
# "emit carbon -> [40g, 120g, 16.4g]" for different transport modes — legitimate lists a
# resolver would have destroyed), while a tight one flagged nothing at all, missing even
# a genuine "shelf life -> ['4-year shelf life', '4 years']". So an LLM classifies the
# predicates instead; there are only a few hundred distinct ones and the verdict caches.
CONFLICT_PROMPT = """Each item below is one subject, one predicate, and the several values recorded for it. Decide whether the values CONTRADICT each other or are a legitimate LIST.

CONTRADICTION: they are competing answers to the same question, so at most one can be true — "costs -> [$800, $1200]", "owns -> [three bikes, four bikes]", "shelf life -> [4 years, 6 years]".
LIST: they coexist happily — themes of a book, steps of a process, benefits of a card, things a role is responsible for.

Judge the VALUES, not the wording of the predicate. When unsure answer LIST: wrongly calling a list a contradiction is the costly error.
Return JSON: {"results": [{"i": <index>, "contradiction": true/false}, ...]}"""


async def _anchor_pairs(
    driver,
    user_id: str,
    source_anchor_ids: Optional[list[str]] = None,
) -> list[tuple[str, str, float]]:
    """Near-duplicate candidates via the existing name-embedding index.

    The legacy v7.12 path leaves ``source_anchor_ids`` unset and therefore starts
    from every anchor in the user's graph.  v7.16 supplies the semantic anchors
    touched by the current ingest batch.  Only the left/source side changes: the
    ANN fan-out, threshold, target eligibility and name-based de-duplication remain
    exactly the v7.12 behavior.
    """
    if source_anchor_ids is not None and not source_anchor_ids:
        return []
    pairs: dict[tuple[str, str], float] = {}
    async with driver.session(database=settings.neo4j_database) as session:
        source_clause = (
            "UNWIND $source_anchor_ids AS source_id\n"
            "MATCH (a:V7Anchor {user_id: $u, id: source_id})"
            if source_anchor_ids is not None
            else "MATCH (a:V7Anchor {user_id: $u})"
        )
        res = await session.run(
            f"""
            {source_clause}
            WHERE a.name_embedding IS NOT NULL
            CALL db.index.vector.queryNodes('v7_anchor_name_emb', 4, a.name_embedding)
            YIELD node AS b, score AS sc
            WHERE b.user_id = $u AND b.name <> a.name AND sc >= $th
            RETURN a.name AS a, b.name AS b, sc
            """,
            u=user_id,
            th=_PAIR_COS,
            source_anchor_ids=source_anchor_ids,
        )
        async for r in res:
            key = tuple(sorted((r["a"], r["b"])))  # undirected, dedup both directions
            pairs[key] = max(pairs.get(key, 0.0), float(r["sc"]))
    return [(a, b, s) for (a, b), s in sorted(pairs.items(), key=lambda kv: -kv[1])]


async def _verify(pairs, agent_state) -> list[tuple[str, str, str]]:
    """LLM-confirmed merges: [(drop, keep, canonical)]. Cosine cannot decide these."""
    from mirix.services.lightrag_extractor import call_openai_chat

    model = llm_model_from_agent(agent_state, default="gpt-4.1-mini")
    confirmed: list[tuple[str, str, str]] = []
    for i in range(0, len(pairs), _BATCH):
        chunk = pairs[i:i + _BATCH]
        payload = json.dumps([{"a": a, "b": b} for a, b, _ in chunk], ensure_ascii=False)
        try:
            raw = await call_openai_chat(MERGE_PROMPT, payload, model, temperature=0.0)
            blob = raw if raw.strip().startswith("{") else raw[raw.find("{"): raw.rfind("}") + 1]
            for item in (json.loads(blob) or {}).get("results", []):
                if not item.get("same"):
                    continue
                a, b = item.get("a"), item.get("b")
                canon = item.get("canonical") or b
                # only accept a canonical that is one of the two names actually shown,
                # so the LLM cannot invent a third name and orphan both anchors
                if a and b and canon in (a, b):
                    confirmed.append((a if canon == b else b, canon, canon))
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconsolidate: verify batch failed (%s)", exc)
    return confirmed


async def _confirm_conflicts(groups: list[dict], agent_state) -> list[dict]:
    """Stage 2 of the funnel: judge the actual competing VALUES.

    Asking an LLM to label a bare predicate does not work — it called "include step"
    and "include specific percentages for" single-valued. Shown the values instead,
    the question becomes concrete. Fails CLOSED (returns nothing on error)."""
    if not groups:
        return []
    from mirix.services.lightrag_extractor import call_openai_chat

    model = llm_model_from_agent(agent_state, default="gpt-4.1-mini")
    out: list[dict] = []
    for i in range(0, len(groups), 25):
        chunk = groups[i:i + 25]
        payload = json.dumps(
            [{"i": j, "subject": g["subject"], "predicate": g["predicate"],
              "values": g["objects"]} for j, g in enumerate(chunk)], ensure_ascii=False)
        try:
            raw = await call_openai_chat(CONFLICT_PROMPT, payload, model, temperature=0.0)
            blob = raw if raw.strip().startswith("{") else raw[raw.find("{"): raw.rfind("}") + 1]
            for item in (json.loads(blob) or {}).get("results", []):
                idx = item.get("i")
                if item.get("contradiction") and isinstance(idx, int) and 0 <= idx < len(chunk):
                    out.append(chunk[idx])
        except Exception as exc:  # noqa: BLE001
            logger.warning("reconsolidate: conflict judging failed (%s)", exc)
    return out


async def _arity_candidates(driver, user_id: str) -> list[dict]:
    """Stage 1 of the funnel: predicates that are USUALLY single-object, judged from
    observed data rather than their wording. `include` is 75% multi-object across its
    259 uses (so: a list); `cost` is 17% (so: a candidate contradiction). This alone
    cuts 1,213 predicates to ~29 and is what makes stage 2 affordable and accurate."""
    async with driver.session(database=settings.neo4j_database) as session:
        res = await session.run(
            """
            MATCH (su)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id:$u})-[:V7_FACT_OBJECT]->(o)
            WITH f.predicate AS p, su.name AS s, collect(DISTINCT o.name) AS objs
            WITH p, count(*) AS uses,
                 sum(CASE WHEN size(objs) > 1 THEN 1 ELSE 0 END) AS multi,
                 collect(CASE WHEN size(objs) > 1 THEN {s: s, objs: objs} END) AS hits
            WHERE uses >= 3 AND multi >= 1 AND toFloat(multi) / uses <= 0.25
            RETURN p, [h IN hits WHERE h IS NOT NULL] AS hits
            """, u=user_id)
        out: list[dict] = []
        async for r in res:
            for h in r["hits"]:
                if len(h["objs"]) <= 3:
                    out.append({"subject": h["s"], "predicate": r["p"] or "",
                                "objects": h["objs"]})
        return out


async def _merge_anchor(session, user_id: str, drop: str, keep: str) -> None:
    """Fold `drop` into `keep`: union the node's own values, redirect every edge,
    then remove `drop`.

    Edges are redirected per relationship type because APOC is not installed.
    MERGE (not CREATE) on the target so redirecting never duplicates an edge that
    already exists.

    The value fold matters as much as the rewiring: a merge that keeps only the
    survivor's properties silently discards evidence. Each property gets the
    reconciliation its meaning demands —
      * ``mention_count``   SUM — both nodes' mentions are real observations, and
        the count is the graph's only frequency/importance signal.
      * ``admission_score`` MAX — the score is "how anchor-worthy is this", so the
        stronger evidence wins rather than whichever node happened to survive.
      * ``anchor_type``     the loser's specific type is promoted when the survivor
        carries the generic ``Other``.
      * ``aliases``         the loser's name (and its own aliases) are kept on the
        survivor, so the surface form stays resolvable after the node is gone.
    Set-union / max / sum are idempotent, commutative and associative, so repeated
    or reordered merges converge on the same node (Swoosh's ICAR properties);
    last-writer-wins would not.
    """
    if drop == keep:
        return
    stmts = [
        """MATCH (d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           SET k.mention_count   = coalesce(k.mention_count, 0) + coalesce(d.mention_count, 0),
               k.admission_score = CASE
                   WHEN coalesce(d.admission_score, 0) > coalesce(k.admission_score, 0)
                   THEN d.admission_score ELSE k.admission_score END,
               k.anchor_type = CASE
                   WHEN (k.anchor_type IS NULL OR toLower(k.anchor_type) = 'other')
                        AND d.anchor_type IS NOT NULL AND toLower(d.anchor_type) <> 'other'
                   THEN d.anchor_type ELSE k.anchor_type END,
               k.updated_at = datetime()
           WITH k, d
           UNWIND coalesce(k.aliases, []) + coalesce(d.aliases, []) + [d.name] AS alias
           WITH k, collect(DISTINCT alias) AS all_aliases
           SET k.aliases = [x IN all_aliases WHERE x IS NOT NULL AND x <> k.name]""",
        # v7.12 keeps the anchor's PG row ids in properties, so the union that used to
        # happen by rewiring V7_APPEARS_IN / V7_DESCRIBED_BY is a list merge. This is
        # the "node1 -> ref 20 25, node2 -> ref 40 70, merged -> 20 25 40 70" case: the
        # survivor must be able to reach every memory BOTH anchors pointed at.
        """MATCH (d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           SET k.episodic_ids = coalesce(k.episodic_ids, [])
               + [x IN coalesce(d.episodic_ids, []) WHERE NOT x IN coalesce(k.episodic_ids, [])],
               k.semantic_ids = coalesce(k.semantic_ids, [])
               + [x IN coalesce(d.semantic_ids, []) WHERE NOT x IN coalesce(k.semantic_ids, [])]""",
        """MATCH (f:V7Fact)-[r:V7_FACT_SUBJECT]->(d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           MERGE (f)-[:V7_FACT_SUBJECT]->(k) DELETE r""",
        """MATCH (f:V7Fact)-[r:V7_FACT_OBJECT]->(d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           MERGE (f)-[:V7_FACT_OBJECT]->(k) DELETE r""",
        # v7.12 arms carry their role, so the rewire MERGEs on (role) too — otherwise
        # a frame with the dropped anchor as both origin and destination would collapse
        # to one arm and lose which role survived.
        """MATCH (f:V7Fact)-[r:V7_FACT_ARG]->(d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           MERGE (f)-[:V7_FACT_ARG {role: r.role}]->(k) DELETE r""",
        """MATCH (d:V7Anchor {user_id:$u, name:$drop})-[r:V7_APPEARS_IN]->(m)
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           MERGE (k)-[:V7_APPEARS_IN]->(m) DELETE r""",
        """MATCH (d:V7Anchor {user_id:$u, name:$drop})-[r:V7_DESCRIBED_BY]->(m)
           MATCH (k:V7Anchor {user_id:$u, name:$keep})
           MERGE (k)-[:V7_DESCRIBED_BY]->(m) DELETE r""",
        """MATCH (d:V7Anchor {user_id:$u, name:$drop})-[r:V7_RELATION]->(o)
           MATCH (k:V7Anchor {user_id:$u, name:$keep}) WHERE o <> k
           MERGE (k)-[:V7_RELATION]->(o) DELETE r""",
        """MATCH (o)-[r:V7_RELATION]->(d:V7Anchor {user_id:$u, name:$drop})
           MATCH (k:V7Anchor {user_id:$u, name:$keep}) WHERE o <> k
           MERGE (o)-[:V7_RELATION]->(k) DELETE r""",
        """MATCH (d:V7Anchor {user_id:$u, name:$drop}) DETACH DELETE d""",
    ]
    for q in stmts:
        await session.run(q, u=user_id, drop=drop, keep=keep)


async def _due(driver, user_id: str, every_n: int) -> tuple[bool, int]:
    """Churn gate. Reconsolidation is LLM-priced, so it runs per N memories CHANGED
    rather than per elapsed time: cost tracks activity instead of firing on an idle
    store or falling far behind during a burst. The mark lives on a marker node so the
    graph carries its own maintenance state.

    Compares |now - mark|, not now - mark. Consolidation makes the store SHRINK — an
    auto_dream cycle here took it 962 -> 678 — so a growth-only test goes negative and
    the gate can then never fire again. Shrinking is exactly when reconsolidation is
    most warranted, since merges are what create new near-duplicates.
    """
    # The churn measure has to count something the CURRENT schema actually writes.
    # v7.12 dropped the ref-node layer, so V7MemoryRef is structurally absent there:
    # `now` was 0, `mark` was 0, and abs(0-0) >= every_n is False forever — node
    # merging silently never ran on a v7.12+ store. Measured on the v712smoke store:
    # 86 anchors, 43 facts, 0 V7MemoryRef, _due False at every_n 1/8/10, while
    # _anchor_pairs had 23 real candidates waiting ("Delta SkyMiles" / "Delta SkyMiles
    # holder", cos 0.975). Worse, maintain_graph's dead-anchor sweep sits in a separate
    # ungated try block, so a dream-enabled v7.12 run would PRUNE without MERGING —
    # exactly the ordering hazard commit 882edbc was written to remove.
    from mirix.services.graph_memory_manager_v7 import is_frame_version

    if is_frame_version():
        cypher = """
            MATCH (a:V7Anchor {user_id: $u})
            WITH sum(size(coalesce(a.episodic_ids, [])) +
                     size(coalesce(a.semantic_ids, []))) AS now
            OPTIONAL MATCH (k:V7Meta {user_id: $u})
            RETURN coalesce(now, 0) AS now,
                   coalesce(k.last_reconsolidated_at_count, 0) AS mark
            """
    else:
        cypher = """
            MATCH (m:V7MemoryRef {user_id: $u})
            WITH count(m) AS now
            OPTIONAL MATCH (k:V7Meta {user_id: $u})
            RETURN now, coalesce(k.last_reconsolidated_at_count, 0) AS mark
            """
    async with driver.session(database=settings.neo4j_database) as session:
        rec = await (await session.run(cypher, u=user_id)).single()
    now = int(rec["now"]) if rec else 0
    mark = int(rec["mark"]) if rec else 0
    return abs(now - mark) >= every_n, now


async def _mark_done(driver, user_id: str, count: int) -> None:
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """MERGE (k:V7Meta {user_id: $u})
               SET k.last_reconsolidated_at_count = $c""", u=user_id, c=count)


async def reconsolidate_graph(
    driver, *, user_id: str, agent_state: Any, dry_run: bool = False,
    every_n_memories: Optional[int] = None,
    source_anchor_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run one semantic reconsolidation cycle. Returns stats.

    Pass ``every_n_memories`` to self-gate: the cycle is skipped unless that many
    memories have been added since the last run.
    """
    stats: dict[str, Any] = {"dry_run": dry_run}
    memory_count = None
    if every_n_memories:
        due, memory_count = await _due(driver, user_id, every_n_memories)
        if not due:
            return {"skipped": "not_due", "memory_count": memory_count}

    # ---- Pass A: cluster near-duplicate anchors ----
    pairs = await _anchor_pairs(driver, user_id, source_anchor_ids)
    stats["candidate_pairs"] = len(pairs)
    confirmed = await _verify(pairs[:_MAX_VERIFY], agent_state) if pairs else []
    stats["llm_confirmed_merges"] = len(confirmed)
    merged = 0
    if confirmed and not dry_run:
        async with driver.session(database=settings.neo4j_database) as session:
            done: set[str] = set()
            for drop, keep, _ in confirmed:
                if drop in done or keep in done:
                    continue  # avoid chaining onto an anchor already removed
                await _merge_anchor(session, user_id, drop, keep)
                done.add(drop)
                merged += 1
            # merging can make two facts identical — collapse, keeping every citation
            res = await session.run(
                """
                MATCH (su)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id:$u})-[:V7_FACT_OBJECT]->(o)
                WITH toLower(su.name)+'|'+coalesce(f.predicate,'')+'|'+toLower(o.name) AS k,
                     collect(f) AS fs
                WHERE size(fs) > 1
                WITH head(fs) AS keep, tail(fs) AS dupes
                UNWIND dupes AS dupe
                OPTIONAL MATCH (dupe)-[:V7_FACT_FROM]->(m:V7MemoryRef)
                FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END |
                         MERGE (keep)-[:V7_FACT_FROM]->(m))
                WITH DISTINCT dupe
                DETACH DELETE dupe
                RETURN count(*) AS n
                """, u=user_id)
            rec = await res.single()
            stats["facts_collapsed_after_merge"] = int(rec["n"]) if rec else 0

            # same collapse for v7.12 n-ary frames: two frames that differed only by an
            # anchor now merged into one are the same assertion. Arms sorted before
            # collect so arg order can't hide a duplicate.
            res_n = await session.run(
                """
                MATCH (f:V7Fact {user_id:$u})-[r:V7_FACT_ARG]->(a:V7Anchor)
                WITH f, r.role + ':' + toLower(a.name) AS arm
                ORDER BY arm
                WITH f, collect(arm) AS arms
                WITH coalesce(f.predicate,'') + '#'
                     + reduce(acc = '', x IN arms | acc + '|' + x) AS k, collect(f) AS fs
                WHERE size(fs) > 1
                WITH head(fs) AS keep, tail(fs) AS dupes
                UNWIND dupes AS dupe
                // v7.12 citations are a property, so the union is a list concat.
                // Roles ride along positionally with the ids they belong to.
                WITH keep, dupe,
                     [i IN range(0, size(coalesce(dupe.memory_ids, [])) - 1)
                      WHERE NOT dupe.memory_ids[i] IN coalesce(keep.memory_ids, [])] AS add
                SET keep.memory_ids = coalesce(keep.memory_ids, [])
                                      + [i IN add | dupe.memory_ids[i]],
                    keep.memory_roles = coalesce(keep.memory_roles, [])
                                      + [i IN add | coalesce(dupe.memory_roles, [])[i]]
                WITH DISTINCT dupe
                DETACH DELETE dupe
                RETURN count(*) AS n
                """, u=user_id)
            rec_n = await res_n.single()
            stats["frames_collapsed_after_merge"] = int(rec_n["n"]) if rec_n else 0
    stats["anchors_merged"] = merged

    # ---- Pass B: REPORT conflicts. Never resolves, never deletes. ----
    # Two-stage funnel: observed arity narrows the field, then an LLM judges the actual
    # competing values. Report-only by design — see the module docstring for why.
    candidates = await _arity_candidates(driver, user_id)
    stats["conflict_candidates"] = len(candidates)
    conflicts = await _confirm_conflicts(candidates, agent_state)
    stats["conflicts_reported"] = len(conflicts)
    stats["conflict_samples"] = conflicts[:6]

    if every_n_memories and not dry_run and memory_count is not None:
        await _mark_done(driver, user_id, memory_count)

    logger.info("graph reconsolidation user=%s: %s", user_id,
                {k: v for k, v in stats.items() if k != "conflict_samples"})
    return stats
