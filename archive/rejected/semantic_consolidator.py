"""Auto-dream v2 — incremental semantic consolidation.

The accepted redesign of the dream's flat-store stage (see
docs/graph_memory_v7/development_history.md, phase 9). Four invariants, all by
construction rather than by prompt:

1. **Episodic is immutable.** This module never reads or writes episodic rows;
   the event log and its temporal chain cannot be damaged.
2. **Raw memories are never modified.** Consolidation is additive: the product
   is a NEW semantic row (``source="auto_dream"``); the only rows ever deleted
   are this module's own earlier products when they get re-consolidated.
3. **Provenance is complete.** The product's graph ref gets a
   ``V7_SUPPORTED_BY {reason:'consolidation'}`` edge to every source ref, so
   any consolidated node can be drilled back to all raw memories (episodic
   evidence transitively via the sources' own support edges).
4. **The synthesis is lossless.** The union-coverage gate (the same
   deterministic-specifics check that guards v1 merges) must pass on the
   synthesized text, with one retry that feeds the gaps back; otherwise the
   cluster is skipped.

Mechanism per cycle (cheap by design — no whole-store batching, no agent loop):
new-since-last-dream semantic rows become PROBES; each probe finds integration
candidates by pgvector cosine against the store plus a shared-anchor signal
from the graph; each accepted neighborhood costs ONE LLM synthesis call.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# pgvector cosine DISTANCE thresholds (0 = identical). A candidate joins a
# probe's neighborhood if it is very close, or moderately close AND shares at
# least one graph anchor with the probe (the dual-signal rule).
_DIST_HIGH = float(os.environ.get("MIRIX_V2_DIST_HIGH", "0.14"))   # ~cos 0.86
_DIST_LOW = float(os.environ.get("MIRIX_V2_DIST_LOW", "0.22"))     # ~cos 0.78
_TOPK = int(os.environ.get("MIRIX_V2_TOPK", "6"))
_PROBE_LIMIT = int(os.environ.get("MIRIX_V2_PROBE_LIMIT", "50"))
# Clusters within a cycle are independent (each is one synthesis call), so the
# LLM stage runs concurrently; the write-back stage stays serial because it
# mutates shared state (products absorbed by one cluster must not be written
# by another).
_CONCURRENCY = max(1, int(os.environ.get("MIRIX_V2_CONCURRENCY", "8")))

_SYNTH_SYSTEM = """You consolidate a user's semantic memory. Given several memory entries about the
same topic, write ONE comprehensive replacement entry.

Rules:
- Preserve EVERY specific from EVERY source VERBATIM: names, dates, numbers,
  quantities, prices, durations, preferences, reasons, symbolic meanings.
- Collapse only redundant wording. If two sources carry distinct facts, keep both.
- If sources CONFLICT (same attribute, different values), keep the newest value
  as current and record the older as history, e.g. "... (previously three bikes,
  as of 2023-05-06)".
- Be COMPACT. State each fact once, in the fewest words that keep it exact; drop
  narrative connectives, repeated framing and any sentence that adds no fact. Aim
  for details under 1500 characters. Length is not a virtue — coverage is.
- Output JSON only: {"name": "<concise topic name>", "summary": "<1-2 sentence
  summary>", "details": "<the full consolidated content>"}"""


@dataclass
class V2Stats:
    probes: int = 0
    clusters: int = 0
    created: int = 0
    updated: int = 0
    gate_retries: int = 0
    gate_skips: int = 0
    skipped_no_candidates: int = 0
    skipped_already_covered: int = 0
    provenance_edges: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


async def _fetch_probes(user_id: str, since: Optional[dt.datetime]) -> list:
    """New raw semantic rows (never our own products) to integrate."""
    from sqlalchemy import text as sa_text
    from mirix.server.server import db_context

    where_since = "AND created_at > :since" if since is not None else ""
    async with db_context() as session:
        rows = (await session.execute(sa_text(
            f"""SELECT id, name, summary, details, created_at
                FROM semantic_memory
                WHERE user_id = :u AND NOT is_deleted
                  AND coalesce(source,'') NOT LIKE 'auto_dream%'
                  {where_since}
                ORDER BY created_at ASC
                LIMIT :lim"""),
            {"u": user_id, "since": since, "lim": _PROBE_LIMIT} if since is not None
            else {"u": user_id, "lim": _PROBE_LIMIT})).fetchall()
    return rows


async def _fetch_candidates(user_id: str, probe_id: str) -> list:
    """Nearest semantic rows by stored-embedding cosine distance (self excluded)."""
    from sqlalchemy import text as sa_text
    from mirix.server.server import db_context

    async with db_context() as session:
        rows = (await session.execute(sa_text(
            """SELECT s.id, s.name, s.summary, s.details, coalesce(s.source,'') AS source,
                      s.summary_embedding <=> p.summary_embedding AS dist
               FROM semantic_memory s,
                    (SELECT summary_embedding FROM semantic_memory WHERE id = :pid) p
               WHERE s.user_id = :u AND NOT s.is_deleted AND s.id <> :pid
                 AND s.summary_embedding IS NOT NULL
               ORDER BY dist ASC
               LIMIT :k"""),
            {"u": user_id, "pid": probe_id, "k": _TOPK})).fetchall()
    return rows


async def _anchor_names(driver, memory_id: str) -> set:
    if driver is None:
        return set()
    try:
        async with driver.session() as s:
            res = await s.run(
                "MATCH (a:V7Anchor)-[:V7_DESCRIBED_BY]->(m:V7ConceptRef {memory_id: $mid}) "
                "RETURN a.name_lower AS n", mid=memory_id)
            return {rec["n"] async for rec in res}
    except Exception:  # noqa: BLE001 — anchor signal is an enhancer, not a dependency
        return set()


async def _carry_provenance_sources(driver, memory_id: str) -> list:
    """When re-consolidating one of our own products, inherit its source list."""
    if driver is None:
        return []
    try:
        async with driver.session() as s:
            res = await s.run(
                "MATCH (:V7ConceptRef {memory_id: $mid})-[r:V7_SUPPORTED_BY]->(m) "
                "WHERE r.reason = 'consolidation' RETURN m.memory_id AS mid", mid=memory_id)
            return [rec["mid"] async for rec in res]
    except Exception:  # noqa: BLE001
        return []



async def _drop_product_ref(driver, memory_id: str) -> None:
    """When a product row is absorbed (deleted), remove its graph ref NOW instead
    of leaving a dangling ConceptRef until the next maintenance sweep — its anchor
    links and fact citations would otherwise point at a deleted row for a whole
    cycle. Provenance has already been carried onto the successor before this."""
    if driver is None:
        return
    try:
        async with driver.session() as s:
            await s.run("MATCH (m:V7ConceptRef {memory_id: $mid}) DETACH DELETE m", mid=memory_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 consolidation: stale product ref cleanup failed (%s)", exc)

async def _write_provenance(driver, product_memory_id: str, source_memory_ids: list) -> int:
    if driver is None or not source_memory_ids:
        return 0
    n = 0
    try:
        async with driver.session() as s:
            for src in source_memory_ids:
                res = await s.run(
                    """MATCH (p:V7ConceptRef {memory_id: $pid})
                       MATCH (m:V7MemoryRef {memory_id: $src})
                       MERGE (p)-[r:V7_SUPPORTED_BY]->(m)
                       SET r.reason = 'consolidation'
                       RETURN count(r) AS c""",
                    pid=product_memory_id, src=src)
                rec = await res.single()
                n += int(rec["c"]) if rec else 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 consolidation: provenance write failed (%s)", exc)
    return n


def _row_text(name: str, summary: str, details: str) -> str:
    return f"{name or ''}\n{summary or ''}\n{details or ''}"


async def _synthesize(sources: list, model: str, gaps_feedback: Optional[list] = None) -> Optional[dict]:
    from mirix.services.lightrag_extractor import call_openai_chat

    payload = "SOURCE ENTRIES (each = one memory; newest last):\n" + "\n---\n".join(
        f"[{i+1}] {t}" for i, t in enumerate(sources))
    if gaps_feedback:
        payload += ("\n\nYOUR PREVIOUS ATTEMPT DROPPED THESE SPECIFICS — the consolidated "
                    "entry MUST contain each of them verbatim: " + ", ".join(gaps_feedback[:15]))
    try:
        raw = await call_openai_chat(_SYNTH_SYSTEM, payload, model, temperature=0.0)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        if data.get("name") and data.get("summary"):
            return {"name": str(data["name"]), "summary": str(data["summary"]),
                    "details": str(data.get("details") or data["summary"])}
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 consolidation: synthesis call failed (%s)", exc)
    return None



async def _synth_and_gate(plan: dict, model: str):
    """One cluster's LLM work: synthesize, gate-check, retry once with the gaps.

    Pure with respect to shared state — safe to run concurrently. Returns
    (synthesized|None, gaps, retried).
    """
    from mirix.services.merge_coverage import coverage_gaps

    def _check(syn):
        return coverage_gaps(plan["bodies"],
                             _row_text(syn["name"], syn["summary"], syn["details"]),
                             label_texts=plan["labels"])

    syn = await _synthesize(plan["texts"], model)
    if not syn:
        return None, ["synthesis failed"], False
    gaps = _check(syn)
    if not gaps:
        return syn, [], False
    syn2 = await _synthesize(plan["texts"], model, gaps_feedback=gaps)
    if not syn2:
        return None, gaps, True
    return syn2, _check(syn2), True


def _probe_already_covered(pid: str, absorbed: list, covered_by_product: dict) -> bool:
    """True when every product in the cluster already lists this probe as a source.

    Then the cluster carries no new information: re-synthesizing it would rewrite
    the same content (at the inflated cost of product-sized payloads) and produce
    an equivalent row.
    """
    owners = covered_by_product.get(pid)
    if not owners:
        return False
    return all(c[0] in owners for c in absorbed)



async def _covered_map(driver, user_id: str) -> dict:
    """Existing consolidation provenance as {source_memory_id: {product_id, ...}}."""
    out: dict = {}
    if driver is None:
        return out
    try:
        async with driver.session() as s:
            res = await s.run(
                "MATCH (p:V7ConceptRef {user_id: $u})-[r:V7_SUPPORTED_BY]->(m) "
                "WHERE r.reason = 'consolidation' "
                "RETURN m.memory_id AS src, p.memory_id AS prod", u=user_id)
            async for rec in res:
                out.setdefault(rec["src"], set()).add(rec["prod"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("v2 consolidation: coverage map unavailable (%s)", exc)
    return out


async def consolidate_incremental(
    *,
    user,
    actor,
    dream_agent_state,
    since: Optional[dt.datetime],
) -> dict:
    """One incremental consolidation cycle. Returns stats (see V2Stats)."""
    from mirix.database.neo4j_client import get_neo4j_driver
    from mirix.services._graph_common import llm_model_from_agent
    from mirix.services.merge_coverage import coverage_gaps
    from mirix.services.semantic_memory_manager import SemanticMemoryManager

    stats = V2Stats()
    driver = get_neo4j_driver()
    model = llm_model_from_agent(dream_agent_state) or "gpt-4.1-mini"
    mgr = SemanticMemoryManager()

    # memory_id -> {product ids that already cite it}. Seeded from the graph so
    # the "nothing new to fold in" check works across cycles, then kept current
    # as this cycle writes.
    covered_by_product = await _covered_map(driver, user.id)

    probes = await _fetch_probes(user.id, since)
    stats.probes = len(probes)
    # When the probe cap is hit, later rows in the window never got their probe
    # turn — checkpointing "now" would skip them forever. Hand the caller the
    # last fetched row's created_at instead so the next cycle resumes there.
    checkpoint_hint = probes[-1][4] if len(probes) >= _PROBE_LIMIT else None

    # ---- phase 1: plan clusters (DB/graph reads only) ----
    plans = []
    for probe in probes:
        pid, pname, psummary, pdetails, _created = probe
        # NB: a probe that was already consumed as another probe's candidate is
        # NOT skipped — top-k similarity is asymmetric, so its own neighborhood
        # can contain partners the earlier cluster missed. Duplication is safe:
        # its neighborhood now contains the freshly minted product, which the
        # absorb path grows instead of minting a sibling.
        cands = await _fetch_candidates(user.id, pid)
        if not cands:
            stats.skipped_no_candidates += 1
            continue

        probe_anchors = await _anchor_names(driver, pid)
        cluster = []
        for cid, cname, csummary, cdetails, csource, dist in cands:
            if dist is None:
                continue
            if dist <= _DIST_HIGH:
                cluster.append((cid, cname, csummary, cdetails, csource))
            elif dist <= _DIST_LOW and probe_anchors:
                shared = probe_anchors & await _anchor_names(driver, cid)
                if shared:
                    cluster.append((cid, cname, csummary, cdetails, csource))
        if not cluster:
            stats.skipped_no_candidates += 1
            continue

        # Existing consolidation products in the neighborhood -> grow instead of
        # minting a sibling. ALL products in the cluster get absorbed (deleting
        # only the first would leave a live overlapping duplicate whose content
        # was just copied into the successor). Only OUR OWN products are ever
        # deleted, never raw rows.
        absorbed = [c for c in cluster if c[4].startswith("auto_dream")]

        # Nothing new to fold in: the cluster is this probe plus products that
        # already consolidated it. Re-synthesizing would rewrite the same content
        # at growing cost (products are ~4x the length of a raw row, so these
        # clusters are the expensive ones) for no information gain.
        raw_partners = [c for c in cluster if not c[4].startswith("auto_dream")]
        if absorbed and not raw_partners and _probe_already_covered(pid, absorbed, covered_by_product):
            stats.skipped_already_covered += 1
            continue

        stats.clusters += 1
        source_rows = [(pid, pname, psummary, pdetails)] + \
                      [(c[0], c[1], c[2], c[3]) for c in cluster]
        # Bodies and titles are gate-checked differently (see coverage_gaps):
        # bodies phrase-level, titles component-level.
        source_texts = [_row_text(n, s, d) for (_i, n, s, d) in source_rows]
        source_bodies = [f"{s or ''}\n{d or ''}" for (_i, _n, s, d) in source_rows]
        source_labels = [n or "" for (_i, n, _s, _d) in source_rows]
        source_ids = [i for (i, _n, _s, _d) in source_rows]

        plans.append({
            "pid": pid, "absorbed": absorbed, "source_ids": source_ids,
            "texts": source_texts, "bodies": source_bodies, "labels": source_labels,
        })

    # ---- phase 2: synthesize + gate-check, CONCURRENTLY ----
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _run(plan):
        async with sem:
            return await _synth_and_gate(plan, model)

    results = await asyncio.gather(*(_run(p) for p in plans), return_exceptions=True)

    # ---- phase 3: write back, SERIALLY (mutates shared state) ----
    for plan, res in zip(plans, results):
        if isinstance(res, Exception):
            stats.errors += 1
            logger.warning("v2 consolidation: synthesis task failed (%s)", res)
            continue
        synthesized, gaps, retried = res
        if retried:
            stats.gate_retries += 1
        if not synthesized or gaps:
            stats.gate_skips += 1
            logger.info("v2 consolidation: cluster skipped (gate gaps=%s)",
                        (gaps or ["synthesis failed"])[:6])
            continue
        pid = plan["pid"]; absorbed = plan["absorbed"]; source_ids = plan["source_ids"]

        # Provenance set: all raw sources, plus whatever the absorbed products
        # already pointed at (so history is never shortened); absorbed products
        # themselves are excluded — they are about to be deleted.
        absorbed_ids = {c[0] for c in absorbed}
        provenance = [i for i in source_ids if i not in absorbed_ids]
        for c in absorbed:
            provenance += await _carry_provenance_sources(driver, c[0])
        provenance = [p for p in provenance if p not in absorbed_ids]

        try:
            inserted = await mgr.insert_semantic_item(
                actor=actor,
                agent_state=dream_agent_state,
                agent_id=dream_agent_state.id,
                name=synthesized["name"],
                summary=synthesized["summary"],
                details=synthesized["details"],
                source="auto_dream",
                organization_id=actor.organization_id,
                user_id=user.id,
            )
            if absorbed:
                for c in absorbed:
                    # Another cluster in this cycle may already have absorbed it;
                    # deleting a gone row is a no-op we must not treat as fatal.
                    try:
                        await mgr.delete_semantic_item_by_id(c[0], actor=actor)
                    except Exception:  # noqa: BLE001
                        pass
                    await _drop_product_ref(driver, c[0])
                    covered_by_product.setdefault(c[0], set())
                stats.updated += 1
            else:
                stats.created += 1
            stats.provenance_edges += await _write_provenance(
                driver, inserted.id, sorted(set(provenance)))
            for src in provenance:
                covered_by_product.setdefault(src, set()).add(inserted.id)
        except Exception as exc:  # noqa: BLE001
            stats.errors += 1
            logger.warning("v2 consolidation: insert failed for probe %s (%s)", pid, exc)

    logger.info("v2 consolidation user=%s: %s", user.id, stats.as_dict())
    out = stats.as_dict()
    out["checkpoint_hint"] = checkpoint_hint
    return out
