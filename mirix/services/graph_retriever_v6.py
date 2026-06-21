"""
v6 graph retriever — lean entity-index path.

Pipeline (no dual-level, no relation vector search, no item nodes):
  1. embed query → vector search on V6Entity.name_embedding → top-K entities
  2. 1-hop expand via V6_COOCCUR (top-N neighbors per seed, ordered by edge count)
  3. union back-refs across all (seed ∪ neighbor) entities → episodic_ids, semantic_ids
  4. PG fetch full rows from episodic_memory and semantic_memory
  5. format markdown context

Returns "" on disabled / no driver / no hits so the dispatcher can degrade.

Why this shape:
- The whole point of v6 is "graph as inverted index". We never traverse
  weighted edges, never compute community, never read description fields off
  the graph — those live in PG. Neo4j only owns the entity → memory_id map.
  Current graphs may store that map either as V6Entity property arrays or as
  V6MemoryRef nodes reached by APPEARS_IN / DESCRIBED_BY edges.
- 1-hop expansion uses raw edge count as a proxy for recall. Hot entities
  (mention_count >> 1) would otherwise dominate; the per-seed N cap keeps
  the expansion proportional.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import embed_batch
from mirix.settings import settings

logger = get_logger(__name__)


# Per-seed neighbor cap during 1-hop expansion. With top_k=15 seeds and
# neighbor_top_n=5 we look at ≤90 entity ids before dedup; in practice the
# back-ref union typically resolves to 30-60 unique memory rows.
DEFAULT_NEIGHBOR_TOP_N = 5

# Hard cap on PG rows fetched per memory type. Prevents pathological queries
# from blowing the LLM context if a hot entity is mentioned in 200+ memories.
DEFAULT_MAX_ITEMS_PER_KIND = 40


@dataclass
class V6EntityHit:
    id: str
    name: str
    entity_type: str
    cosine: float
    source: str  # "seed" or "neighbor"


@dataclass
class V6MemoryRow:
    id: str
    kind: str  # "episodic" or "semantic"
    summary: str
    details: str
    timestamp: Optional[str] = None  # occurred_at for episodic, created_at for semantic
    extra: dict = field(default_factory=dict)


class V6Retriever:
    """Stateless. Construct one per request."""

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        top_k: int = 15,
        neighbor_top_n: int = DEFAULT_NEIGHBOR_TOP_N,
        max_items_per_kind: int = DEFAULT_MAX_ITEMS_PER_KIND,
    ) -> str:
        """Run the full v6 retrieval. Returns formatted markdown context."""
        if not settings.enable_graph_memory or settings.graph_version != "v6":
            return ""

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return ""

        # ─── Step 1: embed query ─────────────────────────────────────────
        if not query or not query.strip():
            return ""
        embs = await embed_batch([query], agent_state)
        q_emb = embs[0] if embs else None
        if q_emb is None:
            logger.info("v6 retrieve: query embedding failed → empty context")
            return ""

        # ─── Step 2: vector search for top-K seed entities ───────────────
        seeds = await self._search_entities(driver, user_id, q_emb, top_k)
        if not seeds:
            return ""

        # ─── Step 3: 1-hop expand via V6_COOCCUR ─────────────────────────
        seed_ids = [s.id for s in seeds]
        neighbors = await self._expand_neighbors(
            driver, user_id=user_id, seed_ids=seed_ids,
            per_seed_n=neighbor_top_n,
        )
        # Combine seeds + neighbors, dedup by id, keep best cosine per id
        by_id: dict[str, V6EntityHit] = {}
        for hit in seeds + neighbors:
            existing = by_id.get(hit.id)
            if existing is None or hit.cosine > existing.cosine:
                by_id[hit.id] = hit
        all_hits = list(by_id.values())

        # ─── Step 4: union back-refs across all entities ─────────────────
        episodic_ids, semantic_ids = await self._collect_backrefs(
            driver, user_id=user_id, entity_ids=[h.id for h in all_hits],
        )
        if not episodic_ids and not semantic_ids:
            return self._format_context(all_hits, [], [])

        # ─── Step 5: PG fetch in parallel ────────────────────────────────
        ep_task = asyncio.create_task(
            self._fetch_episodic(user_id, episodic_ids[:max_items_per_kind])
        )
        sem_task = asyncio.create_task(
            self._fetch_semantic(user_id, semantic_ids[:max_items_per_kind])
        )
        ep_rows, sem_rows = await asyncio.gather(ep_task, sem_task, return_exceptions=True)
        if isinstance(ep_rows, Exception):
            logger.warning("v6 episodic PG fetch failed: %s", ep_rows)
            ep_rows = []
        if isinstance(sem_rows, Exception):
            logger.warning("v6 semantic PG fetch failed: %s", sem_rows)
            sem_rows = []

        ctx = self._format_context(all_hits, ep_rows, sem_rows)
        logger.info(
            "v6 retrieve: %d seeds, %d neighbors, %d ep, %d sem → %d chars",
            len(seeds), len(neighbors), len(ep_rows), len(sem_rows), len(ctx),
        )
        return ctx

    # ─────────────────────────────────────────────── Neo4j: vector search

    async def _search_entities(
        self, driver, user_id: str, emb: list[float], top_k: int,
    ) -> list[V6EntityHit]:
        cypher = """
        CALL db.index.vector.queryNodes('v6_entity_name_emb', $top_k, $emb)
        YIELD node AS e, score AS sim
        WHERE e.user_id = $user_id
        RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type, sim AS sim
        ORDER BY sim DESC
        """
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, top_k=top_k, emb=emb, user_id=user_id)
            return [
                V6EntityHit(
                    id=rec["id"], name=rec["name"] or "",
                    entity_type=rec["entity_type"] or "Other",
                    cosine=float(rec["sim"] or 0.0), source="seed",
                )
                async for rec in result
            ]

    # ─────────────────────────────────────────────── Neo4j: 1-hop expand

    async def _expand_neighbors(
        self, driver, *, user_id: str, seed_ids: list[str], per_seed_n: int,
    ) -> list[V6EntityHit]:
        """For each seed, pull its top-N most-frequent co-occurring neighbors."""
        if not seed_ids:
            return []
        cypher = """
        UNWIND $seed_ids AS sid
        MATCH (seed:V6Entity {id: sid})
        MATCH (seed)-[r:V6_COOCCUR]-(nbr:V6Entity {user_id: $user_id})
        WHERE nbr.id <> sid
        WITH sid, nbr, r.count AS w
        ORDER BY w DESC
        WITH sid, collect({nbr: nbr, w: w})[..$per_seed_n] AS top_nbrs
        UNWIND top_nbrs AS row
        WITH row.nbr AS nbr, row.w AS w
        RETURN DISTINCT nbr.id AS id, nbr.name AS name,
               nbr.entity_type AS entity_type, w AS edge_count
        """
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                cypher, seed_ids=seed_ids, user_id=user_id, per_seed_n=per_seed_n,
            )
            # Neighbors don't have a real cosine; use a small synthetic score so
            # they rank below seeds in dedup.
            return [
                V6EntityHit(
                    id=rec["id"], name=rec["name"] or "",
                    entity_type=rec["entity_type"] or "Other",
                    cosine=0.3, source="neighbor",
                )
                async for rec in result
            ]

    # ─────────────────────────────────────────────── Neo4j: union back-refs

    async def _collect_backrefs(
        self, driver, *, user_id: str, entity_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        if not entity_ids:
            return [], []
        cypher = """
        UNWIND $eids AS eid
        MATCH (e:V6Entity {id: eid, user_id: $user_id})
        OPTIONAL MATCH (e)-[:APPEARS_IN]-(ep_ref:V6EpisodeRef)
        OPTIONAL MATCH (e)-[:DESCRIBED_BY]-(sem_ref:V6ConceptRef)
        RETURN
            coalesce(e.episodic_ids, []) AS eps,
            coalesce(e.semantic_ids, []) AS sems,
            collect(DISTINCT ep_ref.id) AS ep_refs,
            collect(DISTINCT sem_ref.id) AS sem_refs
        """
        ep_set: set[str] = set()
        sem_set: set[str] = set()

        def _strip_kind(raw: object, kind: str) -> Optional[str]:
            if raw is None:
                return None
            value = str(raw)
            prefix = f"{kind}:"
            return value[len(prefix):] if value.startswith(prefix) else value

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, eids=entity_ids, user_id=user_id)
            async for rec in result:
                for raw in rec["eps"] or []:
                    mid = _strip_kind(raw, "episodic")
                    if mid:
                        ep_set.add(mid)
                for raw in rec["ep_refs"] or []:
                    mid = _strip_kind(raw, "episodic")
                    if mid:
                        ep_set.add(mid)
                for raw in rec["sems"] or []:
                    mid = _strip_kind(raw, "semantic")
                    if mid:
                        sem_set.add(mid)
                for raw in rec["sem_refs"] or []:
                    mid = _strip_kind(raw, "semantic")
                    if mid:
                        sem_set.add(mid)
        return sorted(ep_set), sorted(sem_set)

    # ─────────────────────────────────────────────── PG fetch

    async def _fetch_episodic(self, user_id: str, ids: list[str]) -> list[V6MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        async with db_context() as session:
            result = await session.execute(
                sa_text(
                    "SELECT id, summary, details, occurred_at "
                    "FROM episodic_memory "
                    "WHERE user_id = :u AND id = ANY(:ids) "
                    "ORDER BY occurred_at DESC NULLS LAST"
                ),
                {"u": user_id, "ids": ids},
            )
            return [
                V6MemoryRow(
                    id=row[0], kind="episodic",
                    summary=row[1] or "", details=row[2] or "",
                    timestamp=row[3].isoformat() if row[3] is not None else None,
                )
                for row in result.fetchall()
            ]

    async def _fetch_semantic(self, user_id: str, ids: list[str]) -> list[V6MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        async with db_context() as session:
            result = await session.execute(
                sa_text(
                    "SELECT id, name, summary, details, source, created_at "
                    "FROM semantic_memory "
                    "WHERE user_id = :u AND id = ANY(:ids) "
                    "ORDER BY created_at DESC NULLS LAST"
                ),
                {"u": user_id, "ids": ids},
            )
            return [
                V6MemoryRow(
                    id=row[0], kind="semantic",
                    summary=row[2] or "", details=row[3] or "",
                    timestamp=row[5].isoformat() if row[5] is not None else None,
                    extra={"name": row[1] or "", "source": row[4] or ""},
                )
                for row in result.fetchall()
            ]

    # ─────────────────────────────────────────────── format

    def _format_context(
        self, entities: list[V6EntityHit],
        ep_rows: list[V6MemoryRow], sem_rows: list[V6MemoryRow],
    ) -> str:
        if not (entities or ep_rows or sem_rows):
            return ""
        lines: list[str] = ["## Memory Index (v6)"]
        if entities:
            seed_names = [f"{e.name} ({e.entity_type})" for e in entities if e.source == "seed"]
            nbr_names = [f"{e.name}" for e in entities if e.source == "neighbor"]
            if seed_names:
                lines.append(f"**Matched entities:** {', '.join(seed_names[:15])}")
            if nbr_names:
                lines.append(f"**Related entities:** {', '.join(nbr_names[:15])}")

        if ep_rows:
            lines.append("\n### Episodic memories")
            for r in ep_rows:
                ts = self._fmt_ts(r.timestamp)
                head = f"- [{ts}] {r.summary}" if ts else f"- {r.summary}"
                lines.append(head.rstrip())
                if r.details and r.details != r.summary:
                    lines.append(f"  {r.details[:400]}")

        if sem_rows:
            lines.append("\n### Semantic memories")
            for r in sem_rows:
                name = r.extra.get("name", "")
                head = f"- {name}: {r.summary}" if name else f"- {r.summary}"
                lines.append(head)
                if r.details and r.details != r.summary:
                    lines.append(f"  {r.details[:400]}")

        return "\n".join(lines)

    @staticmethod
    def _fmt_ts(ts: Optional[str]) -> str:
        if not ts:
            return ""
        # already ISO; trim to YYYY-MM-DD for readability
        return ts[:10]
