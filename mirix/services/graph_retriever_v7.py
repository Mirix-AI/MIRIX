"""
v7 graph retriever - minimal graph links, full details from flat memory.

Retrieval path:
  1. query embedding -> V7Anchor vector search
  2. anchors -> episodic refs / semantic refs
  3. semantic refs -> supporting episodic refs, when provenance edges exist
  4. fetch full rows from PostgreSQL episodic_memory / semantic_memory
  5. format a compact context for QA
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


DEFAULT_MAX_ITEMS_PER_KIND = 36


@dataclass
class V7AnchorHit:
    id: str
    name: str
    anchor_type: str
    cosine: float


@dataclass
class V7MemoryRow:
    id: str
    kind: str
    summary: str
    details: str
    timestamp: Optional[str] = None
    extra: dict = field(default_factory=dict)


class V7Retriever:
    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        top_k: int = 18,
        max_items_per_kind: int = DEFAULT_MAX_ITEMS_PER_KIND,
    ) -> str:
        if not settings.enable_graph_memory or settings.graph_version != "v7":
            return ""

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None or not query or not query.strip():
            return ""

        embs = await embed_batch([query], agent_state)
        q_emb = embs[0] if embs else None
        if q_emb is None:
            return ""

        anchors = await self._search_anchors(driver, user_id, q_emb, top_k)
        if not anchors:
            return ""

        episodic_ids, semantic_ids = await self._collect_memory_refs(
            driver,
            user_id=user_id,
            anchor_ids=[a.id for a in anchors],
        )
        if not episodic_ids and not semantic_ids:
            return self._format_context(anchors, [], [])

        ep_task = asyncio.create_task(self._fetch_episodic(user_id, episodic_ids[:max_items_per_kind]))
        sem_task = asyncio.create_task(self._fetch_semantic(user_id, semantic_ids[:max_items_per_kind]))
        ep_rows, sem_rows = await asyncio.gather(ep_task, sem_task, return_exceptions=True)
        if isinstance(ep_rows, Exception):
            logger.warning("v7 episodic PG fetch failed: %s", ep_rows)
            ep_rows = []
        if isinstance(sem_rows, Exception):
            logger.warning("v7 semantic PG fetch failed: %s", sem_rows)
            sem_rows = []

        ctx = self._format_context(anchors, ep_rows, sem_rows)
        logger.info(
            "v7 retrieve: %d anchors, %d ep, %d sem -> %d chars",
            len(anchors), len(ep_rows), len(sem_rows), len(ctx),
        )
        return ctx

    async def _search_anchors(
        self, driver, user_id: str, emb: list[float], top_k: int
    ) -> list[V7AnchorHit]:
        cypher = """
        CALL db.index.vector.queryNodes('v7_anchor_name_emb', $top_k, $emb)
        YIELD node AS a, score AS sim
        WHERE a.user_id = $user_id
        RETURN a.id AS id, a.name AS name, a.anchor_type AS anchor_type, sim AS sim
        ORDER BY sim DESC
        """
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, top_k=top_k, emb=emb, user_id=user_id)
            return [
                V7AnchorHit(
                    id=rec["id"],
                    name=rec["name"] or "",
                    anchor_type=rec["anchor_type"] or "Other",
                    cosine=float(rec["sim"] or 0.0),
                )
                async for rec in result
            ]

    async def _collect_memory_refs(
        self, driver, *, user_id: str, anchor_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        if not anchor_ids:
            return [], []
        cypher = """
        UNWIND $anchor_ids AS aid
        MATCH (a:V7Anchor {id: aid, user_id: $user_id})
        OPTIONAL MATCH (a)-[:V7_APPEARS_IN]->(ep:V7EpisodeRef)
        OPTIONAL MATCH (a)-[:V7_DESCRIBED_BY]->(sem:V7ConceptRef)
        OPTIONAL MATCH (sem)-[:V7_SUPPORTED_BY]->(support_ep:V7EpisodeRef)
        OPTIONAL MATCH (ep)<-[:V7_SUPPORTED_BY]-(support_sem:V7ConceptRef)
        OPTIONAL MATCH (ep)-[:V7_NEXT_MEMORY]-(near_ep:V7EpisodeRef)
        RETURN
            collect(DISTINCT ep.memory_id) AS direct_ep,
            collect(DISTINCT support_ep.memory_id) AS support_ep,
            collect(DISTINCT near_ep.memory_id) AS near_ep,
            collect(DISTINCT sem.memory_id) AS direct_sem,
            collect(DISTINCT support_sem.memory_id) AS support_sem
        """
        ep_ids: list[str] = []
        sem_ids: list[str] = []

        def add_unique(target: list[str], values: list[object]) -> None:
            seen = set(target)
            for raw in values or []:
                if raw is None:
                    continue
                value = str(raw)
                if value and value not in seen:
                    target.append(value)
                    seen.add(value)

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, anchor_ids=anchor_ids, user_id=user_id)
            async for rec in result:
                add_unique(ep_ids, rec["direct_ep"])
                add_unique(ep_ids, rec["support_ep"])
                add_unique(ep_ids, rec["near_ep"])
                add_unique(sem_ids, rec["direct_sem"])
                add_unique(sem_ids, rec["support_sem"])
        return ep_ids, sem_ids

    async def _fetch_episodic(self, user_id: str, ids: list[str]) -> list[V7MemoryRow]:
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
                V7MemoryRow(
                    id=row[0],
                    kind="episodic",
                    summary=row[1] or "",
                    details=row[2] or "",
                    timestamp=row[3].isoformat() if row[3] is not None else None,
                )
                for row in result.fetchall()
            ]

    async def _fetch_semantic(self, user_id: str, ids: list[str]) -> list[V7MemoryRow]:
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
                V7MemoryRow(
                    id=row[0],
                    kind="semantic",
                    summary=row[2] or "",
                    details=row[3] or "",
                    timestamp=row[5].isoformat() if row[5] is not None else None,
                    extra={"name": row[1] or "", "source": row[4] or ""},
                )
                for row in result.fetchall()
            ]

    def _format_context(
        self,
        anchors: list[V7AnchorHit],
        ep_rows: list[V7MemoryRow],
        sem_rows: list[V7MemoryRow],
    ) -> str:
        lines: list[str] = ["## Memory Linkage Graph (v7)"]
        if anchors:
            names = [f"{a.name} ({a.anchor_type})" for a in anchors[:18]]
            lines.append(f"**Matched anchors:** {', '.join(names)}")

        if sem_rows:
            lines.append("\n### Semantic memories (PG flat)")
            for row in sem_rows:
                name = row.extra.get("name", "")
                head = f"- {name}: {row.summary}" if name else f"- {row.summary}"
                lines.append(head.rstrip())
                if row.details and row.details != row.summary:
                    lines.append(f"  {row.details[:500]}")

        if ep_rows:
            lines.append("\n### Episodic memories (PG flat evidence)")
            for row in ep_rows:
                ts = row.timestamp[:10] if row.timestamp else ""
                head = f"- [{ts}] {row.summary}" if ts else f"- {row.summary}"
                lines.append(head.rstrip())
                if row.details and row.details != row.summary:
                    lines.append(f"  {row.details[:500]}")

        return "\n".join(lines)
