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
import os
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
    async def retrieve_rows(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        top_k: int = 18,
        max_items_per_kind: int = DEFAULT_MAX_ITEMS_PER_KIND,
    ) -> tuple[list, list, list]:
        """Graph retrieval as STRUCTURED rows: (anchors, episodic_rows, semantic_rows).

        `retrieve()` renders these into a prompt blob. That blob turned out to be easy
        for an answerer to ignore — measured: the graph context was injected on 60/60
        questions and moved nothing, while the same facts merged into the answerer's own
        search RESULTS did move QA. Callers that want the graph to actually influence an
        answer should use these rows and merge them into the tool output.
        """
        if not settings.enable_graph_memory or not (
            settings.graph_version.startswith("v7") or settings.graph_version == "v8"
        ):
            return [], [], []

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None or not query or not query.strip():
            return [], [], []

        embs = await embed_batch([query], agent_state)
        q_emb = embs[0] if embs else None
        if q_emb is None:
            return [], [], []

        anchors = await self._search_anchors(driver, user_id, q_emb, top_k)
        if not anchors:
            return anchors, [], []

        # Single retrieval path for the whole v7 family. The experimental variants
        # that used to branch here — v7.7 Personalized PageRank (retrieval richer, QA
        # flat) and v7.2 per-anchor coverage round-robin (neutral) — are archived; see
        # docs/graph_memory_v7/development_history.md.
        episodic_ids, semantic_ids = await self._collect_memory_refs(
            driver, user_id=user_id, anchor_ids=[a.id for a in anchors],
        )
        if not episodic_ids and not semantic_ids:
            return anchors, [], []
        # v7.1+: rerank the anchor-collected candidates by query full-text
        # similarity (anchor match = recall, text-cosine = precision) so a
        # salient-but-wrong same-name entity from another document sinks below
        # the true answer. Only plain v7 keeps the original anchor-traversal/date
        # order (the unranked baseline). This used to be `== "v7.1"`, which
        # silently switched the rerank OFF for every later version (v7.3+, v7.10,
        # v8): they fell through to a plain traversal-order truncation — the same
        # stale-exact-match guard bug as the ingest routing one fixed earlier.
        # MIRIX_GRAPH_RERANK=0 restores the unranked traversal-order truncation —
        # an A/B toggle so "graph without reranker" can be measured explicitly.
        rerank = None if (
            settings.graph_version == "v7"
            or os.environ.get("MIRIX_GRAPH_RERANK") == "0"
        ) else q_emb
        ep_arg = episodic_ids if rerank else episodic_ids[:max_items_per_kind]
        sem_arg = semantic_ids if rerank else semantic_ids[:max_items_per_kind]
        ep_task = asyncio.create_task(
            self._fetch_episodic(user_id, ep_arg, q_emb=rerank, limit=max_items_per_kind))
        sem_task = asyncio.create_task(
            self._fetch_semantic(user_id, sem_arg, q_emb=rerank, limit=max_items_per_kind))
        ep_rows, sem_rows = await asyncio.gather(ep_task, sem_task, return_exceptions=True)
        if isinstance(ep_rows, Exception):
            logger.warning("v7 episodic PG fetch failed: %s", ep_rows)
            ep_rows = []
        if isinstance(sem_rows, Exception):
            logger.warning("v7 semantic PG fetch failed: %s", sem_rows)
            sem_rows = []

        logger.info("v7 retrieve_rows: %d anchors, %d ep, %d sem",
                    len(anchors), len(ep_rows), len(sem_rows))
        return anchors, ep_rows, sem_rows

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        top_k: int = 18,
        max_items_per_kind: int = DEFAULT_MAX_ITEMS_PER_KIND,
    ) -> str:
        """Graph retrieval rendered as a prompt context blob (the original API)."""
        anchors, ep_rows, sem_rows = await self.retrieve_rows(
            query=query, user_id=user_id, agent_state=agent_state,
            top_k=top_k, max_items_per_kind=max_items_per_kind)
        if not anchors:
            return ""
        ctx = self._format_context(anchors, ep_rows, sem_rows)

        # Hybrid wrap: append flat query-similarity memories alongside the graph
        # context. The graph anchors on entity NAMES (precise where flat hits a
        # same-name collision, but mis-fires where the query term matches a
        # salient-but-wrong entity in another document); flat anchors on full
        # memory TEXT (the complementary disambiguation). Giving the answerer both
        # recovers the union of their correct answers. Gated for clean A/B.
        if os.getenv("MIRIX_GRAPH_HYBRID_WRAP") == "1":
            embs = await embed_batch([query], agent_state)
            if embs and embs[0] is not None:
                flat = await self._flat_section(user_id, embs[0])
                if flat:
                    ctx = ctx + "\n\n" + flat

        logger.info("v7 retrieve: %d anchors -> %d chars", len(anchors), len(ctx))
        return ctx

    async def _flat_section(self, user_id: str, q_emb: list[float], limit: int = 8) -> str:
        """Top-k episodic + semantic by raw query-embedding similarity (flat
        pgvector) — the disambiguation signal the graph alone lacks."""
        from sqlalchemy import text as sa_text
        from mirix.constants import MAX_EMBEDDING_DIM
        from mirix.server.server import db_context

        qp = str(list(q_emb) + [0.0] * (MAX_EMBEDDING_DIM - len(q_emb)))
        try:
            async with db_context() as session:
                ep = (await session.execute(sa_text(
                    "SELECT summary, details FROM episodic_memory "
                    "WHERE user_id = :u AND summary_embedding IS NOT NULL "
                    "ORDER BY summary_embedding <=> CAST(:q AS vector) LIMIT :k"
                ), {"u": user_id, "q": qp, "k": limit})).fetchall()
                sem = (await session.execute(sa_text(
                    "SELECT name, summary, details FROM semantic_memory "
                    "WHERE user_id = :u AND summary_embedding IS NOT NULL "
                    "ORDER BY summary_embedding <=> CAST(:q AS vector) LIMIT :k"
                ), {"u": user_id, "q": qp, "k": limit})).fetchall()
        except Exception as e:
            logger.warning("v7 hybrid flat section failed: %s", e)
            return ""

        lines: list[str] = ["## Most similar memories (flat text match)"]
        for name, summary, details in sem:
            head = f"- {name}: {summary}" if name else f"- {summary}"
            lines.append(head.rstrip())
            if details and details != summary:
                lines.append(f"  {details[:400]}")
        for summary, details in ep:
            lines.append(f"- {summary}".rstrip())
            if details and details != summary:
                lines.append(f"  {details[:400]}")
        return "\n".join(lines)

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

    # (v7.7 PPR and v7.2 coverage retrieval helpers — _load_ppr_graph, _retrieve_ppr,
    #  _collect_per_anchor, _retrieve_coverage, _round_robin — are archived; see
    #  docs/graph_memory_v7/development_history.md.)

    async def _fetch_episodic(
        self, user_id: str, ids: list[str],
        q_emb: Optional[list[float]] = None, limit: Optional[int] = None,
    ) -> list[V7MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        if q_emb is not None:  # v7.1: rank candidates by query full-text similarity
            from mirix.constants import MAX_EMBEDDING_DIM
            qp = str(list(q_emb) + [0.0] * (MAX_EMBEDDING_DIM - len(q_emb)))
            sql = (
                "SELECT id, summary, details, occurred_at, actor FROM episodic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) AND summary_embedding IS NOT NULL "
                "ORDER BY summary_embedding <=> CAST(:q AS vector) LIMIT :k"
            )
            params = {"u": user_id, "ids": ids, "q": qp, "k": limit or DEFAULT_MAX_ITEMS_PER_KIND}
        else:
            sql = (
                "SELECT id, summary, details, occurred_at, actor FROM episodic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) ORDER BY occurred_at DESC NULLS LAST"
            )
            params = {"u": user_id, "ids": ids}

        async with db_context() as session:
            result = await session.execute(sa_text(sql), params)
            return [
                V7MemoryRow(
                    id=row[0],
                    kind="episodic",
                    summary=row[1] or "",
                    details=row[2] or "",
                    timestamp=row[3].isoformat() if row[3] is not None else None,
                    extra={"actor": row[4] or ""},
                )
                for row in result.fetchall()
            ]

    async def _fetch_semantic(
        self, user_id: str, ids: list[str],
        q_emb: Optional[list[float]] = None, limit: Optional[int] = None,
    ) -> list[V7MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        if q_emb is not None:  # v7.1: rank candidates by query full-text similarity
            from mirix.constants import MAX_EMBEDDING_DIM
            qp = str(list(q_emb) + [0.0] * (MAX_EMBEDDING_DIM - len(q_emb)))
            sql = (
                "SELECT id, name, summary, details, source, created_at FROM semantic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) AND summary_embedding IS NOT NULL "
                "ORDER BY summary_embedding <=> CAST(:q AS vector) LIMIT :k"
            )
            params = {"u": user_id, "ids": ids, "q": qp, "k": limit or DEFAULT_MAX_ITEMS_PER_KIND}
        else:
            sql = (
                "SELECT id, name, summary, details, source, created_at FROM semantic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) ORDER BY created_at DESC NULLS LAST"
            )
            params = {"u": user_id, "ids": ids}

        async with db_context() as session:
            result = await session.execute(sa_text(sql), params)
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
            # (The v7.9 role-split rendering — user-domain vs assistant-domain
            # sections — was rejected at QA 34→30 and is archived.)
            lines.append("\n### Episodic memories (PG flat evidence)")
            for row in ep_rows:
                ts = row.timestamp[:10] if row.timestamp else ""
                head = f"- [{ts}] {row.summary}" if ts else f"- {row.summary}"
                lines.append(head.rstrip())
                if row.details and row.details != row.summary:
                    lines.append(f"  {row.details[:500]}")

        return "\n".join(lines)
