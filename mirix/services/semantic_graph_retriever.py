"""
Semantic graph retriever (v4) — reads G_semantic in Neo4j.

Pipeline:
  1. ll embedding → sem_entity_name_emb vector → seed SemanticEntities + 1-hop SEM_RELATES
  2. hl embedding → sem_rel_kw_emb vector → seed SEM_RELATES + endpoints
  3. Round-robin merge
  4. MENTIONS reverse: entities → Concepts that mention them
  5. CONCEPT_RELATES one-hop: each Concept → adjacent Concepts
  6. Score + dedup
  7. PG fetch full concept details

Unlike episodic, there is no timestamp ordering — concepts are ordered by
cosine score (recency_decay defaults to 0.5 when timestamp is missing).
"""

from __future__ import annotations

from typing import Optional

from mirix.log import get_logger
from mirix.services._graph_retriever_base import (
    DEFAULT_TOP_K,
    GraphRetrieverBase,
    GraphSearchResult,
    ItemHit,
    final_score,
)

logger = get_logger(__name__)


class SemanticRetriever(GraphRetrieverBase):
    ENTITY_LABEL = "SemanticEntity"
    ITEM_LABEL = "Concept"
    REL_TYPE = "SEM_RELATES"
    ENTITY_VECTOR_INDEX = "sem_entity_name_emb"
    REL_VECTOR_INDEX = "sem_rel_kw_emb"
    SECTION_TITLE = "Semantic"

    async def retrieve(
        self,
        *,
        driver,
        user_id: str,
        ll_embedding: Optional[list[float]],
        hl_embedding: Optional[list[float]],
        top_k: int = DEFAULT_TOP_K,
        item_top_k: int = 15,
    ) -> GraphSearchResult:
        entities, relations = await self.search(
            driver=driver,
            user_id=user_id,
            ll_embedding=ll_embedding,
            hl_embedding=hl_embedding,
            top_k=top_k,
        )

        entity_ids = [e.id for e in entities]
        concepts_via_mentions = await self._fetch_concepts_via_mentions(
            driver, user_id=user_id, entity_ids=entity_ids, limit=item_top_k * 2,
        )

        concept_ids = [it.id for it in concepts_via_mentions]
        concepts_via_one_hop = await self._fetch_concepts_one_hop(
            driver, user_id=user_id, concept_ids=concept_ids, limit=item_top_k,
        )

        seen: set[str] = set()
        merged: list[ItemHit] = []
        for it in concepts_via_mentions + concepts_via_one_hop:
            if it.id in seen:
                continue
            seen.add(it.id)
            merged.append(it)

        for it in merged:
            it.score = final_score(it.cosine, it.timestamp)
        merged.sort(key=lambda x: x.score, reverse=True)
        merged = merged[:item_top_k]

        await self._enrich_with_pg(merged, user_id=user_id)

        return GraphSearchResult(entities=entities, relations=relations, items=merged)

    async def _fetch_concepts_via_mentions(
        self, driver, *, user_id: str, entity_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not entity_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (e:SemanticEntity {id: eid})<-[:MENTIONS]-(c:Concept {user_id: $user_id})
                WITH DISTINCT c
                ORDER BY c.created_at DESC
                LIMIT $limit
                RETURN c.id AS id, c.name AS name, c.summary AS summary, c.created_at AS created_at
                """,
                eids=entity_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Concept",
                    summary=rec["name"] or "",          # concept "summary" line uses name
                    detail=rec["summary"] or "",        # detail line uses summary
                    timestamp=rec["created_at"], cosine=0.5, source="mentions",
                )
                async for rec in result
            ]

    async def _fetch_concepts_one_hop(
        self, driver, *, user_id: str, concept_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not concept_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $cids AS cid
                MATCH (c:Concept {id: cid})
                OPTIONAL MATCH (c)-[:CONCEPT_RELATES]-(n:Concept {user_id: $user_id})
                WITH n WHERE n IS NOT NULL
                RETURN DISTINCT n.id AS id, n.name AS name, n.summary AS summary, n.created_at AS created_at
                LIMIT $limit
                """,
                cids=concept_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Concept",
                    summary=rec["name"] or "",
                    detail=rec["summary"] or "",
                    timestamp=rec["created_at"], cosine=0.3, source="one_hop",
                )
                async for rec in result
            ]

    async def _enrich_with_pg(self, items: list[ItemHit], *, user_id: str) -> None:
        """Pull full semantic_memory.details. Best-effort; graph summary covers basics."""
        if not items:
            return
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        ids = [it.id for it in items]
        try:
            async with db_context() as session:
                result = await session.execute(
                    sa_text(
                        "SELECT id, details FROM semantic_memory "
                        "WHERE user_id = :u AND id = ANY(:ids)"
                    ),
                    {"u": user_id, "ids": ids},
                )
                detail_map = {row[0]: (row[1] or "") for row in result.fetchall()}
        except Exception as e:
            logger.debug("PG enrich for semantic failed: %s", e)
            return

        for it in items:
            if it.id in detail_map:
                # If PG details is more informative than graph summary, use it
                pg_detail = detail_map[it.id]
                if pg_detail and pg_detail != it.detail:
                    it.detail = pg_detail
