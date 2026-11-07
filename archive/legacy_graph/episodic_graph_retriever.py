"""
Episodic graph retriever (v4) — reads G_episodic in Neo4j.

Pipeline:
  1. ll embedding → ep_entity_name_emb vector → seed EpisodicEntities + 1-hop EP_RELATES
  2. hl embedding → ep_rel_kw_emb vector → seed EP_RELATES + both endpoints
  3. Round-robin merge entities, round-robin merge relations
  4. MENTIONS reverse: entities → Episodes that mention them
  5. NEXT one-hop expansion: each Episode → ±1 temporal neighbors
  6. Score + dedup items
  7. PG fetch full episode details (summary + details + occurred_at)
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


class EpisodicRetriever(GraphRetrieverBase):
    ENTITY_LABEL = "EpisodicEntity"
    ITEM_LABEL = "Episode"
    REL_TYPE = "EP_RELATES"
    ENTITY_VECTOR_INDEX = "ep_entity_name_emb"
    REL_VECTOR_INDEX = "ep_rel_kw_emb"
    SECTION_TITLE = "Episodic"

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
        """Full episodic pipeline: search → MENTIONS reverse → NEXT one-hop → PG fetch."""
        entities, relations = await self.search(
            driver=driver,
            user_id=user_id,
            ll_embedding=ll_embedding,
            hl_embedding=hl_embedding,
            top_k=top_k,
        )

        # Reverse MENTIONS: get Episodes that mention the surviving entities
        entity_ids = [e.id for e in entities]
        episodes_via_mentions = await self._fetch_episodes_via_mentions(
            driver, user_id=user_id, entity_ids=entity_ids, limit=item_top_k * 2,
        )

        # NEXT one-hop expansion
        episode_ids = [it.id for it in episodes_via_mentions]
        episodes_via_one_hop = await self._fetch_episodes_one_hop(
            driver, user_id=user_id, episode_ids=episode_ids, limit=item_top_k,
        )

        # Merge + dedup
        seen_ids: set[str] = set()
        merged_items: list[ItemHit] = []
        for it in episodes_via_mentions + episodes_via_one_hop:
            if it.id in seen_ids:
                continue
            seen_ids.add(it.id)
            merged_items.append(it)

        # Score & sort by recency-aware score
        for it in merged_items:
            it.score = final_score(it.cosine, it.timestamp)
        merged_items.sort(key=lambda x: x.score, reverse=True)
        merged_items = merged_items[:item_top_k]

        # PG fetch full details for the kept items (summary already in graph;
        # PG has details). Best-effort — degrade gracefully if PG miss.
        await self._enrich_with_pg(merged_items, user_id=user_id)

        return GraphSearchResult(entities=entities, relations=relations, items=merged_items)

    async def _fetch_episodes_via_mentions(
        self, driver, *, user_id: str, entity_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not entity_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (e:EpisodicEntity {id: eid})<-[:MENTIONS]-(ep:Episode {user_id: $user_id})
                WITH DISTINCT ep
                ORDER BY ep.occurred_at DESC
                LIMIT $limit
                RETURN ep.id AS id, ep.summary AS summary, ep.occurred_at AS occurred_at
                """,
                eids=entity_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Episode",
                    summary=rec["summary"] or "", detail="",
                    timestamp=rec["occurred_at"], cosine=0.5, source="mentions",
                )
                async for rec in result
            ]

    async def _fetch_episodes_one_hop(
        self, driver, *, user_id: str, episode_ids: list[str], limit: int
    ) -> list[ItemHit]:
        """For each Episode, fetch its NEXT predecessor/successor (±1 hop)."""
        if not episode_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (ep:Episode {id: eid})
                OPTIONAL MATCH (ep)-[:NEXT]->(next:Episode {user_id: $user_id})
                OPTIONAL MATCH (prev:Episode {user_id: $user_id})-[:NEXT]->(ep)
                WITH collect(DISTINCT next) + collect(DISTINCT prev) AS neighbors
                UNWIND neighbors AS n
                WITH n WHERE n IS NOT NULL
                RETURN DISTINCT n.id AS id, n.summary AS summary, n.occurred_at AS occurred_at
                LIMIT $limit
                """,
                eids=episode_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Episode",
                    summary=rec["summary"] or "", detail="",
                    timestamp=rec["occurred_at"], cosine=0.3, source="one_hop",
                )
                async for rec in result
            ]

    async def _enrich_with_pg(self, items: list[ItemHit], *, user_id: str) -> None:
        """Pull full episodic_memory.details for kept items. Graceful on miss."""
        if not items:
            return
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        ids = [it.id for it in items]
        try:
            async with db_context() as session:
                result = await session.execute(
                    sa_text(
                        "SELECT id, details FROM episodic_memory "
                        "WHERE user_id = :u AND id = ANY(:ids)"
                    ),
                    {"u": user_id, "ids": ids},
                )
                detail_map = {row[0]: (row[1] or "") for row in result.fetchall()}
        except Exception as e:
            logger.debug("PG enrich for episodic failed: %s", e)
            return

        for it in items:
            if it.id in detail_map:
                it.detail = detail_map[it.id]
