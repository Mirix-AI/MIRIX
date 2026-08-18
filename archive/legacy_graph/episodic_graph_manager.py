"""
Episodic graph manager (v4) — writes G_episodic in Neo4j.

Hooked from EpisodicMemoryManager.insert_event after the PG row has been
committed. Failures are non-fatal: PG remains the source of truth.

Graph elements written here:
  (:Episode {id, user_id, organization_id, summary, occurred_at})
  (:EpisodicEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
  (:Episode)-[:NEXT]->(:Episode)
  (:Episode)-[:MENTIONS {role}]->(:EpisodicEntity)
  (:EpisodicEntity)-[:EP_RELATES {id, keywords, description, weight,
                                  source_episode_ids, valid_at, invalid_at,
                                  expired_at, keywords_embedding}]
                                  ->(:EpisodicEntity)

CAUSED_BY edges are reserved for a future optional LLM step (P2 leaves them
unused — write path stays at ~1 LLM call/insert).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import (
    embed_batch,
    gen_id,
    iso,
    llm_model_from_agent,
    normalize_name,
)
from mirix.services.lightrag_extractor import (
    ExtractedEntity,
    ExtractedRelation,
    extract_entities_and_relations,
)
from mirix.services.lightrag_merger import merge_descriptions
from mirix.settings import settings

logger = get_logger(__name__)


class EpisodicGraphManager:
    """Stateless coordinator. Construct one per call."""

    async def process_episode(
        self,
        *,
        episode_id: str,
        summary: str,
        details: str,
        occurred_at: datetime,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Run the full episodic write path. Never raises."""
        if not settings.enable_graph_memory:
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        text = (summary or "") + ("\n" + details if details else "")

        # W2: extract first so we know what to embed/upsert
        extraction = await extract_entities_and_relations(
            text=text, llm_model=llm_model_from_agent(agent_state)
        )

        # W1: always create the Episode node, even when extraction is empty
        await self._upsert_episode(
            driver,
            episode_id=episode_id,
            summary=summary,
            occurred_at=occurred_at,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W6: connect Episode to previous Episode by occurred_at (auto NEXT)
        await self._link_next(driver, user_id=user_id, episode_id=episode_id, occurred_at=occurred_at)

        if not extraction.entities and not extraction.relations:
            return {"entities": 0, "relations": 0}

        # W3: upsert EpisodicEntity nodes
        merged_entities = await self._upsert_entities(
            driver,
            entities=extraction.entities,
            episode_id=episode_id,
            agent_state=agent_state,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W4: upsert EP_RELATES edges
        merged_relations = await self._upsert_relations(
            driver,
            relations=extraction.relations,
            episode_id=episode_id,
            occurred_at=occurred_at,
            agent_state=agent_state,
            user_id=user_id,
            llm_model=llm_model_from_agent(agent_state),
        )

        # W7: refresh rank (degree)
        touched_names = sorted({e.name for e in extraction.entities})
        await self._refresh_ranks(driver, names=touched_names, user_id=user_id)

        return {
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
            "merged_entities": merged_entities,
            "merged_relations": merged_relations,
        }

    # --------------------------------------------------------- W1: Episode

    async def _upsert_episode(
        self,
        driver,
        *,
        episode_id: str,
        summary: str,
        occurred_at: datetime,
        user_id: str,
        organization_id: str,
    ) -> None:
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MERGE (e:Episode {id: $id})
                ON CREATE SET e.user_id = $user_id,
                              e.organization_id = $org_id,
                              e.summary = $summary,
                              e.occurred_at = $occurred_at
                ON MATCH SET e.summary = $summary,
                             e.occurred_at = $occurred_at
                """,
                id=episode_id,
                user_id=user_id,
                org_id=organization_id,
                summary=summary or "",
                occurred_at=iso(occurred_at),
            )

    # ------------------------------------------------------- W6: NEXT edges

    async def _link_next(
        self, driver, *, user_id: str, episode_id: str, occurred_at: datetime
    ) -> None:
        """
        Connect the new episode to the most recent prior episode (same user)
        with a :NEXT edge. Idempotent: if a NEXT edge from the same prior
        episode already exists, MERGE keeps it.
        """
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MATCH (current:Episode {id: $id})
                OPTIONAL MATCH (prev:Episode {user_id: $user_id})
                WHERE prev.id <> $id AND prev.occurred_at < $occurred_at
                WITH current, prev
                ORDER BY prev.occurred_at DESC
                LIMIT 1
                FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
                    MERGE (prev)-[:NEXT]->(current)
                )
                """,
                id=episode_id,
                user_id=user_id,
                occurred_at=iso(occurred_at),
            )

    # -------------------------------------------------------- W3: Entities

    async def _upsert_entities(
        self,
        driver,
        *,
        entities: list[ExtractedEntity],
        episode_id: str,
        agent_state: AgentState,
        user_id: str,
        organization_id: str,
    ) -> int:
        if not entities:
            return 0

        # Fetch existing entities by (user_id, name_lower) in one round-trip
        name_lowers = [normalize_name(e.name) for e in entities]
        existing = await self._fetch_existing_entities(driver, user_id, name_lowers)

        # Embed names for entities not yet in the graph
        new_entities = [e for e in entities if normalize_name(e.name) not in existing]
        new_embeddings = await embed_batch([e.name for e in new_entities], agent_state)
        new_emb_map: dict[str, Optional[list[float]]] = {
            normalize_name(e.name): emb for e, emb in zip(new_entities, new_embeddings)
        }

        now = iso(datetime.now(timezone.utc))
        merged_count = 0
        llm_model = llm_model_from_agent(agent_state)

        new_rows: list[dict[str, Any]] = []
        for e in new_entities:
            nl = normalize_name(e.name)
            new_rows.append({
                "id": gen_id("epent"),
                "name": e.name,
                "name_lower": nl,
                "entity_type": e.entity_type,
                "description": e.description,
                "name_embedding": new_emb_map.get(nl),
                "user_id": user_id,
                "organization_id": organization_id,
                "created_at": now,
                "updated_at": now,
            })

        # Update path: merge descriptions for entities that already exist
        update_rows: list[dict[str, Any]] = []
        for e in entities:
            nl = normalize_name(e.name)
            existing_row = existing.get(nl)
            if existing_row is None:
                continue
            old_desc = existing_row.get("description") or ""
            new_desc = e.description or ""
            if not new_desc.strip() or new_desc.strip() == old_desc.strip():
                continue
            merged, llm_used = await merge_descriptions(
                description_type="episodic entity",
                name=existing_row["name"],
                descriptions=[old_desc, new_desc] if old_desc else [new_desc],
                llm_model=llm_model,
            )
            if llm_used:
                merged_count += 1
            update_rows.append({"id": existing_row["id"], "description": merged, "updated_at": now})

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (e:EpisodicEntity {
                        id: row.id,
                        name: row.name,
                        name_lower: row.name_lower,
                        entity_type: row.entity_type,
                        description: row.description,
                        rank: 0,
                        user_id: row.user_id,
                        organization_id: row.organization_id,
                        created_at: row.created_at,
                        updated_at: row.updated_at
                    })
                    WITH e, row
                    CALL {
                        WITH e, row
                        WITH e, row WHERE row.name_embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'name_embedding', row.name_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(e) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:EpisodicEntity {id: row.id})
                    SET e.description = row.description, e.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

            # MENTIONS edges from Episode → EpisodicEntity (covers both new + existing)
            mention_rows = [
                {"episode_id": episode_id, "name_lower": normalize_name(e.name), "user_id": user_id}
                for e in entities
            ]
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (ep:Episode {id: row.episode_id})
                MATCH (e:EpisodicEntity {user_id: row.user_id, name_lower: row.name_lower})
                MERGE (ep)-[m:MENTIONS]->(e)
                ON CREATE SET m.role = 'MENTIONED'
                """,
                rows=mention_rows,
            )

        return merged_count

    async def _fetch_existing_entities(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:EpisodicEntity {user_id: $user_id, name_lower: nl})
                RETURN e.id AS id, e.name AS name, e.name_lower AS name_lower,
                       e.description AS description, e.entity_type AS entity_type
                """,
                names=name_lowers,
                user_id=user_id,
            )
            out: dict[str, dict[str, Any]] = {}
            async for rec in result:
                out[rec["name_lower"]] = dict(rec)
        return out

    # -------------------------------------------------------- W4: EP_RELATES

    async def _upsert_relations(
        self,
        driver,
        *,
        relations: list[ExtractedRelation],
        episode_id: str,
        occurred_at: datetime,
        agent_state: AgentState,
        user_id: str,
        llm_model: str,
    ) -> int:
        if not relations:
            return 0

        kw_embeddings = await embed_batch(
            [r.keywords or r.description for r in relations], agent_state
        )

        pairs = [(normalize_name(r.src), normalize_name(r.tgt)) for r in relations]
        existing_edges = await self._fetch_existing_edges(driver, user_id, pairs)

        now = iso(datetime.now(timezone.utc))
        valid_at = iso(occurred_at)
        merged_count = 0

        new_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        for r, kw_emb in zip(relations, kw_embeddings):
            a = normalize_name(r.src)
            b = normalize_name(r.tgt)
            key = tuple(sorted([a, b]))
            existing = existing_edges.get(key)
            if existing is None:
                new_rows.append({
                    "id": gen_id("eprel"),
                    "src_lower": a,
                    "tgt_lower": b,
                    "user_id": user_id,
                    "keywords": r.keywords,
                    "description": r.description,
                    "weight": float(r.weight),
                    "valid_at": valid_at,
                    "created_at": now,
                    "source_episode_ids": [episode_id],
                    "keywords_embedding": kw_emb,
                })
                continue

            # Merge description, average weight, accumulate source_episode_ids
            old_desc = existing.get("description") or ""
            new_desc = r.description or ""
            if old_desc.strip() and new_desc.strip() and old_desc.strip() != new_desc.strip():
                merged_desc, llm_used = await merge_descriptions(
                    description_type="episodic relation",
                    name=f"{r.src} <-> {r.tgt}",
                    descriptions=[old_desc, new_desc],
                    llm_model=llm_model,
                )
                if llm_used:
                    merged_count += 1
            else:
                merged_desc = new_desc or old_desc

            old_weight = float(existing.get("weight") or 0.5)
            new_weight = (old_weight + float(r.weight)) / 2.0
            old_sources: list[str] = list(existing.get("source_episode_ids") or [])
            if episode_id not in old_sources:
                old_sources.append(episode_id)
            update_rows.append({
                "id": existing["id"],
                "description": merged_desc,
                "weight": new_weight,
                "source_episode_ids": old_sources,
                "updated_at": now,
            })

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:EpisodicEntity {user_id: row.user_id, name_lower: row.src_lower})
                    MATCH (b:EpisodicEntity {user_id: row.user_id, name_lower: row.tgt_lower})
                    CREATE (a)-[r:EP_RELATES {
                        id: row.id,
                        keywords: row.keywords,
                        description: row.description,
                        weight: row.weight,
                        valid_at: row.valid_at,
                        created_at: row.created_at,
                        source_episode_ids: row.source_episode_ids
                    }]->(b)
                    WITH r, row
                    CALL {
                        WITH r, row
                        WITH r, row WHERE row.keywords_embedding IS NOT NULL
                        CALL db.create.setRelationshipVectorProperty(r, 'keywords_embedding', row.keywords_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(r) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH ()-[r:EP_RELATES {id: row.id}]->()
                    SET r.description = row.description,
                        r.weight = row.weight,
                        r.source_episode_ids = row.source_episode_ids,
                        r.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

        return merged_count

    async def _fetch_existing_edges(
        self, driver, user_id: str, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not pairs:
            return {}
        rows = [{"a": a, "b": b} for a, b in pairs]
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $rows AS row
                MATCH (x:EpisodicEntity {user_id: $user_id, name_lower: row.a})
                MATCH (y:EpisodicEntity {user_id: $user_id, name_lower: row.b})
                MATCH (x)-[r:EP_RELATES]-(y)
                WHERE r.expired_at IS NULL
                RETURN row.a AS a, row.b AS b,
                       r.id AS id, r.description AS description,
                       r.weight AS weight, r.source_episode_ids AS source_episode_ids
                """,
                rows=rows,
                user_id=user_id,
            )
            out: dict[tuple[str, str], dict[str, Any]] = {}
            async for rec in result:
                key = tuple(sorted([rec["a"], rec["b"]]))
                out.setdefault(key, {
                    "id": rec["id"],
                    "description": rec["description"],
                    "weight": rec["weight"],
                    "source_episode_ids": rec["source_episode_ids"],
                })
        return out

    # -------------------------------------------------------- W7: ranks

    async def _refresh_ranks(self, driver, *, names: list[str], user_id: str) -> None:
        if not names:
            return
        name_lowers = [normalize_name(n) for n in names]
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:EpisodicEntity {user_id: $user_id, name_lower: nl})
                OPTIONAL MATCH (e)-[r:EP_RELATES]-()
                WHERE r.expired_at IS NULL
                WITH e, count(r) AS deg
                SET e.rank = deg
                """,
                names=name_lowers,
                user_id=user_id,
            )
