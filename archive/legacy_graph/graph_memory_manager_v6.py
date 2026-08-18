"""
v6 graph manager — lean entity index.

Design contrast with v5:
- v5 builds two full LightRAG graphs (G_episodic + G_semantic), each with
  Episode/Concept item nodes, EpisodicEntity/SemanticEntity entity nodes,
  weighted EP_RELATES/SEM_RELATES edges, plus CONCEPT_RELATES edges built
  via an LLM judgement pass. Retrieval walks the graph with dual-level
  (entity name + relation keyword) vector search and 1-hop expansion.
- v6 throws all that away. It builds a single :V6Entity label with one
  vector index on the entity name. Each V6Entity carries back-references
  (episodic_ids / semantic_ids) pointing at the PG rows that mention it.
  A single edge type :V6_COOCCUR captures co-occurrence within the same
  source chunk, with a count property — no description, no weight, no
  embedding. Retrieval is purely:
      query embed → vector search on name → 1-hop co-occur expand → PG fetch

Write cost per insert: 1 LLM call (LightRAG entity extraction, reused
from v5) + N embedding calls (one per new entity name) + a few Cypher
upserts. Roughly half the LLM/Neo4j work of v5 because we skip relation
extraction's prompting overhead, skip merge_descriptions, and skip
CONCEPT_RELATES LLM judgement.

Both EpisodicMemoryManager.insert_event and SemanticMemoryManager.insert_item
hook into the same process_chunk() entry point with different source_kind.
The V6Entity rows are shared across the two sources — "Alice" mentioned in
both an episode and a semantic item collapses into one node with both
episodic_ids and semantic_ids populated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import (
    embed_batch,
    gen_id,
    iso,
    llm_model_from_agent,
    normalize_name,
)
from mirix.services.lightrag_extractor import extract_entities_and_relations
from mirix.settings import settings

logger = get_logger(__name__)


SourceKind = Literal["episodic", "semantic"]


class V6GraphManager:
    """Stateless. Construct one per call."""

    async def process_chunk(
        self,
        *,
        source_kind: SourceKind,
        source_id: str,
        text: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Extract entities, upsert :V6Entity + back-refs, write co-occur edges.

        Never raises. Returns a small stats dict for logging.
        """
        if not settings.enable_graph_memory or settings.graph_version != "v6":
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        if not text or not text.strip():
            return {"entities": 0}

        # Reuse v5's LightRAG extractor. We only need entity names; the
        # relations payload is discarded (co-occurrence edges below are
        # derived from entity pairs in the same chunk, not from LightRAG's
        # relation tuples — they're noisier and add LLM cost we don't need).
        extraction = await extract_entities_and_relations(
            text=text, llm_model=llm_model_from_agent(agent_state)
        )
        entity_names_raw = [e.name for e in extraction.entities if e.name and e.name.strip()]
        # dedup within a single chunk on normalized name
        seen: set[str] = set()
        entity_names: list[str] = []
        entity_types: dict[str, str] = {}
        for e in extraction.entities:
            nl = normalize_name(e.name)
            if not nl or nl in seen:
                continue
            seen.add(nl)
            entity_names.append(e.name)
            entity_types[nl] = e.entity_type or "Other"

        if not entity_names:
            return {"entities": 0}

        merged_count = await self._upsert_entities_with_backref(
            driver,
            names=entity_names,
            entity_types=entity_types,
            source_kind=source_kind,
            source_id=source_id,
            agent_state=agent_state,
            user_id=user_id,
            organization_id=organization_id,
        )

        # Co-occurrence edges: all entity pairs in this chunk.
        # Skip when only one entity (no pair).
        if len(entity_names) >= 2:
            await self._upsert_cooccur_edges(
                driver,
                names=entity_names,
                user_id=user_id,
            )

        return {
            "entities": len(entity_names),
            "merged": merged_count,
            "pairs": len(entity_names) * (len(entity_names) - 1) // 2,
        }

    async def _upsert_entities_with_backref(
        self,
        driver,
        *,
        names: list[str],
        entity_types: dict[str, str],
        source_kind: SourceKind,
        source_id: str,
        agent_state: AgentState,
        user_id: str,
        organization_id: str,
    ) -> int:
        """Upsert V6Entity nodes; append source_id to the right back-ref list.

        Returns the number of rows that were merged (already existed).
        """
        name_lowers = [normalize_name(n) for n in names]

        # One round-trip to see which (user_id, name_lower) already exist
        existing = await self._fetch_existing(driver, user_id, name_lowers)

        new_names = [n for n in names if normalize_name(n) not in existing]
        new_embeddings = await embed_batch(new_names, agent_state) if new_names else []
        new_emb_map: dict[str, Optional[list[float]]] = {
            normalize_name(n): emb for n, emb in zip(new_names, new_embeddings)
        }

        now = iso(datetime.now(timezone.utc))
        backref_field = "episodic_ids" if source_kind == "episodic" else "semantic_ids"

        new_rows: list[dict[str, Any]] = []
        for n in new_names:
            nl = normalize_name(n)
            new_rows.append({
                "id": gen_id("v6ent"),
                "name": n,
                "name_lower": nl,
                "entity_type": entity_types.get(nl, "Other"),
                "name_embedding": new_emb_map.get(nl),
                "user_id": user_id,
                "organization_id": organization_id,
                "created_at": now,
                "updated_at": now,
                # initial back-ref list contains just this source_id
                "episodic_ids": [source_id] if source_kind == "episodic" else [],
                "semantic_ids": [source_id] if source_kind == "semantic" else [],
            })

        # For existing nodes, append the source_id (idempotent via list dedup)
        update_rows = [
            {
                "name_lower": nl,
                "source_id": source_id,
                "updated_at": now,
            }
            for nl in name_lowers
            if nl in existing
        ]

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (e:V6Entity {
                        id: row.id,
                        name: row.name,
                        name_lower: row.name_lower,
                        entity_type: row.entity_type,
                        user_id: row.user_id,
                        organization_id: row.organization_id,
                        episodic_ids: row.episodic_ids,
                        semantic_ids: row.semantic_ids,
                        mention_count: 1,
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
                # Append source_id to the matching back-ref list, dedup via APOC-free
                # Cypher list comprehension. mention_count tracks total appearances.
                await session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (e:V6Entity {{user_id: $user_id, name_lower: row.name_lower}})
                    SET e.{backref_field} = CASE
                        WHEN row.source_id IN coalesce(e.{backref_field}, [])
                            THEN e.{backref_field}
                        ELSE coalesce(e.{backref_field}, []) + row.source_id
                    END,
                        e.mention_count = coalesce(e.mention_count, 0) + 1,
                        e.updated_at = row.updated_at
                    """,
                    rows=update_rows, user_id=user_id,
                )

        return len(update_rows)

    async def _fetch_existing(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:V6Entity {user_id: $user_id, name_lower: nl})
                RETURN e.id AS id, e.name AS name, e.name_lower AS name_lower
                """,
                names=name_lowers, user_id=user_id,
            )
            out: dict[str, dict[str, Any]] = {}
            async for rec in result:
                out[rec["name_lower"]] = dict(rec)
        return out

    async def _upsert_cooccur_edges(
        self,
        driver,
        *,
        names: list[str],
        user_id: str,
    ) -> None:
        """For each unordered pair, MERGE a V6_COOCCUR edge and bump count.

        Edges are undirected by convention but Neo4j requires direction —
        we always store with src.name_lower < tgt.name_lower so traversal
        from either side hits the same edge instance.
        """
        nls = sorted({normalize_name(n) for n in names if normalize_name(n)})
        if len(nls) < 2:
            return

        pairs: list[dict[str, str]] = []
        for i in range(len(nls)):
            for j in range(i + 1, len(nls)):
                pairs.append({"src": nls[i], "tgt": nls[j]})

        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $pairs AS p
                MATCH (a:V6Entity {user_id: $user_id, name_lower: p.src})
                MATCH (b:V6Entity {user_id: $user_id, name_lower: p.tgt})
                MERGE (a)-[e:V6_COOCCUR]->(b)
                ON CREATE SET e.count = 1
                ON MATCH SET e.count = e.count + 1
                """,
                pairs=pairs, user_id=user_id,
            )
