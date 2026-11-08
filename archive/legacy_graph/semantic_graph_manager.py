"""
Semantic graph manager (v4) — writes G_semantic in Neo4j.

Hooked from SemanticMemoryManager.insert_semantic_item after the PG row has
been committed. Failures are non-fatal.

Graph elements written here:
  (:Concept {id, user_id, organization_id, name, summary, created_at})
  (:SemanticEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
  (:Concept)-[:CONCEPT_RELATES {keywords, description, weight,
                                 keywords_embedding}]->(:Concept)
  (:Concept)-[:MENTIONS]->(:SemanticEntity)
  (:SemanticEntity)-[:SEM_RELATES {id, keywords, description, weight,
                                    source_concept_ids, keywords_embedding}]
                                    ->(:SemanticEntity)

Concept-Concept edges are LLM-judged: when a new Concept is inserted, the
top-K most similar existing Concepts (by name embedding) are candidates;
one LLM call decides which actually have a meaningful relationship.

Cost per insert: ~1 LLM call (entity extraction) + ~0.3-1 LLM call (description
merging + concept relation judgement). Heavier than episodic by design — the
semantic graph is small and dense, so investing in good edges pays off.
"""

from __future__ import annotations

import json
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
    call_openai_chat,
    extract_entities_and_relations,
)
from mirix.services.lightrag_merger import merge_descriptions
from mirix.settings import settings

logger = get_logger(__name__)


# Concept-concept relation candidate pool size. Top-K nearest concepts (by
# name embedding) get sent to the LLM for relation judgement in one batch.
DEFAULT_CONCEPT_REL_TOP_K = 5

# Concept-concept relation judgement prompt. Asks the LLM to return JSON for
# which candidates actually relate to the new concept and how.
_CONCEPT_REL_PROMPT = """You are a knowledge graph editor. A new concept has been added to the user's semantic memory. Decide which of the candidate concepts have a meaningful relationship with the new one.

New concept:
  name: {new_name}
  summary: {new_summary}

Candidate concepts (existing in the graph):
{candidates_block}

For each candidate that genuinely relates to the new concept (e.g. IS_A, PART_OF, RELATES_TO, CONTRADICTS, ENABLES, CAUSED_BY), output one JSON object per line. Skip candidates that are unrelated or duplicates. Output strict JSON, one object per line, no markdown fences. If nothing relates, output nothing.

Each object must have:
  "candidate_name": str  // exact name from the list above
  "keywords": str        // short phrase summarizing the relation type (e.g. "subclass", "part of", "contradicts")
  "description": str     // one sentence explaining the relationship
  "weight": float        // 0.0-1.0 strength
"""


class SemanticGraphManager:
    """Stateless coordinator. Construct one per call."""

    async def process_concept(
        self,
        *,
        concept_id: str,
        name: str,
        summary: str,
        details: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Run the full semantic write path. Never raises."""
        if not settings.enable_graph_memory:
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        text = f"{name}: {summary}\n{details or ''}"
        llm_model = llm_model_from_agent(agent_state)

        # W2: extract entities + entity-entity relations from the concept text
        extraction = await extract_entities_and_relations(text=text, llm_model=llm_model)

        # W1: always create the Concept node
        concept_name_emb = (await embed_batch([name], agent_state))[0]
        await self._upsert_concept(
            driver,
            concept_id=concept_id,
            name=name,
            summary=summary,
            user_id=user_id,
            organization_id=organization_id,
            name_embedding=concept_name_emb,
        )

        # W6: concept-concept relation discovery
        concept_rels_added = await self._discover_concept_relations(
            driver,
            new_concept_id=concept_id,
            new_concept_name=name,
            new_concept_summary=summary,
            new_concept_name_emb=concept_name_emb,
            user_id=user_id,
            agent_state=agent_state,
            llm_model=llm_model,
        )

        if not extraction.entities and not extraction.relations:
            return {"entities": 0, "relations": 0, "concept_rels": concept_rels_added}

        # W3: upsert SemanticEntity nodes
        merged_entities = await self._upsert_entities(
            driver,
            entities=extraction.entities,
            concept_id=concept_id,
            agent_state=agent_state,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W4: upsert SEM_RELATES edges
        merged_relations = await self._upsert_relations(
            driver,
            relations=extraction.relations,
            concept_id=concept_id,
            agent_state=agent_state,
            user_id=user_id,
            llm_model=llm_model,
        )

        # W7: refresh rank
        await self._refresh_ranks(
            driver,
            names=sorted({e.name for e in extraction.entities}),
            user_id=user_id,
        )

        return {
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
            "concept_rels": concept_rels_added,
            "merged_entities": merged_entities,
            "merged_relations": merged_relations,
        }

    # --------------------------------------------------------- W1: Concept

    async def _upsert_concept(
        self,
        driver,
        *,
        concept_id: str,
        name: str,
        summary: str,
        user_id: str,
        organization_id: str,
        name_embedding: Optional[list[float]],
    ) -> None:
        now = iso(datetime.now(timezone.utc))
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MERGE (c:Concept {id: $id})
                ON CREATE SET c.user_id = $user_id,
                              c.organization_id = $org_id,
                              c.name = $name,
                              c.summary = $summary,
                              c.created_at = $now
                ON MATCH SET c.name = $name, c.summary = $summary
                """,
                id=concept_id,
                user_id=user_id,
                org_id=organization_id,
                name=name,
                summary=summary or "",
                now=now,
            )
            if name_embedding:
                # Attach name embedding to the concept itself so we can find
                # similar concepts via vector search later.
                await session.run(
                    """
                    MATCH (c:Concept {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'name_embedding', $emb)
                    RETURN count(*) AS _
                    """,
                    id=concept_id,
                    emb=name_embedding,
                )

    # ----------------------------------------- W6: concept-concept relations

    async def _discover_concept_relations(
        self,
        driver,
        *,
        new_concept_id: str,
        new_concept_name: str,
        new_concept_summary: str,
        new_concept_name_emb: Optional[list[float]],
        user_id: str,
        agent_state: AgentState,
        llm_model: str,
        top_k: int = DEFAULT_CONCEPT_REL_TOP_K,
    ) -> int:
        """Find candidate concepts by name vector, ask LLM which actually relate."""
        if new_concept_name_emb is None:
            return 0

        # Find top-K similar concepts via raw cypher (Concept nodes don't yet
        # have a dedicated vector index by design — we use cosine over the
        # property we set above; small graphs make this affordable). For
        # larger deployments switch to a dedicated index on Concept.
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (c:Concept {user_id: $user_id})
                WHERE c.id <> $id AND c.name_embedding IS NOT NULL
                WITH c, vector.similarity.cosine(c.name_embedding, $emb) AS sim
                ORDER BY sim DESC LIMIT $k
                RETURN c.id AS id, c.name AS name, c.summary AS summary, sim
                """,
                user_id=user_id,
                id=new_concept_id,
                emb=new_concept_name_emb,
                k=top_k,
            )
            candidates = [dict(rec) async for rec in result]

        if not candidates:
            return 0

        candidates_block = "\n".join(
            f"  - {c['name']}: {(c.get('summary') or '')[:200]}" for c in candidates
        )
        prompt = _CONCEPT_REL_PROMPT.format(
            new_name=new_concept_name,
            new_summary=(new_concept_summary or "")[:300],
            candidates_block=candidates_block,
        )

        try:
            raw = await call_openai_chat(
                system_prompt="You are a precise knowledge graph editor. Output JSON only.",
                user_prompt=prompt,
                model=llm_model,
                temperature=0.0,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("Concept relation LLM call failed: %s", e)
            return 0

        # Parse line-delimited JSON, tolerant to LLM noise
        relations: list[dict[str, Any]] = []
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cand_name = (obj.get("candidate_name") or "").strip()
            if not cand_name:
                continue
            # Find candidate id by name (case-insensitive)
            cand = next((c for c in candidates if c["name"].lower() == cand_name.lower()), None)
            if cand is None:
                continue
            relations.append({
                "src_id": new_concept_id,
                "tgt_id": cand["id"],
                "keywords": (obj.get("keywords") or "")[:120],
                "description": (obj.get("description") or "")[:500],
                "weight": float(obj.get("weight") or 0.5),
            })

        if not relations:
            return 0

        # Embed keywords for the new edges
        kw_embs = await embed_batch([r["keywords"] or r["description"] for r in relations], agent_state)
        for r, emb in zip(relations, kw_embs):
            r["keywords_embedding"] = emb

        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Concept {id: row.src_id})
                MATCH (b:Concept {id: row.tgt_id})
                MERGE (a)-[r:CONCEPT_RELATES]->(b)
                ON CREATE SET r.keywords = row.keywords,
                              r.description = row.description,
                              r.weight = row.weight
                ON MATCH SET r.keywords = row.keywords,
                             r.description = row.description,
                             r.weight = (coalesce(r.weight, 0.5) + row.weight) / 2.0
                WITH r, row
                CALL {
                    WITH r, row
                    WITH r, row WHERE row.keywords_embedding IS NOT NULL
                    CALL db.create.setRelationshipVectorProperty(r, 'keywords_embedding', row.keywords_embedding)
                    RETURN count(*) AS _
                }
                RETURN count(r) AS created
                """,
                rows=relations,
            )

        return len(relations)

    # --------------------------------------------------- W3: SemanticEntity

    async def _upsert_entities(
        self,
        driver,
        *,
        entities: list[ExtractedEntity],
        concept_id: str,
        agent_state: AgentState,
        user_id: str,
        organization_id: str,
    ) -> int:
        if not entities:
            return 0

        name_lowers = [normalize_name(e.name) for e in entities]
        existing = await self._fetch_existing_entities(driver, user_id, name_lowers)

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
                "id": gen_id("sement"),
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
                description_type="semantic entity",
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
                    CREATE (e:SemanticEntity {
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
                    MATCH (e:SemanticEntity {id: row.id})
                    SET e.description = row.description, e.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

            # MENTIONS edges from Concept → SemanticEntity
            mention_rows = [
                {"concept_id": concept_id, "name_lower": normalize_name(e.name), "user_id": user_id}
                for e in entities
            ]
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.concept_id})
                MATCH (e:SemanticEntity {user_id: row.user_id, name_lower: row.name_lower})
                MERGE (c)-[m:MENTIONS]->(e)
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
                MATCH (e:SemanticEntity {user_id: $user_id, name_lower: nl})
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

    # -------------------------------------------------------- W4: SEM_RELATES

    async def _upsert_relations(
        self,
        driver,
        *,
        relations: list[ExtractedRelation],
        concept_id: str,
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
        merged_count = 0

        new_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        for r, kw_emb in zip(relations, kw_embeddings):
            a, b = normalize_name(r.src), normalize_name(r.tgt)
            key = tuple(sorted([a, b]))
            existing = existing_edges.get(key)
            if existing is None:
                new_rows.append({
                    "id": gen_id("semrel"),
                    "src_lower": a,
                    "tgt_lower": b,
                    "user_id": user_id,
                    "keywords": r.keywords,
                    "description": r.description,
                    "weight": float(r.weight),
                    "created_at": now,
                    "source_concept_ids": [concept_id],
                    "keywords_embedding": kw_emb,
                })
                continue

            old_desc = existing.get("description") or ""
            new_desc = r.description or ""
            if old_desc.strip() and new_desc.strip() and old_desc.strip() != new_desc.strip():
                merged_desc, llm_used = await merge_descriptions(
                    description_type="semantic relation",
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
            old_sources: list[str] = list(existing.get("source_concept_ids") or [])
            if concept_id not in old_sources:
                old_sources.append(concept_id)
            update_rows.append({
                "id": existing["id"],
                "description": merged_desc,
                "weight": new_weight,
                "source_concept_ids": old_sources,
                "updated_at": now,
            })

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:SemanticEntity {user_id: row.user_id, name_lower: row.src_lower})
                    MATCH (b:SemanticEntity {user_id: row.user_id, name_lower: row.tgt_lower})
                    CREATE (a)-[r:SEM_RELATES {
                        id: row.id,
                        keywords: row.keywords,
                        description: row.description,
                        weight: row.weight,
                        created_at: row.created_at,
                        source_concept_ids: row.source_concept_ids
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
                    MATCH ()-[r:SEM_RELATES {id: row.id}]->()
                    SET r.description = row.description,
                        r.weight = row.weight,
                        r.source_concept_ids = row.source_concept_ids,
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
                MATCH (x:SemanticEntity {user_id: $user_id, name_lower: row.a})
                MATCH (y:SemanticEntity {user_id: $user_id, name_lower: row.b})
                MATCH (x)-[r:SEM_RELATES]-(y)
                RETURN row.a AS a, row.b AS b,
                       r.id AS id, r.description AS description,
                       r.weight AS weight, r.source_concept_ids AS source_concept_ids
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
                    "source_concept_ids": rec["source_concept_ids"],
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
                MATCH (e:SemanticEntity {user_id: $user_id, name_lower: nl})
                OPTIONAL MATCH (e)-[r:SEM_RELATES]-()
                WITH e, count(r) AS deg
                SET e.rank = deg
                """,
                names=name_lowers,
                user_id=user_id,
            )
