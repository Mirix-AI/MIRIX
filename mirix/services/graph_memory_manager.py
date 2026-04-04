"""
Graph Memory Manager for MIRIX-v2.

Write path: entity/relation extraction, entity dedup, edge conflict detection.
Read path: seed discovery, graph expansion (2-hop BFS), scoring, context formatting.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text as sa_text

from mirix.embeddings import embedding_model
from mirix.orm.graph_memory import EntityEdge, EntityNode, EpisodeNode, InvolvesEdge
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.settings import settings

logger = logging.getLogger("Mirix.GraphMemoryManager")


def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


class GraphMemoryManager:
    """Manages the temporal knowledge graph (entity nodes, edges, episodes)."""

    # -----------------------------------------------------------------
    # Write path
    # -----------------------------------------------------------------

    async def extract_entities_and_relations(
        self,
        text: str,
        ref_timestamp: str,
        llm_config,
    ) -> Dict[str, Any]:
        """Step W2: Extract entities and relations from text using LLM (1 call)."""
        from mirix.llm_api.helpers import make_post_request

        system_prompt = (
            "You are a memory extraction agent. Given a conversation snippet, "
            "extract structured information in the following JSON format.\n\n"
            "{\n"
            '  "entities": [\n'
            '    {"name": "Alice Chen", "type": "PERSON"},\n'
            '    {"name": "Meta", "type": "ORGANIZATION"}\n'
            "  ],\n"
            '  "relations": [\n'
            "    {\n"
            '      "src": "Alice Chen",\n'
            '      "rel_type": "WORKS_AT",\n'
            '      "dst": "Meta",\n'
            '      "fact_text": "Alice Chen works at Meta",\n'
            '      "valid_at": "2024-03-01T00:00:00Z",\n'
            '      "invalid_at": null\n'
            "    }\n"
            "  ],\n"
            '  "episode_entities": ["Alice Chen", "Meta"]\n'
            "}\n\n"
            "Rules:\n"
            "- Entity names: use full names, be consistent\n"
            "- Only extract temporal info directly stated or clearly implied\n"
            "- If only relative time (e.g., 'last year'), calculate from reference "
            f"timestamp: {ref_timestamp}\n"
            "- If no temporal info, leave valid_at/invalid_at null\n"
            "- Output valid JSON only, no markdown fences"
        )

        try:
            # Use OpenAI API directly (simpler than going through MIRIX LLM client)
            import os
            import httpx

            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = llm_config.model if hasattr(llm_config, "model") else "gpt-4.1-mini"
            endpoint = "https://api.openai.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:8000]},  # Truncate to avoid token limits
                ],
                "temperature": 0,
                "max_tokens": 2000,
            }

            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"]
            # Strip markdown fences if present
            if "```" in content:
                content = content.split("```json")[-1].split("```")[0] if "```json" in content else content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            logger.warning("Graph extraction failed: %s", e)
            return {"entities": [], "relations": [], "episode_entities": []}

    async def find_or_create_entity(
        self,
        name: str,
        entity_type: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> str:
        """Step W3: Entity dedup — find existing entity by name or create new one."""
        from mirix.server.server import db_context

        async with db_context() as session:
            # Exact name match (case-insensitive), filter by user_id for consistency with read path
            result = await session.execute(
                sa_text(
                    "SELECT id, name FROM entity_nodes "
                    "WHERE lower(name) = lower(:name) "
                    "AND user_id = :user_id "
                    "AND (is_deleted = false OR is_deleted IS NULL) "
                    "LIMIT 1"
                ),
                {"name": name, "user_id": user_id},
            )
            row = result.fetchone()
            if row:
                return row[0]

            # No match — create new entity
            entity_id = _gen_id("entity")
            now = datetime.now(timezone.utc)

            # Optionally compute embedding
            embedding = None
            try:
                embed_model = await embedding_model(agent_state.embedding_config)
                embedding = await embed_model.get_text_embedding(f"{name} ({entity_type})")
            except Exception as e:
                logger.debug("Embedding failed for entity %s: %s", name, e)

            entity = EntityNode(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                summary=None,
                embedding=embedding,
                organization_id=organization_id,
                user_id=user_id,
            )
            session.add(entity)
            await session.commit()
            return entity_id

    async def insert_edge(
        self,
        src_id: str,
        dst_id: str,
        rel_type: str,
        fact_text: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
        source_episode_id: Optional[str] = None,
        valid_at: Optional[datetime] = None,
        invalid_at: Optional[datetime] = None,
    ) -> str:
        """Step W4: Insert edge with simple conflict detection (no LLM call for now)."""
        from mirix.server.server import db_context

        async with db_context() as session:
            # Check for conflicting active edge with same (src, rel_type)
            result = await session.execute(
                sa_text(
                    "SELECT id, fact_text FROM entity_edges "
                    "WHERE src_id = :src AND rel_type = :rel "
                    "AND expired_at IS NULL "
                    "AND user_id = :user_id "
                    "AND (is_deleted = false OR is_deleted IS NULL) "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"src": src_id, "rel": rel_type, "user_id": user_id},
            )
            old = result.fetchone()
            if old and old[1] != fact_text:
                # Expire old edge
                await session.execute(
                    sa_text("UPDATE entity_edges SET expired_at = :now WHERE id = :eid"),
                    {"now": datetime.now(timezone.utc), "eid": old[0]},
                )

            # Insert new edge
            edge_id = _gen_id("edge")
            embedding = None
            try:
                embed_model = await embedding_model(agent_state.embedding_config)
                embedding = await embed_model.get_text_embedding(fact_text)
            except Exception as e:
                logger.debug("Embedding failed for edge: %s", e)

            edge = EntityEdge(
                id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                rel_type=rel_type,
                fact_text=fact_text,
                embedding=embedding,
                valid_at=valid_at,
                invalid_at=invalid_at,
                source_episode_id=source_episode_id,
                organization_id=organization_id,
                user_id=user_id,
            )
            session.add(edge)
            await session.commit()
            return edge_id

    async def insert_episode_node(
        self,
        summary: str,
        details: Optional[str],
        event_time: datetime,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
        source_type: str = "conversation",
    ) -> str:
        """Step W1: Create episode node in the graph."""
        from mirix.server.server import db_context

        episode_id = _gen_id("episode")
        embedding = None
        try:
            embed_model = await embedding_model(agent_state.embedding_config)
            embedding = await embed_model.get_text_embedding(summary)
        except Exception as e:
            logger.debug("Embedding failed for episode: %s", e)

        async with db_context() as session:
            ep = EpisodeNode(
                id=episode_id,
                summary=summary,
                details=details,
                embedding=embedding,
                event_time=event_time,
                source_type=source_type,
                organization_id=organization_id,
                user_id=user_id,
            )
            session.add(ep)
            await session.commit()
        return episode_id

    async def insert_involves_edge(
        self,
        episode_id: str,
        entity_id: str,
        role: str = "MENTIONED",
    ) -> None:
        """Step W5: Link episode to entity."""
        from mirix.server.server import db_context

        async with db_context() as session:
            # Check if already exists
            result = await session.execute(
                sa_text(
                    "SELECT id FROM involves_edges "
                    "WHERE episode_id = :ep AND entity_id = :en "
                    "LIMIT 1"
                ),
                {"ep": episode_id, "en": entity_id},
            )
            if result.fetchone():
                return

            ie = InvolvesEdge(
                id=_gen_id("involves"),
                episode_id=episode_id,
                entity_id=entity_id,
                role=role,
            )
            session.add(ie)
            await session.commit()

    async def process_for_graph(
        self,
        text: str,
        summary: str,
        details: Optional[str],
        event_time: datetime,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> None:
        """Full write path: W1 → W2 → W3 → W4 → W5."""
        if not settings.enable_graph_memory:
            return

        try:
            # W1: Episode node
            episode_id = await self.insert_episode_node(
                summary=summary,
                details=details,
                event_time=event_time,
                agent_state=agent_state,
                organization_id=organization_id,
                user_id=user_id,
            )

            # W2: Extract entities and relations
            ref_ts = event_time.isoformat() if event_time else datetime.now(timezone.utc).isoformat()
            extraction = await self.extract_entities_and_relations(
                text=text,
                ref_timestamp=ref_ts,
                llm_config=agent_state.llm_config,
            )

            # W3: Find or create entities
            entity_name_to_id = {}
            for ent in extraction.get("entities", []):
                eid = await self.find_or_create_entity(
                    name=ent["name"],
                    entity_type=ent.get("type", "GENERIC"),
                    agent_state=agent_state,
                    organization_id=organization_id,
                    user_id=user_id,
                )
                entity_name_to_id[ent["name"]] = eid

            # W4: Insert edges
            for rel in extraction.get("relations", []):
                src_name = rel.get("src", "")
                dst_name = rel.get("dst", "")
                if src_name not in entity_name_to_id or dst_name not in entity_name_to_id:
                    continue
                valid_at = None
                invalid_at = None
                if rel.get("valid_at"):
                    try:
                        valid_at = datetime.fromisoformat(rel["valid_at"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass
                if rel.get("invalid_at"):
                    try:
                        invalid_at = datetime.fromisoformat(rel["invalid_at"].replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        pass

                await self.insert_edge(
                    src_id=entity_name_to_id[src_name],
                    dst_id=entity_name_to_id[dst_name],
                    rel_type=rel.get("rel_type", "RELATED_TO"),
                    fact_text=rel.get("fact_text", ""),
                    agent_state=agent_state,
                    organization_id=organization_id,
                    user_id=user_id,
                    source_episode_id=episode_id,
                    valid_at=valid_at,
                    invalid_at=invalid_at,
                )

            # W5: Involves edges
            for ent_name in extraction.get("episode_entities", []):
                if ent_name in entity_name_to_id:
                    await self.insert_involves_edge(
                        episode_id=episode_id,
                        entity_id=entity_name_to_id[ent_name],
                    )

            logger.info(
                "Graph memory: extracted %d entities, %d relations for episode %s",
                len(extraction.get("entities", [])),
                len(extraction.get("relations", [])),
                episode_id,
            )

        except Exception as e:
            logger.error("Graph memory write failed: %s", e, exc_info=True)

    # -----------------------------------------------------------------
    # Read path
    # -----------------------------------------------------------------

    @staticmethod
    def _score_candidate(
        query_embedding: Optional[list],
        candidate_embedding: Optional[list],
        candidate_ts: Optional[datetime],
        hop_distance: int = 0,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        half_life_days: float = 30.0,
    ) -> float:
        """R3: Score a candidate edge/episode by relevance, recency, and proximity."""
        score = 0.0

        # Semantic relevance (cosine similarity)
        if query_embedding and candidate_embedding:
            q = np.array(query_embedding)
            c = np.array(candidate_embedding)
            norm = np.linalg.norm(q) * np.linalg.norm(c)
            if norm > 0:
                score += alpha * (np.dot(q, c) / norm)

        # Recency decay
        if candidate_ts:
            age_days = (datetime.now(timezone.utc) - candidate_ts).total_seconds() / 86400
            recency = np.exp(-0.693 * max(age_days, 0) / half_life_days)
            score += beta * recency

        # Hop proximity
        score += gamma * (1.0 / (1.0 + hop_distance))

        return score

    async def retrieve_graph_context(
        self,
        query: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
        top_k_edges: int = 15,
        top_k_episodes: int = 5,
        max_hops: int = 2,
    ) -> str:
        """Full read path: R1 → R2 → R3 → R4. Returns formatted context string."""
        if not settings.enable_graph_memory:
            return ""

        from mirix.server.server import db_context

        try:
            # Compute query embedding for scoring
            query_embedding = None
            try:
                embed_model = await embedding_model(agent_state.embedding_config)
                query_embedding = await embed_model.get_text_embedding(query or "general")
            except Exception:
                pass

            # ============================================================
            # R1: Seed discovery — union of query-relevant + recent entities
            # ============================================================
            async with db_context() as session:
                seed_ids_set = set()

                # R1a: BM25 keyword search — match multiple query words
                if query:
                    words = [w.lower() for w in query.split() if len(w) > 2][:3]
                    for word in words:
                        kw_result = await session.execute(
                            sa_text(
                                "SELECT id FROM entity_nodes "
                                "WHERE user_id = :user_id "
                                "AND (is_deleted = false OR is_deleted IS NULL) "
                                "AND (lower(name) LIKE :pattern OR lower(COALESCE(summary,'')) LIKE :pattern) "
                                "LIMIT 5"
                            ),
                            {"user_id": user_id, "pattern": f"%{word}%"},
                        )
                        for row in kw_result.fetchall():
                            seed_ids_set.add(row[0])

                # R1b: Embedding similarity search
                if query_embedding:
                    emb_result = await session.execute(
                        sa_text(
                            "SELECT id FROM entity_nodes "
                            "WHERE user_id = :user_id "
                            "AND (is_deleted = false OR is_deleted IS NULL) "
                            "AND embedding IS NOT NULL "
                            "ORDER BY embedding <=> cast(:qemb as vector) "
                            "LIMIT 5"
                        ),
                        {"user_id": user_id, "qemb": str(query_embedding)},
                    )
                    for row in emb_result.fetchall():
                        seed_ids_set.add(row[0])

                # R1c: Always include recent entities (union, not fallback)
                recent_result = await session.execute(
                    sa_text(
                        "SELECT id FROM entity_nodes "
                        "WHERE user_id = :user_id "
                        "AND (is_deleted = false OR is_deleted IS NULL) "
                        "ORDER BY created_at DESC LIMIT 5"
                    ),
                    {"user_id": user_id},
                )
                for row in recent_result.fetchall():
                    seed_ids_set.add(row[0])

            if not seed_ids_set:
                return ""

            seed_ids = list(seed_ids_set)

            # R1d: Direct BM25 search on edge fact_text (for temporal/multi-hop)
            # This finds edges whose fact contains query keywords, even if
            # the connected entities weren't found by R1a-R1c
            if query:
                async with db_context() as session:
                    words = [w.lower() for w in query.split() if len(w) > 2][:4]
                    for word in words:
                        edge_kw = await session.execute(
                            sa_text(
                                "SELECT DISTINCT e.src_id, e.dst_id "
                                "FROM entity_edges e "
                                "WHERE lower(e.fact_text) LIKE :pattern "
                                "AND e.expired_at IS NULL "
                                "AND e.user_id = :user_id "
                                "AND (e.is_deleted = false OR e.is_deleted IS NULL) "
                                "LIMIT 5"
                            ),
                            {"pattern": f"%{word}%", "user_id": user_id},
                        )
                        for row in edge_kw.fetchall():
                            seed_ids_set.add(row[0])
                            seed_ids_set.add(row[1])

                seed_ids = list(seed_ids_set)

            logger.info("Graph R1: %d seed entities for query '%s'", len(seed_ids), (query or "")[:40])

            # ============================================================
            # R2: 2-hop graph expansion — seeds → hop1 neighbors → hop2
            # ============================================================
            async with db_context() as session:
                # Hop 1: edges from seeds
                ph0 = ", ".join(f":s{i}" for i in range(len(seed_ids)))
                p0 = {f"s{i}": sid for i, sid in enumerate(seed_ids)}

                hop1_result = await session.execute(
                    sa_text(
                        f"SELECT DISTINCT "
                        f"CASE WHEN e.src_id IN ({ph0}) THEN e.dst_id ELSE e.src_id END AS neighbor_id "
                        f"FROM entity_edges e "
                        f"WHERE (e.src_id IN ({ph0}) OR e.dst_id IN ({ph0})) "
                        f"AND e.expired_at IS NULL "
                        f"AND (e.is_deleted = false OR e.is_deleted IS NULL)"
                    ),
                    p0,
                )
                hop1_ids = set(row[0] for row in hop1_result.fetchall())

                # All expanded entity IDs (seeds + hop1) with hop tracking
                all_entity_ids = set(seed_ids) | hop1_ids
                all_ids = list(all_entity_ids)
                seed_set = set(seed_ids)  # for hop distance calculation

                placeholders = ", ".join(f":id{i}" for i in range(len(all_ids)))
                params = {f"id{i}": eid for i, eid in enumerate(all_ids)}

                # Get all active edges touching expanded entities
                edge_result = await session.execute(
                    sa_text(
                        f"SELECT DISTINCT e.id, e.fact_text, e.embedding, "
                        f"e.valid_at, e.invalid_at, e.created_at, e.src_id, e.dst_id "
                        f"FROM entity_edges e "
                        f"WHERE (e.src_id IN ({placeholders}) OR e.dst_id IN ({placeholders})) "
                        f"AND e.expired_at IS NULL "
                        f"AND (e.is_deleted = false OR e.is_deleted IS NULL) "
                        f"LIMIT 80"
                    ),
                    params,
                )
                raw_edges = edge_result.fetchall()

                # Get episodes connected to expanded entities
                episode_result = await session.execute(
                    sa_text(
                        f"SELECT DISTINCT ep.id, ep.summary, ep.embedding, ep.event_time "
                        f"FROM episode_nodes ep "
                        f"JOIN involves_edges ie ON ie.episode_id = ep.id "
                        f"WHERE ie.entity_id IN ({placeholders}) "
                        f"AND (ep.is_deleted = false OR ep.is_deleted IS NULL) "
                        f"LIMIT 30"
                    ),
                    params,
                )
                raw_episodes = episode_result.fetchall()

            logger.info("Graph R2: expanded %d->%d entities, found %d edges, %d episodes",
                        len(seed_ids), len(all_ids), len(raw_edges), len(raw_episodes))

            # ============================================================
            # R3: Score & prune
            # ============================================================
            def _parse_embedding(emb):
                """Parse pgvector embedding (may be string, list, or numpy array)."""
                if emb is None:
                    return None
                if isinstance(emb, (list, np.ndarray)):
                    return list(emb)
                # pgvector returns string like "[0.01,0.02,...]"
                try:
                    return json.loads(str(emb))
                except (json.JSONDecodeError, TypeError):
                    return None

            scored_edges = []
            for e in raw_edges:
                eid, fact, emb, valid_at, invalid_at, created_at, src_id, dst_id = e
                emb_list = _parse_embedding(emb)
                # Use valid_at for recency (when the fact happened), fallback to created_at
                ts_for_recency = valid_at or created_at
                # Hop distance: 0 if either end is a seed entity, 1 otherwise
                hop = 0 if (src_id in seed_set or dst_id in seed_set) else 1
                score = self._score_candidate(
                    query_embedding=query_embedding,
                    candidate_embedding=emb_list,
                    candidate_ts=ts_for_recency,
                    hop_distance=hop,
                )
                scored_edges.append((score, fact, valid_at, invalid_at))

            scored_edges.sort(key=lambda x: x[0], reverse=True)
            top_edges = scored_edges[:top_k_edges]

            scored_episodes = []
            for ep in raw_episodes:
                epid, summary, emb, event_time = ep
                emb_list = _parse_embedding(emb)
                score = self._score_candidate(
                    query_embedding=query_embedding,
                    candidate_embedding=emb_list,
                    candidate_ts=event_time,
                    hop_distance=0,
                )
                scored_episodes.append((score, summary, event_time))

            scored_episodes.sort(key=lambda x: x[0], reverse=True)
            top_episodes = scored_episodes[:top_k_episodes]

            logger.info(
                "Graph R3: %d/%d edges, %d/%d episodes after scoring",
                len(top_edges), len(raw_edges), len(top_episodes), len(raw_episodes),
            )

            # ============================================================
            # R4: Format context (full dates for temporal reasoning)
            # ============================================================
            lines = []
            if top_edges:
                lines.append("## Relevant Facts (from knowledge graph)")
                for score, fact, valid_at, invalid_at in top_edges:
                    validity = ""
                    if valid_at and invalid_at:
                        validity = f" (from {valid_at.strftime('%d %B %Y')} to {invalid_at.strftime('%d %B %Y')})"
                    elif valid_at:
                        validity = f" (on/since {valid_at.strftime('%d %B %Y')})"
                    lines.append(f"- {fact}{validity}")

            if top_episodes:
                lines.append("\n## Recent Related Events (from knowledge graph)")
                for score, summary, event_time in top_episodes:
                    ts = event_time.strftime("%d %B %Y") if event_time else "unknown"
                    lines.append(f"- [{ts}] {summary}")

            return "\n".join(lines)

        except Exception as e:
            logger.error("Graph memory retrieval failed: %s", e, exc_info=True)
            return ""
