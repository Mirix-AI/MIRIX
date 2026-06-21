"""
v7 graph manager - minimal semantic+episodic linkage graph.

v7 keeps the useful part of v6 (Neo4j as an index into PG flat memory), but
tightens the ontology:

- Details stay in PostgreSQL. Graph nodes store ids, types, canonical names,
  timestamps, and a short title/preview only for debugging.
- Anchors must be specific enough to be useful. Generic noun phrases are
  discarded instead of becoming graph nodes.
- Semantic and episodic memory refs live in one graph and share the same
  anchors. Semantic refs are linked back to episodic refs from the same
  source chunk when provenance is available.
- No entity-entity co-occurrence edges. Every edge must be a retrieval path:
  anchor -> memory ref, semantic ref -> supporting episode, or temporal next.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
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
from mirix.services.lightrag_extractor import ExtractedEntity, extract_entities_and_relations
from mirix.settings import settings

logger = get_logger(__name__)


SourceKind = Literal["episodic", "semantic"]

MAX_ANCHORS_PER_EPISODE = 8
MAX_ANCHORS_PER_SEMANTIC = 10
PREVIEW_CHARS = 160

_GENERIC_NAMES = {
    "advice", "approach", "benefits", "best practices", "challenge",
    "challenges", "concept", "considerations", "details", "example",
    "examples", "experience", "feedback", "flexibility", "goal", "goals",
    "guidance", "habit", "help", "idea", "ideas", "information",
    "insights", "issue", "issues", "method", "methods", "option",
    "options", "plan", "plans", "practice", "practices", "preference",
    "recommendation", "recommendations", "routine", "schedule", "skills",
    "social media", "steps", "strategy", "support", "task", "tasks",
    "thing", "things", "tips", "topic", "topics", "update", "updates",
    "way", "ways",
}

_GENERIC_SUFFIXES = (
    " advice", " approach", " benefits", " considerations", " details",
    " examples", " experience", " feedback", " guidance", " ideas",
    " information", " method", " methods", " options", " plan", " plans",
    " recommendations", " routine", " schedule", " strategy", " tips",
)

_SPECIFIC_HINT_RE = re.compile(r"(\d|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|['\u2019])")


@dataclass(frozen=True)
class V7AnchorCandidate:
    name: str
    name_lower: str
    anchor_type: str
    score: float


class V7GraphManager:
    """Stateless. Construct one per graph write."""

    async def process_memory(
        self,
        *,
        source_kind: SourceKind,
        source_id: str,
        text: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        occurred_at: Optional[object] = None,
        source_meta: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not settings.enable_graph_memory or settings.graph_version != "v7":
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}
        if not text or not text.strip():
            return {"anchors": 0}

        extraction = await extract_entities_and_relations(
            text=text, llm_model=llm_model_from_agent(agent_state)
        )
        candidates = self._select_anchors(
            extraction.entities,
            max_anchors=MAX_ANCHORS_PER_EPISODE if source_kind == "episodic" else MAX_ANCHORS_PER_SEMANTIC,
        )

        memory_ref_id = f"{source_kind}:{source_id}"
        source_key = self._source_key(source_meta)
        timestamp = self._to_iso(occurred_at) or (source_meta or {}).get("occurred_at")
        preview = self._preview(summary or title or text)

        await self._upsert_memory_ref(
            driver,
            source_kind=source_kind,
            ref_id=memory_ref_id,
            memory_id=source_id,
            title=title or "",
            preview=preview,
            timestamp=timestamp,
            source_key=source_key,
            source_meta=source_meta or {},
            organization_id=organization_id,
            user_id=user_id,
        )

        if candidates:
            await self._upsert_anchors_and_edges(
                driver,
                anchors=candidates,
                source_kind=source_kind,
                memory_ref_id=memory_ref_id,
                agent_state=agent_state,
                organization_id=organization_id,
                user_id=user_id,
            )

        await self._link_support_edges(
            driver,
            source_kind=source_kind,
            memory_ref_id=memory_ref_id,
            user_id=user_id,
            source_key=source_key,
        )
        if source_kind == "episodic":
            await self._link_temporal_edge(driver, memory_ref_id=memory_ref_id, user_id=user_id, timestamp=timestamp)

        return {
            "anchors": len(candidates),
            "memory_ref": memory_ref_id,
            "source_key": source_key,
        }

    # ------------------------------------------------------------------ gate

    def _select_anchors(self, entities: list[ExtractedEntity], *, max_anchors: int) -> list[V7AnchorCandidate]:
        by_name: dict[str, V7AnchorCandidate] = {}
        for entity in entities:
            name = self._clean_name(entity.name)
            nl = normalize_name(name)
            if not name or not nl:
                continue
            score = self._specificity_score(name, entity.entity_type or "Other")
            if score <= 0:
                continue
            candidate = V7AnchorCandidate(
                name=name,
                name_lower=nl,
                anchor_type=entity.entity_type or "Other",
                score=score,
            )
            existing = by_name.get(nl)
            if existing is None or candidate.score > existing.score:
                by_name[nl] = candidate

        return sorted(by_name.values(), key=lambda c: (-c.score, c.name_lower))[:max_anchors]

    def _specificity_score(self, name: str, entity_type: str) -> float:
        nl = normalize_name(name)
        if not nl or nl in _GENERIC_NAMES:
            return 0.0
        if any(nl.endswith(suffix) for suffix in _GENERIC_SUFFIXES):
            return 0.0
        if len(nl) < 3:
            return 0.0

        score = 0.0
        type_norm = entity_type.strip().lower()
        if type_norm in {"person", "location", "organization", "event"}:
            score += 4.0
        elif type_norm in {"content", "object"}:
            score += 3.0
        elif type_norm in {"concept", "method"}:
            score += 1.0
        else:
            score += 0.5

        words = nl.split()
        if len(words) >= 2:
            score += 1.5
        if len(words) >= 3:
            score += 0.5
        if any(ch.isdigit() for ch in name):
            score += 2.0
        if _SPECIFIC_HINT_RE.search(name):
            score += 1.0
        if name[:1].isupper():
            score += 0.5
        if len(words) == 1 and name[:1].isupper() and len(nl) >= 4:
            score += 2.5
        if len(words) == 1 and type_norm in {"concept", "method", "other"} and score < 3.0:
            return 0.0
        return score

    @staticmethod
    def _clean_name(name: str) -> str:
        return " ".join((name or "").strip().strip("\"'`").split())

    # ------------------------------------------------------------- neo4j write

    async def _upsert_memory_ref(
        self,
        driver,
        *,
        source_kind: SourceKind,
        ref_id: str,
        memory_id: str,
        title: str,
        preview: str,
        timestamp: Optional[str],
        source_key: Optional[str],
        source_meta: dict[str, Any],
        organization_id: str,
        user_id: str,
    ) -> None:
        label = "V7EpisodeRef" if source_kind == "episodic" else "V7ConceptRef"
        now = iso(datetime.now(timezone.utc))
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                f"""
                MERGE (m:V7MemoryRef:{label} {{id: $id}})
                SET m.memory_id = $memory_id,
                    m.memory_type = $memory_type,
                    m.user_id = $user_id,
                    m.organization_id = $organization_id,
                    m.title = $title,
                    m.preview = $preview,
                    m.source_key = $source_key,
                    m.source_meta_json = $source_meta_json,
                    m.updated_at = $now,
                    m.created_at = coalesce(m.created_at, $now)
                SET m.timestamp = $timestamp
                """,
                id=ref_id,
                memory_id=memory_id,
                memory_type=source_kind,
                user_id=user_id,
                organization_id=organization_id,
                title=title[:120],
                preview=preview,
                source_key=source_key,
                source_meta_json=json.dumps(source_meta, sort_keys=True),
                timestamp=timestamp,
                now=now,
            )

    async def _upsert_anchors_and_edges(
        self,
        driver,
        *,
        anchors: list[V7AnchorCandidate],
        source_kind: SourceKind,
        memory_ref_id: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> None:
        existing = await self._fetch_existing_anchors(driver, user_id, [a.name_lower for a in anchors])
        new_anchors = [a for a in anchors if a.name_lower not in existing]
        embeddings = await embed_batch([a.name for a in new_anchors], agent_state) if new_anchors else []
        emb_by_lower = {a.name_lower: emb for a, emb in zip(new_anchors, embeddings)}

        now = iso(datetime.now(timezone.utc))
        rows = [
            {
                "id": existing.get(a.name_lower, {}).get("id") or gen_id("v7anc"),
                "name": a.name,
                "name_lower": a.name_lower,
                "anchor_type": a.anchor_type,
                "score": a.score,
                "name_embedding": emb_by_lower.get(a.name_lower),
            }
            for a in anchors
        ]
        rel_type = "V7_APPEARS_IN" if source_kind == "episodic" else "V7_DESCRIBED_BY"

        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                f"""
                UNWIND $rows AS row
                MERGE (a:V7Anchor {{user_id: $user_id, name_lower: row.name_lower}})
                ON CREATE SET
                    a.id = row.id,
                    a.name = row.name,
                    a.anchor_type = row.anchor_type,
                    a.organization_id = $organization_id,
                    a.created_at = $now,
                    a.mention_count = 0
                SET a.updated_at = $now,
                    a.admission_score = row.score,
                    a.mention_count = coalesce(a.mention_count, 0) + 1
                WITH a, row
                CALL (a, row) {{
                    WITH a, row WHERE row.name_embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty(a, 'name_embedding', row.name_embedding)
                    RETURN count(*) AS _
                }}
                WITH a
                MATCH (m:V7MemoryRef {{id: $memory_ref_id}})
                MERGE (a)-[r:{rel_type}]->(m)
                ON CREATE SET r.created_at = $now
                """,
                rows=rows,
                user_id=user_id,
                organization_id=organization_id,
                memory_ref_id=memory_ref_id,
                now=now,
            )

    async def _fetch_existing_anchors(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (a:V7Anchor {user_id: $user_id, name_lower: nl})
                RETURN a.id AS id, a.name_lower AS name_lower
                """,
                names=name_lowers,
                user_id=user_id,
            )
            return {rec["name_lower"]: dict(rec) async for rec in result}

    async def _link_support_edges(
        self,
        driver,
        *,
        source_kind: SourceKind,
        memory_ref_id: str,
        user_id: str,
        source_key: Optional[str],
    ) -> None:
        if not source_key:
            return
        now = iso(datetime.now(timezone.utc))
        if source_kind == "semantic":
            cypher = """
            MATCH (sem:V7ConceptRef {id: $memory_ref_id, user_id: $user_id})
            MATCH (ep:V7EpisodeRef {user_id: $user_id, source_key: $source_key})
            MERGE (sem)-[r:V7_SUPPORTED_BY]->(ep)
            ON CREATE SET r.created_at = $now, r.reason = 'same_source_chunk'
            """
        else:
            cypher = """
            MATCH (ep:V7EpisodeRef {id: $memory_ref_id, user_id: $user_id})
            MATCH (sem:V7ConceptRef {user_id: $user_id, source_key: $source_key})
            MERGE (sem)-[r:V7_SUPPORTED_BY]->(ep)
            ON CREATE SET r.created_at = $now, r.reason = 'same_source_chunk'
            """
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                cypher,
                memory_ref_id=memory_ref_id,
                user_id=user_id,
                source_key=source_key,
                now=now,
            )

    async def _link_temporal_edge(
        self, driver, *, memory_ref_id: str, user_id: str, timestamp: Optional[str]
    ) -> None:
        if not timestamp:
            return
        now = iso(datetime.now(timezone.utc))
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MATCH (cur:V7EpisodeRef {id: $memory_ref_id, user_id: $user_id})
                MATCH (prev:V7EpisodeRef {user_id: $user_id})
                WHERE prev.id <> cur.id AND prev.timestamp <= $timestamp
                WITH cur, prev
                ORDER BY prev.timestamp DESC
                LIMIT 1
                MERGE (prev)-[r:V7_NEXT_MEMORY]->(cur)
                ON CREATE SET r.created_at = $now
                """,
                memory_ref_id=memory_ref_id,
                user_id=user_id,
                timestamp=timestamp,
                now=now,
            )

    # ---------------------------------------------------------------- utils

    @staticmethod
    def _preview(text: str) -> str:
        return " ".join((text or "").split())[:PREVIEW_CHARS]

    @staticmethod
    def _to_iso(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _source_key(source_meta: Optional[dict[str, Any]]) -> Optional[str]:
        if not source_meta:
            return None
        for key in ("chunk_id", "turn_id", "serial"):
            if source_meta.get(key) is not None:
                return f"{key}:{source_meta[key]}"
        if source_meta.get("occurred_at"):
            return f"occurred_at:{source_meta['occurred_at']}"
        return None
