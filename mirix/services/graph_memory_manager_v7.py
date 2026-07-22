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
import hashlib
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

# Anchor merge key (A): collapses case / punctuation / word-order and
# singular/plural so "Mobile App"/"Mobile Apps", "Flight"/"Flights",
# "Price"/"Prices" merge onto ONE V7Anchor instead of fragmenting into
# near-duplicate nodes. Deliberately conservative \u2014 numbers are preserved
# (so "10 Gallons" != "20 Gallons", and distinct dates stay distinct) and only
# plural endings are normalized (no verb stemming, so "Training" != "Trains").
_ANCHOR_STOPWORDS = {"the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "with"}


def _singularize(token: str) -> str:
    if token.isdigit() or len(token) < 4:
        return token
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", token):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def anchor_canonical_key(name: str) -> str:
    """Canonical merge key for a V7 anchor name (see note above)."""
    cleaned = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    tokens = {_singularize(t) for t in cleaned.split() if t and t not in _ANCHOR_STOPWORDS}
    return " ".join(sorted(tokens))


_PRED_COPULA = re.compile(r"^(is|are|was|were|be|been|being)\s+", re.I)


def canon_predicate(pred: str) -> str:
    """Canonical predicate so surface variants are ONE relation, not several:
    ``includes``->``include``, ``is located in``->``located in``, ``offers``->``offer``.
    Applied at write time so the graph never accumulates the variants in the first
    place (a corpus-wide cleanup previously had to fold 99 of them)."""
    if not pred:
        return "related to"
    p = _PRED_COPULA.sub("", pred.lower().strip())
    tokens = p.split()
    if tokens:
        head = tokens[0]
        # singularize the verb, but keep has/is/ss-words intact
        if len(head) > 4 and head.endswith("s") and not head.endswith(("ss", "us", "is", "as")):
            tokens[0] = head[:-1]
    return " ".join(tokens) or "related to"


def fact_identity(user_id: str, subj_key: str, predicate: str, obj_key: str) -> str:
    """Deterministic id for a fact, so the SAME (subject, predicate, object) asserted
    in N memories MERGEs to ONE V7Fact cited N times — instead of N duplicate nodes.
    A random id here was what generated the duplicate-triple redundancy."""
    raw = f"{user_id}|{subj_key}|{predicate}|{obj_key}"
    return "v7fact-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


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
        entities: Optional[list[ExtractedEntity]] = None,
        role: Optional[str] = None,
    ) -> dict[str, Any]:
        if not settings.enable_graph_memory or settings.graph_version not in ("v7", "v7.1", "v7.2", "v7.3", "v7.4", "v7.6", "v7.7", "v7.8", "v7.9", "v7.10", "v8"):
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}
        if not text or not text.strip():
            return {"anchors": 0}

        # Only name/entity_type are consumed downstream, so a caller that already
        # knows the entities (v7.3 extracts them in the same call that produces the
        # proposition) can pass them and skip the LightRAG round-trip entirely.
        relations: list[tuple[str, str, str]] = []
        if entities is None:
            if settings.graph_version == "v7.4":
                # v7.4: local GLiNER encoder instead of the per-memory LightRAG LLM
                # call (~60x faster; drops User/Assistant noise hubs).
                from mirix.services.gliner_extractor import extract_entities_gliner
                entities = await extract_entities_gliner(text)
            elif settings.graph_version in ("v7.6", "v7.8", "v7.10"):
                # v7.6 (direction D): one LLM call -> typed entities + relations.
                # Keeps abstraction (GLiNER can't) and the relations v7 discarded,
                # which become anchor->anchor edges below.
                from mirix.services.triple_extractor import extract_triples
                res = await extract_triples(text, model=llm_model_from_agent(agent_state))
                entities = res.entities
                relations = res.relations
                if settings.graph_version in ("v7.8", "v7.10"):
                    # v7.8: registry-guided canonicalization — reuse existing anchor
                    # names for the same entity so the graph stops fragmenting into
                    # near-duplicate singletons. Also rename relation endpoints so the
                    # anchor->anchor edges land on the canonical anchors.
                    from mirix.services.entity_resolver import canonicalize_entities
                    rename = await canonicalize_entities(
                        entities, driver=driver, user_id=user_id, agent_state=agent_state)
                    if rename:
                        relations = [(rename.get(s, s), r, rename.get(o, o)) for s, r, o in relations]
            else:
                extraction = await extract_entities_and_relations(
                    text=text, llm_model=llm_model_from_agent(agent_state)
                )
                entities = extraction.entities
        candidates = self._select_anchors(
            entities,
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

        # v7.6 (direction D): materialise the extracted relations as anchor->anchor
        # edges. v7 discarded relations entirely; these give retrieval a graph to
        # propagate over (PPR) and are what make multi-hop paths traversable.
        if relations:
            await self._link_relation_edges(driver, relations=relations, user_id=user_id)

        # v7.10 (hypergraph): reify each triple as a V7Fact HYPEREDGE node linking its
        # subject entity, object entity, the source memory, plus role (who) + time (when)
        # as node properties — a multi-dimensional (entity × person/role × time) index
        # over facts, instead of the binary anchor→anchor edge alone.
        if settings.graph_version == "v7.10" and relations:
            await self._upsert_facts(
                driver, relations=relations, memory_ref_id=memory_ref_id,
                role=role, timestamp=timestamp, user_id=user_id)

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
            nl = anchor_canonical_key(name)
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
                    m.preview = $preview,
                    m.source_key = $source_key,
                    m.updated_at = $now,
                    m.created_at = coalesce(m.created_at, $now)
                SET m.timestamp = $timestamp
                """,
                # NOTE: title and source_meta_json are intentionally NOT stored.
                # The retriever only reads memory_id (then fetches full details
                # from PG), and V7_SUPPORTED_BY is built from source_key — so both
                # fields were write-only dead weight. preview is kept as a short
                # debug snippet (it already falls back to title when no summary).
                id=ref_id,
                memory_id=memory_id,
                memory_type=source_kind,
                user_id=user_id,
                organization_id=organization_id,
                preview=preview,
                source_key=source_key,
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

    async def _upsert_facts(
        self, driver, *, relations: list[tuple[str, str, str]], memory_ref_id: str,
        role: Optional[str], timestamp: Optional[str], user_id: str,
    ) -> None:
        """v7.10 hypergraph: each triple becomes a V7Fact HYPEREDGE node connecting its
        subject anchor, object anchor and the source memory, carrying role (user/
        assistant/shared) and time as properties. This reifies the n-ary fact so it can
        be queried on any dimension (entity, who, when) — e.g. "assistant-role facts
        about X in month M" — which a binary anchor→anchor edge cannot express."""
        rows = []
        seen: set[str] = set()
        for subj, rel, obj in relations:
            sk = anchor_canonical_key(self._clean_name(subj))
            ok = anchor_canonical_key(self._clean_name(obj))
            # Skip tautologies ("french -include-> french"): an extraction artifact that
            # asserts nothing. Mirrors the sk != ok guard in _link_relation_edges.
            if not sk or not ok or sk == ok:
                continue
            pred = canon_predicate(rel)[:80]
            fid = fact_identity(user_id, sk, pred, ok)
            if fid in seen:  # same triple repeated inside one memory
                continue
            seen.add(fid)
            rows.append({"fid": fid, "s": sk, "o": ok, "pred": pred})
        if not rows:
            return
        now = iso(datetime.now(timezone.utc))
        role_norm = (role or "shared").lower()
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MATCH (m:V7MemoryRef {id: $mref, user_id: $user_id})
                UNWIND $rows AS row
                MATCH (s:V7Anchor {user_id: $user_id, name_lower: row.s})
                MATCH (o:V7Anchor {user_id: $user_id, name_lower: row.o})
                MERGE (f:V7Fact {id: row.fid})
                  ON CREATE SET f.user_id = $user_id, f.predicate = row.pred,
                                f.role = $role, f.timestamp = $ts, f.created_at = $now
                MERGE (f)-[:V7_FACT_SUBJECT]->(s)
                MERGE (f)-[:V7_FACT_OBJECT]->(o)
                MERGE (f)-[cite:V7_FACT_FROM]->(m)
                  ON CREATE SET cite.role = $role, cite.timestamp = $ts
                """,
                mref=memory_ref_id, user_id=user_id, rows=rows,
                role=role_norm, ts=timestamp, now=now)

    async def _link_relation_edges(
        self, driver, *, relations: list[tuple[str, str, str]], user_id: str
    ) -> None:
        """v7.6: materialise (subject, relation, object) triples as anchor->anchor
        V7_RELATION edges. Endpoints are matched by the same canonical key anchors
        merge on, so an edge is only created when both entities survived as anchors.
        Distinct relation phrases between the same pair are distinct edges."""
        rows = []
        for subj, rel, obj in relations:
            sk = anchor_canonical_key(self._clean_name(subj))
            ok = anchor_canonical_key(self._clean_name(obj))
            if sk and ok and sk != ok:
                rows.append({"s": sk, "o": ok, "r": (rel or "related to")[:80]})
        if not rows:
            return
        now = iso(datetime.now(timezone.utc))
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (a:V7Anchor {user_id: $user_id, name_lower: row.s})
                MATCH (b:V7Anchor {user_id: $user_id, name_lower: row.o})
                MERGE (a)-[e:V7_RELATION {rel: row.r}]->(b)
                ON CREATE SET e.created_at = $now
                """,
                rows=rows,
                user_id=user_id,
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

    # ------------------------------------------------------------- v8 finalize

    async def prune_singletons(self, user_id: str) -> dict[str, Any]:
        """v8 finalize pass: delete degree-1 anchors — anchors that link only a
        single memory ref and therefore create no cross-memory retrieval path
        (the value a graph adds over flat PG vector search). An anchor's final
        degree is only known after ALL of a user's memories are ingested, so
        this must run ONCE at the end of ingestion, not incrementally.

        No-op unless ``graph_version == 'v8'`` — v7 keeps every admitted anchor.
        Safe: only anchor nodes are removed (never memory refs or PG rows), and
        every memory remains reachable via its multi-degree anchors, the
        V7_SUPPORTED_BY / V7_NEXT_MEMORY bridges, and flat PG search.
        """
        if not settings.enable_graph_memory or settings.graph_version != "v8":
            return {"skipped": "not_v8"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        async with driver.session(database=settings.neo4j_database) as session:
            count_res = await session.run(
                """
                MATCH (a:V7Anchor {user_id: $uid})-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(m:V7MemoryRef)
                WITH a, count(DISTINCT m) AS deg WHERE deg = 1
                RETURN count(a) AS n
                """,
                uid=user_id,
            )
            rec = await count_res.single()
            n = int(rec["n"]) if rec else 0
            if n:
                await session.run(
                    """
                    MATCH (a:V7Anchor {user_id: $uid})-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(m:V7MemoryRef)
                    WITH a, count(DISTINCT m) AS deg WHERE deg = 1
                    DETACH DELETE a
                    """,
                    uid=user_id,
                )
        logger.info("v8 prune_singletons: removed %d degree-1 anchors for user=%s", n, user_id)
        return {"pruned": n}
