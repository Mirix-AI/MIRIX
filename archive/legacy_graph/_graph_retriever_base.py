"""
Shared retriever base for v4 graph readers.

EpisodicRetriever and SemanticRetriever are structurally identical — they
walk one graph (entities → relations → items), dedup, and score. The only
differences are which labels/types they MATCH, which vector indexes they
query, and what extra item-item edges they expand (NEXT for episodes,
CONCEPT_RELATES for concepts). This base factors out the shared algorithm
and lets subclasses fill in label/type strings.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import tiktoken

from mirix.log import get_logger

logger = get_logger(__name__)


# Token budgets and ranking constants. Tuned for chat-memory scale.
DEFAULT_TOP_K = 30
DEFAULT_CHUNK_TOP_K = 10
RECENCY_HALF_LIFE_DAYS = 30.0
COSINE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3


_tokenizer = None


def count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    if not text:
        return 0
    return len(_tokenizer.encode(text))


def recency_decay(ts: Optional[Any]) -> float:
    """Exponential decay over days; missing ts → 0.5."""
    if ts is None:
        return 0.5
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
        if age_days < 0:
            return 1.0
        return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5


def final_score(cosine: float, ts: Optional[Any]) -> float:
    return COSINE_WEIGHT * float(cosine or 0.0) + RECENCY_WEIGHT * recency_decay(ts)


def fmt_date(ts: Optional[Any]) -> str:
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts.strftime("%d %B %Y")
    except Exception:
        return str(ts)


# ────────────────────────────────────────────────────────────── dataclasses

@dataclass
class EntityHit:
    id: str
    name: str
    entity_type: str
    description: str
    rank: int
    cosine: float
    updated_at: Optional[Any]
    score: float = 0.0


@dataclass
class RelationHit:
    id: str
    src_name: str
    tgt_name: str
    keywords: str
    description: str
    weight: float
    cosine: float
    valid_at: Optional[Any]
    score: float = 0.0


@dataclass
class ItemHit:
    """Episode or Concept — same shape so format/budget code is shared."""
    id: str
    label: str           # 'Episode' or 'Concept'
    summary: str
    detail: str          # episode.details or concept.summary (fallback)
    timestamp: Optional[Any]   # occurred_at for Episode, created_at for Concept
    cosine: float
    score: float = 0.0
    source: str = "mentions"  # 'mentions', 'one_hop', etc — for debugging


@dataclass
class GraphSearchResult:
    entities: list[EntityHit] = field(default_factory=list)
    relations: list[RelationHit] = field(default_factory=list)
    items: list[ItemHit] = field(default_factory=list)


# ────────────────────────────────────────── round-robin merge helpers

def round_robin_merge_entities(
    locals_: list[EntityHit], globals_: list[EntityHit]
) -> list[EntityHit]:
    merged: list[EntityHit] = []
    seen: set[str] = set()
    n = max(len(locals_), len(globals_))
    for i in range(n):
        for source in (locals_, globals_):
            if i < len(source):
                hit = source[i]
                key = (hit.name or "").lower() or hit.id
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
    return merged


def round_robin_merge_relations(
    locals_: list[RelationHit], globals_: list[RelationHit]
) -> list[RelationHit]:
    merged: list[RelationHit] = []
    seen: set[tuple[str, str]] = set()
    n = max(len(locals_), len(globals_))
    for i in range(n):
        for source in (locals_, globals_):
            if i < len(source):
                hit = source[i]
                key = tuple(sorted([(hit.src_name or "").lower(), (hit.tgt_name or "").lower()]))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
    return merged


# ────────────────────────────────────────── token budget application

def apply_budget_to_search(
    search: GraphSearchResult,
    *,
    max_entity_tokens: int,
    max_relation_tokens: int,
    max_item_tokens: int,
) -> GraphSearchResult:
    def _trim(items: list, render, budget: int) -> list:
        kept = []
        used = 0
        for it in items:
            t = count_tokens(render(it))
            if used + t > budget:
                break
            kept.append(it)
            used += t
        return kept

    kept_e = _trim(
        search.entities,
        lambda e: f"- {e.name} ({e.entity_type}, rank={e.rank}): {e.description}",
        max_entity_tokens,
    )
    kept_r = _trim(
        search.relations,
        lambda r: f"- {r.src_name} <-> {r.tgt_name} [{r.keywords}]: {r.description}",
        max_relation_tokens,
    )
    kept_i = _trim(
        search.items,
        lambda it: f"- [{fmt_date(it.timestamp)}] {it.summary} {it.detail[:200]}",
        max_item_tokens,
    )
    return GraphSearchResult(entities=kept_e, relations=kept_r, items=kept_i)


# ────────────────────────────────────────── retriever base

class GraphRetrieverBase:
    """Subclasses set these as class attributes."""

    # Labels
    ENTITY_LABEL: str = ""        # 'EpisodicEntity' or 'SemanticEntity'
    ITEM_LABEL: str = ""          # 'Episode' or 'Concept'
    REL_TYPE: str = ""            # 'EP_RELATES' or 'SEM_RELATES'

    # Vector index names
    ENTITY_VECTOR_INDEX: str = "" # 'ep_entity_name_emb' or 'sem_entity_name_emb'
    REL_VECTOR_INDEX: str = ""    # 'ep_rel_kw_emb' or 'sem_rel_kw_emb'

    # Title used in markdown output
    SECTION_TITLE: str = ""       # 'Episodic' or 'Semantic'

    # ──────────────────────────────────────────────────────── low-level path

    def _ll_cypher(self) -> str:
        # Vector search on entity names → seed entities. For each seed, pull
        # one-hop neighbor entities via REL_TYPE (no expired_at on semantic
        # rels — but the predicate is harmless: missing property tests as
        # NULL which evaluates the WHERE filter as "IS NULL = true").
        return f"""
        CALL db.index.vector.queryNodes('{self.ENTITY_VECTOR_INDEX}', $top_k, $emb)
        YIELD node AS e, score AS sim
        WHERE e.user_id = $user_id
        WITH e, sim
        ORDER BY sim DESC LIMIT $top_k
        WITH collect({{e: e, sim: sim}}) AS seeds
        UNWIND seeds AS s
        WITH s.e AS seed, s.sim AS sim
        OPTIONAL MATCH (seed)-[r:{self.REL_TYPE}]-(other:{self.ENTITY_LABEL} {{user_id: $user_id}})
        WHERE coalesce(r.expired_at, datetime('9999-01-01')) > datetime()
        RETURN
            seed.id AS sid, seed.name AS sname, seed.entity_type AS stype,
            seed.description AS sdesc, coalesce(seed.rank, 0) AS srank,
            seed.updated_at AS supdated, sim AS sim,
            r.id AS rid, r.description AS rdesc, r.keywords AS rkw,
            coalesce(r.weight, 0.5) AS rweight, r.valid_at AS rvalid,
            other.name AS oname
        """

    def _hl_cypher(self) -> str:
        # Vector search on relation keywords → seed relations. Pull both
        # endpoint entities. We don't have to filter by endpoint label since
        # REL_TYPE alone identifies the graph (EP_RELATES vs SEM_RELATES).
        return f"""
        CALL db.index.vector.queryRelationships('{self.REL_VECTOR_INDEX}', $top_k, $emb)
        YIELD relationship AS r, score AS sim
        WHERE coalesce(r.expired_at, datetime('9999-01-01')) > datetime()
        MATCH (a:{self.ENTITY_LABEL})-[r]-(b:{self.ENTITY_LABEL})
        WHERE a.user_id = $user_id AND b.user_id = $user_id
        WITH r, sim, a, b
        ORDER BY sim DESC LIMIT $top_k
        RETURN
            r.id AS rid, r.description AS rdesc, r.keywords AS rkw,
            coalesce(r.weight, 0.5) AS rweight, r.valid_at AS rvalid, sim AS sim,
            a.id AS aid, a.name AS aname, a.entity_type AS atype,
            a.description AS adesc, coalesce(a.rank, 0) AS arank, a.updated_at AS aupdated,
            b.id AS bid, b.name AS bname, b.entity_type AS btype,
            b.description AS bdesc, coalesce(b.rank, 0) AS brank, b.updated_at AS bupdated
        """

    # ─────────────────────────────────────── high-level orchestration

    async def search(
        self,
        *,
        driver,
        user_id: str,
        ll_embedding: Optional[list[float]],
        hl_embedding: Optional[list[float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[list[EntityHit], list[RelationHit]]:
        """Run ll + hl in parallel where embeddings are available."""
        from mirix.settings import settings

        tasks: dict[str, asyncio.Task] = {}
        if ll_embedding is not None:
            tasks["ll"] = asyncio.create_task(
                self._run_ll(driver, user_id, ll_embedding, top_k, settings.neo4j_database)
            )
        if hl_embedding is not None:
            tasks["hl"] = asyncio.create_task(
                self._run_hl(driver, user_id, hl_embedding, top_k, settings.neo4j_database)
            )
        if not tasks:
            return [], []
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        local_e, local_r, global_e, global_r = [], [], [], []
        for purpose, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                logger.warning("%s retrieval branch %s failed: %s", self.SECTION_TITLE, purpose, res)
                continue
            if purpose == "ll":
                local_e, local_r = res
            elif purpose == "hl":
                global_e, global_r = res

        final_e = round_robin_merge_entities(local_e, global_e)
        final_r = round_robin_merge_relations(local_r, global_r)
        for e in final_e:
            e.score = final_score(e.cosine, e.updated_at)
        for r in final_r:
            r.score = final_score(r.cosine, r.valid_at)
        final_e.sort(key=lambda x: x.score, reverse=True)
        final_r.sort(key=lambda x: x.score, reverse=True)
        return final_e, final_r

    async def _run_ll(self, driver, user_id, emb, top_k, database):
        entities: dict[str, EntityHit] = {}
        relations: dict[str, RelationHit] = {}
        async with driver.session(database=database) as session:
            result = await session.run(self._ll_cypher(), user_id=user_id, emb=emb, top_k=top_k)
            async for rec in result:
                sid = rec["sid"]
                if sid and sid not in entities:
                    entities[sid] = EntityHit(
                        id=sid, name=rec["sname"],
                        entity_type=rec["stype"] or "Other",
                        description=rec["sdesc"] or "",
                        rank=int(rec["srank"] or 0),
                        cosine=float(rec["sim"] or 0.0),
                        updated_at=rec["supdated"],
                    )
                rid = rec["rid"]
                if rid and rid not in relations:
                    relations[rid] = RelationHit(
                        id=rid,
                        src_name=rec["sname"], tgt_name=rec["oname"] or "",
                        keywords=rec["rkw"] or "", description=rec["rdesc"] or "",
                        weight=float(rec["rweight"] or 0.5),
                        cosine=float(rec["sim"] or 0.0),
                        valid_at=rec["rvalid"],
                    )
        return list(entities.values()), list(relations.values())

    async def _run_hl(self, driver, user_id, emb, top_k, database):
        entities: dict[str, EntityHit] = {}
        relations: dict[str, RelationHit] = {}
        async with driver.session(database=database) as session:
            result = await session.run(self._hl_cypher(), user_id=user_id, emb=emb, top_k=top_k)
            async for rec in result:
                rid = rec["rid"]
                if rid and rid not in relations:
                    relations[rid] = RelationHit(
                        id=rid,
                        src_name=rec["aname"] or "", tgt_name=rec["bname"] or "",
                        keywords=rec["rkw"] or "", description=rec["rdesc"] or "",
                        weight=float(rec["rweight"] or 0.5),
                        cosine=float(rec["sim"] or 0.0),
                        valid_at=rec["rvalid"],
                    )
                for prefix in ("a", "b"):
                    eid = rec[f"{prefix}id"]
                    if not eid or eid in entities:
                        continue
                    entities[eid] = EntityHit(
                        id=eid, name=rec[f"{prefix}name"],
                        entity_type=rec[f"{prefix}type"] or "Other",
                        description=rec[f"{prefix}desc"] or "",
                        rank=int(rec[f"{prefix}rank"] or 0),
                        cosine=float(rec["sim"] or 0.0),
                        updated_at=rec[f"{prefix}updated"],
                    )
        return list(entities.values()), list(relations.values())
