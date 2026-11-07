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
import importlib
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import embed_batch
from mirix.services.graph_memory_manager_v7 import is_frame_version
from mirix.settings import settings

logger = get_logger(__name__)


DEFAULT_MAX_ITEMS_PER_KIND = 36

# v7.12 frame-hop bounds. A popular anchor sits in hundreds of frames and the
# co-participant fan-out is quadratic in arity, so both hops are capped before the
# reranker ever sees them.
FRAME_FACT_CAP = 240
FRAME_MEMORY_CAP = 60

# v7.12 PG expansion bounds (replacing V7_SUPPORTED_BY / V7_NEXT_MEMORY).
# v7.26 candidate admission. The PG fetch that follows anchor search used to take
# max(max_items_per_kind*3, 45) rows; the sweep showed the whole effect lands by 120 and a
# larger window only helps the longest rows, because evidence_lexical_score has no
# document-length penalty.
CANDIDATE_WINDOW = 120

PG_EXPAND_SEEDS = 24
PG_EXPAND_LIMIT = 60
# Degeneracy guard only. If the seeds' chunk ids match essentially the WHOLE store,
# chunk membership carries no information and the same-chunk hop is skipped. Set high
# on purpose: seeds legitimately span several chunks, and on conv-26 one session alone
# holds 44% of the rows, so a tighter threshold suppresses real expansion.
PG_EXPAND_MAX_CHUNK_SHARE = 0.90

# Over-fetch multipliers for the anchor vector search — see _search_anchors. The cap
# bounds the work when a user genuinely owns fewer than top_k anchors.
ANCHOR_OVERFETCH_STEPS = (1, 8, 40)
ANCHOR_OVERFETCH_CAP = 2000


def _add_unique(target: list[str], values) -> None:
    seen = set(target)
    for raw in values or []:
        if raw is None:
            continue
        v = str(raw)
        if v and v not in seen:
            target.append(v)
            seen.add(v)


@dataclass
class V7AnchorHit:
    id: str
    name: str
    anchor_type: str
    cosine: float
    degree: int = 0
    predicates: list[str] = field(default_factory=list)
    policy_score: Optional[float] = None


@dataclass
class V7MemoryRow:
    id: str
    kind: str
    summary: str
    details: str
    timestamp: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class V7FactCandidate:
    id: str
    predicate: str
    args: list[dict]
    memory_ids: list[str]
    timestamp: Optional[str] = None
    mentioned_at: Optional[str] = None
    lit_keys: list[str] = field(default_factory=list)
    lit_vals: list[str] = field(default_factory=list)
    policy_score: Optional[float] = None


def _source_provenance(source_refs) -> dict:
    """Keep bounded PG citation provenance available to graph-owned QA."""

    refs = [dict(item) for item in (source_refs or []) if isinstance(item, dict)]
    mentioned_at = next(
        (
            str(item.get("mentioned_at") or item.get("occurred_at"))
            for item in refs
            if item.get("mentioned_at") or item.get("occurred_at")
        ),
        None,
    )
    return {"source_refs": refs, "mentioned_at": mentioned_at}



def _policy_minor(version: str | None = None) -> int | None:
    """Minor number of a v7.N graph version, or None if it is not one."""
    m = re.match(r"^v7\.(\d+)$", str(version if version is not None
                                     else settings.graph_version or ""))
    return int(m.group(1)) if m else None


def is_policy_version(version: str | None = None) -> bool:
    """v7.20 and up: the versions that run a retrieval policy at all."""
    n = _policy_minor(version)
    return n is not None and n >= 20


def is_merge_policy_version(version: str | None = None) -> bool:
    """v7.22 and up: the versions whose policy exposes merge_memory_rows.

    v7.20 and v7.21 expose rank_memory_rows instead, a different call shape, so they
    stay on their own branches.
    """
    n = _policy_minor(version)
    return n is not None and n >= 22


def merge_policy_module():
    """The highest retrieval_policy_v7NN at or below the configured version.

    Replaces a chain of exact equality checks that had to be extended by hand for every
    release. There were 25 of those in this file; a new version matched none of them and
    silently fell back to pre-policy behaviour, so its arm would have measured nothing
    while looking like a regression. Resolving downwards means a new version inherits
    its predecessor's policy until it ships one of its own.
    """
    want = _policy_minor()
    if want is None or want < 22:
        return None
    for n in range(want, 21, -1):
        try:
            return importlib.import_module(f"mirix.services.retrieval_policy_v7{n}")
        except ModuleNotFoundError:
            continue
    return None


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
        answer should use these rows as the graph-owned tool output.
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

        vector_anchors = await self._search_anchors(
            driver, user_id, q_emb, top_k, query=query
        )
        exact_anchors: list[V7AnchorHit] = []
        relation_rows: list[V7MemoryRow] = []
        relation_memory_ids: list[str] = []
        query_plan = None
        if is_merge_policy_version():
            _pol = merge_policy_module()
            plan_query = getattr(_pol, "plan_query")
            rank_anchor_hits = getattr(_pol, "rank_anchor_hits")

            query_plan = plan_query(query)
            if query_plan.is_relation_query:
                exact_anchors = await self._search_exact_anchors(driver, user_id, query)
                by_id = {hit.id: hit for hit in vector_anchors}
                for hit in exact_anchors:
                    by_id[hit.id] = hit
                anchors = rank_anchor_hits(query, list(by_id.values()), top_k)
                relation_rows, relation_memory_ids = await self._collect_relation_facts(
                    driver,
                    user_id=user_id,
                    query=query,
                    exact_anchors=exact_anchors,
                    limit=query_plan.relation_quota,
                )
            else:
                anchors = vector_anchors
        elif settings.graph_version == "v7.21":
            from mirix.services.retrieval_policy_v721 import predicate_hints, rank_anchor_hits

            if predicate_hints(query):
                exact_anchors = await self._search_exact_anchors(driver, user_id, query)
                by_id = {hit.id: hit for hit in vector_anchors}
                for hit in exact_anchors:
                    by_id[hit.id] = hit
                anchors = rank_anchor_hits(query, list(by_id.values()), top_k)
                relation_rows, relation_memory_ids = await self._collect_relation_facts(
                    driver,
                    user_id=user_id,
                    query=query,
                    exact_anchors=exact_anchors,
                    limit=min(8, max(4, max_items_per_kind)),
                )
            else:
                anchors = vector_anchors
        else:
            anchors = vector_anchors
        if not anchors:
            return anchors, [], []

        # Single retrieval path for the whole v7 family. The experimental variants
        # that used to branch here — v7.7 Personalized PageRank (retrieval richer, QA
        # flat) and v7.2 per-anchor coverage round-robin (neutral) — are archived; see
        # docs/graph_memory_v7/development_history.md.
        anchor_ids = [a.id for a in anchors]
        if is_frame_version():
            episodic_ids, semantic_ids = await self._collect_ids_from_anchors(
                driver, user_id=user_id, anchor_ids=anchor_ids)
        else:
            episodic_ids, semantic_ids = await self._collect_memory_refs(
                driver, user_id=user_id, anchor_ids=anchor_ids,
            )
        if is_frame_version():
            # v7.12 adds the read side of the fact layer: memories that assert a frame
            # this anchor participates in, then memories about its co-participants.
            # Appended, never substituted — the direct anchor->memory paths above are
            # what every earlier version retrieved on, and the frame hop is additive
            # recall on top of them.
            f_ep, f_sem = await self._collect_frame_refs(
                driver, user_id=user_id, anchor_ids=anchor_ids)
            seen_ep, seen_sem = set(episodic_ids), set(semantic_ids)
            episodic_ids += [i for i in f_ep if i not in seen_ep]
            semantic_ids += [i for i in f_sem if i not in seen_sem]
            logger.info("v7.12 frame hop: +%d ep, +%d sem candidates",
                        len([i for i in f_ep if i not in seen_ep]),
                        len([i for i in f_sem if i not in seen_sem]))
            # same-chunk + temporal neighbours, recomputed from PG in place of the
            # V7_SUPPORTED_BY / V7_NEXT_MEMORY edges the ref layer used to carry.
            x_ep, x_sem = await self._expand_via_pg(user_id, episodic_ids)
            _add_unique(episodic_ids, x_ep)
            _add_unique(semantic_ids, x_sem)
        # Preserve the ordinary graph traversal candidate set before relation
        # citations are appended. v7.22 fetches and ranks the two lanes separately,
        # which prevents high-confidence relation evidence from crowding every
        # base-recall row out of a bounded SQL shortlist.
        base_episodic_ids = list(episodic_ids)
        base_semantic_ids = list(semantic_ids)
        # Relation-fact citations are offered to both PG tables because a V7Fact
        # intentionally stores compact PG ids without duplicating their kind.  Each
        # fetcher is user-scoped and silently ignores ids belonging to the other table.
        if relation_memory_ids:
            _add_unique(episodic_ids, relation_memory_ids)
            _add_unique(semantic_ids, relation_memory_ids)
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
        wide = settings.graph_version == "v7.26"
        fetch_limit = (
            (min(len(ep_arg), CANDIDATE_WINDOW) if wide else max(max_items_per_kind * 3, 45))
            if is_policy_version()
            else max_items_per_kind
        )
        if is_merge_policy_version():
            base_ep_arg = base_episodic_ids if rerank else base_episodic_ids[:max_items_per_kind]
            base_sem_arg = base_semantic_ids if rerank else base_semantic_ids[:max_items_per_kind]
            relation_fetch_limit = (
                min(len(relation_memory_ids), CANDIDATE_WINDOW) if wide
                else max(max_items_per_kind * 2, 24)
            )
            tasks = (
                asyncio.create_task(self._fetch_episodic(
                    user_id, base_ep_arg, q_emb=rerank, limit=fetch_limit,
                )),
                asyncio.create_task(self._fetch_semantic(
                    user_id, base_sem_arg, q_emb=rerank, limit=fetch_limit,
                )),
                asyncio.create_task(self._fetch_episodic(
                    user_id, relation_memory_ids, q_emb=rerank,
                    limit=relation_fetch_limit,
                )),
                asyncio.create_task(self._fetch_semantic(
                    user_id, relation_memory_ids, q_emb=rerank,
                    limit=relation_fetch_limit,
                )),
            )
            ep_rows, sem_rows, relation_ep_rows, relation_sem_rows = await asyncio.gather(
                *tasks, return_exceptions=True
            )
        else:
            ep_task = asyncio.create_task(
                self._fetch_episodic(
                    user_id, ep_arg, q_emb=rerank, limit=fetch_limit,
                    priority_ids=relation_memory_ids,
                ))
            sem_task = asyncio.create_task(
                self._fetch_semantic(
                    user_id, sem_arg, q_emb=rerank, limit=fetch_limit,
                    priority_ids=relation_memory_ids,
                ))
            ep_rows, sem_rows = await asyncio.gather(
                ep_task, sem_task, return_exceptions=True
            )
            relation_ep_rows, relation_sem_rows = [], []
        if isinstance(ep_rows, Exception):
            logger.warning("v7 episodic PG fetch failed: %s", ep_rows)
            ep_rows = []
        if isinstance(sem_rows, Exception):
            logger.warning("v7 semantic PG fetch failed: %s", sem_rows)
            sem_rows = []
        if isinstance(relation_ep_rows, Exception):
            logger.warning("v7.22 relation episodic PG fetch failed: %s", relation_ep_rows)
            relation_ep_rows = []
        if isinstance(relation_sem_rows, Exception):
            logger.warning("v7.22 relation semantic PG fetch failed: %s", relation_sem_rows)
            relation_sem_rows = []

        if is_merge_policy_version():
            _pol = merge_policy_module()
            merge_memory_rows = getattr(_pol, "merge_memory_rows")

            # v7.25 scores both lanes on ONE key instead of stitching them by quota.
            # fact_scores carries the structural evidence per PG row; when it is empty
            # merge_memory_rows falls back to the v7.24 quota merge, so a query with no
            # relation span behaves exactly as before rather than approximately.
            merge_kwargs = {}
            if hasattr(_pol, "fact_scores_from_rows"):
                merge_kwargs["fact_scores"] = _pol.fact_scores_from_rows(relation_rows)
            ep_rows = merge_memory_rows(
                query, ep_rows, relation_ep_rows, max_items_per_kind, **merge_kwargs
            )
            sem_rows = merge_memory_rows(
                query, sem_rows, relation_sem_rows, max_items_per_kind, **merge_kwargs
            )
            sem_rows = relation_rows + sem_rows
        elif settings.graph_version == "v7.21":
            from mirix.services.retrieval_policy_v721 import rank_memory_rows

            ep_rows = rank_memory_rows(
                query, ep_rows, max_items_per_kind, priority_ids=relation_memory_ids
            )
            sem_rows = rank_memory_rows(
                query, sem_rows, max_items_per_kind, priority_ids=relation_memory_ids
            )
            sem_rows = relation_rows + sem_rows
        elif settings.graph_version == "v7.20":
            from mirix.services.retrieval_policy_v720 import rank_memory_rows

            ep_rows = rank_memory_rows(query, ep_rows, max_items_per_kind)
            sem_rows = rank_memory_rows(query, sem_rows, max_items_per_kind)

        # Verification column. A null arm means nothing unless the arm demonstrably ran:
        # MIRIX_SEARCH_LIMIT was once "measured" as having no effect when the code path
        # never fired at all. These three numbers say whether v7.26 actually widened
        # anything on this call.
        logger.info("v7.26 admission: candidates=%d ep_in=%d fetch_limit=%d wide=%s",
                    len(anchors), len(ep_arg), fetch_limit,
                    settings.graph_version == "v7.26")
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
        # Initial retrieval is deliberately compact in v7.20.  The answerer's
        # explicit search tool still asks retrieve_rows for its requested limit;
        # only the automatically injected context is capped here.
        initial_limit = (
            10 if is_policy_version()
            else max_items_per_kind
        )
        anchors, ep_rows, sem_rows = await self.retrieve_rows(
            query=query, user_id=user_id, agent_state=agent_state,
            top_k=top_k, max_items_per_kind=initial_limit)
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
        self, driver, user_id: str, emb: list[float], top_k: int,
        *, query: str = "",
    ) -> list[V7AnchorHit]:
        """Vector search over anchor names, OVER-FETCHED because the user filter runs
        after the index.

        Neo4j's vector index has no pre-filter: ``queryNodes(k)`` returns the global
        top-k and the ``WHERE a.user_id`` clause then throws away everyone else's.
        With more than one user's anchors in one index the caller silently gets far
        fewer than top_k. Measured with conv-26 (717 anchors) alongside longmem_s_0
        (509): the query "how many art events did I attend" returned 18 anchors of
        which 1 belonged to the querying user, and "what did I say about my sister"
        2 of 18. Every longmem eval arm run while the LoCoMo subgraph was resident
        retrieved on a crippled anchor set — the arms are still comparable to each
        other (all were handicapped alike) but none of them measured the graph at
        full strength.

        Fix is to over-fetch and escalate until enough of the user's own anchors
        survive the filter, bounded so a single-user store pays nothing.
        """
        # The tuple was EMPTY, so candidate_k has always collapsed to top_k and this whole
        # branch was dead code. v7.26 turns it on: 64 anchors admitted instead of 15.
        #
        # Admission, not ranking, is what the error triage points at. Of 169 wrong answers,
        # 38 have the answer sitting in the store and never retrieved, and reading them
        # shows a topically adjacent row winning — the right row exists and loses. Widening
        # what is considered before ranking is the one change measured to help BOTH sides
        # offline (gold-supporting row present in the final evidence: 18->24 on wrong
        # questions, 53->60 on right ones). Every answering-side fix tried so far gained on
        # the wrong questions by breaking more of the right ones; this one does not trade.
        precision_policy = settings.graph_version in ("v7.23", "v7.26")
        candidate_k = max(top_k * 4, 64) if precision_policy else top_k
        if precision_policy:
            cypher = """
            CALL db.index.vector.queryNodes('v7_anchor_name_emb', $fetch_k, $emb)
            YIELD node AS a, score AS sim
            WHERE a.user_id = $user_id
            OPTIONAL MATCH (a)<-[:V7_FACT_ARG]-(f:V7Fact)
            WITH a, sim, count(DISTINCT f) AS degree,
                 collect(DISTINCT f.predicate)[..16] AS predicates
            RETURN a.id AS id, a.name AS name, a.anchor_type AS anchor_type,
                   sim AS sim, degree AS degree, predicates AS predicates
            ORDER BY sim DESC
            LIMIT $candidate_k
            """
        else:
            cypher = """
            CALL db.index.vector.queryNodes('v7_anchor_name_emb', $fetch_k, $emb)
            YIELD node AS a, score AS sim
            WHERE a.user_id = $user_id
            RETURN a.id AS id, a.name AS name, a.anchor_type AS anchor_type,
                   sim AS sim, 0 AS degree, [] AS predicates
            ORDER BY sim DESC
            LIMIT $candidate_k
            """
        hits: list[V7AnchorHit] = []
        async with driver.session(database=settings.neo4j_database) as session:
            for factor in ANCHOR_OVERFETCH_STEPS:
                fetch_k = min(candidate_k * factor, ANCHOR_OVERFETCH_CAP)
                result = await session.run(
                    cypher, fetch_k=fetch_k, candidate_k=candidate_k,
                    emb=emb, user_id=user_id)
                hits = [
                    V7AnchorHit(
                        id=rec["id"],
                        name=rec["name"] or "",
                        anchor_type=rec["anchor_type"] or "Other",
                        cosine=float(rec["sim"] or 0.0),
                        degree=int(rec["degree"] or 0),
                        predicates=[str(p) for p in (rec["predicates"] or []) if p],
                    )
                    async for rec in result
                ]
                # Enough survivors, or the index is exhausted so escalating cannot help.
                if len(hits) >= candidate_k or fetch_k >= ANCHOR_OVERFETCH_CAP:
                    break
        if is_merge_policy_version():
            _pol = merge_policy_module()
            rank_anchor_hits = getattr(_pol, "rank_anchor_hits")

            hits = rank_anchor_hits(query, hits, top_k)
        elif settings.graph_version == "v7.21":
            from mirix.services.retrieval_policy_v721 import rank_anchor_hits

            hits = rank_anchor_hits(query, hits, top_k)
        elif settings.graph_version == "v7.20":
            from mirix.services.retrieval_policy_v720 import rank_anchor_hits

            hits = rank_anchor_hits(query, hits, top_k)
        else:
            hits = hits[:top_k]
        if len(hits) < top_k:
            logger.info("v7 anchor search: %d/%d after user filter (store may be "
                        "smaller than top_k)", len(hits), top_k)
        return hits

    async def _search_exact_anchors(
        self, driver, user_id: str, query: str, *, limit: int = 16,
    ) -> list[V7AnchorHit]:
        """Indexed exact query-name lookups used by v7.21+ relation lanes."""

        if is_merge_policy_version():
            _pol = merge_policy_module()
            exact_anchor_terms = getattr(_pol, "exact_anchor_terms")
        else:
            from mirix.services.retrieval_policy_v721 import exact_anchor_terms

        names = exact_anchor_terms(query)
        if not names:
            return []
        cypher = """
        UNWIND $names AS nl
        MATCH (a:V7Anchor {user_id: $user_id, name_lower: nl})
        OPTIONAL MATCH (a)<-[:V7_FACT_ARG]-(f:V7Fact)
        WITH a, count(DISTINCT f) AS degree,
             collect(DISTINCT f.predicate)[..24] AS predicates
        RETURN a.id AS id, a.name AS name, a.anchor_type AS anchor_type,
               degree AS degree, predicates AS predicates
        ORDER BY CASE WHEN toLower(a.anchor_type) = 'person' THEN 0 ELSE 1 END,
                 size(a.name_lower) DESC
        LIMIT $limit
        """
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                cypher, names=names, user_id=user_id, limit=limit,
            )
            return [
                V7AnchorHit(
                    id=rec["id"],
                    name=rec["name"] or "",
                    anchor_type=rec["anchor_type"] or "Other",
                    cosine=1.0,
                    degree=int(rec["degree"] or 0),
                    predicates=[str(p) for p in (rec["predicates"] or []) if p],
                )
                async for rec in result
            ]

    async def _collect_relation_facts(
        self,
        driver,
        *,
        user_id: str,
        query: str,
        exact_anchors: list[V7AnchorHit],
        limit: int = 8,
    ) -> tuple[list[V7MemoryRow], list[str]]:
        """Return top adjacent facts and the PG citations they prioritize.

        The predicate filter is applied inside Neo4j before the cap, so even an
        explicitly named person hub does not trigger a full-graph traversal.
        """

        if is_merge_policy_version():
            _pol = merge_policy_module()
            predicate_hints = getattr(_pol, "predicate_hints")
            query_mentions_anchor = getattr(_pol, "query_mentions_anchor")
            rank_relation_facts = getattr(_pol, "rank_relation_facts")
        else:
            from mirix.services.retrieval_policy_v721 import (
                predicate_hints,
                query_mentions_anchor,
                rank_relation_facts,
            )

        hints = predicate_hints(query)
        if not exact_anchors or not hints:
            return [], []
        anchor_ids = [hit.id for hit in exact_anchors]
        cypher = """
        UNWIND $anchor_ids AS aid
        MATCH (seed:V7Anchor {id: aid, user_id: $user_id})
              <-[:V7_FACT_ARG]-(f:V7Fact)
        WHERE any(token IN split(
                    replace(replace(toLower(f.predicate), '_', ' '), '-', ' '), ' ')
                  WHERE token IN $predicate_hints)
        WITH DISTINCT f
        // Deterministic truncation. This LIMIT had no ORDER BY, so WHICH facts
        // survived the cap depended on Neo4j's traversal order — meaning any change
        // that raises an anchor's degree (entity consolidation, in particular) would
        // silently reshuffle the surviving set and any measured effect would be
        // partly arbitrary truncation. f.id is a content hash, so this is
        // deterministic without pretending to be a relevance ranking; choosing a
        // MEANINGFUL key (citation count, recency) is a separate change that has to
        // be measured on its own.
        ORDER BY f.id
        LIMIT $fact_cap
        MATCH (f)-[r:V7_FACT_ARG]->(arg:V7Anchor)
        WITH f, collect({id: arg.id, name: arg.name,
                         anchor_type: arg.anchor_type, role: r.role}) AS args
        RETURN f.id AS id, f.predicate AS predicate, args AS args,
               f.memory_ids AS memory_ids, f.timestamp AS timestamp,
               properties(f)['mentioned_at'] AS mentioned_at,
               f.lit_keys AS lit_keys, f.lit_vals AS lit_vals
        """
        candidates: list[V7FactCandidate] = []
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                cypher,
                anchor_ids=anchor_ids,
                user_id=user_id,
                predicate_hints=hints,
                fact_cap=240,
            )
            async for rec in result:
                candidates.append(V7FactCandidate(
                    id=str(rec["id"]),
                    predicate=str(rec["predicate"] or "related to"),
                    args=[dict(arg) for arg in (rec["args"] or [])],
                    memory_ids=[str(mid) for mid in (rec["memory_ids"] or []) if mid],
                    timestamp=str(rec["timestamp"]) if rec["timestamp"] else None,
                    mentioned_at=(
                        str(rec["mentioned_at"]) if rec["mentioned_at"] else None
                    ),
                    lit_keys=[str(value) for value in (rec["lit_keys"] or [])],
                    lit_vals=[str(value) for value in (rec["lit_vals"] or [])],
                ))

        # If the question explicitly names a concrete non-person anchor, keep
        # this lane on that object instead of filling synonym-reserve slots from
        # the named person's entire hub. Generic answer categories stay open so
        # list questions such as "what books/items" can discover their members.
        generic_categories = {
            "activity", "activities", "book", "books", "event", "events",
            "instrument", "instruments", "item", "items", "painting",
            "paintings", "place", "places", "plan", "plans", "subject",
            "subjects", "thing", "things", "type", "types",
        }
        required_context_ids = {
            hit.id for hit in exact_anchors
            if hit.anchor_type.lower() != "person"
            and hit.name.strip().lower() not in generic_categories
            and query_mentions_anchor(query, hit.name)
        }
        if required_context_ids:
            candidates = [
                fact for fact in candidates
                if any(str(arg.get("id")) in required_context_ids for arg in fact.args)
            ]

        rows: list[V7MemoryRow] = []
        memory_ids: list[str] = []
        chain_rows = self._relation_chain_rows(
            query, candidates, exact_anchor_ids=set(anchor_ids),
        )
        ranked = rank_relation_facts(
            query,
            candidates,
            exact_anchor_ids=anchor_ids,
            limit=max(0, limit - len(chain_rows)),
        )
        rows.extend(chain_rows)
        for chain in chain_rows:
            _add_unique(memory_ids, (chain.extra or {}).get("citation_memory_ids", []))
        for fact in ranked:
            rendered_args = "; ".join(
                f"{arg.get('role') or 'participant'}={arg.get('name') or '?'}"
                for arg in fact.args
            )
            literals = "; ".join(
                f"{key}={value}" for key, value in zip(fact.lit_keys, fact.lit_vals)
            )
            body = "; ".join(part for part in (rendered_args, literals) if part)
            summary = f"Graph fact: {fact.predicate}({body})."
            rows.append(V7MemoryRow(
                id=fact.id,
                kind="graph_fact",
                summary=summary,
                details="",
                timestamp=fact.timestamp,
                extra={
                    "name": fact.predicate,
                    "source": "graph_fact",
                    "citation_memory_ids": list(fact.memory_ids),
                    "v721_fact_score": fact.policy_score,
                    "v722_fact_score": (
                        fact.policy_score
                        if is_merge_policy_version()
                        else None
                    ),
                    "v724_role_score": getattr(fact, "v724_role_score", None),
                    "v724_temporal_state_score": getattr(
                        fact, "v724_temporal_state_score", None
                    ),
                    "mentioned_at": fact.mentioned_at,
                },
            ))
            _add_unique(memory_ids, fact.memory_ids)
        logger.info(
            "%s relation lane: %d exact anchors, %d adjacent candidates, "
            "%d ranked facts, %d citations",
            settings.graph_version,
            len(exact_anchors), len(candidates), len(rows), len(memory_ids),
        )
        return rows, memory_ids

    @staticmethod
    def _relation_chain_rows(
        query: str,
        facts: list[V7FactCandidate],
        *,
        exact_anchor_ids: set[str],
    ) -> list[V7MemoryRow]:
        """Resolve one conservative typed recommendation chain.

        Memory conversations commonly split ``X read that book`` and ``Y
        recommended TITLE`` across sessions.  The graph may preserve the first as
        read(X, Books) and the second as recommend(Y, TITLE).  When the question
        explicitly asks for an item *from a recommendation*, two distinct named
        actors are present, and Y has exactly one specific recommended item of the
        requested type, surface that composition as an inference candidate.  If
        there is more than one title, emit nothing rather than guess.
        """

        low = (query or "").lower()
        if not re.search(r"\b(?:recommend|recommendation|suggest|suggestion|advise)\w*\b", low):
            return []
        if not re.search(r"\b(?:read|watch|listen|try|use)\w*\b", low):
            return []
        answer_match = re.search(
            r"\bwhat\s+(book|movie|film|song|album|show|product|item|tool|app)\b",
            low,
        )
        if not answer_match:
            return []
        answer_type = answer_match.group(1)
        if answer_type == "film":
            answer_type = "movie"
        type_compatibility = {
            "book": {"content", "object", "publication", "work"},
            "movie": {"content", "film", "object", "work"},
            "song": {"content", "music", "object", "work"},
            "album": {"content", "music", "object", "work"},
            "show": {"content", "event", "object", "work"},
            "app": {"content", "object", "software", "tool"},
            "tool": {"object", "product", "software", "tool"},
            "product": {"content", "object", "product"},
            "item": {"content", "object", "product", "work"},
        }

        def norm(value: object) -> str:
            tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
            return " ".join(tokens)

        def is_generic(value: object) -> bool:
            name = norm(value)
            singular = name[:-1] if name.endswith("s") else name
            return singular in {answer_type, "item", "thing", "content"}

        object_roles = {"theme", "patient", "content", "object", "item"}
        actor_roles = {
            "agent", "actor", "reader", "viewer", "listener", "user",
            "recommender", "subject", "experiencer",
        }
        action_facts = []
        recommend_facts = []
        for fact in facts:
            pred = norm(fact.predicate)
            if any(word in pred.split() for word in ("read", "watch", "listen", "try", "use")):
                action_facts.append(fact)
            if any(word in pred.split() for word in ("recommend", "suggest", "advise")):
                recommend_facts.append(fact)

        bridges = []
        for fact in action_facts:
            actors = {
                str(arg.get("id")) for arg in fact.args
                if str(arg.get("id")) in exact_anchor_ids
                and str(arg.get("role") or "").lower() in actor_roles
            }
            generic_objects = [
                arg for arg in fact.args
                if str(arg.get("role") or "").lower() in object_roles
                and is_generic(arg.get("name"))
            ]
            if actors and generic_objects:
                bridges.append((fact, actors, generic_objects[0]))
        if not bridges:
            return []

        specifics: dict[str, tuple[V7FactCandidate, dict, set[str]]] = {}
        for fact in recommend_facts:
            actors = {
                str(arg.get("id")) for arg in fact.args
                if str(arg.get("id")) in exact_anchor_ids
                and str(arg.get("role") or "").lower() in actor_roles
            }
            for arg in fact.args:
                role = str(arg.get("role") or "").lower()
                name = str(arg.get("name") or "").strip()
                if role not in object_roles or not name or is_generic(name):
                    continue
                anchor_type = str(arg.get("anchor_type") or "").lower()
                allowed_types = type_compatibility.get(answer_type, set())
                if allowed_types and anchor_type and anchor_type not in allowed_types:
                    continue
                specifics[norm(name)] = (fact, arg, actors)
        if len(specifics) != 1:
            return []

        rec_fact, specific, rec_actors = next(iter(specifics.values()))
        bridge_fact, action_actors, generic = bridges[0]
        if not rec_actors or not action_actors or not (rec_actors - action_actors):
            return []

        actor_names = {
            str(arg.get("id")): str(arg.get("name") or "")
            for fact in (bridge_fact, rec_fact) for arg in fact.args
            if str(arg.get("id")) in exact_anchor_ids
        }
        reader = ", ".join(actor_names.get(value, value) for value in action_actors)
        recommender = ", ".join(actor_names.get(value, value) for value in rec_actors)
        title = str(specific.get("name") or "")
        citations: list[str] = []
        _add_unique(citations, bridge_fact.memory_ids)
        _add_unique(citations, rec_fact.memory_ids)
        return [V7MemoryRow(
            id=f"v721chain:{bridge_fact.id}:{rec_fact.id}",
            kind="graph_fact",
            summary=(
                "Graph resolved relation (unique typed chain): "
                f"{reader} {bridge_fact.predicate} a {generic.get('name')}; "
                f"{recommender} {rec_fact.predicate} {title}; "
                f"therefore requested {answer_type}={title}."
            ),
            details="Unique cited typed composition for the requested recommendation chain.",
            timestamp=rec_fact.timestamp or bridge_fact.timestamp,
            extra={
                "name": "relation chain",
                "source": "graph_relation_inference",
                "citation_memory_ids": citations,
                "v721_chain": True,
            },
        )]

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

    async def _collect_ids_from_anchors(
        self, driver, *, user_id: str, anchor_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """v7.12: the anchor carries its PG row ids, so this is a property read.

        The v7.10 equivalent (_collect_memory_refs) traversed anchor -> ref node to
        learn the same thing. The ref node held a copy of the PG id and nothing else
        PG did not already have.
        """
        if not anchor_ids:
            return [], []
        cypher = """
        UNWIND $anchor_ids AS aid
        MATCH (a:V7Anchor {id: aid, user_id: $user_id})
        RETURN a.episodic_ids AS ep, a.semantic_ids AS sem
        """
        ep_ids: list[str] = []
        sem_ids: list[str] = []
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, anchor_ids=anchor_ids, user_id=user_id)
            async for rec in result:
                _add_unique(ep_ids, rec["ep"])
                _add_unique(sem_ids, rec["sem"])
        return ep_ids, sem_ids

    async def _expand_via_pg(
        self, user_id: str, ep_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """Recompute in PG what V7_SUPPORTED_BY and V7_NEXT_MEMORY used to materialise.

        * same-chunk: rows sharing a ``source_refs[].chunk_id`` with a seed episode —
          this is exactly what V7_SUPPORTED_BY encoded, and what made it 337 edges out
          of 12 chunks (a near-clique per chunk) rather than 337 distinct facts.
        * temporal neighbour: the episode immediately before and after each seed by
          ``occurred_at`` — V7_NEXT_MEMORY was a materialised ORDER BY.

        Kept so that dropping the ref layer does not silently change recall: the
        candidate set is the same, it is just derived where the data actually lives.
        """
        if not ep_ids:
            return [], []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        seeds = ep_ids[:PG_EXPAND_SEEDS]
        same_ep: list[str] = []
        same_sem: list[str] = []
        try:
            async with db_context() as session:
                chunks = (await session.execute(sa_text(
                    "SELECT DISTINCT jsonb_array_elements(source_refs::jsonb)->>'chunk_id' AS c "
                    "FROM episodic_memory WHERE user_id = :u AND id = ANY(:ids)"
                ), {"u": user_id, "ids": seeds})).fetchall()
                chunk_ids = [r[0] for r in chunks if r[0] is not None]
                # Degeneracy guard. If a store ever writes one chunk id on every
                # row, this hop would match the entire store and inject an arbitrary
                # LIMIT-worth of memories as if they were evidence. Measured on
                # conv-26 the ids are properly per-session (3/3/5/7/4/17 rows), so
                # this normally does nothing — it exists so a badly-populated
                # source_refs degrades to "no expansion" rather than "random 60".
                if chunk_ids:
                    row = (await session.execute(sa_text(
                        "SELECT count(*) FILTER (WHERE EXISTS ("
                        "  SELECT 1 FROM jsonb_array_elements(source_refs::jsonb) e"
                        "  WHERE e->>'chunk_id' = ANY(:c))) AS matched, count(*) AS tot "
                        "FROM episodic_memory WHERE user_id = :u AND NOT is_deleted"
                    ), {"u": user_id, "c": chunk_ids})).fetchone()
                    if row and row[1] and row[0] / row[1] > PG_EXPAND_MAX_CHUNK_SHARE:
                        logger.info("v7.12 same-chunk hop skipped: chunk ids match "
                                    "%d/%d rows, not selective", row[0], row[1])
                        chunk_ids = []
                if chunk_ids:
                    rows = (await session.execute(sa_text(
                        "SELECT id FROM episodic_memory WHERE user_id = :u AND NOT is_deleted "
                        "AND EXISTS (SELECT 1 FROM jsonb_array_elements(source_refs::jsonb) e "
                        "            WHERE e->>'chunk_id' = ANY(:c)) LIMIT :k"
                    ), {"u": user_id, "c": chunk_ids, "k": PG_EXPAND_LIMIT})).fetchall()
                    same_ep = [r[0] for r in rows]
                    # semantic_memory.source_refs is EMPTY on every row ever written
                    # (0/74 on the live conv-26 store, 0/672 on lm60_v2g):
                    # insert_semantic_item never populates it, and the one path that
                    # does — upsert_with_conflict_resolution — is gated on " / " being
                    # in the name, which these names never contain. The provenance is
                    # there, just under filter_tags->source_meta (74/74 and 545/672).
                    # Reading source_refs here made this branch return nothing, always.
                    rows = (await session.execute(sa_text(
                        "SELECT id FROM semantic_memory WHERE user_id = :u AND NOT is_deleted "
                        "AND filter_tags->'source_meta'->>'chunk_id' = ANY(:c) LIMIT :k"
                    ), {"u": user_id, "c": chunk_ids, "k": PG_EXPAND_LIMIT})).fetchall()
                    same_sem = [r[0] for r in rows]
                # Exactly the memory before and the memory after each seed. The
                # V7_NEXT_MEMORY hop this replaces was matched undirected on a
                # linked list, so it yielded precisely those two. A time WINDOW is
                # not the same thing: LoCoMo's episodes cluster into a handful of
                # session days months apart, so any window wide enough to catch a
                # neighbour also drags in that entire session.
                neigh = (await session.execute(sa_text(
                    "WITH seeds AS (SELECT occurred_at FROM episodic_memory "
                    "               WHERE user_id = :u AND id = ANY(:ids) "
                    "                 AND occurred_at IS NOT NULL) "
                    "SELECT DISTINCT n.id FROM seeds s CROSS JOIN LATERAL ( "
                    "  (SELECT m.id, m.occurred_at FROM episodic_memory m "
                    "     WHERE m.user_id = :u AND NOT m.is_deleted "
                    "       AND m.occurred_at < s.occurred_at "
                    "     ORDER BY m.occurred_at DESC LIMIT 1) "
                    "  UNION ALL "
                    "  (SELECT m.id, m.occurred_at FROM episodic_memory m "
                    "     WHERE m.user_id = :u AND NOT m.is_deleted "
                    "       AND m.occurred_at > s.occurred_at "
                    "     ORDER BY m.occurred_at ASC LIMIT 1) "
                    ") n LIMIT :k"
                ), {"u": user_id, "ids": seeds, "k": PG_EXPAND_LIMIT})).fetchall()
                _add_unique(same_ep, [r[0] for r in neigh])
        except Exception as exc:  # noqa: BLE001
            logger.warning("v7.12 PG expansion failed: %s", exc)
            return [], []
        return same_ep, same_sem

    async def _collect_frame_refs(
        self, driver, *, user_id: str, anchor_ids: list[str]
    ) -> tuple[list[str], list[str]]:
        """v7.12: expand a hit anchor through the FRAMES it participates in.

        This is the read side the fact layer never had. v7.10 wrote V7Fact nodes and
        then never traversed them — every retrieval path went anchor -> memory ref
        directly — so reifying the triples bought nothing at query time. Two hops are
        worth different amounts and are returned separately:

        * **frame-cited memories** — the memories that actually assert a frame this
          anchor takes part in. High precision: the anchor is a named participant.
        * **co-participant memories** — memories about the OTHER args of those frames.
          This is the n-ary payoff: asking about Boston reaches the Delta flight's
          companion and destination through one shared frame, a link a binary store can
          only reconstruct by joining several triples through the trip's subject.

        Both hops are capped; a popular anchor can sit in hundreds of frames and the
        co-participant fan-out is quadratic in arity.
        """
        if not anchor_ids:
            return [], []
        cypher = """
        UNWIND $anchor_ids AS aid
        MATCH (a:V7Anchor {id: aid, user_id: $user_id})<-[:V7_FACT_ARG]-(f:V7Fact)
        WITH DISTINCT f
        ORDER BY f.id          // deterministic truncation, see _fact_rows above
        LIMIT $fact_cap
        OPTIONAL MATCH (f)-[:V7_FACT_ARG]->(co:V7Anchor)
            WHERE NOT co.id IN $anchor_ids
        WITH collect(DISTINCT f) AS fs,
             collect(DISTINCT coalesce(co.episodic_ids, [])) AS co_ep_lists,
             collect(DISTINCT coalesce(co.semantic_ids, [])) AS co_sem_lists
        RETURN
            reduce(acc = [], x IN fs | acc + coalesce(x.memory_ids, []))[..$mem_cap]
                AS frame_mem,
            reduce(acc = [], x IN co_ep_lists | acc + x)[..$mem_cap]  AS co_ep,
            reduce(acc = [], x IN co_sem_lists | acc + x)[..$mem_cap] AS co_sem
        """
        # NB: the co-participant lists are collected as coalesce(...,[]) rather than as
        # nodes. OPTIONAL MATCH yields null when a frame has no other args, and
        # collect(DISTINCT <null node>) makes Neo4j emit an AggregationSkippedNull
        # warning on EVERY retrieval — hundreds of lines per eval run. Collecting the
        # id lists with an empty-list default has no null to skip.
        ep_ids: list[str] = []
        sem_ids: list[str] = []
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                cypher, anchor_ids=anchor_ids, user_id=user_id,
                fact_cap=FRAME_FACT_CAP, mem_cap=FRAME_MEMORY_CAP)
            async for rec in result:
                # A frame's citations are PG ids without a kind tag (the ref node used
                # to carry the label). Both fetchers filter by user_id and id, so an
                # id offered to the wrong table simply misses — cheaper than storing
                # the kind twice.
                _add_unique(ep_ids, rec["frame_mem"])
                _add_unique(sem_ids, rec["frame_mem"])
                # co-participants after frame citations: order encodes precision for
                # the unranked path (the reranker reorders by text cosine regardless).
                _add_unique(ep_ids, rec["co_ep"])
                _add_unique(sem_ids, rec["co_sem"])
        return ep_ids, sem_ids

    # (v7.7 PPR and v7.2 coverage retrieval helpers — _load_ppr_graph, _retrieve_ppr,
    #  _collect_per_anchor, _retrieve_coverage, _round_robin — are archived; see
    #  docs/graph_memory_v7/development_history.md.)

    async def _fetch_episodic(
        self, user_id: str, ids: list[str],
        q_emb: Optional[list[float]] = None, limit: Optional[int] = None,
        priority_ids: Optional[list[str]] = None,
    ) -> list[V7MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        if q_emb is not None:  # v7.1: rank candidates by query full-text similarity
            from mirix.constants import MAX_EMBEDDING_DIM
            qp = str(list(q_emb) + [0.0] * (MAX_EMBEDDING_DIM - len(q_emb)))
            sql = (
                "SELECT id, summary, details, occurred_at, actor, source_refs "
                "FROM episodic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) AND summary_embedding IS NOT NULL "
                "ORDER BY CASE WHEN id = ANY(:priority) THEN 0 ELSE 1 END, "
                "summary_embedding <=> CAST(:q AS vector) LIMIT :k"
            )
            params = {
                "u": user_id, "ids": ids, "q": qp,
                "priority": list(priority_ids or []),
                "k": limit or DEFAULT_MAX_ITEMS_PER_KIND,
            }
        else:
            sql = (
                "SELECT id, summary, details, occurred_at, actor, source_refs "
                "FROM episodic_memory "
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
                    extra={"actor": row[4] or "", **_source_provenance(row[5])},
                )
                for row in result.fetchall()
            ]

    async def _fetch_semantic(
        self, user_id: str, ids: list[str],
        q_emb: Optional[list[float]] = None, limit: Optional[int] = None,
        priority_ids: Optional[list[str]] = None,
    ) -> list[V7MemoryRow]:
        if not ids:
            return []
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        if q_emb is not None:  # v7.1: rank candidates by query full-text similarity
            from mirix.constants import MAX_EMBEDDING_DIM
            qp = str(list(q_emb) + [0.0] * (MAX_EMBEDDING_DIM - len(q_emb)))
            sql = (
                "SELECT id, name, summary, details, source, created_at, source_refs "
                "FROM semantic_memory "
                "WHERE user_id = :u AND id = ANY(:ids) AND summary_embedding IS NOT NULL "
                "ORDER BY CASE WHEN id = ANY(:priority) THEN 0 ELSE 1 END, "
                "summary_embedding <=> CAST(:q AS vector) LIMIT :k"
            )
            params = {
                "u": user_id, "ids": ids, "q": qp,
                "priority": list(priority_ids or []),
                "k": limit or DEFAULT_MAX_ITEMS_PER_KIND,
            }
        else:
            sql = (
                "SELECT id, name, summary, details, source, created_at, source_refs "
                "FROM semantic_memory "
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
                    extra={
                        "name": row[1] or "",
                        "source": row[4] or "",
                        **_source_provenance(row[6]),
                    },
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

        fact_rows = [row for row in sem_rows if row.kind == "graph_fact"]
        semantic_rows = [row for row in sem_rows if row.kind != "graph_fact"]

        if fact_rows:
            lines.append("\n### Relation-matched graph facts")
            for row in fact_rows:
                ts = row.timestamp[:10] if row.timestamp else ""
                head = f"- [{ts}] {row.summary}" if ts else f"- {row.summary}"
                lines.append(head.rstrip())

        if semantic_rows:
            lines.append("\n### Semantic memories (PG flat)")
            for row in semantic_rows:
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
