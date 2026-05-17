"""
Top-level dispatcher that runs both graph retrievers in parallel.

Entry point from rest_api.retrieve_memories_by_keywords. Owns:
- keyword extraction (1 LLM call, cached, shared between graphs)
- batch embed [ll_kw, hl_kw] (1 API call)
- parallel dispatch to EpisodicRetriever + SemanticRetriever
- token-budget split (50/50 between graphs)
- combined markdown formatting

Returns an empty string when graph memory is disabled, when Neo4j is down,
or when no hits across either graph. Callers treat empty as "no graph context".
"""

from __future__ import annotations

import asyncio
from typing import Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import embed_batch, llm_model_from_agent
from mirix.services._graph_retriever_base import (
    GraphSearchResult,
    apply_budget_to_search,
    fmt_date,
)
from mirix.services.episodic_graph_retriever import EpisodicRetriever
from mirix.services.lightrag_keyword_extractor import extract_keywords
from mirix.services.semantic_graph_retriever import SemanticRetriever
from mirix.settings import settings

logger = get_logger(__name__)


# Total token budget across both graphs (split 50/50 per Q2 decision).
DEFAULT_MAX_TOTAL_TOKENS = 12000


class GraphRetrieverDispatcher:
    """Stateless. Create one per request."""

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        top_k: int = 30,
        item_top_k: int = 15,
    ) -> str:
        """Full v4 retrieval. Returns markdown context string."""
        if not settings.enable_graph_memory:
            return ""

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return ""

        # ─── Step 1: keyword extraction (1 LLM call, cached) ───────────────
        llm_model = llm_model_from_agent(agent_state)
        kw = await extract_keywords(query or "", user_id=user_id, llm_model=llm_model)

        # ─── Step 2: batch embed [ll, hl] ──────────────────────────────────
        ll_str = ", ".join(kw.low_level) if kw.low_level else ""
        hl_str = ", ".join(kw.high_level) if kw.high_level else ""
        texts: list[str] = []
        purposes: list[str] = []
        if ll_str:
            texts.append(ll_str); purposes.append("ll")
        if hl_str:
            texts.append(hl_str); purposes.append("hl")

        emb_by_purpose: dict[str, Optional[list[float]]] = {"ll": None, "hl": None}
        if texts:
            embeddings = await embed_batch(texts, agent_state)
            for p, e in zip(purposes, embeddings):
                emb_by_purpose[p] = e

        ll_emb = emb_by_purpose["ll"]
        hl_emb = emb_by_purpose["hl"]

        if ll_emb is None and hl_emb is None:
            logger.info("Graph retrieve: no embeddings → empty context")
            return ""

        # ─── Step 3: dispatch both retrievers in parallel ──────────────────
        ep_task = asyncio.create_task(
            EpisodicRetriever().retrieve(
                driver=driver, user_id=user_id,
                ll_embedding=ll_emb, hl_embedding=hl_emb,
                top_k=top_k, item_top_k=item_top_k,
            )
        )
        sem_task = asyncio.create_task(
            SemanticRetriever().retrieve(
                driver=driver, user_id=user_id,
                ll_embedding=ll_emb, hl_embedding=hl_emb,
                top_k=top_k, item_top_k=item_top_k,
            )
        )
        ep_result, sem_result = await asyncio.gather(ep_task, sem_task, return_exceptions=True)

        if isinstance(ep_result, Exception):
            logger.warning("Episodic retrieve failed: %s", ep_result)
            ep_result = GraphSearchResult()
        if isinstance(sem_result, Exception):
            logger.warning("Semantic retrieve failed: %s", sem_result)
            sem_result = GraphSearchResult()

        # ─── Step 4: token budget split 50/50, then format ─────────────────
        per_graph_budget = max_total_tokens // 2
        # Within each graph, split: 30% entity, 35% relations, 35% items
        e_budget = int(per_graph_budget * 0.30)
        r_budget = int(per_graph_budget * 0.35)
        i_budget = per_graph_budget - e_budget - r_budget

        ep_trim = apply_budget_to_search(
            ep_result, max_entity_tokens=e_budget,
            max_relation_tokens=r_budget, max_item_tokens=i_budget,
        )
        sem_trim = apply_budget_to_search(
            sem_result, max_entity_tokens=e_budget,
            max_relation_tokens=r_budget, max_item_tokens=i_budget,
        )

        ep_md = _format_section(ep_trim, "Episodic")
        sem_md = _format_section(sem_trim, "Semantic")

        parts = []
        if ep_md:
            parts.append(ep_md)
        if sem_md:
            parts.append(sem_md)
        ctx = "\n\n".join(parts)
        logger.info(
            "Graph retrieve: ep[%dE/%dR/%dI] sem[%dE/%dR/%dI] total %d chars",
            len(ep_trim.entities), len(ep_trim.relations), len(ep_trim.items),
            len(sem_trim.entities), len(sem_trim.relations), len(sem_trim.items),
            len(ctx),
        )
        return ctx


def _format_section(s: GraphSearchResult, title: str) -> str:
    if not (s.entities or s.relations or s.items):
        return ""
    lines = [f"## {title} Knowledge Graph"]
    if s.entities:
        lines.append("### Entities")
        for e in s.entities:
            lines.append(f"- {e.name} ({e.entity_type}, rank={e.rank}): {e.description}")
    if s.relations:
        lines.append("\n### Relationships")
        for r in s.relations:
            validity = f" (on/since {fmt_date(r.valid_at)})" if r.valid_at else ""
            lines.append(
                f"- {r.src_name} <-> {r.tgt_name} [{r.keywords}]: {r.description}{validity}"
            )
    if s.items:
        item_label = "Episodes" if title == "Episodic" else "Concepts"
        lines.append(f"\n### Related {item_label}")
        for it in s.items:
            ts = fmt_date(it.timestamp) if it.timestamp else ""
            ts_part = f"[{ts}] " if ts else ""
            head = f"- {ts_part}{it.summary}".rstrip()
            lines.append(head)
            if it.detail and it.detail != it.summary:
                lines.append(f"  {it.detail[:400]}")
    return "\n".join(lines)
