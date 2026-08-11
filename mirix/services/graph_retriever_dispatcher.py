"""
Top-level graph-retrieval dispatcher.

Entry point from rest_api.retrieve_memories_by_keywords. Routes to the retriever
for the configured ``graph_version`` and returns its markdown context string.

Only the v7 family (v7, v7.x, v8) is live. The earlier generations this file
used to fan out to — the v5 dual-graph pipeline (keyword-LLM call +
EpisodicRetriever/SemanticRetriever over v5 node labels) and the v6 lean entity
index — are archived under ``archive/legacy_graph/``; an unrecognised
``graph_version`` now returns empty context instead of silently running a
pipeline whose node labels no ingest path writes anymore.

Returns an empty string when graph memory is disabled, when the version is not
recognised, or when there are no hits. Callers treat empty as "no graph context".
"""

from __future__ import annotations

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.settings import settings

logger = get_logger(__name__)



class GraphRetrieverDispatcher:
    """Stateless. Create one per request."""

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        item_top_k: int = 15,
    ) -> str:
        if not settings.enable_graph_memory:
            return ""

        if settings.graph_version.startswith("v7") or settings.graph_version == "v8":
            from mirix.services.graph_retriever_v7 import V7Retriever

            return await V7Retriever().retrieve(
                query=query, user_id=user_id, agent_state=agent_state,
                top_k=item_top_k,
            )

        logger.warning(
            "Graph retrieve: graph_version=%r has no live retriever (v5/v6 are "
            "archived); returning empty context", settings.graph_version,
        )
        return ""

    async def retrieve_rows(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        item_top_k: int = 15,
        max_items_per_kind: int = 15,
    ) -> list:
        """Same retrieval as ``retrieve`` but as STRUCTURED rows, flattened.

        The blob form of graph context proved easy for an answerer to ignore. This is
        the form /search serves as the answerer's graph-owned tool results, which is
        the channel that actually reaches an answer.
        """
        if not settings.enable_graph_memory:
            return []
        if not (settings.graph_version.startswith("v7") or settings.graph_version == "v8"):
            return []

        from mirix.services.graph_retriever_v7 import V7Retriever

        _anchors, ep_rows, sem_rows = await V7Retriever().retrieve_rows(
            query=query, user_id=user_id, agent_state=agent_state,
            top_k=item_top_k, max_items_per_kind=max_items_per_kind,
        )
        return list(ep_rows) + list(sem_rows)
