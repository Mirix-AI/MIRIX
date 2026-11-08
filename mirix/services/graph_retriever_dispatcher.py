"""
Top-level graph-retrieval dispatcher.

Entry point from rest_api.retrieve_memories_by_keywords. Runs the hypergraph
retriever when graph memory is on and returns its markdown context string, or an
empty string when graph memory is off or nothing matched. Callers treat empty as
"no graph context".
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

        from mirix.services.graph_retriever_v7 import V7Retriever

        return await V7Retriever().retrieve(
            query=query, user_id=user_id, agent_state=agent_state,
            top_k=item_top_k,
        )

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

        from mirix.services.graph_retriever_v7 import V7Retriever

        _anchors, ep_rows, sem_rows = await V7Retriever().retrieve_rows(
            query=query, user_id=user_id, agent_state=agent_state,
            top_k=item_top_k, max_items_per_kind=max_items_per_kind,
        )
        return list(ep_rows) + list(sem_rows)
