from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_graph_enabled_all_search_never_calls_flat_ep_or_sem(monkeypatch):
    """Graph-backed kinds must enter through graph traversal, not full-table search."""
    from mirix.server import rest_api
    from mirix.services.graph_retriever_dispatcher import GraphRetrieverDispatcher

    client = SimpleNamespace(id="client", read_scopes=["test"])
    user = SimpleNamespace(id="conv-26", timezone="UTC")
    agent = SimpleNamespace()

    episodic_flat = AsyncMock(side_effect=AssertionError("flat episodic search called"))
    semantic_flat = AsyncMock(side_effect=AssertionError("flat semantic search called"))
    server = SimpleNamespace(
        client_manager=SimpleNamespace(
            get_client_by_id=AsyncMock(return_value=client),
        ),
        agent_manager=SimpleNamespace(
            list_agents=AsyncMock(return_value=[agent]),
        ),
        user_manager=SimpleNamespace(
            get_user_by_id=AsyncMock(return_value=user),
        ),
        episodic_memory_manager=SimpleNamespace(
            list_episodic_memory=episodic_flat,
        ),
        semantic_memory_manager=SimpleNamespace(
            list_semantic_items=semantic_flat,
        ),
        resource_memory_manager=SimpleNamespace(
            list_resources=AsyncMock(return_value=[]),
        ),
        procedural_memory_manager=SimpleNamespace(
            list_procedures=AsyncMock(return_value=[]),
        ),
        knowledge_vault_manager=SimpleNamespace(
            list_knowledge=AsyncMock(return_value=[]),
        ),
    )

    graph_rows = [
        SimpleNamespace(
            kind="episodic",
            id="ep-1",
            timestamp="2023-05-07T00:00:00",
            summary="Caroline attended the support group.",
            details="Graph-selected evidence.",
            extra={"actor": "Caroline"},
        ),
        SimpleNamespace(
            kind="semantic",
            id="sem-1",
            timestamp=None,
            summary="Caroline is active in LGBTQ support.",
            details="Graph-selected evidence.",
            extra={"name": "Caroline", "source": "conversation"},
        ),
    ]

    monkeypatch.setattr(rest_api, "get_server", lambda: server)
    monkeypatch.setattr(rest_api.settings, "enable_graph_memory", True)
    monkeypatch.setattr(
        rest_api,
        "_precompute_embedding_for_search",
        AsyncMock(return_value=([0.1], [0.1])),
    )
    monkeypatch.setattr(
        GraphRetrieverDispatcher,
        "retrieve_rows",
        AsyncMock(return_value=graph_rows),
    )

    result = await rest_api.search_memory.__wrapped__(
        user_id="conv-26",
        query="support group",
        memory_type="all",
        search_field="null",
        search_method="embedding",
        limit=15,
        authorization=None,
        filter_tags=None,
        similarity_threshold=None,
        start_date=None,
        end_date=None,
        include_core_memory=False,
        x_client_id="client",
        x_org_id="org",
    )

    episodic_flat.assert_not_awaited()
    semantic_flat.assert_not_awaited()
    assert [row["id"] for row in result["results"]] == ["ep-1", "sem-1"]
    assert all(row["retrieved_by"] == "graph" for row in result["results"])

