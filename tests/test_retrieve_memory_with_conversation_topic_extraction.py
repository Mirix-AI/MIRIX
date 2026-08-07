"""Unit tests for retrieve_memory_with_conversation's topic-extraction
call-site fix (ECMS-522 task 7, R4.1 / R4.4 / R3.2).

Pins the call-arg shape: the route must fetch the client's FULL agent
roster (no `limit=1`) and delegate the topic_extraction_agent lookup/miss
case to the shared get_or_create_topic_extraction_agent helper -- not the
old unconditional `all_agents[0].llm_config` grab.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig


def _make_client(client_id="client-1"):
    return Client(
        id=client_id,
        organization_id="org-1",
        name="Test Client",
        status="active",
    )


def _make_llm_config(model="gpt-4o-mini"):
    return LLMConfig(
        model=model,
        model_endpoint_type="openai",
        context_window=8192,
    )


def _make_agent_state(agent_id, agent_type, llm_config):
    return AgentState(
        id=agent_id,
        name=f"{agent_id}-name",
        system="",
        agent_type=agent_type,
        llm_config=llm_config,
        embedding_config=None,
        tools=[],
        organization_id="org-1",
    )


def _mock_server_with_agents(client, all_agents):
    mock_server = MagicMock()
    mock_server.client_manager.get_client_by_id = AsyncMock(return_value=client)
    mock_server.agent_manager.list_agents = AsyncMock(return_value=all_agents)
    mock_server.agent_manager.create_agent = AsyncMock()
    mock_server.agent_manager.get_agent_by_id = AsyncMock()
    return mock_server


class _EmptyMessagesRequest:
    """Minimal stand-in for RetrieveMemoryRequest with no message content --
    keeps the route on the has_content=False branch so it never calls the
    real LLM, letting this test isolate the llm_config-resolution call shape.
    """

    user_id = "user-1"
    messages = []
    limit = 10
    local_model_for_retrieval = None
    filter_tags = None
    use_cache = True
    start_date = None
    end_date = None
    include_citations = False


@pytest.mark.asyncio
async def test_fetches_full_roster_with_no_limit_and_include_tools_false():
    from mirix.server.rest_api import retrieve_memory_with_conversation

    client = _make_client()
    all_agents = [_make_agent_state("agent-core", AgentType.core_memory_agent, _make_llm_config())]
    mock_server = _mock_server_with_agents(client, all_agents)

    with (
        patch("mirix.server.rest_api.get_server", return_value=mock_server),
        patch("mirix.server.rest_api.retrieve_memories_by_keywords", new=AsyncMock(return_value={})),
    ):
        await retrieve_memory_with_conversation(
            request=_EmptyMessagesRequest(),
            x_client_id=client.id,
            x_org_id="org-1",
        )

    mock_server.agent_manager.list_agents.assert_awaited_once()
    call_kwargs = mock_server.agent_manager.list_agents.await_args.kwargs
    assert "limit" not in call_kwargs
    assert call_kwargs["include_tools"] is False
    assert call_kwargs["actor"] is client


@pytest.mark.asyncio
async def test_uses_existing_topic_extraction_agent_row_when_present():
    from mirix.server.rest_api import retrieve_memory_with_conversation

    client = _make_client()
    topic_llm_config = _make_llm_config(model="gpt-4o-mini-topic")
    all_agents = [
        _make_agent_state("agent-core", AgentType.core_memory_agent, _make_llm_config("gpt-4o")),
        _make_agent_state("agent-topic", AgentType.topic_extraction_agent, topic_llm_config),
    ]
    mock_server = _mock_server_with_agents(client, all_agents)

    with (
        patch("mirix.server.rest_api.get_server", return_value=mock_server),
        patch("mirix.server.rest_api.retrieve_memories_by_keywords", new=AsyncMock(return_value={})),
    ):
        await retrieve_memory_with_conversation(
            request=_EmptyMessagesRequest(),
            x_client_id=client.id,
            x_org_id="org-1",
        )

    # No agents[0] fallback identity leaks through the shared helper when the
    # topic_extraction_agent row exists -- and no create is attempted.
    mock_server.agent_manager.create_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_lazily_creates_row_on_miss_with_no_fallback_override():
    """R3.2/R4.4: on a miss, the retrieve path delegates to the shared helper
    with NO fallback_llm_config override -- keeping its existing
    all_agents[0] fallback identity unchanged.
    """
    from mirix.server.rest_api import retrieve_memory_with_conversation

    client = _make_client()
    seed_llm_config = _make_llm_config(model="gpt-4o-agent-0")
    all_agents = [_make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config)]
    mock_server = _mock_server_with_agents(client, all_agents)
    created_state = _make_agent_state("agent-topic-new", AgentType.topic_extraction_agent, seed_llm_config)
    mock_server.agent_manager.create_agent = AsyncMock(return_value=created_state)

    with (
        patch("mirix.server.rest_api.get_server", return_value=mock_server),
        patch("mirix.server.rest_api.retrieve_memories_by_keywords", new=AsyncMock(return_value={})),
    ):
        await retrieve_memory_with_conversation(
            request=_EmptyMessagesRequest(),
            x_client_id=client.id,
            x_org_id="org-1",
        )

    mock_server.agent_manager.create_agent.assert_awaited_once()
    agent_create = mock_server.agent_manager.create_agent.await_args.kwargs["agent_create"]
    assert agent_create.agent_type == AgentType.topic_extraction_agent
    assert agent_create.llm_config.model == "gpt-4o-agent-0"


@pytest.mark.asyncio
async def test_no_agents_still_returns_existing_no_agents_response():
    """The existing zero-agents guard ("No agents found for this user") is
    unchanged -- it must fire before any topic-extraction logic runs.
    """
    from mirix.server.rest_api import retrieve_memory_with_conversation

    client = _make_client()
    mock_server = _mock_server_with_agents(client, all_agents=[])

    with patch("mirix.server.rest_api.get_server", return_value=mock_server):
        result = await retrieve_memory_with_conversation(
            request=_EmptyMessagesRequest(),
            x_client_id=client.id,
            x_org_id="org-1",
        )

    assert result == {
        "success": False,
        "error": "No agents found for this user",
        "topics": None,
        "memories": {},
    }
    mock_server.agent_manager.create_agent.assert_not_awaited()
