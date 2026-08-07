"""Unit tests for Agent._extract_topics_from_messages's topic-extraction
call-site fix (ECMS-522 task 8, R4.2 / R4.4 / R3.3).

Pins the call-arg shape: the save path (invoked from Agent.step() for
meta_memory_agent) must fetch the client's full agent roster via
self.agent_manager/self.actor and delegate the topic_extraction_agent
lookup/miss case to the shared get_or_create_topic_extraction_agent helper,
passing fallback_llm_config=self.agent_state.llm_config explicitly -- not
the old unconditional self.agent_state.llm_config read, and not the
retrieve path's all_agents[0] fallback identity.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.agent.agent import Agent
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message, MessageRole
from mirix.schemas.mirix_message_content import TextContent


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


def _make_message():
    return Message(
        agent_id="meta-agent-1",
        role=MessageRole.user,
        content=[TextContent(text="hello")],
    )


def _build_extraction_test_agent(client, all_agents, meta_agent_llm_config):
    """Build a minimal Agent instance with only the fields
    _extract_topics_from_messages actually reads: agent_state (for its own
    llm_config and id), agent_manager, and actor."""
    agent = Agent.__new__(Agent)
    agent.agent_state = _make_agent_state("meta-agent-1", AgentType.meta_memory_agent, meta_agent_llm_config)
    agent.agent_manager = SimpleNamespace(
        list_agents=AsyncMock(return_value=all_agents),
        create_agent=AsyncMock(),
        get_agent_by_id=AsyncMock(),
    )
    agent.actor = client
    return agent


def _patch_llm_call(response_topic="topic-a;topic-b"):
    """LLMClient.create's real send_llm_request path talks to a real
    provider -- stub it so these tests isolate llm_config resolution."""
    mock_llm_client = MagicMock()
    mock_llm_client.send_llm_request = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        tool_calls=[
                            MagicMock(
                                function=MagicMock(
                                    name="update_topic",
                                    arguments=f'{{"topic": "{response_topic}"}}',
                                )
                            )
                        ]
                    )
                )
            ]
        )
    )
    return patch("mirix.agent.agent.LLMClient.create", return_value=mock_llm_client)


@pytest.mark.asyncio
async def test_fetches_full_roster_via_self_agent_manager_and_self_actor():
    client = _make_client()
    meta_llm_config = _make_llm_config(model="gpt-4o-meta")
    all_agents = [_make_agent_state("agent-meta", AgentType.meta_memory_agent, meta_llm_config)]
    agent = _build_extraction_test_agent(client, all_agents, meta_llm_config)

    with _patch_llm_call():
        await agent._extract_topics_from_messages([_make_message()])

    agent.agent_manager.list_agents.assert_awaited_once()
    call_kwargs = agent.agent_manager.list_agents.await_args.kwargs
    assert call_kwargs["actor"] is client
    assert call_kwargs["include_tools"] is False
    assert "limit" not in call_kwargs


@pytest.mark.asyncio
async def test_uses_existing_topic_extraction_agent_row_when_present():
    client = _make_client()
    meta_llm_config = _make_llm_config(model="gpt-4o-meta")
    topic_llm_config = _make_llm_config(model="gpt-4o-mini-topic")
    all_agents = [
        _make_agent_state("agent-meta", AgentType.meta_memory_agent, meta_llm_config),
        _make_agent_state("agent-topic", AgentType.topic_extraction_agent, topic_llm_config),
    ]
    agent = _build_extraction_test_agent(client, all_agents, meta_llm_config)

    with _patch_llm_call() as mock_create:
        await agent._extract_topics_from_messages([_make_message()])

    # The topic_extraction_agent row's config is used, not meta_memory_agent's
    # own (agent.agent_state.llm_config) -- proves R4.2's independence bar.
    mock_create.assert_called_once()
    used_llm_config = mock_create.call_args.kwargs["llm_config"]
    assert used_llm_config.model == "gpt-4o-mini-topic"
    agent.agent_manager.create_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_passes_meta_memory_agents_own_config_as_fallback_override():
    """R3.3/R4.4: on a miss, the save path delegates to the shared helper
    WITH fallback_llm_config=self.agent_state.llm_config -- keeping its
    existing, deterministic fallback identity (meta_memory_agent's own
    config), not the retrieve path's all_agents[0] fallback identity.
    """
    client = _make_client()
    meta_llm_config = _make_llm_config(model="gpt-4o-meta-fallback")
    # all_agents[0] is deliberately a DIFFERENT agent/config than
    # meta_memory_agent's own, so a test that leaked the retrieve path's
    # all_agents[0] fallback (rather than the save path's own
    # self.agent_state.llm_config) would be caught here.
    other_llm_config = _make_llm_config(model="gpt-4o-other-agent")
    all_agents = [_make_agent_state("agent-other", AgentType.core_memory_agent, other_llm_config)]
    agent = _build_extraction_test_agent(client, all_agents, meta_llm_config)
    created_state = _make_agent_state("agent-topic-new", AgentType.topic_extraction_agent, meta_llm_config)
    agent.agent_manager.create_agent = AsyncMock(return_value=created_state)

    with _patch_llm_call():
        await agent._extract_topics_from_messages([_make_message()])

    agent.agent_manager.create_agent.assert_awaited_once()
    agent_create = agent.agent_manager.create_agent.await_args.kwargs["agent_create"]
    assert agent_create.agent_type == AgentType.topic_extraction_agent
    # Seeded from meta_memory_agent's own config (the fallback override),
    # NOT other_llm_config (all_agents[0] -- the retrieve path's fallback).
    assert agent_create.llm_config.model == "gpt-4o-meta-fallback"


@pytest.mark.asyncio
async def test_no_agents_at_all_falls_back_to_own_config_directly():
    """Defensive edge case (design's Proposed changes #11's else branch):
    shouldn't happen in practice since this Agent instance is itself one of
    the client's agents, but must fail safe exactly like today if it does.
    """
    client = _make_client()
    meta_llm_config = _make_llm_config(model="gpt-4o-meta-only")
    agent = _build_extraction_test_agent(client, all_agents=[], meta_agent_llm_config=meta_llm_config)

    with _patch_llm_call() as mock_create:
        await agent._extract_topics_from_messages([_make_message()])

    used_llm_config = mock_create.call_args.kwargs["llm_config"]
    assert used_llm_config.model == "gpt-4o-meta-only"
    agent.agent_manager.create_agent.assert_not_awaited()
