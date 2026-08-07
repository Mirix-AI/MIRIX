"""Unit tests for the shared get_or_create_topic_extraction_agent helper and
_topic_extraction_agent_id (ECMS-522 task 6).

Both topic-extraction call sites (retrieve path: rest_api.py's
retrieve_memory_with_conversation; save path: agent.py's
Agent._extract_topics_from_messages) delegate their "find the
topic_extraction_agent row, or lazily create it from a fallback llm_config"
logic to this one shared helper. Covers the design's Testing strategy #5 four
behaviors plus the race-safety concurrent-create test (#9).
"""

from unittest.mock import AsyncMock

import pytest

from mirix.errors import ProviderConflictError
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import (
    _topic_extraction_agent_id,
    get_or_create_topic_extraction_agent,
)


def _make_actor(client_id="client-1"):
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


def _make_agent_state(agent_id, agent_type, llm_config, embedding_config=None):
    return AgentState(
        id=agent_id,
        name=f"{agent_id}-name",
        system="",
        agent_type=agent_type,
        llm_config=llm_config,
        embedding_config=embedding_config,
        tools=[],
        organization_id="org-1",
    )


class TestTopicExtractionAgentId:
    def test_deterministic_id_is_stable_for_same_client(self):
        assert _topic_extraction_agent_id("client-1") == _topic_extraction_agent_id("client-1")

    def test_deterministic_id_differs_across_clients(self):
        assert _topic_extraction_agent_id("client-1") != _topic_extraction_agent_id("client-2")

    def test_deterministic_id_has_expected_shape(self):
        assert _topic_extraction_agent_id("client-1") == "agent-topic-extraction-client-1"


class TestGetOrCreateTopicExtractionAgentBehaviors:
    @pytest.mark.asyncio
    async def test_a_returns_existing_row_llm_config_when_present(self):
        """(a) returns the existing row's llm_config when one is present in
        all_agents."""
        actor = _make_actor()
        existing_llm_config = _make_llm_config(model="gpt-4o-mini-topic")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, _make_llm_config("gpt-4o")),
            _make_agent_state("agent-topic", AgentType.topic_extraction_agent, existing_llm_config),
        ]
        agent_manager = AsyncMock()

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
        )

        assert result.model == "gpt-4o-mini-topic"
        agent_manager.create_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_b_creates_row_seeded_from_all_agents_0_on_miss_with_no_fallback_override(self):
        """(b) creates a row and returns its llm_config on a miss, seeding
        from all_agents[0].llm_config when no fallback_llm_config is passed."""
        actor = _make_actor()
        seed_llm_config = _make_llm_config(model="gpt-4o-seed")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config),
        ]
        created_state = _make_agent_state(
            _topic_extraction_agent_id(actor.id), AgentType.topic_extraction_agent, seed_llm_config
        )
        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(return_value=created_state)

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
        )

        assert result.model == "gpt-4o-seed"
        agent_manager.create_agent.assert_awaited_once()
        call_kwargs = agent_manager.create_agent.await_args.kwargs
        agent_create = call_kwargs["agent_create"]
        assert agent_create.id == _topic_extraction_agent_id(actor.id)
        assert agent_create.agent_type == AgentType.topic_extraction_agent
        assert agent_create.llm_config.model == "gpt-4o-seed"

    @pytest.mark.asyncio
    async def test_c_uses_passed_fallback_llm_config_instead_of_all_agents_0(self):
        """(c) uses the passed fallback_llm_config instead of
        all_agents[0].llm_config when one is given (the save-path case)."""
        actor = _make_actor()
        all_agents_0_config = _make_llm_config(model="gpt-4o-agent-0")
        save_path_fallback = _make_llm_config(model="gpt-4o-meta-memory")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, all_agents_0_config),
        ]
        created_state = _make_agent_state(
            _topic_extraction_agent_id(actor.id), AgentType.topic_extraction_agent, save_path_fallback
        )
        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(return_value=created_state)

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
            fallback_llm_config=save_path_fallback,
        )

        assert result.model == "gpt-4o-meta-memory"
        call_kwargs = agent_manager.create_agent.await_args.kwargs
        assert call_kwargs["agent_create"].llm_config.model == "gpt-4o-meta-memory"

    @pytest.mark.asyncio
    async def test_d_non_conflict_create_failure_logs_and_returns_fallback_without_raising(self):
        """(d) on a create_agent failure (non-conflict), logs a warning and
        returns the fallback without raising."""
        actor = _make_actor()
        seed_llm_config = _make_llm_config(model="gpt-4o-seed")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config),
        ]
        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(side_effect=RuntimeError("transient write failure"))

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
        )

        assert result.model == "gpt-4o-seed"

    @pytest.mark.asyncio
    async def test_empty_all_agents_raises_value_error(self):
        """Caller contract: all_agents must be non-empty -- the caller already
        handles the empty-roster case (e.g. retrieve path's "No agents found"
        response) before ever calling this helper."""
        agent_manager = AsyncMock()
        with pytest.raises(ValueError):
            await get_or_create_topic_extraction_agent(
                agent_manager=agent_manager,
                actor=_make_actor(),
                all_agents=[],
            )


class TestGetOrCreateTopicExtractionAgentRaceSafety:
    @pytest.mark.asyncio
    async def test_losing_create_race_rereads_and_uses_winners_row(self):
        """On a ProviderConflictError (lost the create race to a concurrent
        caller with the same deterministic id), re-read once and use the
        winner's row -- never raise, never retry more than once."""
        actor = _make_actor()
        seed_llm_config = _make_llm_config(model="gpt-4o-seed")
        winner_llm_config = _make_llm_config(model="gpt-4o-winner")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config),
        ]
        winner_state = _make_agent_state(
            _topic_extraction_agent_id(actor.id), AgentType.topic_extraction_agent, winner_llm_config
        )
        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(side_effect=ProviderConflictError("duplicate id"))
        agent_manager.get_agent_by_id = AsyncMock(return_value=winner_state)

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
        )

        assert result.model == "gpt-4o-winner"
        agent_manager.get_agent_by_id.assert_awaited_once()
        call_kwargs = agent_manager.get_agent_by_id.await_args.kwargs
        assert call_kwargs["agent_id"] == _topic_extraction_agent_id(actor.id)

    @pytest.mark.asyncio
    async def test_conflict_and_reread_failure_falls_back_without_raising(self):
        """If the post-conflict re-read itself fails, fall back to the
        fallback llm_config rather than raising."""
        actor = _make_actor()
        seed_llm_config = _make_llm_config(model="gpt-4o-seed")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config),
        ]
        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(side_effect=ProviderConflictError("duplicate id"))
        agent_manager.get_agent_by_id = AsyncMock(side_effect=RuntimeError("re-read failed"))

        result = await get_or_create_topic_extraction_agent(
            agent_manager=agent_manager,
            actor=actor,
            all_agents=all_agents,
        )

        assert result.model == "gpt-4o-seed"

    @pytest.mark.asyncio
    async def test_concurrent_first_ever_calls_yield_exactly_one_row(self):
        """Two concurrent calls for the same client, both missing the row,
        race against a fake backend that enforces the deterministic id's
        uniqueness (raising ProviderConflictError on the loser's insert).
        Assert exactly one row is ever created, and both callers' returned
        llm_config matches the winner's row.
        """
        import asyncio

        actor = _make_actor()
        seed_llm_config = _make_llm_config(model="gpt-4o-seed")
        all_agents = [
            _make_agent_state("agent-core", AgentType.core_memory_agent, seed_llm_config),
        ]

        # Shared fake backend: a dict keyed by deterministic id, guarded so
        # only the first writer wins -- mirrors a real primary-key constraint.
        store = {}
        store_lock = asyncio.Lock()

        async def _fake_create_agent(agent_create, actor):
            async with store_lock:
                if agent_create.id in store:
                    raise ProviderConflictError("duplicate id")
                created = _make_agent_state(agent_create.id, agent_create.agent_type, agent_create.llm_config)
                store[agent_create.id] = created
                return created

        async def _fake_get_agent_by_id(agent_id, actor):
            return store[agent_id]

        agent_manager = AsyncMock()
        agent_manager.create_agent = AsyncMock(side_effect=_fake_create_agent)
        agent_manager.get_agent_by_id = AsyncMock(side_effect=_fake_get_agent_by_id)

        results = await asyncio.gather(
            get_or_create_topic_extraction_agent(agent_manager=agent_manager, actor=actor, all_agents=all_agents),
            get_or_create_topic_extraction_agent(agent_manager=agent_manager, actor=actor, all_agents=all_agents),
        )

        assert len(store) == 1
        winner_config = next(iter(store.values())).llm_config
        assert results[0].model == winner_config.model
        assert results[1].model == winner_config.model
