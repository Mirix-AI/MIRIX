"""
M4.4: When a relational provider is registered, insert_* paths skip embedding
computation and delegate to the provider.

Run: pytest tests/test_embedding_skip.py -v
"""

from datetime import datetime, timezone
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.user_manager import UserManager


def _actor() -> PydanticClient:
    return PydanticClient(
        id="client-test-1",
        name="Test Client",
        organization_id="org-test-1",
    )


def _episodic_agent_state() -> AgentState:
    return AgentState(
        id="agent-ep-1",
        name="episodic_memory_agent",
        system="sys",
        agent_type=AgentType.episodic_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
    )


def _semantic_agent_state() -> AgentState:
    return AgentState(
        id="agent-sem-1",
        name="semantic_memory_agent",
        system="sys",
        agent_type=AgentType.semantic_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
    )


def _episodic_provider_result(actor: PydanticClient, agent_state: AgentState) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": "ep_mem_testprovider1",
        "event_type": "user_message",
        "summary": "sum",
        "details": "det",
        "actor": "user",
        "user_id": UserManager.ADMIN_USER_ID,
        "organization_id": "org-test-1",
        "occurred_at": now,
        "client_id": actor.id,
        "agent_id": agent_state.id,
        "filter_tags": {},
        "embedding_config": agent_state.embedding_config,
        "last_modify": {"timestamp": now.isoformat(), "operation": "created"},
    }


def _semantic_provider_result(actor: PydanticClient, agent_state: AgentState) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": "sem_item_testprovider1",
        "name": "n",
        "summary": "sum",
        "details": "det",
        "source": "src",
        "user_id": UserManager.ADMIN_USER_ID,
        "organization_id": "org-test-1",
        "client_id": actor.id,
        "agent_id": agent_state.id,
        "filter_tags": {},
        "embedding_config": agent_state.embedding_config,
        "last_modify": {"timestamp": now.isoformat(), "operation": "created"},
    }


def _manager_no_session(manager: EpisodicMemoryManager | SemanticMemoryManager) -> None:
    """Fail fast if the SQL path accidentally opens a session."""
    manager.session_maker = MagicMock(side_effect=AssertionError("session_maker must not be called"))


# --- EpisodicMemoryManager.insert_event ---


@pytest.mark.asyncio
async def test_insert_event_with_provider_does_not_call_embedding_model():
    actor = _actor()
    agent_state = _episodic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_episodic_provider_result(actor, agent_state))

    manager = EpisodicMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.episodic_memory_manager.embedding_model", new_callable=AsyncMock) as mock_embedding_model,
    ):
        await manager.insert_event(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            event_type="user_message",
            timestamp=datetime.now(timezone.utc),
            event_actor="user",
            details="det",
            summary="sum",
            organization_id="org-test-1",
        )

    mock_embedding_model.assert_not_called()


@pytest.mark.asyncio
async def test_insert_event_with_provider_create_has_no_embedding_keys():
    actor = _actor()
    agent_state = _episodic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_episodic_provider_result(actor, agent_state))

    manager = EpisodicMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.episodic_memory_manager.embedding_model", new_callable=AsyncMock),
    ):
        await manager.insert_event(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            event_type="user_message",
            timestamp=datetime.now(timezone.utc),
            event_actor="user",
            details="det",
            summary="sum",
            organization_id="org-test-1",
        )

    provider.create.assert_awaited_once()
    assert provider.create.await_args.args[0] == "episodic_memory"
    data_dict = provider.create.await_args.args[1]
    assert not any(k.endswith("_embedding") for k in data_dict)


@pytest.mark.asyncio
async def test_insert_event_without_provider_calls_embedding_model_when_build_enabled():
    actor = _actor()
    agent_state = _episodic_agent_state()
    manager = EpisodicMemoryManager()

    mock_embed = AsyncMock()
    mock_embed.get_text_embedding = AsyncMock(return_value=[0.01] * 16)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
        patch("mirix.services.episodic_memory_manager.BUILD_EMBEDDINGS_FOR_MEMORY", True),
        patch(
            "mirix.services.episodic_memory_manager.embedding_model", new_callable=AsyncMock, return_value=mock_embed
        ) as mock_embedding_model_factory,
        mock.patch.object(EpisodicMemoryManager, "create_episodic_memory", new_callable=AsyncMock) as mock_create,
    ):
        await manager.insert_event(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            event_type="user_message",
            timestamp=datetime.now(timezone.utc),
            event_actor="user",
            details="det",
            summary="sum",
            organization_id="org-test-1",
        )

    mock_embedding_model_factory.assert_awaited_once()
    mock_embed.get_text_embedding.assert_awaited()
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_event_with_provider_passes_embedding_config_to_create():
    actor = _actor()
    agent_state = _episodic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_episodic_provider_result(actor, agent_state))

    manager = EpisodicMemoryManager()
    _manager_no_session(manager)

    with patch("mirix.database.relational_provider.get_relational_provider", return_value=provider):
        await manager.insert_event(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            event_type="user_message",
            timestamp=datetime.now(timezone.utc),
            event_actor="user",
            details="det",
            summary="sum",
            organization_id="org-test-1",
        )

    data_dict = provider.create.await_args.args[1]
    assert "embedding_config" in data_dict
    assert data_dict["embedding_config"] == agent_state.embedding_config


# --- SemanticMemoryManager.insert_semantic_item ---


@pytest.mark.asyncio
async def test_insert_semantic_item_with_provider_does_not_call_embedding_model():
    actor = _actor()
    agent_state = _semantic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_semantic_provider_result(actor, agent_state))

    manager = SemanticMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.semantic_memory_manager.embedding_model", new_callable=AsyncMock) as mock_embedding_model,
    ):
        await manager.insert_semantic_item(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            name="n",
            summary="sum",
            details="det",
            source="src",
            organization_id="org-test-1",
        )

    mock_embedding_model.assert_not_called()


@pytest.mark.asyncio
async def test_insert_semantic_item_with_provider_create_has_no_embedding_keys():
    actor = _actor()
    agent_state = _semantic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_semantic_provider_result(actor, agent_state))

    manager = SemanticMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.semantic_memory_manager.embedding_model", new_callable=AsyncMock),
    ):
        await manager.insert_semantic_item(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            name="n",
            summary="sum",
            details="det",
            source="src",
            organization_id="org-test-1",
        )

    provider.create.assert_awaited_once()
    assert provider.create.await_args.args[0] == "semantic_memory"
    data_dict = provider.create.await_args.args[1]
    assert not any(k.endswith("_embedding") for k in data_dict)


@pytest.mark.asyncio
async def test_insert_semantic_item_without_provider_calls_embedding_model_when_build_enabled():
    actor = _actor()
    agent_state = _semantic_agent_state()
    manager = SemanticMemoryManager()

    mock_embed = AsyncMock()
    mock_embed.get_text_embedding = AsyncMock(return_value=[0.02] * 16)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
        patch("mirix.services.semantic_memory_manager.BUILD_EMBEDDINGS_FOR_MEMORY", True),
        patch(
            "mirix.services.semantic_memory_manager.embedding_model", new_callable=AsyncMock, return_value=mock_embed
        ) as mock_embedding_model_factory,
        mock.patch.object(SemanticMemoryManager, "create_item", new_callable=AsyncMock) as mock_create_item,
    ):
        await manager.insert_semantic_item(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            name="n",
            summary="sum",
            details="det",
            source="src",
            organization_id="org-test-1",
        )

    mock_embedding_model_factory.assert_awaited_once()
    mock_embed.get_text_embedding.assert_awaited()
    mock_create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_semantic_item_with_provider_passes_embedding_config_to_create():
    actor = _actor()
    agent_state = _semantic_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_semantic_provider_result(actor, agent_state))

    manager = SemanticMemoryManager()
    _manager_no_session(manager)

    with patch("mirix.database.relational_provider.get_relational_provider", return_value=provider):
        await manager.insert_semantic_item(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            name="n",
            summary="sum",
            details="det",
            source="src",
            organization_id="org-test-1",
        )

    data_dict = provider.create.await_args.args[1]
    assert "embedding_config" in data_dict
    assert data_dict["embedding_config"] == agent_state.embedding_config
