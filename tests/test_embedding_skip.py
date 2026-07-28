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
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
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


def _manager_no_session(manager) -> None:
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


# ---------------------------------------------------------------------------
# ResourceMemoryManager.insert_resource
# ---------------------------------------------------------------------------


def _resource_agent_state() -> AgentState:
    return AgentState(
        id="agent-res-1",
        name="resource_memory_agent",
        system="sys",
        agent_type=AgentType.resource_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
    )


def _resource_provider_result(actor: PydanticClient, agent_state: AgentState) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": "res_mem_testprovider1",
        "title": "ttl",
        "summary": "sum",
        "content": "cnt",
        "resource_type": "doc",
        "user_id": UserManager.ADMIN_USER_ID,
        "organization_id": "org-test-1",
        "client_id": actor.id,
        "agent_id": agent_state.id,
        "filter_tags": {},
        "embedding_config": agent_state.embedding_config,
        "last_modify": {"timestamp": now.isoformat(), "operation": "created"},
    }


@pytest.mark.asyncio
async def test_insert_resource_with_provider_does_not_call_embedding_model():
    actor = _actor()
    agent_state = _resource_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_resource_provider_result(actor, agent_state))

    manager = ResourceMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.resource_memory_manager.embedding_model", new_callable=AsyncMock) as mock_embedding_model,
    ):
        await manager.insert_resource(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            title="ttl",
            summary="sum",
            resource_type="doc",
            content="cnt",
            organization_id="org-test-1",
        )

    mock_embedding_model.assert_not_called()


@pytest.mark.asyncio
async def test_insert_resource_with_provider_create_has_no_embedding_keys():
    actor = _actor()
    agent_state = _resource_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_resource_provider_result(actor, agent_state))

    manager = ResourceMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.resource_memory_manager.embedding_model", new_callable=AsyncMock),
    ):
        await manager.insert_resource(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            title="ttl",
            summary="sum",
            resource_type="doc",
            content="cnt",
            organization_id="org-test-1",
        )

    provider.create.assert_awaited_once()
    assert provider.create.await_args.args[0] == "resource_memory"
    data_dict = provider.create.await_args.args[1]
    assert not any(k.endswith("_embedding") for k in data_dict)


@pytest.mark.asyncio
async def test_insert_resource_without_provider_calls_embedding_model_when_build_enabled():
    actor = _actor()
    agent_state = _resource_agent_state()
    manager = ResourceMemoryManager()

    mock_embed = AsyncMock()
    mock_embed.get_text_embedding = AsyncMock(return_value=[0.03] * 16)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
        patch("mirix.services.resource_memory_manager.BUILD_EMBEDDINGS_FOR_MEMORY", True),
        patch(
            "mirix.services.resource_memory_manager.embedding_model", new_callable=AsyncMock, return_value=mock_embed
        ) as mock_embedding_model_factory,
        mock.patch.object(ResourceMemoryManager, "create_item", new_callable=AsyncMock) as mock_create_item,
    ):
        await manager.insert_resource(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            title="ttl",
            summary="sum",
            resource_type="doc",
            content="cnt",
            organization_id="org-test-1",
        )

    mock_embedding_model_factory.assert_awaited_once()
    mock_embed.get_text_embedding.assert_awaited()
    mock_create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_resource_with_provider_passes_embedding_config_to_create():
    actor = _actor()
    agent_state = _resource_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_resource_provider_result(actor, agent_state))

    manager = ResourceMemoryManager()
    _manager_no_session(manager)

    with patch("mirix.database.relational_provider.get_relational_provider", return_value=provider):
        await manager.insert_resource(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            title="ttl",
            summary="sum",
            resource_type="doc",
            content="cnt",
            organization_id="org-test-1",
        )

    data_dict = provider.create.await_args.args[1]
    assert "embedding_config" in data_dict
    assert data_dict["embedding_config"] == agent_state.embedding_config


# ---------------------------------------------------------------------------
# ProceduralMemoryManager.insert_procedure
# ---------------------------------------------------------------------------


def _procedural_agent_state() -> AgentState:
    return AgentState(
        id="agent-proc-1",
        name="procedural_memory_agent",
        system="sys",
        agent_type=AgentType.procedural_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
    )


def _procedural_provider_result(actor: PydanticClient, agent_state: AgentState) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": "proc_mem_testprovider1",
        "entry_type": "task",
        "summary": "sum",
        "steps": ["step1", "step2"],
        "user_id": UserManager.ADMIN_USER_ID,
        "organization_id": "org-test-1",
        "client_id": actor.id,
        "agent_id": agent_state.id,
        "filter_tags": {},
        "last_modify": {"timestamp": now.isoformat(), "operation": "created"},
    }


@pytest.mark.asyncio
async def test_insert_procedure_with_provider_does_not_call_embedding_model():
    actor = _actor()
    agent_state = _procedural_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_procedural_provider_result(actor, agent_state))

    manager = ProceduralMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch(
            "mirix.services.procedural_memory_manager.embedding_model", new_callable=AsyncMock
        ) as mock_embedding_model,
    ):
        await manager.insert_procedure(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            entry_type="task",
            summary="sum",
            steps=["step1", "step2"],
            organization_id="org-test-1",
        )

    mock_embedding_model.assert_not_called()


@pytest.mark.asyncio
async def test_insert_procedure_with_provider_create_has_no_embedding_keys():
    actor = _actor()
    agent_state = _procedural_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_procedural_provider_result(actor, agent_state))

    manager = ProceduralMemoryManager()
    _manager_no_session(manager)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=provider),
        patch("mirix.services.procedural_memory_manager.embedding_model", new_callable=AsyncMock),
    ):
        await manager.insert_procedure(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            entry_type="task",
            summary="sum",
            steps=["step1", "step2"],
            organization_id="org-test-1",
        )

    provider.create.assert_awaited_once()
    assert provider.create.await_args.args[0] == "procedural_memory"
    data_dict = provider.create.await_args.args[1]
    assert not any(k.endswith("_embedding") for k in data_dict)


@pytest.mark.asyncio
async def test_insert_procedure_without_provider_calls_embedding_model_when_build_enabled():
    actor = _actor()
    agent_state = _procedural_agent_state()
    manager = ProceduralMemoryManager()

    mock_embed = AsyncMock()
    mock_embed.get_text_embedding = AsyncMock(return_value=[0.04] * 16)

    with (
        patch("mirix.database.relational_provider.get_relational_provider", return_value=None),
        patch("mirix.services.procedural_memory_manager.BUILD_EMBEDDINGS_FOR_MEMORY", True),
        patch(
            "mirix.services.procedural_memory_manager.embedding_model", new_callable=AsyncMock, return_value=mock_embed
        ) as mock_embedding_model_factory,
        mock.patch.object(ProceduralMemoryManager, "create_item", new_callable=AsyncMock) as mock_create_item,
    ):
        await manager.insert_procedure(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            entry_type="task",
            summary="sum",
            steps=["step1", "step2"],
            organization_id="org-test-1",
        )

    mock_embedding_model_factory.assert_awaited_once()
    mock_embed.get_text_embedding.assert_awaited()
    mock_create_item.assert_awaited_once()


@pytest.mark.asyncio
async def test_insert_procedure_with_provider_passes_no_embedding_vectors():
    actor = _actor()
    agent_state = _procedural_agent_state()
    provider = AsyncMock()
    provider.create = AsyncMock(return_value=_procedural_provider_result(actor, agent_state))

    manager = ProceduralMemoryManager()
    _manager_no_session(manager)

    with patch("mirix.database.relational_provider.get_relational_provider", return_value=provider):
        await manager.insert_procedure(
            actor=actor,
            agent_state=agent_state,
            agent_id=agent_state.id,
            entry_type="task",
            summary="sum",
            steps=["step1", "step2"],
            organization_id="org-test-1",
        )

    data_dict = provider.create.await_args.args[1]
    # Procedural insert does not forward embedding_config on the provider
    # branch (it was never part of the provider data_dict), so just verify
    # no embedding vectors are present.
    assert not any(k.endswith("_embedding") for k in data_dict)
