"""
Provider-mode embedding behavior for the knowledge vault manager.

In provider mode (a relational provider is registered) embeddings are owned by
the search index, not Mirix. The provider write path must therefore NOT compute
a caption embedding -- it is wasted work that is stripped before persistence.
This mirrors every other memory manager, whose provider branch never calls the
embedding model.

Usage:
    pytest tests/test_knowledge_vault_provider_embedding.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.database.relational_provider import (
    get_registered_relational_providers,
    register_relational_provider,
    reset_provider_mode_latch,
    unregister_relational_provider,
)
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def cleanup_relational_registry():
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()
    yield
    for name in list(get_registered_relational_providers().keys()):
        unregister_relational_provider(name)
    reset_provider_mode_latch()


class _EchoProvider:
    """Minimal relational provider whose create() echoes the row back."""

    def __init__(self):
        self.created = []

    async def create(self, table, data, actor=None):
        self.created.append((table, data))
        # insert_knowledge constructs PydanticKnowledgeVaultItem(**result),
        # so return the row as-is (it already carries all required fields).
        return data


def _agent_state() -> AgentState:
    embedding_config = EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536,
    )
    return AgentState(
        id="agent-test",
        name="kv-test-agent",
        system="test",
        agent_type="knowledge_vault_memory_agent",
        llm_config=LLMConfig.default_config(model_name="gpt-4"),
        embedding_config=embedding_config,
        tools=[],
    )


async def test_provider_path_does_not_compute_caption_embedding():
    """The provider branch must not invoke the embedding model."""
    provider = _EchoProvider()
    register_relational_provider("ips_relational", provider)

    actor = PydanticClient(id="client-test", name="Test Client", organization_id="org-test")
    agent_state = _agent_state()
    mgr = KnowledgeVaultManager()

    with patch(
        "mirix.services.knowledge_vault_manager.embedding_model", new_callable=AsyncMock
    ) as mock_embedding_model:
        await mgr.insert_knowledge(
            actor=actor,
            agent_state=agent_state,
            agent_id="agent-test",
            entry_type="credential",
            source="test",
            sensitivity="low",
            secret_value="value",
            caption="a caption to embed",
            organization_id="org-test",
            user_id="user-test",
        )

    # The embedding model must never be constructed/called on the provider path.
    mock_embedding_model.assert_not_called()

    # Exactly one row was written to the provider...
    assert len(provider.created) == 1
    table, row = provider.created[0]
    assert table == "knowledge_vault"
    # ...with no computed vector, and embedding_config forwarded as metadata.
    assert row.get("caption_embedding") is None
    assert row["embedding_config"] == agent_state.embedding_config
