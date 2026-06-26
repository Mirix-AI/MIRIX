"""Unit tests for AgentManager._update_agent read-amplification reductions (VEPAGE-1283).

Under the IPS Relational provider, _update_agent previously did TWO reads per
write: a pre-read to capture old parent_id for cache invalidation, and a
post-write re-read with include_relationships=["tools"] to hydrate the return
value. For the hot path (config / system-prompt updates that never change
parent or tools — e.g. the ECMS force_update gather, N agents at once) both
reads are avoidable:

* The parent pre-read is only needed when parent_id is actually changing.
* The post-write hydrate read can be skipped when the caller passes its
  already-loaded ``current_state`` and the update didn't touch tool_ids; the
  return is then built locally by applying the changed scalar fields.

These tests pin that behavior while preserving the read-after-write path for
callers that don't pass current_state (or that change tools/parent).
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.agent import AgentType, UpdateAgent
from mirix.schemas.client import Client
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.services.agent_manager import AgentManager


def _actor():
    return Client(id="client-1", organization_id="org-1", name="Test Client", status="active")


def _state(model="gpt-4o-mini"):
    return PydanticAgentState(
        id="agent-1",
        name="episodic_memory_agent",
        agent_type=AgentType.episodic_memory_agent,
        system="old system",
        llm_config=LLMConfig(model=model, model_endpoint_type="openai", context_window=8192),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        ),
        organization_id="org-1",
        tools=[],
    )


@pytest.mark.asyncio
async def test_non_tool_update_with_current_state_skips_reads():
    """current_state + no tool change -> no pre-read, no hydrate re-read; the
    returned state reflects the applied scalar fields."""
    am = AgentManager()
    provider = AsyncMock()
    provider.read = AsyncMock()  # must NOT be called
    provider.update = AsyncMock()

    current = _state(model="old-model")
    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=provider,
    ):
        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            result = await am._update_agent(
                agent_id="agent-1",
                agent_update=UpdateAgent(system="new system"),
                actor=_actor(),
                current_state=current,
            )

    provider.update.assert_awaited_once()
    provider.read.assert_not_awaited()  # neither pre-read nor hydrate read
    assert result.system == "new system"
    assert result.id == "agent-1"


@pytest.mark.asyncio
async def test_no_current_state_falls_back_to_reread():
    """Without current_state, keep the post-write hydrate read (return-consuming
    callers like the SDK / admin endpoints rely on a freshly hydrated row)."""
    am = AgentManager()
    provider = AsyncMock()
    hydrated = _state(model="new-model").model_dump()
    provider.read = AsyncMock(return_value=hydrated)
    provider.update = AsyncMock()

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=provider,
    ):
        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            await am._update_agent(
                agent_id="agent-1",
                agent_update=UpdateAgent(system="new system"),
                actor=_actor(),
            )

    # One hydrate re-read (no current_state to build from), no parent pre-read
    # (parent_id not changing).
    provider.read.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_change_forces_reread_even_with_current_state():
    """When tool_ids change, the local copy can't reflect new tools, so re-read."""
    am = AgentManager()
    provider = AsyncMock()
    hydrated = _state().model_dump()
    provider.read = AsyncMock(return_value=hydrated)
    provider.update = AsyncMock()

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=provider,
    ):
        with patch("mirix.database.cache_provider.get_cache_provider", return_value=None):
            await am._update_agent(
                agent_id="agent-1",
                agent_update=UpdateAgent(tool_ids=["tool-x", "tool-y"]),
                actor=_actor(),
                current_state=_state(),
            )

    provider.read.assert_awaited_once()  # hydrate read happened despite current_state
