"""The six per-memory-type retrievals in
build_system_prompt_with_memories must run CONCURRENTLY (asyncio.gather), not
serially, and a failure in any one retrieval must propagate rather than being
silently swallowed.

These tests pin the *new* behavior introduced by the parallelization. The
byte-identical-output and per-type-span guarantees are covered by
test_build_prompt_retrieval_spans.py and the existing prompt tests.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mirix.agent.agent import Agent
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User


def _make_user(id="user-1", org_id="org-1") -> User:
    return User(
        id=id,
        organization_id=org_id,
        name="Test User",
        status="active",
        timezone="UTC",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def _make_agent_for_prompt_build() -> Agent:
    """A non-owning (meta) Agent with every retrieval manager method stubbed to
    return empty/zero, so the per-type gates all fire and no IPS round-trips
    happen. Mirrors the fixture in test_build_prompt_retrieval_spans.py."""
    agent_state = AgentState(
        id="agent-meta",
        name=AgentType.meta_memory_agent.value,
        system="System prompt",
        agent_type=AgentType.meta_memory_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
        tools=[],
        parent_id=None,
    )
    user = _make_user()

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.user_id = user.id
    agent._save_scopes = None

    agent.block_manager = SimpleNamespace(
        get_blocks=AsyncMock(return_value=[]),
        get_block_by_id=AsyncMock(return_value=None),
    )
    agent.knowledge_vault_manager = SimpleNamespace(
        list_knowledge=AsyncMock(return_value=[]),
        get_total_number_of_items=AsyncMock(return_value=0),
    )
    agent.episodic_memory_manager = SimpleNamespace(
        list_episodic_memory=AsyncMock(return_value=[]),
        get_total_number_of_items=AsyncMock(return_value=0),
    )
    agent.resource_memory_manager = SimpleNamespace(
        list_resources=AsyncMock(return_value=[]),
        get_total_number_of_items=AsyncMock(return_value=0),
    )
    agent.procedural_memory_manager = SimpleNamespace(
        list_procedures=AsyncMock(return_value=[]),
        get_total_number_of_items=AsyncMock(return_value=0),
    )
    agent.semantic_memory_manager = SimpleNamespace(
        list_semantic_items=AsyncMock(return_value=[]),
        get_total_number_of_items=AsyncMock(return_value=0),
    )
    agent.build_system_prompt = MagicMock(return_value="MEMORY PROMPT")
    return agent


@pytest.mark.asyncio
async def test_retrievals_run_concurrently(monkeypatch):
    """The six retrieval blocks must overlap in flight. We instrument the six
    distinct entry-point manager coroutines with a shared concurrency counter
    that gates on a barrier: each call increments the in-flight count, waits
    until all six have arrived (or a short timeout), then returns. If the
    retrievals were serial, the barrier would never reach >1 in flight and the
    test would hang/time out on the first call."""
    agent = _make_agent_for_prompt_build()

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    all_arrived = asyncio.Event()
    EXPECTED = 6

    async def barrier(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if in_flight >= EXPECTED:
                all_arrived.set()
        # Wait for everyone to arrive; if serial, only one ever arrives and we
        # rely on the outer wait_for to fail the test rather than hang forever.
        try:
            await asyncio.wait_for(all_arrived.wait(), timeout=2.0)
        finally:
            async with lock:
                in_flight -= 1
        return []

    # The single first awaited backend coroutine per memory type.
    agent.block_manager.get_blocks = AsyncMock(side_effect=barrier)
    agent.knowledge_vault_manager.list_knowledge = AsyncMock(side_effect=barrier)
    agent.episodic_memory_manager.list_episodic_memory = AsyncMock(side_effect=barrier)
    agent.resource_memory_manager.list_resources = AsyncMock(side_effect=barrier)
    agent.procedural_memory_manager.list_procedures = AsyncMock(side_effect=barrier)
    agent.semantic_memory_manager.list_semantic_items = AsyncMock(side_effect=barrier)

    await asyncio.wait_for(
        agent.build_system_prompt_with_memories(raw_system="SYS", topics="hello"),
        timeout=5.0,
    )

    assert max_in_flight == EXPECTED, (
        f"expected all {EXPECTED} retrievals in flight at once, "
        f"saw at most {max_in_flight} — retrievals are not concurrent"
    )


@pytest.mark.asyncio
async def test_retrieval_failure_propagates(monkeypatch):
    """A failure in one retrieval must surface, not be swallowed by gather."""
    agent = _make_agent_for_prompt_build()

    agent.semantic_memory_manager.list_semantic_items = AsyncMock(side_effect=RuntimeError("semantic backend down"))

    with pytest.raises(RuntimeError, match="semantic backend down"):
        await agent.build_system_prompt_with_memories(raw_system="SYS", topics="hello")
