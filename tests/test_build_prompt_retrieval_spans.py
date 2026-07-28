"""Each per-memory-type retrieval in build_system_prompt_with_memories must emit
its own timed_span so per-type duration + backend are attributable in the trace.
This is the instrumentation that proves whether the retrievals run serially."""

from contextlib import asynccontextmanager
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
    """Build a NON-owning (meta) Agent with every manager method used by
    build_system_prompt_with_memories stubbed to return empty/zero.

    A meta_memory_agent is not the owner of any of the 6 memory types, so the
    ``or "<type>" not in retrieved_memories`` guard fires for every type.
    Providers are left unregistered (the process default), so
    ``_fetch_recent_indexing_lag_window`` returns [] immediately without
    touching IPS-Relational.
    """
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
    agent._block_scopes = None

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
    # build_system_prompt is sync and just formats the dict; stub it out.
    agent.build_system_prompt = MagicMock(return_value="MEMORY PROMPT")
    return agent


@pytest.mark.asyncio
async def test_each_memory_type_emits_a_retrieve_span(monkeypatch):
    captured = []

    @asynccontextmanager
    async def fake_timed_span(name, metadata=None, **kwargs):
        # Mirror the real contract: yield a mutable ``rec`` dict the body may
        # populate (incl. the reserved ``span_output`` key).
        rec = {}
        captured.append((name, dict(metadata or {}), rec))
        yield rec

    # This patches the source module attribute. It intercepts correctly ONLY
    # because agent.py imports timedspan function-locally (binding at call
    # time). If that import is ever hoisted to module top, this patch would stop
    # intercepting and the test would fail to capture spans -- at which point
    # patch ``mirix.agent.agent.timedspan`` instead.
    monkeypatch.setattr("mirix.observability.timed.timedspan", fake_timed_span)

    agent = _make_agent_for_prompt_build()
    await agent.build_system_prompt_with_memories(raw_system="SYS", topics="hello")

    names = [n for n, _, _ in captured]
    for mem_type in ("core", "knowledge_vault", "episodic", "resource", "procedural", "semantic"):
        assert f"Retrieve {mem_type}" in names, f"missing Retrieve {mem_type} span; got {names}"


@pytest.mark.asyncio
async def test_each_retrieve_span_records_output_counts(monkeypatch):
    """R3 AC2: every retrieval span carries a non-empty output — the item
    counts it produced — instead of rendering Output: undefined."""
    captured = []

    @asynccontextmanager
    async def fake_timed_span(name, metadata=None, **kwargs):
        rec = {}
        captured.append((name, rec))
        yield rec

    monkeypatch.setattr("mirix.observability.timed.timedspan", fake_timed_span)

    agent = _make_agent_for_prompt_build()
    await agent.build_system_prompt_with_memories(raw_system="SYS", topics="hello")

    outputs = {name: rec.get("span_output") for name, rec in captured}
    assert outputs["Retrieve core"] == {"block_count": 0}
    assert outputs["Retrieve knowledge_vault"] == {"merged_count": 0, "total_items": 0}
    assert outputs["Retrieve episodic"] == {
        "recent_count": 0,
        "relevant_count": 0,
        "total_items": 0,
    }
    for mem_type in ("resource", "procedural", "semantic"):
        assert outputs[f"Retrieve {mem_type}"] == {"merged_count": 0, "total_items": 0}
