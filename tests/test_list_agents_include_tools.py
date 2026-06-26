"""Unit tests for AgentManager.list_agents(include_tools=...) (VEPAGE-1283).

Under the IPS Relational provider, hydrating an agent's ``tools`` relationship
costs one ``tool_manager.list_tools_by_ids`` roundtrip PER agent (an N+1). The
search / topic-extraction read paths only need a single agent's
``llm_config`` / ``embedding_config`` and never its tools, so they pass
``include_tools=False`` to suppress that per-agent tool fan-out.

These tests pin the contract that ``include_tools`` controls whether the
provider list query requests the ``tools`` relationship at all — the structural
guarantee behind the read-path fix (the end-to-end "no per-agent tool spans on
the search path" behaviour is guarded by the ECMS full-stack test
``test_sdk_search_perf.py``).
"""

from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.client import Client
from mirix.services.agent_manager import AgentManager


def _make_actor():
    return Client(
        id="client-1",
        organization_id="org-1",
        name="Test Client",
        status="active",
    )


async def _call_list_agents(**kwargs):
    """Invoke list_agents against a recording fake relational provider and
    return the ``include_relationships`` kwarg the provider list query received.
    """
    am = AgentManager()
    fake_rp = AsyncMock()
    # The default (parent_id=None, no query_text) path returns raw rows; an
    # empty list is enough since we only assert on the call kwargs.
    fake_rp.find_using_named_query = AsyncMock(return_value=[])

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        await am.list_agents(actor=_make_actor(), **kwargs)

    assert fake_rp.find_using_named_query.await_count == 1
    return fake_rp.find_using_named_query.await_args.kwargs.get("include_relationships")


@pytest.mark.asyncio
async def test_include_tools_false_skips_tool_relationship():
    """include_tools=False must NOT request the tools relationship, so the
    provider does not fire a per-agent list_tools_by_ids hydration."""
    include_relationships = await _call_list_agents(limit=1, include_tools=False)
    assert include_relationships is None


@pytest.mark.asyncio
async def test_include_tools_false_yields_valid_agent_state_with_empty_tools():
    """Regression guard: with include_tools=False the provider returns rows
    WITHOUT a ``tools`` key, but AgentState.tools is a required field. list_agents
    must still produce a valid AgentState (tools defaulted to []), not raise a
    ValidationError. (A missing default here previously crashed every search /
    PII scan with "1 validation error for AgentState".)"""
    am = AgentManager()
    fake_rp = AsyncMock()
    # A realistic flat row as from_entity produces it when tools are NOT
    # hydrated: no ``tools`` key at all.
    tools_less_row = {
        "id": "agent-1",
        "name": "meta_memory_agent",
        "agent_type": "meta_memory_agent",
        "system": "you are a meta agent",
        "llm_config": {
            "model": "gpt-4o-mini",
            "model_endpoint_type": "openai",
            "context_window": 8192,
        },
        "embedding_config": {
            "embedding_endpoint_type": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        },
        "organization_id": "org-1",
    }
    fake_rp.find_using_named_query = AsyncMock(return_value=[tools_less_row])

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agents = await am.list_agents(actor=_make_actor(), limit=1, include_tools=False)

    assert len(agents) == 1
    assert agents[0].id == "agent-1"
    # The required ``tools`` field is defaulted to an empty list, not missing.
    assert agents[0].tools == []


@pytest.mark.asyncio
async def test_include_tools_default_hydrates_tools():
    """The default preserves the existing behaviour for callers that rely on
    populated ``tools`` (e.g. agent construction / fan-out)."""
    include_relationships = await _call_list_agents(limit=1000)
    assert include_relationships == ["tools"]


@pytest.mark.asyncio
async def test_include_tools_true_explicit_hydrates_tools():
    """Passing include_tools=True explicitly matches the default."""
    include_relationships = await _call_list_agents(limit=1000, include_tools=True)
    assert include_relationships == ["tools"]
