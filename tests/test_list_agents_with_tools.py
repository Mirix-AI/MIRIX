"""Unit tests for AgentManager.list_agents_with_tools (VEPAGE-1228).

The method fetches all child agents of a meta-agent joined with their tools in a
single IPS-R named-query roundtrip, replacing the N+1 where ``list_agents`` is
followed by one ``list_tools_by_ids`` per agent. Rows come back from the
``agent_manager.list_agents_with_tools_by_parent`` joined NQ as raw projection
dicts (skip_entity_mapping=True) and are grouped agent-side into
``PydanticAgentState`` objects with their ``tools`` populated.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from mirix.schemas.agent import AgentType
from mirix.services.agent_manager import AgentManager


def _make_actor():
    class _Actor:
        organization_id = "org-1"
        id = "client-1"

    return _Actor()


def _agent_row(agent_id, agent_type, tool_id, tool_name):
    """One flat (agent, tool) join row as the NQ returns it."""
    return {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "agent_name": agent_type,
        "agent_description": None,
        "agent_llm_config": json.dumps(
            {"model": "gpt-4o-mini", "model_endpoint_type": "openai", "context_window": 8192}
        ),
        "agent_embedding_config": json.dumps(
            {
                "embedding_endpoint_type": "openai",
                "embedding_model": "text-embedding-3-small",
                "embedding_dim": 1536,
            }
        ),
        "agent_system": "you are a memory agent",
        "agent_tool_rules": None,
        "agent_mcp_tools": None,
        "agent_parent_id": "meta-1",
        "agent_organization_id": "org-1",
        "agent_owner": "APP=client-1",
        "agent_created_at": "2026-06-04T00:00:00+00:00",
        "agent_updated_at": "2026-06-04T00:00:00+00:00",
        "tool_id": tool_id,
        "tool_name": tool_name,
        "tool_description": None,
        "tool_json_schema": json.dumps({"name": tool_name, "parameters": {}}),
        "tool_type": "mirix_memory_core",
        "tool_organization_id": "org-1",
    }


@pytest.mark.asyncio
async def test_single_roundtrip_groups_tools_per_agent():
    am = AgentManager()
    fake_rp = AsyncMock()
    # Two agents, each with two tools -> 4 flat rows, ONE NQ call.
    fake_rp.find_using_named_query = AsyncMock(
        return_value=[
            _agent_row("agent-epi", "episodic_memory_agent", "tool-aaaaaaaa", "episodic_memory_insert"),
            _agent_row("agent-epi", "episodic_memory_agent", "tool-bbbbbbbb", "conversation_search"),
            _agent_row("agent-sem", "semantic_memory_agent", "tool-cccccccc", "semantic_memory_insert"),
            _agent_row("agent-sem", "semantic_memory_agent", "tool-bbbbbbbb", "conversation_search"),
        ]
    )

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agents = await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    # Exactly one IPS-R roundtrip.
    assert fake_rp.find_using_named_query.await_count == 1
    called_table, called_nq = fake_rp.find_using_named_query.await_args.args[:2]
    assert called_table == "agents"
    assert called_nq == "agent_manager.list_agents_with_tools_by_parent"

    # Grouped into 2 distinct agents.
    assert {a.id for a in agents} == {"agent-epi", "agent-sem"}
    by_id = {a.id: a for a in agents}
    assert {t.name for t in by_id["agent-epi"].tools} == {
        "episodic_memory_insert",
        "conversation_search",
    }
    assert {t.name for t in by_id["agent-sem"].tools} == {
        "semantic_memory_insert",
        "conversation_search",
    }
    # Agent scalar fields hydrated from the projection.
    assert by_id["agent-epi"].system == "you are a memory agent"
    assert by_id["agent-epi"].llm_config.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_agent_with_no_tools_is_returned():
    am = AgentManager()
    row = _agent_row("agent-core", "core_memory_agent", None, None)
    # LEFT JOIN: an agent with no tools yields a row with null tool columns.
    row["tool_id"] = None
    row["tool_name"] = None
    row["tool_json_schema"] = None
    fake_rp = AsyncMock()
    fake_rp.find_using_named_query = AsyncMock(return_value=[row])

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agents = await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    assert len(agents) == 1
    assert agents[0].id == "agent-core"
    assert agents[0].tools == []


@pytest.mark.asyncio
async def test_stray_null_tool_row_does_not_create_phantom_empty():
    """Grouping is robust to a null-tool row arriving alongside a real tool row
    for the same agent.

    Post-VEPAGE-1228-fix the NQ's INNER-wrapped LEFT JOIN should not emit a
    null-tool row for an agent that has at least one in-scope tool (a
    soft-deleted/cross-org mapping yields no row at all). This test pins the
    client-side grouping contract defensively: even if such a row did arrive,
    the agent must end with exactly its real tool, not a phantom empty entry.
    """
    am = AgentManager()
    real = _agent_row("agent-epi", "episodic_memory_agent", "tool-aaaaaaaa", "episodic_memory_insert")
    stray = _agent_row("agent-epi", "episodic_memory_agent", None, None)
    stray["tool_id"] = None
    stray["tool_name"] = None
    stray["tool_json_schema"] = None
    fake_rp = AsyncMock()
    fake_rp.find_using_named_query = AsyncMock(return_value=[real, stray])

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agents = await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    assert len(agents) == 1
    assert [t.name for t in agents[0].tools] == ["episodic_memory_insert"]


@pytest.mark.asyncio
async def test_no_relational_provider_falls_back_to_list_agents():
    am = AgentManager()
    sentinel = [object()]
    with (
        patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=None,
        ),
        patch.object(am, "list_agents", new=AsyncMock(return_value=sentinel)) as la,
    ):
        result = await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    la.assert_awaited_once()
    assert result is sentinel


# --- VEPAGE-1228 (reopened): positional / non-dict wire rows ---------------
#
# The grouping originally assumed every row was a dict and did
# ``row.get("agent_id")``. Against the *deployed* NQ via the local IPS-R SDK
# the joined multi-column projection comes back as POSITIONAL LISTS, not named
# dicts, so ``.get`` blew up with ``AttributeError: 'list' object has no
# attribute 'get'`` on every save. The fix passes a ``result_set_entity_class``
# so the relational provider hydrates positional rows into dicts (the same
# positional-zip contract ECMS's async_ips_relational_provider uses), keyed by
# the dataclass field order, which must match the NQ SELECT order.

# Authoritative NQ SELECT order (ipsr-mgd-model
# agent_manager.list_agents_with_tools_by_parent).
_NQ_COLUMN_ORDER = [
    "agent_id",
    "agent_type",
    "agent_name",
    "agent_description",
    "agent_llm_config",
    "agent_embedding_config",
    "agent_system",
    "agent_tool_rules",
    "agent_mcp_tools",
    "agent_parent_id",
    "agent_organization_id",
    "agent_owner",
    "agent_created_at",
    "agent_updated_at",
    "tool_id",
    "tool_name",
    "tool_description",
    "tool_json_schema",
    "tool_type",
    "tool_organization_id",
]


def _positional_row(agent_id, agent_type, tool_id, tool_name):
    """One flat (agent, tool) join row as POSITIONAL values, in NQ SELECT
    order — the shape the deployed NQ returns through the local IPS-R SDK."""
    d = _agent_row(agent_id, agent_type, tool_id, tool_name)
    return [d[col] for col in _NQ_COLUMN_ORDER]


def _provider_that_hydrates_positional_rows(positional_rows):
    """A fake relational provider mirroring ECMS
    async_ips_relational_provider.find_using_named_query(skip_entity_mapping=
    True): positional list rows are zipped into dicts ONLY when a
    ``result_set_entity_class`` is supplied (its field order is contractual
    with the NQ projection order). With no class, rows pass through as lists.
    """
    import dataclasses

    async def _fake(table, query_name, **kwargs):
        cls = kwargs.get("result_set_entity_class")
        if cls is None:
            return list(positional_rows)
        fields = list(getattr(cls, "__dataclass_fields__", {}).keys())
        hydrated = []
        for row in positional_rows:
            positional = row if isinstance(row, list) else [row]
            kw = dict(zip(fields, positional))
            hydrated.append(dataclasses.asdict(cls(**kw)))
        return hydrated

    rp = AsyncMock()
    rp.find_using_named_query = AsyncMock(side_effect=_fake)
    return rp


@pytest.mark.asyncio
async def test_positional_rows_are_grouped_without_attribute_error():
    """Repro of the reopened defect: deployed-NQ rows arrive positional, not as
    dicts. Grouping must still produce agents with their tools.
    """
    am = AgentManager()
    fake_rp = _provider_that_hydrates_positional_rows(
        [
            _positional_row("agent-epi", "episodic_memory_agent", "tool-aaaaaaaa", "episodic_memory_insert"),
            _positional_row("agent-epi", "episodic_memory_agent", "tool-bbbbbbbb", "conversation_search"),
            _positional_row("agent-sem", "semantic_memory_agent", "tool-cccccccc", "semantic_memory_insert"),
        ]
    )

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        agents = await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    assert {a.id for a in agents} == {"agent-epi", "agent-sem"}
    by_id = {a.id: a for a in agents}
    assert {t.name for t in by_id["agent-epi"].tools} == {
        "episodic_memory_insert",
        "conversation_search",
    }
    assert by_id["agent-epi"].llm_config.model == "gpt-4o-mini"
    assert by_id["agent-sem"].agent_type == AgentType.semantic_memory_agent


@pytest.mark.asyncio
async def test_passes_result_set_entity_class_to_named_query():
    """The grouping must delegate positional-row hydration to the provider by
    supplying a ``result_set_entity_class`` whose field order matches the NQ
    SELECT order — otherwise positional rows reach the grouping as lists.
    """
    am = AgentManager()
    fake_rp = _provider_that_hydrates_positional_rows([_positional_row("agent-core", "core_memory_agent", None, None)])

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_rp,
    ):
        await am.list_agents_with_tools(parent_id="meta-1", actor=_make_actor())

    cls = fake_rp.find_using_named_query.await_args.kwargs.get("result_set_entity_class")
    assert cls is not None, "list_agents_with_tools must pass result_set_entity_class"
    field_order = list(getattr(cls, "__dataclass_fields__", {}).keys())
    assert field_order == _NQ_COLUMN_ORDER, "dataclass field order must match the NQ SELECT projection order"
