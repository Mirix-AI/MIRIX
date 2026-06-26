"""trigger_memory_update resolves its child agents (with tools) via the joined
NQ ``list_agents_with_tools`` instead of ``list_agents`` + per-agent tool
hydration (VEPAGE-1228).

The sub-agent is constructed from the resolved ``agent_state``; because
``list_agents_with_tools`` returns states with ``.tools`` already populated, the
sub-agent step does not re-fetch tools from the provider.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mirix.functions.function_sets.memory_tools as mt


def _child_state(agent_type):
    return SimpleNamespace(
        id=f"agent-{agent_type}",
        name=f"{agent_type}_memory_agent",
        agent_type=f"{agent_type}_memory_agent",
        tools=[SimpleNamespace(id="tool-deadbeef", name="episodic_memory_insert")],
    )


@pytest.mark.asyncio
async def test_resolves_children_via_list_agents_with_tools():
    parent = MagicMock()
    parent.agent_state = SimpleNamespace(id="meta-1", name="meta_memory_agent")
    parent.actor = SimpleNamespace(id="client-1", organization_id="org-1")
    parent.interface = MagicMock()
    parent.user = SimpleNamespace(id="user-1")
    parent.filter_tags = None
    parent.use_cache = True

    # The new path: list_agents_with_tools returns tool-carrying child states.
    parent.agent_manager.list_agents_with_tools = AsyncMock(return_value=[_child_state("episodic")])
    # The old path must NOT be used.
    parent.agent_manager.list_agents = AsyncMock()

    # Stub the sub-agent class so .step() is a no-op (we only care about resolution).
    fake_agent = MagicMock()
    fake_agent.step = AsyncMock(return_value=None)
    fake_agent_cls = MagicMock(return_value=fake_agent)

    with (
        patch("mirix.agent.EpisodicMemoryAgent", fake_agent_cls),
        patch(
            "mirix.functions.function_sets.memory_tools.get_trace_context",
            return_value={},
        ),
        patch(
            "mirix.functions.function_sets.memory_tools.get_langfuse_client",
            return_value=None,
        ),
    ):
        await mt.trigger_memory_update(
            parent,
            {"message": SimpleNamespace(content="hi", model_copy=lambda deep: SimpleNamespace(content="hi"))},
            ["episodic"],
        )

    parent.agent_manager.list_agents_with_tools.assert_awaited_once_with(parent_id="meta-1", actor=parent.actor)
    parent.agent_manager.list_agents.assert_not_awaited()
    # Sub-agent constructed from the tool-carrying state -> no provider re-fetch.
    _, kwargs = fake_agent_cls.call_args
    assert kwargs["agent_state"].tools  # tools came pre-populated
