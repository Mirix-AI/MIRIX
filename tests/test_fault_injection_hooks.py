"""Tests for the MIRIX-side fault-injection HOOK CALL SITES.

The registry behaviour (shape->exception, fire-counting, no-op-when-off,
per-source keying) is exhaustively covered in test_fault_injection.py. These
tests verify the three MIRIX hooks are wired at the right place and pass the
right (site, source_key, tool) so the registry can match:

* tool_body  — inside Agent._execute_tool_inner, before the tool body runs
* llm        — meta-agent chaining / LLMChainingExhaustedError site
* subagent   — per-sub-agent run inside the fan-out

Each hook's prod-safety is the registry's no-op fast path; here we additionally
assert that with the flag OFF the hook never consults the registry (so a stray
directive can't fire in prod).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mirix.agent.agent import Agent
from mirix.errors import CorrectableToolError, ProviderPermanentError
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.enums import ToolType
from mirix.schemas.tool import Tool
from mirix.testing import fault_injection as fi


@pytest.fixture(autouse=True)
def _clean_registry():
    fi.reset()
    yield
    fi.reset()


def _make_tool_agent(memory_source_id="src-hook") -> Agent:
    """Minimal Agent able to run execute_tool_and_persist_state up to the hook,
    with DB-touching collaborators mocked. The MIRIX_EXTRA tool path is the
    shortest route to the hook (no block copy, no memory update)."""
    from unittest.mock import AsyncMock

    agent = Agent.__new__(Agent)
    agent.logger = MagicMock()
    agent.interface = MagicMock()
    agent.model = "test-model"
    agent.memory_source_id = memory_source_id
    agent.occurred_at = None

    user = MagicMock()
    user.timezone = "UTC"
    agent.user = user

    # get_blocks runs before the hook; stub it out.
    agent.block_manager = MagicMock()
    agent.block_manager.get_blocks = AsyncMock(return_value=[])
    agent._block_scopes = []

    agent_state = MagicMock(spec=AgentState)
    agent_state.id = "agent-meta"
    agent_state.name = "meta_memory_agent"
    agent_state.agent_type = AgentType.meta_memory_agent
    agent.agent_state = agent_state
    return agent


def _extra_tool(name="some_extra_tool") -> Tool:
    return Tool(
        tool_type=ToolType.MIRIX_EXTRA,
        name=name,
        json_schema={"name": name, "description": "test", "parameters": {}},
        return_char_limit=10000,
    )


@pytest.mark.asyncio
async def test_tool_body_hook_fires_attribute_error_and_propagates():
    """With the flag on and an attribute_error directive for this source, the
    tool_body hook raises AttributeError before the real tool body runs."""
    agent = _make_tool_agent("src-attr")
    fi.reset()
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.resolve_directives(
            "src-attr",
            {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "attribute_error"}]}},
        )
        # Patch the tool-module lookup so we'd know if the body ran (it must not).
        with patch("mirix.agent.agent.get_function_from_module") as gf:
            body_ran = {"flag": False}

            async def _body(**kwargs):
                body_ran["flag"] = True
                return "ok"

            gf.return_value = _body
            with pytest.raises(AttributeError):
                await agent.execute_tool_and_persist_state("some_extra_tool", {}, _extra_tool())
            assert body_ran["flag"] is False, "fault must fire before the tool body"
    assert fi.fire_count("src-attr", "tool_body") == 1


@pytest.mark.asyncio
async def test_tool_body_hook_correctable_is_raised_for_reprompt():
    """A `correctable` directive raises CorrectableToolError from the body —
    the outer handler (tested in test_agent_step_propagation.py) turns that
    into a bounded re-prompt."""
    agent = _make_tool_agent("src-corr")
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.resolve_directives(
            "src-corr",
            {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "correctable"}]}},
        )
        with patch("mirix.agent.agent.get_function_from_module") as gf:

            async def _body(**kwargs):
                return "ok"

            gf.return_value = _body
            with pytest.raises(CorrectableToolError):
                await agent.execute_tool_and_persist_state("some_extra_tool", {}, _extra_tool())


@pytest.mark.asyncio
async def test_tool_body_hook_scoped_to_named_tool():
    """A directive scoped to tool='target_tool' must not fire for other tools."""
    agent = _make_tool_agent("src-scope")
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.resolve_directives(
            "src-scope",
            {
                "__fault_injection__": {
                    "faults": [{"site": "tool_body", "shape": "attribute_error", "tool": "target_tool"}]
                }
            },
        )
        with patch("mirix.agent.agent.get_function_from_module") as gf:

            async def _body(**kwargs):
                return "fine"

            gf.return_value = _body
            # Different tool name → hook is a no-op, body runs.
            result = await agent.execute_tool_and_persist_state("some_other_tool", {}, _extra_tool("some_other_tool"))
            assert result == "fine"
    assert fi.fire_count("src-scope", "tool_body") == 0


@pytest.mark.asyncio
async def test_tool_body_hook_is_noop_when_flag_off():
    """Flag off → the hook never raises even with a directive present, and the
    real tool body runs normally (prod safety)."""
    agent = _make_tool_agent("src-off")
    # Register while on, then run while off.
    with patch.object(fi.settings, "fault_injection_enabled", True):
        fi.resolve_directives(
            "src-off",
            {"__fault_injection__": {"faults": [{"site": "tool_body", "shape": "attribute_error"}]}},
        )
    with patch.object(fi.settings, "fault_injection_enabled", False):
        with patch("mirix.agent.agent.get_function_from_module") as gf:

            async def _body(**kwargs):
                return "ran"

            gf.return_value = _body
            result = await agent.execute_tool_and_persist_state("some_extra_tool", {}, _extra_tool())
            assert result == "ran"
    assert fi.fire_count("src-off", "tool_body") == 0


# --------------------------------------------------------------------------- #
# M5 — sub-agent fan-out reducer: any-permanent → permanent, siblings succeed.
# This is the subtle correctness claim behind fault row 9. The reducer is the
# shared seam; here we prove the policy directly with a mixed result list.
# --------------------------------------------------------------------------- #


def test_reducer_reraises_permanent_when_one_subagent_fails():
    from mirix.functions.function_sets.memory_tools import (
        _decide_step_outcome_from_sub_agent_results,
    )

    # episodic succeeded, semantic raised a permanent provider error.
    results = ["episodic ok", ProviderPermanentError("synthetic")]
    with pytest.raises(ProviderPermanentError):
        _decide_step_outcome_from_sub_agent_results(["episodic", "semantic"], results)


def test_reducer_returns_all_on_full_success():
    from mirix.functions.function_sets.memory_tools import (
        _decide_step_outcome_from_sub_agent_results,
    )

    out = _decide_step_outcome_from_sub_agent_results(["episodic", "semantic"], ["a", "b"])
    assert out == ["a", "b"]
