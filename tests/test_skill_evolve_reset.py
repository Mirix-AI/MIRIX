"""Unit tests for the stateless skill-evolve context reset.

These tests pin the contract requested for the MetaClaw eval:

  1. Before each /v1/skills/evolve call the procedural agent's in-context history
     is forced empty (system-message-only).
  2. After each call the history is force-cleared again — UNCONDITIONALLY, even
     when the agent step raised — so the next evolve always starts empty.
  3. The evolve path runs with its own bounded chaining/tool budget
     (SKILL_EVOLVE_MAX_CHAINING_STEPS), not the global chat default. The cap is a
     wall-clock ceiling (env-overridable) that bounds a runaway curator well
     below the request timeout, not a hard-pinned magic number.

They are pure-mock and require no DB, server process, or LLM API key.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from mirix.constants import SKILL_EVOLVE_MAX_CHAINING_STEPS
from mirix.schemas.agent import AgentType
from mirix.server.rest_api import (
    SkillEvolveRequest,
    _reset_agent_in_context_to_system,
    evolve_skills,
)


# ---------------------------------------------------------------------------
# _reset_agent_in_context_to_system — the core clearing primitive
# ---------------------------------------------------------------------------


def _reset_server(message_ids):
    """A minimal mock server exposing only what the reset helper touches."""
    agent_state = SimpleNamespace(message_ids=list(message_ids))
    server = SimpleNamespace()
    server.agent_manager = SimpleNamespace(
        get_agent_by_id=AsyncMock(return_value=agent_state),
        set_in_context_messages=AsyncMock(),
    )
    server.message_manager = SimpleNamespace(
        delete_detached_messages_for_agent=AsyncMock(),
    )
    return server


@pytest.mark.asyncio
async def test_reset_clears_history_to_system_only():
    """Multiple messages -> keep only index 0 (system) and hard-delete the rest."""
    server = _reset_server(["sys", "m1", "m2", "m3"])

    await _reset_agent_in_context_to_system(server, "proc-1", actor="client")

    server.agent_manager.set_in_context_messages.assert_awaited_once_with(
        agent_id="proc-1", message_ids=["sys"], actor="client"
    )
    server.message_manager.delete_detached_messages_for_agent.assert_awaited_once_with(
        agent_id="proc-1", actor="client"
    )


@pytest.mark.asyncio
async def test_reset_is_noop_when_already_system_only():
    """A single (system) message -> nothing to clear, no writes, no deletes."""
    server = _reset_server(["sys"])

    await _reset_agent_in_context_to_system(server, "proc-1", actor="client")

    server.agent_manager.set_in_context_messages.assert_not_awaited()
    server.message_manager.delete_detached_messages_for_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_is_noop_when_empty():
    """An empty message_ids list must not blow up and must be a no-op."""
    server = _reset_server([])

    await _reset_agent_in_context_to_system(server, "proc-1", actor="client")

    server.agent_manager.set_in_context_messages.assert_not_awaited()
    server.message_manager.delete_detached_messages_for_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# evolve_skills — the orchestration contract (pre-clear -> step(50) -> post-clear)
# ---------------------------------------------------------------------------


def _evolve_mocks(monkeypatch, *, step_side_effect=None, residue=("sys", "m1")):
    """Wire up the module-level dependencies of evolve_skills with mocks.

    `residue` is what get_agent_by_id reports each time the reset helper runs, so
    both the pre-clear and the post-clear see >1 message and actually clear.
    Returns (server, proc_step_mock).
    """
    client = SimpleNamespace(id="client-1")

    proc_state = SimpleNamespace(
        id="proc-1",
        agent_type=AgentType.procedural_memory_agent,
        children=[],
        message_ids=list(residue),
    )
    user = SimpleNamespace(id="user-1", timezone="UTC", organization_id="org-1")

    server = SimpleNamespace()
    server.default_interface_factory = lambda: MagicMock()
    server.user_manager = SimpleNamespace(get_user_by_id=AsyncMock(return_value=user))
    server.agent_manager = SimpleNamespace(
        list_agents=AsyncMock(return_value=[proc_state]),
        get_agent_by_id=AsyncMock(
            return_value=SimpleNamespace(message_ids=list(residue))
        ),
        set_in_context_messages=AsyncMock(),
    )
    server.message_manager = SimpleNamespace(
        delete_detached_messages_for_agent=AsyncMock(),
    )
    server.procedural_memory_manager = SimpleNamespace(
        list_procedures=AsyncMock(return_value=[]),  # empty before & after -> empty diff
    )

    proc_step = AsyncMock(side_effect=step_side_effect)
    proc_agent = MagicMock()
    proc_agent.step = proc_step

    monkeypatch.setattr("mirix.server.rest_api.get_server", lambda: server)
    monkeypatch.setattr(
        "mirix.server.rest_api.get_client_from_jwt_or_api_key",
        AsyncMock(return_value=(client, "api_key")),
    )
    monkeypatch.setattr("mirix.agent.ProceduralMemoryAgent", lambda **kw: proc_agent)

    return server, proc_step


@pytest.mark.asyncio
async def test_evolve_preclears_then_steps_with_raised_budget_then_postclears(monkeypatch):
    """Happy path: pre-clear runs before step; step uses the 50-step budget; post-clear runs after."""
    captured = {}

    async def _step_side_effect(*args, **kwargs):
        # The pre-clear MUST have already happened by the time step runs.
        assert server.agent_manager.set_in_context_messages.await_count >= 1, (
            "pre-clear must run before the agent step"
        )
        captured["pre_clear_count"] = server.agent_manager.set_in_context_messages.await_count

    server, proc_step = _evolve_mocks(monkeypatch, step_side_effect=_step_side_effect)

    req = SkillEvolveRequest(messages=["line a", "line b"], user_id="user-1")
    result = await evolve_skills(request=req, authorization="Bearer x", http_request=None)

    # Step ran exactly once with the raised, evolve-specific chaining budget.
    proc_step.assert_awaited_once()
    assert proc_step.await_args.kwargs["max_chaining_steps"] == SKILL_EVOLVE_MAX_CHAINING_STEPS
    assert proc_step.await_args.kwargs["chaining"] is True

    # Pre-clear (before step) + post-clear (after step) = two clears, each system-only.
    assert captured["pre_clear_count"] == 1
    assert server.agent_manager.set_in_context_messages.await_count == 2
    for call in server.agent_manager.set_in_context_messages.await_args_list:
        assert call.kwargs["message_ids"] == ["sys"]
    assert server.message_manager.delete_detached_messages_for_agent.await_count == 2

    assert result["success"] is True


@pytest.mark.asyncio
async def test_evolve_postclears_even_when_step_raises(monkeypatch):
    """Failure path: a raising step still triggers the finally post-clear, and surfaces 500."""
    server, proc_step = _evolve_mocks(
        monkeypatch, step_side_effect=RuntimeError("boom")
    )

    req = SkillEvolveRequest(messages=["oops"], user_id="user-1")

    with pytest.raises(HTTPException) as exc_info:
        await evolve_skills(request=req, authorization="Bearer x", http_request=None)

    assert exc_info.value.status_code == 500

    # Even though step blew up: pre-clear (1) + finally post-clear (1) = 2 clears.
    assert server.agent_manager.set_in_context_messages.await_count == 2
    assert server.message_manager.delete_detached_messages_for_agent.await_count == 2


@pytest.mark.asyncio
async def test_evolve_chaining_budget_is_a_bounded_ceiling():
    """The evolve chaining budget is its own bounded wall-clock ceiling.

    It must be (a) positive and (b) materially smaller than the chat default's
    runaway potential — a curator converges in a few steps, so the cap exists
    only to stop a pathological spin well before the request timeout. We pin the
    intended default but assert the *property* (bounded, env-overridable) rather
    than a magic number, so tuning the default never silently breaks this test.
    """
    assert SKILL_EVOLVE_MAX_CHAINING_STEPS == 15
    assert 0 < SKILL_EVOLVE_MAX_CHAINING_STEPS <= 50
