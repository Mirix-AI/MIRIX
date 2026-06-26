"""Tests for the single finalize chokepoint and "complete = success only" contract.

S2 of VEPAGE-1251. Before this story:

- `mark_processing_complete` was called unconditionally after the LLM loop
  even when the loop exited because `function_failed` exhausted re-prompts
  (agent.py:1597).
- "Mark complete" was scattered across four sites (agent direct-write,
  agent LLM, numaflow permanent, internal-loop finalize) with subtly
  different conditions.

After this story:

- A single `MemorySourceManager.finalize_source(source_id, outcome)`
  chokepoint exists. Today it writes `processing_complete=True` for any
  outcome that should NOT be reprocessed; VEPAGE-1250 will diversify this
  to write `status`.
- The LLM-path mark is gated on success — if the last loop iteration had
  `function_failed=True`, the source is NOT finalized.
- Direct-write, numaflow permanent, and internal-loop finalize all route
  through the chokepoint.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.usage import MirixUsageStatistics  # noqa: F401

# Reuse the fixture-style helpers from test_processing_skip.py to drive step().


def _make_agent_state(name: str):
    """Mirror test_processing_skip's helper — proven shape for driving step()."""
    from mirix.schemas.agent import AgentType

    state = MagicMock()
    state.id = "agent-123"
    state.name = name
    state.created_by_id = "client-1"
    state.parent_id = None
    state.agent_type = AgentType.meta_memory_agent
    state.is_type = lambda t: t == AgentType.meta_memory_agent
    state.llm_config = MagicMock()
    state.llm_config.model = "gpt-4o-mini"
    state.llm_config.model_endpoint_type = "openai"
    state.llm_config.context_window = 128000
    state.tools = []
    state.tool_rules = []
    state.system = "test system prompt"
    state.embedding_config = MagicMock()
    return state


def _make_user():
    return MagicMock(id="user-1", timezone="UTC")


def _make_actor():
    actor = MagicMock(id="actor-1", organization_id="org-1")
    actor.message_set_retention_count = 0  # No retention — avoids DB reads
    return actor


def _make_pydantic_source(processing_complete: bool = False):
    src = MagicMock()
    src.id = "src-abc"
    src.processing_complete = processing_complete
    return src


def _setup_agent(memory_source_id: str):
    from mirix.agent.agent import Agent

    agent_state = _make_agent_state("meta_memory_agent")
    user = _make_user()
    actor = _make_actor()

    agent = Agent.__new__(Agent)
    agent.agent_state = agent_state
    agent.user = user
    agent.actor = actor
    agent.user_id = user.id
    agent.memory_source_id = memory_source_id
    agent.direct_writes = None
    agent.memory_source_manager = MagicMock()
    agent.memory_source_manager.get_by_id = AsyncMock(return_value=_make_pydantic_source(processing_complete=False))
    agent.memory_source_manager.mark_processing_complete = AsyncMock()
    agent.memory_source_manager.finalize_source = AsyncMock()
    agent._persist_memory_source = AsyncMock()
    agent.interface = MagicMock()
    agent.filter_tags = None
    agent.block_filter_tags = None
    agent.use_cache = True
    agent.client_id = "client-1"
    agent.logger = MagicMock()
    agent.model = "gpt-4o-mini"
    agent.summarize = False
    agent.source_summary = None
    agent.source_summary_source = None

    agent.message_manager = MagicMock()
    agent.message_manager.get_messages_for_agent = AsyncMock(return_value=[])
    agent.message_manager.get_messages_for_agent_user = AsyncMock(return_value=[])
    agent.message_manager.create_many_messages = AsyncMock()
    agent.message_manager.hard_delete_user_messages_for_agent = AsyncMock()
    agent.source_message_manager = MagicMock()
    return agent, actor, user


def _make_step_response(function_failed: bool, continue_chaining: bool = False):
    from mirix.schemas.openai.chat_completion_response import UsageStatistics

    resp = MagicMock()
    resp.continue_chaining = continue_chaining
    resp.function_failed = function_failed
    resp.usage = UsageStatistics(completion_tokens=10, prompt_tokens=20, total_tokens=30)
    resp.messages = []
    return resp


# ---------- finalize_source chokepoint contract ----------


@pytest.mark.asyncio
async def test_finalize_source_success_marks_complete():
    """finalize_source(SUCCESS) writes processing_complete=True under
    today's boolean schema."""
    from mirix.queue.error_policy import SaveOutcome
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)
    mgr.mark_processing_complete = AsyncMock()

    await mgr.finalize_source("src-abc", SaveOutcome.SUCCESS)

    mgr.mark_processing_complete.assert_awaited_once_with("src-abc")


@pytest.mark.asyncio
async def test_finalize_source_permanent_failure_marks_complete():
    """finalize_source(PERMANENT_FAILURE) marks complete: the input is
    poison; redeliveries should short-circuit via the L2 check. (Under
    VEPAGE-1250 this will write status='failed_permanent' instead.)"""
    from mirix.queue.error_policy import SaveOutcome
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)
    mgr.mark_processing_complete = AsyncMock()

    await mgr.finalize_source("src-abc", SaveOutcome.PERMANENT_FAILURE)

    mgr.mark_processing_complete.assert_awaited_once_with("src-abc")


@pytest.mark.asyncio
async def test_finalize_source_transient_exhausted_marks_complete():
    """finalize_source(TRANSIENT_EXHAUSTED) marks complete only when used
    on a path that has no redelivery (internal loop). Numaflow does NOT
    call finalize on TRANSIENT_EXHAUSTED — it re-raises for redelivery
    (see queue/__init__.py)."""
    from mirix.queue.error_policy import SaveOutcome
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)
    mgr.mark_processing_complete = AsyncMock()

    await mgr.finalize_source("src-abc", SaveOutcome.TRANSIENT_EXHAUSTED)

    mgr.mark_processing_complete.assert_awaited_once_with("src-abc")


@pytest.mark.asyncio
async def test_finalize_source_none_id_is_noop():
    """finalize_source with no source id is a safe no-op."""
    from mirix.queue.error_policy import SaveOutcome
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)
    mgr.mark_processing_complete = AsyncMock()

    await mgr.finalize_source(None, SaveOutcome.SUCCESS)

    mgr.mark_processing_complete.assert_not_awaited()


# ---------- deduped source: finalize must not log a spurious ERROR ----------
#
# When a source is deduped (external_id / batch_hash conflict — either caught
# up front or lost an INSERT ... ON CONFLICT race), the minted memory_source_id
# never became a persisted row (the *winning* row from the prior submission is
# already processing_complete=True). finalize_source still runs for the deduped
# id with outcome=SUCCESS — marking complete is the correct intent (incomplete
# = actively-processing ONLY) — but the underlying row doesn't exist, so the
# provider's update raises not-found. That must be a benign no-op, NOT an
# ERROR-level "Failed to finalize" line (which fails assert_no_error_logs in the
# idempotency FST). See VEPAGE-1228 expansion / VEPAGE-1165.


@pytest.mark.asyncio
async def test_mark_processing_complete_tolerates_missing_row():
    """mark_processing_complete on a row that doesn't exist (deduped loser)
    is a benign no-op — it must not raise."""
    from mirix.errors import ProviderNotFoundError
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)

    fake_provider = MagicMock()
    fake_provider.update = AsyncMock(side_effect=ProviderNotFoundError("memory_sources/src-x not found"))

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_provider,
    ):
        # Must not raise.
        await mgr.mark_processing_complete("src-deduped")

    fake_provider.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_processing_complete_reraises_real_update_errors():
    """A genuine update failure (not a missing row) must still propagate, so
    finalize_source's ERROR safety net still fires for real problems."""
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)

    fake_provider = MagicMock()
    fake_provider.update = AsyncMock(side_effect=RuntimeError("connection reset"))

    with patch(
        "mirix.database.relational_provider.get_relational_provider",
        return_value=fake_provider,
    ):
        with pytest.raises(RuntimeError):
            await mgr.mark_processing_complete("src-real")


@pytest.mark.asyncio
async def test_finalize_source_does_not_log_error_for_deduped_source():
    """End-to-end: finalize_source(SUCCESS) on a deduped (missing) source must
    complete without going through the logger.exception ERROR branch."""
    from mirix.errors import ProviderNotFoundError
    from mirix.queue.error_policy import SaveOutcome
    from mirix.services.memory_source_manager import MemorySourceManager

    mgr = MemorySourceManager.__new__(MemorySourceManager)

    fake_provider = MagicMock()
    fake_provider.update = AsyncMock(
        side_effect=ProviderNotFoundError("Entity memory_sources/src-x not found for update")
    )

    with (
        patch(
            "mirix.database.relational_provider.get_relational_provider",
            return_value=fake_provider,
        ),
        patch("mirix.services.memory_source_manager.logger") as mock_logger,
    ):
        await mgr.finalize_source("src-deduped", SaveOutcome.SUCCESS)

    mock_logger.exception.assert_not_called()


# ---------- LLM-path: complete = success only ----------


@pytest.mark.asyncio
async def test_llm_path_raises_chaining_exhausted_on_exhausted_function_failed():
    """When the LLM loop exits with function_failed=True (meta-agent emitted
    malformed tool calls past chaining budget), step() raises
    LLMChainingExhaustedError. classify() routes this to Bucket.PERMANENT so
    dispatch_save records SaveOutcome.PERMANENT_FAILURE. step() itself does
    NOT call finalize — that's the dispatcher's job (VEPAGE-1251 Option B)."""
    from mirix.errors import LLMChainingExhaustedError
    from mirix.schemas.message import MessageCreate

    agent, actor, user = _setup_agent("src-abc")

    failed_resp = _make_step_response(function_failed=True)
    agent.inner_step = AsyncMock(return_value=failed_resp)
    agent._extract_topics_from_messages = AsyncMock(return_value=["topic"])

    input_msg = MessageCreate(role="user", content="hello")

    with patch("mirix.agent.agent.LLMClient"):
        with pytest.raises(LLMChainingExhaustedError):
            await agent.step(
                input_messages=[input_msg],
                chaining=False,
                max_chaining_steps=1,
                stream=False,
                skip_verify=True,
                actor=actor,
                user=user,
            )

    # step() does not finalize on the failure path either.
    agent.memory_source_manager.finalize_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_path_does_not_finalize_on_clean_exit():
    """Under Option B, step() returns successfully WITHOUT calling
    finalize_source. dispatch_save is responsible for calling finalize on
    the post-policy SUCCESS verdict."""
    from mirix.schemas.message import MessageCreate

    agent, actor, user = _setup_agent("src-abc")

    ok_resp = _make_step_response(function_failed=False, continue_chaining=False)
    agent.inner_step = AsyncMock(return_value=ok_resp)
    agent._extract_topics_from_messages = AsyncMock(return_value=["topic"])

    input_msg = MessageCreate(role="user", content="hello")

    with patch("mirix.agent.agent.LLMClient"):
        await agent.step(
            input_messages=[input_msg],
            chaining=False,
            max_chaining_steps=1,
            stream=False,
            skip_verify=True,
            actor=actor,
            user=user,
        )

    agent.memory_source_manager.finalize_source.assert_not_awaited()
    agent.memory_source_manager.mark_processing_complete.assert_not_awaited()
