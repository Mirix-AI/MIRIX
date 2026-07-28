"""Tests for the unified save dispatcher (`dispatch_save`).

After VEPAGE-1251, all three run modes (numaflow, kafka, in-memory) route
through `error_policy.dispatch_save`. The dispatcher:

1. Runs the save under `process_with_policy` (classify + bounded retry).
2. Routes the verdict to the single finalize chokepoint
   (`MemorySourceManager.finalize_source`).

step() does NOT finalize internally anymore — the dispatcher handles ALL
outcomes including SUCCESS (Option B).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.errors import (
    LLMChainingExhaustedError,
    LLMRateLimitError,
    LLMUnprocessableEntityError,
)
from mirix.queue.error_policy import SaveOutcome, dispatch_save


@pytest.mark.asyncio
async def test_dispatch_save_success_finalizes_with_success():
    """A clean save returns SaveOutcome.SUCCESS and the dispatcher calls
    finalize_source(SUCCESS)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        return None

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-ok")

    assert outcome.kind is SaveOutcome.SUCCESS
    finalize.assert_awaited_once_with("src-ok", SaveOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_dispatch_save_permanent_finalizes_with_permanent():
    """A Permanent classification routes through finalize_source(PERMANENT_FAILURE)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMUnprocessableEntityError("422 rejected")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-perm")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    finalize.assert_awaited_once_with("src-perm", SaveOutcome.PERMANENT_FAILURE)


@pytest.mark.asyncio
async def test_dispatch_save_llm_chaining_exhausted_is_permanent(monkeypatch):
    """LLMChainingExhaustedError (raised by step() when function_failed
    exhausts chaining) classifies as PERMANENT and finalizes accordingly.
    This is the case where the meta-agent LLM produced malformed tool calls
    past its budget — retrying the whole step won't help."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMChainingExhaustedError("LLM gave up")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-chain")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    finalize.assert_awaited_once_with("src-chain", SaveOutcome.PERMANENT_FAILURE)


@pytest.mark.asyncio
async def test_dispatch_save_transient_exhausted_finalizes_with_transient(monkeypatch):
    """A transient that exhausts the retry budget returns
    TRANSIENT_EXHAUSTED and the dispatcher finalizes. No conscious
    redelivery — all 3 modes treat exhausted-transient as dead-letter."""
    from mirix.queue import error_policy as ep

    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        raise LLMRateLimitError("429 always")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-trans")

    assert outcome.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    finalize.assert_awaited_once_with("src-trans", SaveOutcome.TRANSIENT_EXHAUSTED)


@pytest.mark.asyncio
async def test_dispatch_save_without_source_id_skips_finalize():
    """No memory_source_id → finalize is not called (nothing to finalize)."""
    finalize = AsyncMock()
    mgr = MagicMock(finalize_source=finalize)

    async def _run():
        return None

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id=None)

    assert outcome.kind is SaveOutcome.SUCCESS
    finalize.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Fault-injection active-source lifecycle (batch safety).
#
# dispatch_save is the per-save boundary for ALL run modes. It publishes the
# save in flight on a ContextVar (read by injection hooks that lack a source-id
# handle) and MUST clear it afterward so that when one worker processes several
# messages sequentially in a single asyncio task (e.g. numaflow batching: a
# user-group loop), one message's active source can never leak into the next.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_dispatch_save_sets_active_source_during_run_and_clears_after():
    from mirix.testing import fault_injection as fi

    fi.reset()
    seen = {}

    async def _run():
        # Inside the save, the active source is visible to provider hooks.
        seen["during"] = fi.get_active_source()

    with (
        patch.object(fi.settings, "fault_injection_enabled", True),
        patch(
            "mirix.services.memory_source_manager.MemorySourceManager",
            return_value=MagicMock(finalize_source=AsyncMock()),
        ),
    ):
        await dispatch_save(_run, memory_source_id="src-active")
        # After the save returns, the ContextVar is cleared.
        assert fi.get_active_source() is None

    assert seen["during"] == "src-active"
    fi.reset()


@pytest.mark.asyncio
async def test_sequential_saves_do_not_leak_active_source():
    """Two saves run back-to-back in the SAME task (the numaflow per-user-group
    loop shape). A fault directive registered for src-1 must NOT fire when the
    provider boundary runs for src-2 — the active source is reset between saves.

    Asserts the leak-safety invariant directly: each save's provider boundary
    sees ITS OWN source, and the src-1 directive fires once (for src-1) and
    never for src-2. (We assert on fire counts + the observed active source,
    not on SaveOutcome, since a raw SyntheticProviderError raised in this stub
    bypasses the ECMS translation layer that would normally classify it.)"""
    from mirix.testing import fault_injection as fi

    fi.reset()
    observed = {}

    with (
        patch.object(fi.settings, "fault_injection_enabled", True),
        patch(
            "mirix.services.memory_source_manager.MemorySourceManager",
            return_value=MagicMock(finalize_source=AsyncMock()),
        ),
    ):
        # Only src-1 has a directive.
        fi.resolve_directives(
            "src-1",
            {"__fault_injection__": {"faults": [{"site": "relational_write", "shape": "permanent"}]}},
        )

        async def _run_src1():
            observed["src1_active"] = fi.get_active_source()
            try:
                # A provider write during src-1's save reads the active source.
                fi.maybe_raise("relational_write", source_key=fi.get_active_source(), tool="memory_sources")
            except fi.SyntheticProviderError:
                pass  # the fault fired for src-1 (expected); swallow so the save "succeeds"

        await dispatch_save(_run_src1, memory_source_id="src-1")

        async def _run_src2():
            # src-2 has NO directive; its provider write reads the active source,
            # which must be "src-2" (set by dispatch_save), NOT the stale "src-1".
            observed["src2_active"] = fi.get_active_source()
            fi.maybe_raise("relational_write", source_key=fi.get_active_source(), tool="memory_sources")

        await dispatch_save(_run_src2, memory_source_id="src-2")

    # Each save saw its own source — no leak across the sequential boundary.
    assert observed["src1_active"] == "src-1"
    assert observed["src2_active"] == "src-2"
    # The src-1 fault fired exactly once (for src-1) and never for src-2.
    assert fi.fire_count("src-1", "relational_write") == 1
    assert fi.fire_count("src-2", "relational_write") == 0
    # And the ContextVar is clean after the last save.
    assert fi.get_active_source() is None
    fi.reset()


@pytest.mark.asyncio
async def test_dispatch_save_keeps_tid_through_finalize_then_clears():
    """The step restores the message's TID into the task context (see
    worker._process_message_async). finalize_source's "Finalized
    memory_source=... outcome=..." log line is the save's terminal correlation
    signal, so the TID must STILL be set when finalize runs — and must be
    cleared once dispatch_save returns (the per-message boundary on a reused
    worker task). Clearing inside the step (the old behavior) left the
    finalize line stamped tid=-."""
    from mirix.observability.context import get_tid, set_tid

    seen = {}

    async def fake_finalize(source_id, outcome_kind):
        seen["tid_at_finalize"] = get_tid()

    mgr = MagicMock(finalize_source=AsyncMock(side_effect=fake_finalize))

    async def _run():
        # Simulates restore_trace_from_queue_message inside the worker step.
        set_tid("tid-dispatch-test")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        await dispatch_save(_run, memory_source_id="src-tid")

    assert seen["tid_at_finalize"] == "tid-dispatch-test"
    assert get_tid() is None


# --------------------------------------------------------------------------- #
# Save-outcome marker (R4) + write-count accumulator lifecycle (R3 AC4).
#
# dispatch_save is the single per-save boundary that (a) knows the outcome for
# ALL paths including refusals without source rows and (b) still holds the
# trace context (its finally clears it AFTER the marker fires). The marker is
# NOT gated on memory_source_id — only the DB finalize is.
# --------------------------------------------------------------------------- #


def _finalize_mgr():
    return MagicMock(finalize_source=AsyncMock())


@pytest.mark.asyncio
async def test_marker_fires_exactly_once_on_success():
    async def _run():
        return None

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span") as marker,
    ):
        await dispatch_save(_run, memory_source_id="src-ok")

    marker.assert_called_once_with(SaveOutcome.SUCCESS, "src-ok", error_type=None)


@pytest.mark.asyncio
async def test_marker_fires_exactly_once_on_permanent_with_error_type():
    async def _run():
        raise LLMUnprocessableEntityError("422 rejected")

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span") as marker,
    ):
        await dispatch_save(_run, memory_source_id="src-perm")

    marker.assert_called_once_with(SaveOutcome.PERMANENT_FAILURE, "src-perm", error_type="LLMUnprocessableEntityError")


@pytest.mark.asyncio
async def test_marker_fires_exactly_once_on_transient_exhausted(monkeypatch):
    from mirix.queue import error_policy as ep

    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    async def _run():
        raise LLMRateLimitError("429 always")

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span") as marker,
    ):
        await dispatch_save(_run, memory_source_id="src-trans")

    marker.assert_called_once_with(SaveOutcome.TRANSIENT_EXHAUSTED, "src-trans", error_type="LLMRateLimitError")


@pytest.mark.asyncio
async def test_no_intermediate_markers_across_in_policy_retries(monkeypatch):
    """Transient retries happen INSIDE process_with_policy — a save that fails
    transiently then succeeds gets exactly ONE marker, with the terminal
    outcome (AC1/AC3: markers only at terminal states)."""
    from mirix.queue import error_policy as ep

    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)
    attempts = {"n": 0}

    async def _run():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise LLMRateLimitError("429 transient")
        return None

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span") as marker,
    ):
        await dispatch_save(_run, memory_source_id="src-retry")

    assert attempts["n"] == 3
    marker.assert_called_once_with(SaveOutcome.SUCCESS, "src-retry", error_type=None)


@pytest.mark.asyncio
async def test_marker_fires_even_without_memory_source_id():
    """Refusal path: no source row, DB finalize skipped — the trace marker
    still fires (a refused save must not look in-flight forever)."""
    finalize = AsyncMock()

    async def _run():
        from mirix.errors import QueueMessageRejectedError

        raise QueueMessageRejectedError("missing client_id")

    with (
        patch(
            "mirix.services.memory_source_manager.MemorySourceManager",
            return_value=MagicMock(finalize_source=finalize),
        ),
        patch("mirix.queue.error_policy.emit_save_outcome_span") as marker,
    ):
        outcome = await dispatch_save(_run, memory_source_id=None)

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    finalize.assert_not_awaited()
    marker.assert_called_once_with(SaveOutcome.PERMANENT_FAILURE, None, error_type="QueueMessageRejectedError")


@pytest.mark.asyncio
async def test_marker_fires_before_trace_context_and_tid_clear():
    """The finally clears trace context + TID; the marker must fire while they
    are still set so it can attach to the right trace."""
    from mirix.observability.context import get_tid, get_trace_context, set_tid, set_trace_context

    seen = {}

    def _capture(outcome, memory_source_id, error_type=None):
        seen["tid"] = get_tid()
        seen["trace_id"] = get_trace_context().get("trace_id")

    async def _run():
        set_tid("tid-marker")
        set_trace_context(trace_id="trace-marker")

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span", side_effect=_capture),
    ):
        await dispatch_save(_run, memory_source_id="src-order")

    assert seen["tid"] == "tid-marker"
    assert seen["trace_id"] == "trace-marker"
    assert get_tid() is None
    assert get_trace_context().get("trace_id") is None


@pytest.mark.asyncio
async def test_write_counts_fresh_during_save_and_reset_after():
    """dispatch_save publishes a fresh write-count dict per save (R3 AC4) and
    resets it in its finally — on the success path."""
    from mirix.observability.trace_attrs import bump_write_count, get_write_counts

    seen = {}

    async def _run():
        bump_write_count("episodic")
        seen["during"] = get_write_counts()

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span"),
    ):
        await dispatch_save(_run, memory_source_id="src-wc")

    assert seen["during"] == {"episodic": 1}
    assert get_write_counts() == {}


@pytest.mark.asyncio
async def test_write_counts_reset_even_when_step_raises():
    from mirix.observability.trace_attrs import bump_write_count, get_write_counts

    async def _run():
        bump_write_count("core")
        raise LLMUnprocessableEntityError("422")

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span"),
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-wc-fail")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    assert get_write_counts() == {}


@pytest.mark.asyncio
async def test_sequential_saves_each_get_fresh_write_counts():
    """Batch-worker shape: two saves back-to-back in the same task must not
    see each other's counts."""
    from mirix.observability.trace_attrs import bump_write_count, get_write_counts

    seen = {}

    async def _run_1():
        bump_write_count("episodic")
        seen["first"] = get_write_counts()

    async def _run_2():
        seen["second_at_start"] = get_write_counts()
        bump_write_count("semantic")
        seen["second"] = get_write_counts()

    with (
        patch("mirix.services.memory_source_manager.MemorySourceManager", return_value=_finalize_mgr()),
        patch("mirix.queue.error_policy.emit_save_outcome_span"),
    ):
        await dispatch_save(_run_1, memory_source_id="src-1")
        await dispatch_save(_run_2, memory_source_id="src-2")

    assert seen["first"] == {"episodic": 1}
    assert seen["second_at_start"] == {}
    assert seen["second"] == {"semantic": 1}


@pytest.mark.asyncio
async def test_dispatch_save_clears_tid_even_when_step_raises_permanent():
    """The TID boundary holds on every exit path: a permanently-failing step
    still finalizes with the TID set and leaves the context clean after."""
    from mirix.observability.context import get_tid, set_tid

    seen = {}

    async def fake_finalize(source_id, outcome_kind):
        seen["tid_at_finalize"] = get_tid()

    mgr = MagicMock(finalize_source=AsyncMock(side_effect=fake_finalize))

    async def _run():
        set_tid("tid-perm-test")
        raise LLMUnprocessableEntityError("422 rejected")

    with patch(
        "mirix.services.memory_source_manager.MemorySourceManager",
        return_value=mgr,
    ):
        outcome = await dispatch_save(_run, memory_source_id="src-perm")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    assert seen["tid_at_finalize"] == "tid-perm-test"
    assert get_tid() is None
