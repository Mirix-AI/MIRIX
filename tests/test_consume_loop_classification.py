"""The in-memory batch consumer dispatches every
save through `dispatch_save`.

Same flow as the numaflow path — and now through the BatchQueueWorker (which
REPLACED the serial _consume_loop). With READ_BATCH_SIZE=1 a single message is
a batch of one user-group through the SAME dispatch_save → finalize chokepoint,
so these outcome assertions are unchanged:

- SUCCESS              → finalize_source(SUCCESS) (Option B: step() doesn't
                         finalize internally; the dispatcher does).
- PERMANENT_FAILURE    → finalize_source(PERMANENT_FAILURE).
- TRANSIENT_EXHAUSTED  → finalize_source(TRANSIENT_EXHAUSTED). No
                         conscious redelivery; the policy already
                         retried in-process up to budget.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from mirix.errors import LLMServerError, LLMUnprocessableEntityError
from mirix.queue.memory_queue import MemoryQueue
from mirix.queue.worker import BatchQueueWorker

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _message(source_id):
    """Build a minimal QueueMessage with optional memory_source_id."""
    from mirix.queue.message_pb2 import QueueMessage

    msg = QueueMessage()
    msg.agent_id = "agent-loop"
    if source_id is not None:
        msg.memory_source_id = source_id
    return msg


async def _worker_with_one_message(monkeypatch, msg) -> BatchQueueWorker:
    """A BatchQueueWorker in serial-equivalent mode (READ_BATCH_SIZE=1) whose
    queue holds exactly ``msg``. One _run_one_iteration() processes it as a
    batch of one — the degenerate serial case."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 1)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 10)

    queue = MemoryQueue()
    await queue.put(msg)

    worker = BatchQueueWorker(queue=queue, server=Mock())
    worker._running = True
    return worker


@pytest.mark.asyncio
async def test_consume_loop_permanent_calls_finalize_with_permanent(monkeypatch):
    """A Permanent classification (422 from the LLM) finalizes with
    PERMANENT_FAILURE — not generic 'mark complete'."""
    from mirix.queue.error_policy import SaveOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )

    source_id = "src-permanent"
    worker = await _worker_with_one_message(monkeypatch, _message(source_id))
    worker._process_message_async = AsyncMock(side_effect=LLMUnprocessableEntityError("422 rejected"))

    await worker._run_one_iteration()

    finalize.assert_awaited()
    args = finalize.call_args
    assert args.args[0] == source_id
    assert args.args[1] == SaveOutcome.PERMANENT_FAILURE


@pytest.mark.asyncio
async def test_consume_loop_transient_retried_then_finalized_exhausted(monkeypatch):
    """A sustained transient (LLM 5xx) runs through process_with_policy's
    retry loop. After exhaustion the batch consumer finalizes with
    TRANSIENT_EXHAUSTED (no redelivery on this path)."""
    from mirix.queue.error_policy import SaveOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )
    # Make backoff sleeps zero-time so the test runs fast.
    monkeypatch.setattr("mirix.queue.error_policy.asyncio.sleep", AsyncMock())

    source_id = "src-transient"
    worker = await _worker_with_one_message(monkeypatch, _message(source_id))
    proc = AsyncMock(side_effect=LLMServerError("503 still"))
    worker._process_message_async = proc

    await worker._run_one_iteration()

    # Sustained transient: the policy retries to budget (more than 1
    # attempt). The exact count depends on whole_step_retry_max_attempts;
    # asserting >= 2 is enough to prove the retry loop ran.
    assert proc.await_count >= 2

    # And then the dispatcher finalizes with the right outcome.
    finalize.assert_awaited()
    args = finalize.call_args
    assert args.args[0] == source_id
    assert args.args[1] == SaveOutcome.TRANSIENT_EXHAUSTED


@pytest.mark.asyncio
async def test_consume_loop_success_finalizes_with_success(monkeypatch):
    """Successful completion: dispatch_save calls finalize_source(SUCCESS).
    step() does NOT finalize internally (Option B); the dispatcher is the
    single finalize call site."""
    from mirix.queue.error_policy import SaveOutcome

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )

    worker = await _worker_with_one_message(monkeypatch, _message("src-ok"))
    worker._process_message_async = AsyncMock(return_value=None)

    await worker._run_one_iteration()

    finalize.assert_awaited_once_with("src-ok", SaveOutcome.SUCCESS)
