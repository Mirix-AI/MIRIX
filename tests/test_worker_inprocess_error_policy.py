"""Worker error-handling contract for the internal in-memory consumer path.

After the batch-consumer consolidation (serial loop replaced by the
BatchQueueWorker):

  * ``_process_message_async`` (the shared per-message core) does NOT classify
    or mark the source complete — it runs the agent step and RE-RAISES on any
    failure.
  * Both the external (numaflow) path and the internal BatchQueueWorker use the
    shared `error_policy.dispatch_save` helper, which runs
    `process_with_policy` then routes the verdict to the single finalize
    chokepoint.
  * step() does not finalize internally — `dispatch_save` calls
    finalize_source for all outcomes including SUCCESS (Option B).

These tests pin the core's re-raise contract and the batch consumer's
delegation to dispatch_save (per message inside process_batch).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mirix.queue.memory_queue import MemoryQueue
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import BatchQueueWorker, QueueWorker


def _message(source_id: str | None) -> QueueMessage:
    m = QueueMessage()
    m.agent_id = "agent-test"
    m.client_id = "client-test"
    m.user_id = "user-test"
    if source_id is not None:
        m.memory_source_id = source_id
    return m


# ---------------------------------------------------------------------------
# Core re-raises (so dispatch_save's process_with_policy can classify)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_core_reraises_on_failure(monkeypatch):
    """_process_message_async must propagate, not swallow, processing failures."""
    actor = SimpleNamespace(id="client-test", organization_id="org-test", write_scope="test-scope")
    server = Mock()
    server.client_manager = Mock(get_client_by_id=AsyncMock(return_value=actor))
    server.send_messages = AsyncMock(side_effect=RuntimeError("boom"))

    user = SimpleNamespace(id="user-test", organization_id="org-test")
    monkeypatch.setattr(
        "mirix.queue.worker.UserManager",
        Mock(return_value=Mock(get_user_by_id=AsyncMock(return_value=user))),
    )

    worker = QueueWorker(queue=Mock(), server=server)

    with pytest.raises(RuntimeError, match="boom"):
        await worker._process_message_async(_message("src-x"))


# ---------------------------------------------------------------------------
# Batch consumer: delegates to dispatch_save (per message in process_batch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_loop_delegates_to_dispatch_save(monkeypatch):
    """One iteration pulls a message, then (inside process_batch) calls
    dispatch_save with a run_step closure that wraps _process_message_async.
    dispatch_save handles classification, retry, and finalize."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 1)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 10)

    source_id = "src-loop"
    queue = MemoryQueue()
    await queue.put(_message(source_id))

    worker = BatchQueueWorker(queue=queue, server=Mock())
    monkeypatch.setattr(worker, "_process_message_async", AsyncMock(return_value=None))

    dispatch_mock = AsyncMock()
    with patch("mirix.queue.worker.dispatch_save", dispatch_mock):
        worker._running = True
        await worker._run_one_iteration()

    dispatch_mock.assert_awaited_once()
    # Called with run_step closure and the source_id.
    kwargs = dispatch_mock.await_args.kwargs
    assert kwargs["memory_source_id"] == source_id


@pytest.mark.asyncio
async def test_consume_loop_continues_on_unexpected_loop_body_error(monkeypatch):
    """If something other than dispatch_save raises in the loop body
    (e.g. a queue protocol error during collect), the loop swallows and keeps
    running. No finalize call happens because we don't know the save's outcome."""
    worker = BatchQueueWorker(queue=Mock(), server=Mock())

    call_count = {"n": 0}

    async def fake_run_one_iteration():
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First iteration: deliberately raise something unrecognized to
            # exercise the bare-except safety net.
            raise RuntimeError("queue protocol blew up")
        # Second iteration: stop the loop.
        worker._running = False

    monkeypatch.setattr(worker, "_run_one_iteration", fake_run_one_iteration)

    worker._running = True
    # Must not raise — the safety-net except swallows and logs.
    await worker._batch_consume_loop()
    assert call_count["n"] >= 1
