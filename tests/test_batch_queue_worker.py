"""Unit tests for ``BatchQueueWorker``.

``BatchQueueWorker`` REPLACES the old serial ``QueueWorker._consume_loop``. Its
consume loop, per iteration:

  * accumulates up to ``READ_BATCH_SIZE`` messages from its queue/partition, OR
    flushes the partial batch once ``FLUSH_INTERVAL_MS`` has elapsed since the
    first message in the current batch,
  * then runs the batch through the shared ``mirix.queue.batch.process_batch``
    core, grouping by ``user_id`` with ``max_in_flight_users=MAX_IN_FLIGHT_USERS``,
  * where the per-message ``process`` wraps ``_process_message_async`` in
    ``dispatch_save`` — the SAME chokepoint the serial loop used (so batch-of-1
    is behaviorally serial: one message in → one ``dispatch_save`` → one
    ``finalize``).

``READ_BATCH_SIZE=1`` is the degenerate serial-equivalent; there is no 4th
"mode".
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mirix.queue.batch import BatchResult
from mirix.queue.memory_queue import MemoryQueue, PartitionedMemoryQueue
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.worker import BatchQueueWorker

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _message(agent_id: str = "agent-batch", *, user_id=None, source_id=None) -> QueueMessage:
    msg = QueueMessage()
    msg.agent_id = agent_id
    if user_id is not None:
        msg.user_id = user_id
    if source_id is not None:
        msg.memory_source_id = source_id
    return msg


# ============================================================================
# Batch accumulation: drains up to READ_BATCH_SIZE then dispatches
# ============================================================================


async def test_collect_batch_drains_up_to_read_batch_size(monkeypatch):
    """With READ_BATCH_SIZE=3 and >=3 messages waiting, one collect pulls
    exactly 3 (not more), without waiting on the flush timer."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 3)

    queue = MemoryQueue()
    for i in range(5):
        await queue.put(_message(f"agent-{i}", user_id=f"u{i}"))

    worker = BatchQueueWorker(queue, server=Mock())

    batch = await worker._collect_batch()

    assert len(batch) == 3
    assert [m.agent_id for m in batch] == ["agent-0", "agent-1", "agent-2"]


async def test_loop_runs_full_batch_through_process_batch(monkeypatch):
    """One loop iteration with READ_BATCH_SIZE=3 dispatches a batch of 3 through
    the shared process_batch core."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 3)
    monkeypatch.setattr("mirix.queue.config.MAX_IN_FLIGHT_USERS", 7)

    queue = MemoryQueue()
    for i in range(3):
        await queue.put(_message(f"agent-{i}", user_id=f"u{i}"))

    worker = BatchQueueWorker(queue, server=Mock())
    worker._process_message_async = AsyncMock(return_value=None)

    captured = {}

    async def fake_process_batch(items, *, user_key, process, max_in_flight_users, on_item_success=None):
        captured["count"] = len(items)
        captured["max_in_flight_users"] = max_in_flight_users
        # Drive the injected process so per-message dispatch is exercised.
        for it in items:
            await process(it)
        return BatchResult(succeeded_items=list(items))

    with patch("mirix.queue.worker.process_batch", side_effect=fake_process_batch):
        worker._running = True
        await worker._run_one_iteration()

    assert captured["count"] == 3
    assert captured["max_in_flight_users"] == 7
    # Per-message processing went through the real per-message path.
    assert worker._process_message_async.await_count == 3


# ============================================================================
# Flush: partial batch flushes after FLUSH_INTERVAL_MS
# ============================================================================


async def test_collect_batch_flushes_partial_after_flush_interval(monkeypatch):
    """READ_BATCH_SIZE=5 but only 2 messages available: collect returns the
    partial batch once FLUSH_INTERVAL_MS elapses rather than blocking forever."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 5)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 30)

    queue = MemoryQueue()
    await queue.put(_message("agent-0", user_id="u0"))
    await queue.put(_message("agent-1", user_id="u1"))

    worker = BatchQueueWorker(queue, server=Mock())

    batch = await asyncio.wait_for(worker._collect_batch(), timeout=2.0)

    # Flush fired with the partial batch (2 < READ_BATCH_SIZE=5).
    assert len(batch) == 2
    assert [m.agent_id for m in batch] == ["agent-0", "agent-1"]


async def test_collect_batch_empty_queue_returns_empty(monkeypatch):
    """Empty queue: collect returns an empty batch after the flush window (no
    message ever arrives) — idle no-op, no dispatch."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 5)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 20)

    queue = MemoryQueue()
    worker = BatchQueueWorker(queue, server=Mock())

    batch = await asyncio.wait_for(worker._collect_batch(), timeout=2.0)

    assert batch == []


async def test_loop_idle_on_empty_queue_does_not_dispatch(monkeypatch):
    """An empty queue iteration must NOT call process_batch (nothing to do)."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 5)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 20)

    queue = MemoryQueue()
    worker = BatchQueueWorker(queue, server=Mock())

    with patch("mirix.queue.worker.process_batch", new=AsyncMock()) as pb:
        worker._running = True
        await worker._run_one_iteration()

    pb.assert_not_called()


# ============================================================================
# READ_BATCH_SIZE=1 is the serial-equivalent
# ============================================================================


async def test_read_batch_size_one_is_serial_equivalent(monkeypatch):
    """READ_BATCH_SIZE=1: one message in → exactly one dispatch_save →
    exactly one finalize, identical to the old serial loop."""
    from mirix.queue.error_policy import SaveOutcome

    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 1)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 10)

    finalize = AsyncMock()
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        Mock(return_value=Mock(finalize_source=finalize)),
    )

    queue = MemoryQueue()
    await queue.put(_message("agent-solo", user_id="u0", source_id="src-solo"))

    worker = BatchQueueWorker(queue, server=Mock())
    worker._process_message_async = AsyncMock(return_value=None)

    worker._running = True
    await worker._run_one_iteration()

    # Exactly one message processed, exactly one finalize(SUCCESS).
    worker._process_message_async.assert_awaited_once()
    finalize.assert_awaited_once_with("src-solo", SaveOutcome.SUCCESS)


# ============================================================================
# user_key extraction: user_id when present, else None
# ============================================================================


async def test_user_key_uses_user_id_when_present():
    worker = BatchQueueWorker(MemoryQueue(), server=Mock())
    msg = _message(user_id="alice")
    assert worker._user_key(msg) == "alice"


async def test_user_key_is_none_when_user_id_absent():
    worker = BatchQueueWorker(MemoryQueue(), server=Mock())
    msg = _message()  # no user_id set
    assert worker._user_key(msg) is None


# ============================================================================
# Per-message dispatch wiring: dispatch_save called with the right source id
# ============================================================================


async def test_per_message_process_dispatches_with_memory_source_id(monkeypatch):
    """The per-message ``process`` passed to process_batch wraps
    _process_message_async in dispatch_save, with the message's
    memory_source_id — same wiring the serial loop had."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 1)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 10)

    queue = MemoryQueue()
    await queue.put(_message("agent-x", user_id="u0", source_id="src-xyz"))

    worker = BatchQueueWorker(queue, server=Mock())
    worker._process_message_async = AsyncMock(return_value=None)

    dispatch_mock = AsyncMock()
    with patch("mirix.queue.worker.dispatch_save", dispatch_mock):
        worker._running = True
        await worker._run_one_iteration()

    dispatch_mock.assert_awaited_once()
    assert dispatch_mock.await_args.kwargs["memory_source_id"] == "src-xyz"


# ============================================================================
# Lifecycle: start()/stop() task management (same contract as old worker)
# ============================================================================


async def test_start_stop_lifecycle():
    worker = BatchQueueWorker(MemoryQueue())

    await worker.start()
    assert worker._running is True
    assert worker._task is not None
    assert not worker._task.done()

    await worker.stop()
    assert worker._running is False
    assert worker._task is None


async def test_partition_id_drains_only_its_partition(monkeypatch):
    """A partitioned BatchQueueWorker reads only from its assigned partition."""
    monkeypatch.setattr("mirix.queue.config.READ_BATCH_SIZE", 5)
    monkeypatch.setattr("mirix.queue.config.FLUSH_INTERVAL_MS", 30)

    queue = PartitionedMemoryQueue(num_partitions=2)

    # Put 3 messages directly on partition 1, 0 on partition 0.
    for i in range(3):
        await queue._partitions[1].put(_message(f"agent-p1-{i}", user_id=f"u{i}"))

    worker = BatchQueueWorker(queue, server=Mock(), partition_id=1)

    batch = await asyncio.wait_for(worker._collect_batch(), timeout=2.0)

    assert len(batch) == 3
    assert all(m.agent_id.startswith("agent-p1-") for m in batch)
