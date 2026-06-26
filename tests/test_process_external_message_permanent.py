"""Permanent-failure handling in mirix.queue.process_external_message.

When the underlying worker raises a Permanent-classified exception, the wrapper
must mark the memory source processing_complete and return normally so the
external consumer acks the message instead of redelivering.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from mirix.errors import LLMUnprocessableEntityError
from mirix.queue import process_external_message
from mirix.queue.message_pb2 import QueueMessage


@pytest.mark.asyncio
async def test_permanent_failure_marks_processing_complete_and_does_not_raise(monkeypatch):
    source_id = "src-permanent-test"

    queue_message = QueueMessage()
    queue_message.agent_id = "agent-test"
    queue_message.memory_source_id = source_id

    monkeypatch.setattr(
        "mirix.queue.queue_util.deserialize_queue_message",
        lambda raw, format=None: queue_message,
    )

    worker = Mock()
    worker.process_external_message = AsyncMock(side_effect=LLMUnprocessableEntityError("rejected by risk screening"))

    fake_manager = Mock()
    fake_manager.is_initialized = True
    fake_manager._workers = [worker]
    monkeypatch.setattr("mirix.queue._manager", fake_manager)

    # After VEPAGE-1251 S2 the permanent callback routes through the single
    # finalize chokepoint (MemorySourceManager.finalize_source), not directly
    # to mark_processing_complete.
    from mirix.queue.error_policy import SaveOutcome

    finalize = AsyncMock()
    fake_source_manager_cls = Mock(return_value=Mock(finalize_source=finalize))
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        fake_source_manager_cls,
    )

    # Permanent → returns normally (no raise).
    await process_external_message(b"ignored-by-stub-deserializer")

    finalize.assert_awaited_once_with(source_id, SaveOutcome.PERMANENT_FAILURE)
