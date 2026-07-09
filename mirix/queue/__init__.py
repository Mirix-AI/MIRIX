"""
Mirix Queue - Async-native message processing system.

This module provides asynchronous message processing for the Mirix library.
The queue must be explicitly initialized by calling initialize_queue() with
a server instance.

Features:
- In-memory async queue (default) or Kafka (via QUEUE_TYPE env var)
- Server integration for processing messages
- asyncio.Task-based background workers

Usage:
    >>> from mirix.queue import initialize_queue, save, QueueMessage
    >>> from mirix.server.server import AsyncServer
    >>>
    >>> # Initialize with server instance (call from async context)
    >>> server = AsyncServer()
    >>> await initialize_queue(server)
    >>>
    >>> # Enqueue messages
    >>> msg = QueueMessage()
    >>> msg.agent_id = "agent-123"
    >>> await save(msg)  # Message will be processed asynchronously via server

The queue should be initialized when the REST API starts (in lifespan event).
"""

import logging

from mirix.queue.manager import get_manager
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

_manager = get_manager()


async def initialize_queue(server=None) -> None:
    """
    Initialize the queue with an optional server instance.

    The queue worker will invoke server.send_messages() when processing messages.
    This should be called during application startup (e.g., in FastAPI lifespan).

    Args:
        server: Server instance for processing messages
    """
    await _manager.initialize(server=server)
    logger.info("Queue initialized with server instance")


async def save(message: QueueMessage) -> None:
    """
    Add a message to the queue.

    The message will be automatically processed by the background worker task.

    Args:
        message: QueueMessage protobuf message to add to the queue

    Raises:
        RuntimeError: If the queue is not initialized
    """
    if not _manager.is_initialized:
        logger.warning("Queue not initialized - call initialize_queue() first")
        await _manager.initialize()

    await _manager.save(message)


async def process_external_message(raw_message: bytes) -> None:
    """
    Process a message consumed by an external system (e.g., Numaflow, custom Kafka consumer).

    Delegates to the shared `dispatch_save` helper — same flow as the
    internal `_consume_loop`. Returns normally on every classified outcome
    so the consumer always acks. We do NOT consciously re-raise to invoke
    broker redelivery; the in-process retry budget already handled transient
    cases, and broker redelivery is reserved for process-death recovery
    (un-ack'd messages on crash).
    """
    if not _manager.is_initialized:
        logger.info("Queue not initialized, auto-initializing with server for external message processing")
        from mirix.server.server import AsyncServer

        server = AsyncServer()
        await _manager.initialize(server=server)
        logger.info("Queue initialized with server instance")

    workers = _manager._workers
    if not workers:
        logger.error("No workers available after initialization - this should not happen!")
        raise RuntimeError("Failed to create queue workers during initialization")

    worker = workers[0]

    from mirix.queue.config import KAFKA_SERIALIZATION_FORMAT
    from mirix.queue.queue_util import deserialize_queue_message
    from mirix.queue.worker import dispatch_incoming_message

    try:
        queue_message = deserialize_queue_message(raw_message, format=KAFKA_SERIALIZATION_FORMAT)
    except ValueError:
        # A message that can't even be deserialized is a poison pill: raising
        # here (before dispatch_save) would make the broker redeliver it
        # forever, wedging the partition — this is exactly how the ECMS-73
        # schema-skew incident presented. Deterministically-bad bytes can never
        # succeed on retry, so log loudly and ack. There is no memory_source_id
        # to finalize (the message never parsed), so the drop is visible only
        # here — keep this log ERROR and structured enough to alert on.
        logger.error(
            "Dropping undeserializable queue message (%s format, %d bytes) — acking to avoid a redelivery loop",
            KAFKA_SERIALIZATION_FORMAT,
            len(raw_message),
            exc_info=True,
        )
        return

    logger.debug(
        "Processing external message (%s format): agent_id=%s, user_id=%s, memory_source_id=%s",
        KAFKA_SERIALIZATION_FORMAT,
        queue_message.agent_id,
        queue_message.user_id if queue_message.HasField("user_id") else "None",
        queue_message.memory_source_id if queue_message.HasField("memory_source_id") else None,
    )

    # Normalize/validate + dispatch through the shared funnel — the same one
    # the internal kafka-manual and in-memory consumers use, so the
    # producer-facing contract (filter_tags shape, id/tid backfill,
    # malformed-message dead-letter) holds identically in every run mode.
    await dispatch_incoming_message(worker, queue_message)


__all__ = ["initialize_queue", "save", "process_external_message", "QueueMessage"]
