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
from typing import Optional

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

    This is the primary high-level API for integrating with external Kafka consumers or event
    processing systems. It handles all internal details of deserialization and processing.

    Args:
        raw_message: Raw message bytes from Kafka or event bus (JSON or protobuf format)

    Raises:
        ValueError: If message parsing fails
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

    queue_message = deserialize_queue_message(raw_message, format=KAFKA_SERIALIZATION_FORMAT)

    logger.debug(
        "Processing external message (%s format): agent_id=%s, user_id=%s",
        KAFKA_SERIALIZATION_FORMAT,
        queue_message.agent_id,
        queue_message.user_id if queue_message.HasField("user_id") else "None",
    )

    await worker.process_external_message(queue_message)


__all__ = ["initialize_queue", "save", "process_external_message", "QueueMessage"]
