"""
Mirix Queue - A lightweight queue-based messaging system

This module provides asynchronous message processing for the Mirix library.
The queue must be explicitly initialized by calling initialize_queue() with
a server instance.

Features:
- In-memory queue (default) or Kafka (via QUEUE_TYPE env var)
- Server integration for processing messages
- Thread-safe background worker

Usage:
    >>> from mirix.queue import initialize_queue, save, QueueMessage
    >>> from mirix.server.server import SyncServer
    >>>
    >>> # Initialize with server instance
    >>> server = SyncServer()
    >>> initialize_queue(server)
    >>>
    >>> # Enqueue messages
    >>> msg = QueueMessage()
    >>> msg.agent_id = "agent-123"
    >>> save(msg)  # Message will be processed asynchronously via server

The queue should be initialized when the REST API starts (in lifespan event).
"""

import logging
from typing import Optional

from mirix.queue.manager import get_manager
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)

# Version
__version__ = "0.1.0"

# Get the global manager instance (singleton)
_manager = get_manager()


def initialize_queue(server=None) -> None:
    """
    Initialize the queue with an optional server instance.

    The queue worker will invoke server.send_messages() when processing messages.
    This should be called during application startup (e.g., in FastAPI lifespan).

    Args:
        server: Server instance (e.g., SyncServer) for processing messages

    Example:
        >>> from mirix.server.server import SyncServer
        >>> from mirix.queue import initialize_queue
        >>>
        >>> server = SyncServer()
        >>> initialize_queue(server)
    """
    _manager.initialize(server=server)
    logger.info("Queue initialized with server instance")


def save(message: QueueMessage) -> None:
    """
    Add a message to the queue

    The message will be automatically processed by the background worker.

    Args:
        message: QueueMessage protobuf message to add to the queue

    Raises:
        RuntimeError: If the queue is not initialized

    Example:
        >>> import mirix.queue as queue
        >>> from mirix.queue.message_pb2 import QueueMessage
        >>> msg = QueueMessage()
        >>> msg.agent_id = "agent-123"
        >>> queue.save(msg)
    """
    if not _manager.is_initialized:
        logger.warning("Queue not initialized - call initialize_queue() first")
        # Auto-initialize without server for backward compatibility
        _manager.initialize()

    _manager.save(message)


def process_external_message(raw_message: bytes) -> None:
    """
    Process a message consumed by an external system (e.g., Numaflow, custom Kafka consumer).

    This is the primary high-level API for integrating with external Kafka consumers or event
    processing systems. It handles all internal details of deserialization and processing,
    providing a clean abstraction layer for external applications like ECMS.

    Args:
        raw_message: Raw message bytes from Kafka or event bus (JSON or protobuf format)

    Raises:
        ValueError: If message parsing fails

    Example:
        >>> # In your Numaflow handler or external consumer
        >>> from mirix.queue import process_external_message
        >>>
        >>> def handle_kafka_message(raw_bytes: bytes):
        >>>     process_external_message(raw_bytes)

    Note:
        - Queue is auto-initialized if not already initialized (with server instance)
        - Set MIRIX_QUEUE_AUTO_START_WORKERS=false to disable internal Kafka consumer
        - Format auto-detection uses KAFKA_SERIALIZATION_FORMAT (json/protobuf)
        - This function is thread-safe and can be called from multiple threads
        - The Kafka producer remains functional for enqueueing messages via save()

    Configuration:
        To use with external consumers (Numaflow, etc.), set environment variables:

        >>> import os
        >>> os.environ["MIRIX_QUEUE_AUTO_START_WORKERS"] = "false"
        >>> os.environ["KAFKA_SERIALIZATION_FORMAT"] = "json"  # or "protobuf"
        >>> # Queue will be auto-initialized on first call to process_external_message()
    """
    # Auto-initialize queue with server if not already initialized
    if not _manager.is_initialized:
        logger.info("Queue not initialized, auto-initializing with server for external message processing")
        # Import here to avoid circular dependency
        from mirix.server.server import SyncServer

        server = SyncServer()
        _manager.initialize(server=server)
        logger.info("Queue initialized with server instance")

    # Get the worker (created but not started if AUTO_START_WORKERS=false)
    workers = _manager._workers
    if not workers:
        logger.error("No workers available after initialization - this should not happen!")
        raise RuntimeError("Failed to create queue workers during initialization")

    worker = workers[0]

    # Deserialize message using configured format
    from mirix.queue.config import KAFKA_SERIALIZATION_FORMAT
    from mirix.queue.queue_util import deserialize_queue_message

    queue_message = deserialize_queue_message(raw_message, format=KAFKA_SERIALIZATION_FORMAT)

    logger.debug(
        "Processing external message (%s format): agent_id=%s, user_id=%s",
        KAFKA_SERIALIZATION_FORMAT,
        queue_message.agent_id,
        queue_message.user_id if queue_message.HasField("user_id") else "None",
    )

    # Delegate to worker for processing
    worker.process_external_message(queue_message)


# Export public API
__all__ = ["initialize_queue", "save", "process_external_message", "QueueMessage"]
