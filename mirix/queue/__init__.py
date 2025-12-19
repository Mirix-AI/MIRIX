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


# Export public API
__all__ = ["initialize_queue", "save", "QueueMessage"]
