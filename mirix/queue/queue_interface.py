"""
Abstract base class for queue implementations.
All queue methods are async-native.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)


class QueueInterface(ABC):
    """Abstract base class for async queue implementations"""

    async def start(self) -> None:
        """
        Start the queue (connect to brokers, etc.).
        No-op for in-memory queues; required for external systems like Kafka.
        Called by QueueManager after construction.
        """

    @abstractmethod
    async def put(self, message: QueueMessage) -> None:
        """
        Add a message to the queue.

        Args:
            message: QueueMessage protobuf message to add to the queue
        """
        ...

    @abstractmethod
    async def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue.

        Args:
            timeout: Optional timeout in seconds to wait for a message

        Returns:
            QueueMessage protobuf message from the queue

        Raises:
            asyncio.QueueEmpty or TimeoutError if no message within timeout
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources and close connections"""
        ...
