"""
Abstract base class for queue implementations
Defines the interface that all queue implementations must follow
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)


class QueueInterface(ABC):
    """Abstract base class for queue implementations"""

    @abstractmethod
    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the queue

        Args:
            message: QueueMessage protobuf message to add to the queue
        """
        pass

    @abstractmethod
    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue

        Args:
            timeout: Optional timeout in seconds to wait for a message

        Returns:
            QueueMessage protobuf message from the queue

        Raises:
            Exception if no message available within timeout
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close connections"""
        pass
