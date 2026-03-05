"""
Async-native in-memory queue implementation using asyncio.Queue.

Includes PartitionedMemoryQueue for simulating Kafka-like partitioning
where messages are routed by user_id to ensure per-user ordering.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface

logger = logging.getLogger(__name__)


class MemoryQueue(QueueInterface):
    """Async in-memory queue implementation (single partition)"""

    def __init__(self):
        self._queue: asyncio.Queue[QueueMessage] = asyncio.Queue()

    async def put(self, message: QueueMessage) -> None:
        logger.debug("Adding message to queue: agent_id=%s", message.agent_id)
        await self._queue.put(message)

    async def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue.

        Args:
            timeout: Optional timeout in seconds (None = block indefinitely)

        Returns:
            QueueMessage from the queue

        Raises:
            asyncio.TimeoutError: If no message available within timeout
        """
        if timeout is not None:
            message = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            message = await self._queue.get()
        logger.debug("Retrieved message from queue: agent_id=%s", message.agent_id)
        return message

    async def close(self) -> None:
        pass


class PartitionedMemoryQueue(QueueInterface):
    """
    Partitioned async in-memory queue that simulates Kafka's partitioning.

    Messages are routed to partitions based on user_id hash, ensuring:
    - All messages for the same user go to the same partition
    - Each partition is consumed by exactly one worker task
    - Per-user message ordering is preserved (same as Kafka behavior)

    Partitioning modes:
    - Hash (default): Uses hash(key) % num_partitions (Kafka-like)
    - Round-robin: Sequential assignment (user1->p0, user2->p1, ...)
    """

    def __init__(self, num_partitions: int = 1, round_robin: bool = False):
        self._num_partitions = max(1, num_partitions)
        self._round_robin = round_robin
        self._partitions: List[asyncio.Queue[QueueMessage]] = [
            asyncio.Queue() for _ in range(self._num_partitions)
        ]

        self._user_partition_map: Dict[str, int] = {}
        self._next_partition: int = 0
        self._partition_lock = asyncio.Lock()

        mode = "round-robin" if round_robin else "hash"
        logger.info(
            "Initialized PartitionedMemoryQueue with %d partitions, mode=%s",
            self._num_partitions,
            mode,
        )

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @property
    def round_robin(self) -> bool:
        return self._round_robin

    async def get_partition_stats(self) -> Dict[str, any]:
        """Get statistics about partition distribution."""
        async with self._partition_lock:
            partition_counts = [0] * self._num_partitions
            for partition_id in self._user_partition_map.values():
                partition_counts[partition_id] += 1

            return {
                "mode": "round-robin" if self._round_robin else "hash",
                "num_partitions": self._num_partitions,
                "total_users": len(self._user_partition_map),
                "users_per_partition": partition_counts,
                "queue_sizes": [p.qsize() for p in self._partitions],
            }

    def _get_partition_key(self, message: QueueMessage) -> str:
        if message.HasField("user_id") and message.user_id:
            return message.user_id
        elif message.client_id:
            return message.client_id
        else:
            return message.agent_id

    async def _compute_partition(self, partition_key: str) -> int:
        if not self._round_robin:
            return hash(partition_key) % self._num_partitions

        async with self._partition_lock:
            if partition_key not in self._user_partition_map:
                assigned_partition = self._next_partition
                self._user_partition_map[partition_key] = assigned_partition
                self._next_partition = (self._next_partition + 1) % self._num_partitions
                logger.debug(
                    "Round-robin: assigned %s -> partition %d",
                    partition_key,
                    assigned_partition,
                )
            return self._user_partition_map[partition_key]

    async def put(self, message: QueueMessage) -> None:
        partition_key = self._get_partition_key(message)
        partition_id = await self._compute_partition(partition_key)

        logger.debug(
            "Routing message to partition %d: agent_id=%s, partition_key=%s",
            partition_id,
            message.agent_id,
            partition_key,
        )

        await self._partitions[partition_id].put(message)

    async def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """Retrieve from partition 0 (for backward compatibility)."""
        return await self.get_from_partition(0, timeout)

    async def get_from_partition(
        self, partition_id: int, timeout: Optional[float] = None
    ) -> QueueMessage:
        """
        Retrieve a message from a specific partition.

        Args:
            partition_id: Partition to consume from (0 to num_partitions-1)
            timeout: Optional timeout in seconds

        Raises:
            asyncio.TimeoutError: If no message available within timeout
            ValueError: If partition_id is out of range
        """
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError(
                f"Invalid partition_id {partition_id}, "
                f"must be 0 to {self._num_partitions - 1}"
            )

        if timeout is not None:
            message = await asyncio.wait_for(
                self._partitions[partition_id].get(), timeout=timeout
            )
        else:
            message = await self._partitions[partition_id].get()

        logger.debug(
            "Retrieved message from partition %d: agent_id=%s",
            partition_id,
            message.agent_id,
        )
        return message

    async def close(self) -> None:
        pass
