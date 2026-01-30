"""
In-memory queue implementation using Python's queue.Queue
Thread-safe and suitable for single-process applications

Includes PartitionedMemoryQueue for simulating Kafka-like partitioning
where messages are routed by user_id to ensure per-user ordering.
"""

import logging
import queue
import threading
from typing import Dict, List, Optional

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface

logger = logging.getLogger(__name__)


class MemoryQueue(QueueInterface):
    """Thread-safe in-memory queue implementation (single partition)"""

    def __init__(self):
        """Initialize the in-memory queue"""
        self._queue = queue.Queue()

    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the in-memory queue

        Args:
            message: QueueMessage protobuf message to add
        """
        logger.debug("Adding message to queue: agent_id=%s", message.agent_id)
        self._queue.put(message)

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue

        Args:
            timeout: Optional timeout in seconds (None = block indefinitely)

        Returns:
            QueueMessage protobuf message from the queue

        Raises:
            queue.Empty: If no message available within timeout
        """
        message = self._queue.get(timeout=timeout)
        logger.debug("Retrieved message from queue: agent_id=%s", message.agent_id)
        return message

    def close(self) -> None:
        """
        Clean up resources
        For in-memory queue, no cleanup is needed
        """
        pass


class PartitionedMemoryQueue(QueueInterface):
    """
    Partitioned in-memory queue that simulates Kafka's partitioning behavior.

    Messages are routed to partitions based on user_id hash, ensuring:
    - All messages for the same user go to the same partition
    - Each partition is consumed by exactly one worker
    - Per-user message ordering is preserved (same as Kafka behavior)

    This allows parallel processing across users while maintaining
    serial processing within each user's message stream.

    Partitioning modes:
    - Hash (default): Uses hash(key) % num_partitions (Kafka-like)
    - Round-robin: Sequential assignment (user1->p0, user2->p1, ...)
      Guarantees even distribution - perfect for benchmarks with known user counts
    """

    def __init__(self, num_partitions: int = 1, round_robin: bool = False):
        """
        Initialize the partitioned queue

        Args:
            num_partitions: Number of partitions (should match NUM_WORKERS)
            round_robin: If True, use round-robin assignment instead of hash.
                        Guarantees even distribution (user1->p0, user2->p1, ...).
                        Perfect for benchmarks where you know exact user counts.
        """
        self._num_partitions = max(1, num_partitions)
        self._round_robin = round_robin
        self._partitions: List[queue.Queue] = [queue.Queue() for _ in range(self._num_partitions)]

        # For even partitioning: track user -> partition assignments
        self._user_partition_map: Dict[str, int] = {}
        self._next_partition: int = 0
        self._partition_lock = threading.Lock()

        mode = "round-robin" if round_robin else "hash"
        logger.info(
            "Initialized PartitionedMemoryQueue with %d partitions, mode=%s",
            self._num_partitions,
            mode,
        )

    @property
    def num_partitions(self) -> int:
        """Get the number of partitions"""
        return self._num_partitions

    @property
    def round_robin(self) -> bool:
        """Check if round-robin partitioning mode is enabled"""
        return self._round_robin

    def get_partition_stats(self) -> Dict[str, any]:
        """
        Get statistics about partition distribution.
        Useful for debugging/benchmarking.

        Returns:
            Dict with partition stats including user counts per partition
        """
        with self._partition_lock:
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
        """
        Extract partition key from message (mirrors KafkaQueue behavior)

        Args:
            message: QueueMessage to extract key from

        Returns:
            Partition key string (user_id, client_id)
        """
        # Match KafkaQueue's partition key logic from kafka_queue.py
        if message.HasField("user_id") and message.user_id:
            return message.user_id
        elif message.client_id:
            return message.client_id
        else:
            # Fallback to agent_id if no user context
            return message.agent_id

    def _compute_partition(self, partition_key: str) -> int:
        """
        Compute partition number from key.

        Behavior depends on partitioning mode:
        - Hash-based (default): hash(key) % num_partitions (Kafka-like)
        - Even partitioning: Round-robin assignment for new keys

        Args:
            partition_key: Key to route

        Returns:
            Partition index (0 to num_partitions-1)
        """
        if not self._round_robin:
            # Default: hash-based partitioning (Kafka behavior)
            return hash(partition_key) % self._num_partitions

        # Even partitioning: round-robin assignment for new users
        with self._partition_lock:
            if partition_key not in self._user_partition_map:
                # Assign new user to next partition in sequence
                assigned_partition = self._next_partition
                self._user_partition_map[partition_key] = assigned_partition
                self._next_partition = (self._next_partition + 1) % self._num_partitions
                logger.debug(
                    "Round-robin: assigned %s -> partition %d",
                    partition_key,
                    assigned_partition,
                )
            return self._user_partition_map[partition_key]

    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the appropriate partition based on user_id

        Args:
            message: QueueMessage protobuf message to add
        """
        partition_key = self._get_partition_key(message)
        partition_id = self._compute_partition(partition_key)

        logger.debug(
            "Routing message to partition %d: agent_id=%s, partition_key=%s",
            partition_id,
            message.agent_id,
            partition_key,
        )

        self._partitions[partition_id].put(message)

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from partition 0 (for backward compatibility)

        Note: For partitioned queues, use get_from_partition() instead.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            QueueMessage from partition 0

        Raises:
            queue.Empty: If no message available within timeout
        """
        return self.get_from_partition(0, timeout)

    def get_from_partition(self, partition_id: int, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from a specific partition

        Args:
            partition_id: Partition to consume from (0 to num_partitions-1)
            timeout: Optional timeout in seconds (None = block indefinitely)

        Returns:
            QueueMessage protobuf message from the specified partition

        Raises:
            queue.Empty: If no message available within timeout
            ValueError: If partition_id is out of range
        """
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError(f"Invalid partition_id {partition_id}, " f"must be 0 to {self._num_partitions - 1}")

        message = self._partitions[partition_id].get(timeout=timeout)
        logger.debug(
            "Retrieved message from partition %d: agent_id=%s",
            partition_id,
            message.agent_id,
        )
        return message

    def close(self) -> None:
        """
        Clean up resources
        For in-memory queue, no cleanup is needed
        """
        pass
