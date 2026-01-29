"""
Queue Manager - Handles queue initialization and lifecycle management
Separates implementation logic from the public API

Supports multiple workers for in-memory queue via NUM_WORKERS config.
When NUM_WORKERS > 1, uses PartitionedMemoryQueue to simulate Kafka's
user_id-based partitioning for parallel processing.
"""

import atexit
import logging
import time
from typing import Any, List, Optional

from mirix.log import get_logger
from mirix.queue import config
from mirix.queue.memory_queue import MemoryQueue, PartitionedMemoryQueue
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface
from mirix.queue.worker import QueueWorker

logger = get_logger(__name__)  # Use Mirix logger for proper configuration


class QueueManager:
    """
    Manages queue lifecycle and worker coordination
    Singleton pattern to ensure only one instance per application

    For in-memory queues with NUM_WORKERS > 1, creates a partitioned queue
    where messages are routed by user_id hash (simulating Kafka behavior).
    """

    def __init__(self):
        """Initialize the queue manager"""
        self._queue: Optional[QueueInterface] = None
        self._workers: List[QueueWorker] = []
        self._server: Optional[Any] = None
        self._initialized = False
        self._num_workers = 1
        self._round_robin = False

    def initialize(self, server: Optional[Any] = None) -> None:
        """
        Initialize the queue and start the background workers
        Creates appropriate queue type based on configuration

        This method is idempotent and thread-safe - calling it multiple times
        will only initialize once. The queue uses a singleton pattern.

        For in-memory queues:
        - NUM_WORKERS=1 (default): Single queue, single worker
        - NUM_WORKERS>1: Partitioned queue with N workers (simulates Kafka)

        Args:
            server: Optional server instance for workers to invoke APIs on
        """
        if self._initialized:
            logger.warning("Queue manager already initialized - skipping duplicate initialization")
            worker_count = len(self._workers)
            running_count = sum(1 for w in self._workers if w._running)
            logger.info(f"   Current state: workers={worker_count}, running={running_count}")
            # Allow updating server if provided
            if server:
                logger.info("Updating queue manager with server instance")
                self._server = server
                for worker in self._workers:
                    worker.set_server(server)
            return  # Already initialized

        # Determine number of workers (only for memory queue)
        if config.QUEUE_TYPE == "memory":
            self._num_workers = config.NUM_WORKERS
            self._round_robin = config.ROUND_ROBIN
        else:
            # Kafka handles its own partitioning/consumer groups
            self._num_workers = 1
            self._round_robin = False  # N/A for Kafka

        partition_mode = "round-robin" if self._round_robin else "hash"
        logger.info(
            "Initializing queue manager: type=%s, num_workers=%d, partitioning=%s, server=%s",
            config.QUEUE_TYPE,
            self._num_workers,
            partition_mode,
            "provided" if server else "None",
        )

        self._server = server

        # Create appropriate queue based on configuration
        logger.info("Creating queue instance...")
        self._queue = self._create_queue()
        logger.info(f"Queue created: type={type(self._queue).__name__}")

        # Create and start background workers
        self._workers = []

        if self._num_workers > 1 and isinstance(self._queue, PartitionedMemoryQueue):
            # Multiple workers with partitioned queue
            logger.info("👷 Creating %d background workers (partitioned)...", self._num_workers)
            for partition_id in range(self._num_workers):
                worker = QueueWorker(self._queue, server=self._server, partition_id=partition_id)
                self._workers.append(worker)
                logger.debug("Worker %d created", partition_id)
        else:
            # Single worker (default behavior)
            logger.info("👷 Creating single background worker...")
            worker = QueueWorker(self._queue, server=self._server)
            self._workers.append(worker)
            logger.debug("Worker created")

        # Start all workers (unless AUTO_START_WORKERS is disabled)
        if config.AUTO_START_WORKERS:
            logger.info("Starting %d background worker thread(s)...", len(self._workers))
            for worker in self._workers:
                worker.start()

            # Give threads a moment to start and verify they're running
            time.sleep(0.1)

            running_count = sum(1 for w in self._workers if w._running)
            alive_count = sum(1 for w in self._workers if w._thread and w._thread.is_alive())

            logger.info(
                f"Worker status: running={running_count}/{len(self._workers)}, threads_alive={alive_count}/{len(self._workers)}"
            )

            if running_count != len(self._workers) or alive_count != len(self._workers):
                logger.error("CRITICAL: Some queue workers failed to start!")
                logger.error(f"   Workers running: {running_count}/{len(self._workers)}")
                logger.error(f"   Threads alive: {alive_count}/{len(self._workers)}")
            else:
                logger.info("All %d queue worker(s) started successfully!", len(self._workers))
        else:
            logger.info(
                "Workers created but NOT started (AUTO_START_WORKERS=false) - "
                "Use process_external_message() to process messages from external consumer"
            )

        # Register cleanup function to stop workers on exit
        atexit.register(self.cleanup)

        self._initialized = True
        logger.info("Queue manager initialized successfully")

    def _create_queue(self) -> QueueInterface:
        """
        Factory method to create the appropriate queue implementation

        Returns:
            QueueInterface instance (MemoryQueue, PartitionedMemoryQueue, or KafkaQueue)

        Raises:
            ImportError: If required dependencies are not installed
        """
        if config.QUEUE_TYPE == "kafka":
            # Import Kafka queue (lazy import to avoid unnecessary dependency)
            try:
                from .kafka_queue import KafkaQueue

                # Build kwargs for KafkaQueue
                kafka_kwargs = {
                    "bootstrap_servers": config.KAFKA_BOOTSTRAP_SERVERS,
                    "topic": config.KAFKA_TOPIC,
                    "group_id": config.KAFKA_GROUP_ID,
                    "serialization_format": config.KAFKA_SERIALIZATION_FORMAT,
                    "security_protocol": config.KAFKA_SECURITY_PROTOCOL,
                    # Consumer configuration (configurable via env vars)
                    "auto_offset_reset": config.KAFKA_AUTO_OFFSET_RESET,
                    "consumer_timeout_ms": config.KAFKA_CONSUMER_TIMEOUT_MS,
                    "max_poll_interval_ms": config.KAFKA_MAX_POLL_INTERVAL_MS,
                    "session_timeout_ms": config.KAFKA_SESSION_TIMEOUT_MS,
                }

                # Add SSL parameters if provided
                if config.KAFKA_SSL_CAFILE:
                    kafka_kwargs["ssl_cafile"] = config.KAFKA_SSL_CAFILE
                if config.KAFKA_SSL_CERTFILE:
                    kafka_kwargs["ssl_certfile"] = config.KAFKA_SSL_CERTFILE
                if config.KAFKA_SSL_KEYFILE:
                    kafka_kwargs["ssl_keyfile"] = config.KAFKA_SSL_KEYFILE

                return KafkaQueue(**kafka_kwargs)
            except ImportError as e:
                raise ImportError(
                    f"Kafka queue requested but dependencies not installed: {e}\n"
                    "Install with: pip install queue-sample[kafka]"
                ) from e
        else:
            # In-memory queue
            if self._num_workers > 1:
                # Use partitioned queue for multiple workers
                mode = "round-robin" if self._round_robin else "hash"
                logger.info(
                    "Using PartitionedMemoryQueue with %d partitions, mode=%s",
                    self._num_workers,
                    mode,
                )
                return PartitionedMemoryQueue(
                    num_partitions=self._num_workers,
                    round_robin=self._round_robin,
                )
            else:
                # Default to simple in-memory queue
                return MemoryQueue()

    def save(self, message: QueueMessage) -> None:
        """
        Add a message to the queue

        Args:
            message: QueueMessage protobuf message to add to the queue

        Raises:
            RuntimeError: If the queue is not initialized
        """
        if self._queue is None:
            logger.error("Attempted to save message to uninitialized queue")
            raise RuntimeError("Queue not initialized. This should not happen - " "please report this as a bug.")

        logger.debug(
            "Saving message to queue: agent_id=%s, user_id=%s",
            message.agent_id,
            message.user_id if message.HasField("user_id") else "None",
        )

        # Delegate to the queue implementation
        # (PartitionedMemoryQueue will route based on user_id hash)
        self._queue.put(message)

    def cleanup(self) -> None:
        """
        Cleanup function called when the program exits
        Stops all workers and closes queue connections gracefully

        Note: No logging during cleanup to avoid errors when logging system
        has already shut down during Python/pytest teardown.
        """
        # Stop all workers (don't close queue until all workers stopped)
        for i, worker in enumerate(self._workers):
            # Only the last worker should close the shared queue
            is_last = i == len(self._workers) - 1
            worker.stop(close_queue=is_last)

        self._workers = []
        self._queue = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the manager is initialized"""
        return self._initialized

    @property
    def queue_type(self) -> str:
        """Get the current queue type (memory or kafka)"""
        return config.QUEUE_TYPE

    @property
    def num_workers(self) -> int:
        """Get the number of workers"""
        return self._num_workers

    @property
    def round_robin(self) -> bool:
        """Check if round-robin partitioning mode is enabled"""
        return self._round_robin

    def get_partition_stats(self) -> Optional[dict]:
        """
        Get statistics about partition distribution (memory queue only).
        Useful for debugging/benchmarking to verify even distribution.

        Returns:
            Dict with partition stats, or None if not using partitioned queue
        """
        if isinstance(self._queue, PartitionedMemoryQueue):
            return self._queue.get_partition_stats()
        return None


# Global singleton instance
_manager = QueueManager()


def get_manager() -> QueueManager:
    """
    Get the global queue manager instance

    Returns:
        QueueManager singleton instance
    """
    return _manager
