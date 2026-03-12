"""
Queue Manager - Handles queue initialization and lifecycle management.
Async-native: workers run as asyncio.Tasks in the main event loop.

Supports multiple workers for in-memory queue via NUM_WORKERS config.
When NUM_WORKERS > 1, uses PartitionedMemoryQueue to simulate Kafka's
user_id-based partitioning for parallel processing.
"""

import logging
from typing import Any, List, Optional

from mirix.log import get_logger
from mirix.queue import config
from mirix.queue.memory_queue import MemoryQueue, PartitionedMemoryQueue
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface
from mirix.queue.worker import QueueWorker

logger = get_logger(__name__)


class QueueManager:
    """
    Manages queue lifecycle and worker coordination (async-native).
    Singleton pattern to ensure only one instance per application.
    """

    def __init__(self):
        self._queue: Optional[QueueInterface] = None
        self._workers: List[QueueWorker] = []
        self._server: Optional[Any] = None
        self._initialized = False
        self._num_workers = 1
        self._round_robin = False

    async def initialize(self, server: Optional[Any] = None) -> None:
        """
        Initialize the queue and start the background workers.

        This method is idempotent — calling it multiple times only initializes once.

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
            if server:
                logger.info("Updating queue manager with server instance")
                self._server = server
                for worker in self._workers:
                    worker.set_server(server)
            return

        if config.QUEUE_TYPE == "memory":
            self._num_workers = config.NUM_WORKERS
            self._round_robin = config.ROUND_ROBIN
        else:
            self._num_workers = 1
            self._round_robin = False

        partition_mode = "round-robin" if self._round_robin else "hash"
        logger.info(
            "Initializing queue manager: type=%s, num_workers=%d, partitioning=%s, server=%s",
            config.QUEUE_TYPE,
            self._num_workers,
            partition_mode,
            "provided" if server else "None",
        )

        self._server = server

        logger.info("Creating queue instance...")
        self._queue = self._create_queue()
        logger.info(f"Queue created: type={type(self._queue).__name__}")

        await self._queue.start()

        self._workers = []

        if self._num_workers > 1 and isinstance(self._queue, PartitionedMemoryQueue):
            logger.info("Creating %d background workers (partitioned)...", self._num_workers)
            for partition_id in range(self._num_workers):
                worker = QueueWorker(self._queue, server=self._server, partition_id=partition_id)
                self._workers.append(worker)
                logger.debug("Worker %d created", partition_id)
        else:
            logger.info("Creating single background worker...")
            worker = QueueWorker(self._queue, server=self._server)
            self._workers.append(worker)
            logger.debug("Worker created")

        if config.AUTO_START_WORKERS:
            logger.info("Starting %d background worker task(s)...", len(self._workers))
            for worker in self._workers:
                await worker.start()

            running_count = sum(1 for w in self._workers if w._running)
            task_count = sum(1 for w in self._workers if w._task and not w._task.done())

            logger.info(
                f"Worker status: running={running_count}/{len(self._workers)}, tasks_alive={task_count}/{len(self._workers)}"
            )

            if running_count != len(self._workers) or task_count != len(self._workers):
                logger.error("CRITICAL: Some queue workers failed to start!")
                logger.error(f"   Workers running: {running_count}/{len(self._workers)}")
                logger.error(f"   Tasks alive: {task_count}/{len(self._workers)}")
            else:
                logger.info("All %d queue worker(s) started successfully!", len(self._workers))
        else:
            logger.info(
                "Workers created but NOT started (AUTO_START_WORKERS=false) - "
                "Use process_external_message() to process messages from external consumer"
            )

        self._initialized = True
        logger.info("Queue manager initialized successfully")

    def _create_queue(self) -> QueueInterface:
        """Factory method to create the appropriate queue implementation."""
        if config.QUEUE_TYPE == "kafka":
            try:
                from .kafka_queue import KafkaQueue

                kafka_kwargs = {
                    "bootstrap_servers": config.KAFKA_BOOTSTRAP_SERVERS,
                    "topic": config.KAFKA_TOPIC,
                    "group_id": config.KAFKA_GROUP_ID,
                    "serialization_format": config.KAFKA_SERIALIZATION_FORMAT,
                    "security_protocol": config.KAFKA_SECURITY_PROTOCOL,
                    "auto_offset_reset": config.KAFKA_AUTO_OFFSET_RESET,
                    "consumer_timeout_ms": config.KAFKA_CONSUMER_TIMEOUT_MS,
                    "max_poll_interval_ms": config.KAFKA_MAX_POLL_INTERVAL_MS,
                    "session_timeout_ms": config.KAFKA_SESSION_TIMEOUT_MS,
                }

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
            if self._num_workers > 1:
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
                return MemoryQueue()

    async def save(self, message: QueueMessage) -> None:
        """
        Add a message to the queue.

        Args:
            message: QueueMessage protobuf message to add

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

        await self._queue.put(message)

    async def cleanup(self) -> None:
        """
        Stop all workers and close queue connections gracefully.

        Note: No logging during cleanup to avoid errors when logging system
        has already shut down during Python/pytest teardown.
        """
        for i, worker in enumerate(self._workers):
            is_last = i == len(self._workers) - 1
            await worker.stop(close_queue=is_last)

        self._workers = []
        self._queue = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def queue_type(self) -> str:
        return config.QUEUE_TYPE

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def round_robin(self) -> bool:
        return self._round_robin

    async def get_partition_stats(self) -> Optional[dict]:
        """Get statistics about partition distribution (memory queue only)."""
        if isinstance(self._queue, PartitionedMemoryQueue):
            return await self._queue.get_partition_stats()
        return None


_manager = QueueManager()


def get_manager() -> QueueManager:
    """Get the global queue manager singleton."""
    return _manager
