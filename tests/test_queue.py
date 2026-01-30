"""
Unit tests for Mirix Queue System

Tests cover:
- Queue initialization and lifecycle
- Message enqueueing and processing
- Worker thread management
- Queue manager functionality
- Memory queue implementation
"""

import threading
import time
from datetime import datetime
from unittest.mock import Mock

import pytest

# Import queue components
from mirix.queue import config as queue_config
from mirix.queue import initialize_queue, save
from mirix.queue.manager import get_manager
from mirix.queue.memory_queue import MemoryQueue, PartitionedMemoryQueue

# Note: ProtoUser and ProtoMessageCreate are generated from message.proto
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.message_pb2 import User as ProtoUser
from mirix.queue.queue_util import put_messages
from mirix.queue.worker import QueueWorker

# Import schemas
from mirix.schemas.client import Client
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import MessageCreate
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.services.organization_manager import OrganizationManager

# Test organization ID used by fixtures
TEST_QUEUE_ORG_ID = "org-456"

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ensure_organization():
    """Ensure the test organization exists in the database"""
    org_mgr = OrganizationManager()
    try:
        org_mgr.get_organization_by_id(TEST_QUEUE_ORG_ID)
    except Exception:
        org_mgr.create_organization(PydanticOrganization(id=TEST_QUEUE_ORG_ID, name="Test Queue Org"))
    return TEST_QUEUE_ORG_ID


@pytest.fixture
def mock_server():
    """Create a mock SyncServer instance"""
    server = Mock()
    server.send_messages = Mock(
        return_value=Mock(model_dump=Mock(return_value={"completion_tokens": 100, "prompt_tokens": 50}))
    )
    return server


@pytest.fixture
def sample_client(ensure_organization):
    """Create a sample Client (represents a client application)"""
    return Client(
        id="client-123",
        organization_id=ensure_organization,
        name="Test Client App",
        status="active",
        scope="read_write",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


@pytest.fixture
def sample_messages():
    """Create sample MessageCreate list"""
    return [
        MessageCreate(role=MessageRole.user, content="Hello, how are you?"),
        MessageCreate(role=MessageRole.user, content="What's the weather like?"),
    ]


@pytest.fixture
def sample_queue_message(sample_client):
    """Create a sample QueueMessage protobuf"""
    msg = QueueMessage()

    # Set client_id (new approach - worker will look up Client from database)
    msg.client_id = sample_client.id

    # Set agent and messages
    msg.agent_id = "agent-789"

    proto_msg = ProtoMessageCreate()
    proto_msg.role = ProtoMessageCreate.ROLE_USER
    proto_msg.text_content = "Test message"
    msg.input_messages.append(proto_msg)

    msg.chaining = True
    msg.verbose = False

    return msg


@pytest.fixture
def clean_manager():
    """Get a fresh QueueManager for testing"""
    # Get the singleton and cleanup if initialized
    manager = get_manager()
    if manager.is_initialized:
        manager.cleanup()
    return manager


@pytest.fixture
def configure_workers(monkeypatch, clean_manager):
    """
    Factory fixture to configure worker count and round-robin mode via config patching.

    Usage:
        def test_something(configure_workers, mock_server):
            configure_workers(num_workers=4, round_robin=False)
            manager = get_manager()
            manager.initialize(server=mock_server)
    """

    def _configure(num_workers: int = 1, round_robin: bool = False):
        monkeypatch.setattr(queue_config, "NUM_WORKERS", num_workers)
        monkeypatch.setattr(queue_config, "ROUND_ROBIN", round_robin)
        return clean_manager

    return _configure


# ============================================================================
# MemoryQueue Tests
# ============================================================================


class TestMemoryQueue:
    """Test the in-memory queue implementation"""

    def test_memory_queue_init(self):
        """Test MemoryQueue initialization"""
        queue = MemoryQueue()
        assert queue is not None
        assert hasattr(queue, "_queue")

    def test_memory_queue_put_get(self, sample_queue_message):
        """Test putting and getting messages from memory queue"""
        queue = MemoryQueue()

        # Put message
        queue.put(sample_queue_message)

        # Get message
        retrieved = queue.get(timeout=1.0)

        assert retrieved.agent_id == sample_queue_message.agent_id
        assert retrieved.client_id == sample_queue_message.client_id
        assert len(retrieved.input_messages) == len(sample_queue_message.input_messages)

    def test_memory_queue_timeout(self):
        """Test that get() raises Empty on timeout"""
        import queue as q

        mem_queue = MemoryQueue()

        with pytest.raises(q.Empty):
            mem_queue.get(timeout=0.1)

    def test_memory_queue_fifo_order(self, sample_client):
        """Test that messages are retrieved in FIFO order"""
        queue = MemoryQueue()

        # Create multiple messages
        messages = []
        for i in range(5):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-{i}"
            messages.append(msg)
            queue.put(msg)

        # Retrieve and verify order
        for i in range(5):
            retrieved = queue.get(timeout=1.0)
            assert retrieved.agent_id == f"agent-{i}"

    def test_memory_queue_close(self):
        """Test queue close (should not raise error)"""
        queue = MemoryQueue()
        queue.close()  # Should complete without error


# ============================================================================
# PartitionedMemoryQueue Tests
# ============================================================================


class TestPartitionedMemoryQueue:
    """Test the partitioned in-memory queue implementation"""

    def test_partitioned_queue_init(self):
        """Test PartitionedMemoryQueue initialization"""
        queue = PartitionedMemoryQueue(num_partitions=4)
        assert queue is not None
        assert queue.num_partitions == 4
        assert len(queue._partitions) == 4

    def test_partitioned_queue_default_partitions(self):
        """Test default partition count is 1"""
        queue = PartitionedMemoryQueue()
        assert queue.num_partitions == 1

    def test_partitioned_queue_min_partitions(self):
        """Test that partition count is at least 1"""
        queue = PartitionedMemoryQueue(num_partitions=0)
        assert queue.num_partitions == 1

        queue = PartitionedMemoryQueue(num_partitions=-5)
        assert queue.num_partitions == 1

    def test_same_user_routes_to_same_partition(self, sample_client):
        """Test that messages with same user_id always go to same partition"""
        queue = PartitionedMemoryQueue(num_partitions=4)

        # Create multiple messages for the same user
        user_id = "user-consistent"
        messages = []
        for i in range(10):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-{i}"
            msg.user_id = user_id
            messages.append(msg)
            queue.put(msg)

        # All messages should be in the same partition
        # Find which partition has messages
        partition_with_messages = None
        for partition_id in range(4):
            try:
                msg = queue.get_from_partition(partition_id, timeout=0.1)
                partition_with_messages = partition_id
                # Put it back for counting
                queue._partitions[partition_id].put(msg)
                break
            except:
                continue

        assert partition_with_messages is not None

        # Count messages in that partition
        count = 0
        while True:
            try:
                queue.get_from_partition(partition_with_messages, timeout=0.1)
                count += 1
            except:
                break

        assert count == 10  # All 10 messages in same partition

    def test_different_users_can_route_to_different_partitions(self, sample_client):
        """Test that different user_ids can go to different partitions"""
        queue = PartitionedMemoryQueue(num_partitions=100)  # Many partitions to increase spread

        # Create messages for many different users
        user_ids = [f"user-{i}" for i in range(50)]

        for user_id in user_ids:
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = "agent-test"
            msg.user_id = user_id
            queue.put(msg)

        # Count messages per partition
        partitions_with_messages = set()
        for partition_id in range(100):
            try:
                queue.get_from_partition(partition_id, timeout=0.01)
                partitions_with_messages.add(partition_id)
            except:
                continue

        # With 50 users and 100 partitions, we should have some spread
        # (not all in one partition)
        assert len(partitions_with_messages) > 1

    def test_get_from_partition_retrieves_correct_partition(self, sample_client):
        """Test that get_from_partition retrieves from the correct partition"""
        queue = PartitionedMemoryQueue(num_partitions=3)

        # Manually put messages in specific partitions
        for i in range(3):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-partition-{i}"
            queue._partitions[i].put(msg)

        # Retrieve from each partition
        for i in range(3):
            msg = queue.get_from_partition(i, timeout=1.0)
            assert msg.agent_id == f"agent-partition-{i}"

    def test_get_from_partition_invalid_partition(self):
        """Test that invalid partition_id raises ValueError"""
        queue = PartitionedMemoryQueue(num_partitions=3)

        with pytest.raises(ValueError):
            queue.get_from_partition(5, timeout=0.1)

        with pytest.raises(ValueError):
            queue.get_from_partition(-1, timeout=0.1)

    def test_get_from_partition_timeout(self):
        """Test that get_from_partition raises Empty on timeout"""
        import queue as q

        pqueue = PartitionedMemoryQueue(num_partitions=2)

        with pytest.raises(q.Empty):
            pqueue.get_from_partition(0, timeout=0.1)

    def test_backward_compatible_get(self, sample_client):
        """Test that get() still works (retrieves from partition 0)"""
        queue = PartitionedMemoryQueue(num_partitions=1)

        msg = QueueMessage()
        msg.client_id = sample_client.id
        msg.agent_id = "agent-compat"
        msg.user_id = "user-compat"
        queue.put(msg)

        # Use regular get()
        retrieved = queue.get(timeout=1.0)
        assert retrieved.agent_id == "agent-compat"


# ============================================================================
# QueueWorker Tests
# ============================================================================


class TestQueueWorker:
    """Test the queue worker functionality"""

    def test_worker_init_without_server(self):
        """Test worker initialization without server"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        assert worker.queue == queue
        assert worker._server is None
        assert worker._running is False

    def test_worker_init_with_server(self, mock_server):
        """Test worker initialization with server"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)

        assert worker.queue == queue
        assert worker._server == mock_server

    def test_worker_set_server(self, mock_server):
        """Test setting server after initialization"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        assert worker._server is None

        worker.set_server(mock_server)

        assert worker._server == mock_server

    def test_worker_start_stop(self):
        """Test worker thread lifecycle"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        # Start worker
        worker.start()
        assert worker._running is True
        assert worker._thread is not None
        assert worker._thread.is_alive()

        # Stop worker
        worker.stop()
        assert worker._running is False

        # Wait a bit for thread to finish
        time.sleep(0.5)
        assert not worker._thread.is_alive()

    def test_worker_process_message_without_server(self, sample_queue_message):
        """Test processing message when no server is available"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        # Should log warning and skip processing
        worker._process_message(sample_queue_message)
        # No error should be raised

    def test_worker_process_message_with_server(self, mock_server, sample_queue_message):
        """Test processing message with server available"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)

        # Process message
        worker._process_message(sample_queue_message)

        # Verify server.send_messages was called
        mock_server.send_messages.assert_called_once()

        # Check call arguments
        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs["agent_id"] == sample_queue_message.agent_id
        assert len(call_args.kwargs["input_messages"]) > 0

    def test_worker_message_processing_integration(self, mock_server, sample_queue_message):
        """Test end-to-end message processing"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)

        # Enqueue message
        queue.put(sample_queue_message)

        # Start worker
        worker.start()

        # Wait for processing
        time.sleep(1.0)

        # Stop worker
        worker.stop()

        # Verify server was called
        assert mock_server.send_messages.call_count >= 1


# ============================================================================
# QueueManager Tests
# ============================================================================


class TestQueueManager:
    """Test the queue manager functionality"""

    def test_manager_singleton(self):
        """Test that get_manager returns singleton"""
        manager1 = get_manager()
        manager2 = get_manager()

        assert manager1 is manager2

    def test_manager_init_without_server(self, clean_manager):
        """Test manager initialization without server"""
        manager = clean_manager

        assert not manager.is_initialized

        manager.initialize()

        assert manager.is_initialized
        assert manager._queue is not None
        assert len(manager._workers) > 0

        # Cleanup
        manager.cleanup()

    def test_manager_init_with_server(self, clean_manager, mock_server):
        """Test manager initialization with server"""
        manager = clean_manager

        manager.initialize(server=mock_server)

        assert manager.is_initialized
        assert manager._server == mock_server
        assert manager._workers[0]._server == mock_server

        # Cleanup
        manager.cleanup()

    def test_manager_idempotent_init(self, clean_manager, mock_server):
        """Test that multiple initialize calls are idempotent"""
        manager = clean_manager

        manager.initialize(server=mock_server)
        first_queue = manager._queue
        first_workers = manager._workers.copy()

        # Call initialize again
        manager.initialize(server=mock_server)

        # Should be same instances
        assert manager._queue is first_queue
        assert manager._workers == first_workers

        # Cleanup
        manager.cleanup()

    def test_manager_update_server_after_init(self, clean_manager, mock_server):
        """Test updating server after initialization"""
        manager = clean_manager

        # Initialize without server
        manager.initialize()
        assert manager._server is None

        # Update with server
        mock_server_2 = Mock()
        manager.initialize(server=mock_server_2)

        assert manager._server == mock_server_2
        assert manager._workers[0]._server == mock_server_2

        # Cleanup
        manager.cleanup()

    def test_manager_save_message(self, clean_manager, sample_queue_message):
        """Test saving message via manager"""
        manager = clean_manager
        manager.initialize()

        # Save message
        manager.save(sample_queue_message)

        # Should be in queue
        retrieved = manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id

        # Cleanup
        manager.cleanup()

    def test_manager_cleanup(self, clean_manager):
        """Test manager cleanup"""
        manager = clean_manager
        manager.initialize()

        assert manager.is_initialized
        assert len(manager._workers) > 0
        assert manager._workers[0]._running

        manager.cleanup()

        assert not manager.is_initialized
        assert manager._queue is None
        assert len(manager._workers) == 0


# ============================================================================
# Multi-Worker Manager Tests
# ============================================================================


class TestMultiWorkerManager:
    """Test the queue manager with multiple workers"""

    def test_manager_single_worker_default(self, configure_workers, mock_server):
        """Test that default is single worker"""
        manager = configure_workers(num_workers=1)
        manager.initialize(server=mock_server)

        assert manager.num_workers == 1
        assert len(manager._workers) == 1
        assert isinstance(manager._queue, MemoryQueue)

        manager.cleanup()

    def test_manager_multiple_workers(self, configure_workers, mock_server):
        """Test manager creates correct number of workers"""
        manager = configure_workers(num_workers=4)
        manager.initialize(server=mock_server)

        assert manager.num_workers == 4
        assert len(manager._workers) == 4
        assert isinstance(manager._queue, PartitionedMemoryQueue)
        assert manager._queue.num_partitions == 4

        manager.cleanup()

    def test_manager_workers_have_unique_partition_ids(self, configure_workers, mock_server):
        """Test each worker is assigned a unique partition_id"""
        manager = configure_workers(num_workers=4)
        manager.initialize(server=mock_server)

        partition_ids = [w._partition_id for w in manager._workers]

        # Should have 0, 1, 2, 3
        assert sorted(partition_ids) == [0, 1, 2, 3]

        manager.cleanup()

    def test_manager_all_workers_running(self, configure_workers, mock_server):
        """Test all workers are started and running"""
        manager = configure_workers(num_workers=3)
        manager.initialize(server=mock_server)

        # Give workers time to start
        time.sleep(0.2)

        for worker in manager._workers:
            assert worker._running
            assert worker._thread is not None
            assert worker._thread.is_alive()

        manager.cleanup()

    def test_manager_cleanup_stops_all_workers(self, configure_workers, mock_server):
        """Test cleanup stops all workers"""
        manager = configure_workers(num_workers=3)
        manager.initialize(server=mock_server)

        workers = manager._workers.copy()

        manager.cleanup()

        # All workers should be stopped
        time.sleep(0.5)
        for worker in workers:
            assert not worker._running

    def test_initialize_queue_with_num_workers(self, configure_workers, mock_server):
        """Test initialize_queue() respects NUM_WORKERS config"""
        manager = configure_workers(num_workers=5)

        initialize_queue(mock_server)

        assert manager.num_workers == 5
        assert len(manager._workers) == 5

        manager.cleanup()

    def test_num_workers_one_uses_simple_queue(self, configure_workers, mock_server):
        """Test that num_workers=1 uses simple MemoryQueue"""
        manager = configure_workers(num_workers=1)
        manager.initialize(server=mock_server)

        assert isinstance(manager._queue, MemoryQueue)
        assert not isinstance(manager._queue, PartitionedMemoryQueue)
        assert len(manager._workers) == 1
        assert manager._workers[0]._partition_id is None

        manager.cleanup()


# ============================================================================
# Worker Partition Assignment Tests
# ============================================================================


class TestWorkerPartitionAssignment:
    """Test workers correctly consume from their assigned partitions"""

    def test_worker_with_partition_id_init(self):
        """Test worker initialization with partition_id"""
        queue = PartitionedMemoryQueue(num_partitions=4)
        worker = QueueWorker(queue, partition_id=2)

        assert worker._partition_id == 2

    def test_worker_consumes_from_assigned_partition(self, mock_server, sample_client):
        """Test worker only consumes from its assigned partition"""
        queue = PartitionedMemoryQueue(num_partitions=3)

        # Create worker for partition 1
        worker = QueueWorker(queue, server=mock_server, partition_id=1)

        # Put message directly in partition 1
        msg = QueueMessage()
        msg.client_id = sample_client.id
        msg.agent_id = "agent-partition-1"
        msg.user_id = "user-1"
        queue._partitions[1].put(msg)

        # Put message in partition 0 (different partition)
        msg2 = QueueMessage()
        msg2.client_id = sample_client.id
        msg2.agent_id = "agent-partition-0"
        msg2.user_id = "user-0"
        queue._partitions[0].put(msg2)

        # Start worker
        worker.start()
        time.sleep(0.5)
        worker.stop(close_queue=False)

        # Worker should have processed only partition 1 message
        call_args_list = mock_server.send_messages.call_args_list
        agent_ids = [call.kwargs["agent_id"] for call in call_args_list]

        assert "agent-partition-1" in agent_ids
        assert "agent-partition-0" not in agent_ids

        # Partition 0 message should still be there
        remaining = queue._partitions[0].get(timeout=0.1)
        assert remaining.agent_id == "agent-partition-0"

    def test_multiple_workers_partition_isolation(self, mock_server, sample_client):
        """Test multiple workers don't interfere with each other's partitions"""
        queue = PartitionedMemoryQueue(num_partitions=2)

        # Track which worker processed which message
        processed = {"worker-0": [], "worker-1": []}
        lock = threading.Lock()

        def track_processing(worker_id):
            def handler(**kwargs):
                with lock:
                    processed[worker_id].append(kwargs["agent_id"])
                return None

            return handler

        # Create two workers with different mocks
        mock_server_0 = Mock()
        mock_server_0.send_messages = track_processing("worker-0")
        worker_0 = QueueWorker(queue, server=mock_server_0, partition_id=0)

        mock_server_1 = Mock()
        mock_server_1.send_messages = track_processing("worker-1")
        worker_1 = QueueWorker(queue, server=mock_server_1, partition_id=1)

        # Put messages in specific partitions
        for i in range(5):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-p0-{i}"
            queue._partitions[0].put(msg)

        for i in range(5):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-p1-{i}"
            queue._partitions[1].put(msg)

        # Start both workers
        worker_0.start()
        worker_1.start()

        time.sleep(1.0)

        worker_0.stop(close_queue=False)
        worker_1.stop(close_queue=False)

        # Worker 0 should only have p0 messages
        for agent_id in processed["worker-0"]:
            assert "p0" in agent_id

        # Worker 1 should only have p1 messages
        for agent_id in processed["worker-1"]:
            assert "p1" in agent_id

        assert len(processed["worker-0"]) == 5
        assert len(processed["worker-1"]) == 5

    def test_partitioned_queue_distributes_by_user_id(
        self, configure_workers, mock_server, sample_client, sample_messages
    ):
        """Test end-to-end: messages route by user_id across workers"""
        manager = configure_workers(num_workers=4)

        # Track processed messages per partition
        processed_by_partition = {}
        lock = threading.Lock()

        original_send = mock_server.send_messages

        def tracking_send(**kwargs):
            # Get the user from kwargs
            user = kwargs.get("user")
            user_id = user.id if user else "unknown"
            with lock:
                if user_id not in processed_by_partition:
                    processed_by_partition[user_id] = []
                processed_by_partition[user_id].append(kwargs["agent_id"])
            return original_send(**kwargs)

        mock_server.send_messages = tracking_send

        # Initialize with multiple workers (config already set via fixture)
        manager.initialize(server=mock_server)

        # Send messages for different users
        user_ids = ["user-a", "user-b", "user-c", "user-d"]
        for user_id in user_ids:
            for i in range(3):
                put_messages(
                    actor=sample_client,
                    agent_id=f"agent-{user_id}-{i}",
                    input_messages=sample_messages,
                    user_id=user_id,
                )

        # Wait for processing
        time.sleep(2.0)

        # Verify messages were processed (at least some)
        total_processed = sum(len(v) for v in processed_by_partition.values())
        assert total_processed > 0

        manager.cleanup()


# ============================================================================
# queue_util Tests
# ============================================================================


class TestQueueUtil:
    """Test queue utility functions"""

    def test_put_messages_basic(self, clean_manager, sample_client, sample_messages):
        """Test put_messages with basic parameters"""
        manager = clean_manager
        manager.initialize()

        put_messages(actor=sample_client, agent_id="agent-789", input_messages=sample_messages)

        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.agent_id == "agent-789"
        assert msg.client_id == sample_client.id
        assert len(msg.input_messages) == len(sample_messages)

        # Cleanup
        manager.cleanup()

    def test_put_messages_with_options(self, clean_manager, sample_client, sample_messages):
        """Test put_messages with optional parameters"""
        manager = clean_manager
        manager.initialize()

        put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=sample_messages,
            chaining=False,
            user_id="user-custom",
            verbose=True,
            filter_tags={"tag1": "value1"},
        )

        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.chaining is False
        assert msg.user_id == "user-custom"
        assert msg.verbose is True
        assert dict(msg.filter_tags) == {"tag1": "value1"}

        # Cleanup
        manager.cleanup()

    def test_put_messages_role_mapping(self, clean_manager, sample_client):
        """Test that message roles are correctly mapped"""
        manager = clean_manager
        manager.initialize()

        messages = [
            MessageCreate(role=MessageRole.user, content="User message"),
            MessageCreate(role=MessageRole.system, content="System message"),
        ]

        put_messages(actor=sample_client, agent_id="agent-789", input_messages=messages)

        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.input_messages[0].role == ProtoMessageCreate.ROLE_USER
        assert msg.input_messages[1].role == ProtoMessageCreate.ROLE_SYSTEM

        # Cleanup
        manager.cleanup()


# ============================================================================
# Queue __init__ Tests
# ============================================================================


class TestQueueInit:
    """Test queue module initialization functions"""

    def test_initialize_queue(self, clean_manager, mock_server):
        """Test initialize_queue function"""
        manager = clean_manager

        initialize_queue(mock_server)

        assert manager.is_initialized
        assert manager._server == mock_server

        # Cleanup
        manager.cleanup()

    def test_save_without_init(self, clean_manager, sample_queue_message):
        """Test save auto-initializes if not initialized"""
        manager = clean_manager

        assert not manager.is_initialized

        save(sample_queue_message)

        # Should auto-initialize
        assert manager.is_initialized

        # Cleanup
        manager.cleanup()

    def test_save_with_init(self, clean_manager, mock_server, sample_queue_message):
        """Test save after manual initialization"""
        manager = clean_manager
        initialize_queue(mock_server)

        save(sample_queue_message)

        # Should be in queue
        retrieved = manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id

        # Cleanup
        manager.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================


class TestQueueIntegration:
    """Integration tests for the complete queue system"""

    def test_end_to_end_message_flow(self, clean_manager, mock_server, sample_client, sample_messages):
        """Test complete message flow from enqueue to processing"""
        manager = clean_manager

        # Initialize with server
        initialize_queue(mock_server)

        # Enqueue message
        put_messages(
            actor=sample_client,
            agent_id="agent-integration",
            input_messages=sample_messages,
        )

        # Wait for worker to process
        time.sleep(1.5)

        # Verify server was called
        assert mock_server.send_messages.call_count >= 1

        # Verify call details
        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs["agent_id"] == "agent-integration"

        # Cleanup
        manager.cleanup()

    def test_multiple_messages_processing(self, clean_manager, mock_server, sample_client, sample_messages):
        """Test processing multiple messages"""
        manager = clean_manager
        initialize_queue(mock_server)

        # Enqueue multiple messages
        for i in range(5):
            put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages,
            )

        # Wait for processing
        time.sleep(2.0)

        # Verify all were processed
        assert mock_server.send_messages.call_count >= 5

        # Cleanup
        manager.cleanup()

    def test_worker_handles_processing_errors(self, clean_manager, sample_client, sample_messages):
        """Test that worker handles errors gracefully"""
        manager = clean_manager

        # Create server that raises exception
        error_server = Mock()
        error_server.send_messages = Mock(side_effect=Exception("Processing error"))

        initialize_queue(error_server)

        # Enqueue message
        put_messages(actor=sample_client, agent_id="agent-error", input_messages=sample_messages)

        # Wait for processing attempt
        time.sleep(1.5)

        # Worker should still be running despite error
        assert manager._workers[0]._running

        # Cleanup
        manager.cleanup()


# ============================================================================
# Performance Tests
# ============================================================================


class TestQueuePerformance:
    """Performance tests for queue operations"""

    def test_enqueue_performance(self, clean_manager, sample_client, sample_messages):
        """Test enqueueing performance"""
        manager = clean_manager
        manager.initialize()

        start = time.time()

        # Enqueue 100 messages
        for i in range(100):
            put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages,
            )

        elapsed = time.time() - start

        # Should be very fast (< 1 second for 100 messages)
        assert elapsed < 1.0

        print(f"\nEnqueued 100 messages in {elapsed:.3f}s")

        # Cleanup
        manager.cleanup()

    def test_concurrent_enqueue(self, sample_client, sample_messages, mock_server):
        """Test concurrent enqueueing from multiple threads"""
        # Initialize the global queue manager with mock server
        # This prevents messages from being discarded
        initialize_queue(server=mock_server)
        manager = get_manager()

        # Track processed messages
        processed_messages = []
        processed_lock = threading.Lock()

        def mock_send_messages(**kwargs):
            with processed_lock:
                processed_messages.append(kwargs["agent_id"])
            return None

        mock_server.send_messages = mock_send_messages

        def enqueue_messages(thread_id, count):
            for i in range(count):
                put_messages(
                    actor=sample_client,
                    agent_id=f"agent-{thread_id}-{i}",
                    input_messages=sample_messages,
                )

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=enqueue_messages, args=(i, 20))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Wait a bit for worker to process all messages
        time.sleep(1.0)

        # Verify all 100 messages were processed
        assert len(processed_messages) == 100  # 5 threads * 20 messages

        # Cleanup
        manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
