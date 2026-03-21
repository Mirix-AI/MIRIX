"""
Unit tests for Mirix Queue System (async-native).

Tests cover:
- Queue initialization and lifecycle
- Message enqueueing and processing
- Worker task management
- Queue manager functionality
- Memory queue implementation
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from mirix.queue import config as queue_config
from mirix.queue import initialize_queue, save
from mirix.queue.manager import get_manager
from mirix.queue.memory_queue import MemoryQueue, PartitionedMemoryQueue
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.message_pb2 import User as ProtoUser
from mirix.queue.queue_util import put_messages
from mirix.queue.worker import QueueWorker
from mirix.schemas.client import Client
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import MessageCreate
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.services.organization_manager import OrganizationManager

TEST_QUEUE_ORG_ID = "org-456"

# Use one event loop per module so DB and async fixtures share it (avoids
# "Future attached to a different loop" / "another operation is in progress").
pytestmark = pytest.mark.asyncio(loop_scope="module")


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _init_db():
    """Create all DB tables before any test in this module touches the database."""
    from mirix.server.server import ensure_tables_created
    await ensure_tables_created()


@pytest_asyncio.fixture(scope="module")
async def ensure_organization():
    """Ensure the test organization exists in the database (once per module)."""
    org_mgr = OrganizationManager()
    try:
        await org_mgr.get_organization_by_id(TEST_QUEUE_ORG_ID)
    except Exception:
        await org_mgr.create_organization(
            PydanticOrganization(id=TEST_QUEUE_ORG_ID, name="Test Queue Org")
        )
    return TEST_QUEUE_ORG_ID


@pytest.fixture
def mock_server(sample_client):
    """Create a mock AsyncServer with async send_messages and client_manager so workers can resolve actor/user."""
    server = Mock()
    server.send_messages = AsyncMock(
        return_value=Mock(model_dump=Mock(return_value={"completion_tokens": 100, "prompt_tokens": 50}))
    )
    server.client_manager = Mock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=sample_client)

    mock_user = Mock(id="admin", organization_id=sample_client.organization_id)

    with patch("mirix.queue.worker.UserManager") as MockUM:
        MockUM.return_value.get_admin_user = AsyncMock(return_value=mock_user)
        MockUM.return_value.get_user_by_id = AsyncMock(side_effect=Exception("not found"))
        MockUM.return_value.create_user = AsyncMock(return_value=mock_user)
        yield server


@pytest.fixture
def sample_client(ensure_organization):
    """Create a sample Client"""
    return Client(
        id="client-123",
        organization_id=ensure_organization,
        name="Test Client App",
        status="active",
        write_scope="test",
        read_scopes=["test"],
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
    msg.client_id = sample_client.id
    msg.agent_id = "agent-789"

    proto_msg = ProtoMessageCreate()
    proto_msg.role = ProtoMessageCreate.ROLE_USER
    proto_msg.text_content = "Test message"
    msg.input_messages.append(proto_msg)

    msg.chaining = True
    msg.verbose = False

    return msg


@pytest_asyncio.fixture
async def clean_manager():
    """Get a fresh QueueManager for testing"""
    manager = get_manager()
    if manager.is_initialized:
        await manager.cleanup()
    return manager


@pytest.fixture
def configure_workers(monkeypatch, clean_manager):
    """Factory fixture to configure worker count and round-robin mode."""

    def _configure(num_workers: int = 1, round_robin: bool = False):
        monkeypatch.setattr(queue_config, "NUM_WORKERS", num_workers)
        monkeypatch.setattr(queue_config, "ROUND_ROBIN", round_robin)
        return clean_manager

    return _configure


# ============================================================================
# MemoryQueue Tests
# ============================================================================


class TestMemoryQueue:
    """Test the async in-memory queue implementation"""

    def test_memory_queue_init(self):
        queue = MemoryQueue()
        assert queue is not None
        assert hasattr(queue, "_queue")

    @pytest.mark.asyncio
    async def test_memory_queue_put_get(self, sample_queue_message):
        queue = MemoryQueue()

        await queue.put(sample_queue_message)
        retrieved = await queue.get(timeout=1.0)

        assert retrieved.agent_id == sample_queue_message.agent_id
        assert retrieved.client_id == sample_queue_message.client_id
        assert len(retrieved.input_messages) == len(sample_queue_message.input_messages)

    @pytest.mark.asyncio
    async def test_memory_queue_timeout(self):
        mem_queue = MemoryQueue()

        with pytest.raises(asyncio.TimeoutError):
            await mem_queue.get(timeout=0.1)

    @pytest.mark.asyncio
    async def test_memory_queue_fifo_order(self, sample_client):
        queue = MemoryQueue()

        messages = []
        for i in range(5):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-{i}"
            messages.append(msg)
            await queue.put(msg)

        for i in range(5):
            retrieved = await queue.get(timeout=1.0)
            assert retrieved.agent_id == f"agent-{i}"

    @pytest.mark.asyncio
    async def test_memory_queue_close(self):
        queue = MemoryQueue()
        await queue.close()


# ============================================================================
# PartitionedMemoryQueue Tests
# ============================================================================


class TestPartitionedMemoryQueue:
    """Test the partitioned async in-memory queue implementation"""

    def test_partitioned_queue_init(self):
        queue = PartitionedMemoryQueue(num_partitions=4)
        assert queue is not None
        assert queue.num_partitions == 4
        assert len(queue._partitions) == 4

    def test_partitioned_queue_default_partitions(self):
        queue = PartitionedMemoryQueue()
        assert queue.num_partitions == 1

    def test_partitioned_queue_min_partitions(self):
        queue = PartitionedMemoryQueue(num_partitions=0)
        assert queue.num_partitions == 1

        queue = PartitionedMemoryQueue(num_partitions=-5)
        assert queue.num_partitions == 1

    @pytest.mark.asyncio
    async def test_same_user_routes_to_same_partition(self, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=4)

        user_id = "user-consistent"
        for i in range(10):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-{i}"
            msg.user_id = user_id
            await queue.put(msg)

        partition_with_messages = None
        for partition_id in range(4):
            try:
                msg = await queue.get_from_partition(partition_id, timeout=0.1)
                partition_with_messages = partition_id
                await queue._partitions[partition_id].put(msg)
                break
            except asyncio.TimeoutError:
                continue

        assert partition_with_messages is not None

        count = 0
        while True:
            try:
                await queue.get_from_partition(partition_with_messages, timeout=0.1)
                count += 1
            except asyncio.TimeoutError:
                break

        assert count == 10

    @pytest.mark.asyncio
    async def test_different_users_can_route_to_different_partitions(self, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=100)

        user_ids = [f"user-{i}" for i in range(50)]
        for user_id in user_ids:
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = "agent-test"
            msg.user_id = user_id
            await queue.put(msg)

        partitions_with_messages = set()
        for partition_id in range(100):
            try:
                await queue.get_from_partition(partition_id, timeout=0.01)
                partitions_with_messages.add(partition_id)
            except asyncio.TimeoutError:
                continue

        assert len(partitions_with_messages) > 1

    @pytest.mark.asyncio
    async def test_get_from_partition_retrieves_correct_partition(self, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=3)

        for i in range(3):
            msg = QueueMessage()
            msg.client_id = sample_client.id
            msg.agent_id = f"agent-partition-{i}"
            await queue._partitions[i].put(msg)

        for i in range(3):
            msg = await queue.get_from_partition(i, timeout=1.0)
            assert msg.agent_id == f"agent-partition-{i}"

    @pytest.mark.asyncio
    async def test_get_from_partition_invalid_partition(self):
        queue = PartitionedMemoryQueue(num_partitions=3)

        with pytest.raises(ValueError):
            await queue.get_from_partition(5, timeout=0.1)

        with pytest.raises(ValueError):
            await queue.get_from_partition(-1, timeout=0.1)

    @pytest.mark.asyncio
    async def test_get_from_partition_timeout(self):
        pqueue = PartitionedMemoryQueue(num_partitions=2)

        with pytest.raises(asyncio.TimeoutError):
            await pqueue.get_from_partition(0, timeout=0.1)

    @pytest.mark.asyncio
    async def test_backward_compatible_get(self, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=1)

        msg = QueueMessage()
        msg.client_id = sample_client.id
        msg.agent_id = "agent-compat"
        msg.user_id = "user-compat"
        await queue.put(msg)

        retrieved = await queue.get(timeout=1.0)
        assert retrieved.agent_id == "agent-compat"


# ============================================================================
# QueueWorker Tests
# ============================================================================


class TestQueueWorker:
    """Test the queue worker functionality"""

    def test_worker_init_without_server(self):
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        assert worker.queue == queue
        assert worker._server is None
        assert worker._running is False

    def test_worker_init_with_server(self, mock_server):
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)

        assert worker.queue == queue
        assert worker._server == mock_server

    def test_worker_set_server(self, mock_server):
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        assert worker._server is None
        worker.set_server(mock_server)
        assert worker._server == mock_server

    @pytest.mark.asyncio
    async def test_worker_start_stop(self):
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        await worker.start()
        assert worker._running is True
        assert worker._task is not None
        assert not worker._task.done()

        await worker.stop()
        assert worker._running is False

    @pytest.mark.asyncio
    async def test_worker_process_message_without_server(self, sample_queue_message):
        queue = MemoryQueue()
        worker = QueueWorker(queue)

        await worker._process_message_async(sample_queue_message)

    @pytest.mark.asyncio
    async def test_worker_message_processing_integration(self, mock_server, sample_queue_message):
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)

        await queue.put(sample_queue_message)
        await worker.start()

        await asyncio.sleep(0.5)
        await worker.stop()

        assert mock_server.send_messages.call_count >= 1


# ============================================================================
# QueueManager Tests
# ============================================================================


class TestQueueManager:
    """Test the queue manager functionality"""

    def test_manager_singleton(self):
        manager1 = get_manager()
        manager2 = get_manager()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_manager_init_without_server(self, clean_manager):
        manager = clean_manager
        assert not manager.is_initialized

        await manager.initialize()
        assert manager.is_initialized
        assert manager._queue is not None
        assert len(manager._workers) > 0

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_init_with_server(self, clean_manager, mock_server):
        manager = clean_manager

        await manager.initialize(server=mock_server)
        assert manager.is_initialized
        assert manager._server == mock_server
        assert manager._workers[0]._server == mock_server

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_idempotent_init(self, clean_manager, mock_server):
        manager = clean_manager

        await manager.initialize(server=mock_server)
        first_queue = manager._queue
        first_workers = manager._workers.copy()

        await manager.initialize(server=mock_server)
        assert manager._queue is first_queue
        assert manager._workers == first_workers

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_update_server_after_init(self, clean_manager, mock_server):
        manager = clean_manager

        await manager.initialize()
        assert manager._server is None

        mock_server_2 = Mock()
        await manager.initialize(server=mock_server_2)
        assert manager._server == mock_server_2
        assert manager._workers[0]._server == mock_server_2

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_save_message(self, clean_manager, sample_queue_message):
        manager = clean_manager
        await manager.initialize()

        await manager.save(sample_queue_message)

        retrieved = await manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_cleanup(self, clean_manager):
        manager = clean_manager
        await manager.initialize()

        assert manager.is_initialized
        assert len(manager._workers) > 0
        assert manager._workers[0]._running

        await manager.cleanup()

        assert not manager.is_initialized
        assert manager._queue is None
        assert len(manager._workers) == 0


# ============================================================================
# Multi-Worker Manager Tests
# ============================================================================


class TestMultiWorkerManager:
    """Test the queue manager with multiple workers"""

    @pytest.mark.asyncio
    async def test_manager_single_worker_default(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=1)
        await manager.initialize(server=mock_server)

        assert manager.num_workers == 1
        assert len(manager._workers) == 1
        assert isinstance(manager._queue, MemoryQueue)

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_multiple_workers(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=4)
        await manager.initialize(server=mock_server)

        assert manager.num_workers == 4
        assert len(manager._workers) == 4
        assert isinstance(manager._queue, PartitionedMemoryQueue)
        assert manager._queue.num_partitions == 4

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_workers_have_unique_partition_ids(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=4)
        await manager.initialize(server=mock_server)

        partition_ids = [w._partition_id for w in manager._workers]
        assert sorted(partition_ids) == [0, 1, 2, 3]

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_all_workers_running(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=3)
        await manager.initialize(server=mock_server)

        await asyncio.sleep(0.1)

        for worker in manager._workers:
            assert worker._running
            assert worker._task is not None
            assert not worker._task.done()

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_manager_cleanup_stops_all_workers(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=3)
        await manager.initialize(server=mock_server)

        workers = manager._workers.copy()
        await manager.cleanup()

        for worker in workers:
            assert not worker._running

    @pytest.mark.asyncio
    async def test_initialize_queue_with_num_workers(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=5)

        await initialize_queue(mock_server)

        assert manager.num_workers == 5
        assert len(manager._workers) == 5

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_num_workers_one_uses_simple_queue(self, configure_workers, mock_server):
        manager = configure_workers(num_workers=1)
        await manager.initialize(server=mock_server)

        assert isinstance(manager._queue, MemoryQueue)
        assert not isinstance(manager._queue, PartitionedMemoryQueue)
        assert len(manager._workers) == 1
        assert manager._workers[0]._partition_id is None

        await manager.cleanup()


# ============================================================================
# Worker Partition Assignment Tests
# ============================================================================


class TestWorkerPartitionAssignment:
    """Test workers correctly consume from their assigned partitions"""

    def test_worker_with_partition_id_init(self):
        queue = PartitionedMemoryQueue(num_partitions=4)
        worker = QueueWorker(queue, partition_id=2)
        assert worker._partition_id == 2

    @pytest.mark.asyncio
    async def test_worker_consumes_from_assigned_partition(self, mock_server, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=3)

        worker = QueueWorker(queue, server=mock_server, partition_id=1)

        msg = QueueMessage()
        msg.client_id = sample_client.id
        msg.agent_id = "agent-partition-1"
        msg.user_id = "user-1"
        await queue._partitions[1].put(msg)

        msg2 = QueueMessage()
        msg2.client_id = sample_client.id
        msg2.agent_id = "agent-partition-0"
        msg2.user_id = "user-0"
        await queue._partitions[0].put(msg2)

        await worker.start()
        await asyncio.sleep(0.5)
        await worker.stop(close_queue=False)

        call_args_list = mock_server.send_messages.call_args_list
        agent_ids = [call.kwargs["agent_id"] for call in call_args_list]

        assert "agent-partition-1" in agent_ids
        assert "agent-partition-0" not in agent_ids

        remaining = await asyncio.wait_for(queue._partitions[0].get(), timeout=0.1)
        assert remaining.agent_id == "agent-partition-0"

    @pytest.mark.asyncio
    async def test_multiple_workers_partition_isolation(self, sample_client):
        queue = PartitionedMemoryQueue(num_partitions=2)

        processed = {"worker-0": [], "worker-1": []}

        mock_user = Mock(id="admin", organization_id=sample_client.organization_id)

        def make_server(worker_key):
            s = Mock()
            s.send_messages = AsyncMock(
                side_effect=lambda **kwargs: processed[worker_key].append(kwargs["agent_id"])
            )
            s.client_manager = Mock()
            s.client_manager.get_client_by_id = AsyncMock(return_value=sample_client)
            return s

        mock_server_0 = make_server("worker-0")
        mock_server_1 = make_server("worker-1")

        worker_0 = QueueWorker(queue, server=mock_server_0, partition_id=0)
        worker_1 = QueueWorker(queue, server=mock_server_1, partition_id=1)

        with patch("mirix.queue.worker.UserManager") as MockUM:
            MockUM.return_value.get_admin_user = AsyncMock(return_value=mock_user)
            MockUM.return_value.get_user_by_id = AsyncMock(side_effect=Exception("not found"))
            MockUM.return_value.create_user = AsyncMock(return_value=mock_user)

            for i in range(5):
                msg = QueueMessage()
                msg.client_id = sample_client.id
                msg.agent_id = f"agent-p0-{i}"
                await queue._partitions[0].put(msg)

            for i in range(5):
                msg = QueueMessage()
                msg.client_id = sample_client.id
                msg.agent_id = f"agent-p1-{i}"
                await queue._partitions[1].put(msg)

            await worker_0.start()
            await worker_1.start()

            await asyncio.sleep(1.0)

            await worker_0.stop(close_queue=False)
            await worker_1.stop(close_queue=False)

        for agent_id in processed["worker-0"]:
            assert "p0" in agent_id

        for agent_id in processed["worker-1"]:
            assert "p1" in agent_id

        assert len(processed["worker-0"]) == 5
        assert len(processed["worker-1"]) == 5

    @pytest.mark.asyncio
    async def test_partitioned_queue_distributes_by_user_id(
        self, configure_workers, mock_server, sample_client, sample_messages
    ):
        manager = configure_workers(num_workers=4)

        processed_by_partition = {}

        original_send = mock_server.send_messages

        async def tracking_send(**kwargs):
            user = kwargs.get("user")
            user_id = user.id if user else "unknown"
            if user_id not in processed_by_partition:
                processed_by_partition[user_id] = []
            processed_by_partition[user_id].append(kwargs["agent_id"])
            return await original_send(**kwargs)

        mock_server.send_messages = tracking_send

        await manager.initialize(server=mock_server)

        user_ids = ["user-a", "user-b", "user-c", "user-d"]
        for user_id in user_ids:
            for i in range(3):
                await put_messages(
                    actor=sample_client,
                    agent_id=f"agent-{user_id}-{i}",
                    input_messages=sample_messages,
                    user_id=user_id,
                )

        await asyncio.sleep(2.0)

        total_processed = sum(len(v) for v in processed_by_partition.values())
        assert total_processed > 0

        await manager.cleanup()


# ============================================================================
# queue_util Tests
# ============================================================================


class TestQueueUtil:
    """Test queue utility functions"""

    @pytest.mark.asyncio
    async def test_put_messages_basic(self, clean_manager, sample_client, sample_messages):
        manager = clean_manager
        await manager.initialize()

        await put_messages(actor=sample_client, agent_id="agent-789", input_messages=sample_messages)

        msg = await manager._queue.get(timeout=1.0)
        assert msg.agent_id == "agent-789"
        assert msg.client_id == sample_client.id
        assert len(msg.input_messages) == len(sample_messages)

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_put_messages_with_options(self, clean_manager, sample_client, sample_messages):
        manager = clean_manager
        await manager.initialize()

        await put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=sample_messages,
            chaining=False,
            user_id="user-custom",
            verbose=True,
            filter_tags={"tag1": "value1"},
        )

        msg = await manager._queue.get(timeout=1.0)
        assert msg.chaining is False
        assert msg.user_id == "user-custom"
        assert msg.verbose is True
        assert dict(msg.filter_tags) == {"tag1": "value1"}

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_put_messages_with_block_filter_tags(self, clean_manager, sample_client, sample_messages):
        manager = clean_manager
        await manager.initialize()

        block_filter_tags = {"env": "staging", "team": "platform"}
        await put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=sample_messages,
            block_filter_tags=block_filter_tags,
        )

        msg = await manager._queue.get(timeout=1.0)
        assert msg.agent_id == "agent-789"
        assert hasattr(msg, "block_filter_tags") and msg.block_filter_tags
        assert dict(msg.block_filter_tags) == block_filter_tags

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_put_messages_role_mapping(self, clean_manager, sample_client):
        manager = clean_manager
        await manager.initialize()

        messages = [
            MessageCreate(role=MessageRole.user, content="User message"),
            MessageCreate(role=MessageRole.system, content="System message"),
        ]

        await put_messages(actor=sample_client, agent_id="agent-789", input_messages=messages)

        msg = await manager._queue.get(timeout=1.0)
        assert msg.input_messages[0].role == ProtoMessageCreate.ROLE_USER
        assert msg.input_messages[1].role == ProtoMessageCreate.ROLE_SYSTEM

        await manager.cleanup()


# ============================================================================
# Queue __init__ Tests
# ============================================================================


class TestQueueInit:
    """Test queue module initialization functions"""

    @pytest.mark.asyncio
    async def test_initialize_queue(self, clean_manager, mock_server):
        manager = clean_manager

        await initialize_queue(mock_server)

        assert manager.is_initialized
        assert manager._server == mock_server

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_save_without_init(self, clean_manager, sample_queue_message):
        manager = clean_manager
        assert not manager.is_initialized

        await save(sample_queue_message)
        assert manager.is_initialized

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_save_with_init(self, clean_manager, mock_server, sample_queue_message):
        manager = clean_manager
        await initialize_queue(mock_server)

        await save(sample_queue_message)

        retrieved = await manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id

        await manager.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================


class TestQueueIntegration:
    """Integration tests for the complete queue system"""

    @pytest.mark.asyncio
    async def test_end_to_end_message_flow(self, clean_manager, mock_server, sample_client, sample_messages):
        manager = clean_manager

        await initialize_queue(mock_server)

        await put_messages(
            actor=sample_client,
            agent_id="agent-integration",
            input_messages=sample_messages,
        )

        await asyncio.sleep(1.0)

        assert mock_server.send_messages.call_count >= 1

        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs["agent_id"] == "agent-integration"

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_block_filter_tags_passed_through_to_send_messages(
        self, clean_manager, mock_server, sample_client, sample_messages
    ):
        manager = clean_manager
        await initialize_queue(mock_server)

        block_filter_tags = {"env": "staging", "team": "platform"}
        await put_messages(
            actor=sample_client,
            agent_id="agent-block-tags",
            input_messages=sample_messages,
            block_filter_tags=block_filter_tags,
        )

        await asyncio.sleep(1.0)

        assert mock_server.send_messages.call_count >= 1
        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs.get("block_filter_tags") == block_filter_tags

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_messages_processing(self, clean_manager, mock_server, sample_client, sample_messages):
        manager = clean_manager
        await initialize_queue(mock_server)

        for i in range(5):
            await put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages,
            )

        await asyncio.sleep(2.0)

        assert mock_server.send_messages.call_count >= 5

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_worker_handles_processing_errors(
        self, clean_manager, mock_server, sample_client, sample_messages
    ):
        manager = clean_manager

        mock_server.send_messages = AsyncMock(side_effect=Exception("Processing error"))

        await initialize_queue(mock_server)

        await put_messages(actor=sample_client, agent_id="agent-error", input_messages=sample_messages)

        await asyncio.sleep(1.0)

        assert manager._workers[0]._running

        await manager.cleanup()


# ============================================================================
# Performance Tests
# ============================================================================


class TestQueuePerformance:
    """Performance tests for queue operations"""

    @pytest.mark.asyncio
    async def test_enqueue_performance(self, clean_manager, sample_client, sample_messages):
        manager = clean_manager
        await manager.initialize()

        import time

        start = time.time()

        for i in range(100):
            await put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages,
            )

        elapsed = time.time() - start
        assert elapsed < 1.0

        print(f"\nEnqueued 100 messages in {elapsed:.3f}s")

        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self, sample_client, sample_messages, mock_server):
        await initialize_queue(server=mock_server)
        manager = get_manager()

        processed_messages = []

        async def mock_send_messages(**kwargs):
            processed_messages.append(kwargs["agent_id"])
            return None

        mock_server.send_messages = mock_send_messages

        async def enqueue_messages(task_id, count):
            for i in range(count):
                await put_messages(
                    actor=sample_client,
                    agent_id=f"agent-{task_id}-{i}",
                    input_messages=sample_messages,
                )

        tasks = [asyncio.create_task(enqueue_messages(i, 20)) for i in range(5)]
        await asyncio.gather(*tasks)

        await asyncio.sleep(1.0)

        assert len(processed_messages) == 100

        await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
