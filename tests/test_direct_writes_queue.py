"""Tests for direct_writes plumbing through put_messages into QueueMessage."""

import json

import pytest


@pytest.mark.asyncio
async def test_put_messages_serializes_direct_writes(monkeypatch):
    """direct_writes arg → QueueMessage.direct_writes populated with memory_type + payload_json."""
    from mirix.queue.message_pb2 import QueueMessage
    from mirix.queue.queue_util import put_messages

    captured = {}

    async def fake_save(message: QueueMessage) -> None:
        captured["msg"] = message

    monkeypatch.setattr("mirix.queue.queue_util.queue.save", fake_save)

    class _Actor:
        id = "client-1"

    await put_messages(
        actor=_Actor(),
        agent_id="agent-1",
        messages=[],
        chaining=None,
        user_id="user-1",
        verbose=False,
        filter_tags={"scope": "s"},
        block_filter_tags=None,
        block_filter_tags_update_mode="merge",
        use_cache=True,
        occurred_at="2026-04-17T10:00:00Z",
        memory_source_id="src-abcdef12",
        external_id="ext-1",
        external_thread_id=None,
        source_type="engagement",
        source_system="test-system",
        source_metadata={"display_id": "D1"},
        summary=None,
        summarize=False,
        direct_writes=[
            {
                "memory_type": "episodic",
                "payload": {
                    "items": [
                        {
                            "event_type": "e",
                            "summary": "s",
                            "details": "d",
                            "actor": "system",
                            "occurred_at": "2026-04-17T10:00:00Z",
                        }
                    ]
                },
            },
        ],
    )

    msg = captured["msg"]
    assert len(msg.direct_writes) == 1
    assert msg.direct_writes[0].memory_type == "episodic"
    payload = json.loads(msg.direct_writes[0].payload_json)
    assert payload["items"][0]["event_type"] == "e"
    assert payload["items"][0]["actor"] == "system"


@pytest.mark.asyncio
async def test_worker_extracts_direct_writes_from_queue_message():
    """Worker _process_message_async → server.send_messages receives direct_writes as a list of dicts."""
    from unittest.mock import AsyncMock, MagicMock

    from mirix.queue.message_pb2 import QueueMessage
    from mirix.queue.worker import QueueWorker

    msg = QueueMessage()
    msg.client_id = "client-1"
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    write = msg.direct_writes.add()
    write.memory_type = "episodic"
    write.payload_json = json.dumps({"event_type": "e", "summary": "s", "details": "d", "event_actor": "system"})

    fake_actor = MagicMock()
    fake_actor.id = "client-1"
    fake_actor.organization_id = "org-1"
    fake_actor.write_scope = "test-scope"  # worker derives filter_tags["scope"] from this
    fake_user = MagicMock()

    server = MagicMock()
    server.client_manager = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=fake_actor)
    server.send_messages = AsyncMock(return_value=None)

    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

    # Stub UserManager so actor/user resolution succeeds.
    import mirix.queue.worker as worker_module

    class _FakeUserManager:
        async def get_user_by_id(self, _):
            return fake_user

        async def get_admin_user(self):
            return fake_user

    original_user_manager = worker_module.UserManager
    worker_module.UserManager = _FakeUserManager
    try:
        await worker._process_message_async(msg)
    finally:
        worker_module.UserManager = original_user_manager

    server.send_messages.assert_awaited_once()
    kwargs = server.send_messages.call_args.kwargs
    assert "direct_writes" in kwargs, f"direct_writes missing from send_messages kwargs: {list(kwargs)}"
    assert kwargs["direct_writes"] == [
        {
            "memory_type": "episodic",
            "payload": {"event_type": "e", "summary": "s", "details": "d", "event_actor": "system"},
        }
    ]


@pytest.mark.asyncio
async def test_server_step_sets_direct_writes_on_agent():
    """AsyncServer._step with direct_writes sets mirix_agent.direct_writes."""
    from unittest.mock import AsyncMock, MagicMock

    from mirix.schemas.agent import AgentType
    from mirix.server.server import AsyncServer

    # Build an AsyncServer instance without running __init__
    server = AsyncServer.__new__(AsyncServer)
    server.max_chaining_steps = None
    server.chaining = False

    fake_agent = MagicMock()
    fake_agent.agent_state = MagicMock()
    fake_agent.agent_state.agent_type = AgentType.meta_memory_agent
    fake_agent.interface = MagicMock()
    fake_agent.interface.streaming_mode = False
    fake_agent.step = AsyncMock(return_value=MagicMock(step_count=0))

    server.load_agent = AsyncMock(return_value=fake_agent)

    await server._step(
        actor=MagicMock(id="c-1"),
        agent_id="a-1",
        input_messages=[],
        user=MagicMock(id="u-1"),
        direct_writes=[{"memory_type": "episodic", "payload": {"x": 1}}],
    )

    assert fake_agent.direct_writes == [{"memory_type": "episodic", "payload": {"x": 1}}]
