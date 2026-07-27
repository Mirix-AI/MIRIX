"""Consumer-entry normalization for messages arriving off the queue.

Covers the ECMS-73 direct-queue-ingestion contract: producers may write
straight onto the Kafka topic (bypassing the ECMS HTTP layer entirely), so the
consumer entry point must perform the normalization that layer used to own —
filter_tags/block_filter_tags canonicalization, memory_source_id and tid
backfill — and dead-letter (not redeliver) deterministically-malformed
messages.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_util import (
    normalize_and_validate_incoming_message,
    normalize_filter_tags_struct,
)


class TestNormalizeFilterTagsStruct:
    def test_scalar_becomes_single_element_list(self):
        assert normalize_filter_tags_struct({"env": "prod"}) == {"env": ["prod"]}

    def test_list_elements_stringified(self):
        assert normalize_filter_tags_struct({"n": [1, 2]}) == {"n": ["1", "2"]}

    def test_list_of_strings_passes_through(self):
        assert normalize_filter_tags_struct({"team": ["a", "b"]}) == {"team": ["a", "b"]}

    def test_nested_dict_value_rejected(self):
        with pytest.raises(ValueError, match="nested dict"):
            normalize_filter_tags_struct({"bad": {"x": 1}})

    def test_nested_container_in_list_rejected(self):
        with pytest.raises(ValueError, match="scalars"):
            normalize_filter_tags_struct({"bad": [["x"]]})


class TestNormalizeAndValidateIncomingMessage:
    def _message(self) -> QueueMessage:
        msg = QueueMessage()
        msg.client_id = "client-1"
        msg.agent_id = "agent-1"
        return msg

    def test_backfills_memory_source_id_and_tid(self):
        msg = self._message()
        normalize_and_validate_incoming_message(msg)
        assert msg.memory_source_id.startswith("src-")
        assert msg.tid

    def test_preserves_producer_supplied_ids(self):
        msg = self._message()
        msg.memory_source_id = "src-mine"
        msg.tid = "tid-mine"
        normalize_and_validate_incoming_message(msg)
        assert msg.memory_source_id == "src-mine"
        assert msg.tid == "tid-mine"

    def test_normalizes_scalar_filter_tag_values_to_lists(self):
        from google.protobuf.json_format import MessageToDict

        msg = self._message()
        msg.filter_tags.update({"env": "prod", "team": ["payments"]})
        normalize_and_validate_incoming_message(msg)
        assert MessageToDict(msg.filter_tags) == {"env": ["prod"], "team": ["payments"]}

    def test_normalizes_block_filter_tags_too(self):
        from google.protobuf.json_format import MessageToDict

        msg = self._message()
        msg.block_filter_tags.update({"env": "staging"})
        normalize_and_validate_incoming_message(msg)
        assert MessageToDict(msg.block_filter_tags) == {"env": ["staging"]}

    def test_malformed_filter_tags_raise_but_ids_are_still_backfilled(self):
        msg = self._message()
        msg.filter_tags.update({"bad": {"nested": "dict"}})
        with pytest.raises(ValueError, match="filter_tags"):
            normalize_and_validate_incoming_message(msg)
        # Backfill runs before validation so the failure can be finalized
        # against a real memory_source_id.
        assert msg.memory_source_id.startswith("src-")
        assert msg.tid


@pytest.mark.asyncio
async def test_malformed_message_finalizes_permanent_and_does_not_raise(monkeypatch):
    """A message with malformed filter_tags dead-letters through the normal
    classify -> PERMANENT -> finalize chokepoint (so Numaflow acks it) instead
    of raising out of process_external_message and redelivering forever."""
    from mirix.queue import process_external_message
    from mirix.queue.error_policy import SaveOutcome

    queue_message = QueueMessage()
    queue_message.client_id = "client-1"
    queue_message.agent_id = "agent-1"
    queue_message.filter_tags.update({"bad": {"nested": "dict"}})

    monkeypatch.setattr(
        "mirix.queue.queue_util.deserialize_queue_message",
        lambda raw, format=None: queue_message,
    )

    worker = Mock()
    worker.process_external_message = AsyncMock()

    fake_manager = Mock()
    fake_manager.is_initialized = True
    fake_manager._workers = [worker]
    monkeypatch.setattr("mirix.queue._manager", fake_manager)

    finalize = AsyncMock()
    fake_source_manager_cls = Mock(return_value=Mock(finalize_source=finalize))
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        fake_source_manager_cls,
    )

    refusal_span = Mock()
    monkeypatch.setattr(
        "mirix.observability.skip_spans.emit_refused_to_process_span",
        refusal_span,
    )

    await process_external_message(b"ignored-by-stub-deserializer")

    # The agent step never ran, and the (backfilled) source id was finalized
    # as a permanent failure.
    worker.process_external_message.assert_not_awaited()
    finalize.assert_awaited_once()
    source_id, outcome = finalize.await_args.args
    assert source_id.startswith("src-")
    assert outcome == SaveOutcome.PERMANENT_FAILURE

    # The refusal is made visible in Langfuse, same pattern as no-write-scope.
    refusal_span.assert_called_once()
    assert refusal_span.call_args.kwargs["reason"] == "malformed-message"


@pytest.mark.asyncio
async def test_worker_unified_messages_field_flattens_and_persists_per_turn():
    """A message using the unified `messages` field (single wire copy) yields
    BOTH a packed agent-input and per-turn source_messages dicts to
    server.send_messages — the work the producer used to do twice."""
    import mirix.queue.worker as worker_module
    from mirix.queue.worker import QueueWorker

    msg = QueueMessage()
    msg.client_id = "client-1"
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    turn1 = msg.messages.add()
    turn1.role = ProtoMessageCreate.ROLE_USER
    turn1.text_content = "hi there"
    turn1.external_message_id = "m1"
    turn2 = msg.messages.add()
    turn2.role = ProtoMessageCreate.ROLE_ASSISTANT
    turn2.text_content = "hello!"
    turn2.external_message_id = "m2"

    fake_actor = MagicMock()
    fake_actor.id = "client-1"
    fake_actor.organization_id = "org-1"
    fake_actor.write_scope = "test-scope"
    fake_user = MagicMock()

    server = MagicMock()
    server.client_manager = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=fake_actor)
    server.send_messages = AsyncMock(return_value=None)

    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

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

    # Packed agent input: one MessageCreate with the whole transcript coalesced
    # into a single text block ([USER]/[ASSISTANT] markers inline).
    input_messages = kwargs["input_messages"]
    assert len(input_messages) == 1
    assert len(input_messages[0].content) == 1
    assert input_messages[0].content[0].text == "[USER]\nhi there\n[ASSISTANT]\nhello!"

    # Per-turn provenance dicts with role + external_message_id intact.
    source_messages = kwargs["source_messages"]
    assert source_messages == [
        {"role": "user", "content": "hi there", "external_message_id": "m1"},
        {"role": "assistant", "content": "hello!", "external_message_id": "m2"},
    ]


@pytest.mark.asyncio
async def test_batch_worker_path_normalizes_and_rejects_like_external(monkeypatch):
    """The internal consumer (BatchQueueWorker: kafka-manual / in-memory modes)
    goes through the SAME dispatch funnel as the external Numaflow path — a
    malformed message is refused, span-marked, and finalized PERMANENT there
    too, not just on the external path."""
    from mirix.queue.error_policy import SaveOutcome
    from mirix.queue.worker import BatchQueueWorker

    msg = QueueMessage()
    msg.client_id = "client-1"
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    msg.filter_tags.update({"bad": {"nested": "dict"}})

    finalize = AsyncMock()
    fake_source_manager_cls = Mock(return_value=Mock(finalize_source=finalize))
    monkeypatch.setattr(
        "mirix.services.memory_source_manager.MemorySourceManager",
        fake_source_manager_cls,
    )
    refusal_span = Mock()
    monkeypatch.setattr(
        "mirix.observability.skip_spans.emit_refused_to_process_span",
        refusal_span,
    )

    worker = BatchQueueWorker.__new__(BatchQueueWorker)
    worker._server = MagicMock()
    worker._partition_id = None
    worker.process_external_message = AsyncMock()

    # Drive the per-message chokepoint BatchQueueWorker._run_one_iteration uses.
    from mirix.queue.worker import dispatch_incoming_message

    await dispatch_incoming_message(worker, msg)

    worker.process_external_message.assert_not_awaited()
    finalize.assert_awaited_once()
    source_id, outcome = finalize.await_args.args
    assert source_id.startswith("src-")
    assert outcome == SaveOutcome.PERMANENT_FAILURE
    refusal_span.assert_called_once()
    assert refusal_span.call_args.kwargs["reason"] == "malformed-message"


@pytest.mark.asyncio
async def test_internal_path_normalizes_filter_tags_before_agent():
    """A message consumed on the internal path (in-memory / kafka-manual) gets
    the same filter_tags canonicalization as the external path: scalars arrive
    at the agent as single-element lists (with worker-injected scalar scope)."""
    import mirix.queue.worker as worker_module
    from mirix.queue.worker import QueueWorker, dispatch_incoming_message

    msg = QueueMessage()
    msg.client_id = "client-1"
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    msg.memory_source_id = "src-normalize-internal"
    turn = msg.messages.add()
    turn.role = ProtoMessageCreate.ROLE_USER
    turn.text_content = "hi"
    msg.filter_tags.update({"env": "prod"})

    fake_actor = MagicMock()
    fake_actor.id = "client-1"
    fake_actor.organization_id = "org-1"
    fake_actor.write_scope = "test-scope"
    fake_user = MagicMock()

    server = MagicMock()
    server.client_manager = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=fake_actor)
    server.send_messages = AsyncMock(return_value=None)

    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

    class _FakeUserManager:
        async def get_user_by_id(self, _):
            return fake_user

        async def get_admin_user(self):
            return fake_user

    from unittest.mock import patch as _patch

    original_user_manager = worker_module.UserManager
    worker_module.UserManager = _FakeUserManager
    try:
        with _patch(
            "mirix.services.memory_source_manager.MemorySourceManager",
            return_value=Mock(finalize_source=AsyncMock()),
        ):
            await dispatch_incoming_message(worker, msg)
    finally:
        worker_module.UserManager = original_user_manager

    server.send_messages.assert_awaited_once()
    filter_tags = server.send_messages.call_args.kwargs["filter_tags"]
    assert filter_tags["env"] == ["prod"]
    assert filter_tags["scope"] == "test-scope"


@pytest.mark.asyncio
async def test_worker_legacy_dual_array_fallback_still_works():
    """Messages already in flight (legacy input_messages + source_messages
    pair) keep processing unchanged during rollout."""
    import mirix.queue.worker as worker_module
    from mirix.queue.worker import QueueWorker

    msg = QueueMessage()
    msg.client_id = "client-1"
    msg.agent_id = "agent-1"
    msg.user_id = "user-1"
    packed = msg.input_messages.add()
    packed.role = ProtoMessageCreate.ROLE_USER
    packed.text_content = "[USER]\nhi there"
    original = msg.source_messages.add()
    original.role = ProtoMessageCreate.ROLE_USER
    original.text_content = "hi there"
    original.external_message_id = "m1"

    fake_actor = MagicMock()
    fake_actor.id = "client-1"
    fake_actor.organization_id = "org-1"
    fake_actor.write_scope = "test-scope"
    fake_user = MagicMock()

    server = MagicMock()
    server.client_manager = MagicMock()
    server.client_manager.get_client_by_id = AsyncMock(return_value=fake_actor)
    server.send_messages = AsyncMock(return_value=None)

    worker = QueueWorker.__new__(QueueWorker)
    worker._server = server

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

    kwargs = server.send_messages.call_args.kwargs
    assert len(kwargs["input_messages"]) == 1
    assert kwargs["source_messages"] == [
        {"role": "user", "content": "hi there", "external_message_id": "m1"},
    ]


class TestJsonForwardCompatibility:
    """JSON wire-format skew safety (the ECMS-73 rollout incident).

    Under KAFKA_SERIALIZATION_FORMAT=json, ParseDict was strict about unknown
    fields, so a producer emitting a newer schema than the consumer wedged the
    consumer in a ParseError redelivery loop. JSON deserialization must match
    protobuf binary semantics: unknown fields are skipped, not fatal.
    """

    def test_unknown_fields_in_json_are_ignored(self):
        import json

        from mirix.queue.queue_util import deserialize_queue_message

        blob = json.dumps(
            {
                "client_id": "client-1",
                "agent_id": "agent-1",
                "a_field_from_the_future": [{"x": 1}],
            }
        ).encode("utf-8")

        msg = deserialize_queue_message(blob, format="json")
        assert msg.client_id == "client-1"
        assert msg.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_undeserializable_message_is_dropped_not_redelivered(self, monkeypatch):
        """Poison bytes ack (with an ERROR log) instead of raising out of
        process_external_message, which would redeliver forever."""
        from mirix.queue import process_external_message

        worker = Mock()
        worker.process_external_message = AsyncMock()

        fake_manager = Mock()
        fake_manager.is_initialized = True
        fake_manager._workers = [worker]
        monkeypatch.setattr("mirix.queue._manager", fake_manager)

        # Returns normally (ack) — no exception escapes to the broker layer.
        await process_external_message(b"\x00not-valid-json-or-protobuf\xff")

        worker.process_external_message.assert_not_awaited()
