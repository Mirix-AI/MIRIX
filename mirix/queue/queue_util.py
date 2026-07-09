import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from google.protobuf.json_format import MessageToDict, ParseDict

import mirix.queue as queue
from mirix.observability import add_trace_to_queue_message
from mirix.queue.message_pb2 import DirectMemoryWrite as ProtoDirectMemoryWrite
from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
from mirix.queue.message_pb2 import QueueMessage
from mirix.schemas.client import Client

logger = logging.getLogger(__name__)

_ROLE_TO_PROTO = {
    "user": ProtoMessageCreate.ROLE_USER,
    "system": ProtoMessageCreate.ROLE_SYSTEM,
    "assistant": ProtoMessageCreate.ROLE_ASSISTANT,
}


def _dict_message_to_proto(msg_dict: dict) -> ProtoMessageCreate:
    """Convert one raw per-turn message dict to a protobuf MessageCreate.

    The producer-side counterpart of the worker's
    _convert_proto_source_message_to_dict — the single place that knows the
    wire shape of a per-turn message dict: {"role", "content",
    "external_message_id"?, "occurred_at"?, "metadata"?}. `role` may be
    "user", "system", or "assistant" (unlike the old packed input_messages
    role, which collapsed to one value). List content is reduced to its text
    parts, matching the old queue boundary's behavior (structured/image
    content never survived serialization to the queue).
    """
    proto_msg = ProtoMessageCreate()
    proto_msg.role = _ROLE_TO_PROTO.get(msg_dict.get("role", "user"), ProtoMessageCreate.ROLE_UNSPECIFIED)

    content = msg_dict.get("content", "")
    if isinstance(content, str):
        proto_msg.text_content = content
    elif isinstance(content, list):
        text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        proto_msg.text_content = "\n".join(text_parts)

    if msg_dict.get("name"):
        proto_msg.name = msg_dict["name"]
    if msg_dict.get("otid"):
        proto_msg.otid = msg_dict["otid"]
    if msg_dict.get("sender_id"):
        proto_msg.sender_id = msg_dict["sender_id"]
    if msg_dict.get("group_id"):
        proto_msg.group_id = msg_dict["group_id"]
    if msg_dict.get("external_message_id"):
        proto_msg.external_message_id = msg_dict["external_message_id"]
    if msg_dict.get("occurred_at"):
        proto_msg.message_occurred_at = msg_dict["occurred_at"]
    if msg_dict.get("metadata"):
        proto_msg.message_metadata.update(msg_dict["metadata"])

    return proto_msg


# Queue message serialization utilities


def serialize_queue_message(message: QueueMessage, format: str = "protobuf") -> bytes:
    """
    Serialize QueueMessage to bytes in the specified format.

    Args:
        message: QueueMessage protobuf to serialize
        format: Serialization format - 'protobuf' or 'json'

    Returns:
        Serialized message bytes

    Raises:
        ValueError: If format is not supported
    """
    if format == "json":
        message_dict = MessageToDict(message, preserving_proto_field_name=True)
        return json.dumps(message_dict).encode("utf-8")
    elif format == "protobuf":
        return message.SerializeToString()
    else:
        raise ValueError(f"Unsupported serialization format: {format}")


def deserialize_queue_message(serialized_msg: bytes, format: str = "protobuf") -> QueueMessage:
    """
    Deserialize bytes to QueueMessage in the specified format.

    Args:
        serialized_msg: Serialized message bytes
        format: Serialization format - 'protobuf' or 'json'

    Returns:
        QueueMessage protobuf object

    Raises:
        ValueError: If format is not supported or deserialization fails
    """
    queue_message = QueueMessage()

    try:
        if format == "json":
            message_dict = json.loads(serialized_msg.decode("utf-8"))
            return ParseDict(message_dict, queue_message)
        elif format == "protobuf":
            queue_message.ParseFromString(serialized_msg)
            return queue_message
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    except Exception as e:
        raise ValueError(f"Failed to deserialize message ({format} format): {e}") from e


def normalize_filter_tags_struct(filter_tags: dict) -> Dict[str, List[str]]:
    """Coerce a raw filter_tags dict to the canonical Dict[str, List[str]] shape.

    A scalar becomes a single-element list, list elements are stringified, and
    an already list-of-strings value is left as-is. Historically this
    normalization only existed in ECMS's Pydantic request validator
    (app/service/utils/filter_tags.py) — a message arriving straight off the
    queue (bypassing that HTTP layer entirely) never got it. Centralized here
    so every consumption path (REST-API-produced or Kafka-direct-produced)
    normalizes identically.

    Raises ValueError if a key isn't a string, or a list element isn't a
    primitive that can be stringified sensibly (dict/list nested inside a
    filter_tags value) — this is a shape a caller must fix, not silently
    coerce, since a malformed shape breaks IPS-R queries silently downstream.
    """
    normalized: Dict[str, List[str]] = {}
    for key, raw in filter_tags.items():
        if not isinstance(key, str):
            raise ValueError(f"filter_tags key must be a string, got {type(key).__name__}: {key!r}")
        if isinstance(raw, dict):
            raise ValueError(f"filter_tags[{key!r}] must be a scalar or list of scalars, got a nested dict")
        if isinstance(raw, list):
            if any(isinstance(el, (dict, list)) for el in raw):
                raise ValueError(f"filter_tags[{key!r}] list elements must be scalars, got a nested dict/list")
            normalized[key] = [str(element) for element in raw]
        else:
            normalized[key] = [str(raw)]
    return normalized


def normalize_and_validate_incoming_message(queue_message: QueueMessage) -> QueueMessage:
    """Consumer-entry normalization/validation, run once per message right after
    deserialization — before the message enters the retry/dispatch pipeline.

    Covers the pieces of save-path work that used to exist ONLY at the ECMS
    HTTP layer (never reachable by a message produced straight onto the Kafka
    topic):

    * filter_tags / block_filter_tags: normalize to Dict[str, List[str]] and
      validate the shape (same canonicalization ECMS's request validator
      applies to both fields). Raises (wrapped as a permanent, non-retryable
      failure by the caller) on a malformed shape rather than silently
      miscoercing it.
    * memory_source_id: backfilled if the producer omitted it. This has to
      happen here (not deeper in worker.py) because `process_external_message`
      reads `queue_message.memory_source_id` immediately after deserializing,
      before the worker ever runs, to drive the idempotency finalize
      chokepoint — a message that reaches that point without one would never
      get its `processing_complete` flag written.
    * tid: backfilled with a fresh id (and a warning log) if the producer
      omitted it, since there is no HTTP middleware on this path to have
      generated one already.

    Mutates and returns the same QueueMessage instance for convenience. ID/tid
    backfill runs FIRST and unconditionally — even when filter_tags validation
    below fails — so a rejected message still carries a memory_source_id and
    tid into dispatch_save's classify/finalize/log path (see the caller in
    mirix/queue/__init__.py for why the raise there must happen inside the
    dispatch_save-wrapped step, not before it).
    """
    if not queue_message.HasField("memory_source_id") or not queue_message.memory_source_id:
        queue_message.memory_source_id = f"src-{uuid.uuid4()}"
        logger.debug(
            "Backfilled memory_source_id=%s for incoming message (agent_id=%s) — producer omitted it",
            queue_message.memory_source_id,
            queue_message.agent_id,
        )

    if not queue_message.HasField("tid") or not queue_message.tid:
        queue_message.tid = uuid.uuid4().hex
        logger.warning(
            "Incoming queue message (agent_id=%s, memory_source_id=%s) had no tid — "
            "generated a fallback. Direct-to-Kafka producers should supply their own "
            "tid for end-to-end log/trace correlation.",
            queue_message.agent_id,
            queue_message.memory_source_id,
        )

    for field_name in ("filter_tags", "block_filter_tags"):
        struct_field = getattr(queue_message, field_name)
        if queue_message.HasField(field_name) and struct_field:
            raw_tags = MessageToDict(struct_field)
            try:
                normalized = normalize_filter_tags_struct(raw_tags)
            except ValueError as e:
                raise ValueError(f"Invalid {field_name} on incoming queue message: {e}") from e
            struct_field.Clear()
            struct_field.update(normalized)

    return queue_message


async def put_messages(
    actor: Client,
    agent_id: str,
    messages: List[dict],
    chaining: Optional[bool] = True,
    user_id: Optional[str] = None,
    verbose: Optional[bool] = None,
    filter_tags: Optional[dict] = None,
    block_filter_tags: Optional[dict] = None,
    block_filter_tags_update_mode: Optional[str] = "merge",
    use_cache: bool = True,
    occurred_at: Optional[str] = None,
    memory_source_id: Optional[str] = None,
    external_id: Optional[str] = None,
    external_thread_id: Optional[str] = None,
    source_type: Optional[str] = None,
    source_system: Optional[str] = None,
    source_metadata: Optional[dict] = None,
    summary: Optional[str] = None,
    summarize: bool = False,
    direct_writes: Optional[List[Dict[str, Any]]] = None,
):
    """
    Create QueueMessage protobuf and send to queue.

    Args:
        actor: The Client performing the action (for auth/write operations)
               Client ID is derived from actor.id
        agent_id: ID of the agent to send message to
        messages: Per-turn conversation messages, sent exactly once on the wire.
            Each entry is a dict {"role": "user"|"system"|"assistant", "content":
            str | list, "external_message_id"?, "occurred_at"?, "metadata"?}. The
            worker derives BOTH the packed agent-input and the source_messages
            provenance records from this single list — callers no longer flatten
            or duplicate messages before enqueueing.
        chaining: Enable/disable chaining
        user_id: Optional user ID (end-user ID)
        verbose: Enable verbose logging
        filter_tags: Filter tags dictionary
        block_filter_tags: Optional dict; applied only when blocks are created (e.g. from default template)
        block_filter_tags_update_mode: "merge" (default) or "replace" for existing block filter_tags
        use_cache: Control Redis cache behavior
        occurred_at: Optional ISO 8601 timestamp string for episodic memory
        memory_source_id: Optional pre-generated "src-{uuid4}" for citation
            tracking. NOT generated here — callers that want provenance (e.g.
            rest_api.add_memory) supply their own; messages that reach the
            consumer without one are backfilled there.
        direct_writes: Optional list of direct memory writes. Each entry is a dict
            {"memory_type": str, "payload": dict}. When set, the meta-agent skips
            LLM dispatch and calls the registered handler per entry instead.

    """
    logger.debug("Creating queue message for agent_id=%s, client_id=%s", agent_id, actor.id)

    if not actor or not actor.id:
        raise ValueError(
            f"Cannot queue message: actor is None or has no ID. "
            f"actor={actor}, actor.id={actor.id if actor else 'N/A'}"
        )

    # Build the QueueMessage
    queue_msg = QueueMessage()

    queue_msg.client_id = actor.id

    queue_msg.agent_id = agent_id
    queue_msg.messages.extend(_dict_message_to_proto(m) for m in (messages or []))

    # Optional fields
    if chaining is not None:
        queue_msg.chaining = chaining
    if user_id:
        queue_msg.user_id = user_id
    if verbose is not None:
        queue_msg.verbose = verbose

    # Convert dict to Struct for filter_tags
    if filter_tags:
        queue_msg.filter_tags.update(filter_tags)

    # Optional block_filter_tags (applied only when blocks are created)
    if block_filter_tags:
        queue_msg.block_filter_tags.update(block_filter_tags)

    if block_filter_tags_update_mode:
        queue_msg.block_filter_tags_update_mode = block_filter_tags_update_mode

    # Set use_cache
    queue_msg.use_cache = use_cache

    # Set occurred_at if provided
    if occurred_at is not None:
        queue_msg.occurred_at = occurred_at

    # Memory source fields
    if memory_source_id is not None:
        queue_msg.memory_source_id = memory_source_id
    if external_id is not None:
        queue_msg.external_id = external_id
    if external_thread_id is not None:
        queue_msg.external_thread_id = external_thread_id
    if source_type is not None:
        queue_msg.source_type = source_type
    if source_system is not None:
        queue_msg.source_system = source_system
    if source_metadata:
        queue_msg.source_metadata.update(source_metadata)
    if summary is not None:
        queue_msg.summary = summary
    if summarize:
        queue_msg.summarize = summarize

    # Serialize direct_writes — each entry is {memory_type: str, payload: dict}.
    # The worker deserializes these back into dicts and passes them to the
    # meta-agent, which short-circuits LLM dispatch when set.
    if direct_writes:
        for write in direct_writes:
            proto_write = ProtoDirectMemoryWrite()
            proto_write.memory_type = write["memory_type"]
            proto_write.payload_json = json.dumps(write["payload"])
            queue_msg.direct_writes.append(proto_write)

    # Add LangFuse trace context for distributed tracing
    queue_msg = add_trace_to_queue_message(queue_msg)

    # Send to queue
    logger.debug(
        "Sending message to queue: agent_id=%s, messages_count=%s, occurred_at=%s",
        agent_id,
        len(messages or []),
        occurred_at,
    )
    await queue.save(queue_msg)
    logger.debug("Message successfully sent to queue")
