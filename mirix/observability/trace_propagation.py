"""
Trace context propagation for distributed tracing across queue systems.

Works with both in-memory queues and Kafka using Protocol Buffer fields.
"""

from typing import Any

from mirix.log import get_logger
from mirix.observability.context import get_trace_context, set_trace_context

logger = get_logger(__name__)

# For dict-based message metadata (e.g., Kafka headers/payloads in older codepaths/tests)
TRACE_METADATA_KEY = "trace_metadata"


def add_trace_to_queue_message(message: Any) -> Any:
    """
    Add current trace context to queue message (Protocol Buffer).

    Works for both in-memory and Kafka queues since both use the same
    protobuf schema.

    Args:
        message: QueueMessage protobuf instance

    Returns:
        The same message with trace fields populated
    """
    context = get_trace_context()

    # Only add if we have an active trace
    if not context.get("trace_id"):
        logger.debug("No active trace context when queueing message - LangFuse tracing will not propagate to worker")
        return message

    # Set trace fields on protobuf message
    if context.get("trace_id"):
        message.langfuse_trace_id = context["trace_id"]
    if context.get("observation_id"):
        message.langfuse_observation_id = context["observation_id"]
    if context.get("session_id"):
        message.langfuse_session_id = context["session_id"]
    if context.get("user_id"):
        message.langfuse_user_id = context["user_id"]

    logger.debug(
        f"Added trace context to queue message: trace_id={context.get('trace_id')}, "
        f"observation_id={context.get('observation_id')}"
    )

    return message


def restore_trace_from_queue_message(message: Any) -> bool:
    """
    Restore trace context from queue message (Protocol Buffer).

    Works for both in-memory and Kafka queues.

    Args:
        message: QueueMessage protobuf instance

    Returns:
        True if trace context was restored, False otherwise
    """
    # Check if message has trace fields
    if not hasattr(message, "langfuse_trace_id"):
        logger.debug("Message does not have trace fields (old schema version?)")
        return False

    trace_id = message.langfuse_trace_id if message.HasField("langfuse_trace_id") else None

    if not trace_id:
        logger.debug("No trace ID in queue message")
        return False

    # Restore trace context
    observation_id = message.langfuse_observation_id if message.HasField("langfuse_observation_id") else None
    session_id = message.langfuse_session_id if message.HasField("langfuse_session_id") else None
    user_id = message.langfuse_user_id if message.HasField("langfuse_user_id") else None

    set_trace_context(
        trace_id=trace_id,
        observation_id=observation_id,
        session_id=session_id,
        user_id=user_id,
    )

    logger.debug(
        f"Restored trace context from queue message: trace_id={trace_id}, "
        f"observation_id={observation_id}, session_id={session_id}"
    )

    return True


# ============================================================================
# Backwards-compatible helpers (dict-based message propagation)
# ============================================================================


def serialize_trace_context() -> dict | None:
    """
    Serialize current trace context to a dict for transport (e.g., Kafka message metadata).

    Returns None if no active trace_id is set.
    """
    context = get_trace_context()
    if not context.get("trace_id"):
        return None
    return {
        "trace_id": context.get("trace_id"),
        "observation_id": context.get("observation_id"),
        "session_id": context.get("session_id"),
        "user_id": context.get("user_id"),
    }


def deserialize_trace_context(message: dict) -> bool:
    """
    Restore trace context from a dict-based message (expects TRACE_METADATA_KEY).

    Returns True if context was restored, False otherwise.
    """
    if not isinstance(message, dict):
        return False
    metadata = message.get(TRACE_METADATA_KEY)
    if not isinstance(metadata, dict):
        return False
    trace_id = metadata.get("trace_id")
    if not trace_id:
        return False
    set_trace_context(
        trace_id=trace_id,
        observation_id=metadata.get("observation_id"),
        session_id=metadata.get("session_id"),
        user_id=metadata.get("user_id"),
    )
    return True


def add_trace_to_message(message: dict) -> dict:
    """
    Add current trace context into a dict-based message under TRACE_METADATA_KEY.

    Returns the message (same object) with metadata added when a trace exists.
    """
    serialized = serialize_trace_context()
    if not serialized:
        return message
    message[TRACE_METADATA_KEY] = serialized
    return message
