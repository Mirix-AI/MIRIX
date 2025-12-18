"""
Trace context propagation for distributed tracing across queue systems.

Works with both in-memory queues and Kafka using Protocol Buffer fields.
"""

from typing import Any
from mirix.observability.context import get_trace_context, set_trace_context
from mirix.log import get_logger

logger = get_logger(__name__)

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
    
    logger.debug(f"Added trace context to queue message: trace_id={context.get('trace_id')}")
    
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
    if not hasattr(message, 'langfuse_trace_id'):
        logger.debug("Message does not have trace fields (old schema version?)")
        return False
    
    trace_id = message.langfuse_trace_id if message.HasField('langfuse_trace_id') else None
    
    if not trace_id:
        logger.debug("No trace ID in queue message")
        return False
    
    # Restore trace context
    set_trace_context(
        trace_id=trace_id,
        observation_id=message.langfuse_observation_id if message.HasField('langfuse_observation_id') else None,
        session_id=message.langfuse_session_id if message.HasField('langfuse_session_id') else None,
        user_id=message.langfuse_user_id if message.HasField('langfuse_user_id') else None,
    )
    
    logger.debug(f"Restored trace context from queue message: trace_id={trace_id}")
    
    return True
