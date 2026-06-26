"""Context management for LangFuse tracing across async operations."""

from contextvars import ContextVar
from typing import Any, Optional

# Import for setting AS_ROOT attribute
try:
    from langfuse._client.attributes import LangfuseOtelSpanAttributes
except ImportError:
    LangfuseOtelSpanAttributes = None  # type: ignore

# Context variables for trace propagation
# These propagate through async/await boundaries automatically
current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)
current_observation_id: ContextVar[Optional[str]] = ContextVar("current_observation_id", default=None)
current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar("current_user_id", default=None)

# Request correlation: the transaction id (TID). Tracked separately from the
# trace context above because it is caller/gateway-provided request identity
# that must exist and propagate even when tracing is disabled. It is carried
# across the queue independently of any active trace (see trace_propagation.py)
# and re-established in the worker so all worker/agent log lines and spans can
# be correlated by it.
current_tid: ContextVar[Optional[str]] = ContextVar("current_tid", default=None)


def set_tid(tid: Optional[str]) -> None:
    """Set the current request's transaction id (TID)."""
    if tid:
        current_tid.set(tid)


def get_tid() -> Optional[str]:
    """Get the current request's transaction id (TID), if any."""
    return current_tid.get()


def set_trace_context(
    trace_id: Optional[str] = None,
    observation_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """
    Set current trace context.

    Used to propagate trace information through async operations
    and across service boundaries (e.g., Kafka messages).
    """
    if trace_id:
        current_trace_id.set(trace_id)
    if observation_id:
        current_observation_id.set(observation_id)
    if session_id:
        current_session_id.set(session_id)
    if user_id:
        current_user_id.set(user_id)


def get_trace_context() -> dict:
    """
    Get current trace context.

    Returns:
        Dictionary with current trace IDs and metadata
    """
    return {
        "trace_id": current_trace_id.get(),
        "observation_id": current_observation_id.get(),
        "session_id": current_session_id.get(),
        "user_id": current_user_id.get(),
    }


def clear_trace_context() -> None:
    """
    Clear trace context.

    Should be called at the end of request/task processing to avoid
    context leaking between unrelated operations.
    """
    current_trace_id.set(None)
    current_observation_id.set(None)
    current_session_id.set(None)
    current_user_id.set(None)


def clear_tid() -> None:
    """Clear the current request's transaction id (TID)."""
    current_tid.set(None)


def mark_observation_as_child(observation: Any) -> None:
    """
    Mark a Langfuse observation as a child (not root).

    The Langfuse SDK sets AS_ROOT=True when trace_context is provided,
    but we want proper nesting for child observations. This function overrides
    that behavior.

    Args:
        observation: A Langfuse observation object from start_as_current_observation
    """
    if LangfuseOtelSpanAttributes is not None and hasattr(observation, "_otel_span"):
        observation._otel_span.set_attribute(LangfuseOtelSpanAttributes.AS_ROOT, False)
