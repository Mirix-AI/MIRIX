"""Context management for LangFuse tracing across async operations, and the
conversation session an outbound model call serves."""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

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


# --------------------------------------------------------------------------- #
# The conversation session an outbound LLM / embedding request serves          #
# --------------------------------------------------------------------------- #

#: The conversation session the current work serves (a retrieval's session, a
#: distilled session, a curator batch of one session's experiences): set with
#: :func:`conversation_session_scope`, per asyncio task, and sent on outbound model
#: requests by :func:`outbound_session_headers`.
current_conversation_session: ContextVar[Optional[str]] = ContextVar("current_conversation_session", default=None)
#: The longest session id sent (MIRIX's session_id rule, ``mirix.schemas.message``).
SESSION_VALUE_MAX_LEN = 64
_SESSION_VALUE_RE = re.compile(r"[A-Za-z0-9_-]{1,%d}" % SESSION_VALUE_MAX_LEN)


@contextmanager
def conversation_session_scope(session_id: Optional[str]) -> Iterator[None]:
    """Serve ``session_id`` inside the block: every outbound model request made in it
    (in this task and the tasks it starts) may carry it (:func:`outbound_session_headers`).
    ``None`` clears an outer session, for work that serves several sessions at once."""
    token = current_conversation_session.set(session_id)
    try:
        yield
    finally:
        current_conversation_session.reset(token)


def outbound_session_headers() -> Dict[str, str]:
    """``{<MIRIX_LLM_SESSION_HEADER>: <session id>}`` for an outbound LLM or embedding
    request, or ``{}``: the setting is off (the default), no session is served, or the
    session is no valid MIRIX session id (``[A-Za-z0-9_-]{1,64}``: a malformed value is
    never sent). Sent by the OpenAI chat client and the OpenRouter, OpenAI-compatible
    endpoint, OpenAI-SDK (with an auth provider), Azure and Ollama embedding clients;
    the other providers' clients do not send it yet."""
    from mirix.settings import settings

    name = getattr(settings, "llm_session_header", None)
    if not name:
        return {}
    value = current_conversation_session.get()
    if not isinstance(value, str) or not _SESSION_VALUE_RE.fullmatch(value):
        return {}
    return {name: value}
