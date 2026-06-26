"""Langfuse spans for idempotency skips and no-ops.

When a memory operation short-circuits (L1 source dedup, L2 processing-complete,
L3 temporal guard), emit a dedicated span so the trace shows the skip reason
instead of looking like processing stopped mid-flight.
"""

from typing import Any, Dict, Optional, cast

from mirix.log import get_logger
from mirix.observability.context import get_trace_context, mark_observation_as_child
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)


def emit_idempotency_skip_span(
    name: str,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a no-op Langfuse span marking that processing was skipped.

    Attaches to the current trace context as a child so it appears under the
    meta-agent / memory-agent span that would otherwise look incomplete. When
    Langfuse is disabled or no trace context is active, this is a no-op.

    Args:
        name: Span name (e.g. "Idempotency Skip: source deduped").
        reason: Short tag describing why the skip happened
            (e.g. "source-deduped", "processing-complete", "temporal-guard").
        metadata: Extra fields merged into span metadata for trace inspection.
    """
    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id") if trace_context else None
    parent_span_id = trace_context.get("observation_id") if trace_context else None

    if not (langfuse and trace_id):
        return

    from langfuse.types import TraceContext

    span_metadata: Dict[str, Any] = {"skip_reason": reason}
    if metadata:
        span_metadata.update(metadata)

    # Stamp the TID into span metadata so the Langfuse OTel export emits
    # ``langfuse.observation.metadata.tid``. Consumers that filter spans by TID
    # (the full-stack-test span capture) would otherwise drop this span whenever
    # it is the first/only worker span — it survives today only because skips
    # usually nest under a tid-stamped parent (the "Meta Agent" observation).
    # Mirrors timed.py. Omitted when there's no active TID so we don't write a
    # misleading ``tid=None``.
    from mirix.observability.context import get_tid

    tid = get_tid()
    if tid:
        span_metadata.setdefault("tid", tid)

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    try:
        with langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
    except Exception as e:
        logger.warning("Failed to emit idempotency skip span %s: %s", name, e)


def emit_refused_to_process_span(
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a no-op Langfuse span marking that a save was refused before processing.

    Unlike an idempotency skip (the work was already done elsewhere), a refusal
    means the message is deterministically rejected — e.g. a read-only client
    (no write_scope) can never create memories, so the worker dead-letters it
    without persisting any source data. Emitting a dedicated span makes the
    refusal explicit in the trace instead of looking like processing stopped
    mid-flight.

    Attaches to the current trace context as a child. When Langfuse is disabled
    or no trace context is active, this is a no-op.

    Args:
        reason: Short tag describing why the message was refused
            (e.g. "no-write-scope").
        metadata: Extra fields merged into span metadata for trace inspection.
    """
    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id") if trace_context else None
    parent_span_id = trace_context.get("observation_id") if trace_context else None

    if not (langfuse and trace_id):
        return

    from langfuse.types import TraceContext

    span_metadata: Dict[str, Any] = {"refusal_reason": reason}
    if metadata:
        span_metadata.update(metadata)

    # Stamp the TID into span metadata so the Langfuse OTel export emits
    # ``langfuse.observation.metadata.tid``. This refusal span fires before any
    # tid-stamped parent (e.g. the Meta Agent observation) exists, so consumers
    # that filter spans by TID (the full-stack-test span capture) would drop it
    # entirely without this. Mirrors timed.py. Omitted when there's no active
    # TID so we don't write a misleading ``tid=None``.
    from mirix.observability.context import get_tid

    tid = get_tid()
    if tid:
        span_metadata.setdefault("tid", tid)

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    try:
        with langfuse.start_as_current_observation(
            name="Refused to Process: no write_scope",
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
    except Exception as e:
        logger.warning("Failed to emit refused-to-process span: %s", e)
