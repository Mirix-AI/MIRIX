"""Langfuse spans for idempotency skips and no-ops.

When a memory operation short-circuits (L1 source dedup, L2 processing-complete,
L3 temporal guard), emit a dedicated span so the trace shows the skip reason
instead of looking like processing stopped mid-flight.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from mirix.log import get_logger
from mirix.observability.context import (
    get_trace_context,
    mark_observation_as_child,
    stamp_tid,
)
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.observability.trace_attrs import update_trace_attributes

if TYPE_CHECKING:
    from mirix.queue.error_policy import SaveOutcome

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

    # Stamp the TID so the TID-filtered FST span capture keeps this span even
    # when it is the first/only worker span (it survives today only because
    # skips usually nest under the tid-stamped "Meta Agent" observation).
    span_metadata = stamp_tid(span_metadata)

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    try:
        with langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            # The skip is the whole step: its metadata is what it operated on
            # (input) and the skip reason is what it decided (output). Without
            # these the span renders Input/Output: undefined in LangFuse.
            input=span_metadata,
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
            span.update(output={"skipped": reason})
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

    # Stamp the TID: this refusal span fires before any tid-stamped parent
    # (e.g. the Meta Agent observation) exists, so the TID-filtered FST span
    # capture would drop it entirely without this.
    span_metadata = stamp_tid(span_metadata)

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    try:
        with langfuse.start_as_current_observation(
            # The name carries the ACTUAL reason. It was previously hardcoded
            # to "no write_scope" for every refusal, so e.g. a missing-client-id
            # refusal rendered with a misleading name.
            name=f"Refused to Process: {reason}",
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            # The refusal is the whole step: metadata is what it operated on
            # (input) and the refusal reason is the result (output).
            input=span_metadata,
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
            span.update(output={"refused": reason})
    except Exception as e:
        logger.warning("Failed to emit refused-to-process span: %s", e)


def emit_save_outcome_span(
    outcome: "SaveOutcome",
    memory_source_id: Optional[str],
    error_type: Optional[str] = None,
) -> None:
    """Mark the worker trace complete with the save's terminal outcome (R4).

    Called from ``dispatch_save`` immediately after the ``finalize_source``
    call — inside the ``try``, before the ``finally`` clears trace context, so
    the marker still attaches to the right trace. Opens an ``Outcome`` span
    (the skip-span pattern: explicit trace_context from the ContextVars, TID
    stamped in metadata for the FST capture, parented to the Meta Agent
    observation — a closed parent is fine for late-arriving children) and,
    while the span is current, writes the ``save_outcome:`` tag through the
    central trace-attribute helper.

    Absence of the marker is meaningful (in-flight or abandoned save — R4 AC3),
    so this is a clean no-op without an active trace, and it NEVER raises into
    the save path.

    Args:
        outcome: The terminal ``SaveOutcome`` — its ``.value`` is the tag
            vocabulary, verbatim (``success`` / ``permanent_failure`` /
            ``transient_exhausted``).
        memory_source_id: The save's source id, or ``None`` (the marker is NOT
            gated on it — a refusal with no source row still gets its tag).
        error_type: Exception TYPE NAME only (never ``str(e)`` — PII posture),
            present only for failure outcomes with a known cause.
    """
    try:
        langfuse = get_langfuse_client()
        trace_context = get_trace_context()
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None

        if not (langfuse and trace_id):
            return

        from langfuse.types import TraceContext

        outcome_value = outcome.value

        span_metadata: Dict[str, Any] = {
            "save_outcome": outcome_value,
            "memory_source_id": memory_source_id,
        }

        # Stamp the TID — the TID-filtered FST span capture would otherwise
        # drop the marker span.
        span_metadata = stamp_tid(span_metadata)

        trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
        if parent_span_id:
            trace_context_dict["parent_span_id"] = parent_span_id

        output: Dict[str, Any] = {"outcome": outcome_value}
        if error_type:
            output["error_type"] = error_type

        # Tags for the trace-level write. The tid tag rides along explicitly:
        # on refusal paths the save never reaches the Meta Agent block that
        # normally seeds the accumulator with tid/client, so without it this
        # final full-set write would drop the trace's tid: tag (clobbering the
        # HTTP leg's write on a stitched trace).
        tid = span_metadata.get("tid")
        trace_tags = [f"save_outcome:{outcome_value}"]
        trace_metadata: Dict[str, Any] = {"save_outcome": outcome_value}
        if tid:
            trace_tags.append(f"tid:{tid}")
            trace_metadata["tid"] = tid

        with langfuse.start_as_current_observation(
            name="Outcome",
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            input={"memory_source_id": memory_source_id},
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
            # While the span is current, surface the outcome at the TRACE level
            # (tag = dashboard-filterable; metadata = visible). The helper
            # rewrites the full accumulated tag set, so this write is a strict
            # superset of the worker's earlier tags — no clobbering.
            update_trace_attributes(tags=trace_tags, metadata=trace_metadata)
            span.update(output=output)
    except Exception as e:  # noqa: BLE001 - instrumentation never raises
        logger.warning("Failed to emit save-outcome span: %s", e)
