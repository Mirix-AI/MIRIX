"""Trace-level attribute helpers: the tag accumulator and the per-save write counts.

Langfuse replaces a trace's tag list wholesale on every
``update_current_trace(tags=...)`` write, and the memory-save trace is written
from multiple independent call sites (the HTTP-entry decorator, the worker's
Meta Agent open, the finalize-time outcome emitter) — so a naive per-site
write silently clobbers whatever the previous site wrote (last writer wins).

:func:`update_trace_attributes` kills that structurally instead of by
discipline: new tags merge into a per-context set and every write sends the
FULL sorted set, so whatever the flush order, the trace converges to the
union. The set lives in a ContextVar with the same lifecycle as the trace
context (``clear_trace_context()`` resets it via :func:`reset_trace_tags`).

This module also hosts the per-save write-count accumulator: a ContextVar
dict set by ``dispatch_save`` and bumped in ``_write_citation`` (the single
funnel every memory write passes through), so the worker root span can close
with ``writes_by_memory_type`` without any extra I/O on the save path.
"""

from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional, Set

from mirix.log import get_logger
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)

# The accumulated tag set for the current trace. ``None`` means "no tags
# accumulated yet" — distinct from an empty set only in that we lazily create
# the set on first use.
_trace_tags: ContextVar[Optional[Set[str]]] = ContextVar("trace_tags", default=None)


def update_trace_attributes(
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge ``tags`` into the per-context tag set and write the FULL set.

    Args:
        tags: NEW tags to add, e.g. ``["client:aqna-client"]``. Tag strings are
            used verbatim (``f"{prefix}:{value}"``; values may themselves
            contain ``:`` — tags are exact-match strings).
        metadata: Trace metadata for this write. Langfuse merges metadata by
            key server-side, so it is passed through per call (no client-side
            accumulation needed).

    No-op when Langfuse is disabled or no trace_id is active. Never raises —
    instrumentation must not break the save path.
    """
    try:
        langfuse = get_langfuse_client()
        if langfuse is None:
            return

        # Local import to avoid a module cycle (context.clear_trace_context
        # calls back into reset_trace_tags below).
        from mirix.observability.context import get_trace_context

        trace_context = get_trace_context()
        if not (trace_context and trace_context.get("trace_id")):
            return

        current = _trace_tags.get()
        if current is None:
            current = set()
            _trace_tags.set(current)
        if tags:
            current.update(tags)

        kwargs: Dict[str, Any] = {}
        if current:
            kwargs["tags"] = sorted(current)
        if metadata is not None:
            kwargs["metadata"] = metadata
        if not kwargs:
            return

        langfuse.update_current_trace(**kwargs)
    except Exception as e:  # noqa: BLE001 - instrumentation never raises
        logger.debug("update_trace_attributes failed (ignored): %s", e)


def reset_trace_tags() -> None:
    """Empty the per-context tag set.

    Called from ``clear_trace_context()`` so the tag lifecycle matches the
    trace-context lifecycle: one save's tags can never leak into the next
    message processed on the same (reused) worker task.
    """
    _trace_tags.set(None)


# --------------------------------------------------------------------------- #
# Per-save write-count accumulator (R3: Meta Agent root-span output).
#
# Same lifecycle pattern as the fault-injection active-source scope:
# dispatch_save sets a fresh dict per save and resets it in its finally;
# tasks spawned under asyncio.gather copy the context but share the parent's
# MUTABLE dict object, so concurrent sub-agents all bump the same counts.
# --------------------------------------------------------------------------- #

_save_write_counts: ContextVar[Optional[Dict[str, int]]] = ContextVar("save_write_counts", default=None)


def set_save_write_counts() -> Token:
    """Publish a fresh, empty write-count dict for the current save.

    Returns the ContextVar token so :func:`reset_save_write_counts` can
    restore the prior state in ``dispatch_save``'s ``finally``.
    """
    return _save_write_counts.set({})


def reset_save_write_counts(token: Optional[Token]) -> None:
    """Restore the write-count state that preceded :func:`set_save_write_counts`."""
    if token is not None:
        try:
            _save_write_counts.reset(token)
        except Exception:  # noqa: BLE001 - cross-context reset is a no-op
            _save_write_counts.set(None)


def bump_write_count(memory_type: str) -> None:
    """Count one memory write of ``memory_type`` for the active save.

    Silent no-op when no save is active (e.g. unit tests exercising a tool
    function directly) — instrumentation must never gate the write path.
    """
    counts = _save_write_counts.get()
    if counts is None:
        return
    counts[memory_type] = counts.get(memory_type, 0) + 1


def get_write_counts() -> Dict[str, int]:
    """Return a snapshot of the active save's write counts (``{}`` when unset)."""
    counts = _save_write_counts.get()
    return dict(counts) if counts else {}
