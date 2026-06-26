"""``timed`` / ``timedspan``: a dual decorator/context-manager timing construct.

Wrapping a block (or coroutine) in ``timed(name)`` measures its wall-clock
duration, logs a timing line (DEBUG by default, WARNING when a ``slow_ms``
threshold is crossed), and feeds the elapsed milliseconds through
``record_timing`` so timing data is available to whatever consumer the ``line``
callback wires up.

``timedspan(name)`` is the same construct but additionally opens a child
Langfuse span around the block for duration attribution in the trace tree. The
span machinery is a verbatim copy of ``mirix.observability.timed_spans.timed_span``
(same guard/error handling and TID stamping); it is a clean no-op when Langfuse
is disabled or no trace context is active, so ``timedspan`` is always safe to use
(including in unit tests with no tracing configured).

Both forms work as either an ``async with`` context manager or an ``async``
function decorator.
"""

import functools
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Dict, Optional, cast

from mirix.log import get_logger
from mirix.observability.context import (
    current_observation_id,
    get_tid,
    get_trace_context,
    mark_observation_as_child,
)
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)

_active_record: ContextVar[Optional[Dict[str, Any]]] = ContextVar("timed_active_record", default=None)
LineBuilder = Callable[[float, Dict[str, Any]], str]


def record_timing(**fields: Any) -> None:
    """From inside a decorated body, contribute after-the-block fields to the timing line."""
    rec = _active_record.get()
    if rec is not None:
        rec.update(fields)


@asynccontextmanager
async def _async_noop() -> AsyncIterator[None]:
    """An async no-op used as the inert context body when tracing is off."""
    yield


@asynccontextmanager
async def _open_span(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[None]:
    """Wrap an ``await`` block in a child Langfuse span for duration attribution.

    Args:
        name: Span name shown in the trace (e.g. "Persist Memory Source").
        metadata: Extra fields merged into span metadata for inspection.

    No-op (still runs the wrapped block) when Langfuse is disabled or no trace
    context is active. Tracing failures never propagate to the wrapped work.
    """
    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id") if trace_context else None
    parent_span_id = trace_context.get("observation_id") if trace_context else None

    if not (langfuse and trace_id):
        # Tracing unavailable: run the block untouched.
        yield
        return

    from langfuse.types import TraceContext

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    # Stamp the TID into span metadata so the Langfuse OTel export emits
    # ``langfuse.observation.metadata.tid``. Consumers that filter spans by TID
    # (the full-stack-test span capture) would otherwise drop every nested
    # worker span — only the root spans that already stamp the tid (HTTP-entry
    # trace, worker "Meta Agent" observation) would survive. Mirrors the worker's
    # Meta Agent span metadata. Omitted when there's no active TID so we don't
    # write a misleading ``tid=None``.
    span_metadata: Dict[str, Any] = dict(metadata or {})
    tid = get_tid()
    if tid:
        span_metadata.setdefault("tid", tid)

    try:
        cm = langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata=span_metadata,
        )
    except Exception as e:
        # If span creation itself fails, don't lose the work.
        logger.warning("timedspan(%s) failed to start: %s", name, e)
        yield
        return

    with cm as span:
        try:
            mark_observation_as_child(span)
        except Exception as e:
            logger.warning("timedspan(%s) failed to mark child: %s", name, e)

        # Publish this span as the current observation while the wrapped block
        # runs so any span opened inside it (including a nested timed_span)
        # nests under THIS span rather than under its parent. Restore the prior
        # observation id afterward so the next sibling span parents back to the
        # original parent.
        span_observation_id = getattr(span, "id", None)
        prior_observation_id = parent_span_id
        if span_observation_id:
            # set_trace_context ignores a falsy observation_id, so set the
            # ContextVar directly (and restore directly below) to also handle
            # the None / no-parent case correctly.
            current_observation_id.set(span_observation_id)
        try:
            yield
        finally:
            if span_observation_id:
                current_observation_id.set(prior_observation_id)


class _TimedOp:
    """Dual decorator / async context-manager engine for a single timed op.

    As a context manager: ``async with timed(name) as rec: ...`` times the
    block and yields a mutable ``rec`` dict the body can populate. As a
    decorator: ``@timed(name)`` wraps an ``async def`` so each call is timed,
    with the body reaching ``rec`` via :func:`record_timing`. ``open_span=True``
    (``timedspan``) additionally opens a child Langfuse span around the region.
    """

    def __init__(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        open_span: bool = False,
        log: Optional[Any] = None,
        slow_ms: Optional[float] = None,
        line: Optional[LineBuilder] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._metadata = metadata
        self._open_span = open_span
        self._log = log or logger
        self._slow_ms = slow_ms
        self._line = line
        self._extra = extra
        self._cm: Optional[Any] = None

    @asynccontextmanager
    async def _run(self) -> AsyncIterator[Dict[str, Any]]:
        rec: Dict[str, Any] = self._extra if self._extra is not None else {}
        span_cm = _open_span(self._name, self._metadata) if self._open_span else _async_noop()
        start = time.monotonic()
        async with span_cm:
            try:
                yield rec
            finally:
                # Emitting the timing line must NEVER raise out of this finally:
                # when the body is unwinding an exception, a faulty ``line``
                # callback (e.g. one that subscripts a ``rec`` key the body only
                # sets on success) would otherwise REPLACE the real exception and
                # mask the true failure all the way up the call stack.
                # Instrumentation is best-effort; the wrapped work's outcome wins.
                try:
                    ms = (time.monotonic() - start) * 1000.0
                    slow = self._slow_ms is not None and ms >= self._slow_ms
                    msg = (
                        self._line(ms, rec) if self._line is not None else f"[{self._name} TIMING] execute_ms={ms:.1f}"
                    )
                    emit = self._log.warning if slow else self._log.debug
                    emit("%s%s", msg, " SLOW" if slow else "")
                except Exception as e:  # noqa: BLE001 - timing must not break the call
                    logger.warning("timed(%s) failed to emit timing line: %s", self._name, e)

    async def __aenter__(self) -> Dict[str, Any]:
        self._cm = self._run()
        return await self._cm.__aenter__()

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._cm.__aexit__(*exc)

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self._run() as rec:
                token = _active_record.set(rec)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _active_record.reset(token)

        return wrapper


def timed(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    logger: Optional[Any] = None,
    slow_ms: Optional[float] = None,
    line: Optional[LineBuilder] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> _TimedOp:
    """Time an async block or coroutine, logging its duration.

    Usable as ``async with timed(name) as rec: ...`` or as ``@timed(name)`` on
    an ``async def``. Logs the default ``[<name> TIMING] execute_ms=<ms>`` line
    at DEBUG, or at WARNING (with a ``SLOW`` suffix) when ``slow_ms`` is crossed.
    """
    return _TimedOp(name, metadata, open_span=False, log=logger, slow_ms=slow_ms, line=line, extra=extra)


def timedspan(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    logger: Optional[Any] = None,
    slow_ms: Optional[float] = None,
    line: Optional[LineBuilder] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> _TimedOp:
    """Like :func:`timed`, but also opens a child Langfuse span around the block.

    The span is a no-op when Langfuse is disabled or no trace context is active,
    so this is always safe to use. Usable as a context manager or decorator.
    """
    return _TimedOp(name, metadata, open_span=True, log=logger, slow_ms=slow_ms, line=line, extra=extra)
