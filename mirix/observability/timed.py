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
    get_trace_context,
    mark_observation_as_child,
    stamp_tid,
)
from mirix.observability.langfuse_client import get_langfuse_client

logger = get_logger(__name__)

_active_record: ContextVar[Optional[Dict[str, Any]]] = ContextVar("timed_active_record", default=None)
LineBuilder = Callable[[float, Dict[str, Any]], str]

# Sentinel for ``timedspan(input=...)``: the default mirrors the caller's
# ``metadata`` dict as the span input (the metadata at these call sites already
# IS "what the step operated on"); ``input=None`` suppresses the mirror; any
# other value is used verbatim.
_MIRROR_METADATA = object()

# Reserved ``rec`` key: the body sets ``rec["span_output"] = {...}`` (or calls
# :func:`record_output` in decorator form) and the exit path delivers it as
# ``span.update(output=...)``. Popped before the timing line renders so ``line``
# callbacks never see it.
_SPAN_OUTPUT_KEY = "span_output"


def record_timing(**fields: Any) -> None:
    """From inside a decorated body, contribute after-the-block fields to the timing line."""
    rec = _active_record.get()
    if rec is not None:
        rec.update(fields)


def record_output(**fields: Any) -> None:
    """From inside a decorated body, contribute fields to the span's output.

    Decorator-form counterpart of ``rec["span_output"] = {...}`` (mirrors
    :func:`record_timing`). Successive calls merge. Silent no-op outside an
    active ``timed``/``timedspan`` body.
    """
    rec = _active_record.get()
    if rec is None:
        return
    out = rec.setdefault(_SPAN_OUTPUT_KEY, {})
    if isinstance(out, dict):
        out.update(fields)


@asynccontextmanager
async def _async_noop() -> AsyncIterator[None]:
    """An async no-op used as the inert context body when tracing is off."""
    yield


@asynccontextmanager
async def _open_span(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    input: Any = _MIRROR_METADATA,
) -> AsyncIterator[Optional[Any]]:
    """Wrap an ``await`` block in a child Langfuse span for duration attribution.

    Args:
        name: Span name shown in the trace (e.g. "Persist Memory Source").
        metadata: Extra fields merged into span metadata for inspection.
        input: Span input. Defaults to a mirror of ``metadata`` (the caller's
            dict, WITHOUT the stamped tid); ``None`` suppresses; any other
            value is used verbatim.

    Yields the span handle (or ``None`` on every no-op path) so the caller can
    attach output on exit. No-op (still runs the wrapped block) when Langfuse
    is disabled or no trace context is active. Tracing failures never propagate
    to the wrapped work.
    """
    langfuse = get_langfuse_client()
    trace_context = get_trace_context()
    trace_id = trace_context.get("trace_id") if trace_context else None
    parent_span_id = trace_context.get("observation_id") if trace_context else None

    if not (langfuse and trace_id):
        # Tracing unavailable: run the block untouched.
        yield None
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
    span_metadata: Dict[str, Any] = stamp_tid(dict(metadata or {}))

    # Resolve the span input: mirror the CALLER's metadata by default (not the
    # tid-stamped span_metadata — the tid is a capture concern, not an input).
    if input is _MIRROR_METADATA:
        span_input = dict(metadata) if metadata else None
    else:
        span_input = input

    span_kwargs: Dict[str, Any] = {
        "name": name,
        "as_type": "span",
        "trace_context": cast(TraceContext, trace_context_dict),
        "metadata": span_metadata,
    }
    if span_input is not None:
        span_kwargs["input"] = span_input

    try:
        cm = langfuse.start_as_current_observation(**span_kwargs)
    except Exception as e:
        # If span creation itself fails, don't lose the work.
        logger.warning("timedspan(%s) failed to start: %s", name, e)
        yield None
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
            yield span
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
        span_input: Any = _MIRROR_METADATA,
        log: Optional[Any] = None,
        slow_ms: Optional[float] = None,
        line: Optional[LineBuilder] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._metadata = metadata
        self._open_span = open_span
        self._span_input = span_input
        self._log = log or logger
        self._slow_ms = slow_ms
        self._line = line
        self._extra = extra
        self._cm: Optional[Any] = None

    @asynccontextmanager
    async def _run(self) -> AsyncIterator[Dict[str, Any]]:
        rec: Dict[str, Any] = self._extra if self._extra is not None else {}
        span_cm = _open_span(self._name, self._metadata, self._span_input) if self._open_span else _async_noop()
        start = time.monotonic()
        async with span_cm as span:
            caught: Optional[BaseException] = None
            try:
                yield rec
            except BaseException as e:
                caught = e
                raise
            finally:
                # Nothing in this finally may EVER raise: when the body is
                # unwinding an exception, a faulty ``line`` callback (e.g. one
                # that subscripts a ``rec`` key the body only sets on success)
                # or a failing ``span.update`` would otherwise REPLACE the real
                # exception and mask the true failure all the way up the call
                # stack. Instrumentation is best-effort; the wrapped work's
                # outcome wins.

                # Consume the reserved output key regardless of span state so
                # the timing line / ``line`` callbacks never see it.
                try:
                    span_output = rec.pop(_SPAN_OUTPUT_KEY, None)
                except Exception:  # noqa: BLE001 - rec is caller-supplied
                    span_output = None

                if span is not None:
                    try:
                        if span_output is not None:
                            span.update(output=span_output)
                        if caught is not None:
                            # PII posture: exception TYPE NAME only — str(e)
                            # can echo user content (see agent.py precedent).
                            span.update(level="ERROR", status_message=type(caught).__name__)
                    except Exception as e:  # noqa: BLE001 - never break the call
                        logger.warning("timed(%s) failed to update span output: %s", self._name, e)

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
    input: Any = _MIRROR_METADATA,
    logger: Optional[Any] = None,
    slow_ms: Optional[float] = None,
    line: Optional[LineBuilder] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> _TimedOp:
    """Like :func:`timed`, but also opens a child Langfuse span around the block.

    The span is a no-op when Langfuse is disabled or no trace context is active,
    so this is always safe to use. Usable as a context manager or decorator.

    Span input/output:

    - ``input`` defaults to a mirror of ``metadata`` (what the step operated
      on); pass ``input=None`` to suppress, or an explicit value to override.
    - The body sets output via ``rec["span_output"] = {...}`` (context-manager
      form) or :func:`record_output` (decorator form); it is delivered as
      ``span.update(output=...)`` on exit. On exception the span is marked
      ``level=ERROR`` with the exception type name only (never ``str(e)``).
    """
    return _TimedOp(
        name, metadata, open_span=True, span_input=input, log=logger, slow_ms=slow_ms, line=line, extra=extra
    )
