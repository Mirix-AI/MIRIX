"""Shared OpenTelemetry TracerProvider and an optional on-disk span sink.

Langfuse v3 is built on the standard OTel SDK: it *adds* its own span processor
to whatever ``TracerProvider`` we hand it rather than owning one exclusively. We
exploit that here to keep a single provider that can fan out to more than one
sink — Langfuse (remote) plus, in tests, a local JSONL file.

The file sink is opt-in via ``settings.span_export_file`` and uses a synchronous
``SimpleSpanProcessor`` so spans are written the moment they end (deterministic,
no batching) — exactly what full-stack tests need to read spans back off disk
and assert on them, with no remote Langfuse required.

Both the provider and the file processor are process singletons.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

logger = get_logger(__name__)

_tracer_provider: Optional["TracerProvider"] = None
_file_processor_attached: bool = False


def get_shared_tracer_provider() -> "TracerProvider":
    """Return the process-wide ``TracerProvider``, creating it once.

    Also attaches the on-disk span exporter the first time, if configured. The
    same provider instance is what we pass to the Langfuse client so Langfuse's
    processor and our file processor both receive every span.
    """
    global _tracer_provider

    from opentelemetry.sdk.trace import TracerProvider

    if _tracer_provider is None:
        _tracer_provider = TracerProvider()
        _maybe_attach_file_exporter(_tracer_provider)

    return _tracer_provider


def _span_to_json_line(span: "ReadableSpan") -> str:
    """Render a finished span as a compact, test-friendly JSON object.

    We emit a small, stable shape (not full OTLP) tuned for assertions: name,
    ids, timing, status, and attributes (which include the TID and any span
    metadata). One JSON object per line (JSONL).
    """
    ctx = span.get_span_context()
    parent = span.parent
    start_ns = span.start_time or 0
    end_ns = span.end_time or 0
    record = {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x") if ctx else None,
        "span_id": format(ctx.span_id, "016x") if ctx else None,
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "duration_ms": (end_ns - start_ns) / 1_000_000.0 if (start_ns and end_ns) else None,
        "status": span.status.status_code.name if span.status else None,
        "attributes": dict(span.attributes or {}),
    }
    return json.dumps(record, default=str)


def _maybe_attach_file_exporter(provider: "TracerProvider") -> None:
    """Attach the JSONL file span exporter to ``provider`` if configured.

    No-op (and never raises) when ``settings.span_export_file`` is unset or the
    path can't be opened — span export must never break request processing.
    """
    global _file_processor_attached

    if _file_processor_attached:
        return

    try:
        from pathlib import Path

        from mirix.settings import settings

        path = settings.span_export_file
        # Only act on a concrete str/Path. Deliberately NOT a broad os.PathLike
        # check: MagicMock auto-creates __fspath__ and so passes isinstance(...,
        # os.PathLike), which would make patched-settings unit tests open a file
        # literally named "<MagicMock ...>". str/Path excludes that.
        if not path or not isinstance(path, (str, Path)):
            return

        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

        class _JsonlFileSpanExporter(SpanExporter):
            """Append finished spans to a JSONL file, one span per line."""

            def __init__(self, file_path: str) -> None:
                # Line-buffered append so concurrently-finished spans (cooperative
                # single-loop tests) each land as one atomic line.
                self._fh = open(file_path, "a", buffering=1)

            def export(self, spans) -> "SpanExportResult":
                try:
                    for span in spans:
                        self._fh.write(_span_to_json_line(span) + "\n")
                    return SpanExportResult.SUCCESS
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("span file export failed: %s", e)
                    return SpanExportResult.FAILURE

            def shutdown(self) -> None:
                try:
                    self._fh.close()
                except Exception:
                    pass

        exporter = _JsonlFileSpanExporter(str(path))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        _file_processor_attached = True
        logger.info("Span file export enabled -> %s", path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to attach span file exporter: %s", e)


def _reset_for_testing() -> None:
    """Reset the provider/file-processor singletons (unit tests only)."""
    global _tracer_provider, _file_processor_attached
    _tracer_provider = None
    _file_processor_attached = False
