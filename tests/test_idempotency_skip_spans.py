"""Unit tests for idempotency skip spans.

Verifies that when L1 source-dedup, L2 processing-complete, or L3 temporal-guard
short-circuits a memory operation, a Langfuse span is emitted under the current
trace context so the trace makes the skip reason visible.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability.skip_spans import (
    emit_idempotency_skip_span,
    emit_refused_to_process_span,
)


def _make_client():
    """Langfuse client whose observation cm yields an inspectable span."""
    client = MagicMock()
    span = MagicMock()
    span_ctx = MagicMock()
    span_ctx.__enter__ = MagicMock(return_value=span)
    span_ctx.__exit__ = MagicMock(return_value=False)
    client.start_as_current_observation.return_value = span_ctx
    return client, span


class TestEmitIdempotencySkipSpan:
    def test_no_op_when_langfuse_disabled(self):
        """Skip span should be a silent no-op when Langfuse isn't initialized."""
        with patch("mirix.observability.skip_spans.get_langfuse_client", return_value=None):
            # Should not raise.
            emit_idempotency_skip_span(name="x", reason="source-deduped")

    def test_no_op_when_no_trace_context(self):
        """Skip span should be a silent no-op when no active trace_id."""
        client = MagicMock()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value={}),
        ):
            emit_idempotency_skip_span(name="x", reason="processing-complete")
        client.start_as_current_observation.assert_not_called()

    def test_emits_span_with_reason_and_metadata(self):
        """Happy path: emits a span under the current parent with reason/metadata."""
        client = MagicMock()
        span_ctx = MagicMock()
        span_ctx.__enter__ = MagicMock(return_value=MagicMock())
        span_ctx.__exit__ = MagicMock(return_value=False)
        client.start_as_current_observation.return_value = span_ctx

        trace_ctx = {"trace_id": "trace-123", "observation_id": "obs-abc"}
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value=trace_ctx),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_idempotency_skip_span(
                name="Idempotency Skip: temporal guard (episodic)",
                reason="temporal-guard",
                metadata={"memory_type": "episodic", "memory_id": "m-1"},
            )

        assert client.start_as_current_observation.called
        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["name"] == "Idempotency Skip: temporal guard (episodic)"
        assert kwargs["as_type"] == "span"
        assert kwargs["trace_context"]["trace_id"] == "trace-123"
        assert kwargs["trace_context"]["parent_span_id"] == "obs-abc"
        md = kwargs["metadata"]
        assert md["skip_reason"] == "temporal-guard"
        assert md["memory_type"] == "episodic"
        assert md["memory_id"] == "m-1"

    def test_swallows_exceptions(self):
        """Observability failures must not break the calling code path."""
        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("boom")
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t1", "observation_id": "o1"},
            ),
        ):
            # Should not raise.
            emit_idempotency_skip_span(name="x", reason="source-deduped")

    def test_input_mirrors_span_metadata_and_output_carries_skip_reason(self):
        """R3 catalog rows for skip/failure-marker spans: input = the span's
        metadata mirror; output = ``{"skipped": <reason>}`` — the skip IS the
        result of the step."""
        client, span = _make_client()
        trace_ctx = {"trace_id": "t-1", "observation_id": "o-1"}
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value=trace_ctx),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_idempotency_skip_span(
                name="Idempotency Skip: temporal guard (episodic)",
                reason="temporal-guard",
                metadata={"memory_type": "episodic", "memory_id": "m-1"},
            )

        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["input"] == kwargs["metadata"]
        assert kwargs["input"]["skip_reason"] == "temporal-guard"
        assert kwargs["input"]["memory_type"] == "episodic"
        span.update.assert_any_call(output={"skipped": "temporal-guard"})

    def test_output_update_failure_is_swallowed(self):
        client, span = _make_client()
        span.update.side_effect = RuntimeError("update boom")
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t-1", "observation_id": "o-1"},
            ),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_idempotency_skip_span(name="x", reason="source-deduped")  # must not raise


class TestEmitRefusedToProcessSpan:
    """The refusal emitter previously hardcoded the span name
    ``"Refused to Process: no write_scope"`` for EVERY refusal reason, so a
    missing-client-id refusal rendered with a misleading name. The name must
    carry the actual reason (R4 AC2 FST assertions depend on it)."""

    @pytest.mark.parametrize(
        "reason",
        ["no-write-scope", "missing-client-id", "client-not-found", "malformed-message"],
    )
    def test_span_name_carries_the_actual_reason(self, reason):
        client, _span = _make_client()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t-1", "observation_id": "o-1"},
            ),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_refused_to_process_span(reason=reason, metadata={"agent_id": "a-1"})

        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["name"] == f"Refused to Process: {reason}"

    def test_input_mirrors_span_metadata_and_output_carries_refusal(self):
        client, span = _make_client()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t-1", "observation_id": "o-1"},
            ),
            patch("mirix.observability.skip_spans.mark_observation_as_child"),
        ):
            emit_refused_to_process_span(
                reason="no-write-scope",
                metadata={"client_id": "c-1", "memory_source_id": "src-1"},
            )

        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["input"] == kwargs["metadata"]
        assert kwargs["input"]["refusal_reason"] == "no-write-scope"
        assert kwargs["input"]["client_id"] == "c-1"
        span.update.assert_any_call(output={"refused": "no-write-scope"})

    def test_no_op_when_no_trace_context(self):
        client, _span = _make_client()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value={}),
        ):
            emit_refused_to_process_span(reason="no-write-scope")
        client.start_as_current_observation.assert_not_called()

    def test_swallows_exceptions(self):
        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("boom")
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t-1", "observation_id": "o-1"},
            ),
        ):
            emit_refused_to_process_span(reason="client-not-found")  # must not raise
