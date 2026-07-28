"""Unit tests for ``emit_save_outcome_span`` (the R4 completion marker).

``finalize_source`` is the only point that knows a save's terminal outcome and
today it tells only the logs. The emitter opens an ``Outcome`` span at the
``dispatch_save`` chokepoint — while the TID and trace context are still set —
and writes the ``save_outcome:`` tag through the central trace-attribute
helper, so ECMS-499 can filter finished traces instead of guessing with a
fetch-lag heuristic. Absence of the marker is meaningful (in-flight or
abandoned), so it must be a clean no-op without an active trace and must
never raise into the save path.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability.skip_spans import emit_save_outcome_span
from mirix.queue.error_policy import SaveOutcome


def _make_client():
    client = MagicMock()
    span = MagicMock()
    span_ctx = MagicMock()
    span_ctx.__enter__ = MagicMock(return_value=span)
    span_ctx.__exit__ = MagicMock(return_value=False)
    client.start_as_current_observation.return_value = span_ctx
    return client, span


def _patches(client, trace_ctx=None, tid="tid-1"):
    if trace_ctx is None:
        trace_ctx = {"trace_id": "t-1", "observation_id": "meta-agent-obs"}
    return (
        patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
        patch("mirix.observability.skip_spans.get_trace_context", return_value=trace_ctx),
        patch("mirix.observability.skip_spans.mark_observation_as_child"),
        patch("mirix.observability.context.get_tid", return_value=tid),
    )


class TestEmitSaveOutcomeSpan:
    @pytest.mark.parametrize(
        "outcome",
        [SaveOutcome.SUCCESS, SaveOutcome.PERMANENT_FAILURE, SaveOutcome.TRANSIENT_EXHAUSTED],
    )
    def test_tag_and_metadata_use_verbatim_saveoutcome_vocabulary(self, outcome):
        """One vocabulary shared by the policy verdict, the DB log line, and the
        trace tag: SaveOutcome.value verbatim. The tid tag rides along so the
        full-set write never drops the trace's tid on refusal paths (where the
        worker accumulator was never seeded)."""
        client, _span = _make_client()
        p1, p2, p3, p4 = _patches(client, tid="tid-1")
        with (
            p1,
            p2,
            p3,
            p4,
            patch("mirix.observability.skip_spans.update_trace_attributes") as upd,
        ):
            emit_save_outcome_span(outcome, memory_source_id="src-1")

        upd.assert_called_once_with(
            tags=[f"save_outcome:{outcome.value}", "tid:tid-1"],
            metadata={"save_outcome": outcome.value, "tid": "tid-1"},
        )

    def test_no_tid_tag_when_tid_absent(self):
        client, _span = _make_client()
        p1, p2, p3, p4 = _patches(client, tid=None)
        with (
            p1,
            p2,
            p3,
            p4,
            patch("mirix.observability.skip_spans.update_trace_attributes") as upd,
        ):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")

        upd.assert_called_once_with(
            tags=["save_outcome:success"],
            metadata={"save_outcome": "success"},
        )

    def test_span_shape_name_parent_input_output(self):
        client, span = _make_client()
        p1, p2, p3, p4 = _patches(client)
        with p1, p2, p3, p4, patch("mirix.observability.skip_spans.update_trace_attributes"):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")

        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["name"] == "Outcome"
        assert kwargs["as_type"] == "span"
        assert kwargs["trace_context"]["trace_id"] == "t-1"
        # Parents to the Meta Agent observation (closed parent is fine for
        # late-arriving children).
        assert kwargs["trace_context"]["parent_span_id"] == "meta-agent-obs"
        assert kwargs["input"] == {"memory_source_id": "src-1"}
        span.update.assert_any_call(output={"outcome": "success"})

    def test_tid_stamped_in_metadata(self):
        """The FST span capture filters by the tid metadata attribute — without
        it the marker span would be dropped from every capture."""
        client, _span = _make_client()
        p1, p2, p3, p4 = _patches(client, tid="tid-fst")
        with p1, p2, p3, p4, patch("mirix.observability.skip_spans.update_trace_attributes"):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")

        md = client.start_as_current_observation.call_args.kwargs["metadata"]
        assert md.get("tid") == "tid-fst"

    def test_fires_with_no_memory_source_id(self):
        """NOT gated on memory_source_id: a refusal with no source row still
        gets its terminal tag (the DB finalize is gated; the marker is not)."""
        client, span = _make_client()
        p1, p2, p3, p4 = _patches(client)
        with p1, p2, p3, p4, patch("mirix.observability.skip_spans.update_trace_attributes") as upd:
            emit_save_outcome_span(SaveOutcome.PERMANENT_FAILURE, memory_source_id=None)

        upd.assert_called_once()
        kwargs = client.start_as_current_observation.call_args.kwargs
        assert kwargs["input"] == {"memory_source_id": None}
        span.update.assert_any_call(output={"outcome": "permanent_failure"})

    def test_error_type_present_only_when_given(self):
        client, span = _make_client()
        p1, p2, p3, p4 = _patches(client)
        with p1, p2, p3, p4, patch("mirix.observability.skip_spans.update_trace_attributes"):
            emit_save_outcome_span(
                SaveOutcome.TRANSIENT_EXHAUSTED,
                memory_source_id="src-1",
                error_type="LLMRateLimitError",
            )

        span.update.assert_any_call(output={"outcome": "transient_exhausted", "error_type": "LLMRateLimitError"})

    def test_no_op_without_active_trace(self):
        """Absence of the marker is meaningful (AC3): with no trace there is
        nothing to mark — no span, no tag write."""
        client, _span = _make_client()
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch("mirix.observability.skip_spans.get_trace_context", return_value={}),
            patch("mirix.observability.skip_spans.update_trace_attributes") as upd,
        ):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")

        client.start_as_current_observation.assert_not_called()
        upd.assert_not_called()

    def test_no_op_when_langfuse_disabled(self):
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=None),
            patch("mirix.observability.skip_spans.update_trace_attributes") as upd,
        ):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")
        upd.assert_not_called()

    def test_never_raises_on_span_creation_failure(self):
        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("boom")
        with (
            patch("mirix.observability.skip_spans.get_langfuse_client", return_value=client),
            patch(
                "mirix.observability.skip_spans.get_trace_context",
                return_value={"trace_id": "t-1", "observation_id": "o-1"},
            ),
        ):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")

    def test_never_raises_when_tag_helper_fails(self):
        client, _span = _make_client()
        p1, p2, p3, p4 = _patches(client)
        with (
            p1,
            p2,
            p3,
            p4,
            patch(
                "mirix.observability.skip_spans.update_trace_attributes",
                side_effect=RuntimeError("tag boom"),
            ),
        ):
            emit_save_outcome_span(SaveOutcome.SUCCESS, memory_source_id="src-1")
