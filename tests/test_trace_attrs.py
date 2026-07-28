"""Unit tests for the central trace-attribute helper (``update_trace_attributes``).

Trace-level tags in Langfuse are replaced wholesale on every
``update_current_trace(tags=...)`` write, and the save trace is written from
two independent processes (HTTP entry + worker) plus the finalize emitter —
last writer wins. The helper kills that bug structurally: it accumulates the
per-context tag set in a ContextVar and always writes the FULL set, so
whatever the flush order, the trace converges to the union.

Pure-python: the Langfuse client is mocked; no DB, no network.
"""

from unittest.mock import MagicMock, patch

import pytest

from mirix.observability import context as obs_context
from mirix.observability.trace_attrs import (
    reset_trace_tags,
    update_trace_attributes,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Each test starts and ends with no trace context and an empty tag set."""
    obs_context.clear_trace_context()
    reset_trace_tags()
    yield
    obs_context.clear_trace_context()
    reset_trace_tags()


def _langfuse():
    return MagicMock()


def _with_trace(trace_id="trace-1"):
    obs_context.set_trace_context(trace_id=trace_id, observation_id="obs-1")


class TestUpdateTraceAttributes:
    def test_tags_merge_across_calls_and_full_set_is_written(self):
        """Successive calls accumulate; every write sends the full sorted set."""
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["tid:t-1"])
            update_trace_attributes(tags=["client:aqna-client"])
            update_trace_attributes(tags=["write_kind:extraction"])

        calls = langfuse.update_current_trace.call_args_list
        assert len(calls) == 3
        assert calls[0].kwargs["tags"] == ["tid:t-1"]
        assert calls[1].kwargs["tags"] == sorted(["tid:t-1", "client:aqna-client"])
        assert calls[2].kwargs["tags"] == sorted(["tid:t-1", "client:aqna-client", "write_kind:extraction"])

    def test_duplicate_tags_are_set_deduplicated(self):
        """Re-adding the same tag (e.g. a deduped redelivery re-tagging the same
        outcome) does not duplicate it — set semantics."""
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["save_outcome:success"])
            update_trace_attributes(tags=["save_outcome:success"])

        last = langfuse.update_current_trace.call_args_list[-1]
        assert last.kwargs["tags"] == ["save_outcome:success"]

    def test_metadata_passed_through_per_call(self):
        """Metadata is forwarded per call (Langfuse merges metadata by key
        server-side, so no client-side accumulation is needed)."""
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["tid:t-1"], metadata={"tid": "t-1"})
            update_trace_attributes(metadata={"write_kind": "direct"})

        calls = langfuse.update_current_trace.call_args_list
        assert calls[0].kwargs["metadata"] == {"tid": "t-1"}
        assert calls[1].kwargs["metadata"] == {"write_kind": "direct"}
        # A metadata-only call still writes the full accumulated tag set.
        assert calls[1].kwargs["tags"] == ["tid:t-1"]

    def test_tag_values_used_verbatim_including_colons(self):
        """Values may themselves contain ':' (the tid: precedent embeds arbitrary
        gateway TIDs) — no escaping or splitting."""
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["tid:gw:12:34"])

        assert langfuse.update_current_trace.call_args.kwargs["tags"] == ["tid:gw:12:34"]

    def test_no_op_when_langfuse_disabled(self):
        _with_trace()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=None,
        ):
            update_trace_attributes(tags=["client:x"])  # must not raise

    def test_no_op_when_no_trace_id_active(self):
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["client:x"])
        langfuse.update_current_trace.assert_not_called()

    def test_never_raises_when_sdk_call_fails(self):
        _with_trace()
        langfuse = _langfuse()
        langfuse.update_current_trace.side_effect = RuntimeError("boom")
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["client:x"])  # must not raise


class TestReset:
    def test_reset_trace_tags_empties_the_set(self):
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["tid:t-1", "client:c-1"])
            reset_trace_tags()
            update_trace_attributes(tags=["tid:t-2"])

        last = langfuse.update_current_trace.call_args_list[-1]
        assert last.kwargs["tags"] == ["tid:t-2"]

    def test_clear_trace_context_resets_the_tag_set(self):
        """dispatch_save's finally calls clear_trace_context(); the tag set must
        reset with it so one save's tags never leak into the next message on a
        reused worker task."""
        _with_trace()
        langfuse = _langfuse()
        with patch(
            "mirix.observability.trace_attrs.get_langfuse_client",
            return_value=langfuse,
        ):
            update_trace_attributes(tags=["save_outcome:success"])
            obs_context.clear_trace_context()
            _with_trace(trace_id="trace-2")
            update_trace_attributes(tags=["tid:t-next"])

        last = langfuse.update_current_trace.call_args_list[-1]
        assert last.kwargs["tags"] == ["tid:t-next"]
