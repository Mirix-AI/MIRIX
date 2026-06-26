"""Tests for LangFuse integration."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import pytest_asyncio

from mirix.observability.langfuse_client import (
    _reset_for_testing,
    flush_langfuse,
    get_langfuse_client,
    initialize_langfuse,
    is_langfuse_enabled,
    shutdown_langfuse,
)
from mirix.observability.trace_propagation import (
    TRACE_METADATA_KEY,
    add_trace_to_message,
    deserialize_trace_context,
    serialize_trace_context,
)


@pytest_asyncio.fixture(autouse=True)
async def reset_singleton():
    """Reset singleton before each test. Must be pytest_asyncio.fixture so async tests await it."""
    await _reset_for_testing()
    yield
    await _reset_for_testing()


@pytest.mark.asyncio
async def test_langfuse_disabled_by_default():
    """Test LangFuse is disabled without configuration."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = False

        client = await initialize_langfuse()
        assert client is None
        assert not is_langfuse_enabled()


@pytest.mark.asyncio
async def test_langfuse_initialization_with_credentials():
    """Test LangFuse initializes with valid credentials."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"
        mock_settings.langfuse_debug = False
        mock_settings.langfuse_flush_interval = 10

        with patch("langfuse.Langfuse") as MockLangfuse, patch("opentelemetry.sdk.trace.TracerProvider"):
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            client = await initialize_langfuse()
            assert client is not None
            assert is_langfuse_enabled()

            mock_client.flush.assert_called_once()


@pytest.mark.asyncio
async def test_langfuse_missing_credentials():
    """Test LangFuse handles missing credentials gracefully."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = None
        mock_settings.langfuse_secret_key = None

        client = await initialize_langfuse()
        assert client is None
        assert not is_langfuse_enabled()


def test_trace_context_serialization():
    """Test trace context can be serialized for Kafka."""
    from mirix.observability.context import set_trace_context

    set_trace_context(trace_id="trace-123", user_id="user-456", session_id="session-789")

    serialized = serialize_trace_context()

    assert serialized is not None
    assert serialized["trace_id"] == "trace-123"
    assert serialized["user_id"] == "user-456"
    assert serialized["session_id"] == "session-789"


def test_trace_context_serialization_no_trace():
    """Test serialization returns None when no trace context."""
    from mirix.observability.context import clear_trace_context

    clear_trace_context()
    serialized = serialize_trace_context()

    assert serialized is None


def test_trace_context_deserialization():
    """Test trace context can be restored from Kafka message."""
    from mirix.observability.context import get_trace_context

    message = {
        TRACE_METADATA_KEY: {
            "trace_id": "trace-abc",
            "user_id": "user-xyz",
        }
    }

    result = deserialize_trace_context(message)

    assert result is True

    context = get_trace_context()
    assert context["trace_id"] == "trace-abc"
    assert context["user_id"] == "user-xyz"


def test_trace_context_deserialization_no_metadata():
    """Test deserialization handles missing metadata."""
    message = {"some_key": "some_value"}

    result = deserialize_trace_context(message)

    assert result is False


def test_add_trace_to_message():
    """Test adding trace context to Kafka message."""
    from mirix.observability.context import set_trace_context

    set_trace_context(trace_id="trace-test-123")

    message = {"data": "test"}
    result = add_trace_to_message(message)

    assert TRACE_METADATA_KEY in result
    assert result[TRACE_METADATA_KEY]["trace_id"] == "trace-test-123"
    assert result["data"] == "test"


@pytest.mark.asyncio
async def test_graceful_degradation():
    """Test that operations work without LangFuse."""
    with patch("mirix.observability.langfuse_client.get_langfuse_client") as mock_get:
        mock_get.return_value = None

        assert not is_langfuse_enabled()
        assert await flush_langfuse() is True
        await shutdown_langfuse()


@pytest.mark.asyncio
async def test_flush_langfuse_with_timeout():
    """Test flush respects timeout parameter."""
    with patch("mirix.settings.settings") as mock_settings:
        mock_settings.langfuse_enabled = True
        mock_settings.langfuse_public_key = "pk-test"
        mock_settings.langfuse_secret_key = "sk-test"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"
        mock_settings.langfuse_debug = False
        mock_settings.langfuse_flush_interval = 10
        mock_settings.langfuse_flush_timeout = 5.0
        mock_settings.span_export_file = None  # no file sink in this test

        import mirix.observability.tracer_provider as _tp

        _tp._reset_for_testing()

        with patch("langfuse.Langfuse") as MockLangfuse:
            mock_client = MagicMock()
            MockLangfuse.return_value = mock_client

            await initialize_langfuse()
            result = await flush_langfuse(timeout=15.0)

            assert result is True
            mock_client.flush.assert_called()


def test_context_isolation():
    """Test that trace contexts don't leak between operations."""
    from mirix.observability.context import clear_trace_context, get_trace_context, set_trace_context

    set_trace_context(trace_id="trace-1")
    assert get_trace_context()["trace_id"] == "trace-1"

    clear_trace_context()
    assert get_trace_context()["trace_id"] is None

    set_trace_context(trace_id="trace-2")
    assert get_trace_context()["trace_id"] == "trace-2"


# ============================================================================
# TID (request correlation) propagation
#
# The TID is propagated independently of LangFuse trace context: it must reach
# the worker even when tracing is disabled (no active trace_id). These tests
# pin that behavior on the protobuf path used by the real Kafka queue.
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_tid():
    """Keep TID state clean around each test in this module."""
    from mirix.observability.context import clear_tid

    clear_tid()
    yield
    clear_tid()


def test_tid_propagates_to_queue_message_without_active_trace():
    """TID is copied onto the protobuf message even when there is NO trace."""
    from mirix.observability.context import clear_trace_context, set_tid
    from mirix.observability.trace_propagation import add_trace_to_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    clear_trace_context()  # ensure no active trace -> trace fields skip
    set_tid("tid-no-trace")

    msg = add_trace_to_queue_message(QueueMessage())

    assert msg.HasField("tid")
    assert msg.tid == "tid-no-trace"
    # And the trace early-return held: no trace fields were populated.
    assert not msg.HasField("langfuse_trace_id")


def test_tid_restored_from_queue_message_without_trace_fields():
    """TID is restored into context independent of trace fields presence."""
    from mirix.observability.context import get_tid
    from mirix.observability.trace_propagation import restore_trace_from_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    msg = QueueMessage()
    msg.tid = "tid-restore"

    # No trace fields set -> returns False for trace, but TID still restored.
    trace_restored = restore_trace_from_queue_message(msg)

    assert trace_restored is False
    assert get_tid() == "tid-restore"


def test_tid_and_trace_propagate_together():
    """When both are present, both make the round trip onto the message."""
    from mirix.observability.context import clear_trace_context, get_tid, set_tid, set_trace_context
    from mirix.observability.trace_propagation import add_trace_to_queue_message, restore_trace_from_queue_message
    from mirix.queue.message_pb2 import QueueMessage

    clear_trace_context()
    set_trace_context(trace_id="trace-xyz", user_id="user-1")
    set_tid("tid-both")

    msg = add_trace_to_queue_message(QueueMessage())
    assert msg.tid == "tid-both"
    assert msg.langfuse_trace_id == "trace-xyz"

    # Fresh worker side: clear, then restore.
    clear_trace_context()
    from mirix.observability.context import clear_tid

    clear_tid()
    restore_trace_from_queue_message(msg)
    assert get_tid() == "tid-both"


def test_tid_isolation():
    """TID does not leak between operations once cleared."""
    from mirix.observability.context import clear_tid, get_tid, set_tid

    set_tid("tid-1")
    assert get_tid() == "tid-1"

    clear_tid()
    assert get_tid() is None

    set_tid("tid-2")
    assert get_tid() == "tid-2"


def test_span_export_file_writes_jsonl(tmp_path):
    """With span_export_file set, finished spans are written to a JSONL file.

    Drives the shared TracerProvider directly with a real OTel span (no remote
    Langfuse needed) — the path full-stack tests rely on to read spans off disk.
    """
    import json
    from unittest.mock import patch

    import mirix.observability.tracer_provider as tp
    from mirix.settings import settings

    span_file = tmp_path / "spans.jsonl"
    tp._reset_for_testing()
    try:
        with patch.object(settings, "span_export_file", span_file):
            provider = tp.get_shared_tracer_provider()
            tracer = provider.get_tracer("test")
            with tracer.start_as_current_span("List Tools By Ids") as span:
                span.set_attribute("tid", "tid-span-export")
                span.set_attribute("tool_id_count", 3)
            provider.force_flush()

        lines = [ln for ln in span_file.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["name"] == "List Tools By Ids"
        assert rec["attributes"]["tid"] == "tid-span-export"
        assert rec["attributes"]["tool_id_count"] == 3
        assert rec["trace_id"] and rec["span_id"]
        assert rec["duration_ms"] is not None
    finally:
        tp._reset_for_testing()


def test_log_filter_injects_tid():
    """The logging filter stamps the current TID onto records ('-' when unset)."""
    import logging

    from mirix.log import _tid_log_filter
    from mirix.observability.context import clear_tid, set_tid

    set_tid("tid-log")
    rec = logging.LogRecord("Mirix", logging.INFO, "/tmp/x.py", 1, "msg", None, None)
    _tid_log_filter.filter(rec)
    assert rec.tid == "tid-log"

    clear_tid()
    rec2 = logging.LogRecord("Mirix", logging.INFO, "/tmp/x.py", 1, "msg", None, None)
    _tid_log_filter.filter(rec2)
    assert rec2.tid == "-"
