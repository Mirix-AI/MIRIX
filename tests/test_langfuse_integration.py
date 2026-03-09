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

        with patch("langfuse.Langfuse") as MockLangfuse, patch("opentelemetry.sdk.trace.TracerProvider"):
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
