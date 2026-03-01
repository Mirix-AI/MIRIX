"""
Unit tests for MirixClient send_message and user_message block_filter_tags.

Verifies that block_filter_tags is included in the request body when calling
POST /agents/{agent_id}/messages (send_message and user_message).
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mirix.client import MirixClient


@pytest.fixture
def client():
    """MirixClient with mocked _request."""
    with patch.object(MirixClient, "_request") as mock_request:
        mock_request.return_value = {"messages": [], "usage": {}}
        c = MirixClient(api_key="test-key", debug=False)
        yield c


class TestMirixClientSendMessageBlockFilterTags:
    """Test that send_message includes block_filter_tags in the request."""

    def test_send_message_includes_block_filter_tags_in_request(self, client):
        """When block_filter_tags is provided, it is sent in the request body."""
        block_filter_tags = {"env": "staging", "team": "platform"}
        client._request = Mock(return_value={"messages": [], "usage": {}})

        client.send_message(
            message="Hello",
            role="user",
            agent_id="agent-123",
            block_filter_tags=block_filter_tags,
        )

        client._request.assert_called_once()
        call_kwargs = client._request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"].get("block_filter_tags") == block_filter_tags

    def test_send_message_without_block_filter_tags(self, client):
        """When block_filter_tags is not provided, it is not in the request body."""
        client._request = Mock(return_value={"messages": [], "usage": {}})

        client.send_message(
            message="Hello",
            role="user",
            agent_id="agent-123",
        )

        client._request.assert_called_once()
        call_kwargs = client._request.call_args.kwargs
        assert "json" in call_kwargs
        assert "block_filter_tags" not in call_kwargs["json"]


class TestMirixClientUserMessageBlockFilterTags:
    """Test that user_message passes block_filter_tags to send_message."""

    def test_user_message_includes_block_filter_tags_in_request(self, client):
        """user_message(block_filter_tags=...) results in block_filter_tags in request."""
        block_filter_tags = {"env": "prod"}
        client._request = Mock(return_value={"messages": [], "usage": {}})

        client.user_message(
            agent_id="agent-456",
            message="Hi",
            block_filter_tags=block_filter_tags,
        )

        client._request.assert_called_once()
        call_kwargs = client._request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"].get("block_filter_tags") == block_filter_tags

    def test_user_message_passes_block_filter_tags_to_send_message(self, client):
        """user_message forwards block_filter_tags to send_message."""
        with patch.object(client, "send_message", wraps=client.send_message) as mock_send:
            client._request = Mock(return_value={"messages": [], "usage": {}})
            client.user_message(
                agent_id="agent-789",
                message="Test",
                user_id="user-1",
                block_filter_tags={"tag": "value"},
            )
            mock_send.assert_called_once()
            assert mock_send.call_args.kwargs.get("block_filter_tags") == {"tag": "value"}
