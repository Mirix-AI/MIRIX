"""
Tests for message handling, particularly the race condition fix.

Tests cover:
1. get_messages_by_ids gracefully handles missing message IDs
2. get_in_context_messages filters by user when user is provided
3. get_in_context_messages returns all messages when user is not provided
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client
from mirix.schemas.message import Message
from mirix.schemas.user import User
from mirix.services.agent_manager import AgentManager
from mirix.services.message_manager import MessageManager


def make_client(id="client-1", org_id="org-1"):
    """Create a real Client object for tests."""
    return Client(
        id=id,
        organization_id=org_id,
        name="Test Client",
        status="active",
        scope="test",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def make_user(id="user-1", org_id="org-1"):
    """Create a real User object for tests."""
    return User(
        id=id,
        organization_id=org_id,
        name="Test User",
        status="active",
        timezone="UTC",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False,
    )


def make_agent_state(message_ids=None):
    """Create a mock AgentState with spec for type checking."""
    agent_state = MagicMock(spec=AgentState)
    agent_state.message_ids = message_ids or []
    return agent_state


class TestGetMessagesByIds:
    """Tests for MessageManager.get_messages_by_ids() - race condition fix"""

    def test_returns_existing_messages_skips_missing(self):
        """
        Test that get_messages_by_ids returns existing messages and skips missing ones.

        This is the key fix for the race condition - when concurrent workers
        delete messages via summarization, other workers should not crash.
        """
        manager = MessageManager()

        # Mock the session and MessageModel.list to return only 2 of 3 requested messages
        mock_session = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.id = "msg-1"
        mock_msg1.to_pydantic.return_value = MagicMock(id="msg-1")

        mock_msg2 = MagicMock()
        mock_msg2.id = "msg-2"
        mock_msg2.to_pydantic.return_value = MagicMock(id="msg-2")

        # msg-3 is "missing" (simulates deletion by another worker)

        with patch.object(manager, "session_maker") as mock_session_maker:
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=mock_session)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_session_maker.return_value = mock_context

            with patch(
                "mirix.services.message_manager.MessageModel"
            ) as MockMessageModel:
                MockMessageModel.list.return_value = [mock_msg1, mock_msg2]

                actor = make_client()

                # Request 3 messages, but only 2 exist
                result = manager.get_messages_by_ids(
                    message_ids=["msg-1", "msg-2", "msg-3"], actor=actor
                )

                # Should return only the 2 that exist, not crash
                assert len(result) == 2
                assert result[0].id == "msg-1"
                assert result[1].id == "msg-2"

    def test_preserves_order_of_existing_messages(self):
        """Test that returned messages maintain the requested order."""
        manager = MessageManager()

        mock_msg2 = MagicMock()
        mock_msg2.id = "msg-2"
        mock_msg2.to_pydantic.return_value = MagicMock(id="msg-2")

        mock_msg1 = MagicMock()
        mock_msg1.id = "msg-1"
        mock_msg1.to_pydantic.return_value = MagicMock(id="msg-1")

        with patch.object(manager, "session_maker") as mock_session_maker:
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=MagicMock())
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_session_maker.return_value = mock_context

            with patch(
                "mirix.services.message_manager.MessageModel"
            ) as MockMessageModel:
                # DB returns in different order
                MockMessageModel.list.return_value = [mock_msg2, mock_msg1]

                actor = make_client()

                result = manager.get_messages_by_ids(
                    message_ids=["msg-1", "msg-2"], actor=actor  # Requested order
                )

                # Should be in requested order, not DB order
                assert result[0].id == "msg-1"
                assert result[1].id == "msg-2"

    def test_returns_empty_list_when_all_missing(self):
        """Test that an empty list is returned when all messages are missing."""
        manager = MessageManager()

        with patch.object(manager, "session_maker") as mock_session_maker:
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=MagicMock())
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_session_maker.return_value = mock_context

            with patch(
                "mirix.services.message_manager.MessageModel"
            ) as MockMessageModel:
                MockMessageModel.list.return_value = []  # All missing

                actor = make_client()

                result = manager.get_messages_by_ids(
                    message_ids=["msg-1", "msg-2"], actor=actor
                )

                assert result == []


class TestGetInContextMessages:
    """Tests for AgentManager.get_in_context_messages() - user filtering fix"""

    def test_filters_by_user_id_when_user_provided(self):
        """
        Test that messages are filtered by user.id when user parameter is provided.

        This fixes the bug where actor.id (client ID) was used instead of user.id.
        """
        manager = AgentManager()

        # Create mock messages
        system_msg = MagicMock()
        system_msg.user_id = "system"

        user_a_msg = MagicMock()
        user_a_msg.user_id = "user-a"

        user_b_msg = MagicMock()
        user_b_msg.user_id = "user-b"

        with patch.object(manager, "message_manager") as mock_msg_manager:
            mock_msg_manager.get_messages_by_ids.return_value = [
                system_msg,
                user_a_msg,
                user_b_msg,
            ]

            agent_state = make_agent_state(message_ids=["sys-1", "msg-a", "msg-b"])
            actor = make_client(id="client-123")
            user = make_user(id="user-a")  # Should filter to this user's messages

            result = manager.get_in_context_messages(
                agent_state=agent_state, actor=actor, user=user
            )

            # Should have system message + only user-a's message
            assert len(result) == 2
            assert result[0] == system_msg
            assert result[1] == user_a_msg

    def test_no_filtering_when_user_not_provided(self):
        """
        Test that all messages are returned when user parameter is not provided.

        This maintains backward compatibility.
        """
        manager = AgentManager()

        system_msg = MagicMock()
        user_a_msg = MagicMock()
        user_b_msg = MagicMock()

        with patch.object(manager, "message_manager") as mock_msg_manager:
            mock_msg_manager.get_messages_by_ids.return_value = [
                system_msg,
                user_a_msg,
                user_b_msg,
            ]

            agent_state = make_agent_state(message_ids=["sys-1", "msg-a", "msg-b"])
            actor = make_client()

            # No user parameter
            result = manager.get_in_context_messages(
                agent_state=agent_state, actor=actor
            )

            # Should return all messages (no filtering)
            assert len(result) == 3

    def test_returns_empty_when_no_messages(self):
        """Test that empty list is returned when agent has no messages."""
        manager = AgentManager()

        with patch.object(manager, "message_manager") as mock_msg_manager:
            mock_msg_manager.get_messages_by_ids.return_value = []

            agent_state = make_agent_state(message_ids=[])
            actor = make_client()

            result = manager.get_in_context_messages(
                agent_state=agent_state, actor=actor
            )

            assert result == []
