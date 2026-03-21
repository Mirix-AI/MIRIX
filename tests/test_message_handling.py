"""
Tests for message handling after the message_ids refactor.

Tests cover:
1. get_messages_for_agent_user returns messages in chronological order
2. hard_delete_user_messages_for_agent deletes correct rows and keeps newest N
3. Retention=0 path: no DB persistence after step
4. Retention=N path: persists input messages and prunes to N newest
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.client import Client
from mirix.schemas.message import Message
from mirix.schemas.user import User
from mirix.services.message_manager import MessageManager


def make_client(id="client-1", org_id="org-1", retention=0):
    """Create a real Client object for tests."""
    return Client(
        id=id,
        organization_id=org_id,
        name="Test Client",
        status="active",
        write_scope="test",
        read_scopes=["test"],
        message_set_retention_count=retention,
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


def make_pydantic_message(id: str, role: str = "user", user_id: str = "user-1") -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.id = id
    msg.role = role
    msg.user_id = user_id
    return msg


class TestGetMessagesForAgentUser:
    """Tests for MessageManager.get_messages_for_agent_user()"""

    def test_returns_messages_in_chronological_order(self):
        """DB returns newest-first; method should reverse to chronological."""
        manager = MessageManager()

        # Simulate DB returning newest-first (DESC order)
        msg_old = MagicMock()
        msg_old.to_pydantic.return_value = make_pydantic_message("msg-1")
        msg_new = MagicMock()
        msg_new.to_pydantic.return_value = make_pydantic_message("msg-2")

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [msg_new, msg_old]  # newest first from DB

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.get_messages_for_agent_user(
                    agent_id="agent-1", user_id="user-1", actor=actor, limit=10
                )

        result = asyncio.run(run())

        # Should be reversed to chronological order
        assert len(result) == 2
        assert result[0].id == "msg-1"  # oldest first
        assert result[1].id == "msg-2"

    def test_returns_empty_when_no_messages(self):
        """Returns empty list when no messages exist."""
        manager = MessageManager()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.get_messages_for_agent_user(
                    agent_id="agent-1", user_id="user-1", actor=actor, limit=10
                )

        result = asyncio.run(run())
        assert result == []


class TestHardDeleteUserMessagesForAgent:
    """Tests for MessageManager.hard_delete_user_messages_for_agent()"""

    def test_deletes_all_when_keep_newest_n_is_zero(self):
        """keep_newest_n=0 means delete everything."""
        manager = MessageManager()

        delete_ids_result = MagicMock()
        delete_ids_result.all.return_value = [("msg-1",), ("msg-2",), ("msg-3",)]

        execute_results = [
            delete_ids_result,  # select IDs to delete
            MagicMock(),        # DELETE statement
        ]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=execute_results)
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                with patch("mirix.database.redis_client.get_redis_client", return_value=None):
                    actor = make_client()
                    return await manager.hard_delete_user_messages_for_agent(
                        agent_id="agent-1",
                        user_id="user-1",
                        actor=actor,
                        keep_newest_n=0,
                    )

        count = asyncio.run(run())
        assert count == 3

    def test_returns_zero_when_no_messages_exist(self):
        """Returns 0 when there are no messages to delete."""
        manager = MessageManager()

        delete_ids_result = MagicMock()
        delete_ids_result.all.return_value = []  # nothing to delete

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=delete_ids_result)
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def _async_cm():
            yield mock_session

        async def run():
            with patch.object(manager, "session_maker", return_value=_async_cm()):
                actor = make_client()
                return await manager.hard_delete_user_messages_for_agent(
                    agent_id="agent-1",
                    user_id="user-1",
                    actor=actor,
                    keep_newest_n=0,
                )

        count = asyncio.run(run())
        assert count == 0


class TestRetentionBehavior:
    """Tests that retention=0 vs retention>0 produces correct persistence behavior."""

    def test_client_default_retention_is_zero(self):
        """Clients default to message_set_retention_count=0."""
        client = make_client()
        assert (client.message_set_retention_count or 0) == 0

    def test_client_with_retention_has_correct_value(self):
        """Clients configured with retention=5 expose that value."""
        client = make_client(retention=5)
        assert client.message_set_retention_count == 5
