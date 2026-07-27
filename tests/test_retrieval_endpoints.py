"""Unit tests for S9 retrieval endpoints — memory sources and threads.

Tests the manager retrieval methods and API route handlers using mocks.
Validates scope-based access control: memory sources use filter_tags->>'scope'
matching the same pattern as memory tables (client.read_scopes).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.memory_source import MemorySource, PaginatedResponse
from mirix.schemas.source_message import SourceMessage

# --- Test data factories ---

SRC_ID_1 = "src-aaaaaaaa-1111-2222-3333-444444444444"
SRC_ID_2 = "src-bbbbbbbb-1111-2222-3333-444444444444"
SMSG_ID_1 = "smsg-aaaaaaaa-1111-2222-3333-444444444444"
SMSG_ID_2 = "smsg-bbbbbbbb-1111-2222-3333-444444444444"


_SENTINEL = object()


def _make_source(
    id=SRC_ID_1,
    client_id="client-1",
    user_id="user-1",
    external_thread_id="thread-1",
    occurred_at=None,
    filter_tags=_SENTINEL,
):
    return MemorySource(
        id=id,
        client_id=client_id,
        user_id=user_id,
        organization_id="org-1",
        external_thread_id=external_thread_id,
        source_type="conversation",
        processing_complete=False,
        occurred_at=occurred_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
        filter_tags={"scope": "sbg"} if filter_tags is _SENTINEL else filter_tags,
    )


def _make_message(id=SMSG_ID_1, memory_source_id=SRC_ID_1, sequence_num=0, role="user", content=None):
    return SourceMessage(
        id=id,
        memory_source_id=memory_source_id,
        role=role,
        content=content or {"text": "hello"},
        sequence_num=sequence_num,
        content_hash="abc123",
    )


def _make_client(client_id="client-1", read_scopes=None, write_scope="sbg"):
    """Create a mock client with scope configuration."""
    client = MagicMock()
    client.id = client_id
    client.read_scopes = read_scopes if read_scopes is not None else ["sbg"]
    client.write_scope = write_scope
    return client


# --- SourceMessageManager.get_messages_by_source_id ---


class TestGetMessagesBySourceId:

    @pytest.mark.asyncio
    async def test_returns_messages_for_source(self):
        from mirix.services.source_message_manager import SourceMessageManager

        msg1 = _make_message(id=SMSG_ID_1, sequence_num=0)
        msg2 = _make_message(id=SMSG_ID_2, sequence_num=1)

        mock_record1 = MagicMock()
        mock_record1.to_pydantic.return_value = msg1
        mock_record2 = MagicMock()
        mock_record2.to_pydantic.return_value = msg2

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_record1, mock_record2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_context():
            yield mock_session

        mgr = SourceMessageManager.__new__(SourceMessageManager)
        mgr.session_maker = mock_context

        result = await mgr.get_messages_by_source_id(memory_source_id="src-1")

        assert len(result.items) == 2
        assert result.items[0].id == SMSG_ID_1
        assert result.items[1].id == SMSG_ID_2
        assert result.has_more is False
        assert result.next_cursor is None


# --- REST API route handlers ---


def _mock_server_with_client(client):
    """Set up mock server that returns the given client from client_manager."""
    mock_server = MagicMock()
    mock_server.client_manager.get_client_by_id = AsyncMock(return_value=client)
    return mock_server


def _mock_server_with_unregistered_client():
    """Set up mock server whose client_manager raises NoResultFound (unregistered client)."""
    from mirix.orm.errors import NoResultFound

    mock_server = MagicMock()
    mock_server.client_manager.get_client_by_id = AsyncMock(side_effect=NoResultFound("Client not found"))
    return mock_server


class TestGetMemorySourceRoute:

    @pytest.mark.asyncio
    async def test_returns_source_when_scope_matches(self):
        from mirix.server.rest_api import get_memory_source

        source = _make_source(filter_tags={"scope": "sbg"})
        client = _make_client(read_scopes=["sbg"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                    result = await get_memory_source(source_id=SRC_ID_1, x_client_id="client-1")

        assert result.id == SRC_ID_1

    @pytest.mark.asyncio
    async def test_cross_client_scope_access(self):
        """Client B can read a source created by Client A if the scope is in Client B's read_scopes."""
        from mirix.server.rest_api import get_memory_source

        source = _make_source(client_id="client-a", filter_tags={"scope": "shared-scope"})
        client_b = _make_client(client_id="client-b", read_scopes=["shared-scope"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-b", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client_b)):
                    result = await get_memory_source(source_id=SRC_ID_1, x_client_id="client-b")

        assert result.id == SRC_ID_1

    @pytest.mark.asyncio
    async def test_404_when_not_found(self):
        from mirix.server.rest_api import get_memory_source

        client = _make_client(read_scopes=["sbg"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=None)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                    from fastapi import HTTPException as FastHTTPException

                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source(source_id="src-missing", x_client_id="client-1")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_scope_not_in_read_scopes(self):
        """Client cannot read a source whose scope is not in its read_scopes."""
        from mirix.server.rest_api import get_memory_source

        source = _make_source(filter_tags={"scope": "tax"})
        client = _make_client(read_scopes=["sbg"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                    from fastapi import HTTPException as FastHTTPException

                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source(source_id=SRC_ID_1, x_client_id="client-1")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_no_filter_tags(self):
        """Source with no filter_tags (legacy) is not accessible."""
        from mirix.server.rest_api import get_memory_source

        source = _make_source(filter_tags=None)
        client = _make_client(read_scopes=["sbg"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockMgr:
            mock_instance = MockMgr.return_value
            mock_instance.get_by_id = AsyncMock(return_value=source)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                    from fastapi import HTTPException as FastHTTPException

                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source(source_id=SRC_ID_1, x_client_id="client-1")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_client_not_registered(self):
        """An unregistered X-Client-UUID raises 404, not an unhandled NoResultFound."""
        from mirix.server.rest_api import get_memory_source

        with patch(
            "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-missing", "org-1")
        ):
            with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_unregistered_client()):
                from fastapi import HTTPException as FastHTTPException

                with pytest.raises(FastHTTPException) as exc_info:
                    await get_memory_source(source_id=SRC_ID_1, x_client_id="client-missing")

                assert exc_info.value.status_code == 404


class TestGetMemorySourceMessagesRoute:

    @pytest.mark.asyncio
    async def test_returns_messages_when_scope_matches(self):
        from mirix.server.rest_api import get_memory_source_messages

        source = _make_source(filter_tags={"scope": "sbg"})
        client = _make_client(read_scopes=["sbg"])
        messages = [_make_message(id=SMSG_ID_1), _make_message(id=SMSG_ID_2, sequence_num=1)]
        page = PaginatedResponse(items=messages, next_cursor=None, has_more=False)

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockSourceMgr:
            mock_source_instance = MockSourceMgr.return_value
            mock_source_instance.get_by_id = AsyncMock(return_value=source)

            with patch("mirix.services.source_message_manager.SourceMessageManager") as MockMsgMgr:
                mock_msg_instance = MockMsgMgr.return_value
                mock_msg_instance.get_messages_by_source_id = AsyncMock(return_value=page)

                with patch(
                    "mirix.server.rest_api.get_client_and_org",
                    new_callable=AsyncMock,
                    return_value=("client-1", "org-1"),
                ):
                    with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                        result = await get_memory_source_messages(source_id=SRC_ID_1, x_client_id="client-1")

        assert len(result.items) == 2
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_404_when_scope_mismatch(self):
        from mirix.server.rest_api import get_memory_source_messages

        source = _make_source(filter_tags={"scope": "tax"})
        client = _make_client(read_scopes=["sbg"])

        with patch("mirix.services.memory_source_manager.MemorySourceManager") as MockSourceMgr:
            mock_source_instance = MockSourceMgr.return_value
            mock_source_instance.get_by_id = AsyncMock(return_value=source)

            with patch(
                "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-1", "org-1")
            ):
                with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
                    from fastapi import HTTPException as FastHTTPException

                    with pytest.raises(FastHTTPException) as exc_info:
                        await get_memory_source_messages(source_id=SRC_ID_1, x_client_id="client-1")

                    assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_client_not_registered(self):
        """An unregistered X-Client-UUID raises 404, not an unhandled NoResultFound."""
        from mirix.server.rest_api import get_memory_source_messages

        with patch(
            "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-missing", "org-1")
        ):
            with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_unregistered_client()):
                from fastapi import HTTPException as FastHTTPException

                with pytest.raises(FastHTTPException) as exc_info:
                    await get_memory_source_messages(source_id=SRC_ID_1, x_client_id="client-missing")

                assert exc_info.value.status_code == 404
