"""Unit tests for get_client_or_404 and its use across rest_api.py routes.

client_manager.get_client_by_id raises NoResultFound (never returns None) on
a missing client. get_client_or_404 is the shared helper that maps that to
HTTPException(404); these tests confirm the helper itself, and a sample of
call sites that previously had no guard at all, now surface a 404 instead
of an unhandled 500.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mirix.orm.errors import NoResultFound


def _mock_server_with_missing_client():
    mock_server = MagicMock()
    mock_server.client_manager.get_client_by_id = AsyncMock(side_effect=NoResultFound("Client not found"))
    return mock_server


def _mock_server_with_client(client):
    mock_server = MagicMock()
    mock_server.client_manager.get_client_by_id = AsyncMock(return_value=client)
    return mock_server


class TestGetClientOr404:
    @pytest.mark.asyncio
    async def test_returns_client_when_found(self):
        from mirix.server.rest_api import get_client_or_404

        client = MagicMock()
        with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_client(client)):
            result = await get_client_or_404("client-1")

        assert result is client

    @pytest.mark.asyncio
    async def test_raises_404_when_client_missing(self):
        from mirix.server.rest_api import get_client_or_404

        with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_missing_client()):
            with pytest.raises(HTTPException) as exc_info:
                await get_client_or_404("client-missing")

        assert exc_info.value.status_code == 404


class TestListAgentsRoute:
    @pytest.mark.asyncio
    async def test_404_when_client_not_registered(self):
        """list_agents previously had no guard at all around get_client_by_id."""
        from mirix.server.rest_api import list_agents

        with patch(
            "mirix.server.rest_api.get_client_and_org", new_callable=AsyncMock, return_value=("client-missing", "org-1")
        ):
            with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_missing_client()):
                with pytest.raises(HTTPException) as exc_info:
                    await list_agents(x_client_id="client-missing")

                assert exc_info.value.status_code == 404


class TestGetClientRoute:
    @pytest.mark.asyncio
    async def test_404_when_client_not_registered(self):
        """get_client previously had a dead if-not-client check (get_client_by_id raises, never returns None)."""
        from mirix.server.rest_api import get_client

        with patch("mirix.server.rest_api.get_current_admin", return_value={"sub": "admin-1"}):
            with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_missing_client()):
                with pytest.raises(HTTPException) as exc_info:
                    await get_client(client_id="client-missing", authorization="Bearer x")

                assert exc_info.value.status_code == 404


class TestGetClientFromJwtOrApiKeyRoute:
    @pytest.mark.asyncio
    async def test_404_when_header_client_not_registered(self):
        """The injected-header branch had no guard at all around get_client_by_id."""
        from mirix.server.rest_api import get_client_from_jwt_or_api_key

        mock_request = MagicMock()
        mock_request.headers = {"x-client-id": "client-missing", "x-org-id": "org-1"}

        with patch(
            "mirix.server.rest_api.get_client_and_org",
            new_callable=AsyncMock,
            return_value=("client-missing", "org-1"),
        ):
            with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_missing_client()):
                with pytest.raises(HTTPException) as exc_info:
                    await get_client_from_jwt_or_api_key(authorization=None, request=mock_request)

                assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_401_when_no_auth_provided(self):
        """No authorization header and no x-client-id header still 401s (unrelated to the client lookup)."""
        from mirix.server.rest_api import get_client_from_jwt_or_api_key

        mock_request = MagicMock()
        mock_request.headers = {}

        with pytest.raises(HTTPException) as exc_info:
            await get_client_from_jwt_or_api_key(authorization=None, request=mock_request)

        assert exc_info.value.status_code == 401


class TestRawMemoryClientResolution:
    @pytest.mark.asyncio
    async def test_401_when_neither_client_nor_client_id_given(self):
        """create_raw_memory's own guard (nothing passed at all) must survive the get_client_or_404 swap."""
        from mirix.schemas.raw_memory import RawMemoryItemCreateRequest
        from mirix.server.rest_api import create_raw_memory

        req = RawMemoryItemCreateRequest(context="c")

        with patch("mirix.server.rest_api.get_server", return_value=MagicMock()):
            with pytest.raises(HTTPException) as exc_info:
                await create_raw_memory(req, user_id="user-1")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_404_when_client_id_given_but_not_registered(self):
        """A client_id that doesn't resolve 404s distinctly from the 'nothing given' 401 case."""
        from mirix.schemas.raw_memory import RawMemoryItemCreateRequest
        from mirix.server.rest_api import create_raw_memory

        req = RawMemoryItemCreateRequest(context="c")

        with patch("mirix.server.rest_api.get_server", return_value=_mock_server_with_missing_client()):
            with pytest.raises(HTTPException) as exc_info:
                await create_raw_memory(req, user_id="user-1", client_id="client-missing")

            assert exc_info.value.status_code == 404
