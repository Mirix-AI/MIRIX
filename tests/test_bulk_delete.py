"""
M4.5: delete_by_client_id uses bulk SQL (session_maker) when no relational
provider delegation applies — flow remains intact after provider changes.

Run: pytest tests/test_bulk_delete.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.schemas.client import Client as PydanticClient
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager


def _async_session_context(mock_session: AsyncMock):
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def _actor() -> PydanticClient:
    return PydanticClient(
        id="client-bulk-del-1",
        name="Bulk Delete Client",
        organization_id="org-bulk-1",
    )


@pytest.mark.asyncio
async def test_episodic_delete_by_client_id_uses_session_maker_sql_path():
    actor = _actor()
    mock_session = _async_session_context(AsyncMock())

    select_result = MagicMock()
    select_result.all.return_value = [("ep-mem-1",), ("ep-mem-2",)]
    delete_result = MagicMock()

    mock_session.execute = AsyncMock(side_effect=[select_result, delete_result])

    mock_session_maker = MagicMock(return_value=mock_session)

    manager = EpisodicMemoryManager()
    manager.session_maker = mock_session_maker

    with patch("mirix.database.redis_client.get_redis_client", return_value=None):
        deleted = await manager.delete_by_client_id(actor)

    assert deleted == 2
    mock_session_maker.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()
    assert mock_session.execute.await_count == 2
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_episodic_delete_by_client_id_returns_zero_without_delete_query():
    """No rows: early return after first SELECT; session still entered."""
    actor = _actor()
    mock_session = _async_session_context(AsyncMock())

    select_result = MagicMock()
    select_result.all.return_value = []

    mock_session.execute = AsyncMock(return_value=select_result)

    mock_session_maker = MagicMock(return_value=mock_session)

    manager = EpisodicMemoryManager()
    manager.session_maker = mock_session_maker

    with patch("mirix.database.redis_client.get_redis_client", return_value=None):
        deleted = await manager.delete_by_client_id(actor)

    assert deleted == 0
    mock_session_maker.assert_called_once_with()
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_semantic_delete_by_client_id_uses_session_maker_sql_path():
    actor = _actor()
    mock_session = _async_session_context(AsyncMock())

    select_result = MagicMock()
    select_result.all.return_value = [("sem-1",)]
    delete_result = MagicMock()

    mock_session.execute = AsyncMock(side_effect=[select_result, delete_result])

    mock_session_maker = MagicMock(return_value=mock_session)

    manager = SemanticMemoryManager()
    manager.session_maker = mock_session_maker

    with patch("mirix.database.redis_client.get_redis_client", return_value=None):
        deleted = await manager.delete_by_client_id(actor)

    assert deleted == 1
    mock_session_maker.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()
    assert mock_session.execute.await_count == 2
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_delete_by_client_id_returns_zero_without_delete_query():
    actor = _actor()
    mock_session = _async_session_context(AsyncMock())

    select_result = MagicMock()
    select_result.all.return_value = []

    mock_session.execute = AsyncMock(return_value=select_result)

    mock_session_maker = MagicMock(return_value=mock_session)

    manager = SemanticMemoryManager()
    manager.session_maker = mock_session_maker

    with patch("mirix.database.redis_client.get_redis_client", return_value=None):
        deleted = await manager.delete_by_client_id(actor)

    assert deleted == 0
    mock_session_maker.assert_called_once_with()
    mock_session.execute.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
