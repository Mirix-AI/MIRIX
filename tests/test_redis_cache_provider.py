"""
Redis cache provider tests for Mirix.

Tests that RedisCacheProvider correctly delegates to RedisMemoryClient
and handles errors gracefully (returns None/False).
Uses async tests and await to match Redis native async API.

Usage:
    pytest tests/test_redis_cache_provider.py -v
"""

from unittest.mock import AsyncMock, Mock

import pytest

from mirix.database.redis_cache_provider import RedisCacheProvider


@pytest.fixture
def mock_redis_client():
    """Mock RedisMemoryClient with async methods returning awaitables."""
    client = Mock()
    client.get_json = AsyncMock(return_value={"test": "data"})
    client.set_json = AsyncMock(return_value=True)
    client.get_hash = AsyncMock(return_value={"hash": "data"})
    client.set_hash = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_redis_provider_get_json(mock_redis_client):
    """get_json delegates to client and returns data."""
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.get_json("test_key")
    assert result == {"test": "data"}
    mock_redis_client.get_json.assert_called_once_with("test_key")


@pytest.mark.asyncio
async def test_redis_provider_set_json(mock_redis_client):
    """set_json delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.set_json("test_key", {"k": "v"}, ttl=300)
    assert result is True
    mock_redis_client.set_json.assert_called_once_with("test_key", {"k": "v"}, ttl=300)


@pytest.mark.asyncio
async def test_redis_provider_get_hash(mock_redis_client):
    """get_hash delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.get_hash("test_key")
    assert result == {"hash": "data"}
    mock_redis_client.get_hash.assert_called_once_with("test_key")


@pytest.mark.asyncio
async def test_redis_provider_set_hash(mock_redis_client):
    """set_hash delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.set_hash("test_key", {"a": 1}, ttl=60)
    assert result is True
    mock_redis_client.set_hash.assert_called_once_with("test_key", {"a": 1}, ttl=60)


@pytest.mark.asyncio
async def test_redis_provider_delete(mock_redis_client):
    """delete delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.delete("test_key")
    assert result is True
    mock_redis_client.delete.assert_called_once_with("test_key")


@pytest.mark.asyncio
async def test_redis_provider_get_returns_none_on_error(mock_redis_client):
    """get_json returns None when client raises."""
    mock_redis_client.get_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.get_json("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_redis_provider_set_returns_false_on_error(mock_redis_client):
    """set_json returns False when client raises."""
    mock_redis_client.set_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = await provider.set_json("test_key", {})
    assert result is False


def test_redis_provider_has_prefix_constants():
    """Provider exposes same key prefixes as RedisMemoryClient."""
    client = Mock()
    provider = RedisCacheProvider(client)
    assert provider.MESSAGE_PREFIX == "msg:"
    assert provider.BLOCK_PREFIX == "block:"
    assert provider.RAW_MEMORY_PREFIX == "raw_memory:"
