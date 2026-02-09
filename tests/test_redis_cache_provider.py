"""
Redis cache provider tests for Mirix.

Tests that RedisCacheProvider correctly delegates to RedisMemoryClient
and handles errors gracefully (returns None/False).

Usage:
    pytest tests/test_redis_cache_provider.py -v
"""

from unittest.mock import Mock

import pytest

from mirix.database.redis_cache_provider import RedisCacheProvider


@pytest.fixture
def mock_redis_client():
    """Mock RedisMemoryClient with standard methods."""
    client = Mock()
    client.get_json.return_value = {"test": "data"}
    client.set_json.return_value = True
    client.get_hash.return_value = {"hash": "data"}
    client.set_hash.return_value = True
    client.delete.return_value = True
    return client


def test_redis_provider_get_json(mock_redis_client):
    """get_json delegates to client and returns data."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_json("test_key")
    assert result == {"test": "data"}
    mock_redis_client.get_json.assert_called_once_with("test_key")


def test_redis_provider_set_json(mock_redis_client):
    """set_json delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_json("test_key", {"k": "v"}, ttl=300)
    assert result is True
    mock_redis_client.set_json.assert_called_once_with(
        "test_key", {"k": "v"}, ttl=300
    )


def test_redis_provider_get_hash(mock_redis_client):
    """get_hash delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_hash("test_key")
    assert result == {"hash": "data"}
    mock_redis_client.get_hash.assert_called_once_with("test_key")


def test_redis_provider_set_hash(mock_redis_client):
    """set_hash delegates to client with ttl."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_hash("test_key", {"a": 1}, ttl=60)
    assert result is True
    mock_redis_client.set_hash.assert_called_once_with(
        "test_key", {"a": 1}, ttl=60
    )


def test_redis_provider_delete(mock_redis_client):
    """delete delegates to client."""
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.delete("test_key")
    assert result is True
    mock_redis_client.delete.assert_called_once_with("test_key")


def test_redis_provider_get_returns_none_on_error(mock_redis_client):
    """get_json returns None when client raises."""
    mock_redis_client.get_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.get_json("test_key")
    assert result is None


def test_redis_provider_set_returns_false_on_error(mock_redis_client):
    """set_json returns False when client raises."""
    mock_redis_client.set_json.side_effect = Exception("Redis error")
    provider = RedisCacheProvider(mock_redis_client)
    result = provider.set_json("test_key", {})
    assert result is False


def test_redis_provider_has_prefix_constants():
    """Provider exposes same key prefixes as RedisMemoryClient."""
    client = Mock()
    provider = RedisCacheProvider(client)
    assert provider.MESSAGE_PREFIX == "msg:"
    assert provider.BLOCK_PREFIX == "block:"
    assert provider.RAW_MEMORY_PREFIX == "raw_memory:"
