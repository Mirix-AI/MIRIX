"""
Cache provider registry tests for Mirix.

Tests the cache provider registration and retrieval used by service managers.
Cache strategy (Redis vs IPS Cache) is selected by the host (e.g. ECMS);
Mirix only provides the registry and Redis provider.

Usage:
    pytest tests/test_cache_provider.py -v
"""

import pytest

from mirix.database.cache_provider import (
    get_cache_provider,
    get_registered_providers,
    register_cache_provider,
    unregister_cache_provider,
)


class MockCacheProvider:
    """Mock cache provider for testing (duck typing)."""

    MESSAGE_PREFIX = "msg:"
    BLOCK_PREFIX = "block:"

    def get(self, key: str):
        return {"mock": "data"}

    def set(self, key: str, data: dict, ttl: int = None):
        return True

    def delete(self, key: str):
        return True

    def get_hash(self, key: str):
        return {"mock": "data"}

    def set_hash(self, key: str, data: dict, ttl: int = None):
        return True

    def get_json(self, key: str):
        return {"mock": "data"}

    def set_json(self, key: str, data: dict, ttl: int = None):
        return True


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up cache provider registry before and after each test."""
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)
    yield
    for name in list(get_registered_providers().keys()):
        unregister_cache_provider(name)


def test_register_and_get_cache_provider():
    """Registering a provider makes it the active one."""
    provider = MockCacheProvider()
    register_cache_provider("mock", provider)

    retrieved = get_cache_provider()
    assert retrieved is provider


def test_get_cache_provider_none_when_empty():
    """get_cache_provider returns None when nothing is registered."""
    assert get_cache_provider() is None


def test_unregister_cache_provider():
    """Unregistering removes the provider; active becomes None."""
    provider = MockCacheProvider()
    register_cache_provider("mock", provider)
    unregister_cache_provider("mock")

    assert get_cache_provider() is None


def test_last_registered_is_active():
    """Last registered provider becomes the active one."""
    p1 = MockCacheProvider()
    p2 = MockCacheProvider()
    register_cache_provider("first", p1)
    register_cache_provider("second", p2)

    assert get_cache_provider() is p2


def test_get_registered_providers():
    """get_registered_providers returns a copy of the registry."""
    provider = MockCacheProvider()
    register_cache_provider("mock", provider)

    reg = get_registered_providers()
    assert reg == {"mock": provider}
    assert "mock" in reg
