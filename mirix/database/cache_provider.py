"""
Cache provider interface and registry for Mirix.

Cache providers implement the interface via duck typing (no base class
required). Similar to the auth_provider pattern in mirix.llm_api.auth_provider.

Expected methods (duck typing):
    - get(key: str) -> Optional[Dict[str, Any]]
    - set(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - delete(key: str) -> bool
    - get_hash(key: str) -> Optional[Dict[str, Any]]
    - set_hash(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - get_json(key: str) -> Optional[Dict[str, Any]]
    - set_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool

Key prefix constants (all providers should define these):
    BLOCK_PREFIX, MESSAGE_PREFIX, EPISODIC_PREFIX, SEMANTIC_PREFIX,
    PROCEDURAL_PREFIX, RESOURCE_PREFIX, KNOWLEDGE_PREFIX, RAW_MEMORY_PREFIX,
    ORGANIZATION_PREFIX, USER_PREFIX, CLIENT_PREFIX, AGENT_PREFIX, TOOL_PREFIX

Usage:
    # In external project (e.g., ECMS)
    from mirix.database.cache_provider import register_cache_provider
    cache_provider = MyCustomCacheProvider(config)
    register_cache_provider("my_cache", cache_provider)

    # In Mirix service managers
    from mirix.database.cache_provider import get_cache_provider
    cache_provider = get_cache_provider()
    if cache_provider:
        data = cache_provider.get_hash(f"{cache_provider.MESSAGE_PREFIX}{msg_id}")
"""

from typing import Any, Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global cache provider registry (simple dictionary)
_cache_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None


def register_cache_provider(name: str, provider: Any) -> None:
    """
    Register a cache provider with Mirix.

    Similar to register_auth_provider() pattern. Last registered provider
    becomes the active one.

    Args:
        name: Provider identifier (e.g., "redis", "ips_cache").
        provider: Provider instance implementing the cache interface.
    """
    global _cache_providers, _active_provider_name

    _cache_providers[name] = provider
    _active_provider_name = name
    logger.info("Registered cache provider: %s", name)


def get_cache_provider() -> Optional[Any]:
    """
    Get the active cache provider.

    Returns None if no provider is registered (graceful fallback to PostgreSQL).

    Returns:
        Cache provider instance or None.
    """
    if _active_provider_name and _active_provider_name in _cache_providers:
        return _cache_providers[_active_provider_name]
    return None


def unregister_cache_provider(name: str) -> None:
    """
    Unregister a cache provider.

    Args:
        name: Provider identifier.
    """
    global _cache_providers, _active_provider_name

    if name in _cache_providers:
        del _cache_providers[name]
        if _active_provider_name == name:
            _active_provider_name = None
        logger.info("Unregistered cache provider: %s", name)


def get_registered_providers() -> Dict[str, Any]:
    """
    Get all registered cache providers (for tests).

    Returns:
        Dictionary of provider_name -> provider_instance.
    """
    return dict(_cache_providers)
