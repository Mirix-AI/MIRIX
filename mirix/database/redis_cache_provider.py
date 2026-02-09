"""
Redis cache provider for Mirix.

Wraps the existing RedisMemoryClient to implement the cache provider interface.
Used when Mirix runs standalone with Redis; ECMS can register IPS Cache instead.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from mirix.log import get_logger

if TYPE_CHECKING:
    from mirix.database.redis_client import RedisMemoryClient

logger = get_logger(__name__)


class RedisCacheProvider:
    """
    Redis cache provider implementation.

    Wraps RedisMemoryClient to provide the cache provider interface used by
    service managers and ORM. All operations delegate to the Redis client;
    errors are logged and return None/False for graceful fallback.
    """

    # Key prefixes (must match RedisMemoryClient for key compatibility)
    BLOCK_PREFIX = "block:"
    MESSAGE_PREFIX = "msg:"
    EPISODIC_PREFIX = "episodic:"
    SEMANTIC_PREFIX = "semantic:"
    PROCEDURAL_PREFIX = "procedural:"
    RESOURCE_PREFIX = "resource:"
    KNOWLEDGE_PREFIX = "knowledge:"
    RAW_MEMORY_PREFIX = "raw_memory:"
    ORGANIZATION_PREFIX = "org:"
    USER_PREFIX = "user:"
    CLIENT_PREFIX = "client:"
    AGENT_PREFIX = "agent:"
    TOOL_PREFIX = "tool:"

    def __init__(self, redis_client: "RedisMemoryClient") -> None:
        """
        Initialize Redis cache provider.

        Args:
            redis_client: Existing RedisMemoryClient instance.
        """
        self.redis_client = redis_client
        logger.info("Initialized RedisCacheProvider")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value (string/JSON) from Redis."""
        try:
            return self.redis_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get failed for key %s: %s", key, e)
            return None

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set value (JSON) in Redis."""
        try:
            return self.redis_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set failed for key %s: %s", key, e)
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return self.redis_client.delete(key)
        except Exception as e:
            logger.warning("Redis delete failed for key %s: %s", key, e)
            return False

    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """Get hash from Redis."""
        try:
            return self.redis_client.get_hash(key)
        except Exception as e:
            logger.warning("Redis get_hash failed for key %s: %s", key, e)
            return None

    def set_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set hash in Redis."""
        try:
            return self.redis_client.set_hash(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_hash failed for key %s: %s", key, e)
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON from Redis."""
        try:
            return self.redis_client.get_json(key)
        except Exception as e:
            logger.warning("Redis get_json failed for key %s: %s", key, e)
            return None

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON in Redis."""
        try:
            return self.redis_client.set_json(key, data, ttl=ttl)
        except Exception as e:
            logger.warning("Redis set_json failed for key %s: %s", key, e)
            return False
