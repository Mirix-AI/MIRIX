"""
Hybrid Redis client for Mirix memory system.

Uses:
- Redis Hash for blocks (fast, flat structure, no embeddings)
- Redis Hash for messages (fast, mostly flat, no embeddings)
- Redis JSON with Vector fields for memory tables (embeddings support)

Provides:
- 40-60% faster operations for blocks and messages via Hash
- 10-40x faster vector similarity search vs PostgreSQL pgvector
- Hybrid text+vector search capabilities
"""

import json
from typing import Any, Dict, List, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global Redis client instance
_redis_client: Optional["RedisMemoryClient"] = None


class RedisMemoryClient:
    """
    Hybrid Redis client for Mirix memory caching and search.

    Architecture:
    - Hash: blocks, messages (no embeddings, flat structure)
    - JSON + Vector: episodic, semantic, procedural, resource, knowledge (has embeddings)
    """

    # Index names
    BLOCK_INDEX = "idx:blocks"
    MESSAGE_INDEX = "idx:messages"
    EPISODIC_INDEX = "idx:episodic_memory"
    SEMANTIC_INDEX = "idx:semantic_memory"
    PROCEDURAL_INDEX = "idx:procedural_memory"
    RESOURCE_INDEX = "idx:resource_memory"
    KNOWLEDGE_INDEX = "idx:knowledge_vault"
    ORGANIZATION_INDEX = "idx:organizations"
    USER_INDEX = "idx:users"
    AGENT_INDEX = "idx:agents"
    TOOL_INDEX = "idx:tools"

    # Key prefixes
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

    def __init__(
        self,
        redis_uri: str,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        socket_keepalive: bool = True,
        retry_on_timeout: bool = True,
    ):
        """
        Initialize Redis client with optimized connection pool.

        Args:
            redis_uri: Redis connection URI
            max_connections: Maximum connections per container (default: 50)
            socket_timeout: Socket timeout in seconds (default: 5)
            socket_connect_timeout: Socket connect timeout in seconds (default: 5)
            socket_keepalive: Enable TCP keepalive (default: True)
            retry_on_timeout: Retry on timeout errors (default: True)
        """
        try:
            import socket

            from redis import ConnectionPool, Redis

            self.redis_uri = redis_uri

            # Socket keepalive options (prevent stale connections)
            socket_keepalive_options = {}
            if socket_keepalive and hasattr(socket, "TCP_KEEPIDLE"):
                # TCP_KEEPIDLE may not be available on all platforms (e.g., Windows)
                socket_keepalive_options = {
                    socket.TCP_KEEPIDLE: 60,  # Start keepalive after 60s idle
                    socket.TCP_KEEPINTVL: 10,  # Send keepalive every 10s
                    socket.TCP_KEEPCNT: 3,  # Close after 3 failed keepalives
                }

            # Create connection pool with optimized settings
            self.pool = ConnectionPool.from_url(
                redis_uri,
                max_connections=max_connections,  # Bounded connection count
                socket_timeout=socket_timeout,  # Read/write timeout
                socket_connect_timeout=socket_connect_timeout,  # Connection timeout
                socket_keepalive=socket_keepalive,  # Enable TCP keepalive
                socket_keepalive_options=socket_keepalive_options,
                retry_on_timeout=retry_on_timeout,  # Retry on timeouts
                decode_responses=True,  # Decode bytes to strings
                health_check_interval=30,  # Check connection health every 30s
            )

            self.client = Redis(connection_pool=self.pool)

            # Log configuration
            logger.info(
                "Redis connection pool initialized: %s (max_connections=%d, " "socket_timeout=%ds, keepalive=%s)",
                self._mask_uri(redis_uri),
                max_connections,
                socket_timeout,
                socket_keepalive,
            )
        except ImportError:
            logger.error("Redis library not installed. Install with: pip install redis[hiredis]")
            raise
        except Exception as e:
            logger.error("Failed to initialize Redis client: %s", e)
            raise

    def _mask_uri(self, uri: str) -> str:
        """Mask password in URI for logging."""
        if "@" in uri and ":" in uri:
            parts = uri.split("@")
            if len(parts) == 2:
                protocol = parts[0].split("://")[0]
                return f"{protocol}://****@{parts[1]}"
        return uri

    def ping(self) -> bool:
        """Test Redis connection."""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error("Redis ping failed: %s", e)
            return False

    def close(self) -> None:
        """Close Redis connection pool."""
        try:
            if self.pool:
                self.pool.disconnect()
                logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error("Error closing Redis pool: %s", e)

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get Redis connection pool information for monitoring.

        Returns:
            Dictionary with connection pool stats and Redis server info
        """
        try:
            # Get Redis server info
            info = self.client.info("clients")
            server_info = self.client.info("server")

            # Get connection pool stats
            pool_info = {
                # Client connections (this container)
                "pool_max_connections": self.pool.max_connections,
                "pool_connection_kwargs": {
                    "socket_timeout": self.pool.connection_kwargs.get("socket_timeout"),
                    "socket_keepalive": self.pool.connection_kwargs.get("socket_keepalive"),
                },
                # Redis server stats (global)
                "server_connected_clients": info.get("connected_clients", 0),
                "server_max_clients": server_info.get("maxclients", 10000),
                "server_version": server_info.get("redis_version", "unknown"),
                # Calculate usage percentage
                "usage_percent": (info.get("connected_clients", 0) / server_info.get("maxclients", 10000)) * 100,
            }

            return pool_info
        except Exception as e:
            logger.error("Failed to get Redis connection info: %s", e)
            return {}

    def log_connection_stats(self) -> None:
        """Log current Redis connection statistics."""
        try:
            info = self.get_connection_info()
            if info:
                logger.info(
                    "Redis connections: %d/%d (%.1f%%) | Pool max: %d | Version: %s",
                    info.get("server_connected_clients", 0),
                    info.get("server_max_clients", 10000),
                    info.get("usage_percent", 0),
                    info.get("pool_max_connections", 50),
                    info.get("server_version", "unknown"),
                )

                # Warn if usage is high
                if info.get("usage_percent", 0) > 80:
                    logger.warning(
                        "Redis connection usage high: %.1f%% (%d/%d)",
                        info.get("usage_percent", 0),
                        info.get("server_connected_clients", 0),
                        info.get("server_max_clients", 10000),
                    )
        except Exception as e:
            logger.error("Failed to log Redis connection stats: %s", e)

    def create_indexes(self) -> None:
        """Create RediSearch indexes for all memory types (hybrid approach)."""
        logger.info(
            "Creating Redis indexes (hybrid: Hash for blocks/messages/orgs/users/agents/tools, JSON+Vectors for memory)..."
        )

        try:
            # Hash-based indexes (no embeddings)
            self._create_block_index()
            self._create_message_index()
            self._create_organization_index()
            self._create_user_index()
            self._create_agent_index()
            self._create_tool_index()

            # JSON-based indexes with vector fields (has embeddings)
            self._create_episodic_index()
            self._create_semantic_index()
            self._create_procedural_index()
            self._create_resource_index()
            self._create_knowledge_index()

            logger.info("All Redis indexes created successfully")
        except Exception as e:
            logger.error("Failed to create some indexes: %s", e)
            # Don't raise - allow system to continue without indexes

    # ========================================================================
    # HASH-BASED METHODS (for blocks and messages - NO embeddings)
    # ========================================================================

    def _create_block_index(self) -> None:
        """Create HASH-based index for blocks (Core Memory)."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.BLOCK_INDEX).info()
                logger.debug("Index %s already exists", self.BLOCK_INDEX)
                return
            except:
                pass

            schema = (
                TextField("organization_id"),
                TextField("user_id"),
                TextField("agent_id"),
                TagField("label"),  # Exact match: "human", "persona"
                TextField("value"),  # Full-text search on content
                NumericField("limit"),
                NumericField("created_at_ts"),
            )

            self.client.ft(self.BLOCK_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.BLOCK_PREFIX], index_type=IndexType.HASH  # Hash type for simple data
                ),
            )
            logger.info("Created HASH index: %s", self.BLOCK_INDEX)

        except Exception as e:
            logger.warning("Failed to create block index: %s", e)

    def _create_message_index(self) -> None:
        """Create HASH-based index for messages."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.MESSAGE_INDEX).info()
                logger.debug("Index %s already exists", self.MESSAGE_INDEX)
                return
            except:
                pass

            schema = (
                TextField("organization_id"),
                TextField("agent_id"),
                TextField("user_id"),
                TagField("role"),  # user, assistant, system, tool
                TextField("text"),  # Message text
                TextField("model"),
                NumericField("created_at_ts"),
                TextField("filter_tags"),  # Filter tags (stored as JSON string in HASH)
            )

            self.client.ft(self.MESSAGE_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.MESSAGE_PREFIX], index_type=IndexType.HASH)  # Hash type
            )
            logger.info("Created HASH index: %s", self.MESSAGE_INDEX)

        except Exception as e:
            logger.warning("Failed to create message index: %s", e)

    def _create_organization_index(self) -> None:
        """Create HASH-based index for organizations."""
        try:
            from redis.commands.search.field import NumericField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.ORGANIZATION_INDEX).info()
                logger.debug("Index %s already exists", self.ORGANIZATION_INDEX)
                return
            except:
                pass

            schema = (
                TextField("id"),
                TextField("name"),
                NumericField("created_at_ts"),
            )

            self.client.ft(self.ORGANIZATION_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.ORGANIZATION_PREFIX], index_type=IndexType.HASH)
            )
            logger.info("Created HASH index: %s", self.ORGANIZATION_INDEX)

        except Exception as e:
            logger.warning("Failed to create organization index: %s", e)

    def _create_user_index(self) -> None:
        """Create HASH-based index for users."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.USER_INDEX).info()
                logger.debug("Index %s already exists", self.USER_INDEX)
                return
            except:
                pass

            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("status"),  # active/inactive
                TextField("timezone"),
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )

            self.client.ft(self.USER_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.USER_PREFIX], index_type=IndexType.HASH)
            )
            logger.info("Created HASH index: %s", self.USER_INDEX)

        except Exception as e:
            logger.warning("Failed to create user index: %s", e)

    def _create_agent_index(self) -> None:
        """Create HASH-based index for agents (with denormalized tool_ids)."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.AGENT_INDEX).info()
                logger.debug("Index %s already exists", self.AGENT_INDEX)
                return
            except:
                pass

            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("agent_type"),
                TextField("description"),
                TextField("parent_id"),
                TextField("system"),  # System prompt
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )

            self.client.ft(self.AGENT_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.AGENT_PREFIX], index_type=IndexType.HASH)
            )
            logger.info("Created HASH index: %s", self.AGENT_INDEX)

        except Exception as e:
            logger.warning("Failed to create agent index: %s", e)

    def _create_tool_index(self) -> None:
        """Create HASH-based index for tools."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            try:
                self.client.ft(self.TOOL_INDEX).info()
                logger.debug("Index %s already exists", self.TOOL_INDEX)
                return
            except:
                pass

            schema = (
                TextField("id"),
                TextField("organization_id"),
                TextField("name"),
                TagField("tool_type"),  # CORE, CUSTOM, etc.
                TextField("description"),
                TagField("tags", separator=","),
                NumericField("return_char_limit"),
                NumericField("created_at_ts"),
                NumericField("updated_at_ts"),
                TagField("is_deleted"),
            )

            self.client.ft(self.TOOL_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.TOOL_PREFIX], index_type=IndexType.HASH)
            )
            logger.info("Created HASH index: %s", self.TOOL_INDEX)

        except Exception as e:
            logger.warning("Failed to create tool index: %s", e)

    def set_hash(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store data as Redis Hash (for flat structures like blocks and messages).

        Args:
            key: Redis key
            data: Data to store (will be flattened)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            # Flatten and convert all values to strings
            flattened = self._flatten_dict(data)

            # HSET creates/updates all fields atomically
            self.client.hset(key, mapping=flattened)

            if ttl:
                self.client.expire(key, ttl)

            logger.debug("Stored Hash: %s (%d fields)", key, len(flattened))
            return True
        except Exception as e:
            logger.error("Failed to set hash for %s: %s", key, e)
            return False

    def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from Redis Hash.

        Args:
            key: Redis key

        Returns:
            Data dictionary or None if not found
        """
        try:
            data = self.client.hgetall(key)
            if not data:
                return None

            # Convert back to proper types
            result = self._unflatten_dict(data)
            logger.debug("Retrieved Hash: %s", key)
            return result
        except Exception as e:
            logger.error("Failed to get hash for %s: %s", key, e)
            return None

    def update_hash_field(self, key: str, field: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Update a single field in Redis Hash (very fast for partial updates!).

        Args:
            key: Redis key
            field: Field name to update
            value: New value
            ttl: Optional TTL reset

        Returns:
            True if successful
        """
        try:
            self.client.hset(key, field, str(value))
            if ttl:
                self.client.expire(key, ttl)
            logger.debug("Updated Hash field: %s.%s", key, field)
            return True
        except Exception as e:
            logger.error("Failed to update hash field %s in %s: %s", field, key, e)
            return False

    def _flatten_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Flatten dictionary for Hash storage. All values converted to strings."""
        flattened = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Store nested dicts as JSON strings
                flattened[key] = json.dumps(value)
            elif isinstance(value, (list, tuple)):
                # Store lists as JSON strings
                flattened[key] = json.dumps(value)
            elif value is None:
                flattened[key] = ""
            elif isinstance(value, bool):
                flattened[key] = "true" if value else "false"
            else:
                flattened[key] = str(value)
        return flattened

    def _unflatten_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Convert Hash data back to proper Python types."""
        result = {}
        for key, value in data.items():
            # Parse numeric fields (e.g., 'limit')
            if key in ("limit",):
                try:
                    result[key] = float(value) if "." in value else int(value)
                    continue
                except (ValueError, AttributeError):
                    pass

            # Parse boolean
            if value.lower() in ("true", "false"):
                result[key] = value.lower() == "true"
                continue

            # Try to parse JSON (for nested structures)
            if value and (value.startswith("{") or value.startswith("[")):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass

            # Keep as string
            result[key] = value if value else None

        return result

    # ========================================================================
    # JSON-BASED METHODS (for memory types with embeddings)
    # ========================================================================

    def _create_episodic_index(self) -> None:
        """Create JSON-based index for episodic memory with 2 VECTOR fields."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            from mirix.constants import MAX_EMBEDDING_DIM

            try:
                self.client.ft(self.EPISODIC_INDEX).info()
                logger.debug("Index %s already exists", self.EPISODIC_INDEX)
                return
            except:
                pass

            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.actor", as_name="actor"),
                TextField("$.event_type", as_name="event_type"),
                TextField("$.summary", as_name="summary"),
                TextField("$.details", as_name="details"),
                NumericField("$.occurred_at_ts", as_name="occurred_at_ts"),
                TagField("$.user_id", as_name="user_id"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),  # Explicit scope field for fast filtering
                TextField("$.filter_tags.*", as_name="filter_tags"),  # Filter tags for flexible filtering
                # Vector fields for embeddings (32KB total)
                VectorField(
                    "$.details_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="details_embedding",
                ),
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="summary_embedding",
                ),
            )

            self.client.ft(self.EPISODIC_INDEX).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.EPISODIC_PREFIX], index_type=IndexType.JSON  # JSON type for complex data
                ),
            )
            logger.info("Created JSON+VECTOR index: %s (2 vectors)", self.EPISODIC_INDEX)

        except Exception as e:
            logger.warning("Failed to create episodic index: %s", e)

    def _create_semantic_index(self) -> None:
        """Create JSON-based index for semantic memory with 3 VECTOR fields."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            from mirix.constants import MAX_EMBEDDING_DIM

            try:
                self.client.ft(self.SEMANTIC_INDEX).info()
                logger.debug("Index %s already exists", self.SEMANTIC_INDEX)
                return
            except:
                pass

            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.name", as_name="name"),
                TextField("$.summary", as_name="summary"),
                TextField("$.details", as_name="details"),
                TextField("$.source", as_name="source"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),  # Explicit scope field
                TextField("$.filter_tags.*", as_name="filter_tags"),  # Filter tags for flexible filtering
                # Three vector fields for comprehensive search (48KB total!)
                VectorField(
                    "$.name_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="name_embedding",
                ),
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="summary_embedding",
                ),
                VectorField(
                    "$.details_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="details_embedding",
                ),
            )

            self.client.ft(self.SEMANTIC_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.SEMANTIC_PREFIX], index_type=IndexType.JSON)
            )
            logger.info("Created JSON+VECTOR index: %s (3 vectors, 48KB!)", self.SEMANTIC_INDEX)

        except Exception as e:
            logger.warning("Failed to create semantic index: %s", e)

    def _create_procedural_index(self) -> None:
        """Create JSON-based index for procedural memory with 2 VECTOR fields."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            from mirix.constants import MAX_EMBEDDING_DIM

            try:
                self.client.ft(self.PROCEDURAL_INDEX).info()
                logger.debug("Index %s already exists", self.PROCEDURAL_INDEX)
                return
            except:
                pass

            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.entry_type", as_name="entry_type"),
                TextField("$.summary", as_name="summary"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),  # Explicit scope field
                TextField("$.filter_tags.*", as_name="filter_tags"),  # Filter tags for flexible filtering
                # Two vector fields (32KB total)
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="summary_embedding",
                ),
                VectorField(
                    "$.steps_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="steps_embedding",
                ),
            )

            self.client.ft(self.PROCEDURAL_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.PROCEDURAL_PREFIX], index_type=IndexType.JSON)
            )
            logger.info("Created JSON+VECTOR index: %s (2 vectors)", self.PROCEDURAL_INDEX)

        except Exception as e:
            logger.warning("Failed to create procedural index: %s", e)

    def _create_resource_index(self) -> None:
        """Create JSON-based index for resource memory with 1 VECTOR field."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            from mirix.constants import MAX_EMBEDDING_DIM

            try:
                self.client.ft(self.RESOURCE_INDEX).info()
                logger.debug("Index %s already exists", self.RESOURCE_INDEX)
                return
            except:
                pass

            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.title", as_name="title"),
                TextField("$.summary", as_name="summary"),
                TextField("$.resource_type", as_name="resource_type"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),  # Explicit scope field
                TextField("$.filter_tags.*", as_name="filter_tags"),  # Filter tags
                # One vector field (16KB)
                VectorField(
                    "$.summary_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="summary_embedding",
                ),
            )

            self.client.ft(self.RESOURCE_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.RESOURCE_PREFIX], index_type=IndexType.JSON)
            )
            logger.info("Created JSON+VECTOR index: %s (1 vector)", self.RESOURCE_INDEX)

        except Exception as e:
            logger.warning("Failed to create resource index: %s", e)

    def _create_knowledge_index(self) -> None:
        """Create JSON-based index for knowledge vault with 1 VECTOR field."""
        try:
            from redis.commands.search.field import NumericField, TagField, TextField, VectorField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            from mirix.constants import MAX_EMBEDDING_DIM

            try:
                self.client.ft(self.KNOWLEDGE_INDEX).info()
                logger.debug("Index %s already exists", self.KNOWLEDGE_INDEX)
                return
            except:
                pass

            schema = (
                TextField("$.organization_id", as_name="organization_id"),
                TextField("$.agent_id", as_name="agent_id"),
                TextField("$.caption", as_name="caption"),
                TagField("$.user_id", as_name="user_id"),
                NumericField("$.created_at_ts", as_name="created_at_ts"),
                TagField("$.filter_tags.scope", as_name="filter_tags_scope"),  # Explicit scope field
                TextField("$.filter_tags.*", as_name="filter_tags"),  # Filter tags
                # One vector field (16KB)
                VectorField(
                    "$.caption_embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": MAX_EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    as_name="caption_embedding",
                ),
            )

            self.client.ft(self.KNOWLEDGE_INDEX).create_index(
                schema, definition=IndexDefinition(prefix=[self.KNOWLEDGE_PREFIX], index_type=IndexType.JSON)
            )
            logger.info("Created JSON+VECTOR index: %s (1 vector)", self.KNOWLEDGE_INDEX)

        except Exception as e:
            logger.warning("Failed to create knowledge index: %s", e)

    def set_json(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Store data as Redis JSON (for complex structures with embeddings).

        Args:
            key: Redis key
            data: Data to store (supports nested structures)
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            # Use JSON.SET command
            self.client.json().set(key, "$", data)

            if ttl:
                self.client.expire(key, ttl)

            logger.debug("Stored JSON: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to set JSON for %s: %s", key, e)
            return False

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from Redis JSON.

        Args:
            key: Redis key

        Returns:
            Data dictionary or None if not found
        """
        try:
            data = self.client.json().get(key)
            if data is None:
                return None

            logger.debug("Retrieved JSON: %s", key)
            return data
        except Exception as e:
            logger.error("Failed to get JSON for %s: %s", key, e)
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            self.client.delete(key)
            logger.debug("Deleted key: %s", key)
            return True
        except Exception as e:
            logger.error("Failed to delete key %s: %s", key, e)
            return False

    def _build_filter_tags_query(self, filter_tags: Dict[str, Any]) -> str:
        """
        Build Redis Search query string from filter_tags dictionary.

        According to MEMORY_METADATA_IMPLEMENTATION_GUIDE.md,
        filter_tags are indexed as TAG fields with names like:
        - $.filter_tags.expert_id -> indexed as "filter_tags_expert_id"
        - $.filter_tags.scope -> indexed as "filter_tags_scope"

        Args:
            filter_tags: Dictionary of filter tag key-value pairs
                        e.g., {"expert_id": "expert-123", "scope": "read"}

        Returns:
            Redis Search query string for filter_tags filters
            e.g., "@filter_tags_expert_id:{expert\\-123} @filter_tags_scope:{read}"

        Example:
            filter_tags = {"expert_id": "expert-123", "scope": "read"}
            # Returns: "@filter_tags_expert_id:{expert\\-123} @filter_tags_scope:{read}"
        """
        if not filter_tags:
            return ""

        import re

        def escape_tag_value(value: str) -> str:
            """Escape special characters for Redis TAG field values."""
            # TAG fields need escaping for: - : . ( ) { } [ ] " ' , < > ; ! @ # $ % ^ & * + = ~
            special_chars = r'[\-:.()\[\]{}"\',<>;!@#$%^&*+=~]'
            return re.sub(special_chars, lambda m: f"\\{m.group(0)}", str(value))

        query_parts = []
        for key, value in filter_tags.items():
            # Convert filter_tags.expert_id -> filter_tags_expert_id
            field_name = f"filter_tags_{key}"
            escaped_value = escape_tag_value(value)
            query_parts.append(f"@{field_name}:{{{escaped_value}}}")

        return " ".join(query_parts)

    def search_text(
        self,
        index_name: str,
        query: str,
        search_fields: List[str],
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search using RediSearch.

        Args:
            index_name: Index to search (e.g., "idx:episodic_memory")
            query: Search query text
            search_fields: Fields to search in (e.g., ["summary", "details"])
            limit: Maximum number of results
            user_id: Filter by user_id (optional)
            organization_id: Filter by organization_id (optional)
            return_fields: Specific fields to return (None = return all)
            filter_tags: Optional filter tags for additional filtering
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of matching documents with BM25-like scores

        Example:
            results = redis_client.search_text(
                index_name="idx:episodic_memory",
                query="meeting Sarah",
                search_fields=["details"],
                limit=10,
                user_id="user-123",
                organization_id="org-456",
                filter_tags={"expert_id": "expert-123"},
                start_date=datetime(2025, 11, 19),
                end_date=datetime(2025, 11, 19, 23, 59, 59)
            )
        """
        try:
            import re
            from datetime import datetime

            from redis.commands.search.query import Query

            # Escape special characters in query for Redis Search
            # Redis Search special characters: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( ) - + = ~
            def escape_redis_query(text: str) -> str:
                """Escape special characters for Redis Search query."""
                special_chars = r'[,.<>{}[\]"\':;!@#$%^&*()\-+=~]'
                return re.sub(special_chars, lambda m: f"\\{m.group(0)}", text)

            escaped_query = escape_redis_query(query)

            # Build query parts
            query_parts = []

            # Add user_id filter if provided (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@user_id:{{{escaped_user_id}}}")

            # Add organization_id filter (TEXT field)
            if organization_id:
                escaped_org_id = organization_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering (numeric range query on occurred_at_ts)
            if start_date or end_date:
                # Convert datetime to Unix timestamp
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                query_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")
                logger.debug("🕐 Redis temporal filter: @occurred_at_ts:[%s %s]", min_ts, max_ts)

            # Add filter_tags filters
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    query_parts.append(filter_query)

            # Build field search query
            if len(search_fields) == 1:
                text_query = f"@{search_fields[0]}:({escaped_query})"
            else:
                field_query = "|".join(search_fields)
                text_query = f"@{field_query}:({escaped_query})"

            query_parts.append(text_query)

            # Combine query parts
            search_query = " ".join(query_parts)

            # Build Query object
            query_obj = Query(search_query).paging(0, limit)

            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields)

            # Execute search
            results = self.client.ft(index_name).search(query_obj)

            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, "json"):
                    documents.append(json.loads(doc.json))
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith("_"):
                            doc_dict[key] = value
                    documents.append(doc_dict)

            logger.debug("Redis text search: found %d results in %s", len(documents), index_name)
            return documents

        except Exception as e:
            logger.warning("Redis text search failed for index %s with query '%s': %s", index_name, query[:50], e)
            return []

    def search_vector(
        self,
        index_name: str,
        embedding: List[float],
        vector_field: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using RediSearch KNN.

        Args:
            index_name: Index to search
            embedding: Query embedding vector (must match dimension of indexed vectors)
            vector_field: Vector field to search (e.g., "summary_embedding", "details_embedding")
            limit: Maximum number of results (K in KNN)
            user_id: Filter by user_id (optional)
            organization_id: Filter by organization_id (optional)
            return_fields: Specific fields to return (None = return all)
            filter_tags: Optional filter tags for additional filtering
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of similar documents sorted by cosine similarity

        Example:
            results = redis_client.search_vector(
                index_name="idx:semantic_memory",
                embedding=[0.1, 0.2, ...],  # 1536-dim vector
                vector_field="summary_embedding",
                limit=10,
                user_id="user-123",
                organization_id="org-456",
                filter_tags={"expert_id": "expert-123"}
            )
        """
        try:
            from datetime import datetime

            import numpy as np
            from redis.commands.search.query import Query

            # Convert embedding to bytes
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # Build query parts for pre-filter
            query_parts = []

            # Add user_id filter if provided (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@user_id:{{{escaped_user_id}}}")

            # Add organization_id filter (TEXT field)
            if organization_id:
                escaped_org_id = organization_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering (numeric range query on occurred_at_ts)
            if start_date or end_date:
                # Convert datetime to Unix timestamp
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                query_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")
                logger.debug("🕐 Redis temporal filter: @occurred_at_ts:[%s %s]", min_ts, max_ts)

            # Add filter_tags filters
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    query_parts.append(filter_query)

            # Build pre-filter part (if any filters)
            pre_filter = " ".join(query_parts) if query_parts else "*"

            # Build KNN query with pre-filter
            # Using KNN syntax: (pre-filter)=>[KNN K @vector_field $vec AS distance]
            knn_query = f"({pre_filter})=>[KNN {limit} @{vector_field} $vec AS vector_distance]"

            # Build Query object
            query_obj = (
                Query(knn_query).sort_by("vector_distance").paging(0, limit).dialect(2)  # Required for vector search
            )

            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields, "vector_distance")
            # If no specific fields requested, return all fields (don't specify return_fields)
            # Redis will return all indexed fields when return_fields is not called

            # Execute search
            results = self.client.ft(index_name).search(query_obj, query_params={"vec": embedding_bytes})

            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, "json"):
                    doc_dict = json.loads(doc.json)
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith("_"):
                            doc_dict[key] = value

                # Add similarity score (convert distance to similarity)
                if hasattr(doc, "vector_distance"):
                    # Cosine distance to similarity: similarity = 1 - distance
                    doc_dict["similarity_score"] = 1.0 - float(doc.vector_distance)

                documents.append(doc_dict)

            logger.debug("Redis vector search: found %d results in %s", len(documents), index_name)
            return documents

        except Exception as e:
            logger.warning(
                "Redis vector search failed for index %s (vector_field: %s): %s", index_name, vector_field, e
            )
            return []

    def search_recent(
        self,
        index_name: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        sort_by: str = "created_at_ts",
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most recent documents from an index.

        Args:
            index_name: Index to search
            limit: Maximum number of results
            user_id: Filter by user_id (optional)
            organization_id: Filter by organization_id (optional)
            sort_by: Field to sort by (default: "created_at_ts")
            return_fields: Specific fields to return (None = return all)
            filter_tags: Optional filter tags for additional filtering
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of recent documents sorted by timestamp (descending)

        Example:
            results = redis_client.search_recent(
                index_name="idx:episodic_memory",
                limit=10,
                user_id="user-123",
                organization_id="org-456",
                sort_by="occurred_at_ts",
                filter_tags={"expert_id": "expert-123"},
                start_date=datetime(2025, 11, 19),
                end_date=datetime(2025, 11, 19, 23, 59, 59)
            )
        """
        try:
            from datetime import datetime

            from redis.commands.search.query import Query

            # Build query parts
            query_parts = []

            # Add user_id filter (escape special characters in TAG field)
            if user_id:
                escaped_user_id = user_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@user_id:{{{escaped_user_id}}}")

            # Add organization_id filter (TEXT field)
            if organization_id:
                escaped_org_id = organization_id.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering (numeric range query on occurred_at_ts)
            if start_date or end_date:
                # Convert datetime to Unix timestamp
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                query_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")
                logger.debug("🕐 Redis temporal filter: @occurred_at_ts:[%s %s]", min_ts, max_ts)

            # Add filter_tags filters
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    query_parts.append(filter_query)

            # Combine query parts
            search_query = " ".join(query_parts) if query_parts else "*"

            # Build Query object with sorting
            query_obj = (
                Query(search_query).sort_by(sort_by, asc=False).paging(0, limit)  # Descending order (most recent first)
            )

            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields)

            # Execute search
            results = self.client.ft(index_name).search(query_obj)

            # Parse results
            documents = []
            for doc in results.docs:
                # For JSON documents
                if hasattr(doc, "json"):
                    documents.append(json.loads(doc.json))
                # For Hash documents
                else:
                    doc_dict = {}
                    for key, value in doc.__dict__.items():
                        if not key.startswith("_"):
                            doc_dict[key] = value
                    documents.append(doc_dict)

            logger.debug("Redis recent search: found %d results in %s", len(documents), index_name)
            return documents

        except Exception as e:
            logger.warning("Redis recent search failed for index %s (sort_by: %s): %s", index_name, sort_by, e)
            return []

    def search_recent_by_org(
        self,
        index_name: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        sort_by: str = "created_at_ts",
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent documents filtered by organization_id and filter_tags.

        Args:
            index_name: Index to search
            limit: Maximum number of results
            organization_id: Filter by organization_id (optional)
            sort_by: Field to sort by (default: "created_at_ts")
            return_fields: Specific fields to return (None = return all)
            filter_tags: Optional filter tags (includes "scope" key for client scope matching)
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of recent documents sorted by timestamp (descending)
        """
        try:
            import re
            from datetime import datetime

            from redis.commands.search.query import Query

            def escape_text_value(value: str) -> str:
                """Escape special characters for Redis TEXT field values."""
                # Escape special chars that have meaning in Redis Search queries
                special_chars = r'[\-:.()\[\]{}"\',<>;!@#$%^&*+=~|]'
                return re.sub(special_chars, lambda m: f"\\{m.group(0)}", str(value))

            # Build query parts
            query_parts = []

            # Add organization_id filter (TEXT field) - MUST ESCAPE
            if organization_id:
                escaped_org_id = escape_text_value(organization_id)
                query_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering (numeric range query)
            if start_date or end_date:
                # Convert datetime to Unix timestamp
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                query_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")
                logger.debug("🕐 Redis temporal filter: @occurred_at_ts:[%s %s]", min_ts, max_ts)

            # Add filter_tags filters (including scope)
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    query_parts.append(filter_query)

            # Combine query parts
            search_query = " ".join(query_parts) if query_parts else "*"

            # Build Query object with sorting
            query_obj = (
                Query(search_query).sort_by(sort_by, asc=False).paging(0, limit)  # Descending order (most recent first)
            )

            # Add return fields if specified
            if return_fields:
                query_obj = query_obj.return_fields(*return_fields)

            # Execute search
            results = self.client.ft(index_name).search(query_obj)

            # Convert results to dictionaries
            return [self._doc_to_dict(doc) for doc in results.docs]

        except Exception as e:
            logger.warning("Redis org search failed for index %s: %s", index_name, e)
            return []

    def search_vector_by_org(
        self,
        index_name: str,
        embedding: List[float],
        vector_field: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        return_fields: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search filtered by organization_id and filter_tags (including scope).

        Args:
            index_name: Index to search
            embedding: Query embedding vector
            vector_field: Name of the vector field to search
            limit: Maximum number of results
            organization_id: Filter by organization_id (optional)
            return_fields: Specific fields to return (None = return all)
            filter_tags: Optional filter tags (includes "scope" key)
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of documents sorted by similarity
        """
        try:
            import re
            from datetime import datetime

            from redis.commands.search.query import Query

            def escape_text_value(value: str) -> str:
                """Escape special characters for Redis TEXT field values."""
                special_chars = r'[\-:.()\[\]{}"\',<>;!@#$%^&*+=~|]'
                return re.sub(special_chars, lambda m: f"\\{m.group(0)}", str(value))

            # Build filter query parts (organization_id + filter_tags + temporal)
            filter_parts = []

            # Add organization_id filter - MUST ESCAPE
            if organization_id:
                escaped_org_id = escape_text_value(organization_id)
                filter_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering
            if start_date or end_date:
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                filter_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")

            # Add filter_tags filters (including scope)
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    filter_parts.append(filter_query)

            # Combine filters
            filter_str = " ".join(filter_parts) if filter_parts else "*"

            # Convert embedding to bytes
            import numpy as np

            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

            # Build KNN query
            query_str = f"({filter_str})=>[KNN {limit} @{vector_field} $embedding AS distance]"

            query_obj = Query(query_str).sort_by("distance").paging(0, limit).dialect(2)

            if return_fields:
                query_obj = query_obj.return_fields(*return_fields, "distance")
            # If no specific fields requested, return all fields (don't specify return_fields)
            # Redis will return all indexed fields when return_fields is not called

            # Execute vector search
            results = self.client.ft(index_name).search(query_obj, query_params={"embedding": embedding_bytes})

            return [self._doc_to_dict(doc) for doc in results.docs]

        except Exception as e:
            logger.warning("Redis vector search failed for org search: %s", e)
            return []

    def search_text_by_org(
        self,
        index_name: str,
        query_text: str,
        search_field: str,
        search_method: str,
        limit: int = 10,
        organization_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Text search (BM25, string match) filtered by organization_id and filter_tags (including scope).

        Args:
            index_name: Index to search
            query_text: Search query text
            search_field: Field to search in
            search_method: Search method ("bm25", "string_match", etc.)
            limit: Maximum number of results
            organization_id: Filter by organization_id (optional)
            filter_tags: Optional filter tags (includes "scope" key)
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of matching documents
        """
        try:
            import re
            from datetime import datetime

            from redis.commands.search.query import Query

            def escape_text_value(value: str) -> str:
                """Escape special characters for Redis TEXT field values."""
                special_chars = r'[\-:.()\[\]{}"\',<>;!@#$%^&*+=~|]'
                return re.sub(special_chars, lambda m: f"\\{m.group(0)}", str(value))

            # Build query parts
            query_parts = []

            # Add organization_id filter - MUST ESCAPE
            if organization_id:
                escaped_org_id = escape_text_value(organization_id)
                query_parts.append(f"@organization_id:{escaped_org_id}")

            # Add temporal filtering
            if start_date or end_date:
                min_ts = int(start_date.timestamp()) if start_date else "-inf"
                max_ts = int(end_date.timestamp()) if end_date else "+inf"
                query_parts.append(f"@occurred_at_ts:[{min_ts} {max_ts}]")

            # Add filter_tags filters (including scope)
            if filter_tags:
                filter_query = self._build_filter_tags_query(filter_tags)
                if filter_query:
                    query_parts.append(filter_query)

            # Add text search query
            if query_text:
                # Escape special characters for Redis query
                escaped_query = query_text.replace("-", "\\-").replace(":", "\\:")
                query_parts.append(f"@{search_field}:{escaped_query}")

            # Combine query parts
            search_query = " ".join(query_parts) if query_parts else "*"

            # Build Query object
            query_obj = Query(search_query).paging(0, limit)

            # Execute search
            results = self.client.ft(index_name).search(query_obj)

            return [self._doc_to_dict(doc) for doc in results.docs]

        except Exception as e:
            logger.warning("Redis text search failed for org search: %s", e)
            return []

    def _doc_to_dict(self, doc) -> Dict[str, Any]:
        """
        Convert a Redis search result document to a dictionary.

        Args:
            doc: Redis document from search results

        Returns:
            Dictionary representation of the document
        """
        import json

        # For JSON documents
        if hasattr(doc, "json"):
            return json.loads(doc.json)
        # For Hash documents
        else:
            doc_dict = {}
            for key, value in doc.__dict__.items():
                if not key.startswith("_"):
                    doc_dict[key] = value
            return doc_dict

    @staticmethod
    def clean_redis_fields(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove Redis-specific fields before Pydantic validation.

        SECURITY: We keep Pydantic's extra="forbid" for API validation security.
        This helper strips Redis-internal fields (_ts timestamps, search metadata) that
        would otherwise cause validation errors, while preserving strict validation for
        user input.

        Redis stores both datetime ISO strings and numeric timestamps (_ts suffix) for sorting.
        This helper removes the Redis-specific _ts fields after search operations.

        Args:
            items: List of dictionaries from Redis Search results

        Returns:
            List of cleaned dictionaries ready for strict Pydantic validation
        """
        from datetime import datetime

        for item in items:
            # Remove Redis-specific timestamp fields (keeping only the datetime ISO strings)
            if "created_at_ts" in item:
                # If we don't have the ISO string, convert from timestamp
                if "created_at" not in item:
                    item["created_at"] = datetime.fromtimestamp(item["created_at_ts"]).isoformat()
                item.pop("created_at_ts")

            if "occurred_at_ts" in item:
                # If we don't have the ISO string, convert from timestamp
                if "occurred_at" not in item:
                    item["occurred_at"] = datetime.fromtimestamp(item["occurred_at_ts"]).isoformat()
                item.pop("occurred_at_ts")

            # Remove search metadata fields
            item.pop("similarity_score", None)
            item.pop("vector_distance", None)

        return items


def initialize_redis_client() -> Optional[RedisMemoryClient]:
    """Initialize global Redis client from settings with optimized connection pool."""
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    try:
        from mirix.settings import settings

        if not settings.redis_enabled:
            logger.info("Redis is disabled (MIRIX_REDIS_ENABLED=false)")
            return None

        redis_uri = settings.mirix_redis_uri
        if not redis_uri:
            logger.warning("Redis enabled but no URI configured")
            return None

        # Initialize with optimized connection pool settings
        _redis_client = RedisMemoryClient(
            redis_uri=redis_uri,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_connect_timeout,
            socket_keepalive=settings.redis_socket_keepalive,
            retry_on_timeout=settings.redis_retry_on_timeout,
        )

        # Test connection
        if not _redis_client.ping():
            logger.error("Redis ping failed - disabling Redis")
            _redis_client = None
            return None

        # Create indexes
        _redis_client.create_indexes()

        # Log connection pool info
        _redis_client.log_connection_stats()

        logger.info("Redis client initialized successfully with optimized connection pool")
        return _redis_client

    except Exception as e:
        logger.error("Failed to initialize Redis client: %s", e)
        logger.info("System will continue without Redis caching")
        _redis_client = None
        return None


def get_redis_client() -> Optional[RedisMemoryClient]:
    """Get the global Redis client instance."""
    return _redis_client


def close_redis_client() -> None:
    """Close the global Redis client."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None
