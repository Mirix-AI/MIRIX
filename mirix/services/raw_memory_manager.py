"""
Manager class for raw memory CRUD operations.

Raw memories are unprocessed task context stored for task sharing use cases,
with a 14-day TTL enforced by nightly cleanup jobs.
"""
import datetime as dt
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.orm.raw_memory import RawMemory
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.raw_memory import (
    RawMemoryItem as PydanticRawMemoryItem,
    RawMemoryItemCreate as PydanticRawMemoryItemCreate,
)
from mirix.schemas.user import User as PydanticUser
from mirix.settings import settings
from mirix.utils import enforce_types

logger = get_logger(__name__)


class RawMemoryManager:
    """
    Manager class to handle business logic related to raw memory items.

    Raw memories are unprocessed task context stored for task sharing use cases,
    with a 14-day TTL enforced by nightly cleanup jobs.
    """

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_raw_memory(
        self,
        raw_memory: PydanticRawMemoryItemCreate,
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticRawMemoryItem:
        """
        Create a new raw memory record (direct write, no queue).

        Args:
            raw_memory: The raw memory data to create
            actor: Client performing the operation (for audit trail)
            client_id: Client application identifier (defaults to actor.id)
            user_id: End-user identifier (defaults to admin user)
            use_cache: If True, cache in Redis. If False, skip caching.

        Returns:
            Created raw memory as Pydantic model
        """
        # Backward compatibility: if client_id not provided, use actor.id as fallback
        if client_id is None:
            client_id = actor.id
            logger.warning(
                "client_id not provided to create_raw_memory, using actor.id as fallback"
            )

        # user_id should be explicitly provided for proper multi-user isolation
        # Fallback to admin user if not provided
        if user_id is None:
            from mirix.services.user_manager import UserManager

            user_id = UserManager.ADMIN_USER_ID
            logger.warning(
                "user_id not provided to create_raw_memory, using ADMIN_USER_ID as fallback"
            )

        # Ensure ID is set before model_dump
        if not raw_memory.id:
            from mirix.utils import generate_unique_short_id

            raw_memory.id = generate_unique_short_id(
                self.session_maker, RawMemory, "raw_mem"
            )

        logger.debug(
            "Creating raw memory: id=%s, client_id=%s, user_id=%s, filter_tags=%s",
            raw_memory.id,
            client_id,
            user_id,
            raw_memory.filter_tags,
        )

        # Convert the Pydantic model into a dict
        raw_memory_dict = raw_memory.model_dump()

        # Set user_id, organization_id, and audit field
        raw_memory_dict["user_id"] = user_id
        raw_memory_dict["organization_id"] = actor.organization_id
        raw_memory_dict["_created_by_id"] = client_id

        # Default timestamps to now if not provided
        now = datetime.now(dt.timezone.utc)
        if not raw_memory_dict.get("occurred_at"):
            raw_memory_dict["occurred_at"] = now
        if not raw_memory_dict.get("created_at"):
            raw_memory_dict["created_at"] = now
        if not raw_memory_dict.get("updated_at"):
            raw_memory_dict["updated_at"] = now

        # Validate required fields
        if not raw_memory_dict.get("context"):
            raise ValueError("Required field 'context' is missing or empty")

        # Create the raw memory item (with conditional Redis caching)
        with self.session_maker() as session:
            raw_memory_item = RawMemory(**raw_memory_dict)
            raw_memory_item.create_with_redis(
                session, actor=actor, use_cache=use_cache
            )

            logger.info("Raw memory created: id=%s", raw_memory_item.id)
            return raw_memory_item.to_pydantic()

    @enforce_types
    def get_raw_memory_by_id(
        self,
        memory_id: str,
        user: PydanticUser,
    ) -> Optional[PydanticRawMemoryItem]:
        """
        Fetch a single raw memory record by ID (with Redis JSON caching).

        Args:
            memory_id: ID of the memory to fetch
            user: User who owns this memory

        Returns:
            Raw memory as Pydantic model

        Raises:
            NoResultFound: If the record doesn't exist or doesn't belong to user
        """
        # Try Redis cache first (JSON-based for memory tables)
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{memory_id}"
                cached_data = redis_client.get_json(redis_key)
                if cached_data:
                    # Cache HIT - return from Redis
                    logger.debug(
                        "✅ Redis cache HIT for raw memory %s", memory_id
                    )
                    return PydanticRawMemoryItem(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            logger.warning(
                "Redis cache read failed for raw memory %s: %s",
                memory_id,
                e,
            )

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                # Construct a PydanticClient for actor using user's organization_id
                actor = PydanticClient(
                    id="system-default-client",
                    organization_id=user.organization_id,
                    name="system-client",
                )

                raw_memory_item = RawMemory.read(
                    db_session=session, identifier=memory_id, actor=actor
                )
                pydantic_memory = raw_memory_item.to_pydantic()

                # Populate Redis cache for next time
                try:
                    if redis_client:
                        data = pydantic_memory.model_dump(mode="json")
                        redis_client.set_json(
                            redis_key, data, ttl=settings.redis_ttl_default
                        )
                        logger.debug(
                            "Populated Redis cache for raw memory %s",
                            memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to populate Redis cache for raw memory %s: %s",
                        memory_id,
                        e,
                    )

                return pydantic_memory
            except NoResultFound:
                raise NoResultFound(
                    f"Raw memory record with id {memory_id} not found."
                )

    @enforce_types
    def update_raw_memory(
        self,
        memory_id: str,
        new_context: Optional[str] = None,
        new_filter_tags: Optional[Dict[str, Any]] = None,
        actor: Optional[PydanticClient] = None,
        context_update_mode: str = "replace",
        tags_merge_mode: str = "replace",
    ) -> PydanticRawMemoryItem:
        """
        Update an existing raw memory record.

        Args:
            memory_id: ID of the memory to update
            new_context: New context text
            new_filter_tags: New or updated filter tags
            actor: Client performing the update (for access control and audit)
            context_update_mode: How to handle context updates ("append" or "replace")
            tags_merge_mode: How to handle filter_tags updates ("merge" or "replace")

        Returns:
            Updated raw memory as Pydantic model

        Raises:
            ValueError: If memory not found or validation fails
        """
        logger.debug(
            "Updating raw memory: id=%s, context_mode=%s, tags_mode=%s",
            memory_id,
            context_update_mode,
            tags_merge_mode,
        )

        with self.session_maker() as session:
            # Fetch the existing memory with row-level lock (SELECT FOR UPDATE)
            # This prevents race conditions when multiple agents append/merge concurrently
            stmt = (
                select(RawMemory)
                .where(RawMemory.id == memory_id)
                .with_for_update()
            )

            result = session.execute(stmt)
            try:
                raw_memory = result.scalar_one()
            except NoResultFound:
                raise ValueError(f"Raw memory {memory_id} not found")

            # Perform access control check (replaces RawMemory.read's built-in check)
            if actor and raw_memory.organization_id != actor.organization_id:
                raise ValueError(
                    f"Access denied: memory {memory_id} belongs to "
                    f"organization {raw_memory.organization_id}, "
                    f"actor belongs to {actor.organization_id}"
                )

            # Update context
            if new_context is not None:
                if context_update_mode == "append":
                    raw_memory.context = (
                        f"{raw_memory.context}\n\n{new_context}"
                    )
                    logger.debug("Appended to context for memory %s", memory_id)
                else:  # replace
                    raw_memory.context = new_context
                    logger.debug("Replaced context for memory %s", memory_id)

            # Update filter_tags
            if new_filter_tags is not None:
                if tags_merge_mode == "merge":
                    # Merge new tags with existing
                    existing_tags = raw_memory.filter_tags or {}
                    raw_memory.filter_tags = {
                        **existing_tags,
                        **new_filter_tags,
                    }
                    logger.debug("Merged filter_tags for memory %s", memory_id)
                else:  # replace
                    raw_memory.filter_tags = new_filter_tags
                    logger.debug("Replaced filter_tags for memory %s", memory_id)

            # Update last_modify and timestamp
            raw_memory.updated_at = datetime.now(timezone.utc)
            raw_memory.last_modify = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": "updated",
            }
            raw_memory._last_update_by_id = actor.id if actor else None

            # Commit changes
            session.commit()

            # Invalidate Redis cache
            try:
                from mirix.database.redis_client import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{memory_id}"
                    redis_client.delete(redis_key)
                    logger.debug(
                        "Invalidated Redis cache for memory %s", memory_id
                    )
            except Exception as e:
                logger.warning(
                    "Failed to invalidate Redis cache for memory %s: %s",
                    memory_id,
                    e,
                )

            logger.info("Raw memory updated: id=%s", memory_id)
            return raw_memory.to_pydantic()

    @enforce_types
    def delete_raw_memory(
        self,
        memory_id: str,
        actor: PydanticClient,
    ) -> bool:
        """
        Delete a raw memory (hard delete, used by cleanup job).

        Args:
            memory_id: ID of the memory to delete
            actor: Client performing the deletion (for access control)

        Returns:
            True if deleted, False if not found
        """
        logger.info("Deleting raw memory: id=%s", memory_id)

        with self.session_maker() as session:
            try:
                raw_memory = RawMemory.read(
                    db_session=session, identifier=memory_id, actor=actor
                )
                session.delete(raw_memory)
                session.commit()

                # Invalidate Redis cache
                try:
                    from mirix.database.redis_client import get_redis_client

                    redis_client = get_redis_client()
                    if redis_client:
                        redis_key = (
                            f"{redis_client.RAW_MEMORY_PREFIX}{memory_id}"
                        )
                        redis_client.delete(redis_key)
                        logger.debug(
                            "Invalidated Redis cache for deleted memory %s",
                            memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to invalidate Redis cache for deleted memory %s: %s",
                        memory_id,
                        e,
                    )

                logger.info("Raw memory deleted: id=%s", memory_id)
                return True
            except NoResultFound:
                logger.warning(
                    "Raw memory not found for deletion: id=%s", memory_id
                )
                return False
