"""
Manager class for raw memory CRUD operations.

Raw memories are unprocessed task context stored for task sharing use cases,
with a 14-day TTL enforced by nightly cleanup jobs.
"""

import base64
import datetime as dt
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, or_, select

from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY
from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.orm.raw_memory import RawMemory
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.raw_memory import RawMemoryItem as PydanticRawMemoryItem
from mirix.schemas.raw_memory import RawMemoryItemCreate as PydanticRawMemoryItemCreate
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
        agent_state: Optional[AgentState] = None,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticRawMemoryItem:
        """
        Create a new raw memory record (direct write, no queue).

        Args:
            raw_memory: The raw memory data to create
            actor: Client performing the operation (for audit trail)
            agent_state: Agent state containing embedding configuration (optional)
            client_id: Client application identifier (defaults to actor.id)
            user_id: End-user identifier (defaults to admin user)
            use_cache: If True, cache in Redis. If False, skip caching.

        Returns:
            Created raw memory as Pydantic model
        """
        # Backward compatibility: if client_id not provided, use actor.id as fallback
        if client_id is None:
            client_id = actor.id
            logger.warning("client_id not provided to create_raw_memory, using actor.id as fallback")

        # user_id should be explicitly provided for proper multi-user isolation
        # Fallback to admin user if not provided
        if user_id is None:
            from mirix.services.user_manager import UserManager

            user_id = UserManager.ADMIN_USER_ID
            logger.warning("user_id not provided to create_raw_memory, using ADMIN_USER_ID as fallback")

        # Ensure ID is set before model_dump
        if not raw_memory.id:
            from mirix.utils import generate_unique_short_id

            raw_memory.id = generate_unique_short_id(self.session_maker, RawMemory, "raw_mem")

        logger.debug(
            "Creating raw memory: id=%s, client_id=%s, user_id=%s, filter_tags=%s",
            raw_memory.id,
            client_id,
            user_id,
            raw_memory.filter_tags,
        )

        # Conditionally calculate embeddings based on BUILD_EMBEDDINGS_FOR_MEMORY flag
        if BUILD_EMBEDDINGS_FOR_MEMORY and agent_state is not None:
            try:
                from mirix.embeddings import embedding_model

                embed_model = embedding_model(agent_state.embedding_config)
                context_embedding = embed_model.get_text_embedding(raw_memory.context)

                # Pad embeddings using Pydantic validator
                raw_memory.context_embedding = PydanticRawMemoryItemCreate.pad_embeddings(context_embedding)
                raw_memory.embedding_config = agent_state.embedding_config
            except Exception as e:
                logger.warning("Failed to generate embeddings for raw memory creation: %s", e)
                raw_memory.context_embedding = None
                raw_memory.embedding_config = None
        else:
            raw_memory.context_embedding = None
            raw_memory.embedding_config = None

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
            raw_memory_item.create_with_redis(session, actor=actor, use_cache=use_cache)

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
                    logger.debug("Redis cache HIT for raw memory %s", memory_id)
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

                raw_memory_item = RawMemory.read(db_session=session, identifier=memory_id, actor=actor)
                pydantic_memory = raw_memory_item.to_pydantic()

                # Populate Redis cache for next time
                try:
                    if redis_client:
                        data = pydantic_memory.model_dump(mode="json")
                        redis_client.set_json(redis_key, data, ttl=settings.redis_ttl_default)
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
                raise NoResultFound(f"Raw memory record with id {memory_id} not found.")

    @enforce_types
    def update_raw_memory(
        self,
        memory_id: str,
        new_context: Optional[str] = None,
        new_filter_tags: Optional[Dict[str, Any]] = None,
        actor: Optional[PydanticClient] = None,
        agent_state: Optional[AgentState] = None,
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
            agent_state: Agent state containing embedding configuration (optional)
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
            stmt = select(RawMemory).where(RawMemory.id == memory_id).with_for_update()

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
                    raw_memory.context = f"{raw_memory.context}\n\n{new_context}"
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

            # Regenerate embeddings if context changed and agent_state provided
            if BUILD_EMBEDDINGS_FOR_MEMORY and agent_state is not None and new_context is not None:
                try:
                    from mirix.embeddings import embedding_model

                    embed_model = embedding_model(agent_state.embedding_config)
                    context_embedding = embed_model.get_text_embedding(raw_memory.context)

                    raw_memory.context_embedding = PydanticRawMemoryItem.pad_embeddings(context_embedding)
                    raw_memory.embedding_config = agent_state.embedding_config
                except Exception as e:
                    logger.warning("Failed to regenerate embeddings for raw memory update: %s", e)

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
                    logger.debug("Invalidated Redis cache for memory %s", memory_id)
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
                raw_memory = RawMemory.read(db_session=session, identifier=memory_id, actor=actor)
                session.delete(raw_memory)
                session.commit()

                # Invalidate Redis cache
                try:
                    from mirix.database.redis_client import get_redis_client

                    redis_client = get_redis_client()
                    if redis_client:
                        redis_key = f"{redis_client.RAW_MEMORY_PREFIX}{memory_id}"
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
                logger.warning("Raw memory not found for deletion: id=%s", memory_id)
                return False

    @enforce_types
    def search_raw_memories(
        self,
        user: PydanticUser,
        filter_tags: Optional[Dict[str, Any]] = None,
        sort: str = "-updated_at",
        cursor: Optional[str] = None,
        time_range: Optional[Dict[str, Optional[datetime]]] = None,
        limit: int = 10,
    ) -> Tuple[List[PydanticRawMemoryItem], Optional[str]]:
        """
        Search raw memories with filtering, sorting, cursor-based pagination, and time range filtering.

        Args:
            user: User to search memories for (used for organization_id filtering)
            filter_tags: AND filter on top-level keys (scope is handled separately)
            sort: Sort field and direction (updated_at, -updated_at, created_at, -created_at, occurred_at, -occurred_at)
            cursor: Opaque Base64-encoded cursor for pagination
            time_range: Dict with keys like created_at_gte, created_at_lte, etc.
            limit: Maximum number of results (max 100, default 10)

        Returns:
            Tuple of (items, next_cursor) where next_cursor is Base64-encoded JSON or None
        """
        # Enforce limit max
        limit = min(limit, 100)

        # Parse sort string
        ascending = not sort.startswith("-")
        sort_field_name = sort.lstrip("-")

        # Validate sort field
        valid_sort_fields = {"updated_at", "created_at", "occurred_at"}
        if sort_field_name not in valid_sort_fields:
            raise ValueError(f"Invalid sort field: {sort_field_name}. Must be one of {valid_sort_fields}")

        # Decode cursor if provided
        decoded_cursor = None
        if cursor:
            try:
                decoded_bytes = base64.b64decode(cursor.encode())
                decoded_str = decoded_bytes.decode()
                decoded_cursor = json.loads(decoded_str)

                # Validate cursor has required fields
                if sort_field_name not in decoded_cursor or "id" not in decoded_cursor:
                    raise ValueError("Invalid cursor format: missing required fields")

                # Parse datetime from cursor and strip timezone for DB comparison
                cursor_sort_value = datetime.fromisoformat(decoded_cursor[sort_field_name])
                if cursor_sort_value.tzinfo:
                    cursor_sort_value = cursor_sort_value.replace(tzinfo=None)
                cursor_id = decoded_cursor["id"]
            except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ValueError(f"Invalid cursor format: {e}")

        with self.session_maker() as session:
            # Base query filtering by organization_id
            base_query = select(RawMemory).where(RawMemory.organization_id == user.organization_id)

            # Apply filter_tags (AND filter on top-level keys)
            if filter_tags:
                for key, value in filter_tags.items():
                    if key == "scope":
                        # Scope matching: input value must be in memory's scope field
                        base_query = base_query.where(
                            or_(
                                func.lower(RawMemory.filter_tags[key].as_string()).contains(str(value).lower()),
                                RawMemory.filter_tags[key].as_string() == str(value),
                            )
                        )
                    else:
                        # Other keys: exact match
                        base_query = base_query.where(RawMemory.filter_tags[key].as_string() == str(value))

            # Apply time range filtering
            if time_range:
                if time_range.get("created_at_gte"):
                    base_query = base_query.where(RawMemory.created_at >= time_range["created_at_gte"])
                if time_range.get("created_at_lte"):
                    base_query = base_query.where(RawMemory.created_at <= time_range["created_at_lte"])
                if time_range.get("occurred_at_gte"):
                    base_query = base_query.where(RawMemory.occurred_at >= time_range["occurred_at_gte"])
                if time_range.get("occurred_at_lte"):
                    base_query = base_query.where(RawMemory.occurred_at <= time_range["occurred_at_lte"])
                if time_range.get("updated_at_gte"):
                    base_query = base_query.where(RawMemory.updated_at >= time_range["updated_at_gte"])
                if time_range.get("updated_at_lte"):
                    base_query = base_query.where(RawMemory.updated_at <= time_range["updated_at_lte"])

            # Apply cursor pagination
            if decoded_cursor:
                sort_field = getattr(RawMemory, sort_field_name)
                if ascending:
                    # Get items where sort_field > cursor.sort_field OR
                    # (sort_field == cursor.sort_field AND id > cursor.id)
                    base_query = base_query.where(
                        or_(
                            sort_field > cursor_sort_value,
                            and_(
                                sort_field == cursor_sort_value,
                                RawMemory.id > cursor_id,
                            ),
                        )
                    )
                else:
                    # Get items where sort_field < cursor.sort_field OR
                    # (sort_field == cursor.sort_field AND id < cursor.id)
                    base_query = base_query.where(
                        or_(
                            sort_field < cursor_sort_value,
                            and_(
                                sort_field == cursor_sort_value,
                                RawMemory.id < cursor_id,
                            ),
                        )
                    )

            # Apply sorting
            sort_field = getattr(RawMemory, sort_field_name)
            if ascending:
                base_query = base_query.order_by(sort_field, RawMemory.id)
            else:
                base_query = base_query.order_by(desc(sort_field), desc(RawMemory.id))

            # Apply limit (fetch one extra to check if there are more results)
            base_query = base_query.limit(limit + 1)

            # Execute query
            result = session.execute(base_query)
            items = result.scalars().all()

            # Determine if there are more results and get next cursor
            has_more = len(items) > limit
            if has_more:
                items = items[:limit]  # Remove the extra item

            # Encode next cursor if there are more results
            next_cursor = None
            if has_more and items:
                last_item = items[-1]
                sort_field_value = getattr(last_item, sort_field_name)
                cursor_data = {
                    sort_field_name: sort_field_value.isoformat(),
                    "id": last_item.id,
                }
                cursor_json = json.dumps(cursor_data)
                next_cursor = base64.b64encode(cursor_json.encode()).decode()

            return [item.to_pydantic() for item in items], next_cursor
