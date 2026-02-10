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
from mirix.services.user_manager import UserManager
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
        user_id: str,
        agent_state: Optional[AgentState] = None,
        client_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticRawMemoryItem:
        """
        Create a new raw memory record (direct write, no queue).

        Args:
            raw_memory: The raw memory data to create
            actor: Client performing the operation (for audit trail)
            user_id: End-user identifier (required)
            agent_state: Agent state containing embedding configuration (optional)
            client_id: Client application identifier (defaults to actor.id, used for audit trail)
            use_cache: If True, cache in Redis. If False, skip caching.

        Returns:
            Created raw memory as Pydantic model

        Raises:
            ValueError: If user_id is not provided
        """
        # Validate user_id is provided
        if not user_id:
            raise ValueError("user_id is required for create_raw_memory")

        # Backward compatibility: if client_id not provided, use actor.id as fallback
        if client_id is None:
            client_id = actor.id
            logger.warning("client_id not provided to create_raw_memory, using actor.id as fallback")

        # Auto-create user if it doesn't exist (users are organization-scoped, not client-scoped)
        user_manager = UserManager()
        try:
            user_manager.get_user_by_id(user_id)
        except NoResultFound:
            logger.info(
                "User with id=%s not found, auto-creating with organization_id=%s",
                user_id,
                actor.organization_id,
            )
            try:
                # Create user with provided user_id and client's organization
                user_manager.create_user(
                    pydantic_user=PydanticUser(
                        id=user_id,
                        name=user_id,  # Use user_id as default name
                        organization_id=actor.organization_id,
                        timezone=user_manager.DEFAULT_TIME_ZONE,
                        status="active",
                        is_deleted=False,
                        is_admin=False,
                    )
                )
                logger.info(
                    "✓ Auto-created user: %s in organization: %s",
                    user_id,
                    actor.organization_id,
                )
            except Exception as create_error:
                logger.error(
                    "Failed to auto-create user with id=%s: %s",
                    user_id,
                    create_error,
                )
                raise create_error

        # Ensure ID is set before model_dump
        if not raw_memory.id:
            from mirix.utils import generate_unique_short_id

            raw_memory.id = generate_unique_short_id(self.session_maker, RawMemory, "raw_mem")

        # Auto-inject scope from actor
        if raw_memory.filter_tags is None:
            raw_memory.filter_tags = {}
        raw_memory.filter_tags["scope"] = actor.scope

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
        actor: PydanticClient,
        user_id: Optional[str] = None,
    ) -> Optional[PydanticRawMemoryItem]:
        """
        Fetch a single raw memory record by ID (with Redis JSON caching).

        Args:
            memory_id: ID of the memory to fetch
            actor: Client performing the operation (for scope validation)
            user_id: Optional user ID - if provided, filters by user (404 if mismatch)

        Returns:
            Raw memory as Pydantic model

        Raises:
            NoResultFound: If the record doesn't exist, scope doesn't match, or user doesn't match
        """
        # Try cache first (cache provider: Redis or IPS Cache)
        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.RAW_MEMORY_PREFIX}{memory_id}"
                cached_data = cache_provider.get_json(cache_key)
                if cached_data:
                    # Cache HIT - validate scope before returning
                    logger.debug("Cache HIT for raw memory %s", memory_id)
                    pydantic_memory = PydanticRawMemoryItem(**cached_data)

                    # Validate scope
                    memory_scope = (pydantic_memory.filter_tags or {}).get("scope")
                    if memory_scope != actor.scope:
                        raise NoResultFound(f"Raw memory record with id {memory_id} not found.")

                    # Validate user_id if provided
                    if user_id and pydantic_memory.user_id != user_id:
                        raise NoResultFound(f"Raw memory record with id {memory_id} not found.")

                    return pydantic_memory
        except NoResultFound:
            raise
        except Exception as e:
            # Log but continue to PostgreSQL on cache error
            logger.warning(
                "Cache read failed for raw memory %s: %s",
                memory_id,
                e,
            )

        # Cache MISS or cache unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                raw_memory_item = RawMemory.read(db_session=session, identifier=memory_id, actor=actor)
                pydantic_memory = raw_memory_item.to_pydantic()

                # Validate scope
                memory_scope = (pydantic_memory.filter_tags or {}).get("scope")
                if memory_scope != actor.scope:
                    raise NoResultFound(f"Raw memory record with id {memory_id} not found.")

                # Validate user_id if provided
                if user_id and pydantic_memory.user_id != user_id:
                    raise NoResultFound(f"Raw memory record with id {memory_id} not found.")

                # Populate cache for next time
                try:
                    if cache_provider:
                        cache_key = f"{cache_provider.RAW_MEMORY_PREFIX}{memory_id}"
                        data = pydantic_memory.model_dump(mode="json")
                        cache_provider.set_json(cache_key, data, ttl=settings.redis_ttl_default)
                        logger.debug(
                            "Populated cache for raw memory %s",
                            memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to populate cache for raw memory %s: %s",
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
        actor: PydanticClient,
        new_context: Optional[str] = None,
        new_filter_tags: Optional[Dict[str, Any]] = None,
        agent_state: Optional[AgentState] = None,
        context_update_mode: str = "replace",
        tags_merge_mode: str = "replace",
        user_id: Optional[str] = None,
    ) -> PydanticRawMemoryItem:
        """
        Update an existing raw memory record.

        Args:
            memory_id: ID of the memory to update
            new_context: New context text
            new_filter_tags: New or updated filter tags
            actor: Client performing the update (required for access control)
            agent_state: Agent state containing embedding configuration (optional)
            context_update_mode: How to handle context updates ("append" or "replace")
            tags_merge_mode: How to handle filter_tags updates ("merge" or "replace")
            user_id: Optional user ID - if provided, filters by user (404 if mismatch)

        Returns:
            Updated raw memory as Pydantic model

        Raises:
            ValueError: If memory not found, scope mismatch, user mismatch, or validation fails
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
            if raw_memory.organization_id != actor.organization_id:
                raise ValueError(
                    f"Access denied: memory {memory_id} belongs to "
                    f"organization {raw_memory.organization_id}, "
                    f"actor belongs to {actor.organization_id}"
                )

            # Perform scope access control check
            memory_scope = (raw_memory.filter_tags or {}).get("scope")
            if memory_scope != actor.scope:
                raise ValueError(
                    f"Access denied: memory {memory_id} has scope '{memory_scope}', " f"actor has scope '{actor.scope}'"
                )

            # Perform user_id access control check if provided
            if user_id and raw_memory.user_id != user_id:
                raise ValueError(f"Raw memory {memory_id} not found")

            # Prevent scope tampering in filter_tags updates
            if new_filter_tags is not None and "scope" in new_filter_tags:
                if new_filter_tags["scope"] != actor.scope:
                    raise ValueError("Cannot change memory scope - scope must match actor.scope")

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
                    # Preserve scope when replacing tags - scope is immutable
                    preserved_scope = (raw_memory.filter_tags or {}).get("scope")
                    raw_memory.filter_tags = new_filter_tags
                    if preserved_scope:
                        raw_memory.filter_tags["scope"] = preserved_scope
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
            # Audit field (_last_updated_by_id) is handled by base class via
            # property accessor when using update_with_redis(), or can be set
            # manually: raw_memory.last_updated_by_id = actor.id
            if actor:
                raw_memory.last_updated_by_id = actor.id

            # Commit changes
            session.commit()

            # Invalidate cache
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.RAW_MEMORY_PREFIX}{memory_id}"
                    cache_provider.delete(cache_key)
                    logger.debug("Invalidated cache for memory %s", memory_id)
            except Exception as e:
                logger.warning(
                    "Failed to invalidate cache for memory %s: %s",
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
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a raw memory (hard delete, used by cleanup job).

        Args:
            memory_id: ID of the memory to delete
            actor: Client performing the deletion (for access control)
            user_id: Optional user ID - if provided, filters by user (returns False if mismatch)

        Returns:
            True if deleted, False if not found or user mismatch
        """
        logger.info("Deleting raw memory: id=%s", memory_id)

        with self.session_maker() as session:
            try:
                raw_memory = RawMemory.read(db_session=session, identifier=memory_id, actor=actor)

                # Perform scope access control check
                memory_scope = (raw_memory.filter_tags or {}).get("scope")
                if memory_scope != actor.scope:
                    raise ValueError(
                        f"Access denied: memory {memory_id} has scope '{memory_scope}', "
                        f"actor has scope '{actor.scope}'"
                    )

                # Perform user_id access control check if provided
                if user_id and raw_memory.user_id != user_id:
                    logger.warning("Raw memory %s not found for deletion (user mismatch)", memory_id)
                    return False

                session.delete(raw_memory)
                session.commit()

                # Invalidate cache
                try:
                    from mirix.database.cache_provider import get_cache_provider

                    cache_provider = get_cache_provider()
                    if cache_provider:
                        cache_key = f"{cache_provider.RAW_MEMORY_PREFIX}{memory_id}"
                        cache_provider.delete(cache_key)
                        logger.debug(
                            "Invalidated cache for deleted memory %s",
                            memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to invalidate cache for deleted memory %s: %s",
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
        organization_id: str,
        user_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        sort: str = "-updated_at",
        cursor: Optional[str] = None,
        time_range: Optional[Dict[str, Optional[datetime]]] = None,
        limit: int = 10,
    ) -> Tuple[List[PydanticRawMemoryItem], Optional[str]]:
        """
        Search raw memories with filtering, sorting, cursor-based pagination, and time range filtering.

        Args:
            organization_id: Organization ID to filter by (required)
            user_id: Optional user ID - if provided, filters by user
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
            base_query = select(RawMemory).where(RawMemory.organization_id == organization_id)

            # Apply user_id filter if provided
            if user_id:
                base_query = base_query.where(RawMemory.user_id == user_id)

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
