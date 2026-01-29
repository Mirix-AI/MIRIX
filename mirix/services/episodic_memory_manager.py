import datetime as dt
import json
import re
import string
from datetime import datetime
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sqlalchemy import func, select, text

from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY
from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.schemas.user import User as PydanticUser
from mirix.services.utils import build_query, update_timezone
from mirix.settings import settings
from mirix.utils import enforce_types

logger = get_logger(__name__)


class EpisodicMemoryManager:
    """Manager class to handle business logic related to Episodic episodic_memory items."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    def _clean_text_for_search(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text with punctuation removed and normalized whitespace
        """
        if not text:
            return ""

        # Remove punctuation using string.punctuation
        # Create translation table that maps each punctuation character to space
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        text = text.translate(translator)

        # Convert to lowercase and normalize whitespace
        text = re.sub(r"\s+", " ", text.lower().strip())

        return text

    def _preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 search by tokenizing and cleaning.

        Args:
            text: Input text to preprocess

        Returns:
            List of cleaned tokens
        """
        if not text:
            return []

        # Clean text first
        cleaned_text = self._clean_text_for_search(text)

        # Split into tokens and filter out empty strings and very short tokens
        tokens = [token for token in cleaned_text.split() if token.strip() and len(token) > 1]
        return tokens

    def _count_word_matches(self, event_data: Dict[str, Any], query_words: List[str], search_field: str = "") -> int:
        """
        Count how many of the query words are present in the event data.

        Args:
            event_data: Dictionary containing event data
            query_words: List of query words to search for
            search_field: Specific field to search in, or empty string to search all text fields

        Returns:
            Number of query words found in the event
        """
        if not query_words:
            return 0

        # Determine which text fields to search in
        if search_field == "summary":
            search_texts = [event_data.get("summary", "")]
        elif search_field == "details":
            search_texts = [event_data.get("details", "")]
        elif search_field == "actor":
            search_texts = [event_data.get("actor", "")]
        elif search_field == "event_type":
            search_texts = [event_data.get("event_type", "")]
        else:
            # Search across all relevant text fields
            search_texts = [
                event_data.get("summary", ""),
                event_data.get("details", ""),
                event_data.get("actor", ""),
                event_data.get("event_type", ""),
            ]

        # Combine all search texts and clean them (remove punctuation)
        combined_text = " ".join(text for text in search_texts if text)
        cleaned_combined_text = self._clean_text_for_search(combined_text)

        # Count how many query words are present
        word_matches = 0
        for word in query_words:
            # Query words are already cleaned, so we can do direct comparison
            if word in cleaned_combined_text:
                word_matches += 1

        return word_matches

    @update_timezone
    @enforce_types
    def get_episodic_memory_by_id(
        self, episodic_memory_id: str, user: PydanticUser, timezone_str: str = None
    ) -> Optional[PydanticEpisodicEvent]:
        """
        Fetch a single episodic memory record by ID (with Redis JSON caching).

        Args:
            episodic_memory_id: ID of the memory to fetch
            user: User who owns this memory
            timezone_str: Optional timezone string

        Raises:
            NoResultFound: If the record doesn't exist or doesn't belong to user
        """
        # Try Redis cache first (JSON-based for memory tables)
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.EPISODIC_PREFIX}{episodic_memory_id}"
                cached_data = redis_client.get_json(redis_key)
                if cached_data:
                    # Cache HIT - return from Redis
                    logger.debug("Redis cache HIT for episodic memory %s", episodic_memory_id)
                    return PydanticEpisodicEvent(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            logger.warning(
                "Redis cache read failed for episodic memory %s: %s",
                episodic_memory_id,
                e,
            )

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                # Construct a PydanticClient for actor using user's organization_id.
                # Note: We can pass in a PydanticClient with a default client ID because
                # EpisodicEvent.read() only uses the organization_id from the actor for
                # access control (see apply_access_predicate in sqlalchemy_base.py).
                # The actual client ID is not used for filtering.
                actor = PydanticClient(
                    id="system-default-client",
                    organization_id=user.organization_id,
                    name="system-client",
                )

                episodic_memory_item = EpisodicEvent.read(
                    db_session=session, identifier=episodic_memory_id, actor=actor
                )
                pydantic_event = episodic_memory_item.to_pydantic()

                # Populate Redis cache for next time
                try:
                    if redis_client:
                        data = pydantic_event.model_dump(mode="json")
                        # model_dump(mode='json') already converts datetime to ISO format strings
                        redis_client.set_json(redis_key, data, ttl=settings.redis_ttl_default)
                        logger.debug(
                            "Populated Redis cache for episodic memory %s",
                            episodic_memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to populate Redis cache for episodic memory %s: %s",
                        episodic_memory_id,
                        e,
                    )

                return pydantic_event
            except NoResultFound:
                raise NoResultFound(f"Episodic episodic_memory record with id {episodic_memory_id} not found.")

    @update_timezone
    @enforce_types
    def get_most_recently_updated_event(
        self, user: PydanticUser, timezone_str: str = None
    ) -> Optional[PydanticEpisodicEvent]:
        """
        Fetch the most recently updated episodic event based on last_modify timestamp.

        Args:
            user: User who owns the memories to query
            timezone_str: Optional timezone string

        Returns:
            Most recent event or None if no events exist
        """
        with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import DateTime, cast, text

            query = (
                select(EpisodicEvent)
                .where(EpisodicEvent.user_id == user.id)
                .order_by(cast(text("episodic_memory.last_modify ->> 'timestamp'"), DateTime).desc())
            )

            result = session.execute(query.limit(1))
            episodic_memory = result.scalar_one_or_none()

            return [episodic_memory.to_pydantic()] if episodic_memory else None

    @enforce_types
    def create_episodic_memory(
        self,
        episodic_memory: PydanticEpisodicEvent,
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticEpisodicEvent:
        """
        Create a new episodic episodic_memory record.
        Uses the provided Pydantic model (PydanticEpisodicEvent) as input data.

        Args:
            episodic_memory: The episodic memory data to create
            actor: Client performing the operation (for audit trail)
            client_id: Client application identifier (defaults to actor.id)
            user_id: End-user identifier (required for multi-user isolation)
            use_cache: If True, cache in Redis. If False, skip caching.
        """

        # Backward compatibility: if client_id not provided, use actor.id as fallback
        if client_id is None:
            client_id = actor.id
            logger.warning("client_id not provided to create_episodic_memory, using actor.id as fallback")

        # user_id should be explicitly provided for proper multi-user isolation
        # Fallback to None if not provided - callers should pass user_id explicitly
        if user_id is None:
            from mirix.services.user_manager import UserManager

            user_id = UserManager.ADMIN_USER_ID
            logger.warning("user_id not provided to create_episodic_memory, using ADMIN_USER_ID as fallback")

        # Ensure ID is set before model_dump
        if not episodic_memory.id:
            from mirix.utils import generate_unique_short_id

            episodic_memory.id = generate_unique_short_id(self.session_maker, EpisodicEvent, "ep")

        # Convert the Pydantic model into a dict
        episodic_memory_dict = episodic_memory.model_dump()

        # Set client_id and user_id on the memory
        episodic_memory_dict["client_id"] = client_id
        episodic_memory_dict["user_id"] = user_id

        logger.debug(
            "create_episodic_memory: client_id=%s, user_id=%s, filter_tags=%s",
            client_id,
            user_id,
            episodic_memory.filter_tags,
        )

        # Validate required fields if necessary (event_type, summary, etc.)
        required_fields = ["event_type", "summary"]
        for field in required_fields:
            if field not in episodic_memory_dict or episodic_memory_dict[field] is None:
                raise ValueError(f"Required field '{field}' is missing or None in episodic episodic_memory data")

        # Set defaults if needed
        episodic_memory_dict.setdefault("organization_id", actor.organization_id)

        # Other fields like occurred_at, created_at, etc.
        # might be auto-generated by the model or the DB

        # Create the episodic memory item (with conditional Redis caching)
        with self.session_maker() as session:
            episodic_memory_item = EpisodicEvent(**episodic_memory_dict)
            episodic_memory_item.create_with_redis(session, actor=actor, use_cache=use_cache)
            return episodic_memory_item.to_pydantic()

    @enforce_types
    def create_many_episodic_memory(
        self,
        episodic_memory: List[PydanticEpisodicEvent],
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[PydanticEpisodicEvent]:
        """
        Create multiple episodic episodic_memory records in one go.
        """
        return [self.create_episodic_memory(e, actor, client_id, user_id) for e in episodic_memory]

    @enforce_types
    def delete_event_by_id(self, id: str, actor: PydanticClient) -> None:
        """
        Delete an episodic memory record by ID (removes from Redis cache).
        """
        with self.session_maker() as session:
            try:
                episodic_memory_item = EpisodicEvent.read(db_session=session, identifier=id, actor=actor)
                # Remove from Redis cache before hard delete
                from mirix.database.redis_client import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.EPISODIC_PREFIX}{id}"
                    redis_client.delete(redis_key)
                episodic_memory_item.hard_delete(session)
            except NoResultFound:
                raise NoResultFound(f"Episodic episodic_memory record with id {id} not found.")

    @enforce_types
    def delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk delete all episodic memory records for a client (removes from Redis cache).
        Optimized with single DB query and batch Redis deletion.

        Args:
            actor: Client whose memories to delete (uses actor.id as client_id)

        Returns:
            Number of records deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Get IDs for Redis cleanup (only fetch IDs, not full objects)
            item_ids = [
                row[0] for row in session.query(EpisodicEvent.id).filter(EpisodicEvent.client_id == actor.id).all()
            ]

            count = len(item_ids)
            if count == 0:
                return 0

            # Bulk delete in single query
            session.query(EpisodicEvent).filter(EpisodicEvent.client_id == actor.id).delete(synchronize_session=False)

            session.commit()

        # Batch delete from Redis cache (outside of session context)
        redis_client = get_redis_client()
        if redis_client and item_ids:
            redis_keys = [f"{redis_client.EPISODIC_PREFIX}{item_id}" for item_id in item_ids]

            # Delete in batches to avoid command size limits
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count

    def soft_delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk soft delete all episodic memory records for a client (updates Redis cache).

        Args:
            actor: Client whose memories to soft delete (uses actor.id as client_id)

        Returns:
            Number of records soft deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Query all non-deleted records for this client (use actor.id)
            items = (
                session.query(EpisodicEvent)
                .filter(
                    EpisodicEvent.client_id == actor.id,
                    EpisodicEvent.is_deleted == False,
                )
                .all()
            )

            count = len(items)
            if count == 0:
                return 0

            # Extract IDs BEFORE committing (to avoid detached instance errors)
            item_ids = [item.id for item in items]

            # Soft delete from database (set is_deleted = True directly, don't call item.delete())
            for item in items:
                item.is_deleted = True
                item.set_updated_at()

            session.commit()

        # Update Redis cache with is_deleted=true (outside session)
        redis_client = get_redis_client()
        if redis_client:
            for item_id in item_ids:
                redis_key = f"{redis_client.EPISODIC_PREFIX}{item_id}"
                try:
                    redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    # If update fails, remove from cache
                    redis_client.delete(redis_key)

        return count

    def soft_delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk soft delete all episodic memory records for a user (updates Redis cache).

        Args:
            user_id: ID of the user whose memories to soft delete

        Returns:
            Number of records soft deleted
        """
        import datetime as dt
        from datetime import datetime

        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Extract IDs BEFORE bulk update (for Redis cleanup)
            item_ids = [
                row[0]
                for row in session.query(EpisodicEvent.id)
                .filter(EpisodicEvent.user_id == user_id, EpisodicEvent.is_deleted == False)
                .all()
            ]

            count = len(item_ids)
            if count == 0:
                return 0

            # Batch soft delete in database using single SQL UPDATE
            session.query(EpisodicEvent).filter(
                EpisodicEvent.user_id == user_id, EpisodicEvent.is_deleted == False
            ).update(
                {"is_deleted": True, "updated_at": datetime.now(dt.UTC)},
                synchronize_session=False,
            )

            session.commit()

        # Batch update Redis cache using pipeline (outside session)
        redis_client = get_redis_client()
        if redis_client and item_ids:
            try:
                # Use Redis pipeline for batch operations
                pipe = redis_client.client.pipeline()
                for item_id in item_ids:
                    redis_key = f"{redis_client.EPISODIC_PREFIX}{item_id}"
                    pipe.hset(redis_key, "is_deleted", "true")
                pipe.execute()
            except Exception as e:
                # If pipeline fails, fall back to individual deletions
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    "Redis pipeline failed for soft_delete_by_user_id, removing keys: %s",
                    e,
                )
                for item_id in item_ids:
                    redis_key = f"{redis_client.EPISODIC_PREFIX}{item_id}"
                    try:
                        redis_client.delete(redis_key)
                    except Exception:
                        pass

        return count

    def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all episodic memory records for a user (removes from Redis cache).
        Optimized with single DB query and batch Redis deletion.

        Args:
            user_id: ID of the user whose memories to delete

        Returns:
            Number of records deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Get IDs for Redis cleanup (only fetch IDs, not full objects)
            item_ids = [
                row[0] for row in session.query(EpisodicEvent.id).filter(EpisodicEvent.user_id == user_id).all()
            ]

            count = len(item_ids)
            if count == 0:
                return 0

            # Bulk delete in single query
            session.query(EpisodicEvent).filter(EpisodicEvent.user_id == user_id).delete(synchronize_session=False)

            session.commit()

        # Batch delete from Redis cache (outside of session context)
        redis_client = get_redis_client()
        if redis_client and item_ids:
            redis_keys = [f"{redis_client.EPISODIC_PREFIX}{item_id}" for item_id in item_ids]

            # Delete in batches to avoid command size limits
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count

    @enforce_types
    def insert_event(
        self,
        actor: PydanticClient,
        agent_state: AgentState,
        agent_id: str,
        event_type: str,
        timestamp: datetime,
        event_actor: str,
        details: str,
        summary: str,
        organization_id: str,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> PydanticEpisodicEvent:
        try:
            logger.debug("insert_event called with filter_tags: %s", filter_tags)

            # Set defaults for required fields
            from mirix.services.user_manager import UserManager

            if client_id is None:
                client_id = actor.id
            if user_id is None:
                user_id = UserManager.ADMIN_USER_ID
                logger.debug("user_id not provided, using ADMIN_USER_ID: %s", user_id)
            # Conditionally calculate embeddings based on BUILD_EMBEDDINGS_FOR_MEMORY flag
            if BUILD_EMBEDDINGS_FOR_MEMORY:
                # TODO: need to check if we need to chunk the text
                embed_model = embedding_model(agent_state.embedding_config)
                details_embedding = embed_model.get_text_embedding(details)
                summary_embedding = embed_model.get_text_embedding(summary)
                embedding_config = agent_state.embedding_config
            else:
                details_embedding = None
                summary_embedding = None
                embedding_config = None

            event = self.create_episodic_memory(
                PydanticEpisodicEvent(
                    occurred_at=timestamp,
                    event_type=event_type,
                    client_id=client_id,  # Required field: client app that created this memory
                    user_id=user_id,  # Required field: end-user who owns this memory
                    agent_id=agent_id,
                    actor=event_actor,
                    summary=summary,
                    details=details,
                    organization_id=organization_id,
                    summary_embedding=summary_embedding,
                    details_embedding=details_embedding,
                    embedding_config=embedding_config,
                    filter_tags=filter_tags,
                    last_modify={
                        "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                        "operation": "created",
                    },
                ),
                actor=actor,
                client_id=client_id,
                user_id=user_id,
                use_cache=use_cache,
            )

            return event

        except Exception as e:
            raise e

    @update_timezone
    @enforce_types
    def list_episodic_memory_around_timestamp(
        self,
        agent_state: AgentState,
        start_time: datetime,
        end_time: datetime,
        user: PydanticUser,
        timezone_str: str = None,
    ) -> List[PydanticEpisodicEvent]:
        """
        List all episodic events around a timestamp.

        Args:
            agent_state: Agent state
            start_time: Start of time window
            end_time: End of time window
            user: User who owns the memories to query
            timezone_str: Optional timezone string
        """
        with self.session_maker() as session:
            # Query for episodic events within the time window
            query = select(EpisodicEvent).where(
                EpisodicEvent.occurred_at.between(start_time, end_time),
                EpisodicEvent.user_id == user.id,
            )

            result = session.execute(query)
            episodic_memory = result.scalars().all()

            return [event.to_pydantic() for event in episodic_memory]

    def get_total_number_of_items(self, user: PydanticUser) -> int:
        """
        Get the total number of items in the episodic memory for the user.

        Args:
            user: User who owns the memories to count
        """
        with self.session_maker() as session:
            query = select(func.count(EpisodicEvent.id)).where(EpisodicEvent.user_id == user.id)
            result = session.execute(query)
            return result.scalar_one()

    @update_timezone
    @enforce_types
    def list_episodic_memory(
        self,
        agent_state: AgentState,
        user: PydanticUser,
        query: str = "",
        embedded_text: Optional[List[float]] = None,
        search_field: str = "",
        search_method: str = "embedding",
        limit: Optional[int] = 50,
        timezone_str: str = None,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticEpisodicEvent]:
        """
        List all episodic events with various search methods and optional temporal filtering.

        Args:
            agent_state: The agent state containing embedding configuration
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('summary', 'details', 'actor', 'event_type', etc.)
            search_method: Search method to use:
                - 'embedding': Vector similarity search using embeddings
                - 'string_match': Simple string containment search
                - 'bm25': **RECOMMENDED** - PostgreSQL native full-text search (ts_rank_cd) when using PostgreSQL,
                               falls back to in-memory BM25 for SQLite
                - 'fuzzy_match': Fuzzy string matching (legacy, kept for compatibility)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            filter_tags: Tag-based filtering (key-value pairs)
            use_cache: If True, try Redis cache first. If False, skip cache and query PostgreSQL directly.
            start_date: Optional start datetime for filtering by occurred_at (inclusive)
            end_date: Optional end datetime for filtering by occurred_at (inclusive)

        Returns:
            List of episodic events matching the search criteria

        Note:
            **For PostgreSQL users**: 'bm25' is now the recommended method for text-based searches as it uses
            PostgreSQL's native full-text search with ts_rank_cd for BM25-like scoring. This is much more efficient
            than loading all documents into memory and leverages your existing GIN indexes.

            **For SQLite users**: 'fts5_match' is recommended for text-based searches as it's efficient and uses
            proper BM25 ranking. 'fts5_match' requires SQLite compiled with FTS5 support.

            Performance comparison:
            - PostgreSQL 'bm25': Native DB search, very fast, scales well
            - Legacy 'bm25' (SQLite): In-memory processing, slow for large datasets

            **Temporal filtering**: When start_date and/or end_date are provided, only events with
            occurred_at within the specified range will be returned. This is particularly useful for
            queries like "What happened today?" or "Show me last week's events".
        """

        # Extract organization_id from user for multi-tenant isolation
        organization_id = user.organization_id

        # Try Redis Search first (if cache enabled and Redis is available)
        from mirix.database.redis_client import get_redis_client

        redis_client = get_redis_client()

        if use_cache and redis_client:
            try:
                # Case 1: No query - get recent items
                if not query or query == "":
                    results = redis_client.search_recent(
                        index_name=redis_client.EPISODIC_INDEX,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        sort_by="occurred_at_ts",  # Sort by occurred_at for episodic
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis cache HIT: returned %d recent episodic events",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

                # Case 2: Vector similarity search
                elif search_method == "embedding":
                    # Generate or use provided embedding
                    if embedded_text is None:
                        import numpy as np

                        from mirix.constants import MAX_EMBEDDING_DIM
                        from mirix.embeddings import embedding_model

                        embedded_text = embedding_model(agent_state.embedding_config).get_text_embedding(query)
                        embedded_text = np.array(embedded_text)
                        embedded_text = np.pad(
                            embedded_text,
                            (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                            mode="constant",
                        ).tolist()

                    # Determine vector field
                    vector_field = f"{search_field}_embedding" if search_field else "details_embedding"

                    results = redis_client.search_vector(
                        index_name=redis_client.EPISODIC_INDEX,
                        embedding=embedded_text,
                        vector_field=vector_field,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis vector search HIT: found %d episodic events",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

                # Case 3: Full-text search (BM25-like)
                elif search_method in ["bm25", "string_match"]:
                    # Determine search field
                    fields = [search_field] if search_field else ["details", "summary"]

                    results = redis_client.search_text(
                        index_name=redis_client.EPISODIC_INDEX,
                        query=query,
                        search_fields=fields,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis text search HIT: found %d episodic events",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

            except Exception as e:
                logger.warning(
                    "Redis search failed for episodic memory, falling back to PostgreSQL: %s",
                    e,
                )
                # Fall through to PostgreSQL

        # Log when bypassing cache or Redis unavailable
        if not use_cache:
            logger.debug("Bypassing Redis cache (use_cache=False), querying PostgreSQL directly for episodic memory")
        elif not redis_client:
            logger.debug("Redis unavailable, querying PostgreSQL directly for episodic memory")

        # Original PostgreSQL implementation (unchanged - serves as fallback)
        with self.session_maker() as session:
            # TODO: handle the case where query is None, we need to extract the 50 most recent results

            if query == "":
                query_stmt = (
                    select(EpisodicEvent)
                    .where(EpisodicEvent.user_id == user.id)
                    .where(EpisodicEvent.organization_id == organization_id)
                    .order_by(EpisodicEvent.occurred_at.desc())
                )

                # Apply filter_tags if provided (e.g., {"scope": "CARE"})
                # This allows clients to filter memories by custom tags for access control
                if filter_tags:
                    for key, value in filter_tags.items():
                        query_stmt = query_stmt.where(EpisodicEvent.filter_tags[key].as_string() == str(value))

                # Apply temporal filtering if provided
                if start_date is not None:
                    query_stmt = query_stmt.where(EpisodicEvent.occurred_at >= start_date)
                if end_date is not None:
                    query_stmt = query_stmt.where(EpisodicEvent.occurred_at <= end_date)

                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = session.execute(query_stmt)
                episodic_memory = result.scalars().all()
                return [event.to_pydantic() for event in episodic_memory]

            else:
                base_query = (
                    select(
                        EpisodicEvent.id.label("id"),
                        EpisodicEvent.created_at.label("created_at"),
                        EpisodicEvent.occurred_at.label("occurred_at"),
                        EpisodicEvent.actor.label("actor"),
                        EpisodicEvent.event_type.label("event_type"),
                        EpisodicEvent.summary.label("summary"),
                        EpisodicEvent.details.label("details"),
                        EpisodicEvent.summary_embedding.label("summary_embedding"),
                        EpisodicEvent.details_embedding.label("details_embedding"),
                        EpisodicEvent.embedding_config.label("embedding_config"),
                        EpisodicEvent.organization_id.label("organization_id"),
                        EpisodicEvent.last_modify.label("last_modify"),
                        EpisodicEvent.user_id.label("user_id"),
                        EpisodicEvent.agent_id.label("agent_id"),
                    )
                    .where(EpisodicEvent.user_id == user.id)
                    .where(EpisodicEvent.organization_id == organization_id)
                )

                # Apply filter_tags if provided (e.g., {"scope": "CARE"})
                # This allows clients to filter memories by custom tags for access control
                if filter_tags:
                    for key, value in filter_tags.items():
                        base_query = base_query.where(EpisodicEvent.filter_tags[key].as_string() == str(value))

                # Apply temporal filtering if provided
                if start_date is not None:
                    base_query = base_query.where(EpisodicEvent.occurred_at >= start_date)
                if end_date is not None:
                    base_query = base_query.where(EpisodicEvent.occurred_at <= end_date)

                if search_method == "embedding":
                    embed_query = True
                    embedding_config = agent_state.embedding_config

                    main_query = build_query(
                        base_query=base_query,
                        query_text=query,
                        embedded_text=embedded_text,
                        embed_query=embed_query,
                        embedding_config=embedding_config,
                        search_field=eval("EpisodicEvent." + search_field + "_embedding"),
                        target_class=EpisodicEvent,
                        similarity_threshold=similarity_threshold,
                    )

                elif search_method == "string_match":
                    search_field = eval("EpisodicEvent." + search_field)
                    main_query = base_query.where(func.lower(search_field).contains(query.lower()))

                elif search_method == "bm25":
                    # Check if we're using PostgreSQL - use native full-text search if available
                    if settings.mirix_pg_uri_no_default:
                        # Use PostgreSQL's native full-text search with ts_rank for BM25-like functionality
                        return self._postgresql_fulltext_search(
                            session,
                            base_query,
                            query,
                            search_field,
                            limit,
                            user,
                            filter_tags=filter_tags,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    else:
                        # Fallback to in-memory BM25 for SQLite (legacy method)
                        # Load all candidate events (memory-intensive, kept for compatibility)
                        result = session.execute(select(EpisodicEvent).where(EpisodicEvent.user_id == user.id))
                        all_events = result.scalars().all()

                        # Apply temporal filtering in memory for SQLite
                        if start_date is not None:
                            all_events = [e for e in all_events if e.occurred_at and e.occurred_at >= start_date]
                        if end_date is not None:
                            all_events = [e for e in all_events if e.occurred_at and e.occurred_at <= end_date]

                        if not all_events:
                            return []

                        # Prepare documents for BM25
                        documents = []
                        valid_events = []

                        for event in all_events:
                            # Determine which field to use for search
                            if search_field and hasattr(event, search_field):
                                text_to_search = getattr(event, search_field) or ""
                            else:
                                text_to_search = event.summary or ""

                            # Preprocess the text into tokens
                            tokens = self._preprocess_text_for_bm25(text_to_search)

                            # Only include events that have tokens after preprocessing
                            if tokens:
                                documents.append(tokens)
                                valid_events.append(event)

                        if not documents:
                            return []

                        # Initialize BM25 with the documents
                        bm25 = BM25Okapi(documents)

                        # Preprocess the query
                        query_tokens = self._preprocess_text_for_bm25(query)

                        if not query_tokens:
                            # If query has no valid tokens, return most recent events
                            return [event.to_pydantic() for event in valid_events[:limit]]

                        # Get BM25 scores for all documents
                        scores = bm25.get_scores(query_tokens)

                        # Create scored events list
                        scored_events = list(zip(scores, valid_events))

                        # Sort by BM25 score in descending order
                        scored_events.sort(key=lambda x: x[0], reverse=True)

                        # Get top events based on limit
                        top_events = [event for score, event in scored_events[:limit]]
                        episodic_memory = top_events

                        # Return the list after converting to Pydantic
                        return [event.to_pydantic() for event in episodic_memory]

                elif search_method == "fuzzy_match":
                    # Load all candidate events (kept for backward compatibility)
                    result = session.execute(select(EpisodicEvent).where(EpisodicEvent.user_id == user.id))
                    all_events = result.scalars().all()
                    scored_events = []
                    for event in all_events:
                        # Determine which field to use:
                        # 1. If a search_field is provided (like "summary" or "details") use that.
                        # 2. Otherwise, fallback to the summary.
                        if search_field and hasattr(event, search_field):
                            text_to_search = getattr(event, search_field)
                        else:
                            text_to_search = event.summary

                        # Use fuzz.partial_ratio for short query matching against long text.
                        score = fuzz.partial_ratio(query.lower(), text_to_search.lower())
                        scored_events.append((score, event))

                    # Sort events in descending order of fuzzy match score.
                    scored_events.sort(key=lambda x: x[0], reverse=True)
                    # Optionally, you could filter out results below a certain score threshold.
                    top_events = [event for score, event in scored_events[:limit]]
                    episodic_memory = top_events
                    # Return the list after converting to Pydantic.
                    return [event.to_pydantic() for event in episodic_memory]

                if limit:
                    main_query = main_query.limit(limit)

                results = list(session.execute(main_query))

                episodic_memory = []
                for row in results:
                    data = dict(row._mapping)
                    episodic_memory.append(EpisodicEvent(**data))

                return [event.to_pydantic() for event in episodic_memory]

    def _postgresql_fulltext_search(
        self,
        session,
        base_query,
        query_text,
        search_field,
        limit,
        user,
        filter_tags=None,
        start_date=None,
        end_date=None,
    ):
        """
        Efficient PostgreSQL-native full-text search using ts_rank for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.

        Args:
            session: Database session
            base_query: Base SQLAlchemy query (not used in raw SQL, but kept for API compatibility)
            query_text: Search query string
            search_field: Field to search in ('summary', 'details', 'actor', 'event_type', etc.)
            limit: Maximum number of results to return
            user: User who owns the memories
            filter_tags: Optional dict of tag key-value pairs to filter by (e.g., {"scope": "CARE"})
            start_date: Optional start datetime for temporal filtering
            end_date: Optional end datetime for temporal filtering

        Returns:
            List of EpisodicEvent objects ranked by relevance

        Note:
            This function builds raw SQL queries for performance. It must explicitly include
            filter_tags in the WHERE clause to ensure proper tag-based filtering, as these
            filters are not applied from base_query.
        """
        from sqlalchemy import func

        # Clean and prepare the search query
        cleaned_query = self._clean_text_for_search(query_text)
        if not cleaned_query.strip():
            return []

        # Split into words and create a tsquery - PostgreSQL will handle the ranking
        query_words = [word.strip() for word in cleaned_query.split() if word.strip()]
        if not query_words:
            return []

        # Create tsquery string with improved logic:
        # 1. Use AND for multiple words when they form a meaningful phrase
        # 2. Use OR for broader matching when words seem unrelated
        # 3. Add prefix matching for partial word matches
        tsquery_parts = []
        for word in query_words:
            # Escape special characters for tsquery
            escaped_word = word.replace("'", "''").replace("&", "").replace("|", "").replace("!", "").replace(":", "")
            if escaped_word and len(escaped_word) > 1:  # Skip very short words
                # Add both exact and prefix matching for better results
                if len(escaped_word) >= 3:
                    tsquery_parts.append(f"('{escaped_word}' | '{escaped_word}':*)")
                else:
                    tsquery_parts.append(f"'{escaped_word}'")

        if not tsquery_parts:
            return []

        # Use AND logic for multiple terms to find more relevant documents
        # but fallback to OR if AND produces no results
        if len(tsquery_parts) > 1:
            tsquery_string_and = " & ".join(tsquery_parts)  # AND logic for precision
            tsquery_string_or = " | ".join(tsquery_parts)  # OR logic for recall
        else:
            tsquery_string_and = tsquery_string_or = tsquery_parts[0]

        # Build the PostgreSQL full-text search query using raw SQL with proper parameterization
        # This avoids the TextClause.op() issue and is more efficient

        # Determine which field to search based on search_field
        if search_field == "summary":
            tsvector_sql = "to_tsvector('english', coalesce(summary, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(summary, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "details":
            tsvector_sql = "to_tsvector('english', coalesce(details, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(details, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "actor":
            tsvector_sql = "to_tsvector('english', coalesce(actor, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(actor, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "event_type":
            tsvector_sql = "to_tsvector('english', coalesce(event_type, ''))"
            rank_sql = (
                "ts_rank_cd(to_tsvector('english', coalesce(event_type, '')), to_tsquery('english', :tsquery), 32)"
            )
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(summary, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(details, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(actor, '')), 'C') ||
                             setweight(to_tsvector('english', coalesce(event_type, '')), 'D')"""
            rank_sql = """ts_rank_cd(
                setweight(to_tsvector('english', coalesce(summary, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(details, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(actor, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(event_type, '')), 'D'),
                to_tsquery('english', :tsquery), 32)"""

        # Build WHERE clause with temporal filtering
        # Note: Must explicitly include all filters here since we're building raw SQL
        # The base_query parameter is not used, but kept for API compatibility
        where_clauses = [
            f"{tsvector_sql} @@ to_tsquery('english', :tsquery)",
            "user_id = :user_id",
        ]
        query_params = {
            "tsquery": tsquery_string_and,
            "user_id": user.id,
            "limit_val": limit or 50,
        }

        # Add filter_tags filtering (e.g., {"scope": "CARE"})
        # This allows clients to filter memories by custom tags for access control
        # CRITICAL: Without this filter, searches return 0 results when filter_tags are provided
        if filter_tags:
            for key, value in filter_tags.items():
                # Use JSONB operator to filter by tag key-value pairs
                where_clauses.append(f"filter_tags->>'{key}' = :filter_tag_{key}")
                query_params[f"filter_tag_{key}"] = str(value)

        # Add temporal filtering if provided
        if start_date is not None:
            where_clauses.append("occurred_at >= :start_date")
            query_params["start_date"] = start_date
        if end_date is not None:
            where_clauses.append("occurred_at <= :end_date")
            query_params["end_date"] = end_date

        where_clause = " AND ".join(where_clauses)

        # Try AND query first for more precise results
        try:
            and_query_sql = text(
                f"""
                SELECT 
                    id, created_at, occurred_at, actor, event_type,
                    summary, details, summary_embedding, details_embedding,
                    embedding_config, organization_id, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM episodic_memory 
                WHERE {where_clause}
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """
            )

            results = list(session.execute(and_query_sql, query_params))

            # If AND query returns sufficient results, use them
            if len(results) >= min(limit or 10, 10):
                episodic_memory = []
                for row in results:
                    data = dict(row._mapping)
                    # Remove the rank_score field before creating the object
                    data.pop("rank_score", None)

                    # Parse JSON fields that are returned as strings from raw SQL
                    json_fields = ["last_modify", "embedding_config"]
                    for field in json_fields:
                        if field in data and isinstance(data[field], str):
                            try:
                                data[field] = json.loads(data[field])
                            except (json.JSONDecodeError, TypeError):
                                pass

                    # Parse embedding fields
                    embedding_fields = ["summary_embedding", "details_embedding"]
                    for field in embedding_fields:
                        if field in data and data[field] is not None:
                            data[field] = self._parse_embedding_field(data[field])

                    episodic_memory.append(EpisodicEvent(**data))

                return [event.to_pydantic() for event in episodic_memory]

        except Exception as e:
            logger.debug("PostgreSQL AND query error: %s", e)

        # If AND query fails or returns too few results, try OR query
        try:
            # Update query params for OR query
            or_query_params = query_params.copy()
            or_query_params["tsquery"] = tsquery_string_or

            or_query_sql = text(
                f"""
                SELECT 
                    id, created_at, occurred_at, actor, event_type,
                    summary, details, summary_embedding, details_embedding,
                    embedding_config, organization_id, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM episodic_memory 
                WHERE {where_clause}
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """
            )

            results = session.execute(or_query_sql, or_query_params)

            episodic_memory = []
            for row in results:
                data = dict(row._mapping)
                # Remove the rank_score field before creating the object
                data.pop("rank_score", None)

                # Parse JSON fields that are returned as strings from raw SQL
                json_fields = ["last_modify", "embedding_config"]
                for field in json_fields:
                    if field in data and isinstance(data[field], str):
                        try:
                            data[field] = json.loads(data[field])
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Parse embedding fields
                embedding_fields = ["summary_embedding", "details_embedding"]
                for field in embedding_fields:
                    if field in data and data[field] is not None:
                        data[field] = self._parse_embedding_field(data[field])

                episodic_memory.append(EpisodicEvent(**data))

            return [event.to_pydantic() for event in episodic_memory]

        except Exception as e:
            # If there's an error with the tsquery (e.g., invalid syntax), fall back to simpler search
            logger.debug("PostgreSQL full-text search error: %s", e)
            # Fall back to simple ILIKE search
            fallback_field = (
                getattr(EpisodicEvent, search_field)
                if search_field and hasattr(EpisodicEvent, search_field)
                else EpisodicEvent.summary
            )
            fallback_query = base_query.where(func.lower(fallback_field).contains(query_text.lower())).order_by(
                EpisodicEvent.created_at.desc()
            )

            if limit:
                fallback_query = fallback_query.limit(limit)

            results = session.execute(fallback_query)
            episodic_memory = [EpisodicEvent(**dict(row._mapping)) for row in results]
            return [event.to_pydantic() for event in episodic_memory]

    def update_event(
        self,
        event_id: str = None,
        new_summary: str = None,
        new_details: str = None,
        actor: PydanticClient = None,
        agent_state: AgentState = None,
        update_mode: str = "append",
    ):
        """
        Update the selected events

        Args:
            event_id: ID of the episodic event to update
            new_summary: New summary text (will overwrite existing summary)
            new_details: New details text
            actor: Client performing the update (for access control and audit trail)
            agent_state: Agent state containing embedding configuration (needed for embedding regeneration)
            update_mode: How to handle new_details - "append" (default) appends to existing,
                        "replace" overwrites existing details entirely
        """

        with self.session_maker() as session:
            # Use the passed actor directly for access control
            # EpisodicEvent.read() uses organization_id from actor for filtering
            selected_event = EpisodicEvent.read(db_session=session, identifier=event_id, actor=actor)

            if not selected_event:
                raise ValueError(f"Episodic episodic_memory record with id {event_id} not found.")

            operations = []
            if new_summary:
                selected_event.summary = new_summary
                operations.append("summary_updated")
            if new_details:
                if update_mode == "replace":
                    selected_event.details = new_details
                else:
                    selected_event.details += "\n" + new_details
                operations.append("details_updated")

            if BUILD_EMBEDDINGS_FOR_MEMORY and agent_state is not None and (new_summary or new_details):
                embed_model = embedding_model(agent_state.embedding_config)

                # Use Pydantic validator to pad embeddings (single source of truth)
                selected_event.summary_embedding = PydanticEpisodicEvent.pad_embeddings(
                    embed_model.get_text_embedding(selected_event.summary)
                )
                selected_event.details_embedding = PydanticEpisodicEvent.pad_embeddings(
                    embed_model.get_text_embedding(selected_event.details)
                )
                selected_event.embedding_config = agent_state.embedding_config

            # Update last_modify field with timestamp and operation info
            selected_event.last_modify = {
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "operation": ", ".join(operations) if operations else "updated",
            }

            selected_event.update_with_redis(session, actor=actor)  # Updates Redis JSON cache
            return selected_event.to_pydantic()

    def _parse_embedding_field(self, embedding_value):
        """
        Helper method to parse embedding field from different PostgreSQL return formats.

        Args:
            embedding_value: The raw embedding value from PostgreSQL query

        Returns:
            List of floats or None if parsing fails
        """
        if embedding_value is None:
            return None

        try:
            # If it's already a list or tuple, convert to list
            if isinstance(embedding_value, (list, tuple)):
                return list(embedding_value)

            # If it's a string, try different parsing approaches
            if isinstance(embedding_value, str):
                # Remove any whitespace
                embedding_value = embedding_value.strip()

                # Check if it's a JSON array string: "[-0.006639634,-0.0114432...]"
                if embedding_value.startswith("[") and embedding_value.endswith("]"):
                    try:
                        return json.loads(embedding_value)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try manual parsing
                        # Remove brackets and split by comma
                        inner = embedding_value[1:-1]  # Remove [ and ]
                        return [float(x.strip()) for x in inner.split(",") if x.strip()]

                # Try comma-separated values
                if "," in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split(",") if x.strip()]

                # Try space-separated values
                if " " in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split() if x.strip()]

            # Try using the original deserialize_vector approach for binary data
            try:
                from mirix.helpers.converters import deserialize_vector

                class MockDialect:
                    name = "postgresql"

                return deserialize_vector(embedding_value, MockDialect())
            except Exception:
                pass

            # If all else fails, return None to avoid validation errors
            return None

        except Exception as e:
            logger.debug("Warning: Failed to parse embedding field: %s", e)
            return None

    @update_timezone
    @enforce_types
    def list_episodic_memory_by_org(
        self,
        agent_state: AgentState,
        organization_id: str,
        query: str = "",
        embedded_text: Optional[List[float]] = None,
        search_field: str = "",
        search_method: str = "embedding",
        limit: Optional[int] = 50,
        timezone_str: str = None,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticEpisodicEvent]:
        """
        List episodic events across ALL users in an organization.
        Filtered by organization_id and filter_tags (including scope).

        Args:
            agent_state: Agent state containing embedding configuration
            organization_id: Organization ID to filter by
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('summary', 'details', etc.)
            search_method: Search method ('embedding', 'bm25', 'string_match', etc.)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            filter_tags: Filter tags dict (should include "scope": client.scope)
            use_cache: If True, try Redis cache first
            start_date: Optional start datetime for filtering by occurred_at
            end_date: Optional end datetime for filtering by occurred_at

        Returns:
            List of episodic events matching org_id and filter_tags["scope"]
        """

        # Try Redis Search first (if cache enabled and Redis is available)
        from mirix.database.redis_client import get_redis_client

        redis_client = get_redis_client()

        if use_cache and redis_client:
            try:
                # Case 1: No query - get recent items
                if not query or query == "":
                    results = redis_client.search_recent_by_org(
                        index_name=redis_client.EPISODIC_INDEX,
                        limit=limit or 50,
                        organization_id=organization_id,
                        sort_by="occurred_at_ts",
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis cache HIT: returned %d recent episodic events for org %s",
                            len(results),
                            organization_id,
                        )
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

                # Case 2: Vector similarity search
                elif search_method == "embedding":
                    # Generate or use provided embedding
                    if embedded_text is None:
                        import numpy as np

                        from mirix.constants import MAX_EMBEDDING_DIM
                        from mirix.embeddings import embedding_model

                        embedded_text = embedding_model(agent_state.embedding_config).get_text_embedding(query)
                        embedded_text = np.array(embedded_text)
                        embedded_text = np.pad(
                            embedded_text,
                            (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                            mode="constant",
                        ).tolist()

                    # Determine vector field
                    vector_field = f"{search_field}_embedding" if search_field else "details_embedding"

                    results = redis_client.search_vector_by_org(
                        index_name=redis_client.EPISODIC_INDEX,
                        embedding=embedded_text,
                        vector_field=vector_field,
                        limit=limit or 50,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis vector search HIT: %d results for org %s",
                            len(results),
                            organization_id,
                        )
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

                # Case 3: Text search (BM25, string match, etc.)
                else:
                    results = redis_client.search_text_by_org(
                        index_name=redis_client.EPISODIC_INDEX,
                        query_text=query,
                        search_field=search_field or "details",
                        search_method=search_method,
                        limit=limit or 50,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if results:
                        logger.debug(
                            "Redis text search HIT: %d results for org %s",
                            len(results),
                            organization_id,
                        )
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticEpisodicEvent(**item) for item in results]

            except Exception as e:
                logger.warning("Redis search failed for org %s: %s", organization_id, e)

        # Fallback to PostgreSQL
        logger.debug(
            "Redis cache MISS or disabled - falling back to PostgreSQL for org %s",
            organization_id,
        )

        with self.session_maker() as session:
            # Base query filtering by organization_id instead of user_id
            # Return full EpisodicEvent objects, not individual columns
            base_query = select(EpisodicEvent).where(EpisodicEvent.organization_id == organization_id)

            # Apply filter_tags (INCLUDING SCOPE)
            # For scope: check if input value is contained in memory's scope
            # For other keys: exact match
            if filter_tags:
                from sqlalchemy import func, or_

                for key, value in filter_tags.items():
                    if key == "scope":
                        # Scope matching: input value must be in memory's scope field
                        base_query = base_query.where(
                            or_(
                                func.lower(EpisodicEvent.filter_tags[key].as_string()).contains(str(value).lower()),
                                EpisodicEvent.filter_tags[key].as_string() == str(value),
                            )
                        )
                    else:
                        # Other keys: exact match
                        base_query = base_query.where(EpisodicEvent.filter_tags[key].as_string() == str(value))

            # Apply temporal filtering if provided
            if start_date is not None:
                base_query = base_query.where(EpisodicEvent.occurred_at >= start_date)
            if end_date is not None:
                base_query = base_query.where(EpisodicEvent.occurred_at <= end_date)

            # Handle empty query - fall back to recent sort
            if not query or query == "":
                base_query = base_query.order_by(EpisodicEvent.occurred_at.desc())
                if limit:
                    base_query = base_query.limit(limit)
                result = session.execute(base_query)
                episodic_memory = result.scalars().all()
                return [event.to_pydantic() for event in episodic_memory]

            if search_method == "embedding":
                embed_query = True
                embedding_config = agent_state.embedding_config

                # Use provided embedding or generate it
                if embedded_text is None:
                    import numpy as np

                    from mirix.constants import MAX_EMBEDDING_DIM
                    from mirix.embeddings import embedding_model

                    embedded_text = embedding_model(embedding_config).get_text_embedding(query)
                    embedded_text = np.array(embedded_text)
                    embedded_text = np.pad(
                        embedded_text,
                        (0, MAX_EMBEDDING_DIM - embedded_text.shape[0]),
                        mode="constant",
                    ).tolist()

                # Determine which embedding field to search
                if not search_field or search_field == "details":
                    embedding_field = EpisodicEvent.details_embedding
                elif search_field == "summary":
                    embedding_field = EpisodicEvent.summary_embedding
                else:
                    embedding_field = EpisodicEvent.details_embedding

                embedding_query_field = embedding_field.cosine_distance(embedded_text).label("distance")

                base_query = base_query.add_columns(embedding_query_field)

                # Apply similarity threshold if provided
                if similarity_threshold is not None:
                    base_query = base_query.where(embedding_query_field < similarity_threshold)

                base_query = base_query.order_by(embedding_query_field)
            elif search_method == "bm25":
                # Use PostgreSQL native full-text search if available
                from sqlalchemy import func, text

                # Determine search field
                if not search_field or search_field == "details":
                    text_field = EpisodicEvent.details
                elif search_field == "summary":
                    text_field = EpisodicEvent.summary
                else:
                    text_field = EpisodicEvent.details

                # Use ts_rank_cd for BM25-like scoring
                tsquery = func.plainto_tsquery("english", query)
                tsvector = func.to_tsvector("english", text_field)
                rank = func.ts_rank_cd(tsvector, tsquery).label("rank")

                base_query = base_query.add_columns(rank).where(tsvector.op("@@")(tsquery)).order_by(rank.desc())

            if limit:
                base_query = base_query.limit(limit)

            result = session.execute(base_query)
            episodic_memory = result.scalars().all()
            return [event.to_pydantic() for event in episodic_memory]
