import json
import re
import string
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sqlalchemy import func, select, text

from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY
from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.orm.semantic_memory import SemanticMemoryItem
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.semantic_memory import SemanticMemoryItem as PydanticSemanticMemoryItem
from mirix.schemas.semantic_memory import (
    SemanticMemoryItemUpdate,
)
from mirix.schemas.user import User as PydanticUser
from mirix.services.utils import build_query, update_timezone
from mirix.settings import settings
from mirix.utils import enforce_types, generate_unique_short_id

logger = get_logger(__name__)


class SemanticMemoryManager:
    """Manager class to handle business logic related to Semantic Memory Items."""

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

    def _count_word_matches(self, item_data: Dict[str, Any], query_words: List[str], search_field: str = "") -> int:
        """
        Count how many of the query words are present in the semantic memory item data.

        Args:
            item_data: Dictionary containing semantic memory item data
            query_words: List of query words to search for
            search_field: Specific field to search in, or empty string to search all text fields

        Returns:
            Number of query words found in the item
        """
        if not query_words:
            return 0

        # Determine which text fields to search in
        if search_field == "name":
            search_texts = [item_data.get("name", "")]
        elif search_field == "summary":
            search_texts = [item_data.get("summary", "")]
        elif search_field == "details":
            search_texts = [item_data.get("details", "")]
        elif search_field == "source":
            search_texts = [item_data.get("source", "")]
        else:
            # Search across all relevant text fields
            search_texts = [
                item_data.get("name", ""),
                item_data.get("summary", ""),
                item_data.get("details", ""),
                item_data.get("source", ""),
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

    def _postgresql_fulltext_search(
        self,
        session,
        base_query,
        query_text,
        search_field,
        limit,
        user_id,
        filter_tags=None,
    ):
        """
        Efficient PostgreSQL-native full-text search using ts_rank_cd for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.

        Args:
            session: Database session
            base_query: Base SQLAlchemy query (not used, kept for API compatibility)
            query_text: Search query string
            search_field: Field to search in ('name', 'summary', 'details', 'source', etc.)
            limit: Maximum number of results to return
            user_id: User ID to filter by
            filter_tags: Optional dict of tag key-value pairs to filter by (e.g., {"scope": "CARE"})

        Returns:
            List of SemanticMemoryItem objects ranked by relevance
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

        # Create tsquery string with improved logic
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

        # Determine which field to search based on search_field
        if search_field == "name":
            tsvector_sql = "to_tsvector('english', coalesce(name, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(name, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "summary":
            tsvector_sql = "to_tsvector('english', coalesce(summary, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(summary, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "details":
            tsvector_sql = "to_tsvector('english', coalesce(details, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(details, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == "source":
            tsvector_sql = "to_tsvector('english', coalesce(source, ''))"
            rank_sql = "ts_rank_cd(to_tsvector('english', coalesce(source, '')), to_tsquery('english', :tsquery), 32)"
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(details, '')), 'C') ||
                             setweight(to_tsvector('english', coalesce(source, '')), 'D')"""
            rank_sql = """ts_rank_cd(
                setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(details, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(source, '')), 'D'),
                to_tsquery('english', :tsquery), 32)"""

        # Build WHERE clauses dynamically
        where_clauses = [
            f"{tsvector_sql} @@ to_tsquery('english', :tsquery)",
            "user_id = :user_id",
        ]
        query_params = {
            "tsquery": tsquery_string_and,
            "user_id": user_id,
            "limit_val": limit or 50,
        }

        # Add filter_tags filtering (e.g., {"scope": "CARE"})
        if filter_tags:
            for key, value in filter_tags.items():
                where_clauses.append(f"filter_tags->>'{key}' = :filter_tag_{key}")
                query_params[f"filter_tag_{key}"] = str(value)

        where_clause = " AND ".join(where_clauses)

        # Try AND query first for more precise results
        try:
            and_query_sql = text(
                f"""
                SELECT 
                    id, created_at, name, summary, details, source,
                    name_embedding, summary_embedding, details_embedding, embedding_config,
                    organization_id, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM semantic_memory 
                WHERE {where_clause}
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """
            )

            results = list(session.execute(and_query_sql, query_params))

            # If AND query returns sufficient results, use them
            if len(results) >= min(limit or 10, 10):
                semantic_items = []
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
                    embedding_fields = [
                        "name_embedding",
                        "summary_embedding",
                        "details_embedding",
                    ]
                    for field in embedding_fields:
                        if field in data and data[field] is not None:
                            data[field] = self._parse_embedding_field(data[field])

                    semantic_items.append(SemanticMemoryItem(**data))

                return [item.to_pydantic() for item in semantic_items]

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
                    id, created_at, name, summary, details, source,
                    name_embedding, summary_embedding, details_embedding, embedding_config,
                    organization_id, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM semantic_memory 
                WHERE {where_clause}
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """
            )

            results = session.execute(or_query_sql, or_query_params)

            semantic_items = []
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
                embedding_fields = [
                    "name_embedding",
                    "summary_embedding",
                    "details_embedding",
                ]
                for field in embedding_fields:
                    if field in data and data[field] is not None:
                        data[field] = self._parse_embedding_field(data[field])

                semantic_items.append(SemanticMemoryItem(**data))

            return [item.to_pydantic() for item in semantic_items]

        except Exception as e:
            # If there's an error with the tsquery, fall back to simpler search
            logger.debug("PostgreSQL full-text search error: %s", e)
            # Fall back to simple ILIKE search
            fallback_field = (
                getattr(SemanticMemoryItem, search_field)
                if search_field and hasattr(SemanticMemoryItem, search_field)
                else SemanticMemoryItem.name
            )
            fallback_query = base_query.where(func.lower(fallback_field).contains(query_text.lower())).order_by(
                SemanticMemoryItem.created_at.desc()
            )

            if limit:
                fallback_query = fallback_query.limit(limit)

            results = session.execute(fallback_query)
            semantic_items = [SemanticMemoryItem(**dict(row._mapping)) for row in results]
            return [item.to_pydantic() for item in semantic_items]

    @update_timezone
    @enforce_types
    def get_semantic_item_by_id(
        self, semantic_memory_id: str, user: PydanticUser, timezone_str: str
    ) -> Optional[PydanticSemanticMemoryItem]:
        """Fetch a semantic memory item by ID (with Redis JSON caching)."""
        # Try Redis cache first (JSON-based for memory tables)
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.SEMANTIC_PREFIX}{semantic_memory_id}"
                cached_data = redis_client.get_json(redis_key)
                if cached_data:
                    logger.debug("Redis cache HIT for semantic memory %s", semantic_memory_id)
                    return PydanticSemanticMemoryItem(**cached_data)
        except Exception as e:
            logger.warning(
                "Redis cache read failed for semantic memory %s: %s",
                semantic_memory_id,
                e,
            )

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                semantic_memory_item = SemanticMemoryItem.read(
                    db_session=session, identifier=semantic_memory_id, user=user
                )
                pydantic_item = semantic_memory_item.to_pydantic()

                # Populate Redis cache for next time
                try:
                    if redis_client:
                        data = pydantic_item.model_dump(mode="json")
                        # model_dump(mode='json') already converts datetime to ISO format strings
                        redis_client.set_json(redis_key, data, ttl=settings.redis_ttl_default)
                        logger.debug(
                            "Populated Redis cache for semantic memory %s",
                            semantic_memory_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to populate Redis cache for semantic memory %s: %s",
                        semantic_memory_id,
                        e,
                    )

                return pydantic_item
            except NoResultFound:
                raise NoResultFound(f"Semantic memory item with id {semantic_memory_id} not found.")

    @update_timezone
    @enforce_types
    def get_most_recently_updated_item(
        self, user: PydanticUser, timezone_str: str = None
    ) -> Optional[PydanticSemanticMemoryItem]:
        """
        Fetch the most recently updated semantic memory item based on last_modify timestamp.
        Filter by user_id from actor.
        Returns None if no items exist.
        """
        with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import DateTime, cast, text

            query = select(SemanticMemoryItem).order_by(
                cast(text("semantic_memory.last_modify ->> 'timestamp'"), DateTime).desc()
            )

            # Filter by user_id for multi-user support
            query = query.where(SemanticMemoryItem.user_id == user.id)

            result = session.execute(query.limit(1))
            item = result.scalar_one_or_none()

            return [item.to_pydantic()] if item else None

    @enforce_types
    def create_item(
        self,
        item_data: PydanticSemanticMemoryItem,
        actor: PydanticClient,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> PydanticSemanticMemoryItem:
        """Create a new semantic memory item.

        Args:
            item_data: The semantic memory data to create
            actor: Client performing the operation (for audit trail)
            client_id: Client application identifier (defaults to actor.id)
            user_id: End-user identifier (optional)
            use_cache: If True, cache in Redis. If False, skip caching.
        """

        # Backward compatibility
        if client_id is None:
            client_id = actor.id

        # Ensure ID is set before model_dump
        if not item_data.id:
            item_data.id = generate_unique_short_id(self.session_maker, SemanticMemoryItem, "sem")

        data_dict = item_data.model_dump()

        # Validate required fields
        required_fields = ["summary", "name"]
        for field in required_fields:
            if field not in data_dict or not data_dict[field]:
                raise ValueError(f"Required field '{field}' missing from semantic memory data")

        # Set client_id and user_id on the memory
        data_dict["client_id"] = client_id
        data_dict["user_id"] = user_id

        logger.debug("create_item: client_id=%s, user_id=%s", client_id, user_id)

        with self.session_maker() as session:
            item = SemanticMemoryItem(**data_dict)
            item.create_with_redis(session, actor=actor, use_cache=use_cache)
            return item.to_pydantic()

    @enforce_types
    def update_item(
        self,
        item_update: SemanticMemoryItemUpdate,
        user: PydanticUser,
        actor: PydanticClient,
    ) -> PydanticSemanticMemoryItem:
        """Update an existing semantic memory item."""
        with self.session_maker() as session:
            item = SemanticMemoryItem.read(db_session=session, identifier=item_update.id, user=user)
            update_data = item_update.model_dump(exclude_unset=True)
            for k, v in update_data.items():
                if k not in [
                    "id",
                    "updated_at",
                ]:  # Exclude updated_at - handled by update() method
                    setattr(item, k, v)
            # updated_at is automatically set to current UTC time by item.update()
            item.update_with_redis(session, actor=actor)  # Updates Redis JSON cache
            return item.to_pydantic()

    @enforce_types
    def create_many_items(
        self,
        items: List[PydanticSemanticMemoryItem],
        user: PydanticUser,
    ) -> List[PydanticSemanticMemoryItem]:
        """Create multiple semantic memory items."""
        return [self.create_item(i, user) for i in items]

    def get_total_number_of_items(self, user: PydanticUser) -> int:
        """Get the total number of items in the semantic memory for the user."""
        with self.session_maker() as session:
            query = select(func.count(SemanticMemoryItem.id)).where(SemanticMemoryItem.user_id == user.id)
            result = session.execute(query)
            return result.scalar_one()

    @update_timezone
    @enforce_types
    def list_semantic_items(
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
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticSemanticMemoryItem]:
        """
        List semantic memory items with various search methods.

        Args:
            agent_state: The agent state containing embedding configuration
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('name', 'summary', 'details', 'source')
            search_method: Search method to use:
                - 'embedding': Vector similarity search using embeddings
                - 'string_match': Simple string containment search
                - 'bm25': **RECOMMENDED** - PostgreSQL native full-text search (ts_rank_cd) when using PostgreSQL,
                               falls back to in-memory BM25 for SQLite
                - 'fuzzy_match': Fuzzy string matching (legacy, kept for compatibility)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            use_cache: If True, try Redis cache first. If False, skip cache and query PostgreSQL directly.

        Returns:
            List of semantic memory items matching the search criteria

        Note:
            **For PostgreSQL users**: 'bm25' is now the recommended method for text-based searches as it uses
            PostgreSQL's native full-text search with ts_rank_cd for BM25-like scoring. This is much more efficient
            than loading all documents into memory and leverages your existing GIN indexes.

            **For SQLite users**: 'bm25' now has fallback support that uses in-memory BM25 processing.

            Performance comparison:
            - PostgreSQL 'bm25': Native DB search, very fast, scales well
            - Fallback 'bm25' (SQLite): In-memory processing, slower for large datasets but still provides
              proper BM25 ranking
        """
        query = query.strip() if query else ""
        is_empty_query = not query or query == ""

        # Extract organization_id from user for multi-tenant isolation
        organization_id = user.organization_id

        # Try Redis Search first (if cache enabled and Redis is available)
        from mirix.database.redis_client import get_redis_client

        redis_client = get_redis_client()

        if use_cache and redis_client:
            try:
                # Case 1: No query - get recent items (regardless of search_method)
                if is_empty_query:
                    logger.debug(
                        "Searching Redis for recent semantic items with filter_tags=%s",
                        filter_tags,
                    )
                    results = redis_client.search_recent(
                        index_name=redis_client.SEMANTIC_INDEX,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        sort_by="created_at_ts",
                        filter_tags=filter_tags,
                    )
                    logger.debug(
                        "Redis search_recent returned %d results",
                        len(results) if results else 0,
                    )
                    if results:
                        logger.debug(
                            "Redis cache HIT: returned %d recent semantic items",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]
                    # If no results, fall through to PostgreSQL (don't return empty list)

                # Case 2: Vector similarity search
                elif search_method == "embedding":
                    if embedded_text is None:
                        from mirix.embeddings import embedding_model

                        embedded_text = embedding_model.embed_and_upload_batch([query], agent_state.embedding_config)[0]

                    vector_field = f"{search_field}_embedding" if search_field else "summary_embedding"

                    results = redis_client.search_vector(
                        index_name=redis_client.SEMANTIC_INDEX,
                        embedding=embedded_text,
                        vector_field=vector_field,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                    )
                    if results:
                        logger.debug(
                            "Redis vector search HIT: found %d semantic items",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]

                # Case 3: Full-text search
                elif search_method in ["bm25", "string_match"]:
                    fields = [search_field] if search_field else ["name", "summary", "details"]

                    results = redis_client.search_text(
                        index_name=redis_client.SEMANTIC_INDEX,
                        query=query,
                        search_fields=fields,
                        limit=limit or 50,
                        user_id=user.id,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                    )
                    if results:
                        logger.debug(
                            "Redis text search HIT: found %d semantic items",
                            len(results),
                        )
                        # Clean Redis-specific fields before Pydantic validation
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]

            except Exception as e:
                logger.warning(
                    "Redis search failed for semantic memory, falling back to PostgreSQL: %s",
                    e,
                )
                # Fall through to PostgreSQL

        # Log when bypassing cache or Redis unavailable
        if not use_cache:
            logger.debug("Bypassing Redis cache (use_cache=False), querying PostgreSQL directly for semantic memory")
        elif not redis_client:
            logger.debug("Redis unavailable, querying PostgreSQL directly for semantic memory")
        else:
            logger.debug("Redis returned no results, falling back to PostgreSQL for semantic memory")

        logger.debug("PostgreSQL fallback: query='%s', filter_tags=%s", query, filter_tags)
        with self.session_maker() as session:
            if query == "":
                # Use proper PostgreSQL JSON text extraction and casting for ordering
                from sqlalchemy import DateTime, cast, text

                query_stmt = (
                    select(SemanticMemoryItem)
                    .where(SemanticMemoryItem.user_id == user.id)
                    .where(SemanticMemoryItem.organization_id == organization_id)
                    .order_by(
                        cast(
                            text("semantic_memory.last_modify ->> 'timestamp'"),
                            DateTime,
                        ).desc()
                    )
                )

                # Apply filter_tags if provided
                if filter_tags:
                    for key, value in filter_tags.items():
                        query_stmt = query_stmt.where(SemanticMemoryItem.filter_tags[key].as_string() == str(value))

                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = session.execute(query_stmt)
                semantic_items = result.scalars().all()
                logger.debug(
                    "PostgreSQL returned %d semantic items (filter_tags=%s)",
                    len(semantic_items),
                    filter_tags,
                )
                return [item.to_pydantic() for item in semantic_items]

            else:
                base_query = (
                    select(
                        SemanticMemoryItem.id.label("id"),
                        SemanticMemoryItem.created_at.label("created_at"),
                        SemanticMemoryItem.name.label("name"),
                        SemanticMemoryItem.summary.label("summary"),
                        SemanticMemoryItem.details.label("details"),
                        SemanticMemoryItem.source.label("source"),
                        SemanticMemoryItem.name_embedding.label("name_embedding"),
                        SemanticMemoryItem.summary_embedding.label("summary_embedding"),
                        SemanticMemoryItem.details_embedding.label("details_embedding"),
                        SemanticMemoryItem.embedding_config.label("embedding_config"),
                        SemanticMemoryItem.organization_id.label("organization_id"),
                        SemanticMemoryItem.last_modify.label("last_modify"),
                        SemanticMemoryItem.user_id.label("user_id"),
                        SemanticMemoryItem.agent_id.label("agent_id"),
                    )
                    .where(SemanticMemoryItem.user_id == user.id)
                    .where(SemanticMemoryItem.organization_id == organization_id)
                )

                # Apply filter_tags if provided
                if filter_tags:
                    for key, value in filter_tags.items():
                        base_query = base_query.where(SemanticMemoryItem.filter_tags[key].as_string() == str(value))

                if search_method == "embedding":
                    embed_query = True
                    embedding_config = agent_state.embedding_config

                    main_query = build_query(
                        base_query=base_query,
                        query_text=query,
                        embedded_text=embedded_text,
                        embed_query=embed_query,
                        embedding_config=embedding_config,
                        search_field=eval("SemanticMemoryItem." + search_field + "_embedding"),
                        target_class=SemanticMemoryItem,
                        similarity_threshold=similarity_threshold,
                    )

                elif search_method == "string_match":
                    search_field = eval("SemanticMemoryItem." + search_field)
                    main_query = base_query.where(func.lower(search_field).contains(query.lower()))

                elif search_method == "bm25":
                    # Check if we're using PostgreSQL - use native full-text search if available
                    if settings.mirix_pg_uri_no_default:
                        # Use PostgreSQL native full-text search
                        return self._postgresql_fulltext_search(
                            session,
                            base_query,
                            query,
                            search_field,
                            limit,
                            user.id,
                            filter_tags=filter_tags,
                        )
                    else:
                        # Fallback to in-memory BM25 for SQLite (legacy method)
                        # Load all candidate items (memory-intensive, kept for compatibility)
                        result = session.execute(
                            select(SemanticMemoryItem).where(SemanticMemoryItem.user_id == user.id)
                        )
                        all_items = result.scalars().all()

                        if not all_items:
                            return []

                        # Prepare documents for BM25
                        documents = []
                        valid_items = []

                        for item in all_items:
                            # Determine which field to use for search
                            if search_field and hasattr(item, search_field):
                                text_to_search = getattr(item, search_field) or ""
                            else:
                                text_to_search = item.name or ""

                            # Preprocess the text into tokens
                            tokens = self._preprocess_text_for_bm25(text_to_search)

                            # Only include items that have tokens after preprocessing
                            if tokens:
                                documents.append(tokens)
                                valid_items.append(item)

                        if not documents:
                            return []

                        # Initialize BM25 with the documents
                        bm25 = BM25Okapi(documents)

                        # Preprocess the query
                        query_tokens = self._preprocess_text_for_bm25(query)

                        if not query_tokens:
                            # If query has no valid tokens, return most recent items
                            return [item.to_pydantic() for item in valid_items[:limit]]

                        # Get BM25 scores for all documents
                        scores = bm25.get_scores(query_tokens)

                        # Create scored items list
                        scored_items = list(zip(scores, valid_items))

                        # Sort by BM25 score in descending order
                        scored_items.sort(key=lambda x: x[0], reverse=True)

                        # Get top items based on limit
                        top_items = [item for score, item in scored_items[:limit]]
                        semantic_items = top_items

                        # Return the list after converting to Pydantic
                        return [item.to_pydantic() for item in semantic_items]

                elif search_method == "fuzzy_match":
                    # Fuzzy matching: load all candidate items into memory and compute a fuzzy match score.
                    result = session.execute(select(SemanticMemoryItem).where(SemanticMemoryItem.user_id == user.id))
                    all_items = result.scalars().all()
                    scored_items = []
                    for item in all_items:
                        # Determine which field to use:
                        # 1. If a search_field is provided (e.g., "concept" or "summary") and exists in the item, use it.
                        # 2. Otherwise, default to using the "concept" field.
                        if search_field and hasattr(item, search_field):
                            text_to_search = getattr(item, search_field)
                        else:
                            text_to_search = item.name
                        # Compute the fuzzy matching score using partial_ratio for better short-to-long matching.
                        score = fuzz.partial_ratio(query.lower(), text_to_search.lower())
                        scored_items.append((score, item))

                    # Sort items descending by score and pick the top ones.
                    scored_items.sort(key=lambda x: x[0], reverse=True)
                    top_items = [item for score, item in scored_items[:limit]]
                    return [item.to_pydantic() for item in top_items]

                if limit:
                    main_query = main_query.limit(limit)

                results = list(session.execute(main_query))

                semantic_items = []
                for row in results:
                    data = dict(row._mapping)
                    semantic_items.append(SemanticMemoryItem(**data))

                return [item.to_pydantic() for item in semantic_items]

    @enforce_types
    def insert_semantic_item(
        self,
        actor: PydanticClient,
        agent_state: AgentState,
        agent_id: str,
        name: str,
        summary: str,
        details: Optional[str],
        source: Optional[str],
        organization_id: str,
        filter_tags: Optional[dict] = None,
        use_cache: bool = True,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> PydanticSemanticMemoryItem:
        """
        Create a new semantic memory entry using provided parameters.
        """
        try:
            # Set defaults for required fields
            from mirix.services.user_manager import UserManager

            if client_id is None:
                client_id = actor.id
            if user_id is None:
                user_id = UserManager.ADMIN_USER_ID
                logger.debug("user_id not provided, using DEFAULT_USER_ID: %s", user_id)

            # Conditionally calculate embeddings based on BUILD_EMBEDDINGS_FOR_MEMORY flag
            if BUILD_EMBEDDINGS_FOR_MEMORY:
                # TODO: need to check if we need to chunk the text
                embed_model = embedding_model(agent_state.embedding_config)
                name_embedding = embed_model.get_text_embedding(name)
                summary_embedding = embed_model.get_text_embedding(summary)
                details_embedding = embed_model.get_text_embedding(details)
                embedding_config = agent_state.embedding_config
            else:
                name_embedding = None
                summary_embedding = None
                details_embedding = None
                embedding_config = None

            semantic_item = self.create_item(
                item_data=PydanticSemanticMemoryItem(
                    client_id=client_id,  # Required field: client app that created this memory
                    user_id=user_id,  # Required field: end-user who owns this memory
                    agent_id=agent_id,
                    name=name,
                    summary=summary,
                    details=details,
                    source=source,
                    organization_id=organization_id,
                    details_embedding=details_embedding,
                    name_embedding=name_embedding,
                    summary_embedding=summary_embedding,
                    embedding_config=embedding_config,
                    filter_tags=filter_tags,
                ),
                actor=actor,
                use_cache=use_cache,
                client_id=client_id,
                user_id=user_id,
            )

            # Note: Item is already added to clustering tree in create_item()
            return semantic_item
        except Exception as e:
            raise e

    def delete_semantic_item_by_id(self, semantic_memory_id: str, actor: PydanticClient) -> None:
        """Delete a semantic memory item by ID (removes from Redis cache)."""
        with self.session_maker() as session:
            try:
                item = SemanticMemoryItem.read(db_session=session, identifier=semantic_memory_id, actor=actor)
                # Remove from Redis cache before hard delete
                from mirix.database.redis_client import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.SEMANTIC_PREFIX}{semantic_memory_id}"
                    redis_client.delete(redis_key)
                item.hard_delete(session)
            except NoResultFound:
                raise NoResultFound(f"Semantic memory item with id {semantic_memory_id} not found.")

    @enforce_types
    def delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk delete all semantic memory records for a client (removes from Redis cache).
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
                row[0]
                for row in session.query(SemanticMemoryItem.id).filter(SemanticMemoryItem.client_id == actor.id).all()
            ]

            count = len(item_ids)
            if count == 0:
                return 0

            # Bulk delete in single query
            session.query(SemanticMemoryItem).filter(SemanticMemoryItem.client_id == actor.id).delete(
                synchronize_session=False
            )

            session.commit()

        # Batch delete from Redis cache (outside of session context)
        redis_client = get_redis_client()
        if redis_client and item_ids:
            redis_keys = [f"{redis_client.SEMANTIC_PREFIX}{item_id}" for item_id in item_ids]

            # Delete in batches to avoid command size limits
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count

    def soft_delete_by_client_id(self, actor: PydanticClient) -> int:
        """
        Bulk soft delete all semantic memory records for a client (updates Redis cache).

        Args:
            actor: Client whose memories to soft delete (uses actor.id as client_id)

        Returns:
            Number of records soft deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Query all non-deleted records for this client (use actor.id)
            items = (
                session.query(SemanticMemoryItem)
                .filter(
                    SemanticMemoryItem.client_id == actor.id,
                    SemanticMemoryItem.is_deleted == False,
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
                redis_key = f"{redis_client.SEMANTIC_PREFIX}{item_id}"
                try:
                    redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    # If update fails, remove from cache
                    redis_client.delete(redis_key)

        return count

    def soft_delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk soft delete all semantic memory records for a user (updates Redis cache).

        Args:
            user_id: ID of the user whose memories to soft delete

        Returns:
            Number of records soft deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Query all non-deleted records for this user
            items = (
                session.query(SemanticMemoryItem)
                .filter(
                    SemanticMemoryItem.user_id == user_id,
                    SemanticMemoryItem.is_deleted == False,
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
                redis_key = f"{redis_client.SEMANTIC_PREFIX}{item_id}"
                try:
                    redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    # If update fails, remove from cache
                    redis_client.delete(redis_key)

        return count

    def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all semantic memory records for a user (removes from Redis cache).
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
                row[0]
                for row in session.query(SemanticMemoryItem.id).filter(SemanticMemoryItem.user_id == user_id).all()
            ]

            count = len(item_ids)
            if count == 0:
                return 0

            # Bulk delete in single query
            session.query(SemanticMemoryItem).filter(SemanticMemoryItem.user_id == user_id).delete(
                synchronize_session=False
            )

            session.commit()

        # Batch delete from Redis cache (outside of session context)
        redis_client = get_redis_client()
        if redis_client and item_ids:
            redis_keys = [f"{redis_client.SEMANTIC_PREFIX}{item_id}" for item_id in item_ids]

            # Delete in batches to avoid command size limits
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count

    @update_timezone
    @enforce_types
    def list_semantic_items_by_org(
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
        similarity_threshold: Optional[float] = None,
    ) -> List[PydanticSemanticMemoryItem]:
        """List semantic memory items across ALL users in an organization."""
        from mirix.database.redis_client import get_redis_client

        redis_client = get_redis_client()

        if use_cache and redis_client:
            try:
                if not query or query == "":
                    results = redis_client.search_recent_by_org(
                        index_name=redis_client.SEMANTIC_INDEX,
                        limit=limit or 50,
                        organization_id=organization_id,
                        sort_by="created_at_ts",
                        filter_tags=filter_tags,
                    )
                    if results:
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]
                elif search_method == "embedding":
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

                    vector_field = (
                        f"{search_field}_embedding"
                        if search_field in ["name", "summary", "details"]
                        else "details_embedding"
                    )
                    results = redis_client.search_vector_by_org(
                        index_name=redis_client.SEMANTIC_INDEX,
                        embedding=embedded_text,
                        vector_field=vector_field,
                        limit=limit or 50,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                    )
                    if results:
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]
                else:
                    results = redis_client.search_text_by_org(
                        index_name=redis_client.SEMANTIC_INDEX,
                        query_text=query,
                        search_field=search_field or "details",
                        search_method=search_method,
                        limit=limit or 50,
                        organization_id=organization_id,
                        filter_tags=filter_tags,
                    )
                    if results:
                        results = redis_client.clean_redis_fields(results)
                        return [PydanticSemanticMemoryItem(**item) for item in results]
            except Exception as e:
                logger.warning("Redis search failed: %s", e)

        with self.session_maker() as session:
            # Return full SemanticMemoryItem objects, not individual columns
            base_query = select(SemanticMemoryItem).where(SemanticMemoryItem.organization_id == organization_id)

            if filter_tags:
                from sqlalchemy import func, or_

                for key, value in filter_tags.items():
                    if key == "scope":
                        # Scope matching: input value must be in memory's scope field
                        base_query = base_query.where(
                            or_(
                                func.lower(SemanticMemoryItem.filter_tags[key].as_string()).contains(
                                    str(value).lower()
                                ),
                                SemanticMemoryItem.filter_tags[key].as_string() == str(value),
                            )
                        )
                    else:
                        # Other keys: exact match
                        base_query = base_query.where(SemanticMemoryItem.filter_tags[key].as_string() == str(value))

            # Handle empty query - fall back to recent sort
            if not query or query == "":
                base_query = base_query.order_by(SemanticMemoryItem.created_at.desc())
                if limit:
                    base_query = base_query.limit(limit)
                result = session.execute(base_query)
                items = result.scalars().all()
                return [item.to_pydantic() for item in items]

            # Embedding search
            if search_method == "embedding":
                embedding_config = agent_state.embedding_config
                if embedded_text is None:
                    from mirix.embeddings import embedding_model

                    embedded_text = embedding_model(embedding_config).get_text_embedding(query)

                # Determine which embedding field to search
                if search_field == "name":
                    embedding_field = SemanticMemoryItem.name_embedding
                elif search_field == "summary":
                    embedding_field = SemanticMemoryItem.summary_embedding
                elif search_field == "details":
                    embedding_field = SemanticMemoryItem.details_embedding
                else:
                    embedding_field = SemanticMemoryItem.details_embedding

                embedding_query_field = embedding_field.cosine_distance(embedded_text).label("distance")
                base_query = base_query.add_columns(embedding_query_field)

                # Apply similarity threshold if provided
                if similarity_threshold is not None:
                    base_query = base_query.where(embedding_query_field < similarity_threshold)

                base_query = base_query.order_by(embedding_query_field)

            # BM25 search
            elif search_method == "bm25":
                from sqlalchemy import func

                # Determine search field
                if search_field == "name":
                    text_field = SemanticMemoryItem.name
                elif search_field == "summary":
                    text_field = SemanticMemoryItem.summary
                elif search_field == "details":
                    text_field = SemanticMemoryItem.details
                else:
                    text_field = SemanticMemoryItem.details

                tsquery = func.plainto_tsquery("english", query)
                tsvector = func.to_tsvector("english", text_field)
                rank = func.ts_rank_cd(tsvector, tsquery).label("rank")

                base_query = base_query.add_columns(rank).where(tsvector.op("@@")(tsquery)).order_by(rank.desc())

            if limit:
                base_query = base_query.limit(limit)

            result = session.execute(base_query)
            items = result.scalars().all()
            return [item.to_pydantic() for item in items]
