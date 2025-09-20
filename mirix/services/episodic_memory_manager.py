import re
import uuid
from typing import List, Optional, Dict, Any
import json
import string
import time
import datetime as dt
from datetime import datetime
from mirix.orm.errors import NoResultFound
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.schemas.user import User as PydanticUser
from sqlalchemy import Select, func, literal, select, union_all, text
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.utils import enforce_types
from pydantic import BaseModel, Field
from sqlalchemy import select
from rapidfuzz import fuzz 
from rank_bm25 import BM25Okapi
from mirix.settings import settings
from mirix.schemas.agent import AgentState
from mirix.embeddings import embedding_model, parse_and_chunk_text
from mirix.services.utils import build_query, update_timezone
from mirix.helpers.converters import deserialize_vector
from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

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
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
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

    def _count_word_matches(self, event_data: Dict[str, Any], query_words: List[str], search_field: str = '') -> int:
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
        if search_field == 'summary':
            search_texts = [event_data.get('summary', '')]
        elif search_field == 'details':
            search_texts = [event_data.get('details', '')]
        elif search_field == 'actor':
            search_texts = [event_data.get('actor', '')]
        elif search_field == 'event_type':
            search_texts = [event_data.get('event_type', '')]
        else:
            # Search across all relevant text fields
            search_texts = [
                event_data.get('summary', ''),
                event_data.get('details', ''),
                event_data.get('actor', ''),
                event_data.get('event_type', '')
            ]
        
        # Combine all search texts and clean them (remove punctuation)
        combined_text = ' '.join(text for text in search_texts if text)
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
    def get_episodic_memory_by_id(self, episodic_memory_id: str, actor: PydanticUser, timezone_str: str=None) -> Optional[PydanticEpisodicEvent]:
        """
        Fetch a single episodic episodic_memory record by ID.
        Raises NoResultFound if the record doesn't exist.
        """
        with self.session_maker() as session:
            try:
                episodic_memory_item = EpisodicEvent.read(db_session=session, identifier=episodic_memory_id, actor=actor)
                return episodic_memory_item.to_pydantic()
            except NoResultFound:
                raise NoResultFound(f"Episodic episodic_memory record with id {episodic_memory_id} not found.")

    @update_timezone
    @enforce_types
    def get_most_recently_updated_event(self, actor: PydanticUser, timezone_str: str = None) -> Optional[PydanticEpisodicEvent]:
        """
        Fetch the most recently updated episodic event based on last_modify timestamp.
        Filter by user_id from actor.
        Returns None if no events exist.
        """
        with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import cast, DateTime, text
            query = select(EpisodicEvent).where(
                EpisodicEvent.user_id == actor.id
            ).order_by(
                cast(text("episodic_memory.last_modify ->> 'timestamp'"), DateTime).desc()
            )
            
            result = session.execute(query.limit(1))
            episodic_memory = result.scalar_one_or_none()
            
            return [episodic_memory.to_pydantic()] if episodic_memory else None

    @enforce_types
    def create_episodic_memory(self, episodic_memory: PydanticEpisodicEvent, actor: PydanticUser) -> PydanticEpisodicEvent:
        """
        Create a new episodic episodic_memory record.
        Uses the provided Pydantic model (PydanticEpisodicEvent) as input data.
        """
        
        # Ensure ID is set before model_dump
        if not episodic_memory.id:
            from mirix.utils import generate_unique_short_id
            episodic_memory.id = generate_unique_short_id(self.session_maker, EpisodicEvent, "ep")

        # Convert the Pydantic model into a dict
        episodic_memory_dict = episodic_memory.model_dump()

        # Validate required fields if necessary (event_type, summary, etc.)
        required_fields = ["event_type", "summary"]
        for field in required_fields:
            if field not in episodic_memory_dict or episodic_memory_dict[field] is None:
                raise ValueError(f"Required field '{field}' is missing or None in episodic episodic_memory data")

        # Set defaults if needed
        episodic_memory_dict.setdefault("organization_id", episodic_memory.organization_id)
        
        # Set user_id from actor for multi-user support
        episodic_memory_dict["user_id"] = actor.id
        
        # Other fields like occurred_at, created_at, etc. 
        # might be auto-generated by the model or the DB

        # Create the episodic episodic_memory item
        with self.session_maker() as session:
            episodic_memory_item = EpisodicEvent(**episodic_memory_dict)
            episodic_memory_item.create(session)
            return episodic_memory_item.to_pydantic()

    @enforce_types
    def create_many_episodic_memory(self, episodic_memory: List[PydanticEpisodicEvent], actor: PydanticUser) -> List[PydanticEpisodicEvent]:
        """
        Create multiple episodic episodic_memory records in one go.
        """
        return [self.create_episodic_memory(e, actor) for e in episodic_memory]

    @enforce_types
    def delete_event_by_id(self, id: str, actor: PydanticUser) -> None:
        """
        Delete an episodic episodic_memory record by ID.
        """
        with self.session_maker() as session:
            try:
                episodic_memory_item = EpisodicEvent.read(db_session=session, identifier=id, actor=actor)
                episodic_memory_item.hard_delete(session)
            except NoResultFound:
                raise NoResultFound(f"Episodic episodic_memory record with id {id} not found.")

    @enforce_types
    def insert_event(self, 
                     actor: PydanticUser,
                     agent_state: AgentState,
                     event_type: str,
                     timestamp: datetime, 
                     event_actor: str, 
                     details: str,
                     summary: str,
                     organization_id: str,
                     tree_path: Optional[List[str]] = None) -> PydanticEpisodicEvent:

        try:

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
                    user_id=actor.id,
                    actor=event_actor,
                    summary=summary,
                    details=details,
                    tree_path=tree_path or [],
                    organization_id=organization_id,
                    summary_embedding=summary_embedding,
                    details_embedding=details_embedding,
                    embedding_config=embedding_config,
                    last_modify={"timestamp": datetime.now(dt.timezone.utc).isoformat(), "operation": "created"},
                ),
                actor=actor
            )

            return event
        
        except Exception as e:
            raise e
    
    @update_timezone
    @enforce_types
    def list_episodic_memory_around_timestamp(self,
                                              agent_state: AgentState,
                                              start_time: datetime,
                                              end_time: datetime,
                                              actor: PydanticUser,
                                              timezone_str: str = None) -> List[PydanticEpisodicEvent]:

        """
        list all episodic events around a timestamp
        time_window: The time window to search around the timestamp. It is in the form of "HH:MM:SS" or "HH:MM".
        """
        with self.session_maker() as session:

            # Query for episodic events within the time window
            query = select(EpisodicEvent).where(
                EpisodicEvent.occurred_at.between(start_time, end_time),
                EpisodicEvent.user_id == actor.id
            )

            result = session.execute(query)
            episodic_memory = result.scalars().all()

            return [event.to_pydantic() for event in episodic_memory]

    def get_total_number_of_items(self, actor: PydanticUser) -> int:
        """Get the total number of items in the episodic memory for the user."""
        with self.session_maker() as session:
            query = select(func.count(EpisodicEvent.id)).where(
                EpisodicEvent.user_id == actor.id
            )
            result = session.execute(query)
            return result.scalar_one()

    @update_timezone
    @enforce_types
    def list_episodic_memory(self, 
                             agent_state: AgentState,
                             actor: PydanticUser,
                             query: str = '', 
                             embedded_text: Optional[List[float]] = None,
                             search_field: str = '',
                             search_method: str = 'embedding',
                             limit: Optional[int] = 50,
                             timezone_str: str = None) -> List[PydanticEpisodicEvent]:
        """
        List all episodic events with various search methods.
        
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
        """

        with self.session_maker() as session:
            
            # TODO: handle the case where query is None, we need to extract the 50 most recent results

            if query == '':
                query_stmt = select(EpisodicEvent).where(
                    EpisodicEvent.user_id == actor.id
                ).order_by(EpisodicEvent.occurred_at.desc())
                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = session.execute(query_stmt)
                episodic_memory = result.scalars().all()
                return [event.to_pydantic() for event in episodic_memory]

            else:

                base_query = select(
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
                    EpisodicEvent.metadata_.label("metadata_"),
                    EpisodicEvent.last_modify.label("last_modify"),
                    EpisodicEvent.tree_path.label("tree_path"),
                    EpisodicEvent.user_id.label("user_id"),
                ).where(
                    EpisodicEvent.user_id == actor.id
                )

                if search_method == 'embedding':

                    embed_query = True
                    embedding_config = agent_state.embedding_config

                    main_query = build_query(
                        base_query=base_query,
                        query_text=query,
                        embedded_text=embedded_text,
                        embed_query=embed_query,
                        embedding_config=embedding_config,
                        search_field = eval("EpisodicEvent." + search_field + "_embedding"),
                        target_class=EpisodicEvent,
                    )
            
                elif search_method == 'string_match':

                    search_field = eval("EpisodicEvent." + search_field)
                    main_query = base_query.where(func.lower(search_field).contains(query.lower()))

                elif search_method == 'bm25':

                    # Check if we're using PostgreSQL - use native full-text search if available
                    if settings.mirix_pg_uri_no_default:
                        # Use PostgreSQL's native full-text search with ts_rank for BM25-like functionality
                        return self._postgresql_fulltext_search(
                            session, base_query, query, search_field, limit, actor
                        )
                    else:
                        # Fallback to in-memory BM25 for SQLite (legacy method)
                        # Load all candidate events (memory-intensive, kept for compatibility)
                        result = session.execute(select(EpisodicEvent).where(
                            EpisodicEvent.user_id == actor.id
                        ))
                        all_events = result.scalars().all()
                        
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

                elif search_method == 'fuzzy_match':

                    # Load all candidate events (kept for backward compatibility)
                    result = session.execute(select(EpisodicEvent).where(
                        EpisodicEvent.user_id == actor.id
                    ))
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

    def _postgresql_fulltext_search(self, session, base_query, query_text, search_field, limit, actor):
        """
        Efficient PostgreSQL-native full-text search using ts_rank for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.
        
        Args:
            session: Database session
            base_query: Base SQLAlchemy query
            query_text: Search query string
            search_field: Field to search in ('summary', 'details', 'actor', 'event_type', etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of EpisodicEvent objects ranked by relevance
        """
        from sqlalchemy import text, func
        
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
            tsquery_string_or = " | ".join(tsquery_parts)   # OR logic for recall
        else:
            tsquery_string_and = tsquery_string_or = tsquery_parts[0]
        
        # Build the PostgreSQL full-text search query using raw SQL with proper parameterization
        # This avoids the TextClause.op() issue and is more efficient
        
        # Determine which field to search based on search_field
        if search_field == 'summary':
            tsvector_sql = "to_tsvector('english', coalesce(summary, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(summary, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'details':
            tsvector_sql = "to_tsvector('english', coalesce(details, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(details, '')), to_tsquery('english', :tsquery), 32)"  
        elif search_field == 'actor':
            tsvector_sql = "to_tsvector('english', coalesce(actor, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(actor, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'event_type':
            tsvector_sql = "to_tsvector('english', coalesce(event_type, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(event_type, '')), to_tsquery('english', :tsquery), 32)"
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(summary, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(details, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(actor, '')), 'C') ||
                             setweight(to_tsvector('english', coalesce(event_type, '')), 'D')"""
            rank_sql = f"""ts_rank_cd(
                setweight(to_tsvector('english', coalesce(summary, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(details, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(actor, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(event_type, '')), 'D'),
                to_tsquery('english', :tsquery), 32)"""
        
        # Try AND query first for more precise results
        try:
            and_query_sql = text(f"""
                SELECT 
                    id, created_at, occurred_at, actor, event_type, tree_path,
                    summary, details, summary_embedding, details_embedding,
                    embedding_config, organization_id, metadata_, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM episodic_memory 
                WHERE {tsvector_sql} @@ to_tsquery('english', :tsquery)
                    AND user_id = :user_id
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """)
            
            results = list(session.execute(and_query_sql, {
                'tsquery': tsquery_string_and,
                'user_id': actor.id,
                'limit_val': limit or 50
            }))
            
            # If AND query returns sufficient results, use them
            if len(results) >= min(limit or 10, 10):
                episodic_memory = []
                for row in results:
                    data = dict(row._mapping)
                    # Remove the rank_score field before creating the object
                    data.pop('rank_score', None)
                    
                    # Parse JSON fields that are returned as strings from raw SQL
                    json_fields = ['last_modify', 'metadata_', 'embedding_config']
                    for field in json_fields:
                        if field in data and isinstance(data[field], str):
                            try:
                                data[field] = json.loads(data[field])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    
                    # Parse embedding fields
                    embedding_fields = ['summary_embedding', 'details_embedding']
                    for field in embedding_fields:
                        if field in data and data[field] is not None:
                            data[field] = self._parse_embedding_field(data[field])
                    
                    episodic_memory.append(EpisodicEvent(**data))
                
                return [event.to_pydantic() for event in episodic_memory]
                
        except Exception as e:
            print(f"PostgreSQL AND query error: {e}")
        
        # If AND query fails or returns too few results, try OR query
        try:
            or_query_sql = text(f"""
                SELECT 
                    id, created_at, occurred_at, actor, event_type, tree_path,
                    summary, details, summary_embedding, details_embedding,
                    embedding_config, organization_id, metadata_, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM episodic_memory 
                WHERE {tsvector_sql} @@ to_tsquery('english', :tsquery)
                    AND user_id = :user_id
                ORDER BY rank_score DESC, created_at DESC
                LIMIT :limit_val
            """)
            
            results = session.execute(or_query_sql, {
                'tsquery': tsquery_string_or,
                'user_id': actor.id,
                'limit_val': limit or 50
            })
            
            episodic_memory = []
            for row in results:
                data = dict(row._mapping)
                # Remove the rank_score field before creating the object
                data.pop('rank_score', None)
                
                # Parse JSON fields that are returned as strings from raw SQL
                json_fields = ['last_modify', 'metadata_', 'embedding_config']
                for field in json_fields:
                    if field in data and isinstance(data[field], str):
                        try:
                            data[field] = json.loads(data[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                # Parse embedding fields
                embedding_fields = ['summary_embedding', 'details_embedding']
                for field in embedding_fields:
                    if field in data and data[field] is not None:
                        data[field] = self._parse_embedding_field(data[field])
                
                episodic_memory.append(EpisodicEvent(**data))
            
            return [event.to_pydantic() for event in episodic_memory]
            
        except Exception as e:
            # If there's an error with the tsquery (e.g., invalid syntax), fall back to simpler search
            print(f"PostgreSQL full-text search error: {e}")
            # Fall back to simple ILIKE search
            fallback_field = getattr(EpisodicEvent, search_field) if search_field and hasattr(EpisodicEvent, search_field) else EpisodicEvent.summary
            fallback_query = base_query.where(
                func.lower(fallback_field).contains(query_text.lower())
            ).order_by(EpisodicEvent.created_at.desc())
            
            if limit:
                fallback_query = fallback_query.limit(limit)
                
            results = session.execute(fallback_query)
            episodic_memory = [EpisodicEvent(**dict(row._mapping)) for row in results]
            return [event.to_pydantic() for event in episodic_memory]
        
    def update_event(self, 
                            event_id: str = None,
                            new_summary: str = None,
                            new_details: str = None,
                            actor: PydanticUser = None):
        """
        Update the selected events
        """

        with self.session_maker() as session:
            
            # query = select(EpisodicEvent)
            # query = query.where(EpisodicEvent.id == event_id)
            # selected_event = session.execute(query).scalar_one_or_none()
            selected_event = EpisodicEvent.read(db_session=session, identifier=event_id, actor=actor)
            
            if not selected_event:
                raise ValueError(f"Episodic episodic_memory record with id {event_id} not found.")

            operations = []
            if new_summary:
                selected_event.summary = new_summary
                operations.append("summary_updated")
            if new_details:
                selected_event.details += "\n" + new_details
                operations.append("details_updated")
            
            # Update last_modify field with timestamp and operation info
            selected_event.last_modify = {
                "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                "operation": ", ".join(operations) if operations else "updated"
            }
            
            selected_event.update(session)
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
                if embedding_value.startswith('[') and embedding_value.endswith(']'):
                    try:
                        return json.loads(embedding_value)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try manual parsing
                        # Remove brackets and split by comma
                        inner = embedding_value[1:-1]  # Remove [ and ]
                        return [float(x.strip()) for x in inner.split(',') if x.strip()]
                
                # Try comma-separated values
                if ',' in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split(',') if x.strip()]
                
                # Try space-separated values
                if ' ' in embedding_value:
                    return [float(x.strip()) for x in embedding_value.split() if x.strip()]
            
            # Try using the original deserialize_vector approach for binary data
            try:
                from mirix.helpers.converters import deserialize_vector
                class MockDialect:
                    name = 'postgresql'
                return deserialize_vector(embedding_value, MockDialect())
            except Exception:
                pass
                
            # If all else fails, return None to avoid validation errors
            return None
            
        except Exception as e:
            print(f"Warning: Failed to parse embedding field: {e}")
            return None
    