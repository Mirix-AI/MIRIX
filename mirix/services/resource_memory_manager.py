import uuid
from typing import List, Optional, Dict, Any
import json
import string
import re
import time

from rank_bm25 import BM25Okapi
from mirix.orm.errors import NoResultFound
from mirix.embeddings import embedding_model, parse_and_chunk_text
from mirix.orm.resource_memory import ResourceMemoryItem
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.resource_memory import (
    ResourceMemoryItem as PydanticResourceMemoryItem,
    ResourceMemoryItemUpdate
)
from mirix.schemas.agent import AgentState
from mirix.utils import enforce_types
from pydantic import BaseModel, Field
from sqlalchemy import select, func, text
from mirix.services.utils import build_query, update_timezone
from mirix.settings import settings
from mirix.helpers.converters import deserialize_vector
from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

class ResourceMemoryManager:
    """Manager class to handle logic related to Resource/Workspace Memory Items."""

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

    def _postgresql_fulltext_search(self, session, base_query, query_text, search_field, limit, actor):
        """
        Efficient PostgreSQL-native full-text search using ts_rank_cd for BM25-like functionality.
        This method leverages PostgreSQL's built-in full-text search capabilities and GIN indexes.
        
        Args:
            session: Database session
            base_query: Base SQLAlchemy query
            query_text: Search query string
            search_field: Field to search in ('title', 'summary', 'content', 'resource_type', etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of ResourceMemoryItem objects ranked by relevance
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
            tsquery_string_or = " | ".join(tsquery_parts)   # OR logic for recall
        else:
            tsquery_string_and = tsquery_string_or = tsquery_parts[0]
        
        # Determine which field to search based on search_field
        if search_field == 'title':
            tsvector_sql = "to_tsvector('english', coalesce(title, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(title, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'summary':
            tsvector_sql = "to_tsvector('english', coalesce(summary, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(summary, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'content':
            tsvector_sql = "to_tsvector('english', coalesce(content, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(content, '')), to_tsquery('english', :tsquery), 32)"
        elif search_field == 'resource_type':
            tsvector_sql = "to_tsvector('english', coalesce(resource_type, ''))"
            rank_sql = f"ts_rank_cd(to_tsvector('english', coalesce(resource_type, '')), to_tsquery('english', :tsquery), 32)"
        else:
            # Search across all relevant text fields with weighting
            tsvector_sql = """setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                             setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                             setweight(to_tsvector('english', coalesce(content, '')), 'C') ||
                             setweight(to_tsvector('english', coalesce(resource_type, '')), 'D')"""
            rank_sql = f"""ts_rank_cd(
                setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(content, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(resource_type, '')), 'D'),
                to_tsquery('english', :tsquery), 32)"""
        
        # Try AND query first for more precise results
        try:
            and_query_sql = text(f"""
                SELECT 
                    id, title, summary, content, tree_path, summary_embedding, embedding_config,
                    created_at, resource_type, organization_id, metadata_, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM resource_memory 
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
                resources = []
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
                    embedding_fields = ['summary_embedding']
                    for field in embedding_fields:
                        if field in data and data[field] is not None:
                            data[field] = self._parse_embedding_field(data[field])
                    
                    resources.append(ResourceMemoryItem(**data))
                
                return [resource.to_pydantic() for resource in resources]
                
        except Exception as e:
            print(f"PostgreSQL AND query error: {e}")
        
        # If AND query fails or returns too few results, try OR query
        try:
            or_query_sql = text(f"""
                SELECT 
                    id, title, summary, content, tree_path, summary_embedding, embedding_config,
                    created_at, resource_type, organization_id, metadata_, last_modify, user_id,
                    {rank_sql} as rank_score
                FROM resource_memory 
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
            
            resources = []
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
                embedding_fields = ['summary_embedding']
                for field in embedding_fields:
                    if field in data and data[field] is not None:
                        data[field] = self._parse_embedding_field(data[field])
                
                resources.append(ResourceMemoryItem(**data))
            
            return [resource.to_pydantic() for resource in resources]
            
        except Exception as e:
            # If there's an error with the tsquery, fall back to simpler search
            print(f"PostgreSQL full-text search error: {e}")
            # Fall back to simple ILIKE search
            fallback_field = getattr(ResourceMemoryItem, search_field) if search_field and hasattr(ResourceMemoryItem, search_field) else ResourceMemoryItem.title
            fallback_query = base_query.where(
                func.lower(fallback_field).contains(query_text.lower())
            ).order_by(ResourceMemoryItem.created_at.desc())
            
            if limit:
                fallback_query = fallback_query.limit(limit)
                
            results = session.execute(fallback_query)
            resources = [ResourceMemoryItem(**dict(row._mapping)) for row in results]
            return [resource.to_pydantic() for resource in resources]

    @update_timezone
    @enforce_types
    def get_item_by_id(self, item_id: str, actor: PydanticUser, timezone_str: str) -> Optional[PydanticResourceMemoryItem]:
        """Fetch a resource memory item by ID."""
        with self.session_maker() as session:
            try:
                item = ResourceMemoryItem.read(db_session=session, identifier=item_id, actor=actor)
                return item.to_pydantic()
            except NoResultFound:
                raise NoResultFound(f"Resource memory item with id {item_id} not found.")

    @update_timezone
    @enforce_types
    def get_most_recently_updated_item(self, actor: PydanticUser, timezone_str: str = None) -> Optional[PydanticResourceMemoryItem]:
        """
        Fetch the most recently updated resource memory item based on last_modify timestamp.
        Filter by user_id from actor.
        Returns None if no items exist.
        """
        with self.session_maker() as session:
            # Use proper PostgreSQL JSON text extraction and casting for ordering
            from sqlalchemy import cast, DateTime, text
            query = select(ResourceMemoryItem).where(
                ResourceMemoryItem.user_id == actor.id
            ).order_by(
                cast(text("resource_memory.last_modify ->> 'timestamp'"), DateTime).desc()
            )
            
            result = session.execute(query.limit(1))
            item = result.scalar_one_or_none()
            
            return [item.to_pydantic()] if item else None

    @enforce_types
    def create_item(self, item_data: PydanticResourceMemoryItem, actor: PydanticUser) -> PydanticResourceMemoryItem:
        """Create a new resource memory item."""
        
        # Ensure ID is set before model_dump
        if not item_data.id:
            from mirix.utils import generate_unique_short_id
            item_data.id = generate_unique_short_id(self.session_maker, ResourceMemoryItem, "res")
        
        data_dict = item_data.model_dump()

        # Validate required fields
        required_fields = ["title", "summary", "content"]
        for field in required_fields:
            if field not in data_dict:
                raise ValueError(f"Required field '{field}' missing from resource memory data")

        data_dict.setdefault("metadata_", {})
        
        # Set user_id from actor for multi-user support
        data_dict["user_id"] = actor.id

        with self.session_maker() as session:
            item = ResourceMemoryItem(**data_dict)
            item.create(session)
            return item.to_pydantic()

    @enforce_types
    def update_item(self, item_update: ResourceMemoryItemUpdate, actor: PydanticUser) -> PydanticResourceMemoryItem:
        """Update an existing resource memory item."""
        with self.session_maker() as session:
            item = ResourceMemoryItem.read(db_session=session, identifier=item_update.id, actor=actor)
            update_data = item_update.model_dump(exclude_unset=True)
            for k, v in update_data.items():
                if k not in ["id", "updated_at"]:
                    setattr(item, k, v)
            item.updated_at = item_update.updated_at
            item.update(session, actor=actor)
            return item.to_pydantic()

    @enforce_types
    def create_many_items(self, items: List[PydanticResourceMemoryItem], actor: PydanticUser, limit: Optional[int] = 50) -> List[PydanticResourceMemoryItem]:
        """Create multiple resource memory items."""
        return [self.create_item(i, actor) for i in items]

    def get_total_number_of_items(self, actor: PydanticUser) -> int:
        """Get the total number of items in the resource memory for the user."""
        with self.session_maker() as session:
            query = select(func.count(ResourceMemoryItem.id)).where(
                ResourceMemoryItem.user_id == actor.id
            )
            result = session.execute(query)
            return result.scalar_one()

    @update_timezone
    @enforce_types
    def list_resources(self,
                       agent_state: AgentState,
                       actor: PydanticUser,
                       query: str = '',
                       embedded_text: Optional[List[float]] = None,
                       search_field: str = 'content',
                       search_method: str = 'string_match',
                       limit: Optional[int] = 50,
                       timezone_str: str = None) -> List[PydanticResourceMemoryItem]:
        """
        Retrieve resource memory items according to the query.
        
        Args:
            agent_state: The agent state containing embedding configuration
            query: Search query string
            embedded_text: Pre-computed embedding for semantic search
            search_field: Field to search in ('title', 'summary', 'content', 'resource_type')
            search_method: Search method to use:
                - 'embedding': Vector similarity search using embeddings
                - 'string_match': Simple string containment search
                - 'bm25': **RECOMMENDED** - PostgreSQL native full-text search (ts_rank_cd) when using PostgreSQL, 
                               falls back to in-memory BM25 for SQLite
                - 'fuzzy_match': Fuzzy string matching (not implemented)
            limit: Maximum number of results to return
            timezone_str: Timezone string for timestamp conversion
            
        Returns:
            List of resource memory items matching the search criteria
            
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

        with self.session_maker() as session:

            if query == '':
                # Use proper PostgreSQL JSON text extraction and casting for ordering
                from sqlalchemy import cast, DateTime, text
                query_stmt = select(ResourceMemoryItem).where(
                    ResourceMemoryItem.user_id == actor.id
                ).order_by(
                    cast(text("resource_memory.last_modify ->> 'timestamp'"), DateTime).desc()
                )
                if limit:
                    query_stmt = query_stmt.limit(limit)
                result = session.execute(query_stmt)
                resource_memory = result.scalars().all()
                return [event.to_pydantic() for event in resource_memory]

            base_query = select(
                ResourceMemoryItem.id.label("id"),
                ResourceMemoryItem.title.label("title"),
                ResourceMemoryItem.summary.label("summary"),
                ResourceMemoryItem.content.label("content"),
                ResourceMemoryItem.summary_embedding.label("summary_embedding"),
                ResourceMemoryItem.embedding_config.label("embedding_config"),
                ResourceMemoryItem.created_at.label("created_at"),
                ResourceMemoryItem.resource_type.label("resource_type"),
                ResourceMemoryItem.organization_id.label("organization_id"),
                ResourceMemoryItem.metadata_.label("metadata_"),
                ResourceMemoryItem.last_modify.label("last_modify"),
                ResourceMemoryItem.tree_path.label("tree_path"),
                ResourceMemoryItem.user_id.label("user_id"),
            ).where(
                ResourceMemoryItem.user_id == actor.id
            )

            if search_method == 'string_match':
                main_query = base_query.where(func.lower(getattr(ResourceMemoryItem, search_field)).contains(query.lower()))
            
            elif search_method == 'embedding':
                embed_query = True
                embedding_config = agent_state.embedding_config

                main_query = build_query(
                    base_query=base_query,
                    query_text=query,
                    embed_query=embed_query,
                    embedded_text=embedded_text,
                    embedding_config=embedding_config,
                    search_field = eval("ResourceMemoryItem." + search_field + "_embedding"),
                    target_class=ResourceMemoryItem,
                )

            elif search_method == 'bm25':
                
                # Check if we're using PostgreSQL - use native full-text search if available
                if settings.mirix_pg_uri_no_default:
                    # Use PostgreSQL native full-text search
                    return self._postgresql_fulltext_search(
                        session, base_query, query, search_field, limit, actor
                    )
                else:
                    # Fallback to in-memory BM25 for SQLite (legacy method)
                    # Load all candidate items (memory-intensive, kept for compatibility)
                    result = session.execute(select(ResourceMemoryItem).where(
                        ResourceMemoryItem.user_id == actor.id
                    ))
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
                            text_to_search = item.content or ""
                        
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
                    resource_memory = top_items
                    
                    # Return the list after converting to Pydantic
                    return [item.to_pydantic() for item in resource_memory]

            elif search_method == "fuzzy_match":
                raise NotImplementedError("Fuzzy matching is not implemented yet.")
        
            else:
                raise ValueError(f"Unknown search method: {search_method}")

            results = list(session.execute(main_query))[:limit]

            resource_memory = []

            for row in results:
                data = dict(row._mapping)
                resource_memory.append(ResourceMemoryItem(**data))
            
            return [item.to_pydantic() for item in resource_memory]

    @enforce_types
    def insert_resource(self, 
                        actor: PydanticUser,
                        agent_state: AgentState,
                        title: str,
                        summary: str,
                        resource_type: str,
                        content: str,
                        organization_id: str,
                        tree_path: Optional[List[str]] = None
                        ) -> PydanticResourceMemoryItem:
        """Create a new resource memory item."""
        try:

            # Conditionally calculate embeddings based on BUILD_EMBEDDINGS_FOR_MEMORY flag
            if BUILD_EMBEDDINGS_FOR_MEMORY:
                embed_model = embedding_model(agent_state.embedding_config)
                summary_embedding = embed_model.get_text_embedding(summary)
                embedding_config = agent_state.embedding_config
            else:
                summary_embedding = None
                embedding_config = None

            resource = self.create_item(
                item_data=PydanticResourceMemoryItem(
                    user_id=actor.id,
                    title=title,
                    summary=summary,
                    content=content,
                    resource_type=resource_type,
                    tree_path=tree_path or [],
                    organization_id=organization_id,
                    summary_embedding=summary_embedding,
                    embedding_config=embedding_config,
                ),
                actor=actor
            )
            return resource
        
        except Exception as e:
            raise e     

    @enforce_types
    def delete_resource_by_id(self, resource_id: str, actor: PydanticUser) -> None:
        """Delete a resource memory item by ID."""
        with self.session_maker() as session:
            try:
                item = ResourceMemoryItem.read(db_session=session, identifier=resource_id, actor=actor)
                item.hard_delete(session)
            except NoResultFound:
                raise NoResultFound(f"Resource Memory record with id {resource_id} not found.")
