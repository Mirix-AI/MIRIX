import asyncio
import random
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

from sqlalchemy import String, and_, desc, func, or_, select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, TimeoutError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from mirix.log import get_logger
from mirix.orm.base import Base, CommonSqlalchemyMetaMixins
from mirix.orm.enums import AccessType
from mirix.orm.errors import (
    DatabaseTimeoutError,
    ForeignKeyConstraintViolationError,
    NoResultFound,
    UniqueConstraintViolationError,
)
from mirix.orm.sqlite_functions import adapt_array

if TYPE_CHECKING:
    from pydantic import BaseModel
    from sqlalchemy import Select

    from mirix.orm.client import Client
    from mirix.orm.user import User

logger = get_logger(__name__)

# Diagnostic flag for MissingGreenlet debugging - set via env var
import os

_TRACE_MISSING_GREENLET = os.getenv("MIRIX_TRACE_MISSING_GREENLET", "false").lower() == "true"


def handle_db_timeout(func):
    """Decorator to handle SQLAlchemy TimeoutError (async-aware)."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TimeoutError as e:
            logger.error("Timeout while executing %s: %s", func.__name__, e)
            raise DatabaseTimeoutError(message=f"Timeout occurred in {func.__name__}.", original_exception=e) from e

    return wrapper


def retry_db_operation(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
):
    """
    Decorator to retry database operations with exponential backoff when encountering database locked errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds between retries
        backoff_factor: Multiplier for exponential backoff
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    if any(
                        msg in error_msg
                        for msg in [
                            "database is locked",
                            "database locked",
                            "sqlite3.operationalerror: database is locked",
                            "could not obtain lock",
                            "busy",
                            "locked",
                        ]
                    ):
                        if attempt == max_retries:
                            logger.error(
                                "Database locked error in %s after %d retries: %s",
                                func.__name__,
                                max_retries,
                                e,
                            )
                            raise
                        delay = min(base_delay * (backoff_factor**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        logger.warning(
                            "Database locked in %s (attempt %d/%d), retrying in %.2fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                            total_delay,
                            e,
                        )
                        await asyncio.sleep(total_delay)
                        continue
                    raise
                except Exception as e:
                    raise e
            raise last_exception

        return wrapper

    return decorator


def transaction_retry(max_retries: int = 3, base_delay: float = 0.1, max_delay: float = 2.0):
    """
    Decorator for database operations that need proper transaction handling with rollback on failures.

    This decorator ensures that:
    1. Transactions are properly committed on success
    2. Transactions are properly rolled back on failure
    3. Database locked errors are retried with exponential backoff
    4. All other exceptions are properly handled

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for first retry
        max_delay: Maximum delay in seconds between retries
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    if any(
                        msg in error_msg
                        for msg in [
                            "database is locked",
                            "database locked",
                            "sqlite3.operationalerror: database is locked",
                            "could not obtain lock",
                            "busy",
                            "locked",
                        ]
                    ):
                        if attempt == max_retries:
                            logger.error(
                                "Database locked error in %s after %d retries: %s",
                                func.__name__,
                                max_retries,
                                e,
                            )
                            raise
                        delay = min(base_delay * (2.0**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        logger.warning(
                            "Database locked in %s (attempt %d/%d), retrying in %.2fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_retries + 1,
                            total_delay,
                            e,
                        )
                        await asyncio.sleep(total_delay)
                        continue
                    raise
                except Exception as e:
                    raise e
            raise last_exception

        return wrapper

    return decorator


class SqlalchemyBase(CommonSqlalchemyMetaMixins, Base):
    __abstract__ = True

    __order_by_default__ = "created_at"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    @classmethod
    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def list(
        cls,
        *,
        db_session: AsyncSession,
        cursor: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        ascending: bool = True,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        actor: Optional["Client"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        join_model: Optional[Base] = None,
        join_conditions: Optional[Union[Tuple, List]] = None,
        **kwargs,
    ) -> List["SqlalchemyBase"]:
        """
        List records with cursor-based pagination, ordering by created_at.
        Cursor is an ID, but pagination is based on the cursor object's created_at value.

        Args:
            db_session: SQLAlchemy session
            cursor: ID of the last item seen (for pagination)
            start_date: Filter items after this date
            end_date: Filter items before this date
            limit: Maximum number of items to return
            query_text: Text to search for
            query_embedding: Vector to search for similar embeddings
            ascending: Sort direction
            tags: List of tags to filter by
            match_all_tags: If True, return items matching all tags. If False, match any tag.
            **kwargs: Additional filters to apply
        """
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must be earlier than or equal to end_date")

        logger.debug("Listing %s with kwarg filters %s", cls.__name__, kwargs)
        session = db_session
        # If cursor provided, get the reference object
        cursor_obj = None
        if cursor:
            cursor_obj = await session.get(cls, cursor)
            if not cursor_obj:
                raise NoResultFound(f"No {cls.__name__} found with id {cursor}")

        query = select(cls)

        if join_model and join_conditions:
            query = query.join(join_model, and_(*join_conditions))

        # Apply access predicate if actor is provided
        if actor:
            query = cls.apply_access_predicate(query, actor, access, access_type)

        # Handle tag filtering if the model has tags
        if tags and hasattr(cls, "tags"):
            query = select(cls)

            if match_all_tags:
                # Match ALL tags - use subqueries
                subquery = (
                    select(cls.tags.property.mapper.class_.agent_id)
                    .where(cls.tags.property.mapper.class_.tag.in_(tags))
                    .group_by(cls.tags.property.mapper.class_.agent_id)
                    .having(func.count() == len(tags))
                )
                query = query.filter(cls.id.in_(subquery))
            else:
                # Match ANY tag - use join and filter
                query = (
                    query.join(cls.tags)
                    .filter(cls.tags.property.mapper.class_.tag.in_(tags))
                    .group_by(cls.id)  # Deduplicate results
                )

            # Group by primary key and all necessary columns to avoid JSON comparison
            query = query.group_by(cls.id)

        # Apply filtering logic from kwargs
        for key, value in kwargs.items():
            if "." in key:
                # Handle joined table columns
                table_name, column_name = key.split(".")
                joined_table = locals().get(table_name) or globals().get(table_name)
                column = getattr(joined_table, column_name)
            else:
                # Handle columns from main table
                column = getattr(cls, key)

            if isinstance(value, (list, tuple, set)):
                query = query.where(column.in_(value))
            else:
                query = query.where(column == value)

        # Date range filtering
        if start_date:
            query = query.filter(cls.created_at > start_date)
        if end_date:
            query = query.filter(cls.created_at < end_date)

        # Cursor-based pagination
        if cursor_obj:
            if ascending:
                query = query.where(cls.created_at >= cursor_obj.created_at).where(
                    or_(
                        cls.created_at > cursor_obj.created_at,
                        cls.id > cursor_obj.id,
                    )
                )
            else:
                query = query.where(cls.created_at <= cursor_obj.created_at).where(
                    or_(
                        cls.created_at < cursor_obj.created_at,
                        cls.id < cursor_obj.id,
                    )
                )

        # Text search
        if query_text:
            if hasattr(cls, "text"):
                query = query.filter(func.lower(cls.text).contains(func.lower(query_text)))
            elif hasattr(cls, "name"):
                # Special case for Agent model - search across name
                query = query.filter(func.lower(cls.name).contains(func.lower(query_text)))

        # Embedding search (for Passages)
        is_ordered = False
        if query_embedding:
            if not hasattr(cls, "embedding"):
                raise ValueError(f"Class {cls.__name__} does not have an embedding column")

            from mirix.settings import settings

            if settings.mirix_pg_uri_no_default:
                # PostgreSQL with pgvector
                query = query.order_by(cls.embedding.cosine_distance(query_embedding).asc())
            else:
                # SQLite with custom vector type
                query_embedding_binary = adapt_array(query_embedding)
                query = query.order_by(
                    func.cosine_distance(cls.embedding, query_embedding_binary).asc(),
                    cls.created_at.asc(),
                    cls.id.asc(),
                )
                is_ordered = True

        # Handle soft deletes
        if hasattr(cls, "is_deleted"):
            query = query.where(~cls.is_deleted)

        # Apply ordering
        if not is_ordered:
            if ascending:
                query = query.order_by(cls.created_at, cls.id)
            else:
                query = query.order_by(desc(cls.created_at), desc(cls.id))

        query = query.limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())

    @classmethod
    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def read(
        cls,
        db_session: AsyncSession,
        identifier: Optional[str] = None,
        actor: Optional["Client"] = None,
        user: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> "SqlalchemyBase":
        """The primary accessor for an ORM record (async)."""
        logger.debug("Reading %s with ID: %s with actor=%s", cls.__name__, identifier, actor)

        query = select(cls)
        query_conditions = []

        if identifier is not None:
            query = query.where(cls.id == identifier)
            query_conditions.append(f"id='{identifier}'")

        if kwargs:
            query = query.filter_by(**kwargs)
            query_conditions.append(", ".join(f"{key}='{value}'" for key, value in kwargs.items()))

        if actor:
            query = cls.apply_access_predicate(query, actor, access, access_type, user)
            query_conditions.append(f"access level in {access} for actor='{actor}'")

        if hasattr(cls, "is_deleted"):
            query = query.where(~cls.is_deleted)
            query_conditions.append("is_deleted=False")

        result = await db_session.execute(query)
        found = result.scalar_one_or_none()
        if found:
            return found

        conditions_str = ", ".join(query_conditions) if query_conditions else "no specific conditions"
        raise NoResultFound(f"{cls.__name__} not found with {conditions_str}")

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    async def create(self, db_session: AsyncSession, actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug("Creating %s with ID: %s with actor=%s", self.__class__.__name__, self.id, actor)

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        try:
            db_session.add(self)
            await db_session.commit()
            await db_session.refresh(self)
            return self
        except (DBAPIError, IntegrityError) as e:
            await db_session.rollback()
            logger.error("Failed to create %s with ID %s: %s", self.__class__.__name__, self.id, e)
            self._handle_dbapi_error(e)
        except Exception as e:
            await db_session.rollback()
            logger.error("Unexpected error creating %s with ID %s: %s", self.__class__.__name__, self.id, e)
            raise

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def delete(self, db_session: AsyncSession, actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug("Soft deleting %s with ID: %s with actor=%s", self.__class__.__name__, self.id, actor)

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return await self.update(db_session)

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def hard_delete(self, db_session: AsyncSession, actor: Optional["Client"] = None) -> None:
        """Permanently removes the record from the database (async)."""
        logger.debug("Hard deleting %s with ID: %s with actor=%s", self.__class__.__name__, self.id, actor)

        try:
            await db_session.delete(self)
            await db_session.commit()
            logger.debug("%s with ID %s successfully hard deleted", self.__class__.__name__, self.id)
        except Exception as e:
            await db_session.rollback()
            logger.exception("Failed to hard delete %s with ID %s", self.__class__.__name__, self.id)
            raise ValueError(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}: {e}") from e

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    async def update(self, db_session: AsyncSession, actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug("Updating %s with ID: %s with actor=%s", self.__class__.__name__, self.id, actor)
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.set_updated_at()

        try:
            db_session.add(self)
            await db_session.commit()
            await db_session.refresh(self)
            return self
        except Exception as e:
            await db_session.rollback()
            logger.error("Failed to update %s with ID %s: %s", self.__class__.__name__, self.id, e)
            raise

    @classmethod
    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def size(
        cls,
        *,
        db_session: AsyncSession,
        actor: Optional["Client"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> int:
        """Get the count of rows that match the provided filters (async)."""
        logger.debug("Calculating size for %s with filters %s", cls.__name__, kwargs)

        query = select(func.count()).select_from(cls)

        if actor:
            query = cls.apply_access_predicate(query, actor, access, access_type)

        for key, value in kwargs.items():
            if value:
                column = getattr(cls, key, None)
                if not column:
                    raise AttributeError(f"{cls.__name__} has no attribute '{key}'")
                if isinstance(value, (list, tuple, set)):
                    query = query.where(column.in_(value))
                else:
                    query = query.where(column == value)

        if hasattr(cls, "is_deleted"):
            query = query.where(~cls.is_deleted)

        try:
            result = await db_session.execute(query)
            count = result.scalar()
            return count if count else 0
        except DBAPIError as e:
            logger.exception("Failed to calculate size for %s", cls.__name__)
            raise e

    @classmethod
    def apply_access_predicate(
        cls,
        query: "Select",
        actor: "Client",
        access: List[Literal["read", "write", "admin"]],
        access_type: AccessType = AccessType.ORGANIZATION,
        user: Optional["User"] = None,
    ) -> "Select":
        """applies a WHERE clause restricting results to the given actor and access level

        For the agents table, this method automatically applies client-level isolation by filtering
        on both organization_id and _created_by_id (client_id). This ensures each client has their
        own independent agent hierarchy (meta agent and sub-agents).

        Args:
            query: The initial sqlalchemy select statement
            actor: The user acting on the query. **Note**: this is called 'actor' to identify the
                   person or system acting. Users can act on users, making naming very sticky otherwise.
            user_id: The user id to restrict the query to.
            access:
                what mode of access should the query restrict to? This will be used with granular permissions,
                but because of how it will impact every query we want to be explicitly calling access ahead of time.
            access_type: The type of access to restrict the query to.
        Returns:
            the sqlalchemy select statement restricted to the given access.
        """
        del access  # entrypoint for row-level permissions. Defaults to "same org as the actor, all permissions" at the moment
        if access_type == AccessType.ORGANIZATION:
            org_id = getattr(actor, "organization_id", None)
            if not org_id:
                raise ValueError(f"object {actor} has no organization accessor")

            # SPECIAL HANDLING FOR AGENTS TABLE: Add client-level isolation
            # Each client gets their own independent agent hierarchy
            if cls.__tablename__ == "agents":
                client_id = getattr(actor, "id", None)
                if not client_id:
                    raise ValueError(f"object {actor} has no client id accessor")
                # Filter by BOTH organization_id AND _created_by_id (client_id)
                return query.where(
                    cls.organization_id == org_id,
                    cls._created_by_id == client_id,  # Client-level isolation
                    ~cls.is_deleted,
                )

            # For all other tables: organization-level filtering only
            return query.where(cls.organization_id == org_id, ~cls.is_deleted)
        elif access_type == AccessType.USER:
            if not user:
                raise ValueError(f"object {actor} has no user accessor")
            return query.where(cls.user_id == user.id, ~cls.is_deleted)
        else:
            raise ValueError(f"unknown access_type: {access_type}")

    @classmethod
    def _handle_dbapi_error(cls, e: DBAPIError):
        """Handle database errors and raise appropriate custom exceptions."""
        orig = e.orig  # Extract the original error from the DBAPIError
        error_code = None
        error_message = str(orig) if orig else str(e)
        logger.info("Handling DBAPIError: %s", error_message)

        # Handle SQLite-specific errors
        if "UNIQUE constraint failed" in error_message:
            raise UniqueConstraintViolationError(
                f"A unique constraint was violated for {cls.__name__}. Check your input for duplicates: {e}"
            ) from e

        if "FOREIGN KEY constraint failed" in error_message:
            raise ForeignKeyConstraintViolationError(
                f"A foreign key constraint was violated for {cls.__name__}. Check your input for missing or invalid references: {e}"
            ) from e

        # For psycopg2
        if hasattr(orig, "pgcode"):
            error_code = orig.pgcode
        # For pg8000
        elif hasattr(orig, "args") and len(orig.args) > 0:
            # The first argument contains the error details as a dictionary
            err_dict = orig.args[0]
            if isinstance(err_dict, dict):
                error_code = err_dict.get("C")  # 'C' is the error code field
        logger.info("Extracted error_code: %s", error_code)

        # Handle unique constraint violations
        if error_code == "23505":
            raise UniqueConstraintViolationError(
                f"A unique constraint was violated for {cls.__name__}. Check your input for duplicates: {e}"
            ) from e

        # Handle foreign key violations
        if error_code == "23503":
            raise ForeignKeyConstraintViolationError(
                f"A foreign key constraint was violated for {cls.__name__}. Check your input for missing or invalid references: {e}"
            ) from e

        # Re-raise for other unhandled DBAPI errors
        raise

    @property
    def __pydantic_model__(self) -> "BaseModel":
        raise NotImplementedError("Sqlalchemy models must declare a __pydantic_model__ property to be convertable.")

    def to_pydantic(self) -> "BaseModel":
        """converts to the basic pydantic model counterpart"""
        if _TRACE_MISSING_GREENLET:
            try:
                return self.__pydantic_model__.model_validate(self)
            except Exception as e:
                if "MissingGreenlet" in str(type(e).__name__) or "greenlet" in str(e).lower():
                    import traceback

                    logger.error(
                        "MissingGreenlet detected in to_pydantic for %s (id=%s)\n" "Full traceback:\n%s",
                        self.__class__.__name__,
                        getattr(self, "id", "no-id"),
                        traceback.format_exc(),
                    )
                raise
        return self.__pydantic_model__.model_validate(self)

    def to_record(self) -> "BaseModel":
        """Deprecated accessor for to_pydantic"""
        logger.warning("to_record is deprecated, use to_pydantic instead.")
        return self.to_pydantic()

    # ========================================================================
    # REDIS INTEGRATION METHODS (Hybrid: Hash for blocks/messages, JSON for memory)
    # ========================================================================

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    async def create_with_redis(
        self, db_session: AsyncSession, actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """Create record in PostgreSQL and optionally cache in Redis (async)."""
        logger.debug(
            "Creating %s with ID: %s (use_cache=%s) with actor=%s",
            self.__class__.__name__,
            self.id,
            use_cache,
            actor,
        )

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        try:
            db_session.add(self)
            await db_session.commit()
            await db_session.refresh(self)

            if use_cache:
                await self._update_redis_cache(operation="create", actor=actor)
                logger.debug("Cached %s to cache", self.__class__.__name__)
            else:
                logger.debug("Skipped cache for %s (use_cache=False)", self.__class__.__name__)

            return self
        except (DBAPIError, IntegrityError) as e:
            await db_session.rollback()
            logger.error("Failed to create %s with ID %s: %s", self.__class__.__name__, self.id, e)
            self._handle_dbapi_error(e)
        except Exception as e:
            await db_session.rollback()
            logger.error("Unexpected error creating %s with ID %s: %s", self.__class__.__name__, self.id, e)
            raise

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    async def update_with_redis(
        self, db_session: AsyncSession, actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """Update record in PostgreSQL and optionally update cache (async)."""
        logger.debug(
            "Updating %s with ID: %s (use_cache=%s) with actor=%s",
            self.__class__.__name__,
            self.id,
            use_cache,
            actor,
        )
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.set_updated_at()

        try:
            db_session.add(self)
            await db_session.commit()
            await db_session.refresh(self)

            if use_cache:
                await self._update_redis_cache(operation="update", actor=actor)
                logger.debug("Updated %s in cache", self.__class__.__name__)
            else:
                logger.debug("Skipped cache update for %s (use_cache=False)", self.__class__.__name__)

            return self
        except Exception as e:
            await db_session.rollback()
            logger.error("Failed to update %s with ID %s: %s", self.__class__.__name__, self.id, e)
            raise

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    async def delete_with_redis(
        self, db_session: AsyncSession, actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """Soft delete record in PostgreSQL and optionally remove from cache (async)."""
        logger.debug(
            "Soft deleting %s with ID: %s (use_cache=%s) with actor=%s",
            self.__class__.__name__,
            self.id,
            use_cache,
            actor,
        )

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True

        if use_cache:
            await self._update_redis_cache(operation="delete", actor=actor)
            logger.debug("Removed %s from cache", self.__class__.__name__)
        else:
            logger.debug("Skipped cache deletion for %s (use_cache=False)", self.__class__.__name__)

        return await self.update(db_session)

    async def _update_redis_cache(self, operation: str = "update", actor: Optional["Client"] = None) -> None:
        """Update cache based on table type (via cache provider). Async."""
        try:
            from mirix.database.cache_provider import get_cache_provider
            from mirix.database.redis_client import get_redis_client
            from mirix.settings import settings

            cache_provider = get_cache_provider()
            if cache_provider is None:
                return

            redis_client = get_redis_client()

            table_name = getattr(self, "__tablename__", None)
            if not table_name:
                return

            # HASH-BASED CACHING (blocks and messages - NO embeddings)
            if table_name == "block":
                cache_key = f"{cache_provider.BLOCK_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_blocks)
                return

            if table_name == "messages":
                cache_key = f"{cache_provider.MESSAGE_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_messages)
                return

            # ORGANIZATION CACHING (Hash-based)
            if table_name == "organizations":
                cache_key = f"{cache_provider.ORGANIZATION_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_organizations)
                return

            # USER CACHING (Hash-based)
            if table_name == "users":
                cache_key = f"{cache_provider.USER_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_users)
                return

            # AGENT CACHING (Hash-based, with denormalized tool_ids)
            if table_name == "agents":
                import json

                cache_key = f"{cache_provider.AGENT_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")

                    if "message_ids" in data and data["message_ids"]:
                        data["message_ids"] = json.dumps(data["message_ids"])
                    if "llm_config" in data and data["llm_config"]:
                        data["llm_config"] = json.dumps(data["llm_config"])
                    if "embedding_config" in data and data["embedding_config"]:
                        data["embedding_config"] = json.dumps(data["embedding_config"])
                    if "tool_rules" in data and data["tool_rules"]:
                        data["tool_rules"] = json.dumps(data["tool_rules"])
                    if "mcp_tools" in data and data["mcp_tools"]:
                        data["mcp_tools"] = json.dumps(data["mcp_tools"])

                    if "tools" in data and data["tools"]:
                        tool_ids = [tool.id if hasattr(tool, "id") else tool["id"] for tool in data["tools"]]
                        data["tool_ids"] = json.dumps(tool_ids)

                        for tool in data["tools"]:
                            tool_data = (
                                tool
                                if isinstance(tool, dict)
                                else tool.model_dump(mode="json") if hasattr(tool, "model_dump") else tool.__dict__
                            )
                            tool_key = f"{cache_provider.TOOL_PREFIX}{tool_data['id']}"

                            if "json_schema" in tool_data and tool_data["json_schema"]:
                                tool_data["json_schema"] = json.dumps(tool_data["json_schema"])
                            if "tags" in tool_data and tool_data["tags"]:
                                tool_data["tags"] = json.dumps(tool_data["tags"])

                            await cache_provider.set_hash(tool_key, tool_data, ttl=settings.redis_ttl_tools)

                    if "memory" in data and data["memory"]:
                        memory_obj = data["memory"]
                        if isinstance(memory_obj, dict) and "blocks" in memory_obj:
                            block_ids = [
                                block.id if hasattr(block, "id") else block["id"] for block in memory_obj["blocks"]
                            ]
                            data["memory_block_ids"] = json.dumps(block_ids)
                            data["memory_prompt_template"] = memory_obj.get("prompt_template", "")

                    if "children" in data and data["children"]:
                        children_ids = [child.id if hasattr(child, "id") else child["id"] for child in data["children"]]
                        data["children_ids"] = json.dumps(children_ids)

                        if redis_client:
                            for child_id in children_ids:
                                reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                                await redis_client.client.set(reverse_key, self.id)
                                await redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    data.pop("tools", None)
                    data.pop("memory", None)
                    data.pop("children", None)

                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_agents)
                return

            # TOOL CACHING (Hash-based)
            if table_name == "tools":
                import json

                cache_key = f"{cache_provider.TOOL_PREFIX}{self.id}"
                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")

                    if "json_schema" in data and data["json_schema"]:
                        data["json_schema"] = json.dumps(data["json_schema"])
                    if "tags" in data and data["tags"]:
                        data["tags"] = json.dumps(data["tags"])

                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_tools)
                return

            # JSON-BASED CACHING (memory tables with embeddings)
            memory_tables = {
                "episodic_memory": cache_provider.EPISODIC_PREFIX,
                "semantic_memory": cache_provider.SEMANTIC_PREFIX,
                "procedural_memory": cache_provider.PROCEDURAL_PREFIX,
                "resource_memory": cache_provider.RESOURCE_PREFIX,
                "knowledge_vault": cache_provider.KNOWLEDGE_PREFIX,
                "raw_memory": cache_provider.RAW_MEMORY_PREFIX,
            }

            if table_name in memory_tables:
                prefix = memory_tables[table_name]
                cache_key = f"{prefix}{self.id}"

                if operation == "delete":
                    await cache_provider.delete(cache_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")

                    if hasattr(self, "created_at") and self.created_at:
                        data["created_at_ts"] = self.created_at.timestamp()
                    if hasattr(self, "occurred_at") and self.occurred_at:
                        data["occurred_at_ts"] = self.occurred_at.timestamp()

                    await cache_provider.set_json(cache_key, data, ttl=settings.redis_ttl_default)

        except Exception as e:
            # Log but don't fail the operation if Redis fails
            logger.error("Failed to update cache for %s %s: %s", self.__class__.__name__, self.id, e)
            logger.info("Operation completed successfully in PostgreSQL despite cache error")
