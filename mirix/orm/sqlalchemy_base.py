import random
import time
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

from sqlalchemy import String, and_, desc, func, or_, select
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError, TimeoutError
from sqlalchemy.orm import Mapped, Session, mapped_column

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
    from sqlalchemy.orm import Session

    from mirix.orm.client import Client
    from mirix.orm.user import User

logger = get_logger(__name__)


def handle_db_timeout(func):
    """Decorator to handle SQLAlchemy TimeoutError and wrap it in a custom exception."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TimeoutError as e:
            logger.error(f"Timeout while executing {func.__name__} with args {args} and kwargs {kwargs}: {e}")
            raise DatabaseTimeoutError(message=f"Timeout occurred in {func.__name__}.", original_exception=e)

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
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Check if this is a database locked error
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
                            logger.error(f"Database locked error in {func.__name__} after {max_retries} retries: {e}")
                            raise e

                        # Calculate delay with exponential backoff and jitter
                        delay = min(base_delay * (backoff_factor**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                        total_delay = delay + jitter

                        logger.warning(
                            f"Database locked in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}), retrying in {total_delay:.2f}s: {e}"
                        )
                        time.sleep(total_delay)
                        continue
                    else:
                        # Not a database locked error, re-raise immediately
                        raise e
                except Exception as e:
                    # Other exceptions should be re-raised immediately
                    raise e

            # If we get here, all retries failed
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
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DBAPIError) as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Check if this is a database locked error
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
                            logger.error(f"Database locked error in {func.__name__} after {max_retries} retries: {e}")
                            raise e

                        # Calculate delay with exponential backoff and jitter
                        delay = min(base_delay * (2.0**attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter

                        logger.warning(
                            f"Database locked in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}), retrying in {total_delay:.2f}s: {e}"
                        )
                        time.sleep(total_delay)
                        continue
                    else:
                        # Not a database locked error, re-raise immediately
                        raise e
                except Exception as e:
                    # Other exceptions should be re-raised immediately
                    raise e

            # If we get here, all retries failed
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
    def list(
        cls,
        *,
        db_session: "Session",
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
        with db_session as session:
            # If cursor provided, get the reference object
            cursor_obj = None
            if cursor:
                cursor_obj = session.get(cls, cursor)
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

            return list(session.execute(query).scalars())

    @classmethod
    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    def read(
        cls,
        db_session: "Session",
        identifier: Optional[str] = None,
        actor: Optional["Client"] = None,
        user: Optional["User"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> "SqlalchemyBase":
        """The primary accessor for an ORM record.
        Args:
            db_session: the database session to use when retrieving the record
            identifier: the identifier of the record to read, can be the id string or the UUID object for backwards compatibility
            actor: if specified, results will be scoped only to records the user is able to access
            access: if actor is specified, records will be filtered to the minimum permission level for the actor
            kwargs: additional arguments to pass to the read, used for more complex objects
        Returns:
            The matching object
        Raises:
            NoResultFound: if the object is not found
        """
        logger.debug("Reading %s with ID: %s with actor=%s", cls.__name__, identifier, actor)

        # Start the query
        query = select(cls)
        # Collect query conditions for better error reporting
        query_conditions = []

        # If an identifier is provided, add it to the query conditions
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
        if found := db_session.execute(query).scalar():
            return found

        # Construct a detailed error message based on query conditions
        conditions_str = ", ".join(query_conditions) if query_conditions else "no specific conditions"
        raise NoResultFound(f"{cls.__name__} not found with {conditions_str}")

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    def create(self, db_session: "Session", actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug(f"Creating {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        with db_session as session:
            try:
                session.add(self)
                session.commit()
                session.refresh(self)
                return self
            except (DBAPIError, IntegrityError) as e:
                session.rollback()
                logger.error(f"Failed to create {self.__class__.__name__} with ID {self.id}: {e}")
                self._handle_dbapi_error(e)
            except Exception as e:
                session.rollback()
                logger.error(f"Unexpected error creating {self.__class__.__name__} with ID {self.id}: {e}")
                raise

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    def delete(self, db_session: "Session", actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug(f"Soft deleting {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True
        return self.update(db_session)

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    def hard_delete(self, db_session: "Session", actor: Optional["Client"] = None) -> None:
        """Permanently removes the record from the database."""
        logger.debug(f"Hard deleting {self.__class__.__name__} with ID: {self.id} with actor={actor}")

        with db_session as session:
            try:
                session.delete(self)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.exception(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}")
                raise ValueError(f"Failed to hard delete {self.__class__.__name__} with ID {self.id}: {e}")
            else:
                logger.debug(f"{self.__class__.__name__} with ID {self.id} successfully hard deleted")

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    def update(self, db_session: "Session", actor: Optional["Client"] = None) -> "SqlalchemyBase":
        logger.debug(f"Updating {self.__class__.__name__} with ID: {self.id} with actor={actor}")
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.set_updated_at()

        with db_session as session:
            try:
                session.add(self)
                session.commit()
                session.refresh(self)
                return self
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update {self.__class__.__name__} with ID {self.id}: {e}")
                raise

    @classmethod
    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    def size(
        cls,
        *,
        db_session: "Session",
        actor: Optional["Client"] = None,
        access: Optional[List[Literal["read", "write", "admin"]]] = ["read"],
        access_type: AccessType = AccessType.ORGANIZATION,
        **kwargs,
    ) -> int:
        """
        Get the count of rows that match the provided filters.

        Args:
            db_session: SQLAlchemy session
            **kwargs: Filters to apply to the query (e.g., column_name=value)

        Returns:
            int: The count of rows that match the filters

        Raises:
            DBAPIError: If a database error occurs
        """
        logger.debug("Calculating size for %s with filters %s", cls.__name__, kwargs)

        with db_session as session:
            query = select(func.count()).select_from(cls)

            if actor:
                query = cls.apply_access_predicate(query, actor, access, access_type)

            # Apply filtering logic based on kwargs
            for key, value in kwargs.items():
                if value:
                    column = getattr(cls, key, None)
                    if not column:
                        raise AttributeError(f"{cls.__name__} has no attribute '{key}'")
                    if isinstance(value, (list, tuple, set)):  # Check for iterables
                        query = query.where(column.in_(value))
                    else:  # Single value for equality filtering
                        query = query.where(column == value)

            # Handle soft deletes if the class has the 'is_deleted' attribute
            if hasattr(cls, "is_deleted"):
                query = query.where(~cls.is_deleted)

            try:
                count = session.execute(query).scalar()
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
    def create_with_redis(
        self, db_session: "Session", actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """
        Create record in PostgreSQL and optionally cache in Redis.
        Uses Hash for blocks/messages, JSON for memory tables.

        Args:
            db_session: Database session
            actor: User performing the operation
            use_cache: If True, cache in Redis. If False, skip caching.
        """
        logger.debug(
            f"Creating {self.__class__.__name__} with ID: {self.id} (use_cache={use_cache}) with actor={actor}"
        )

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        with db_session as session:
            try:
                # Write to PostgreSQL (source of truth)
                session.add(self)
                session.commit()
                session.refresh(self)

                # Conditional Redis caching
                if use_cache:
                    # Write to Redis cache
                    self._update_redis_cache(operation="create", actor=actor)
                    logger.debug(f"Cached {self.__class__.__name__} to Redis")
                else:
                    logger.debug(f"Skipped Redis cache for {self.__class__.__name__} (use_cache=False)")

                return self
            except (DBAPIError, IntegrityError) as e:
                session.rollback()
                logger.error(f"Failed to create {self.__class__.__name__} with ID {self.id}: {e}")
                self._handle_dbapi_error(e)
            except Exception as e:
                session.rollback()
                logger.error(f"Unexpected error creating {self.__class__.__name__} with ID {self.id}: {e}")
                raise

    @handle_db_timeout
    @transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0)
    def update_with_redis(
        self, db_session: "Session", actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """
        Update record in PostgreSQL and optionally update Redis cache.

        Args:
            db_session: Database session
            actor: User performing the operation
            use_cache: If True, update Redis cache. If False, skip caching.
        """
        logger.debug(
            f"Updating {self.__class__.__name__} with ID: {self.id} (use_cache={use_cache}) with actor={actor}"
        )
        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.set_updated_at()

        with db_session as session:
            try:
                # Update PostgreSQL
                session.add(self)
                session.commit()
                session.refresh(self)

                # Conditional Redis cache update
                if use_cache:
                    # Update Redis cache
                    self._update_redis_cache(operation="update", actor=actor)
                    logger.debug(f"Updated {self.__class__.__name__} in Redis")
                else:
                    logger.debug(f"Skipped Redis cache update for {self.__class__.__name__} (use_cache=False)")

                return self
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update {self.__class__.__name__} with ID {self.id}: {e}")
                raise

    @handle_db_timeout
    @retry_db_operation(max_retries=3, base_delay=0.1, max_delay=2.0)
    def delete_with_redis(
        self, db_session: "Session", actor: Optional["Client"] = None, use_cache: bool = True
    ) -> "SqlalchemyBase":
        """
        Soft delete record in PostgreSQL and optionally remove from Redis cache.

        Args:
            db_session: Database session
            actor: User performing the operation
            use_cache: If True, remove from Redis cache. If False, skip cache deletion.
        """
        logger.debug(
            f"Soft deleting {self.__class__.__name__} with ID: {self.id} (use_cache={use_cache}) with actor={actor}"
        )

        if actor:
            self._set_created_and_updated_by_fields(actor.id)

        self.is_deleted = True

        # Conditional Redis cache deletion
        if use_cache:
            self._update_redis_cache(operation="delete", actor=actor)
            logger.debug(f"Removed {self.__class__.__name__} from Redis")
        else:
            logger.debug(f"Skipped Redis cache deletion for {self.__class__.__name__} (use_cache=False)")

        return self.update(db_session)

    def _update_redis_cache(self, operation: str = "update", actor: Optional["Client"] = None) -> None:
        """
        Update Redis cache based on table type.

        Args:
            operation: "create", "update", or "delete"
            actor: User performing the operation
        """
        try:
            from mirix.database.redis_client import get_redis_client
            from mirix.settings import settings

            redis_client = get_redis_client()
            if redis_client is None:
                return  # Redis not configured, skip

            table_name = getattr(self, "__tablename__", None)
            if not table_name:
                return

            # HASH-BASED CACHING (blocks and messages - NO embeddings)
            if table_name == "block":
                redis_key = f"{redis_client.BLOCK_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_blocks)
                return

            if table_name == "messages":
                redis_key = f"{redis_client.MESSAGE_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_messages)
                return

            # ORGANIZATION CACHING (Hash-based)
            if table_name == "organizations":
                redis_key = f"{redis_client.ORGANIZATION_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_organizations)
                return

            # USER CACHING (Hash-based)
            if table_name == "users":
                redis_key = f"{redis_client.USER_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_users)
                return

            # AGENT CACHING (Hash-based, with denormalized tool_ids)
            if table_name == "agents":
                import json

                redis_key = f"{redis_client.AGENT_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")

                    # Serialize complex JSON fields for Hash storage
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

                    # model_dump(mode='json') already converts datetime to ISO format strings

                    # Denormalize tools_agents: Cache tools separately, store tool_ids with agent
                    if "tools" in data and data["tools"]:
                        tool_ids = [tool.id if hasattr(tool, "id") else tool["id"] for tool in data["tools"]]
                        data["tool_ids"] = json.dumps(tool_ids)

                        # Cache each tool separately
                        for tool in data["tools"]:
                            tool_data = (
                                tool
                                if isinstance(tool, dict)
                                else tool.model_dump(mode="json") if hasattr(tool, "model_dump") else tool.__dict__
                            )
                            tool_key = f"{redis_client.TOOL_PREFIX}{tool_data['id']}"

                            # Serialize tool JSON fields
                            if "json_schema" in tool_data and tool_data["json_schema"]:
                                tool_data["json_schema"] = json.dumps(tool_data["json_schema"])
                            if "tags" in tool_data and tool_data["tags"]:
                                tool_data["tags"] = json.dumps(tool_data["tags"])

                            # model_dump(mode='json') already converts datetime to ISO format strings

                            redis_client.set_hash(tool_key, tool_data, ttl=settings.redis_ttl_tools)

                    # Denormalize memory: Store block IDs for reconstruction
                    if "memory" in data and data["memory"]:
                        memory_obj = data["memory"]
                        if isinstance(memory_obj, dict) and "blocks" in memory_obj:
                            block_ids = [
                                block.id if hasattr(block, "id") else block["id"] for block in memory_obj["blocks"]
                            ]
                            data["memory_block_ids"] = json.dumps(block_ids)
                            data["memory_prompt_template"] = memory_obj.get("prompt_template", "")

                            # Maintain reverse mapping: block -> agents (for cache invalidation)
                            for block_id in block_ids:
                                reverse_key = f"{redis_client.BLOCK_PREFIX}{block_id}:agents"
                                redis_client.client.sadd(reverse_key, self.id)
                                redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    # Denormalize children: Store child agent IDs for reconstruction (list_agents only)
                    if "children" in data and data["children"]:
                        children_ids = [child.id if hasattr(child, "id") else child["id"] for child in data["children"]]
                        data["children_ids"] = json.dumps(children_ids)

                        # Maintain reverse mapping: child -> parent (for cache invalidation)
                        for child_id in children_ids:
                            reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                            redis_client.client.set(reverse_key, self.id)
                            redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    # Remove relationship fields (cached separately or reconstructed on demand)
                    data.pop("tools", None)
                    data.pop("memory", None)
                    data.pop("children", None)  # Reconstructed in list_agents() only

                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_agents)
                return

            # TOOL CACHING (Hash-based)
            if table_name == "tools":
                import json

                redis_key = f"{redis_client.TOOL_PREFIX}{self.id}"
                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")

                    # Serialize JSON fields
                    if "json_schema" in data and data["json_schema"]:
                        data["json_schema"] = json.dumps(data["json_schema"])
                    if "tags" in data and data["tags"]:
                        data["tags"] = json.dumps(data["tags"])

                    # model_dump(mode='json') already converts datetime to ISO format strings

                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_tools)
                return

            # JSON-BASED CACHING (memory tables with embeddings)
            memory_tables = {
                "episodic_memory": redis_client.EPISODIC_PREFIX,
                "semantic_memory": redis_client.SEMANTIC_PREFIX,
                "procedural_memory": redis_client.PROCEDURAL_PREFIX,
                "resource_memory": redis_client.RESOURCE_PREFIX,
                "knowledge_vault": redis_client.KNOWLEDGE_PREFIX,
                "raw_memory": redis_client.RAW_MEMORY_PREFIX,
            }

            if table_name in memory_tables:
                prefix = memory_tables[table_name]
                redis_key = f"{prefix}{self.id}"

                if operation == "delete":
                    redis_client.delete(redis_key)
                else:
                    data = self.to_pydantic().model_dump(mode="json")
                    # model_dump(mode='json') converts datetime to ISO format strings

                    # ADD NUMERIC TIMESTAMP FIELDS FOR REDIS SEARCH SORTING
                    # RediSearch needs numeric fields to sort by (not ISO strings)
                    if hasattr(self, "created_at") and self.created_at:
                        data["created_at_ts"] = self.created_at.timestamp()
                    if hasattr(self, "occurred_at") and self.occurred_at:
                        data["occurred_at_ts"] = self.occurred_at.timestamp()

                    redis_client.set_json(redis_key, data, ttl=settings.redis_ttl_default)

        except Exception as e:
            # Log but don't fail the operation if Redis fails
            logger.error("Failed to update Redis cache for %s %s: %s", self.__class__.__name__, self.id, e)
            logger.info("Operation completed successfully in PostgreSQL despite Redis error")
