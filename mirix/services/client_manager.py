from typing import List, Optional

from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.orm.client import Client as ClientModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.client import ClientUpdate
from mirix.services.organization_manager import OrganizationManager
from mirix.utils import enforce_types


class ClientManager:
    """Manager class to handle business logic related to Clients."""

    DEFAULT_CLIENT_NAME = "default_client"
    DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_default_client(
        self, org_id: str = OrganizationManager.DEFAULT_ORG_ID
    ) -> PydanticClient:
        """Create the default client."""
        with self.session_maker() as session:
            # Make sure the org id exists
            try:
                OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(
                    f"No organization with {org_id} exists in the organization table."
                )

            # Try to retrieve the client
            try:
                client = ClientModel.read(
                    db_session=session, identifier=self.DEFAULT_CLIENT_ID
                )
            except NoResultFound:
                # If it doesn't exist, make it
                client = ClientModel(
                    id=self.DEFAULT_CLIENT_ID,
                    name=self.DEFAULT_CLIENT_NAME,
                    status="active",
                    scope="",
                    organization_id=org_id,
                )
                client.create(session)

            return client.to_pydantic()

    @enforce_types
    def create_client(self, pydantic_client: PydanticClient) -> PydanticClient:
        """Create a new client if it doesn't already exist (with Redis caching)."""
        with self.session_maker() as session:
            new_client = ClientModel(**pydantic_client.model_dump())
            new_client.create_with_redis(session, actor=None)  # Auto-caches to Redis
            return new_client.to_pydantic()

    @enforce_types
    def update_client(self, client_update: ClientUpdate) -> PydanticClient:
        """Update client details (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing client by ID
            existing_client = ClientModel.read(
                db_session=session, identifier=client_update.id
            )
            
            # Update only the fields that are provided in ClientUpdate
            update_data = client_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_client, key, value)
            
            # Commit the updated client and update cache
            existing_client.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_client.to_pydantic()

    @enforce_types
    def update_client_status(self, client_id: str, status: str) -> PydanticClient:
        """Update the status of a client (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing client by ID
            existing_client = ClientModel.read(db_session=session, identifier=client_id)

            # Update the status
            existing_client.status = status

            # Commit the updated client and update cache
            existing_client.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_client.to_pydantic()

    @enforce_types
    def soft_delete_client(self, client_id: str) -> PydanticClient:
        """
        Soft delete a client (marks as deleted, keeps in database).
        
        Args:
            client_id: The client ID to soft delete
            
        Returns:
            The soft-deleted client
            
        Raises:
            NoResultFound: If client not found
        """
        with self.session_maker() as session:
            # Retrieve the client
            client = ClientModel.read(db_session=session, identifier=client_id)
            
            # Soft delete using ORM's delete method (sets is_deleted=True)
            client.delete(session, actor=None)
            
            # Update Redis cache (remove from cache since it's deleted)
            try:
                from mirix.database.redis_client import get_redis_client
                from mirix.log import get_logger
                
                logger = get_logger(__name__)
                redis_client = get_redis_client()
                if redis_client:
                    # Remove from cache since it's deleted
                    redis_key = f"{redis_client.CLIENT_PREFIX}{client_id}"
                    redis_client.delete(redis_key)
                    logger.debug("Removed soft-deleted client %s from Redis cache", client_id)
            except Exception as e:
                from mirix.log import get_logger
                logger = get_logger(__name__)
                logger.warning("Failed to update Redis for soft-deleted client %s: %s", client_id, e)
            
            return client.to_pydantic()

    @enforce_types
    def delete_client_by_id(self, client_id: str):
        """Hard delete a client and their associated records (removes from Redis cache)."""
        with self.session_maker() as session:
            # Delete from client table
            client = ClientModel.read(db_session=session, identifier=client_id)

            # Remove from Redis cache before hard delete
            try:
                from mirix.database.redis_client import get_redis_client
                from mirix.log import get_logger

                logger = get_logger(__name__)
                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.CLIENT_PREFIX}{client_id}"
                    redis_client.delete(redis_key)
                    logger.debug("Removed client %s from Redis cache", client_id)
            except Exception as e:
                from mirix.log import get_logger
                logger = get_logger(__name__)
                logger.warning("Failed to remove client %s from Redis cache: %s", client_id, e)

            client.hard_delete(session)
            session.commit()

    @enforce_types
    def get_client_by_id(self, client_id: str) -> PydanticClient:
        """Fetch a client by ID (with Redis Hash caching)."""
        # Try Redis cache first
        try:
            from mirix.database.redis_client import get_redis_client
            from mirix.log import get_logger

            logger = get_logger(__name__)
            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.CLIENT_PREFIX}{client_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    logger.debug("✅ Redis cache HIT for client %s", client_id)
                    return PydanticClient(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            from mirix.log import get_logger
            logger = get_logger(__name__)
            logger.warning("Redis cache read failed for client %s: %s", client_id, e)

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            client = ClientModel.read(db_session=session, identifier=client_id)
            pydantic_client = client.to_pydantic()

            # Populate Redis cache for next time
            try:
                if redis_client:
                    from mirix.settings import settings
                    data = pydantic_client.model_dump(mode='json')
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_clients)
                    logger.debug("Populated Redis cache for client %s", client_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for client %s: %s", client_id, e)

            return pydantic_client

    @enforce_types
    def get_default_client(self) -> PydanticClient:
        """Fetch the default client."""
        return self.get_client_by_id(self.DEFAULT_CLIENT_ID)

    @enforce_types
    def get_client_or_default(self, client_id: Optional[str] = None):
        """Fetch the client or default client."""
        if not client_id:
            return self.get_default_client()

        try:
            return self.get_client_by_id(client_id=client_id)
        except NoResultFound:
            return self.get_default_client()

    @enforce_types
    def list_clients(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[PydanticClient]:
        """List clients with pagination using cursor (id) and limit."""
        with self.session_maker() as session:
            results = ClientModel.list(db_session=session, cursor=cursor, limit=limit)
            return [client.to_pydantic() for client in results]
