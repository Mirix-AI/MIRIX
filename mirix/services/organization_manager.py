from typing import List, Optional

from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.utils import create_random_username, enforce_types


class OrganizationManager:
    """Manager class to handle business logic related to Organizations."""

    DEFAULT_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
    DEFAULT_ORG_NAME = "default_org"

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def get_default_organization(self) -> PydanticOrganization:
        """Fetch the default organization, creating it if it doesn't exist."""
        try:
            return self.get_organization_by_id(self.DEFAULT_ORG_ID)
        except NoResultFound:
            # Default organization doesn't exist, create it
            return self.create_default_organization()

    @enforce_types
    def get_organization_by_id(self, org_id: str) -> Optional[PydanticOrganization]:
        """Fetch an organization by ID (with Redis Hash caching)."""
        # Try Redis cache first
        try:
            from mirix.database.redis_client import get_redis_client
            from mirix.log import get_logger

            logger = get_logger(__name__)
            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.ORGANIZATION_PREFIX}{org_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    logger.debug("Redis cache HIT for organization %s", org_id)
                    return PydanticOrganization(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            from mirix.log import get_logger

            logger = get_logger(__name__)
            logger.warning("Redis cache read failed for organization %s: %s", org_id, e)

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            organization = OrganizationModel.read(db_session=session, identifier=org_id)
            pydantic_org = organization.to_pydantic()

            # Populate Redis cache for next time
            try:
                if redis_client:
                    from mirix.settings import settings

                    data = pydantic_org.model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_organizations)
                    logger.debug("Populated Redis cache for organization %s", org_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for organization %s: %s", org_id, e)

            return pydantic_org

    @enforce_types
    def create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Create a new organization."""
        try:
            org = self.get_organization_by_id(pydantic_org.id)
            return org
        except NoResultFound:
            return self._create_organization(pydantic_org=pydantic_org)

    @enforce_types
    def _create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        with self.session_maker() as session:
            # Generate a random name if none provided
            org_data = pydantic_org.model_dump()
            if org_data.get("name") is None:
                org_data["name"] = create_random_username()

            org = OrganizationModel(**org_data)
            org.create_with_redis(session, actor=None)  # Auto-caches to Redis
            return org.to_pydantic()

    @enforce_types
    def create_default_organization(self) -> PydanticOrganization:
        """Create the default organization."""
        return self.create_organization(PydanticOrganization(name=self.DEFAULT_ORG_NAME, id=self.DEFAULT_ORG_ID))

    @enforce_types
    def update_organization_name_using_id(self, org_id: str, name: Optional[str] = None) -> PydanticOrganization:
        """Update an organization (with Redis cache invalidation)."""
        with self.session_maker() as session:
            org = OrganizationModel.read(db_session=session, identifier=org_id)
            if name:
                org.name = name
            org.update_with_redis(session, actor=None)  # Updates Redis cache
            return org.to_pydantic()

    @enforce_types
    def delete_organization_by_id(self, org_id: str):
        """Delete an organization (removes from Redis cache)."""
        with self.session_maker() as session:
            organization = OrganizationModel.read(db_session=session, identifier=org_id)

            # Remove from Redis cache before hard delete
            try:
                from mirix.database.redis_client import get_redis_client
                from mirix.log import get_logger

                logger = get_logger(__name__)
                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.ORGANIZATION_PREFIX}{org_id}"
                    redis_client.delete(redis_key)
                    logger.debug("Removed organization %s from Redis cache", org_id)
            except Exception as e:
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning("Failed to remove organization %s from Redis cache: %s", org_id, e)

            organization.hard_delete(session)

    @enforce_types
    def list_organizations(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticOrganization]:
        """List organizations with pagination based on cursor (org_id) and limit."""
        with self.session_maker() as session:
            results = OrganizationModel.list(db_session=session, cursor=cursor, limit=limit)
            return [org.to_pydantic() for org in results]
