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
    async def get_default_organization(self) -> PydanticOrganization:
        """Fetch the default organization, creating it if it doesn't exist."""
        try:
            return await self.get_organization_by_id(self.DEFAULT_ORG_ID)
        except NoResultFound:
            return await self.create_default_organization()

    @enforce_types
    async def get_organization_by_id(self, org_id: str) -> Optional[PydanticOrganization]:
        """Fetch an organization by ID (with cache - Redis or IPS Cache)."""
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.ORGANIZATION_PREFIX}{org_id}"
                cached_data = await cache_provider.get_hash(cache_key)
                if cached_data:
                    logger.debug("Cache HIT for organization %s", org_id)
                    return PydanticOrganization(**cached_data)
        except Exception as e:
            logger.warning("Cache read failed for organization %s: %s", org_id, e)

        async with self.session_maker() as session:
            organization = await OrganizationModel.read(db_session=session, identifier=org_id)
            pydantic_org = organization.to_pydantic()

            try:
                if cache_provider:
                    from mirix.settings import settings

                    cache_key = f"{cache_provider.ORGANIZATION_PREFIX}{org_id}"
                    data = pydantic_org.model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_organizations)
                    logger.debug("Populated cache for organization %s", org_id)
            except Exception as e:
                logger.warning("Failed to populate cache for organization %s: %s", org_id, e)

            return pydantic_org

    @enforce_types
    async def create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        """Create a new organization."""
        try:
            org = await self.get_organization_by_id(pydantic_org.id)
            return org
        except NoResultFound:
            return await self._create_organization(pydantic_org=pydantic_org)

    @enforce_types
    async def _create_organization(self, pydantic_org: PydanticOrganization) -> PydanticOrganization:
        async with self.session_maker() as session:
            org_data = pydantic_org.model_dump()
            if org_data.get("name") is None:
                org_data["name"] = create_random_username()

            org = OrganizationModel(**org_data)
            await org.create_with_redis(session, actor=None)
            return org.to_pydantic()

    @enforce_types
    async def create_default_organization(self) -> PydanticOrganization:
        """Create the default organization."""
        return await self.create_organization(
            PydanticOrganization(name=self.DEFAULT_ORG_NAME, id=self.DEFAULT_ORG_ID)
        )

    @enforce_types
    async def update_organization_name_using_id(
        self, org_id: str, name: Optional[str] = None
    ) -> PydanticOrganization:
        """Update an organization (with cache invalidation)."""
        async with self.session_maker() as session:
            org = await OrganizationModel.read(db_session=session, identifier=org_id)
            if name:
                org.name = name
            await org.update_with_redis(session, actor=None)
            return org.to_pydantic()

    @enforce_types
    async def delete_organization_by_id(self, org_id: str) -> None:
        """Delete an organization (removes from cache)."""
        async with self.session_maker() as session:
            organization = await OrganizationModel.read(db_session=session, identifier=org_id)

            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.log import get_logger

                logger = get_logger(__name__)
                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.ORGANIZATION_PREFIX}{org_id}"
                    await cache_provider.delete(cache_key)
                    logger.debug("Removed organization %s from cache", org_id)
            except Exception as e:
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning("Failed to remove organization %s from cache: %s", org_id, e)

            await organization.hard_delete(session)

    @enforce_types
    async def list_organizations(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[PydanticOrganization]:
        """List organizations with pagination based on cursor (org_id) and limit."""
        async with self.session_maker() as session:
            results = await OrganizationModel.list(db_session=session, cursor=cursor, limit=limit)
            return [org.to_pydantic() for org in results]
