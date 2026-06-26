from typing import List, Optional

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.orm.user import User as UserModel
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.user import UserUpdate
from mirix.services.organization_manager import OrganizationManager
from mirix.utils import enforce_types

logger = get_logger(__name__)


class UserManager:
    """Manager class to handle business logic related to Users."""

    ADMIN_USER_NAME = "admin_user"
    ADMIN_USER_ID = "user-00000000-0000-4000-8000-000000000000"
    DEFAULT_USER_NAME = "default_user"  # Organization-specific default user for block templates
    DEFAULT_TIME_ZONE = "UTC (UTC+00:00)"

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create_admin_user(self, org_id: str = OrganizationManager.DEFAULT_ORG_ID) -> PydanticUser:
        """Create the admin user (async).

        ``ADMIN_USER_ID`` is a global constant; the admin row is a shared stub
        across orgs (see ``UserManager.create_user``). The pre-check is the fast
        path; the catch-and-swallow handles startup races where another process
        won between our read and our create.
        """
        from mirix.database.provider_write_retry import is_conflict
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # Verify the organization exists in Relational DB provider
            org = await provider.read("organizations", org_id)
            if org is None:
                raise ValueError(f"No organization with {org_id} exists in the organization table.")

            # Return existing admin user if already present
            existing = await provider.read("users", self.ADMIN_USER_ID)
            if existing:
                return PydanticUser(**existing)

            try:
                result = await provider.create(
                    "users",
                    {
                        "id": self.ADMIN_USER_ID,
                        "name": self.ADMIN_USER_NAME,
                        "status": "active",
                        "timezone": self.DEFAULT_TIME_ZONE,
                        "organization_id": org_id,
                        "is_admin": True,
                    },
                )
                return PydanticUser(**result)
            except Exception as exc:
                if not is_conflict(exc):
                    raise
                # Lost the race; another caller created the row. Return it.
                existing = await provider.read("users", self.ADMIN_USER_ID)
                if existing is None:
                    raise
                return PydanticUser(**existing)

        async with self.session_maker() as session:
            try:
                await OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(f"No organization with {org_id} exists in the organization table.") from None

            try:
                user = await UserModel.read(db_session=session, identifier=self.ADMIN_USER_ID)
            except NoResultFound:
                user = UserModel(
                    id=self.ADMIN_USER_ID,
                    name=self.ADMIN_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                    is_admin=True,
                )
                try:
                    await user.create(session)
                except Exception as exc:
                    if not is_conflict(exc):
                        raise
                    user = await UserModel.read(db_session=session, identifier=self.ADMIN_USER_ID)

            return user.to_pydantic()

    @enforce_types
    async def create_user(self, pydantic_user: PydanticUser) -> PydanticUser:
        """Create a new user, or return the existing row if id is already taken.

        ``users.id`` is globally unique (id-only PK on every backend). The
        ``users`` row is a shared stub across orgs — per-org isolation lives on
        child tables (memories, blocks, messages), not on this row. Two callers
        using the same user id under different orgs share this row and that is
        the intended behavior. Catch the PK conflict and return the existing
        row instead of erroring.

        Args:
            pydantic_user: The user data
        """
        from mirix.database.provider_write_retry import is_conflict
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            user_data = pydantic_user.model_dump()
            try:
                result = await provider.create("users", user_data)
                return PydanticUser(**result)
            except Exception as exc:
                if not is_conflict(exc):
                    raise
                existing = await provider.read("users", pydantic_user.id)
                if existing is None:
                    raise
                return PydanticUser(**existing)

        async with self.session_maker() as session:
            user_data = pydantic_user.model_dump()
            new_user = UserModel(**user_data)
            try:
                await new_user.create_with_redis(session, actor=None)
                return new_user.to_pydantic()
            except Exception as exc:
                if not is_conflict(exc):
                    raise
            existing = await UserModel.read(db_session=session, identifier=pydantic_user.id)
            return existing.to_pydantic()

    @enforce_types
    async def update_user(self, user_update: UserUpdate) -> PydanticUser:
        """Update user details (with cache invalidation)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
            result = await provider.update("users", user_update.id, update_data)
            return PydanticUser(**result)

        async with self.session_maker() as session:
            existing_user = await UserModel.read(db_session=session, identifier=user_update.id)
            update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_user, key, value)
            await existing_user.update_with_redis(session, actor=None)
            return existing_user.to_pydantic()

    @enforce_types
    async def update_user_timezone(self, timezone_str: str, user_id: str) -> PydanticUser:
        """Update the timezone of a user (with cache invalidation)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.update("users", user_id, {"timezone": timezone_str})
            return PydanticUser(**result)

        async with self.session_maker() as session:
            existing_user = await UserModel.read(db_session=session, identifier=user_id)
            existing_user.timezone = timezone_str
            await existing_user.update_with_redis(session, actor=None)
            return existing_user.to_pydantic()

    @enforce_types
    async def update_user_status(self, user_id: str, status: str) -> PydanticUser:
        """Update the status of a user (with cache invalidation)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.update("users", user_id, {"status": status})
            return PydanticUser(**result)

        async with self.session_maker() as session:
            existing_user = await UserModel.read(db_session=session, identifier=user_id)
            existing_user.status = status
            await existing_user.update_with_redis(session, actor=None)
            return existing_user.to_pydantic()

    @enforce_types
    async def delete_user_by_id(self, user_id: str):
        """
        Soft delete a user and cascade soft delete to all associated records using memory managers.

        Cleanup workflow:
        1. Soft delete all memory records using memory managers:
           - Episodic memories
           - Semantic memories
           - Procedural memories
           - Resource memories
           - Knowledge vault items
           - Messages
           - Blocks

        2. Database (PostgreSQL):
           - Set user.is_deleted = True

        3. Redis Cache:
           - Update user hash with is_deleted=true
           - Memory cache entries updated by managers with is_deleted=true

        Args:
            user_id: ID of the user to soft delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Soft deleting user %s and all associated records using memory managers...", user_id)

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # raw_memory is included here; messages is handled separately below
            # via mutate_using_named_query (messages is an engine table, not IEDM).
            memory_tables = [
                "episodic_memory",
                "semantic_memory",
                "procedural_memory",
                "resource_memory",
                "knowledge_vault",
                "raw_memory",
                "block",
            ]
            soft_deleted = 0
            for table in memory_tables:
                rows = await provider.find_using_named_query(
                    table,
                    f"user_manager.list_ids_{table}_by_user",
                    params={"userId": user_id},
                    page_size=5000,
                )
                ids = [r.get("id") for r in rows if r.get("id")]
                if not ids:
                    continue
                result = await provider.bulk_delete(table, ids, soft=True)
                soft_deleted += int(result.get("success", 0) or 0)

            # Soft delete messages for this user (engine table — no domain events needed).
            await provider.mutate_using_named_query(
                "messages",
                "message_manager.update_by_user_id",
                params={"userId": user_id},
            )

            # Soft delete user record through provider path.
            await provider.delete("users", user_id, soft=True)

            # Invalidate user cache using cache provider abstraction.
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    await cache_provider.delete(f"{cache_provider.USER_PREFIX}{user_id}")
            except Exception as e:
                logger.warning("Failed to invalidate user cache for %s: %s", user_id, e)

            logger.info(
                "Soft deleted user %s via provider path (%d related records)",
                user_id,
                soft_deleted,
            )
            return

        # Import memory managers
        from mirix.services.block_manager import BlockManager
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
        from mirix.services.message_manager import MessageManager
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager
        from mirix.services.resource_memory_manager import ResourceMemoryManager
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        # 1. Soft delete all memory records using memory managers
        episodic_manager = EpisodicMemoryManager()
        semantic_manager = SemanticMemoryManager()
        procedural_manager = ProceduralMemoryManager()
        resource_manager = ResourceMemoryManager()
        knowledge_manager = KnowledgeVaultManager()
        message_manager = MessageManager()
        block_manager = BlockManager()

        episodic_count = await episodic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d episodic memories for user %s", episodic_count, user_id)

        semantic_count = await semantic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d semantic memories for user %s", semantic_count, user_id)

        procedural_count = await procedural_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d procedural memories for user %s", procedural_count, user_id)

        resource_count = await resource_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d resource memories for user %s", resource_count, user_id)

        knowledge_count = await knowledge_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d knowledge vault items for user %s", knowledge_count, user_id)

        message_count = await message_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d messages for user %s", message_count, user_id)

        block_count = await block_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d blocks for user %s", block_count, user_id)

        # 2. Soft delete user
        async with self.session_maker() as session:
            # Find user
            user = await UserModel.read(db_session=session, identifier=user_id)
            if not user:
                logger.warning("User %s not found", user_id)
                return

            # Soft delete user (set is_deleted = True directly, don't call user.delete())
            user.is_deleted = True
            user.set_updated_at()
            await session.commit()
            logger.info("Soft deleted user %s from database", user_id)

            # 3. Invalidate Redis cache (remove key so soft-deleted user is not served from cache)
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    user_key = f"{cache_provider.USER_PREFIX}{user_id}"
                    await cache_provider.delete(user_key)
                    logger.debug("Removed soft-deleted user %s from cache", user_id)

                logger.info(
                    "User %s and all associated records soft deleted: "
                    "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages, %d blocks",
                    user_id,
                    episodic_count,
                    semantic_count,
                    procedural_count,
                    resource_count,
                    knowledge_count,
                    message_count,
                    block_count,
                )
            except Exception as e:
                logger.warning("Failed to update cache for user %s: %s", user_id, e)

    async def _invalidate_agent_cache_for_user(self, user_id: str) -> None:
        """Scan and delete all Redis agent-cache entries.

        Called after bulk-deleting a user's memories so stale agent state
        (which may embed message/memory references) is evicted.  This mirrors
        the equivalent loop in the PG branch of ``delete_memories_by_user_id``.
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()
            if redis_client:
                scan_cursor = 0
                invalidated = 0
                while True:
                    scan_cursor, keys = await redis_client.client.scan(
                        cursor=scan_cursor,
                        match=f"{redis_client.AGENT_PREFIX}*",
                        count=100,
                    )
                    if keys:
                        await redis_client.client.delete(*keys)
                        invalidated += len(keys)
                    if scan_cursor == 0:
                        break
                if invalidated:
                    logger.debug(
                        "Invalidated %d agent caches after deleting memories for user %s",
                        invalidated,
                        user_id,
                    )
        except Exception as e:
            logger.warning("Failed to invalidate agent cache for user %s: %s", user_id, e)

    async def delete_memories_by_user_id(self, user_id: str):
        """
        Hard delete memories, messages, and blocks for a user using memory managers' bulk delete.

        This permanently removes data records while preserving the user record.
        Uses optimized bulk delete methods in each manager for efficient deletion.

        Cleanup workflow:
        1. Call each memory manager's delete_by_user_id() method
           - EpisodicMemoryManager.delete_by_user_id()
           - SemanticMemoryManager.delete_by_user_id()
           - ProceduralMemoryManager.delete_by_user_id()
           - ResourceMemoryManager.delete_by_user_id()
           - KnowledgeVaultManager.delete_by_user_id()
           - MessageManager.delete_by_user_id()
           - BlockManager.delete_by_user_id()
        2. Each manager handles:
           - Bulk database deletion
           - Redis cache cleanup
           - Business logic
        3. PRESERVE: user record

        Args:
            user_id: ID of the user whose memories to delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Bulk deleting memories for user %s using memory managers (preserving user record)...", user_id)

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # raw_memory is included here; messages is handled separately below
            # via mutate_using_named_query (messages is an engine table, not IEDM).
            memory_tables = [
                "episodic_memory",
                "semantic_memory",
                "procedural_memory",
                "resource_memory",
                "knowledge_vault",
                "raw_memory",
                "block",
            ]
            total_deleted = 0
            for table in memory_tables:
                rows = await provider.find_using_named_query(
                    table,
                    f"user_manager.list_ids_{table}_by_user",
                    params={"userId": user_id},
                    page_size=5000,
                )
                ids = [r.get("id") for r in rows if r.get("id")]
                if not ids:
                    continue
                result = await provider.bulk_delete(table, ids, soft=False)
                deleted = int(result.get("success", 0) or 0)
                total_deleted += deleted
                logger.debug("Bulk deleted %d %s records via provider", deleted, table)

            # Hard delete messages for this user (engine table — no domain events needed).
            await provider.mutate_using_named_query(
                "messages",
                "message_manager.update_by_user_id",
                params={"userId": user_id},
            )

            logger.info(
                "Bulk deleted memories for user %s via provider path (%d records)",
                user_id,
                total_deleted,
            )
            # Invalidate agent caches that might reference deleted messages/memories
            # for this user — mirrors the equivalent cleanup in the PG branch.
            await self._invalidate_agent_cache_for_user(user_id)
            return

        # Import managers
        from mirix.services.block_manager import BlockManager
        from mirix.services.episodic_memory_manager import EpisodicMemoryManager
        from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
        from mirix.services.message_manager import MessageManager
        from mirix.services.procedural_memory_manager import ProceduralMemoryManager
        from mirix.services.resource_memory_manager import ResourceMemoryManager
        from mirix.services.semantic_memory_manager import SemanticMemoryManager

        # Initialize managers
        episodic_manager = EpisodicMemoryManager()
        semantic_manager = SemanticMemoryManager()
        procedural_manager = ProceduralMemoryManager()
        resource_manager = ResourceMemoryManager()
        knowledge_manager = KnowledgeVaultManager()
        message_manager = MessageManager()
        block_manager = BlockManager()

        # Use managers' bulk delete methods
        try:
            # Bulk delete memories using manager methods
            episodic_count = await episodic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d episodic memories", episodic_count)

            semantic_count = await semantic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d semantic memories", semantic_count)

            procedural_count = await procedural_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d procedural memories", procedural_count)

            resource_count = await resource_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d resource memories", resource_count)

            knowledge_count = await knowledge_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d knowledge vault items", knowledge_count)

            message_count = await message_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d messages", message_count)

            block_count = await block_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d blocks", block_count)

            # Messages for this user are already deleted by delete_by_user_id above.
            # No message_ids maintenance needed (column removed).

            # Invalidate agent caches that might reference deleted messages for this user.
            await self._invalidate_agent_cache_for_user(user_id)

            logger.info(
                "Bulk deleted all memories for user %s: "
                "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages, %d blocks "
                "(user record preserved)",
                user_id,
                episodic_count,
                semantic_count,
                procedural_count,
                resource_count,
                knowledge_count,
                message_count,
                block_count,
            )
        except Exception as e:
            logger.error("Failed to bulk delete memories for user %s: %s", user_id, e)
            raise

    @enforce_types
    async def get_user_by_id(self, user_id: str) -> PydanticUser:
        """Fetch a user by ID (with cache - Redis or Cache provider)."""
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.USER_PREFIX}{user_id}"
                cached_data = await cache_provider.get_hash(cache_key)
                if cached_data:
                    logger.debug("Cache HIT for user %s", user_id)
                    return PydanticUser(**cached_data)
        except Exception as e:
            logger.debug("Cache read failed for user %s: %s", user_id, e)

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            rows = await provider.find_using_named_query(
                "users",
                "user_manager.get_user_by_id",
                params={"id": user_id},
                page_size=1,
            )
            if not rows:
                raise NoResultFound(f"User {user_id} not found")
            pydantic_user = PydanticUser(**rows[0])
            try:
                if cache_provider:
                    from mirix.settings import settings

                    cache_key = f"{cache_provider.USER_PREFIX}{user_id}"
                    data = pydantic_user.model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_users)
            except Exception as e:
                logger.warning("Failed to populate cache for user %s: %s", user_id, e)
            return pydantic_user

        async with self.session_maker() as session:
            user = await UserModel.read(db_session=session, identifier=user_id)
            pydantic_user = user.to_pydantic()

            try:
                if cache_provider:
                    from mirix.settings import settings

                    cache_key = f"{cache_provider.USER_PREFIX}{user_id}"
                    data = pydantic_user.model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_users)
                    logger.debug("Populated cache for user %s", user_id)
            except Exception as e:
                logger.warning("Failed to populate cache for user %s: %s", user_id, e)

            return pydantic_user

    @enforce_types
    async def get_admin_user(self) -> PydanticUser:
        """Fetch the admin user, creating it if it doesn't exist."""
        try:
            return await self.get_user_by_id(self.ADMIN_USER_ID)
        except NoResultFound:
            # Admin user doesn't exist, create it
            # First ensure the default organization exists
            from mirix.services.organization_manager import OrganizationManager

            org_mgr = OrganizationManager()
            await org_mgr.get_default_organization()  # Auto-creates if missing
            return await self.create_admin_user(org_id=OrganizationManager.DEFAULT_ORG_ID)

    @enforce_types
    async def get_or_create_org_default_user(self, org_id: str) -> PydanticUser:
        """
        Get or create the default template user for an organization.
        This user serves as the template for copying blocks to new users.

        Args:
            org_id: Organization ID

        Returns:
            PydanticUser: The default user for this organization
        """
        from mirix.database.relational_provider import get_relational_provider

        # Deterministic ID keeps get-or-create idempotent in both backends
        default_user_id = f"user-default-{org_id}"

        provider = get_relational_provider()
        if provider:
            existing = await provider.read("users", default_user_id)
            if existing:
                logger.debug("Found existing default user %s for organization %s", default_user_id, org_id)
                return PydanticUser(**existing)

            logger.info("Creating default template user for organization %s", org_id)
            try:
                result = await provider.create(
                    "users",
                    {
                        "id": default_user_id,
                        "name": self.DEFAULT_USER_NAME,
                        "status": "active",
                        "timezone": self.DEFAULT_TIME_ZONE,
                        "organization_id": org_id,
                    },
                )
                logger.info("Created default template user %s for organization %s", default_user_id, org_id)
                return PydanticUser(**result)
            except Exception as create_err:
                # Handle race condition: another request may have created it concurrently
                error_msg = str(create_err).lower()
                if "unique" in error_msg or "duplicate" in error_msg or "already exists" in error_msg:
                    logger.debug("Default user creation race condition, retrying lookup: %s", create_err)
                    existing = await provider.read("users", default_user_id)
                    if existing:
                        return PydanticUser(**existing)
                raise

        # PostgreSQL fallback path (no Relational DB provider registered)
        # Try to find existing default user for this org
        async with self.session_maker() as session:
            try:
                stmt = (
                    select(UserModel)
                    .where(
                        UserModel.name == self.DEFAULT_USER_NAME,
                        UserModel.organization_id == org_id,
                        UserModel.is_deleted == False,
                    )
                    .limit(1)
                )
                result = await session.execute(stmt)
                user = result.scalar_one_or_none()

                if user:
                    logger.debug("Found existing default user %s for organization %s", user.id, org_id)
                    return user.to_pydantic()
            except Exception as e:
                logger.debug("Error finding default user: %s", e)

        # Default user doesn't exist, create it
        logger.info("Creating default template user for organization %s", org_id)

        try:
            # Try to get by ID first (in case it exists with that ID)
            return await self.get_user_by_id(default_user_id)
        except NoResultFound:
            pass

        # Create the default user (handle race condition from concurrent requests)
        try:
            async with self.session_maker() as session:
                user = UserModel(
                    id=default_user_id,
                    name=self.DEFAULT_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                )
                await user.create(session)
                logger.info("Created default template user %s for organization %s", default_user_id, org_id)
                return user.to_pydantic()
        except Exception as create_err:
            error_msg = str(create_err).lower()
            if "unique" in error_msg or "duplicate" in error_msg or "already exists" in error_msg:
                logger.debug("Default user creation race condition, retrying lookup: %s", create_err)
                return await self.get_user_by_id(default_user_id)
            raise

    @enforce_types
    async def get_user_or_admin(self, user_id: Optional[str] = None):
        """Fetch the user or admin user."""
        if not user_id:
            return await self.get_admin_user()

        try:
            return await self.get_user_by_id(user_id=user_id)
        except NoResultFound:
            return await self.get_admin_user()

    @enforce_types
    async def list_users(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        organization_id: Optional[str] = None,
    ) -> List[PydanticUser]:
        """List users with pagination using cursor (id) and limit.

        Args:
            cursor: Cursor for pagination
            limit: Maximum number of users to return
            organization_id: Filter by organization ID
        """
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            results = await provider.find_using_named_query(
                "users",
                "user_manager.list_users",
                params={"cursor": cursor},
                page_size=limit or 50,
            )
            return [PydanticUser(**r) for r in results]

        async with self.session_maker() as session:
            stmt = select(UserModel).where(UserModel.is_deleted == False)

            if organization_id:
                stmt = stmt.where(UserModel.organization_id == organization_id)

            stmt = stmt.order_by(UserModel.created_at.desc())

            if cursor:
                stmt = stmt.where(UserModel.id < cursor)

            if limit:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            results = result.scalars().all()
            return [user.to_pydantic() for user in results]
