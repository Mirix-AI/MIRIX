from typing import List, Optional, Tuple

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
    def create_admin_user(self, org_id: str = OrganizationManager.DEFAULT_ORG_ID) -> PydanticUser:
        """Create the admin user."""
        with self.session_maker() as session:
            # Make sure the org id exists
            try:
                OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(f"No organization with {org_id} exists in the organization table.")

            # Try to retrieve the user
            try:
                user = UserModel.read(db_session=session, identifier=self.ADMIN_USER_ID)
            except NoResultFound:
                # If it doesn't exist, make it
                user = UserModel(
                    id=self.ADMIN_USER_ID,
                    name=self.ADMIN_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                    is_admin=True,
                )
                user.create(session)

            return user.to_pydantic()

    @enforce_types
    def create_user(self, pydantic_user: PydanticUser, client_id: Optional[str] = None) -> PydanticUser:
        """Create a new user if it doesn't already exist (with Redis caching).

        Args:
            pydantic_user: The user data
            client_id: Optional client ID to associate the user with
        """
        with self.session_maker() as session:
            user_data = pydantic_user.model_dump()
            if client_id:
                user_data["client_id"] = client_id
            new_user = UserModel(**user_data)
            new_user.create_with_redis(session, actor=None)  # Auto-caches to Redis
            return new_user.to_pydantic()

    @enforce_types
    def update_user(self, user_update: UserUpdate) -> PydanticUser:
        """Update user details (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_update.id)

            # Update only the fields that are provided in UserUpdate
            update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_user, key, value)

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_timezone(self, timezone_str: str, user_id: str) -> PydanticUser:
        """Update the timezone of a user (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the timezone
            existing_user.timezone = timezone_str

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_status(self, user_id: str, status: str) -> PydanticUser:
        """Update the status of a user (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the status
            existing_user.status = status

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def delete_user_by_id(self, user_id: str):
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
        from mirix.database.redis_client import get_redis_client
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Soft deleting user %s and all associated records using memory managers...", user_id)

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

        episodic_count = episodic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d episodic memories for user %s", episodic_count, user_id)

        semantic_count = semantic_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d semantic memories for user %s", semantic_count, user_id)

        procedural_count = procedural_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d procedural memories for user %s", procedural_count, user_id)

        resource_count = resource_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d resource memories for user %s", resource_count, user_id)

        knowledge_count = knowledge_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d knowledge vault items for user %s", knowledge_count, user_id)

        message_count = message_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d messages for user %s", message_count, user_id)

        block_count = block_manager.soft_delete_by_user_id(user_id=user_id)
        logger.debug("Soft deleted %d blocks for user %s", block_count, user_id)

        # 2. Soft delete user
        with self.session_maker() as session:
            # Find user
            user = UserModel.read(db_session=session, identifier=user_id)
            if not user:
                logger.warning("User %s not found", user_id)
                return

            # Soft delete user (set is_deleted = True directly, don't call user.delete())
            user.is_deleted = True
            user.set_updated_at()
            session.commit()
            logger.info("Soft deleted user %s from database", user_id)

            # 3. Update Redis cache to reflect soft delete
            try:
                redis_client = get_redis_client()
                if redis_client:
                    # Update user hash with is_deleted=true
                    user_key = f"{redis_client.USER_PREFIX}{user_id}"
                    try:
                        redis_client.client.hset(user_key, "is_deleted", "true")
                        logger.debug("Updated user %s in Redis (is_deleted=true)", user_id)
                    except Exception as e:
                        logger.warning("Failed to update user in Redis, removing instead: %s", e)
                        redis_client.delete(user_key)

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
                logger.warning("Failed to update Redis cache for user %s: %s", user_id, e)

    def delete_memories_by_user_id(self, user_id: str):
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
            episodic_count = episodic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d episodic memories", episodic_count)

            semantic_count = semantic_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d semantic memories", semantic_count)

            procedural_count = procedural_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d procedural memories", procedural_count)

            resource_count = resource_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d resource memories", resource_count)

            knowledge_count = knowledge_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d knowledge vault items", knowledge_count)

            message_count = message_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d messages", message_count)

            block_count = block_manager.delete_by_user_id(user_id=user_id)
            logger.debug("Bulk deleted %d blocks", block_count)

            # Clear message_ids from ALL agents in PostgreSQL (messages are user-scoped, agents are client-scoped)
            # IMPORTANT: Keep the first message (system message) as agents need it to function
            # We need to clear message_ids from all agents that might have cached this user's messages
            with self.session_maker() as session:
                from mirix.orm.agent import Agent as AgentModel

                # Update ALL agents to keep only system messages
                # (We can't know which agents have which user's messages, so clean all)
                agents = session.query(AgentModel).all()

                for agent in agents:
                    if agent.message_ids and len(agent.message_ids) > 1:  # Has conversation messages
                        agent.message_ids = [agent.message_ids[0]]  # Keep system message only

                session.commit()
                logger.debug(
                    "Cleared conversation message_ids from %d agents in PostgreSQL (kept system messages)", len(agents)
                )

            # Invalidate agent caches that might reference deleted messages for this user
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()
            if redis_client:
                # Use SCAN to find all agent keys and delete them
                cursor = 0
                invalidated_count = 0
                while True:
                    cursor, keys = redis_client.client.scan(
                        cursor=cursor, match=f"{redis_client.AGENT_PREFIX}*", count=100
                    )
                    if keys:
                        redis_client.client.delete(*keys)
                        invalidated_count += len(keys)
                    if cursor == 0:
                        break
                if invalidated_count > 0:
                    logger.debug("Invalidated %d agent caches due to user deletion", invalidated_count)

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
    def get_user_by_id(self, user_id: str) -> PydanticUser:
        """Fetch a user by ID (with Redis Hash caching)."""
        # Try Redis cache first
        try:
            from mirix.database.redis_client import get_redis_client
            from mirix.log import get_logger

            logger = get_logger(__name__)
            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.USER_PREFIX}{user_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    logger.debug("Redis cache HIT for user %s", user_id)
                    return PydanticUser(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            from mirix.log import get_logger

            logger = get_logger(__name__)
            logger.warning("Redis cache read failed for user %s: %s", user_id, e)

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            user = UserModel.read(db_session=session, identifier=user_id)
            pydantic_user = user.to_pydantic()

            # Populate Redis cache for next time
            try:
                if redis_client:
                    from mirix.settings import settings

                    data = pydantic_user.model_dump(mode="json")
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_users)
                    logger.debug("Populated Redis cache for user %s", user_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for user %s: %s", user_id, e)

            return pydantic_user

    @enforce_types
    def get_admin_user(self) -> PydanticUser:
        """Fetch the admin user, creating it if it doesn't exist."""
        try:
            return self.get_user_by_id(self.ADMIN_USER_ID)
        except NoResultFound:
            # Admin user doesn't exist, create it
            # First ensure the default organization exists
            from mirix.services.organization_manager import OrganizationManager

            org_mgr = OrganizationManager()
            org_mgr.get_default_organization()  # Auto-creates if missing
            return self.create_admin_user(org_id=OrganizationManager.DEFAULT_ORG_ID)

    @enforce_types
    def get_or_create_org_default_user(self, org_id: str, client_id: Optional[str] = None) -> PydanticUser:
        """
        Get or create the default template user for an organization.
        This user serves as the template for copying blocks to new users.

        Args:
            org_id: Organization ID
            client_id: Optional client ID (for audit trail)

        Returns:
            PydanticUser: The default user for this organization
        """
        # Try to find existing default user for this org
        with self.session_maker() as session:
            try:
                user = (
                    session.query(UserModel)
                    .filter(
                        UserModel.name == self.DEFAULT_USER_NAME,
                        UserModel.organization_id == org_id,
                        UserModel.is_deleted == False,
                    )
                    .first()
                )

                if user:
                    logger.debug("Found existing default user %s for organization %s", user.id, org_id)
                    return user.to_pydantic()
            except Exception as e:
                logger.debug("Error finding default user: %s", e)

        # Default user doesn't exist, create it
        logger.info("Creating default template user for organization %s", org_id)

        # Generate a deterministic user_id for the default user
        default_user_id = f"user-default-{org_id}"

        try:
            # Try to get by ID first (in case it exists with that ID)
            return self.get_user_by_id(default_user_id)
        except NoResultFound:
            pass

        # Create the default user
        with self.session_maker() as session:
            user = UserModel(
                id=default_user_id,
                name=self.DEFAULT_USER_NAME,
                status="active",
                timezone=self.DEFAULT_TIME_ZONE,
                organization_id=org_id,
                client_id=client_id,  # Optional - which client created this user
            )
            user.create(session)
            logger.info("Created default template user %s for organization %s", default_user_id, org_id)
            return user.to_pydantic()

    @enforce_types
    def get_user_or_admin(self, user_id: Optional[str] = None):
        """Fetch the user or admin user."""
        if not user_id:
            return self.get_admin_user()

        try:
            return self.get_user_by_id(user_id=user_id)
        except NoResultFound:
            return self.get_admin_user()

    @enforce_types
    def list_users(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        client_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[PydanticUser]:
        """List users with pagination using cursor (id) and limit.

        Args:
            cursor: Cursor for pagination
            limit: Maximum number of users to return
            client_id: Filter by client ID (users belonging to this client)
            organization_id: Filter by organization ID
        """
        with self.session_maker() as session:
            query = session.query(UserModel).filter(UserModel.is_deleted == False)

            if client_id:
                query = query.filter(UserModel.client_id == client_id)

            if organization_id:
                query = query.filter(UserModel.organization_id == organization_id)

            query = query.order_by(UserModel.created_at.desc())

            if cursor:
                query = query.filter(UserModel.id < cursor)

            if limit:
                query = query.limit(limit)

            results = query.all()
            return [user.to_pydantic() for user in results]
