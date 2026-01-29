import os
from typing import List, Optional

from mirix.log import get_logger
from mirix.orm import user
from mirix.orm.block import Block as BlockModel
from mirix.orm.enums import AccessType
from mirix.orm.errors import NoResultFound
from mirix.schemas.block import Block
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.block import BlockUpdate, Human, Persona
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types, list_human_files, list_persona_files

logger = get_logger(__name__)


class BlockManager:
    """Manager class to handle business logic related to Blocks."""

    def __init__(self):
        # Fetching the db_context similarly as in ToolManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_or_update_block(
        self,
        block: Block,
        actor: PydanticClient,
        agent_id: Optional[str] = None,
        user: Optional["PydanticUser"] = None,  # NEW: Explicit user parameter for data scoping
    ) -> PydanticBlock:
        """
        Create a new block based on the Block schema (with Redis Hash caching).

        Args:
            block: Block data to create
            actor: Client for audit trail (created_by_id, last_updated_by_id)
            agent_id: Optional agent_id to associate with this block
            user_id: Optional user_id for data scoping (if None, block is not user-scoped)

        Returns:
            PydanticBlock: The created or updated block
        """
        # Check if block exists (user parameter not needed for existence check)
        db_block = self.get_block_by_id(block.id, user=None)
        if db_block:
            update_data = BlockUpdate(**block.model_dump(exclude_none=True))
            return self.update_block(block.id, update_data, actor, user=user)
        else:
            with self.session_maker() as session:
                data = block.model_dump(exclude_none=True)
                # Use explicit user_id parameter (not actor.id which is client_id)
                final_user_id = user.id if user else None
                logger.debug(
                    f"Creating block with user_id={final_user_id}, agent_id={agent_id}, org_id={actor.organization_id}"
                )
                block = BlockModel(
                    **data, organization_id=actor.organization_id, user_id=final_user_id, agent_id=agent_id
                )
                block.create_with_redis(session, actor=actor)  # Use Redis integration
                logger.debug(f"Block {block.id} created with user_id={block.user_id}, agent_id={block.agent_id}")
            return block.to_pydantic()

    @enforce_types
    def _invalidate_agent_caches_for_block(self, block_id: str) -> None:
        """
        Invalidate all agent caches that reference this block.
        Called when a block is updated or deleted to maintain cache consistency.
        """
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if redis_client:
                # Get all agent IDs that reference this block
                reverse_key = f"{redis_client.BLOCK_PREFIX}{block_id}:agents"
                agent_ids = redis_client.client.smembers(reverse_key)

                if agent_ids:
                    logger.debug("Invalidating %s agent caches due to block %s change", len(agent_ids), block_id)

                    # Delete each agent's cache
                    for agent_id in agent_ids:
                        agent_key = f"{redis_client.AGENT_PREFIX}{agent_id.decode() if isinstance(agent_id, bytes) else agent_id}"
                        redis_client.delete(agent_key)

                    # Clean up the reverse mapping
                    redis_client.delete(reverse_key)

                    logger.debug("Invalidated %s agent caches for block %s", len(agent_ids), block_id)
        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            logger.warning("Failed to invalidate agent caches for block %s: %s", block_id, e)

    def update_block(
        self,
        block_id: str,
        block_update: BlockUpdate,
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,  # NEW: Optional user parameter for consistency
    ) -> PydanticBlock:
        """
        Update a block by its ID (with Redis Hash caching and agent cache invalidation).

        Args:
            block_id: ID of the block to update
            block_update: BlockUpdate with fields to update
            actor: Client for audit trail (last_updated_by_id)
            user: Optional user if updating user field
        """
        with self.session_maker() as session:
            block = BlockModel.read(
                db_session=session, identifier=block_id, actor=actor, user=user, access_type=AccessType.USER
            )
            update_data = block_update.model_dump(exclude_unset=True, exclude_none=True)

            for key, value in update_data.items():
                setattr(block, key, value)

            # Update user_id if provided (allows changing ownership)
            if user is not None:
                block.user_id = user.id

            block.update_with_redis(db_session=session, actor=actor)  # Use Redis integration

            # Invalidate agent caches that reference this block
            self._invalidate_agent_caches_for_block(block_id)

            return block.to_pydantic()

    @enforce_types
    def delete_block(self, block_id: str, actor: PydanticClient) -> PydanticBlock:
        """Delete a block by its ID (removes from Redis cache and invalidates agent caches)."""
        with self.session_maker() as session:
            block = BlockModel.read(db_session=session, identifier=block_id)
            # Use hard_delete and manually update Redis cache
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()
            if redis_client:
                redis_key = f"{redis_client.BLOCK_PREFIX}{block_id}"
                redis_client.delete(redis_key)

            # Invalidate agent caches that reference this block
            self._invalidate_agent_caches_for_block(block_id)

            block.hard_delete(db_session=session, actor=actor)
            return block.to_pydantic()

    @enforce_types
    def get_blocks(
        self,
        user: PydanticUser,
        agent_id: Optional[str] = None,
        label: Optional[str] = None,
        id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        auto_create_from_default: bool = True,  # NEW: Auto-create user blocks from default user's template
    ) -> List[PydanticBlock]:
        """
        Retrieve blocks based on various optional filters.

        If auto_create_from_default=True and no blocks found for this user+agent,
        automatically copies blocks from the default user as a template.
        This enables lazy initialization of user-specific core memory blocks.

        Args:
            user: User to get blocks for
            agent_id: Optional agent filter
            label: Optional label filter
            id: Optional block ID filter
            cursor: Pagination cursor
            limit: Max results
            auto_create_from_default: If True, copy default user's blocks when none exist
        """
        with self.session_maker() as session:
            # Build filters
            filters = {
                "organization_id": user.organization_id,
                "user_id": user.id,
            }
            if agent_id:
                filters["agent_id"] = agent_id
            if label:
                filters["label"] = label
            if id:
                filters["id"] = id

            blocks = BlockModel.list(db_session=session, cursor=cursor, limit=limit, **filters)

            # NEW: If no blocks found and auto-create is enabled, copy from default user
            if not blocks and auto_create_from_default and agent_id:
                logger.debug(
                    "No blocks found for user %s, agent %s. Creating from default user template.", user.id, agent_id
                )
                blocks = self._copy_blocks_from_default_user(
                    session=session, target_user=user, agent_id=agent_id, organization_id=user.organization_id
                )

            return [block.to_pydantic() for block in blocks]

    def _copy_blocks_from_default_user(
        self, session, target_user: PydanticUser, agent_id: str, organization_id: str
    ) -> List[BlockModel]:
        """
        Copy blocks from the default user to the target user.

        This enables lazy initialization: when a user first uses the system,
        their core memory blocks are automatically created from the default template.

        Args:
            session: Database session
            target_user: User to create blocks for
            agent_id: Agent ID to associate blocks with
            organization_id: Organization ID

        Returns:
            List of newly created BlockModel instances
        """
        from mirix.services.user_manager import UserManager
        from mirix.utils import generate_unique_short_id

        # NEW: Get the organization-specific default user
        # This ensures we copy blocks from the correct template within the same organization
        user_manager = UserManager()
        try:
            org_default_user = user_manager.get_or_create_org_default_user(
                org_id=organization_id, client_id=None  # No specific client needed for lookup
            )
            default_user_id = org_default_user.id
            logger.debug(
                "Using organization default user %s as template for user %s in org %s",
                default_user_id,
                target_user.id,
                organization_id,
            )
        except Exception as e:
            # Fallback to global admin user if org default user creation fails
            logger.warning("Failed to get org default user, falling back to global admin: %s", e)
            default_user_id = UserManager.ADMIN_USER_ID

        # Find default user's blocks for this agent
        default_blocks = BlockModel.list(
            db_session=session,
            user_id=default_user_id,
            agent_id=agent_id,
            organization_id=organization_id,
            limit=100,  # Core memory typically has 2-10 blocks
        )

        logger.debug(
            "Found %d default template blocks for agent %s (user_id=%s, org_id=%s)",
            len(default_blocks),
            agent_id,
            default_user_id,
            organization_id,
        )

        if not default_blocks:
            logger.warning(
                "No default template blocks found for agent %s. User %s will have no blocks.", agent_id, target_user.id
            )
            return []

        # Create copies for the target user
        new_blocks = []
        logger.debug("Starting to copy %d blocks for user %s (agent=%s)", len(default_blocks), target_user.id, agent_id)

        for template_block in default_blocks:
            logger.debug("Copying block %s (label=%s) from template user", template_block.id, template_block.label)

            try:
                # Generate new unique ID using Block schema's standard ID generator
                from mirix.schemas.block import Block as PydanticBlock

                new_block_id = PydanticBlock._generate_id()
                logger.debug(f"Generated new block ID: {new_block_id}")

                # Create copy with user's ID
                new_block = BlockModel(
                    id=new_block_id,
                    label=template_block.label,
                    value=template_block.value,  # Copy the value from template
                    limit=template_block.limit,
                    user_id=target_user.id,  # Associate with target user
                    agent_id=agent_id,
                    organization_id=organization_id,
                    created_by_id=target_user.id,  # User "created" their own blocks
                    last_updated_by_id=target_user.id,
                )
                logger.debug(f"Created BlockModel instance for {new_block_id}")

                # Save to database (directly to session, not via create_with_redis to avoid nested context manager issues)
                session.add(new_block)
                logger.debug(f"Added block {new_block_id} to session")

                session.commit()
                logger.debug(f"Committed block {new_block_id} to database")

                session.refresh(new_block)
                logger.debug(f"Refreshed block {new_block_id} from database")

                # Update Redis cache manually
                try:
                    new_block._update_redis_cache(operation="create", actor=None)
                    logger.debug(f"Cached copied block {new_block.id} to Redis")
                except Exception as e:
                    logger.warning(f"Failed to cache block {new_block.id} to Redis: {e}")

                new_blocks.append(new_block)

                logger.debug(
                    "Created block %s (label=%s) for user %s from default template %s",
                    new_block.id,
                    new_block.label,
                    target_user.id,
                    template_block.id,
                )
            except Exception as e:
                logger.error(
                    "Failed to copy block %s for user %s: %s", template_block.id, target_user.id, e, exc_info=True
                )
                session.rollback()
                # Continue with next block instead of failing completely
                continue

        logger.info(
            "Created %d blocks for user %s from default user template (agent=%s)",
            len(new_blocks),
            target_user.id,
            agent_id,
        )

        return new_blocks

    @enforce_types
    def get_block_by_id(self, block_id: str, user: Optional[PydanticUser] = None) -> Optional[PydanticBlock]:
        """Retrieve a block by its ID (with Redis Hash caching - 40-60% faster!)."""
        # Try Redis cache first (Hash-based for blocks)
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.BLOCK_PREFIX}{block_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    # Normalize block data: ensure 'value' is never None (use empty string instead)
                    if "value" not in cached_data or cached_data["value"] is None:
                        cached_data["value"] = ""
                    # Cache HIT - return from Redis
                    return PydanticBlock(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            logger.warning("Redis cache read failed for block %s: %s", block_id, e)

        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                block = BlockModel.read(
                    db_session=session,
                    identifier=block_id,
                    user=user,
                    access_type=AccessType.USER,
                )
                pydantic_block = block.to_pydantic()

                # Populate Redis cache for next time
                try:
                    if redis_client:
                        from mirix.settings import settings

                        data = pydantic_block.model_dump(mode="json")
                        # model_dump(mode='json') already converts datetime to ISO format strings
                        redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_blocks)
                except Exception as e:
                    # Log but don't fail on cache population error
                    logger.warning("Failed to populate Redis cache for block %s: %s", block_id, e)

                return pydantic_block
            except NoResultFound:
                return None

    @enforce_types
    def get_all_blocks_by_ids(self, block_ids: List[str], user: Optional[PydanticUser] = None) -> List[PydanticBlock]:
        # TODO: We can do this much more efficiently by listing, instead of executing individual queries per block_id
        blocks = []
        for block_id in block_ids:
            block = self.get_block_by_id(block_id, user=user)
            blocks.append(block)
        return blocks

    def soft_delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk soft delete all blocks for a user (updates Redis cache).

        Args:
            user_id: ID of the user whose blocks to soft delete

        Returns:
            Number of records soft deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Query all non-deleted records for this user
            blocks = (
                session.query(BlockModel).filter(BlockModel.user_id == user_id, BlockModel.is_deleted == False).all()
            )

            count = len(blocks)
            if count == 0:
                return 0

            # Extract IDs BEFORE committing (to avoid detached instance errors)
            block_ids = [block.id for block in blocks]

            # Soft delete from database (set is_deleted = True directly, don't call block.delete())
            for block in blocks:
                block.is_deleted = True
                block.set_updated_at()

            session.commit()

        # Invalidate agent caches and update Redis (outside session)
        for block_id in block_ids:
            self._invalidate_agent_caches_for_block(block_id)

        redis_client = get_redis_client()
        if redis_client:
            for block_id in block_ids:
                redis_key = f"{redis_client.BLOCK_PREFIX}{block_id}"
                try:
                    redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    # If update fails, remove from cache
                    redis_client.delete(redis_key)

        return count

    def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all blocks for a user (removes from Redis cache).
        Optimized with single DB query and batch Redis deletion.

        Args:
            user_id: ID of the user whose blocks to delete

        Returns:
            Number of records deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            # Get IDs for Redis cleanup (only fetch IDs, not full objects)
            block_ids = [row[0] for row in session.query(BlockModel.id).filter(BlockModel.user_id == user_id).all()]

            count = len(block_ids)
            if count == 0:
                return 0

            # Invalidate agent caches that reference these blocks (before deletion)
            for block_id in block_ids:
                self._invalidate_agent_caches_for_block(block_id)

            # Bulk delete in single query
            session.query(BlockModel).filter(BlockModel.user_id == user_id).delete(synchronize_session=False)

            session.commit()

        # Batch delete from Redis cache (outside of session context)
        redis_client = get_redis_client()
        if redis_client and block_ids:
            redis_keys = [f"{redis_client.BLOCK_PREFIX}{block_id}" for block_id in block_ids]

            # Delete in batches to avoid command size limits
            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count

    @enforce_types
    def add_default_blocks(self, actor: PydanticClient, user: Optional[PydanticUser] = None):
        """
        Add default persona and human blocks.

        Args:
            actor: Client for audit trail
            user_id: Optional user_id for block ownership (uses default if not provided)
        """
        # Use admin user if not provided
        if user is None:
            from mirix.services.user_manager import UserManager

            user = UserManager().get_admin_user()

        for persona_file in list_persona_files():
            text = open(persona_file, "r", encoding="utf-8").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            self.create_or_update_block(Persona(value=text), actor=actor, user=user)

        for human_file in list_human_files():
            text = open(human_file, "r", encoding="utf-8").read()
            name = os.path.basename(human_file).replace(".txt", "")
            self.create_or_update_block(Human(value=text), actor=actor, user=user)
