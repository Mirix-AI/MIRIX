from typing import Any, Dict, List, Optional

from mirix.log import get_logger
from mirix.orm.block import Block as BlockModel
from mirix.orm.enums import AccessType
from mirix.orm.errors import NoResultFound
from mirix.schemas.block import Block
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.block import BlockUpdate
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types

logger = get_logger(__name__)


class BlockManager:
    """Manager class to handle business logic related to Blocks."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_or_update_block(
        self,
        block: Block,
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
    ) -> PydanticBlock:
        """
        Create a new block or update an existing one (with Redis Hash caching).

        Scope is always auto-injected from actor.write_scope unless filter_tags
        already contains a "scope" key. This ensures every block is scoped to the
        client's write_scope. Callers do not need to pass scope explicitly.

        Args:
            block: Block data to create
            actor: Client for audit trail and scope resolution
            user: Optional user for data scoping
            filter_tags: Optional extra filter tags. Scope is auto-injected from
                         actor.write_scope if not already present.

        Returns:
            PydanticBlock: The created or updated block
        """
        if filter_tags is None:
            filter_tags = {}
        if "scope" not in filter_tags and actor.write_scope:
            filter_tags["scope"] = actor.write_scope

        db_block = self.get_block_by_id(block.id, user=None)
        if db_block:
            update_data = BlockUpdate(**block.model_dump(exclude_none=True))
            return self.update_block(block.id, update_data, actor, user=user)
        else:
            with self.session_maker() as session:
                data = block.model_dump(
                    exclude_none=True,
                    exclude={"organization_id", "user_id", "filter_tags"},
                )
                final_user_id = user.id if user else None
                scope = filter_tags.get("scope")
                logger.debug(
                    "Creating block with user_id=%s, scope=%s, org_id=%s",
                    final_user_id,
                    scope,
                    actor.organization_id,
                )
                block = BlockModel(
                    **data,
                    organization_id=actor.organization_id,
                    user_id=final_user_id,
                    filter_tags=filter_tags or None,
                )
                block.create_with_redis(session, actor=actor)
                logger.debug("Block %s created with user_id=%s, scope=%s", block.id, block.user_id, scope)
            return block.to_pydantic()

    @enforce_types
    def seed_template_block_for_actor_scope_if_necessary(
        self,
        label: str,
        value: str,
        limit: int,
        actor: PydanticClient,
        default_user: PydanticUser,
    ) -> PydanticBlock | None:
        """
        Ensure a template block exists for (user, scope, label). Idempotent.

        If a block with the same (user_id, scope, label) already exists, this
        function no-ops.

        Args:
            label: Block label (e.g. "human", "persona")
            value: Initial block content
            limit: Character limit
            actor: Client for audit trail and scope resolution

        Returns:
            PydanticBlock: The created or updated block
        """
        assert actor.write_scope is not None
        scope = actor.write_scope

        # Look for existing block by key: (user_id, scope, label)
        existing = self.get_blocks(
            user=default_user,
            any_scopes=[scope],
            label=label,
            auto_create_from_default=False,
        )
        if existing:
            return None

        # Create new block
        new_block = Block(
            label=label,
            value=value,
            limit=limit,
            filter_tags={"scope": scope},
            organization_id=actor.organization_id,
            user_id=default_user.id,
            created_by_id=default_user.id,
            last_updated_by_id=default_user.id,
        )
        logger.debug(
            "Creating template block: label=%s, scope=%s, user_id=%s",
            label,
            scope,
            default_user.id,
        )
        return self.create_or_update_block(
            block=new_block,
            actor=actor,
            user=default_user,
        )

    @enforce_types
    def _invalidate_block_cache(self, block_id: str) -> None:
        """
        Invalidate caches for a block.
        Called when a block is updated or deleted to maintain cache consistency.
        """
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                cache_provider.delete(cache_key)
        except Exception as e:
            logger.warning("Failed to invalidate cache for block %s: %s", block_id, e)

    def update_block(
        self,
        block_id: str,
        block_update: BlockUpdate,
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,
    ) -> PydanticBlock:
        """
        Update a block by its ID (with Redis Hash caching).

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

            if user is not None:
                block.user_id = user.id

            block.update_with_redis(db_session=session, actor=actor)

            return block.to_pydantic()

    @enforce_types
    def delete_block(self, block_id: str, actor: PydanticClient) -> PydanticBlock:
        """Delete a block by its ID (removes from cache)."""
        from mirix.database.cache_provider import get_cache_provider

        with self.session_maker() as session:
            block = BlockModel.read(db_session=session, identifier=block_id)

            cache_provider = get_cache_provider()
            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                cache_provider.delete(cache_key)

            self._invalidate_block_cache(block_id)

            block.hard_delete(db_session=session, actor=actor)
            return block.to_pydantic()

    @enforce_types
    def get_blocks(
        self,
        user: PydanticUser,
        any_scopes: Optional[List[str]] = None,
        label: Optional[str] = None,
        id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        auto_create_from_default: bool = True,
    ) -> List[PydanticBlock]:
        """
        Retrieve blocks based on various optional filters.

        Args:
            user: User to get blocks for
            any_scopes: If provided, only return blocks whose scope matches any value
                        in this list. Pass a single-element list for exact scope match
                        (e.g. [client.write_scope]) or a multi-element list for read
                        access across scopes (e.g. client.read_scopes).
                        An empty list means no scope access — returns [].
            label: Optional label filter
            id: Optional block ID filter
            cursor: Pagination cursor
            limit: Max results
            auto_create_from_default: If True and any_scopes has exactly one scope,
                                      copy default user's template blocks when none exist
        """
        with self.session_maker() as session:
            if any_scopes is not None:
                if not any_scopes:
                    return []
                # Scope-filtered query — pushed into SQL, hits the btree index
                blocks = BlockModel.list_by_scopes(
                    db_session=session,
                    user_id=user.id,
                    organization_id=user.organization_id,
                    scopes=any_scopes,
                    label=label,
                    id=id,
                    limit=limit or 50,
                )
            else:
                # Unscoped query — returns all blocks for this user
                filters = {
                    "organization_id": user.organization_id,
                    "user_id": user.id,
                }
                if label:
                    filters["label"] = label
                if id:
                    filters["id"] = id
                blocks = BlockModel.list(db_session=session, cursor=cursor, limit=limit, **filters)

            # Auto-create from default user template if no blocks found.
            # Only auto-create when filtering by exactly one scope (write path).
            if not blocks and auto_create_from_default and any_scopes and len(any_scopes) == 1:
                scope = any_scopes[0]
                assert user.organization_id is not None
                logger.debug(
                    "No blocks found for user %s, scope %s. Creating from default user template.",
                    user.id,
                    scope,
                )
                blocks = self._copy_blocks_from_default_user(
                    session=session,
                    target_user=user,
                    scope=scope,
                    organization_id=user.organization_id,
                )

            return [block.to_pydantic() for block in blocks]

    def _copy_blocks_from_default_user(
        self, session, target_user: PydanticUser, scope: str, organization_id: str
    ) -> List[BlockModel]:
        """
        Copy template blocks from the default user to the target user for a given scope.

        Template blocks are seeded by create_meta_agent when a client provides a blocks
        config. They live on the org's default user with filter_tags={"scope": "<write_scope>"}.
        Clients sharing the same write_scope share the same template blocks.

        Args:
            session: Database session
            target_user: User to create blocks for
            scope: Scope to match template blocks and assign to new blocks
            organization_id: Organization ID

        Returns:
            List of newly created BlockModel instances
        """
        from mirix.services.user_manager import UserManager

        user_manager = UserManager()
        try:
            org_default_user = user_manager.get_or_create_org_default_user(org_id=organization_id)
            default_user_id = org_default_user.id
            logger.debug(
                "Using organization default user %s as template for user %s in org %s",
                default_user_id,
                target_user.id,
                organization_id,
            )
        except Exception as e:
            logger.warning("Failed to get org default user, falling back to global admin: %s", e)
            default_user_id = UserManager.ADMIN_USER_ID

        # Find template blocks for this scope on the default user (SQL-level scope filter)
        default_blocks = BlockModel.list_by_scopes(
            db_session=session,
            user_id=default_user_id,
            organization_id=organization_id,
            scopes=[scope],
            limit=100,
        )

        logger.debug(
            "Found %d template blocks for scope %s (default_user=%s, org=%s)",
            len(default_blocks),
            scope,
            default_user_id,
            organization_id,
        )

        if not default_blocks:
            logger.warning(
                "No template blocks found for scope %s. "
                "Ensure create_meta_agent was called with a blocks config for this scope. "
                "User %s will have no blocks.",
                scope,
                target_user.id,
            )
            return []

        new_blocks = []
        logger.debug("Starting to copy %d blocks for user %s (scope=%s)", len(default_blocks), target_user.id, scope)

        for template_block in default_blocks:
            logger.debug("Copying block %s (label=%s) from template user", template_block.id, template_block.label)

            try:
                from mirix.schemas.block import Block as PydanticBlock

                new_block_id = PydanticBlock._generate_id()

                new_block = BlockModel(
                    id=new_block_id,
                    label=template_block.label,
                    value=template_block.value,
                    limit=template_block.limit,
                    user_id=target_user.id,
                    organization_id=organization_id,
                    filter_tags={"scope": scope},
                    created_by_id=target_user.id,
                    last_updated_by_id=target_user.id,
                )

                session.add(new_block)
                session.commit()
                session.refresh(new_block)

                try:
                    new_block._update_redis_cache(operation="create", actor=None)
                    logger.debug("Cached copied block %s to cache", new_block.id)
                except Exception as e:
                    logger.warning("Failed to cache block %s to cache: %s", new_block.id, e)

                new_blocks.append(new_block)

                logger.debug(
                    "Created block %s (label=%s) for user %s, scope=%s from template %s",
                    new_block.id,
                    new_block.label,
                    target_user.id,
                    scope,
                    template_block.id,
                )
            except Exception as e:
                logger.error(
                    "Failed to copy block %s for user %s: %s", template_block.id, target_user.id, e, exc_info=True
                )
                session.rollback()
                continue

        logger.info(
            "Created %d blocks for user %s from default user template (scope=%s)",
            len(new_blocks),
            target_user.id,
            scope,
        )

        return new_blocks

    @enforce_types
    def get_block_by_id(self, block_id: str, user: Optional[PydanticUser] = None) -> Optional[PydanticBlock]:
        """Retrieve a block by its ID (with cache - Redis or IPS Cache)."""
        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                cached_data = cache_provider.get_hash(cache_key)
                if cached_data:
                    if "value" not in cached_data or cached_data["value"] is None:
                        cached_data["value"] = ""
                    return PydanticBlock(**cached_data)
        except Exception as e:
            logger.warning("Cache read failed for block %s: %s", block_id, e)

        with self.session_maker() as session:
            try:
                block = BlockModel.read(
                    db_session=session,
                    identifier=block_id,
                    user=user,
                    access_type=AccessType.USER,
                )
                pydantic_block = block.to_pydantic()

                try:
                    if cache_provider:
                        from mirix.settings import settings

                        cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                        data = pydantic_block.model_dump(mode="json")
                        cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_blocks)
                except Exception as e:
                    logger.warning("Failed to populate cache for block %s: %s", block_id, e)

                return pydantic_block
            except NoResultFound:
                return None

    @enforce_types
    def get_all_blocks_by_ids(self, block_ids: List[str], user: Optional[PydanticUser] = None) -> List[PydanticBlock]:
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
            blocks = (
                session.query(BlockModel).filter(BlockModel.user_id == user_id, BlockModel.is_deleted == False).all()
            )

            count = len(blocks)
            if count == 0:
                return 0

            block_ids = [block.id for block in blocks]

            for block in blocks:
                block.is_deleted = True
                block.set_updated_at()

            session.commit()

        for block_id in block_ids:
            self._invalidate_block_cache(block_id)

        redis_client = get_redis_client()
        if redis_client:
            for block_id in block_ids:
                redis_key = f"{redis_client.BLOCK_PREFIX}{block_id}"
                try:
                    redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    redis_client.delete(redis_key)

        return count

    def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all blocks for a user (removes from Redis cache).

        Args:
            user_id: ID of the user whose blocks to delete

        Returns:
            Number of records deleted
        """
        from mirix.database.redis_client import get_redis_client

        with self.session_maker() as session:
            block_ids = [row[0] for row in session.query(BlockModel.id).filter(BlockModel.user_id == user_id).all()]

            count = len(block_ids)
            if count == 0:
                return 0

            for block_id in block_ids:
                self._invalidate_block_cache(block_id)

            session.query(BlockModel).filter(BlockModel.user_id == user_id).delete(synchronize_session=False)

            session.commit()

        redis_client = get_redis_client()
        if redis_client and block_ids:
            redis_keys = [f"{redis_client.BLOCK_PREFIX}{block_id}" for block_id in block_ids]

            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                redis_client.client.delete(*batch)

        return count
