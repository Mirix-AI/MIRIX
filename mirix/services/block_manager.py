import os
from typing import List, Optional

from mirix.log import get_logger
from mirix.orm.block import Block as BlockModel
from mirix.orm.errors import NoResultFound
from mirix.schemas.block import Block, BlockUpdate, Human, Persona
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types, list_human_files, list_persona_files

from mirix.orm import user
from mirix.orm.enums import AccessType

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
        user: Optional["PydanticUser"] = None  # NEW: Explicit user parameter for data scoping
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
                # ✅ Use explicit user_id parameter (not actor.id which is client_id)
                block = BlockModel(**data, organization_id=actor.organization_id, user_id=user.id if user else None, agent_id=agent_id)
                block.create_with_redis(session, actor=actor)  # ⭐ Use Redis integration
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
                    
                    logger.debug("✅ Invalidated %s agent caches for block %s", len(agent_ids), block_id)
        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            logger.warning("Failed to invalidate agent caches for block %s: %s", block_id, e)
    
    def update_block(
        self, 
        block_id: str, 
        block_update: BlockUpdate, 
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None  # NEW: Optional user parameter for consistency
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

            block.update_with_redis(db_session=session, actor=actor)  # ⭐ Use Redis integration
            
            # ⭐ Invalidate agent caches that reference this block
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
            
            # ⭐ Invalidate agent caches that reference this block
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
    ) -> List[PydanticBlock]:
        """Retrieve blocks based on various optional filters."""
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

            blocks = BlockModel.list(
                db_session=session, cursor=cursor, limit=limit, **filters
            )

            return [block.to_pydantic() for block in blocks]

    @enforce_types
    def get_block_by_id(
        self, block_id: str, user: Optional[PydanticUser] = None
    ) -> Optional[PydanticBlock]:
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
                    if 'value' not in cached_data or cached_data['value'] is None:
                        cached_data['value'] = ''
                    # Cache HIT - return from Redis
                    return PydanticBlock(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            logger.warning("Redis cache read failed for block %s: %s", block_id, e)
        
        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            try:
                block = BlockModel.read(
                    db_session=session, identifier=block_id, actor=user, user=user, access_type=AccessType.USER
                )
                pydantic_block = block.to_pydantic()
                
                # Populate Redis cache for next time
                try:
                    if redis_client:
                        from mirix.settings import settings
                        data = pydantic_block.model_dump(mode='json')
                        # model_dump(mode='json') already converts datetime to ISO format strings
                        redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_blocks)
                except Exception as e:
                    # Log but don't fail on cache population error
                    logger.warning("Failed to populate Redis cache for block %s: %s", block_id, e)
                
                return pydantic_block
            except NoResultFound:
                return None

    @enforce_types
    def get_all_blocks_by_ids(
        self, block_ids: List[str], user: Optional[PydanticUser] = None
    ) -> List[PydanticBlock]:
        # TODO: We can do this much more efficiently by listing, instead of executing individual queries per block_id
        blocks = []
        for block_id in block_ids:
            block = self.get_block_by_id(block_id, user=user)
            blocks.append(block)
        return blocks

    @enforce_types
    def add_default_blocks(self, actor: PydanticClient, user: Optional[PydanticUser] = None):
        """
        Add default persona and human blocks.
        
        Args:
            actor: Client for audit trail
            user_id: Optional user_id for block ownership (uses default if not provided)
        """
        # Use default user if not provided
        if user is None:
            from mirix.services.user_manager import UserManager
            user = UserManager().get_default_user()
        
        for persona_file in list_persona_files():
            text = open(persona_file, "r", encoding="utf-8").read()
            name = os.path.basename(persona_file).replace(".txt", "")
            self.create_or_update_block(
                Persona(value=text), actor=actor, user=user
            )

        for human_file in list_human_files():
            text = open(human_file, "r", encoding="utf-8").read()
            name = os.path.basename(human_file).replace(".txt", "")
            self.create_or_update_block(
                Human(value=text), actor=actor, user=user
            )
