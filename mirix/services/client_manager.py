from typing import List, Optional

from mirix.orm.client import Client as ClientModel
from mirix.orm.client_api_key import ClientApiKey as ClientApiKeyModel
from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.client import ClientUpdate
from mirix.schemas.client_api_key import ClientApiKey as PydanticClientApiKey
from mirix.schemas.client_api_key import ClientApiKeyCreate
from mirix.security.api_keys import hash_api_key
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
    def create_default_client(self, org_id: str = OrganizationManager.DEFAULT_ORG_ID) -> PydanticClient:
        """Create the default client."""
        with self.session_maker() as session:
            # Make sure the org id exists
            try:
                OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(f"No organization with {org_id} exists in the organization table.")

            # Try to retrieve the client
            try:
                client = ClientModel.read(db_session=session, identifier=self.DEFAULT_CLIENT_ID)
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
            existing_client = ClientModel.read(db_session=session, identifier=client_update.id)

            # Update only the fields that are provided in ClientUpdate
            update_data = client_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_client, key, value)

            # Commit the updated client and update cache
            existing_client.update_with_redis(session, actor=None)  # Updates Redis cache
            return existing_client.to_pydantic()

    @enforce_types
    def create_client_api_key(
        self,
        client_id: str,
        api_key: str,
        name: Optional[str] = None,
        permission: str = "all",
        user_id: Optional[str] = None,
    ) -> PydanticClientApiKey:
        """Create a new API key for a client."""
        hashed = hash_api_key(api_key)
        with self.session_maker() as session:
            # Verify client exists
            existing_client = ClientModel.read(db_session=session, identifier=client_id)

            # Create new API key with full Pydantic model (generates ID)
            api_key_pydantic = PydanticClientApiKey(
                client_id=client_id,
                organization_id=existing_client.organization_id,
                api_key_hash=hashed,
                name=name,
                status="active",
                permission=permission,
                user_id=user_id,
            )
            # Convert to ORM model
            new_api_key = ClientApiKeyModel(**api_key_pydantic.model_dump())
            new_api_key.create(session)
            return new_api_key.to_pydantic()

    @enforce_types
    def set_client_api_key(self, client_id: str, api_key: str, name: Optional[str] = None) -> PydanticClientApiKey:
        """
        Create a new API key for a client (deprecated name, use create_client_api_key).

        This method now creates a new API key entry in the client_api_keys table.
        For backward compatibility with existing scripts.
        """
        return self.create_client_api_key(client_id, api_key, name)

    @enforce_types
    def get_client_by_api_key(self, api_key: str) -> Optional[PydanticClient]:
        """Lookup a client via API key (hash match) from the client_api_keys table."""
        hashed = hash_api_key(api_key)
        with self.session_maker() as session:
            # Query the ClientApiKey table
            api_key_record = (
                session.query(ClientApiKeyModel)
                .filter(
                    ClientApiKeyModel.api_key_hash == hashed,
                    ClientApiKeyModel.status == "active",
                    ClientApiKeyModel.is_deleted == False,
                )
                .first()
            )
            if not api_key_record:
                return None

            # Get the associated client
            client = ClientModel.read(db_session=session, identifier=api_key_record.client_id)
            if client.is_deleted or client.status != "active":
                return None

            return client.to_pydantic()

    @enforce_types
    def list_client_api_keys(self, client_id: str) -> List[PydanticClientApiKey]:
        """List all API keys for a client."""
        with self.session_maker() as session:
            api_keys = (
                session.query(ClientApiKeyModel)
                .filter(ClientApiKeyModel.client_id == client_id, ClientApiKeyModel.is_deleted == False)
                .all()
            )
            return [key.to_pydantic() for key in api_keys]

    @enforce_types
    def revoke_client_api_key(self, api_key_id: str) -> PydanticClientApiKey:
        """Revoke an API key (set status to 'revoked')."""
        with self.session_maker() as session:
            api_key = ClientApiKeyModel.read(db_session=session, identifier=api_key_id)
            api_key.status = "revoked"
            api_key.update(session, actor=None)
            return api_key.to_pydantic()

    @enforce_types
    def delete_client_api_key(self, api_key_id: str) -> None:
        """Permanently delete an API key from the database."""
        with self.session_maker() as session:
            api_key = ClientApiKeyModel.read(db_session=session, identifier=api_key_id)
            session.delete(api_key)
            session.commit()

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
        """
        Soft delete a client and cascade soft delete to all associated records.

        Cleanup workflow:
        1. Soft delete all memory records using memory managers:
           - Episodic memories
           - Semantic memories
           - Procedural memories
           - Resource memories
           - Knowledge vault items
           - Messages

        2. Database (PostgreSQL):
           - Set client.is_deleted = True
           - Set agents.is_deleted = True for agents created by this client
           - Set tools.is_deleted = True for tools created by this client
           - Set blocks.is_deleted = True for blocks created by this client

        3. Redis Cache:
           - Update client hash with is_deleted=true
           - Update agent hashes with is_deleted=true
           - Update memory cache entries with is_deleted=true

        Args:
            client_id: ID of the client to soft delete
        """
        from mirix.database.redis_client import get_redis_client
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info("Soft deleting client %s and all associated records...", client_id)

        # Get client for actor parameter
        client = self.get_client_by_id(client_id)

        # Import memory managers
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

        episodic_count = episodic_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d episodic memories", episodic_count)

        semantic_count = semantic_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d semantic memories", semantic_count)

        procedural_count = procedural_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d procedural memories", procedural_count)

        resource_count = resource_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d resource memories", resource_count)

        knowledge_count = knowledge_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d knowledge vault items", knowledge_count)

        message_count = message_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d messages", message_count)

        # 2. Soft delete client metadata records
        with self.session_maker() as session:
            # Find client
            client_orm = ClientModel.read(db_session=session, identifier=client_id)
            if not client_orm:
                logger.warning("Client %s not found", client_id)
                return

            # Find all agents created by this client
            from mirix.orm.agent import Agent as AgentModel
            from mirix.orm.block import Block as BlockModel
            from mirix.orm.tool import Tool as ToolModel

            agents_created_by_client = (
                session.query(AgentModel)
                .filter(AgentModel._created_by_id == client_id, AgentModel.is_deleted == False)
                .all()
            )
            agent_ids = [agent.id for agent in agents_created_by_client]
            logger.debug("Found %d agents created by client %s", len(agent_ids), client_id)

            # Soft delete agents (set is_deleted = True directly, don't call agent.delete())
            for agent in agents_created_by_client:
                agent.is_deleted = True
                agent.set_updated_at()
            logger.debug("Soft deleted %d agents", len(agent_ids))

            # Soft delete tools created by this client
            tools = (
                session.query(ToolModel)
                .filter(ToolModel._created_by_id == client_id, ToolModel.is_deleted == False)
                .all()
            )
            for tool in tools:
                tool.is_deleted = True
                tool.set_updated_at()
            logger.debug("Soft deleted %d tools", len(tools))

            # Soft delete blocks created by this client
            blocks = (
                session.query(BlockModel)
                .filter(BlockModel._created_by_id == client_id, BlockModel.is_deleted == False)
                .all()
            )
            for block in blocks:
                block.is_deleted = True
                block.set_updated_at()
            logger.debug("Soft deleted %d blocks", len(blocks))

            # Soft delete client (set is_deleted = True directly, don't call client_orm.delete())
            client_orm.is_deleted = True
            client_orm.set_updated_at()
            session.commit()
            logger.info("Soft deleted client %s from database", client_id)

            # 3. Update Redis cache to reflect soft delete
            try:
                redis_client = get_redis_client()
                if redis_client:
                    # Update client hash with is_deleted=true
                    client_key = f"{redis_client.CLIENT_PREFIX}{client_id}"
                    try:
                        # Update the is_deleted field in Redis
                        redis_client.client.hset(client_key, "is_deleted", "true")
                        logger.debug("Updated client %s in Redis (is_deleted=true)", client_id)
                    except Exception as e:
                        logger.warning("Failed to update client in Redis, removing instead: %s", e)
                        redis_client.delete(client_key)

                    # Update agent hashes with is_deleted=true
                    for agent_id in agent_ids:
                        agent_key = f"{redis_client.AGENT_PREFIX}{agent_id}"
                        try:
                            redis_client.client.hset(agent_key, "is_deleted", "true")
                        except Exception:
                            # If update fails, remove from cache
                            redis_client.delete(agent_key)
                    logger.debug("Updated %d agents in Redis cache (is_deleted=true)", len(agent_ids))

                    logger.info(
                        "Client %s and all associated records soft deleted: "
                        "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages",
                        client_id,
                        episodic_count,
                        semantic_count,
                        procedural_count,
                        resource_count,
                        knowledge_count,
                        message_count,
                    )
            except Exception as e:
                logger.warning("Failed to update Redis cache for client %s: %s", client_id, e)

    def delete_memories_by_client_id(self, client_id: str):
        """
        Hard delete memories, messages, and blocks for a client using memory managers' bulk delete.

        This permanently removes data records while preserving the client, agents, and tools.
        Uses optimized bulk delete methods in each manager for efficient deletion.

        Cleanup workflow:
        1. Call each memory manager's delete_by_client_id() method
           - EpisodicMemoryManager.delete_by_client_id()
           - SemanticMemoryManager.delete_by_client_id()
           - ProceduralMemoryManager.delete_by_client_id()
           - ResourceMemoryManager.delete_by_client_id()
           - KnowledgeVaultManager.delete_by_client_id()
           - MessageManager.delete_by_client_id()
        2. Delete blocks (via _created_by_id)
        3. Each manager handles:
           - Bulk database deletion
           - Redis cache cleanup
           - Business logic
        4. PRESERVE: client record, agents, tools

        Args:
            client_id: ID of the client whose memories to delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Bulk deleting memories for client %s using memory managers (preserving client, agents, tools)...",
            client_id,
        )

        # Import managers
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

        # Get client as actor for manager methods
        client = self.get_client_by_id(client_id)
        if not client:
            logger.warning("Client %s not found", client_id)
            return

        # Use managers' bulk delete methods (much more efficient)
        try:
            # Bulk delete memories using manager methods (actor.id is used as client_id)
            episodic_count = episodic_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d episodic memories", episodic_count)

            semantic_count = semantic_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d semantic memories", semantic_count)

            procedural_count = procedural_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d procedural memories", procedural_count)

            resource_count = resource_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d resource memories", resource_count)

            knowledge_count = knowledge_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d knowledge vault items", knowledge_count)

            message_count = message_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d messages", message_count)

            # Delete blocks created by this client (using bulk operations)
            block_count = 0
            block_ids = []
            with self.session_maker() as session:
                from mirix.orm.block import Block as BlockModel

                # Get block IDs first (for Redis cleanup and agent cache invalidation)
                block_ids = [
                    row[0] for row in session.query(BlockModel.id).filter(BlockModel._created_by_id == client_id).all()
                ]

                block_count = len(block_ids)
                if block_count > 0:
                    # Invalidate agent caches for all blocks (before deletion)
                    from mirix.services.block_manager import BlockManager

                    block_manager = BlockManager()
                    for block_id in block_ids:
                        block_manager._invalidate_agent_caches_for_block(block_id)

                    # Bulk delete in single query
                    session.query(BlockModel).filter(BlockModel._created_by_id == client_id).delete(
                        synchronize_session=False
                    )

                    session.commit()

            # Remove blocks from Redis cache (outside session context)
            if block_ids:
                from mirix.database.redis_client import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    redis_keys = [f"{redis_client.BLOCK_PREFIX}{block_id}" for block_id in block_ids]
                    # Delete in batches
                    BATCH_SIZE = 1000
                    for i in range(0, len(redis_keys), BATCH_SIZE):
                        batch = redis_keys[i : i + BATCH_SIZE]
                        redis_client.client.delete(*batch)

            logger.debug("Bulk deleted %d blocks", block_count)

            # Clear message_ids from agents in PostgreSQL (they reference deleted messages)
            # IMPORTANT: Keep the first message (system message) as agents need it to function
            with self.session_maker() as session:
                from mirix.orm.agent import Agent as AgentModel

                # Update all agents for this client to keep only the system message
                agents = session.query(AgentModel).filter(AgentModel._created_by_id == client_id).all()

                agent_ids = [agent.id for agent in agents]
                for agent in agents:
                    # Keep only the first message_id (system message), clear the rest
                    if agent.message_ids and len(agent.message_ids) > 0:
                        agent.message_ids = [agent.message_ids[0]]  # Keep system message only

                session.commit()
                logger.debug(
                    "Cleared conversation message_ids from %d agents in PostgreSQL (kept system messages)",
                    len(agent_ids),
                )

            # Invalidate ALL agent caches for this client (force reload from PostgreSQL with cleared message_ids)
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()
            if redis_client and agent_ids:
                logger.debug("Invalidating %d agent caches for client %s", len(agent_ids), client_id)
                for agent_id in agent_ids:
                    agent_key = f"{redis_client.AGENT_PREFIX}{agent_id}"
                    redis_client.delete(agent_key)
                logger.debug("Invalidated %d agent caches", len(agent_ids))

            logger.info(
                "Bulk deleted all memories for client %s: "
                "%d episodic, %d semantic, %d procedural, %d resource, %d knowledge_vault, %d messages, %d blocks "
                "(client, agents, tools preserved)",
                client_id,
                episodic_count,
                semantic_count,
                procedural_count,
                resource_count,
                knowledge_count,
                message_count,
                block_count,
            )
        except Exception as e:
            logger.error("Failed to bulk delete memories for client %s: %s", client_id, e)
            raise

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
                    logger.debug("Redis cache HIT for client %s", client_id)
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

                    data = pydantic_client.model_dump(mode="json")
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_clients)
                    logger.debug("Populated Redis cache for client %s", client_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for client %s: %s", client_id, e)

            return pydantic_client

    @enforce_types
    def get_default_client(self) -> PydanticClient:
        """Fetch the default client, creating it if it doesn't exist."""
        try:
            return self.get_client_by_id(self.DEFAULT_CLIENT_ID)
        except NoResultFound:
            # Default client doesn't exist, create it
            # First ensure the default organization exists
            from mirix.services.organization_manager import OrganizationManager

            org_mgr = OrganizationManager()
            org_mgr.get_default_organization()  # Auto-creates if missing
            return self.create_default_client(org_id=OrganizationManager.DEFAULT_ORG_ID)

    @enforce_types
    def get_client_or_default(self, client_id: Optional[str] = None, organization_id: Optional[str] = None):
        """
        Fetch the client or create/return default client.

        Args:
            client_id: The client ID to retrieve (optional)
            organization_id: The organization ID for creating new clients (optional)

        Returns:
            PydanticClient: The client object
        """
        if not client_id:
            return self.get_default_client()

        try:
            return self.get_client_by_id(client_id=client_id)
        except NoResultFound:
            # If organization_id is provided, create a new client
            if organization_id:
                return self.create_client(
                    PydanticClient(
                        id=client_id,
                        organization_id=organization_id,
                        name=f"Local Client {client_id}",
                        status="active",
                        scope="local",
                    )
                )
            # Otherwise return default client
            return self.get_default_client()

    @enforce_types
    def list_clients(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[PydanticClient]:
        """List clients with pagination using cursor (id) and limit."""
        with self.session_maker() as session:
            results = ClientModel.list(db_session=session, cursor=cursor, limit=limit)
            return [client.to_pydantic() for client in results]
