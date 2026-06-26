from typing import List, Optional

from sqlalchemy import delete, select

from mirix.orm.client import Client as ClientModel
from mirix.orm.client_api_key import ClientApiKey as ClientApiKeyModel
from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.client import ClientUpdate
from mirix.schemas.client_api_key import ClientApiKey as PydanticClientApiKey
from mirix.security.api_keys import hash_api_key
from mirix.services.organization_manager import OrganizationManager
from mirix.utils import enforce_types


class ClientManager:
    """Manager class to handle business logic related to Clients.

    Identity model: clients have an id-only primary key (no composite
    ``(organization_id, id)``). Named clients get random UUIDs at creation time,
    so collisions are impossible by construction. The "default client" is the
    only client whose id would otherwise be a constant — to keep each org's
    default-client row isolated we derive it per-org via :func:`default_client_id`
    rather than using a single global id.
    """

    DEFAULT_CLIENT_NAME = "default_client"
    # Global default-client id, used only where there is no organization in
    # scope to derive a per-org id from:
    #   * agent.py — Agent fallback when constructed without an actor; the
    #     agent_state's organization_id is optional and absent in standalone use.
    #   * local_client.py / rest_api.py — single-org standalone-MIRIX paths.
    # When an org IS available, callers must use ``default_client_id(org_id)``
    # so each org gets its own default-client row (a single shared id would
    # collide on the id-only primary key the first time a second org bootstraps).
    DEFAULT_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"

    @classmethod
    def default_client_id(cls, org_id: str) -> str:
        """Per-org id for "the default client" — distinct per org.

        Mirrors the ``user-default-{org_id}`` pattern used elsewhere. Required
        because clients carry per-org-meaningful state (write_scope,
        read_scopes) and so must be isolated per org, but the id-only primary
        key cannot enforce ``(organization_id, id)`` composite identity. Deriving
        the id from the org makes each org's default client a distinct row
        without composite PKs.
        """
        return f"client-default-{org_id}"

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create_default_client(self, org_id: str = OrganizationManager.DEFAULT_ORG_ID) -> PydanticClient:
        """Create the default client for *org_id*, or return the existing row.

        Per-org default client (id derived from ``org_id``); see
        ``default_client_id``. Pre-check is the fast path; catch-and-swallow
        handles concurrent-startup races.
        """
        from mirix.database.provider_write_retry import is_conflict
        from mirix.database.relational_provider import get_relational_provider

        client_id = self.default_client_id(org_id)
        provider = get_relational_provider()
        if provider:
            # Verify the organization exists in Relational DB provider
            org = await provider.read("organizations", org_id)
            if org is None:
                raise ValueError(f"No organization with {org_id} exists in the organization table.")

            # Return existing default client if already present
            existing = await provider.read("clients", client_id)
            if existing:
                return PydanticClient(**existing)

            try:
                result = await provider.create(
                    "clients",
                    {
                        "id": client_id,
                        "name": self.DEFAULT_CLIENT_NAME,
                        "status": "active",
                        "write_scope": None,
                        "read_scopes": [],
                        "organization_id": org_id,
                    },
                )
                return PydanticClient(**result)
            except Exception as exc:
                if not is_conflict(exc):
                    raise
                existing = await provider.read("clients", client_id)
                if existing is None:
                    raise
                return PydanticClient(**existing)

        async with self.session_maker() as session:
            try:
                await OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(f"No organization with {org_id} exists in the organization table.") from None

            try:
                client = await ClientModel.read(db_session=session, identifier=client_id)
            except NoResultFound:
                client = ClientModel(
                    id=client_id,
                    name=self.DEFAULT_CLIENT_NAME,
                    status="active",
                    write_scope=None,
                    read_scopes=[],
                    organization_id=org_id,
                )
                try:
                    await client.create(session)
                except Exception as exc:
                    if not is_conflict(exc):
                        raise
                    client = await ClientModel.read(db_session=session, identifier=client_id)

            return client.to_pydantic()

    @enforce_types
    async def create_client(self, pydantic_client: PydanticClient) -> PydanticClient:
        """Create a new client, or return the existing row if id is already taken.

        ``clients.id`` is globally unique. Named clients use random UUIDs at
        creation so collisions are not expected — the catch-and-swallow exists
        as defense-in-depth and to handle concurrent-startup races on the
        default client.
        """
        from mirix.database.provider_write_retry import is_conflict
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            client_data = pydantic_client.model_dump()
            try:
                result = await provider.create("clients", client_data)
                return PydanticClient(**result)
            except Exception as exc:
                if not is_conflict(exc):
                    raise
                existing = await provider.read("clients", pydantic_client.id)
                if existing is None:
                    raise
                return PydanticClient(**existing)
        async with self.session_maker() as session:
            new_client = ClientModel(**pydantic_client.model_dump())
            try:
                await new_client.create_with_redis(session, actor=None)  # Auto-caches to Redis
                return new_client.to_pydantic()
            except Exception as exc:
                if not is_conflict(exc):
                    raise
            existing = await ClientModel.read(db_session=session, identifier=pydantic_client.id)
            return existing.to_pydantic()

    @enforce_types
    async def update_client(self, client_update: ClientUpdate) -> PydanticClient:
        """Update client details (with cache invalidation)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            update_data = client_update.model_dump(exclude_unset=True, exclude_none=True)
            result = await provider.update("clients", client_update.id, update_data)
            return PydanticClient(**result)
        async with self.session_maker() as session:
            existing_client = await ClientModel.read(db_session=session, identifier=client_update.id)
            update_data = client_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_client, key, value)
            await existing_client.update_with_redis(session, actor=None)
            return existing_client.to_pydantic()

    @enforce_types
    async def create_client_api_key(
        self,
        client_id: str,
        api_key: str,
        name: Optional[str] = None,
        permission: str = "all",
        user_id: Optional[str] = None,
    ) -> PydanticClientApiKey:
        """Create a new API key for a client."""
        from mirix.database.relational_provider import get_relational_provider

        hashed = hash_api_key(api_key)

        provider = get_relational_provider()
        if provider:
            client_data = await provider.read("clients", client_id)
            org_id = client_data.get("organization_id", "") if client_data else ""
            data = {
                "client_id": client_id,
                "organization_id": org_id,
                "api_key_hash": hashed,
                "name": name,
                "status": "active",
                "permission": permission,
                "user_id": user_id,
            }
            result = await provider.create("client_api_keys", data)
            return PydanticClientApiKey(**result)

        async with self.session_maker() as session:
            existing_client = await ClientModel.read(db_session=session, identifier=client_id)
            api_key_pydantic = PydanticClientApiKey(
                client_id=client_id,
                organization_id=existing_client.organization_id,
                api_key_hash=hashed,
                name=name,
                status="active",
                permission=permission,
                user_id=user_id,
            )
            new_api_key = ClientApiKeyModel(**api_key_pydantic.model_dump())
            await new_api_key.create(session)
            return new_api_key.to_pydantic()

    @enforce_types
    async def set_client_api_key(
        self, client_id: str, api_key: str, name: Optional[str] = None
    ) -> PydanticClientApiKey:
        """
        Create a new API key for a client (deprecated name, use create_client_api_key).

        This method now creates a new API key entry in the client_api_keys table.
        For backward compatibility with existing scripts.
        """
        return await self.create_client_api_key(client_id, api_key, name)

    @enforce_types
    async def get_client_by_api_key(self, api_key: str) -> Optional[PydanticClient]:
        """Lookup a client via API key (hash match) from the client_api_keys table."""
        from mirix.database.relational_provider import get_relational_provider

        hashed = hash_api_key(api_key)

        provider = get_relational_provider()
        if provider:
            keys = await provider.find_using_named_query(
                "client_api_keys",
                "client_manager.get_client_by_api_key",
                params={"apiKeyHash": hashed},
                page_size=1,
            )
            if not keys:
                return None
            client_id = keys[0].get("client_id", "")
            client_rows = await provider.find_using_named_query(
                "clients",
                "client_manager.get_client_by_id",
                params={"id": client_id},
                page_size=1,
            )
            if not client_rows:
                return None
            client_data = client_rows[0]
            # client_manager.get_client_by_api_key YAML already enforces
            # status = 'active' and is_deleted = false on the api_key row,
            # so no additional Python status check is needed here.
            return PydanticClient(**client_data)

        async with self.session_maker() as session:
            stmt = (
                select(ClientApiKeyModel)
                .where(
                    ClientApiKeyModel.api_key_hash == hashed,
                    ClientApiKeyModel.status == "active",
                    ClientApiKeyModel.is_deleted == False,
                )
                .limit(1)
            )
            result = await session.execute(stmt)
            api_key_record = result.scalar_one_or_none()
            if not api_key_record:
                return None

            client = await ClientModel.read(db_session=session, identifier=api_key_record.client_id)
            if client.is_deleted or client.status != "active":
                return None

            return client.to_pydantic()

    @enforce_types
    async def list_client_api_keys(self, client_id: str) -> List[PydanticClientApiKey]:
        """List all API keys for a client."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            records = await provider.find_using_named_query(
                "client_api_keys",
                "client_manager.list_client_api_keys",
                params={"clientId": client_id},
                page_size=100,
            )
            return [PydanticClientApiKey(**r) for r in records]

        async with self.session_maker() as session:
            stmt = select(ClientApiKeyModel).where(
                ClientApiKeyModel.client_id == client_id,
                ClientApiKeyModel.is_deleted == False,
            )
            result = await session.execute(stmt)
            api_keys = result.scalars().all()
            return [key.to_pydantic() for key in api_keys]

    @enforce_types
    async def revoke_client_api_key(self, api_key_id: str) -> PydanticClientApiKey:
        """Revoke an API key (set status to 'revoked')."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # Read the row first so we can synthesize the response without relying on
            # Relational DB provider read-after-write consistency for the mutation NQ.
            existing = await provider.read("client_api_keys", api_key_id)
            if existing is None:
                from mirix.orm.errors import NoResultFound as _NRF

                raise _NRF(f"Client API key {api_key_id} not found")
            await provider.mutate_using_named_query(
                "client_api_keys",
                "client_manager.revoke_client_api_key",
                params={"id": api_key_id},
            )
            existing["status"] = "revoked"
            # Invalidate the cached client so callers get fresh api_key status data.
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            client_id = existing.get("client_id")
            if cache_provider and client_id:
                await cache_provider.delete(f"{cache_provider.CLIENT_PREFIX}{client_id}")
            return PydanticClientApiKey(**existing)

        async with self.session_maker() as session:
            api_key = await ClientApiKeyModel.read(db_session=session, identifier=api_key_id)
            api_key.status = "revoked"
            await api_key.update(session, actor=None)
            return api_key.to_pydantic()

    @enforce_types
    async def delete_client_api_key(self, api_key_id: str) -> None:
        """Permanently delete an API key from the database."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # Read before delete so we can extract client_id for cache invalidation.
            existing = await provider.read("client_api_keys", api_key_id)
            await provider.hard_delete("client_api_keys", api_key_id)
            # Invalidate cached client to reflect removal of the api key.
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            client_id = (existing or {}).get("client_id")
            if cache_provider and client_id:
                await cache_provider.delete(f"{cache_provider.CLIENT_PREFIX}{client_id}")
            return

        async with self.session_maker() as session:
            api_key = await ClientApiKeyModel.read(db_session=session, identifier=api_key_id)
            session.delete(api_key)
            await session.commit()

    @enforce_types
    async def update_client_status(self, client_id: str, status: str) -> PydanticClient:
        """Update the status of a client (with cache invalidation)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.update("clients", client_id, {"status": status})
            return PydanticClient(**result)
        async with self.session_maker() as session:
            existing_client = await ClientModel.read(db_session=session, identifier=client_id)
            existing_client.status = status
            await existing_client.update_with_redis(session, actor=None)
            return existing_client.to_pydantic()

    @enforce_types
    async def soft_delete_client(self, client_id: str) -> PydanticClient:
        """
        Soft delete a client (marks as deleted, keeps in database).

        Args:
            client_id: The client ID to soft delete

        Returns:
            The soft-deleted client

        Raises:
            NoResultFound: If client not found
        """
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            await provider.mutate_using_named_query(
                "clients",
                "client_manager.update_client_by_id",
                params={"id": client_id},
            )
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.CLIENT_PREFIX}{client_id}"
                    await cache_provider.delete(cache_key)
            except Exception:
                pass
            return PydanticClient(id=client_id, name="", status="deleted", read_scopes=[])
        async with self.session_maker() as session:
            client = await ClientModel.read(db_session=session, identifier=client_id)
            await client.delete(session, actor=None)

            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.log import get_logger

                logger = get_logger(__name__)
                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.CLIENT_PREFIX}{client_id}"
                    await cache_provider.delete(cache_key)
                    logger.debug("Removed soft-deleted client %s from cache", client_id)
            except Exception as e:
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    "Failed to update cache for soft-deleted client %s: %s",
                    client_id,
                    e,
                )

            return client.to_pydantic()

    @enforce_types
    async def delete_client_by_id(self, client_id: str):
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

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            # Soft delete memory records owned by the client via named queries.
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
            for table in memory_tables:
                rows = await provider.find_using_named_query(
                    table,
                    f"client_manager.list_ids_{table}_by_client",
                    params={"createdById": client_id},
                    page_size=5000,
                )
                ids = [r.get("id") for r in rows if r.get("id")]
                if ids:
                    await provider.bulk_delete(table, ids, soft=True)

            # Soft delete messages owned by this client (engine table — no domain events needed).
            await provider.mutate_using_named_query(
                "messages",
                "message_manager.update_by_client_id",
                params={"clientId": client_id},
            )

            # Soft delete agents/tools created by this client (A-9/T-8 collect + A-10 bulk mutate).
            for table, list_nq, mutate_nq in (
                ("agents", "client_manager.list_agent_by_client_id", "client_manager.update_agent_by_client_id"),
                ("tools", "client_manager.list_tools_by_client_id", "client_manager.update_tool_by_client_id"),
            ):
                records = await provider.find_using_named_query(
                    table,
                    list_nq,
                    params={"createdById": client_id},
                    page_size=5000,
                )
                ids = [r.get("id") for r in records if r.get("id")]
                if ids:
                    await provider.mutate_using_named_query(
                        table,
                        mutate_nq,
                        params={"createdById": client_id},
                    )

            # Soft delete the client record.
            await provider.mutate_using_named_query(
                "clients",
                "client_manager.update_client_by_id",
                params={"id": client_id},
            )

            # Cache invalidation via cache provider abstraction.
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    await cache_provider.delete(f"{cache_provider.CLIENT_PREFIX}{client_id}")
            except Exception as e:
                logger.warning(
                    "Failed to invalidate cache for soft-deleted client %s: %s",
                    client_id,
                    e,
                )
            logger.info("Soft deleted client %s via provider path", client_id)
            return

        # Get client for actor parameter
        client = await self.get_client_by_id(client_id)

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

        episodic_count = await episodic_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d episodic memories", episodic_count)

        semantic_count = await semantic_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d semantic memories", semantic_count)

        procedural_count = await procedural_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d procedural memories", procedural_count)

        resource_count = await resource_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d resource memories", resource_count)

        knowledge_count = await knowledge_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d knowledge vault items", knowledge_count)

        message_count = await message_manager.soft_delete_by_client_id(actor=client)
        logger.debug("Soft deleted %d messages", message_count)

        # 2. Soft delete client metadata records
        from mirix.orm.agent import Agent as AgentModel
        from mirix.orm.block import Block as BlockModel
        from mirix.orm.tool import Tool as ToolModel

        async with self.session_maker() as session:
            client_orm = await ClientModel.read(db_session=session, identifier=client_id)
            if not client_orm:
                logger.warning("Client %s not found", client_id)
                return

            stmt_agents = select(AgentModel).where(
                AgentModel._created_by_id == client_id,
                AgentModel.is_deleted == False,
            )
            result_agents = await session.execute(stmt_agents)
            agents_created_by_client = result_agents.scalars().all()
            agent_ids = [agent.id for agent in agents_created_by_client]
            logger.debug("Found %d agents created by client %s", len(agent_ids), client_id)

            for agent in agents_created_by_client:
                agent.is_deleted = True
                agent.set_updated_at()
            logger.debug("Soft deleted %d agents", len(agent_ids))

            stmt_tools = select(ToolModel).where(
                ToolModel._created_by_id == client_id,
                ToolModel.is_deleted == False,
            )
            result_tools = await session.execute(stmt_tools)
            tools = result_tools.scalars().all()
            for tool in tools:
                tool.is_deleted = True
                tool.set_updated_at()
            logger.debug("Soft deleted %d tools", len(tools))

            stmt_blocks = select(BlockModel).where(
                BlockModel._created_by_id == client_id,
                BlockModel.is_deleted == False,
            )
            result_blocks = await session.execute(stmt_blocks)
            blocks = result_blocks.scalars().all()
            for block in blocks:
                block.is_deleted = True
                block.set_updated_at()
            logger.debug("Soft deleted %d blocks", len(blocks))

            client_orm.is_deleted = True
            client_orm.set_updated_at()
            await session.commit()
            logger.info("Soft deleted client %s from database", client_id)

        # 3. Update Redis cache to reflect soft delete
        try:
            redis_client = get_redis_client()
            if redis_client:
                client_key = f"{redis_client.CLIENT_PREFIX}{client_id}"
                try:
                    await redis_client.client.hset(client_key, "is_deleted", "true")
                    logger.debug("Updated client %s in cache (is_deleted=true)", client_id)
                except Exception as e:
                    logger.warning("Failed to update client in Redis, removing instead: %s", e)
                    await redis_client.delete(client_key)

                for agent_id in agent_ids:
                    agent_key = f"{redis_client.AGENT_PREFIX}{agent_id}"
                    try:
                        await redis_client.client.hset(agent_key, "is_deleted", "true")
                    except Exception:
                        await redis_client.delete(agent_key)
                logger.debug("Updated %d agents in Redis cache (is_deleted=true)", len(agent_ids))

                logger.info(
                    "Client %s and all associated records soft deleted: "
                    "%d episodic, %d semantic, %d procedural, %d resource, "
                    "%d knowledge_vault, %d messages",
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

    async def delete_memories_by_client_id(self, client_id: str):
        """
        Hard delete memories, messages, and blocks for a client.

        When Relational DB provider is registered, uses bulk_delete API.
        Otherwise uses memory managers' bulk delete via SQLAlchemy.

        Preserves the client, agents, and tools.

        Args:
            client_id: ID of the client whose memories to delete
        """
        from mirix.log import get_logger

        logger = get_logger(__name__)

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            logger.info("Bulk deleting memories via Relational DB provider for client %s", client_id)
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
            total = 0
            for table in memory_tables:
                rows = await provider.find_using_named_query(
                    table,
                    f"client_manager.list_ids_{table}_by_client",
                    params={"createdById": client_id},
                    page_size=5000,
                )
                if rows:
                    ids = [r.get("id", "") for r in rows if r.get("id")]
                    if ids:
                        result = await provider.bulk_delete(table, ids, soft=False)
                        deleted = int(result.get("success", 0) or 0)
                        total += deleted
                        logger.debug("Bulk deleted %d %s records", deleted, table)

            # Hard delete messages owned by this client (engine table — no domain events needed).
            await provider.mutate_using_named_query(
                "messages",
                "message_manager.update_by_client_id",
                params={"clientId": client_id},
            )

            logger.info("Provider bulk delete complete for client %s: %d total records", client_id, total)
            return

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
        client = await self.get_client_by_id(client_id)
        if not client:
            logger.warning("Client %s not found", client_id)
            return

        # Use managers' bulk delete methods (much more efficient)
        try:
            # Bulk delete memories using manager methods (actor.id is used as client_id)
            episodic_count = await episodic_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d episodic memories", episodic_count)

            semantic_count = await semantic_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d semantic memories", semantic_count)

            procedural_count = await procedural_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d procedural memories", procedural_count)

            resource_count = await resource_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d resource memories", resource_count)

            knowledge_count = await knowledge_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d knowledge vault items", knowledge_count)

            message_count = await message_manager.delete_by_client_id(actor=client)
            logger.debug("Bulk deleted %d messages", message_count)

            # Delete blocks created by this client (using bulk operations)
            block_count = 0
            block_ids: List[str] = []
            async with self.session_maker() as session:
                from mirix.orm.block import Block as BlockModel

                stmt_ids = select(BlockModel.id).where(BlockModel._created_by_id == client_id)
                result_ids = await session.execute(stmt_ids)
                block_ids = [row[0] for row in result_ids.all()]

                block_count = len(block_ids)
                if block_count > 0:
                    from mirix.services.block_manager import BlockManager

                    block_manager = BlockManager()
                    for block_id in block_ids:
                        await block_manager._invalidate_block_cache(block_id)

                    stmt_del = delete(BlockModel).where(BlockModel._created_by_id == client_id)
                    await session.execute(stmt_del)
                    await session.commit()

            if block_ids:
                from mirix.database.redis_client import get_redis_client

                redis_client = get_redis_client()
                if redis_client:
                    redis_keys = [f"{redis_client.BLOCK_PREFIX}{block_id}" for block_id in block_ids]
                    BATCH_SIZE = 1000
                    for i in range(0, len(redis_keys), BATCH_SIZE):
                        batch = redis_keys[i : i + BATCH_SIZE]
                        await redis_client.client.delete(*batch)

            logger.debug("Bulk deleted %d blocks", block_count)

            # Collect agent IDs for cache invalidation (messages already deleted above)
            agent_ids: List[str] = []
            async with self.session_maker() as session:
                from mirix.orm.agent import Agent as AgentModel

                stmt_agents = select(AgentModel).where(AgentModel._created_by_id == client_id)
                result_agents = await session.execute(stmt_agents)
                agents = result_agents.scalars().all()
                agent_ids = [agent.id for agent in agents]

            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            if cache_provider and agent_ids:
                logger.debug(
                    "Invalidating %d agent caches for client %s",
                    len(agent_ids),
                    client_id,
                )
                for agent_id in agent_ids:
                    agent_key = f"{cache_provider.AGENT_PREFIX}{agent_id}"
                    await cache_provider.delete(agent_key)
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
    async def get_client_by_id(self, client_id: str) -> PydanticClient:
        """Fetch a client by ID (with cache - Redis or Cache provider)."""
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.CLIENT_PREFIX}{client_id}"
                cached_data = await cache_provider.get_hash(cache_key)
                if cached_data:
                    logger.debug("Cache HIT for client %s", client_id)
                    return PydanticClient(**cached_data)
        except Exception as e:
            logger.debug("Cache read failed for client %s: %s", client_id, e)

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            from mirix.orm.errors import NoResultFound as _NRF

            rows = await provider.find_using_named_query(
                "clients",
                "client_manager.get_client_by_id",
                params={"id": client_id},
                page_size=1,
            )
            if not rows:
                raise _NRF(f"Client {client_id} not found")
            pydantic_client = PydanticClient(**rows[0])
            try:
                if cache_provider:
                    from mirix.settings import settings

                    cache_key = f"{cache_provider.CLIENT_PREFIX}{client_id}"
                    data = pydantic_client.model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_clients)
            except Exception as e:
                logger.warning("Failed to populate cache for client %s: %s", client_id, e)
            return pydantic_client

        async with self.session_maker() as session:
            client = await ClientModel.read(db_session=session, identifier=client_id)
            pydantic_client = client.to_pydantic()

            try:
                if cache_provider:
                    from mirix.settings import settings

                    cache_key = f"{cache_provider.CLIENT_PREFIX}{client_id}"
                    data = pydantic_client.model_dump(mode="json")
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_clients)
                    logger.debug("Populated cache for client %s", client_id)
            except Exception as e:
                logger.warning("Failed to populate cache for client %s: %s", client_id, e)

            return pydantic_client

    @enforce_types
    async def get_default_client(self) -> PydanticClient:
        """Fetch the default client, creating it if it doesn't exist."""
        try:
            return await self.get_client_by_id(self.DEFAULT_CLIENT_ID)
        except NoResultFound:
            from mirix.services.organization_manager import OrganizationManager

            org_mgr = OrganizationManager()
            await org_mgr.get_default_organization()
            return await self.create_default_client(org_id=OrganizationManager.DEFAULT_ORG_ID)

    @enforce_types
    async def get_client_or_default(
        self,
        client_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ):
        """
        Fetch the client or create/return default client.

        Args:
            client_id: The client ID to retrieve (optional)
            organization_id: The organization ID for creating new clients (optional)

        Returns:
            PydanticClient: The client object
        """
        if not client_id:
            return await self.get_default_client()

        try:
            return await self.get_client_by_id(client_id=client_id)
        except NoResultFound:
            if organization_id:
                return await self.create_client(
                    PydanticClient(
                        id=client_id,
                        organization_id=organization_id,
                        name=f"Local Client {client_id}",
                        status="active",
                        write_scope="local",
                        read_scopes=["local"],
                    )
                )
            return await self.get_default_client()

    @enforce_types
    async def list_clients(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticClient]:
        """List clients with pagination using cursor (id) and limit."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            results = await provider.find_using_named_query(
                "clients",
                "client_manager.list_clients",
                params={"cursor": cursor},
                page_size=limit or 50,
            )
            return [PydanticClient(**r) for r in results]
        async with self.session_maker() as session:
            results = await ClientModel.list(db_session=session, cursor=cursor, limit=limit)
            return [client.to_pydantic() for client in results]
