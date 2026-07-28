import importlib
import warnings
from typing import List, Optional

from sqlalchemy import select

from mirix.constants import (
    ALL_TOOLS,
    BASE_TOOLS,
    CHAT_AGENT_TOOLS,
    CORE_MEMORY_TOOLS,
    EPISODIC_MEMORY_TOOLS,
    EXTRAS_TOOLS,
    KNOWLEDGE_VAULT_TOOLS,
    MCP_TOOLS,
    META_MEMORY_TOOLS,
    PROCEDURAL_MEMORY_TOOLS,
    RESOURCE_MEMORY_TOOLS,
    SEMANTIC_MEMORY_TOOLS,
    UNIVERSAL_MEMORY_TOOLS,
)
from mirix.functions.functions import derive_openai_json_schema, load_function_set
from mirix.log import get_logger

# TODO: Remove this once we translate all of these to the ORM
from mirix.orm.errors import NoResultFound
from mirix.orm.tool import Tool as ToolModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import ToolType
from mirix.schemas.tool import Tool as PydanticTool
from mirix.schemas.tool import ToolUpdate
from mirix.utils import enforce_types, printd

logger = get_logger(__name__)

# Max IN-list size per IPSR named-query call. IPSR's Hibernate-on-NQ stack has
# a server-side cap on IN expansions; 200 leaves headroom while keeping
# request counts low for the ~30-tool init path.
_BATCH_CHUNK_SIZE = 200


class ToolManager:
    """Manager class to handle business logic related to Tools."""

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context
        # Per-org memoization for the request-path base-tool existence check.
        # Populated by ``ensure_base_tools_exist`` and ``upsert_base_tools``
        # so subsequent calls within the same process lifetime skip all
        # tool-table roundtrips for that org.
        self._base_tools_verified_org_ids: set = set()

    # TODO: Refactor this across the codebase to use CreateTool instead of passing in a Tool object
    @enforce_types
    async def create_or_update_tool(self, pydantic_tool: PydanticTool, actor: PydanticClient) -> PydanticTool:
        """Create or update a tool (async)."""
        tool = await self.get_tool_by_name(tool_name=pydantic_tool.name, actor=actor)
        if tool:
            update_data = pydantic_tool.model_dump(exclude_unset=True, exclude_none=True)
            if update_data:
                return await self.update_tool_by_id(tool.id, ToolUpdate(**update_data), actor)
            printd(
                "`create_or_update_tool` was called with name=%s but found existing tool with nothing to update.",
                pydantic_tool.name,
            )
            return tool
        return await self.create_tool(pydantic_tool, actor=actor)

    @enforce_types
    async def create_tool(self, pydantic_tool: PydanticTool, actor: PydanticClient) -> PydanticTool:
        """Create a new tool (async)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            pydantic_tool.organization_id = actor.organization_id
            if pydantic_tool.description is None:
                pydantic_tool.description = pydantic_tool.json_schema.get("description", None)
            tool_data = pydantic_tool.model_dump()
            tool_data["_created_by_id"] = actor.id
            result = await provider.create("tools", tool_data)
            return PydanticTool(**result)

        async with self.session_maker() as session:
            pydantic_tool.organization_id = actor.organization_id
            if pydantic_tool.description is None:
                pydantic_tool.description = pydantic_tool.json_schema.get("description", None)
            tool_data = pydantic_tool.model_dump()
            tool = ToolModel(**tool_data)
            await tool.create(session, actor=actor)
        return tool.to_pydantic()

    @enforce_types
    async def get_tool_by_id(self, tool_id: str, actor: PydanticClient) -> PydanticTool:
        """Fetch a tool by its ID (async, with read-through cache on provider path)."""
        from mirix.database.cache_provider import get_cache_provider
        from mirix.database.relational_provider import get_relational_provider
        from mirix.log import get_logger

        logger = get_logger(__name__)
        cache_provider = get_cache_provider()

        if cache_provider:
            try:
                cache_key = f"{cache_provider.TOOL_PREFIX}{tool_id}"
                cached = await cache_provider.get_hash(cache_key)
                if cached:
                    import json as _json

                    if "json_schema" in cached and isinstance(cached["json_schema"], str):
                        cached["json_schema"] = _json.loads(cached["json_schema"])
                    if "tags" in cached and isinstance(cached["tags"], str):
                        cached["tags"] = _json.loads(cached["tags"])
                    return PydanticTool(**cached)
            except Exception as e:
                logger.debug("Cache read failed for tool %s: %s", tool_id, e)

        provider = get_relational_provider()
        if provider:
            rows = await provider.find_using_named_query(
                "tools",
                "tool_manager.get_tool_by_id",
                params={"id": tool_id, "organizationId": actor.organization_id},
                page_size=1,
            )
            if not rows:
                raise NoResultFound(f"Tool {tool_id} not found")
            tool = PydanticTool(**rows[0])

            if cache_provider:
                try:
                    import json as _json

                    from mirix.settings import settings

                    data = tool.model_dump(mode="json")
                    if "json_schema" in data and data["json_schema"]:
                        data["json_schema"] = _json.dumps(data["json_schema"])
                    if "tags" in data and data["tags"]:
                        data["tags"] = _json.dumps(data["tags"])
                    cache_key = f"{cache_provider.TOOL_PREFIX}{tool_id}"
                    await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_tools)
                except Exception as e:
                    logger.warning("Failed to populate cache for tool %s: %s", tool_id, e)

            return tool

        async with self.session_maker() as session:
            tool = await ToolModel.read(db_session=session, identifier=tool_id, actor=actor)
            return tool.to_pydantic()

    @enforce_types
    async def get_tool_by_name(self, tool_name: str, actor: PydanticClient) -> Optional[PydanticTool]:
        """Retrieve a tool by name (async)."""
        try:
            from mirix.database.relational_provider import get_relational_provider

            provider = get_relational_provider()
            if provider:
                results = await provider.find_using_named_query(
                    "tools",
                    "tool_manager.get_tool_by_name",
                    params={"name": tool_name, "organizationId": actor.organization_id},
                    page_size=1,
                )
                return PydanticTool(**results[0]) if results else None
            async with self.session_maker() as session:
                tool = await ToolModel.read(db_session=session, name=tool_name, actor=actor)
                return tool.to_pydantic()
        except NoResultFound:
            return None

    @enforce_types
    async def list_tools(
        self,
        actor: PydanticClient,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticTool]:
        """List all tools with optional pagination using cursor and limit."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            results = await provider.find_using_named_query(
                "tools",
                "tool_manager.list_tools",
                params={"organizationId": actor.organization_id, "cursor": cursor},
                page_size=limit or 50,
            )
            return [PydanticTool(**r) for r in results]

        async with self.session_maker() as session:
            tools = await ToolModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id,
            )
            return [tool.to_pydantic() for tool in tools]

    @enforce_types
    async def list_tools_by_names(self, tool_names: List[str], actor: PydanticClient) -> List[PydanticTool]:
        """Batch-fetch tools by name.

        Returns the tools whose ``name`` is in ``tool_names`` and whose
        ``organization_id`` matches ``actor.organization_id``. Replaces the
        N+1 per-name ``get_tool_by_name`` loop in agent_manager paths.

        On the IPSR provider path, the input is de-duped and chunked into
        ``_BATCH_CHUNK_SIZE`` slices (one named-query call per chunk). On the
        PG path, SQLAlchemy ``in_()`` handles expansion natively in a single
        query, so no chunking is needed.
        """
        unique_names = list({n for n in tool_names if n})
        if not unique_names:
            return []

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        results: List[PydanticTool] = []
        if provider:
            for start in range(0, len(unique_names), _BATCH_CHUNK_SIZE):
                chunk = unique_names[start : start + _BATCH_CHUNK_SIZE]
                # NQ uses ``= ANY(string_to_array(CAST(:names AS varchar), ','))``
                # so the bind value must be a single comma-separated string,
                # not a Python list. Chunking still applies to keep the bound
                # string and resulting array small.
                rows = await provider.find_using_named_query(
                    "tools",
                    "tool_manager.list_tools_by_names",
                    params={
                        "names": ",".join(chunk),
                        "organizationId": actor.organization_id,
                    },
                    page_size=len(chunk),
                )
                results.extend(PydanticTool(**r) for r in rows)
        else:
            async with self.session_maker() as session:
                stmt = (
                    select(ToolModel)
                    .where(ToolModel.name.in_(unique_names))
                    .where(ToolModel.organization_id == actor.organization_id)
                    .where(~ToolModel.is_deleted)
                )
                rows = await session.execute(stmt)
                results = [t.to_pydantic() for t in rows.scalars().all()]

        if len(results) < len(unique_names):
            returned = {t.name for t in results}
            missing = [n for n in unique_names if n not in returned]
            logger.warning(
                "list_tools_by_names: %d of %d names not found (missing=%s)",
                len(missing),
                len(unique_names),
                missing,
            )
        return results

    @enforce_types
    async def list_tools_by_ids(self, tool_ids: List[str], actor: PydanticClient) -> List[PydanticTool]:
        """Batch-fetch tools by id (used by the ECMS provider's M2M
        post-create resolution). Same shape as ``list_tools_by_names``."""
        unique_ids = list({i for i in tool_ids if i})
        if not unique_ids:
            return []

        from mirix.database.relational_provider import get_relational_provider
        from mirix.observability.timed import timedspan

        provider = get_relational_provider()
        results: List[PydanticTool] = []
        # Span the batch tool resolution: it is one of the save-path network
        # reads, and the chunk count makes a regressed N+1 visible in the trace.
        async with timedspan(
            "List Tools By Ids",
            metadata={
                "tool_id_count": len(unique_ids),
                "chunks": (len(unique_ids) + _BATCH_CHUNK_SIZE - 1) // _BATCH_CHUNK_SIZE,
                "provider": "ips_relational" if provider else "postgres",
            },
        ) as rec:
            if provider:
                for start in range(0, len(unique_ids), _BATCH_CHUNK_SIZE):
                    chunk = unique_ids[start : start + _BATCH_CHUNK_SIZE]
                    # See ``list_tools_by_names``: the NQ uses
                    # ``= ANY(string_to_array(CAST(:ids AS varchar), ','))``
                    # so the bind value must be a comma-separated string.
                    rows = await provider.find_using_named_query(
                        "tools",
                        "tool_manager.list_tools_by_ids",
                        params={
                            "ids": ",".join(chunk),
                            "organizationId": actor.organization_id,
                        },
                        page_size=len(chunk),
                    )
                    results.extend(PydanticTool(**r) for r in rows)
            else:
                async with self.session_maker() as session:
                    stmt = (
                        select(ToolModel)
                        .where(ToolModel.id.in_(unique_ids))
                        .where(ToolModel.organization_id == actor.organization_id)
                        .where(~ToolModel.is_deleted)
                    )
                    rows = await session.execute(stmt)
                    results = [t.to_pydantic() for t in rows.scalars().all()]
            rec["span_output"] = {"tools_loaded": len(results)}

        if len(results) < len(unique_ids):
            returned = {t.id for t in results}
            missing = [i for i in unique_ids if i not in returned]
            logger.warning(
                "list_tools_by_ids: %d of %d ids not found (missing=%s)",
                len(missing),
                len(unique_ids),
                missing,
            )
        return results

    @enforce_types
    async def update_tool_by_id(self, tool_id: str, tool_update: ToolUpdate, actor: PydanticClient) -> PydanticTool:
        """Update a tool by its ID (async)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            update_data = tool_update.model_dump(exclude_none=True)
            if "source_code" in update_data and "json_schema" not in update_data:
                new_schema = derive_openai_json_schema(source_code=update_data.get("source_code"))
                update_data["json_schema"] = new_schema
            result = await provider.update("tools", tool_id, update_data)
            return PydanticTool(**result)

        async with self.session_maker() as session:
            tool = await ToolModel.read(db_session=session, identifier=tool_id, actor=actor)
            update_data = tool_update.model_dump(exclude_none=True)
            for key, value in update_data.items():
                setattr(tool, key, value)
            if "source_code" in update_data.keys() and "json_schema" not in update_data.keys():
                pydantic_tool = tool.to_pydantic()
                new_schema = derive_openai_json_schema(source_code=pydantic_tool.source_code)
                tool.json_schema = new_schema
            updated = await tool.update(db_session=session, actor=actor)
            return updated.to_pydantic()

    @enforce_types
    async def delete_tool_by_id(self, tool_id: str, actor: PydanticClient) -> None:
        """Delete a tool by its ID."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            await provider.hard_delete("tools", tool_id)
            # Invalidate tool cache to prevent stale reads after deletion.
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            if cache_provider:
                tool_cache_key = f"{cache_provider.TOOL_PREFIX}{tool_id}"
                await cache_provider.delete(tool_cache_key)
            return

        async with self.session_maker() as session:
            try:
                tool = await ToolModel.read(db_session=session, identifier=tool_id, actor=actor)
                await tool.hard_delete(db_session=session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Tool with id {tool_id} not found.")

    @staticmethod
    def _load_base_tool_names() -> List[str]:
        """Return the expected base/memory/extras tool names in import order.

        Mirrors ``upsert_base_tools`` discovery: walks the public functions in
        ``mirix.functions.function_sets.{base,memory_tools,extras}`` and
        filters them against ``ALL_TOOLS``.
        """
        functions_to_schema: dict = {}
        for module_name in ("base", "memory_tools", "extras"):
            full_module_name = f"mirix.functions.function_sets.{module_name}"
            module = importlib.import_module(full_module_name)
            try:
                functions_to_schema.update(load_function_set(module))
            except ValueError as e:
                warnings.warn(f"Error loading function set '{module_name}': {e}")
        return [name for name in functions_to_schema.keys() if name in ALL_TOOLS]

    @staticmethod
    def _classify_base_tool(name: str) -> tuple:
        """Return ``(tool_type, tags)`` for one of the known base tool names."""
        if name in BASE_TOOLS:
            tool_type = ToolType.MIRIX_CORE
            return tool_type, [tool_type.value]
        if name in (
            CORE_MEMORY_TOOLS
            + EPISODIC_MEMORY_TOOLS
            + PROCEDURAL_MEMORY_TOOLS
            + RESOURCE_MEMORY_TOOLS
            + KNOWLEDGE_VAULT_TOOLS
            + META_MEMORY_TOOLS
            + SEMANTIC_MEMORY_TOOLS
            + UNIVERSAL_MEMORY_TOOLS
            + CHAT_AGENT_TOOLS
        ):
            tool_type = ToolType.MIRIX_MEMORY_CORE
            return tool_type, [tool_type.value]
        if name in EXTRAS_TOOLS:
            tool_type = ToolType.MIRIX_EXTRA
            return tool_type, [tool_type.value]
        if name in MCP_TOOLS:
            tool_type = ToolType.MIRIX_EXTRA
            return tool_type, [tool_type.value, "mcp_wrapper"]
        raise ValueError(f"Tool name {name} is not in the list of tool names")

    @enforce_types
    async def upsert_base_tools(self, actor: PydanticClient) -> List[PydanticTool]:
        """Add or update default tools (async).

        This is the full-write path used by ``server.ensure_defaults()`` at
        process startup and by admin maintenance flows: every known tool has
        its source/schema re-derived and written back. Memoized per org per
        process so subsequent calls within the same worker are no-ops.

        Request-path callers (e.g. ``agent_manager.create_meta_agent``) should
        use ``ensure_base_tools_exist`` instead, which only writes when a tool
        is missing.
        """
        org_id = actor.organization_id
        if org_id and org_id in self._base_tools_verified_org_ids:
            return []

        expected = self._load_base_tool_names()
        tools: List[PydanticTool] = []
        for name in expected:
            tool_type, tags = self._classify_base_tool(name)
            tool = await self.create_or_update_tool(
                PydanticTool(
                    name=name,
                    tags=tags,
                    source_type="python",
                    tool_type=tool_type,
                ),
                actor=actor,
            )
            tools.append(tool)

        if org_id:
            self._base_tools_verified_org_ids.add(org_id)
        return tools

    @enforce_types
    async def ensure_base_tools_exist(self, actor: PydanticClient) -> List[PydanticTool]:
        """Lightweight request-path base-tool verification.

        Used by ``agent_manager.create_meta_agent`` on every cold-start init
        request. Performs at most one batched tools-read followed by per-tool
        creates only for tools that do not yet exist (typically zero in
        production where ``ensure_defaults()`` at startup has already seeded
        the table). Subsequent calls for the same org in the same process
        lifetime are zero-roundtrip via per-org memoization.

        Returns the list of tools created (possibly empty). Does not update
        existing tools — for full update semantics, use ``upsert_base_tools``.
        """
        org_id = actor.organization_id
        if org_id and org_id in self._base_tools_verified_org_ids:
            return []

        expected = self._load_base_tool_names()
        existing = await self.list_tools_by_names(expected, actor)
        existing_names = {t.name for t in existing}
        missing = [n for n in expected if n not in existing_names]

        created: List[PydanticTool] = []
        for name in missing:
            tool_type, tags = self._classify_base_tool(name)
            tool = await self.create_tool(
                PydanticTool(
                    name=name,
                    tags=tags,
                    source_type="python",
                    tool_type=tool_type,
                ),
                actor=actor,
            )
            created.append(tool)

        if org_id:
            self._base_tools_verified_org_ids.add(org_id)
        return created
