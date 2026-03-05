import importlib
import warnings
from typing import List, Optional

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

# TODO: Remove this once we translate all of these to the ORM
from mirix.orm.errors import NoResultFound
from mirix.orm.tool import Tool as ToolModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.enums import ToolType
from mirix.schemas.tool import Tool as PydanticTool
from mirix.schemas.tool import ToolUpdate
from mirix.utils import enforce_types, printd


class ToolManager:
    """Manager class to handle business logic related to Tools."""

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    # TODO: Refactor this across the codebase to use CreateTool instead of passing in a Tool object
    @enforce_types
    async def create_or_update_tool(
        self, pydantic_tool: PydanticTool, actor: PydanticClient
    ) -> PydanticTool:
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
        """Fetch a tool by its ID (async)."""
        async with self.session_maker() as session:
            tool = await ToolModel.read(db_session=session, identifier=tool_id, actor=actor)
            return tool.to_pydantic()

    @enforce_types
    async def get_tool_by_name(
        self, tool_name: str, actor: PydanticClient
    ) -> Optional[PydanticTool]:
        """Retrieve a tool by name (async)."""
        try:
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
        async with self.session_maker() as session:
            tools = await ToolModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id,
            )
            return [tool.to_pydantic() for tool in tools]

    @enforce_types
    async def update_tool_by_id(
        self, tool_id: str, tool_update: ToolUpdate, actor: PydanticClient
    ) -> PydanticTool:
        """Update a tool by its ID (async)."""
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
        async with self.session_maker() as session:
            try:
                tool = await ToolModel.read(
                    db_session=session, identifier=tool_id, actor=actor
                )
                await tool.hard_delete(db_session=session, actor=actor)
            except NoResultFound:
                raise ValueError(f"Tool with id {tool_id} not found.")

    @enforce_types
    async def upsert_base_tools(self, actor: PydanticClient) -> List[PydanticTool]:
        """Add default tools in base.py (async)."""
        functions_to_schema = {}
        module_names = ["base", "memory_tools", "extras"]

        for module_name in module_names:
            full_module_name = f"mirix.functions.function_sets.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
            except Exception as e:
                raise e

            try:
                functions_to_schema.update(load_function_set(module))
            except ValueError as e:
                err = f"Error loading function set '{module_name}': {e}"
                warnings.warn(err)

        tools = []
        for name, schema in functions_to_schema.items():
            if name in ALL_TOOLS:
                if name in BASE_TOOLS:
                    tool_type = ToolType.MIRIX_CORE
                    tags = [tool_type.value]
                elif (
                    name
                    in CORE_MEMORY_TOOLS
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
                    tags = [tool_type.value]
                elif name in EXTRAS_TOOLS:
                    tool_type = ToolType.MIRIX_EXTRA
                    tags = [tool_type.value]
                elif name in MCP_TOOLS:
                    tool_type = ToolType.MIRIX_EXTRA
                    tags = [tool_type.value, "mcp_wrapper"]
                else:
                    raise ValueError(f"Tool name {name} is not in the list of tool names")

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
        return tools
