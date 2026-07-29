import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from mirix.constants import (
    BASE_TOOLS,
    CHAT_AGENT_TOOLS,
    CORE_MEMORY_BLOCK_CHAR_LIMIT,
    CORE_MEMORY_TOOLS,
    EPISODIC_MEMORY_TOOLS,
    EXTRAS_TOOLS,
    KNOWLEDGE_VAULT_TOOLS,
    MCP_TOOLS,
    META_MEMORY_TOOLS,
    PROCEDURAL_MEMORY_TOOLS,
    RESOURCE_MEMORY_TOOLS,
    SEARCH_MEMORY_TOOLS,
    SEMANTIC_MEMORY_TOOLS,
    UNIVERSAL_MEMORY_TOOLS,
)
from mirix.log import get_logger
from mirix.orm import Agent as AgentModel
from mirix.orm import Tool as ToolModel
from mirix.orm.errors import NoResultFound

logger = get_logger(__name__)

# Diagnostic flag for MissingGreenlet debugging

_TRACE_MISSING_GREENLET = os.getenv("MIRIX_TRACE_MISSING_GREENLET", "false").lower() == "true"
from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.agent import AgentType, CreateAgent, CreateMetaAgent, UpdateAgent, UpdateMetaAgent
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import ToolType
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.tool_rule import ToolRule as PydanticToolRule
from mirix.schemas.user import User as PydanticUser
from mirix.services.block_manager import BlockManager
from mirix.services.helpers.agent_manager_helper import (
    _process_relationship,
    check_supports_structured_output,
    derive_system_message,
)
from mirix.services.message_manager import MessageManager
from mirix.services.tool_manager import ToolManager
from mirix.services.user_manager import UserManager
from mirix.utils import create_random_username, enforce_types

logger = get_logger(__name__)


# Agent Manager Class
class AgentManager:
    """Manager class to handle business logic related to Agents."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context
        self.tool_manager = ToolManager()
        self.message_manager = MessageManager()
        self.block_manager = BlockManager()

    def _monitor_cache_conflict(self, agent_id: str, db_updated_at: datetime, loaded_at: datetime) -> None:
        """
        Monitor for potential cache invalidation conflicts (stale writes).

        This method logs when an agent is being written with stale data,
        which could indicate a cache conflict from concurrent modifications.

        Args:
            agent_id: The agent ID being updated
            db_updated_at: When the agent was last updated in the database
            loaded_at: When the agent was loaded into memory

        Note:
            This is monitoring only - does not prevent writes.
            Use for collecting data to inform whether optimistic locking is needed.
        """
        if db_updated_at > loaded_at:
            # Potential stale write detected
            time_diff = (db_updated_at - loaded_at).total_seconds()
            logger.warning(
                "Potential stale agent write detected: agent=%s, " "db_updated_at=%s, loaded_at=%s, diff=%.2fs",
                agent_id,
                db_updated_at.isoformat(),
                loaded_at.isoformat(),
                time_diff,
            )
            # Future: Add metrics here
            # metrics.increment('agent.stale_write_detected', tags={'agent_id': agent_id})

    # ======================================================================================================================
    # Basic CRUD operations
    # ======================================================================================================================
    @enforce_types
    async def create_agent(
        self,
        agent_create: CreateAgent,
        actor: PydanticClient,
    ) -> PydanticAgentState:
        system = derive_system_message(agent_type=agent_create.agent_type, system=agent_create.system)

        if not agent_create.llm_config:
            raise ValueError("llm_config is required")

        # Check tool rules are valid
        if agent_create.tool_rules:
            check_supports_structured_output(model=agent_create.llm_config.model, tool_rules=agent_create.tool_rules)

        # TODO: Remove this block once we deprecate the legacy `tools` field
        # create passed in `tools`
        tool_names = []
        if agent_create.include_base_tools:
            tool_names.extend(BASE_TOOLS)
        if agent_create.tools:
            tool_names.extend(agent_create.tools)
        if agent_create.agent_type == AgentType.chat_agent:
            tool_names.extend(CHAT_AGENT_TOOLS + EXTRAS_TOOLS + MCP_TOOLS)
        if agent_create.agent_type == AgentType.episodic_memory_agent:
            tool_names.extend(EPISODIC_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.procedural_memory_agent:
            tool_names.extend(PROCEDURAL_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.resource_memory_agent:
            tool_names.extend(RESOURCE_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.knowledge_vault_memory_agent:
            tool_names.extend(KNOWLEDGE_VAULT_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.core_memory_agent:
            tool_names.extend(CORE_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.semantic_memory_agent:
            tool_names.extend(SEMANTIC_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.meta_memory_agent:
            tool_names.extend(META_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_create.agent_type == AgentType.reflexion_agent:
            tool_names.extend(SEARCH_MEMORY_TOOLS + CHAT_AGENT_TOOLS + UNIVERSAL_MEMORY_TOOLS + EXTRAS_TOOLS)

        # Remove duplicates
        tool_names = list(set(tool_names))

        tool_ids = list(agent_create.tool_ids or [])
        if tool_names:
            resolved = await self.tool_manager.list_tools_by_names(tool_names=tool_names, actor=actor)
            by_name = {t.name: t for t in resolved}
            for tool_name in tool_names:
                tool = by_name.get(tool_name)
                if tool:
                    tool_ids.append(tool.id)
                else:
                    logger.debug("Tool %s not found", tool_name)

        # Remove duplicates
        tool_ids = list(set(tool_ids))

        # Create the agent
        agent_state = await self._create_agent(
            name=agent_create.name,
            system=system,
            agent_type=agent_create.agent_type,
            llm_config=agent_create.llm_config,
            embedding_config=agent_create.embedding_config,
            tool_ids=tool_ids,
            tool_rules=agent_create.tool_rules,
            parent_id=agent_create.parent_id,
            actor=actor,
        )

        return agent_state

    async def create_meta_agent(
        self,
        meta_agent_create: CreateMetaAgent,
        actor: PydanticClient,
    ) -> Dict[str, PydanticAgentState]:
        """
        Create a meta agent by first creating a meta_memory_agent as the parent,
        then creating all the sub-agents specified in the meta_agent_create.agents list
        with their parent_id set to the meta_memory_agent.

        Args:
            meta_agent_create: CreateMetaAgent schema with configuration for all sub-agents
            actor: Client performing the action (for audit trail)

        Returns:
            Dict[str, PydanticAgentState]: Dictionary mapping agent names to their agent states,
                                           including the "meta_memory_agent" parent
        """

        if not meta_agent_create.llm_config:
            raise ValueError("llm_config is required")

        # Get organization's default user to serve as the template for block seeding
        user_manager = UserManager()
        assert actor.organization_id is not None
        # write_scope can be None for read-only clients — agents are still created
        # for search (embedding config), but block seeding is skipped.
        default_user = await user_manager.get_or_create_org_default_user(org_id=actor.organization_id)
        logger.debug(
            "Using organization default user %s for block templates in org %s",
            default_user.id,
            actor.organization_id,
        )

        # Ensure base tools are available in the database for this organization.
        # Uses the lightweight existence-check path: one batched read + creates
        # only for missing tools (zero on the common case where the startup
        # ``upsert_base_tools`` has already seeded the org's tool rows).
        await self.tool_manager.ensure_base_tools_exist(actor=actor)

        # Map agent names to their corresponding AgentType
        agent_name_to_type = {
            "core_memory_agent": AgentType.core_memory_agent,
            "resource_memory_agent": AgentType.resource_memory_agent,
            "semantic_memory_agent": AgentType.semantic_memory_agent,
            "episodic_memory_agent": AgentType.episodic_memory_agent,
            "procedural_memory_agent": AgentType.procedural_memory_agent,
            "knowledge_vault_memory_agent": AgentType.knowledge_vault_memory_agent,
            "meta_memory_agent": AgentType.meta_memory_agent,
            "reflexion_agent": AgentType.reflexion_agent,
            "background_agent": AgentType.background_agent,
            "chat_agent": AgentType.chat_agent,
        }

        # Load default system prompts from base folder
        default_system_prompts = {}
        base_prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system", "base")

        for filename in os.listdir(base_prompts_dir):
            if filename.endswith(".txt"):
                agent_name = filename[:-4]  # Strip .txt suffix
                prompt_file = os.path.join(base_prompts_dir, filename)
                with open(prompt_file, "r", encoding="utf-8") as f:
                    default_system_prompts[agent_name] = f.read()

        # First, create the meta_memory_agent as the parent
        meta_agent_name = meta_agent_create.name or "meta_memory_agent"
        meta_system_prompt = None
        if meta_agent_create.system_prompts and "meta_memory_agent" in meta_agent_create.system_prompts:
            meta_system_prompt = meta_agent_create.system_prompts["meta_memory_agent"]
        else:
            meta_system_prompt = default_system_prompts["meta_memory_agent"]

        meta_agent_create_schema = CreateAgent(
            name=meta_agent_name,
            agent_type=AgentType.meta_memory_agent,
            system=meta_system_prompt,
            llm_config=meta_agent_create.llm_config,
            embedding_config=meta_agent_create.embedding_config,
            include_base_tools=True,
        )

        meta_agent_state = await self.create_agent(
            agent_create=meta_agent_create_schema,
            actor=actor,
        )
        logger.debug(f"Created meta_memory_agent: {meta_agent_name} with id: {meta_agent_state.id}")

        # Store the parent agent
        created_agents = {}

        # Now create all sub-agents with parent_id set to the meta_memory_agent
        for agent_item in meta_agent_create.agents:
            # Parse agent_item - can be string or dict
            agent_name = None
            agent_config = {}

            if isinstance(agent_item, str):
                agent_name = agent_item
            elif isinstance(agent_item, dict):
                # Dict format: {agent_name: {config}}
                agent_name = list(agent_item.keys())[0]
                agent_config = agent_item[agent_name] or {}
            else:
                logger.warning("Invalid agent item format: %s, skipping...", agent_item)
                continue

            # Skip meta_memory_agent since we already created it as the parent
            if agent_name == "meta_memory_agent":
                continue

            # Get the agent type
            agent_type = agent_name_to_type.get(agent_name)
            if not agent_type:
                logger.warning("Unknown agent type: %s, skipping...", agent_name)
                continue

            # Get custom system prompt if provided, fallback to default
            custom_system = None
            if meta_agent_create.system_prompts and agent_name in meta_agent_create.system_prompts:
                custom_system = meta_agent_create.system_prompts[agent_name]
            elif agent_name in default_system_prompts:
                custom_system = default_system_prompts[agent_name]

            # Create the agent using CreateAgent schema with parent_id
            agent_create = CreateAgent(
                name=f"{meta_agent_name}_{agent_name}",
                agent_type=agent_type,
                system=custom_system,  # Uses custom prompt or default from base folder
                llm_config=meta_agent_create.llm_config,
                embedding_config=meta_agent_create.embedding_config,
                include_base_tools=True,
                parent_id=meta_agent_state.id,  # Set the parent_id
            )

            # Create the agent
            agent_state = await self.create_agent(
                agent_create=agent_create,
                actor=actor,
            )
            created_agents[agent_name] = agent_state
            logger.debug(f"Created sub-agent: {agent_name} with id: {agent_state.id}, parent_id: {meta_agent_state.id}")

            # Seed template blocks for this client's scope (e.g., blocks for core_memory_agent).
            # These are created for the org's default user with the client's write_scope.
            # When a real user first interacts via this scope, get_blocks() lazy-copies
            # these templates to create the user's own blocks for that scope.
            # Clients sharing the same write_scope share the same template blocks.
            if "blocks" in agent_config:
                memory_block_configs = agent_config["blocks"]
                for block_cfg in memory_block_configs:
                    await self.block_manager.seed_template_block_for_actor_scope_if_necessary(
                        label=block_cfg["label"],
                        value=block_cfg["value"],
                        limit=block_cfg.get("limit", CORE_MEMORY_BLOCK_CHAR_LIMIT),
                        actor=actor,
                        default_user=default_user,
                    )
                logger.debug(
                    "Seeded %d template blocks for scope=%s, agent=%s",
                    len(memory_block_configs),
                    actor.write_scope,
                    agent_name,
                )

            # Future: Add handling for other agent-specific configs here if needed
            # E.g., if 'initial_data' in agent_config: ...

        if created_agents:
            meta_agent_state.children = list(created_agents.values())
        return meta_agent_state

    async def update_meta_agent(
        self,
        meta_agent_id: str,
        meta_agent_update: UpdateMetaAgent,
        actor: PydanticClient,
    ) -> PydanticAgentState:
        """
        Update an existing meta agent and its sub-agents.

        Args:
            meta_agent_id: ID of the meta agent to update
            meta_agent_update: UpdateMetaAgent schema with fields to update
            actor: User performing the action

        Returns:
            PydanticAgentState: The updated meta agent state with children
        """
        # Get the existing meta agent
        meta_agent_state = await self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)

        # Verify this is actually a meta_memory_agent
        if meta_agent_state.agent_type != AgentType.meta_memory_agent:
            raise ValueError(f"Agent {meta_agent_id} is not a meta_memory_agent")

        # Map agent names to their corresponding AgentType
        agent_name_to_type = {
            "core_memory_agent": AgentType.core_memory_agent,
            "resource_memory_agent": AgentType.resource_memory_agent,
            "semantic_memory_agent": AgentType.semantic_memory_agent,
            "episodic_memory_agent": AgentType.episodic_memory_agent,
            "procedural_memory_agent": AgentType.procedural_memory_agent,
            "knowledge_vault_memory_agent": AgentType.knowledge_vault_memory_agent,
            "meta_memory_agent": AgentType.meta_memory_agent,
            "reflexion_agent": AgentType.reflexion_agent,
            "background_agent": AgentType.background_agent,
            "chat_agent": AgentType.chat_agent,
        }

        # Load default system prompts from base folder
        default_system_prompts = {}
        base_prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system", "base")

        for filename in os.listdir(base_prompts_dir):
            if filename.endswith(".txt"):
                agent_name = filename[:-4]  # Strip .txt suffix
                prompt_file = os.path.join(base_prompts_dir, filename)
                with open(prompt_file, "r", encoding="utf-8") as f:
                    default_system_prompts[agent_name] = f.read()

        # Build update fields for meta agent
        meta_agent_update_fields = {}
        if meta_agent_update.name is not None:
            meta_agent_update_fields["name"] = meta_agent_update.name
        if meta_agent_update.llm_config is not None:
            meta_agent_update_fields["llm_config"] = meta_agent_update.llm_config
        if meta_agent_update.embedding_config is not None:
            meta_agent_update_fields["embedding_config"] = meta_agent_update.embedding_config

        # Update meta agent with all fields at once. update_agent already returns
        # the hydrated state, so reuse it instead of a follow-up get_agent_by_id
        # (which is a guaranteed cache miss — the update just invalidated it).
        if meta_agent_update_fields:
            meta_agent_state = await self.update_agent(
                agent_id=meta_agent_id,
                agent_update=UpdateAgent(**meta_agent_update_fields),
                actor=actor,
            )

        # Update meta agent's system prompt if provided (reuse the return).
        if meta_agent_update.system_prompts and "meta_memory_agent" in meta_agent_update.system_prompts:
            meta_agent_state = await self.update_system_prompt(
                agent_id=meta_agent_id,
                system_prompt=meta_agent_update.system_prompts["meta_memory_agent"],
                actor=actor,
            )

        # Get existing sub-agents
        existing_children = await self.list_agents(actor=actor, parent_id=meta_agent_id)
        existing_agent_names = set()
        existing_agents_by_name = {}

        # for child in existing_children:
        #     # Extract agent type name from the full name (e.g., "meta_memory_agent_core_memory_agent" -> "core_memory_agent")
        #     for agent_type_name in agent_name_to_type.keys():
        #         if agent_type_name in child.name:
        #             existing_agent_names.add(agent_type_name)
        #             existing_agents_by_name[agent_type_name] = child
        #             break
        existing_agent_names = set([child.name.replace("meta_memory_agent_", "") for child in existing_children])
        existing_agents_by_name = {child.name.replace("meta_memory_agent_", ""): child for child in existing_children}

        # If agents list is provided, determine what needs to be created or deleted
        if meta_agent_update.agents is not None:
            # Parse the desired agent names from the update request
            desired_agent_names = set()
            agent_configs = {}

            for agent_item in meta_agent_update.agents:
                if isinstance(agent_item, str):
                    agent_name = agent_item
                    agent_configs[agent_name] = {}
                elif isinstance(agent_item, dict):
                    agent_name = list(agent_item.keys())[0]
                    agent_configs[agent_name] = agent_item[agent_name] or {}
                else:
                    logger.warning("Invalid agent item format: %s, skipping...", agent_item)
                    continue

                # Skip meta_memory_agent as it's the parent
                if agent_name != "meta_memory_agent":
                    desired_agent_names.add(agent_name)

            # Determine agents to create and delete
            agents_to_create = desired_agent_names - existing_agent_names
            agents_to_delete = existing_agent_names - desired_agent_names

            # Delete agents that are no longer needed
            for agent_name in agents_to_delete:
                if agent_name in existing_agents_by_name:
                    child_agent = existing_agents_by_name[agent_name]
                    logger.debug("Deleting sub-agent: %s with id: %s", agent_name, child_agent.id)
                    await self.delete_agent(agent_id=child_agent.id, actor=actor)

            # Create new agents
            for agent_name in agents_to_create:
                agent_type = agent_name_to_type.get(agent_name)
                if not agent_type:
                    logger.warning("Unknown agent type: %s, skipping...", agent_name)
                    continue

                # Get custom system prompt if provided, fallback to default
                custom_system = None
                if meta_agent_update.system_prompts and agent_name in meta_agent_update.system_prompts:
                    custom_system = meta_agent_update.system_prompts[agent_name]
                elif agent_name in default_system_prompts:
                    custom_system = default_system_prompts[agent_name]

                # Use the updated configs or fall back to meta agent's configs
                llm_config = meta_agent_update.llm_config or meta_agent_state.llm_config
                embedding_config = meta_agent_update.embedding_config or meta_agent_state.embedding_config

                # Create the agent using CreateAgent schema with parent_id
                agent_create = CreateAgent(
                    name=f"{meta_agent_state.name}_{agent_name}",
                    agent_type=agent_type,
                    system=custom_system,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                    include_base_tools=True,
                    parent_id=meta_agent_id,
                )

                # Create the agent
                new_agent_state = await self.create_agent(
                    agent_create=agent_create,
                    actor=actor,
                )
                existing_agents_by_name[agent_name] = new_agent_state
                logger.debug(
                    f"Created sub-agent: {agent_name} with id: {new_agent_state.id}, parent_id: {meta_agent_id}"
                )

        # Update system prompts for existing sub-agents
        if meta_agent_update.system_prompts:
            for agent_name, system_prompt in meta_agent_update.system_prompts.items():
                # Skip meta_memory_agent as we already updated it
                if agent_name == "meta_memory_agent":
                    continue

                if agent_name in existing_agents_by_name:
                    child_agent = existing_agents_by_name[agent_name]
                    if child_agent.system != system_prompt:
                        logger.debug("Updating system prompt for sub-agent: %s", agent_name)
                        await self.update_system_prompt(
                            agent_id=child_agent.id,
                            system_prompt=system_prompt,
                            actor=actor,
                        )

        # Update llm_config and embedding_config for all sub-agents if provided.
        # Pass the already-loaded child as current_state so update_agent applies
        # the changed scalars locally and skips the post-write hydrate read.
        if meta_agent_update.llm_config or meta_agent_update.embedding_config:
            for agent_name, child_agent in existing_agents_by_name.items():
                update_fields = {}
                if meta_agent_update.llm_config is not None:
                    update_fields["llm_config"] = meta_agent_update.llm_config
                if meta_agent_update.embedding_config is not None:
                    update_fields["embedding_config"] = meta_agent_update.embedding_config

                if update_fields:
                    logger.debug("Updating configs for sub-agent: %s", agent_name)
                    await self.update_agent(
                        agent_id=child_agent.id,
                        agent_update=UpdateAgent(**update_fields),
                        actor=actor,
                        current_state=child_agent,
                    )

        # Refresh the meta agent state with updated children
        meta_agent_state = await self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)
        updated_children = await self.list_agents(actor=actor, parent_id=meta_agent_id)
        meta_agent_state.children = updated_children

        return meta_agent_state

    async def update_agent_tools_and_system_prompts(
        self,
        agent_id: str,
        actor: PydanticClient,
        system_prompt: Optional[str] = None,
    ):
        agent_state = await self.get_agent_by_id(agent_id=agent_id, actor=actor)

        # update the system prompt
        if system_prompt is not None:
            if not agent_state.system == system_prompt:
                await self.update_system_prompt(agent_id=agent_id, system_prompt=system_prompt, actor=actor)

        # update the tools
        ## get the new tool names
        tool_names = []
        if agent_state.agent_type == AgentType.episodic_memory_agent:
            tool_names.extend(EPISODIC_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.procedural_memory_agent:
            tool_names.extend(PROCEDURAL_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.resource_memory_agent:
            tool_names.extend(RESOURCE_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.knowledge_vault_memory_agent:
            tool_names.extend(KNOWLEDGE_VAULT_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.core_memory_agent:
            tool_names.extend(CORE_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.semantic_memory_agent:
            tool_names.extend(SEMANTIC_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.meta_memory_agent:
            tool_names.extend(META_MEMORY_TOOLS + UNIVERSAL_MEMORY_TOOLS)
        if agent_state.agent_type == AgentType.chat_agent:
            tool_names.extend(BASE_TOOLS + CHAT_AGENT_TOOLS + EXTRAS_TOOLS)
        if agent_state.agent_type == AgentType.reflexion_agent:
            tool_names.extend(SEARCH_MEMORY_TOOLS + CHAT_AGENT_TOOLS + UNIVERSAL_MEMORY_TOOLS + EXTRAS_TOOLS)

        ## extract the existing tool names for the agent
        existing_tools = agent_state.tools
        existing_tool_names = set([tool.name for tool in existing_tools])
        existing_tool_ids = [tool.id for tool in existing_tools]

        # Separate MCP tools from native tools - preserve MCP tools
        mcp_tools = [tool for tool in existing_tools if tool.tool_type == ToolType.MIRIX_MCP]
        mcp_tool_names = set([tool.name for tool in mcp_tools])
        mcp_tool_ids = [tool.id for tool in mcp_tools]

        new_tool_names = [tool_name for tool_name in tool_names if tool_name not in existing_tool_names]
        # Only remove non-MCP tools that aren't in the expected tool list
        tool_names_to_remove = [
            tool_name
            for tool_name in existing_tool_names
            if tool_name not in tool_names and tool_name not in mcp_tool_names
        ]

        # Start with existing tool IDs, ensuring MCP tools are always preserved
        tool_ids = existing_tool_ids.copy()

        # Ensure all MCP tools are preserved (in case they were missed)
        for mcp_tool_id in mcp_tool_ids:
            if mcp_tool_id not in tool_ids:
                tool_ids.append(mcp_tool_id)

        # Resolve adds and removes in a single batched lookup to avoid N+1
        # IPSR roundtrips on the force_update=true path.
        names_to_resolve = list(set(new_tool_names) | set(tool_names_to_remove))
        resolved_by_name: dict = {}
        if names_to_resolve:
            resolved = await self.tool_manager.list_tools_by_names(tool_names=names_to_resolve, actor=actor)
            resolved_by_name = {t.name: t for t in resolved}

        # Add new tools
        if new_tool_names:
            for tool_name in new_tool_names:
                tool = resolved_by_name.get(tool_name)
                if tool:
                    tool_ids.append(tool.id)

        # Remove tools that should no longer be attached
        if tool_names_to_remove:
            tools_to_remove_ids = []
            for tool_name in tool_names_to_remove:
                tool = resolved_by_name.get(tool_name)
                if tool:
                    tools_to_remove_ids.append(tool.id)

            # Filter out the tools to be removed
            tool_ids = [tool_id for tool_id in tool_ids if tool_id not in tools_to_remove_ids]

        # Update the agent if there are any changes
        if len(new_tool_names) > 0 or len(tool_names_to_remove) > 0:
            await self.update_agent(
                agent_id=agent_id,
                agent_update=UpdateAgent(tool_ids=tool_ids),
                actor=actor,
            )

    @enforce_types
    async def _create_agent(
        self,
        actor: PydanticClient,
        name: str,
        system: str,
        agent_type: AgentType,
        llm_config: LLMConfig,
        embedding_config: Optional[EmbeddingConfig],
        tool_ids: List[str],
        tool_rules: Optional[List[PydanticToolRule]] = None,
        parent_id: Optional[str] = None,
    ) -> PydanticAgentState:
        """Create a new agent."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            if name is None:
                name = create_random_username()

            data_dict = {
                # Pre-generate a UUID so Relational DB provider uses it as the system entity.id.
                # Relational DB provider requires a valid UUID for engine table entity.id.
                # Using str(uuid.uuid4()) (no prefix) ensures the provider accepts it directly.
                # The matching entity_key stores this UUID for natural-key lookups.
                "id": str(uuid.uuid4()),
                "name": name,
                "system": system,
                "agent_type": agent_type,
                "llm_config": llm_config.model_dump() if hasattr(llm_config, "model_dump") else llm_config,
                "embedding_config": (
                    embedding_config.model_dump() if embedding_config and hasattr(embedding_config, "model_dump") else embedding_config
                ),
                "organization_id": actor.organization_id,
                "tools": tool_ids,
                "tool_rules": (
                    [tr.model_dump() if hasattr(tr, "model_dump") else tr for tr in tool_rules] if tool_rules else None
                ),
                "parent_id": parent_id,
                # Pass the client UUID so the Relational DB provider can store it
                # in ipsr_entity_owner, enabling correct created_by_id round-trip.
                "_created_by_id": actor.id,
            }
            result = await provider.create("agents", data_dict)
            agent_state = PydanticAgentState(**result)

            if parent_id:
                await self._invalidate_parent_cache_for_child(agent_state.id, parent_id)

            return agent_state

        async with self.session_maker() as session:
            # Generate a random name if none provided
            if name is None:
                name = create_random_username()

            # Prepare the agent data
            data = {
                "name": name,
                "system": system,
                "agent_type": agent_type,
                "llm_config": llm_config,
                "embedding_config": embedding_config,
                "organization_id": actor.organization_id,
                "tool_rules": tool_rules,
                "parent_id": parent_id,
            }

            # Create the new agent using SqlalchemyBase.create_with_redis
            new_agent = AgentModel(**data)
            await _process_relationship(session, new_agent, "tools", ToolModel, tool_ids, replace=True)
            await new_agent.create_with_redis(session, actor=actor)  # Auto-caches to Redis

            # Invalidate parent cache if this is a child agent
            if parent_id:
                await self._invalidate_parent_cache_for_child(new_agent.id, parent_id)

            # Convert to PydanticAgentState and return
            return new_agent.to_pydantic()

    @enforce_types
    async def update_agent(
        self,
        agent_id: str,
        agent_update: UpdateAgent,
        actor: PydanticClient,
        current_state: Optional[PydanticAgentState] = None,
    ) -> PydanticAgentState:
        # Update agent (system prompt and all other fields are persisted directly).
        # ``current_state`` (the caller's already-loaded agent) lets _update_agent
        # skip the post-write hydrate read on non-tool updates — see _update_agent.
        return await self._update_agent(
            agent_id=agent_id,
            agent_update=agent_update,
            actor=actor,
            current_state=current_state,
        )

    @enforce_types
    async def update_llm_config(
        self, agent_id: str, llm_config: LLMConfig, actor: PydanticClient
    ) -> PydanticAgentState:
        # Skip the write when the persisted config already matches.
        # Callers on the registration / initialize-meta-agent path invoke this
        # per agent; an unchanged config otherwise produces a needless write.
        current = await self.get_agent_by_id(agent_id=agent_id, actor=actor)
        if current.llm_config == llm_config:
            return current
        return await self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(llm_config=llm_config),
            actor=actor,
        )

    @enforce_types
    async def update_system_prompt(
        self, agent_id: str, system_prompt: str, actor: PydanticClient
    ) -> PydanticAgentState:
        return await self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(system=system_prompt),
            actor=actor,
        )

    @enforce_types
    async def update_mcp_tools(
        self,
        agent_id: str,
        mcp_tools: List[str],
        actor: PydanticClient,
        tool_ids: List[str],
    ) -> PydanticAgentState:
        """Update the MCP tools connected to an agent."""
        return await self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(mcp_tools=mcp_tools, tool_ids=tool_ids),
            actor=actor,
        )

    @enforce_types
    async def add_mcp_tool(
        self,
        agent_id: str,
        mcp_tool_name: str,
        tool_ids: List[str],
        actor: PydanticClient,
    ) -> PydanticAgentState:
        """Add a single MCP tool to an agent."""
        # First get the current agent state
        agent_state = await self.get_agent_by_id(agent_id=agent_id, actor=actor)
        current_mcp_tools = agent_state.mcp_tools or []

        # Add the new MCP tool if not already present
        if mcp_tool_name not in current_mcp_tools:
            current_mcp_tools.append(mcp_tool_name)
            return await self.update_mcp_tools(
                agent_id=agent_id,
                mcp_tools=current_mcp_tools,
                actor=actor,
                tool_ids=tool_ids,
            )

        return agent_state

    @enforce_types
    async def _update_agent(
        self,
        agent_id: str,
        agent_update: UpdateAgent,
        actor: PydanticClient,
        current_state: Optional[PydanticAgentState] = None,
    ) -> PydanticAgentState:
        """
        Update an existing agent.

        Args:
            agent_id: The ID of the agent to update.
            agent_update: UpdateAgent object containing the updated fields.
            actor: User performing the action.
            current_state: The agent's already-loaded state, if the caller has
                it. When supplied AND the update does not touch ``tool_ids``, the
                returned state is built by applying the changed scalar fields onto
                this state instead of re-reading the row with its tools — saving
                the post-write hydrate read (the dominant per-write cost under the
                IPS Relational provider). Omit it (the default) to keep the
                read-after-write behaviour for callers that need a freshly
                hydrated row.

        Returns:
            PydanticAgentState: The updated agent as a Pydantic model.
        """
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            changing_parent = agent_update.parent_id is not None
            # Only pre-read to capture the OLD parent for parent-cache
            # invalidation when the parent is actually changing. For the common
            # case (config / system-prompt updates that never touch parent_id)
            # this read is pure waste — skip it.
            old_parent_id = None
            if changing_parent:
                existing = await provider.read("agents", agent_id)
                old_parent_id = existing.get("parent_id") if existing else None

            scalar_fields = ["name", "system", "llm_config", "embedding_config", "tool_rules", "mcp_tools", "parent_id"]
            update_data: dict = {}
            for field in scalar_fields:
                value = getattr(agent_update, field, None)
                if value is not None:
                    update_data[field] = value.model_dump() if hasattr(value, "model_dump") else value

            tools_changed = agent_update.tool_ids is not None
            if tools_changed:
                update_data["tools"] = agent_update.tool_ids

            await provider.update("agents", agent_id, update_data)

            # Build the returned state. provider.update returns only scalar
            # columns (no hydrated ``tools``), and AgentState.tools is required,
            # so normally we re-read with include_relationships=["tools"]. When
            # the caller passed its already-loaded ``current_state`` AND this
            # update didn't change tools, apply the changed scalar fields onto a
            # copy of that state instead — avoiding the hydrate read (1 read +
            # per-tool join) entirely. Otherwise fall back to the re-read.
            if current_state is not None and not tools_changed:
                agent_state = current_state.model_copy(
                    update={
                        k: getattr(agent_update, k) for k in scalar_fields if getattr(agent_update, k, None) is not None
                    }
                )
            else:
                result = await provider.read("agents", agent_id, include_relationships=["tools"])
                if result is None:
                    raise ValueError(f"Agent {agent_id} disappeared between update and read-back")
                agent_state = PydanticAgentState(**result)

            # Invalidate cache if available
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    await cache_provider.delete(f"{cache_provider.AGENT_PREFIX}{agent_id}")
            except Exception as e:
                logger.warning("Cache invalidation failed for agent %s: %s", agent_id, e)

            # Invalidate parent caches only when the parent actually changed
            # (old_parent_id is only populated in that case).
            new_parent_id = agent_state.parent_id
            if old_parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, old_parent_id)
            if changing_parent and new_parent_id and new_parent_id != old_parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, new_parent_id)

            return agent_state

        async with self.session_maker() as session:
            # Retrieve the existing agent
            agent = await AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Track old parent_id for cache invalidation
            old_parent_id = agent.parent_id

            # Update scalar fields directly
            scalar_fields = {
                "name",
                "system",
                "llm_config",
                "embedding_config",
                "tool_rules",
                "mcp_tools",
                "parent_id",
            }
            for field in scalar_fields:
                value = getattr(agent_update, field, None)
                if value is not None:
                    setattr(agent, field, value)

            # Update relationships using _process_relationship
            if agent_update.tool_ids is not None:
                await _process_relationship(
                    session,
                    agent,
                    "tools",
                    ToolModel,
                    agent_update.tool_ids,
                    replace=True,
                )

            # Commit and refresh the agent, update Redis cache
            await agent.update_with_redis(session, actor=actor)  # Updates Redis cache

            # Invalidate parent caches if parent_id changed or agent has a parent
            if old_parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, old_parent_id)
            if agent.parent_id and agent.parent_id != old_parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, agent.parent_id)

            # Convert to PydanticAgentState and return
            return agent.to_pydantic()

    async def _invalidate_parent_cache_for_child(self, child_agent_id: str, parent_id: Optional[str] = None) -> None:
        """
        Invalidate parent agent cache when a child agent is created/updated/deleted.

        Args:
            child_agent_id: ID of the child agent that changed
            parent_id: Optional parent_id if known, otherwise will look up from reverse mapping
        """
        try:
            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()

            if not redis_client:
                return

            # If parent_id not provided, try to get it from reverse mapping
            if not parent_id:
                reverse_key = f"{redis_client.AGENT_PREFIX}{child_agent_id}:parent"
                parent_id_bytes = await redis_client.client.get(reverse_key)
                if parent_id_bytes:
                    parent_id = (
                        parent_id_bytes.decode("utf-8") if isinstance(parent_id_bytes, bytes) else parent_id_bytes
                    )

            # Invalidate parent agent cache
            if parent_id:
                parent_key = f"{redis_client.AGENT_PREFIX}{parent_id}"
                await redis_client.delete(parent_key)
                logger.debug(
                    "Invalidated parent agent %s cache due to child %s change",
                    parent_id,
                    child_agent_id,
                )

                # Clean up reverse mapping if this is a deletion
                reverse_key = f"{redis_client.AGENT_PREFIX}{child_agent_id}:parent"
                await redis_client.delete(reverse_key)

        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            logger.warning("Failed to invalidate parent cache for child %s: %s", child_agent_id, e)

    async def _reconstruct_children_from_cache(
        self,
        agent_states: List[PydanticAgentState],
        session: Session,
        actor: PydanticClient,
    ) -> dict:
        """
        Reconstruct children for parent agents from Redis cache.
        Falls back to PostgreSQL if Redis is unavailable or data is missing.

        Args:
            agent_states: List of parent agents
            session: Database session for fallback
            actor: User performing the operation

        Returns:
            Dictionary mapping parent_id -> list of child agents
        """
        import json

        from mirix.database.redis_client import get_redis_client
        from mirix.schemas.tool import Tool as PydanticTool

        children_by_parent = {}
        parent_ids = [agent.id for agent in agent_states]

        try:
            redis_client = get_redis_client()
            if not redis_client:
                # Redis not available, fall back to PostgreSQL
                return await self._get_children_from_db(parent_ids, session, actor)

            # Step 1: Fetch parent agents from Redis to get children_ids
            pipe = redis_client.client.pipeline()
            for parent_id in parent_ids:
                pipe.hgetall(f"{redis_client.AGENT_PREFIX}{parent_id}")
            parent_results = await pipe.execute()

            # Extract all children IDs
            all_children_ids = []
            parent_to_children_ids = {}
            for i, parent_data in enumerate(parent_results):
                if parent_data and "children_ids" in parent_data:
                    children_ids_str = parent_data["children_ids"]
                    if isinstance(children_ids_str, bytes):
                        children_ids_str = children_ids_str.decode("utf-8")
                    children_ids = json.loads(children_ids_str) if children_ids_str else []
                    parent_to_children_ids[parent_ids[i]] = children_ids
                    all_children_ids.extend(children_ids)

            if not all_children_ids:
                # No children IDs found, return empty
                return children_by_parent

            # Step 2: Fetch all child agents from Redis using pipeline
            pipe = redis_client.client.pipeline()
            for child_id in all_children_ids:
                pipe.hgetall(f"{redis_client.AGENT_PREFIX}{child_id}")
            child_results = await pipe.execute()

            # Build mapping of child_id -> reconstructed child agent
            children_cache = {}
            missing_child_ids = []

            for i, child_data in enumerate(child_results):
                child_id = all_children_ids[i]
                if not child_data:
                    missing_child_ids.append(child_id)
                    continue

                # Deserialize JSON fields
                if "llm_config" in child_data:
                    child_data["llm_config"] = (
                        json.loads(child_data["llm_config"])
                        if isinstance(child_data["llm_config"], (str, bytes))
                        else child_data["llm_config"]
                    )
                if "embedding_config" in child_data:
                    child_data["embedding_config"] = (
                        json.loads(child_data["embedding_config"])
                        if isinstance(child_data["embedding_config"], (str, bytes))
                        else child_data["embedding_config"]
                    )
                if "tool_rules" in child_data:
                    child_data["tool_rules"] = (
                        json.loads(child_data["tool_rules"])
                        if isinstance(child_data["tool_rules"], (str, bytes))
                        else child_data["tool_rules"]
                    )
                if "mcp_tools" in child_data:
                    child_data["mcp_tools"] = (
                        json.loads(child_data["mcp_tools"])
                        if isinstance(child_data["mcp_tools"], (str, bytes))
                        else child_data["mcp_tools"]
                    )

                # Reconstruct tools from Redis
                tools = []
                if "tool_ids" in child_data and child_data["tool_ids"]:
                    tool_ids = (
                        json.loads(child_data["tool_ids"])
                        if isinstance(child_data["tool_ids"], (str, bytes))
                        else child_data["tool_ids"]
                    )

                    tool_pipe = redis_client.client.pipeline()
                    for tool_id in tool_ids:
                        tool_pipe.hgetall(f"{redis_client.TOOL_PREFIX}{tool_id}")
                    tool_results = await tool_pipe.execute()

                    for tool_data in tool_results:
                        if tool_data:
                            if "json_schema" in tool_data and isinstance(tool_data["json_schema"], (str, bytes)):
                                tool_data["json_schema"] = json.loads(tool_data["json_schema"])
                            if "tags" in tool_data and isinstance(tool_data["tags"], (str, bytes)):
                                tool_data["tags"] = json.loads(tool_data["tags"])
                            tools.append(PydanticTool(**tool_data))

                child_data["tools"] = tools
                child_data.pop("tool_ids", None)
                child_data.pop("memory_block_ids", None)
                child_data.pop("memory_prompt_template", None)
                # Strip legacy fields no longer on AgentState
                for legacy_key in "memory":
                    child_data.pop(legacy_key, None)

                # Children don't need their own children reconstructed (1-level depth only)
                child_data["children"] = None
                child_data.pop("children_ids", None)

                children_cache[child_id] = PydanticAgentState(**child_data)

            # If any children are missing from cache, fall back to PostgreSQL for ALL children
            if missing_child_ids:
                logger.warning(
                    "Some children not found in Redis cache (%s missing), falling back to PostgreSQL",
                    len(missing_child_ids),
                )
                return await self._get_children_from_db(parent_ids, session, actor)

            # Step 3: Group children by parent_id
            for parent_id, children_ids in parent_to_children_ids.items():
                children_by_parent[parent_id] = [
                    children_cache[child_id] for child_id in children_ids if child_id in children_cache
                ]

            logger.debug(
                "Reconstructed children for %s parent agents from Redis cache",
                len(children_by_parent),
            )
            return children_by_parent

        except Exception as e:
            # Log error and fall back to PostgreSQL
            logger.warning("Failed to reconstruct children from cache: %s", e)
            return await self._get_children_from_db(parent_ids, session, actor)

    async def _get_children_from_db(self, parent_ids: List[str], session: Session, actor: PydanticClient) -> dict:
        """
        Fallback method to get children from PostgreSQL with client-level filtering.

        Args:
            parent_ids: List of parent agent IDs
            session: Database session
            actor: Client performing the operation (for client-level isolation)

        Returns:
            Dictionary mapping parent_id -> list of child agents
        """
        # Query all agents for this client (triggers client-level filtering via apply_access_predicate)
        children = await AgentModel.list(
            db_session=session,
            actor=actor,  # Triggers client-level filtering (organization_id + _created_by_id)
        )

        # Filter children by parent_id and group them
        children_by_parent = {}
        for child in children:
            if child.parent_id in parent_ids:
                if child.parent_id not in children_by_parent:
                    children_by_parent[child.parent_id] = []
                children_by_parent[child.parent_id].append(child.to_pydantic())

        logger.debug(
            "Retrieved children for %s parent agents from PostgreSQL (client-filtered)",
            len(children_by_parent),
        )
        return children_by_parent

    async def _get_children_from_redis(
        self, parent_id: str, actor: PydanticClient
    ) -> Optional[List[PydanticAgentState]]:
        """
        Fetch children from Redis cache using parent's children_ids.

        Args:
            parent_id: ID of the parent agent
            actor: User performing the operation

        Returns:
            List of child agents if found in cache, None if cache miss
        """
        try:
            import json

            from mirix.database.redis_client import get_redis_client

            redis_client = get_redis_client()
            if not redis_client:
                return None

            # Get parent's cache to retrieve children_ids
            parent_key = f"{redis_client.AGENT_PREFIX}{parent_id}"
            parent_data = await redis_client.get_hash(parent_key)

            if not parent_data or "children_ids" not in parent_data:
                # Parent not in cache or doesn't have children_ids
                return None

            # Parse children_ids
            children_ids_str = parent_data["children_ids"]
            if isinstance(children_ids_str, bytes):
                children_ids_str = children_ids_str.decode("utf-8")
            children_ids = json.loads(children_ids_str) if children_ids_str else []

            if not children_ids:
                # Parent has no children
                return []

            # Fetch each child using get_agent_by_id (which uses Redis cache)
            children = []
            for child_id in children_ids:
                try:
                    child = await self.get_agent_by_id(child_id, actor)
                    children.append(child)
                except NoResultFound:
                    # Child not found - cache inconsistency
                    logger.warning(
                        "Child agent %s not found for parent %s, cache inconsistent",
                        child_id,
                        parent_id,
                    )
                    return None  # Fall back to PostgreSQL for consistency

            logger.debug(
                "Retrieved %s children for parent %s from Redis cache",
                len(children),
                parent_id,
            )
            return children

        except Exception as e:
            # Log error and return None to trigger PostgreSQL fallback
            logger.warning("Failed to get children from cache for parent %s: %s", parent_id, e)
            return None

    async def _cache_children_ids_for_parents(self, agent_states: List[PydanticAgentState]) -> None:
        """
        Cache children_ids for parent agents that have children populated.
        This enables future list_agents(parent_id=X) calls to use Redis cache.

        Args:
            agent_states: List of parent agents with children populated
        """
        try:
            import json

            from mirix.database.redis_client import get_redis_client
            from mirix.settings import settings

            redis_client = get_redis_client()
            if not redis_client:
                return

            for agent_state in agent_states:
                if agent_state.children:
                    # Extract children IDs
                    children_ids = [child.id for child in agent_state.children]

                    # Update parent's cache with children_ids
                    parent_key = f"{redis_client.AGENT_PREFIX}{agent_state.id}"
                    await redis_client.client.hset(parent_key, "children_ids", json.dumps(children_ids))

                    # Maintain reverse mapping for cache invalidation
                    for child_id in children_ids:
                        reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                        await redis_client.client.set(reverse_key, agent_state.id)
                        await redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

            logger.debug(
                "Cached children_ids for %s parent agents",
                len([a for a in agent_states if a.children]),
            )
        except Exception as e:
            # Log but don't fail if caching fails
            logger.warning("Failed to cache children_ids for parent agents: %s", e)

    @enforce_types
    async def list_agents(
        self,
        actor: PydanticClient,
        match_all_tags: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        parent_id: Optional[str] = None,
        user: Optional[PydanticUser] = None,
        sort_desc: bool = False,
        include_tools: bool = True,
        **kwargs,
    ) -> List[PydanticAgentState]:
        """
        List agents that have the specified tags.
        By default, only returns top-level agents (parent_id is None) with their children populated.
        If parent_id is provided, only returns agents with that parent_id.

        When parent_id is provided, tries to use Redis cache via parent's children_ids first,
        then falls back to PostgreSQL if cache miss.

        ``sort_desc`` selects descending vs ascending creation-time ordering for the
        Relational DB provider path (``agent_manager.list_agents_desc`` vs ``agent_manager.list_agents_asc``).

        ``include_tools`` controls whether each returned agent's ``tools``
        relationship is hydrated. Under the IPS Relational provider that
        hydration costs one ``tool_manager.list_tools_by_ids`` round-trip PER
        agent (an N+1). Callers that only need an agent's config (e.g. the
        search / topic-extraction read paths, which use a single agent's
        ``llm_config`` / ``embedding_config`` and never its tools) should pass
        ``include_tools=False`` — typically with ``limit=1`` — to avoid that
        per-agent tool fan-out. Defaults to True so existing callers that rely
        on populated ``tools`` are unchanged.
        """
        from mirix.database.relational_provider import get_relational_provider

        # Tools are an M2M relationship hydrated per-agent by the relational
        # provider; only request it when the caller actually needs tools.
        rel_include = ["tools"] if include_tools else None

        def _to_agent_state(row: dict) -> PydanticAgentState:
            # When tools are not hydrated (include_tools=False) the provider row
            # has no ``tools`` key, but AgentState.tools is a required field.
            # Default it to an empty list so the model still validates — callers
            # that pass include_tools=False (search / embedding-config reads) do
            # not consume ``tools``.
            if "tools" not in row:
                row = {**row, "tools": []}
            return PydanticAgentState(**row)

        rel_provider = get_relational_provider()
        # A-4: query_text path (new provider branch — replaces SQLAlchemy fallback)
        if rel_provider and query_text and parent_id is None and not kwargs:
            results = await rel_provider.find_using_named_query(
                "agents",
                "agent_manager.list_agents_by_query_text",
                params={
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                    "queryText": f"%{query_text}%",
                },
                page_size=limit or 50,
                include_relationships=rel_include,
            )
            return [_to_agent_state(r) for r in results]

        # A-3: default provider list path — choose ascending vs descending NQ based on sort_desc
        if rel_provider and parent_id is None and not query_text and not kwargs:
            query_name = "agent_manager.list_agents_desc" if sort_desc else "agent_manager.list_agents_asc"
            results = await rel_provider.find_using_named_query(
                "agents",
                query_name,
                params={
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                    "name": None,
                    "parentId": None,
                },
                page_size=limit or 50,
                include_relationships=rel_include,
            )
            return [_to_agent_state(r) for r in results]

        # A-11: parent_id wide list (named query lists up to 1000 by org+client, parent_id
        # filtered in Python). PostgreSQL doesn't hold the data when
        # ENGINE_DB_STRATEGY=ips_relational.
        if rel_provider and parent_id is not None and not query_text and not kwargs:
            all_agents = await rel_provider.find_using_named_query(
                "agents",
                "agent_manager.list_agents_rel_provider",
                params={
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                },
                page_size=1000,
                include_relationships=rel_include,
            )
            results = [r for r in all_agents if r.get("parent_id") == parent_id]
            return [_to_agent_state(r) for r in results]

        # Optimization: Use Redis cache for list_agents(parent_id=X)
        if parent_id is not None:
            cached_children = await self._get_children_from_redis(parent_id, actor)
            if cached_children is not None:
                logger.debug("Cache HIT for children of parent %s", parent_id)
                return cached_children
            # Cache miss - fall through to PostgreSQL query
            logger.debug(
                "Cache MISS for children of parent %s, querying PostgreSQL",
                parent_id,
            )

        async with self.session_maker() as session:
            # Get agents filtered by parent_id (None for top-level agents, or specific parent_id)
            # Actor triggers apply_access_predicate which filters by both organization_id and _created_by_id (client isolation)
            agents = await AgentModel.list(
                db_session=session,
                actor=actor,  # Triggers client-level filtering via apply_access_predicate
                match_all_tags=match_all_tags,
                cursor=cursor,
                limit=limit,
                query_text=query_text,
                parent_id=parent_id,
                **kwargs,
            )

            # Convert to Pydantic
            if _TRACE_MISSING_GREENLET:
                logger.info("Converting %d agents to Pydantic in list_agents", len(agents))
                agent_states = []
                for i, agent in enumerate(agents):
                    try:
                        agent_states.append(agent.to_pydantic())
                    except Exception as e:
                        if "MissingGreenlet" in str(type(e).__name__) or "greenlet" in str(e).lower():
                            import traceback

                            logger.error(
                                "MissingGreenlet in list_agents at index %d, agent_id=%s\n" "Full traceback:\n%s",
                                i,
                                agent.id,
                                traceback.format_exc(),
                            )
                        raise
            else:
                agent_states = [agent.to_pydantic() for agent in agents]

            # If there are no agents, return early
            if not agent_states:
                return agent_states

            # Only populate children if we're listing top-level agents (parent_id is None)
            if parent_id is None:
                children_by_parent = await self._reconstruct_children_from_cache(agent_states, session, actor)

                # Assign children to their parent agents
                for agent_state in agent_states:
                    agent_state.children = children_by_parent.get(agent_state.id, [])

                # Cache children_ids for future list_agents(parent_id=X) calls
                await self._cache_children_ids_for_parents(agent_states)

            return agent_states

    async def list_agents_with_tools(self, parent_id: str, actor: PydanticClient) -> List[PydanticAgentState]:
        """Fetch all child agents of ``parent_id`` WITH their tools in a single
        relational-provider roundtrip.

        Replaces the N+1 where ``list_agents(parent_id=...)`` returns the agents
        and the provider then resolves each agent's ``tools`` relationship with a
        separate ``list_tools_by_ids`` call. The joined named query
        ``agent_manager.list_agents_with_tools_by_parent`` returns one flat
        ``(agent, tool)`` row per pair (tool columns NULL for an agent with no
        tools, via LEFT JOIN); we group them agent-side into
        ``PydanticAgentState`` objects with ``tools`` populated.

        Rows are projected by the NQ (``skip_entity_mapping=True``) and do NOT go
        through ``FieldMapper`` camel/snake mapping. The deployed NQ via the
        IPS-R SDK returns each row as a *positional* list (the column-order
        contract of a multi-column projection), not a named dict — so we pass a
        ``result_set_entity_class`` (``AgentToolRow``) whose field order matches
        the NQ SELECT order. The relational provider zips each positional row
        against those field names and hands back uniform dicts keyed by the
        aliases, which we then JSON-parse and group below. This is the same
        ``result_set_entity_class`` mechanism the other manager NQs use for
        non-entity projections (see ``mirix/database/named_query_results.py``).

        Falls back to ``list_agents`` (legacy per-agent tool hydration) when no
        relational provider is active (e.g. PostgreSQL-backed tests).
        """
        import json

        from mirix.database.named_query_results import AgentToolRow
        from mirix.database.relational_provider import get_relational_provider
        from mirix.schemas.tool import Tool as PydanticTool

        rel_provider = get_relational_provider()
        if rel_provider is None:
            return await self.list_agents(actor=actor, parent_id=parent_id)

        rows = await rel_provider.find_using_named_query(
            "agents",
            "agent_manager.list_agents_with_tools_by_parent",
            params={
                "organizationId": actor.organization_id,
                "createdById": actor.id,
                "parentId": parent_id,
            },
            page_size=1000,
            skip_entity_mapping=True,
            result_set_entity_class=AgentToolRow,
        )

        def _parse_json(value):
            if value is None or isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("list_agents_with_tools: failed to parse JSON column")
                    return None
            return value

        agents_by_id: Dict[str, PydanticAgentState] = {}
        tool_ids_seen: Dict[str, set] = {}

        for row in rows:
            aid = row.get("agent_id")
            if not aid:
                continue
            if aid not in agents_by_id:
                agents_by_id[aid] = PydanticAgentState(
                    id=aid,
                    name=row.get("agent_name") or "",
                    agent_type=AgentType(row.get("agent_type")),
                    system=row.get("agent_system") or "",
                    description=row.get("agent_description"),
                    parent_id=row.get("agent_parent_id"),
                    organization_id=row.get("agent_organization_id"),
                    llm_config=LLMConfig(**_parse_json(row.get("agent_llm_config"))),
                    tool_rules=_parse_json(row.get("agent_tool_rules")),
                    mcp_tools=_parse_json(row.get("agent_mcp_tools")),
                    tools=[],
                )
                tool_ids_seen[aid] = set()

            tid = row.get("tool_id")
            if tid and tid not in tool_ids_seen[aid]:
                tool_ids_seen[aid].add(tid)
                agents_by_id[aid].tools.append(
                    PydanticTool(
                        id=tid,
                        name=row.get("tool_name"),
                        description=row.get("tool_description"),
                        json_schema=_parse_json(row.get("tool_json_schema")),
                        tool_type=ToolType(row.get("tool_type")) if row.get("tool_type") else ToolType.CUSTOM,
                        organization_id=row.get("tool_organization_id"),
                    )
                )

        return list(agents_by_id.values())

    @enforce_types
    async def get_agent_by_id(self, agent_id: str, actor: PydanticClient) -> PydanticAgentState:
        """Fetch an agent by its ID (with cache and Redis pipeline for tools/blocks)."""
        import json

        from mirix.database.cache_provider import get_cache_provider
        from mirix.database.redis_client import get_redis_client
        from mirix.log import get_logger
        from mirix.schemas.tool import Tool as PydanticTool

        logger = get_logger(__name__)
        cache_provider = get_cache_provider()
        redis_client = get_redis_client()

        try:
            if cache_provider:
                cache_key = f"{cache_provider.AGENT_PREFIX}{agent_id}"
                cached_data = await cache_provider.get_hash(cache_key)
                if cached_data and redis_client:
                    logger.debug("Cache HIT for agent %s", agent_id)

                    # Deserialize JSON fields
                    if "llm_config" in cached_data:
                        cached_data["llm_config"] = (
                            json.loads(cached_data["llm_config"])
                            if isinstance(cached_data["llm_config"], str)
                            else cached_data["llm_config"]
                        )
                    if "embedding_config" in cached_data:
                        cached_data["embedding_config"] = (
                            json.loads(cached_data["embedding_config"])
                            if isinstance(cached_data["embedding_config"], str)
                            else cached_data["embedding_config"]
                        )
                    if "tool_rules" in cached_data:
                        cached_data["tool_rules"] = (
                            json.loads(cached_data["tool_rules"])
                            if isinstance(cached_data["tool_rules"], str)
                            else cached_data["tool_rules"]
                        )
                    if "mcp_tools" in cached_data:
                        cached_data["mcp_tools"] = (
                            json.loads(cached_data["mcp_tools"])
                            if isinstance(cached_data["mcp_tools"], str)
                            else cached_data["mcp_tools"]
                        )

                    # Retrieve tools from Redis using pipeline (denormalized tools_agents)
                    tools = []
                    if "tool_ids" in cached_data and cached_data["tool_ids"]:
                        tool_ids = (
                            json.loads(cached_data["tool_ids"])
                            if isinstance(cached_data["tool_ids"], str)
                            else cached_data["tool_ids"]
                        )

                        # Use pipeline for efficient parallel retrieval
                        pipe = redis_client.client.pipeline()
                        for tool_id in tool_ids:
                            pipe.hgetall(f"{redis_client.TOOL_PREFIX}{tool_id}")
                        tool_results = await pipe.execute()

                        # Deserialize tool data
                        for tool_data in tool_results:
                            if tool_data:
                                # Convert Redis hash data to proper types
                                if "json_schema" in tool_data and isinstance(tool_data["json_schema"], str):
                                    tool_data["json_schema"] = json.loads(tool_data["json_schema"])
                                if "tags" in tool_data and isinstance(tool_data["tags"], str):
                                    tool_data["tags"] = json.loads(tool_data["tags"])
                                tools.append(PydanticTool(**tool_data))

                    cached_data["tools"] = tools
                    cached_data.pop("tool_ids", None)  # Remove denormalized field
                    cached_data.pop("memory_block_ids", None)
                    cached_data.pop("memory_prompt_template", None)
                    # Strip legacy fields no longer on AgentState
                    for legacy_key in "memory":
                        cached_data.pop(legacy_key, None)

                    agent_state = PydanticAgentState(**cached_data)

                    # SECURITY CHECK: Verify agent belongs to this client
                    # Prevents cross-client access via Redis cache
                    if agent_state.created_by_id != actor.id:
                        raise NoResultFound(f"Agent {agent_id} not found or not accessible to client {actor.id}")

                    return agent_state  # Cache HIT (agent + tools + memory)
        except Exception as e:
            logger.debug("Cache read failed for agent %s: %s", agent_id, e)

        # Relational DB provider delegation (named query with include_relationships for tools)
        from mirix.database.relational_provider import get_relational_provider

        rel_provider = get_relational_provider()
        if rel_provider:
            # Use the named query (which filters by organizationId + ipsrentityowner) rather
            # than provider.read(actor=actor) so we get include_relationships=["tools"] in
            # a single round trip.  The actor param on provider.read would handle org/client
            # isolation identically but does not support relationship expansion.
            rows = await rel_provider.find_using_named_query(
                "agents",
                "agent_manager.get_agent_by_id",
                params={
                    "id": agent_id,
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                },
                page_size=1,
                include_relationships=["tools"],
            )
            if not rows:
                raise NoResultFound(f"Agent {agent_id} not found")
            agent_state = PydanticAgentState(**rows[0])
            # created_by_id security check is enforced at the query level (ipsrentityowner
            # == :createdById in the YAML NQ), so no additional Python check is needed.

            try:
                if cache_provider:
                    from mirix.settings import settings

                    data = agent_state.model_dump(mode="json")
                    if "llm_config" in data and data["llm_config"]:
                        data["llm_config"] = json.dumps(data["llm_config"])
                    if "embedding_config" in data and data["embedding_config"]:
                        data["embedding_config"] = json.dumps(data["embedding_config"])
                    if "tool_rules" in data and data["tool_rules"]:
                        data["tool_rules"] = json.dumps(data["tool_rules"])
                    if "mcp_tools" in data and data["mcp_tools"]:
                        data["mcp_tools"] = json.dumps(data["mcp_tools"])
                    if "tools" in data and data["tools"]:
                        tool_ids = [t["id"] for t in data["tools"]]
                        data["tool_ids"] = json.dumps(tool_ids)
                        for tool in data["tools"]:
                            tool_key = f"{cache_provider.TOOL_PREFIX}{tool['id']}"
                            tool_data = dict(tool)
                            if "json_schema" in tool_data and tool_data["json_schema"]:
                                tool_data["json_schema"] = json.dumps(tool_data["json_schema"])
                            if "tags" in tool_data and tool_data["tags"]:
                                tool_data["tags"] = json.dumps(tool_data["tags"])
                            await cache_provider.set_hash(tool_key, tool_data, ttl=settings.redis_ttl_tools)
                    data.pop("tools", None)
                    data.pop("children", None)
                    agent_cache_key = f"{cache_provider.AGENT_PREFIX}{agent_id}"
                    await cache_provider.set_hash(agent_cache_key, data, ttl=settings.redis_ttl_agents)
            except Exception as e:
                logger.warning("Failed to populate cache after provider read for agent %s: %s", agent_id, e)

            return agent_state

        # Cache MISS or no cache - fetch from PostgreSQL with client filtering
        async with self.session_maker() as session:
            # AgentModel.read calls apply_access_predicate, which now filters by client (organization_id + _created_by_id)
            # If agent doesn't belong to this client, read() will raise NoResultFound automatically
            agent = await AgentModel.read(
                db_session=session,
                identifier=agent_id,
                actor=actor,  # Triggers client-level filtering via apply_access_predicate
            )

            if _TRACE_MISSING_GREENLET:
                try:
                    logger.info("Converting agent %s to Pydantic in get_agent_by_id", agent_id)
                    pydantic_agent = agent.to_pydantic()
                except Exception as e:
                    if "MissingGreenlet" in str(type(e).__name__) or "greenlet" in str(e).lower():
                        import traceback

                        logger.error(
                            "MissingGreenlet in get_agent_by_id for agent_id=%s\n" "Full traceback:\n%s",
                            agent_id,
                            traceback.format_exc(),
                        )
                    raise
            else:
                pydantic_agent = agent.to_pydantic()

            # Populate cache for next time
            try:
                if cache_provider:
                    from mirix.settings import settings

                    data = pydantic_agent.model_dump(mode="json")

                    if "llm_config" in data and data["llm_config"]:
                        data["llm_config"] = json.dumps(data["llm_config"])
                    if "embedding_config" in data and data["embedding_config"]:
                        data["embedding_config"] = json.dumps(data["embedding_config"])
                    if "tool_rules" in data and data["tool_rules"]:
                        data["tool_rules"] = json.dumps(data["tool_rules"])
                    if "mcp_tools" in data and data["mcp_tools"]:
                        data["mcp_tools"] = json.dumps(data["mcp_tools"])

                    if "tools" in data and data["tools"]:
                        tool_ids = [tool["id"] for tool in data["tools"]]
                        data["tool_ids"] = json.dumps(tool_ids)

                        for tool in data["tools"]:
                            tool_key = f"{cache_provider.TOOL_PREFIX}{tool['id']}"
                            tool_data = dict(tool)
                            if "json_schema" in tool_data and tool_data["json_schema"]:
                                tool_data["json_schema"] = json.dumps(tool_data["json_schema"])
                            if "tags" in tool_data and tool_data["tags"]:
                                tool_data["tags"] = json.dumps(tool_data["tags"])
                            await cache_provider.set_hash(tool_key, tool_data, ttl=settings.redis_ttl_tools)

                    if "children" in data and data["children"]:
                        children_ids = [
                            child["id"] if isinstance(child, dict) else child.id for child in data["children"]
                        ]
                        data["children_ids"] = json.dumps(children_ids)

                        if redis_client:
                            for child_id in children_ids:
                                reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                                await redis_client.client.set(reverse_key, agent_id)
                                await redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    data.pop("tools", None)
                    data.pop("children", None)

                    agent_cache_key = f"{cache_provider.AGENT_PREFIX}{agent_id}"
                    await cache_provider.set_hash(agent_cache_key, data, ttl=settings.redis_ttl_agents)
                    logger.debug("Populated cache for agent %s with tools", agent_id)
            except Exception as e:
                logger.warning("Failed to populate cache for agent %s: %s", agent_id, e)

            return pydantic_agent

    @enforce_types
    async def get_agent_by_name(self, agent_name: str, actor: PydanticClient) -> PydanticAgentState:
        """Fetch an agent by its name."""
        from mirix.database.relational_provider import get_relational_provider

        rel_provider = get_relational_provider()
        if rel_provider:
            results = await rel_provider.find_using_named_query(
                "agents",
                "agent_manager.get_agent_by_name",
                params={
                    "organizationId": actor.organization_id,
                    "name": agent_name,
                    "createdById": actor.id,
                },
                page_size=1,
                include_relationships=["tools"],
            )
            if not results:
                raise NoResultFound(f"Agent with name {agent_name} not found")
            return PydanticAgentState(**results[0])

        async with self.session_maker() as session:
            agent = await AgentModel.read(db_session=session, name=agent_name, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    async def size_agents(self, actor: PydanticClient) -> int:
        """Return the count of non-deleted agents for the actor's organization.

        Routes through the ``sqlalchemy_base.size_agents`` named query when the
        Relational DB provider is registered; falls back to ``AgentModel.size`` otherwise.
        ``SqlalchemyBase.size()`` is deliberately not modified to keep the ORM base
        class decoupled from the provider.
        """
        from mirix.database.relational_provider import get_relational_provider

        rel_provider = get_relational_provider()
        if rel_provider is not None:
            from mirix.database.named_query_results import CountResult

            rows = await rel_provider.find_using_named_query(
                "agents",
                "sqlalchemy_base.size_agents",
                params={
                    "organizationId": actor.organization_id,
                    "createdById": actor.id,
                },
                result_set_entity_class=CountResult,
                page_size=1,
            )
            if not rows:
                return 0
            return int(rows[0].get("count") or 0)

        async with self.session_maker() as session:
            return await AgentModel.size(db_session=session, actor=actor)

    @enforce_types
    async def delete_agent(self, agent_id: str, actor: PydanticClient) -> None:
        """
        Deletes an agent and its associated relationships.
        Ensures proper permission checks and cascades where applicable.

        Args:
            agent_id: ID of the agent to be deleted.
            actor: User performing the action.

        Raises:
            NoResultFound: If agent doesn't exist
        """
        from mirix.database.relational_provider import get_relational_provider

        rel_provider = get_relational_provider()
        if rel_provider:
            existing = await rel_provider.read("agents", agent_id)
            if existing is None:
                raise NoResultFound(f"Agent {agent_id} not found")
            parent_id = existing.get("parent_id")
            await rel_provider.hard_delete("agents", agent_id)
            try:
                from mirix.database.cache_provider import get_cache_provider

                cache_provider = get_cache_provider()
                if cache_provider:
                    await cache_provider.delete(f"{cache_provider.AGENT_PREFIX}{agent_id}")
            except Exception:
                pass
            if parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, parent_id)
            return

        async with self.session_maker() as session:
            # Retrieve the agent
            agent = await AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Track parent_id for cache invalidation
            parent_id = agent.parent_id

            # Remove from cache before hard delete
            try:
                from mirix.database.cache_provider import get_cache_provider
                from mirix.log import get_logger

                logger = get_logger(__name__)
                cache_provider = get_cache_provider()
                if cache_provider:
                    cache_key = f"{cache_provider.AGENT_PREFIX}{agent_id}"
                    await cache_provider.delete(cache_key)
                    logger.debug("Removed agent %s from cache", agent_id)
            except Exception as e:
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning("Failed to remove agent %s from cache: %s", agent_id, e)

            await agent.hard_delete(session)

            # Invalidate parent cache if this was a child agent
            if parent_id:
                await self._invalidate_parent_cache_for_child(agent_id, parent_id)

    # ======================================================================================================================
    # Message Management
    # ======================================================================================================================
    @enforce_types
    async def reset_messages(
        self,
        agent_id: str,
        actor: PydanticClient,
        user_id: Optional[str] = None,
    ) -> PydanticAgentState:
        """
        Removes messages belonging to the specified user from the agent's conversation history.

        This action is destructive and cannot be undone once committed.

        Args:
            agent_id (str): The ID of the agent whose messages will be reset.
            actor (PydanticClient): The Client performing this action.
            user_id (str): The user whose messages will be removed. If None, removes all non-system messages.

        Returns:
            PydanticAgentState: The updated agent state.
        """
        if user_id:
            await self.message_manager.hard_delete_user_messages_for_agent(
                agent_id=agent_id,
                user_id=user_id,
                actor=actor,
                keep_newest_n=0,
            )
        else:
            # Delete all non-system messages for every user of this agent
            from sqlalchemy import delete

            from mirix.orm.message import Message as MessageModel
            from mirix.schemas.message import MessageRole

            async with self.session_maker() as session:
                await session.execute(
                    delete(MessageModel).where(
                        MessageModel.agent_id == agent_id,
                        MessageModel.organization_id == actor.organization_id,
                        MessageModel.role != MessageRole.system,
                    )
                )
                await session.commit()

        return await self.get_agent_by_id(agent_id=agent_id, actor=actor)

    # ======================================================================================================================
    # Tool Management
    # ======================================================================================================================
    @enforce_types
    async def attach_tool(self, agent_id: str, tool_id: str, actor: PydanticClient) -> PydanticAgentState:
        """
        Attaches a tool to an agent.

        Args:
            agent_id: ID of the agent to attach the tool to.
            tool_id: ID of the tool to attach.
            actor: User performing the action.

        Raises:
            NoResultFound: If the agent or tool is not found.

        Returns:
            PydanticAgentState: The updated agent state.
        """
        async with self.session_maker() as session:
            # Verify the agent exists and user has permission to access it
            agent = await AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            await _process_relationship(
                session=session,
                agent=agent,
                relationship_name="tools",
                model_class=ToolModel,
                item_ids=[tool_id],
                allow_partial=False,
                replace=False,
            )

            # Commit and refresh the agent
            await agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    async def detach_tool(self, agent_id: str, tool_id: str, actor: PydanticClient) -> PydanticAgentState:
        """
        Detaches a tool from an agent.

        Args:
            agent_id: ID of the agent to detach the tool from.
            tool_id: ID of the tool to detach.
            actor: User performing the action.

        Raises:
            NoResultFound: If the agent or tool is not found.

        Returns:
            PydanticAgentState: The updated agent state.
        """
        async with self.session_maker() as session:
            # Verify the agent exists and user has permission to access it
            agent = await AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Filter out the tool to be detached
            remaining_tools = [tool for tool in agent.tools if tool.id != tool_id]

            if len(remaining_tools) == len(agent.tools):  # Tool ID was not in the relationship
                logger.warning(
                    f"Attempted to remove unattached tool id={tool_id} from agent id={agent_id} by actor={actor}"
                )

            # Update the tools relationship
            agent.tools = remaining_tools

            # Commit and refresh the agent
            await agent.update(session, actor=actor)
            return agent.to_pydantic()
