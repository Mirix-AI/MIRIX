import os
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
from mirix.orm import Block as BlockModel
from mirix.orm import Tool as ToolModel
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.agent import (
    AgentType,
    CreateAgent,
    CreateMetaAgent,
    UpdateAgent,
    UpdateMetaAgent,
)
from mirix.schemas.block import Block
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import ToolType
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.message import MessageCreate
from mirix.schemas.tool_rule import ToolRule as PydanticToolRule
from mirix.schemas.user import User as PydanticUser
from mirix.services.block_manager import BlockManager
from mirix.services.helpers.agent_manager_helper import (
    _process_relationship,
    check_supports_structured_output,
    derive_system_message,
    initialize_message_sequence,
    package_initial_message_sequence,
)
from mirix.services.message_manager import MessageManager
from mirix.services.tool_manager import ToolManager
from mirix.services.user_manager import UserManager
from mirix.utils import create_random_username, enforce_types, get_utc_time

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
    def create_agent(
        self,
        agent_create: CreateAgent,
        actor: PydanticClient,
    ) -> PydanticAgentState:
        system = derive_system_message(agent_type=agent_create.agent_type, system=agent_create.system)

        if not agent_create.llm_config or not agent_create.embedding_config:
            raise ValueError("llm_config and embedding_config are required")

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

        tool_ids = agent_create.tool_ids or []
        for tool_name in tool_names:
            tool = self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
            if tool:
                tool_ids.append(tool.id)
            else:
                logger.debug("Tool %s not found", tool_name)

        # Remove duplicates
        tool_ids = list(set(tool_ids))

        # Create the agent
        agent_state = self._create_agent(
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

        return self.append_initial_message_sequence_to_in_context_messages(
            actor, agent_state, agent_create.initial_message_sequence
        )

    def create_meta_agent(
        self,
        meta_agent_create: CreateMetaAgent,
        actor: PydanticClient,
        user_id: Optional[str] = None,  # NEW: user_id for block creation
    ) -> Dict[str, PydanticAgentState]:
        """
        Create a meta agent by first creating a meta_memory_agent as the parent,
        then creating all the sub-agents specified in the meta_agent_create.agents list
        with their parent_id set to the meta_memory_agent.

        Args:
            meta_agent_create: CreateMetaAgent schema with configuration for all sub-agents
            actor: Client performing the action (for audit trail)
            user_id: Optional user_id for block creation (uses default user if not provided)

        Returns:
            Dict[str, PydanticAgentState]: Dictionary mapping agent names to their agent states,
                                           including the "meta_memory_agent" parent
        """

        if not meta_agent_create.llm_config or not meta_agent_create.embedding_config:
            raise ValueError("llm_config and embedding_config are required")

        # NEW: Get or create organization-specific default user for block templates
        user_manager = UserManager()

        if user_id:
            # Specific user_id provided - use it
            try:
                user = user_manager.get_user_by_id(user_id)
            except Exception as e:
                logger.warning(
                    "Failed to load user with id=%s, falling back to org default user: %s",
                    user_id,
                    e,
                )
                # Fall back to organization's default user (not global admin)
                user = user_manager.get_or_create_org_default_user(org_id=actor.organization_id, client_id=actor.id)
        else:
            # No user_id provided - use organization's default template user
            # This user will serve as the template for copying blocks to new users
            user = user_manager.get_or_create_org_default_user(org_id=actor.organization_id, client_id=actor.id)
            logger.debug(
                "Using organization default user %s for block templates in org %s",
                user.id,
                actor.organization_id,
            )

        # Ensure base tools are available in the database for this organization
        self.tool_manager.upsert_base_tools(actor=actor)

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

        meta_agent_state = self.create_agent(
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
            agent_state = self.create_agent(
                agent_create=agent_create,
                actor=actor,
            )
            created_agents[agent_name] = agent_state
            logger.debug(f"Created sub-agent: {agent_name} with id: {agent_state.id}, parent_id: {meta_agent_state.id}")

            # FIX: Process agent-specific initialization (e.g., blocks for core_memory_agent)
            # This handles any agent configuration provided in the meta_agent creation request
            if "blocks" in agent_config:
                # Create memory blocks for this agent (typically core_memory_agent)
                memory_block_configs = agent_config["blocks"]
                for block in memory_block_configs:
                    self.block_manager.create_or_update_block(
                        block=Block(
                            value=block["value"],
                            limit=block.get("limit", CORE_MEMORY_BLOCK_CHAR_LIMIT),
                            label=block["label"],
                        ),
                        actor=actor,
                        agent_id=agent_state.id,  # Use child agent's ID, not parent's
                        user=user,  # Pass user for block creation
                    )
                logger.debug(
                    f"Created {len(memory_block_configs)} memory blocks for {agent_name} (agent_id: {agent_state.id})"
                )

                # Ensure blocks are committed to database before proceeding
                # This is critical for template block copying to work correctly
                logger.debug(f"Flushing database session to ensure blocks are committed for agent {agent_state.id}")
                with self.block_manager.session_maker() as session:
                    session.commit()  # Explicit commit to ensure blocks are visible to other sessions

            # Future: Add handling for other agent-specific configs here if needed
            # E.g., if 'initial_data' in agent_config: ...

        if created_agents:
            meta_agent_state.children = list(created_agents.values())
        return meta_agent_state

    def update_meta_agent(
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
        meta_agent_state = self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)

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

        # Update meta agent with all fields at once
        if meta_agent_update_fields:
            self.update_agent(
                agent_id=meta_agent_id,
                agent_update=UpdateAgent(**meta_agent_update_fields),
                actor=actor,
            )
            meta_agent_state = self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)

        # Update meta agent's system prompt if provided (separate call needed for rebuild_system_prompt)
        if meta_agent_update.system_prompts and "meta_memory_agent" in meta_agent_update.system_prompts:
            self.update_system_prompt(
                agent_id=meta_agent_id,
                system_prompt=meta_agent_update.system_prompts["meta_memory_agent"],
                actor=actor,
            )
            meta_agent_state = self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)

        # Get existing sub-agents
        existing_children = self.list_agents(actor=actor, parent_id=meta_agent_id)
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
                    self.delete_agent(agent_id=child_agent.id, actor=actor)

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
                new_agent_state = self.create_agent(
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
                        self.update_system_prompt(
                            agent_id=child_agent.id,
                            system_prompt=system_prompt,
                            actor=actor,
                        )

        # Update llm_config and embedding_config for all sub-agents if provided
        if meta_agent_update.llm_config or meta_agent_update.embedding_config:
            for agent_name, child_agent in existing_agents_by_name.items():
                update_fields = {}
                if meta_agent_update.llm_config is not None:
                    update_fields["llm_config"] = meta_agent_update.llm_config
                if meta_agent_update.embedding_config is not None:
                    update_fields["embedding_config"] = meta_agent_update.embedding_config

                if update_fields:
                    logger.debug("Updating configs for sub-agent: %s", agent_name)
                    self.update_agent(
                        agent_id=child_agent.id,
                        agent_update=UpdateAgent(**update_fields),
                        actor=actor,
                    )

        # Refresh the meta agent state with updated children
        meta_agent_state = self.get_agent_by_id(agent_id=meta_agent_id, actor=actor)
        updated_children = self.list_agents(actor=actor, parent_id=meta_agent_id)
        meta_agent_state.children = updated_children

        return meta_agent_state

    def update_agent_tools_and_system_prompts(
        self,
        agent_id: str,
        actor: PydanticClient,
        system_prompt: Optional[str] = None,
    ):
        agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)

        # update the system prompt
        if system_prompt is not None:
            if not agent_state.system == system_prompt:
                self.update_system_prompt(agent_id=agent_id, system_prompt=system_prompt, actor=actor)

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

        # Add new tools
        if len(new_tool_names) > 0:
            for tool_name in new_tool_names:
                tool = self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
                if tool:
                    tool_ids.append(tool.id)

        # Remove tools that should no longer be attached
        if len(tool_names_to_remove) > 0:
            tools_to_remove_ids = []
            for tool_name in tool_names_to_remove:
                tool = self.tool_manager.get_tool_by_name(tool_name=tool_name, actor=actor)
                if tool:
                    tools_to_remove_ids.append(tool.id)

            # Filter out the tools to be removed
            tool_ids = [tool_id for tool_id in tool_ids if tool_id not in tools_to_remove_ids]

        # Update the agent if there are any changes
        if len(new_tool_names) > 0 or len(tool_names_to_remove) > 0:
            self.update_agent(
                agent_id=agent_id,
                agent_update=UpdateAgent(tool_ids=tool_ids),
                actor=actor,
            )

    @enforce_types
    def _generate_initial_message_sequence(
        self,
        actor: PydanticClient,
        agent_state: PydanticAgentState,
        supplied_initial_message_sequence: Optional[List[MessageCreate]] = None,
        user_id: Optional[str] = None,
    ) -> List[PydanticMessage]:
        init_messages = initialize_message_sequence(
            agent_state=agent_state,
            memory_edit_timestamp=get_utc_time(),
            include_initial_boot_message=True,
        )
        if supplied_initial_message_sequence is not None:
            # We always need the system prompt up front
            system_message_obj = PydanticMessage.dict_to_message(
                agent_id=agent_state.id,
                model=agent_state.llm_config.model,
                openai_message_dict=init_messages[0],
            )
            # Don't use anything else in the pregen sequence, instead use the provided sequence
            init_messages = [system_message_obj]
            init_messages.extend(
                package_initial_message_sequence(
                    agent_state.id,
                    supplied_initial_message_sequence,
                    agent_state.llm_config.model,
                    actor,
                    user_id=user_id,
                )
            )
        else:
            init_messages = [
                PydanticMessage.dict_to_message(
                    agent_id=agent_state.id,
                    model=agent_state.llm_config.model,
                    openai_message_dict=msg,
                )
                for msg in init_messages
            ]

        return init_messages

    @enforce_types
    def append_initial_message_sequence_to_in_context_messages(
        self,
        actor: PydanticClient,
        agent_state: PydanticAgentState,
        initial_message_sequence: Optional[List[MessageCreate]] = None,
        user_id: Optional[str] = None,
    ) -> PydanticAgentState:
        init_messages = self._generate_initial_message_sequence(
            actor, agent_state, initial_message_sequence, user_id=user_id
        )
        return self.append_to_in_context_messages(init_messages, agent_id=agent_state.id, actor=actor, user_id=user_id)

    @enforce_types
    def _create_agent(
        self,
        actor: PydanticClient,
        name: str,
        system: str,
        agent_type: AgentType,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        tool_ids: List[str],
        tool_rules: Optional[List[PydanticToolRule]] = None,
        parent_id: Optional[str] = None,
    ) -> PydanticAgentState:
        """Create a new agent."""
        with self.session_maker() as session:
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
            _process_relationship(session, new_agent, "tools", ToolModel, tool_ids, replace=True)
            new_agent.create_with_redis(session, actor=actor)  # Auto-caches to Redis

            # Invalidate parent cache if this is a child agent
            if parent_id:
                self._invalidate_parent_cache_for_child(new_agent.id, parent_id)

            # Convert to PydanticAgentState and return
            return new_agent.to_pydantic()

    @enforce_types
    def update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: PydanticClient) -> PydanticAgentState:
        # Get current state BEFORE update to detect changes
        old_agent_state = None
        if agent_update.system:
            old_agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)

        # Update agent (including system field in database)
        agent_state = self._update_agent(agent_id=agent_id, agent_update=agent_update, actor=actor)

        # Rebuild the system prompt if it changed
        if agent_update.system and old_agent_state and agent_update.system != old_agent_state.system:
            agent_state = self.rebuild_system_prompt(
                agent_id=agent_state.id,
                system_prompt=agent_update.system,  # Pass the new system prompt
                actor=actor,
                force=True,
            )

        return agent_state

    @enforce_types
    def update_llm_config(self, agent_id: str, llm_config: LLMConfig, actor: PydanticClient) -> PydanticAgentState:
        return self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(llm_config=llm_config),
            actor=actor,
        )

    @enforce_types
    def update_system_prompt(self, agent_id: str, system_prompt: str, actor: PydanticClient) -> PydanticAgentState:
        agent_state = self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(system=system_prompt),
            actor=actor,
        )
        # Rebuild the system prompt if it's different
        agent_state = self.rebuild_system_prompt(
            agent_id=agent_state.id,
            system_prompt=system_prompt,
            actor=actor,
            force=True,
        )
        return agent_state

    @enforce_types
    def update_mcp_tools(
        self,
        agent_id: str,
        mcp_tools: List[str],
        actor: PydanticClient,
        tool_ids: List[str],
    ) -> PydanticAgentState:
        """Update the MCP tools connected to an agent."""
        return self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(mcp_tools=mcp_tools, tool_ids=tool_ids),
            actor=actor,
        )

    @enforce_types
    def add_mcp_tool(
        self,
        agent_id: str,
        mcp_tool_name: str,
        tool_ids: List[str],
        actor: PydanticClient,
    ) -> PydanticAgentState:
        """Add a single MCP tool to an agent."""
        # First get the current agent state
        agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)
        current_mcp_tools = agent_state.mcp_tools or []

        # Add the new MCP tool if not already present
        if mcp_tool_name not in current_mcp_tools:
            current_mcp_tools.append(mcp_tool_name)
            return self.update_mcp_tools(
                agent_id=agent_id,
                mcp_tools=current_mcp_tools,
                actor=actor,
                tool_ids=tool_ids,
            )

        return agent_state

    @enforce_types
    def _update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: PydanticClient) -> PydanticAgentState:
        """
        Update an existing agent.

        Args:
            agent_id: The ID of the agent to update.
            agent_update: UpdateAgent object containing the updated fields.
            actor: User performing the action.

        Returns:
            PydanticAgentState: The updated agent as a Pydantic model.
        """
        with self.session_maker() as session:
            # Retrieve the existing agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Track old parent_id for cache invalidation
            old_parent_id = agent.parent_id

            # Update scalar fields directly
            scalar_fields = {
                "name",
                "system",
                "llm_config",
                "embedding_config",
                "message_ids",
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
                _process_relationship(
                    session,
                    agent,
                    "tools",
                    ToolModel,
                    agent_update.tool_ids,
                    replace=True,
                )

            # Commit and refresh the agent, update Redis cache
            agent.update_with_redis(session, actor=actor)  # Updates Redis cache

            # Invalidate parent caches if parent_id changed or agent has a parent
            if old_parent_id:
                self._invalidate_parent_cache_for_child(agent_id, old_parent_id)
            if agent.parent_id and agent.parent_id != old_parent_id:
                self._invalidate_parent_cache_for_child(agent_id, agent.parent_id)

            # Convert to PydanticAgentState and return
            return agent.to_pydantic()

    def _invalidate_parent_cache_for_child(self, child_agent_id: str, parent_id: Optional[str] = None) -> None:
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
                parent_id_bytes = redis_client.client.get(reverse_key)
                if parent_id_bytes:
                    parent_id = (
                        parent_id_bytes.decode("utf-8") if isinstance(parent_id_bytes, bytes) else parent_id_bytes
                    )

            # Invalidate parent agent cache
            if parent_id:
                parent_key = f"{redis_client.AGENT_PREFIX}{parent_id}"
                redis_client.delete(parent_key)
                logger.debug(
                    "Invalidated parent agent %s cache due to child %s change",
                    parent_id,
                    child_agent_id,
                )

                # Clean up reverse mapping if this is a deletion
                reverse_key = f"{redis_client.AGENT_PREFIX}{child_agent_id}:parent"
                redis_client.delete(reverse_key)

        except Exception as e:
            # Log but don't fail the operation if cache invalidation fails
            logger.warning("Failed to invalidate parent cache for child %s: %s", child_agent_id, e)

    def _reconstruct_children_from_cache(
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
        from mirix.schemas.block import Block as PydanticBlock
        from mirix.schemas.memory import Memory as PydanticMemory
        from mirix.schemas.tool import Tool as PydanticTool

        children_by_parent = {}
        parent_ids = [agent.id for agent in agent_states]

        try:
            redis_client = get_redis_client()
            if not redis_client:
                # Redis not available, fall back to PostgreSQL
                return self._get_children_from_db(parent_ids, session, actor)

            # Step 1: Fetch parent agents from Redis to get children_ids
            pipe = redis_client.client.pipeline()
            for parent_id in parent_ids:
                pipe.hgetall(f"{redis_client.AGENT_PREFIX}{parent_id}")
            parent_results = pipe.execute()

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
            child_results = pipe.execute()

            # Build mapping of child_id -> reconstructed child agent
            children_cache = {}
            missing_child_ids = []

            for i, child_data in enumerate(child_results):
                child_id = all_children_ids[i]
                if not child_data:
                    missing_child_ids.append(child_id)
                    continue

                # Deserialize JSON fields
                if "message_ids" in child_data:
                    child_data["message_ids"] = (
                        json.loads(child_data["message_ids"])
                        if isinstance(child_data["message_ids"], (str, bytes))
                        else child_data["message_ids"]
                    )
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
                    tool_results = tool_pipe.execute()

                    for tool_data in tool_results:
                        if tool_data:
                            if "json_schema" in tool_data and isinstance(tool_data["json_schema"], (str, bytes)):
                                tool_data["json_schema"] = json.loads(tool_data["json_schema"])
                            if "tags" in tool_data and isinstance(tool_data["tags"], (str, bytes)):
                                tool_data["tags"] = json.loads(tool_data["tags"])
                            tools.append(PydanticTool(**tool_data))

                child_data["tools"] = tools
                child_data.pop("tool_ids", None)

                # Reconstruct memory from blocks
                blocks = []
                prompt_template = ""

                if "memory_block_ids" in child_data and child_data["memory_block_ids"]:
                    block_ids = (
                        json.loads(child_data["memory_block_ids"])
                        if isinstance(child_data["memory_block_ids"], (str, bytes))
                        else child_data["memory_block_ids"]
                    )
                    prompt_template = child_data.get("memory_prompt_template", "")

                    block_pipe = redis_client.client.pipeline()
                    for block_id in block_ids:
                        block_pipe.hgetall(f"{redis_client.BLOCK_PREFIX}{block_id}")
                    block_results = block_pipe.execute()

                    for block_data in block_results:
                        if block_data:
                            # Normalize block data: ensure 'value' is never None (use empty string instead)
                            if "value" not in block_data or block_data["value"] is None:
                                block_data["value"] = ""
                            blocks.append(PydanticBlock(**block_data))

                # Always create a Memory object (even if empty) - never None
                memory = PydanticMemory(blocks=blocks, prompt_template=prompt_template)

                child_data["memory"] = memory
                child_data.pop("memory_block_ids", None)
                child_data.pop("memory_prompt_template", None)

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
                return self._get_children_from_db(parent_ids, session, actor)

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
            logger.warning("Failed to reconstruct children from Redis cache: %s", e)
            return self._get_children_from_db(parent_ids, session, actor)

    def _get_children_from_db(self, parent_ids: List[str], session: Session, actor: PydanticClient) -> dict:
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
        children = AgentModel.list(
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

    def _get_children_from_redis(self, parent_id: str, actor: PydanticClient) -> Optional[List[PydanticAgentState]]:
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
            parent_data = redis_client.get_hash(parent_key)

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
                    child = self.get_agent_by_id(child_id, actor)
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
            logger.warning("Failed to get children from Redis for parent %s: %s", parent_id, e)
            return None

    def _cache_children_ids_for_parents(self, agent_states: List[PydanticAgentState]) -> None:
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
                    redis_client.client.hset(parent_key, "children_ids", json.dumps(children_ids))

                    # Maintain reverse mapping for cache invalidation
                    for child_id in children_ids:
                        reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                        redis_client.client.set(reverse_key, agent_state.id)
                        redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

            logger.debug(
                "Cached children_ids for %s parent agents",
                len([a for a in agent_states if a.children]),
            )
        except Exception as e:
            # Log but don't fail if caching fails
            logger.warning("Failed to cache children_ids for parent agents: %s", e)

    @enforce_types
    def list_agents(
        self,
        actor: PydanticClient,
        match_all_tags: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        parent_id: Optional[str] = None,
        user: Optional[PydanticUser] = None,
        **kwargs,
    ) -> List[PydanticAgentState]:
        """
        List agents that have the specified tags.
        By default, only returns top-level agents (parent_id is None) with their children populated.
        If parent_id is provided, only returns agents with that parent_id.

        When parent_id is provided, tries to use Redis cache via parent's children_ids first,
        then falls back to PostgreSQL if cache miss.
        """
        # Optimization: Use Redis cache for list_agents(parent_id=X)
        if parent_id is not None:
            cached_children = self._get_children_from_redis(parent_id, actor)
            if cached_children is not None:
                logger.debug("Redis cache HIT for children of parent %s", parent_id)
                return cached_children
            # Cache miss - fall through to PostgreSQL query
            logger.debug(
                "Redis cache MISS for children of parent %s, querying PostgreSQL",
                parent_id,
            )

        with self.session_maker() as session:
            # Get agents filtered by parent_id (None for top-level agents, or specific parent_id)
            # Actor triggers apply_access_predicate which filters by both organization_id and _created_by_id (client isolation)
            agents = AgentModel.list(
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
            agent_states = [agent.to_pydantic() for agent in agents]

            # If there are no agents, return early
            if not agent_states:
                return agent_states

            # Only populate children if we're listing top-level agents (parent_id is None)
            if parent_id is None:
                children_by_parent = self._reconstruct_children_from_cache(agent_states, session, actor)

                # Assign children to their parent agents
                for agent_state in agent_states:
                    agent_state.children = children_by_parent.get(agent_state.id, [])

                # Cache children_ids for future list_agents(parent_id=X) calls
                self._cache_children_ids_for_parents(agent_states)

            return agent_states

    @enforce_types
    def get_agent_by_id(self, agent_id: str, actor: PydanticClient) -> PydanticAgentState:
        """Fetch an agent by its ID (with Redis Hash caching and tool retrieval pipeline)."""
        # Try Redis cache first
        try:
            import json

            from mirix.database.redis_client import get_redis_client
            from mirix.log import get_logger
            from mirix.schemas.tool import Tool as PydanticTool

            logger = get_logger(__name__)
            redis_client = get_redis_client()

            if redis_client:
                redis_key = f"{redis_client.AGENT_PREFIX}{agent_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    logger.debug("Redis cache HIT for agent %s", agent_id)

                    # Deserialize JSON fields
                    if "message_ids" in cached_data:
                        cached_data["message_ids"] = (
                            json.loads(cached_data["message_ids"])
                            if isinstance(cached_data["message_ids"], str)
                            else cached_data["message_ids"]
                        )
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
                        tool_results = pipe.execute()

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

                    # Reconstruct memory from block IDs
                    from mirix.schemas.block import Block as PydanticBlock
                    from mirix.schemas.memory import Memory as PydanticMemory

                    blocks = []
                    prompt_template = ""

                    if "memory_block_ids" in cached_data and cached_data["memory_block_ids"]:
                        block_ids = (
                            json.loads(cached_data["memory_block_ids"])
                            if isinstance(cached_data["memory_block_ids"], str)
                            else cached_data["memory_block_ids"]
                        )
                        prompt_template = cached_data.get("memory_prompt_template", "")

                        # Use pipeline for efficient parallel block retrieval
                        pipe = redis_client.client.pipeline()
                        for block_id in block_ids:
                            pipe.hgetall(f"{redis_client.BLOCK_PREFIX}{block_id}")
                        block_results = pipe.execute()

                        # Reconstruct blocks
                        for block_data in block_results:
                            if block_data:
                                # Normalize block data: ensure 'value' is never None (use empty string instead)
                                if "value" not in block_data or block_data["value"] is None:
                                    block_data["value"] = ""
                                blocks.append(PydanticBlock(**block_data))

                        logger.debug(
                            "Reconstructed memory with %s blocks for agent %s",
                            len(blocks),
                            agent_id,
                        )

                    # Always create a Memory object (even if empty) - never None
                    memory = PydanticMemory(blocks=blocks, prompt_template=prompt_template)

                    cached_data["memory"] = memory
                    cached_data.pop("memory_block_ids", None)
                    cached_data.pop("memory_prompt_template", None)

                    agent_state = PydanticAgentState(**cached_data)

                    # SECURITY CHECK: Verify agent belongs to this client
                    # Prevents cross-client access via Redis cache
                    if agent_state.created_by_id != actor.id:
                        from sqlalchemy.exc import NoResultFound

                        raise NoResultFound(f"Agent {agent_id} not found or not accessible to client {actor.id}")

                    return agent_state  # Cache HIT (agent + tools + memory)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            from mirix.log import get_logger

            logger = get_logger(__name__)
            logger.warning("Redis cache read failed for agent %s: %s", agent_id, e)

        # Cache MISS or Redis unavailable - fetch from PostgreSQL with client filtering
        with self.session_maker() as session:
            # AgentModel.read calls apply_access_predicate, which now filters by client (organization_id + _created_by_id)
            # If agent doesn't belong to this client, read() will raise NoResultFound automatically
            agent = AgentModel.read(
                db_session=session,
                identifier=agent_id,
                actor=actor,  # Triggers client-level filtering via apply_access_predicate
            )
            pydantic_agent = agent.to_pydantic()

            # Populate Redis cache for next time
            try:
                if redis_client:
                    import json

                    from mirix.settings import settings

                    data = pydantic_agent.model_dump(mode="json")

                    # Serialize JSON fields for Hash storage
                    if "message_ids" in data and data["message_ids"]:
                        data["message_ids"] = json.dumps(data["message_ids"])
                    if "llm_config" in data and data["llm_config"]:
                        data["llm_config"] = json.dumps(data["llm_config"])
                    if "embedding_config" in data and data["embedding_config"]:
                        data["embedding_config"] = json.dumps(data["embedding_config"])
                    if "tool_rules" in data and data["tool_rules"]:
                        data["tool_rules"] = json.dumps(data["tool_rules"])
                    if "mcp_tools" in data and data["mcp_tools"]:
                        data["mcp_tools"] = json.dumps(data["mcp_tools"])

                    # model_dump(mode='json') already converts datetime to ISO format strings

                    # Cache tools separately and store tool_ids
                    if "tools" in data and data["tools"]:
                        tool_ids = [tool["id"] for tool in data["tools"]]
                        data["tool_ids"] = json.dumps(tool_ids)

                        for tool in data["tools"]:
                            tool_key = f"{redis_client.TOOL_PREFIX}{tool['id']}"
                            if "json_schema" in tool and tool["json_schema"]:
                                tool["json_schema"] = json.dumps(tool["json_schema"])
                            if "tags" in tool and tool["tags"]:
                                tool["tags"] = json.dumps(tool["tags"])
                            redis_client.set_hash(tool_key, tool, ttl=settings.redis_ttl_tools)

                    # Cache memory_block_ids for reconstruction
                    if "memory" in data and data["memory"]:
                        memory_obj = data["memory"]
                        if isinstance(memory_obj, dict) and "blocks" in memory_obj:
                            block_ids = [
                                block["id"] if isinstance(block, dict) else block.id for block in memory_obj["blocks"]
                            ]
                            data["memory_block_ids"] = json.dumps(block_ids)
                            data["memory_prompt_template"] = memory_obj.get("prompt_template", "")

                            # Maintain reverse mapping for cache invalidation
                            for block_id in block_ids:
                                reverse_key = f"{redis_client.BLOCK_PREFIX}{block_id}:agents"
                                redis_client.client.sadd(reverse_key, agent_id)
                                redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    # Cache children_ids for reconstruction (list_agents only)
                    if "children" in data and data["children"]:
                        children_ids = [
                            child["id"] if isinstance(child, dict) else child.id for child in data["children"]
                        ]
                        data["children_ids"] = json.dumps(children_ids)

                        # Maintain reverse mapping for cache invalidation
                        for child_id in children_ids:
                            reverse_key = f"{redis_client.AGENT_PREFIX}{child_id}:parent"
                            redis_client.client.set(reverse_key, agent_id)
                            redis_client.client.expire(reverse_key, settings.redis_ttl_agents)

                    # Remove relationship fields
                    data.pop("tools", None)
                    data.pop("memory", None)
                    data.pop("children", None)

                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_agents)
                    logger.debug("Populated Redis cache for agent %s with tools", agent_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for agent %s: %s", agent_id, e)

            return pydantic_agent

    @enforce_types
    def get_agent_by_name(self, agent_name: str, actor: PydanticClient) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, name=agent_name, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def delete_agent(self, agent_id: str, actor: PydanticClient) -> None:
        """
        Deletes an agent and its associated relationships.
        Ensures proper permission checks and cascades where applicable.

        Args:
            agent_id: ID of the agent to be deleted.
            actor: User performing the action.

        Raises:
            NoResultFound: If agent doesn't exist
        """
        with self.session_maker() as session:
            # Retrieve the agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Track parent_id for cache invalidation
            parent_id = agent.parent_id

            # Remove from Redis cache before hard delete
            try:
                from mirix.database.redis_client import get_redis_client
                from mirix.log import get_logger

                logger = get_logger(__name__)
                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.AGENT_PREFIX}{agent_id}"
                    redis_client.delete(redis_key)
                    logger.debug("Removed agent %s from Redis cache", agent_id)
            except Exception as e:
                from mirix.log import get_logger

                logger = get_logger(__name__)
                logger.warning("Failed to remove agent %s from Redis cache: %s", agent_id, e)

            agent.hard_delete(session)

            # Invalidate parent cache if this was a child agent
            if parent_id:
                self._invalidate_parent_cache_for_child(agent_id, parent_id)

    # ======================================================================================================================
    # In Context Messages Management
    # ======================================================================================================================
    # TODO: There are several assumptions here that are not explicitly checked
    # TODO: 1) These message ids are valid
    # TODO: 2) These messages are ordered from oldest to newest
    # TODO: This can be fixed by having an actual relationship in the ORM for message_ids
    # TODO: This can also be made more efficient, instead of getting, setting, we can do it all in one db session for one query.
    # @enforce_types
    # def get_in_context_messages(
    #     self, agent_id: str, actor: PydanticClient
    # ) -> List[PydanticMessage]:
    #     message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
    #     messages = self.message_manager.get_messages_by_ids(
    #         message_ids=message_ids, actor=actor
    #     )
    #     messages = [messages[0]] + [
    #         message for message in messages[1:] if message.user_id == actor.id
    #     ]
    #     return messages
    @enforce_types
    def get_in_context_messages(
        self,
        agent_state: PydanticAgentState,
        actor: PydanticClient,
        user: Optional[PydanticUser] = None,
    ) -> List[PydanticMessage]:
        message_ids = agent_state.message_ids
        messages = self.message_manager.get_messages_by_ids(message_ids=message_ids, actor=actor)
        # Handle empty message list (e.g., after deletion)
        if not messages:
            return []

        # Keep first message (system message) and filter rest by user_id
        if user:
            messages = [messages[0]] + [message for message in messages[1:] if message.user_id == user.id]
        return messages

    @enforce_types
    def get_system_message(self, agent_id: str, actor: PydanticClient) -> PydanticMessage:
        agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)
        message_ids = agent_state.message_ids

        # Handle empty message_ids (e.g., after deletion)
        if not message_ids:
            return None

        return self.message_manager.get_message_by_id(message_id=message_ids[0], actor=actor)

    @enforce_types
    def rebuild_system_prompt(
        self, agent_id: str, system_prompt: str, actor: PydanticClient, force=False
    ) -> PydanticAgentState:
        """Rebuld the system prompt, put the system_prompt at the first position in the list of messages."""

        agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)
        # Swap the system message out (only if there is a diff)
        message = PydanticMessage.dict_to_message(
            agent_id=agent_id,
            model=agent_state.llm_config.model,
            openai_message_dict={"role": "system", "content": system_prompt},
        )
        message = self.message_manager.create_message(message, actor=actor)
        message_ids = [message.id] + agent_state.message_ids[1:]  # swap index 0 (system)
        return self.set_in_context_messages(agent_id=agent_id, message_ids=message_ids, actor=actor)

    @enforce_types
    def set_in_context_messages(
        self, agent_id: str, message_ids: List[str], actor: PydanticClient
    ) -> PydanticAgentState:
        return self.update_agent(
            agent_id=agent_id,
            agent_update=UpdateAgent(message_ids=message_ids),
            actor=actor,
        )

    @enforce_types
    def trim_older_in_context_messages(
        self,
        num: int,
        agent_id: str,
        actor: PydanticClient,
        user_id: Optional[str] = None,
    ) -> PydanticAgentState:
        """
        Trim older messages from the in-context message list, keeping `num` most recent messages
        for the specified user. Messages from other users are preserved.

        Args:
            num: Number of most recent user messages to keep.
            agent_id: The agent ID.
            actor: The Client performing the operation.
            user_id: The user whose messages to trim. If None, trims all non-system messages.
        """
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        system_message_id = message_ids[0]
        message_ids = message_ids[1:]

        message_id_indices_belonging_to_user = [
            idx
            for idx, message_id in enumerate(message_ids)
            if self.message_manager.get_message_by_id(message_id=message_id, actor=actor).user_id == user_id
        ]
        message_ids_belonging_to_user = [message_ids[idx] for idx in message_id_indices_belonging_to_user]
        message_ids_to_keep = [message_ids[idx] for idx in message_id_indices_belonging_to_user[num - 1 :]]

        message_ids_belonging_to_user = set(message_ids_belonging_to_user)
        message_ids_to_keep = set(message_ids_to_keep)

        # new_messages = [message_ids[0]] + message_ids[num:]  # 0 is system message
        new_messages = [system_message_id] + [
            msg_id
            for msg_id in message_ids
            if (msg_id not in message_ids_belonging_to_user or msg_id in message_ids_to_keep)
        ]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=new_messages, actor=actor)

    @enforce_types
    def trim_all_in_context_messages_except_system(
        self, agent_id: str, actor: PydanticClient, user_id: Optional[str] = None
    ) -> PydanticAgentState:
        """
        Remove all messages except the system message for a specific user.
        Messages from other users are preserved.

        Args:
            agent_id: The agent ID.
            actor: The Client performing the operation.
            user_id: The user whose messages to remove. If None, removes all non-system messages.
        """
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        system_message_id = message_ids[0]  # 0 is system message

        # Keep system message and only filter out messages belonging to the specified user
        new_message_ids = [system_message_id]
        for message_id in message_ids[1:]:  # Skip system message
            message = self.message_manager.get_message_by_id(message_id=message_id, actor=actor)
            if message.user_id != user_id:
                new_message_ids.append(message_id)

        return self.set_in_context_messages(agent_id=agent_id, message_ids=new_message_ids, actor=actor)

    @enforce_types
    def prepend_to_in_context_messages(
        self,
        messages: List[PydanticMessage],
        agent_id: str,
        actor: PydanticClient,
        user_id: Optional[str] = None,
    ) -> PydanticAgentState:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        new_messages = self.message_manager.create_many_messages(messages, actor=actor, user_id=user_id)
        message_ids = [message_ids[0]] + [m.id for m in new_messages] + message_ids[1:]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=message_ids, actor=actor)

    @enforce_types
    def append_to_in_context_messages(
        self,
        messages: List[PydanticMessage],
        agent_id: str,
        actor: PydanticClient,
        user_id: Optional[str] = None,
    ) -> PydanticAgentState:
        messages = self.message_manager.create_many_messages(messages, actor=actor, user_id=user_id)
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids or []
        message_ids += [m.id for m in messages]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=message_ids, actor=actor)

    @enforce_types
    def reset_messages(
        self,
        agent_id: str,
        actor: PydanticClient,
        user_id: Optional[str] = None,
        add_default_initial_messages: bool = False,
    ) -> PydanticAgentState:
        """
        Removes messages belonging to the specified user from the agent's conversation history.
        Preserves system messages and messages from other users.

        This action is destructive and cannot be undone once committed.

        Args:
            agent_id (str): The ID of the agent whose messages will be reset.
            actor (PydanticClient): The Client performing this action.
            user_id (str): The user whose messages will be removed. If None, removes all non-system messages.
            add_default_initial_messages: If true, adds the default initial messages after resetting.

        Returns:
            PydanticAgentState: The updated agent state with user's messages removed.
        """
        with self.session_maker() as session:
            # Retrieve the existing agent (will raise NoResultFound if invalid)
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Get current messages to filter
            current_messages = agent.messages

            # Filter out messages belonging to the specific user, but keep:
            # 1. System messages (role='system') - always keep
            # 2. Messages from other users (user_id != specified user_id)
            messages_to_keep = []
            messages_to_remove = []

            for message in current_messages:
                if message.role == "system":
                    # Always keep system messages
                    messages_to_keep.append(message)
                elif user_id is None or message.user_id == user_id:
                    # Remove this user's messages (or all if user_id is None)
                    messages_to_remove.append(message)
                else:
                    # Keep messages from other users
                    messages_to_keep.append(message)

            # Update the agent's messages relationship to only keep filtered messages
            agent.messages = messages_to_keep

            # Update message_ids to reflect the remaining messages
            # Keep the order based on created_at timestamp
            agent.message_ids = [msg.id for msg in messages_to_keep]

            # Commit the update
            agent.update(db_session=session, actor=actor)

            agent_state = agent.to_pydantic()

        if add_default_initial_messages:
            return self.append_initial_message_sequence_to_in_context_messages(actor, agent_state, user_id=user_id)
        else:
            # We still want to always have a system message
            init_messages = initialize_message_sequence(
                agent_state=agent_state,
                memory_edit_timestamp=get_utc_time(),
                include_initial_boot_message=True,
            )
            system_message = PydanticMessage.dict_to_message(
                agent_id=agent_state.id,
                user_id=agent_state.created_by_id,
                model=agent_state.llm_config.model,
                openai_message_dict=init_messages[0],
            )
            return self.append_to_in_context_messages([system_message], agent_id=agent_state.id, actor=actor)

    # ======================================================================================================================
    # Tool Management
    # ======================================================================================================================
    @enforce_types
    def attach_tool(self, agent_id: str, tool_id: str, actor: PydanticClient) -> PydanticAgentState:
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
        with self.session_maker() as session:
            # Verify the agent exists and user has permission to access it
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Use the _process_relationship helper to attach the tool
            _process_relationship(
                session=session,
                agent=agent,
                relationship_name="tools",
                model_class=ToolModel,
                item_ids=[tool_id],
                allow_partial=False,  # Ensure the tool exists
                replace=False,  # Extend the existing tools
            )

            # Commit and refresh the agent
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def detach_tool(self, agent_id: str, tool_id: str, actor: PydanticClient) -> PydanticAgentState:
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
        with self.session_maker() as session:
            # Verify the agent exists and user has permission to access it
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Filter out the tool to be detached
            remaining_tools = [tool for tool in agent.tools if tool.id != tool_id]

            if len(remaining_tools) == len(agent.tools):  # Tool ID was not in the relationship
                logger.warning(
                    f"Attempted to remove unattached tool id={tool_id} from agent id={agent_id} by actor={actor}"
                )

            # Update the tools relationship
            agent.tools = remaining_tools

            # Commit and refresh the agent
            agent.update(session, actor=actor)
            return agent.to_pydantic()
