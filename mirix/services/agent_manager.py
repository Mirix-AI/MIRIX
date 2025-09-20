from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import Select, func, literal, select, union_all

from mirix.constants import (
    CORE_MEMORY_TOOLS, BASE_TOOLS, MAX_EMBEDDING_DIM,
    EPISODIC_MEMORY_TOOLS, PROCEDURAL_MEMORY_TOOLS, SEMANTIC_MEMORY_TOOLS,
    RESOURCE_MEMORY_TOOLS, KNOWLEDGE_VAULT_TOOLS, META_MEMORY_TOOLS, UNIVERSAL_MEMORY_TOOLS, CHAT_AGENT_TOOLS, SEARCH_MEMORY_TOOLS, EXTRAS_TOOLS, MCP_TOOLS
)
from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.orm import Agent as AgentModel
from mirix.orm import Block as BlockModel
from mirix.orm import Tool as ToolModel
from mirix.orm.errors import NoResultFound
from mirix.orm.sandbox_config import AgentEnvironmentVariable as AgentEnvironmentVariableModel
from mirix.orm.sqlite_functions import adapt_array
from mirix.orm.enums import ToolType
from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.agent import AgentType, CreateAgent, UpdateAgent
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.message import MessageCreate
from mirix.schemas.tool_rule import ToolRule as PydanticToolRule
from mirix.schemas.user import User as PydanticUser
from mirix.services.block_manager import BlockManager
from mirix.services.helpers.agent_manager_helper import (
    _process_relationship,
    _process_tags,
    check_supports_structured_output,
    compile_system_message,
    derive_system_message,
    initialize_message_sequence,
    package_initial_message_sequence,
)
from mirix.services.message_manager import MessageManager
from mirix.services.tool_manager import ToolManager
from mirix.settings import settings
from mirix.utils import enforce_types, get_utc_time, united_diff

logger = get_logger(__name__)


# Agent Manager Class
class AgentManager:
    """Manager class to handle business logic related to Agents."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context
        self.block_manager = BlockManager()
        self.tool_manager = ToolManager()
        self.message_manager = MessageManager()

    # ======================================================================================================================
    # Basic CRUD operations
    # ======================================================================================================================
    @enforce_types
    def create_agent(
        self,
        agent_create: CreateAgent,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        system = derive_system_message(agent_type=agent_create.agent_type, system=agent_create.system)

        if not agent_create.llm_config or not agent_create.embedding_config:
            raise ValueError("llm_config and embedding_config are required")

        # Check tool rules are valid
        if agent_create.tool_rules:
            check_supports_structured_output(model=agent_create.llm_config.model, tool_rules=agent_create.tool_rules)

        # create blocks (note: cannot be linked into the agent_id is created)
        block_ids = list(agent_create.block_ids or [])  # Create a local copy to avoid modifying the original
        if agent_create.memory_blocks:
            for create_block in agent_create.memory_blocks:
                block = self.block_manager.create_or_update_block(PydanticBlock(**create_block.model_dump()), actor=actor)
                block_ids.append(block.id)

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
        if agent_create.agent_type == AgentType.knowledge_vault_agent:
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
                print(f"Tool {tool_name} not found")
            
        # Remove duplicates
        tool_ids = list(set(tool_ids))

        # Create the agent
        agent_state = self._create_agent(
            name=agent_create.name,
            system=system,
            agent_type=agent_create.agent_type,
            llm_config=agent_create.llm_config,
            embedding_config=agent_create.embedding_config,
            block_ids=block_ids,
            tool_ids=tool_ids,
            tags=agent_create.tags or [],
            description=agent_create.description,
            metadata_=agent_create.metadata_,
            tool_rules=agent_create.tool_rules,
            actor=actor,
        )

        # If there are provided environment variables, add them in
        if agent_create.tool_exec_environment_variables:
            agent_state = self._set_environment_variables(
                agent_id=agent_state.id,
                env_vars=agent_create.tool_exec_environment_variables,
                actor=actor,
            )

        return self.append_initial_message_sequence_to_in_context_messages(actor, agent_state, agent_create.initial_message_sequence)

    def update_agent_tools_and_system_prompts(
        self,
        agent_id: str,
        actor: PydanticUser,
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
        if agent_state.agent_type == AgentType.knowledge_vault_agent:
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
        tool_names_to_remove = [tool_name for tool_name in existing_tool_names 
                               if tool_name not in tool_names and tool_name not in mcp_tool_names]

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
            self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(tool_ids=tool_ids), actor=actor)

    @enforce_types
    def _generate_initial_message_sequence(
        self, actor: PydanticUser, agent_state: PydanticAgentState, supplied_initial_message_sequence: Optional[List[MessageCreate]] = None
    ) -> List[PydanticMessage]:
        init_messages = initialize_message_sequence(
            agent_state=agent_state, memory_edit_timestamp=get_utc_time(), include_initial_boot_message=True
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
                package_initial_message_sequence(agent_state.id, supplied_initial_message_sequence, agent_state.llm_config.model, actor)
            )
        else:
            init_messages = [
                PydanticMessage.dict_to_message(agent_id=agent_state.id, model=agent_state.llm_config.model, openai_message_dict=msg)
                for msg in init_messages
            ]

        return init_messages

    @enforce_types
    def append_initial_message_sequence_to_in_context_messages(
        self, actor: PydanticUser, agent_state: PydanticAgentState, initial_message_sequence: Optional[List[MessageCreate]] = None
    ) -> PydanticAgentState:
        init_messages = self._generate_initial_message_sequence(actor, agent_state, initial_message_sequence)
        return self.append_to_in_context_messages(init_messages, agent_id=agent_state.id, actor=actor)

    @enforce_types
    def _create_agent(
        self,
        actor: PydanticUser,
        name: str,
        system: str,
        agent_type: AgentType,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        block_ids: List[str],
        tool_ids: List[str],
        tags: List[str],
        description: Optional[str] = None,
        metadata_: Optional[Dict] = None,
        tool_rules: Optional[List[PydanticToolRule]] = None,
    ) -> PydanticAgentState:
        """Create a new agent."""
        with self.session_maker() as session:
            # Prepare the agent data
            data = {
                "name": name,
                "system": system,
                'topic': '', # TODO: make this nullable
                "agent_type": agent_type,
                "llm_config": llm_config,
                "embedding_config": embedding_config,
                "organization_id": actor.organization_id,
                "description": description,
                "metadata_": metadata_,
                "tool_rules": tool_rules,
            }

            # Create the new agent using SqlalchemyBase.create
            new_agent = AgentModel(**data)
            _process_relationship(session, new_agent, "tools", ToolModel, tool_ids, replace=True)
            _process_relationship(session, new_agent, "core_memory", BlockModel, block_ids, replace=True)
            _process_tags(new_agent, tags, replace=True)
            new_agent.create(session, actor=actor)

            # Convert to PydanticAgentState and return
            return new_agent.to_pydantic()

    @enforce_types
    def update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: PydanticUser) -> PydanticAgentState:
        agent_state = self._update_agent(agent_id=agent_id, agent_update=agent_update, actor=actor)

        # If there are provided environment variables, add them in
        if agent_update.tool_exec_environment_variables:
            agent_state = self._set_environment_variables(
                agent_id=agent_state.id,
                env_vars=agent_update.tool_exec_environment_variables,
                actor=actor,
            )

        # Rebuild the system prompt if it's different
        if agent_update.system and agent_update.system != agent_state.system:
            agent_state = self.rebuild_system_prompt(agent_id=agent_state.id, actor=actor, force=True, update_timestamp=False)

        return agent_state

    @enforce_types
    def update_llm_config(self, agent_id: str, llm_config: LLMConfig, actor: PydanticUser) -> PydanticAgentState:
        return self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(llm_config=llm_config), actor=actor)

    @enforce_types
    def update_system_prompt(self, agent_id: str, system_prompt: str, actor: PydanticUser) -> PydanticAgentState:
        agent_state = self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(system=system_prompt), actor=actor)
        # Rebuild the system prompt if it's different
        agent_state = self.rebuild_system_prompt(agent_id=agent_state.id, system_prompt=system_prompt, actor=actor, force=True)
        return agent_state

    @enforce_types
    def update_mcp_tools(self, agent_id: str, mcp_tools: List[str], actor: PydanticUser, tool_ids: List[str]) -> PydanticAgentState:
        """Update the MCP tools connected to an agent."""
        return self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(mcp_tools=mcp_tools, tool_ids=tool_ids), actor=actor)

    @enforce_types
    def add_mcp_tool(self, agent_id: str, mcp_tool_name: str, tool_ids: List[str], actor: PydanticUser) -> PydanticAgentState:
        """Add a single MCP tool to an agent."""
        # First get the current agent state
        agent_state = self.get_agent_by_id(agent_id=agent_id, actor=actor)
        current_mcp_tools = agent_state.mcp_tools or []
        
        # Add the new MCP tool if not already present
        if mcp_tool_name not in current_mcp_tools:
            current_mcp_tools.append(mcp_tool_name)
            return self.update_mcp_tools(agent_id=agent_id, mcp_tools=current_mcp_tools, actor=actor, tool_ids=tool_ids)
        
        return agent_state

    @enforce_types
    def _update_agent(self, agent_id: str, agent_update: UpdateAgent, actor: PydanticUser) -> PydanticAgentState:
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

            # Update scalar fields directly
            scalar_fields = {"name", "system", "topic", "llm_config", "embedding_config", "message_ids", "tool_rules", "description", "metadata_", "mcp_tools"}
            for field in scalar_fields:
                value = getattr(agent_update, field, None)
                if value is not None:
                    setattr(agent, field, value)

            # Update relationships using _process_relationship and _process_tags
            if agent_update.tool_ids is not None:
                _process_relationship(session, agent, "tools", ToolModel, agent_update.tool_ids, replace=True)
            if agent_update.block_ids is not None:
                _process_relationship(session, agent, "core_memory", BlockModel, agent_update.block_ids, replace=True)
            if agent_update.tags is not None:
                _process_tags(agent, agent_update.tags, replace=True)

            # Commit and refresh the agent
            agent.update(session, actor=actor)

            # Convert to PydanticAgentState and return
            return agent.to_pydantic()

    @enforce_types
    def list_agents(
        self,
        actor: PydanticUser,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        query_text: Optional[str] = None,
        **kwargs,
    ) -> List[PydanticAgentState]:
        """
        List agents that have the specified tags.
        """
        with self.session_maker() as session:
            agents = AgentModel.list(
                db_session=session,
                tags=tags,
                match_all_tags=match_all_tags,
                cursor=cursor,
                limit=limit,
                organization_id=actor.organization_id if actor else None,
                query_text=query_text,
                **kwargs,
            )

            return [agent.to_pydantic() for agent in agents]

    @enforce_types
    def get_agent_by_id(self, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def get_agent_by_name(self, agent_name: str, actor: PydanticUser) -> PydanticAgentState:
        """Fetch an agent by its ID."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, name=agent_name, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def delete_agent(self, agent_id: str, actor: PydanticUser) -> None:
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
            agent.hard_delete(session)

    # ======================================================================================================================
    # Per Agent Environment Variable Management
    # ======================================================================================================================
    @enforce_types
    def _set_environment_variables(
        self,
        agent_id: str,
        env_vars: Dict[str, str],
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """
        Adds or replaces the environment variables for the specified agent.

        Args:
            agent_id: The agent id.
            env_vars: A dictionary of environment variable key-value pairs.
            actor: The user performing the action.

        Returns:
            PydanticAgentState: The updated agent as a Pydantic model.
        """
        with self.session_maker() as session:
            # Retrieve the agent
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Fetch existing environment variables as a dictionary
            existing_vars = {var.key: var for var in agent.tool_exec_environment_variables}

            # Update or create environment variables
            updated_vars = []
            for key, value in env_vars.items():
                if key in existing_vars:
                    # Update existing variable
                    existing_vars[key].value = value
                    updated_vars.append(existing_vars[key])
                else:
                    # Create new variable
                    updated_vars.append(
                        AgentEnvironmentVariableModel(
                            key=key,
                            value=value,
                            agent_id=agent_id,
                            organization_id=actor.organization_id,
                        )
                    )

            # Remove stale variables
            stale_keys = set(existing_vars) - set(env_vars)
            agent.tool_exec_environment_variables = [var for var in updated_vars if var.key not in stale_keys]

            # Update the agent in the database
            agent.update(session, actor=actor)

            # Return the updated agent state
            return agent.to_pydantic()

    # ======================================================================================================================
    # In Context Messages Management
    # ======================================================================================================================
    # TODO: There are several assumptions here that are not explicitly checked
    # TODO: 1) These message ids are valid
    # TODO: 2) These messages are ordered from oldest to newest
    # TODO: This can be fixed by having an actual relationship in the ORM for message_ids
    # TODO: This can also be made more efficient, instead of getting, setting, we can do it all in one db session for one query.
    @enforce_types
    def get_in_context_messages(self, agent_id: str, actor: PydanticUser) -> List[PydanticMessage]:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        messages = self.message_manager.get_messages_by_ids(message_ids=message_ids, actor=actor)
        messages = [messages[0]] + [message for message in messages[1:] if message.user_id == actor.id]
        return messages

    @enforce_types
    def get_system_message(self, agent_id: str, actor: PydanticUser) -> PydanticMessage:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        return self.message_manager.get_message_by_id(message_id=message_ids[0], actor=actor)

    @enforce_types
    def rebuild_system_prompt(self, agent_id: str, system_prompt: str, actor: PydanticUser, force=False) -> PydanticAgentState:
        
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
    def update_topic(self, agent_id: str, topic: str, actor: PydanticUser) -> PydanticAgentState:
        return self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(topic=topic), actor=actor)

    @enforce_types
    def set_in_context_messages(self, agent_id: str, message_ids: List[str], actor: PydanticUser) -> PydanticAgentState:
        return self.update_agent(agent_id=agent_id, agent_update=UpdateAgent(message_ids=message_ids), actor=actor)

    @enforce_types
    def trim_older_in_context_messages(self, num: int, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        system_message_id = message_ids[0]
        message_ids = message_ids[1:]

        message_id_indices_belonging_to_actor = [idx for idx, message_id in enumerate(message_ids) if self.message_manager.get_message_by_id(message_id=message_id, actor=actor).user_id == actor.id]
        message_ids_belonging_to_actor = [message_ids[idx] for idx in message_id_indices_belonging_to_actor]
        message_ids_to_keep = [message_ids[idx] for idx in message_id_indices_belonging_to_actor[num-1:]]

        message_ids_belonging_to_actor = set(message_ids_belonging_to_actor)
        message_ids_to_keep = set(message_ids_to_keep)

        # new_messages = [message_ids[0]] + message_ids[num:]  # 0 is system message
        new_messages = [system_message_id] + [msg_id for msg_id in message_ids if (msg_id not in message_ids_belonging_to_actor or msg_id in message_ids_to_keep)]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=new_messages, actor=actor)

    @enforce_types
    def trim_all_in_context_messages_except_system(self, agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        system_message_id = message_ids[0]  # 0 is system message
        
        # Keep system message and only filter out messages belonging to the current actor
        new_message_ids = [system_message_id]
        for message_id in message_ids[1:]:  # Skip system message
            message = self.message_manager.get_message_by_id(message_id=message_id, actor=actor)
            if message.user_id != actor.id:
                new_message_ids.append(message_id)
        
        return self.set_in_context_messages(agent_id=agent_id, message_ids=new_message_ids, actor=actor)

    @enforce_types
    def prepend_to_in_context_messages(self, messages: List[PydanticMessage], agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids
        new_messages = self.message_manager.create_many_messages(messages, actor=actor)
        message_ids = [message_ids[0]] + [m.id for m in new_messages] + message_ids[1:]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=message_ids, actor=actor)

    @enforce_types
    def append_to_in_context_messages(self, messages: List[PydanticMessage], agent_id: str, actor: PydanticUser) -> PydanticAgentState:
        messages = self.message_manager.create_many_messages(messages, actor=actor)
        message_ids = self.get_agent_by_id(agent_id=agent_id, actor=actor).message_ids or []
        message_ids += [m.id for m in messages]
        return self.set_in_context_messages(agent_id=agent_id, message_ids=message_ids, actor=actor)

    @enforce_types
    def reset_messages(self, agent_id: str, actor: PydanticUser, add_default_initial_messages: bool = False) -> PydanticAgentState:
        """
        Removes messages belonging to the specified actor from the agent's conversation history.
        Preserves system messages and messages from other actors.

        This action is destructive and cannot be undone once committed.

        Args:
            add_default_initial_messages: If true, adds the default initial messages after resetting.
            agent_id (str): The ID of the agent whose messages will be reset.
            actor (PydanticUser): The user performing this action - only their messages will be removed.

        Returns:
            PydanticAgentState: The updated agent state with actor's messages removed.
        """
        with self.session_maker() as session:
            # Retrieve the existing agent (will raise NoResultFound if invalid)
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)

            # Get current messages to filter
            current_messages = agent.messages
            
            # Filter out messages belonging to the specific actor, but keep:
            # 1. System messages (role='system')
            # 2. Messages from other actors (user_id != actor.id)
            messages_to_keep = []
            messages_to_remove = []
            
            for message in current_messages:
                if message.role == 'system' or message.user_id == actor.id:
                    messages_to_remove.append(message)
                else:
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
            return self.append_initial_message_sequence_to_in_context_messages(actor, agent_state)
        else:
            # We still want to always have a system message
            init_messages = initialize_message_sequence(
                agent_state=agent_state, memory_edit_timestamp=get_utc_time(), include_initial_boot_message=True
            )
            system_message = PydanticMessage.dict_to_message(
                agent_id=agent_state.id,
                user_id=agent_state.created_by_id,
                model=agent_state.llm_config.model,
                openai_message_dict=init_messages[0],
            )
            return self.append_to_in_context_messages([system_message], agent_id=agent_state.id, actor=actor)

    # ======================================================================================================================
    # Block management
    # ======================================================================================================================
    @enforce_types
    def get_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        actor: PydanticUser,
    ) -> PydanticBlock:
        """Gets a block attached to an agent by its label."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            for block in agent.core_memory:
                if block.label == block_label:
                    return block.to_pydantic()
            raise NoResultFound(f"No block with label '{block_label}' found for agent '{agent_id}'")

    @enforce_types
    def update_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        new_block_id: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Updates which block is assigned to a specific label for an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            new_block = BlockModel.read(db_session=session, identifier=new_block_id, actor=actor)

            if new_block.label != block_label:
                raise ValueError(f"New block label '{new_block.label}' doesn't match required label '{block_label}'")

            # Remove old block with this label if it exists
            agent.core_memory = [b for b in agent.core_memory if b.label != block_label]

            # Add new block
            agent.core_memory.append(new_block)
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def attach_block(self, agent_id: str, block_id: str, actor: PydanticUser) -> PydanticAgentState:
        """Attaches a block to an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            block = BlockModel.read(db_session=session, identifier=block_id, actor=actor)

            agent.core_memory.append(block)
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def detach_block(
        self,
        agent_id: str,
        block_id: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a block from an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            original_length = len(agent.core_memory)

            agent.core_memory = [b for b in agent.core_memory if b.id != block_id]

            if len(agent.core_memory) == original_length:
                raise NoResultFound(f"No block with id '{block_id}' found for agent '{agent_id}' with actor id: '{actor.id}'")

            agent.update(session, actor=actor)
            return agent.to_pydantic()

    @enforce_types
    def detach_block_with_label(
        self,
        agent_id: str,
        block_label: str,
        actor: PydanticUser,
    ) -> PydanticAgentState:
        """Detaches a block with the specified label from an agent."""
        with self.session_maker() as session:
            agent = AgentModel.read(db_session=session, identifier=agent_id, actor=actor)
            original_length = len(agent.core_memory)

            agent.core_memory = [b for b in agent.core_memory if b.label != block_label]

            if len(agent.core_memory) == original_length:
                raise NoResultFound(f"No block with label '{block_label}' found for agent '{agent_id}' with actor id: '{actor.id}'")

            agent.update(session, actor=actor)
            return agent.to_pydantic()

    # ======================================================================================================================
    # Tool Management
    # ======================================================================================================================
    @enforce_types
    def attach_tool(self, agent_id: str, tool_id: str, actor: PydanticUser) -> PydanticAgentState:
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
    def detach_tool(self, agent_id: str, tool_id: str, actor: PydanticUser) -> PydanticAgentState:
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
                logger.warning(f"Attempted to remove unattached tool id={tool_id} from agent id={agent_id} by actor={actor}")

            # Update the tools relationship
            agent.tools = remaining_tools

            # Commit and refresh the agent
            agent.update(session, actor=actor)
            return agent.to_pydantic()

    # ======================================================================================================================
    # Tag Management
    # ======================================================================================================================
    @enforce_types
    def list_tags(
        self, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50, query_text: Optional[str] = None
    ) -> List[str]:
        """
        Get all tags a user has created, ordered alphabetically.

        Args:
            actor: User performing the action.
            cursor: Cursor for pagination.
            limit: Maximum number of tags to return.
            query_text: Query text to filter tags by.

        Returns:
            List[str]: List of all tags.
        """
        with self.session_maker() as session:
            query = (
                session.query(AgentsTags.tag)
                .join(AgentModel, AgentModel.id == AgentsTags.agent_id)
                .filter(AgentModel.organization_id == actor.organization_id)
                .distinct()
            )

            if query_text:
                query = query.filter(AgentsTags.tag.ilike(f"%{query_text}%"))

            if cursor:
                query = query.filter(AgentsTags.tag > cursor)

            query = query.order_by(AgentsTags.tag).limit(limit)
            results = [tag[0] for tag in query.all()]
            return results
