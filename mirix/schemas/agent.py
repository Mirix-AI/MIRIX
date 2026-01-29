from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from mirix.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from mirix.helpers import ToolRulesSolver
from mirix.schemas.block import CreateBlock
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import Memory
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_base import OrmMetadataBase
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import ToolRule

# Removed create_random_username import - server generates names if not provided


class AgentType(str, Enum):
    """
    Enum to represent the type of agent.
    """

    coder_agent = "coder_agent"
    chat_agent = "chat_agent"
    reflexion_agent = "reflexion_agent"
    background_agent = "background_agent"
    episodic_memory_agent = "episodic_memory_agent"
    procedural_memory_agent = "procedural_memory_agent"
    resource_memory_agent = "resource_memory_agent"
    knowledge_vault_memory_agent = "knowledge_vault_memory_agent"
    meta_memory_agent = "meta_memory_agent"
    semantic_memory_agent = "semantic_memory_agent"
    core_memory_agent = "core_memory_agent"


class AgentState(OrmMetadataBase, validate_assignment=True):
    """
    Representation of an agent's state. This is the state of the agent at a given time, and is persisted in the DB backend. The state has all the information needed to recreate a persisted agent.

    Parameters:
        id (str): The unique identifier of the agent.
        name (str): The name of the agent (must be unique to the user).
        created_at (datetime): The datetime the agent was created.
        memory (Memory): The in-context memory of the agent.
        tools (List[str]): The tools used by the agent. This includes any memory editing functions specified in `memory`.
        system (str): The system prompt used by the agent.
        llm_config (LLMConfig): The LLM configuration used by the agent.
        embedding_config (EmbeddingConfig): The embedding configuration used by the agent.

    """

    __id_prefix__ = "agent"

    # NOTE: this is what is returned to the client and also what is used to initialize `Agent`
    id: str = Field(..., description="The id of the agent. Assigned by the database.")
    name: str = Field(..., description="The name of the agent.")
    # tool rules
    tool_rules: Optional[List[ToolRule]] = Field(default=None, description="The list of tool rules.")

    # in-context memory
    message_ids: Optional[List[str]] = Field(
        default=None,
        description="The ids of the messages in the agent's in-context memory.",
    )

    # system prompt
    system: str = Field(..., description="The system prompt used by the agent.")

    # agent configuration
    agent_type: AgentType = Field(..., description="The type of agent.")

    # llm information
    llm_config: LLMConfig = Field(..., description="The LLM configuration used by the agent.")
    embedding_config: EmbeddingConfig = Field(..., description="The embedding configuration used by the agent.")

    # This is an object representing the in-process state of a running `Agent`
    # Field in this object can be theoretically edited by tools, and will be persisted by the ORM
    organization_id: Optional[str] = Field(
        None,
        description="The unique identifier of the organization associated with the agent.",
    )

    description: Optional[str] = Field(None, description="The description of the agent.")
    parent_id: Optional[str] = Field(None, description="The parent agent ID (for sub-agents in a meta-agent).")
    children: Optional[List["AgentState"]] = Field(
        default=None, description="Child agents (sub-agents) if this is a parent agent."
    )

    memory: Memory = Field(..., description="The in-context memory of the agent.")
    tools: List[Tool] = Field(..., description="The tools used by the agent.")
    mcp_tools: Optional[List[str]] = Field(
        default_factory=list,
        description="List of connected MCP server names (e.g., ['gmail-native'])",
    )

    def is_type(self, *agent_types: "AgentType") -> bool:
        """Check if agent matches any of the given types (by enum or name fallback).

        Args:
            *agent_types: One or more AgentType enum values to check against.

        Returns:
            True if the agent matches any of the provided types.

        Example:
            agent_state.is_type(AgentType.core_memory_agent)
            agent_state.is_type(AgentType.episodic_memory_agent, AgentType.reflexion_agent)
        """
        for at in agent_types:
            # Direct enum comparison (works for both enum and string values due to str, Enum)
            if self.agent_type == at or at.value in self.name:
                return True
        return False


class CreateAgent(BaseModel, validate_assignment=True):  #
    # all optional as server can generate defaults
    name: Optional[str] = Field(
        None,
        description="The name of the agent. If not provided, server will generate one.",
    )

    # memory creation
    memory_blocks: Optional[List[CreateBlock]] = Field(
        None,
        description="The blocks to create in the agent's in-context memory.",
    )
    # TODO: This is a legacy field and should be removed ASAP to force `tool_ids` usage
    tools: Optional[List[str]] = Field(None, description="The tools used by the agent.")
    tool_ids: Optional[List[str]] = Field(None, description="The ids of the tools used by the agent.")
    tool_rules: Optional[List[ToolRule]] = Field(None, description="The tool rules governing the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    agent_type: AgentType = Field(default_factory=lambda: AgentType.chat_agent, description="The type of agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the agent."
    )
    # Note: if this is None, then we'll populate with the standard "more human than human" initial message sequence
    # If the client wants to make this empty, then the client can set the arg to an empty list
    initial_message_sequence: Optional[List[MessageCreate]] = Field(
        None,
        description="The initial set of messages to put in the agent's in-context memory.",
    )
    include_base_tools: bool = Field(
        True,
        description="If true, attaches the Mirix core tools (e.g. archival_memory and core_memory related functions).",
    )
    include_multi_agent_tools: bool = Field(
        False,
        description="If true, attaches the Mirix multi-agent tools (e.g. sending a message to another agent).",
    )
    parent_id: Optional[str] = Field(None, description="The parent agent ID (for sub-agents in a meta-agent).")
    model: Optional[str] = Field(
        None,
        description="The LLM configuration handle used by the agent, specified in the format "
        "provider/model-name, as an alternative to specifying llm_config.",
    )
    embedding: Optional[str] = Field(
        None,
        description="The embedding configuration handle used by the agent, specified in the format provider/model-name.",
    )
    context_window_limit: Optional[int] = Field(None, description="The context window limit used by the agent.")
    embedding_chunk_size: Optional[int] = Field(
        DEFAULT_EMBEDDING_CHUNK_SIZE,
        description="The embedding chunk size used by the agent.",
    )
    from_template: Optional[str] = Field(None, description="The template id used to configure the agent")
    template: bool = Field(False, description="Whether the agent is a template")
    project: Optional[str] = Field(None, description="The project slug that the agent will be associated with.")
    tool_exec_environment_variables: Optional[Dict[str, str]] = Field(
        None,
        description="The environment variables for tool execution specific to this agent.",
    )
    memory_variables: Optional[Dict[str, str]] = Field(
        None, description="The variables that should be set for the agent."
    )
    mcp_tools: Optional[List[str]] = Field(None, description="List of MCP server names to connect to this agent.")

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validate the requested new agent name (prevent bad inputs)"""

        import re

        if not name:
            # don't check if not provided
            return name

        # TODO: this check should also be added to other model (e.g. User.name)
        # Length check
        if not (1 <= len(name) <= 50):
            raise ValueError("Name length must be between 1 and 50 characters.")

        # Regex for allowed characters (alphanumeric, spaces, hyphens, underscores)
        if not re.match("^[A-Za-z0-9 _-]+$", name):
            raise ValueError("Name contains invalid characters.")

        # Further checks can be added here...
        # TODO

        return name

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: Optional[str]) -> Optional[str]:
        if not model:
            return model

        provider_name, model_name = model.split("/", 1)
        if not provider_name or not model_name:
            raise ValueError("The llm config handle should be in the format provider/model-name")

        return model

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, embedding: Optional[str]) -> Optional[str]:
        if not embedding:
            return embedding

        provider_name, embedding_name = embedding.split("/", 1)
        if not provider_name or not embedding_name:
            raise ValueError("The embedding config handle should be in the format provider/model-name")

        return embedding


class UpdateAgent(BaseModel):
    name: Optional[str] = Field(None, description="The name of the agent.")
    tool_ids: Optional[List[str]] = Field(None, description="The ids of the tools used by the agent.")
    block_ids: Optional[List[str]] = Field(None, description="The ids of the blocks used by the agent.")
    system: Optional[str] = Field(None, description="The system prompt used by the agent.")
    tool_rules: Optional[List[ToolRule]] = Field(None, description="The tool rules governing the agent.")
    llm_config: Optional[LLMConfig] = Field(None, description="The LLM configuration used by the agent.")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the agent."
    )
    message_ids: Optional[List[str]] = Field(
        None, description="The ids of the messages in the agent's in-context memory."
    )
    description: Optional[str] = Field(None, description="The description of the agent.")
    parent_id: Optional[str] = Field(None, description="The parent agent ID (for sub-agents in a meta-agent).")
    mcp_tools: Optional[List[str]] = Field(None, description="List of MCP server names to connect to this agent.")

    class Config:
        extra = "ignore"  # Ignores extra fields


class CreateMetaAgent(BaseModel):
    """Request schema for creating a MetaAgent."""

    name: Optional[str] = Field(
        None,
        description="Optional name for the MetaAgent. If None, a random name will be generated.",
    )
    agents: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=lambda: [
            "core_memory_agent",
            "resource_memory_agent",
            "semantic_memory_agent",
            "episodic_memory_agent",
            "procedural_memory_agent",
            "knowledge_vault_memory_agent",
            "meta_memory_agent",
            "reflexion_agent",
            "background_agent",
        ],
        description="List of memory agent names or dicts with agent configs. Supports both 'agent_name' strings and {'agent_name': {'blocks': [...], ...}} dicts.",
    )
    system_prompts: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary mapping agent names to their system prompt text. Takes precedence over system_prompts_folder.",
    )
    llm_config: Optional[LLMConfig] = Field(
        None,
        description="LLM configuration for memory agents. Required if no default is set.",
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        None,
        description="Embedding configuration for memory agents. Required if no default is set.",
    )


class UpdateMetaAgent(BaseModel):
    """Request schema for updating a MetaAgent."""

    name: Optional[str] = Field(
        None,
        description="Optional new name for the MetaAgent.",
    )
    agents: Optional[List[Union[str, Dict[str, Any]]]] = Field(
        None,
        description="List of memory agent names or dicts with agent configs. Will be compared with existing agents to determine what to add/remove.",
    )
    system_prompts: Optional[Dict[str, str]] = Field(
        None,
        description="Dictionary mapping agent names to their system prompt text. Updates only the specified agents.",
    )
    llm_config: Optional[LLMConfig] = Field(
        None,
        description="LLM configuration for meta agent and its sub-agents.",
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        None,
        description="Embedding configuration for meta agent and its sub-agents.",
    )

    class Config:
        extra = "ignore"  # Ignores extra fields


class AgentStepResponse(BaseModel):
    messages: List[Message] = Field(..., description="The messages generated during the agent's step.")
    continue_chaining: bool = Field(
        ...,
        description="Whether the agent requested a contine_chaining (i.e. follow-up execution).",
    )
    function_failed: bool = Field(..., description="Whether the agent step ended because a function call failed.")
    in_context_memory_warning: bool = Field(
        ...,
        description="Whether the agent step ended because the in-context memory is near its limit.",
    )
    usage: UsageStatistics = Field(..., description="Usage statistics of the LLM call during the agent's step.")
    traj: Optional[dict] = Field(None, description="Action, Observation, State at the current step")


class AgentStepState(BaseModel):
    step_number: int = Field(..., description="The current step number in the agent loop")
    tool_rules_solver: ToolRulesSolver = Field(..., description="The current state of the ToolRulesSolver")


def get_prompt_template_for_agent_type(agent_type: Optional[AgentType] = None):
    if agent_type == AgentType.sleeptime_agent:
        return (
            "{% for block in blocks %}"
            '<{{ block.label }} characters="{{ block.value|length }}/{{ block.limit }}">\n'
            "{% for line in block.value.split('\\n') %}"
            "Line {{ loop.index }}: {{ line }}\n"
            "{% endfor %}"
            "</{{ block.label }}>"
            "{% if not loop.last %}\n{% endif %}"
            "{% endfor %}"
        )
    return (
        "{% for block in blocks %}"
        '<{{ block.label }} characters="{{ block.value|length }}/{{ block.limit }}">\n'
        "{{ block.value }}\n"
        "</{{ block.label }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %}"
    )


# Rebuild model to support forward references (children field)
AgentState.model_rebuild()
