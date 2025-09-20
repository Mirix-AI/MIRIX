import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from mirix.orm.block import Block
from mirix.orm.custom_columns import EmbeddingConfigColumn, LLMConfigColumn, ToolRulesColumn
from mirix.orm.message import Message
from mirix.orm.mixins import OrganizationMixin
from mirix.orm.organization import Organization
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.agent import AgentState as PydanticAgentState
from mirix.schemas.agent import AgentType
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import Memory
from mirix.schemas.tool_rule import ToolRule

if TYPE_CHECKING:
    from mirix.orm.agents_tags import AgentsTags
    from mirix.orm.organization import Organization
    from mirix.orm.tool import Tool
    from mirix.orm.user import User


class Agent(SqlalchemyBase, OrganizationMixin):
    __tablename__ = "agents"
    __pydantic_model__ = PydanticAgentState

    # agent generates its own id
    # TODO: We want to migrate all the ORM models to do this, so we will need to move this to the SqlalchemyBase
    # TODO: Some still rely on the Pydantic object to do this
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"agent-{uuid.uuid4()}")

    # Descriptor fields
    agent_type: Mapped[Optional[AgentType]] = mapped_column(String, nullable=True, doc="The type of Agent")
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="a human-readable identifier for an agent, non-unique.")
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The description of the agent.")

    # System prompt
    system: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The system prompt used by the agent.")

    # Current Topic
    topic: Mapped[Optional[str]] = mapped_column(String, nullable=True, doc="The current topic between the agent and the user.")

    # In context memory
    # TODO: This should be a separate mapping table
    # This is dangerously flexible with the JSON type
    message_ids: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, doc="List of message IDs in in-context memory.")

    # Metadata and configs
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, doc="metadata for the agent.")
    llm_config: Mapped[Optional[LLMConfig]] = mapped_column(
        LLMConfigColumn, nullable=True, doc="the LLM backend configuration object for this agent."
    )
    embedding_config: Mapped[Optional[EmbeddingConfig]] = mapped_column(
        EmbeddingConfigColumn, doc="the embedding configuration object for this agent."
    )

    # Tool rules
    tool_rules: Mapped[Optional[List[ToolRule]]] = mapped_column(ToolRulesColumn, doc="the tool rules for this agent.")
    
    # MCP tools - list of connected MCP server names
    mcp_tools: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, doc="List of connected MCP server names (e.g., ['gmail-native'])")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="agents")
    tool_exec_environment_variables: Mapped[List["AgentEnvironmentVariable"]] = relationship(
        "AgentEnvironmentVariable",
        back_populates="agent",
        cascade="all, delete-orphan",
        lazy="selectin",
        doc="Environment variables associated with this agent.",
    )
    tools: Mapped[List["Tool"]] = relationship("Tool", secondary="tools_agents", lazy="selectin", passive_deletes=True)
    core_memory: Mapped[List["Block"]] = relationship("Block", secondary="blocks_agents", lazy="selectin")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="agent",
        lazy="selectin",
        cascade="all, delete-orphan",  # Ensure messages are deleted when the agent is deleted
        passive_deletes=True,
    )
    tags: Mapped[List["AgentsTags"]] = relationship(
        "AgentsTags",
        back_populates="agent",
        cascade="all, delete-orphan",
        lazy="selectin",
        doc="Tags associated with the agent.",
    )

    def to_pydantic(self) -> PydanticAgentState:
        """converts to the basic pydantic model counterpart"""
        state = {
            "id": self.id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "message_ids": self.message_ids,
            "tools": self.tools,
            "tags": [t.tag for t in self.tags],
            "tool_rules": self.tool_rules,
            "system": self.system,
            "topic": self.topic,
            "agent_type": self.agent_type,
            "llm_config": self.llm_config,
            "embedding_config": self.embedding_config,
            "metadata_": self.metadata_,
            "memory": Memory(blocks=[b.to_pydantic() for b in self.core_memory]),
            "created_by_id": self.created_by_id,
            "last_updated_by_id": self.last_updated_by_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_exec_environment_variables": self.tool_exec_environment_variables,
            "mcp_tools": self.mcp_tools,
        }
        return self.__pydantic_model__(**state)
