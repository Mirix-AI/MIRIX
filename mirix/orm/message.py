from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, CheckConstraint, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.custom_columns import (
    MessageContentColumn,
    ToolCallColumn,
    ToolReturnColumn,
)
from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.message import (
    SESSION_ID_MAX_LEN,
    SESSION_ID_SQL_PATTERN,
    Message as PydanticMessage,
    ToolReturn,
)
from mirix.schemas.mirix_message_content import MessageContent
from mirix.schemas.mirix_message_content import TextContent as PydanticTextContent
from mirix.schemas.openai.openai import ToolCall as OpenAIToolCall

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.step import Step
    from mirix.orm.user import User


class Message(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """Defines data model for storing Message objects"""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_agent_created_at", "agent_id", "created_at"),
        Index("ix_messages_created_at", "created_at", "id"),
        Index("ix_messages_client_user", "client_id", "user_id"),
        Index("ix_messages_agent_client_user", "agent_id", "client_id", "user_id"),
        # Accelerates "list messages of an agent in a session, newest first".
        Index(
            "ix_messages_agent_session_created_at",
            "agent_id",
            "session_id",
            "created_at",
        ),
        # Backstop the app-level validator: DB must never store an invalid session_id.
        # Uses the Postgres `~` operator, so emit the constraint only on Postgres
        # (SQLite, used for some local/test setups, has no POSIX regex operator).
        # Pattern derived from mirix.schemas.message so there's one source of truth.
        CheckConstraint(
            f"session_id IS NULL OR session_id ~ '{SESSION_ID_SQL_PATTERN}'",
            name="ck_messages_session_id_format",
        ).ddl_if(dialect="postgresql"),
    )
    __pydantic_model__ = PydanticMessage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique message identifier")
    role: Mapped[str] = mapped_column(doc="Message role (user/assistant/system/tool)")
    text: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Message content")
    content: Mapped[List[MessageContent]] = mapped_column(
        MessageContentColumn, nullable=True, doc="Message content parts"
    )
    model: Mapped[Optional[str]] = mapped_column(nullable=True, doc="LLM model used")
    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Name for multi-agent scenarios")
    tool_calls: Mapped[List[OpenAIToolCall]] = mapped_column(ToolCallColumn, doc="Tool call information")
    tool_call_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="ID of the tool call")

    # NEW: Filter tags for flexible filtering and categorization
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags for filtering and categorization"
    )

    # Foreign key to client (for access control and filtering)
    client_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this message",
    )

    step_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("steps.id", ondelete="SET NULL"),
        nullable=True,
        doc="ID of the step that this message belongs to",
    )
    otid: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="The offline threading ID associated with this message"
    )
    tool_returns: Mapped[List[ToolReturn]] = mapped_column(
        ToolReturnColumn,
        nullable=True,
        doc="Tool execution return information for prior tool calls",
    )
    group_id: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="The multi-agent group that the message was sent in"
    )
    sender_id: Mapped[Optional[str]] = mapped_column(
        nullable=True,
        doc="The id of the sender of the message, can be an identity id or agent id",
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(SESSION_ID_MAX_LEN),
        nullable=True,
        doc="Top-level conversation/session identifier for grouping messages. "
        "Enforced by app validator and DB CHECK constraint "
        f"(pattern {SESSION_ID_SQL_PATTERN}).",
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="messages", lazy="selectin")
    organization: Mapped["Organization"] = relationship("Organization", back_populates="messages", lazy="selectin")
    step: Mapped["Step"] = relationship("Step", back_populates="messages", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this message.
        """
        return relationship("User", lazy="selectin")

    def to_pydantic(self) -> PydanticMessage:
        """Custom pydantic conversion to handle data using legacy text field"""
        model = self.__pydantic_model__.model_validate(self)
        if self.text and not model.content:
            model.content = [PydanticTextContent(text=self.text)]
        # If there are no tool calls, set tool_calls to None
        if len(self.tool_calls) == 0:
            model.tool_calls = None
        return model
