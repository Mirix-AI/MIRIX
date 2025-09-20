from typing import List, Optional

from mirix.schemas.openai.openai import ToolCall as OpenAIToolCall
from sqlalchemy import BigInteger, FetchedValue, ForeignKey, Index, event, text
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship, declared_attr

from mirix.orm.custom_columns import MessageContentColumn, ToolCallColumn, ToolReturnColumn
from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.mirix_message_content import MessageContent
from mirix.schemas.mirix_message_content import TextContent as PydanticTextContent
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.message import ToolReturn
from mirix.settings import settings


class Message(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """Defines data model for storing Message objects"""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_agent_created_at", "agent_id", "created_at"),
        Index("ix_messages_created_at", "created_at", "id"),
    )
    __pydantic_model__ = PydanticMessage

    id: Mapped[str] = mapped_column(primary_key=True, doc="Unique message identifier")
    role: Mapped[str] = mapped_column(doc="Message role (user/assistant/system/tool)")
    text: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Message content")
    content: Mapped[List[MessageContent]] = mapped_column(MessageContentColumn, nullable=True, doc="Message content parts")
    model: Mapped[Optional[str]] = mapped_column(nullable=True, doc="LLM model used")
    name: Mapped[Optional[str]] = mapped_column(nullable=True, doc="Name for multi-agent scenarios")
    tool_calls: Mapped[List[OpenAIToolCall]] = mapped_column(ToolCallColumn, doc="Tool call information")
    tool_call_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="ID of the tool call")
    step_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("steps.id", ondelete="SET NULL"), nullable=True, doc="ID of the step that this message belongs to"
    )
    otid: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The offline threading ID associated with this message")
    tool_returns: Mapped[List[ToolReturn]] = mapped_column(
        ToolReturnColumn, nullable=True, doc="Tool execution return information for prior tool calls"
    )
    group_id: Mapped[Optional[str]] = mapped_column(nullable=True, doc="The multi-agent group that the message was sent in")
    sender_id: Mapped[Optional[str]] = mapped_column(
        nullable=True, doc="The id of the sender of the message, can be an identity id or agent id"
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
        return relationship(
            "User",
            lazy="selectin"
        )

    def to_pydantic(self) -> PydanticMessage:
        """Custom pydantic conversion to handle data using legacy text field"""
        model = self.__pydantic_model__.model_validate(self)
        if self.text and not model.content:
            model.content = [PydanticTextContent(text=self.text)]
        # If there are no tool calls, set tool_calls to None
        if len(self.tool_calls) == 0:
            model.tool_calls = None
        return model
