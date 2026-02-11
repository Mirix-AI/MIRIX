from typing import TYPE_CHECKING, Optional, Type

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, UniqueConstraint, event
from sqlalchemy.orm import (
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
)

from mirix.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.block import Human, Persona

if TYPE_CHECKING:
    from mirix.orm import Organization
    from mirix.orm.agent import Agent
    from mirix.orm.user import User


class Block(OrganizationMixin, UserMixin, SqlalchemyBase):
    """Blocks are sections of the LLM context, representing a specific part of the total Memory"""

    __tablename__ = "block"
    __pydantic_model__ = PydanticBlock
    __table_args__ = (
        UniqueConstraint("id", "label", name="unique_block_id_label"),
        Index("idx_block_id_label", "id", "label", unique=True),
    )

    label: Mapped[str] = mapped_column(doc="the type of memory block in use, ie 'human', 'persona', 'system'")
    value: Mapped[str] = mapped_column(doc="Text content of the block for the respective section of core memory.")
    limit: Mapped[BigInteger] = mapped_column(
        Integer,
        default=CORE_MEMORY_BLOCK_CHAR_LIMIT,
        doc="Character limit of the block.",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this block belongs to",
    )

    # relationships
    organization: Mapped[Optional["Organization"]] = relationship("Organization")

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        """
        Relationship to the Agent that owns this block.
        """
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this block.
        """
        return relationship("User", lazy="selectin")

    def to_pydantic(self) -> Type:
        if self.label == "human":
            Schema = Human
        elif self.label == "persona":
            Schema = Persona
        else:
            Schema = PydanticBlock
        return Schema.model_validate(self)


@event.listens_for(Block, "before_insert")
@event.listens_for(Block, "before_update")
def validate_value_length(mapper, connection, target):
    """Ensure the value length does not exceed the limit."""
    if target.value and len(target.value) > target.limit:
        raise ValueError(
            f"Value length ({len(target.value)}) exceeds the limit ({target.limit}) for block with label '{target.label}' and id '{target.id}'."
        )
