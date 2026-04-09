import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.procedural_memory import ProceduralMemoryItem as PydanticProceduralMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class ProceduralMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores procedural knowledge as reusable skills.

    name:         Short skill identifier (e.g. 'deploy-production')
    entry_type:   Category or tag of the skill (e.g. 'workflow', 'guide', 'script')
    description:  Short descriptive text about what this skill accomplishes
    instructions: Step-by-step instructions as plain text
    triggers:     Conditions that indicate this skill is relevant
    examples:     Input/output examples for this skill
    version:      Semantic version of this skill
    """

    __tablename__ = "procedural_memory"
    __pydantic_model__ = PydanticProceduralMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this procedural memory entry",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this procedural memory item belongs to",
    )

    # Foreign key to client (for access control and filtering)
    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this item",
    )

    # Short skill identifier
    name: Mapped[str] = mapped_column(String, nullable=False, doc="Short skill identifier (e.g. 'deploy-production')")

    # Distinguish the type/category of the skill
    entry_type: Mapped[str] = mapped_column(String, doc="Category or type (e.g. 'workflow', 'guide', 'script')")

    # A human-friendly description of this skill
    description: Mapped[str] = mapped_column(String, doc="Short descriptive text about what this skill accomplishes")

    # Step-by-step instructions as plain text
    instructions: Mapped[str] = mapped_column(String, doc="Step-by-step instructions as a single string")

    # Conditions that indicate this skill is relevant
    triggers: Mapped[list] = mapped_column(JSON, default=list, doc="Conditions that indicate this skill is relevant")

    # Input/output examples
    examples: Mapped[list] = mapped_column(JSON, default=list, doc="Input/output examples for this skill")

    # Semantic version
    version: Mapped[str] = mapped_column(String, default="0.1.0", doc="Semantic version of this skill")

    # NEW: Filter tags for flexible filtering and categorization
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags for filtering and categorization"
    )

    # When was this item last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "operation": "created",
        },
        doc="Last modification info including timestamp and operation type",
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, nullable=True, doc="Embedding configuration"
    )

    # Vector embedding field based on database type
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        description_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        instructions_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        description_embedding = Column(CommonVector, nullable=True)
        instructions_embedding = Column(CommonVector, nullable=True)

    # Database indexes for efficient querying
    __table_args__ = tuple(
        filter(
            None,
            [
                # Organization-level query optimization indexes
                (
                    Index("ix_procedural_memory_organization_id", "organization_id")
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_created_at",
                        "organization_id",
                        "created_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_procedural_memory_org_user_name",
                        "organization_id",
                        "user_id",
                        "name",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # SQLite indexes
                (
                    Index("ix_procedural_memory_organization_id_sqlite", "organization_id")
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        """
        Relationship to the Agent that owns this procedural memory item.
        """
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        Adjust 'back_populates' to match the collection name in your `Organization` model.
        """
        return relationship("Organization", back_populates="procedural_memory", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this procedural memory item.
        """
        return relationship("User", lazy="selectin")
