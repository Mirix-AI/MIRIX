import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.resource_memory import ResourceMemoryItem as PydanticResourceMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class ResourceMemoryItem(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Stores references to user's documents, files, or resources for easy retrieval & linking to tasks.

    title:   A short name/title of the resource (e.g. 'MarketingPlan2025')
    summary:        A brief description or summary of the resource.
    content:         The text/content of the file (can be partial or full)
    resource_type:   Category or type of the resource (e.g. 'doc', 'text', 'markdown', 'spreadsheet')
    """

    __tablename__ = "resource_memory"
    __pydantic_model__ = PydanticResourceMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this resource memory entry",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this resource memory item belongs to",
    )

    # Foreign key to client (for access control and filtering)
    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this item",
    )

    title: Mapped[str] = mapped_column(String, doc="Short name or title of the resource")

    summary: Mapped[str] = mapped_column(String, doc="Brief description or summary of the resource")

    resource_type: Mapped[str] = mapped_column(
        String,
        doc="Type or format of the resource (e.g. 'doc', 'markdown', 'pdf_text')",
    )

    content: Mapped[str] = mapped_column(String, doc="Full text or partial content of this resource")

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

        summary_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        summary_embedding = Column(CommonVector, nullable=True)

    # Database indexes for efficient querying
    __table_args__ = tuple(
        filter(
            None,
            [
                # Organization-level query optimization indexes
                (
                    Index("ix_resource_memory_organization_id", "organization_id")
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_resource_memory_org_created_at",
                        "organization_id",
                        "created_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_resource_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_resource_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # SQLite indexes
                (
                    Index("ix_resource_memory_organization_id_sqlite", "organization_id")
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        """
        Relationship to the Agent that owns this resource memory item.
        """
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to organization (mirroring your existing patterns).
        """
        return relationship("Organization", back_populates="resource_memory", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this resource memory item.
        """
        return relationship("User", lazy="selectin")
