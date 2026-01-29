import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, String, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.episodic_memory import EpisodicEvent as PydanticEpisodicEvent
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class EpisodicEvent(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    Represents an event in the 'episodic memory' system, capturing
    timestamped interactions or observations with a short summary
    and optional detailed notes or metadata.
    """

    __tablename__ = "episodic_memory"
    __pydantic_model__ = PydanticEpisodicEvent

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for the episodic event",
    )

    # Foreign key to agent
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the agent this episodic event belongs to",
    )

    # Foreign key to client (for access control and filtering)
    client_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=True,
        doc="ID of the client application that created this event",
    )

    # When did this event occur? (You can store creation time or an explicit event time.)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, doc="Timestamp when the event occurred or was recorded")

    # When was this event last modified and what operation?
    last_modify: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "timestamp": datetime.now(dt.timezone.utc).isoformat(),
            "operation": "created",
        },
        doc="Last modification info including timestamp and operation type",
    )

    # Who or what triggered this event (e.g., 'user', 'assistant', 'system', etc.)
    actor: Mapped[str] = mapped_column(String, doc="Identifies the actor/source of this event")

    event_type: Mapped[str] = mapped_column(
        String,
        doc="Type/category of the episodic event (e.g., user_message, inference, system_notification)",
    )

    # A brief summary/title of the event
    summary: Mapped[str] = mapped_column(String, doc="Short summary of the event")

    details: Mapped[str] = mapped_column(
        String, nullable=False, doc="Detailed description or narrative about this event"
    )

    # NEW: Filter tags for flexible filtering and categorization
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=None, doc="Custom filter tags for filtering and categorization"
    )

    embedding_config: Mapped[Optional[dict]] = mapped_column(
        EmbeddingConfigColumn, nullable=True, doc="Embedding configuration"
    )

    # Vector embedding field based on database type
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        details_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
        summary_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        details_embedding = Column(CommonVector, nullable=True)
        summary_embedding = Column(CommonVector, nullable=True)

    # Full-text search indexes - handled by migration script
    # PostgreSQL: GIN indexes on tsvector expressions
    # SQLite: FTS5 virtual table with triggers
    __table_args__ = tuple(
        filter(
            None,
            [
                # PostgreSQL full-text search indexes
                (
                    Index(
                        "ix_episodic_memory_summary_fts",
                        text("to_tsvector('english', summary)"),
                        postgresql_using="gin",
                        postgresql_where=text("summary IS NOT NULL"),
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_episodic_memory_details_fts",
                        text("to_tsvector('english', details)"),
                        postgresql_using="gin",
                        postgresql_where=text("details IS NOT NULL"),
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_episodic_memory_combined_fts",
                        text("to_tsvector('english', coalesce(summary, '') || ' ' || coalesce(details, ''))"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Organization-level query optimization indexes
                (
                    Index("ix_episodic_memory_organization_id", "organization_id")
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_episodic_memory_org_occurred_at",
                        "organization_id",
                        "occurred_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_episodic_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_episodic_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Standard indexes for SQLite (FTS5 virtual table handled separately)
                Index("ix_episodic_memory_summary_sqlite", "summary") if not settings.mirix_pg_uri_no_default else None,
                Index("ix_episodic_memory_details_sqlite", "details") if not settings.mirix_pg_uri_no_default else None,
                (
                    Index("ix_episodic_memory_organization_id_sqlite", "organization_id")
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        """
        Relationship to the Agent that owns this episodic event.
        """
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """
        Relationship to the Organization that owns this event.
        Matches back_populates on the 'EpisodicEvent' relationship in Organization.
        """
        return relationship("Organization", back_populates="episodic_memory", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """
        Relationship to the User that owns this episodic event.
        """
        return relationship("User", lazy="selectin")
