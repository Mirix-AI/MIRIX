"""
ORM model for raw (unprocessed) task memories.

Raw memories store task context without LLM extraction, intended for
task sharing use cases with a 14-day TTL.
"""

import datetime as dt
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, DateTime, Index, String, Text, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.raw_memory import RawMemoryItem as PydanticRawMemoryItem
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class RawMemory(SqlalchemyBase, OrganizationMixin, UserMixin):
    """
    ORM model for raw (unprocessed) task memories.

    Raw memories store task context without LLM extraction, intended for
    task sharing use cases with a 14-day TTL.
    """

    __tablename__ = "raw_memory"
    __pydantic_model__ = PydanticRawMemoryItem

    # Primary key
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this raw memory entry",
    )

    # Note: user_id is provided by UserMixin with ForeignKey to users table
    # Note: organization_id is provided by OrganizationMixin with ForeignKey to organizations table

    # Content field
    context: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Raw task context string (unprocessed)",
    )

    # filter_tags stores scope and other metadata (matching episodic_memory pattern)
    filter_tags: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Custom filter tags including scope for access control",
    )

    # Last modification tracking (standard MIRIX pattern)
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

        context_embedding = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        context_embedding = Column(CommonVector, nullable=True)

    # Timestamps
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When the event occurred or was recorded",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When record was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        doc="When record was last updated",
    )

    # Audit fields (track which client created/updated the record)
    _created_by_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client ID that created this memory",
    )
    _last_update_by_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client ID that last updated this memory",
    )

    # Indexes following standard MIRIX memory table pattern
    __table_args__ = tuple(
        filter(
            None,
            [
                # PostgreSQL indexes
                Index("ix_raw_memory_organization_id", "organization_id") if settings.mirix_pg_uri_no_default else None,
                (
                    Index(
                        "ix_raw_memory_org_updated_at",
                        "organization_id",
                        "updated_at",
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_raw_memory_filter_tags_gin",
                        text("(filter_tags::jsonb)"),
                        postgresql_using="gin",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                (
                    Index(
                        "ix_raw_memory_org_filter_scope",
                        "organization_id",
                        text("((filter_tags->>'scope')::text)"),
                        postgresql_using="btree",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # SQLite fallback indexes
                (
                    Index(
                        "ix_raw_memory_organization_id_sqlite",
                        "organization_id",
                    )
                    if not settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )

    @declared_attr
    def organization(cls) -> Mapped["Organization"]:
        """Relationship to the Organization."""
        return relationship("Organization", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped["User"]:
        """Relationship to the User."""
        return relationship("User", lazy="selectin")
