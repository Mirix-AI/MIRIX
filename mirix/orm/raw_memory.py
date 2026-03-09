"""
ORM model for raw (unprocessed) task memories.

Raw memories store task context without LLM extraction, intended for
task sharing use cases with a 14-day TTL.
"""

import datetime as dt
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column, Index, String, Text, text
from sqlalchemy.event import listens_for
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector, DateTimeNaiveUTC, EmbeddingConfigColumn
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

    # Timestamps (DateTimeNaiveUTC ensures bind params are naive for TIMESTAMP WITHOUT TIME ZONE)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTimeNaiveUTC(),
        nullable=False,
        doc="When the event occurred or was recorded",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTimeNaiveUTC(),
        nullable=False,
        doc="When record was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTimeNaiveUTC(),
        nullable=False,
        doc="When record was last updated",
    )

    def set_updated_at(self, timestamp: Optional[datetime] = None) -> None:
        """
        Set updated_at to naive UTC (RawMemory uses TIMESTAMP WITHOUT TIME ZONE).
        Overrides mixin to avoid asyncpg "offset-naive and offset-aware" errors.
        """
        now = timestamp or datetime.now(timezone.utc)
        self.updated_at = now.replace(tzinfo=None) if now.tzinfo else now

    # Note: Audit fields (_created_by_id, _last_updated_by_id) are inherited
    # from CommonSqlalchemyMetaMixins via SqlalchemyBase

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


def _naive_utc(dt_val: Optional[datetime]) -> Optional[datetime]:
    """Return naive UTC datetime for TIMESTAMP WITHOUT TIME ZONE columns."""
    if dt_val is None:
        return None
    if dt_val.tzinfo is not None:
        return dt_val.astimezone(timezone.utc).replace(tzinfo=None)
    return dt_val


@listens_for(RawMemory, "before_update")
def _raw_memory_before_update(mapper, connection, target: RawMemory) -> None:
    """Ensure timestamp columns are naive UTC before UPDATE (asyncpg compatibility)."""
    for attr in ("occurred_at", "created_at", "updated_at"):
        val = getattr(target, attr, None)
        if val is not None and getattr(val, "tzinfo", None) is not None:
            setattr(target, attr, _naive_utc(val))
