from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.source_message import SourceMessage as PydanticSourceMessage
from mirix.settings import settings


class SourceMessage(SqlalchemyBase):
    """
    ORM model for individual messages within a memory source.

    Each message in a POST /memory/add request is stored as a SourceMessage,
    linked to its parent MemorySource. Used for citation retrieval and
    content-hash deduplication.
    """

    __tablename__ = "source_messages"
    __pydantic_model__ = PydanticSourceMessage

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        doc="Unique ID for this source message",
    )

    memory_source_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="ID of the parent memory source",
    )

    external_thread_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Denormalized thread ID from parent source",
    )

    external_message_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Client-provided stable message ID",
    )

    role: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Message role: user, assistant, or system",
    )

    content: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        doc="Message content structure",
    )

    occurred_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Per-message timestamp",
    )

    sequence_num: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Ordering within the source",
    )

    content_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="SHA-256 of (role, content) for dedup",
    )

    message_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        doc="Client-provided per-message property bag",
    )

    __table_args__ = tuple(
        filter(
            None,
            [
                # Partial unique index: external_message_id dedup (PostgreSQL only)
                (
                    Index(
                        "uq_source_messages_ext_id",
                        "memory_source_id",
                        "external_message_id",
                        unique=True,
                        postgresql_where=text("external_message_id IS NOT NULL"),
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Full unique constraint: content_hash dedup
                (
                    Index(
                        "uq_source_messages_hash",
                        "memory_source_id",
                        "content_hash",
                        unique=True,
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
                # Thread query index
                (
                    Index(
                        "ix_source_messages_thread",
                        "external_thread_id",
                        "occurred_at",
                    )
                    if settings.mirix_pg_uri_no_default
                    else None
                ),
            ],
        )
    )
