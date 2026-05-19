import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.sqlalchemy_base import SqlalchemyBase


class MemoryAccessLog(SqlalchemyBase):
    """
    Lightweight access log recording which memory entries were retrieved and when.

    Used by the Auto-Dreamer v2 consolidator to build the working region:
        R = {newly written in window} ∪ {retrieved by task agent in window}

    Rows are append-only and should be pruned periodically (e.g. older than 90 days).
    """

    __tablename__ = "memory_access_log"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"mal_{uuid.uuid4().hex[:12]}",
    )
    memory_id: Mapped[str] = mapped_column(String, nullable=False)
    memory_type: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    organization_id: Mapped[Optional[str]] = mapped_column(
        String,
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_memory_access_log_user_type_time", "user_id", "memory_type", "accessed_at"),
        Index("ix_memory_access_log_memory_id", "memory_id"),
    )
