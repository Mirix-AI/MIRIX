"""ORM model for the retrieval log.

One row per memory-retrieval call, id returned as `retrieval_id`. Kept for the
retention window (`MIRIX_RETRIEVAL_LOG_RETENTION_DAYS`) and deleted with the
user's data.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin

`client_id` and `session_id` are plain Strings (no FK): a log row must not
block deleting a client, and sessions are not a table of their own for every
memory type. Status columns are plain Strings validated in
`schemas/retrieval_log.py`. No relationships are declared.
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.message import SESSION_ID_MAX_LEN
from mirix.schemas.retrieval_log import RetrievalLog as PydanticRetrievalLog


class RetrievalLog(SqlalchemyBase, OrganizationMixin, UserMixin):
    """One memory-retrieval call and its ranking trace."""

    __tablename__ = "retrieval_log"
    __pydantic_model__ = PydanticRetrievalLog

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"rlog-{uuid.uuid4()}",
        doc="Returned to the caller as retrieval_id.",
    )
    client_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The calling client."
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(SESSION_ID_MAX_LEN),
        nullable=True,
        doc="The caller's session, when given.",
    )
    memory_types: Mapped[Optional[List[str]]] = mapped_column(
        JSON, nullable=True, doc="Types searched."
    )
    query_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Raw query text."
    )
    extracted_topics: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Extracted topics."
    )
    extraction_status: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="ok | fallback | skipped."
    )
    embedding_status: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="ok | failed | skipped."
    )
    per_type: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, doc="Per-type ranking trace."
    )
    llm_usage: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, doc="Extraction token usage."
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="End-to-end latency (ms)."
    )
    retry_events: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
        doc="One entry per retry during the retrieval: "
        "{kind, attempt, waited_s, at, stage?, error?}.",
    )
    embedding_chunked_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Embedding calls of the retrieval whose input had to be chunked.",
    )

    __table_args__ = (
        # GET /memory/retrieval_logs?session_id=... for one user, newest first.
        Index(
            "ix_retrieval_log_org_user_created",
            "organization_id",
            "user_id",
            "created_at",
        ),
        # GET /memory/retrieval_logs lists one client's rows, newest first;
        # a client's erasure deletes them by client_id.
        Index(
            "ix_retrieval_log_org_client_created",
            "organization_id",
            "client_id",
            "created_at",
        ),
        Index("ix_retrieval_log_session_id", "session_id"),
        # Retention sweep: DELETE ... WHERE created_at < cutoff.
        Index("ix_retrieval_log_created_at", "created_at"),
    )
