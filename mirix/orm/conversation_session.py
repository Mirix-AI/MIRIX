"""ORM model for the conversation-session lifecycle store.

One row per (organization_id, user_id, session_id) in the Conversation Message
Store. It records the session's lifetime (first/last message, explicit or idle
close, reopens), the meta agent that wrote it (the scope for triggering,
distillation and curation), and the skill learner's bookkeeping: distill
failure count / terminal failure time, the infrastructure-error backoff
(`distill_retry_after`, doubling with `distill_infra_failures`), and
storage-cap truncation totals.

`agent_context` (optional) is what the writing client said about how
the agent's prompt was built (its prompt template / setup), attached with an
add of the session's turns; it is not a conversation turn. The distiller shows
it, verbatim, beside the session's transcript.

Whether a session still needs distilling is derived per message row
(`conversation_message.distilled_at IS NULL`), never stored here.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin

`agent_id` is a plain String (no FK): sessions backfilled from pre-existing
conversation rows may not be attributable to a meta agent, and the column must
survive agent re-creation. `session_tag` and `close_reason` are plain Strings
validated in `schemas/conversation_session.py`. No `user`/`organization`
relationships are declared: this bookkeeping table is read on hot paths (every
task write, every sweep) and must not pay for selectin loads nobody uses.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.conversation_session import (
    ConversationSession as PydanticConversationSession,
)
from mirix.schemas.message import SESSION_ID_MAX_LEN, SESSION_ID_SQL_PATTERN


class ConversationSession(SqlalchemyBase, OrganizationMixin, UserMixin):
    """Lifecycle row for one external conversation session."""

    __tablename__ = "conversation_session"
    __pydantic_model__ = PydanticConversationSession

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"convsess-{uuid.uuid4()}",
        doc="Unique ID for this session row.",
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Meta memory agent that wrote this session (scope for trigger/distill/curate).",
    )
    session_id: Mapped[str] = mapped_column(
        String(SESSION_ID_MAX_LEN),
        nullable=False,
        doc="External session identifier (same format as conversation_message.session_id).",
    )
    session_tag: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="'task' | 'conversation', fixed by the first write. NULL = legacy backfill.",
    )
    first_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="created_at of the first message."
    )
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, doc="created_at of the latest message."
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Stored message count.",
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the session ended; NULL while open (an idle open session is materialized on first check).",
    )
    close_reason: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="'explicit' | 'idle'."
    )
    reopen_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Times a write reopened it.",
    )
    distill_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Counted distill failures.",
    )
    distill_failed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Terminal distill failure; the session is no longer selected.",
    )
    distill_retry_after: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Infrastructure-error backoff: not counted or claimed before this time.",
    )
    distill_infra_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Consecutive infrastructure failures since the last success or reopen; "
        "the backoff doubles with each. Never counted as distill_failures.",
    )
    truncated_message_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Messages truncated by the cap.",
    )
    truncated_chars: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Characters removed by the cap.",
    )
    agent_context: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="How the agent's prompt was built, as its client attached it; "
        "the latest one given wins. Not a conversation turn.",
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            "session_id",
            name="uq_conversation_session_org_user_session",
        ),
        # Same backstop as conversation_message: a create_all-built database must
        # never store an invalid session_id. Postgres-only (POSIX regex operator).
        CheckConstraint(
            f"session_id ~ '{SESSION_ID_SQL_PATTERN}'",
            name="ck_conversation_session_session_id_format",
        ).ddl_if(dialect="postgresql"),
        # Trigger counting / flush claim: pending task sessions of one scope.
        Index(
            "ix_conversation_session_scope",
            "organization_id",
            "user_id",
            "agent_id",
            "session_tag",
        ),
        # Sweep: open sessions whose last message is older than the idle cutoff.
        Index(
            "ix_conversation_session_closed_last_message",
            "closed_at",
            "last_message_at",
        ),
    )
