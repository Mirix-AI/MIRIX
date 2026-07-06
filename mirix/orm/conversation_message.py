"""ORM model for the Conversation Message Store.

A `ConversationMessage` is one external conversation turn — a real `user` or
`assistant` message that arrived through the memory-add API *carrying a
session_id*. It is the canonical, learnable record of a conversation and the
SINGLE source the procedural-memory (skill) distiller reads.

This store is deliberately SEPARATE from the `messages` table that backs the
agent loop. The `messages` table also holds the meta agent's own synthesized
bookkeeping (`[System Message]` bootstraps, `trigger_memory_update` tool calls,
`continue_chaining` heartbeats); reading skills from there made the distiller
learn from MIRIX operating *itself*. By giving the real conversation its own
home — with its REAL roles preserved, not the `[USER]`/`[ASSISTANT]`
role-collapsed form the meta agent receives — correctness comes from structure
(which table we read) rather than from fragile string conventions.

Only turns that arrived with a `session_id` land here (`session_id` is NOT
NULL): this store holds session'd conversation turns and nothing else. A
session's order is defined by `MIN(created_at)` over its turns; `distilled_at`
marks a session that a rolling distill round has already consumed so the barrier
advances and history is never reprocessed.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin

`role` is a plain String (NOT a pg ENUM); its value space ('user' | 'assistant')
is Literal-validated in `schemas/conversation_message.py`, so the value space can
change without a DB migration. The `id` PK carries a `convmsg-` prefix matching
the `sexp-`/`proc-` convention on sibling tables.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.conversation_message import (
    ConversationMessage as PydanticConversationMessage,
)
from mirix.schemas.message import SESSION_ID_MAX_LEN, SESSION_ID_SQL_PATTERN
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class ConversationMessage(SqlalchemyBase, OrganizationMixin, UserMixin):
    """One external conversation turn that arrived with a session_id."""

    __tablename__ = "conversation_message"
    __pydantic_model__ = PydanticConversationMessage

    # Override the bare SqlalchemyBase PK to attach a prefixed default, matching
    # the `sexp-`/`proc-` convention on sibling tables.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"convmsg-{uuid.uuid4()}",
        doc="Unique ID for this conversation message.",
    )

    session_id: Mapped[str] = mapped_column(
        String(SESSION_ID_MAX_LEN),
        nullable=False,
        index=True,
        doc="The conversation this turn belongs to. NOT NULL — this store only "
        "holds session'd turns. MIN(created_at) per session defines order.",
    )
    role: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="The real turn role: 'user' or 'assistant' (validated in Pydantic, "
        "not a pg ENUM). NOT the role-collapsed form the meta agent receives.",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="The verbatim text of this conversation turn.",
    )
    distilled_at: Mapped[Optional["datetime"]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="Set when this session's turns were consumed by a distill round. "
        "NULL = not yet distilled; the rolling barrier reads only NULL sessions.",
    )

    # Indexes mirror the skill_experience pg/sqlite guard pattern: an explicit
    # `is not None` filter (not `filter(None, ...)`) because an un-attached Index
    # is falsy until bound. The composite index backs the primary access pattern
    # — "list/seal/order this (org, user)'s sessions by first-appearance time".
    __table_args__ = tuple(
        item
        for item in [
            # Backstop the app-level validator so a create_all-built database
            # matches the SQL migration: the DB must never store an invalid
            # session_id. Unlike `messages`, session_id here is NOT NULL (this
            # store only holds session'd turns), so the NULL branch is omitted.
            # Uses the Postgres `~` operator, so emit only on Postgres (SQLite,
            # used for some local/test setups, has no POSIX regex operator).
            # Pattern derived from mirix.schemas.message — one source of truth.
            CheckConstraint(
                f"session_id ~ '{SESSION_ID_SQL_PATTERN}'",
                name="ck_conversation_message_session_id_format",
            ).ddl_if(dialect="postgresql"),
            Index(
                "ix_conversation_message_org_user_session_created",
                "organization_id",
                "user_id",
                "session_id",
                "created_at",
            ),
            # Accelerates the per-session ascending fetch in list_turns_for_session.
            Index(
                "ix_conversation_message_session_created",
                "session_id",
                "created_at",
            ),
            (
                Index(
                    "ix_conversation_message_organization_id",
                    "organization_id",
                )
                if settings.mirix_pg_uri_no_default
                else None
            ),
        ]
        if item is not None
    )

    @declared_attr
    def user(cls) -> Mapped[Optional["User"]]:
        return relationship("User", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped[Optional["Organization"]]:
        return relationship("Organization", lazy="selectin")
