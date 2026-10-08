"""ORM model for skill learning runs.

One row per learner execution for one (organization, user, meta agent) scope.
The row is written as `queued` in the same transaction that claims its
sessions (with their watermarks), then driven by a background task that
refreshes `heartbeat_at` every 30 seconds. Runs whose heartbeat lapses past the
lease are reclaimed as `failed` / `lease_expired`, releasing their sessions.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin
  - agent_id (FK agents.id ON DELETE CASCADE) -- AgentMixin

`trigger`, `mode` and `status` are plain Strings validated in
`schemas/skill_learning_run.py`. No relationships are declared (hot-path
bookkeeping; see conversation_session).
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.skill_learning_run import (
    SkillLearningRun as PydanticSkillLearningRun,
)


class SkillLearningRun(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """One skill-learner execution (distill -> curate) for one scope."""

    __tablename__ = "skill_learning_run"
    __pydantic_model__ = PydanticSkillLearningRun

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"slrun-{uuid.uuid4()}",
        doc="The run id; curator batches are `{id}:b{i}`.",
    )
    trigger: Mapped[str] = mapped_column(
        String, nullable=False, doc="threshold | close | sweep | flush | manual."
    )
    mode: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="procedural",
        server_default=text("'procedural'"),
        doc="The auto_dream mode run: 'procedural' (skill learning) or another "
        "auto_dream mode (core, episodic, semantic, resource, knowledge, "
        "experience) — every auto_dream run is recorded here.",
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="queued",
        server_default=text("'queued'"),
        doc="queued | running | succeeded | failed.",
    )
    session_ids: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
        doc="Claimed session_ids.",
    )
    session_watermarks: Mapped[Dict[str, str]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
        doc="session_id -> ISO timestamp of its newest row at claim time.",
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="Lease heartbeat."
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Failure reason."
    )
    distill_metrics: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, doc="One entry per distilled session."
    )
    curator_metrics: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON, nullable=True, doc="One entry per curator batch."
    )
    llm_usage: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True, doc="Aggregated token usage."
    )
    embedding_chunked_calls: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Chunked embedding calls.",
    )
    retry_events: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
        doc="One entry per application-level retry: "
        "{kind, attempt, waited_s, at, stage?, session_id?, error?}.",
    )

    __table_args__ = (
        # "Active runs of this scope" (claim exclusion, wait-until-idle).
        Index(
            "ix_skill_learning_run_scope_status",
            "organization_id",
            "user_id",
            "agent_id",
            "status",
        ),
        # Lease reaper: queued/running runs with a stale heartbeat.
        Index("ix_skill_learning_run_status_heartbeat", "status", "heartbeat_at"),
        # GET /memory/learning_runs listing, newest first.
        Index(
            "ix_skill_learning_run_org_user_created",
            "organization_id",
            "user_id",
            "created_at",
        ),
    )
