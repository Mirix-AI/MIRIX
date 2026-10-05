"""Pydantic schemas for skill learning runs.

A `SkillLearningRun` records one execution of the skill learner
(distill -> curate) for one (organization, user, meta agent) scope: what
triggered it, its status, the sessions it claimed with their watermarks,
its lease heartbeat, and per-session / per-batch metrics.

Lifecycle: ``queued -> running -> succeeded | failed``. A run whose
`heartbeat_at` is older than `MIRIX_LEARNING_RUN_LEASE_SECONDS` while
queued/running is reclaimed as ``failed`` (error ``lease_expired``), which
releases its claimed sessions.

`session_watermarks` maps each claimed session_id to the ISO-8601 timestamp of
its newest row at claim time; the run only reads and marks rows at or before
the watermark, so rows written after the claim (a reopened session) stay
pending. Values are stored as strings so the JSON column round-trips exactly.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.message import _validate_session_id
from mirix.schemas.mirix_base import MirixBase

LearningRunTrigger = Literal["threshold", "close", "sweep", "flush", "manual"]
LearningRunStatus = Literal["queued", "running", "succeeded", "failed"]
# The auto_dream mode a run executes (every auto_dream run is recorded).
LearningRunMode = Literal[
    "procedural", "core", "episodic", "semantic", "resource", "knowledge", "experience"
]

# Statuses that still hold a claim on their sessions (lease applies).
ACTIVE_LEARNING_RUN_STATUSES = ("queued", "running")


def _normalize_watermark(session_id: str, value: Any) -> str:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"session_watermarks[{session_id!r}] is not an ISO-8601 timestamp: {value!r}"
            ) from exc
    else:
        raise ValueError(
            f"session_watermarks[{session_id!r}] must be a datetime or ISO string, got {type(value).__name__}"
        )
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.isoformat()


class SkillLearningRunBase(MirixBase):
    __id_prefix__ = "slrun"

    trigger: LearningRunTrigger = Field(
        ..., description="threshold | close | sweep | flush | manual."
    )
    mode: LearningRunMode = Field(
        "procedural",
        description="The auto_dream mode: 'procedural' (skill learning from "
        "claimed sessions) or another auto_dream mode (no sessions).",
    )
    session_ids: List[str] = Field(
        default_factory=list, description="Sessions claimed by this run."
    )
    session_watermarks: Dict[str, str] = Field(
        default_factory=dict,
        description="session_id -> ISO timestamp of its newest row at claim time.",
    )

    @field_validator("session_ids")
    @classmethod
    def _check_session_ids(cls, value: List[str]) -> List[str]:
        for session_id in value:
            if _validate_session_id(session_id) is None:
                raise ValueError("session_ids must not contain empty values")
        if len(set(value)) != len(value):
            raise ValueError("session_ids must be unique")
        return value

    @field_validator("session_watermarks", mode="before")
    @classmethod
    def _check_watermarks(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("session_watermarks must be an object")
        return {str(k): _normalize_watermark(str(k), v) for k, v in value.items()}

    @model_validator(mode="after")
    def _watermarks_belong_to_claimed_sessions(self):
        stray = set(self.session_watermarks) - set(self.session_ids)
        if stray:
            raise ValueError(
                f"session_watermarks has sessions not in session_ids: {sorted(stray)}"
            )
        if self.mode != "procedural" and self.session_ids:
            raise ValueError(
                f"a {self.mode!r} auto_dream run claims no sessions (only "
                "procedural runs learn from sessions)"
            )
        return self


class SkillLearningRunCreate(SkillLearningRunBase):
    """Create schema — the scope plus the claim."""

    organization_id: str = Field(..., description="The owning organization id.")
    user_id: str = Field(..., description="The user whose sessions are learned from.")
    agent_id: str = Field(
        ..., description="The meta memory agent whose config drives the run."
    )
    status: LearningRunStatus = Field(
        "queued", description="Initial status (normally 'queued')."
    )


class SkillLearningRun(SkillLearningRunBase):
    """Full learning-run schema, with database-related fields."""

    id: Optional[str] = Field(
        None, description="The run id (curator batches are `{id}:b{i}`)."
    )
    organization_id: str = Field(..., description="The owning organization id.")
    user_id: str = Field(..., description="The user whose sessions are learned from.")
    agent_id: str = Field(
        ..., description="The meta memory agent whose config drives the run."
    )
    status: LearningRunStatus = Field(
        "queued", description="queued | running | succeeded | failed."
    )
    heartbeat_at: Optional[datetime] = Field(
        None, description="Lease heartbeat, refreshed every 30s while alive."
    )
    started_at: Optional[datetime] = Field(None, description="When execution began.")
    finished_at: Optional[datetime] = Field(
        None, description="When the run succeeded or failed."
    )
    error: Optional[str] = Field(
        None, description="Failure reason (e.g. 'lease_expired' or the exception)."
    )
    distill_metrics: Optional[List[Dict[str, Any]]] = Field(
        None, description="One entry per distilled session."
    )
    curator_metrics: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "One entry per curator batch; plus one entry with "
            "stage 'reembed' when the run's closing re-embedding pass found "
            "skills of its user without a retrieval vector, with "
            "error_kind 'embedding' when some are still without one."
        ),
    )
    llm_usage: Optional[Dict[str, Any]] = Field(
        None, description="Aggregated LLM token usage."
    )
    embedding_chunked_calls: int = Field(
        0, ge=0, description="Embedding calls that had to be chunked."
    )
    retry_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Every application-level retry of the run: kind, attempt, waited_s, at, and optionally stage, session_id, error.",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Row creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class SkillLearningRunResponse(SkillLearningRun):
    """Response schema for a learning run."""

    pass
