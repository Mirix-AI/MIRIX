"""Pydantic schemas for the general session-experience store.

A `SkillExperience` is one transferable lesson distilled from a single work
session's transcript. Each is either `worth_learning` (an approach worth
repeating) or `worth_avoiding` (a pitfall to avoid), scored by `importance` and
`credibility` in [0,1]. Its `evidence` JSON holds the lesson's grounding
(`quote`, `signal_type`) and — when the distiller stated it — the outcome of
its session as judged from the transcript alone (`session_outcome`, one of
:data:`SESSION_OUTCOMES`, and `outcome_evidence`), with where the transcript
shows a succeeded / failed outcome (`outcome_source`) and, when it does not,
the outcome the distiller stated (`stated_outcome`); experiences stored before
that have none of these keys. Records start `pending` and are `consumed` by a
curator batch, which records their `disposition`: `used` (a skill
write cited it), `covered`, `not_worth`, `deferred`, or `unaddressed` (closed
out after its third attempt). Consumed `deferred` / `not_worth` experiences
form the experience pool until they are retired (`pool_retired_at`).
`superseded` (later evidence replaced a pending lesson) is never used by the
curator.

This module has a Base with the user-facing fields, a full schema with DB
columns, an Update schema, and a Response alias. Enums are `Literal`s (validated
here in Pydantic), NOT pg ENUMs — the ORM column is a plain String, so adding a
value never requires a DB migration. `importance` and `credibility` are clamped
to [0,1] by validators (garbage -> 0.0).
"""

import math
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.schemas.mirix_base import MirixBase

# experience_type / status value spaces. Kept as Literals so the schema is the
# single source of truth; the ORM stores them as plain strings.
ExperienceType = Literal["worth_learning", "worth_avoiding"]
ExperienceStatus = Literal["pending", "consumed", "superseded"]
# What became of a consumed experience, and why one left the pool.
ExperienceDisposition = Literal["used", "covered", "not_worth", "deferred", "unaddressed"]
PoolRetireReason = Literal["used", "covered", "ignored", "merged", "unembeddable"]
# Whether an experience's session achieved its task, as the distiller judged
# it from the transcript alone (stored in the evidence JSON, `session_outcome`).
SessionOutcome = Literal["succeeded", "failed", "unclear"]
SESSION_OUTCOMES = ("succeeded", "failed", "unclear")

# Length caps. Wide enough that a normal distilled lesson is never
# cut, bounded so a pathological output can't bloat the DB. The distiller keeps
# the head and tail of a field past its cap (with an explicit marker) and
# counts it in the run's distill metrics; it never cuts silently.
SKILL_EXPERIENCE_MAX_TITLE_LEN = 1024
SKILL_EXPERIENCE_MAX_CONTENT_LEN = 65536
SKILL_EXPERIENCE_MAX_EVIDENCE_LEN = 65536


def _clamp01(value, default: float = 0.0) -> float:
    """Coerce a distiller-reported score into [0.0, 1.0]; default on garbage.

    Kept in sync with the distiller so validation and persistence agree on
    out-of-range/garbage handling.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    # NaN slips past BOTH the < 0 and > 1 comparisons (all NaN comparisons are
    # False), which would poison the importance*credibility priority ordering —
    # reject it. (+-inf is fine: it clamps to the bounds below.)
    if math.isnan(f):
        return default
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


class SkillExperienceBase(MirixBase):
    """Shared, user-facing fields for a session-experience."""

    __id_prefix__ = "sexp"

    session_id: str = Field(
        ..., description="Provenance: the session this experience was distilled from."
    )
    experience_type: ExperienceType = Field(
        ...,
        description="'worth_learning' (approach to repeat) or 'worth_avoiding' (pitfall).",
    )
    title: str = Field(
        ...,
        max_length=SKILL_EXPERIENCE_MAX_TITLE_LEN,
        description="Short headline for the experience.",
    )
    content: str = Field(
        default="",
        max_length=SKILL_EXPERIENCE_MAX_CONTENT_LEN,
        description="The transferable lesson: when-to-apply (learn) or how-to-avoid (avoid).",
    )
    importance: float = Field(
        default=0.0,
        description="How impactful/worth-acting-on the lesson is, in [0,1].",
    )
    credibility: float = Field(
        default=0.0,
        description="How well-grounded in a direct signal the lesson is, in [0,1].",
    )
    evidence: str = Field(
        default="",
        max_length=SKILL_EXPERIENCE_MAX_EVIDENCE_LEN,
        description=(
            "JSON string of the in-conversation signal: {quote, signal_type}, plus "
            "the session's outcome {session_outcome, outcome_evidence, "
            "outcome_source, stated_outcome} when the distiller stated it."
        ),
    )
    status: ExperienceStatus = Field(
        default="pending",
        description="Lifecycle: pending -> consumed (by an evolution run) or superseded.",
    )

    @field_validator("importance", "credibility", mode="before")
    @classmethod
    def _clamp_scores(cls, v):
        """Clamp importance/credibility into [0,1]; coerce garbage -> 0.0."""
        return _clamp01(v)


class SkillExperienceCreate(SkillExperienceBase):
    """Create schema — carries the owner ids needed to persist the row."""

    agent_id: str = Field(..., description="The id of the agent that owns this experience.")
    user_id: str = Field(..., description="The id of the user that owns this experience.")
    organization_id: str = Field(..., description="The owning organization id.")


class SkillExperience(SkillExperienceBase):
    """Full experience schema, with database-related fields."""

    id: Optional[str] = Field(None, description="Unique identifier for this experience.")
    agent_id: Optional[str] = Field(
        None, description="The id of the agent that owns this experience."
    )
    user_id: str = Field(..., description="The id of the user that owns this experience.")
    organization_id: str = Field(..., description="The owning organization id.")
    consumed_by: Optional[str] = Field(
        None,
        description="Evolution-run id that consumed this experience (set on mark_consumed).",
    )
    influenced_skill_ids: Optional[List[str]] = Field(
        None, description="Skill ids this experience influenced post-evolution (lineage)."
    )
    curation_attempts: int = Field(
        0,
        ge=0,
        description="Curator batches that returned this experience to pending unaddressed.",
    )
    # The experience pool. The pool vector itself (pool_embedding) is
    # deliberately not part of the schema: it is a deferred column, and
    # reading it here would lazy-load it on an async session.
    disposition: Optional[ExperienceDisposition] = Field(
        None, description="What became of it once consumed (NULL while pending)."
    )
    cited_skill_ids: Optional[List[str]] = Field(
        None, description="Skills whose committed write cited this experience."
    )
    pool_scope: Optional[str] = Field(
        None, description="The curator's write scope when it entered the pool."
    )
    pool_embedding_model: Optional[str] = Field(
        None, description="The embedding model of its pool vector (``model@dim``)."
    )
    pool_embedding_failures: int = Field(
        0,
        ge=0,
        description="Finished batches in which its pool embedding failed.",
    )
    recurrence_count: int = Field(
        1, ge=1, description="Times the same lesson came up (itself counts once)."
    )
    ignored_pushes: int = Field(
        0, ge=0, description="Decay counter of the pool (reset when it recurs)."
    )
    pushed_count: int = Field(
        0, ge=0, description="Finished batches that showed it as an earlier lesson."
    )
    last_pushed_at: Optional[datetime] = Field(
        None, description="When a batch last showed it."
    )
    pool_retired_at: Optional[datetime] = Field(
        None, description="When it left the pool."
    )
    pool_retire_reason: Optional[PoolRetireReason] = Field(
        None, description="Why it left the pool."
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class SkillExperienceUpdate(MirixBase):
    """Schema for updating an existing experience (status transitions + lineage)."""

    id: str = Field(..., description="Unique ID for this experience.")
    status: Optional[ExperienceStatus] = Field(None, description="New lifecycle status.")
    consumed_by: Optional[str] = Field(
        None, description="Evolution-run id that consumed this experience."
    )
    influenced_skill_ids: Optional[List[str]] = Field(
        None, description="Skill ids this experience influenced post-evolution."
    )
    curation_attempts: Optional[int] = Field(
        None, ge=0, description="New curation-attempt count."
    )
    disposition: Optional[ExperienceDisposition] = Field(
        None, description="What became of it once consumed."
    )
    cited_skill_ids: Optional[List[str]] = Field(
        None, description="Skills whose committed write cited it."
    )
    pool_scope: Optional[str] = Field(None, description="Its pool's write scope.")
    pool_embedding_model: Optional[str] = Field(
        None, description="The embedding model of its pool vector."
    )
    pool_embedding_failures: Optional[int] = Field(
        None, ge=0, description="New count of failed pool embeddings."
    )
    recurrence_count: Optional[int] = Field(None, ge=1, description="New recurrence count.")
    ignored_pushes: Optional[int] = Field(None, ge=0, description="New decay counter.")
    pushed_count: Optional[int] = Field(None, ge=0, description="New push count.")
    last_pushed_at: Optional[datetime] = Field(None, description="Last push time.")
    pool_retired_at: Optional[datetime] = Field(None, description="Retirement time.")
    pool_retire_reason: Optional[PoolRetireReason] = Field(
        None, description="Why it left the pool."
    )
    updated_at: datetime = Field(
        default_factory=get_utc_time, description="Update timestamp."
    )


class SkillExperienceResponse(SkillExperience):
    """Response schema for a session-experience."""

    pass
