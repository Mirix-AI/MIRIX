"""Pydantic schemas for the general session-experience store (Goal 2).

A `SkillExperience` is one transferable lesson distilled from a single work
session's transcript. Each is either `worth_learning` (an approach worth
repeating) or `worth_avoiding` (a pitfall to avoid), scored by `importance` and
`credibility` in [0,1]. Records start `pending`, are `consumed` by a skill
evolution run, or are `superseded` when later evidence overrides them.

Mirrors `schemas/skill_evolution_record.py`: a Base with the user-facing fields,
a full schema with DB columns, an Update schema, and a Response alias. Enums are
`Literal`s (validated here in Pydantic), NOT pg ENUMs — the ORM column is a
plain String, so adding a value never requires a DB migration. `importance` and
`credibility` are clamped to [0,1] by validators (garbage -> 0.0), mirroring
`_clamp01` in skill_session_distiller.
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

# Length caps. Generous — these hold distilled prose, not transcripts — but
# bounded so a pathological distiller output can't bloat the DB or the curator
# payload.
SKILL_EXPERIENCE_MAX_TITLE_LEN = 256
SKILL_EXPERIENCE_MAX_CONTENT_LEN = 8192
SKILL_EXPERIENCE_MAX_EVIDENCE_LEN = 2048


def _clamp01(value, default: float = 0.0) -> float:
    """Coerce a distiller-reported score into [0.0, 1.0]; default on garbage.

    Kept identical in spirit to `skill_session_distiller._clamp01` so the
    validator and the distiller agree on out-of-range/garbage handling.
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
        description="JSON string of the in-conversation signal: {quote, signal_type}.",
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
    updated_at: datetime = Field(
        default_factory=get_utc_time, description="Update timestamp."
    )


class SkillExperienceResponse(SkillExperience):
    """Response schema for a session-experience."""

    pass
