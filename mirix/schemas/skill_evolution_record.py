"""Pydantic schemas for the skill-evolution record store (C2).

A `SkillEvolutionRecord` is one distilled success/failure observation produced
per graded round by the C1 distiller and consumed, every N rounds, by the C3
curator. The store is the durable hand-off between the two: records start
`pending`, are `consumed` by an evolution run, or are `superseded` (soft-delete)
when later evidence overrides them.

Mirrors `schemas/procedural_memory.py`: a Base with the user-facing fields, a
full schema with DB columns, an Update schema, and a Response alias. Enums are
`Literal`s (validated here in Pydantic), NOT pg ENUMs — the ORM column is a
plain String, so adding a value never requires a DB migration.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import Field

from mirix.client.utils import get_utc_time
from mirix.schemas.mirix_base import MirixBase

# record_type / status value spaces. Kept as Literals so the schema is the
# single source of truth; the ORM stores them as plain strings.
RecordType = Literal["success", "failure"]
RecordStatus = Literal["pending", "consumed", "superseded"]

# Length caps. Generous — these hold distilled prose, not transcripts — but
# bounded so a pathological distiller output can't bloat the DB or the curator
# payload. `detail` is the largest because it carries root_cause + what_to_avoid
# (failure) or what_worked (success).
SKILL_RECORD_MAX_TITLE_LEN = 256
SKILL_RECORD_MAX_DESCRIPTION_LEN = 1024
SKILL_RECORD_MAX_DETAIL_LEN = 8192


class SkillEvolutionRecordBase(MirixBase):
    """Shared, user-facing fields for a skill-evolution record."""

    __id_prefix__ = "sevr"

    day: str = Field(
        ..., description="MetaClaw day this record was distilled from (e.g. 'day03')."
    )
    round_id: str = Field(
        ..., description="Round identifier within the day (e.g. 'r4')."
    )
    round_index: int = Field(
        ..., description="Monotonic round index within the day; used by the watermark."
    )
    record_type: RecordType = Field(
        ...,
        description="Whether this round was distilled as a 'success' or a 'failure'.",
    )
    title: str = Field(
        ...,
        max_length=SKILL_RECORD_MAX_TITLE_LEN,
        description="Short headline for the observation.",
    )
    description: str = Field(
        ...,
        max_length=SKILL_RECORD_MAX_DESCRIPTION_LEN,
        description="One-paragraph description of what happened.",
    )
    detail: str = Field(
        ...,
        max_length=SKILL_RECORD_MAX_DETAIL_LEN,
        description="Actionable detail: what_worked (success) or root_cause + what_to_avoid (failure).",
    )
    evidence_round_ids: List[str] = Field(
        default_factory=list, description="Round ids cited as evidence for this record."
    )
    quality_score: float = Field(
        default=0.0,
        description="Distiller-assigned quality in [0,1]; ranking only, never grants budget.",
    )
    generality: float = Field(
        default=0.0, description="Distiller-assigned generality in [0,1]; ranking only."
    )
    status: RecordStatus = Field(
        default="pending",
        description="Lifecycle: pending -> consumed (by an evolution run) or superseded (soft-delete).",
    )


class SkillEvolutionRecordCreate(SkillEvolutionRecordBase):
    """Create schema — carries the owner ids needed to persist the row."""

    agent_id: str = Field(..., description="The id of the agent that owns this record.")
    user_id: str = Field(..., description="The id of the user that owns this record.")
    organization_id: str = Field(..., description="The owning organization id.")


class SkillEvolutionRecord(SkillEvolutionRecordBase):
    """Full record schema, with database-related fields."""

    id: Optional[str] = Field(None, description="Unique identifier for this record.")
    agent_id: Optional[str] = Field(
        None, description="The id of the agent that owns this record."
    )
    user_id: str = Field(..., description="The id of the user that owns this record.")
    organization_id: str = Field(..., description="The owning organization id.")
    consumed_by: Optional[str] = Field(
        None,
        description="Evolution-run id that consumed this record (set on mark_consumed).",
    )
    influenced_skill_ids: Optional[List[str]] = Field(
        None, description="Skill ids this record influenced post-evolution (lineage)."
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class SkillEvolutionRecordUpdate(MirixBase):
    """Schema for updating an existing record (status transitions + lineage)."""

    id: str = Field(..., description="Unique ID for this record.")
    status: Optional[RecordStatus] = Field(None, description="New lifecycle status.")
    consumed_by: Optional[str] = Field(
        None, description="Evolution-run id that consumed this record."
    )
    influenced_skill_ids: Optional[List[str]] = Field(
        None, description="Skill ids this record influenced post-evolution."
    )
    updated_at: datetime = Field(
        default_factory=get_utc_time, description="Update timestamp."
    )


class SkillEvolutionRecordResponse(SkillEvolutionRecord):
    """Response schema for a skill-evolution record."""

    pass
