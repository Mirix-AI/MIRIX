"""ORM model for the general session-experience store (Goal 2).

A `SkillExperience` is one transferable lesson distilled from ONE work
session's transcript (Goal-2 distillation) and later consumed by the procedural
skill agent to create/edit skills (Goal-3 evolution).

Unlike the MetaClaw-specific `SkillEvolutionRecord` (day/round semantics,
success/failure ordering), this store is general: every experience is either
`worth_learning` (an approach worth repeating) or `worth_avoiding` (a pitfall to
avoid), scored by `importance` and `credibility` in [0,1]. There is NO external
oracle — both are derived purely from the conversation content.

Records start `pending`; an evolution run flips them to `consumed` (recording
its run id in `consumed_by` + the skills it influenced in `influenced_skill_ids`),
or later evidence flips them to `superseded`.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin
  - agent_id (FK agents.id ON DELETE CASCADE) -- AgentMixin

`experience_type` and `status` are plain Strings (NOT pg ENUMs); their value
spaces are Literal-validated in `schemas/skill_experience.py`, so adding a value
never needs a DB migration.
"""

import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, Float, Index, String, Text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.skill_experience import (
    SkillExperience as PydanticSkillExperience,
)
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class SkillExperience(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """One distilled, transferable experience from a single work session."""

    __tablename__ = "skill_experience"
    __pydantic_model__ = PydanticSkillExperience

    # Override the bare SqlalchemyBase PK to attach a prefixed default, matching
    # the `proc-`/`sevr-` convention on sibling tables.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"sexp-{uuid.uuid4()}",
        doc="Unique ID for this skill-experience.",
    )

    session_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="Provenance: the session this experience was distilled from.",
    )
    experience_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="'worth_learning' or 'worth_avoiding' (validated in Pydantic, not a pg ENUM).",
    )
    title: Mapped[str] = mapped_column(
        String, nullable=False, doc="Short headline for the experience."
    )
    # content/evidence are NOT NULL (with a server-side '' default for the
    # migration path) because the pydantic full schema requires non-null strings;
    # leaving them nullable would let a NULL row break to_pydantic()/list.
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="The transferable lesson: when-to-apply (learn) or how-to-avoid (avoid).",
    )
    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="How impactful/worth-acting-on the lesson is, in [0,1].",
    )
    credibility: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="How well-grounded in a direct signal the lesson is, in [0,1].",
    )
    evidence: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="JSON string of the in-conversation signal: {quote, signal_type}.",
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        doc="Lifecycle: pending | consumed | superseded (validated in Pydantic).",
    )
    consumed_by: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Evolution-run id that consumed this experience."
    )
    influenced_skill_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Skill ids this experience influenced post-evolution (lineage).",
    )

    # Indexes mirror the procedural_memory pg/sqlite guard pattern: an explicit
    # `is not None` filter (not `filter(None, ...)`) because an un-attached
    # Index is falsy until bound.
    __table_args__ = tuple(
        item
        for item in [
            # Primary access pattern: list_experiences(agent_id, status).
            Index(
                "ix_skill_experience_agent_status",
                "agent_id",
                "status",
            ),
            (
                Index("ix_skill_experience_organization_id", "organization_id")
                if settings.mirix_pg_uri_no_default
                else None
            ),
        ]
        if item is not None
    )

    @declared_attr
    def agent(cls) -> Mapped[Optional["Agent"]]:
        return relationship("Agent", lazy="selectin")

    @declared_attr
    def user(cls) -> Mapped[Optional["User"]]:
        return relationship("User", lazy="selectin")

    @declared_attr
    def organization(cls) -> Mapped[Optional["Organization"]]:
        return relationship("Organization", lazy="selectin")
