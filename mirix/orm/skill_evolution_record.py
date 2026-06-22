import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.skill_evolution_record import (
    SkillEvolutionRecord as PydanticSkillEvolutionRecord,
)
from mirix.settings import settings

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class SkillEvolutionRecord(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """
    One distilled success/failure observation produced per graded round (C1)
    and consumed every N rounds by the curator (C3).

    The store is the durable hand-off between distiller and curator. Records
    start `pending`; an evolution run flips them to `consumed` (recording its
    run id in `consumed_by`), or later evidence flips them to `superseded`
    (soft-delete; excluded from `list_pending`).

    Inherited (do NOT redeclare):
      - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
      - organization_id                          -- OrganizationMixin
      - user_id                                  -- UserMixin
      - agent_id (FK agents.id ON DELETE CASCADE) -- AgentMixin

    `record_type` and `status` are plain Strings (NOT pg ENUMs); their value
    spaces are Literal-validated in `schemas/skill_evolution_record.py`, so
    adding a value never needs a DB migration.
    """

    __tablename__ = "skill_evolution_record"
    __pydantic_model__ = PydanticSkillEvolutionRecord

    # Override the bare SqlalchemyBase PK to attach a prefixed default, matching
    # the `proc-`/`ats-` convention on sibling tables.
    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"sevr-{uuid.uuid4()}",
        doc="Unique ID for this skill-evolution record.",
    )

    day: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="MetaClaw day this record was distilled from (e.g. 'day03').",
    )
    round_id: Mapped[str] = mapped_column(
        String, nullable=False, doc="Round identifier within the day (e.g. 'r4')."
    )
    round_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Monotonic round index within the day; used by the watermark.",
    )
    record_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
        doc="'success' or 'failure' (validated in Pydantic, not a pg ENUM).",
    )
    title: Mapped[str] = mapped_column(
        String, nullable=False, doc="Short headline for the observation."
    )
    # description/detail are NOT NULL (with a server-side '' default for the
    # migration path) because the pydantic full schema requires non-null strings;
    # leaving them nullable would let a NULL row break to_pydantic()/list_pending.
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="One-paragraph description of what happened.",
    )
    detail: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Actionable detail: what_worked, or root_cause + what_to_avoid.",
    )
    evidence_round_ids: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        doc="Round ids cited as evidence for this record.",
    )
    quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Distiller-assigned quality in [0,1]; ranking only.",
    )
    generality: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Distiller-assigned generality in [0,1]; ranking only.",
    )
    status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
        doc="Lifecycle: pending | consumed | superseded (validated in Pydantic).",
    )
    consumed_by: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Evolution-run id that consumed this record."
    )
    influenced_skill_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Skill ids this record influenced post-evolution (lineage).",
    )

    # Indexes mirror the procedural_memory pg/sqlite guard pattern: an explicit
    # `is not None` filter (not `filter(None, ...)`) because an un-attached
    # Index is falsy until bound.
    __table_args__ = tuple(
        item
        for item in [
            # Primary access pattern: list_pending(agent_id, status, round_index).
            Index(
                "ix_skill_evolution_record_agent_status",
                "agent_id",
                "status",
                "round_index",
            ),
            (
                Index("ix_skill_evolution_record_organization_id", "organization_id")
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
