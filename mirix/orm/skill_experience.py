"""ORM model for the general session-experience store.

A `SkillExperience` is one transferable lesson distilled from ONE work
session's transcript and later consumed by the procedural skill agent to
create/edit skills. Every experience is either `worth_learning` (an approach
worth repeating) or `worth_avoiding` (a pitfall to avoid), scored by
`importance` and `credibility` in [0,1]. There is no external oracle: both are
derived purely from the conversation content.

Records start `pending`; a curator batch flips them to `consumed` (recording
its batch id in `consumed_by` + the skills the batch changed in
`influenced_skill_ids`) and records what became of each one
(`disposition`): `used` when a skill write cited it (`cited_skill_ids`), `covered`,
`not_worth`, `deferred`, or `unaddressed` when it was closed out after its
third attempt. Consumed experiences that are `deferred` or `not_worth` form the
experience pool of their (organization, user, meta agent) and write scope
(`pool_scope`) until they are retired (`pool_retired_at`): a later batch shows
the most similar of them again as earlier lessons. A new experience that
recurred a lesson still in the pool is merged into it (retired `merged` as it
enters), so one lesson is one pool row. `superseded` stays reserved
for pending lessons that later evidence replaced; the curator never uses it.

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
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.constants import MAX_EMBEDDING_DIM
from mirix.orm.custom_columns import CommonVector
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
        doc="JSON string of the in-conversation signal: {quote, signal_type}, "
        "plus the session's outcome {session_outcome, outcome_evidence} when the "
        "distiller stated it (kept here, not in columns of its own, so older "
        "rows and databases need no migration).",
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
    curation_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Curator batches that ended (context_limit / step_cap, or a "
        "non-infrastructure curator failure) without addressing this experience; "
        "after 3 it is closed out as consumed/unaddressed.",
    )

    # --- The experience pool ---------------------------------------------- #
    disposition: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="What became of it once consumed: used | covered | not_worth | "
        "deferred | unaddressed (validated in Pydantic). NULL while pending, and "
        "for experiences consumed before the pool existed.",
    )
    cited_skill_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        default=None,
        doc="Skills whose committed write cited this experience (per-experience "
        "lineage; influenced_skill_ids stays the batch's).",
    )
    pool_scope: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="The curator's write scope when the experience entered the pool; "
        "only batches of the same scope see it.",
    )
    # The similarity vector of title + content, zero-padded to
    # MAX_EMBEDDING_DIM like procedural_memory.retrieval_embedding. Deferred:
    # listing and settling experiences never ship it (and to_pydantic never
    # touches it, which would lazy-load on an async session).
    if settings.mirix_pg_uri_no_default:
        from pgvector.sqlalchemy import Vector

        pool_embedding = mapped_column(
            Vector(MAX_EMBEDDING_DIM), nullable=True, deferred=True
        )
    else:
        pool_embedding = mapped_column(CommonVector, nullable=True, deferred=True)
    pool_embedding_model: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="The embedding model pool_embedding came from, with its dimension "
        "(``model@dim``): a vector of another model or width is embedded again, "
        "never compared.",
    )
    pool_embedding_failures: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Finished batches in which this pool experience's own embedding "
        "failed while the batch's new experiences embedded; it is retired "
        "(unembeddable) at MIRIX_SKILL_EXPERIENCE_POOL_EMBED_MAX_FAILURES.",
    )
    recurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
        doc="Times the same lesson came up (the experience itself counts once).",
    )
    ignored_pushes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Decay counter: +1 per batch that showed it and neither cited nor "
        "covered it (+2 when listed not worth); reset when it recurs.",
    )
    pushed_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        doc="Finished batches that showed it as an earlier lesson.",
    )
    last_pushed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When a batch last showed it."
    )
    pool_retired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, doc="When it left the pool."
    )
    pool_retire_reason: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Why it left the pool: used | covered | ignored | merged (a new "
        "experience that recurred a lesson still in the pool, which counts it) | "
        "unembeddable.",
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
            # The experience pool's scan: agent + disposition.
            Index(
                "ix_skill_experience_agent_disposition",
                "agent_id",
                "disposition",
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
