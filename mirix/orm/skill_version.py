"""ORM model for skill version history.

A `SkillVersion` is one "commit" of a skill: its per-skill `version_no`
(1, 2, ...; unique per skill), parent version, change type, provenance
(curator batch `run_id`, `actor_kind`, `reason`) and the sha256 of each of the
seven versioned fields, which key rows in `skill_object`. Rebuilding version n
means reading those seven objects back.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin

`skill_id` cascades on delete: a hard-deleted skill takes its whole history
with it (erasure); soft-deleted skills keep theirs and can be restored.
`change_type` is a plain String validated in `schemas/skill_version.py`. No
relationships are declared (history rows are read in bulk by skill id).
"""

import uuid
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.skill_version import SkillVersion as PydanticSkillVersion


class SkillVersion(SqlalchemyBase, OrganizationMixin, UserMixin):
    """One version of a procedural-memory skill."""

    __tablename__ = "skill_version"
    __pydantic_model__ = PydanticSkillVersion

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"skver-{uuid.uuid4()}",
        doc="Unique ID for this version row.",
    )
    skill_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("procedural_memory.id", ondelete="CASCADE"),
        nullable=False,
        doc="The skill (procedural_memory.id) this version belongs to.",
    )
    version_no: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Per-skill version number from 1."
    )
    parent_version_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Previous version id (NULL for version 1)."
    )
    change_type: Mapped[str] = mapped_column(
        String, nullable=False, doc="baseline | create | edit | delete | restore."
    )
    name_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_type_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    description_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    instructions_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    triggers_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    examples_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    filter_tags_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
        doc="Curator batch id `{learning_run_id}:b{i}`, when a curator made the change.",
    )
    actor_kind: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Who made the change."
    )
    reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, doc="Why the change was made."
    )

    __table_args__ = (
        UniqueConstraint(
            "skill_id", "version_no", name="uq_skill_version_skill_version_no"
        ),
        Index("ix_skill_version_org_skill", "organization_id", "skill_id"),
        Index("ix_skill_version_run_id", "run_id"),
    )
