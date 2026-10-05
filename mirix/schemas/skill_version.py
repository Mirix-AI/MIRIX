"""Pydantic schemas for skill version history.

Skill history is a git-style content-addressed store in Postgres:

* `SkillObject` — one blob per (organization_id, sha256): the zlib-compressed
  canonical bytes of one skill field value. Canonical form: text fields are
  their UTF-8 bytes; JSON fields are
  ``json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))``.
  The sha256 is taken over the canonical bytes BEFORE compression. Objects are
  scoped per organization so deduplication never crosses tenants.
* `SkillVersion` — one "commit" per change of a skill: which skill, its
  monotonically increasing `version_no` (from 1), the parent version, the change
  type, and the sha256 of each of the seven versioned fields
  (`SKILL_VERSIONED_FIELDS`). Embeddings are never versioned; they are
  recomputed from content.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.mirix_base import MirixBase

# The seven skill fields captured by every version, in canonical order. Each
# has a `<field>_sha256` column on skill_version pointing at a skill_object.
SKILL_VERSIONED_FIELDS = (
    "name",
    "entry_type",
    "description",
    "instructions",
    "triggers",
    "examples",
    "filter_tags",
)

# change_type value space (plain String in the ORM): the first version written
# for a pre-existing skill is a 'baseline'; later ones record what happened.
SkillChangeType = Literal["baseline", "create", "edit", "delete", "restore"]

SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


def _sha_field(field_name: str):
    return Field(
        ...,
        pattern=SHA256_HEX_PATTERN,
        description=f"sha256 (hex) of the canonical bytes of `{field_name}` — a skill_object key.",
    )


class SkillObject(MirixBase):
    """One content-addressed, zlib-compressed skill field value."""

    organization_id: str = Field(
        ..., description="Owning organization (objects never dedupe across tenants)."
    )
    sha256: str = Field(
        ...,
        pattern=SHA256_HEX_PATTERN,
        description="sha256 of the canonical, uncompressed bytes.",
    )
    content_zlib: bytes = Field(..., description="zlib-compressed canonical bytes.")
    raw_length: int = Field(
        ..., ge=0, description="Length in bytes of the uncompressed canonical content."
    )
    created_at: Optional[datetime] = Field(
        None, description="When the object was first written."
    )
    last_referenced_at: Optional[datetime] = Field(
        None,
        description="Last time a version write referenced this object (refreshed on conflict).",
    )


class SkillVersionBase(MirixBase):
    __id_prefix__ = "skver"


class SkillVersion(SkillVersionBase):
    """One version ("commit") of a skill."""

    id: Optional[str] = Field(
        None, description="Unique identifier for this version row."
    )
    organization_id: str = Field(..., description="The owning organization id.")
    user_id: str = Field(..., description="The user that owns the skill.")
    skill_id: str = Field(
        ..., description="The procedural_memory row this version belongs to."
    )
    version_no: int = Field(
        ..., ge=1, description="Per-skill version number, starting at 1."
    )
    parent_version_id: Optional[str] = Field(
        None, description="The previous version (None for the first)."
    )
    change_type: SkillChangeType = Field(
        ..., description="baseline | create | edit | delete | restore."
    )
    name_sha256: str = _sha_field("name")
    entry_type_sha256: str = _sha_field("entry_type")
    description_sha256: str = _sha_field("description")
    instructions_sha256: str = _sha_field("instructions")
    triggers_sha256: str = _sha_field("triggers")
    examples_sha256: str = _sha_field("examples")
    filter_tags_sha256: str = _sha_field("filter_tags")
    run_id: Optional[str] = Field(
        None,
        description="Curator batch id (`{learning_run_id}:b{i}`) that made the change, if any.",
    )
    actor_kind: Optional[str] = Field(
        None,
        description="Who made the change (e.g. 'curator', 'api', 'backfill', 'restore').",
    )
    reason: Optional[str] = Field(
        None, description="Why (required for curator deletes)."
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="When the version was written."
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp (versions are immutable)."
    )


class SkillVersionResponse(SkillVersion):
    """Response schema for a skill version."""

    pass
