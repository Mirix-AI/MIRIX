"""ORM model for content-addressed skill field values.

A `SkillObject` stores one zlib-compressed canonical skill-field value, keyed
by (organization_id, sha256-of-uncompressed-bytes). Versions in
`skill_version` reference objects by hash, so an edit that only changes
`instructions` re-uses the other six fields' objects. Writes are
`INSERT ... ON CONFLICT (organization_id, sha256) DO UPDATE SET
last_referenced_at = now()`; unreferenced objects are removed when a skill is
hard-deleted.

This table deliberately does NOT use SqlalchemyBase: it has a natural composite
primary key and none of the id / updated_at / is_deleted / created_by columns.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.base import Base
from mirix.schemas.skill_version import SkillObject as PydanticSkillObject


class SkillObject(Base):
    """One content-addressed, compressed skill field value."""

    __tablename__ = "skill_object"
    __pydantic_model__ = PydanticSkillObject

    organization_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("organizations.id"),
        primary_key=True,
        doc="Owning organization; objects are never shared across tenants.",
    )
    sha256: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        doc="sha256 hex of the canonical bytes before compression.",
    )
    content_zlib: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, doc="zlib-compressed canonical bytes."
    )
    raw_length: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="Uncompressed length in bytes."
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), doc="First write."
    )
    last_referenced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        doc="Last version write that referenced it.",
    )

    def to_pydantic(self) -> PydanticSkillObject:
        return PydanticSkillObject.model_validate(self)
