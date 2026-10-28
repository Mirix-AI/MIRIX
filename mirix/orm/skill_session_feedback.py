"""ORM model of per-session feedback on served skills.

One row per (distilled session, served skill): whether the agent followed the
skill and with what effect, as the distiller judged it from the session's
transcript, and the session's outcome — see
``mirix/schemas/skill_session_feedback.py``. Written in the same transaction
that commits the session's experiences and marks its rows distilled; read by
the curator as each skill's usage record.

Inherited (do NOT redeclare):
  - id, created_at, updated_at, is_deleted  -- SqlalchemyBase + meta mixin
  - organization_id                          -- OrganizationMixin
  - user_id                                  -- UserMixin

``skill_id`` cascades on delete: a hard-deleted skill takes its feedback with
it. ``session_id`` is the external id (no FK: the transcript may be pruned
while the feedback stays). The value columns are plain Strings validated in
the schema.
"""

import uuid
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from mirix.orm.mixins import OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.message import SESSION_ID_MAX_LEN
from mirix.schemas.skill_session_feedback import (
    SkillSessionFeedback as PydanticSkillSessionFeedback,
)


class SkillSessionFeedback(SqlalchemyBase, OrganizationMixin, UserMixin):
    """A served skill's feedback from one distilled session."""

    __tablename__ = "skill_session_feedback"
    __pydantic_model__ = PydanticSkillSessionFeedback

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"sfb-{uuid.uuid4()}",
        doc="Unique ID for this feedback row.",
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="Meta agent that owns the session."
    )
    session_id: Mapped[str] = mapped_column(
        String(SESSION_ID_MAX_LEN), nullable=False, doc="External session id."
    )
    learning_run_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The learning run that distilled the session."
    )
    skill_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("procedural_memory.id", ondelete="CASCADE"),
        nullable=False,
        doc="The served skill.",
    )
    skill_name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="The skill's name in the version served."
    )
    skill_version_no: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, doc="The version served (NULL: not recorded)."
    )
    served_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
        doc="Retrievals of the session that served it.",
    )
    session_outcome: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, doc="succeeded | failed | unclear (NULL: none stated)."
    )
    followed: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="unclear",
        server_default=text("'unclear'"),
        doc="yes | partly | no | unclear.",
    )
    followed_evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    followed_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stated_followed: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    effect: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="unclear",
        server_default=text("'unclear'"),
        doc="helped | no_effect | hurt | unclear.",
    )
    effect_evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    effect_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stated_effect: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    __table_args__ = (
        # The usage record of a skill.
        Index(
            "ix_skill_session_feedback_owner_skill",
            "organization_id",
            "user_id",
            "skill_id",
        ),
        # The skills served in a batch's sessions.
        Index(
            "ix_skill_session_feedback_owner_session",
            "organization_id",
            "user_id",
            "session_id",
        ),
    )
