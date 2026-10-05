"""Pydantic schemas for per-session feedback on served skills.

When a learning run distills a session, the distiller also judges each skill
MIRIX served to the agent in that session (the retrieval log of the session
says which, and at which version): whether the agent ``followed`` it and what
``effect`` it had, each with a short quote of the transcript. One
``skill_session_feedback`` row per (distilled session, served skill) keeps
that judgement — also when the distiller gave none (both ``unclear``) — with
the session's own outcome, so the curator can see each skill's usage record
(how often it was served, followed, and with what effect and outcome).

Labels never come from outside: every value is MIRIX's own judgement from the
transcript. A ``followed`` value other than ``unclear`` needs a quote found in
the messages the distiller was shown (the agent's own words may show it); an
``effect`` other than ``unclear`` needs a quote from a message of the user or
a tool (the environment) — otherwise it is stored ``unclear`` and the stated
value is kept in ``stated_followed`` / ``stated_effect``.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.mirix_base import MirixBase

#: Whether the agent followed a served skill.
FOLLOWED_VALUES = ("yes", "partly", "no", "unclear")
#: What a served skill did for the session.
EFFECT_VALUES = ("helped", "no_effect", "hurt", "unclear")

SkillFollowed = Literal["yes", "partly", "no", "unclear"]
SkillEffect = Literal["helped", "no_effect", "hurt", "unclear"]


class SkillSessionFeedbackBase(MirixBase):
    __id_prefix__ = "sfb"


class SkillSessionFeedback(SkillSessionFeedbackBase):
    """One served skill's feedback from one distilled session."""

    id: Optional[str] = Field(None, description="Unique identifier of the row.")
    organization_id: Optional[str] = Field(None, description="The owning organization.")
    user_id: str = Field(..., description="The user whose session it was.")
    agent_id: Optional[str] = Field(
        None, description="The meta agent that owns the session (the learning scope)."
    )
    session_id: str = Field(..., description="The external session id.")
    learning_run_id: Optional[str] = Field(
        None, description="The learning run that distilled the session."
    )
    skill_id: str = Field(..., description="The served skill (procedural_memory.id).")
    skill_name: Optional[str] = Field(
        None, description="The skill's name in the version served."
    )
    skill_version_no: Optional[int] = Field(
        None, description="The version served (None: not recorded at serve time)."
    )
    served_count: int = Field(
        1, ge=0, description="How many retrievals of the session served it."
    )
    session_outcome: Optional[str] = Field(
        None, description="The session's outcome as the distiller judged it."
    )
    followed: SkillFollowed = Field(
        "unclear", description="yes | partly | no | unclear."
    )
    followed_evidence: Optional[str] = Field(
        None, description="The quote that shows whether it was followed."
    )
    followed_source: Optional[str] = Field(
        None,
        description="Where the quote was found: user_or_tool, agent_only or not_in_transcript.",
    )
    stated_followed: Optional[str] = Field(
        None, description="What the distiller stated when it was stored as unclear."
    )
    effect: SkillEffect = Field(
        "unclear", description="helped | no_effect | hurt | unclear."
    )
    effect_evidence: Optional[str] = Field(
        None, description="The quote that shows its effect."
    )
    effect_source: Optional[str] = Field(
        None,
        description="Where the quote was found: user_or_tool, agent_only or not_in_transcript.",
    )
    stated_effect: Optional[str] = Field(
        None, description="What the distiller stated when it was stored as unclear."
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="When the row was written."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")
