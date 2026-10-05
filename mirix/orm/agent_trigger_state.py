import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

from mirix.orm.mixins import AgentMixin, OrganizationMixin, UserMixin
from mirix.orm.sqlalchemy_base import SqlalchemyBase
from mirix.schemas.agent_trigger_state import (
    AgentTriggerState as PydanticAgentTriggerState,
)

if TYPE_CHECKING:
    from mirix.orm.agent import Agent
    from mirix.orm.organization import Organization
    from mirix.orm.user import User


class AgentTriggerState(SqlalchemyBase, OrganizationMixin, UserMixin, AgentMixin):
    """
    Per-(agent, user, trigger_type) cursor row of the skill-learning trigger.

    The row is the scope's lock: every trigger claim locks it with
    ``SELECT ... FOR UPDATE`` so concurrent checks of one (meta agent, user)
    serialize. Its fields are diagnostics only — the count of pending sessions
    is always derived live from conversation_session / conversation_message /
    skill_learning_run (see AgentTriggerStateManager.check_and_claim_fire).
    """

    __tablename__ = "agent_trigger_state"
    __pydantic_model__ = PydanticAgentTriggerState

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: f"ats-{uuid.uuid4()}"
    )

    trigger_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="Kind of trigger this row bookkeeps (e.g. 'procedural_skill').",
    )

    last_fired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the scope last fired a skill learning run (diagnostics).",
    )

    last_fired_session_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="session_id whose write/close triggered the last fire (diagnostics).",
    )

    last_fired_tied_session_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        doc="session_ids claimed by the last fired learning run (diagnostics "
        "only; claims are tracked on skill_learning_run.session_ids).",
    )

    __table_args__ = (
        UniqueConstraint(
            "agent_id",
            "user_id",
            "trigger_type",
            name="uq_agent_trigger_state_agent_user_type",
        ),
        Index(
            "ix_agent_trigger_state_agent_user_type",
            "agent_id",
            "user_id",
            "trigger_type",
        ),
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
