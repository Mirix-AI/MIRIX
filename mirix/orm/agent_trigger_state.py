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
    Persists per-(agent, user, trigger_type) bookkeeping for interval-driven
    memory triggers (e.g. "fire a procedural-memory extraction every N sessions").

    Only the last-fire cursor is stored here. The live counter (e.g. "how many
    sessions since the last fire") is derived at read time from the messages
    table via a COUNT(DISTINCT session_id) — the single source of truth for
    session accrual.
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
        doc="Created_at of the message that caused the last fire. Used as the "
        "lower bound when counting new sessions.",
    )

    last_fired_session_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        doc="session_id that was in-progress on the agent when the last fire "
        "claimed. Stored for audit/logging only; the fire filter is MIN-based "
        "and subsumes in-progress-session dedup automatically.",
    )

    last_fired_tied_session_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        doc="session_ids whose MIN(created_at) (first-appearance timestamp) "
        "equalled last_fired_at in the last fire's counted window. The fire "
        "filter uses MIN semantics so each session_id can only contribute to "
        "exactly one window; this tied set handles the exact-microsecond case "
        "where a session's first message commits at the same timestamp as our "
        "watermark but was not visible to our SELECT. The next window lets "
        "such sessions qualify via `first_ts = last_fired_at AND session_id "
        "NOT IN last_fired_tied_session_ids`.",
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
