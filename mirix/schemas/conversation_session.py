"""Pydantic schemas for the conversation-session lifecycle store.

A `ConversationSession` is the per-(organization, user, session_id) lifecycle
row for the Conversation Message Store: when the session started and last
received a message, whether it has ended (explicitly, or by idling past
`MIRIX_SESSION_IDLE_TIMEOUT_SECONDS`), and the bookkeeping the skill learner
needs — distill failures and backoff, reopen count, and
storage-cap truncation totals.

"Distilled" is NOT a session-level flag: it is tracked per message row
(`conversation_message.distilled_at`), so a reopened session's new rows stay
pending while its earlier rows remain distilled.

`session_tag` is fixed by the first write; later writes carrying another
tag keep the first one and get a warning. NULL is a legacy value for sessions
backfilled from pre-tag rows.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, field_validator

from mirix.constants import VALID_SESSION_TAGS
from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.message import _validate_session_id
from mirix.schemas.mirix_base import MirixBase

# close_reason value space: 'explicit' (POST /memory/sessions/{id}/close) or
# 'idle' (materialized by the idle-timeout check). Plain String in the ORM.
SessionCloseReason = Literal["explicit", "idle"]


def _validate_session_tag(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value not in VALID_SESSION_TAGS:
        raise ValueError(
            f"session_tag must be one of {VALID_SESSION_TAGS} or None; got {value!r}"
        )
    return value


class ConversationSessionBase(MirixBase):
    """Shared, user-facing fields for a conversation session."""

    __id_prefix__ = "convsess"

    session_id: str = Field(..., description="The external session identifier.")
    session_tag: Optional[str] = Field(
        None,
        description="'task' | 'conversation', fixed by the first write; NULL only for legacy backfilled sessions.",
    )
    agent_id: Optional[str] = Field(
        None,
        description="The meta memory agent that wrote this session; its config drives distillation and curation.",
    )

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        validated = _validate_session_id(value)
        if validated is None:
            raise ValueError("session_id is required for a conversation session")
        return validated

    @field_validator("session_tag")
    @classmethod
    def _check_session_tag(cls, value: Optional[str]) -> Optional[str]:
        return _validate_session_tag(value)


class ConversationSessionCreate(ConversationSessionBase):
    """Create schema — the owner ids and the first write's timestamp."""

    user_id: str = Field(..., description="The id of the user that owns this session.")
    organization_id: str = Field(..., description="The owning organization id.")
    first_message_at: datetime = Field(
        ..., description="created_at of the session's first message."
    )
    last_message_at: datetime = Field(
        ..., description="created_at of the session's latest message."
    )
    message_count: int = Field(
        0, ge=0, description="Number of stored messages in the session."
    )


class ConversationSession(ConversationSessionBase):
    """Full conversation-session schema, with database-related fields."""

    id: Optional[str] = Field(
        None, description="Unique identifier for this session row."
    )
    user_id: str = Field(..., description="The id of the user that owns this session.")
    organization_id: str = Field(..., description="The owning organization id.")
    first_message_at: datetime = Field(
        ..., description="created_at of the session's first message."
    )
    last_message_at: datetime = Field(
        ..., description="created_at of the session's latest message."
    )
    message_count: int = Field(
        0, ge=0, description="Number of stored messages in the session."
    )
    closed_at: Optional[datetime] = Field(
        None,
        description="When the session ended (explicit close, or last_message_at + idle timeout).",
    )
    close_reason: Optional[SessionCloseReason] = Field(
        None, description="'explicit' or 'idle'."
    )
    reopen_count: int = Field(
        0, ge=0, description="How many times a write reopened this ended session."
    )
    distill_failures: int = Field(
        0,
        ge=0,
        description="Counted distill failures (truncation, parse, timeout); infrastructure errors excluded.",
    )
    distill_failed_at: Optional[datetime] = Field(
        None,
        description="Set once distill_failures reaches the limit; the session is no longer selected.",
    )
    distill_retry_after: Optional[datetime] = Field(
        None,
        description="Backoff after an infrastructure error; before it the session is neither counted nor claimed.",
    )
    distill_infra_failures: int = Field(
        0,
        ge=0,
        description="Consecutive infrastructure failures; the backoff doubles with each (1 min .. 1 h). Not distill failures.",
    )
    truncated_message_count: int = Field(
        0, ge=0, description="Messages truncated by the storage cap."
    )
    truncated_chars: int = Field(
        0, ge=0, description="Characters removed by the storage cap."
    )
    agent_context: Optional[str] = Field(
        None,
        description="How the agent's prompt was built (its prompt template or setup), "
        "as the client attached it to the session; not a conversation turn.",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Row creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class ConversationSessionResponse(ConversationSession):
    """Response schema for a conversation session."""

    pass
