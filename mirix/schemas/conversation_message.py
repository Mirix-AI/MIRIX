"""Pydantic schemas for the Conversation Message Store.

A `ConversationMessage` is one external conversation turn — a real `user` or
`assistant` message that arrived through the memory-add API carrying a
`session_id`. It is the canonical, learnable record of a conversation and the
single source the procedural-memory (skill) distiller reads.

Mirrors `schemas/skill_experience.py`: a Base with the user-facing fields, a
`Create` schema that carries the owner ids needed to persist, a full schema with
DB columns, and a Response alias. `role` is a `Literal` (validated here in
Pydantic), NOT a pg ENUM — the ORM column is a plain String, so changing the
value space never requires a DB migration. `session_id` reuses the shared
validator from `schemas/message.py` so the conversation store and the agent-loop
`messages` table agree on exactly one session_id format.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import Field, field_validator

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.message import _validate_session_id
from mirix.schemas.mirix_base import MirixBase

# role value space. Kept as a Literal so the schema is the single source of
# truth; the ORM stores it as a plain string.
ConversationRole = Literal["user", "assistant"]

# Length cap on a single turn. Generous — conversation turns can be long — but
# bounded so a pathological caller can't bloat the DB.
CONVERSATION_MESSAGE_MAX_CONTENT_LEN = 65536


class ConversationMessageBase(MirixBase):
    """Shared, user-facing fields for a conversation turn."""

    __id_prefix__ = "convmsg"

    session_id: str = Field(
        ...,
        description="The conversation this turn belongs to (NOT NULL in this store).",
    )
    role: ConversationRole = Field(
        ...,
        description="The real turn role: 'user' or 'assistant'.",
    )
    content: str = Field(
        default="",
        max_length=CONVERSATION_MESSAGE_MAX_CONTENT_LEN,
        description="The verbatim text of this conversation turn.",
    )

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, v):
        """Enforce the shared session_id format; reject None/empty.

        `_validate_session_id` permits None for the agent-loop `messages` table,
        but this store only ever holds session'd turns, so a None/empty value is
        rejected here.
        """
        validated = _validate_session_id(v)
        if validated is None:
            raise ValueError("session_id is required for a conversation message")
        return validated


class ConversationMessageCreate(ConversationMessageBase):
    """Create schema — carries the owner ids needed to persist the row."""

    user_id: str = Field(..., description="The id of the user that owns this turn.")
    organization_id: str = Field(..., description="The owning organization id.")


class ConversationMessage(ConversationMessageBase):
    """Full conversation-message schema, with database-related fields."""

    id: Optional[str] = Field(None, description="Unique identifier for this turn.")
    user_id: str = Field(..., description="The id of the user that owns this turn.")
    organization_id: str = Field(..., description="The owning organization id.")
    distilled_at: Optional[datetime] = Field(
        None,
        description="Set when this session was consumed by a distill round "
        "(NULL = not yet distilled).",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class ConversationMessageResponse(ConversationMessage):
    """Response schema for a conversation message."""

    pass
