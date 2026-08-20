from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator

from mirix.schemas.mirix_base import MirixBase

TRIGGER_TYPE_MAX_LEN = 64
TRIGGER_TYPE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"

# Registered trigger kinds. Adding a new trigger? Add its string here so
# callers can't typo their way into silently creating a parallel bookkeeping
# row. Shared with the manager and with memory_tools.
TRIGGER_TYPE_PROCEDURAL_SKILL = "procedural_skill"
KNOWN_TRIGGER_TYPES = frozenset({TRIGGER_TYPE_PROCEDURAL_SKILL})


def _validate_trigger_type(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"trigger_type must be a string, got {type(value).__name__}")
    if len(value) == 0 or len(value) > TRIGGER_TYPE_MAX_LEN:
        raise ValueError(
            f"trigger_type must be 1..{TRIGGER_TYPE_MAX_LEN} characters, got len={len(value)}"
        )
    import re

    if not re.fullmatch(TRIGGER_TYPE_PATTERN, value):
        raise ValueError(
            f"trigger_type '{value}' does not match pattern {TRIGGER_TYPE_PATTERN}"
        )
    # Membership check closes the "typo creates a parallel bookkeeping row"
    # footgun. Adding a new trigger means adding it to KNOWN_TRIGGER_TYPES —
    # that is the whole point of the registry. The length/pattern checks
    # above stay as a defensive belt for future additions.
    if value not in KNOWN_TRIGGER_TYPES:
        raise ValueError(
            f"trigger_type '{value}' is not in the registry "
            f"{sorted(KNOWN_TRIGGER_TYPES)}. Add it to KNOWN_TRIGGER_TYPES "
            f"in mirix/schemas/agent_trigger_state.py."
        )
    return value


class AgentTriggerStateBase(MirixBase):
    """Shared fields for agent_trigger_state."""

    __id_prefix__ = "ats"

    agent_id: str = Field(..., description="ID of the agent this trigger state belongs to.")
    user_id: str = Field(..., description="ID of the user this trigger state belongs to.")
    trigger_type: str = Field(
        ...,
        max_length=TRIGGER_TYPE_MAX_LEN,
        description="Kind of trigger (e.g. 'procedural_skill').",
    )

    @field_validator("trigger_type")
    @classmethod
    def _check_trigger_type(cls, value: str) -> str:
        return _validate_trigger_type(value)


class AgentTriggerState(AgentTriggerStateBase):
    """Persisted trigger-state row."""

    id: Optional[str] = Field(None, description="Unique identifier for this trigger state row.")
    organization_id: Optional[str] = Field(None, description="Owning organization id.")
    last_fired_at: Optional[datetime] = Field(
        None,
        description="created_at of the message that caused the last fire; lower "
        "bound for 'new sessions since last fire' counts.",
    )
    last_fired_session_id: Optional[str] = Field(
        None,
        max_length=64,
        description="session_id of the message that caused the last fire.",
    )
    last_fired_tied_session_ids: Optional[List[str]] = Field(
        default_factory=list,
        description="session_ids whose first-appearance timestamp equals "
        "last_fired_at. The fire filter uses MIN(created_at) per session_id so "
        "a given session_id counts in exactly one window; this tied set covers "
        "the rare exact-microsecond tie at the watermark.",
    )
    created_at: Optional[datetime] = Field(None, description="Row creation timestamp.")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")
