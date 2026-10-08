"""Pydantic schemas for the Conversation Message Store.

A `ConversationMessage` is one external conversation turn — a real `user`,
`assistant`, or `tool` message that arrived through the memory-add API carrying
a `session_id`. It is the canonical, learnable record of a conversation and the
single source the procedural-memory (skill) distiller reads. Tool turns matter:
the work-process lessons (a tool error, a retry, a fix that finally worked) live
in tool calls and tool results, so dropping them would blind the distiller to
exactly the worth_avoiding/worth_learning signal it exists to extract.

Mirrors `schemas/skill_experience.py`: a Base with the user-facing fields, a
`Create` schema that carries the owner ids needed to persist, a full schema with
DB columns, and a Response alias. `role` is a `Literal` (validated here in
Pydantic), NOT a pg ENUM — the ORM column is a plain String, so changing the
value space never requires a DB migration. `session_id` reuses the shared
validator from `schemas/message.py` so the conversation store and the agent-loop
`messages` table agree on exactly one session_id format.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.message import _validate_session_id
from mirix.schemas.mirix_base import MirixBase

# role value space. Kept as a Literal so the schema is the single source of
# truth; the ORM stores it as a plain string. 'tool' covers both tool results
# (role: "tool" messages) and serialized assistant tool calls.
ConversationRole = Literal["user", "assistant", "tool"]

# There is no static length cap on a turn: the storage cap is
# derived from the writing meta agent's context window and applied by
# ConversationMessageManager.record_turns (head + tail kept, with an explicit
# truncation marker and `original_text_length` recorded).

# content_parts part types. A text part is {"type": "text", "text": str};
# an image reference is {"type": "image_ref", "file_id": ...} for a stored image
# (plus mime_type/width/height/sha256/bytes) or {"type": "image_ref", "url": ...}
# for a remote image that is recorded but never downloaded.
CONTENT_PART_TEXT = "text"
CONTENT_PART_IMAGE_REF = "image_ref"
CONTENT_PART_TYPES = (CONTENT_PART_TEXT, CONTENT_PART_IMAGE_REF)


def _validate_content_parts(
    parts: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Validate the shape of `content_parts` (None = text-only message)."""
    if parts is None:
        return parts
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError(f"content_parts[{i}] must be an object")
        part_type = part.get("type")
        if part_type not in CONTENT_PART_TYPES:
            raise ValueError(
                f"content_parts[{i}].type must be one of {CONTENT_PART_TYPES}; got {part_type!r}"
            )
        if part_type == CONTENT_PART_TEXT and not isinstance(part.get("text"), str):
            raise ValueError(f"content_parts[{i}] (text) needs a string 'text'")
        has_file_id = isinstance(part.get("file_id"), str) and bool(part["file_id"])
        has_url = isinstance(part.get("url"), str) and bool(part["url"])
        if part_type == CONTENT_PART_IMAGE_REF and not (has_file_id or has_url):
            raise ValueError(f"content_parts[{i}] (image_ref) needs a 'file_id' or a 'url'")
    return parts


def image_placeholder(part: Dict[str, Any]) -> str:
    """The `content` text standing in for one image_ref part."""
    return f"[image: {part.get('file_id') or part.get('url')}]"


def render_content_parts_text(parts: List[Dict[str, Any]]) -> str:
    """The `content` column for a turn with `content_parts`.

    Text parts verbatim and each image as an `[image: <file_id>]` placeholder,
    in order, joined by newlines — so readers of `content` see where the images
    were without the store ever holding image bytes.
    """
    pieces: List[str] = []
    for part in parts:
        if part.get("type") == CONTENT_PART_IMAGE_REF:
            pieces.append(image_placeholder(part))
        else:
            pieces.append(str(part.get("text", "")))
    return "\n".join(pieces)


def content_parts_have_images(parts: Optional[List[Dict[str, Any]]]) -> bool:
    return any(
        isinstance(p, dict) and p.get("type") == CONTENT_PART_IMAGE_REF for p in parts or []
    )


class ConversationMessageBase(MirixBase):
    """Shared, user-facing fields for a conversation turn."""

    __id_prefix__ = "convmsg"

    session_id: str = Field(
        ...,
        description="The conversation this turn belongs to (NOT NULL in this store).",
    )
    role: ConversationRole = Field(
        ...,
        description="The real turn role: 'user', 'assistant', or 'tool'.",
    )
    content: str = Field(
        default="",
        description="The verbatim text of this conversation turn; for a turn with "
        "images, its text with `[image: <file_id>]` placeholders (never base64).",
    )
    content_parts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Ordered text parts + image references; set only for messages "
        "that carry an image (then `content` holds placeholder text).",
    )
    original_text_length: Optional[int] = Field(
        default=None,
        ge=0,
        description="Text length before the storage cap truncated it (None = untruncated).",
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

    @field_validator("content_parts")
    @classmethod
    def _check_content_parts(cls, v):
        return _validate_content_parts(v)


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
    session_tag: Optional[str] = Field(
        None,
        description="Routing tag stamped on the row: 'task' | 'conversation' | "
        "None (legacy).",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="Creation timestamp."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")


class ConversationMessageResponse(ConversationMessage):
    """Response schema for a conversation message."""

    pass
