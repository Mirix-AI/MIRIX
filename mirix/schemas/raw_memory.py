"""
Pydantic schemas for raw task memory.

Raw memories store unprocessed task context without LLM extraction.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field

from mirix.client.utils import get_utc_time
from mirix.schemas.mirix_base import MirixBase


class RawMemoryItemBase(MirixBase):
    """Base schema for raw task memory."""

    __id_prefix__ = "raw_mem"

    context: str = Field(
        ...,
        description="Raw task context string (unprocessed)",
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        None,
        description="Filter tags for categorization and access control (includes scope)",
        examples=[
            {
                "scope": "CARE",
                "engagement_id": "tsk_9f3c2a",
                "priority": "high",
            }
        ],
    )


class RawMemoryItem(RawMemoryItemBase):
    """
    Full raw memory response schema.

    Represents a complete raw memory record with all database fields including
    timestamps, relationships, and metadata.
    
    Note: Audit fields (_created_by_id, _last_update_by_id) are tracked internally
    in the ORM layer but not exposed in the API response schema, consistent with
    other MIRIX memory types.
    """

    id: str = Field(..., description="Unique identifier (UUIDv7)")
    user_id: str = Field(..., description="User ID this memory belongs to")
    organization_id: str = Field(..., description="Organization ID")

    # Last modification tracking (standard MIRIX pattern)
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": get_utc_time().isoformat(),
            "operation": "created",
        },
        description="Last modification info including timestamp and operation type",
    )

    # Timestamps
    occurred_at: datetime = Field(
        default_factory=get_utc_time,
        description="When the event occurred",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time,
        description="When record was created",
    )
    updated_at: datetime = Field(
        default_factory=get_utc_time,
        description="When record was last updated",
    )


class RawMemoryItemCreate(RawMemoryItemBase):
    """
    Schema for creating a raw memory.

    Args:
        user_id: User ID this memory belongs to
        organization_id: Organization ID
        occurred_at: When the event occurred (defaults to now if omitted)
        id: Unique identifier (server generates UUIDv7 if omitted)
    """

    user_id: str = Field(..., description="User ID")
    organization_id: str = Field(..., description="Organization ID")
    occurred_at: Optional[datetime] = Field(
        None,
        description="When the event occurred (defaults to now)",
    )
    id: Optional[str] = Field(
        None,
        description="Unique identifier (server generates if omitted)",
    )


class RawMemoryItemUpdate(MirixBase):
    """
    Schema for updating a raw memory (used by REST API and service layer).

    All fields are optional - only provided fields will be updated.

    Args:
        context: New context text
        filter_tags: New or updated filter tags
        context_update_type: How to handle context updates ("append" or "replace")
        tags_update_type: How to handle filter_tags updates ("merge" or "replace")
    """

    context: Optional[str] = Field(
        None,
        description="New context text",
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        None,
        description="New or updated filter tags",
    )
    context_update_type: str = Field(
        "replace",
        pattern="^(append|replace)$",
        description="How to handle context updates: 'append' adds to existing, 'replace' overwrites",
    )
    tags_update_type: str = Field(
        "replace",
        pattern="^(merge|replace)$",
        description="How to handle filter_tags updates: 'merge' combines with existing, 'replace' overwrites",
    )
