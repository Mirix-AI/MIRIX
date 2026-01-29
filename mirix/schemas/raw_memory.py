"""
Pydantic schemas for raw task memory.

Raw memories store unprocessed task context without LLM extraction.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.schemas.embedding_config import EmbeddingConfig
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

    context_embedding: Optional[List[float]] = Field(None, description="The embedding of the context")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used for this memory"
    )

    @field_validator("context_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to MAX_EMBEDDING_DIM."""
        import numpy as np

        from mirix.constants import MAX_EMBEDDING_DIM

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(
                np_embedding,
                (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]),
                mode="constant",
            )
            return padded_embedding.tolist()
        return embedding

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
        context_embedding: Optional embedding of the context (computed by manager)
        embedding_config: Optional embedding configuration used
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

    # Embedding fields (set by manager during creation)
    context_embedding: Optional[List[float]] = Field(None, description="The embedding of the context")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used for this memory"
    )

    @field_validator("context_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to MAX_EMBEDDING_DIM."""
        import numpy as np

        from mirix.constants import MAX_EMBEDDING_DIM

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(
                np_embedding,
                (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]),
                mode="constant",
            )
            return padded_embedding.tolist()
        return embedding


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
    context_embedding: Optional[List[float]] = Field(
        None,
        description="The embedding of the context (regenerated on context update)",
    )
    embedding_config: Optional[EmbeddingConfig] = Field(
        None,
        description="The embedding configuration",
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


class TimeRangeFilter(BaseModel):
    """Time range filter for searching raw memories."""

    created_at_gte: Optional[datetime] = Field(None, description="Filter memories created at or after this time")
    created_at_lte: Optional[datetime] = Field(None, description="Filter memories created at or before this time")
    occurred_at_gte: Optional[datetime] = Field(None, description="Filter memories that occurred at or after this time")
    occurred_at_lte: Optional[datetime] = Field(
        None, description="Filter memories that occurred at or before this time"
    )
    updated_at_gte: Optional[datetime] = Field(None, description="Filter memories updated at or after this time")
    updated_at_lte: Optional[datetime] = Field(None, description="Filter memories updated at or before this time")


class SearchRawMemoryRequest(BaseModel):
    """Request model for searching raw memories."""

    filter_tags: Optional[Dict[str, Any]] = Field(
        None,
        description="AND filter on top-level keys. Scope is automatically set based on authenticated client.",
    )
    sort: str = Field(
        "-updated_at",
        description="Sort field and direction. Options: updated_at, -updated_at, created_at, -created_at, occurred_at, -occurred_at",
    )
    cursor: Optional[str] = Field(
        None,
        description="Opaque Base64-encoded cursor for pagination (contains sort field value + ID)",
    )
    time_range: Optional[TimeRangeFilter] = Field(
        None, description="Time range filters for created_at, occurred_at, or updated_at"
    )
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results to return (default: 10, max: 100)")


class SearchRawMemoryResponse(BaseModel):
    """Response model for searching raw memories."""

    items: List[RawMemoryItem] = Field(..., description="List of matching raw memories")
    cursor: Optional[str] = Field(
        None,
        description="Next opaque cursor for pagination (Base64-encoded JSON with sort field value + ID of last item). Null if no more pages.",
    )
    count: int = Field(..., description="Number of items returned")
