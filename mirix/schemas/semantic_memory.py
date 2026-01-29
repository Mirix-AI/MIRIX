from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_base import MirixBase


class SemanticMemoryItemBase(MirixBase):
    """
    Base schema for storing semantic memory items (e.g., general knowledge, concepts, facts).
    """

    __id_prefix__ = "sem_item"
    name: str = Field(..., description="The name or main concept/object for the knowledge entry")
    summary: str = Field(..., description="A concise explanation or summary of the concept")
    details: str = Field(..., description="Detailed explanation or additional context for the concept")
    source: str = Field(
        ...,
        description="Reference or origin of this information (e.g., book, article, movie)",
    )


class SemanticMemoryItem(SemanticMemoryItemBase):
    """
    Full semantic memory item schema, including database-related fields.
    """

    id: Optional[str] = Field(None, description="Unique identifier for the semantic memory item")
    agent_id: Optional[str] = Field(None, description="The id of the agent this semantic memory item belongs to")
    client_id: Optional[str] = Field(None, description="The id of the client application that created this item")
    user_id: str = Field(..., description="The id of the user who generated the semantic memory")
    created_at: datetime = Field(default_factory=get_utc_time, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": get_utc_time().isoformat(),
            "operation": "created",
        },
        description="Last modification info including timestamp and operation type",
    )
    organization_id: str = Field(..., description="The unique identifier of the organization")
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the details")
    name_embedding: Optional[List[float]] = Field(None, description="The embedding of the name")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the event"
    )

    # NEW: Filter tags for flexible filtering and categorization
    filter_tags: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom filter tags for filtering and categorization",
        examples=[
            {"project_id": "proj-abc", "session_id": "sess-xyz", "tags": ["important", "work"], "priority": "high"}
        ],
    )

    # need to validate both details_embedding and summary_embedding to ensure they are the same size
    @field_validator("details_embedding", "summary_embedding", "name_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(
                np_embedding,
                (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]),
                mode="constant",
            )
            return padded_embedding.tolist()
        return embedding


class SemanticMemoryItemUpdate(MirixBase):
    """
    Schema for updating an existing semantic memory item.
    """

    id: str = Field(..., description="Unique ID for this semantic memory entry")
    agent_id: Optional[str] = Field(None, description="The id of the agent this semantic memory item belongs to")
    name: Optional[str] = Field(None, description="The name or main concept for the knowledge entry")
    summary: Optional[str] = Field(None, description="A concise explanation or summary of the concept")
    details: Optional[str] = Field(None, description="Detailed explanation or additional context for the concept")
    source: Optional[str] = Field(
        None,
        description="Reference or origin of this information (e.g., book, article, movie)",
    )
    actor: Optional[str] = Field(
        None,
        description="The actor who generated the semantic memory (user or assistant)",
    )
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type",
    )
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the details")
    name_embedding: Optional[List[float]] = Field(None, description="The embedding of the name")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the event"
    )
    filter_tags: Optional[Dict[str, Any]] = Field(
        None, description="Custom filter tags for filtering and categorization"
    )


class SemanticMemoryItemResponse(SemanticMemoryItem):
    """
    Response schema for semantic memory item.
    """

    pass
