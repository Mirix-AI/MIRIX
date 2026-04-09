from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_base import MirixBase


class ProceduralMemoryItemBase(MirixBase):
    """
    Base schema for storing procedural knowledge as reusable skills.
    """

    __id_prefix__ = "proc_item"
    name: str = Field(..., description="Short skill identifier (e.g., 'deploy-production')")
    entry_type: str = Field(..., description="Category (e.g., 'workflow', 'guide', 'script')")
    description: str = Field(..., description="Short descriptive text about the skill")
    instructions: str = Field(..., description="Step-by-step instructions as a single string")
    triggers: List[str] = Field(default_factory=list, description="Conditions indicating this skill is relevant")
    examples: List[dict] = Field(default_factory=list, description="Input/output examples for this skill")


class ProceduralMemoryItem(ProceduralMemoryItemBase):
    """
    Full procedural memory item schema, with database-related fields.
    """

    id: Optional[str] = Field(None, description="Unique identifier for the procedural memory item")
    agent_id: Optional[str] = Field(None, description="The id of the agent this procedural memory item belongs to")
    client_id: Optional[str] = Field(None, description="The id of the client application that created this item")
    user_id: str = Field(..., description="The id of the user who generated the procedure")
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
    version: str = Field(default="0.1.0", description="Semantic version of this skill")
    description_embedding: Optional[List[float]] = Field(None, description="The embedding of the description")
    instructions_embedding: Optional[List[float]] = Field(None, description="The embedding of the instructions")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the event"
    )

    # Filter tags for flexible filtering and categorization
    filter_tags: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom filter tags for filtering and categorization",
        examples=[
            {"project_id": "proj-abc", "session_id": "sess-xyz", "tags": ["important", "work"], "priority": "high"}
        ],
    )

    @field_validator("description_embedding", "instructions_embedding")
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


class ProceduralMemoryItemUpdate(MirixBase):
    """Schema for updating an existing procedural memory item."""

    id: str = Field(..., description="Unique ID for this procedural memory entry")
    agent_id: Optional[str] = Field(None, description="The id of the agent this procedural memory item belongs to")
    name: Optional[str] = Field(None, description="Short skill identifier")
    entry_type: Optional[str] = Field(None, description="Category (e.g., 'workflow', 'guide', 'script')")
    description: Optional[str] = Field(None, description="Short descriptive text about the skill")
    instructions: Optional[str] = Field(None, description="Step-by-step instructions as a single string")
    triggers: Optional[List[str]] = Field(None, description="Conditions indicating this skill is relevant")
    examples: Optional[List[dict]] = Field(None, description="Input/output examples for this skill")
    version: Optional[str] = Field(None, description="Semantic version of this skill")
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type",
    )
    instructions_embedding: Optional[List[float]] = Field(None, description="The embedding of the instructions")
    description_embedding: Optional[List[float]] = Field(None, description="The embedding of the description")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the event"
    )

    filter_tags: Optional[Dict[str, Any]] = Field(
        None, description="Custom filter tags for filtering and categorization"
    )


class ProceduralMemoryItemResponse(ProceduralMemoryItem):
    """Response schema for procedural memory item."""

    pass
