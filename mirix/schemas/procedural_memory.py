from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.utils import get_utc_time
from mirix.schemas.embedding_config import EmbeddingConfig

class ProceduralMemoryItemBase(MirixBase):
    """
    Base schema for storing procedural knowledge (e.g., workflows, methods).
    """
    __id_prefix__ = "proc_item"
    entry_type: str = Field(..., description="Category (e.g., 'workflow', 'guide', 'script')")
    summary: str = Field(..., description="Short descriptive text about the procedure")
    steps: List[str] = Field(..., description="Step-by-step instructions as a list of strings")
    tree_path: List[str] = Field(..., description="Hierarchical categorization path as an array of strings (e.g., ['workflows', 'development', 'testing'])")

class ProceduralMemoryItem(ProceduralMemoryItemBase):
    """
    Full procedural memory item schema, with database-related fields.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the procedural memory item")
    user_id: str = Field(..., description="The id of the user who generated the procedure")
    created_at: datetime = Field(default_factory=get_utc_time, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    organization_id: str = Field(..., description="The unique identifier of the organization")
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    steps_embedding: Optional[List[float]] = Field(None, description="The embedding of the steps")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

    # need to validate both steps_embedding and summary_embedding to ensure they are the same size
    @field_validator("summary_embedding", "steps_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class ProceduralMemoryItemUpdate(MirixBase):
    """Schema for updating an existing procedural memory item."""
    id: str = Field(..., description="Unique ID for this procedural memory entry")
    entry_type: Optional[str] = Field(None, description="Category (e.g., 'workflow', 'guide', 'script')")
    summary: Optional[str] = Field(None, description="Short descriptive text")
    steps: Optional[List[str]] = Field(None, description="Step-by-step instructions as a list of strings")
    tree_path: Optional[List[str]] = Field(
        None, 
        description="Hierarchical categorization path as an array of strings (e.g., ['workflows', 'development', 'testing'])"
    )
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Arbitrary additional metadata")
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    steps_embedding: Optional[List[float]] = Field(None, description="The embedding of the event")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

class ProceduralMemoryItemResponse(ProceduralMemoryItem):
    """Response schema for procedural memory item."""
    pass
