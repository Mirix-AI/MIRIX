from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import get_utc_time

class ResourceMemoryItemBase(MirixBase):
    """
    Base schema for resource memory items - storing docs, user files, references, etc.
    """
    __id_prefix__ = "res_item"
    title: str = Field(..., description="Short name/title of the resource")
    summary: str = Field(..., description="Short description or summary of the resource")
    resource_type: str = Field(..., description="File type or format (e.g. 'doc', 'markdown', 'pdf_text')")
    content: str = Field(..., description="Full or partial text content of the resource")
    tree_path: List[str] = Field(..., description="Hierarchical categorization path as an array of strings (e.g., ['documents', 'work', 'projects'])")

class ResourceMemoryItem(ResourceMemoryItemBase):
    """
    Full schema for resource memory items with DB fields.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the resource memory item")
    user_id: str = Field(..., description="The id of the user who generated the resource")
    created_at: datetime = Field(default_factory=get_utc_time, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    organization_id: str = Field(..., description="The unique identifier of the organization")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata (tags, creation date, etc.)")

    @field_validator("summary_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class ResourceMemoryItemUpdate(MirixBase):
    """Schema for updating an existing resource memory item."""
    id: str = Field(..., description="Unique ID for this resource memory entry")
    title: Optional[str] = Field(None, description="Short name/title of the resource")
    summary: Optional[str] = Field(None, description="Short description or summary of the resource")
    resource_type: Optional[str] = Field(None, description="File type/format (e.g. 'doc', 'markdown')")
    content: Optional[str] = Field(None, description="Full or partial text content")
    tree_path: Optional[List[str]] = Field(
        None, 
        description="Hierarchical categorization path as an array of strings (e.g., ['documents', 'work', 'projects'])"
    )
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Arbitrary additional metadata")

class ResourceMemoryItemResponse(ResourceMemoryItem):
    """Response schema for resource memory item with additional fields if needed."""
    pass
