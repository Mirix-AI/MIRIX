from datetime import datetime
from typing import Dict, Optional, Any, List

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import get_utc_time


class KnowledgeVaultItemBase(MirixBase):
    """
    Base schema for knowledge vault items containing common fields.
    """
    __id_prefix__ = "kv_item"
    entry_type: str = Field(..., description="Category (e.g., 'credential', 'bookmark', 'api_key')")
    source: str = Field(..., description="Information on who/where it was provided")
    sensitivity: str = Field(..., description="Data sensitivity level ('low', 'medium', 'high')")
    secret_value: str = Field(..., description="The actual credential or data value")
    caption: str = Field(..., description="Description of the knowledge vault item (e.g. 'API key for OpenAI Service')")


class KnowledgeVaultItem(KnowledgeVaultItemBase):
    """
    Representation of a knowledge vault item for storing credentials, bookmarks, etc.
    
    Additional Parameters:
        id (str): Unique ID for this knowledge vault entry.
        created_at (datetime): Creation timestamp.
        updated_at (Optional[datetime]): Last update timestamp.
    """
    id: Optional[str] = Field(None, description="Unique identifier for the knowledge vault item")
    user_id: str = Field(..., description="The id of the user who generated the knowledge vault item")
    created_at: datetime = Field(default_factory=get_utc_time, description="The creation date of the knowledge vault item")
    updated_at: Optional[datetime] = Field(None, description="The last update date of the knowledge vault item")
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")
    organization_id: str = Field(..., description="The unique identifier of the organization")
    caption_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

    # need to validate both details_embedding and summary_embedding to ensure they are the same size
    @field_validator("caption_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class KnowledgeVaultItemCreate(KnowledgeVaultItemBase):
    """
    Schema for creating a new knowledge vault item.
    
    Inherits all required fields from KnowledgeVaultItemBase.
    """
    pass


class KnowledgeVaultItemUpdate(MirixBase):
    """
    Schema for updating an existing knowledge vault item.
    
    All fields (except id) are optional so that only provided fields are updated.
    """
    id: str = Field(..., description="Unique ID for this knowledge vault entry")
    entry_type: Optional[str] = Field(None, description="Category (e.g., 'credential', 'bookmark', 'api_key')")
    source: Optional[str] = Field(None, description="Information on who/where it was provided")
    sensitivity: Optional[str] = Field(None, description="Data sensitivity level ('low', 'medium', 'high')")
    secret_value: Optional[str] = Field(None, description="The actual credential or data value")
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Arbitrary additional metadata")
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization")
    updated_at: datetime = Field(default_factory=get_utc_time, description="The update date")
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    caption_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

class KnowledgeVaultItemResponse(KnowledgeVaultItem):
    """
    Response schema for knowledge vault items with additional fields that might be needed by the API.
    """
    pass
