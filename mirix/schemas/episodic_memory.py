from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import Field, field_validator
from mirix.constants import MAX_EMBEDDING_DIM

from mirix.schemas.mirix_base import MirixBase
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.utils import get_utc_time


class EpisodicEventBase(MirixBase):
    """
    Base schema for episodic memory events containing common fields.
    """

    __id_prefix__ = "ep_mem"
    event_type: str = Field(..., description="Type/category of the episodic event (e.g., user_message, inference, system_notification)")
    summary: str = Field(..., description="Short textual summary of the event")
    details: str = Field(..., description="Detailed description or text for the event")
    actor: str = Field(..., description="The actor who generated the event (user or assistant)")
    tree_path: List[str] = Field(..., description="Hierarchical categorization path as an array of strings (e.g., ['work', 'projects', 'ai-research'])")

class EpisodicEventForLLM(EpisodicEventBase):
    """
    Schema for creating a new episodic memory record.
    """
    # TODO: make `occurred_at` optional
    occurred_at: str = Field(..., description="When the event happened (it should be mentioned in the user's response and it should be in the format of 'YYYY-MM-DD HH:MM:SS')")

class EpisodicEvent(EpisodicEventBase):
    """
    Representation of a single episodic memory event in the system.

    Additional Parameters:
        id (str): Unique identifier for this memory item
        occurred_at (datetime): When the event occurred or was recorded
        created_at (datetime): When the memory record was created in the system
        updated_at (Optional[datetime]): Last update timestamp
    """
    id: Optional[str] = Field(None, description="Unique identifier for the episodic event")

    user_id: str = Field(..., description="The id of the user who generated the episodic event")

    occurred_at: datetime = Field(
        default_factory=get_utc_time,
        description="When the event actually happened (recorded or user-labeled)."
    )
    created_at: datetime = Field(
        default_factory=get_utc_time,
        description="Timestamp when this memory record was created"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="When this memory record was last updated"
    )
    last_modify: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": get_utc_time().isoformat(), "operation": "created"},
        description="Last modification info including timestamp and operation type"
    )
    metadata_: Dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata for the event")
    organization_id: str = Field(..., description="Unique identifier of the organization")
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the event")
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

    # need to validate both details_embedding and summary_embedding to ensure they are the same size
    @field_validator("details_embedding", "summary_embedding")
    @classmethod
    def pad_embeddings(cls, embedding: List[float]) -> List[float]:
        """Pad embeddings to `MAX_EMBEDDING_SIZE`. This is necessary to ensure all stored embeddings are the same size."""
        import numpy as np

        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding

class EpisodicEventUpdate(MirixBase):
    """
    Schema for updating an existing episodic memory record.

    All fields (except id) are optional so that only provided fields are updated.
    """
    id: str = Field(..., description="Unique ID for this episodic memory record")
    event_type: Optional[str] = Field(None, description="Type/category of the event")
    summary: Optional[str] = Field(None, description="Short textual summary of the event")
    details: Optional[str] = Field(None, description="Detailed text describing the event")
    tree_path: Optional[List[str]] = Field(
        None, 
        description="Hierarchical categorization path as an array of strings (e.g., ['work', 'projects', 'ai-research'])"
    )
    metadata_: Optional[Dict[str, Any]] = Field(None, description="Any additional metadata")
    organization_id: Optional[str] = Field(None, description="Unique identifier of the organization")
    occurred_at: Optional[datetime] = Field(None, description="If the event's time is updated")
    updated_at: datetime = Field(
        default_factory=get_utc_time,
        description="Timestamp when this memory record was last updated"
    )
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type"
    )
    summary_embedding: Optional[List[float]] = Field(None, description="The embedding of the summary")
    details_embedding: Optional[List[float]] = Field(None, description="The embedding of the event")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the event")

