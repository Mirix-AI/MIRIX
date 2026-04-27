from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from mirix.client.utils import get_utc_time
from mirix.constants import MAX_EMBEDDING_DIM
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_base import MirixBase

# Allowed skill entry_type values. Kept as a set so tests and REST handlers
# can import and reuse the same source of truth.
SKILL_ENTRY_TYPES = frozenset({"workflow", "guide", "script"})

# Length caps for skill fields. They are intentionally generous: real skills
# can be long. The goal is to prevent pathological inputs (e.g. multi-megabyte
# instruction blobs) from hitting the DB and embedding pipeline, not to
# police prose. Revisit if users routinely bump against them.
SKILL_MAX_NAME_LEN = 128
SKILL_MAX_DESCRIPTION_LEN = 1024
SKILL_MAX_INSTRUCTIONS_LEN = 65_536
SKILL_MAX_TRIGGER_LEN = 512
SKILL_MAX_EXAMPLES_COUNT = 50


def _validate_entry_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    if value not in SKILL_ENTRY_TYPES:
        raise ValueError(
            f"Invalid entry_type '{value}'. Must be one of: {sorted(SKILL_ENTRY_TYPES)}."
        )
    return value


def _validate_triggers(triggers: Optional[List[str]]) -> Optional[List[str]]:
    if triggers is None:
        return triggers
    for i, trigger in enumerate(triggers):
        if not isinstance(trigger, str):
            raise ValueError(f"triggers[{i}] must be a string, got {type(trigger).__name__}.")
        if len(trigger) > SKILL_MAX_TRIGGER_LEN:
            raise ValueError(
                f"triggers[{i}] exceeds max length {SKILL_MAX_TRIGGER_LEN}."
            )
    return triggers


def _validate_examples(examples: Optional[List[dict]]) -> Optional[List[dict]]:
    if examples is None:
        return examples
    if len(examples) > SKILL_MAX_EXAMPLES_COUNT:
        raise ValueError(
            f"examples has {len(examples)} entries; max is {SKILL_MAX_EXAMPLES_COUNT}."
        )
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"examples[{i}] must be an object, got {type(example).__name__}.")
    return examples


class ProceduralMemoryItemBase(MirixBase):
    """
    Base schema for storing procedural knowledge as reusable skills.
    """

    __id_prefix__ = "proc_item"
    name: str = Field(
        ..., max_length=SKILL_MAX_NAME_LEN, description="Short skill identifier (e.g., 'deploy-production')"
    )
    entry_type: str = Field(..., description="Category (one of: 'workflow', 'guide', 'script')")
    description: str = Field(
        ..., max_length=SKILL_MAX_DESCRIPTION_LEN, description="Short descriptive text about the skill"
    )
    instructions: str = Field(
        ..., max_length=SKILL_MAX_INSTRUCTIONS_LEN, description="Step-by-step instructions as a single string"
    )
    triggers: List[str] = Field(default_factory=list, description="Conditions indicating this skill is relevant")
    examples: List[dict] = Field(default_factory=list, description="Input/output examples for this skill")

    @field_validator("entry_type")
    @classmethod
    def _check_entry_type(cls, value: str) -> str:
        return _validate_entry_type(value)

    @field_validator("triggers")
    @classmethod
    def _check_triggers(cls, value: List[str]) -> List[str]:
        return _validate_triggers(value) or []

    @field_validator("examples")
    @classmethod
    def _check_examples(cls, value: List[dict]) -> List[dict]:
        return _validate_examples(value) or []


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
        None, description="The embedding configuration used by the skill"
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
    name: Optional[str] = Field(None, max_length=SKILL_MAX_NAME_LEN, description="Short skill identifier")
    entry_type: Optional[str] = Field(None, description="Category (one of: 'workflow', 'guide', 'script')")
    description: Optional[str] = Field(
        None, max_length=SKILL_MAX_DESCRIPTION_LEN, description="Short descriptive text about the skill"
    )
    instructions: Optional[str] = Field(
        None, max_length=SKILL_MAX_INSTRUCTIONS_LEN, description="Step-by-step instructions as a single string"
    )
    triggers: Optional[List[str]] = Field(None, description="Conditions indicating this skill is relevant")
    examples: Optional[List[dict]] = Field(None, description="Input/output examples for this skill")
    version: Optional[str] = Field(None, description="Semantic version of this skill")
    organization_id: Optional[str] = Field(None, description="The organization ID")
    updated_at: datetime = Field(default_factory=get_utc_time, description="Update timestamp")

    @field_validator("entry_type")
    @classmethod
    def _check_entry_type(cls, value: Optional[str]) -> Optional[str]:
        return _validate_entry_type(value)

    @field_validator("triggers")
    @classmethod
    def _check_triggers(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _validate_triggers(value)

    @field_validator("examples")
    @classmethod
    def _check_examples(cls, value: Optional[List[dict]]) -> Optional[List[dict]]:
        return _validate_examples(value)
    last_modify: Optional[Dict[str, Any]] = Field(
        None,
        description="Last modification info including timestamp and operation type",
    )
    instructions_embedding: Optional[List[float]] = Field(None, description="The embedding of the instructions")
    description_embedding: Optional[List[float]] = Field(None, description="The embedding of the description")
    embedding_config: Optional[EmbeddingConfig] = Field(
        None, description="The embedding configuration used by the skill"
    )

    filter_tags: Optional[Dict[str, Any]] = Field(
        None, description="Custom filter tags for filtering and categorization"
    )


class ProceduralMemoryItemResponse(ProceduralMemoryItem):
    """Response schema for procedural memory item."""

    pass
