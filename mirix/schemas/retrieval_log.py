"""Pydantic schemas for the retrieval log.

One `RetrievalLog` row per memory-retrieval call: the raw query, the extracted
topics and how extraction went, the query-embedding status, and a per-memory
type trace (method, every candidate's bm25 / cosine / rrf scores and rank, gate
parameters, returned ids, skills missing a vector). Its id is returned to the
caller as `retrieval_id`.

Rows are kept `MIRIX_RETRIEVAL_LOG_RETENTION_DAYS` (default 30), deleted with
the user's data, and not written at all when `MIRIX_RETRIEVAL_LOG_ENABLED` is
false.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from mirix.helpers.datetime_helpers import get_utc_time
from mirix.schemas.message import _validate_session_id
from mirix.schemas.mirix_base import MirixBase

# The six memory components a retrieval can cover, in the order a retrieval
# returns them (memory_types=None retrieves all of them).
RETRIEVAL_MEMORY_TYPE_ORDER = (
    "episodic",
    "semantic",
    "resource",
    "procedural",
    "knowledge_vault",
    "core",
)
RETRIEVAL_MEMORY_TYPES = frozenset(RETRIEVAL_MEMORY_TYPE_ORDER)

# ok = topics extracted by the LLM; fallback = extraction failed, the raw text
# was used as the query; skipped = no extraction was attempted.
ExtractionStatus = Literal["ok", "fallback", "skipped"]
# ok = query vector computed; failed = could not embed (BM25-only, no gate);
# skipped = no vector needed for the requested memory types.
EmbeddingStatus = Literal["ok", "failed", "skipped"]


class RetrievalLogBase(MirixBase):
    __id_prefix__ = "rlog"


class RetrievalLog(RetrievalLogBase):
    """Full retrieval-log schema."""

    id: Optional[str] = Field(
        None, description="Returned to the caller as `retrieval_id`."
    )
    organization_id: str = Field(..., description="The owning organization id.")
    user_id: str = Field(..., description="The user whose memories were retrieved.")
    client_id: Optional[str] = Field(None, description="The calling client.")
    session_id: Optional[str] = Field(
        None, description="The caller's session, when given."
    )
    memory_types: Optional[List[str]] = Field(
        None, description="Memory types searched (None = all)."
    )
    query_text: Optional[str] = Field(
        None, description="The raw query text (kept for the retention window)."
    )
    extracted_topics: Optional[str] = Field(
        None, description="Topics extracted from the messages."
    )
    extraction_status: Optional[ExtractionStatus] = Field(
        None, description="ok | fallback | skipped."
    )
    embedding_status: Optional[EmbeddingStatus] = Field(
        None, description="ok | failed | skipped."
    )
    per_type: Optional[Dict[str, Any]] = Field(
        None,
        description="memory_type -> method, candidate scores/ranks, gate params, returned ids.",
    )
    llm_usage: Optional[Dict[str, Any]] = Field(
        None, description="Token usage of topic extraction."
    )
    latency_ms: Optional[int] = Field(
        None, ge=0, description="End-to-end latency in milliseconds."
    )
    retry_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Every retry during the retrieval: kind, attempt, waited_s, at, and optionally stage, error.",
    )
    embedding_chunked_calls: int = Field(
        0,
        ge=0,
        description="Embedding calls of the retrieval whose input had to be chunked.",
    )
    created_at: datetime = Field(
        default_factory=get_utc_time, description="When the retrieval happened."
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: Optional[str]) -> Optional[str]:
        return _validate_session_id(value)

    @field_validator("memory_types")
    @classmethod
    def _check_memory_types(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        unknown = [t for t in value if t not in RETRIEVAL_MEMORY_TYPES]
        if unknown:
            raise ValueError(
                f"unknown memory_types {unknown}; allowed: {sorted(RETRIEVAL_MEMORY_TYPES)}"
            )
        return value


class RetrievalLogResponse(RetrievalLog):
    """Response schema for a retrieval-log row."""

    pass
