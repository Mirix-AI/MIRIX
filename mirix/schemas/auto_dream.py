import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


AUTO_DREAM_MODES = {"core", "episodic", "semantic", "resource", "procedural", "knowledge", "experience"}


class AutoDreamRequest(BaseModel):
    start_date: Optional[dt.datetime] = Field(None, description="Start of time window; defaults to last dream time")
    end_date: Optional[dt.datetime] = Field(None, description="End of time window; defaults to now")
    mode: str = Field(
        default="experience",
        description=(
            "Auto-dream mode. One of: core, episodic, semantic, resource, procedural, knowledge, experience. "
            "experience processes episodic, semantic, and knowledge together."
        ),
    )
    dry_run: bool = Field(False, description="If true, return plan without applying changes")
    graph_only: bool = Field(
        True,
        description=(
            "Refine the graph only: structural maintenance plus node merging (value "
            "union + edge rewiring) and conflict flagging. The flat PG memories — "
            "their ids, content and embeddings — are left untouched, so flat "
            "retrieval is unchanged. THIS IS THE DEFAULT: consolidation belongs on "
            "the graph, and rewriting the flat store measured -7.7 QA points on "
            "LongMemEval-S before the coverage gate, break-even after it. Pass "
            "false to opt into the legacy LLM memory-merge pass (`mode` then "
            "selects which memory components it rewrites)."
        ),
    )
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")
    source_chunk_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Exact source_meta.chunk_id values covered by this dream batch. Incremental "
            "graph revisions use them to resolve the semantic-memory delta; older "
            "graph versions ignore this field."
        ),
    )
    semantic_memory_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Optional explicit semantic-memory delta. This takes precedence over "
            "source_chunk_ids and is primarily useful for non-conversation ingesters."
        ),
    )
    final_full_graph: bool = Field(
        False,
        description=(
            "True only for the Dream immediately after the final ingest chunk. "
            "Hybrid graph revisions use it to run their deferred full semantic sweep."
        ),
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, mode: str) -> str:
        mode = mode.lower()
        if mode not in AUTO_DREAM_MODES:
            raise ValueError(f"mode must be one of {sorted(AUTO_DREAM_MODES)}")
        return mode


class MemoryTypeStats(BaseModel):
    total: int = 0
    removed: int = 0
    merged: int = 0
    conflicts_resolved: int = 0


class AutoDreamResponse(BaseModel):
    start_date: Optional[dt.datetime]
    end_date: Optional[dt.datetime]
    processed: Dict[str, MemoryTypeStats]
    last_dream_at: dt.datetime
    dry_run: bool
    message: str = ""
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
