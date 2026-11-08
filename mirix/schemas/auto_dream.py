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
    graph_only: Optional[bool] = Field(
        None,
        description=(
            "Refine the graph only: structural maintenance plus node merging (value "
            "union + edge rewiring) and conflict flagging. The flat PG memories — "
            "their ids, content and embeddings — are left untouched, so flat "
            "retrieval is unchanged. Unset, it is true when graph memory is on "
            "(MIRIX_ENABLE_GRAPH_MEMORY=true): consolidation belongs on the graph, and rewriting the flat store "
            "measured -7.7 QA points on LongMemEval-S before the coverage gate, "
            "break-even after it. Outside it, it is false: the LLM memory-merge "
            "pass (`mode` then selects which memory components it rewrites). "
            "mode='procedural' does not read it."
        ),
    )
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")
    meta_agent_id: Optional[str] = Field(
        None,
        description=(
            "Optional meta memory agent id to run auto-dream against. When omitted, "
            "the server uses the client's first meta memory agent for backward "
            "compatibility."
        ),
    )
    last_n_sessions: Optional[int] = Field(
        None,
        description=(
            "For mode='procedural': claim at most this many pending sessions "
            "(oldest first) into the learning run. Default (or <= 0): every "
            "pending session of the meta agent's scope (flush semantics)."
        ),
    )
    wait: bool = Field(
        True,
        description=(
            "Every non-dry auto_dream run is recorded as a learning run (any "
            "mode). true (default) runs it synchronously and returns its "
            "result; false queues it in the background and returns its id in "
            "run_ids at once (poll GET /memory/learning_runs/{id})."
        ),
    )
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
    # Structured evolution counts for the procedural path so drivers can
    # health-gate without parsing `message`. Default 0/empty keeps every
    # existing caller (and non-procedural modes) working unchanged.
    skills_changed: int = 0
    changes: Dict[str, list] = Field(default_factory=dict)
    # The recorded runs (skill_learning_run ids) this call created or ran —
    # every mode; empty only for dry runs.
    run_ids: List[str] = Field(default_factory=list)
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
