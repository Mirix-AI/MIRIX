import datetime as dt
from typing import Dict, Optional

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
        False,
        description=(
            "If true, skip the LLM memory-merge pass entirely and only refine the graph "
            "(maintenance + semantic reconsolidation). The flat PG memories — their ids, "
            "content and embeddings — are left untouched, so flat retrieval is unchanged; "
            "only the hypergraph structure is consolidated. This avoids the retrieval "
            "degradation that memory merging causes on fact-recall QA."
        ),
    )
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")

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
