import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mirix.schemas.usage import MirixUsageStatistics


class AutoDreamRequest(BaseModel):
    start_date: Optional[dt.datetime] = Field(None, description="Start of time window; defaults to last dream time")
    end_date: Optional[dt.datetime] = Field(None, description="End of time window; defaults to now")
    memory_types: List[str] = Field(
        default=["episodic", "semantic", "procedural", "resource", "knowledge_vault"],
        description="Which memory types to process",
    )
    dry_run: bool = Field(False, description="If true, return plan without applying changes")
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")
    temperature: Optional[float] = Field(None, description="Override LLM temperature (e.g. 0.0 for deterministic testing)")
    raw_sessions: Optional[List[str]] = Field(
        None,
        description=(
            "Raw session texts to inject into the dream payload as cheating context. "
            "Intended for the FIRST dream in a run, where memories alone are sparse/empty. "
            "When provided, the agent sees the raw conversations alongside any fetched memories."
        ),
    )


class MemoryTypeStats(BaseModel):
    total: int = 0
    removed: int = 0
    merged: int = 0
    conflicts_resolved: int = 0


class AutoDreamResponse(BaseModel):
    start_date: Optional[dt.datetime] = None
    end_date: Optional[dt.datetime] = None
    processed: Dict[str, MemoryTypeStats]
    last_dream_at: Optional[dt.datetime] = None
    dry_run: bool
    message: str = ""
    usage: Optional[MirixUsageStatistics] = None
