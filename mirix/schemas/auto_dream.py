import datetime as dt
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AutoDreamRequest(BaseModel):
    start_date: Optional[dt.datetime] = Field(None, description="Start of time window; defaults to last dream time")
    end_date: Optional[dt.datetime] = Field(None, description="End of time window; defaults to now")
    memory_types: List[str] = Field(
        default=["episodic", "semantic", "procedural", "resource", "knowledge_vault"],
        description="Which memory types to process",
    )
    dry_run: bool = Field(False, description="If true, return plan without applying changes")
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")


class MemoryTypeStats(BaseModel):
    total: int = 0
    removed: int = 0
    merged: int = 0
    conflicts_resolved: int = 0


class AutoDreamResponse(BaseModel):
    start_date: dt.datetime
    end_date: dt.datetime
    processed: Dict[str, MemoryTypeStats]
    last_dream_at: dt.datetime
    dry_run: bool
    message: str = ""
