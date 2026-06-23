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
    model: Optional[str] = Field(None, description="Override LLM model (e.g. gpt-4.1-mini for testing)")
    last_n_sessions: Optional[int] = Field(
        None,
        description=(
            "For mode='procedural' (general session-experience distillation): how "
            "many of the meta agent's most-recent retained sessions to distill. "
            "Defaults to MESSAGE_RETAIN_LAST_N_SESSIONS."
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
