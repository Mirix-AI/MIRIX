from enum import Enum


class ToolType(str, Enum):
    CUSTOM = "custom"
    MIRIX_CORE = "mirix_core"
    MIRIX_CODER_CORE = "mirix_coder_core"
    MIRIX_MEMORY_CORE = "mirix_memory_core"
    MIRIX_EXTRA = "mirix_extra"
    MIRIX_MULTI_AGENT_CORE = "mirix_multi_agent_core"


class JobType(str, Enum):
    JOB = "job"
    RUN = "run"


class ToolSourceType(str, Enum):
    """Defines what a tool was derived from"""

    python = "python"
    json = "json"
