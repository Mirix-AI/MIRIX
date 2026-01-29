from enum import Enum

# Import ToolType from schemas (moved to avoid circular imports)
from mirix.schemas.enums import ToolType  # noqa: F401


class JobType(str, Enum):
    JOB = "job"
    RUN = "run"


class ToolSourceType(str, Enum):
    """Defines what a tool was derived from"""

    python = "python"
    json = "json"


class AccessType(str, Enum):
    """Defines the access scope for ORM operations"""

    ORGANIZATION = "organization"
    USER = "user"
