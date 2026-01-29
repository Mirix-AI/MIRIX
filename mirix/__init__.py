__version__ = "0.1.0"


# Client imports (always available)
from mirix.client import MirixClient as MirixClient

# Server-only imports (only available when full package is installed)
try:
    from mirix.local_client import LocalClient as LocalClient
    from mirix.local_client import create_client as create_client
except ImportError:
    LocalClient = None
    create_client = None

try:
    from mirix.sdk import Mirix as Mirix
    from mirix.sdk import load_config as load_config
except ImportError:
    Mirix = None
    load_config = None

# Schema imports for easier access (available in both client and server)
from mirix.schemas.agent import AgentState as AgentState
from mirix.schemas.block import Block as Block
from mirix.schemas.embedding_config import EmbeddingConfig as EmbeddingConfig
from mirix.schemas.enums import JobStatus as JobStatus
from mirix.schemas.llm_config import LLMConfig as LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary as ArchivalMemorySummary
from mirix.schemas.memory import BasicBlockMemory as BasicBlockMemory
from mirix.schemas.memory import ChatMemory as ChatMemory
from mirix.schemas.memory import Memory as Memory
from mirix.schemas.memory import RecallMemorySummary as RecallMemorySummary
from mirix.schemas.message import Message as Message
from mirix.schemas.mirix_message import MirixMessage as MirixMessage
from mirix.schemas.openai.chat_completion_response import UsageStatistics as UsageStatistics
from mirix.schemas.organization import Organization as Organization
from mirix.schemas.tool import Tool as Tool
from mirix.schemas.usage import MirixUsageStatistics as MirixUsageStatistics
from mirix.schemas.user import User as User
