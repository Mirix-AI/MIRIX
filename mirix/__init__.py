__version__ = "0.1.2"


# import clients
from mirix.client.client import LocalClient, create_client

# # imports for easier access
from mirix.schemas.agent import AgentState
from mirix.schemas.block import Block
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.enums import JobStatus
from mirix.schemas.mirix_message import MirixMessage
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, BasicBlockMemory, ChatMemory, Memory, RecallMemorySummary
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.organization import Organization
from mirix.schemas.tool import Tool
from mirix.schemas.usage import MirixUsageStatistics
from mirix.schemas.user import User

# Import the new SDK interface
from mirix.sdk import Mirix