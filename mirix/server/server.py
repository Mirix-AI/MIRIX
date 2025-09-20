# inspecting tools
import asyncio
import os
import traceback
import warnings
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple, Union

# from composio.client import Composio
# from composio.client.collections import ActionModel, AppModel
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

import mirix.constants as constants
import mirix.server.utils as server_utils
import mirix.system as system
from mirix.agent import Agent, save_agent

# TODO use custom interface
from mirix.interface import AgentInterface  # abstract
from mirix.interface import CLIInterface  # for printing to terminal
from mirix.log import get_logger
from mirix.agent import EpisodicMemoryAgent, ProceduralMemoryAgent, ResourceMemoryAgent, KnowledgeVaultAgent, MetaMemoryAgent, SemanticMemoryAgent, CoreMemoryAgent, ReflexionAgent, BackgroundAgent
from mirix.orm import Base
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType, CreateAgent
from mirix.schemas.block import BlockUpdate
from mirix.schemas.embedding_config import EmbeddingConfig

# openai schemas
from mirix.schemas.enums import JobStatus, MessageStreamStatus
from mirix.schemas.environment_variables import SandboxEnvironmentVariableCreate
from mirix.schemas.mirix_message import LegacyMirixMessage, MirixMessage, ToolReturnMessage
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, ContextWindowOverview, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate, MessageRole, MessageUpdate
from mirix.schemas.organization import Organization
from mirix.schemas.providers import (
    AnthropicBedrockProvider,
    AnthropicProvider,
    AzureProvider,
    GoogleAIProvider,
    GroqProvider,
    MirixProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    TogetherProvider,
    VLLMChatCompletionsProvider,
    VLLMCompletionsProvider,
)
from mirix.schemas.sandbox_config import SandboxType
from mirix.schemas.source import Source
from mirix.schemas.tool import Tool
from mirix.schemas.usage import MirixUsageStatistics
from mirix.schemas.user import User
from mirix.services.agent_manager import AgentManager
from mirix.services.block_manager import BlockManager
from mirix.services.message_manager import MessageManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.knowledge_vault_manager import KnowledgeVaultManager
from mirix.services.episodic_memory_manager import EpisodicMemoryManager
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.services.resource_memory_manager import ResourceMemoryManager
from mirix.services.semantic_memory_manager import SemanticMemoryManager
from mirix.services.per_agent_lock_manager import PerAgentLockManager
from mirix.services.cloud_file_mapping_manager import CloudFileMappingManager
from mirix.services.sandbox_config_manager import SandboxConfigManager
from mirix.services.provider_manager import ProviderManager
from mirix.services.step_manager import StepManager
from mirix.services.tool_execution_sandbox import ToolExecutionSandbox
from mirix.services.tool_manager import ToolManager
from mirix.services.user_manager import UserManager
from mirix.utils import get_friendly_error_msg, get_utc_time, json_dumps, json_loads

logger = get_logger(__name__)


class Server(object):
    """Abstract server class that supports multi-agent multi-user"""

    @abstractmethod
    def list_agents(self, user_id: str) -> dict:
        """List all available agents to a user"""
        raise NotImplementedError

    @abstractmethod
    def get_agent_memory(self, user_id: str, agent_id: str) -> dict:
        """Return the memory of an agent (core memory + non-core statistics)"""
        raise NotImplementedError

    @abstractmethod
    def get_server_config(self, user_id: str) -> dict:
        """Return the base config"""
        raise NotImplementedError

    @abstractmethod
    def update_agent_core_memory(self, user_id: str, agent_id: str, label: str, actor: User) -> Memory:
        """Update the agents core memory block, return the new state"""
        raise NotImplementedError

    @abstractmethod
    def create_agent(
        self,
        request: CreateAgent,
        actor: User,
        # interface
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        """Create a new agent using a config"""
        raise NotImplementedError

    @abstractmethod
    def user_message(self, user_id: str, agent_id: str, message: str) -> None:
        """Process a message from the user, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def system_message(self, user_id: str, agent_id: str, message: str) -> None:
        """Process a message from the system, internally calls step"""
        raise NotImplementedError

    @abstractmethod
    def send_messages(self, user_id: str, agent_id: str, messages: Union[MessageCreate, List[Message]]) -> None:
        """Send a list of messages to the agent"""
        raise NotImplementedError

    @abstractmethod
    def run_command(self, user_id: str, agent_id: str, command: str) -> Union[str, None]:
        """Run a command on the agent, e.g. /memory

        May return a string with a message generated by the command
        """
        raise NotImplementedError


from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mirix.config import MirixConfig

# NOTE: hack to see if single session management works
from mirix.settings import model_settings, settings, tool_settings

config = MirixConfig.load()


def print_sqlite_schema_error():
    """Print a formatted error message for SQLite schema issues"""
    console = Console()
    error_text = Text()
    error_text.append("Existing SQLite DB schema is invalid, and schema migrations are not supported for SQLite. ", style="bold red")
    error_text.append("To have migrations supported between Mirix versions, please run Mirix with Docker (", style="white")
    error_text.append("https://docs.mirix.com/server/docker", style="blue underline")
    error_text.append(") or use Postgres by setting ", style="white")
    error_text.append("MIRIX_PG_URI", style="yellow")
    error_text.append(".\n\n", style="white")
    error_text.append("If you wish to keep using SQLite, you can reset your database by removing the DB file with ", style="white")
    error_text.append("rm ~/.mirix/sqlite.db", style="yellow")
    error_text.append(" or downgrade to your previous version of Mirix.", style="white")

    console.print(Panel(error_text, border_style="red"))


@contextmanager
def db_error_handler():
    """Context manager for handling database errors"""
    try:
        yield
    except Exception as e:
        # Handle other SQLAlchemy errors
        print(e)
        print_sqlite_schema_error()
        # raise ValueError(f"SQLite DB error: {str(e)}")
        exit(1)

# Check for PGlite mode
USE_PGLITE = os.environ.get('MIRIX_USE_PGLITE', 'false').lower() == 'true'

if USE_PGLITE:
    print("PGlite mode detected - setting up PGlite adapter")
    
    # Import PGlite connector
    try:
        from mirix.database.pglite_connector import pglite_connector
        
        # Create a simple adapter to make PGlite work with existing code
        class PGliteSession:
            """Adapter to make PGlite work with SQLAlchemy-style code"""
            
            def __init__(self, connector):
                self.connector = connector
                
            def execute(self, query, params=None):
                """Execute a query using PGlite bridge"""
                if hasattr(query, 'compile'):
                    # Handle SQLAlchemy query objects
                    compiled = query.compile(compile_kwargs={"literal_binds": True})
                    query_str = str(compiled)
                else:
                    query_str = str(query)
                
                result = self.connector.execute_query(query_str, params)
                
                # Create a simple result wrapper
                class ResultWrapper:
                    def __init__(self, data):
                        self.rows = data.get('rows', [])
                        self.rowcount = data.get('rowCount', 0)
                        
                    def scalars(self):
                        return self.rows
                        
                    def all(self):
                        return self.rows
                        
                    def first(self):
                        return self.rows[0] if self.rows else None
                        
                return ResultWrapper(result)
                
            def commit(self):
                pass  # PGlite handles commits automatically
                
            def rollback(self):
                pass  # Basic implementation
                
            def close(self):
                pass  # No need to close PGlite sessions
        
        class PGliteEngine:
            """Engine adapter for PGlite"""
            
            def __init__(self, connector):
                self.connector = connector
                
            def connect(self):
                return PGliteSession(self.connector)
                
        # Create the engine
        engine = PGliteEngine(pglite_connector)
        
        # Create sessionmaker
        class PGliteSessionMaker:
            def __init__(self, engine):
                self.engine = engine
                
            def __call__(self):
                return self.engine.connect()
                
        SessionLocal = PGliteSessionMaker(engine)
        
        # Set config for PGlite mode
        config.recall_storage_type = "pglite"
        config.recall_storage_uri = "pglite://local"
        config.archival_storage_type = "pglite" 
        config.archival_storage_uri = "pglite://local"
        
        print("PGlite adapter initialized successfully")
        
    except ImportError as e:
        print(f"Failed to import PGlite connector: {e}")
        print("Falling back to SQLite mode")
        USE_PGLITE = False

if not USE_PGLITE and settings.mirix_pg_uri_no_default:
    print("Creating engine", settings.mirix_pg_uri)
    config.recall_storage_type = "postgres"
    config.recall_storage_uri = settings.mirix_pg_uri_no_default
    config.archival_storage_type = "postgres"
    config.archival_storage_uri = settings.mirix_pg_uri_no_default

    # create engine
    engine = create_engine(
        settings.mirix_pg_uri,
        pool_size=settings.pg_pool_size,
        max_overflow=settings.pg_max_overflow,
        pool_timeout=settings.pg_pool_timeout,
        pool_recycle=settings.pg_pool_recycle,
        echo=settings.pg_echo,
    )
    
    # Create all tables for PostgreSQL
    Base.metadata.create_all(bind=engine)
elif not USE_PGLITE:
    # TODO: don't rely on config storage
    sqlite_db_path = os.path.join(config.recall_storage_path, "sqlite.db")
    
    # Configure SQLite engine with proper concurrency settings
    engine = create_engine(
        f"sqlite:///{sqlite_db_path}",
        # Connection pooling configuration for better concurrency
        pool_size=20,
        max_overflow=30,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        # Enable SQLite-specific options
        connect_args={
            "check_same_thread": False,  # Allow sharing connections between threads
            "timeout": 30,  # 30 second timeout for database locks
        },
        echo=False,
    )

    # Store the original connect method
    original_connect = engine.connect

    def wrapped_connect(*args, **kwargs):
        with db_error_handler():
            # Get the connection
            connection = original_connect(*args, **kwargs)

            # Store the original execution method
            original_execute = connection.execute

            # Wrap the execute method of the connection
            def wrapped_execute(*args, **kwargs):
                with db_error_handler():
                    return original_execute(*args, **kwargs)

            # Replace the connection's execute method
            connection.execute = wrapped_execute

            return connection

    # Replace the engine's connect method
    engine.connect = wrapped_connect

    Base.metadata.create_all(bind=engine)

if not USE_PGLITE:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from contextlib import contextmanager

db_context = contextmanager(get_db)


class SyncServer(Server):
    """Simple single-threaded / blocking server process"""

    def __init__(
        self,
        chaining: bool = True,
        max_chaining_steps: Optional[bool] = None,
        default_interface_factory: Callable[[], AgentInterface] = lambda: CLIInterface(),
        init_with_default_org_and_user: bool = True,
        # default_interface: AgentInterface = CLIInterface(),
        # default_persistence_manager_cls: PersistenceManager = LocalStateManager,
        # auth_mode: str = "none",  # "none, "jwt", "external"
    ):
        """Server process holds in-memory agents that are being run"""
        # chaining = whether or not to run again if continue_chaining=true
        self.chaining = chaining

        # if chaining == true, what's the max number of times we'll chain before yielding?
        # none = no limit, can go on forever
        self.max_chaining_steps = max_chaining_steps

        # The default interface that will get assigned to agents ON LOAD
        self.default_interface_factory = default_interface_factory

        # Initialize the metadata store
        config = MirixConfig.load()
        if settings.mirix_pg_uri_no_default:
            config.recall_storage_type = "postgres"
            config.recall_storage_uri = settings.mirix_pg_uri_no_default
            config.archival_storage_type = "postgres"
            config.archival_storage_uri = settings.mirix_pg_uri_no_default
        config.save()
        self.config = config

        # Managers that interface with data models
        self.organization_manager = OrganizationManager()
        self.user_manager = UserManager()
        self.tool_manager = ToolManager()
        self.block_manager = BlockManager()
        self.sandbox_config_manager = SandboxConfigManager(tool_settings)
        self.message_manager = MessageManager()
        self.agent_manager = AgentManager()
        self.step_manager = StepManager()

        # Newly added managers
        self.knowledge_vault_manager = KnowledgeVaultManager()
        self.episodic_memory_manager =  EpisodicMemoryManager()
        self.procedural_memory_manager = ProceduralMemoryManager()
        self.resource_memory_manager = ResourceMemoryManager()
        self.semantic_memory_manager = SemanticMemoryManager()

        # API Key Manager
        self.provider_manager = ProviderManager()

        # CloudFileManager
        self.cloud_file_mapping_manager = CloudFileMappingManager()

        # Managers that interface with parallelism
        self.per_agent_lock_manager = PerAgentLockManager()

        # Make default user and org
        if init_with_default_org_and_user:
            self.default_org = self.organization_manager.create_default_organization()
            self.default_user = self.user_manager.create_default_user()
            # self.block_manager.add_default_blocks(actor=self.default_user)
            self.tool_manager.upsert_base_tools(actor=self.default_user)

            # # Add composio keys to the tool sandbox env vars of the org
            # if tool_settings.composio_api_key:
            #     manager = SandboxConfigManager(tool_settings)
            #     sandbox_config = manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=self.default_user)

            #     manager.create_sandbox_env_var(
            #         SandboxEnvironmentVariableCreate(key="COMPOSIO_API_KEY", value=tool_settings.composio_api_key),
            #         sandbox_config_id=sandbox_config.id,
            #         actor=self.default_user,
            #     )

        # collect providers (always has Mirix as a default)
        self._enabled_providers: List[Provider] = [MirixProvider()]
        
        # Check for database-stored API key first, fall back to model_settings
        openai_override_key = ProviderManager().get_openai_override_key()
        openai_api_key = openai_override_key if openai_override_key else model_settings.openai_api_key
        
        if openai_api_key:
            self._enabled_providers.append(
                OpenAIProvider(
                    api_key=openai_api_key,
                    base_url=model_settings.openai_api_base,
                )
            )
        if model_settings.anthropic_api_key:
            self._enabled_providers.append(
                AnthropicProvider(
                    api_key=model_settings.anthropic_api_key,
                )
            )
        if model_settings.ollama_base_url:
            self._enabled_providers.append(
                OllamaProvider(
                    base_url=model_settings.ollama_base_url,
                    api_key=None,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
        # Check for database-stored API key first, fall back to model_settings
        gemini_override_key = ProviderManager().get_gemini_override_key()
        gemini_api_key = gemini_override_key if gemini_override_key else model_settings.gemini_api_key
        
        if gemini_api_key:
            self._enabled_providers.append(
                GoogleAIProvider(
                    api_key=gemini_api_key,
                )
            )
        if model_settings.azure_api_key and model_settings.azure_base_url:
            assert model_settings.azure_api_version, "AZURE_API_VERSION is required"
            self._enabled_providers.append(
                AzureProvider(
                    api_key=model_settings.azure_api_key,
                    base_url=model_settings.azure_base_url,
                    api_version=model_settings.azure_api_version,
                )
            )
        if model_settings.groq_api_key:
            self._enabled_providers.append(
                GroqProvider(
                    api_key=model_settings.groq_api_key,
                )
            )
        if model_settings.together_api_key:
            self._enabled_providers.append(
                TogetherProvider(
                    api_key=model_settings.together_api_key,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
        if model_settings.vllm_api_base:
            # vLLM exposes both a /chat/completions and a /completions endpoint
            self._enabled_providers.append(
                VLLMCompletionsProvider(
                    base_url=model_settings.vllm_api_base,
                    default_prompt_formatter=model_settings.default_prompt_formatter,
                )
            )
            # NOTE: to use the /chat/completions endpoint, you need to specify extra flags on vLLM startup
            # see: https://docs.vllm.ai/en/latest/getting_started/examples/openai_chat_completion_client_with_tools.html
            # e.g. "... --enable-auto-tool-choice --tool-call-parser hermes"
            self._enabled_providers.append(
                VLLMChatCompletionsProvider(
                    base_url=model_settings.vllm_api_base,
                )
            )
        if model_settings.aws_access_key and model_settings.aws_secret_access_key and model_settings.aws_region:
            self._enabled_providers.append(
                AnthropicBedrockProvider(
                    aws_region=model_settings.aws_region,
                )
            )

    def load_agent(self, agent_id: str, actor: User, interface: Union[AgentInterface, None] = None) -> Agent:
        """Updated method to load agents from persisted storage"""
        agent_lock = self.per_agent_lock_manager.get_lock(agent_id)
        with agent_lock:
            agent_state = self.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor)

            interface = interface or self.default_interface_factory()
            if agent_state.agent_type == AgentType.chat_agent:
                agent = Agent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.episodic_memory_agent:
                agent = EpisodicMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.knowledge_vault_agent:
                agent = KnowledgeVaultAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.procedural_memory_agent:
                agent = ProceduralMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.resource_memory_agent:
                agent = ResourceMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.meta_memory_agent:
                agent = MetaMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.semantic_memory_agent:
                agent = SemanticMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.core_memory_agent:
                agent = CoreMemoryAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.reflexion_agent:
                agent = ReflexionAgent(agent_state=agent_state, interface=interface, user=actor)
            elif agent_state.agent_type == AgentType.background_agent:
                agent = BackgroundAgent(agent_state=agent_state, interface=interface, user=actor)
            else:
                raise ValueError(f"Invalid agent type {agent_state.agent_type}")

            return agent

    def _step(
        self,
        actor: User,
        agent_id: str,
        input_messages: Union[Message, List[Message]],
        interface: Union[AgentInterface, None] = None,  # needed to getting responses
        put_inner_thoughts_first: bool = True,
        existing_file_uris: Optional[List[str]] = None,
        force_response: bool = False,
        display_intermediate_message: any = None,
        request_user_confirmation: any = None,
        chaining: Optional[bool] = None,
        extra_messages: Optional[List[dict]] = None,
        message_queue: Optional[any] = None,
        retrieved_memories: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> MirixUsageStatistics:
        """Send the input message through the agent"""
        logger.debug(f"Got input messages: {input_messages}")
        mirix_agent = None
        try:
            mirix_agent = self.load_agent(agent_id=agent_id, interface=interface, actor=actor)

            if mirix_agent is None:
                raise KeyError(f"Agent (user={actor.id}, agent={agent_id}) is not loaded")

            # Determine whether or not to token stream based on the capability of the interface
            token_streaming = mirix_agent.interface.streaming_mode if hasattr(mirix_agent.interface, "streaming_mode") else False

            logger.debug(f"Starting agent step")
            if interface:
                metadata = interface.metadata if hasattr(interface, "metadata") else None
            else:
                metadata = None

            # Use provided chaining value or fall back to server default
            effective_chaining = chaining if chaining is not None else self.chaining

            usage_stats = mirix_agent.step(
                input_messages=input_messages,
                chaining=effective_chaining,
                max_chaining_steps=self.max_chaining_steps,
                stream=token_streaming,
                skip_verify=True,
                metadata=metadata,
                force_response=force_response,
                existing_file_uris=existing_file_uris,
                display_intermediate_message=display_intermediate_message,
                request_user_confirmation=request_user_confirmation,
                put_inner_thoughts_first=put_inner_thoughts_first,
                extra_messages=extra_messages,
                message_queue=message_queue,
                user_id=user_id
            )

        except Exception as e:
            logger.error(f"Error in server._step: {e}")
            print(traceback.print_exc())
            raise
        finally:
            logger.debug("Calling step_yield()")
            if mirix_agent:
                mirix_agent.interface.step_yield()

        return usage_stats

    def _command(self, user_id: str, agent_id: str, command: str) -> MirixUsageStatistics:
        """Process a CLI command"""
        # TODO: Thread actor directly through this function, since the top level caller most likely already retrieved the user
        actor = self.user_manager.get_user_or_default(user_id=user_id)

        logger.debug(f"Got command: {command}")

        # Get the agent object (loaded in memory)
        mirix_agent = self.load_agent(agent_id=agent_id, actor=actor)
        usage = None

        if command.lower() == "exit":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "save" or command.lower() == "savechat":
            save_agent(mirix_agent)

        elif command.lower() == "dump" or command.lower().startswith("dump "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 0
            if amount == 0:
                mirix_agent.interface.print_messages(mirix_agent.messages, dump=True)
            else:
                mirix_agent.interface.print_messages(mirix_agent.messages[-min(amount, len(mirix_agent.messages)) :], dump=True)

        elif command.lower() == "dumpraw":
            mirix_agent.interface.print_messages_raw(mirix_agent.messages)

        elif command.lower() == "memory":
            ret_str = f"\nDumping memory contents:\n" + f"\n{str(mirix_agent.agent_state.memory)}"
            return ret_str

        elif command.lower() == "pop" or command.lower().startswith("pop "):
            # Check if there's an additional argument that's an integer
            command = command.strip().split()
            pop_amount = int(command[1]) if len(command) > 1 and command[1].isdigit() else 3
            n_messages = len(mirix_agent.messages)
            MIN_MESSAGES = 2
            if n_messages <= MIN_MESSAGES:
                logger.debug(f"Agent only has {n_messages} messages in stack, none left to pop")
            elif n_messages - pop_amount < MIN_MESSAGES:
                logger.debug(f"Agent only has {n_messages} messages in stack, cannot pop more than {n_messages - MIN_MESSAGES}")
            else:
                logger.debug(f"Popping last {pop_amount} messages from stack")
                for _ in range(min(pop_amount, len(mirix_agent.messages))):
                    mirix_agent.messages.pop()

        elif command.lower() == "retry":
            # TODO this needs to also modify the persistence manager
            logger.debug(f"Retrying for another answer")
            while len(mirix_agent.messages) > 0:
                if mirix_agent.messages[-1].get("role") == "user":
                    # we want to pop up to the last user message and send it again
                    mirix_agent.messages[-1].get("content")
                    mirix_agent.messages.pop()
                    break
                mirix_agent.messages.pop()

        elif command.lower() == "rethink" or command.lower().startswith("rethink "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rethink "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(mirix_agent.messages) - 1, 0, -1):
                    if mirix_agent.messages[x].get("role") == "assistant":
                        text = command[len("rethink ") :].strip()
                        mirix_agent.messages[x].update({"content": text})
                        break

        elif command.lower() == "rewrite" or command.lower().startswith("rewrite "):
            # TODO this needs to also modify the persistence manager
            if len(command) < len("rewrite "):
                logger.warning("Missing text after the command")
            else:
                for x in range(len(mirix_agent.messages) - 1, 0, -1):
                    if mirix_agent.messages[x].get("role") == "assistant":
                        text = command[len("rewrite ") :].strip()
                        args = json_loads(mirix_agent.messages[x].get("function_call").get("arguments"))
                        args["message"] = text
                        mirix_agent.messages[x].get("function_call").update({"arguments": json_dumps(args)})
                        break

        # No skip options
        elif command.lower() == "wipe":
            # exit not supported on server.py
            raise ValueError(command)

        elif command.lower() == "contine_chaining":
            input_message = system.get_contine_chaining()
            usage = self._step(actor=actor, agent_id=agent_id, input_message=input_message)

        elif command.lower() == "memorywarning":
            input_message = system.get_token_limit_warning()
            usage = self._step(actor=actor, agent_id=agent_id, input_message=input_message)

        if not usage:
            usage = MirixUsageStatistics()

        return usage

    def user_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> MirixUsageStatistics:
        """Process an incoming user message and feed it through the Mirix agent"""
        try:
            actor = self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        try:
            agent = self.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor)
        except NoResultFound:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_user_message = system.package_user_message(
                user_message=message,
                time=timestamp.isoformat() if timestamp else None,
            )

            # NOTE: eventually deprecate and only allow passing Message types
            # Convert to a Message object
            if timestamp:
                message = Message(
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    agent_id=agent_id,
                    role="user",
                    text=packaged_user_message,
                )

        # Run the agent state forward
        usage = self._step(actor=actor, agent_id=agent_id, input_messages=message)
        return usage

    def system_message(
        self,
        user_id: str,
        agent_id: str,
        message: Union[str, Message],
        timestamp: Optional[datetime] = None,
    ) -> MirixUsageStatistics:
        """Process an incoming system message and feed it through the Mirix agent"""
        try:
            actor = self.user_manager.get_user_by_id(user_id=user_id)
        except NoResultFound:
            raise ValueError(f"User user_id={user_id} does not exist")

        try:
            agent = self.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor)
        except NoResultFound:
            raise ValueError(f"Agent agent_id={agent_id} does not exist")

        # Basic input sanitization
        if isinstance(message, str):
            if len(message) == 0:
                raise ValueError(f"Invalid input: '{message}'")

            # If the input begins with a command prefix, reject
            elif message.startswith("/"):
                raise ValueError(f"Invalid input: '{message}'")

            packaged_system_message = system.package_system_message(system_message=message)

            # NOTE: eventually deprecate and only allow passing Message types
            # Convert to a Message object

            if timestamp:
                message = Message(
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                    created_at=timestamp,
                )
            else:
                message = Message(
                    agent_id=agent_id,
                    role="system",
                    text=packaged_system_message,
                )

        if isinstance(message, Message):
            # Can't have a null text field
            if message.text is None or len(message.text) == 0:
                raise ValueError(f"Invalid input: '{message.text}'")
            # If the input begins with a command prefix, reject
            elif message.text.startswith("/"):
                raise ValueError(f"Invalid input: '{message.text}'")

        else:
            raise TypeError(f"Invalid input: '{message}' - type {type(message)}")

        if timestamp:
            # Override the timestamp with what the caller provided
            message.created_at = timestamp

        # Run the agent state forward
        return self._step(actor=actor, agent_id=agent_id, input_messages=message)

    def construct_system_message(self, agent_id: str, message: str, actor: User) -> str:
        """
        Construct a system message from a message.
        """
        logger.debug(f"Got message: {message}")
        mirix_agent = None
        mirix_agent = self.load_agent(agent_id=agent_id, actor=actor)
        if mirix_agent is None:
            raise KeyError(f"Agent (user={actor.id}, agent={agent_id}) is not loaded")
        return mirix_agent.construct_system_message(message=message)

    def send_messages(
        self,
        actor: User,
        agent_id: str,
        input_messages: List[MessageCreate],
        interface: Union[AgentInterface, None] = None,  # needed for responses
        metadata: Optional[dict] = None,  # Pass through metadata to interface
        put_inner_thoughts_first: bool = True,
        display_intermediate_message: callable = None,
        request_user_confirmation: callable = None,
        force_response: bool = False,
        chaining: Optional[bool] = True,
        existing_file_uris: Optional[List[str]] = None,
        extra_messages: Optional[List[dict]] = None,
        message_queue: Optional[any] = None,
        retrieved_memories: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> MirixUsageStatistics:
        """Send a list of messages to the agent."""

        # Store metadata in interface if provided
        if metadata and hasattr(interface, "metadata"):
            interface.metadata = metadata

        # Run the agent state forward
        return self._step(
            actor=actor,
            agent_id=agent_id,
            input_messages=input_messages,
            interface=interface,
            force_response=force_response,
            put_inner_thoughts_first=put_inner_thoughts_first,
            display_intermediate_message=display_intermediate_message,
            request_user_confirmation=request_user_confirmation,
            chaining=chaining,
            existing_file_uris=existing_file_uris,
            extra_messages=extra_messages,
            message_queue=message_queue,
            retrieved_memories=retrieved_memories,
            user_id=user_id
        )

    # @LockingServer.agent_lock_decorator
    def run_command(self, user_id: str, agent_id: str, command: str) -> MirixUsageStatistics:
        """Run a command on the agent"""
        # If the input begins with a command prefix, attempt to process it as a command
        if command.startswith("/"):
            if len(command) > 1:
                command = command[1:]  # strip the prefix
        return self._command(user_id=user_id, agent_id=agent_id, command=command)

    def create_agent(
        self,
        request: CreateAgent,
        actor: User,
        # interface
        interface: Union[AgentInterface, None] = None,
    ) -> AgentState:
        if request.llm_config is None:
            if request.model is None:
                raise ValueError("Must specify either model or llm_config in request")
            request.llm_config = self.get_llm_config_from_handle(handle=request.model, context_window_limit=request.context_window_limit)

        if request.embedding_config is None:
            if request.embedding is None:
                raise ValueError("Must specify either embedding or embedding_config in request")
            request.embedding_config = self.get_embedding_config_from_handle(
                handle=request.embedding, embedding_chunk_size=request.embedding_chunk_size or constants.DEFAULT_EMBEDDING_CHUNK_SIZE
            )

        """Create a new agent using a config"""
        # Invoke manager
        return self.agent_manager.create_agent(
            agent_create=request,
            actor=actor,
        )

    # convert name->id

    # TODO: These can be moved to agent_manager
    def get_agent_memory(self, agent_id: str, actor: User) -> Memory:
        """Return the memory of an agent (core memory)"""
        return self.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor).memory

    def get_recall_memory_summary(self, agent_id: str, actor: User) -> RecallMemorySummary:
        return RecallMemorySummary(size=self.message_manager.size(actor=actor, agent_id=agent_id))

    def get_agent_recall_cursor(
        self,
        user_id: str,
        agent_id: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: Optional[int] = 100,
        reverse: Optional[bool] = False,
        return_message_object: bool = True,
        assistant_message_tool_name: str = constants.DEFAULT_MESSAGE_TOOL,
        assistant_message_tool_kwarg: str = constants.DEFAULT_MESSAGE_TOOL_KWARG,
    ) -> Union[List[Message], List[MirixMessage]]:
        # TODO: Thread actor directly through this function, since the top level caller most likely already retrieved the user

        actor = self.user_manager.get_user_or_default(user_id=user_id)
        start_date = self.message_manager.get_message_by_id(after, actor=actor).created_at if after else None
        end_date = self.message_manager.get_message_by_id(before, actor=actor).created_at if before else None

        records = self.message_manager.list_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            ascending=not reverse,
        )

        if not return_message_object:
            records = [
                msg
                for m in records
                for msg in m.to_mirix_message(
                    assistant_message_tool_name=assistant_message_tool_name,
                    assistant_message_tool_kwarg=assistant_message_tool_kwarg,
                )
            ]

        if reverse:
            records = records[::-1]

        return records

    def get_server_config(self, include_defaults: bool = False) -> dict:
        """Return the base config"""

        def clean_keys(config):
            config_copy = config.copy()
            for k, v in config.items():
                if k == "key" or "_key" in k:
                    config_copy[k] = server_utils.shorten_key_middle(v, chars_each_side=5)
            return config_copy

        # TODO: do we need a separate server config?
        base_config = vars(self.config)
        clean_base_config = clean_keys(base_config)

        response = {"config": clean_base_config}

        if include_defaults:
            default_config = vars(MirixConfig())
            clean_default_config = clean_keys(default_config)
            response["defaults"] = clean_default_config

        return response

    def update_agent_core_memory(self, agent_id: str, label: str, value: str, actor: User) -> Memory:
        """Update the value of a block in the agent's memory"""

        # get the block id
        block = self.agent_manager.get_block_with_label(agent_id=agent_id, block_label=label, actor=actor)

        # update the block
        self.block_manager.update_block(block_id=block.id, block_update=BlockUpdate(value=value), actor=actor)

        # rebuild system prompt for agent, potentially changed
        return self.agent_manager.rebuild_system_prompt(agent_id=agent_id, actor=actor).memory

    def update_agent_message(self, message_id: str, request: MessageUpdate, actor: User) -> Message:
        """Update the details of a message associated with an agent"""

        # Get the current message
        return self.message_manager.update_message_by_id(message_id=message_id, message_update=request, actor=actor)

    def get_organization_or_default(self, org_id: Optional[str]) -> Organization:
        """Get the organization object for org_id if it exists, otherwise return the default organization object"""
        if org_id is None:
            org_id = self.organization_manager.DEFAULT_ORG_ID

        try:
            return self.organization_manager.get_organization_by_id(org_id=org_id)
        except NoResultFound:
            raise HTTPException(status_code=404, detail=f"Organization with id {org_id} not found")

    def list_llm_models(self) -> List[LLMConfig]:
        """List available models"""

        llm_models = []
        for provider in self.get_enabled_providers():
            try:
                llm_models.extend(provider.list_llm_models())
            except Exception as e:
                warnings.warn(f"An error occurred while listing LLM models for provider {provider}: {e}")
        return llm_models

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        """List available embedding models"""
        embedding_models = []
        for provider in self.get_enabled_providers():
            try:
                embedding_models.extend(provider.list_embedding_models())
            except Exception as e:
                warnings.warn(f"An error occurred while listing embedding models for provider {provider}: {e}")
        return embedding_models

    def get_enabled_providers(self):
        providers_from_env = {p.name: p for p in self._enabled_providers}
        providers_from_db = {p.name: p for p in self.provider_manager.list_providers()}
        # Merge the two dictionaries, keeping the values from providers_from_db where conflicts occur
        return {**providers_from_env, **providers_from_db}.values()

    def get_llm_config_from_handle(self, handle: str, context_window_limit: Optional[int] = None) -> LLMConfig:
        provider_name, model_name = handle.split("/", 1)
        provider = self.get_provider_from_name(provider_name)

        llm_configs = [config for config in provider.list_llm_models() if config.model == model_name]
        if not llm_configs:
            raise ValueError(f"LLM model {model_name} is not supported by {provider_name}")
        elif len(llm_configs) > 1:
            raise ValueError(f"Multiple LLM models with name {model_name} supported by {provider_name}")
        else:
            llm_config = llm_configs[0]

        if context_window_limit:
            if context_window_limit > llm_config.context_window:
                raise ValueError(f"Context window limit ({context_window_limit}) is greater than maximum of ({llm_config.context_window})")
            llm_config.context_window = context_window_limit

        return llm_config

    def get_embedding_config_from_handle(
        self, handle: str, embedding_chunk_size: int = constants.DEFAULT_EMBEDDING_CHUNK_SIZE
    ) -> EmbeddingConfig:
        provider_name, model_name = handle.split("/", 1)
        provider = self.get_provider_from_name(provider_name)

        embedding_configs = [config for config in provider.list_embedding_models() if config.embedding_model == model_name]
        if not embedding_configs:
            raise ValueError(f"Embedding model {model_name} is not supported by {provider_name}")
        elif len(embedding_configs) > 1:
            raise ValueError(f"Multiple embedding models with name {model_name} supported by {provider_name}")
        else:
            embedding_config = embedding_configs[0]

        if embedding_chunk_size:
            embedding_config.embedding_chunk_size = embedding_chunk_size

        return embedding_config

    def get_provider_from_name(self, provider_name: str) -> Provider:
        providers = [provider for provider in self._enabled_providers if provider.name == provider_name]
        if not providers:
            raise ValueError(f"Provider {provider_name} is not supported")
        elif len(providers) > 1:
            raise ValueError(f"Multiple providers with name {provider_name} supported")
        else:
            provider = providers[0]

        return provider

    def add_llm_model(self, request: LLMConfig) -> LLMConfig:
        """Add a new LLM model"""

    def add_embedding_model(self, request: EmbeddingConfig) -> EmbeddingConfig:
        """Add a new embedding model"""

    def get_agent_context_window(self, agent_id: str, actor: User) -> ContextWindowOverview:
        mirix_agent = self.load_agent(agent_id=agent_id, actor=actor)
        return mirix_agent.get_context_window()

    def run_tool_from_source(
        self,
        actor: User,
        tool_args: Dict[str, str],
        tool_source: str,
        tool_env_vars: Optional[Dict[str, str]] = None,
        tool_source_type: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> ToolReturnMessage:
        """Run a tool from source code"""
        if tool_source_type is not None and tool_source_type != "python":
            raise ValueError("Only Python source code is supported at this time")

        # NOTE: we're creating a floating Tool object and NOT persisting to DB
        tool = Tool(
            name=tool_name,
            source_code=tool_source,
        )
        assert tool.name is not None, "Failed to create tool object"

        # TODO eventually allow using agent state in tools
        agent_state = None

        # Next, attempt to run the tool with the sandbox
        try:
            sandbox_run_result = ToolExecutionSandbox(tool.name, tool_args, actor, tool_object=tool).run(
                agent_state=agent_state, additional_env_vars=tool_env_vars
            )
            return ToolReturnMessage(
                id="null",
                tool_call_id="null",
                date=get_utc_time(),
                status=sandbox_run_result.status,
                tool_return=str(sandbox_run_result.func_return),
                stdout=sandbox_run_result.stdout,
                stderr=sandbox_run_result.stderr,
            )

        except Exception as e:
            func_return = get_friendly_error_msg(function_name=tool.name, exception_name=type(e).__name__, exception_message=str(e))
            return ToolReturnMessage(
                id="null",
                tool_call_id="null",
                date=get_utc_time(),
                status="error",
                tool_return=func_return,
                stdout=[],
                stderr=[traceback.format_exc()],
            )

    # Composio wrappers
    # def get_composio_client(self, api_key: Optional[str] = None):
    #     if api_key:
    #         return Composio(api_key=api_key)
    #     elif tool_settings.composio_api_key:
    #         return Composio(api_key=tool_settings.composio_api_key)
    #     else:
    #         return Composio()

    # def get_composio_apps(self, api_key: Optional[str] = None) -> List["AppModel"]:
    #     """Get a list of all Composio apps with actions"""
    #     apps = self.get_composio_client(api_key=api_key).apps.get()
    #     apps_with_actions = []
    #     for app in apps:
    #         # A bit of hacky logic until composio patches this
    #         if app.meta["actionsCount"] > 0 and not app.name.lower().endswith("_beta"):
    #             apps_with_actions.append(app)

    #     return apps_with_actions

    # def get_composio_actions_from_app_name(self, composio_app_name: str, api_key: Optional[str] = None) -> List["ActionModel"]:
    #     actions = self.get_composio_client(api_key=api_key).actions.get(apps=[composio_app_name])
    #     return actions

    async def send_message_to_agent(
        self,
        agent_id: str,
        actor: User,
        # role: MessageRole,
        messages: Union[List[Message], List[MessageCreate]],
        stream_steps: bool,
        stream_tokens: bool,
        # related to whether or not we return `MirixMessage`s or `Message`s
        chat_completion_mode: bool = False,
        # Support for AssistantMessage
        use_assistant_message: bool = True,
        assistant_message_tool_name: str = constants.DEFAULT_MESSAGE_TOOL,
        assistant_message_tool_kwarg: str = constants.DEFAULT_MESSAGE_TOOL_KWARG,
        metadata: Optional[dict] = None,
    ) -> Union[StreamingResponse, MirixResponse]:
        """Split off into a separate function so that it can be imported in the /chat/completion proxy."""

        # TODO: @charles is this the correct way to handle?
        include_final_message = True

        if not stream_steps and stream_tokens:
            raise HTTPException(status_code=400, detail="stream_steps must be 'true' if stream_tokens is 'true'")

        # For streaming response
        try:

            # TODO: move this logic into server.py

            # Get the generator object off of the agent's streaming interface
            # This will be attached to the POST SSE request used under-the-hood
            mirix_agent = self.load_agent(agent_id=agent_id, actor=actor)

            # Disable token streaming if not OpenAI
            # TODO: cleanup this logic
            llm_config = mirix_agent.agent_state.llm_config
            if stream_tokens and (llm_config.model_endpoint_type != "openai" or "inference.memgpt.ai" in llm_config.model_endpoint):
                warnings.warn(
                    "Token streaming is only supported for models with type 'openai' or `inference.memgpt.ai` in the model_endpoint: agent has endpoint type {llm_config.model_endpoint_type} and {llm_config.model_endpoint}. Setting stream_tokens to False."
                )
                stream_tokens = False

            # Create a new interface per request
            mirix_agent.interface = StreamingServerInterface(
                # multi_step=True,  # would we ever want to disable this?
                use_assistant_message=use_assistant_message,
                assistant_message_tool_name=assistant_message_tool_name,
                assistant_message_tool_kwarg=assistant_message_tool_kwarg,
                inner_thoughts_in_kwargs=(
                    llm_config.put_inner_thoughts_in_kwargs if llm_config.put_inner_thoughts_in_kwargs is not None else False
                ),
                # inner_thoughts_kwarg=INNER_THOUGHTS_KWARG,
            )
            streaming_interface = mirix_agent.interface
            if not isinstance(streaming_interface, StreamingServerInterface):
                raise ValueError(f"Agent has wrong type of interface: {type(streaming_interface)}")

            # Enable token-streaming within the request if desired
            streaming_interface.streaming_mode = stream_tokens
            # "chatcompletion mode" does some remapping and ignores inner thoughts
            streaming_interface.streaming_chat_completion_mode = chat_completion_mode

            # streaming_interface.allow_assistant_message = stream
            # streaming_interface.function_call_legacy_mode = stream

            # Allow AssistantMessage is desired by client
            # streaming_interface.use_assistant_message = use_assistant_message
            # streaming_interface.assistant_message_tool_name = assistant_message_tool_name
            # streaming_interface.assistant_message_tool_kwarg = assistant_message_tool_kwarg

            # Related to JSON buffer reader
            # streaming_interface.inner_thoughts_in_kwargs = (
            #     llm_config.put_inner_thoughts_in_kwargs if llm_config.put_inner_thoughts_in_kwargs is not None else False
            # )

            # Offload the synchronous message_func to a separate thread
            streaming_interface.stream_start()
            task = asyncio.create_task(
                asyncio.to_thread(
                    self.send_messages,
                    actor=actor,
                    agent_id=agent_id,
                    messages=messages,
                    interface=streaming_interface,
                    metadata=metadata,
                )
            )

            if stream_steps:
                # return a stream
                return StreamingResponse(
                    sse_async_generator(
                        streaming_interface.get_generator(),
                        usage_task=task,
                        finish_message=include_final_message,
                    ),
                    media_type="text/event-stream",
                )

            else:
                # buffer the stream, then return the list
                generated_stream = []
                async for message in streaming_interface.get_generator():
                    assert (
                        isinstance(message, MirixMessage)
                        or isinstance(message, LegacyMirixMessage)
                        or isinstance(message, MessageStreamStatus)
                    ), type(message)
                    generated_stream.append(message)
                    if message == MessageStreamStatus.done:
                        break

                # Get rid of the stream status messages
                filtered_stream = [d for d in generated_stream if not isinstance(d, MessageStreamStatus)]
                usage = await task

                # By default the stream will be messages of type MirixMessage or MirixLegacyMessage
                # If we want to convert these to Message, we can use the attached IDs
                # NOTE: we will need to de-duplicate the Messsage IDs though (since Assistant->Inner+Func_Call)
                # TODO: eventually update the interface to use `Message` and `MessageChunk` (new) inside the deque instead
                return MirixResponse(messages=filtered_stream, usage=usage)

        except HTTPException:
            raise
        except Exception as e:
            print(e)
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"{e}")
