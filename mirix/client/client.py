import logging
import time
import os
import base64
import hashlib
import shutil
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Union
from urllib.parse import urlparse

import requests

import mirix.utils
from mirix.constants import ADMIN_PREFIX, META_MEMORY_TOOLS, CORE_MEMORY_TOOLS, BASE_TOOLS, DEFAULT_HUMAN, DEFAULT_PERSONA, FUNCTION_RETURN_CHAR_LIMIT
from mirix.functions.functions import parse_source_code
from mirix.orm.errors import NoResultFound
from mirix.schemas.agent import AgentState, AgentType, CreateAgent, UpdateAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_message_content import TextContent, ImageContent, FileContent, CloudFileContent, MessageContentType

# new schemas
from mirix.schemas.enums import JobStatus, MessageRole
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.file import FileMetadata as PydanticFileMetadata
from mirix.schemas.mirix_message import MirixMessage, MirixMessageUnion
from mirix.schemas.mirix_request import MirixRequest, MirixStreamingRequest
from mirix.schemas.mirix_response import MirixResponse, MirixStreamingResponse
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, ChatMemory, CreateArchivalMemory, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate, MessageUpdate
from mirix.schemas.openai.chat_completion_response import UsageStatistics
from mirix.schemas.openai.chat_completions import ToolCall
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import E2BSandboxConfig, LocalSandboxConfig, SandboxConfig, SandboxConfigCreate, SandboxConfigUpdate
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.interface import QueuingInterface
from mirix.prompts import gpt_persona


def create_client():
    return LocalClient()


class AbstractClient(object):
    def __init__(
        self,
        debug: bool = False,
    ):
        self.debug = debug

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        raise NotImplementedError

    def create_agent(
        self,
        name: Optional[str] = None,
        agent_type: Optional[AgentType] = AgentType.chat_agent,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        memory=None,
        block_ids: Optional[List[str]] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        tool_rules: Optional[List[BaseToolRule]] = None,
        include_base_tools: Optional[bool] = True,
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentState:
        raise NotImplementedError

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
        memory: Optional[Memory] = None,
        tags: Optional[List[str]] = None,
    ):
        raise NotImplementedError

    def get_tools_from_agent(self, agent_id: str):
        raise NotImplementedError

    def add_tool_to_agent(self, agent_id: str, tool_id: str):
        raise NotImplementedError

    def remove_tool_from_agent(self, agent_id: str, tool_id: str):
        raise NotImplementedError

    def rename_agent(self, agent_id: str, new_name: str):
        raise NotImplementedError

    def delete_agent(self, agent_id: str):
        raise NotImplementedError

    def get_agent(self, agent_id: str) -> AgentState:
        raise NotImplementedError

    def get_agent_id(self, agent_name: str) -> AgentState:
        raise NotImplementedError

    def get_in_context_memory(self, agent_id: str) -> Memory:
        raise NotImplementedError

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        raise NotImplementedError

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        raise NotImplementedError

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        raise NotImplementedError

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        raise NotImplementedError

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        stream: Optional[bool] = False,
        stream_steps: bool = False,
        stream_tokens: bool = False,
    ) -> MirixResponse:
        raise NotImplementedError

    def user_message(self, agent_id: str, message: str) -> MirixResponse:
        raise NotImplementedError

    def create_human(self, name: str, text: str) -> Human:
        raise NotImplementedError

    def create_persona(self, name: str, text: str) -> Persona:
        raise NotImplementedError

    def list_humans(self) -> List[Human]:
        raise NotImplementedError

    def list_personas(self) -> List[Persona]:
        raise NotImplementedError

    def update_human(self, human_id: str, text: str) -> Human:
        raise NotImplementedError

    def update_persona(self, persona_id: str, text: str) -> Persona:
        raise NotImplementedError

    def get_persona(self, id: str) -> Persona:
        raise NotImplementedError

    def get_human(self, id: str) -> Human:
        raise NotImplementedError

    def get_persona_id(self, name: str) -> str:
        raise NotImplementedError

    def get_human_id(self, name: str) -> str:
        raise NotImplementedError

    def delete_persona(self, id: str):
        raise NotImplementedError

    def delete_human(self, id: str):
        raise NotImplementedError

    def load_langchain_tool(self, langchain_tool: "LangChainBaseTool", additional_imports_module_attr_map: dict[str, str] = None) -> Tool:
        raise NotImplementedError

    def load_composio_tool(self, action: "ActionType") -> Tool:
        raise NotImplementedError

    def create_tool(
        self, func, name: Optional[str] = None, tags: Optional[List[str]] = None, return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT
    ) -> Tool:
        raise NotImplementedError

    def create_or_update_tool(
        self, func, name: Optional[str] = None, tags: Optional[List[str]] = None, return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT
    ) -> Tool:
        raise NotImplementedError

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        func: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        raise NotImplementedError

    def list_tools(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Tool]:
        raise NotImplementedError

    def get_tool(self, id: str) -> Tool:
        raise NotImplementedError

    def delete_tool(self, id: str):
        raise NotImplementedError

    def get_tool_id(self, name: str) -> Optional[str]:
        raise NotImplementedError

    def upsert_base_tools(self) -> List[Tool]:
        raise NotImplementedError

    def get_messages(
        self, agent_id: str, before: Optional[str] = None, after: Optional[str] = None, limit: Optional[int] = 1000
    ) -> List[Message]:
        raise NotImplementedError

    def list_model_configs(self) -> List[LLMConfig]:
        raise NotImplementedError

    def list_embedding_configs(self) -> List[EmbeddingConfig]:
        raise NotImplementedError

    def create_org(self, name: Optional[str] = None) -> Organization:
        raise NotImplementedError

    def list_orgs(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Organization]:
        raise NotImplementedError

    def delete_org(self, org_id: str) -> Organization:
        raise NotImplementedError

    def create_sandbox_config(self, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """
        Create a new sandbox configuration.

        Args:
            config (Union[LocalSandboxConfig, E2BSandboxConfig]): The sandbox settings.

        Returns:
            SandboxConfig: The created sandbox configuration.
        """
        raise NotImplementedError

    def update_sandbox_config(self, sandbox_config_id: str, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """
        Update an existing sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to update.
            config (Union[LocalSandboxConfig, E2BSandboxConfig]): The updated sandbox settings.

        Returns:
            SandboxConfig: The updated sandbox configuration.
        """
        raise NotImplementedError

    def delete_sandbox_config(self, sandbox_config_id: str) -> None:
        """
        Delete a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to delete.
        """
        raise NotImplementedError

    def list_sandbox_configs(self, limit: int = 50, cursor: Optional[str] = None) -> List[SandboxConfig]:
        """
        List all sandbox configurations.

        Args:
            limit (int, optional): The maximum number of sandbox configurations to return. Defaults to 50.
            cursor (Optional[str], optional): The pagination cursor for retrieving the next set of results.

        Returns:
            List[SandboxConfig]: A list of sandbox configurations.
        """
        raise NotImplementedError

    def create_sandbox_env_var(
        self, sandbox_config_id: str, key: str, value: str, description: Optional[str] = None
    ) -> SandboxEnvironmentVariable:
        """
        Create a new environment variable for a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to associate the environment variable with.
            key (str): The name of the environment variable.
            value (str): The value of the environment variable.
            description (Optional[str], optional): A description of the environment variable. Defaults to None.

        Returns:
            SandboxEnvironmentVariable: The created environment variable.
        """
        raise NotImplementedError

    def update_sandbox_env_var(
        self, env_var_id: str, key: Optional[str] = None, value: Optional[str] = None, description: Optional[str] = None
    ) -> SandboxEnvironmentVariable:
        """
        Update an existing environment variable.

        Args:
            env_var_id (str): The ID of the environment variable to update.
            key (Optional[str], optional): The updated name of the environment variable. Defaults to None.
            value (Optional[str], optional): The updated value of the environment variable. Defaults to None.
            description (Optional[str], optional): The updated description of the environment variable. Defaults to None.

        Returns:
            SandboxEnvironmentVariable: The updated environment variable.
        """
        raise NotImplementedError

    def delete_sandbox_env_var(self, env_var_id: str) -> None:
        """
        Delete an environment variable by its ID.

        Args:
            env_var_id (str): The ID of the environment variable to delete.
        """
        raise NotImplementedError

    def list_sandbox_env_vars(
        self, sandbox_config_id: str, limit: int = 50, cursor: Optional[str] = None
    ) -> List[SandboxEnvironmentVariable]:
        """
        List all environment variables associated with a sandbox configuration.

        Args:
            sandbox_config_id (str): The ID of the sandbox configuration to retrieve environment variables for.
            limit (int, optional): The maximum number of environment variables to return. Defaults to 50.
            cursor (Optional[str], optional): The pagination cursor for retrieving the next set of results.

        Returns:
            List[SandboxEnvironmentVariable]: A list of environment variables.
        """
        raise NotImplementedError

class LocalClient(AbstractClient):
    """
    A local client for Mirix, which corresponds to a single user.

    Attributes:
        user_id (str): The user ID.
        debug (bool): Whether to print debug information.
        interface (QueuingInterface): The interface for the client.
        server (SyncServer): The server for the client.
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        debug: bool = False,
        default_llm_config: Optional[LLMConfig] = None,
        default_embedding_config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initializes a new instance of Client class.

        Args:
            user_id (str): The user ID.
            debug (bool): Whether to print debug information.
        """

        from mirix.server.server import SyncServer

        # set logging levels
        mirix.utils.DEBUG = debug
        logging.getLogger().setLevel(logging.CRITICAL)

        # save default model config
        self._default_llm_config = default_llm_config
        self._default_embedding_config = default_embedding_config

        # create server
        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface_factory=lambda: self.interface)
        
        # initialize file manager
        from mirix.services.file_manager import FileManager
        self.file_manager = FileManager()

        # save org_id that `LocalClient` is associated with
        if org_id:
            self.org_id = org_id
        else:
            self.org_id = self.server.organization_manager.DEFAULT_ORG_ID
        # save user_id that `LocalClient` is associated with
        if user_id:
            self.user_id = user_id
        else:
            # get default user
            self.user_id = self.server.user_manager.DEFAULT_USER_ID

        self.user = self.server.user_manager.get_user_or_default(self.user_id)
        self.organization = self.server.get_organization_or_default(self.org_id)
        
        # get images directory from settings and ensure it exists
        # Can be customized via MIRIX_IMAGES_DIR environment variable
        from mirix.settings import settings
        self.images_dir = Path(settings.images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _generate_file_hash(self, content: bytes) -> str:
        """Generate a unique hash for file content to avoid duplicates."""
        return hashlib.sha256(content).hexdigest()[:16]

    def _save_image_from_base64(self, base64_data: str, detail: str = "auto") -> FileMetadata:
        """Save an image from base64 data and return FileMetadata."""
        try:
            # Parse the data URL format: data:image/jpeg;base64,{data}
            if base64_data.startswith('data:'):
                header, encoded = base64_data.split(',', 1)
                # Extract MIME type from header
                mime_type = header.split(':')[1].split(';')[0]
                file_extension = mime_type.split('/')[-1]
            else:
                # Assume it's just base64 data without header
                encoded = base64_data
                mime_type = 'image/jpeg'
                file_extension = 'jpg'
            
            # Decode base64 data
            image_data = base64.b64decode(encoded)
            
            # Generate unique filename using hash
            file_hash = self._generate_file_hash(image_data)
            file_name = f"image_{file_hash}.{file_extension}"
            file_path = self.images_dir / file_name
            
            # Check if file already exists
            if not file_path.exists():
                # Save the image data
                with open(file_path, 'wb') as f:
                    f.write(image_data)

            # Create FileMetadata
            file_metadata = self.file_manager.create_file_metadata_from_path(
                file_path=str(file_path),
                organization_id=self.org_id
            )
            
            return file_metadata
            
        except Exception as e:
            raise ValueError(f"Failed to save base64 image: {str(e)}")

    def _save_image_from_url(self, url: str, detail: str = "auto") -> FileMetadata:
        """Download and save an image from URL and return FileMetadata."""
        try:
            # Download the image
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get content type and determine file extension
            content_type = response.headers.get('content-type', 'image/jpeg')
            file_extension = content_type.split('/')[-1]
            if file_extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                file_extension = 'jpg'
            
            # Get the image content
            image_data = response.content
            
            # Generate unique filename using hash
            file_hash = self._generate_file_hash(image_data)
            file_name = f"image_{file_hash}.{file_extension}"
            file_path = self.images_dir / file_name
            
            # Check if file already exists
            if not file_path.exists():
                # Save the image data
                with open(file_path, 'wb') as f:
                    f.write(image_data)
            
            # Create FileMetadata
            file_metadata = self.file_manager.create_file_metadata_from_path(
                file_path=str(file_path),
                organization_id=self.org_id
            )
            
            return file_metadata
            
        except Exception as e:
            raise ValueError(f"Failed to download and save image from URL {url}: {str(e)}")

    def _save_image_from_file_uri(self, file_uri: str) -> FileMetadata:
        """Copy an image from file URI and return FileMetadata."""
        try:
            # Parse file URI (could be file:// or just a local path)
            if file_uri.startswith('file://'):
                source_path = file_uri[7:]  # Remove 'file://' prefix
            else:
                source_path = file_uri
            
            source_path = Path(source_path)
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Read the file content
            with open(source_path, 'rb') as f:
                image_data = f.read()
            
            # Generate unique filename using hash
            file_hash = self._generate_file_hash(image_data)
            file_extension = source_path.suffix.lstrip('.') or 'jpg'
            file_name = f"image_{file_hash}.{file_extension}"
            file_path = self.images_dir / file_name
            
            # Check if file already exists
            if not file_path.exists():
                # Copy the file
                shutil.copy2(source_path, file_path)
            
            # Create FileMetadata
            file_metadata = self.file_manager.create_file_metadata_from_path(
                file_path=str(file_path),
                organization_id=self.org_id
            )
            
            return file_metadata
            
        except Exception as e:
            raise ValueError(f"Failed to copy image from file URI {file_uri}: {str(e)}")

    def _save_image_from_google_cloud_uri(self, cloud_uri: str) -> FileMetadata:
        """Create FileMetadata from Google Cloud URI without downloading the image.
        
        Google Cloud URIs are not directly downloadable and should be stored as remote references
        in the source_url field, similar to how regular HTTP URLs are handled.
        """
        # Parse URI to get file name - Google Cloud URIs typically come in the format:
        # https://generativelanguage.googleapis.com/v1beta/files/{file_id}
        from urllib.parse import urlparse
        parsed_uri = urlparse(cloud_uri)
        
        # Extract file ID from path if available, otherwise use generic name
        file_id = os.path.basename(parsed_uri.path) or "google_cloud_file"
        file_name = f"google_cloud_{file_id}"
        
        # Ensure file name has an extension
        if not os.path.splitext(file_name)[1]:
            file_name += ".jpg"  # Default to jpg for images without extension
        
        # Determine MIME type from extension or default to image/jpeg
        file_extension = os.path.splitext(file_name)[1].lower()
        file_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml'
        }
        file_type = file_type_map.get(file_extension, 'image/jpeg')

        # Create FileMetadata with Google Cloud URI in google_cloud_url field
        file_metadata = self.file_manager.create_file_metadata(
            PydanticFileMetadata(
                organization_id=self.org_id,
                file_name=file_name,
                file_path=None,  # No local path for Google Cloud URIs
                source_url=None,  # No regular source URL for Google Cloud files
                google_cloud_url=cloud_uri,  # Store Google Cloud URI in the dedicated field
                file_type=file_type,
                file_size=None,  # Unknown size for remote Google Cloud files
                file_creation_date=None,
                file_last_modified_date=None,
            )
        )
        
        return file_metadata

    def _save_file_from_path(self, file_path: str) -> FileMetadata:
        """Save a file from local path and return FileMetadata."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Create FileMetadata using the file manager
            file_metadata = self.file_manager.create_file_metadata_from_path(
                file_path=str(file_path),
                organization_id=self.org_id
            )
            
            return file_metadata
            
        except Exception as e:
            raise ValueError(f"Failed to save file from path {file_path}: {str(e)}")

    def _determine_file_type(self, file_path: str) -> str:
        """Determine file type from file extension."""
        file_extension = os.path.splitext(file_path)[1].lower()
        file_type_map = {
            # Images
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
            # Documents
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.rtf': 'application/rtf',
            '.html': 'text/html',
            '.htm': 'text/html',
            # Spreadsheets
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.csv': 'text/csv',
            # Presentations
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            # Other common formats
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.zip': 'application/zip',
        }
        return file_type_map.get(file_extension, 'application/octet-stream')

    def _create_file_metadata_from_url(self, url: str, detail: str = "auto") -> FileMetadata:
        """Create FileMetadata from URL without downloading the image.
        
        The URL is stored in the source_url field, not file_path, to clearly
        distinguish between local files and remote resources.
        """
        try:
            # Parse URL to get file name
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path) or "remote_image"
            
            # Ensure file name has an extension
            if not os.path.splitext(file_name)[1]:
                file_name += ".jpg"  # Default to jpg for images without extension
            
            # Determine MIME type from extension or default to image/jpeg
            file_extension = os.path.splitext(file_name)[1].lower()
            file_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml'
            }
            file_type = file_type_map.get(file_extension, 'image/jpeg')
            
            # Create FileMetadata with URL in source_url field
            file_metadata = self.file_manager.create_file_metadata(
                PydanticFileMetadata(
                    organization_id=self.org_id,
                    file_name=file_name,
                    file_path=None,  # No local path for remote URLs
                    source_url=url,  # Store URL in the dedicated field
                    file_type=file_type,
                    file_size=None,  # Unknown size for remote URLs
                    file_creation_date=None,
                    file_last_modified_date=None,
                )
            )
            
            return file_metadata
            
        except Exception as e:
            raise ValueError(f"Failed to create file metadata from URL {url}: {str(e)}")

    # agents
    def list_agents(
        self, query_text: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 100, cursor: Optional[str] = None
    ) -> List[AgentState]:
        self.interface.clear()

        return self.server.agent_manager.list_agents(actor=self.server.user_manager.get_user_by_id(self.user.id), tags=tags, query_text=query_text, limit=limit, cursor=cursor)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        """
        Check if an agent exists

        Args:
            agent_id (str): ID of the agent
            agent_name (str): Name of the agent

        Returns:
            exists (bool): `True` if the agent exists, `False` otherwise
        """

        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existing = self.list_agents()
        if agent_id:
            return str(agent_id) in [str(agent.id) for agent in existing]
        else:
            return agent_name in [str(agent.name) for agent in existing]

    def create_agent(
        self,
        name: Optional[str] = None,
        # agent config
        agent_type: Optional[AgentType] = AgentType.chat_agent,
        # model configs
        embedding_config: EmbeddingConfig = None,
        llm_config: LLMConfig = None,
        # memory
        memory: Memory = None,
        block_ids: Optional[List[str]] = None,
        system: Optional[str] = None,
        # tools
        tool_ids: Optional[List[str]] = None,
        tool_rules: Optional[List[BaseToolRule]] = None,
        include_base_tools: Optional[bool] = True,
        include_meta_memory_tools: Optional[bool] = False,
        # metadata
        metadata: Optional[Dict] = {"human:": DEFAULT_HUMAN, "persona": DEFAULT_PERSONA},
        description: Optional[str] = None,
        initial_message_sequence: Optional[List[Message]] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentState:
        """Create an agent

        Args:
            name (str): Name of the agent
            embedding_config (EmbeddingConfig): Embedding configuration
            llm_config (LLMConfig): LLM configuration
            memory_blocks (List[Dict]): List of configurations for the memory blocks (placed in core-memory)
            system (str): System configuration
            tools (List[str]): List of tools
            tool_rules (Optional[List[BaseToolRule]]): List of tool rules
            include_base_tools (bool): Include base tools
            metadata (Dict): Metadata
            description (str): Description
            tags (List[str]): Tags for filtering agents

        Returns:
            agent_state (AgentState): State of the created agent
        """
        # construct list of tools
        tool_ids = tool_ids or []
        tool_names = []
        if include_base_tools:
            tool_names += BASE_TOOLS
        if include_meta_memory_tools:
            tool_names += META_MEMORY_TOOLS
        tool_ids += [self.server.tool_manager.get_tool_by_name(tool_name=name, actor=self.server.user_manager.get_user_by_id(self.user.id)).id for name in tool_names]

        # check if default configs are provided
        assert embedding_config or self._default_embedding_config, f"Embedding config must be provided"
        assert llm_config or self._default_llm_config, f"LLM config must be provided"

        # TODO: This should not happen here, we need to have clear separation between create/add blocks
        for block in memory.get_blocks():
            self.server.block_manager.create_or_update_block(block, actor=self.server.user_manager.get_user_by_id(self.user.id))

        # Also get any existing block_ids passed in
        block_ids = block_ids or []

        # create agent
        # Create the base parameters
        create_params = {
            "description": description,
            "metadata_": metadata,
            "memory_blocks": [],
            "block_ids": [b.id for b in memory.get_blocks()] + block_ids,
            "tool_ids": tool_ids,
            "tool_rules": tool_rules,
            "include_base_tools": include_base_tools,
            "system": system,
            "agent_type": agent_type,
            "llm_config": llm_config if llm_config else self._default_llm_config,
            "embedding_config": embedding_config if embedding_config else self._default_embedding_config,
            "initial_message_sequence": initial_message_sequence,
            "tags": tags,
        }

        # Only add name if it's not None
        if name is not None:
            create_params["name"] = name

        agent_state = self.server.create_agent(
            CreateAgent(**create_params),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )

        # TODO: get full agent state
        return self.server.agent_manager.get_agent_by_id(agent_state.id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def update_message(
        self,
        agent_id: str,
        message_id: str,
        role: Optional[MessageRole] = None,
        text: Optional[str] = None,
        name: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        message = self.server.update_agent_message(
            agent_id=agent_id,
            message_id=message_id,
            request=MessageUpdate(
                role=role,
                text=text,
                name=name,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            ),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )
        return message

    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        message_ids: Optional[List[str]] = None,
    ):
        """
        Update an existing agent

        Args:
            agent_id (str): ID of the agent
            name (str): Name of the agent
            description (str): Description of the agent
            system (str): System configuration
            tools (List[str]): List of tools
            metadata (Dict): Metadata
            llm_config (LLMConfig): LLM configuration
            embedding_config (EmbeddingConfig): Embedding configuration
            message_ids (List[str]): List of message IDs
            tags (List[str]): Tags for filtering agents

        Returns:
            agent_state (AgentState): State of the updated agent
        """
        # TODO: add the abilitty to reset linked block_ids
        self.interface.clear()
        agent_state = self.server.agent_manager.update_agent(
            agent_id,
            UpdateAgent(
                name=name,
                system=system,
                tool_ids=tool_ids,
                tags=tags,
                description=description,
                metadata_=metadata,
                llm_config=llm_config,
                embedding_config=embedding_config,
                message_ids=message_ids,
            ),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )
        return agent_state

    def get_tools_from_agent(self, agent_id: str) -> List[Tool]:
        """
        Get tools from an existing agent.

        Args:
            agent_id (str): ID of the agent

        Returns:
            List[Tool]: A list of Tool objs
        """
        self.interface.clear()
        return self.server.agent_manager.get_agent_by_id(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id)).tools

    def add_tool_to_agent(self, agent_id: str, tool_id: str):
        """
        Add tool to an existing agent

        Args:
            agent_id (str): ID of the agent
            tool_id (str): A tool id

        Returns:
            agent_state (AgentState): State of the updated agent
        """
        self.interface.clear()
        agent_state = self.server.agent_manager.attach_tool(agent_id=agent_id, tool_id=tool_id, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return agent_state

    def remove_tool_from_agent(self, agent_id: str, tool_id: str):
        """
        Removes tools from an existing agent

        Args:
            agent_id (str): ID of the agent
            tool_id (str): The tool id

        Returns:
            agent_state (AgentState): State of the updated agent
        """
        self.interface.clear()
        agent_state = self.server.agent_manager.detach_tool(agent_id=agent_id, tool_id=tool_id, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return agent_state

    def rename_agent(self, agent_id: str, new_name: str):
        """
        Rename an agent

        Args:
            agent_id (str): ID of the agent
            new_name (str): New name for the agent
        """
        self.update_agent(agent_id, name=new_name)

    def delete_agent(self, agent_id: str):
        """
        Delete an agent

        Args:
            agent_id (str): ID of the agent to delete
        """
        self.server.agent_manager.delete_agent(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_agent_by_name(self, agent_name: str) -> AgentState:
        """
        Get an agent by its name

        Args:
            agent_name (str): Name of the agent

        Returns:
            agent_state (AgentState): State of the agent
        """
        self.interface.clear()
        return self.server.agent_manager.get_agent_by_name(agent_name=agent_name, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_agent(self, agent_id: str) -> AgentState:
        """
        Get an agent's state by its ID.

        Args:
            agent_id (str): ID of the agent

        Returns:
            agent_state (AgentState): State representation of the agent
        """
        self.interface.clear()
        return self.server.agent_manager.get_agent_by_id(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_agent_id(self, agent_name: str) -> Optional[str]:
        """
        Get the ID of an agent by name (names are unique per user)

        Args:
            agent_name (str): Name of the agent

        Returns:
            agent_id (str): ID of the agent
        """

        self.interface.clear()
        assert agent_name, f"Agent name must be provided"

        # TODO: Refactor this futher to not have downstream users expect Optionals - this should just error
        try:
            return self.server.agent_manager.get_agent_by_name(agent_name=agent_name, actor=self.server.user_manager.get_user_by_id(self.user.id)).id
        except NoResultFound:
            return None

    # memory
    def get_in_context_memory(self, agent_id: str) -> Memory:
        """
        Get the in-context (i.e. core) memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): In-context memory of the agent
        """
        memory = self.server.get_agent_memory(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return memory

    def get_core_memory(self, agent_id: str) -> Memory:
        return self.get_in_context_memory(agent_id)

    def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        """
        Update the in-context memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            memory (Memory): The updated in-context memory of the agent

        """
        # TODO: implement this (not sure what it should look like)
        memory = self.server.update_agent_core_memory(agent_id=agent_id, label=section, value=value, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return memory

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        """
        Get a summary of the archival memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (ArchivalMemorySummary): Summary of the archival memory

        """
        return self.server.get_archival_memory_summary(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        """
        Get a summary of the recall memory of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            summary (RecallMemorySummary): Summary of the recall memory
        """
        return self.server.get_recall_memory_summary(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """
        Get in-context messages of an agent

        Args:
            agent_id (str): ID of the agent

        Returns:
            messages (List[Message]): List of in-context messages
        """
        return self.server.agent_manager.get_in_context_messages(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    # agent interactions

    def construct_system_message(self, agent_id: str, message: str, user_id: str) -> str:
        """
        Construct a system message from a message.
        """
        return self.server.construct_system_message(agent_id=agent_id, message=message, actor=self.server.user_manager.get_user_by_id(user_id))

    def send_messages(
        self,
        agent_id: str,
        messages: List[Union[Message | MessageCreate]],
    ):
        """
        Send pre-packed messages to an agent.

        Args:
            agent_id (str): ID of the agent
            messages (List[Union[Message | MessageCreate]]): List of messages to send

        Returns:
            response (MirixResponse): Response from the agent
        """
        self.interface.clear()
        usage = self.server.send_messages(actor=self.server.user_manager.get_user_by_id(self.user.id), agent_id=agent_id, 
                                          messages=messages)

        # format messages
        return MirixResponse(messages=messages, usage=usage)

    def send_message(
        self,
        message: str | list[dict],
        role: str,
        name: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        force_response: bool = False,
        existing_file_uris: Optional[List[str]] = None,
        extra_messages: Optional[List[dict]] = None,
        display_intermediate_message: any = None,
        request_user_confirmation: any = None,
        chaining: Optional[bool] = None,
        message_queue: Optional[any] = None,
        retrieved_memories: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> MirixResponse:
        """
        Send a message to an agent

        Args:
            message (str): Message to send
            role (str): Role of the message
            agent_id (str): ID of the agent
            name(str): Name of the sender
            stream (bool): Stream the response (default: `False`)
            extra_message (str): Extra message to send. It will be inserted before the last message
            chaining (bool): Whether to enable chaining for this message

        Returns:
            response (MirixResponse): Response from the agent
        """

        if not agent_id:
            # lookup agent by name
            assert agent_name, f"Either agent_id or agent_name must be provided"
            agent_id = self.get_agent_id(agent_name=agent_name)
            assert agent_id, f"Agent with name {agent_name} not found"

        if stream_steps or stream_tokens:
            # TODO: implement streaming with stream=True/False
            raise NotImplementedError
        self.interface.clear()

        if isinstance(message, str):
            content = [TextContent(text=message)]
            input_messages = [MessageCreate(role=MessageRole(role), content=content, name=name)]
        elif isinstance(message, list):
            def convert_message(m):
                if m['type'] == 'text':
                    return TextContent(**m)
                elif m['type'] == 'image_url':
                    url = m['image_url']['url']
                    detail = m['image_url'].get("detail", "auto")
                    
                    # Handle the image based on URL type
                    if url.startswith('data:'):
                        # Base64 encoded image - save locally
                        file_metadata = self._save_image_from_base64(url, detail)
                    else:
                        # HTTP URL - just create FileMetadata without downloading
                        file_metadata = self._create_file_metadata_from_url(url, detail)
                    
                    return ImageContent(
                        type=MessageContentType.image_url,
                        image_id=file_metadata.id,
                        detail=detail
                    )
                    
                elif m['type'] == 'image_data':
                    # Base64 image data (new format)
                    data = m['image_data']['data']
                    detail = m['image_data'].get("detail", "auto")
                    
                    # Save the base64 image to file_manager
                    file_metadata = self._save_image_from_base64(data, detail)
                    
                    return ImageContent(
                        type=MessageContentType.image_url,
                        image_id=file_metadata.id,
                        detail=detail
                    )
                elif m['type'] == 'file_uri':

                    # File URI (local file path)  
                    file_path = m['file_uri']
                    
                    # Check if it's an image or other file type
                    file_type = self._determine_file_type(file_path)
                    
                    if file_type.startswith('image/'):
                        # Handle as image
                        file_metadata = self._save_image_from_file_uri(file_path)
                        return ImageContent(
                            type=MessageContentType.image_url,
                            image_id=file_metadata.id,
                            detail="auto"
                        )
                    else:
                        # Handle as general file (e.g., PDF, DOC, etc.)
                        file_metadata = self._save_file_from_path(file_path)
                        return FileContent(
                            type=MessageContentType.file_uri,
                            file_id=file_metadata.id
                        )
                
                elif m['type'] == 'google_cloud_file_uri':
                    # Google Cloud file URI
                    # Handle both the typo version and the correct version from the test file
                    file_uri = m.get('google_cloud_file_uri') or m.get('file_uri')

                    file_metadata = self._save_image_from_google_cloud_uri(file_uri)
                    return CloudFileContent(
                        type=MessageContentType.google_cloud_file_uri,
                        cloud_file_uri=file_metadata.id,
                    )

                elif m['type'] == 'database_image_id':
                    return ImageContent(
                        type=MessageContentType.image_url,
                        image_id=m['image_id'],
                        detail="auto"
                    )
                
                elif m['type'] == 'database_file_id':
                    return FileContent(
                        type=MessageContentType.file_uri,
                        file_id=m['file_id'],
                    )
                
                elif m['type'] == 'database_google_cloud_file_uri':
                    return CloudFileContent(
                        type=MessageContentType.google_cloud_file_uri,
                        cloud_file_uri=m['cloud_file_uri'],
                    )

                else:
                    raise ValueError(f"Unknown message type: {m['type']}")
            
            content = [convert_message(m) for m in message]
            input_messages = [MessageCreate(role=MessageRole(role), content=content, name=name)]
            if extra_messages is not None:
                extra_messages = [MessageCreate(role=MessageRole(role), content=[convert_message(m) for m in extra_messages], name=name)]

        else:
            raise ValueError(f"Invalid message type: {type(message)}")

        usage = self.server.send_messages(
            actor=self.server.user_manager.get_user_by_id(self.user.id),
            agent_id=agent_id,
            input_messages=input_messages,
            force_response=force_response,
            display_intermediate_message=display_intermediate_message,
            request_user_confirmation=request_user_confirmation,
            chaining=chaining,
            existing_file_uris=existing_file_uris,
            extra_messages=extra_messages,
            message_queue=message_queue,
            user_id=user_id
        )

        # format messages
        messages = self.interface.to_list()

        mirix_messages = []
        for m in messages:
            mirix_messages += m.to_mirix_message()

        return MirixResponse(messages=mirix_messages, usage=usage)

    def user_message(self, agent_id: str, message: str) -> MirixResponse:
        """
        Send a message to an agent as a user

        Args:
            agent_id (str): ID of the agent
            message (str): Message to send

        Returns:
            response (MirixResponse): Response from the agent
        """
        self.interface.clear()
        return self.send_message(role="user", agent_id=agent_id, message=message)

    def run_command(self, agent_id: str, command: str) -> MirixResponse:
        """
        Run a command on the agent

        Args:
            agent_id (str): The agent ID
            command (str): The command to run

        Returns:
            MirixResponse: The response from the agent

        """
        self.interface.clear()
        usage = self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

        # NOTE: messages/usage may be empty, depending on the command
        return MirixResponse(messages=self.interface.to_list(), usage=usage)

    # archival memory

    # humans / personas

    def get_block_id(self, name: str, label: str) -> str:
        block = self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), template_name=name, label=label, is_template=True)
        if not block:
            return None
        return block[0].id

    def create_human(self, name: str, text: str):
        """
        Create a human block template (saved human string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Human block
        """
        return self.server.block_manager.create_or_update_block(Human(template_name=name, value=text), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def create_persona(self, name: str, text: str):
        """
        Create a persona block template (saved persona string to pre-fill `ChatMemory`)

        Args:
            name (str): Name of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Persona block
        """
        return self.server.block_manager.create_or_update_block(Persona(template_name=name, value=text), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def list_humans(self):
        """
        List available human block templates

        Returns:
            humans (List[Human]): List of human blocks
        """
        return self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), label="human", is_template=True)

    def list_personas(self) -> List[Persona]:
        """
        List available persona block templates

        Returns:
            personas (List[Persona]): List of persona blocks
        """
        return self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), label="persona", is_template=True)

    def update_human(self, human_id: str, text: str):
        """
        Update a human block template

        Args:
            human_id (str): ID of the human block
            text (str): Text of the human block

        Returns:
            human (Human): Updated human block
        """

        return self.server.block_manager.update_block(
            block_id=human_id, block_update=UpdateHuman(value=text, is_template=True), actor=self.server.user_manager.get_user_by_id(self.user.id)
        )

    def update_persona(self, persona_id: str, text: str):
        """
        Update a persona block template

        Args:
            persona_id (str): ID of the persona block
            text (str): Text of the persona block

        Returns:
            persona (Persona): Updated persona block
        """
        blocks = self.server.block_manager.get_blocks(self.user)
        persona_block = [block for block in blocks if block.label == 'persona'][0]
        return self.server.block_manager.update_block(
            block_id=persona_block.id, block_update=BlockUpdate(value=text), actor=self.server.user_manager.get_user_by_id(self.user.id)
        )

    def update_persona_text(self, persona_name: str, text: str):
        """
        Update a persona block template by template name

        Args:
            persona_name (str): Name of the persona template
            text (str): Text of the persona block

        Returns:
            persona (Persona): Updated persona block
        """
        persona_id = self.get_persona_id(persona_name)
        if persona_id:
            # Update existing persona
            return self.server.block_manager.update_block(
                block_id=persona_id, 
                block_update=BlockUpdate(value=text, is_template=True), 
                actor=self.server.user_manager.get_user_by_id(self.user.id)
            )
        else:
            # Create new persona if it doesn't exist
            return self.create_persona(persona_name, text)

    def get_persona(self, id: str) -> Persona:
        """
        Get a persona block template

        Args:
            id (str): ID of the persona block

        Returns:
            persona (Persona): Persona block
        """
        assert id, f"Persona ID must be provided"
        return Persona(**self.server.block_manager.get_block_by_id(id, actor=self.server.user_manager.get_user_by_id(self.user.id)).model_dump())

    def get_human(self, id: str) -> Human:
        """
        Get a human block template

        Args:
            id (str): ID of the human block

        Returns:
            human (Human): Human block
        """
        assert id, f"Human ID must be provided"
        return Human(**self.server.block_manager.get_block_by_id(id, actor=self.server.user_manager.get_user_by_id(self.user.id)).model_dump())

    def get_persona_id(self, name: str) -> str:
        """
        Get the ID of a persona block template

        Args:
            name (str): Name of the persona block

        Returns:
            id (str): ID of the persona block
        """
        persona = self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), template_name=name, label="persona", is_template=True)
        if not persona:
            return None
        return persona[0].id

    def get_human_id(self, name: str) -> str:
        """
        Get the ID of a human block template

        Args:
            name (str): Name of the human block

        Returns:
            id (str): ID of the human block
        """
        human = self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), template_name=name, label="human", is_template=True)
        if not human:
            return None
        return human[0].id

    def delete_persona(self, id: str):
        """
        Delete a persona block template

        Args:
            id (str): ID of the persona block
        """
        self.delete_block(id)

    def delete_human(self, id: str):
        """
        Delete a human block template

        Args:
            id (str): ID of the human block
        """
        self.delete_block(id)

    # tools
    def load_langchain_tool(self, langchain_tool: "LangChainBaseTool", additional_imports_module_attr_map: dict[str, str] = None) -> Tool:
        tool_create = ToolCreate.from_langchain(
            langchain_tool=langchain_tool,
            additional_imports_module_attr_map=additional_imports_module_attr_map,
        )
        return self.server.tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def load_crewai_tool(self, crewai_tool: "CrewAIBaseTool", additional_imports_module_attr_map: dict[str, str] = None) -> Tool:
        tool_create = ToolCreate.from_crewai(
            crewai_tool=crewai_tool,
            additional_imports_module_attr_map=additional_imports_module_attr_map,
        )
        return self.server.tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def load_composio_tool(self, action: "ActionType") -> Tool:
        tool_create = ToolCreate.from_composio(action_name=action.name)
        return self.server.tool_manager.create_or_update_tool(pydantic_tool=Tool(**tool_create.model_dump()), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """
        Create a tool. This stores the source code of function on the server, so that the server can execute the function and generate an OpenAI JSON schemas for it when using with an agent.

        Args:
            func (callable): The function to create a tool for.
            name: (str): Name of the tool (must be unique per-user.)
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            description (str, optional): The description.
            return_char_limit (int): The character limit for the tool's return value. Defaults to FUNCTION_RETURN_CHAR_LIMIT.

        Returns:
            tool (Tool): The created tool.
        """
        # TODO: check if tool already exists
        # TODO: how to load modules?
        # parse source code/schema
        source_code = parse_source_code(func)
        source_type = "python"
        if not tags:
            tags = []

        # call server function
        return self.server.tool_manager.create_tool(
            Tool(
                source_type=source_type,
                source_code=source_code,
                name=name,
                tags=tags,
                description=description,
                return_char_limit=return_char_limit,
            ),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )

    def create_or_update_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """
        Creates or updates a tool. This stores the source code of function on the server, so that the server can execute the function and generate an OpenAI JSON schemas for it when using with an agent.

        Args:
            func (callable): The function to create a tool for.
            name: (str): Name of the tool (must be unique per-user.)
            tags (Optional[List[str]], optional): Tags for the tool. Defaults to None.
            description (str, optional): The description.
            return_char_limit (int): The character limit for the tool's return value. Defaults to FUNCTION_RETURN_CHAR_LIMIT.

        Returns:
            tool (Tool): The created tool.
        """
        source_code = parse_source_code(func)
        source_type = "python"
        if not tags:
            tags = []

        # call server function
        return self.server.tool_manager.create_or_update_tool(
            Tool(
                source_type=source_type,
                source_code=source_code,
                name=name,
                tags=tags,
                description=description,
                return_char_limit=return_char_limit,
            ),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        func: Optional[callable] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """
        Update a tool with provided parameters (name, func, tags)

        Args:
            id (str): ID of the tool
            name (str): Name of the tool
            func (callable): Function to wrap in a tool
            tags (List[str]): Tags for the tool
            return_char_limit (int): The character limit for the tool's return value. Defaults to FUNCTION_RETURN_CHAR_LIMIT.

        Returns:
            tool (Tool): Updated tool
        """
        update_data = {
            "source_type": "python",  # Always include source_type
            "source_code": parse_source_code(func) if func else None,
            "tags": tags,
            "name": name,
            "description": description,
            "return_char_limit": return_char_limit,
        }

        # Filter out any None values from the dictionary
        update_data = {key: value for key, value in update_data.items() if value is not None}

        return self.server.tool_manager.update_tool_by_id(tool_id=id, tool_update=ToolUpdate(**update_data), actor=self.server.user_manager.get_user_by_id(self.user.id))

    def list_tools(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Tool]:
        """
        List available tools for the user.

        Returns:
            tools (List[Tool]): List of tools
        """
        return self.server.tool_manager.list_tools(cursor=cursor, limit=limit, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_tool(self, id: str) -> Optional[Tool]:
        """
        Get a tool given its ID.

        Args:
            id (str): ID of the tool

        Returns:
            tool (Tool): Tool
        """
        return self.server.tool_manager.get_tool_by_id(id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def delete_tool(self, id: str):
        """
        Delete a tool given the ID.

        Args:
            id (str): ID of the tool
        """
        return self.server.tool_manager.delete_tool_by_id(id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def get_tool_id(self, name: str) -> Optional[str]:
        """
        Get the ID of a tool from its name. The client will use the org_id it is configured with.

        Args:
            name (str): Name of the tool

        Returns:
            id (str): ID of the tool (`None` if not found)
        """
        tool = self.server.tool_manager.get_tool_by_name(tool_name=name, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return tool.id if tool else None

    # recall memory

    def get_messages(self, agent_id: str, cursor: Optional[str] = None, limit: Optional[int] = 1000) -> List[Message]:
        """
        Get messages from an agent with pagination.

        Args:
            agent_id (str): ID of the agent
            cursor (str): Get messages after a certain time
            limit (int): Limit number of messages

        Returns:
            messages (List[Message]): List of messages
        """

        self.interface.clear()
        return self.server.get_agent_recall_cursor(
            user_id=self.user_id,
            agent_id=agent_id,
            before=cursor,
            limit=limit,
            reverse=True,
        )

    def list_blocks(self, label: Optional[str] = None, templates_only: Optional[bool] = True) -> List[Block]:
        """
        List available blocks

        Args:
            label (str): Label of the block
            templates_only (bool): List only templates

        Returns:
            blocks (List[Block]): List of blocks
        """
        return self.server.block_manager.get_blocks(actor=self.server.user_manager.get_user_by_id(self.user.id), label=label, is_template=templates_only)

    def create_block(
        self, label: str, value: str, limit: Optional[int] = None, template_name: Optional[str] = None, is_template: bool = False
    ) -> Block:  #
        """
        Create a block

        Args:
            label (str): Label of the block
            name (str): Name of the block
            text (str): Text of the block
            limit (int): Character of the block

        Returns:
            block (Block): Created block
        """
        block = Block(label=label, template_name=template_name, value=value, is_template=is_template, limit=limit)
        # if limit:
        #     block.limit = limit
        return self.server.block_manager.create_or_update_block(block, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def update_block(self, block_id: str, name: Optional[str] = None, text: Optional[str] = None, limit: Optional[int] = None) -> Block:
        """
        Update a block

        Args:
            block_id (str): ID of the block
            name (str): Name of the block
            text (str): Text of the block

        Returns:
            block (Block): Updated block
        """
        return self.server.block_manager.update_block(
            block_id=block_id,
            block_update=BlockUpdate(template_name=name, value=text, limit=limit if limit else self.get_block(block_id).limit),
            actor=self.server.user_manager.get_user_by_id(self.user.id),
        )

    def get_block(self, block_id: str) -> Block:
        """
        Get a block

        Args:
            block_id (str): ID of the block

        Returns:
            block (Block): Block
        """
        return self.server.block_manager.get_block_by_id(block_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def delete_block(self, id: str) -> Block:
        """
        Delete a block

        Args:
            id (str): ID of the block

        Returns:
            block (Block): Deleted block
        """
        return self.server.block_manager.delete_block(id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def set_default_llm_config(self, llm_config: LLMConfig):
        """
        Set the default LLM configuration for agents.

        Args:
            llm_config (LLMConfig): LLM configuration
        """
        self._default_llm_config = llm_config

    def set_default_embedding_config(self, embedding_config: EmbeddingConfig):
        """
        Set the default embedding configuration for agents.

        Args:
            embedding_config (EmbeddingConfig): Embedding configuration
        """
        self._default_embedding_config = embedding_config

    def list_llm_configs(self) -> List[LLMConfig]:
        """
        List available LLM configurations

        Returns:
            configs (List[LLMConfig]): List of LLM configurations
        """
        return self.server.list_llm_models()

    def list_embedding_configs(self) -> List[EmbeddingConfig]:
        """
        List available embedding configurations

        Returns:
            configs (List[EmbeddingConfig]): List of embedding configurations
        """
        return self.server.list_embedding_models()

    def create_org(self, name: Optional[str] = None) -> Organization:
        return self.server.organization_manager.create_organization(pydantic_org=Organization(name=name))

    def list_orgs(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[Organization]:
        return self.server.organization_manager.list_organizations(cursor=cursor, limit=limit)

    def delete_org(self, org_id: str) -> Organization:
        return self.server.organization_manager.delete_organization_by_id(org_id=org_id)

    def create_sandbox_config(self, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """
        Create a new sandbox configuration.
        """
        config_create = SandboxConfigCreate(config=config)
        return self.server.sandbox_config_manager.create_or_update_sandbox_config(sandbox_config_create=config_create, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def update_sandbox_config(self, sandbox_config_id: str, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """
        Update an existing sandbox configuration.
        """
        sandbox_update = SandboxConfigUpdate(config=config)
        return self.server.sandbox_config_manager.update_sandbox_config(
            sandbox_config_id=sandbox_config_id, sandbox_update=sandbox_update, actor=self.server.user_manager.get_user_by_id(self.user.id)
        )

    def delete_sandbox_config(self, sandbox_config_id: str) -> None:
        """
        Delete a sandbox configuration.
        """
        return self.server.sandbox_config_manager.delete_sandbox_config(sandbox_config_id=sandbox_config_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def list_sandbox_configs(self, limit: int = 50, cursor: Optional[str] = None) -> List[SandboxConfig]:
        """
        List all sandbox configurations.
        """
        return self.server.sandbox_config_manager.list_sandbox_configs(actor=self.server.user_manager.get_user_by_id(self.user.id), limit=limit, cursor=cursor)

    def create_sandbox_env_var(
        self, sandbox_config_id: str, key: str, value: str, description: Optional[str] = None
    ) -> SandboxEnvironmentVariable:
        """
        Create a new environment variable for a sandbox configuration.
        """
        env_var_create = SandboxEnvironmentVariableCreate(key=key, value=value, description=description)
        return self.server.sandbox_config_manager.create_sandbox_env_var(
            env_var_create=env_var_create, sandbox_config_id=sandbox_config_id, actor=self.server.user_manager.get_user_by_id(self.user.id)
        )

    def update_sandbox_env_var(
        self, env_var_id: str, key: Optional[str] = None, value: Optional[str] = None, description: Optional[str] = None
    ) -> SandboxEnvironmentVariable:
        """
        Update an existing environment variable.
        """
        env_var_update = SandboxEnvironmentVariableUpdate(key=key, value=value, description=description)
        return self.server.sandbox_config_manager.update_sandbox_env_var(
            env_var_id=env_var_id, env_var_update=env_var_update, actor=self.server.user_manager.get_user_by_id(self.user.id)
        )

    def delete_sandbox_env_var(self, env_var_id: str) -> None:
        """
        Delete an environment variable by its ID.
        """
        return self.server.sandbox_config_manager.delete_sandbox_env_var(env_var_id=env_var_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def list_sandbox_env_vars(
        self, sandbox_config_id: str, limit: int = 50, cursor: Optional[str] = None
    ) -> List[SandboxEnvironmentVariable]:
        """
        List all environment variables associated with a sandbox configuration.
        """
        return self.server.sandbox_config_manager.list_sandbox_env_vars(
            sandbox_config_id=sandbox_config_id, actor=self.server.user_manager.get_user_by_id(self.user.id), limit=limit, cursor=cursor
        )

    # file management methods
    def save_file(self, file_path: str, source_id: Optional[str] = None) -> FileMetadata:
        """
        Save a file to the file manager and return its metadata.
        
        Args:
            file_path (str): Path to the file to save
            source_id (Optional[str]): Optional source ID to associate with the file
            
        Returns:
            FileMetadata: The created file metadata
        """
        return self.file_manager.create_file_metadata_from_path(
            file_path=file_path,
            organization_id=self.org_id,
            source_id=source_id
        )

    def list_files(self, cursor: Optional[str] = None, limit: Optional[int] = 50) -> List[FileMetadata]:
        """
        List files for the current organization.
        
        Args:
            cursor (Optional[str]): Pagination cursor
            limit (Optional[int]): Maximum number of files to return
            
        Returns:
            List[FileMetadata]: List of file metadata
        """
        return self.file_manager.get_files_by_organization_id(
            organization_id=self.org_id,
            cursor=cursor,
            limit=limit
        )

    def get_file(self, file_id: str) -> FileMetadata:
        """
        Get file metadata by ID.
        
        Args:
            file_id (str): ID of the file
            
        Returns:
            FileMetadata: The file metadata
        """
        return self.file_manager.get_file_metadata_by_id(file_id)

    def delete_file(self, file_id: str) -> None:
        """
        Delete a file by ID.
        
        Args:
            file_id (str): ID of the file to delete
        """
        self.file_manager.delete_file_metadata(file_id)

    def search_files(self, name_pattern: str) -> List[FileMetadata]:
        """
        Search files by name pattern.
        
        Args:
            name_pattern (str): Pattern to search for in file names
            
        Returns:
            List[FileMetadata]: List of matching files
        """
        return self.file_manager.search_files_by_name(
            file_name=name_pattern,
            organization_id=self.org_id
        )

    def get_file_stats(self) -> dict:
        """
        Get file statistics for the current organization.
        
        Returns:
            dict: File statistics including total files, size, and types
        """
        return self.file_manager.get_file_stats(organization_id=self.org_id)

    def update_agent_memory_block_label(self, agent_id: str, current_label: str, new_label: str) -> Memory:
        """Rename a block in the agent's core memory

        Args:
            agent_id (str): The agent ID
            current_label (str): The current label of the block
            new_label (str): The new label of the block

        Returns:
            memory (Memory): The updated memory
        """
        block = self.get_agent_memory_block(agent_id, current_label)
        return self.update_block(block.id, label=new_label)

    # TODO: remove this
    def add_agent_memory_block(self, agent_id: str, create_block: CreateBlock) -> Memory:
        """
        Create and link a memory block to an agent's core memory

        Args:
            agent_id (str): The agent ID
            create_block (CreateBlock): The block to create

        Returns:
            memory (Memory): The updated memory
        """
        block_req = Block(**create_block.model_dump())
        block = self.server.block_manager.create_or_update_block(actor=self.server.user_manager.get_user_by_id(self.user.id), block=block_req)
        # Link the block to the agent
        agent = self.server.agent_manager.attach_block(agent_id=agent_id, block_id=block.id, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return agent.memory

    def link_agent_memory_block(self, agent_id: str, block_id: str) -> Memory:
        """
        Link a block to an agent's core memory

        Args:
            agent_id (str): The agent ID
            block_id (str): The block ID

        Returns:
            memory (Memory): The updated memory
        """
        return self.server.agent_manager.attach_block(agent_id=agent_id, block_id=block_id, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def remove_agent_memory_block(self, agent_id: str, block_label: str) -> Memory:
        """
        Unlike a block from the agent's core memory

        Args:
            agent_id (str): The agent ID
            block_label (str): The block label

        Returns:
            memory (Memory): The updated memory
        """
        return self.server.agent_manager.detach_block_with_label(agent_id=agent_id, block_label=block_label, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def list_agent_memory_blocks(self, agent_id: str) -> List[Block]:
        """
        Get all the blocks in the agent's core memory

        Args:
            agent_id (str): The agent ID

        Returns:
            blocks (List[Block]): The blocks in the agent's core memory
        """
        agent = self.server.agent_manager.get_agent_by_id(agent_id=agent_id, actor=self.server.user_manager.get_user_by_id(self.user.id))
        return agent.memory.blocks

    def get_agent_memory_block(self, agent_id: str, label: str) -> Block:
        """
        Get a block in the agent's core memory by its label

        Args:
            agent_id (str): The agent ID
            label (str): The label in the agent's core memory

        Returns:
            block (Block): The block corresponding to the label
        """
        return self.server.agent_manager.get_block_with_label(agent_id=agent_id, block_label=label, actor=self.server.user_manager.get_user_by_id(self.user.id))

    def update_agent_memory_block(
        self,
        agent_id: str,
        label: str,
        value: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Update a block in the agent's core memory by specifying its label

        Args:
            agent_id (str): The agent ID
            label (str): The label of the block
            value (str): The new value of the block
            limit (int): The new limit of the block

        Returns:
            block (Block): The updated block
        """
        block = self.get_agent_memory_block(agent_id, label)
        data = {}
        if value:
            data["value"] = value
        if limit:
            data["limit"] = limit
        return self.server.block_manager.update_block(block.id, actor=self.server.user_manager.get_user_by_id(self.user.id), block_update=BlockUpdate(**data))

    def update_block(
        self,
        block_id: str,
        label: Optional[str] = None,
        value: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Update a block given the ID with the provided fields

        Args:
            block_id (str): ID of the block
            label (str): Label to assign to the block
            value (str): Value to assign to the block
            limit (int): Token limit to assign to the block

        Returns:
            block (Block): Updated block
        """
        data = {}
        if value:
            data["value"] = value
        if limit:
            data["limit"] = limit
        if label:
            data["label"] = label
        return self.server.block_manager.update_block(block_id, actor=self.server.user_manager.get_user_by_id(self.user.id), block_update=BlockUpdate(**data))

    def get_tags(
        self,
        cursor: str = None,
        limit: int = 100,
        query_text: str = None,
    ) -> List[str]:
        """
        Get all tags.

        Returns:
            tags (List[str]): List of tags
        """
        return self.server.agent_manager.list_tags(actor=self.server.user_manager.get_user_by_id(self.user.id), cursor=cursor, limit=limit, query_text=query_text)
