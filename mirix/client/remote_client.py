"""
MirixClient implementation for Mirix.
This client communicates with a remote Mirix server via REST API.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Set, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mirix.client.client import AbstractClient
from mirix.constants import FUNCTION_RETURN_CHAR_LIMIT
from mirix.schemas.agent import AgentState, AgentType, CreateAgent, CreateMetaAgent
from mirix.schemas.block import Block, BlockUpdate, CreateBlock, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
)
from mirix.schemas.file import FileMetadata
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message, MessageCreate
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    SandboxConfigCreate,
    SandboxConfigUpdate,
)
from mirix.schemas.tool import Tool, ToolCreate, ToolUpdate
from mirix.schemas.tool_rule import BaseToolRule
from mirix.log import get_logger

logger = get_logger(__name__)


class MirixClient(AbstractClient):
    """
    Client that communicates with a remote Mirix server via REST API.
    
    This client runs on the user's local machine and makes HTTP requests
    to a Mirix server hosted in the cloud.
    
    Example:
        >>> client = MirixClient(
        ...     base_url="https://api.mirix.ai",
        ...     org_id="my-org",
        ... )
        >>> meta_agent = client.initialize_meta_agent(
        ...     user_id="my-user",
        ...     config={"llm_config": {...}, "embedding_config": {...}},
        ... )
        >>> response = client.add(
        ...     user_id="my-user",
        ...     messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        ... )
    """

    def __init__(
        self,
        org_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_name: Optional[str] = None,
        client_scope: str = "",     
        org_name: Optional[str] = None,
        debug: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize MirixClient.
        
        This client represents a CLIENT APPLICATION (tenant), not an end-user.
        End-user IDs are passed per-request in the add() method.

        Args:
            org_id: Organization ID (required)
            base_url: Base URL of the Mirix API server (optional, can also be set via MIRIX_API_URL env var, default: "http://localhost:8000")
            client_id: Client ID representing this application (optional, will be auto-generated if not provided)
            client_name: Client name (optional, defaults to client_id if not provided)
            client_scope: Client scope (read, write, read_write, admin), default: "read_write"
            org_id: Organization ID (optional, will be auto-generated if not provided)
            org_name: Organization name (optional, defaults to org_id if not provided)
            debug: Whether to enable debug logging
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
        """
        super().__init__(debug=debug)
        
        # Get base URL from parameter or environment variable
        self.base_url = (base_url or os.environ.get("MIRIX_API_URL", "http://localhost:8000")).rstrip("/")

        # Generate IDs if not provided
        if not client_id:
            import uuid
            client_id = f"client-{uuid.uuid4().hex[:8]}"
        
        if not org_id:
            import uuid
            org_id = f"org-{uuid.uuid4().hex[:8]}"
        
        self.client_id = client_id
        self.client_name = client_name or client_id
        self.client_scope = client_scope            
        self.org_id = org_id
        self.org_name = org_name or org_id
        self.timeout = timeout
        self._known_users: Set[str] = set()
        
        # Track initialized meta agent for this project
        self._meta_agent: Optional[AgentState] = None
        
        # Create session with retry logic
        self.session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        if self.client_id:
            self.session.headers.update({"X-Client-ID": self.client_id})
        
        if self.org_id:
            self.session.headers.update({"X-Org-ID": self.org_id})
                
        self.session.headers.update({"Content-Type": "application/json"})

        # Create organization and client if they don't exist
        self._ensure_org_and_client_exist()

    def _ensure_org_and_client_exist(self):
        """
        Ensure that the organization and client exist on the server.
        Creates them if they don't exist.
        
        Note: This method does NOT create users. Users are created per-request
        based on the user_id parameter in add() and other methods.
        """
        try:
            # Create or get organization first
            org_response = self._request(
                "POST",
                "/organizations/create_or_get",
                json={"org_id": self.org_id, "name": self.org_name}
            )
            if self.debug:
                logger.debug("[MirixClient] Organization initialized: %s (name: %s)", self.org_id, self.org_name)
            
            # Create or get client
            client_response = self._request(
                "POST",
                "/clients/create_or_get",
                json={
                    "client_id": self.client_id,
                    "name": self.client_name,
                    "org_id": self.org_id,
                    "scope": self.client_scope,
                    "status": "active"
                }
            )
            if self.debug:
                logger.debug("[MirixClient] Client initialized: %s (name: %s, scope: %s)", 
                           self.client_id, self.client_name, self.client_scope)
        except Exception as e:
            # Don't fail initialization if this fails - the server might handle it
            if self.debug:
                logger.debug("[MirixClient] Note: Could not pre-create org/client: %s", e)
                logger.debug("[MirixClient] Server will create them on first request if needed")

    def create_or_get_user(
        self,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> str:
        """
        Create a user if it doesn't exist, or get existing user.
        
        This method ensures a user exists in the backend database before performing
        operations that require a user_id. If the user already exists, it returns
        the existing user_id. If not, it creates a new user.
        
        Args:
            user_id: Optional user ID. If not provided, a random ID will be generated.
            user_name: Optional user name. Defaults to user_id if not provided.
            org_id: Optional organization ID. Defaults to client's org_id if not provided.
            
        Returns:
            str: The user_id (either existing or newly created)
            
        Example:
            >>> client = MirixClient(client_id="my-app", org_id="my-org")
            >>> 
            >>> # Create user with specific ID
            >>> user_id = client.create_or_get_user(
            ...     user_id="demo-user",
            ...     user_name="Demo User"
            ... )
            >>> print(f"User ready: {user_id}")
            >>> 
            >>> # Create user with auto-generated ID
            >>> user_id = client.create_or_get_user(user_name="Alice")
            >>> print(f"User created with ID: {user_id}")
            >>> 
            >>> # Now use the user_id for memory operations
            >>> result = client.add(
            ...     user_id=user_id,
            ...     messages=[...]
            ... )
        """
        # Use client's org_id if not specified
        if not org_id:
            org_id = self.org_id
        
        # Prepare request data
        request_data = {
            "user_id": user_id,
            "name": user_name,
            "org_id": org_id,
        }
        
        # Make API request
        response = self._request("POST", "/users/create_or_get", json=request_data)
        
        # Extract and return user_id from response
        if isinstance(response, dict) and "id" in response:
            created_user_id = response["id"]
            if self.debug:
                logger.debug("User ready: %s", created_user_id)
            return created_user_id
        else:
            raise ValueError(f"Unexpected response from /users/create_or_get: {response}")

    def _ensure_user_exists(self, user_id: str):
        """Ensure that the given user exists for this org."""
        if not user_id:
            raise ValueError("user_id is required")
        if user_id in self._known_users:
            return
        try:
            self._request(
                "POST",
                "/users/create_or_get",
                json={
                    "user_id": user_id,
                    "name": user_id,
                    "org_id": self.org_id,
                },
            )
            self._known_users.add(user_id)
            if self.debug:
                logger.debug("[MirixClient] User ensured: %s (org: %s)", user_id, self.org_id)
        except Exception as e:
            if self.debug:
                logger.debug("[MirixClient] Note: Could not ensure user %s: %s", user_id, e)

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Any:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/agents")
            json: JSON body for the request
            params: Query parameters
            
        Returns:
            Response data (parsed JSON)
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        if self.debug:
            logger.debug("[MirixClient] %s %s", method, url)
            if json:
                logger.debug("[MirixClient] Request body: %s", json)
        
        response = self.session.request(
            method=method,
            url=url,
            json=json,
            params=params,
            timeout=self.timeout,
        )
        
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # Try to extract error message from response
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            raise requests.HTTPError(f"API request failed: {error_detail}") from e
        
        # Return parsed JSON if there's content
        if response.content:
            return response.json()
        return None

    # ========================================================================
    # Agent Methods
    # ========================================================================

    def list_agents(
        self,
        query_text: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> List[AgentState]:
        """List all agents."""
        params = {"limit": limit}
        if query_text:
            params["query_text"] = query_text
        if tags:
            params["tags"] = ",".join(tags)
        if cursor:
            params["cursor"] = cursor
        if parent_id:
            params["parent_id"] = parent_id
        
        data = self._request("GET", "/agents", params=params)
        return [AgentState(**agent) for agent in data]

    def agent_exists(
        self, agent_id: Optional[str] = None, agent_name: Optional[str] = None
    ) -> bool:
        """Check if an agent exists."""
        if not (agent_id or agent_name):
            raise ValueError("Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError("Only one of agent_id or agent_name can be provided")
        
        existing = self.list_agents()
        if agent_id:
            return str(agent_id) in [str(agent.id) for agent in existing]
        else:
            return agent_name in [str(agent.name) for agent in existing]

    def create_agent(
        self,
        name: Optional[str] = None,
        agent_type: Optional[AgentType] = AgentType.chat_agent,
        embedding_config: Optional[EmbeddingConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        memory: Optional[Memory] = None,
        block_ids: Optional[List[str]] = None,
        system: Optional[str] = None,
        tool_ids: Optional[List[str]] = None,
        tool_rules: Optional[List[BaseToolRule]] = None,
        include_base_tools: Optional[bool] = True,
        include_meta_memory_tools: Optional[bool] = False,
        metadata: Optional[Dict] = None,
        description: Optional[str] = None,
        initial_message_sequence: Optional[List[Message]] = None,
        tags: Optional[List[str]] = None,
    ) -> AgentState:
        """Create an agent."""
        request_data = {
            "name": name,
            "agent_type": agent_type,
            "embedding_config": embedding_config.model_dump() if embedding_config else None,
            "llm_config": llm_config.model_dump() if llm_config else None,
            "memory": memory.model_dump() if memory else None,
            "block_ids": block_ids,
            "system": system,
            "tool_ids": tool_ids,
            "tool_rules": [rule.model_dump() if hasattr(rule, 'model_dump') else rule for rule in (tool_rules or [])],
            "include_base_tools": include_base_tools,
            "include_meta_memory_tools": include_meta_memory_tools,
            "metadata": metadata,
            "description": description,
            "initial_message_sequence": [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in (initial_message_sequence or [])],
            "tags": tags,
        }
        
        data = self._request("POST", "/agents", json=request_data)
        return AgentState(**data)

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
        """Update an agent."""
        request_data = {
            "name": name,
            "description": description,
            "system": system,
            "tool_ids": tool_ids,
            "metadata": metadata,
            "llm_config": llm_config.model_dump() if llm_config else None,
            "embedding_config": embedding_config.model_dump() if embedding_config else None,
            "message_ids": message_ids,
            "memory": memory.model_dump() if memory else None,
            "tags": tags,
        }
        
        data = self._request("PATCH", f"/agents/{agent_id}", json=request_data)
        return AgentState(**data)

    def get_agent(self, agent_id: str) -> AgentState:
        """Get an agent by ID."""
        data = self._request("GET", f"/agents/{agent_id}")
        return AgentState(**data)

    def get_agent_id(self, agent_name: str) -> Optional[str]:
        """Get agent ID by name."""
        agents = self.list_agents()
        for agent in agents:
            if agent.name == agent_name:
                return agent.id
        return None

    def delete_agent(self, agent_id: str):
        """Delete an agent."""
        self._request("DELETE", f"/agents/{agent_id}")

    def rename_agent(self, agent_id: str, new_name: str):
        """Rename an agent."""
        self.update_agent(agent_id, name=new_name)

    def get_tools_from_agent(self, agent_id: str) -> List[Tool]:
        """Get tools from an agent."""
        agent = self.get_agent(agent_id)
        return agent.tools

    def add_tool_to_agent(self, agent_id: str, tool_id: str):
        """Add a tool to an agent."""
        raise NotImplementedError("add_tool_to_agent not yet implemented in REST API")

    def remove_tool_from_agent(self, agent_id: str, tool_id: str):
        """Remove a tool from an agent."""
        raise NotImplementedError("remove_tool_from_agent not yet implemented in REST API")

    # ========================================================================
    # Memory Methods
    # ========================================================================

    def get_in_context_memory(self, agent_id: str) -> Memory:
        """Get in-context memory of an agent."""
        data = self._request("GET", f"/agents/{agent_id}/memory")
        return Memory(**data)

    def update_in_context_memory(
        self, agent_id: str, section: str, value: Union[List[str], str]
    ) -> Memory:
        """Update in-context memory."""
        raise NotImplementedError("update_in_context_memory not yet implemented in REST API")

    def get_archival_memory_summary(self, agent_id: str) -> ArchivalMemorySummary:
        """Get archival memory summary."""
        data = self._request("GET", f"/agents/{agent_id}/memory/archival")
        return ArchivalMemorySummary(**data)

    def get_recall_memory_summary(self, agent_id: str) -> RecallMemorySummary:
        """Get recall memory summary."""
        data = self._request("GET", f"/agents/{agent_id}/memory/recall")
        return RecallMemorySummary(**data)

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get in-context messages."""
        raise NotImplementedError("get_in_context_messages not yet implemented in REST API")

    # ========================================================================
    # Message Methods
    # ========================================================================

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        stream: Optional[bool] = False,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> MirixResponse:
        """Send a message to an agent.
        
        Args:
            message: The message text to send
            role: The role of the message sender (user/system)
            agent_id: The ID of the agent to send the message to
            name: Optional name of the message sender
            stream: Enable streaming (not yet implemented)
            stream_steps: Stream intermediate steps
            stream_tokens: Stream tokens as they are generated
            filter_tags: Optional filter tags for categorization and filtering.
                Example: {"project_id": "proj-alpha", "session_id": "sess-123"}
            use_cache: Control Redis cache behavior (default: True)
        
        Returns:
            MirixResponse: The response from the agent
        
        Example:
            >>> response = client.send_message(
            ...     message="What's the status?",
            ...     role="user",
            ...     agent_id="agent123",
            ...     filter_tags={"project": "alpha", "priority": "high"}
            ... )
        """
        if stream or stream_steps or stream_tokens:
            raise NotImplementedError("Streaming not yet implemented in REST API")
        
        request_data = {
            "message": message,
            "role": role,
            "name": name,
            "stream_steps": stream_steps,
            "stream_tokens": stream_tokens,
        }
        
        # Include filter_tags if provided
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        # Include use_cache if not default
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        data = self._request("POST", f"/agents/{agent_id}/messages", json=request_data)
        return MirixResponse(**data)

    def user_message(self, agent_id: str, message: str) -> MirixResponse:
        """Send a user message to an agent."""
        return self.send_message(message=message, role="user", agent_id=agent_id)

    def get_messages(
        self,
        agent_id: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = 1000,
        use_cache: bool = True,
    ) -> List[Message]:
        """Get messages from an agent.
        
        Args:
            agent_id: The ID of the agent
            before: Get messages before this cursor
            after: Get messages after this cursor
            limit: Maximum number of messages to retrieve
            use_cache: Control Redis cache behavior (default: True)
        
        Returns:
            List of messages
        """
        params = {"limit": limit}
        if before:
            params["cursor"] = before
        if not use_cache:
            params["use_cache"] = "false"
        
        data = self._request("GET", f"/agents/{agent_id}/messages", params=params)
        return [Message(**msg) for msg in data]

    # ========================================================================
    # Tool Methods
    # ========================================================================

    def list_tools(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[Tool]:
        """List all tools."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        
        data = self._request("GET", "/tools", params=params)
        return [Tool(**tool) for tool in data]

    def get_tool(self, id: str) -> Tool:
        """Get a tool by ID."""
        data = self._request("GET", f"/tools/{id}")
        return Tool(**data)

    def create_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """Create a tool."""
        raise NotImplementedError(
            "create_tool with function not supported in MirixClient. "
            "Tools must be created on the server side."
        )

    def create_or_update_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """Create or update a tool."""
        raise NotImplementedError(
            "create_or_update_tool with function not supported in MirixClient. "
            "Tools must be created on the server side."
        )

    def update_tool(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        func: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """Update a tool."""
        raise NotImplementedError("update_tool not yet implemented in REST API")

    def delete_tool(self, id: str):
        """Delete a tool."""
        self._request("DELETE", f"/tools/{id}")

    def get_tool_id(self, name: str) -> Optional[str]:
        """Get tool ID by name."""
        tools = self.list_tools()
        for tool in tools:
            if tool.name == name:
                return tool.id
        return None

    def upsert_base_tools(self) -> List[Tool]:
        """Upsert base tools."""
        raise NotImplementedError("upsert_base_tools must be done on server side")

    # ========================================================================
    # Block Methods
    # ========================================================================

    def list_blocks(
        self, label: Optional[str] = None, templates_only: Optional[bool] = True
    ) -> List[Block]:
        """List blocks."""
        params = {}
        if label:
            params["label"] = label
        
        data = self._request("GET", "/blocks", params=params)
        return [Block(**block) for block in data]

    def get_block(self, block_id: str) -> Block:
        """Get a block by ID."""
        data = self._request("GET", f"/blocks/{block_id}")
        return Block(**data)

    def create_block(
        self,
        label: str,
        value: str,
        limit: Optional[int] = None,
    ) -> Block:
        """Create a block."""
        block_data = {
            "label": label,
            "value": value,
            "limit": limit,
        }
        
        block = Block(**block_data)
        data = self._request("POST", "/blocks", json=block.model_dump())
        return Block(**data)

    def delete_block(self, id: str) -> Block:
        """Delete a block."""
        self._request("DELETE", f"/blocks/{id}")

    # ========================================================================
    # Human/Persona Methods
    # ========================================================================

    def create_human(self, name: str, text: str) -> Human:
        """Create a human block."""
        human = Human(value=text)
        data = self._request("POST", "/blocks", json=human.model_dump())
        return Human(**data)

    def create_persona(self, name: str, text: str) -> Persona:
        """Create a persona block."""
        persona = Persona(value=text)
        data = self._request("POST", "/blocks", json=persona.model_dump())
        return Persona(**data)

    def list_humans(self) -> List[Human]:
        """List human blocks."""
        blocks = self.list_blocks(label="human")
        return [Human(**block.model_dump()) for block in blocks]

    def list_personas(self) -> List[Persona]:
        """List persona blocks."""
        blocks = self.list_blocks(label="persona")
        return [Persona(**block.model_dump()) for block in blocks]

    def update_human(self, human_id: str, text: str) -> Human:
        """Update a human block."""
        raise NotImplementedError("update_human not yet implemented in REST API")

    def update_persona(self, persona_id: str, text: str) -> Persona:
        """Update a persona block."""
        raise NotImplementedError("update_persona not yet implemented in REST API")

    def get_persona(self, id: str) -> Persona:
        """Get a persona block."""
        data = self._request("GET", f"/blocks/{id}")
        return Persona(**data)

    def get_human(self, id: str) -> Human:
        """Get a human block."""
        data = self._request("GET", f"/blocks/{id}")
        return Human(**data)

    def get_persona_id(self, name: str) -> str:
        """Get persona ID by name."""
        personas = self.list_personas()
        if personas:
            return personas[0].id
        return None

    def get_human_id(self, name: str) -> str:
        """Get human ID by name."""
        humans = self.list_humans()
        if humans:
            return humans[0].id
        return None

    def delete_persona(self, id: str):
        """Delete a persona."""
        self.delete_block(id)

    def delete_human(self, id: str):
        """Delete a human."""
        self.delete_block(id)

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def list_model_configs(self) -> List[LLMConfig]:
        """List available LLM configurations."""
        data = self._request("GET", "/config/llm")
        return [LLMConfig(**config) for config in data]

    def list_embedding_configs(self) -> List[EmbeddingConfig]:
        """List available embedding configurations."""
        data = self._request("GET", "/config/embedding")
        return [EmbeddingConfig(**config) for config in data]

    # ========================================================================
    # Organization Methods
    # ========================================================================

    def create_org(self, name: Optional[str] = None) -> Organization:
        """Create an organization."""
        data = self._request("POST", "/organizations", json={"name": name})
        return Organization(**data)

    def list_orgs(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> List[Organization]:
        """List organizations."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        
        data = self._request("GET", "/organizations", params=params)
        return [Organization(**org) for org in data]

    def delete_org(self, org_id: str) -> Organization:
        """Delete an organization."""
        raise NotImplementedError("delete_org not yet implemented in REST API")

    # ========================================================================
    # Sandbox Methods (Not Implemented)
    # ========================================================================

    def create_sandbox_config(
        self, config: Union[LocalSandboxConfig, E2BSandboxConfig]
    ) -> SandboxConfig:
        """Create sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    def update_sandbox_config(
        self,
        sandbox_config_id: str,
        config: Union[LocalSandboxConfig, E2BSandboxConfig],
    ) -> SandboxConfig:
        """Update sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    def delete_sandbox_config(self, sandbox_config_id: str) -> None:
        """Delete sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    def list_sandbox_configs(
        self, limit: int = 50, cursor: Optional[str] = None
    ) -> List[SandboxConfig]:
        """List sandbox configs."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    def create_sandbox_env_var(
        self,
        sandbox_config_id: str,
        key: str,
        value: str,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """Create sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    def update_sandbox_env_var(
        self,
        env_var_id: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """Update sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    def delete_sandbox_env_var(self, env_var_id: str) -> None:
        """Delete sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    def list_sandbox_env_vars(
        self, sandbox_config_id: str, limit: int = 50, cursor: Optional[str] = None
    ) -> List[SandboxEnvironmentVariable]:
        """List sandbox environment variables."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    # ========================================================================
    # New Memory API Methods
    # ========================================================================

    def _load_system_prompts(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Load all system prompts from the system_prompts_folder.
        
        Args:
            config: Configuration dictionary that may contain 'system_prompts_folder'
            
        Returns:
            Dict mapping agent names to their prompt text
        """
        import os
        import logging
        
        logger = logging.getLogger(__name__)
        prompts = {}
        
        system_prompts_folder = config.get("system_prompts_folder")
        if not system_prompts_folder:
            return prompts
        
        if not os.path.exists(system_prompts_folder):
            return prompts
        
        # Load all .txt files from the system prompts folder
        for filename in os.listdir(system_prompts_folder):
            if filename.endswith(".txt"):
                agent_name = filename[:-4]  # Strip .txt suffix
                prompt_file = os.path.join(system_prompts_folder, filename)
                
                try:
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        prompts[agent_name] = f.read()
                except Exception as e:
                    # Log warning but continue
                    logger.warning(
                        f"Failed to load system prompt for {agent_name} from {prompt_file}: {e}"
                    )

        return prompts
    
    def initialize_meta_agent(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        update_agents: Optional[bool] = False,
    ) -> AgentState:
        """
        Initialize a meta agent with the given configuration.
        
        This creates a meta memory agent that manages multiple specialized memory agents
        (episodic, semantic, procedural, etc.) for the current project.
        
        Args:
            config: Configuration dictionary with llm_config, embedding_config, etc.
            config_path: Path to YAML config file (alternative to config dict)
            
        Returns:
            AgentState: The initialized meta agent
            
        Example:
            >>> client = MirixClient(org_id="org-123")
            >>> config = {
            ...     "llm_config": {"model": "gemini-2.0-flash"},
            ...     "embedding_config": {"model": "text-embedding-004"}
            ... }
            >>> meta_agent = client.initialize_meta_agent(config=config)
        """
        #self._ensure_user_exists(user_id)
        
        # Load config from file if provided
        if config_path:
            import yaml
            from pathlib import Path
            
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
        
        if not config:
            raise ValueError("Either config or config_path must be provided")
        
        # Load system prompts from folder if specified and not already provided
        if config.get("meta_agent_config") and config['meta_agent_config'].get("system_prompts_folder") and not config.get("system_prompts"):
            config['meta_agent_config']["system_prompts"] = self._load_system_prompts(config['meta_agent_config'])
            del config['meta_agent_config']['system_prompts_folder']

        # Prepare request data
        request_data = {
            #"user_id": user_id,
            "org_id": self.org_id,
            "config": config,
            "update_agents": update_agents,
        }
        
        # Make API request to initialize meta agent
        data = self._request("POST", "/agents/meta/initialize", json=request_data)
        self._meta_agent = AgentState(**data)
        return self._meta_agent
    
    def add(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        chaining: bool = True,
        verbose: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Add conversation turns to memory (asynchronous processing).
        
        This method queues conversation turns for background processing by queue workers.
        The messages are stored in the appropriate memory systems asynchronously.
        
        Args:
            user_id: User ID for the conversation
            messages: List of message dicts with role and content.
                     Messages should end with an assistant turn.
                     Format: [
                         {"role": "user", "content": [{"type": "text", "text": "..."}]},
                         {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
                     ]
            verbose: If True, enable verbose output during memory processing
            filter_tags: Optional dict of tags for filtering and categorization.
                        Example: {"project_id": "proj-123", "session_id": "sess-456"}
            use_cache: Control Redis cache behavior (default: True)
        
        Returns:
            Dict containing:
                - success (bool): True if message was queued successfully
                - message (str): Status message
                - status (str): "queued" - indicates async processing
                - agent_id (str): Meta agent ID processing the messages
                - message_count (int): Number of messages queued
            
        Note:
            Processing happens asynchronously. The response indicates the message
            was successfully queued, not that processing is complete.
            
        Example:
            >>> response = client.add(
            ...     user_id='user_123',
            ...     messages=[
            ...         {"role": "user", "content": [{"type": "text", "text": "I went to dinner"}]},
            ...         {"role": "assistant", "content": [{"type": "text", "text": "That's great!"}]}
            ...     ],
            ...     verbose=True,
            ...     filter_tags={"session_id": "sess-789", "tags": ["personal"]}
            ... )
            >>> logger.debug(response)
            {
                "success": True,
                "message": "Memory queued for processing",
                "status": "queued",
                "agent_id": "agent-456",
                "message_count": 2
            }
        """
        if not self._meta_agent:
            raise ValueError("Meta agent not initialized. Call initialize_meta_agent() first.")
        
        self._ensure_user_exists(user_id)
        
        request_data = {
            "user_id": user_id,
            "org_id": self.org_id,
            "meta_agent_id": self._meta_agent.id,
            "messages": messages,
            "chaining": chaining,
            "verbose": verbose,
        }
        
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        return self._request("POST", "/memory/add", json=request_data)
    
    def retrieve_with_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        local_model_for_retrieval: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories based on conversation context.
        
        This method analyzes the conversation and retrieves relevant memories
        from all memory systems.
        
        Args:
            user_id: User ID for the conversation
            messages: List of message dicts with role and content.
                     Messages should end with a user turn.
                     Format: [
                         {"role": "user", "content": [{"type": "text", "text": "..."}]}
                     ]
            limit: Maximum number of items to retrieve per memory type (default: 10)
            filter_tags: Optional dict of tags for filtering results.
                        Only memories matching these tags will be returned.
            use_cache: Control Redis cache behavior (default: True)
        
        Returns:
            Dict containing retrieved memories organized by type
            
        Example:
            >>> memories = client.retrieve_with_conversation(
            ...     user_id='user_123',
            ...     messages=[
            ...         {"role": "user", "content": [{"type": "text", "text": "Where did I go yesterday?"}]}
            ...     ],
            ...     limit=5,
            ...     filter_tags={"session_id": "sess-789"}
            ... )
        """
        if not self._meta_agent:
            raise ValueError("Meta agent not initialized. Call initialize_meta_agent() first.")
        
        self._ensure_user_exists(user_id)
        
        request_data = {
            "user_id": user_id,
            "org_id": self.org_id,
            "messages": messages,
            "limit": limit,
            "local_model_for_retrieval": local_model_for_retrieval
        }
        
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        return self._request("POST", "/memory/retrieve/conversation", json=request_data)
    
    def retrieve_with_topic(
        self,
        user_id: str,
        topic: str,
        limit: int = 10,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories based on a topic.
        
        This method searches for memories related to a specific topic or keyword.
        
        Args:
            user_id: User ID for the conversation
            topic: Topic or keyword to search for
            limit: Maximum number of items to retrieve per memory type (default: 10)
            #org_id: Optional organization scope override (defaults to client's org)
        
        Returns:
            Dict containing retrieved memories organized by type
            
        Example:
            >>> memories = client.retrieve_with_topic(
            ...     user_id='user_123',
            ...     topic="dinner",
            ...     limit=5
            ... )
        """
        if not self._meta_agent:
            raise ValueError("Meta agent not initialized. Call initialize_meta_agent() first.")
        
        self._ensure_user_exists(user_id)
        
        params = {
            "user_id": user_id,
            "org_id": org_id or self.org_id,
            "topic": topic,
            "limit": limit,
        }
        
        return self._request("GET", "/memory/retrieve/topic", params=params)
    
    def search(
        self,
        user_id: str,
        query: str,
        memory_type: str = "all",
        search_field: str = "null",
        search_method: str = "bm25",
        limit: int = 10,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories using various search methods.
        Similar to the search_in_memory tool function.
        
        This method performs a search across specified memory types and returns
        a flat list of results.
        
        Args:
            user_id: User ID for the conversation
            query: Search query
            memory_type: Type of memory to search. Options: "episodic", "resource", 
                        "procedural", "knowledge_vault", "semantic", "all" (default: "all")
            search_field: Field to search in. Options vary by memory type:
                         - episodic: "summary", "details"
                         - resource: "summary", "content"
                         - procedural: "summary", "steps"
                         - knowledge_vault: "caption", "secret_value"
                         - semantic: "name", "summary", "details"
                         - For "all": use "null" (default)
            search_method: Search method. Options: "bm25" (default), "embedding"
            limit: Maximum number of results per memory type (default: 10)
            org_id: Optional organization scope override (defaults to client's org)
        
        Returns:
            Dict containing:
                - success: bool
                - query: str (the search query)
                - memory_type: str (the memory type searched)
                - search_field: str (the field searched)
                - search_method: str (the search method used)
                - results: List[Dict] (flat list of results from all memory types)
                - count: int (total number of results)
            
        Example:
            >>> # Search all memory types
            >>> results = client.search(
            ...     user_id='user_123',
            ...     query="restaurants",
            ...     limit=5
            ... )
            logger.debug("Found %s results", results['count'])
            >>> 
            >>> # Search only episodic memories in details field
            >>> episodic_results = client.search(
            ...     user_id='user_123',
            ...     query="meeting",
            ...     memory_type="episodic",
            ...     search_field="details",
            ...     limit=10
            ... )
        """
        if not self._meta_agent:
            raise ValueError("Meta agent not initialized. Call initialize_meta_agent() first.")
        
        self._ensure_user_exists(user_id)
        
        params = {
            "user_id": user_id,
            "org_id": org_id or self.org_id,
            "query": query,
            "memory_type": memory_type,
            "search_field": search_field,
            "search_method": search_method,
            "limit": limit,
        }
        
        return self._request("GET", "/memory/search", params=params)

    # ========================================================================
    # LangChain/Composio/CrewAI Integration (Not Supported)
    # ========================================================================
