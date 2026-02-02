"""
MirixTerminalClient implementation for Mirix.
This client communicates with a remote Mirix server via REST API.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mirix.constants import FUNCTION_RETURN_CHAR_LIMIT
from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.block import Block, Human, Persona
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.environment_variables import (
    SandboxEnvironmentVariable,
)
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.memory import ArchivalMemorySummary, Memory, RecallMemorySummary
from mirix.schemas.message import Message
from mirix.schemas.mirix_response import MirixResponse
from mirix.schemas.organization import Organization
from mirix.schemas.sandbox_config import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
)
from mirix.schemas.tool import Tool
from mirix.schemas.tool_rule import BaseToolRule

logger = get_logger(__name__)


# Default configurations for different LLM providers
PROVIDER_DEFAULTS = {
    "openai": {
        "llm_config": {
            "model": "gpt-4o-mini",
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "context_window": 128000,
        },
        "topic_extraction_llm_config": {
            "model": "gpt-4.1-nano",
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "context_window": 128000,
        },
        "build_embeddings_for_memory": True,
        "embedding_config": {
            "embedding_model": "text-embedding-3-small",
            "embedding_endpoint": "https://api.openai.com/v1",
            "embedding_endpoint_type": "openai",
            "embedding_dim": 1536,
        },
    },
    "anthropic": {
        "llm_config": {
            "model": "claude-sonnet-4-20250514",
            "model_endpoint_type": "anthropic",
            "model_endpoint": "https://api.anthropic.com/v1",
            "context_window": 200000,
        },
        "topic_extraction_llm_config": {
            "model": "claude-sonnet-4-20250514",
            "model_endpoint_type": "anthropic",
            "model_endpoint": "https://api.anthropic.com/v1",
            "context_window": 200000,
        },
        # Anthropic doesn't provide embeddings, use without embeddings
        "build_embeddings_for_memory": False,
    },
    "google_ai": {
        "llm_config": {
            "model": "gemini-2.0-flash",
            "model_endpoint_type": "google_ai",
            "model_endpoint": "https://generativelanguage.googleapis.com",
            "context_window": 1000000,
        },
        "topic_extraction_llm_config": {
            "model": "gemini-2.0-flash-lite",
            "model_endpoint_type": "google_ai",
            "model_endpoint": "https://generativelanguage.googleapis.com",
            "context_window": 1000000,
        },
        "build_embeddings_for_memory": True,
        "embedding_config": {
            "embedding_model": "text-embedding-004",
            "embedding_endpoint_type": "google_ai",
            "embedding_endpoint": "https://generativelanguage.googleapis.com",
            "embedding_dim": 768,
        },
    },
}

# Default meta agent configuration shared across all providers
DEFAULT_META_AGENT_CONFIG = {
    "system_prompts_folder": "mirix/prompts/system/base",
    "agents": [
        "core_memory_agent",
        "resource_memory_agent",
        "semantic_memory_agent",
        "episodic_memory_agent",
        "procedural_memory_agent",
        "knowledge_memory_agent",
        "reflexion_agent",
        "background_agent",
    ],
    "memory": {
        "core": [
            {"label": "human", "value": ""},
            {"label": "persona", "value": "I am a helpful assistant."},
        ],
        "decay": {
            "fade_after_days": 30,
            "expire_after_days": 90,
        },
    },
}


def _get_provider_config(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a configuration dictionary for a specific LLM provider.
    
    Args:
        provider: The LLM provider name ("openai", "anthropic", "google_ai")
        api_key: Optional API key for the provider. If not provided, the config
                 won't include api_key (user can set via environment variables).
        model: Optional model name to override the default
    
    Returns:
        A complete configuration dictionary ready for initialize_meta_agent
    """
    import copy
    
    provider = provider.lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {list(PROVIDER_DEFAULTS.keys())}"
        )
    
    # Deep copy to avoid modifying the defaults
    config = copy.deepcopy(PROVIDER_DEFAULTS[provider])
    
    # Set API key for LLM configs (only if provided)
    if api_key:
        config["llm_config"]["api_key"] = api_key
        config["topic_extraction_llm_config"]["api_key"] = api_key
    
    # Override model if specified
    if model:
        config["llm_config"]["model"] = model
        config["topic_extraction_llm_config"]["model"] = model
    
    # Set embedding API key (only for providers that use embeddings)
    if api_key and config.get("build_embeddings_for_memory") and "embedding_config" in config:
        config["embedding_config"]["api_key"] = api_key
    
    # Add meta agent config
    config["meta_agent_config"] = copy.deepcopy(DEFAULT_META_AGENT_CONFIG)
    
    return config


def _validate_occurred_at(occurred_at: Optional[str]) -> Optional[datetime]:
    """
    Validate occurred_at format and convert to datetime.

    Args:
        occurred_at: ISO 8601 datetime string (e.g., "2025-11-18T10:30:00" or "2025-11-18T10:30:00+00:00")

    Returns:
        datetime object if valid, None if input is None

    Raises:
        ValueError: If format is invalid
    """
    if occurred_at is None:
        return None

    if not isinstance(occurred_at, str):
        raise ValueError(
            f"occurred_at must be a string in ISO 8601 format, got {type(occurred_at).__name__}"
        )

    # Validate ISO 8601 format
    iso_pattern = (
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
    )
    if not re.match(iso_pattern, occurred_at):
        raise ValueError(
            f"occurred_at must be in ISO 8601 format (e.g., '2025-11-18T10:30:00' or '2025-11-18T10:30:00+00:00'), got: {occurred_at}"
        )

    try:
        # Parse and validate the datetime
        dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        return dt
    except ValueError as e:
        raise ValueError(
            f"Invalid occurred_at datetime: {occurred_at}. Error: {str(e)}"
        )


class MirixTerminalClient():
    """
    Client that communicates with a terminal-mode Mirix server via REST API.
    
    This client runs on the user's local machine and makes HTTP requests
    to a Mirix server hosted in the cloud.
    
    The client_id identifies both the client and organization, so no explicit
    org_id is needed.
    
    Example:
        >>> client = MirixTerminalClient(client_id="your-client-id")
        >>> # or bootstrap by name:
        >>> client = MirixTerminalClient(client_name="my-terminal-client")
        >>> meta_agent = client.initialize_meta_agent(
        ...     config={"llm_config": {...}, "embedding_config": {...}},
        ... )
        >>> response = client.add(
        ...     user_id="my-user",
        ...     messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        ... )
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_name: Optional[str] = None,
        debug: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize MirixTerminalClient.
        
        This client represents a CLIENT APPLICATION (tenant), not an end-user.
        End-user IDs are passed per-request in the add() method.
        
        The client_id identifies the client and organization, so no explicit org_id is needed.

        Args:
            client_id: Client ID for authentication (optional; can also be set via MIRIX_CLIENT_ID env var)
            client_name: Client name for bootstrap creation if client_id is not provided
            debug: Whether to enable debug logging
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
            headers: Optional headers to include in the initialization requests
        """

        self.debug = debug

        # Get base URL from environment variable
        self.base_url = base_url or (
            os.environ.get("MIRIX_API_URL", "http://localhost:8531")
        ).rstrip("/")

        self.timeout = timeout
        self._known_users: Set[str] = set()

        # Create session with retry logic
        self.session = requests.Session()

        # Track initialized meta agent for this project
        self._meta_agent: Optional[AgentState] = None
        
        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=[
                "HEAD",
                "GET",
                "PUT",
                "DELETE",
                "OPTIONS",
                "TRACE",
                "POST",
            ],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({"Content-Type": "application/json"})
        if headers:
            self.session.headers.update(headers)

        env_client_id = os.environ.get("MIRIX_CLIENT_ID")
        if client_id:
            requested_client_id = client_id
        elif client_name:
            requested_client_id = None
        else:
            requested_client_id = env_client_id

        if not requested_client_id and not client_name:
            raise ValueError(
                "client_id or client_name is required; set MIRIX_CLIENT_ID or pass client_id/client_name to MirixTerminalClient."
            )

        self.client_id = self._bootstrap_client(
            client_id=requested_client_id,
            client_name=client_name,
        )

        # Set headers - client_id identifies client and org
        self.session.headers.update({"X-Client-Id": self.client_id})

    def _bootstrap_client(
        self,
        client_id: Optional[str],
        client_name: Optional[str],
    ) -> str:
        """
        Create or load a client via the terminal bootstrap endpoint.
        """
        payload = {
            "client_id": client_id,
            "client_name": client_name,
        }

        url = f"{self.base_url}/terminal/clients"
        if self.debug:
            logger.debug("[MirixTerminalClient] POST %s", url)
            logger.debug("[MirixTerminalClient] Bootstrap payload: %s", payload)

        response = self.session.post(
            url=url,
            json=payload,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise ValueError(
                f"Terminal client bootstrap failed: {error_detail}"
            ) from e

        data = response.json() if response.content else {}
        client_id_value = data.get("client_id")
        if not client_id_value:
            raise ValueError(
                f"Terminal client bootstrap returned no client_id: {data}"
            )

        if self.debug:
            logger.debug("[MirixTerminalClient] client_id: %s", client_id_value)

        return client_id_value

    def create_or_get_user(
        self,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a user if it doesn't exist, or get existing user.
        
        This method ensures a user exists in the backend database before performing
        operations that require a user_id. If the user already exists, it returns
        the existing user_id. If not, it creates a new user.
        
        The organization is automatically determined from the client_id.
        
        Args:
            user_id: Optional user ID. If not provided, a random ID will be generated.
            user_name: Optional user name. Defaults to user_id if not provided.
            
        Returns:
            str: The user_id (either existing or newly created)
            
        Example:
            >>> client = MirixTerminalClient(client_id="your-client-id")
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
        # Prepare request data - org is determined from client_id on server side
        request_data = {
            "user_id": user_id,
            "name": user_name,
        }

        # Make API request
        response = self._request(
            "POST", "/users/create_or_get", json=request_data, headers=headers
        )
        
        # Extract and return user_id from response
        if isinstance(response, dict) and "id" in response:
            created_user_id = response["id"]
            if self.debug:
                logger.debug("User ready: %s", created_user_id)
            return created_user_id
        else:
            raise ValueError(
                f"Unexpected response from /users/create_or_get: {response}"
            )

    def _ensure_user_exists(
        self, user_id: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ):
        """Ensure that the given user exists for the client's organization.

        Args:
            user_id: User ID to ensure exists
            headers: Optional headers to include in the request
        """
        if not user_id:
            return
        if user_id in self._known_users:
            return
        try:
            self._request(
                "POST",
                "/users/create_or_get",
                json={
                    "user_id": user_id,
                    "name": user_id,
                },
                headers=headers,
            )
            self._known_users.add(user_id)
            if self.debug:
                logger.debug("[MirixTerminalClient] User ensured: %s", user_id)
        except Exception as e:
            if self.debug:
                logger.debug(
                    "[MirixTerminalClient] Note: Could not ensure user %s: %s", user_id, e
                )

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/agents")
            json: JSON body for the request
            params: Query parameters
            headers: Optional headers to merge with session headers
            
        Returns:
            Response data (parsed JSON)
            
        Raises:
            requests.HTTPError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        if self.debug:
            logger.debug("[MirixTerminalClient] %s %s", method, url)
            if json:
                logger.debug("[MirixTerminalClient] Request body: %s", json)
        
        response = self.session.request(
            method=method,
            url=url,
            json=json,
            params=params,
            timeout=self.timeout,
            headers=headers,
        )
        
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            # Try to extract error message from response
            try:
                error_detail = response.json().get("detail", str(e))
            except Exception:
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
        headers: Optional[Dict[str, str]] = None,
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
        
        data = self._request("GET", "/agents", params=params, headers=headers)
        return [AgentState(**agent) for agent in data]

    def agent_exists(
        self,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Check if an agent exists."""
        if not (agent_id or agent_name):
            raise ValueError("Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError("Only one of agent_id or agent_name can be provided")
        
        existing = self.list_agents(headers=headers)
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
        headers: Optional[Dict[str, str]] = None,
    ) -> AgentState:
        """Create an agent."""
        request_data = {
            "name": name,
            "agent_type": agent_type,
            "embedding_config": (
                embedding_config.model_dump() if embedding_config else None
            ),
            "llm_config": llm_config.model_dump() if llm_config else None,
            "memory": memory.model_dump() if memory else None,
            "block_ids": block_ids,
            "system": system,
            "tool_ids": tool_ids,
            "tool_rules": [
                rule.model_dump() if hasattr(rule, "model_dump") else rule
                for rule in (tool_rules or [])
            ],
            "include_base_tools": include_base_tools,
            "include_meta_memory_tools": include_meta_memory_tools,
            "metadata": metadata,
            "description": description,
            "initial_message_sequence": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in (initial_message_sequence or [])
            ],
            "tags": tags,
        }
        
        data = self._request("POST", "/agents", json=request_data, headers=headers)
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
        headers: Optional[Dict[str, str]] = None,
    ):
        """Update an agent."""
        request_data = {
            "name": name,
            "description": description,
            "system": system,
            "tool_ids": tool_ids,
            "metadata": metadata,
            "llm_config": llm_config.model_dump() if llm_config else None,
            "embedding_config": (
                embedding_config.model_dump() if embedding_config else None
            ),
            "message_ids": message_ids,
            "memory": memory.model_dump() if memory else None,
            "tags": tags,
        }
        
        data = self._request(
            "PATCH", f"/agents/{agent_id}", json=request_data, headers=headers
        )
        return AgentState(**data)

    def update_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> AgentState:
        """
        Update an agent's system prompt by agent name.
        
        This method updates the agent's system prompt and triggers a rebuild
        of the system message in the agent's message history.
        
        The method accepts short agent names like "episodic", "semantic", "core",
        or full names like "meta_memory_agent_episodic_memory_agent".
        
        Under the hood, this:
        1. Resolves the agent name to agent_id for the authenticated client
        2. Updates the agent.system field in PostgreSQL
        3. Updates the agent.system field in Redis cache
        4. Creates a new system message
        5. Updates message_ids[0] to reference the new system message
        
        Args:
            agent_name: Name of the agent to update. Can be:
                - Short name: "episodic", "semantic", "core", "procedural", 
                  "resource", "knowledge", "reflexion", "meta_memory_agent"
                - Full name: "meta_memory_agent_episodic_memory_agent", etc.
            system_prompt: The new system prompt text
            headers: Optional HTTP headers
            
        Returns:
            AgentState: The updated agent state
            
        Raises:
            Exception: If agent with the given name is not found
            
        Example:
            >>> client = MirixTerminalClient(client_id="your-client-id")
            >>> 
            >>> # Update episodic memory agent's system prompt
            >>> updated_agent = client.update_system_prompt(
            ...     agent_name="episodic",
            ...     system_prompt='''You are an episodic memory agent specialized in 
            ...     sales conversations. Focus on extracting key customer interactions,
            ...     pain points, and buying signals.'''
            ... )
            >>> 
            >>> print(f"Updated agent: {updated_agent.name}")
            >>> print(f"New system prompt: {updated_agent.system[:100]}...")
            
            >>> # Update semantic memory agent
            >>> updated_agent = client.update_system_prompt(
            ...     agent_name="semantic",
            ...     system_prompt="You are a semantic memory agent..."
            ... )
            
            >>> # Can also use full name
            >>> updated_agent = client.update_system_prompt(
            ...     agent_name="meta_memory_agent_core_memory_agent",
            ...     system_prompt="You are a core memory agent..."
            ... )
        
        Note:
            Common agent names:
            - "episodic" or "meta_memory_agent_episodic_memory_agent"
            - "semantic" or "meta_memory_agent_semantic_memory_agent"
            - "core" or "meta_memory_agent_core_memory_agent"
            - "procedural" or "meta_memory_agent_procedural_memory_agent"
            - "resource" or "meta_memory_agent_resource_memory_agent"
            - "knowledge" or "meta_memory_agent_knowledge_memory_agent"
            - "reflexion" or "meta_memory_agent_reflexion_agent"
            - "meta_memory_agent" (the parent meta agent)
        """
        request_data = {"system_prompt": system_prompt}
        data = self._request(
            "PATCH", 
            f"/agents/by-name/{agent_name}/system", 
            json=request_data, 
            headers=headers
        )
        return AgentState(**data)

    def get_agent(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> AgentState:
        """Get an agent by ID."""
        data = self._request("GET", f"/agents/{agent_id}", headers=headers)
        return AgentState(**data)

    def get_agent_id(
        self, agent_name: str, headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Get agent ID by name."""
        agents = self.list_agents(headers=headers)
        for agent in agents:
            if agent.name == agent_name:
                return agent.id
        return None

    def delete_agent(self, agent_id: str, headers: Optional[Dict[str, str]] = None):
        """Delete an agent."""
        self._request("DELETE", f"/agents/{agent_id}", headers=headers)

    def rename_agent(
        self, agent_id: str, new_name: str, headers: Optional[Dict[str, str]] = None
    ):
        """Rename an agent."""
        self.update_agent(agent_id, name=new_name, headers=headers)

    def get_tools_from_agent(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> List[Tool]:
        """Get tools from an agent."""
        agent = self.get_agent(agent_id, headers=headers)
        return agent.tools

    def add_tool_to_agent(self, agent_id: str, tool_id: str):
        """Add a tool to an agent."""
        raise NotImplementedError("add_tool_to_agent not yet implemented in REST API")

    def remove_tool_from_agent(self, agent_id: str, tool_id: str):
        """Remove a tool from an agent."""
        raise NotImplementedError(
            "remove_tool_from_agent not yet implemented in REST API"
        )

    # ========================================================================
    # Memory Methods
    # ========================================================================

    def get_in_context_memory(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> Memory:
        """Get in-context memory of an agent."""
        data = self._request("GET", f"/agents/{agent_id}/memory", headers=headers)
        return Memory(**data)

    def update_in_context_memory(
        self, agent_id: str, section: str, value: Union[List[str], str]
    ) -> Memory:
        """Update in-context memory."""
        raise NotImplementedError(
            "update_in_context_memory not yet implemented in REST API"
        )

    def get_archival_memory_summary(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> ArchivalMemorySummary:
        """Get archival memory summary."""
        data = self._request(
            "GET", f"/agents/{agent_id}/memory/archival", headers=headers
        )
        return ArchivalMemorySummary(**data)

    def get_recall_memory_summary(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> RecallMemorySummary:
        """Get recall memory summary."""
        data = self._request(
            "GET", f"/agents/{agent_id}/memory/recall", headers=headers
        )
        return RecallMemorySummary(**data)

    def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get in-context messages."""
        raise NotImplementedError(
            "get_in_context_messages not yet implemented in REST API"
        )

    # ========================================================================
    # Message Methods
    # ========================================================================

    def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,  # End-user ID for message attribution
        name: Optional[str] = None,
        stream: Optional[bool] = False,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> MirixResponse:
        """Send a message to an agent.
        
        Args:
            message: The message text to send
            role: The role of the message sender (user/system)
            agent_id: The ID of the agent to send the message to
            user_id: Optional end-user ID for message attribution. If not provided,
                    messages will be associated with the default user. This is critical
                    for multi-tenant applications to properly isolate user conversations.
            name: Optional name of the message sender
            stream: Enable streaming (not yet implemented)
            stream_steps: Stream intermediate steps
            stream_tokens: Stream tokens as they are generated
            filter_tags: Optional filter tags for categorization and filtering.
                Example: {"project_id": "proj-alpha", "session_id": "sess-123"}
            use_cache: Control Redis cache behavior (default: True)
            headers: Optional headers to include in the request
        
        Returns:
            MirixResponse: The response from the agent
        
        Example:
            >>> response = client.send_message(
            ...     message="What's the status?",
            ...     role="user",
            ...     agent_id="agent123",
            ...     user_id="user-456",
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
        
        # Include user_id if provided
        if user_id is not None:
            request_data["user_id"] = user_id
        
        # Include filter_tags if provided
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        # Include use_cache if not default
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        data = self._request(
            "POST", f"/agents/{agent_id}/messages", json=request_data, headers=headers
        )
        return MirixResponse(**data)

    def user_message(
        self, 
        agent_id: str, 
        message: str, 
        user_id: Optional[str] = None,  # End-user ID
        headers: Optional[Dict[str, str]] = None
    ) -> MirixResponse:
        """Send a user message to an agent.
        
        Args:
            agent_id: The ID of the agent to send the message to
            message: The message text to send
            user_id: Optional end-user ID for message attribution
            headers: Optional headers to include in the request
            
        Returns:
            MirixResponse: The response from the agent
        """
        return self.send_message(
            message=message, 
            role="user", 
            agent_id=agent_id, 
            user_id=user_id,  # Pass user_id
            headers=headers
        )

    def get_messages(
        self,
        agent_id: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
        limit: Optional[int] = 1000,
        use_cache: bool = True,
        headers: Optional[Dict[str, str]] = None,
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
        
        data = self._request(
            "GET", f"/agents/{agent_id}/messages", params=params, headers=headers
        )
        return [Message(**msg) for msg in data]

    # ========================================================================
    # Tool Methods
    # ========================================================================

    def list_tools(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Tool]:
        """List all tools."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        
        data = self._request("GET", "/tools", params=params, headers=headers)
        return [Tool(**tool) for tool in data]

    def get_tool(self, id: str, headers: Optional[Dict[str, str]] = None) -> Tool:
        """Get a tool by ID."""
        data = self._request("GET", f"/tools/{id}", headers=headers)
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
            "create_tool with function not supported in MirixTerminalClient. "
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
            "create_or_update_tool with function not supported in MirixTerminalClient. "
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

    def delete_tool(self, id: str, headers: Optional[Dict[str, str]] = None):
        """Delete a tool."""
        self._request("DELETE", f"/tools/{id}", headers=headers)

    def get_tool_id(
        self, name: str, headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Get tool ID by name."""
        tools = self.list_tools(headers=headers)
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
        self,
        label: Optional[str] = None,
        templates_only: Optional[bool] = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Block]:
        """List blocks."""
        params = {}
        if label:
            params["label"] = label
        
        data = self._request("GET", "/blocks", params=params, headers=headers)
        return [Block(**block) for block in data]

    def get_block(
        self, block_id: str, headers: Optional[Dict[str, str]] = None
    ) -> Block:
        """Get a block by ID."""
        data = self._request("GET", f"/blocks/{block_id}", headers=headers)
        return Block(**data)

    def create_block(
        self,
        label: str,
        value: str,
        limit: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Block:
        """Create a block."""
        block_data = {
            "label": label,
            "value": value,
            "limit": limit,
        }
        
        block = Block(**block_data)
        data = self._request(
            "POST", "/blocks", json=block.model_dump(), headers=headers
        )
        return Block(**data)

    def delete_block(self, id: str, headers: Optional[Dict[str, str]] = None) -> Block:
        """Delete a block."""
        self._request("DELETE", f"/blocks/{id}", headers=headers)

    # ========================================================================
    # Human/Persona Methods
    # ========================================================================

    def create_human(
        self, name: str, text: str, headers: Optional[Dict[str, str]] = None
    ) -> Human:
        """Create a human block."""
        human = Human(value=text)
        data = self._request(
            "POST", "/blocks", json=human.model_dump(), headers=headers
        )
        return Human(**data)

    def create_persona(
        self, name: str, text: str, headers: Optional[Dict[str, str]] = None
    ) -> Persona:
        """Create a persona block."""
        persona = Persona(value=text)
        data = self._request(
            "POST", "/blocks", json=persona.model_dump(), headers=headers
        )
        return Persona(**data)

    def list_humans(self, headers: Optional[Dict[str, str]] = None) -> List[Human]:
        """List human blocks."""
        blocks = self.list_blocks(label="human", headers=headers)
        return [Human(**block.model_dump()) for block in blocks]

    def list_personas(self, headers: Optional[Dict[str, str]] = None) -> List[Persona]:
        """List persona blocks."""
        blocks = self.list_blocks(label="persona", headers=headers)
        return [Persona(**block.model_dump()) for block in blocks]

    def update_human(
        self, human_id: str, text: str, headers: Optional[Dict[str, str]] = None
    ) -> Human:
        """Update a human block."""
        raise NotImplementedError("update_human not yet implemented in REST API")

    def update_persona(
        self, persona_id: str, text: str, headers: Optional[Dict[str, str]] = None
    ) -> Persona:
        """Update a persona block."""
        raise NotImplementedError("update_persona not yet implemented in REST API")

    def get_persona(self, id: str, headers: Optional[Dict[str, str]] = None) -> Persona:
        """Get a persona block."""
        data = self._request("GET", f"/blocks/{id}", headers=headers)
        return Persona(**data)

    def get_human(self, id: str, headers: Optional[Dict[str, str]] = None) -> Human:
        """Get a human block."""
        data = self._request("GET", f"/blocks/{id}", headers=headers)
        return Human(**data)

    def get_persona_id(
        self, name: str, headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Get persona ID by name."""
        personas = self.list_personas(headers=headers)
        if personas:
            return personas[0].id
        return None

    def get_human_id(self, name: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Get human ID by name."""
        humans = self.list_humans(headers=headers)
        if humans:
            return humans[0].id
        return None

    def delete_persona(self, id: str, headers: Optional[Dict[str, str]] = None):
        """Delete a persona."""
        self.delete_block(id, headers=headers)

    def delete_human(self, id: str, headers: Optional[Dict[str, str]] = None):
        """Delete a human."""
        self.delete_block(id, headers=headers)

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def list_model_configs(
        self, headers: Optional[Dict[str, str]] = None
    ) -> List[LLMConfig]:
        """List available LLM configurations."""
        data = self._request("GET", "/config/llm", headers=headers)
        return [LLMConfig(**config) for config in data]

    def list_embedding_configs(
        self, headers: Optional[Dict[str, str]] = None
    ) -> List[EmbeddingConfig]:
        """List available embedding configurations."""
        data = self._request("GET", "/config/embedding", headers=headers)
        return [EmbeddingConfig(**config) for config in data]

    # ========================================================================
    # Organization Methods
    # ========================================================================

    def create_org(
        self, name: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Organization:
        """Create an organization."""
        data = self._request(
            "POST", "/organizations", json={"name": name}, headers=headers
        )
        return Organization(**data)

    def list_orgs(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Organization]:
        """List organizations."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        
        data = self._request("GET", "/organizations", params=params, headers=headers)
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
        import logging
        import os
        
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
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        update_agents: Optional[bool] = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> AgentState:
        """
        Initialize a meta agent with the given configuration.
        
        This creates a meta memory agent that manages multiple specialized memory agents
        (episodic, semantic, procedural, etc.) for the current project.
        
        There are two ways to provide configuration (in order of precedence):
        1. provider (+ optional api_key): Use default config for the specified provider
        2. config: Provide a complete configuration dictionary
        
        Args:
            provider: LLM provider name ("openai", "anthropic", "google_ai").
                     Uses default configuration for that provider.
            api_key: Optional API key for the LLM provider. If not provided,
                     you can set it via environment variables (e.g., OPENAI_API_KEY).
            model: Optional model name to override the provider's default model
            config: Configuration dictionary with llm_config, embedding_config, etc.
            update_agents: Whether to update existing agents
            headers: Optional HTTP headers for the request
            
        Returns:
            AgentState: The initialized meta agent
            
        Examples:
            # Simplest setup - just provider (api_key from environment variable)
            >>> client = MirixTerminalClient(client_id="your-client-id")
            >>> meta_agent = client.initialize_meta_agent(provider="openai")
            
            # With explicit API key
            >>> meta_agent = client.initialize_meta_agent(
            ...     provider="openai",
            ...     api_key="sk-proj-xxx"
            ... )
            
            # With custom model
            >>> meta_agent = client.initialize_meta_agent(
            ...     provider="openai",
            ...     api_key="sk-proj-xxx",
            ...     model="gpt-4o"
            ... )
            
            # Using Anthropic (no embeddings)
            >>> meta_agent = client.initialize_meta_agent(
            ...     provider="anthropic",
            ...     api_key="sk-ant-xxx"
            ... )
            
            # Using Google AI (Gemini models)
            >>> meta_agent = client.initialize_meta_agent(provider="google_ai")
            
            # Using a config dictionary
            >>> config = {
            ...     "llm_config": {"model": "gemini-2.0-flash"},
            ...     "embedding_config": {"model": "text-embedding-004"}
            ... }
            >>> meta_agent = client.initialize_meta_agent(config=config)
        """
        
        # Option 1: Generate config from provider (api_key is optional)
        if provider:
            config = _get_provider_config(
                provider=provider,
                api_key=api_key,
                model=model,
            )
        
        if not config:
            raise ValueError(
                "Configuration required. Provide one of:\n"
                "  - provider for quick setup (e.g., provider='openai')\n"
                "  - config for a configuration dictionary"
            )
        
        # Load system prompts from folder if specified and not already provided
        if (
            config.get("meta_agent_config")
            and config["meta_agent_config"].get("system_prompts_folder")
            and not config.get("system_prompts")
        ):
            config["meta_agent_config"]["system_prompts"] = self._load_system_prompts(
                config["meta_agent_config"]
            )
            del config["meta_agent_config"]["system_prompts_folder"]

        # Prepare request data - org is determined from client_id on server side
        request_data = {
            "config": config,
            "update_agents": update_agents,
        }
        
        # Make API request to initialize meta agent
        data = self._request(
            "POST", "/agents/meta/initialize", json=request_data, headers=headers
        )
        self._meta_agent = AgentState(**data)
        return self._meta_agent
    
    def add(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        chaining: bool = False,
        verbose: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        occurred_at: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
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
            chaining: Enable/disable chaining (default: True)
            verbose: If True, enable verbose output during memory processing
            filter_tags: Optional dict of tags for filtering and categorization.
                        Example: {"project_id": "proj-123", "session_id": "sess-456"}
            use_cache: Control Redis cache behavior (default: True)
            occurred_at: Optional ISO 8601 timestamp string for episodic memory.
                        If provided, episodic memories will use this timestamp instead of current time.
                        Format: "2025-11-18T10:30:00" or "2025-11-18T10:30:00+00:00"
                        Example: "2025-11-18T15:30:00"
            headers: Optional headers dict to include in the request. Useful for passing
                    per-request authentication tokens. Example: {"Authorization": "Bearer token123"}
        
        Returns:
            Dict containing:
                - success (bool): True if message was queued successfully
                - message (str): Status message
                - status (str): "queued" - indicates async processing
                - agent_id (str): Meta agent ID processing the messages
                - message_count (int): Number of messages queued
            
        Raises:
            ValueError: If occurred_at format is invalid

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
            ...     filter_tags={"session_id": "sess-789"},
            ...     occurred_at="2025-11-18T15:30:00"
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
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first."
            )

        # Validate occurred_at format if provided
        if occurred_at is not None:
            _validate_occurred_at(occurred_at)  # Raises ValueError if invalid
        
        self._ensure_user_exists(user_id, headers=headers)
        
        # Prepare request data - org is determined from client_id on server side
        request_data = {
            "user_id": user_id,
            "meta_agent_id": self._meta_agent.id,
            "messages": messages,
            "chaining": chaining,
            "verbose": verbose,
        }
        
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        if occurred_at is not None:
            request_data["occurred_at"] = occurred_at

        return self._request("POST", "/memory/add", json=request_data, headers=headers)

    def clear_memory(self, user_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Clear all memories for a specific user.
        
        Args:
            user_id: User ID whose memory should be cleared
            headers: Optional HTTP headers
            
        Returns:
            Dict containing success status and message
        """
        request_data = {"user_id": user_id}
        return self._request("POST", "/memory/clear", json=request_data, headers=headers)

    def delete_memory(self, memory_type: str, memory_id: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Delete a specific memory by type and ID.
        
        Args:
            memory_type: Type of memory ('episodic', 'semantic', 'procedural', 'resource', 'knowledge')
            memory_id: ID of the memory to delete
            headers: Optional HTTP headers
            
        Returns:
            Dict containing success status and message
        """
        valid_types = ['episodic', 'semantic', 'procedural', 'resource', 'knowledge']
        if memory_type not in valid_types:
            raise ValueError(f"Invalid memory_type. Must be one of {valid_types}")
            
        return self._request("DELETE", f"/memory/{memory_type}/{memory_id}", headers=headers)

    def retrieve_with_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        local_model_for_retrieval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories based on conversation context with optional temporal filtering.
        
        This method analyzes the conversation and retrieves relevant memories from all memory systems.
        It can automatically extract temporal expressions from queries (e.g., "today", "yesterday")
        or accept explicit date ranges for filtering episodic memories.
        
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
            local_model_for_retrieval: Optional local Ollama model for topic extraction
            start_date: Optional start date/time for filtering episodic memories (ISO 8601 format).
                       Only episodic memories with occurred_at >= start_date will be returned.
                       Examples: "2025-11-19T00:00:00" or "2025-11-19T00:00:00+00:00"
            end_date: Optional end date/time for filtering episodic memories (ISO 8601 format).
                     Only episodic memories with occurred_at <= end_date will be returned.
                     Examples: "2025-11-19T23:59:59" or "2025-11-19T23:59:59+00:00"
        
        Returns:
            Dict containing:
            - success: Boolean indicating success
            - topics: Extracted topics from the conversation
            - temporal_expression: Extracted temporal phrase (if any)
            - date_range: Applied date range filter (if any)
            - memories: Retrieved memories organized by type
            
        Examples:
            >>> # Automatic temporal parsing from query
            >>> memories = client.retrieve_with_conversation(
            ...     user_id='user_123',
            ...     messages=[
            ...         {"role": "user", "content": [{"type": "text", "text": "What happened today?"}]}
            ...     ]
            ... )

            >>> # Explicit date range
            >>> memories = client.retrieve_with_conversation(
            ...     user_id='user_123',
            ...     messages=[
            ...         {"role": "user", "content": [{"type": "text", "text": "What meetings did I have?"}]}
            ...     ],
            ...     start_date="2025-11-19T00:00:00",
            ...     end_date="2025-11-19T23:59:59"
            ... )

            >>> # Combine with filter_tags
            >>> memories = client.retrieve_with_conversation(
            ...     user_id='user_123',
            ...     messages=[
            ...         {"role": "user", "content": [{"type": "text", "text": "Show me yesterday's work"}]}
            ...     ],
            ...     filter_tags={"category": "work"}
            ... )
        """
        if not self._meta_agent:
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first."
            )
        
        self._ensure_user_exists(user_id)
        
        # Prepare request data - org is determined from client_id on server side
        request_data = {
            "user_id": user_id,
            "messages": messages,
            "limit": limit,
            "local_model_for_retrieval": local_model_for_retrieval,
        }
        
        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags
        
        if not use_cache:
            request_data["use_cache"] = use_cache
        
        # Add temporal filtering parameters
        if start_date is not None:
            request_data["start_date"] = start_date
        if end_date is not None:
            request_data["end_date"] = end_date

        return self._request(
            "POST", "/memory/retrieve/conversation", json=request_data, headers=headers
        )
    
    def retrieve_with_topic(
        self,
        user_id: str,
        topic: str,
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant memories based on a topic.
        
        This method searches for memories related to a specific topic or keyword.
        
        Args:
            user_id: User ID for the conversation
            topic: Topic or keyword to search for
            limit: Maximum number of items to retrieve per memory type (default: 10)
            filter_tags: Optional dict of tags for filtering results.
                        Only memories matching these tags will be returned.
            use_cache: Control Redis cache behavior (default: True)
        
        Returns:
            Dict containing retrieved memories organized by type
            
        Example:
            >>> memories = client.retrieve_with_topic(
            ...     user_id='user_123',
            ...     topic="dinner",
            ...     limit=5,
            ...     filter_tags={"session_id": "sess-789"}
            ... )
        """
        if not self._meta_agent:
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first."
            )
        
        self._ensure_user_exists(user_id, headers=headers)
        
        params = {
            "user_id": user_id,
            "topic": topic,
            "limit": limit,
            "use_cache": use_cache,
        }

        # Encode filter_tags as JSON string for query parameter
        if filter_tags is not None:
            params["filter_tags"] = json.dumps(filter_tags)
        
        return self._request(
            "GET", "/memory/retrieve/topic", params=params, headers=headers
        )
    
    def search(
        self,
        user_id: str,
        query: str,
        memory_type: str = "all",
        search_field: str = "null",
        search_method: str = "bm25",
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories using various search methods with optional temporal filtering.
        Similar to the search_in_memory tool function.
        
        This method performs a search across specified memory types and returns
        a flat list of results.
        
        The organization is automatically determined from the client_id.
        
        Args:
            user_id: User ID for the conversation
            query: Search query
            memory_type: Type of memory to search. Options: "episodic", "resource", 
                        "procedural", "knowledge", "semantic", "all" (default: "all")
            search_field: Field to search in. Options vary by memory type:
                         - episodic: "summary", "details"
                         - resource: "summary", "content"
                         - procedural: "summary", "steps"
                         - knowledge: "caption", "secret_value"
                         - semantic: "name", "summary", "details"
                         - For "all": use "null" (default)
            search_method: Search method. Options: "bm25" (default), "embedding"
            limit: Maximum number of results per memory type (default: 10)
            filter_tags: Optional filter tags for additional filtering (scope added automatically)
            similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0).
                                 Only results with cosine distance < threshold are returned.
                                 Recommended values:
                                 - 0.5 (strict: only highly relevant results)
                                 - 0.7 (moderate: reasonably relevant results)
                                 - 0.9 (loose: loosely related results)
                                 - None (no filtering, returns all top N results)
                                 Only applies when search_method="embedding". Default: None
            start_date: Optional start date/time for filtering episodic memories (ISO 8601 format).
                       Only episodic memories with occurred_at >= start_date will be returned.
                       Examples: "2025-12-05T00:00:00" or "2025-12-05T00:00:00Z"
            end_date: Optional end date/time for filtering episodic memories (ISO 8601 format).
                     Only episodic memories with occurred_at <= end_date will be returned.
                     Examples: "2025-12-05T23:59:59" or "2025-12-05T23:59:59Z"
        
        Returns:
            Dict containing:
                - success: bool
                - query: str (the search query)
                - memory_type: str (the memory type searched)
                - search_field: str (the field searched)
                - search_method: str (the search method used)
                - date_range: dict (applied date range, if any)
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
            >>> 
            >>> # Search with additional filter tags
            >>> filtered_results = client.search(
            ...     user_id='user_123',
            ...     query="QuickBooks",
            ...     filter_tags={"project": "alpha", "expert_id": "expert-123"}
            ... )
            >>> 
            >>> # Search with temporal filtering (episodic memories only)
            >>> temporal_results = client.search(
            ...     user_id='user_123',
            ...     query="meetings",
            ...     start_date="2025-12-01T00:00:00",
            ...     end_date="2025-12-05T23:59:59"
            ... )
            >>> 
            >>> # Search with similarity threshold (embedding only)
            >>> relevant_results = client.search(
            ...     user_id='user_123',
            ...     query="database optimization",
            ...     search_method="embedding",
            ...     similarity_threshold=0.7
            ... )
        """
        if not self._meta_agent:
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first."
            )
        
        self._ensure_user_exists(user_id, headers=headers)
        
        # Prepare params - org is determined from client_id on server side
        params = {
            "user_id": user_id,
            "query": query,
            "memory_type": memory_type,
            "search_field": search_field,
            "search_method": search_method,
            "limit": limit,
        }
        
        # Add filter_tags if provided
        if filter_tags:
            import json
            params["filter_tags"] = json.dumps(filter_tags)
        
        # Add similarity threshold if provided
        if similarity_threshold is not None:
            params["similarity_threshold"] = similarity_threshold
        
        # Add temporal filtering parameters
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        
        return self._request("GET", "/memory/search", params=params, headers=headers)

    def search_all_users(
        self,
        query: str,
        memory_type: str = "all",
        search_field: str = "null",
        search_method: str = "bm25",
        limit: int = 10,
        filter_tags: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories across ALL users in the organization with optional temporal filtering.
        Results are automatically filtered by client scope.
        
        The organization and client are automatically determined from the client_id.
        
        Args:
            query: Search query
            memory_type: Type of memory to search. Options: "episodic", "resource", 
                        "procedural", "knowledge", "semantic", "all" (default: "all")
            search_field: Field to search in. Options vary by memory type:
                         - episodic: "summary", "details"
                         - resource: "summary", "content"
                         - procedural: "summary", "steps"
                         - knowledge: "caption", "secret_value"
                         - semantic: "name", "summary", "details"
                         - For "all": use "null" (default)
            search_method: Search method. Options: "bm25" (default), "embedding"
            limit: Maximum results per memory type (total across all users)
            filter_tags: Optional additional filter tags (scope added automatically)
            similarity_threshold: Optional similarity threshold for embedding search (0.0-2.0).
                                 Only results with cosine distance < threshold are returned.
                                 Recommended: 0.5 (strict), 0.7 (moderate), 0.9 (loose), None (no filter).
                                 Only applies when search_method="embedding". Default: None
            start_date: Optional start date/time for filtering episodic memories (ISO 8601 format).
                       Only episodic memories with occurred_at >= start_date will be returned.
                       Examples: "2025-12-05T00:00:00" or "2025-12-05T00:00:00Z"
            end_date: Optional end date/time for filtering episodic memories (ISO 8601 format).
                     Only episodic memories with occurred_at <= end_date will be returned.
                     Examples: "2025-12-05T23:59:59" or "2025-12-05T23:59:59Z"
        
        Returns:
            Dict containing:
                - success: bool
                - query: str (the search query)
                - memory_type: str (the memory type searched)
                - search_field: str (the field searched)
                - search_method: str (the search method used)
                - date_range: dict (applied date range, if any)
                - results: List[Dict] (flat list of results with user_id for each)
                - count: int (total number of results)
                - client_id: str (which client was used)
                - organization_id: str (which org was searched)
                - client_scope: str (scope used for filtering)
                - filter_tags: dict (applied filter tags)
            
        Example:
            >>> # Search all users' episodic memories
            >>> results = client.search_all_users(
            ...     query="meeting notes",
            ...     memory_type="episodic",
            ...     limit=20
            ... )
            >>> print(f"Found {results['count']} memories across users")
            >>> 
            >>> # Search with additional filters
            >>> results = client.search_all_users(
            ...     query="project documentation",
            ...     filter_tags={"project": "alpha"},
            ...     memory_type="resource"
            ... )
            >>> 
            >>> # Search across all users with temporal filtering
            >>> results = client.search_all_users(
            ...     query="project updates",
            ...     start_date="2025-12-01T00:00:00",
            ...     end_date="2025-12-05T23:59:59"
            ... )
            >>> 
            >>> # Search across all users with similarity threshold
            >>> results = client.search_all_users(
            ...     query="QuickBooks troubleshooting",
            ...     search_method="embedding",
            ...     similarity_threshold=0.7,
            ...     limit=20
            ... )
        """
        if not self._meta_agent:
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first."
            )
        
        params = {
            "query": query,
            "memory_type": memory_type,
            "search_field": search_field,
            "search_method": search_method,
            "limit": limit,
        }
        
        # Add filter_tags if provided
        if filter_tags:
            import json
            params["filter_tags"] = json.dumps(filter_tags)
        
        # Add similarity threshold if provided
        if similarity_threshold is not None:
            params["similarity_threshold"] = similarity_threshold
        
        # Add temporal filtering parameters
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        
        return self._request("GET", "/memory/search_all_users", params=params, headers=headers)

    # ========================================================================
    # LangChain/Composio/CrewAI Integration (Not Supported)
    # ========================================================================
