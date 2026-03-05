"""
MirixClient implementation for Mirix.
This client communicates with a remote Mirix server via REST API.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

import httpx

from mirix.client.client import AbstractClient
from mirix.constants import FUNCTION_RETURN_CHAR_LIMIT
from mirix.log import get_logger
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

logger = get_logger(__name__)


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
        raise ValueError(f"occurred_at must be a string in ISO 8601 format, got {type(occurred_at).__name__}")

    # Validate ISO 8601 format
    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
    if not re.match(iso_pattern, occurred_at):
        raise ValueError(
            f"occurred_at must be in ISO 8601 format (e.g., '2025-11-18T10:30:00' or '2025-11-18T10:30:00+00:00'), got: {occurred_at}"
        )

    try:
        # Parse and validate the datetime
        dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        return dt
    except ValueError as e:
        raise ValueError(f"Invalid occurred_at datetime: {occurred_at}. Error: {str(e)}")


class RetryTransport(httpx.AsyncBaseTransport):
    """Async HTTP transport with retry on specific status codes."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        wrapped_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._wrapped = wrapped_transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._wrapped.handle_async_request(request)
                if response.status_code not in self.RETRYABLE_STATUS_CODES or attempt == self._max_retries:
                    return response
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    raise
            delay = self._backoff_factor * (2 ** attempt)
            await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class MirixClient(AbstractClient):
    """
    Client that communicates with a remote Mirix server via REST API.

    This client runs on the user's local machine and makes HTTP requests
    to a Mirix server hosted in the cloud.

    The API key identifies both the client and organization, so no explicit
    org_id is needed.

    Example:
        >>> client = MirixClient(
        ...     api_key="your-api-key",
        ...     base_url="https://api.mirix.ai",
        ... )
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_name: Optional[str] = None,
        client_scope: str = "",
        write_scope: Optional[str] = None,
        read_scopes: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        org_name: Optional[str] = None,
        org_id: Optional[str] = None,
        debug: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize MirixClient.

        This client represents a CLIENT APPLICATION (tenant), not an end-user.
        End-user IDs are passed per-request in the add() method.

        The API key identifies the client and organization, so no explicit org_id is needed.

        Args:
            api_key: API key for authentication (required; can also be set via MIRIX_API_KEY env var)
            base_url: Base URL of the Mirix API server (optional, can also be set via MIRIX_API_URL env var, default: "http://localhost:8000")
            client_name: Client name (optional, defaults to a generic label)
            client_scope: DEPRECATED - Use write_scope and read_scopes instead. For backward compatibility,
                          if client_scope is provided and write_scope/read_scopes are not, it will be used
                          as both write_scope and read_scopes[0].
            write_scope: Scope for writing memories (None = read-only client)
            read_scopes: List of scopes this client can read from
            debug: Whether to enable debug logging
            timeout: Request timeout in seconds
            max_retries: Number of retries for failed requests
            headers: Optional headers to include in the initialization requests
        """
        super().__init__(debug=debug)

        # Get base URL from parameter or environment variable
        self.base_url = (base_url or os.environ.get("MIRIX_API_URL", "http://localhost:8531")).rstrip("/")

        # Handle backward compatibility with client_scope
        if write_scope is None and read_scopes is None and client_scope:
            # Legacy mode: use client_scope for both write and read
            self.write_scope = client_scope
            self.read_scopes = [client_scope]
        else:
            self.write_scope = write_scope
            self.read_scopes = read_scopes if read_scopes is not None else []

        # Keep client_scope for backward compatibility in logging
        self.client_scope = client_scope or write_scope or ""
        self.timeout = timeout
        self._known_users: Set[str] = set()
        self.api_key = api_key or os.environ.get("MIRIX_API_KEY")
        self._init_headers = headers
        self.max_retries = max_retries

        if not self.api_key:
            import uuid

            if not client_id:
                client_id = f"client-{uuid.uuid4().hex[:8]}"
            if not org_id:
                org_id = f"org-{uuid.uuid4().hex[:8]}"

        self.client_id = client_id
        self.client_name = client_name or (client_id or "client")
        self.org_id = org_id
        self.org_name = org_name or (org_id or "org")

        # Build headers for the async HTTP client
        client_headers = {"Content-Type": "application/json"}
        if self.api_key:
            client_headers["X-API-Key"] = self.api_key
        else:
            client_headers["X-Client-ID"] = self.client_id
            client_headers["X-Org-ID"] = self.org_id

        transport = RetryTransport(
            max_retries=max_retries,
            backoff_factor=1.0,
            wrapped_transport=httpx.AsyncHTTPTransport(),
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers=client_headers,
        )

        # Track initialized meta agent for this project
        self._meta_agent: Optional[AgentState] = None

    @classmethod
    async def create(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_name: Optional[str] = None,
        client_scope: str = "",
        write_scope: Optional[str] = None,
        read_scopes: Optional[List[str]] = None,
        client_id: Optional[str] = None,
        org_name: Optional[str] = None,
        org_id: Optional[str] = None,
        debug: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
    ) -> "MirixClient":
        """Create and initialize a MirixClient. Use this instead of __init__ for async usage."""
        inst = cls(
            api_key=api_key,
            base_url=base_url,
            client_name=client_name,
            client_scope=client_scope,
            write_scope=write_scope,
            read_scopes=read_scopes,
            client_id=client_id,
            org_name=org_name,
            org_id=org_id,
            debug=debug,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
        )
        await inst._ensure_org_and_client_exist(headers=inst._init_headers)
        return inst

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _ensure_org_and_client_exist(self, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Ensure that the organization and client exist on the server.
        Creates them if they don't exist.

        Note: This method does NOT create users. Users are created per-request
        based on the user_id parameter in add() and other methods.

        Args:
            headers: Optional headers to include in the request
        """
        try:
            await self._request(
                "POST",
                "/organizations/create_or_get",
                json={"org_id": self.org_id, "name": self.org_name},
                headers=headers,
            )
            if self.debug:
                logger.debug(
                    "[MirixClient] Organization initialized: %s (name: %s)",
                    self.org_id,
                    self.org_name,
                )
            await self._request(
                "POST",
                "/clients/create_or_get",
                json={
                    "client_id": self.client_id,
                    "name": self.client_name,
                    "org_id": self.org_id,
                    "write_scope": self.write_scope,
                    "read_scopes": self.read_scopes,
                    "status": "active",
                },
                headers=headers,
            )
            if self.debug:
                logger.debug(
                    "[MirixClient] Client initialized: %s (name: %s, write_scope: %s, read_scopes: %s)",
                    self.client_id,
                    self.client_name,
                    self.write_scope,
                    self.read_scopes,
                )
        except Exception as e:
            if self.debug:
                logger.debug("[MirixClient] Note: Could not pre-create org/client: %s", e)
                logger.debug("[MirixClient] Server will create them on first request if needed")

    async def create_or_get_user(
        self,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        org_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a user if it doesn't exist, or get existing user.

        This method ensures a user exists in the backend database before performing
        operations that require a user_id. If the user already exists, it returns
        the existing user_id. If not, it creates a new user.

        The organization is automatically determined from the API key.

        Args:
            user_id: Optional user ID. If not provided, a random ID will be generated.
            user_name: Optional user name. Defaults to user_id if not provided.
            org_id: Optional organization ID. Defaults to client's org_id if not provided.

        Returns:
            str: The user_id (either existing or newly created)
        """
        request_data = {"user_id": user_id, "name": user_name}
        if not (headers and "X-API-Key" in headers) and (org_id or self.org_id):
            request_data["org_id"] = org_id or self.org_id

        response = await self._request(
            "POST", "/users/create_or_get", json=request_data, headers=headers
        )
        if isinstance(response, dict) and "id" in response:
            if self.debug:
                logger.debug("User ready: %s", response["id"])
            return response["id"]
        raise ValueError(f"Unexpected response from /users/create_or_get: {response}")

    async def _ensure_user_exists(
        self, user_id: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Ensure that the given user exists for the client's organization."""
        if not user_id or user_id in self._known_users:
            return
        try:
            await self._request(
                "POST",
                "/users/create_or_get",
                json={"user_id": user_id, "name": user_id},
                headers=headers,
            )
            self._known_users.add(user_id)
            if self.debug:
                logger.debug("[MirixClient] User ensured: %s", user_id)
        except Exception as e:
            if self.debug:
                logger.debug("[MirixClient] Note: Could not ensure user %s: %s", user_id, e)

    async def _request(
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
            httpx.HTTPStatusError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        if self.debug:
            logger.debug("[MirixClient] %s %s", method, url)
            if json:
                logger.debug("[MirixClient] Request body: %s", json)

        response = await self._client.request(
            method=method, url=url, json=json, params=params, headers=headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            raise httpx.HTTPStatusError(
                error_detail, request=e.request, response=e.response
            ) from e
        if response.content:
            return response.json()
        return None

    # ========================================================================
    # Agent Methods
    # ========================================================================

    async def list_agents(
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

        data = await self._request("GET", "/agents", params=params, headers=headers)
        return [AgentState(**agent) for agent in data]

    async def agent_exists(
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

        existing = await self.list_agents(headers=headers)
        if agent_id:
            return str(agent_id) in [str(agent.id) for agent in existing]
        else:
            return agent_name in [str(agent.name) for agent in existing]

    async def create_agent(
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
            "embedding_config": (embedding_config.model_dump() if embedding_config else None),
            "llm_config": llm_config.model_dump() if llm_config else None,
            "memory": memory.model_dump() if memory else None,
            "block_ids": block_ids,
            "system": system,
            "tool_ids": tool_ids,
            "tool_rules": [rule.model_dump() if hasattr(rule, "model_dump") else rule for rule in (tool_rules or [])],
            "include_base_tools": include_base_tools,
            "include_meta_memory_tools": include_meta_memory_tools,
            "metadata": metadata,
            "description": description,
            "initial_message_sequence": [
                msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in (initial_message_sequence or [])
            ],
            "tags": tags,
        }

        data = await self._request("POST", "/agents", json=request_data, headers=headers)
        return AgentState(**data)

    async def update_agent(
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
            "embedding_config": (embedding_config.model_dump() if embedding_config else None),
            "message_ids": message_ids,
            "memory": memory.model_dump() if memory else None,
            "tags": tags,
        }

        data = await self._request("PATCH", f"/agents/{agent_id}", json=request_data, headers=headers)
        return AgentState(**data)

    async def update_system_prompt(
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
                  "resource", "knowledge_vault", "reflexion", "meta_memory_agent"
                - Full name: "meta_memory_agent_episodic_memory_agent", etc.
            system_prompt: The new system prompt text
            headers: Optional HTTP headers

        Returns:
            AgentState: The updated agent state

        Raises:
            Exception: If agent with the given name is not found

        Example:
            >>> client = MirixClient(api_key="your-key")
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
            - "knowledge_vault" or "meta_memory_agent_knowledge_vault_memory_agent"
            - "reflexion" or "meta_memory_agent_reflexion_agent"
            - "meta_memory_agent" (the parent meta agent)
        """
        request_data = {"system_prompt": system_prompt}
        data = await self._request("PATCH", f"/agents/by-name/{agent_name}/system", json=request_data, headers=headers)
        return AgentState(**data)

    async def get_agent(self, agent_id: str, headers: Optional[Dict[str, str]] = None) -> AgentState:
        """Get an agent by ID."""
        data = await self._request("GET", f"/agents/{agent_id}", headers=headers)
        return AgentState(**data)

    async def get_agent_id(self, agent_name: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Get agent ID by name."""
        agents = await self.list_agents(headers=headers)
        for agent in agents:
            if agent.name == agent_name:
                return agent.id
        return None

    async def delete_agent(self, agent_id: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Delete an agent."""
        await self._request("DELETE", f"/agents/{agent_id}", headers=headers)

    async def rename_agent(self, agent_id: str, new_name: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Rename an agent."""
        await self.update_agent(agent_id, name=new_name, headers=headers)

    async def get_tools_from_agent(self, agent_id: str, headers: Optional[Dict[str, str]] = None) -> List[Tool]:
        """Get tools from an agent."""
        agent = await self.get_agent(agent_id, headers=headers)
        return agent.tools

    async def add_tool_to_agent(self, agent_id: str, tool_id: str) -> None:
        """Add a tool to an agent."""
        raise NotImplementedError("add_tool_to_agent not yet implemented in REST API")

    async def remove_tool_from_agent(self, agent_id: str, tool_id: str) -> None:
        """Remove a tool from an agent."""
        raise NotImplementedError("remove_tool_from_agent not yet implemented in REST API")

    # ========================================================================
    # Memory Methods
    # ========================================================================

    async def update_in_context_memory(self, agent_id: str, section: str, value: Union[List[str], str]) -> Memory:
        """Update in-context memory."""
        raise NotImplementedError("update_in_context_memory not yet implemented in REST API")

    async def get_archival_memory_summary(
        self, agent_id: str, headers: Optional[Dict[str, str]] = None
    ) -> ArchivalMemorySummary:
        """Get archival memory summary."""
        data = await self._request("GET", f"/agents/{agent_id}/memory/archival", headers=headers)
        return ArchivalMemorySummary(**data)

    async def get_recall_memory_summary(self, agent_id: str, headers: Optional[Dict[str, str]] = None) -> RecallMemorySummary:
        """Get recall memory summary."""
        data = await self._request("GET", f"/agents/{agent_id}/memory/recall", headers=headers)
        return RecallMemorySummary(**data)

    async def get_in_context_messages(self, agent_id: str) -> List[Message]:
        """Get in-context messages."""
        raise NotImplementedError("get_in_context_messages not yet implemented in REST API")

    # ========================================================================
    # Message Methods
    # ========================================================================

    async def send_message(
        self,
        message: str,
        role: str,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        user_id: Optional[str] = None,  # End-user ID for message attribution
        name: Optional[str] = None,
        stream: Optional[bool] = False,
        stream_steps: bool = False,
        stream_tokens: bool = False,
        chaining: Optional[bool] = None,
        verbose: Optional[bool] = None,
        block_filter_tags: Optional[Dict[str, Any]] = None,
        block_filter_tags_update_mode: Optional[str] = "merge",
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
            block_filter_tags: Optional dict; applied to block filter_tags when core memory agent runs.
            block_filter_tags_update_mode: "merge" (default) or "replace" for existing block filter_tags.
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
        if not (agent_id or agent_name):
            raise ValueError("Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError("Only one of agent_id or agent_name can be provided")
        resolved_agent_id = (await self.get_agent_id(agent_name, headers=headers)) if agent_name else agent_id
        if not resolved_agent_id:
            raise ValueError(f"Agent not found: {agent_name!r}")

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

        if block_filter_tags is not None:
            request_data["block_filter_tags"] = block_filter_tags

        if block_filter_tags_update_mode != "merge":
            request_data["block_filter_tags_update_mode"] = block_filter_tags_update_mode

        # Include use_cache if not default
        if not use_cache:
            request_data["use_cache"] = use_cache

        data = await self._request(
            "POST", f"/agents/{resolved_agent_id}/messages", json=request_data, headers=headers
        )
        return MirixResponse(**data)

    async def user_message(
        self,
        agent_id: str,
        message: str,
        user_id: Optional[str] = None,  # End-user ID
        block_filter_tags: Optional[Dict[str, Any]] = None,
        block_filter_tags_update_mode: Optional[str] = "merge",
        headers: Optional[Dict[str, str]] = None,
    ) -> MirixResponse:
        """Send a user message to an agent.

        Args:
            agent_id: The ID of the agent to send the message to
            message: The message text to send
            user_id: Optional end-user ID for message attribution
            block_filter_tags: Optional dict; applied to block filter_tags when core memory agent runs.
            block_filter_tags_update_mode: "merge" (default) or "replace" for existing block filter_tags.
            headers: Optional headers to include in the request

        Returns:
            MirixResponse: The response from the agent
        """
        return await self.send_message(
            message=message,
            role="user",
            agent_id=agent_id,
            user_id=user_id,
            block_filter_tags=block_filter_tags,
            block_filter_tags_update_mode=block_filter_tags_update_mode,
            headers=headers,
        )

    async def get_messages(
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

        data = await self._request("GET", f"/agents/{agent_id}/messages", params=params, headers=headers)
        return [Message(**msg) for msg in data]

    # ========================================================================
    # Tool Methods
    # ========================================================================

    async def list_tools(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Tool]:
        """List all tools."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/tools", params=params, headers=headers)
        return [Tool(**tool) for tool in data]

    async def get_tool(self, id: str, headers: Optional[Dict[str, str]] = None) -> Tool:
        """Get a tool by ID."""
        data = await self._request("GET", f"/tools/{id}", headers=headers)
        return Tool(**data)

    async def create_tool(
        self,
        func,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        return_char_limit: int = FUNCTION_RETURN_CHAR_LIMIT,
    ) -> Tool:
        """Create a tool."""
        raise NotImplementedError(
            "create_tool with function not supported in MirixClient. " "Tools must be created on the server side."
        )

    async def create_or_update_tool(
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

    async def update_tool(
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

    async def delete_tool(self, id: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Delete a tool."""
        await self._request("DELETE", f"/tools/{id}", headers=headers)

    async def get_tool_id(self, name: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Get tool ID by name."""
        tools = await self.list_tools(headers=headers)
        for tool in tools:
            if tool.name == name:
                return tool.id
        return None

    async def upsert_base_tools(self) -> List[Tool]:
        """Upsert base tools."""
        raise NotImplementedError("upsert_base_tools must be done on server side")

    # ========================================================================
    # Block Methods
    # ========================================================================

    async def list_blocks(
        self,
        label: Optional[str] = None,
        templates_only: Optional[bool] = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Block]:
        """List blocks."""
        params = {}
        if label:
            params["label"] = label

        data = await self._request("GET", "/blocks", params=params, headers=headers)
        return [Block(**block) for block in data]

    async def get_block(self, block_id: str, headers: Optional[Dict[str, str]] = None) -> Block:
        """Get a block by ID."""
        data = await self._request("GET", f"/blocks/{block_id}", headers=headers)
        return Block(**data)

    async def create_block(
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
        data = await self._request("POST", "/blocks", json=block.model_dump(), headers=headers)
        return Block(**data)

    async def delete_block(self, id: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Delete a block."""
        await self._request("DELETE", f"/blocks/{id}", headers=headers)

    # ========================================================================
    # Human/Persona Methods
    # ========================================================================

    async def create_human(self, name: str, text: str, headers: Optional[Dict[str, str]] = None) -> Human:
        """Create a human block."""
        human = Human(value=text)
        data = await self._request("POST", "/blocks", json=human.model_dump(), headers=headers)
        return Human(**data)

    async def create_persona(self, name: str, text: str, headers: Optional[Dict[str, str]] = None) -> Persona:
        """Create a persona block."""
        persona = Persona(value=text)
        data = await self._request("POST", "/blocks", json=persona.model_dump(), headers=headers)
        return Persona(**data)

    async def list_humans(self, headers: Optional[Dict[str, str]] = None) -> List[Human]:
        """List human blocks."""
        blocks = await self.list_blocks(label="human", headers=headers)
        return [Human(**block.model_dump()) for block in blocks]

    async def list_personas(self, headers: Optional[Dict[str, str]] = None) -> List[Persona]:
        """List persona blocks."""
        blocks = await self.list_blocks(label="persona", headers=headers)
        return [Persona(**block.model_dump()) for block in blocks]

    async def update_human(self, human_id: str, text: str, headers: Optional[Dict[str, str]] = None) -> Human:
        """Update a human block."""
        raise NotImplementedError("update_human not yet implemented in REST API")

    async def update_persona(self, persona_id: str, text: str, headers: Optional[Dict[str, str]] = None) -> Persona:
        """Update a persona block."""
        raise NotImplementedError("update_persona not yet implemented in REST API")

    async def get_persona(self, id: str, headers: Optional[Dict[str, str]] = None) -> Persona:
        """Get a persona block."""
        data = await self._request("GET", f"/blocks/{id}", headers=headers)
        return Persona(**data)

    async def get_human(self, id: str, headers: Optional[Dict[str, str]] = None) -> Human:
        """Get a human block."""
        data = await self._request("GET", f"/blocks/{id}", headers=headers)
        return Human(**data)

    async def get_persona_id(self, name: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Get persona ID by name."""
        personas = await self.list_personas(headers=headers)
        if personas:
            return personas[0].id
        return None

    async def get_human_id(self, name: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Get human ID by name."""
        humans = await self.list_humans(headers=headers)
        if humans:
            return humans[0].id
        return None

    async def delete_persona(self, id: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Delete a persona."""
        await self.delete_block(id, headers=headers)

    async def delete_human(self, id: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Delete a human."""
        await self.delete_block(id, headers=headers)

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    async def list_model_configs(self, headers: Optional[Dict[str, str]] = None) -> List[LLMConfig]:
        """List available LLM configurations."""
        data = await self._request("GET", "/config/llm", headers=headers)
        return [LLMConfig(**config) for config in data]

    async def list_embedding_configs(self, headers: Optional[Dict[str, str]] = None) -> List[EmbeddingConfig]:
        """List available embedding configurations."""
        data = await self._request("GET", "/config/embedding", headers=headers)
        return [EmbeddingConfig(**config) for config in data]

    # ========================================================================
    # Organization Methods
    # ========================================================================

    async def create_org(self, name: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> Organization:
        """Create an organization."""
        data = await self._request("POST", "/organizations", json={"name": name}, headers=headers)
        return Organization(**data)

    async def list_orgs(
        self,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Organization]:
        """List organizations."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = await self._request("GET", "/organizations", params=params, headers=headers)
        return [Organization(**org) for org in data]

    async def delete_org(self, org_id: str) -> Organization:
        """Delete an organization."""
        raise NotImplementedError("delete_org not yet implemented in REST API")

    # ========================================================================
    # Sandbox Methods (Not Implemented)
    # ========================================================================

    async def create_sandbox_config(self, config: Union[LocalSandboxConfig, E2BSandboxConfig]) -> SandboxConfig:
        """Create sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    async def update_sandbox_config(
        self,
        sandbox_config_id: str,
        config: Union[LocalSandboxConfig, E2BSandboxConfig],
    ) -> SandboxConfig:
        """Update sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    async def delete_sandbox_config(self, sandbox_config_id: str) -> None:
        """Delete sandbox config."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    async def list_sandbox_configs(self, limit: int = 50, cursor: Optional[str] = None) -> List[SandboxConfig]:
        """List sandbox configs."""
        raise NotImplementedError("Sandbox config not yet implemented in REST API")

    async def create_sandbox_env_var(
        self,
        sandbox_config_id: str,
        key: str,
        value: str,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """Create sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    async def update_sandbox_env_var(
        self,
        env_var_id: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SandboxEnvironmentVariable:
        """Update sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    async def delete_sandbox_env_var(self, env_var_id: str) -> None:
        """Delete sandbox environment variable."""
        raise NotImplementedError("Sandbox env vars not yet implemented in REST API")

    async def list_sandbox_env_vars(
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
                    logger.warning(f"Failed to load system prompt for {agent_name} from {prompt_file}: {e}")

        return prompts

    async def initialize_meta_agent(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        update_agents: Optional[bool] = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[AgentState]:
        """
        Initialize a meta agent with the given configuration.

        This creates a meta memory agent that manages multiple specialized memory agents
        (episodic, semantic, procedural, etc.) for the current project.

        Returns None if the client has no write_scope (read-only clients don't need agents).

        Args:
            config: Configuration dictionary with llm_config, embedding_config, etc.
            config_path: Path to YAML config file (alternative to config dict)

        Returns:
            AgentState if created/updated, None if client is read-only (no write_scope)

        Example:
            >>> client = MirixClient(api_key="your-api-key")
            >>> config = {
            ...     "llm_config": {"model": "gemini-2.0-flash"},
            ...     "embedding_config": {"model": "text-embedding-004"}
            ... }
            >>> meta_agent = client.initialize_meta_agent(config=config)
        """

        # Load config from file if provided
        if config_path:
            from pathlib import Path

            import yaml

            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

        if not config:
            raise ValueError("Either config or config_path must be provided")

        # Load system prompts from folder if specified and not already provided
        if (
            config.get("meta_agent_config")
            and config["meta_agent_config"].get("system_prompts_folder")
            and not config.get("system_prompts")
        ):
            config["meta_agent_config"]["system_prompts"] = self._load_system_prompts(config["meta_agent_config"])
            del config["meta_agent_config"]["system_prompts_folder"]

        # Prepare request data - org is determined from API key on server side
        request_data = {
            "config": config,
            "update_agents": update_agents,
        }

        # Make API request to initialize meta agent
        data = await self._request("POST", "/agents/meta/initialize", json=request_data, headers=headers)

        # Server returns null for read-only clients (no write_scope)
        if not data:
            self._meta_agent = None
            return None

        self._meta_agent = AgentState(**data)
        return self._meta_agent

    async def add(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        chaining: bool = True,
        verbose: bool = False,
        filter_tags: Optional[Dict[str, Any]] = None,
        block_filter_tags: Optional[Dict[str, Any]] = None,
        block_filter_tags_update_mode: Optional[str] = "merge",
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
            block_filter_tags: Optional dict applied to block filter_tags when core memory agent runs.
                              e.g. {"env": "staging", "team": "platform"}.
            block_filter_tags_update_mode: "merge" (default) or "replace" for existing block filter_tags.
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
                "Meta agent not initialized. Call initialize_meta_agent() first. "
                "If you already called it, the client may not have a write_scope configured."
            )

        # Validate occurred_at format if provided
        if occurred_at is not None:
            _validate_occurred_at(occurred_at)  # Raises ValueError if invalid

        await self._ensure_user_exists(user_id, headers=headers)

        # Prepare request data - org is determined from API key on server side
        request_data = {
            "user_id": user_id,
            "meta_agent_id": self._meta_agent.id,
            "messages": messages,
            "chaining": chaining,
            "verbose": verbose,
        }

        if filter_tags is not None:
            request_data["filter_tags"] = filter_tags

        if block_filter_tags is not None:
            request_data["block_filter_tags"] = block_filter_tags

        if block_filter_tags_update_mode != "merge":
            request_data["block_filter_tags_update_mode"] = block_filter_tags_update_mode

        if not use_cache:
            request_data["use_cache"] = use_cache

        if occurred_at is not None:
            request_data["occurred_at"] = occurred_at

        return await self._request("POST", "/memory/add", json=request_data, headers=headers)

    async def retrieve_with_conversation(
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
                "Meta agent not initialized. Call initialize_meta_agent() first. "
                "If you already called it, the client may not have a write_scope configured."
            )

        await self._ensure_user_exists(user_id, headers=headers)

        # Prepare request data - org is determined from API key on server side
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

        return await self._request("POST", "/memory/retrieve/conversation", json=request_data, headers=headers)

    async def retrieve_with_topic(
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
                "Meta agent not initialized. Call initialize_meta_agent() first. "
                "If you already called it, the client may not have a write_scope configured."
            )

        await self._ensure_user_exists(user_id, headers=headers)

        params = {
            "user_id": user_id,
            "topic": topic,
            "limit": limit,
            "use_cache": use_cache,
        }

        # Encode filter_tags as JSON string for query parameter
        if filter_tags is not None:
            params["filter_tags"] = json.dumps(filter_tags)

        return await self._request("GET", "/memory/retrieve/topic", params=params, headers=headers)

    async def search(
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
        org_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories using various search methods with optional temporal filtering.
        Similar to the search_in_memory tool function.

        This method performs a search across specified memory types and returns
        a flat list of results.

        The organization is automatically determined from the API key.

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
            org_id: Optional organization scope override (defaults to client's org)

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
                "Meta agent not initialized. Call initialize_meta_agent() first. "
                "If you already called it, the client may not have a write_scope configured."
            )

        await self._ensure_user_exists(user_id, headers=headers)

        # Prepare params - org is determined from API key on server side
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

        return await self._request("GET", "/memory/search", params=params, headers=headers)

    async def search_all_users(
        self,
        query: str,
        memory_type: str = "all",
        search_field: str = "null",
        search_method: str = "bm25",
        limit: int = 10,
        client_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        org_id: Optional[str] = None,
        include_core_memory: bool = False,
        block_filter_tags: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories across ALL users in the organization with optional temporal filtering.
        Results are automatically filtered by client scope.

        If client_id is provided, uses that client's organization.

        Args:
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
            limit: Maximum results per memory type (total across all users)
            client_id: Optional client ID (uses its org_id and scope for filtering)
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
            org_id: Optional organization scope (overridden by client's org if client_id provided)
            include_core_memory: When True, include a "core" section with block memory (scoped by client).
            block_filter_tags: When include_core_memory is True, only blocks whose filter_tags contain
                              these keys/values are returned. Scope is auto-injected from client.

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
            >>> # Search with specific client and additional filters
            >>> results = client.search_all_users(
            ...     query="project documentation",
            ...     client_id="client-123",
            ...     filter_tags={"project": "alpha"},
            ...     memory_type="resource"
            ... )
            >>>
            >>> # Search across all users with temporal filtering
            >>> results = client.search_all_users(
            ...     query="project updates",
            ...     client_id="client-123",
            ...     start_date="2025-12-01T00:00:00",
            ...     end_date="2025-12-05T23:59:59"
            ... )
            >>>
            >>> # Search across all users with similarity threshold
            >>> results = client.search_all_users(
            ...     query="QuickBooks troubleshooting",
            ...     client_id="client-123",
            ...     search_method="embedding",
            ...     similarity_threshold=0.7,
            ...     limit=20
            ... )
        """
        if not self._meta_agent:
            raise ValueError(
                "Meta agent not initialized. Call initialize_meta_agent() first. "
                "If you already called it, the client may not have a write_scope configured."
            )

        params = {
            "query": query,
            "memory_type": memory_type,
            "search_field": search_field,
            "search_method": search_method,
            "limit": limit,
        }

        # Add client_id if provided (server will use this client's org_id)
        if client_id:
            params["client_id"] = client_id

        # Add org_id if provided (used only if no client_id specified)
        if org_id:
            params["org_id"] = org_id
        elif not client_id:
            # Use current client's org_id if neither client_id nor org_id provided
            params["org_id"] = self.org_id

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

        if include_core_memory:
            params["include_core_memory"] = "true"
        if block_filter_tags is not None:
            import json as _json

            params["block_filter_tags"] = _json.dumps(block_filter_tags)

        return await self._request("GET", "/memory/search_all_users", params=params, headers=headers)

    # ========================================================================
    # LangChain/Composio/CrewAI Integration (Not Supported)
    # ========================================================================
