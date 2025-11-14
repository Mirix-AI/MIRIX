"""
Mirix Client Module

This module provides client implementations for interacting with Mirix agents:
- LocalClient: For embedded/local deployments (server runs in-process)
- MirixClient: For cloud deployments (server accessed via REST API)
"""

from mirix.client.client import AbstractClient, LocalClient
from mirix.client.remote_client import MirixClient

__all__ = ["AbstractClient", "LocalClient", "MirixClient", "create_client"]


def create_client(
    mode: str = "local",
    **kwargs,
):
    """
    Factory function to create a Mirix client.
    
    Args:
        mode: Client mode - "local" or "remote"
        **kwargs: Additional arguments passed to the client constructor
        
    For LocalClient (mode="local"):
        - user_id: Optional user ID
        - org_id: Optional organization ID
        - debug: Enable debug logging
        - default_llm_config: Default LLM configuration
        - default_embedding_config: Default embedding configuration
        
    For MirixClient (mode="remote"):
        - base_url: API server URL (required)
        - api_key: API key for authentication
        - client_id: Client ID (required)
        - org_id: Organization ID (required)
        - debug: Enable debug logging
        - timeout: Request timeout in seconds (default: 60)
        - max_retries: Max number of retries (default: 3)
    
    Returns:
        AbstractClient: A LocalClient or MirixClient instance
        
    Examples:
        >>> # Create a local client (embedded server)
        >>> client = create_client(mode="local")
        
        >>> # Create a remote client (cloud deployment)
        >>> client = create_client(
        ...     mode="remote",
        ...     base_url="https://api.mirix.ai",
        ...     api_key="sk-..."
        ... )
        
        >>> # Both clients have the same interface
        >>> agent = client.create_agent(name="my_agent")
        >>> response = client.send_message(
        ...     agent_id=agent.id,
        ...     message="Hello!",
        ...     role="user"
        ... )
    """
    if mode == "local":
        return LocalClient(**kwargs)
    elif mode == "remote":
        if "base_url" not in kwargs:
            raise ValueError("base_url is required for remote mode")
        if "org_id" not in kwargs:
            raise ValueError("org_id is required for remote mode")
        return MirixClient(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'remote'")
