"""
Authentication provider system for dynamic header injection in LLM requests.

This module provides a global registry system for authentication providers that can
inject custom headers (like claims-based tickets) at request time. This is useful for
custom OpenAI-compatible endpoints that require dynamic, short-lived authentication tokens.

Example:
    >>> from mirix.llm_api.auth_provider import register_auth_provider
    >>>
    >>> class MyAuthProvider(AuthProvider):
    ...     def get_auth_headers(self):
    ...         token = get_fresh_token()  # Use synchronous calls
    ...         return {"Authorization": f"Bearer {token}"}
    >>>
    >>> register_auth_provider("my_provider", MyAuthProvider())
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mirix.log import get_logger

logger = get_logger(__name__)

# Global registry for auth providers
_AUTH_PROVIDERS: Dict[str, "AuthProvider"] = {}


class AuthProvider(ABC):
    """
    Abstract base class for authentication providers.

    Authentication providers are responsible for generating dynamic authentication
    headers that are injected into LLM API requests at request time.

    All auth providers must implement get_auth_headers() as a synchronous method.
    For I/O operations (HTTP requests, database queries), use synchronous libraries
    like `requests` instead of async libraries.
    """

    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for the current request (synchronous).

        This method is called at request time to fetch fresh authentication headers.
        For short-lived tokens (e.g., 10-minute claims-based tickets), this method
        should handle token refresh logic internally using synchronous I/O.

        Returns:
            Dict[str, str]: A dictionary of HTTP headers to inject into the request.
                           Typically includes Authorization header, but can include
                           any custom headers needed.

        Example:
            >>> def get_auth_headers(self):
            ...     token = self.refresh_token()  # Use synchronous call
            ...     return {
            ...         "Authorization": f"Bearer {token}",
            ...         "X-Custom-Header": "value"
            ...     }
        """
        raise NotImplementedError

    async def get_auth_headers_async(self) -> Dict[str, str]:
        """
        Async variant of :meth:`get_auth_headers`, called from MIRIX's async
        LLM/embedding request paths.

        The default offloads the synchronous :meth:`get_auth_headers` to a
        worker thread so a blocking auth round-trip never stalls the event loop.
        Providers that can refresh tokens natively async — and especially those
        that cache/coalesce token refreshes — should override this to avoid the
        thread hop entirely.
        """
        return await asyncio.to_thread(self.get_auth_headers)


def register_auth_provider(name: str, provider: AuthProvider) -> None:
    """
    Register an authentication provider in the global registry.

    Args:
        name: Unique name for this auth provider. This name is referenced
              in LLMConfig.auth_provider field.
        provider: An instance of AuthProvider to register.

    Example:
        >>> class MyAuthProvider(AuthProvider):
        ...     def get_auth_headers(self):
        ...         return {"Authorization": "Bearer my-token"}
        >>>
        >>> register_auth_provider("my_provider", MyAuthProvider())
    """
    if name in _AUTH_PROVIDERS:
        logger.warning(f"Overwriting existing auth provider: {name}")
    _AUTH_PROVIDERS[name] = provider
    logger.info(f"Registered auth provider: {name}")


def get_auth_provider(name: str) -> Optional[AuthProvider]:
    """
    Retrieve an authentication provider from the global registry.

    Args:
        name: Name of the auth provider to retrieve.

    Returns:
        The registered AuthProvider instance, or None if not found.
    """
    return _AUTH_PROVIDERS.get(name)


def unregister_auth_provider(name: str) -> bool:
    """
    Remove an authentication provider from the global registry.

    Args:
        name: Name of the auth provider to remove.

    Returns:
        True if the provider was removed, False if it didn't exist.
    """
    if name in _AUTH_PROVIDERS:
        del _AUTH_PROVIDERS[name]
        logger.info(f"Unregistered auth provider: {name}")
        return True
    return False


def list_auth_providers() -> list[str]:
    """
    List all registered auth provider names.

    Returns:
        List of registered provider names.
    """
    return list(_AUTH_PROVIDERS.keys())
