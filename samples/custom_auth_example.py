"""
Example: Using Custom Authentication Providers with Mirix

This example demonstrates how to use custom authentication providers for LLM endpoints
that require dynamic, short-lived authentication tokens

Use cases:
- Authentication systems with short lived token expiry
- Custom authentication schemes for enterprise LLM endpoints

The auth provider system works by:
1. Registering an auth provider globally in your server process
2. Referencing it by name in your LLMConfig
3. The provider is called at request time to get fresh headers (synchronously)
"""

import time
from datetime import datetime, timedelta
from typing import Dict

from mirix.llm_api.auth_provider import (
    AuthProvider,
    list_auth_providers,
    register_auth_provider,
)
from mirix.schemas.llm_config import LLMConfig


class SampleAuthProvider(AuthProvider):
    """
    Auth provider with synchronous token refresh.

    This shows the proper pattern for auth providers that need to
    make HTTP calls to refresh tokens using synchronous I/O.
    """

    def __init__(self, auth_endpoint: str, token_lifetime_seconds: int = 600):
        self.auth_endpoint = auth_endpoint
        self.token_lifetime_seconds = token_lifetime_seconds
        self.current_token = None
        self.token_expiry = None

    def _is_token_expired(self) -> bool:
        """Check if the current token is expired or about to expire."""
        if self.token_expiry is None:
            return True
        # Refresh if less than 60 seconds remaining
        return datetime.now() >= (self.token_expiry - timedelta(seconds=60))

    def _refresh_token(self) -> str:
        """
        Refresh the authentication token by making an HTTP request.

        In a real implementation, this would:
        - Use requests library for synchronous HTTP requests
        - Call your auth service to get a fresh claims-based ticket
        - Handle errors and retries
        - Log authentication events

        Example:
            import requests

            response = requests.post(
                self.auth_endpoint,
                json={"client_id": "...", "client_secret": "..."}
            )
            data = response.json()
            return data["access_token"]
        """
        # Simulate HTTP request with a small delay
        time.sleep(0.1)  # Represents network latency

        new_token = f"claims_token_{int(time.time())}"
        self.current_token = new_token
        self.token_expiry = datetime.now() + timedelta(seconds=self.token_lifetime_seconds)
        print(f"[Auth] Refreshed token (sync HTTP), expires at {self.token_expiry}")
        return new_token

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers, refreshing token if necessary.

        This method is called before each LLM request to get fresh headers.
        All auth providers must implement this as a synchronous method.
        """
        if self._is_token_expired():
            self._refresh_token()

        return {
            "Authorization": f"Bearer {self.current_token}",
            "X-Token-Expiry": self.token_expiry.isoformat(),
        }


# Register the auth provider
sample_auth_provider = SampleAuthProvider(auth_endpoint="https://auth.example.com/token", token_lifetime_seconds=600)
register_auth_provider("sample_provider", sample_auth_provider)


def create_llm_config_with_auth():
    """
    Example showing how to use registered auth providers in your LLM configs.
    """
    sample_config = LLMConfig(
        model="gpt-4",
        model_endpoint_type="openai",
        context_window=8192,
        model_endpoint="https://custom-endpoint.example.com/v1",
        auth_provider="sample_provider",  # Uses SampleAuthProvider
    )

    return sample_config


def test_auth_provider():
    """
    Example of testing your auth providers.
    """
    from mirix.llm_api.auth_provider import get_auth_provider

    print("\n=== Testing Auth Provider ===\n")

    # Test sample provider with token refresh
    print("\n1. Testing sample_provider (with refresh):")
    provider = get_auth_provider("sample_provider")
    if provider:
        # First call - should trigger refresh
        headers1 = provider.get_auth_headers()
        print(f"   First call headers: {headers1}")

        # Second call - should use cached token
        headers2 = provider.get_auth_headers()
        print(f"   Second call headers: {headers2}")


if __name__ == "__main__":
    # List all registered providers
    print("Registered auth providers:", list_auth_providers())

    sample_config = create_llm_config_with_auth()
    print(f"\nCreated LLM config with auth provider:")
    print(f"  - {sample_config.auth_provider}")

    test_auth_provider()
