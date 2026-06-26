"""Tests for OpenAIClient auth-header behavior in provider mode.

Regression coverage for the save-path 401 pattern: when an auth provider is
configured and its sign-in fails transiently, the client must NOT silently
send the request with dummy/no credentials (which the gateway rejects as a
misleading permanent 401). It must raise so the failure is classified as a
transient, retryable token-generation error.
"""

import pytest

from mirix.errors import ProviderTransientError
from mirix.llm_api.auth_provider import (
    AuthProvider,
    register_auth_provider,
)
from mirix.llm_api.openai_client import OpenAIClient
from mirix.queue.error_policy import Bucket, classify
from mirix.schemas.llm_config import LLMConfig


@pytest.fixture(autouse=True)
def cleanup_registry():
    from mirix.llm_api.auth_provider import (
        list_auth_providers,
        unregister_auth_provider,
    )

    for name in list_auth_providers():
        unregister_auth_provider(name)
    yield
    for name in list_auth_providers():
        unregister_auth_provider(name)


def _llm_config(auth_provider=None):
    return LLMConfig(
        model="gpt-4.1",
        model_endpoint_type="openai",
        model_endpoint="https://example.invalid/v1",
        context_window=8192,
        auth_provider=auth_provider,
    )


class _FailingProvider(AuthProvider):
    def get_auth_headers(self):
        raise RuntimeError("Offline ticket generation failed: ConnectTimeout")

    async def get_auth_headers_async(self):
        raise RuntimeError("Offline ticket generation failed: ConnectTimeout")


class _GoodProvider(AuthProvider):
    def get_auth_headers(self):
        return {"Authorization": "Intuit_IAM_Authentication real-token"}


@pytest.mark.asyncio
async def test_provider_mode_auth_failure_raises_not_swallowed():
    """A configured provider whose sign-in fails must raise, never send dummy creds.

    It must raise ProviderTransientError so the failure is classified as a
    retryable token-generation error (not a misleading permanent 401), and the
    original cause is preserved on the chain.
    """
    register_auth_provider("failing_provider", _FailingProvider())
    client = OpenAIClient(llm_config=_llm_config(auth_provider="failing_provider"))

    with pytest.raises(ProviderTransientError) as excinfo:
        await client._prepare_client_kwargs()

    assert "Offline ticket generation failed" in str(excinfo.value.__cause__)
    assert classify(excinfo.value) is Bucket.TRANSIENT


@pytest.mark.asyncio
async def test_no_provider_configured_does_not_raise():
    """With no auth provider configured (local dev), behavior is unchanged."""
    client = OpenAIClient(llm_config=_llm_config(auth_provider=None))
    kwargs = await client._prepare_client_kwargs()
    assert "default_headers" not in kwargs


@pytest.mark.asyncio
async def test_provider_mode_success_injects_headers():
    """A working provider still injects its headers as default_headers."""
    register_auth_provider("good_provider", _GoodProvider())
    client = OpenAIClient(llm_config=_llm_config(auth_provider="good_provider"))
    kwargs = await client._prepare_client_kwargs()
    assert kwargs["default_headers"]["Authorization"].startswith("Intuit_IAM_Authentication")
