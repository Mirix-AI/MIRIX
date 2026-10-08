"""The opt-in session header on MIRIX's outbound LLM and embedding requests.

``MIRIX_LLM_SESSION_HEADER`` names a request header that carries the conversation
session a call serves (a proxy's per-session cost attribution: a gateway, Helicone,
LiteLLM). It is off by default; when on, only a value that is a valid MIRIX session id
is ever sent. Offline: the fake model server (``tests/fakes``) or a capturing fake SDK
client; nothing reaches a provider.
"""

import asyncio
import re

import pytest

from mirix.observability.context import (
    SESSION_VALUE_MAX_LEN,
    conversation_session_scope,
    current_conversation_session,
    outbound_session_headers,
)
from mirix.schemas.llm_config import LLMConfig
from mirix.settings import Settings, settings

HEADER = "x-skill-bench-item"
SID = "t0123456789abcdef0123"  # a harness task uid: "t" + 20 hex


@pytest.fixture
def header_on(monkeypatch):
    monkeypatch.setattr(settings, "llm_session_header", HEADER)
    return HEADER


def test_the_header_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MIRIX_LLM_SESSION_HEADER", raising=False)
    assert Settings().llm_session_header is None
    monkeypatch.setattr(settings, "llm_session_header", None)
    with conversation_session_scope(SID):
        assert outbound_session_headers() == {}


def test_the_setting_is_a_header_name_and_never_a_credential(monkeypatch):
    monkeypatch.setenv("MIRIX_LLM_SESSION_HEADER", "X-Session-Id")
    assert Settings().llm_session_header == "X-Session-Id"
    monkeypatch.setenv("MIRIX_LLM_SESSION_HEADER", "  ")
    assert Settings().llm_session_header is None  # blank: off
    # Surrounding whitespace (a .env line, shell quoting) is not a bad name:
    # importing the settings must not fail over it.
    for padded in ("x-item ", " x-item", "x-item\n", "\tx-item\r\n"):
        monkeypatch.setenv("MIRIX_LLM_SESSION_HEADER", padded)
        assert Settings().llm_session_header == "x-item", repr(padded)
    for bad in (
        "Authorization",
        "proxy-authorization",
        "Cookie",
        "x-api-key",
        "api-key",
        "Content-Type",
        "Host",
        "x-stainless-retry-count",
        "openai-organization",
        "x session",
        "x-se\nssion",
        "x:y",
    ):
        with pytest.raises(ValueError):
            Settings(llm_session_header=bad)


def test_only_a_valid_session_id_is_ever_sent(header_on):
    assert outbound_session_headers() == {}  # no session served
    with conversation_session_scope(SID):
        assert outbound_session_headers() == {HEADER: SID}
    longest = "a" * SESSION_VALUE_MAX_LEN
    with conversation_session_scope(longest):
        assert outbound_session_headers() == {HEADER: longest}
    for bad in (
        "a" * (SESSION_VALUE_MAX_LEN + 1),
        "two words",
        SID + "\n",
        "",
        "sess:1",
        "é",
        None,
    ):
        with conversation_session_scope(bad):
            assert outbound_session_headers() == {}, bad


def test_the_value_rule_is_mirix_session_id_rule():
    from mirix.schemas.message import SESSION_ID_MAX_LEN, SESSION_ID_SQL_PATTERN

    assert SESSION_VALUE_MAX_LEN == SESSION_ID_MAX_LEN
    rule = re.compile(SESSION_ID_SQL_PATTERN)
    for value in (SID, "a" * 64, "a-b_C9"):
        assert rule.fullmatch(value)


@pytest.mark.asyncio
async def test_scopes_nest_and_concurrent_tasks_keep_their_own(header_on):
    with conversation_session_scope("outer"):
        with conversation_session_scope("inner"):
            assert outbound_session_headers() == {HEADER: "inner"}
        with conversation_session_scope(None):  # a call serving several sessions
            assert outbound_session_headers() == {}
        assert outbound_session_headers() == {HEADER: "outer"}
    assert current_conversation_session.get() is None

    async def serve(sid: str) -> list:
        seen = []
        with conversation_session_scope(sid):
            for _ in range(3):
                seen.append(outbound_session_headers().get(HEADER))
                await asyncio.sleep(0)
        return seen

    first, second = await asyncio.gather(serve("s-one"), serve("s-two"))
    assert first == ["s-one"] * 3 and second == ["s-two"] * 3


def _llm_config(base_url: str) -> LLMConfig:
    return LLMConfig(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=base_url,
        context_window=200_000,
        max_output_tokens=16_000,
        api_key="fake-key",
    )


@pytest.mark.asyncio
async def test_the_openai_client_sends_the_session_of_its_scope(
    fake_llm_server, header_on, monkeypatch
):
    from mirix import constants
    from mirix.llm_api.openai_client import OpenAIClient

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)
    fake_llm_server.queue_text("one").queue_text("two").queue_text("three")
    client = OpenAIClient(_llm_config(fake_llm_server.base_url))
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    with conversation_session_scope(SID):
        await client.request(dict(body))
    await client.request(dict(body))  # outside any scope
    monkeypatch.setattr(settings, "llm_session_header", None)
    with conversation_session_scope(SID):
        await client.request(dict(body))  # the setting off
    sent = [r.headers.get(HEADER.lower()) for r in fake_llm_server.chat_requests()]
    assert sent == [SID, None, None]
    # The header never reaches the request body the model sees.
    assert all(SID not in str(r.json) for r in fake_llm_server.chat_requests())


@pytest.mark.asyncio
async def test_the_openai_client_stream_sends_the_session_too(header_on, monkeypatch):
    from mirix.llm_api import openai_client

    created = []

    class FakeCompletions:
        async def create(self, **kwargs):
            created.append(kwargs)
            return "a stream"

    class FakeSDK:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(openai_client, "AsyncOpenAI", FakeSDK)
    client = openai_client.OpenAIClient(_llm_config("http://127.0.0.1:9/v1"))
    with conversation_session_scope(SID):
        assert await client.stream({"model": "m", "messages": []}) == "a stream"
    assert await client.stream({"model": "m", "messages": []}) == "a stream"
    assert created[0]["extra_headers"] == {HEADER: SID} and created[0]["stream"]
    assert "extra_headers" not in created[1]  # nothing to send: the call as before


@pytest.mark.asyncio
async def test_the_http_embedding_clients_send_the_session(
    fake_llm_server, header_on, monkeypatch
):
    from mirix.embeddings import EmbeddingEndpoint, OpenRouterEmbedding

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    openrouter = OpenRouterEmbedding(
        api_key="fake-key",
        model="google/gemini-embedding-001",
        base_url=fake_llm_server.base_url,
        user="u",
    )
    endpoint = EmbeddingEndpoint(
        model="bge", base_url=fake_llm_server.base_url, user="u"
    )
    with conversation_session_scope(SID):
        assert await openrouter.get_text_embedding("a query")
        assert await endpoint.get_text_embedding("a query")
    assert await openrouter.get_text_embedding("no session")
    requests = fake_llm_server.embedding_requests()
    assert [r.headers.get(HEADER) for r in requests] == [SID, SID, None]
    # The client's own headers always win over the session header.
    assert requests[0].headers["authorization"] == "Bearer fake-key"


@pytest.mark.asyncio
async def test_the_sdk_embedding_clients_send_the_session(header_on, monkeypatch):
    from mirix import embeddings
    from mirix.schemas.embedding_config import EmbeddingConfig

    created = []

    class FakeEmbeddings:
        async def create(self, **kwargs):
            created.append(kwargs)
            vector = type("V", (), {"embedding": [1.0, 0.0]})()
            return type("R", (), {"data": [vector]})()

    azure = embeddings.AzureOpenAIEmbedding(
        api_endpoint="https://example.invalid",
        api_key="k",
        api_version="2024-02-01",
        model="m",
    )
    azure.client = type("C", (), {"embeddings": FakeEmbeddings()})()
    with conversation_session_scope(SID):
        await azure.get_text_embedding("x")
    await azure.get_text_embedding("y")
    assert created[0]["extra_headers"] == {HEADER: SID}
    assert "extra_headers" not in created[1]

    class FakeSDK:
        def __init__(self, **kwargs):
            self.embeddings = FakeEmbeddings()

    class Provider:
        def get_auth_headers(self):
            return {"Authorization": "Bearer token"}

    import mirix.llm_api.auth_provider as auth_provider

    monkeypatch.setattr(auth_provider, "get_auth_provider", lambda name: Provider())
    monkeypatch.setattr("openai.AsyncOpenAI", FakeSDK)
    custom = embeddings.OpenAIEmbeddingWithCustomAuth(
        EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="m",
            embedding_dim=2,
            auth_provider="p",
        ),
        auth_provider="p",
    )
    created.clear()
    with conversation_session_scope(SID):
        assert await custom.get_text_embedding("x") == [1.0, 0.0]
    assert created[0]["extra_headers"] == {HEADER: SID}
