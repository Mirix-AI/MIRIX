"""Tests for OpenRouter support: LLM client, embedding client, schema validation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mirix.llm_api.llm_client import LLMClient
from mirix.llm_api.openrouter_client import OpenRouterClient
from mirix.embeddings import OpenRouterEmbedding
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.embedding_config import EmbeddingConfig


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_llm_config_accepts_openrouter(self):
        config = LLMConfig(
            model="anthropic/claude-haiku-4.5",
            model_endpoint_type="openrouter",
            model_endpoint="https://openrouter.ai/api/v1",
            context_window=128000,
        )
        assert config.model_endpoint_type == "openrouter"

    def test_embedding_config_accepts_openrouter(self):
        config = EmbeddingConfig(
            embedding_model="google/gemini-embedding-001",
            embedding_endpoint_type="openrouter",
            embedding_endpoint="https://openrouter.ai/api/v1",
            embedding_dim=3072,
        )
        assert config.embedding_endpoint_type == "openrouter"


# ---------------------------------------------------------------------------
# LLMClient factory
# ---------------------------------------------------------------------------

class TestLLMClientFactory:
    def test_creates_openrouter_client(self):
        config = LLMConfig(
            model="anthropic/claude-haiku-4.5",
            model_endpoint_type="openrouter",
            model_endpoint="https://openrouter.ai/api/v1",
            context_window=128000,
        )
        client = LLMClient.create(config)
        assert isinstance(client, OpenRouterClient)

    def test_openai_still_creates_openai_client(self):
        from mirix.llm_api.openai_client import OpenAIClient
        config = LLMConfig(
            model="gpt-4o-mini",
            model_endpoint_type="openai",
            context_window=128000,
        )
        client = LLMClient.create(config)
        assert isinstance(client, OpenAIClient)
        assert not isinstance(client, OpenRouterClient)


# ---------------------------------------------------------------------------
# OpenRouterClient.build_request_data
# ---------------------------------------------------------------------------

class TestOpenRouterClient:
    @pytest.fixture
    def llm_config(self):
        return LLMConfig(
            model="anthropic/claude-haiku-4.5",
            model_endpoint_type="openrouter",
            model_endpoint="https://openrouter.ai/api/v1",
            context_window=128000,
            max_tokens=1024,
            temperature=0.7,
        )

    @pytest.fixture
    def client(self, llm_config):
        return OpenRouterClient(llm_config=llm_config)

    @pytest.fixture
    def mock_message(self):
        msg = MagicMock()
        msg.to_openai_dict.return_value = {
            "role": "user",
            "content": "Hello",
        }
        return msg

    @pytest.fixture
    def sample_tools(self):
        return [
            {
                "name": "search_memory",
                "description": "Search memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_no_strict_mode_in_tools(self, client, llm_config, mock_message, sample_tools):
        """OpenRouter must NOT add strict/additionalProperties to tool schemas."""
        data = await client.build_request_data(
            messages=[mock_message],
            llm_config=llm_config,
            tools=sample_tools,
        )
        for tool in data.get("tools", []):
            func = tool.get("function", {})
            assert "strict" not in func, "strict should not be set for OpenRouter"
            params = func.get("parameters", {})
            assert "additionalProperties" not in params, "additionalProperties should not be set for OpenRouter"

    @pytest.mark.asyncio
    async def test_tool_choice_required_with_tools(self, client, llm_config, mock_message, sample_tools):
        """tool_choice should be 'required' when tools are provided."""
        data = await client.build_request_data(
            messages=[mock_message],
            llm_config=llm_config,
            tools=sample_tools,
        )
        assert data.get("tool_choice") == "required"

    @pytest.mark.asyncio
    async def test_no_tool_choice_without_tools(self, client, llm_config, mock_message):
        """tool_choice should be absent when no tools are provided."""
        data = await client.build_request_data(
            messages=[mock_message],
            llm_config=llm_config,
            tools=None,
        )
        assert "tool_choice" not in data

    @pytest.mark.asyncio
    async def test_force_tool_call(self, client, llm_config, mock_message, sample_tools):
        """force_tool_call should set tool_choice to specific function."""
        data = await client.build_request_data(
            messages=[mock_message],
            llm_config=llm_config,
            tools=sample_tools,
            force_tool_call="search_memory",
        )
        tc = data.get("tool_choice", {})
        assert tc.get("type") == "function"
        assert tc.get("function", {}).get("name") == "search_memory"

    @pytest.mark.asyncio
    async def test_model_passed_through(self, client, llm_config, mock_message):
        """Model name should be passed as-is (e.g. 'anthropic/claude-haiku-4.5')."""
        data = await client.build_request_data(
            messages=[mock_message],
            llm_config=llm_config,
        )
        assert data["model"] == "anthropic/claude-haiku-4.5"

    @pytest.mark.asyncio
    async def test_inherits_request_method(self, client):
        """OpenRouterClient should inherit the request() method from OpenAIClient."""
        assert hasattr(client, "request")
        assert hasattr(client, "convert_response_to_chat_completion")


# ---------------------------------------------------------------------------
# OpenRouterEmbedding
# ---------------------------------------------------------------------------

class TestOpenRouterEmbedding:
    def test_init(self):
        emb = OpenRouterEmbedding(
            api_key="test-key",
            model="google/gemini-embedding-001",
            base_url="https://openrouter.ai/api/v1",
            user="test-user",
        )
        assert emb._api_key == "test-key"
        assert emb.model_name == "google/gemini-embedding-001"

    @pytest.mark.asyncio
    async def test_call_api_sends_bearer_auth(self):
        """Embedding requests must include Bearer auth header."""
        emb = OpenRouterEmbedding(
            api_key="sk-test-123",
            model="google/gemini-embedding-001",
            base_url="https://openrouter.ai/api/v1",
            user="user-1",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await emb._call_api("test text")

            assert result == [0.1, 0.2, 0.3]

            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert headers["Authorization"] == "Bearer sk-test-123"
            assert headers["Content-Type"] == "application/json"

            json_data = call_args.kwargs.get("json") or call_args[1].get("json")
            assert json_data["model"] == "google/gemini-embedding-001"
            assert json_data["input"] == "test text"

    @pytest.mark.asyncio
    async def test_call_api_error_handling(self):
        """Should raise TypeError on unexpected response format."""
        emb = OpenRouterEmbedding(
            api_key="sk-test",
            model="test-model",
            base_url="https://openrouter.ai/api/v1",
            user="user-1",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "invalid model"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(TypeError, match="Unexpected embedding response"):
                await emb._call_api("test text")


# ---------------------------------------------------------------------------
# OpenAI client comparison — ensure OpenAI path still adds strict mode
# ---------------------------------------------------------------------------

class TestOpenAIClientNotAffected:
    @pytest.mark.asyncio
    async def test_openai_client_adds_structured_output(self):
        """Verify the original OpenAIClient still converts to structured output
        (adds additionalProperties: false to parameters)."""
        from mirix.llm_api.openai_client import OpenAIClient

        config = LLMConfig(
            model="gpt-4o-mini",
            model_endpoint_type="openai",
            context_window=128000,
            max_tokens=1024,
            temperature=0.7,
        )
        client = OpenAIClient(llm_config=config)
        msg = MagicMock()
        msg.to_openai_dict.return_value = {"role": "user", "content": "Hi"}

        tools = [{
            "name": "test_fn",
            "description": "A test function",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "query"}},
                "required": ["q"],
            },
        }]

        data = await client.build_request_data(
            messages=[msg], llm_config=config, tools=tools,
        )

        # OpenAI client should have additionalProperties (from convert_to_structured_output)
        for tool in data.get("tools", []):
            params = tool.get("function", {}).get("parameters", {})
            assert params.get("additionalProperties") is False, \
                "OpenAI client should add additionalProperties=False"
