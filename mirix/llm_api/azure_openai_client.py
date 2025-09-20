import os
from typing import List, Optional

from openai import AsyncAzureOpenAI, AsyncStream, AzureOpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from mirix.llm_api.openai_client import OpenAIClient
from mirix.log import get_logger
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.services.provider_manager import ProviderManager
from mirix.settings import model_settings

logger = get_logger(__name__)


class AzureOpenAIClient(OpenAIClient):
    """
    Azure OpenAI client that extends the base OpenAI client with Azure-specific configurations.
    Most of the implementation is inherited from OpenAIClient since Azure OpenAI uses the same API interface.
    """

    def _prepare_client_kwargs(self) -> dict:
        """
        Prepare Azure-specific client initialization parameters.
        Azure requires api_key, api_version, azure_endpoint, and azure_deployment.
        """
        # Check for custom API key in LLMConfig first (for custom models)
        custom_api_key = getattr(self.llm_config, 'api_key', None)
        if custom_api_key:
            api_key = custom_api_key
        else:
            # Check for database-stored API key first, fall back to model_settings and environment
            override_key = ProviderManager().get_azure_openai_override_key()
            api_key = override_key or model_settings.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            
        # Get Azure-specific configurations
        azure_endpoint = getattr(self.llm_config, 'azure_endpoint', None) or model_settings.azure_base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = getattr(self.llm_config, 'api_version', None) or model_settings.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
        azure_deployment = getattr(self.llm_config, 'azure_deployment', None) or self.llm_config.model
        
        # Validate required parameters
        if not api_key:
            raise ValueError("Azure OpenAI API key is required. Set it via LLMConfig, settings, or AZURE_OPENAI_API_KEY environment variable.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set it via LLMConfig, settings, or AZURE_OPENAI_ENDPOINT environment variable.")
        if not azure_deployment:
            raise ValueError("Azure OpenAI deployment name is required. Set it via LLMConfig or use the model name.")
        
        kwargs = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
        }
        
        logger.debug(f"Azure OpenAI client initialized with endpoint: {azure_endpoint}, deployment: {azure_deployment}")
        return kwargs

    def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        """
        Build request data for Azure OpenAI API.
        Azure uses deployment names as the model parameter.
        """
        # Call parent method to build the base request
        request_data = super().build_request_data(
            messages=messages,
            llm_config=llm_config,
            tools=tools,
            force_tool_call=force_tool_call,
            existing_file_uris=existing_file_uris,
        )
        
        # For Azure, ensure we have the model field set to the deployment name
        # The deployment name can come from azure_deployment or fall back to model
        azure_deployment = getattr(llm_config, 'azure_deployment', None) or llm_config.model
        if azure_deployment:
            request_data['model'] = azure_deployment
            
        return request_data

    def request(self, request_data: dict) -> dict:
        """
        Performs synchronous request to Azure OpenAI API.
        """
        client = AzureOpenAI(**self._prepare_client_kwargs())
        response: ChatCompletion = client.chat.completions.create(**request_data)
        return response.model_dump()

    async def request_async(self, request_data: dict) -> dict:
        """
        Performs asynchronous request to Azure OpenAI API.
        """
        client = AsyncAzureOpenAI(**self._prepare_client_kwargs())
        response: ChatCompletion = await client.chat.completions.create(**request_data)
        return response.model_dump()

    def stream(self, request_data: dict) -> Stream[ChatCompletionChunk]:
        """
        Performs streaming request to Azure OpenAI API.
        """
        client = AzureOpenAI(**self._prepare_client_kwargs())
        response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(**request_data, stream=True)
        return response_stream

    async def stream_async(self, request_data: dict) -> AsyncStream[ChatCompletionChunk]:
        """
        Performs asynchronous streaming request to Azure OpenAI API.
        """
        client = AsyncAzureOpenAI(**self._prepare_client_kwargs())
        response_stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(**request_data, stream=True)
        return response_stream 