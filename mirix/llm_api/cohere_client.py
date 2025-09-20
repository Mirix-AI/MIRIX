from typing import List, Optional

from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse


class CohereClient(LLMClientBase):
    """Cohere client - currently a stub that falls back to original implementation"""

    def build_request_data(
        self,
        messages: List[Message],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
    ) -> dict:
        # TODO: Implement cohere-specific request building
        raise NotImplementedError("CohereClient not yet implemented - use fallback")

    def request(self, request_data: dict) -> dict:
        # TODO: Implement cohere-specific request
        raise NotImplementedError("CohereClient not yet implemented - use fallback")

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
    ) -> ChatCompletionResponse:
        # TODO: Implement cohere-specific response conversion
        raise NotImplementedError("CohereClient not yet implemented - use fallback")

    def handle_llm_error(self, e: Exception) -> Exception:
        # TODO: Implement cohere-specific error handling
        return super().handle_llm_error(e) 