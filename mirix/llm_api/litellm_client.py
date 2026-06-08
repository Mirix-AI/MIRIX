from typing import List, Optional

from mirix.errors import (
    ErrorCode,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
)
from mirix.llm_api.helpers import convert_to_structured_output
from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.log import get_logger
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.openai.chat_completion_request import (
    ChatCompletionRequest,
    FunctionSchema,
    ToolFunctionChoice,
    cast_message_to_subtype,
)
from mirix.schemas.openai.chat_completion_request import (
    FunctionCall as ToolFunctionChoiceFunctionCall,
)
from mirix.schemas.openai.chat_completion_request import Tool as OpenAITool
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse

logger = get_logger(__name__)


class LiteLLMClient(LLMClientBase):
    """LLM client that uses LiteLLM to route requests to 100+ providers."""

    async def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        openai_message_list = [
            cast_message_to_subtype(m.to_openai_dict()) for m in messages
        ]

        model = llm_config.model
        if not model:
            logger.warning(
                f"Model type not set in llm_config: {llm_config.model_dump_json(indent=4)}"
            )

        tool_choice = None
        if tools:
            tool_choice = "required"

        if force_tool_call is not None:
            tool_choice = ToolFunctionChoice(
                type="function",
                function=ToolFunctionChoiceFunctionCall(name=force_tool_call),
            )

        data = ChatCompletionRequest(
            model=model,
            messages=openai_message_list,
            tools=(
                [OpenAITool(type="function", function=f) for f in tools]
                if tools
                else None
            ),
            tool_choice=tool_choice,
            user=str(),
            max_completion_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
        )

        if data.tools is not None and len(data.tools) > 0:
            for tool in data.tools:
                try:
                    structured_output_version = convert_to_structured_output(
                        tool.function.model_dump()
                    )
                    tool.function = FunctionSchema(**structured_output_version)
                except ValueError as e:
                    logger.warning(
                        f"Failed to convert tool function to structured output, tool={tool}, error={e}"
                    )
        else:
            delattr(data, "tool_choice")

        return data.model_dump(exclude_unset=True)

    async def request(self, request_data: dict) -> dict:
        import litellm

        api_key = getattr(self.llm_config, "api_key", None)
        base_url = self.llm_config.model_endpoint

        kwargs = {
            **request_data,
            "drop_params": True,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url

        logger.debug(
            "LiteLLM Request - Model: %s, Endpoint: %s",
            request_data.get("model"),
            base_url,
        )

        response = await litellm.acompletion(**kwargs)
        return response.model_dump()

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[PydanticMessage],
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(**response_data)

    def handle_llm_error(self, e: Exception) -> Exception:
        qualname = f"{type(e).__module__}.{type(e).__qualname__}"

        if "AuthenticationError" in qualname:
            logger.error(f"[LiteLLM] Authentication error: {e}")
            return LLMAuthenticationError(
                message=f"Authentication failed: {e}",
                code=ErrorCode.UNAUTHENTICATED,
            )

        if "RateLimitError" in qualname:
            logger.warning("[LiteLLM] Rate limited: %s", e)
            return LLMRateLimitError(
                message=f"Rate limited: {e}",
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
            )

        if "BadRequestError" in qualname:
            logger.warning("[LiteLLM] Bad request: %s", e)
            return LLMBadRequestError(
                message=f"Bad request: {e}",
                code=ErrorCode.INVALID_ARGUMENT,
            )

        if "NotFoundError" in qualname:
            logger.warning("[LiteLLM] Not found: %s", e)
            return LLMNotFoundError(
                message=f"Resource not found: {e}",
                code=ErrorCode.NOT_FOUND,
            )

        if "APIConnectionError" in qualname or "Timeout" in qualname:
            logger.warning("[LiteLLM] Connection error: %s", e)
            return LLMConnectionError(
                message=f"Connection failed: {e}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        if "PermissionDeniedError" in qualname:
            logger.error(f"[LiteLLM] Permission denied: {e}")
            return LLMPermissionDeniedError(
                message=f"Permission denied: {e}",
                code=ErrorCode.PERMISSION_DENIED,
            )

        status_code = getattr(e, "status_code", None)
        if status_code and status_code >= 500:
            logger.warning("[LiteLLM] Server error (%s): %s", status_code, e)
            return LLMServerError(
                message=f"Server error: {e}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
            )

        return super().handle_llm_error(e)
