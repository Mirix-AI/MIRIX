from typing import List, Optional

from mirix.llm_api.openai_client import OpenAIClient
from mirix.log import get_logger
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.openai.chat_completion_request import (
    ChatCompletionRequest,
    Tool as OpenAITool,
    ToolFunctionChoice,
    cast_message_to_subtype,
)
from mirix.schemas.openai.chat_completion_request import FunctionCall as ToolFunctionChoiceFunctionCall

logger = get_logger(__name__)


class OpenRouterClient(OpenAIClient):
    """LLM client for OpenRouter API.

    Inherits from OpenAIClient and overrides behaviour that is
    incompatible with non-OpenAI models routed through OpenRouter:

    1. Skips ``convert_to_structured_output`` (``strict`` mode is
       OpenAI-specific and breaks Anthropic / Gemini models).
    2. Defaults ``tool_choice`` to ``"auto"`` instead of ``"required"``
       so that models can emit reasoning text between tool calls.
    """

    async def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        use_developer_message = llm_config.model.startswith("o1") or llm_config.model.startswith("o3")

        openai_message_list = [
            cast_message_to_subtype(m.to_openai_dict(use_developer_message=use_developer_message))
            for m in messages
        ]

        model = llm_config.model or None

        tool_choice = "required" if tools else None

        if force_tool_call is not None:
            tool_choice = ToolFunctionChoice(
                type="function",
                function=ToolFunctionChoiceFunctionCall(name=force_tool_call),
            )

        data = ChatCompletionRequest(
            model=model,
            messages=await self.fill_image_content_in_messages(openai_message_list),
            tools=([OpenAITool(type="function", function=f) for f in tools] if tools else None),
            tool_choice=tool_choice,
            user=str(),
            max_completion_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
        )

        if not (data.tools is not None and len(data.tools) > 0):
            delattr(data, "tool_choice")

        # Skip convert_to_structured_output entirely — non-OpenAI models
        # do not support strict mode / additionalProperties.

        return data.model_dump(exclude_unset=True)
