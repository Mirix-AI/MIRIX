import os
import json
import base64
from typing import List, Optional
from mirix.utils import parse_json

import openai
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from mirix.errors import (
    ErrorCode,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMNotFoundError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)
from mirix.llm_api.helpers import add_inner_thoughts_to_functions, convert_to_structured_output, unpack_all_inner_thoughts_from_kwargs
from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION, INNER_THOUGHTS_KWARG_DESCRIPTION_GO_FIRST
from mirix.log import get_logger
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.openai.chat_completion_request import ChatCompletionRequest
from mirix.schemas.openai.chat_completion_request import FunctionCall as ToolFunctionChoiceFunctionCall
from mirix.schemas.openai.chat_completion_request import FunctionSchema
from mirix.schemas.openai.chat_completion_request import Tool as OpenAITool
from mirix.schemas.openai.chat_completion_request import ToolFunctionChoice, cast_message_to_subtype
from mirix.schemas.openai.chat_completion_request import UserMessage
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.services.provider_manager import ProviderManager
from mirix.settings import model_settings

logger = get_logger(__name__)


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 format with data URL prefix.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image with data URL prefix (e.g., "data:image/jpeg;base64,...")
    """
    import mimetypes
    
    # Get the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith('image/'):
        # Default to jpeg if we can't determine the type
        mime_type = 'image/jpeg'
    
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_string}"


class OpenAIClient(LLMClientBase):
    def _prepare_client_kwargs(self) -> dict:
        # Check for custom API key in LLMConfig first (for custom models)
        custom_api_key = getattr(self.llm_config, 'api_key', None)
        if custom_api_key:
            api_key = custom_api_key
        else:
            # Check for database-stored API key first, fall back to model_settings and environment
            override_key = ProviderManager().get_openai_override_key()
            api_key = override_key or model_settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            # supposedly the openai python client requires a dummy API key
            api_key = api_key or "DUMMY_API_KEY"
        kwargs = {"api_key": api_key, "base_url": self.llm_config.model_endpoint}
        return kwargs

    def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,  # Keep as dict for now as per base class
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        """
        Constructs a request object in the expected data format for the OpenAI API.
        """
        if tools and llm_config.put_inner_thoughts_in_kwargs:
            # Special case for LM Studio backend since it needs extra guidance to force out the thoughts first
            # TODO(fix)
            inner_thoughts_desc = (
                INNER_THOUGHTS_KWARG_DESCRIPTION_GO_FIRST if ":1234" in llm_config.model_endpoint else INNER_THOUGHTS_KWARG_DESCRIPTION
            )
            tools = add_inner_thoughts_to_functions(
                functions=tools,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=inner_thoughts_desc,
                put_inner_thoughts_first=True,
            )

        use_developer_message = llm_config.model.startswith("o1") or llm_config.model.startswith("o3")  # o-series models

        openai_message_list = [
            cast_message_to_subtype(
                m.to_openai_dict(
                    put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
                    use_developer_message=use_developer_message,
                )
            )
            for m in messages
        ]

        if llm_config.model:
            model = llm_config.model
        else:
            logger.warning(f"Model type not set in llm_config: {llm_config.model_dump_json(indent=4)}")
            model = None

        # force function calling for reliability, see https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        # TODO(matt) move into LLMConfig
        # TODO: This vllm checking is very brittle and is a patch at most
        tool_choice = None
        if llm_config.model_endpoint == "https://inference.memgpt.ai" or (llm_config.handle and "vllm" in self.llm_config.handle):
            tool_choice = "auto"  # TODO change to "required" once proxy supports it
        elif tools:
            # only set if tools is non-Null
            tool_choice = "required"

        if force_tool_call is not None:
            tool_choice = ToolFunctionChoice(type="function", function=ToolFunctionChoiceFunctionCall(name=force_tool_call))

        data = ChatCompletionRequest(
            model=model,
            messages=self.fill_image_content_in_messages(openai_message_list),
            tools=[OpenAITool(type="function", function=f) for f in tools] if tools else None,
            tool_choice=tool_choice,
            user=str(),
            max_completion_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
        )

        if "inference.memgpt.ai" in llm_config.model_endpoint:
            # override user id for inference.memgpt.ai
            import uuid

            data.user = str(uuid.UUID(int=0))
            data.model = "memgpt-openai"

        if data.tools is not None and len(data.tools) > 0:
            # Convert to structured output style (which has 'strict' and no optionals)
            for tool in data.tools:
                try:
                    structured_output_version = convert_to_structured_output(tool.function.model_dump())
                    tool.function = FunctionSchema(**structured_output_version)
                except ValueError as e:
                    logger.warning(f"Failed to convert tool function to structured output, tool={tool}, error={e}")

        return data.model_dump(exclude_unset=True)

    def fill_image_content_in_messages(self, openai_message_list):
        """
        Converts image URIs in the message to base64 format.
        """

        from mirix.constants import LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY

        global_image_idx = 0
        new_message_list = []

        image_content_loaded = False # it will always be false if `LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY` is False

        for message_idx, message in enumerate(openai_message_list[::-1]):

            if message.role != 'user':
                new_message_list.append(message)
                continue

            # It it is not a list, then it is not a message with image.
            # TODO: (yu) Single image as message should be the list, this probably needs to be warned in the beginning.
            if not isinstance(message.content, list):
                new_message_list.append(message)
                continue

            has_image = False

            message_content = []
            for m in message.content:
                if m['type'] == 'image_url':
                    if LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY and image_content_loaded:
                        message_content.append({
                            'type': 'text',
                            'text': "[System Message] There was an image here but now the image has been deleted to save space.",
                        })

                    else:
                        message_content.append({
                            'type': 'text',
                            'text': f"<image {global_image_idx}>",
                        })
                        file = self.file_manager.get_file_metadata_by_id(m['image_id'])
                        if file.source_url is not None:
                            message_content.append({
                                'type': 'image_url',
                                'image_url': {'url': file.source_url, 'detail': m['detail']},
                            })
                        elif file.file_path is not None:
                            message_content.append({
                                'type': 'image_url',
                                'image_url': {'url': encode_image(file.file_path), 'detail': m['detail']},
                            })
                        else:
                            raise ValueError(f"File {file.file_path} has no source_url or file_path")
                        global_image_idx += 1
                        has_image = True
                elif m['type'] == 'google_cloud_file_uri':
                    file = self.file_manager.get_file_metadata_by_id(m['cloud_file_uri'])
                    try:
                        local_path = self.cloud_file_mapping_manager.get_local_file(file.google_cloud_url)
                    except Exception as e:
                        local_path = None

                    if local_path is not None and os.path.exists(local_path):
                        message_content.append({
                            'type': 'image_url',
                            'image_url': {'url': encode_image(local_path)},
                        })
                    else:
                        message_content.append({
                            'type': 'text',
                            'text': f"[System Message] There was an image here but now the image has been deleted to save space.",
                        })
                    
                elif m['type'] == 'file_uri':
                    raise NotImplementedError("File URI is currently not supported for OpenAI")
                else:
                    message_content.append(m)
            message.content = message_content
            new_message_list.append(message)

            if has_image:
                if LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY:
                    # Load image content for the last message only.
                    image_content_loaded = True
        
        new_message_list = new_message_list[::-1]

        return new_message_list

    def request(self, request_data: dict) -> dict:
        """
        Performs underlying synchronous request to OpenAI API and returns raw response dict.
        """
        client = OpenAI(**self._prepare_client_kwargs())
        response: ChatCompletion = client.chat.completions.create(**request_data)
        return response.model_dump()

    async def request_async(self, request_data: dict) -> dict:
        """
        Performs underlying asynchronous request to OpenAI API and returns raw response dict.
        """
        client = AsyncOpenAI(**self._prepare_client_kwargs())
        response: ChatCompletion = await client.chat.completions.create(**request_data)
        return response.model_dump()

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[PydanticMessage],  # Included for consistency, maybe used later
    ) -> ChatCompletionResponse:
        """
        Converts raw OpenAI response dict into the ChatCompletionResponse Pydantic model.
        Handles potential extraction of inner thoughts if they were added via kwargs.
        """
        # OpenAI's response structure directly maps to ChatCompletionResponse
        # We just need to instantiate the Pydantic model for validation and type safety.
        chat_completion_response = ChatCompletionResponse(**response_data)

        # Unpack inner thoughts if they were embedded in function arguments
        if self.llm_config.put_inner_thoughts_in_kwargs:
            chat_completion_response = unpack_all_inner_thoughts_from_kwargs(
                response=chat_completion_response, inner_thoughts_key=INNER_THOUGHTS_KWARG
            )

        return chat_completion_response

    def stream(self, request_data: dict) -> Stream[ChatCompletionChunk]:
        """
        Performs underlying streaming request to OpenAI and returns the stream iterator.
        """
        client = OpenAI(**self._prepare_client_kwargs())
        response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(**request_data, stream=True)
        return response_stream

    async def stream_async(self, request_data: dict) -> AsyncStream[ChatCompletionChunk]:
        """
        Performs underlying asynchronous streaming request to OpenAI and returns the async stream iterator.
        """
        client = AsyncOpenAI(**self._prepare_client_kwargs())
        response_stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(**request_data, stream=True)
        return response_stream

    def handle_llm_error(self, e: Exception) -> Exception:
        """
        Maps OpenAI-specific errors to common LLMError types.
        """
        if isinstance(e, openai.APIConnectionError):
            logger.warning(f"[OpenAI] API connection error: {e}")
            return LLMConnectionError(
                message=f"Failed to connect to OpenAI: {str(e)}",
                code=ErrorCode.INTERNAL_SERVER_ERROR,
                details={"cause": str(e.__cause__) if e.__cause__ else None},
            )

        if isinstance(e, openai.RateLimitError):
            logger.warning(f"[OpenAI] Rate limited (429). Consider backoff. Error: {e}")
            return LLMRateLimitError(
                message=f"Rate limited by OpenAI: {str(e)}",
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                details=e.body,  # Include body which often has rate limit details
            )

        if isinstance(e, openai.BadRequestError):
            logger.warning(f"[OpenAI] Bad request (400): {str(e)}")
            # BadRequestError can signify different issues (e.g., invalid args, context length)
            # Check message content if finer-grained errors are needed
            # Example: if "context_length_exceeded" in str(e): return LLMContextLengthExceededError(...)
            return LLMBadRequestError(
                message=f"Bad request to OpenAI: {str(e)}",
                code=ErrorCode.INVALID_ARGUMENT,  # Or more specific if detectable
                details=e.body,
            )

        if isinstance(e, openai.AuthenticationError):
            logger.error(f"[OpenAI] Authentication error (401): {str(e)}")  # More severe log level
            return LLMAuthenticationError(
                message=f"Authentication failed with OpenAI: {str(e)}", code=ErrorCode.UNAUTHENTICATED, details=e.body
            )

        if isinstance(e, openai.PermissionDeniedError):
            logger.error(f"[OpenAI] Permission denied (403): {str(e)}")  # More severe log level
            return LLMPermissionDeniedError(
                message=f"Permission denied by OpenAI: {str(e)}", code=ErrorCode.PERMISSION_DENIED, details=e.body
            )

        if isinstance(e, openai.NotFoundError):
            logger.warning(f"[OpenAI] Resource not found (404): {str(e)}")
            # Could be invalid model name, etc.
            return LLMNotFoundError(message=f"Resource not found in OpenAI: {str(e)}", code=ErrorCode.NOT_FOUND, details=e.body)

        if isinstance(e, openai.UnprocessableEntityError):
            logger.warning(f"[OpenAI] Unprocessable entity (422): {str(e)}")
            return LLMUnprocessableEntityError(
                message=f"Invalid request content for OpenAI: {str(e)}",
                code=ErrorCode.INVALID_ARGUMENT,  # Usually validation errors
                details=e.body,
            )

        # General API error catch-all
        if isinstance(e, openai.APIStatusError):
            logger.warning(f"[OpenAI] API status error ({e.status_code}): {str(e)}")
            # Map based on status code potentially
            if e.status_code >= 500:
                error_cls = LLMServerError
                error_code = ErrorCode.INTERNAL_SERVER_ERROR
            else:
                # Treat other 4xx as bad requests if not caught above
                error_cls = LLMBadRequestError
                error_code = ErrorCode.INVALID_ARGUMENT

            return error_cls(
                message=f"OpenAI API error: {str(e)}",
                code=error_code,
                details={
                    "status_code": e.status_code,
                    "response": str(e.response),
                    "body": e.body,
                },
            )

        # Fallback for unexpected errors
        return super().handle_llm_error(e)
