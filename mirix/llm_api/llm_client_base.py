import time
from abc import abstractmethod
from typing import Dict, List, Optional, Union

from mirix.errors import LLMError
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.services.cloud_file_mapping_manager import CloudFileMappingManager
from mirix.services.file_manager import FileManager
from mirix.llm_api.llm_debug_logger import get_llm_debug_logger

class LLMClientBase:
    """
    Abstract base class for LLM clients, formatting the request objects,
    handling the downstream request and parsing into chat completions response format
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        put_inner_thoughts_first: Optional[bool] = True,
        use_tool_naming: bool = True,
    ):
        self.llm_config = llm_config
        self.put_inner_thoughts_first = put_inner_thoughts_first
        self.use_tool_naming = use_tool_naming
        self.file_manager = FileManager()
        self.cloud_file_mapping_manager = CloudFileMappingManager()

    def send_llm_request(
        self,
        messages: List[Message],
        tools: Optional[List[dict]] = None,
        stream: bool = False,
        force_tool_call: Optional[str] = None,
        get_input_data_for_debugging: bool = False,
        existing_file_uris: Optional[List[str]] = None,
    ) -> ChatCompletionResponse:
        """
        Issues a request to the downstream model endpoint and parses response.
        """
        # 获取调试日志记录器
        debug_logger = get_llm_debug_logger()
        
        request_data = self.build_request_data(messages, self.llm_config, tools, force_tool_call, existing_file_uris=existing_file_uris)

        if get_input_data_for_debugging:
            return request_data

        # 记录请求
        request_id = debug_logger.log_request(
            model_name=self.llm_config.model,
            endpoint=self.llm_config.model_endpoint,
            request_data=request_data,
            additional_info={
                "tools_count": len(tools) if tools else 0,
                "messages_count": len(messages),
                "stream": stream,
                "force_tool_call": force_tool_call
            }
        )

        start_time = time.time()
        
        try:
            response_data = self.request(request_data)
            response_time_ms = (time.time() - start_time) * 1000
            
            # 记录成功响应
            debug_logger.log_response(
                request_id=request_id,
                response_data=response_data,
                response_time_ms=response_time_ms,
                additional_info={
                    "success": True
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            # 记录错误
            debug_logger.log_error(
                request_id=request_id,
                error=e,
                error_context={
                    "response_time_ms": response_time_ms,
                    "model_name": self.llm_config.model,
                    "endpoint": self.llm_config.model_endpoint
                }
            )
            raise self.handle_llm_error(e)

        try:
            chat_completion_data = self.convert_response_to_chat_completion(response_data, messages)
            return chat_completion_data
        except Exception as e:
            # 记录响应转换错误
            debug_logger.log_error(
                request_id=request_id,
                error=e,
                error_context={
                    "stage": "response_conversion",
                    "response_data_keys": list(response_data.keys()) if isinstance(response_data, dict) else "not_dict"
                }
            )
            raise

    @abstractmethod
    def build_request_data(
        self,
        messages: List[Message],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        """
        Constructs a request object in the expected data format for this client.
        """
        raise NotImplementedError

    @abstractmethod
    def request(self, request_data: dict) -> dict:
        """
        Performs underlying request to llm and returns raw response.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[Message],
    ) -> ChatCompletionResponse:
        """
        Converts custom response format from llm client into an OpenAI
        ChatCompletionsResponse object.
        """
        raise NotImplementedError

    @abstractmethod
    def handle_llm_error(self, e: Exception) -> Exception:
        """
        Maps provider-specific errors to common LLMError types.
        Each LLM provider should implement this to translate their specific errors.

        Args:
            e: The original provider-specific exception

        Returns:
            An LLMError subclass that represents the error in a provider-agnostic way
        """
        return LLMError(f"Unhandled LLM error: {str(e)}") 