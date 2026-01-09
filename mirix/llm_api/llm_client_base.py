import logging
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, cast

from mirix.errors import LLMError
from mirix.observability.context import get_trace_context
from mirix.observability.langfuse_client import get_langfuse_client
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.services.cloud_file_mapping_manager import CloudFileMappingManager
from mirix.services.file_manager import FileManager

if TYPE_CHECKING:
    from langfuse.types import TraceContext

# Import for setting AS_ROOT attribute
try:
    from langfuse._client.attributes import LangfuseOtelSpanAttributes
except ImportError:
    LangfuseOtelSpanAttributes = None  # type: ignore


class LLMClientBase:
    """
    Abstract base class for LLM clients, formatting the request objects,
    handling the downstream request and parsing into chat completions response format
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        use_tool_naming: bool = True,
    ):
        self.llm_config = llm_config
        self.use_tool_naming = use_tool_naming
        self.file_manager = FileManager()
        self.cloud_file_mapping_manager = CloudFileMappingManager()
        self.logger = logging.getLogger("Mirix.LLMClientBase")

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
        request_data = self.build_request_data(
            messages,
            self.llm_config,
            tools,
            force_tool_call,
            existing_file_uris=existing_file_uris,
        )

        if get_input_data_for_debugging:
            return request_data

        langfuse = get_langfuse_client()
        trace_context = get_trace_context() if langfuse else {}
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None
        if langfuse and trace_id:
            return self._execute_with_langfuse(
                langfuse, request_data, messages, tools, trace_id, parent_span_id
            )
        else:
            self.logger.debug("Sending LLM request without LangFuse tracing")
            return self._execute_without_langfuse(request_data, messages)

    def _execute_without_langfuse(
        self, request_data: dict, messages: List[Message]
    ) -> ChatCompletionResponse:
        """Execute LLM request without LangFuse tracing."""
        try:
            t1 = time.time()
            response_data = self.request(request_data)
            t2 = time.time()
            self.logger.debug("LLM request time: %.2f seconds", t2 - t1)
        except Exception as e:
            raise self.handle_llm_error(e)

        return self.convert_response_to_chat_completion(response_data, messages)

    def _execute_with_langfuse(
        self,
        langfuse,
        request_data: dict,
        messages: List[Message],
        tools: Optional[List[dict]],
        trace_id: str,
        parent_span_id: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Execute LLM request with LangFuse generation tracing (context manager)."""
        # Transform messages for LangFuse input
        messages_for_trace = []
        for m in messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            # Extract text content from Message.content list
            content_text = ""
            if hasattr(m, "content") and m.content:
                text_parts = []
                for part in m.content:
                    if hasattr(part, "text"):  # TextContent
                        text_parts.append(part.text)
                    elif hasattr(part, "reasoning"):  # ReasoningContent
                        text_parts.append(f"[reasoning]{part.reasoning}")
                content_text = "\n".join(text_parts) if text_parts else str(m.content)

            msg_dict: dict = {"role": role, "content": content_text}

            # Include tool_calls if present (for assistant messages that call tools)
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", None),
                        "function": {
                            "name": (
                                getattr(tc.function, "name", None)
                                if hasattr(tc, "function")
                                else None
                            ),
                            "arguments": (
                                getattr(tc.function, "arguments", None)
                                if hasattr(tc, "function")
                                else None
                            ),
                        },
                    }
                    for tc in m.tool_calls
                ]

            messages_for_trace.append(msg_dict)
        self.logger.debug(f"LLM generation linked to trace_id={trace_id}")

        # Build trace_context with parent_span_id (per Langfuse v3 docs)
        trace_context_dict: dict = {"trace_id": trace_id}
        if parent_span_id:
            trace_context_dict["parent_span_id"] = parent_span_id

        # Prepare input that includes both messages and tools (like OpenAI API format)
        trace_input: dict = {"messages": messages_for_trace}
        if tools:
            trace_input["tools"] = tools

        try:
            # Use context manager for proper OTel context propagation
            with langfuse.start_as_current_observation(
                name="llm_completion",
                as_type="generation",
                trace_context=cast("TraceContext", trace_context_dict),
                model=self.llm_config.model,
                input=trace_input,
                metadata={
                    "provider": self.llm_config.model_endpoint_type,
                    "tools_count": len(tools) if tools else 0,
                },
            ) as generation:
                # Override AS_ROOT to False - this span is a child, not a root
                # The SDK sets AS_ROOT=True when trace_context is provided, but we want proper nesting
                if LangfuseOtelSpanAttributes is not None and hasattr(
                    generation, "_otel_span"
                ):
                    generation._otel_span.set_attribute(
                        LangfuseOtelSpanAttributes.AS_ROOT, False
                    )
                try:
                    t1 = time.time()
                    response_data = self.request(request_data)
                    t2 = time.time()
                    self.logger.debug("LLM request time: %.2f seconds", t2 - t1)
                except Exception as e:
                    generation.update(
                        status_message=str(e),
                        level="ERROR",
                        metadata={"error_type": type(e).__name__},
                    )
                    raise self.handle_llm_error(e)

                chat_completion_data = self.convert_response_to_chat_completion(
                    response_data, messages
                )

                # Update generation with output
                try:
                    output_message = None
                    if (
                        hasattr(chat_completion_data, "choices")
                        and len(chat_completion_data.choices) > 0
                    ):
                        choice = chat_completion_data.choices[0]
                        if hasattr(choice, "message"):
                            msg = choice.message
                            output_message = {
                                "role": getattr(msg, "role", "assistant"),
                                "content": getattr(msg, "content", None),
                            }
                            # Include tool_calls if present (common for agent responses)
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                output_message["tool_calls"] = [
                                    {
                                        "id": getattr(tc, "id", None),
                                        "type": getattr(tc, "type", "function"),
                                        "function": {
                                            "name": (
                                                getattr(tc.function, "name", None)
                                                if hasattr(tc, "function")
                                                else None
                                            ),
                                            "arguments": (
                                                getattr(tc.function, "arguments", None)
                                                if hasattr(tc, "function")
                                                else None
                                            ),
                                        },
                                    }
                                    for tc in msg.tool_calls
                                ]

                    usage_dict = None
                    if (
                        hasattr(chat_completion_data, "usage")
                        and chat_completion_data.usage
                    ):
                        usage_dict = {
                            "input": getattr(
                                chat_completion_data.usage, "prompt_tokens", 0
                            ),
                            "output": getattr(
                                chat_completion_data.usage, "completion_tokens", 0
                            ),
                            "total": getattr(
                                chat_completion_data.usage, "total_tokens", 0
                            ),
                        }

                    generation.update(output=output_message, usage=usage_dict)
                except Exception as update_err:
                    self.logger.warning(f"Failed to update generation: {update_err}")

                return chat_completion_data

        except LLMError:
            # LLM errors should propagate for retry handling
            raise
        except Exception as e:
            # Only catch Langfuse-specific failures, fall back to non-traced execution
            self.logger.warning(f"LangFuse tracing failed, executing without: {e}")
            return self._execute_without_langfuse(request_data, messages)

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
