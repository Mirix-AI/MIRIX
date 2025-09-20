import os
import json
import uuid
from typing import List, Optional, Tuple

import requests
from google.genai.types import FunctionCallingConfig, FunctionCallingConfigMode, ToolConfig

from mirix.constants import NON_USER_MSG_PREFIX
from mirix.helpers.datetime_helpers import get_utc_time
from mirix.helpers.json_helpers import json_dumps
from mirix.llm_api.helpers import make_post_request
from mirix.llm_api.llm_client_base import LLMClientBase
from mirix.utils import clean_json_string_extra_backslash, count_tokens
from mirix.log import get_logger
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message as PydanticMessage
from mirix.schemas.openai.chat_completion_request import Tool
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice, FunctionCall, Message, ToolCall, UsageStatistics
from mirix.settings import model_settings
from mirix.services.provider_manager import ProviderManager
from mirix.utils import get_tool_call_id

logger = get_logger(__name__)


class GoogleAIClient(LLMClientBase):

    def request(self, request_data: dict) -> dict:
        """
        Performs underlying request to llm and returns raw response.
        """
        # print("[google_ai request]", json.dumps(request_data, indent=2))

        # Check for database-stored API key first, fall back to model_settings
        override_key = ProviderManager().get_gemini_override_key()
        api_key = str(override_key) if override_key else str(model_settings.gemini_api_key)

        url, headers = get_gemini_endpoint_and_headers(
            base_url=str(self.llm_config.model_endpoint),
            model=self.llm_config.model,
            api_key=api_key,
            key_in_header=True,
            generate_content=True,
        )
        return make_post_request(url, headers, request_data)

    def build_request_data(
        self,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: List[dict],
        force_tool_call: Optional[str] = None,
        existing_file_uris: Optional[List[str]] = None,
    ) -> dict:
        """
        Constructs a request object in the expected data format for this client.
        """
        if tools:
            tools = [{"type": "function", "function": f} for f in tools]
            tool_objs = [Tool(**t) for t in tools]
            tool_names = [t.function.name for t in tool_objs]
            # Convert to the exact payload style Google expects
            tools = self.convert_tools_to_google_ai_format(tool_objs)
        else:
            tool_names = []

        contents = self.add_dummy_model_messages(
            [m.to_google_ai_dict() for m in messages],
        )

        generation_config = {
            "temperature": llm_config.temperature,
            "max_output_tokens": llm_config.max_tokens,
        }
        
        # Only add thinkingConfig for models that support it
        # gemini-2.0-flash and gemini-1.5-pro don't support thinking
        if not ("2.0" in llm_config.model or "1.5" in llm_config.model):
            generation_config['thinkingConfig'] = {'thinkingBudget': 1024}  # TODO: put into llm_config
        
        contents = self.combine_tool_responses(contents)

        request_data = {
            "contents": self.fill_image_content_in_messages(contents, existing_file_uris=existing_file_uris),
            "tools": tools,
            "generation_config": generation_config,
        }

        # write tool config
        tool_config = ToolConfig(
            function_calling_config=FunctionCallingConfig(
                # ANY mode forces the model to predict only function calls
                mode=FunctionCallingConfigMode.ANY,
                # Provide the list of tools (though empty should also work, it seems not to)
                allowed_function_names=tool_names,
            )
        )
        request_data["tool_config"] = tool_config.model_dump()
        return request_data

    def combine_tool_responses(self, contents: List[dict]) -> List[dict]:
        idx = 0
        new_contents = []
        while idx < len(contents):
            message = contents[idx]
            new_contents.append(message)

            if message['role'] == 'model':
                is_multiple_tool_call = False
                if len(message['parts']) > 1:
                    is_multiple_tool_call = True
                    for part in message['parts']:
                        if not 'functionCall' in part:
                            is_multiple_tool_call = False

                if is_multiple_tool_call:
                    # need to combine the next len(message['parts]) into one:
                    messages_for_function_responses = contents[idx+1:idx+len(message['parts'])+1]
                    new_contents.append(
                        {
                            'role': 'function',
                            'parts': []
                        }
                    )
                    for msg_for_func_res in messages_for_function_responses:
                        assert msg_for_func_res['role'] == 'function' and "functionResponse" in msg_for_func_res['parts'][0]
                        new_contents[-1]['parts'].extend(msg_for_func_res['parts'])
                    
                    idx += len(message['parts'])
            idx += 1
        return new_contents

    def fill_image_content_in_messages(self, google_ai_message_list, existing_file_uris: Optional[List[str]] = None):
        """
        Converts image URIs in the message to base64 format.
        """
        from mirix.constants import LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY

        global_image_idx = 0
        new_message_list = []

        image_content_loaded = False  # it will always be false if `LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY` is False

        for message_idx, message in enumerate(google_ai_message_list[::-1]):

            if message['role'] != 'user':
                new_message_list.append(message)
                continue
                
            if isinstance(message['parts'], str):
                message['parts'] = [{'text': message['parts']}]
                new_message_list.append(message)
                continue
            
            assert isinstance(message['parts'], list), f"Expected list of parts, got {type(message['parts'])}"
            
            has_image = False
            message_parts = []
            for part in message['parts']:
                if 'text' in part:
                    message_parts.append(part)
                elif 'image_id' in part:
                    if LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY and image_content_loaded:
                        message_parts.append({
                            'text': "[System Message] There was an image here but now the image has been deleted to save space."
                        })
                    else:
                        message_parts.append({'text': f"<image {global_image_idx}>"})
                        file = self.file_manager.get_file_metadata_by_id(part['image_id'])
                        if file.source_url is not None:
                            # For Google AI, we need to convert URL to base64
                            import requests
                            response = requests.get(file.source_url)
                            import base64
                            base64_data = base64.b64encode(response.content).decode('utf-8')
                            # Determine mime type from URL or default to jpeg
                            mime_type = file.file_type
                            message_parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data,
                                }
                            })
                        elif file.file_path is not None:
                            # Read from file path and convert to base64
                            import base64
                            mime_type = file.file_type
                            with open(file.file_path, "rb") as img_file:
                                base64_data = base64.b64encode(img_file.read()).decode("utf-8")
                            message_parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data,
                                }
                            })
                        else:
                            raise ValueError(f"File with id {part['image_id']} has neither source_url nor file_path")
                        global_image_idx += 1
                        has_image = True
                elif 'cloud_file_uri' in part:
                    file = self.file_manager.get_file_metadata_by_id(part['cloud_file_uri'])
                    if existing_file_uris is not None and file.google_cloud_url not in existing_file_uris:
                        message_parts.append({
                            'text': f"[System Message] There was an image here but now the image has been deleted to save space."
                        })
                    else:
                        message_parts.append({
                            "file_data": {
                                "mime_type": file.file_type,
                                "file_uri": file.google_cloud_url,
                            }
                        })
                else:
                    raise ValueError(f"Unknown part type in message: {part}")
            message['parts'] = message_parts
            new_message_list.append(message)

            if has_image:
                if LOAD_IMAGE_CONTENT_FOR_LAST_MESSAGE_ONLY:
                    # Load image content for the last message only.
                    image_content_loaded = True

        new_message_list = new_message_list[::-1]

        return new_message_list

    def convert_response_to_chat_completion(
        self,
        response_data: dict,
        input_messages: List[PydanticMessage],
    ) -> ChatCompletionResponse:
        """
        Converts custom response format from llm client into an OpenAI
        ChatCompletionsResponse object.

        Example Input:
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": " OK. Barbie is showing in two theaters in Mountain View, CA: AMC Mountain View 16 and Regal Edwards 14."
                            }
                        ]
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 9,
                "candidatesTokenCount": 27,
                "totalTokenCount": 36
            }
        }
        """
        # print("[google_ai response]", json.dumps(response_data, indent=2))

        try:
            choices = []
            index = 0
            for candidate in response_data["candidates"]:
                content = candidate["content"]

                role = content["role"]
                assert role == "model", f"Unknown role in response: {role}"

                parts = content["parts"]

                # NOTE: we aren't properly supported multi-parts here anyways (we're just appending choices),
                #       so let's disable it for now

                # NOTE(Apr 9, 2025): there's a very strange bug on 2.5 where the response has a part with broken text
                # {'candidates': [{'content': {'parts': [{'functionCall': {'name': 'send_message', 'args': {'request_contine_chaining': False, 'message': 'Hello! How can I make your day better?', 'inner_thoughts': 'User has initiated contact. Sending a greeting.'}}}], 'role': 'model'}, 'finishReason': 'STOP', 'avgLogprobs': -0.25891534213362066}], 'usageMetadata': {'promptTokenCount': 2493, 'candidatesTokenCount': 29, 'totalTokenCount': 2522, 'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 2493}], 'candidatesTokensDetails': [{'modality': 'TEXT', 'tokenCount': 29}]}, 'modelVersion': 'gemini-1.5-pro-002'}
                # To patch this, if we have multiple parts we can take the last one

                # NOTE(May 9, 2025 from Yu) I found that sometimes when the respones have multiple parts, they are actually multiple function calls. In my experiments, gemini-2.5-flash can call two `archival_memory_insert` in one output.
                # if len(parts) > 1:
                #     logger.warning(f"Unexpected multiple parts in response from Google AI: {parts}")
                #     parts = [parts[-1]]

                # TODO support parts / multimodal
                # TODO support parallel tool calling natively
                # TODO Alternative here is to throw away everything else except for the first part
                for response_message in parts:
                    # Convert the actual message style to OpenAI style
                    if "functionCall" in response_message and response_message["functionCall"] is not None:
                        function_call = response_message["functionCall"]
                        assert isinstance(function_call, dict), function_call
                        function_name = function_call["name"]
                        assert isinstance(function_name, str), function_name
                        function_args = function_call["args"]
                        assert isinstance(function_args, dict), function_args

                        # NOTE: this also involves stripping the inner monologue out of the function
                        if self.llm_config.put_inner_thoughts_in_kwargs:
                            from mirix.constants import INNER_THOUGHTS_KWARG

                            assert INNER_THOUGHTS_KWARG in function_args, f"Couldn't find inner thoughts in function args:\n{function_call}"
                            inner_thoughts = function_args.pop(INNER_THOUGHTS_KWARG)
                            assert inner_thoughts is not None, f"Expected non-null inner thoughts function arg:\n{function_call}"
                        else:
                            inner_thoughts = None

                        # Google AI API doesn't generate tool call IDs
                        openai_response_message = Message(
                            role="assistant",  # NOTE: "model" -> "assistant"
                            content=inner_thoughts,
                            tool_calls=[
                                ToolCall(
                                    id=get_tool_call_id(),
                                    type="function",
                                    function=FunctionCall(
                                        name=function_name,
                                        arguments=clean_json_string_extra_backslash(json_dumps(function_args)),
                                    ),
                                )
                            ],
                        )

                    else:

                        # Inner thoughts are the content by default
                        inner_thoughts = response_message["text"]

                        # Google AI API doesn't generate tool call IDs
                        openai_response_message = Message(
                            role="assistant",  # NOTE: "model" -> "assistant"
                            content=inner_thoughts,
                        )

                    # Google AI API uses different finish reason strings than OpenAI
                    # OpenAI: 'stop', 'length', 'function_call', 'content_filter', null
                    #   see: https://platform.openai.com/docs/guides/text-generation/chat-completions-api
                    # Google AI API: FINISH_REASON_UNSPECIFIED, STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
                    #   see: https://ai.google.dev/api/python/google/ai/generativelanguage/Candidate/FinishReason
                    finish_reason = candidate["finishReason"]
                    if finish_reason == "STOP":
                        openai_finish_reason = (
                            "function_call"
                            if openai_response_message.tool_calls is not None and len(openai_response_message.tool_calls) > 0
                            else "stop"
                        )
                    elif finish_reason == "MAX_TOKENS":
                        openai_finish_reason = "length"
                    elif finish_reason == "SAFETY":
                        openai_finish_reason = "content_filter"
                    elif finish_reason == "RECITATION":
                        openai_finish_reason = "content_filter"
                    else:
                        raise ValueError(f"Unrecognized finish reason in Google AI response: {finish_reason}")

                    choices.append(
                        Choice(
                            finish_reason=openai_finish_reason,
                            index=index,
                            message=openai_response_message,
                        )
                    )
                    index += 1

            # if len(choices) > 1:
            #     raise UserWarning(f"Unexpected number of candidates in response (expected 1, got {len(choices)})")

            # NOTE: some of the Google AI APIs show UsageMetadata in the response, but it seems to not exist?
            #  "usageMetadata": {
            #     "promptTokenCount": 9,
            #     "candidatesTokenCount": 27,
            #     "totalTokenCount": 36
            #   }
            if "usageMetadata" in response_data:
                usage_data = response_data["usageMetadata"]
                if "promptTokenCount" not in usage_data:
                    raise ValueError(f"promptTokenCount not found in usageMetadata:\n{json.dumps(usage_data, indent=2)}")
                if "totalTokenCount" not in usage_data:
                    raise ValueError(f"totalTokenCount not found in usageMetadata:\n{json.dumps(usage_data, indent=2)}")
                if "candidatesTokenCount" not in usage_data:
                    raise ValueError(f"candidatesTokenCount not found in usageMetadata:\n{json.dumps(usage_data, indent=2)}")

                prompt_tokens = usage_data["promptTokenCount"]
                completion_tokens = usage_data["candidatesTokenCount"]
                total_tokens = usage_data["totalTokenCount"]

                usage = UsageStatistics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            else:
                # Count it ourselves
                assert input_messages is not None, f"Didn't get UsageMetadata from the API response, so input_messages is required"
                prompt_tokens = count_tokens(json_dumps(input_messages))  # NOTE: this is a very rough approximation
                completion_tokens = count_tokens(json_dumps(openai_response_message.model_dump()))  # NOTE: this is also approximate
                total_tokens = prompt_tokens + completion_tokens
                usage = UsageStatistics(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

            response_id = str(uuid.uuid4())
            return ChatCompletionResponse(
                id=response_id,
                choices=choices,
                model=self.llm_config.model,  # NOTE: Google API doesn't pass back model in the response
                created=get_utc_time(),
                usage=usage,
            )
        except KeyError as e:
            raise e

    def convert_tools_to_google_ai_format(self, tools: List[Tool]) -> List[dict]:
        """
        OpenAI style:
        "tools": [{
            "type": "function",
            "function": {
                "name": "find_movies",
                "description": "find ....",
                "parameters": {
                "type": "object",
                "properties": {
                    PARAM: {
                    "type": PARAM_TYPE,  # eg "string"
                    "description": PARAM_DESCRIPTION,
                    },
                    ...
                },
                "required": List[str],
                }
            }
        }
        ]

        Google AI style:
        "tools": [{
            "functionDeclarations": [{
            "name": "find_movies",
            "description": "find movie titles currently playing in theaters based on any description, genre, title words, etc.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state, e.g. San Francisco, CA or a zip code e.g. 95616"
                },
                "description": {
                    "type": "STRING",
                    "description": "Any kind of description including category or genre, title words, attributes, etc."
                }
                },
                "required": ["description"]
            }
            }, {
            "name": "find_theaters",
            ...
        """
        function_list = [
            dict(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,  # TODO need to unpack
            )
            for t in tools
        ]

        # Add inner thoughts if needed
        for func in function_list:
            # Note: Google AI API used to have weird casing requirements, but not any more

            # Add inner thoughts
            if self.llm_config.put_inner_thoughts_in_kwargs:
                from mirix.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION

                func["parameters"]["properties"][INNER_THOUGHTS_KWARG] = {
                    "type": "string",
                    "description": INNER_THOUGHTS_KWARG_DESCRIPTION,
                }
                func["parameters"]["required"].append(INNER_THOUGHTS_KWARG)

        return [{"functionDeclarations": function_list}]

    def add_dummy_model_messages(self, messages: List[dict]) -> List[dict]:
        """Google AI API requires all function call returns are immediately followed by a 'model' role message.

        In Letta, the 'model' will often call a function (e.g. send_message) that itself yields to the user,
        so there is no natural follow-up 'model' role message.

        To satisfy the Google AI API restrictions, we can add a dummy 'yield' message
        with role == 'model' that is placed in-betweeen and function output
        (role == 'tool') and user message (role == 'user').
        """
        dummy_yield_message = {
            "role": "model",
            "parts": [{"text": f"{NON_USER_MSG_PREFIX}Function call returned, waiting for user response."}],
        }
        messages_with_padding = []
        for i, message in enumerate(messages):
            messages_with_padding.append(message)
            # Check if the current message role is 'tool' and the next message role is 'user'
            if message["role"] in ["tool", "function"] and (i + 1 < len(messages) and messages[i + 1]["role"] == "user"):
                messages_with_padding.append(dummy_yield_message)

        return messages_with_padding


def get_gemini_endpoint_and_headers(
    base_url: str, model: Optional[str], api_key: str, key_in_header: bool = True, generate_content: bool = False
) -> Tuple[str, dict]:
    """
    Dynamically generate the model endpoint and headers.
    """
    url = f"{base_url}/v1beta/models"

    # Add the model
    if model is not None:
        url += f"/{model}"

    # Add extension for generating content if we're hitting the LM
    if generate_content:
        url += ":generateContent"

    # Decide if api key should be in header or not
    # Two ways to pass the key: https://ai.google.dev/tutorials/setup
    if key_in_header:
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        url += f"?key={api_key}"
        headers = {"Content-Type": "application/json"}

    return url, headers


def google_ai_get_model_list(base_url: str, api_key: str, key_in_header: bool = True) -> List[dict]:
    from mirix.utils import printd

    url, headers = get_gemini_endpoint_and_headers(base_url, None, api_key, key_in_header)

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string

        # Grab the models out
        model_list = response["models"]
        return model_list

    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        # Print the HTTP status code
        print(f"HTTP Error: {http_err.response.status_code}")
        # Print the response content (error message from server)
        print(f"Message: {http_err.response.text}")
        raise http_err

    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err

    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def google_ai_get_model_details(base_url: str, api_key: str, model: str, key_in_header: bool = True) -> List[dict]:
    from mirix.utils import printd

    url, headers = get_gemini_endpoint_and_headers(base_url, model, api_key, key_in_header)

    try:
        response = requests.get(url, headers=headers)
        printd(f"response = {response}")
        response.raise_for_status()  # Raises HTTPError for 4XX/5XX status
        response = response.json()  # convert to dict from string
        printd(f"response.json = {response}")

        # Grab the models out
        return response

    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (e.g., response 4XX, 5XX)
        printd(f"Got HTTPError, exception={http_err}")
        # Print the HTTP status code
        print(f"HTTP Error: {http_err.response.status_code}")
        # Print the response content (error message from server)
        print(f"Message: {http_err.response.text}")
        raise http_err

    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err

    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def google_ai_get_model_context_window(base_url: str, api_key: str, model: str, key_in_header: bool = True) -> int:
    model_details = google_ai_get_model_details(base_url=base_url, api_key=api_key, model=model, key_in_header=key_in_header)
    # TODO should this be:
    # return model_details["inputTokenLimit"] + model_details["outputTokenLimit"]
    return int(model_details["inputTokenLimit"])
