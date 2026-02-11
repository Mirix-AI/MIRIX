import uuid
from typing import List, Optional, Tuple

import requests

from mirix.constants import MAX_IMAGES_TO_PROCESS, NON_USER_MSG_PREFIX
from mirix.llm_api.helpers import make_post_request
from mirix.log import get_logger
from mirix.schemas.openai.chat_completion_request import Tool
from mirix.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageStatistics,
)
from mirix.utils import (
    clean_json_string_extra_backslash,
    count_tokens,
    get_tool_call_id,
    get_utc_time,
    json_dumps,
)

logger = get_logger(__name__)


def get_gemini_endpoint_and_headers(
    base_url: str,
    model: Optional[str],
    api_key: str,
    key_in_header: bool = True,
    generate_content: bool = False,
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
        logger.debug("HTTP Error: %s", http_err.response.status_code)
        # Print the response content (error message from server)
        logger.debug("Message: %s", http_err.response.text)
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
    model_details = google_ai_get_model_details(
        base_url=base_url, api_key=api_key, model=model, key_in_header=key_in_header
    )
    # TODO should this be:
    # return model_details["inputTokenLimit"] + model_details["outputTokenLimit"]
    return int(model_details["inputTokenLimit"])


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
        logger.debug("HTTP Error: %s", http_err.response.status_code)
        # Print the response content (error message from server)
        logger.debug("Message: %s", http_err.response.text)
        raise http_err

    except requests.exceptions.RequestException as req_err:
        # Handle other requests-related errors (e.g., connection error)
        printd(f"Got RequestException, exception={req_err}")
        raise req_err

    except Exception as e:
        # Handle other potential errors
        printd(f"Got unknown Exception, exception={e}")
        raise e


def add_dummy_model_messages(messages: List[dict]) -> List[dict]:
    """Google AI API requires all function call returns are immediately followed by a 'model' role message.

    In Mirix, the 'model' will often call a function (e.g. send_message) that itself yields to the user,
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


# TODO use pydantic model as input
def to_google_ai(openai_message_dict: dict) -> dict:
    # TODO supports "parts" as part of multimodal support
    assert not isinstance(openai_message_dict["content"], list), "Multi-part content is message not yet supported"
    if openai_message_dict["role"] == "user":
        google_ai_message_dict = {
            "role": "user",
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "assistant":
        google_ai_message_dict = {
            "role": "model",  # NOTE: diff
            "parts": [{"text": openai_message_dict["content"]}],
        }
    elif openai_message_dict["role"] == "tool":
        google_ai_message_dict = {
            "role": "function",  # NOTE: diff
            "parts": [{"text": openai_message_dict["content"]}],
        }
    else:
        raise ValueError(f"Unsupported conversion (OpenAI -> Google AI) from role {openai_message_dict['role']}")
    return google_ai_message_dict


# TODO convert return type to pydantic
def convert_tools_to_google_ai_format(tools: List[Tool], inner_thoughts_in_kwargs: Optional[bool] = True) -> List[dict]:
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

    # Correct casing + add inner thoughts if needed
    for func in function_list:
        func["parameters"]["type"] = "OBJECT"
        for param_name, param_fields in func["parameters"]["properties"].items():
            param_fields["type"] = param_fields["type"].upper()
    return [{"functionDeclarations": function_list}]


def convert_google_ai_response_to_chatcompletion(
    response_json: dict,  # REST response from Google AI API
    model: str,  # Required since not returned
    input_messages: Optional[List[dict]] = None,  # Required if the API doesn't return UsageMetadata
) -> ChatCompletionResponse:
    """Google AI API response format is not the same as ChatCompletion, requires unpacking

    Example:
    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                'thought': True, 'thoughtSignature': 'AVSoXO6Gl3O8QoOMyK1PI5LF9bTeN9v8iohxLr7UVHAlzQ4Ekhg3Exk6/tCeYHADfcVT6+nXThDXHWBii7bsbCvpyI30LdaG66i0DEYzP1vRJi1R7xGqFntgUCkahG5wVgr3t7D+umGzCERF83N+Xc1TVC1eocsfuc4dHuzccPNFp2MwsxtwPcdXUoyHbmtQ8j8S4mmme8j74b+KhhV0KV+YnWjr8VY0Sk7n1966LUo5TPaRgUfEzTikDUoGh3wZBXs0UzPgWYy2AxcZN7h/ZgLBOEdsuOkzs6RPrTpnAst/L38tmmf0HKracEWjtxPMDdPjTVnH5TuFyV58hsXUyNDZJ3lLGP7eSQQmPX/fqn6f6Bv+V6We/GdNvDaQRbdOzR0MmJE7YATEFT1BAJsi1BA/Gu8tCvLOIGu/Uty69bgNDyBDf8Sdqpp3yTSTDyHQawCIgk+P8dApEPkmi/G4dwnxZ175NmWpJpDU81ZUYq0rPA2Ijdg4vWOgce9YJhKQjjFTqYcWe+8X29hhSA9hwC9ewv4POPDZRbsFCU9I6tUuAMn4hxW2XB231AA87AJMvn6Av8MAcdf28tYAWbO/CQWakuhERKeZN2jeg6uhjTkmmT56JN4KqtF15u/sROI6/DLALUcX16jVC8O6o4hynGUdLzOoO7W8Kc/a2LwYJJeraDsynhyMqipwdIS2aXHu+gMFpXl2FkIZxLtVbPg177GAvosV+ZyWaV6FWorGGsnq+eR9E5r5syz6zXsbk93lzqqteLA0WWtew6dXDlDNaJ5gCbIVnKwLzxYZ7xxhk4Ko+5SSN8cKN0odbeC+RdQXkrs='
              },
              {
                "text": " OK. Barbie is showing in two theaters in Mountain View, CA: AMC Mountain View 16 and Regal Edwards 14."
              },
              {
                'functionCall': {'name': 'update_topic', 'args': {'topic': 'greeting'}}
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
    try:
        choices = []
        index = 0
        for candidate in response_json["candidates"]:
            content = candidate["content"]

            role = content["role"]
            assert role == "model", f"Unknown role in response: {role}"

            parts = content["parts"]
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

                    # Google AI API doesn't generate tool call IDs
                    openai_response_message = Message(
                        role="assistant",  # NOTE: "model" -> "assistant"
                        content=None,
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

                elif "thought" in response_message:
                    continue

                else:
                    # Google AI API doesn't generate tool call IDs
                    openai_response_message = Message(
                        role="assistant",  # NOTE: "model" -> "assistant"
                        content=response_message["text"],
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
                        if openai_response_message.tool_calls is not None
                        and len(openai_response_message.tool_calls) > 0
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
        if "usageMetadata" in response_json:
            usage = UsageStatistics(
                prompt_tokens=response_json["usageMetadata"]["promptTokenCount"],
                completion_tokens=response_json["usageMetadata"]["candidatesTokenCount"],
                total_tokens=response_json["usageMetadata"]["totalTokenCount"],
            )
        else:
            # Count it ourselves
            assert (
                input_messages is not None
            ), "Didn't get UsageMetadata from the API response, so input_messages is required"
            prompt_tokens = count_tokens(json_dumps(input_messages))  # NOTE: this is a very rough approximation
            completion_tokens = count_tokens(
                json_dumps(openai_response_message.model_dump())
            )  # NOTE: this is also approximate
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
            model=model,  # NOTE: Google API doesn't pass back model in the response
            created=get_utc_time(),
            usage=usage,
        )
    except KeyError as e:
        raise e


def extract_content(content):
    import re

    pattern = r"([^<]+)?(?:<image>\s*(.*?)\s*</image>)?"
    matches = re.findall(pattern, content)

    result = []
    for text, image in matches:
        if text and text.strip():
            result.append({"type": "text", "text": text.strip()})
        if image and image.strip():
            result.append({"type": "image_url", "image_url": image.strip()})

    return result


# TODO convert 'data' type to pydantic
def google_ai_chat_completions_request(
    base_url: str,
    model: str,
    api_key: str,
    data: dict,
    key_in_header: bool = True,
    add_postfunc_model_messages: bool = True,
    # NOTE: Google AI API doesn't support mixing parts 'text' and 'function',
    # so there's no clean way to put inner thoughts in the same message as a function call
    inner_thoughts_in_kwargs: bool = True,
    image_uris: Optional[List[str]] = None,
    get_input_data_for_debugging: bool = False,
) -> ChatCompletionResponse:
    """https://ai.google.dev/docs/function_calling

    From https://ai.google.dev/api/rest#service-endpoint:
    "A service endpoint is a base URL that specifies the network address of an API service.
    One service might have multiple service endpoints.
    This service has the following service endpoint and all URIs below are relative to this service endpoint:
    https://xxx.googleapis.com
    """

    assert api_key is not None, "Missing api_key when calling Google AI"

    url, headers = get_gemini_endpoint_and_headers(base_url, model, api_key, key_in_header, generate_content=True)

    # data["contents"][-1]["role"] = "model"
    if add_postfunc_model_messages:
        data["contents"] = add_dummy_model_messages(data["contents"])

    num_processed_images = 0

    for contents in data["contents"][::-1]:
        if contents["role"] == "user":
            assert len(contents["parts"]) == 1
            part = contents["parts"][0]

            if "</image>" in part["text"] and "<image>" in part["text"]:
                new_parts = []

                # for item in extract_content(json.loads(part['text'])['message']):
                for item in extract_content(part["text"]):
                    if item["type"] == "text":
                        new_parts.append({"text": item["text"]})

                    elif item["type"] == "image_url":
                        if (
                            image_uris is not None
                            and "existing_image_uris" in image_uris
                            and item["image_url"] in image_uris["existing_image_uris"]
                        ):
                            new_parts.append(
                                {
                                    "file_data": {
                                        "mime_type": "image/png",
                                        "file_uri": item["image_url"],
                                    }
                                }
                            )
                            num_processed_images += 1
                        else:
                            new_parts.append(
                                {
                                    "text": "[This image has been deleted (maybe because it is either repeated or unimportant)]"
                                }
                            )

                contents["parts"] = new_parts

                if num_processed_images >= MAX_IMAGES_TO_PROCESS:
                    break

    if get_input_data_for_debugging:
        return data

    import time

    s1 = time.time()
    response_json = make_post_request(url, headers, data)
    s2 = time.time()
    logger.debug("Query Takes time:", s2 - s1)

    if len(response_json["candidates"][0]["content"]) == 0:
        raise ValueError("Empty response from Google AI API")

    return convert_google_ai_response_to_chatcompletion(
        response_json=response_json,
        model=data.get("model"),
        input_messages=data["contents"],
    )
