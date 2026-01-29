import copy
import json
import logging
import warnings
from collections import OrderedDict
from typing import Any, List, Union

import requests

from mirix.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice
from mirix.settings import summarizer_settings
from mirix.utils import count_tokens, json_dumps, printd

logger = logging.getLogger(__name__)


def _convert_to_structured_output_helper(property: dict) -> dict:
    """Convert a single JSON schema property to structured output format (recursive)"""

    if "type" not in property:
        raise ValueError(f"Property {property} is missing a type")
    param_type = property["type"]

    if "description" not in property:
        # raise ValueError(f"Property {property} is missing a description")
        param_description = None
    else:
        param_description = property["description"]

    if param_type == "object":
        if "properties" not in property:
            raise ValueError(f"Property {property} of type object is missing properties")
        properties = property["properties"]
        property_dict = {
            "type": "object",
            "properties": {k: _convert_to_structured_output_helper(v) for k, v in properties.items()},
            "additionalProperties": False,
            "required": list(properties.keys()),
        }
        if param_description is not None:
            property_dict["description"] = param_description
        return property_dict

    elif param_type == "array":
        if "items" not in property:
            raise ValueError(f"Property {property} of type array is missing items")
        items = property["items"]
        property_dict = {
            "type": "array",
            "items": _convert_to_structured_output_helper(items),
        }
        if param_description is not None:
            property_dict["description"] = param_description
        return property_dict

    else:
        property_dict = {
            "type": param_type,  # simple type
        }
        if param_description is not None:
            property_dict["description"] = param_description
        return property_dict


def convert_to_structured_output(openai_function: dict, allow_optional: bool = False) -> dict:
    """Convert function call objects to structured output objects

    See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas
    """
    description = openai_function["description"] if "description" in openai_function else ""

    structured_output = {
        "name": openai_function["name"],
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            "required": [],
        },
    }

    # This code needs to be able to handle nested properties
    # For example, the param details may have "type" + "description",
    # but if "type" is "object" we expected "properties", where each property has details
    # and if "type" is "array" we expect "items": <type>
    for param, details in openai_function["parameters"]["properties"].items():
        param_type = details["type"]
        description = details["description"]

        if param_type == "object":
            if "properties" not in details:
                # Structured outputs requires the properties on dicts be specified ahead of time
                raise ValueError(f"Property {param} of type object is missing properties")
            structured_output["parameters"]["properties"][param] = {
                "type": "object",
                "description": description,
                "properties": {k: _convert_to_structured_output_helper(v) for k, v in details["properties"].items()},
                "additionalProperties": False,
                "required": list(details["properties"].keys()),
            }

        elif param_type == "array":
            structured_output["parameters"]["properties"][param] = {
                "type": "array",
                "description": description,
                "items": _convert_to_structured_output_helper(details["items"]),
            }

        else:
            structured_output["parameters"]["properties"][param] = {
                "type": param_type,  # simple type
                "description": description,
            }

        if "enum" in details:
            structured_output["parameters"]["properties"][param]["enum"] = details["enum"]

    if not allow_optional:
        # Add all properties to required list
        structured_output["parameters"]["required"] = list(structured_output["parameters"]["properties"].keys())

    else:
        # See what parameters exist that aren't required
        # Those are implied "optional" types
        # For those types, turn each of them into a union type with "null"
        # e.g.
        # "type": "string" -> "type": ["string", "null"]
        # TODO
        raise NotImplementedError

    return structured_output


def make_post_request(url: str, headers: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    printd(f"Sending request to {url}")
    try:
        response = requests.post(url, headers=headers, json=data)
        printd(f"Response status code: {response.status_code}")

        # Raise for 4XX/5XX HTTP errors
        response.raise_for_status()

        # Check if the response content type indicates JSON and attempt to parse it
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            try:
                response_data = response.json()  # Attempt to parse the response as JSON
                printd(f"Response JSON: {response_data}")
            except ValueError as json_err:
                # Handle the case where the content type says JSON but the body is invalid
                error_message = f"Failed to parse JSON despite Content-Type being {content_type}: {json_err}"
                printd(error_message)
                raise ValueError(error_message) from json_err
        else:
            error_message = f"Unexpected content type returned: {response.headers.get('Content-Type')}"
            printd(error_message)
            raise ValueError(error_message)
        # Process the response using the callback function
        return response_data

    except requests.exceptions.HTTPError as http_err:
        # HTTP errors (4XX, 5XX)
        error_message = f"HTTP error occurred: {http_err}"
        if http_err.response is not None:
            error_message += f" | Status code: {http_err.response.status_code}, Message: {http_err.response.text}"
        printd(error_message)
        raise requests.exceptions.HTTPError(error_message) from http_err

    except requests.exceptions.Timeout as timeout_err:
        # Handle timeout errors
        error_message = f"Request timed out: {timeout_err}"
        printd(error_message)
        raise requests.exceptions.Timeout(error_message) from timeout_err

    except requests.exceptions.RequestException as req_err:
        # Non-HTTP errors (e.g., connection, SSL errors)
        error_message = f"Request failed: {req_err}"
        printd(error_message)
        raise requests.exceptions.RequestException(error_message) from req_err

    except ValueError as val_err:
        # Handle content-type or non-JSON response issues
        error_message = f"ValueError: {val_err}"
        printd(error_message)
        raise ValueError(error_message) from val_err

    except Exception as e:
        # Catch any other unknown exceptions
        error_message = f"An unexpected error occurred: {e}"
        printd(error_message)
        raise Exception(error_message) from e


def calculate_summarizer_cutoff(
    in_context_messages: List[Message],
    token_counts: List[int],
    logger: "logging.Logger",
) -> int:
    if len(in_context_messages) != len(token_counts):
        raise ValueError(
            f"Given in_context_messages has different length from given token_counts: {len(in_context_messages)} != {len(token_counts)}"
        )

    in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]

    if summarizer_settings.evict_all_messages:
        logger.debug("Evicting all messages...")
        return len(in_context_messages)
    else:
        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # We do the inverse of `desired_memory_token_pressure` to get what we need to remove
        desired_token_count_to_summarize = int(
            sum(token_counts) * (1 - summarizer_settings.desired_memory_token_pressure)
        )
        logger.debug(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")

        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(in_context_messages_openai):
            # Skip system
            if i == 0:
                continue
            cutoff = i
            tokens_so_far += token_counts[i]

            if msg["role"] not in ["user", "tool", "function"] and tokens_so_far >= desired_token_count_to_summarize:
                # The intent of this code is to break on an assistant message boundary,
                # so that we don't summarize in the middle of a back and forth turn.
                # Break if the role is NOT a user or tool/function and tokens_so_far is enough
                break
            elif len(in_context_messages) - cutoff - 1 <= summarizer_settings.keep_last_n_messages:
                # Also break if we reached the `keep_last_n_messages` threshold
                # NOTE: This may be on a user, tool, or function in theory
                logger.warning(
                    f"Breaking summary cutoff early on role={msg['role']} because we hit the `keep_last_n_messages`={summarizer_settings.keep_last_n_messages}"
                )
                break
        # If the next message is a tool call result, then include it in the set of messages to summarize as well.
        # The intent of this code is so that tool calls and their results stay together. They are either both summarized
        # or neither is.
        while (
            cutoff + 1 < len(in_context_messages_openai)
            and in_context_messages_openai[cutoff + 1]["role"] == MessageRole.tool
        ):
            cutoff += 1

        logger.debug("Evicting %s/%s messages...", cutoff, len(in_context_messages))
        return cutoff + 1


def get_token_counts_for_messages(in_context_messages: List[Message]) -> List[int]:
    in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]
    token_counts = [count_tokens(str(msg)) for msg in in_context_messages_openai]
    return token_counts


def is_context_overflow_error(
    exception: Union[requests.exceptions.RequestException, Exception],
) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)"""
    from mirix.utils import printd

    match_string = OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING

    # Backwards compatibility with openai python package/client v0.28 (pre-v1 client migration)
    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    # Based on python requests + OpenAI REST API (/v1)
    elif isinstance(exception, requests.exceptions.HTTPError):
        if exception.response is not None and "application/json" in exception.response.headers.get("Content-Type", ""):
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occurred, but couldn't find error field: {error_details}")
                    return False
                else:
                    error_details = error_details["error"]

                # Check for the specific error code
                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occurred, caught error code {error_details.get('code')}")
                    return True
                # Soft-check for "maximum context length" inside of the message
                elif error_details.get("message") and "maximum context length" in error_details.get("message"):
                    printd(f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})")
                    return True
                else:
                    printd(f"HTTPError occurred, but unknown error message: {error_details}")
                    return False
            except ValueError:
                # JSON decoding failed
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")

    # Generic fail
    else:
        return False
