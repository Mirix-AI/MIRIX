import logging
from typing import Any, List, Union

import httpx

from mirix.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from mirix.schemas.message import Message
from mirix.utils import count_tokens, printd

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


async def make_post_request(url: str, headers: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    """Async HTTP POST using httpx. Call with await."""
    printd(f"Sending request to {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
        printd(f"Response status code: {response.status_code}")

        # Raise for 4XX/5XX HTTP errors
        response.raise_for_status()

        # Check if the response content type indicates JSON and attempt to parse it
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            try:
                response_data = response.json()
                printd(f"Response JSON: {response_data}")
            except ValueError as json_err:
                error_message = f"Failed to parse JSON despite Content-Type being {content_type}: {json_err}"
                printd(error_message)
                raise ValueError(error_message) from json_err
        else:
            error_message = f"Unexpected content type returned: {response.headers.get('Content-Type')}"
            printd(error_message)
            raise ValueError(error_message)
        return response_data

    except httpx.HTTPStatusError as http_err:
        error_message = f"HTTP error occurred: {http_err}"
        if http_err.response is not None:
            error_message += f" | Status code: {http_err.response.status_code}, Message: {http_err.response.text}"
        printd(error_message)
        raise httpx.HTTPStatusError(error_message, request=http_err.request, response=http_err.response) from http_err

    except httpx.TimeoutException as timeout_err:
        error_message = f"Request timed out: {timeout_err}"
        printd(error_message)
        raise httpx.TimeoutException(error_message) from timeout_err

    except httpx.RequestError as req_err:
        error_message = f"Request failed: {req_err}"
        printd(error_message)
        raise httpx.RequestError(error_message) from req_err

    except ValueError as val_err:
        error_message = f"ValueError: {val_err}"
        printd(error_message)
        raise ValueError(error_message) from val_err

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        printd(error_message)
        raise Exception(error_message) from e


def get_token_counts_for_messages(in_context_messages: List[Message]) -> List[int]:
    in_context_messages_openai = [m.to_openai_dict() for m in in_context_messages]
    token_counts = [count_tokens(str(msg)) for msg in in_context_messages_openai]
    return token_counts


def is_context_overflow_error(
    exception: Union[httpx.HTTPError, Exception],
) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages)."""
    from mirix.utils import printd

    match_string = OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING

    if match_string in str(exception):
        printd(f"Found '{match_string}' in str(exception)={(str(exception))}")
        return True

    if isinstance(exception, httpx.HTTPStatusError) and exception.response is not None:
        ct = exception.response.headers.get("Content-Type", "")
        if "application/json" in ct:
            try:
                error_details = exception.response.json()
                if "error" not in error_details:
                    printd(f"HTTPError occurred, but couldn't find error field: {error_details}")
                    return False
                error_details = error_details["error"]
                if error_details.get("code") == "context_length_exceeded":
                    printd(f"HTTPError occurred, caught error code {error_details.get('code')}")
                    return True
                if error_details.get("message") and "maximum context length" in error_details.get("message", ""):
                    printd(f"HTTPError occurred, found '{match_string}' in error message contents ({error_details})")
                    return True
                printd(f"HTTPError occurred, but unknown error message: {error_details}")
                return False
            except ValueError:
                printd(f"HTTPError occurred ({exception}), but no JSON error message.")
    return False
