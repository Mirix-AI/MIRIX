import copy
import json
import logging
import warnings
from collections import OrderedDict
from typing import Any, Awaitable, Callable, List, Optional, Union

import httpx

from mirix.constants import OPENAI_CONTEXT_WINDOW_ERROR_SUBSTRING
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse, Choice
from mirix.settings import summarizer_settings
from mirix.utils import count_tokens, json_dumps, printd

logger = logging.getLogger(__name__)


#: Connecting to a model provider never waits longer than this (the SDKs' own
#: default): an unreachable host is an infrastructure error in seconds, not
#: after the whole call budget.
LLM_CONNECT_TIMEOUT_SECONDS = 5.0


def llm_timeout(sdk: Any = None) -> Any:
    """The per-attempt HTTP timeout of a model call: read, write
    and pool waits up to ``MIRIX_LLM_TIMEOUT_SECONDS`` (default 600), the
    connect at most :data:`LLM_CONNECT_TIMEOUT_SECONDS`. The whole call —
    every transport retry included — is bounded separately by the same
    ``MIRIX_LLM_TIMEOUT_SECONDS`` as a wall-clock deadline (:func:`with_llm_deadline`).

    ``sdk`` is the SDK module whose client takes the timeout (``openai``,
    ``anthropic``): the timeout is built with that SDK's OWN ``Timeout`` type,
    so it is valid whichever HTTP stack the installed SDK version runs on (the
    ``httpx``-based SDKs export ``httpx.Timeout``; ``anthropic`` 1.x rejects an
    ``httpx.Timeout`` because it runs on ``httpx2``). Without ``sdk`` — for
    MIRIX's own ``httpx`` calls — it is an :class:`httpx.Timeout`. An SDK whose
    ``Timeout`` type cannot be built this way gets the plain float seconds
    (every SDK accepts one; only the shorter connect timeout is lost).
    """
    from mirix import constants

    seconds = float(constants.LLM_TIMEOUT_SECONDS)
    connect = min(LLM_CONNECT_TIMEOUT_SECONDS, seconds)
    if sdk is None:
        return httpx.Timeout(seconds, connect=connect)
    timeout_type = getattr(sdk, "Timeout", None)
    if isinstance(timeout_type, type):
        try:
            return timeout_type(seconds, connect=connect)
        except Exception:  # noqa: BLE001 — an SDK with another Timeout signature
            logger.warning(
                "%s.Timeout(seconds, connect=...) failed; passing the timeout as "
                "%.0f s (no separate connect timeout)",
                getattr(sdk, "__name__", sdk),
                seconds,
            )
    return seconds


def llm_transport_options(sdk: Any = None) -> dict:
    """``timeout`` and ``max_retries`` for a model SDK client.

    Read at call time from ``mirix.constants`` (``MIRIX_LLM_TIMEOUT_SECONDS``,
    default 600; ``MIRIX_LLM_MAX_RETRIES``, default 2) and always passed
    explicitly, so MIRIX never depends on an SDK's own defaults. ``timeout``
    is built with ``sdk``'s own ``Timeout`` type (see :func:`llm_timeout`);
    pass the module of the SDK whose client takes these options.
    """
    from mirix import constants

    return {
        "timeout": llm_timeout(sdk),
        "max_retries": int(constants.LLM_MAX_RETRIES),
    }


async def with_llm_deadline(call: Awaitable[Any]) -> Any:
    """Await one model call — its transport retries included — within the
    wall-clock deadline ``MIRIX_LLM_TIMEOUT_SECONDS`` (the call timeout,
    10 min by default).

    The HTTP timeouts bound each read or write, not the whole exchange: a
    provider (or proxy) that keeps trickling bytes would hold a call forever,
    and SDK retries would stack attempt after attempt. Past the deadline the
    call is cancelled and :class:`~mirix.errors.LLMTimeoutError` raised.
    """
    import asyncio

    from mirix import constants
    from mirix.errors import ErrorCode, LLMTimeoutError

    deadline = float(constants.LLM_TIMEOUT_SECONDS)
    try:
        return await asyncio.wait_for(call, timeout=deadline)
    except asyncio.TimeoutError as exc:
        raise LLMTimeoutError(
            message=(
                f"the model call did not complete within its {deadline:g} s "
                "deadline (MIRIX_LLM_TIMEOUT_SECONDS)"
            ),
            code=ErrorCode.DEPENDENCY_TIMEOUT,
        ) from exc


def _transport_retry_kind(response: Optional[httpx.Response]) -> str:
    """The retry-policy kind of the failure an SDK transport retry follows."""
    from mirix.services.retry_policy import ErrorKind

    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status in (401, 403, 404):
        return ErrorKind.CONFIG.value
    return ErrorKind.INFRA.value  # 408/409/429/5xx, or no answer at all


_RECORDING_SDK_CLASSES: dict = {}


def recording_sdk_class(sdk_class: type) -> type:
    """``sdk_class`` (``AsyncOpenAI``, ``AsyncAzureOpenAI``,
    ``AsyncAnthropic`` …) with its transport retries recorded (every
    retry lands in the learning run's or retrieval log's
    ``retry_events``).

    The OpenAI and Anthropic SDKs retry inside one call (connection errors,
    timeouts, 408/409/429/5xx) and announce each retry through their async
    ``_sleep_for_retry`` hook; the subclass records the retry with the
    recorder of the enclosing
    :func:`~mirix.services.retry_policy.recording_retries` block (a no-op
    outside one), then waits as the SDK would. An SDK without that hook is
    returned unchanged — its retries still happen, unrecorded.
    """
    import inspect

    cached = _RECORDING_SDK_CLASSES.get(sdk_class)
    if cached is not None:
        return cached
    hook = getattr(sdk_class, "_sleep_for_retry", None)
    if hook is None or not inspect.iscoroutinefunction(hook):
        _RECORDING_SDK_CLASSES[sdk_class] = sdk_class
        return sdk_class

    async def _sleep_for_retry(self, **kwargs):  # type: ignore[no-untyped-def]
        from mirix.services.retry_policy import record_retry

        response = kwargs.get("response")
        retries_taken = int(kwargs.get("retries_taken") or 0)
        waited = 0.0
        try:
            remaining = int(kwargs.get("max_retries") or 0) - retries_taken
            waited = float(
                self._calculate_retry_timeout(
                    remaining,
                    kwargs.get("options"),
                    response.headers if response is not None else None,
                )
            )
        except Exception:  # noqa: BLE001 — the wait is informative only
            waited = 0.0
        await record_retry(
            _transport_retry_kind(response),
            retries_taken + 1,
            waited,
            stage="transport",
        )
        return await hook(self, **kwargs)

    recording = type(
        f"Recording{sdk_class.__name__}",
        (sdk_class,),
        {"_sleep_for_retry": _sleep_for_retry, "__module__": __name__},
    )
    _RECORDING_SDK_CLASSES[sdk_class] = recording
    return recording


# Transport-level retry of one HTTP model call (mirrors the OpenAI SDK).
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
_INITIAL_RETRY_DELAY = 0.5
_MAX_RETRY_DELAY = 8.0
_MAX_RETRY_AFTER = 60.0


def _retry_after_seconds(value: Optional[str]) -> Optional[float]:
    """Seconds from a Retry-After header (delta-seconds or HTTP date)."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        from email.utils import parsedate_to_datetime

        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        import datetime as _dt

        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        seconds = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    return seconds if 0 <= seconds <= _MAX_RETRY_AFTER else None


def transport_retry_delay(attempt: int, retry_after: Optional[str] = None) -> float:
    """Wait before transport retry ``attempt`` (0-based): the server's
    Retry-After when sane (<= 60 s), else 0.5 s doubling to 8 s, with jitter."""
    import random

    hinted = _retry_after_seconds(retry_after)
    if hinted is not None:
        return hinted
    delay = min(_INITIAL_RETRY_DELAY * (2**attempt), _MAX_RETRY_DELAY)
    return delay * (1 - 0.25 * random.random())


async def post_json_with_retries(
    url: str,
    headers: dict,
    data: dict,
    *,
    timeout: Union[float, httpx.Timeout],
    max_retries: int,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
) -> dict:
    """POST ``data`` as JSON and return the JSON body, with transport retries.

    For HTTP model APIs without an SDK (the Google client). ``timeout`` bounds
    each attempt; up to ``max_retries`` retries follow connection errors,
    timeouts and HTTP 408/409/429/5xx (Retry-After honored), waiting with
    ``asyncio.sleep`` — never blocking the event loop. Every retry is recorded
    with the enclosing ``recording_retries`` block's recorder. Other
    HTTP errors raise ``httpx.HTTPStatusError`` at once (message includes the
    response body).
    """
    import asyncio

    from mirix.services.retry_policy import ErrorKind, record_retry

    sleep = sleep or asyncio.sleep
    attempt = 0
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        while True:
            retry_after = None
            failure: Optional[BaseException] = None
            try:
                response = await client.post(url, headers=headers, json=data)
            except httpx.TransportError as exc:
                if attempt >= max_retries:
                    raise
                reason = type(exc).__name__
                failure = exc
            else:
                status = response.status_code
                if status < 400:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" not in content_type.lower():
                        raise ValueError(
                            f"Unexpected content type returned: {content_type or 'none'}"
                        )
                    return response.json()
                retryable = status in _RETRYABLE_STATUS_CODES or status >= 500
                if not retryable or attempt >= max_retries:
                    message = (
                        f"HTTP error occurred: {status} for {url.split('?', 1)[0]}"
                        f" | Status code: {status}, Message: {response.text[:2000]}"
                    )
                    raise httpx.HTTPStatusError(
                        message, request=response.request, response=response
                    )
                retry_after = response.headers.get("retry-after")
                reason = f"HTTP {status}"
            delay = transport_retry_delay(attempt, retry_after)
            attempt += 1
            logger.info(
                "Model call failed (%s); transport retry %d/%d in %.2fs",
                reason,
                attempt,
                max_retries,
                delay,
            )
            await record_retry(
                ErrorKind.INFRA,
                attempt,
                delay,
                stage="transport",
                error=failure or RuntimeError(reason),
            )
            await sleep(delay)


def read_file_base64(path: str) -> str:
    """A file's bytes, base64-encoded. Blocking — call via ``asyncio.to_thread``
    (image payloads are read on the request path; never on the event loop)."""
    import base64

    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


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
    exception: Union[httpx.HTTPError, Exception],
) -> bool:
    """Checks if an exception is due to context overflow (based on common OpenAI response messages).

    A request refused BEFORE sending by the context budget
    (``ContextBudgetExceeded``) is the same situation — the input is
    too large for the window — so callers that shrink their context on
    overflow (the agent summarizer) handle it too.
    """
    from mirix.errors import ContextBudgetExceeded
    from mirix.utils import printd

    if isinstance(exception, ContextBudgetExceeded):
        return True

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
