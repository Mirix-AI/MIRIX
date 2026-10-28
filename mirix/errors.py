import json
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

# Avoid circular imports
if TYPE_CHECKING:
    from mirix.schemas.message import Message
    from mirix.schemas.mirix_message import MirixMessage


class ErrorCode(Enum):
    """Enum for error codes used across all LLM client implementations.

    These codes provide a standardized way to categorize errors from different
    LLM providers (OpenAI, Anthropic, Google AI, etc.) into common categories.
    """

    # Server-side errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DEPENDENCY_TIMEOUT = "DEPENDENCY_TIMEOUT"  # External service timeout (e.g., 424 errors)

    # Request/response errors
    CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"  # Bad request / validation errors

    # Authentication/authorization errors
    UNAUTHENTICATED = "UNAUTHENTICATED"  # 401 - Missing/invalid credentials
    PERMISSION_DENIED = "PERMISSION_DENIED"  # 403 - Valid credentials but no access

    # Resource errors
    NOT_FOUND = "NOT_FOUND"  # 404 - Model/resource doesn't exist


class MirixError(Exception):
    """Base class for all Mirix related errors."""

    def __init__(self, message: str, code: Optional[ErrorCode] = None, details: dict = {}):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.code:
            return f"{self.code.value}: {self.message}"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', code='{self.code}', details={self.details})"


class MirixToolCreateError(MirixError):
    """Error raised when a tool cannot be created."""

    default_error_message = "Error creating tool."

    def __init__(self, message=None):
        super().__init__(message=message or self.default_error_message)


class MirixConfigurationError(MirixError):
    """Error raised when there are configuration-related issues."""

    def __init__(self, message: str, missing_fields: Optional[List[str]] = None):
        self.missing_fields = missing_fields or []
        super().__init__(message=message, details={"missing_fields": self.missing_fields})


class MirixAgentNotFoundError(MirixError):
    """Error raised when an agent is not found."""


class MirixUserNotFoundError(MirixError):
    """Error raised when a user is not found."""


class LLMError(MirixError):
    pass


class LLMAuthenticationError(LLMError):
    """Error raised when LLM authentication fails."""

    pass


class LLMBadRequestError(LLMError):
    """Error raised when LLM request is malformed."""

    pass


class LLMConnectionError(LLMError):
    """Error raised when LLM connection fails."""

    pass


class LLMNotFoundError(LLMError):
    """Error raised when LLM resource is not found."""

    pass


class LLMPermissionDeniedError(LLMError):
    """Error raised when LLM permission is denied."""

    pass


class LLMRateLimitError(LLMError):
    """Error raised when LLM rate limit is exceeded."""

    pass


class LLMServerError(LLMError):
    """Error raised when LLM server encounters an error."""

    pass


class LLMUnprocessableEntityError(LLMError):
    """Error raised when LLM cannot process the entity."""

    pass


class LLMInvalidResponseError(LLMError):
    """The model answered, but not with anything usable (no choices, an empty
    message, a malformed answer) — still so after the agent's retries.

    A failure of the model exchange (for the retry policy, like an
    unparseable answer), not of the infrastructure or of MIRIX itself.
    """

    pass


class LLMRequestBuildError(MirixError):
    """MIRIX failed to make a model call before any answer came back: building
    the request (``build_request_data``) raised, or the SDK refused how MIRIX
    called it (a ``TypeError`` for an argument it does not take, …).

    Deliberately NOT an :class:`LLMError` nor a ``ValueError`` / ``KeyError``:
    it is MIRIX's own failure (a bug, a stored state that no longer
    validates), never the model's or the provider's. The agent does not retry
    it as an unusable answer, and the skill curator never counts it as a
    curation attempt; the original error is ``__cause__``.
    ``stage`` is ``"build"`` (building the request) or ``"sdk_call"``
    (handing it to the SDK). ``kind`` is MIRIX's retry-policy verdict on what
    it wraps (``internal``, or ``infra`` / ``db`` for a transient failure met
    on the way), which :meth:`wrapping` also writes into the message.
    """

    def __init__(self, message: str, *, stage: str = "build", kind: Optional[str] = None):
        self.stage = stage
        self.kind = kind
        details = {"stage": stage}
        if kind is not None:
            details["kind"] = kind
        super().__init__(message=message, details=details)

    @classmethod
    def wrapping(cls, what: str, cause: BaseException, *, stage: str) -> "LLMRequestBuildError":
        """``<what> failed [kind=<kind>] (<Cause>: <cause>)``, caused by ``cause``.

        ``kind`` is the retry policy's verdict on ``cause`` and the chain it
        wraps (``retry_policy.classify_build_cause``). It stands before the
        cause's own text, so no cut of the recorded text loses it, and a reader
        of that text (the eval harness judging a learning run's or a
        retrieval's error) never has to guess it from the words of the cause,
        which name only one level of the chain and may hold anything (bind
        parameters, a dictionary key).
        """
        from mirix.services.retry_policy import classify_build_cause

        try:
            kind = classify_build_cause(cause).value
        except Exception:  # noqa: BLE001 - never lose the original error to its label
            kind = "unknown"
        error = cls(
            f"{what} failed [kind={kind}] ({type(cause).__name__}: {cause})",
            stage=stage,
            kind=kind,
        )
        error.__cause__ = cause
        return error


class LearningRunInternalError(MirixError):
    """A learning run finished its work, but MIRIX's own failure left some of it
    undone: a session whose distill request could not be built
    (:class:`LLMRequestBuildError`, whatever it wraps). Such a session is
    neither counted against nor backed off — it stays pending for a fixed
    MIRIX — and the run is recorded ``failed`` with this error instead of
    ``succeeded`` (retry kind ``internal``).
    """

    def __init__(self, message: str, *, session_ids=()):
        self.session_ids = list(session_ids)
        super().__init__(message=message, details={"session_ids": self.session_ids})


class LLMTimeoutError(LLMConnectionError):
    """The model call itself timed out (read/write/pool timeout, or the overall
    request timeout — ``MIRIX_LLM_TIMEOUT_SECONDS``).

    A subclass of :class:`LLMConnectionError` so existing handlers keep
    catching it, but retry policy tells them apart: a failed
    CONNECTION is an infrastructure error (backoff, never counted against the
    session), a timed-out CALL is retried once and then counted.
    """

    pass


class ContextBudgetExceeded(MirixError):
    """A request would not leave the model room for a normal answer.

    Raised BEFORE anything is sent when ``output_cap(estimated input)`` — the
    most output the budget allows next to this input — is below the minimum
    (1,024 tokens). Callers that go through the budget module never see it; it
    is the guard for a caller that bypassed it. Retry policy: shrink the
    input (halve the chunk budget) and try again; never truncate silently.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        estimated_input_tokens: Optional[int] = None,
        output_cap: Optional[int] = None,
        context_window: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        minimum_output_tokens: Optional[int] = None,
    ):
        self.estimated_input_tokens = estimated_input_tokens
        self.output_cap = output_cap
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.minimum_output_tokens = minimum_output_tokens
        details = {
            "estimated_input_tokens": estimated_input_tokens,
            "output_cap": output_cap,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "minimum_output_tokens": minimum_output_tokens,
        }
        super().__init__(
            message=message
            or (
                f"the request's estimated input ({estimated_input_tokens} tokens) "
                f"leaves only {output_cap} output tokens inside the context budget "
                f"(minimum {minimum_output_tokens}); shrink the input"
            ),
            code=ErrorCode.CONTEXT_WINDOW_EXCEEDED,
            details=details,
        )


class AbortAgentStep(Exception):
    """Raised by a tool to end the whole agent step at once.

    Every other exception a tool raises becomes an error tool result that the
    model sees and the step goes on; this one (and its subclasses, e.g. a skill
    learning run that is no longer active) propagates out of ``Agent.step`` so
    the caller stops, instead of letting the model keep calling tools.
    """


class ToolCallRefused(MirixError):
    """Raised by a tool that declines to run this call (e.g. a
    ``finish_memory_update`` that would end a skill curation batch while some
    of its experiences are still unaddressed).

    Its message becomes the tool result the model reads. The call counts as
    neither executed nor failed (``Agent._handle_ai_response``), is never run
    again by the agent's tracing fallback, and the chain goes on so the model
    can act on the message.
    """


class ToolArgumentError(ValueError):
    """The arguments of a tool call do not fit the tool's signature (a required
    argument missing, an unknown one): the model's mistake, found before the
    tool runs (``Agent.execute_tool_and_persist_state`` binds them first), so
    it is never taken for a ``TypeError`` of MIRIX's own code."""


class AgentContextLimitReached(MirixError):
    """``Agent.step`` stopped before an LLM call whose estimated input would
    pass the step's context limit (``context_stop_fraction``), or —
    with auto-summary off — after the provider refused the input as too long.

    Control flow only: ``Agent.step`` catches it and ends with stop reason
    ``context_limit``; it never reaches the caller.
    """

    def __init__(self, estimated_input_tokens: Optional[int], limit: Optional[int]):
        self.estimated_input_tokens = estimated_input_tokens
        self.limit = limit
        super().__init__(
            message=(
                f"the next call's estimated input ({estimated_input_tokens} tokens) "
                f"passes the step's context limit ({limit} tokens)"
            ),
            code=ErrorCode.CONTEXT_WINDOW_EXCEEDED,
            details={
                "estimated_input_tokens": estimated_input_tokens,
                "limit": limit,
            },
        )


class LLMOutputTruncatedError(MirixError):
    """The model stopped because it hit the output cap (``finish_reason=length``).

    The truncated output is never used: the caller shrinks
    the input and retries.
    """

    def __init__(
        self, message: str = "the model output was truncated (finish_reason=length)"
    ):
        super().__init__(message=message, code=ErrorCode.CONTEXT_WINDOW_EXCEEDED)


class EmbeddingCallError(MirixError):
    """An embedding call failed after its own retries (never counted
    against a session; the vector is recomputed later or BM25 is used alone)."""

    def __init__(self, message: str = "the embedding call failed"):
        super().__init__(message=message)


class BedrockPermissionError(MirixError):
    """Exception raised for errors in the Bedrock permission process."""

    def __init__(
        self,
        message="User does not have access to the Bedrock model with the specified ID.",
    ):
        super().__init__(message=message)


class BedrockError(MirixError):
    """Exception raised for errors in the Bedrock process."""

    def __init__(self, message="Error with Bedrock model."):
        super().__init__(message=message)


class LLMJSONParsingError(MirixError):
    """Exception raised for errors in the JSON parsing process."""

    def __init__(self, message="Error parsing JSON generated by LLM"):
        super().__init__(message=message)


class LocalLLMError(MirixError):
    """Generic catch-all error for local LLM problems"""

    def __init__(self, message="Encountered an error while running local LLM"):
        super().__init__(message=message)


class LocalLLMConnectionError(MirixError):
    """Error for when local LLM cannot be reached with provided IP/port"""

    def __init__(self, message="Could not connect to local LLM"):
        super().__init__(message=message)


class ContextWindowExceededError(MirixError):
    """Error raised when the context window is exceeded but further summarization fails."""

    def __init__(self, message: str, details: dict = {}):
        error_message = f"{message} ({details})"
        super().__init__(
            message=error_message,
            code=ErrorCode.CONTEXT_WINDOW_EXCEEDED,
            details=details,
        )


class RateLimitExceededError(MirixError):
    """Error raised when the llm rate limiter throttles api requests."""

    def __init__(self, message: str, max_retries: int):
        error_message = f"{message} ({max_retries})"
        super().__init__(
            message=error_message,
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details={"max_retries": max_retries},
        )


class MirixMessageError(MirixError):
    """Base error class for handling message-related errors."""

    messages: List[Union["Message", "MirixMessage"]]
    default_error_message: str = "An error occurred with the message."

    def __init__(
        self,
        *,
        messages: List[Union["Message", "MirixMessage"]],
        explanation: Optional[str] = None,
    ) -> None:
        error_msg = self.construct_error_message(messages, self.default_error_message, explanation)
        super().__init__(error_msg)
        self.messages = messages

    @staticmethod
    def construct_error_message(
        messages: List[Union["Message", "MirixMessage"]],
        error_msg: str,
        explanation: Optional[str] = None,
    ) -> str:
        """Helper method to construct a clean and formatted error message."""
        if explanation:
            error_msg += f" (Explanation: {explanation})"

        # Pretty print out message JSON
        message_json = json.dumps([message.model_dump() for message in messages], indent=4)
        return f"{error_msg}\n\n{message_json}"


class MissingToolCallError(MirixMessageError):
    """Error raised when a message is missing a tool call."""

    default_error_message = "The message is missing a tool call."


class InvalidToolCallError(MirixMessageError):
    """Error raised when a message uses an invalid tool call."""

    default_error_message = "The message uses an invalid tool call or has improper usage of a tool call."


class MissingInnerMonologueError(MirixMessageError):
    """Error raised when a message is missing an inner monologue."""

    default_error_message = "The message is missing an inner monologue."


class InvalidInnerMonologueError(MirixMessageError):
    """Error raised when a message has a malformed inner monologue."""

    default_error_message = "The message has a malformed inner monologue."
