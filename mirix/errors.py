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


class LLMBadResponseShapeError(LLMError):
    """LLM returned a response with the wrong shape — empty choices, empty
    content + no tool calls, or an unexpected finish_reason like ``length``.

    These look like transient provider quirks: the LLM should produce a
    well-shaped response on retry. Classified TRANSIENT; ``_get_ai_reply``'s
    inner retry loop handles it like any other transient.
    """

    pass


class LLMChainingExhaustedError(MirixError):
    """The meta-agent LLM loop ran out of chaining budget with the last
    iteration still flagging a CorrectableToolError-shape failure.

    Concretely: the LLM emitted a malformed tool call (unknown tool name /
    bad JSON args / failed validation), the loop injected a "please finish"
    re-prompt, and the LLM either emitted another malformed call OR
    `max_chaining_steps` was reached without recovery.

    Classified as Bucket.PERMANENT — retrying the whole step would just
    re-run the same prompt against the same LLM and get the same broken
    output. Surfaces as a save-path PERMANENT_FAILURE so the source can be
    dead-lettered with a distinct outcome value when a future status-column
    migration distinguishes outcomes at the DB level.
    """

    pass


class CorrectableToolError(MirixError):
    """Tool execution failed for a reason the LLM can self-correct on a re-prompt.

    Examples: missing required tool argument, malformed JSON in tool args,
    argument type mismatch caught by preprocessing.

    Distinguishes "LLM produced something wrong" — which the bounded re-prompt
    in `Agent.step()` is designed to fix — from "something downstream broke"
    (DB/provider/code-bug) which must propagate so `process_with_policy` can
    classify and decide retry/finalize.

    Raising this from tool argument preprocessing keeps the existing
    feed-back-to-LLM behavior; everything else (`AttributeError`,
    `OperationalError`, `ProviderTransientError`, …) propagates out of
    `step()` instead of being string-erased at the tool-execution swallow site.
    """

    pass


class ProviderTransientError(MirixError):
    """Provider call failed with a retryable condition (5xx, 429, timeout).

    The ECMS provider boundary translates SDK-specific transient exceptions
    into this type. `error_policy.classify()` maps it to `Bucket.TRANSIENT`
    by isinstance — no string-matching in the MIRIX core.
    """

    pass


class ProviderPermanentError(MirixError):
    """Provider call failed with a non-retryable condition (4xx auth, bad request).

    The ECMS provider boundary translates SDK-specific permanent exceptions
    into this type. `error_policy.classify()` maps it to `Bucket.PERMANENT`
    by isinstance.
    """

    pass


class ProviderNotFoundError(ProviderPermanentError):
    """Provider call returned 404 / "entity does not exist".

    Subclass of ProviderPermanentError so it classifies the same way at the
    policy layer, but distinct enough that read paths can convert it to
    `None` rather than crashing. Translated from IPS-R's
    ``UnknownEntityError`` at the provider boundary.
    """

    pass


class RelationalProviderRequiredError(ProviderTransientError):
    """A relational provider was required but none is currently registered.

    Raised by ``get_relational_provider()`` once provider mode has latched (a
    relational provider was registered at least once this process) but the
    active provider is now missing — e.g. it was unregistered during shutdown
    teardown while a save was still in flight. Subclasses
    ``ProviderTransientError`` so the queue classifies it TRANSIENT and
    redelivers the save: on the next startup the provider is registered again
    and the save succeeds, rather than silently reading/writing the wrong store
    (PostgreSQL) under provider mode.
    """

    pass


class ProviderConflictError(MirixError):
    """Provider call hit a unique-constraint / duplicate-key conflict.

    Callers (`source_message_manager`, `user_manager`, `client_manager`,
    `memory_citation_manager`) treat this as an idempotent no-op — the
    intended row already exists.

    Kept distinct from `ProviderPermanentError`: collapsing the two would
    break the dedup control flow (L1 source-message uniqueness, L3 citation
    dedup). Never classified into a transient/permanent bucket; the
    `is_conflict(exc)` predicate is the only consumer.
    """

    pass


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
