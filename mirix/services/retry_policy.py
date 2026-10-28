"""Retry primitives shared by the distiller, the curator and retrieval.

Two layers of retry exist and must not be confused:

* **transport** — the model SDK retries ONE call by itself (connection errors,
  timeouts, 408/409/429/5xx, honoring ``Retry-After``), bounded by
  ``MIRIX_LLM_MAX_RETRIES`` (see :mod:`mirix.llm_api.helpers`). Embedding
  calls have no transport layer: their SDK retries are off and the embedding
  policy below is their only retry (:mod:`mirix.embeddings`);
* **application** — what MIRIX does once a call (or a transaction) finally
  failed. That is this module.

It provides:

* :func:`classify_error` — one :class:`ErrorKind` for any exception (and
  :func:`classify_finish_reason` for a truncated answer), across the
  OpenAI/Anthropic SDK errors, raw ``httpx`` errors, MIRIX's ``LLM*`` errors,
  SQLAlchemy / asyncpg / SQLite database errors and the budget errors;
* :data:`RETRY_POLICIES` — the retry table as data: how the current run
  retries each kind, and whether a final failure counts toward the session's
  3 distill failures or defers it across runs;
* :func:`retry_call` — the generic in-run retry for the kinds whose retry is
  "call it again" (a timed-out call once, a database transaction up to 5
  times, an embedding call up to 3 times, all with ``asyncio.sleep``); the
  kinds whose retry changes the input (halve the chunk budget, add a repair
  prompt) are re-raised for the caller to handle;
* :func:`distill_retry_delay` / :func:`next_distill_retry_after` — the
  cross-run backoff written to ``conversation_session.distill_retry_after``
  after an infrastructure failure: 1 minute, doubling, capped at 1 hour
  (``MIRIX_DISTILL_RETRY_BACKOFF_MIN_SECONDS`` / ``_MAX_SECONDS``);
* :class:`RetryEventRecorder` — appends ``{kind, attempt, waited_s, ...}`` for
  every retry to a learning run's ``retry_events`` or a retrieval log's.

Every wait is ``asyncio.sleep``: nothing here blocks the event loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional, TypeVar

from mirix.log import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ErrorKind(str, Enum):
    """What went wrong, as far as retrying is concerned."""

    #: Network error, connection timeout, HTTP 408/409/424/429/5xx, provider
    #: unavailable. Transient and not the session's fault.
    INFRA = "infra"
    #: Authentication / permission / unknown model (401/403/404): the
    #: deployment is misconfigured. Not the session's fault either.
    CONFIG = "config"
    #: The answer hit the output cap (``finish_reason=length``).
    TRUNCATED = "truncated"
    #: The input does not fit: ``ContextBudgetExceeded`` before sending, or the
    #: provider's own context-length refusal.
    BUDGET = "budget"
    #: The answer is not what was asked for (e.g. not a JSON array).
    PARSE = "parse"
    #: The call itself timed out (``MIRIX_LLM_TIMEOUT_SECONDS``), or a proxy
    #: that enforces the call's deadline itself answered that the model's own
    #: attempt used all of it (the eval gateway's 504 ``deadline_exceeded``
    #: with ``blame: "model"``): the same event, seen from the
    #: other side of the proxy.
    TIMEOUT = "timeout"
    #: Deadlock, serialization failure, lost connection, locked database.
    DB = "db"
    #: An embedding call failed.
    EMBEDDING = "embedding"
    #: MIRIX's own failure before any answer came back: a request it could not
    #: build or an SDK it called wrongly (``LLMRequestBuildError``), whatever
    #: that wraps — a ``KeyError``, a ``json.JSONDecodeError``, a missing image
    #: file (``FileNotFoundError``) … — but a network or transient database
    #: failure met while building. Never the model's, the provider's or the
    #: network's, so it is never retried, never counted against a session or
    #: an experience, and never backed off: the work stays pending for a fixed
    #: MIRIX and the learning run fails with the error recorded.
    INTERNAL = "internal"
    #: The provider refused the request for its content: its moderation
    #: flagged the input (a 403 saying so). The same input is refused again,
    #: and it is no misconfigured deployment (``CONFIG`` would defer the work
    #: across runs, again and again): not retried, counted — like an
    #: answer that stays unusable.
    REFUSED = "refused"
    #: Anything else (a bad request, a bug): not retried.
    OTHER = "other"


# Strategies: how the CURRENT run retries a kind.
STRATEGY_TRANSPORT = "transport"  # the SDK already retried; nothing more now
STRATEGY_HALVE_CHUNK_BUDGET = "halve_chunk_budget"  # caller re-chunks smaller
STRATEGY_REPAIR_PROMPT = "repair_prompt"  # caller asks again with a repair prompt
STRATEGY_RETRY_SAME = "retry_same"  # the same call again, at once
STRATEGY_RETRY_TRANSACTION = "retry_transaction"  # the whole transaction again
STRATEGY_BACKOFF = "backoff"  # the same call again after a growing wait
STRATEGY_NONE = "none"

#: Strategies :func:`retry_call` performs itself (the input stays the same).
REPEAT_STRATEGIES = frozenset(
    {STRATEGY_RETRY_SAME, STRATEGY_RETRY_TRANSACTION, STRATEGY_BACKOFF}
)


@dataclass(frozen=True)
class RetryPolicy:
    """One row of the retry table.

    ``in_run_retries`` — retries within the current run (halvings, repair
    prompts, repeats). ``counts_as_failure`` — whether a failure left after
    them counts toward the session's 3 distill failures (for the curator,
    ``curation_attempts``). ``defer_across_runs`` — whether the session is
    left pending with a ``distill_retry_after`` backoff instead.
    """

    kind: ErrorKind
    strategy: str
    in_run_retries: int
    counts_as_failure: bool
    defer_across_runs: bool
    base_delay: float = 0.0
    max_delay: float = 0.0

    def delay(self, retry_index: int) -> float:
        """Seconds to wait before in-run retry ``retry_index`` (0-based)."""
        if self.base_delay <= 0:
            return 0.0
        return min(
            self.base_delay * (2**retry_index), self.max_delay or self.base_delay
        )


RETRY_POLICIES: Dict[ErrorKind, RetryPolicy] = {
    ErrorKind.INFRA: RetryPolicy(
        ErrorKind.INFRA,
        STRATEGY_TRANSPORT,
        in_run_retries=0,
        counts_as_failure=False,
        defer_across_runs=True,
    ),
    ErrorKind.CONFIG: RetryPolicy(
        ErrorKind.CONFIG,
        STRATEGY_NONE,
        in_run_retries=0,
        counts_as_failure=False,
        defer_across_runs=True,
    ),
    ErrorKind.TRUNCATED: RetryPolicy(
        ErrorKind.TRUNCATED,
        STRATEGY_HALVE_CHUNK_BUDGET,
        in_run_retries=3,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
    ErrorKind.BUDGET: RetryPolicy(
        ErrorKind.BUDGET,
        STRATEGY_HALVE_CHUNK_BUDGET,
        in_run_retries=3,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
    ErrorKind.PARSE: RetryPolicy(
        ErrorKind.PARSE,
        STRATEGY_REPAIR_PROMPT,
        in_run_retries=2,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
    ErrorKind.TIMEOUT: RetryPolicy(
        ErrorKind.TIMEOUT,
        STRATEGY_RETRY_SAME,
        in_run_retries=1,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
    # Like transaction_retry(max_retries=5, base_delay=0.1, max_delay=3.0); a
    # failure left after that fails the run (its sessions are released).
    ErrorKind.DB: RetryPolicy(
        ErrorKind.DB,
        STRATEGY_RETRY_TRANSACTION,
        in_run_retries=5,
        counts_as_failure=False,
        defer_across_runs=False,
        base_delay=0.1,
        max_delay=3.0,
    ),
    # The vector is recomputed later (skills) or BM25 is used alone (queries).
    ErrorKind.EMBEDDING: RetryPolicy(
        ErrorKind.EMBEDDING,
        STRATEGY_BACKOFF,
        in_run_retries=3,
        counts_as_failure=False,
        defer_across_runs=False,
        base_delay=1.0,
        max_delay=8.0,
    ),
    # The provider's moderation refused the input: the same input would be
    # refused again; counted, never deferred.
    ErrorKind.REFUSED: RetryPolicy(
        ErrorKind.REFUSED,
        STRATEGY_NONE,
        in_run_retries=0,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
    # MIRIX's own failure: neither counted nor deferred; the run fails.
    ErrorKind.INTERNAL: RetryPolicy(
        ErrorKind.INTERNAL,
        STRATEGY_NONE,
        in_run_retries=0,
        counts_as_failure=False,
        defer_across_runs=False,
    ),
    ErrorKind.OTHER: RetryPolicy(
        ErrorKind.OTHER,
        STRATEGY_NONE,
        in_run_retries=0,
        counts_as_failure=True,
        defer_across_runs=False,
    ),
}


def policy_for(kind: ErrorKind) -> RetryPolicy:
    return RETRY_POLICIES[ErrorKind(kind)]


# --------------------------------------------------------------------------- #
# Classification                                                               #
# --------------------------------------------------------------------------- #

_INFRA_STATUS = frozenset({408, 409, 424, 429})
_CONFIG_STATUS = frozenset({401, 403, 404})
_CONTEXT_LENGTH_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "too many tokens",
    "exceeds the maximum number of tokens",
    "input token count",
)
# SQLSTATEs of transient database failures (Postgres).
_TRANSIENT_SQLSTATES = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
        "53300",  # too_many_connections
    }
)
# Driver messages of transient failures, matched against the DRIVER's error
# (``exc.orig``) only — never ``str()`` of the SQLAlchemy error, which carries
# the statement and its bind parameters (user content), so a row whose text
# happens to say "database is locked" must not make a deterministic failure
# look transient.
_TRANSIENT_DB_MESSAGES = (
    "deadlock detected",
    "could not serialize access",
    "database is locked",
    "database table is locked",
    "connection is closed",
    "connection was closed",
    "server closed the connection",
    "terminating connection",
    "connection reset",
    "could not obtain lock",
)
# SQLite primary result codes of a transient failure (the extended code's low
# byte): SQLITE_BUSY, SQLITE_LOCKED.
_TRANSIENT_SQLITE_CODES = frozenset({5, 6})


def _status_code(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        value = details.get("status_code")
        if isinstance(value, int):
            return value
    return None


def _mentions_context_length(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_LENGTH_MARKERS)


def _error_body(exc: BaseException) -> Optional[Dict[str, Any]]:
    """The error object of the HTTP answer ``exc`` carries, if any: an SDK
    error's ``body`` (the OpenAI SDK keeps the answer's ``error`` object), a
    MIRIX LLM error's ``details`` (``{"body": ...}``, or the body itself)."""
    candidates: List[Any] = []
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        candidates.append(details.get("body") if "body" in details else details)
    candidates.append(getattr(exc, "body", None))
    for body in candidates:
        if isinstance(body, dict):
            inner = body.get("error")
            return inner if isinstance(inner, dict) else body
    return None


def _proxy_timed_out_the_call(exc: BaseException) -> bool:
    """A 504 from a proxy that enforces the call's deadline itself and says the
    model's own attempt used all of it — the eval gateway:
    ``{"type": "deadline_exceeded", "blame": "model", ...}``. The call timed
    out (retried once, then counted), exactly as if MIRIX's own
    ``MIRIX_LLM_TIMEOUT_SECONDS`` had fired. A deadline the proxy lost to its
    queue or its retries (``blame`` anything else) stays infrastructure."""
    body = _error_body(exc)
    return (
        body is not None
        and body.get("type") == "deadline_exceeded"
        and body.get("blame") == "model"
    )


def _is_moderation_refusal(exc: BaseException) -> bool:
    """A 403 because the provider's moderation flagged the input (OpenRouter:
    "<model> requires moderation on <provider>. Your input was flagged for
    ..."): :attr:`ErrorKind.REFUSED`, not a configuration error. Judged on the
    answer's error message when the exception carries it, else on its text."""
    body = _error_body(exc)
    message = body.get("message") if body is not None else None
    text = str(message) if message else str(exc)
    return "moderat" in text.lower()


def _kind_for_status(status: int, exc: BaseException) -> ErrorKind:
    if status == 504 and _proxy_timed_out_the_call(exc):
        return ErrorKind.TIMEOUT
    if status in _INFRA_STATUS or status >= 500:
        return ErrorKind.INFRA
    if status == 403 and _is_moderation_refusal(exc):
        return ErrorKind.REFUSED
    if status in _CONFIG_STATUS:
        return ErrorKind.CONFIG
    if status == 413 or _mentions_context_length(exc):
        return ErrorKind.BUDGET
    return ErrorKind.OTHER


def _sqlstate(exc: BaseException) -> Optional[str]:
    for candidate in (exc, getattr(exc, "orig", None), exc.__cause__):
        if candidate is None:
            continue
        for attr in ("sqlstate", "pgcode"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and value:
                return value
    return None


def _is_transient_db_error(exc: BaseException) -> bool:
    """Deadlock / serialization failure / lost connection / locked database.

    Judged on the driver's error only: its SQLSTATE (Postgres), its result
    code (SQLite), else its own message — never the SQLAlchemy wrapper's text,
    which includes the bind parameters. An integrity or data error is never
    transient: the same statement fails the same way again.
    """
    try:
        from sqlalchemy import exc as sa_exc
    except ImportError:  # pragma: no cover - SQLAlchemy is a hard dependency
        return False
    if isinstance(exc, (sa_exc.DisconnectionError, sa_exc.TimeoutError)):
        return True  # pool checkout timeout / invalidated connection
    if isinstance(exc, (sa_exc.IntegrityError, sa_exc.DataError)):
        return False
    if isinstance(exc, sa_exc.DBAPIError):
        if getattr(exc, "connection_invalidated", False):
            return True
        state = _sqlstate(exc)
        if state:
            return state in _TRANSIENT_SQLSTATES or state.startswith("08")
        orig = getattr(exc, "orig", None)
        code = getattr(orig, "sqlite_errorcode", None)
        if isinstance(code, int):
            return (code & 0xFF) in _TRANSIENT_SQLITE_CODES
        if orig is None:
            return False
        text = str(orig).lower()
        return any(marker in text for marker in _TRANSIENT_DB_MESSAGES)
    try:
        import asyncpg.exceptions as pg_exc
    except ImportError:  # pragma: no cover
        return False
    return isinstance(
        exc,
        (
            pg_exc.DeadlockDetectedError,
            pg_exc.SerializationError,
            pg_exc.PostgresConnectionError,
            pg_exc.ConnectionDoesNotExistError,
            pg_exc.TooManyConnectionsError,
        ),
    )


def is_connect_timeout(exc: Optional[BaseException]) -> bool:
    """Whether ``exc`` is an HTTP stack's connect timeout: ``httpx``'s, or the
    same-named class of another stack an SDK may run on (``httpx2`` for
    ``anthropic`` 1.x), which is not an ``httpx.ConnectTimeout`` subclass."""
    import httpx

    if exc is None:
        return False
    if isinstance(exc, httpx.ConnectTimeout):
        return True
    return any(cls.__name__ == "ConnectTimeout" for cls in type(exc).__mro__)


def _classify_sdk_error(exc: BaseException) -> Optional[ErrorKind]:
    """OpenAI / Anthropic SDK exceptions (they share the httpx-based design)."""
    for sdk_name in ("openai", "anthropic"):
        try:
            sdk = __import__(sdk_name)
        except ImportError:  # pragma: no cover
            continue
        if isinstance(exc, sdk.APITimeoutError):
            if is_connect_timeout(exc.__cause__):
                return ErrorKind.INFRA
            return ErrorKind.TIMEOUT
        if isinstance(exc, sdk.APIConnectionError):
            return ErrorKind.INFRA
        if isinstance(exc, sdk.APIStatusError):
            return _kind_for_status(int(exc.status_code), exc)
    # google-genai (the ``google_ai`` embedding type): an HTTP error answer
    # carries its status as ``code``. Never imported here: its error can only
    # exist once the SDK is loaded (and importing it costs ~0.4 s).
    genai_errors = sys.modules.get("google.genai.errors")
    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if isinstance(code, int) and 100 <= code <= 599:
            return _kind_for_status(code, exc)
    return None


def _classify_aiohttp_error(exc: BaseException) -> Optional[ErrorKind]:
    """``aiohttp``'s network failures (the HTTP stack google-genai runs its
    async calls on when aiohttp is installed — the ``google_ai`` embedding
    type): a dropped or refused connection, a broken payload (``INFRA``), a
    timed-out read (``TIMEOUT``). None for anything else. Never imported
    here: such an error can only exist once aiohttp is loaded."""
    aiohttp = sys.modules.get("aiohttp")
    if aiohttp is None:
        return None
    timeout_error = getattr(aiohttp, "ServerTimeoutError", None)
    if isinstance(timeout_error, type) and isinstance(exc, timeout_error):
        return ErrorKind.TIMEOUT
    network = tuple(
        cls
        for cls in (
            getattr(aiohttp, "ClientConnectionError", None),
            getattr(aiohttp, "ClientPayloadError", None),
        )
        if isinstance(cls, type)
    )
    if network and isinstance(exc, network):
        return ErrorKind.INFRA
    return None


def _classify_build_error(exc: BaseException) -> ErrorKind:
    """The kind of an ``LLMRequestBuildError``: ``INTERNAL`` —
    MIRIX's own failure, whatever it wraps (a ``KeyError``, a
    ``json.JSONDecodeError``, a ``TimeoutError``, a local file missing:
    ``FileNotFoundError``) — except for the two transient failures building a
    request can meet on the way, which keep their own kind: the network while
    fetching something (an ``httpx`` transport error, a dropped connection, a
    408/429/5xx answer: ``INFRA``) and a transient database error while
    reading stored state (``DB``). See :func:`classify_build_cause`."""
    return classify_build_cause(exc.__cause__ or exc.__context__)


def classify_build_cause(cause: Optional[BaseException]) -> ErrorKind:
    """The kind of ``cause``, a failure met while building a model request (or
    handing it to the SDK), judged on it and on what it wraps
    (``__cause__`` / ``__context__``, up to 5 levels): ``INFRA`` or ``DB`` for
    the transient failures :func:`_classify_build_error` names, else
    ``INTERNAL``. ``LLMRequestBuildError`` records it in its text
    (``[kind=...]``), so whoever reads only the recorded error text — a
    learning run's or a retrieval's — sees MIRIX's own verdict on the whole
    chain, not just the one cause the text names."""
    import httpx

    seen = set()
    current: Optional[BaseException] = cause
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, (httpx.TransportError, ConnectionError)):
            return ErrorKind.INFRA
        if isinstance(current, httpx.HTTPStatusError):
            status = current.response.status_code
            if status in _INFRA_STATUS or status >= 500:
                return ErrorKind.INFRA
        if _is_transient_db_error(current):
            return ErrorKind.DB
        current = current.__cause__ or current.__context__
    return ErrorKind.INTERNAL


def _classify_one(exc: BaseException) -> ErrorKind:
    import httpx

    from mirix import errors

    # MIRIX's own failure first: what it wraps (a JSONDecodeError, a
    # TimeoutError, an OSError …) happened while MIRIX built the request, not
    # in any answer, so it is never a parse failure or a timed-out call.
    if isinstance(exc, errors.LLMRequestBuildError):
        return _classify_build_error(exc)
    if isinstance(exc, errors.LearningRunInternalError):
        return ErrorKind.INTERNAL
    # Budget / truncation / parsing next: they are the most specific.
    if isinstance(
        exc, (errors.ContextBudgetExceeded, errors.ContextWindowExceededError)
    ):
        return ErrorKind.BUDGET
    if isinstance(exc, errors.LLMOutputTruncatedError):
        return ErrorKind.TRUNCATED
    if isinstance(
        exc,
        (
            errors.LLMJSONParsingError,
            errors.LLMInvalidResponseError,
            json.JSONDecodeError,
        ),
    ):
        return ErrorKind.PARSE
    if isinstance(exc, errors.EmbeddingCallError):
        return ErrorKind.EMBEDDING
    # MIRIX's provider-neutral LLM errors (what LLM clients raise). A proxy's
    # verdict on its own deadline and a moderation refusal first: they are an
    # LLMServerError (504) and an LLMPermissionDeniedError (403) below.
    if isinstance(exc, errors.LLMError):
        if _status_code(exc) == 504 and _proxy_timed_out_the_call(exc):
            return ErrorKind.TIMEOUT
        if (
            isinstance(exc, errors.LLMPermissionDeniedError)
            or _status_code(exc) == 403
        ) and _is_moderation_refusal(exc):
            return ErrorKind.REFUSED
    if isinstance(exc, errors.LLMTimeoutError):
        return ErrorKind.TIMEOUT
    if isinstance(
        exc,
        (
            errors.LLMRateLimitError,
            errors.LLMServerError,
            errors.LLMConnectionError,
            errors.RateLimitExceededError,
            errors.LocalLLMConnectionError,
        ),
    ):
        return ErrorKind.INFRA
    if isinstance(
        exc,
        (
            errors.LLMAuthenticationError,
            errors.LLMPermissionDeniedError,
            errors.LLMNotFoundError,
            errors.MirixConfigurationError,  # e.g. no API key for the provider
        ),
    ):
        return ErrorKind.CONFIG
    if isinstance(exc, (errors.LLMBadRequestError, errors.LLMUnprocessableEntityError)):
        status = _status_code(exc)
        if status is not None:
            return _kind_for_status(status, exc)
        return ErrorKind.BUDGET if _mentions_context_length(exc) else ErrorKind.OTHER
    if isinstance(exc, errors.LLMError):
        status = _status_code(exc)
        return _kind_for_status(status, exc) if status is not None else ErrorKind.OTHER
    # Raw SDK / HTTP errors (a caller that did not go through an LLM client).
    sdk_kind = _classify_sdk_error(exc)
    if sdk_kind is not None:
        return sdk_kind
    aiohttp_kind = _classify_aiohttp_error(exc)
    if aiohttp_kind is not None:
        return aiohttp_kind
    if isinstance(exc, httpx.HTTPStatusError):
        return _kind_for_status(exc.response.status_code, exc)
    if isinstance(exc, httpx.ConnectTimeout):
        return ErrorKind.INFRA
    if isinstance(exc, httpx.TimeoutException):
        return ErrorKind.TIMEOUT
    if isinstance(exc, httpx.TransportError):
        return ErrorKind.INFRA
    # Databases (before generic OS errors: drivers raise OSError subclasses).
    if _is_transient_db_error(exc):
        return ErrorKind.DB
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorKind.TIMEOUT
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorKind.INFRA
    return ErrorKind.OTHER


#: Kinds of an embedding call's failure that are worth the embedding backoff:
#: the provider or the network may recover. A configuration error (401/403,
#: an unknown model), a bad request or a bug fails the same way again, so it
#: is not retried — the skill is stored without a vector at once.
_TRANSIENT_EMBEDDING_KINDS = frozenset(
    {ErrorKind.INFRA, ErrorKind.TIMEOUT, ErrorKind.EMBEDDING, ErrorKind.PARSE}
)


def classify_error(exc: BaseException, *, operation: Optional[str] = None) -> ErrorKind:
    """The :class:`ErrorKind` of ``exc``.

    ``operation="embedding"``: a transient failure of an embedding call
    (network, 5xx/429, timeout, a garbled answer) is
    :attr:`ErrorKind.EMBEDDING` — retried with the embedding backoff and never
    counted against a session; any other failure keeps its own kind
    (``CONFIG``, ``OTHER`` …), which :func:`retry_call` does not repeat. A
    generic wrapper exception (``OTHER``) is judged by the error it wraps
    (``__cause__`` / ``__context__``, a few levels deep) — but never past an
    ``LLMRequestBuildError``: that is ``INTERNAL`` whatever it wraps (a
    ``json.JSONDecodeError`` raised while MIRIX built a request is no
    unparseable answer, a missing local file no network failure), save a
    network or transient database failure met while building (``INFRA`` /
    ``DB``, see :func:`_classify_build_error`). Cancellation is not
    an error: ``asyncio.CancelledError`` is always re-raised, never retried —
    classifying it is a programming error.
    """
    if isinstance(exc, asyncio.CancelledError) or not isinstance(exc, Exception):
        raise TypeError(f"{type(exc).__name__} is not a retryable error; re-raise it")
    seen = set()
    current: Optional[BaseException] = exc
    kind = ErrorKind.OTHER
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        kind = _classify_one(current)
        if kind is not ErrorKind.OTHER:
            break
        current = current.__cause__ or current.__context__
    if operation == "embedding" and kind in _TRANSIENT_EMBEDDING_KINDS:
        return ErrorKind.EMBEDDING
    return kind


def classify_finish_reason(finish_reason: Optional[str]) -> Optional[ErrorKind]:
    """:attr:`ErrorKind.TRUNCATED` for an answer that hit the output cap."""
    if finish_reason in ("length", "max_tokens", "MAX_TOKENS"):
        return ErrorKind.TRUNCATED
    return None


def raise_if_truncated(response: Any) -> None:
    """Raise ``LLMOutputTruncatedError`` when a ChatCompletionResponse was cut
    off by the output cap — its content must never be used."""
    from mirix.errors import LLMOutputTruncatedError

    for choice in getattr(response, "choices", None) or []:
        if classify_finish_reason(getattr(choice, "finish_reason", None)):
            raise LLMOutputTruncatedError()


def is_retryable_in_run(kind: ErrorKind) -> bool:
    """True for the kinds :func:`retry_call` repeats itself."""
    policy = policy_for(kind)
    return policy.strategy in REPEAT_STRATEGIES and policy.in_run_retries > 0


# --------------------------------------------------------------------------- #
# Cross-run backoff (conversation_session.distill_retry_after)                  #
# --------------------------------------------------------------------------- #


def distill_retry_delay(
    attempt: int,
    *,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> float:
    """Seconds a session waits after its ``attempt``-th consecutive
    infrastructure failure: ``min * 2**(attempt - 1)``, capped at ``max``
    (defaults 60 s and 3600 s)."""
    from mirix import constants

    low = (
        float(constants.DISTILL_RETRY_BACKOFF_MIN_SECONDS)
        if min_seconds is None
        else float(min_seconds)
    )
    high = (
        float(constants.DISTILL_RETRY_BACKOFF_MAX_SECONDS)
        if max_seconds is None
        else float(max_seconds)
    )
    high = max(high, low)
    attempt = max(1, int(attempt))
    # 2**n overflows float math long after the cap binds; stop doubling early.
    exponent = min(attempt - 1, 64)
    return min(low * (2.0**exponent), high)


def next_distill_retry_after(
    now: dt.datetime,
    attempt: int,
    *,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> dt.datetime:
    """``now + distill_retry_delay(attempt)``."""
    return now + dt.timedelta(
        seconds=distill_retry_delay(
            attempt, min_seconds=min_seconds, max_seconds=max_seconds
        )
    )


# --------------------------------------------------------------------------- #
# Retry events                                                                 #
# --------------------------------------------------------------------------- #

_MAX_ERROR_CHARS = 500

RetrySink = Callable[[List[Dict[str, Any]]], Awaitable[bool]]


class RetryEventRecorder:
    """Collects one event per retry and appends them to a record.

    Each event is ``{"kind", "attempt", "waited_s", "at"}`` plus the optional
    ``"stage"`` (e.g. ``distill``, ``curate``, ``extract``, ``embed``),
    ``"session_id"`` and ``"error"`` (type and message, truncated). With a
    ``sink`` (a learning run's or a retrieval log's ``retry_events``) every
    event is persisted when recorded; a failed write is logged, never raised —
    recording a retry must not break the operation being retried — and the
    event is re-sent with the next one. Without a sink the events stay in
    :attr:`events` (e.g. for a retrieval log row written at the end).
    """

    def __init__(
        self,
        sink: Optional[RetrySink] = None,
        *,
        clock: Optional[Callable[[], dt.datetime]] = None,
    ):
        self._sink = sink
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self._events: List[Dict[str, Any]] = []
        self._unsent: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    @property
    def events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._events]

    @property
    def unsent(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self._unsent]

    async def record(
        self,
        kind: Any,
        attempt: int,
        waited_s: float,
        *,
        stage: Optional[str] = None,
        session_id: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "kind": ErrorKind(kind).value,
            "attempt": int(attempt),
            "waited_s": round(float(waited_s), 3),
            "at": self._clock().isoformat(),
        }
        if stage:
            event["stage"] = stage
        if session_id:
            event["session_id"] = session_id
        if error is not None:
            event["error"] = f"{type(error).__name__}: {error}"[:_MAX_ERROR_CHARS]
        self._events.append(event)
        self._unsent.append(event)
        await self.flush()
        return dict(event)

    async def flush(self) -> bool:
        """Send pending events to the sink. True when nothing is left pending."""
        if self._sink is None:
            return not self._unsent
        async with self._lock:
            if not self._unsent:
                return True
            batch = list(self._unsent)
            try:
                written = await self._sink(batch)
            except Exception:  # noqa: BLE001 — recording must never break the retry
                logger.warning(
                    "Could not record %d retry event(s)", len(batch), exc_info=True
                )
                return False
            if written:
                del self._unsent[: len(batch)]
            return bool(written)

    @classmethod
    def for_learning_run(
        cls, run_id: str, *, run_manager: Any = None
    ) -> "RetryEventRecorder":
        """Append to ``skill_learning_run.retry_events`` of ``run_id``."""
        if run_manager is None:
            from mirix.services.skill_learning_run_manager import (
                SkillLearningRunManager,
            )

            run_manager = SkillLearningRunManager()

        async def sink(events: List[Dict[str, Any]]) -> bool:
            return await run_manager.append_retry_events(run_id, events)

        return cls(sink)

    @classmethod
    def for_retrieval_log(
        cls, retrieval_id: str, *, session_maker: Any = None
    ) -> "RetryEventRecorder":
        """Append to ``retrieval_log.retry_events`` of ``retrieval_id``."""

        async def sink(events: List[Dict[str, Any]]) -> bool:
            return await append_retrieval_retry_events(
                retrieval_id, events, session_maker=session_maker
            )

        return cls(sink)


# --------------------------------------------------------------------------- #
# The recorder of the current stage (context-scoped)                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RecordingContext:
    recorder: RetryEventRecorder
    stage: Optional[str]
    session_id: Optional[str]


_current_recording: contextvars.ContextVar[Optional[_RecordingContext]] = (
    contextvars.ContextVar("mirix_retry_recording", default=None)
)

# Set while retry_call(operation="embedding") runs its call: that layer owns
# the retries of the embedding (3 backoff retries), so the provider
# models' own legacy retry loop (mirix.embeddings.embedding_with_retry) steps
# aside instead of multiplying them (one query embedding under a sustained
# 429 used to make 16 provider calls and wait 70-100 s before BM25).
_embedding_retries_owned: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mirix_embedding_retries_owned", default=False
)


def embedding_retries_owned() -> bool:
    """Whether an enclosing ``retry_call(operation="embedding")`` retries the
    embedding call being made (the model then calls its provider once)."""
    return _embedding_retries_owned.get()


@contextmanager
def recording_retries(
    recorder: Optional[RetryEventRecorder],
    *,
    stage: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Iterator[None]:
    """Record every retry made inside this block with ``recorder``.

    A learning run wraps its distill and curator stages in it, so the retries
    made deep below — a skill write's transaction retries, an embedding's
    backoff, a batch settle, the SDK's transport retries, the agent's own
    retries of an unusable answer — land in the run's ``retry_events``
    without every layer passing a recorder down. ``stage`` / ``session_id``
    label events whose caller named none. The context is copied into tasks
    started inside the block (``asyncio.gather`` / ``create_task``).
    ``recorder=None`` is a no-op block.
    """
    if recorder is None:
        yield
        return
    token = _current_recording.set(_RecordingContext(recorder, stage, session_id))
    try:
        yield
    finally:
        _current_recording.reset(token)


def current_retry_recorder() -> Optional[RetryEventRecorder]:
    """The recorder of the enclosing :func:`recording_retries` block, if any."""
    context = _current_recording.get()
    return context.recorder if context is not None else None


async def record_retry(
    kind: Any,
    attempt: int,
    waited_s: float,
    *,
    stage: Optional[str] = None,
    session_id: Optional[str] = None,
    error: Optional[BaseException] = None,
    recorder: Optional[RetryEventRecorder] = None,
) -> None:
    """Record one retry with ``recorder``, else with the enclosing block's
    recorder (its stage / session as defaults). No recorder: nothing to do.
    Never raises — recording must not break the operation being retried."""
    context = _current_recording.get()
    if recorder is None:
        if context is None:
            return
        recorder = context.recorder
    if context is not None:
        stage = stage or context.stage
        session_id = session_id or context.session_id
    try:
        await recorder.record(
            kind, attempt, waited_s, stage=stage, session_id=session_id, error=error
        )
    except Exception:  # noqa: BLE001 — see the docstring
        logger.warning("Could not record a retry event", exc_info=True)


async def append_retrieval_retry_events(
    retrieval_id: str,
    events: List[Dict[str, Any]],
    *,
    session_maker: Any = None,
) -> bool:
    """Append ``events`` to one retrieval log row's ``retry_events`` (row
    locked, one transaction). False when the row does not exist (yet)."""
    from sqlalchemy import select

    from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogModel

    if not events:
        return True
    if session_maker is None:
        from mirix.server.server import db_context

        session_maker = db_context
    async with session_maker() as db:
        row = (
            await db.execute(
                select(RetrievalLogModel)
                .where(RetrievalLogModel.id == retrieval_id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        row.retry_events = list(row.retry_events or []) + [dict(e) for e in events]
        await db.commit()
        return True


# --------------------------------------------------------------------------- #
# In-run retry                                                                 #
# --------------------------------------------------------------------------- #


async def retry_call(
    fn: Callable[[], Awaitable[T]],
    *,
    operation: Optional[str] = None,
    stage: Optional[str] = None,
    session_id: Optional[str] = None,
    recorder: Optional[RetryEventRecorder] = None,
    policies: Optional[Dict[ErrorKind, RetryPolicy]] = None,
    sleep: Optional[Callable[[float], Awaitable[Any]]] = None,
    jitter: bool = True,
) -> T:
    """Await ``fn()``, retrying the failures whose retry is "call it again".

    Per :data:`RETRY_POLICIES`: a timed-out call is repeated once at once, a
    transient database failure up to 5 times (0.1 s doubling to 3 s), a failed
    embedding call up to 3 times (1 s doubling to 8 s); ``fn`` must therefore
    be safe to repeat (a whole transaction, not half of one). Every other kind
    — and a kind whose retries are used up — is re-raised unchanged for the
    caller (which may shrink its input, add a repair prompt, defer the
    session, or fail the run). Each retry is recorded before its
    ``asyncio.sleep`` wait — with ``recorder``, else with the recorder of the
    enclosing :func:`recording_retries` block. Cancellation always propagates.
    With ``operation="embedding"`` this is the ONLY application-level retry
    of the call: :func:`embedding_retries_owned` tells the embedding models to
    skip their own legacy retry loop while ``fn`` runs.
    """
    policies = policies or RETRY_POLICIES
    sleep = sleep or asyncio.sleep
    used: Dict[ErrorKind, int] = {}
    while True:
        owned = _embedding_retries_owned.set(True) if operation == "embedding" else None
        try:
            return await fn()
        except Exception as exc:
            kind = classify_error(exc, operation=operation)
            # A partial custom table never masks the failure with a KeyError:
            # a kind it leaves out is not retried here.
            policy = policies.get(kind)
            count = used.get(kind, 0)
            if (
                policy is None
                or policy.strategy not in REPEAT_STRATEGIES
                or count >= policy.in_run_retries
            ):
                raise
            used[kind] = count + 1
            delay = policy.delay(count)
            if jitter and delay > 0:
                delay *= 1 - 0.25 * random.random()
            logger.info(
                "%s failed (%s: %s); retry %d/%d in %.2fs",
                stage or getattr(fn, "__name__", "call"),
                kind.value,
                type(exc).__name__,
                count + 1,
                policy.in_run_retries,
                delay,
            )
            await record_retry(
                kind,
                count + 1,
                delay,
                stage=stage,
                session_id=session_id,
                error=exc,
                recorder=recorder,
            )
            if delay > 0:
                await sleep(delay)
        finally:
            if owned is not None:
                _embedding_retries_owned.reset(owned)
