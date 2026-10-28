"""Retry primitives.

* the classification table: every error family the distiller, the curator and
  retrieval can meet maps to one ErrorKind;
* the retry policy table as data (in-run retries, counted vs deferred);
* the cross-run backoff: 1 minute, doubling, capped at 1 hour;
* retry_call: repeats only what "call it again" can fix, records every retry,
  waits with asyncio.sleep — the event loop keeps running;
* the retry-event recorder never lets a recording failure break the retry.

(The manager-level behavior — distill_retry_after on the session row, events
on the learning run / retrieval log — is in test_skill_learning_lifecycle.)
"""

import asyncio
import datetime as dt
import json
import time
from types import SimpleNamespace

import anthropic
import asyncpg.exceptions
import httpx
import openai
import pytest
from sqlalchemy import exc as sa_exc

from fakes.gateway_errors import (
    FORBIDDEN_403,
    GATEWAY_LOST_DEADLINE,
    GATEWAY_MODEL_TIMEOUT,
    GATEWAY_MODERATION_403,
)
from mirix import constants, errors
from mirix.services import retry_policy
from mirix.services.retry_policy import (
    RETRY_POLICIES,
    ErrorKind,
    RetryEventRecorder,
    classify_error,
    classify_finish_reason,
    distill_retry_delay,
    next_distill_retry_after,
    raise_if_truncated,
    retry_call,
)

REQUEST = httpx.Request("POST", "https://api.example/v1/chat/completions")


def _response(status, body=None):
    return httpx.Response(status, request=REQUEST, json=body or {"error": {}})


def _chained(outer, cause):
    try:
        try:
            raise cause
        except BaseException as inner:
            raise outer from inner
    except BaseException as exc:
        return exc


class _DBAPIOrig(Exception):
    def __init__(self, message, sqlstate=None):
        super().__init__(message)
        self.sqlstate = sqlstate


def _db_error(message, sqlstate=None, cls=sa_exc.OperationalError, invalidated=False):
    return cls(
        "SELECT 1",
        {},
        _DBAPIOrig(message, sqlstate),
        connection_invalidated=invalidated,
    )


# --------------------------------------------------------------------------- #
# Classification table                                                         #
# --------------------------------------------------------------------------- #

CASES = [
    # --- MIRIX's provider-neutral LLM errors -------------------------------
    ("LLMRateLimitError", errors.LLMRateLimitError("429"), ErrorKind.INFRA),
    ("LLMServerError", errors.LLMServerError("503"), ErrorKind.INFRA),
    ("LLMConnectionError", errors.LLMConnectionError("reset"), ErrorKind.INFRA),
    ("RateLimitExceededError", errors.RateLimitExceededError("x", 3), ErrorKind.INFRA),
    ("LLMTimeoutError", errors.LLMTimeoutError("slow"), ErrorKind.TIMEOUT),
    ("LLMAuthenticationError", errors.LLMAuthenticationError("401"), ErrorKind.CONFIG),
    (
        "LLMPermissionDeniedError",
        errors.LLMPermissionDeniedError("403"),
        ErrorKind.CONFIG,
    ),
    ("LLMNotFoundError", errors.LLMNotFoundError("no model"), ErrorKind.CONFIG),
    (
        "LLMBadRequestError context",
        errors.LLMBadRequestError("This model's maximum context length is 8192"),
        ErrorKind.BUDGET,
    ),
    (
        "LLMBadRequestError other",
        errors.LLMBadRequestError("bad enum"),
        ErrorKind.OTHER,
    ),
    (
        "LLMBadRequestError 413",
        errors.LLMBadRequestError("too big", details={"status_code": 413}),
        ErrorKind.BUDGET,
    ),
    # --- budget / truncation / parse / embedding ---------------------------
    (
        "ContextBudgetExceeded",
        errors.ContextBudgetExceeded(estimated_input_tokens=9, output_cap=1),
        ErrorKind.BUDGET,
    ),
    (
        "ContextWindowExceededError",
        errors.ContextWindowExceededError("prompt is too long"),
        ErrorKind.BUDGET,
    ),
    ("LLMOutputTruncatedError", errors.LLMOutputTruncatedError(), ErrorKind.TRUNCATED),
    ("LLMJSONParsingError", errors.LLMJSONParsingError(), ErrorKind.PARSE),
    (
        "JSONDecodeError",
        json.JSONDecodeError("Expecting value", "x", 0),
        ErrorKind.PARSE,
    ),
    ("EmbeddingCallError", errors.EmbeddingCallError(), ErrorKind.EMBEDDING),
    # --- MIRIX's own failure: never walked past, whatever it wraps ------
    (
        "build error wrapping JSONDecodeError",
        _chained(
            errors.LLMRequestBuildError("build failed"),
            json.JSONDecodeError("Expecting value", "x", 0),
        ),
        ErrorKind.INTERNAL,
    ),
    (
        "build error wrapping TimeoutError",
        _chained(errors.LLMRequestBuildError("build failed"), TimeoutError("x")),
        ErrorKind.INTERNAL,
    ),
    (
        "build error wrapping FileNotFoundError",
        _chained(
            errors.LLMRequestBuildError("build failed"),
            FileNotFoundError(2, "No such file", "/images/gone.png"),
        ),
        ErrorKind.INTERNAL,
    ),
    (
        "sdk_call error",
        _chained(
            errors.LLMRequestBuildError("sdk refused", stage="sdk_call"),
            TypeError("unexpected keyword argument"),
        ),
        ErrorKind.INTERNAL,
    ),
    (
        "build error wrapping a network failure",
        _chained(
            errors.LLMRequestBuildError("build failed"),
            httpx.ConnectError("refused", request=REQUEST),
        ),
        ErrorKind.INFRA,
    ),
    (
        "build error wrapping a transient database error",
        _chained(
            errors.LLMRequestBuildError("build failed"),
            _db_error("deadlock detected", "40P01"),
        ),
        ErrorKind.DB,
    ),
    (
        "LearningRunInternalError",
        errors.LearningRunInternalError("1 session not distilled"),
        ErrorKind.INTERNAL,
    ),
    # --- OpenAI / Anthropic SDK errors (a caller that bypassed a client) ---
    (
        "openai read timeout",
        _chained(openai.APITimeoutError(request=REQUEST), httpx.ReadTimeout("x")),
        ErrorKind.TIMEOUT,
    ),
    (
        "openai connect timeout",
        _chained(openai.APITimeoutError(request=REQUEST), httpx.ConnectTimeout("x")),
        ErrorKind.INFRA,
    ),
    ("openai connection", openai.APIConnectionError(request=REQUEST), ErrorKind.INFRA),
    (
        "openai 429",
        openai.RateLimitError("rl", response=_response(429), body=None),
        ErrorKind.INFRA,
    ),
    (
        "openai 500",
        openai.InternalServerError("down", response=_response(500), body=None),
        ErrorKind.INFRA,
    ),
    (
        "openai 408",
        openai.APIStatusError("timeout", response=_response(408), body=None),
        ErrorKind.INFRA,
    ),
    (
        "openai 401",
        openai.AuthenticationError("key", response=_response(401), body=None),
        ErrorKind.CONFIG,
    ),
    (
        "openai 400 context",
        openai.BadRequestError(
            "Error code: 400 - context_length_exceeded",
            response=_response(400),
            body=None,
        ),
        ErrorKind.BUDGET,
    ),
    (
        "openai 400",
        openai.BadRequestError("invalid tool", response=_response(400), body=None),
        ErrorKind.OTHER,
    ),
    (
        "anthropic 529",
        anthropic.APIStatusError("overloaded", response=_response(529), body=None),
        ErrorKind.INFRA,
    ),
    (
        "anthropic timeout",
        _chained(anthropic.APITimeoutError(request=REQUEST), httpx.ReadTimeout("x")),
        ErrorKind.TIMEOUT,
    ),
    # --- raw httpx ----------------------------------------------------------
    (
        "httpx 429",
        httpx.HTTPStatusError("x", request=REQUEST, response=_response(429)),
        ErrorKind.INFRA,
    ),
    (
        "httpx 503",
        httpx.HTTPStatusError("x", request=REQUEST, response=_response(503)),
        ErrorKind.INFRA,
    ),
    (
        "httpx 403",
        httpx.HTTPStatusError("x", request=REQUEST, response=_response(403)),
        ErrorKind.CONFIG,
    ),
    ("httpx read timeout", httpx.ReadTimeout("x", request=REQUEST), ErrorKind.TIMEOUT),
    (
        "httpx connect timeout",
        httpx.ConnectTimeout("x", request=REQUEST),
        ErrorKind.INFRA,
    ),
    ("httpx connect error", httpx.ConnectError("x", request=REQUEST), ErrorKind.INFRA),
    (
        "httpx protocol error",
        httpx.RemoteProtocolError("x", request=REQUEST),
        ErrorKind.INFRA,
    ),
    # --- asyncio / OS -------------------------------------------------------
    ("asyncio timeout", asyncio.TimeoutError(), ErrorKind.TIMEOUT),
    ("builtin timeout", TimeoutError("x"), ErrorKind.TIMEOUT),
    ("connection reset", ConnectionResetError("x"), ErrorKind.INFRA),
    # --- databases ----------------------------------------------------------
    ("pg deadlock", _db_error("deadlock detected", "40P01"), ErrorKind.DB),
    ("pg serialization", _db_error("could not serialize", "40001"), ErrorKind.DB),
    ("pg connection", _db_error("server closed", "08006"), ErrorKind.DB),
    ("pg admin shutdown", _db_error("terminating", "57P01"), ErrorKind.DB),
    ("sqlite locked", _db_error("database is locked"), ErrorKind.DB),
    ("invalidated connection", _db_error("gone", invalidated=True), ErrorKind.DB),
    ("pool timeout", sa_exc.TimeoutError("QueuePool limit"), ErrorKind.DB),
    (
        "asyncpg deadlock",
        asyncpg.exceptions.DeadlockDetectedError("deadlock"),
        ErrorKind.DB,
    ),
    (
        "unique violation",
        _db_error("duplicate key", "23505", cls=sa_exc.IntegrityError),
        ErrorKind.OTHER,
    ),
    # --- a moderation refusal -------------------------------------
    (
        "openai 403 moderation",
        openai.PermissionDeniedError(
            "Error code: 403",
            response=_response(403),
            body={"code": 403, "message": "m requires moderation on P"},
        ),
        ErrorKind.REFUSED,
    ),
    # --- the rest -----------------------------------------------------------
    ("ValueError", ValueError("bug"), ErrorKind.OTHER),
    (
        "wrapped infra",
        _chained(RuntimeError("step failed"), errors.LLMRateLimitError("429")),
        ErrorKind.INFRA,
    ),
]


@pytest.mark.parametrize(
    "error, kind", [(e, k) for _, e, k in CASES], ids=[name for name, _, _ in CASES]
)
def test_classification_table(error, kind):
    assert classify_error(error) is kind


def test_every_kind_is_classified_by_the_table():
    assert {kind for _, _, kind in CASES} == set(ErrorKind)


def _cause_chain(*errors_in_order):
    """The first error, each one ``__cause__`` of the one before it."""
    for outer, inner in zip(errors_in_order, errors_in_order[1:]):
        outer.__cause__ = inner
    return errors_in_order[0]


@pytest.mark.parametrize(
    "cause, kind",
    [
        (KeyError("image"), ErrorKind.INTERNAL),
        (TimeoutError("reading the attachment"), ErrorKind.INTERNAL),
        (httpx.ConnectError("refused", request=REQUEST), ErrorKind.INFRA),
        (_db_error("deadlock detected", "40P01"), ErrorKind.DB),
        (
            _cause_chain(
                RuntimeError("fetching the image failed"),
                ConnectionResetError(54, "Connection reset by peer"),
            ),
            ErrorKind.INFRA,
        ),
        (
            _cause_chain(
                ValueError("no image"),
                RuntimeError("download failed"),
                asyncpg.exceptions.DeadlockDetectedError("deadlock detected"),
            ),
            ErrorKind.DB,
        ),
        (
            _cause_chain(RuntimeError("bad state"), KeyError("image")),
            ErrorKind.INTERNAL,
        ),
        (  # a bind parameter naming a network class is no network failure
            sa_exc.IntegrityError(
                "INSERT INTO t VALUES (?)", ("httpx.ReadTimeout",), _DBAPIOrig("UNIQUE")
            ),
            ErrorKind.INTERNAL,
        ),
    ],
    ids=[
        "key",
        "timeout",
        "connect",
        "deadlock",
        "runtime_from_reset",
        "three_levels_to_deadlock",
        "runtime_from_key",
        "bind_names_httpx",
    ],
)
def test_a_request_build_error_records_the_verdict_on_its_whole_chain(cause, kind):
    """A request MIRIX could not build carries the retry policy's verdict on what it
    wraps, the whole chain, in its text (``[kind=...]``, before the cause's own words):
    whoever reads only the recorded text of a learning run's or a retrieval's error
    sees what ``classify_error`` decided, not a guess from the one cause it names."""
    assert retry_policy.classify_build_cause(cause) is kind
    error = errors.LLMRequestBuildError.wrapping(
        "building the OpenAIClient request", cause, stage="build"
    )
    assert error.__cause__ is cause and error.stage == "build"
    assert classify_error(error) is kind
    assert error.kind == kind.value and error.details["kind"] == kind.value
    assert str(error).startswith(
        f"building the OpenAIClient request failed [kind={kind.value}] "
        f"({type(cause).__name__}: "
    )


def test_a_request_build_error_whose_verdict_fails_keeps_the_original_error(
    monkeypatch,
):
    """Labelling the error must never replace it: a classification that raises leaves
    ``[kind=unknown]`` and the cause intact (a reader then judges the cause itself)."""

    def broken(cause):
        raise RuntimeError("classifier bug")

    monkeypatch.setattr(retry_policy, "classify_build_cause", broken)
    cause = KeyError("image")
    error = errors.LLMRequestBuildError.wrapping(
        "building the OpenAIClient request", cause, stage="build"
    )
    assert error.kind == "unknown" and error.__cause__ is cause
    assert str(error).startswith(
        "building the OpenAIClient request failed [kind=unknown] (KeyError: "
    )


class _SqliteOrig(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.sqlite_errorcode = code


def test_user_content_in_bind_parameters_never_makes_a_db_error_transient():
    # str() of a SQLAlchemy error carries the statement and its bind
    # parameters — the experience or skill text being written. Only the
    # driver's own error decides.
    text = "tip: if sqlite says database is locked, retry; deadlock detected"
    duplicate = sa_exc.IntegrityError(
        "INSERT INTO skill_experience ...", {"content": text}, _DBAPIOrig("UNIQUE")
    )
    assert text in str(duplicate)
    assert classify_error(duplicate) is ErrorKind.OTHER
    bad_data = sa_exc.DataError(
        "INSERT ...", {"content": text}, _DBAPIOrig("invalid byte 0x00", "22021")
    )
    assert classify_error(bad_data) is ErrorKind.OTHER
    no_table = sa_exc.OperationalError(
        "SELECT ...", {"q": text}, _SqliteOrig("no such table: skills", 1)
    )
    assert classify_error(no_table) is ErrorKind.OTHER
    plain_other = sa_exc.OperationalError(
        "UPDATE ...", {"q": text}, _DBAPIOrig("disk I/O error")
    )
    assert classify_error(plain_other) is ErrorKind.OTHER


def test_the_driver_error_decides_transience():
    busy = sa_exc.OperationalError(
        "UPDATE ...", {}, _SqliteOrig("database is locked", 5)
    )
    locked = sa_exc.OperationalError("UPDATE ...", {}, _SqliteOrig("table locked", 262))
    assert classify_error(busy) is ErrorKind.DB
    assert classify_error(locked) is ErrorKind.DB  # SQLITE_LOCKED_SHAREDCACHE
    # An integrity error is never transient, whatever its driver says.
    assert (
        classify_error(
            sa_exc.IntegrityError("INSERT ...", {}, _DBAPIOrig("deadlock detected"))
        )
        is ErrorKind.OTHER
    )


def test_a_transient_embedding_failure_is_an_embedding_failure():
    for error in (
        errors.LLMServerError("503"),
        httpx.ConnectError("x", request=REQUEST),
        httpx.ReadTimeout("x", request=REQUEST),
        errors.EmbeddingCallError(),
        json.JSONDecodeError("garbled", "<html>", 0),
    ):
        assert classify_error(error, operation="embedding") is ErrorKind.EMBEDDING


def test_a_permanent_embedding_failure_keeps_its_kind():
    # A revoked key, an unknown model or a bug fails the same way again;
    # it is not retried with the embedding backoff.
    assert (
        classify_error(errors.LLMAuthenticationError("401"), operation="embedding")
        is ErrorKind.CONFIG
    )
    assert (
        classify_error(
            httpx.HTTPStatusError("403", request=REQUEST, response=_response(403)),
            operation="embedding",
        )
        is ErrorKind.CONFIG
    )
    assert classify_error(ValueError("x"), operation="embedding") is ErrorKind.OTHER
    assert classify_error(TypeError("bug"), operation="embedding") is ErrorKind.OTHER


def test_cancellation_is_never_classified():
    with pytest.raises(TypeError):
        classify_error(asyncio.CancelledError())


def test_finish_reason():
    assert classify_finish_reason("length") is ErrorKind.TRUNCATED
    assert classify_finish_reason("MAX_TOKENS") is ErrorKind.TRUNCATED
    assert classify_finish_reason("stop") is None
    assert classify_finish_reason(None) is None
    truncated = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    with pytest.raises(errors.LLMOutputTruncatedError):
        raise_if_truncated(truncated)
    raise_if_truncated(SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]))


# --------------------------------------------------------------------------- #
# The retry table                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kind, strategy, retries, counted, deferred",
    [
        # infrastructure: transport retries, then back off across runs, not counted
        (ErrorKind.INFRA, "transport", 0, False, True),
        (ErrorKind.CONFIG, "none", 0, False, True),
        # truncation / budget: halve the chunk budget up to 3 times, then count
        (ErrorKind.TRUNCATED, "halve_chunk_budget", 3, True, False),
        (ErrorKind.BUDGET, "halve_chunk_budget", 3, True, False),
        # parse failure: up to 2 repair prompts, then count
        (ErrorKind.PARSE, "repair_prompt", 2, True, False),
        # call timeout: once more as is, then count
        (ErrorKind.TIMEOUT, "retry_same", 1, True, False),
        # database: the whole transaction up to 5 times; then the run fails
        (ErrorKind.DB, "retry_transaction", 5, False, False),
        # embedding: 3 retries with backoff; never counted
        (ErrorKind.EMBEDDING, "backoff", 3, False, False),
        # a moderation refusal: the same input is refused again; counted
        (ErrorKind.REFUSED, "none", 0, True, False),
        (ErrorKind.OTHER, "none", 0, True, False),
    ],
)
def test_policy_table(kind, strategy, retries, counted, deferred):
    policy = RETRY_POLICIES[kind]
    assert policy.strategy == strategy
    assert policy.in_run_retries == retries
    assert policy.counts_as_failure is counted
    assert policy.defer_across_runs is deferred


def test_in_run_delays():
    db = RETRY_POLICIES[ErrorKind.DB]
    assert [db.delay(i) for i in range(6)] == [0.1, 0.2, 0.4, 0.8, 1.6, 3.0]
    embedding = RETRY_POLICIES[ErrorKind.EMBEDDING]
    assert [embedding.delay(i) for i in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert RETRY_POLICIES[ErrorKind.TIMEOUT].delay(0) == 0.0


# --------------------------------------------------------------------------- #
# Cross-run backoff                                                            #
# --------------------------------------------------------------------------- #


def test_backoff_doubles_from_one_minute_to_one_hour():
    assert constants.DISTILL_RETRY_BACKOFF_MIN_SECONDS == 60
    assert constants.DISTILL_RETRY_BACKOFF_MAX_SECONDS == 3600
    assert [distill_retry_delay(n) for n in range(1, 9)] == [
        60,
        120,
        240,
        480,
        960,
        1920,
        3600,
        3600,
    ]
    assert distill_retry_delay(10_000) == 3600  # no overflow
    assert distill_retry_delay(0) == 60 and distill_retry_delay(-3) == 60


def test_backoff_bounds_are_configurable(monkeypatch):
    monkeypatch.setattr(constants, "DISTILL_RETRY_BACKOFF_MIN_SECONDS", 5)
    monkeypatch.setattr(constants, "DISTILL_RETRY_BACKOFF_MAX_SECONDS", 30)
    assert [distill_retry_delay(n) for n in range(1, 6)] == [5, 10, 20, 30, 30]
    assert distill_retry_delay(3, min_seconds=1, max_seconds=100) == 4
    # A max below the min is raised to the min.
    assert distill_retry_delay(4, min_seconds=50, max_seconds=10) == 50


def test_next_retry_after():
    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
    assert next_distill_retry_after(now, 3) == now + dt.timedelta(minutes=4)


def test_settings_are_validated():
    """How the settings parse (the module constants use these helpers)."""
    assert constants._positive_int_setting("X", "-4", 60) == 60  # invalid -> default
    assert constants._positive_int_setting("X", "90", 60) == 90
    assert (
        constants._non_negative_int_setting("X", "0", 2) == 0
    )  # eval: gateway retries
    assert constants._non_negative_int_setting("X", "-1", 2) == 2
    assert constants._non_negative_int_setting("X", "many", 2) == 2
    assert constants._positive_float_setting("X", "zero", 600.0) == 600.0
    assert constants._positive_float_setting("X", "0", 600.0) == 600.0
    assert constants._positive_float_setting("X", "12.5", 600.0) == 12.5
    assert constants._positive_float_setting("X", None, 600.0) == 600.0
    # The maximum backoff is never below the minimum.
    assert (
        constants.DISTILL_RETRY_BACKOFF_MAX_SECONDS
        >= constants.DISTILL_RETRY_BACKOFF_MIN_SECONDS
    )


# --------------------------------------------------------------------------- #
# retry_call                                                                   #
# --------------------------------------------------------------------------- #


class _Flaky:
    def __init__(self, *failures, result="ok"):
        self.failures = list(failures)
        self.calls = 0
        self.result = result

    async def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.result


@pytest.mark.asyncio
class TestRetryCall:
    async def test_a_timeout_is_retried_once_at_once(self):
        slept = []

        async def sleep(seconds):
            slept.append(seconds)

        fn = _Flaky(errors.LLMTimeoutError("slow"))
        assert await retry_call(fn, sleep=sleep) == "ok"
        assert fn.calls == 2 and slept == []
        fn = _Flaky(errors.LLMTimeoutError("slow"), errors.LLMTimeoutError("again"))
        with pytest.raises(errors.LLMTimeoutError):
            await retry_call(fn, sleep=sleep)
        assert fn.calls == 2

    async def test_a_kind_missing_from_a_custom_table_is_raised_unchanged(self):
        # A partial policy table must never mask the real failure with a
        # KeyError: the kind it leaves out is simply not retried.
        only_timeouts = {ErrorKind.TIMEOUT: RETRY_POLICIES[ErrorKind.TIMEOUT]}
        fn = _Flaky(errors.LLMRateLimitError("slow down"))
        with pytest.raises(errors.LLMRateLimitError):
            await retry_call(fn, policies=only_timeouts)
        assert fn.calls == 1
        fn = _Flaky(errors.LLMTimeoutError("slow"))
        assert await retry_call(fn, policies=only_timeouts) == "ok"
        assert fn.calls == 2

    async def test_db_failures_retry_the_transaction_five_times(self):
        slept = []

        async def sleep(seconds):
            slept.append(seconds)

        deadlock = _db_error("deadlock detected", "40P01")
        fn = _Flaky(*[deadlock] * 5)
        assert await retry_call(fn, sleep=sleep, jitter=False) == "ok"
        assert fn.calls == 6
        assert slept == [0.1, 0.2, 0.4, 0.8, 1.6]
        fn = _Flaky(*[deadlock] * 6)
        with pytest.raises(sa_exc.OperationalError):
            await retry_call(fn, sleep=sleep, jitter=False)
        assert fn.calls == 6

    async def test_embedding_failures_back_off_three_times(self):
        slept = []

        async def sleep(seconds):
            slept.append(seconds)

        fn = _Flaky(*[errors.LLMServerError("502 bad gateway")] * 3)
        assert (
            await retry_call(fn, operation="embedding", sleep=sleep, jitter=False)
            == "ok"
        )
        assert slept == [1.0, 2.0, 4.0]

    @pytest.mark.parametrize(
        "error",
        [
            errors.LLMAuthenticationError("401 revoked key"),
            errors.LLMNotFoundError("404 unknown model"),
            TypeError("bug"),
            errors.MirixConfigurationError("no OpenRouter key"),
        ],
    )
    async def test_a_permanent_embedding_failure_is_not_retried(self, error):
        slept = []

        async def sleep(seconds):
            slept.append(seconds)

        fn = _Flaky(error)
        with pytest.raises(type(error)):
            await retry_call(fn, operation="embedding", sleep=sleep, jitter=False)
        assert fn.calls == 1 and slept == []

    @pytest.mark.parametrize(
        "error",
        [
            errors.LLMRateLimitError("429"),  # the transport already retried
            errors.ContextBudgetExceeded(),  # the caller shrinks the input
            errors.LLMOutputTruncatedError(),
            errors.LLMJSONParsingError(),  # the caller adds a repair prompt
            errors.LLMAuthenticationError("401"),
            ValueError("bug"),
        ],
    )
    async def test_everything_else_goes_back_to_the_caller(self, error):
        fn = _Flaky(error)
        with pytest.raises(type(error)):
            await retry_call(fn)
        assert fn.calls == 1

    async def test_every_retry_is_recorded_before_its_wait(self):
        order = []

        async def sink(events):
            order.extend(("recorded", e["attempt"]) for e in events)
            return True

        async def sleep(seconds):
            order.append(("slept", seconds))

        recorder = RetryEventRecorder(sink)
        fn = _Flaky(*[_db_error("database is locked")] * 2)
        await retry_call(
            fn,
            recorder=recorder,
            stage="commit",
            session_id="s-9",
            sleep=sleep,
            jitter=False,
        )
        assert order == [
            ("recorded", 1),
            ("slept", 0.1),
            ("recorded", 2),
            ("slept", 0.2),
        ]
        assert [(e["kind"], e["attempt"], e["waited_s"]) for e in recorder.events] == [
            ("db", 1, 0.1),
            ("db", 2, 0.2),
        ]
        assert recorder.events[0]["stage"] == "commit"
        assert recorder.events[0]["session_id"] == "s-9"
        assert "database is locked" in recorder.events[0]["error"]

    async def test_cancellation_propagates(self):
        async def cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await retry_call(cancelled)

    async def test_waits_never_block_the_event_loop(self, monkeypatch):
        """Real asyncio.sleep waits: a concurrent task keeps running, and the
        blocking time.sleep is never used."""

        def forbidden(seconds):
            raise AssertionError("time.sleep would block the event loop")

        monkeypatch.setattr(time, "sleep", forbidden)
        monkeypatch.setitem(
            RETRY_POLICIES,
            ErrorKind.DB,
            retry_policy.RetryPolicy(
                ErrorKind.DB,
                retry_policy.STRATEGY_RETRY_TRANSACTION,
                in_run_retries=3,
                counts_as_failure=False,
                defer_across_runs=False,
                base_delay=0.05,
                max_delay=0.05,
            ),
        )
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        task = asyncio.create_task(ticker())
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            fn = _Flaky(*[_db_error("database is locked")] * 3)
            assert await retry_call(fn, jitter=False) == "ok"
        finally:
            task.cancel()
        assert loop.time() - started >= 0.15
        assert ticks >= 10


@pytest.mark.asyncio
class TestRecorder:
    async def test_without_a_sink_events_stay_in_memory(self):
        recorder = RetryEventRecorder()
        event = await recorder.record(ErrorKind.EMBEDDING, 1, 1.0, stage="embed")
        assert event["kind"] == "embedding" and recorder.events == [event]
        assert recorder.unsent == [event] and await recorder.flush() is False

    async def test_a_failing_sink_never_breaks_the_retry_and_resends(self):
        calls = []

        async def flaky_sink(events):
            calls.append([e["attempt"] for e in events])
            if len(calls) == 1:
                raise RuntimeError("database down")
            return True

        recorder = RetryEventRecorder(flaky_sink)
        await recorder.record("infra", 1, 60)  # the sink raises: logged only
        assert len(recorder.unsent) == 1
        await recorder.record("infra", 2, 120)  # both are sent now
        assert calls == [[1], [1, 2]]
        assert recorder.unsent == []

    async def test_errors_are_truncated(self):
        recorder = RetryEventRecorder()
        event = await recorder.record("other", 1, 0, error=ValueError("x" * 5000))
        assert len(event["error"]) == 500

    async def test_unknown_kinds_are_rejected(self):
        with pytest.raises(ValueError):
            await RetryEventRecorder().record("not-a-kind", 1, 0)


# --------------------------------------------------------------------------- #
# A proxy's verdict on its own deadline; a moderation refusal                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status, body, kind",
    [
        (504, GATEWAY_MODEL_TIMEOUT, ErrorKind.TIMEOUT),
        (504, GATEWAY_LOST_DEADLINE, ErrorKind.INFRA),
        (504, {"error": {"message": "upstream timed out"}}, ErrorKind.INFRA),
        (403, GATEWAY_MODERATION_403, ErrorKind.REFUSED),
        (403, FORBIDDEN_403, ErrorKind.CONFIG),
    ],
    ids=["model-timeout", "lost-deadline", "plain-504", "moderation", "forbidden"],
)
def test_a_proxys_verdict_and_a_moderation_refusal_on_raw_sdk_errors(
    status, body, kind
):
    response = httpx.Response(status, request=REQUEST, json=body)
    # What the OpenAI client builds: ``body`` is the answer's ``error`` object.
    error = openai.APIStatusError(
        f"Error code: {status} - {body}", response=response, body=body["error"]
    )
    assert classify_error(error) is kind
    wrapped = errors.LLMServerError(
        f"OpenAI API error: {error}",
        details={"status_code": status, "body": body["error"]},
    )
    if status == 403:
        wrapped = errors.LLMPermissionDeniedError(
            f"Permission denied by OpenAI: {error}", details=body["error"]
        )
    assert classify_error(wrapped) is kind


@pytest.mark.asyncio
class TestThroughTheOpenAIClient:
    """The production OpenAI-compatible client against the offline fake server,
    which answers the way the eval gateway does: the retry-policy kind
    of what ``send_llm_request`` raises."""

    @pytest.fixture(autouse=True)
    def _direct(self, monkeypatch):
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
        monkeypatch.setattr(constants, "LLM_MAX_RETRIES", 0)

    @pytest.mark.parametrize(
        "status, body, kind",
        [
            (504, GATEWAY_MODEL_TIMEOUT, ErrorKind.TIMEOUT),
            (504, GATEWAY_LOST_DEADLINE, ErrorKind.INFRA),
            (403, GATEWAY_MODERATION_403, ErrorKind.REFUSED),
            (403, FORBIDDEN_403, ErrorKind.CONFIG),
        ],
        ids=["model-timeout", "lost-deadline", "moderation", "forbidden"],
    )
    async def test_the_kind_of_what_the_client_raises(
        self, fake_llm_server, status, body, kind
    ):
        from fakes.fake_llm_server import Fault
        from mirix.llm_api.llm_client import LLMClient
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.llm_config import LLMConfig
        from mirix.schemas.message import Message
        from mirix.schemas.mirix_message_content import TextContent

        fake_llm_server.inject_fault(Fault(status=status, body=body))
        client = LLMClient.create(
            LLMConfig(
                model="fake-model",
                model_endpoint_type="openai",
                model_endpoint=fake_llm_server.base_url,
                context_window=200_000,
                max_output_tokens=32_000,
                api_key="fake-key",
            )
        )
        with pytest.raises(errors.LLMError) as raised:
            await client.send_llm_request(
                messages=[
                    Message(role=MessageRole.user, content=[TextContent(text="hi")])
                ]
            )
        assert classify_error(raised.value) is kind
        assert len(fake_llm_server.chat_requests()) == 1


def test_a_timeout_verdict_needs_the_proxys_word_for_it():
    # Only ``blame: model`` on a ``deadline_exceeded`` body: a provider's own
    # 504 that happens to mention a deadline stays infrastructure.
    for error in (
        {**GATEWAY_MODEL_TIMEOUT["error"], "blame": "infra"},
        {**GATEWAY_MODEL_TIMEOUT["error"], "type": "upstream_error"},
        {k: v for k, v in GATEWAY_MODEL_TIMEOUT["error"].items() if k != "blame"},
    ):
        wrapped = errors.LLMServerError(
            "x", details={"status_code": 504, "body": error}
        )
        assert classify_error(wrapped) is ErrorKind.INFRA


def test_aiohttp_network_failures_are_infrastructure():
    # google-genai runs its async calls on aiohttp when it is
    # installed (the google_ai embedding type).
    aiohttp = pytest.importorskip("aiohttp")
    for error in (
        aiohttp.ServerDisconnectedError(),
        aiohttp.ClientPayloadError("cut"),
        aiohttp.ClientConnectionError("refused"),
    ):
        assert classify_error(error) is ErrorKind.INFRA
        assert classify_error(error, operation="embedding") is ErrorKind.EMBEDDING
    timed_out = aiohttp.ServerTimeoutError("slow")
    assert classify_error(timed_out) is ErrorKind.TIMEOUT
    assert classify_error(timed_out, operation="embedding") is ErrorKind.EMBEDDING
