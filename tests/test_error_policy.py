"""Tests for mirix.queue.error_policy.

Covers:
- classify() across every Permanent and Transient class in the mapping table
- Unknown exceptions default to Transient and emit a one-shot warning
- process_with_policy() success / Permanent / Transient-retry / Transient-exhausted paths
- on_permanent callback invocation and error-message shape
- on_permanent failure does not mask the underlying Permanent outcome
"""

from __future__ import annotations

import pytest

from mirix.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMConnectionError,
    LLMPermissionDeniedError,
    LLMRateLimitError,
    LLMServerError,
    LLMUnprocessableEntityError,
)
from mirix.queue import error_policy as ep
from mirix.queue.error_policy import Bucket, Outcome, SaveOutcome, classify, process_with_policy

# ---------- classify() ----------


@pytest.mark.parametrize(
    "exc_cls",
    [
        LLMUnprocessableEntityError,  # 422
        LLMBadRequestError,  # 400
        LLMAuthenticationError,  # 401
        LLMPermissionDeniedError,  # 403
    ],
)
def test_classify_permanent(exc_cls):
    assert classify(exc_cls("boom")) is Bucket.PERMANENT


@pytest.mark.parametrize(
    "exc_cls",
    [
        LLMRateLimitError,  # 429
        LLMServerError,  # 5xx
        LLMConnectionError,  # network
    ],
)
def test_classify_transient(exc_cls):
    assert classify(exc_cls("boom")) is Bucket.TRANSIENT


def test_classify_unknown_defaults_to_transient(caplog):
    # Reset the one-shot dedup set so we can assert the warning fires.
    ep._unknown_warned.discard(RuntimeError)
    with caplog.at_level("WARNING", logger="mirix.queue.error_policy"):
        assert classify(RuntimeError("something weird")) is Bucket.TRANSIENT
    assert any("RuntimeError" in r.message for r in caplog.records)


def test_classify_unknown_warns_once_per_class(caplog):
    """Unknown types that aren't pure-Python bug shapes default to
    TRANSIENT with a one-shot warning. (After VEPAGE-1251 S4, bug shapes
    like KeyError without a provider traceback frame are routed to
    PERMANENT by the origin-split, so they intentionally bypass this
    warning path — see test_classify_db_and_unknown_origin.py.)"""

    class _UnmappedException(Exception):
        pass

    ep._unknown_warned.discard(_UnmappedException)
    with caplog.at_level("WARNING", logger="mirix.queue.error_policy"):
        classify(_UnmappedException("a"))
        classify(_UnmappedException("b"))
    warnings = [r for r in caplog.records if "_UnmappedException" in r.message]
    assert len(warnings) == 1, "expected one-shot warning per class"


# ---------- process_with_policy() ----------


@pytest.mark.asyncio
async def test_process_with_policy_completed_path():
    calls = {"n": 0}

    async def run_step():
        calls["n"] += 1

    out = await process_with_policy(run_step, memory_source_id="src-ok")
    assert out.kind is SaveOutcome.SUCCESS
    assert out.cause is None
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_process_with_policy_permanent_path_invokes_callback_and_short_circuits():
    perm_calls: list[tuple[str, str, str]] = []

    async def on_perm(source_id, error_message, exc):
        perm_calls.append((source_id, error_message, type(exc).__name__))

    async def run_step():
        raise LLMUnprocessableEntityError("content rejected")

    out = await process_with_policy(
        run_step,
        memory_source_id="src-422",
        on_permanent=on_perm,
    )
    assert out.kind is SaveOutcome.PERMANENT_FAILURE
    assert out.bucket is Bucket.PERMANENT
    assert isinstance(out.cause, LLMUnprocessableEntityError)
    # Single attempt — Permanent does not retry.
    assert perm_calls == [("src-422", "LLMUnprocessableEntityError: content rejected", "LLMUnprocessableEntityError")]


@pytest.mark.asyncio
async def test_process_with_policy_permanent_callback_failure_is_swallowed():
    """If on_permanent raises, the Outcome is still PERMANENT_FAILURE.

    We must not let a logging/DB hiccup convert a permanent failure into a
    transient retry — that would re-create the cascade.
    """

    async def on_perm(source_id, error_message, exc):
        raise RuntimeError("DB hiccup writing status")

    async def run_step():
        raise LLMBadRequestError("400 schema")

    out = await process_with_policy(run_step, memory_source_id="src-cb", on_permanent=on_perm)
    assert out.kind is SaveOutcome.PERMANENT_FAILURE
    assert out.bucket is Bucket.PERMANENT


@pytest.mark.asyncio
async def test_process_with_policy_transient_retry_then_success(monkeypatch):
    # Make sleep instant.
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    attempts = {"n": 0}

    async def run_step():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise LLMRateLimitError("429 try again")

    out = await process_with_policy(run_step, memory_source_id="src-flaky")
    assert out.kind is SaveOutcome.SUCCESS
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_process_with_policy_transient_exhausted(monkeypatch):
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)
    # Settings default: whole_step_retry_max_attempts=2 → 3 total attempts
    from mirix.settings import settings as svc_settings

    expected_total = svc_settings.whole_step_retry_max_attempts + 1

    attempts = {"n": 0}

    async def run_step():
        attempts["n"] += 1
        raise LLMRateLimitError("still 429")

    out = await process_with_policy(run_step, memory_source_id="src-tx")
    assert out.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    assert out.bucket is Bucket.TRANSIENT
    assert isinstance(out.cause, LLMRateLimitError)
    assert attempts["n"] == expected_total


@pytest.mark.asyncio
async def test_process_with_policy_unknown_exception_treated_as_transient(monkeypatch):
    """Unknown exceptions walk the Transient path, not the Permanent one.

    Guards against accidentally hiding bugs by classifying them as Permanent
    (which would silently ack the message and never retry).
    """
    monkeypatch.setattr(ep, "_backoff_seconds", lambda *_: 0.0)

    perm_calls: list[str] = []

    async def on_perm(source_id, error_message, exc):
        perm_calls.append(source_id)

    async def run_step():
        raise ValueError("oops, a bug")

    out = await process_with_policy(run_step, memory_source_id="src-bug", on_permanent=on_perm)
    assert out.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    assert perm_calls == [], "on_permanent must not fire for Transient outcomes"


# ---------- Outcome dataclass ----------


def test_outcome_is_frozen():
    out = Outcome(kind=SaveOutcome.SUCCESS)
    with pytest.raises(Exception):
        out.kind = SaveOutcome.PERMANENT_FAILURE  # type: ignore[misc]


# ---------- classify() walks __cause__ ----------


def test_classify_unwraps_cause_to_find_permanent():
    """A Permanent error wrapped in another exception type must still classify
    as Permanent. Defends against any code path that wraps exceptions for
    diagnostic reasons (e.g. trigger_memory_update used to wrap in RuntimeError)."""
    inner = LLMUnprocessableEntityError("content rejected")
    try:
        try:
            raise inner
        except LLMUnprocessableEntityError as e:
            raise RuntimeError("wrapper for diagnostic context") from e
    except RuntimeError as wrapped:
        assert classify(wrapped) is Bucket.PERMANENT


def test_classify_unwraps_cause_to_find_transient():
    """Same property in the Transient direction."""
    inner = LLMRateLimitError("429")
    try:
        try:
            raise inner
        except LLMRateLimitError as e:
            raise RuntimeError("wrapper") from e
    except RuntimeError as wrapped:
        assert classify(wrapped) is Bucket.TRANSIENT


# ---------- format_exc_chain ----------


def test_format_exc_chain_single_exception():
    exc = ValueError("bad value")
    assert ep.format_exc_chain(exc) == "ValueError: bad value"


def test_format_exc_chain_two_levels():
    try:
        try:
            raise LLMUnprocessableEntityError("422 unprocessable")
        except LLMUnprocessableEntityError as inner:
            raise RuntimeError("wrapper failed") from inner
    except RuntimeError as wrapped:
        chain = ep.format_exc_chain(wrapped)
    assert "RuntimeError: wrapper failed" in chain
    assert "LLMUnprocessableEntityError: 422 unprocessable" in chain
    assert chain.index("RuntimeError") < chain.index("LLMUnprocessableEntityError")
    assert " <- " in chain


def test_format_exc_chain_respects_max_depth():
    # Build a 5-link chain manually with __cause__ links
    e1 = ValueError("1")
    e2 = ValueError("2")
    e2.__cause__ = e1
    e3 = ValueError("3")
    e3.__cause__ = e2
    chain = ep.format_exc_chain(e3, max_depth=2)
    # max_depth=2 means only 2 frames included
    assert chain.count("ValueError") == 2


def test_format_exc_chain_handles_cycles():
    # A cause-chain that points back at itself MUST not infinite-loop.
    e = ValueError("self")
    e.__cause__ = e  # pathological but possible if someone screws up
    chain = ep.format_exc_chain(e)
    assert chain == "ValueError: self"  # only one entry, no loop
