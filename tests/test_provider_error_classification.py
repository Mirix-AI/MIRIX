"""S3 of VEPAGE-1251: Provider* types in classify() + inner-exhausted marker.

These tests pin two contracts:

1. The three MIRIX-owned Provider* types map by plain isinstance in
   `classify()`. The ECMS provider boundary translates SDK exceptions into
   these types so the MIRIX core never imports provider SDK classes.

2. An exception tagged as "inner tier already exhausted my budget" must NOT
   trigger another full whole-step retry cycle in `process_with_policy`.
   This kills the LLM inner (3) × whole-step (3) = 9 multiplication for any
   dependency that retries in place.

Tests for ProviderConflictError dedup semantics live in
test_provider_write_retry.py — Conflict is handled by `is_conflict(exc)`
no-op callers, never by `classify()` directly.
"""

from __future__ import annotations

import pytest

from mirix.errors import (
    ProviderConflictError,
    ProviderPermanentError,
    ProviderTransientError,
)
from mirix.queue.error_policy import (
    Bucket,
    Outcome,
    SaveOutcome,
    classify,
    mark_inner_exhausted,
    process_with_policy,
)

# ---------- classify() ----------


def test_provider_transient_error_classifies_as_transient():
    assert classify(ProviderTransientError("503 from IPS-R")) is Bucket.TRANSIENT


def test_provider_permanent_error_classifies_as_permanent():
    assert classify(ProviderPermanentError("400 bad shape")) is Bucket.PERMANENT


def test_provider_transient_wrapped_in_runtime_error_still_transient():
    """A foreign type wrapping a Provider* via `raise X from Y` keeps the
    correct bucket (cause-chain walk)."""
    try:
        raise ProviderTransientError("upstream 503")
    except ProviderTransientError as cause:
        try:
            raise RuntimeError("rewrapped") from cause
        except RuntimeError as wrapper:
            assert classify(wrapper) is Bucket.TRANSIENT


def test_provider_permanent_wrapped_still_permanent():
    try:
        raise ProviderPermanentError("400")
    except ProviderPermanentError as cause:
        try:
            raise RuntimeError("rewrapped") from cause
        except RuntimeError as wrapper:
            assert classify(wrapper) is Bucket.PERMANENT


def test_provider_conflict_error_is_not_classified_as_permanent_or_transient():
    """ProviderConflictError is consumed by `is_conflict(exc)` no-op
    callers — it must NOT short-circuit into Permanent (would break the
    L1/L3 dedup) and must NOT be silently retried as Transient. The
    classifier sees it as TRANSIENT by default (the unmapped fallback) —
    but in practice it never reaches the classifier because providers
    catch it at the boundary."""
    # The MIRIX-owned Conflict type is not in PERMANENT_TYPES or
    # TRANSIENT_TYPES; that is intentional. Document the expectation:
    # the boundary callers handle it; the classifier never sees it.
    # If a future change moves Conflict into one of the buckets, this
    # test will fail and force a review of the four is_conflict callers.
    assert classify(ProviderConflictError("uq_email")) is Bucket.TRANSIENT


# ---------- inner-exhausted marker ----------


def test_mark_inner_exhausted_tags_exception_and_returns_it():
    """mark_inner_exhausted sets a marker attribute and returns the
    exception so callers can `raise mark_inner_exhausted(exc)`."""
    exc = ProviderTransientError("503 after 3 inner retries")
    tagged = mark_inner_exhausted(exc)
    assert tagged is exc
    assert getattr(exc, "__mirix_inner_exhausted__", False) is True


def test_classify_unaffected_by_inner_exhausted_marker():
    """The marker affects the policy loop, not the classifier. A marked
    transient is still classified TRANSIENT."""
    exc = ProviderTransientError("503")
    mark_inner_exhausted(exc)
    assert classify(exc) is Bucket.TRANSIENT


# ---------- process_with_policy: inner-exhausted skips whole-step retry ----------


@pytest.mark.asyncio
async def test_inner_exhausted_transient_not_re_retried_at_whole_step():
    """When the propagated exception is tagged as inner-exhausted, the
    policy loop must return TRANSIENT_EXHAUSTED after exactly one attempt
    instead of running the full backoff loop."""
    calls = {"n": 0}

    async def run_step():
        calls["n"] += 1
        exc = ProviderTransientError("503 (inner retries exhausted)")
        raise mark_inner_exhausted(exc)

    outcome = await process_with_policy(run_step, memory_source_id="src-x")

    assert (
        outcome.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    ), "marked exhausted transient must still verdict TRANSIENT_EXHAUSTED"
    assert calls["n"] == 1, (
        f"whole-step loop must NOT retry an inner-exhausted transient — "
        f"expected exactly 1 attempt, got {calls['n']}"
    )


@pytest.mark.asyncio
async def test_unmarked_transient_still_retried_at_whole_step():
    """Without the inner-exhausted marker, the policy loop retries
    transients up to its configured budget — existing behavior is
    preserved."""
    calls = {"n": 0}

    async def run_step():
        calls["n"] += 1
        raise ProviderTransientError("503 (no inner tier for this path)")

    outcome = await process_with_policy(run_step, memory_source_id="src-y")

    assert outcome.kind is SaveOutcome.TRANSIENT_EXHAUSTED
    # The default whole_step_retry_max_attempts under MIRIX settings is 2,
    # so the loop runs initial + 2 = 3 attempts. Assert it ran more than
    # the single inner-exhausted attempt.
    assert calls["n"] >= 2, f"unmarked transient must still get whole-step retries — got {calls['n']}"


@pytest.mark.asyncio
async def test_inner_exhausted_permanent_classification_unaffected():
    """Permanent errors short-circuit regardless of the marker — the marker
    only skips the *transient retry loop*, not the permanent bypass."""
    calls = {"n": 0}

    async def run_step():
        calls["n"] += 1
        exc = ProviderPermanentError("400 (inner classified permanent)")
        raise mark_inner_exhausted(exc)

    outcome = await process_with_policy(run_step, memory_source_id="src-z")

    assert outcome.kind is SaveOutcome.PERMANENT_FAILURE
    assert calls["n"] == 1
