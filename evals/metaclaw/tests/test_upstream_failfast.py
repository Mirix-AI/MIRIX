"""H1 — Fail-fast on upstream OpenRouter ConnectError storm.

A total upstream network drop at a day boundary previously turned into a ~3h
silent 0% run: every proxy forward raised ``httpx.ConnectError`` and openclaw
retried in a tight loop. These tests pin the hardening:

  * a *bounded* retry-with-backoff on ConnectError (a single transient blip
    must NOT trip the fail-fast — it should be retried and, if it recovers,
    succeed),
  * a process-wide consecutive-connect-failure counter that, once it crosses a
    threshold, trips a DISTINCT fatal ``UpstreamUnreachableError`` so the run
    aborts within minutes instead of spinning,
  * normal HTTP errors (4xx/5xx WITH a status) are NOT this failure mode and
    must never trip the connect-failure fail-fast.
"""

from __future__ import annotations

import httpx
import pytest

from evals.metaclaw.vendor.metaclaw import api_server


# --------------------------------------------------------------------------- #
# The consecutive-connect-failure tracker (process-wide fail-fast)            #
# --------------------------------------------------------------------------- #


def test_tracker_trips_after_threshold_consecutive_connect_failures():
    tracker = api_server._ConnectFailureTracker(threshold=5)
    # First threshold-1 failures must NOT trip.
    for _ in range(4):
        assert tracker.record_connect_failure() is False
    # The Nth consecutive failure trips the fail-fast.
    assert tracker.record_connect_failure() is True
    assert tracker.tripped is True


def test_tracker_success_resets_the_counter():
    tracker = api_server._ConnectFailureTracker(threshold=3)
    assert tracker.record_connect_failure() is False
    assert tracker.record_connect_failure() is False
    # A single success in between clears the streak — a transient blip recovers.
    tracker.record_success()
    assert tracker.record_connect_failure() is False
    assert tracker.record_connect_failure() is False
    assert tracker.record_connect_failure() is True


def test_tracker_assert_alive_raises_once_tripped():
    tracker = api_server._ConnectFailureTracker(threshold=2)
    tracker.record_connect_failure()
    tracker.record_connect_failure()  # trips
    with pytest.raises(api_server.UpstreamUnreachableError):
        tracker.assert_alive()


def test_tracker_stays_tripped_after_late_success():
    """codex P2: once the latch trips, a later record_success() must NOT clear
    the fatal state — the run is already doomed and must abort."""
    tracker = api_server._ConnectFailureTracker(threshold=2)
    tracker.record_connect_failure()
    tracker.record_connect_failure()  # trips
    tracker.record_success()  # a late, racing success
    assert tracker.tripped is True
    with pytest.raises(api_server.UpstreamUnreachableError):
        tracker.assert_alive()


# --------------------------------------------------------------------------- #
# UpstreamUnreachableError is a DISTINCT, legible fatal error                  #
# --------------------------------------------------------------------------- #


def test_upstream_unreachable_error_is_distinct_and_legible():
    err = api_server.UpstreamUnreachableError("openrouter.ai", 5)
    assert isinstance(err, Exception)
    msg = str(err)
    assert "openrouter.ai" in msg or "unreachable" in msg.lower()
    # Not an HTTPException — it must propagate up and abort, not be returned as
    # a normal HTTP response the agent would retry.
    from fastapi import HTTPException

    assert not isinstance(err, HTTPException)


# --------------------------------------------------------------------------- #
# Bounded retry: a single ConnectError that then recovers must succeed         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_bounded_retry_recovers_from_single_blip(monkeypatch):
    """One ConnectError followed by a success → forward succeeds, no fail-fast."""
    calls = {"n": 0}

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("All connection attempts failed")
            return _FakeResp()

    monkeypatch.setattr(api_server.httpx, "AsyncClient", _FakeClient)

    server = _make_minimal_server(monkeypatch)
    # Make backoff instant so the test is fast.
    monkeypatch.setattr(api_server.asyncio, "sleep", _noop_sleep)

    result = await server._forward_to_openai_compat({"model": "x", "messages": []})
    assert result["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 2  # failed once, retried, succeeded
    assert server._connect_failure_tracker.tripped is False


@pytest.mark.asyncio
async def test_sustained_connect_storm_raises_upstream_unreachable(monkeypatch):
    """Every forward ConnectErrors → after the bounded retries are exhausted a
    single forward raises UpstreamUnreachableError (fatal, aborts the run)."""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr(api_server.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(api_server.asyncio, "sleep", _noop_sleep)

    server = _make_minimal_server(monkeypatch)
    # Drive the tracker right up to (threshold-1) so this one forward trips it.
    for _ in range(server._connect_failure_tracker.threshold - 1):
        server._connect_failure_tracker.record_connect_failure()

    with pytest.raises(api_server.UpstreamUnreachableError):
        await server._forward_to_openai_compat({"model": "x", "messages": []})


@pytest.mark.asyncio
async def test_connect_timeout_is_treated_as_connect_failure(monkeypatch):
    """codex P1: httpx.ConnectTimeout is NOT a subclass of ConnectError. A
    blackholed upstream that times out on connect must ALSO drive the fail-fast,
    not fall through to a plain 502."""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(api_server.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(api_server.asyncio, "sleep", _noop_sleep)

    server = _make_minimal_server(monkeypatch)
    for _ in range(server._connect_failure_tracker.threshold - 1):
        server._connect_failure_tracker.record_connect_failure()

    with pytest.raises(api_server.UpstreamUnreachableError):
        await server._forward_to_openai_compat({"model": "x", "messages": []})


@pytest.mark.asyncio
async def test_http_status_error_does_not_trip_failfast(monkeypatch):
    """A 429/500 WITH a status is NOT the connect-failure mode — it must surface
    as an HTTPException and leave the connect-failure tracker untouched."""
    from fastapi import HTTPException

    class _FakeResp:
        status_code = 429
        text = "rate limited"

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "429", request=httpx.Request("POST", "http://x"), response=self  # type: ignore[arg-type]
            )

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _FakeResp()

    monkeypatch.setattr(api_server.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(api_server.asyncio, "sleep", _noop_sleep)

    server = _make_minimal_server(monkeypatch)
    with pytest.raises(HTTPException):
        await server._forward_to_openai_compat({"model": "x", "messages": []})
    # The connect-failure tracker must be pristine: this is not its failure mode.
    assert server._connect_failure_tracker._consecutive == 0
    assert server._connect_failure_tracker.tripped is False


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


async def _noop_sleep(_seconds):  # pragma: no cover - trivial
    return None


def _make_minimal_server(monkeypatch):
    """Build a MetaClawAPIServer with just enough wiring to call the forward.

    We bypass __init__ (which touches the filesystem / config) and set only the
    attributes ``_forward_to_openai_compat`` reads.
    """
    import types

    server = api_server.MetaClawAPIServer.__new__(api_server.MetaClawAPIServer)
    server.config = types.SimpleNamespace(
        llm_api_base="https://openrouter.ai/api/v1",
        llm_api_key="sk-test",
        llm_model_id="test-model",
    )
    server._connect_failure_tracker = api_server._ConnectFailureTracker(threshold=8)
    return server
