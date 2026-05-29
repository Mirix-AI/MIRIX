"""Unit tests for :mod:`evals.metaclaw.mirix_adapters.evolver_adapter`.

Uses :class:`httpx.MockTransport` so all assertions are at the HTTP wire
level — no patching of private methods, no inspection of internal state.

Coverage (per issue #04 acceptance criteria):

  1.  ``evolve`` POSTs to ``/v1/skills/evolve`` with body shape
      ``{messages: [str, ...], user_id: str}``.
  2.  ``_sample_to_message`` caps prompt at 1500 tail + response at 1500 head.
  3.  ``_sample_to_message`` does NOT inject ``outcome=FAIL`` / any fake label.
  4.  ``evolve([])`` short-circuits — zero HTTP requests observed.
  5.  HTTP error from mock → returns ``[]`` and logs warning.
  6.  Return value is always ``[]`` regardless of mock response shape.
  7.  ``__init__`` preflight raises clear error when ``/v1/skills/evolve``
      is missing from ``/openapi.json``.
  8.  One retry on transient 5xx, second attempt succeeds.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, List

import httpx
import pytest

from evals.metaclaw.mirix_adapters.evolver_adapter import (
    EVOLVE_ENDPOINT_PATH,
    PROMPT_TAIL_CHARS,
    RESPONSE_HEAD_CHARS,
    MirixEvolverAdapter,
    _parse_evolve_diff_counts,
    _sample_to_message,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


_OPENAPI_OK = {
    "openapi": "3.1.0",
    "paths": {
        "/v1/skills": {},
        "/v1/skills/evolve": {"post": {"summary": "Evolve Skills"}},
    },
}

_OPENAPI_MISSING = {
    "openapi": "3.1.0",
    "paths": {"/v1/skills": {}},
}


class _Sample:
    """Mimics paper's ``SimpleNamespace(prompt_text, response_text, reward)``."""

    def __init__(self, prompt_text: str, response_text: str, reward: float = 0.0):
        self.prompt_text = prompt_text
        self.response_text = response_text
        self.reward = reward


def _make_adapter(
    handler: Callable[[httpx.Request], httpx.Response],
    **kw,
) -> MirixEvolverAdapter:
    """Construct an evolver adapter wired to a MockTransport."""
    transport = httpx.MockTransport(handler)
    return MirixEvolverAdapter(
        base_url=kw.pop("base_url", "http://mock.test"),
        user_id=kw.pop("user_id", "u-test"),
        transport=transport,
        retry_sleep_s=kw.pop("retry_sleep_s", 0.0),  # don't sleep in tests
        **kw,
    )


def _ok_evolve_handler(captured: List[httpx.Request]) -> Callable:
    """Build a handler that ack's preflight + returns a canonical evolve diff."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if request.url.path == EVOLVE_ENDPOINT_PATH:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "changes": {
                        "created": [{"id": "s1"}, {"id": "s2"}],
                        "edited": [{"id": "s3"}],
                        "deleted": [],
                    },
                    "summary": {"total_after": 4},
                },
            )
        return httpx.Response(404, text=f"unexpected path {request.url.path}")

    return handler


# --------------------------------------------------------------------------- #
# 1. evolve() POSTs the right URL and payload shape                           #
# --------------------------------------------------------------------------- #


def test_evolve_posts_correct_url_and_payload_shape():
    captured: List[httpx.Request] = []
    a = _make_adapter(_ok_evolve_handler(captured), user_id="u-arm-mirix")

    samples = [
        _Sample("prompt one", "response one"),
        _Sample("prompt two", "response two"),
    ]
    out = asyncio.run(a.evolve(samples, current_skills={}))
    assert out == []

    # Find the evolve POST among captured requests (after the preflight GET).
    evolve_reqs = [r for r in captured if r.url.path == EVOLVE_ENDPOINT_PATH]
    assert len(evolve_reqs) == 1
    req = evolve_reqs[0]
    assert req.method == "POST"

    import json as _json

    body = _json.loads(req.content)
    assert set(body.keys()) == {"messages", "user_id"}
    assert body["user_id"] == "u-arm-mirix"
    assert isinstance(body["messages"], list)
    assert len(body["messages"]) == 2
    assert all(isinstance(m, str) for m in body["messages"])
    # Each message should mention its turn index
    assert "Turn 1" in body["messages"][0]
    assert "Turn 2" in body["messages"][1]
    # Auth header propagated
    assert req.headers["X-Client-Id"].startswith("client-")


# --------------------------------------------------------------------------- #
# 2. _sample_to_message caps prompt tail 1500 + response head 1500            #
# --------------------------------------------------------------------------- #


def test_sample_to_message_caps_payload_sizes():
    # Distinct head/tail markers so we can verify which slice survived.
    big_prompt = "PROMPT_HEAD_MARKER" + ("X" * 5000) + "PROMPT_TAIL_MARKER"
    big_response = "RESPONSE_HEAD_MARKER" + ("Y" * 5000) + "RESPONSE_TAIL_MARKER"
    s = _Sample(big_prompt, big_response)
    msg = _sample_to_message(s, idx=0)

    # The kept slices (prompt tail, response head) must include their markers.
    assert "PROMPT_TAIL_MARKER" in msg
    assert "RESPONSE_HEAD_MARKER" in msg

    # The dropped slices (prompt head, response tail) must NOT appear.
    assert "PROMPT_HEAD_MARKER" not in msg
    assert "RESPONSE_TAIL_MARKER" not in msg

    # Total bytes of the "X" filler (prompt tail) must be <= PROMPT_TAIL_CHARS.
    assert msg.count("X") <= PROMPT_TAIL_CHARS
    # Total bytes of the "Y" filler (response head) must be <= RESPONSE_HEAD_CHARS.
    assert msg.count("Y") <= RESPONSE_HEAD_CHARS


# --------------------------------------------------------------------------- #
# 3. _sample_to_message does NOT inject FAIL / fake quality label             #
# --------------------------------------------------------------------------- #


def test_sample_to_message_does_not_inject_fake_quality_label():
    s = _Sample("ctx", "resp", reward=0.0)
    msg = _sample_to_message(s, idx=0)
    forbidden = (
        "FAIL",
        "PASS",
        "outcome=",
        "reward=",
        "label=",
        "success=",
        "failure=",
        "ground_truth=",
    )
    for tok in forbidden:
        assert tok not in msg, f"adapter must not inject fake quality label {tok!r}"


# --------------------------------------------------------------------------- #
# 4. evolve([]) short-circuits — zero HTTP requests post-preflight            #
# --------------------------------------------------------------------------- #


def test_evolve_empty_samples_short_circuits_no_http():
    captured: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        return httpx.Response(500, text="should not be called")

    a = _make_adapter(handler)
    # Reset captures so we only count what evolve() does.
    captured.clear()

    out = asyncio.run(a.evolve([], current_skills={}))
    assert out == []
    # No requests at all should have been issued during evolve().
    assert captured == []


# --------------------------------------------------------------------------- #
# 5. HTTP error → return [] and emit warning                                  #
# --------------------------------------------------------------------------- #


def test_evolve_http_error_returns_empty_and_logs_warning(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        # Both attempts (initial + retry) fail.
        return httpx.Response(503, text="server overloaded")

    a = _make_adapter(handler)
    samples = [_Sample("p", "r")]

    with caplog.at_level(
        logging.WARNING, logger="evals.metaclaw.mirix_adapters.evolver_adapter"
    ):
        out = asyncio.run(a.evolve(samples, current_skills={}))

    assert out == []
    # Warning text should mention HTTP error or the endpoint
    joined = " ".join(rec.message for rec in caplog.records)
    assert "HTTP error" in joined or "HTTP 503" in joined or "evolve" in joined


# --------------------------------------------------------------------------- #
# 6. Return value is always [] regardless of mock response                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mock_payload",
    [
        # Canonical good shape
        {
            "success": True,
            "changes": {"created": [{"id": "x"}], "edited": [], "deleted": []},
        },
        # Empty diff
        {"success": True, "changes": {"created": [], "edited": [], "deleted": []}},
        # Older shape (counts only)
        {"success": True, "created": 3, "edited": 1, "deleted": 0},
        # Unexpected shape — MIRIX returns something weird
        {"unexpected": True, "foo": "bar"},
        # Empty dict
        {},
    ],
)
def test_evolve_returns_empty_regardless_of_response_shape(mock_payload):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        return httpx.Response(200, json=mock_payload)

    a = _make_adapter(handler)
    out = asyncio.run(a.evolve([_Sample("p", "r")], current_skills={}))
    assert out == []


# --------------------------------------------------------------------------- #
# 7. Preflight raises when /v1/skills/evolve is missing                       #
# --------------------------------------------------------------------------- #


def test_init_raises_when_evolve_endpoint_missing_from_openapi():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_MISSING)
        return httpx.Response(404, text="never reached")

    with pytest.raises(RuntimeError) as excinfo:
        _make_adapter(handler)
    assert EVOLVE_ENDPOINT_PATH in str(excinfo.value)
    assert (
        "feat/skill-evolve" in str(excinfo.value)
        or "branch" in str(excinfo.value).lower()
    )


def test_init_tolerates_openapi_fetch_failure():
    """If /openapi.json itself returns 500 the adapter should warn but
    continue — we'd rather see the real evolve error than crash startup."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(500, text="oops")
        return httpx.Response(200, json={"success": True, "changes": {}})

    # Should not raise
    a = _make_adapter(handler)
    assert a is not None


# --------------------------------------------------------------------------- #
# 8. One retry on 5xx                                                          #
# --------------------------------------------------------------------------- #


def test_evolve_retries_once_on_5xx_then_succeeds():
    calls = {"evolve_n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        if request.url.path == EVOLVE_ENDPOINT_PATH:
            calls["evolve_n"] += 1
            if calls["evolve_n"] == 1:
                return httpx.Response(502, text="bad gateway")
            return httpx.Response(
                200,
                json={"success": True, "changes": {"created": [{"id": "x"}]}},
            )
        return httpx.Response(404)

    a = _make_adapter(handler)
    out = asyncio.run(a.evolve([_Sample("p", "r")], current_skills={}))
    assert out == []  # always []
    assert calls["evolve_n"] == 2  # initial + 1 retry


# --------------------------------------------------------------------------- #
# Misc: _parse_evolve_diff_counts                                              #
# --------------------------------------------------------------------------- #


def test_parse_evolve_diff_counts_handles_canonical_shape():
    payload = {
        "success": True,
        "changes": {"created": [1, 2, 3], "edited": [4], "deleted": [5, 6]},
    }
    assert _parse_evolve_diff_counts(payload) == (3, 1, 2)


def test_parse_evolve_diff_counts_handles_int_counts():
    payload = {"created": 5, "edited": 0, "deleted": 1}
    assert _parse_evolve_diff_counts(payload) == (5, 0, 1)


def test_parse_evolve_diff_counts_handles_unknown_shape():
    assert _parse_evolve_diff_counts({"whatever": True}) == (0, 0, 0)
    assert _parse_evolve_diff_counts(None) == (0, 0, 0)
    assert _parse_evolve_diff_counts("string") == (0, 0, 0)


def test_evolve_exhausts_all_retries_on_sustained_5xx():
    """Sustained 5xx -> adapter tries max_http_retries+1 times, then returns [].

    Locks in the 2026-05-28 hardening: a single retry could not outlast a
    sustained PG-pool outage, silently degrading the mirix arm to
    retrieval-only.  We now retry several times with exponential backoff.
    """
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        call_count["n"] += 1
        return httpx.Response(500, json={"detail": "pg pool down"})

    adapter = _make_adapter(handler, max_http_retries=4)
    out = asyncio.run(adapter.evolve([_Sample("ctx", "resp")], current_skills={}))
    assert out == []
    # 1 initial attempt + 4 retries = 5 calls.
    assert call_count["n"] == adapter.max_http_retries + 1 == 5


def test_evolve_recovers_after_burst_of_5xx():
    """A burst of 5xx followed by a 200 -> the retry loop rides through it."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        call_count["n"] += 1
        if call_count["n"] < 4:
            return httpx.Response(503, json={"detail": "reconnecting"})
        return httpx.Response(200, json={"changes": {"created": [{"id": "s1"}]}})

    adapter = _make_adapter(handler, max_http_retries=4)
    out = asyncio.run(adapter.evolve([_Sample("ctx", "resp")], current_skills={}))
    # Still [] (sole-writer model) but the call SUCCEEDED on attempt 4.
    assert out == []
    assert call_count["n"] == 4


def test_negative_max_http_retries_clamps_to_single_attempt():
    """codex LOW fix: a negative max_http_retries must not yield attempts==0
    (which would dereference an unbound resp). It clamps to 0 → one attempt."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        calls["n"] += 1
        return httpx.Response(200, json={"changes": {"created": [{"id": "s1"}]}})

    adapter = _make_adapter(handler, max_http_retries=-5)
    assert adapter.max_http_retries == 0
    out = asyncio.run(adapter.evolve([_Sample("c", "r")], current_skills={}))
    assert out == []
    assert calls["n"] == 1  # exactly one attempt, no retries


def test_4xx_is_not_retried_and_returns_empty():
    """A 4xx (e.g. 404 no-agent) must NOT be retried; surfaces once, returns []."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        calls["n"] += 1
        return httpx.Response(404, json={"detail": "No procedural memory agent"})

    adapter = _make_adapter(handler, max_http_retries=4)
    out = asyncio.run(adapter.evolve([_Sample("c", "r")], current_skills={}))
    assert out == []
    assert calls["n"] == 1  # 4xx not retried


def test_transport_error_on_final_attempt_returns_empty():
    """A transport error (not an HTTP response) on every attempt → [] after
    exhausting retries, never crashing paper's loop."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=_OPENAPI_OK)
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    adapter = _make_adapter(handler, max_http_retries=2)
    out = asyncio.run(adapter.evolve([_Sample("c", "r")], current_skills={}))
    assert out == []
    assert calls["n"] == adapter.max_http_retries + 1 == 3
