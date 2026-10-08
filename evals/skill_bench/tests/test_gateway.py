"""Gateway contract, against a local fake upstream only."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.infra import gateway as gw_mod
from evals.skill_bench.infra import gateway as gateway_module
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.gateway import (
    LOCAL_TOKEN,
    gateway_url,
    parse_retry_after,
    prepare_request,
)
from evals.skill_bench.tests.fakes import (
    FAKE_UPSTREAM_KEY,
    FakeUpstream,
    FlakyTransport,
    Reply,
    chat_body,
)
from evals.skill_bench.tests.fakes import running_gateway as running

LUNA = "openai/gpt-6-luna"
LUNA56 = "openai/gpt-5.6-luna"


def chat_request(
    model: str = LUNA, content: str = "hello", **extra: Any
) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


def ledger(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    path = tmp_path / "runs" / run_id / "gateway.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def final_calls(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    return [row for row in ledger(tmp_path, run_id) if row.get("event") == "call"]


# --- request rewrite ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_cap"), [(LUNA, 102_400), (LUNA56, 102_400)]
)
async def test_rewrite_applies_f9_reasoning_seed_and_drops_sampling(
    tmp_path, model, expected_cap
):
    fake = FakeUpstream()
    caller = chat_request(
        model,
        max_tokens=50,
        max_completion_tokens=60,
        temperature=0.7,
        top_p=0.9,
        stream_options={"include_usage": True},
        reasoning={"effort": "high"},
        reasoning_effort="high",
        seed=7,
        provider={"order": ["OpenAI"], "allow_fallbacks": False},
        models=["openai/gpt-4o"],
        route="fallback",
        plugins=[{"id": "web"}],
        transforms=["middle-out"],
        tools=[
            {
                "type": "function",
                "function": {"name": "look", "parameters": {"type": "object"}},
            }
        ],
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            f"/runs/cell-{model.split('/')[0]}/actor/v1/chat/completions", json=caller
        )
    assert response.status_code == 200
    sent = fake.calls("chat/completions")[0].json
    assert sent["max_tokens"] == expected_cap
    assert sent["reasoning"] == {"effort": "medium"}
    assert sent["seed"] == protocol.SEED
    assert sent["stream"] is False
    assert sent["usage"] == {"include": True}
    assert sent["transforms"] == []
    for dropped in (
        "temperature",
        "top_p",
        "stream_options",
        "reasoning_effort",
        "max_completion_tokens",
        "models",
        "route",
        "plugins",
    ):
        assert dropped not in sent, dropped
    # The caller's own provider routing never reaches upstream; the protocol's own
    # policy names neither luna, so none is sent.
    assert sent.get("provider") == protocol.provider_routing(model)
    assert sent["tools"] == caller["tools"] and sent["messages"] == caller["messages"]
    headers = fake.calls("chat/completions")[0].headers
    assert headers["authorization"] == f"Bearer {FAKE_UPSTREAM_KEY}"


def test_prepare_request_does_not_mutate_caller_body():
    limits = {m: protocol.model_limits(m) for m in protocol.CHAT_MODELS}
    body = chat_request(temperature=1.0, max_tokens=5)
    raw = json.dumps(body).encode()
    prepared = prepare_request(
        raw,
        "chat/completions",
        limits=limits,
        seed_models=frozenset(limits),
        seed=42,
        effort="medium",
    )
    assert body == chat_request(temperature=1.0, max_tokens=5)
    assert prepared.max_tokens_requested == 5
    assert prepared.max_tokens_rewritten == 102_400
    assert json.loads(prepared.content) == prepared.body


def test_seed_only_sent_to_models_that_support_it():
    limits = {m: protocol.model_limits(m) for m in protocol.CHAT_MODELS}
    raw = json.dumps(chat_request(seed=3)).encode()
    prepared = prepare_request(
        raw,
        "chat/completions",
        limits=limits,
        seed_models=frozenset(),
        seed=42,
        effort="medium",
    )
    assert "seed" not in prepared.body


@pytest.mark.asyncio
async def test_f9_cap_shrinks_when_input_is_large(tmp_path):
    # ~780k tokens for luna: window80 - input < max_out80, so the cap is window-bound.
    fake = FakeUpstream()
    caller = chat_request(LUNA, " hello" * 780_000)
    expected_input = tokens.estimate_chat_input(caller).total
    limits = protocol.model_limits(LUNA)
    assert limits.window80 - expected_input < limits.max_out80
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/big/learner/v1/chat/completions", json=caller
        )
    assert response.status_code == 200
    # The F9 cap leaves ceil(1.1 * input) + max_tokens below luna's top provider window
    # (1,050,000), so the provider-window clamp (W10, N5) does not bind.
    assert 737_600 < expected_input <= limits.gateway_input_budget
    f9_cap = 840_000 - expected_input
    provider_cap = 1_050_000 - (-(-expected_input * 11 // 10)) - 1
    assert fake.calls("chat/completions")[0].json["max_tokens"] == f9_cap
    row = final_calls(tmp_path, "big")[0]
    assert row["input_tokens_estimated"] == expected_input
    assert row["max_tokens_rewritten"] == f9_cap == row["max_tokens_f9"]
    assert row["max_tokens_provider"] == provider_cap > f9_cap
    assert row["max_tokens_bound"] == "window80"
    assert row["tokenizer_margin"] == "1/10"


@pytest.mark.asyncio
async def test_input_over_budget_is_rejected_before_upstream(tmp_path, monkeypatch):
    budget = protocol.model_limits(LUNA).gateway_input_budget
    real = tokens.count_text

    def fake_count(text: str) -> int:
        if text == "AT_BUDGET":
            return budget - (
                tokens.MESSAGE_OVERHEAD_TOKENS
                + real("user")
                + tokens.REPLY_PRIMING_TOKENS
            )
        if text == "OVER_BUDGET":
            return budget
        return real(text)

    monkeypatch.setattr(tokens, "count_text", fake_count)
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        over = await client.post(
            "/runs/r1/actor/v1/chat/completions",
            json=chat_request(content="OVER_BUDGET"),
        )
        at = await client.post(
            "/runs/r1/actor/v1/chat/completions", json=chat_request(content="AT_BUDGET")
        )
    assert over.status_code == 400
    error = over.json()["error"]
    assert error["type"] == "context_budget_exceeded" and error["source"] == "gateway"
    assert error["input_budget"] == budget and error["input_tokens_estimated"] > budget
    assert at.status_code == 200
    assert len(fake.calls("chat/completions")) == 1
    assert (
        fake.calls("chat/completions")[0].json["max_tokens"]
        == protocol.MIN_OUTPUT_TOKENS
    )
    statuses = [row["status"] for row in final_calls(tmp_path, "r1")]
    assert statuses == ["rejected", "ok"]


# --- whitelist, identity ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whitelist_and_identity_checks(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        cases = {
            "unknown_chat": await client.post(
                "/runs/w/actor/v1/chat/completions", json=chat_request("openai/gpt-4o")
            ),
            "online_variant": await client.post(
                "/runs/w/actor/v1/chat/completions", json=chat_request(LUNA + ":online")
            ),
            "batch_variant": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request(LUNA56 + ":batch"),
            ),
            # Another variant is another model (the same weights in another mode), and
            # the alias OpenRouter re-points. (openai/gpt-5.6-luna is a model under test
            # of its own: luna56.)
            "luna_variant": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request("openai/gpt-6-luna-pro"),
            ),
            "latest_alias": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request("~openai/gpt-luna-latest"),
            ),
            "chat_as_embedding": await client.post(
                "/runs/w/embedding/v1/embeddings", json={"model": LUNA, "input": "x"}
            ),
            "embedding_as_chat": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request(protocol.EMBEDDING_MODEL),
            ),
            "role_mismatch": await client.post(
                "/runs/w/actor/v1/embeddings",
                json={"model": protocol.EMBEDDING_MODEL, "input": "x"},
            ),
            "bad_role": await client.post(
                "/runs/w/admin/v1/chat/completions", json=chat_request()
            ),
            "bad_endpoint": await client.post(
                "/runs/w/actor/v1/completions", json=chat_request()
            ),
            "bad_run_id": await client.post(
                "/runs/bad.run/actor/v1/chat/completions", json=chat_request()
            ),
            "bad_json": await client.post(
                "/runs/w/actor/v1/chat/completions", content=b"{not json"
            ),
            "bad_item": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request(),
                headers={"x-skill-bench-item": "train:0001"},
            ),
            "no_token": await client.post(
                "/runs/w/actor/v1/chat/completions",
                json=chat_request(),
                headers={"Authorization": "Bearer sk-real"},
            ),
        }
    expected = {
        "unknown_chat": 403,
        "online_variant": 403,
        "batch_variant": 403,
        "luna_variant": 403,
        "latest_alias": 403,
        "chat_as_embedding": 403,
        "embedding_as_chat": 403,
        "role_mismatch": 400,
        "bad_role": 400,
        "bad_endpoint": 404,
        "bad_run_id": 400,
        "bad_json": 400,
        "bad_item": 400,
        "no_token": 401,
    }
    assert {k: r.status_code for k, r in cases.items()} == expected
    assert fake.requests == []
    assert all(r.json()["error"]["source"] == "gateway" for r in cases.values())
    for key in ("luna_variant", "latest_alias"):
        assert cases[key].json()["error"]["type"] == "model_not_allowed", key
    rows = final_calls(tmp_path, "w")
    assert len(rows) == len(expected) - 1  # the invalid run_id is never used as a path
    assert {row["status"] for row in rows} == {"rejected"}


@pytest.mark.asyncio
async def test_run_is_pinned_to_one_chat_model(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        first = await client.post(
            "/runs/cell/actor/v1/chat/completions", json=chat_request(LUNA)
        )
        mixed = await client.post(
            "/runs/cell/learner/v1/chat/completions", json=chat_request(LUNA56)
        )
        other_run = await client.post(
            "/runs/cell2/learner/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert (first.status_code, mixed.status_code, other_run.status_code) == (
        200,
        403,
        200,
    )
    assert mixed.json()["error"]["type"] == "model_mismatch_for_run"
    assert len(fake.calls("chat/completions")) == 2


@pytest.mark.asyncio
async def test_luna56_is_whitelisted_and_never_mixed_with_luna(tmp_path):
    """``luna56`` (openai/gpt-5.6-luna) is a model under test: its
    calls pass the whitelist with its own F9 cap, on runs of its own. A luna run that
    asks for it (or a luna56 run that asks for luna) is refused: the ids are compared
    whole, never by prefix."""
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (gateway, client):
        own = await client.post(
            "/runs/c56/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
        luna = await client.post(
            "/runs/c/actor/v1/chat/completions", json=chat_request(LUNA)
        )
        into_luna = await client.post(
            "/runs/c/learner/v1/chat/completions", json=chat_request(LUNA56)
        )
        into_luna56 = await client.post(
            "/runs/c56/learner/v1/chat/completions", json=chat_request(LUNA)
        )
        health = (await client.get("/health")).json()
    assert (own.status_code, luna.status_code) == (200, 200)
    assert (into_luna.status_code, into_luna56.status_code) == (403, 403)
    for refused in (into_luna, into_luna56):
        assert refused.json()["error"]["type"] == "model_mismatch_for_run"
    assert health["effort_supported"][LUNA56] is True
    assert LUNA56 in health["models"]
    row = final_calls(tmp_path, "c56")[0]
    assert row["status"] == "ok"
    sent = fake.calls("chat/completions")[0].json
    assert sent["model"] == LUNA56
    assert sent["max_tokens"] == row["max_tokens_rewritten"]
    assert row["max_tokens_rewritten"] <= protocol.model_limits(LUNA56).max_out80


# --- retries and deadline ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_5xx_are_retried_within_the_deadline(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(503, {"error": {"message": "busy"}}),
        Reply(502),
        Reply(500),
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 200
    assert response.headers["x-skill-bench-retries"] == "3"
    row = final_calls(tmp_path, "r")[0]
    assert row["retries"] == 3 and row["status"] == "ok"
    assert [a["outcome"] for a in row["attempts"]] == [
        "http_503",
        "http_502",
        "http_500",
        "ok",
    ]
    assert len(fake.calls("chat/completions")) == 4


@pytest.mark.asyncio
async def test_transport_errors_are_retried(tmp_path):
    fake = FakeUpstream()
    flaky = FlakyTransport(
        fake.asgi_transport(),
        [httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError],
    )
    async with running(tmp_path, fake, transport=flaky) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 200
    assert flaky.raised == ["ConnectError", "ReadError", "RemoteProtocolError"]
    row = final_calls(tmp_path, "r")[0]
    assert [a["outcome"] for a in row["attempts"]] == [
        "ConnectError",
        "ReadError",
        "RemoteProtocolError",
        "ok",
    ]
    # A connection that dropped after the request went out (a read error,
    # the upstream closing mid-answer) may be billed like a deadline-cut attempt; one
    # that never connected is not.
    assert [a.get("billed") for a in row["attempts"]] == [
        None,
        "unknown",
        "unknown",
        None,
    ]
    assert row["unknown_cost_attempts"] == 2 and row["unbilled_deadline_attempts"] == 0
    from evals.skill_bench.cell_records import summarize_ledger

    totals = summarize_ledger(tmp_path / "runs" / "r" / "gateway.jsonl")["totals"]
    assert totals["unknown_cost_calls"] == 1 and totals["unknown_cost_attempts"] == 2


class _TracedHang(httpx.AsyncBaseTransport):
    """A transport that reports a POST's progress through the request's ``trace``
    extension (as httpcore does) up to ``events``, then never answers; anything else (the
    balance check) goes to ``inner``."""

    def __init__(self, inner: httpx.AsyncBaseTransport, events: list[str]) -> None:
        self.inner = inner
        self.events = events

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return await self.inner.handle_async_request(request)
        trace = request.extensions.get("trace")
        for name in self.events:
            if trace is not None:
                await trace(name, {})
        await asyncio.sleep(60)
        raise AssertionError("unreachable")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("events", "billed"),
    [
        # Still connecting (or waiting for a pooled connection) at the deadline: the
        # request never reached the provider.
        (["connection.connect_tcp.started"], None),
        (
            [
                "connection.connect_tcp.started",
                "connection.connect_tcp.complete",
                "http11.send_request_headers.started",
            ],
            None,
        ),
        # The body went out: the provider may be generating, OpenRouter may bill it.
        (
            [
                "http11.send_request_headers.started",
                "http11.send_request_body.started",
                "http11.send_request_body.complete",
                "http11.receive_response_headers.started",
            ],
            "unknown",
        ),
        # A transport that says nothing: an attempt cut off may have been billed.
        ([], "unknown"),
    ],
)
async def test_a_deadline_before_the_request_was_sent_is_not_billed(
    tmp_path, events, billed
):
    fake = FakeUpstream()
    async with running(
        tmp_path,
        fake,
        transport=_TracedHang(fake.asgi_transport(), events),
        deadline_s=0.3,
    ) as (_, client):
        started = time.monotonic()
        response = await client.post(
            "/runs/x3/actor/v1/chat/completions", json=chat_request()
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 504
    # The deadline plus the grace httpx's own timeouts get to fire first.
    assert elapsed < 0.3 + gateway_module.DEADLINE_GRACE_S + 1.0
    (attempt,) = final_calls(tmp_path, "x3")[0]["attempts"]
    assert attempt["outcome"] == "deadline" and attempt.get("billed") == billed


@pytest.mark.asyncio
async def test_provider_unavailable_bodies_are_retried(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(200, {"error": {"code": 502, "message": "Provider returned error"}}),
        Reply(200, chat_body(content=None, finish_reason="error")),
        Reply(200, {"id": "x", "choices": []}),
        Reply(200, raw=b"<html>gateway</html>"),
        Reply(200, {"error": {"message": "unexpected provider failure"}}),
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 200
    outcomes = [a["outcome"] for a in final_calls(tmp_path, "r")[0]["attempts"]]
    assert outcomes == [
        "provider_error_body",
        "choice_error",
        "empty_choices",
        "invalid_json",
        "provider_error_body",
        "ok",
    ]


@pytest.mark.asyncio
async def test_client_error_inside_200_body_is_not_retried(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(
            200, {"error": {"code": 400, "message": "context too long for provider"}}
        ),
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 400
    assert response.json()["error"]["source"] == "upstream"
    assert len(fake.calls("chat/completions")) == 1


@pytest.mark.asyncio
async def test_mirix_calls_are_ledgered_with_the_session_they_serve(tmp_path):
    """B32: a MIRIX run with MIRIX_LLM_SESSION_HEADER=x-skill-bench-item names the
    session it serves (the harness's task uid) on its chat and embedding calls; the
    gateway ledgers it as the call's item for the mirix and embedding roles alike, and
    the header never reaches the provider. A call without it keeps item null."""
    fake = FakeUpstream()
    uid = "t0123456789abcdef0123"
    tagged = {"x-skill-bench-item": uid}
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/m/phase", json={"phase": "test"})
        replies = [
            await client.post(
                "/runs/m/mirix/v1/chat/completions", json=chat_request(), headers=tagged
            ),
            await client.post(
                "/runs/m/embedding/v1/embeddings",
                json={"model": protocol.EMBEDDING_MODEL, "input": "a query"},
                headers=tagged,
            ),
            await client.post("/runs/m/mirix/v1/chat/completions", json=chat_request()),
        ]
    assert [r.status_code for r in replies] == [200, 200, 200]
    rows = final_calls(tmp_path, "m")
    assert [(r["role"], r["item"]) for r in rows] == [
        ("mirix", uid),
        ("embedding", uid),
        ("mirix", None),
    ]
    assert fake.requests and all(
        uid not in json.dumps(r.headers) and uid not in json.dumps(r.json)
        for r in fake.requests
    )


@pytest.mark.asyncio
async def test_empty_embedding_input_is_rejected(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/e/embedding/v1/embeddings",
            json={"model": "google/gemini-embedding-001", "input": []},
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "empty_input"
    assert fake.requests == []


@pytest.mark.asyncio
async def test_429_honors_retry_after(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(
            429, {"error": {"message": "rate limited"}}, headers={"Retry-After": "1"}
        ),
    )
    async with running(tmp_path, fake) as (_, client):
        started = time.monotonic()
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 200
    assert elapsed >= 1.0
    assert final_calls(tmp_path, "r")[0]["attempts"][0]["wait_s"] == 1.0


@pytest.mark.asyncio
async def test_retry_after_beyond_deadline_fails_fast_with_504(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(429, headers={"Retry-After": "30"}))
    async with running(tmp_path, fake, deadline_s=5.0) as (_, client):
        started = time.monotonic()
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 504
    assert response.json()["error"]["reason"] == "retry_after_exceeds_deadline"
    assert elapsed < 2.0
    assert len(fake.calls("chat/completions")) == 1


@pytest.mark.asyncio
async def test_deadline_returns_504_and_is_never_retried(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(delay_s=5.0))
    async with running(tmp_path, fake, deadline_s=0.5) as (_, client):
        started = time.monotonic()
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 504
    error = response.json()["error"]
    assert (
        error["type"] == "deadline_exceeded"
        and error["retries"] == 0
        and error["upstream_attempts"] == 1
    )
    assert elapsed < 2.0
    assert len(fake.calls("chat/completions")) == 1
    row = final_calls(tmp_path, "r")[0]
    assert (
        row["status"] == "deadline_exceeded"
        and row["http_status"] == 504
        and row["retries"] == 0
    )
    # The provider ran on after the gateway gave up and OpenRouter billed it;
    # the cost is unknown, which the line says instead of implying "not billed".
    (attempt,) = row["attempts"]
    assert attempt["outcome"] == "deadline" and attempt["billed"] == "unknown"
    assert row["unbilled_deadline_attempts"] == 1
    assert row["billed_attempts"] == 0 and row["cost_total"] is None


@pytest.mark.asyncio
async def test_a_deadline_attempt_after_a_billed_retry_is_counted_apart(tmp_path):
    """A billed attempt that was retried keeps its cost in ``cost_total``; the attempt the
    deadline then cut off is ``billed: unknown`` and counted in
    ``unbilled_deadline_attempts``, never mistaken for an unbilled one."""
    fake = FakeUpstream()
    failed = chat_body(
        finish_reason="error",
        usage={"prompt_tokens": 500, "completion_tokens": 90, "cost": 0.002},
    )
    fake.queue("chat/completions", Reply(json=failed), Reply(delay_s=5.0))
    async with running(tmp_path, fake, deadline_s=0.8) as (_, client):
        response = await client.post(
            "/runs/cut/learner/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 504
    row = final_calls(tmp_path, "cut")[0]
    first, second = row["attempts"]
    assert first["cost"] == 0.002 and "billed" not in first
    assert second["outcome"] == "deadline" and second["billed"] == "unknown"
    assert row["billed_attempts"] == 1 and row["cost_total"] == pytest.approx(0.002)
    assert row["unbilled_deadline_attempts"] == 1
    from evals.skill_bench.cell_records import summarize_ledger

    totals = summarize_ledger(tmp_path / "runs" / "cut" / "gateway.jsonl")["totals"]
    assert totals["unknown_cost_calls"] == 1 and totals["billed_attempts"] == 1
    assert totals["cost_total"] == pytest.approx(0.002)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout", "billed"),
    [
        (httpx.ReadTimeout, "unknown"),  # sent: the provider may be generating
        (httpx.PoolTimeout, None),  # never sent
        (httpx.WriteTimeout, None),  # the body never arrived whole
    ],
)
async def test_only_a_deadline_after_the_request_was_sent_is_billed_unknown(
    tmp_path, timeout, billed
):
    fake = FakeUpstream()
    flaky = FlakyTransport(fake.asgi_transport(), [timeout])
    async with running(tmp_path, fake, transport=flaky) as (_, client):
        response = await client.post(
            "/runs/to/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 504 and flaky.raised == [timeout.__name__]
    row = final_calls(tmp_path, "to")[0]
    (attempt,) = row["attempts"]
    assert attempt["outcome"] == "deadline" and attempt.get("billed") == billed
    assert row["unbilled_deadline_attempts"] == (1 if billed else 0)


@pytest.mark.asyncio
async def test_every_final_line_records_when_the_call_completed(tmp_path):
    """``ts`` is the call's arrival and ``completed_at`` its end, for every final line
    (ok, deadline, rejected), so a ledger lines up with the key's usage over time."""
    from datetime import datetime

    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(), Reply(delay_s=5.0))
    async with running(tmp_path, fake, deadline_s=0.5) as (_, client):
        ok = await client.post("/runs/t/actor/v1/chat/completions", json=chat_request())
        late = await client.post(
            "/runs/t/actor/v1/chat/completions", json=chat_request(content="again")
        )
        bad = await client.post(
            "/runs/t/actor/v1/chat/completions", json=chat_request("gpt-4o")
        )
    assert (ok.status_code, late.status_code, bad.status_code) == (200, 504, 403)
    rows = final_calls(tmp_path, "t")
    assert [r["status"] for r in rows] == ["ok", "deadline_exceeded", "rejected"]
    for row in rows:
        arrived = datetime.fromisoformat(row["ts"])
        completed = datetime.fromisoformat(row["completed_at"])
        assert completed >= arrived
        assert (completed - arrived).total_seconds() == pytest.approx(
            row["latency_s"], abs=0.25
        )
    assert rows[0]["unbilled_deadline_attempts"] == 0
    assert (
        datetime.fromisoformat(rows[1]["completed_at"])
        - datetime.fromisoformat(rows[1]["ts"])
    ).total_seconds() >= 0.5


@pytest.mark.asyncio
async def test_every_error_body_says_whose_failure_it_is(tmp_path):
    """``blame`` is the harness's own H8 verdict on the body and role, so a
    caller that cannot run it (MIRIX's M3.5 policy) tells a model timeout (the model's
    single attempt used the whole deadline) from a deadline the gateway lost."""
    from evals.skill_bench.infra.failures import classify_gateway_error

    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(delay_s=5.0),  # a model timeout on the mirix role
        Reply(delay_s=5.0),  # and on the actor role
        Reply(
            403,
            {
                "error": {
                    "code": 403,
                    "message": "openai/gpt-6-luna requires moderation on OpenAI. "
                    'Your input was flagged for "violence". No credits were charged.',
                    "metadata": {"reasons": ["violence"], "flagged_input": "..."},
                }
            },
        ),
        *[Reply(503) for _ in range(10)],
    )
    # A 1 s deadline leaves the first call 250 ms (a quarter) to reach its single
    # attempt; 0.5 s left 125 ms, which a loaded machine sometimes spent before the first
    # call's attempt started, so the gateway rightly blamed its own lost time (infra).
    async with running(tmp_path, fake, deadline_s=1.0, backoff_base_s=0.3) as (
        _,
        client,
    ):
        roles = {
            "mirix_timeout": "mirix",
            "actor_timeout": "actor",
            "moderation": "mirix",
            "lost_to_retries": "mirix",
            "rejected": "nobody",  # not a role: the gateway refuses the request
        }
        answers = {}
        for name, role in roles.items():
            answers[name] = await client.post(
                f"/runs/r/{role}/v1/chat/completions", json=chat_request()
            )
    blames = {name: r.json()["error"]["blame"] for name, r in answers.items()}
    assert blames == {
        "mirix_timeout": "model",
        "actor_timeout": "model",
        "moderation": "model",
        "lost_to_retries": "infra",
        "rejected": "harness",
    }
    for name, response in answers.items():
        verdict = classify_gateway_error(
            response.status_code, response.json(), roles[name]
        )
        assert blames[name] == verdict.kind.value, name
    lines = final_calls(tmp_path, "r")
    assert [line["error"]["blame"] for line in lines] == list(blames.values())


@pytest.mark.asyncio
async def test_retries_share_one_total_deadline(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(503) for _ in range(10)])
    async with running(tmp_path, fake, deadline_s=1.0, backoff_base_s=0.3) as (
        _,
        client,
    ):
        started = time.monotonic()
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
        elapsed = time.monotonic() - started
    assert response.status_code == 504
    assert response.json()["error"]["reason"] == "backoff_exceeds_deadline"
    assert elapsed <= 1.0 + 0.3
    row = final_calls(tmp_path, "r")[0]
    assert (
        row["retries"] == 2
    )  # waits 0.3 and 0.6; the next 1.2 s wait would pass the deadline
    assert len(fake.calls("chat/completions")) == 3


@pytest.mark.asyncio
async def test_retries_are_capped_at_five(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(503) for _ in range(10)])
    async with running(tmp_path, fake, backoff_base_s=0.001) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 503
    error = response.json()["error"]
    assert (
        error["type"] == "upstream_infra_failure"
        and error["retries"] == protocol.GATEWAY_MAX_RETRIES
    )
    assert len(fake.calls("chat/completions")) == protocol.GATEWAY_MAX_RETRIES + 1
    assert final_calls(tmp_path, "r")[0]["status"] == "infra_error"


@pytest.mark.asyncio
async def test_client_errors_pass_through_without_retry(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(400, {"error": {"code": 400, "message": "Invalid tool schema"}}),
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/r/actor/v1/chat/completions", json=chat_request(stream=True)
        )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["source"] == "upstream" and "Invalid tool schema" in error["message"]
    assert len(fake.calls("chat/completions")) == 1
    assert final_calls(tmp_path, "r")[0]["status"] == "upstream_error"


def test_parse_retry_after():
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("-1") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert (
        parse_retry_after("Mon, 28 Sep 2026 12:00:10 GMT", now=1_790_596_800.0) == 10.0
    )


# --- ledger ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ledger_records_every_required_field(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        phase = await client.post("/runs/cellA/phase", json={"phase": "learning"})
        response = await client.post(
            "/runs/cellA/judge/v1/chat/completions",
            json=chat_request(max_tokens=50),
            headers={"x-skill-bench-item": "t0123456789abcdef0123"},
        )
    assert phase.status_code == 200 and response.status_code == 200
    rows = ledger(tmp_path, "cellA")
    assert [r["event"] for r in rows] == ["phase", "call_started", "call"]
    started, final = rows[1], rows[2]
    assert (
        started["call_id"]
        == final["call_id"]
        == response.headers["x-skill-bench-call-id"]
    )
    expected = {
        "run_id": "cellA",
        "role": "judge",
        "phase": "learning",
        "item": "t0123456789abcdef0123",
        "endpoint": "chat/completions",
        "model": LUNA,
        "model_requested": LUNA,
        "status": "ok",
        "http_status": 200,
        "stream_requested": False,
        "max_tokens_requested": 50,
        "max_tokens_rewritten": 102_400,
        "input_tokens_actual": 42,
        "output_tokens": 17,
        "reasoning_tokens": 9,
        "cached_tokens": 0,
        "finish_reasons": ["stop"],
        "cost": 0.000123,
        "retries": 0,
        "provider": "FakeProvider",
        "generation_id": "gen-fake-1",
    }
    assert {k: final[k] for k in expected} == expected
    assert (
        final["input_tokens_estimated"]
        == tokens.estimate_chat_input(chat_request(max_tokens=50)).total
    )
    assert final["latency_s"] >= 0 and final["queue_wait_s"] >= 0
    assert final["completed_at"] >= final["ts"]
    assert final["billed_attempts"] == 1 and final["unbilled_deadline_attempts"] == 0
    assert (
        final["wire"]["reasoning"] == {"effort": "medium"}
        and final["wire"]["seed"] == 42
    )
    sent = fake.calls("chat/completions")[0].json
    assert (
        final["request_sha256"]
        == gw_mod.hashlib.sha256(
            json.dumps(sent, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
    )
    text = (tmp_path / "runs" / "cellA" / "gateway.jsonl").read_text()
    assert FAKE_UPSTREAM_KEY not in text


@pytest.mark.asyncio
async def test_run_state_survives_gateway_restart(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/cellB/phase", json={"phase": "test"})
        assert (
            await client.post(
                "/runs/cellB/actor/v1/chat/completions", json=chat_request(LUNA)
            )
        ).status_code == 200
    async with running(tmp_path, fake) as (_, client):
        mixed = await client.post(
            "/runs/cellB/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
        same = await client.post(
            "/runs/cellB/actor/v1/chat/completions", json=chat_request(LUNA)
        )
    assert mixed.status_code == 403 and same.status_code == 200
    assert final_calls(tmp_path, "cellB")[-1]["phase"] == "test"


# --- streaming, embeddings ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_callers_get_complete_response_as_sse(tmp_path):
    from openai import AsyncOpenAI

    fake = FakeUpstream()
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "total"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
        },
    ]
    fake.queue(
        "chat/completions",
        Reply(
            200,
            chat_body(
                content="thinking done", tool_calls=calls, finish_reason="tool_calls"
            ),
        ),
    )
    async with running(tmp_path, fake) as (gateway, _):
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app))
        sdk = AsyncOpenAI(
            api_key=LOCAL_TOKEN,
            base_url=gateway_url("http://gateway", "s", "actor"),
            http_client=http,
        )
        stream = await sdk.chat.completions.create(
            model=LUNA,
            messages=[{"role": "user", "content": "find the total"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        content, args, names, finish, usage = "", {}, {}, None, None
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            for choice in chunk.choices:
                content += choice.delta.content or ""
                for call in choice.delta.tool_calls or []:
                    names[call.index] = (
                        call.function.name if call.function else None
                    ) or names.get(call.index)
                    args[call.index] = args.get(call.index, "") + (
                        (call.function.arguments if call.function else "") or ""
                    )
                finish = choice.finish_reason or finish
        await http.aclose()
    assert content == "thinking done"
    assert names == {0: "grep", 1: "read"}
    assert args == {0: '{"pattern": "total"}', 1: '{"path": "a.txt"}'}
    assert finish == "tool_calls"
    assert usage is not None and usage.total_tokens == 59
    assert fake.calls("chat/completions")[0].json["stream"] is False
    assert final_calls(tmp_path, "s")[0]["stream_requested"] is True


def test_sse_frames_have_one_usage_frame_and_done():
    frames = list(gw_mod.sse_frames(chat_body()))
    assert frames[-1] == "data: [DONE]\n\n"
    assert sum('"usage"' in f for f in frames) == 1
    assert all(f.startswith("data: ") and f.endswith("\n\n") for f in frames)


@pytest.mark.asyncio
async def test_embeddings_pass_through_unchanged(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/e/embedding/v1/embeddings",
            json={
                "model": "gemini-embedding-001",
                "input": ["alpha", "beta"],
                "encoding_format": "float",
            },
        )
    assert response.status_code == 200
    sent = fake.calls("embeddings")[0].json
    assert sent == {
        "model": protocol.EMBEDDING_MODEL,
        "input": ["alpha", "beta"],
        "encoding_format": "float",
    }
    body = response.json()
    assert [d["index"] for d in body["data"]] == [0, 1] and len(
        body["data"][0]["embedding"]
    ) == 8
    row = final_calls(tmp_path, "e")[0]
    assert (
        row["model"] == protocol.EMBEDDING_MODEL
        and row["model_requested"] == "gemini-embedding-001"
    )
    assert row["max_tokens_rewritten"] is None and row["input_tokens_estimated"] > 0
    assert row["cost"] == 0.000001


# --- balance ----------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_balance_pauses_with_402_until_it_recovers(tmp_path):
    fake = FakeUpstream(limit_remaining=5.0)
    async with running(tmp_path, fake, balance_floor_usd=10.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        paused = await client.post(
            "/runs/b/actor/v1/chat/completions", json=chat_request()
        )
        health = (await client.get("/health")).json()
        fake.limit_remaining = 50.0
        resumed = await client.post(
            "/runs/b/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused is False
    assert (
        paused.status_code == 402 and paused.json()["error"]["type"] == "budget_paused"
    )
    assert health["paused"] is True and health["pause_reason"] == "balance_below_floor"
    assert resumed.status_code == 200
    assert len(fake.calls("chat/completions")) == 1
    assert [r["status"] for r in final_calls(tmp_path, "b")] == ["budget_paused", "ok"]
    # The held call was placed like any other: its call_started line comes first, so a
    # learning resume discards it and quiescence waits for it.
    events = [(r["event"], r.get("call_id")) for r in ledger(tmp_path, "b")]
    held = final_calls(tmp_path, "b")[0]["call_id"]
    assert events.index(("call_started", held)) < events.index(("call", held))
    balance_rows = (
        (tmp_path / "runs" / "_gateway" / "balance.jsonl").read_text().splitlines()
    )
    assert json.loads(balance_rows[0])["limit_remaining"] == 5.0


@pytest.mark.asyncio
async def test_unlimited_key_falls_back_to_account_credits(tmp_path):
    fake = FakeUpstream(limit_remaining=None, credits=(20.0, 17.5))
    async with running(tmp_path, fake, balance_floor_usd=10.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        low = await client.post(
            "/runs/c/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.balance is not None
        assert gateway.balance["remaining_usd"] == 2.5
        assert gateway.balance["remaining_source"] == "credits"
        fake.credits = (200.0, 17.5)
        ok = await client.post("/runs/c/actor/v1/chat/completions", json=chat_request())
        fake.credits = None  # neither source known: never blocks on its own
        unknown = await client.post(
            "/runs/c/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.balance["remaining_usd"] is None
    assert (low.status_code, ok.status_code, unknown.status_code) == (402, 200, 200)


@pytest.mark.asyncio
async def test_upstream_402_pauses_later_calls(tmp_path):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(402, {"error": {"code": 402, "message": "Insufficient credits"}}),
    )
    async with running(tmp_path, fake, balance_ttl_s=3600) as (_, client):
        first = await client.post(
            "/runs/b/actor/v1/chat/completions", json=chat_request()
        )
        second = await client.post(
            "/runs/b/actor/v1/chat/completions", json=chat_request()
        )
    assert first.status_code == second.status_code == 402
    assert len(fake.calls("chat/completions")) == 1


# --- launcher (real sockets, real child process) ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_gateway_reads_key_only_from_env_file(tmp_path):
    from evals.skill_bench.infra.processes import find_free_port

    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    with fake.serve() as upstream:
        process = await gw_mod.launch_gateway(
            port=find_free_port(),
            env_file=env_file,
            runs_dir=tmp_path / "runs",
            upstream=upstream,
            startup_timeout_s=60,
        )
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                response = await client.post(
                    process.url("live", "actor") + "/chat/completions",
                    json=chat_request(),
                    headers={"Authorization": f"Bearer {LOCAL_TOKEN}"},
                )
                health = (await client.get(process.base_url + "/health")).json()
        finally:
            await process.stop()
    assert response.status_code == 200
    assert (
        fake.calls("chat/completions")[0].headers["authorization"]
        == f"Bearer {FAKE_UPSTREAM_KEY}"
    )
    assert health["ok"] is True and health["encoding"] in (
        protocol.TOKEN_ENCODING,
        tokens.FALLBACK_ENCODING,
    )
    assert process.process.alive() is False
    log_text = (tmp_path / "runs" / "_gateway" / "gateway.log").read_text()
    assert FAKE_UPSTREAM_KEY not in log_text
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "runs" / "_gateway" / "owned_processes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert events == ["started", "stopped"]


def gateway_rows(runs: Path) -> list[dict[str, Any]]:
    path = runs / "_gateway" / "owned_processes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_a_second_harness_never_kills_a_live_harness_gateway(tmp_path):
    """N1: two harnesses share one runs dir (different ports); starting B leaves A running."""
    from evals.skill_bench.infra.processes import OwnedProcessError, find_free_port

    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    with fake.serve() as upstream:

        async def launch(port: int) -> gw_mod.GatewayProcess:
            return await gw_mod.launch_gateway(
                port=port, env_file=env_file, runs_dir=runs, upstream=upstream
            )

        port_a = find_free_port()
        first = await launch(port_a)
        second = None
        try:
            second = await launch(find_free_port())
            # The same port again, while its harness (this one) holds it: refused, not killed.
            with pytest.raises(OwnedProcessError, match="another live harness"):
                await launch(port_a)
            assert first.process.alive() and second.process.alive()
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                for gateway in (first, second):
                    health = await client.get(gateway.base_url + "/health")
                    assert health.status_code == 200
        finally:
            if second is not None:
                await second.stop()
            await first.stop()
    assert [row["event"] for row in gateway_rows(runs)] == [
        "started",
        "started",
        "stopped",
        "stopped",
    ]
    assert first.owner is not None and not first.owner.held


GATEWAY_HARNESS = """
import asyncio, sys, time
from pathlib import Path
from evals.skill_bench.infra.gateway import launch_gateway

async def main():
    gateway = await launch_gateway(
        port=int(sys.argv[1]), env_file=Path(sys.argv[2]), runs_dir=Path(sys.argv[3]),
        upstream=sys.argv[4],
    )
    print(gateway.process.pid, flush=True)
    time.sleep(120)  # "crashes" (SIGKILL) without stopping its gateway

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_a_crashed_harness_gateway_is_cleaned_before_its_port_is_reused(
    tmp_path,
):
    """W24 still holds under N1: a dead owner's gateway is killed (identity-checked)."""
    import signal
    import subprocess
    import sys

    from evals.skill_bench.infra.processes import find_free_port, process_start_time

    fake = FakeUpstream()
    env_file = tmp_path / "gateway.env"
    env_file.write_text(f"OPENROUTER_API_KEY={FAKE_UPSTREAM_KEY}\n")
    runs = tmp_path / "runs"
    port = find_free_port()
    with fake.serve() as upstream:
        harness = subprocess.Popen(
            [
                sys.executable,
                "-c",
                GATEWAY_HARNESS,
                str(port),
                str(env_file),
                str(runs),
                upstream,
            ],
            cwd=protocol.REPO_ROOT,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        orphan = 0
        try:
            assert harness.stdout is not None
            line = await asyncio.wait_for(
                asyncio.to_thread(harness.stdout.readline), 90
            )
            orphan = int(line)
            harness.send_signal(signal.SIGKILL)
            await asyncio.to_thread(harness.wait, 10)
            assert process_start_time(orphan) is not None  # still serving, unowned
            gateway = await gw_mod.launch_gateway(
                port=port, env_file=env_file, runs_dir=runs, upstream=upstream
            )
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                    health = await client.get(gateway.base_url + "/health")
                assert health.status_code == 200
                assert gateway.process.pid != orphan
            finally:
                await gateway.stop()
        finally:
            if harness.poll() is None:
                harness.kill()
            if orphan and process_start_time(orphan) is not None:
                os.kill(orphan, signal.SIGKILL)
    cleaned = [row for row in gateway_rows(runs) if row["event"] == "cleaned"]
    assert [(row["pid"], row["reason"]) for row in cleaned] == [(orphan, "killed")]


def test_load_api_key_never_echoes_the_value(tmp_path, monkeypatch):
    env_file = tmp_path / "e.env"
    env_file.write_text("OTHER=1\n")
    with pytest.raises(SystemExit) as excinfo:
        gw_mod.load_api_key(env_file)
    assert "OPENROUTER_API_KEY not found" in str(excinfo.value)
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-or-test-env ")
    assert gw_mod.load_api_key(None) == "sk-or-test-env"


def test_gateway_url_scheme():
    assert gateway_url("http://127.0.0.1:18540/", "alf-luna-mirix", "mirix") == (
        "http://127.0.0.1:18540/runs/alf-luna-mirix/mirix/v1"
    )
    with pytest.raises(ValueError):
        gateway_url("http://x", "a/b", "actor")
    with pytest.raises(ValueError):
        gateway_url("http://x", "a", "root")


# --- gateway edge cases ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_allowlisted_chat_fields_reach_upstream_and_drops_are_logged(
    tmp_path,
):
    """W18: sampling and routing fields beyond F8 never reach OpenRouter silently."""
    fake = FakeUpstream()
    caller = chat_request(
        n=8,
        top_k=5,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        repetition_penalty=1.1,
        min_p=0.1,
        logit_bias={"50256": -100},
        logprobs=True,
        top_logprobs=5,
        verbosity="high",
        preset="@preset/fancy",
        temperature=0.3,
        provider={"order": ["X"]},
        stop=["END"],
        user="cell-user",
        parallel_tool_calls=False,
        response_format={"type": "text"},
    )
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/allow/judge/v1/chat/completions", json=caller
        )
    assert response.status_code == 200
    sent = fake.calls("chat/completions")[0].json
    assert set(sent) == {
        "model",
        "messages",
        "stop",
        "user",
        "parallel_tool_calls",
        "response_format",
        "stream",
        "reasoning",
        "max_tokens",
        "seed",
        "usage",
        "transforms",
    }
    row = final_calls(tmp_path, "allow")[0]
    assert row["dropped_fields"] == sorted(
        [
            "n",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "min_p",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "verbosity",
            "preset",
            "temperature",
            "provider",
        ]
    )


@pytest.mark.asyncio
async def test_billed_attempts_that_were_retried_keep_their_cost(tmp_path):
    """W11: a 200 with finish_reason=error is retried, but it may have been billed."""
    fake = FakeUpstream()
    failed = chat_body(
        finish_reason="error",
        usage={"prompt_tokens": 50_000, "completion_tokens": 900, "cost": 0.25},
    )
    failed["id"] = "gen-failed"
    failed["provider"] = "FlakyProvider"
    fake.queue("chat/completions", Reply(json=failed), Reply())
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/bill/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 200
    row = final_calls(tmp_path, "bill")[0]
    assert row["retries"] == 1 and row["cost"] == 0.000123  # the answer's own cost
    first, second = row["attempts"]
    assert (first["outcome"], first["cost"], first["generation_id"]) == (
        "choice_error",
        0.25,
        "gen-failed",
    )
    assert (
        first["provider"] == "FlakyProvider"
        and first["usage"]["prompt_tokens"] == 50_000
    )
    assert second["cost"] == 0.000123 and second["generation_id"] == "gen-fake-1"
    assert row["cost_total"] == pytest.approx(0.250123)
    assert row["billed_attempts"] == 2
    assert first["budget_s"] > 0 and second["budget_s"] <= first["budget_s"]


@pytest.mark.asyncio
async def test_an_empty_unfinished_reply_is_retried_as_a_provider_failure(tmp_path):
    """W16: no content, no tool calls and no finish_reason is a dropped generation."""
    fake = FakeUpstream()
    dropped = chat_body(content="", finish_reason="stop")
    dropped["choices"][0]["finish_reason"] = None
    dropped["choices"][0]["native_finish_reason"] = "provider_disconnect"
    silent = chat_body(
        content="", finish_reason="stop"
    )  # the model chose to say nothing
    fake.queue("chat/completions", Reply(json=dropped), Reply(json=silent))
    async with running(tmp_path, fake) as (_, client):
        response = await client.post(
            "/runs/empty/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 200
    assert len(fake.calls("chat/completions")) == 2
    row = final_calls(tmp_path, "empty")[0]
    assert row["attempts"][0]["outcome"] == "empty_unfinished_choice"
    assert row["attempts"][0]["error"]["native_finish_reason"] == "provider_disconnect"
    assert row["finish_reasons"] == ["stop"]


@pytest.mark.asyncio
async def test_a_failed_ledger_write_never_loses_a_billed_response(
    tmp_path, monkeypatch
):
    """W17: ENOSPC on the final line: the caller still gets the paid completion."""
    fake = FakeUpstream()
    real = gw_mod._append_all
    broken = {"on": True}

    def flaky(path, records):
        if broken["on"] and any(r.get("event") == "call" for r in records):
            raise OSError(28, "No space left on device")
        return real(path, records)

    monkeypatch.setattr(gw_mod, "_append_all", flaky)
    async with running(tmp_path, fake) as (gateway, client):
        response = await client.post(
            "/runs/disk/actor/v1/chat/completions", json=chat_request()
        )
        assert response.status_code == 200 and response.json()["id"] == "gen-fake-1"
        health = (await client.get("/health")).json()
        assert health["ledger_errors"] == 1 and health["ledger_lines_pending"] == 1
        assert len(fake.calls("chat/completions")) == 1  # paid once, returned once
        broken["on"] = False
        await client.post("/runs/disk/phase", json={"phase": "test"})
        assert gateway._unlogged == {}
    events = [row["event"] for row in ledger(tmp_path, "disk")]
    assert events == [
        "call_started",
        "call",
        "phase",
    ]  # the held line was written first


@pytest.mark.asyncio
async def test_ledger_records_the_served_model_and_system_fingerprint(tmp_path):
    """W20: the dated snapshot OpenRouter served, for the H1.8 one-model-per-cell audit."""
    fake = FakeUpstream()
    body = chat_body()
    body["model"] = "openai/gpt-6-luna-20260922"
    body["system_fingerprint"] = "fp_abc123"
    fake.queue("chat/completions", Reply(json=body))
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/fp/actor/v1/chat/completions", json=chat_request())
    row = final_calls(tmp_path, "fp")[0]
    assert row["model"] == LUNA
    assert row["model_served"] == "openai/gpt-6-luna-20260922"
    assert row["system_fingerprint"] == "fp_abc123"


@pytest.mark.asyncio
async def test_a_torn_state_file_is_rebuilt_from_the_ledger(tmp_path):
    """W21: after a power loss the state file may be empty; the pin must survive."""
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/torn/phase", json={"phase": "learning"})
        await client.post(
            "/runs/torn/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    state = tmp_path / "runs" / "torn" / "gateway_state.json"
    assert json.loads(state.read_text())["model"] == LUNA56
    state.write_text("")  # torn write
    async with running(tmp_path, fake) as (_, client):
        mixed = await client.post(
            "/runs/torn/actor/v1/chat/completions", json=chat_request(LUNA)
        )
        same = await client.post(
            "/runs/torn/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert mixed.status_code == 403 and same.status_code == 200
    assert json.loads(state.read_text()) == {
        "model": LUNA56,
        "phase": "learning",
        "abandoned": False,
    }
    assert "state_rebuilt" in [row["event"] for row in ledger(tmp_path, "torn")]


@pytest.mark.asyncio
async def test_a_corrupt_state_with_an_ambiguous_ledger_fails_closed(tmp_path):
    run_dir = tmp_path / "runs" / "amb"
    run_dir.mkdir(parents=True)
    rows = [
        {"event": "call_started", "endpoint": "chat/completions", "model": m}
        for m in (LUNA, LUNA56)
    ]
    (run_dir / "gateway.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "gateway_state.json").write_text("{not json")
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        with pytest.raises(gw_mod.GatewayStateError):
            await client.post(
                "/runs/amb/actor/v1/chat/completions", json=chat_request()
            )
    assert fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_an_unknown_balance_never_lifts_an_upstream_402_pause(tmp_path):
    """W34: /key without a limit and /credits failing must not resume the gateway."""
    fake = FakeUpstream(limit_remaining=None, credits=None)
    fake.queue(
        "chat/completions",
        Reply(402, {"error": {"code": 402, "message": "Insufficient credits"}}),
    )
    async with running(tmp_path, fake, balance_ttl_s=0.0) as (gateway, client):
        first = await client.post(
            "/runs/p/actor/v1/chat/completions", json=chat_request()
        )
        second = await client.post(
            "/runs/p/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused and gateway.pause_reason == "upstream_402"
        fake.limit_remaining = 50.0  # a known balance above the floor resumes
        third = await client.post(
            "/runs/p/actor/v1/chat/completions", json=chat_request()
        )
        assert not gateway.paused
    assert (first.status_code, second.status_code, third.status_code) == (402, 402, 200)
    assert len(fake.calls("chat/completions")) == 2


@pytest.mark.asyncio
async def test_a_paused_gateway_holds_calls_until_the_balance_recovers(tmp_path):
    """W24: a short dip below the floor no longer fails (and restarts) every running cell."""
    fake = FakeUpstream(limit_remaining=5.0)
    async with running(
        tmp_path, fake, balance_floor_usd=10.0, balance_ttl_s=0.05, pause_hold_s=30.0
    ) as (gateway, client):
        call = asyncio.ensure_future(
            client.post("/runs/hold/actor/v1/chat/completions", json=chat_request())
        )
        await asyncio.sleep(0.3)
        assert not call.done() and gateway.paused
        fake.limit_remaining = 50.0  # topped up
        response = await asyncio.wait_for(call, 10)
    assert response.status_code == 200
    row = final_calls(tmp_path, "hold")[0]
    assert row["status"] == "ok" and row["balance_wait_s"] >= 0.2


@pytest.mark.asyncio
async def test_held_calls_share_one_balance_check_per_ttl(tmp_path):
    """N3: 40 held calls must not each poll /key (a rate-limited /key would pin the pause)."""
    fake = FakeUpstream(limit_remaining=5.0)
    ttl, held_s = 0.5, 2.0
    async with running(
        tmp_path, fake, balance_floor_usd=10.0, balance_ttl_s=ttl, pause_hold_s=30.0
    ) as (gateway, client):
        calls = [
            asyncio.ensure_future(
                client.post(
                    "/runs/herd/actor/v1/chat/completions",
                    json=chat_request(content=f"call {i}"),
                )
            )
            for i in range(40)
        ]
        await asyncio.sleep(held_s)
        assert gateway.paused and not any(call.done() for call in calls)
        fake.limit_remaining = 50.0
        responses = await asyncio.wait_for(asyncio.gather(*calls), 30)
    assert [r.status_code for r in responses] == [200] * 40
    # One fetch per TTL while held, plus the first check and the one that saw the top-up.
    assert fake.key_calls <= held_s / ttl + 4, fake.key_calls
    balance_rows = (tmp_path / "runs" / "_gateway" / "balance.jsonl").read_text()
    assert len(balance_rows.splitlines()) == fake.key_calls


@pytest.mark.asyncio
async def test_a_spend_budget_pauses_the_gateway_once_the_key_spent_it(tmp_path):
    """The balance floor alone lets a run spend up to the key's daily limit;
    ``budget_usd`` pauses once the key's all-time usage has grown by the budget's
    amount, counted from when the budget was first set for the runs directory."""
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 40.0
    async with running(tmp_path, fake, budget_usd=8.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        # Established at start: /health is ready without a pause.
        health = (await client.get("/health")).json()
        assert health["paused"] is False and health["budget_usd"] == 8.0
        assert health["budget"]["start_usage_usd"] == 40.0
        assert health["budget"]["left_usd"] == 8.0
        fake.usage = 47.5
        spending = await client.post(
            "/runs/bud/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.budget is not None and gateway.budget["spent_usd"] == 7.5
        fake.usage = 48.0  # the budget of 8 USD is spent
        spent = await client.post(
            "/runs/bud/actor/v1/chat/completions", json=chat_request()
        )
        health = (await client.get("/health")).json()
        assert health["paused"] and health["pause_reason"] == "budget_spent"
        assert health["budget"]["left_usd"] == 0.0
    assert (spending.status_code, spent.status_code) == (200, 402)
    assert spent.json()["error"]["type"] == "budget_paused"
    assert len(fake.calls("chat/completions")) == 1  # nothing sent past the budget
    record = json.loads((tmp_path / "runs" / "_gateway" / "budget.json").read_text())
    assert record["start_usage_usd"] == 40.0
    assert [b["budget_usd"] for b in record["budgets"]] == [8.0]
    rows = (tmp_path / "runs" / "_gateway" / "balance.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["budget"]["spent_usd"] == 8.0


@pytest.mark.asyncio
async def test_a_spent_budget_is_not_shadowed_by_the_balance_floor(tmp_path):
    """A key that holds just the floor plus the budget goes below the floor exactly as
    the budget is used up. The pause must say ``budget_spent`` (a person raises the
    budget; the supervisor exits 10 and alerts), not the waitable
    ``balance_below_floor`` a scheduler would poll forever."""
    fake = FakeUpstream(limit_remaining=260.0)
    fake.usage = 0.0
    async with running(
        tmp_path, fake, budget_usd=250.0, balance_ttl_s=0.0, balance_floor_usd=10.0
    ) as (_gateway, client):
        assert (await client.get("/balance")).json()["paused"] is False
        fake.usage, fake.limit_remaining = 250.4, 9.6
        health = (await client.get("/balance")).json()
        assert health["budget"]["left_usd"] < 0
        assert health["paused"] and health["pause_reason"] == "budget_spent"
        # A budget still left keeps the floor's (waitable) pause.
        fake.usage, fake.limit_remaining = 0.0, 9.6
    async with running(
        tmp_path / "other",
        fake,
        budget_usd=250.0,
        balance_ttl_s=0.0,
        balance_floor_usd=10.0,
    ) as (_gateway, client):
        health = (await client.get("/balance")).json()
        assert health["pause_reason"] == "balance_below_floor"


@pytest.mark.asyncio
async def test_a_restarted_gateway_keeps_the_runs_directorys_budget(tmp_path):
    """A relaunch after a pause does not start a new budget: the starting usage is
    kept in ``_gateway/budget.json``; a new amount (a person raised it) counts from
    the same start, and is recorded."""
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 10.0
    async with running(tmp_path, fake, budget_usd=5.0, balance_ttl_s=0.0):
        pass
    fake.usage = 15.0
    async with running(tmp_path, fake, budget_usd=5.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        assert gateway.paused and gateway.pause_reason == "budget_spent"
        refused = await client.post(
            "/runs/r2/actor/v1/chat/completions", json=chat_request()
        )
    assert refused.status_code == 402
    async with running(tmp_path, fake, budget_usd=7.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        assert not gateway.paused
        assert gateway.budget is not None and gateway.budget["left_usd"] == 2.0
        ok = await client.post(
            "/runs/r3/actor/v1/chat/completions", json=chat_request()
        )
    assert ok.status_code == 200
    record = json.loads((tmp_path / "runs" / "_gateway" / "budget.json").read_text())
    assert record["start_usage_usd"] == 10.0
    assert [b["budget_usd"] for b in record["budgets"]] == [5.0, 7.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "record", "reason"),
    [
        (None, None, "budget_unknown"),  # the key info carries no usage
        (3.0, {"start_usage_usd": 9.0, "budgets": []}, "budget_invalid"),  # another key
        (3.0, {"budgets": []}, "budget_unknown"),  # an unreadable record
    ],
    ids=["no_usage", "usage_below_start", "record_without_start"],
)
async def test_a_budget_that_cannot_be_established_sends_nothing(
    tmp_path, usage, record, reason
):
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = usage
    if record is not None:
        path = tmp_path / "runs" / "_gateway" / "budget.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
    async with running(tmp_path, fake, budget_usd=5.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        response = await client.post(
            "/runs/u/actor/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused and gateway.pause_reason == reason
        assert gateway.budget is not None and "error" in gateway.budget
    assert response.status_code == 402 and fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_a_budget_pause_lifts_on_a_known_budget_even_without_a_balance(tmp_path):
    """A key without its own limit whose credits cannot be read has an unknown balance,
    which never lifts an upstream 402; the budget's own pause lifts once it is known."""
    fake = FakeUpstream(limit_remaining=None, credits=None)
    fake.usage = None
    async with running(tmp_path, fake, budget_usd=5.0, balance_ttl_s=0.0) as (
        gateway,
        client,
    ):
        assert gateway.paused and gateway.pause_reason == "budget_unknown"
        fake.usage = 2.0  # the first readable usage starts the budget
        ok = await client.post("/runs/k/actor/v1/chat/completions", json=chat_request())
        assert not gateway.paused
    assert ok.status_code == 200


def test_a_budget_must_be_positive():
    with pytest.raises(ValueError, match="budget"):
        gw_mod.Gateway(
            gw_mod.GatewayConfig(
                api_key="k", models=protocol.load_models(), budget_usd=0.0
            )
        )


def costing(usd: float) -> Reply:
    """A chat reply the provider bills ``usd`` for (the ledger's ``cost_total``)."""
    usage = {
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "total_tokens": 59,
        "cost": usd,
    }
    return Reply(json=chat_body(usage=usage))


def write_ledger(runs_dir: Path, run_id: str, rows: list[Any]) -> Path:
    """A ledger by hand: dict rows as JSON lines, str rows verbatim (torn lines)."""
    path = runs_dir / run_id / "gateway.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join((json.dumps(r) if isinstance(r, dict) else r) + "\n" for r in rows)
    path.write_text(text)
    return path


def experiment_record(path: Path, start: float, *amounts: float) -> Path:
    """An experiment gateway's key-basis ``budget.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "start_usage_usd": start,
                "started_at": "2026-09-01T00:00:00+00:00",
                "budgets": [
                    {"budget_usd": a, "set_at": "t", "usage_usd": start}
                    for a in amounts
                ],
            }
        )
    )
    return path


@pytest.mark.asyncio
async def test_a_ledger_budget_counts_only_this_runs_directorys_ledgers(tmp_path):
    """On the ledger basis the key's usage (the experiment's spend on the same
    key) never uses the budget up; the cost_total of this runs directory's own calls
    does, and the call that spends it pauses the gateway at once (budget_spent, a stop
    reason), without waiting for the next balance check."""
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 40.0
    fake.queue("chat/completions", *(costing(0.4) for _ in range(4)))
    async with running(
        tmp_path, fake, budget_usd=1.0, budget_basis="ledger", balance_ttl_s=3600.0
    ) as (gateway, client):
        health = (await client.get("/health")).json()
        assert health["paused"] is False and health["budget_basis"] == "ledger"
        view = health["budget"]
        assert view["basis"] == "ledger" and view["start_ledger_usd"] == 0.0
        assert (view["spent_usd"], view["left_usd"]) == (0.0, 1.0)
        assert "start_usage_usd" not in view
        fake.usage = 140.0  # the experiment spent 100 USD on the key meanwhile
        await gateway.refresh_balance(force=True)
        assert not gateway.paused
        statuses = []
        for _ in range(4):
            response = await client.post(
                "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
            )
            statuses.append(response.status_code)
        health = (await client.get("/health")).json()
    assert statuses == [200, 200, 200, 402]
    assert len(fake.calls("chat/completions")) == 3
    assert health["paused"] and health["pause_reason"] == "budget_spent"
    view = health["budget"]
    assert view["spent_usd"] == pytest.approx(1.2) and view["left_usd"] < 0
    assert view["ledger_usd"] == pytest.approx(1.2)
    assert view["unknown_cost_attempts"] == 0
    assert fake.key_calls == 2  # the pause came from the ledger count, not from /key
    record = json.loads(
        (tmp_path / "runs" / "_gateway" / "budget_ledger.json").read_text()
    )
    assert record["start_ledger_usd"] == 0.0
    assert [b["budget_usd"] for b in record["budgets"]] == [1.0]
    assert record["budgets"][0]["ledger_usd"] == 0.0
    assert not (tmp_path / "runs" / "_gateway" / "budget.json").exists()
    rows = (tmp_path / "runs" / "_gateway" / "balance.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["budget"]["basis"] == "ledger"


@pytest.mark.asyncio
async def test_a_ledger_budget_counts_another_gateways_calls_at_each_balance_check(
    tmp_path,
):
    """A second gateway keeping its ledgers in the same runs directory (another one on
    another port) spends the same budget: its calls are seen at the next
    balance check, and nothing is counted twice (the larger of the ledgers' total and
    the gateway's own count)."""
    fake = FakeUpstream(limit_remaining=1000.0)
    runs = tmp_path / "runs"
    async with running(
        tmp_path, fake, budget_usd=1.0, budget_basis="ledger", balance_ttl_s=3600.0
    ) as (gateway, client):
        write_ledger(runs, "grade-other", [{"event": "call", "cost_total": 0.7}])
        ok = await client.post(
            "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
        )
        assert ok.status_code == 200 and not gateway.paused
        assert gateway.budget is not None
        assert gateway.budget["spent_usd"] == pytest.approx(0.000123)  # not seen yet
        await gateway.refresh_balance(force=True)
        assert gateway.budget["spent_usd"] == pytest.approx(0.700123)
        assert not gateway.paused
        write_ledger(
            runs,
            "grade-other",
            [
                {"event": "call", "cost_total": 0.7},
                {"event": "call", "cost_total": 0.4},
            ],
        )
        await gateway.refresh_balance(force=True)
        assert gateway.paused and gateway.pause_reason == "budget_spent"
        assert gateway.budget["spent_usd"] == pytest.approx(1.100123)
        await gateway.refresh_balance(force=True)  # counted once, however often read
        assert gateway.budget["spent_usd"] == pytest.approx(1.100123)


@pytest.mark.asyncio
async def test_a_restarted_ledger_gateway_keeps_its_start(tmp_path):
    """The start is the ledgers' total when a ledger budget was first set here
    (ledger lines before it never count); a restart keeps it; a new amount counts from
    the same start; torn lines, non-call lines and non-numeric costs are skipped; the
    attempts of unknown cost are counted apart; ledgers removed below the start are
    budget_invalid."""
    runs = tmp_path / "runs"
    write_ledger(
        runs,
        "grade-old",
        [
            {"event": "call", "cost_total": 0.3, "unknown_cost_attempts": 2},
            {"event": "phase", "cost_total": 50.0},  # not a final call line
            {"event": "call", "cost_total": None},
            {"event": "call", "cost_total": True},
            '{"event": "call", "cost_total": 9',  # torn (X15)
        ],
    )
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.queue("chat/completions", costing(0.4))
    async with running(
        tmp_path, fake, budget_usd=1.0, budget_basis="ledger", balance_ttl_s=0.0
    ) as (gateway, client):
        assert gateway.budget is not None
        assert gateway.budget["start_ledger_usd"] == pytest.approx(0.3)
        assert gateway.budget["spent_usd"] == 0.0
        assert gateway.budget["unknown_cost_attempts"] == 2
        ok = await client.post(
            "/runs/grade-new/judge/v1/chat/completions", json=chat_request()
        )
        assert ok.status_code == 200
        assert gateway.budget["spent_usd"] == pytest.approx(0.4)
    async with running(
        tmp_path, fake, budget_usd=2.0, budget_basis="ledger", balance_ttl_s=0.0
    ) as (gateway, _client):
        assert gateway.budget is not None
        assert gateway.budget["start_ledger_usd"] == pytest.approx(0.3)
        assert gateway.budget["ledger_usd"] == pytest.approx(0.7)
        assert gateway.budget["spent_usd"] == pytest.approx(0.4)
        assert gateway.budget["left_usd"] == pytest.approx(1.6)
        assert not gateway.paused
    record = json.loads((runs / "_gateway" / "budget_ledger.json").read_text())
    assert record["start_ledger_usd"] == pytest.approx(0.3)
    assert [b["budget_usd"] for b in record["budgets"]] == [1.0, 2.0]
    assert record["budgets"][1]["ledger_usd"] == pytest.approx(0.7)
    # Ledgers removed: less than the start is never read as money left.
    for path in runs.glob("*/gateway.jsonl"):
        path.unlink()
    async with running(
        tmp_path, fake, budget_usd=2.0, budget_basis="ledger", balance_ttl_s=0.0
    ) as (gateway, _client):
        assert gateway.paused and gateway.pause_reason == "budget_invalid"
        assert (
            gateway.budget is not None and "ledgers removed?" in gateway.budget["error"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["unreadable_ledger", "record_without_start"])
async def test_a_ledger_budget_that_cannot_be_counted_sends_nothing(tmp_path, problem):
    """A ledger that cannot be read, or a record without its start, is
    budget_unknown (waitable, never an undercount): nothing is sent."""
    runs = tmp_path / "runs"
    if problem == "unreadable_ledger":
        (runs / "grade-x" / "gateway.jsonl").mkdir(parents=True)  # read_bytes fails
    else:
        path = runs / "_gateway" / "budget_ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"budgets": []}))
    fake = FakeUpstream(limit_remaining=1000.0)
    async with running(
        tmp_path, fake, budget_usd=1.0, budget_basis="ledger", balance_ttl_s=0.0
    ) as (gateway, client):
        response = await client.post(
            "/runs/grade-y/judge/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused and gateway.pause_reason == "budget_unknown"
        assert gateway.budget is not None and gateway.budget["error"]
        assert gateway.budget["basis"] == "ledger"
    assert response.status_code == 402 and fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_budget_basis_is_in_the_start_record_and_health(tmp_path):
    """starts.jsonl and /health say what the budget counts and which
    experiment records it enforces; the key basis stays the default."""
    record = experiment_record(tmp_path / "exp" / "_gateway" / "budget.json", 1.0, 250)
    fake = FakeUpstream(limit_remaining=1000.0)
    async with running(
        tmp_path,
        fake,
        budget_usd=5.0,
        budget_basis="ledger",
        experiment_budgets=(record,),
    ) as (_gateway, client):
        health = (await client.get("/health")).json()
    assert health["budget_basis"] == "ledger"
    assert [e["file"] for e in health["budget"]["experiment"]] == [str(record)]
    starts = [
        json.loads(line)
        for line in (tmp_path / "runs" / "_gateway" / "starts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert starts[-1]["budget_basis"] == "ledger"
    assert starts[-1]["experiment_budgets"] == [str(record)]
    async with running(tmp_path / "key", fake, budget_usd=5.0) as (_gateway, client):
        health = (await client.get("/health")).json()
    assert health["budget_basis"] == "key" and health["budget"]["basis"] == "key"
    assert "experiment" not in health["budget"]
    async with running(tmp_path / "none", fake) as (_gateway, client):
        health = (await client.get("/health")).json()
    assert health["budget_basis"] == "key" and health["budget"] is None


@pytest.mark.asyncio
async def test_an_experiment_cap_pauses_a_ledger_gateway_once_the_key_spent_it(
    tmp_path,
):
    """A ledger-basis gateway on the experiment's key stops once the key has spent the
    experiment record's last amount from its start, whatever its own ledger budget says;
    the record is read again when it changes (a person raised it)."""
    record = experiment_record(tmp_path / "exp" / "_gateway" / "budget.json", 100, 250)
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 349.0
    async with running(
        tmp_path,
        fake,
        budget_usd=5.0,
        budget_basis="ledger",
        experiment_budgets=(record,),
        balance_ttl_s=0.0,
    ) as (gateway, client):
        assert not gateway.paused
        assert gateway.budget is not None
        (cap,) = gateway.budget["experiment"]
        assert cap["spent_usd"] == 249.0 and cap["left_usd"] == 1.0 and cap["checked"]
        assert (cap["start_usage_usd"], cap["budget_usd"]) == (100.0, 250.0)
        ok = await client.post(
            "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
        )
        fake.usage = 351.0
        refused = await client.post(
            "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused and gateway.pause_reason == "budget_spent"
        assert gateway.budget["spent_usd"] < 0.001  # its own ledger budget is untouched
        assert gateway.budget["experiment"][0]["left_usd"] == -1.0
        experiment_record(record, 100, 250, 300)  # the cap is raised to 300 USD
        again = await client.post(
            "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
        )
        assert not gateway.paused
        assert gateway.budget["experiment"][0]["left_usd"] == 49.0
    assert (ok.status_code, refused.status_code, again.status_code) == (200, 402, 200)
    assert len(fake.calls("chat/completions")) == 2
    # A key-basis gateway may enforce a cap too.
    fake.usage = 351.0
    async with running(
        tmp_path / "key",
        fake,
        budget_usd=500.0,
        experiment_budgets=(experiment_record(tmp_path / "e2.json", 100, 250),),
        balance_ttl_s=0.0,
    ) as (gateway, _client):
        assert gateway.paused and gateway.pause_reason == "budget_spent"
        assert gateway.budget is not None and gateway.budget["left_usd"] == 500.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["missing", "not_json", "no_start", "no_budgets", "no_usage"]
)
async def test_an_unreadable_experiment_record_pauses_waitably(tmp_path, problem):
    """A cap that cannot be read (or no usage to count it on) is never guessed:
    budget_unknown, a waitable pause, the error in the view."""
    path = tmp_path / "exp" / "budget.json"
    fake = FakeUpstream(limit_remaining=1000.0)
    if problem == "not_json":
        path.parent.mkdir(parents=True)
        path.write_text("{torn")
    elif problem == "no_start":
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"budgets": [{"budget_usd": 250}]}))
    elif problem == "no_budgets":
        experiment_record(path, 100)
    elif problem == "no_usage":
        experiment_record(path, 100, 250)
        fake.usage = None
    async with running(
        tmp_path,
        fake,
        budget_usd=5.0,
        budget_basis="ledger",
        experiment_budgets=(path,),
        balance_ttl_s=0.0,
    ) as (gateway, client):
        response = await client.post(
            "/runs/grade-x/judge/v1/chat/completions", json=chat_request()
        )
        assert gateway.paused and gateway.pause_reason == "budget_unknown"
        assert gateway.pause_reason in gw_mod.WAITABLE_PAUSE_REASONS
        assert gateway.budget is not None
        assert gateway.budget["experiment"][0]["error"]
    assert response.status_code == 402 and fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_an_experiment_record_of_another_key_is_recorded_not_checked(tmp_path):
    """A key usage below the record's start means this gateway's key is not the
    experiment's: the cap is shown as not checked and does not pause."""
    record = experiment_record(tmp_path / "budget.json", 100, 250)
    fake = FakeUpstream(limit_remaining=1000.0)
    fake.usage = 3.0
    async with running(
        tmp_path,
        fake,
        budget_usd=5.0,
        budget_basis="ledger",
        experiment_budgets=(record,),
    ) as (gateway, _client):
        assert not gateway.paused
        assert gateway.budget is not None
        (cap,) = gateway.budget["experiment"]
        assert cap["checked"] is False and cap["left_usd"] is None
        assert "another key" in cap["error"]


def test_worse_pause_prefers_a_stop_reason():
    assert gw_mod._worse_pause(None, None) is None
    assert gw_mod._worse_pause("budget_unknown", None) == "budget_unknown"
    assert gw_mod._worse_pause(None, "budget_spent") == "budget_spent"
    assert gw_mod._worse_pause("budget_unknown", "budget_spent") == "budget_spent"
    assert gw_mod._worse_pause("budget_invalid", "budget_spent") == "budget_invalid"


def test_ledger_spend_reads_final_call_lines_only(tmp_path):
    write_ledger(
        tmp_path,
        "a",
        [
            {"event": "call", "cost_total": 0.25, "unknown_cost_attempts": 1},
            {"event": "attempt", "cost_total": 7.0},
            "",
            "not json",
            '["a", "list"]',  # JSON, not an object
        ],
    )
    write_ledger(tmp_path, "b", [{"event": "call", "cost_total": 1}])
    (tmp_path / "_gateway").mkdir()
    (tmp_path / "_gateway" / "balance.jsonl").write_text('{"event": "call"}\n')
    assert gw_mod.ledger_spend(tmp_path) == (1.25, 1)
    assert gw_mod.ledger_spend(tmp_path / "nothing") == (0.0, 0)


def test_a_budget_basis_is_validated_and_needs_a_budget():
    models = protocol.load_models()
    with pytest.raises(ValueError, match="budget_basis"):
        gw_mod.Gateway(
            gw_mod.GatewayConfig(
                api_key="k", models=models, budget_usd=1.0, budget_basis="usage"
            )
        )
    for extra in (
        {"budget_basis": "ledger"},
        {"experiment_budgets": (Path("/x/budget.json"),)},
    ):
        with pytest.raises(ValueError, match="need budget_usd"):
            gw_mod.Gateway(gw_mod.GatewayConfig(api_key="k", models=models, **extra))


def test_cli_needs_a_budget_for_the_ledger_basis_or_a_cap(capsys):
    for argv in (
        ["--budget-basis", "ledger"],
        ["--experiment-budget", "/x/budget.json"],
    ):
        with pytest.raises(SystemExit) as info:
            gw_mod.main(argv)
        assert info.value.code == 2
        assert "need --budget" in capsys.readouterr().err
    with pytest.raises(SystemExit) as info:
        gw_mod.main(["--budget", "5", "--budget-basis", "usage"])
    assert info.value.code == 2


@pytest.mark.asyncio
async def test_a_forced_refresh_joins_the_fetch_in_flight(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake, balance_ttl_s=3600) as (gateway, _):
        await asyncio.gather(*(gateway.refresh_balance(force=True) for _ in range(10)))
        assert fake.key_calls == 1
        await gateway.refresh_balance()  # fresh cache: no fetch
        assert fake.key_calls == 1
        await gateway.refresh_balance(force=True)
        assert fake.key_calls == 2


@pytest.mark.asyncio
async def test_a_hold_never_outlasts_the_deadline(tmp_path):
    fake = FakeUpstream(limit_remaining=5.0)
    async with running(
        tmp_path,
        fake,
        balance_floor_usd=10.0,
        balance_ttl_s=0.05,
        pause_hold_s=600.0,
        deadline_s=0.6,
    ) as (_, client):
        started = time.monotonic()
        response = await client.post(
            "/runs/hold2/actor/v1/chat/completions", json=chat_request()
        )
    assert response.status_code == 402 and time.monotonic() - started < 5
    assert response.json()["error"]["held_s"] >= 0.5
    assert fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_abandoned_runs_are_refused(tmp_path):
    """W24: a restarted learning attempt's orphaned MIRIX server cannot keep spending."""
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        ok = await client.post(
            "/runs/old/mirix/v1/chat/completions", json=chat_request()
        )
        abandon = await client.post(
            "/runs/old/abandon", json={"reason": "learning attempt 1 invalid"}
        )
        refused = await client.post(
            "/runs/old/mirix/v1/chat/completions", json=chat_request()
        )
        other = await client.post(
            "/runs/new/mirix/v1/chat/completions", json=chat_request()
        )
    assert (ok.status_code, abandon.status_code) == (200, 200)
    assert (
        refused.status_code == 410
        and refused.json()["error"]["type"] == "run_abandoned"
    )
    assert other.status_code == 200
    assert len(fake.calls("chat/completions")) == 2
    async with running(tmp_path, fake) as (_, client):  # survives a gateway restart
        again = await client.post(
            "/runs/old/mirix/v1/chat/completions", json=chat_request()
        )
    assert again.status_code == 410


@pytest.mark.asyncio
async def test_identical_requests_after_a_failure_are_flagged(tmp_path):
    """W9: SDK-level retries inside MIRIX show up as repeats in the ledger."""
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(503) for _ in range(2)], Reply())
    async with running(tmp_path, fake, max_retries=1) as (_, client):
        first = await client.post(
            "/runs/rep/mirix/v1/chat/completions", json=chat_request()
        )
        second = await client.post(
            "/runs/rep/mirix/v1/chat/completions", json=chat_request()
        )
    assert first.status_code == 503 and second.status_code == 200
    rows = final_calls(tmp_path, "rep")
    assert "repeat_of" not in rows[0]
    assert rows[1]["repeat_of"] == rows[0]["call_id"]
    assert rows[1]["repeat_after_status"] == "infra_error"


@pytest.mark.asyncio
async def test_calls_and_starts_record_the_effective_protocol_settings(tmp_path):
    """W14 / W25: a non-protocol gateway cannot pass for a protocol one."""
    fake = FakeUpstream()
    backoff = protocol.GATEWAY_BACKOFF_BASE_S  # the fakes shorten it by default
    async with running(tmp_path, fake, backoff_base_s=backoff) as (gateway, client):
        await client.post("/runs/ok/actor/v1/chat/completions", json=chat_request())
        health = (await client.get("/health")).json()
    assert gateway.protocol_conformant is True
    assert health["effort_supported"] == {LUNA: True, LUNA56: True}
    assert health["settings"]["deadline_s"] == protocol.CALL_DEADLINE_S
    row = final_calls(tmp_path, "ok")[0]
    assert row["protocol_conformant"] is True
    assert row["gateway_settings_sha256"] == gateway.settings_sha256
    assert row["reasoning_effort_supported"] is True
    starts = [
        json.loads(line)
        for line in (tmp_path / "runs" / "_gateway" / "starts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert starts[-1]["protocol_conformant"] is True
    assert starts[-1]["protocol_fingerprint"] == protocol.protocol_fingerprint()
    assert starts[-1]["token_estimate"] == dict(protocol.TOKEN_ESTIMATE)
    assert FAKE_UPSTREAM_KEY not in json.dumps(starts)
    async with running(tmp_path, fake, deadline_s=1200.0, max_retries=10) as (
        gateway,
        client,
    ):
        await client.post("/runs/off/actor/v1/chat/completions", json=chat_request())
    assert gateway.protocol_conformant is False
    assert final_calls(tmp_path, "off")[0]["protocol_conformant"] is False


@pytest.mark.asyncio
async def test_small_inputs_keep_the_max_out80_cap_and_record_its_bound(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post(
            "/runs/small/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    row = final_calls(tmp_path, "small")[0]
    assert row["max_tokens_rewritten"] == 102_400 == row["max_tokens_f9"]
    assert row["max_tokens_bound"] == "max_out80"
    assert row["max_tokens_provider"] > row["max_tokens_rewritten"]


@pytest.mark.asyncio
async def test_a_different_tokenizer_margin_is_not_the_protocol(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake, tokenizer_margin=Fraction(1, 5)) as (
        gateway,
        client,
    ):
        await client.post(
            "/runs/m/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert gateway.protocol_conformant is False
    assert gateway.limits[LUNA56].tokenizer_margin == Fraction(1, 5)
    row = final_calls(tmp_path, "m")[0]
    assert row["protocol_conformant"] is False and row["tokenizer_margin"] == "1/5"


def test_cli_refuses_protocol_overrides_without_the_flag(capsys):
    for argv in (
        ["--deadline", "1200"],
        ["--max-retries", "10"],
        ["--tokenizer-margin", "1/5"],
    ):
        with pytest.raises(SystemExit) as info:
            gw_mod.main(argv)
        assert info.value.code == 2
        assert "--allow-protocol-overrides" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_file_parts_are_refused_instead_of_counted_as_base64(tmp_path):
    """W19: a 1.5 MB PDF must not be 'estimated' as ~1.4M text tokens."""
    pdf = "data:application/pdf;base64," + base64.b64encode(b"%PDF" * 400_000).decode()
    caller = {
        "model": LUNA,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarise"},
                    {"type": "file", "file": {"filename": "a.pdf", "file_data": pdf}},
                ],
            }
        ],
    }
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        response = await client.post("/runs/pdf/mirix/v1/chat/completions", json=caller)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_content_part"
    assert fake.calls("chat/completions") == []


# --- more gateway edge cases -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_call_tagged_with_another_phase_than_the_run_is_refused(tmp_path):
    """X25/X6: a learner or judge that tags its call 'learning' after run.py put the run in
    its test phase is refused (409 phase_mismatch). Every line records the run's phase and
    the caller's tag apart, so the audit never trusts a self-declared phase."""
    fake = FakeUpstream()
    item = {"x-skill-bench-item": "t0123456789abcdef0123"}
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/ph/phase", json={"phase": "learning"})
        ok = await client.post(
            "/runs/ph/judge/v1/chat/completions",
            json=chat_request(),
            headers={**item, "x-skill-bench-phase": "learning"},
        )
        await client.post("/runs/ph/phase", json={"phase": "test"})
        forged = await client.post(
            "/runs/ph/judge/v1/chat/completions",
            json=chat_request(),
            headers={**item, "x-skill-bench-phase": "learning"},
        )
        untagged = await client.post(
            "/runs/ph/actor/v1/chat/completions", json=chat_request(), headers=item
        )
    assert ok.status_code == 200 and untagged.status_code == 200
    assert forged.status_code == 409
    assert forged.json()["error"]["type"] == "phase_mismatch"
    calls = final_calls(tmp_path, "ph")
    assert [
        (c["role"], c["status"], c["run_phase"], c["phase_header"], c["phase"])
        for c in calls
    ] == [
        ("judge", "ok", "learning", "learning", "learning"),
        ("judge", "rejected", "test", "learning", "learning"),
        ("actor", "ok", "test", None, "test"),
    ]
    assert len(fake.calls("chat/completions")) == 2  # the forged call never went out
    started = [r for r in ledger(tmp_path, "ph") if r["event"] == "call_started"]
    assert [r["run_phase"] for r in started] == ["learning", "test"]


@pytest.mark.asyncio
async def test_every_ledger_line_names_the_gateway_process_that_wrote_it(tmp_path):
    """X34: a start id on the start record and on every ledger line (pending lines too),
    so the audit matches a call to its own gateway even when gateways overlap."""
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (first, client):
        await client.post("/runs/sid/phase", json={"phase": "learning"})
        await client.post("/runs/sid/actor/v1/chat/completions", json=chat_request())
        health = (await client.get("/health")).json()
    async with running(tmp_path, fake) as (second, client):
        await client.post("/runs/sid/actor/v1/chat/completions", json=chat_request())
    assert first.start_id != second.start_id and len(first.start_id) == 16
    assert health["start_id"] == first.start_id and health["pid"] == os.getpid()
    rows = ledger(tmp_path, "sid")
    assert all(r["gateway_start"] in (first.start_id, second.start_id) for r in rows)
    assert [r["gateway_start"] for r in rows if r["event"] == "call"] == [
        first.start_id,
        second.start_id,
    ]
    starts = [
        json.loads(line)
        for line in (tmp_path / "runs" / "_gateway" / "starts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [s["start_id"] for s in starts] == [first.start_id, second.start_id]


@pytest.mark.asyncio
async def test_a_torn_ledger_tail_does_not_swallow_the_next_call(tmp_path):
    """X15: the gateway killed (or the disk full) mid-append leaves a fragment; the next
    line starts on a line of its own, so the call after it is never lost."""
    from evals.skill_bench.cell_records import read_jsonl_counted, summarize_ledger

    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/tt/phase", json={"phase": "test"})
        await client.post("/runs/tt/actor/v1/chat/completions", json=chat_request())
    path = tmp_path / "runs" / "tt" / "gateway.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "call", "call_id": "torn-call", "cost_tot')
    async with running(tmp_path, fake) as (_, client):
        again = await client.post(
            "/runs/tt/actor/v1/chat/completions", json=chat_request()
        )
    assert again.status_code == 200
    rows, torn = read_jsonl_counted(path)
    assert torn == 1
    ids = [r["call_id"] for r in rows if r.get("event") == "call"]
    assert again.headers["x-skill-bench-call-id"] in ids and len(ids) == 2
    summary = summarize_ledger(path)
    assert summary["torn_lines"] == 1 and summary["unfinished_calls"] == 0


def test_the_users_lock_of_a_loopback_gateway_is_one_file(tmp_path):
    """X12: 'localhost' and '127.0.0.1' name one gateway, so one users lock."""
    assert gw_mod.gateway_users_lock(tmp_path, "localhost", 8765) == (
        tmp_path / "_gateway" / "gateway-127.0.0.1-8765.users.lock"
    )
    assert gw_mod.gateway_users_lock(
        tmp_path, "127.0.0.1", 8765
    ) == gw_mod.gateway_users_lock(tmp_path, "localhost", 8765)
    assert gw_mod.gateway_owner_lock(tmp_path, "localhost", 8765).name == (
        "gateway-127.0.0.1-8765.lock"
    )


@pytest.mark.asyncio
async def test_health_names_the_users_lock(tmp_path):
    fake = FakeUpstream()
    lock = tmp_path / "runs" / "_gateway" / "gateway-127.0.0.1-1.users.lock"
    async with running(tmp_path, fake, users_lock=lock) as (gateway, client):
        health = (await client.get("/health")).json()
    assert health["users_lock"] == str(lock)
    assert health["start_id"] == gateway.start_id


# --- learning checkpoint and resume markers -----------------------------------------------


SHA = "ab" * 32


def checkpoint_mark(position: int = 3, kind: str = "played") -> dict[str, Any]:
    return {
        "event": "checkpoint",
        "id": f"c{position:04d}-{kind}",
        "kind": kind,
        "position": position,
        "snapshot_sha256": SHA,
    }


def resume_mark(segment: int = 1, **fields: Any) -> dict[str, Any]:
    body = {
        "event": "resume",
        "segment": segment,
        "checkpoint": "c0003-played",
        "checkpoint_offset": 0,
        "launch": "l1",
    }
    return {**body, **fields}


def test_the_ledger_events_are_the_ones_the_records_name():
    from evals.skill_bench import cell_records

    assert gw_mod.LEDGER_EVENT_CHECKPOINT == cell_records.LEDGER_EVENT_CHECKPOINT
    assert gw_mod.LEDGER_EVENT_RESUME == cell_records.LEDGER_EVENT_RESUME
    assert gw_mod.MARK_EVENTS == ("checkpoint", "resume")


@pytest.mark.asyncio
async def test_a_mark_is_an_fsynced_ledger_line_at_the_offset_it_answers(
    tmp_path, monkeypatch
):
    """The marker goes into the ledger in its own write order, between calls, stamped
    like every line; the answer's offset is where its line starts, as
    ``cell_records.read_ledger_lines`` counts it."""
    from evals.skill_bench.cell_records import read_ledger_lines

    fsynced: list[int] = []
    real_fsync = os.fsync

    def counting(fd: int) -> None:
        fsynced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(gw_mod.os, "fsync", counting)
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (gateway, client):
        await client.post("/runs/mk/phase", json={"phase": "learning"})
        await client.post("/runs/mk/actor/v1/chat/completions", json=chat_request())
        before = len(fsynced)
        first = await client.post("/runs/mk/mark", json=checkpoint_mark())
        assert len(fsynced) > before
        await client.post("/runs/mk/judge/v1/chat/completions", json=chat_request())
        second = await client.post(
            "/runs/mk/mark",
            json=resume_mark(checkpoint_offset=first.json()["offset"]),
        )
    assert first.status_code == second.status_code == 200
    lines = read_ledger_lines(tmp_path / "runs" / "mk" / "gateway.jsonl")
    by_offset = {line.offset: line.row for line in lines}
    for answer, event in ((first.json(), "checkpoint"), (second.json(), "resume")):
        assert answer["ok"] is True
        row = by_offset[answer["offset"]]
        assert row["event"] == event and row["ts"] == answer["ts"]
        assert row["run_id"] == "mk" and row["gateway_start"] == gateway.start_id
    events = [line.row["event"] for line in lines]
    assert events == [
        "phase",
        "call_started",
        "call",
        "checkpoint",
        "call_started",
        "call",
        "resume",
    ]
    checkpoint = by_offset[first.json()["offset"]]
    assert {k: checkpoint[k] for k in checkpoint_mark()} == checkpoint_mark()


@pytest.mark.asyncio
async def test_the_ledger_returns_where_each_line_starts_even_after_a_torn_tail(
    tmp_path,
):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (gateway, _client):
        path = tmp_path / "runs" / "off" / "gateway.jsonl"
        a = await gateway.ledger("off", {"event": "phase", "phase": "learning"})
        b = await gateway.ledger("off", {"event": "phase", "phase": "test"})
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"event": "call", "call_id": "torn')
        c = await gateway.ledger("off", {"event": "phase", "phase": "learning"})
    data = path.read_bytes()
    assert a == 0 and data[b - 1 : b] == b"\n"
    for offset, phase in ((a, "learning"), (b, "test"), (c, "learning")):
        line = data[offset:].split(b"\n", 1)[0]
        assert json.loads(line)["phase"] == phase
    assert data[c - 1 : c] == b"\n"  # the fragment was ended first


@pytest.mark.asyncio
async def test_a_mark_goes_after_the_lines_a_failed_write_kept_in_memory(tmp_path):
    """A final line kept in memory (a failed write) goes out before the marker, and the
    answer's offset is still the marker's."""
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (gateway, client):
        gateway._unlogged["kept"] = [
            {"event": "call", "call_id": "late", "run_id": "kept"}
        ]
        answer = (await client.post("/runs/kept/mark", json=checkpoint_mark())).json()
    rows = ledger(tmp_path, "kept")
    assert [r["event"] for r in rows] == ["call", "checkpoint"]
    data = (tmp_path / "runs" / "kept" / "gateway.jsonl").read_bytes()
    assert json.loads(data[answer["offset"] :].split(b"\n", 1)[0])["event"] == (
        "checkpoint"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"event": "probe"},  # run.py's probe of the route
        {"event": "call", "call_id": "x"},
        {"event": "checkpoint"},
        {**checkpoint_mark(), "kind": "done"},  # the id says played
        {**checkpoint_mark(), "position": 4},
        {**checkpoint_mark(), "position": True},
        {**checkpoint_mark(0, "init"), "id": "c0001-init", "position": 1},
        {**checkpoint_mark(), "snapshot_sha256": "not-a-sha"},
        {**checkpoint_mark(), "ts": "forged"},
        {**checkpoint_mark(), "gateway_start": "forged"},
        {**checkpoint_mark(), "call_id": "abc"},
        {**checkpoint_mark(), "nested": {"a": 1}},
        {**checkpoint_mark(), "listed": [1]},
        {**checkpoint_mark(), "Upper": 1},
        {**checkpoint_mark(), "long": "x" * 300},
        resume_mark(segment=0),
        resume_mark(checkpoint=None),  # offset without a checkpoint
        resume_mark(checkpoint_offset=None),
        resume_mark(checkpoint="latest"),
        resume_mark(checkpoint_offset=-1),
        resume_mark(launch=5),
        [checkpoint_mark()],
    ],
)
async def test_a_bad_mark_is_refused_before_anything_is_written(tmp_path, body):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        response = await client.post("/runs/bad/mark", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_mark"
    assert not (tmp_path / "runs" / "bad").exists()


@pytest.mark.asyncio
async def test_a_mark_is_refused_for_bad_json_size_credential_and_run_id(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        not_json = await client.post(
            "/runs/bad/mark",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        big = await client.post(
            "/runs/bad/mark",
            json={**checkpoint_mark(), **{f"f{i}": "y" * 200 for i in range(30)}},
        )
        stranger = await client.post(
            "/runs/bad/mark",
            json=checkpoint_mark(),
            headers={"Authorization": "Bearer sk-or-real-key"},
        )
        bad_id = await client.post("/runs/-bad/mark", json=checkpoint_mark())
    assert (not_json.status_code, big.status_code) == (400, 400)
    assert stranger.status_code == 401 and bad_id.status_code == 400
    assert not (tmp_path / "runs" / "bad").exists()


@pytest.mark.asyncio
async def test_a_mark_on_an_abandoned_run_is_410_and_writes_nothing(tmp_path):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post("/runs/gone/abandon", json={"reason": "operator re-roll"})
        before = ledger(tmp_path, "gone")
        refused = await client.post("/runs/gone/mark", json=resume_mark())
    assert refused.status_code == 410
    assert refused.json()["error"]["type"] == "run_abandoned"
    assert ledger(tmp_path, "gone") == before


@pytest.mark.asyncio
async def test_run_py_finds_the_mark_route_and_uses_its_offset(tmp_path):
    """The contract run.py relies on (GROUP-A, open issue 2): the probe gets 400 from
    this gateway (never 404/405), and a mark answers ok with an int offset."""
    from evals.skill_bench.tests.fakes import install_mark_route

    fake = FakeUpstream()
    fresh = gw_mod.Gateway(
        gw_mod.GatewayConfig(api_key="k", models=protocol.load_models())
    )
    assert install_mark_route(fresh) is False  # the gateway's own route, no shim
    async with running(tmp_path, fake) as (gateway, client):
        probe = await client.post(
            "/runs/skill-bench-mark-probe/mark", json={"event": "probe"}
        )
        answer = await client.post("/runs/r1/mark", json=checkpoint_mark(0, "init"))
        health = (await client.get("/health")).json()
    assert probe.status_code == 400
    assert not (tmp_path / "runs" / "skill-bench-mark-probe").exists()
    assert answer.status_code == 200 and isinstance(answer.json()["offset"], int)
    assert health["start_id"] == gateway.start_id


# --- provider policy ------------------------------------------------------------------------

#: A test-only provider policy (the protocol names no model): luna56 ignores two
#: providers.
POLICY = {LUNA56: {"ignore": ("open-inference", "wafer")}}


@pytest.fixture
def provider_policy(monkeypatch):
    """The protocol with :data:`POLICY` as its provider policy: the gateway's default
    routing, the providers' names and the settings a conformant gateway has."""
    monkeypatch.setattr(protocol, "PROVIDER_POLICY", POLICY)
    monkeypatch.setattr(
        protocol,
        "PROVIDER_SLUG_NAMES",
        {"open-inference": "OpenInference", "wafer": "Wafer"},
    )
    monkeypatch.setitem(
        gw_mod.PROTOCOL_SETTINGS,
        "provider_policy",
        protocol.provider_policy_settings(POLICY),
    )


@pytest.mark.asyncio
async def test_policy_calls_ignore_the_policy_providers_and_luna_is_unrestricted(
    tmp_path, provider_policy
):
    fake = FakeUpstream()
    async with running(tmp_path, fake) as (_, client):
        await client.post(
            "/runs/l56/actor/v1/chat/completions",
            json=chat_request(LUNA56, provider={"only": ["OpenInference"]}),
        )
        await client.post(
            "/runs/luna/actor/v1/chat/completions",
            json=chat_request(LUNA, provider={"only": ["Azure"]}),
        )
        await client.post(
            "/runs/emb/embedding/v1/embeddings",
            json={
                "model": protocol.EMBEDDING_MODEL,
                "input": "x",
                "provider": {"only": ["x"]},
            },
        )
    sent = [c.json for c in fake.calls("chat/completions")]
    assert sent[0]["provider"] == {"ignore": ["open-inference", "wafer"]}
    assert "provider" not in sent[1]
    assert "provider" not in fake.calls("embeddings")[0].json
    l56, luna = final_calls(tmp_path, "l56")[0], final_calls(tmp_path, "luna")[0]
    assert l56["wire"]["provider"] == {"ignore": ["open-inference", "wafer"]}
    assert "provider" not in luna["wire"]
    # The caller's own routing was dropped, and said so.
    assert "provider" in l56["dropped_fields"] and "provider" in luna["dropped_fields"]
    assert "provider" not in final_calls(tmp_path, "emb")[0]["wire"]


def test_the_policy_sits_last_in_the_rewritten_body(provider_policy):
    """The audit rebuilds the sent body from the ledger's wire (H10.2 request hashes):
    the provider object is the last key, after the gateway's own fields."""
    body = gw_mod.rewrite_chat_body(
        chat_request(LUNA56),
        model=LUNA56,
        max_tokens=10,
        seed=None,
        effort="medium",
        provider=protocol.provider_routing(LUNA56),
    )
    assert list(body)[-1] == "provider"
    plain = gw_mod.rewrite_chat_body(
        chat_request(LUNA), model=LUNA, max_tokens=10, seed=42, effort="medium"
    )
    assert "provider" not in plain and list(plain)[-1] == "transforms"


@pytest.mark.asyncio
async def test_the_policy_is_part_of_the_protocol_settings(tmp_path, provider_policy):
    """A gateway without the provider policy is not the protocol's gateway (H10.6): its
    settings hash and conformance say so on every call line and in starts.jsonl."""
    assert gw_mod.PROTOCOL_SETTINGS["provider_policy"] == {
        LUNA56: {"ignore": ["open-inference", "wafer"]}
    }
    fake = FakeUpstream()
    backoff = protocol.GATEWAY_BACKOFF_BASE_S
    async with running(tmp_path, fake, backoff_base_s=backoff) as (gateway, client):
        health = (await client.get("/health")).json()
    assert gateway.protocol_conformant is True
    assert health["provider_policy"] == gw_mod.PROTOCOL_SETTINGS["provider_policy"]
    assert health["settings"]["provider_policy"] == health["provider_policy"]
    async with running(tmp_path, fake, backoff_base_s=backoff, provider_policy={}) as (
        gateway,
        client,
    ):
        await client.post(
            "/runs/np/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert gateway.protocol_conformant is False
    row = final_calls(tmp_path, "np")[0]
    assert row["protocol_conformant"] is False and "provider" not in row["wire"]
    starts = [
        json.loads(line)
        for line in (tmp_path / "runs" / "_gateway" / "starts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert starts[0]["settings"]["provider_policy"] == {
        LUNA56: {"ignore": ["open-inference", "wafer"]}
    }
    assert starts[1]["settings"]["provider_policy"] == {}
    assert starts[1]["protocol_conformant"] is False


@pytest.mark.asyncio
async def test_no_endpoint_left_is_retried_as_provider_unavailability(tmp_path):
    """OpenRouter's 404 "no endpoints found" (every provider the routing leaves is down)
    is infrastructure: retried, then an INFRA failure; any other 404, and one that says
    no endpoint can take this request, is not retried."""
    from evals.skill_bench.infra.failures import FailureKind, classify_gateway_error

    fake = FakeUpstream()
    message = "No endpoints found for openai/gpt-5.6-luna."
    fake.queue(
        "chat/completions",
        Reply(404, {"error": {"code": 404, "message": message}}),
        Reply(),
    )
    async with running(tmp_path, fake) as (_, client):
        ok = await client.post(
            "/runs/ne/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert ok.status_code == 200
    attempts = final_calls(tmp_path, "ne")[0]["attempts"]
    assert [a["outcome"] for a in attempts] == ["no_endpoint", "ok"]
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        *[
            Reply(404, {"error": {"code": 404, "message": "No allowed providers"}})
            for _ in range(10)
        ],
    )
    async with running(tmp_path, fake, backoff_base_s=0.001) as (_, client):
        gone = await client.post(
            "/runs/ne2/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert gone.status_code == 502
    failure = classify_gateway_error(gone.status_code, gone.json(), "actor")
    assert failure.kind is FailureKind.INFRA
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(404, {"error": {"message": "Not Found"}}))
    async with running(tmp_path, fake) as (_, client):
        plain = await client.post(
            "/runs/ne3/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert plain.status_code == 404 and len(fake.calls("chat/completions")) == 1
    # A 404 that says no endpoint can take this request or this account is
    # never retried: it fails at once, for a person (HARNESS), not after 600 s as INFRA.
    for n, final in enumerate(
        (
            "No endpoints found that support tool use. Try disabling tools.",
            "No endpoints found that can handle the requested parameters.",
            "No endpoints found matching your data policy (Free model publication).",
            "No endpoints found matching your data policy and provider preferences",
        )
    ):
        fake = FakeUpstream()
        fake.queue("chat/completions", Reply(404, {"error": {"message": final}}))
        async with running(tmp_path, fake) as (_, client):
            refused = await client.post(
                f"/runs/nf{n}/actor/v1/chat/completions", json=chat_request(LUNA56)
            )
        assert refused.status_code == 404, final
        assert len(fake.calls("chat/completions")) == 1, final
        failure = classify_gateway_error(404, refused.json(), "actor")
        assert failure.kind is FailureKind.HARNESS, final


@pytest.mark.asyncio
async def test_an_answer_from_an_ignored_provider_is_never_returned(
    tmp_path, provider_policy
):
    """OpenRouter served a call from a provider its routing ignores (by
    name or slug): the gateway records the billed attempt, retries, and returns the next
    answer; the audit's ignored_provider_served stays a backstop. Other models, with no
    routing, take an answer from any provider."""
    from evals.skill_bench.infra.failures import FailureKind, classify_gateway_error

    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        Reply(200, chat_body(LUNA56, "from wafer", provider="Wafer")),
        Reply(200, chat_body(LUNA56, "from oi", provider="open-inference")),
        Reply(200, chat_body(LUNA56, "fine", provider="OpenAI")),
    )
    async with running(tmp_path, fake, backoff_base_s=0.001) as (_, client):
        ok = await client.post(
            "/runs/ig/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert ok.status_code == 200
    assert ok.json()["choices"][0]["message"]["content"] == "fine"
    row = final_calls(tmp_path, "ig")[0]
    assert [a["outcome"] for a in row["attempts"]] == [
        gw_mod.IGNORED_PROVIDER_OUTCOME,
        gw_mod.IGNORED_PROVIDER_OUTCOME,
        "ok",
    ]
    assert [a["provider"] for a in row["attempts"]] == [
        "Wafer",
        "open-inference",
        "OpenAI",
    ]
    assert row["attempts"][0]["error"]["ignored"] == "wafer"
    assert row["attempts"][1]["error"]["ignored"] == "open-inference"
    # Every refused answer was billed, and is in the call's cost.
    assert row["billed_attempts"] == 3 and row["retries"] == 2
    assert row["cost_total"] == pytest.approx(3 * 0.000123)
    assert row["provider"] == "OpenAI" and row["status"] == "ok"

    # Only ignored providers answering: an INFRA failure once the retries run out, and
    # no ignored answer ever reaches the caller.
    fake = FakeUpstream()
    fake.queue(
        "chat/completions",
        *[Reply(200, chat_body(LUNA56, "no", provider="Wafer")) for _ in range(10)],
    )
    async with running(tmp_path, fake, backoff_base_s=0.001) as (_, client):
        gone = await client.post(
            "/runs/ig2/actor/v1/chat/completions", json=chat_request(LUNA56)
        )
    assert gone.status_code == 502
    assert classify_gateway_error(502, gone.json(), "actor").kind is FailureKind.INFRA
    row = final_calls(tmp_path, "ig2")[0]
    assert row["status"] == "infra_error"
    assert {a["outcome"] for a in row["attempts"]} == {gw_mod.IGNORED_PROVIDER_OUTCOME}

    # luna has no routing: Wafer's answer is its answer.
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(200, chat_body(LUNA, "hi", provider="Wafer")))
    async with running(tmp_path, fake) as (_, client):
        luna = await client.post(
            "/runs/ig3/actor/v1/chat/completions", json=chat_request(LUNA)
        )
    assert luna.status_code == 200 and len(fake.calls("chat/completions")) == 1


def test_the_refusal_follows_the_routing_the_call_carried():
    """``refuse_ignored_provider`` checks the answer against the routing sent (a gateway
    started with another policy refuses by that one), and leaves every non-ok verdict
    and an answer without a provider alone."""
    ok = gw_mod._Verdict("ok", 200, chat_body(LUNA56, provider="Wafer"))
    routing = {"ignore": ["wafer"]}
    refused = gw_mod.refuse_ignored_provider(ok, routing)
    assert refused.kind == "retry" and refused.failure == "ignored_provider"
    assert gw_mod.refuse_ignored_provider(ok, None) is ok
    assert gw_mod.refuse_ignored_provider(ok, {"ignore": ["open-inference"]}) is ok
    anon = gw_mod._Verdict("ok", 200, chat_body(LUNA56, provider=""))
    assert gw_mod.refuse_ignored_provider(anon, routing) is anon
    failed = gw_mod._Verdict(
        "fail", 400, chat_body(LUNA56, provider="Wafer"), "http_400"
    )
    assert gw_mod.refuse_ignored_provider(failed, routing) is failed
    assert protocol.ignored_by(routing, "WAFER") == "wafer"
    assert protocol.ignored_by({"ignore": []}, "Wafer") is None
    assert protocol.ignored_by("not a routing", "Wafer") is None


# --- pauses and slots ----------------------------------------------------------------------


def test_every_pause_reason_is_waitable_or_a_stop():
    """The scheduler and supervisor sort /health's pause_reason by these two sets:
    waitable ones lift by themselves, stop ones need a person."""
    waitable, stop = gw_mod.WAITABLE_PAUSE_REASONS, gw_mod.STOP_PAUSE_REASONS
    assert not waitable & stop
    assert stop == {"budget_spent", "budget_invalid"}
    assert waitable == {"balance_below_floor", "upstream_402", "budget_unknown"}
    assert gw_mod.BUDGET_PAUSE_REASONS <= waitable | stop
    # Every reason the gateway can set is in one of them.
    source = Path(gw_mod.__file__).read_text(encoding="utf-8")
    import re as _re

    set_here = set(_re.findall(r'_pause\("([a-z_0-9]+)"\)', source))
    set_here |= {gw_mod.BUDGET_UNKNOWN, gw_mod.BUDGET_SPENT, gw_mod.BUDGET_INVALID}
    assert set_here <= waitable | stop, set_here


@pytest.mark.asyncio
async def test_health_reports_the_upstream_slots(tmp_path):
    import threading

    release = threading.Event()
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(hold=release), Reply())
    async with running(tmp_path, fake, max_concurrency=1) as (_, client):
        idle = (await client.get("/health")).json()["slots"]
        first = asyncio.create_task(
            client.post("/runs/s/actor/v1/chat/completions", json=chat_request())
        )
        second = asyncio.create_task(
            client.post(
                "/runs/s/actor/v1/chat/completions", json=chat_request(content="b")
            )
        )
        busy = idle
        for _ in range(200):
            busy = (await client.get("/health")).json()["slots"]
            if busy["in_use"] == 1 and busy["waiting"] == 1:
                break
            await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(first, second)
        after = (await client.get("/health")).json()["slots"]
    assert idle == {"max": 1, "in_use": 0, "waiting": 0}
    assert busy == {"max": 1, "in_use": 1, "waiting": 1}
    assert after == {"max": 1, "in_use": 0, "waiting": 0}
