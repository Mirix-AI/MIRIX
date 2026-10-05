"""The actor: gateway ``actor`` role calls and their records."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import fields

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import (
    Actor,
    ActorCall,
    ActorCallError,
    request_bytes,
)
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import (
    CONTENT_FILTER,
    GATEWAY_REJECTED,
    GATEWAY_UNREACHABLE,
    MODEL_TIMEOUT,
    TRUNCATED,
    UPSTREAM_INFRA,
    FailureKind,
)
from evals.skill_bench.infra.gateway import LOCAL_TOKEN
from evals.skill_bench.tests.fakes import FAKE_UPSTREAM_KEY, FakeUpstream, Reply
from evals.skill_bench.tests.rollout_support import (
    LUNA,
    RUN_ID,
    gateway_actor,
    ledger_calls,
    text_reply,
)

MESSAGES = [
    {"role": "system", "content": "You are an expert agent."},
    {"role": "user", "content": "MEMORY TEXT\n\nobservation"},
]
ITEM = "t" + "3" * 20


def all_record_fields() -> set[str]:
    return {f.name for f in fields(ActorCall)}


@pytest.mark.asyncio
async def test_successful_call_records_every_field(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", text_reply("<action>look</action>", reasoning="why"))
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        reply = await actor.complete(
            MESSAGES,
            index=0,
            item=ITEM,
            phase="test",
            memory="MEMORY TEXT",
            memory_tokens=tokens.count_text("MEMORY TEXT"),
        )
    call = reply.call
    record = call.to_dict()
    assert set(record) == all_record_fields()
    # The request, exactly as sent.
    assert call.request == {"model": LUNA, "messages": MESSAGES}
    sent = request_bytes(call.request)
    assert call.request_bytes == len(sent)
    assert call.request_sha256 == hashlib.sha256(sent).hexdigest()
    upstream = fake.calls("chat/completions")[0].json
    assert upstream["messages"] == MESSAGES
    # F9: the gateway applied the cap the actor expected from the same estimate.
    estimate = tokens.estimate_chat_input(call.request)
    assert call.input_tokens_estimated == estimate.total
    assert (
        call.max_tokens_expected
        == upstream["max_tokens"]
        == min(
            protocol.model_limits(LUNA).max_out80,
            protocol.model_limits(LUNA).window80 - estimate.total,
        )
    )
    # Memory, identity, timing.
    assert call.memory_sha256 == hashlib.sha256(b"MEMORY TEXT").hexdigest()
    assert call.memory_chars == 11 and call.memory_tokens == tokens.count_text(
        "MEMORY TEXT"
    )
    assert (call.index, call.role, call.model, call.item, call.phase) == (
        0,
        "actor",
        LUNA,
        ITEM,
        "test",
    )
    assert call.url == f"http://gateway/runs/{RUN_ID}/actor/v1/chat/completions"
    assert call.latency_s >= 0 and call.started_at.endswith("+00:00")
    # Response, usage and the join key into the gateway ledger.
    assert call.http_status == 200 and call.gateway_retries == 0
    assert call.response is not None and call.response["id"] == "gen-fake-1"
    assert call.usage is not None and call.usage["prompt_tokens"] == 42
    assert (call.cost, call.provider, call.generation_id, call.finish_reason) == (
        0.000123,
        "FakeProvider",
        "gen-fake-1",
        "stop",
    )
    assert call.failure is None and call.error is None
    ledger = ledger_calls(tmp_path)
    assert len(ledger) == 1
    row = ledger[0]
    assert row["call_id"] == call.gateway_call_id
    assert (row["role"], row["item"], row["phase"], row["status"]) == (
        "actor",
        ITEM,
        "test",
        "ok",
    )
    assert row["max_tokens_rewritten"] == call.max_tokens_expected
    assert row["input_tokens_estimated"] == call.input_tokens_estimated
    # The reply.
    assert reply.content == "<action>look</action>" and reply.reasoning == "why"
    assert reply.failure is None
    json.dumps(record)


@pytest.mark.asyncio
async def test_actor_sends_only_model_and_messages_with_the_local_token(tmp_path):
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            headers={"x-skill-bench-call-id": "abc", "x-skill-bench-retries": "2"},
        )

    actor = Actor(
        gateway_base_url="http://gw",
        run_id=RUN_ID,
        model=LUNA,
        transport=httpx.MockTransport(handler),
    )
    async with actor:
        reply = await actor.complete(MESSAGES, index=3, item=None, phase=None)
        plain = await actor.complete(MESSAGES, index=4, item=ITEM, phase="learning")
    body = json.loads(seen[0].content)
    assert body == {"model": LUNA, "messages": MESSAGES}
    assert set(json.loads(seen[1].content)) == {"model", "messages"}
    for request in seen:
        assert request.headers["authorization"] == f"Bearer {LOCAL_TOKEN}"
        assert FAKE_UPSTREAM_KEY not in request.headers.get("authorization", "")
    assert "x-skill-bench-item" not in seen[0].headers
    assert seen[1].headers["x-skill-bench-item"] == ITEM
    assert seen[1].headers["x-skill-bench-phase"] == "learning"
    assert reply.call.gateway_call_id == "abc" and reply.call.gateway_retries == 2
    assert reply.call.index == 3 and plain.call.index == 4
    assert reply.call.max_tokens_expected <= protocol.model_limits(LUNA).max_out80


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "code"),
    [("length", TRUNCATED), ("content_filter", CONTENT_FILTER)],
)
async def test_truncated_or_filtered_replies_carry_a_model_failure(
    tmp_path, finish_reason, code
):
    fake = FakeUpstream()
    fake.queue(
        "chat/completions", text_reply("<think>partial", finish_reason=finish_reason)
    )
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        reply = await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert reply.failure is not None
    assert reply.failure.kind is FailureKind.MODEL and reply.failure.code == code
    assert reply.call.failure is not None and reply.call.failure["code"] == code
    assert reply.call.finish_reason == finish_reason


@pytest.mark.asyncio
async def test_single_attempt_timeout_is_a_model_failure(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(delay_s=1.0))
    async with gateway_actor(tmp_path, fake, deadline_s=0.2) as (_gateway, actor):
        with pytest.raises(ActorCallError) as caught:
            await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    failure, call = caught.value.failure, caught.value.call
    assert failure.kind is FailureKind.MODEL and failure.code == MODEL_TIMEOUT
    assert failure.role == "actor"
    assert call.http_status == 504 and call.error is not None
    assert call.error["type"] == "deadline_exceeded"
    assert call.failure == failure.to_dict()
    assert call.request["messages"] == MESSAGES and call.response is not None


@pytest.mark.asyncio
async def test_exhausted_retries_are_infra(tmp_path):
    fake = FakeUpstream()
    fake.queue("chat/completions", *[Reply(status=503) for _ in range(3)])
    async with gateway_actor(tmp_path, fake, max_retries=2) as (_gateway, actor):
        with pytest.raises(ActorCallError) as caught:
            await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert caught.value.failure.kind is FailureKind.INFRA
    assert caught.value.failure.code == UPSTREAM_INFRA
    assert caught.value.call.gateway_retries == 2
    assert len(fake.calls("chat/completions")) == 3


@pytest.mark.asyncio
async def test_gateway_refusal_is_a_harness_failure(tmp_path):
    fake = FakeUpstream()
    async with gateway_actor(tmp_path, fake) as (gateway, _actor):
        wrong = Actor(
            gateway_base_url="http://gateway",
            run_id=RUN_ID,
            model=LUNA,
            token="not-the-local-token",
            transport=httpx.ASGITransport(app=gateway.app),
        )
        async with wrong:
            with pytest.raises(ActorCallError) as caught:
                await wrong.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert caught.value.failure.kind is FailureKind.HARNESS
    assert caught.value.failure.code == GATEWAY_REJECTED
    assert caught.value.call.http_status == 401
    assert fake.calls("chat/completions") == []


@pytest.mark.asyncio
async def test_unreachable_gateway_is_infra():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with Actor(
        gateway_base_url="http://gw",
        run_id=RUN_ID,
        model=LUNA,
        transport=httpx.MockTransport(handler),
    ) as actor:
        with pytest.raises(ActorCallError) as caught:
            await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert caught.value.failure.kind is FailureKind.INFRA
    assert caught.value.failure.code == GATEWAY_UNREACHABLE
    call = caught.value.call
    assert call.http_status is None and call.error is not None
    assert call.error["type"] == "ConnectError"


@pytest.mark.asyncio
async def test_invalid_gateway_success_body_is_a_harness_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    async with Actor(
        gateway_base_url="http://gw",
        run_id=RUN_ID,
        model=LUNA,
        transport=httpx.MockTransport(handler),
    ) as actor:
        with pytest.raises(ActorCallError) as caught:
            await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert caught.value.failure.kind is FailureKind.HARNESS
    assert caught.value.call.response == {"unexpected": True}


def test_actor_refuses_models_outside_the_protocol():
    with pytest.raises(KeyError):
        Actor(gateway_base_url="http://gw", run_id=RUN_ID, model="openai/gpt-4o")
    with pytest.raises(ValueError):
        Actor(gateway_base_url="http://gw", run_id="bad run id", model=LUNA)


@pytest.mark.asyncio
async def test_one_actor_serves_concurrent_episodes(tmp_path):
    fake = FakeUpstream(
        responder=lambda endpoint, body: text_reply(body["messages"][-1]["content"])
    )
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        replies = await asyncio.gather(
            *(
                actor.complete(
                    [{"role": "user", "content": f"echo {i}"}],
                    index=0,
                    item=f"t{i:020x}",
                    phase="test",
                )
                for i in range(8)
            )
        )
    assert [r.content for r in replies] == [f"echo {i}" for i in range(8)]
    assert sorted(row["item"] for row in ledger_calls(tmp_path)) == sorted(
        f"t{i:020x}" for i in range(8)
    )


@pytest.mark.asyncio
async def test_request_encoding_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    from evals.skill_bench.agent import actor as actor_module

    loop_thread = threading.get_ident()
    threads: list[int] = []
    real = actor_module._encode

    def spy(body):
        threads.append(threading.get_ident())
        return real(body)

    monkeypatch.setattr(actor_module, "_encode", spy)
    fake = FakeUpstream()
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        reply = await actor.complete(MESSAGES, index=0, item=ITEM, phase="test")
    assert threads and all(t != loop_thread for t in threads)
    assert (
        reply.call.request_sha256
        == hashlib.sha256(request_bytes(reply.call.request)).hexdigest()
    )


@pytest.mark.asyncio
async def test_the_recorded_client_serves_other_chat_roles(tmp_path):
    """J and the baselines' learner calls reuse the actor's recorded client."""
    from evals.skill_bench.agent.actor import Actor
    from evals.skill_bench.tests.fakes import running_gateway

    with pytest.raises(ValueError, match="role"):
        Actor(
            gateway_base_url="http://gateway", run_id="r", model=LUNA, role="embedding"
        )
    fake = FakeUpstream()
    fake.queue("chat/completions", Reply(503))
    async with running_gateway(tmp_path, fake, max_retries=0) as (gateway, _client):
        judge = Actor(
            gateway_base_url="http://gateway",
            run_id="cell-test-a1",
            model=LUNA,
            role="judge",
            transport=httpx.ASGITransport(app=gateway.app),
        )
        async with judge:
            with pytest.raises(ActorCallError) as info:
                await judge.complete(MESSAGES, index=0, item=ITEM, phase="learning")
            reply = await judge.complete(MESSAGES, index=1, item=ITEM, phase="learning")
    assert judge.url.endswith("/runs/cell-test-a1/judge/v1/chat/completions")
    assert info.value.failure.role == "judge" and info.value.call.role == "judge"
    assert reply.call.role == "judge"
    assert [row["role"] for row in ledger_calls(tmp_path)] == ["judge", "judge"]
