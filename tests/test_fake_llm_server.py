"""Tests for the offline OpenAI-compatible fake model server.

The fake is shared infrastructure for every offline MIRIX and eval-framework
test, so its wire format is checked with the real OpenAI SDK:
if the SDK can parse what the fake serves, MIRIX's OpenAI-compatible clients
can too.
"""

import asyncio
import base64
import json
import math
import struct
import time

import httpx
import openai
import pytest
from fakes.fake_llm_server import (
    DEFAULT_EMBEDDING_DIM,
    ROUTE_CHAT,
    ROUTE_EMBEDDINGS,
    ChatResponse,
    FakeLLMServer,
    Fault,
    cosine,
    text_hash_embedding,
    token_hash_embedding,
)


def _sdk(server: FakeLLMServer, **kwargs) -> openai.AsyncOpenAI:
    kwargs.setdefault("max_retries", 0)
    return openai.AsyncOpenAI(
        base_url=server.base_url,
        api_key="fake-key",
        http_client=httpx.AsyncClient(trust_env=False),
        **kwargs,
    )


def _http(timeout: float = 5.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(trust_env=False, timeout=timeout)


_MESSAGES = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# lifecycle                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_context_manager_binds_a_free_port_and_stops():
    with FakeLLMServer() as server:
        assert server.is_running
        assert server.port > 0
        assert server.base_url == f"http://127.0.0.1:{server.port}/v1"
        async with _http() as client:
            resp = await client.get(f"{server.base_url}/models")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "fake-model"
    assert not server.is_running


@pytest.mark.asyncio
async def test_two_servers_run_side_by_side():
    with FakeLLMServer() as a, FakeLLMServer() as b:
        assert a.port != b.port
        a.queue_text("from a")
        b.queue_text("from b")
        ra = await _sdk(a).chat.completions.create(model="m", messages=_MESSAGES)
        rb = await _sdk(b).chat.completions.create(model="m", messages=_MESSAGES)
    assert ra.choices[0].message.content == "from a"
    assert rb.choices[0].message.content == "from b"


@pytest.mark.asyncio
async def test_pytest_fixture_yields_running_server(fake_llm_server):
    assert fake_llm_server.is_running
    fake_llm_server.queue_text("fixture ok")
    resp = await _sdk(fake_llm_server).chat.completions.create(
        model="m", messages=_MESSAGES
    )
    assert resp.choices[0].message.content == "fixture ok"


@pytest.mark.asyncio
async def test_stop_does_not_hang_on_a_delayed_request():
    server = FakeLLMServer().start()
    server.inject_fault(Fault.slow(30))

    async def slow_call():
        async with _http(timeout=10) as client:
            try:
                await client.post(
                    f"{server.base_url}/chat/completions",
                    json={"model": "m", "messages": _MESSAGES},
                )
            except httpx.HTTPError:
                pass

    task = asyncio.create_task(slow_call())
    await asyncio.sleep(0.2)
    started = time.monotonic()
    await asyncio.to_thread(server.stop)
    assert time.monotonic() - started < 3
    await asyncio.wait_for(task, timeout=5)


# --------------------------------------------------------------------------- #
# chat completions                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scripted_responses_are_served_fifo(fake_llm_server):
    fake_llm_server.queue_text("first").queue_text("second")
    client = _sdk(fake_llm_server)
    r1 = await client.chat.completions.create(model="m", messages=_MESSAGES)
    r2 = await client.chat.completions.create(model="m", messages=_MESSAGES)
    assert [r1.choices[0].message.content, r2.choices[0].message.content] == [
        "first",
        "second",
    ]
    assert r1.choices[0].finish_reason == "stop"
    assert fake_llm_server.pending_responses == 0


@pytest.mark.asyncio
async def test_plain_string_and_raw_payload_are_accepted(fake_llm_server):
    raw = {
        "id": "chatcmpl-raw",
        "object": "chat.completion",
        "created": 0,
        "model": "raw-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "raw"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    fake_llm_server.queue_response("plain", raw)
    client = _sdk(fake_llm_server)
    r1 = await client.chat.completions.create(model="m", messages=_MESSAGES)
    r2 = await client.chat.completions.create(model="m", messages=_MESSAGES)
    assert r1.choices[0].message.content == "plain"
    assert r2.id == "chatcmpl-raw"
    assert r2.model == "raw-model"


@pytest.mark.asyncio
async def test_tool_calls_are_parsed_by_the_openai_sdk(fake_llm_server):
    fake_llm_server.queue_tool_calls(
        [
            {
                "name": "skill_create",
                "arguments": {"name": "wash-mug", "triggers": ["mug"]},
            },
            {"name": "finish_memory_update", "arguments": "{}", "id": "call_custom"},
        ]
    )
    resp = await _sdk(fake_llm_server).chat.completions.create(
        model="m",
        messages=_MESSAGES,
        tools=[
            {
                "type": "function",
                "function": {"name": "skill_create", "parameters": {"type": "object"}},
            }
        ],
    )
    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    calls = choice.message.tool_calls
    assert [c.function.name for c in calls] == ["skill_create", "finish_memory_update"]
    assert json.loads(calls[0].function.arguments) == {
        "name": "wash-mug",
        "triggers": ["mug"],
    }
    assert calls[1].id == "call_custom"
    assert calls[0].id and calls[0].type == "function"


@pytest.mark.asyncio
async def test_finish_reason_length_and_usage_override(fake_llm_server):
    fake_llm_server.queue_text(
        "truncated outp",
        finish_reason="length",
        usage={"prompt_tokens": 1000, "completion_tokens": 4096},
    )
    resp = await _sdk(fake_llm_server).chat.completions.create(
        model="m", messages=_MESSAGES
    )
    assert resp.choices[0].finish_reason == "length"
    assert resp.usage.prompt_tokens == 1000
    assert resp.usage.completion_tokens == 4096
    assert resp.usage.total_tokens == 5096


@pytest.mark.asyncio
async def test_usage_is_estimated_when_not_scripted(fake_llm_server):
    fake_llm_server.queue_text("x" * 400)
    resp = await _sdk(fake_llm_server).chat.completions.create(
        model="m", messages=_MESSAGES
    )
    assert resp.usage.completion_tokens == 100
    assert resp.usage.prompt_tokens >= 1
    assert (
        resp.usage.total_tokens
        == resp.usage.prompt_tokens + resp.usage.completion_tokens
    )


@pytest.mark.asyncio
async def test_tool_call_with_content_and_message_extra(fake_llm_server):
    fake_llm_server.queue_response(
        ChatResponse(
            content="thinking aloud",
            tool_calls=[{"name": "f", "arguments": {}}],
            finish_reason="stop",
            message_extra={"reasoning_content": "hidden"},
        )
    )
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
    message = resp.json()["choices"][0]["message"]
    assert message["content"] == "thinking aloud"
    assert message["reasoning_content"] == "hidden"
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_default_responder_sees_the_request(fake_llm_server):
    seen = []

    def responder(body):
        seen.append(body)
        return f"echo:{body['messages'][-1]['content']}"

    fake_llm_server.set_default_responder(responder)
    fake_llm_server.queue_text("scripted wins first")
    client = _sdk(fake_llm_server)
    r1 = await client.chat.completions.create(model="m", messages=_MESSAGES)
    r2 = await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "ping"}]
    )
    assert r1.choices[0].message.content == "scripted wins first"
    assert r2.choices[0].message.content == "echo:ping"
    assert len(seen) == 1
    assert [r.outcome for r in fake_llm_server.chat_requests()] == [
        "scripted",
        "default",
    ]


@pytest.mark.asyncio
async def test_default_responder_may_return_a_fault(fake_llm_server):
    fake_llm_server.set_default_responder(lambda body: Fault.http_error(502))
    with pytest.raises(openai.InternalServerError) as exc:
        await _sdk(fake_llm_server).chat.completions.create(
            model="m", messages=_MESSAGES
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_raising_default_responder_returns_500_not_a_hang(fake_llm_server):
    def boom(body):
        raise RuntimeError("responder bug")

    fake_llm_server.set_default_responder(boom)
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
    assert resp.status_code == 500
    assert "responder bug" in resp.json()["error"]["message"]


@pytest.mark.asyncio
async def test_unscripted_request_fails_loudly_and_is_counted(fake_llm_server):
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "fake_llm_unscripted"
    assert fake_llm_server.unscripted_requests == 1


@pytest.mark.asyncio
async def test_stream_and_malformed_requests_are_rejected(fake_llm_server):
    async with _http() as client:
        stream = await client.post(
            f"{fake_llm_server.base_url}/chat/completions",
            json={"messages": _MESSAGES, "stream": True},
        )
        no_messages = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"model": "m"}
        )
        bad_json = await client.post(
            f"{fake_llm_server.base_url}/chat/completions",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        unknown = await client.post(f"{fake_llm_server.base_url}/completions", json={})
    assert stream.status_code == 400
    assert no_messages.status_code == 400
    assert bad_json.status_code == 400
    assert unknown.status_code == 404
    # A rejected request must not consume a scripted response.
    assert fake_llm_server.pending_responses == 0


@pytest.mark.asyncio
async def test_routes_work_without_the_v1_prefix(fake_llm_server):
    fake_llm_server.queue_text("no prefix")
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.url}/chat/completions", json={"messages": _MESSAGES}
        )
        emb = await client.post(
            f"{fake_llm_server.url}/embeddings", json={"input": "x"}
        )
    assert resp.json()["choices"][0]["message"]["content"] == "no prefix"
    assert len(emb.json()["data"][0]["embedding"]) == DEFAULT_EMBEDDING_DIM


@pytest.mark.asyncio
async def test_concurrent_requests_each_get_one_scripted_response(fake_llm_server):
    for i in range(10):
        fake_llm_server.queue_text(f"r{i}")
    client = _sdk(fake_llm_server)
    results = await asyncio.gather(
        *[
            client.chat.completions.create(model="m", messages=_MESSAGES)
            for _ in range(10)
        ]
    )
    assert sorted(r.choices[0].message.content for r in results) == sorted(
        f"r{i}" for i in range(10)
    )


# --------------------------------------------------------------------------- #
# request log                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_request_log_records_method_path_and_json(fake_llm_server):
    fake_llm_server.queue_text("ok")
    await _sdk(fake_llm_server).chat.completions.create(
        model="gpt-x", messages=_MESSAGES, max_completion_tokens=321, temperature=0.2
    )
    request = fake_llm_server.last_chat_request()
    assert request.method == "POST"
    assert request.path == "/v1/chat/completions"
    assert request.route == ROUTE_CHAT
    assert request.json["model"] == "gpt-x"
    assert request.json["max_completion_tokens"] == 321
    assert request.json["messages"] == _MESSAGES
    assert request.headers["authorization"] == "Bearer fake-key"
    assert request.outcome == "scripted"


@pytest.mark.asyncio
async def test_reset_clears_script_faults_and_log(fake_llm_server):
    fake_llm_server.queue_text("left over")
    fake_llm_server.inject_fault(Fault.http_error(500))
    async with _http() as client:
        await client.post(f"{fake_llm_server.base_url}/embeddings", json={"input": "x"})
    fake_llm_server.reset()
    assert fake_llm_server.pending_responses == 0
    assert fake_llm_server.pending_faults == 0
    assert fake_llm_server.requests == []
    with pytest.raises(AssertionError):
        fake_llm_server.last_chat_request()


# --------------------------------------------------------------------------- #
# embeddings                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_embeddings_are_deterministic_unit_vectors_of_default_dim(
    fake_llm_server,
):
    async with _http() as client:
        first = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": "wash the mug"}
        )
        again = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": "wash the mug"}
        )
    v1 = first.json()["data"][0]["embedding"]
    v2 = again.json()["data"][0]["embedding"]
    assert len(v1) == DEFAULT_EMBEDDING_DIM == 3072
    assert v1 == v2
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-9)
    assert v1 == fake_llm_server.embed("wash the mug")
    # Deterministic across server instances (hash-seeded, no process state).
    with FakeLLMServer() as other:
        assert other.embed("wash the mug") == v1
    assert first.json()["usage"]["prompt_tokens"] >= 1


@pytest.mark.asyncio
async def test_embeddings_list_input_dimensions_and_indices(fake_llm_server):
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/embeddings",
            json={
                "input": ["alpha", "beta", [1, 2, 3]],
                "dimensions": 8,
                "model": "emb-x",
            },
        )
    body = resp.json()
    assert body["model"] == "emb-x"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(len(d["embedding"]) == 8 for d in body["data"])
    assert body["data"][0]["embedding"] != body["data"][1]["embedding"]


@pytest.mark.asyncio
async def test_embeddings_token_id_input_is_a_single_input(fake_llm_server):
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/embeddings",
            json={"input": [5, 6, 7], "dimensions": 4},
        )
    assert len(resp.json()["data"]) == 1


@pytest.mark.asyncio
async def test_embeddings_via_openai_sdk_base64_default(fake_llm_server):
    resp = await _sdk(fake_llm_server).embeddings.create(
        model="emb", input=["one", "two"]
    )
    assert [len(d.embedding) for d in resp.data] == [DEFAULT_EMBEDDING_DIM] * 2
    # The SDK asked for base64; float32 round-trip matches the float vector closely.
    request = fake_llm_server.embedding_requests()[-1]
    assert request.json["encoding_format"] == "base64"
    expected = fake_llm_server.embed("one")
    assert max(abs(a - b) for a, b in zip(resp.data[0].embedding, expected)) < 1e-6


@pytest.mark.asyncio
async def test_embeddings_base64_encoding_is_little_endian_float32(fake_llm_server):
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/embeddings",
            json={"input": "abc", "encoding_format": "base64", "dimensions": 4},
        )
    raw = base64.b64decode(resp.json()["data"][0]["embedding"])
    decoded = struct.unpack("<4f", raw)
    assert all(
        abs(a - b) < 1e-6 for a, b in zip(decoded, fake_llm_server.embed("abc", 4))
    )


@pytest.mark.asyncio
async def test_embeddings_reject_missing_input(fake_llm_server):
    async with _http() as client:
        missing = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"model": "m"}
        )
        empty = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": []}
        )
    assert missing.status_code == 400
    assert empty.status_code == 400


def test_token_hash_mode_gives_similar_vectors_to_overlapping_text():
    a = token_hash_embedding("clean the mug in the sink", 256)
    b = token_hash_embedding("mug sink clean", 256)
    c = token_hash_embedding("quarterly revenue table", 256)
    assert cosine(a, b) > 0.5
    assert abs(cosine(a, c)) < 0.3
    # text_hash mode is order/format sensitive by design.
    assert text_hash_embedding("a b", 16) != text_hash_embedding("b a", 16)


@pytest.mark.asyncio
async def test_server_embedding_mode_and_custom_fn():
    with FakeLLMServer(embedding_mode="token_hash", embedding_dim=32) as server:
        assert server.embed("x y") == token_hash_embedding("x y", 32)
    with FakeLLMServer(
        embedding_fn=lambda text, dim: [float(len(text))] + [0.0] * (dim - 1)
    ) as server:
        async with _http() as client:
            resp = await client.post(
                f"{server.base_url}/embeddings", json={"input": "four", "dimensions": 3}
            )
        assert resp.json()["data"][0]["embedding"] == [4.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        FakeLLMServer(embedding_mode="nope")


# --------------------------------------------------------------------------- #
# fault injection                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_5xx_fault_does_not_consume_the_scripted_response(fake_llm_server):
    fake_llm_server.queue_text("after the outage")
    fake_llm_server.inject_fault(Fault.http_error(503), times=2)
    client = _sdk(fake_llm_server)
    for _ in range(2):
        with pytest.raises(openai.InternalServerError) as exc:
            await client.chat.completions.create(model="m", messages=_MESSAGES)
        assert exc.value.status_code == 503
    resp = await client.chat.completions.create(model="m", messages=_MESSAGES)
    assert resp.choices[0].message.content == "after the outage"
    assert [r.outcome for r in fake_llm_server.chat_requests()] == [
        "fault:503",
        "fault:503",
        "scripted",
    ]


@pytest.mark.asyncio
async def test_sdk_transport_retry_recovers_from_a_5xx(fake_llm_server):
    fake_llm_server.queue_text("recovered")
    fake_llm_server.inject_fault(Fault.http_error(500))
    resp = await _sdk(fake_llm_server, max_retries=1).chat.completions.create(
        model="m", messages=_MESSAGES
    )
    assert resp.choices[0].message.content == "recovered"
    assert len(fake_llm_server.chat_requests()) == 2


@pytest.mark.asyncio
async def test_rate_limit_fault_sends_retry_after(fake_llm_server):
    fake_llm_server.inject_fault(Fault.rate_limit(retry_after=7))
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    assert resp.json()["error"]["type"] == "rate_limit_exceeded"
    fake_llm_server.inject_fault(Fault.rate_limit(retry_after=0))
    with pytest.raises(openai.RateLimitError):
        await _sdk(fake_llm_server).chat.completions.create(
            model="m", messages=_MESSAGES
        )


@pytest.mark.asyncio
async def test_slow_fault_trips_the_client_timeout_then_recovers(fake_llm_server):
    fake_llm_server.inject_fault(Fault.slow(0.8))
    fake_llm_server.queue_text("served to nobody")
    client = _sdk(fake_llm_server, timeout=0.2)
    with pytest.raises(openai.APITimeoutError):
        await client.chat.completions.create(model="m", messages=_MESSAGES)
    # The abandoned slow request still consumes the next scripted response once
    # its delay elapses (Fault.slow is "slow but successful" server-side).
    deadline = time.monotonic() + 5
    while fake_llm_server.pending_responses and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert fake_llm_server.pending_responses == 0
    fake_llm_server.queue_text("retry")
    resp = await _sdk(fake_llm_server).chat.completions.create(
        model="m", messages=_MESSAGES
    )
    assert resp.choices[0].message.content == "retry"


@pytest.mark.asyncio
async def test_timeout_fault_returns_an_error_after_the_delay(fake_llm_server):
    fake_llm_server.inject_fault(Fault.timeout(0.2, status=504))
    started = time.monotonic()
    async with _http() as client:
        resp = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
    assert resp.status_code == 504
    assert time.monotonic() - started >= 0.2


@pytest.mark.asyncio
async def test_drop_connection_fault_is_a_network_error(fake_llm_server):
    fake_llm_server.inject_fault(Fault.drop_connection())
    async with _http() as client:
        with pytest.raises(httpx.TransportError):
            await client.post(
                f"{fake_llm_server.base_url}/chat/completions",
                json={"messages": _MESSAGES},
            )
    assert fake_llm_server.chat_requests()[-1].outcome == "fault:disconnect"


@pytest.mark.asyncio
async def test_faults_are_scoped_to_their_route(fake_llm_server):
    fake_llm_server.inject_fault(Fault.http_error(500), route=ROUTE_EMBEDDINGS)
    fake_llm_server.queue_text("chat unaffected")
    async with _http() as client:
        chat = await client.post(
            f"{fake_llm_server.base_url}/chat/completions", json={"messages": _MESSAGES}
        )
        emb_fail = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": "x"}
        )
        emb_ok = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": "x"}
        )
    assert chat.status_code == 200
    assert emb_fail.status_code == 500
    assert emb_ok.status_code == 200


@pytest.mark.asyncio
async def test_route_none_fault_hits_the_next_request_of_any_route(fake_llm_server):
    fake_llm_server.inject_fault(Fault.http_error(502), route=None)
    async with _http() as client:
        emb = await client.post(
            f"{fake_llm_server.base_url}/embeddings", json={"input": "x"}
        )
    assert emb.status_code == 502
    assert fake_llm_server.pending_faults == 0


def test_inject_fault_validates_arguments():
    server = FakeLLMServer()
    with pytest.raises(ValueError):
        server.inject_fault(Fault(), route="bogus")
    with pytest.raises(ValueError):
        server.inject_fault(Fault(), times=0)
    with pytest.raises(RuntimeError):
        _ = server.port
