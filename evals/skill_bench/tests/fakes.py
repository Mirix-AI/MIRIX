"""Offline fakes for skill_bench tests: never contacts a real provider.

* :class:`FakeUpstream` - a scriptable OpenRouter-compatible upstream
  (``/chat/completions``, ``/embeddings``, ``/key``). Use ``httpx.ASGITransport(app=fake.app)``
  in-process, or ``with fake.serve() as url`` for a real loopback socket.
* :class:`FlakyTransport` - raises scripted transport errors before delegating.
* :func:`write_stub_server` - a tiny HTTP server standing in for MIRIX in process tests.
* :func:`install_mark_route` - ``POST /runs/{run_id}/mark`` (learning checkpoint and
  resume markers) on a gateway of a checkout that does not have it yet;
  :func:`running_gateway` and :func:`serve_gateway` install it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import textwrap
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from evals.skill_bench import protocol
from evals.skill_bench.infra import gateway as gateway_module
from evals.skill_bench.infra.gateway import LOCAL_TOKEN, Gateway, GatewayConfig
from evals.skill_bench.infra.processes import find_free_port

FAKE_UPSTREAM_KEY = "sk-or-test-not-a-real-key"


@dataclass
class Reply:
    """One scripted upstream reply. ``json`` None means the endpoint's default success body."""

    status: int = 200
    json: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    delay_s: float = 0.0
    raw: bytes | None = None
    #: The reply waits until the test sets this event (a call still in flight upstream
    #: while the harness that sent it is killed); None: no wait.
    hold: threading.Event | None = None


@dataclass
class Recorded:
    path: str
    headers: dict[str, str]
    json: Any
    at: float


def chat_body(
    model: str = "openai/gpt-6-luna",
    content: str | None = "ok",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
    usage: Mapping[str, Any] | None = None,
    provider: str = "FakeProvider",
    reasoning: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "gen-fake-1",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "native_finish_reason": finish_reason,
            }
        ],
        "usage": dict(
            usage
            or {
                "prompt_tokens": 42,
                "completion_tokens": 17,
                "total_tokens": 59,
                "cost": 0.000123,
                "completion_tokens_details": {"reasoning_tokens": 9},
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        ),
    }


def embedding_body(
    inputs: list[str], model: str = "google/gemini-embedding-001"
) -> dict[str, Any]:
    data = []
    for i, text in enumerate(inputs):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": [b / 255.0 for b in digest[:8]],
            }
        )
    return {
        "object": "list",
        "data": data,
        "model": model,
        "provider": "Google",
        "usage": {
            "prompt_tokens": 3 * len(inputs),
            "total_tokens": 3 * len(inputs),
            "cost": 0.000001,
        },
    }


#: ``responder(endpoint, body)`` -> a reply for a request no queued reply covers (None = default).
Responder = Callable[[str, dict[str, Any]], "Reply | None"]


class FakeUpstream:
    """Scriptable upstream. Queue replies per endpoint; unqueued calls go to ``responder``
    (if set) and otherwise get default successes."""

    def __init__(
        self,
        *,
        limit_remaining: float | None = 100.0,
        credits: tuple[float, float] | None = None,
        responder: Responder | None = None,
    ) -> None:
        self.requests: list[Recorded] = []
        self.responder = responder
        self.replies: dict[str, deque[Reply]] = {
            "chat/completions": deque(),
            "embeddings": deque(),
        }
        self.limit_remaining = limit_remaining
        self.credits = credits  # (total_credits, total_usage); None -> /credits fails
        #: The key's all-time ``usage`` /key reports (the spend budget counts on it).
        self.usage: float | None = 1.5
        self.key_calls = 0
        self.app = self._build_app()

    def queue(self, endpoint: str, *replies: Reply) -> None:
        self.replies[endpoint].extend(replies)

    def calls(self, endpoint: str) -> list[Recorded]:
        return [r for r in self.requests if r.path == endpoint]

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/credits")
        async def credits() -> Response:
            if self.credits is None:
                return JSONResponse(
                    {"error": {"code": 403, "message": "forbidden"}}, 403
                )
            total, used = self.credits
            return JSONResponse({"data": {"total_credits": total, "total_usage": used}})

        @app.get("/key")
        async def key() -> dict[str, Any]:
            self.key_calls += 1
            return {
                "data": {
                    "label": "fake",
                    "limit": 500,
                    "limit_remaining": self.limit_remaining,
                    "usage": self.usage,
                }
            }

        @app.post("/{endpoint:path}")
        async def post(endpoint: str, request: Request) -> Response:
            body = await request.json()
            headers = {k.lower(): v for k, v in request.headers.items()}
            self.requests.append(Recorded(endpoint, headers, body, time.monotonic()))
            queue = self.replies.get(endpoint)
            if queue is None:
                return JSONResponse({"error": {"message": "not found"}}, 404)
            if queue:
                reply = queue.popleft()
            else:
                scripted = self.responder(endpoint, body) if self.responder else None
                reply = scripted if scripted is not None else Reply()
            if reply.delay_s:
                await asyncio.sleep(reply.delay_s)
            while reply.hold is not None and not reply.hold.is_set():
                await asyncio.sleep(0.01)
            if reply.raw is not None:
                return Response(
                    reply.raw,
                    reply.status,
                    headers=dict(reply.headers),
                    media_type="application/json",
                )
            payload = reply.json
            if payload is None:
                if endpoint == "chat/completions":
                    payload = chat_body(model=body.get("model", "openai/gpt-6-luna"))
                else:
                    raw_input = body.get("input")
                    payload = embedding_body(
                        raw_input if isinstance(raw_input, list) else [str(raw_input)]
                    )
            return JSONResponse(payload, reply.status, headers=dict(reply.headers))

        return app

    def asgi_transport(self) -> httpx.ASGITransport:
        return httpx.ASGITransport(app=self.app)

    @contextmanager
    def serve(self) -> Iterator[str]:
        """Serve on a real loopback port in a background thread; yield the base URL."""
        import uvicorn

        port = find_free_port()
        server = uvicorn.Server(
            uvicorn.Config(
                self.app,
                host="127.0.0.1",
                port=port,
                log_level="error",
                timeout_graceful_shutdown=1,
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("fake upstream did not start")
            time.sleep(0.02)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


class FlakyTransport(httpx.AsyncBaseTransport):
    """Raise the scripted exceptions (None = pass through) for successive POSTs, then delegate."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        failures: list[type[httpx.TransportError] | None],
    ) -> None:
        self.inner = inner
        self.failures = deque(failures)
        self.raised: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and self.failures:
            failure = self.failures.popleft()
            if failure is not None:
                self.raised.append(failure.__name__)
                raise failure("injected failure", request=request)
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


STUB_SERVER = textwrap.dedent(
    '''
    """Stand-in for scripts/start_server.py: /health, /env, and a write under ~/.mirix like MIRIX."""
    import argparse, json, os
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    mirix_dir = os.path.join(os.path.expanduser("~"), ".mirix")
    os.makedirs(mirix_dir, exist_ok=True)
    with open(os.path.join(mirix_dir, "stub_marker"), "w") as fh:
        fh.write(str(os.getpid()))
    print("stub mirix server starting on", args.port, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = {"status": "healthy", "service": "mirix-api"}
            elif self.path == "/env":
                body = dict(os.environ)
            else:
                self.send_response(404)
                self.end_headers()
                return
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    HTTPServer((args.host, args.port), Handler).serve_forever()
    '''
)


def write_stub_server(directory: Path) -> Path:
    path = directory / "stub_mirix_server.py"
    path.write_text(STUB_SERVER, encoding="utf-8")
    return path


#: The events ``POST /runs/{run_id}/mark`` takes, and the size bound of its body.
MARK_EVENTS = ("checkpoint", "resume")
MARK_MAX_BYTES = 4096


def install_mark_route(gateway: Gateway) -> bool:
    """Give ``gateway`` the ``POST /runs/{run_id}/mark`` route (learning checkpoint and
    resume markers) when its checkout does not have it yet (a test shim; a gateway that
    has the route keeps its own). The marker line goes through ``Gateway.ledger``
    (fsync'd, in the ledger's own write order, with ``gateway_start``); the answer
    carries the byte offset where it starts. 400 for another event or an oversized
    body, 410 for an abandoned run. Returns whether the shim was installed."""
    app = gateway.app
    if any(getattr(r, "path", None) == "/runs/{run_id}/mark" for r in app.routes):
        return False

    @app.post("/runs/{run_id}/mark")
    async def mark(run_id: str, request: Request) -> Response:
        token = gateway.config.local_token
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return JSONResponse(
                {"error": {"code": 401, "message": "unauthorized"}}, 401
            )
        if not gateway_module.RUN_ID_RE.fullmatch(run_id):
            return JSONResponse(
                {"error": {"code": 400, "message": "invalid run_id"}}, 400
            )
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if (
            not isinstance(body, dict)
            or body.get("event") not in MARK_EVENTS
            or len(raw) > MARK_MAX_BYTES
        ):
            return JSONResponse(
                {
                    "error": {
                        "code": 400,
                        "message": f"event must be one of {MARK_EVENTS}",
                    }
                },
                400,
            )
        if (await gateway._state(run_id)).abandoned:
            return JSONResponse(
                {"error": {"code": 410, "message": f"run {run_id} was abandoned"}}, 410
            )
        record = {**body, "ts": gateway_module._now(), "run_id": run_id}
        await gateway.ledger(run_id, record)
        line = json.dumps(
            {**record, "gateway_start": gateway.start_id},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        path = gateway.config.runs_dir / run_id / "gateway.jsonl"
        data = await asyncio.to_thread(path.read_bytes)
        return JSONResponse(
            {"ok": True, "offset": data.rfind(line), "ts": record["ts"]}
        )

    return True


@asynccontextmanager
async def running_gateway(
    tmp_path: Path,
    fake: FakeUpstream,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    **overrides: Any,
) -> AsyncIterator[tuple[Gateway, httpx.AsyncClient]]:
    config = GatewayConfig(
        api_key=FAKE_UPSTREAM_KEY,
        models=protocol.load_models(),
        runs_dir=tmp_path / "runs",
        upstream="http://upstream",
        trust_env=False,
        transport=transport or fake.asgi_transport(),
        backoff_base_s=overrides.pop("backoff_base_s", 0.01),
        # A paused gateway answers 402 at once unless a test asks it to hold calls.
        pause_hold_s=overrides.pop("pause_hold_s", 0.0),
        **overrides,
    )
    gateway = Gateway(config)
    install_mark_route(gateway)
    await gateway.start()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway.app),
        base_url="http://gateway",
        headers={"Authorization": f"Bearer {LOCAL_TOKEN}"},
        timeout=60,
    )
    try:
        yield gateway, client
    finally:
        await client.aclose()
        await gateway.aclose()


@contextmanager
def serve_gateway(
    tmp_path: Path,
    fake: FakeUpstream,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    **overrides: Any,
) -> Iterator[tuple[Gateway, str]]:
    """A real gateway over ``fake`` on a loopback port (background thread); yields it and its base URL.

    Methods reach the gateway by URL (``CellContext.gateway_base_url``), as in a real
    cell; :func:`running_gateway` is the in-process variant for code that takes a
    transport. The gateway's event loop runs in the server thread.
    """
    import uvicorn

    config = GatewayConfig(
        api_key=FAKE_UPSTREAM_KEY,
        models=protocol.load_models(),
        runs_dir=tmp_path / "runs",
        upstream="http://upstream",
        trust_env=False,
        transport=transport or fake.asgi_transport(),
        backoff_base_s=overrides.pop("backoff_base_s", 0.01),
        pause_hold_s=overrides.pop("pause_hold_s", 0.0),
        **overrides,
    )
    gateway = Gateway(config)
    install_mark_route(gateway)
    port = find_free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            gateway.app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            lifespan="on",
            timeout_graceful_shutdown=1,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("gateway did not start")
        time.sleep(0.02)
    try:
        yield gateway, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
