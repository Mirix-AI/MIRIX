"""An OpenAI-compatible fake model server for offline tests.

Every test in this repository must run without paid or real model-provider
calls. ``FakeLLMServer`` stands in for an OpenAI-compatible endpoint: it binds a
free localhost port, serves requests from a background thread, and lets a test
script exactly what each call returns. Point any OpenAI-compatible client
(``openai.AsyncOpenAI``, MIRIX's ``LLMConfig.model_endpoint`` /
``EmbeddingConfig.embedding_endpoint``, raw ``httpx``) at :attr:`base_url`.

Endpoints (both with and without the ``/v1`` prefix):

* ``POST /v1/chat/completions`` — answers from, in order: an injected fault for
  this route, the scripted FIFO of responses, the optional default responder.
  With none of those it returns HTTP 500 (``type: fake_llm_unscripted``) and
  counts the miss in :attr:`FakeLLMServer.unscripted_requests`, so a test that
  forgot to script a call fails loudly instead of hanging. ``stream=true`` is
  rejected with HTTP 400 (not supported).
* ``POST /v1/embeddings`` — deterministic vectors for every input (string, list
  of strings, or token-id lists). Default: a unit vector seeded by the SHA-256
  of the text (``embedding_mode="text_hash"``), 3072 dimensions like
  gemini-embedding-001. ``embedding_mode="token_hash"`` sums per-token hash
  vectors instead, so texts that share words get a positive cosine — useful for
  similarity-gate tests. Honors ``dimensions`` and ``encoding_format=base64``
  (the OpenAI Python SDK requests base64 by default).
* ``GET /v1/models`` — a one-model listing.

Fault injection (:meth:`FakeLLMServer.inject_fault`) applies to the next
request(s) on a route and does NOT consume a scripted response, so the retry
that follows receives it: HTTP 5xx, 429 with ``Retry-After``, an artificial
delay (to trip client timeouts), or a dropped connection.

Every request is logged (:attr:`FakeLLMServer.requests`) with its method, path,
JSON body, headers and outcome, for assertions such as "the client sent
``max_completion_tokens=…``".

The module is stdlib-only (pytest is optional) so the eval framework can import
it too. Typical use::

    def test_something(fake_llm_server):
        fake_llm_server.queue_text("hello")
        fake_llm_server.queue_tool_call("finish", {"ok": True})
        fake_llm_server.inject_fault(Fault.rate_limit(retry_after=0), route="chat")
        client = AsyncOpenAI(base_url=fake_llm_server.base_url, api_key="fake")
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import re
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

DEFAULT_EMBEDDING_DIM = 3072
DEFAULT_MODEL_NAME = "fake-model"

ROUTE_CHAT = "chat"
ROUTE_EMBEDDINGS = "embeddings"
ROUTE_MODELS = "models"
_ROUTES_BY_SUFFIX = {
    "/chat/completions": ROUTE_CHAT,
    "/embeddings": ROUTE_EMBEDDINGS,
    "/models": ROUTE_MODELS,
}

EMBEDDING_MODE_TEXT_HASH = "text_hash"
EMBEDDING_MODE_TOKEN_HASH = "token_hash"
_EMBEDDING_MODES = (EMBEDDING_MODE_TEXT_HASH, EMBEDDING_MODE_TOKEN_HASH)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Scripted responses and faults                                               #
# --------------------------------------------------------------------------- #


@dataclass
class ChatResponse:
    """One scripted ``/chat/completions`` answer.

    ``tool_calls`` entries are ``{"name": str, "arguments": dict | str}`` with an
    optional ``"id"``; dict arguments are JSON-encoded as the OpenAI wire format
    requires. ``finish_reason`` defaults to ``"tool_calls"`` when tool calls are
    present and ``"stop"`` otherwise; set ``"length"`` to simulate truncation.
    ``usage`` overrides the estimated token counts (missing keys are filled in).
    ``message_extra`` is merged into the returned ``message`` object (for
    provider-specific fields such as ``reasoning_content``).
    """

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    message_extra: Dict[str, Any] = field(default_factory=dict)

    def to_payload(
        self, request_json: Dict[str, Any], completion_id: str
    ) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "assistant", "content": self.content}
        wire_tool_calls = None
        if self.tool_calls:
            wire_tool_calls = []
            for index, call in enumerate(self.tool_calls):
                arguments = call.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                wire_tool_calls.append(
                    {
                        "id": call.get("id") or f"call_{completion_id}_{index}",
                        "type": "function",
                        "function": {"name": call["name"], "arguments": arguments},
                    }
                )
            message["tool_calls"] = wire_tool_calls
        message.update(self.message_extra)

        finish_reason = self.finish_reason or (
            "tool_calls" if self.tool_calls else "stop"
        )
        completion_chars = len(self.content or "") + len(
            json.dumps(wire_tool_calls or [])
        )
        prompt_chars = len(json.dumps(request_json.get("messages", []))) + len(
            json.dumps(request_json.get("tools", []))
        )
        usage = {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, completion_chars // 4),
        }
        if self.usage:
            usage.update(self.usage)
        if "total_tokens" not in (self.usage or {}):
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        return {
            "id": f"chatcmpl-{completion_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model or request_json.get("model") or DEFAULT_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": usage,
        }


@dataclass
class Fault:
    """A failure served instead of the normal response.

    * ``status`` / ``body`` / ``headers`` — the HTTP error to return.
    * ``delay`` — seconds to wait before answering (cut short by ``stop()``).
    * ``passthrough`` — after the delay, serve the NORMAL response instead of an
      error (a slow-but-successful call, for client-timeout tests).
    * ``disconnect`` — close the socket without any response (network error).
    * ``trickle`` — answer 200 at once, then send the body one whitespace byte
      every ``trickle`` seconds for ``delay`` seconds before a small completion:
      every read gets data in time, so only a WALL-CLOCK deadline stops it.
    """

    status: int = 500
    body: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    passthrough: bool = False
    disconnect: bool = False
    trickle: float = 0.0

    @classmethod
    def http_error(cls, status: int = 500, message: Optional[str] = None) -> "Fault":
        return cls(
            status=status,
            body=_error_body(message or f"injected HTTP {status}", "fake_llm_injected"),
        )

    @classmethod
    def rate_limit(
        cls, retry_after: float = 1.0, message: str = "injected rate limit"
    ) -> "Fault":
        value = (
            str(int(retry_after))
            if float(retry_after).is_integer()
            else str(retry_after)
        )
        return cls(
            status=429,
            body=_error_body(message, "rate_limit_exceeded"),
            headers={"Retry-After": value},
        )

    @classmethod
    def slow(cls, seconds: float) -> "Fault":
        """Delay, then answer normally — trips any client timeout below ``seconds``.

        The delayed request still takes the next scripted response when its delay
        elapses, even if the client already gave up (like a real slow upstream).
        """
        return cls(delay=seconds, passthrough=True)

    @classmethod
    def timeout(cls, seconds: float, status: int = 504) -> "Fault":
        """Delay, then answer with an error (``status``) if the client is still waiting."""
        return cls(
            status=status,
            delay=seconds,
            body=_error_body("injected upstream timeout", "timeout"),
        )

    @classmethod
    def drop_connection(cls) -> "Fault":
        return cls(disconnect=True)

    @classmethod
    def trickling(cls, seconds: float, interval: float = 0.05) -> "Fault":
        """A 200 whose body trickles in (one byte per ``interval``) for
        ``seconds``: no read ever times out, the whole exchange takes long."""
        return cls(status=200, delay=seconds, trickle=interval)

    def label(self) -> str:
        if self.disconnect:
            return "fault:disconnect"
        if self.trickle:
            return f"fault:trickle:{self.delay}"
        if self.passthrough:
            return f"fault:slow:{self.delay}"
        return f"fault:{self.status}"


@dataclass(frozen=True)
class RecordedRequest:
    """One request the fake received, for test assertions."""

    method: str
    path: str
    route: Optional[str]
    json: Any
    headers: Dict[str, str]
    timestamp: float
    outcome: str


ScriptedChat = Union[ChatResponse, str, Dict[str, Any], Fault]
DefaultResponder = Callable[[Dict[str, Any]], ScriptedChat]
EmbeddingFn = Callable[[str, int], List[float]]


def _error_body(message: str, error_type: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "code": error_type}}


# --------------------------------------------------------------------------- #
# Deterministic embeddings                                                    #
# --------------------------------------------------------------------------- #


def _seeded_unit_vector(seed_text: str, dim: int) -> List[float]:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def text_hash_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
    """Unit vector seeded by the whole text: identical text -> identical vector."""
    return _seeded_unit_vector("text:" + text, dim)


def token_hash_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
    """Normalized sum of per-token seeded vectors: shared words -> positive cosine."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return text_hash_embedding(text, dim)
    total = [0.0] * dim
    for token in tokens:
        for i, v in enumerate(_seeded_unit_vector("token:" + token, dim)):
            total[i] += v
    norm = math.sqrt(sum(v * v for v in total)) or 1.0
    return [v / norm for v in total]


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity helper for tests."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# HTTP plumbing                                                               #
# --------------------------------------------------------------------------- #


class _FakeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # stop() must not join request threads that are sleeping in a delay fault.
    block_on_close = False
    allow_reuse_address = True

    fake: "FakeLLMServer"


class _Handler(BaseHTTPRequestHandler):
    server: _FakeHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return  # keep test output quiet

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self.server.fake._dispatch(self, "GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self.server.fake._dispatch(self, "POST")


def _route_for(path: str) -> Optional[str]:
    clean = path.split("?", 1)[0].rstrip("/")
    for suffix, route in _ROUTES_BY_SUFFIX.items():
        if clean == suffix or clean == "/v1" + suffix:
            return route
    return None


# --------------------------------------------------------------------------- #
# The fake server                                                             #
# --------------------------------------------------------------------------- #


class FakeLLMServer:
    """OpenAI-compatible fake served from a background thread on localhost."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        embedding_mode: str = EMBEDDING_MODE_TEXT_HASH,
        embedding_fn: Optional[EmbeddingFn] = None,
        default_responder: Optional[DefaultResponder] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        if embedding_mode not in _EMBEDDING_MODES:
            raise ValueError(
                f"embedding_mode must be one of {_EMBEDDING_MODES}, got {embedding_mode!r}"
            )
        self.host = host
        self._requested_port = port
        self.embedding_dim = embedding_dim
        self.embedding_mode = embedding_mode
        self.embedding_fn = embedding_fn
        self.default_responder = default_responder
        self.model_name = model_name

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._responses: Deque[ScriptedChat] = deque()
        # route (or None = any route) -> FIFO of faults
        self._faults: Deque[Tuple[Optional[str], Fault]] = deque()
        self._requests: List[RecordedRequest] = []
        self._completion_counter = 0
        self._unscripted = 0
        self._httpd: Optional[_FakeHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ----------------------------- lifecycle ------------------------------ #

    def start(self) -> "FakeLLMServer":
        if self._httpd is not None:
            return self
        self._stop_event.clear()
        httpd = _FakeHTTPServer((self.host, self._requested_port), _Handler)
        httpd.fake = self
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="FakeLLMServer",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        self._stop_event.set()  # wake any request sleeping in a delay fault
        httpd.shutdown()
        httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "FakeLLMServer":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._httpd is not None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("FakeLLMServer is not running")
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        """OpenAI-style base URL (``…/v1``) to hand to clients."""
        return f"{self.url}/v1"

    # ------------------------------ scripting ----------------------------- #

    def queue_response(self, *responses: ScriptedChat) -> "FakeLLMServer":
        """Append chat responses (ChatResponse, plain text, raw payload dict, or Fault)."""
        with self._lock:
            self._responses.extend(responses)
        return self

    def queue_text(
        self,
        text: str,
        *,
        finish_reason: str = "stop",
        usage: Optional[Dict[str, int]] = None,
    ) -> "FakeLLMServer":
        return self.queue_response(
            ChatResponse(content=text, finish_reason=finish_reason, usage=usage)
        )

    def queue_tool_call(
        self,
        name: str,
        arguments: Union[Dict[str, Any], str, None] = None,
        *,
        content: Optional[str] = None,
        finish_reason: str = "tool_calls",
        usage: Optional[Dict[str, int]] = None,
        call_id: Optional[str] = None,
    ) -> "FakeLLMServer":
        call: Dict[str, Any] = {
            "name": name,
            "arguments": arguments if arguments is not None else {},
        }
        if call_id:
            call["id"] = call_id
        return self.queue_tool_calls(
            [call], content=content, finish_reason=finish_reason, usage=usage
        )

    def queue_tool_calls(
        self,
        calls: List[Dict[str, Any]],
        *,
        content: Optional[str] = None,
        finish_reason: str = "tool_calls",
        usage: Optional[Dict[str, int]] = None,
    ) -> "FakeLLMServer":
        return self.queue_response(
            ChatResponse(
                content=content,
                tool_calls=list(calls),
                finish_reason=finish_reason,
                usage=usage,
            )
        )

    def inject_fault(
        self, fault: Fault, *, route: Optional[str] = ROUTE_CHAT, times: int = 1
    ) -> "FakeLLMServer":
        """Serve ``fault`` for the next ``times`` requests on ``route`` (None = any route)."""
        if route is not None and route not in (
            ROUTE_CHAT,
            ROUTE_EMBEDDINGS,
            ROUTE_MODELS,
        ):
            raise ValueError(f"unknown route {route!r}")
        if times < 1:
            raise ValueError("times must be >= 1")
        with self._lock:
            for _ in range(times):
                self._faults.append((route, fault))
        return self

    def set_default_responder(
        self, responder: Optional[DefaultResponder]
    ) -> "FakeLLMServer":
        self.default_responder = responder
        return self

    def reset(self) -> None:
        """Forget scripted responses, pending faults and the request log."""
        with self._lock:
            self._responses.clear()
            self._faults.clear()
            self._requests.clear()
            self._unscripted = 0

    # ----------------------------- inspection ----------------------------- #

    @property
    def requests(self) -> List[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    def chat_requests(self) -> List[RecordedRequest]:
        return [r for r in self.requests if r.route == ROUTE_CHAT]

    def embedding_requests(self) -> List[RecordedRequest]:
        return [r for r in self.requests if r.route == ROUTE_EMBEDDINGS]

    def last_chat_request(self) -> RecordedRequest:
        chats = self.chat_requests()
        if not chats:
            raise AssertionError("FakeLLMServer received no chat request")
        return chats[-1]

    @property
    def pending_responses(self) -> int:
        with self._lock:
            return len(self._responses)

    @property
    def pending_faults(self) -> int:
        with self._lock:
            return len(self._faults)

    @property
    def unscripted_requests(self) -> int:
        with self._lock:
            return self._unscripted

    def embed(self, text: str, dim: Optional[int] = None) -> List[float]:
        """The exact vector the server returns for ``text`` (for assertions)."""
        dim = dim or self.embedding_dim
        if self.embedding_fn is not None:
            return list(self.embedding_fn(text, dim))
        if self.embedding_mode == EMBEDDING_MODE_TOKEN_HASH:
            return token_hash_embedding(text, dim)
        return text_hash_embedding(text, dim)

    # ------------------------------ serving ------------------------------- #

    def _record(
        self,
        handler: _Handler,
        method: str,
        route: Optional[str],
        body: Any,
        outcome: str,
    ) -> None:
        with self._lock:
            self._requests.append(
                RecordedRequest(
                    method=method,
                    path=handler.path,
                    route=route,
                    json=body,
                    headers={k.lower(): v for k, v in handler.headers.items()},
                    timestamp=time.time(),
                    outcome=outcome,
                )
            )

    def _take_fault(self, route: Optional[str]) -> Optional[Fault]:
        with self._lock:
            for index, (fault_route, fault) in enumerate(self._faults):
                if fault_route is None or fault_route == route:
                    del self._faults[index]
                    return fault
        return None

    @staticmethod
    def _send_json(
        handler: _Handler,
        status: int,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        data = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(data)
        handler.wfile.flush()

    def _dispatch(self, handler: _Handler, method: str) -> None:
        route = _route_for(handler.path)
        body: Any = None
        raw = b""
        length = int(handler.headers.get("Content-Length") or 0)
        if length:
            raw = handler.rfile.read(length)
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._record(
                    handler, method, route, raw.decode("utf-8", "replace"), "bad_json"
                )
                self._send_json(
                    handler,
                    400,
                    _error_body(
                        "request body is not valid JSON", "invalid_request_error"
                    ),
                )
                return

        if route is None or (route == ROUTE_MODELS) != (method == "GET"):
            self._record(handler, method, route, body, "not_found")
            self._send_json(
                handler,
                404,
                _error_body(f"no route for {method} {handler.path}", "not_found"),
            )
            return

        fault = self._take_fault(route)
        if fault is not None:
            if self._serve_fault(handler, method, route, body, fault):
                return

        if route == ROUTE_MODELS:
            self._record(handler, method, route, body, "models")
            self._send_json(
                handler,
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": self.model_name, "object": "model", "owned_by": "fake"}
                    ],
                },
            )
        elif route == ROUTE_EMBEDDINGS:
            self._serve_embeddings(handler, method, body)
        else:
            self._serve_chat(handler, method, body)

    def _serve_fault(
        self, handler: _Handler, method: str, route: str, body: Any, fault: Fault
    ) -> bool:
        """Serve ``fault``. Returns False when the normal response should follow."""
        self._record(handler, method, route, body, fault.label())
        if fault.trickle > 0:
            self._serve_trickle(handler, body, fault)
            return True
        if fault.delay > 0:
            self._stop_event.wait(fault.delay)
            if self._stop_event.is_set():
                handler.close_connection = True
                return True
        if fault.passthrough:
            return False
        if fault.disconnect:
            handler.close_connection = True
            return True
        payload = (
            fault.body
            if fault.body is not None
            else _error_body(f"injected HTTP {fault.status}", "injected")
        )
        try:
            self._send_json(handler, fault.status, payload, fault.headers)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True  # the client gave up (timeout)
        return True

    def _serve_trickle(self, handler: _Handler, body: Any, fault: Fault) -> None:
        request = body if isinstance(body, dict) else {}
        data = json.dumps(
            ChatResponse(content="(trickled)").to_payload(request, "trickle")
        ).encode("utf-8")
        filler = max(1, int(math.ceil(fault.delay / fault.trickle)))
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(filler + len(data)))
            handler.end_headers()
            handler.wfile.flush()
            for _ in range(filler):
                if self._stop_event.wait(fault.trickle):
                    handler.close_connection = True
                    return
                handler.wfile.write(b" ")
                handler.wfile.flush()
            handler.wfile.write(data)
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True  # the client gave up

    def _serve_embeddings(self, handler: _Handler, method: str, body: Any) -> None:
        if not isinstance(body, dict) or "input" not in body:
            self._record(handler, method, ROUTE_EMBEDDINGS, body, "bad_request")
            self._send_json(
                handler,
                400,
                _error_body(
                    "embeddings request needs 'input'", "invalid_request_error"
                ),
            )
            return
        raw_inputs = body["input"]
        if isinstance(raw_inputs, str) or (
            isinstance(raw_inputs, list)
            and raw_inputs
            and all(isinstance(t, int) for t in raw_inputs)
        ):
            raw_inputs = [raw_inputs]
        if not isinstance(raw_inputs, list) or not raw_inputs:
            self._record(handler, method, ROUTE_EMBEDDINGS, body, "bad_request")
            self._send_json(
                handler,
                400,
                _error_body("'input' must be non-empty", "invalid_request_error"),
            )
            return
        texts = [
            item if isinstance(item, str) else json.dumps(item) for item in raw_inputs
        ]
        dim = int(body.get("dimensions") or self.embedding_dim)
        as_base64 = body.get("encoding_format") == "base64"

        data = []
        for index, text in enumerate(texts):
            vector = self.embed(text, dim)
            encoded: Any = vector
            if as_base64:
                encoded = base64.b64encode(
                    struct.pack(f"<{len(vector)}f", *vector)
                ).decode("ascii")
            data.append({"object": "embedding", "index": index, "embedding": encoded})
        tokens = sum(max(1, len(t) // 4) for t in texts)
        self._record(handler, method, ROUTE_EMBEDDINGS, body, "embedding")
        self._send_json(
            handler,
            200,
            {
                "object": "list",
                "data": data,
                "model": body.get("model") or self.model_name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )

    def _next_chat_response(
        self, body: Dict[str, Any]
    ) -> Tuple[Optional[ScriptedChat], str]:
        with self._lock:
            if self._responses:
                return self._responses.popleft(), "scripted"
        if self.default_responder is not None:
            return self.default_responder(body), "default"
        with self._lock:
            self._unscripted += 1
        return None, "unscripted"

    def _serve_chat(self, handler: _Handler, method: str, body: Any) -> None:
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            self._record(handler, method, ROUTE_CHAT, body, "bad_request")
            self._send_json(
                handler,
                400,
                _error_body(
                    "chat request needs a 'messages' list", "invalid_request_error"
                ),
            )
            return
        if body.get("stream"):
            self._record(handler, method, ROUTE_CHAT, body, "stream_rejected")
            self._send_json(
                handler,
                400,
                _error_body(
                    "stream=true is not supported by FakeLLMServer",
                    "invalid_request_error",
                ),
            )
            return

        try:
            scripted, source = self._next_chat_response(body)
        except Exception as exc:  # a buggy default responder must surface, not hang
            self._record(handler, method, ROUTE_CHAT, body, "responder_error")
            self._send_json(
                handler,
                500,
                _error_body(f"default responder raised: {exc!r}", "fake_llm_responder"),
            )
            return

        if scripted is None:
            self._record(handler, method, ROUTE_CHAT, body, "unscripted")
            self._send_json(
                handler,
                500,
                _error_body(
                    "FakeLLMServer has no scripted response for this request",
                    "fake_llm_unscripted",
                ),
            )
            return
        if isinstance(scripted, Fault):
            if not self._serve_fault(handler, method, ROUTE_CHAT, body, scripted):
                self._record(handler, method, ROUTE_CHAT, body, "unscripted")
                self._send_json(
                    handler,
                    500,
                    _error_body(
                        "a passthrough Fault cannot be a scripted response",
                        "fake_llm_script",
                    ),
                )
            return

        with self._lock:
            self._completion_counter += 1
            completion_id = f"fake{self._completion_counter}"
        if isinstance(scripted, str):
            payload = ChatResponse(content=scripted).to_payload(body, completion_id)
        elif isinstance(scripted, ChatResponse):
            payload = scripted.to_payload(body, completion_id)
        else:
            payload = scripted  # raw payload dict, served verbatim
        self._record(handler, method, ROUTE_CHAT, body, source)
        try:
            self._send_json(handler, 200, payload)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True


# --------------------------------------------------------------------------- #
# pytest integration                                                          #
# --------------------------------------------------------------------------- #

try:  # pytest is optional so non-test code (the eval framework) can import this module
    import pytest
except ImportError:  # pragma: no cover - exercised only outside the test environment
    pytest = None  # type: ignore[assignment]

if pytest is not None:

    @pytest.fixture
    def fake_llm_server():
        """A running FakeLLMServer, stopped after the test."""
        with FakeLLMServer() as server:
            yield server
