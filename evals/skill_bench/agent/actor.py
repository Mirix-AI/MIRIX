"""The actor: model calls through the gateway ``actor`` role, each one recorded.

:class:`Actor` posts OpenAI chat-completions requests to
``<gateway>/runs/<run_id>/actor/v1/chat/completions`` with httpx (async), authenticated
with the gateway's local token, never a provider key. It sends only ``model`` and
``messages``: the gateway owns the rest
(reasoning effort, no temperature, the F9 ``max_tokens``, seed, usage accounting,
retries inside the 600 s deadline).

Every call returns or raises with an :class:`ActorCall` record:

* the full request body exactly as sent, its byte length and sha256;
* the injected memory's sha256, character and ``o200k_base`` token counts (the memory
  text itself is in the request and, once per episode, in the trajectory's injection);
* the input-token estimate and the F9 output cap the gateway will apply to it
  (``min(max_out80, window80 - input)``, o200k text);
* the gateway call id and retry count (to join the gateway ledger), HTTP status,
  latency, the full response body, usage, cost, provider, finish reason;
* the classified failure, if any (``infra/failures.py``).

Failures:

* non-2xx from the gateway, or no gateway at all: :class:`ActorCallError` with the
  classified :class:`~evals.skill_bench.infra.failures.Failure` (a 504 from one attempt
  that had the whole deadline is a MODEL timeout, F9; exhausted retries, time lost in the
  gateway queue, 402 and transport errors are INFRA; gateway refusals are HARNESS);
* HTTP 200 with ``finish_reason`` ``length`` or ``content_filter``: returned, with
  :attr:`ActorReply.failure` set (MODEL). The rollout treats the reply as SkillOpt does
  (see ``agent/rollout.py``) and records the failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import (
    Failure,
    FailureKind,
    GatewayCallError,
    classify_completion,
    classify_exception,
    classify_gateway_error,
)
from evals.skill_bench.infra.gateway import CHAT, LOCAL_TOKEN, gateway_url

ACTOR_ROLE = "actor"
#: Gateway roles an :class:`Actor` can call. The framework's actor uses ``actor``; the
#: judge J (``methods/judge.py``) and the baselines' learner calls reuse the same recorded
#: client with ``judge`` / ``learner`` (MIRIX calls the ``mirix`` role itself).
CHAT_ROLES: tuple[str, ...] = ("actor", "learner", "judge", "mirix")
#: Longer than the gateway's total deadline, so the client sees the gateway's classified 504
#: (the same protocol constant sets MIRIX's timeout).
CLIENT_TIMEOUT_S = protocol.CLIENT_TIMEOUT_S
CONNECT_TIMEOUT_S = 30.0
INVALID_GATEWAY_RESPONSE = "invalid_gateway_response"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_bytes(body: Mapping[str, Any]) -> bytes:
    """The exact bytes the actor sends for ``body`` (compact JSON, UTF-8)."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _encode(body: Mapping[str, Any]) -> tuple[bytes, Any, str]:
    """Wire bytes, the request parsed back from them (a detached copy) and their sha256.

    Pure CPU work on payloads that may be large: run in a worker thread.
    """
    content = request_bytes(body)
    return content, json.loads(content), hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ActorCall:
    """Record of one actor call (H5). ``request`` is the full body as sent."""

    index: int
    run_id: str
    role: str
    model: str
    item: str | None
    phase: str | None
    url: str
    started_at: str
    latency_s: float
    request: Mapping[str, Any]
    request_sha256: str
    request_bytes: int
    memory_sha256: str
    memory_chars: int
    memory_tokens: int
    input_tokens_estimated: int
    input_estimate: Mapping[str, Any]
    max_tokens_expected: int
    http_status: int | None
    gateway_call_id: str | None
    gateway_retries: int | None
    response: Mapping[str, Any] | None
    response_sha256: str | None
    usage: Mapping[str, Any] | None
    cost: float | None
    provider: str | None
    generation_id: str | None
    finish_reason: str | None
    failure: Mapping[str, Any] | None
    error: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActorReply:
    """The model's reply to one successful call.

    ``content`` is ``message.content or ""`` (SkillOpt's reading of a response);
    ``reasoning`` is reasoning text the provider returned separately (audit only, Q20).
    ``failure`` is set for a truncated or filtered reply (a MODEL failure the rollout
    records).
    """

    call: ActorCall
    content: str
    reasoning: str | None
    finish_reason: str | None
    failure: Failure | None


class ActorCallError(GatewayCallError):
    """An actor call without a usable reply; ``call`` is its record, ``failure`` its class."""

    def __init__(self, failure: Failure, call: ActorCall) -> None:
        super().__init__(failure)
        self.call = call


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return text parts
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") in (None, "text")
        )
    return str(content)


def _reasoning_text(message: Mapping[str, Any]) -> str | None:
    for key in ("reasoning", "reasoning_content"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    details = message.get("reasoning_details")
    if isinstance(details, list):
        texts = [
            str(d.get("text") or d.get("summary") or "")
            for d in details
            if isinstance(d, Mapping)
        ]
        joined = "\n".join(t for t in texts if t)
        return joined or None
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class Actor:
    """Calls one chat model through one gateway chat role (``actor`` by default) of one run.

    One instance serves every episode of the run and may be used concurrently. Pass
    ``transport`` (tests: ``httpx.ASGITransport(app=gateway.app)``) or a ready ``client``.
    ``role`` selects another chat role (``judge``, ``learner``): the call records and the
    failure classification then name that role.
    """

    def __init__(
        self,
        *,
        gateway_base_url: str,
        run_id: str,
        model: str,
        token: str = LOCAL_TOKEN,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = CLIENT_TIMEOUT_S,
        models: Mapping[str, Any] | None = None,
        role: str = ACTOR_ROLE,
    ) -> None:
        if role not in CHAT_ROLES:
            raise ValueError(f"role must be one of {CHAT_ROLES}, not {role!r}")
        self.run_id = run_id
        self.model = model
        self.role = role
        self.limits = protocol.model_limits(model, models)
        self.base_url = gateway_url(gateway_base_url, run_id, role)
        self.url = f"{self.base_url}/{CHAT}"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(timeout_s, connect=CONNECT_TIMEOUT_S),
            trust_env=False,  # loopback gateway: never through a proxy
        )
        self._ready = False

    async def start(self) -> None:
        """Load the token encoding off the event loop (its first use may download it)."""
        if not self._ready:
            await tokens.preload_encoding()
            self._ready = True

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    async def __aenter__(self) -> Actor:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def body(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The request body the actor sends for ``messages``."""
        return {"model": self.model, "messages": list(messages)}

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        index: int,
        item: str | None,
        phase: str | None,
        memory: str = "",
        memory_tokens: int = 0,
    ) -> ActorReply:
        """Send one call and return the reply; raise :class:`ActorCallError` without one."""
        await self.start()
        body = self.body(messages)
        content, sent, digest = await asyncio.to_thread(_encode, body)
        estimate = await tokens.estimate_chat_input_async(body, len(content))
        headers = dict(self._headers)
        if item is not None:
            headers["x-skill-bench-item"] = item
        if phase is not None:
            headers["x-skill-bench-phase"] = phase
        base = {
            "index": index,
            "run_id": self.run_id,
            "role": self.role,
            "model": self.model,
            "item": item,
            "phase": phase,
            "url": self.url,
            "started_at": _now(),
            # Parsed back from the bytes sent: exactly the wire content, detached from
            # the caller's message objects.
            "request": sent,
            "request_sha256": digest,
            "request_bytes": len(content),
            "memory_sha256": hashlib.sha256(memory.encode("utf-8")).hexdigest(),
            "memory_chars": len(memory),
            "memory_tokens": memory_tokens,
            "input_tokens_estimated": estimate.total,
            "input_estimate": estimate.to_dict(),
            "max_tokens_expected": self.limits.output_cap(estimate.total),
        }
        started = time.monotonic()

        def record(**values: Any) -> ActorCall:
            fields: dict[str, Any] = {
                "latency_s": round(time.monotonic() - started, 3),
                "http_status": None,
                "gateway_call_id": None,
                "gateway_retries": None,
                "response": None,
                "response_sha256": None,
                "usage": None,
                "cost": None,
                "provider": None,
                "generation_id": None,
                "finish_reason": None,
                "failure": None,
                "error": None,
            }
            fields.update(values)
            return ActorCall(**base, **fields)

        try:
            response = await self._client.post(
                self.url, content=content, headers=headers
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # transport errors, client timeout: no reply at all
            failure = classify_exception(exc, self.role)
            call = record(
                failure=failure.to_dict(),
                error={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
            raise ActorCallError(failure, call) from exc

        gateway = {
            "http_status": response.status_code,
            "gateway_call_id": response.headers.get("x-skill-bench-call-id"),
            "gateway_retries": _as_int(response.headers.get("x-skill-bench-retries")),
            "response_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code != 200:
            failure = classify_gateway_error(response.status_code, data, self.role)
            error = data.get("error") if isinstance(data, Mapping) else None
            call = record(
                **gateway,
                response=data if isinstance(data, Mapping) else None,
                failure=failure.to_dict(),
                error=error
                if isinstance(error, Mapping)
                else {"message": response.text[:500]},
            )
            raise ActorCallError(failure, call)

        choices = data.get("choices") if isinstance(data, Mapping) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], Mapping)
            else None
        )
        if not isinstance(message, Mapping):
            failure = Failure(
                FailureKind.HARNESS,
                INVALID_GATEWAY_RESPONSE,
                self.role,
                response.status_code,
                "gateway returned 200 without choices[0].message",
            )
            call = record(
                **gateway,
                response=data if isinstance(data, Mapping) else None,
                failure=failure.to_dict(),
                error={"message": failure.detail},
            )
            raise ActorCallError(failure, call)

        assert isinstance(data, Mapping) and isinstance(choices, list)
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else None
        finish_reason = choices[0].get("finish_reason")
        failure = classify_completion(data, self.role)
        call = record(
            **gateway,
            response=data,
            usage=usage,
            cost=usage.get("cost") if usage else None,
            provider=data.get("provider"),
            generation_id=data.get("id"),
            finish_reason=finish_reason,
            failure=failure.to_dict() if failure else None,
        )
        return ActorReply(
            call=call,
            content=_content_text(message.get("content")),
            reasoning=_reasoning_text(message),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            failure=failure,
        )
