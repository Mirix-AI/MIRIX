"""Loopback OpenRouter gateway: the only process that reads the real key.

URL scheme (OpenAI-compatible base URL per cell and role)::

    http://127.0.0.1:<port>/runs/<run_id>/<role>/v1/chat/completions
    http://127.0.0.1:<port>/runs/<run_id>/<role>/v1/embeddings

``role`` is one of ``actor | learner | judge | mirix`` (chat) or ``embedding``
(embeddings). Callers authenticate with :data:`LOCAL_TOKEN`, never a provider key.
Optional request headers ``x-skill-bench-item`` (opaque task uid) and
``x-skill-bench-phase`` tag the ledger line; ``POST /runs/<run_id>/phase``
sets the phase for every later call of that run (MIRIX sends no phase header; it
sends the item header, the session a call serves, with ``MIRIX_LLM_SESSION_HEADER``,
which ``run.py`` sets: B32).

Request rewrite (chat): only the fields in ``protocol.CHAT_FORWARDED_FIELDS`` are
forwarded (model, messages, tools, tool_choice, response_format, stop,
parallel_tool_calls, user); everything else the caller sends (temperature, top_p,
n, penalties, logprobs, presets, provider routing, plugins ...) is dropped and the
dropped names are logged (F8, Q23: OpenRouter chooses the provider and may fail
over; the chosen provider is logged). The one exception is the protocol's own provider
policy (``protocol.PROVIDER_POLICY``):
for a model the policy names, the gateway sets ``provider={"ignore": [...]}`` itself,
and OpenRouter chooses among the other providers; the ledger's
``wire`` records it (a caller's own ``provider`` is still dropped). A 404 saying no
endpoint or allowed provider is left is provider unavailability (retried, INFRA); one
saying no endpoint supports the request or the account's data policy is not retried.
An answer served by a provider the routing ignores is never returned: the billed
attempt is recorded (``outcome: ignored_provider``) and retried like provider
unavailability. Never
stream upstream (a streaming caller gets
the full response re-framed as SSE); ``reasoning={"effort": "medium"}``;
``max_tokens`` overridden with the F9 cap ``min(max_out80, window80 - input)``
computed from this request (o200k text), also kept below the
top provider's window with the tokenizer margin
(``context_length - ceil(input * (1 + margin)) - 1``, ``protocol.ModelLimits``), 400
when the input leaves less than MIN_OUTPUT_TOKENS; ``seed=42`` when the model supports it; usage accounting
requested.

Deadline and retries: every call has one total deadline (600 s) that includes
queueing, retries and back-off. Only infrastructure failures are retried
(connect/read errors, 408, 429 honoring Retry-After, 5xx, provider-unavailable
bodies, a reply without any output and no finish reason), at most 5 times with
exponential back-off capped by the remaining time. Hitting the deadline returns 504
and is never retried; its body carries ``attempt_budget_s`` (the deadline left when
the last attempt started), ``queue_wait_s`` and ``pre_attempt_s`` so the time lost
in the gateway is never blamed on the model (``infra/failures.py``). Every error body
also says whose failure it is under H8, ``blame`` (``model``, ``infra`` or ``harness``:
``infra/failures.classify_gateway_error`` on that body and role), so a caller that does
not run the harness's classifier can still tell them apart: MIRIX's M3.5 retry policy
treats a 504 with ``blame: model`` as the call itself timing out (retried once, then
counted), not as an infrastructure failure.

Ledger: ``<runs_dir>/<run_id>/gateway.jsonl``, one fsync'd line per event: a
``call_started`` line and a final ``call`` line sharing ``call_id``. Every admitted call
gets its ``call_started`` line before a pause can hold it, so a call the pause answers
402 is placed and quiesced like any other. The final line's ``ts`` is when the call
arrived and ``completed_at`` when its line was written. Every attempt that returned a
body keeps its own usage, cost, provider and generation id (``cost_total`` sums them:
retried attempts may be billed). An attempt the deadline cut off after its request was
sent returned no body, yet the provider ran on and OpenRouter billed it: it is marked
``billed: "unknown"`` and counted in the call's ``unbilled_deadline_attempts``, since
its cost is in no ``cost_total``. So is an attempt whose connection dropped after the
request was sent (a read error, the upstream closing mid-answer), which is retried;
``unknown_cost_attempts`` counts both kinds. Whether the request went out is what
httpx's own typed timeouts and httpcore's trace of the request body say: a pool, connect
or write timeout, or a deadline met before the body was sent, never reached the provider
and is not marked. Each call line carries the sha256 of the gateway's effective protocol
settings and whether they equal the frozen protocol (``protocol_conformant``);
``runs/_gateway/starts.jsonl`` records the full settings of every gateway start. A
second identical request (same run, role and request sha256) within an hour of a failed
one is flagged ``repeat_of``. It is information for the audit, which judges it by role:
on the ``mirix`` role (and ``embedding`` in MIRIX cells) MIRIX's own retry policy makes
such application-level retries, while SDK-level retries are ruled out by
``MIRIX_LLM_MAX_RETRIES=0`` (set and probed by ``infra/mirix_server.py``; ``audit.py``).
If the final ledger line cannot be written after a billed success, the response is still
returned and the line is kept in memory until the next write succeeds.

Budget: ``GET <upstream>/key`` is polled (30 s cache); below the floor the gateway
pauses. A call that arrives while paused waits (within its deadline) for the balance
to recover, then returns 402 if it has not. The pause only lifts when a known
balance is back above the floor.

Spend budget (``--budget USD``, optional): the most this runs directory may spend. Two
bases (``--budget-basis``):

* ``key`` (the default; every experiment gateway): counted on the key's all-time
  ``usage`` from the value it had when a budget was first set here
  (``<runs_dir>/_gateway/budget.json``, kept across gateway restarts, so a relaunch after
  a pause does not start a new budget; a later ``--budget`` replaces the amount, not the
  starting point). The gateway pauses once ``usage`` has grown by the budget
  (``budget_spent``), and also while the budget cannot be established
  (``budget_unknown``: no usage read yet, an unreadable record) or the usage reads below
  the record's starting point (``budget_invalid``: another key). OpenRouter updates
  ``usage`` a little after each call, so the spend can pass the budget by what the
  in-flight calls cost.
* ``ledger`` (a gateway whose spend must not include the experiment's): counted on the
  ``cost_total`` of the final call lines of this runs directory's own ledgers
  (``<runs_dir>/*/gateway.jsonl``, read at start and at each balance check, and counted
  as each call finishes, so a second gateway on the same runs directory spends it too),
  from their total when a ledger budget was first set here
  (``<runs_dir>/_gateway/budget_ledger.json``, ``start_ledger_usd``; same rules for
  restarts and amounts). The experiment's spend on the same key never uses it up. A call
  that spends it pauses the gateway at once (``budget_spent``); ledgers that cannot be
  read or a record that cannot be read make it ``budget_unknown``; ledgers holding less
  than the start (removed) ``budget_invalid``. Attempts of unknown cost
  (``unknown_cost_attempts``) are counted apart, not in the spend.

``--experiment-budget PATH`` (repeatable, with a budget): another runs directory's
key-basis record (``budget.json``) this gateway also stops on: once the key's ``usage``
has grown from its ``start_usage_usd`` by its last ``budget_usd``, the pause is
``budget_spent`` whatever this gateway's own budget says, so a second gateway on the
experiment's key can never pass the experiment's cap. A record that cannot be read, or
no usage, is ``budget_unknown`` (waitable, never guessed); a usage below its start (this
gateway uses another key) is recorded as not checked. A stop reason of either budget
wins over a waitable one. ``/health`` and ``_gateway/balance.jsonl`` show the budget
view (``basis``, ``spent_usd``, ``left_usd``, ``experiment``); ``starts.jsonl`` records
``budget_basis`` and ``experiment_budgets``. Unlike the floor, a budget does not depend
on the key's daily limit reset. Calls already sent when it pauses still finish and are
billed.

Pause reasons (``/health`` ``pause_reason``) come in two sets the scheduler and the
supervisor act on: :data:`WAITABLE_PAUSE_REASONS` lift by themselves (a top-up, the
key's daily limit reset, the next usage read), :data:`STOP_PAUSE_REASONS` need a person
(the budget set for the runs directory is spent, or the budget record belongs to another
key).

Runs can be abandoned (``POST /runs/<run_id>/abandon``, e.g. a learning attempt that
H8 invalidates): later calls of that run get 410, so an orphaned MIRIX server of a dead
attempt cannot keep spending.

Markers (a learning phase resumes from checkpoints):
``POST /runs/<run_id>/mark`` with ``{"event": "checkpoint" | "resume", ...}`` appends one
fsync'd ``checkpoint`` or ``resume`` line to the run's ledger, in the ledger's own write
order and stamped like every line (``ts``, ``run_id``, ``gateway_start``), and answers
``{"ok": true, "offset": <byte offset where the line starts>, "ts": ...}``. The calls whose
``call_started`` lies between a checkpoint marker and a later resume marker that restores
it are that resume's discarded segment (``run.py``, ``audit.py``). Any other event, a body
that is not a small flat JSON object, or a marker without its fields is refused with 400
before anything is written (``run.py`` probes the route with ``{"event": "probe"}``: 404
or 405 means a gateway without it); an abandoned run answers 410.

``/health`` also reports the upstream slots (``slots``: ``max``, ``in_use`` and the calls
``waiting`` for one), for monitoring.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import email.utils
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from evals.skill_bench import protocol
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import classify_gateway_error
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    cleanup_owned,
    ends_torn,
    port_is_free,
)

log = logging.getLogger(__name__)

KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18540
DEFAULT_BALANCE_FLOOR_USD = 10.0
LOCAL_TOKEN = "skill-bench-local-credential-not-a-provider-key"
CHAT = "chat/completions"
EMBEDDINGS = "embeddings"
ENDPOINTS = (CHAT, EMBEDDINGS)
PHASES = ("setup", "learning", "test")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
REPEAT_WINDOW_S = (
    3600.0  # identical requests after a failure within this window are flagged
)
ITEM_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: ``billed`` of an attempt the deadline cut off: OpenRouter bills it, the ledger never
#: learns how much.
BILLED_UNKNOWN = "unknown"
EMBEDDING_ALIASES = {"gemini-embedding-001": protocol.EMBEDDING_MODEL}
RETRYABLE_STATUS = frozenset({408, 429, *range(500, 600)})
PROVIDER_UNAVAILABLE_PATTERNS = (
    "provider returned error",
    "provider unavailable",
    "no available provider",
    "temporarily unavailable",
    "overloaded",
    "upstream error",
    "rate limit",
)
#: OpenRouter's 404 when no endpoint of the model can take the request, for example while
#: every provider the protocol's routing leaves (``protocol.PROVIDER_POLICY``) is down:
#: provider unavailability, retried like a 5xx (INFRA), never the caller's fault.
NO_ENDPOINT_PATTERNS = ("no endpoints found", "no allowed providers")
#: ... except the 404s that say no endpoint can take *this* request or *this* account
#: (``No endpoints found that support tool use``, ``... that can handle the requested
#: parameters``, ``... matching your data policy``): retrying cannot change that, so they
#: are not retried, and the call fails at once as the upstream's refusal (HARNESS) for a
#: person to look at, instead of spending the whole deadline and then looping through
#: resumes as INFRA.
NO_ENDPOINT_FINAL_PATTERNS = (
    "that support",
    "that can handle",
    "supporting",
    "requested parameters",
    "data policy",
)
#: The attempt outcome of an answer the provider routing the gateway sent says to ignore
#: (``wire.provider``; ``protocol.ignored_by``): OpenRouter served the call from a
#: provider the protocol excludes (a routing fault, or a new name for the same provider).
#: The answer is never returned: the attempt, billed, is retried like provider
#: unavailability (INFRA), and the audit's ``ignored_provider_served`` stays a backstop
#: for an answer that got through.
IGNORED_PROVIDER_OUTCOME = "ignored_provider"
CONNECT_TIMEOUT_S = 30.0
#: How long past a call's deadline ``asyncio.wait_for`` waits before it cuts the attempt
#: off itself: httpx's own timeouts (set to the deadline) fire first and say what was cut
#: off — a pool, connect or write timeout never delivered the request.
#: ``wait_for`` then only ends an answer that keeps trickling in past the deadline.
DEADLINE_GRACE_S = 0.5
BALANCE_TIMEOUT_S = 15.0
#: The spend budget's record, under ``<runs_dir>/_gateway`` (:meth:`Gateway._update_budget`).
BUDGET_FILE = "budget.json"
#: What the spend budget counts (``--budget-basis``): the key's all-time ``usage``
#: (every experiment gateway), or the ``cost_total`` of this runs directory's own
#: ledgers (the experiment's spend on the same key never uses it up).
BUDGET_BASIS_KEY = "key"
BUDGET_BASIS_LEDGER = "ledger"
BUDGET_BASES: tuple[str, ...] = (BUDGET_BASIS_KEY, BUDGET_BASIS_LEDGER)
#: The ledger-basis budget's record, under ``<runs_dir>/_gateway``.
BUDGET_LEDGER_FILE = "budget_ledger.json"
BUDGET_UNKNOWN = "budget_unknown"
BUDGET_SPENT = "budget_spent"
BUDGET_INVALID = "budget_invalid"
#: Pauses of the spend budget: they lift once the budget is known and not spent, whatever
#: the balance (an upstream 402 lifts only on a known balance above the floor).
BUDGET_PAUSE_REASONS = frozenset({BUDGET_UNKNOWN, BUDGET_SPENT, BUDGET_INVALID})
#: The key's balance (``limit_remaining``, else the account's credits) is below the floor;
#: the key's daily limit running out reads the same.
BALANCE_BELOW_FLOOR = "balance_below_floor"
#: OpenRouter answered 402 (insufficient credit).
UPSTREAM_402 = "upstream_402"
#: Pauses that lift by themselves (a top-up, the daily limit's reset, the next usage
#: read): a scheduler with ``--auto-resume`` waits them out.
WAITABLE_PAUSE_REASONS = frozenset({BALANCE_BELOW_FLOOR, UPSTREAM_402, BUDGET_UNKNOWN})
#: Pauses only a person lifts: the budget set for the runs directory is spent, or the
#: budget record belongs to another key. Nothing more is launched on them.
STOP_PAUSE_REASONS = frozenset({BUDGET_SPENT, BUDGET_INVALID})
#: Ledger events of ``POST /runs/<run_id>/mark`` (``cell_records`` names the same values).
LEDGER_EVENT_CHECKPOINT = "checkpoint"
LEDGER_EVENT_RESUME = "resume"
MARK_EVENTS: tuple[str, ...] = (LEDGER_EVENT_CHECKPOINT, LEDGER_EVENT_RESUME)
#: The largest marker body the gateway takes, and the longest string value in it.
MARK_MAX_BYTES = 4096
MARK_MAX_STRING = 256
#: Fields the gateway writes itself on a marker line (a caller's are refused).
MARK_RESERVED_FIELDS = frozenset({"ts", "run_id", "gateway_start", "call_id"})
_CHECKPOINT_ID_RE = re.compile(r"^c(\d{4,})-(init|played|done)$")
#: How far the key's usage may read below the budget's starting point before it counts as
#: another key (OpenRouter reports usage as a float).
BUDGET_USAGE_TOLERANCE_USD = 0.01
SERVER_KEEP_ALIVE_S = 75
ERROR_MESSAGE_MAX = 500
# Caller fields the gateway sets itself (not reported as "dropped").
_CHAT_OWNED_FIELDS = frozenset(
    {
        "stream",
        "stream_options",
        "reasoning",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "usage",
        "transforms",
    }
)
_CHAT_FORWARDED_FIELDS = frozenset(protocol.CHAT_FORWARDED_FIELDS)
_EMBEDDING_OWNED_FIELDS = frozenset(
    {"stream", "stream_options", "provider", "models", "route"}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usd(value: Any) -> float | None:
    """A JSON number as a float (None: missing, a bool or not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _worse_pause(a: str | None, b: str | None) -> str | None:
    """Of two budget pause reasons (None: none), the one to pause on: a stop reason
    (``budget_spent``, ``budget_invalid``) before a waitable one, ``a`` first."""
    for reason in (a, b):
        if reason in STOP_PAUSE_REASONS:
            return reason
    return a or b


def ledger_spend(runs_dir: Path) -> tuple[float, int]:
    """The ``cost_total`` sum and the ``unknown_cost_attempts`` of every final call line
    of the ledgers directly under ``runs_dir`` (``<run_id>/gateway.jsonl``): what the
    ledger budget basis counts. Lines split on ``\\n`` only; a torn or foreign line is
    skipped. Blocking; an unreadable ledger raises OSError (the budget is then unknown,
    never undercounted)."""
    total = 0.0
    unknown = 0
    for path in sorted(runs_dir.glob("*/gateway.jsonl")):
        for raw in path.read_bytes().split(b"\n"):
            try:
                row = json.loads(raw)
            except ValueError:
                continue  # a torn line (X15) or an empty one
            if not isinstance(row, dict) or row.get("event") != "call":
                continue
            cost = row.get("cost_total")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total += float(cost)
            count = row.get("unknown_cost_attempts")
            if isinstance(count, int) and not isinstance(count, bool):
                unknown += count
    return total, unknown


def _append_all(path: Path, records: Sequence[Mapping[str, Any]]) -> int:
    """Append several JSON lines with one write and one fsync; returns the byte offset
    where the last record's line starts (the ``mark`` answer's ``offset``).

    A torn last line (the gateway killed or the disk full mid-append) is ended with a
    newline first, so the new lines never merge with the fragment (X15). The file is
    opened ``O_APPEND``: the write lands at the end whoever else appended, and the offset
    is read back from where it ended.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    ]
    data = b"".join(lines)
    if ends_torn(path):
        data = b"\n" + data
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        end = os.lseek(fd, 0, os.SEEK_CUR)
        os.fsync(fd)
    finally:
        os.close(fd)
    return end - len(lines[-1]) if lines else end


def _fsync_dir(directory: Path) -> None:
    """Make a rename in ``directory`` durable (no-op where directories cannot be opened)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def gateway_url(base_url: str, run_id: str, role: str) -> str:
    """OpenAI-compatible base URL of one run and role, e.g. ``.../runs/<run>/actor/v1``."""
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"invalid run_id {run_id!r}")
    if role not in protocol.ROLES:
        raise ValueError(f"invalid role {role!r}")
    return f"{base_url.rstrip('/')}/runs/{run_id}/{role}/v1"


# ---------------------------------------------------------------------------
# Request preparation (pure; runs in a worker thread for large payloads)
# ---------------------------------------------------------------------------


class GatewayReject(Exception):
    """A request the gateway refuses before any upstream call."""

    def __init__(self, status: int, kind: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.kind, self.message, self.extra = status, kind, message, extra


@dataclass(frozen=True)
class PreparedRequest:
    endpoint: str
    model: str
    model_requested: Any
    stream_requested: bool
    body: dict[str, Any]
    content: bytes
    sha256: str
    estimate: tokens.TokenEstimate
    max_tokens_requested: Any
    max_tokens_rewritten: int | None
    max_tokens_f9: int | None = None  # the F9 formula alone (before the provider clamp)
    #: The top provider's window with the tokenizer margin (N5); None if unknown.
    max_tokens_provider: int | None = None
    #: Which limit set ``max_tokens_rewritten``: max_out80, window80 or provider_window.
    max_tokens_bound: str | None = None
    dropped_fields: tuple[str, ...] = ()  # caller fields the allowlist removed


def dropped_chat_fields(body: Mapping[str, Any]) -> tuple[str, ...]:
    """Caller fields the chat rewrite removes (not forwarded, not set by the gateway)."""
    return tuple(
        sorted(
            str(k)
            for k in body
            if k not in _CHAT_FORWARDED_FIELDS and k not in _CHAT_OWNED_FIELDS
        )
    )


def rewrite_chat_body(
    body: Mapping[str, Any],
    *,
    model: str,
    max_tokens: int,
    seed: int | None,
    effort: str,
    provider: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Apply the H3 chat rewrite to a caller body (the caller's dict is not modified).

    Only ``protocol.CHAT_FORWARDED_FIELDS`` survive; the gateway then sets its own fields,
    ``provider`` last and only when the model has a routing policy
    (``protocol.provider_routing``; the ledger's ``wire.provider``), so a body without one
    is byte for byte what it was before the policy existed.
    """
    out = {k: v for k, v in body.items() if k in _CHAT_FORWARDED_FIELDS}
    out["model"] = model
    out["stream"] = False
    out["reasoning"] = {"effort": effort}
    out["max_tokens"] = max_tokens
    if seed is not None:
        out["seed"] = seed
    out["usage"] = {"include": True}
    out["transforms"] = []  # no prompt compression, ever
    if provider:
        out["provider"] = {str(k): list(v) for k, v in provider.items()}
    return out


def rewrite_embedding_body(body: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    out = {k: v for k, v in body.items() if k not in _EMBEDDING_OWNED_FIELDS}
    out["model"] = model
    return out


def prepare_request(
    raw: bytes,
    endpoint: str,
    *,
    limits: Mapping[str, protocol.ModelLimits],
    seed_models: frozenset[str],
    seed: int,
    effort: str,
    provider_policy: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> PreparedRequest:
    """Parse, whitelist, budget-check and rewrite one request; raise :class:`GatewayReject`.

    ``provider_policy`` is the gateway's routing per chat model (``GatewayConfig``; None:
    the protocol's, ``protocol.PROVIDER_POLICY``)."""
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise GatewayReject(
            400, "invalid_json", "request body is not valid JSON"
        ) from exc
    if not isinstance(body, dict):
        raise GatewayReject(400, "invalid_json", "request body must be a JSON object")
    requested = body.get("model")
    stream = bool(body.get("stream"))
    if endpoint == CHAT:
        model = requested if isinstance(requested, str) else ""
        if model not in limits:
            raise GatewayReject(
                403,
                "model_not_allowed",
                f"chat model {requested!r} is not whitelisted",
                requested=str(requested)[:200],
                allowed=sorted(limits),
            )
        try:
            estimate = tokens.estimate_chat_input(body)
        except tokens.UnsupportedContentError as exc:
            raise GatewayReject(400, "unsupported_content_part", str(exc)) from exc
        except ValueError as exc:
            raise GatewayReject(400, "invalid_messages", str(exc)) from exc
        lim = limits[model]
        if estimate.total > lim.gateway_input_budget:
            raise GatewayReject(
                400,
                "context_budget_exceeded",
                f"estimated input {estimate.total} tokens exceeds the input budget {lim.gateway_input_budget} "
                f"for {model} (the largest input that leaves {protocol.MIN_OUTPUT_TOKENS} output "
                f"tokens under F9 and the top provider's window with a {lim.tokenizer_margin} "
                "tokenizer margin)",
                input_tokens_estimated=estimate.total,
                input_budget=lim.gateway_input_budget,
                window80=lim.window80,
            )
        cap = lim.output_cap(estimate.total)
        f9_cap: int | None = lim.f9_output_cap(estimate.total)
        provider_cap = lim.provider_output_cap(estimate.total)
        bound: str | None = lim.output_cap_bound(estimate.total)
        max_requested = body.get("max_completion_tokens", body.get("max_tokens"))
        dropped = dropped_chat_fields(body)
        out = rewrite_chat_body(
            body,
            model=model,
            max_tokens=cap,
            seed=seed if model in seed_models else None,
            effort=effort,
            provider=protocol.provider_routing(model, provider_policy),
        )
    elif endpoint == EMBEDDINGS:
        model = (
            EMBEDDING_ALIASES.get(requested, requested)
            if isinstance(requested, str)
            else ""
        )
        if model != protocol.EMBEDDING_MODEL:
            raise GatewayReject(
                403,
                "model_not_allowed",
                f"embedding model {requested!r} is not whitelisted",
                requested=str(requested)[:200],
                allowed=[protocol.EMBEDDING_MODEL],
            )
        if body.get("input") in (None, "", []):
            raise GatewayReject(400, "empty_input", "embedding input is empty")
        estimate = tokens.estimate_embedding_input(body)
        cap = max_requested = f9_cap = provider_cap = bound = None
        dropped = ()
        out = rewrite_embedding_body(body, model=model)
    else:
        raise GatewayReject(
            404, "unsupported_endpoint", f"unsupported endpoint {endpoint!r}"
        )
    content = json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return PreparedRequest(
        endpoint=endpoint,
        model=model,
        model_requested=requested,
        stream_requested=stream,
        body=out,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        estimate=estimate,
        max_tokens_requested=max_requested,
        max_tokens_rewritten=cap,
        max_tokens_f9=f9_cap,
        max_tokens_provider=provider_cap,
        max_tokens_bound=bound,
        dropped_fields=dropped,
    )


_MARK_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_mark(raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """A ``POST /runs/<run_id>/mark`` body, or why it is refused (400, nothing written).

    The body is a flat JSON object of at most :data:`MARK_MAX_BYTES`: lower-case field
    names, and values that are null, a boolean, a finite number or a string of at most
    :data:`MARK_MAX_STRING` characters; never a field the gateway writes itself
    (:data:`MARK_RESERVED_FIELDS`). Per event:

    * ``checkpoint``: ``id`` a checkpoint id (``c0000-init``, ``c<p:04d>-played``,
      ``c<p:04d>-done``) and ``kind`` / ``position`` equal to its kind and position;
      ``snapshot_sha256`` 64 lower-case hex characters;
    * ``resume``: ``segment`` an integer from 1; ``checkpoint`` a checkpoint id and
      ``checkpoint_offset`` an integer from 0, or both null (a resume before the first
      checkpoint); ``launch`` a string or null.
    """
    if len(raw) > MARK_MAX_BYTES:
        return None, f"a marker body is at most {MARK_MAX_BYTES} bytes"
    try:
        body = json.loads(raw)
    except ValueError:
        return None, "the marker body is not JSON"
    if not isinstance(body, dict):
        return None, "the marker body must be a JSON object"
    event = body.get("event")
    if event not in MARK_EVENTS:
        return None, f"event must be one of {list(MARK_EVENTS)}"
    for key, value in body.items():
        if not isinstance(key, str) or not _MARK_FIELD_RE.fullmatch(key):
            return None, f"field {str(key)[:40]!r} is not a lower-case name"
        if key in MARK_RESERVED_FIELDS:
            return None, f"field {key!r} is written by the gateway"
        if isinstance(value, str):
            if len(value) > MARK_MAX_STRING:
                return (
                    None,
                    f"field {key!r} is longer than {MARK_MAX_STRING} characters",
                )
        elif isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None, f"field {key!r} is not a finite number"
        elif value is not None and not isinstance(value, (bool, int)):
            return None, f"field {key!r} is not a flat value"
    if event == LEDGER_EVENT_CHECKPOINT:
        match = _CHECKPOINT_ID_RE.fullmatch(str(body.get("id")))
        if match is None or not isinstance(body.get("id"), str):
            return None, "a checkpoint marker needs its checkpoint id"
        position, kind = int(match.group(1)), match.group(2)
        if body.get("kind") != kind or not _is_int(body.get("position")):
            return None, "the checkpoint marker's kind and position must match its id"
        if body["position"] != position or (kind == "init" and position != 0):
            return None, "the checkpoint marker's kind and position must match its id"
        sha = body.get("snapshot_sha256")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            return None, "a checkpoint marker needs the snapshot's sha256"
    else:
        segment = body.get("segment")
        if not _is_int(segment) or segment < 1:
            return None, "a resume marker needs its segment number (from 1)"
        checkpoint, offset = body.get("checkpoint"), body.get("checkpoint_offset")
        if (checkpoint is None) != (offset is None):
            return None, "checkpoint and checkpoint_offset are both set or both null"
        if checkpoint is not None and (
            not isinstance(checkpoint, str)
            or _CHECKPOINT_ID_RE.fullmatch(checkpoint) is None
            or not _is_int(offset)
            or offset < 0
        ):
            return None, "a resume marker names a checkpoint id and its ledger offset"
        if body.get("launch") is not None and not isinstance(body.get("launch"), str):
            return None, "launch must be a string"
    return body, None


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def sse_frames(data: Mapping[str, Any]) -> Iterator[str]:
    """Re-frame a complete chat completion as OpenAI SSE chunks, ending with usage and ``[DONE]``."""
    common = {
        k: data[k]
        for k in ("id", "created", "model", "provider", "system_fingerprint")
        if k in data
    }
    common["object"] = "chat.completion.chunk"
    for choice in data.get("choices") or []:
        index = choice.get("index", 0)
        delta = dict(choice.get("message") or {})
        calls = delta.get("tool_calls")
        if calls:
            delta["tool_calls"] = [dict(call, index=i) for i, call in enumerate(calls)]
        yield _frame(
            common
            | {"choices": [{"index": index, "delta": delta, "finish_reason": None}]}
        )
        final: dict[str, Any] = {
            "index": index,
            "delta": {},
            "finish_reason": choice.get("finish_reason"),
        }
        if "native_finish_reason" in choice:
            final["native_finish_reason"] = choice["native_finish_reason"]
        yield _frame(common | {"choices": [final]})
    yield _frame(common | {"choices": [], "usage": data.get("usage")})
    yield "data: [DONE]\n\n"


def _frame(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _error_body(
    status: int, kind: str, message: str, source: str = "gateway", **extra: Any
) -> dict[str, Any]:
    return {
        "error": {
            "source": source,
            "type": kind,
            "code": status,
            "message": message,
            **extra,
        }
    }


def _with_blame(error: dict[str, Any], status: int, role: str | None) -> None:
    """Add ``blame`` to an error body's ``error`` object: whose failure H8 says it is
    (``model``, ``infra`` or ``harness``), by the harness's own classifier."""
    error["blame"] = classify_gateway_error(status, {"error": error}, role).kind.value


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP date)."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def _provider_unavailable(error: Mapping[str, Any]) -> bool:
    if _as_int(error.get("code")) in RETRYABLE_STATUS:
        return True
    message = str(error.get("message", "")).lower()
    return any(p in message for p in PROVIDER_UNAVAILABLE_PATTERNS)


def _no_endpoint(error: Any, response: httpx.Response) -> bool:
    """Whether a 404 is OpenRouter's "no endpoints found" / "no allowed providers" (see
    :data:`NO_ENDPOINT_PATTERNS`), not a wrong URL, nor one that says no endpoint can
    take this request or account (:data:`NO_ENDPOINT_FINAL_PATTERNS`)."""
    if isinstance(error, Mapping):
        message = str(error.get("message", ""))
    else:
        try:
            message = response.text
        except Exception:  # noqa: BLE001 - an unreadable body is not the marker
            message = ""
    lowered = message.lower()
    return any(p in lowered for p in NO_ENDPOINT_PATTERNS) and not any(
        p in lowered for p in NO_ENDPOINT_FINAL_PATTERNS
    )


def _short_error(error: Any) -> dict[str, Any]:
    if not isinstance(error, Mapping):
        return {"message": str(error)[:ERROR_MESSAGE_MAX]}
    return {
        "code": error.get("code"),
        "type": error.get("type"),
        "message": str(error.get("message", ""))[:ERROR_MESSAGE_MAX],
    }


def _choice_is_empty(choice: Mapping[str, Any]) -> bool:
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return True
    content = message.get("content")
    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    return not has_content and not message.get("tool_calls")


def _billing(data: Any) -> dict[str, Any]:
    """Usage, cost, provider and generation id of one upstream body (for the attempt log)."""
    if not isinstance(data, Mapping):
        return {}
    usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else None
    out: dict[str, Any] = {}
    if usage:
        out["usage"] = dict(usage)
        out["cost"] = usage.get("cost")
    for key, name in (("id", "generation_id"), ("provider", "provider")):
        if data.get(key) is not None:
            out[name] = data.get(key)
    return out


@dataclass
class _Verdict:
    kind: str  # ok | retry | fail | paused
    status: int
    data: Any = None
    failure: str = ""
    retry_after: float | None = None
    error: dict[str, Any] | None = None


def judge_response(endpoint: str, response: httpx.Response) -> _Verdict:
    """Classify one upstream response: success, retryable infra failure, hard failure or out of credit."""
    status = response.status_code
    try:
        data = response.json()
    except ValueError:
        data = None
    error = data.get("error") if isinstance(data, dict) else None
    if status == 402 or (
        isinstance(error, Mapping) and _as_int(error.get("code")) == 402
    ):
        return _Verdict(
            "paused",
            402,
            data,
            "upstream_402",
            error=_short_error(error or "payment required"),
        )
    if status in RETRYABLE_STATUS:
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        return _Verdict(
            "retry",
            status,
            data,
            f"http_{status}",
            retry_after,
            _short_error(error or response.text),
        )
    if status == 404 and _no_endpoint(error, response):
        # No endpoint of the model can take the request right now (every provider the
        # routing leaves is down, or none serves this request): provider unavailability.
        return _Verdict(
            "retry",
            status,
            data,
            "no_endpoint",
            error=_short_error(error or response.text),
        )
    if status != 200:
        return _Verdict(
            "fail",
            status if status >= 400 else 502,
            data,
            f"http_{status}",
            error=_short_error(error or response.text),
        )
    if not isinstance(data, dict):
        return _Verdict("retry", 502, None, "invalid_json")
    if isinstance(error, Mapping):
        code = _as_int(error.get("code"))
        if (
            code is not None
            and 400 <= code < 500
            and code not in (408, 429)
            and not _provider_unavailable(error)
        ):
            return _Verdict("fail", code, data, "error_body", error=_short_error(error))
        # An error object inside HTTP 200 without a client-error code is a provider-side
        # failure (OpenRouter reports mid-flight provider errors this way): retry.
        return _Verdict(
            "retry",
            code if code in RETRYABLE_STATUS else 502,
            data,
            "provider_error_body",
            error=_short_error(error),
        )
    if endpoint == CHAT:
        choices = data.get("choices")
        if not choices:
            return _Verdict("retry", 502, data, "empty_choices")
        for choice in choices:
            if not isinstance(choice, Mapping):
                return _Verdict("retry", 502, data, "invalid_choice")
            if choice.get("finish_reason") == "error" or choice.get("error"):
                return _Verdict(
                    "retry",
                    502,
                    data,
                    "choice_error",
                    error=_short_error(choice.get("error") or "error"),
                )
            if choice.get("finish_reason") is None and _choice_is_empty(choice):
                # No output and no finish reason: the provider dropped the generation.
                # A model that chose to say nothing reports finish_reason "stop".
                return _Verdict(
                    "retry",
                    502,
                    data,
                    "empty_unfinished_choice",
                    error={
                        "message": "choice without content, tool calls or finish_reason",
                        "native_finish_reason": choice.get("native_finish_reason"),
                    },
                )
    elif not data.get("data"):
        return _Verdict("retry", 502, data, "empty_embeddings")
    return _Verdict("ok", 200, data)


def refuse_ignored_provider(verdict: _Verdict, routing: Any) -> _Verdict:
    """An ok ``verdict`` whose answer came from a provider ``routing`` (the ``provider``
    object the call went out with) ignores becomes a retry
    (:data:`IGNORED_PROVIDER_OUTCOME`); any other verdict is returned as it is."""
    if verdict.kind != "ok" or not isinstance(verdict.data, Mapping):
        return verdict
    served = verdict.data.get("provider")
    slug = protocol.ignored_by(routing, served)
    if slug is None:
        return verdict
    return _Verdict(
        "retry",
        502,
        verdict.data,
        IGNORED_PROVIDER_OUTCOME,
        error={
            "message": f"answer served by {served}, a provider the call's routing "
            f"ignores ({slug}); not returned",
            "provider": served,
            "ignored": slug,
        },
    )


@dataclass
class Outcome:
    status: str  # ok | upstream_error | infra_error | deadline_exceeded | budget_paused
    http_status: int
    data: Any = None
    content: bytes | None = None
    retries: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatewayConfig:
    api_key: str = field(repr=False)
    models: Mapping[str, Any]  # parsed config/models.json
    runs_dir: Path = protocol.RUNS_DIR
    upstream: str = DEFAULT_UPSTREAM
    deadline_s: float = protocol.CALL_DEADLINE_S
    max_retries: int = protocol.GATEWAY_MAX_RETRIES
    backoff_base_s: float = protocol.GATEWAY_BACKOFF_BASE_S
    backoff_cap_s: float = protocol.GATEWAY_BACKOFF_CAP_S
    balance_floor_usd: float = DEFAULT_BALANCE_FLOOR_USD
    #: The most this runs directory may spend (USD, on the key's ``usage``), or None: only
    #: the balance floor stops spending (see the module docstring, "Spend budget").
    budget_usd: float | None = None
    #: What ``budget_usd`` counts: :data:`BUDGET_BASIS_KEY` (the key's usage) or
    #: :data:`BUDGET_BASIS_LEDGER` (this runs directory's own ledgers).
    budget_basis: str = BUDGET_BASIS_KEY
    #: Other runs directories' key-basis budget records (``budget.json``) this gateway
    #: also stops on once the key has spent their amount (so it never passes the
    #: experiment's cap); needs ``budget_usd``.
    experiment_budgets: tuple[Path, ...] = ()
    balance_ttl_s: float = 30.0
    #: How long a call that arrives while paused waits for the balance to recover (always
    #: bounded by its own deadline); 0 answers 402 at once.
    pause_hold_s: float = protocol.CALL_DEADLINE_S
    max_concurrency: int = 32
    local_token: str | None = LOCAL_TOKEN
    seed: int = protocol.SEED
    reasoning_effort: str = protocol.REASONING_EFFORT
    #: The provider-window clamp's tokenizer margin (N5, ``protocol.TOKENIZER_MARGIN``).
    tokenizer_margin: Fraction = protocol.TOKENIZER_MARGIN
    #: OpenRouter provider routing per chat model (``protocol.PROVIDER_POLICY``);
    #: another value makes the gateway non-conformant.
    provider_policy: Mapping[str, Mapping[str, Sequence[str]]] = field(
        default_factory=lambda: protocol.PROVIDER_POLICY
    )
    trust_env: bool = True
    transport: httpx.AsyncBaseTransport | None = field(
        default=None, repr=False, compare=False
    )
    #: The lock every ``run.py`` that uses this gateway holds shared for its lifetime
    #: (:func:`gateway_users_lock`); reported in ``/health`` so attached workers find it.
    users_lock: Path | None = None

    def protocol_settings(self) -> dict[str, Any]:
        """The settings that decide what every call sends upstream and how it is judged."""
        return {
            "deadline_s": self.deadline_s,
            "max_retries": self.max_retries,
            "backoff_base_s": self.backoff_base_s,
            "backoff_cap_s": self.backoff_cap_s,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
            "tokenizer_margin": str(Fraction(self.tokenizer_margin)),
            "provider_policy": protocol.provider_policy_settings(self.provider_policy),
        }


#: What the frozen protocol says those settings are.
PROTOCOL_SETTINGS: Mapping[str, Any] = {
    "deadline_s": protocol.CALL_DEADLINE_S,
    "max_retries": protocol.GATEWAY_MAX_RETRIES,
    "backoff_base_s": protocol.GATEWAY_BACKOFF_BASE_S,
    "backoff_cap_s": protocol.GATEWAY_BACKOFF_CAP_S,
    "reasoning_effort": protocol.REASONING_EFFORT,
    "seed": protocol.SEED,
    "tokenizer_margin": str(protocol.TOKENIZER_MARGIN),
    "provider_policy": protocol.provider_policy_settings(),
}


class GatewayStateError(RuntimeError):
    """A run's state file is corrupt and cannot be rebuilt unambiguously from its ledger."""


@dataclass
class _RunState:
    model: str | None = None
    phase: str | None = None
    abandoned: bool = False


class Gateway:
    """State and request handling; :func:`build_app` exposes it over HTTP."""

    def __init__(self, config: GatewayConfig) -> None:
        if not config.api_key:
            raise ValueError("api_key is required")
        if config.budget_usd is not None and not config.budget_usd > 0:
            raise ValueError("budget_usd must be positive")
        if config.budget_basis not in BUDGET_BASES:
            raise ValueError(
                f"budget_basis must be one of {list(BUDGET_BASES)}, not "
                f"{config.budget_basis!r}"
            )
        if config.budget_usd is None and (
            config.budget_basis != BUDGET_BASIS_KEY or config.experiment_budgets
        ):
            raise ValueError(
                "a ledger budget basis or experiment budgets need budget_usd"
            )
        self.config = config
        self.limits = {
            m: protocol.model_limits(
                m, config.models, tokenizer_margin=Fraction(config.tokenizer_margin)
            )
            for m in protocol.CHAT_MODELS
        }
        entries = config.models["models"]
        if entries.get(protocol.EMBEDDING_MODEL, {}).get("kind") != "embedding":
            raise KeyError(f"{protocol.EMBEDDING_MODEL} missing from models.json")
        self.seed_models = frozenset(
            m
            for m in protocol.CHAT_MODELS
            if "seed" in (entries[m].get("supported_parameters") or [])
        )
        # H1.9: the effort is sent to every model; models.json says whether each one lists
        # it (a model may not list "medium"). Not refused, but logged here, in /health
        # and on every call line.
        self.effort_supported: dict[str, bool | None] = {}
        for m in protocol.CHAT_MODELS:
            efforts = (entries[m].get("reasoning") or {}).get("supported_efforts")
            supported = (
                config.reasoning_effort in efforts
                if isinstance(efforts, list)
                else None
            )
            self.effort_supported[m] = supported
            if supported is False:
                log.warning(
                    "reasoning effort %r is not in %s's supported_efforts %s",
                    config.reasoning_effort,
                    m,
                    efforts,
                )
        settings = config.protocol_settings()
        self.settings_sha256 = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.protocol_conformant = settings == dict(PROTOCOL_SETTINGS)
        self.ledger_errors = 0
        self._unlogged: dict[str, list[dict[str, Any]]] = {}
        self._recent: dict[tuple[str, str, str], tuple[str, str, float]] = {}
        self._client: httpx.AsyncClient | None = None
        self._slots: asyncio.Semaphore | None = None
        #: Upstream slots held, and calls queued for one (``/health`` ``slots``).
        self._slots_in_use = 0
        self._slots_waiting = 0
        self._state_lock: asyncio.Lock | None = None
        self._balance_task: asyncio.Task[None] | None = None
        self._ledger_locks: dict[str, asyncio.Lock] = {}
        self._runs: dict[str, _RunState] = {}
        self.balance: dict[str, Any] | None = None
        self.balance_error: str | None = None
        self._balance_at = float("-inf")
        #: The spend budget as last computed (:meth:`_update_budget`); None without one.
        self.budget: dict[str, Any] | None = None
        self._budget_record: dict[str, Any] | None = None
        #: Ledger basis: the ``cost_total`` of every final call line of this runs
        #: directory's ledgers (read at start, then counted as calls finish), the
        #: attempts of unknown cost, the record's starting point once known, and why the
        #: ledgers could not be read (the budget is then unknown, never undercounted).
        self._ledger_spent = 0.0
        self._ledger_unknown = 0
        self._ledger_start: float | None = None
        self._ledger_error: str | None = None
        #: Experiment budget records as last read: path -> ((mtime_ns, size), record).
        self._experiment_cache: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}
        # With a budget, nothing is sent until it is known (the first balance fetch).
        self.paused = config.budget_usd is not None
        self.pause_reason: str | None = BUDGET_UNKNOWN if self.paused else None
        #: This gateway process's identity: stamped on its start record and on every
        #: ledger line it writes, so the audit matches a call to the gateway that served
        #: it, not to whichever gateway started last (X34).
        self.start_id = uuid.uuid4().hex[:16]
        self.app = build_app(self)

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        cfg = self.config
        self._client = httpx.AsyncClient(
            base_url=cfg.upstream.rstrip("/"),
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "X-Title": "mirix-skill-bench",
            },
            timeout=httpx.Timeout(cfg.deadline_s, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(
                max_connections=cfg.max_concurrency + 4,
                max_keepalive_connections=cfg.max_concurrency,
            ),
            trust_env=cfg.trust_env,
            transport=cfg.transport,
        )
        self._slots = asyncio.Semaphore(cfg.max_concurrency)
        self._state_lock = asyncio.Lock()
        cfg.runs_dir.mkdir(parents=True, exist_ok=True)
        encoding = await tokens.preload_encoding()
        if encoding != protocol.TOKEN_ENCODING:
            self.protocol_conformant = False
        record = await asyncio.to_thread(self.start_record)  # hashes the protocol tree
        await asyncio.to_thread(
            append_jsonl, cfg.runs_dir / "_gateway" / "starts.jsonl", record
        )
        if cfg.budget_basis == BUDGET_BASIS_LEDGER:
            await self._scan_ledgers()
        if cfg.budget_usd is not None:
            # Establish the budget before /health reports ready: a cell that checks
            # /health at its start must not find a gateway paused only for not knowing yet.
            await self.refresh_balance(force=True)
        log.info("gateway ready: encoding=%s upstream=%s", encoding, cfg.upstream)

    def start_record(self) -> dict[str, Any]:
        """Everything about this gateway that shapes calls (never the key)."""
        cfg = self.config
        return {
            "event": "gateway_start",
            "ts": _now(),
            "pid": os.getpid(),
            "start_id": self.start_id,
            "users_lock": str(cfg.users_lock) if cfg.users_lock else None,
            "upstream": cfg.upstream,
            "settings": cfg.protocol_settings(),
            "settings_sha256": self.settings_sha256,
            "protocol_settings": dict(PROTOCOL_SETTINGS),
            "protocol_conformant": self.protocol_conformant,
            "protocol_fingerprint": protocol.protocol_fingerprint(),
            "encoding": tokens.encoding_name(),
            "token_estimate": dict(protocol.TOKEN_ESTIMATE),
            "max_concurrency": cfg.max_concurrency,
            "balance_floor_usd": cfg.balance_floor_usd,
            "budget_usd": cfg.budget_usd,
            "budget_basis": cfg.budget_basis,
            "experiment_budgets": [str(p) for p in cfg.experiment_budgets],
            "pause_hold_s": cfg.pause_hold_s,
            "effort_supported": dict(self.effort_supported),
            "limits": {m: lim.f9() for m, lim in self.limits.items()},
        }

    async def aclose(self) -> None:
        task, self._balance_task = self._balance_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("gateway not started")
        return self._client

    # -- run state, ledger ---------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self.config.runs_dir / run_id

    async def ledger(self, run_id: str, record: Mapping[str, Any]) -> int:
        """Append one fsync'd ledger line; earlier lines that could not be written go first.

        Every line carries this gateway's ``gateway_start`` id (X34). Returns the byte
        offset where ``record``'s line starts.
        """
        lock = self._ledger_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            pending = self._unlogged.get(run_id) or []
            lines = [
                line
                if "gateway_start" in line
                else {**line, "gateway_start": self.start_id}
                for line in (*pending, record)
            ]
            offset = await asyncio.to_thread(
                _append_all, self._run_dir(run_id) / "gateway.jsonl", lines
            )
            if pending:
                log.warning(
                    "wrote %d delayed ledger line(s) of %s", len(pending), run_id
                )
                self._unlogged.pop(run_id, None)
            return offset

    async def ledger_final(self, run_id: str, record: dict[str, Any]) -> None:
        """Write a call's final line; a failed write never costs the (billed) response.

        The line stays in memory and is written before the run's next ledger line. Until
        then /health reports it, and the next call of the run fails at its own
        ``call_started`` line (500, INFRA) if the disk is still unusable, so a broken
        ledger stops the run instead of losing the record of paid calls.
        """
        try:
            await self.ledger(run_id, record)
        except Exception as exc:
            self.ledger_errors += 1
            self._unlogged.setdefault(run_id, []).append(record)
            log.error(
                "ledger write failed for %s call %s (%s); line kept in memory",
                run_id,
                record.get("call_id"),
                type(exc).__name__,
            )

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "gateway_state.json"

    def _load_state(self, run_id: str) -> _RunState:
        path = self._state_path(run_id)
        if not path.exists():
            return _RunState()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state is not a JSON object")
            return _RunState(
                model=data.get("model"),
                phase=data.get("phase"),
                abandoned=bool(data.get("abandoned", False)),
            )
        except ValueError as exc:  # torn write after a crash: rebuild from the ledger
            state = self._state_from_ledger(run_id)
            log.warning("rebuilt corrupt %s from the ledger (%s)", path, exc)
            self._save_state(run_id, state)
            append_jsonl(
                self._run_dir(run_id) / "gateway.jsonl",
                {
                    "event": "state_rebuilt",
                    "ts": _now(),
                    "run_id": run_id,
                    "model": state.model,
                    "phase": state.phase,
                    "abandoned": state.abandoned,
                },
            )
            return state

    def _state_from_ledger(self, run_id: str) -> _RunState:
        """The pinned model, phase and abandonment as the ledger recorded them.

        The model is the one of every chat ``call_started`` line (a line is only written
        after the model was pinned); more than one model means the state cannot be
        trusted, and the run is refused (fail closed).
        """
        path = self._run_dir(run_id) / "gateway.jsonl"
        models: set[str] = set()
        phase: str | None = None
        abandoned = False
        if path.exists():
            # "\n" only: a record may hold U+2028 (ensure_ascii=False).
            for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a torn line (X15)
                if not isinstance(row, dict):
                    continue  # a line that parses but is no record
                event = row.get("event")
                if event == "call_started" and row.get("endpoint") == CHAT:
                    models.add(str(row.get("model")))
                elif event == "phase":
                    phase = row.get("phase")
                elif event == "abandoned":
                    abandoned = True
        if len(models) > 1:
            raise GatewayStateError(
                f"run {run_id}: state file corrupt and the ledger names several models {sorted(models)}"
            )
        return _RunState(
            model=next(iter(models), None), phase=phase, abandoned=abandoned
        )

    def _save_state(self, run_id: str, state: _RunState) -> None:
        """Atomic, durable replace: fsync the temporary file, rename, fsync the directory."""
        path = self._state_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = {
            "model": state.model,
            "phase": state.phase,
            "abandoned": state.abandoned,
        }
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_dir(path.parent)

    async def _state(self, run_id: str) -> _RunState:
        assert self._state_lock is not None
        async with self._state_lock:
            state = self._runs.get(run_id)
            if state is None:
                state = await asyncio.to_thread(self._load_state, run_id)
                self._runs[run_id] = state
            return state

    async def bind_model(self, run_id: str, model: str) -> str | None:
        """Pin the run to its first chat model (H1.8); return the pinned model on mismatch."""
        state = await self._state(run_id)
        assert self._state_lock is not None
        async with self._state_lock:
            if state.model is None:
                state.model = model
                await asyncio.to_thread(self._save_state, run_id, state)
            return None if state.model == model else state.model

    async def set_phase(self, run_id: str, phase: str) -> None:
        state = await self._state(run_id)
        assert self._state_lock is not None
        async with self._state_lock:
            state.phase = phase
            await asyncio.to_thread(self._save_state, run_id, state)
        await self.ledger(
            run_id, {"event": "phase", "ts": _now(), "run_id": run_id, "phase": phase}
        )

    async def abandon(self, run_id: str, reason: str = "") -> None:
        """Refuse every later call of ``run_id`` (410): a dead attempt must not keep spending."""
        state = await self._state(run_id)
        assert self._state_lock is not None
        async with self._state_lock:
            state.abandoned = True
            await asyncio.to_thread(self._save_state, run_id, state)
        await self.ledger(
            run_id,
            {
                "event": "abandoned",
                "ts": _now(),
                "run_id": run_id,
                "reason": reason[:500],
            },
        )

    async def mark(self, run_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """Append a ``checkpoint`` or ``resume`` marker line to ``run_id``'s ledger (the
        body already passed :func:`mark_problem`); returns ``{"ok", "offset", "ts"}``,
        ``offset`` being where the line starts. Raises :class:`GatewayReject` (410) for an
        abandoned run: a dead attempt's markers would describe nothing."""
        if (await self._state(run_id)).abandoned:
            raise GatewayReject(
                410, "run_abandoned", f"run {run_id} was abandoned: nothing to mark"
            )
        ts = _now()
        record = {**body, "event": str(body["event"]), "ts": ts, "run_id": run_id}
        offset = await self.ledger(run_id, record)
        return {"ok": True, "offset": offset, "ts": ts}

    # -- balance ---------------------------------------------------------------

    def _pause(self, reason: str) -> None:
        if not self.paused:
            log.warning("gateway paused: %s", reason)
        self.paused, self.pause_reason = True, reason

    async def refresh_balance(
        self, force: bool = False, *, timeout_s: float | None = None
    ) -> None:
        """Refresh the cached balance, one fetch at a time (single flight, N3).

        A caller that arrives while a fetch is in flight waits for that fetch instead of
        starting another, forced or not. Otherwise a fetch starts only when ``force`` is
        set or the cache is older than ``balance_ttl_s``. Calls held by a pause share the
        same fetches, so N held calls cost one ``/key`` request per TTL, not N.
        ``timeout_s`` bounds this caller's wait (the fetch itself goes on for the others).
        """
        task = self._balance_task
        if task is None or task.done():
            fresh = time.monotonic() - self._balance_at < self.config.balance_ttl_s
            if fresh and not force:
                return
            task = self._balance_task = asyncio.create_task(self._fetch_balance())
        # One waiter's cancellation (its own deadline) must not cancel the shared fetch.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout_s)
        except asyncio.TimeoutError:
            return

    async def _fetch_balance(self) -> None:
        cfg = self.config
        self._balance_at = time.monotonic()
        try:
            response = await self.client.get("/key", timeout=BALANCE_TIMEOUT_S)
            response.raise_for_status()
            data = response.json()["data"]
        except (
            Exception
        ) as exc:  # keep the last known balance; the upstream 402 still pauses
            self.balance_error = type(exc).__name__
            log.warning("balance check failed: %s", self.balance_error)
            return
        self.balance_error = None
        self.balance = {
            k: data.get(k)
            for k in (
                "limit",
                "limit_remaining",
                "limit_reset",
                "usage",
                "usage_daily",
                "is_free_tier",
            )
        }
        remaining = self.balance.get("limit_remaining")
        source = "key"
        if remaining is None:  # key without its own limit: fall back to account credits
            remaining, source = await self._account_credit(), "credits"
        self.balance["remaining_usd"] = remaining
        self.balance["remaining_source"] = source if remaining is not None else None
        known = isinstance(remaining, (int, float))
        budget_problem = await self._update_budget(data)
        if budget_problem in STOP_PAUSE_REASONS:
            # A spent or invalid budget needs a person whatever the balance: checked
            # first, or the key running below the floor as the budget is used up (a key
            # may hold just the floor plus the budget) would read as the waitable floor,
            # and the supervisor would never stop and tell a person.
            self._pause(budget_problem)
        elif known and remaining < cfg.balance_floor_usd:
            self._pause("balance_below_floor")
        elif budget_problem is not None:
            self._pause(budget_problem)
        elif self.paused and (known or self.pause_reason in BUDGET_PAUSE_REASONS):
            log.warning(
                "gateway resumed (limit_remaining=%s, budget left %s)",
                remaining,
                (self.budget or {}).get("left_usd"),
            )
            self.paused, self.pause_reason = False, None
        # An unknown balance never lifts a pause (an upstream 402 stays in force); a
        # budget pause lifts once the budget is known and left.
        record = {
            "ts": _now(),
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "floor_usd": cfg.balance_floor_usd,
            **self.balance,
        }
        if cfg.budget_usd is not None:
            record["budget"] = self.budget
        await asyncio.to_thread(
            append_jsonl, cfg.runs_dir / "_gateway" / "balance.jsonl", record
        )

    async def _update_budget(self, data: Mapping[str, Any]) -> str | None:
        """Recompute the spend budget from the key info ``data``: None when there is no
        budget or some of it is left, else the pause reason (``budget_spent``,
        ``budget_unknown``, ``budget_invalid``). Sets :attr:`budget`: its own amount on
        its basis, and each experiment cap it enforces (``experiment``); a stop reason
        of either wins over a waitable one."""
        cfg = self.config
        if cfg.budget_usd is None:
            return None
        if cfg.budget_basis == BUDGET_BASIS_LEDGER:
            problem = await self._update_ledger_budget()
        else:
            problem = await self._update_key_budget(data)
        if cfg.experiment_budgets:
            caps, cap_problem = await self._experiment_caps(data)
            assert self.budget is not None
            self.budget["experiment"] = caps
            problem = _worse_pause(problem, cap_problem)
        return problem

    async def _update_key_budget(self, data: Mapping[str, Any]) -> str | None:
        """The key basis: the key's all-time ``usage`` since the record's start."""
        cfg = self.config
        assert cfg.budget_usd is not None
        path = cfg.runs_dir / "_gateway" / BUDGET_FILE
        view: dict[str, Any] = {
            "budget_usd": cfg.budget_usd,
            "file": str(path),
            "basis": BUDGET_BASIS_KEY,
        }
        usage = data.get("usage")
        if isinstance(usage, bool) or not isinstance(usage, (int, float)):
            self.budget = {**view, "error": "the key info carries no usage"}
            return BUDGET_UNKNOWN
        try:
            record = await asyncio.to_thread(self._budget_start, path, float(usage))
        except (OSError, ValueError, TypeError) as exc:
            self.budget = {
                **view,
                "usage_usd": float(usage),
                "error": f"{type(exc).__name__}: {exc} (fix or remove {path})"[:500],
            }
            return BUDGET_UNKNOWN
        start = float(record["start_usage_usd"])
        spent = float(usage) - start
        self.budget = {
            **view,
            "start_usage_usd": start,
            "started_at": record.get("started_at"),
            "usage_usd": float(usage),
            "spent_usd": round(spent, 6),
            "left_usd": round(cfg.budget_usd - spent, 6),
        }
        if spent < -BUDGET_USAGE_TOLERANCE_USD:
            self.budget["error"] = (
                f"the key's usage ${float(usage):.2f} is below the budget's starting "
                f"point ${start:.2f}: another key? Remove {path} to start a new budget"
            )
            return BUDGET_INVALID
        return BUDGET_SPENT if spent >= cfg.budget_usd else None

    async def _update_ledger_budget(self) -> str | None:
        """The ledger basis: the ``cost_total`` of this runs directory's own ledgers since
        the record's start (``_gateway/budget_ledger.json``, created at the ledgers' total
        the first time, kept across restarts; a later ``--budget`` replaces the amount,
        not the start)."""
        cfg = self.config
        assert cfg.budget_usd is not None
        path = cfg.runs_dir / "_gateway" / BUDGET_LEDGER_FILE
        view: dict[str, Any] = {
            "budget_usd": cfg.budget_usd,
            "file": str(path),
            "basis": BUDGET_BASIS_LEDGER,
        }
        # Read the ledgers again at every balance check: another gateway keeping its
        # ledgers in this runs directory (a second grader on another port) spends the
        # same budget, and a ledger that could not be read is tried again.
        await self._scan_ledgers()
        if self._ledger_error is not None:
            self.budget = {**view, "error": self._ledger_error}
            return BUDGET_UNKNOWN
        try:
            record = await asyncio.to_thread(
                self._budget_start,
                path,
                self._ledger_spent,
                start_field="start_ledger_usd",
                value_field="ledger_usd",
            )
        except (OSError, ValueError, TypeError) as exc:
            self.budget = {
                **view,
                "ledger_usd": round(self._ledger_spent, 10),
                "error": f"{type(exc).__name__}: {exc} (fix or remove {path})"[:500],
            }
            return BUDGET_UNKNOWN
        self._ledger_start = float(record["start_ledger_usd"])
        self.budget = {
            **view,
            "started_at": record.get("started_at"),
            **self._ledger_view(),
        }
        spent = self.budget["spent_usd"]
        if spent < -BUDGET_USAGE_TOLERANCE_USD:
            self.budget["error"] = (
                f"the ledgers of {cfg.runs_dir} hold ${self._ledger_spent:.4f}, less than "
                f"the budget's starting point ${self._ledger_start:.4f}: ledgers removed? "
                f"Remove {path} to start a new budget"
            )
            return BUDGET_INVALID
        return BUDGET_SPENT if spent >= cfg.budget_usd else None

    def _ledger_view(self) -> dict[str, Any]:
        """The ledger budget's counts (after its start is known)."""
        assert self._ledger_start is not None and self.config.budget_usd is not None
        spent = self._ledger_spent - self._ledger_start
        return {
            "start_ledger_usd": self._ledger_start,
            "ledger_usd": round(self._ledger_spent, 10),
            "spent_usd": round(spent, 6),
            "left_usd": round(self.config.budget_usd - spent, 6),
            "unknown_cost_attempts": self._ledger_unknown,
        }

    async def _scan_ledgers(self) -> None:
        """Ledger basis: read every ledger of this runs directory (at start and at each
        balance check). The count is the larger of what the ledgers hold and what this
        gateway counted itself (its calls count as they finish, before their line is
        written): each is at most the true spend, so nothing is counted twice, and the
        calls of another gateway on this runs directory are seen at the next check."""
        try:
            spent, unknown = await asyncio.to_thread(ledger_spend, self.config.runs_dir)
        except OSError as exc:
            self._ledger_error = (
                f"cannot read the ledgers of {self.config.runs_dir}: "
                f"{type(exc).__name__}: {exc}"
            )[:500]
            log.warning("%s", self._ledger_error)
            return
        self._ledger_spent = max(self._ledger_spent, spent)
        self._ledger_unknown = max(self._ledger_unknown, unknown)
        self._ledger_error = None

    def _count_ledger_call(self, record: Mapping[str, Any]) -> None:
        """Ledger basis: count a finished call, and pause at once when it spent the
        budget (the next balance check would, a little later)."""
        cost = record.get("cost_total")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self._ledger_spent += float(cost)
        unknown = record.get("unknown_cost_attempts")
        if isinstance(unknown, int) and not isinstance(unknown, bool):
            self._ledger_unknown += unknown
        if self._ledger_start is None or self.config.budget_usd is None:
            return
        view = self._ledger_view()
        if self.budget is not None:
            self.budget.update(view)
        if view["spent_usd"] >= self.config.budget_usd:
            self._pause(BUDGET_SPENT)

    async def _experiment_caps(
        self, data: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Each experiment budget record (``budget.json``, key basis) against the key's
        ``usage``: its view, and the pause it asks for. Spent (``usage - start_usage_usd
        >= budgets[-1].budget_usd``): ``budget_spent``; an unreadable record or no usage:
        ``budget_unknown`` (waitable; the cap is never guessed); a usage below its start
        (another key than this gateway's): recorded, not checked, no pause."""
        usage = _usd(data.get("usage"))
        caps: list[dict[str, Any]] = []
        problem: str | None = None
        for path in self.config.experiment_budgets:
            entry: dict[str, Any] = {"file": str(path)}
            caps.append(entry)
            try:
                record = await asyncio.to_thread(self._experiment_record, path)
                start = _usd(record.get("start_usage_usd"))
                budgets = [
                    b for b in record.get("budgets") or () if isinstance(b, dict)
                ]
                amount = _usd(budgets[-1].get("budget_usd")) if budgets else None
                if start is None:
                    raise ValueError("no start_usage_usd")
                if amount is None:
                    raise ValueError("no budgets[-1].budget_usd")
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
                problem = _worse_pause(problem, BUDGET_UNKNOWN)
                continue
            entry.update(start_usage_usd=start, budget_usd=amount)
            if usage is None:
                entry["error"] = "the key info carries no usage"
                problem = _worse_pause(problem, BUDGET_UNKNOWN)
                continue
            spent = usage - start
            entry["spent_usd"] = round(spent, 6)
            if spent < -BUDGET_USAGE_TOLERANCE_USD:
                entry["left_usd"] = None
                entry["checked"] = False
                entry["error"] = (
                    f"the key's usage ${usage:.2f} is below this record's start "
                    f"${start:.2f}: another key than the experiment's; not checked"
                )
                continue
            entry["left_usd"] = round(amount - spent, 6)
            entry["checked"] = True
            if spent >= amount:
                problem = _worse_pause(problem, BUDGET_SPENT)
        return caps, problem

    def _experiment_record(self, path: Path) -> dict[str, Any]:
        """An experiment budget record, read again only when it changed (blocking)."""
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._experiment_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path} is not a JSON object")
        self._experiment_cache[path] = (key, record)
        return record

    def _budget_start(
        self,
        path: Path,
        value: float,
        *,
        start_field: str = "start_usage_usd",
        value_field: str = "usage_usd",
    ) -> dict[str, Any]:
        """The budget record of this runs directory, created at ``value`` (the key's usage,
        or the ledgers' total on the ledger basis) the first time (blocking). Every change
        of the amount is appended to its ``budgets``."""
        if self._budget_record is not None:
            return self._budget_record
        record: dict[str, Any]
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            start = loaded.get(start_field) if isinstance(loaded, dict) else None
            if isinstance(start, bool) or not isinstance(start, (int, float)):
                raise ValueError(f"{path} has no {start_field}")
            record = loaded
        else:
            record = {start_field: value, "started_at": _now(), "budgets": []}
        budgets = [b for b in record.get("budgets") or () if isinstance(b, dict)]
        if not budgets or budgets[-1].get("budget_usd") != self.config.budget_usd:
            budgets.append(
                {
                    "budget_usd": self.config.budget_usd,
                    "set_at": _now(),
                    value_field: value,
                    "gateway_start_id": self.start_id,
                }
            )
            record["budgets"] = budgets
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            tmp.replace(path)
        self._budget_record = record
        return record

    async def _account_credit(self) -> float | None:
        """Remaining account credit (``total_credits - total_usage`` from ``/credits``), or None."""
        try:
            response = await self.client.get("/credits", timeout=BALANCE_TIMEOUT_S)
            response.raise_for_status()
            data = response.json()["data"]
            return float(data["total_credits"]) - float(data["total_usage"])
        except Exception as exc:
            self.balance_error = f"credits:{type(exc).__name__}"
            log.warning("credit check failed: %s", self.balance_error)
            return None

    async def check_paused(self) -> bool:
        await self.refresh_balance()
        return self.paused

    async def wait_while_paused(self, deadline: float) -> bool:
        """Hold a call while the gateway is paused; True if it is still paused at the end.

        The call waits at most ``pause_hold_s`` and never past its own deadline, checking
        the balance every ``balance_ttl_s``: a short dip below the floor
        (topped up meanwhile) no longer invalidates every running learning cell (H8).
        The checks respect the balance cache and share one in-flight fetch
        (:meth:`refresh_balance`), so however many calls are held, the gateway asks
        OpenRouter about once per ``balance_ttl_s``.
        """
        if not await self.check_paused():
            return False
        cfg = self.config
        hold_until = min(deadline, time.monotonic() + cfg.pause_hold_s)
        poll = max(cfg.balance_ttl_s, 0.05)
        while self.paused:
            left = hold_until - time.monotonic()
            if left <= 0:
                return True
            await asyncio.sleep(min(poll, left))
            await self.refresh_balance(
                timeout_s=max(0.0, hold_until - time.monotonic())
            )
        return False

    # -- upstream -----------------------------------------------------------------

    def _backoff(self, retries: int) -> float:
        return min(self.config.backoff_cap_s, self.config.backoff_base_s * (2**retries))

    async def forward(self, prepared: PreparedRequest, deadline: float) -> Outcome:
        """Send with retries on infrastructure failures, all inside ``deadline`` (monotonic).

        Every attempt records the deadline it started with (``budget_s``) and, when the
        upstream returned a body, that body's usage, cost, provider and generation id:
        a retried attempt may have been billed. An answer from a provider the call's
        routing ignores is such a retried attempt (:func:`refuse_ignored_provider`).
        """
        attempts: list[dict[str, Any]] = []
        retries = 0
        headers = {"Content-Type": "application/json"}
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._deadline(retries, attempts, "deadline_before_attempt")
            started = time.monotonic()
            attempt: dict[str, Any] = {
                "n": len(attempts) + 1,
                "budget_s": round(remaining, 3),
            }
            attempts.append(attempt)
            retry_after: float | None = None
            # Whether the request body went out (httpcore's trace): None while the
            # transport says nothing (then an attempt cut off may have been billed).
            progress: dict[str, bool | None] = {"sent": None}

            async def trace(
                name: str, info: Mapping[str, Any], progress=progress
            ) -> None:
                if name.endswith("send_request_body.complete"):
                    progress["sent"] = True
                elif progress["sent"] is None:
                    progress["sent"] = False

            try:
                response = await asyncio.wait_for(
                    self.client.post(
                        "/" + prepared.endpoint,
                        content=prepared.content,
                        headers=headers,
                        timeout=httpx.Timeout(
                            remaining, connect=min(CONNECT_TIMEOUT_S, remaining)
                        ),
                        extensions={"trace": trace},
                    ),
                    timeout=remaining + DEADLINE_GRACE_S,
                )
            except asyncio.TimeoutError:
                attempt.update(
                    outcome="deadline", latency_s=round(time.monotonic() - started, 3)
                )
                # Cut off by the deadline itself: billed unless the trace shows the
                # request never went out.
                if progress["sent"] is not False:
                    attempt["billed"] = BILLED_UNKNOWN
                return self._deadline(retries, attempts, "deadline_during_attempt")
            except httpx.ConnectTimeout:
                failure, last_status, last_error = "connect_timeout", 502, None
            except (
                httpx.TimeoutException
            ) as exc:  # read/write/pool timeout == remaining deadline
                attempt.update(
                    outcome="deadline", latency_s=round(time.monotonic() - started, 3)
                )
                attempt["timeout"] = type(exc).__name__
                # A request that never fully reached the upstream (no pooled connection,
                # a body cut off mid-write) cannot have been billed.
                if not isinstance(exc, (httpx.PoolTimeout, httpx.WriteTimeout)):
                    attempt["billed"] = BILLED_UNKNOWN
                return self._deadline(retries, attempts, "deadline_during_attempt")
            except httpx.TransportError as exc:
                failure, last_status, last_error = type(exc).__name__, 502, None
                # The request went out and its answer was lost (the connection dropped
                # mid-answer): OpenRouter may bill it like a deadline-cut attempt, and
                # the retry pays again. A connect or write error never
                # delivered it.
                if progress["sent"] is True or (
                    progress["sent"] is None
                    and isinstance(exc, (httpx.ReadError, httpx.RemoteProtocolError))
                ):
                    attempt["billed"] = BILLED_UNKNOWN
            else:
                attempt["latency_s"] = round(time.monotonic() - started, 3)
                verdict = refuse_ignored_provider(
                    judge_response(prepared.endpoint, response),
                    prepared.body.get("provider"),
                )
                attempt["outcome"] = verdict.failure or "ok"
                attempt["http_status"] = response.status_code
                attempt.update(_billing(verdict.data))
                if verdict.kind != "ok" and verdict.error is not None:
                    attempt["error"] = verdict.error  # e.g. native_finish_reason
                if verdict.kind == "ok":
                    return Outcome(
                        "ok", 200, verdict.data, response.content, retries, attempts
                    )
                if verdict.kind == "paused":
                    self._pause("upstream_402")
                    return Outcome(
                        "budget_paused",
                        402,
                        verdict.data,
                        None,
                        retries,
                        attempts,
                        _error_body(
                            402,
                            "budget_paused",
                            "OpenRouter reports insufficient credit",
                            source="upstream",
                        ),
                    )
                if verdict.kind == "fail":
                    message = (verdict.error or {}).get(
                        "message"
                    ) or "upstream request failed"
                    return Outcome(
                        "upstream_error",
                        verdict.status,
                        verdict.data,
                        None,
                        retries,
                        attempts,
                        _error_body(
                            verdict.status,
                            "upstream_error",
                            message,
                            source="upstream",
                            upstream=verdict.error,
                        ),
                    )
                failure, last_status, last_error = (
                    verdict.failure,
                    verdict.status,
                    verdict.error,
                )
                retry_after = verdict.retry_after
            attempt.setdefault("outcome", failure)
            attempt.setdefault("latency_s", round(time.monotonic() - started, 3))
            if retries >= self.config.max_retries:
                message = f"infrastructure failure persisted after {retries} retries (last: {failure})"
                status = last_status if last_status in RETRYABLE_STATUS else 502
                body = _error_body(
                    status,
                    "upstream_infra_failure",
                    message,
                    retries=retries,
                    upstream=last_error,
                )
                return Outcome(
                    "infra_error", status, None, None, retries, attempts, body
                )
            wait = retry_after if retry_after is not None else self._backoff(retries)
            if time.monotonic() + wait >= deadline:
                reason = (
                    "retry_after_exceeds_deadline"
                    if retry_after is not None
                    else "backoff_exceeds_deadline"
                )
                return self._deadline(retries, attempts, reason)
            attempt["wait_s"] = round(wait, 3)
            await asyncio.sleep(wait)
            retries += 1

    def _deadline(
        self, retries: int, attempts: list[dict[str, Any]], reason: str
    ) -> Outcome:
        """A 504 whose body lets ``infra/failures.py`` tell model time from gateway time.

        ``attempt_budget_s`` is the deadline left when the last attempt started (None if
        no attempt started); ``relay`` adds ``queue_wait_s`` and ``pre_attempt_s``.
        """
        message = f"call exceeded the {self.config.deadline_s:.0f}s deadline ({reason})"
        body = _error_body(
            504,
            "deadline_exceeded",
            message,
            reason=reason,
            retries=retries,
            upstream_attempts=len(attempts),
            deadline_s=self.config.deadline_s,
            attempt_budget_s=attempts[-1].get("budget_s") if attempts else None,
        )
        return Outcome("deadline_exceeded", 504, None, None, retries, attempts, body)

    # -- HTTP handler -------------------------------------------------------------------

    async def relay(
        self, request: Request, run_id: str, role: str, endpoint: str
    ) -> Response:
        arrived = time.monotonic()
        deadline = arrived + self.config.deadline_s
        call_id = uuid.uuid4().hex
        if not RUN_ID_RE.fullmatch(run_id):
            return JSONResponse(
                _error_body(
                    400, "invalid_run_id", "run_id must match " + RUN_ID_RE.pattern
                ),
                400,
            )
        item = request.headers.get("x-skill-bench-item")
        phase_header = request.headers.get("x-skill-bench-phase")
        record: dict[str, Any] = {
            "event": "call",
            "ts": _now(),
            "call_id": call_id,
            "run_id": run_id,
            "role": role,
            "endpoint": endpoint,
            "item": item,
            "gateway_settings_sha256": self.settings_sha256,
            "protocol_conformant": self.protocol_conformant,
        }

        async def reject(
            status: int, kind: str, message: str, **extra: Any
        ) -> Response:
            body = _error_body(status, kind, message, **extra)
            _with_blame(body["error"], status, role)
            state = await self._state(run_id)
            record.update(
                status="rejected",
                http_status=status,
                phase=phase_header or state.phase,
                run_phase=state.phase,
                phase_header=phase_header,
                error=body["error"],
                latency_s=round(time.monotonic() - arrived, 3),
                completed_at=_now(),
            )
            await self.ledger(run_id, record)
            return JSONResponse(
                body, status, headers={"x-skill-bench-call-id": call_id}
            )

        if endpoint not in ENDPOINTS:
            return await reject(
                404, "unsupported_endpoint", f"endpoint must be one of {ENDPOINTS}"
            )
        if role not in protocol.ROLES:
            return await reject(
                400, "invalid_role", f"role must be one of {protocol.ROLES}"
            )
        if (role == "embedding") != (endpoint == EMBEDDINGS):
            return await reject(
                400,
                "role_endpoint_mismatch",
                "role 'embedding' is used exactly for /embeddings",
            )
        token = self.config.local_token
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return await reject(
                401, "unauthorized", "use the local skill-bench credential"
            )
        if item is not None and not ITEM_RE.fullmatch(item):
            return await reject(
                400, "invalid_item", "x-skill-bench-item must match " + ITEM_RE.pattern
            )
        if phase_header is not None and phase_header not in PHASES:
            return await reject(
                400, "invalid_phase", f"x-skill-bench-phase must be one of {PHASES}"
            )
        if (await self._state(run_id)).abandoned:
            return await reject(
                410,
                "run_abandoned",
                f"run {run_id} was abandoned (its learning attempt was restarted)",
            )
        run_phase = (await self._state(run_id)).phase
        if phase_header is not None and run_phase is not None:
            if phase_header != run_phase:
                # X25: a caller's own phase tag never overrides the run's phase. A learner
                # or judge call tagged "learning" while run.py has put the run in its
                # test phase is refused (and the audit reports the refusal).
                return await reject(
                    409,
                    "phase_mismatch",
                    f"x-skill-bench-phase is {phase_header!r} but run {run_id} is in "
                    f"its {run_phase!r} phase",
                )

        raw = await request.body()
        kwargs = {
            "limits": self.limits,
            "seed_models": self.seed_models,
            "seed": self.config.seed,
            "effort": self.config.reasoning_effort,
            "provider_policy": self.config.provider_policy,
        }
        try:
            if len(raw) > protocol.LARGE_PAYLOAD_CHARS:
                prepared = await asyncio.to_thread(
                    prepare_request, raw, endpoint, **kwargs
                )
            else:
                prepared = prepare_request(raw, endpoint, **kwargs)
        except GatewayReject as exc:
            return await reject(exc.status, exc.kind, exc.message, **exc.extra)

        record.update(
            model=prepared.model,
            model_requested=prepared.model_requested,
            stream_requested=prepared.stream_requested,
            max_tokens_requested=prepared.max_tokens_requested,
            max_tokens_rewritten=prepared.max_tokens_rewritten,
            max_tokens_f9=prepared.max_tokens_f9,
            max_tokens_provider=prepared.max_tokens_provider,
            max_tokens_bound=prepared.max_tokens_bound,
            tokenizer_margin=str(Fraction(self.config.tokenizer_margin)),
            dropped_fields=list(prepared.dropped_fields),
            reasoning_effort_supported=self.effort_supported.get(prepared.model),
            input_tokens_estimated=prepared.estimate.total,
            input_estimate=prepared.estimate.to_dict(),
            request_sha256=prepared.sha256,
            request_bytes=len(prepared.content),
            wire={
                k: prepared.body[k]
                for k in (
                    "model",
                    "stream",
                    "reasoning",
                    "max_tokens",
                    "seed",
                    "usage",
                    "transforms",
                    "provider",
                )
                if k in prepared.body
            },
        )
        repeat = self._recent.get((run_id, role, prepared.sha256))
        if repeat is not None and time.monotonic() - repeat[2] <= REPEAT_WINDOW_S:
            # The same request again after a failed one: an M3.5 application-level retry
            # (mirix / embedding roles), an H8 test-phase item re-run (actor role), or a
            # retry nothing should make (learner / judge). Flag it; the audit decides by role.
            record.update(repeat_of=repeat[0], repeat_after_status=repeat[1])
        if endpoint == CHAT:
            pinned = await self.bind_model(run_id, prepared.model)
            if pinned is not None:
                return await reject(
                    403,
                    "model_mismatch_for_run",
                    f"run {run_id} is pinned to {pinned}",
                    pinned=pinned,
                )
        # ``phase`` is the caller's tag or, without one, the run's phase; ``run_phase`` is
        # what run.py set (POST /runs/<id>/phase) when the call was admitted (the value the
        # phase check above saw) and ``phase_header`` the caller's tag, so the audit never
        # has to trust a self-declared phase (X6, X25).
        record["phase"] = phase_header or run_phase
        record["run_phase"] = run_phase
        record["phase_header"] = phase_header

        # Every admitted call has its ``call_started`` line before it can be held by a
        # pause, so a call the pause answers 402 is placed in the ledger like any other:
        # quiescence waits for it, and a learning resume's discarded segment holds it
        # (a held call without the line could never be discarded).
        await self.ledger(
            run_id,
            {
                "event": "call_started",
                "ts": _now(),
                "call_id": call_id,
                "run_id": run_id,
                "role": role,
                "phase": record["phase"],
                "run_phase": record["run_phase"],
                "item": item,
                "endpoint": endpoint,
                "model": prepared.model,
                "request_sha256": prepared.sha256,
                "input_tokens_estimated": prepared.estimate.total,
                "max_tokens_rewritten": prepared.max_tokens_rewritten,
            },
        )
        paused_at = time.monotonic()
        if await self.wait_while_paused(deadline):
            outcome = Outcome(
                "budget_paused",
                402,
                error=_error_body(
                    402,
                    "budget_paused",
                    f"gateway paused ({self.pause_reason}); preserve checkpoints",
                    held_s=round(time.monotonic() - paused_at, 3),
                ),
            )
            return await self._finish(
                record, prepared, outcome, arrived, queue_wait=0.0, call_id=call_id
            )
        held = time.monotonic() - paused_at
        if held >= 0.001:
            record["balance_wait_s"] = round(held, 3)

        assert self._slots is not None
        queued = time.monotonic()
        self._slots_waiting += 1
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=max(0.0, deadline - queued)
            )
            timed_out = False
        except asyncio.TimeoutError:
            timed_out = True
        finally:
            self._slots_waiting -= 1
        if timed_out:
            outcome = self._deadline(0, [], "deadline_while_queued")
            return await self._finish(
                record, prepared, outcome, arrived, time.monotonic() - queued, call_id
            )
        queue_wait = time.monotonic() - queued
        self._slots_in_use += 1
        try:
            outcome = await self.forward(prepared, deadline)
        finally:
            self._slots_in_use -= 1
            self._slots.release()
        return await self._finish(
            record, prepared, outcome, arrived, queue_wait, call_id
        )

    async def _finish(
        self,
        record: dict[str, Any],
        prepared: PreparedRequest,
        outcome: Outcome,
        arrived: float,
        queue_wait: float,
        call_id: str,
    ) -> Response:
        data = outcome.data if isinstance(outcome.data, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        choices = data.get("choices") if isinstance(data.get("choices"), list) else []
        if outcome.error and outcome.status == "deadline_exceeded":
            error = outcome.error["error"]
            first = outcome.attempts[0].get("budget_s") if outcome.attempts else None
            error["queue_wait_s"] = round(queue_wait, 3)
            error["pre_attempt_s"] = (
                round(self.config.deadline_s - float(first), 3)
                if first is not None
                else None
            )
        if outcome.error:
            _with_blame(outcome.error["error"], outcome.http_status, record["role"])
        costs = [a["cost"] for a in outcome.attempts if a.get("cost") is not None]
        unknown = [a for a in outcome.attempts if a.get("billed") == BILLED_UNKNOWN]
        unbilled = sum(1 for a in unknown if a.get("outcome") == "deadline")
        record.update(
            status=outcome.status,
            http_status=outcome.http_status,
            retries=outcome.retries,
            attempts=outcome.attempts,
            latency_s=round(time.monotonic() - arrived, 3),
            queue_wait_s=round(queue_wait, 3),
            provider=data.get("provider"),
            generation_id=data.get("id"),
            model_served=data.get("model"),
            system_fingerprint=data.get("system_fingerprint"),
            finish_reasons=[
                c.get("finish_reason") for c in choices if isinstance(c, Mapping)
            ],
            native_finish_reasons=[
                c.get("native_finish_reason") for c in choices if isinstance(c, Mapping)
            ],
            input_tokens_actual=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            cost=usage.get("cost"),
            cost_total=round(sum(float(c) for c in costs), 10) if costs else None,
            billed_attempts=len(costs),
            # Attempts the deadline cut off: billed, at a cost cost_total leaves out.
            unbilled_deadline_attempts=unbilled,
            # All attempts of unknown cost: those, and those whose connection dropped
            # after the request was sent.
            unknown_cost_attempts=len(unknown),
            usage=usage or None,
            error=outcome.error["error"] if outcome.error else None,
            completed_at=_now(),
        )
        key = (record["run_id"], record["role"], prepared.sha256)
        if outcome.status == "ok":
            self._recent.pop(key, None)
        else:
            self._recent[key] = (record["call_id"], outcome.status, time.monotonic())
        if self.config.budget_basis == BUDGET_BASIS_LEDGER:
            # Counted before the line is written (a delayed line still counts).
            self._count_ledger_call(record)
        await self.ledger_final(record["run_id"], record)
        headers = {
            "x-skill-bench-call-id": call_id,
            "x-skill-bench-retries": str(outcome.retries),
        }
        if outcome.status != "ok":
            return JSONResponse(outcome.error, outcome.http_status, headers=headers)
        if prepared.stream_requested and prepared.endpoint == CHAT:
            return StreamingResponse(
                sse_frames(data), media_type="text/event-stream", headers=headers
            )
        return Response(
            outcome.content, 200, media_type="application/json", headers=headers
        )

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "balance": self.balance,
            "balance_error": self.balance_error,
            "balance_floor_usd": self.config.balance_floor_usd,
            "budget_usd": self.config.budget_usd,
            "budget_basis": self.config.budget_basis,
            "budget": self.budget,
            "encoding": tokens.encoding_name(),
            "models": [*protocol.CHAT_MODELS, protocol.EMBEDDING_MODEL],
            "deadline_s": self.config.deadline_s,
            "max_retries": self.config.max_retries,
            "settings": self.config.protocol_settings(),
            "settings_sha256": self.settings_sha256,
            "protocol_conformant": self.protocol_conformant,
            "effort_supported": dict(self.effort_supported),
            "pid": os.getpid(),
            "start_id": self.start_id,
            "users_lock": str(self.config.users_lock)
            if self.config.users_lock
            else None,
            "ledger_errors": self.ledger_errors,
            "ledger_lines_pending": sum(len(v) for v in self._unlogged.values()),
            # Upstream slots (monitoring): queue waits spend the calls' deadlines.
            "slots": {
                "max": self.config.max_concurrency,
                "in_use": self._slots_in_use,
                "waiting": self._slots_waiting,
            },
            "provider_policy": protocol.provider_policy_settings(
                self.config.provider_policy
            ),
        }


def build_app(gateway: Gateway) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await gateway.start()
        try:
            yield
        finally:
            await gateway.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await gateway.health()

    @app.get("/balance")
    async def balance() -> dict[str, Any]:
        await gateway.refresh_balance(force=True)
        return await gateway.health()

    @app.post("/runs/{run_id}/phase")
    async def phase(run_id: str, request: Request) -> Response:
        token = gateway.config.local_token
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return JSONResponse(
                _error_body(
                    401, "unauthorized", "use the local skill-bench credential"
                ),
                401,
            )
        if not RUN_ID_RE.fullmatch(run_id):
            return JSONResponse(
                _error_body(400, "invalid_run_id", "invalid run_id"), 400
            )
        try:
            value = (await request.json()).get("phase")
        except (ValueError, AttributeError):
            value = None
        if value not in PHASES:
            return JSONResponse(
                _error_body(400, "invalid_phase", f"phase must be one of {PHASES}"), 400
            )
        await gateway.set_phase(run_id, value)
        return JSONResponse({"run_id": run_id, "phase": value})

    @app.post("/runs/{run_id}/abandon")
    async def abandon(run_id: str, request: Request) -> Response:
        token = gateway.config.local_token
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return JSONResponse(
                _error_body(
                    401, "unauthorized", "use the local skill-bench credential"
                ),
                401,
            )
        if not RUN_ID_RE.fullmatch(run_id):
            return JSONResponse(
                _error_body(400, "invalid_run_id", "invalid run_id"), 400
            )
        try:
            reason = str((await request.json()).get("reason") or "")
        except (ValueError, AttributeError):
            reason = ""
        await gateway.abandon(run_id, reason)
        return JSONResponse({"run_id": run_id, "abandoned": True})

    @app.post("/runs/{run_id}/mark")
    async def mark(run_id: str, request: Request) -> Response:
        # Every refusal comes before anything is written (run.py probes the route with an
        # unknown event: 400 here, 404/405 from a gateway without it).
        token = gateway.config.local_token
        if (
            token is not None
            and request.headers.get("authorization") != f"Bearer {token}"
        ):
            return JSONResponse(
                _error_body(
                    401, "unauthorized", "use the local skill-bench credential"
                ),
                401,
            )
        if not RUN_ID_RE.fullmatch(run_id):
            return JSONResponse(
                _error_body(400, "invalid_run_id", "invalid run_id"), 400
            )
        body, problem = parse_mark(await request.body())
        if body is None:
            return JSONResponse(_error_body(400, "invalid_mark", str(problem)), 400)
        try:
            answer = await gateway.mark(run_id, body)
        except GatewayReject as exc:
            return JSONResponse(
                _error_body(exc.status, exc.kind, exc.message), exc.status
            )
        return JSONResponse(answer)

    @app.post("/runs/{run_id}/{role}/v1/{endpoint:path}")
    async def relay(
        run_id: str, role: str, endpoint: str, request: Request
    ) -> Response:
        return await gateway.relay(request, run_id, role, endpoint)

    return app


# ---------------------------------------------------------------------------
# Key loading, CLI, launcher
# ---------------------------------------------------------------------------


def load_api_key(env_file: Path | None) -> str:
    """Read the key from ``env_file`` if given, else from ``$OPENROUTER_API_KEY``. Never logs the value."""
    if env_file is not None:
        from dotenv import dotenv_values

        key = dotenv_values(env_file).get(KEY_ENV)
        source = f"--env-file {env_file}"
    else:
        key = os.environ.get(KEY_ENV)
        source = f"${KEY_ENV}"
    if not key or not key.strip():
        raise SystemExit(f"{KEY_ENV} not found in {source}")
    return key.strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="skill-bench OpenRouter gateway")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--env-file",
        type=Path,
        help=f"dotenv file holding {KEY_ENV} (default: environment)",
    )
    parser.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    parser.add_argument("--models", type=Path, default=protocol.MODELS_JSON)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument(
        "--balance-floor", type=float, default=DEFAULT_BALANCE_FLOOR_USD, help="USD"
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="USD this runs directory may spend, on the key's usage from when a budget "
        "was first set here (runs-dir/_gateway/budget.json); the gateway pauses once it "
        "is spent (default: no budget, only the balance floor)",
    )
    parser.add_argument(
        "--budget-basis",
        choices=BUDGET_BASES,
        default=BUDGET_BASIS_KEY,
        help="what --budget counts: the key's all-time usage (key, every experiment "
        "gateway) or the cost_total of this runs directory's own ledgers from when a "
        "ledger budget was first set here (ledger, runs-dir/_gateway/budget_ledger.json: "
        "the experiment's spend on the key never uses it up)",
    )
    parser.add_argument(
        "--experiment-budget",
        type=Path,
        action="append",
        default=[],
        help="repeatable: another runs directory's budget.json whose amount this gateway "
        "also stops on once the key's usage has grown by it (so this gateway never "
        "passes the experiment's cap); needs --budget",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=protocol.CALL_DEADLINE_S,
        help="seconds per call (protocol value; changing it needs --allow-protocol-overrides)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=protocol.GATEWAY_MAX_RETRIES,
        help="protocol value; changing it needs --allow-protocol-overrides",
    )
    parser.add_argument(
        "--tokenizer-margin",
        type=Fraction,
        default=protocol.TOKENIZER_MARGIN,
        help="provider-window clamp margin, e.g. 1/10 (protocol value; changing it needs "
        "--allow-protocol-overrides)",
    )
    parser.add_argument(
        "--allow-protocol-overrides",
        action="store_true",
        help="accept a --deadline/--max-retries/--tokenizer-margin that differs from the "
        "protocol (tests and debugging only: every call line is then marked "
        "protocol_conformant=false)",
    )
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument(
        "--no-trust-env",
        action="store_true",
        help="ignore proxy environment for upstream",
    )
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args(argv)
    overridden = [
        name
        for name, value, frozen in (
            ("--deadline", args.deadline, protocol.CALL_DEADLINE_S),
            ("--max-retries", args.max_retries, protocol.GATEWAY_MAX_RETRIES),
            ("--tokenizer-margin", args.tokenizer_margin, protocol.TOKENIZER_MARGIN),
        )
        if value != frozen
    ]
    if args.budget is not None and not args.budget > 0:
        parser.error("--budget must be positive")
    if args.budget is None and (
        args.budget_basis != BUDGET_BASIS_KEY or args.experiment_budget
    ):
        parser.error("--budget-basis ledger and --experiment-budget need --budget")
    if overridden and not args.allow_protocol_overrides:
        parser.error(
            f"{', '.join(overridden)} differ from the frozen protocol (H3); pass "
            "--allow-protocol-overrides to run a non-protocol gateway anyway"
        )
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = GatewayConfig(
        api_key=load_api_key(args.env_file),
        models=protocol.load_models(args.models),
        runs_dir=args.runs_dir,
        upstream=args.upstream,
        deadline_s=args.deadline,
        max_retries=args.max_retries,
        tokenizer_margin=args.tokenizer_margin,
        balance_floor_usd=args.balance_floor,
        budget_usd=args.budget,
        budget_basis=args.budget_basis,
        experiment_budgets=tuple(p.absolute() for p in args.experiment_budget),
        max_concurrency=args.max_concurrency,
        trust_env=not args.no_trust_env,
        users_lock=gateway_users_lock(args.runs_dir, args.host, args.port).resolve(),
    )
    import uvicorn

    uvicorn.run(
        Gateway(config).app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
        # Clients (httpx, openai SDK) drop idle connections after 5 s; the server must keep
        # them longer, or a reused connection can be closed under the client mid-request
        # and surface as a spurious transport (infra) failure.
        timeout_keep_alive=SERVER_KEEP_ALIVE_S,
    )


@dataclass
class GatewayProcess:
    """A gateway child process started by :func:`launch_gateway`.

    ``owner`` is the harness's :class:`~evals.skill_bench.infra.processes.OwnerLock` on
    this gateway: held until :meth:`stop`, released by the kernel if the harness dies.
    """

    process: OwnedProcess
    base_url: str
    owner: OwnerLock | None = None

    def url(self, run_id: str, role: str) -> str:
        return gateway_url(self.base_url, run_id, role)

    async def stop(self) -> int | None:
        try:
            return await self.process.stop()
        finally:
            if self.owner is not None:
                self.owner.release()


def _host_slug(host: str) -> str:
    # "localhost" names the same loopback gateway as "127.0.0.1" (one lock for both).
    host = "127.0.0.1" if host.strip().lower() == "localhost" else host
    return re.sub(r"[^A-Za-z0-9.-]+", "_", host)


def gateway_owner_lock(runs_dir: Path, host: str, port: int) -> Path:
    """The owner lock of the gateway on ``host:port`` under ``runs_dir`` (N1)."""
    return runs_dir / "_gateway" / f"gateway-{_host_slug(host)}-{port}.lock"


def gateway_users_lock(runs_dir: Path, host: str, port: int) -> Path:
    """The users lock of the gateway on ``host:port`` under ``runs_dir`` (X12).

    Every ``run.py`` that uses the gateway holds it shared for its whole lifetime; no
    cleanup kills a gateway while it is held, whatever became of the gateway's owner.
    """
    return runs_dir / "_gateway" / f"gateway-{_host_slug(host)}-{port}.users.lock"


_PROVIDER_KEY_RE = re.compile(
    r"(API_KEY|API_TOKEN|SECRET_KEY|ACCESS_KEY_ID|SECRET_ACCESS_KEY)$"
)


async def launch_gateway(
    *,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    env_file: Path | None = None,
    runs_dir: Path = protocol.RUNS_DIR,
    models_path: Path = protocol.MODELS_JSON,
    upstream: str = DEFAULT_UPSTREAM,
    balance_floor_usd: float = DEFAULT_BALANCE_FLOOR_USD,
    budget_usd: float | None = None,
    extra_args: Sequence[str] = (),
    log_path: Path | None = None,
    registry: Path | None = None,
    startup_timeout_s: float = 60.0,
    python: str = sys.executable,
) -> GatewayProcess:
    """Start ``python -m evals.skill_bench.infra.gateway`` as an owned process and wait for /health.

    The caller never reads the key: the child loads it from ``env_file`` or, when
    that is None, inherits ``$OPENROUTER_API_KEY``. Other provider keys are not passed on.

    Several harnesses may share one ``runs_dir`` (each with its own gateway port). The
    calling harness holds the gateway's owner lock (:func:`gateway_owner_lock`) until
    :meth:`GatewayProcess.stop`; the cleanup before the start kills only registered
    gateways whose owner lock is free, i.e. whose harness has died (N1). A port whose lock
    a live harness holds is refused.
    """
    control = runs_dir / "_gateway"
    registry = registry or control / "owned_processes.jsonl"
    lock_path = gateway_owner_lock(runs_dir, host, port)
    try:
        owner = await asyncio.to_thread(OwnerLock.acquire, lock_path)
    except OwnedProcessError as exc:
        raise OwnedProcessError(
            f"the gateway on {host}:{port} belongs to another live harness ({exc})"
        ) from exc
    try:
        return await _launch_owned_gateway(
            owner,
            port=port,
            host=host,
            env_file=env_file,
            runs_dir=runs_dir,
            models_path=models_path,
            upstream=upstream,
            balance_floor_usd=balance_floor_usd,
            budget_usd=budget_usd,
            extra_args=extra_args,
            log_path=log_path or control / "gateway.log",
            registry=registry,
            startup_timeout_s=startup_timeout_s,
            python=python,
        )
    except BaseException:
        owner.release()
        raise


async def _launch_owned_gateway(
    owner: OwnerLock,
    *,
    port: int,
    host: str,
    env_file: Path | None,
    runs_dir: Path,
    models_path: Path,
    upstream: str,
    balance_floor_usd: float,
    budget_usd: float | None,
    extra_args: Sequence[str],
    log_path: Path,
    registry: Path,
    startup_timeout_s: float,
    python: str,
) -> GatewayProcess:
    # Gateways left by crashed harnesses (registered, never stopped, owner lock free) are
    # killed; a live harness's gateway (lock held) and any PID that now belongs to another
    # program are never signalled.
    killed = await asyncio.to_thread(cleanup_owned, registry, held_locks=(owner,))
    # A killed gateway's socket closes as the kernel reaps it: give it a moment.
    deadline = time.monotonic() + (5.0 if killed else 0.0)
    while not await asyncio.to_thread(port_is_free, port, host):
        if time.monotonic() >= deadline:
            raise OwnedProcessError(f"port {port} is already in use")
        await asyncio.sleep(0.1)
    argv = [
        python,
        "-m",
        "evals.skill_bench.infra.gateway",
        "--host",
        host,
        "--port",
        str(port),
        "--runs-dir",
        str(runs_dir),
        "--models",
        str(models_path),
        "--upstream",
        upstream,
        "--balance-floor",
        str(balance_floor_usd),
        *(["--budget", str(budget_usd)] if budget_usd is not None else []),
        *(["--env-file", str(env_file)] if env_file is not None else []),
        *extra_args,
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not _PROVIDER_KEY_RE.search(k) or k == KEY_ENV
    }
    if env_file is not None:
        env.pop(KEY_ENV, None)
    loopback = "127.0.0.1,localhost,::1"
    env["NO_PROXY"] = ",".join(filter(None, [env.get("NO_PROXY", ""), loopback]))
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONUNBUFFERED"] = "1"
    process = await asyncio.to_thread(
        OwnedProcess.spawn,
        "gateway",
        argv,
        env=env,
        cwd=protocol.REPO_ROOT,
        log_path=log_path,
        registry=registry,
        owner_lock=owner.path,
        users_lock=gateway_users_lock(runs_dir, host, port),
    )
    base_url = f"http://{host}:{port}"
    try:
        await process.wait_healthy(base_url + "/health", timeout_s=startup_timeout_s)
    except BaseException:
        await process.stop()
        raise
    return GatewayProcess(process, base_url, owner)


if __name__ == "__main__":
    main()
