"""Failure taxonomy, counters and recovery rules.

Three kinds, never mixed:

* ``INFRA`` - the environment failed: gateway retries exhausted (connect/read
  errors, 408, 429, 5xx, provider unavailable), the 600 s deadline spent partly
  on retries, in the gateway queue or on the gateway's balance check, the budget
  pause (402), a MIRIX cell server that crashed or became unhealthy, an owned
  process that cannot be cleaned up, a database outage, the WebShop
  server unreachable, not ready or failing a request (``webshop_server``), or the
  Q18 wait for idle learning timing out. **Never a model failure.**
* ``MODEL`` - the model's own behaviour: format errors, truncated output
  (``finish_reason=length``), content filtering, and a call whose single upstream
  attempt had (nearly) the whole deadline and still ran past it (F9: an actor
  timeout is a model failure; a learner/judge timeout is counted in the method's
  metrics). The gateway's 504 body says how much of the deadline that attempt had
  (``attempt_budget_s``); anything more than
  ``min(protocol.MODEL_TIMEOUT_MAX_OVERHEAD_S, 25% of the deadline)`` lost before it
  makes the timeout INFRA (``deadline_after_queue``). Scored normally.
* ``HARNESS`` - our bug or a protocol violation: requests the gateway refuses
  (budget, whitelist, identity), malformed requests the provider rejects,
  unexpected exceptions, audit/fingerprint violations, and a MIRIX cell database
  that cannot be proven the attempt's (``mirix_database_conflict``: another run's
  memory may hold the name, so it is never resumed or dropped automatically).

Recovery (H8): model failures are scored (actor) or counted (learning-side roles) and
the run continues. An infra failure in the **learning** phase resumes the attempt from
its latest learning checkpoint; a harness failure stops it for a person. Recorded, not
stopped (rules harness-stop-learning and gw-404-no-endpoint): an upstream refusal
(:func:`upstream_refusal`: a payload the provider refuses, HTTP 413 or a 400 naming its
size or images, or a no-endpoint 404) of the actor, the learner or J in the learning
phase is counted and the phase goes on (``count_and_continue``; the actor's item ends
unanswered and is scored 0, ``provider_unserved``); the same refusal on
:data:`REFUSAL_STREAK_ITEMS` consecutive calls of one role stops it
(``upstream_refusal_streak``, HARNESS). A MIRIX learning batch lost to infrastructure
after :data:`MIRIX_BATCH_ROLLBACKS` consecutive rollbacks of its checkpoint is accepted
and recorded (``accept_failed_batch``). In the **test** phase memory is frozen, so only
the failed item is re-run later and never scored as a model failure, except an item the
provider refused no-endpoint on :data:`PROVIDER_UNSERVED_TRIES` consecutive tries: that
try is scored unanswered and flagged (``provider_unserved``). Nothing is re-run
automatically.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.infra.processes import OwnedProcessError, append_jsonl


class FailureKind(str, Enum):
    INFRA = "infra"
    MODEL = "model"
    HARNESS = "harness"


class Phase(str, Enum):
    LEARNING = "learning"
    TEST = "test"


class Recovery(str, Enum):
    SCORE_AND_CONTINUE = (
        "score_and_continue"  # actor model failure: the item scores as failed
    )
    COUNT_AND_CONTINUE = (
        "count_and_continue"  # learning-side model failure: counted in method metrics
    )
    RESTART_CELL = (
        "restart_cell"  # learning phase: new attempt with a fresh DB and HOME
    )
    RERUN_ITEM = "rerun_item"  # test phase: re-run only this item, not scored now
    #: A MIRIX learning batch lost to infrastructure after MIRIX_BATCH_ROLLBACKS
    #: consecutive rollbacks of its checkpoint: accepted and recorded (run.py writes it).
    ACCEPT_FAILED_BATCH = "accept_failed_batch"


# Stable failure codes (the ``code`` field).
FORMAT_ERROR = "format_error"
TRUNCATED = "truncated"
CONTENT_FILTER = "content_filter"
MODEL_TIMEOUT = "model_timeout"
UNANSWERED = "unanswered"
DEADLINE_AFTER_RETRIES = "deadline_after_retries"
DEADLINE_WHILE_QUEUED = "deadline_while_queued"
DEADLINE_AFTER_QUEUE = (
    "deadline_after_queue"  # one attempt, but the deadline was spent before it
)
OWNED_PROCESS = "owned_process"
UPSTREAM_INFRA = "upstream_infra"
BUDGET_PAUSED = "budget_paused"
GATEWAY_UNREACHABLE = "gateway_unreachable"
GATEWAY_CLIENT_TIMEOUT = "gateway_client_timeout"
MIRIX_SERVER = "mirix_server"
#: The WebShop server (``envs/webshop.WebShopServerError``): unreachable, not ready, a
#: timeout, a 5xx or an episode it lost. INFRA: the item is re-run, never scored.
WEBSHOP_SERVER = "webshop_server"
DATABASE = "database"
IDLE_WAIT_TIMEOUT = "idle_wait_timeout"
#: MIRIX's own learning hit infrastructure: a failed learning run, a session deferred
#: after an infrastructure error (M3.5 ``distill_retry_after``), a curator batch stopped
#: by one, or a mirix/embedding call no retry recovered (X3, X13). INFRA.
MIRIX_LEARNING = "mirix_learning"
#: MIRIX answered a retrieval without its topic extraction or its query vector because
#: an infrastructure (or harness) failure took it away: MIRIX falls back silently (M7.1,
#: M7.3), so the method raises instead and the item is not scored on the degraded
#: retrieval (H8; rule N1). A model failure of the extraction is MIRIX's own and scored.
MIRIX_RETRIEVAL = "mirix_retrieval"
#: A new attempt found its MIRIX cell database already there, or the database under the
#: attempt's name is marked as another run's, or a resume would move aside and drop one
#: without an owner marker (``infra/mirix_server.CellDatabaseConflict``, B02). HARNESS:
#: whatever holds the name may be another run's memory, so a person decides.
MIRIX_DATABASE_CONFLICT = "mirix_database_conflict"
GATEWAY_REJECTED = "gateway_rejected"
UPSTREAM_REJECTED = "upstream_rejected"
UPSTREAM_AUTH = "upstream_auth"
UNEXPECTED_EXCEPTION = "unexpected_exception"
PROTOCOL_DRIFT = "protocol_drift"
AUDIT_VIOLATION = "audit_violation"

#: Roles whose calls happen inside a memory method (their model failures are counted, not scored).
LEARNING_SIDE_ROLES = frozenset({"learner", "judge", "mirix", "embedding"})

# --- Upstream refusals, recorded instead of stopping ------------------------------------------
#: An actor call the provider refused (:func:`upstream_refusal`, of a class the caller
#: tolerates): the item ended unanswered and is scored 0, its row flagged
#: ``provider_unserved`` (rules harness-stop-learning, gw-404-no-endpoint). HARNESS kind
#: (never the model's failure), recovery ``score_and_continue``.
PROVIDER_UNSERVED = "provider_unserved"
#: The same upstream refusal on :data:`REFUSAL_STREAK_ITEMS` consecutive calls of one role
#: (:class:`RefusalStreaks`, :func:`learning_refusal_streak`): the learning phase stops for
#: a person (HARNESS, ``restart_cell``; the scheduler never releases it by itself).
REFUSAL_STREAK = "upstream_refusal_streak"
#: Refusal classes (:func:`refusal_class`): a payload the provider refuses (upstream HTTP
#: 413, or a 400 whose message says so, :data:`PAYLOAD_REFUSAL_MARKERS`), no provider
#: endpoint for this request (a final no-endpoint 404,
#: ``infra/gateway.NO_ENDPOINT_FINAL_PATTERNS``), and a request the gateway itself refused
#: (``gateway_rejected``; counted for MIRIX's calls only).
REFUSAL_PAYLOAD = "payload"
REFUSAL_NO_ENDPOINT = "no_endpoint"
REFUSAL_GATEWAY = "gateway"
#: Gateway refusal types (``error.type``) that mean a caller broke the protocol (its
#: credential, role, model, run or phase): a copy of ``audit.PROTOCOL_REJECTIONS`` (a
#: test keeps the two equal). The audit fails a cell for any of them (H10.3), so a MIRIX
#: learning call refused this way is never counted and learned past: it stops the
#: attempt at once.
PROTOCOL_REJECTION_TYPES = frozenset(
    {
        "unauthorized",
        "invalid_role",
        "role_endpoint_mismatch",
        "unsupported_endpoint",
        "model_mismatch_for_run",
        "model_not_allowed",
        "run_abandoned",
        "invalid_run_id",
        "invalid_item",
        "invalid_phase",
        "phase_mismatch",
    }
)
#: The upstream refusal classes a caller may tolerate (``rollout.run_episode(unserved=)``).
REFUSAL_CLASSES: tuple[str, ...] = (REFUSAL_PAYLOAD, REFUSAL_NO_ENDPOINT)
PAYLOAD_REFUSAL_STATUSES = frozenset({400, 413})
#: The texts (lower case) of an upstream 400 that refuses the request's size or images.
#: "provider returned error" is OpenRouter's generic upstream message, which may be all
#: a provider says when it refuses a payload. Any other 400 (a ``max_tokens``
#: the model does not take, a ``tool_call_id`` the history lacks, an unsupported
#: parameter) is a request the harness built wrong: not a refusal, so it stops. A 413
#: is a payload refusal whatever it says.
PAYLOAD_REFUSAL_MARKERS: tuple[str, ...] = (
    "provider returned error",
    "payload",
    "too large",
    "too long",
    "size",
    "image",
    "context length",
)
#: Texts of an upstream 400 that name the request's own parameters: a ``max_tokens`` or
#: completion budget the model does not take, a parameter it does not know, a tool call
#: id the history lacks, an image it cannot read. Every retry repeats such a request,
#: whatever else the text says ("too large", "image"): it is the harness's to fix, so it
#: stops (HARNESS), never counted as a payload refusal.
REQUEST_PARAMETER_MARKERS: tuple[str, ...] = (
    "max_tokens",
    "max_completion_tokens",
    "completion tokens",
    "unsupported parameter",
    "unrecognized request argument",
    "tool_call_id",
    "invalid image",
    "image url",
    "mime type",
)
#: vLLM's context-length refusal splits what the request asked for ("... maximum context
#: length is N tokens. However, you requested M tokens (A in the messages, B in the
#: completion)"): a completion budget B that alone reaches N is the harness's
#: ``max_tokens``, not the input's size.
_CONTEXT_SPLIT = re.compile(
    r"maximum context length is (\d+).{0,200}?(\d+) in the completion", re.DOTALL
)


def names_request_parameter(detail: str) -> bool:
    """Whether an upstream 400's text (lower case) names the request's own parameters
    (:data:`REQUEST_PARAMETER_MARKERS`) or a completion budget that alone fills the
    model's context (pure)."""
    if any(marker in detail for marker in REQUEST_PARAMETER_MARKERS):
        return True
    split = _CONTEXT_SPLIT.search(detail)
    return bool(split and int(split.group(2)) >= int(split.group(1)))


#: The texts of an upstream 404 that says no provider endpoint can serve the request: a
#: copy of ``infra/gateway.NO_ENDPOINT_PATTERNS`` (not imported: the gateway imports this
#: module; a test keeps the two equal).
NO_ENDPOINT_MARKERS: tuple[str, ...] = ("no endpoints found", "no allowed providers")
#: Refusals of the account, not of the request (a key-level data policy fails every
#: call): never tolerated, they stop at once.
ACCOUNT_REFUSAL_MARKERS: tuple[str, ...] = ("data policy",)
#: Roles whose upstream refusals the learning phase counts and goes on with.
TOLERATED_REFUSAL_ROLES = frozenset({"actor", "learner", "judge"})
#: Consecutive calls of one role with the same refusal that stop the learning phase.
REFUSAL_STREAK_ITEMS = 3
#: Consecutive no-endpoint tries of a test item after which its try is scored unanswered.
PROVIDER_UNSERVED_TRIES = 2
#: Consecutive rollbacks of one learning checkpoint after which the MIRIX batch it
#: replays is accepted despite its infrastructure losses.
MIRIX_BATCH_ROLLBACKS = 3


@dataclass(frozen=True)
class Failure:
    kind: FailureKind
    code: str
    role: str | None = None
    status: int | None = None
    detail: str = ""
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Failure:
        return cls(
            kind=FailureKind(data["kind"]),
            code=str(data["code"]),
            role=data.get("role"),
            status=data.get("status"),
            detail=str(data.get("detail", "")),
            retries=int(data.get("retries", 0)),
        )


class GatewayCallError(RuntimeError):
    """Raised by gateway clients (actor, methods) for a non-2xx gateway response."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(f"{failure.kind.value}/{failure.code}: {failure.detail}")
        self.failure = failure


class LearningAttemptInvalid(RuntimeError):
    """A learning-phase infra/harness failure: the cell must restart with a fresh attempt."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(
            f"learning attempt invalid: {failure.kind.value}/{failure.code}: {failure.detail}"
        )
        self.failure = failure


class MirixRetrievalDegraded(RuntimeError):
    """MIRIX's retrieval lost its topic extraction or query vector to a failure that is not
    the model's (``MIRIX_RETRIEVAL``, rule N1). Raised for the first two consecutive
    degraded retrievals of a test item (the item stays pending for a re-run, H8) and for
    the third consecutive learning item with a degraded retrieval (the attempt resumes
    from its checkpoint). The third degraded try of a test item and the first two of the
    learning phase are injected and flagged instead (``methods/mirix.py``,
    ``Injection.degraded``; rule N1)."""

    def __init__(self, failure: Failure) -> None:
        super().__init__(
            f"MIRIX retrieval degraded: {failure.kind.value}/{failure.code}: "
            f"{failure.detail}"
        )
        self.failure = failure


def _error_of(body: Any) -> Mapping[str, Any]:
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        return body["error"]
    return {}


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_timeout_allowance(deadline_s: float) -> float:
    """Deadline time that may be lost before the one attempt of a model timeout (H8)."""
    return min(
        protocol.MODEL_TIMEOUT_MAX_OVERHEAD_S,
        float(protocol.MODEL_TIMEOUT_MAX_OVERHEAD_FRACTION) * deadline_s,
    )


def attempt_had_whole_deadline(error: Mapping[str, Any]) -> bool:
    """True when the timed-out attempt started with (nearly) the whole deadline.

    Needs the gateway's ``deadline_s`` and ``attempt_budget_s``; without them the time
    cannot be attributed and the answer is False (never blame the model without evidence).
    """
    deadline = _as_float(error.get("deadline_s"))
    budget = _as_float(error.get("attempt_budget_s"))
    if deadline is None or budget is None or deadline <= 0:
        return False
    return budget >= deadline - model_timeout_allowance(deadline)


def classify_gateway_error(status: int, body: Any, role: str | None = None) -> Failure:
    """Classify a non-2xx gateway response (see ``infra/gateway.py`` error bodies)."""
    error = _error_of(body)
    kind_name = str(error.get("type", ""))
    source = error.get("source")
    detail = str(error.get("message", ""))[:500]
    retries = int(error.get("retries", 0) or 0)

    def make(kind: FailureKind, code: str) -> Failure:
        return Failure(kind, code, role, status, detail, retries)

    if kind_name == "deadline_exceeded":
        reason = error.get("reason")
        if reason == "deadline_while_queued" or error.get("upstream_attempts") == 0:
            return make(FailureKind.INFRA, DEADLINE_WHILE_QUEUED)
        if retries == 0 and reason == "deadline_during_attempt":
            if not attempt_had_whole_deadline(error):
                # Queueing for a slot or a slow balance check ate the deadline first.
                return make(FailureKind.INFRA, DEADLINE_AFTER_QUEUE)
            if role != "embedding":
                return make(
                    FailureKind.MODEL, MODEL_TIMEOUT
                )  # the model alone used the whole budget
        return make(FailureKind.INFRA, DEADLINE_AFTER_RETRIES)
    if status == 402:
        return make(FailureKind.INFRA, BUDGET_PAUSED)
    if kind_name == "upstream_infra_failure" or status in (408, 429) or status >= 500:
        return make(FailureKind.INFRA, UPSTREAM_INFRA)
    if source == "gateway":
        return make(FailureKind.HARNESS, GATEWAY_REJECTED)
    if status == 401:
        return make(
            FailureKind.INFRA, UPSTREAM_AUTH
        )  # the gateway's key, not the experiment
    if status == 403 and "moderat" in detail.lower():
        return make(FailureKind.MODEL, CONTENT_FILTER)
    return make(FailureKind.HARNESS, UPSTREAM_REJECTED)


def ledger_line_failure(line: Mapping[str, Any]) -> Failure | None:
    """The failure a final gateway ledger line (``event: call``) records; None for a success."""
    if line.get("status") == "ok":
        return None
    try:
        status = int(line.get("http_status") or 0)
    except (TypeError, ValueError):
        status = 0
    error = line.get("error") if isinstance(line.get("error"), Mapping) else {}
    role = line.get("role")
    return classify_gateway_error(
        status, {"error": error}, str(role) if role is not None else None
    )


def line_phase(line: Mapping[str, Any]) -> str | None:
    """The phase a ledger line belongs to: the run's phase set by ``run.py`` when the
    gateway recorded it (``run_phase``), else the line's (possibly self-declared) phase."""
    value = line.get("run_phase") or line.get("phase")
    return str(value) if value is not None else None


def unrecovered_infra_calls(
    lines: Iterable[Mapping[str, Any]],
    *,
    roles: Iterable[str],
    phases: Iterable[str],
) -> list[Mapping[str, Any]]:
    """Final call lines on ``roles`` in ``phases`` that failed on infrastructure (INFRA) and
    that no later successful repeat of the same request recovered (H8, X13).

    A caller's application-level retry (MIRIX's M3.5 back-off, its embedding retries)
    sends the same request again; the gateway flags it ``repeat_of`` the failed call. A
    success at the end of such a chain recovers every failure in it. What is left was
    swallowed by the caller: MIRIX deferred the work, fell back (retrieval without topic
    extraction) or stored a skill without its vector, and the learning attempt is no
    longer what an attempt without the outage would have learned.
    """
    wanted_roles, wanted_phases = set(roles), set(phases)
    calls = [line for line in lines if line.get("event") == "call"]
    by_id = {str(line.get("call_id")): line for line in calls}
    recovered: set[str] = set()
    for line in calls:
        if line.get("status") != "ok":
            continue
        previous, seen = line.get("repeat_of"), set()
        while previous and str(previous) not in seen:
            seen.add(str(previous))
            recovered.add(str(previous))
            previous = (by_id.get(str(previous)) or {}).get("repeat_of")
    found = []
    for line in calls:
        if (
            line.get("role") not in wanted_roles
            or line_phase(line) not in wanted_phases
        ):
            continue
        failure = ledger_line_failure(line)
        if failure is None or failure.kind is not FailureKind.INFRA:
            continue
        if str(line.get("call_id")) not in recovered:
            found.append(line)
    return found


def classify_completion(
    data: Mapping[str, Any], role: str | None = None
) -> Failure | None:
    """Model-side problems inside a successful chat response (truncation, content filter)."""
    for choice in data.get("choices") or []:
        reason = choice.get("finish_reason")
        if reason == "length":
            return Failure(
                FailureKind.MODEL, TRUNCATED, role, 200, "finish_reason=length"
            )
        if reason == "content_filter":
            return Failure(
                FailureKind.MODEL,
                CONTENT_FILTER,
                role,
                200,
                "finish_reason=content_filter",
            )
    return None


def _is_webshop_server_error(exc: BaseException) -> bool:
    """``envs/webshop.WebShopServerError`` (by name: this module never imports the
    environments): the WebShop server failed, not the harness or the model."""
    return any(
        cls.__name__ == "WebShopServerError"
        and cls.__module__ == "evals.skill_bench.envs.webshop"
        for cls in type(exc).__mro__
    )


def classify_exception(exc: BaseException, role: str | None = None) -> Failure:
    """Map an exception raised while running an item to a failure."""
    if isinstance(
        exc, (GatewayCallError, LearningAttemptInvalid, MirixRetrievalDegraded)
    ):
        return exc.failure
    detail = f"{type(exc).__name__}: {exc}"[:500]
    module = type(exc).__module__
    if _is_webshop_server_error(exc):
        # Before the httpx branches: its transport errors are the server's, not the
        # gateway's (it wraps them).
        return Failure(FailureKind.INFRA, WEBSHOP_SERVER, role, None, detail)
    if isinstance(exc, httpx.TimeoutException):
        return Failure(FailureKind.INFRA, GATEWAY_CLIENT_TIMEOUT, role, None, detail)
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return Failure(FailureKind.INFRA, GATEWAY_UNREACHABLE, role, None, detail)
    if type(exc).__name__ == "CellDatabaseConflict":
        return Failure(FailureKind.HARNESS, MIRIX_DATABASE_CONFLICT, role, None, detail)
    if type(exc).__name__ == "MirixServerError":
        return Failure(FailureKind.INFRA, MIRIX_SERVER, role, None, detail)
    if isinstance(exc, OwnedProcessError):
        return Failure(FailureKind.INFRA, OWNED_PROCESS, role, None, detail)
    if module.startswith(("asyncpg", "sqlalchemy")):
        return Failure(FailureKind.INFRA, DATABASE, role, None, detail)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Failure(FailureKind.INFRA, IDLE_WAIT_TIMEOUT, role, None, detail)
    if type(exc).__name__ == "ProtocolDriftError":
        return Failure(FailureKind.HARNESS, PROTOCOL_DRIFT, role, None, detail)
    return Failure(FailureKind.HARNESS, UNEXPECTED_EXCEPTION, role, None, detail)


def recovery_for(phase: Phase | str, failure: Failure) -> Recovery:
    """H8 recovery rule for a failure observed in ``phase``."""
    phase = Phase(phase)
    if failure.code == PROVIDER_UNSERVED:
        # The actor's refused item was scored unanswered.
        return Recovery.SCORE_AND_CONTINUE
    if failure.kind is FailureKind.MODEL:
        if failure.role in LEARNING_SIDE_ROLES:
            return Recovery.COUNT_AND_CONTINUE
        return Recovery.SCORE_AND_CONTINUE
    if (
        phase is Phase.LEARNING
        and failure.role in TOLERATED_REFUSAL_ROLES
        and upstream_refusal(failure) is not None
    ):
        return Recovery.COUNT_AND_CONTINUE
    return Recovery.RESTART_CELL if phase is Phase.LEARNING else Recovery.RERUN_ITEM


def is_scored_as_model_failure(failure: Failure) -> bool:
    return failure.kind is FailureKind.MODEL and failure.role not in LEARNING_SIDE_ROLES


class FailureCounters:
    """Counts by (phase, kind, code, role); JSON round-trippable for reports."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, str, str]] = Counter()

    def add(self, phase: Phase | str, failure: Failure, n: int = 1) -> None:
        self._counts[
            (Phase(phase).value, failure.kind.value, failure.code, failure.role or "-")
        ] += n

    def total(
        self, *, phase: Phase | str | None = None, kind: FailureKind | None = None
    ) -> int:
        return sum(
            n
            for (p, k, _, _), n in self._counts.items()
            if (phase is None or p == Phase(phase).value)
            and (kind is None or k == kind.value)
        )

    def to_dict(self) -> dict[str, Any]:
        rows = [
            {
                "phase": p,
                "kind": k,
                "code": c,
                "role": None if r == "-" else r,
                "count": n,
            }
            for (p, k, c, r), n in sorted(self._counts.items())
        ]
        totals = {
            phase.value: {
                kind.value: self.total(phase=phase, kind=kind) for kind in FailureKind
            }
            for phase in Phase
        }
        return {"rows": rows, "totals": totals}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FailureCounters:
        counters = cls()
        for row in data.get("rows", []):
            key = (row["phase"], row["kind"], row["code"], row.get("role") or "-")
            counters._counts[key] += int(row["count"])
        return counters


class FailureLog:
    """Append-only, fsync'd ``failures.jsonl`` of one cell."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(
        self,
        phase: Phase | str,
        failure: Failure,
        *,
        item: str | None = None,
        attempt: int | None = None,
        stage: str | None = None,
    ) -> Recovery:
        """Append one row; ``stage`` (the rollout stage, e.g. ``actor`` for a test item
        scored ``provider_unserved``) is written only when given, so every other row
        keeps its shape."""
        recovery = recovery_for(phase, failure)
        append_jsonl(
            self.path,
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "phase": Phase(phase).value,
                "item": item,
                "attempt": attempt,
                **({"stage": stage} if stage is not None else {}),
                "recovery": recovery.value,
                **failure.to_dict(),
            },
        )
        return recovery

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            # "\n" only: a record may hold U+2028 (ensure_ascii=False).
            for line in self.path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]

    def counters(self) -> FailureCounters:
        counters = FailureCounters()
        for row in self.rows():
            counters.add(row["phase"], Failure.from_dict(row))
        return counters


def pending_test_items(
    item_ids: Sequence[str], results: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Test-phase resume: items without a scored result, in the original order.

    A result counts only when it carries a score; infra/harness failures leave
    ``score`` empty and are therefore re-run, while scored model failures are final.
    Later rows for the same item win.
    """
    scored: dict[str, bool] = {}
    for row in results:
        key = row.get("raw_id") or row.get("item")
        if key is not None:
            scored[str(key)] = row.get("score") is not None
    return [item for item in item_ids if not scored.get(item, False)]


# ---------------------------------------------------------------------------
# Upstream refusals: recorded, and stopped only when they repeat
# ---------------------------------------------------------------------------


def upstream_refusal(failure: Failure) -> str | None:
    """The class of a request the provider refused (pure): ``payload`` for an upstream
    413, or a 400 whose message names its size or images (:data:`PAYLOAD_REFUSAL_MARKERS`)
    and none of the request's own parameters (:func:`names_request_parameter`),
    ``no_endpoint`` for an upstream 404 saying no provider endpoint serves the
    request (:data:`NO_ENDPOINT_MARKERS`); None for anything else - every other failure,
    an upstream 403 or plain 404, and a refusal of the account
    (:data:`ACCOUNT_REFUSAL_MARKERS`: a data policy fails every call, so it stays a stop).
    """
    if failure.kind is not FailureKind.HARNESS or failure.code != UPSTREAM_REJECTED:
        return None
    detail = failure.detail.lower()
    if any(marker in detail for marker in ACCOUNT_REFUSAL_MARKERS):
        return None
    if failure.status == 413:
        return REFUSAL_PAYLOAD
    if failure.status == 400 and names_request_parameter(detail):
        return None
    if failure.status in PAYLOAD_REFUSAL_STATUSES and any(
        marker in detail for marker in PAYLOAD_REFUSAL_MARKERS
    ):
        return REFUSAL_PAYLOAD
    if failure.status == 404 and any(m in detail for m in NO_ENDPOINT_MARKERS):
        return REFUSAL_NO_ENDPOINT
    return None


def refusal_class(failure: Failure) -> str | None:
    """:func:`upstream_refusal`, plus the class of an actor call already scored
    ``provider_unserved`` and ``gateway`` for a request the gateway itself refused
    (``gateway_rejected``): the classes ``costs.json`` counts per phase and role."""
    if failure.code == PROVIDER_UNSERVED:
        return REFUSAL_NO_ENDPOINT if failure.status == 404 else REFUSAL_PAYLOAD
    if failure.kind is FailureKind.HARNESS and failure.code == GATEWAY_REJECTED:
        return REFUSAL_GATEWAY
    return upstream_refusal(failure)


def refusal_signature(failure: Failure) -> str | None:
    """``role/class/status`` of a refusal (:func:`refusal_class`), None for a failure that
    is not one: two refusals are "the same" when their signatures are equal."""
    cls = refusal_class(failure)
    if cls is None:
        return None
    return f"{failure.role or '-'}/{cls}/{failure.status}"


def refusal_streak_failure(refusal: Failure, count: int) -> Failure:
    """The HARNESS stop after ``count`` consecutive calls with the same refusal."""
    signature = refusal_signature(refusal) or f"{refusal.role or '-'}/{refusal.code}"
    detail = (
        f"the same upstream refusal ({signature}) on {count} consecutive "
        f"{refusal.role or 'unknown-role'} calls (rule harness-stop-learning): "
        f"{refusal.detail}"
    )
    return Failure(
        FailureKind.HARNESS,
        REFUSAL_STREAK,
        refusal.role,
        refusal.status,
        detail[:500],
        refusal.retries,
    )


class RefusalStreaks:
    """Consecutive refusals per role (a method's learner, J or MIRIX calls).

    :meth:`observe` is told each call's outcome: a refusal extends its role's streak when
    its signature (:func:`refusal_signature`) is the role's last one and starts a new one
    otherwise; a served call (or a failure that is not a refusal) ends it. The
    :data:`REFUSAL_STREAK_ITEMS`-th refusal in a row returns the HARNESS stop
    (:func:`refusal_streak_failure`) for the caller to raise. :meth:`to_dict` is the JSON
    a learning checkpoint keeps (``{role: {"signature", "count"}}``); the constructor reads
    it back, ignoring rows it cannot read.
    """

    def __init__(self, recorded: Mapping[str, Any] | None = None) -> None:
        self._streaks: dict[str, tuple[str, int]] = {}
        if not isinstance(recorded, Mapping):
            return
        for role, row in recorded.items():
            if not isinstance(row, Mapping):
                continue
            signature, count = row.get("signature"), row.get("count")
            if (
                isinstance(signature, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            ):
                self._streaks[str(role)] = (signature, count)

    def observe(
        self, role: str, refusal: Failure | None, signature: str | None = None
    ) -> Failure | None:
        """Record one call of ``role``; the stop failure when the streak reached
        :data:`REFUSAL_STREAK_ITEMS`, else None."""
        if refusal is not None and signature is None:
            signature = refusal_signature(refusal)
        if refusal is None or signature is None:
            self._streaks.pop(role, None)
            return None
        last = self._streaks.get(role)
        count = last[1] + 1 if last is not None and last[0] == signature else 1
        self._streaks[role] = (signature, count)
        if count >= REFUSAL_STREAK_ITEMS:
            return refusal_streak_failure(refusal, count)
        return None

    def count(self, role: str) -> int:
        """The length of ``role``'s current streak (0 without one)."""
        streak = self._streaks.get(role)
        return streak[1] if streak is not None else 0

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            role: {"signature": signature, "count": count}
            for role, (signature, count) in sorted(self._streaks.items())
        }


def _row_failure(row: Mapping[str, Any]) -> Failure | None:
    """The failure dict of a results or learning row, None when absent or unreadable."""
    data = row.get("failure")
    if not isinstance(data, Mapping):
        return None
    try:
        return Failure.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def provider_unserved_row(row: Mapping[str, Any]) -> bool:
    """A results or learning row of an item no provider could serve (pure): ``run.py``'s
    ``provider_unserved`` flag, or the ``end_reason`` or failure code of a row written
    without it. The audit and the report share it."""
    failure = row.get("failure")
    return (
        row.get("provider_unserved") is True
        or row.get("end_reason") == PROVIDER_UNSERVED
        or (isinstance(failure, Mapping) and failure.get("code") == PROVIDER_UNSERVED)
    )


def refused_no_endpoint_try(row: Mapping[str, Any]) -> bool:
    """A results row of a test try left unscored because the provider refused its actor
    call no-endpoint (:func:`upstream_refusal`): what :func:`unserved_retries` counts, and
    what the audit requires before a test item scored ``provider_unserved`` (pure)."""
    failure = _row_failure(row)
    return (
        row.get("score") is None
        and failure is not None
        and failure.role == "actor"
        and upstream_refusal(failure) == REFUSAL_NO_ENDPOINT
    )


def unserved_retries(results: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Test items whose next try is scored unanswered if the provider refuses it
    no-endpoint again (rule gw-404-no-endpoint; ``run.py`` passes
    ``unserved=("no_endpoint",)`` to ``rollout.run_episode`` for them).

    Walking ``results.jsonl`` rows in order, per item (``raw_id``): the trailing tries
    that were unscored because the provider refused the actor no-endpoint
    (:func:`upstream_refusal`); any other row (a scored one, another failure) ends the
    run of tries. Returns the items with at least ``PROVIDER_UNSERVED_TRIES - 1``.
    """
    trailing: dict[str, int] = {}
    for row in results:
        key = row.get("raw_id") or row.get("item")
        if key is None:
            continue
        refused = refused_no_endpoint_try(row)
        trailing[str(key)] = trailing.get(str(key), 0) + 1 if refused else 0
    need = max(1, PROVIDER_UNSERVED_TRIES - 1)
    return frozenset(key for key, n in trailing.items() if n >= need)


def learning_refusal_streak(
    rows: Iterable[Mapping[str, Any]], items: int = REFUSAL_STREAK_ITEMS
) -> Failure | None:
    """The HARNESS stop when the last ``items`` learning positions each ended with the
    actor's call refused the same way (scored ``provider_unserved``, one
    :func:`refusal_signature`), else None (pure; ``run.py`` reads ``learning.jsonl``).

    The latest row of each integer ``position`` counts (a replayed position keeps its
    newest row); the positions must be consecutive.
    """
    latest: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        position = row.get("position")
        if isinstance(position, int) and not isinstance(position, bool):
            latest[position] = row
    if items < 1 or len(latest) < items:
        return None
    positions = sorted(latest)[-items:]
    if positions[-1] - positions[0] != items - 1:
        return None
    signatures: set[str] = set()
    last: Failure | None = None
    for position in positions:
        row = latest[position]
        failure = _row_failure(row)
        if (
            row.get("score") is None
            or failure is None
            or failure.code != PROVIDER_UNSERVED
        ):
            return None
        signature = refusal_signature(failure)
        if signature is None:
            return None
        signatures.add(signature)
        last = failure
    if len(signatures) != 1 or last is None:
        return None
    return refusal_streak_failure(last, items)
