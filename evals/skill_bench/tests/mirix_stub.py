"""A stub MIRIX server for the MIRIX method's tests.

:class:`StubMirix` implements exactly the endpoints ``methods/mirix.py`` calls through
MIRIX's own ``MirixClient``, with MIRIX's request and response shapes
(``mirix/server/rest_api.py``: an agent is a whole ``AgentState``, a retrieval writes a
retrieval log row), and records every request byte for byte. Learning is simulated the
way MIRIX schedules it: ended task sessions are claimed by a learning run once
``threshold`` of them are pending (M1.5) or on ``flush`` (M1.7); a run goes
``queued -> running -> succeeded`` in the background, calls the gateway like MIRIX's
distiller would (the meta agent's ``llm_config`` / ``embedding_config``), and writes one
skill with a version history. ``hold_runs`` keeps runs in flight until :meth:`release`.
A curator batch can return experiences with a counted attempt (``curator_returns``): they
wait for the curator, and a flush with no pending session starts a curate-only run for
them (M1.7), as MIRIX does. A run can delete an earlier skill (``delete_skill_in_run``),
a skill can lack its retrieval vector (``skills_without_vector``, the history's
``has_retrieval_embedding``), and a batch can report the experience pool
(``curator_pool``).

A retrieval calls the gateway like MIRIX's topic extraction and query embedding do and
logs the outcome as MIRIX does (M7.4): a failed chat call is ``extraction_status:
fallback`` with MIRIX's error text (the OpenAI SDK's ``Error code: NNN - {...}``), a failed
embedding call ``embedding_status: failed``; ``extraction_outcomes`` scripts a model
failure of the extraction instead (``TopicExtractionError``, MIRIX's own fallback).

:class:`StubMirixServer` serves one stub on a loopback port (the method's
``server_factory`` handle); :class:`StubMirixServers` keeps each cell's stub across
server restarts, like a cell database, so a test-phase ``resume`` sees the frozen memory.
Rendering is MIRIX's own (``mirix.client.render``), as in a real cell.

Learning checkpoints (``run.py``): :meth:`StubMirixServer.checkpoint_store` is the
stub's :class:`~evals.skill_bench.infra.mirix_server.CheckpointStore`. It dumps the
stub's state (what MIRIX keeps in its database: tenants, agents, sessions, runs, skills,
versions, the retrieval log; never the scripted test controls, which a redo consumes as
a real outage would pass) into the checkpoint, digests it, and a rollback puts the
dumped state back and exports what only the post-crash state had.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import itertools
import json
import pickle
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from evals.skill_bench.infra.mirix_server import (
    CellDatabaseConflict,
    MirixServerError,
    export_summary,
)
from evals.skill_bench.infra.processes import find_free_port
from evals.skill_bench.methods.base import CellContext

#: MIRIX's own LLMConfig / EmbeddingConfig defaults for the fields the method leaves out.
LLM_DEFAULTS = {
    "model_wrapper": None,
    "handle": None,
    "temperature": 0.7,
    "max_tokens": 4096,
    "enable_reasoner": False,
    "reasoning_effort": None,
    "max_reasoning_tokens": 0,
    "put_inner_thoughts_in_kwargs": None,
    "auth_provider": None,
}
EMBEDDING_DEFAULTS = {
    "embedding_chunk_size": 300,
    "handle": None,
    "auth_provider": None,
}
STUB_DISTILLER_PROMPT = "stub MIRIX distiller"
STUB_TOPIC_PROMPT = "stub MIRIX topic extraction"
#: In a stub call's text, asks :func:`with_stub_mirix`'s upstream to fail it (502).
STUB_FAIL_MARKER = "stub: fail upstream"
#: MIRIX's text for a topic extraction the model's answer failed (M7.1: its own fallback).
STUB_MODEL_EXTRACTION_ERROR = (
    "TopicExtractionError: the extraction answer has no usable topic: no tool call"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StubRequest:
    method: str
    path: str
    query: dict[str, str]
    json: Any
    raw: bytes
    client_id: str | None


class RetrieveMemoryRequest(BaseModel):
    """MIRIX's request model with the session-retrieval fields (the preflight reads the
    schema)."""

    user_id: str | None = None
    messages: list[dict[str, Any]]
    limit: int = 10
    local_model_for_retrieval: str | None = None
    memory_types: list[str] | None = None
    session_id: str | None = None


class LegacyRetrieveMemoryRequest(BaseModel):
    """The request model of a MIRIX without the session-retrieval fields."""

    user_id: str | None = None
    messages: list[dict[str, Any]]
    limit: int = 10
    local_model_for_retrieval: str | None = None


class LegacyAddMemoryRequest(BaseModel):
    """MIRIX's ``/memory/add`` request model without ``agent_context``; the
    other fields the client sends are kept as they come (the preflight reads the
    schema's named fields only)."""

    model_config = ConfigDict(extra="allow")

    user_id: str | None = None
    meta_agent_id: str | None = None
    messages: Any = None
    session_id: str | None = None
    session_tag: str | None = None


class AddMemoryRequest(LegacyAddMemoryRequest):
    """MIRIX's ``/memory/add`` request model with ``agent_context``: stored on the
    session, the latest non-blank one winning."""

    agent_context: str | None = None


@dataclass
class StubMirix:
    """In-memory MIRIX: tenants, task sessions, learning runs, skills and their history."""

    threshold: int = 5
    call_gateway: bool = True
    hold_runs: bool = False
    run_delay_s: float = 0.0
    #: An older MIRIX (no retrieval fields, no history endpoint).
    legacy: bool = False
    #: An older MIRIX: ``/memory/add`` has no ``agent_context`` field.
    no_agent_context: bool = False
    #: (METHOD, path) -> HTTP status to answer once, before handling the request.
    fail_once: dict[tuple[str, str], int] = field(default_factory=dict)
    #: The reply of /memory/add for task sessions (tests of an old MIRIX's "queued").
    add_status: str = "stored"
    #: LLMConfig fields this MIRIX does not know and drops (an older MIRIX).
    drop_llm_fields: tuple[str, ...] = ()
    #: session id -> the outcomes of its successive distillations (M4.4 ``status``):
    #: ``ok`` (the default), ``failed`` (a counted failure: pending again), ``deferred``
    #: (an infrastructure back-off: not pending any more) or ``failed_permanently``.
    session_outcomes: dict[str, list[str]] = field(default_factory=dict)
    #: session id -> the ``failure_kind`` and ``error`` its ``deferred`` distillations
    #: record (default ``infra`` and an upstream 502), e.g. ``config`` and the refusal
    #: MIRIX defers across runs (M3.5: 401/403/404).
    deferred_errors: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Whether each of the next flushes may start a run, one per flush: ``False`` starts
    #: none even with sessions pending, a MIRIX whose flush does not take up a session
    #: with a counted failure. Once empty, every flush claims what is pending.
    flush_claims: list[bool] = field(default_factory=list)
    #: Whether MIRIX's sweeper claims what is pending just before each of the next
    #: flushes, one per flush (a ``sweep`` run, whatever the threshold): the flush then
    #: finds nothing to claim.
    sweep_before_flush: list[bool] = field(default_factory=list)
    #: Curator ``error_kind`` of the next runs' batch (e.g. ``infra``), one per run; ``""``
    #: is a run whose batch succeeds (its ``curator_error_texts`` entry is skipped too).
    curator_errors: list[str] = field(default_factory=list)
    #: Experiences the next runs' curator batch returns with a counted attempt (M5.4),
    #: one count per run: they wait for the curator, and the next flush curates them.
    curator_returns: list[int] = field(default_factory=list)
    #: run number (1-based) -> the id of an earlier skill that run's curator deletes.
    delete_skill_in_run: dict[int, str] = field(default_factory=dict)
    #: The experience-pool keys of the next runs' curator batch entry, one
    #: mapping per run (``dispositions``, ``history_skipped``, ``history``,
    #: ``pool_active_after`` ...), as a MIRIX with the pool reports them. A batch without
    #: one reports no pool, as a MIRIX without the pool does.
    curator_pool: list[dict[str, Any]] = field(default_factory=list)
    #: Skills whose history says ``has_retrieval_embedding: false``.
    skills_without_vector: set[str] = field(default_factory=set)
    #: Outcomes of the next retrievals' topic extraction, one per retrieval, before the
    #: gateway's answer is looked at: ``model`` (the model's answer is unusable: MIRIX's
    #: own fallback, ``TopicExtractionError``). Otherwise the gateway call decides.
    extraction_outcomes: list[str] = field(default_factory=list)
    #: Write no retrieval log row (MIRIX_RETRIEVAL_LOG_ENABLED=false): ``retrieval_id`` null.
    retrieval_log_disabled: bool = False
    #: ``{"mirix" | "embedding": n}``: once the memory is frozen (a flush run exists),
    #: the next ``n`` retrievals send that call with :data:`STUB_FAIL_MARKER`, which
    #: :func:`with_stub_mirix`'s upstream answers with a 502 (every gateway retry too).
    degrade_test_retrievals: dict[str, int] = field(default_factory=dict)
    #: The same before the memory is frozen: the next ``n`` learning-phase retrievals
    #: lose that call (a degraded learning retrieval is injected).
    degrade_learning_retrievals: dict[str, int] = field(default_factory=dict)
    #: The ``error`` of the next ``curator_errors`` batches, one per batch (e.g. the
    #: text of a gateway refusal MIRIX recorded); ``"stub failure"`` once empty.
    curator_error_texts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requests: list[StubRequest] = []
        self.orgs: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, dict[str, Any]] = {}
        self.users: dict[str, dict[str, Any]] = {}
        self.agents: dict[str, dict[str, Any]] = {}
        #: agent id -> the client that owns it (MIRIX keeps it in its database; an
        #: ``AgentState`` has no such field).
        self.agent_clients: dict[str, str] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, Any]] = {}
        self.history: dict[str, list[dict[str, Any]]] = {}
        #: skill id -> version_no -> the stored content of that version (M6.1).
        self.versions: dict[str, dict[int, dict[str, Any]]] = {}
        self.retrieval_logs: list[dict[str, Any]] = []
        self.gateway_calls: list[dict[str, Any]] = []
        #: Skill ids a run's curator deleted (their history stays, M6.2).
        self.deleted: set[str] = set()
        #: (user, agent) -> experiences waiting for the curator.
        self.waiting_experiences: dict[tuple[str, str], int] = {}
        self._ids = itertools.count(1)
        self._released = threading.Event()
        if not self.hold_runs:
            self._released.set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.app = self._build_app()

    # --- the "database" (learning checkpoints) ---------------------------------------

    #: What MIRIX keeps in its database: a learning checkpoint dumps exactly these.
    STATE_FIELDS = (
        "orgs",
        "clients",
        "users",
        "agents",
        "agent_clients",
        "sessions",
        "runs",
        "skills",
        "history",
        "versions",
        "retrieval_logs",
        "deleted",
        "waiting_experiences",
    )

    def _next_id(self) -> int:
        value = next(self._ids)
        self._ids = itertools.count(value)
        return value

    def dump_state(self) -> dict[str, Any]:
        """A deep copy of the stub's database state (and its id counter)."""
        state = {name: copy.deepcopy(getattr(self, name)) for name in self.STATE_FIELDS}
        state["next_id"] = self._next_id()
        return state

    def load_state(self, state: Mapping[str, Any] | None) -> None:
        """Replace the database state (None: empty, a fresh database)."""
        if state is None:
            empty: dict[str, Any] = {"deleted": set(), "retrieval_logs": []}
            for name in self.STATE_FIELDS:
                setattr(self, name, empty.get(name, {}))
            self._ids = itertools.count(1)
            return
        for name in self.STATE_FIELDS:
            setattr(self, name, copy.deepcopy(state[name]))
        self._ids = itertools.count(int(state["next_id"]))

    def state_digest(self) -> str:
        """sha256 of the database state in a canonical form (sets sorted, tuple keys
        joined): equal exactly when the state is."""

        def plain(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    "|".join(map(str, k)) if isinstance(k, tuple) else str(k): plain(v)
                    for k, v in value.items()
                }
            if isinstance(value, (set, frozenset)):
                return sorted(plain(v) for v in value)
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        data = json.dumps(
            plain(self.dump_state()), sort_keys=True, ensure_ascii=False, default=repr
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    # --- test controls -------------------------------------------------------------

    def release(self) -> None:
        self._released.set()

    def active_runs(self) -> list[dict[str, Any]]:
        return [r for r in self.runs.values() if r["status"] in ("queued", "running")]

    def writes(self) -> list[StubRequest]:
        """Requests that change MIRIX state (retrieval only writes its log)."""
        return [
            r
            for r in self.requests
            if r.method != "GET" and r.path != "/memory/retrieve/conversation"
        ]

    # --- simulated learning --------------------------------------------------------

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids):04d}"

    def _pending(self, user_id: str, agent_id: str) -> list[str]:
        return [
            sid
            for sid, s in self.sessions.items()
            if s["user_id"] == user_id
            and s["agent_id"] == agent_id
            and s["tag"] == "task"
            and s["closed_at"] is not None
            and s["claimed_by"] is None
            and not s["distilled"]
            and not s.get("deferred")
        ]

    def _claim(
        self, user_id: str, agent_id: str, trigger: str, *, threshold_met: bool = False
    ) -> str | None:
        pending = self._pending(user_id, agent_id)
        waiting = self.waiting_experiences.get((user_id, agent_id), 0)
        curate_only = trigger == "flush" and not pending and waiting > 0
        below = (
            trigger != "flush" and not threshold_met and len(pending) < self.threshold
        )
        if not curate_only and (not pending or below):
            return None
        run_id = self._new_id("slr")
        for sid in pending:
            self.sessions[sid]["claimed_by"] = run_id
        self.runs[run_id] = {
            "id": run_id,
            "organization_id": self.agents[agent_id]["organization_id"],
            "user_id": user_id,
            "agent_id": agent_id,
            "trigger": trigger,
            "status": "queued",
            "session_ids": sorted(pending),
            "session_watermarks": {
                sid: len(self.sessions[sid]["messages"]) for sid in pending
            },
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "distill_metrics": [],
            "curator_metrics": [],
        }
        task = asyncio.get_running_loop().create_task(self._execute(run_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run_id

    def _frozen(self) -> bool:
        return any(run["trigger"] == "flush" for run in self.runs.values())

    def _degrade(self) -> str | None:
        """The role this retrieval's call fails on (``degrade_test_retrievals`` once the
        memory is frozen, ``degrade_learning_retrievals`` before)."""
        countdown = (
            self.degrade_test_retrievals
            if self._frozen()
            else self.degrade_learning_retrievals
        )
        for role, left in countdown.items():
            if left > 0:
                countdown[role] = left - 1
                return role
        return None

    async def _gateway_calls(
        self, agent: Mapping[str, Any], text: str, *, fail: str | None = None
    ) -> tuple[httpx.Response, httpx.Response] | None:
        """One chat and one embedding call through the gateway, as MIRIX makes them;
        ``fail`` (a role) marks that call for the upstream to fail."""
        if not self.call_gateway:
            return None
        llm, emb = agent["llm_config"], agent["embedding_config"]
        user = STUB_FAIL_MARKER if fail == "mirix" else "stub"
        query = f"{text} {STUB_FAIL_MARKER}" if fail == "embedding" else text
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            chat = await client.post(
                f"{llm['model_endpoint']}/chat/completions",
                headers={"Authorization": f"Bearer {llm.get('api_key')}"},
                json={
                    "model": llm["model"],
                    "messages": [
                        {"role": "system", "content": text},
                        {"role": "user", "content": user},
                    ],
                },
            )
            embedding = await client.post(
                f"{emb['embedding_endpoint']}/embeddings",
                headers={"Authorization": f"Bearer {emb.get('api_key')}"},
                json={"input": query, "model": emb["embedding_model"]},
            )
        self.gateway_calls.append(
            {"chat": chat.status_code, "embedding": embedding.status_code, "text": text}
        )
        return chat, embedding

    async def _gateway(
        self,
        agent: Mapping[str, Any],
        text: str,
        usage: dict[str, Any] | None = None,
    ) -> str | None:
        """The calls of :meth:`_gateway_calls`; the error of a failed one, else None.

        ``usage`` receives the chat call's ``llm_usage`` (MIRIX aggregates it on
        the learning run, which the audit reconciles with the gateway ledger)."""
        answered = await self._gateway_calls(agent, text)
        if answered is None:
            return None
        chat, embedding = answered
        if chat.status_code != 200 or embedding.status_code != 200:
            return f"gateway answered {chat.status_code} / {embedding.status_code}"
        if usage is not None:
            reply = (chat.json() or {}).get("usage") or {}
            totals = {
                "calls": 1,
                "prompt_tokens": int(reply.get("prompt_tokens") or 0),
                "completion_tokens": int(reply.get("completion_tokens") or 0),
                "total_tokens": int(reply.get("total_tokens") or 0),
            }
            usage.update(totals, by_stage={"distill": dict(totals)})
        return None

    async def _execute(self, run_id: str) -> None:
        run = self.runs[run_id]
        while not self._released.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(self.run_delay_s)
        run.update(status="running", started_at=_now())
        usage: dict[str, Any] = {}
        error = await self._gateway(
            self.agents[run["agent_id"]], STUB_DISTILLER_PROMPT, usage
        )
        if error is not None:
            for sid in run["session_ids"]:
                self.sessions[sid]["claimed_by"] = None
            run.update(status="failed", error=error, finished_at=_now())
            return
        scope = (run["user_id"], run["agent_id"])
        n = len(self.skills) + len(self.deleted) + 1
        skill_id = f"proc-{n:04d}"
        self.skills[skill_id] = {
            "memory_type": "procedural",
            "id": skill_id,
            "entry_type": "workflow",
            "name": f"stub skill {n}",
            "description": f"learned from {len(run['session_ids'])} task sessions",
            "instructions": "1. look around\n2. act",
            "triggers": ["a task like the ones seen"],
            "examples": [],
            "version": 1,
            "created_at": _now(),
            "updated_at": _now(),
        }
        self.history[skill_id] = [
            {"version_no": 1, "change_type": "create", "run_id": f"{run_id}:b0"}
        ]
        self.versions[skill_id] = {1: version_content(self.skills[skill_id])}
        metrics = []
        for sid in run["session_ids"]:
            queue = self.session_outcomes.get(sid) or []
            outcome = queue.pop(0) if queue else "ok"
            session = self.sessions[sid]
            session["claimed_by"] = None
            if outcome in ("ok", "failed_permanently"):
                session["distilled"] = True
            elif outcome == "deferred":
                session["deferred"] = True  # distill_retry_after: not pending for now
            entry: dict[str, Any] = {
                "session_id": sid,
                "status": outcome,
                "experiences": 1 if outcome == "ok" else 0,
            }
            if outcome == "deferred":
                kind, error = self.deferred_errors.get(
                    sid, ("infra", "stub: upstream 502")
                )
                entry.update(failure_kind=kind, error=error)
            metrics.append(entry)
        changes: dict[str, list[str]] = {
            "created": [skill_id],
            "edited": [],
            "deleted": [],
        }
        doomed = self.delete_skill_in_run.get(len(self.runs))
        if doomed is not None and doomed in self.skills:
            del self.skills[doomed]
            self.deleted.add(doomed)
            self.history[doomed].append(
                {
                    "version_no": len(self.history[doomed]) + 1,
                    "change_type": "delete",
                    "run_id": f"{run_id}:b0",
                }
            )
            self.versions[doomed][len(self.history[doomed])] = self.versions[doomed][1]
            changes["deleted"].append(doomed)
        batch: dict[str, Any] = {
            "batch_id": f"{run_id}:b0",
            "writes": 1,
            "changes": changes,
            "outcome": "consume",
        }
        if self.curator_pool:
            batch.update(copy.deepcopy(self.curator_pool.pop(0)))
        # The experiences this run distilled (one per ``ok`` session) and those an
        # earlier batch returned: this batch returns ``curator_returns`` of them.
        waiting = self.waiting_experiences.get(scope, 0) + sum(
            e["experiences"] for e in metrics
        )
        returned = (
            min(waiting, self.curator_returns.pop(0)) if self.curator_returns else 0
        )
        self.waiting_experiences[scope] = returned
        if returned:
            batch.update(outcome="return", returned=returned)
        status, run_error = "succeeded", None
        if self.curator_errors:
            # MIRIX keeps the batch pending (M3.5); its own failure (a bug, a request
            # it could not build) also fails the run, with the error recorded.
            kind = self.curator_errors.pop(0)
            error = (
                self.curator_error_texts.pop(0)
                if self.curator_error_texts
                else "stub failure"
            )
            if kind:  # "" is a run whose batch succeeds
                batch.update(error_kind=kind, error=error, outcome="keep")
            if kind in ("internal", "other"):
                status, run_error = "failed", "KeyError: 'stub curator bug'"
        run.update(
            status=status,
            error=run_error,
            finished_at=_now(),
            distill_metrics=metrics,
            curator_metrics=[batch],
            llm_usage=usage or None,
        )

    # --- the app ---------------------------------------------------------------------

    def _build_app(self) -> FastAPI:  # noqa: C901 - one route per MIRIX endpoint
        app = FastAPI()
        stub = self

        @app.middleware("http")
        async def record(request: Request, call_next: Any) -> Response:
            raw = await request.body()
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                body = None
            path = request.url.path
            if path != "/openapi.json":
                stub.requests.append(
                    StubRequest(
                        request.method,
                        path,
                        dict(request.query_params),
                        body,
                        raw,
                        request.headers.get("x-client-id"),
                    )
                )
            status = stub.fail_once.pop((request.method, path), None)
            if status is not None:
                return JSONResponse({"detail": "injected failure"}, status)
            return await call_next(request)

        def client_of(x_client_id: str | None) -> dict[str, Any]:
            if x_client_id not in stub.clients:
                raise _Http(404, f"Client {x_client_id} not found")
            return stub.clients[x_client_id]

        @app.exception_handler(_Http)
        async def http_error(_request: Request, exc: Exception) -> Response:
            assert isinstance(exc, _Http)
            return JSONResponse({"detail": exc.detail}, exc.status)

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"status": "healthy"}

        @app.post("/organizations/create_or_get")
        async def org(body: dict[str, Any]) -> dict[str, Any]:
            org_id = body["org_id"]
            return stub.orgs.setdefault(
                org_id, {"id": org_id, "name": body.get("name")}
            )

        @app.post("/clients/create_or_get")
        async def client(body: dict[str, Any]) -> dict[str, Any]:
            cid = body["client_id"]
            return stub.clients.setdefault(
                cid,
                {
                    "id": cid,
                    "name": body.get("name"),
                    "organization_id": body.get("org_id"),
                    "write_scope": body.get("write_scope"),
                    "read_scopes": body.get("read_scopes"),
                    "status": body.get("status"),
                },
            )

        @app.post("/users/create_or_get")
        async def create_user(
            body: dict[str, Any], x_client_id: str | None = Header(None)
        ) -> dict[str, Any]:
            owner = client_of(x_client_id)
            uid = body["user_id"]
            return stub.users.setdefault(
                uid,
                {
                    "id": uid,
                    "name": body.get("name"),
                    "organization_id": owner["organization_id"],
                },
            )

        @app.get("/users/{user_id}")
        async def get_user(user_id: str) -> dict[str, Any]:
            if user_id not in stub.users:
                raise _Http(404, f"User {user_id} not found")
            return stub.users[user_id]

        @app.post("/agents/meta/initialize")
        async def initialize(
            body: dict[str, Any], x_client_id: str | None = Header(None)
        ) -> Any:
            owner = client_of(x_client_id)
            if not owner.get("write_scope"):
                return None
            mine = [
                a
                for a in stub.agents.values()
                if stub.agent_clients[a["id"]] == owner["id"]
            ]
            if mine:
                return mine[0]
            config = body["config"]
            agent_id = stub._new_id("agent")
            agent = {
                "id": agent_id,
                "name": "meta_memory_agent",
                "agent_type": "meta_memory_agent",
                "system": "stub meta memory agent",
                "tools": [],
                "organization_id": owner["organization_id"],
                "llm_config": {
                    k: v
                    for k, v in {**LLM_DEFAULTS, **config["llm_config"]}.items()
                    if k not in stub.drop_llm_fields
                },
                "embedding_config": {
                    **EMBEDDING_DEFAULTS,
                    **config["embedding_config"],
                },
            }
            stub.agents[agent_id] = agent
            stub.agent_clients[agent_id] = owner["id"]
            return agent

        @app.get("/agents")
        async def agents(
            x_client_id: str | None = Header(None),
        ) -> list[dict[str, Any]]:
            owner = client_of(x_client_id)
            return [
                a
                for a in stub.agents.values()
                if stub.agent_clients[a["id"]] == owner["id"]
            ]

        async def add(body: dict[str, Any], x_client_id: str | None) -> Any:
            client_of(x_client_id)
            if body.get("meta_agent_id") not in stub.agents:
                raise _Http(404, "meta agent not found")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise _Http(422, "messages must be a non-empty list")
            sid, tag = body.get("session_id"), body.get("session_tag") or "conversation"
            if tag == "task" and not sid:
                raise _Http(422, "session_tag 'task' requires a session_id")
            agent_context = body.get("agent_context")
            if isinstance(agent_context, str) and agent_context.strip() and not sid:
                raise _Http(422, "agent_context needs a session_id")
            session = stub.sessions.get(sid) if sid else None
            warnings = []
            if session is None and sid:
                session = stub.sessions[sid] = {
                    "user_id": body.get("user_id"),
                    "agent_id": body["meta_agent_id"],
                    "tag": tag,
                    "messages": [],
                    "closed_at": None,
                    "claimed_by": None,
                    "distilled": False,
                    "reopen_count": 0,
                }
            elif session is not None and session["tag"] != tag:
                warnings.append(f"session_tag mismatch: {tag} vs {session['tag']}")
            if session is not None:
                if session["closed_at"] is not None:  # Q1: a write reopens the session
                    session["closed_at"] = None
                    session["reopen_count"] += 1
                session["messages"].extend(copy.deepcopy(messages))
                if isinstance(agent_context, str) and agent_context.strip():
                    session["agent_context"] = agent_context  # the latest one wins
            return {
                "success": True,
                "message": "Stored in the session conversation store for skill learning",
                "status": stub.add_status,
                "agent_id": body["meta_agent_id"],
                "session_id": sid,
                "message_count": len(messages),
                "warnings": warnings,
                "learning_run_id": None,
            }

        async def retrieve(body: dict[str, Any], x_client_id: str | None) -> Any:
            owner = client_of(x_client_id)
            agent = next(
                a
                for a in stub.agents.values()
                if stub.agent_clients[a["id"]] == owner["id"]
            )
            answered = await stub._gateway_calls(
                agent, STUB_TOPIC_PROMPT, fail=stub._degrade()
            )
            outcome = (
                stub.extraction_outcomes.pop(0) if stub.extraction_outcomes else None
            )
            extraction_status, extraction_error = "ok", None
            embedding_status, embedding_error = "ok", None
            if outcome == "model":
                extraction_status = "fallback"
                extraction_error = STUB_MODEL_EXTRACTION_ERROR
            if answered is not None:
                chat, embedding = answered
                if chat.status_code != 200 and extraction_status == "ok":
                    extraction_status = "fallback"
                    extraction_error = openai_sdk_error(chat)
                if embedding.status_code != 200:
                    embedding_status = "failed"
                    embedding_error = embedding_http_error(embedding)
            items = sorted(stub.skills.values(), key=lambda s: s["id"])[
                : body.get("limit", 10)
            ]
            items = [
                {**copy.deepcopy(s), "retrieval_scores": {"bm25": 1.0, "cosine": 0.9}}
                for s in items
            ]
            retrieval_id = None
            if not stub.retrieval_log_disabled:
                retrieval_id = stub._new_id("rlog")
                extraction: dict[str, Any] = {"calls": 1, "status": extraction_status}
                if extraction_error:
                    extraction["error"] = extraction_error
                stub.retrieval_logs.append(
                    {
                        "id": retrieval_id,
                        "user_id": body.get("user_id"),
                        "client_id": owner["id"],
                        "session_id": body.get("session_id"),
                        "memory_types": body.get("memory_types"),
                        "extracted_topics": None
                        if extraction_status == "fallback"
                        else "stub topics",
                        "extraction_status": extraction_status,
                        "embedding_status": embedding_status,
                        "per_type": {
                            "procedural": {
                                "embedding_status": embedding_status,
                                "embedding_error": embedding_error,
                                "returned_ids": [s["id"] for s in items],
                            }
                        },
                        "llm_usage": {"topic_extraction": extraction},
                        "retry_events": [],
                        "created_at": _now(),
                    }
                )
            return {
                "success": True,
                "retrieval_id": retrieval_id,
                "topics": None if extraction_status == "fallback" else "stub topics",
                "procedural_query": (
                    None if extraction_status == "fallback" else "stub task"
                ),
                "extraction_status": extraction_status,
                "temporal_expression": None,
                "date_range": None,
                "memories": {
                    "procedural": {"total_count": len(stub.skills), "items": items}
                },
                "errors": {},
            }

        if self.no_agent_context:

            @app.post("/memory/add")
            async def add_before_p16(
                request: LegacyAddMemoryRequest, x_client_id: str | None = Header(None)
            ) -> Any:
                return await add(request.model_dump(), x_client_id)

        else:

            @app.post("/memory/add")
            async def add_p16(
                request: AddMemoryRequest, x_client_id: str | None = Header(None)
            ) -> Any:
                return await add(request.model_dump(), x_client_id)

        if self.legacy:

            @app.post("/memory/retrieve/conversation")
            async def retrieve_legacy(
                request: LegacyRetrieveMemoryRequest,
                x_client_id: str | None = Header(None),
            ) -> Any:
                return await retrieve(request.model_dump(), x_client_id)

        else:

            @app.post("/memory/retrieve/conversation")
            async def retrieve_m71(
                request: RetrieveMemoryRequest, x_client_id: str | None = Header(None)
            ) -> Any:
                return await retrieve(request.model_dump(), x_client_id)

        @app.post("/memory/sessions/{session_id}/close")
        async def close(session_id: str, body: dict[str, Any]) -> Any:
            session = stub.sessions.get(session_id)
            if session is None or session["user_id"] != body.get("user_id"):
                raise _Http(404, f"Session {session_id} not found")
            if session["closed_at"] is not None:
                return {
                    "session_id": session_id,
                    "closed_at": session["closed_at"],
                    "close_reason": "explicit",
                    "already_closed": True,
                    "learning_run_id": None,
                }
            session["closed_at"] = _now()
            run_id = stub._claim(session["user_id"], session["agent_id"], "close")
            return {
                "session_id": session_id,
                "closed_at": session["closed_at"],
                "close_reason": "explicit",
                "already_closed": False,
                "learning_run_id": run_id,
            }

        @app.post("/memory/skills/flush")
        async def flush(body: dict[str, Any]) -> Any:
            user_id, agent_id = body.get("user_id"), body.get("meta_agent_id")
            if user_id not in stub.users or agent_id not in stub.agents:
                raise _Http(404, "not found")
            closed = []
            if body.get("close_open_sessions"):
                for sid, s in stub.sessions.items():
                    if s["user_id"] == user_id and s["closed_at"] is None:
                        s["closed_at"] = _now()
                        closed.append(sid)
            if stub.sweep_before_flush and stub.sweep_before_flush.pop(0):
                stub._claim(user_id, agent_id, "sweep", threshold_met=True)
            count = len(stub._pending(user_id, agent_id))
            waiting = stub.waiting_experiences.get((user_id, agent_id), 0)
            claims = stub.flush_claims.pop(0) if stub.flush_claims else True
            run_id = stub._claim(user_id, agent_id, "flush") if claims else None
            return {
                "learning_run_id": run_id,
                "session_count": count,
                "closed_session_ids": closed,
                "pending_experiences": waiting,
            }

        @app.get("/memory/learning_runs")
        async def runs(
            user_id: str | None = None,
            status: str | None = None,
            limit: int = 50,
            meta_agent_id: str | None = None,
        ) -> Any:
            selected = [
                r
                for r in stub.runs.values()
                if (user_id is None or r["user_id"] == user_id)
                and (meta_agent_id is None or r["agent_id"] == meta_agent_id)
                and (
                    status is None
                    or r["status"] == status
                    or (status == "active" and r["status"] in ("queued", "running"))
                )
            ]
            selected.sort(key=lambda r: r["created_at"], reverse=True)
            return {"learning_runs": copy.deepcopy(selected[:limit])}

        @app.get("/memory/learning_runs/{run_id}")
        async def run(run_id: str) -> Any:
            if run_id not in stub.runs:
                raise _Http(404, f"Learning run {run_id} not found")
            return stub.runs[run_id]

        @app.get("/memory/components")
        async def components(
            user_id: str, memory_type: str = "all", limit: int = 1000
        ) -> Any:
            items = sorted(stub.skills.values(), key=lambda s: s["id"])[:limit]
            return {
                "success": True,
                "user_id": user_id,
                "memory_type": memory_type,
                "limit": limit,
                "memories": {
                    "procedural": {
                        "total_count": len(stub.skills),
                        "items": copy.deepcopy(items),
                    }
                },
            }

        if not self.legacy:

            @app.get("/memory/skills/{skill_id}/history")
            async def history(skill_id: str, user_id: str | None = None) -> Any:
                if skill_id not in stub.history:
                    raise _Http(404, f"Skill {skill_id} not found")
                return {
                    "skill_id": skill_id,
                    "name": (stub.skills.get(skill_id) or {}).get("name"),
                    "is_deleted": skill_id in stub.deleted,
                    "has_retrieval_embedding": skill_id
                    not in stub.skills_without_vector,
                    "versions": copy.deepcopy(stub.history[skill_id]),
                }

            @app.get("/memory/skills/{skill_id}/versions/{version_no}")
            async def version(
                skill_id: str, version_no: int, user_id: str | None = None
            ) -> Any:
                stored = (stub.versions.get(skill_id) or {}).get(version_no)
                if stored is None:
                    raise _Http(404, f"Version {version_no} of {skill_id} not found")
                entry = next(
                    v for v in stub.history[skill_id] if v["version_no"] == version_no
                )
                return {
                    "skill_id": skill_id,
                    "version": copy.deepcopy(entry),
                    "content": copy.deepcopy(stored),
                }

            @app.get("/memory/retrieval_logs")
            async def retrieval_logs(
                session_id: str | None = None,
                user_id: str | None = None,
                limit: int = 50,
                x_client_id: str | None = Header(None),
            ) -> Any:
                owner = client_of(x_client_id)
                rows = [
                    r
                    for r in reversed(stub.retrieval_logs)
                    if r["client_id"] == owner["id"]
                    and (session_id is None or r["session_id"] == session_id)
                    and (user_id is None or r["user_id"] == user_id)
                ]
                return {"retrieval_logs": copy.deepcopy(rows[:limit])}

        return app


class _Http(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


class StubMirixServer:
    """Serves one :class:`StubMirix` on a loopback port (a ``server_factory`` handle)."""

    def __init__(self, stub: StubMirix, mode: str = "create") -> None:
        self.stub = stub
        self.mode = mode
        self.port = find_free_port()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.stopped = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        await asyncio.to_thread(self._start)

    def _start(self) -> None:
        import uvicorn

        self._server = uvicorn.Server(
            uvicorn.Config(
                self.stub.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                timeout_graceful_shutdown=1,
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise MirixServerError("stub MIRIX did not start")
            time.sleep(0.01)

    async def stop(self, grace_s: float = 20.0) -> int | None:
        if self._server is not None and not self.stopped:
            self._server.should_exit = True
            if self._thread is not None:
                await asyncio.to_thread(self._thread.join, 10)
        self.stopped = True
        return 0

    async def ensure_alive(self) -> None:
        if self.stopped or self._thread is None or not self._thread.is_alive():
            raise MirixServerError("stub MIRIX server is not running")

    def checkpoint_store(self) -> StubCheckpointStore:
        """The stub's "database" for ``run.py``'s learning checkpoints (one per stub, so
        a test's hooks on it hold across server restarts)."""
        store = getattr(self.stub, "_checkpoint_store", None)
        if store is None:
            store = StubCheckpointStore(self.stub)
            self.stub._checkpoint_store = store  # type: ignore[attr-defined]
        return store


#: The dump file of a stub's state in a learning checkpoint.
STUB_DUMP_FILE = "stub_mirix_state.pickle"


class StubCheckpointStore:
    """The stub's "database" as a :class:`~evals.skill_bench.infra.mirix_server.
    CheckpointStore` (``run.py`` learning checkpoints)."""

    def __init__(self, stub: StubMirix, name: str = "stub_mirix") -> None:
        self.stub = stub
        self._name = name
        #: Test hooks: extra connections :meth:`connections` reports, and a digest
        #: :meth:`digest` returns instead of the real one (a restore that went wrong).
        self.extra_connections = 0
        self.digest_override: dict[str, Any] | None = None
        #: Test hook: the stub's "database" is another run's (B02), so
        #: :meth:`check_owner` and :meth:`rollback` raise ``CellDatabaseConflict`` with
        #: this text; and what they were asked (``owner_checks``: the claims and
        #: segments; ``rollbacks``: the segments rolled back).
        self.owner_conflict: str | None = None
        self.owner_checks: list[tuple[bool, int | None]] = []
        self.rollbacks: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    async def check_available(self) -> None:
        return None

    async def check_owner(
        self, *, claim_unmarked: bool = False, segment: int | None = None
    ) -> dict[str, Any]:
        self.owner_checks.append((claim_unmarked, segment))
        if self.owner_conflict is not None:
            raise CellDatabaseConflict(self.owner_conflict)
        return {
            "database": self._name,
            "owner": None,
            "states": {self._name: "ours"},
            "claimed": [],
        }

    async def connections(self) -> int:
        return self.extra_connections

    async def active_runs(self) -> int:
        return len(self.stub.active_runs())

    def _digest(self) -> dict[str, Any]:
        return {"stub": [len(self.stub.sessions), self.stub.state_digest()]}

    async def checkpoint(self, out_dir: Path) -> dict[str, Any]:
        data = pickle.dumps(self.stub.dump_state())
        path = out_dir / STUB_DUMP_FILE
        await asyncio.to_thread(path.write_bytes, data)
        return {
            "db_name": self._name,
            "dump_sha256": hashlib.sha256(data).hexdigest(),
            "dump_bytes": len(data),
            "db_digest": self._digest(),
            "active_runs": len(self.stub.active_runs()),
            "snapshot_id": None,
        }

    async def digest(self) -> dict[str, Any]:
        if self.digest_override is not None:
            return self.digest_override
        return self._digest()

    async def rollback(
        self,
        segment: int,
        checkpoint_dir: Path | None,
        export_path: Path,
        *,
        keep_discarded: bool = False,
        claim_unmarked: bool = False,
    ) -> dict[str, Any]:
        owners = await self.check_owner(claim_unmarked=claim_unmarked, segment=segment)
        self.rollbacks.append(segment)
        before = self.stub.dump_state()
        restored = (
            pickle.loads((checkpoint_dir / STUB_DUMP_FILE).read_bytes())
            if checkpoint_dir is not None
            else None
        )
        self.stub.load_state(restored)
        after = restored or {"runs": {}, "sessions": {}, "retrieval_logs": []}
        kept_logs = {r["id"] for r in after["retrieval_logs"]}
        rows = {
            "retrieval_log": [
                {
                    "id": r["id"],
                    "session_id": r.get("session_id"),
                    "extraction_status": r.get("extraction_status"),
                    "embedding_status": r.get("embedding_status"),
                }
                for r in before["retrieval_logs"]
                if r["id"] not in kept_logs
            ],
            "skill_learning_run": [
                {
                    "id": run_id,
                    "status": run.get("status"),
                    "trigger": run.get("trigger"),
                    "session_ids": run.get("session_ids"),
                    "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at"),
                }
                for run_id, run in before["runs"].items()
                if run_id not in after["runs"]
            ],
            "conversation_session": [
                {
                    "id": sid,
                    "session_id": sid,
                    "message_count": len(session.get("messages") or ()),
                    "closed_at": session.get("closed_at"),
                }
                for sid, session in before["sessions"].items()
                if sid not in after["sessions"]
            ],
        }
        changed = sorted(
            sid
            for sid, session in before["sessions"].items()
            if sid in after["sessions"] and session != after["sessions"][sid]
        )
        export = {
            "discarded_db": f"{self._name}_d{segment:02d}",
            "live_db": self._name if checkpoint_dir is not None else None,
            "tables": {
                "conversation_session": {
                    "added": len(rows["conversation_session"]),
                    "removed": 0,
                    "changed": changed,
                }
            },
            "rows": rows,
        }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(export, sort_keys=True, default=str))
        return {
            **export_summary(export),
            "post_crash_db": f"kept:{export['discarded_db']}"
            if keep_discarded
            else "dropped",
            "owner": owners,
        }


class StubMirixServers:
    """``server_factory`` keeping one :class:`StubMirix` per cell directory (its "database").

    ``fresh`` refuses a cell that already has one; ``resume`` refuses one that does not,
    as ``infra/mirix_server.py`` does.
    """

    def __init__(self, **stub_options: Any) -> None:
        self.stub_options = stub_options
        self.stubs: dict[Path, StubMirix] = {}
        self.servers: list[StubMirixServer] = []

    def __call__(self, cell: CellContext, mode: str) -> StubMirixServer:
        key = Path(cell.cell_dir or cell.work_dir)
        if mode == "fresh":
            if key in self.stubs:
                raise MirixServerError(f"{key} already has a MIRIX database")
            self.stubs[key] = StubMirix(**self.stub_options)
        elif key not in self.stubs:
            raise MirixServerError(f"{key} has no MIRIX database to resume")
        server = StubMirixServer(self.stubs[key], mode)
        self.servers.append(server)
        return server

    def stub_for(self, cell_dir: Path) -> StubMirix:
        return self.stubs[Path(cell_dir)]


def version_content(skill: Mapping[str, Any]) -> dict[str, Any]:
    """The seven versioned fields of a skill, as MIRIX stores a version (M6.1)."""
    fields = (
        "name",
        "entry_type",
        "description",
        "instructions",
        "triggers",
        "examples",
    )
    return {
        **{name: copy.deepcopy(skill.get(name)) for name in fields},
        "filter_tags": {"scope": "skill_bench"},
    }


def openai_sdk_error(response: httpx.Response) -> str:
    """MIRIX's log text for a topic-extraction call the gateway failed: the OpenAI SDK's
    ``APIStatusError`` wrapped in MIRIX's ``LLMServerError`` / ``LLMBadRequestError``
    (``mirix/llm_api/openai_client.py``), cut to 500 characters like MIRIX's."""
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    name = "LLMServerError" if response.status_code >= 500 else "LLMBadRequestError"
    code = (
        "INTERNAL_SERVER_ERROR" if response.status_code >= 500 else "INVALID_ARGUMENT"
    )
    text = f"{name}: {code}: OpenAI API error: Error code: {response.status_code} - {body!r}"
    return text[:500]


def embedding_http_error(response: httpx.Response) -> str:
    """MIRIX's log text for a query embedding the gateway failed
    (``mirix/embeddings.py`` ``raise_for_embedding_status``), cut like MIRIX's."""
    url = str(response.request.url).split("?", 1)[0]
    text = (
        f"HTTPStatusError: OpenRouter embeddings endpoint answered HTTP "
        f"{response.status_code} for {url}: {response.text[:500]}"
    )
    return text[:500]


def stub_mirix_prompt(body: Mapping[str, Any]) -> str | None:
    """The stub MIRIX's own prompt when ``body`` is one of its chat calls (a retrieval's
    topic extraction or a learning run's distiller), else None."""
    messages = body.get("messages") or []
    if messages and isinstance(messages[0], Mapping):
        content = messages[0].get("content")
        if content in (STUB_TOPIC_PROMPT, STUB_DISTILLER_PROMPT):
            return str(content)
    return None


def with_stub_mirix(
    responder: Callable[[str, dict[str, Any]], Any],
) -> Callable[[str, dict[str, Any]], Any]:
    """A ``FakeUpstream`` responder that answers the stub MIRIX's own chat calls with the
    fake's default reply and leaves every other request to ``responder`` (a script of the
    actor's replies, say). A stub call carrying :data:`STUB_FAIL_MARKER` gets an upstream
    502, so the gateway retries it and gives up (an infrastructure failure, H8)."""
    from evals.skill_bench.tests.fakes import Reply

    def respond(endpoint: str, body: dict[str, Any]) -> Any:
        if STUB_FAIL_MARKER in json.dumps(body):
            # Retry-After: 0 lets the gateway spend its five retries at once.
            return Reply(
                status=502,
                json={"error": {"message": "stub upstream down"}},
                headers={"Retry-After": "0"},
            )
        if endpoint == "chat/completions" and stub_mirix_prompt(body):
            return None
        return responder(endpoint, body)

    return respond
