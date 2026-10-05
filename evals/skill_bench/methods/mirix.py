"""The MIRIX method: MIRIX through its native API only.

The harness adds no memory logic of its own. It hands MIRIX each raw message as the
rollout produces it, asks MIRIX's own retrieval for skills before a task, and injects
MIRIX's own rendering of the result unchanged (Q21). Everything MIRIX learns, it learns
in its own background learning runs.

**Hooks** (H7.2):

* ``setup``: start (or, with ``options["mirix_base_url"]``, connect to) the cell's MIRIX
  server (``infra/mirix_server.py``: its own database and HOME, H8), check that it has the
  API this method needs (:func:`missing_features`), and create the organization, client,
  meta agent and user through MIRIX's own client (``MirixClient.initialize_meta_agent``,
  ``create_or_get_user``). The meta agent's ``llm_config`` is endpoint type ``openai`` at
  the gateway role ``mirix`` with ``context_window`` and ``max_output_tokens`` from
  ``config/models.json``; its ``embedding_config`` is type ``openrouter``,
  ``google/gemini-embedding-001``, at the gateway role ``embedding``. Both carry the
  gateway's local credential, never a provider key. Every other setting is MIRIX's
  default (``methods/templates/mirix/setup.json``).
* ``before_task``: ``retrieve_with_conversation(messages=first_messages,
  memory_types=["procedural"], session_id=task_uid)``, then MIRIX's own
  ``render_memories(response)`` (``mirix.client.render.render_retrieved_memories``, M7.5);
  the rendered text is the injection. The retrieval's log row (``list_retrieval_logs``,
  its ``retrieval_id``, M7.4) says whether the topic extraction or the query vector was
  lost: when anything but the model took it away (:func:`retrieval_problem`), the method
  raises :class:`~evals.skill_bench.infra.failures.MirixRetrievalDegraded` (INFRA or
  HARNESS ``mirix_retrieval``) for the first two consecutive degraded retrievals of a
  test item, so the item is re-run (H8, rule N1); the third is injected and flagged
  (``Injection.degraded``: the item's row says ``retrieval_degraded``). In the learning
  phase a degraded retrieval is injected, flagged and counted
  (``tolerated.counts.retrievals_degraded``), and the third consecutive learning item
  with one raises (the attempt resumes from its last checkpoint; rule N1). The
  retrieval's record (``mirix_retrievals.jsonl``) keeps
  that evidence and the row's ranking trace (:func:`procedural_trace`: gate, limit and
  candidates), so the R2.5 gate can be re-checked for the skills it filtered out too.
* ``on_message``: ``add(messages=[message], session_id=task_uid, session_tag="task")``;
  anything but ``status: "stored"`` raises, and MIRIX's ``warnings`` are counted and kept.
  The first add of a learning session also carries ``agent_context``: the
  ``TaskView.agent_context`` its ``before_task`` received (the actor's prompt
  template, the same for every item), which
  MIRIX stores on the session (``conversation_session.agent_context``; a task-session add
  stores at least one turn, so it is stored with it) and its distiller shows beside the
  transcript (``<agent_context>``). Test-phase sessions are never distilled and get none;
  an empty agent context is never sent.
* ``after_task``: ``close_session(task_uid)``.
* ``wait_until_idle``: poll ``list_learning_runs()`` until no run is queued or running;
  past the timeout, :class:`TimeoutError` (INFRA, Q18). Once idle in the learning phase,
  MIRIX's own record of what it learned is checked (:func:`learning_problems`): a failed
  learning run, a session MIRIX deferred after an infrastructure error (M3.5
  ``distill_retry_after``), a curator batch an infrastructure error stopped, or a curator
  batch whose earlier lessons an experience-pool embedding failure cost (none shown, or
  a pool experience left out; the embedding model is infrastructure, never
  the model under test) means the attempt no longer learns what it would have learned
  without the outage, so it raises ``LearningAttemptInvalid`` (INFRA ``mirix_learning``;
  HARNESS when the gateway rejected MIRIX's embedding request, when MIRIX failed on its
  own without a gateway answer, or when MIRIX's configuration skipped its curation or
  deferred a session: :func:`curation_configuration_skip`,
  :func:`distill_configuration_deferral`). An INFRA one resumes the attempt from the
  ``played`` checkpoint of the item whose close started the batch, so the whole batch
  runs again from the database as it was before it, never as a re-flush of the failed
  session (a smaller batch than an uninterrupted run curates); a HARNESS one stops the
  attempt for a person (H8, X3, X13). MIRIX's model failures
  (counted per session, Q8) are not infrastructure. An exception inside a curator tool
  that MIRIX answered to the model as a failed tool result (MIRIX's own, or an
  infrastructure one; never the model's bad arguments) is logged as a warning
  (:func:`learning_warnings`, B31), not a reason to invalidate the attempt; the audit
  reports it per attempt.
* ``finalize``: ``flush_skills()`` (M1.7), then the same poll and ``get_learning_run`` of
  the flush's run. A session whose flush run distilled it without success counts a
  failure (Q8) and stays pending: MIRIX takes it up again, so the flush repeats until
  every closed session is settled (distilled, or failed for good after its third counted
  failure) and nothing is left for a sweep in the test phase to learn from
  (:func:`unsettled_sessions`). The frozen memory's snapshot hash is recorded for the
  test phase.
* ``snapshot``: the skills, each one's version history and every version's stored content
  (``get_skill_history`` / ``get_skill_version``, M6.1) and the learning runs (M9, M10),
  plus the experience pool's state as the runs' curator batches report it
  (:func:`experience_pool`: MIRIX has no endpoint for experiences), which the
  harness derives from the runs and keeps outside the frozen hash (the runs are in it).

**Learning checkpoints** (``methods/base.py`` contract item 6): :meth:`checkpoint`
returns the method's state; ``run.py`` dumps the cell database and records HOME with each
checkpoint and restores both before a resumed learning phase restarts MIRIX (mode
``resume``). ``setup`` with ``cell.resume`` then binds the meta agent MIRIX has, checks
its configuration (the cell's gateway run), loads the state and requires the rebuilt
snapshot to hash to the checkpoint's (``CheckpointMismatch``).

**Test phase**: ``setup`` with ``phase="test"`` resumes the attempt's MIRIX (mode
``resume``) using GET requests only, and refuses a memory whose snapshot differs from the
one frozen by ``finalize``. Afterwards only ``before_task`` runs (retrieval reads memory
and writes nothing but MIRIX's retrieval log); ``on_message`` / ``after_task`` raise.

**The API** (:class:`MirixApi`) is MIRIX's own Python client, ``MirixClient``
(``mirix/client/remote_client.py``), imported when the method is set up (never before: an
import of MIRIX loads a ``.env`` it can see, so the import refuses one). Three settings
make it the harness's:

* ``max_retries=0``: its ``RetryTransport`` would repeat a POST after a 5xx or a dropped
  connection and could store a message twice behind the harness's back; a failed write is
  an infrastructure failure of the learning attempt instead (H8);
* the harness creates its user first (``create_or_get_user``), so the client's per-call
  "ensure the user exists" POST is skipped (a test phase makes GET requests only);
* a resumed test phase binds the client to the frozen meta agent it lists (``GET
  /agents``), not to one ``initialize_meta_agent`` would create or update.

Failures (H8): a transport error or a 5xx from MIRIX raises
:class:`~evals.skill_bench.infra.mirix_server.MirixServerError` (INFRA ``mirix_server``);
a 4xx, an unexpected reply or a missing feature raises :class:`MirixApiError` (HARNESS).
MIRIX handles its own model failures (M3.5) and records them in its learning runs, which
the snapshot exports.

**Tolerated failures** (the rule that a MIRIX learning problem stops the attempt, and
rule N1): a learning
problem whose error is a gateway or provider refusal of one of MIRIX's calls
(:func:`refused_call_in`: a payload refused with HTTP 413, or 400 naming its size or
images, a no-endpoint 404, a request the gateway itself refused) is counted and learning
goes on; the same refusal in ``REFUSAL_STREAK_ITEMS`` (3) consecutive learning runs
stops the attempt for a person (``upstream_refusal_streak``; an idle check without a new
run leaves the streak as it is). Never counted, a stop at once (HARNESS,
:func:`untolerated_refusal`): a refusal of a session MIRIX deferred (it is never learned,
so the memory could not be frozen) and a gateway refusal of a protocol type
(the audit fails the cell for it). An INFRA problem of the batch a resumed learning phase
replays, after its checkpoint was rolled back ``MIRIX_BATCH_ROLLBACKS`` (3) times in a row
(``LearningResume.rollbacks``), is accepted at the first idle check after the resume.
Each accepted problem is written to ``mirix_learning.jsonl`` (event ``tolerated``) and
kept in the state and in the snapshot's ``tolerated`` block (``counts``: by reason, and
``mirix_learning_failed`` for both; ``problems``: reason, text and rollbacks), which is
absent while nothing was tolerated, so a memory without tolerance snapshots as before.
``finalize``'s own checks (sessions not learned before the freeze, skills without a
vector) are never accepted: either would let the test phase change the memory (H10.4).
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar
from urllib.parse import quote

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.envs.base import Message
from evals.skill_bench.infra.failures import (
    ACCOUNT_REFUSAL_MARKERS,
    GATEWAY_REJECTED,
    MIRIX_BATCH_ROLLBACKS,
    MIRIX_LEARNING,
    MIRIX_RETRIEVAL,
    PROTOCOL_REJECTION_TYPES,
    REFUSAL_STREAK_ITEMS,
    Failure,
    FailureKind,
    LearningAttemptInvalid,
    MirixRetrievalDegraded,
    RefusalStreaks,
    classify_exception,
    classify_gateway_error,
    refusal_class,
)
from evals.skill_bench.infra.mirix_server import (
    CellServerSpec,
    MirixServer,
    MirixServerError,
    ServerMode,
    dotenv_files_visible_to_mirix,
)
from evals.skill_bench.infra.processes import append_jsonl
from evals.skill_bench.methods.base import (
    CellContext,
    CheckpointMismatch,
    FrozenMemoryMismatch,
    Injection,
    MemoryMethodBase,
    SignalLevel,
    TaskView,
    TrajectoryView,
    check_resumed_snapshot,
)

T = TypeVar("T")
log = logging.getLogger(__name__)

SETUP_FILE = "methods/templates/mirix/setup.json"
STATE_FILE = "mirix_method.json"
RETRIEVAL_RECORD_FILE = "mirix_retrievals.jsonl"
LEARNING_RECORD_FILE = "mirix_learning.jsonl"
STATE_VERSION = 1

#: What a retrieval record keeps of MIRIX's ranking trace (the log row's
#: ``per_type.procedural``, M7.4; ``SkillRetrievalDiagnostics.to_dict``).
PROCEDURAL_TRACE_KEYS = (
    "requested_method",
    "method",
    "candidate_count",
    "missing_vector_count",
    "gate",
    "limit",
    "max_cosine_distance",
    "candidates",
    "candidates_omitted",
    "returned_ids",
)

#: MIRIX's M3.5 embedding policy: one attempt and 3 back-off retries, waiting 1, 2 and
#: 4 s between them (``mirix.services.retry_policy``; a test checks the two agree).
MIRIX_EMBEDDING_ATTEMPTS = 4
MIRIX_EMBEDDING_BACKOFF_S = 1.0 + 2.0 + 4.0
#: How long the harness waits for one MIRIX answer. The slowest one is a retrieval: one
#: topic-extraction call, then the query embedding with its whole M3.5 retry chain, every
#: call bounded by MIRIX's own client timeout (``MIRIX_LLM_TIMEOUT_SECONDS`` and
#: ``MIRIX_EMBEDDING_TIMEOUT_SECONDS``, both ``protocol.CLIENT_TIMEOUT_S``, above the
#: gateway's deadline). A fallback query long enough to need several embedding chunks
#: (``RETRIEVAL_FALLBACK_QUERY_MAX_CHARS``), every one of them held to the deadline, could
#: take longer; the harness then gives up on MIRIX (``MirixServerError``, INFRA), so such a
#: retrieval is re-run (test) or its attempt restarted (learning), never scored (H8).
MIRIX_HTTP_TIMEOUT_S = (
    (1 + MIRIX_EMBEDDING_ATTEMPTS) * protocol.CLIENT_TIMEOUT_S
    + MIRIX_EMBEDDING_BACKOFF_S
    + protocol.CLIENT_TIMEOUT_MARGIN_S
)
#: How often ``wait_until_idle`` asks MIRIX for active learning runs.
DEFAULT_POLL_S = 2.0
#: ``finalize`` waits this long for the flush run (a few sessions, each call bounded by
#: the gateway deadline and MIRIX's M3.5 retries). ``options["mirix_finalize_timeout_s"]``.
DEFAULT_FINALIZE_TIMEOUT_S = 7200.0
#: Learning runs fetched for the snapshot; MIRIX caps ``limit`` at 500 (M9).
LEARNING_RUNS_LIMIT = 500
#: Skills fetched for the snapshot; MIRIX caps ``/memory/components`` at 1,000.
SKILLS_LIMIT = 1000
#: Retrieval log rows fetched to find one retrieval's row: the rows of its task session,
#: newest first (a test item re-run under H8 has one row per run).
RETRIEVAL_LOG_LIMIT = 50
#: GET requests the snapshot export keeps in flight (skill histories and versions).
EXPORT_CONCURRENCY = 8
CLIENT_MODULE = "mirix.client.remote_client"
CLIENT_CLASS = "MirixClient"
RENDER_MODULE = "mirix.client.render"
RENDER_FUNCTION = "render_retrieved_memories"
#: ``MirixClient`` methods the method calls.
REQUIRED_CLIENT_METHODS: tuple[str, ...] = (
    "initialize_meta_agent",
    "create_or_get_user",
    "list_agents",
    "add",
    "retrieve_with_conversation",
    "render_memories",
    "close_session",
    "flush_skills",
    "list_learning_runs",
    "get_learning_run",
    "list_retrieval_logs",
    "list_memory_components",
    "get_skill_history",
    "get_skill_version",
    "close",
)
ACTIVE = "active"
ACTIVE_RUN_STATUSES = ("queued", "running")
TERMINAL_RUN_STATUSES = ("succeeded", "failed")
#: ``distill_metrics`` session statuses that settle a session for good (M4.4, Q8).
SETTLED_SESSION_STATUSES = ("ok", "failed_permanently")
#: A session MIRIX distilled without success that it counted as a failure (Q8): it stays
#: pending and is taken up again.
COUNTED_FAILURE_STATUS = "failed"
#: A session MIRIX backed off after an infrastructure error (M3.5).
DEFERRED_STATUS = "deferred"
#: The ``config_deferral`` names of a session the distiller deferred for its own
#: configuration (MIRIX's ``DISTILL_CONFIGURATION_DEFERRALS``): an input budget too small
#: for any chunk, no LLM client or config. Every resume defers it again, so it is
#: HARNESS (:func:`distill_configuration_deferral`).
DISTILL_CONFIGURATION_DEFERRALS = ("input_budget_too_small", "no_llm_client")
#: Curator ``error_kind`` values that are the infrastructure's, not the model's (M3.5's
#: ``ErrorKind``: the provider or network, the deployment's configuration, the database,
#: an embedding call left failing after its back-off) — except on a curation
#: configuration skip (:func:`curation_configuration_skip`), which is HARNESS.
INFRA_ERROR_KINDS = ("infra", "config", "db", "embedding")
#: The ``curator_metrics`` status and ``error_kind`` of a curator stage MIRIX's
#: configuration stopped before any batch (MIRIX's ``curation_skip_entry``: the owner
#: has no write scope, the procedural agent lacks a curator tool, its context window is
#: too small ...; the reason in ``skipped``).
CURATION_SKIPPED_STATUS = "skipped"
CONFIG_ERROR_KIND = "config"
#: Counted failures that settle a session for good (Q8), and counted curation attempts
#: after which the curator marks an experience ``unaddressed`` (M5.1).
SESSION_FAILURES_TO_SETTLE = 3
CURATION_ATTEMPTS_TO_SETTLE = 3
#: The flush repeats until MIRIX starts no run for it: while sessions are left with a
#: counted failure, and while experiences a curator batch returned (a counted attempt)
#: wait for the curator. A session settles within 3 rounds and the experiences of the
#: round that distilled it within 3 more (distillation and curation share a run), so the
#: 7th flush finds nothing to learn.
MAX_FLUSH_ROUNDS = SESSION_FAILURES_TO_SETTLE + CURATION_ATTEMPTS_TO_SETTLE + 1
#: The consecutive degraded retrieval of a test item that is injected and flagged instead
#: of re-run (rule N1): two re-runs, then the third.
MIRIX_DEGRADED_RETRIEVAL_TRIES = 3
#: ``MirixState.tolerated`` keys (the snapshot's ``tolerated.counts``): learning problems
#: accepted because a gateway or provider refused MIRIX's call, or after
#: ``MIRIX_BATCH_ROLLBACKS`` rollbacks of their batch; both together; and learning
#: retrievals injected although degraded.
TOLERATED_REFUSED = "gateway_refusal"
TOLERATED_ROLLED_BACK = "rolled_back"
TOLERATED_TOTAL = "mirix_learning_failed"
TOLERATED_RETRIEVALS = "retrievals_degraded"
#: The refusal streak role of MIRIX's own calls (:class:`RefusalStreaks`).
MIRIX_ROLE = "mirix"

#: (HTTP method, path template) of every endpoint the method calls (MIRIX's API).
#: :func:`missing_features` checks them against MIRIX's OpenAPI schema.
REQUIRED_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("get", "/health"),
    ("post", "/organizations/create_or_get"),
    ("post", "/clients/create_or_get"),
    ("post", "/users/create_or_get"),
    ("get", "/users/{user_id}"),
    ("post", "/agents/meta/initialize"),
    ("get", "/agents"),
    ("post", "/memory/add"),
    ("post", "/memory/retrieve/conversation"),
    ("get", "/memory/retrieval_logs"),
    ("post", "/memory/sessions/{session_id}/close"),
    ("post", "/memory/skills/flush"),
    ("get", "/memory/learning_runs"),
    ("get", "/memory/learning_runs/{run_id}"),
    ("get", "/memory/components"),
    ("get", "/memory/skills/{skill_id}/history"),
    ("get", "/memory/skills/{skill_id}/versions/{version_no}"),
)
#: Request fields MIRIX's session retrieval adds to ``/memory/retrieve/conversation``.
REQUIRED_RETRIEVE_FIELDS: tuple[str, ...] = ("memory_types", "session_id")
#: Request fields of ``/memory/add`` the method sends beyond the messages: the session
#: and the session's agent context.
REQUIRED_ADD_FIELDS: tuple[str, ...] = ("session_id", "session_tag", "agent_context")

#: MIRIX's own exception names (``llm_usage.topic_extraction.error`` of a retrieval log
#: row starts with one) for a topic extraction that failed on the model's answer or on
#: MIRIX itself: no usable tool call, an answer cut off or not parseable, an input above
#: the budget, a request MIRIX could not build (unless what it wraps is a network or
#: database failure, :func:`extraction_failure`). MIRIX retrieves with the raw text then
#: (M7.1): that is the method's own behaviour, scored like any other (H8). Every other
#: failure is judged by the gateway's answer it carries (:func:`retrieval_problem`).
MODEL_OR_METHOD_EXTRACTION_ERRORS: tuple[str, ...] = (
    "TopicExtractionError",
    "LLMOutputTruncatedError",
    "LLMInvalidResponseError",
    "LLMJSONParsingError",
    "ContextBudgetExceeded",
    "LLMRequestBuildError",
)


#: Exception names, in the error texts MIRIX records (a retrieval's per-type ``errors``, a
#: learning run's ``error``), of failures that are the environment's: the database or the
#: network went away, or MIRIX gave up waiting on them (SQLAlchemy, asyncpg, httpx, the
#: OpenAI SDK and the standard library's names). Any other exception MIRIX reports is its
#: own failure (a ``TypeError``, an ``IntegrityError``, a schema that no longer matches):
#: it fails the same way again, so it is HARNESS, for a person, and never an INFRA re-run
#: or restart paid for again and again. A request MIRIX could not build
#: (``LLMRequestBuildError``) is judged by what it wraps instead, as MIRIX itself judges
#: it (:func:`build_error_kind`).
INFRA_EXCEPTION_NAMES = frozenset(
    {
        # databases
        "OperationalError",
        "InterfaceError",
        "DisconnectionError",
        "DeadlockDetectedError",
        "SerializationError",
        "PostgresConnectionError",
        "ConnectionDoesNotExistError",
        "ConnectionFailureError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "AdminShutdownError",
        "CrashShutdownError",
        "QueryCanceledError",
        "LockNotAvailableError",
        # network and waiting
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ReadError",
        "WriteError",
        "NetworkError",
        "RemoteProtocolError",
        "TransportError",
        "APIConnectionError",
        "APITimeoutError",
        "LLMConnectionError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "CancelledError",
        "ServerDisconnectedError",
        "ClientConnectionError",
        "ClientOSError",
    }
)
_EXCEPTION_NAME = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Timeout))\b")


def exception_names(text: Any) -> list[str]:
    """The exception class names an error text MIRIX recorded mentions, in order
    (``"OperationalError (ConnectionDoesNotExistError): ..."`` gives both)."""
    return _EXCEPTION_NAME.findall(str(text or ""))


#: What MIRIX's ``_classify_build_error`` (``mirix/services/retry_policy.py``, M3.5) calls
#: transient in a request it could not build: an ``httpx.TransportError`` or a built-in
#: ``ConnectionError`` on the way (its ``INFRA``), and a transient database error while
#: reading stored state (its ``DB``: the asyncpg classes it names and those of the
#: SQLSTATEs it lists). Everything else it wraps is MIRIX's own failure (``INTERNAL``):
#: a ``KeyError``, a local file missing, and also a ``TimeoutError`` or a
#: ``CancelledError`` (reading a stored attachment timed out), which
#: :data:`INFRA_EXCEPTION_NAMES` would call infrastructure.
BUILD_ERROR_INFRA_NAMES = frozenset(
    {
        # httpx.TransportError and its subclasses
        "TransportError",
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "ConnectError",
        "ReadError",
        "WriteError",
        "CloseError",
        "ProtocolError",
        "LocalProtocolError",
        "RemoteProtocolError",
        "ProxyError",
        "UnsupportedProtocol",
        # the built-in ConnectionError and its subclasses
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        # transient database errors (asyncpg; SQLAlchemy's lost connection)
        "DisconnectionError",
        "DeadlockDetectedError",
        "SerializationError",
        "LockNotAvailableError",
        "AdminShutdownError",
        "CrashShutdownError",
        "CannotConnectNowError",
        "TooManyConnectionsError",
        "PostgresConnectionError",
        "ConnectionDoesNotExistError",
        "ConnectionFailureError",
        "ClientCannotConnectError",
        "ConnectionRejectionError",
        "TransactionResolutionUnknownError",
        "ProtocolViolationError",
    }
)
#: HTTP statuses of an ``httpx.HTTPStatusError`` MIRIX calls transient while building a
#: request (its ``_INFRA_STATUS``, and every 5xx).
BUILD_ERROR_INFRA_STATUSES = frozenset({408, 409, 424, 429})
#: SQLAlchemy's wrappers of a driver error: transient only when the driver's message says
#: so (MIRIX's ``_TRANSIENT_DB_MESSAGES``), as a locked SQLite database does.
DB_WRAPPER_NAMES = frozenset(
    {"OperationalError", "InterfaceError", "DatabaseError", "DBAPIError"}
)
TRANSIENT_DB_MESSAGES: tuple[str, ...] = (
    "deadlock detected",
    "could not serialize access",
    "database is locked",
    "database table is locked",
    "connection is closed",
    "connection was closed",
    "server closed the connection",
    "terminating connection",
    "connection reset",
    "could not obtain lock",
)
#: SQLAlchemy's connection-pool timeout (``sqlalchemy.exc.TimeoutError``, transient for
#: MIRIX) has the built-in ``TimeoutError``'s name; its message tells them apart.
POOL_TIMEOUT_MARKER = "queuepool limit"
#: MIRIX's own verdict on a request it could not build, which it writes into the error
#: text right after the words it opens with (``LLMRequestBuildError.wrapping``:
#: ``building the <Client> request failed [kind=<kind>] (<Cause>: <cause>)``): its retry
#: policy's ``classify_build_cause`` on the whole chain of causes.
_BUILD_ERROR_VERDICT = re.compile(
    r":\s*(?:building|calling) the \S+ (?:request|SDK) failed \[kind=([a-z_]+)\]"
)
#: MIRIX's kinds of such a failure, as H8 kinds: a network or database failure on the way
#: is INFRA, anything else (``internal``) MIRIX's own, which a re-run repeats.
BUILD_ERROR_VERDICTS: Mapping[str, FailureKind] = MappingProxyType(
    {
        "infra": FailureKind.INFRA,
        "db": FailureKind.INFRA,
        "internal": FailureKind.HARNESS,
    }
)
#: Without that verdict (a MIRIX from before it): where MIRIX's text names classes,
#: never its free text. The cause's class right after ``failed (`` (with the driver's
#: class after it, ``Name (Driver):``, as MIRIX writes a database error it reports), then,
#: at the start of its message, SQLAlchemy's driver class ``(module.Class)`` and the
#: asyncpg class its adapter names (``<class 'asyncpg.exceptions.Class'>``).
_BUILD_ERROR_CAUSE = re.compile(
    r":\s*(?:building|calling) the \S+ (?:request|SDK) failed"
    r"(?: \[kind=[a-z_]+\])? \(([A-Za-z_]\w*)(?: \(([A-Za-z_]\w*)\))?: ?"
)
_DRIVER_CLASSES = re.compile(r"\((?:\w+\.)*(\w+)\)(?:\s*<class '(?:\w+\.)*(\w+)'>)?")
_HTTP_STATUS = re.compile(r"[^']*'([1-5][0-9][0-9])\b")


def build_error_kind(text: Any) -> FailureKind:
    """H8 kind of an error text naming ``LLMRequestBuildError``: MIRIX's own verdict in
    that text (:data:`BUILD_ERROR_VERDICTS`), which judges every
    cause in the chain, as MIRIX's ``_classify_build_error`` does. A text without it is
    judged on the classes MIRIX formats into it (:func:`build_error_kind_from_classes`)."""
    _, found, wrapped = str(text or "").partition("LLMRequestBuildError")
    if not found:
        return FailureKind.HARNESS
    verdict = _BUILD_ERROR_VERDICT.match(wrapped)
    if verdict is not None and verdict.group(1) in BUILD_ERROR_VERDICTS:
        return BUILD_ERROR_VERDICTS[verdict.group(1)]
    return build_error_kind_from_classes(wrapped)


def build_error_kind_from_classes(wrapped: str) -> FailureKind:
    """H8 kind of the text after ``LLMRequestBuildError`` when MIRIX wrote no verdict:
    INFRA when the cause's class, or the driver class SQLAlchemy names at the start of
    its message (:data:`_BUILD_ERROR_CAUSE`, :data:`_DRIVER_CLASSES`), is one MIRIX's
    ``_classify_build_error`` calls transient (:data:`BUILD_ERROR_INFRA_NAMES`), an
    ``HTTPStatusError`` of :data:`BUILD_ERROR_INFRA_STATUSES` or 5xx, a database wrapper
    whose driver message is one of :data:`TRANSIENT_DB_MESSAGES`, or the connection
    pool's timeout; else HARNESS: MIRIX's ``INTERNAL``, its own failure. Words elsewhere
    in the text (a bind parameter, a dictionary key naming an exception) never count.
    The text names one cause only: a transient failure deeper in the chain is not seen,
    which only MIRIX's verdict covers."""
    cause = _BUILD_ERROR_CAUSE.match(wrapped)
    if cause is None:
        return FailureKind.HARNESS
    names = [name for name in cause.groups() if name]
    message = wrapped[cause.end() :]
    # The driver's own message only, as MIRIX judges it: SQLAlchemy's text goes on with
    # the statement and its bind parameters, which may hold any words.
    driver = re.split(r"\[sql:", message, maxsplit=1, flags=re.IGNORECASE)[0]
    classes = _DRIVER_CLASSES.match(driver)
    if classes is not None:
        names += [name for name in classes.groups() if name]
    if any(name in BUILD_ERROR_INFRA_NAMES for name in names):
        return FailureKind.INFRA
    if names[0] == "HTTPStatusError":
        found = _HTTP_STATUS.match(message)
        status = int(found.group(1)) if found else 0
        if status in BUILD_ERROR_INFRA_STATUSES or status >= 500:
            return FailureKind.INFRA
    lowered = driver.lower()
    if "TimeoutError" in names and POOL_TIMEOUT_MARKER in lowered:
        return FailureKind.INFRA
    if any(name in DB_WRAPPER_NAMES for name in names) and any(
        marker in lowered for marker in TRANSIENT_DB_MESSAGES
    ):
        return FailureKind.INFRA
    return FailureKind.HARNESS


def mirix_error_kind(text: Any) -> FailureKind:
    """INFRA when an error MIRIX recorded names a database or network failure (any of
    :data:`INFRA_EXCEPTION_NAMES`, a lease MIRIX lost, a run it cancelled), else HARNESS:
    MIRIX's own failure, which a re-run repeats. A request MIRIX could not build is
    judged by what it wraps, as MIRIX judges it (:func:`build_error_kind`)."""
    value = str(text or "")
    if value.strip() in ("lease_expired", "cancelled"):
        return FailureKind.INFRA  # MIRIX's process stalled or stopped mid-run
    names = exception_names(value)
    if "LLMRequestBuildError" in names:
        return build_error_kind(value)
    if any(name in INFRA_EXCEPTION_NAMES for name in names):
        return FailureKind.INFRA
    return FailureKind.HARNESS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    """sha256 of ``value`` as sorted, compact JSON (snapshot identity)."""
    return _sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


class MirixApiError(RuntimeError):
    """MIRIX refused a request (4xx), answered unexpectedly or lacks a needed feature (HARNESS)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def setup_template() -> Mapping[str, Any]:
    """``methods/templates/mirix/setup.json`` (fingerprinted)."""
    return MappingProxyType(json.loads(protocol.read_protocol_text(SETUP_FILE)))


def meta_agent_config(cell: CellContext) -> dict[str, Any]:
    """The ``config`` of ``/agents/meta/initialize`` for ``cell`` (H7.2).

    Only these fields are set; every other LLM, embedding and meta-agent setting keeps
    MIRIX's default.
    """
    template = setup_template()
    return {
        "llm_config": {
            **template["llm_config"],
            "model": cell.model,
            "model_endpoint": cell.role_url("mirix"),
            "context_window": cell.context_length,
            "max_output_tokens": cell.max_output_tokens,
            "api_key": cell.gateway_token,
        },
        "embedding_config": {
            **template["embedding_config"],
            "embedding_endpoint": cell.role_url("embedding"),
            "embedding_model": cell.embedding_model,
            "api_key": cell.gateway_token,
        },
    }


def without_secrets(config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with every ``api_key`` removed (for records)."""
    return {
        key: without_secrets(value) if isinstance(value, Mapping) else value
        for key, value in config.items()
        if key != "api_key"
    }


# ---------------------------------------------------------------------------
# MIRIX's own client library (imported at setup)
# ---------------------------------------------------------------------------

Renderer = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class MirixLibrary:
    """What the method uses of MIRIX's Python package: its client class (as the harness
    subclasses it, :func:`_harness_client_class`) and its M7.5 renderer."""

    client_class: type
    render: Renderer


def _harness_client_class(base: type) -> type:
    """``MirixClient`` as the harness uses it (see the module docstring)."""

    class HarnessMirixClient(base):  # type: ignore[misc, valid-type]
        """MIRIX's ``MirixClient`` with the harness's three settings: no transport retries
        (the constructor's ``max_retries=0``), no implicit user creation, and a meta agent
        bound from a listing for a resumed test phase."""

        async def _ensure_user_exists(
            self, user_id: str | None = None, headers: Any = None
        ) -> None:
            # The harness called create_or_get_user (or, resuming, GET /users/{id}) first.
            return None

        async def get_user(self, user_id: str) -> Any:
            return await self._request("GET", f"/users/{quote(user_id, safe='')}")

        async def openapi(self) -> Any:
            return await self._request("GET", "/openapi.json")

        def bind_meta_agent(self, agent: Any) -> None:
            self._meta_agent = agent

    return HarnessMirixClient


def _load_library() -> MirixLibrary:
    leaks = dotenv_files_visible_to_mirix()
    # Importing mirix runs load_dotenv(): no provider key may reach the harness.
    if leaks:
        raise MirixServerError(
            f"importing MIRIX would load {leaks[0]} via load_dotenv(); move it out of the way"
        )
    try:
        client_module = importlib.import_module(CLIENT_MODULE)
        render_module = importlib.import_module(RENDER_MODULE)
        base = getattr(client_module, CLIENT_CLASS)
        render = getattr(render_module, RENDER_FUNCTION)
    except (ImportError, AttributeError) as exc:
        raise MirixApiError(
            f"this MIRIX has no {CLIENT_MODULE}.{CLIENT_CLASS} or "
            f"{RENDER_MODULE}.{RENDER_FUNCTION}: {exc}"
        ) from exc
    missing = [name for name in REQUIRED_CLIENT_METHODS if not hasattr(base, name)]
    if missing:
        raise MirixApiError(f"this MIRIX's {CLIENT_CLASS} lacks {', '.join(missing)}")
    if not callable(render):
        raise MirixApiError(f"{RENDER_MODULE}.{RENDER_FUNCTION} is not callable")

    def rendered(response: Mapping[str, Any]) -> str:
        text = render(response)
        if not isinstance(text, str):
            raise MirixApiError(
                f"{RENDER_MODULE}.{RENDER_FUNCTION} returned {type(text).__name__}"
            )
        return text

    return MirixLibrary(_harness_client_class(base), rendered)


_library_cache: list[MirixLibrary] = []


async def load_mirix() -> MirixLibrary:
    """MIRIX's client and renderer (imported once, off the event loop)."""
    if not _library_cache:
        _library_cache.append(await asyncio.to_thread(_load_library))
    return _library_cache[0]


# ---------------------------------------------------------------------------
# The API: MirixClient with H8's failure classes
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> Any:
    """A pydantic model MIRIX's client returned (``AgentState``) as JSON data."""
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else value


class MirixApi:
    """MIRIX's REST API for one client application (tenant), through ``MirixClient``.

    Authentication is MIRIX's client headers (``X-Client-ID`` / ``X-Org-ID``: the client
    has no API key; one in ``MIRIX_API_KEY`` would switch it to another tenant, so it is
    refused). Requests are never retried here: a failed write is an infrastructure
    failure of the learning attempt (H8), not something to repeat behind MIRIX's back.
    Every call maps MIRIX's answer to H8: a transport error or a 5xx is
    :class:`MirixServerError` (INFRA), a 4xx or a reply MIRIX's own client cannot read is
    :class:`MirixApiError` (HARNESS).
    """

    def __init__(
        self,
        library: MirixLibrary,
        base_url: str,
        *,
        client: Mapping[str, Any],
        organization: Mapping[str, Any],
        timeout_s: float = MIRIX_HTTP_TIMEOUT_S,
        on_transport_error: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        if os.environ.get("MIRIX_API_KEY"):
            raise MirixApiError(
                "MIRIX_API_KEY is set in the harness's environment: MirixClient would "
                "authenticate as that key's client instead of the cell's; unset it"
            )
        self.base_url = base_url.rstrip("/")
        self.client_id = str(client["client_id"])
        self.org_id = str(organization["org_id"])
        self._on_transport_error = on_transport_error
        self.client: Any = library.client_class(
            base_url=self.base_url,
            client_id=self.client_id,
            client_name=str(client["name"]),
            write_scope=client["write_scope"],
            read_scopes=list(client["read_scopes"]),
            org_id=self.org_id,
            org_name=str(organization["name"]),
            timeout=timeout_s,
            max_retries=0,
        )
        if getattr(self.client, "api_key", None):  # pragma: no cover - guarded above
            raise MirixApiError("MirixClient picked up an API key; refusing")

    async def aclose(self) -> None:
        await self.client.close()

    async def _call(self, what: str, call: Awaitable[T]) -> T:
        try:
            return await call
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = f"{exc}"[:500]
            if status >= 500:
                raise MirixServerError(
                    f"MIRIX {what} answered {status}: {detail}"
                ) from exc
            raise MirixApiError(
                f"MIRIX refused {what} ({status}): {detail}", status=status
            ) from exc
        except httpx.HTTPError as exc:
            if self._on_transport_error is not None:
                await self._on_transport_error()  # raises if the server died
            raise MirixServerError(
                f"MIRIX {what} failed: {type(exc).__name__}: {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:
            # A non-JSON body, a reply MIRIX's client schema rejects (pydantic's
            # ValidationError is a ValueError) or a request it refuses to send.
            raise MirixApiError(
                f"MIRIX {what}: {type(exc).__name__}: {str(exc)[:500]}"
            ) from exc

    # --- tenant ---------------------------------------------------------------------

    async def openapi(self) -> Any:
        return await self._call("GET /openapi.json", self.client.openapi())

    async def initialize_meta_agent(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """``MirixClient.initialize_meta_agent``: org and client ``create_or_get``, then
        ``/agents/meta/initialize`` (``update_agents=False``); the client keeps the agent."""
        agent = await self._call(
            "POST /agents/meta/initialize",
            self.client.initialize_meta_agent(
                config=json.loads(json.dumps(config)), update_agents=False
            ),
        )
        if agent is None:
            raise MirixApiError(
                "MIRIX created no meta agent (does the client have a write scope?)"
            )
        return _as_dict(agent)

    async def create_or_get_user(self, user_id: str, user_name: str) -> str:
        return await self._call(
            "POST /users/create_or_get",
            self.client.create_or_get_user(user_id=user_id, user_name=user_name),
        )

    async def get_user(self, user_id: str) -> Any:
        return await self._call(f"GET /users/{user_id}", self.client.get_user(user_id))

    async def bind_meta_agent(self, agent_id: str) -> dict[str, Any]:
        """The client's meta agent, found in ``GET /agents`` (a resumed test phase)."""
        agents = await self._call("GET /agents", self.client.list_agents())
        match = [a for a in agents or () if getattr(a, "id", None) == agent_id]
        if len(match) != 1:
            raise MirixApiError(
                f"the resumed MIRIX has no meta agent {agent_id} for this client"
            )
        self.client.bind_meta_agent(match[0])
        return _as_dict(match[0])

    # --- memory ---------------------------------------------------------------------

    async def add(
        self,
        user_id: str,
        messages: Sequence[Message],
        *,
        session_id: str,
        session_tag: str,
        agent_context: str | None = None,
    ) -> Any:
        """``MirixClient.add``; ``agent_context`` is passed only when given, so an
        add without one sends the request body it always sent."""
        extra = {"agent_context": agent_context} if agent_context is not None else {}
        return await self._call(
            "POST /memory/add",
            self.client.add(
                user_id=user_id,
                messages=list(messages),
                session_id=session_id,
                session_tag=session_tag,
                **extra,
            ),
        )

    async def retrieve_with_conversation(
        self,
        user_id: str,
        messages: Sequence[Message],
        *,
        limit: int,
        memory_types: Sequence[str],
        session_id: str,
    ) -> Any:
        return await self._call(
            "POST /memory/retrieve/conversation",
            self.client.retrieve_with_conversation(
                user_id=user_id,
                messages=list(messages),
                limit=limit,
                memory_types=list(memory_types),
                session_id=session_id,
            ),
        )

    async def list_retrieval_logs(
        self, *, session_id: str, user_id: str, limit: int = RETRIEVAL_LOG_LIMIT
    ) -> list[dict[str, Any]]:
        rows = await self._call(
            "GET /memory/retrieval_logs",
            self.client.list_retrieval_logs(
                session_id=session_id, user_id=user_id, limit=limit
            ),
        )
        return _dict_rows(rows, "retrieval log listing")

    # --- sessions and learning -------------------------------------------------------

    async def close_session(self, session_id: str, user_id: str) -> Any:
        return await self._call(
            f"POST /memory/sessions/{session_id}/close",
            self.client.close_session(session_id=session_id, user_id=user_id),
        )

    async def flush_skills(
        self, *, user_id: str, close_open_sessions: bool, meta_agent_id: str
    ) -> Any:
        return await self._call(
            "POST /memory/skills/flush",
            self.client.flush_skills(
                user_id=user_id,
                close_open_sessions=close_open_sessions,
                meta_agent_id=meta_agent_id,
            ),
        )

    async def list_learning_runs(
        self, *, user_id: str, meta_agent_id: str, limit: int = LEARNING_RUNS_LIMIT
    ) -> list[dict[str, Any]]:
        runs = await self._call(
            "GET /memory/learning_runs",
            self.client.list_learning_runs(
                user_id=user_id, limit=limit, meta_agent_id=meta_agent_id
            ),
        )
        return _dict_rows(runs, "learning-run listing")

    async def get_learning_run(self, run_id: str) -> dict[str, Any]:
        run = await self._call(
            f"GET /memory/learning_runs/{run_id}", self.client.get_learning_run(run_id)
        )
        if not isinstance(run, dict) or run.get("id") != run_id:
            raise MirixApiError(f"unexpected learning run {run_id}: {str(run)[:300]}")
        return run

    # --- export ---------------------------------------------------------------------

    async def list_skills(self, user_id: str, limit: int = SKILLS_LIMIT) -> Any:
        return await self._call(
            "GET /memory/components",
            self.client.list_memory_components(
                user_id=user_id, memory_type="procedural", limit=limit
            ),
        )

    async def get_skill_history(self, skill_id: str, user_id: str) -> Any:
        return await self._call(
            f"GET /memory/skills/{skill_id}/history",
            self.client.get_skill_history(skill_id, user_id=user_id),
        )

    async def get_skill_version(
        self, skill_id: str, version_no: int, user_id: str
    ) -> Any:
        return await self._call(
            f"GET /memory/skills/{skill_id}/versions/{version_no}",
            self.client.get_skill_version(skill_id, version_no, user_id=user_id),
        )


def _dict_rows(rows: Any, what: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise MirixApiError(f"unexpected {what}: {str(rows)[:300]}")
    return rows


# ---------------------------------------------------------------------------
# Feature check
# ---------------------------------------------------------------------------

_PATH_PARAM = re.compile(r"\{[^}]*\}")


def _normal_path(path: str) -> str:
    return _PATH_PARAM.sub("{}", path.rstrip("/") or "/")


def _resolve(schema: Any, document: Mapping[str, Any]) -> Any:
    seen = 0
    while isinstance(schema, Mapping) and "$ref" in schema and seen < 16:
        ref = str(schema["$ref"])
        node: Any = document
        for part in ref.lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, Mapping) else {}
        schema, seen = node, seen + 1
    return schema


def _properties(schema: Any, document: Mapping[str, Any]) -> set[str]:
    schema = _resolve(schema, document)
    if not isinstance(schema, Mapping):
        return set()
    names = set((schema.get("properties") or {}).keys())
    for key in ("allOf", "anyOf", "oneOf"):
        for part in schema.get(key) or []:
            names |= _properties(part, document)
    return names


def missing_features(openapi: Any) -> list[str]:
    """What this method needs and MIRIX's OpenAPI document does not offer (empty: all there)."""
    if not isinstance(openapi, Mapping) or not isinstance(
        openapi.get("paths"), Mapping
    ):
        return ["an OpenAPI document (/openapi.json)"]
    offered = {
        (method.lower(), _normal_path(path))
        for path, operations in openapi["paths"].items()
        if isinstance(operations, Mapping)
        for method in operations
    }
    missing = [
        f"{method.upper()} {path}"
        for method, path in REQUIRED_ENDPOINTS
        if (method, _normal_path(path)) not in offered
    ]
    for path, names, why in (
        (
            "/memory/retrieve/conversation",
            REQUIRED_RETRIEVE_FIELDS,
            "retrieval by memory type and session",
        ),
        ("/memory/add", REQUIRED_ADD_FIELDS, "task sessions with an agent context"),
    ):
        operation = next(
            (
                operations.get("post")
                for offered_path, operations in openapi["paths"].items()
                if _normal_path(offered_path) == path
                and isinstance(operations, Mapping)
            ),
            None,
        )
        if not isinstance(operation, Mapping):
            continue  # the endpoint itself is reported missing above
        schema = (
            ((operation.get("requestBody") or {}).get("content") or {}).get(
                "application/json"
            )
            or {}
        ).get("schema")
        fields = _properties(schema, openapi)
        missing += [
            f"{path} request field {name!r} ({why})"
            for name in names
            if name not in fields
        ]
    return missing


# ---------------------------------------------------------------------------
# A degraded retrieval (MIRIX's retrieval fallbacks and log; rule N1)
# ---------------------------------------------------------------------------

#: The HTTP status in the error text MIRIX logs for a failed model or embedding call:
#: the OpenAI SDK's ``Error code: 502 - {...}`` (topic extraction) or MIRIX's
#: ``embeddings endpoint answered HTTP 502 for <url>: {...}`` (query embedding).
_STATUS_IN_ERROR = re.compile(r"(?:Error code:|answered HTTP) (\d{3})\b")


_CLOSERS = {"{": "}", "[": "]", "(": ")"}
#: How many cut points :func:`_truncated_mapping` tries, from the end, before giving up.
_MAX_REPAIRS = 64


def _parse_mapping(text: str) -> Mapping[str, Any] | None:
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _truncated_mapping(text: str) -> Mapping[str, Any] | None:
    """The mapping a JSON or Python-repr text starts with, when the text was cut short:
    every complete item before the cut, the open mappings and lists closed (the item the
    cut split is dropped). None when not even that parses.

    MIRIX keeps 500 characters of an error (``_error_text``). A gateway body that repeats
    a long upstream message (a moderation refusal) is cut inside it, and the fields that
    came first, ``type``, ``code`` and ``message`` among them, would otherwise be lost
    with the rest."""
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    cuts: list[tuple[int, str]] = []  # (end of a complete item, the closers it needs)
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in _CLOSERS:
            stack.append(_CLOSERS[char])
        elif char in "}])":
            if not stack or stack.pop() != char:
                break  # not a mapping any more
            if not stack:
                return _parse_mapping(text[: index + 1])
            cuts.append((index + 1, "".join(reversed(stack))))
        elif char == "," and stack:
            cuts.append((index, "".join(reversed(stack))))
    for end, closers in reversed(cuts[-_MAX_REPAIRS:]):
        value = _parse_mapping(text[:end] + closers)
        if value is not None:
            return value
    return None


def gateway_error_in(text: Any) -> tuple[int | None, Mapping[str, Any] | None]:
    """The HTTP status and the error body of the gateway answer an error text quotes:
    ``(None, None)`` for an error without one (a dropped connection, MIRIX's own
    exception). A body MIRIX's 500-character cut left incomplete keeps its complete
    fields (:func:`_truncated_mapping`); ``(status, None)`` when none survived."""
    if not isinstance(text, str):
        return None, None
    match = _STATUS_IN_ERROR.search(text)
    if match is None:
        return None, None
    status = int(match.group(1))
    rest = text[match.end() :]
    start = rest.find("{")
    if start < 0:
        return status, None
    candidate = rest[start:]
    return status, _parse_mapping(candidate) or _truncated_mapping(candidate)


def _call_failure(role: str, error: Any, what: str) -> Failure | None:
    """The H8 failure of a model or embedding call MIRIX made through the gateway and
    lost, judged by the gateway's answer (``infra/failures.classify_gateway_error``, the
    same rules as for the harness's own calls); None when the gateway blamed the model
    (a model timeout on the ``mirix`` role). Without a gateway answer (a dropped
    connection, a timeout on MIRIX's side, an error MIRIX raised itself) the failure is
    INFRA: the model is never blamed without evidence."""
    text = str(error) if error else f"MIRIX recorded no error for the {what}"
    status, body = gateway_error_in(text)
    if status is None:
        return Failure(
            FailureKind.INFRA, MIRIX_RETRIEVAL, role, None, f"{what}: {text}"[:500]
        )
    judged = classify_gateway_error(status, {"error": body} if body else {}, role)
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        judged = classify_gateway_error(status, body, role)
    if judged.kind is FailureKind.MODEL:
        return None
    if _moderation_refusal_in(status, body, text):
        return None  # the model's content filter: MIRIX's own fallback, scored (H7.2)
    return Failure(
        judged.kind,
        MIRIX_RETRIEVAL,
        role,
        status,
        f"{what} ({judged.kind.value}/{judged.code}): {text}"[:500],
        judged.retries,
    )


def _moderation_refusal_in(
    status: int, body: Mapping[str, Any] | None, text: str
) -> bool:
    """A 403 whose text says the provider's moderation refused the call, when not even
    its message survived MIRIX's cut (the harness's rule, ``classify_gateway_error``,
    judges the message): the model's content filter, never a request the gateway itself
    refused (``source: gateway``)."""
    if status != 403 or "moderat" not in text.lower():
        return False
    error = body.get("error") if isinstance(body, Mapping) else None
    error = error if isinstance(error, Mapping) else body or {}
    return error.get("source") != "gateway" and "'source': 'gateway'" not in text


def extraction_failure(error: Any) -> Failure | None:
    """H8 for a topic extraction MIRIX fell back from (``extraction_status: fallback``):
    None when the model's answer or MIRIX itself failed it
    (:data:`MODEL_OR_METHOD_EXTRACTION_ERRORS`: MIRIX's own M7.1 behaviour, scored), else
    the failure of the call (:func:`_call_failure`). A request MIRIX could not build
    because a network or database call on the way failed is infrastructure, exactly as
    MIRIX's own ``_classify_build_error`` has it (:func:`build_error_kind`);
    one that wraps anything else (a ``KeyError``, a ``TimeoutError``) is MIRIX's own
    failure, after which it retrieved with the raw text: its behaviour, scored."""
    text = str(error or "")
    name = text.split(":", 1)[0].strip()
    if name == "LLMRequestBuildError":
        if build_error_kind(text) is FailureKind.INFRA:
            return Failure(
                FailureKind.INFRA,
                MIRIX_RETRIEVAL,
                "mirix",
                None,
                f"topic extraction: {text}"[:500],
            )
        return None
    if name in MODEL_OR_METHOD_EXTRACTION_ERRORS:
        return None
    return _call_failure("mirix", error, "topic extraction")


def retrieval_evidence(
    response: Mapping[str, Any], row: Mapping[str, Any] | None
) -> dict[str, Any]:
    """What :func:`retrieval_problem` judges, from the retrieval's answer and its log row
    (M7.4): kept in ``mirix_retrievals.jsonl`` so the audit judges the same data."""
    row = row or {}
    usage = row.get("llm_usage") if isinstance(row.get("llm_usage"), Mapping) else {}
    extraction = usage.get("topic_extraction") if isinstance(usage, Mapping) else None
    per_type = row.get("per_type") if isinstance(row.get("per_type"), Mapping) else {}
    procedural = per_type.get("procedural") if isinstance(per_type, Mapping) else None
    events = [e for e in row.get("retry_events") or () if isinstance(e, Mapping)]
    return {
        "success": response.get("success"),
        "errors": dict(response.get("errors") or {}),
        "extraction_status": row.get(
            "extraction_status", response.get("extraction_status")
        ),
        "extraction_error": extraction.get("error")
        if isinstance(extraction, Mapping)
        else None,
        "embedding_status": row.get("embedding_status"),
        "embedding_error": procedural.get("embedding_error")
        if isinstance(procedural, Mapping)
        else None,
        "retry_events": [
            {
                "kind": e.get("kind"),
                "stage": e.get("stage"),
                "attempt": e.get("attempt"),
                "error": str(e.get("error") or "")[:300],
            }
            for e in events
        ],
    }


def procedural_trace(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The gate and candidate part of a retrieval log row's procedural trace (M7.4):
    the :data:`PROCEDURAL_TRACE_KEYS` MIRIX logged (a key it did not log is left out),
    or None for a row without that trace. ``candidates`` stays as MIRIX bounded it: the
    returned skills plus the next ``RETRIEVAL_LOG_TRACE_CANDIDATES`` in fused order, the
    rest counted in ``candidates_omitted`` (R24)."""
    per_type = (row or {}).get("per_type")
    procedural = per_type.get("procedural") if isinstance(per_type, Mapping) else None
    if not isinstance(procedural, Mapping):
        return None
    return {k: procedural[k] for k in PROCEDURAL_TRACE_KEYS if k in procedural}


def retrieval_problem(evidence: Mapping[str, Any]) -> Failure | None:
    """Why a retrieval must not be scored, or None (pure; rule N1, H8).

    * MIRIX reported that every requested type failed (``success: false``): INFRA when a
      search error names a database or network failure (:func:`mirix_error_kind`), else
      HARNESS: MIRIX's own failure, which a re-run repeats.
    * The query vector failed (``embedding_status: failed``: MIRIX ranked by BM25 alone,
      without the similarity gate, M7.3). The embedding model is infrastructure, never
      the model under test, so this is not the method's behaviour whatever the cause.
    * The topic extraction fell back to the raw text (M7.1): the method's own behaviour
      when the model's answer (or MIRIX itself) failed it, else the lost call's failure
      (:func:`extraction_failure`). ``retry_events`` are kept as evidence: an
      infrastructure failure the retries recovered left nothing degraded.
    """
    if evidence.get("success") is False:
        errors = evidence.get("errors") or {}
        kind = (
            FailureKind.INFRA
            if any(mirix_error_kind(e) is FailureKind.INFRA for e in errors.values())
            else FailureKind.HARNESS
        )
        return Failure(
            kind,
            MIRIX_RETRIEVAL,
            "mirix",
            None,
            f"MIRIX's retrieval failed: {json.dumps(errors, default=str)}"[:500],
        )
    if evidence.get("embedding_status") == "failed":
        failure = _call_failure(
            "embedding", evidence.get("embedding_error"), "query embedding"
        )
        return failure or Failure(
            FailureKind.INFRA,
            MIRIX_RETRIEVAL,
            "embedding",
            None,
            f"query embedding: {evidence.get('embedding_error')}"[:500],
        )
    if evidence.get("extraction_status") == "fallback":
        return extraction_failure(evidence.get("extraction_error"))
    return None


def refused_call_in(text: Any) -> Failure | None:
    """The refusal a MIRIX learning problem's text quotes, or None (pure).

    The gateway answer the error carries (:func:`gateway_error_in`) is judged by the
    harness's own rules (``classify_gateway_error``); it is a refusal when its class
    (``infra/failures.refusal_class``) is a payload the provider refused (413, or a 400
    naming its size or images), a
    no-endpoint 404, or a request the gateway itself refused. Not refusals: an answer
    without an HTTP 4xx, infrastructure (401, 402, 408, 429), the model's content filter,
    a plain 403 or 404, a 400 that names no size or image, and a refusal of
    the account (a data policy). When MIRIX's 500-character cut left the body without a
    complete ``message``, the text after the status stands in for it, so a cut 400 is
    judged by what survived of its words."""
    raw = str(text or "")
    if any(marker in raw.lower() for marker in ACCOUNT_REFUSAL_MARKERS):
        return None
    status, body = gateway_error_in(raw)
    if status is None or not 400 <= status < 500:
        return None
    if isinstance(body, Mapping) and isinstance(body.get("error"), Mapping):
        error = dict(body["error"])
    else:
        error = dict(body) if isinstance(body, Mapping) else {}
    if "message" not in error:
        match = _STATUS_IN_ERROR.search(raw)
        error["message"] = raw[match.end() :].strip(" -") if match else ""
    judged = classify_gateway_error(status, {"error": error}, MIRIX_ROLE)
    return judged if refusal_class(judged) is not None else None


#: The text :func:`learning_problem_kinds` gives a session MIRIX deferred.
_DEFERRED_PROBLEM = re.compile(r"^run \S+ deferred session (\S+) after ")


def untolerated_refusal(text: str, refusal: Failure) -> str | None:
    """Why the learning phase must not go on past this refusal (:func:`refused_call_in`)
    of a MIRIX learning problem, or None when it may be counted (pure).

    * MIRIX deferred the session (its M3.5 policy for a 401/403/404 and a gateway 409:
      ``defer_across_runs``). It refuses the session again on every retry and never
      settles it, so the freeze could never happen: going on would spend the whole
      learning phase and then fail at ``finalize``.
    * The gateway refused the call as a protocol break (a phase, model, role or run the
      call broke: :data:`PROTOCOL_REJECTION_TYPES`): a configuration bug every later
      call repeats, which the audit fails the cell for (H10.3).
    """
    deferred = _DEFERRED_PROBLEM.match(text)
    if deferred is not None:
        return (
            f"MIRIX deferred session {deferred.group(1)} after a refused call: it "
            "refuses the session again on every retry and never learns it, so the "
            "memory could not be frozen"
        )
    if refusal.code == GATEWAY_REJECTED:
        _, body = gateway_error_in(text)
        error = body.get("error") if isinstance(body, Mapping) else None
        error = error if isinstance(error, Mapping) else (body or {})
        kind = str(error.get("type") or "")
        if kind in PROTOCOL_REJECTION_TYPES:
            return (
                f"the gateway refused a MIRIX call as a protocol break ({kind}): every "
                "later call repeats it, and the audit fails the cell for it (H10.3)"
            )
    return None


def degraded_streaks(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Consecutive degraded test-phase retrievals per task uid, as the method's own
    ``mirix_retrievals.jsonl`` records them (pure): walking the ``retrieval`` rows of the
    frozen memory in order, a row with evidence and a retrieval problem (refused:
    ``failure``; injected: ``degraded``) extends its item's run, any other row (a clean
    retrieval, one MIRIX did not answer) ends it. Only the non-zero runs are returned."""
    streaks: dict[str, int] = {}
    for row in records:
        if row.get("event") != "retrieval" or row.get("frozen") is not True:
            continue
        uid = row.get("task_uid")
        if not isinstance(uid, str):
            continue
        degraded = isinstance(row.get("evidence"), Mapping) and bool(
            row.get("failure") or row.get("degraded")
        )
        streaks[uid] = streaks.get(uid, 0) + 1 if degraded else 0
    return {uid: n for uid, n in streaks.items() if n}


def _read_records(path: Path) -> list[dict[str, Any]]:
    """The JSON rows of one of the method's own JSONL records (blocking): split on "\n"
    only (a record may hold U+2028), torn lines skipped, a missing file
    none."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# The cell's server
# ---------------------------------------------------------------------------


class ServerHandle(Protocol):
    """What the method needs of the cell's MIRIX server (``MirixServer`` has it)."""

    @property
    def base_url(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self, grace_s: float = ...) -> int | None: ...

    async def ensure_alive(self) -> None: ...


ServerFactory = Callable[[CellContext, ServerMode], ServerHandle]


def default_server(cell: CellContext, mode: ServerMode) -> ServerHandle:
    """The cell attempt's own MIRIX (``infra/mirix_server.py``) on ``options["mirix_port"]``."""
    port = cell.options.get("mirix_port")
    if port is None:
        raise ValueError(
            "the MIRIX method needs options['mirix_port'] (a server it starts) or "
            "options['mirix_base_url'] (a server started by run.py)"
        )
    assert cell.cell_dir is not None  # CellContext fills it
    spec = CellServerSpec(cell.cell_id, cell.attempt, cell.cell_dir, port=int(port))
    return MirixServer(spec, mode=mode)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class MirixCounters:
    """Learning-side counts (written during the learning phase, frozen afterwards)."""

    messages_added: int = 0
    sessions_closed: int = 0
    #: Learning runs MIRIX reported as started by an add, a close or the flush.
    runs_started: int = 0
    #: ``warnings`` MIRIX returned for task-session writes (Q2: a session_tag mismatch).
    add_warnings: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class MirixState:
    cell_id: str
    attempt: int
    benchmark: str
    model: str
    org_id: str
    client_id: str
    user_id: str
    meta_agent_id: str
    config: dict[str, Any]
    counters: MirixCounters = field(default_factory=MirixCounters)
    frozen: bool = False
    frozen_snapshot_sha256: str | None = None
    flush: dict[str, Any] | None = None
    #: The task sessions ``after_task`` closed, in order (every one must be settled
    #: before the memory freezes, X3).
    closed_sessions: list[str] = field(default_factory=list)
    #: How many flushes ``finalize`` made (more than one: sessions with counted failures).
    flush_rounds: int = 0
    #: What the learning phase went on despite: counts by
    #: ``TOLERATED_*`` key, the accepted learning problems (``reason``, ``text``,
    #: ``rollbacks``), the consecutive learning items with a degraded retrieval, and the
    #: refusal streak of MIRIX's calls (``RefusalStreaks.to_dict``), and the learning
    #: runs that streak has seen, in the order it saw them (each finished run is one
    #: observation). A state of the code before them reads back with them
    #: empty.
    tolerated: dict[str, int] = field(default_factory=dict)
    accepted_problems: list[dict[str, Any]] = field(default_factory=list)
    degraded_streak: int = 0
    refusal_streaks: dict[str, Any] = field(default_factory=dict)
    refusal_runs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = STATE_VERSION
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MirixState:
        if data.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported state version {data.get('version')!r}")
        values = {k: v for k, v in data.items() if k != "version"}
        values["counters"] = MirixCounters(**values["counters"])
        return cls(**values)


def _write_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _procedural_items(response: Any) -> list[Mapping[str, Any]]:
    memories = response.get("memories") if isinstance(response, Mapping) else None
    procedural = memories.get("procedural") if isinstance(memories, Mapping) else None
    items = procedural.get("items") if isinstance(procedural, Mapping) else None
    return (
        [i for i in items if isinstance(i, Mapping)] if isinstance(items, list) else []
    )


def _entries(run: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = run.get(key)
    return (
        [e for e in value if isinstance(e, Mapping)] if isinstance(value, list) else []
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int:
    """A count MIRIX recorded, 0 for anything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


#: A session's ``distill_metrics`` status after MIRIX's own failure (a distill request it
#: could not build: kind ``internal``, M3.5): not counted, not deferred, the run fails.
INTERNAL_ERROR_STATUS = "internal_error"
#: Curator ``error_kind`` values of MIRIX's own failure: a request it could not build
#: (``internal``) or a bug (``other`` on a batch MIRIX kept pending instead of returning).
INTERNAL_ERROR_KINDS = ("internal", "other")
#: How a batch MIRIX kept pending settles (``_batch_outcome``: ``keep``).
BATCH_KEPT = "keep"
#: ``history_skipped`` of a curator batch that showed no earlier lessons because the
#: experience pool's embedding failed after MIRIX's M3.5 retries.
POOL_EMBEDDING_FAILED = "embedding_failed"


def pool_embedding_failure(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``history`` of a curator batch whose earlier lessons an experience-pool
    embedding failure cost, or None: it showed none
    (``history_skipped: embedding_failed``), or it left a pool experience out of its
    ranking (``history.embedding_error`` recorded while it still showed the rest)."""
    history = _mapping(entry.get("history"))
    if entry.get("history_skipped") == POOL_EMBEDDING_FAILED or history.get(
        "embedding_error"
    ):
        return history
    return None


def curation_configuration_skip(entry: Mapping[str, Any]) -> bool:
    """Whether a ``curator_metrics`` entry records a curator stage that MIRIX's
    configuration stopped before any batch. Its ``error_kind`` is
    ``config``, as for an LLM call MIRIX classified as a misconfigured deployment
    (401/403/404), but no call failed: the same configuration skips the stage on every
    run, so an INFRA resume only repeats it — a person must look (HARNESS)."""
    return (
        entry.get("status") == CURATION_SKIPPED_STATUS
        and entry.get("error_kind") == CONFIG_ERROR_KIND
    )


def distill_configuration_deferral(entry: Mapping[str, Any]) -> bool:
    """Whether a ``distill_metrics`` entry records a session the distiller deferred for
    its own configuration (:data:`DISTILL_CONFIGURATION_DEFERRALS`). Its
    ``failure_kind`` is ``config``, as after an LLM call MIRIX classified as a
    misconfigured deployment (401/403/404), but the same configuration defers the
    session on every run, so an INFRA resume only repeats it — a person must look
    (HARNESS)."""
    return (
        entry.get("status") == DEFERRED_STATUS
        and entry.get("config_deferral") in DISTILL_CONFIGURATION_DEFERRALS
    )


def learning_problem_kinds(
    runs: Sequence[Mapping[str, Any]],
) -> list[tuple[FailureKind, str]]:
    """What in MIRIX's learning runs (M9) an infrastructure failure, or MIRIX's own
    failure, cost, each with its H8 kind (pure; X3, X13).

    * a run that ``failed``: HARNESS when MIRIX failed it for its own failure (a session
      left ``internal_error``, a curator batch MIRIX kept after an ``internal`` or
      ``other`` error, a ``LearningRunInternalError``, or an error naming no database or
      network failure: :func:`mirix_error_kind`), else INFRA (a database error left after
      its retries, an expired lease);
    * a session ``deferred`` after an infrastructure or configuration error (INFRA, M3.5:
      it waits ``distill_retry_after``, up to an hour, and is learned later or never) —
      unless the distiller's own configuration deferred it
      (:func:`distill_configuration_deferral`: an input budget too small for any chunk,
      no LLM client): HARNESS, since every resume defers it again;
    * a session MIRIX could not distill for its own failure (``internal_error``: HARNESS,
      a re-run fails the same way);
    * a curator batch stopped by an infrastructure, configuration, database or embedding
      error (INFRA: its experiences wait for a later run), or kept after MIRIX's own
      failure (HARNESS);
    * a curator stage MIRIX's configuration skipped before any batch
      (:func:`curation_configuration_skip`: no write scope, no curator tool, a context
      window too small ...): HARNESS, since every resume skips it again;
    * a curator batch whose earlier lessons an experience-pool embedding failure cost
      (:func:`pool_embedding_failure`): it showed none (``history_skipped:
      embedding_failed``) or left a pool experience out of its ranking, so the memory is
      not what the attempt learns without the failure. The embedding model is
      infrastructure, never the model under test, so the failure is judged by the
      gateway's answer MIRIX recorded (:func:`_call_failure`: HARNESS when the gateway
      rejected MIRIX's request). Without one it is INFRA, as a lost query vector is
      (rule N1) — unless MIRIX classified the error as its own (``internal`` /
      ``other``: a vector it could not store, an empty answer it refused): HARNESS, as
      for a curator batch MIRIX kept after its own failure, since a restart repeats it.

    Model failures (truncated, unparseable or refused output, timeouts: counted per
    session, Q8) are MIRIX's own business and are not listed.
    """
    problems: list[tuple[FailureKind, str]] = []
    for run in runs:
        run_id = run.get("id")
        internal = False
        for entry in _entries(run, "distill_metrics"):
            if distill_configuration_deferral(entry):
                problems.append(
                    (
                        FailureKind.HARNESS,
                        f"run {run_id} deferred session {entry.get('session_id')} for "
                        f"MIRIX's configuration ({entry.get('config_deferral')}): "
                        f"{str(entry.get('error'))[:200]}",
                    )
                )
            elif entry.get("status") == DEFERRED_STATUS:
                problems.append(
                    (
                        FailureKind.INFRA,
                        f"run {run_id} deferred session {entry.get('session_id')} "
                        f"after {entry.get('failure_kind')}: "
                        f"{str(entry.get('error'))[:200]}",
                    )
                )
            elif entry.get("status") == INTERNAL_ERROR_STATUS:
                internal = True
                problems.append(
                    (
                        FailureKind.HARNESS,
                        f"run {run_id} could not distill session "
                        f"{entry.get('session_id')} for MIRIX's own failure: "
                        f"{str(entry.get('error'))[:200]}",
                    )
                )
        for entry in _entries(run, "curator_metrics"):
            kind = entry.get("error_kind")
            if curation_configuration_skip(entry):
                problems.append(
                    (
                        FailureKind.HARNESS,
                        f"run {run_id} skipped skill curation for MIRIX's configuration "
                        f"({entry.get('skipped')}, {entry.get('batch_id')}): "
                        f"{str(entry.get('error'))[:200]}",
                    )
                )
            elif kind in INFRA_ERROR_KINDS:
                problems.append(
                    (
                        FailureKind.INFRA,
                        f"run {run_id} curator batch {entry.get('batch_id')} stopped "
                        f"on {kind}: {str(entry.get('error'))[:200]}",
                    )
                )
            elif kind in INTERNAL_ERROR_KINDS and entry.get("outcome") == BATCH_KEPT:
                internal = True
                problems.append(
                    (
                        FailureKind.HARNESS,
                        f"run {run_id} curator batch {entry.get('batch_id')} stopped "
                        f"on MIRIX's own failure ({kind}): "
                        f"{str(entry.get('error'))[:200]}",
                    )
                )
            history = pool_embedding_failure(entry)
            if history is not None:
                error = history.get("embedding_error")
                error_kind = history.get("embedding_error_kind")
                failure = _call_failure("embedding", error, "experience pool embedding")
                kind = failure.kind if failure is not None else FailureKind.INFRA
                if (
                    failure is not None
                    and failure.status is None
                    and error_kind in INTERNAL_ERROR_KINDS
                ):
                    # No gateway answer, and MIRIX judged the error its own: a restart
                    # meets it again.
                    kind = FailureKind.HARNESS
                lost = (
                    "showed no earlier lessons"
                    if entry.get("history_skipped") == POOL_EMBEDDING_FAILED
                    else f"left {_count(history.get('embedding_failed'))} pool "
                    "experience(s) out of its earlier lessons"
                )
                problems.append(
                    (
                        kind,
                        f"run {run_id} curator batch {entry.get('batch_id')} {lost}: "
                        f"the experience pool's embedding failed ({error_kind}): "
                        f"{str(error)[:200]}",
                    )
                )
        if run.get("status") == "failed":
            error = str(run.get("error"))
            kind = (
                FailureKind.HARNESS
                if internal or error.startswith("LearningRunInternalError")
                else mirix_error_kind(error)
            )
            problems.append((kind, f"learning run {run_id} failed ({error[:200]})"))
    return problems


def learning_problems(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    """:func:`learning_problem_kinds` as text, each prefixed with its kind."""
    return [f"[{kind.value}] {text}" for kind, text in learning_problem_kinds(runs)]


#: Exceptions a MIRIX tool raises while it runs to reject the model's arguments (a time
#: string it cannot parse, a memory type it does not know, a payload its model refuses):
#: the model's own mistake, answered to it like a validation error, never MIRIX's (B31).
#: Only for a record without MIRIX's own verdict (``argument_error``): MIRIX decides it
#: on the exception object, so a subclass (a date parser's ``ParserError``), arguments
#: that do not fit the tool's signature and an id that names nothing count too.
MODEL_ARGUMENT_EXCEPTIONS = frozenset(
    {"ValueError", "ValidationError", "JSONDecodeError"}
)
#: The stage of a tool failure MIRIX records (``mirix/agent/tool_failures.py``) whose
#: exception came from the tool running, or from MIRIX right after it ran.
TOOL_EXECUTION_STAGE = "execution"


def curator_tool_failures(
    runs: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """The exceptions curator tools raised while they ran that were not the model's own
    mistakes (B31, pure): ``(label, text)`` with label ``infra`` when the exception names
    a database or network failure (:func:`mirix_error_kind`), else ``mirix`` (MIRIX's own
    error, e.g. a ``DBAPIError`` its query caused).

    MIRIX answers every tool failure to the model as a failed tool result, so none of
    them fails a batch or a run (:func:`learning_problem_kinds` never sees them); a
    MIRIX that records why its tool calls failed lists them in each curator batch's
    ``tool_failures`` (tool, stage, exception class, count, first error). Only the
    ``execution`` stage can be MIRIX's or the environment's: an unknown tool, arguments
    that are not JSON or fail validation, an error text a tool returned, and an
    exception MIRIX judged the model's (``argument_error``; a record without it: the
    exceptions of :data:`MODEL_ARGUMENT_EXCEPTIONS`) are the model's. A record without
    ``tool_failures`` (a MIRIX before it) gives nothing."""
    found: list[tuple[str, str]] = []
    for run in runs:
        for entry in _entries(run, "curator_metrics"):
            failures = entry.get("tool_failures")
            if not isinstance(failures, list):
                continue
            for failure in failures:
                if not isinstance(failure, Mapping):
                    continue
                exception = failure.get("exception")
                verdict = failure.get("argument_error")
                if (
                    failure.get("stage") != TOOL_EXECUTION_STAGE
                    or not isinstance(exception, str)
                    or not exception
                    or verdict is True
                    or (
                        not isinstance(verdict, bool)
                        and exception in MODEL_ARGUMENT_EXCEPTIONS
                    )
                ):
                    continue
                error = str(failure.get("error") or "")
                label = (
                    "infra"
                    if mirix_error_kind(f"{exception}: {error}") is FailureKind.INFRA
                    else "mirix"
                )
                found.append(
                    (
                        label,
                        f"run {run.get('id')} curator batch {entry.get('batch_id')}: "
                        f"{_count(failure.get('count'))} call(s) of "
                        f"{failure.get('tool')} raised {exception} while it ran and were "
                        f"answered to the model as failed tool results: {error[:200]}",
                    )
                )
    return found


def learning_warnings(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    """:func:`curator_tool_failures` as text, each prefixed with its label: what the
    attempt learned despite (logged by the method, reported by the audit as warnings;
    never a reason to invalidate the attempt)."""
    return [f"[{label}] {text}" for label, text in curator_tool_failures(runs)]


def curated_skill_ids(runs: Sequence[Mapping[str, Any]]) -> set[str]:
    """Every skill id a curator batch of ``runs`` created, edited or deleted
    (``curator_metrics[].changes``, M9)."""
    ids: set[str] = set()
    for run in runs:
        for entry in _entries(run, "curator_metrics"):
            changes = entry.get("changes")
            if not isinstance(changes, Mapping):
                continue
            for kind in ("created", "edited", "deleted"):
                value = changes.get(kind)
                if isinstance(value, list):
                    ids.update(str(i) for i in value if i)
    return ids


def unsettled_sessions(
    runs: Sequence[Mapping[str, Any]], sessions: Sequence[str]
) -> dict[str, str]:
    """Each of ``sessions`` MIRIX has not settled, with its state (pure; X3).

    A session is settled when the last run that claimed it succeeded and distilled it
    (``ok``) or gave it up after its third counted failure (``failed_permanently``).
    Otherwise its state is ``failed`` (a counted failure: MIRIX takes it up again),
    ``deferred`` (an infrastructure back-off), the run's status (still ``queued`` /
    ``running``, or ``failed``), ``unreported`` (the run has no metrics for it) or
    ``unclaimed`` (no run ever claimed it).
    """
    ordered = sorted(
        runs, key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or ""))
    )
    last: dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]] = {}
    for run in ordered:
        metrics = {
            str(e.get("session_id")): e for e in _entries(run, "distill_metrics")
        }
        for sid in run.get("session_ids") or ():
            last[str(sid)] = (run, metrics.get(str(sid)))
    found = {}
    for sid in sessions:
        if sid not in last:
            found[sid] = "unclaimed"
            continue
        run, entry = last[sid]
        if run.get("status") != "succeeded":
            found[sid] = str(run.get("status"))
        elif entry is None:
            found[sid] = "unreported"
        elif entry.get("status") not in SETTLED_SESSION_STATUSES:
            found[sid] = str(entry.get("status"))
    return found


# ---------------------------------------------------------------------------
# The experience pool
# ---------------------------------------------------------------------------

#: What became of a curator batch's new experiences (the ``dispositions`` of a settled
#: ``curator_metrics`` entry): a committed skill write cited it (``used``, also when it
#: was closed out), an existing skill covers it, it is not worth a skill, or it is kept
#: for later (``deferred``; ``defaulted`` counts the deferred ones the curator did not
#: triage, so it is part of ``deferred``); ``unaddressed``: closed out after its third
#: curation attempt with no citation.
POOL_DISPOSITIONS: tuple[str, ...] = (
    "used",
    "covered",
    "not_worth",
    "deferred",
    "defaulted",
    "unaddressed",
)
#: The dispositions that put an experience into the pool (``unaddressed`` never does).
POOL_ENTERING: tuple[str, ...] = ("not_worth", "deferred")
#: Why a pool experience left the pool (``retired``): a skill write cited it, the curator
#: found it covered, it was ignored too often, it was a new experience that recurred a
#: lesson still in the pool (``merged`` into it), or its embedding kept failing
#: (``unembeddable``). A MIRIX before the last two reports neither: they count 0.
POOL_RETIRE_REASONS: tuple[str, ...] = (
    "used",
    "covered",
    "ignored",
    "merged",
    "unembeddable",
)
#: Counts a settled curator batch reports, summed over batches: the earlier lessons it
#: showed (``history_pushed``, a returned batch's included) and what became of them
#: (cited, covered, listed not worth, ignored), the pool experiences it saw again
#: (``recurrences``), the pool experiences whose failed embedding it counted
#: (``embedding_failures``), its creates / edits and its deletions that cited no
#: experience, its accepted ``experience_triage`` calls and the citations it did not
#: count for want of a quote of the written text that applies them
#: (``citations_unverified``, B12; a MIRIX before B12 reports none: 0).
POOL_COUNTS: tuple[str, ...] = (
    "history_pushed",
    "history_cited",
    "history_covered",
    "history_not_worth",
    "history_ignored",
    "recurrences",
    "embedding_failures",
    "uncited_writes",
    "uncited_deletes",
    "triage_calls",
    "citations_unverified",
)
#: Per-batch counts of how the earlier lessons were chosen (a batch's ``history``),
#: summed over batches: the pool experiences ranked, those at or above the push and the
#: recurrence thresholds, those left out for lack of room, the texts embedded, those
#: whose embedding failed and the pool vectors that could not be compared.
POOL_HISTORY_COUNTS: tuple[str, ...] = (
    "candidates",
    "above_min",
    "recurrence_candidates",
    "dropped_for_room",
    "embedded",
    "embedding_failed",
    "unusable_vectors",
)


def _pool_batches(run: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int]:
    """A run's curator batches that report the pool (settled by a MIRIX with the
    experience pool: their entry carries ``dispositions``), in batch order, and how many
    do not."""
    entries = _entries(run, "curator_metrics")
    reported = [e for e in entries if isinstance(e.get("dispositions"), Mapping)]
    return reported, len(entries) - len(reported)


def run_pool_summary(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """What one learning run's curator batches did to the experience pool (pure, from
    their ``curator_metrics`` entries), or None when none of them reports it (a MIRIX
    without the experience pool, or a run without a settled curator batch).

    ``pool_active_after`` is the pool's size after the run's last batch settled."""
    reported, unreported = _pool_batches(run)
    if not reported:
        return None
    last = reported[-1].get("pool_active_after")
    skipped = Counter(
        str(e["history_skipped"]) for e in reported if e.get("history_skipped")
    )
    return {
        "run_id": run.get("id"),
        "batches": len(reported),
        "unreported_batches": unreported,
        "dispositions": {
            key: sum(_count(_mapping(e.get("dispositions")).get(key)) for e in reported)
            for key in POOL_DISPOSITIONS
        },
        **{key: sum(_count(e.get(key)) for e in reported) for key in POOL_COUNTS},
        "retired": {
            key: sum(_count(_mapping(e.get("retired")).get(key)) for e in reported)
            for key in POOL_RETIRE_REASONS
        },
        "history_skipped": dict(sorted(skipped.items())),
        "pool_active_after": _count(last) if last is not None else None,
    }


def _nearest_rank(ordered: Sequence[float], q: float) -> float:
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _pool_history(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How the earlier lessons were chosen, over ``batches``: :data:`POOL_HISTORY_COUNTS`
    summed, the batches whose pool scan was cut, the max cosines MIRIX logged (at most
    50 per batch, for calibrating the two provisional thresholds: their count and
    nearest-rank p50, p90 and max) and the thresholds the last batch used."""
    histories = [_mapping(e.get("history")) for e in batches]
    cosines = sorted(
        float(c["max_cosine"])
        for h in histories
        for c in h.get("max_cosines") or ()
        if isinstance(c, Mapping)
        and isinstance(c.get("max_cosine"), (int, float))
        and not isinstance(c.get("max_cosine"), bool)
        and math.isfinite(c["max_cosine"])
    )
    thresholds = next(
        (
            dict(h["thresholds"])
            for h in reversed(histories)
            if isinstance(h.get("thresholds"), Mapping)
        ),
        None,
    )
    return {
        **{
            key: sum(_count(h.get(key)) for h in histories)
            for key in POOL_HISTORY_COUNTS
        },
        # Earlier lessons shown with their content cut to the room left (M5.7).
        "lessons_cut": sum(
            len(h["lessons_cut"])
            for h in histories
            if isinstance(h.get("lessons_cut"), list)
        ),
        "scan_truncated_batches": sum(
            1 for h in histories if h.get("pool_scan_truncated") is True
        ),
        "max_cosines": {
            "logged": len(cosines),
            "p50": round(_nearest_rank(cosines, 0.5), 4),
            "p90": round(_nearest_rank(cosines, 0.9), 4),
            "max": round(cosines[-1], 4),
        }
        if cosines
        else {"logged": 0},
        "thresholds": thresholds,
    }


def experience_pool(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The experience pool of the frozen memory (pure), from the learning runs' curator
    batches, or None when no run reports it (a MIRIX without the experience pool).

    MIRIX has no endpoint for experiences; every settled curator batch reports what it
    did to the pool (its ``curator_metrics`` entry), and nothing else changes the pool,
    so the runs tell its state at the freeze: ``active_at_freeze`` (the size after the
    last batch settled, runs by ``(created_at, id)``), the experiences that ``entered``
    it (disposition ``not_worth`` or ``deferred``) and those ``retired`` (used, covered,
    ignored). ``unaccounted`` = entered - retired - active is 0 when MIRIX's pool
    metrics add up. Also the dispositions and :data:`POOL_COUNTS` summed, the skip
    reasons of the batches that showed no earlier lessons (``history_skipped``), how
    the lessons were chosen (``history``: :func:`_pool_history`) and each run's own
    summary (``per_run``, :func:`run_pool_summary`). The snapshot carries it, so the
    frozen hash covers it as it covers the runs it comes from."""
    ordered = sorted(
        runs, key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or ""))
    )
    per_run = [s for s in (run_pool_summary(r) for r in ordered) if s is not None]
    if not per_run:
        return None
    dispositions = {
        key: sum(s["dispositions"][key] for s in per_run) for key in POOL_DISPOSITIONS
    }
    retired = {
        key: sum(s["retired"][key] for s in per_run) for key in POOL_RETIRE_REASONS
    }
    entered = sum(dispositions[key] for key in POOL_ENTERING)
    active = per_run[-1]["pool_active_after"]
    skipped: Counter[str] = Counter()
    for summary in per_run:
        skipped.update(summary["history_skipped"])
    return {
        "batches": sum(s["batches"] for s in per_run),
        # Every run's: one whose batches all lack the pool has no summary of its own.
        "unreported_batches": sum(_pool_batches(run)[1] for run in ordered),
        "dispositions": dispositions,
        "entered": entered,
        "retired": retired,
        "active_at_freeze": active,
        "unaccounted": entered - sum(retired.values()) - active
        if active is not None
        else None,
        **{key: sum(s[key] for s in per_run) for key in POOL_COUNTS},
        "history_skipped": dict(sorted(skipped.items())),
        "history": _pool_history(
            [batch for run in ordered for batch in _pool_batches(run)[0]]
        ),
        "per_run": per_run,
    }


#: The ``skill_feedback`` counts a distilled session reports, summed over
#: sessions: the skills served, those whose header the context cut left out, those the
#: distiller gave an entry for, and the stated ``followed`` / ``effect`` values no quote
#: verified (stored unclear).
FEEDBACK_COUNTS: tuple[str, ...] = (
    "served",
    "not_shown",
    "given",
    "followed_unverified",
    "effect_unverified",
)


def _distilled_sessions(
    runs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """The ``distill_metrics`` entry each session was learned from (status ``ok``; the
    last one, runs by ``(created_at, id)``), in that order."""
    ordered = sorted(
        runs, key=lambda r: (str(r.get("created_at") or ""), str(r.get("id") or ""))
    )
    last: dict[str, Mapping[str, Any]] = {}
    for run in ordered:
        for entry in _entries(run, "distill_metrics"):
            if entry.get("status") == "ok":
                last.pop(str(entry.get("session_id")), None)
                last[str(entry.get("session_id"))] = entry
    return list(last.values())


def learning_context(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """What MIRIX's distiller was shown with each learned session and what it said of
    the skills served in it, and the usage records its curator batches showed, from the
    runs' ``distill_metrics`` and ``curator_metrics`` (pure). None when no learned
    session reports its context (a MIRIX without agent contexts).

    ``sessions``: the learned sessions (each once, :func:`_distilled_sessions`);
    ``with_agent_context`` of them were distilled with the agent context the client
    attached (``agent_context_chars`` > 0), ``without_agent_context``
    names the others that had messages, ``agent_context_chars`` the lengths MIRIX
    stored (each once); ``context_cut``: sessions whose context MIRIX cut to fit
    (``context_truncated_tokens`` > 0, summed in ``context_truncated_tokens``).
    ``served_sessions`` were served at least one skill (``served_skills`` in all,
    ``served_per_session`` by count, ``served_skills_hidden`` the served skills the
    context left out). ``feedback``: :data:`FEEDBACK_COUNTS` summed and the stored
    ``followed`` / ``effect`` values counted; ``skills``: each served skill's (by
    ``skill_id``, most served first). ``skills_recorded`` is False when sessions had
    feedback but MIRIX did not record each skill's values (a MIRIX before it did).
    ``usage``: the curator batches that report a usage section (``batches``), those
    that showed records (``shown``), the records shown (``records``; ``served`` of them
    for skills served in the batch's sessions), the batches that showed one line per
    skill (``compact``), the skills left out for room (``left_out``), the section's
    ``tokens`` and why a batch showed none (``skipped``). ``citations_unverified``:
    the curator's citations not counted for want of a verified quote (B12)."""
    sessions = [e for e in _distilled_sessions(runs) if "agent_context_chars" in e]
    if not sessions:
        return None
    feedback = {key: 0 for key in FEEDBACK_COUNTS}
    followed: Counter[str] = Counter()
    effect: Counter[str] = Counter()
    skills: dict[str, dict[str, Any]] = {}
    recorded = True
    for entry in sessions:
        given = _mapping(entry.get("skill_feedback"))
        if not given:
            continue
        for key in FEEDBACK_COUNTS:
            feedback[key] += _count(given.get(key))
        rows = given.get("skills")
        if not isinstance(rows, list):
            recorded = recorded and not _count(given.get("served"))
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            skill = skills.setdefault(
                str(row.get("skill_id")),
                {
                    "skill_id": str(row.get("skill_id")),
                    "name": row.get("name"),
                    "served": 0,
                    "followed": Counter(),
                    "effect": Counter(),
                },
            )
            skill["name"] = row.get("name") or skill["name"]
            skill["served"] += 1
            skill["followed"][str(row.get("followed"))] += 1
            skill["effect"][str(row.get("effect"))] += 1
            followed[str(row.get("followed"))] += 1
            effect[str(row.get("effect"))] += 1
    served = Counter(_count(e.get("served_skills")) for e in sessions)
    batches = [
        e
        for run in runs
        for e in _entries(run, "curator_metrics")
        if isinstance(e.get("skill_usage"), Mapping)
    ]
    usages = [_mapping(e.get("skill_usage")) for e in batches]

    def _ids(usage: Mapping[str, Any], key: str) -> int:
        value = usage.get(key)
        return len(value) if isinstance(value, list) else 0

    return {
        "sessions": len(sessions),
        "with_agent_context": sum(
            1 for e in sessions if _count(e.get("agent_context_chars")) > 0
        ),
        "without_agent_context": sorted(
            str(e.get("session_id"))
            for e in sessions
            if _count(e.get("agent_context_chars")) == 0
            and ("messages" not in e or _count(e.get("messages")) > 0)
        ),
        "agent_context_chars": sorted(
            {_count(e.get("agent_context_chars")) for e in sessions} - {0}
        ),
        "context_cut": sum(
            1 for e in sessions if _count(e.get("context_truncated_tokens")) > 0
        ),
        "context_truncated_tokens": sum(
            _count(e.get("context_truncated_tokens")) for e in sessions
        ),
        "served_sessions": sum(n for k, n in served.items() if k > 0),
        "served_skills": sum(k * n for k, n in served.items()),
        "served_per_session": {str(k): n for k, n in sorted(served.items())},
        "served_skills_hidden": sum(
            _count(e.get("served_skills_hidden")) for e in sessions
        ),
        "feedback": {
            **feedback,
            "followed": dict(sorted(followed.items())),
            "effect": dict(sorted(effect.items())),
        },
        "skills_recorded": recorded,
        "skills": [
            {
                **s,
                "followed": dict(sorted(s["followed"].items())),
                "effect": dict(sorted(s["effect"].items())),
            }
            for s in sorted(
                skills.values(), key=lambda s: (-s["served"], str(s["name"]))
            )
        ],
        "usage": {
            "batches": len(batches),
            "shown": sum(1 for u in usages if _ids(u, "skill_ids")),
            "records": sum(_ids(u, "skill_ids") for u in usages),
            "served": sum(_ids(u, "served_ids") for u in usages),
            "compact": sum(1 for u in usages if u.get("compact") is True),
            "left_out": sum(_ids(u, "left_out_ids") for u in usages),
            "tokens": sum(_count(u.get("tokens")) for u in usages),
            "skipped": dict(
                sorted(
                    Counter(
                        str(u["skipped"]) for u in usages if u.get("skipped")
                    ).items()
                )
            ),
        },
        "citations_unverified": sum(
            _count(e.get("citations_unverified"))
            for run in runs
            for e in _entries(run, "curator_metrics")
        ),
    }


def run_outcome(run: Mapping[str, Any]) -> dict[str, Any]:
    """A learning run in a few numbers (``mirix_learning.jsonl``, the learning summary):
    its status, sessions and their distill outcomes, experiences written, curator writes,
    what its curator batches did to the experience pool (:func:`run_pool_summary`) and
    what its sessions were distilled with (:func:`learning_context`)."""
    distill = _entries(run, "distill_metrics")
    curator = _entries(run, "curator_metrics")
    statuses: dict[str, int] = {}
    for entry in distill:
        key = str(entry.get("status"))
        statuses[key] = statuses.get(key, 0) + 1
    return {
        "run_id": run.get("id"),
        "trigger": run.get("trigger"),
        "status": run.get("status"),
        "error": run.get("error"),
        "sessions": len(run.get("session_ids") or ()),
        "session_statuses": statuses,
        "experiences": sum(int(e.get("experiences") or 0) for e in distill),
        "curator_batches": len(curator),
        "curator_writes": sum(int(e.get("writes") or 0) for e in curator),
        "experience_pool": run_pool_summary(run),
        "learning_context": learning_context([run]),
        "llm_usage": run.get("llm_usage"),
    }


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------


class MirixMethod(MemoryMethodBase):
    """MIRIX through its native API (S0: it learns from raw trajectories, no labels).

    Zero-argument construction is what ``run.py`` uses: the server comes from
    ``cell.options`` (``mirix_port``: the method starts and stops the cell's server;
    ``mirix_base_url``: a server someone else runs). Tests pass ``server_factory`` (a
    stub MIRIX server) and, where they need to, ``renderer`` in place of MIRIX's own.
    ``poll_s`` / ``finalize_timeout_s`` default to ``options["mirix_poll_s"]`` /
    ``options["mirix_finalize_timeout_s"]`` or the module constants.
    """

    name = "mirix"
    signal_level: SignalLevel = "S0"

    def __init__(
        self,
        *,
        server_factory: ServerFactory | None = None,
        renderer: Renderer | None = None,
        poll_s: float | None = None,
        finalize_timeout_s: float | None = None,
    ) -> None:
        super().__init__()
        self._server_factory = server_factory
        self._renderer = renderer
        self._poll_s = poll_s
        self._finalize_timeout_s = finalize_timeout_s
        self.server: ServerHandle | None = None
        self._api: MirixApi | None = None
        self._state: MirixState | None = None
        #: Test phase: consecutive degraded retrievals per task uid (never in the state
        #: or the snapshot: the test phase does not change the memory, H10.4).
        self._degraded_test: dict[str, int] = {}
        #: Learning resume: how often in a row this checkpoint was rolled back
        #: (``LearningResume.rollbacks``), and whether the first idle check after the
        #: resume is still to come.
        self._rollbacks = 0
        self._after_resume = False
        #: ``tolerated`` records the idle check accepted, written after it.
        self._pending_records: list[dict[str, Any]] = []
        self._record_lock = asyncio.Lock()
        #: The learning runs the last idle wait saw (every run of this tenant).
        self._runs: list[dict[str, Any]] = []
        #: The learning warnings already logged (each idle check lists every run).
        self._warned: set[str] = set()
        #: Learning phase: the agent context each task's ``before_task`` received,
        #: sent with the first add of its session and dropped once MIRIX stored it (so
        #: none is ever pending at a checkpoint, which is taken between items or after
        #: an item's last message).
        self._agent_contexts: dict[str, str] = {}

    # --- H6 hooks -------------------------------------------------------------

    async def setup(self, cell: CellContext) -> None:
        await super().setup(cell)
        self._agent_contexts = {}
        library = await load_mirix()
        self._renderer = self._renderer or library.render
        template = setup_template()
        try:
            base_url = await self._start_server(cell)
            self._api = MirixApi(
                library,
                base_url,
                client=template["client"],
                organization=template["organization"],
                on_transport_error=self._server_alive,
            )
            missing = missing_features(await self._api.openapi())
            if missing:
                raise MirixApiError(
                    "this MIRIX lacks what the method needs: " + "; ".join(missing)
                )
            if cell.learning and cell.resume is not None:
                await self._resume_learning(cell)
            elif cell.learning:
                await self._set_up_learning(cell)
            else:
                await self._resume(cell)
        except BaseException:
            self._state = None
            await self.close()  # never leave a server running behind a failed setup
            raise

    async def checkpoint(self) -> dict[str, Any]:
        """Contract item 6: the method's state (ids, counters, closed sessions); MIRIX's
        memory itself is the cell database, which ``run.py`` dumps with the checkpoint."""
        state, _ = self._require_learning()
        return state.to_dict()

    async def before_task(self, task: TaskView) -> Injection:
        state, api = self._require()
        template = setup_template()
        started = _now()
        retrieval_id: Any = None
        try:
            response = await api.retrieve_with_conversation(
                state.user_id,
                list(task.first_messages),
                limit=int(template["retrieve_limit"]),
                memory_types=list(template["memory_types"]),
                session_id=task.task_uid,
            )
            if not isinstance(response, Mapping):
                raise MirixApiError(f"MIRIX retrieval answered {str(response)[:500]}")
            retrieval_id = response.get("retrieval_id")
            row = await self._retrieval_row(task.task_uid, retrieval_id)
        except Exception as exc:
            # MIRIX did not answer (a timeout, a 5xx) or its log row is missing: the item
            # is not scored (H8), and the gateway calls MIRIX made meanwhile belong to a
            # refused retrieval, whose window the audit reads. Not a degraded
            # retrieval: it ends the item's run of them, as degraded_streaks reads it.
            if state.frozen:
                self._degraded_test.pop(task.task_uid, None)
            await self._record(
                RETRIEVAL_RECORD_FILE,
                {
                    "event": "retrieval",
                    "started_at": started,
                    "at": _now(),
                    "task_uid": task.task_uid,
                    "frozen": state.frozen,
                    "retrieval_id": retrieval_id,
                    "item_ids": [],
                    "evidence": None,
                    "procedural": None,
                    "failure": classify_exception(exc, "mirix").to_dict(),
                    "injection_sha256": None,
                    "injection_chars": None,
                },
            )
            raise
        evidence = retrieval_evidence(response, row)
        problem = retrieval_problem(evidence)
        streak, accepted = self._degraded(task.task_uid, state, problem)
        injected = problem is None or accepted
        items = _procedural_items(response)
        ids = tuple(str(i["id"]) for i in items if "id" in i)
        text = ""
        if injected:
            assert self._renderer is not None
            text = self._renderer(response)
            if not isinstance(text, str):
                raise MirixApiError(
                    f"MIRIX's renderer returned {type(text).__name__}, not text"
                )
        await self._record(
            RETRIEVAL_RECORD_FILE,
            {
                "event": "retrieval",
                "started_at": started,
                "at": _now(),
                "task_uid": task.task_uid,
                "frozen": state.frozen,
                "retrieval_id": retrieval_id,
                "item_ids": list(ids),
                "retrieval_scores": {
                    str(i["id"]): i.get("retrieval_scores") for i in items if "id" in i
                },
                "topics": response.get("topics"),
                # The task-level part of the topics that ranked the skills
                # (goal and operations); None when MIRIX ranked with the topics.
                "procedural_query": response.get("procedural_query"),
                "evidence": evidence,
                "procedural": procedural_trace(row),
                "failure": problem.to_dict()
                if problem is not None and not accepted
                else None,
                "injection_sha256": _sha256(text) if injected else None,
                "injection_chars": len(text) if injected else None,
                # Only with a retrieval problem: the item's consecutive degraded
                # retrievals, and the problem of one injected despite it.
                **({"degraded_streak": streak} if problem is not None else {}),
                **({"degraded": problem.to_dict()} if accepted and problem else {}),
            },
        )
        if problem is not None and not accepted:
            raise MirixRetrievalDegraded(problem)
        if not state.frozen and task.agent_context.strip():
            # Only learning sessions are distilled, so only they carry it.
            self._agent_contexts[task.task_uid] = task.agent_context
        return Injection(text, item_ids=ids, degraded=accepted)

    def _degraded(
        self, task_uid: str, state: MirixState, problem: Failure | None
    ) -> tuple[int, bool]:
        """The item's consecutive degraded retrievals and whether this one is injected
        (rule N1): in the test phase the
        :data:`MIRIX_DEGRADED_RETRIEVAL_TRIES`-th in a row of one item (kept outside the
        state: the test phase never changes the memory); in the learning phase every one
        but the ``REFUSAL_STREAK_ITEMS``-th consecutive learning item with one (counted
        in ``tolerated``)."""
        if state.frozen:
            if problem is None:
                self._degraded_test.pop(task_uid, None)
                return 0, False
            streak = self._degraded_test.get(task_uid, 0) + 1
            self._degraded_test[task_uid] = streak
            return streak, streak >= MIRIX_DEGRADED_RETRIEVAL_TRIES
        if problem is None:
            state.degraded_streak = 0
            return 0, False
        streak = state.degraded_streak + 1
        state.degraded_streak = streak
        accepted = streak < REFUSAL_STREAK_ITEMS
        if accepted:
            state.tolerated[TOLERATED_RETRIEVALS] = (
                state.tolerated.get(TOLERATED_RETRIEVALS, 0) + 1
            )
        return streak, accepted

    async def on_message(self, task_uid: str, message: Message) -> None:
        state, api = self._require_learning()
        agent_context = self._agent_contexts.get(task_uid)
        response = await api.add(
            state.user_id,
            [message],
            session_id=task_uid,
            session_tag=str(setup_template()["session_tag"]),
            agent_context=agent_context,
        )
        if not isinstance(response, Mapping) or response.get("status") != "stored":
            raise MirixApiError(
                "MIRIX did not store the message as a task session: "
                f"{str(response)[:500]}"
            )
        if agent_context is not None:
            # A stored task-session add stored a turn, and the agent context with it
            # (MIRIX refuses a task-session add without one, 422).
            self._agent_contexts.pop(task_uid, None)
        state.counters.messages_added += 1
        warnings = response.get("warnings") or []
        state.counters.add_warnings += len(warnings)
        if warnings or response.get("learning_run_id"):
            if response.get("learning_run_id"):
                state.counters.runs_started += 1
            await self._record(
                LEARNING_RECORD_FILE,
                {
                    "event": "add",
                    "at": _now(),
                    "task_uid": task_uid,
                    "warnings": warnings,
                    "learning_run_id": response.get("learning_run_id"),
                },
            )

    async def after_task(self, trajectory: TrajectoryView) -> None:
        state, api = self._require_learning()
        self._agent_contexts.pop(trajectory.task_uid, None)
        response = await api.close_session(trajectory.task_uid, state.user_id)
        if not isinstance(response, Mapping):
            raise MirixApiError(f"unexpected close_session reply: {response!r}")
        if response.get("already_closed"):
            raise MirixApiError(
                f"session {trajectory.task_uid} was already closed: a task session is "
                "learned from once (H8 fairness)"
            )
        state.counters.sessions_closed += 1
        state.closed_sessions.append(trajectory.task_uid)
        if response.get("learning_run_id"):
            state.counters.runs_started += 1
        await self._record(
            LEARNING_RECORD_FILE,
            {
                "event": "close",
                "at": _now(),
                "task_uid": trajectory.task_uid,
                "learning_run_id": response.get("learning_run_id"),
                "closed_at": response.get("closed_at"),
            },
        )
        await self._persist()

    async def wait_until_idle(self, timeout_s: float) -> None:
        state, api = self._require()
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            runs = await api.list_learning_runs(
                user_id=state.user_id,
                limit=LEARNING_RUNS_LIMIT,
                meta_agent_id=state.meta_agent_id,
            )
            if len(runs) >= LEARNING_RUNS_LIMIT:
                raise MirixApiError(
                    f"{len(runs)} learning runs: more than one page, the idle check "
                    "would be incomplete"
                )
            self._runs = runs
            active = [r for r in runs if r.get("status") in ACTIVE_RUN_STATUSES]
            if not active:
                if not state.frozen:
                    self._check_learning(runs)
                    await self._write_tolerated()
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                listed = ", ".join(
                    f"{run.get('id')}={run.get('status')}" for run in active[:10]
                )
                raise TimeoutError(
                    f"MIRIX learning still in flight after {timeout_s:g} s: {listed}"
                )
            await asyncio.sleep(min(self.poll_s, remaining))

    async def finalize(self) -> None:
        """Flush until MIRIX has nothing left to learn, then freeze (M1.7, Q8, X3).

        Each flush starts a run for what is pending: the closed sessions not yet settled
        (a counted failure is taken up again) and the experiences a curator batch
        returned with a counted attempt (a curate-only run). The loop ends at the first
        flush that starts no run, within :data:`MAX_FLUSH_ROUNDS` and one deadline,
        ``finalize_timeout_s`` for all the rounds together. A flush that starts no run
        while a session with a counted failure is still unsettled raises
        :class:`MirixApiError` instead of freezing without it, unless a run that started
        since MIRIX last listed its runs (its sweeper's) claimed that session just
        before the flush: then the flush is repeated."""
        state, api = self._require_learning()
        deadline = time.monotonic() + self.finalize_timeout_s
        retried: set[str] = set()
        for _ in range(MAX_FLUSH_ROUNDS):
            # The runs MIRIX listed last, while idle (the previous round's wait, or the
            # last learning item's): a run not among them started since, e.g. MIRIX's
            # sweeper claiming a session just before this flush.
            listed = {str(r.get("id")) for r in self._runs}
            response = await api.flush_skills(
                user_id=state.user_id,
                close_open_sessions=bool(setup_template()["flush_close_open_sessions"]),
                meta_agent_id=state.meta_agent_id,
            )
            if not isinstance(response, Mapping):
                raise MirixApiError(f"unexpected flush_skills reply: {response!r}")
            state.flush = {k: response.get(k) for k in sorted(response)}
            state.flush_rounds += 1
            run_id = response.get("learning_run_id")
            if run_id:
                state.counters.runs_started += 1
            await self._record(
                LEARNING_RECORD_FILE,
                {
                    "event": "flush",
                    "at": _now(),
                    "round": state.flush_rounds,
                    **state.flush,
                },
            )
            await self.wait_until_idle(max(0.0, deadline - time.monotonic()))
            if run_id:
                run = await api.get_learning_run(str(run_id))
                if run.get("status") not in TERMINAL_RUN_STATUSES:
                    raise MirixApiError(
                        f"flush run {run_id} is {run.get('status')!r} although MIRIX "
                        "lists no active run"
                    )
                await self._record(
                    LEARNING_RECORD_FILE,
                    {
                        "event": "flush_run",
                        "at": _now(),
                        "round": state.flush_rounds,
                        **run_outcome(run),
                    },
                )
            left = unsettled_sessions(self._runs, state.closed_sessions)
            retried = {s for s, why in left.items() if why == COUNTED_FAILURE_STATUS}
            if set(left) - retried:
                raise self._infra(
                    "sessions not learned before the freeze: "
                    + ", ".join(f"{s}={why}" for s, why in sorted(left.items()))
                )
            if not run_id:
                claimed_elsewhere = sorted(
                    str(r.get("id"))
                    for r in self._runs
                    if str(r.get("id")) not in listed
                    and retried & {str(s) for s in r.get("session_ids") or ()}
                )
                if retried and claimed_elsewhere:
                    # Another run (MIRIX's sweeper, a threshold check) claimed the session
                    # just before this flush, which then had nothing to claim; that run
                    # failed it again. Not MIRIX breaking its claim rules: flush again
                    # (MAX_FLUSH_ROUNDS still bounds the rounds).
                    await self._record(
                        LEARNING_RECORD_FILE,
                        {
                            "event": "flush_preempted",
                            "at": _now(),
                            "round": state.flush_rounds,
                            "runs": claimed_elsewhere,
                            "sessions": sorted(retried),
                        },
                    )
                    continue
                if retried:
                    # A session with a counted failure is pending again at once (Q8: no
                    # retry_after), so a flush must take it up. One that starts no run
                    # while no other run took the session up would freeze without that
                    # session's learning: MIRIX contradicts its own claim
                    # rules, a person looks (HARNESS).
                    raise MirixApiError(
                        f"flush round {state.flush_rounds} started no learning run "
                        "although sessions with counted failures are not settled: "
                        + ", ".join(sorted(retried))
                        + f" (flush reply: {state.flush})"
                    )
                break  # no pending session and no experience waiting for the curator
        else:
            waiting = (state.flush or {}).get("pending_experiences")
            raise MirixApiError(
                f"MIRIX still had something to learn after {MAX_FLUSH_ROUNDS} flushes "
                f"(Q8 settles a session after {SESSION_FAILURES_TO_SETTLE} counted "
                f"failures, M5.1 an experience after {CURATION_ATTEMPTS_TO_SETTLE} "
                f"curation attempts): sessions with counted failures "
                f"{', '.join(sorted(retried)) or 'none'}; experiences waiting for the "
                f"curator at the last flush: {waiting}"
            )
        exported = await self._export(state, api)
        vectorless = exported["skills_without_vector"]
        if vectorless:
            # A skill whose embedding still failed after MIRIX's back-off and its
            # end-of-run re-embedding: invisible to every gated retrieval, and a later
            # re-embedding pass (the test-phase server's startup) would write to the
            # frozen memory. The outage cost learning: the attempt restarts.
            raise self._infra(
                "skills without a retrieval vector at the freeze: "
                + ", ".join(vectorless)
            )
        state.frozen = True
        state.frozen_snapshot_sha256 = canonical_sha256(exported)
        await self._persist()

    async def snapshot(self) -> dict[str, Any]:
        state, api = self._require()
        exported = await self._export(state, api)
        return {
            "method": self.name,
            **exported,
            # Derived by the harness from the learning runs the frozen hash covers:
            # outside the hash, so a change to the summary's code never makes a frozen
            # memory look changed.
            "experience_pool": experience_pool(exported["learning_runs"]),
            "counters": state.counters.to_dict(),
            "frozen": state.frozen,
            "frozen_snapshot_sha256": state.frozen_snapshot_sha256,
            # What learning went on despite; absent while
            # nothing was, so a memory without tolerance snapshots as before.
            **(
                {
                    "tolerated": {
                        "counts": dict(sorted(state.tolerated.items())),
                        "problems": [dict(p) for p in state.accepted_problems],
                    }
                }
                if state.tolerated or state.accepted_problems
                else {}
            ),
        }

    async def close(self) -> None:
        if self._api is not None:
            api, self._api = self._api, None
            await api.aclose()
        if self.server is not None:
            server, self.server = self.server, None
            await server.stop()

    # --- configuration --------------------------------------------------------------

    @property
    def poll_s(self) -> float:
        if self._poll_s is not None:
            return self._poll_s
        cell = self.cell
        return (
            float(cell.options.get("mirix_poll_s", DEFAULT_POLL_S))
            if cell
            else DEFAULT_POLL_S
        )

    @property
    def finalize_timeout_s(self) -> float:
        if self._finalize_timeout_s is not None:
            return self._finalize_timeout_s
        cell = self.cell
        default = DEFAULT_FINALIZE_TIMEOUT_S
        return (
            float(cell.options.get("mirix_finalize_timeout_s", default))
            if cell
            else default
        )

    @property
    def state(self) -> MirixState:
        """The method's state (still readable after :meth:`close`)."""
        if self._state is None:
            raise RuntimeError("setup() has not run")
        return self._state

    @property
    def api(self) -> MirixApi:
        return self._require()[1]

    # --- internals ------------------------------------------------------------------

    def _require(self) -> tuple[MirixState, MirixApi]:
        if self._state is None or self._api is None:
            raise RuntimeError("setup() has not run (or close() already did)")
        return self._state, self._api

    def _require_learning(self) -> tuple[MirixState, MirixApi]:
        state, api = self._require()
        if state.frozen:
            raise RuntimeError(
                "MIRIX's memory is frozen: the test phase only retrieves (H1.6, H10.4)"
            )
        return state, api

    @property
    def state_path(self) -> Path:
        assert self.cell is not None
        return self.cell.work_dir / STATE_FILE

    async def _start_server(self, cell: CellContext) -> str:
        base_url = cell.options.get("mirix_base_url")
        if base_url:
            return str(base_url)
        # A resumed learning phase restarts the attempt's MIRIX on its restored database.
        mode: ServerMode = (
            "fresh" if cell.learning and cell.resume is None else "resume"
        )
        factory = self._server_factory or default_server
        server = factory(cell, mode)
        await server.start()
        self.server = server
        return server.base_url

    async def _server_alive(self) -> None:
        if self.server is not None:
            await self.server.ensure_alive()

    @staticmethod
    def _infra(detail: str) -> LearningAttemptInvalid:
        return LearningAttemptInvalid(
            Failure(FailureKind.INFRA, MIRIX_LEARNING, "mirix", None, detail[:1000])
        )

    def _check_learning(self, runs: Sequence[Mapping[str, Any]]) -> None:
        """Raise (the attempt is invalid, H8) if an outage, or MIRIX's own failure, cost
        MIRIX any learning: HARNESS when MIRIX's own failure or a curation configuration
        skip is among the problems (a restart repeats it: a person must look), else
        INFRA. First log, once each, the exceptions curator tools met that MIRIX
        answered to the model as failed tool results (:func:`learning_warnings`, B31):
        warnings only."""
        for warning in learning_warnings(runs):
            if warning not in self._warned:
                self._warned.add(warning)
                log.warning("MIRIX curator tool exception: %s", warning)
        problems = learning_problem_kinds(runs)
        problems = self._tolerate(runs, problems)
        if not problems:
            return
        text = "; ".join(f"[{kind.value}] {detail}" for kind, detail in problems)
        if any(kind is FailureKind.HARNESS for kind, _ in problems):
            raise LearningAttemptInvalid(
                Failure(
                    FailureKind.HARNESS,
                    MIRIX_LEARNING,
                    "mirix",
                    None,
                    "MIRIX's learning failed on MIRIX's own error or its "
                    f"configuration: {text}"[:1000],
                )
            )
        raise self._infra("MIRIX's learning hit an infrastructure failure: " + text)

    def _tolerate(
        self,
        runs: Sequence[Mapping[str, Any]],
        problems: Sequence[tuple[FailureKind, str]],
    ) -> list[tuple[FailureKind, str]]:
        """The learning problems :meth:`_check_learning` still raises (a MIRIX learning
        problem stops the attempt unless it is tolerated). Problems accepted before are
        dropped (every idle check lists every run). A new refusal
        (:func:`refused_call_in`) that
        :func:`untolerated_refusal` names (a session MIRIX deferred, a protocol break)
        stops the attempt at once (HARNESS). Each learning run the check sees for the
        first time, oldest first, is one observation of MIRIX's refusal streak: its
        first refusal extends the streak, and the ``REFUSAL_STREAK_ITEMS``-th refused
        run in a row with the same one raises ``upstream_refusal_streak`` (HARNESS); a
        run without one ends it, and a check without a new run (MIRIX learns once a
        batch of sessions is pending) leaves it as it is. Refusals are accepted
        (``gateway_refusal``). The rest is accepted (``rolled_back``) only at the first
        check after a resume whose checkpoint was rolled back ``MIRIX_BATCH_ROLLBACKS``
        times in a row, and only when all of it is INFRA. Nothing is accepted by a check
        that raises."""
        state = self.state
        after_resume, self._after_resume = self._after_resume, False
        known = {
            str(p.get("text"))
            for p in state.accepted_problems
            if isinstance(p, Mapping)
        }
        refused: list[tuple[str, Failure]] = []
        rest: list[tuple[FailureKind, str]] = []
        for kind, text in problems:
            if text in known:
                continue
            failure = refused_call_in(text)
            if failure is not None:
                refused.append((text, failure))
            else:
                rest.append((kind, text))
        for text, failure in refused:
            why = untolerated_refusal(text, failure)
            if why is not None:
                raise LearningAttemptInvalid(
                    Failure(
                        FailureKind.HARNESS,
                        MIRIX_LEARNING,
                        MIRIX_ROLE,
                        failure.status,
                        f"{why} (rule harness-stop-learning): {text}"[:1000],
                    )
                )
        streaks = RefusalStreaks(state.refusal_streaks)
        seen = set(state.refusal_runs)
        new_runs = [r for r in runs if str(r.get("id")) not in seen]
        # MIRIX lists its runs newest first: oldest first, by creation time.
        new_runs = sorted(reversed(new_runs), key=lambda r: str(r.get("created_at")))
        stop = None
        for run in new_runs:
            state.refusal_runs.append(str(run.get("id")))
            first = next(
                (
                    failure
                    for _, text in learning_problem_kinds([run])
                    if (failure := refused_call_in(text)) is not None
                ),
                None,
            )
            stop = streaks.observe(MIRIX_ROLE, first)
            if stop is not None:
                break
        state.refusal_streaks = streaks.to_dict()
        if stop is not None:
            raise LearningAttemptInvalid(stop)
        if rest and not (
            after_resume
            and self._rollbacks >= MIRIX_BATCH_ROLLBACKS
            and all(kind is FailureKind.INFRA for kind, _ in rest)
        ):
            return rest
        for text, _ in refused:
            self._accept(TOLERATED_REFUSED, text)
        for _, text in rest:
            self._accept(TOLERATED_ROLLED_BACK, text)
        return []

    def _accept(self, reason: str, text: str) -> None:
        """Record one accepted learning problem in the state and queue its record."""
        state = self.state
        state.accepted_problems.append(
            {"reason": reason, "text": text, "rollbacks": self._rollbacks}
        )
        state.tolerated[reason] = state.tolerated.get(reason, 0) + 1
        state.tolerated[TOLERATED_TOTAL] = state.tolerated.get(TOLERATED_TOTAL, 0) + 1
        self._pending_records.append(
            {
                "event": "tolerated",
                "at": _now(),
                "reason": reason,
                "problem": text,
                "rollbacks": self._rollbacks,
            }
        )

    async def _write_tolerated(self) -> None:
        """Write the ``tolerated`` records the last idle check queued, then the state."""
        records, self._pending_records = self._pending_records, []
        for record in records:
            await self._record(LEARNING_RECORD_FILE, record)
        if records:
            await self._persist()

    async def _retrieval_row(self, task_uid: str, retrieval_id: Any) -> dict[str, Any]:
        """The retrieval's log row (M7.4). Without it nothing tells whether the retrieval
        lost its extraction or its vector, so the item cannot be scored (HARNESS)."""
        state, api = self._require()
        if not retrieval_id:
            raise MirixApiError(
                "MIRIX's retrieval returned no retrieval_id (is its retrieval log off, "
                "MIRIX_RETRIEVAL_LOG_ENABLED=false, or did the row fail to write?): "
                "whether it lost its topic extraction or query vector cannot be checked"
            )
        rows = await api.list_retrieval_logs(session_id=task_uid, user_id=state.user_id)
        match = [r for r in rows if r.get("id") == retrieval_id]
        if len(match) != 1:
            raise MirixApiError(
                f"MIRIX lists {len(match)} retrieval log rows with id {retrieval_id} for "
                f"session {task_uid}"
            )
        return match[0]

    async def _persist(self) -> None:
        state, _ = self._require()
        await asyncio.to_thread(_write_atomic, self.state_path, state.to_dict())

    async def _record(self, name: str, record: Mapping[str, Any]) -> None:
        assert self.cell is not None
        async with self._record_lock:
            await asyncio.to_thread(append_jsonl, self.cell.work_dir / name, record)

    async def _set_up_learning(self, cell: CellContext) -> None:
        assert self._api is not None
        api = self._api
        if await asyncio.to_thread(self.state_path.exists):
            raise RuntimeError(
                f"{self.state_path} exists: a learning attempt starts in a fresh work_dir (H8)"
            )
        template = setup_template()
        org, client, user = (
            template["organization"],
            template["client"],
            template["user"],
        )
        config = meta_agent_config(cell)
        agent = await api.initialize_meta_agent(config)
        if not agent.get("id"):
            raise MirixApiError(f"MIRIX created no meta agent: {str(agent)[:300]}")
        _check_agent_config(agent, config)
        created = await api.create_or_get_user(user["user_id"], user["name"])
        if created != user["user_id"]:
            raise MirixApiError(f"unexpected user from MIRIX: {str(created)[:300]}")
        self._state = MirixState(
            cell_id=cell.cell_id,
            attempt=cell.attempt,
            benchmark=cell.benchmark,
            model=cell.model,
            org_id=org["org_id"],
            client_id=client["client_id"],
            user_id=user["user_id"],
            meta_agent_id=str(agent["id"]),
            config=without_secrets(config),
        )
        exported = await self._export(self._state, api)
        if exported["skills"] or exported["learning_runs"]:
            raise RuntimeError(
                "this MIRIX already holds skills or learning runs: a learning attempt "
                "starts from empty memory (H8)"
            )
        await self._persist()

    async def _resume_learning(self, cell: CellContext) -> None:
        """A learning phase resumed from a checkpoint (contract item 6): ``run.py`` has
        restored the cell database and HOME to the checkpoint and restarted MIRIX on them.
        Bind the meta agent MIRIX has (never initialize a new one), refuse one that does
        not carry this cell's configuration (another gateway or run), load the method's
        state from the checkpoint, and require the rebuilt snapshot to hash to the one the
        checkpoint recorded. Writes are allowed from here on, as in any learning phase."""
        assert self._api is not None and cell.resume is not None
        api, resume = self._api, cell.resume
        try:
            state = MirixState.from_dict(dict(resume.state))
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointMismatch(
                f"mirix: checkpoint {resume.checkpoint_id} holds no method state "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if state.frozen:
            raise CheckpointMismatch(
                f"mirix: checkpoint {resume.checkpoint_id} holds a frozen memory; a "
                "learning checkpoint is taken before the freeze"
            )
        identity = (state.cell_id, state.attempt, state.benchmark, state.model)
        expected = (cell.cell_id, cell.attempt, cell.benchmark, cell.model)
        if identity != expected:
            raise CheckpointMismatch(
                f"mirix: checkpoint {resume.checkpoint_id} belongs to {identity}, not to "
                f"this cell attempt {expected}"
            )
        user = await api.get_user(state.user_id)
        if not isinstance(user, Mapping) or user.get("id") != state.user_id:
            raise CheckpointMismatch(
                f"mirix: the restored MIRIX has no user {state.user_id}"
            )
        agent = await api.bind_meta_agent(state.meta_agent_id)
        _check_agent_config(agent, meta_agent_config(cell))
        self._state = state
        # The batch this resume replays may be accepted despite its INFRA losses at the
        # first idle check, once its checkpoint was rolled back often enough.
        self._rollbacks = resume.rollbacks
        self._after_resume = True
        # What finalize's first flush compares with (the runs MIRIX listed while idle).
        self._runs = await api.list_learning_runs(
            user_id=state.user_id,
            limit=LEARNING_RUNS_LIMIT,
            meta_agent_id=state.meta_agent_id,
        )
        await self._persist()
        check_resumed_snapshot(self.name, await self.snapshot(), resume)

    async def _resume(self, cell: CellContext) -> None:
        assert self._api is not None
        api = self._api
        path = self.state_path
        if not await asyncio.to_thread(path.is_file):
            raise FrozenMemoryMismatch(
                f"no frozen MIRIX state at {path}: the test phase resumes this attempt's "
                "frozen memory and never starts empty (H8)"
            )
        state = MirixState.from_dict(await asyncio.to_thread(_read_json, path))
        if not state.frozen or state.frozen_snapshot_sha256 is None:
            raise FrozenMemoryMismatch(
                f"{path} was never frozen (finalize did not finish): the learning phase "
                "of this attempt did not end"
            )
        identity = (state.cell_id, state.attempt, state.benchmark, state.model)
        expected = (cell.cell_id, cell.attempt, cell.benchmark, cell.model)
        if identity != expected:
            raise FrozenMemoryMismatch(
                f"{path} belongs to {identity}, not to this cell attempt {expected}"
            )
        user = await api.get_user(state.user_id)
        if not isinstance(user, Mapping) or user.get("id") != state.user_id:
            raise MirixApiError(f"the resumed MIRIX has no user {state.user_id}")
        agent = await api.bind_meta_agent(state.meta_agent_id)
        _check_agent_config(agent, meta_agent_config(cell))
        self._state = state
        exported = await self._export(state, api)
        if canonical_sha256(exported) != state.frozen_snapshot_sha256:
            raise FrozenMemoryMismatch(
                "the resumed MIRIX memory differs from the memory frozen at the end of the "
                "learning phase (H8, H10.4)"
            )
        # The degraded retrievals earlier invocations left each test item with (only
        # the method's own record: the scorer's records are never read, H10.1).
        self._degraded_test = degraded_streaks(
            await asyncio.to_thread(
                _read_records, cell.work_dir / RETRIEVAL_RECORD_FILE
            )
        )

    async def _export(self, state: MirixState, api: MirixApi) -> dict[str, Any]:
        """Skills, their version history, every version's stored content and
        the learning runs (M9, M10); also the skills a learning run deleted (their
        history and versions: ``deleted_skills``) and the live skills without a
        retrieval vector (``skills_without_vector``, from each history's
        ``has_retrieval_embedding``). The frozen snapshot's hash covers all of it, so a
        vector a re-embedding pass adds in the test phase is a changed memory.

        Only what MIRIX returns: the experience pool's summary (``experience_pool``) is
        the harness's own pure function of the learning runs, which this hash covers,
        and
        :meth:`snapshot` adds it outside the hash (a change
        to that function between the freeze and a resumed test phase must not turn a
        valid attempt into an H10.4 mismatch)."""
        listing = await api.list_skills(state.user_id)
        memories = listing.get("memories") if isinstance(listing, Mapping) else None
        procedural = (
            memories.get("procedural") if isinstance(memories, Mapping) else None
        )
        if not isinstance(procedural, Mapping) or not isinstance(
            procedural.get("items"), list
        ):
            raise MirixApiError(f"unexpected skill listing: {str(listing)[:300]}")
        skills = sorted(
            (dict(item) for item in procedural["items"]), key=lambda s: str(s.get("id"))
        )
        total = procedural.get("total_count")
        if isinstance(total, int) and total > len(skills):
            raise MirixApiError(
                f"MIRIX lists {len(skills)} of {total} skills: the export would be incomplete"
            )
        gate = asyncio.Semaphore(EXPORT_CONCURRENCY)

        async def bounded(call: Awaitable[T]) -> T:
            async with gate:
                return await call

        runs = await api.list_learning_runs(
            user_id=state.user_id,
            limit=LEARNING_RUNS_LIMIT,
            meta_agent_id=state.meta_agent_id,
        )
        if len(runs) >= LEARNING_RUNS_LIMIT:
            raise MirixApiError(
                f"{len(runs)} learning runs: more than one page, the export would be incomplete"
            )
        live = [str(skill["id"]) for skill in skills]
        # A skill the curator deleted is gone from the listing; its history (and every
        # version's content) is still MIRIX's to give.
        deleted = sorted(curated_skill_ids(runs) - set(live))
        ids = live + deleted
        histories = await asyncio.gather(
            *(bounded(api.get_skill_history(i, state.user_id)) for i in ids)
        )
        history = dict(zip(ids, histories, strict=True))
        for skill_id in deleted:
            if not (history[skill_id] or {}).get("is_deleted"):
                raise MirixApiError(
                    f"skill {skill_id}, which a learning run changed, is neither listed "
                    "nor deleted: the export would be incomplete"
                )
        vectorless = []
        for skill_id in live:
            has = (history[skill_id] or {}).get("has_retrieval_embedding")
            if not isinstance(has, bool):
                raise MirixApiError(
                    f"the history of skill {skill_id} does not say whether it has its "
                    "retrieval vector (has_retrieval_embedding)"
                )
            if not has:
                vectorless.append(skill_id)
        wanted = []
        for skill_id in ids:
            entries = history[skill_id]
            listed = entries.get("versions") if isinstance(entries, Mapping) else None
            if not isinstance(listed, list):
                raise MirixApiError(
                    f"unexpected history of skill {skill_id}: {str(entries)[:300]}"
                )
            wanted += [
                (skill_id, int(v["version_no"]))
                for v in listed
                if isinstance(v, Mapping) and v.get("version_no") is not None
            ]
        stored = await asyncio.gather(
            *(bounded(api.get_skill_version(s, n, state.user_id)) for s, n in wanted)
        )
        versions: dict[str, dict[str, Any]] = {skill_id: {} for skill_id in ids}
        for (skill_id, number), answer in zip(wanted, stored, strict=True):
            content = answer.get("content") if isinstance(answer, Mapping) else None
            if not isinstance(content, Mapping):
                raise MirixApiError(
                    f"unexpected version {number} of skill {skill_id}: {str(answer)[:300]}"
                )
            versions[skill_id][str(number)] = dict(content)
        return {
            "meta_agent_id": state.meta_agent_id,
            "user_id": state.user_id,
            "skills": skills,
            "deleted_skills": deleted,
            "skills_without_vector": vectorless,
            "skill_history": history,
            "skill_versions": versions,
            "learning_runs": sorted(runs, key=lambda r: str(r.get("id"))),
        }


def _check_agent_config(agent: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Refuse a meta agent whose stored configuration is not the one this cell asked for.

    Catches a MIRIX that silently drops a field (``max_output_tokens`` in older MIRIX)
    and a resumed agent that points at another gateway run.
    """
    problems = []
    for section in ("llm_config", "embedding_config"):
        stored = agent.get(section)
        if not isinstance(stored, Mapping):
            problems.append(f"{section} missing")
            continue
        for key, value in config[section].items():
            if key == "api_key":
                continue  # MIRIX may not echo secrets
            if stored.get(key) != value:
                problems.append(
                    f"{section}.{key}={stored.get(key)!r}, expected {value!r}"
                )
    if problems:
        raise MirixApiError(
            "MIRIX's meta agent does not carry the cell's configuration: "
            + "; ".join(problems)
        )
