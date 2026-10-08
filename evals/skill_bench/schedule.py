"""Matrix scheduler (H8, H1.3).

Runs the cells of the benchmarks x models x methods matrix (``protocol.BENCHMARKS``,
``protocol.MODELS``, ``protocol.METHODS``; a filter or a study selects a batch) as
``run.py`` worker processes against one shared gateway::

    python -m evals.skill_bench.schedule --benchmark alfworld --concurrency 3 \\
        --env-file /path/to/.env                  # launch (or adopt) the gateway
    python -m evals.skill_bench.schedule --gateway-url http://127.0.0.1:18540   # attach
    python -m evals.skill_bench.schedule --dry-run  # show what would run
    python -m evals.skill_bench.schedule --auto-resume ...   # re-run interrupted cells

Rules:

* **File lock.** One scheduler per runs directory (``runs/_schedule/scheduler.lock``).
* **Concurrency 1-24** (``protocol.SCHEDULE_CONCURRENCY_RANGE``) cells at a time.
* **PID takeover.** A restarted scheduler adopts the workers that are still running (each
  live ``run.py`` holds its cell's ``cell.lock``; ``process.json`` names its pid) instead
  of starting them twice, and counts them against the concurrency. Workers run in their
  own sessions, so stopping the scheduler never stops them; a scheduler that launched the
  gateway leaves it running while workers are alive, and the next scheduler adopts it.
  Every live ``run.py`` holds the gateway's users lock (shared): no scheduler stops or
  sweeps a gateway that is held, whichever batch its workers belong to. A worker
  that finds another live harness holding its cell (``busy``) leaves it to that harness,
  which the scheduler adopts.
* **Re-runs.** Without ``--auto-resume`` nothing is re-run automatically: a cell that
  ended in ``test_incomplete``, ``learning_interrupted``, ``learning_failed``,
  ``infra_unavailable``, ``interrupted`` or ``crashed`` (or whose worker died) is reported
  and left alone, and ``--rerun-interrupted`` continues those cells (``run.py`` then
  resumes the learning phase from its checkpoint or re-runs the pending test items, H8).
  **``--auto-resume``** re-queues the cells whose status is in
  ``AUTO_RESUMABLE_STATUSES`` (and the cells whose worker died) by itself, each after a
  backoff that doubles with every consecutive resume that made no progress (1, 2, 4, 8,
  16, then 30 minutes; progress, a new checkpoint or more scored test items, resets it),
  up to ``--max-resumes-without-progress`` such resumes (2 for ``crashed``) and
  ``--max-resumes`` in all; past a cap the cell needs a person. An ``interrupted`` ending
  counts only as someone else's (no count without progress) when run.py recorded the
  signal it received (``status.json`` ``signal``) or the scheduler stopped it itself: a
  cancellation nobody sent is the harness's own and counts. A new attempt is
  progress only once it got past its setup. The counters live in
  ``runs/_schedule/resume_state.json`` and survive restarts; ``--reset-resume`` clears
  them once a person has looked. ``refused`` and ``violation`` cells are never launched:
  they need a person. Nor is a cell whose learning a HARNESS failure stopped (the
  harness's or MIRIX's own failure, which a resume repeats), with or without
  ``--auto-resume``: a person looks first, then passes ``--rerun-harness-failures
  [CELL ...]``, whose launches of that cell carry ``--new-attempt`` (a new attempt learns
  from scratch) until that new attempt exists: a launch that ended before it started
  (the gateway down at the start, a budget pause, a drain) leaves the release for the
  next one, and a new attempt that exists is resumed, never replaced. The
  decision covers the HARNESS stops that exist when that scheduler starts (the named
  cells, or all of them), each for exactly one new attempt: it is kept in
  ``resume_state.json`` (``harness_approved`` with the stopped ``harness_attempt``;
  ``harness_relaunched`` once the new attempt exists, by the ending's status marker), so
  a HARNESS stop that happens later, or the same one seen by a restarted scheduler,
  waits for a person again. A HARNESS release's launches never carry
  ``--note``: the stopped attempt only learned (``run.py`` asks for a note only to
  supersede an attempt that scored a test item or sent a test actor call).
  ``infra_unavailable`` and ``learning_interrupted`` go through the same backoff and
  caps as the other resumable endings; an attempt without learning
  checkpoints that infrastructure stops ends ``learning_interrupted`` too (the next
  launch restarts it as a new attempt), and the INFRA ``learning_failed`` an older
  ``run.py`` wrote for it is resumed the same way.
* **New attempts a person asks for.** The one-off
  ``--new-attempt-cells CELL [CELL ...] --new-attempt-note TEXT`` starts one new attempt
  of each named cell, whatever its latest attempt reached (a complete, tested cell
  too): its launches carry ``--new-attempt --note TEXT`` until that attempt exists
  (``run.py`` marks a tested attempt it replaces superseded and keeps it), later
  launches carry neither and resume it. The request is kept per note in
  ``resume_state.json`` (``rerun_requests``, key ``sha256(note)[:16]``): a scheduler
  restarted with the same command line requests nothing again, a deliberate second
  rerun takes a new note. Its estimate (what the replaced attempt cost, its
  ``costs.json``) is recorded in ``state.json`` ``new_attempts`` and the
  ``new_attempts_requested`` event; it spends from the run's ``--budget`` like any
  other launch.
  A cell that never ran needs no request; attempts that cannot be read defer the
  launch like a HARNESS release (never a blind ``--new-attempt``); blocked cells stay
  blocked.
* **The selection is recorded** (a deployment may run the baselines first and MIRIX's
  cells later, into the same runs directories). The command line's cells
  are recorded in ``_schedule/selection.json`` at the first launch (a runs directory
  that a scheduler without the record ran is read from the ``start`` events of its
  ``events.jsonl``), and a later launch that selects a cell no earlier launch selected
  is refused (exit 4) unless ``--widen-selection NOTE`` says why, once: the supervisor
  passes it one-off, and the record keeps each widening (when, the note, the cells and
  methods it added; event ``selection_widened``). A narrower selection always starts.
  ``tools/supervise.py`` refuses such a start itself (exit 2), so neither ``tools/phase16.sh``
  ``up``/``recover`` nor the systemd unit can widen a replicate's methods by accident.
* **MIRIX gateways.** A MIRIX agent calls the gateway it was set up with,
  and ``run.py`` refuses a resume on another. A MIRIX cell whose resumed attempt
  recorded another gateway (a cell first run by a standalone ``run.py``, a scheduler
  restarted with another ``--gateway-port``) gets that one: this scheduler attaches to
  it (another live harness owns it), adopts it (its owner died) or launches it on the
  recorded port, with the same budget, as an extra gateway (``state.json``
  ``gateways``, event ``gateway_extra``), closed or released at the end like its own.
  When it cannot (an attaching scheduler, a port taken by something else, another
  budget), the cell needs a person (``gateway_port_unavailable``, event
  ``launch_refused_gateway``). A restarted scheduler first adopts the gateways of the
  runs directory whose owner died while workers still use them.
* **Pause on low balance.** No cell is launched while the gateway is paused or its known
  balance is below the floor (an unknown balance counts as low; the gateway's own floor,
  0 included, unless ``--balance-floor``). Without ``--auto-resume`` the scheduler exits
  once nothing is running while paused. With it, it keeps polling through the pauses
  that lift by themselves (``WAITABLE_PAUSE_REASONS``: a low balance, the key's daily
  limit, an upstream 402, an unknown budget or balance) and launches again when they
  lift; on a spent or invalid budget (``STOP_PAUSE_REASONS``) and an unreachable gateway
  it launches nothing and exits (3) once nothing runs, with ``pause_reason`` in
  ``state.json``. A worker that found the gateway paused ran nothing: its cell goes back
  to the front of the queue and is launched once the balance allows.
* **Spend budget.** ``--budget USD`` launches the gateway with that budget: it pauses once
  the key's usage has grown by it since a budget was first set for this runs directory
  (``_gateway/budget.json``; see ``infra/gateway.py``), which pauses the scheduler as
  above. An attached or adopted gateway must already run with the same budget, and no
  gateway is launched without one in a runs directory that has a budget record.
* **Resource guard** (keep the host's memory healthy; ``infra/resources.py``). A cell
  is launched only while the available memory minus its estimated peak (and minus the
  headroom the running workers have not taken yet) stays at or above
  ``--min-free-memory-gb`` and the disk has ``--min-free-disk-gb`` free. A cell the
  memory gate holds never holds the queue: every lighter cell behind it starts
  whenever it fits (event
  ``resource_skip`` with the running count), and the held cell starts the first time it
  is admitted (it is tried first on every poll; the queue is finite, so it is never
  starved). The disk gate holds the whole queue. A WebShop cell waits for the WebShop
  server to answer ``ready`` (gate ``webshop``, read at most every
  :data:`WEBSHOP_READY_POLL_S`); the other benchmarks' cells go on meanwhile. A server
  that answers ``failed`` needs a person: the wait is recorded once more as the event
  ``webshop_server_failed``.
  ``--benchmark-concurrency BENCHMARK=N`` (e.g. ``alfworld=2``) caps one benchmark's
  cells (skipped over, not held back); by default only WebShop's are capped
  (:data:`DEFAULT_BENCHMARK_CAPS`: its cells share one server). The host load is logged
  every ``--resources-log-interval`` in ``_schedule/resources.jsonl`` and the report
  lists it next to each cell's sandbox timeouts, so a cap for another benchmark can be
  set from those numbers. After
  three consecutive samples with less than ``--memory-emergency-gb``
  available, the most recently launched worker gets ``SIGTERM``: it stops ``interrupted``
  and resumable, and ``--auto-resume`` brings it back once admission passes (that resume
  counts towards ``--max-resumes`` only). No further worker is stopped until the stopped
  one has exited and :data:`EMERGENCY_SETTLE_S` have passed (its checkpoint capture is
  still holding its memory), and the victim is the newest of the workers stopped the
  fewest times, so one relaunched cell does not take every stop. A worker
  that cannot be signalled (a stale adopted pid) passes the stop to the next one; a
  victim still running :data:`EMERGENCY_KILL_AFTER_S` after its ``SIGTERM``
  gets ``SIGKILL`` (its process group; its MIRIX server is cleaned up once it is gone)
  and no longer holds back the next stop. The host's resources are
  logged to ``runs/_schedule/resources.jsonl``.
* **Schedulers that share a host** (for example several replicate runs directories, each
  under its own scheduler and supervisor; ``host_registry.py``). Every scheduler keeps a
  record of its workers in the host directory (``$SKILL_BENCH_HOST_DIR``, else
  ``~/.cache/skill_bench/host``) and reads the others' under the host lock: the memory gate also subtracts the headroom the other
  schedulers' workers have not taken yet, ``--host-benchmark-concurrency BENCHMARK=N``
  caps a benchmark's cells over every scheduler of the host (by default WebShop at
  :data:`DEFAULT_HOST_BENCHMARK_CAPS`: one server with one global lock serves every
  replicate; a held cell is skipped like a per-benchmark cap, event ``host_cap_wait``,
  ``state.json`` ``host``; every scheduler publishes its caps and enforces the lowest a
  live one publishes, ``host_cap_mismatch`` when a peer's differs, and keeps a slot free
  for a peer the cap holds back while it runs none of the benchmark's cells, first come
  first served), and a memory emergency stops one worker host-wide (a
  scheduler whose peer stopped one less than :data:`EMERGENCY_SETTLE_S` ago, or whose
  peer's victim still runs, leaves it at that: ``memory_emergency_left_to_peer``). A
  scheduler publishes its record only once it has adopted the workers it found, so a
  restart never replaces its predecessor's orphaned record with an empty one. A host
  directory that cannot be used is recorded (``host_registry_unavailable``) and the
  scheduler goes on with its own view. ``--budget-start-from RUNS_DIR`` copies another
  runs directory's ``_gateway/budget.json`` before the first budgeted gateway here
  (:func:`share_budget_start`), so the replicates on one key count from one start and
  ``--budget`` is their total; each replicate needs its own ``--gateway-port`` (a port
  another runs directory's gateway holds is refused, never shared).

Before anything else, under the scheduler lock: attach to or adopt/launch the gateway,
then ``processes.sweep_owned(runs_dir)`` cleans the MIRIX servers of crashed cells (an
attaching scheduler leaves ``runs/_gateway`` alone: that gateway is not its to stop).
Workers never see a provider key: they only know the gateway URL. A merged read-only
view of several hosts' runs (``_merged.json``, ``tools/sync_runs.py``) is refused.

Records: ``runs/_schedule/state.json`` (rewritten every poll with the live queue, the
waiting cells and their wake-ups, the workers, the pause, the last resource sample,
the resume counters, the extra gateways and the ``--new-attempt-cells`` requests; its
``exit_code`` is null until the scheduler stops), ``runs/_schedule/events.jsonl``
(launches, adoptions, exits, resumes, pauses, resource waits and skips, emergency
stops, extra gateways, new attempts), ``runs/_schedule/resume_state.json`` and
``runs/_schedule/resources.jsonl``. Every path option is made absolute when it is read,
so a relative ``--runs-dir`` names the same directory for the scheduler, its workers and
their MIRIX servers. ``tools/supervise.py`` keeps this scheduler running.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeGuard

import httpx

from evals.skill_bench import host_registry, protocol, study
from evals.skill_bench.cell_records import (
    ATTEMPT_FROZEN,
    ATTEMPT_LEARNING,
    ATTEMPT_LEARNING_INTERRUPTED,
    ATTEMPT_TESTING,
    AUTO_RESUMABLE_STATUSES,
    BLOCKED_STATUSES,
    EXIT_CODES,
    MERGED_MARKER,
    RESUMABLE_STATUSES,
    STATUS_BUDGET_PAUSED,
    STATUS_BUSY,
    STATUS_COMPLETE,
    STATUS_CRASHED,
    STATUS_INTERRUPTED,
    STATUS_LEARNING_FAILED,
    STATUS_RUNNING,
    CellPaths,
    cell_is_live,
    cell_progress,
    now,
    read_json,
    read_jsonl,
    read_status,
    write_json_atomic,
)
from evals.skill_bench.infra import resources
from evals.skill_bench.infra.failures import FailureKind
from evals.skill_bench.infra.gateway import (
    BUDGET_FILE,
    BUDGET_SPENT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_UPSTREAM,
    LOCAL_TOKEN,
    STOP_PAUSE_REASONS,
    WAITABLE_PAUSE_REASONS,
    GatewayProcess,
    gateway_owner_lock,
    gateway_users_lock,
    launch_gateway,
)
from evals.skill_bench.infra.mirix_server import is_secret_env_name
from evals.skill_bench.infra.processes import (
    OwnedProcessError,
    OwnerLock,
    append_jsonl,
    cleanup_owned,
    lock_is_held,
    port_is_free,
    process_start_time,
    registered_process_groups,
    signal_group,
    sweep_owned,
)
from evals.skill_bench.run import (
    DEFAULT_FINALIZE_TIMEOUT_S,
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_TEST_CONCURRENCY,
    MIRIX_METHOD,
    SPLIT_MODES,
    CellSpec,
    same_gateway,
)

log = logging.getLogger("evals.skill_bench.schedule")

#: The benchmarks in launch order: ALFWorld, then WebShop. Every
#: benchmark of ``protocol.BENCHMARKS``, and only those (checked at import, like
#: METHOD_ORDER).
BENCHMARK_ORDER: tuple[str, ...] = ("alfworld", "webshop")
if set(BENCHMARK_ORDER) != set(protocol.BENCHMARKS) or len(BENCHMARK_ORDER) != len(
    protocol.BENCHMARKS
):  # a protocol benchmark nobody could schedule, or one the protocol no longer has
    raise RuntimeError(
        f"schedule.BENCHMARK_ORDER {BENCHMARK_ORDER} is not an order of "
        f"protocol.BENCHMARKS {protocol.BENCHMARKS}"
    )
#: How long a WebShop server readiness reading is reused before the server is asked again
#: (the gate below; every cell's preflight checks the server fully).
WEBSHOP_READY_POLL_S = 30.0
#: How a reading of a server that failed its own checks starts (it needs a person: the
#: gate holds WebShop cells and records ``webshop_server_failed`` once per reason).
WEBSHOP_FAILED_PREFIX = "webshop: server 'failed'"
#: Caps on concurrent cells that apply unless ``--benchmark-concurrency`` names the
#: benchmark (WebShop's cells share one server with one global lock: at most 4 cells,
#: 16 test episodes at the default test concurrency, use it at once).
DEFAULT_BENCHMARK_CAPS: Mapping[str, int] = MappingProxyType({"webshop": 4})
#: Caps on concurrent cells of one benchmark over every scheduler of the host (the host
#: registry, ``host_registry.py``) that apply unless ``--host-benchmark-concurrency`` names
#: the benchmark. Several replicate schedulers on one host share one WebShop server
#: with one global lock: three per-scheduler caps of 4 would admit 12 WebShop cells (48
#: test episodes at --test-concurrency 4); the host cap keeps it at 6 cells, 24 test
#: episodes, the load ``envs/webshop.py``'s request timeout was set for. A single
#: scheduler with its cap of 4 never meets it.
DEFAULT_HOST_BENCHMARK_CAPS: Mapping[str, int] = MappingProxyType({"webshop": 6})


async def webshop_server_ready() -> tuple[bool, str]:
    """Whether the WebShop server answers ``ready`` (``envs/webshop.py``), and why not."""
    from evals.skill_bench.envs import webshop

    client = webshop.ServerClient(
        webshop.default_server_url(), timeout_s=webshop.READY_TIMEOUT_S
    )
    try:
        code, payload = await client.ready()
    except webshop.WebShopServerError as exc:
        return False, f"webshop: server unreachable ({exc})"
    finally:
        await client.aclose()
    if code == 200 and payload.get("state") == "ready":
        return True, "ready"
    # A failed server reads WEBSHOP_FAILED_PREFIX (+ its problems).
    return False, (
        f"webshop: server {payload.get('state')!r}"
        + (
            f" ({'; '.join(map(str, payload.get('problems') or []))[:300]})"
            if payload.get("problems")
            else ""
        )
    )


#: Within a benchmark, the long learning cells start first; the no-memory cells (test
#: phase only) fill the tail. Every method of ``protocol.METHODS``, and only those: a
#: method the study removed (``study.REMOVED_METHODS``) is refused by name (``matrix``,
#: ``--method``), and its cells left on disk are never selected.
METHOD_ORDER: tuple[str, ...] = (
    "mirix",
    "reasoningbank",
    "awm",
    "none",
)
if set(METHOD_ORDER) != set(protocol.METHODS) or len(METHOD_ORDER) != len(
    protocol.METHODS
):  # a protocol method nobody could schedule, or one the protocol no longer has
    raise RuntimeError(
        f"schedule.METHOD_ORDER {METHOD_ORDER} is not an order of protocol.METHODS "
        f"{protocol.METHODS}"
    )
MODEL_ORDER: tuple[str, ...] = tuple(protocol.MODELS)  # luna, luna56
DEFAULT_CONCURRENCY = 3
DEFAULT_POLL_S = 10.0
#: How often the balance is refreshed upstream (``GET /balance``); in between, the
#: gateway's cached ``/health`` is read.
DEFAULT_BALANCE_POLL_S = 30.0
DEFAULT_BALANCE_FLOOR_USD = 10.0
WORKER_MODULE = "evals.skill_bench.run"

#: ``--auto-resume``: the delay before a cell's k-th consecutive
#: resume without progress is ``min(cap, base * 2**(k-1))``: 1, 2, 4, 8, 16, 30, 30 ... min.
DEFAULT_RESUME_BACKOFF_BASE_S = 60.0
DEFAULT_RESUME_BACKOFF_CAP_S = 1800.0
DEFAULT_MAX_RESUMES_WITHOUT_PROGRESS = 5
#: A ``crashed`` cell (an unexpected exception in run.py, likely to repeat) gets fewer.
CRASHED_MAX_RESUMES_WITHOUT_PROGRESS = 2
DEFAULT_MAX_RESUMES = 20
RESUME_HISTORY = 20
RESUME_STATE_FILE = "resume_state.json"
RESOURCES_LOG_FILE = "resources.jsonl"
DEFAULT_RESOURCES_LOG_INTERVAL_S = 60.0
#: ``docker stats`` (Postgres's and the WebShop server's memory) is read this often.
DOCKER_STATS_INTERVAL_S = 600.0
#: Consecutive samples below ``--memory-emergency-gb`` before a worker is stopped.
EMERGENCY_SAMPLES = 3
#: After a worker the emergency stop signalled has exited, how long before another may be
#: stopped: its memory takes a moment to be freed and read.
EMERGENCY_SETTLE_S = 60.0
#: A worker the emergency stop signalled that is still running this long after its
#: ``SIGTERM`` is hung: it gets ``SIGKILL`` (its process group) and no longer holds back
#: the next stop. run.py's ``SIGTERM`` waits at most 60 s for a
#: checkpoint being written (``run.CHECKPOINT_SHIELD_S``), then stops its MIRIX server.
EMERGENCY_KILL_AFTER_S = 120.0

# Scheduler outcomes per cell.
DONE = "done"
QUEUED = "queued"
LIVE = "live"
ATTENTION = "needs_attention"
BLOCKED = "blocked"
#: ``--auto-resume``: re-queued once its backoff has passed (``resume_state.json``).
WAITING = "waiting"

#: ``resume_state.json`` ``last_status`` of a cell whose worker died without a status
#: (``status.json`` still says ``running`` and no live process holds ``cell.lock``).
STATUS_WORKER_DIED = "worker_died"
#: Why a cell needs a person (``state.json`` ``attention_reasons``).
REASON_RESUME_CAP = "resume_cap"
REASON_RESUME_CAP_TOTAL = "resume_cap_total"
REASON_HARNESS_FAILURE = "harness_failure"
#: A released HARNESS stop whose attempts could not be read for
#: RELEASE_UNREADABLE_MAX_DEFERRALS launches in a row; the release is kept.
REASON_RELEASE_UNREADABLE = "harness_release_unreadable"
REASON_NOT_RERUN = "rerun_not_requested"
#: A MIRIX cell whose agent calls the gateway its attempt recorded, when this
#: scheduler cannot supply that gateway (it only attaches, the port is taken by
#: something else, the gateway does not start or runs with another budget).
REASON_GATEWAY_PORT = "gateway_port_unavailable"
#: A pending ``--new-attempt-cells`` request whose attempts could not be read for
#: RELEASE_UNREADABLE_MAX_DEFERRALS launches in a row; the request is kept.
REASON_RERUN_UNREADABLE = "new_attempt_unreadable"
#: The worker the resource guard stopped (its resume does not count as one without
#: progress).
STOPPED_MEMORY_EMERGENCY = "memory_emergency_stop"

#: Pause reasons of the scheduler's own (the gateway's are in ``infra/gateway.py``).
PAUSE_GATEWAY_UNREACHABLE = "gateway_unreachable"
PAUSE_BALANCE_UNKNOWN = "balance_unknown"
PAUSE_BALANCE_LOW = "balance_below_floor"
#: Pauses ``--auto-resume`` waits through: they lift by themselves (a top-up, the key's
#: daily limit reset, the next usage read). Every other pause (a spent or invalid budget,
#: an unreachable gateway) stops the scheduler once nothing runs.
AUTO_RESUME_WAITABLE_PAUSES = frozenset(WAITABLE_PAUSE_REASONS) | {
    PAUSE_BALANCE_UNKNOWN
}

#: Scheduler exit codes.
EXIT_ALL_COMPLETE = 0
EXIT_ATTENTION = 2
EXIT_PAUSED = 3
EXIT_REFUSED = 4


class ScheduleError(RuntimeError):
    """The scheduler cannot start (lock held, gateway unusable)."""


def iso(epoch: float) -> str:
    """UTC ISO 8601 of a ``time.time()`` value."""
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def epoch(value: Any) -> float | None:
    """``time.time()`` value of an ISO 8601 string (UTC when it names no zone), or None."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def _pick(values: Sequence[str] | None, order: Sequence[str], what: str) -> list[str]:
    if not values:
        return list(order)
    unknown = sorted(set(values) - set(order))
    if unknown:
        raise ValueError(f"unknown {what}: {unknown}; choose from {list(order)}")
    return [v for v in order if v in set(values)]


def matrix(
    benchmarks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    *,
    split_mode: str = "protocol",
    n_train: int | None = None,
    n_test: int | None = None,
) -> list[CellSpec]:
    """The H1.3 matrix (2 benchmarks x 2 models x 4 methods = 16 cells), filtered, in
    launch order.

    ``models`` takes keys (``luna``, ``luna56``) or model ids, each matched whole
    (``luna`` never selects ``luna56``). A method the study removed
    (``study.REMOVED_METHODS``) is refused naming the decision (``ValueError``), even
    beside study methods: a selection is never silently narrowed.
    """
    study.refuse_removed(methods)
    keys = None
    if models:
        by_id = {v: k for k, v in protocol.MODELS.items()}
        keys = [by_id.get(m, m) for m in models]
    return [
        CellSpec(benchmark, model, method, split_mode, n_train, n_test)
        for benchmark in _pick(benchmarks, BENCHMARK_ORDER, "benchmarks")
        for method in _pick(methods, METHOD_ORDER, "methods")
        for model in _pick(keys, MODEL_ORDER, "models")
    ]


def study_matrix(
    name: str,
    benchmarks: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> list[CellSpec]:
    """The cells of the named study (``study.STUDIES``), in launch order, narrowed by
    ``benchmarks``, ``models`` and ``methods`` (a name outside the study is refused,
    ``ValueError``). A smoke study's cells learn on the whole train split and test on
    the whole val split of their benchmark (``study.Study.sizes``)."""
    entry = study.STUDIES.get(name)
    if entry is None:
        raise ValueError(f"unknown study {name!r}; studies: {', '.join(study.STUDIES)}")
    picked_b, picked_m, picked_methods = study.narrowed(
        entry, benchmarks=benchmarks, models=models, methods=methods
    )
    cells = []
    for benchmark in _pick(picked_b, BENCHMARK_ORDER, "benchmarks"):
        n_train, n_test = entry.sizes(benchmark)
        for method in _pick(picked_methods, METHOD_ORDER, "methods"):
            for model in _pick(picked_m, MODEL_ORDER, "models"):
                cells.append(
                    CellSpec(
                        benchmark, model, method, entry.split_mode, n_train, n_test
                    )
                )
    return cells


def cells_from_args(args: argparse.Namespace) -> list[CellSpec]:
    """The cells a ``schedule.py`` (or ``tools/supervise.py``) command line selects:
    ``--study`` with its narrowing filters, else the matrix of ``--benchmark``,
    ``--model``, ``--method`` and ``--split-mode`` (default ``protocol``). A study sets
    its split mode and sizes itself: ``--split-mode``, ``--n-train`` or ``--n-test``
    beside it is refused (``ValueError``)."""
    if getattr(args, "study", None):
        given = [
            flag
            for flag, value in (
                ("--split-mode", args.split_mode),
                ("--n-train", args.n_train),
                ("--n-test", args.n_test),
            )
            if value is not None
        ]
        if given:
            raise ValueError(
                f"--study {args.study} sets the split mode and sizes itself: leave out "
                f"{', '.join(given)}"
            )
        return study_matrix(args.study, args.benchmark, args.model, args.method)
    return matrix(
        args.benchmark,
        args.model,
        args.method,
        split_mode=args.split_mode or "protocol",
        n_train=args.n_train,
        n_test=args.n_test,
    )


def named_cells_problem(
    flag: str, named: Sequence[str] | None, cells: Sequence[CellSpec]
) -> str | None:
    """Why ``flag``'s cell ids cannot be acted on (None: all are selected cells). A cell
    of a method the study removed is named as such, with the decision: it is outside
    every selection and stays on disk untouched."""
    selected = {c.cell_id for c in cells}
    removed = sorted(
        (method, cell_id)
        for cell_id in set(named or ()) - selected
        if study.is_removed(method := study.cell_id_method(cell_id))
    )
    if removed:
        return (
            f"{flag} names cells of a removed method {[c for _, c in removed]}: "
            + study.removed_message(str(removed[0][0]))
        )
    unknown = sorted(set(named or ()) - selected)
    if unknown:
        return f"{flag} names cells outside the selection: {unknown}"
    return None


#: ``run.py --note``: one line of at most this many characters.
NOTE_MAX_CHARS = 1000


def note_problem(note: str | None) -> str | None:
    """Why ``note`` cannot be ``run.py``'s ``--note`` (one non-blank line of at most
    :data:`NOTE_MAX_CHARS` characters), or None."""
    if note is None or not note.strip():
        return "the note is empty (say why the cell runs again)"
    if "\n" in note or "\r" in note:
        return "the note must be one line"
    if len(note) > NOTE_MAX_CHARS:
        return f"the note is longer than {NOTE_MAX_CHARS} characters"
    return None


def rerun_key(note: str) -> str:
    """The key of a ``--new-attempt-cells`` request: one per note."""
    return hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]


#: ``_schedule/selection.json``: the cells the launches of a runs directory selected
#: (module docstring).
SELECTION_FILE = "selection.json"
SELECTION_VERSION = 1


def _cells_methods(cell_ids: Iterable[str]) -> list[str]:
    """The methods the cell ids name, sorted."""
    return sorted({m for c in cell_ids if (m := study.cell_id_method(c)) is not None})


def recorded_selection(control: Path) -> dict[str, Any] | None:
    """The selection the earlier launches of a runs directory made (``control``: its
    ``_schedule``): ``selection.json``, else the cells the ``start`` events of its
    ``events.jsonl`` named (a scheduler from before the record; ``inferred_from``
    ``events``), else None (no scheduler started there). Raises ``ValueError`` when the
    record cannot be read as one (nobody can then tell what was selected). Blocking."""
    record = read_json(control / SELECTION_FILE)
    if record is not None:
        if not isinstance(record, dict) or not isinstance(record.get("cells"), list):
            raise ValueError(f"{control / SELECTION_FILE} is not a selection record")
        return record
    cells: set[str] = set()
    for line in read_jsonl(control / "events.jsonl"):
        if line.get("event") == "start" and isinstance(line.get("cells"), list):
            cells |= {str(c) for c in line["cells"]}
    if not cells:
        return None
    return {
        "version": SELECTION_VERSION,
        "cells": sorted(cells),
        "methods": _cells_methods(cells),
        "inferred_from": "events",
        "first": None,
        "widened": [],
    }


def selection_problem(
    control: Path, cells: Sequence[CellSpec], widen_note: str | None
) -> str | None:
    """Why a launch of ``cells`` must not start in the runs directory of ``control``: it
    selects cells no earlier launch selected, without ``--widen-selection``; or the
    record cannot be read. None: it may start. Blocking."""
    try:
        record = recorded_selection(control)
    except (OSError, ValueError) as exc:
        return f"the selection record cannot be read ({exc}): repair or remove it"
    if record is None:
        return None
    added = sorted({c.cell_id for c in cells} - set(record["cells"]))
    if not added or widen_note is not None:
        return None
    new_methods = sorted(set(_cells_methods(added)) - set(record.get("methods") or ()))
    return (
        f"this launch selects {len(added)} cell(s) no earlier launch of "
        f"{control.parent} selected ({', '.join(added[:8])}"
        + (", ..." if len(added) > 8 else "")
        + ")"
        + (
            f", of methods it never selected: {', '.join(new_methods)}"
            if new_methods
            else ""
        )
        + f"; earlier launches selected the methods {', '.join(record.get('methods') or ()) or 'none'}. "
        "Narrow the selection (tools/phase16.sh: P16_METHODS), or pass "
        "--widen-selection 'NOTE' once to add them on purpose (recorded in "
        f"{SELECTION_FILE})"
    )


def record_selection(
    control: Path, cells: Sequence[CellSpec], widen_note: str | None, *, pid: int
) -> dict[str, Any] | None:
    """Record this launch's selection: the first one, or a widening
    ``--widen-selection`` asked for (the note, when, the cells and methods it added).
    Returns the widening or first record written, None when nothing changed (a launch
    within the recorded cells). Raises ``ValueError`` for a selection
    :func:`selection_problem` refuses. Blocking."""
    problem = selection_problem(control, cells, widen_note)
    if problem is not None:
        raise ValueError(problem)
    record = recorded_selection(control)
    selected = sorted({c.cell_id for c in cells})
    at = now()
    if record is None:
        entry = {
            "at": at,
            "pid": pid,
            "cells": selected,
            "methods": _cells_methods(selected),
        }
        record = {
            "version": SELECTION_VERSION,
            "cells": selected,
            "methods": _cells_methods(selected),
            "inferred_from": None,
            "first": entry,
            "widened": [],
        }
        write_json_atomic(control / SELECTION_FILE, record)
        return {"kind": "first", **entry}
    added = sorted(set(selected) - set(record["cells"]))
    # A record read from the start events is written down once.
    inferred = not (control / SELECTION_FILE).is_file()
    if not added and not inferred:
        return None
    cells_now = sorted(set(record["cells"]) | set(selected))
    widening = None
    if added:
        widening = {
            "at": at,
            "pid": pid,
            "note": widen_note,
            "cells": added,
            "methods": sorted(
                set(_cells_methods(added)) - set(record.get("methods") or ())
            ),
        }
    record = {
        **record,
        "version": SELECTION_VERSION,
        "cells": cells_now,
        "methods": _cells_methods(cells_now),
        "widened": [*(record.get("widened") or []), *([widening] if widening else [])],
    }
    write_json_atomic(control / SELECTION_FILE, record)
    if widening is not None:
        return {"kind": "widened", **widening}
    return {"kind": "inferred", "at": at, "cells": record["cells"]}


def attempt_cost_usd(paths: CellPaths, attempt: int) -> float | None:
    """What attempt ``attempt`` cost (its ``costs.json`` ``totals.cost_total``), the
    estimate of a rerun that replaces it; None when unreadable. Blocking."""
    try:
        costs = read_json(paths.attempt(attempt).costs)
    except (OSError, ValueError):
        return None
    totals = costs.get("totals") if isinstance(costs, dict) else None
    value = totals.get("cost_total") if isinstance(totals, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleOptions:
    runs_dir: Path = protocol.RUNS_DIR
    concurrency: int = DEFAULT_CONCURRENCY
    gateway_url: str | None = None
    #: Where the gateway writes its ledgers (default ``runs_dir``, the gateway it launches).
    gateway_runs_dir: Path | None = None
    gateway_host: str = DEFAULT_HOST
    gateway_port: int = DEFAULT_PORT
    #: Upstream of a launched gateway (tests point it at the fake upstream).
    gateway_upstream: str = DEFAULT_UPSTREAM
    env_file: Path | None = None
    #: The floor below which nothing is launched (default: the gateway's own floor).
    balance_floor_usd: float | None = None
    #: The spend budget of a launched gateway (USD); an attached or adopted gateway must
    #: run with the same one. None: no budget.
    budget_usd: float | None = None
    #: Another runs directory whose budget start this one shares (replicates that
    #: share one key): before this runs directory's first budgeted gateway, its
    #: ``_gateway/budget.json`` is copied from there (:func:`share_budget_start`), so
    #: every gateway counts the key's usage from one start and ``--budget`` is the total
    #: of all of them.
    budget_start_from: Path | None = None
    test_concurrency: int = DEFAULT_TEST_CONCURRENCY
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S
    finalize_timeout_s: float = DEFAULT_FINALIZE_TIMEOUT_S
    max_learning_attempts: int = 1
    rerun_interrupted: bool = False
    #: With ``rerun_interrupted`` or ``auto_resume``, also launch (with ``--new-attempt``)
    #: a cell whose learning a HARNESS failure stopped (a person looked): the
    #: HARNESS stops that exist when this scheduler starts, once each.
    rerun_harness_failures: bool = False
    #: Only these cells' HARNESS stops (empty: every selected cell's).
    rerun_harness_cells: tuple[str, ...] = ()
    #: One-off: start one new attempt of each of these cells, whatever their
    #: latest attempt reached (a complete cell too), with ``--new-attempt --note
    #: <new_attempt_note>`` (run.py supersedes a tested attempt it replaces). Kept per
    #: note in ``resume_state.json``: the same note never reruns a cell twice.
    new_attempt_cells: tuple[str, ...] = ()
    #: The one-line reason ``run.py`` records (``--note``); required with
    #: ``new_attempt_cells``, ignored without them (the supervisor may withhold the cells).
    new_attempt_note: str | None = None
    #: Record the selection in ``_schedule/selection.json`` and refuse a wider one than
    #: every earlier launch's (the command line always does; tests that
    #: build a scheduler by hand may leave it off).
    guard_selection: bool = False
    #: One-off: why this launch selects cells no earlier launch selected (recorded).
    widen_selection: str | None = None
    #: Re-queue ``AUTO_RESUMABLE_STATUSES`` and dead workers by themselves, with backoff
    #: and caps, and wait through the pauses that lift by themselves.
    auto_resume: bool = False
    resume_backoff_base_s: float = DEFAULT_RESUME_BACKOFF_BASE_S
    resume_backoff_cap_s: float = DEFAULT_RESUME_BACKOFF_CAP_S
    max_resumes_without_progress: int = DEFAULT_MAX_RESUMES_WITHOUT_PROGRESS
    max_resumes: int = DEFAULT_MAX_RESUMES
    #: Clear these cells' resume counters at start (an empty tuple: every selected cell);
    #: None: keep them.
    reset_resume: tuple[str, ...] | None = None
    #: Resource guard (infra/resources.py); None: that gate is off.
    min_free_memory_gb: float | None = None
    min_free_disk_gb: float | None = None
    memory_emergency_gb: float | None = None
    #: Per-benchmark caps on concurrent cells, e.g. ``(("alfworld", 2),)``, on top of
    #: :data:`DEFAULT_BENCHMARK_CAPS` (WebShop's).
    benchmark_concurrency: tuple[tuple[str, int], ...] = ()
    #: Caps on concurrent cells of one benchmark over every scheduler of the host, on top
    #: of :data:`DEFAULT_HOST_BENCHMARK_CAPS` (WebShop's; ``host_registry.py``).
    host_benchmark_concurrency: tuple[tuple[str, int], ...] = ()
    #: The host directory the schedulers of this host coordinate in (None:
    #: ``host_registry.default_host_dir()``, ``$SKILL_BENCH_HOST_DIR`` or
    #: ``~/.cache/skill_bench/host``).
    host_dir: Path | None = None
    resources_log_interval_s: float = DEFAULT_RESOURCES_LOG_INTERVAL_S
    poll_s: float = DEFAULT_POLL_S
    balance_poll_s: float = DEFAULT_BALANCE_POLL_S
    python: str = sys.executable
    worker_module: str = WORKER_MODULE
    #: Extra ``run.py`` arguments for every worker (e.g. tests' non-protocol gateway).
    worker_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Every path absolute, once, here: workers and a launched
        # gateway run in the repository root, whatever the scheduler's working directory,
        # and each worker's MIRIX server in its attempt directory. absolute(), not
        # resolve(): a symlinked runs directory keeps the spelling the operator gave.
        object.__setattr__(self, "runs_dir", Path(self.runs_dir).absolute())
        for name in ("gateway_runs_dir", "env_file", "host_dir", "budget_start_from"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).absolute())
        low, high = protocol.SCHEDULE_CONCURRENCY_RANGE
        if not low <= self.concurrency <= high:
            raise ValueError(f"concurrency must be between {low} and {high}")
        if self.budget_usd is not None and not self.budget_usd > 0:
            raise ValueError("the budget must be positive")
        if self.budget_start_from is not None:
            if self.budget_usd is None:
                raise ValueError(
                    "--budget-start-from shares a budget start: it needs --budget (the "
                    "total of every runs directory that shares it)"
                )
            if os.path.normpath(self.budget_start_from) == os.path.normpath(
                self.runs_dir
            ):
                raise ValueError("--budget-start-from names this runs directory")
        if not 0 <= self.resume_backoff_base_s <= self.resume_backoff_cap_s:
            raise ValueError(
                "the resume backoff needs 0 <= --resume-backoff-base <= "
                "--resume-backoff-cap"
            )
        if self.max_resumes_without_progress < 0 or self.max_resumes < 0:
            raise ValueError("resume caps cannot be negative")
        for name in ("min_free_memory_gb", "min_free_disk_gb", "memory_emergency_gb"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if not self.resources_log_interval_s > 0:
            raise ValueError("resources_log_interval_s must be positive")
        for name, flag in (
            ("benchmark_concurrency", "--benchmark-concurrency"),
            ("host_benchmark_concurrency", "--host-benchmark-concurrency"),
        ):
            caps = tuple((str(b), int(n)) for b, n in getattr(self, name))
            for benchmark, cap in caps:
                if benchmark not in BENCHMARK_ORDER:
                    raise ValueError(
                        f"unknown benchmark {benchmark!r} in {flag}; "
                        f"choose from {list(BENCHMARK_ORDER)}"
                    )
                if cap < 1:
                    raise ValueError(
                        f"the {benchmark} cap of {flag} must be at least 1"
                    )
            if len({b for b, _ in caps}) != len(caps):
                raise ValueError(f"{flag} names a benchmark twice")
            object.__setattr__(self, name, caps)
        if self.reset_resume is not None:
            object.__setattr__(self, "reset_resume", tuple(self.reset_resume))
        object.__setattr__(self, "rerun_harness_cells", tuple(self.rerun_harness_cells))
        object.__setattr__(self, "new_attempt_cells", tuple(self.new_attempt_cells))
        if self.new_attempt_cells:
            problem = note_problem(self.new_attempt_note)
            if problem:
                raise ValueError(
                    f"--new-attempt-cells needs --new-attempt-note: {problem}"
                )
        if self.widen_selection is not None:
            problem = note_problem(self.widen_selection)
            if problem:
                raise ValueError(f"--widen-selection: {problem}")

    @property
    def ledger_dir(self) -> Path:
        return self.gateway_runs_dir or self.runs_dir

    @property
    def control(self) -> Path:
        return self.runs_dir / "_schedule"

    @property
    def benchmark_caps(self) -> dict[str, int]:
        """The caps in force: :data:`DEFAULT_BENCHMARK_CAPS`, then the options' own."""
        return {**DEFAULT_BENCHMARK_CAPS, **dict(self.benchmark_concurrency)}

    @property
    def host_benchmark_caps(self) -> dict[str, int]:
        """The host-wide caps in force: :data:`DEFAULT_HOST_BENCHMARK_CAPS`, then the
        options' own."""
        return {**DEFAULT_HOST_BENCHMARK_CAPS, **dict(self.host_benchmark_concurrency)}

    @property
    def host_directory(self) -> Path:
        """The host directory of ``host_registry.py`` (``host_dir``, else its default)."""
        return self.host_dir or host_registry.default_host_dir()

    def cell_dir(self, spec: CellSpec) -> Path:
        return self.runs_dir / spec.cell_id


# ---------------------------------------------------------------------------
# Gateway: attach, adopt or launch
# ---------------------------------------------------------------------------


def live_gateways(runs_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    """Registered gateways of ``runs_dir`` that still run as the registered process (same
    pid and start time), with their owner lock paths (blocking)."""
    registry = runs_dir / "_gateway" / "owned_processes.jsonl"
    if not registry.is_file():
        return []
    active: dict[int, dict[str, Any]] = {}
    # "\n" only: a record may hold U+2028 (ensure_ascii=False).
    for line in registry.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue  # a line that parses but is no record
        if row.get("event") == "started":
            active[int(row["pid"])] = row
        else:
            active.pop(int(row.get("pid", -1)), None)
    found = []
    for pid, row in active.items():
        value = row.get("owner_lock")
        if row.get("name") != "gateway" or not value:
            continue
        path = Path(str(value))
        path = path if path.is_absolute() else registry.parent / path
        started = process_start_time(pid)
        if started and started == " ".join(str(row.get("started") or "").split()):
            found.append((row, path))
    return found


def registered_gateway(runs_dir: Path, lock_path: Path) -> dict[str, Any] | None:
    """The live, registered gateway whose owner lock is ``lock_path`` (blocking), or None."""
    for row, path in live_gateways(runs_dir):
        if path.resolve() == lock_path.resolve():
            return row
    return None


def live_cells(runs_dir: Path) -> list[str]:
    """Cells of ``runs_dir`` whose run.py is alive (it holds ``cell.lock``; blocking)."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        child.name
        for child in runs_dir.iterdir()
        if (child / "cell.lock").is_file() and lock_is_held(child / "cell.lock")
    )


def loopback_port(url: str | None) -> int | None:
    """The port of a loopback gateway URL, None for another URL or an unreadable one."""
    if not url:
        return None
    try:
        parsed = httpx.URL(url)
    except Exception:
        return None
    if parsed.port is None or parsed.host.lower() not in _LOOPBACK_HOSTS:
        return None
    return parsed.port


def live_cells_recording(runs_dir: Path, port: int) -> list[str]:
    """The live cells of ``runs_dir`` (:func:`live_cells`) whose resumed MIRIX attempt
    recorded the loopback gateway on ``port`` (:func:`mirix_gateway_url`). Blocking."""
    return [
        name
        for name in live_cells(runs_dir)
        if loopback_port(mirix_gateway_url(CellPaths(runs_dir / name))) == port
    ]


@dataclass
class SchedulerGateway:
    """The gateway the scheduler's workers use: ``attach``, ``launched`` or ``adopted``."""

    base_url: str
    mode: str
    runs_dir: Path
    process: GatewayProcess | None = None
    owner: OwnerLock | None = None
    pid: int | None = None
    #: The lock the gateway's live users (``run.py`` workers) hold shared.
    users_lock: Path | None = None
    #: Why an orphan this scheduler adopted at its start does not serve it as asked (not
    #: healthy, or another budget than ``--budget``: :func:`_adoption_problem`). It is
    #: adopted all the same (a live worker uses it) and the problem is recorded
    #: (``state.json`` ``gateways``, event ``gateway_extra``) for the supervisor to alert.
    problem: str | None = None
    #: The port of an extra gateway (a MIRIX cell's recorded one); None for the
    #: scheduler's own.
    extra_port: int | None = None
    #: What ``--budget-start-from`` did before this gateway was launched or adopted
    #: (:func:`share_budget_start`; event ``budget_start_shared``), None without it.
    budget_share: dict[str, Any] | None = None

    def in_use(self) -> bool:
        """True while any live ``run.py`` uses this gateway (its users lock is held), else
        (a worker between its start and its users lock) while a cell of this runs
        directory is live: for the scheduler's own gateway any live cell, for an extra
        gateway only a live cell whose attempt recorded its port (a live cell on another
        gateway never keeps it up). Blocking."""
        if self.users_lock is not None and lock_is_held(self.users_lock):
            return True
        if self.extra_port is not None:
            return bool(live_cells_recording(self.runs_dir, self.extra_port))
        return bool(live_cells(self.runs_dir))

    async def close(self) -> None:
        """Stop a gateway the scheduler owns, unless live workers still use it.

        The scheduler's own workers are gone when it calls this, but another batch's
        workers (another scheduler, a ``run.py`` attached by URL) may still use the
        gateway: then it is only released, like :meth:`release`, and stays up.
        """
        if self.mode == "attach":
            return
        if await asyncio.to_thread(self.in_use):
            self.release()
            return
        if self.process is not None:
            await self.process.stop()
        elif self.owner is not None:
            # Adopted: release its owner lock, then the orphan cleanup stops it (it checks
            # the process identity and never touches a gateway with live users).
            self.owner.release()
            registry = self.runs_dir / "_gateway" / "owned_processes.jsonl"
            await asyncio.to_thread(cleanup_owned, registry)

    def release(self) -> None:
        """Leave the gateway running for live workers; the next scheduler adopts it."""
        if self.process is not None and self.process.owner is not None:
            self.process.owner.release()
        if self.owner is not None:
            self.owner.release()


async def _healthy(url: str) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(url + "/health")
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if response.status_code == 200 and isinstance(data, dict) else None


def _check_budget(
    options: ScheduleOptions, health: Mapping[str, Any], where: str
) -> None:
    """Refuse a gateway this scheduler did not launch whose spend budget is not the one
    asked for: ``--budget`` would silently not apply."""
    if options.budget_usd is None:
        return
    running = health.get("budget_usd")
    if running != options.budget_usd:
        raise ScheduleError(
            f"the gateway at {where} runs with budget {running!r}, not --budget "
            f"{options.budget_usd:g}: stop it (once its cells are done) and launch the "
            "scheduler again, or leave out --budget to use that gateway as it is"
        )


async def ensure_gateway(options: ScheduleOptions) -> SchedulerGateway:
    """Attach to ``gateway_url``; else adopt this runs directory's orphaned gateway on the
    configured port (its scheduler died, its workers may still use it); else launch one."""
    if options.gateway_url:
        health = await _healthy(options.gateway_url.rstrip("/"))
        if health is None:
            raise ScheduleError(f"gateway {options.gateway_url} is not healthy")
        _check_budget(options, health, options.gateway_url)
        return SchedulerGateway(
            options.gateway_url.rstrip("/"), "attach", options.ledger_dir
        )
    runs_dir = options.runs_dir
    host, port = options.gateway_host, options.gateway_port
    lock_path = gateway_owner_lock(runs_dir, host, port)
    url = f"http://{host}:{port}"
    if await asyncio.to_thread(lock_is_held, lock_path):
        raise ScheduleError(
            f"the gateway on {host}:{port} belongs to another live harness; pass "
            f"--gateway-url {url} to share it"
        )
    if not await asyncio.to_thread(port_is_free, port, host):
        row = await asyncio.to_thread(registered_gateway, runs_dir, lock_path)
        health = await _healthy(url) if row is not None else None
        if row is None or health is None:
            raise ScheduleError(
                f"port {port} is in use by something that is not this runs directory's "
                "orphaned gateway"
            )
        _check_budget(options, health, url)
        share = await asyncio.to_thread(share_budget_start, options)
        try:
            owner = await asyncio.to_thread(OwnerLock.acquire, lock_path)
        except OwnedProcessError as exc:
            raise ScheduleError(str(exc)) from exc
        return SchedulerGateway(
            url,
            "adopted",
            runs_dir,
            owner=owner,
            pid=int(row["pid"]),
            users_lock=gateway_users_lock(runs_dir, host, port),
            budget_share=share,
        )
    # launch_gateway kills every orphaned gateway of this runs directory. One that live
    # workers still use (its scheduler died; it listens on another port) must not be.
    orphans = [
        row
        for row, path in await asyncio.to_thread(live_gateways, runs_dir)
        if path.resolve() != lock_path.resolve()
        and not await asyncio.to_thread(lock_is_held, path)
    ]
    if orphans:
        busy = await asyncio.to_thread(live_cells, runs_dir)
        if busy:
            raise ScheduleError(
                f"an orphaned gateway (pid {orphans[0]['pid']}, owner lock "
                f"{orphans[0].get('owner_lock')}) may serve the live cells {busy}; "
                "restart the scheduler on that gateway's port to adopt it"
            )
    # Before the budget kwargs: a record copied from --budget-start-from is this runs
    # directory's from now on (and needs --budget, which the options checked).
    share = await asyncio.to_thread(share_budget_start, options)
    kwargs = await asyncio.to_thread(_gateway_budget_kwargs, options)
    process = await launch_gateway(
        port=port,
        host=host,
        env_file=options.env_file,
        runs_dir=runs_dir,
        upstream=options.gateway_upstream,
        **kwargs,
    )
    return SchedulerGateway(
        process.base_url,
        "launched",
        runs_dir,
        process=process,
        pid=process.process.pid,
        users_lock=gateway_users_lock(runs_dir, host, port),
        budget_share=share,
    )


#: An owner lock's file name (``infra/gateway.gateway_owner_lock``):
#: ``gateway-<host slug>-<port>.lock``.
_GATEWAY_LOCK_NAME = re.compile(
    r"^gateway-(?P<host>[A-Za-z0-9.-]+)-(?P<port>\d+)\.lock$"
)
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def mirix_gateway_url(paths: CellPaths) -> str | None:
    """The gateway a MIRIX cell's next launch must give its worker, or None: the
    latest attempt's ``attempt.json`` ``gateway_base_url`` when that attempt is resumed
    (``learning`` / ``learning_interrupted`` with a ``checkpoints/`` directory, or
    ``frozen`` / ``testing``). A MIRIX agent keeps the gateway URL it was set up with
    (its ``llm_config`` and ``embedding_config`` endpoints), and ``run.py`` refuses a
    resume on another gateway. None for no attempt, one that ends in a new attempt (no
    checkpoints, complete, invalid, violation) and anything unreadable. The same rule as
    ``run.CellRunner._resume_gateway_port`` (duplicated so ``run.py`` stays untouched; a
    test pins the two together). Blocking."""
    try:
        numbers = paths.attempt_numbers()
    except OSError:
        return None
    if not numbers:
        return None
    attempt = paths.attempt(numbers[-1])
    try:
        state = read_json(attempt.state)
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    current = state.get("state")
    if current in (ATTEMPT_LEARNING, ATTEMPT_LEARNING_INTERRUPTED):
        if not attempt.checkpoints.is_dir():
            return None  # restarts as a new attempt: any gateway
    elif current not in (ATTEMPT_FROZEN, ATTEMPT_TESTING):
        return None
    recorded = state.get("gateway_base_url")
    return str(recorded) if isinstance(recorded, str) and recorded else None


#: The fields ``share_budget_start`` adds to a copied budget record (the gateway keeps
#: every field of the record it reads).
BUDGET_SHARED_FROM = "start_shared_from"
BUDGET_SHARED_AT = "start_shared_at"


def _budget_start_of(path: Path) -> tuple[float, Any]:
    """``(start_usage_usd, started_at)`` of a key-basis budget record (blocking);
    ``ValueError`` when it has no numeric start."""
    record = json.loads(path.read_text(encoding="utf-8"))
    start = record.get("start_usage_usd") if isinstance(record, dict) else None
    if isinstance(start, bool) or not isinstance(start, (int, float)):
        raise ValueError(f"{path} has no start_usage_usd")
    return float(start), record.get("started_at")


def share_budget_start(options: ScheduleOptions) -> dict[str, Any] | None:
    """``--budget-start-from DIR`` (replicate runs directories that share one key): make
    this runs directory's budget count from DIR's start. Blocking.

    A key-basis budget counts the key's whole usage growth since its start, so every
    gateway on the key sees the others' spend: with one start, each one pauses when the
    key has spent ``--budget`` since then, which makes ``--budget`` the total of every
    runs directory that shares it (the same as copying the record by hand). Before this
    runs directory has a record, DIR's is copied here
    (with :data:`BUDGET_SHARED_FROM` and :data:`BUDGET_SHARED_AT` added) and
    ``{"shared": "copied", ...}`` is returned; DIR without a record yet is a
    :class:`ScheduleError` (exit 4: the supervisor retries until DIR's first gateway has
    written it). When this runs directory has a record already, nothing is changed: a
    start that differs from DIR's is returned as ``{"shared": "differs", ...}`` (recorded,
    not refused), an equal one as ``{"shared": "same", ...}``. None without the option."""
    source_dir = options.budget_start_from
    if source_dir is None:
        return None
    source = source_dir / "_gateway" / BUDGET_FILE
    own = options.runs_dir / "_gateway" / BUDGET_FILE
    info: dict[str, Any] = {"from": str(source), "file": str(own)}
    if own.is_file():
        try:
            mine = _budget_start_of(own)
            theirs = _budget_start_of(source) if source.is_file() else None
        except (OSError, ValueError) as exc:
            return {**info, "shared": "unreadable", "error": f"{exc}"[:300]}
        if theirs is None:
            return {**info, "shared": "source_missing", "start": mine[0]}
        return {
            **info,
            "shared": "same" if mine == theirs else "differs",
            "start": mine[0],
            "started_at": mine[1],
            "source_start": theirs[0],
            "source_started_at": theirs[1],
        }
    if not source.is_file():
        raise ScheduleError(
            f"--budget-start-from: {source} does not exist yet; start the scheduler of "
            f"{source_dir} first (its gateway writes the record at its first balance "
            "check), then this one counts from the same start (the supervisor retries "
            "a refused scheduler by itself)"
        )
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
        start = _budget_start_of(source)
    except (OSError, ValueError) as exc:
        raise ScheduleError(
            f"--budget-start-from: {source} cannot be read ({exc}); fix it, or leave "
            "out the option to start a budget of this runs directory's own"
        ) from exc
    record[BUDGET_SHARED_FROM] = str(source)
    record[BUDGET_SHARED_AT] = now()
    write_json_atomic(own, record)
    return {**info, "shared": "copied", "start": start[0], "started_at": start[1]}


def _gateway_budget_kwargs(options: ScheduleOptions) -> dict[str, Any]:
    """The budget and floor a gateway this scheduler launches runs with; refuses a
    launch without ``--budget`` in a runs directory with a budget record. Blocking."""
    record = options.runs_dir / "_gateway" / BUDGET_FILE
    if options.budget_usd is None and record.is_file():
        # A budget was set for this runs directory: a gateway launched without one would
        # spend past it unnoticed (a step whose command lost --budget).
        raise ScheduleError(
            f"{record} sets a spend budget for this runs directory: pass --budget (the "
            "directory's total spend budget), or remove the file to run without one"
        )
    kwargs: dict[str, Any] = {}
    if options.balance_floor_usd is not None:
        kwargs["balance_floor_usd"] = options.balance_floor_usd
    if options.budget_usd is not None:
        kwargs["budget_usd"] = options.budget_usd
    return kwargs


async def gateway_on_port(options: ScheduleOptions, port: int) -> SchedulerGateway:
    """A gateway of this runs directory on ``port`` (another than the scheduler's own) for
    a MIRIX cell whose agent recorded it: attach to it when another live
    harness owns it, adopt it when it is this runs directory's registered live gateway
    whose owner died, else launch one there (with the scheduler's budget and floor).
    Each is checked healthy and running with ``--budget``. :class:`ScheduleError` when
    none of these is possible."""
    runs_dir = options.runs_dir
    host = options.gateway_host
    url = f"http://{host}:{port}"
    lock_path = gateway_owner_lock(runs_dir, host, port)
    users_lock = gateway_users_lock(runs_dir, host, port)
    if await asyncio.to_thread(lock_is_held, lock_path):
        # Another live harness (a scheduler of another batch, a run.py in launch mode)
        # owns it: share it, as --gateway-url would.
        health = await _healthy(url)
        if health is None:
            raise ScheduleError(
                f"the gateway on {host}:{port} belongs to another live harness and is "
                "not healthy"
            )
        _check_budget(options, health, url)
        return SchedulerGateway(
            url, "attach", runs_dir, users_lock=users_lock, extra_port=port
        )
    if not await asyncio.to_thread(port_is_free, port, host):
        row = await asyncio.to_thread(registered_gateway, runs_dir, lock_path)
        health = await _healthy(url) if row is not None else None
        if row is None or health is None:
            raise ScheduleError(
                f"port {port} is in use by something that is not this runs directory's "
                "gateway"
            )
        _check_budget(options, health, url)
        try:
            owner = await asyncio.to_thread(OwnerLock.acquire, lock_path)
        except OwnedProcessError as exc:
            raise ScheduleError(str(exc)) from exc
        return SchedulerGateway(
            url,
            "adopted",
            runs_dir,
            owner=owner,
            pid=int(row["pid"]),
            users_lock=users_lock,
            extra_port=port,
        )
    kwargs = await asyncio.to_thread(_gateway_budget_kwargs, options)
    try:
        process = await launch_gateway(
            port=port,
            host=host,
            env_file=options.env_file,
            runs_dir=runs_dir,
            upstream=options.gateway_upstream,
            **kwargs,
        )
    except Exception as exc:  # not started: the cell goes to a person, the matrix on
        raise ScheduleError(
            f"cannot launch a gateway on {host}:{port}: {type(exc).__name__}: {exc}"
        ) from exc
    return SchedulerGateway(
        process.base_url,
        "launched",
        runs_dir,
        process=process,
        pid=process.process.pid,
        users_lock=users_lock,
        extra_port=port,
    )


def _orphans_in_use(
    options: ScheduleOptions,
) -> list[tuple[dict[str, Any], Path, str, int]]:
    """This runs directory's live registered gateways on another port than the
    scheduler's whose owner died (owner lock free) while ``run.py`` workers still use
    them (users lock held): ``(row, owner lock, host, port)``. Blocking."""
    runs_dir = options.runs_dir
    main = gateway_owner_lock(runs_dir, options.gateway_host, options.gateway_port)
    found = []
    for row, path in live_gateways(runs_dir):
        if path.resolve() == main.resolve() or lock_is_held(path):
            continue
        match = _GATEWAY_LOCK_NAME.match(path.name)
        if match is None:
            continue
        host, port = match["host"], int(match["port"])
        if lock_is_held(gateway_users_lock(runs_dir, host, port)):
            found.append((row, path, host, port))
    return found


async def _adoption_problem(options: ScheduleOptions, url: str) -> str | None:
    """Why the gateway at ``url`` does not serve this scheduler as asked: it does not
    answer ``/health``, or it runs with another budget than ``--budget`` (the checks of
    every other adoption: :func:`ensure_gateway`, :func:`gateway_on_port`). None when it
    does."""
    health = await _healthy(url)
    if health is None:
        return f"the gateway at {url} is not healthy (no answer from /health)"
    try:
        _check_budget(options, health, url)
    except ScheduleError as exc:
        return str(exc)
    return None


async def adopt_live_gateways(options: ScheduleOptions) -> dict[int, SchedulerGateway]:
    """Adopt, as extra gateways of this scheduler, every gateway of the runs
    directory on another port whose owner died while workers still use it (a MIRIX
    cell's recorded gateway a previous scheduler launched): its owner lock is taken, so
    :func:`ensure_gateway` no longer sees an orphan that may serve live cells, and the
    scheduler closes (or releases) it at its end like its own. An attaching scheduler
    adopts nothing. Each is checked like every other adoption (healthy, running
    with ``--budget``: :func:`_adoption_problem`); one that fails is adopted all the
    same, since a live worker uses it (left an orphan it would refuse the whole matrix,
    stopped it would kill the worker), with its ``problem`` set: the scheduler records
    it and the supervisor alerts it, and the cells that recorded it go to a person at
    their next launch (:meth:`Scheduler._gateway_at` checks again). Returns ``{port:
    gateway}``."""
    if options.gateway_url:
        return {}
    adopted: dict[int, SchedulerGateway] = {}
    for row, path, host, port in await asyncio.to_thread(_orphans_in_use, options):
        try:
            owner = await asyncio.to_thread(OwnerLock.acquire, path)
        except OwnedProcessError:
            continue  # another harness took it meanwhile
        url = f"http://{host}:{port}"
        adopted[port] = SchedulerGateway(
            url,
            "adopted",
            options.runs_dir,
            owner=owner,
            pid=int(row["pid"]),
            users_lock=gateway_users_lock(options.runs_dir, host, port),
            problem=await _adoption_problem(options, url),
            extra_port=port,
        )
    return adopted


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def worker_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The workers' environment: no provider key or other credential (they only reach the
    gateway), loopback without proxies, unbuffered output."""
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if not is_secret_env_name(k)}
    loopback = "127.0.0.1,localhost,::1"
    env["NO_PROXY"] = ",".join(filter(None, [env.get("NO_PROXY", ""), loopback]))
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONUNBUFFERED"] = "1"
    return env


def worker_argv(
    spec: CellSpec,
    options: ScheduleOptions,
    gateway_url: str,
    ledger_dir: Path,
    *,
    new_attempt: bool = False,
    note: str | None = None,
) -> list[str]:
    """The ``run.py`` command line of one cell (``new_attempt``: the launch a person asked
    for with ``--rerun-harness-failures`` or ``--new-attempt-cells``, which learns again
    from scratch; ``note``: the reason ``run.py`` records for a ``--new-attempt-cells``
    rerun; a HARNESS release carries none)."""
    argv = [
        options.python,
        "-m",
        options.worker_module,
        "--benchmark",
        spec.benchmark,
        "--model",
        spec.model_key,
        "--method",
        spec.method,
        "--split-mode",
        spec.split_mode,
    ]
    if spec.smoke:
        argv += ["--n-train", str(spec.n_train), "--n-test", str(spec.n_test)]
    argv += [
        "--out",
        str(options.cell_dir(spec)),
        "--gateway-url",
        gateway_url,
        "--gateway-runs-dir",
        str(ledger_dir),
        "--test-concurrency",
        str(options.test_concurrency),
        "--idle-timeout",
        f"{options.idle_timeout_s:g}",
        "--finalize-timeout",
        f"{options.finalize_timeout_s:g}",
        "--max-learning-attempts",
        str(options.max_learning_attempts),
        *options.worker_args,
    ]
    if new_attempt:
        argv.append("--new-attempt")
        if note is not None:
            # One token, so argparse never reads a note that starts with '-' (one the
            # scheduler took as --new-attempt-note=-...) as an option of its own.
            argv.append(f"--note={note}")
    return argv


@dataclass
class Worker:
    """A ``run.py`` process of one cell: launched here, or adopted from a dead scheduler."""

    spec: CellSpec
    paths: CellPaths
    pid: int
    popen: subprocess.Popen[bytes] | None = None
    started_at: float = field(default_factory=time.monotonic)
    #: The extra gateway's port this scheduler gave the worker; None for its
    #: own gateway or an adopted worker.
    gateway_port: int | None = None

    @property
    def adopted(self) -> bool:
        return self.popen is None

    def poll(self) -> int | None:
        """The exit code once the worker has exited, else None (blocking; tiny)."""
        if self.popen is not None:
            return self.popen.poll()
        if cell_is_live(self.paths):  # an adopted worker holds its cell lock until exit
            return None
        status = read_status(self.paths) or {}
        code = status.get("exit_code")
        return int(code) if isinstance(code, int) else EXIT_CODES[STATUS_CRASHED]

    def process_group(self) -> int:
        """The worker's process group (blocking): its own pid for a launched worker (its
        own session), the group an adopted one runs in."""
        if self.popen is not None:
            return self.pid
        try:
            return os.getpgid(self.pid)
        except (ProcessLookupError, PermissionError, OSError):
            return self.pid

    def refresh(self) -> None:
        """Re-read an adopted worker's ``process.json`` (its pid and start; blocking).
        A launched worker knows both."""
        if self.popen is None:
            self._adopted_still_runs()

    def _adopted_still_runs(self, switch: bool = True) -> bool:
        """An adopted worker's pid is still that worker (``process.json``'s start time
        matches), never a stranger that reused it. Blocking. An adopted worker is the
        cell lock's holder (:meth:`poll`): a worker adopted before its run.py wrote
        ``process.json`` (pid -1) takes the pid it names now, or it could never be
        stopped. ``switch`` False: only the pid it has (a kill of a victim that was
        sent SIGTERM must never land on a run.py that took the cell after it)."""
        try:
            info = (
                read_json(self.paths.process) if self.paths.process.exists() else None
            )
        except (OSError, ValueError):
            info = None
        pid = info.get("pid") if isinstance(info, dict) else None
        started = info.get("started") if isinstance(info, dict) else None
        if not (_is_count(pid) and pid > 0 and started):
            return False
        live = process_start_time(pid) == " ".join(str(started).split())
        if live and pid != self.pid and switch:
            # Adopted before its run.py rewrote process.json: the pid then named was the
            # previous run's, dead. process.json is written by the
            # lock holder alone, so the live process it names now is this worker; its
            # start is that run's too (the memory emergency stops the newest first).
            self.pid = pid
            launched = epoch(info.get("at")) if isinstance(info, dict) else None
            if launched is not None:
                self.started_at = time.monotonic() - max(0.0, time.time() - launched)
        return live and pid == self.pid

    def terminate(self) -> bool:
        """SIGTERM the worker (only the process this worker is: a launched child, or an
        adopted one whose ``process.json`` start time still matches). Blocking."""
        if self.popen is not None:
            if self.popen.poll() is not None:
                return False
            self.popen.send_signal(signal.SIGTERM)
            return True
        if not self._adopted_still_runs():
            return False
        try:
            os.kill(self.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def kill(self) -> bool:
        """SIGKILL the worker and its process group (its ALFWorld games; a memory
        emergency victim that outlived its ``SIGTERM``), under the same
        identity check as :meth:`terminate`: a launched child leads its own session; an
        adopted one is killed with its group when it leads one, else alone (as the
        supervisor kills). Blocking."""
        if self.popen is not None:
            if self.popen.poll() is not None:
                return False
            return signal_group(self.pid, signal.SIGKILL)
        if not self._adopted_still_runs(switch=False):
            return False
        if self.process_group() == self.pid:
            return signal_group(self.pid, signal.SIGKILL)
        try:
            os.kill(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            return False
        return True


def learning_failure_kind(paths: CellPaths, record: Mapping[str, Any]) -> str | None:
    """The H8 kind of the failure that ended a ``learning_failed`` cell (blocking):
    ``status.json``'s ``failure_kind``, else (a status written before it carried one)
    the last attempt's recorded ``failure``."""
    kind = record.get("failure_kind")
    if isinstance(kind, str):
        return kind
    numbers = paths.attempt_numbers()
    if not numbers:
        return None
    state = read_json(paths.attempt(numbers[-1]).state)
    failure = state.get("failure") if isinstance(state, dict) else None
    kind = failure.get("kind") if isinstance(failure, dict) else None
    return kind if isinstance(kind, str) else None


def _harness_learning_failure(paths: CellPaths, record: Mapping[str, Any]) -> bool:
    return (
        record.get("status") == STATUS_LEARNING_FAILED
        and learning_failure_kind(paths, record) == FailureKind.HARNESS.value
    )


def classify_cell(
    paths: CellPaths,
    *,
    rerun_interrupted: bool,
    rerun_harness_failures: bool = False,
    auto_resume: bool = False,
) -> tuple[str, str | None]:
    """What the scheduler does with a cell (blocking): (``DONE`` | ``LIVE`` | ``QUEUED`` |
    ``WAITING`` | ``ATTENTION`` | ``BLOCKED``, the cell's status).

    * A cell whose learning a HARNESS failure stopped stays ``ATTENTION`` unless
      ``rerun_harness_failures`` too; that launch carries ``--new-attempt``.
    * With ``auto_resume``, an ``AUTO_RESUMABLE_STATUSES`` cell, or one whose worker died
      (``running`` without a live lock), is ``WAITING``: the scheduler re-queues it once
      its backoff has passed, unless a resume cap sends it to a person
      (:class:`ResumeTracker`). An attempt without learning checkpoints that
      infrastructure stopped ends ``learning_interrupted`` (``WAITING`` like any
      other; the next launch restarts it as a new attempt); the INFRA
      ``learning_failed`` an older ``run.py`` wrote for it is ``QUEUED``, as
      ``rerun_interrupted`` does.
    * Without it, those cells are ``ATTENTION`` unless ``rerun_interrupted``.
    """
    if cell_is_live(paths):
        return LIVE, STATUS_RUNNING
    record = read_status(paths) or {}
    status = record.get("status")
    if status == STATUS_COMPLETE:
        return DONE, status
    if status in BLOCKED_STATUSES:
        return BLOCKED, status
    rerun = rerun_interrupted or auto_resume
    if status == STATUS_LEARNING_FAILED:
        if _harness_learning_failure(paths, record):
            return (QUEUED if rerun and rerun_harness_failures else ATTENTION), status
        return (QUEUED if rerun else ATTENTION), status
    if (
        status in RESUMABLE_STATUSES
        or status in AUTO_RESUMABLE_STATUSES
        or status == STATUS_RUNNING
    ):
        # STATUS_RUNNING without a live lock: the worker died without a word.
        if auto_resume and (
            status in AUTO_RESUMABLE_STATUSES or status == STATUS_RUNNING
        ):
            return WAITING, status
        return (QUEUED if rerun_interrupted else ATTENTION), status
    return QUEUED, status  # never run, or stopped at the start (budget paused)


#: How long the sweep after a worker's exit waits for its cell lock.
SWEEP_LOCK_WAIT_S = 0.5
#: How long a launch waits when a pending HARNESS release meets attempts it cannot list,
#: and how many such waits in a row before the cell goes to a person.
RELEASE_UNREADABLE_RETRY_S = 60.0
RELEASE_UNREADABLE_MAX_DEFERRALS = 10
#: Ways a release is used up without a new attempt: a person may release it again.
RELEASE_NEVER_USED = frozenset({"unknown", "reset"})


def sweep_dead_cell(paths: CellPaths) -> list[int]:
    """Kill what a dead worker of the cell left running (its MIRIX server), blocking;
    the leader pids killed. Only while this scheduler holds the cell lock itself: every
    run.py holds it for its whole life, so with it held no run.py of the cell lives and
    the registry's live rows are orphans. A MIRIX server's row carries no owner lock of
    its own, so the cell lock is what keeps a live run.py's server alive: a lock someone
    holds (the run.py that made a worker exit busy, one a person started by hand) means
    no sweep, and that run.py's own start cleans up its cell."""
    registry = paths.root / "owned_processes.jsonl"
    if not registry.exists():
        return []
    try:
        # A moment's wait: a status probe (lock_is_held) may hold it briefly, while a
        # real holder keeps it for its whole life (and run.py retries for 2 s).
        lock = OwnerLock.acquire(paths.lock, wait_s=SWEEP_LOCK_WAIT_S)
    except OwnedProcessError:
        return []
    try:
        return cleanup_owned(registry, held_locks=(lock,))
    finally:
        lock.release()


def adopted_worker(spec: CellSpec, paths: CellPaths) -> Worker:
    """A live worker of another scheduler (blocking). Its ``started_at`` is its own start
    (``process.json`` ``at``) on this process's monotonic clock, so the memory emergency
    stop picks the most recently launched worker, adopted or not."""
    try:
        info = read_json(paths.process) if paths.process.exists() else None
    except ValueError:
        info = None
    pid = int(info["pid"]) if isinstance(info, dict) and "pid" in info else -1
    launched = epoch(info.get("at")) if isinstance(info, dict) else None
    started_at = time.monotonic()
    if launched is not None:
        started_at -= max(0.0, time.time() - launched)
    return Worker(spec, paths, pid, started_at=started_at)


# ---------------------------------------------------------------------------
# Auto-resume: backoff, caps and progress (resume_state.json)
# ---------------------------------------------------------------------------

#: How far a cell's attempt has come, for comparing progress (cell_progress's ``state``).
_STATE_RANK = {"frozen": 1, "testing": 2, "complete": 3}


def _is_count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _count(value: Any) -> int:
    return value if _is_count(value) else -1


def progress_key(progress: Any) -> tuple[int, ...] | None:
    """An orderable form of ``cell_records.cell_progress``: attempt, how far its state
    has come, checkpoint ordinal, learning items done, scored test items."""
    if not isinstance(progress, Mapping):
        return None
    return (
        _count(progress.get("attempt")),
        _STATE_RANK.get(str(progress.get("state")), 0),
        _count(progress.get("checkpoint_ordinal")),
        _count(progress.get("learning_done")),
        _count(progress.get("test_scored")),
    )


def progress_grew(before: Any, after: Any) -> bool:
    """True when ``after`` is further than ``before`` (a new checkpoint, more scored test
    items, a later state), or is a later attempt that got past its setup (a checkpoint
    after init, a learning item done, a later state, a scored test item). A new attempt
    alone is not progress: an attempt that an infrastructure failure stops in its setup,
    again and again, must reach the no-progress cap. Without a baseline the
    cell gets the benefit of the doubt; without a reading (an unreadable cell) it made
    none."""
    later = progress_key(after)
    if later is None:
        return False
    earlier = progress_key(before)
    if earlier is None:
        return True
    if later[0] != earlier[0]:
        return later[0] > earlier[0] and any(v > 0 for v in later[1:])
    return later[1:] > earlier[1:]


def status_marker(record: Mapping[str, Any]) -> str:
    """Identifies one ending of a cell (its ``status.json`` as last written), so an exit a
    scheduler has counted is not counted again by the next one."""
    return "|".join(
        str(record.get(k))
        for k in ("status", "updated_at", "launch", "pid", "exit_code")
    )


#: Joins the status marker a worker's death before its first status of its own was
#: counted on, and that death.
DIED_MARKER_SEP = "|worker_died:"


def died_marker(record: Mapping[str, Any], pid: int, exit_code: int) -> str:
    """Identifies the death of the worker ``pid`` before it wrote a status of its own:
    the ``status.json`` it left as it was (``record``, an earlier run's), that worker and
    its exit code. The same ending always gets the same marker, and a restarted scheduler
    that reads that status again sees it counted (:func:`marker_counted`; a marker with
    the clock in it would let it count that ending a second time)."""
    return f"{status_marker(record)}{DIED_MARKER_SEP}{pid}:{exit_code}"


def marker_counted(last: Any, marker: str) -> bool:
    """True when the last ending counted (``last``, ``resume_state.json``
    ``last_marker``) is the ending ``marker`` names, or a worker's death that left that
    very status behind: a scheduler that reads that status again (after a restart) must
    not count it again."""
    return isinstance(last, str) and (
        last == marker or last.startswith(marker + DIED_MARKER_SEP)
    )


def _attempt_of(progress: Any) -> int:
    """The latest attempt of a ``cell_progress`` reading (0: none, or unreadable)."""
    value = progress.get("attempt") if isinstance(progress, Mapping) else None
    return value if _is_count(value) else 0


def operator_resumed_since(paths: CellPaths, attempt: int, since: Any) -> bool:
    """Whether a person resumed attempt ``attempt`` by hand (``--force-resume``: an
    ``attempt.json`` ``resumes[]`` entry with ``operator_resume``) at or after ``since``
    (ISO 8601; the release of its HARNESS stop), blocking. A resume before the release
    belongs to an earlier stop."""
    if attempt <= 0:
        return False
    try:
        state = read_json(paths.attempt(attempt).state)
    except (OSError, ValueError):
        return False
    resumes = state.get("resumes") if isinstance(state, dict) else None
    floor = epoch(since)
    for entry in resumes if isinstance(resumes, list) else ():
        if not (isinstance(entry, Mapping) and entry.get("operator_resume") is True):
            continue
        at = epoch(entry.get("at"))
        if floor is None or (at is not None and at >= floor):
            return True
    return False


def attempt_run_id(paths: CellPaths, attempt: int) -> str | None:
    """The gateway run id of the cell's attempt ``attempt`` (``attempt.json``
    ``run_id``), or None (blocking): what tells an attempt from a started-over cell's
    attempt of the same number."""
    try:
        state = read_json(paths.attempt(attempt).state)
    except (OSError, ValueError):
        return None
    value = state.get("run_id") if isinstance(state, dict) else None
    return str(value) if value else None


def latest_attempt(paths: CellPaths) -> tuple[int, Any]:
    """The cell's latest attempt (0: none; -1: the attempts cannot be listed) and that
    attempt's state, read directly (blocking): what a released HARNESS stop is checked
    against when ``cell_progress`` cannot read the cell (a torn results file), so that a
    new attempt that exists never looks missing."""
    try:
        numbers = paths.attempt_numbers()
    except OSError:
        return -1, None
    if not numbers:
        return 0, None
    try:
        state = read_json(paths.attempt(numbers[-1]).state)
    except (OSError, ValueError):
        state = None
    return numbers[-1], state.get("state") if isinstance(state, dict) else None


@dataclass(frozen=True)
class ResumeDecision:
    """Where an auto-resumable cell goes: back to the queue at ``next_at`` (epoch), or to
    a person (``attention``: :data:`REASON_RESUME_CAP` or :data:`REASON_RESUME_CAP_TOTAL`)."""

    next_at: float | None
    delay_s: float | None
    attention: str | None
    #: False when the exit had been counted already (by this or an earlier scheduler).
    new: bool
    entry: Mapping[str, Any]


class ResumeTracker:
    """Per-cell resume counters of ``--auto-resume``, kept in
    ``runs/_schedule/resume_state.json`` (atomic; blocking methods, async callers use
    ``asyncio.to_thread``).

    Each cell's entry: ``resumes`` (every resume scheduled), ``without_progress`` (the
    consecutive exits without progress), ``backoff_index`` (the exponent of the current
    delay), ``last_status``, ``last_exit_code``, ``last_exit_at``, ``next_at``,
    ``progress`` (``cell_progress`` at the last exit or launch), ``attention`` (a cap
    reached), ``last_marker`` (the last ending counted) and the last
    :data:`RESUME_HISTORY` exits in ``history``; for a released HARNESS stop,
    ``harness_approved`` (its marker), ``harness_attempt`` (the attempt it stopped) and,
    once its new attempt exists, ``harness_relaunched`` and ``harness_new_attempt``;
    for ``--new-attempt-cells``, ``rerun_requests``: ``{key: {note,
    requested_at, from_attempt, from_run_id, estimate_usd, state, new_attempt}}`` with
    key :func:`rerun_key` (``state``: ``pending`` until the new attempt exists, then
    ``started``; ``replaced`` (with ``replaced_by``) when a later request with another
    note arrived while it was pending; ``reset`` when the cell was started over).
    """

    def __init__(
        self,
        path: Path,
        *,
        base_s: float = DEFAULT_RESUME_BACKOFF_BASE_S,
        cap_s: float = DEFAULT_RESUME_BACKOFF_CAP_S,
        max_without_progress: int = DEFAULT_MAX_RESUMES_WITHOUT_PROGRESS,
        max_resumes: int = DEFAULT_MAX_RESUMES,
    ) -> None:
        self.path = path
        self.base_s = base_s
        self.cap_s = cap_s
        self.max_without_progress = max_without_progress
        self.max_resumes = max_resumes
        self.cells: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            value = read_json(self.path)
        except (OSError, ValueError) as exc:
            # Written atomically, so this is a hand edit gone wrong: start over, loudly.
            self.load_error = f"{type(exc).__name__}: {exc}"[:300]
            value = None
        self.cells = (
            {k: dict(v) for k, v in value.items() if isinstance(v, dict)}
            if isinstance(value, dict)
            else {}
        )

    def save(self) -> None:
        write_json_atomic(self.path, self.cells)

    def delay_s(self, without_progress: int) -> float:
        """The wait before a resume after ``without_progress`` consecutive exits without
        progress: ``min(cap, base * 2**(k-1))`` (k = 0 after progress: ``base``)."""
        exponent = max(0, without_progress - 1)
        return min(self.cap_s, self.base_s * (2 ** min(exponent, 32)))

    def cap_for(self, status: str | None) -> int:
        if status == STATUS_CRASHED:
            return min(self.max_without_progress, CRASHED_MAX_RESUMES_WITHOUT_PROGRESS)
        return self.max_without_progress

    def _entry(self, cell_id: str) -> dict[str, Any]:
        return self.cells.setdefault(
            cell_id,
            {
                "resumes": 0,
                "without_progress": 0,
                "backoff_index": 0,
                "history": [],
            },
        )

    def decision(self, cell_id: str, *, new: bool = False) -> ResumeDecision:
        entry = self._entry(cell_id)
        attention = entry.get("attention")
        next_at = None if attention else epoch(entry.get("next_at"))
        return ResumeDecision(
            next_at=next_at,
            delay_s=entry.get("delay_s"),
            attention=attention,
            new=new,
            entry=dict(entry),
        )

    def note_launch(self, cell_id: str, progress: Any, at: float) -> None:
        """A worker starts: its progress now is the baseline its exit is compared with."""
        entry = self._entry(cell_id)
        entry["progress"] = progress
        entry["launched_at"] = iso(at)
        entry["next_at"] = None
        self.save()

    def note_exit(
        self,
        cell_id: str,
        *,
        status: str | None,
        exit_code: int | None,
        progress: Any,
        marker: str | None,
        at: float,
        counts: bool = True,
        reason: str | None = None,
    ) -> ResumeDecision:
        """Count one auto-resumable ending of a cell and decide its resume.

        ``counts=False``: the ending was not the cell's (``reason``: the scheduler's
        memory emergency stop, or a ``SIGTERM``/``SIGINT`` from the supervisor's drain,
        a reboot or an operator): it is resumed after ``base`` and adds to ``resumes``
        only.
        An ending already counted (same ``marker``, or a worker's death that left that
        status behind: :func:`marker_counted`) changes nothing.
        """
        entry = self._entry(cell_id)
        if marker is not None and marker_counted(entry.get("last_marker"), marker):
            return self.decision(cell_id)
        grew = progress_grew(entry.get("progress"), progress)
        if grew:
            entry["without_progress"] = 0
        elif counts:
            entry["without_progress"] = int(entry.get("without_progress", 0)) + 1
        without = int(entry.get("without_progress", 0))
        attention = None
        if without > self.cap_for(status):
            attention = REASON_RESUME_CAP
        elif int(entry.get("resumes", 0)) >= self.max_resumes:
            attention = REASON_RESUME_CAP_TOTAL
        delay = None
        if attention is None:
            delay = self.delay_s(without if counts else 0)
            entry["backoff_index"] = max(0, without - 1) if counts else 0
            entry["resumes"] = int(entry.get("resumes", 0)) + 1
            entry["next_at"] = iso(at + delay)
        else:
            entry["next_at"] = None
        entry.update(
            attention=attention,
            delay_s=delay,
            last_status=status,
            last_exit_code=exit_code,
            last_exit_at=iso(at),
            last_marker=marker,
            progress=progress,
        )
        history = list(entry.get("history") or [])
        history.append(
            {
                "at": iso(at),
                "status": status,
                "exit_code": exit_code,
                "delay_s": delay,
                "progress": progress,
                "progressed": grew,
                **({"reason": reason} if reason else {}),
                **({"attention": attention} if attention else {}),
            }
        )
        entry["history"] = history[-RESUME_HISTORY:]
        self.save()
        return self.decision(cell_id, new=True)

    def died_on(self, cell_id: str, marker: str) -> bool:
        """True when the last ending counted for ``cell_id`` is a worker's death that
        left the status ``marker`` names behind."""
        last = (self.cells.get(cell_id) or {}).get("last_marker")
        return isinstance(last, str) and last.startswith(marker + DIED_MARKER_SEP)

    def reset(self, cell_ids: Sequence[str], at: float) -> list[str]:
        """Clear the counters of ``cell_ids`` (a person looked): a cell a cap stopped is
        resumed at once, from counters at zero. Returns the cells that had an entry."""
        done = []
        for cell_id in cell_ids:
            entry = self.cells.get(cell_id)
            if entry is None:
                continue
            entry.update(
                resumes=0,
                without_progress=0,
                backoff_index=0,
                attention=None,
                next_at=None,
                delay_s=None,
            )
            history = list(entry.get("history") or [])
            history.append({"at": iso(at), "event": "reset"})
            entry["history"] = history[-RESUME_HISTORY:]
            done.append(cell_id)
        self.save()
        return done

    # -- HARNESS stops a person released (--rerun-harness-failures) --------

    def approve_harness(
        self,
        cell_id: str,
        marker: str,
        at: float,
        *,
        attempt: int | None,
        run_id: str | None = None,
    ) -> bool:
        """A person released the HARNESS stop ``marker`` of ``cell_id`` (which stopped
        attempt ``attempt``, of gateway run ``run_id``) for one new attempt; False when
        that very stop had its new attempt already. Releasing a stop that is released
        already keeps the attempt it recorded. ``attempt`` None: it could
        not be read; :meth:`harness_release` takes the latest while the status is still
        that stop."""
        entry = self._entry(cell_id)
        if (
            entry.get("harness_relaunched") == marker
            and entry.get("harness_relaunched_reason") not in RELEASE_NEVER_USED
        ):
            return False
        if entry.get("harness_relaunched") == marker:
            # Used up without a new attempt (the cell looked started over, or the stop
            # was gone): released again.
            entry.pop("harness_relaunched", None)
            entry.pop("harness_relaunched_reason", None)
        recorded = entry.get("harness_attempt")
        if not (entry.get("harness_approved") == marker and _is_count(recorded)):
            if attempt is None:
                entry.pop("harness_attempt", None)
            else:
                entry["harness_attempt"] = attempt
            if run_id:
                entry["harness_run_id"] = run_id
            else:
                entry.pop("harness_run_id", None)
        entry["harness_approved"] = marker
        entry["harness_approved_at"] = iso(at)
        self.save()
        return True

    def harness_pending(self, cell_id: str) -> str | None:
        """The released HARNESS stop of ``cell_id`` whose new attempt does not exist yet
        (as last seen: :meth:`harness_release` looks)."""
        entry = self.cells.get(cell_id) or {}
        approved = entry.get("harness_approved")
        if approved and approved != entry.get("harness_relaunched"):
            return str(approved)
        return None

    def harness_release(
        self,
        cell_id: str,
        *,
        latest: int,
        state: Any,
        marker: str,
        at: float,
        hand_resumed: bool = False,
        latest_run_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Whether the next launch of ``cell_id`` still starts the new attempt of its
        released HARNESS stop (``--new-attempt``): ``(that stop's marker or None, why
        the release was used up now or None)``. ``latest``/``state``: the cell's latest
        attempt and its state now; ``marker``: its status marker now;
        ``hand_resumed``: a person resumed the stopped attempt (``--force-resume``) since
        the release (:func:`operator_resumed_since`).

        The release is used up once the new attempt exists, i.e. once the latest attempt
        is past the stopped one (``new_attempt``), or when the stopped attempt went on
        without one (a person resumed it by hand: ``resumed``); never at a launch, which
        may end before run.py reaches the stopped attempt. A release kept
        before the stopped attempt was recorded counts from the latest attempt while the
        status is still that stop, else it is used up (``unknown``: never a second new
        attempt)."""
        pending = self.harness_pending(cell_id)
        if pending is None:
            return None, None
        entry = self._entry(cell_id)
        stopped = entry.get("harness_attempt")
        if not _is_count(stopped):
            if marker != pending:
                self.note_harness_relaunch(
                    cell_id, pending, at, attempt=latest, reason="unknown"
                )
                return None, "unknown"
            stopped = entry["harness_attempt"] = latest
            if latest_run_id:
                # Its run too, so a started-over cell is told apart.
                entry["harness_run_id"] = latest_run_id
            self.save()
        if latest > stopped:
            self.note_harness_relaunch(
                cell_id, pending, at, attempt=latest, reason="new_attempt"
            )
            return None, "new_attempt"
        if latest == stopped and (state in _STATE_RANK or hand_resumed):
            # The stopped attempt froze, is testing or complete: --new-attempt would
            # throw that away (or be refused once a test item is scored). Or a person
            # resumed it by hand and it is still learning, or was interrupted again:
            # --new-attempt would throw that progress away too.
            self.note_harness_relaunch(
                cell_id, pending, at, attempt=latest, reason="resumed"
            )
            return None, "resumed"
        return pending, None

    def note_harness_relaunch(
        self,
        cell_id: str,
        marker: str,
        at: float,
        *,
        attempt: int | None = None,
        reason: str | None = None,
    ) -> None:
        """The released stop ``marker`` is used up: its new attempt (``attempt``) exists
        (``reason``: :meth:`harness_release`)."""
        entry = self._entry(cell_id)
        entry["harness_relaunched"] = marker
        entry["harness_relaunched_at"] = iso(at)
        if reason:
            entry["harness_relaunched_reason"] = reason
        else:
            entry.pop("harness_relaunched_reason", None)
        if attempt is not None:
            entry["harness_new_attempt"] = attempt
        history = list(entry.get("history") or [])
        history.append(
            {
                "at": iso(at),
                "event": "harness_new_attempt",
                **({"attempt": attempt} if attempt is not None else {}),
                **({"reason": reason} if reason else {}),
            }
        )
        entry["history"] = history[-RESUME_HISTORY:]
        self.save()

    # -- new attempts a person asked for (--new-attempt-cells) ------------------

    def approve_rerun(
        self,
        cell_id: str,
        *,
        key: str,
        note: str,
        from_attempt: int | None,
        from_run_id: str | None,
        estimate_usd: float | None,
        at: float,
    ) -> bool:
        """Record a request for one new attempt of ``cell_id`` replacing attempt
        ``from_attempt``; False when this note (``key``) was requested for the cell
        before, pending or done: a scheduler restarted with the same command line never
        reruns twice, a deliberate second rerun needs a new note. Another pending request
        of the cell is replaced by this one (one new attempt at a time)."""
        entry = self._entry(cell_id)
        requests = entry.get("rerun_requests")
        requests = dict(requests) if isinstance(requests, dict) else {}
        if key in requests:
            return False
        for other in requests.values():
            if isinstance(other, dict) and other.get("state") == "pending":
                other.update(state="replaced", replaced_by=key, replaced_at=iso(at))
        requests[key] = {
            "note": note,
            "requested_at": iso(at),
            "from_attempt": from_attempt,
            "from_run_id": from_run_id,
            "estimate_usd": estimate_usd,
            "state": "pending",
            "new_attempt": None,
        }
        entry["rerun_requests"] = requests
        history = list(entry.get("history") or [])
        history.append(
            {
                "at": iso(at),
                "event": "new_attempt_requested",
                "key": key,
                "from_attempt": from_attempt,
            }
        )
        entry["history"] = history[-RESUME_HISTORY:]
        self.save()
        return True

    def rerun_pending(self, cell_id: str) -> tuple[str, dict[str, Any]] | None:
        """The pending ``--new-attempt-cells`` request of ``cell_id`` (key, request)."""
        requests = (self.cells.get(cell_id) or {}).get("rerun_requests")
        for key, request in (requests if isinstance(requests, dict) else {}).items():
            if isinstance(request, dict) and request.get("state") == "pending":
                return str(key), dict(request)
        return None

    def set_rerun_from(
        self, cell_id: str, key: str, attempt: int, run_id: str | None
    ) -> None:
        """A request recorded while the attempts could not be read replaces the latest
        attempt, read now."""
        request = self.cells[cell_id]["rerun_requests"][key]
        request.update(from_attempt=attempt, from_run_id=run_id)
        self.save()

    def note_rerun_started(
        self,
        cell_id: str,
        key: str,
        attempt: int | None,
        at: float,
        *,
        state: str = "started",
    ) -> None:
        """The request ``key`` is used up: its new attempt ``attempt`` exists (``state``
        ``started``), or the cell was started over (``reset``)."""
        entry = self._entry(cell_id)
        request = entry["rerun_requests"][key]
        request.update(state=state, new_attempt=attempt, started_at=iso(at))
        history = list(entry.get("history") or [])
        history.append(
            {
                "at": iso(at),
                "event": f"new_attempt_{state}",
                "key": key,
                **({"attempt": attempt} if attempt is not None else {}),
            }
        )
        entry["history"] = history[-RESUME_HISTORY:]
        self.save()

    def reruns(self, cell_ids: Sequence[str]) -> dict[str, Any]:
        """Each cell's pending (else latest) ``--new-attempt-cells`` request, for
        ``state.json`` ``new_attempts``."""
        out = {}
        for cell_id in cell_ids:
            requests = (self.cells.get(cell_id) or {}).get("rerun_requests")
            if not isinstance(requests, dict) or not requests:
                continue
            found = self.rerun_pending(cell_id)
            request = (
                found[1]
                if found
                else max(
                    (r for r in requests.values() if isinstance(r, dict)),
                    key=lambda r: str(r.get("requested_at")),
                    default=None,
                )
            )
            if request is None:
                continue
            out[cell_id] = {
                k: request.get(k)
                for k in (
                    "note",
                    "from_attempt",
                    "estimate_usd",
                    "state",
                    "new_attempt",
                )
            }
        return out

    def summary(self, cell_ids: Sequence[str]) -> dict[str, Any]:
        """The counters of ``cell_ids`` for ``state.json`` (cells without an entry left out)."""
        out = {}
        for cell_id in cell_ids:
            entry = self.cells.get(cell_id)
            if entry is None:
                continue
            out[cell_id] = {
                k: entry.get(k)
                for k in (
                    "resumes",
                    "without_progress",
                    "backoff_index",
                    "last_status",
                    "last_exit_code",
                    "last_exit_at",
                    "next_at",
                    "attention",
                )
            }
        return out


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------


@dataclass
class ScheduleResult:
    complete: list[str] = field(default_factory=list)
    attention: dict[str, str | None] = field(default_factory=dict)
    #: Why each ``attention`` cell needs a person (``REASON_*``).
    attention_reasons: dict[str, str] = field(default_factory=dict)
    blocked: dict[str, str | None] = field(default_factory=dict)
    launched: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    queued: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    #: ``--auto-resume``: cells waiting out their backoff, with their wake-up (ISO).
    waiting: dict[str, str] = field(default_factory=dict)
    paused: bool = False
    #: Why the scheduler stopped paused (a gateway or scheduler pause reason).
    pause_reason: str | None = None
    stopped: bool = False

    @property
    def exit_code(self) -> int:
        if self.paused or self.stopped:
            return EXIT_PAUSED
        if (
            self.attention
            or self.blocked
            or self.queued
            or self.running
            or self.waiting
        ):
            return EXIT_ATTENTION
        return EXIT_ALL_COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "needs_attention": self.attention,
            "attention_reasons": self.attention_reasons,
            "blocked": self.blocked,
            "launched": self.launched,
            "adopted": self.adopted,
            "queued": self.queued,
            "running": self.running,
            "waiting": self.waiting,
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "stopped": self.stopped,
            "exit_code": self.exit_code,
        }


#: Reads the host: ``(runs_dir, workers={cell: [pgid, ...]}) -> HostResources`` (blocking).
Sampler = Callable[..., resources.HostResources]


class Scheduler:
    def __init__(
        self,
        cells: Sequence[CellSpec],
        options: ScheduleOptions,
        *,
        clock: Callable[[], float] = time.time,
        sampler: Sampler | None = None,
        docker_sampler: Callable[[], dict[str, float] | None] | None = None,
        webshop_ready: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
    ) -> None:
        """``clock`` (wall clock, for resume wake-ups), ``sampler`` (the host reading),
        ``docker_sampler`` and ``webshop_ready`` (async; whether the WebShop server is
        ready, and why not) default to the real ones."""
        ids = [c.cell_id for c in cells]
        if len(set(ids)) != len(ids):
            raise ValueError("the matrix names a cell twice")
        self.cells = list(cells)
        self.options = options
        self.gateway: SchedulerGateway | None = None
        self.active: dict[str, Worker] = {}
        self.queue: list[CellSpec] = []
        #: ``--auto-resume``: cell -> when it is re-queued (epoch).
        self.waiting: dict[str, float] = {}
        self.result = ScheduleResult()
        self.stop = asyncio.Event()
        self._balance_at = float("-inf")
        self.pause_reason: str | None = None
        self.budget: dict[str, Any] = {}
        self.started: str | None = None
        #: Wall clock for resume wake-ups (persisted, so they survive a restart).
        self.clock = clock
        self.sampler = sampler or resources.sample
        self.docker_sampler = docker_sampler or resources.docker_memory
        self.webshop_ready = webshop_ready or webshop_server_ready
        #: The last WebShop readiness reading (monotonic time, ready, why).
        self._webshop_reading: tuple[float, bool, str] | None = None
        #: The failed server's reason last recorded (``webshop_server_failed``).
        self._webshop_failed: str | None = None
        self.tracker = ResumeTracker(
            options.control / RESUME_STATE_FILE,
            base_s=options.resume_backoff_base_s,
            cap_s=options.resume_backoff_cap_s,
            max_without_progress=options.max_resumes_without_progress,
            max_resumes=options.max_resumes,
        )
        self._order = {c.cell_id: i for i, c in enumerate(self.cells)}
        self._specs = {c.cell_id: c for c in self.cells}
        #: Workers the scheduler itself stopped, with the reason.
        self._stopping: dict[str, str] = {}
        #: Memory emergency stops per cell (the victim rotates), and no new
        #: victim before this monotonic time (a stopped worker's memory settles).
        self._emergency_stops: dict[str, int] = {}
        self._emergency_settle_until = float("-inf")
        #: When (monotonic) each emergency victim got its SIGTERM, and the victims that
        #: outlived it by EMERGENCY_KILL_AFTER_S: whether their SIGKILL went out. Those
        #: no longer hold back the next stop.
        self._emergency_signalled: dict[str, float] = {}
        self._emergency_killed: dict[str, bool] = {}
        #: Cells whose pending HARNESS release met attempts it could not list, and how
        #: often in a row their launch was deferred for it.
        self._release_unreadable: set[str] = set()
        self._release_deferrals: dict[str, int] = {}
        #: Cells whose pending --new-attempt-cells request met attempts it could not
        #: read (their launch is deferred like a HARNESS release's).
        self._rerun_unreadable: set[str] = set()
        #: Gateways on other ports than ``gateway``'s, for MIRIX cells whose agent
        #: recorded one: ``{port: gateway}``, closed like the main one.
        self.extra_gateways: dict[int, SchedulerGateway] = {}
        #: Launches that went past a cell the memory gate held, per held cell (recorded
        #: in the resource_skip events; no limit).
        self._skips: dict[str, int] = {}
        #: The cell the resource gate holds: its id, the gate and since when (ISO).
        self._resource_wait_info: dict[str, Any] | None = None
        #: The one-off options (``--reset-resume``, ``--rerun-harness-failures``) were
        #: applied (``state.json`` says so: the supervisor stops passing them).
        self.one_off_applied = False
        #: The last host reading, for ``state.json`` and the resource guard.
        self.resources: dict[str, Any] | None = None
        self._docker: dict[str, Any] | None = None
        self._docker_task: asyncio.Task[Any] | None = None
        self._docker_at = float("-inf")
        self._low_memory_samples = 0
        self._resources_logged_at = float("-inf")
        self._resource_wait: tuple[str, str] | None = None
        # Containers whose memory ``docker stats`` reads into the resources log: the
        # compose Postgres of the MIRIX cells and the WebShop server (logged next to the
        # host's own memory, to size the caps on).
        self._has_containers = any(
            c.benchmark == "webshop" or c.method == "mirix" for c in self.cells
        )
        #: The host's other schedulers (``host_registry.py``; for example replicate
        #: schedulers on one host): the memory gate reserves their workers'
        #: headroom too, ``--host-benchmark-concurrency`` counts their cells, and one
        #: memory emergency stops one worker host-wide.
        self.host = host_registry.HostRegistry(options.host_directory, options.runs_dir)
        #: Cells admitted under the host lock whose launch has not returned yet: in this
        #: scheduler's host record, so no peer admits on the memory they will take.
        self._host_pending: dict[str, CellSpec] = {}
        #: The peers last read (``state.json`` ``host``).
        self._host_peers: list[host_registry.Peer] = []
        #: When (epoch seconds) this scheduler last stopped a worker in a memory
        #: emergency (its host record: a peer stops none for EMERGENCY_SETTLE_S).
        self._host_emergency_at: float | None = None
        #: Cells the host-wide cap holds back, with the reason (``state.json``).
        self._host_held: dict[str, str] = {}
        #: The host directory's last problem, and every problem recorded so far (each
        #: one is logged and recorded once: ``host_registry_unavailable``).
        self._host_problem: str | None = None
        self._host_problems_seen: set[str] = set()
        #: The peers this scheduler last left a memory emergency stop to (recorded once
        #: per stretch they hold it: ``memory_emergency_left_to_peer``).
        self._host_emergency_seen: tuple[Any, ...] = ()
        #: Whether this scheduler has adopted the workers it found (``classify``): only
        #: then does it publish its host record, so a restarted scheduler never replaces
        #: its predecessor's orphaned record of those workers with an empty one, and a
        #: scheduler that stops before that leaves the predecessor's record as it is.
        self._host_ready = False
        #: The peers' host caps that differ from this scheduler's own, recorded once
        #: each (``host_cap_mismatch``).
        self._host_cap_mismatches: set[tuple[Any, ...]] = set()
        #: Since when (epoch seconds) the host cap holds this scheduler back from each
        #: benchmark while it runs none of its cells (its record's ``waiting``).
        self._host_waiting_since: dict[str, float] = {}

    # -- records -----------------------------------------------------------------------

    async def event(self, name: str, **fields: Any) -> None:
        record = {"at": now(), "event": name, "pid": os.getpid(), **fields}
        await asyncio.to_thread(
            append_jsonl, self.options.control / "events.jsonl", record
        )

    async def write_state(self, *, final: bool = False) -> None:
        """``state.json``: the live queue and workers on every poll; ``exit_code`` stays
        null until the ``final`` write, when the scheduler stops (the result's
        ``queued``, ``running`` and exit code are filled only then)."""
        state = {
            "at": now(),
            "pid": os.getpid(),
            # The process's start time: the supervisor adopts a live scheduler by it.
            "started": self.started,
            "gateway": None
            if self.gateway is None
            else {
                "url": self.gateway.base_url,
                "mode": self.gateway.mode,
                "pid": self.gateway.pid,
            },
            # The extra gateways of MIRIX cells' recorded ports.
            "gateways": {
                str(port): {
                    "url": gw.base_url,
                    "mode": gw.mode,
                    "pid": gw.pid,
                    # An adopted one that is not healthy or runs with another budget:
                    # the supervisor alerts it (gateway_needs_attention).
                    "problem": gw.problem,
                }
                for port, gw in sorted(self.extra_gateways.items())
            },
            "concurrency": self.options.concurrency,
            "benchmark_concurrency": self.options.benchmark_caps,
            "auto_resume": self.options.auto_resume,
            "budget": self.budget,
            **self.result.to_dict(),
            "active": {c: w.pid for c, w in self.active.items()},
            "queued": [c.cell_id for c in self.queue],
            "running": sorted(self.active),
            "waiting": {
                c: iso(t) for c, t in sorted(self.waiting.items(), key=lambda i: i[1])
            },
            "pause_reason": self.pause_reason,
            # The cell the resource gate holds and since when (the supervisor alerts on
            # a long hold with nothing running).
            "resource_wait": self._resource_wait_info,
            # The host's other schedulers and the host-wide caps (host_registry.py).
            "host": self._host_state(),
            "one_off_applied": self.one_off_applied,
            "resources": self.resources,
            "resume": self.tracker.summary([c.cell_id for c in self.cells])
            if self.options.auto_resume
            else {},
            # --new-attempt-cells requests and their estimates.
            "new_attempts": self.tracker.reruns([c.cell_id for c in self.cells]),
            "exit_code": self.result.exit_code if final else None,
        }
        await asyncio.to_thread(
            write_json_atomic, self.options.control / "state.json", state
        )
        if not final:
            await self._publish_host()

    # -- budget --------------------------------------------------------------------------

    async def gateway_pause(self) -> str | None:
        """Why nothing may launch now, or None: the gateway is paused (its
        ``pause_reason``), unreachable, or its balance is unknown or below the floor."""
        assert self.gateway is not None
        fresh = time.monotonic() - self._balance_at >= self.options.balance_poll_s
        path = "/balance" if fresh else "/health"
        try:
            async with httpx.AsyncClient(
                base_url=self.gateway.base_url,
                headers={"Authorization": f"Bearer {LOCAL_TOKEN}"},
                timeout=30,
                trust_env=False,
            ) as client:
                response = await client.get(path)
            response.raise_for_status()
            health = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.budget = {"reachable": False, "error": type(exc).__name__}
            return PAUSE_GATEWAY_UNREACHABLE
        if fresh:
            self._balance_at = time.monotonic()
        balance = health.get("balance") or {}
        remaining = balance.get("remaining_usd")
        floor = self.options.balance_floor_usd
        if floor is None:
            reported = health.get("balance_floor_usd")
            floor = (
                float(reported)
                if isinstance(reported, (int, float)) and not isinstance(reported, bool)
                else DEFAULT_BALANCE_FLOOR_USD
            )
        known = isinstance(remaining, (int, float)) and not isinstance(remaining, bool)
        spend = health.get("budget") if isinstance(health.get("budget"), dict) else {}
        left = spend.get("left_usd")
        reason: str | None = None
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and left <= 0
            and not spend.get("error")
        ):
            # The budget is used up, whatever the gateway's pause says (a key below its
            # floor too reads balance_below_floor on an older gateway): a person raises
            # it.
            reason = BUDGET_SPENT
        elif health.get("paused"):
            reason = str(health.get("pause_reason") or "paused")
        elif not known:
            reason = PAUSE_BALANCE_UNKNOWN
        elif remaining < floor:
            reason = PAUSE_BALANCE_LOW
        self.budget = {
            "reachable": True,
            "paused": reason is not None,
            "gateway_paused": bool(health.get("paused")),
            "pause_reason": health.get("pause_reason"),
            "scheduler_pause_reason": reason,
            "remaining_usd": remaining,
            "floor_usd": floor,
            # The gateway's spend budget (--budget), when it runs with one.
            "spend": health.get("budget"),
            "checked_at": now(),
        }
        return reason

    def _waits_through(self, reason: str) -> bool:
        return self._pause_kind(reason) == "wait"

    def _pause_kind(self, reason: str) -> str:
        if reason in STOP_PAUSE_REASONS:
            return "stop"
        if self.options.auto_resume and reason in AUTO_RESUME_WAITABLE_PAUSES:
            return "wait"
        return "idle_exit"

    # -- queue ---------------------------------------------------------------------------

    def _enqueue(self, spec: CellSpec) -> None:
        """Queue ``spec`` in matrix order among the queued cells (the cells a pause sent
        back stay at the front)."""
        index = self._order.get(spec.cell_id, len(self._order))
        for position, queued in enumerate(self.queue):
            if self._order.get(queued.cell_id, len(self._order)) > index:
                self.queue.insert(position, spec)
                return
        self.queue.append(spec)

    def _attend(self, cell_id: str, status: str | None, reason: str) -> None:
        self.result.attention[cell_id] = status
        self.result.attention_reasons[cell_id] = reason

    async def _progress(self, paths: CellPaths) -> Any:
        try:
            return await asyncio.to_thread(cell_progress, paths)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.warning("cannot read the progress of %s: %s", paths.root.name, exc)
            return None

    async def _place(
        self,
        spec: CellSpec,
        kind: str,
        status: str | None,
        *,
        exit_code: int | None = None,
        stopped_by: str | None = None,
    ) -> None:
        """Put a cell that is not running where ``kind`` (``classify_cell``) sends it."""
        cell_id = spec.cell_id
        paths = CellPaths(self.options.cell_dir(spec))
        if kind in (DONE, QUEUED, ATTENTION) and (
            await self._rerun_release(spec, paths) is not None
            or cell_id in self._rerun_unreadable
        ):
            # A person asked for a new attempt of this cell (--new-attempt-cells) and
            # it does not exist yet: the cell runs, whatever its latest
            # attempt reached (a complete cell too, a HARNESS stop too); its launches
            # carry --new-attempt --note until that attempt exists. Blocked cells stay
            # blocked; a cell waiting out a resume backoff keeps it.
            self._enqueue(spec)
            return
        if kind == DONE:
            self.result.complete.append(cell_id)
        elif kind == BLOCKED:
            self.result.blocked[cell_id] = status
        elif kind == QUEUED:
            record = await asyncio.to_thread(read_status, paths) or {}
            marker = status_marker(record)
            harness = await asyncio.to_thread(_harness_learning_failure, paths, record)
            if harness and self.tracker.harness_pending(cell_id) != marker:
                # --rerun-harness-failures: one new attempt per HARNESS stop a person
                # released (the stops that existed when that scheduler started, kept in
                # resume_state.json). A stop that happened since, or the new attempt
                # failing the same way, waits for the person again. The
                # launches carry --new-attempt until it exists (_harness_release).
                self._attend(cell_id, status, REASON_HARNESS_FAILURE)
                return
            if self.options.auto_resume and (
                self.tracker.died_on(cell_id, marker)
                or (not harness and status == STATUS_LEARNING_FAILED)
            ):
                # A worker that died before its first status on top of this status (a
                # budget pause, a released HARNESS stop; counted by an earlier
                # scheduler: its backoff and caps still hold after a restart), or an
                # attempt without checkpoints that infrastructure stopped
                # (run.py starts a new one): the same backoff and caps as every other
                # resumable ending, also at a scheduler's start (a cap reached stays
                # reached).
                await self._schedule_resume(spec, paths, status, exit_code=exit_code)
                return
            self._enqueue(spec)
        elif kind == WAITING:
            await self._schedule_resume(
                spec, paths, status, exit_code=exit_code, stopped_by=stopped_by
            )
        else:
            record = await asyncio.to_thread(read_status, paths) or {}
            harness = await asyncio.to_thread(_harness_learning_failure, paths, record)
            self._attend(
                cell_id, status, REASON_HARNESS_FAILURE if harness else REASON_NOT_RERUN
            )

    async def _schedule_resume(
        self,
        spec: CellSpec,
        paths: CellPaths,
        status: str | None,
        *,
        exit_code: int | None,
        stopped_by: str | None = None,
        marker: str | None = None,
    ) -> None:
        """Count an auto-resumable ending and put the cell in ``waiting`` (or, past a
        cap, to a person). ``marker`` overrides the status-based one (a worker that died
        before writing a status of its own)."""
        record = await asyncio.to_thread(read_status, paths) or {}
        label = STATUS_WORKER_DIED if status in (STATUS_RUNNING, None) else status
        if exit_code is None and isinstance(record.get("exit_code"), int):
            exit_code = int(record["exit_code"])
        progress = await self._progress(paths)
        # A SIGTERM/SIGINT is someone else's (the supervisor's drain, a reboot, an
        # operator, the memory emergency stop), never the cell's failure: it is resumed
        # after the base wait and counts towards --max-resumes only. run.py records the
        # signal it received; an ``interrupted`` without one is a cancellation nobody
        # sent (a harness bug) and counts like any failure.
        signalled = label == STATUS_INTERRUPTED and bool(record.get("signal"))
        external = stopped_by or ("signal" if signalled else None)
        decision = await asyncio.to_thread(
            self.tracker.note_exit,
            spec.cell_id,
            status=label,
            exit_code=exit_code,
            progress=progress,
            marker=marker or status_marker(record),
            at=self.clock(),
            counts=external is None,
            reason=external,
        )
        entry = decision.entry
        if decision.attention:
            self._attend(spec.cell_id, status, decision.attention)
            if decision.new:
                await self.event(
                    "resume_cap",
                    cell=spec.cell_id,
                    status=label,
                    reason=decision.attention,
                    resumes=entry.get("resumes"),
                    without_progress=entry.get("without_progress"),
                )
            return
        next_at = decision.next_at if decision.next_at is not None else self.clock()
        self.waiting[spec.cell_id] = next_at
        if decision.new:
            await self.event(
                "resume_scheduled",
                cell=spec.cell_id,
                status=label,
                exit_code=exit_code,
                delay_s=decision.delay_s,
                next_at=iso(next_at),
                resumes=entry.get("resumes"),
                without_progress=entry.get("without_progress"),
                progress=progress,
                **({"reason": stopped_by} if stopped_by else {}),
            )

    def _promote_due(self) -> list[str]:
        """Move the waiting cells whose wake-up has come into the queue, in matrix order."""
        at = self.clock()
        due = sorted(
            (c for c, t in self.waiting.items() if t <= at),
            key=lambda c: self._order.get(c, len(self._order)),
        )
        for cell_id in due:
            del self.waiting[cell_id]
            self._enqueue(self._specs[cell_id])
        return due

    # -- workers -------------------------------------------------------------------------

    async def launch(self, spec: CellSpec) -> bool:
        """Spawn the cell's worker; False when the launch was deferred or handed to a
        person instead (a pending HARNESS release that cannot be read)."""
        assert self.gateway is not None
        paths = CellPaths(self.options.cell_dir(spec))
        progress = await self._progress(paths) if self.options.auto_resume else None
        # A released HARNESS stop: every launch carries --new-attempt until its new
        # attempt exists. A --new-attempt-cells request likewise, with
        # its note.
        released = await self._harness_release(spec, paths, progress)
        note = await self._rerun_release(spec, paths)
        unreadable = (
            REASON_RELEASE_UNREADABLE
            if spec.cell_id in self._release_unreadable
            else REASON_RERUN_UNREADABLE
            if spec.cell_id in self._rerun_unreadable
            else None
        )
        if unreadable is not None:
            # A pending release, and the attempts cannot be listed: a launch without
            # --new-attempt would be refused by run.py (needs_decision) and the person's
            # release wasted; with it, it could reroll a new attempt that exists. Try
            # again shortly, and hand it to a person (the release kept) when it stays
            # so. The same for a --new-attempt-cells request: never a blind
            # --new-attempt.
            deferrals = self._release_deferrals.get(spec.cell_id, 0) + 1
            self._release_deferrals[spec.cell_id] = deferrals
            self._skips.pop(spec.cell_id, None)
            if (
                self._resource_wait is not None
                and self._resource_wait[0] == spec.cell_id
            ):
                # It leaves the queue: no resource wait is its any more (reporting).
                self._resource_wait = None
                self._resource_wait_info = None
            if deferrals >= RELEASE_UNREADABLE_MAX_DEFERRALS:
                try:
                    record = await asyncio.to_thread(read_status, paths) or {}
                except OSError:
                    record = {}
                self._attend(spec.cell_id, record.get("status"), unreadable)
                await self.event(
                    "launch_deferred",
                    cell=spec.cell_id,
                    reason=unreadable,
                    deferrals=deferrals,
                    attention=True,
                )
                return False
            self.waiting[spec.cell_id] = self.clock() + RELEASE_UNREADABLE_RETRY_S
            await self.event(
                "launch_deferred",
                cell=spec.cell_id,
                reason=unreadable,
                deferrals=deferrals,
                retry_s=RELEASE_UNREADABLE_RETRY_S,
            )
            return False
        self._release_deferrals.pop(spec.cell_id, None)
        new_attempt = released is not None or note is not None
        gateway = await self._worker_gateway(spec, paths, new_attempt=new_attempt)
        if gateway is None:
            return False  # sent to a person: its recorded gateway cannot be supplied
        argv = worker_argv(
            spec,
            self.options,
            gateway.base_url,
            gateway.runs_dir,
            new_attempt=new_attempt,
            note=note,
        )
        if self.options.auto_resume:
            await asyncio.to_thread(
                self.tracker.note_launch, spec.cell_id, progress, self.clock()
            )

        def spawn() -> subprocess.Popen[bytes]:
            paths.root.mkdir(parents=True, exist_ok=True)
            with paths.worker_log.open("ab") as log_fh:
                return subprocess.Popen(
                    argv,
                    cwd=protocol.REPO_ROOT,
                    env=worker_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # a scheduler restart never stops a cell
                )

        popen = await asyncio.to_thread(spawn)
        self.active[spec.cell_id] = Worker(
            spec, paths, popen.pid, popen, gateway_port=gateway.extra_port
        )
        self.result.launched.append(spec.cell_id)
        await self.event("launched", cell=spec.cell_id, worker=popen.pid, argv=argv)
        return True

    async def _worker_gateway(
        self, spec: CellSpec, paths: CellPaths, *, new_attempt: bool
    ) -> SchedulerGateway | None:
        """The gateway this launch's worker uses: the scheduler's, or, for a MIRIX cell
        that resumes an attempt whose agent recorded another gateway, that one on
        its recorded port (:meth:`_gateway_at`). None when it cannot be supplied:
        the cell goes to a person (``gateway_port_unavailable``) instead of a launch
        run.py would refuse. A launch with ``--new-attempt`` sets up a new agent: any
        gateway."""
        assert self.gateway is not None
        if spec.method != MIRIX_METHOD or new_attempt:
            return self.gateway
        recorded = await asyncio.to_thread(mirix_gateway_url, paths)
        if recorded is None or same_gateway(recorded, self.gateway.base_url):
            return self.gateway
        try:
            return await self._gateway_at(recorded, spec.cell_id)
        except ScheduleError as exc:
            try:
                record = await asyncio.to_thread(read_status, paths) or {}
            except OSError:
                record = {}
            self._attend(spec.cell_id, record.get("status"), REASON_GATEWAY_PORT)
            await self.event(
                "launch_refused_gateway",
                cell=spec.cell_id,
                recorded=recorded,
                error=str(exc),
            )
            return None

    async def _gateway_at(self, url: str, cell_id: str) -> SchedulerGateway:
        """The gateway at ``url`` (a MIRIX attempt's recorded one) as an extra gateway of
        this scheduler: reused while healthy; else attached, adopted or launched on its
        port (:func:`gateway_on_port`). A scheduler that only attaches launches nothing.
        :class:`ScheduleError` when it cannot be had."""
        assert self.gateway is not None
        try:
            parsed = httpx.URL(url)
            port = parsed.port
            host = parsed.host.lower()
        except Exception as exc:
            raise ScheduleError(
                f"the recorded gateway URL {url!r} is unreadable"
            ) from exc
        if port is None or host not in _LOOPBACK_HOSTS:
            raise ScheduleError(
                f"the recorded gateway {url} is not a loopback gateway with a port: "
                "this scheduler can only supply gateways on this host"
            )
        existing = self.extra_gateways.get(port)
        if existing is not None:
            health = await _healthy(existing.base_url)
            if health is None:
                raise ScheduleError(
                    f"the gateway on port {port} (mode {existing.mode}) is not healthy"
                )
            _check_budget(self.options, health, existing.base_url)
            return existing
        if self.gateway.mode == "attach":
            raise ScheduleError(
                f"this scheduler attaches to {self.gateway.base_url} and launches no "
                f"gateway: run a scheduler on port {port} (--gateway-port {port}) or "
                f"attach one to {url}"
            )
        gateway = await gateway_on_port(self.options, port)
        self.extra_gateways[port] = gateway
        await self.event(
            "gateway_extra",
            port=port,
            mode=gateway.mode,
            cell=cell_id,
            url=gateway.base_url,
            pid=gateway.pid,
        )
        return gateway

    def _cells_recording_port(self, port: int) -> list[str]:
        """The matrix's MIRIX cells whose resumed attempt recorded a loopback gateway on
        ``port`` (:func:`mirix_gateway_url`). Blocking."""
        return [
            spec.cell_id
            for spec in self.cells
            if spec.method == MIRIX_METHOD and self._recorded_port(spec) == port
        ]

    def _recorded_port(self, spec: CellSpec) -> int | None:
        """The loopback gateway port ``spec``'s resumed MIRIX attempt recorded (None for
        no such attempt, another host or an unreadable URL). Blocking."""
        paths = CellPaths(self.options.cell_dir(spec))
        return loopback_port(mirix_gateway_url(paths))

    async def _announce_extra_gateways(self) -> None:
        """One ``gateway_extra`` event per gateway adopted at the start with the
        matrix's cells that recorded its port; an adopted one that failed its checks
        (:func:`adopt_live_gateways`) carries its ``problem``, logged as a warning."""
        for port, extra in sorted(self.extra_gateways.items()):
            cells = await asyncio.to_thread(self._cells_recording_port, port)
            if extra.problem:
                log.warning(
                    "adopted the gateway on port %s (cells %s) although %s",
                    port,
                    cells,
                    extra.problem,
                )
            await self.event(
                "gateway_extra",
                port=port,
                mode=extra.mode,
                cell=None,
                url=extra.base_url,
                pid=extra.pid,
                cells=cells,
                **({"problem": extra.problem} if extra.problem else {}),
            )

    def _ports_needed(self) -> set[int]:
        """The extra gateways' ports this scheduler's cells still need: the port each
        running worker was given, and the recorded gateway port of every queued, waiting
        or running MIRIX cell (its next launch resumes on it). Blocking."""
        ports = {
            port
            for worker in self.active.values()
            if (port := getattr(worker, "gateway_port", None)) is not None
        }
        specs = [
            *self.queue,
            *(self._specs[c] for c in self.waiting if c in self._specs),
            *(w.spec for w in self.active.values()),
        ]
        for spec in specs:
            if spec.method == MIRIX_METHOD:
                port = self._recorded_port(spec)
                if port is not None:
                    ports.add(port)
        return ports

    async def _close_idle_extra_gateways(self) -> None:
        """Close every extra gateway nothing needs any more, on the first poll
        after its last user, not at the scheduler's end (else it stays up, its port
        taken, as long as the scheduler runs): no worker or queued, waiting or running
        MIRIX cell of this scheduler needs its port (:meth:`_ports_needed`) and no live worker
        uses it (:meth:`SchedulerGateway.in_use`: its users lock, a live cell recording
        its port). A MIRIX cell that needs it later gets it again (:meth:`_gateway_at`).
        Event ``gateway_extra_closed``."""
        if not self.extra_gateways:
            return
        needed = await asyncio.to_thread(self._ports_needed)
        for port, gateway in sorted(self.extra_gateways.items()):
            if port in needed or await asyncio.to_thread(gateway.in_use):
                continue
            del self.extra_gateways[port]
            try:
                await gateway.close()
            except Exception as exc:  # forgotten all the same; the next one adopts it
                log.warning("cannot close the gateway %s: %s", gateway.base_url, exc)
            await self.event(
                "gateway_extra_closed",
                port=port,
                mode=gateway.mode,
                url=gateway.base_url,
                pid=gateway.pid,
            )

    async def _close_gateways(self) -> None:
        """At the scheduler's end: the main and every extra gateway are closed, or only
        released while workers go on without us (the next scheduler adopts them)."""
        assert self.gateway is not None
        gateways = [*self.extra_gateways.values(), self.gateway]
        for gateway in gateways:
            if self.active:
                gateway.release()
            else:
                try:
                    await gateway.close()
                except Exception as exc:  # the others are still closed
                    log.warning(
                        "cannot close the gateway %s: %s", gateway.base_url, exc
                    )

    async def _adopt(self, spec: CellSpec, paths: CellPaths, why: str) -> None:
        worker = await asyncio.to_thread(adopted_worker, spec, paths)
        self.active[spec.cell_id] = worker
        self.result.adopted.append(spec.cell_id)
        await self.event("adopted", cell=spec.cell_id, worker=worker.pid, reason=why)

    async def _release_harness_stops(self) -> None:
        """``--rerun-harness-failures``: release, for one new attempt each, the HARNESS
        stops of the selected (or named) cells as they are now, each with the attempt it
        stopped (the status's ``attempt``, else the cell's latest); a stop
        that happens later is not covered."""
        wanted = set(self.options.rerun_harness_cells)
        released: dict[str, str] = {}
        for spec in self.cells:
            if wanted and spec.cell_id not in wanted:
                continue
            paths = CellPaths(self.options.cell_dir(spec))

            def stop(paths: CellPaths = paths) -> tuple[str, Any] | None:
                if cell_is_live(paths):
                    return None
                record = read_status(paths) or {}
                if not _harness_learning_failure(paths, record):
                    return None
                return status_marker(record), record.get("attempt")

            found = await asyncio.to_thread(stop)
            if found is None:
                continue
            marker, attempt = found
            progress = await self._progress(paths)
            if not _is_count(attempt):
                attempt, _state = await self._latest_attempt(paths, progress)
            # Unreadable (-1): released without its attempt, which harness_release
            # takes from the latest while the status is still this stop (the person's
            # release is never dropped).
            stopped = attempt if attempt >= 0 else None
            run_id = (
                await asyncio.to_thread(attempt_run_id, paths, stopped)
                if stopped
                else None
            )
            if not await asyncio.to_thread(
                self.tracker.approve_harness,
                spec.cell_id,
                marker,
                self.clock(),
                attempt=stopped,
                run_id=run_id,
            ):
                continue
            # A released stop whose new attempt exists already (a scheduler launched it,
            # then died) is used up here, not released again.
            if await self._harness_release(spec, paths, progress) is not None:
                released[spec.cell_id] = marker
        await self.event("harness_stops_released", cells=sorted(released))

    async def _harness_release(
        self, spec: CellSpec, paths: CellPaths, progress: Any = None
    ) -> str | None:
        """The released HARNESS stop whose new attempt the cell's next launch still has
        to start (that launch carries ``--new-attempt``), or None. The release is used up
        only once that new attempt exists (the cell's latest attempt is past the stopped
        one; :meth:`ResumeTracker.harness_release`), not when a worker is spawned: an
        approved worker that ends before run.py's plan step reaches the stopped attempt
        (the gateway or Postgres down at its start, a budget pause, a drain or emergency
        SIGTERM, a death before its first status) leaves it for the next launch, which
        run.py would otherwise refuse (``needs_decision``); and a new attempt that
        exists is resumed, never replaced by a third. ``progress``: the
        cell's ``cell_progress`` if read already.

        Whatever the decision needs and cannot read (the attempts, the status, the
        stopped attempt's run or state) decides nothing: the cell is marked unreadable,
        its launch deferred and the release kept. Any other OSError on the way
        (resume_state.json not saved on a full disk, after the decision was taken in
        memory) defers this one launch too; the next acts on that decision."""
        cell_id = spec.cell_id
        self._release_unreadable.discard(cell_id)
        if self.tracker.harness_pending(cell_id) is None:
            return None
        try:
            return await self._harness_release_read(cell_id, paths, progress)
        except OSError as exc:
            entry = self.tracker.cells.get(cell_id) or {}
            await self._release_cannot_read(
                cell_id, entry.get("harness_attempt"), error=f"{exc}"
            )
            return None

    async def _release_cannot_read(
        self, cell_id: str, stopped: Any, **why: Any
    ) -> None:
        """Mark ``cell_id``'s pending release unreadable: its launch is deferred."""
        self._release_unreadable.add(cell_id)
        await self.event(
            "harness_release_unreadable", cell=cell_id, stopped_attempt=stopped, **why
        )

    async def _harness_release_read(
        self, cell_id: str, paths: CellPaths, progress: Any
    ) -> str | None:
        """:meth:`_harness_release` for a cell with a pending release."""
        pending = self.tracker.harness_pending(cell_id)
        if pending is None:
            return None
        latest, state = await self._latest_attempt(paths, progress)
        raw = await asyncio.to_thread(read_status, paths)
        record = raw or {}
        entry = self.tracker.cells.get(cell_id) or {}
        stopped = entry.get("harness_attempt")
        if latest < 0:
            # The attempts could not be listed: whether the new attempt exists is
            # unknown, and --new-attempt would invalidate it if it did. No launch now
            # (launch defers it); the release waits for a reading.
            await self._release_cannot_read(cell_id, stopped)
            return None
        if not _is_count(stopped) and latest > 0 and raw is None:
            # Released without its attempt, which is taken from the latest while the
            # status is still that stop: a status that cannot be read would use the
            # release up (unknown) and launch without --new-attempt.
            await self._release_cannot_read(cell_id, stopped, status_unreadable=True)
            return None
        if not _is_count(stopped) and latest == 0:
            # Released without its attempt (it could not be read), and no attempt is
            # there now: the directory is moved aside. Kept for its return, never used
            # up on a status that is not there.
            return None
        if _is_count(stopped) and latest < stopped:
            # Listed, and below the stopped attempt: the cell's directory was moved out
            # (started over, or aside for a moment). --new-attempt would reroll a fresh
            # cell's attempt: this launch goes without, and the release is kept for the
            # directory's return.
            return None
        wanted = entry.get("harness_run_id")
        if _is_count(stopped) and latest == stopped and wanted:
            current = await asyncio.to_thread(attempt_run_id, paths, latest)
            if current is None:
                # Its run cannot be read: whether this is the stopped attempt or a
                # started-over cell's is unknown. No launch now (launch defers it: one
                # without --new-attempt would be refused, needs_decision, and block the
                # cell); kept.
                await self._release_cannot_read(
                    cell_id, stopped, run_id_unreadable=True
                )
                return None
            if current != wanted:
                # An attempt of the stopped number, of another run: the cell was
                # started over and learned that far again. The stop is gone with the
                # old directory: used up, never a reroll of the new cell.
                await asyncio.to_thread(
                    self.tracker.note_harness_relaunch,
                    cell_id,
                    pending,
                    self.clock(),
                    attempt=latest,
                    reason="reset",
                )
                await self.event(
                    "harness_release_used",
                    cell=cell_id,
                    attempt=latest,
                    stopped_attempt=stopped,
                    reason="reset",
                )
                return None
        latest_run_id = (
            await asyncio.to_thread(attempt_run_id, paths, latest)
            if not _is_count(stopped) and latest > 0
            else None
        )
        if not _is_count(stopped) and latest > 0 and latest_run_id is None:
            # Recording the stopped attempt without its run would lose the check that
            # tells a started-over cell apart, for good.
            await self._release_cannot_read(cell_id, stopped, run_id_unreadable=True)
            return None
        if _is_count(stopped) and latest == stopped and latest > 0 and state is None:
            # The stopped attempt's state cannot be read: whether a person resumed it
            # (frozen, testing) is unknown, and --new-attempt would throw that away.
            await self._release_cannot_read(cell_id, stopped, state_unreadable=True)
            return None
        hand_resumed = (
            _is_count(stopped)
            and latest == stopped
            and await asyncio.to_thread(
                operator_resumed_since, paths, latest, entry.get("harness_approved_at")
            )
        )
        released, used = await asyncio.to_thread(
            self.tracker.harness_release,
            cell_id,
            latest=latest,
            state=state,
            marker=status_marker(record),
            at=self.clock(),
            hand_resumed=hand_resumed,
            latest_run_id=latest_run_id,
        )
        if used is not None:
            await self.event(
                "harness_release_used",
                cell=cell_id,
                attempt=latest,
                stopped_attempt=(self.tracker.cells.get(cell_id) or {}).get(
                    "harness_attempt"
                ),
                reason=used,
            )
        return released

    # -- --new-attempt-cells ------------------------------------------------------

    async def _request_new_attempts(self) -> None:
        """``--new-attempt-cells``: record, for each named cell, one new attempt that
        replaces its latest attempt now, with ``--new-attempt-note`` and the estimate
        (what the replaced attempt cost, ``costs.json``). The same note never reruns a
        cell twice (a restarted scheduler with the same command line requests nothing);
        a cell that never ran needs nothing (its first attempt starts anyway). Reruns
        spend from the run's ``--budget`` like any other launch: the estimate is
        recorded and sent (the supervisor's notice), then they run."""
        wanted = set(self.options.new_attempt_cells)
        note = self.options.new_attempt_note
        if not wanted or note is None:
            return
        key = rerun_key(note)
        requested: list[str] = []
        already: list[str] = []
        estimates: dict[str, float | None] = {}
        for spec in self.cells:
            if spec.cell_id not in wanted:
                continue
            paths = CellPaths(self.options.cell_dir(spec))
            latest, _state = await asyncio.to_thread(latest_attempt, paths)
            if latest == 0:
                await self.event("new_attempt_not_needed", cell=spec.cell_id, note=note)
                continue
            # Unreadable (-1): recorded without its attempt, which the launch takes
            # from the latest once it can read it (never a blind --new-attempt).
            known = latest if latest > 0 else None
            run_id = (
                await asyncio.to_thread(attempt_run_id, paths, known) if known else None
            )
            estimate = (
                await asyncio.to_thread(attempt_cost_usd, paths, known)
                if known
                else None
            )
            if await asyncio.to_thread(
                self.tracker.approve_rerun,
                spec.cell_id,
                key=key,
                note=note,
                from_attempt=known,
                from_run_id=run_id,
                estimate_usd=estimate,
                at=self.clock(),
            ):
                requested.append(spec.cell_id)
                estimates[spec.cell_id] = estimate
            else:
                already.append(spec.cell_id)
        known_costs = [v for v in estimates.values() if v is not None]
        await self.event(
            "new_attempts_requested",
            cells=requested,
            note=note,
            key=key,
            estimates=estimates,
            estimate_total_usd=round(sum(known_costs), 6) if known_costs else None,
            # This note was requested for them before: never a second rerun.
            already_requested=already,
        )

    async def _rerun_release(self, spec: CellSpec, paths: CellPaths) -> str | None:
        """The note the cell's next launch carries with ``--new-attempt`` (a pending
        ``--new-attempt-cells`` request whose new attempt does not exist yet), or None.
        The request is used up once the cell's latest attempt is past the one it
        replaces (``new_attempt_started``); an attempt of that number from another run
        (the cell started over) uses it up without a rerun (``reset``). Attempts or a
        run id that cannot be read decide nothing: the launch is deferred and the
        request kept (like a HARNESS release)."""
        cell_id = spec.cell_id
        self._rerun_unreadable.discard(cell_id)
        pending = self.tracker.rerun_pending(cell_id)
        if pending is None:
            return None
        key, request = pending
        try:
            latest, _state = await asyncio.to_thread(latest_attempt, paths)
            if latest < 0:
                raise OSError("the attempts cannot be listed")
            replaced = request.get("from_attempt")
            if not _is_count(replaced):
                if latest == 0:
                    return None  # the directory is away: kept for its return
                run_id = await asyncio.to_thread(attempt_run_id, paths, latest)
                if run_id is None:
                    raise OSError(f"attempt {latest}'s run id cannot be read")
                await asyncio.to_thread(
                    self.tracker.set_rerun_from, cell_id, key, latest, run_id
                )
                replaced = latest
                request["from_run_id"] = run_id
            if latest > replaced:
                await asyncio.to_thread(
                    self.tracker.note_rerun_started, cell_id, key, latest, self.clock()
                )
                await self.event("new_attempt_started", cell=cell_id, attempt=latest)
                return None
            if latest < replaced:
                return None  # moved out or started over: kept, never a reroll
            wanted = request.get("from_run_id")
            if wanted:
                current = await asyncio.to_thread(attempt_run_id, paths, latest)
                if current is None:
                    raise OSError(f"attempt {latest}'s run id cannot be read")
                if current != wanted:
                    await asyncio.to_thread(
                        self.tracker.note_rerun_started,
                        cell_id,
                        key,
                        None,
                        self.clock(),
                        state="reset",
                    )
                    await self.event(
                        "new_attempt_reset",
                        cell=cell_id,
                        attempt=latest,
                        run_id=current,
                    )
                    return None
        except OSError as exc:
            self._rerun_unreadable.add(cell_id)
            await self.event("new_attempt_unreadable", cell=cell_id, error=str(exc))
            return None
        return str(request["note"])

    async def _latest_attempt(
        self, paths: CellPaths, progress: Any = None
    ) -> tuple[int, Any]:
        """The cell's latest attempt and its state, from ``progress`` (``cell_progress``,
        read here unless given), else read directly (:func:`latest_attempt`)."""
        if progress is None:
            progress = await self._progress(paths)
        if isinstance(progress, Mapping) and _is_count(progress.get("attempt")):
            return _attempt_of(progress), progress.get("state")
        # No reading, or no attempt in it (cell_progress also says None when it cannot
        # list the attempts): read them directly.
        return await asyncio.to_thread(latest_attempt, paths)

    async def _classify_one(self, spec: CellSpec, **place: Any) -> None:
        paths = CellPaths(self.options.cell_dir(spec))
        # A released HARNESS stop whose new attempt does not exist yet; _place checks
        # it is this very stop.
        released = await self._harness_release(spec, paths)
        kind, status = await asyncio.to_thread(
            classify_cell,
            paths,
            rerun_interrupted=self.options.rerun_interrupted,
            # A release whose attempts cannot be read now is still the person's: queued,
            # and its launch deferred, not sent back as harness_failure.
            rerun_harness_failures=released is not None
            or spec.cell_id in self._release_unreadable,
            auto_resume=self.options.auto_resume,
        )
        if kind == LIVE:
            await self._adopt(spec, paths, "live")
        else:
            await self._place(spec, kind, status, **place)

    async def reap(self) -> None:
        for cell_id, worker in list(self.active.items()):
            code = await asyncio.to_thread(worker.poll)
            if code is None:
                continue
            del self.active[cell_id]
            stopped_by = self._stopping.pop(cell_id, None)
            self._emergency_signalled.pop(cell_id, None)
            self._emergency_killed.pop(cell_id, None)
            if stopped_by == STOPPED_MEMORY_EMERGENCY:
                self._emergency_settle_until = time.monotonic() + EMERGENCY_SETTLE_S
                # The host's other schedulers settle from the victim's exit too (its
                # memory is freed only now), as this scheduler does.
                self._host_emergency_at = time.time()
            swept: list[int] = []
            # A worker that died without cleaning up (the scheduler's SIGKILL of a hung
            # emergency victim; the OOM killer or any other signal) left its MIRIX server
            # (a group of its own) running, holding memory until the cell's next run.py,
            # a resume backoff later. Swept only under the cell lock (sweep_dead_cell): a
            # busy exit, or a run.py a person started, holds it, and its MIRIX server is
            # alive and in use. A cleanup that fails is left to the cell's next run.py.
            try:
                swept = await asyncio.to_thread(sweep_dead_cell, worker.paths)
            except Exception as exc:
                log.warning("cannot clean up after %s: %s", cell_id, exc)
            status = (await asyncio.to_thread(read_status, worker.paths)) or {}
            # A launched worker's statuses carry its pid: any other status is an
            # earlier run's, left as it was by a worker that died before its first
            # status (an import error, a kill at its start). Checked before the status
            # is read as this worker's ending: a stale budget_paused would put the cell
            # back at the front of the queue on every poll, without backoff or cap.
            own = worker.popen is None or status.get("pid") == worker.pid
            name = status.get("status") if own else None
            await self.event(
                "exited",
                cell=cell_id,
                worker=worker.pid,
                returncode=code,
                status=name,
                detail=status.get("detail") if own else None,
                adopted=worker.adopted,
                **({"stopped_by": stopped_by} if stopped_by else {}),
                **({} if own else {"stale_status": status.get("status")}),
                **({"swept": swept} if swept else {}),
            )
            # A released HARNESS stop whose new attempt now exists is used up;
            # a --new-attempt-cells request too.
            await self._harness_release(worker.spec, worker.paths)
            await self._rerun_release(worker.spec, worker.paths)
            if worker.popen is not None and code == EXIT_CODES[STATUS_BUSY]:
                # Another live harness holds the cell (run.py wrote no status): that
                # harness's run is the cell's; adopt it rather than start one more.
                if cell_id in self.result.launched:
                    self.result.launched.remove(cell_id)
                await self._classify_one(worker.spec)
            elif code == EXIT_CODES[STATUS_COMPLETE] and (
                status.get("status") == STATUS_COMPLETE
            ):
                # Complete, whoever wrote it: a complete cell is never launched, so this
                # status cannot be a stale one.
                self.result.complete.append(cell_id)
            elif not own:
                # Its ending is this worker's death, counted like any death (backoff and
                # caps), with a marker that stays the same for this ending, so a
                # restarted scheduler that reads the stale status again does not count
                # it again.
                if self.options.auto_resume:
                    await self._schedule_resume(
                        worker.spec,
                        worker.paths,
                        STATUS_RUNNING,
                        exit_code=code,
                        stopped_by=stopped_by,
                        marker=died_marker(status, worker.pid, code),
                    )
                else:
                    self._attend(cell_id, f"exit {code}", REASON_NOT_RERUN)
            elif name == STATUS_BUDGET_PAUSED:
                # Nothing ran (run.py found the gateway paused at its start): the cell
                # waits in the queue and is launched again once the balance recovers
                # ("pause, then continue"; not a re-run of anything, H8). The next poll
                # asks the gateway for a fresh balance first.
                self.queue.insert(0, worker.spec)
                if cell_id in self.result.launched:
                    self.result.launched.remove(cell_id)
                self._balance_at = float("-inf")
            elif name in BLOCKED_STATUSES:
                self.result.blocked[cell_id] = name
            elif self.options.auto_resume:
                await self._classify_one(
                    worker.spec, exit_code=code, stopped_by=stopped_by
                )
            else:
                self._attend(cell_id, name or f"exit {code}", REASON_NOT_RERUN)

    # -- resource guard ------------------------------------------------------------------

    def _worker_groups(self) -> dict[str, list[int]]:
        """Each active cell's process groups: its worker's and its MIRIX server's (blocking)."""
        groups = {}
        for cell_id, worker in self.active.items():
            pgids = {worker.process_group()}
            pgids.update(
                registered_process_groups(worker.paths.root / "owned_processes.jsonl")
            )
            groups[cell_id] = sorted(p for p in pgids if p > 0)
        return groups

    async def sample_resources(self) -> resources.HostResources:
        groups = await asyncio.to_thread(self._worker_groups)
        reading = await asyncio.to_thread(
            self.sampler, self.options.runs_dir, workers=groups
        )
        self.resources = {
            **reading.to_dict(),
            **({"docker_gb": self._docker} if self._docker is not None else {}),
        }
        return reading

    def _reserved_gb(self, reading: resources.HostResources) -> float:
        return resources.reserved_gb(
            {c: w.spec for c, w in self.active.items()}, reading.workers_rss_gb
        )

    def _benchmark_allows(self, spec: CellSpec) -> bool:
        cap = self.options.benchmark_caps.get(spec.benchmark)
        if cap is None:
            return True
        running = sum(
            1 for w in self.active.values() if w.spec.benchmark == spec.benchmark
        )
        return running < cap

    @property
    def _gated(self) -> bool:
        return (
            self.options.min_free_memory_gb is not None
            or self.options.min_free_disk_gb is not None
        )

    # -- the host's other schedulers (host_registry.py) --------------------------------

    async def _host_unavailable(self, exc: BaseException) -> None:
        """The host directory cannot be used: this scheduler goes on with its own view
        (record, don't block), and says so once per problem."""
        text = f"{type(exc).__name__}: {exc}"[:300]
        self._host_problem = text
        if text in self._host_problems_seen:
            return
        self._host_problems_seen.add(text)
        log.warning(
            "host directory %s: %s; the memory gate, --host-benchmark-concurrency and "
            "the memory emergency see this scheduler's own cells only",
            self.host.directory,
            text,
        )
        await self.event(
            "host_registry_unavailable", directory=str(self.host.directory), error=text
        )

    async def _host_lock(self) -> OwnerLock | None:
        """The host lock, or None when it cannot be had (recorded)."""
        try:
            return await asyncio.to_thread(self.host.lock)
        except (OSError, OwnedProcessError) as exc:
            await self._host_unavailable(exc)
            return None

    async def _read_peers(self, *, prune: bool) -> list[host_registry.Peer]:
        try:
            peers = await asyncio.to_thread(self.host.peers, prune=prune)
        except OSError as exc:
            await self._host_unavailable(exc)
            return []
        self._host_peers = peers
        return peers

    def _host_workers(self) -> list[dict[str, Any]]:
        """This scheduler's cells for its host record: the running workers with the
        resident memory last read, and the admitted cells whose launch is under way."""
        rss = (self.resources or {}).get("workers_rss_gb") or {}
        rows: list[dict[str, Any]] = []
        for cell_id, worker in sorted(self.active.items()):
            spec = worker.spec
            value = rss.get(cell_id) if isinstance(rss, Mapping) else None
            rows.append(
                {
                    "cell": cell_id,
                    "benchmark": spec.benchmark,
                    "method": spec.method,
                    "estimate_gb": resources.estimate_gb(spec.benchmark, spec.method),
                    "rss_gb": value
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else None,
                    "pending": False,
                }
            )
        for cell_id, spec in sorted(self._host_pending.items()):
            if cell_id in self.active:
                continue
            rows.append(
                {
                    "cell": cell_id,
                    "benchmark": spec.benchmark,
                    "method": spec.method,
                    "estimate_gb": resources.estimate_gb(spec.benchmark, spec.method),
                    "rss_gb": None,
                    "pending": True,
                }
            )
        return rows

    def _emergency_pending(self) -> bool:
        return any(
            why == STOPPED_MEMORY_EMERGENCY and c not in self._emergency_killed
            for c, why in self._stopping.items()
        )

    def _host_running(self, benchmark: str) -> int:
        """This scheduler's cells of ``benchmark`` that run or are admitted."""
        return sum(
            1 for w in self.active.values() if w.spec.benchmark == benchmark
        ) + sum(
            1
            for c, s in self._host_pending.items()
            if s.benchmark == benchmark and c not in self.active
        )

    def _host_waiting(self) -> dict[str, float]:
        """The benchmarks whose queued cells the host cap holds back while this
        scheduler runs none of their cells and has room for one, with since when (its
        record's ``waiting``; every peer keeps a slot free for it, first come first
        served). Updates the times kept."""
        queued = {c.cell_id: c.benchmark for c in self.queue}
        room = len(self.active) < self.options.concurrency
        waiting = {
            queued[cell]
            for cell in self._host_held
            if room and cell in queued and self._host_running(queued[cell]) == 0
        }
        now_s = time.time()
        self._host_waiting_since = {
            b: self._host_waiting_since.get(b, now_s) for b in sorted(waiting)
        }
        return dict(self._host_waiting_since)

    async def _publish_host(self) -> None:
        """Write this scheduler's host record (only once it knows its start time, which
        its peers check it by, and has adopted the workers it found)."""
        if not self.started or not self._host_ready:
            return
        try:
            await asyncio.to_thread(
                self.host.publish,
                pid=os.getpid(),
                started=self.started,
                workers=self._host_workers(),
                test_concurrency=self.options.test_concurrency,
                emergency_at=self._host_emergency_at,
                emergency_pending=self._emergency_pending(),
                host_benchmark_caps=self.options.host_benchmark_caps,
                waiting=self._host_waiting(),
            )
        except OSError as exc:
            await self._host_unavailable(exc)

    def _host_state(self) -> dict[str, Any]:
        """``state.json`` ``host``: the directory, the host-wide caps, the peers last
        read and the cells the host cap holds."""
        caps = self.options.host_benchmark_caps
        benchmarks = set(caps) | {
            b for p in self._host_peers for b in (p.host_benchmark_caps or {})
        }
        in_force = {
            b: host_registry.host_cap(caps.get(b), self._host_peers, b)[0]
            for b in sorted(benchmarks)
        }
        return {
            "directory": str(self.host.directory),
            "benchmark_concurrency": caps,
            # The lowest cap any live scheduler of the host publishes.
            "benchmark_concurrency_in_force": in_force,
            "peers": [p.summary() for p in self._host_peers],
            "held": dict(sorted(self._host_held.items())),
            "waiting": self._host_waiting(),
            "problem": self._host_problem,
        }

    def _host_cap_problem(
        self, spec: CellSpec, peers: Sequence[host_registry.Peer]
    ) -> str | None:
        """Why the host-wide cap holds ``spec`` back, or None. The cap in force is the
        lowest any live scheduler of the host publishes (``host_registry.host_cap``),
        and a scheduler that runs some of the benchmark's cells keeps one slot free for
        each peer the cap holds back while it runs none
        (``host_registry.starved_peers``)."""
        benchmark = spec.benchmark
        own = self.options.host_benchmark_caps.get(benchmark)
        cap, _other = host_registry.host_cap(own, peers, benchmark)
        if cap is None:
            return None
        here = self._host_running(benchmark)
        there = host_registry.peers_cells(peers, benchmark)
        since = self._host_waiting_since.get(benchmark) if here == 0 else None
        starved = host_registry.starved_peers(
            peers,
            benchmark,
            before=None if since is None else (since, str(self.options.runs_dir)),
        )
        if here + there + len(starved) < cap:
            return None
        lower = (
            f" (the lowest of this host's schedulers; this scheduler's own is {own})"
            if own != cap
            else ""
        )
        kept = (
            f"; {len(starved)} slot(s) kept for scheduler(s) waiting with none: "
            + ", ".join(p.runs_dir for p in starved)
            if starved
            else ""
        )
        return (
            f"host: {here + there} {benchmark} cells run on this host (this "
            f"scheduler {here}, {len(peers)} other scheduler(s) {there}) >= "
            f"--host-benchmark-concurrency {benchmark}={cap}{lower}{kept}"
        )

    async def _note_host_cap_mismatch(
        self, peers: Sequence[host_registry.Peer]
    ) -> None:
        """A peer that publishes another host cap than this scheduler's own: recorded
        once per peer, benchmark and pair of values (``host_cap_mismatch``, a warning);
        the lowest is enforced."""
        own = self.options.host_benchmark_caps
        for peer in peers:
            theirs = peer.host_benchmark_caps
            if theirs is None:
                continue
            for benchmark in sorted(set(own) | set(theirs)):
                ours, other = own.get(benchmark), theirs.get(benchmark)
                if ours == other:
                    continue
                key = (peer.runs_dir, benchmark, other, ours)
                if key in self._host_cap_mismatches:
                    continue
                self._host_cap_mismatches.add(key)
                in_force = min(c for c in (ours, other) if c is not None)
                log.warning(
                    "host cap of %s: %s=%s, this scheduler's %s=%s; every scheduler "
                    "of the host enforces the lowest (%d): pass the same "
                    "--host-benchmark-concurrency to all of them",
                    peer.runs_dir,
                    benchmark,
                    other,
                    benchmark,
                    ours,
                    in_force,
                )
                await self.event(
                    "host_cap_mismatch",
                    peer=peer.runs_dir,
                    benchmark=benchmark,
                    theirs=other,
                    ours=ours,
                    in_force=in_force,
                )

    async def _admit(self, spec: CellSpec) -> tuple[bool, str]:
        """May ``spec`` start now? Under the host lock: the host-wide benchmark cap, then
        (with a gate) the memory and disk gate with the other schedulers' unreserved
        headroom subtracted too; an admitted cell enters this scheduler's host record
        before the lock is let go, so no peer admits on the memory it will take."""
        lock = await self._host_lock()
        try:
            peers = await self._read_peers(prune=lock is not None)
            await self._note_host_cap_mismatch(peers)
            problem = self._host_cap_problem(spec, peers)
            if problem is not None:
                return False, problem
            if self._gated:
                reading = await self.sample_resources()
                theirs = host_registry.peers_reserved_gb(peers)
                admitted, why = resources.admit(
                    reading,
                    spec,
                    min_free_memory_gb=self.options.min_free_memory_gb,
                    min_free_disk_gb=self.options.min_free_disk_gb,
                    reserved_gb=self._reserved_gb(reading) + theirs,
                )
                if theirs > 0:
                    why += (
                        f" (reserved includes {theirs:.2f} GB for {len(peers)} other "
                        "scheduler(s) on this host)"
                    )
            else:
                admitted, why = True, "no gate"
            if admitted and lock is not None:
                self._host_pending[spec.cell_id] = spec
                await self._publish_host()
            return admitted, why
        finally:
            if lock is not None:
                lock.release()

    async def _launch_ready(self) -> None:
        """Launch queued cells while the concurrency, the per-benchmark caps and the
        resource gate allow, and never once a stop was asked for (a SIGTERM
        that lands while the gateway or the host is read must not start a worker nobody
        drains). A cell a benchmark cap holds is skipped. A cell the memory gate holds
        never holds the queue (there is no skip limit): every cell behind
        it that fits starts (``resource_skip``, with the running count), and the held
        cell, tried first on every poll, starts the first time it is admitted. The disk
        gate holds the whole queue."""
        index = 0
        held: CellSpec | None = None
        while index < len(self.queue) and len(self.active) < self.options.concurrency:
            if self.stop.is_set():
                return
            spec = self.queue[index]
            if not self._benchmark_allows(spec):
                index += 1
                continue
            if spec.benchmark == "webshop":
                # No WebShop cell starts before its server is ready (loading
                # the product set takes minutes; its memory is outside the cell's own).
                ready, why = await self._webshop_gate()
                if self.stop.is_set():
                    return
                await self._note_webshop_failed(None if ready else why)
                if not ready:
                    if held is None:
                        held = spec
                        await self._note_resource_wait(spec, "webshop", why)
                    index += 1  # the other benchmarks' cells go on
                    continue
            admitted, why = await self._admit(spec)
            if self.stop.is_set():
                self._host_pending.pop(spec.cell_id, None)
                return
            if not admitted:
                gate = why.split(":", 1)[0]
                if gate == "host":
                    # The host-wide cap of its benchmark (host_registry.py): skipped
                    # over like a per-benchmark cap, never holding the queue.
                    await self._note_host_hold(spec, why)
                    index += 1
                    continue
                # Past the host cap: another gate holds it, so no peer keeps a slot
                # free for it meanwhile (its record's waiting).
                self._host_held.pop(spec.cell_id, None)
                if held is None:
                    held = spec
                    await self._note_resource_wait(spec, gate, why)
                if gate == "memory":
                    index += 1  # a lighter cell behind it may still fit
                    continue
                return  # the disk gate holds every cell
            self._host_held.pop(spec.cell_id, None)
            self.queue.pop(index)
            try:
                launched = await self.launch(spec)
            finally:
                # Launched: an active worker now; deferred: nothing to reserve.
                self._host_pending.pop(spec.cell_id, None)
            if not launched:
                continue  # deferred: no launch to count
            if held is not None:
                self._skips[held.cell_id] = self._skips.get(held.cell_id, 0) + 1
                await self.event(
                    "resource_skip",
                    held=held.cell_id,
                    launched=spec.cell_id,
                    skips=self._skips[held.cell_id],
                )
            else:
                self._skips.pop(spec.cell_id, None)
                if self._resource_wait is not None:
                    await self.event(
                        "resource_wait_end",
                        cell=self._resource_wait[0],
                        admitted=spec.cell_id,
                    )
                    self._resource_wait = None
                    self._resource_wait_info = None
        if held is None and self._resource_wait is not None and not self.queue:
            self._resource_wait = None
            self._resource_wait_info = None

    async def _webshop_gate(self) -> tuple[bool, str]:
        """The WebShop server's readiness, read at most every
        :data:`WEBSHOP_READY_POLL_S` (a failed probe counts as not ready)."""
        now = time.monotonic()
        reading = self._webshop_reading
        if reading is None or now - reading[0] >= WEBSHOP_READY_POLL_S:
            try:
                ready, why = await self.webshop_ready()
            except Exception as exc:  # noqa: BLE001 - not ready, and why
                ready, why = False, f"{type(exc).__name__}: {exc}"
            reading = (now, bool(ready), str(why))
            self._webshop_reading = reading
        return reading[1], reading[2]

    async def _note_webshop_failed(self, why: str | None) -> None:
        """A server that failed its own checks (a manifest that differs, a load that
        raised) never becomes ready by itself: say so once per reason (event
        ``webshop_server_failed`` and a warning), while the gate holds its cells."""
        failed = (
            why if why is not None and why.startswith(WEBSHOP_FAILED_PREFIX) else None
        )
        if failed is None or failed == self._webshop_failed:
            self._webshop_failed = failed
            return
        self._webshop_failed = failed
        log.warning(
            "%s: WebShop cells wait until it is fixed and restarted "
            "(python -m evals.skill_bench.envs.webshop --stop, --start)",
            failed,
        )
        await self.event("webshop_server_failed", reason=failed)

    async def _note_host_hold(self, spec: CellSpec, why: str) -> None:
        """A cell the host-wide cap holds back: ``state.json`` ``host.held``, and the
        event ``host_cap_wait`` once until it is launched."""
        first = spec.cell_id not in self._host_held
        self._host_held[spec.cell_id] = why
        if first:
            await self.event("host_cap_wait", cell=spec.cell_id, reason=why)

    async def _note_resource_wait(self, spec: CellSpec, gate: str, why: str) -> None:
        held = (spec.cell_id, gate)
        if held == self._resource_wait:
            return
        self._resource_wait = held
        self._resource_wait_info = {
            "cell": spec.cell_id,
            "gate": gate,
            "reason": why,
            "since": now(),
        }
        await self.event("resource_wait", cell=spec.cell_id, reason=why)

    async def _emergency_kill_hung(self) -> None:
        """SIGKILL (its process group) an emergency victim still running
        :data:`EMERGENCY_KILL_AFTER_S` after its ``SIGTERM``: it is hung, and it would
        hold back every later emergency stop for the rest of the run. Once killed, or
        when the kill cannot be delivered (it is not the process it was), it no longer
        holds them back; a delivered kill starts the settle period (its memory is freed
        a moment later), and reap cleans up its MIRIX server."""
        clock = time.monotonic()
        for cell_id, since in list(self._emergency_signalled.items()):
            if cell_id in self._emergency_killed:
                continue
            worker = self.active.get(cell_id)
            if (
                worker is None
                or self._stopping.get(cell_id) != STOPPED_MEMORY_EMERGENCY
            ):
                self._emergency_signalled.pop(cell_id, None)
                continue
            if clock - since < EMERGENCY_KILL_AFTER_S:
                continue
            delivered = await asyncio.to_thread(worker.kill)
            self._emergency_killed[cell_id] = delivered
            if delivered:
                self._emergency_settle_until = time.monotonic() + EMERGENCY_SETTLE_S
                self._host_emergency_at = time.time()
            await self.event(
                "memory_emergency_kill",
                cell=cell_id,
                worker=worker.pid,
                killed=delivered,
                waited_s=round(clock - since, 1),
            )

    async def _emergency_check(self, reading: resources.HostResources) -> None:
        await self._emergency_kill_hung()
        floor = self.options.memory_emergency_gb
        if not resources.emergency(reading, floor):
            self._low_memory_samples = 0
            return
        self._low_memory_samples += 1
        if self._low_memory_samples < EMERGENCY_SAMPLES:
            return
        stopping = [
            c
            for c, why in self._stopping.items()
            if why == STOPPED_MEMORY_EMERGENCY and c not in self._emergency_killed
        ]
        if stopping or time.monotonic() < self._emergency_settle_until:
            # A stopped worker is still capturing its checkpoint, or has only just
            # exited: its memory is not free yet, so stopping one more now would stop
            # them all within a minute. A hung one is killed after
            # EMERGENCY_KILL_AFTER_S and holds nothing back then.
            return
        # The newest of those stopped the fewest times first: one relaunched cell does
        # not take every stop and run into --max-resumes. An adopted worker
        # re-reads process.json first, so one adopted with the previous run's record
        # sorts by its own run's start.
        for cell_id, worker in list(self.active.items()):
            if cell_id not in self._stopping:
                await asyncio.to_thread(worker.refresh)
        candidates = sorted(
            (w for c, w in self.active.items() if c not in self._stopping),
            key=lambda w: (self._emergency_stops.get(w.spec.cell_id, 0), -w.started_at),
        )
        if not candidates:
            return
        # One stop host-wide (host_registry.py): the other schedulers of the host read
        # the same low memory; a peer that stopped a worker less than the settle period
        # ago, or whose victim still runs, has made this stop already. The decision and
        # this scheduler's record of its own stop are taken under the host lock.
        lock = await self._host_lock()
        try:
            if lock is not None:
                peers = await self._read_peers(prune=True)
                busy = host_registry.recent_emergency(
                    peers, now=time.time(), settle_s=EMERGENCY_SETTLE_S
                )
                if busy:
                    # Once per stretch in which these peers hold the stop.
                    marker = tuple(sorted(p.runs_dir for p in busy))
                    if marker != self._host_emergency_seen:
                        self._host_emergency_seen = marker
                        await self.event(
                            "memory_emergency_left_to_peer",
                            peers=[p.summary() for p in busy],
                            mem_available_gb=reading.mem_available_gb,
                            floor_gb=floor,
                        )
                    return
                self._host_emergency_seen = ()
            self._low_memory_samples = 0
            for victim in candidates:
                cell_id = victim.spec.cell_id
                delivered = await asyncio.to_thread(victim.terminate)
                if delivered:
                    self._stopping[cell_id] = STOPPED_MEMORY_EMERGENCY
                    self._emergency_signalled[cell_id] = time.monotonic()
                    self._emergency_stops[cell_id] = (
                        self._emergency_stops.get(cell_id, 0) + 1
                    )
                    self._host_emergency_at = time.time()
                    if lock is not None:
                        await self._publish_host()
                await self.event(
                    STOPPED_MEMORY_EMERGENCY,
                    cell=cell_id,
                    worker=victim.pid,
                    signalled=delivered,
                    mem_available_gb=reading.mem_available_gb,
                    floor_gb=floor,
                    samples=EMERGENCY_SAMPLES,
                )
                if delivered:
                    return
                # Not signalled (an adopted worker whose pid is stale, one that has just
                # exited): the next worker takes the stop, or this one would be picked
                # forever and no other ever stopped.
        finally:
            if lock is not None:
                lock.release()

    async def _docker_stats(self) -> None:
        try:
            found = await asyncio.to_thread(self.docker_sampler)
        except Exception as exc:  # optional reading: never a reason to stop
            log.debug("docker stats failed: %s", exc)
            return
        if found is not None:
            self._docker = {
                "at": now(),
                "containers": found,
                "total_gb": round(sum(found.values()), 3),
            }

    async def _housekeeping(self) -> None:
        """The docker stats, the resources log and the memory emergency check, each when
        due (one host reading per poll at most)."""
        clock = time.monotonic()
        if self._has_containers and clock - self._docker_at >= DOCKER_STATS_INTERVAL_S:
            self._docker_at = clock
            if self._docker_task is None or self._docker_task.done():
                self._docker_task = asyncio.create_task(self._docker_stats())
        log_due = (
            clock - self._resources_logged_at >= self.options.resources_log_interval_s
        )
        watch = self.options.memory_emergency_gb is not None and bool(self.active)
        if not (log_due or watch):
            return
        reading = await self.sample_resources()
        if log_due:
            self._resources_logged_at = clock
            await asyncio.to_thread(
                append_jsonl,
                self.options.control / RESOURCES_LOG_FILE,
                {**(self.resources or {}), "active": sorted(self.active)},
            )
        if watch:
            await self._emergency_check(reading)

    # -- main loop -----------------------------------------------------------------------

    async def classify(self) -> None:
        for spec in self.cells:
            await self._classify_one(spec)

    async def run(self) -> ScheduleResult:
        control = self.options.control
        merged = self.options.runs_dir / MERGED_MARKER
        if await asyncio.to_thread(merged.exists):
            raise ScheduleError(
                f"{self.options.runs_dir} is a merged read-only view of several hosts' "
                f"runs ({MERGED_MARKER}, tools/sync_runs.py): run cells in a host's own "
                "runs directory"
            )
        await asyncio.to_thread(control.mkdir, parents=True, exist_ok=True)
        try:
            lock = await asyncio.to_thread(
                OwnerLock.acquire, control / "scheduler.lock", wait_s=0.0
            )
        except OwnedProcessError as exc:
            raise ScheduleError(
                f"another scheduler is running on {self.options.runs_dir}"
            ) from exc
        try:
            return await self._run_locked()
        finally:
            lock.release()

    async def _run_locked(self) -> ScheduleResult:
        # A selection wider than every earlier launch's is refused
        # before anything starts, unless --widen-selection says why (recorded).
        selection: dict[str, Any] | None = None
        if self.options.guard_selection:
            try:
                selection = await asyncio.to_thread(
                    record_selection,
                    self.options.control,
                    self.cells,
                    self.options.widen_selection,
                    pid=os.getpid(),
                )
            except (OSError, ValueError) as exc:
                raise ScheduleError(str(exc)) from exc
        # Gateways on other ports whose owner died while workers still use them (a
        # MIRIX cell's recorded gateway a previous scheduler launched) are adopted
        # first, so ensure_gateway does not take them for orphans.
        self.extra_gateways = await adopt_live_gateways(self.options)
        try:
            self.gateway = await ensure_gateway(self.options)
        except BaseException:
            for extra in self.extra_gateways.values():
                extra.release()
            raise
        await asyncio.to_thread(self.tracker.load)
        self.started = await asyncio.to_thread(process_start_time, os.getpid())
        await self.event(
            "start",
            gateway=self.gateway.base_url,
            gateway_mode=self.gateway.mode,
            cells=[c.cell_id for c in self.cells],
            concurrency=self.options.concurrency,
            auto_resume=self.options.auto_resume,
            started=self.started,
            host_directory=str(self.host.directory),
            host_benchmark_concurrency=self.options.host_benchmark_caps,
            **(
                {"resume_state_error": self.tracker.load_error}
                if self.tracker.load_error
                else {}
            ),
        )
        host_caps = self.options.host_benchmark_caps
        for benchmark, cap in sorted(self.options.benchmark_caps.items()):
            if benchmark in host_caps and host_caps[benchmark] < cap:
                # Recorded, not refused: the host-wide cap is the lower one in force.
                log.warning(
                    "--benchmark-concurrency %s=%d is above the host-wide cap %s=%d "
                    "(--host-benchmark-concurrency): at most %d %s cells run on this host",
                    benchmark,
                    cap,
                    benchmark,
                    host_caps[benchmark],
                    host_caps[benchmark],
                    benchmark,
                )
        if selection is not None:
            await self.event(f"selection_{selection.pop('kind')}", **selection)
        if self.options.reset_resume is not None:
            wanted = list(self.options.reset_resume) or [c.cell_id for c in self.cells]
            cleared = await asyncio.to_thread(self.tracker.reset, wanted, self.clock())
            await self.event("resume_reset", cells=cleared)
        await self._announce_extra_gateways()
        share = self.gateway.budget_share
        if share is not None:
            # --budget-start-from: one budget start for the replicates.
            await self.event("budget_start_shared", **share)
            if share.get("shared") not in ("copied", "same"):
                log.warning(
                    "--budget-start-from %s: this runs directory's budget record %s "
                    "(%s): its gateway counts from its own start, not the shared one",
                    share.get("from"),
                    share.get("file"),
                    share.get("shared"),
                )
        if self.options.rerun_harness_failures:
            await self._release_harness_stops()
        await self._request_new_attempts()
        # The one-off options are applied: say so at once, so a supervisor that restarts
        # this scheduler stops passing them only now.
        self.one_off_applied = True
        await self.write_state()
        # A scheduler that only attaches to a gateway never touches the gateway registry:
        # that gateway (or another harness's) is not its to stop.
        skip = ("_gateway",) if self.gateway.mode == "attach" else ()
        swept = await asyncio.to_thread(
            sweep_owned, self.options.runs_dir, skip_dirs=skip
        )
        if swept:
            await self.event("sweep", killed=swept)
        try:
            await self.classify()
            # The workers found are adopted: the host record may name them now.
            self._host_ready = True
            await self._publish_host()
            await self._loop()
        finally:
            self.result.queued = [c.cell_id for c in self.queue]
            self.result.running = sorted(self.active)
            self.result.waiting = {
                c: iso(t) for c, t in sorted(self.waiting.items(), key=lambda i: i[1])
            }
            task = self._docker_task
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            # Workers that go on without us keep their gateways (the next scheduler
            # adopts them); else every gateway this scheduler owns is closed.
            await self._close_gateways()
            await self.write_state(final=True)
            # This scheduler no longer admits anything on the host; the workers it
            # leaves running keep their reservation until they exit.
            await self._leave_host()
            await self.event("stop", **self.result.to_dict())
        return self.result

    async def _leave_host(self) -> None:
        """Withdraw this scheduler's host record, or, when workers go on without it (a
        stop, a crash path; the supervisor drains them or the next scheduler adopts
        them), leave an orphaned record of those workers, each with its pid and start
        time: the peers count them by their own liveness (``host_registry``), and
        neither the WebShop host cap nor the memory headroom forgets them meanwhile."""
        if not self._host_ready:
            # It never adopted the workers it found (it stopped before classify): the
            # record there, an orphaned one of a predecessor's workers included, stays
            # as it is.
            return
        orphans: list[dict[str, Any]] = []
        victims = {
            c
            for c, why in self._stopping.items()
            if why == STOPPED_MEMORY_EMERGENCY and c not in self._emergency_killed
        }
        if self.active and self.started:
            rows = {r["cell"]: r for r in self._host_workers() if not r["pending"]}
            for cell_id, worker in sorted(self.active.items()):
                started = await asyncio.to_thread(process_start_time, worker.pid)
                if cell_id in rows and started:
                    orphans.append(
                        {
                            **rows[cell_id],
                            "pid": worker.pid,
                            "started": started,
                            # The peers wait while it captures its checkpoint, and
                            # for the settle period after.
                            "emergency_victim": cell_id in victims,
                        }
                    )
        if not orphans:
            with contextlib.suppress(OSError):  # nothing left to tell the peers
                await asyncio.to_thread(self.host.withdraw)
            return
        try:
            await asyncio.to_thread(
                self.host.publish,
                pid=os.getpid(),
                started=self.started,
                workers=orphans,
                test_concurrency=self.options.test_concurrency,
                emergency_at=self._host_emergency_at,
                emergency_pending=any(o["emergency_victim"] for o in orphans),
                orphaned=True,
            )
        except OSError as exc:
            await self._host_unavailable(exc)
            return
        await self.event("host_record_orphaned", workers=[o["cell"] for o in orphans])

    async def _loop(self) -> None:
        while True:
            await self.reap()
            if self.stop.is_set():
                self.result.stopped = True
                return
            await self._close_idle_extra_gateways()
            promoted = self._promote_due()
            if promoted:
                await self.event("resume_due", cells=promoted)
            await self._housekeeping()
            reason = await self.gateway_pause()
            if reason != self.pause_reason:
                await self.event(
                    "budget_paused" if reason else "budget_resumed",
                    reason=reason,
                    # stop: exit once idle, for a person (budget); wait: auto-resume
                    # waits it out; idle_exit: exit once idle (the supervisor restarts).
                    kind=None if reason is None else self._pause_kind(reason),
                    budget=self.budget,
                )
                self.pause_reason = reason
            if reason is None and not self.stop.is_set():
                await self._launch_ready()
            await self.write_state()
            if not self.queue and not self.waiting and not self.active:
                return
            if (
                reason is not None
                and not self.active
                and not self._waits_through(reason)
            ):
                self.result.paused = True
                self.result.pause_reason = reason
                return
            try:
                await asyncio.wait_for(self.stop.wait(), self.options.poll_s)
            except asyncio.TimeoutError:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def benchmark_cap(value: str) -> tuple[str, int]:
    """``alfworld=2`` -> ``("alfworld", 2)`` (argparse type)."""
    name, sep, number = value.partition("=")
    if not sep or not number.strip().isdigit():
        raise argparse.ArgumentTypeError(f"expected BENCHMARK=N, got {value!r}")
    return name.strip(), int(number)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.schedule",
        description="Run the skill_bench matrix (H1.3, H8) with bounded concurrency.",
        # No abbreviations: the supervisor finds the one-off options by their full
        # names, and an abbreviated one (--reset, --rerun-harness) would reach every
        # restarted scheduler unrecorded.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--study",
        type=study.study_argument,
        help="a named selection of cells (study.STUDIES: "
        + ", ".join(f"{n}: {e.description}" for n, e in study.STUDIES.items())
        + "); --benchmark, --model and --method narrow it, the split mode and sizes "
        "are the study's",
    )
    parser.add_argument("--benchmark", nargs="*", choices=BENCHMARK_ORDER)
    parser.add_argument(
        "--model", nargs="*", choices=[*protocol.MODELS, *protocol.CHAT_MODELS]
    )
    parser.add_argument(
        "--method",
        nargs="*",
        type=study.method_argument,
        choices=METHOD_ORDER,
        help="default: every method; a method the study removed "
        "(study.REMOVED_METHODS) is refused, naming the decision",
    )
    parser.add_argument(
        "--split-mode", choices=SPLIT_MODES, help="default: protocol (or the study's)"
    )
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    parser.add_argument("--gateway-url", help="attach to this running gateway")
    parser.add_argument(
        "--gateway-runs-dir",
        type=Path,
        help="where an attached gateway writes its ledgers (default --runs-dir)",
    )
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--env-file", type=Path, help="key for a launched gateway")
    parser.add_argument(
        "--balance-floor",
        type=float,
        help="USD; launch nothing below it (default: the gateway's floor)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="the USD budget for this runs directory: the "
        "launched gateway pauses once the key's usage has grown by it since a budget was "
        "first set here, and the scheduler then launches nothing more",
    )
    parser.add_argument(
        "--budget-start-from",
        type=Path,
        metavar="RUNS_DIR",
        help="share another runs directory's budget start (replicate runs directories "
        "on one key): before the first budgeted gateway here, its _gateway/budget.json "
        "is copied from RUNS_DIR (refused, and retried by the supervisor, until RUNS_DIR's "
        "gateway has written it), so --budget is the total of every runs directory that "
        "shares the start; a record here already is kept (a different start is recorded)",
    )
    parser.add_argument(
        "--test-concurrency", type=int, default=DEFAULT_TEST_CONCURRENCY
    )
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT_S)
    parser.add_argument(
        "--finalize-timeout", type=float, default=DEFAULT_FINALIZE_TIMEOUT_S
    )
    parser.add_argument("--max-learning-attempts", type=int, default=1)
    parser.add_argument(
        "--rerun-interrupted",
        action="store_true",
        help="continue cells that stopped early (pending test items, an interrupted "
        "learning phase) once; see --auto-resume for doing it by itself. A learning "
        "phase that a HARNESS failure stopped is not relaunched unless "
        "--rerun-harness-failures too",
    )
    parser.add_argument(
        "--rerun-harness-failures",
        nargs="*",
        metavar="CELL",
        help="with --rerun-interrupted or --auto-resume: also relaunch, with "
        "--new-attempt (learning again from scratch), the cells (none named: every "
        "selected cell) whose learning a HARNESS failure has stopped when this "
        "scheduler starts (the harness's or MIRIX's own failure, which a resume "
        "repeats), once a person has looked at each; each such stop gets one new "
        "attempt (its launches carry --new-attempt until that attempt exists), and a "
        "HARNESS stop that happens later waits for a person again",
    )
    parser.add_argument(
        "--new-attempt-cells",
        nargs="+",
        metavar="CELL",
        help="one-off: start one new attempt of each of these cells whatever their "
        "latest attempt reached (a complete cell too), with --new-attempt --note "
        "(run.py marks a tested attempt it replaces superseded, and keeps it); needs "
        "--new-attempt-note. Recorded per note in _schedule/resume_state.json: the same "
        "note never reruns a cell twice (a restarted scheduler requests nothing), a "
        "deliberate second rerun takes a new note. The estimate (what the replaced "
        "attempt cost) is recorded in state.json new_attempts; the reruns spend from "
        "--budget like any other launch",
    )
    parser.add_argument(
        "--widen-selection",
        metavar="NOTE",
        help="one-off: this launch selects cells no earlier launch of the runs directory "
        "selected (e.g. MIRIX's cells after the baselines), on purpose: NOTE (one "
        "line) says why and is recorded in _schedule/selection.json; without it such a "
        "launch is refused",
    )
    parser.add_argument(
        "--new-attempt-note",
        metavar="TEXT",
        help="why the --new-attempt-cells run again: one line of at most "
        f"{NOTE_MAX_CHARS} characters, recorded by run.py (ignored without "
        "--new-attempt-cells: a supervisor may withhold the cells once applied)",
    )
    resume = parser.add_argument_group("automatic resume")
    resume.add_argument(
        "--auto-resume",
        action="store_true",
        help="re-queue cells that ended learning_interrupted, test_incomplete, "
        "infra_unavailable, interrupted or crashed, or whose worker died, by themselves "
        "(with backoff and caps), and wait through the gateway pauses that lift by "
        "themselves (low balance, daily limit, upstream 402, unknown budget)",
    )
    resume.add_argument(
        "--resume-backoff-base",
        type=float,
        default=DEFAULT_RESUME_BACKOFF_BASE_S,
        dest="resume_backoff_base_s",
        help="seconds before the first resume; doubled for each further resume "
        "without progress (default %(default)g)",
    )
    resume.add_argument(
        "--resume-backoff-cap",
        type=float,
        default=DEFAULT_RESUME_BACKOFF_CAP_S,
        dest="resume_backoff_cap_s",
        help="the longest wait before a resume, seconds (default %(default)g)",
    )
    resume.add_argument(
        "--max-resumes-without-progress",
        type=int,
        default=DEFAULT_MAX_RESUMES_WITHOUT_PROGRESS,
        help="consecutive resumes without progress before the cell needs a person "
        f"(default %(default)s; {CRASHED_MAX_RESUMES_WITHOUT_PROGRESS} for crashed)",
    )
    resume.add_argument(
        "--max-resumes",
        type=int,
        default=DEFAULT_MAX_RESUMES,
        help="resumes per cell in all before it needs a person (default %(default)s)",
    )
    resume.add_argument(
        "--reset-resume",
        nargs="*",
        metavar="CELL",
        help="clear the resume counters of these cells (none named: every selected "
        "cell) once a person has looked, so a capped cell is resumed again",
    )
    guard = parser.add_argument_group("resource guard")
    guard.add_argument(
        "--min-free-memory-gb",
        type=float,
        help="launch a cell only while the available memory minus its estimated peak "
        "(infra/resources.py CELL_MEMORY_GB) stays at or above this",
    )
    guard.add_argument(
        "--min-free-disk-gb",
        type=float,
        help="launch a cell only while the runs directory's disk has this much free",
    )
    guard.add_argument(
        "--memory-emergency-gb",
        type=float,
        help="SIGTERM the most recently launched worker (resumable) after "
        f"{EMERGENCY_SAMPLES} consecutive samples with less memory available "
        f"(SIGKILL if it still runs {EMERGENCY_KILL_AFTER_S:g} s later)",
    )
    guard.add_argument(
        "--benchmark-concurrency",
        nargs="*",
        type=benchmark_cap,
        default=[],
        metavar="BENCHMARK=N",
        help="per-benchmark caps on concurrent cells, e.g. alfworld=2 (by default only "
        f"{', '.join(f'{b}={n}' for b, n in DEFAULT_BENCHMARK_CAPS.items())}: its "
        "cells share one server; the host load is logged in _schedule/resources.jsonl, "
        "to size a cap for another benchmark if its sandbox timeouts grow)",
    )
    guard.add_argument(
        "--host-benchmark-concurrency",
        nargs="*",
        type=benchmark_cap,
        default=[],
        metavar="BENCHMARK=N",
        help="caps on concurrent cells of a benchmark over every scheduler of this host "
        "(the host directory, $SKILL_BENCH_HOST_DIR or ~/.cache/skill_bench/host), e.g. "
        "webshop=6 (by default only "
        f"{', '.join(f'{b}={n}' for b, n in DEFAULT_HOST_BENCHMARK_CAPS.items())}: one "
        "WebShop server serves the episodes of every scheduler of the host). The memory "
        "gate and the memory emergency see every scheduler of the host the same way",
    )
    guard.add_argument(
        "--resources-log-interval",
        type=float,
        default=DEFAULT_RESOURCES_LOG_INTERVAL_S,
        dest="resources_log_interval_s",
        help="seconds between rows of _schedule/resources.jsonl (default %(default)g)",
    )
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_S, dest="poll_s")
    parser.add_argument(
        "--dry-run", action="store_true", help="show each cell's plan and exit"
    )
    parser.add_argument("--log-level", default="info")
    return parser


#: ``plan_table``'s plan of a cell a ``--new-attempt-cells`` request relaunches.
PLAN_RERUN = "rerun"


def plan_table(
    cells: Sequence[CellSpec], options: ScheduleOptions
) -> list[dict[str, Any]]:
    tracker = ResumeTracker(options.control / RESUME_STATE_FILE)
    tracker.load()
    asked = set(options.new_attempt_cells)
    key = rerun_key(options.new_attempt_note) if options.new_attempt_note else None
    rows = []
    for spec in cells:
        paths = CellPaths(options.cell_dir(spec))
        kind, status = classify_cell(
            paths,
            rerun_interrupted=options.rerun_interrupted,
            rerun_harness_failures=options.rerun_harness_failures,
            auto_resume=options.auto_resume,
        )
        row: dict[str, Any] = {"cell": spec.cell_id, "plan": kind, "status": status}
        entry = tracker.cells.get(spec.cell_id)
        pending = tracker.rerun_pending(spec.cell_id)
        requests = (entry or {}).get("rerun_requests") or {}
        if kind != BLOCKED and pending is not None:
            row["plan"] = PLAN_RERUN
            row["rerun"] = {
                k: pending[1].get(k)
                for k in ("note", "from_attempt", "estimate_usd", "state")
            }
        elif kind != BLOCKED and spec.cell_id in asked and key not in requests:
            latest = latest_attempt(paths)[0]
            if latest != 0:  # a cell that never ran starts its first attempt anyway
                row["plan"] = PLAN_RERUN
                row["rerun"] = {
                    "note": options.new_attempt_note,
                    "from_attempt": latest if latest > 0 else None,
                    "estimate_usd": attempt_cost_usd(paths, latest)
                    if latest > 0
                    else None,
                    "state": "to_request",
                }
        if kind == WAITING and entry is not None:
            row["resume"] = {
                k: entry.get(k)
                for k in ("resumes", "without_progress", "next_at", "attention")
            }
        rows.append(row)
    return rows


def options_from_args(args: argparse.Namespace) -> ScheduleOptions:
    return ScheduleOptions(
        runs_dir=args.runs_dir,
        concurrency=args.concurrency,
        gateway_url=args.gateway_url,
        gateway_runs_dir=args.gateway_runs_dir,
        gateway_port=args.gateway_port,
        env_file=args.env_file,
        balance_floor_usd=args.balance_floor,
        budget_usd=args.budget,
        budget_start_from=args.budget_start_from,
        test_concurrency=args.test_concurrency,
        idle_timeout_s=args.idle_timeout,
        finalize_timeout_s=args.finalize_timeout,
        max_learning_attempts=args.max_learning_attempts,
        rerun_interrupted=args.rerun_interrupted,
        rerun_harness_failures=args.rerun_harness_failures is not None,
        rerun_harness_cells=tuple(args.rerun_harness_failures or ()),
        new_attempt_cells=tuple(args.new_attempt_cells or ()),
        new_attempt_note=args.new_attempt_note,
        guard_selection=True,
        widen_selection=args.widen_selection,
        auto_resume=args.auto_resume,
        resume_backoff_base_s=args.resume_backoff_base_s,
        resume_backoff_cap_s=args.resume_backoff_cap_s,
        max_resumes_without_progress=args.max_resumes_without_progress,
        max_resumes=args.max_resumes,
        reset_resume=None if args.reset_resume is None else tuple(args.reset_resume),
        min_free_memory_gb=args.min_free_memory_gb,
        min_free_disk_gb=args.min_free_disk_gb,
        memory_emergency_gb=args.memory_emergency_gb,
        benchmark_concurrency=tuple(args.benchmark_concurrency or ()),
        host_benchmark_concurrency=tuple(args.host_benchmark_concurrency or ()),
        resources_log_interval_s=args.resources_log_interval_s,
        poll_s=args.poll_s,
    )


async def _amain(scheduler: Scheduler) -> ScheduleResult:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, scheduler.stop.set)
    return await scheduler.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cells = cells_from_args(args)
        options = options_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    for flag, named in (
        ("--reset-resume", args.reset_resume),
        ("--rerun-harness-failures", args.rerun_harness_failures),
        ("--new-attempt-cells", args.new_attempt_cells),
    ):
        problem = named_cells_problem(flag, named, cells)
        if problem:
            parser.error(problem)
    if args.new_attempt_note is not None and not args.new_attempt_cells:
        # The supervisor withholds --new-attempt-cells once one scheduler applied them
        # and passes the rest of the command line on.
        log.info("--new-attempt-note without --new-attempt-cells: ignored")
    if args.dry_run:
        for row in plan_table(cells, options):
            print(json.dumps(row))
        return 0
    try:
        result = asyncio.run(_amain(Scheduler(cells, options)))
    except ScheduleError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(result.to_dict()))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
