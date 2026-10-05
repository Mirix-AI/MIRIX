"""Cell directory layout, crash-safe record files, cell states and the gateway ledger summary.

Shared by ``run.py`` (writes a cell), ``schedule.py`` (reads cell states) and, later,
``report.py`` / ``audit.py`` (read the records). H1 (the learning stream and the
frozen test phase), H8 (attempts, restarts and resume), H9 (cost inputs) and H10.4 (the
frozen snapshot).

Layout of one cell (``runs/<cell_id>/``, :class:`CellPaths`)::

    cell.json            manifest: the immutable cell definition (items, fingerprint) and
                         the latest invocation (git commit, gateway, providers, attempts)
    status.json          scheduler-facing status of the cell (CellStatus)
    process.json         identity of the live run.py (pid, start time, argv): PID takeover
    cell.lock            flock held by the live run.py for its whole lifetime
    launches.jsonl       run.py invocations: a ``start`` line for each, and a ``work`` line
                         (the attempt, phase, git commit and model-visible code, and the
                         version ``segment``, ``note`` and ``auto_freeze`` status) when
                         it goes on to run an attempt's learning or test phase
    operator_notes/      notes an operator recorded with ``--acknowledge-rerun`` (cell.json
                         ``operator_notes`` lists them with their sha256)
    private/salt         the cell's secret salt for opaque task uids (H5), mode 0600
    results.jsonl        complete cells: the final test result per item, in test order
    learning.jsonl       complete cells: the learning-curve rows of the final attempt
    summary.json         complete cells: headline numbers and pointers
    owned_processes.jsonl, server.lock   the MIRIX server (infra/mirix_server.py)
    attempt-NN/          one learning attempt and its test phase (H8)
      reservation.json   the attempt number and gateway run reserved before the gateway
                         hears of the run; a directory with it and no attempt.json is an
                         attempt that never started (its run is abandoned)
      attempt.json       the attempt's state (AttemptState) and its history; also
                         version_segments and versions_mixed (the versions it ran
                         under), note and supersedes (a --new-attempt --note that
                         replaced a tested attempt), superseded (on the replaced one)
                         and accepted_learning_calls (MIRIX learning calls lost again
                         after 3 rollbacks and accepted)
      versions/SS/version.json   one version segment: its fingerprint, constants,
                         file map and per-file code hashes (run.py)
      versions/SS/tree.diff      that segment's uncommitted changes as a binary patch
                         (``git apply`` on its commit rebuilds the tree)
      method/            the method's work_dir (CellContext.work_dir); write-once bulk data
                         under method/objects/ (methods/base.OBJECTS_DIR)
      home/, logs/       the MIRIX server's HOME and log (infra/mirix_server.py)
      learning.jsonl     one row per learning item: stream position and private score (H9)
      results.jsonl      test rows, append-only; the last row of an item wins
      injections.jsonl   one row per before_task: phase, o200k tokens, item ids, text
      failures.jsonl     every H8 failure (infra/failures.FailureLog)
      trajectories/<phase>/<uid>.json    private trajectory records (images in the blob dir)
      trajectories/<phase>/failed/<uid>-<launch>-<n>.json   items that failed (forensics)
      snapshots/frozen.json              the memory frozen after learning (H1.5)
      snapshots/<check>-<k>.json         snapshots compared with it (resume, after_test,
                                         learning_resume)
      costs.json         the attempt's gateway ledger summary (H9 cost inputs)
      checkpoints/       learning checkpoints (run.py):
        LATEST           {"id", "ordinal", "sha256" of its checkpoint.json}
        c0000-init/, c<p>-played/, c<p>-done/   the latest two checkpoints, each with
                         checkpoint.json (written last), method/ (a copy of the work_dir,
                         objects/** hard-linked), snapshot.json, and for MIRIX
                         mirix_db.dump, home_manifest.json and home_files/
      discarded.jsonl    one row per discarded segment: what a learning resume rolled back
      discarded/seg-SS/  what that resume moved aside (the method directory, the tails of
                         learning.jsonl and injections.jsonl, trajectories, MIRIX's new
                         HOME files, mirix_export.json)

Everything under a cell directory is scorer-side: methods only ever receive the views of
``agent/trajectory.py`` and their own ``work_dir``.

A runs directory also holds ``_gateway/``, ``_schedule/``, ``_blobs/``, ``_audit/``,
``_report/``, ``_supervise/``, ``_removed/`` (the manifests of a removed method's data,
``tools/remove_method_data.py``) and, in a merged view of several hosts
(``tools/sync_runs.py``), ``_hosts/<name>/`` and ``_merged.json``
(:data:`NON_CELL_DIRS`, :data:`MERGED_MARKER`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.agent.trajectory import canonical_json
from evals.skill_bench.infra.failures import (
    PROVIDER_UNSERVED,
    REFUSAL_STREAK,
    Failure,
    Recovery,
    ledger_line_failure,
    refusal_class,
)
from evals.skill_bench.infra.processes import append_jsonl, lock_is_held

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

#: ``attempt.json`` ``state`` values (H8).
ATTEMPT_LEARNING = "learning"  # the learning stream is running (or its harness died)
#: The learning phase stopped on an infrastructure failure or an interruption, or on a
#: harness failure (``needs_decision: "harness"``): the next invocation resumes it from
#: its latest checkpoint in the same attempt and gateway run.
ATTEMPT_LEARNING_INTERRUPTED = "learning_interrupted"
ATTEMPT_INVALID = "invalid"  # invalidated: the cell starts a new attempt
ATTEMPT_FROZEN = "frozen"  # memory frozen after learning; the test phase may start
ATTEMPT_TESTING = "testing"  # the test phase has started; resumable per item
ATTEMPT_COMPLETE = "complete"  # every test item scored, snapshot unchanged (H10.4)
ATTEMPT_VIOLATION = "violation"  # the frozen memory changed or could not be restored
ATTEMPT_STATES: tuple[str, ...] = (
    ATTEMPT_LEARNING,
    ATTEMPT_LEARNING_INTERRUPTED,
    ATTEMPT_INVALID,
    ATTEMPT_FROZEN,
    ATTEMPT_TESTING,
    ATTEMPT_COMPLETE,
    ATTEMPT_VIOLATION,
)
#: ``attempt.json`` ``needs_decision`` of an attempt a harness failure stopped: a person
#: picks ``--new-attempt`` or ``--force-resume``.
NEEDS_DECISION_HARNESS = "harness"

#: ``status.json`` ``status`` values, and the exit code ``run.py`` returns with each.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_TEST_INCOMPLETE = "test_incomplete"
#: The learning phase stopped for a person: a harness failure (``failure_kind:
#: harness``), a corrupt or mismatching checkpoint, or (legacy) an attempt without
#: checkpoints that an infrastructure failure invalidated.
STATUS_LEARNING_FAILED = "learning_failed"
#: The learning phase stopped on an infrastructure failure or an interruption; running
#: the cell again resumes it from its latest checkpoint.
STATUS_LEARNING_INTERRUPTED = "learning_interrupted"
STATUS_REFUSED = "refused"
STATUS_VIOLATION = "violation"
STATUS_BUDGET_PAUSED = "budget_paused"
STATUS_BUSY = "busy"
STATUS_INTERRUPTED = "interrupted"
STATUS_CRASHED = "crashed"
#: The cell's infrastructure was unreachable before anything ran (the gateway unhealthy
#: or refusing a run, Postgres down for a MIRIX cell, the WebShop server unreachable or
#: still loading): transient, resumable.
STATUS_INFRA_UNAVAILABLE = "infra_unavailable"
EXIT_CODES: Mapping[str, int] = {
    STATUS_COMPLETE: 0,
    STATUS_CRASHED: 1,
    STATUS_TEST_INCOMPLETE: 2,
    STATUS_LEARNING_FAILED: 3,
    STATUS_REFUSED: 4,
    STATUS_VIOLATION: 4,
    STATUS_BUSY: 5,
    STATUS_BUDGET_PAUSED: 6,
    STATUS_INFRA_UNAVAILABLE: 7,
    STATUS_LEARNING_INTERRUPTED: 8,
    STATUS_INTERRUPTED: 130,
}
#: Statuses after which re-running the cell continues it (a learning resume, a new
#: learning attempt, or the pending test items).
RESUMABLE_STATUSES = frozenset(
    {
        STATUS_TEST_INCOMPLETE,
        STATUS_LEARNING_FAILED,
        STATUS_LEARNING_INTERRUPTED,
        STATUS_INTERRUPTED,
        STATUS_CRASHED,
        STATUS_INFRA_UNAVAILABLE,
    }
)
#: The statuses ``schedule.py --auto-resume`` re-queues by itself, with a backoff.
#: ``learning_failed`` is not among them: it waits for a person.
AUTO_RESUMABLE_STATUSES = frozenset(
    {
        STATUS_LEARNING_INTERRUPTED,
        STATUS_TEST_INCOMPLETE,
        STATUS_INFRA_UNAVAILABLE,
        STATUS_INTERRUPTED,
        STATUS_CRASHED,
    }
)
#: Statuses that need a person: re-running would only refuse again.
BLOCKED_STATUSES = frozenset({STATUS_REFUSED, STATUS_VIOLATION})

#: Failure codes of the learning checkpoints (``failures.jsonl``, ``attempt.json``
#: ``failure``): a restored checkpoint that does not reproduce its recorded state and a
#: checkpoint that fails its own hashes are HARNESS (a person looks); a checkpoint that
#: could not be taken because a call was still in flight, and a full disk, are INFRA.
FAILURE_CHECKPOINT_MISMATCH = "checkpoint_mismatch"
FAILURE_CHECKPOINT_CORRUPT = "checkpoint_corrupt"
FAILURE_CHECKPOINT_NOT_QUIESCENT = "checkpoint_not_quiescent"
FAILURE_DISK_FULL = "disk_full"
#: The recovery of an infrastructure failure in the learning phase: the next invocation
#: resumes the attempt from its checkpoint (``infra/failures.Recovery`` names the other
#: recoveries).
RECOVERY_RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"

#: Learning checkpoint kinds (run.py): ``init`` after ``setup``; per learning item
#: ``played`` (scored and recorded, before ``after_task``) and ``done`` (after
#: ``wait_until_idle`` and the ledger check).
CHECKPOINT_INIT = "init"
CHECKPOINT_PLAYED = "played"
CHECKPOINT_DONE = "done"
CHECKPOINT_KINDS: tuple[str, ...] = (
    CHECKPOINT_INIT,
    CHECKPOINT_PLAYED,
    CHECKPOINT_DONE,
)
#: The checkpoint file inside a checkpoint directory, written last.
CHECKPOINT_RECORD = "checkpoint.json"
#: Gateway ledger events ``run.py`` marks (``POST /runs/<run_id>/mark``).
LEDGER_EVENT_CHECKPOINT = "checkpoint"
LEDGER_EVENT_RESUME = "resume"
_CHECKPOINT_ID = re.compile(r"^c(\d{4,})-(init|played|done)$")

#: Directories of a runs directory that are not cells.
NON_CELL_DIRS = frozenset(
    {
        "_gateway",
        "_schedule",
        "_blobs",
        "_report",
        "_audit",
        "_hosts",
        "_supervise",
        "_removed",
    }
)
#: A runs directory holding this file is a merged read-only view of several hosts
#: (``tools/sync_runs.py``): cells are never run in it.
MERGED_MARKER = "_merged.json"
#: Where a merged view keeps each host's mirrored runs directory (``_hosts/<name>/``).
HOSTS_DIR = "_hosts"

#: ``launches.jsonl`` ``event`` values: every invocation's start line, and the line of an
#: invocation that went on to run an attempt's learning or test phase, with the code it
#: ran (the audit's H10.6 code check reads only these). Lines written before work
#: lines existed have no ``event`` and carry the code of every invocation.
LAUNCH_START = "start"
LAUNCH_WORK = "work"
#: ``cell.json`` ``operator_notes`` kind: the operator acknowledged that this cell directory
#: runs a cell again whose earlier instance had run test items (``--acknowledge-rerun``).
NOTE_RERUN_ACKNOWLEDGED = "rerun_acknowledged"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checkpoint_ordinal(kind: str, position: int) -> int:
    """The order of a learning checkpoint in its attempt: ``init`` 0, ``played`` of item
    ``p`` 2p+1, ``done`` of item ``p`` 2p+2 (a later checkpoint has a larger ordinal)."""
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"checkpoint kind must be one of {CHECKPOINT_KINDS}")
    if position < 0:
        raise ValueError("a checkpoint position is never negative")
    if kind == CHECKPOINT_INIT:
        if position != 0:
            raise ValueError("the init checkpoint is at position 0")
        return 0
    return 2 * position + (1 if kind == CHECKPOINT_PLAYED else 2)


def checkpoint_id(kind: str, position: int) -> str:
    """``c0000-init``, ``c<p:04d>-played`` or ``c<p:04d>-done``."""
    checkpoint_ordinal(kind, position)  # validates
    return f"c{position:04d}-{kind}"


def parse_checkpoint_id(cid: str) -> tuple[str, int]:
    """``(kind, position)`` of a checkpoint id; ValueError for anything else."""
    match = _CHECKPOINT_ID.fullmatch(str(cid))
    if match is None:
        raise ValueError(f"not a checkpoint id: {cid!r}")
    kind, position = match.group(2), int(match.group(1))
    checkpoint_ordinal(kind, position)
    return kind, position


#: Where :func:`machine_id` reads the machine's own id.
IOREG = "/usr/sbin/ioreg"
LINUX_MACHINE_ID_FILES: tuple[str, ...] = (
    "/etc/machine-id",
    "/var/lib/dbus/machine-id",
)
_IOPLATFORM_UUID = re.compile(r'"IOPlatformUUID"\s*=\s*"([0-9A-Fa-f-]{8,})"')
#: How :func:`_raw_machine_id` asks ``ioreg``: a record written while one ``ioreg``
#: timed out has no machine id and is compared by hostname, which a Mac
#: rename then breaks. A run that failed (timed out, could not start, exited non-zero,
#: printed nothing) is tried again, up to :data:`IOREG_TRIES` times, each with
#: :data:`IOREG_TIMEOUT_S`, :data:`IOREG_RETRY_WAIT_S` apart; a run that answered
#: without an ``IOPlatformUUID`` is not (it would answer the same).
IOREG_TRIES = 3
IOREG_TIMEOUT_S = 20.0
IOREG_RETRY_WAIT_S = 1.0


def _ioreg_platform_uuid() -> str | None:
    """The IOPlatformUUID ``ioreg`` prints, tried again after a failed run (blocking)."""
    for attempt in range(IOREG_TRIES):
        if attempt:
            time.sleep(IOREG_RETRY_WAIT_S)
        try:
            done = subprocess.run(
                [IOREG, "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=IOREG_TIMEOUT_S,
                check=False,
            )
        except FileNotFoundError:
            return None  # no ioreg on this machine: trying again cannot help
        except (OSError, subprocess.SubprocessError):
            continue  # timed out, or could not start (EAGAIN under load)
        match = _IOPLATFORM_UUID.search(done.stdout or "")
        if match:
            return match.group(1).upper()
        if done.returncode == 0 and (done.stdout or "").strip():
            return None  # it answered, without the id
    return None


def _raw_machine_id(system: str) -> str | None:
    """The machine's own id (blocking): macOS's IOPlatformUUID, Linux's machine-id; None
    when it cannot be read."""
    if system == "Darwin":
        return _ioreg_platform_uuid()
    if system == "Linux":
        for name in LINUX_MACHINE_ID_FILES:
            try:
                value = Path(name).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if value:
                return value
    return None


_machine_id: str | None = None
_machine_id_lock = threading.Lock()
#: After a failed read, the machine is not asked again for this long: a hung
#: ``ioreg`` costs about a minute of retries, and every record an invocation
#: writes asks for the id, under a lock.
MACHINE_ID_FAILURE_HOLD_S = 300.0
_machine_id_failed_at: float | None = None


def _hold_clock() -> float:
    """The clock of :data:`MACHINE_ID_FAILURE_HOLD_S` (a seam for the tests)."""
    return time.monotonic()


def machine_id() -> str | None:
    """A stable id of this machine (blocking until read, then cached), or None.

    The hostname is not one: on a Mac without a ``scutil`` HostName it follows the
    network (DHCP, a VPN, a Bonjour rename such as ``my-laptop-2.local``), so a laptop
    that joins another network mid-phase would look like another host. The raw id
    is kept private, as systemd asks of ``/etc/machine-id``: this is the sha256 of it
    under a skill-bench prefix, first 32 hex digits.

    Only an id that was read is cached: after a failure a later call
    asks the machine again, so one slow ``ioreg`` does not leave a whole process (every
    record it writes) without an id; not before :data:`MACHINE_ID_FAILURE_HOLD_S` though,
    so a hung one does not stall every record."""
    global _machine_id, _machine_id_failed_at
    if _machine_id is not None:
        return _machine_id
    with _machine_id_lock:
        failed = _machine_id_failed_at
        if _machine_id is None and (
            failed is None or _hold_clock() - failed >= MACHINE_ID_FAILURE_HOLD_S
        ):
            raw = _raw_machine_id(platform.system())
            if raw:
                _machine_id = hashlib.sha256(
                    f"skill-bench-host:{raw}".encode()
                ).hexdigest()[:32]
                _machine_id_failed_at = None
            else:
                _machine_id_failed_at = _hold_clock()
    return _machine_id


def host_info() -> dict[str, str | None]:
    """The host an attempt runs on (``attempt.json`` ``host``, ``launches.jsonl``): a
    learning resume must run on the same host (its database and HOME are local), and the
    audit compares ``system`` / ``machine`` across the cells of a benchmark.
    ``machine_id`` (:func:`machine_id`) is what identifies the host; the hostname is
    kept for people to read."""
    return {
        "hostname": socket.gethostname(),
        "machine_id": machine_id(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


#: The fields of :func:`host_info` that name a host (a new kernel or Python patch
#: release is not another host). :func:`same_host` compares ``machine_id``, ``system``
#: and ``machine`` when both records have a machine id, else (a record written before
#: machine ids, or a machine whose id cannot be read) the hostname, system and machine.
HOST_IDENTITY_FIELDS: tuple[str, ...] = ("hostname", "machine_id", "system", "machine")


def same_host(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    """Whether two :func:`host_info` records name the same host (see
    :data:`HOST_IDENTITY_FIELDS`); a missing record is never the same host."""
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    if a.get("machine_id") and b.get("machine_id"):
        fields: tuple[str, ...] = ("machine_id", "system", "machine")
    else:
        fields = ("hostname", "system", "machine")
    return all(a.get(k) is not None and a.get(k) == b.get(k) for k in fields)


def host_key(host: Mapping[str, Any]) -> str:
    """What tells two hosts apart across cells: the machine id, else the hostname."""
    machine = host.get("machine_id")
    return f"id:{machine}" if machine else f"hostname:{host.get('hostname')}"


def host_text(host: Mapping[str, Any] | None) -> str:
    """``hostname (system machine, id 1234abcd)`` of a :func:`host_info` record."""
    if not isinstance(host, Mapping):
        return "an unrecorded host"
    machine = host.get("machine_id")
    tail = f", id {str(machine)[:8]}" if machine else ""
    return f"{host.get('hostname')} ({host.get('system')} {host.get('machine')}{tail})"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellPaths:
    """Paths of one cell directory (see the module docstring)."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "cell.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def process(self) -> Path:
        return self.root / "process.json"

    @property
    def lock(self) -> Path:
        return self.root / "cell.lock"

    @property
    def launches(self) -> Path:
        return self.root / "launches.jsonl"

    @property
    def salt(self) -> Path:
        return self.root / "private" / "salt"

    @property
    def operator_notes(self) -> Path:
        return self.root / "operator_notes"

    @property
    def results(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def learning(self) -> Path:
        return self.root / "learning.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def worker_log(self) -> Path:
        return self.root / "worker.log"

    def attempt(self, n: int) -> AttemptPaths:
        return AttemptPaths(self.root / f"attempt-{n:02d}", n)

    def attempt_numbers(self) -> list[int]:
        """Attempts that have an ``attempt.json``, in order."""
        found = []
        if self.root.is_dir():
            for child in self.root.iterdir():
                name = child.name
                if (
                    name.startswith("attempt-")
                    and name[8:].isdigit()
                    and (child / "attempt.json").is_file()
                ):
                    found.append(int(name[8:]))
        return sorted(found)


@dataclass(frozen=True)
class AttemptPaths:
    """Paths of one attempt directory (``attempt-NN``, the MIRIX server's ``attempt_dir`` too)."""

    root: Path
    number: int

    @property
    def state(self) -> Path:
        return self.root / "attempt.json"

    @property
    def reservation(self) -> Path:
        return self.root / "reservation.json"

    @property
    def method_dir(self) -> Path:
        return self.root / "method"

    @property
    def learning(self) -> Path:
        return self.root / "learning.jsonl"

    @property
    def results(self) -> Path:
        return self.root / "results.jsonl"

    @property
    def injections(self) -> Path:
        return self.root / "injections.jsonl"

    @property
    def failures(self) -> Path:
        return self.root / "failures.jsonl"

    @property
    def costs(self) -> Path:
        return self.root / "costs.json"

    def trajectories(self, phase: str) -> Path:
        return self.root / "trajectories" / phase

    @property
    def trajectories_root(self) -> Path:
        return self.root / "trajectories"

    def snapshot(self, name: str) -> Path:
        return self.root / "snapshots" / f"{name}.json"

    @property
    def frozen_snapshot(self) -> Path:
        return self.snapshot("frozen")

    @property
    def home(self) -> Path:
        """The MIRIX server's HOME (``infra/mirix_server.CellServerSpec.home``)."""
        return self.root / "home"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        """``checkpoints/LATEST``: the id of the newest complete checkpoint."""
        return self.checkpoints / "LATEST"

    def checkpoint(self, cid: str) -> Path:
        return self.checkpoints / cid

    @property
    def discarded_log(self) -> Path:
        """``discarded.jsonl``: one row per discarded segment of a resumed learning phase."""
        return self.root / "discarded.jsonl"

    def discarded_dir(self, segment: int) -> Path:
        return self.root / "discarded" / f"seg-{segment:02d}"


# ---------------------------------------------------------------------------
# Crash-safe files
# ---------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
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


def write_json_atomic(path: Path, value: Any, *, mode: int | None = None) -> None:
    """Replace ``path`` with ``value`` as JSON: temporary file, fsync, rename, fsync the directory.

    Blocking; async callers use ``asyncio.to_thread``. Safe against torn files after a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as handle:
        if mode is not None:
            os.chmod(tmp, mode)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def read_json(path: Path) -> Any | None:
    """The JSON value of ``path``, or None when the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)


def read_jsonl_counted(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Rows of a JSONL file and the number of unparsable (torn) lines skipped.

    A line torn by a crash or a full disk mid-append is skipped wherever it is:
    ``append_jsonl`` ends a torn tail with a newline before the next record, so the
    fragment is a line of its own and every record after it stays readable.
    """
    try:
        # Split on "\n" only: writers use ensure_ascii=False, so a record may hold U+2028,
        # U+2029 or U+0085, which str.splitlines() would treat as line breaks and tear
        # the record in two.
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except FileNotFoundError:
        return [], 0
    rows = []
    torn = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            torn += 1  # a fragment torn mid-write; the record it held is lost
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            torn += 1
    return rows, torn


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file (missing file: none). Torn lines are skipped (see
    :func:`read_jsonl_counted`, which also counts them)."""
    return read_jsonl_counted(path)[0]


class JsonlWriter:
    """Appends fsync'd JSON lines from concurrent coroutines, one line at a time.

    Each append runs in a worker thread (blocking I/O off the event loop) under an asyncio
    lock, so lines of concurrent test items never interleave.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def append(self, record: Mapping[str, Any]) -> None:
        async with self._lock:
            await asyncio.to_thread(append_jsonl, self.path, record)


def canonical_sha256(value: Any) -> str:
    """sha256 of the canonical JSON of ``value`` (sorted keys, no spaces): snapshot hashes."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    """sha256 of newline-joined strings (item orders in the manifest)."""
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


#: The cell-level files whose rows report.py publishes (its metrics and learning curves).
PUBLISHED_RECORDS = ("results.jsonl", "learning.jsonl")


def records_digest(paths: CellPaths) -> str:
    """sha256 over the bytes of the cell files report.py publishes numbers from. The audit
    records it and the report compares it, so a report only calls a cell audited when the
    audit saw the very records it prints (blocking)."""
    digest = hashlib.sha256()
    for name in PUBLISHED_RECORDS:
        path = paths.root / name
        digest.update(name.encode("utf-8") + b"\0")
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            digest.update(b"-\0")
            continue
        digest.update(str(len(data)).encode("ascii") + b"\0" + data)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


#: Tracked files a change to which does not make the checkout dirty: the frozen protocol
#: record, which a run itself rewrites when it freezes the protocol (the
#: auto-freeze); its content is recorded per attempt instead.
GIT_DIRTY_EXCLUDED: tuple[str, ...] = ("evals/skill_bench/config/protocol_frozen.json",)

#: The frozen record's untracked side files, never part of a version (the fingerprint
#: skips dot files): the lock file ``tools/freeze_protocol.record_lock`` takes where the
#: record's directory cannot be locked (NFS) and the temporary file of the
#: record's atomic replace (left behind by a writer killed mid-write).
FROZEN_RECORD_SIDE_FILES: tuple[str, ...] = tuple(
    f"{GIT_DIRTY_EXCLUDED[0].rsplit('/', 1)[0]}/.protocol_frozen.json.{suffix}"
    for suffix in ("lock", "tmp")
)

#: What a version segment's tree diff covers (``run.capture_tree``; pathspecs relative to
#: the repository's top): the harness, without its tests, runs, data and the frozen
#: record (another cell's automatic freeze changes it) and its side files...
TREE_PATHSPECS_HARNESS: tuple[str, ...] = (
    "evals/skill_bench",
    ":(exclude)evals/skill_bench/tests",
    ":(exclude)evals/skill_bench/runs",
    ":(exclude)evals/skill_bench/data",
    f":(exclude){GIT_DIRTY_EXCLUDED[0]}",
    *(f":(exclude){path}" for path in FROZEN_RECORD_SIDE_FILES),
)
#: ...and, for a MIRIX cell, the MIRIX package its server runs and the launcher that
#: starts that server (``infra/mirix_server.py`` runs ``scripts/start_server.py``).
TREE_PATHSPECS_MIRIX: tuple[str, ...] = (
    "mirix",
    ":(exclude)mirix/tests",
    protocol.MIRIX_LAUNCHER,
)


def tree_pathspecs(*, mirix: bool) -> list[str]:
    """The pathspecs a version segment's tree diff covers (pure): the harness and, for a
    MIRIX cell, the MIRIX package and its launcher."""
    return [*TREE_PATHSPECS_HARNESS, *(TREE_PATHSPECS_MIRIX if mirix else ())]


def _top_pathspec(spec: str) -> str:
    """``spec`` (relative to the repository's top) for a git run from any directory."""
    if spec.startswith(":(exclude)"):
        return ":(top,exclude)" + spec[len(":(exclude)") :]
    return ":(top)" + spec


def git_info(root: Path = protocol.REPO_ROOT) -> dict[str, Any]:
    """The checkout's commit and whether the commit alone does not rebuild it (blocking):
    ``dirty`` when a tracked file differs from it (leaving out
    :data:`GIT_DIRTY_EXCLUDED`) or an untracked, not ignored file lies under the paths a
    version's tree diff covers (:func:`tree_pathspecs` of a MIRIX cell, which includes
    a harness cell's), counted in ``untracked`` (the fingerprint and the code
    digests walk the disk, so a new prompt or module never ``git add``-ed is part of the
    version, and run.py captures a dirty tree's untracked files in ``tree.diff``). Both
    are None when git failed. A gitignored file a version hashes is run.py's
    ``tree.unreproducible``."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ":(top)",
        *(f":(top,exclude){path}" for path in GIT_DIRTY_EXCLUDED),
    )
    others = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *(_top_pathspec(s) for s in tree_pathspecs(mirix=True)),
    )
    untracked = None if others is None else len([n for n in others.split("\0") if n])
    return {
        "commit": commit,
        "dirty": None
        if status is None or untracked is None
        else bool(status) or untracked > 0,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "untracked": untracked,
    }


# ---------------------------------------------------------------------------
# Status (scheduler-facing)
# ---------------------------------------------------------------------------


def read_status(paths: CellPaths) -> dict[str, Any] | None:
    try:
        value = read_json(paths.status)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def cell_is_live(paths: CellPaths) -> bool:
    """True while a live run.py holds the cell lock (whatever its status file says)."""
    return lock_is_held(paths.lock)


def cell_progress(paths: CellPaths) -> dict[str, Any]:
    """How far a cell got, for the scheduler's no-progress count (blocking):
    ``{"attempt", "state", "checkpoint_ordinal", "learning_done", "test_scored"}`` of its
    latest attempt. A new checkpoint (a larger ordinal), more scored test items or a new
    attempt is progress. A cell without attempts reports ``attempt`` None."""
    progress: dict[str, Any] = {
        "attempt": None,
        "state": None,
        "checkpoint_ordinal": None,
        "learning_done": 0,
        "test_scored": 0,
    }
    try:
        numbers = paths.attempt_numbers()
    except OSError:
        return progress
    if not numbers:
        return progress
    attempt = paths.attempt(numbers[-1])
    progress["attempt"] = attempt.number
    try:
        state = read_json(attempt.state)
    except (OSError, ValueError):
        state = None
    if isinstance(state, dict):
        progress["state"] = state.get("state")
        learning = state.get("learning")
        if isinstance(learning, Mapping) and isinstance(learning.get("done"), int):
            progress["learning_done"] = learning["done"]
    try:
        latest = read_json(attempt.latest_checkpoint)
    except (OSError, ValueError):
        latest = None
    if isinstance(latest, dict) and isinstance(latest.get("ordinal"), int):
        progress["checkpoint_ordinal"] = latest["ordinal"]
    rows = read_jsonl(attempt.results)
    progress["test_scored"] = len(
        {str(r.get("raw_id")) for r in rows if r.get("score") is not None}
    )
    return progress


# ---------------------------------------------------------------------------
# Discarded segments of a resumed learning phase
# ---------------------------------------------------------------------------


def read_discarded(paths: AttemptPaths) -> list[dict[str, Any]]:
    """The rows of ``discarded.jsonl``, one per discarded segment, in order."""
    return read_jsonl(paths.discarded_log)


def consecutive_rollbacks(
    resumes: Sequence[Mapping[str, Any]] | None, checkpoint: str | None, *, code: str
) -> int:
    """How many learning resumes in a row, from the newest ``attempt.json`` ``resumes``
    row back, restored ``checkpoint`` after an INFRA failure with ``code`` (pure;
    ``run.py`` passes it to the method as ``LearningResume.rollbacks``). The first row
    that differs (an interruption, another checkpoint, another failure) ends the count."""
    count = 0
    for row in reversed(list(resumes or ())):
        failure = row.get("failure") if isinstance(row, Mapping) else None
        if (
            not isinstance(failure, Mapping)
            or row.get("checkpoint") != checkpoint
            or failure.get("kind") != "infra"
            or failure.get("code") != code
        ):
            break
        count += 1
    return count


def discarded_call_ids(paths: AttemptPaths) -> frozenset[str]:
    """Every gateway call id a learning resume of this attempt discarded (the union of the
    ``calls`` of every ``discarded.jsonl`` row)."""
    return frozenset(
        str(call["call_id"])
        for row in read_discarded(paths)
        for call in row.get("calls") or ()
        if isinstance(call, Mapping) and call.get("call_id")
    )


@dataclass(frozen=True)
class LedgerLine:
    """One parsed ledger line and the byte offset where it starts."""

    offset: int
    row: Mapping[str, Any]


def read_ledger_lines(path: Path) -> list[LedgerLine]:
    """Every parsable line of a gateway ledger with its byte offset (blocking; torn
    lines are skipped, a missing file has none)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    out: list[LedgerLine] = []
    offset = 0
    for raw in data.split(b"\n"):
        start, offset = offset, offset + len(raw) + 1
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(LedgerLine(start, row))
    return out


def segment_calls(
    lines: Iterable[LedgerLine],
    *,
    checkpoint_offset: int | None,
    resume_offset: int,
    exclude: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """The calls of one discarded segment: every call whose ``call_started`` line lies
    strictly between the restored checkpoint's marker (``checkpoint_offset``; None: the
    ledger's start, a resume before the first checkpoint) and the resume marker, less the
    calls an earlier segment already discarded (``exclude``: a resume that restored an
    older checkpoint covers the ledger an earlier resume discarded too). A call's final
    ``call`` line may come after the marker (in flight when the harness died): its
    ``final_line`` is then False until the gateway writes it.

    A final line without any ``call_started`` line (an older gateway answered a call it
    held during a budget pause 402 before writing one) is placed by its own offset, so
    it is discarded like every other call of its segment."""
    rows = list(lines)
    skip = {str(c) for c in exclude}
    finals: dict[str, Mapping[str, Any]] = {}
    for line in rows:
        if line.row.get("event") == "call" and line.row.get("call_id"):
            finals[str(line.row["call_id"])] = line.row
    lower = -1 if checkpoint_offset is None else checkpoint_offset
    calls: list[dict[str, Any]] = []
    for line in call_placements(rows):
        row = line.row
        call_id = str(row["call_id"])
        if not lower < line.offset < resume_offset or call_id in skip:
            continue
        final = finals.get(call_id)
        calls.append(
            {
                "call_id": call_id,
                "role": row.get("role"),
                "item": row.get("item"),
                "phase": row.get("run_phase") or row.get("phase"),
                "started_offset": line.offset,
                "gateway_start": row.get("gateway_start"),
                "final_line": final is not None,
                "status": final.get("status") if final is not None else None,
                "cost_total": _num(final.get("cost_total")) if final else 0.0,
                "unknown_cost": bool(final and unknown_cost_attempts(final) > 0),
            }
        )
    return calls


def discarded_segment_costs(
    rows: Iterable[Mapping[str, Any]], ledger_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """What each discarded segment's calls cost by the ledger's final lines now, one
    entry per ``discarded.jsonl`` row, in order.

    A row's own ``cost_total`` is what the final lines held when the resume wrote it: a
    call still in flight then (a zombie, or a call a gateway pause held) has
    ``final_line: false`` and no cost in it, and the gateway writes its final line later.
    ``discarded.jsonl`` stays as written; this recomputes from the ledger instead, as
    ``summarize_ledger``'s ``discarded`` bucket and ``report.py``'s resumes do:

    * ``cost_total``, ``unknown_cost_calls``: over the row's calls with a final line
      now; ``calls`` and ``by_role``: all of the row's calls;
    * ``unfinished_calls``: the row's calls still without one (cost unknown);
    * ``cost_total_at_resume``: the row's recorded figure; ``finished_after_resume``:
      the calls the row listed without a final line that have one now."""
    finals: dict[str, Mapping[str, Any]] = {}
    for line in ledger_rows:
        if line.get("event") == "call" and line.get("call_id"):
            finals[str(line["call_id"])] = line
    out: list[dict[str, Any]] = []
    for row in rows:
        calls = [c for c in row.get("calls") or () if isinstance(c, Mapping)]
        cost = 0.0
        unknown = unfinished = late = 0
        roles: Counter[str] = Counter()
        for call in calls:
            final = finals.get(str(call.get("call_id")))
            roles[str(call.get("role"))] += 1
            if final is None:
                unfinished += 1
                continue
            cost += _num(final.get("cost_total"))
            unknown += int(unknown_cost_attempts(final) > 0)
            late += int(call.get("final_line") is False)
        out.append(
            {
                "segment": row.get("segment"),
                "calls": len(calls),
                "cost_total": round(cost, 10),
                "unknown_cost_calls": unknown,
                "unfinished_calls": unfinished,
                "by_role": dict(sorted(roles.items())),
                "cost_total_at_resume": row.get("cost_total"),
                "finished_after_resume": late,
            }
        )
    return out


def call_placements(lines: Iterable[LedgerLine]) -> list[LedgerLine]:
    """The line that places each call of a ledger, in ledger order: its
    ``call_started`` line, or for a call that has none its final ``call`` line (see
    :func:`segment_calls`)."""
    rows = list(lines)
    started = {
        str(line.row["call_id"])
        for line in rows
        if line.row.get("event") == "call_started" and line.row.get("call_id")
    }
    placed: list[LedgerLine] = []
    for line in rows:
        event, call_id = line.row.get("event"), line.row.get("call_id")
        if not call_id:
            continue
        if event == "call_started" or (event == "call" and str(call_id) not in started):
            placed.append(line)
    return placed


# ---------------------------------------------------------------------------
# Host-portable lookups (a runs directory copied from another host)
# ---------------------------------------------------------------------------


def resolve_ledger_path(
    state: Mapping[str, Any],
    *,
    runs_dir: Path,
    gateway_runs_dirs: Iterable[Path] = (),
) -> Path:
    """The ledger of an attempt's gateway run R (``attempt.json`` ``run_id``), the first
    that exists of:

    1. ``<runs_dir>/R/gateway.jsonl``;
    2. ``<gateway_runs_dir>/R/gateway.jsonl`` for each ``gateway_runs_dirs``;
    3. ``<runs_dir>/_hosts/*/R/gateway.jsonl`` (a merged view of several hosts);
    4. the recorded absolute ``ledger``, only when its directory is named R (a path
       recorded on another host could name a stale local file).

    Without one, the path of step 1 (which the caller reports missing)."""
    run_id = str(state.get("run_id") or "")
    runs_dir = Path(runs_dir)
    first = runs_dir / run_id / "gateway.jsonl"
    if not run_id:
        return first
    candidates = [first]
    candidates += [Path(d) / run_id / "gateway.jsonl" for d in gateway_runs_dirs]
    hosts = runs_dir / HOSTS_DIR
    try:
        mirrors = sorted(hosts.iterdir()) if hosts.is_dir() else []
    except OSError:
        mirrors = []
    candidates += [m / run_id / "gateway.jsonl" for m in mirrors]
    recorded = state.get("ledger")
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        if path.parent.name == run_id:
            candidates.append(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return first


def resolve_blob_dir(manifest: Mapping[str, Any], runs_dir: Path) -> Path:
    """Where a cell's trajectory images are: ``<runs_dir>/_blobs`` when it exists, else
    the ``blob_dir`` its ``cell.json`` recorded (else ``<runs_dir>/_blobs``)."""
    local = Path(runs_dir) / "_blobs"
    if local.is_dir():
        return local
    recorded = manifest.get("blob_dir")
    if isinstance(recorded, str) and recorded and Path(recorded).is_dir():
        return Path(recorded)
    return local


# ---------------------------------------------------------------------------
# Gateway ledger summary (H9 cost inputs, providers, H10.3 linkage)
# ---------------------------------------------------------------------------


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "ok": 0,
        "failed": 0,
        "cost_total": 0.0,
        "billed_attempts": 0,
        # Calls with an attempt cut off after its request was sent (the deadline, or a
        # connection that dropped mid-answer): billed, at a cost the ledger never learns,
        # so ``cost_total`` leaves it out.
        "unknown_cost_calls": 0,
        "unknown_cost_attempts": 0,
        "retries": 0,
        "input_tokens_estimated": 0,
        "input_tokens_actual": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "latency_s": 0.0,
    }


#: Upper edges (seconds) of the latency histogram of :func:`summarize_ledger`'s
#: ``pricing``; the last bucket holds everything above. Additive across ledgers, so the
#: report bounds the latency quantiles of a whole matrix.
LATENCY_EDGES_S: tuple[float, ...] = (1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
#: Upper edges of the reasoning-token histogram of ``pricing`` (a 1-2-5 series; the last
#: bucket holds everything above), additive like the latency one: the report bounds a
#: provider's reasoning p95 by it (a model's reasoning may vary with the provider
#: OpenRouter picks).
REASONING_EDGES: tuple[int, ...] = (
    0,
    10,
    20,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
)
#: An ok call whose answering attempt took longer than this share of the call deadline
#: counts as ``slow_calls``: how close a provider brings calls to a model-blamed 504.
SLOW_CALL_DEADLINE_FRACTION = 0.5


@lru_cache(maxsize=1)
def _list_prices() -> Mapping[str, Mapping[str, Any]]:
    """``config/models.json`` pricing (USD per token) per model id (blocking, cached)."""
    try:
        models = protocol.load_models()["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return {m: dict(e.get("pricing") or {}) for m, e in models.items()}


def list_price_usd(
    model: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    cache_write_tokens: int = 0,
) -> tuple[float | None, bool]:
    """What a call costs at the model's list price in ``config/models.json`` (the
    tier of an ``overrides`` entry once the prompt reaches its ``min_prompt_tokens``:
    luna above 272K input), and whether that tier applied. None without a price.

    Cached prompt tokens count at ``input_cache_read`` and prompt tokens written to the
    cache (``usage.prompt_tokens_details.cache_write_tokens``) at ``input_cache_write``
    (the prompt price for a model that lists none); both are part of the prompt
    tokens, so the rest counts at ``prompt``. Reasoning tokens are part of the
    completion. OpenRouter bills the provider's own price, so ``cost / list`` tells the
    tier a call was served at (a discounted tier below 1, a priority tier above)."""
    pricing = _list_prices().get(model)
    if not pricing:
        return None, False
    tier: dict[str, Any] = dict(pricing)
    over = False
    for override in pricing.get("overrides") or ():
        minimum = (
            override.get("min_prompt_tokens") if isinstance(override, dict) else None
        )
        if isinstance(minimum, (int, float)) and prompt_tokens >= minimum:
            tier.update(override)
            over = True
    prompt_price = _num(tier.get("prompt"))
    cache_price = (
        _num(tier.get("input_cache_read"))
        if tier.get("input_cache_read") is not None
        else prompt_price
    )
    write_price = (
        _num(tier.get("input_cache_write"))
        if tier.get("input_cache_write") is not None
        else prompt_price
    )
    cached = max(0, min(cached_tokens, prompt_tokens))
    written = max(0, min(cache_write_tokens, prompt_tokens - cached))
    listed = (
        (prompt_tokens - cached - written) * prompt_price
        + cached * cache_price
        + written * write_price
        + completion_tokens * _num(tier.get("completion"))
    )
    return listed, over


def _pricing_entry() -> dict[str, Any]:
    return {
        "calls": 0,
        "priced_calls": 0,
        "cost_usd": 0.0,
        "list_cost_usd": 0.0,
        "ratio_min": None,
        "ratio_max": None,
        "over_tier_calls": 0,
        "latency_s_max": 0.0,
        "latency_buckets": [0] * (len(LATENCY_EDGES_S) + 1),
        "slow_calls": 0,
        "reasoning_calls": 0,
        "reasoning_tokens": 0,
        "reasoning_tokens_max": 0,
        "reasoning_buckets": [0] * (len(REASONING_EDGES) + 1),
    }


def _slot(value: float, edges: tuple[float, ...]) -> int:
    return next((i for i, edge in enumerate(edges) if value <= edge), len(edges))


def _add_pricing(entry: dict[str, Any], row: Mapping[str, Any]) -> None:
    """One ok call into its ``pricing[model][provider]`` entry."""
    entry["calls"] += 1
    attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
    last = attempts[-1] if attempts and isinstance(attempts[-1], Mapping) else {}
    # The attempt that answered, not the call: queueing and retries are not the tier's.
    latency = _num(last.get("latency_s", row.get("latency_s")))
    entry["latency_s_max"] = round(max(entry["latency_s_max"], latency), 3)
    entry["latency_buckets"][_slot(latency, LATENCY_EDGES_S)] += 1
    if latency > protocol.CALL_DEADLINE_S * SLOW_CALL_DEADLINE_FRACTION:
        entry["slow_calls"] += 1
    reasoning = row.get("reasoning_tokens")
    if isinstance(reasoning, (int, float)) and not isinstance(reasoning, bool):
        entry["reasoning_calls"] += 1
        entry["reasoning_tokens"] += int(reasoning)
        entry["reasoning_tokens_max"] = max(
            entry["reasoning_tokens_max"], int(reasoning)
        )
        entry["reasoning_buckets"][_slot(reasoning, REASONING_EDGES)] += 1
    cost = row.get("cost")
    usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
    details = usage.get("prompt_tokens_details")
    written = (
        details.get("cache_write_tokens") if isinstance(details, Mapping) else None
    )
    listed, over = list_price_usd(
        str(row.get("model")),
        int(_num(row.get("input_tokens_actual"))),
        int(_num(row.get("cached_tokens"))),
        int(_num(row.get("output_tokens"))),
        int(_num(written)),
    )
    entry["over_tier_calls"] += int(over)
    if cost is None or listed is None or listed <= 0:
        return
    entry["priced_calls"] += 1
    entry["cost_usd"] = round(entry["cost_usd"] + _num(cost), 10)
    entry["list_cost_usd"] = round(entry["list_cost_usd"] + listed, 10)
    ratio = round(_num(cost) / listed, 4)
    for key, pick in (("ratio_min", min), ("ratio_max", max)):
        entry[key] = ratio if entry[key] is None else pick(entry[key], ratio)


def unknown_cost_attempts(row: Mapping[str, Any]) -> int:
    """The attempts of a final ``call`` line cut off after their request was sent: by the
    deadline, or by a connection that dropped mid-answer.
    OpenRouter billed them for what the provider generated, at a cost the ledger never
    learns. The gateway's count (``unknown_cost_attempts``); a line written before it
    counted dropped connections, its ``unbilled_deadline_attempts``; a line written
    before the gateway counted anything, its attempts whose ``outcome`` is ``deadline``
    — an upper bound: such a line does not say which of them never delivered the
    request (a pool or write timeout, which the gateway now leaves unmarked), nor
    which dropped connections were billed."""
    for key in ("unknown_cost_attempts", "unbilled_deadline_attempts"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
    return sum(
        1 for a in attempts if isinstance(a, Mapping) and a.get("outcome") == "deadline"
    )


def summarize_ledger(
    path: Path, *, discarded: Iterable[str] = frozenset()
) -> dict[str, Any]:
    """Summarise one gateway run's ledger (``<runs>/<run_id>/gateway.jsonl``; blocking).

    ``discarded`` names the calls a learning resume discarded (``discarded.jsonl``):
    they stay in every count below (they were made and billed) and are also summed on
    their own in ``discarded`` (the same fields as a ``by_phase_role`` bucket, plus
    ``by_role``), which the report shows as the cell's discarded learning cost.

    * ``by_phase_role``: calls, successes, cost (``cost_total``, which includes billed
      retried attempts), tokens and retries per ``phase/role``, and
      ``unknown_cost_calls``: calls with an attempt cut off after its request was sent
      (:func:`unknown_cost_attempts`), billed at a cost ``cost_total`` leaves out;
    * ``providers``: the providers OpenRouter actually used per ``phase/role``, and
      ``model_served`` per chat model (H1.8);
    * ``pricing``: per model and provider, the ok calls' billed cost against the list
      price of ``config/models.json`` (:func:`list_price_usd`: the ratio tells the
      service tier, which the ``provider`` name does not), the calls in the model's
      long-input price tier, and the answering attempt's latency (max and a histogram,
      :data:`LATENCY_EDGES_S`; ``slow_calls`` above :data:`SLOW_CALL_DEADLINE_FRACTION`
      of the deadline) and reasoning tokens (sum, max and a histogram,
      :data:`REASONING_EDGES`, over the calls that report them);
      ``timeouts_model_blamed``: calls per model that hit the deadline on the model's
      own attempt;
    * ``items``: actor cost per opaque task uid and phase (the per-item cost of H9), and
      ``items_unknown_cost``: the actor calls per uid and phase whose cost is unknown;
    * ``repeat_of``: flagged repeated requests per role (judged by the audit, H10.3);
    * ``routing``: per model the call went out as, the OpenRouter provider routing its
      ``wire`` carried (canonical JSON of ``wire.provider``, ``"none"`` without one) and
      how many calls carried it; ``routing_unrecorded``: calls per model whose line has no
      ``wire`` (a gateway before it was recorded; a call refused before the rewrite sent
      nothing and is in neither). The report states the policy from these;
    * ``refusals``: per ``phase/role``, the failed calls that were refusals, by
      ``<class>/<http_status>`` (``infra/failures.refusal_class``: ``payload``,
      ``no_endpoint`` or ``gateway``; discarded calls included);
    * ``protocol_conformant``: False if any call line was served by a non-protocol gateway;
    * ``call_ids``: every final call line's id (to link trajectories to the ledger);
    * ``unfinished_calls``: calls whose final line is missing (their gateway stopped
      mid-call; their cost is unknown), ``torn_lines``: unparsable lines skipped.

    A call's phase is the run's phase ``run.py`` set (``run_phase``), not the caller's tag.
    """
    rows, torn = read_jsonl_counted(path)
    by: dict[str, dict[str, Any]] = defaultdict(_bucket)
    dropped = {str(c) for c in discarded}
    lost = _bucket()
    lost_roles: Counter[str] = Counter()
    started: set[str] = set()
    providers: dict[str, Counter[str]] = defaultdict(Counter)
    served: dict[str, Counter[str]] = defaultdict(Counter)
    pricing: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    timeouts: Counter[str] = Counter()
    items: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    items_unknown: dict[str, Counter[str]] = defaultdict(Counter)
    repeats: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    routing: dict[str, Counter[str]] = defaultdict(Counter)
    routing_unrecorded: Counter[str] = Counter()
    refusals: dict[str, Counter[str]] = defaultdict(Counter)
    call_ids: list[str] = []
    events: Counter[str] = Counter()
    nonconformant = 0
    phases: list[str] = []
    abandoned = False
    for row in rows:
        event = str(row.get("event"))
        events[event] += 1
        if event == "phase":
            phases.append(str(row.get("phase")))
            continue
        if event == "abandoned":
            abandoned = True
            continue
        if event == "call_started" and row.get("call_id"):
            started.add(str(row["call_id"]))
        if event != "call":
            continue
        role = str(row.get("role"))
        # The run's phase run.py set, never a caller's own tag; older lines: phase.
        phase = str(row.get("run_phase") or row.get("phase"))
        key = f"{phase}/{role}"
        status = str(row.get("status"))
        statuses[status] += 1
        unbilled = unknown_cost_attempts(row)
        targets = [by[key]]
        if str(row.get("call_id")) in dropped:
            targets.append(lost)
            lost_roles[role] += 1
        for bucket in targets:
            bucket["calls"] += 1
            bucket["ok" if status == "ok" else "failed"] += 1
            bucket["cost_total"] += _num(row.get("cost_total"))
            bucket["billed_attempts"] += int(_num(row.get("billed_attempts")))
            bucket["unknown_cost_attempts"] += unbilled
            bucket["unknown_cost_calls"] += int(unbilled > 0)
            bucket["retries"] += int(_num(row.get("retries")))
            for field in (
                "input_tokens_estimated",
                "input_tokens_actual",
                "output_tokens",
                "reasoning_tokens",
            ):
                bucket[field] += int(_num(row.get(field)))
            bucket["latency_s"] += _num(row.get("latency_s"))
        if row.get("provider"):
            providers[key][str(row["provider"])] += 1
        if row.get("model_served") and row.get("model"):
            served[str(row["model"])][str(row["model_served"])] += 1
        if status == "ok" and row.get("model"):
            by_provider = pricing[str(row["model"])]
            provider = str(row.get("provider") or "unknown")
            _add_pricing(by_provider.setdefault(provider, _pricing_entry()), row)
        error = row.get("error") if isinstance(row.get("error"), Mapping) else {}
        if status == "deadline_exceeded" and error.get("blame") == "model":
            timeouts[str(row.get("model"))] += 1
        if status != "ok":
            failed = ledger_line_failure(row)
            refused = refusal_class(failed) if failed is not None else None
            if refused is not None:
                refusals[key][f"{refused}/{row.get('http_status')}"] += 1
        if row.get("item") and role == "actor":
            items[str(row["item"])][phase] += _num(row.get("cost_total"))
            if unbilled:
                items_unknown[str(row["item"])][phase] += 1
        if row.get("repeat_of"):
            repeats[role] += 1
        wire = row.get("wire")
        if isinstance(wire, Mapping) and wire.get("model"):
            sent = wire.get("provider")
            routing[str(wire["model"])][
                "none" if sent is None else json.dumps(sent, sort_keys=True)
            ] += 1
        elif row.get("model") and row.get("request_sha256") and "wire" not in row:
            routing_unrecorded[str(row["model"])] += 1
        if row.get("protocol_conformant") is False:
            nonconformant += 1
        if row.get("call_id"):
            call_ids.append(str(row["call_id"]))
    for bucket in by.values():
        bucket["cost_total"] = round(bucket["cost_total"], 10)
        bucket["latency_s"] = round(bucket["latency_s"], 3)
    totals = _bucket()
    for bucket in by.values():
        for field, value in bucket.items():
            totals[field] += value
    totals["cost_total"] = round(totals["cost_total"], 10)
    totals["latency_s"] = round(totals["latency_s"], 3)
    lost["cost_total"] = round(lost["cost_total"], 10)
    lost["latency_s"] = round(lost["latency_s"], 3)
    lost["by_role"] = dict(sorted(lost_roles.items()))
    return {
        "ledger": str(path),
        "exists": path.is_file(),
        "events": dict(sorted(events.items())),
        "phases": phases,
        "abandoned": abandoned,
        "statuses": dict(sorted(statuses.items())),
        "by_phase_role": {k: by[k] for k in sorted(by)},
        "totals": totals,
        "providers": {k: dict(providers[k].most_common()) for k in sorted(providers)},
        "model_served": {k: dict(served[k].most_common()) for k in sorted(served)},
        "pricing": {
            model: {p: pricing[model][p] for p in sorted(pricing[model])}
            for model in sorted(pricing)
        },
        "timeouts_model_blamed": dict(sorted(timeouts.items())),
        "items": {
            item: {p: round(c, 10) for p, c in sorted(costs.items())}
            for item, costs in sorted(items.items())
        },
        "items_unknown_cost": {
            item: dict(sorted(counts.items()))
            for item, counts in sorted(items_unknown.items())
        },
        "repeat_of": dict(sorted(repeats.items())),
        "routing": {m: dict(sorted(routing[m].items())) for m in sorted(routing)},
        "routing_unrecorded": dict(sorted(routing_unrecorded.items())),
        "refusals": {k: dict(sorted(refusals[k].items())) for k in sorted(refusals)},
        "protocol_conformant": nonconformant == 0,
        "nonconformant_calls": nonconformant,
        # Calls whose gateway stopped before their final line: their cost is unknown.
        "unfinished_calls": len(started - set(call_ids)),
        "torn_lines": torn,
        "call_ids": call_ids,
        # The calls a learning resume discarded (also counted above).
        "discarded": lost,
    }


# ---------------------------------------------------------------------------
# Every gateway run of a cell id
# ---------------------------------------------------------------------------


def run_id_pattern(cell_id: str) -> re.Pattern[str]:
    """The gateway run ids of ``cell_id``'s attempts: ``<cell_id>-aNN-<instance>``,
    or ``<cell_id>-aNN`` for an attempt started before instance ids."""
    return re.compile(rf"{re.escape(cell_id)}-a(\d{{2,}})(?:-([0-9a-f]+))?")


@dataclass(frozen=True)
class GatewayRun:
    """One gateway run of a cell id, whichever cell directory ran it."""

    run_id: str
    attempt: int
    #: The instance id of the cell directory that ran it (None: before instance ids).
    instance: str | None
    #: The run was put in its test phase, or served a call in it.
    test_phase: bool
    #: Actor calls of its test phase: test items were run and scored (the same evidence
    #: ``run.py``'s ``--new-attempt`` refusal and the audit's ``tested_attempt_replaced``
    #: read).
    test_actor_calls: int
    calls: int
    #: When the gateway admitted the run's first test-phase actor call (its ledger ``ts``),
    #: None when it made none: from then on a test score of the run could be seen. The
    #: audit compares an operator note's time with it.
    first_test_actor_at: str | None = None

    @property
    def tested(self) -> bool:
        return self.test_actor_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt": self.attempt,
            "instance": self.instance,
            "test_phase": self.test_phase,
            "test_actor_calls": self.test_actor_calls,
            "first_test_actor_at": self.first_test_actor_at,
            "calls": self.calls,
        }


def cell_gateway_runs(runs_dir: Path, cell_id: str) -> list[GatewayRun]:
    """Every run of ``cell_id`` in a gateway runs directory, in run id order (blocking).

    A cell started over in a new directory gets a new instance id, so its runs never mix
    with the old directory's; the old runs stay here. The audit compares them with the
    cell directory's attempts: a run of another instance that ran test items is the cell
    run again after its test scores could be seen (H1.3)."""
    pattern = run_id_pattern(cell_id)
    found: list[GatewayRun] = []
    try:
        children = sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []
    except OSError:
        children = []
    for child in children:
        match = pattern.fullmatch(child.name)
        if match is None or not child.is_dir():
            continue
        test_phase = False
        test_actor = calls = 0
        first: tuple[datetime, str] | None = None
        for row in read_jsonl(child / "gateway.jsonl"):
            event = row.get("event")
            if event == "phase":
                test_phase = test_phase or row.get("phase") == "test"
            elif event == "call":
                calls += 1
                if (row.get("run_phase") or row.get("phase")) == "test":
                    test_phase = True
                    if row.get("role") == "actor":
                        test_actor += 1
                        at = _parse_ts(row.get("ts"))
                        if at is not None and (first is None or at < first[0]):
                            first = (at, str(row["ts"]))
        try:
            state = read_json(child / "gateway_state.json")
        except ValueError:
            state = None
        if isinstance(state, dict) and state.get("phase") == "test":
            test_phase = True
        found.append(
            GatewayRun(
                run_id=child.name,
                attempt=int(match.group(1)),
                instance=match.group(2),
                test_phase=test_phase,
                test_actor_calls=test_actor,
                calls=calls,
                first_test_actor_at=first[1] if first else None,
            )
        )
    return found


def _parse_ts(value: Any) -> datetime | None:
    """A ledger ``ts`` (ISO 8601, UTC when it names no zone), None when unreadable."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def call_completed_at(row: Mapping[str, Any]) -> datetime | None:
    """When a final ``call`` line's call ended: its ``completed_at``, or for a line
    written before the gateway recorded it, its ``ts`` (the arrival) plus ``latency_s``.
    None when neither can be read."""
    completed = _parse_ts(row.get("completed_at")) if row.get("completed_at") else None
    if completed is not None:
        return completed
    arrived = _parse_ts(row.get("ts"))
    latency = row.get("latency_s")
    if arrived is None:
        return None
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        return arrived + timedelta(seconds=float(latency))
    return arrived


# ---------------------------------------------------------------------------
# What an attempt learned or was scored despite
# ---------------------------------------------------------------------------

#: The refusal counters the methods snapshot, only when non-zero
#: (``methods/awm.REFUSAL_COUNTERS`` and ``methods/reasoningbank.REFUSAL_COUNTERS``; a
#: test keeps them equal). A missing key is 0.
TOLERATED_COUNTER_KEYS: tuple[str, ...] = (
    "induction_rejected",
    "induction_retries",
    "judge_rejected",
    "extraction_rejected",
)
#: The methods' own learning records under ``attempt-NN/method/`` that
#: :func:`recorded_problems` reads (``methods/awm.INDUCTIONS_FILE``,
#: ``methods/reasoningbank.EXTRACTIONS_FILE``, ``methods/mirix.LEARNING_RECORD_FILE``;
#: this module never imports a method, a test keeps the names equal).
METHOD_INDUCTIONS_FILE = "inductions.jsonl"
METHOD_EXTRACTIONS_FILE = "extractions.jsonl"
METHOD_MIRIX_LEARNING_FILE = "mirix_learning.jsonl"
#: ``recorded_problems`` keys, in the order a report lists them.
RECORDED_PROBLEM_KEYS: tuple[str, ...] = (
    "provider_unserved",
    "upstream_refusal_streak",
    "mirix_batch_accepted",
    "retrieval_degraded",
    "induction_rejected",
    "induction_retries",
    "extraction_rejected",
    "judge_rejected",
    "mirix_gateway_refusal",
    "mirix_rolled_back",
)


def _safe_rows(path: Path) -> list[dict[str, Any]]:
    """:func:`read_jsonl`, and none for a file that cannot be read at all."""
    try:
        return read_jsonl(path)
    except (OSError, ValueError):
        return []


def _safe_json(path: Path) -> Any | None:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _row_failure(row: Mapping[str, Any]) -> Failure | None:
    data = row.get("failure")
    if not isinstance(data, Mapping):
        return None
    try:
        return Failure.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def _unserved(row: Mapping[str, Any]) -> bool:
    return (
        row.get("provider_unserved") is True
        or row.get("end_reason") == PROVIDER_UNSERVED
    )


def _latest_by(
    rows: Iterable[Mapping[str, Any]], key: str, *, scored_only: bool = False
) -> list[Mapping[str, Any]]:
    """The latest row per ``key`` value, in the order each value first appeared."""
    latest: dict[Any, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or (scored_only and row.get("score") is None):
            continue
        if isinstance(value, (dict, list)):
            continue
        latest[value] = row
    return list(latest.values())


def tolerated_failures(paths: AttemptPaths) -> dict[str, Any]:
    """What one attempt learned, or was scored, despite (blocking; the one reader of
    these records for the report and the audit's info; every key is always present).

    * ``learning``: from the latest ``learning.jsonl`` row of each position, the items
      whose actor call the provider refused (``provider_unserved``: position, raw id,
      task uid, HTTP status and refusal class) and those injected with a degraded MIRIX
      retrieval (``retrieval_degraded``);
    * ``test``: the same two lists of raw ids, from the latest scored ``results.jsonl``
      row of each item;
    * ``method``: the refusal counters of the frozen snapshot
      (:data:`TOLERATED_COUNTER_KEYS`, the ones it has) and MIRIX's ``tolerated`` block
      (None without one, or without a frozen snapshot);
    * ``refusals``: ``costs.json`` ``refusals`` (refused calls per phase/role and class);
    * ``accepted_learning_calls``: the MIRIX calls ``attempt.json`` records as lost and
      accepted after ``MIRIX_BATCH_ROLLBACKS`` rollbacks.
    """
    learning_unserved: list[dict[str, Any]] = []
    learning_degraded: list[dict[str, Any]] = []
    for row in _latest_by(_safe_rows(paths.learning), "position"):
        where = {
            "position": row.get("position"),
            "raw_id": row.get("raw_id"),
            "task_uid": row.get("task_uid"),
        }
        if _unserved(row):
            failure = _row_failure(row)
            learning_unserved.append(
                {
                    **where,
                    "status": failure.status if failure else None,
                    "class": refusal_class(failure) if failure else None,
                }
            )
        if row.get("retrieval_degraded") is True:
            learning_degraded.append(where)
    test_rows = _latest_by(_safe_rows(paths.results), "raw_id", scored_only=True)
    snapshot_doc = _safe_json(paths.frozen_snapshot)
    snapshot = (
        snapshot_doc.get("snapshot") if isinstance(snapshot_doc, Mapping) else None
    )
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    counters = snapshot.get("counters")
    counters = counters if isinstance(counters, Mapping) else {}
    mirix = snapshot.get("tolerated")
    costs = _safe_json(paths.costs)
    refusals = costs.get("refusals") if isinstance(costs, Mapping) else None
    state = _safe_json(paths.state)
    accepted = (
        state.get("accepted_learning_calls") if isinstance(state, Mapping) else None
    )
    return {
        "learning": {
            "provider_unserved": learning_unserved,
            "retrieval_degraded": learning_degraded,
        },
        "test": {
            "provider_unserved": [r.get("raw_id") for r in test_rows if _unserved(r)],
            "retrieval_degraded": [
                r.get("raw_id")
                for r in test_rows
                if r.get("retrieval_degraded") is True
            ],
        },
        "method": {
            "counters": {
                key: counters[key]
                for key in TOLERATED_COUNTER_KEYS
                if isinstance(counters.get(key), int) and counters[key]
            },
            "mirix": dict(mirix) if isinstance(mirix, Mapping) else None,
        },
        "refusals": dict(refusals) if isinstance(refusals, Mapping) else {},
        "accepted_learning_calls": list(accepted) if isinstance(accepted, list) else [],
    }


def recorded_problems(paths: AttemptPaths) -> dict[str, int]:
    """How often one attempt recorded a problem instead of stopping on it
    (blocking, never raises; a missing or unreadable file counts 0). Only the non-zero
    :data:`RECORDED_PROBLEM_KEYS` are returned:

    * ``provider_unserved`` / ``upstream_refusal_streak``: ``failures.jsonl`` rows with
      that code; ``mirix_batch_accepted``: rows with recovery ``accept_failed_batch``;
    * ``retrieval_degraded``: ``learning.jsonl`` and ``results.jsonl`` rows flagged so;
    * ``induction_rejected`` / ``induction_retries``: AWM induction rows with
      ``rejected`` / ``retry``; ``extraction_rejected``: RB extraction rows with
      ``refused``; ``judge_rejected``: ``judge_refused`` events of either;
    * ``mirix_gateway_refusal`` / ``mirix_rolled_back``: MIRIX ``tolerated`` events by
      reason.

    The supervisor alerts on these (the cell's highest attempt), the report lists them.
    """
    counts: Counter[str] = Counter()
    for row in _safe_rows(paths.failures):
        code = row.get("code")
        if code == PROVIDER_UNSERVED:
            counts["provider_unserved"] += 1
        elif code == REFUSAL_STREAK:
            counts["upstream_refusal_streak"] += 1
        if row.get("recovery") == Recovery.ACCEPT_FAILED_BATCH.value:
            counts["mirix_batch_accepted"] += 1
    for path in (paths.learning, paths.results):
        counts["retrieval_degraded"] += sum(
            1 for row in _safe_rows(path) if row.get("retrieval_degraded") is True
        )
    method = paths.method_dir
    for row in _safe_rows(method / METHOD_INDUCTIONS_FILE):
        if row.get("event") == "judge_refused":
            counts["judge_rejected"] += 1
            continue
        counts["induction_rejected"] += int(row.get("rejected") is not None)
        counts["induction_retries"] += int(row.get("retry") is not None)
    for row in _safe_rows(method / METHOD_EXTRACTIONS_FILE):
        if row.get("event") == "judge_refused":
            counts["judge_rejected"] += 1
            continue
        counts["extraction_rejected"] += int(row.get("refused") is not None)
    for row in _safe_rows(method / METHOD_MIRIX_LEARNING_FILE):
        if row.get("event") != "tolerated":
            continue
        if row.get("reason") == "gateway_refusal":
            counts["mirix_gateway_refusal"] += 1
        elif row.get("reason") == "rolled_back":
            counts["mirix_rolled_back"] += 1
    return {key: counts[key] for key in RECORDED_PROBLEM_KEYS if counts[key]}
