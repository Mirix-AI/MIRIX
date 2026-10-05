"""Owned child processes: spawn, health wait, stop, and crash-safe cleanup.

Ported from ``skill_main_tables/owned_processes.py`` and ``server.py``. Every
child runs in its own session (process group) and is recorded in a JSONL
registry. :func:`cleanup_owned` later kills only groups that were registered and
whose owner is proven gone:

* **Owner.** A row may name an *owner lock* (:class:`OwnerLock`): a file the harness
  that started the child holds with ``flock`` for as long as the child is its own. The
  kernel drops the lock when that harness dies. A row whose owner lock is still held
  belongs to a live harness and is left alone: no signal, no ``cleaned`` row. The
  gateway registry is shared by every harness of a runs directory, so each gateway row
  carries such a lock (N1). Rows without one rely on a registry-wide lock instead (the
  cell ``server.lock`` checked by :func:`registry_in_use`).
* **Users.** A row may also name a *users lock*, held shared (``LOCK_SH``) by every
  process that uses the child: each live ``run.py`` holds its gateway's for its whole
  lifetime. A child whose users lock is held is never killed, even after its owner died
  (a scheduler that stopped while its workers go on), by any cleanup (X12).
* **Identity.** A registered PID that now belongs to another program (checked through
  the process start time, read with ``LC_ALL=C`` so the check does not depend on the
  harness's locale) is never signalled: it is recorded as ``cleaned`` with reason
  ``pid_reused`` (for example after a reboot) and skipped.
* **Orphans.** When the registered leader is gone, its process group id may still be
  alive: our own orphaned grandchildren, or (after a reboot or PID wrap-around) an
  unrelated program's group. Every child gets a random token in its environment
  (:data:`OWNER_TOKEN_ENV`, inherited by its descendants), recorded in its row. Only
  members whose environment carries that token are killed, one by one; a group whose
  members cannot be proven ours is recorded (``gone``) and never signalled (N2).

:func:`sweep_owned` runs the cleanup over every registry below a runs directory (a
harness-wide orphan sweep at scheduler start).

For the resource guard and the supervisor:
:func:`process_group_rss` sums resident memory per process group from one ``ps`` call,
:func:`registered_process_groups` names the live children of a registry (a cell's MIRIX
server), :func:`live_cell_workers` lists the live ``run.py`` of a runs directory (its
``cell.lock`` held, its ``process.json`` naming the same process), and
:func:`signal_group` signals a process group.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


class OwnedProcessError(RuntimeError):
    """A child failed to start, became unhealthy, or cannot be cleaned up safely."""


#: Environment variable every owned child (and, by inheritance, its descendants) carries:
#: the random token of its registry row. It is how a cleanup proves that a process whose
#: group leader has exited is one of ours before signalling it.
OWNER_TOKEN_ENV = "SKILL_BENCH_OWNER_MARK"
#: How long :meth:`OwnerLock.acquire` retries: another harness's :func:`lock_is_held`
#: probe holds the lock for a moment, which must not look like a live owner.
OWNER_LOCK_WAIT_S = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ends_torn(path: Path) -> bool:
    """True when ``path`` is non-empty and does not end with a newline: a line torn by a
    crash or a full disk mid-append (blocking; a missing file is not torn)."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return False
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) != b"\n"
    except FileNotFoundError:
        return False


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line and fsync it (crash-safe audit trail).

    A torn last line (a crash or ENOSPC mid-append) is ended with a newline first, so
    the new record never merges with the fragment: the fragment stays one unparsable line
    that readers skip, and every record after it stays readable (X15).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    if ends_torn(path):
        line = "\n" + line
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


#: ``ps`` prints ``lstart`` in the current locale ("Mon Sep 28" vs "lun. 28 sept."); the
#: identity check must not change when a harness resumes under another LANG / LC_ALL.
_C_LOCALE = {"LC_ALL": "C", "LANG": "C", "LC_TIME": "C"}


def process_start_time(pid: int) -> str | None:
    """``ps -o lstart`` of ``pid`` in the C locale (identity against PID reuse), or None if gone."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **_C_LOCALE},
    )
    text = " ".join(result.stdout.split())
    return text if result.returncode == 0 and text else None


def process_group_members(pgid: int) -> list[int]:
    """PIDs whose process group is ``pgid`` (``ps -A``, C locale); empty if none."""
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=,pgid="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **_C_LOCALE},
    )
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == str(pgid) and fields[0].isdigit():
            members.append(int(fields[0]))
    return sorted(members)


#: One ``ps`` call listing every process's pid, process group and resident memory (KiB on
#: macOS and Linux alike).
PS_TABLE_ARGV: tuple[str, ...] = ("ps", "-A", "-o", "pid=,pgid=,rss=")


def parse_ps_table(text: str) -> list[tuple[int, int, int]]:
    """``(pid, pgid, rss KiB)`` rows of :data:`PS_TABLE_ARGV`'s output; other lines skipped."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(f.isdigit() for f in fields):
            rows.append((int(fields[0]), int(fields[1]), int(fields[2])))
    return rows


def process_group_rss(
    pgids: Collection[int], *, ps_output: str | None = None
) -> dict[int, int]:
    """Resident memory in bytes of each process group in ``pgids`` (blocking): the sum over
    its members, from one ``ps`` call (or ``ps_output``, its text). A group with no member
    left is absent."""
    wanted = set(pgids)
    if not wanted:
        return {}
    if ps_output is None:
        result = subprocess.run(
            list(PS_TABLE_ARGV),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **_C_LOCALE},
        )
        ps_output = result.stdout
    totals: dict[int, int] = {}
    for _pid, pgid, rss_kib in parse_ps_table(ps_output):
        if pgid in wanted:
            totals[pgid] = totals.get(pgid, 0) + rss_kib * 1024
    return totals


def process_environment_has(pid: int, entry: str) -> bool:
    """True only when ``entry`` (``NAME=value``) is provably in the environment of ``pid``.

    Linux reads ``/proc/<pid>/environ``; macOS and the BSDs use ``ps -E``. A process whose
    environment cannot be read (another user, a platform binary under SIP, a race with
    its exit) is *not* proven, so it is never signalled.
    """
    if sys.platform.startswith("linux"):
        try:
            data = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            return False
        return entry.encode() in data.split(b"\0")
    result = subprocess.run(
        ["ps", "-E", "-ww", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **_C_LOCALE},
    )
    return result.returncode == 0 and entry in result.stdout.split()


def proven_group_members(pgid: int, token: str | None) -> list[int]:
    """Members of process group ``pgid`` that carry our ``token`` (never anyone else)."""
    if not token:
        return []
    entry = f"{OWNER_TOKEN_ENV}={token}"
    return [
        pid
        for pid in process_group_members(pgid)
        if process_environment_has(pid, entry)
    ]


def _kill_each(pids: Sequence[int]) -> list[int]:
    """SIGKILL each PID (not a group); returns the ones that were still there."""
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
        killed.append(pid)
    return killed


# ---------------------------------------------------------------------------
# Owner locks
# ---------------------------------------------------------------------------


def lock_is_held(path: Path) -> bool:
    """True when a live process holds ``path`` with ``flock`` (a missing file is free).

    The probe takes the lock for a moment and releases it at once; owners acquire with
    :meth:`OwnerLock.acquire`, which retries across such a probe.
    """
    try:
        handle = path.open("r")
    except FileNotFoundError:
        return False
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


class OwnerLock:
    """An exclusive ``flock`` a harness holds while a child process is its own.

    The kernel releases it when the harness exits or dies, however it dies, so a
    cleanup can tell a crashed owner (lock free) from a live one (lock held). Acquire and
    release are blocking file operations: async callers use ``asyncio.to_thread`` for
    :meth:`acquire` (it may wait up to ``wait_s``).
    """

    def __init__(self, path: Path, handle: Any) -> None:
        self.path = path
        self._handle: Any = handle

    @classmethod
    def acquire(
        cls, path: Path, *, wait_s: float = OWNER_LOCK_WAIT_S, shared: bool = False
    ) -> OwnerLock:
        """Take the lock or raise :class:`OwnedProcessError` if a live harness holds it.

        ``shared`` takes a shared lock (``LOCK_SH``): any number of users may hold it at
        once, and :func:`lock_is_held` reports the file held while one of them lives
        (the users of a gateway, X12).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        deadline = time.monotonic() + wait_s
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        while True:
            try:
                fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
                return cls(path, handle)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise OwnedProcessError(
                        f"{path} is held by another live harness process"
                    ) from None
                time.sleep(0.05)
            except BaseException:
                handle.close()
                raise

    @property
    def held(self) -> bool:
        return self._handle is not None

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None


def _owner_lock_path(
    registry: Path, row: Mapping[str, Any], key: str = "owner_lock"
) -> Path | None:
    value = row.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else registry.parent / path


def row_in_use(registry: Path, row: Mapping[str, Any]) -> bool:
    """True when a live process still uses a registered child: some process holds its
    ``users_lock`` (shared, a gateway's live workers, X12). Blocking."""
    users = _owner_lock_path(registry, row, "users_lock")
    return users is not None and lock_is_held(users)


def _row_lock_value(registry: Path, lock: Path) -> str:
    """How a row names its owner lock: relative to the registry's directory when inside it."""
    try:
        return lock.resolve().relative_to(registry.parent.resolve()).as_posix()
    except ValueError:
        return str(lock.resolve())


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing accepts connections on ``host:port``.

    A connect probe rather than a bind test: TIME_WAIT leftovers of a stopped server
    must not block a restart (servers bind with SO_REUSEADDR), while any live
    listener, including one on the wildcard address, must be detected.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return True
        return False


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def tail(path: Path, max_bytes: int = 4000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


@dataclass
class OwnedProcess:
    """A registered child process group."""

    name: str
    popen: subprocess.Popen[bytes]
    started: str
    log_path: Path
    registry: Path
    token: str | None = field(default=None, repr=False)
    _log_fh: Any = field(default=None, repr=False)

    @property
    def pid(self) -> int:
        return self.popen.pid

    @classmethod
    def spawn(
        cls,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        log_path: Path,
        registry: Path,
        owner_lock: Path | None = None,
        users_lock: Path | None = None,
    ) -> OwnedProcess:
        """Start ``argv`` in a new session with stdout/stderr appended to ``log_path`` (blocking; use to_thread).

        ``owner_lock`` names the :class:`OwnerLock` the caller holds for this child's
        lifetime; a cleanup never touches the child while that lock is held.
        ``users_lock`` names a lock the child's users hold shared while they use it (a
        gateway's live ``run.py`` workers, X12); a cleanup never touches the child while
        any of them holds it, even after its owner died.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        log_fh = log_path.open("ab")
        try:
            popen = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env={**env, OWNER_TOKEN_ENV: token},
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_fh.close()
            raise
        started = process_start_time(popen.pid) or ""
        row: dict[str, Any] = {
            "event": "started",
            "name": name,
            "pid": popen.pid,
            "pgid": popen.pid,
            "started": started,
            "token": token,
            "at": _now(),
        }
        if owner_lock is not None:
            row["owner_lock"] = _row_lock_value(registry, owner_lock)
        if users_lock is not None:
            row["users_lock"] = _row_lock_value(registry, users_lock)
        append_jsonl(registry, row)
        return cls(name, popen, started, log_path, registry, token, log_fh)

    def alive(self) -> bool:
        return self.popen.poll() is None

    async def wait_healthy(
        self,
        url: str,
        *,
        timeout_s: float,
        accept: Callable[[httpx.Response], bool] = lambda r: r.status_code == 200,
        interval_s: float = 0.25,
    ) -> httpx.Response:
        """Poll ``url`` until ``accept`` holds; fail early if the child exits."""
        deadline = time.monotonic() + timeout_s
        last_error = "no response"
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            while time.monotonic() < deadline:
                if not self.alive():
                    raise OwnedProcessError(
                        f"{self.name} exited with code {self.popen.returncode} before becoming healthy; "
                        f"log tail:\n{tail(self.log_path)}"
                    )
                try:
                    response = await client.get(url)
                    if accept(response) and self.alive():
                        return response
                    last_error = f"HTTP {response.status_code}"
                except httpx.HTTPError as exc:
                    last_error = type(exc).__name__
                await asyncio.sleep(interval_s)
        raise OwnedProcessError(
            f"{self.name} not healthy at {url} after {timeout_s:.0f}s ({last_error})"
        )

    async def stop(self, grace_s: float = 15.0) -> int | None:
        """SIGTERM the process group, SIGKILL after ``grace_s``; record the stop."""
        if self.alive():
            _signal_group(self.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.popen.wait), timeout=grace_s
                )
            except asyncio.TimeoutError:
                _signal_group(self.pid, signal.SIGKILL)
                await asyncio.to_thread(self.popen.wait)
        elif await asyncio.to_thread(process_start_time, self.pid) is None:
            # The leader is gone; kill the members of its group that carry our token
            # (orphaned grandchildren), never the group as a whole.
            members = await asyncio.to_thread(
                proven_group_members, self.pid, self.token
            )
            await asyncio.to_thread(_kill_each, members)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        await asyncio.to_thread(
            append_jsonl,
            self.registry,
            {
                "event": "stopped",
                "name": self.name,
                "pid": self.pid,
                "returncode": self.popen.returncode,
                "at": _now(),
            },
        )
        return self.popen.returncode


def signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to process group ``pgid``; False when the group is gone or not ours."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    signal_group(pgid, sig)


def _registry_rows(registry: Path) -> dict[int, dict[str, Any]]:
    """The ``started`` rows of a registry that no later row settled, by pid (blocking)."""
    active: dict[int, dict[str, Any]] = {}
    try:
        text = registry.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return active
    for line in text.split("\n"):  # "\n" only: U+2028 stays inside a record (P15-G1)
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a torn line after a crash (X15)
        if not isinstance(row, dict) or "pid" not in row:
            continue
        if row.get("event") == "started":
            active[int(row["pid"])] = row
        else:
            active.pop(int(row["pid"]), None)
    return active


def registered_process_groups(registry: Path) -> list[int]:
    """Process groups of the children a registry still lists as running (blocking): a
    cell's MIRIX server, for the resource guard's per-cell memory. Rows are not checked
    against PID reuse (one ``ps`` per row); a stale row can at worst count a stranger's
    memory against its cell until the next cleanup settles it."""
    return sorted(
        {int(row.get("pgid", pid)) for pid, row in _registry_rows(registry).items()}
    )


def live_cell_workers(runs_dir: Path) -> list[tuple[str, int]]:
    """``(cell_id, pid)`` of every live ``run.py`` of ``runs_dir`` (blocking): its cell
    directory's ``cell.lock`` is held and its ``process.json`` names a running process
    with the recorded start time (never a reused PID). The supervisor's drain signals
    exactly these."""
    found: list[tuple[str, int]] = []
    if not runs_dir.is_dir():
        return found
    for child in sorted(runs_dir.iterdir()):
        lock = child / "cell.lock"
        if not lock.is_file() or not lock_is_held(lock):
            continue
        try:
            info = json.loads((child / "process.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(info, dict):
            continue
        pid, started = info.get("pid"), info.get("started")
        if not isinstance(pid, int) or isinstance(pid, bool) or not started:
            continue
        if process_start_time(pid) == " ".join(str(started).split()):
            found.append((child.name, pid))
    return found


def cleanup_owned(
    registry: Path, *, held_locks: Collection[OwnerLock] = ()
) -> list[int]:
    """Kill registered process groups whose owner is gone (after a harness crash).

    * A row whose owner lock is held belongs to a live harness: skipped, nothing written.
      ``held_locks`` are the caller's own locks: a row naming one of them was left by a
      previous owner that has died (the caller could not have acquired it otherwise),
      so it is cleaned like any row whose lock is free.
    * A row whose ``users_lock`` is held is still used by a live process (a gateway whose
      scheduler died while its ``run.py`` workers go on, X12): skipped, nothing written,
      whoever asks for the cleanup and whatever its own locks.
    * A registered PID that now belongs to a different process (a reboot, or plain PID
      reuse) or that does not lead its own group is never signalled (``pid_reused``,
      ``not_group_leader``).
    * A leader that is gone (``gone``): only members of its group that carry the row's
      token are killed, one by one (``orphans_killed``); anything unproven is left alone.

    Every row it settles gets a ``cleaned`` row naming the reason, so a stale registry
    cannot block its cell forever. Returns the leader PIDs that were running and killed.
    """
    if not registry.exists():
        return []
    active = _registry_rows(registry)
    mine = {lock.path.resolve() for lock in held_locks if lock.held}
    killed = []
    for pid, row in active.items():
        lock = _owner_lock_path(registry, row)
        if lock is not None and lock.resolve() not in mine and lock_is_held(lock):
            continue  # its harness is alive: not ours to touch
        if row_in_use(registry, row):
            continue  # live processes still use it (X12): never killed under them
        current = process_start_time(pid)
        record: dict[str, Any] = {}
        reason: str
        if current is None:
            orphans = _kill_each(
                proven_group_members(int(row.get("pgid", pid)), row.get("token"))
            )
            reason = "orphans_killed" if orphans else "gone"
            if orphans:
                record["orphans"] = orphans
        elif current != " ".join(str(row.get("started") or "").split()):
            reason = "pid_reused"  # another program has this PID now: never signal it
        elif int(row["pgid"]) != pid:
            reason = "not_group_leader"
        else:
            reason = "killed"
            _signal_group(pid, signal.SIGKILL)
            killed.append(pid)
        append_jsonl(
            registry,
            {
                "event": "cleaned",
                "name": row.get("name"),
                "pid": pid,
                "reason": reason,
                **record,
                "at": _now(),
            },
        )
    return killed


def stop_orphaned(registry: Path, *, grace_s: float = 15.0) -> list[int]:
    """Stop, gently, the registered children nobody owns or uses any more (blocking).

    Like :func:`cleanup_owned`, but a leader that is provably still the registered process
    (same start time, leading its own group) whose owner lock and users lock are both free
    gets ``SIGTERM`` first and ``grace_s`` to exit (a gateway flushes its ledger); then
    :func:`cleanup_owned` settles every row (``SIGKILL`` for what is left). Used after the
    supervisor's drain, when the last worker has released a gateway that a stopped
    scheduler launched. Returns the pids that got ``SIGTERM``.
    """
    termed: list[int] = []
    for pid, row in _registry_rows(registry).items():
        lock = _owner_lock_path(registry, row)
        if lock is not None and lock_is_held(lock):
            continue
        if row_in_use(registry, row):
            continue
        current = process_start_time(pid)
        if current is None or current != " ".join(
            str(row.get("started") or "").split()
        ):
            continue
        if int(row.get("pgid", pid)) != pid:
            continue
        if signal_group(pid, signal.SIGTERM):
            termed.append(pid)
    deadline = time.monotonic() + grace_s
    while termed and time.monotonic() < deadline:
        if all(process_start_time(pid) is None for pid in termed):
            break
        time.sleep(0.1)
    cleanup_owned(registry)
    return termed


def registry_in_use(registry: Path) -> bool:
    """True when a live harness holds the cell lock next to ``registry`` (``server.lock``)."""
    return lock_is_held(registry.parent / "server.lock")


def sweep_owned(
    runs_dir: Path,
    *,
    name: str = "owned_processes.jsonl",
    skip_dirs: Collection[str] = (),
) -> dict[str, list[int]]:
    """Clean every owned-process registry below ``runs_dir`` (H8 orphan sweep; blocking).

    Run by the scheduler at start, while it holds the scheduler lock: MIRIX servers of
    crashed cells and a crashed harness's gateway otherwise keep running (and spending)
    until the same cell restarts. Registries whose cell lock is held by a live harness
    are skipped, and so are rows whose own owner lock is held (the gateways of other
    live harnesses) or whose users lock is held (a gateway live workers still use).
    Registries in a directory named in ``skip_dirs`` (relative to ``runs_dir``, e.g.
    ``_gateway`` for a scheduler that only attaches to a gateway) are left alone.
    Returns the killed PIDs per registry (relative path).
    """
    killed: dict[str, list[int]] = {}
    if not runs_dir.is_dir():
        return killed
    skipped = {(runs_dir / d).resolve() for d in skip_dirs}
    for registry in sorted(runs_dir.rglob(name)):
        if registry.parent.resolve() in skipped:
            continue
        if registry_in_use(registry):
            continue
        pids = cleanup_owned(registry)
        if pids:
            killed[registry.relative_to(runs_dir).as_posix()] = pids
    return killed
