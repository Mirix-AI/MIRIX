"""Freeze the protocol fingerprint, or check the tree against the frozen value.

Usage (from the repository root)::

    python -m evals.skill_bench.tools.freeze_protocol --check             # exit 1 on drift
    python -m evals.skill_bench.tools.freeze_protocol --reason "why"      # (re)freeze
    python -m evals.skill_bench.tools.freeze_protocol --backfill-constants
    python -m evals.skill_bench.tools.freeze_protocol --backfill-model-slugs

The frozen value lives in ``config/protocol_frozen.json`` (``protocol.FROZEN_PROTOCOL_JSON``;
every reader and writer not given a path uses ``protocol.frozen_record_path()``, which the
``SKILL_BENCH_FROZEN_RECORD`` environment variable can point at a copy, as the tests do):
the fingerprint, the protocol constants (per key, ``constants``, and their sha256,
``constants_sha256``), the sha256 of every fingerprinted file, when it was frozen, the git
commit it was frozen from, and why. That file is not fingerprinted itself.

``run.py`` freezes the tree automatically (:func:`auto_freeze`) when a new attempt, a
learning resume or a test resume starts under a tree that is not the frozen value: once,
under the record's lock, with a reason naming the cell, the attempt, the commit and the
files and constant keys that changed (the record's ``auto_freeze`` block keeps them). A
tree that differs from the record is therefore no error: ``--check`` is a query (exit 1
on drift), and the next launch freezes it. Every writer
(:func:`freeze`, :func:`backfill_constants`, :func:`auto_freeze`) holds
:func:`record_lock`, so two writers never interleave; a missing or malformed record is
never recreated automatically (that would drop its history).

Re-freezing is a protocol change. A re-freeze that changes the fingerprint needs
``--reason``; the previous value is kept in ``history`` with its constants (per key and
hash) and file map, so it stays a known version (``protocol.known_versions``) and the
audit and report can name the files and the constant keys that changed between two
versions (``protocol.version_diff``).

``--backfill-constants`` adds the per-key constants to a record frozen before it kept them
(such a record compared the constants as one hash, so removing a method from
``protocol.METHODS`` could not be told from a change every cell ran under).
It changes nothing else, and only when the tree is exactly the frozen value: its
fingerprint equals the record's, and its constants hash to the record's
``constants_sha256``. History entries stay as they are: an entry that shares the
backfilled hash reads its constants from the backfilled record
(``protocol.version_diff``), one that does not keeps "not recorded".
Versions are recorded and reported per cell, not enforced across cells (so a method
fixed after its run re-runs only its own cells): ``audit.py`` passes a cell under the
frozen value, warns about a cell under an earlier one (``protocol_version``) and fails a
cell under a fingerprint no freeze produced.

That warning is all a cell under the earlier value gets only while the re-freeze changed
nothing the audit checks against this tree: its per-call checks read this tree's
constants (gateway settings, reasoning effort, provider policy) and it rebuilds the
first prompts from this tree's environment templates and splits. A re-freeze that
changes a ``protocol.py`` constant other than ``protocol.COMPOSITION_CONSTANTS`` (the
method list) or a fingerprinted file outside ``methods/prompts/``
(``protocol.unauditable_changes``) leaves every cell under the old value failing those
checks as not re-verified under their own version (warnings: the audit's
``protocol_version_unauditable`` summary lists which checks and why): audit such cells
from a checkout of their own protocol to re-verify them. The tool prints a warning when a
re-freeze does that, naming how to find that checkout
(``protocol.version_checkout_hint``: the record's ``git`` is the commit a value was
frozen *from*, before the freeze wrote the record, not a commit that holds it).

Every freeze also keeps, per sha256 of ``config/models.json``, the ``canonical_slug`` of
each model (``model_slugs``, carried over from the replaced record; ``--backfill-model-slugs``
adds the frozen file's to a record written before): the audit judges which model family
OpenRouter may serve a cell's model as against the slugs of the cell's own version, so an
edit of models.json does not turn the model-substitution check into a warning for every
earlier cell.

``--check`` fails too when the record's fingerprint is the tree's but its constants
hash, per-key constants or file map are not (``protocol.protocol_drift``,
``record_problems``): the fingerprint covers them, so the record was edited by hand.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol

SCHEMA_VERSION = 1
#: What a ``history`` entry keeps of the record it replaced.
HISTORY_KEYS: tuple[str, ...] = (
    "fingerprint",
    "constants_sha256",
    "constants",
    "files",
    "frozen_at",
    "git",
    "reason",
)
#: How long a writer waits for another one to release the record (:func:`record_lock`); a
#: freeze holds it for milliseconds.
FREEZE_LOCK_WAIT_S = 60.0
FREEZE_LOCK_POLL_S = 0.05
#: How many file names an automatic freeze's reason lists per kind (then "+N more").
REASON_FILES_SHOWN = 10


class FreezeLockError(RuntimeError):
    """The frozen record's lock could not be taken (nothing was written)."""


class FreezeLockTimeout(FreezeLockError):
    """Another writer held the frozen record's lock for longer than the wait."""


class FreezeLockUnavailable(FreezeLockError):
    """The filesystem that holds the frozen record supports no lock this tool can take."""


def lock_file_path(path: Path) -> Path:
    """The lock file beside the frozen record ``path`` that :func:`record_lock` uses
    where the record's directory cannot be locked (NFS)."""
    path = Path(path).absolute()
    return path.with_name(f".{path.name}.lock")


def _flock_until(fd: int, what: Path, wait_s: float, deadline: float) -> None:
    """``flock`` ``fd`` exclusively, retrying every :data:`FREEZE_LOCK_POLL_S` until
    ``deadline`` (:class:`FreezeLockTimeout`); any other ``OSError`` propagates."""
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise FreezeLockTimeout(
                    f"the frozen record's lock ({what}) is held by another writer "
                    f"after {wait_s:.1f} s"
                ) from None
            time.sleep(FREEZE_LOCK_POLL_S)


def _take_lock(path: Path, wait_s: float) -> int:
    """A descriptor that holds :func:`record_lock`'s lock on ``path`` (blocking)."""
    directory = Path(path).absolute().parent
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, wait_s)
    fd = os.open(directory, os.O_RDONLY)
    try:
        _flock_until(fd, directory, wait_s, deadline)
        return fd
    except FreezeLockTimeout:
        os.close(fd)
        raise
    except OSError as exc:
        # NFS: flock is a whole-file fcntl lock there, which needs a file opened for
        # writing (EBADF on a directory). A filesystem refuses the directory lock to
        # every writer or to none, so all of them fall back to the same lock file.
        os.close(fd)
        first = exc
    lock = lock_file_path(path)
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        raise FreezeLockUnavailable(
            f"the frozen record's directory {directory} cannot be locked ({first}) "
            f"and the lock file {lock} cannot be opened ({exc})"
        ) from exc
    try:
        _flock_until(fd, lock, wait_s, deadline)
        return fd
    except FreezeLockTimeout:
        os.close(fd)
        raise
    except OSError as exc:
        os.close(fd)
        raise FreezeLockUnavailable(
            f"the filesystem of {directory} supports no lock this tool can take: "
            f"neither the directory ({first}) nor the lock file {lock} ({exc}); "
            "freeze from a checkout on a local filesystem"
        ) from exc


@contextmanager
def record_lock(path: Path, wait_s: float | None = None) -> Iterator[None]:
    """Hold the frozen record ``path`` exclusively (blocking): an ``flock`` on the
    directory that holds it, so it serializes writers across the atomic replace of the
    file (a lock on the file's inode would not) and leaves no lock file in the
    repository. Where the directory cannot be locked (NFS) every writer locks
    :func:`lock_file_path` instead, a file that is never replaced. Retries every
    :data:`FREEZE_LOCK_POLL_S` until ``wait_s``, then raises :class:`FreezeLockTimeout`;
    :class:`FreezeLockUnavailable` when neither can be locked (both are
    :class:`FreezeLockError`). Never nest it in one process: two descriptors of one
    process conflict (:func:`auto_freeze` calls :func:`_freeze_locked`, not
    :func:`freeze`)."""
    fd = _take_lock(path, FREEZE_LOCK_WAIT_S if wait_s is None else wait_s)
    try:
        yield
    finally:
        os.close(fd)  # releases the lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git() -> dict[str, Any]:
    from evals.skill_bench.cell_records import git_info

    return git_info()


def record_tracked_by_git(path: Path) -> bool | None:
    """Whether git tracks the frozen record ``path`` (blocking): True when its checkout
    tracks it (an automatic freeze then leaves it changed on this host),
    False when the checkout does not, None when git cannot tell (no git, no checkout)."""
    path = Path(path).absolute()
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(done.returncode)


def tracked_record_note(path: Path, commit: str) -> str:
    """What an automatic freeze into the git-tracked record ``path`` leaves to do (pure;
    the record now differs from ``commit`` on this host, so a ``git pull``
    that brings another record stops."""
    return (
        f"the frozen record {path} is tracked by git and this automatic freeze changed "
        f"it on this host (commit {commit}): commit it with the code, or before a "
        "`git pull` that brings another record set it aside (`git stash`, or copy it "
        "and `git checkout` it) and freeze again after the pull if needed. The cells "
        "keep every version they ran under in versions/SS/version.json, so an audit "
        "of a record that no longer knows a version reads it there (a warning)"
    )


def _write(path: Path, record: Mapping[str, Any]) -> None:
    """Replace the record at ``path`` atomically and durably: the new
    content is fsynced before it replaces the old one, and the directory after, so a
    crash or power loss leaves the old record or the new one, never an empty file (which
    :func:`auto_freeze` would never recreate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    data = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze(
    *,
    reason: str | None,
    root: Path = protocol.SKILL_BENCH_DIR,
    path: Path | None = None,
    now: Callable[[], str] = _now,
    git: Callable[[], Mapping[str, Any]] = _git,
    manifest: Mapping[str, Any] | None = None,
    auto_freeze: Mapping[str, Any] | None = None,
    lock_wait_s: float | None = None,
) -> dict[str, Any]:
    """Write the frozen record for the tree at ``root`` (``manifest``: its
    ``protocol.fingerprint_manifest``, computed when not given) to ``path`` (default
    ``protocol.frozen_record_path()``), under :func:`record_lock`; returns it.

    Re-freezing to the same fingerprint changes nothing (a record without per-key
    constants gets them from :func:`backfill_constants`). Replacing a different frozen
    value needs a ``reason`` and keeps the old value in ``history`` (:data:`HISTORY_KEYS`:
    with its constants, per key and hash, and file map; and its ``auto_freeze`` block
    when an automatic freeze wrote it). ``auto_freeze`` is the block an automatic freeze
    records on the new value (:func:`auto_freeze`).
    """
    path = protocol.frozen_record_path() if path is None else path
    with record_lock(path, lock_wait_s):
        return _freeze_locked(
            reason=reason,
            root=root,
            path=path,
            now=now,
            git=git,
            manifest=manifest,
            auto_freeze=auto_freeze,
        )


def _freeze_locked(
    *,
    reason: str | None,
    root: Path,
    path: Path,
    now: Callable[[], str],
    git: Callable[[], Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    auto_freeze: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """:func:`freeze`'s body; the caller holds :func:`record_lock`."""
    manifest = (
        dict(manifest) if manifest is not None else protocol.fingerprint_manifest(root)
    )
    previous: dict[str, Any] | None = None
    if path.is_file():
        previous = protocol.load_frozen_protocol(path)
        if previous["fingerprint"] == manifest["fingerprint"]:
            return previous
        if not reason:
            raise ValueError(
                f"{path} already freezes {previous['fingerprint']}; replacing it is a "
                "protocol change and needs --reason"
            )
    history = list((previous or {}).get("history") or [])
    if previous is not None:
        # The replaced value keeps its constants and file map:
        # cells run under it stay a known earlier version, and the files and constant
        # keys that changed since can be named. Entries written before this kept only
        # fingerprint, frozen_at, git and reason; before the per-key record, no constants.
        entry = {key: previous.get(key) for key in HISTORY_KEYS}
        if isinstance(previous.get("auto_freeze"), Mapping):
            entry["auto_freeze"] = dict(previous["auto_freeze"])
        history.append(entry)
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "protocol_id": protocol.PROTOCOL_ID,
        "fingerprint": manifest["fingerprint"],
        "constants_sha256": manifest["constants_sha256"],
        "constants": manifest["constants"],
        "files": dict(sorted(manifest["files"].items())),
        "frozen_at": now(),
        "git": dict(git()),
        "reason": reason or "initial freeze",
        "history": history,
        "how_to_refreeze": (
            "python -m evals.skill_bench.tools.freeze_protocol --reason '<why>'"
        ),
    }
    if auto_freeze is not None:
        record["auto_freeze"] = dict(auto_freeze)
    # What each model is requested as under every models.json frozen so far (the
    # audit's model_served check reads a version's own slugs, by its file's hash).
    slugs = {
        **protocol.recorded_model_slugs(previous),
        **protocol.model_slugs_of(manifest, root),
    }
    if slugs:
        record["model_slugs"] = dict(sorted(slugs.items()))
    _write(path, record)
    return record


def backfill_model_slugs(
    *,
    root: Path = protocol.SKILL_BENCH_DIR,
    path: Path | None = None,
    lock_wait_s: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Add the canonical slugs of the frozen ``config/models.json`` to a record frozen
    before it kept them (``model_slugs``, keyed by the file's sha256);
    returns the record and whether it was written. Only the tree's own models.json is
    read, and only when its hash is the one the record's file map gives it (``ValueError``
    otherwise, nothing written). Nothing else changes. Holds :func:`record_lock`."""
    path = protocol.frozen_record_path() if path is None else path
    with record_lock(path, lock_wait_s):
        record = protocol.load_frozen_protocol(path)
        wanted = (record.get("files") or {}).get(protocol.MODELS_JSON_REL)
        found = protocol.model_slugs_record(root)
        if not wanted or wanted not in found:
            raise ValueError(
                f"the tree's {protocol.MODELS_JSON_REL} hashes to "
                f"{', '.join(found)}, not to the frozen {wanted}: its slugs are not the "
                "frozen ones"
            )
        have = protocol.recorded_model_slugs(record)
        if have.get(wanted) == found[wanted]:
            return record, False
        if wanted in have:
            raise ValueError(
                f"{path} already records other slugs for {protocol.MODELS_JSON_REL} "
                f"{wanted}: fix the record by hand"
            )
        slugs = dict(sorted({**have, **found}.items()))
        _write(path, {**record, "model_slugs": slugs})
        return protocol.load_frozen_protocol(path), True


def backfill_constants(
    *,
    root: Path = protocol.SKILL_BENCH_DIR,
    path: Path | None = None,
    lock_wait_s: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Add the tree's protocol constants, per key, to the frozen record at ``path``
    (``constants``); returns the record and whether it was written.

    Refuses (``ValueError``, nothing written) unless the tree is exactly the frozen
    value: the tree's fingerprint equals the record's, and its constants hash to the
    record's ``constants_sha256``. A record that already keeps those constants is left
    as it is; one that keeps others is refused. Nothing else changes: not the
    fingerprint, not the file map, not the history (whose entries stay as they were).
    Holds :func:`record_lock` (``path`` defaults to ``protocol.frozen_record_path()``)."""
    path = protocol.frozen_record_path() if path is None else path
    with record_lock(path, lock_wait_s):
        return _backfill_locked(root=root, path=path)


def _backfill_locked(*, root: Path, path: Path) -> tuple[dict[str, Any], bool]:
    record = protocol.load_frozen_protocol(path)
    manifest = protocol.fingerprint_manifest(root)
    constants = manifest["constants"]
    computed = protocol.constants_sha256(constants)
    if manifest["fingerprint"] != record["fingerprint"]:
        raise ValueError(
            f"the tree's protocol fingerprint {manifest['fingerprint']} is not the "
            f"frozen {record['fingerprint']}: the per-key constants can only be read "
            "from the tree the record was frozen from"
        )
    if computed != record.get("constants_sha256"):
        raise ValueError(
            f"the tree's constants hash to {computed}, not to the recorded "
            f"constants_sha256 {record.get('constants_sha256')}: they are not the "
            "frozen constants"
        )
    existing = record.get("constants")
    if existing is not None:
        if isinstance(existing, Mapping) and (
            protocol.constants_sha256(existing) == computed
        ):
            return record, False
        raise ValueError(
            f"{path} already records per-key constants, and they are not the frozen "
            "ones (they do not hash to its constants_sha256): fix the record by hand"
        )
    _write(path, {**record, "constants": constants})
    written = protocol.load_frozen_protocol(path)
    readback = protocol.recorded_constants(written)
    if (
        written["fingerprint"] != record["fingerprint"]
        or readback is None
        or protocol.constants_sha256(readback) != computed
    ):  # pragma: no cover - _write is atomic; this checks that the file reads back
        raise ValueError(f"{path} does not read back as written")
    return written, True


def refreeze_warning(record: Mapping[str, Any]) -> str | None:
    """The warning for a re-freeze (``record`` as :func:`freeze` wrote it) that changed
    something the audit checks cells of the replaced value against this tree with
    (``protocol.unauditable_changes``; see the module docstring), else None (pure)."""
    history = record.get("history") or ()
    if not history or not isinstance(history[-1], Mapping):
        return None
    previous = history[-1]
    if previous.get("fingerprint") == record.get("fingerprint"):
        return None
    versions = {
        str(entry.get("fingerprint")): dict(entry)
        for entry in (previous, record)
        if isinstance(entry.get("fingerprint"), str)
    }
    diff = protocol.version_diff(
        previous.get("fingerprint"), record.get("fingerprint"), versions
    )
    changes = protocol.unauditable_changes(diff)
    if not changes:
        return None
    # A change limited to some benchmarks' entries and files leaves the cells of the
    # others auditable as they ran; name those.
    benchmarks = protocol.benchmarks_of(
        protocol.recorded_constants(previous), protocol.recorded_constants(record)
    )
    spared = [b for b in benchmarks if not protocol.unauditable_changes(diff, b)]
    return (
        f"warning: this re-freeze changes {', '.join(changes)}. Cells run under the "
        f"replaced value {str(previous.get('fingerprint'))[:12]} stay a known earlier "
        "version, but an audit from this tree cannot re-verify the checks that read "
        "the new constants or templates for them: it lists those checks as not "
        "re-verified under their version (warnings, protocol_version_unauditable)"
        + (
            f"; the cells of {', '.join(spared)} are not affected (only other "
            "benchmarks' entries and files changed)"
            if spared
            else ""
        )
        + ". To re-verify them, audit them from "
        + protocol.version_checkout_hint(
            str(previous.get("fingerprint")), previous.get("git")
        )
    )


def _names(paths: Sequence[str]) -> str:
    """``[a, b, ...]`` with at most :data:`REASON_FILES_SHOWN` names, then ``+N more``."""
    shown = list(paths)[:REASON_FILES_SHOWN]
    more = len(paths) - len(shown)
    return "[" + ", ".join([*shown, *([f"+{more} more"] if more > 0 else [])]) + "]"


def _attempt_text(value: Any) -> str:
    return f"{value:02d}" if isinstance(value, int) else str(value)


def auto_freeze(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    root: Path = protocol.SKILL_BENCH_DIR,
    path: Path | None = None,
    now: Callable[[], str] = _now,
    git: Callable[[], Mapping[str, Any]] = _git,
    lock_wait_s: float | None = None,
    tracked: Callable[[Path], bool | None] | None = None,
) -> dict[str, Any]:
    """Freeze ``manifest`` (the ``protocol.fingerprint_manifest`` a launch runs under)
    unless the record already freezes it: ``run.py`` calls this when a new attempt, a
    learning resume or a test resume starts.
    Blocking; never raises an ``Exception`` (whatever goes wrong is returned as status
    ``error`` and recorded by the caller, and the attempt runs on).

    A record that already freezes the manifest's fingerprint gives ``already_frozen``
    (read without the lock: every writer replaces the file atomically; with a warning
    when the record does not describe the tree, ``protocol.protocol_drift``
    ``record_problems``). Otherwise, under :func:`record_lock` (so two cells starting at
    once freeze once: the second re-reads the record and finds its fingerprint), the
    manifest is frozen with a reason naming the cell (``context``: ``cell_id``,
    ``attempt``, ``why``, ``launch``, ``host``, ``cell_dir``), the commit, the files and
    the constant keys that changed, and whether it goes back to an earlier version
    (``revert_to_known``), kept in the record's ``auto_freeze`` block. A missing or
    malformed record gives ``error`` and is never recreated (that would drop its
    history; the attempt keeps its own ``versions/SS/version.json`` instead).

    A freeze into a record git tracks (``tracked``: :func:`record_tracked_by_git`)
    leaves it changed on this host, which a later ``git pull`` that brings another
    record stops on (``tracked`` None: :func:`record_tracked_by_git`, looked up when
    called): ``record_tracked`` says so and ``note`` says what to do
    (:func:`tracked_record_note`; None otherwise).

    Returns ``{status: already_frozen | frozen | error, record, fingerprint,
    previous_fingerprint, frozen_at, reason, revert_to_known, changed, added, removed,
    constants_changed_keys, warning, error, waited_s, record_tracked, note, commit}``
    (``commit``: the abbreviated commit a freeze was made from)."""
    t0 = time.monotonic()
    result: dict[str, Any] = {
        "status": "error",
        "record": None,
        "fingerprint": None,
        "previous_fingerprint": None,
        "frozen_at": None,
        "reason": None,
        "revert_to_known": False,
        "changed": [],
        "added": [],
        "removed": [],
        "constants_changed_keys": None,
        "warning": None,
        "error": None,
        "waited_s": 0.0,
        "record_tracked": None,
        "note": None,
        "commit": None,
    }
    try:
        path = protocol.frozen_record_path() if path is None else Path(path)
        result["record"] = str(path)
        if not isinstance(manifest, Mapping):
            raise ValueError("incomplete manifest: not a mapping")
        result["fingerprint"] = manifest.get("fingerprint")
        missing = [
            key
            for key in ("fingerprint", "constants_sha256", "constants", "files")
            if manifest.get(key) is None
        ]
        if missing:
            raise ValueError(f"incomplete manifest (no {', '.join(missing)})")
        # The common case, without the lock: the record (replaced atomically by every
        # writer) already freezes this fingerprint.
        if _already_frozen(result, manifest=manifest, root=root, path=path):
            return result
        with record_lock(path, lock_wait_s):
            result["waited_s"] = round(time.monotonic() - t0, 3)
            _auto_freeze_locked(
                result,
                manifest=manifest,
                context=context,
                root=root,
                path=path,
                now=now,
                git=git,
            )
    except Exception as exc:  # noqa: BLE001 - recorded by the caller; the attempt runs
        result.update(status="error", error=f"{type(exc).__name__}: {exc}"[:1000])
        result["waited_s"] = round(time.monotonic() - t0, 3)
    if result["status"] == "frozen":
        try:
            tracks = record_tracked_by_git if tracked is None else tracked
            result["record_tracked"] = tracks(Path(str(result["record"])))
        except Exception:  # noqa: BLE001 - the freeze is done; whether git tracks
            result["record_tracked"] = None  # its record is only advice
        if result["record_tracked"]:
            result["note"] = tracked_record_note(
                Path(str(result["record"])), str(result["commit"] or "unknown")
            )
    return result


def _already_frozen(
    result: dict[str, Any], *, manifest: Mapping[str, Any], root: Path, path: Path
) -> bool:
    """Fill ``result`` as ``already_frozen`` when the record at ``path`` holds the
    manifest's fingerprint (read without the lock); False otherwise (also when it cannot
    be read: the locked path decides)."""
    try:
        record = protocol.load_frozen_protocol(path)
    except (protocol.ProtocolDriftError, ValueError, OSError):
        return False
    if str(record["fingerprint"]) != str(manifest["fingerprint"]):
        return False
    drift = protocol.protocol_drift(record, root, path=path, current=manifest)
    result.update(
        status="already_frozen",
        previous_fingerprint=str(record["fingerprint"]),
        frozen_at=record.get("frozen_at"),
    )
    if drift["record_problems"]:
        result["warning"] = (
            "the frozen record's fingerprint is this tree's, but the record does "
            "not describe the tree (edited by hand?): "
            + "; ".join(drift["record_problems"])
        )[:1000]
    return True


def _auto_freeze_locked(
    result: dict[str, Any],
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    root: Path,
    path: Path,
    now: Callable[[], str],
    git: Callable[[], Mapping[str, Any]],
) -> None:
    """:func:`auto_freeze`'s body under the lock: fills ``result``."""
    try:
        record = protocol.load_frozen_protocol(path)
    except (protocol.ProtocolDriftError, ValueError) as exc:
        result["error"] = (
            f"the frozen record cannot be read, so nothing was frozen (a missing or "
            f"malformed record is never recreated automatically: that would drop its "
            f"history): {exc}"
        )[:1000]
        return
    previous = str(record["fingerprint"])
    fingerprint = str(manifest["fingerprint"])
    if previous == fingerprint:
        # Another writer froze it while this one waited for the lock.
        _already_frozen(result, manifest=manifest, root=root, path=path)
        return
    result["previous_fingerprint"] = previous
    drift = protocol.protocol_drift(record, root, path=path, current=manifest)
    earlier = [
        entry
        for entry in record.get("history") or ()
        if isinstance(entry, Mapping) and entry.get("fingerprint") == fingerprint
    ]
    revert = bool(earlier)
    keys = drift["constants_changed_keys"] if drift["constants_changed"] else []
    if not drift["constants_changed"]:
        constants = "none"
    elif keys is None:
        constants = "changed (keys not recorded)"
    else:
        constants = ", ".join(keys) or "changed (keys not recorded)"
    git_now = dict(git())
    commit = str(git_now.get("commit") or "")[:12] or "unknown"
    result["commit"] = commit
    reason = (
        f"auto-freeze: {context.get('cell_id')} "
        f"attempt {_attempt_text(context.get('attempt'))} ({context.get('why')}), "
        f"commit {commit}"
        + (" with uncommitted changes" if git_now.get("dirty") else "")
        + f"; files changed {_names(drift['changed'])}, added "
        f"{_names(drift['added'])}, removed {_names(drift['removed'])}; "
        f"constants [{constants}]"
        + (
            f"; back to the earlier version frozen at {earlier[-1].get('frozen_at')}"
            if revert
            else ""
        )
    )
    block = {
        "cell_id": context.get("cell_id"),
        "attempt": context.get("attempt"),
        "why": context.get("why"),
        "launch": context.get("launch"),
        "host": context.get("host"),
        "cell_dir": context.get("cell_dir"),
        "previous_fingerprint": previous,
        "revert_to_known": revert,
        "changed": list(drift["changed"]),
        "added": list(drift["added"]),
        "removed": list(drift["removed"]),
        "constants_changed_keys": keys,
    }
    written = _freeze_locked(
        reason=reason,
        root=root,
        path=path,
        now=now,
        git=lambda: git_now,
        manifest=manifest,
        auto_freeze=block,
    )
    result.update(
        status="frozen",
        frozen_at=written.get("frozen_at"),
        reason=reason,
        revert_to_known=revert,
        changed=block["changed"],
        added=block["added"],
        removed=block["removed"],
        constants_changed_keys=keys,
        warning=refreeze_warning(written),
    )


def check(
    root: Path = protocol.SKILL_BENCH_DIR, path: Path | None = None
) -> dict[str, Any]:
    """The drift between the tree and the frozen record (``protocol.protocol_drift``;
    ``path`` defaults to ``protocol.frozen_record_path()``). A query: nothing is
    written."""
    return protocol.protocol_drift(root=root, path=path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="compare the tree with the frozen value"
    )
    parser.add_argument("--reason", help="why the protocol is (re)frozen")
    parser.add_argument(
        "--backfill-constants",
        action="store_true",
        help="add the per-key constants to a record frozen before it kept them (only "
        "when the tree is exactly the frozen value; nothing else changes)",
    )
    parser.add_argument(
        "--backfill-model-slugs",
        action="store_true",
        help="add the canonical slugs of the frozen config/models.json to a record "
        "frozen before it kept them (only when the tree's models.json is the frozen "
        "one; nothing else changes)",
    )
    parser.add_argument("--root", type=Path, default=protocol.SKILL_BENCH_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="the frozen record (default: $SKILL_BENCH_FROZEN_RECORD, else "
        "config/protocol_frozen.json)",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = protocol.frozen_record_path()
    if args.backfill_constants and (args.check or args.reason):
        parser.error("--backfill-constants takes neither --check nor --reason")
    if args.backfill_model_slugs and (
        args.check or args.reason or args.backfill_constants
    ):
        parser.error("--backfill-model-slugs takes no other option but --root/--out")
    if args.backfill_model_slugs:
        try:
            record, written = backfill_model_slugs(root=args.root, path=args.out)
        except (
            ValueError,
            OSError,
            protocol.ProtocolDriftError,
            FreezeLockError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"{'backfilled' if written else 'already records'} the canonical slugs of "
            f"{protocol.MODELS_JSON_REL} "
            f"{record['files'][protocol.MODELS_JSON_REL][:12]} in {args.out}; nothing "
            "else changed"
        )
        return 0
    if args.backfill_constants:
        try:
            record, written = backfill_constants(root=args.root, path=args.out)
        except (
            ValueError,
            OSError,
            protocol.ProtocolDriftError,
            FreezeLockError,
        ) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"{'backfilled' if written else 'already records'} "
            f"{len(record['constants'])} protocol constants per key in {args.out}: the "
            f"tree's fingerprint equals the frozen {record['fingerprint']} and its "
            f"constants hash to the recorded {record['constants_sha256']}; the "
            "fingerprint, file map and history are unchanged"
        )
        return 0
    if args.check:
        try:
            drift = check(args.root, args.out)
        except protocol.ProtocolDriftError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(protocol.describe_drift(drift))
        return 1 if drift["drifted"] else 0
    before = None
    try:
        if args.out.is_file():
            before = protocol.load_frozen_protocol(args.out)["fingerprint"]
        record = freeze(reason=args.reason, root=args.root, path=args.out)
    except (ValueError, OSError, protocol.ProtocolDriftError, FreezeLockError) as exc:
        # A malformed existing record (not JSON, no fingerprint) too: exit 2, no traceback.
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"frozen {record['fingerprint']} ({len(record['files'])} files) in {args.out}"
    )
    history = record.get("history") or ()
    if before != record["fingerprint"] and history:
        previous = history[-1]
        diff = protocol.version_diff(
            previous.get("fingerprint"),
            record["fingerprint"],
            {
                str(entry.get("fingerprint")): dict(entry)
                for entry in (previous, record)
            },
        )
        print(
            f"since {str(previous.get('fingerprint'))[:12]}: "
            + protocol.describe_version_diff(diff)
        )
    warning = refreeze_warning(record) if before != record["fingerprint"] else None
    if warning:
        print(warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
