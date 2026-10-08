"""Pull each host's runs directory into one merged, read-only view (a study may run on
several hosts, split by benchmark).

Usage (from the repository root)::

    python -m evals.skill_bench.tools.sync_runs \\
        --source remote=user@remote:<checkout>/evals/skill_bench/runs/<run> \\
        --source local=evals/skill_bench/runs/<local-run> \\
        --into evals/skill_bench/runs/<view>
    python -m evals.skill_bench.tools.sync_runs --mirror-only \\
        --source remote=... --into ...

A source is ``NAME=PATH`` (a local runs directory) or ``NAME=HOST:PATH`` (over ssh, as
rsync reads it: ``PATH`` relative to the remote home). ``NAME`` is letters, digits, ``_``
and ``-``.

1. **Mirror** each source into ``<into>/_hosts/<NAME>/``, a directory only this tool
   writes, with ``rsync -a --delete-after --delay-updates --itemize-changes`` (plus
   ``-z``, ``--timeout`` and ``-e "ssh -o BatchMode=yes -o ServerAliveInterval=30"`` for
   a remote source). Every option is one macOS's ``openrsync`` (protocol 29, "rsync
   version 2.6.9 compatible") and GNU rsync both accept (:data:`OPENRSYNC_OPTIONS`).

   - The mirror is a faithful copy: a file the source rewrote is copied whatever its
     time (a learning resume restores files from a checkpoint with their older times,
     so ``--update`` would keep the newer, discarded copy), and a file the source moved
     away (``discarded/``) is deleted after the transfer. No ``--partial``: a cut
     transfer never leaves a truncated record in the mirror.
   - Left out: locks (``*.lock``), atomic-write temporaries (``.*.tmp``), checkpoints and
     discarded segments still being written (``*.partial/``), the source's own
     ``_audit/``, ``_report/`` and ``_hosts/``; and, unless ``--with-home`` /
     ``--with-checkpoints``, MIRIX's per-attempt ``HOME`` (``attempt-*/home/``,
     ``attempt-*/discarded/*/home/``) and the learning checkpoints
     (``attempt-*/checkpoints/``, hard links kept with ``-H``). Excluded files already in
     the mirror are never deleted, so a local ``_audit/`` there survives.
   - Then ``_gateway/starts.jsonl`` is copied once more, alone (:func:`starts_argv`):
     a gateway appends its start record before it serves any call, so every ledger line
     in the mirror has its start record there too, even of a gateway that started on
     the host while the main rsync ran (else the audit's fail-closed
     ``gateway_start_missing`` on a healthy cell). Its exit 23 (no such file: no gateway
     started there yet) is fine; any other failure fails the source.
   - rsync's exit 24 (files vanished while a run was writing) is a warning; any other
     failure stops the tool before it touches the view (exit 3).
   - **Removed on purpose.** ``tools/remove_method_data.py`` deletes a
     removed method's cells and gateway runs on a host after writing their spend
     manifest, and lists them in ``<runs>/_removed/index.json``. That index is read
     with the identity (one rsync of that one file for a remote source; exit 23: no
     removals; any other failure: nothing is synced, exit 3). Every entry it lists is
     excluded from the rsync (never copied back, even from a host that still holds
     part of it, and never a deletion for the guard below); the entries of a removal
     in state ``complete`` are then deleted from the mirror (real directories only),
     those of an unfinished one stay there, excluded and unlinked, with a warning. A
     name that is not a plain top-level cell or gateway run of a method the study
     removed (``_blobs``, ``a/b``, a study cell ...) is ignored with a warning, so the
     deletion guard sees it as before. ``_removed/`` itself is mirrored like any record (the report and audit
     read the manifests under ``_hosts/<NAME>/_removed/``), never linked.
   - A sync cut off (Ctrl-C, a kill, a dropped connection) leaves rsync's
     ``--delay-updates`` staging directories (``.~tmp~/``) and the temporary file of the
     file it was receiving (``.<name>.`` + 6 random letters and digits in GNU rsync, 10
     in openrsync) in the mirror. They are no data: never counted as the
     mirror's files or as deletions, and the next sync removes the staging directories
     before it copies (rsync deletes the temporary files itself; GNU rsync keeps
     ``.~tmp~/`` for ever). So one sync of a view runs at a time
     (``<into>/.sync_runs.lock``; another is refused, exit 2); a dry run takes no lock.
   - Each mirror holds its own ``_merged.json`` (``{"mirror_of": spec, ...}``), so
     ``run.py`` and ``schedule.py`` refuse to run cells in it, and this tool refuses it
     as a source. ``_hosts/`` and every mirror are real directories this tool makes: a
     symlink there would write that ``_merged.json`` into what it points to (a live runs
     directory, whose cells ``run.py`` would then refuse), and is refused.

2. **The merged view** in ``<into>`` (unless ``--mirror-only``), built from every mirror
   under ``_hosts/``:

   - one relative symlink per cell directory (``cell.json``) and gateway run directory
     (``gateway.jsonl``) of each mirror (``<entry> -> _hosts/<NAME>/<entry>``), except
     the entries that mirror's own ``_removed/index.json`` lists;
   - ``_blobs/`` as a real directory of per-file relative symlinks
     (``_blobs/<sha256> -> ../_hosts/<NAME>/_blobs/<sha256>``). Blobs are
     content-addressed: the same name from two hosts must have the same size;
   - ``_gateway``, ``_schedule`` and ``_supervise`` stay under ``_hosts/<NAME>/``, where
     the readers look for them (the audit's gateway start records, the report's spend);
   - links into ``_hosts/`` that no mirror backs any more are removed. Nothing else in
     ``<into>`` is touched (``_audit/`` and ``_report/`` are the local tools' output).

   ``summarize_learning``, ``audit`` and ``report`` then run over ``<into>``; they find
   every ledger relative to it (``cell_records.resolve_ledger_path``).

3. **Conflicts.** The same entry from two mirrors stops the tool before it touches the
   view, unless both trees are identical, or every copy is a cell directory whose
   ``cell.json`` ``instance`` is readable and differs from the others' (independent
   cells of one id, e.g. a rerun on another host). Those are all linked: the
   copy the view already links keeps the plain name (with no link yet, the earliest
   ``cell.json`` ``created_at``; ties: host name order), every other copy is linked as
   ``<entry>@<host>``; the readers group them by ``cell.json`` cell id (the newer
   complete one is reported, the other listed as superseded, both copies' money
   counted). A copied cell that diverged (one instance), an unreadable ``cell.json``,
   gateway run directories and a blob name with two sizes still stop the tool
   (sync-faithful). They are listed and the exit status is 1.

4. ``<into>/_merged.json`` (:data:`cell_records.MERGED_MARKER`) records every source
   (its spec, when it was synced, rsync's version and exit status, the starts copy's
   exit and time (``starts_exit``, ``starts_synced_at``), the files transferred and
   deleted, the staging directories removed, what it removed on purpose
   (``removed_from_mirror``: the entries this sync deleted from the mirror;
   ``removed_from_mirror_at``: every entry any sync deleted from it -> when, carried
   from one sync to the next; ``removed_manifests``, ``removed_problems``), the
   mirror's files and bytes), the
   view's ``entries`` (link name -> host), ``links`` (link name -> entry name where they
   differ), the entries identical on several hosts (``duplicates``), the independent
   cells of one id (``cells_on_several_hosts``: ``{entry: {host: link}}``), what each
   host removed on purpose (``removed_on_purpose``: ``{host: {entry:
   '<method>/<removal_id>/manifest.json'}}``), the conflicts, when the view was built
   (``view_built_at``) and how the last sync ended (``last_sync``: ``view``,
   ``mirror-only``, ``conflicts`` or ``rsync failed``). ``run.py`` and ``schedule.py``
   refuse a directory holding it: cells never run in a merged view. ``audit.py`` and
   ``report.py`` show what it says (:func:`merged_view_status`): when
   the view was built and each host synced, and why it is not final.

``--dry-run`` runs ``rsync -n`` and changes nothing. ``--checksum`` compares contents,
not sizes and times (``rsync -c``; slower, for the final sync).

**Deletion guard.** A mirror may be the only copy of a host's cells
once the host is gone, and ``--delete-after`` would empty it from an emptied, re-created
or mistyped source, or pass on a mistyped ``rm`` on the host. So before anything is
copied:

- every source must be the runs directory its mirror was made from: its identity is the
  first gateway start in its ``_gateway/starts.jsonl`` (``start_id``; read over ssh for a
  remote source), kept in ``_merged.json`` (``sources.<NAME>.first_start``). A source
  without that file, or with another first start, is refused while its mirror holds
  files; ``--allow-other-source`` lets the mirror follow it from then on;
- a sync whose dry run (``rsync -n``) would delete a whole top-level entry of a mirror (a
  cell, a gateway run, ``_gateway/``, ``_schedule/`` ...; hidden files aside), a record
  anywhere in it (:data:`PROTECTED_RECORDS`), or more than half of its files (and more
  than :data:`DELETE_GUARD_MIN_FILES`) is refused; ``--allow-deletes`` lets it through.
  It never lifts the identity check: following another runs directory
  takes both flags. Only the entries the source's own ``_removed/index.json`` lists
  (removed on purpose, above) are left out of that dry run; every other deletion is
  guarded as before.

The tool refuses (exit 2) an ``<into>`` that is not empty and not a merged view, that
holds a real cell or gateway directory, or that is inside a cell, a gateway run, another
view or a mirror (its path resolved); a symlink as ``<into>/_hosts`` or as a mirror;
a local source that is not a directory, is a merged view itself, or
overlaps ``<into>`` (paths resolved: an ``<into>`` linked into a source too); bad or
repeated source names; a source the deletion guard stops; and a sync while another sync
of the same view runs.

Exit status: 0 synced; 1 conflicts; 2 refused; 3 rsync failed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.skill_bench import study
from evals.skill_bench.cell_records import (
    HOSTS_DIR,
    MERGED_MARKER,
    NON_CELL_DIRS,
    now,
    read_json,
    write_json_atomic,
)

EXIT_OK = 0
EXIT_CONFLICT = 1
EXIT_REFUSED = 2
EXIT_RSYNC = 3

MERGED_SCHEMA = 1
SOURCE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
#: ``HOST:PATH`` (rsync's remote syntax: no ``/`` before the first ``:``).
_REMOTE = re.compile(r"^(?P<host>[^/:]+):(?P<path>.*)$")
BLOB_NAME = re.compile(r"^[0-9a-f]{64}$")
BLOBS_DIR = "_blobs"
#: rsync exit status: some source files vanished during the transfer (a live run).
RSYNC_VANISHED = 24
RSYNC_TIMEOUT_S = 600
SSH_COMMAND = "ssh -o BatchMode=yes -o ServerAliveInterval=30"
#: rsync exit status: a requested file does not exist (among other partial transfers).
RSYNC_PARTIAL = 23
#: A source runs directory's identity: its first gateway start. Copied once
#: more after everything else (:func:`starts_argv`), so every ledger line in a mirror has
#: its gateway's start record there too.
STARTS_FILE = "_gateway/starts.jsonl"
#: What ``tools/remove_method_data.py`` writes in a runs directory
#: (``remove_method_data.REMOVED_DIR`` / ``INDEX_FILE``): the removals of a removed
#: method's data, with the top-level entries each deleted on purpose.
REMOVED_DIR = "_removed"
REMOVED_INDEX = f"{REMOVED_DIR}/index.json"
#: A sync may not delete more than this share of a mirror's files, when that is more
#: than DELETE_GUARD_MIN_FILES files, without --allow-deletes.
DELETE_GUARD_FRACTION = 0.5
DELETE_GUARD_MIN_FILES = 50
#: Records a sync may not delete from a mirror, wherever they are, without
#: --allow-deletes: a cell's definition, status and published rows, an
#: attempt's record and rows, a gateway run's ledger.
PROTECTED_RECORDS = frozenset(
    {
        "cell.json",
        "status.json",
        "results.jsonl",
        "learning.jsonl",
        "attempt.json",
        "gateway.jsonl",
    }
)
#: The --delay-updates staging directory of openrsync and GNU rsync: a sync cut off
#: leaves one, with the files it had received, in every directory it was updating.
RSYNC_STAGING_DIR = ".~tmp~"
#: rsync's temporary file of a file being received: "." + its name + "." + 6 random
#: letters and digits (GNU rsync's mkstemp) or 10 (openrsync), left by a sync cut off.
#: The harness's own temporaries end in ".tmp" or ".link".
RSYNC_TEMP_FILE = re.compile(r"^\..+\.(?:[A-Za-z0-9]{6}|[A-Za-z0-9]{10})$")
#: Held by a sync for its whole run, in ``<into>``: one sync of a view at a time.
LOCK_FILE = ".sync_runs.lock"

#: Always left out of a mirror (anchored ones only at the runs directory's top).
EXCLUDES: tuple[str, ...] = (
    "*.lock",
    ".*.tmp",
    "*.partial/",
    "/_audit/",
    "/_report/",
    f"/{HOSTS_DIR}/",
    f"/{MERGED_MARKER}",
)
#: MIRIX's per-attempt HOME and what a learning resume moved aside of it.
HOME_EXCLUDES: tuple[str, ...] = ("attempt-*/home/", "attempt-*/discarded/*/home/")
CHECKPOINT_EXCLUDES: tuple[str, ...] = ("attempt-*/checkpoints/",)

#: The options this tool passes, all in ``openrsync``'s usage (macOS 15+) and GNU rsync's.
OPENRSYNC_OPTIONS = frozenset(
    {
        "-a",
        "-z",
        "-n",
        "-c",
        "-H",
        "-e",
        "--delete-after",
        "--delay-updates",
        "--itemize-changes",
        "--exclude",
        "--timeout",
        "--ignore-errors",
    }
)


class Refused(Exception):
    """The tool will not run as asked (exit 2)."""


@dataclass(frozen=True)
class Source:
    name: str
    spec: str
    host: str | None  # None: a local path
    path: str

    @property
    def rsync_arg(self) -> str:
        path = self.path.rstrip("/") or "/"
        path = path if path.endswith("/") else path + "/"
        return f"{self.host}:{path}" if self.host else path


def parse_source(text: str) -> Source:
    name, sep, spec = text.partition("=")
    if not sep or not spec:
        raise Refused(f"--source {text!r}: expected NAME=PATH or NAME=HOST:PATH")
    if not SOURCE_NAME.match(name):
        raise Refused(
            f"--source {text!r}: the name must be letters, digits, '_' or '-'"
        )
    if spec.startswith("-"):
        raise Refused(f"--source {text!r}: a path or host may not start with '-'")
    remote = _REMOTE.match(spec)
    if remote:
        if not remote["path"]:
            raise Refused(f"--source {text!r}: no path after the host")
        return Source(name, spec, remote["host"], remote["path"])
    return Source(name, spec, None, spec)


def rsync_argv(
    source: Source,
    mirror: Path,
    *,
    rsync: str = "rsync",
    dry_run: bool = False,
    with_home: bool = False,
    with_checkpoints: bool = False,
    checksum: bool = False,
    ignore_errors: bool = False,
    extra_excludes: Iterable[str] = (),
) -> list[str]:
    """The rsync command that brings ``mirror`` up to ``source``. ``ignore_errors``
    (the deletion guard's dry run only): list the deletions even after an I/O error.
    ``extra_excludes``: top-level entries the source removed on purpose
    (:func:`source_removed`), never copied again and never deleted by rsync."""
    argv = [rsync, "-a", "--delete-after", "--delay-updates", "--itemize-changes"]
    if dry_run:
        argv.append("-n")
    if ignore_errors:
        argv.append("--ignore-errors")
    if checksum:
        argv.append("-c")
    if with_checkpoints:
        argv.append("-H")  # checkpoints hard-link the method's objects
    if source.host:
        argv += ["-z", f"--timeout={RSYNC_TIMEOUT_S}", "-e", SSH_COMMAND]
    patterns = list(EXCLUDES)
    if not with_home:
        patterns += HOME_EXCLUDES
    if not with_checkpoints:
        patterns += CHECKPOINT_EXCLUDES
    patterns += [f"/{entry}/" for entry in sorted(set(extra_excludes))]
    argv += [f"--exclude={p}" for p in patterns]
    argv += [source.rsync_arg, f"{mirror}/"]
    return argv


def starts_argv(source: Source, mirror: Path, *, rsync: str = "rsync") -> list[str]:
    """The rsync command that copies the source's ``_gateway/starts.jsonl`` alone into
    the mirror's ``_gateway/``, after the main rsync. A gateway appends its
    start record (fsync'd) before it serves any call, so the copy made last holds the
    start of every ledger line the main rsync brought, even of a gateway that started
    while it ran (the audit's fail-closed ``gateway_start_missing``)."""
    argv = [rsync, "-a"]
    if source.host:
        argv += ["-z", f"--timeout={RSYNC_TIMEOUT_S}", "-e", SSH_COMMAND]
    argv += [
        f"{source.rsync_arg}{STARTS_FILE}",
        f"{mirror}/{Path(STARTS_FILE).parent}/",
    ]
    return argv


def rsync_version(rsync: str = "rsync") -> str | None:
    try:
        proc = subprocess.run(
            [rsync, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in (proc.stdout or proc.stderr).splitlines()]
    return " / ".join(line for line in lines[:2] if line) or None


def is_rsync_leftover(path: str) -> bool:
    """Whether ``path`` (relative, as rsync itemizes it: a directory ends in ``/``) is
    what a sync cut off left behind: in a ``.~tmp~/`` staging directory
    (or that directory), or rsync's temporary file of a file being received."""
    parts = path.rstrip("/").split("/")
    if RSYNC_STAGING_DIR in parts:
        return True
    return not path.endswith("/") and bool(RSYNC_TEMP_FILE.match(parts[-1]))


@dataclass
class Itemized:
    transferred: list[str] = field(default_factory=list)
    #: files (and symlinks) deleted; directories and leftovers are not files
    deleted: list[str] = field(default_factory=list)
    #: directories deleted (``path/``)
    deleted_dirs: list[str] = field(default_factory=list)
    #: deletions of what a sync cut off left behind (:func:`is_rsync_leftover`)
    leftovers: list[str] = field(default_factory=list)


def parse_itemized(output: str) -> Itemized:
    """Files rsync sent (``>f``/``<f``/``cf``) and deleted (``*deleting``), from
    ``--itemize-changes`` (both rsyncs print ``YXcstpoguax path``; openrsync
    ``*deleting path``, GNU rsync ``*deleting   path``; a directory ends in ``/``).
    Directories and a cut-off sync's leftovers are kept apart from the deleted files,
    so the deletion guard never counts them as files."""
    items = Itemized()
    for line in output.splitlines():
        if line.startswith("*deleting"):
            path = line.split(None, 1)[1].strip() if " " in line else ""
            if is_rsync_leftover(path):
                items.leftovers.append(path)
            elif path.endswith("/"):
                items.deleted_dirs.append(path)
            else:
                items.deleted.append(path)
        elif len(line) > 2 and line[0] in "<>c" and line[1] == "f":
            items.transferred.append(
                line.split(None, 1)[1].strip() if " " in line else ""
            )
    return items


def _walk(root: Path) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` (symlinks not followed) without a cut-off sync's leftovers: its
    staging directories are not entered, its temporary files not listed."""
    for directory, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != RSYNC_STAGING_DIR]
        yield directory, dirs, [n for n in names if not RSYNC_TEMP_FILE.match(n)]


def tree_size(root: Path) -> tuple[int, int]:
    """Regular files and their bytes under ``root`` (symlinks not followed; a cut-off
    sync's leftovers are not the mirror's files)."""
    files = size = 0
    for directory, _dirs, names in _walk(root):
        for name in names:
            try:
                st = os.lstat(os.path.join(directory, name))
            except OSError:
                continue
            if not os.path.islink(os.path.join(directory, name)):
                files += 1
                size += st.st_size
    return files, size


def remove_staging_dirs(mirror: Path) -> int:
    """Remove the ``.~tmp~/`` staging directories a sync cut off left in ``mirror``,
    just before a sync of it copies from its source again: rsync's
    copies of files whose older version is still in place. openrsync would delete them
    itself, GNU rsync (which protects its staging directory) never. Returns how many."""
    removed = 0
    for directory, dirs, _names in os.walk(mirror):
        if RSYNC_STAGING_DIR in dirs:
            dirs.remove(RSYNC_STAGING_DIR)
            staging = os.path.join(directory, RSYNC_STAGING_DIR)
            if not os.path.islink(staging):
                shutil.rmtree(staging)
                removed += 1
    return removed


def tree_digest(root: Path) -> str:
    """One hash of a tree's relative paths, file contents and symlink targets (without a
    cut-off sync's leftovers)."""
    digest = hashlib.sha256()
    for directory, dirs, names in _walk(root):
        dirs.sort()
        rel_dir = os.path.relpath(directory, root)
        for name in sorted(
            names + [d for d in dirs if os.path.islink(os.path.join(directory, d))]
        ):
            path = os.path.join(directory, name)
            digest.update(os.path.join(rel_dir, name).encode() + b"\0")
            if os.path.islink(path):
                digest.update(b"L" + os.readlink(path).encode() + b"\0")
                continue
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _is_real_run_entry(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.is_symlink()
        and ((path / "cell.json").is_file() or (path / "gateway.jsonl").is_file())
    )


def check_into(into: Path) -> None:
    """``into`` is absent, empty or a merged view, and holds no directory a run wrote."""
    if into.exists() and not into.is_dir():
        raise Refused(f"--into {into} is not a directory")
    if not into.is_dir():
        return
    entries = list(into.iterdir())
    if entries and not (into / MERGED_MARKER).is_file():
        names = {e.name for e in entries if not e.name.startswith(".")}
        if names - {HOSTS_DIR}:
            raise Refused(
                f"--into {into} is neither empty nor a merged view (no {MERGED_MARKER}):"
                " it may be a runs directory where cells ran"
            )
    for entry in entries:
        if (
            entry.name in ("_gateway", "_schedule", "_supervise")
            and not entry.is_symlink()
        ):
            raise Refused(
                f"--into {into} holds {entry.name}/: a runs directory, not a view"
            )
        if _is_real_run_entry(entry):
            raise Refused(
                f"--into {into} holds a real cell or gateway directory {entry.name}/"
            )
    blobs = into / BLOBS_DIR
    if blobs.is_symlink() or (blobs.exists() and not blobs.is_dir()):
        raise Refused(f"--into {into}: {BLOBS_DIR} is not a directory")


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_sources(sources: Sequence[Source], into: Path) -> None:
    names = [s.name for s in sources]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise Refused(f"source names given more than once: {', '.join(repeated)}")
    target = into.resolve()
    for source in sources:
        if source.host:
            continue
        path = Path(source.path)
        if not path.is_dir():
            raise Refused(f"--source {source.name}: {path} is not a directory")
        if (path / MERGED_MARKER).is_file():
            raise Refused(
                f"--source {source.name}: {path} is a merged view, not a host's runs "
                "directory"
            )
        if _overlaps(path.resolve(), target):
            raise Refused(f"--source {source.name}: {path} overlaps --into {into}")


def _container_marker(path: Path) -> str | None:
    """What makes ``path`` a cell, a gateway run, a merged view or a mirror (the file it
    holds), or None. Never a runs directory's ``_gateway/``: runs directories nest by
    design (``runs/`` may hold ``runs/<local-run>`` and the view ``runs/<view>``, and
    gets a ``_gateway/`` of its own from a gateway started with the default
    ``--runs-dir``)."""
    for name in ("cell.json", "gateway.jsonl", MERGED_MARKER):
        if (path / name).is_file():
            return name
    return None


def check_mirror_paths(into: Path) -> None:
    """The mirrors are real directories this tool makes under ``<into>/_hosts/``, and
    ``<into>`` (its path resolved) is inside no cell, gateway run, view or mirror. A
    mirror symlinked into a live runs directory would get ``_merged.json`` written into
    it, and ``run.py`` would then refuse every cell there. An ``<into>`` in or inside a
    local source is refused by :func:`check_sources` (paths resolved)."""
    hosts = into / HOSTS_DIR
    if hosts.is_symlink():
        raise Refused(
            f"--into {into}: {HOSTS_DIR} is a symlink (to {os.readlink(hosts)}): this "
            f"tool makes {HOSTS_DIR}/ and every mirror in it as real directories, and a "
            "sync through a link would write into what it points to"
        )
    if hosts.is_dir():
        for child in sorted(hosts.iterdir()):
            if child.is_symlink():
                raise Refused(
                    f"--into {into}: {HOSTS_DIR}/{child.name} is a symlink (to "
                    f"{os.readlink(child)}): a mirror is a real directory this tool "
                    "makes, and a sync through a link would write its _merged.json "
                    "into what it points to (a live runs directory, whose cells run.py "
                    f"would then refuse); remove the link"
                )
    for ancestor in into.resolve().parents:
        marker = _container_marker(ancestor)
        if marker is not None:
            raise Refused(
                f"--into {into} is inside {ancestor}, which holds {marker} (a cell, a "
                "gateway run, a merged view or a mirror): make the view elsewhere"
            )


# ---------------------------------------------------------------------------
# The deletion guard
# ---------------------------------------------------------------------------


def starts_identity(text: str) -> str | None:
    """The identity of a runs directory from its ``_gateway/starts.jsonl``: the first
    line's ``start_id`` (the sha256 of that line when it has none); None when empty."""
    for line in text.split("\n"):  # "\n" only: U+2028 stays inside a record
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict) and row.get("start_id"):
            return str(row["start_id"])
        return "sha256:" + hashlib.sha256(line.encode()).hexdigest()[:16]
    return None


@dataclass
class Identity:
    """What :func:`source_identity` found: the identity (None: no starts file), or the
    rsync failure that kept it from reading one."""

    first_start: str | None
    rsync_exit: int = 0
    stderr: str = ""


def source_identity(source: Source, *, rsync: str = "rsync") -> Identity:
    """The first gateway start of ``source`` (blocking): read directly for a local path,
    with one rsync of that single file over ssh for a remote one."""
    if source.host is None:
        path = Path(source.path) / STARTS_FILE
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return Identity(starts_identity(handle.read(1 << 16)))
        except OSError:
            return Identity(None)
    with tempfile.TemporaryDirectory(prefix="sync-runs-") as tmp:
        remote = f"{source.host}:{source.path.rstrip('/')}/{STARTS_FILE}"
        argv = [rsync, f"--timeout={RSYNC_TIMEOUT_S}", "-e", SSH_COMMAND, remote, tmp]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except OSError as exc:
            return Identity(None, 127, f"{rsync}: {exc}")
        if proc.returncode == RSYNC_PARTIAL:
            return Identity(None)  # the file does not exist there
        if proc.returncode != 0:
            return Identity(None, proc.returncode, proc.stderr)
        try:
            text = (Path(tmp) / Path(STARTS_FILE).name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return Identity(None)
        return Identity(starts_identity(text))


@dataclass
class Removed:
    """What a runs directory's ``_removed/index.json`` lists: the top-level
    entries its own ``tools/remove_method_data.py`` deleted on purpose, after writing
    their spend manifest.

    ``entries`` maps each entry to its removal's manifest
    (``<method>/<removal_id>/manifest.json``, under ``_removed/``); ``finished`` are the
    entries of removals in state ``complete``. ``problems``: index lines ignored (a name
    that is not a plain top-level cell or gateway run, an unreadable index). A remote
    read that failed has ``rsync_exit`` set: nothing is synced then."""

    entries: dict[str, str] = field(default_factory=dict)
    finished: frozenset[str] = frozenset()
    problems: list[str] = field(default_factory=list)
    rsync_exit: int = 0
    stderr: str = ""

    @property
    def manifests(self) -> list[str]:
        return sorted(set(self.entries.values()))


#: A gateway run directory's name: ``<cell_id>-aNN`` and the cell's instance id.
_RUN_NAME = re.compile(r"^(?P<cell>.+)-a\d{2,}(?:-[0-9a-f]+)?$")


def _entry_problem(entry: Any) -> str | None:
    """Why ``entry`` cannot be a top-level entry removed on purpose, or None. A runs
    directory's own entries (``_blobs``, ``_gateway``, ``_removed`` ...) and hidden
    ones are never excluded from a mirror nor deleted from it; nor is anything but a
    cell or a gateway run of a method the study removed (``study.REMOVED_METHODS``, by
    the name alone: the removal tool deletes nothing else), so a wrong index can
    neither hide a study cell from the view nor delete its mirror copy."""
    if not isinstance(entry, str) or not entry:
        return "not a name"
    if (
        any(c in entry for c in "/\\\0*?[]")
        or entry in (".", "..")
        or entry != entry.strip()
    ):
        return "not a top-level entry name"
    if entry.startswith(("_", ".")) or entry in NON_CELL_DIRS:
        return "a runs directory's own or hidden entry is never removed"
    run = _RUN_NAME.fullmatch(entry)
    if not (
        study.removed_cell_id(entry)
        or (run is not None and study.removed_cell_id(run["cell"]))
    ):
        return (
            "not a cell or gateway run of a removed method "
            f"({', '.join(study.REMOVED_METHODS)})"
        )
    return None


def _index_manifests(text: str, states: Iterable[str] | None = None) -> dict[str, str]:
    """Each entry of ``index.json``'s removals (of those in ``states``, when given) ->
    its removal's manifest path."""
    try:
        record = json.loads(text)
    except ValueError:
        return {}
    removals = record.get("removals") if isinstance(record, Mapping) else None
    manifests: dict[str, str] = {}
    wanted = None if states is None else set(states)
    for removal in removals if isinstance(removals, list) else ():
        if not isinstance(removal, Mapping):
            continue
        if wanted is not None and removal.get("state") not in wanted:
            continue
        manifest = removal.get("manifest")
        if not isinstance(manifest, str) or not manifest:
            manifest = (
                f"{removal.get('method')}/{removal.get('removal_id')}/manifest.json"
            )
        entries = removal.get("entries")
        for entry in entries if isinstance(entries, list) else ():
            if isinstance(entry, str):
                manifests.setdefault(entry, manifest)
    return manifests


def removed_from_index(text: str, *, where: str = REMOVED_INDEX) -> Removed:
    """:class:`Removed` from the text of an ``index.json`` (pure). The entries are
    ``remove_method_data.removed_entries_from_index`` (``finished_only`` for
    ``finished``); a name that is not a plain top-level entry is a problem, ignored."""
    removed = Removed()
    unreadable = "nothing is excluded, so the deletion guard sees every removed entry"
    try:
        record = json.loads(text)
    except ValueError as exc:
        record, why = None, f"not JSON: {exc}"
    else:
        why = "not a JSON object" if not isinstance(record, Mapping) else None
    if why is not None:
        # The reader gives no entry for such a text; say so rather than exclude nothing
        # in silence (the guard then refuses the deletions it cannot account for).
        removed.problems.append(f"{where} unreadable ({why}): {unreadable}")
        return removed
    try:
        from evals.skill_bench.tools import remove_method_data

        listed = remove_method_data.removed_entries_from_index(text)
        finished = remove_method_data.removed_entries_from_index(
            text, finished_only=True
        )
        states = (remove_method_data.STATE_DELETING, remove_method_data.STATE_COMPLETE)
    except Exception as exc:  # the reader is not ours to trust blindly
        removed.problems.append(
            f"{where} unreadable ({type(exc).__name__}: {exc}): {unreadable}"
        )
        return removed
    manifests = _index_manifests(text)
    # Every name the index lists for a removal the reader counts, so a name the reader
    # leaves out (not a plain top-level entry, a runs directory's own) is warned about,
    # never dropped in silence.
    named = set(_index_manifests(text, states)) | set(listed)
    for entry in sorted(named, key=str):
        problem = _entry_problem(entry)
        if problem is None and entry not in listed:
            problem = "not a plain top-level entry name"
        if problem is not None:
            removed.problems.append(f"{where}: {entry!r} ignored ({problem})")
            continue
        removed.entries[entry] = manifests.get(entry, "unknown manifest")
    removed.finished = frozenset(e for e in finished if e in removed.entries)
    return removed


def source_removed(source: Source, *, rsync: str = "rsync") -> Removed:
    """What ``source`` removed on purpose (blocking): its ``_removed/index.json``, read
    directly for a local path, with one rsync of that single file over ssh for a remote
    one (exit 23: the file does not exist, no removal). Any other rsync failure is
    returned (``rsync_exit``): the sync then copies nothing."""
    where = f"{source.name}: {REMOVED_INDEX}"
    if source.host is None:
        path = Path(source.path) / REMOVED_INDEX
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, NotADirectoryError):
            return Removed()
        except OSError as exc:
            return Removed(
                problems=[
                    f"{where} unreadable ({exc}): nothing is excluded, so the deletion "
                    "guard sees every removed entry"
                ]
            )
        return removed_from_index(text, where=where)
    with tempfile.TemporaryDirectory(prefix="sync-runs-") as tmp:
        remote = f"{source.host}:{source.path.rstrip('/')}/{REMOVED_INDEX}"
        argv = [rsync, f"--timeout={RSYNC_TIMEOUT_S}", "-e", SSH_COMMAND, remote, tmp]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except OSError as exc:
            return Removed(rsync_exit=127, stderr=f"{rsync}: {exc}")
        if proc.returncode == RSYNC_PARTIAL:
            return Removed()  # no index there: nothing removed on purpose
        if proc.returncode != 0:
            return Removed(rsync_exit=proc.returncode, stderr=proc.stderr)
        try:
            text = (Path(tmp) / Path(REMOVED_INDEX).name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return Removed()
        return removed_from_index(text, where=where)


def mirror_removed(mirror: Path) -> Removed:
    """What a mirror's own copy of its source's ``_removed/index.json`` lists (the main
    rsync mirrors ``_removed/``): the entries the view never links from that mirror."""
    try:
        text = (mirror / REMOVED_INDEX).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return Removed()
    return removed_from_index(text, where=f"{mirror.name}: {REMOVED_INDEX}")


def mirror_identity(into: Path, name: str, record: Mapping[str, Any]) -> str | None:
    """The identity a mirror was made from: ``_merged.json``'s, else its own copy of
    the source's ``_gateway/starts.jsonl``."""
    sources = (
        record.get("sources") if isinstance(record.get("sources"), Mapping) else {}
    )
    entry = sources.get(name) if isinstance(sources, Mapping) else None
    if isinstance(entry, Mapping) and entry.get("first_start"):
        return str(entry["first_start"])
    path = into / HOSTS_DIR / name / STARTS_FILE
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return starts_identity(handle.read(1 << 16))
    except OSError:
        return None


def _mirror_files(into: Path, name: str) -> int:
    mirror = into / HOSTS_DIR / name
    return tree_size(mirror)[0] if mirror.is_dir() else 0


def check_identity(
    source: Source, into: Path, *, identity: Identity, record: Mapping[str, Any]
) -> None:
    """Refuse a sync that would replace a mirror: the source is not the runs
    directory the mirror was made from (another first gateway start, or none).
    ``--allow-other-source`` lifts it; ``--allow-deletes`` does not."""
    mirror = into / HOSTS_DIR / source.name
    files = _mirror_files(into, source.name)
    if files <= 1:  # nothing to lose (a new mirror, or its _merged.json alone)
        return
    known = mirror_identity(into, source.name, record)
    hint = (
        "; pass --allow-other-source to make the mirror follow it (what that deletes "
        "still needs --allow-deletes)"
    )
    if identity.first_start is None:
        raise Refused(
            f"--source {source.name}: {source.spec} has no {STARTS_FILE} (an emptied, "
            f"re-created or mistyped runs directory?), and syncing it would delete the "
            f"{files} files of its mirror {mirror}" + hint
        )
    if known is not None and identity.first_start != known:
        raise Refused(
            f"--source {source.name}: {source.spec} is another runs directory (first "
            f"gateway start {identity.first_start}, the mirror's {known}): syncing it "
            f"would replace the mirror {mirror}" + hint
        )


@dataclass
class Deletions:
    """What a sync would delete from a mirror, without a cut-off sync's leftovers."""

    #: every file, also those under a deleted directory
    files: list[str]
    #: the mirror's top-level entries deleted whole (hidden ones aside)
    entries: list[str]
    #: the :data:`PROTECTED_RECORDS` among ``files``
    records: list[str]


def planned_deletions(items: Itemized, mirror: Path) -> Deletions:
    """What a dry run's ``items`` delete from ``mirror``. A deleted directory's files are
    read from the mirror too, whichever rsync printed them."""
    files = set(items.deleted)
    for directory in items.deleted_dirs:
        root = mirror / directory.rstrip("/")
        if root.is_dir() and not root.is_symlink():
            for path, _dirs, names in _walk(root):
                files.update(
                    Path(path, name).relative_to(mirror).as_posix() for name in names
                )
    tops = {p.rstrip("/") for p in (*items.deleted, *items.deleted_dirs)}
    return Deletions(
        files=sorted(files),
        entries=sorted(t for t in tops if "/" not in t and not t.startswith(".")),
        records=sorted(p for p in files if p.rsplit("/", 1)[-1] in PROTECTED_RECORDS),
    )


def _some(names: Sequence[str], limit: int = 5) -> str:
    return ", ".join(names[:limit]) + (", ..." if len(names) > limit else "")


def check_deletions(
    source: Source,
    into: Path,
    *,
    rsync: str = "rsync",
    with_home: bool = False,
    with_checkpoints: bool = False,
    removed: Iterable[str] = (),
) -> None:
    """Refuse a sync whose dry run deletes a whole top-level entry of the mirror, a
    record anywhere, or more than :data:`DELETE_GUARD_FRACTION` of its files; a
    cut-off sync's leftovers are never counted. ``--allow-deletes`` lifts it. The
    entries the source's ``_removed/index.json`` lists (``removed``) are excluded from
    the dry run as from the sync itself: their deletion was recorded on purpose and
    never counts; every other deletion does."""
    mirror = into / HOSTS_DIR / source.name
    files = _mirror_files(into, source.name)
    if files <= 1:  # nothing to lose (a new mirror, or its _merged.json alone)
        return
    argv = rsync_argv(
        source,
        mirror,
        rsync=rsync,
        dry_run=True,
        with_home=with_home,
        with_checkpoints=with_checkpoints,
        extra_excludes=removed,
        # After an I/O error (a file that vanished or cannot be read, rsync exit 23)
        # both rsyncs skip every deletion unless told otherwise: without this the
        # dry run would list none and the guard pass, while the real run, with no
        # error at its moment, deletes whole cells. With it the listing is
        # a superset of what the real run can delete (it still skips deletions on an
        # error).
        ignore_errors=True,
    )
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError:
        return  # the sync itself reports the missing rsync
    if proc.returncode not in (0, RSYNC_PARTIAL, RSYNC_VANISHED):
        # The guard fails closed. A dry run cut off by a flaky link
        # (ssh's 255) says nothing about what the sync would delete; the real rsync
        # that followed could still succeed and remove whole cells from what may be
        # their only copy. 23 (macOS openrsync's for a file that vanished or cannot be
        # read during a live run) and 24 still list every deletion, thanks to
        # --ignore-errors.
        tail = " ".join(proc.stderr.strip().splitlines()[-3:])
        raise Refused(
            f"--source {source.name}: the dry run that checks what syncing "
            f"{source.spec} would delete failed (rsync exit {proc.returncode}"
            + (f": {tail}" if tail else "")
            + "); nothing was copied: sync again, or pass --allow-deletes to sync "
            "without the check"
        )
    deletions = planned_deletions(parse_itemized(proc.stdout), mirror)
    limit = max(DELETE_GUARD_MIN_FILES, int(DELETE_GUARD_FRACTION * files))
    problems = []
    if deletions.entries:
        n = len(deletions.entries)
        problems.append(
            f"{n} whole {'entry' if n == 1 else 'entries'} ({_some(deletions.entries)})"
        )
    if deletions.records:
        n = len(deletions.records)
        problems.append(
            f"{n} record{'' if n == 1 else 's'} ({_some(deletions.records)})"
        )
    if len(deletions.files) > limit:
        problems.append(
            f"{len(deletions.files)} of its {files} files (more than {limit})"
        )
    if problems:
        # After an I/O error (23) the listing may name what the real run would keep (a
        # directory unreadable for a moment): say so, with rsync's words. A
        # 24 (files vanished) does not stop deletions: its listing is what happens.
        tail = " ".join(proc.stderr.strip().splitlines()[-3:])
        note = (
            f" (the dry run met an I/O error, rsync exit {proc.returncode}: {tail}; "
            "while it lasts the real sync may delete nothing, so a directory that "
            "cannot be read may show here as deleted: check it on the host)"
            if proc.returncode == RSYNC_PARTIAL
            else ""
        )
        raise Refused(
            f"--source {source.name}: syncing {source.spec} would delete "
            + ", ".join(problems)
            + f" from its mirror {mirror}, which may be their only copy{note}; restore "
            "them on the host, or pass --allow-deletes to sync it anyway"
        )


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------


@dataclass
class Mirrored:
    source: Source
    exit: int
    stderr: str
    items: Itemized
    record: dict[str, Any]
    #: what the source removed on purpose
    removed: Removed = field(default_factory=Removed)
    #: the entries of finished removals deleted from the mirror by this sync
    removed_from_mirror: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.exit not in (0, RSYNC_VANISHED)


def _run_rsync(argv: list[str], rsync: str) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        return 127, "", f"{rsync}: {exc}"
    return proc.returncode, proc.stdout, proc.stderr


def remove_finished_entries(mirror: Path, removed: Removed) -> list[str]:
    """Delete from ``mirror`` each entry of a FINISHED removal its source lists:
    real directories only (never a link, never ``_blobs`` or any ``_`` entry: see
    :func:`_entry_problem`). An unfinished removal's entries stay, excluded and
    unlinked. Returns the entries deleted; what is left is added to its problems."""
    deleted: list[str] = []
    for entry in sorted(removed.finished):
        if _entry_problem(entry) is not None:  # never: removed_from_index drops them
            continue
        path = mirror / entry
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_dir():
            removed.problems.append(
                f"{entry} in the mirror is not a real directory: left in place"
            )
            continue
        shutil.rmtree(path)
        deleted.append(entry)
    return deleted


def removed_from_mirror_at(
    previous: Mapping[str, Any], deleted_now: Iterable[str], synced_at: str
) -> dict[str, str]:
    """Every entry a sync deleted from a mirror -> when: the previous
    record's ``removed_from_mirror_at`` (its ``removed_from_mirror`` too, for a record
    written before this map existed, at that record's ``synced_at``) plus
    ``deleted_now`` at ``synced_at``. An entry deleted again keeps its first time."""
    carried = previous.get("removed_from_mirror_at")
    at: dict[str, str] = {
        str(entry): str(when)
        for entry, when in (carried.items() if isinstance(carried, Mapping) else ())
    }
    legacy = previous.get("removed_from_mirror")
    for entry in legacy if isinstance(legacy, list) else ():
        at.setdefault(str(entry), str(previous.get("synced_at")))
    for entry in deleted_now:
        at.setdefault(entry, synced_at)
    return dict(sorted(at.items()))


def mirror_source(
    source: Source,
    into: Path,
    *,
    rsync: str = "rsync",
    version: str | None = None,
    dry_run: bool = False,
    with_home: bool = False,
    with_checkpoints: bool = False,
    checksum: bool = False,
    removed: Removed | None = None,
) -> Mirrored:
    """Bring ``<into>/_hosts/<name>/`` up to ``source``: the main rsync (without the
    entries ``removed`` lists), then ``_gateway/starts.jsonl`` alone once more
    (:func:`starts_argv`; its exit 23, no such file, is fine; any other failure fails
    the source), then the entries of finished removals leave the mirror
    (:func:`remove_finished_entries`). A dry run does only the first, with
    ``-n``."""
    removed = removed if removed is not None else Removed()
    mirror = into / HOSTS_DIR / source.name
    staging_removed = 0
    if not dry_run:
        mirror.mkdir(parents=True, exist_ok=True)
        # Under the sync's lock, so no other rsync of this mirror is using them.
        staging_removed = remove_staging_dirs(mirror)
    argv = rsync_argv(
        source,
        mirror,
        rsync=rsync,
        dry_run=dry_run,
        with_home=with_home,
        with_checkpoints=with_checkpoints,
        checksum=checksum,
        extra_excludes=removed.entries,
    )
    code, out, err = _run_rsync(argv, rsync)
    rsync_exit = code
    items = parse_itemized(out)
    starts_exit: int | None = None
    starts_synced_at: str | None = None
    removed_now: list[str] = []
    if not dry_run and code in (0, RSYNC_VANISHED):
        (mirror / Path(STARTS_FILE).parent).mkdir(parents=True, exist_ok=True)
        starts_exit, _out, starts_err = _run_rsync(
            starts_argv(source, mirror, rsync=rsync), rsync
        )
        if starts_exit in (0, RSYNC_VANISHED):
            starts_synced_at = now()
            if starts_exit == RSYNC_VANISHED and code == 0:
                code = RSYNC_VANISHED  # a warning, as for the main rsync
        elif starts_exit != RSYNC_PARTIAL:  # 23: no starts file there (no gateway yet)
            code = starts_exit
            err = (
                f"{err.rstrip()}\ncopy of {STARTS_FILE} failed (exit {starts_exit}): "
                f"{starts_err.strip()}"
            ).lstrip()
    if not dry_run and code in (0, RSYNC_VANISHED):
        removed_now = remove_finished_entries(mirror, removed)
    if not dry_run and rsync_exit in (0, RSYNC_VANISHED):
        # The mirror is read-only too: run.py and schedule.py refuse to run cells in
        # it, and a later sync's --delete-after keeps this file (it is excluded). Also
        # when only the starts copy failed: the cells the main rsync brought are there.
        write_json_atomic(
            mirror / MERGED_MARKER,
            {
                "mirror_of": source.spec,
                "name": source.name,
                "view": "../..",
                "at": now(),
            },
        )
    files, size = tree_size(mirror) if mirror.is_dir() else (0, 0)
    record = {
        "spec": source.spec,
        "synced_at": now(),
        "rsync_version": version,
        "rsync_exit": rsync_exit,
        "starts_exit": starts_exit,
        "starts_synced_at": starts_synced_at,
        "transferred": len(items.transferred),
        "deleted": len(items.deleted),
        "leftovers_removed": staging_removed,
        "removed_from_mirror": removed_now,
        "removed_manifests": removed.manifests,
        "removed_problems": list(removed.problems),
        "files": files,
        "bytes": size,
        "with_home": with_home,
        "with_checkpoints": with_checkpoints,
    }
    return Mirrored(source, code, err, items, record, removed, removed_now)


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------


@dataclass
class View:
    #: link name -> the host whose mirror it links to (a link name is the entry's own
    #: name, or ``<entry>@<host>`` for a second independent cell of that name)
    entries: dict[str, str] = field(default_factory=dict)
    #: link name -> the entry's name in ``_hosts/<host>/``, only where they differ
    links: dict[str, str] = field(default_factory=dict)
    #: entry name -> every host that has it, when identical on several
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    #: entry name -> {host: link name}: independent cells of one name on several hosts
    #: (different ``cell.json`` instances, e.g. a rerun on another host), all linked
    cells_on_several_hosts: dict[str, dict[str, str]] = field(default_factory=dict)
    #: host -> {entry: its removal's manifest}: what each mirror's source removed on
    #: purpose (``_removed/index.json``); never linked
    removed: dict[str, dict[str, str]] = field(default_factory=dict)
    #: blob name -> host
    blobs: dict[str, str] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    cells: int = 0

    def target(self, link: str) -> str:
        """Where the view's ``link`` points (relative to ``<into>``)."""
        return f"{HOSTS_DIR}/{self.entries[link]}/{self.links.get(link, link)}"


def mirrors_of(into: Path) -> list[Path]:
    hosts = into / HOSTS_DIR
    if not hosts.is_dir():
        return []
    return sorted(
        d
        for d in hosts.iterdir()
        if d.is_dir() and not d.is_symlink() and SOURCE_NAME.match(d.name)
    )


def _cell_identity(path: Path) -> tuple[str, datetime | None] | None:
    """A cell directory's ``cell.json`` ``instance`` and ``created_at``; None when it is
    not a cell, or its ``cell.json`` or instance cannot be read."""
    if not (path / "cell.json").is_file():
        return None
    try:
        manifest = read_json(path / "cell.json")
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    instance = manifest.get("instance")
    if not isinstance(instance, str) or not instance:
        return None
    return instance, _time(manifest.get("created_at"))


def _independent_cells(paths: Sequence[Path]) -> dict[str, datetime | None] | None:
    """``{host: created_at}`` when every copy of one entry is a cell directory whose
    ``cell.json`` instance is readable and differs from every other copy's (independent
    cells of one id); else None (a copied cell that diverged: a conflict)."""
    created: dict[str, datetime | None] = {}
    instances: set[str] = set()
    for path in paths:
        identity = _cell_identity(path)
        if identity is None or identity[0] in instances:
            return None
        instances.add(identity[0])
        created[path.parent.name] = identity[1]
    return created


def _plain_name_host(
    into: Path, name: str, created: Mapping[str, datetime | None]
) -> str:
    """Which copy of independent cells keeps the plain ``name`` in the view: the one the
    view already links it to, else the earliest ``cell.json`` ``created_at`` (a missing
    time last; ties: host name order)."""
    link = into / name
    if link.is_symlink():
        target = os.readlink(link)
        for host in created:
            if target == f"{HOSTS_DIR}/{host}/{name}":
                return host
    return min(
        created,
        key=lambda host: (
            created[host] is None,
            created[host] or datetime.min.replace(tzinfo=timezone.utc),
            host,
        ),
    )


def plan_view(into: Path) -> View:
    """What the view links, and what conflicts, from every mirror under ``_hosts/``.

    An entry a mirror's own ``_removed/index.json`` lists is never linked from that
    mirror. The same entry on several mirrors is one link when the trees are
    identical; independent cells of one name (different ``cell.json`` instances) are
    all linked, the plain name to one and ``<entry>@<host>`` to each other;
    anything else is a conflict."""
    view = View()
    found: dict[str, list[Path]] = {}
    blobs: dict[str, list[tuple[str, int]]] = {}
    for mirror in mirrors_of(into):
        removed = mirror_removed(mirror)
        if removed.entries:
            view.removed[mirror.name] = dict(removed.entries)
        for child in sorted(mirror.iterdir()):
            if child.name.startswith(("_", ".")) or child.name in NON_CELL_DIRS:
                continue
            if child.name in removed.entries:
                continue
            if _is_real_run_entry(child):
                found.setdefault(child.name, []).append(child)
        blob_dir = mirror / BLOBS_DIR
        if blob_dir.is_dir() and not blob_dir.is_symlink():
            for blob in sorted(blob_dir.iterdir()):
                if (
                    BLOB_NAME.match(blob.name)
                    and blob.is_file()
                    and not blob.is_symlink()
                ):
                    blobs.setdefault(blob.name, []).append(
                        (mirror.name, blob.stat().st_size)
                    )
    for name, paths in sorted(found.items()):
        hosts = [p.parent.name for p in paths]
        if len(paths) > 1:
            digests = {tree_digest(p) for p in paths}
            if len(digests) > 1:
                created = _independent_cells(paths)
                links = (
                    None
                    if created is None
                    else _several_hosts_links(into, name, created, found)
                )
                if links is None:
                    view.conflicts.append(
                        {"entry": name, "hosts": hosts, "reason": "different content"}
                    )
                    continue
                view.cells_on_several_hosts[name] = links
                for host, link in links.items():
                    view.entries[link] = host
                    if link != name:
                        view.links[link] = name
                    view.cells += 1
                continue
            view.duplicates[name] = hosts
        view.entries[name] = hosts[0]
        if (paths[0] / "cell.json").is_file():
            view.cells += 1
    for name, copies in sorted(blobs.items()):
        if len({size for _host, size in copies}) > 1:
            view.conflicts.append(
                {
                    "entry": f"{BLOBS_DIR}/{name}",
                    "hosts": [h for h, _s in copies],
                    "reason": "different sizes",
                }
            )
            continue
        view.blobs[name] = copies[0][0]
    for name in view.entries:
        existing = into / name
        if existing.exists() and not existing.is_symlink():
            view.conflicts.append(
                {
                    "entry": name,
                    "hosts": [view.entries[name]],
                    "reason": "not a link in --into",
                }
            )
    return view


def _several_hosts_links(
    into: Path,
    name: str,
    created: Mapping[str, datetime | None],
    found: Mapping[str, Sequence[Path]],
) -> dict[str, str] | None:
    """``{host: link name}`` for independent cells of one ``name``; None when a
    ``<name>@<host>`` link name is itself an entry of some mirror (never shadowed)."""
    plain = _plain_name_host(into, name, created)
    links = {
        host: name if host == plain else f"{name}@{host}" for host in sorted(created)
    }
    if any(link in found for link in links.values() if link != name):
        return None
    return links


def _link(path: Path, target: str) -> bool:
    """Point ``path`` at ``target`` (atomic replace); False when it already did."""
    if path.is_symlink() and os.readlink(path) == target:
        return False
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.link")
    os.symlink(target, tmp)
    os.replace(tmp, path)
    return True


def _ours(path: Path, prefix: str) -> bool:
    return path.is_symlink() and os.readlink(path).startswith(prefix)


def apply_view(into: Path, view: View) -> dict[str, int]:
    """Create and update the view's links; remove the links into ``_hosts/`` it no longer
    has. Only symlinks are ever removed."""
    counts = {"linked": 0, "removed": 0}
    wanted = {link: view.target(link) for link in view.entries}
    for child in into.iterdir():
        if (
            child.is_symlink()
            and child.name.startswith(".")
            and child.name.endswith(".link")
        ):
            child.unlink()  # a temporary link of a sync that was cut off
        elif _ours(child, f"{HOSTS_DIR}/") and wanted.get(child.name) != os.readlink(
            child
        ):
            child.unlink()
            counts["removed"] += 1
    for name, target in sorted(wanted.items()):
        counts["linked"] += _link(into / name, target)
    blob_dir = into / BLOBS_DIR
    blob_dir.mkdir(exist_ok=True)
    wanted_blobs = {
        name: f"../{HOSTS_DIR}/{host}/{BLOBS_DIR}/{name}"
        for name, host in view.blobs.items()
    }
    for child in blob_dir.iterdir():
        if _ours(child, f"../{HOSTS_DIR}/") and wanted_blobs.get(
            child.name
        ) != os.readlink(child):
            child.unlink()
            counts["removed"] += 1
    for name, target in sorted(wanted_blobs.items()):
        path = blob_dir / name
        if path.exists() and not path.is_symlink():
            continue  # a real blob placed here by hand: content-addressed, keep it
        counts["linked"] += _link(path, target)
    return counts


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _merged_record(into: Path) -> dict[str, Any]:
    record = read_json(into / MERGED_MARKER)
    return dict(record) if isinstance(record, Mapping) else {}


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


@contextmanager
def sync_lock(into: Path) -> Iterator[None]:
    """Hold ``<into>/.sync_runs.lock`` for a whole sync: another sync of
    the same view is refused (exit 2) while one runs, so neither removes, nor deletes
    with ``--delete-after``, the other's staging directories mid-transfer, nor writes
    over the other's ``_merged.json``. The kernel releases it however the process ends."""
    into.mkdir(parents=True, exist_ok=True)
    handle = (into / LOCK_FILE).open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "pid unknown"
            raise Refused(
                f"another sync of {into} is running ({holder}); wait for it to end"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()}, since {now()}\n")
        handle.flush()
        yield
    finally:
        handle.close()


def _print_removed(result: Mirrored, *, dry_run: bool, out: Any, err: Any) -> None:
    """What a source removed on purpose, on one line."""
    removed = result.removed
    if not removed.entries:
        return
    n = len(removed.entries)
    line = (
        f"{result.source.name}: {n} {'entry' if n == 1 else 'entries'} removed on "
        f"purpose ({', '.join(removed.manifests)}): never copied again nor linked"
    )
    if dry_run:
        line += f"; {len(removed.finished)} would leave the mirror"
    else:
        line += f"; {len(result.removed_from_mirror)} deleted from the mirror now"
    print(line, file=out)
    unfinished = sorted(set(removed.entries) - removed.finished)
    if unfinished:
        print(
            f"warning: {result.source.name}: {_some(unfinished)} "
            f"{'is' if len(unfinished) == 1 else 'are'} listed by a removal not "
            "finished on the host: excluded and unlinked, the mirror keeps its copy "
            "until the removal completes",
            file=err,
        )


def _last_sync(mode: str, sources: Sequence[Source]) -> dict[str, Any]:
    """How a sync that wrote ``_merged.json`` ended."""
    return {"at": now(), "mode": mode, "sources": [s.name for s in sources]}


def sync(
    sources: Sequence[Source],
    into: Path,
    *,
    rsync: str = "rsync",
    dry_run: bool = False,
    mirror_only: bool = False,
    with_home: bool = False,
    with_checkpoints: bool = False,
    checksum: bool = False,
    verbose: bool = False,
    allow_deletes: bool = False,
    allow_other_source: bool = False,
    out: Any = None,
    err: Any = None,
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    check_sources(sources, into)
    check_into(into)
    check_mirror_paths(into)
    with ExitStack() as stack:
        if not dry_run:  # a dry run only reads
            stack.enter_context(sync_lock(into))
        version = rsync_version(rsync)
        record = _merged_record(into)
        # Every source is checked before any mirror changes.
        identities = {s.name: source_identity(s, rsync=rsync) for s in sources}
        unreadable = {n: i for n, i in identities.items() if i.rsync_exit}
        for name, identity in unreadable.items():
            print(
                f"error: {name}: cannot read its {STARTS_FILE} (rsync exit "
                f"{identity.rsync_exit}): {identity.stderr.strip()[-2000:]}",
                file=err,
            )
        # What each source removed on purpose, read like its identity.
        removals = {
            s.name: source_removed(s, rsync=rsync)
            for s in sources
            if s.name not in unreadable
        }
        unread_removals = {n: r for n, r in removals.items() if r.rsync_exit}
        for name, removed in unread_removals.items():
            print(
                f"error: {name}: cannot read its {REMOVED_INDEX} (rsync exit "
                f"{removed.rsync_exit}): {removed.stderr.strip()[-2000:]}",
                file=err,
            )
        if unreadable or unread_removals:
            print("nothing was synced; the merged view was not changed", file=err)
            return EXIT_RSYNC
        for name, removed in removals.items():
            for problem in removed.problems:
                print(f"warning: {problem}", file=err)
        # Two guards, two flags: --allow-deletes never lets a mirror
        # follow another runs directory.
        if not allow_other_source:
            for source in sources:
                check_identity(
                    source, into, identity=identities[source.name], record=record
                )
        if not allow_deletes:
            for source in sources:
                check_deletions(
                    source,
                    into,
                    rsync=rsync,
                    with_home=with_home,
                    with_checkpoints=with_checkpoints,
                    removed=removals[source.name].entries,
                )
        mirrored = []
        for source in sources:
            removed = removals[source.name]
            problems_before = len(removed.problems)
            result = mirror_source(
                source,
                into,
                rsync=rsync,
                version=version,
                dry_run=dry_run,
                with_home=with_home,
                with_checkpoints=with_checkpoints,
                checksum=checksum,
                removed=removed,
            )
            mirrored.append(result)
            for problem in removed.problems[problems_before:]:
                print(f"warning: {source.name}: {problem}", file=err)
            _print_removed(result, dry_run=dry_run, out=out, err=err)
            verb = "would transfer" if dry_run else "transferred"
            print(
                f"{source.name}: rsync exit {result.exit}; {verb} "
                f"{len(result.items.transferred)} files, "
                f"{'would delete' if dry_run else 'deleted'} "
                f"{len(result.items.deleted)}"
                + (
                    ""
                    if dry_run
                    else f"; mirror {result.record['files']} files, "
                    f"{_human(result.record['bytes'])}"
                ),
                file=out,
            )
            if result.record["leftovers_removed"]:
                print(
                    f"{source.name}: removed {result.record['leftovers_removed']} "
                    f"staging directories an interrupted sync left ({RSYNC_STAGING_DIR})",
                    file=out,
                )
            if verbose:
                for path in result.items.transferred:
                    print(f"  > {path}", file=out)
                for path in result.items.deleted:
                    print(f"  - {path}", file=out)
            if result.exit == RSYNC_VANISHED:
                print(
                    f"warning: {source.name}: files vanished during the transfer (a run "
                    "is writing); sync again once it is idle",
                    file=err,
                )
            elif result.failed:
                print(
                    f"error: {source.name}: rsync failed (exit {result.exit}): "
                    f"{result.stderr.strip()[-2000:]}",
                    file=err,
                )
        if dry_run:
            print("dry run: nothing changed", file=out)
            return EXIT_RSYNC if any(m.failed for m in mirrored) else EXIT_OK
        record["schema"] = MERGED_SCHEMA
        record.setdefault("sources", {})
        for result in mirrored:
            previous = record["sources"].get(result.source.name) or {}
            first_start = identities[result.source.name].first_start
            record["sources"][result.source.name] = {
                **result.record,
                # The identity the mirror now follows; a failed sync keeps
                # the previous one.
                "first_start": first_start
                if not result.failed and first_start is not None
                else previous.get("first_start"),
                # Every entry any sync deleted from this mirror, and when:
                # ``removed_from_mirror`` is this sync's alone.
                "removed_from_mirror_at": removed_from_mirror_at(
                    previous, result.removed_from_mirror, result.record["synced_at"]
                ),
            }
        # What every mirror's source removed on purpose, from the mirrors'
        # own copies of their _removed/index.json: never linked into the view.
        record["removed_on_purpose"] = {
            mirror.name: dict(removed.entries)
            for mirror in mirrors_of(into)
            if (removed := mirror_removed(mirror)).entries
        }
        record["at"] = now()
        if any(m.failed for m in mirrored):
            record["last_sync"] = _last_sync("rsync failed", sources)
            write_json_atomic(into / MERGED_MARKER, record)
            print("the merged view was not changed", file=err)
            return EXIT_RSYNC
        if mirror_only:
            record.setdefault("entries", {})
            record.setdefault("conflicts", [])
            record["last_sync"] = _last_sync("mirror-only", sources)
            write_json_atomic(into / MERGED_MARKER, record)
            print(f"mirror only: the view in {into} was not changed", file=out)
            return EXIT_OK
        view = plan_view(into)
        record["conflicts"] = view.conflicts
        if view.conflicts:
            record["last_sync"] = _last_sync("conflicts", sources)
            write_json_atomic(into / MERGED_MARKER, record)
            for conflict in view.conflicts:
                print(
                    f"conflict: {conflict['entry']} from "
                    f"{', '.join(conflict['hosts'])} ({conflict['reason']})",
                    file=err,
                )
            print("the merged view was not changed", file=err)
            return EXIT_CONFLICT
        counts = apply_view(into, view)
        record.update(
            entries=view.entries,
            links=view.links,
            duplicates=view.duplicates,
            cells_on_several_hosts=view.cells_on_several_hosts,
            blobs=len(view.blobs),
            view_built_at=now(),
            last_sync=_last_sync("view", sources),
        )
        write_json_atomic(into / MERGED_MARKER, record)
        runs = len(view.entries) - view.cells
        for name, links in sorted(view.cells_on_several_hosts.items()):
            shown = ", ".join(
                f"{link} ({host})" for host, link in sorted(links.items())
            )
            print(
                f"{name}: independent cells of one id on several hosts, all linked: "
                f"{shown}",
                file=out,
            )
        print(
            f"view {into}: {view.cells} cells, {runs} gateway runs, {len(view.blobs)} "
            f"blobs from {len(mirrors_of(into))} hosts; {counts['linked']} links made, "
            f"{counts['removed']} removed",
            file=out,
        )
        return EXIT_OK


# ---------------------------------------------------------------------------
# What the readers are told
# ---------------------------------------------------------------------------


def _time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hours(since: Any, at: datetime) -> float | None:
    start = _time(since)
    return None if start is None else round((at - start).total_seconds() / 3600, 2)


#: How the sync that left a view behind its mirrors ended (``last_sync.mode``).
_BEHIND = {
    "mirror-only": "a --mirror-only sync",
    "conflicts": "a sync stopped by conflicts",
    "rsync failed": "a sync whose rsync failed",
}


def merged_view_status(
    runs_dir: Path, *, at: datetime | None = None
) -> dict[str, Any] | None:
    """What ``<runs_dir>/_merged.json`` says, for ``audit.py`` and ``report.py``; None
    when there is none. ``warnings`` (``check``, ``detail``) say why the
    view is not final:

    * ``merged_view_not_final``: it was never built (``--mirror-only`` only); a host was
      synced after it was built (a ``--mirror-only`` sync, or one stopped by conflicts or
      an rsync failure: its links miss the entries added since and keep those removed);
      a host's last rsync did not exit 0 (24: a run was writing);
    * ``merged_view_conflict``: each conflict it lists (an entry that differs between
      hosts, which the view links from at most one of them);
    * ``merged_view_unreadable``: the file is not a JSON object.

    ``age_hours`` is the time since the view was built (for a host's mirror, since it
    was synced), at ``at`` (default now). ``cells_on_several_hosts`` (``{entry: {host:
    link}}``) and ``removed_on_purpose`` (``{host: {entry: manifest}}``) are
    information, never a warning. Blocking."""
    path = runs_dir / MERGED_MARKER
    at = at or datetime.now(timezone.utc)
    try:
        record = read_json(path)
    except (OSError, ValueError) as exc:
        record = f"{type(exc).__name__}: {exc}"
    if record is None:
        return None
    status: dict[str, Any] = {
        "path": str(path),
        "kind": "view",
        "view_built_at": None,
        "age_hours": None,
        "hosts": {},
        "conflicts": [],
        "cells_on_several_hosts": {},
        "removed_on_purpose": {},
        "warnings": [],
    }

    def warn(check: str, detail: str) -> None:
        status["warnings"].append({"check": check, "detail": detail})

    if not isinstance(record, Mapping):
        why = record if isinstance(record, str) else "not a JSON object"
        warn(
            "merged_view_unreadable",
            f"{path} is unreadable ({why}): what the view holds is unknown",
        )
        status.update(final=False, summary=f"{path} is unreadable")
        return status
    if record.get("mirror_of") is not None:
        # A host's mirror (``_hosts/<name>/``), read on its own.
        status.update(
            kind="mirror",
            mirror_of=str(record["mirror_of"]),
            age_hours=_hours(record.get("at"), at),
            final=True,
            summary=f"the mirror of {record['mirror_of']}, synced {record.get('at')} "
            f"({_hours(record.get('at'), at)} h ago)",
        )
        return status
    sources = record.get("sources")
    for name, entry in sorted(
        (sources if isinstance(sources, Mapping) else {}).items()
    ):
        entry = entry if isinstance(entry, Mapping) else {}
        status["hosts"][str(name)] = {
            key: entry.get(key)
            for key in ("spec", "synced_at", "rsync_exit", "starts_exit")
        }
    for key in ("cells_on_several_hosts", "removed_on_purpose"):
        value = record.get(key)
        if isinstance(value, Mapping):
            status[key] = {
                str(k): dict(v) for k, v in value.items() if isinstance(v, Mapping)
            }
    conflicts = record.get("conflicts")
    status["conflicts"] = [
        dict(c)
        for c in (conflicts if isinstance(conflicts, list) else ())
        if isinstance(c, Mapping)
    ]
    built = record.get("view_built_at")
    last = (
        record.get("last_sync") if isinstance(record.get("last_sync"), Mapping) else {}
    )
    status.update(view_built_at=built, age_hours=_hours(built, at))
    if not built:
        warn(
            "merged_view_not_final",
            "the view was never built: every sync of it was --mirror-only (or stopped), "
            "so it links no cell; run tools/sync_runs.py without --mirror-only",
        )
    else:
        built_time = _time(built)
        behind = sorted(
            name
            for name, host in status["hosts"].items()
            if built_time is not None
            and (_time(host["synced_at"]) or built_time) > built_time
        )
        if behind:
            how = _BEHIND.get(
                str(last.get("mode")),
                "a --mirror-only sync, or one stopped by conflicts or an rsync failure",
            )
            warn(
                "merged_view_not_final",
                f"{', '.join(behind)} synced after the view was built ({built}), by "
                f"{how}: its links miss the entries added since and keep those removed "
                "since; sync again without --mirror-only (after resolving any conflict)",
            )
    for name, host in status["hosts"].items():
        code = host["rsync_exit"]
        starts = host.get("starts_exit")
        problems = []
        if code not in (0, None):
            why = (
                "files vanished while a run wrote them"
                if code == RSYNC_VANISHED
                else "rsync failed"
            )
            problems.append(f"the last sync of {name} exited {code} ({why})")
        if starts not in (0, RSYNC_PARTIAL, None):
            problems.append(
                f"the last copy of its {STARTS_FILE} exited {starts} (the mirror may "
                "miss the start record of a gateway whose ledger it holds)"
            )
        if problems:
            warn(
                "merged_view_not_final",
                "; ".join(problems) + ": sync again once its runs are idle",
            )
    for conflict in status["conflicts"]:
        hosts = ", ".join(str(h) for h in conflict.get("hosts") or ())
        warn(
            "merged_view_conflict",
            f"conflict: {conflict.get('entry')} from {hosts} ({conflict.get('reason')}): "
            "the view links at most one of them; keep the entry on one host and sync "
            "again",
        )
    hosts_text = ", ".join(
        f"{name} synced {host['synced_at']}" for name, host in status["hosts"].items()
    )
    status["final"] = not status["warnings"]
    status["summary"] = f"{len(status['hosts'])} host(s) ({hosts_text}); " + (
        f"view built {built} ({status['age_hours']} h ago)"
        if built
        else "view never built"
    )
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=[HOST:]PATH",
        help="a host's runs directory (repeatable)",
    )
    parser.add_argument("--into", type=Path, required=True, help="the merged view")
    parser.add_argument(
        "--dry-run", action="store_true", help="rsync -n; change nothing"
    )
    parser.add_argument(
        "--mirror-only",
        action="store_true",
        help="update _hosts/<NAME>/ only (monitoring), not the view",
    )
    parser.add_argument(
        "--with-home", action="store_true", help="also copy MIRIX's per-attempt HOME"
    )
    parser.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="also copy the learning checkpoints",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help="compare file contents, not sizes and times (slower; the final sync)",
    )
    parser.add_argument(
        "--allow-deletes",
        action="store_true",
        help="sync a source whatever it deletes from its mirror: a whole cell, gateway "
        "run or other top-level entry, a record (cell.json, status.json, "
        "results.jsonl, learning.jsonl, attempt.json, gateway.jsonl), or more than "
        "half of the mirror's files (off by default: a mirror may be a gone host's "
        "only copy). It never lets a mirror follow another runs directory",
    )
    parser.add_argument(
        "--allow-other-source",
        action="store_true",
        help="sync a source that is not the runs directory its mirror was made from "
        "(another first gateway start, or no _gateway/starts.jsonl): the mirror "
        "follows it from now on; what that deletes still needs --allow-deletes",
    )
    parser.add_argument("--rsync", default="rsync", help="the rsync binary")
    parser.add_argument(
        "--verbose", action="store_true", help="list every file sent and deleted"
    )
    args = parser.parse_args(argv)
    if shutil.which(args.rsync) is None:
        print(f"error: {args.rsync} not found", file=sys.stderr)
        return EXIT_REFUSED
    try:
        sources = [parse_source(text) for text in args.source]
        return sync(
            sources,
            args.into,
            rsync=args.rsync,
            dry_run=args.dry_run,
            mirror_only=args.mirror_only,
            with_home=args.with_home,
            with_checkpoints=args.with_checkpoints,
            checksum=args.checksum,
            verbose=args.verbose,
            allow_deletes=args.allow_deletes,
            allow_other_source=args.allow_other_source,
        )
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
