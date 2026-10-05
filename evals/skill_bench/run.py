"""One cell end to end: (benchmark, model, method) under protocol P1 (H1, H8, H9, H10.4).

Usage::

    python -m evals.skill_bench.run --benchmark alfworld --model luna --method mirix \\
        --gateway-url http://127.0.0.1:18540            # attach to a running gateway
    python -m evals.skill_bench.run ... --env-file .env  # or launch a gateway for this cell
    python -m evals.skill_bench.run ... --split-mode smoke --n-train 5 --n-test 5
    python -m evals.skill_bench.run --benchmark webshop --model luna --method none \
        --split-mode pilot ...                           # WebShop's pilot split

A cell runs in ``--out`` (default ``runs/<cell_id>``; layout in ``cell_records.py``):

1. **Prepare.** Take the cell lock (one live harness per cell), load the items, compute
   the protocol fingerprint and the git commit, check the environment (ALFWorld game
   files, WebShop's final manifest and ready server), attach
   to or launch the gateway and check that it is protocol-conformant and not paused.
2. **Learning** (H1.4, a new attempt, gateway run ``<cell_id>-aNN-<instance>``, its number
   reserved in ``attempt-NN/reservation.json`` before the gateway hears of it): a fresh
   method (and, for MIRIX, a fresh per-cell MIRIX server: new database and HOME, H8),
   ``setup``, then the training items in the
   seed-42 order, each ``before_task`` -> rollout with ``on_message`` -> score ->
   ``after_task`` -> ``wait_until_idle`` (``agent/rollout.py``). Each item's score is
   recorded on the scorer side only (``learning.jsonl``: the learning curve of H9), as
   soon as it is scored and before ``after_task``. The attempt is checkpointed after
   ``setup`` (``init``), after each item's score (``played``) and after its idle wait
   (``done``); see "Learning checkpoints" below.
3. **Freeze** (H1.5): ``finalize`` -> ``wait_until_idle`` -> ``snapshot``, saved with its
   sha256 (``snapshots/frozen.json``).
4. **Test** (H1.6): only ``before_task``, items concurrently (``--test-concurrency``), in
   the seed-42 order of the test split. Every item's result is appended to
   ``results.jsonl`` as it finishes, so a restarted harness resumes with the items that
   have no scored result.
5. **Check** (H10.4): the snapshot after the test phase (also of an interrupted one) must
   equal the frozen one. A complete cell gets its final ``results.jsonl``,
   ``learning.jsonl`` and ``summary.json`` (written before the attempt is marked
   complete); every attempt gets ``costs.json`` from its gateway ledger (H9 cost inputs)
   and records the model-visible code it ran (``protocol.code_fingerprint``), as
   does the ``launches.jsonl`` ``work`` line of every invocation that runs its learning
   or test phase (an invocation that runs nothing writes only its ``start`` line).

Failures (H8, ``infra/failures.py``): model failures are scored (the actor) or counted
(the method) and the cell goes on. In the **learning** phase (an interruption must never
make the experiment start over):

* an infrastructure failure or an interruption (a crash, SIGTERM/SIGINT, SIGKILL, the
  machine sleeping, the gateway or MIRIX dying, a failure the gateway's retries did not
  recover, and an infrastructure failure on the ``mirix`` or ``embedding`` role that the
  method carried on after) leaves the attempt ``learning_interrupted`` (exit code
  8; ``interrupted``, 130, for a signal). Its attempt number and gateway run stay; the
  next invocation **resumes** it from its latest learning checkpoint;
* a harness failure (the harness's or MIRIX's own, which a resume would repeat) leaves it
  ``learning_interrupted`` with ``needs_decision: "harness"`` and the cell
  ``learning_failed`` (exit code 3): a person picks ``--force-resume`` (resume from the
  checkpoint) or ``--new-attempt`` (invalidate it, abandon its gateway run and learn
  again from scratch). So is a checkpoint that is corrupt or that a resume could not
  restore exactly (``checkpoint_corrupt``, ``checkpoint_mismatch``);
* an attempt without learning checkpoints (a cell without learning items: the no-memory
  method or a smoke cell with ``--n-train 0``; or an older harness's) stops the same way,
  and the next invocation restarts it as a new attempt, since it has nothing to resume
  (setup and freeze are all it did).

Recorded, not stopped (the limits are ``infra/failures.py``'s):

* the provider refuses an actor call in the learning phase (an upstream 400/413 payload
  refusal, or a 404 "no endpoints"): the item ends unanswered and is scored 0
  (``end_reason`` and ``failure.code`` ``provider_unserved``, the row flagged
  ``provider_unserved: true``, a ``failures.jsonl`` row with recovery
  ``score_and_continue``), and the cell goes on. The same refusal on
  ``REFUSAL_STREAK_ITEMS`` (3) consecutive items stops the learning phase as a harness
  failure (``upstream_refusal_streak``, ``restart_cell``) after the last item's ``done``
  checkpoint, so ``--force-resume`` goes on with the next item. Learner and judge
  refusals inside a method are counted in the method's own records;
* a test item whose actor call the provider refused with "no endpoints" is re-run like
  any failed item (unscored, ``rerun_item``); its second consecutive such try
  (``PROVIDER_UNSERVED_TRIES``) is scored 0, flagged ``provider_unserved``;
* a MIRIX retrieval the method injected although it knows part of its input was lost
  (degraded) flags the row ``retrieval_degraded: true`` and the ``injections.jsonl`` row
  ``degraded: true``;
* a learning checkpoint's batch of MIRIX calls lost to infrastructure is rolled
  back and replayed; once the same checkpoint was rolled back ``MIRIX_BATCH_ROLLBACKS``
  (3) times in a row, the calls it lost again are accepted: listed in ``attempt.json``
  ``accepted_learning_calls`` and one ``failures.jsonl`` row (``mirix_learning``,
  ``accept_failed_batch``), and the learning phase goes on.

In the **test** phase only the failed item is marked (never scored) and
the next invocation re-runs it; those re-runs are listed in ``attempt.json``
(``test_reruns``) for the H10.3 audit. Each invocation does one step of this plan and
exits with the status's code (``cell_records.EXIT_CODES``); ``--max-learning-attempts``
lets it resume an interrupted learning phase at once, and ``schedule.py --auto-resume``
re-runs resumable cells with a backoff. Unreachable infrastructure before anything ran is
``infra_unavailable`` (resumable). Every live invocation holds its gateway's users lock
(shared), so no cleanup stops the gateway under it.

Learning checkpoints (H8; ``cell_records.py`` has the layout).
A learning phase resumes at item boundaries. ``checkpoints/`` keeps the latest two of:
``init`` (after ``setup``), per item ``played`` (scored and recorded, before
``after_task``) and ``done`` (after the idle wait and the ledger check). Each holds a
copy of the method's ``work_dir`` (``objects/**`` hard-linked), the method's own
``checkpoint()`` state, the sha256 of its ``snapshot()``, the lengths of
``learning.jsonl`` and ``injections.jsonl`` and the trajectories written so far, and for
MIRIX a ``pg_dump`` of the cell database taken under an exported snapshot (with the
table digest of that same snapshot) and HOME's manifest. A checkpoint is taken only when
no call of the run is in flight and MIRIX has no active learning run, and it is marked in
the gateway ledger (``POST /runs/<run_id>/mark``). A resume checks the host, the checkpoint
format, the gateway and the database (other code or another protocol is recorded as a new
version segment, not refused), stops what the dead invocation left running, marks the resume
in the ledger, moves everything written after the checkpoint into
``discarded/seg-SS/`` (the database is renamed aside, its extra rows exported, and
dropped), restores the checkpoint and verifies it (file hashes, the database digest
before MIRIX starts, then the method's snapshot), records the discarded segment in
``discarded.jsonl`` (its calls, recomputed from the ledger markers, and their cost so
far: a call still in flight is in ``pending_calls``, and ``costs.json``
``discarded_segments`` prices it once its final line is written), and continues: from
``done(p)`` with item p+1, from ``played(p)`` with item p's ``after_task`` on the view
rebuilt from its recorded trajectory (an image blob a crash lost is re-created from the
benchmark data first). Every item's effects
reach the method's live memory exactly once; the calls of a discarded try are listed and
priced, never used.

The no-memory method has no learning stream (H1.6): its cell is setup, freeze and test.

Protocol versions are recorded per attempt, not enforced across cells (a baseline fixed
after its run re-runs only its own cells; the conditions of each run are recorded instead
of refusing it). ``cell.json``
``cell`` holds the cell definition and the protocol its latest attempt started under. A
tree whose protocol fingerprint differs from it no longer refuses the cell (any other
difference of the definition still does: the items, splits, model and method of a cell
never change):

* every launch that runs work (a new attempt, a learning resume, a test resume) first
  freezes the tree when it is not the frozen value (``tools/freeze_protocol.auto_freeze``,
  under the record's lock, once; a missing or corrupt record is recorded and the attempt
  runs on), and records the version it runs under;
* a new attempt adopts the tree's protocol: the replaced values go to ``cell.json``
  ``protocol_history`` (``{fingerprint, constants_sha256, attempts}``, the attempts run
  under them) and the manifest is written before the attempt's number is reserved;
* a learning or test resume goes on under the tree's protocol, code and commit. When any
  of them (or the uncommitted changes of the tree) differ from the version before, the
  attempt gets a new version segment: ``attempt.json`` ``version_segments`` (from which
  learning position or test count, which protocol, code and commit, which files changed,
  ``protocol.attempt_versions``), ``versions/SS/version.json`` (the per-key constants and
  per-file hashes) and, for a dirty tree, ``versions/SS/tree.diff`` (``git apply`` on the
  commit rebuilds the tree), and ``versions_mixed`` says whether its protocol or code
  changed (the report labels such a cell a mixed version). ``--allow-code-change`` is
  accepted and no longer needed. The validity guards stay: the host, the checkpoint
  format, the frozen memory's hash and its equality after a resume (H10.4), the ledger and
  the MIRIX agent's gateway;
* a complete cell stays as it is (reported under its own version).

Every ``attempt.json`` records the ``protocol`` (``fingerprint``, ``constants_sha256``)
and ``code`` it started under, every ``work`` line of ``launches.jsonl`` the protocol,
code, version ``segment``, ``note`` and automatic freeze of its launch, and ``cell.json``'s
per-attempt summary each attempt's versions (``protocol.attempt_protocol`` reads an
attempt written before this from the history or the manifest). The audit and the report
show each cell's versions.

A cell is run once (H1.3). ``--new-attempt --note '<one line: why>'`` replaces an
attempt that already ran its test phase: the old attempt stays in the directory, marked
superseded (``attempt.json`` ``invalidation: superseded`` with the note, what it scored
and the attempt that replaced it), its gateway run is abandoned, and the report lists
both. Repeating the same ``--new-attempt --note`` resumes the
attempt it started instead of replacing it again. Starting a cell over in a new directory
(the old one moved out of the runs directory) is recorded with an operator note
(``--acknowledge-rerun``), also when it comes after the new directory's test phase or
names no earlier run here (both are recorded and the audit reports them). A smoke cell
(val items only) is outside the H1.3 matrix: running it again is only a warning.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import errno
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import sys
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import httpx

from evals.skill_bench import protocol, study
from evals.skill_bench.agent.actor import Actor
from evals.skill_bench.agent.rollout import (
    END_PROVIDER_UNSERVED,
    IDLE_WAIT_GRACE_S,
    STAGE_ACTOR,
    STAGE_AFTER_TASK,
    STAGE_BEFORE_TASK,
    STAGE_IDLE_WAIT,
    STAGE_PLAYED,
    EpisodeRun,
    LearningHookError,
    RolloutError,
    finish_learning,
    run_episode,
)
from evals.skill_bench.agent.trajectory import (
    Trajectory,
    canonical_json,
    externalize_blobs,
    to_jsonable,
    trajectory_view_from_record,
    unreadable_blobs,
    write_trajectory,
)
from evals.skill_bench.cell_records import (
    ATTEMPT_COMPLETE,
    ATTEMPT_FROZEN,
    ATTEMPT_INVALID,
    ATTEMPT_LEARNING,
    ATTEMPT_LEARNING_INTERRUPTED,
    ATTEMPT_TESTING,
    ATTEMPT_VIOLATION,
    CHECKPOINT_DONE,
    CHECKPOINT_INIT,
    CHECKPOINT_PLAYED,
    CHECKPOINT_RECORD,
    EXIT_CODES,
    FAILURE_CHECKPOINT_CORRUPT,
    FAILURE_CHECKPOINT_MISMATCH,
    FAILURE_CHECKPOINT_NOT_QUIESCENT,
    FAILURE_DISK_FULL,
    LAUNCH_START,
    LAUNCH_WORK,
    LEDGER_EVENT_CHECKPOINT,
    LEDGER_EVENT_RESUME,
    MERGED_MARKER,
    NEEDS_DECISION_HARNESS,
    NOTE_RERUN_ACKNOWLEDGED,
    RECOVERY_RESUME_FROM_CHECKPOINT,
    STATUS_BUDGET_PAUSED,
    STATUS_BUSY,
    STATUS_COMPLETE,
    STATUS_CRASHED,
    STATUS_INFRA_UNAVAILABLE,
    STATUS_INTERRUPTED,
    STATUS_LEARNING_FAILED,
    STATUS_LEARNING_INTERRUPTED,
    STATUS_REFUSED,
    STATUS_RUNNING,
    STATUS_TEST_INCOMPLETE,
    STATUS_VIOLATION,
    AttemptPaths,
    CellPaths,
    JsonlWriter,
    LedgerLine,
    canonical_sha256,
    cell_gateway_runs,
    checkpoint_id,
    checkpoint_ordinal,
    consecutive_rollbacks,
    discarded_call_ids,
    discarded_segment_costs,
    git_info,
    host_info,
    host_text,
    now,
    parse_checkpoint_id,
    read_discarded,
    read_json,
    read_jsonl,
    read_ledger_lines,
    records_digest,
    same_host,
    segment_calls,
    sha256_lines,
    summarize_ledger,
    tree_pathspecs,
    write_json_atomic,
)
from evals.skill_bench.envs.agent_context import context_record, env_agent_context
from evals.skill_bench.envs.base import Env, EpisodeResult, Split, Task, new_cell_salt
from evals.skill_bench.infra.failures import (
    MIRIX_BATCH_ROLLBACKS,
    MIRIX_LEARNING,
    MIRIX_RETRIEVAL,
    REFUSAL_CLASSES,
    REFUSAL_NO_ENDPOINT,
    WEBSHOP_SERVER,
    Failure,
    FailureKind,
    FailureLog,
    LearningAttemptInvalid,
    Recovery,
    classify_exception,
    learning_refusal_streak,
    line_phase,
    pending_test_items,
    recovery_for,
    unrecovered_infra_calls,
    unserved_retries,
)
from evals.skill_bench.infra.gateway import (
    LOCAL_TOKEN,
    gateway_users_lock,
    launch_gateway,
)
from evals.skill_bench.infra.mirix_server import (
    EXPORT_FILE,
    HOME_MANIFEST_FILE,
    MIRIX_SESSION_HEADER_ENV,
    CellDatabaseConflict,
    CellServerSpec,
    CheckpointStore,
    MirixServer,
    MirixServerError,
    PostgresAdmin,
    ServerMode,
    build_server_env,
    capture_home,
    check_llm_contract,
    dotenv_files_visible_to_mirix,
    home_manifest,
    move_new_home_files,
    probe_llm_contract,
    restore_home,
)
from evals.skill_bench.infra.processes import (
    OwnedProcessError,
    OwnerLock,
    cleanup_owned,
    find_free_port,
    lock_is_held,
    process_start_time,
)
from evals.skill_bench.methods.base import (
    ITEM_HEADER,
    OBJECTS_DIR,
    CellContext,
    CheckpointMismatch,
    FrozenMemoryMismatch,
    LearningResume,
    MemoryMethod,
    checkpoint_state_json,
    method_class,
)

log = logging.getLogger("evals.skill_bench.run")

SplitMode = Literal["protocol", "smoke", "pilot"]
#: ``pilot``: a benchmark's extra pilot split
#: (``protocol.EXTRA_SPLIT_SIZES``, WebShop's 50 items), played once by the no-memory
#: method to measure the headroom before the benchmark enters the formal protocol.
SPLIT_MODES: tuple[str, ...] = ("protocol", "smoke", "pilot")
#: The method a pilot cell runs (the no-memory method only).
PILOT_METHOD = "none"

DEFAULT_TEST_CONCURRENCY = 4
#: Bound of every ``wait_until_idle``; a timeout is INFRA (H8), which restarts a
#: learning attempt, so it only guards against a hung method and is generous.
DEFAULT_IDLE_TIMEOUT_S = 7200.0
#: Bound of ``finalize`` (MIRIX flushes the remaining sessions and waits for that run,
#: H7.2); passed to the MIRIX method as ``options["mirix_finalize_timeout_s"]``.
DEFAULT_FINALIZE_TIMEOUT_S = 7200.0
#: Bounds of ``setup`` and ``snapshot``: a hung method must not hang the cell forever.
SETUP_TIMEOUT_S = 900.0
SNAPSHOT_TIMEOUT_S = 600.0
#: Bound of the after-test snapshot taken when a test phase is interrupted.
CANCEL_SNAPSHOT_TIMEOUT_S = 60.0
#: A port picked for a MIRIX server can be taken by another cell before it binds.
MIRIX_PORT_TRIES = 3
#: The method whose cells get a per-cell MIRIX server (H7.2, H8).
MIRIX_METHOD = "mirix"
#: Size bound of an ``--acknowledge-rerun`` note.
MAX_OPERATOR_NOTE_BYTES = 64 * 1024
#: The ``cell.json`` ``cell`` keys that name the protocol rather than the cell: a tree
#: under another protocol does not refuse the cell over them, its next new attempt adopts
#: them (``_adopt_protocol``).
PROTOCOL_KEYS: tuple[str, ...] = ("protocol_fingerprint", "protocol_constants_sha256")
#: How long a learning checkpoint waits for the run to be quiescent (no call in flight,
#: no active MIRIX learning run) before it is ``checkpoint_not_quiescent`` (INFRA).
CHECKPOINT_QUIESCE_S = 60.0
CHECKPOINT_POLL_S = 0.1
#: How long a SIGTERM waits for a checkpoint being written to finish (shielded).
CHECKPOINT_SHIELD_S = 60.0
#: How long a learning resume waits for the dead invocation's MIRIX server to leave the
#: cell database before it gives up (INFRA ``mirix_server``).
DB_RELEASE_S = 60.0
#: Learning checkpoints kept per attempt: the latest and its predecessor.
CHECKPOINTS_KEPT = 2
#: ``attempt.json`` ``checkpoint_format``: the attempt writes learning checkpoints, in
#: this layout (``checkpoint.json``, ``attempt_files``, the move-aside of a discarded
#: segment). A resume refuses an attempt of another format: bump it whenever
#: that layout changes, since run.py itself is outside the code digest a resume compares.
CHECKPOINT_FORMAT = 1
#: ``attempt.json`` ``resume_from`` naming no checkpoint: roll back to the empty state.
RESUME_FROM_START = "start"
#: ``snapshot_checks`` kind of the snapshot a learning resume rebuilt.
SNAPSHOT_CHECK_LEARNING_RESUME = "learning_resume"
#: Errnos of a full disk or quota (``disk_full``, INFRA).
DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})
RERUN_NOTE_HINT = "--acknowledge-rerun NOTE: who re-runs the cell, when and why"


#: A version segment's directory in its attempt (``attempt-NN/versions/SS/``), its
#: ``version.json`` (the per-key constants and per-file hashes of the version) and, for a
#: tree with uncommitted changes, ``tree.diff`` (``capture_tree``).
VERSIONS_DIR = "versions"
VERSION_FILE = "version.json"
TREE_DIFF = "tree.diff"
#: Bounds of the untracked files a tree diff carries (more, or larger, are listed in
#: ``untracked_skipped``).
TREE_MAX_UNTRACKED = 200
TREE_MAX_FILE_BYTES = 5 * 2**20
#: The longest ``--note`` (one line: why this invocation runs).
MAX_NOTE_CHARS = 1000
NOTE_RULE = (
    "--note is one line: why this run (for example 'rerun with the fixed MIRIX curator'), "
    f"at most {MAX_NOTE_CHARS} characters"
)


def capture_tree(
    dest: Path,
    *,
    pathspecs: Sequence[str],
    repo: Path = protocol.REPO_ROOT,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """The uncommitted changes of the checkout under ``pathspecs``, as one binary patch at
    ``dest`` that ``git apply`` on the commit turns back into the tree: the tracked
    changes (``git diff --binary HEAD``) and each untracked file (at most
    :data:`TREE_MAX_UNTRACKED` regular files of at most :data:`TREE_MAX_FILE_BYTES`; the
    others are named in ``untracked_skipped``). Blocking; never raises. Returns
    ``{dirty, diff, diff_sha256, diff_bytes, untracked, untracked_skipped, error}``:
    ``dirty`` False and no file written for a clean tree, ``error`` (and ``dirty`` None)
    when git failed."""
    import subprocess

    result: dict[str, Any] = {
        "dirty": None,
        "diff": None,
        "diff_sha256": None,
        "diff_bytes": 0,
        "untracked": 0,
        "untracked_skipped": [],
        "error": None,
    }

    def git(*args: str, stdout: Any = subprocess.PIPE) -> Any:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            stdout=stdout,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )

    try:
        status = git(
            "status", "--porcelain", "-z", "--untracked-files=all", "--", *pathspecs
        )
        if status.returncode != 0:
            raise RuntimeError(
                "git status: " + status.stderr.decode("utf-8", "replace").strip()
            )
        if not status.stdout.strip(b"\0 \n"):
            result["dirty"] = False
            return result
        others = git(
            "ls-files", "--others", "--exclude-standard", "-z", "--", *pathspecs
        )
        if others.returncode != 0:
            raise RuntimeError(
                "git ls-files: " + others.stderr.decode("utf-8", "replace").strip()
            )
        names = [
            n for n in others.stdout.decode("utf-8", "surrogateescape").split("\0") if n
        ]
        skipped: list[str] = []
        added = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            tracked = git("diff", "--binary", "HEAD", "--", *pathspecs, stdout=handle)
            if tracked.returncode != 0:
                raise RuntimeError(
                    "git diff: " + tracked.stderr.decode("utf-8", "replace").strip()
                )
            for name in names:
                full = repo / name
                if (
                    added >= TREE_MAX_UNTRACKED
                    or full.is_symlink()
                    or not full.is_file()
                    or full.stat().st_size > TREE_MAX_FILE_BYTES
                ):
                    skipped.append(name)
                    continue
                handle.flush()
                new = git(
                    "diff",
                    "--no-index",
                    "--binary",
                    "--",
                    "/dev/null",
                    name,
                    stdout=handle,
                )
                if new.returncode not in (0, 1):  # 1: the files differ, as they do
                    skipped.append(name)
                    continue
                added += 1
            handle.flush()
            os.fsync(handle.fileno())
        result.update(
            dirty=True,
            diff=str(dest),
            diff_sha256=_file_sha256(dest),
            diff_bytes=dest.stat().st_size,
            untracked=added,
            untracked_skipped=skipped,
        )
    except Exception as exc:  # noqa: BLE001 - recorded; the attempt runs on
        result.update(dirty=None, diff=None, error=f"{type(exc).__name__}: {exc}"[:500])
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
    return result


#: The most hashed files a segment's ``tree.unreproducible`` names (its ``count`` is all).
TREE_MAX_UNREPRODUCIBLE = 200


def hashed_repo_paths(
    files: Mapping[str, Any] | None, code: Mapping[str, Any]
) -> set[str]:
    """Every file a version hashes, as a path from the repository's top (pure): the
    fingerprinted files and the harness code (``files`` and ``code['harness_sha256']``,
    relative to ``evals/skill_bench``) and the MIRIX code with its launcher
    (``code['mirix_sha256']``, relative to ``mirix/``)."""
    import posixpath

    bench = protocol.SKILL_BENCH_DIR.relative_to(protocol.REPO_ROOT).as_posix()
    package = protocol.MIRIX_PACKAGE_DIR.relative_to(protocol.REPO_ROOT).as_posix()
    harness = code.get("harness_sha256")
    mirix = code.get("mirix_sha256")
    paths = [f"{bench}/{p}" for p in (files or {})]
    paths += [f"{bench}/{p}" for p in (harness if isinstance(harness, Mapping) else {})]
    paths += [f"{package}/{p}" for p in (mirix if isinstance(mirix, Mapping) else {})]
    return {posixpath.normpath(p) for p in paths}


def unreproducible_files(
    hashed: Iterable[str],
    tree: Mapping[str, Any],
    *,
    pathspecs: Sequence[str],
    repo: Path = protocol.REPO_ROOT,
    timeout_s: float = 60.0,
    max_files: int = TREE_MAX_UNREPRODUCIBLE,
) -> dict[str, Any]:
    """The hashed files (``hashed``, paths from the top of ``repo``) that the commit plus
    the segment's tree diff (``tree``, :func:`capture_tree`'s record) cannot rebuild
    (blocking; never raises): a file git ignores (neither ``git status`` nor
    the diff sees it; ``why`` ``ignored``) and an untracked file the diff does not carry
    (skipped, or no diff was written; ``not_captured``). Only files under ``pathspecs``
    are checked (every file a version hashes is). Returns ``{files, count, error}``:
    ``files`` sorted by path, at most ``max_files`` of them; both None and ``error`` set
    when git failed."""
    import subprocess

    def others(*extra: str) -> set[str]:
        done = subprocess.run(
            ["git", "ls-files", "--others", *extra, "-z", "--", *pathspecs],
            cwd=repo,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        if done.returncode != 0:
            raise RuntimeError(
                "git ls-files: " + done.stderr.decode("utf-8", "replace").strip()
            )
        out = done.stdout.decode("utf-8", "surrogateescape")
        return {n for n in out.split("\0") if n}

    try:
        wanted = set(hashed)
        untracked = others() & wanted
        ignored = (
            others("--ignored", "--exclude-standard") & untracked
            if untracked
            else set()
        )
    except Exception as exc:  # noqa: BLE001 - recorded; the attempt runs on
        return {
            "files": None,
            "count": None,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    skipped = set(tree.get("untracked_skipped") or ())
    lost = []
    for path in sorted(untracked):
        if path in ignored:
            lost.append({"path": path, "why": "ignored"})
        elif not tree.get("diff") or path in skipped:
            lost.append({"path": path, "why": "not_captured"})
    return {"files": lost[:max_files], "count": len(lost), "error": None}


def start_over_hint(smoke: bool) -> str:
    """How to start a cell over, for a refusal: the alternative to
    ``--new-attempt --note`` (a new attempt in the same directory). The old directory
    leaves the runs directory, not only its name: the audit reads every cell directory
    there, and two with one cell id each report the other's runs as the cell run twice.
    Its gateway runs stay, so the audit sees them."""
    move = (
        "move its directory out of the runs directory (for example to "
        "runs-superseded/) and leave its gateway runs where they are"
    )
    if smoke:
        return (
            f"A smoke cell (val items) may be run again: {move}, then run it "
            "again. The audit lists the earlier runs as a warning (smoke_cell_rerun)."
        )
    return (
        f"Or start the whole cell over in a new directory: {move}; record why with "
        f"{RERUN_NOTE_HINT} (the audit lists the earlier runs)."
    )


EnvFactory = Callable[[str], Env]
MirixPreflight = Callable[[str, Path], Awaitable[Mapping[str, Any]]]
MethodFactory = Callable[[], MemoryMethod]
MirixServerFactory = Callable[[CellServerSpec, ServerMode], MirixServer]
GitInfo = Callable[[], Mapping[str, Any]]
#: :func:`capture_tree`'s signature (a seam for tests: they never diff the real checkout).
TreeCapture = Callable[..., Mapping[str, Any]]
#: :func:`unreproducible_files`'s signature (``CellRunner.tree_check``).
TreeCheck = Callable[..., Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Cell definition
# ---------------------------------------------------------------------------


def resolve_model(name: str) -> str:
    """The protocol model id for a key (``luna``, ``luna56``) or an id, matched whole."""
    if name in protocol.MODELS:
        return protocol.MODELS[name]
    if name in protocol.CHAT_MODELS:
        return name
    raise ValueError(
        f"unknown model {name!r}; use one of {sorted(protocol.MODELS)} or "
        f"{list(protocol.CHAT_MODELS)}"
    )


def model_key(model: str) -> str:
    return next(k for k, v in protocol.MODELS.items() if v == model)


@dataclass(frozen=True)
class CellSpec:
    """One cell of the H1.3 matrix, in protocol mode or as a smoke subset, or a
    benchmark's pilot (the no-memory method on the pilot split)."""

    benchmark: str
    model: str
    method: str
    split_mode: str = "protocol"
    n_train: int | None = None
    n_test: int | None = None

    def __post_init__(self) -> None:
        if self.benchmark not in protocol.BENCHMARKS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}")
        object.__setattr__(self, "model", resolve_model(self.model))
        # A method the study removed: its code is gone, and the refusal names why.
        study.refuse_removed([self.method])
        if self.method not in protocol.METHODS:
            raise ValueError(
                f"unknown method {self.method!r}; protocol methods: {protocol.METHODS}"
            )
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"split mode must be one of {SPLIT_MODES}")
        if self.split_mode in ("protocol", "pilot"):
            if self.n_train is not None or self.n_test is not None:
                raise ValueError(
                    f"{self.split_mode} mode runs whole splits; --n-train and --n-test "
                    "are for --split-mode smoke"
                )
            if self.pilot and "pilot" not in protocol.EXTRA_SPLIT_SIZES.get(
                self.benchmark, {}
            ):
                raise ValueError(f"{self.benchmark} has no pilot split")
            if self.pilot and self.method != PILOT_METHOD:
                raise ValueError(f"a pilot cell runs the {PILOT_METHOD!r} method only")
        else:
            if self.n_train is None or self.n_test is None:
                raise ValueError("smoke mode needs --n-train and --n-test")
            if self.n_train < 0 or self.n_test < 1:
                raise ValueError("smoke mode needs n_train >= 0 and n_test >= 1")
            sizes = protocol.SPLIT_SIZES[self.benchmark]
            if self.n_train > sizes["train"] or self.n_test > sizes["val"]:
                raise ValueError(
                    f"{self.benchmark} has {sizes['train']} train and {sizes['val']} val "
                    "items"
                )

    @property
    def model_key(self) -> str:
        return model_key(self.model)

    @property
    def smoke(self) -> bool:
        return self.split_mode == "smoke"

    @property
    def pilot(self) -> bool:
        return self.split_mode == "pilot"

    @property
    def cell_id(self) -> str:
        """``<benchmark>-<model key>-<method>``, with ``-smoke-l<n>-t<n>`` for a smoke cell
        and ``-pilot`` for a pilot cell.

        Also the prefix of the gateway ``run_id`` of each attempt (``<cell_id>-aNN``).
        """
        base = f"{self.benchmark}-{self.model_key}-{self.method}"
        if self.smoke:
            base += f"-smoke-l{self.n_train}-t{self.n_test}"
        if self.pilot:
            base += "-pilot"
        return base

    @property
    def learns(self) -> bool:
        """The no-memory method only runs the test phase (H1.6)."""
        return self.method != "none"

    @property
    def test_split(self) -> str:
        """Smoke cells never touch the test split: their test items come from val;
        a pilot cell plays the pilot split."""
        return "val" if self.smoke else ("pilot" if self.pilot else "test")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "benchmark": self.benchmark,
            "model": self.model,
            "model_key": self.model_key,
            "method": self.method,
            "split_mode": self.split_mode,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


def seeded_order(tasks: Sequence[Task]) -> list[Task]:
    """The tasks in the protocol's seed-42 order (``random.Random(42).shuffle`` of the manifest order)."""
    order = list(tasks)
    random.Random(protocol.SEED).shuffle(order)
    return order


def select_items(env: Env, spec: CellSpec) -> tuple[list[Task], list[Task]]:
    """The learning stream and the test items of a cell (H1.1, H1.2, H1.4; blocking).

    * learning: the train split in the seed-42 order, the same for every method (none for
      the no-memory method); a smoke cell takes the head of that order;
    * test: the test split in the seed-42 order (the manifest groups ALFWorld items by task
      type, and the order items reach ``before_task`` must not carry that, H5; scoring
      does not depend on it); a smoke cell takes the head of the val split in the seed-42
      order and never loads the test split; a pilot cell (no-memory only) plays the
      pilot split in the seed-42 order.
    """
    learning: list[Task] = []
    if spec.learns and (not spec.smoke or (spec.n_train or 0) > 0):
        learning = seeded_order(env.load_tasks("train"))
        if spec.smoke:
            learning = learning[: spec.n_train]
    if spec.smoke:
        test = seeded_order(env.load_tasks("val"))[: spec.n_test]
    else:
        test = seeded_order(env.load_tasks(cast(Split, spec.test_split)))
    for task in (*learning, *test):
        if task.benchmark != spec.benchmark:
            raise ValueError(f"{task.raw_id} is a {task.benchmark} task")
    if len({t.raw_id for t in test}) != len(test):
        raise ValueError("test items must have distinct raw ids")
    return learning, test


def default_env(benchmark: str) -> Env:
    """The benchmark's environment with its default data (blocking: constructors verify assets)."""
    if benchmark == "alfworld":
        from evals.skill_bench.envs.alfworld import AlfWorldEnv

        return AlfWorldEnv()
    if benchmark == "webshop":
        from evals.skill_bench.envs.webshop import WebShopEnv

        # The frozen files only; the server is checked by its preflight.
        return WebShopEnv()
    raise ValueError(f"unknown benchmark {benchmark!r}")


def default_mirix_server(spec: CellServerSpec, mode: ServerMode) -> MirixServer:
    return MirixServer(spec, mode=mode)


# ---------------------------------------------------------------------------
# Options and outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOptions:
    """How to run a cell (not part of the protocol; recorded in the manifest).

    Gateway: ``gateway_url`` attaches to a running gateway, whose ledgers live in
    ``gateway_runs_dir/<run_id>/gateway.jsonl``. Without it the cell launches its own
    gateway (``gateway_port``, default a free port; the key comes from ``env_file`` or the
    environment) and stops it at the end.
    """

    out_dir: Path
    gateway_url: str | None = None
    gateway_runs_dir: Path = protocol.RUNS_DIR
    gateway_port: int | None = None
    env_file: Path | None = None
    balance_floor_usd: float | None = None
    #: The spend budget of a launched gateway (USD, ``infra/gateway.py``); None: none.
    budget_usd: float | None = None
    test_concurrency: int = DEFAULT_TEST_CONCURRENCY
    idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S
    finalize_timeout_s: float = DEFAULT_FINALIZE_TIMEOUT_S
    #: Start a new learning attempt although the last one is frozen, testing, complete or
    #: in violation; one that ran its test phase needs a ``note`` and is superseded.
    new_attempt: bool = False
    #: Learning steps this invocation may take: 1 (default) stops after an interrupted
    #: learning phase; more resumes it at once from its checkpoint (never after a harness
    #: failure).
    max_learning_attempts: int = 1
    #: Resume a learning phase a harness failure stopped (``needs_decision``), from its
    #: checkpoint, after a person has looked (recorded as ``operator_resume``).
    force_resume: bool = False
    #: Keep the post-crash MIRIX database of a learning resume (``<db>_dSS``) for
    #: forensics instead of dropping it once its rows are exported.
    keep_discarded_db: bool = False
    #: No longer needed (accepted so older command lines still parse): a resume under
    #: another commit, code or protocol runs and is recorded in ``version_segments``.
    allow_code_change: bool = False
    #: Accept a gateway whose settings differ from the protocol (tests and debugging only).
    allow_nonconformant_gateway: bool = False
    #: Check the environment's data before a cell starts.
    preflight: bool = True
    #: Content-addressed store of trajectory images (default ``<out>/../_blobs``).
    blob_dir: Path | None = None
    #: An operator's note acknowledging that this cell directory runs the cell again after
    #: an earlier directory of the same cell id ran test items: recorded in
    #: ``cell.json`` ``operator_notes`` with the earlier runs it covers.
    acknowledge_rerun: Path | None = None
    #: One line: why this invocation runs (``--note``). Recorded on its work line and, for
    #: a new attempt, in ``attempt.json`` ``note``; required by ``--new-attempt`` over an
    #: attempt that already ran its test phase (that attempt is then superseded).
    note: str | None = None

    def __post_init__(self) -> None:
        # Every path absolute, once, here: the MIRIX server runs in its
        # attempt directory and a launched gateway in the repository root, so a relative
        # path handed to them would name another directory. absolute(), not resolve():
        # a symlinked runs directory keeps the spelling the operator gave.
        object.__setattr__(self, "out_dir", Path(self.out_dir).absolute())
        object.__setattr__(
            self, "gateway_runs_dir", Path(self.gateway_runs_dir).absolute()
        )
        for name in ("env_file", "blob_dir", "acknowledge_rerun"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).absolute())
        if self.test_concurrency < 1:
            raise ValueError("test concurrency must be at least 1")
        if self.idle_timeout_s <= 0 or self.finalize_timeout_s <= 0:
            raise ValueError("the idle and finalize timeouts must be positive")
        if self.max_learning_attempts < 1:
            raise ValueError("max_learning_attempts must be at least 1")
        if self.budget_usd is not None and not self.budget_usd > 0:
            raise ValueError("the budget must be positive")
        if self.note is not None:
            note = str(self.note).strip()
            if not note or "\n" in note or "\r" in note or len(note) > MAX_NOTE_CHARS:
                raise ValueError(NOTE_RULE)
            object.__setattr__(self, "note", note)

    @property
    def blobs(self) -> Path:
        return self.blob_dir or self.out_dir.parent / "_blobs"

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "gateway_url": self.gateway_url,
            "gateway_runs_dir": str(self.gateway_runs_dir),
            "gateway_port": self.gateway_port,
            "test_concurrency": self.test_concurrency,
            "idle_timeout_s": self.idle_timeout_s,
            "finalize_timeout_s": self.finalize_timeout_s,
            "new_attempt": self.new_attempt,
            "max_learning_attempts": self.max_learning_attempts,
            "force_resume": self.force_resume,
            "keep_discarded_db": self.keep_discarded_db,
            "allow_code_change": self.allow_code_change,
            "allow_nonconformant_gateway": self.allow_nonconformant_gateway,
            "preflight": self.preflight,
            "blob_dir": str(self.blobs),
            "acknowledge_rerun": (
                str(self.acknowledge_rerun) if self.acknowledge_rerun else None
            ),
            "note": self.note,
        }


@dataclass(frozen=True)
class CellOutcome:
    """What one invocation achieved; ``exit_code`` is the process exit code."""

    cell_id: str
    status: str
    attempt: int | None = None
    detail: str = ""
    pending: int = 0
    #: The H8 kind of the failure that ended a learning phase (``learning_failed``).
    failure_kind: str | None = None

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "attempt": self.attempt,
            "detail": self.detail,
            "pending": self.pending,
            "failure_kind": self.failure_kind,
        }


#: Why an attempt was invalidated (``attempt.json`` ``invalidation``):
#: ``learning_failure`` (H8: a learning phase restarted as a new attempt),
#: ``operator_reroll`` (``--new-attempt`` before any test item was scored) and
#: ``superseded`` (``--new-attempt --note`` after its test phase ran: kept, listed with
#: its note).
INVALIDATION_LEARNING = "learning_failure"
INVALIDATION_OPERATOR = "operator_reroll"
INVALIDATION_SUPERSEDED = "superseded"


class _Refused(Exception):
    """A precondition failed before any attempt was consumed (or a resume was refused)."""


class _Paused(Exception):
    """The gateway is paused on the budget: nothing is started."""


#: The roles whose unrecovered infrastructure failures in the learning phase restart the
#: attempt although the method carried on: MIRIX's own calls and embeddings.
LEARNING_CHECK_ROLES: tuple[str, ...] = ("mirix", "embedding")


class _LedgerReader:
    """The run's gateway ledger, read incrementally (blocking): each :meth:`read` parses
    only what the gateway appended since the last one, so checking before every learning
    checkpoint stays linear in the ledger. It keeps where each call started, which calls
    have their final line, the final lines of :data:`LEARNING_CHECK_ROLES` and the
    ``checkpoint`` / ``resume`` markers, each with its byte offset."""

    def __init__(self, roles: Sequence[str] = LEARNING_CHECK_ROLES) -> None:
        self.roles = frozenset(roles)
        self.path: Path | None = None
        self.offset = 0
        #: Every started call: its ``call_started`` offset (for a call without one, its
        #: final line's) and gateway start only (a long learning phase has tens of
        #: thousands of calls).
        self.started: dict[str, LedgerLine] = {}
        #: The started calls without a final line yet.
        self.pending: dict[str, LedgerLine] = {}
        #: ``call_started`` offsets in ledger order, and their call ids.
        self._offsets: list[int] = []
        self._order: list[str] = []
        self.finished: set[str] = set()
        self.role_lines: list[dict[str, Any]] = []
        self.markers: list[LedgerLine] = []

    def read(self, path: Path) -> None:
        if path != self.path:
            self.__init__(tuple(self.roles))  # type: ignore[misc]
            self.path = path
        try:
            with path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return
        end = data.rfind(b"\n") + 1  # a line being written is read next time
        position = self.offset
        for raw in data[:end].split(b"\n")[:-1]:
            line_offset, position = position, position + len(raw) + 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue  # a torn line (a gateway killed mid-write)
            if not isinstance(row, dict):
                continue
            event = row.get("event")
            call_id = str(row.get("call_id") or "")
            if event == "call_started" and call_id:
                line = LedgerLine(
                    line_offset, {"gateway_start": row.get("gateway_start")}
                )
                self.started[call_id] = line
                if call_id not in self.finished:
                    self.pending[call_id] = line
                self._offsets.append(line_offset)
                self._order.append(call_id)
            elif event == "call" and call_id:
                if call_id not in self.started:
                    # A final line without a call_started line (an older gateway
                    # answered a call it held during a pause 402 before writing one):
                    # placed by its own offset.
                    self.started[call_id] = LedgerLine(
                        line_offset, {"gateway_start": row.get("gateway_start")}
                    )
                    self._offsets.append(line_offset)
                    self._order.append(call_id)
                self.finished.add(call_id)
                self.pending.pop(call_id, None)
                if row.get("role") in self.roles:
                    self.role_lines.append(row)
            elif event in (LEDGER_EVENT_CHECKPOINT, LEDGER_EVENT_RESUME):
                self.markers.append(LedgerLine(line_offset, row))
        self.offset += end

    def open_calls(
        self, *, exclude: frozenset[str], gateway_start: str | None
    ) -> list[str]:
        """Calls started and not finished: in flight, unless the gateway that started
        them is gone (another ``gateway_start``: they never finish) or a learning resume
        discarded them (a dead invocation's call the gateway may still be finishing)."""
        return [
            call_id
            for call_id, line in self.pending.items()
            if call_id not in exclude
            and (
                gateway_start is None
                or line.row.get("gateway_start") in (None, gateway_start)
            )
        ]

    def started_after(self, offset: int) -> list[str]:
        """The calls whose ``call_started`` line lies after byte ``offset``."""
        index = bisect.bisect_right(self._offsets, offset)
        return list(dict.fromkeys(self._order[index:]))

    def started_since(self, offset: int) -> list[str]:
        """The calls whose ``call_started`` line starts at byte ``offset`` or later (an
        :attr:`offset` this reader had reached: the lines written since)."""
        index = bisect.bisect_left(self._offsets, offset)
        return list(dict.fromkeys(self._order[index:]))

    def marker(self, event: str, **match: Any) -> LedgerLine | None:
        """The last marker of ``event`` whose fields equal ``match``."""
        for line in reversed(self.markers):
            if line.row.get("event") == event and all(
                line.row.get(k) == v for k, v in match.items()
            ):
                return line
        return None


class _InfraUnavailable(Exception):
    """The cell's infrastructure is unreachable before anything ran (the gateway, the
    MIRIX Postgres, the WebShop server): transient, the next invocation goes on."""


class _LearningFailed(LearningAttemptInvalid):
    """The learning phase stopped on ``failure`` at ``stage`` (item ``position``).

    A :class:`~evals.skill_bench.infra.failures.LearningAttemptInvalid`, so the rollout
    classifies it by its own failure when it comes out of the ``on_played`` hook.
    ``resume_from`` names the checkpoint a resume must restore when the latest one
    already holds the failure's effects (a lost ``mirix`` / ``embedding`` call before its
    marker; :data:`RESUME_FROM_START` for the empty state). ``keep_cause``: the failure
    stopped a resume before it changed anything (the database owner check), so the
    attempt keeps what stopped it before (its failure, ``resume_from`` and interruption,
    which the resume a person decides still needs) and records this one next to it
    (``resume_conflict``)."""

    def __init__(
        self,
        failure: Failure,
        *,
        stage: str,
        position: int | None = None,
        resume_from: str | None = None,
        keep_cause: bool = False,
    ) -> None:
        super().__init__(failure)
        self.stage = stage
        self.position = position
        self.resume_from = resume_from
        self.keep_cause = keep_cause


def classify_learning_exception(exc: BaseException) -> Failure:
    """:func:`~evals.skill_bench.infra.failures.classify_exception` plus the learning
    checkpoints' own failures: a restore that does not reproduce the checkpoint
    (``checkpoint_mismatch``, HARNESS) and a full disk (``disk_full``, INFRA)."""
    detail = f"{type(exc).__name__}: {exc}"[:500]
    if isinstance(exc, CheckpointMismatch):
        return Failure(
            FailureKind.HARNESS, FAILURE_CHECKPOINT_MISMATCH, None, None, detail
        )
    if isinstance(exc, OSError) and exc.errno in DISK_FULL_ERRNOS:
        return Failure(FailureKind.INFRA, FAILURE_DISK_FULL, None, None, detail)
    return classify_exception(exc)


def _harness_stop_choices(code: str | None, *, checkpointed: bool) -> str:
    """What a person may do about a learning phase a HARNESS failure stopped. A
    ``checkpoint_mismatch`` comes from the memory the method rebuilds from its own
    checkpoint: ``--force-resume`` rebuilds it the same way and stops again, so it is
    not offered."""
    if code == FAILURE_CHECKPOINT_MISMATCH:
        return (
            "The memory rebuilt from the checkpoint does not match it, so "
            "--force-resume would stop the same way: a person fixes the code that "
            "rebuilds it, or picks --new-attempt (learn again from scratch)"
        )
    return "A person picks --force-resume (resume from its checkpoint) or " + (
        "--new-attempt (learn again from scratch)"
        + (
            ""
            if checkpointed
            else " (an attempt without checkpoints restarts as a new attempt)"
        )
    )


def learning_recovery(failure: Failure) -> str:
    """The recovery of a learning-phase failure (``failures.jsonl``, the learning row): an
    infrastructure failure resumes from the checkpoint, a harness failure waits for a
    person (who may start the cell over, ``restart_cell``)."""
    if failure.kind is FailureKind.INFRA:
        return RECOVERY_RESUME_FROM_CHECKPOINT
    return recovery_for("learning", failure).value


def _record_learning_failure(
    path: Path,
    failure: Failure,
    *,
    item: str | None,
    attempt: int,
    stage: str | None,
    checkpointed: bool,
    recovery: str | None = None,
) -> None:
    """One ``failures.jsonl`` row of a learning-phase failure (the
    :class:`~evals.skill_bench.infra.failures.FailureLog` row with its ``stage``, and
    the learning recovery of an attempt that has checkpoints unless ``recovery`` names
    it: ``accept_failed_batch`` for a MIRIX batch accepted after its rollbacks)."""
    if recovery is None:
        recovery = (
            learning_recovery(failure)
            if checkpointed
            else recovery_for("learning", failure).value
        )
    _append(
        path,
        {
            "at": now(),
            "phase": "learning",
            "item": item,
            "attempt": attempt,
            "stage": stage,
            "recovery": recovery,
            **failure.to_dict(),
        },
    )


def _accepted_call_ids(state: Mapping[str, Any]) -> frozenset[str]:
    """The lost learning calls ``attempt.json`` records as accepted: after their batch's
    rollbacks (``accepted_learning_calls``) or as calls of a degraded retrieval the
    method injected (``degraded_retrieval_calls``)."""
    return frozenset(
        str(row.get("call_id"))
        for key in ("accepted_learning_calls", DEGRADED_RETRIEVAL_CALLS)
        for row in state.get(key) or ()
        if isinstance(row, Mapping) and row.get("call_id")
    )


#: ``attempt.json``: the lost learning calls of degraded retrievals the method injected.
DEGRADED_RETRIEVAL_CALLS = "degraded_retrieval_calls"


#: The MIRIX method's retrieval record (``methods/mirix.RETRIEVAL_RECORD_FILE``) in its
#: ``work_dir``: the learning ledger check reads the windows of the degraded retrievals
#: the method injected from it.
MIRIX_RETRIEVAL_RECORD = "mirix_retrievals.jsonl"


def _stamp(value: Any) -> datetime | None:
    """An ISO 8601 time (UTC when it names no zone), None when unreadable."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class DegradedRetrieval:
    """A learning-phase MIRIX retrieval the method injected although it knew part of its
    input was lost (its record's ``degraded`` evidence), and the item's ``injections.jsonl``
    row says ``degraded: true``: its gateway calls ran between ``start`` and ``end``."""

    start: datetime
    end: datetime
    record: Mapping[str, Any]
    position: int | None

    def holds(self, line: Mapping[str, Any]) -> bool:
        """Whether the ledger line's call arrived inside the retrieval's window."""
        stamp = _stamp(line.get("ts"))
        return stamp is not None and self.start <= stamp <= self.end


def degraded_retrievals(
    records: Iterable[Mapping[str, Any]], injections: Iterable[Mapping[str, Any]]
) -> list[DegradedRetrieval]:
    """The learning retrievals MIRIX injected degraded (pure): a ``retrieval`` record of
    the learning phase (not ``frozen``) with its ``degraded`` evidence, no ``failure``
    and a readable ``started_at``..``at`` window, whose task uid has a learning-phase
    ``injections.jsonl`` row flagged ``degraded`` (what ``learning.jsonl`` flags
    ``retrieval_degraded``). Anything else excuses no lost call."""
    flagged: dict[str, int | None] = {}
    for row in injections:
        if row.get("phase") == "learning" and row.get("degraded") is True:
            position = row.get("position")
            flagged[str(row.get("task_uid"))] = (
                position if isinstance(position, int) else None
            )
    found = []
    for record in records:
        if (
            record.get("event") != "retrieval"
            or record.get("frozen") is not False
            or not isinstance(record.get("degraded"), Mapping)
            or record.get("failure")
            or str(record.get("task_uid")) not in flagged
        ):
            continue
        start, end = _stamp(record.get("started_at")), _stamp(record.get("at"))
        if start is None or end is None or end < start:
            continue
        found.append(
            DegradedRetrieval(start, end, record, flagged[str(record["task_uid"])])
        )
    return found


class _CheckpointCorrupt(Exception):
    """A learning checkpoint on disk fails its own hashes."""


@dataclass(frozen=True)
class _Checkpoint:
    """A learning checkpoint on disk, its ``checkpoint.json`` verified."""

    cid: str
    kind: str
    position: int
    ordinal: int
    path: Path
    record: Mapping[str, Any]

    @property
    def offset(self) -> int:
        return int(self.record["ledger_marker"]["offset"])

    @property
    def next_position(self) -> int:
        """The first learning item a resume from here plays from its start."""
        return self.position + 1 if self.kind == CHECKPOINT_DONE else self.position


# ---------------------------------------------------------------------------
# Checkpoint files (blocking helpers, run in threads)
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tree_files(root: Path) -> list[str]:
    """Every regular file under ``root`` (relative POSIX paths, sorted)."""
    if not root.is_dir():
        return []
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            found.append((Path(dirpath) / name).relative_to(root).as_posix())
    return sorted(found)


def _is_object(rel: str) -> bool:
    return rel == OBJECTS_DIR or rel.startswith(OBJECTS_DIR + "/")


def _copy_method_dir(
    src: Path, dest: Path, *, cache: dict[str, tuple[int, int, str]] | None = None
) -> dict[str, str]:
    """Copy a method's ``work_dir`` to ``dest`` (write-once ``objects/**`` hard-linked,
    every other file copied and fsync'd); returns ``{relpath: sha256}``. ``cache`` keeps
    the sha256 of object files by (size, mtime), so they are hashed once."""
    files: dict[str, str] = {}
    dest.mkdir(parents=True, exist_ok=True)
    for rel in _tree_files(src):
        source, target = src / rel, dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if _is_object(rel):
            stat = source.stat()
            known = (cache or {}).get(rel)
            if known is not None and known[:2] == (stat.st_size, stat.st_mtime_ns):
                sha = known[2]
            else:
                sha = _file_sha256(source)
                if cache is not None:
                    cache[rel] = (stat.st_size, stat.st_mtime_ns, sha)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
                _fsync_path(target)
        else:
            shutil.copy2(source, target)
            _fsync_path(target)
            sha = _file_sha256(target)
        files[rel] = sha
    return files


def _restore_method_dir(checkpoint_method: Path, work_dir: Path) -> dict[str, str]:
    """Replace ``work_dir`` by a checkpoint's copy (objects hard-linked back)."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    return _copy_method_dir(checkpoint_method, work_dir)


def _write_json_durable(path: Path, value: Any) -> bytes:
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return data


def _verify_checkpoint(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    """``checkpoint.json`` of a checkpoint directory, after checking its own sha256
    (``LATEST`` records it) and every file it lists; raises :class:`_CheckpointCorrupt`."""
    record_path = path / CHECKPOINT_RECORD
    try:
        data = record_path.read_bytes()
    except OSError as exc:
        raise _CheckpointCorrupt(f"{record_path}: {exc}") from exc
    if (
        expected_sha256 is not None
        and hashlib.sha256(data).hexdigest() != expected_sha256
    ):
        raise _CheckpointCorrupt(f"{record_path} does not hash to the one LATEST names")
    try:
        record = json.loads(data)
    except ValueError as exc:
        raise _CheckpointCorrupt(f"{record_path} is not JSON: {exc}") from exc
    if not isinstance(record, dict) or record.get("id") != path.name:
        raise _CheckpointCorrupt(f"{record_path} does not describe {path.name}")
    for rel, sha in sorted((record.get("files") or {}).items()):
        target = path / rel
        try:
            actual = _file_sha256(target)
        except OSError as exc:
            raise _CheckpointCorrupt(f"{target}: {exc}") from exc
        if actual != sha:
            raise _CheckpointCorrupt(f"{target} does not hash to its recorded sha256")
    return record


def _checkpoint_dirs(root: Path) -> list[tuple[int, str, Path]]:
    """Complete checkpoint directories under ``checkpoints/`` as (ordinal, id, path),
    newest first (``*.partial`` and anything else left out)."""
    found = []
    if not root.is_dir():
        return found
    for child in root.iterdir():
        try:
            kind, position = parse_checkpoint_id(child.name)
        except ValueError:
            continue
        if child.is_dir():
            found.append((checkpoint_ordinal(kind, position), child.name, child))
    return sorted(found, reverse=True)


def _load_checkpoint(path: Path, expected_sha256: str | None = None) -> _Checkpoint:
    record = _verify_checkpoint(path, expected_sha256)
    kind, position = parse_checkpoint_id(path.name)
    return _Checkpoint(
        path.name, kind, position, checkpoint_ordinal(kind, position), path, record
    )


def _truncate_jsonl(
    live: Path,
    size: int,
    moved: Path,
    *,
    expected_sha256: str | None,
    annotate: Callable[[dict[str, Any]], dict[str, Any]],
) -> int:
    """Move the tail of an append-only JSONL file past ``size`` bytes to ``moved`` (each
    row through ``annotate``) and rewrite ``live`` to exactly its first ``size`` bytes.
    Safe to repeat: a file already cut keeps the rows moved before. Returns the rows
    moved; raises :class:`CheckpointMismatch` when the kept prefix is not the one the
    checkpoint recorded."""
    try:
        data = live.read_bytes()
    except FileNotFoundError:
        data = b""
    if len(data) < size:
        raise CheckpointMismatch(
            f"{live.name} holds {len(data)} bytes, fewer than the {size} its checkpoint "
            "recorded"
        )
    head, tail = data[:size], data[size:]
    if (
        expected_sha256 is not None
        and hashlib.sha256(head).hexdigest() != expected_sha256
    ):
        raise CheckpointMismatch(
            f"the first {size} bytes of {live.name} are not the ones its checkpoint "
            "recorded"
        )
    rows = 0
    if tail.strip():
        out = []
        for raw in tail.split(b"\n"):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                out.append(json.dumps({"torn_line": raw.decode("utf-8", "replace")}))
                continue
            out.append(json.dumps(annotate(row), ensure_ascii=False, sort_keys=True))
            rows += 1
        moved.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(moved, ("\n".join(out) + "\n").encode("utf-8"))
    if tail:
        _write_bytes_atomic(live, head)
    return rows


def _prefix_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return hashlib.sha256(b"").hexdigest()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _learning_row_trajectory(rows: Sequence[Mapping[str, Any]], position: int) -> str:
    """The trajectory path of the live learning row of ``position``."""
    for row in reversed(rows):
        if row.get("position") == position and row.get("failure") is None:
            path = row.get("trajectory")
            if isinstance(path, str) and path:
                return path
    raise CheckpointMismatch(
        f"no recorded learning row of item {position}: its played checkpoint cannot be "
        "continued"
    )


class _BlobsLost(ValueError):
    """Image blobs a played item's recorded trajectory references are missing or corrupt,
    and the item's benchmark data does not re-create them: its ``played``
    checkpoint cannot be continued as recorded. The message leads with the blob
    directory and the first blob's name, then what the operator does: the failure detail
    keeps 500 characters (the whole list is in the message, which worker.log keeps)."""

    def __init__(
        self, position: int, blob_dir: Path, digests: Sequence[str], why: str | None
    ) -> None:
        self.digests = list(digests)
        more = len(self.digests) - 1
        first = self.digests[0] if self.digests else "?"
        super().__init__(
            f"{len(self.digests)} image blob(s) missing or corrupt in {blob_dir}: "
            f"{first}" + (f" and {more} more" if more > 0 else "") + "; item "
            f"{position}'s data does not re-create them. Copy them back (a mirror or "
            "backup of the runs directory) and --force-resume, or --new-attempt. "
            f"All: {', '.join(self.digests)}"
            + (f"; re-creating them failed: {why}" if why else "")
        )


# ---------------------------------------------------------------------------
# Gateway link
# ---------------------------------------------------------------------------


class GatewayLink:
    """The cell's connection to the gateway: attached (``--gateway-url``) or launched.

    For as long as the link is open, ``run.py`` holds the gateway's users lock shared
    (:func:`~evals.skill_bench.infra.gateway.gateway_users_lock`): no scheduler sweep, no
    ``launch_gateway`` cleanup and no adopting scheduler kills a gateway a live cell still
    uses, whatever became of the harness that launched it.
    """

    def __init__(
        self,
        base_url: str,
        runs_dir: Path,
        *,
        mode: str,
        owned: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.runs_dir = runs_dir
        self.mode = mode
        self.owned = owned
        self.users: OwnerLock | None = None
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {LOCAL_TOKEN}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
            trust_env=False,  # loopback: never through a proxy
        )

    @classmethod
    async def open(cls, options: RunOptions, *, port: int | None = None) -> GatewayLink:
        """Attach to ``options.gateway_url``, or launch a gateway on ``port`` (default
        ``options.gateway_port``, else a free port)."""
        if options.gateway_url:
            return cls(options.gateway_url, options.gateway_runs_dir, mode="attach")
        port = port or options.gateway_port or await asyncio.to_thread(find_free_port)
        kwargs: dict[str, Any] = {}
        if options.balance_floor_usd is not None:
            kwargs["balance_floor_usd"] = options.balance_floor_usd
        if options.budget_usd is not None:
            kwargs["budget_usd"] = options.budget_usd
        owned = await launch_gateway(
            port=port,
            env_file=options.env_file,
            runs_dir=options.gateway_runs_dir,
            **kwargs,
        )
        return cls(owned.base_url, options.gateway_runs_dir, mode="launch", owned=owned)

    def users_lock_path(self, health: Mapping[str, Any]) -> Path:
        """The gateway's users lock: as its ``/health`` names it, else derived from its
        runs directory, host and port (a gateway started without one)."""
        reported = health.get("users_lock")
        if isinstance(reported, str) and reported:
            return Path(reported)
        url = httpx.URL(self.base_url)
        return gateway_users_lock(self.runs_dir, url.host, url.port or 80)

    async def hold_users_lock(self, health: Mapping[str, Any]) -> None:
        """Hold the gateway's users lock (shared) until :meth:`close`."""
        if self.users is None:
            path = self.users_lock_path(health)
            self.users = await asyncio.to_thread(OwnerLock.acquire, path, shared=True)

    async def health(self) -> dict[str, Any]:
        response = await self.client.get("/health")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise RuntimeError(f"gateway /health is not ok: {data!r}"[:500])
        return data

    async def set_phase(self, run_id: str, phase: str) -> None:
        response = await self.client.post(
            f"/runs/{run_id}/phase", json={"phase": phase}
        )
        response.raise_for_status()

    async def abandon(self, run_id: str, reason: str) -> None:
        response = await self.client.post(
            f"/runs/{run_id}/abandon", json={"reason": reason}
        )
        response.raise_for_status()

    async def mark(self, run_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /runs/<run_id>/mark``: the gateway appends a ``checkpoint`` or ``resume``
        marker line to the run's ledger (fsync'd) and answers with its byte ``offset``."""
        response = await self.client.post(f"/runs/{run_id}/mark", json=dict(body))
        if response.status_code in (404, 405):
            raise RuntimeError(
                f"gateway {self.base_url} has no POST /runs/<run_id>/mark (learning "
                "checkpoints need a gateway of this checkout)"
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise RuntimeError(f"gateway mark answered {data!r}"[:500])
        return data

    async def supports_marks(self) -> bool:
        """Whether the gateway has the ``mark`` route (a bad event is refused with 400)."""
        response = await self.client.post(
            "/runs/skill-bench-mark-probe/mark", json={"event": "probe"}
        )
        return response.status_code not in (404, 405)

    def ledger_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "gateway.jsonl"

    async def close(self) -> None:
        """Release the users lock; stop a gateway this cell launched unless another live
        process uses it (then it only gives up ownership, and whoever cleans up after the
        last user stops it)."""
        try:
            await self.client.aclose()
        finally:
            users, self.users = self.users, None
            path = users.path if users is not None else None
            if users is not None:
                users.release()
            if self.owned is not None:
                shared = path is not None and await asyncio.to_thread(
                    lock_is_held, path
                )
                if shared:
                    owner = getattr(self.owned, "owner", None)
                    if owner is not None:
                        owner.release()
                else:
                    await self.owned.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def new_instance_id() -> str:
    """A cell directory's random instance id: part of every gateway run id of its attempts
    (``<cell_id>-aNN-<instance>``), so a cell started over in a new directory, or a second
    runs directory on the same gateway, never writes into another cell's ledger."""
    return uuid.uuid4().hex[:8]


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def same_gateway(a: str, b: str) -> bool:
    """True when two gateway base URLs reach the same gateway: same scheme and port, and
    the same host or both loopback (``localhost`` / ``127.0.0.1``)."""
    try:
        x, y = httpx.URL(a), httpx.URL(b)
    except Exception:
        return False
    if x.scheme != y.scheme or x.port != y.port:
        return False
    return x.host == y.host or (
        x.host.lower() in _LOOPBACK_HOSTS and y.host.lower() in _LOOPBACK_HOSTS
    )


def load_or_create_salt(path: Path) -> bytes:
    """The cell's secret uid salt (H5), created once with mode 0600 (blocking)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        salt = new_cell_salt()
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(salt)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        return salt
    if len(data) < 16:
        raise ValueError(f"{path} does not hold a usable salt")
    return data


def snapshot_record(snapshot: Any) -> dict[str, Any]:
    """The snapshot as plain JSON with its sha256 (canonical JSON; refuses non-JSON values)."""
    plain = to_jsonable(snapshot)
    if not isinstance(plain, dict):
        raise TypeError(
            f"snapshot() must return a JSON object, got {type(snapshot).__name__}"
        )
    json.loads(canonical_json(plain))  # raises on values JSON cannot represent
    return {"sha256": canonical_sha256(plain), "snapshot": plain}


#: Failure codes of an environment preflight that are transient: the cell is
#: ``infra_unavailable`` (resumed with backoff), never ``refused`` (a person). The
#: WebShop server unreachable or still loading (its readiness gate is read every 30 s, and
#: a restart can land between that reading and the cell's start).
TRANSIENT_PREFLIGHT_CODES: frozenset[str] = frozenset({WEBSHOP_SERVER})


async def preflight_env(
    env: Env, benchmark: str, splits: Sequence[str]
) -> dict[str, Any]:
    """The environment's own checks before a cell starts.
    A failure is :class:`_Refused`, except one of :data:`TRANSIENT_PREFLIGHT_CODES`
    (:class:`_InfraUnavailable`)."""
    info: dict[str, Any] = {"splits": list(splits)}
    verify = getattr(env, "verify_gamefiles", None)
    if callable(verify):
        problems: list[str] = []
        for split in dict.fromkeys(splits):
            found = cast(Sequence[str], await asyncio.to_thread(verify, split))
            problems += [str(p) for p in found]
        if problems:
            shown = "; ".join(problems[:10])
            more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
            raise _Refused(
                f"{benchmark} game files are not the pinned ones: {shown}{more}"
            )
        info["gamefiles_verified"] = True
    check = getattr(env, "preflight", None)
    if callable(check):
        try:
            result = await check()
        except Exception as exc:
            text = f"{benchmark} preflight failed: {type(exc).__name__}: {exc}"
            if classify_exception(exc).code in TRANSIENT_PREFLIGHT_CODES:
                raise _InfraUnavailable(text) from exc
            raise _Refused(text)
        info["env"] = to_jsonable(result)
    return info


async def _pg_connect(**kwargs: Any) -> Any:
    import asyncpg

    return await asyncpg.connect(**kwargs)


async def mirix_preflight(
    cell_id: str,
    cell_dir: Path,
    *,
    pg: PostgresAdmin | None = None,
    connect: Callable[..., Awaitable[Any]] = _pg_connect,
    probe: Callable[[Mapping[str, str]], Awaitable[Mapping[str, Any]]] = (
        probe_llm_contract
    ),
) -> dict[str, Any]:
    """What a MIRIX cell needs before an attempt is started (H7.2, H8), so a missing
    database or a MIRIX that would retry on its own refuses the cell instead of consuming
    learning attempts: no ``.env`` visible to MIRIX, no ``MIRIX_API_KEY`` in the
    harness's environment, the compose Postgres and its template reachable, and this
    checkout's MIRIX reading the retry/timeout settings."""
    leaks = await asyncio.to_thread(dotenv_files_visible_to_mirix)
    if leaks:
        raise _Refused(
            f"MIRIX would load {leaks[0]} via load_dotenv(); move it out of the way"
        )
    if os.environ.get("MIRIX_API_KEY"):
        # The method refuses it at setup (methods/mirix.MirixApi): MirixClient would
        # authenticate as that key's client instead of the cell's.
        raise _Refused(
            "MIRIX_API_KEY is set in the harness's environment: every MIRIX cell would "
            "fail at setup (MirixClient would authenticate as that key's client); unset it"
        )
    pg = pg or PostgresAdmin()
    try:
        conn = await connect(
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=pg.maintenance_db,
            timeout=10,
        )
        try:
            template = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", pg.template
            )
        finally:
            await conn.close()
    except Exception as exc:
        # Transient (a Postgres restart) or not yet started: nothing ran, run again.
        raise _InfraUnavailable(
            f"MIRIX cells need the compose Postgres on {pg.host}:{pg.port} "
            f"(docker compose -f evals/skill_bench/infra/compose.yaml up -d): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not template:
        raise _Refused(f"the Postgres template database {pg.template} is missing")
    env = build_server_env(CellServerSpec(cell_id, 1, cell_dir, port=0, pg=pg))
    try:
        values = dict(await probe(env))
    except Exception as exc:
        raise _Refused(f"the MIRIX settings probe failed: {exc}") from exc
    problems = check_llm_contract(values)
    if problems:
        raise _Refused(
            "this MIRIX does not honour the retry/timeout contract (H3): "
            + "; ".join(problems)
        )
    return {
        "postgres": f"{pg.host}:{pg.port}",
        "template": pg.template,
        "llm_contract": to_jsonable(values),
    }


async def _run_all(workers: Sequence[Callable[[], Any]]) -> None:
    """Run coroutine factories concurrently; the first error cancels the others."""
    tasks = [asyncio.ensure_future(w()) for w in workers]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            error = None if task.cancelled() else task.exception()
            if error is not None:
                raise error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


@dataclass
class _Attempt:
    """Everything one attempt's phases share within this invocation."""

    number: int
    paths: AttemptPaths
    run_id: str
    method: Any
    actor: Actor
    server: MirixServer | None
    learning_rows: JsonlWriter
    result_rows: JsonlWriter
    injections: JsonlWriter
    failures: FailureLog
    counter: dict[str, int] = field(default_factory=dict)
    #: MIRIX's cell database (learning checkpoints); None for the other methods.
    store: CheckpointStore | None = None
    #: MIRIX's HOME (learning checkpoints); None for the other methods.
    home: Path | None = None
    #: The last checkpoint's HOME manifest: unchanged files are not hashed again.
    home_manifest: dict[str, Any] | None = None
    #: sha256 of the method's write-once objects, by (size, mtime).
    object_hashes: dict[str, tuple[int, int, str]] = field(default_factory=dict)
    #: The calls earlier learning resumes of this attempt discarded.
    discarded: frozenset[str] = frozenset()
    #: The lost MIRIX calls this attempt accepted after their batch was rolled back
    #: ``MIRIX_BATCH_ROLLBACKS`` times (``attempt.json`` ``accepted_learning_calls``) or
    #: as calls of a degraded retrieval the method injected (``degraded_retrieval_calls``):
    #: left out of the ledger check like the discarded ones.
    accepted: frozenset[str] = frozenset()
    #: How many learning resumes in a row, this one included, restored
    #: ``rollback_checkpoint`` after an INFRA ``mirix_learning`` failure
    #: (``cell_records.consecutive_rollbacks``); set by a learning resume.
    rollbacks: int = 0
    rollback_checkpoint: str | None = None
    #: A learning resume opened this attempt and its first ledger check has not run
    #: yet: that check sees the replayed batch (and may accept it).
    after_resume: bool = False


class CellRunner:
    """Runs one step of one cell's plan (see the module docstring); :meth:`run` returns the outcome.

    ``env_factory``, ``method_factory``, ``mirix_server_factory`` and ``git`` are seams for
    tests; the defaults are the real environments, :func:`methods.base.method_class`, the
    real MIRIX server and :func:`cell_records.git_info`.
    """

    def __init__(
        self,
        spec: CellSpec,
        options: RunOptions,
        *,
        env_factory: EnvFactory | None = None,
        method_factory: MethodFactory | None = None,
        mirix_server_factory: MirixServerFactory | None = None,
        mirix_check: MirixPreflight | None = None,
        git: GitInfo | None = None,
    ) -> None:
        self.spec = spec
        self.options = options
        self.paths = CellPaths(options.out_dir)
        self.env_factory = env_factory or default_env
        self.method_factory = method_factory
        self.mirix_server_factory = mirix_server_factory or default_mirix_server
        self.mirix_check = mirix_check or mirix_preflight
        self.git = git or git_info
        #: How a dirty tree's diff is captured (a seam: ``tests/run_support.runner``
        #: replaces it); None: :func:`capture_tree`, looked up when called.
        self.capture_tree: TreeCapture | None = None
        #: Which hashed files the commit plus the tree diff cannot rebuild:
        #: :func:`unreproducible_files` on the real checkout. A runner given its own
        #: ``git`` record (the tests) has no checkout to check: None, recorded as not
        #: checked (``tree.unreproducible`` None).
        self.tree_check: TreeCheck | None = (
            unreproducible_files if git is None else None
        )
        self.launch = uuid.uuid4().hex[:12]
        self.env: Env | None = None
        self.learning: list[Task] = []
        self.test: list[Task] = []
        self.salt = b""
        self.gateway: GatewayLink | None = None
        self.manifest: dict[str, Any] = {}
        self.git_now: dict[str, Any] = {}
        self.code_now: dict[str, Any] = {}
        #: This tree's protocol (``fingerprint``, ``constants_sha256``), set with the
        #: manifest: what a new attempt and a resume run under.
        self.protocol_now: dict[str, Any] = {}
        #: This tree's whole ``protocol.fingerprint_manifest`` (per-key constants and
        #: per-file hashes): what the automatic freeze freezes and a version segment's
        #: ``version.json`` keeps.
        self.protocol_manifest: dict[str, Any] = {}
        #: ``protocol.code_fingerprint(files=True)``: the digests with their per-file
        #: hashes (``version.json``); ``code_now`` is the same without the maps.
        self.code_files: dict[str, Any] = {}
        #: The gateway URL the method's own calls use (a resumed MIRIX agent keeps the one
        #: it was configured with); None: the gateway link's.
        self._method_gateway_url: str | None = None
        self._status_extra: dict[str, Any] = {}
        self._learning_ledger = _LedgerReader()
        self.host = host_info()
        #: The signal (``SIGTERM`` / ``SIGINT``) that cancelled this invocation, recorded
        #: in its ``interrupted`` status: the scheduler counts an interruption nobody
        #: signalled (a leaked cancellation) as a failure.
        self.received_signal: str | None = None

    # -- entry point -------------------------------------------------------------

    async def run(self) -> CellOutcome:
        merged = self.paths.root.parent / MERGED_MARKER
        if await asyncio.to_thread(merged.exists):
            # A merged read-only view of several hosts (tools/sync_runs.py): its cells
            # belong to the hosts that ran them; nothing here is ever written.
            return CellOutcome(
                self.spec.cell_id,
                STATUS_REFUSED,
                detail=f"{self.paths.root.parent} is a merged read-only view of several "
                f"hosts ({MERGED_MARKER}); run the cell in its host's runs directory",
            )
        self.paths.root.mkdir(parents=True, exist_ok=True)
        try:
            lock = await asyncio.to_thread(OwnerLock.acquire, self.paths.lock)
        except OwnedProcessError:
            # Another live harness owns the cell; its status file is not ours to touch.
            return CellOutcome(
                self.spec.cell_id,
                STATUS_BUSY,
                detail="another live harness process holds the cell lock",
            )
        try:
            return await self._run_locked()
        finally:
            lock.release()

    async def _run_locked(self) -> CellOutcome:
        previous = await asyncio.to_thread(read_json, self.paths.status)
        await self._write_process()
        await self._set_status(STATUS_RUNNING, phase="prepare")
        try:
            outcome = await self._prepare_and_run()
        except _Refused as exc:
            outcome = CellOutcome(self.spec.cell_id, STATUS_REFUSED, detail=str(exc))
        except _Paused as exc:
            outcome = CellOutcome(
                self.spec.cell_id, STATUS_BUDGET_PAUSED, detail=str(exc)
            )
        except _InfraUnavailable as exc:
            outcome = CellOutcome(
                self.spec.cell_id, STATUS_INFRA_UNAVAILABLE, detail=str(exc)
            )
        except asyncio.CancelledError:
            # Blocking write on purpose: a second cancellation must not lose the status.
            self._write_status_now(
                STATUS_INTERRUPTED,
                detail="interrupted"
                + (f" ({self.received_signal})" if self.received_signal else ""),
                signal=self.received_signal,
            )
            raise
        except Exception as exc:
            log.exception("cell %s crashed", self.spec.cell_id)
            await self._set_status(
                STATUS_CRASHED, detail=f"{type(exc).__name__}: {exc}"[:1000]
            )
            raise
        if (
            outcome.status
            in (STATUS_REFUSED, STATUS_BUDGET_PAUSED, STATUS_INFRA_UNAVAILABLE)
            and isinstance(previous, dict)
            and previous.get("status") == STATUS_COMPLETE
        ):
            # Nothing ran: a refusal must not turn a finished cell into a blocked one.
            await asyncio.to_thread(
                write_json_atomic,
                self.paths.status,
                {
                    **previous,
                    "last_refusal": {
                        "status": outcome.status,
                        "detail": outcome.detail,
                        "launch": self.launch,
                        "at": now(),
                    },
                },
            )
            return outcome
        await self._set_status(
            outcome.status,
            attempt=outcome.attempt,
            phase="exited",
            detail=outcome.detail,
            pending=outcome.pending,
            # The scheduler's --rerun-interrupted leaves a HARNESS learning failure for a
            # person: it reads the kind here.
            failure_kind=outcome.failure_kind,
        )
        return outcome

    # -- status and identity -------------------------------------------------------

    def _status_record(self, status: str, **extra: Any) -> dict[str, Any]:
        self._status_extra.update({k: v for k, v in extra.items() if v is not None})
        return {
            "cell_id": self.spec.cell_id,
            "status": status,
            "exit_code": EXIT_CODES[status] if status in EXIT_CODES else None,
            "pid": os.getpid(),
            "launch": self.launch,
            "updated_at": now(),
            **self._status_extra,
        }

    async def _set_status(self, status: str, **extra: Any) -> None:
        record = self._status_record(status, **extra)
        await asyncio.to_thread(write_json_atomic, self.paths.status, record)

    def _write_status_now(self, status: str, **extra: Any) -> None:
        write_json_atomic(self.paths.status, self._status_record(status, **extra))

    async def _write_process(self) -> None:
        pid = os.getpid()
        started = await asyncio.to_thread(process_start_time, pid)
        await asyncio.to_thread(
            write_json_atomic,
            self.paths.process,
            {
                "pid": pid,
                "started": started,
                "argv": list(sys.argv),
                "cell_id": self.spec.cell_id,
                "launch": self.launch,
                "at": now(),
            },
        )

    # -- preparation ---------------------------------------------------------------

    async def _prepare_and_run(self) -> CellOutcome:
        spec = self.spec
        try:
            env = await asyncio.to_thread(self.env_factory, spec.benchmark)
            self.learning, self.test = await asyncio.to_thread(select_items, env, spec)
        except Exception as exc:
            raise _Refused(
                f"cannot load the {spec.benchmark} items: {type(exc).__name__}: {exc}"
            ) from exc
        self.env = env
        factory = await self._method_factory()
        fingerprint = await asyncio.to_thread(protocol.fingerprint_manifest)
        self.protocol_manifest = dict(fingerprint)
        self.git_now = dict(await asyncio.to_thread(self.git))
        full = dict(
            await asyncio.to_thread(
                protocol.code_fingerprint, mirix=spec.method == MIRIX_METHOD, files=True
            )
        )
        self.code_files = full
        # attempt.json and the work lines keep the digests alone, as before.
        self.code_now = {k: v for k, v in full.items() if not k.endswith("_sha256")}
        if self.options.allow_code_change:
            log.info(
                "%s: --allow-code-change is no longer needed: a resume under another "
                "commit, code or protocol runs and is recorded in attempt.json "
                "version_segments",
                spec.cell_id,
            )
        await self._load_manifest(fingerprint)
        if not await asyncio.to_thread(self.paths.salt.is_file) and (
            await asyncio.to_thread(self.paths.attempt_numbers)
        ):
            raise _Refused(
                f"{self.paths.salt} is gone although the cell has attempts: its task "
                "uids cannot be reproduced (H5)"
            )
        self.salt = await asyncio.to_thread(load_or_create_salt, self.paths.salt)
        # Every invocation leaves a start line. The code it runs is recorded only once it
        # goes on to run an attempt's work (``_record_work``): an invocation that finds the
        # cell complete, is refused or paused, or finds the gateway down ran nothing, and
        # must not count as a code change of the attempt.
        await asyncio.to_thread(
            _append,
            self.paths.launches,
            {
                "event": LAUNCH_START,
                "launch": self.launch,
                "at": now(),
                "pid": os.getpid(),
                "host": self.host,
                "git": self.git_now,
                "options": self.options.to_dict(),
            },
        )
        await self._acknowledge_rerun()
        preflight: dict[str, Any] = {"skipped": True}
        if self.options.preflight:
            splits = (["train"] if self.learning else []) + [spec.test_split]
            preflight = await preflight_env(env, spec.benchmark, splits)
            if spec.method == MIRIX_METHOD:
                preflight["mirix"] = dict(
                    await self.mirix_check(spec.cell_id, self.paths.root)
                )
        self.manifest["preflight"] = preflight
        self.gateway = await self._open_gateway()
        try:
            return await self._plan_and_run(factory)
        finally:
            await self.gateway.close()

    async def _method_factory(self) -> MethodFactory:
        if self.method_factory is not None:
            return self.method_factory
        try:
            cls = await asyncio.to_thread(method_class, self.spec.method)
        except Exception as exc:
            raise _Refused(f"method {self.spec.method}: {exc}") from exc
        if cls is None:
            raise _Refused(
                f"method {self.spec.method!r} is not implemented yet "
                f"(methods/{self.spec.method}.py is a placeholder)"
            )
        return cls

    def _immutable(self, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
        spec = self.spec
        learning = [t.raw_id for t in self.learning]
        test = [t.raw_id for t in self.test]
        return {
            **spec.to_dict(),
            "protocol_id": protocol.PROTOCOL_ID,
            "seed": protocol.SEED,
            "learning_split": "train" if learning else None,
            "test_split": spec.test_split,
            "learning_items": learning,
            "test_items": test,
            "learning_items_sha256": sha256_lines(learning),
            "test_items_sha256": sha256_lines(test),
            "protocol_fingerprint": fingerprint["fingerprint"],
            "protocol_constants_sha256": fingerprint["constants_sha256"],
        }

    async def _load_manifest(self, fingerprint: Mapping[str, Any]) -> None:
        """Read ``cell.json`` (or start it) and check that it describes this cell.

        Every key of the definition but the protocol's (:data:`PROTOCOL_KEYS`) must equal
        this invocation's: the items, splits, model and method never change in a cell
        directory. A protocol fingerprint that differs is not refused: the manifest
        keeps the one its latest attempt started under, a new attempt adopts this tree's
        (``_adopt_protocol``) and a resume goes on under it, recorded as a new version
        segment (``_version_segment``).
        ``self.protocol_now`` holds this tree's."""
        immutable = self._immutable(fingerprint)
        self.protocol_now = {
            "fingerprint": immutable["protocol_fingerprint"],
            "constants_sha256": immutable["protocol_constants_sha256"],
        }
        existing = await asyncio.to_thread(read_json, self.paths.manifest)
        if existing is None:
            self.manifest = {
                "schema": 1,
                "cell": immutable,
                "instance": new_instance_id(),
                "created_at": now(),
                "git_at_creation": self.git_now,
            }
            return
        before = existing.get("cell")
        before = before if isinstance(before, dict) else {}
        changed = sorted(
            k
            for k in set(before) | set(immutable)
            if k not in PROTOCOL_KEYS and before.get(k) != immutable.get(k)
        )
        if changed:
            raise _Refused(
                f"{self.paths.manifest} describes another cell (changed: "
                f"{', '.join(changed)}): a cell directory holds one cell definition. "
                "(A protocol change alone is not refused: a new attempt adopts it, and "
                "the audit and the report show each attempt's version.) "
                + start_over_hint(self.spec.smoke)
            )
        self.manifest = dict(existing)
        # A cell directory made before instance ids keeps its attempts' run ids (they
        # are read from attempt.json); its new attempts get an instance id.
        if not self.manifest.get("instance"):
            self.manifest["instance"] = new_instance_id()

    def _own_run_ids(self) -> set[str]:
        """The gateway runs this cell directory's attempts ran (and reserved)."""
        own: set[str] = set()
        if not self.paths.root.is_dir():
            return own
        for child in self.paths.root.iterdir():
            if not (child.name.startswith("attempt-") and child.name[8:].isdigit()):
                continue
            for name in ("attempt.json", "reservation.json"):
                try:
                    record = read_json(child / name)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("run_id"):
                    own.add(str(record["run_id"]))
        return own

    def _scored_attempts(self) -> list[int]:
        """This directory's attempts with a scored test row (blocking)."""
        return [
            number
            for number in self.paths.attempt_numbers()
            if any(
                r.get("score") is not None
                for r in read_jsonl(self.paths.attempt(number).results)
            )
        ]

    async def _acknowledge_rerun(self) -> None:
        """Record the operator's ``--acknowledge-rerun`` note.

        A cell started over in a new directory after its earlier directory ran test items
        runs the cell twice (H1.3): the audit finds the earlier directory's gateway runs
        (same cell id, another instance id) and reports a violation, unless this cell's
        ``cell.json`` records an operator note covering them. The note is copied into
        ``operator_notes/`` and recorded with its sha256 and the earlier runs as they are
        now; a run that goes on to run more test items later is no longer covered.

        Recorded, not refused: a note that names no
        earlier run in this gateway runs directory (an earlier run on another host, or
        none) is recorded with ``superseded: []``; a protocol cell's note that comes once
        this directory has run test items itself is recorded with ``after_test: true`` and
        ``tested_runs`` (with both runs' scores in view a note could pick the better run:
        the audit reports it, ``rerun_acknowledged_after_test``). A smoke cell's re-run is
        only a warning in the audit. An unreadable, empty or oversized note is still
        refused: an input error, caught before anything runs."""
        note = self.options.acknowledge_rerun
        if note is None:
            return
        try:
            data = await asyncio.to_thread(note.read_bytes)
        except OSError as exc:
            raise _Refused(
                f"cannot read the --acknowledge-rerun note {note}: {exc}"
            ) from exc
        if not data.strip():
            raise _Refused(
                f"the --acknowledge-rerun note {note} is empty: say who re-runs the "
                "cell, when and why"
            )
        if len(data) > MAX_OPERATOR_NOTE_BYTES:
            raise _Refused(
                f"the --acknowledge-rerun note {note} is {len(data)} bytes, more than "
                f"{MAX_OPERATOR_NOTE_BYTES}: a note, not a log"
            )
        instance = self.manifest["instance"]
        own = await asyncio.to_thread(self._own_run_ids)
        runs = await asyncio.to_thread(
            cell_gateway_runs, self.options.gateway_runs_dir, self.spec.cell_id
        )
        earlier = [r for r in runs if r.run_id not in own and r.instance != instance]
        digest = hashlib.sha256(data).hexdigest()
        superseded = [r.to_dict() for r in earlier]
        notes = [
            n for n in self.manifest.get("operator_notes") or () if isinstance(n, dict)
        ]
        if any(
            n.get("kind") == NOTE_RERUN_ACKNOWLEDGED
            and n.get("sha256") == digest
            and n.get("superseded") == superseded
            for n in notes
        ):
            return  # the same note over the same runs: already recorded
        if not earlier:
            log.warning(
                "%s: --acknowledge-rerun: no gateway run of %s from another cell "
                "directory in %s: the note is recorded without earlier runs (an earlier "
                "run on another host, or none)",
                self.spec.cell_id,
                self.spec.cell_id,
                self.options.gateway_runs_dir,
            )
        where: list[str] = []
        if not self.spec.smoke:
            tested = [
                r.run_id
                for r in runs
                if (r.run_id in own or r.instance == instance) and r.tested
            ]
            scored = await asyncio.to_thread(self._scored_attempts)
            where = [f"gateway run {r}" for r in tested] + [
                f"attempt {n}" for n in scored
            ]
            if where:
                log.warning(
                    "%s: --acknowledge-rerun recorded after this directory ran test "
                    "items (%s); the audit reports it (rerun_acknowledged_after_test)",
                    self.spec.cell_id,
                    ", ".join(where),
                )
        dest = self.paths.operator_notes / f"{digest[:12]}-{note.name}"
        await asyncio.to_thread(dest.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_write_bytes_atomic, dest, data)
        notes.append(
            {
                "kind": NOTE_RERUN_ACKNOWLEDGED,
                "file": self._rel(dest),
                "sha256": digest,
                "source": str(note),
                "at": now(),
                "launch": self.launch,
                "superseded": superseded,
                **({"after_test": True, "tested_runs": where} if where else {}),
            }
        )
        self.manifest["operator_notes"] = notes
        await asyncio.to_thread(write_json_atomic, self.paths.manifest, self.manifest)
        log.warning(
            "%s: operator note %s acknowledges %d earlier gateway run(s) of this cell "
            "(%d of them ran test items)",
            self.spec.cell_id,
            dest,
            len(earlier),
            sum(1 for r in earlier if r.tested),
        )

    async def _open_gateway(self) -> GatewayLink:
        options = self.options
        try:
            gateway = await GatewayLink.open(
                options, port=await asyncio.to_thread(self._resume_gateway_port)
            )
        except Exception as exc:
            raise _InfraUnavailable(
                f"cannot start the gateway: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            health = await gateway.health()
            await gateway.hold_users_lock(health)
        except Exception as exc:
            await gateway.close()
            raise _InfraUnavailable(
                f"gateway {gateway.base_url} is not healthy: {type(exc).__name__}: {exc}"
            ) from exc
        conformant = bool(health.get("protocol_conformant")) and (
            health.get("encoding") == protocol.TOKEN_ENCODING
        )
        # The gateway conforms to the protocol it imported when it started; a live one
        # (kept or adopted by the scheduler) may predate this tree's. Its
        # model settings are held to this tree's before any call: the audit fails every
        # call sent under others (H10.6:gateway_start_settings); the rest are recorded.
        manifest = self.protocol_manifest or {}
        constants = manifest.get("constants") or protocol.protocol_constants()
        served = health.get("settings")
        differ = protocol.gateway_settings_differ(served, constants)
        model_differ = [k for k in differ if k in protocol.GATEWAY_MODEL_SETTINGS]
        conformant = conformant and not model_differ
        self.manifest["gateway"] = {
            "base_url": gateway.base_url,
            "mode": gateway.mode,
            "runs_dir": str(gateway.runs_dir),
            "settings": health.get("settings"),
            "settings_sha256": health.get("settings_sha256"),
            "protocol_conformant": conformant,
            "encoding": health.get("encoding"),
            "effort_supported": health.get("effort_supported"),
            # The settings whose value is not the one this tree's protocol gives.
            "tree_settings_differ": differ,
            "checked_at": now(),
        }
        if model_differ and not options.allow_nonconformant_gateway:
            await gateway.close()
            served_map = served if isinstance(served, Mapping) else {}
            tree = protocol.gateway_settings_of(constants)
            raise _Refused(
                f"gateway {gateway.base_url} runs "
                + ", ".join(f"{k}={served_map.get(k)!r}" for k in model_differ)
                + f"; this tree's protocol {str(manifest.get('fingerprint'))[:12]} says "
                + ", ".join(f"{k}={tree.get(k)!r}" for k in model_differ)
                + ": every call it sent would fail the audit (H10.6:"
                "gateway_start_settings). The gateway started before the protocol "
                "changed: restart it from this checkout (stop it; the scheduler, or "
                "run.py without --gateway-url, starts one under this tree), or put the "
                "tree back; H10.6"
            )
        if not conformant and not options.allow_nonconformant_gateway:
            await gateway.close()
            raise _Refused(
                f"gateway {gateway.base_url} does not run the protocol settings "
                f"(protocol_conformant={health.get('protocol_conformant')}, encoding="
                f"{health.get('encoding')}); H3 / H10.6"
            )
        if differ:
            log.warning(
                "%s: gateway %s runs other settings than this tree's protocol (%s); "
                "recorded (manifest gateway.tree_settings_differ), the audit warns "
                "(H10.6-gateway-settings)",
                self.spec.cell_id,
                gateway.base_url,
                ", ".join(differ),
            )
        if health.get("paused"):
            await gateway.close()
            raise _Paused(
                f"gateway paused ({health.get('pause_reason')}); no cell starts until "
                "the balance recovers"
            )
        if self.learning:
            # Learning checkpoints are marked in the ledger: a gateway of an older
            # checkout would fail the first one, after setup had already spent.
            try:
                marks = await gateway.supports_marks()
            except Exception as exc:
                await gateway.close()
                raise _InfraUnavailable(
                    f"gateway {gateway.base_url} did not answer: {type(exc).__name__}: "
                    f"{exc}"
                ) from exc
            if not marks:
                await gateway.close()
                raise _Refused(
                    f"gateway {gateway.base_url} has no POST /runs/<run_id>/mark: learning "
                    "checkpoints need a gateway of this checkout (restart it)"
                )
        return gateway

    # -- the plan --------------------------------------------------------------------

    def _read_attempt(self, number: int) -> dict[str, Any]:
        state = read_json(self.paths.attempt(number).state)
        if not isinstance(state, dict):
            raise _Refused(f"{self.paths.attempt(number).state} is not a JSON object")
        return state

    def _next_attempt_number(self) -> int:
        """One past every ``attempt-NN`` directory, with or without an ``attempt.json``: a
        number reserved by an invocation that stopped before its attempt existed is never
        handed out again, so its gateway run id is never reused."""
        taken = [0]
        if self.paths.root.is_dir():
            for child in self.paths.root.iterdir():
                if child.name.startswith("attempt-") and child.name[8:].isdigit():
                    taken.append(int(child.name[8:]))
        return max(taken) + 1

    async def _plan_and_run(self, factory: MethodFactory) -> CellOutcome:
        """One step of the cell's plan. With ``max_learning_attempts`` > 1, a learning phase
        that stopped on infrastructure is resumed at once from its checkpoint (an attempt
        without checkpoints restarts as a new attempt), up to that many learning steps.
        Never after a HARNESS failure (the harness's or MIRIX's own): it repeats, so a
        person looks first."""
        left = self.options.max_learning_attempts
        while True:
            outcome = await self._plan_step(factory)
            if self.options.new_attempt:
                # --new-attempt is for this invocation's first step: a later step
                # resumes the attempt it started, never invalidates it for another.
                self.options = replace(self.options, new_attempt=False)
            if outcome.status not in (
                STATUS_LEARNING_FAILED,
                STATUS_LEARNING_INTERRUPTED,
            ):
                return outcome
            if outcome.failure_kind == FailureKind.HARNESS.value:
                if left > 1:
                    log.warning(
                        "%s: %s; not resumed although %d more learning step(s) are "
                        "allowed: a HARNESS failure repeats",
                        self.spec.cell_id,
                        outcome.detail,
                        left - 1,
                    )
                return outcome
            left -= 1
            if left <= 0:
                return outcome
            log.warning(
                "%s: %s; %s at once (%d more allowed)",
                self.spec.cell_id,
                outcome.detail,
                "resuming the learning phase"
                if outcome.status == STATUS_LEARNING_INTERRUPTED
                else "starting a new attempt",
                left,
            )

    async def _plan_step(self, factory: MethodFactory) -> CellOutcome:
        numbers = await asyncio.to_thread(self.paths.attempt_numbers)
        if not numbers:
            return await self._new_attempt(factory, why="first attempt")
        number = numbers[-1]
        state = await asyncio.to_thread(self._read_attempt, number)
        current = state.get("state")
        await self._link_superseded(number, state)
        self._repeated_request(number, state)
        if current == ATTEMPT_COMPLETE:
            if self.options.new_attempt:
                scored, calls = await self._tested(number)
                return await self._supersede(
                    number, state, factory, scored=scored, calls=calls
                )
            # A crash after the attempt was marked complete (an older run.py marked it
            # before writing them) must not lose the cell's own records.
            await self._ensure_cell_records(number)
            await self._write_manifest(number)
            return CellOutcome(
                self.spec.cell_id, STATUS_COMPLETE, number, "already complete"
            )
        if current in (ATTEMPT_LEARNING, ATTEMPT_LEARNING_INTERRUPTED):
            if self.options.new_attempt:
                await self._invalidate(
                    number,
                    "the operator asked for a new attempt instead of resuming the "
                    "learning phase",
                    kind=INVALIDATION_OPERATOR,
                )
                return await self._new_attempt(
                    factory, why=f"--new-attempt replacing interrupted attempt {number}"
                )
            return await self._resume_learning(number, state, factory)
        if current == ATTEMPT_INVALID:
            if not state.get("abandoned"):
                await self._invalidate(number, str(state.get("invalid_reason") or ""))
            superseded = state.get("superseded")
            unfinished = (
                state.get("invalidation") == INVALIDATION_SUPERSEDED
                and isinstance(superseded, Mapping)
                and superseded.get("by_attempt") is None
            )
            # A crash between superseding an attempt and writing its replacement's
            # attempt.json: the new attempt still names the one it supersedes.
            return await self._new_attempt(
                factory,
                why=f"after invalid attempt {number}",
                supersedes=number if unfinished else None,
            )
        if current in (ATTEMPT_FROZEN, ATTEMPT_TESTING, ATTEMPT_VIOLATION):
            if self.options.new_attempt:
                scored, calls = await self._tested(number)
                if scored or calls:
                    return await self._supersede(
                        number, state, factory, scored=scored, calls=calls
                    )
                after = " after a violation" if current == ATTEMPT_VIOLATION else ""
                await self._invalidate(
                    number,
                    f"the operator asked for a new attempt{after} before any test item "
                    "was scored",
                    kind=INVALIDATION_OPERATOR,
                )
                return await self._new_attempt(
                    factory,
                    why=f"--new-attempt replacing untested attempt {number} ({current})",
                )
            if current == ATTEMPT_VIOLATION:
                raise _Refused(
                    f"attempt {number} violated the frozen-memory invariant "
                    f"({state.get('violation', {}).get('detail', 'see attempt.json')}); "
                    "pass --new-attempt (with --note '<one line: why>' once a test item "
                    "was scored or a test-phase actor call sent) to learn again from "
                    "scratch: the violated attempt stays, marked superseded. "
                    + start_over_hint(self.spec.smoke)
                )
            return await self._resume_test(number, state, factory)
        raise _Refused(f"attempt {number} has unknown state {current!r}")

    async def _link_superseded(self, number: int, state: Mapping[str, Any]) -> None:
        """A crash between a new attempt's attempt.json and the ``superseded.by_attempt``
        of the attempt it replaces: name the replacement now (it exists)."""
        old = state.get("supersedes")
        if not isinstance(old, int) or old == number:
            return
        try:
            replaced = await asyncio.to_thread(self._read_attempt, old)
        except (_Refused, ValueError, OSError):
            return
        superseded = replaced.get("superseded")
        if isinstance(superseded, Mapping) and superseded.get("by_attempt") is None:
            await self._update_attempt(
                old, superseded={**superseded, "by_attempt": number}
            )

    def _repeated_request(self, number: int, state: Mapping[str, Any]) -> None:
        """An identical ``--new-attempt --note`` repeated (a supervisor restart of the same
        argv) goes on with the attempt that request started instead of replacing it again
        (and paying for its learning twice)."""
        note = state.get("note")
        if (
            self.options.new_attempt
            and self.options.note
            and isinstance(note, Mapping)
            and note.get("text") == self.options.note
            and note.get("new_attempt") is True
        ):
            log.warning(
                "%s: attempt %d was started by this same --new-attempt --note request; "
                "it is not replaced again (change the note to replace it)",
                self.spec.cell_id,
                number,
            )
            self.options = replace(self.options, new_attempt=False)

    async def _supersede(
        self,
        number: int,
        state: Mapping[str, Any],
        factory: MethodFactory,
        *,
        scored: int,
        calls: int,
    ) -> CellOutcome:
        """``--new-attempt`` over an attempt that already ran its test phase (H1.3): with
        ``--note`` the old attempt stays in the directory, marked superseded
        (``invalidation: superseded``, its note, what it scored, the digest of the cell
        records it published, the attempt that replaces it), its gateway run is abandoned (a stray call gets 410), and a new attempt
        starts under this tree's protocol. Without ``--note`` it is refused before
        anything changes."""
        note = self.options.note
        current = state.get("state")
        if not note:
            raise _Refused(
                f"attempt {number} has already run its test phase ({scored} scored test "
                f"item(s), {calls} test-phase actor call(s)): --new-attempt replaces it "
                "only with --note '<one line: why>'; the old attempt stays in this "
                "directory, marked superseded, and the report lists both (H1.3). "
                + start_over_hint(self.spec.smoke)
            )
        digest = (
            await asyncio.to_thread(records_digest, self.paths)
            if current == ATTEMPT_COMPLETE
            else None
        )
        await self._invalidate(
            number,
            f"superseded by a new attempt (--new-attempt --note): {note}",
            kind=INVALIDATION_SUPERSEDED,
            extra={
                "superseded": {
                    "note": note,
                    "at": now(),
                    "launch": self.launch,
                    "scored_items": scored,
                    "test_actor_calls": calls,
                    "cell_records_sha256": digest,
                    "by_attempt": None,
                }
            },
        )
        log.warning(
            "%s: attempt %d (%s, %d scored test item(s)) is superseded by a new attempt: "
            "%s",
            self.spec.cell_id,
            number,
            current,
            scored,
            note,
        )
        return await self._new_attempt(
            factory,
            why=f"superseding tested attempt {number} ({current})",
            supersedes=number,
        )

    async def _tested(self, number: int) -> tuple[int, int]:
        """How far the attempt's test phase got: distinct test items with a score, and
        actor calls the gateway recorded in the run's test phase (the audit's
        ``tested_attempt_replaced`` reads the same two)."""
        rows = await asyncio.to_thread(read_jsonl, self.paths.attempt(number).results)
        scored = len({str(r.get("raw_id")) for r in rows if r.get("score") is not None})
        calls = 0
        run_id = (await asyncio.to_thread(self._read_attempt, number)).get("run_id")
        if self.gateway is not None and run_id:
            ledger = await asyncio.to_thread(
                read_jsonl, self.gateway.ledger_path(str(run_id))
            )
            calls = sum(
                1
                for r in ledger
                if r.get("event") == "call"
                and r.get("role") == "actor"
                and line_phase(r) == "test"
            )
        return scored, calls

    async def _invalidate(
        self,
        number: int,
        reason: str,
        failure: Failure | None = None,
        *,
        stage: str | None = None,
        kind: str = INVALIDATION_LEARNING,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Mark an attempt invalid (H8) and abandon its gateway run so nothing of it keeps spending.

        The attempt keeps its trail: the state it had (``previous_state``), a violation it
        recorded, and ``invalidation``: ``learning_failure`` (H8 restarts a failed learning
        phase), ``operator_reroll`` (``--new-attempt`` before its test phase) or
        ``superseded`` (``--new-attempt --note`` after it), for the audit. ``extra``
        is merged into its ``attempt.json`` (the ``superseded`` block)."""
        assert self.gateway is not None
        state = await asyncio.to_thread(self._read_attempt, number)
        if state.get("state") != ATTEMPT_INVALID:
            state["previous_state"] = state.get("state")
            state["invalidation"] = kind
        state.update(
            state=ATTEMPT_INVALID,
            invalid_reason=reason,
            invalidated_at=now(),
            invalidated_by=self.launch,
        )
        if failure is not None:
            state["failure"] = failure.to_dict()
            state["failure_stage"] = stage
        if extra:
            state.update(extra)
        try:
            await self.gateway.abandon(state["run_id"], reason[:500])
            state["abandoned"] = True
            state.pop("abandon_error", None)
        except Exception as exc:
            state["abandoned"] = False
            state["abandon_error"] = f"{type(exc).__name__}: {exc}"[:500]
        await self._save_attempt(number, state)

    async def _save_attempt(self, number: int, state: Mapping[str, Any]) -> None:
        record = {**state, "updated_at": now()}
        await asyncio.to_thread(
            write_json_atomic, self.paths.attempt(number).state, record
        )

    async def _update_attempt(self, number: int, **changes: Any) -> dict[str, Any]:
        state = await asyncio.to_thread(self._read_attempt, number)
        state.update(changes)
        await self._save_attempt(number, state)
        return state

    # -- attempts --------------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        """A path inside the cell, relative to the cell directory (records stay movable)."""
        return path.relative_to(self.paths.root).as_posix()

    def _new_run_id(self, number: int) -> str:
        """The gateway run of a new attempt: ``<cell_id>-aNN-<instance>``."""
        return f"{self.spec.cell_id}-a{number:02d}-{self.manifest['instance']}"

    def _attempt_run_id(self, number: int) -> str:
        """The gateway run of an existing attempt, as its ``attempt.json`` records it."""
        run_id = self._read_attempt(number).get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise _Refused(f"{self.paths.attempt(number).state} names no gateway run")
        return run_id

    def _resume_gateway_port(self) -> int | None:
        """The port a launched gateway must use so a MIRIX attempt can resume.

        A MIRIX agent keeps the gateway URL it was set up with (its ``llm_config`` and
        ``embedding_config`` endpoints): resuming its learning or test phase in launch
        mode reuses the port its attempt recorded. None when nothing needs it
        (blocking)."""
        options = self.options
        if (
            options.gateway_url
            or options.gateway_port
            or self.spec.method != MIRIX_METHOD
        ):
            return None
        numbers = self.paths.attempt_numbers()
        if not numbers:
            return None
        try:
            state = read_json(self.paths.attempt(numbers[-1]).state)
        except ValueError:
            return None
        if not isinstance(state, dict) or state.get("state") not in (
            ATTEMPT_LEARNING,
            ATTEMPT_LEARNING_INTERRUPTED,
            ATTEMPT_FROZEN,
            ATTEMPT_TESTING,
        ):
            return None
        if state.get("state") in (
            ATTEMPT_LEARNING,
            ATTEMPT_LEARNING_INTERRUPTED,
        ) and (
            options.new_attempt
            or not self.paths.attempt(numbers[-1]).checkpoints.is_dir()
        ):
            return None  # a new attempt follows: any port
        recorded = state.get("gateway_base_url")
        try:
            return httpx.URL(str(recorded)).port if recorded else None
        except Exception:
            return None

    async def _check_ledger_link(self, run_id: str) -> None:
        """The run's ledger must be where the manifest says (H9 cost inputs, H10.3)."""
        assert self.gateway is not None
        path = self.gateway.ledger_path(run_id)
        if not await asyncio.to_thread(path.is_file):
            raise _Refused(
                f"the gateway does not write {path}: pass the --gateway-runs-dir of the "
                "gateway at --gateway-url, so costs and calls can be linked to the cell"
            )

    def _unstarted_reservations(self) -> list[tuple[int, dict[str, Any]]]:
        """Attempt numbers reserved by an invocation that stopped before the attempt
        existed (a ``reservation.json`` and no ``attempt.json``) and not settled yet."""
        found: list[tuple[int, dict[str, Any]]] = []
        if not self.paths.root.is_dir():
            return found
        for child in sorted(self.paths.root.iterdir()):
            if not (child.name.startswith("attempt-") and child.name[8:].isdigit()):
                continue
            paths = self.paths.attempt(int(child.name[8:]))
            if paths.state.is_file():
                continue
            try:
                record = read_json(paths.reservation)
            except ValueError:
                continue  # a torn marker is written atomically: never seen in practice
            if isinstance(record, dict) and not record.get("settled_at"):
                found.append((paths.number, record))
        return found

    async def _settle_reservations(self) -> None:
        """Abandon the gateway run of every attempt that never started.

        The invocation that reserved it stopped between the reservation and
        ``attempt.json`` (a SIGTERM, a crash, the gateway refusing the run). Nothing ran
        under that run: no method or actor gets a run id before ``attempt.json`` exists.
        Its number stays taken (``_next_attempt_number``) and the run is abandoned so that
        nothing can ever spend under it; the reservation records how it was settled. An
        abandon the gateway failed is tried again by the next invocation."""
        assert self.gateway is not None
        instance = self.manifest["instance"]
        for number, record in await asyncio.to_thread(self._unstarted_reservations):
            run_id = str(record.get("run_id") or "")
            settled: dict[str, Any] = {"settled_by": self.launch, "abandoned": False}
            ours = record.get("instance") == instance and run_id == (
                f"{self.spec.cell_id}-a{number:02d}-{instance}"
            )
            known = bool(run_id) and await asyncio.to_thread(
                (self.gateway.ledger_path(run_id).parent / "gateway_state.json").exists
            )
            if not ours:
                settled["gateway_run"] = "not this cell directory's run: left alone"
            elif not known:
                settled["gateway_run"] = "never created"
            else:
                try:
                    await self.gateway.abandon(
                        run_id,
                        f"attempt {number} of {self.spec.cell_id} never started",
                    )
                    settled["abandoned"] = True
                except Exception as exc:
                    settled["abandon_error"] = f"{type(exc).__name__}: {exc}"[:500]
            merged = {**record, **settled}
            if "abandon_error" not in settled:
                merged["settled_at"] = now()
                merged.pop("abandon_error", None)  # an earlier try's
            await asyncio.to_thread(
                write_json_atomic, self.paths.attempt(number).reservation, merged
            )
            log.warning(
                "%s: attempt %d was reserved by launch %s but never started; its number "
                "is skipped (gateway run %s: %s)",
                self.spec.cell_id,
                number,
                record.get("launch"),
                run_id,
                "abandoned"
                if settled["abandoned"]
                else settled.get("abandon_error") or settled.get("gateway_run"),
            )

    async def _record_work(
        self,
        number: int,
        phase: str,
        *,
        why: str,
        learning_from: Mapping[str, Any] | None = None,
        test_from: Mapping[str, Any] | None = None,
    ) -> None:
        """This invocation goes on to run attempt ``number`` from ``phase`` (``learning``: a
        new attempt, which goes on to its test phase, or a learning resume; ``test``: a
        resumed test phase): freeze the tree when it is not the frozen value
        (:meth:`_auto_freeze`), record the version it runs under (:meth:`_version_segment`)
        and append its work line: the git commit, the model-visible code, the protocol,
        the version segment, the ``--note`` and the automatic freeze's status. The audit
        compares the code (and the protocol) of these lines only, so an invocation that
        ran nothing never counts as a change of the attempt."""
        freeze = await self._auto_freeze(number, why)
        index = await self._version_segment(
            number,
            phase,
            why=why,
            freeze=freeze,
            learning_from=learning_from,
            test_from=test_from,
        )
        line: dict[str, Any] = {
            "event": LAUNCH_WORK,
            "launch": self.launch,
            "at": now(),
            "attempt": number,
            "phase": phase,
            "host": self.host,
            "git": self.git_now,
            "code": self.code_now,
            # The audit's protocol_changed_within_attempt.
            "protocol": dict(self.protocol_now),
            "segment": index,
            "note": self.options.note,
            "auto_freeze": freeze.get("status"),
        }
        if freeze.get("status") == "error":
            line["auto_freeze_error"] = freeze.get("error")
        await asyncio.to_thread(_append, self.paths.launches, line)

    async def _auto_freeze(self, number: int, why: str) -> dict[str, Any]:
        """Freeze this launch's protocol manifest when the frozen record does not hold it
        yet, once, under the record's lock, naming this cell, attempt and why
        (``tools/freeze_protocol.auto_freeze``). Never raises (a cancellation aside): an
        error (a missing or corrupt record, the lock held too long) is recorded on the work
        line and the version segment, and the attempt runs on."""
        from evals.skill_bench.tools import freeze_protocol

        context = {
            "cell_id": self.spec.cell_id,
            "attempt": number,
            "why": why,
            "launch": self.launch,
            "host": {
                "hostname": self.host.get("hostname"),
                "machine_id": self.host.get("machine_id"),
            },
            "cell_dir": str(self.paths.root),
        }
        git_now = dict(self.git_now)
        try:
            result = dict(
                await asyncio.to_thread(
                    freeze_protocol.auto_freeze,
                    manifest=self.protocol_manifest,
                    context=context,
                    git=lambda: dict(git_now),
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded; the attempt runs on
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
        status = result.get("status")
        if status == "frozen":
            log.warning(
                "%s: froze protocol %s automatically: %s",
                self.spec.cell_id,
                result.get("fingerprint"),
                result.get("reason"),
            )
        if status == "error":
            log.warning(
                "%s: the automatic freeze of protocol %s failed (recorded; the attempt "
                "runs on): %s",
                self.spec.cell_id,
                self.protocol_now.get("fingerprint"),
                result.get("error"),
            )
        elif result.get("warning"):
            log.warning("%s: %s", self.spec.cell_id, result["warning"])
        if result.get("note"):
            # The record git tracks now differs from the commit on this host.
            log.warning("%s: %s", self.spec.cell_id, result["note"])
        return result

    def _tree_pathspecs(self) -> list[str]:
        """What a version segment's tree diff covers (``cell_records.tree_pathspecs``):
        the harness (without its tests, runs, data and the frozen record, which another
        cell's automatic freeze changes) and, for a MIRIX cell, the MIRIX package and the
        launcher of its server."""
        return tree_pathspecs(mirix=self.spec.method == MIRIX_METHOD)

    async def _version_segment(
        self,
        number: int,
        phase: str,
        *,
        why: str,
        freeze: Mapping[str, Any],
        learning_from: Mapping[str, Any] | None,
        test_from: Mapping[str, Any] | None,
    ) -> int:
        """Record the version this launch runs attempt ``number`` under: the same as the
        attempt's last version segment (protocol fingerprint, code digests, git commit
        and uncommitted tree diff all equal) adds this launch to it; otherwise a new
        segment is opened, with its ``versions/SS/version.json`` (and ``tree.diff`` for a
        dirty tree), saying from which learning position or test count it runs and what
        changed. Returns the segment's index (the work line's ``segment``)."""
        paths = self.paths.attempt(number)
        state = await asyncio.to_thread(self._read_attempt, number)
        lines = await asyncio.to_thread(read_jsonl, self.paths.launches)
        segments = await asyncio.to_thread(
            protocol.attempt_versions, self.manifest, state, number, lines
        )
        pending = paths.root / VERSIONS_DIR / f".pending-{self.launch}.diff.tmp"
        # Another launch's pending diff is a crashed one's (the cell lock admits one).
        await asyncio.to_thread(self._drop_pending_diffs, paths)
        if self.git_now.get("dirty") is not False:
            tree = dict(
                await asyncio.to_thread(
                    self.capture_tree or capture_tree,
                    pending,
                    pathspecs=self._tree_pathspecs(),
                )
            )
        else:
            tree = {
                "dirty": False,
                "diff": None,
                "diff_sha256": None,
                "diff_bytes": 0,
                "untracked": 0,
                "untracked_skipped": [],
                "error": None,
            }
        fingerprint = self.protocol_now.get("fingerprint")
        last = segments[-1] if segments else None

        def key(segment: Mapping[str, Any] | None, tree_sha: Any) -> tuple[Any, ...]:
            segment = segment or {}
            proto = segment.get("protocol") or {}
            code = segment.get("code") or {}
            git = segment.get("git") or {}
            return (
                proto.get("fingerprint"),
                code.get("harness"),
                code.get("mirix"),
                git.get("commit"),
                tree_sha,
            )

        here = (
            fingerprint,
            self.code_now.get("harness"),
            self.code_now.get("mirix"),
            self.git_now.get("commit"),
            tree.get("diff_sha256"),
        )
        if (
            last is not None
            and key(last, (last.get("tree") or {}).get("diff_sha256")) == here
        ):
            if self.launch not in last.setdefault("launches", []):
                last["launches"].append(self.launch)
            await asyncio.to_thread(pending.unlink, missing_ok=True)
            index = int(last.get("index", len(segments) - 1))
        else:
            index = len(segments)
            segment = await asyncio.to_thread(
                self._write_version,
                paths,
                index,
                why=why,
                phase=phase,
                freeze=freeze,
                tree=tree,
                pending=pending,
                previous=last,
                learning_from=learning_from,
                test_from=test_from,
            )
            segments.append(segment)
            if index > 0:
                log.warning(
                    "%s: attempt %d goes on under another version from %s: %s; recorded "
                    "in attempt.json version_segments",
                    self.spec.cell_id,
                    number,
                    json.dumps(learning_from or test_from, sort_keys=True),
                    json.dumps(segment["changed"], sort_keys=True, default=str)[:1000],
                )
        changed = [s.get("changed") or {} for s in segments]
        state["version_segments"] = segments
        state["versions_mixed"] = {
            "protocol": any(bool(c.get("protocol")) for c in changed),
            "code": any(bool(c.get("code")) for c in changed),
            "commit": any(bool(c.get("commit") or c.get("tree")) for c in changed),
        }
        await self._save_attempt(number, state)
        return index

    @staticmethod
    def _drop_pending_diffs(paths: AttemptPaths) -> None:
        directory = paths.root / VERSIONS_DIR
        if directory.is_dir():
            for stale in directory.glob(".pending-*.diff.tmp"):
                stale.unlink(missing_ok=True)

    def _write_version(
        self,
        paths: AttemptPaths,
        index: int,
        *,
        why: str,
        phase: str,
        freeze: Mapping[str, Any],
        tree: dict[str, Any],
        pending: Path,
        previous: Mapping[str, Any] | None,
        learning_from: Mapping[str, Any] | None,
        test_from: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """A new version segment's directory and its entry (blocking): the pending tree
        diff moved to ``tree.diff``, ``version.json`` written atomically (an earlier
        half-written directory of this index is overwritten), and what changed since the
        segment before. ``tree.unreproducible`` names the hashed files the commit plus
        the diff cannot rebuild (:func:`unreproducible_files`; None: not checked); a tree
        with any is dirty (the audit's dirty_tree warning)."""
        seg_dir = paths.root / VERSIONS_DIR / f"{index:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        tree = dict(tree)
        tree["unreproducible"] = None
        if self.tree_check is not None:
            hashed = hashed_repo_paths(
                (self.protocol_manifest or {}).get("files"), self.code_files
            )
            check = dict(
                self.tree_check(hashed, tree, pathspecs=self._tree_pathspecs())
            )
            tree["unreproducible"] = check
            if check.get("count"):
                tree["dirty"] = True
                log.warning(
                    "%s: attempt %d runs %d hashed file(s) its commit and tree diff "
                    "cannot rebuild (gitignored, or untracked and not captured: %s); "
                    "recorded in version segment %d tree.unreproducible: git add them "
                    "(or move them out of the ignored paths) to make the version "
                    "reproducible",
                    self.spec.cell_id,
                    paths.number,
                    check["count"],
                    ", ".join(f["path"] for f in (check.get("files") or [])[:5]),
                    index,
                )
        if tree.get("diff"):
            target = seg_dir / TREE_DIFF
            os.replace(pending, target)
            _fsync_path(seg_dir)
            tree["diff"] = self._rel(target)
        else:
            pending.unlink(missing_ok=True)
        manifest = self.protocol_manifest
        fingerprint = self.protocol_now.get("fingerprint")
        try:
            known: dict[str, Any] | None = dict(protocol.known_versions())
        except (protocol.ProtocolDriftError, OSError, ValueError):
            known = None
        version = (known or {}).get(str(fingerprint))
        frozen = {
            "known": None if known is None else version is not None,
            "current": bool(version and version.get("current")),
            "frozen_at": (version or {}).get("frozen_at"),
            "reason": (version or {}).get("reason"),
        }
        document = {
            "schema": 1,
            "cell_id": self.spec.cell_id,
            "attempt": paths.number,
            "segment": index,
            "launch": self.launch,
            "written_at": now(),
            "fingerprint": fingerprint,
            "constants_sha256": manifest.get("constants_sha256"),
            "constants": manifest.get("constants"),
            "files": manifest.get("files"),
            # What each model was requested as under this version's models.json, by its
            # hash (the audit's model_served check reads the cell's own).
            "model_slugs": protocol.model_slugs_of(manifest),
            "code": dict(self.code_files),
            "git": dict(self.git_now),
            "host": dict(self.host),
            "tree": tree,
            "frozen": frozen,
        }
        target = seg_dir / VERSION_FILE
        write_json_atomic(target, document)
        entry: dict[str, Any] = {
            "index": index,
            "legacy": False,
            "why": why,
            "phase": phase,
            "first_launch": self.launch,
            "started_at": document["written_at"],
            "launches": [self.launch],
            "learning_from": dict(learning_from) if learning_from else None,
            "test_from": dict(test_from) if test_from else None,
            "protocol": {
                "fingerprint": fingerprint,
                "constants_sha256": self.protocol_now.get("constants_sha256"),
            },
            "code": dict(self.code_now),
            "git": dict(self.git_now),
            "host": dict(self.host),
            "tree": tree,
            "version_file": self._rel(target),
            "auto_freeze": dict(freeze),
            "known_version": frozen["known"],
            "note": self.options.note,
            "changed": None,
        }
        if previous is not None:
            entry["changed"] = self._version_changes(previous, document, known)
        return entry

    def _version_changes(
        self,
        previous: Mapping[str, Any],
        document: Mapping[str, Any],
        known: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """What a new version segment changes since ``previous`` (blocking: reads the
        previous segment's ``version.json``)."""
        before_doc: Mapping[str, Any] = {}
        rel = previous.get("version_file")
        if isinstance(rel, str) and rel:
            try:
                loaded = read_json(self.paths.root / rel)
            except ValueError:
                loaded = None
            before_doc = loaded if isinstance(loaded, Mapping) else {}
        a = (previous.get("protocol") or {}).get("fingerprint")
        b = document.get("fingerprint")
        diff = None
        if a != b and a and b:
            versions = dict(known or {})
            for fp, doc in ((a, before_doc), (b, document)):
                recorded = protocol.recorded_version(doc) if doc else None
                if recorded is not None and recorded["fingerprint"] == fp:
                    versions[str(fp)] = recorded
            diff = protocol.version_diff(a, b, versions)
        before_code = previous.get("code") or {}
        code = [
            name
            for name in ("harness", "mirix")
            if before_code.get(name) != self.code_now.get(name)
        ]
        code_files: dict[str, Any] | None = None
        old_maps = before_doc.get("code") if isinstance(before_doc, Mapping) else None
        if isinstance(old_maps, Mapping) and isinstance(
            old_maps.get("harness_sha256"), Mapping
        ):
            code_files = {}
            for name in ("harness", "mirix"):
                old = old_maps.get(f"{name}_sha256")
                new = self.code_files.get(f"{name}_sha256")
                if isinstance(old, Mapping) and isinstance(new, Mapping):
                    code_files[name] = {
                        "changed": sorted(
                            p for p in new if p in old and old[p] != new[p]
                        ),
                        "added": sorted(p for p in new if p not in old),
                        "removed": sorted(p for p in old if p not in new),
                    }
        before_tree = (previous.get("tree") or {}).get("diff_sha256")
        return {
            "protocol": a != b,
            "protocol_diff": diff,
            "code": code,
            "code_files": code_files,
            "commit": (previous.get("git") or {}).get("commit")
            != self.git_now.get("commit"),
            "tree": before_tree != (document.get("tree") or {}).get("diff_sha256"),
        }

    async def _adopt_protocol(self) -> bool:
        """A new attempt runs under this tree's protocol.

        When ``cell.json`` ``cell`` names another one, its values move to
        ``protocol_history`` (``fingerprint``, ``constants_sha256`` and ``attempts``: the
        attempts that ran under them and no earlier entry lists, as
        ``protocol.attempt_protocol`` reads them, so an attempt written before attempts
        recorded their protocol stays attributed) and this tree's replace them. Returns
        whether the manifest changed; the caller writes it before the attempt's number is
        reserved. Only reached when no attempt can resume (the last one is invalid, or
        there is none), so no live attempt changes protocol."""
        cell = self.manifest["cell"]
        old = {
            "fingerprint": cell.get("protocol_fingerprint"),
            "constants_sha256": cell.get("protocol_constants_sha256"),
        }
        if old == self.protocol_now:
            return False
        history = list(self.manifest.get("protocol_history") or [])
        listed = {
            n
            for entry in history
            if isinstance(entry, Mapping)
            for n in entry.get("attempts") or ()
        }
        ran = []
        for n in await asyncio.to_thread(self.paths.attempt_numbers):
            state = await asyncio.to_thread(self._read_attempt, n)
            recorded = protocol.attempt_protocol(self.manifest, state, n)
            if n not in listed and recorded["fingerprint"] == old["fingerprint"]:
                ran.append(n)
        history.append(
            {**old, "attempts": ran, "replaced_at": now(), "replaced_by": self.launch}
        )
        self.manifest["protocol_history"] = history
        self.manifest["cell"] = {
            **cell,
            "protocol_fingerprint": self.protocol_now["fingerprint"],
            "protocol_constants_sha256": self.protocol_now["constants_sha256"],
        }
        log.warning(
            "%s: the new attempt adopts protocol %s; attempt(s) %s ran under %s "
            "(recorded in cell.json protocol_history)",
            self.spec.cell_id,
            self.protocol_now["fingerprint"],
            ", ".join(str(n) for n in ran) or "none",
            old["fingerprint"],
        )
        return True

    async def _new_attempt(
        self, factory: MethodFactory, *, why: str, supersedes: int | None = None
    ) -> CellOutcome:
        """Start a new attempt (``why``: what started it, recorded in its first version
        segment; ``supersedes``: the tested attempt it replaces). It runs under this
        tree's protocol, frozen automatically when needed (``_record_work``)."""
        assert self.gateway is not None
        await self._settle_reservations()
        number = await asyncio.to_thread(self._next_attempt_number)
        run_id = self._new_run_id(number)
        ledger = self.gateway.ledger_path(run_id)
        if await asyncio.to_thread(ledger.exists) or await asyncio.to_thread(
            (ledger.parent / "gateway_state.json").exists
        ):
            # Another cell directory with this instance id (a copied cell) already ran
            # this gateway run: its calls must never be counted as this attempt's.
            raise _Refused(
                f"the gateway already has a run {run_id} ({ledger.parent}); this cell "
                "directory shares its instance id with another one. Start the cell in "
                "a fresh directory."
            )
        paths = self.paths.attempt(number)
        adopted = await self._adopt_protocol()
        persisted = await asyncio.to_thread(read_json, self.paths.manifest)
        if (
            adopted
            or not isinstance(persisted, dict)
            or (persisted.get("instance") != self.manifest["instance"])
        ):
            # The instance id in the run id is the cell directory's for good from now on:
            # an interrupted first attempt (or the first new attempt of a directory made
            # before instance ids) must not leave a run of an instance id the next
            # invocation would not know. Likewise the protocol this attempt adopts is on
            # disk before its number is reserved.
            await self._write_manifest(None)
        # Reserve the number before the gateway hears of the run. An interruption
        # between set_phase and attempt.json leaves this reservation, so the next
        # invocation takes the next number (and abandons this run) instead of finding
        # the gateway run already there and refusing the cell for good.
        await asyncio.to_thread(paths.root.mkdir, parents=True, exist_ok=False)
        await asyncio.to_thread(
            write_json_atomic,
            paths.reservation,
            {
                "attempt": number,
                "run_id": run_id,
                "instance": self.manifest["instance"],
                "launch": self.launch,
                "at": now(),
            },
        )
        try:
            await self.gateway.set_phase(run_id, "setup")
        except Exception as exc:
            raise _InfraUnavailable(
                f"gateway {self.gateway.base_url} refused the new run {run_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        await self._check_ledger_link(run_id)
        await asyncio.to_thread(paths.method_dir.mkdir, parents=True, exist_ok=False)
        checkpointed = bool(self.learning)
        if checkpointed:
            # Its presence marks an attempt that writes learning checkpoints: without it
            # (an older harness, or no learning items) a stopped learning phase is not
            # resumed but restarted (H8).
            await asyncio.to_thread(
                paths.checkpoints.mkdir, parents=True, exist_ok=False
            )
        await self._save_attempt(
            number,
            {
                "attempt": number,
                "run_id": run_id,
                "instance": self.manifest["instance"],
                "state": ATTEMPT_LEARNING,
                "started_at": now(),
                "started_by": self.launch,
                "host": self.host,
                "git": self.git_now,
                "code": self.code_now,
                # The agent context is harness code outside the protocol fingerprint:
                # its digest, so cells and replicates can be compared on it
                # (tools/aggregate_replicates.py).
                "agent_context": context_record(env_agent_context(self.env)),
                # The protocol the attempt started under; later ones are in
                # version_segments.
                "protocol": dict(self.protocol_now),
                "version_segments": [],
                "versions_mixed": {"protocol": False, "code": False, "commit": False},
                "note": (
                    {
                        "text": self.options.note,
                        "at": now(),
                        "launch": self.launch,
                        "new_attempt": bool(self.options.new_attempt),
                    }
                    if self.options.note
                    else None
                ),
                "supersedes": supersedes,
                "gateway_base_url": self.gateway.base_url,
                "ledger": str(self.gateway.ledger_path(run_id)),
                # Where the ledger is relative to a gateway runs directory, so a runs
                # directory copied to another host still finds it.
                "ledger_rel": f"{run_id}/gateway.jsonl",
                "checkpoint_format": CHECKPOINT_FORMAT if checkpointed else None,
                "learning": {
                    "items": len(self.learning),
                    "done": 0,
                    "played": 0,
                    "checkpoint": None,
                },
                "resumes": [],
                "test": {"items": len(self.test)},
                "test_reruns": [],
                "snapshot_checks": [],
            },
        )
        if supersedes is not None:
            old = await asyncio.to_thread(self._read_attempt, supersedes)
            superseded = old.get("superseded")
            await self._update_attempt(
                supersedes,
                superseded={
                    **(superseded if isinstance(superseded, Mapping) else {}),
                    "by_attempt": number,
                },
            )
        await self._record_work(
            number,
            "learning",
            why=why,
            learning_from={"position": 0, "stage": "setup", "checkpoint": None},
        )
        await self._write_manifest(number)
        await self._set_status(STATUS_RUNNING, attempt=number, phase="learning")

        async def open_new() -> _Attempt:
            return await self._open_attempt(
                number, factory, phase="learning", mirix_mode="fresh"
            )

        return await self._run_learning(number, open_new, init=checkpointed)

    async def _run_learning(
        self,
        number: int,
        open_attempt: Callable[[], Awaitable[_Attempt]],
        *,
        init: bool,
        start: int = 0,
        played: _Checkpoint | None = None,
    ) -> CellOutcome:
        """Open the attempt, take its ``init`` checkpoint (a new learning phase), finish a
        ``played`` item (a resume from its ``played`` checkpoint), learn the items from
        ``start``, freeze, and go on to the test phase."""
        attempt: _Attempt | None = None
        try:
            try:
                attempt = await open_attempt()
                if init:
                    await self._write_checkpoint(attempt, CHECKPOINT_INIT, 0)
                if played is not None:
                    await self._finish_played(attempt, played)
                await self._learn(attempt, start=start)
                frozen = await self._freeze(attempt)
            except asyncio.CancelledError:
                await asyncio.shield(self._learning_interrupted(number))
                raise
            except _LearningFailed as exc:
                return await self._learning_failed(number, exc)
            except Exception as exc:
                return await self._learning_failed(
                    number,
                    _LearningFailed(classify_learning_exception(exc), stage="cell"),
                )
            return await self._test_phase(attempt, frozen)
        finally:
            await self._close_attempt(attempt)

    def _checkpointed(self, number: int) -> bool:
        """Whether the attempt writes learning checkpoints (blocking)."""
        return self.paths.attempt(number).checkpoints.is_dir()

    async def _learning_interrupted(self, number: int) -> None:
        """SIGINT/SIGTERM in the learning phase: the attempt stays resumable (no abandon);
        the next invocation resumes it, or restarts it as a new attempt when it has no
        learning checkpoints (``_resume_learning``)."""
        state = await asyncio.to_thread(self._read_attempt, number)
        if state.get("state") not in (ATTEMPT_LEARNING, ATTEMPT_LEARNING_INTERRUPTED):
            return
        state.update(
            state=ATTEMPT_LEARNING_INTERRUPTED,
            failure=None,
            failure_stage=None,
            resume_from=None,
            interruption={"reason": "interrupted", "at": now(), "launch": self.launch},
        )
        await self._save_attempt(number, state)

    async def _learning_failed(self, number: int, exc: _LearningFailed) -> CellOutcome:
        """The learning phase stopped on ``exc``. An infrastructure failure leaves the
        attempt ``learning_interrupted`` (resumed by the next invocation from its
        checkpoint); a harness failure leaves it for a person (``needs_decision``). An
        attempt without checkpoints stops the same way; the next invocation restarts it as
        a new attempt (``_resume_learning``)."""
        failure = exc.failure
        paths = self.paths.attempt(number)
        item = (
            self.learning[exc.position].uid(self.salt)
            if exc.position is not None and exc.position < len(self.learning)
            else None
        )
        checkpointed = await asyncio.to_thread(self._checkpointed, number)
        await asyncio.to_thread(
            _record_learning_failure,
            paths.failures,
            failure,
            item=item,
            attempt=number,
            stage=exc.stage,
            checkpointed=checkpointed,
        )
        where = f" at learning item {exc.position}" if exc.position is not None else ""
        harness = failure.kind is FailureKind.HARNESS
        if harness:
            reason = (
                f"learning stopped{where} ({exc.stage}: {failure.kind.value}/"
                f"{failure.code}): the harness's or MIRIX's own failure, which a "
                "resume would repeat; nothing resumes it automatically. "
                + _harness_stop_choices(failure.code, checkpointed=checkpointed)
            )
            status = STATUS_LEARNING_FAILED
        else:
            reason = (
                f"learning interrupted{where} ({exc.stage}: {failure.kind.value}/"
                + (
                    f"{failure.code}); the next invocation resumes it from its checkpoint"
                    if checkpointed
                    else f"{failure.code}); the next invocation restarts it as a new attempt "
                    "(it has no learning checkpoints)"
                )
            )
            status = STATUS_LEARNING_INTERRUPTED
        state = await asyncio.to_thread(self._read_attempt, number)
        if exc.keep_cause:
            # The resume stopped before anything changed: the attempt keeps the cause
            # the resume a person decides must still discard (its failure, the
            # checkpoint it must restore, how it was interrupted), so the discarded row
            # names that cause and not this stop.
            state.update(
                needs_decision=(
                    NEEDS_DECISION_HARNESS if harness else state.get("needs_decision")
                ),
                resume_conflict={
                    "failure": failure.to_dict(),
                    "stage": exc.stage,
                    "at": now(),
                    "launch": self.launch,
                },
            )
        else:
            # A conflict an earlier resume recorded is not this stop's (it would name
            # the conflict, and its advice, in place of this failure's); the discarded
            # rows keep it.
            state.pop("resume_conflict", None)
            state.update(
                state=ATTEMPT_LEARNING_INTERRUPTED,
                failure=failure.to_dict(),
                failure_stage=exc.stage,
                failure_position=exc.position,
                failure_at=now(),
                failure_launch=self.launch,
                resume_from=exc.resume_from,
                needs_decision=NEEDS_DECISION_HARNESS if harness else None,
                interruption=None,
            )
        await self._save_attempt(number, state)
        await self._write_costs(number)
        await self._write_manifest(number)
        return CellOutcome(
            self.spec.cell_id,
            status,
            number,
            f"{reason}: {failure.detail}"[:1000],
            failure_kind=failure.kind.value,
        )

    async def _open_attempt(
        self,
        number: int,
        factory: MethodFactory,
        *,
        phase: str,
        mirix_mode: ServerMode,
        resume: LearningResume | None = None,
        server: tuple[Any, CellServerSpec] | None = None,
    ) -> _Attempt:
        """Start the attempt's MIRIX server (if any), its actor and a new method instance
        (``setup`` in ``phase``; ``resume`` restores a learning checkpoint)."""
        assert self.gateway is not None
        paths = self.paths.attempt(number)
        run_id = await asyncio.to_thread(self._attempt_run_id, number)
        started, options = await self._start_mirix(
            number, mode=mirix_mode, prepared=server
        )
        actor = Actor(
            gateway_base_url=self.gateway.base_url,
            run_id=run_id,
            model=self.spec.model,
        )
        attempt = _Attempt(
            number=number,
            paths=paths,
            run_id=run_id,
            method=None,
            actor=actor,
            server=started,
            learning_rows=JsonlWriter(paths.learning),
            result_rows=JsonlWriter(paths.results),
            injections=JsonlWriter(paths.injections),
            failures=FailureLog(paths.failures),
            discarded=await asyncio.to_thread(discarded_call_ids, paths),
            accepted=_accepted_call_ids(
                await asyncio.to_thread(self._read_attempt, number)
            ),
        )
        try:
            if started is not None:
                attempt.store = self._checkpoint_store(started)
                attempt.home = paths.home
            await actor.start()
            method = factory()
            name = getattr(method, "name", None)
            if name != self.spec.method:
                raise RuntimeError(
                    f"the method instance is named {name!r}, the cell's method is "
                    f"{self.spec.method!r}"
                )
            attempt.method = method
            cell = CellContext(
                cell_id=self.spec.cell_id,
                attempt=number,
                benchmark=self.spec.benchmark,
                method=self.spec.method,
                model=self.spec.model,
                gateway_base_url=self._method_gateway_url or self.gateway.base_url,
                run_id=run_id,
                gateway_token=LOCAL_TOKEN,
                # cell_dir is left to default to work_dir: a method sees its own
                # directory only, never the scorer's records next to it (H10.1). run.py
                # runs the MIRIX server, so the method needs no cell directory.
                work_dir=paths.method_dir,
                phase=phase,  # type: ignore[arg-type]
                options=options,
                resume=resume,
            )
            await asyncio.wait_for(method.setup(cell), SETUP_TIMEOUT_S)
        except BaseException:
            await self._close_attempt(attempt)
            raise
        return attempt

    def _new_mirix_server(
        self, number: int, mode: ServerMode
    ) -> tuple[Any, CellServerSpec]:
        """A MIRIX server of attempt ``number`` (blocking: reads ``attempt.json``). Its
        database is the one the attempt recorded (``mirix.db_name``: an older attempt
        keeps its name), else the name bound to the cell directory's instance id, so
        two runs directories of one cell id never share one; the attempt's gateway run
        id marks it as the attempt's. MIRIX sends the session each call serves as the
        gateway's item."""
        state = self._read_attempt(number)
        mirix = state.get("mirix")
        recorded = mirix.get("db_name") if isinstance(mirix, Mapping) else None
        instance = state.get("instance") or self.manifest.get("instance")
        port = find_free_port()
        spec = CellServerSpec(
            self.spec.cell_id,
            number,
            self.paths.root,
            port,
            # MIRIX names the session (task uid) each call serves in the gateway's
            # item header, so its calls are ledgered per item (headers only: nothing
            # the model sees changes).
            extra_env=MappingProxyType({MIRIX_SESSION_HEADER_ENV: ITEM_HEADER}),
            instance=str(instance) if instance else None,
            owner=str(state["run_id"]) if state.get("run_id") else None,
            database=recorded if isinstance(recorded, str) and recorded else None,
        )
        return self.mirix_server_factory(spec, mode), spec

    @staticmethod
    def _checkpoint_store(server: Any) -> CheckpointStore:
        """The cell database of a MIRIX server, for learning checkpoints."""
        getter = getattr(server, "checkpoint_store", None)
        if not callable(getter):
            raise RuntimeError(
                f"the MIRIX server {type(server).__name__} has no checkpoint_store(): "
                "its database cannot be checkpointed"
            )
        return cast(CheckpointStore, getter())

    async def _start_mirix(
        self,
        number: int,
        *,
        mode: ServerMode,
        prepared: tuple[Any, CellServerSpec] | None = None,
    ) -> tuple[MirixServer | None, dict[str, Any]]:
        """The per-cell MIRIX server of the MIRIX method (H7.2, H8): fresh for a new attempt,
        resumed (same database and HOME) for its test phase or a learning resume.
        ``prepared`` is a server made (not started) by :meth:`_new_mirix_server`."""
        if self.spec.method != MIRIX_METHOD:
            return None, {}
        for attempt in range(MIRIX_PORT_TRIES):
            if prepared is not None and attempt == 0:
                server, spec = prepared
            else:
                server, spec = await asyncio.to_thread(
                    self._new_mirix_server, number, mode
                )
            port = spec.port
            try:
                await server.start()
            except MirixServerError as exc:
                # Another cell took the port between the probe and the bind; nothing
                # (database, HOME) was created yet at that check.
                if "already in use" in str(exc) and attempt + 1 < MIRIX_PORT_TRIES:
                    continue
                raise
            await self._update_attempt(
                number,
                mirix={
                    "base_url": server.base_url,
                    "port": port,
                    "mode": mode,
                    "db_name": getattr(
                        getattr(server, "spec", None), "db_name", spec.db_name
                    ),
                    "home": str(spec.home),
                    "llm_contract": getattr(server, "llm_contract", None),
                },
            )
            return server, {
                "mirix_base_url": server.base_url,
                "mirix_port": port,
                "mirix_finalize_timeout_s": self.options.finalize_timeout_s,
            }
        raise MirixServerError("no free port for the MIRIX server")  # pragma: no cover

    async def _close_attempt(self, attempt: _Attempt | None) -> None:
        if attempt is None:
            return
        errors: list[str] = []
        close = getattr(attempt.method, "close", None)
        if callable(close):
            try:
                await close()
            except Exception as exc:
                errors.append(f"method.close: {type(exc).__name__}: {exc}")
        try:
            await attempt.actor.aclose()
        except Exception as exc:
            errors.append(f"actor: {type(exc).__name__}: {exc}")
        if attempt.server is not None:
            try:
                await attempt.server.stop()
            except Exception as exc:
                errors.append(f"mirix server: {type(exc).__name__}: {exc}")
        for error in errors:
            log.warning("closing attempt %s: %s", attempt.number, error)

    # -- learning (H1.4) -----------------------------------------------------------------

    async def _learn(self, attempt: _Attempt, *, start: int = 0) -> None:
        """The learning items from ``start``: each played and scored, recorded and
        checkpointed (``played``) before ``after_task``, then checkpointed again after its
        idle wait (``done``). An actor call the provider refuses (a payload or no-endpoint
        refusal) ends the item unanswered, scored 0 and flagged
        ``provider_unserved``; the same refusal on ``REFUSAL_STREAK_ITEMS`` consecutive
        items stops the learning phase (HARNESS ``upstream_refusal_streak``) after the last
        one's ``done`` checkpoint, so ``--force-resume`` goes on with the next item."""
        assert self.gateway is not None and self.env is not None
        await self.gateway.set_phase(attempt.run_id, "learning")
        for position in range(start, len(self.learning)):
            task = self.learning[position]
            if attempt.server is not None:
                try:
                    await attempt.server.ensure_alive()
                except Exception as exc:
                    raise _LearningFailed(
                        classify_exception(exc), stage="mirix_server", position=position
                    ) from exc

            async def on_played(
                run: EpisodeRun, position: int = position, task: Task = task
            ) -> None:
                # The item is persisted as soon as it is scored, and checkpointed
                # before the method learns from it.
                try:
                    await self._record(
                        attempt, "learning", position, task, run.trajectory, run.result
                    )
                    if run.trajectory.end_reason == END_PROVIDER_UNSERVED:
                        # Recorded with the row, before the checkpoint that keeps it: a
                        # resume from played(p) never records it twice, and one before
                        # it plays the item (and records its own refusal) again.
                        await self._record_unserved(attempt, task, run.trajectory)
                    await self._write_checkpoint(attempt, CHECKPOINT_PLAYED, position)
                except _LearningFailed:
                    raise
                except Exception as exc:
                    raise _LearningFailed(
                        classify_learning_exception(exc),
                        stage=STAGE_PLAYED,
                        position=position,
                    ) from exc

            try:
                run = await run_episode(
                    self.env,
                    task,
                    method=attempt.method,
                    actor=attempt.actor,
                    salt=self.salt,
                    phase="learning",
                    idle_timeout_s=self.options.idle_timeout_s,
                    on_played=on_played,
                    unserved=REFUSAL_CLASSES,
                )
            except RolloutError as err:
                cause = err.__cause__
                if isinstance(cause, _LearningFailed):
                    raise _LearningFailed(
                        cause.failure,
                        stage=cause.stage,
                        position=position,
                        resume_from=cause.resume_from,
                    ) from err
                if err.stage not in (STAGE_PLAYED, STAGE_AFTER_TASK, STAGE_IDLE_WAIT):
                    # Not played: the failed try is recorded (never scored) for
                    # forensics; a resume moves it aside with the rest of the try.
                    await self._record(
                        attempt,
                        "learning",
                        position,
                        task,
                        err.trajectory,
                        err.result,
                        failure=err.failure,
                        stage=err.stage,
                    )
                # Played: its row exists; a second one is never written (failures.jsonl
                # records the failure).
                raise _LearningFailed(
                    err.failure, stage=err.stage, position=position
                ) from err
            await self._item_learned(
                attempt,
                position,
                {
                    k: v
                    for k, v in run.trajectory.timings.items()
                    if k in ("after_task_s", "idle_wait_s", "total_s")
                },
            )
            if run.trajectory.end_reason == END_PROVIDER_UNSERVED:
                await self._stop_on_refusal_streak(attempt, position)

    async def _record_unserved(
        self, attempt: _Attempt, task: Task, trajectory: Trajectory
    ) -> None:
        """The ``failures.jsonl`` row of a learning item whose actor call the provider
        refused (scored unanswered): ``provider_unserved``, ``score_and_continue``."""
        recorded = trajectory.failure
        if not isinstance(recorded, Mapping):
            return
        await asyncio.to_thread(
            _record_learning_failure,
            attempt.paths.failures,
            Failure.from_dict(recorded),
            item=task.uid(self.salt),
            attempt=attempt.number,
            stage=STAGE_ACTOR,
            checkpointed=True,
        )

    async def _stop_on_refusal_streak(self, attempt: _Attempt, position: int) -> None:
        """After item ``position``'s ``done`` checkpoint: the same provider refusal on the
        last ``REFUSAL_STREAK_ITEMS`` items stops the learning phase for a person (the
        refusal repeats: continuing would score every item unanswered)."""
        rows = await asyncio.to_thread(read_jsonl, attempt.paths.learning)
        stop = learning_refusal_streak(rows)
        if stop is not None:
            raise _LearningFailed(stop, stage="refusal_streak", position=position)

    async def _item_learned(
        self, attempt: _Attempt, position: int, timings: Mapping[str, float]
    ) -> None:
        """After an item's idle wait: the ledger check, then its ``done`` checkpoint."""
        await self._check_learning_calls(attempt, position)
        await self._write_checkpoint(
            attempt, CHECKPOINT_DONE, position, item_timings=dict(timings)
        )

    async def _finish_played(self, attempt: _Attempt, checkpoint: _Checkpoint) -> None:
        """A resume from item p's ``played`` checkpoint: ``after_task`` and the idle wait on
        the view rebuilt from p's recorded trajectory, then p's ``done`` checkpoint."""
        position = checkpoint.position
        if attempt.server is not None:
            try:
                await attempt.server.ensure_alive()
            except Exception as exc:
                raise _LearningFailed(
                    classify_exception(exc), stage="mirix_server", position=position
                ) from exc
        try:
            view = await self._rebuilt_played_view(attempt, position)
        except Exception as exc:
            failure = classify_learning_exception(exc)
            if failure.code != FAILURE_DISK_FULL and isinstance(
                exc, (OSError, ValueError, KeyError)
            ):
                # A missing or corrupt trajectory record, or an image blob the
                # benchmark data does not re-create: the checkpoint cannot be continued
                # as recorded (a person looks; _BlobsLost says what to do).
                cause = (
                    str(exc)
                    if isinstance(exc, _BlobsLost)
                    else f"{type(exc).__name__}: {exc}"
                )
                failure = Failure(
                    FailureKind.HARNESS,
                    FAILURE_CHECKPOINT_MISMATCH,
                    None,
                    None,
                    f"the trajectory of item {position} cannot be rebuilt: {cause}"[
                        :500
                    ],
                )
            raise _LearningFailed(failure, stage="resume", position=position) from exc
        try:
            timings = await finish_learning(
                attempt.method, view, idle_timeout_s=self.options.idle_timeout_s
            )
        except LearningHookError as err:
            raise _LearningFailed(
                err.failure, stage=err.stage, position=position
            ) from err
        await self._item_learned(attempt, position, timings)
        rows = await asyncio.to_thread(read_jsonl, attempt.paths.learning)
        played = [r for r in rows if r.get("position") == position]
        if played and played[-1].get("end_reason") == END_PROVIDER_UNSERVED:
            await self._stop_on_refusal_streak(attempt, position)

    async def _rebuilt_played_view(self, attempt: _Attempt, position: int) -> Any:
        """:meth:`_played_view` of item ``position``, after re-creating the image blobs
        its record references that are missing or corrupt. Blobs are content-addressed,
        and an image is the benchmark's own data: a fresh episode's ``reset`` of the
        same task shows the same data URL, so a blob a crash emptied (a filesystem's
        delayed allocation) is written again, durably, under the same name. Only a blob
        the benchmark data does not re-create stops the resume (:class:`_BlobsLost`,
        which says what to do)."""
        blobs = self.options.blobs
        record = await asyncio.to_thread(self._played_record, attempt, position)
        lost = await asyncio.to_thread(unreadable_blobs, record, blobs)
        if lost:
            why = await self._recreate_blobs(position, blobs)
            still = await asyncio.to_thread(unreadable_blobs, record, blobs)
            if still:
                error = _BlobsLost(position, blobs, still, why)
                log.error("%s: %s", self.spec.cell_id, error)
                raise error
            log.warning(
                "%s: item %d's image blob(s) %s were missing or corrupt in %s; "
                "re-created from the benchmark data",
                self.spec.cell_id,
                position,
                ", ".join(lost),
                blobs,
            )
        return await asyncio.to_thread(self._played_view, attempt, position, record)

    async def _recreate_blobs(self, position: int, blobs: Path) -> str | None:
        """Write the image blobs of learning item ``position`` again from a fresh
        episode's ``reset``; None, or why that failed (the resume then
        stops for a person with the blobs still missing)."""
        assert self.env is not None
        task = self.learning[position]
        try:
            episode = self.env.episode(task)
            try:
                reset = await episode.reset()
            finally:
                await episode.close()
            await asyncio.to_thread(
                externalize_blobs,
                {"messages": list(reset.messages), "fields": dict(reset.fields)},
                blobs,
            )
        except Exception as exc:
            log.warning(
                "%s: cannot re-create item %d's image blobs: %s: %s",
                self.spec.cell_id,
                position,
                type(exc).__name__,
                exc,
            )
            return f"{type(exc).__name__}: {exc}"
        return None

    def _played_record(self, attempt: _Attempt, position: int) -> dict[str, Any]:
        """Item ``position``'s recorded trajectory, as written (blobs not restored)."""
        rows = read_jsonl(attempt.paths.learning)
        rel = _learning_row_trajectory(rows, position)  # relative to the cell directory
        return json.loads((self.paths.root / rel).read_text(encoding="utf-8"))

    def _played_view(
        self,
        attempt: _Attempt,
        position: int,
        record: Mapping[str, Any] | None = None,
    ) -> Any:
        if record is None:
            record = self._played_record(attempt, position)
        view = trajectory_view_from_record(record, self.options.blobs)
        expected = self.learning[position].uid(self.salt)
        if view.task_uid != expected:
            raise CheckpointMismatch(
                f"the recorded trajectory of item {position} is of {view.task_uid}, not "
                f"{expected}"
            )
        return view

    async def _check_learning_calls(
        self, attempt: _Attempt, position: int | None = None
    ) -> None:
        """H8 for calls a method's own retries swallow: an infrastructure failure on
        the ``mirix`` or ``embedding`` role in this attempt's learning phase that no later
        retry of the same request recovered means the memory is not what it would have
        been without the outage (MIRIX deferred a session, fell back to retrieving with
        the raw text, or stored a skill without its vector). The learning phase stops;
        its resume restores the latest checkpoint taken before that call started
        (``resume_from``), so the memory never keeps the call's effect. Calls a learning
        resume discarded are not the live memory's and are left out.

        A batch rolled back ``MIRIX_BATCH_ROLLBACKS`` times in a row (the same checkpoint
        restored after the same failure) is accepted when its replay loses calls again
        (recorded, not blocked): the first check after such a
        resume lists them in ``attempt.json`` ``accepted_learning_calls`` and one
        ``failures.jsonl`` row (``accept_failed_batch``), and the learning phase goes on;
        the method records what it tolerated itself (MIRIX's ``tolerated`` block)."""
        assert self.gateway is not None
        reader = self._learning_ledger
        await asyncio.to_thread(reader.read, self.gateway.ledger_path(attempt.run_id))
        lines = [
            row
            for row in reader.role_lines
            if str(row.get("call_id")) not in attempt.discarded
            and str(row.get("call_id")) not in attempt.accepted
        ]
        lost = unrecovered_infra_calls(
            lines, roles=LEARNING_CHECK_ROLES, phases=("setup", "learning")
        )
        after_resume, attempt.after_resume = attempt.after_resume, False
        if lost:
            lost = await self._tolerate_degraded_retrievals(attempt, lost)
        if not lost:
            return
        if after_resume and attempt.rollbacks >= MIRIX_BATCH_ROLLBACKS:
            await self._accept_lost_calls(attempt, lost, position)
            return
        first = lost[0]
        error = first.get("error") if isinstance(first.get("error"), Mapping) else {}
        detail = (
            f"{len(lost)} {first.get('role')} call(s) failed on infrastructure and no "
            f"retry recovered them (first: call {first.get('call_id')}, HTTP "
            f"{first.get('http_status')}, {error.get('type')}: {error.get('message')})"
        )
        # Every lost call is placed (its call_started line, else its own final line),
        # so the resume restores the newest checkpoint before the first of them, never
        # the start for want of a line.
        started = [
            reader.started[str(line.get("call_id"))].offset
            for line in lost
            if str(line.get("call_id")) in reader.started
        ]
        resume_from = await asyncio.to_thread(
            self._checkpoint_before,
            attempt.paths,
            min(started) if started else reader.offset,
        )
        raise _LearningFailed(
            Failure(
                FailureKind.INFRA,
                MIRIX_LEARNING,
                str(first.get("role")),
                first.get("http_status"),
                detail[:500],
            ),
            stage="learning_calls",
            position=position,
            resume_from=resume_from,
        )

    async def _tolerate_degraded_retrievals(
        self, attempt: _Attempt, lost: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """A learning retrieval MIRIX injected although it lost its topic extraction or
        query vector is learned on, not rolled back. Each lost call that arrived inside the window of such a
        retrieval (:func:`degraded_retrievals`) is recorded in ``attempt.json``
        ``degraded_retrieval_calls``, each retrieval in one ``failures.jsonl`` row (INFRA
        ``mirix_retrieval``, ``count_and_continue``), and left out of this and every later
        check (``attempt.accepted``). The other lost calls are returned."""
        paths = attempt.paths
        retrievals = await asyncio.to_thread(
            lambda: degraded_retrievals(
                read_jsonl(paths.method_dir / MIRIX_RETRIEVAL_RECORD),
                read_jsonl(paths.injections),
            )
        )
        held: dict[int, list[Mapping[str, Any]]] = {}
        rest: list[Mapping[str, Any]] = []
        for line in lost:
            index = next((i for i, r in enumerate(retrievals) if r.holds(line)), None)
            if index is None:
                rest.append(line)
            else:
                held.setdefault(index, []).append(line)
        if not held:
            return rest
        state = await asyncio.to_thread(self._read_attempt, attempt.number)
        recorded = list(state.get(DEGRADED_RETRIEVAL_CALLS) or [])
        known = {
            str(row.get("call_id")) for row in recorded if isinstance(row, Mapping)
        }
        at = now()
        fresh: dict[int, list[dict[str, Any]]] = {}
        for index, lines in held.items():
            retrieval = retrievals[index]
            for line in lines:
                call_id = str(line.get("call_id"))
                attempt.accepted = attempt.accepted | {call_id}
                if call_id in known:
                    continue
                error = line.get("error")
                fresh.setdefault(index, []).append(
                    {
                        "call_id": call_id,
                        "role": line.get("role"),
                        "http_status": line.get("http_status"),
                        "error_type": error.get("type")
                        if isinstance(error, Mapping)
                        else None,
                        "position": retrieval.position,
                        "task_uid": retrieval.record.get("task_uid"),
                        "retrieval_id": retrieval.record.get("retrieval_id"),
                        "started_at": retrieval.record.get("started_at"),
                        "ended_at": retrieval.record.get("at"),
                        "launch": self.launch,
                        "at": at,
                    }
                )
        if not fresh:
            return rest
        state[DEGRADED_RETRIEVAL_CALLS] = [
            *recorded,
            *(row for rows in fresh.values() for row in rows),
        ]
        await self._save_attempt(attempt.number, state)
        for rows in fresh.values():
            await self._record_degraded_retrieval(attempt, rows)
        return rest

    async def _record_degraded_retrieval(
        self, attempt: _Attempt, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """One ``failures.jsonl`` row (INFRA ``mirix_retrieval``, ``count_and_continue``)
        for the lost calls of one degraded learning retrieval the method injected
        (:meth:`_tolerate_degraded_retrievals`), and a warning in the worker log."""
        first = rows[0]
        ids = [str(row["call_id"]) for row in rows]
        detail = (
            f"{len(rows)} lost call(s) of retrieval {first.get('retrieval_id')} "
            f"({first.get('started_at')}..{first.get('ended_at')}), which MIRIX injected "
            f"flagged degraded, learned on: {', '.join(ids[:10])}"
            + (f" (+{len(ids) - 10} more)" if len(ids) > 10 else "")
        )
        task_uid = first.get("task_uid")
        await asyncio.to_thread(
            _record_learning_failure,
            attempt.paths.failures,
            Failure(
                FailureKind.INFRA,
                MIRIX_RETRIEVAL,
                str(first.get("role")) if first.get("role") else None,
                first.get("http_status"),
                detail[:500],
            ),
            item=str(task_uid) if task_uid is not None else None,
            attempt=attempt.number,
            stage=MIRIX_RETRIEVAL,
            checkpointed=True,
            recovery=Recovery.COUNT_AND_CONTINUE.value,
        )
        log.warning(
            "%s: attempt %d learns on a degraded retrieval: %s",
            self.spec.cell_id,
            attempt.number,
            detail,
        )

    async def _accept_lost_calls(
        self,
        attempt: _Attempt,
        lost: Sequence[Mapping[str, Any]],
        position: int | None,
    ) -> None:
        """Accept the calls a batch lost again after ``attempt.rollbacks`` rollbacks of
        its checkpoint: each in ``attempt.json`` ``accepted_learning_calls`` and one
        ``failures.jsonl`` row (``mirix_learning``, ``accept_failed_batch``)."""
        at = now()
        rows = []
        for line in lost:
            error = line.get("error") if isinstance(line.get("error"), Mapping) else {}
            rows.append(
                {
                    "call_id": str(line.get("call_id")),
                    "role": line.get("role"),
                    "http_status": line.get("http_status"),
                    "error_type": error.get("type"),
                    "position": position,
                    "checkpoint": attempt.rollback_checkpoint,
                    "rollbacks": attempt.rollbacks,
                    "launch": self.launch,
                    "at": at,
                }
            )
        state = await asyncio.to_thread(self._read_attempt, attempt.number)
        state["accepted_learning_calls"] = [
            *(state.get("accepted_learning_calls") or []),
            *rows,
        ]
        await self._save_attempt(attempt.number, state)
        attempt.accepted = attempt.accepted | {row["call_id"] for row in rows}
        first = lost[0]
        ids = [row["call_id"] for row in rows]
        detail = (
            f"{len(rows)} lost call(s) accepted after {attempt.rollbacks} rollbacks of "
            f"checkpoint {attempt.rollback_checkpoint}: {', '.join(ids[:10])}"
            + (f" (+{len(ids) - 10} more)" if len(ids) > 10 else "")
        )
        await asyncio.to_thread(
            _record_learning_failure,
            attempt.paths.failures,
            Failure(
                FailureKind.INFRA,
                MIRIX_LEARNING,
                str(first.get("role")) if first.get("role") else None,
                first.get("http_status"),
                detail[:500],
            ),
            item=(
                self.learning[position].uid(self.salt)
                if position is not None and position < len(self.learning)
                else None
            ),
            attempt=attempt.number,
            stage=MIRIX_LEARNING,
            checkpointed=True,
            recovery=Recovery.ACCEPT_FAILED_BATCH.value,
        )
        log.warning(
            "%s: attempt %d accepts %s (the method records what it tolerated)",
            self.spec.cell_id,
            attempt.number,
            detail,
        )

    @staticmethod
    def _checkpoint_before(paths: AttemptPaths, offset: int) -> str:
        """The newest kept checkpoint whose ledger marker precedes ``offset`` (a lost
        call's ``call_started``), else :data:`RESUME_FROM_START` (blocking)."""
        for _ordinal, cid, path in _checkpoint_dirs(paths.checkpoints):
            try:
                record = json.loads((path / CHECKPOINT_RECORD).read_text("utf-8"))
                marker = int(record["ledger_marker"]["offset"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if marker < offset:
                return cid
        return RESUME_FROM_START

    async def _freeze(self, attempt: _Attempt) -> dict[str, Any]:
        """H1.5: finalize, wait until idle, snapshot; the frozen record with its sha256."""
        method = attempt.method
        idle = self.options.idle_timeout_s
        stage = "finalize"
        try:
            await asyncio.wait_for(
                method.finalize(), self.options.finalize_timeout_s + IDLE_WAIT_GRACE_S
            )
            stage = "wait_until_idle"
            await asyncio.wait_for(
                method.wait_until_idle(idle), idle + IDLE_WAIT_GRACE_S
            )
            await self._check_learning_calls(attempt)
            stage = "snapshot"
            snapshot = await asyncio.wait_for(method.snapshot(), SNAPSHOT_TIMEOUT_S)
            record = await asyncio.to_thread(snapshot_record, snapshot)
        except _LearningFailed:
            raise
        except Exception as exc:
            raise _LearningFailed(
                classify_learning_exception(exc), stage=stage
            ) from exc
        frozen = {**record, "at": now(), "attempt": attempt.number}
        await asyncio.to_thread(
            write_json_atomic, attempt.paths.frozen_snapshot, frozen
        )
        await self._update_attempt(
            attempt.number,
            state=ATTEMPT_FROZEN,
            frozen={
                "sha256": record["sha256"],
                "at": frozen["at"],
                "path": self._rel(attempt.paths.frozen_snapshot),
            },
            learning={
                **(
                    (await asyncio.to_thread(self._read_attempt, attempt.number)).get(
                        "learning"
                    )
                    or {}
                ),
                "items": len(self.learning),
                "done": len(self.learning),
                "played": len(self.learning),
            },
        )
        return frozen

    # -- learning checkpoints ------------------------------------------------------------

    async def _write_checkpoint(
        self,
        attempt: _Attempt,
        kind: str,
        position: int,
        *,
        item_timings: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Take one learning checkpoint (see the module docstring): wait until the run is
        quiescent, then capture it. A SIGTERM during the capture waits for it (at most
        :data:`CHECKPOINT_SHIELD_S`), so an interruption never leaves a checkpoint half
        written behind a LATEST that names it."""
        t0 = time.monotonic()
        stage_position = position if kind != CHECKPOINT_INIT else None
        try:
            open_calls = await self._quiesce(attempt)
        except _LearningFailed:
            raise
        except Exception as exc:
            raise _LearningFailed(
                classify_learning_exception(exc),
                stage="checkpoint",
                position=stage_position,
            ) from exc
        if open_calls is not None:
            raise _LearningFailed(
                Failure(
                    FailureKind.INFRA,
                    FAILURE_CHECKPOINT_NOT_QUIESCENT,
                    None,
                    None,
                    f"checkpoint {checkpoint_id(kind, position)}: still busy after "
                    f"{CHECKPOINT_QUIESCE_S:.0f} s ({open_calls})"[:500],
                ),
                stage="checkpoint",
                position=stage_position,
            )
        # Where the ledger stood when the run was quiescent: a call that starts after
        # it, even before the checkpoint's marker, ran while the capture read the state.
        quiet_offset = self._learning_ledger.offset
        # Every call started before the marker has its final line now: a lost MIRIX or
        # embedding call is caught before a checkpoint could keep its effect.
        await self._check_learning_calls(attempt, stage_position)
        quiesce_s = time.monotonic() - t0
        work = asyncio.ensure_future(
            self._capture_checkpoint(
                attempt,
                kind,
                position,
                item_timings=item_timings,
                quiesce_s=quiesce_s,
                quiet_offset=quiet_offset,
            )
        )
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            if not work.done():
                try:
                    await asyncio.wait_for(work, CHECKPOINT_SHIELD_S)
                except BaseException:  # noqa: BLE001 - the interruption wins
                    pass
            raise
        except _LearningFailed:
            raise
        except Exception as exc:
            raise _LearningFailed(
                classify_learning_exception(exc),
                stage="checkpoint",
                position=stage_position,
            ) from exc

    async def _quiesce(self, attempt: _Attempt) -> str | None:
        """Wait until no call of the run is in flight and MIRIX has no active learning run;
        None when quiescent, else what was still busy after the bound."""
        assert self.gateway is not None
        health = await self.gateway.health()
        start_id = health.get("start_id")
        gateway_start = str(start_id) if start_id else None
        path = self.gateway.ledger_path(attempt.run_id)
        deadline = time.monotonic() + CHECKPOINT_QUIESCE_S
        while True:
            await asyncio.to_thread(self._learning_ledger.read, path)
            busy = self._learning_ledger.open_calls(
                exclude=attempt.discarded, gateway_start=gateway_start
            )
            active = await attempt.store.active_runs() if attempt.store else 0
            if not busy and not active:
                return None
            if time.monotonic() >= deadline:
                return (
                    f"{len(busy)} call(s) in flight {busy[:5]}, {active} active MIRIX "
                    "learning run(s)"
                )
            await asyncio.sleep(CHECKPOINT_POLL_S)

    async def _capture_checkpoint(
        self,
        attempt: _Attempt,
        kind: str,
        position: int,
        *,
        item_timings: Mapping[str, float] | None,
        quiesce_s: float,
        quiet_offset: int | None = None,
    ) -> dict[str, Any]:
        assert self.gateway is not None
        t0 = time.monotonic()
        cid = checkpoint_id(kind, position)
        paths = attempt.paths
        method = attempt.method
        stage_position = position if kind != CHECKPOINT_INIT else None
        state = checkpoint_state_json(await method.checkpoint())
        snapshot = await asyncio.wait_for(method.snapshot(), SNAPSHOT_TIMEOUT_S)
        record = await asyncio.to_thread(snapshot_record, snapshot)
        marked = await self.gateway.mark(
            attempt.run_id,
            {
                "event": LEDGER_EVENT_CHECKPOINT,
                "id": cid,
                "kind": kind,
                "position": position,
                "snapshot_sha256": record["sha256"],
            },
        )
        marker = await self._locate_marker(
            attempt.run_id, marked, LEDGER_EVENT_CHECKPOINT, id=cid
        )
        partial = paths.checkpoints / f"{cid}.partial"
        final = paths.checkpoint(cid)

        def build() -> dict[str, Any]:
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            method_files = _copy_method_dir(
                paths.method_dir, partial / "method", cache=attempt.object_hashes
            )
            (partial / "snapshot.json").write_bytes(canonical_json(record["snapshot"]))
            _fsync_path(partial / "snapshot.json")
            trajectories = [
                (Path("trajectories") / rel).as_posix()
                for rel in _tree_files(paths.trajectories_root)
            ]
            return {
                "method_files": method_files,
                "trajectories": trajectories,
                "learning_bytes": _file_size(paths.learning),
                "injections_bytes": _file_size(paths.injections),
                "learning_sha256": _prefix_sha256(paths.learning),
                "injections_sha256": _prefix_sha256(paths.injections),
            }

        built = await asyncio.to_thread(build)
        mirix: dict[str, Any] | None = None
        if attempt.store is not None:
            mirix = dict(await attempt.store.checkpoint(partial))
            if mirix.get("active_runs"):
                await asyncio.to_thread(shutil.rmtree, partial, True)
                raise _LearningFailed(
                    Failure(
                        FailureKind.INFRA,
                        FAILURE_CHECKPOINT_NOT_QUIESCENT,
                        None,
                        None,
                        f"checkpoint {cid}: {mirix['active_runs']} MIRIX learning run(s) "
                        "active in the dumped database",
                    ),
                    stage="checkpoint",
                    position=stage_position,
                )
            assert attempt.home is not None
            home = attempt.home

            def capture() -> dict[str, Any]:
                manifest = home_manifest(home, previous=attempt.home_manifest)
                capture_home(home, manifest, partial)
                data = _write_json_durable(partial / HOME_MANIFEST_FILE, manifest)
                return {
                    "manifest": manifest,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }

            captured = await asyncio.to_thread(capture)
            attempt.home_manifest = captured["manifest"]
            mirix["home_manifest_sha256"] = captured["sha256"]
            mirix["home_files"] = len(captured["manifest"]["files"])
            # The database was dumped after the snapshot the marker records. A MIRIX
            # write in between (its sweeper, with or without a gateway call) would leave
            # a dump that does not rebuild that snapshot, and the resume from it a
            # HARNESS checkpoint_mismatch: the memory must read the same after the dump.
            after = await asyncio.wait_for(method.snapshot(), SNAPSHOT_TIMEOUT_S)
            after_sha = (await asyncio.to_thread(snapshot_record, after))["sha256"]
            if after_sha != record["sha256"]:
                await asyncio.to_thread(shutil.rmtree, partial, True)
                raise _LearningFailed(
                    Failure(
                        FailureKind.INFRA,
                        FAILURE_CHECKPOINT_NOT_QUIESCENT,
                        None,
                        None,
                        f"checkpoint {cid}: the memory changed while it was captured "
                        f"(snapshot {record['sha256'][:12]} before the database dump, "
                        f"{after_sha[:12]} after it)",
                    ),
                    stage="checkpoint",
                    position=stage_position,
                )
        # A call that started after the run was found quiescent (so also one before the
        # marker, while the snapshot was read) ran while the capture read the state: the
        # capture may hold part of its effect.
        await asyncio.to_thread(
            self._learning_ledger.read, self.gateway.ledger_path(attempt.run_id)
        )
        reader = self._learning_ledger
        late = [
            c
            for c in (
                reader.started_after(marker.offset)
                if quiet_offset is None
                else reader.started_since(quiet_offset)
            )
            if c not in attempt.discarded
        ]
        if late:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            raise _LearningFailed(
                Failure(
                    FailureKind.INFRA,
                    FAILURE_CHECKPOINT_NOT_QUIESCENT,
                    None,
                    None,
                    f"checkpoint {cid}: {len(late)} call(s) started during the capture "
                    f"({late[:5]})",
                ),
                stage="checkpoint",
                position=stage_position,
            )
        task_uid = (
            self.learning[position].uid(self.salt)
            if kind != CHECKPOINT_INIT and position < len(self.learning)
            else None
        )
        checkpoint = {
            "id": cid,
            "kind": kind,
            "position": position,
            "ordinal": checkpoint_ordinal(kind, position),
            "attempt": attempt.number,
            "run_id": attempt.run_id,
            "task_uid": task_uid,
            "at": now(),
            "launch": self.launch,
            "ledger_marker": {
                "offset": marker.offset,
                "ts": marker.row.get("ts"),
                "gateway_start": marker.row.get("gateway_start"),
            },
            "method": {
                "state": state,
                "snapshot_sha256": record["sha256"],
                "files": built["method_files"],
            },
            "attempt_files": {
                "learning.jsonl": built["learning_bytes"],
                "injections.jsonl": built["injections_bytes"],
                "trajectories": built["trajectories"],
            },
            "attempt_files_sha256": {
                "learning.jsonl": built["learning_sha256"],
                "injections.jsonl": built["injections_sha256"],
            },
            "mirix": mirix,
            "item_timings": dict(item_timings or {}),
        }

        def finish() -> dict[str, Any]:
            files = {rel: _file_sha256(partial / rel) for rel in _tree_files(partial)}
            checkpoint["files"] = files
            checkpoint["timings"] = {
                "quiesce_s": round(quiesce_s, 3),
                "capture_s": round(time.monotonic() - t0, 3),
            }
            data = _write_json_durable(partial / CHECKPOINT_RECORD, checkpoint)
            for rel in files:
                if not rel.startswith(f"method/{OBJECTS_DIR}/"):
                    _fsync_path(partial / rel)  # objects are linked, already durable
            for directory in sorted(
                {(partial / rel).parent for rel in files} | {partial}
            ):
                _fsync_path(directory)
            if final.exists():
                shutil.rmtree(final)
            partial.rename(final)
            _fsync_path(paths.checkpoints)
            latest = {
                "id": cid,
                "ordinal": checkpoint["ordinal"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            write_json_atomic(paths.latest_checkpoint, latest)
            _fsync_path(paths.checkpoints)
            self._prune_checkpoints(paths, cid)
            return checkpoint

        written = await asyncio.to_thread(finish)
        state_now = await asyncio.to_thread(self._read_attempt, attempt.number)
        learning = dict(state_now.get("learning") or {})
        learning.update(
            items=len(self.learning),
            done=position + 1 if kind == CHECKPOINT_DONE else position,
            played=position if kind == CHECKPOINT_INIT else position + 1,
            checkpoint=cid,
        )
        await self._update_attempt(attempt.number, learning=learning)
        log.info(
            "%s: checkpoint %s (%.2f s + %.2f s quiescence)",
            self.spec.cell_id,
            cid,
            written["timings"]["capture_s"],
            quiesce_s,
        )
        return written

    @staticmethod
    def _prune_checkpoints(paths: AttemptPaths, latest: str) -> None:
        """Keep LATEST and its predecessor; remove every other checkpoint and partial."""
        root = paths.checkpoints
        kind, position = parse_checkpoint_id(latest)
        top = checkpoint_ordinal(kind, position)
        kept = [cid for ordinal, cid, _p in _checkpoint_dirs(root) if ordinal <= top][
            :CHECKPOINTS_KEPT
        ]
        for child in root.iterdir():
            if child.name == paths.latest_checkpoint.name or child.name in kept:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            elif child.name.startswith("."):
                child.unlink(missing_ok=True)  # a LATEST temporary file left by a crash

    async def _locate_marker(
        self, run_id: str, answer: Mapping[str, Any], event: str, **match: Any
    ) -> LedgerLine:
        """The marker line the gateway just wrote: at the offset it answered with when that
        line is the marker (always, for this checkout's gateway), else found by content."""
        assert self.gateway is not None
        path = self.gateway.ledger_path(run_id)
        offset = answer.get("offset")

        def at_offset() -> LedgerLine | None:
            if not isinstance(offset, int) or offset < 0:
                return None
            try:
                with path.open("rb") as handle:
                    handle.seek(offset)
                    raw = handle.readline()
            except OSError:
                return None
            try:
                row = json.loads(raw)
            except ValueError:
                return None
            if (
                isinstance(row, dict)
                and row.get("event") == event
                and all(row.get(k) == v for k, v in match.items())
            ):
                return LedgerLine(offset, row)
            return None

        found = await asyncio.to_thread(at_offset)
        if found is not None:
            return found
        await asyncio.to_thread(self._learning_ledger.read, path)
        line = self._learning_ledger.marker(event, **match)
        if line is None:
            raise RuntimeError(
                f"the gateway acknowledged a {event} marker ({dict(match)}) that is not in "
                f"{path}"
            )
        return line

    # -- learning resume -----------------------------------------------------------------

    async def _resume_learning(
        self, number: int, state: Mapping[str, Any], factory: MethodFactory
    ) -> CellOutcome:
        """Resume attempt ``number``'s learning phase from a checkpoint (see the module
        docstring). Every step before the ledger marker changes nothing, and every step
        after it is safe to repeat: an invocation that dies during a resume leaves the
        next one to finish it."""
        assert self.gateway is not None
        paths = self.paths.attempt(number)
        if not await asyncio.to_thread(self._checkpointed, number):
            # The single restart point of an attempt without learning checkpoints: it
            # has nothing to resume (setup and freeze are all it did). A harness failure
            # still waits for a person first.
            if state.get("needs_decision") and not self.options.force_resume:
                raise _Refused(self._needs_decision_refusal(number, state))
            failure = state.get("failure")
            await self._invalidate(
                number,
                "the learning phase stopped in an attempt with no learning checkpoints "
                "(a cell without learning items, or an older harness): it restarts as a "
                "new attempt (H8)",
                Failure.from_dict(failure) if isinstance(failure, Mapping) else None,
                stage=state.get("failure_stage"),
            )
            return await self._new_attempt(
                factory, why=f"restart of attempt {number} (no learning checkpoints)"
            )
        run_id = str(state.get("run_id") or "")
        await self._resume_guards(number, state, run_id)
        await self._settle_reservations()
        forced = bool(state.get("needs_decision")) and self.options.force_resume

        # Pick the checkpoint (read-only).
        try:
            checkpoint, notes = await asyncio.to_thread(
                self._pick_checkpoint, paths, state.get("resume_from")
            )
        except _CheckpointCorrupt as exc:
            return await self._learning_failed(
                number,
                _LearningFailed(
                    Failure(
                        FailureKind.HARNESS,
                        FAILURE_CHECKPOINT_CORRUPT,
                        None,
                        None,
                        str(exc)[:500],
                    ),
                    stage="resume",
                ),
            )
        mode: ServerMode = "resume" if checkpoint is not None else "fresh"
        prepared: tuple[Any, CellServerSpec] | None = None
        store: CheckpointStore | None = None
        segment = len(await asyncio.to_thread(read_discarded, paths)) + 1
        if self.spec.method == MIRIX_METHOD:
            prepared = await asyncio.to_thread(self._new_mirix_server, number, mode)
            store = self._checkpoint_store(prepared[0])
            try:
                await store.check_available()
            except Exception as exc:
                raise _InfraUnavailable(
                    f"attempt {number}'s MIRIX database cannot be restored: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            # The rollback renames and drops the database under the attempt's
            # name, so it must be the attempt's (read-only, before anything changes).
            # Another run's database, or one without a marker that nobody claimed,
            # waits for a person, whatever stopped the attempt.
            try:
                await store.check_owner(claim_unmarked=forced, segment=segment)
            except CellDatabaseConflict as exc:
                # A MIRIX server the dead invocation left still writes the database
                # and spends on the gateway run: stop this cell's own processes
                # (nothing of the database changes) before waiting for the person.
                await asyncio.to_thread(
                    cleanup_owned, self.paths.root / "owned_processes.jsonl"
                )
                return await self._learning_failed(
                    number,
                    _LearningFailed(
                        classify_learning_exception(exc),
                        stage="resume",
                        keep_cause=True,
                    ),
                )
            except Exception as exc:
                raise _InfraUnavailable(
                    f"attempt {number}'s MIRIX database owner cannot be read: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        # Stop what the dead invocation left running.
        await self._kill_leftovers(number, store)

        # Mark the resume in the ledger.
        try:
            await self.gateway.set_phase(run_id, "learning")
            answer = await self.gateway.mark(
                run_id,
                {
                    "event": LEDGER_EVENT_RESUME,
                    "segment": segment,
                    "checkpoint": checkpoint.cid if checkpoint else None,
                    "checkpoint_offset": checkpoint.offset if checkpoint else None,
                    "launch": self.launch,
                },
            )
            marker = await self._locate_marker(
                run_id, answer, LEDGER_EVENT_RESUME, segment=segment, launch=self.launch
            )
        except Exception as exc:
            raise _InfraUnavailable(
                f"gateway {self.gateway.base_url} did not mark the resume of {run_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            row = await self._discard_segment(
                number, state, checkpoint, segment, marker, store, forced=forced
            )
        except asyncio.CancelledError:
            raise  # the state is unchanged: the next invocation repeats the resume
        except _LearningFailed as exc:
            return await self._learning_failed(number, exc)
        except Exception as exc:
            return await self._learning_failed(
                number,
                _LearningFailed(classify_learning_exception(exc), stage="resume"),
            )
        if checkpoint is None:
            learning_from = {"position": 0, "stage": "setup", "checkpoint": None}
        elif checkpoint.kind == CHECKPOINT_PLAYED:
            learning_from = {
                "position": checkpoint.position,
                "stage": STAGE_AFTER_TASK,
                "checkpoint": checkpoint.cid,
            }
        else:
            learning_from = {
                "position": checkpoint.next_position,
                "stage": STAGE_BEFORE_TASK,
                "checkpoint": checkpoint.cid,
            }
        await self._record_work(
            number,
            "learning",
            why=f"learning resume of attempt {number}",
            learning_from=learning_from,
        )
        await self._write_manifest(number)
        await self._set_status(STATUS_RUNNING, attempt=number, phase="learning")
        log.warning(
            "%s: attempt %d resumes its learning phase from %s (segment %d discarded: "
            "%d call(s), $%.6f%s)%s",
            self.spec.cell_id,
            number,
            checkpoint.cid if checkpoint else "the empty state",
            segment,
            len(row["calls"]),
            row["cost_total"],
            # Their cost comes with their final lines, after this.
            f" + {len(row['pending_calls'])} still in flight"
            if row["pending_calls"]
            else "",
            f"; {notes}" if notes else "",
        )

        # How often in a row this checkpoint was restored after a lost MIRIX batch (this
        # resume included, its resumes row saved by _discard_segment): the method and the
        # ledger check accept the batch's failure once it reaches MIRIX_BATCH_ROLLBACKS.
        rollbacks = consecutive_rollbacks(
            (await asyncio.to_thread(self._read_attempt, number)).get("resumes"),
            checkpoint.cid if checkpoint else None,
            code=MIRIX_LEARNING,
        )

        def resumed(attempt: _Attempt) -> _Attempt:
            attempt.rollbacks = rollbacks
            attempt.rollback_checkpoint = checkpoint.cid if checkpoint else None
            attempt.after_resume = True
            return attempt

        async def open_resumed() -> _Attempt:
            if checkpoint is None:
                # Nothing was checkpointed: setup runs again on the empty state, in the
                # setup phase as in a new attempt.
                await self.gateway.set_phase(run_id, "setup")  # type: ignore[union-attr]
                return resumed(
                    await self._open_attempt(
                        number,
                        factory,
                        phase="learning",
                        mirix_mode=mode,
                        server=prepared,
                    )
                )
            resume = LearningResume(
                checkpoint_id=checkpoint.cid,
                kind=checkpoint.kind,  # type: ignore[arg-type]
                position=checkpoint.position,
                state=dict(checkpoint.record["method"]["state"]),
                snapshot_sha256=str(checkpoint.record["method"]["snapshot_sha256"]),
                rollbacks=rollbacks,
            )
            try:
                attempt = await self._open_attempt(
                    number,
                    factory,
                    phase="learning",
                    mirix_mode=mode,
                    resume=resume,
                    server=prepared,
                )
            except CheckpointMismatch as exc:
                raise _LearningFailed(
                    classify_learning_exception(exc), stage="resume"
                ) from exc
            try:
                await self._check_resumed_snapshot(attempt, checkpoint, segment)
            except BaseException:
                await self._close_attempt(attempt)
                raise
            return resumed(attempt)

        if checkpoint is None:
            start, played = 0, None
        elif checkpoint.kind == CHECKPOINT_PLAYED:
            start, played = checkpoint.position + 1, checkpoint
        else:
            start, played = checkpoint.next_position, None
        return await self._run_learning(
            number,
            open_resumed,
            init=checkpoint is None,
            start=start,
            played=played,
        )

    async def _resume_guards(
        self, number: int, state: Mapping[str, Any], run_id: str
    ) -> None:
        """Step 1 of a resume: the same host, the same checkpoint format, the same run on a
        gateway that takes it, and a person's decision for a harness failure (else
        refused). Other code, another commit or another protocol is not refused: the
        resume goes on under this tree's and records a new version segment
        (``_version_segment``)."""
        assert self.gateway is not None
        recorded_host = state.get("host")
        if not same_host(recorded_host, self.host):
            # The machine id, not the hostname, names a host when both records have
            # one: a Mac's hostname follows its network.
            where, here = host_text(recorded_host), host_text(self.host)
            raise _Refused(
                f"attempt {number} learned on {where}, this is {here}: its checkpoints "
                "(and a MIRIX cell's database and HOME) are local to that host. Resume it "
                "there, or pass --new-attempt to learn again from scratch"
            )
        layout = state.get("checkpoint_format")
        if layout != CHECKPOINT_FORMAT:
            commit = (state.get("git") or {}).get("commit")
            raise _Refused(
                f"attempt {number}'s checkpoints are in format {layout}, this run.py "
                f"reads format {CHECKPOINT_FORMAT}: check out commit {commit} to resume "
                "it, or pass --new-attempt to learn again from scratch"
            )
        if state.get("needs_decision") and not self.options.force_resume:
            raise _Refused(self._needs_decision_refusal(number, state))
        if await asyncio.to_thread(self._run_abandoned, run_id):
            raise _Refused(
                f"attempt {number}'s gateway run {run_id} was abandoned: it cannot be "
                "resumed; pass --new-attempt"
            )
        await self._check_ledger_link(run_id)
        recorded = state.get("ledger")
        live = self.gateway.ledger_path(run_id)
        if (
            isinstance(recorded, str)
            and recorded
            and Path(recorded) != live
            and await asyncio.to_thread(Path(recorded).is_file)
        ):
            raise _Refused(
                f"attempt {number}'s ledger is {recorded}, this gateway writes {live}: "
                "a resume must append to the same ledger (pass the --gateway-runs-dir "
                "of its gateway)"
            )
        agent_url = state.get("gateway_base_url")
        if self.spec.method == MIRIX_METHOD and agent_url:
            # The MIRIX agent calls the gateway it was set up with.
            if not same_gateway(str(agent_url), self.gateway.base_url):
                port = httpx.URL(str(agent_url)).port
                raise _Refused(
                    f"attempt {number}'s MIRIX agent calls the gateway at {agent_url} "
                    f"(its llm_config and embedding_config endpoints), not "
                    f"{self.gateway.base_url}: attach to that gateway (--gateway-url "
                    f"{agent_url}) or launch this cell's gateway on port {port} "
                    f"(--gateway-port {port})"
                )
            self._method_gateway_url = str(agent_url)

    @staticmethod
    def _needs_decision_refusal(number: int, state: Mapping[str, Any]) -> str:
        """Why a harness-stopped learning phase is not resumed without a person: the
        failure of the resume that stopped on a database owner conflict
        (``resume_conflict``, kept apart from the attempt's own cause), else the
        attempt's failure."""
        conflict = state.get("resume_conflict")
        failure = (
            (conflict.get("failure") if isinstance(conflict, Mapping) else None)
            or state.get("failure")
            or {}
        )
        if failure.get("code") == FAILURE_CHECKPOINT_MISMATCH:
            choices = (
                "the memory it rebuilds from its checkpoint does not match the "
                "checkpoint, so --force-resume would stop the same way: fix the code "
                "that rebuilds it, or pass --new-attempt to learn again from scratch"
            )
        else:
            choices = (
                "pass --force-resume to resume it from its checkpoint (an attempt "
                "without checkpoints restarts as a new attempt), or --new-attempt to "
                "learn again from scratch"
            )
        return (
            f"attempt {number}'s learning phase stopped on a harness failure "
            f"({failure.get('code')}: {failure.get('detail')}) and waits for a "
            f"person: {choices}"
        )

    def _run_abandoned(self, run_id: str) -> bool:
        assert self.gateway is not None
        path = self.gateway.ledger_path(run_id).parent / "gateway_state.json"
        try:
            data = read_json(path)
        except ValueError:
            return False
        return bool(isinstance(data, dict) and data.get("abandoned"))

    def _pick_checkpoint(
        self, paths: AttemptPaths, resume_from: Any
    ) -> tuple[_Checkpoint | None, str]:
        """Step 3 (blocking): LATEST if it verifies, else the newest kept checkpoint that
        does; an older one when the failure named it (``resume_from``); None when no
        checkpoint was ever written (the attempt stopped during setup). Raises
        :class:`_CheckpointCorrupt` when checkpoints exist and none verifies."""
        notes: list[str] = []
        root = paths.checkpoints
        dirs = _checkpoint_dirs(root)
        latest_record: Any = None
        try:
            latest_record = read_json(paths.latest_checkpoint)
        except ValueError:
            latest_record = {"corrupt": True}
        if resume_from == RESUME_FROM_START:
            return None, "the failure preceded every kept checkpoint"
        if not dirs and latest_record is None:
            return None, ""
        chosen: _Checkpoint | None = None
        if isinstance(latest_record, dict) and latest_record.get("id"):
            try:
                chosen = _load_checkpoint(
                    root / str(latest_record["id"]), str(latest_record.get("sha256"))
                )
            except (_CheckpointCorrupt, ValueError) as exc:
                notes.append(
                    f"LATEST ({latest_record.get('id')}) failed its check: {exc}"
                )
        if chosen is None:
            for _ordinal, cid, path in dirs:
                try:
                    chosen = _load_checkpoint(path)
                    notes.append(f"using the previous checkpoint {cid}")
                    break
                except (_CheckpointCorrupt, ValueError) as exc:
                    notes.append(f"{cid} failed its check: {exc}")
        if chosen is None:
            raise _CheckpointCorrupt(
                "no learning checkpoint verifies: " + "; ".join(notes)
            )
        if isinstance(resume_from, str) and resume_from and resume_from != chosen.cid:
            try:
                kind, position = parse_checkpoint_id(resume_from)
            except ValueError:
                kind, position = "", -1
            if kind and checkpoint_ordinal(kind, position) < chosen.ordinal:
                try:
                    chosen = _load_checkpoint(root / resume_from)
                except (_CheckpointCorrupt, ValueError) as exc:
                    raise _CheckpointCorrupt(
                        f"the checkpoint {resume_from} the failure needs failed its "
                        f"check: {exc}"
                    ) from exc
                notes.append(
                    f"restoring {resume_from}: the failure precedes the latest checkpoint"
                )
        return chosen, "; ".join(notes)

    async def _kill_leftovers(self, number: int, store: CheckpointStore | None) -> None:
        """Step 2: stop the dead invocation's MIRIX server, and make sure nothing uses the
        cell database any more (the fence that also keeps a zombie from calling the
        gateway under this run)."""
        await asyncio.to_thread(
            cleanup_owned, self.paths.root / "owned_processes.jsonl"
        )
        if store is None:
            return
        deadline = time.monotonic() + DB_RELEASE_S
        while True:
            try:
                connected = await store.connections()
            except Exception as exc:
                raise _InfraUnavailable(
                    f"cannot read the connections of {store.name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not connected:
                return
            if time.monotonic() >= deadline:
                raise _InfraUnavailable(
                    f"{connected} connection(s) still use the cell database {store.name} "
                    f"after {DB_RELEASE_S:.0f} s: a MIRIX server this harness cannot "
                    "prove dead may still write it (INFRA mirix_server); stop it, then "
                    "run the cell again"
                )
            await asyncio.sleep(0.5)

    async def _discard_segment(
        self,
        number: int,
        state: Mapping[str, Any],
        checkpoint: _Checkpoint | None,
        segment: int,
        marker: LedgerLine,
        store: CheckpointStore | None,
        *,
        forced: bool,
    ) -> dict[str, Any]:
        """Steps 5 to 7: the segment's calls, the move aside, the restore and its checks,
        the ``discarded.jsonl`` row and the attempt's new state."""
        assert self.gateway is not None
        paths = self.paths.attempt(number)
        ledger = self.gateway.ledger_path(str(state.get("run_id")))
        earlier = await asyncio.to_thread(discarded_call_ids, paths)
        lines = await asyncio.to_thread(read_ledger_lines, ledger)
        calls = segment_calls(
            lines,
            checkpoint_offset=checkpoint.offset if checkpoint else None,
            resume_offset=marker.offset,
            exclude=earlier,
        )
        moved = await asyncio.to_thread(self._move_aside, paths, checkpoint, segment)
        final = paths.discarded_dir(segment)
        work = final if final.is_dir() else final.with_name(final.name + ".partial")
        mirix = moved.get("mirix")
        if mirix is None and store is not None:
            summary = await store.rollback(
                segment,
                checkpoint.path if checkpoint else None,
                work / EXPORT_FILE,
                keep_discarded=self.options.keep_discarded_db,
                claim_unmarked=forced,
            )
            mirix = {
                "export": self._rel(final / EXPORT_FILE),
                **{k: v for k, v in summary.items()},
            }
            moved["mirix"] = mirix
        if not final.is_dir():

            def seal() -> None:
                _write_json_durable(work / "moved.json", moved)
                work.rename(final)
                _fsync_path(final.parent)

            await asyncio.to_thread(seal)

        # Step 7: restore and verify, before MIRIX starts.
        restore = await self._restore_checkpoint(paths, checkpoint, store)
        in_flight = self._in_flight(state, checkpoint, moved)
        failure = state.get("failure")
        interruption = state.get("interruption") or {}
        if state.get("state") == ATTEMPT_LEARNING:
            reason = "process_killed"
        elif interruption.get("reason"):
            reason = str(interruption["reason"])
        else:
            reason = str((failure or {}).get("code") or "unknown")
        cost = round(sum(float(c.get("cost_total") or 0.0) for c in calls), 10)
        unknown = sum(1 for c in calls if c.get("unknown_cost"))
        # The calls without a final line yet (a zombie, or a call a
        # gateway pause holds past the resume). ``cost_total`` is what the ledger held
        # at the resume and leaves them out; the row stays as written, and costs.json
        # (``discarded_segments``), report.py and summarize_learning read their cost
        # from the final line the gateway writes later.
        pending = [str(c["call_id"]) for c in calls if not c.get("final_line")]
        row = {
            "segment": segment,
            "attempt": number,
            "run_id": state.get("run_id"),
            "launch": self.launch,
            "at": now(),
            "from_state": state.get("state"),
            "failure": failure,
            "failure_stage": state.get("failure_stage"),
            "reason": reason,
            "operator_resume": forced,
            # A resume stopped for a person before this one (the database owner
            # check), recorded next to the cause this row discards.
            "resume_conflict": state.get("resume_conflict"),
            "checkpoint": {
                "id": checkpoint.cid if checkpoint else None,
                "offset": checkpoint.offset if checkpoint else None,
            },
            "resume_marker_offset": marker.offset,
            "in_flight": in_flight,
            "calls": calls,
            "cost_total": cost,
            "unknown_cost_calls": unknown,
            "pending_calls": pending,
            "moved": {k: v for k, v in moved.items() if k != "mirix"},
            "mirix": mirix,
            "restore": restore,
        }
        if not restore["files_equal"] or restore["db_digest_equal"] is False:
            await asyncio.to_thread(_append, paths.discarded_log, row)
            raise _LearningFailed(
                Failure(
                    FailureKind.HARNESS,
                    FAILURE_CHECKPOINT_MISMATCH,
                    None,
                    None,
                    f"checkpoint {checkpoint.cid if checkpoint else None} was not "
                    f"restored exactly: {restore.get('problems')}"[:500],
                ),
                stage="resume",
            )
        await asyncio.to_thread(_append, paths.discarded_log, row)
        await asyncio.to_thread(self._drop_newer_checkpoints, paths, checkpoint)
        fresh = await asyncio.to_thread(self._read_attempt, number)
        for key in (
            "failure",
            "failure_stage",
            "failure_position",
            "failure_at",
            "failure_launch",
            "resume_from",
            "needs_decision",
            "interruption",
            "resume_conflict",
        ):
            fresh.pop(key, None)
        position = checkpoint.position if checkpoint else 0
        fresh.update(
            state=ATTEMPT_LEARNING,
            learning={
                **(fresh.get("learning") or {}),
                "items": len(self.learning),
                "done": checkpoint.next_position if checkpoint else 0,
                "played": (
                    position + 1
                    if checkpoint and checkpoint.kind != CHECKPOINT_INIT
                    else 0
                ),
                "checkpoint": checkpoint.cid if checkpoint else None,
            },
            resumes=[
                *(fresh.get("resumes") or []),
                {
                    "segment": segment,
                    "checkpoint": checkpoint.cid if checkpoint else None,
                    "checkpoint_offset": checkpoint.offset if checkpoint else None,
                    "resume_marker_offset": marker.offset,
                    "at": now(),
                    "launch": self.launch,
                    "host": self.host,
                    "from_state": state.get("state"),
                    "reason": reason,
                    "failure": failure,
                    "failure_stage": state.get("failure_stage"),
                    "operator_resume": forced,
                    "discarded_calls": len(calls),
                    "cost": cost,
                    "unknown_cost_calls": unknown,
                    "pending_calls": len(pending),
                },
            ],
        )
        await self._save_attempt(number, fresh)
        return row

    def _move_aside(
        self, paths: AttemptPaths, checkpoint: _Checkpoint | None, segment: int
    ) -> dict[str, Any]:
        """Step 6 (blocking; safe to repeat): everything written after the checkpoint goes
        to ``discarded/seg-SS.partial/`` (sealed by :meth:`_discard_segment`): the method
        directory, the tails of ``learning.jsonl`` and ``injections.jsonl`` (each row
        pointing at its moved trajectory), the trajectories the checkpoint does not list
        and, for MIRIX, HOME's new files. A sealed ``seg-SS`` is only read back."""
        final = paths.discarded_dir(segment)
        if final.is_dir():
            data = read_json(final / "moved.json")
            if isinstance(data, dict):
                return data
            raise CheckpointMismatch(f"{final}/moved.json is missing or unreadable")
        work = final.with_name(final.name + ".partial")
        work.mkdir(parents=True, exist_ok=True)
        rel_final = self._rel(final)
        # 1. the method directory, whole
        if paths.method_dir.exists():
            target = work / "method"
            if target.exists():
                target = work / f"method.{uuid.uuid4().hex[:8]}"
            paths.method_dir.rename(target)
        # 2. trajectories the checkpoint does not list (failed tries included)
        kept = set(
            checkpoint.record["attempt_files"]["trajectories"] if checkpoint else ()
        )
        for rel in _tree_files(paths.trajectories_root):
            full = f"trajectories/{rel}"
            if full in kept:
                continue
            dest = work / "trajectories" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(paths.trajectories_root / rel, dest)
        # Keyed like the rows' ``trajectory``: relative to the cell directory.
        moved_trajectories = {
            self._rel(paths.trajectories_root / rel): f"{rel_final}/trajectories/{rel}"
            for rel in _tree_files(work / "trajectories")
        }
        # 3. the tails of the append-only records
        sizes = checkpoint.record["attempt_files"] if checkpoint else {}
        digests = (
            (checkpoint.record.get("attempt_files_sha256") or {}) if checkpoint else {}
        )
        by_uid: dict[str, str] = {}
        for old, new in moved_trajectories.items():
            name = old.rsplit("/", 1)[-1]
            uid = name[:-5] if name.endswith(".json") else name
            by_uid.setdefault(uid.split("-", 1)[0] if "/failed/" in old else uid, new)

        def learning_row(row: dict[str, Any]) -> dict[str, Any]:
            return {
                **row,
                "discarded_segment": segment,
                "trajectory_discarded": moved_trajectories.get(
                    str(row.get("trajectory"))
                ),
            }

        def injection_row(row: dict[str, Any]) -> dict[str, Any]:
            return {
                **row,
                "discarded_segment": segment,
                "trajectory_discarded": by_uid.get(str(row.get("task_uid"))),
            }

        _truncate_jsonl(
            paths.learning,
            int(sizes.get("learning.jsonl") or 0),
            work / "learning_rows.jsonl",
            expected_sha256=digests.get("learning.jsonl"),
            annotate=learning_row,
        )
        _truncate_jsonl(
            paths.injections,
            int(sizes.get("injections.jsonl") or 0),
            work / "injections.jsonl",
            expected_sha256=digests.get("injections.jsonl"),
            annotate=injection_row,
        )
        # 4. MIRIX's HOME: what the checkpoint's manifest does not list
        home_moved: list[str] = []
        if self.spec.method == MIRIX_METHOD:
            manifest = None
            if checkpoint is not None:
                manifest = read_json(checkpoint.path / HOME_MANIFEST_FILE)
                if not isinstance(manifest, dict):
                    raise CheckpointMismatch(
                        f"{checkpoint.path}/{HOME_MANIFEST_FILE} is missing"
                    )
            home_moved = move_new_home_files(paths.home, manifest, work / "home")
        rows = read_jsonl(work / "learning_rows.jsonl")
        return {
            "method": f"{rel_final}/method",
            "learning_rows": len(rows),
            "learning_positions": [r.get("position") for r in rows],
            "injections": len(read_jsonl(work / "injections.jsonl")),
            "trajectories": sorted(moved_trajectories.values()),
            "home_files": len(_tree_files(work / "home")),
            "home_moved_now": len(home_moved),
            "mirix": None,
        }

    async def _restore_checkpoint(
        self,
        paths: AttemptPaths,
        checkpoint: _Checkpoint | None,
        store: CheckpointStore | None,
    ) -> dict[str, Any]:
        """Step 7: the checkpoint's method directory and HOME back in place, checked
        against its hashes, and the restored database against its digest (MIRIX is not
        running). Returns the ``restore`` record (the snapshot is checked after setup)."""
        problems: list[str] = []

        def files() -> bool:
            if checkpoint is None:
                if paths.method_dir.exists():
                    shutil.rmtree(paths.method_dir)
                paths.method_dir.mkdir(parents=True)
                return True
            restored = _restore_method_dir(checkpoint.path / "method", paths.method_dir)
            expected = dict(checkpoint.record["method"]["files"])
            if restored != expected:
                differ = sorted(
                    k
                    for k in set(restored) | set(expected)
                    if restored.get(k) != expected.get(k)
                )
                problems.append(f"method files differ: {differ[:10]}")
            if self.spec.method == MIRIX_METHOD:
                manifest = read_json(checkpoint.path / HOME_MANIFEST_FILE) or {}
                problems.extend(restore_home(paths.home, checkpoint.path, manifest))
            return not problems

        files_equal = await asyncio.to_thread(files)
        db_equal: bool | None = None
        if store is not None and checkpoint is not None:
            expected = (checkpoint.record.get("mirix") or {}).get("db_digest")
            actual = await store.digest()
            db_equal = json.loads(json.dumps(actual)) == json.loads(
                json.dumps(expected)
            )
            if not db_equal:
                problems.append("the restored database's digest differs")
        return {
            "snapshot_sha256_expected": (
                checkpoint.record["method"]["snapshot_sha256"] if checkpoint else None
            ),
            "snapshot_sha256_actual": None,  # set once the method has rebuilt it
            "db_digest_equal": db_equal,
            "files_equal": files_equal,
            "problems": problems,
        }

    def _in_flight(
        self,
        state: Mapping[str, Any],
        checkpoint: _Checkpoint | None,
        moved: Mapping[str, Any],
    ) -> dict[str, Any]:
        """What the discarded segment was doing: the item after the checkpoint, and the
        stage (the recorded failure's, else inferred from what was moved aside)."""
        n = len(self.learning)
        position = checkpoint.next_position if checkpoint else 0
        uid = self.learning[position].uid(self.salt) if position < n else None
        stage = state.get("failure_stage")
        if not stage:
            if checkpoint is None and not moved.get("learning_rows"):
                stage = "setup"
            elif position >= n:
                stage = "freeze"
            elif checkpoint is not None and checkpoint.kind == CHECKPOINT_PLAYED:
                stage = STAGE_AFTER_TASK
            elif position in (moved.get("learning_positions") or ()):
                stage = STAGE_PLAYED
            else:
                stage = "rollout"
        recorded = bool(uid) and (
            (checkpoint is not None and checkpoint.kind == CHECKPOINT_PLAYED)
            or any(uid in str(t) for t in moved.get("trajectories") or ())
        )
        return {
            "position": position if position < n else None,
            "task_uid": uid,
            "stage": stage,
            "trajectory_recorded": recorded,
        }

    @staticmethod
    def _drop_newer_checkpoints(
        paths: AttemptPaths, checkpoint: _Checkpoint | None
    ) -> None:
        """After a restore to ``checkpoint``, it is LATEST again: newer checkpoints hold
        what was discarded (blocking)."""
        root = paths.checkpoints
        top = checkpoint.ordinal if checkpoint else -1
        for ordinal, _cid, path in _checkpoint_dirs(root):
            if ordinal > top:
                shutil.rmtree(path, ignore_errors=True)
        for child in root.iterdir():
            if child.name.endswith(".partial") and child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        if checkpoint is None:
            paths.latest_checkpoint.unlink(missing_ok=True)
        else:
            data = (checkpoint.path / CHECKPOINT_RECORD).read_bytes()
            write_json_atomic(
                paths.latest_checkpoint,
                {
                    "id": checkpoint.cid,
                    "ordinal": checkpoint.ordinal,
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
        _fsync_path(root)

    async def _check_resumed_snapshot(
        self, attempt: _Attempt, checkpoint: _Checkpoint, segment: int
    ) -> None:
        """Step 10: the snapshot of the resumed method must hash to the checkpoint's."""
        snapshot = await asyncio.wait_for(attempt.method.snapshot(), SNAPSHOT_TIMEOUT_S)
        record = await asyncio.to_thread(snapshot_record, snapshot)
        expected = str(checkpoint.record["method"]["snapshot_sha256"])
        equal = record["sha256"] == expected
        count = attempt.counter.get(SNAPSHOT_CHECK_LEARNING_RESUME, 0) + 1
        attempt.counter[SNAPSHOT_CHECK_LEARNING_RESUME] = count
        name = f"{SNAPSHOT_CHECK_LEARNING_RESUME}-{self.launch}-{count}"
        await asyncio.to_thread(
            write_json_atomic,
            attempt.paths.snapshot(name),
            {**record, "at": now(), "expected_sha256": expected, "equal": equal},
        )
        state = await asyncio.to_thread(self._read_attempt, attempt.number)
        checks = [
            *state.get("snapshot_checks", []),
            {
                "kind": SNAPSHOT_CHECK_LEARNING_RESUME,
                "sha256": record["sha256"],
                "expected_sha256": expected,
                "equal": equal,
                "checkpoint": checkpoint.cid,
                "segment": segment,
                "path": self._rel(attempt.paths.snapshot(name)),
                "launch": self.launch,
                "at": now(),
            },
        ]
        await self._update_attempt(attempt.number, snapshot_checks=checks)
        await asyncio.to_thread(
            self._set_restored_snapshot, attempt.paths, segment, record["sha256"]
        )
        if not equal:
            raise _LearningFailed(
                Failure(
                    FailureKind.HARNESS,
                    FAILURE_CHECKPOINT_MISMATCH,
                    None,
                    None,
                    f"the memory resumed from {checkpoint.cid} hashes to "
                    f"{record['sha256']}, not to the recorded {expected}",
                ),
                stage="resume",
            )

    @staticmethod
    def _set_restored_snapshot(paths: AttemptPaths, segment: int, sha256: str) -> None:
        """Fill ``restore.snapshot_sha256_actual`` of segment ``segment``'s row once the
        method rebuilt its snapshot (the file rewritten atomically; blocking)."""
        rows = read_jsonl(paths.discarded_log)
        for row in reversed(rows):
            if row.get("segment") == segment:
                row.setdefault("restore", {})["snapshot_sha256_actual"] = sha256
                break
        else:
            return
        _write_jsonl_atomic(paths.discarded_log, rows)

    # -- test (H1.6) ------------------------------------------------------------------------

    async def _resume_test(
        self, number: int, state: Mapping[str, Any], factory: MethodFactory
    ) -> CellOutcome:
        """Restart the test phase of a frozen attempt: resume the memory, check it, go on.
        Under another commit, code or protocol it goes on and records a new version
        segment (from which test count, what changed): the frozen
        memory itself must still hash to its record and equal the frozen snapshot once
        resumed (H10.4), which no option overrides."""
        paths = self.paths.attempt(number)
        frozen = await asyncio.to_thread(read_json, paths.frozen_snapshot)
        if not isinstance(frozen, dict) or "sha256" not in frozen:
            raise _Refused(f"{paths.frozen_snapshot} is missing or unreadable")
        recorded = (state.get("frozen") or {}).get("sha256")
        actual = await asyncio.to_thread(canonical_sha256, frozen.get("snapshot"))
        if not (recorded == frozen["sha256"] == actual):
            raise _Refused(
                f"{paths.frozen_snapshot} does not match the sha256 recorded when the "
                f"memory was frozen ({recorded}); the frozen record is corrupt"
            )
        assert self.gateway is not None
        agent_url = state.get("gateway_base_url")
        if self.spec.method == MIRIX_METHOD and agent_url:
            # The frozen MIRIX agent calls the gateway it was set up with; a test
            # phase on another gateway would fail its retrieval calls (or, worse, let
            # MIRIX fall back to retrieving without topic extraction).
            if not same_gateway(str(agent_url), self.gateway.base_url):
                port = httpx.URL(str(agent_url)).port
                raise _Refused(
                    f"attempt {number}'s MIRIX agent calls the gateway at {agent_url} "
                    f"(its llm_config and embedding_config endpoints), not "
                    f"{self.gateway.base_url}: attach to that gateway (--gateway-url "
                    f"{agent_url}) or launch this cell's gateway on port {port} "
                    f"(--gateway-port {port})"
                )
            self._method_gateway_url = str(agent_url)
        run_id = str(state.get("run_id"))
        try:
            # Every call of the resumed attempt, MIRIX's own at its restart too,
            # belongs to the test phase.
            await self.gateway.set_phase(run_id, "test")
        except Exception as exc:
            raise _InfraUnavailable(
                f"gateway {self.gateway.base_url} refused the phase of {run_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        rows = await asyncio.to_thread(read_jsonl, paths.results)
        await self._record_work(
            number,
            "test",
            why=f"test resume of attempt {number}",
            test_from={
                "scored": len(
                    {str(r.get("raw_id")) for r in rows if r.get("score") is not None}
                ),
                "pending": len(await self._pending(paths)),
            },
        )
        await self._write_manifest(number)
        await self._set_status(STATUS_RUNNING, attempt=number, phase="test")
        attempt: _Attempt | None = None
        try:
            try:
                attempt = await self._open_attempt(
                    number, factory, phase="test", mirix_mode="resume"
                )
                resumed = await asyncio.wait_for(
                    attempt.method.snapshot(), SNAPSHOT_TIMEOUT_S
                )
                record = await asyncio.to_thread(snapshot_record, resumed)
            except (asyncio.CancelledError, _Refused):
                raise
            except FrozenMemoryMismatch as exc:
                # The method found its frozen memory missing or changed: an H10.4
                # violation of the attempt, never a pending item.
                return await self._violation(
                    number,
                    "the frozen memory could not be restored for the test phase "
                    f"({type(exc).__name__}: {exc})"[:1000],
                )
            except Exception as exc:
                failure = classify_exception(exc)
                await asyncio.to_thread(
                    FailureLog(paths.failures).record,
                    "test",
                    failure,
                    item=None,
                    attempt=number,
                )
                pending = await self._pending(paths)
                return CellOutcome(
                    self.spec.cell_id,
                    STATUS_TEST_INCOMPLETE,
                    number,
                    f"the frozen memory could not be resumed ({failure.kind.value}/"
                    f"{failure.code}: {failure.detail})"[:1000],
                    pending=len(pending),
                )
            equal = await self._check_snapshot(attempt, "resume", record, frozen)
            if not equal:
                return await self._violation(
                    number,
                    "the memory resumed for the test phase differs from the frozen one",
                )
            return await self._test_phase(attempt, frozen)
        finally:
            await self._close_attempt(attempt)

    async def _pending(self, paths: AttemptPaths) -> list[str]:
        rows = await asyncio.to_thread(read_jsonl, paths.results)
        return pending_test_items([t.raw_id for t in self.test], rows)

    async def _test_phase(
        self, attempt: _Attempt, frozen: Mapping[str, Any]
    ) -> CellOutcome:
        assert self.gateway is not None
        number = attempt.number
        await self.gateway.set_phase(attempt.run_id, "test")
        await self._set_status(STATUS_RUNNING, attempt=number, phase="test")
        state = await self._update_attempt(number, state=ATTEMPT_TESTING)
        pending_ids = await self._pending(attempt.paths)
        reruns = await self._reruns(attempt, pending_ids)
        if reruns:
            state = await self._update_attempt(
                number, test_reruns=[*state.get("test_reruns", []), *reruns]
            )
        by_id = {t.raw_id: (i, t) for i, t in enumerate(self.test)}
        # Items whose last try the provider refused "no endpoints": this try is scored
        # unanswered if it is refused again (PROVIDER_UNSERVED_TRIES).
        unserved = unserved_retries(
            await asyncio.to_thread(read_jsonl, attempt.paths.results)
        )
        try:
            await self._run_test_items(
                attempt, [by_id[r] for r in pending_ids], unserved=unserved
            )
        except asyncio.CancelledError:
            # The items this launch scored ran on the in-memory state that serves
            # them; compare it with the frozen memory before the process goes.
            await asyncio.shield(self._snapshot_after_cancel(attempt, frozen))
            raise

        await self._write_costs(number)
        try:
            after = await asyncio.wait_for(
                attempt.method.snapshot(), SNAPSHOT_TIMEOUT_S
            )
            record = await asyncio.to_thread(snapshot_record, after)
        except Exception as exc:
            failure = classify_exception(exc)
            await asyncio.to_thread(
                attempt.failures.record, "test", failure, item=None, attempt=number
            )
            remaining = await self._pending(attempt.paths)
            await self._write_manifest(number)
            return CellOutcome(
                self.spec.cell_id,
                STATUS_TEST_INCOMPLETE,
                number,
                "the snapshot after the test phase failed, so H10.4 is unchecked "
                f"({failure.kind.value}/{failure.code}: {failure.detail})"[:1000],
                pending=len(remaining),
            )
        equal = await self._check_snapshot(attempt, "after_test", record, frozen)
        if not equal:
            return await self._violation(
                number,
                "the memory changed during the test phase (H10.4: the snapshot after "
                "the test phase differs from the frozen one)",
            )
        remaining = await self._pending(attempt.paths)
        await self._update_attempt(
            number,
            test={
                "items": len(self.test),
                "scored": len(self.test) - len(remaining),
                "pending": remaining,
            },
        )
        if remaining:
            await self._write_manifest(number)
            return CellOutcome(
                self.spec.cell_id,
                STATUS_TEST_INCOMPLETE,
                number,
                f"{len(remaining)} test item(s) failed on infrastructure and wait for a "
                "re-run (H8)",
                pending=len(remaining),
            )
        await self._complete(attempt)
        return CellOutcome(self.spec.cell_id, STATUS_COMPLETE, number)

    async def _snapshot_after_cancel(
        self, attempt: _Attempt, frozen: Mapping[str, Any]
    ) -> None:
        """The after-test snapshot check of an interrupted test phase (bounded)."""
        try:
            after = await asyncio.wait_for(
                attempt.method.snapshot(), CANCEL_SNAPSHOT_TIMEOUT_S
            )
            record = await asyncio.to_thread(snapshot_record, after)
            equal = await self._check_snapshot(
                attempt, "after_test", record, frozen, interrupted=True
            )
            if not equal:
                await self._violation(
                    attempt.number,
                    "the memory changed during the test phase (H10.4: the snapshot "
                    "taken when the test phase was interrupted differs from the frozen "
                    "one)",
                )
        except Exception as exc:  # noqa: BLE001 - the next resume check covers it
            log.warning(
                "%s: no after-test snapshot at interruption (%s: %s); the next resume "
                "checks the stored memory",
                self.spec.cell_id,
                type(exc).__name__,
                exc,
            )

    async def _reruns(
        self, attempt: _Attempt, pending_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Pending items that were attempted before (a failed row or test calls in the ledger):
        the H8 test-phase re-runs the audit allows actor ``repeat_of`` on (H10.3)."""
        if not pending_ids:
            return []
        rows = await asyncio.to_thread(read_jsonl, attempt.paths.results)
        failed = {
            str(r.get("raw_id")): r.get("failure")
            for r in rows
            if r.get("score") is None
        }
        assert self.gateway is not None
        ledger = await asyncio.to_thread(
            read_jsonl, self.gateway.ledger_path(attempt.run_id)
        )
        touched = {
            str(r.get("item"))
            for r in ledger
            if r.get("event") in ("call", "call_started")
            and r.get("role") == "actor"
            and r.get("phase") == "test"
        }
        by_id = {t.raw_id: t for t in self.test}
        reruns = []
        for raw_id in pending_ids:
            uid = by_id[raw_id].uid(self.salt)
            if raw_id in failed or uid in touched:
                reruns.append(
                    {
                        "raw_id": raw_id,
                        "task_uid": uid,
                        "previous_failure": failed.get(raw_id),
                        "launch": self.launch,
                        "at": now(),
                    }
                )
        return reruns

    async def _run_test_items(
        self,
        attempt: _Attempt,
        items: Sequence[tuple[int, Task]],
        *,
        unserved: frozenset[str] = frozenset(),
    ) -> None:
        """Test items concurrently; after an infra or harness failure, stop taking new items
        if the cell's infrastructure is down (they stay pending for the next invocation).
        ``unserved``: the raw ids whose try is scored unanswered when the provider refuses
        it "no endpoints" again."""
        queue = deque(items)
        stop = asyncio.Event()

        async def worker() -> None:
            while queue and not stop.is_set():
                position, task = queue.popleft()
                failure = await self._test_item(
                    attempt, position, task, unserved=task.raw_id in unserved
                )
                if failure is not None and not await self._infra_ok(attempt):
                    stop.set()

        count = max(1, min(self.options.test_concurrency, len(items)))
        await _run_all([worker for _ in range(count)])

    async def _infra_ok(self, attempt: _Attempt) -> bool:
        assert self.gateway is not None
        try:
            health = await self.gateway.health()
        except Exception:
            return False
        if health.get("paused"):
            return False
        if attempt.server is not None:
            try:
                await attempt.server.ensure_alive()
            except Exception:
                return False
        return True

    async def _test_item(
        self, attempt: _Attempt, position: int, task: Task, *, unserved: bool = False
    ) -> Failure | None:
        assert self.env is not None
        try:
            run: EpisodeRun = await run_episode(
                self.env,
                task,
                method=attempt.method,
                actor=attempt.actor,
                salt=self.salt,
                phase="test",
                unserved=(REFUSAL_NO_ENDPOINT,) if unserved else (),
            )
        except RolloutError as err:
            await self._record(
                attempt,
                "test",
                position,
                task,
                err.trajectory,
                err.result,
                failure=err.failure,
                stage=err.stage,
            )
            await asyncio.to_thread(
                attempt.failures.record,
                "test",
                err.failure,
                item=err.trajectory.task_uid,
                attempt=attempt.number,
            )
            return err.failure
        await self._record(attempt, "test", position, task, run.trajectory, run.result)
        if run.trajectory.end_reason == END_PROVIDER_UNSERVED and isinstance(
            run.trajectory.failure, Mapping
        ):
            # The second consecutive no-endpoint refusal: scored unanswered, recorded.
            await asyncio.to_thread(
                attempt.failures.record,
                "test",
                Failure.from_dict(run.trajectory.failure),
                item=run.trajectory.task_uid,
                attempt=attempt.number,
            )
        return None

    async def _check_snapshot(
        self,
        attempt: _Attempt,
        kind: str,
        record: Mapping[str, Any],
        frozen: Mapping[str, Any],
        **extra: Any,
    ) -> bool:
        equal = record["sha256"] == frozen["sha256"]
        count = attempt.counter.get(kind, 0) + 1
        attempt.counter[kind] = count
        name = f"{kind}-{self.launch}-{count}"
        await asyncio.to_thread(
            write_json_atomic,
            attempt.paths.snapshot(name),
            {**record, "at": now(), "equal_to_frozen": equal},
        )
        state = await asyncio.to_thread(self._read_attempt, attempt.number)
        checks = [
            *state.get("snapshot_checks", []),
            {
                "kind": kind,
                "sha256": record["sha256"],
                "frozen_sha256": frozen["sha256"],
                "equal": equal,
                "path": self._rel(attempt.paths.snapshot(name)),
                "launch": self.launch,
                "at": now(),
                **extra,
            },
        ]
        await self._update_attempt(attempt.number, snapshot_checks=checks)
        return equal

    async def _violation(self, number: int, detail: str) -> CellOutcome:
        await self._update_attempt(
            number,
            state=ATTEMPT_VIOLATION,
            violation={"detail": detail, "at": now(), "launch": self.launch},
        )
        await self._write_manifest(number)
        return CellOutcome(self.spec.cell_id, STATUS_VIOLATION, number, detail)

    # -- records -------------------------------------------------------------------------------

    async def _record(
        self,
        attempt: _Attempt,
        phase: str,
        position: int,
        task: Task,
        trajectory: Trajectory,
        result: EpisodeResult,
        *,
        failure: Failure | None = None,
        stage: str | None = None,
    ) -> None:
        """Write one item's trajectory, injection row and result row (scorer side)."""
        uid = trajectory.task_uid
        directory = attempt.paths.trajectories(phase)
        if failure is None:
            path = directory / f"{uid}.json"
        else:
            key = f"failed-{uid}"
            attempt.counter[key] = attempt.counter.get(key, 0) + 1
            path = (
                directory
                / "failed"
                / f"{uid}-{self.launch}-{attempt.counter[key]}.json"
            )
        await asyncio.to_thread(
            write_trajectory, path, trajectory, blob_dir=self.options.blobs
        )
        injection = trajectory.injection
        # A MIRIX retrieval injected although part of its input was lost: flagged here,
        # never model-visible.
        degraded = injection is not None and bool(injection.degraded)
        if injection is not None:
            await attempt.injections.append(
                {
                    "phase": phase,
                    "position": position,
                    "raw_id": task.raw_id,
                    "task_uid": uid,
                    "tokens": injection.tokens,
                    "encoding": trajectory.injection_encoding,
                    "item_ids": list(injection.item_ids),
                    "chars": len(injection.text),
                    "sha256": hashlib.sha256(
                        injection.text.encode("utf-8")
                    ).hexdigest(),
                    "text": injection.text,
                    "actor_calls": len(trajectory.calls),
                    "injected_tokens_total": trajectory.injected_tokens_total,
                    "launch": self.launch,
                    "at": now(),
                    **({"degraded": True} if degraded else {}),
                }
            )
        row = result.to_dict()
        score = row.get("score")
        row.update(
            cell_id=self.spec.cell_id,
            attempt=attempt.number,
            run_id=attempt.run_id,
            phase=phase,
            position=position,
            task_type=task.raw.get("task_type")
            if self.spec.benchmark == "alfworld"
            else None,
            primary=score.get("primary") if isinstance(score, dict) else None,
            injected_tokens_total=trajectory.injected_tokens_total,
            actor_calls=len(trajectory.calls),
            gateway_call_ids=[
                getattr(call, "gateway_call_id", None) for call in trajectory.calls
            ],
            end_reason=trajectory.end_reason,
            timings=dict(trajectory.timings),
            trajectory=self._rel(path),
            launch=self.launch,
            at=now(),
        )
        # Flags present only when true (recorded, not refused).
        if degraded:
            row["retrieval_degraded"] = True
        if trajectory.end_reason == END_PROVIDER_UNSERVED:
            row["provider_unserved"] = True
        if failure is not None:
            # H8: an item with an infra or harness failure is never scored, even if its
            # score was computed before the failure (it is kept for forensics only).
            row["unused_score"] = score
            row["score"] = None
            row["primary"] = None
            row["failure"] = failure.to_dict()
            row["failure_stage"] = stage
            row["recovery"] = (
                learning_recovery(failure)
                if phase == "learning" and attempt.paths.checkpoints.is_dir()
                else recovery_for(phase, failure).value
            )
        writer = attempt.learning_rows if phase == "learning" else attempt.result_rows
        await writer.append(row)

    async def _write_costs(self, number: int) -> dict[str, Any]:
        """The attempt's ledger summary (H9 costs by role and phase, the providers) with
        the link check: every actor call of every trajectory is in the ledger. Written for
        failed attempts too: their spend is part of the experiment's cost."""
        assert self.gateway is not None
        paths = self.paths.attempt(number)
        run_id = await asyncio.to_thread(self._attempt_run_id, number)
        discarded = await asyncio.to_thread(discarded_call_ids, paths)
        ledger = self.gateway.ledger_path(run_id)
        summary = await asyncio.to_thread(summarize_ledger, ledger, discarded=discarded)
        # Each discarded segment's cost by the final lines now, beside
        # ``discarded`` (their sum): a call in flight at its resume is in it once the
        # gateway wrote its final line, which discarded.jsonl's row cannot show.
        segments = await asyncio.to_thread(read_discarded, paths)
        summary["discarded_segments"] = (
            discarded_segment_costs(
                segments, await asyncio.to_thread(read_jsonl, ledger)
            )
            if segments
            else []
        )
        rows = [
            *await asyncio.to_thread(read_jsonl, paths.learning),
            *await asyncio.to_thread(read_jsonl, paths.results),
        ]
        in_ledger = set(summary["call_ids"])
        referenced = [
            str(call_id)
            for row in rows
            for call_id in row.get("gateway_call_ids") or ()
            if call_id
        ]
        summary["linkage"] = {
            "actor_calls_referenced": len(referenced),
            "missing_from_ledger": sorted(set(referenced) - in_ledger),
        }
        summary["attempt"] = number
        summary["run_id"] = run_id
        summary["at"] = now()
        await asyncio.to_thread(write_json_atomic, paths.costs, summary)
        return summary

    async def _complete(self, attempt: _Attempt) -> None:
        """Every test item scored and the memory unchanged: write the cell's final records,
        then mark the attempt complete (in that order)."""
        await self._write_cell_records(attempt.number)
        await self._update_attempt(
            attempt.number, state=ATTEMPT_COMPLETE, completed_at=now()
        )
        await self._write_manifest(attempt.number)

    async def _ensure_cell_records(self, number: int) -> None:
        """A complete attempt whose cell-level records are missing gets them rewritten
        from its own records."""
        paths = self.paths
        files = (paths.results, paths.learning, paths.summary)
        if not all(
            await asyncio.gather(*(asyncio.to_thread(f.is_file) for f in files))
        ):
            log.warning(
                "%s: attempt %d is complete but the cell records are missing; "
                "rewriting them",
                self.spec.cell_id,
                number,
            )
            await self._write_cell_records(number)

    async def _write_cell_records(self, number: int) -> None:
        """The cell's ``results.jsonl`` (the last scored row of each test item, in test
        order), ``learning.jsonl`` and ``summary.json``, from attempt ``number``."""
        paths = self.paths.attempt(number)
        state = await asyncio.to_thread(self._read_attempt, number)
        rows = await asyncio.to_thread(read_jsonl, paths.results)
        latest = {str(r.get("raw_id")): r for r in rows if r.get("score") is not None}
        missing = [t.raw_id for t in self.test if t.raw_id not in latest]
        if missing:
            raise _Refused(
                f"attempt {number} has no scored result for {len(missing)} test "
                f"item(s) ({', '.join(missing[:5])}): its cell records cannot be written"
            )
        final = [latest[t.raw_id] for t in self.test]
        learning = await asyncio.to_thread(read_jsonl, paths.learning)
        costs = await asyncio.to_thread(read_json, paths.costs) or {}
        lines = await asyncio.to_thread(read_jsonl, self.paths.launches)
        segments = await asyncio.to_thread(
            protocol.attempt_versions, self.manifest, state, number, lines
        )
        await asyncio.to_thread(_write_jsonl_atomic, self.paths.results, final)
        await asyncio.to_thread(_write_jsonl_atomic, self.paths.learning, learning)

        def mean(values: Sequence[float]) -> float | None:
            return round(sum(values) / len(values), 6) if values else None

        metric_names = sorted(
            {
                name
                for row in final
                for name in ((row.get("score") or {}).get("metrics") or {})
            }
        )
        summary = {
            **self.spec.to_dict(),
            "attempt": number,
            "run_id": state.get("run_id"),
            # The attempt's own: a complete attempt of an earlier version keeps it
            # after the tree moved on.
            "protocol_fingerprint": protocol.attempt_protocol(
                self.manifest, state, number
            )["fingerprint"],
            "code": state.get("code"),
            "test": {
                "items": len(final),
                "primary_mean": mean([float(r["primary"]) for r in final]),
                "metrics_mean": {
                    name: mean(
                        [
                            float((r["score"].get("metrics") or {}).get(name, 0.0))
                            for r in final
                        ]
                    )
                    for name in metric_names
                },
                "reruns": len(state.get("test_reruns", [])),
                "injected_tokens_mean": mean(
                    [float(r.get("injected_tokens") or 0) for r in final]
                ),
            },
            "learning": {
                "items": len(learning),
                "primary_mean": mean(
                    [
                        float(r["primary"])
                        for r in learning
                        if r.get("primary") is not None
                    ]
                ),
            },
            "frozen_snapshot_sha256": (state.get("frozen") or {}).get("sha256"),
            # Every version the attempt ran under: a "mixed version" cell changed its
            # protocol or code within the attempt.
            "versions": {
                "segments": len(segments),
                "mixed": state.get("versions_mixed")
                or {"protocol": False, "code": False, "commit": False},
                "protocol_fingerprints": list(
                    dict.fromkeys(
                        (s.get("protocol") or {}).get("fingerprint")
                        for s in segments
                        if (s.get("protocol") or {}).get("fingerprint")
                    )
                ),
            },
            "costs": costs.get("totals"),
            "cost_by_phase_role": costs.get("by_phase_role"),
            "providers": costs.get("providers"),
            "completed_at": now(),
        }
        await asyncio.to_thread(write_json_atomic, self.paths.summary, summary)

    async def _write_manifest(self, current: int | None) -> None:
        numbers = await asyncio.to_thread(self.paths.attempt_numbers)
        attempts = []
        for n in numbers:
            state = await asyncio.to_thread(self._read_attempt, n)
            spent = await asyncio.to_thread(read_json, self.paths.attempt(n).costs)
            attempts.append(
                {
                    key: state.get(key)
                    for key in (
                        "attempt",
                        "run_id",
                        "state",
                        "started_at",
                        "invalid_reason",
                        "invalidation",
                        "previous_state",
                        "abandoned",
                        "ledger",
                        "ledger_rel",
                        "git",
                        "host",
                        "needs_decision",
                    )
                }
                | {
                    # Each attempt's versions: the model-visible code it started with
                    # and the protocol it ran under.
                    "code": state.get("code"),
                    "protocol": protocol.attempt_protocol(self.manifest, state, n),
                    "version_segments": len(state.get("version_segments") or ()),
                    "versions_mixed": state.get("versions_mixed"),
                    "note": (state.get("note") or {}).get("text")
                    if isinstance(state.get("note"), Mapping)
                    else None,
                    "supersedes": state.get("supersedes"),
                    "superseded": state.get("superseded"),
                    "resumes": len(state.get("resumes") or ()),
                    "learning_checkpoint": (state.get("learning") or {}).get(
                        "checkpoint"
                    ),
                    "frozen_sha256": (state.get("frozen") or {}).get("sha256"),
                    "cost_total": (spent or {}).get("totals", {}).get("cost_total")
                    if isinstance(spent, dict)
                    else None,
                    # Beside it, the calls cost_total cannot count (cut off after
                    # their request was sent: billed, cost unknown).
                    "unknown_cost_calls": (spent or {})
                    .get("totals", {})
                    .get("unknown_cost_calls")
                    if isinstance(spent, dict)
                    else None,
                }
            )
        costs = None
        if current is not None:
            costs = await asyncio.to_thread(
                read_json, self.paths.attempt(current).costs
            )
        self.manifest.update(
            updated_at=now(),
            git=self.git_now,
            options=self.options.to_dict(),
            attempts=attempts,
            current_attempt=current,
            blob_dir=str(self.options.blobs),
            last_host=self.host,
        )
        if isinstance(costs, dict):
            self.manifest["providers"] = costs.get("providers")
            self.manifest["model_served"] = costs.get("model_served")
            self.manifest["costs"] = {
                "totals": costs.get("totals"),
                "by_phase_role": costs.get("by_phase_role"),
            }
        await asyncio.to_thread(write_json_atomic, self.paths.manifest, self.manifest)


def _append(path: Path, record: Mapping[str, Any]) -> None:
    from evals.skill_bench.infra.processes import append_jsonl

    append_jsonl(path, record)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.run",
        description="Run one skill_bench cell (H1, H8): learning, freeze, test.",
    )
    parser.add_argument("--benchmark", required=True, choices=protocol.BENCHMARKS)
    parser.add_argument(
        "--model",
        required=True,
        choices=[*protocol.MODELS, *protocol.CHAT_MODELS],
        help=f"{', '.join(protocol.MODELS)}, or the model id",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=study.method_argument,
        choices=protocol.METHODS,
        help="a method the study removed is refused, with the reason it was removed",
    )
    parser.add_argument("--split-mode", default="protocol", choices=SPLIT_MODES)
    parser.add_argument(
        "--n-train",
        type=int,
        help="smoke: the first N items of the seed-42 train order",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        help="smoke: the first N items of the seed-42 val order (never the test split)",
    )
    parser.add_argument(
        "--out", type=Path, help="cell directory (default runs/<cell_id>)"
    )
    gateway = parser.add_argument_group("gateway")
    gateway.add_argument("--gateway-url", help="attach to this running gateway")
    gateway.add_argument(
        "--gateway-runs-dir",
        type=Path,
        default=protocol.RUNS_DIR,
        help="where the gateway writes <run_id>/gateway.jsonl (default %(default)s)",
    )
    gateway.add_argument(
        "--gateway-port", type=int, help="port of a gateway this cell launches"
    )
    gateway.add_argument(
        "--env-file", type=Path, help="dotenv file with the key, for a launched gateway"
    )
    gateway.add_argument(
        "--balance-floor", type=float, help="USD, for a launched gateway"
    )
    gateway.add_argument(
        "--budget",
        type=float,
        help="USD the launched gateway's runs directory may spend (infra/gateway.py)",
    )
    gateway.add_argument(
        "--allow-nonconformant-gateway",
        action="store_true",
        help="accept a gateway with non-protocol settings (tests and debugging only)",
    )
    parser.add_argument(
        "--test-concurrency", type=int, default=DEFAULT_TEST_CONCURRENCY
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT_S,
        help="seconds for each wait_until_idle (a timeout is INFRA)",
    )
    parser.add_argument(
        "--finalize-timeout",
        type=float,
        default=DEFAULT_FINALIZE_TIMEOUT_S,
        help="seconds for finalize (MIRIX: the flush of the remaining sessions)",
    )
    parser.add_argument(
        "--new-attempt",
        action="store_true",
        help="learn again from scratch instead of resuming an interrupted learning phase, "
        "or although the last attempt is frozen, testing, complete or in violation; once "
        "its test phase scored an item or sent an actor call it needs --note, and that "
        "attempt stays, marked superseded. The new attempt runs under this tree's "
        "protocol, frozen automatically (a changed one is adopted and the old one "
        "recorded in cell.json protocol_history)",
    )
    parser.add_argument(
        "--note",
        metavar="TEXT",
        help="one line, why this invocation runs (at most "
        f"{MAX_NOTE_CHARS} characters); recorded on its work line and, for a new "
        "attempt, in attempt.json note; required with --new-attempt when the attempt it "
        "replaces has already run its test phase. Repeating the same --new-attempt "
        "--note resumes the attempt it started instead of replacing it again",
    )
    parser.add_argument(
        "--max-learning-attempts",
        type=int,
        default=1,
        help="learning steps this invocation may take (default 1: an interrupted "
        "learning phase is resumed by the next invocation); more resumes it at once from "
        "its checkpoint. A HARNESS learning failure is never resumed automatically",
    )
    parser.add_argument(
        "--force-resume",
        action="store_true",
        help="resume a learning phase that a harness failure stopped (needs_decision), "
        "from its checkpoint, once a person has looked (recorded as operator_resume)",
    )
    parser.add_argument(
        "--keep-discarded-db",
        action="store_true",
        help="MIRIX: keep the post-crash database of a learning resume (<db>_dSS) for "
        "forensics instead of dropping it after its rows are exported",
    )
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="no longer needed: a resume under another commit, code or protocol runs "
        "and is recorded in attempt.json version_segments; accepted so older command "
        "lines still parse",
    )
    parser.add_argument(
        "--acknowledge-rerun",
        type=Path,
        metavar="NOTE",
        help="an operator's note (a text file: who, when, why) acknowledging that this "
        "cell directory runs the cell again although an earlier directory of the same "
        "cell id ran test items (H1.3); it is copied into the cell and recorded in "
        "cell.json with those earlier gateway runs, and the audit reports them as a "
        "warning instead of a violation. Record it before this directory's test phase: "
        "one recorded after it, or naming no earlier run here (another host), is "
        "recorded and reported by the audit",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[CellSpec, RunOptions, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = CellSpec(
            benchmark=args.benchmark,
            model=args.model,
            method=args.method,
            split_mode=args.split_mode,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        options = RunOptions(
            out_dir=args.out or protocol.RUNS_DIR / spec.cell_id,
            gateway_url=args.gateway_url,
            gateway_runs_dir=args.gateway_runs_dir,
            gateway_port=args.gateway_port,
            env_file=args.env_file,
            balance_floor_usd=args.balance_floor,
            budget_usd=args.budget,
            test_concurrency=args.test_concurrency,
            idle_timeout_s=args.idle_timeout,
            finalize_timeout_s=args.finalize_timeout,
            new_attempt=args.new_attempt,
            max_learning_attempts=args.max_learning_attempts,
            force_resume=args.force_resume,
            keep_discarded_db=args.keep_discarded_db,
            allow_code_change=args.allow_code_change,
            allow_nonconformant_gateway=args.allow_nonconformant_gateway,
            acknowledge_rerun=args.acknowledge_rerun,
            note=args.note,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return spec, options, args


async def _amain(runner: CellRunner) -> CellOutcome:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is not None:
        current = task

        def on_signal(sig: signal.Signals) -> None:
            # The first signal names the interruption in status.json.
            runner.received_signal = runner.received_signal or sig.name
            current.cancel()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, on_signal, sig)
    return await runner.run()


def run_cell(runner: CellRunner) -> int:
    """Run one cell as the CLI does and return its exit code: SIGTERM cancels it (an
    interrupted learning phase is resumed by the next invocation) and its outcome is
    printed as one JSON line. ``main`` uses it, and so does the tests' kill-and-resume
    worker (``tests/resume_worker.py``), so both have the same signal handling."""
    try:
        outcome = asyncio.run(_amain(runner))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            json.dumps({"cell_id": runner.spec.cell_id, "status": STATUS_INTERRUPTED})
        )
        return EXIT_CODES[STATUS_INTERRUPTED]
    print(json.dumps(outcome.to_dict(), ensure_ascii=False))
    return outcome.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    spec, options, args = parse_args(argv)
    logging.basicConfig(
        level=str(args.log_level).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_cell(CellRunner(spec, options))


if __name__ == "__main__":
    raise SystemExit(main())
