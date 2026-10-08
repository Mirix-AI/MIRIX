"""Delete a removed method's leftover data, after writing down what it cost.

Usage (from the repository root)::

    python -m evals.skill_bench.tools.remove_method_data --runs-dir <host runs dir> \\
        --method <method>                      # dry run: prints the plan, writes nothing
    python -m evals.skill_bench.tools.remove_method_data --runs-dir <host runs dir> \\
        --method <method> --yes [--note TEXT]

A method the study removed (``study.REMOVED_METHODS``) may leave cells and gateway runs
on every host, to be deleted once that is decided (:data:`DELETION_DECISIONS`). Their
calls were billed, so the report must keep counting their money after the files are
gone. This tool therefore writes the spend first, and deletes only after that record is
safely on disk:

1. ``<runs>/_removed/<method>/<removal_id>/calls.jsonl``: one row per final ``call`` line
   of every gateway ledger it deletes (the fields a spend window reads: ``run_id``,
   ``call_id``, ``ts``, ``completed_at``, ``latency_s``, ``cost_total``,
   ``unknown_cost_attempts`` ...);
2. ``manifest.json`` beside it (:data:`KIND`, schema 1): the decision, each deleted cell
   with its status and its costs exactly as ``report.cell_costs`` computed them (the
   report shows them in place of the cell), each deleted gateway run with its totals,
   each deleted entry's size and tree digest, and ``calls.jsonl``'s row count and
   sha256; state ``deleting``;
3. ``<runs>/_removed/index.json``: every removal of this runs directory with the
   top-level entries it deletes (``sync_runs`` reads it to let those deletions through
   its guard);
4. each entry is renamed into ``.trash.tmp/`` under the removal (atomic, same
   filesystem; ``sync_runs`` never mirrors a ``.*.tmp`` name) and removed there;
5. the manifest and the index get state ``complete``.

A run stopped anywhere is finished by the next run (``deleting`` removals are completed
first); no second manifest is written for the same entries.

Targets: every direct child of ``--runs-dir`` with a ``cell.json`` whose method
(``study.cell_method``) is ``--method``, and every direct child with a ``gateway.jsonl``
whose name is a run id of such a cell or of the method's cell id pattern (an orphan run
whose cell directory is already gone).

Refused (exit 2, nothing written): a method that is not removed or has no deletion
decision here (:data:`DELETION_DECISIONS`); a merged view or a host mirror
(``_merged.json``: mirrors follow their hosts through ``sync_runs``); a target that is a
symlink or lies outside the runs directory; a live target cell (its ``cell.lock`` held);
a removed cell whose ledger lies outside the runs directory; a target run that a study
cell's attempt names; another removal holding ``_removed/.lock``.

Never touched: ``_blobs`` (content-addressed, shared), ``_gateway`` (the key's spend
record: the budget is the key's usage, which this tool cannot change), ``_schedule``,
``_supervise``, ``_audit``, ``_report``, any study cell or gateway run.

Exit status: 0 planned, done or nothing left; 2 refused; 3 failed mid-way (re-run to
finish).

The readers (:func:`load_manifests`, :func:`manifest_calls`,
:func:`removed_entries_from_index`, :func:`removed_spend`) serve ``report.py``, the audit
and ``sync_runs``. This module imports ``report`` only inside the tool's functions:
``report.py`` imports the readers.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol, study
from evals.skill_bench.cell_records import (
    HOSTS_DIR,
    MERGED_MARKER,
    CellPaths,
    cell_is_live,
    git_info,
    host_info,
    now,
    read_json,
    read_jsonl,
    records_digest,
    resolve_ledger_path,
    run_id_pattern,
    unknown_cost_attempts,
    write_json_atomic,
)

REMOVED_DIR = "_removed"
INDEX_FILE = f"{REMOVED_DIR}/index.json"
MANIFEST_FILE = "manifest.json"
CALLS_FILE = "calls.jsonl"
#: Where entries go before they are removed (a ``.*.tmp`` name: never mirrored).
TRASH_DIR = ".trash.tmp"
LOCK_FILE = ".lock"
SCHEMA = 1
KIND = "skill_bench.removed_method_data"
STATE_DELETING = "deleting"
STATE_COMPLETE = "complete"
KIND_CELL = "cell"
KIND_GATEWAY_RUN = "gateway_run"

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_FAILED = 3

#: The decision to delete a removed method's data, per method: an operator records an
#: entry (``{"decision": ..., "quote": ...}``: who decided, and in what words) for a
#: method of ``study.REMOVED_METHODS`` once the deletion of its data is decided; it is
#: copied into each removal's manifest. No method has one in this release.
DELETION_DECISIONS: Mapping[str, Mapping[str, str]] = {}

#: The fields of a final ledger ``call`` line kept in ``calls.jsonl`` (what the report's
#: spend windows read; ``unknown_cost_attempts`` as the int the gateway's line implies).
CALL_FIELDS: tuple[str, ...] = (
    "run_id",
    "call_id",
    "event",
    "ts",
    "completed_at",
    "latency_s",
    "cost_total",
    "billed_attempts",
    "unknown_cost_attempts",
    "status",
    "run_phase",
    "phase",
    "role",
    "model",
    "provider",
    "item",
    "input_tokens_actual",
    "output_tokens",
    "reasoning_tokens",
)

KEPT: tuple[Mapping[str, str], ...] = (
    {
        "path": "_blobs",
        "why": "content-addressed, shared by every cell of the benchmark",
    },
    {"path": "_gateway", "why": "balance/budget/starts: the key's spend record"},
    {"path": "_schedule", "why": "the scheduler's state"},
    {"path": "_supervise", "why": "the supervisor's state"},
)
#: Top-level names of a runs directory that are never targets.
PROTECTED = frozenset(
    {
        "_blobs",
        "_gateway",
        "_schedule",
        "_supervise",
        "_audit",
        "_report",
        "_hosts",
        REMOVED_DIR,
    }
)


class Refused(Exception):
    """The removal is refused before anything is written (exit 2)."""


# ---------------------------------------------------------------------------
# Readers (report.py, the audit, sync_runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovalManifest:
    """One removal's ``manifest.json`` as read: ``problems`` lists why it (or its
    ``calls.jsonl``, once :func:`manifest_calls` read it) cannot be trusted as a whole."""

    path: Path
    data: dict[str, Any]
    calls_path: Path
    problems: list[str] = field(default_factory=list)

    @property
    def removal_id(self) -> str:
        return str(self.data.get("removal_id") or self.path.parent.name)

    @property
    def runs_dir(self) -> Path:
        """The runs directory the removal was made in (``<runs>/_removed/<m>/<id>/``)."""
        return self.path.parents[3]


def _manifest_paths(root: Path) -> list[Path]:
    try:
        return sorted((root / REMOVED_DIR).glob(f"*/*/{MANIFEST_FILE}"))
    except OSError:
        return []


def _manifest_problem(path: Path, loaded: Any) -> str | None:
    """Why a parsed ``manifest.json`` is not a removal manifest this reader knows, None
    when it is. ``kind`` and ``schema`` are checked when present (a file without them,
    such as one written by hand, is read); ``cells`` must be a list."""
    if not isinstance(loaded, dict):
        return f"{path}: not a JSON object"
    kind = loaded.get("kind", KIND)
    if kind != KIND:
        return f"{path}: kind {kind!r} is not {KIND!r}"
    schema = loaded.get("schema", SCHEMA)
    if isinstance(schema, bool) or schema != SCHEMA:
        return (
            f"{path}: schema {schema!r} is not {SCHEMA} (another version of the tool)"
        )
    if not isinstance(loaded.get("cells", []), list):
        return f"{path}: its cells are not a list"
    return None


def load_manifests(directory: Path, *, hosts: bool = True) -> list[RemovalManifest]:
    """Every removal manifest of ``directory`` (``<dir>/_removed/*/*/manifest.json``)
    and, with ``hosts``, of each host mirror of a merged view
    (``<dir>/_hosts/*/_removed/...``), one per ``removal_id`` (a readable copy before an
    unreadable one, else the first found), in path order. Never raises: an unreadable or
    foreign file is returned with its ``problems`` (blocking)."""
    directory = Path(directory)
    roots = [directory]
    if hosts:
        try:
            mirrors = sorted((directory / HOSTS_DIR).iterdir())
        except OSError:
            mirrors = []
        roots += [m for m in mirrors if m.is_dir()]
    found: list[RemovalManifest] = []
    for root in roots:
        for path in _manifest_paths(root):
            problems: list[str] = []
            data: dict[str, Any] = {}
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{path}: unreadable ({type(exc).__name__}: {exc})")
            else:
                problem = _manifest_problem(path, loaded)
                if problem is None:
                    data = loaded
                else:
                    problems.append(problem)
            found.append(
                RemovalManifest(
                    path,
                    data,
                    path.parent / str(data.get("calls_file") or CALLS_FILE),
                    problems,
                )
            )
    chosen: dict[str, RemovalManifest] = {}
    for manifest in found:
        held = chosen.get(manifest.removal_id)
        if held is None or (held.problems and not manifest.problems):
            chosen[manifest.removal_id] = manifest
    kept = {id(m) for m in chosen.values()}
    return [m for m in found if id(m) in kept]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def cell_row_problem(row: Any) -> str | None:
    """Why a manifest's ``cells`` row cannot stand in for the deleted cell in the report
    or the learning summary, None when it can (pure): it names its cell (``cell_id``,
    ``cell_dir``, ``method``, ``benchmark``) and carries its costs as
    ``report.cell_costs`` computed them (``total_cost``, ``other_attempts.cost``, and
    each side's ``cost`` and ``by_role``). The audit reads only the ids."""
    if not isinstance(row, Mapping):
        return "a cell row that is not an object"
    for key in ("cell_id", "cell_dir", "method", "benchmark"):
        if not isinstance(row.get(key), str) or not row.get(key):
            return f"a cell row without {key}"
    name = row["cell_id"]
    costs = row.get("costs")
    if not isinstance(costs, Mapping) or not _is_number(costs.get("total_cost")):
        return f"{name}: no costs.total_cost"
    other = costs.get("other_attempts")
    if not isinstance(other, Mapping) or not _is_number(other.get("cost")):
        return f"{name}: no costs.other_attempts.cost"
    for side in ("learning", "test"):
        entry = costs.get(side)
        roles = entry.get("by_role") if isinstance(entry, Mapping) else None
        if (
            not isinstance(entry, Mapping)
            or not _is_number(entry.get("cost"))
            or not isinstance(roles, Mapping)
            or not all(
                isinstance(v, Mapping) and _is_number(v.get("cost"))
                for v in roles.values()
            )
        ):
            return f"{name}: no costs.{side} (cost and by_role)"
    return None


def manifest_calls(manifest: RemovalManifest) -> list[dict[str, Any]]:
    """The rows of a removal's ``calls.jsonl`` (split on ``\\n`` only, like
    ``cell_records.read_jsonl``). A row count or sha256 that differs from the manifest's
    adds a problem; the rows are returned anyway (blocking)."""
    try:
        data = manifest.calls_path.read_bytes()
    except OSError as exc:
        manifest.problems.append(
            f"{manifest.calls_path}: unreadable ({type(exc).__name__}: {exc})"
        )
        return []
    rows: list[dict[str, Any]] = []
    torn = 0
    for line in data.decode("utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            torn += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            torn += 1
    expected_rows = manifest.data.get("calls_rows")
    expected_sha = manifest.data.get("calls_sha256")
    if torn or (isinstance(expected_rows, int) and expected_rows != len(rows)):
        manifest.problems.append(
            f"{manifest.calls_path}: {len(rows)} row(s) ({torn} unreadable), the manifest "
            f"says {expected_rows}"
        )
    if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
        manifest.problems.append(
            f"{manifest.calls_path}: its sha256 differs from the manifest's"
        )
    return rows


_SAFE_ENTRY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")


def removed_entries_from_index(
    text: str, *, finished_only: bool = False
) -> frozenset[str]:
    """The top-level entry names ``_removed/index.json`` (its text) says were deleted on
    purpose: of every removal in state ``deleting`` or ``complete``, or with
    ``finished_only`` of the ``complete`` ones. A name with a ``/``, or starting with
    ``_`` or ``.``, is never one (pure; unreadable text gives none)."""
    try:
        data = json.loads(text)
    except ValueError:
        return frozenset()
    if not isinstance(data, Mapping):
        return frozenset()
    states = {STATE_COMPLETE} if finished_only else {STATE_DELETING, STATE_COMPLETE}
    out: set[str] = set()
    for removal in data.get("removals") or ():
        if not isinstance(removal, Mapping) or removal.get("state") not in states:
            continue
        for entry in removal.get("entries") or ():
            if isinstance(entry, str) and _SAFE_ENTRY.fullmatch(entry):
                out.add(entry)
    return frozenset(out)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def removed_spend(runs_dir: Path) -> dict[str, Any]:
    """What the deleted gateway ledgers of ``runs_dir`` (and its host mirrors) recorded,
    from the removals' ``calls.jsonl``: ``{cost_usd, calls, unknown_cost_calls,
    removals, problems}`` (for a ledger-based spend reader; blocking)."""
    cost = 0.0
    calls = unknown = 0
    removals: list[str] = []
    problems: list[str] = []
    for manifest in load_manifests(runs_dir):
        rows = manifest_calls(manifest) if not manifest.problems else []
        problems += manifest.problems
        removals.append(manifest.removal_id)
        for row in rows:
            if row.get("event", "call") != "call":
                continue
            calls += 1
            cost += _number(row.get("cost_total"))
            unknown += int(unknown_cost_attempts(row) > 0)
    return {
        "cost_usd": round(cost, 10),
        "calls": calls,
        "unknown_cost_calls": unknown,
        "removals": removals,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _method_run_pattern(method: str) -> re.Pattern[str]:
    """Any gateway run id of a cell of ``method``: ``<benchmark>-<model>-<method>``
    (``-smoke-l<n>-t<n>``) ``-aNN`` (``-<instance>``)."""
    # Longest names first (``luna56`` before ``luna``).
    benchmarks = sorted(protocol.BENCHMARKS, key=len, reverse=True)
    models = sorted(protocol.MODELS, key=len, reverse=True)
    return re.compile(
        rf"^(?:{'|'.join(map(re.escape, benchmarks))})-"
        rf"(?:{'|'.join(map(re.escape, models))})-{re.escape(method)}"
        r"(?:-smoke-l\d+-t\d+)?-a\d{2,}(?:-[0-9a-f]+)?$"
    )


@dataclass
class Plan:
    """The entries a removal deletes."""

    method: str
    cells: list[Path] = field(default_factory=list)
    runs: list[Path] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def entries(self) -> list[Path]:
        return sorted([*self.cells, *self.runs], key=lambda p: p.name)


def _inside(path: Path, runs_dir: Path) -> bool:
    try:
        path.resolve().relative_to(runs_dir.resolve())
    except (OSError, ValueError):
        return False
    return True


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        data = read_json(root / "cell.json")
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def check_method(method: str) -> None:
    """Refuse a method that is not removed, or has no deletion decision here."""
    if not study.is_removed(method):
        raise Refused(
            f"{method!r} is not a method the study removed (study.REMOVED_METHODS: "
            f"{', '.join(sorted(study.REMOVED_METHODS))}): its data is never deleted"
        )
    if method not in DELETION_DECISIONS:
        raise Refused(
            f"no deletion decision recorded for {method!r} (DELETION_DECISIONS): its "
            "data stays"
        )


def check_runs_dir(runs_dir: Path) -> None:
    """Refuse a directory that is no host's runs directory: a merged view or a mirror."""
    if not runs_dir.is_dir():
        raise Refused(f"{runs_dir} is not a directory")
    if (runs_dir / MERGED_MARKER).exists():
        raise Refused(
            f"{runs_dir} is a merged view ({MERGED_MARKER}): delete on each host's own "
            "runs directory; the view follows through sync_runs"
        )
    parent = runs_dir.absolute().parent
    if parent.name == HOSTS_DIR and (parent.parent / MERGED_MARKER).exists():
        raise Refused(
            f"{runs_dir} is a host mirror of a merged view: delete on the host itself; "
            "the mirror follows through sync_runs"
        )


def plan_removal(runs_dir: Path, method: str) -> Plan:
    """The target entries of ``runs_dir`` and every reason to refuse (blocking, read
    only)."""
    plan = Plan(method)
    pattern = _method_run_pattern(method)
    try:
        children = sorted(runs_dir.iterdir())
    except OSError as exc:
        plan.refusals.append(f"{runs_dir}: {exc}")
        return plan
    cell_ids: set[str] = set()
    study_runs: set[str] = set()
    for child in children:
        if child.name in PROTECTED or child.name.startswith("."):
            continue
        if not (child / "cell.json").is_file():
            continue
        manifest = _read_manifest(child)
        definition = (
            manifest.get("cell") if isinstance(manifest.get("cell"), Mapping) else {}
        )
        cell_id = str(definition.get("cell_id") or child.name)
        if study.cell_method(definition, cell_id) == method:
            plan.cells.append(child)
            cell_ids.add(cell_id)
        else:
            paths = CellPaths(child)
            for number in paths.attempt_numbers():
                try:
                    state = read_json(paths.attempt(number).state) or {}
                except (OSError, ValueError):
                    state = {}
                if isinstance(state, Mapping) and state.get("run_id"):
                    study_runs.add(str(state["run_id"]))
    patterns = [run_id_pattern(c) for c in sorted(cell_ids)]
    for child in children:
        if child.name in PROTECTED or child.name.startswith("."):
            continue
        if child in plan.cells or not (child / "gateway.jsonl").is_file():
            continue
        if pattern.fullmatch(child.name) or any(
            p.fullmatch(child.name) for p in patterns
        ):
            plan.runs.append(child)
    for entry in plan.entries:
        if entry.is_symlink():
            plan.refusals.append(f"{entry.name} is a symlink: refused")
        elif not _inside(entry, runs_dir):
            plan.refusals.append(f"{entry.name} lies outside {runs_dir}: refused")
    for run in plan.runs:
        if run.name in study_runs:
            plan.refusals.append(
                f"{run.name} is the gateway run of a study cell's attempt: refused"
            )
    for root in plan.cells:
        paths = CellPaths(root)
        if cell_is_live(paths):
            plan.refusals.append(
                f"{root.name} is live (its cell.lock is held): refused"
            )
        for number in paths.attempt_numbers():
            try:
                state = read_json(paths.attempt(number).state) or {}
            except (OSError, ValueError):
                continue
            if not isinstance(state, Mapping):
                continue
            ledger = resolve_ledger_path(state, runs_dir=runs_dir)
            if ledger.exists() and not _inside(ledger, runs_dir):
                plan.refusals.append(
                    f"{root.name} attempt {number}'s ledger {ledger} lies outside "
                    f"{runs_dir}: its money could not be recorded with it; refused"
                )
    return plan


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def _tree(path: Path) -> dict[str, Any]:
    """Files, bytes and a digest of a directory tree (relative paths, file contents,
    symlink targets; links are never followed; blocking)."""
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(path):
        base = Path(current)
        found += [base / name for name in filenames]
        found += [base / name for name in dirnames if (base / name).is_symlink()]
    digest = hashlib.sha256()
    size = 0
    for full in sorted(found, key=lambda p: p.relative_to(path).as_posix()):
        digest.update(full.relative_to(path).as_posix().encode("utf-8") + b"\0")
        if full.is_symlink():
            digest.update(b"link:" + os.readlink(full).encode("utf-8") + b"\0")
            continue
        try:
            with open(full, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError:
            digest.update(b"unreadable\0")
    return {"files": len(found), "bytes": size, "tree_sha256": digest.hexdigest()}


def _ledger_calls(run: Path) -> list[dict[str, Any]]:
    """The final ``call`` lines of a gateway run's ledger as :data:`CALL_FIELDS` rows."""
    out = []
    for row in read_jsonl(run / "gateway.jsonl"):
        if row.get("event") != "call":
            continue
        kept = {k: row.get(k) for k in CALL_FIELDS if k in row}
        kept["run_id"] = run.name
        kept["event"] = "call"
        kept["unknown_cost_attempts"] = unknown_cost_attempts(row)
        out.append(kept)
    return out


def _run_record(runs_dir: Path, run: Path, cell_ids: Sequence[str]) -> dict[str, Any]:
    rows = read_jsonl(run / "gateway.jsonl")
    events = Counter(str(r.get("event")) for r in rows)
    calls = [r for r in rows if r.get("event") == "call"]
    started = {
        str(r["call_id"])
        for r in rows
        if r.get("event") == "call_started" and r.get("call_id")
    }
    finished = {str(r.get("call_id")) for r in calls}
    stamps = sorted(str(r["ts"]) for r in calls if r.get("ts"))
    ledger = run / "gateway.jsonl"
    try:
        data = ledger.read_bytes()
    except OSError:
        data = b""
    owner = next((c for c in cell_ids if run_id_pattern(c).fullmatch(run.name)), None)
    if owner is None:
        owner = _cell_id_of_run(run.name)
    return {
        "run_id": run.name,
        "cell_id": owner,
        "path": run.relative_to(runs_dir).as_posix(),
        "calls": len(calls),
        "cost_total": round(sum(_number(r.get("cost_total")) for r in calls), 10),
        "unknown_cost_calls": sum(1 for r in calls if unknown_cost_attempts(r) > 0),
        "unfinished_calls": len(started - finished),
        "first_call_at": stamps[0] if stamps else None,
        "last_call_at": stamps[-1] if stamps else None,
        "ledger_bytes": len(data),
        "ledger_sha256": hashlib.sha256(data).hexdigest(),
        "events": dict(sorted(events.items())),
    }


def _cell_id_of_run(run_id: str) -> str | None:
    """The cell id a run id names (``<cell_id>-aNN(-<instance>)``), None when none."""
    match = re.fullmatch(r"(.+)-a\d{2,}(?:-[0-9a-f]+)?", run_id)
    return match.group(1) if match else None


def _cell_record(runs_dir: Path, root: Path) -> dict[str, Any]:
    """One deleted cell as the manifest keeps it: the report's row of it (costs exactly
    as ``report.cell_costs`` computes them)."""
    from evals.skill_bench import report  # lazy: report imports this module

    cell = report.load_cell(root, runs_dir)
    costs = report.cell_costs(cell)
    reported = cell.attempt(cell.reported)
    state = reported.state if reported else {}
    totals: Counter[str] = Counter()
    unfinished = 0
    for attempt in cell.attempts:
        ledger = attempt.ledger or {}
        for key, value in (ledger.get("totals") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value
        unfinished += int(ledger.get("unfinished_calls") or 0)
    scored = {
        str(r.get("raw_id"))
        for r in (
            read_jsonl(cell.paths.attempt(cell.reported).results)
            if cell.reported is not None
            else []
        )
        if r.get("score") is not None
    }
    method = str(study.method_of(cell))
    return {
        "cell_id": cell.cell_id,
        "cell_dir": root.name,
        "instance": cell.manifest.get("instance"),
        "method": method,
        "benchmark": cell.benchmark,
        "model_key": cell.model_key,
        "split_mode": cell.split_mode,
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "attempts": len(cell.attempts),
        "attempt_states": {str(a.number): a.state.get("state") for a in cell.attempts},
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
        "protocol_fingerprint": protocol.attempt_protocol(
            cell.manifest, state, cell.reported
        )["fingerprint"],
        "host": report.cell_host(cell),
        "records_sha256": records_digest(cell.paths)
        if cell.paths.results.is_file()
        else None,
        "test_items_scored": len(scored),
        "costs": costs,
        "not_counted": report.uncounted_costs(costs),
        "calls": {
            "calls": int(totals.get("calls", 0)),
            "ok": int(totals.get("ok", 0)),
            "failed": int(totals.get("failed", 0)),
            "unknown_cost_calls": int(totals.get("unknown_cost_calls", 0)),
            "unfinished_calls": unfinished,
        },
        "tokens": {
            key: int(totals.get(key, 0))
            for key in (
                "input_tokens_estimated",
                "input_tokens_actual",
                "output_tokens",
                "reasoning_tokens",
            )
        },
        "gateway_runs": [
            str(a.state.get("run_id")) for a in cell.attempts if a.state.get("run_id")
        ],
    }


def build_manifest(
    runs_dir: Path,
    plan: Plan,
    *,
    removal_id: str,
    note: str | None,
    argv: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The manifest (state ``deleting``) and the ``calls.jsonl`` rows of ``plan``
    (blocking, read only)."""
    method = plan.method
    decision = DELETION_DECISIONS[method]
    cells = [_cell_record(runs_dir, root) for root in plan.cells]
    cell_ids = sorted({c["cell_id"] for c in cells})
    runs = [_run_record(runs_dir, run, cell_ids) for run in plan.runs]
    calls = [row for run in plan.runs for row in _ledger_calls(run)]
    deleted = [
        {
            "entry": entry.name,
            "kind": KIND_CELL if entry in plan.cells else KIND_GATEWAY_RUN,
            "path": entry.relative_to(runs_dir).as_posix(),
            **_tree(entry),
        }
        for entry in plan.entries
    ]
    manifest = {
        "schema": SCHEMA,
        "kind": KIND,
        "removal_id": removal_id,
        "method": method,
        "decision": decision["decision"],
        "quote": decision["quote"],
        "reason": study.REMOVED_METHODS[method].reason,
        "note": note,
        "state": STATE_DELETING,
        "created_at": now(),
        "completed_at": None,
        "host": host_info(),
        "runs_dir": str(runs_dir.absolute()),
        "tool": {"argv": list(argv), "git": git_info()},
        "cells": cells,
        "gateway_runs": runs,
        "deleted": deleted,
        "kept": [dict(k) for k in KEPT],
        "totals": {
            "cells": len(cells),
            "gateway_runs": len(runs),
            "calls": len(calls),
            "cost_total": round(sum(_number(r.get("cost_total")) for r in calls), 10),
            "unknown_cost_calls": sum(
                1 for r in calls if int(r.get("unknown_cost_attempts") or 0) > 0
            ),
            "unfinished_calls": sum(r["unfinished_calls"] for r in runs),
            "bytes": sum(d["bytes"] for d in deleted),
        },
        "calls_file": CALLS_FILE,
        "calls_rows": len(calls),
        "calls_sha256": None,
    }
    return manifest, calls


# ---------------------------------------------------------------------------
# Writing and deleting
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


def _write_calls(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """``calls.jsonl`` atomically (tmp, fsync, rename, fsync the directory); its sha256."""
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)
    return hashlib.sha256(data).hexdigest()


def _manifest_rel(manifest: Mapping[str, Any]) -> str:
    return f"{manifest['method']}/{manifest['removal_id']}/{MANIFEST_FILE}"


def _update_index(runs_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Add or replace this removal's line in ``_removed/index.json`` (atomic)."""
    path = runs_dir / INDEX_FILE
    try:
        index = read_json(path)
    except (OSError, ValueError):
        index = None
    if not isinstance(index, dict) or not isinstance(index.get("removals"), list):
        index = {"schema": SCHEMA, "removals": []}
    entry = {
        "removal_id": manifest["removal_id"],
        "method": manifest["method"],
        "state": manifest["state"],
        "manifest": _manifest_rel(manifest),
        "entries": [d["entry"] for d in manifest.get("deleted") or ()],
        "created_at": manifest.get("created_at"),
        "completed_at": manifest.get("completed_at"),
    }
    removals = [
        r
        for r in index["removals"]
        if not (isinstance(r, Mapping) and r.get("removal_id") == entry["removal_id"])
    ]
    index["removals"] = sorted(
        [*removals, entry], key=lambda r: str(r.get("removal_id"))
    )
    write_json_atomic(path, index)


def _delete_entries(
    runs_dir: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> None:
    """Move each entry into the removal's trash, then remove it there; an entry already
    moved is removed from the trash (a stopped run resumed; blocking)."""
    trash = manifest_path.parent / TRASH_DIR
    trash.mkdir(parents=True, exist_ok=True)
    for deleted in manifest.get("deleted") or ():
        name = str(deleted["entry"])
        if not _SAFE_ENTRY.fullmatch(name):
            raise Refused(f"the manifest names an unsafe entry {name!r}")
        live = runs_dir / name
        moved = trash / name
        if live.exists() or live.is_symlink():
            if live.is_symlink():
                raise Refused(f"{name} became a symlink since the manifest was written")
            if deleted.get("kind") == KIND_CELL:
                instance = _read_manifest(live).get("instance")
                expected = next(
                    (
                        c.get("instance")
                        for c in manifest.get("cells") or ()
                        if c.get("cell_dir") == name
                    ),
                    None,
                )
                if expected is not None and instance != expected:
                    raise Refused(
                        f"{name} holds another cell instance ({instance}) than the one "
                        f"the manifest recorded ({expected}): not deleted"
                    )
            os.rename(live, moved)
            _fsync_dir(runs_dir)
            _fsync_dir(trash)
        if moved.exists():
            shutil.rmtree(moved)
    shutil.rmtree(trash, ignore_errors=True)


def _complete(runs_dir: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["state"] = STATE_COMPLETE
    manifest["completed_at"] = now()
    write_json_atomic(manifest_path, manifest)
    _update_index(runs_dir, manifest)


def finish_pending(runs_dir: Path, method: str | None = None) -> list[str]:
    """Complete every removal of ``runs_dir`` left in state ``deleting`` (of
    ``method`` when given): the removal ids finished (blocking)."""
    finished = []
    for found in load_manifests(runs_dir, hosts=False):
        data = found.data
        if found.problems or data.get("state") != STATE_DELETING:
            continue
        if method is not None and data.get("method") != method:
            continue
        _delete_entries(runs_dir, found.path, data)
        _complete(runs_dir, found.path, dict(data))
        finished.append(found.removal_id)
    return finished


def _removal_id() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{secrets.token_hex(2)}"
    )


def _lock_file(path: Path, stack: ExitStack) -> bool:
    """Hold an exclusive ``flock`` on ``path`` for the stack's life; False when another
    process holds it. A missing file is created only for the removal's own lock."""
    try:
        handle = stack.enter_context(open(path, "a"))
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    stack.callback(fcntl.flock, handle.fileno(), fcntl.LOCK_UN)
    return True


def _summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cells": [
            {
                "cell_id": c["cell_id"],
                "cell_dir": c["cell_dir"],
                "status": c["status"],
                "total_cost": c["costs"]["total_cost"],
            }
            for c in manifest["cells"]
        ],
        "gateway_runs": [
            {"run_id": r["run_id"], "calls": r["calls"], "cost_total": r["cost_total"]}
            for r in manifest["gateway_runs"]
        ],
        "totals": manifest["totals"],
    }


def run(
    runs_dir: Path,
    method: str,
    *,
    yes: bool = False,
    note: str | None = None,
    argv: Sequence[str] = (),
) -> tuple[int, dict[str, Any]]:
    """The tool (blocking): ``(exit status, what to print)``."""
    runs_dir = Path(runs_dir)
    out: dict[str, Any] = {
        "runs_dir": str(runs_dir),
        "method": method,
        "dry_run": not yes,
    }
    try:
        check_method(method)
        check_runs_dir(runs_dir)
        if note is not None and ("\n" in note or "\r" in note or len(note) > 1000):
            raise Refused("--note is one line of at most 1000 characters")
        pending = [
            m.removal_id
            for m in load_manifests(runs_dir, hosts=False)
            if m.data.get("state") == STATE_DELETING
        ]
        plan = plan_removal(runs_dir, method)
        if not yes:
            out["pending_removals"] = pending
            out["refusals"] = plan.refusals
            if plan.entries and not plan.refusals:
                manifest, _calls = build_manifest(
                    runs_dir, plan, removal_id="(dry run)", note=note, argv=argv
                )
                out["plan"] = _summary(manifest)
            else:
                out["plan"] = {"entries": [e.name for e in plan.entries]}
            return (EXIT_REFUSED if plan.refusals else EXIT_OK), out
        if not pending:
            # Nothing is written before these pass (a stopped removal is finished first).
            if plan.refusals:
                raise Refused("; ".join(plan.refusals))
            if not plan.entries:
                out["deleted"] = []
                out["detail"] = f"nothing of {method} is left in {runs_dir}"
                return EXIT_OK, out
        with ExitStack() as stack:
            (runs_dir / REMOVED_DIR).mkdir(exist_ok=True)
            if not _lock_file(runs_dir / REMOVED_DIR / LOCK_FILE, stack):
                raise Refused(f"another removal holds {REMOVED_DIR}/{LOCK_FILE}")
            try:
                out["finished_pending"] = finish_pending(runs_dir)
            except OSError as exc:
                out["error"] = (
                    f"finishing a stopped removal: {type(exc).__name__}: {exc}"
                )
                return EXIT_FAILED, out
            plan = plan_removal(runs_dir, method)
            if plan.refusals:
                raise Refused("; ".join(plan.refusals))
            if not plan.entries:
                out["deleted"] = []
                out["detail"] = f"nothing of {method} is left in {runs_dir}"
                return EXIT_OK, out
            for root in plan.cells:
                lock = CellPaths(root).lock
                if lock.exists() and not _lock_file(lock, stack):
                    raise Refused(
                        f"{root.name} is live (its cell.lock is held): refused"
                    )
            removal_id = _removal_id()
            manifest, calls = build_manifest(
                runs_dir, plan, removal_id=removal_id, note=note, argv=argv
            )
            directory = runs_dir / REMOVED_DIR / method / removal_id
            manifest_path = directory / MANIFEST_FILE
            manifest["calls_sha256"] = _write_calls(directory / CALLS_FILE, calls)
            write_json_atomic(manifest_path, manifest)
            _update_index(runs_dir, manifest)
            out.update(removal_id=removal_id, manifest=str(manifest_path))
            out["plan"] = _summary(manifest)
            try:
                _delete_entries(runs_dir, manifest_path, manifest)
            except OSError as exc:
                out["error"] = (
                    f"stopped mid-way ({type(exc).__name__}: {exc}); the manifest is "
                    "written: run the tool again to finish"
                )
                return EXIT_FAILED, out
            _complete(runs_dir, manifest_path, manifest)
            out["deleted"] = [d["entry"] for d in manifest["deleted"]]
            out["state"] = STATE_COMPLETE
            return EXIT_OK, out
    except Refused as exc:
        out["refused"] = str(exc)
        return EXIT_REFUSED, out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        required=True,
        help="a method of study.REMOVED_METHODS with a deletion decision",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="delete (default: a dry run that writes nothing)",
    )
    parser.add_argument("--note", help="one line, recorded in the manifest")
    args = parser.parse_args(argv)
    code, out = run(
        args.runs_dir,
        args.method,
        yes=args.yes,
        note=args.note,
        argv=list(sys.argv if argv is None else argv),
    )
    print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
