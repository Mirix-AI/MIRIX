"""Each cell over its replicate runs directories: mean, spread and completeness.

Usage (from the repository root)::

    python -m evals.skill_bench.tools.aggregate_replicates --study phase16 \\
        evals/skill_bench/runs/phase16-r1 evals/skill_bench/runs/phase16-r2 \\
        evals/skill_bench/runs/phase16-r3 [--expect 3] [--out DIR]
    python -m evals.skill_bench.tools.aggregate_replicates RUNS_DIR [RUNS_DIR ...] \\
        [--split-mode {protocol,smoke,all}] [--expect N] [--out DIR]

A study with replicates (``study.STUDIES``; ``phase16``: three) runs every cell that many
times, in separate runs directories with identical cell ids. ``report.py`` reports one
runs directory; this tool puts the cells of several side by side and writes
``replicates.json`` (the source of truth) and ``replicates.md`` (default ``<parent of the
first runs directory>/_replicates/``):

* **Per cell id** (one benchmark, model and method; every cell id any runs directory
  holds, of the split mode selected): each metric ``report.py`` reports for the
  benchmark (``report.METRICS``: the main table's metric first, ``main``; ALFWorld's
  success rate, WebShop's success rate, mean score and success within 15 steps) per
  replicate, and over the replicates that have it: ``n``, ``mean``, the sample standard
  deviation (``std``, n - 1 in the denominator; None below two), ``min`` and ``max``.
  Each replicate also names its status, the reported attempt, its test items (their
  count and the sha256 of their sorted raw ids), its protocol fingerprint, its harness
  and MIRIX code digests, the digest of its agent context (``attempt.json``
  ``agent_context``), its audit status (from that runs directory's
  ``_audit/audit.json``) and its cost.
* **Flags** per cell id: ``missing_replicate`` (a runs directory without the cell),
  ``incomplete`` (not complete: still running, interrupted, waiting), ``blocked``
  (``refused`` or ``violation``: a person looks), ``no_metric`` (complete, but no metric:
  no scored test row), ``superseded_attempts`` (an attempt ``--new-attempt --note``
  replaced), ``duplicate_directories`` (another directory of the cell id in one runs
  directory; the newest complete one is used), ``rerun_after_test``
  (``report.cell_reruns``: an operator note or the audit says an earlier run of the cell
  id scored test items), ``mixed_protocol`` / ``mixed_code`` / ``mixed_mirix_code`` /
  ``mixed_agent_context`` (the replicates' reported attempts ran different protocol
  fingerprints, harness digests, MIRIX digests or agent contexts),
  ``mixed_version_segments`` (one replicate's attempt ran more than one version) and
  ``audit_not_passed`` (an audit exists and does not vouch for the records). Nothing is
  left out for a flag: the numbers say over which replicates they are.
* **Missing cells** (with ``--study``): every (benchmark, model, method) of the study
  that no runs directory holds a cell of.
* **Spend**: each runs directory's key spend against its ledgers
  (``report.gateway_spend``) and, for runs directories that share one budget start
  (``schedule.py --budget-start-from``), the union over them (``report.union_spend``).
  Unattributed spend is whatever the key spent in the window that no named runs
  directory's ledger holds: the other replicates' calls, development runs on the same
  key and any other use of the key.

The Markdown opens with a summary: per benchmark, each model and method's metrics as
mean ± std (n).

Offline and read-only: it reads the records ``run.py`` wrote with ``report.py``'s own
readers (``load_cells``, the removed methods set aside, ``select_reported``,
``test_metrics``, ``cell_protocol``, ``cell_code``, ``cell_agent_context``,
``cell_costs``, ``cell_reruns``, ``audit_status``), never an environment, a model or
the network, and scores nothing itself. Blocking file reads: async callers use
:func:`aggregate_async` (a thread), never ``asyncio.run`` inside a running loop. Exit 0
once written, 2 when no runs directory holds a complete cell.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol, report, study
from evals.skill_bench.cell_records import BLOCKED_STATUSES
from evals.skill_bench.report import CellData

SCHEMA_VERSION = 3
OUT_DIR_NAME = "_replicates"
REPLICATES_JSON = "replicates.json"
REPLICATES_MD = "replicates.md"
#: The order rows are listed in within a benchmark and group: ``schedule.METHOD_ORDER``
#: (a test pins the two together; this tool does not import the scheduler).
METHOD_ORDER: tuple[str, ...] = ("mirix", "reasoningbank", "awm", "none")

#: Flags (module docstring).
FLAG_MISSING = "missing_replicate"
FLAG_INCOMPLETE = "incomplete"
FLAG_BLOCKED = "blocked"
FLAG_NO_METRIC = "no_metric"
FLAG_SUPERSEDED = "superseded_attempts"
FLAG_DUPLICATES = "duplicate_directories"
FLAG_RERUN = "rerun_after_test"
FLAG_MIXED_PROTOCOL = "mixed_protocol"
FLAG_MIXED_CODE = "mixed_code"
FLAG_MIXED_MIRIX = "mixed_mirix_code"
FLAG_MIXED_AGENT_CONTEXT = "mixed_agent_context"
FLAG_MIXED_SEGMENTS = "mixed_version_segments"
FLAG_AUDIT = "audit_not_passed"


@dataclass
class Replicate:
    """One runs directory, read with report.py's readers."""

    runs_dir: Path
    label: str
    cells: dict[str, CellData] = field(default_factory=dict)
    #: Other directories of a cell id (``report.select_reported``), by cell id.
    set_apart: dict[str, list[str]] = field(default_factory=dict)
    audit: Mapping[str, Any] | None = None
    removed: list[str] = field(default_factory=list)


def _labels(runs_dirs: Sequence[Path]) -> list[str]:
    """Each runs directory's label: its name, or its whole path when names repeat."""
    names = [d.name for d in runs_dirs]
    if len(set(names)) == len(names):
        return names
    return [str(d) for d in runs_dirs]


def load_replicate(
    runs_dir: Path,
    label: str,
    *,
    split_mode: str | None = "protocol",
    gateway_runs_dirs: Sequence[Path] = (),
) -> Replicate:
    """The cells of one runs directory as ``report.build_report`` reads them: the removed
    methods' cells set aside, one directory per cell id. Blocking."""
    loaded = report.load_cells(
        runs_dir, split_mode=split_mode, gateway_runs_dirs=gateway_runs_dirs
    )
    kept, removed = study.split_cells(loaded)
    chosen, set_apart = report.select_reported(kept)
    found = Replicate(
        runs_dir=runs_dir,
        label=label,
        cells={c.cell_id: c for c in chosen},
        audit=report.load_audit(runs_dir / report.AUDIT_FILE),
        removed=sorted(c.cell_id for c in removed),
    )
    for cell, _winner, role in set_apart:
        found.set_apart.setdefault(cell.cell_id, []).append(
            f"{cell.paths.root.name} ({role})"
        )
    return found


def summarize(values: Sequence[float]) -> dict[str, Any]:
    """``n``, ``mean``, sample ``std`` (None below two values), ``min`` and ``max``."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) >= 2 else None,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def test_items_digest(cell: CellData) -> str | None:
    """sha256 of the sorted raw ids of the cell's test rows (None without rows): two
    cells tested the same items exactly when these are equal."""
    ids = sorted(str(r.get("raw_id")) for r in cell.results)
    if not ids:
        return None
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _replicate_entry(
    cell: CellData, replicate: Replicate, frozen: str | None
) -> dict[str, Any]:
    """What one replicate's directory says about the cell."""
    protocol_record = report.cell_protocol(cell, frozen=frozen)
    versions = report.cell_version_record(cell, frozen=frozen)
    code = report.cell_code(cell)
    context = report.cell_agent_context(cell)
    costs = report.cell_costs(cell)
    test = report.test_metrics(cell) if cell.complete else None
    return {
        "cell_dir": cell.paths.root.name,
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "attempts": len(cell.attempts),
        "superseded_attempts": [
            a.number for a in cell.attempts if report.attempt_superseded(a.state)
        ],
        "newer_attempt": cell.newer_attempt,
        "record_problems": list(cell.record_problems),
        "protocol_fingerprint": protocol_record["fingerprint"],
        "protocol_current": protocol_record["current"],
        "harness": code.get("harness"),
        "mirix": code.get("mirix"),
        "commit": code.get("commit"),
        "agent_context_sha256": context["sha256"] if context else None,
        "versions_mixed": versions["mixed"],
        "audit": report.audit_status(cell, replicate.audit).get("status"),
        "reruns": report.cell_reruns(cell, replicate.audit) is not None,
        "cost_usd": costs.get("total_cost"),
        "items": test["items"] if test else None,
        "test_items_sha256": test_items_digest(cell) if test else None,
        "metrics": dict(test["metrics"]) if test else {},
    }


def _flag(name: str, replicates: Sequence[str], detail: str) -> dict[str, Any]:
    return {"flag": name, "replicates": list(replicates), "detail": detail}


def cell_flags(
    cell_id: str,
    entries: Mapping[str, dict[str, Any] | None],
    replicates: Sequence[Replicate],
) -> list[dict[str, Any]]:
    """The flags of one cell id over the replicates (module docstring)."""
    flags: list[dict[str, Any]] = []
    missing = [label for label, e in entries.items() if e is None]
    if missing:
        flags.append(
            _flag(FLAG_MISSING, missing, f"no {cell_id} in {', '.join(missing)}")
        )
    present = {label: e for label, e in entries.items() if e is not None}
    blocked = [label for label, e in present.items() if e["status"] in BLOCKED_STATUSES]
    if blocked:
        flags.append(
            _flag(
                FLAG_BLOCKED,
                blocked,
                ", ".join(f"{label}: {present[label]['status']}" for label in blocked)
                + " (a person looks)",
            )
        )
    incomplete = [
        label
        for label, e in present.items()
        if not e["complete"] and label not in blocked
    ]
    if incomplete:
        flags.append(
            _flag(
                FLAG_INCOMPLETE,
                incomplete,
                ", ".join(
                    f"{label}: {present[label]['status'] or 'no status'}"
                    + (
                        f", attempt {present[label]['newer_attempt']['number']} running"
                        if present[label]["newer_attempt"]
                        else ""
                    )
                    for label in incomplete
                ),
            )
        )
    no_metric = [
        label
        for label, e in present.items()
        if e["complete"] and all(v is None for v in e["metrics"].values())
    ]
    if no_metric:
        flags.append(
            _flag(
                FLAG_NO_METRIC,
                no_metric,
                "complete without a metric (no scored test row)",
            )
        )
    superseded = [label for label, e in present.items() if e["superseded_attempts"]]
    if superseded:
        flags.append(
            _flag(
                FLAG_SUPERSEDED,
                superseded,
                ", ".join(
                    f"{label}: attempt(s) "
                    + ", ".join(str(n) for n in present[label]["superseded_attempts"])
                    for label in superseded
                ),
            )
        )
    by_label = {r.label: r for r in replicates}
    duplicates = [label for label in present if by_label[label].set_apart.get(cell_id)]
    if duplicates:
        flags.append(
            _flag(
                FLAG_DUPLICATES,
                duplicates,
                "; ".join(
                    f"{label}: {', '.join(by_label[label].set_apart[cell_id])} not used"
                    for label in duplicates
                ),
            )
        )
    reruns = [label for label, e in present.items() if e.get("reruns")]
    if reruns:
        flags.append(
            _flag(
                FLAG_RERUN,
                reruns,
                "an earlier run of the cell id scored test items (an operator note or "
                "the audit says so; report.py 'Runs and attempts')",
            )
        )
    for name, key, what in (
        (FLAG_MIXED_PROTOCOL, "protocol_fingerprint", "protocol fingerprints"),
        (FLAG_MIXED_CODE, "harness", "harness digests"),
        (FLAG_MIXED_MIRIX, "mirix", "MIRIX code digests"),
        (FLAG_MIXED_AGENT_CONTEXT, "agent_context_sha256", "agent contexts"),
    ):
        values = {
            label: e.get(key)
            for label, e in present.items()
            if e["complete"] and e.get(key)
        }
        if len(set(values.values())) > 1:
            flags.append(
                _flag(
                    name,
                    sorted(values),
                    f"{len(set(values.values()))} {what}: "
                    + ", ".join(
                        f"{label} {str(v)[:12]}" for label, v in sorted(values.items())
                    ),
                )
            )
    segments = [label for label, e in present.items() if e["versions_mixed"]]
    if segments:
        flags.append(
            _flag(
                FLAG_MIXED_SEGMENTS,
                segments,
                "the reported attempt ran more than one protocol or code version",
            )
        )
    audited = [
        label
        for label, e in present.items()
        if e["complete"]
        and by_label[label].audit is not None
        and e["audit"] != report.AUDIT_PASSED
    ]
    if audited:
        flags.append(
            _flag(
                FLAG_AUDIT,
                audited,
                ", ".join(f"{label}: {present[label]['audit']}" for label in audited),
            )
        )
    return flags


def main_metric(benchmark: str) -> str | None:
    """The benchmark's main-table metric (``report.METRICS``: the first ``main`` one)."""
    return next((m.name for m in report.METRICS.get(benchmark, ()) if m.main), None)


def aggregate_cells(
    replicates: Sequence[Replicate], *, frozen: str | None = None
) -> list[dict[str, Any]]:
    """One row per cell id any replicate holds, in launch order (benchmark, method,
    model as ``schedule.py`` orders them, then the cell id)."""
    ids = sorted({cell_id for r in replicates for cell_id in r.cells})
    rows = []
    for cell_id in ids:
        sample = next(r.cells[cell_id] for r in replicates if cell_id in r.cells)
        entries: dict[str, dict[str, Any] | None] = {
            r.label: _replicate_entry(r.cells[cell_id], r, frozen)
            if cell_id in r.cells
            else None
            for r in replicates
        }
        metrics: dict[str, Any] = {}
        for metric in report.METRICS.get(sample.benchmark, ()):
            values = {
                label: (e["metrics"].get(metric.name) if e is not None else None)
                for label, e in entries.items()
            }
            known = [float(v) for v in values.values() if v is not None]
            metrics[metric.name] = {
                "label": metric.label,
                "main": metric.main,
                "binary": metric.binary,
                "values": values,
                **summarize(known),
            }
        rows.append(
            {
                "cell_id": cell_id,
                "benchmark": sample.benchmark,
                "model_key": sample.model_key,
                "method": sample.method,
                "split_mode": sample.split_mode,
                "group": sample.group,
                "main_metric": main_metric(sample.benchmark),
                "metrics": metrics,
                "replicates": entries,
                "flags": cell_flags(cell_id, entries, replicates),
            }
        )
    order = {b: i for i, b in enumerate(protocol.BENCHMARKS)}
    methods = {m: i for i, m in enumerate(METHOD_ORDER)}
    models = {m: i for i, m in enumerate(protocol.MODELS)}
    rows.sort(
        key=lambda r: (
            order.get(r["benchmark"], 99),
            r["group"],
            methods.get(r["method"], 99),
            models.get(r["model_key"], 99),
            r["cell_id"],
        )
    )
    return rows


def missing_cells(
    entry: study.Study, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    """Every (benchmark, model, method) of the study that no row holds, in the study's
    order (pure)."""
    held = {(r["benchmark"], r["model_key"], r["method"]) for r in rows}
    return [
        {"benchmark": b, "model_key": m, "method": method}
        for b in entry.benchmarks
        for m in entry.models
        for method in entry.methods
        if (b, m, method) not in held
    ]


def replicate_spend(replicates: Sequence[Replicate]) -> dict[str, Any]:
    """Each runs directory's key spend against its ledgers, and the union over the ones
    that share a budget start (``report.gateway_spend`` / ``report.union_spend``)."""
    spends = []
    for replicate in replicates:
        spend = report.gateway_spend(replicate.runs_dir)
        if spend is not None:
            spends.append({"replicate": replicate.label, **spend})
    union = report.union_spend(spends) if len(spends) >= 2 else None
    return {"directories": spends, "union": union}


def aggregate(
    runs_dirs: Sequence[Path],
    *,
    split_mode: str | None = "protocol",
    expect: int | None = None,
    study_name: str | None = None,
) -> dict[str, Any]:
    """The whole aggregation as JSON-ready data (blocking; module docstring).
    ``study_name`` (``study.STUDIES``) takes the study's split mode and replicate count
    (``expect`` by default, whatever the number of directories named), keeps only the
    study's cells and lists the study's cells no runs directory holds
    (``missing_cells``). ``warnings`` names a runs directory count that differs from the
    expected replicates."""
    if not runs_dirs:
        raise ValueError("name at least one runs directory")
    entry = None
    warnings: list[str] = []
    if study_name is not None:
        entry = study.STUDIES.get(study_name)
        if entry is None:
            raise ValueError(
                f"unknown study {study_name!r}; studies: {', '.join(study.STUDIES)}"
            )
        split_mode = entry.split_mode
    runs_dirs = [Path(d).absolute() for d in runs_dirs]
    if len({str(d) for d in runs_dirs}) != len(runs_dirs):
        raise ValueError("a runs directory is named twice")
    if expect is None:
        expect = entry.replicates if entry is not None else len(runs_dirs)
    if expect < 1:
        raise ValueError("--expect must be at least 1")
    if len(runs_dirs) != expect:
        warnings.append(
            f"{len(runs_dirs)} runs directories named, {expect} replicates expected"
            + (f" (--study {study_name})" if entry is not None else "")
            + ": each cell's numbers are over the runs directories named"
        )
    try:
        frozen: str | None = protocol.frozen_fingerprint()
    except protocol.ProtocolDriftError:
        frozen = None
    labels = _labels(runs_dirs)
    replicates = [
        load_replicate(d, label, split_mode=split_mode)
        for d, label in zip(runs_dirs, labels)
    ]
    if entry is not None:
        wanted = {
            (b, m, method)
            for b in entry.benchmarks
            for m in entry.models
            for method in entry.methods
        }
        for replicate in replicates:
            outside = sorted(
                cell_id
                for cell_id, cell in replicate.cells.items()
                if (cell.benchmark, cell.model_key, cell.method) not in wanted
            )
            for cell_id in outside:
                del replicate.cells[cell_id]
    rows = aggregate_cells(replicates, frozen=frozen)
    flagged = sorted({f["flag"] for row in rows for f in row["flags"]})
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": study_name,
        "split_mode": split_mode or "all",
        "frozen_fingerprint": frozen,
        "replicates": [
            {
                "label": r.label,
                "runs_dir": str(r.runs_dir),
                "cells": len(r.cells),
                "complete": sum(1 for c in r.cells.values() if c.complete),
                "audit": r.audit is not None and not r.audit.get("unreadable"),
                "removed_cells": r.removed,
            }
            for r in replicates
        ],
        "expect_replicates": expect,
        "warnings": warnings,
        "cells": rows,
        "flags": flagged,
        "missing_cells": missing_cells(entry, rows) if entry is not None else [],
        "spend": replicate_spend(replicates),
    }


async def aggregate_async(runs_dirs: Sequence[Path], **kwargs: Any) -> dict[str, Any]:
    """:func:`aggregate` from async code: its file reads run in a thread."""
    return await asyncio.to_thread(aggregate, runs_dirs, **kwargs)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def mean_std_text(stats: Mapping[str, Any]) -> str:
    """A metric's replicate statistics as ``mean ± std (n=N)`` (no ``± std`` below two
    values); ``-`` without a value."""
    if not stats.get("n"):
        return "-"
    text = _fmt(stats["mean"])
    if stats.get("std") is not None:
        text += f" ± {_fmt(stats['std'])}"
    return f"{text} (n={stats['n']})"


def _summary_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per benchmark: one line per cell (its group, model and method), each metric as
    mean ± std over the replicates."""
    out: list[str] = []
    for benchmark in dict.fromkeys(r["benchmark"] for r in rows):
        mine = [r for r in rows if r["benchmark"] == benchmark]
        names = list(mine[0]["metrics"])
        labels = [
            mine[0]["metrics"][n]["label"]
            + (" (main)" if mine[0]["metrics"][n]["main"] else "")
            for n in names
        ]
        header = ["Group", "Model", "Method", *labels]
        out += ["", f"### {benchmark}", ""]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for row in mine:
            values = [mean_std_text(row["metrics"].get(n) or {}) for n in names]
            out.append(
                "| "
                + " | ".join([row["group"], row["model_key"], row["method"], *values])
                + " |"
            )
    if not out:
        return ["", "## Summary", "", "No cell in any runs directory."]
    return ["", "## Summary", *out]


def render_markdown(data: Mapping[str, Any]) -> str:
    """The Markdown view of :func:`aggregate`'s data."""
    labels = [r["label"] for r in data["replicates"]]
    selection = f"split mode `{data['split_mode']}`"
    if data.get("study"):
        selection = f"Study `{data['study']}`, {selection}"
    lines = [
        "# Replicates",
        "",
        f"{selection[0].upper()}{selection[1:]}, {len(labels)} runs directories, "
        f"{data['expect_replicates']} replicates expected per cell; frozen protocol "
        f"`{str(data['frozen_fingerprint'])[:12]}`. Generated {data['generated_at']}.",
        "",
    ]
    for r in data["replicates"]:
        lines.append(
            f"- `{r['label']}`: {r['runs_dir']} ({r['complete']}/{r['cells']} cells "
            f"complete{'' if r['audit'] else ', no audit'})"
        )
    for warning in data.get("warnings") or ():
        lines.append(f"- **Warning:** {warning}")
    missing = data.get("missing_cells") or []
    if missing:
        lines.append(
            "- **Missing cells** (the study's, in no runs directory): "
            + ", ".join(
                f"{m['benchmark']}/{m['model_key']}/{m['method']}" for m in missing
            )
        )
    lines += _summary_section(data["cells"])
    lines += ["", "## Cells", ""]
    header = ["Cell", "Metric", *labels, "n", "mean", "std", "Flags"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in data["cells"]:
        flags = ", ".join(f["flag"] for f in row["flags"]) or "-"
        for name, stats in row["metrics"].items():
            label = stats["label"] + (" (main)" if stats["main"] else "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["cell_id"],
                        label,
                        *(_fmt(stats["values"].get(lab)) for lab in labels),
                        str(stats["n"]),
                        _fmt(stats["mean"]),
                        _fmt(stats["std"]),
                        flags,
                    ]
                )
                + " |"
            )
            flags = ""  # once per cell
    flagged = [row for row in data["cells"] if row["flags"]]
    if flagged:
        lines += ["", "## Flags", ""]
        for row in flagged:
            for f in row["flags"]:
                lines.append(f"- `{row['cell_id']}` {f['flag']}: {f['detail']}")
    spend = data["spend"]
    lines += ["", "## Spend", ""]
    for s in spend["directories"]:
        lines.append(
            f"- `{s['replicate']}`: key usage +${_fmt(s['key_usage_delta_usd'], 2)}, "
            f"ledgers ${_fmt(s['ledger_cost_usd'], 2)}, unattributed "
            f"${_fmt(s['unattributed_usd'], 2)} (any spend on the key in its window "
            "that its own ledgers do not hold: the other replicates' calls, development "
            "runs and any other use of the same key)"
        )
    union = spend["union"]
    if union is not None:
        if union.get("shared") and union.get("key_usage_delta_usd") is not None:
            lines.append(
                f"- Shared budget start, all replicates: key usage "
                f"+${_fmt(union['key_usage_delta_usd'], 2)}, ledgers "
                f"${_fmt(union['ledger_cost_usd'], 2)}, unattributed "
                f"${_fmt(union['unattributed_usd'], 2)} (any spend on the key since the "
                "shared start that no named runs directory's ledger holds: development "
                "runs on the same key, other steps)"
            )
        else:
            lines.append(f"- No union: {union.get('detail')}")
    if not spend["directories"]:
        lines.append("No balance record in any runs directory.")
    return "\n".join(lines) + "\n"


def write(data: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / REPLICATES_JSON
    md_path = out_dir / REPLICATES_MD
    json_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(data), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.tools.aggregate_replicates",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("runs_dirs", nargs="+", type=Path, metavar="RUNS_DIR")
    parser.add_argument(
        "--study",
        type=study.study_argument,
        help="the named study (study.STUDIES, e.g. phase16): its split mode, its "
        "replicate count and its cells only, and the study's cells no runs directory "
        "holds",
    )
    parser.add_argument(
        "--split-mode",
        choices=("protocol", "smoke", "all"),
        help="which cells (default: protocol cells only, as report.py; with --study, "
        "the study's)",
    )
    parser.add_argument(
        "--expect",
        type=int,
        help="replicates expected per cell (default: the study's replicate count with "
        "--study, else the number of runs directories); another number of runs "
        "directories is a warning",
    )
    parser.add_argument(
        "--out", type=Path, help="default <parent of the first runs dir>/_replicates"
    )
    args = parser.parse_args(argv)
    if args.study and args.split_mode is not None:
        parser.error(f"--study {args.study} sets the split mode itself")
    mode = args.split_mode or "protocol"
    try:
        data = aggregate(
            args.runs_dirs,
            split_mode=None if mode == "all" else mode,
            expect=args.expect,
            study_name=args.study,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for warning in data["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    out = args.out or args.runs_dirs[0].absolute().parent / OUT_DIR_NAME
    json_path, md_path = write(data, out)
    complete = sum(r["complete"] for r in data["replicates"])
    print(
        json.dumps(
            {
                "cells": len(data["cells"]),
                "complete": complete,
                "flags": data["flags"],
                "missing_cells": len(data["missing_cells"]),
                "json": str(json_path),
                "markdown": str(md_path),
            }
        )
    )
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
