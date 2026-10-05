"""What every cell's method learned before its memory froze, what it injected at test time,
and what the cell cost (a readiness check: did each learning method learn anything?).

Usage (from the repository root)::

    python -m evals.skill_bench.tools.summarize_learning --runs-dir evals/skill_bench/runs
    python -m evals.skill_bench.tools.summarize_learning --split-mode smoke --json

It reads only the scorer-side records ``run.py`` writes (the reported attempt's
``snapshots/frozen.json``, ``injections.jsonl`` and gateway ledger, through ``report.py``'s
loader), never a method's live state, and prints for every cell:

* **what the method learned** by the freeze, from its frozen snapshot:

  - ``mirix``: the skills (count; each skill's name, version and stored versions with
    their change types), the skills a run deleted (with their history), any skill frozen
    without its retrieval vector, the experiences its learning runs distilled, the
    outcomes their sessions were stored with (and the stated outcomes the transcript
    did not show, or only the agent's own words did), the curator's writes, the
    learning runs by status (with the first errors), the experience pool (its size
    at the freeze and how many experiences entered it, what became of each new
    experience, the earlier lessons shown and what became of them, recurrences,
    retirements, the curator's citations without a verified quote), and what the
    distiller was shown and said (``methods.mirix.learning_context``: the learned
    sessions distilled with the agent context, the skills served per session, each
    served skill's stored followed / effect values, the usage records the curator
    batches showed);
  - ``reasoningbank``: the memory items (count and titles) and the trajectories they came
    from, by J's verdict;
  - ``awm``: the workflow (characters), the inductions run and the workflows written;
  - ``none``: nothing, by design;

* **test-phase injections** (o200k tokens counted by the framework): items, empty
  injections, mean / median / max tokens and the total;
* **cost** from the gateway ledgers (``cost_total``, billed retries included): learning
  side, test phase, invalidated attempts, and the total; and what the total leaves
  out (``not_counted``, as ``report.py`` prints it): attempts without a ledger, calls
  without a final ledger line and cut-off calls (the deadline or a dropped connection
  cut an attempt off after its request was sent), billed by OpenRouter at a cost the
  ledger never learns;
* **learning resumes**: how many times the learning phase was rolled back to a
  checkpoint and resumed, and what the discarded calls cost (part of the learning
  side's cost, shown apart); and the **host** the reported attempt ran on.

The ledgers are found where they are on this host (``cell_records.resolve_ledger_path``:
relative to ``--runs-dir``, then each ``--gateway-runs-dir``, then a merged view's
``_hosts/*``), so a runs directory copied from another host summarizes the same.

A learning method *produced nothing* when its frozen memory is empty (no MIRIX skill, no
ReasoningBank memory item, an empty AWM workflow) or when the cell has no frozen memory
to read (learning did not finish). A learning method whose test
phase injected nothing although it learned something gets a warning (for MIRIX: no skill
passed the retrieval gate). So does a MIRIX curator batch that showed no earlier lessons
because the experience pool's embedding failed (the attempt should have restarted) or
because the pool or MIRIX's embeddings were off (not MIRIX's default), pool
metrics that do not add up (entered - retired - active is not 0), and a cell whose
benchmark offers the agent context (the attempt recorded one, or, for an attempt
recorded before it named one, :data:`AGENT_CONTEXT_BENCHMARKS`) with a learned session
MIRIX distilled without it, or with a different length than the attempt recorded
(a wrong tree or a lost attach shows here, not only in SQL).

The cells of a method the study removed (``study.REMOVED_METHODS``) are set aside:
listed under ``removed`` with the decision, their status and their ledgers' cost
(``removed_cost``; ``total_cost_all_cells`` is the study's total and theirs;
``removed_unknown_cost_calls`` and ``removed_unfinished_calls`` count their calls of
unknown cost, in neither), never summarized, and never part of the exit status. A cell
is a removed method's by the method its ``cell.json`` names, else by the one its id
names (``study.method_of``). Once such cells are deleted
(``tools/remove_method_data.py``), they are listed from the removal manifests
(``_removed/<method>/<removal_id>/manifest.json``, a merged view's host mirrors
included), with ``on_disk`` false and their cost as the manifest recorded it, so the
totals stay what they were; an unreadable manifest is named in
``removed_manifest_problems``.

What a learning method accepted and went on from (recorded, not a reason to stop) is
listed per cell and warned about: MIRIX's ``tolerated`` block (learning calls
lost to the gateway or rolled back three times, retrievals degraded), and the learning
calls the provider refused that AWM and ReasoningBank counted (``refusals``).

Exit status: 0 every learning cell learned something; 1 at least one produced nothing;
2 no cell of the study was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol, study
from evals.skill_bench.report import (
    METHOD_REFUSAL_COUNTERS,
    POOL_SKIPS_NAMED,
    CellData,
    cell_agent_context,
    cell_costs,
    cell_host,
    cell_injections,
    cell_resumes,
    load_cells,
    mirix_pool,
    uncounted_costs,
)
from evals.skill_bench.tools.remove_method_data import cell_row_problem, load_manifests

EXIT_OK = 0
EXIT_NOTHING_LEARNED = 1
EXIT_NO_CELLS = 2
#: Skill names / memory titles printed per cell (the JSON has them all).
LIST_LIMIT = 20
#: Learning-run errors kept per cell.
ERROR_LIMIT = 5
#: Session ids named per warning (the JSON has them all).
SESSION_LIMIT = 5
#: The benchmarks whose environment offers the agent context (those with an
#: ``agent_context()``, ``envs/agent_context.py``), for an attempt recorded before
#: ``attempt.json`` named its context; a test keeps it equal to the environments.
AGENT_CONTEXT_BENCHMARKS: tuple[str, ...] = ("alfworld", "webshop")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _entries(run: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    return [e for e in run.get(key) or () if isinstance(e, Mapping)]


# ---------------------------------------------------------------------------
# What each method learned (its frozen snapshot)
# ---------------------------------------------------------------------------


def learned_mirix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Skills, their versions, experiences (and the outcomes their sessions were stored
    with), learning runs and the experience pool (``report.mirix_pool``, None for a
    MIRIX before the pool), the exceptions curator tools met that MIRIX answered to the
    model as failed tool results (``methods.mirix.learning_warnings``; none for a MIRIX
    before it recorded them), and what the distiller was shown and said of the served
    skills, with the usage records the curator saw (``learning_context``:
    ``methods.mirix.learning_context``; None for a MIRIX before it recorded them)."""
    from evals.skill_bench.methods.mirix import learning_context, learning_warnings

    history = _mapping(snapshot.get("skill_history"))
    stored = _mapping(snapshot.get("skill_versions"))
    skills = []
    for skill in snapshot.get("skills") or ():
        if not isinstance(skill, Mapping):
            continue
        skill_id = str(skill.get("id"))
        versions = [
            v
            for v in _mapping(history.get(skill_id)).get("versions") or ()
            if isinstance(v, Mapping)
        ]
        skills.append(
            {
                "id": skill_id,
                "name": skill.get("name"),
                "version": skill.get("version"),
                "entry_type": skill.get("entry_type"),
                "versions": len(versions),
                "changes": [str(v.get("change_type")) for v in versions],
                "versions_stored": len(_mapping(stored.get(skill_id))),
            }
        )
    runs = [r for r in snapshot.get("learning_runs") or () if isinstance(r, Mapping)]
    distilled = [e for r in runs for e in _entries(r, "distill_metrics")]
    experiences = sum(_int(e.get("experiences")) for e in distilled)
    # The outcome each session's experiences were stored with (a MIRIX before it
    # stores none), and the chunks whose stated succeeded / failed outcome the
    # transcript did not show (stored unclear) or only the agent's own words showed.
    outcomes = Counter(
        str(e["session_outcome"])
        for e in distilled
        if e.get("session_outcome") is not None
    )
    writes = sum(
        _int(e.get("writes")) for r in runs for e in _entries(r, "curator_metrics")
    )
    deleted = []
    for skill_id in snapshot.get("deleted_skills") or ():
        entry = _mapping(history.get(str(skill_id)))
        versions = [v for v in entry.get("versions") or () if isinstance(v, Mapping)]
        deleted.append(
            {
                "id": str(skill_id),
                "name": entry.get("name"),
                "versions": len(versions),
                "changes": [str(v.get("change_type")) for v in versions],
            }
        )
    return {
        "skills": len(skills),
        "skill_list": skills,
        "deleted_skills": deleted,
        "skills_without_vector": list(snapshot.get("skills_without_vector") or ()),
        "experiences": experiences,
        "session_outcomes": dict(sorted(outcomes.items())),
        "outcomes_unverified": sum(
            _int(e.get("outcomes_unverified")) for e in distilled
        ),
        "outcomes_self_reported": sum(
            _int(e.get("outcomes_self_reported")) for e in distilled
        ),
        "curator_writes": writes,
        "learning_runs": len(runs),
        "runs_by_status": dict(Counter(str(r.get("status")) for r in runs)),
        "run_errors": [str(r["error"]) for r in runs if r.get("error")][:ERROR_LIMIT],
        "curator_tool_exceptions": learning_warnings(runs)[:ERROR_LIMIT],
        "experience_pool": mirix_pool(snapshot),
        "learning_context": learning_context(runs),
        "counters": dict(_mapping(snapshot.get("counters"))),
    }


def context_warnings(
    learned: Mapping[str, Any] | None,
    benchmark: str,
    record: Mapping[str, Any] | None,
) -> list[str]:
    """What in a MIRIX cell's ``learning_context`` says the agent context did not
    reach the distiller as the attempt sent it: learned sessions MIRIX
    distilled without one, although the cell's benchmark offers it (``record``: the
    attempt's ``agent_context``, ``report.cell_agent_context``; None, an attempt from
    before it was recorded: :data:`AGENT_CONTEXT_BENCHMARKS`), and a stored length that
    is not the attempt's. Nothing for a MIRIX without a ``learning_context``."""
    context = _mapping((learned or {}).get("learning_context"))
    if not context:
        return []
    sent = _int(record.get("chars")) if record is not None else None
    offered = sent > 0 if sent is not None else benchmark in AGENT_CONTEXT_BENCHMARKS
    if not offered:
        return []
    warnings = []
    missing = [str(s) for s in context.get("without_agent_context") or ()]
    if missing:
        named = ", ".join(missing[:SESSION_LIMIT])
        more = len(missing) - SESSION_LIMIT
        warnings.append(
            f"{len(missing)} of {_int(context.get('sessions'))} learned session(s) were "
            "distilled without the agent context although the benchmark offers it "
            f"(the agent context did not reach MIRIX's distiller): {named}"
            + (f" and {more} more" if more > 0 else "")
        )
    if sent:
        other = [_int(n) for n in context.get("agent_context_chars") or () if n != sent]
        if other:
            warnings.append(
                "MIRIX stored an agent context of "
                + ", ".join(str(n) for n in other)
                + f" chars on learned sessions; the attempt sent {sent}"
            )
    return warnings


def learned_reasoningbank(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Memory items (their titles) and the trajectories they were extracted from."""
    entries = [e for e in snapshot.get("entries") or () if isinstance(e, Mapping)]
    titles = [
        str(item.get("title"))
        for entry in entries
        for item in entry.get("memory_items") or ()
        if isinstance(item, Mapping)
    ]
    return {
        "memory_items": len(titles),
        "titles": titles,
        "entries": len(entries),
        "entries_by_status": dict(Counter(str(e.get("status")) for e in entries)),
        "counters": dict(_mapping(snapshot.get("counters"))),
    }


def learned_awm(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The induced workflow and how it came about."""
    workflow = str(snapshot.get("workflow") or "")
    counters = _mapping(snapshot.get("counters"))
    inductions = snapshot.get("inductions")
    return {
        "workflow_chars": len(workflow),
        "workflow_empty": not workflow.strip(),
        "inductions": len(inductions)
        if isinstance(inductions, list)
        else _int(counters.get("inductions")),
        "workflows_written": _int(counters.get("workflows_written")),
        "induction_model_failures": _int(counters.get("induction_model_failures")),
        "pool": len(snapshot.get("pool") or ()),
        "counters": dict(counters),
    }


LEARNERS = {
    "mirix": learned_mirix,
    "reasoningbank": learned_reasoningbank,
    "awm": learned_awm,
}


def nothing_learned(method: str, learned: Mapping[str, Any] | None) -> str | None:
    """Why a learning method's frozen memory holds nothing (None: it learned something,
    or the method learns nothing by design)."""
    if method == "none":
        return None
    if learned is None:
        return "no frozen memory: learning did not finish"
    if method == "mirix" and not learned["skills"]:
        runs = ", ".join(
            f"{s} {n}" for s, n in sorted(learned["runs_by_status"].items())
        )
        return (
            f"0 skills after the freeze ({learned['experiences']} experiences, "
            f"{learned['learning_runs']} learning runs{': ' + runs if runs else ''})"
        )
    if method == "reasoningbank" and not learned["memory_items"]:
        return f"0 memory items ({learned['entries']} bank entries)"
    if method == "awm" and learned["workflow_empty"]:
        return (
            f"empty workflow ({learned['inductions']} inductions, "
            f"{learned['induction_model_failures']} model failures)"
        )
    if method not in LEARNERS:
        return f"unknown method {method!r}: cannot tell what it learned"
    return None


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


def _attempt_state(cell: CellData) -> str | None:
    attempt = cell.attempt(cell.reported)
    return str(attempt.state.get("state")) if attempt else None


def summarize_cell(cell: CellData) -> dict[str, Any]:
    """What the cell's method learned, injected at test time and cost (JSON-ready)."""
    snapshot = (cell.frozen or {}).get("snapshot")
    learner = LEARNERS.get(cell.method)
    learned: dict[str, Any] | None
    if cell.method == "none":
        learned = {}
    elif isinstance(snapshot, Mapping) and learner is not None:
        learned = learner(snapshot)
    elif isinstance(snapshot, Mapping):
        learned = {"snapshot_keys": sorted(snapshot)}
    else:
        learned = None
    nothing = nothing_learned(cell.method, learned)
    injections = cell_injections(cell).get("test")
    costs = cell_costs(cell)
    discarded = costs["learning_discarded"]
    resumes = cell_resumes(cell)
    warnings = []
    if (
        nothing is None
        and cell.method != "none"
        and injections
        and injections["items"]
        and injections["empty"] == injections["items"]
    ):
        warnings.append(
            f"learned something but injected nothing in all {injections['items']} "
            "test items"
        )
    if costs["unfinished_calls"]:
        warnings.append(
            f"{costs['unfinished_calls']} calls without a final ledger line (cost unknown)"
        )
    if costs["ledgers_missing"]:
        warnings.append(f"attempts without a ledger: {costs['ledgers_missing']}")
    if discarded["unknown_cost_calls"]:
        warnings.append(
            f"{discarded['unknown_cost_calls']} discarded learning calls were cut off "
            "(billed at a cost the ledger never learns)"
        )
    if cell.method == "mirix" and learned:
        warnings += pool_warnings(learned.get("experience_pool"))
        warnings += context_warnings(
            learned, str(cell.benchmark), cell_agent_context(cell)
        )
    tolerated, refusals = _accepted(cell.method, snapshot)
    if tolerated and tolerated.get("counts"):
        warnings.append(
            "MIRIX learning went on past "
            + ", ".join(f"{k} {v}" for k, v in sorted(tolerated["counts"].items()))
            + " (recorded, not stopped)"
        )
    if any(refusals.values()):
        warnings.append(
            "the provider refused learning calls, counted and gone on from: "
            + ", ".join(f"{k} {v}" for k, v in refusals.items() if v)
        )
    return {
        "cell_id": cell.cell_id,
        "benchmark": cell.benchmark,
        "model_key": cell.model_key,
        "method": cell.method,
        "split_mode": cell.split_mode,
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "attempt_state": _attempt_state(cell),
        "frozen": isinstance(snapshot, Mapping),
        "learned": learned,
        "nothing_learned": nothing,
        "test_injections": injections,
        # What the method accepted and went on from (snapshot ``tolerated``).
        "tolerated": tolerated,
        "refusals": refusals,
        "host": cell_host(cell),
        "resumes": len(resumes),
        "resume_checkpoints": [r.get("checkpoint") for r in resumes],
        "cost": {
            "learning": costs["learning"]["cost"],
            "learning_calls": costs["learning"]["calls"],
            "learning_discarded": discarded["cost"],
            "learning_discarded_calls": discarded["calls"],
            "test": costs["test"]["cost"],
            "test_calls": costs["test"]["calls"],
            "other_attempts": costs["other_attempts"]["cost"],
            "total": costs["total_cost"],
            "by_role": {
                side: {r: v["cost"] for r, v in costs[side]["by_role"].items()}
                for side in ("learning", "test")
            },
            "unknown_cost_calls": costs["unknown_cost_calls"],
            "unknown_cost_calls_by_side": {
                side: costs[side].get("unknown_cost_calls", 0)
                for side in ("learning", "test")
            },
            "not_counted": uncounted_costs(costs),
        },
        "warnings": warnings,
    }


def _accepted(
    method: str, snapshot: Any
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """MIRIX's ``tolerated`` block (None when absent: nothing was tolerated) and the
    refusal counters of AWM / ReasoningBank (a missing counter is 0), from the frozen
    snapshot."""
    snapshot = _mapping(snapshot)
    tolerated = snapshot.get("tolerated")
    counters = _mapping(snapshot.get("counters"))
    return (
        dict(tolerated) if isinstance(tolerated, Mapping) else None,
        {k: _int(counters.get(k)) for k in METHOD_REFUSAL_COUNTERS.get(method, ())},
    )


def removed_cell(cell: CellData) -> dict[str, Any]:
    """A removed method's cell as the summary lists it (JSON-ready, no absolute path):
    what it is, where it stopped, and what its ledgers cost. What its method learned is
    not read: the method left the study (``study.REMOVED_METHODS``). Its method is read
    as ``study.split_cells`` read it (``study.method_of``: a definition without one is
    no crash)."""
    costs = cell_costs(cell)
    method = str(study.method_of(cell))
    return {
        "cell_id": cell.cell_id,
        "cell_dir": cell.paths.root.name,
        "benchmark": cell.benchmark,
        "model_key": cell.model_key,
        "method": method,
        "split_mode": cell.split_mode,
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "attempt_state": _attempt_state(cell),
        "decision": study.REMOVED_METHODS[method].decision,
        "cost": {
            "total": costs["total_cost"],
            "unknown_cost_calls": costs["unknown_cost_calls"],
            "unfinished_calls": costs["unfinished_calls"],
            "not_counted": uncounted_costs(costs),
        },
        "on_disk": True,
        "deleted": None,
    }


def deleted_cell(row: Mapping[str, Any], manifest: Any) -> dict[str, Any]:
    """A removed method's cell whose data was deleted, as its removal manifest recorded
    it (``tools/remove_method_data.py``): the same fields as :func:`removed_cell`."""
    costs = _mapping(row.get("costs"))
    method = str(row.get("method"))
    states = _mapping(row.get("attempt_states"))
    data = manifest.data
    return {
        "cell_id": row.get("cell_id"),
        "cell_dir": row.get("cell_dir"),
        "benchmark": row.get("benchmark"),
        "model_key": row.get("model_key"),
        "method": method,
        "split_mode": row.get("split_mode"),
        "status": row.get("status"),
        "complete": row.get("complete"),
        "reported_attempt": row.get("reported_attempt"),
        "attempt_state": states.get(str(row.get("reported_attempt"))),
        "decision": study.REMOVED_METHODS[method].decision
        if study.is_removed(method)
        else data.get("decision"),
        "cost": {
            "total": costs.get("total_cost"),
            "unknown_cost_calls": _int(costs.get("unknown_cost_calls")),
            "unfinished_calls": _int(costs.get("unfinished_calls")),
            "not_counted": row.get("not_counted") or uncounted_costs(costs),
        },
        "on_disk": False,
        "deleted": {
            "removal_id": manifest.removal_id,
            "at": data.get("completed_at") or data.get("created_at"),
            "host": _mapping(data.get("host")).get("hostname"),
        },
    }


def deleted_cells(
    runs_dir: Path,
    gateway_runs_dirs: Sequence[Path],
    split_mode: str | None,
    on_disk: Sequence[CellData],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The deleted cells of every removal manifest of ``runs_dir`` (its host mirrors
    included) and of ``gateway_runs_dirs``, one per removal, of ``split_mode``; a cell
    still on disk (an interrupted deletion) is left to :func:`removed_cell`. Also the
    manifests' problems and the rows that cannot stand in for their cell
    (``remove_method_data.cell_row_problem``), which are left out (blocking)."""
    seen: set[str] = set()
    live = {(c.paths.root.name, c.manifest.get("instance")) for c in on_disk}
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for directory in [runs_dir, *gateway_runs_dirs]:
        for manifest in load_manifests(directory):
            if manifest.removal_id in seen:
                continue
            seen.add(manifest.removal_id)
            if manifest.problems:
                problems += manifest.problems
                continue
            for row in manifest.data.get("cells") or ():
                problem = cell_row_problem(row)
                if problem is not None:
                    problems.append(f"{manifest.path}: {problem}")
                    continue
                if split_mode is not None and row.get("split_mode") != split_mode:
                    continue
                if (row.get("cell_dir"), row.get("instance")) in live:
                    continue
                rows.append(deleted_cell(row, manifest))
    return rows, problems


def summarize(
    runs_dir: Path,
    *,
    split_mode: str | None = None,
    gateway_runs_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Every cell of ``runs_dir`` (``split_mode`` None: protocol and smoke cells); the
    ledgers are looked up as ``report.load_cells`` does (``gateway_runs_dirs``: where
    else this host keeps gateway run directories).

    The cells of a method the study removed (``study.REMOVED_METHODS``) are set aside:
    ``removed`` lists them with their status and cost, and ``removed_methods`` says
    why; they are in no other key (``total_cost`` is the study's cells',
    ``total_cost_all_cells`` adds theirs) and never make the summary fail
    (:func:`main`)."""
    study_cells, removed = study.split_cells(
        load_cells(runs_dir, split_mode=split_mode, gateway_runs_dirs=gateway_runs_dirs)
    )
    cells = [summarize_cell(c) for c in study_cells]
    gone, problems = deleted_cells(runs_dir, gateway_runs_dirs, split_mode, removed)
    set_aside = sorted(
        [removed_cell(c) for c in removed] + gone,
        key=lambda c: (str(c["cell_id"]), str(c["cell_dir"])),
    )
    nothing = [c["cell_id"] for c in cells if c["nothing_learned"]]
    total = round(sum(float(c["cost"]["total"] or 0) for c in cells), 10)
    removed_cost = round(sum(float(c["cost"]["total"] or 0) for c in set_aside), 10)
    return {
        "runs_dir": str(runs_dir),
        "cells": cells,
        "nothing_learned": nothing,
        "total_cost": total,
        "unknown_cost_calls": sum(int(c["cost"]["unknown_cost_calls"]) for c in cells),
        "resumes": sum(int(c["resumes"]) for c in cells),
        "learning_discarded_cost": round(
            sum(float(c["cost"]["learning_discarded"] or 0) for c in cells), 10
        ),
        "removed": set_aside,
        "removed_methods": study.removed_methods_record(
            (c["method"], c["cell_id"]) for c in set_aside
        ),
        "removed_cost": removed_cost,
        # Not in removed_cost either, so the all-cells line names them.
        "removed_unknown_cost_calls": sum(
            int(c["cost"]["unknown_cost_calls"]) for c in set_aside
        ),
        "removed_unfinished_calls": sum(
            int(c["cost"]["unfinished_calls"]) for c in set_aside
        ),
        "total_cost_all_cells": round(total + removed_cost, 10),
        "removed_deleted": sum(1 for c in set_aside if not c["on_disk"]),
        "removed_manifest_problems": problems,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _usd(value: Any) -> str:
    return f"${float(value or 0):.4f}"


def _listed(names: Sequence[str], indent: str) -> list[str]:
    lines = [f"{indent}- {name}" for name in names[:LIST_LIMIT]]
    if len(names) > LIST_LIMIT:
        lines.append(f"{indent}  ... and {len(names) - LIST_LIMIT} more")
    return lines


def _learned_lines(method: str, learned: Mapping[str, Any] | None) -> list[str]:
    if method == "none":
        return ["  learned: nothing, by design (no-memory baseline)"]
    if learned is None:
        return ["  learned: (no frozen memory)"]
    if method == "mirix":
        runs = ", ".join(
            f"{s} {n}" for s, n in sorted(learned["runs_by_status"].items())
        )
        lines = [
            f"  learned: {learned['skills']} skills, {learned['experiences']} experiences, "
            f"{learned['curator_writes']} curator writes; {learned['learning_runs']} "
            f"learning runs ({runs or 'none'})"
        ]
        lines += _listed(
            [
                f"{s['name']} (v{s['version']}, {s['versions']} versions: "
                f"{', '.join(s['changes']) or '-'})"
                for s in learned["skill_list"]
            ],
            "    ",
        )
        lines += _listed(
            [
                f"deleted: {s['name']} ({s['versions']} versions: "
                f"{', '.join(s['changes']) or '-'})"
                for s in learned["deleted_skills"]
            ],
            "    ",
        )
        if learned.get("session_outcomes"):
            lines.append(
                "    session outcomes: "
                + ", ".join(f"{o} {n}" for o, n in learned["session_outcomes"].items())
                + f" ({_int(learned.get('outcomes_unverified'))} stated outcome the "
                "transcript did not show, "
                f"{_int(learned.get('outcomes_self_reported'))} only the agent's own "
                "words showed)"
            )
        if learned["skills_without_vector"]:
            lines.append(
                "    without a retrieval vector: "
                + ", ".join(learned["skills_without_vector"])
            )
        lines += [f"    run error: {e[:200]}" for e in learned["run_errors"]]
        lines += [
            f"    curator tool exception: {e[:200]}"
            for e in learned.get("curator_tool_exceptions") or ()
        ]
        lines.append(_pool_line(learned.get("experience_pool")))
        lines += context_lines(learned.get("learning_context"))
        return lines
    if method == "reasoningbank":
        by = ", ".join(
            f"{s} {n}" for s, n in sorted(learned["entries_by_status"].items())
        )
        return [
            f"  learned: {learned['memory_items']} memory items from "
            f"{learned['entries']} trajectories ({by or 'none'})"
        ] + _listed(learned["titles"], "    ")
    if method == "awm":
        return [
            f"  learned: workflow of {learned['workflow_chars']} chars; "
            f"{learned['inductions']} inductions, {learned['workflows_written']} written, "
            f"{learned['induction_model_failures']} model failures"
        ]
    return [f"  learned: unknown method; snapshot keys {learned.get('snapshot_keys')}"]


def _pool_line(pool: Mapping[str, Any] | None) -> str:
    """The MIRIX experience pool in one line."""
    if not pool:
        return "    experience pool: not reported (a MIRIX without the experience pool)"
    d = _mapping(pool.get("dispositions"))
    retired = _mapping(pool.get("retired"))
    active = pool.get("active_at_freeze")
    return (
        f"    experience pool: {'?' if active is None else active} active at the freeze "
        f"({pool.get('entered')} entered; retired: used {_int(retired.get('used'))}, "
        f"covered {_int(retired.get('covered'))}, ignored "
        f"{_int(retired.get('ignored'))}, merged {_int(retired.get('merged'))}, "
        f"unembeddable {_int(retired.get('unembeddable'))}); new experiences: used "
        f"{_int(d.get('used'))}, "
        f"covered {_int(d.get('covered'))}, not worth {_int(d.get('not_worth'))}, "
        f"deferred {_int(d.get('deferred'))} ({_int(d.get('defaulted'))} by default), "
        f"unaddressed {_int(d.get('unaddressed'))}; earlier lessons shown "
        f"{_int(pool.get('history_pushed'))} (cited {_int(pool.get('history_cited'))}, "
        f"covered {_int(pool.get('history_covered'))}, not worth "
        f"{_int(pool.get('history_not_worth'))}, ignored "
        f"{_int(pool.get('history_ignored'))}), recurrences "
        f"{_int(pool.get('recurrences'))}; citations without a verified quote "
        f"{_int(pool.get('citations_unverified'))}"
    )


def _counts(values: Any) -> str:
    """``yes 3, unclear 1`` (``-`` for none)."""
    return ", ".join(f"{k} {_int(v)}" for k, v in _mapping(values).items()) or "-"


def context_lines(context: Mapping[str, Any] | None) -> list[str]:
    """MIRIX's ``learning_context`` in a few lines: the learned sessions
    and the agent context, the skills served, the feedback stored for them (each
    skill's too) and the usage records the curator batches showed."""
    if not context:
        return ["    session context: not reported (a MIRIX that records none)"]
    chars = ", ".join(str(n) for n in context.get("agent_context_chars") or ()) or "-"
    per = ", ".join(
        f"{k} x{_int(n)}"
        for k, n in _mapping(context.get("served_per_session")).items()
    )
    feedback = _mapping(context.get("feedback"))
    usage = _mapping(context.get("usage"))
    lines = [
        f"    session context: {_int(context.get('with_agent_context'))} of "
        f"{_int(context.get('sessions'))} learned sessions with the agent context "
        f"({chars} chars), {_int(context.get('context_cut'))} cut to fit "
        f"({_int(context.get('context_truncated_tokens'))} tokens); "
        f"{_int(context.get('served_skills'))} skills served in "
        f"{_int(context.get('served_sessions'))} sessions (per session: {per or '-'}; "
        f"{_int(context.get('served_skills_hidden'))} left out of the context)",
        f"    skill feedback: entries for {_int(feedback.get('given'))} of "
        f"{_int(feedback.get('served'))} served ({_int(feedback.get('not_shown'))} not "
        f"shown); followed {_counts(feedback.get('followed'))}; effect "
        f"{_counts(feedback.get('effect'))}; stated without a verifying quote: followed "
        f"{_int(feedback.get('followed_unverified'))}, effect "
        f"{_int(feedback.get('effect_unverified'))}",
    ]
    if not context.get("skills_recorded", True):
        lines.append(
            "      (each skill's values not recorded in the run: a MIRIX before it "
            "recorded them; skill_session_feedback holds them)"
        )
    lines += _listed(
        [
            f"{s.get('name')}: served {_int(s.get('served'))}; followed "
            f"{_counts(s.get('followed'))}; effect {_counts(s.get('effect'))}"
            for s in context.get("skills") or ()
            if isinstance(s, Mapping)
        ],
        "      ",
    )
    skipped = _counts(usage.get("skipped"))
    lines.append(
        f"    curator usage records: shown in {_int(usage.get('shown'))} of "
        f"{_int(usage.get('batches'))} batches ({_int(usage.get('records'))} records, "
        f"{_int(usage.get('served'))} of skills served there; "
        f"{_int(usage.get('compact'))} batch(es) with one line per skill, "
        f"{_int(usage.get('left_out'))} skills left out for room, "
        f"{_int(usage.get('tokens'))} tokens; none shown: {skipped}); citations "
        f"without a verified quote {_int(context.get('citations_unverified'))}"
    )
    return lines


def pool_warnings(pool: Mapping[str, Any] | None) -> list[str]:
    """What in a MIRIX experience pool deserves a look: batches that showed
    no earlier lessons because the pool's embedding failed or the pool was off, pool
    experiences a failed embedding left out of a batch's earlier lessons, and pool
    metrics that do not add up."""
    if not pool:
        return []
    warnings = []
    skipped = _mapping(pool.get("history_skipped"))
    failed = _int(skipped.get("embedding_failed"))
    if failed:
        warnings.append(
            f"{failed} curator batch(es) showed no earlier lessons because the "
            "experience pool's embedding failed: the attempt should have restarted"
        )
    left_out = _int(_mapping(pool.get("history")).get("embedding_failed"))
    if left_out and not failed:
        warnings.append(
            f"{left_out} experience-pool embedding(s) failed, leaving pool experiences "
            "out of the earlier lessons: the attempt should have restarted"
        )
    off = {r: _int(skipped.get(r)) for r in POOL_SKIPS_NAMED if r != "embedding_failed"}
    if any(off.values()):
        warnings.append(
            "the experience pool was off in "
            + ", ".join(f"{n} batch(es) ({r})" for r, n in off.items() if n)
            + ": not MIRIX's default"
        )
    unaccounted = pool.get("unaccounted")
    if unaccounted not in (0, None):
        warnings.append(
            f"the experience pool's metrics do not add up: {pool.get('entered')} "
            f"entered, {sum(_int(v) for v in _mapping(pool.get('retired')).values())} "
            f"retired, {pool.get('active_at_freeze')} active at the freeze"
        )
    return warnings


def _host_text(host: Mapping[str, Any]) -> str:
    """``hostname (system machine, id 1234abcd)``: the machine id is the host's identity,
    the hostname only a name."""
    machine_id = host.get("machine_id")
    tail = f", id {str(machine_id)[:8]}" if machine_id else ""
    return (
        f"{host.get('hostname') or '?'} ({host.get('system') or '?'} "
        f"{host.get('machine') or '?'}{tail})"
    )


def render_text(summary: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for cell in summary["cells"]:
        state = cell["attempt_state"] or "-"
        host = cell.get("host")
        lines.append(
            f"{cell['cell_id']}  [{cell['status'] or '-'}; attempt "
            f"{cell['reported_attempt']} {state}"
            + (f"; host {_host_text(host)}" if host else "")
            + "]"
        )
        lines += _learned_lines(cell["method"], cell["learned"])
        if cell["nothing_learned"]:
            lines.append(f"  NOTHING LEARNED: {cell['nothing_learned']}")
        inj = cell["test_injections"]
        if inj and inj["items"]:
            tokens = inj["tokens"]
            lines.append(
                f"  test injections: {inj['items']} items, {inj['empty']} empty; tokens "
                f"mean {tokens['mean']:.0f}, median {tokens['median']:.0f}, "
                f"max {tokens['max']:.0f}, total {tokens['total']:.0f}"
            )
        else:
            lines.append("  test injections: none recorded")
        cost = cell["cost"]
        lines.append(
            f"  cost: learning {_usd(cost['learning'])} ({cost['learning_calls']} calls), "
            f"test {_usd(cost['test'])} ({cost['test_calls']} calls), invalidated "
            f"attempts {_usd(cost['other_attempts'])}, total {_usd(cost['total'])}"
            + (
                f"; not counted: {cost['not_counted']}"
                if cost.get("not_counted", "-") != "-"
                else ""
            )
        )
        if cell.get("resumes"):
            checkpoints = ", ".join(str(c) for c in cell["resume_checkpoints"])
            lines.append(
                f"  learning resumes: {cell['resumes']} (from {checkpoints}); discarded "
                f"{_usd(cost['learning_discarded'])} "
                f"({cost['learning_discarded_calls']} calls, part of learning)"
            )
        tolerated = cell.get("tolerated") or {}
        if tolerated.get("problems"):
            lines += [
                f"  accepted ({p.get('reason')}, after {p.get('rollbacks')} "
                f"rollback(s)): {str(p.get('text'))[:200]}"
                for p in tolerated["problems"][:ERROR_LIMIT]
            ]
        lines += [f"  warning: {w}" for w in cell["warnings"]]
        lines.append("")
    removed = summary.get("removed") or []
    for method, entry in sorted((summary.get("removed_methods") or {}).items()):
        cells = [c for c in removed if c["method"] == method]
        if not cells:
            continue
        lines.append(
            f"removed from the study, set aside (not summarized): {method}, "
            f"{entry['decision']}: {entry['reason']}"
        )
        for cell in cells:
            cost = cell["cost"]
            lines.append(
                f"  {cell['cell_id']}  [{cell['status'] or '-'}; attempt "
                f"{cell['reported_attempt']} {cell['attempt_state'] or '-'}] cost "
                f"{_usd(cost['total'])}"
                + (
                    f"; not counted: {cost['not_counted']}"
                    if cost.get("not_counted", "-") != "-"
                    else ""
                )
                + (
                    f"; data deleted {cell['deleted'].get('at')} on "
                    f"{cell['deleted'].get('host')} (removal "
                    f"{cell['deleted'].get('removal_id')}; cost from its manifest)"
                    if cell.get("deleted")
                    else ""
                )
            )
        lines.append("")
    for problem in summary.get("removed_manifest_problems") or ():
        lines.append(f"warning: a removal manifest could not be read: {problem}")
    nothing = summary["nothing_learned"]
    unknown = summary.get("unknown_cost_calls") or 0
    resumes = summary.get("resumes") or 0
    removed_unknown = [
        f"{n} {what}"
        for n, what in (
            (int(summary.get("removed_unknown_cost_calls") or 0), "cut-off"),
            (int(summary.get("removed_unfinished_calls") or 0), "unfinished"),
        )
        if n
    ]
    lines.append(
        f"{len(summary['cells'])} cells, total cost {_usd(summary['total_cost'])}"
        + (
            f" (not counting {unknown} cut-off call(s) billed at an unknown cost)"
            if unknown
            else ""
        )
        + (
            f"; {resumes} learning resume(s) discarded "
            f"{_usd(summary.get('learning_discarded_cost'))}"
            if resumes
            else ""
        )
        + (
            f"; {len(removed)} cell(s) of removed methods set aside, "
            f"{_usd(summary.get('removed_cost'))} more"
            + (
                f" (not counting their {' and '.join(removed_unknown)} call(s) of "
                "unknown cost)"
                if removed_unknown
                else ""
            )
            + f" ({_usd(summary.get('total_cost_all_cells'))} with them)"
            if removed
            else ""
        )
        + "; "
        + (
            f"{len(nothing)} learned nothing: {', '.join(nothing)}"
            if nothing
            else "every learning method learned something"
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    parser.add_argument(
        "--split-mode",
        default="all",
        choices=("protocol", "smoke", "all"),
        help="which cells to summarize (default: all)",
    )
    parser.add_argument(
        "--gateway-runs-dir",
        type=Path,
        action="append",
        default=[],
        help="where else this host keeps gateway run directories (repeatable; the "
        "ledgers are looked for relative to --runs-dir first)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON, not text")
    args = parser.parse_args(argv)
    summary = summarize(
        args.runs_dir,
        split_mode=None if args.split_mode == "all" else args.split_mode,
        gateway_runs_dirs=args.gateway_runs_dir,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        sys.stdout.write(render_text(summary))
    if not summary["cells"]:
        removed = summary["removed"]
        print(
            f"no cells in {args.runs_dir}"
            + (
                f" but {len(removed)} of methods removed from the study, set aside: "
                + ", ".join(c["cell_id"] for c in removed)
                if removed
                else ""
            ),
            file=sys.stderr,
        )
        return EXIT_NO_CELLS
    return EXIT_NOTHING_LEARNED if summary["nothing_learned"] else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
