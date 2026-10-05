"""tools/remove_method_data.py: a removed method's leftover cells and gateway runs are
deleted only after their spend is written to a manifest, and the report counts the same
money before and after (costs, spend windows, the union of two hosts). The report's
reading of the manifests is tested here too."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol, report, study
from evals.skill_bench.cell_records import write_json_atomic
from evals.skill_bench.tests.conftest import REMOVED_METHOD
from evals.skill_bench.tests.test_report import (
    FP,
    _call,
    _host_runs,
    _row,
    _timed,
    _write_balance,
    make_cell,
    shop,
)
from evals.skill_bench.tools import remove_method_data as tool
from evals.skill_bench.tools import summarize_learning

START = "2026-09-29T05:00:00+00:00"
AT = "2026-09-29T06:00:00+00:00"
END = "2026-09-29T08:00:00+00:00"
#: The method the tests register as removed (the ``removed_method`` fixture).
REMOVED = REMOVED_METHOD
REMOVED_WEB = f"webshop-luna-{REMOVED}"
REMOVED_ALF56 = f"alfworld-luna56-{REMOVED}"
REMOVED_ALF = f"alfworld-luna-{REMOVED}"
REMOVED_WEB56 = f"webshop-luna56-{REMOVED}"
#: A gateway run of a smoke cell no directory holds.
ORPHAN = f"alfworld-luna56-{REMOVED}-smoke-l2-t2-a01-deadbeef"
PROTECTED = ("_blobs", "_gateway", "_schedule", "_supervise")


def _tree_digest(path: Path) -> str:
    """Every file under ``path`` (names and bytes), or '-' when it does not exist."""
    if not path.exists():
        return "-"
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()


def _timed_call(role: str, phase: str, cost: float, **kw: Any) -> dict[str, Any]:
    return _timed(_call(role, phase, cost, **kw), AT, completed_at=AT)


@pytest.fixture
def deletion_decided(removed_method: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """:data:`REMOVED` removed from the study (``removed_method``) with its deletion decided
    (``DELETION_DECISIONS``), for one test."""
    monkeypatch.setattr(
        tool,
        "DELETION_DECISIONS",
        {removed_method: {"decision": "example decision", "quote": "delete it"}},
    )
    return removed_method


def _leftover_cells(runs: Path) -> dict[str, Path]:
    """The removed method's cells a runs directory still holds: a complete WebShop
    cell, a complete ALFWorld cell of luna56, and an ALFWorld cell stopped mid-learning.
    Their ledgers: $0.05, $0.01 and $0.02."""
    web = make_cell(
        runs,
        "webshop",
        REMOVED,
        test=shop("w", [1.0, 0.0, 1.0, 0.0]),
        calls=[
            _call("actor", "learning", 0.02, item="l1"),
            _call("learner", "learning", 0.02, item="l1"),
            _call("actor", "test", 0.01, item="u1"),
        ],
        snapshot={"method": REMOVED, "notes": ""},
    )
    alf56 = make_cell(
        runs,
        "alfworld",
        REMOVED,
        model_key="luna56",
        test=[_row("a1", 1.0)],
        calls=[
            _call("actor", "test", 0.01, item="u1", model=protocol.MODELS["luna56"])
        ],
    )
    alf = make_cell(
        runs,
        "alfworld",
        REMOVED,
        test=[],
        status="interrupted",
        calls=[_call("actor", "learning", 0.02, item="l1")],
    )
    return {REMOVED_WEB: web, REMOVED_ALF56: alf56, REMOVED_ALF: alf}


@pytest.fixture
def removed_runs(tmp_path: Path, deletion_decided: str) -> Path:
    """A host's runs directory: study cells (WebShop, ALFWorld), the removed method's
    leftovers (three cells, a WebShop one of luna56, an orphan gateway run), the shared
    directories and a balance record covering every call."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "webshop",
        "none",
        test=shop("w", [1.0]),
        calls=[_timed_call("actor", "test", 0.01, item="u1")],
    )
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[_timed_call("actor", "test", 0.02, item="u1")],
    )
    _leftover_cells(runs)
    make_cell(
        runs,
        "webshop",
        REMOVED,
        model_key="luna56",
        test=shop("w", [1.0]),
        calls=[
            _timed_call(
                "actor", "test", 0.04, item="u1", model=protocol.MODELS["luna56"]
            )
        ],
    )
    orphan = runs / ORPHAN
    orphan.mkdir()
    (orphan / "gateway.jsonl").write_text(
        json.dumps(_timed_call("actor", "learning", 0.07, item="l1")) + "\n"
    )
    (runs / "_blobs").mkdir()
    (runs / "_blobs" / "ab").write_bytes(b"blob")
    (runs / "_schedule").mkdir()
    (runs / "_schedule" / "state.json").write_text("{}")
    (runs / "_supervise").mkdir()
    (runs / "_supervise" / "events.jsonl").write_text("{}\n")
    budget = {"budget_usd": 50.0, "started_at": START, "start_usage_usd": 100.0}
    _write_balance(
        runs,
        [
            {"ts": START, "usage": 100.0, "budget": {**budget, "spent_usd": 0.0}},
            {"ts": END, "usage": 100.5, "budget": {**budget, "spent_usd": 0.5}},
        ],
    )
    return runs


REMOVED_ENTRIES = sorted(
    [
        REMOVED_ALF,
        REMOVED_WEB,
        REMOVED_ALF56,
        REMOVED_WEB56,
        f"{REMOVED_ALF}-a01",
        f"{REMOVED_WEB}-a01",
        f"{REMOVED_ALF56}-a01",
        f"{REMOVED_WEB56}-a01",
        ORPHAN,
    ]
)


def _run(runs: Path, *extra: str, capsys: Any = None) -> tuple[int, dict[str, Any]]:
    code = tool.main(["--runs-dir", str(runs), "--method", REMOVED, *extra])
    printed = json.loads(capsys.readouterr().out) if capsys is not None else {}
    return code, printed


def _manifest(runs: Path) -> tuple[Path, dict[str, Any]]:
    (path,) = sorted((runs / tool.REMOVED_DIR / REMOVED).glob("*/manifest.json"))
    return path, json.loads(path.read_text())


def test_a_dry_run_writes_nothing_and_prints_the_plan(removed_runs, capsys):
    before = _tree_digest(removed_runs)
    code, printed = _run(removed_runs, capsys=capsys)
    assert code == tool.EXIT_OK and printed["dry_run"] is True
    assert printed["refusals"] == []
    plan = printed["plan"]
    assert sorted(c["cell_id"] for c in plan["cells"]) == sorted(
        [REMOVED_ALF, REMOVED_WEB, REMOVED_ALF56, REMOVED_WEB56]
    )
    assert {r["run_id"] for r in plan["gateway_runs"]} == set(REMOVED_ENTRIES) - {
        REMOVED_ALF,
        REMOVED_WEB,
        REMOVED_ALF56,
        REMOVED_WEB56,
    }
    assert plan["totals"]["cost_total"] == pytest.approx(
        0.02 + 0.05 + 0.01 + 0.04 + 0.07
    )
    assert _tree_digest(removed_runs) == before
    assert not (removed_runs / tool.REMOVED_DIR).exists()


def test_it_deletes_exactly_the_method_cells_and_their_gateway_runs(
    removed_runs, capsys
):
    kept = {
        name: _tree_digest(removed_runs / name)
        for name in [
            *PROTECTED,
            "webshop-luna-none",
            "webshop-luna-none-a01",
            "alfworld-luna-none",
            "alfworld-luna-none-a01",
        ]
    }
    code, printed = _run(
        removed_runs, "--yes", "--note", "delete the leftovers", capsys=capsys
    )
    assert code == tool.EXIT_OK and printed["state"] == tool.STATE_COMPLETE
    assert sorted(printed["deleted"]) == REMOVED_ENTRIES
    for entry in REMOVED_ENTRIES:
        assert not (removed_runs / entry).exists(), entry
    for name, digest in kept.items():
        assert _tree_digest(removed_runs / name) == digest, name
    path, manifest = _manifest(removed_runs)
    assert manifest["state"] == tool.STATE_COMPLETE and manifest["completed_at"]
    assert manifest["note"] == "delete the leftovers"
    assert not (path.parent / tool.TRASH_DIR).exists()
    index = json.loads((removed_runs / tool.INDEX_FILE).read_text())
    (entry,) = index["removals"]
    assert entry["state"] == tool.STATE_COMPLETE
    assert entry["manifest"] == f"{REMOVED}/{manifest['removal_id']}/manifest.json"
    assert sorted(entry["entries"]) == REMOVED_ENTRIES
    # The report no longer finds a cell there, and no hidden entry is one.
    assert not [p for p in report.discover_cells(removed_runs) if REMOVED in p.name]


def test_the_manifest_is_written_before_anything_is_deleted(
    removed_runs, capsys, monkeypatch
):
    """Stopped at the first entry: the manifest, calls.jsonl and the index are on disk
    (state deleting) and nothing is deleted yet; the next run finishes that removal and
    writes no second manifest."""
    real = os.rename

    def refuse(src: Any, dst: Any) -> None:
        if tool.TRASH_DIR in str(dst):
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(tool.os, "rename", refuse)
    code, printed = _run(removed_runs, "--yes", capsys=capsys)
    assert code == tool.EXIT_FAILED and "run the tool again" in printed["error"]
    path, manifest = _manifest(removed_runs)
    assert manifest["state"] == tool.STATE_DELETING
    calls = path.parent / tool.CALLS_FILE
    assert calls.is_file()
    assert hashlib.sha256(calls.read_bytes()).hexdigest() == manifest["calls_sha256"]
    index = json.loads((removed_runs / tool.INDEX_FILE).read_text())
    assert [r["state"] for r in index["removals"]] == [tool.STATE_DELETING]
    assert all((removed_runs / entry).exists() for entry in REMOVED_ENTRIES)
    # Meanwhile the report counts every cell once, from disk.
    data = report.build_report(removed_runs, frozen=FP)
    assert {c["on_disk"] for c in data["removed"]["cells"]} == {True}
    monkeypatch.setattr(tool.os, "rename", real)
    code, printed = _run(removed_runs, "--yes", capsys=capsys)
    assert code == tool.EXIT_OK
    assert printed["finished_pending"] == [manifest["removal_id"]]
    assert printed["deleted"] == [] and "nothing" in printed["detail"]
    path, finished = _manifest(removed_runs)
    assert finished["removal_id"] == manifest["removal_id"]
    assert finished["state"] == tool.STATE_COMPLETE
    assert not any((removed_runs / entry).exists() for entry in REMOVED_ENTRIES)


def test_a_second_run_finds_nothing_and_exits_0(removed_runs, capsys):
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    code, printed = _run(removed_runs, "--yes", capsys=capsys)
    assert code == tool.EXIT_OK and printed["deleted"] == []
    assert (
        len(list((removed_runs / tool.REMOVED_DIR / REMOVED).glob("*/manifest.json")))
        == 1
    )


def test_manifest_content(removed_runs, capsys):
    loaded = {
        c.cell_id: c
        for c in report.load_cells(removed_runs, split_mode=None)
        if study.method_of(c) == REMOVED
    }
    expected_costs = {cid: report.cell_costs(c) for cid, c in loaded.items()}
    trees = {entry: tool._tree(removed_runs / entry) for entry in REMOVED_ENTRIES}
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    _, manifest = _manifest(removed_runs)
    assert (manifest["schema"], manifest["kind"]) == (tool.SCHEMA, tool.KIND)
    assert (manifest["decision"], manifest["quote"]) == (
        "example decision",
        "delete it",
    )
    assert manifest["reason"] == study.REMOVED_METHODS[REMOVED].reason
    assert manifest["host"]["hostname"] and manifest["created_at"]
    cells = {c["cell_id"]: c for c in manifest["cells"]}
    assert set(cells) == set(loaded)
    for cell_id, cell in cells.items():
        assert cell["costs"] == json.loads(json.dumps(expected_costs[cell_id]))
        assert cell["not_counted"] == report.uncounted_costs(expected_costs[cell_id])
        assert cell["status"] == loaded[cell_id].status.get("status")
        assert cell["attempts"] == len(loaded[cell_id].attempts)
    assert (cells[REMOVED_ALF]["status"], cells[REMOVED_ALF]["complete"]) == (
        "interrupted",
        False,
    )
    assert cells[REMOVED_WEB]["calls"]["calls"] == 3
    assert cells[REMOVED_WEB]["tokens"]["input_tokens_actual"] == 3 * 110
    assert cells[REMOVED_WEB]["gateway_runs"] == [f"{REMOVED_WEB}-a01"]
    runs = {r["run_id"]: r for r in manifest["gateway_runs"]}
    assert runs[ORPHAN]["cell_id"] == f"alfworld-luna56-{REMOVED}-smoke-l2-t2"
    assert runs[ORPHAN]["cost_total"] == pytest.approx(0.07)
    deleted = {d["entry"]: d for d in manifest["deleted"]}
    assert set(deleted) == set(REMOVED_ENTRIES)
    for entry, tree in trees.items():
        assert {k: deleted[entry][k] for k in tree} == tree
        assert deleted[entry]["path"] == entry
    assert deleted[REMOVED_WEB]["kind"] == tool.KIND_CELL
    assert deleted[ORPHAN]["kind"] == tool.KIND_GATEWAY_RUN
    assert manifest["totals"]["calls"] == manifest["calls_rows"] == 7
    assert {k["path"] for k in manifest["kept"]} >= {"_blobs", "_gateway"}


def _report_view(data: Mapping[str, Any]) -> dict[str, Any]:
    removed = [
        {k: v for k, v in c.items() if k not in ("on_disk", "deleted")}
        for c in data["removed"]["cells"]
    ]
    return {
        "costs": data["costs"],
        "spend": data["spend"],
        "spend_union": data["spend_union"],
        "methods": data["removed"]["methods"],
        "removed_cells": removed,
        "cells": data["cells"],
        "comparisons": data["comparisons"],
    }


def test_the_report_is_identical_after_the_deletion(removed_runs, capsys):
    """Every number of the report is the same before and after the deletion: the
    matrix, the removed cells' costs (from the manifest), all cells' total and the spend
    window (the deleted ledgers' calls from calls.jsonl)."""
    before = report.build_report(removed_runs, frozen=FP)
    # In the window: 2 study calls and the removed method's timed ones (WebShop of luna56, the orphan
    # run); test_report's leftover cells' calls carry no time: outside it.
    spent = before["spend"][0]
    assert (spent["ledger_calls"], spent["calls_outside_window"]) == (4, 5)
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    after = report.build_report(removed_runs, frozen=FP)
    assert _report_view(after) == _report_view(before)
    assert after["costs"]["all_cells_total"] == before["costs"]["all_cells_total"]
    deleted = {c["cell_id"]: c for c in after["removed"]["cells"]}
    assert {c["on_disk"] for c in deleted.values()} == {False}
    assert deleted[REMOVED_WEB]["deleted"]["manifest"].startswith(
        f"{tool.REMOVED_DIR}/{REMOVED}/"
    )
    assert after["removed"]["deleted"]["cells"] == 4
    assert after["removed"]["deleted"]["cost_usd"] == pytest.approx(
        before["removed"]["cost_usd"]
    )
    assert not [w for w in after["warnings"] if w["check"].startswith("removed_")]
    md = report.render_markdown(after)
    assert "The data of 4 cell(s) was deleted" in md
    assert "deleted " in "\n".join(md.split("## Removed methods")[1].splitlines()[:30])
    # The spend of the deleted runs, for a ledger-based reader.
    spent = tool.removed_spend(removed_runs)
    assert spent["calls"] == 7 and spent["cost_usd"] == pytest.approx(0.19)


def test_two_hosts_keep_their_union_after_one_deletes(
    tmp_path, capsys, deletion_decided
):
    """Over a merged view of two hosts sharing one budget start, the per-host spend and
    their union are the same after one host deleted the removed method's data and the view mirrored
    the deletion (the manifest under _hosts/<host>/_removed)."""
    view = tmp_path / "view"
    shared = {"budget_usd": 50.0, "started_at": START, "start_usage_usd": 100.0}
    _host_runs(
        view,
        "cloud",
        "alfworld",
        0.30,
        AT,
        END,
        {**shared, "spent_usd": 0.5, "usage": 100.5},
    )
    _host_runs(
        view,
        "local",
        "webshop",
        0.20,
        AT,
        END,
        {**shared, "spent_usd": 0.6, "usage": 100.6},
    )
    mirror = view / "_hosts" / "cloud"
    removed = make_cell(
        mirror,
        "alfworld",
        REMOVED,
        test=[_row("a1", 1.0)],
        calls=[_timed_call("actor", "test", 0.05, item="u1")],
    )
    for entry in (removed, mirror / f"{REMOVED_ALF}-a01"):
        (view / entry.name).symlink_to(os.path.relpath(entry, view))
    before = report.build_report(view, frozen=FP)
    assert [c["cell_id"] for c in before["removed"]["cells"]] == [REMOVED_ALF]
    assert _run(mirror, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    after = report.build_report(view, frozen=FP)
    assert _report_view(after) == _report_view(before)
    assert before["spend_union"]["ledger_cost_usd"] == pytest.approx(0.55)
    (row,) = after["removed"]["cells"]
    assert row["on_disk"] is False


@pytest.mark.parametrize(
    "case",
    ["study_method", "merged_view", "mirror", "live", "symlink", "ledger_outside"],
)
def test_refusals_write_nothing(removed_runs, tmp_path, capsys, case):
    method = REMOVED
    runs = removed_runs
    holder = None
    if case == "study_method":
        method = "awm"
    elif case == "merged_view":
        (removed_runs / "_merged.json").write_text("{}")
    elif case == "mirror":
        view = tmp_path / "view"
        (view / "_hosts").mkdir(parents=True)
        (view / "_merged.json").write_text("{}")
        runs = view / "_hosts" / "cloud"
        removed_runs.rename(runs)
    elif case == "live":
        holder = open(removed_runs / REMOVED_WEB / "cell.lock", "a")  # noqa: SIM115
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif case == "symlink":
        real = tmp_path / "elsewhere"
        (removed_runs / REMOVED_ALF).rename(real)
        (removed_runs / REMOVED_ALF).symlink_to(real)
    elif case == "ledger_outside":
        outside = tmp_path / "outside" / f"{REMOVED_WEB}-a01"
        outside.parent.mkdir()
        (removed_runs / f"{REMOVED_WEB}-a01").rename(outside)
        state = removed_runs / REMOVED_WEB / "attempt-01" / "attempt.json"
        data = json.loads(state.read_text())
        data["ledger"] = str(outside / "gateway.jsonl")
        write_json_atomic(state, data)
    before = _tree_digest(runs)
    try:
        code = tool.main(["--runs-dir", str(runs), "--method", method, "--yes"])
        printed = json.loads(capsys.readouterr().out)
    finally:
        if holder is not None:
            holder.close()
    assert code == tool.EXIT_REFUSED, printed
    assert printed["refused"]
    assert not (runs / tool.REMOVED_DIR).exists()
    assert _tree_digest(runs) == before


def test_removed_entries_from_index_parses_what_sync_runs_reads():
    text = json.dumps(
        {
            "schema": 1,
            "removals": [
                {"removal_id": "r1", "state": "complete", "entries": ["a", "b-a01"]},
                {"removal_id": "r2", "state": "deleting", "entries": ["c"]},
                {"removal_id": "r3", "state": "planned", "entries": ["d"]},
                {
                    "removal_id": "r4",
                    "state": "complete",
                    "entries": ["_blobs", ".trash.tmp", "x/y", "", 7],
                },
            ],
        }
    )
    assert tool.removed_entries_from_index(text) == frozenset({"a", "b-a01", "c"})
    assert tool.removed_entries_from_index(text, finished_only=True) == frozenset(
        {"a", "b-a01"}
    )
    assert tool.removed_entries_from_index("not json") == frozenset()
    assert tool.removed_entries_from_index("[1]") == frozenset()


def test_manifests_are_read_once_per_removal_and_never_raise(
    removed_runs, tmp_path, capsys
):
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    view = tmp_path / "view"
    (view / "_hosts").mkdir(parents=True)
    (view / "_hosts" / "cloud").symlink_to(removed_runs)
    (view / "_hosts" / "copy").symlink_to(removed_runs)  # the same removal twice
    (found,) = tool.load_manifests(view)
    assert found.problems == [] and found.runs_dir.name == "cloud"
    assert len(tool.manifest_calls(found)) == 7
    broken = view / tool.REMOVED_DIR / REMOVED / "r0"
    broken.mkdir(parents=True)
    (broken / tool.MANIFEST_FILE).write_text("{torn")
    manifests = tool.load_manifests(view)
    assert [bool(m.problems) for m in manifests] == [True, False]


def test_a_manifest_and_a_live_cell_of_one_directory_count_once(
    removed_runs, capsys, monkeypatch
):
    """An interrupted deletion leaves a cell both on disk and in its manifest: the
    report reads it from disk, once."""
    before = report.build_report(removed_runs, frozen=FP)

    def refuse(src: Any, dst: Any) -> None:
        raise OSError("stopped")

    monkeypatch.setattr(tool.os, "rename", refuse)
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_FAILED
    during = report.build_report(removed_runs, frozen=FP)
    assert _report_view(during) == _report_view(before)
    assert len(during["removed"]["cells"]) == 4


def test_an_unreadable_manifest_is_a_warning_not_silence(removed_runs, capsys):
    """A manifest the report cannot read is named in a warning (its money unknown);
    a calls.jsonl that differs from its manifest is named too, its rows still counted."""
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    path, manifest = _manifest(removed_runs)
    calls = path.parent / tool.CALLS_FILE
    calls.write_text(
        calls.read_text()
        + json.dumps({"event": "call", "cost_total": 1.0, "run_id": "x"})
        + "\n"
    )
    data = report.build_report(removed_runs, frozen=FP)
    (mismatch,) = [
        w for w in data["warnings"] if w["check"] == "removed_manifest_mismatch"
    ]
    assert (
        "8 row(s)" in mismatch["detail"]
        and "its rows are counted" in mismatch["detail"]
    )
    assert (
        data["spend"][0]["ledger_calls"],
        data["spend"][0]["calls_outside_window"],
    ) == (4, 6)
    path.write_text("{torn")
    data = report.build_report(removed_runs, frozen=FP)
    (unreadable,) = [
        w for w in data["warnings"] if w["check"] == "removed_manifest_unreadable"
    ]
    assert "its money unknown" in unreadable["detail"]
    assert [c for c in data["removed"]["cells"] if not c["on_disk"]] == []
    assert "- **Warning (removed_manifest_unreadable):**" in report.render_markdown(
        data
    )


def _write_manifest(base: Path, removal_id: str, data: Mapping[str, Any]) -> Path:
    path = base / tool.REMOVED_DIR / REMOVED / removal_id / tool.MANIFEST_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    return path


def test_a_manifest_the_tool_did_not_write_is_read_by_its_fields(tmp_path):
    """The reader checks ``kind`` and ``schema`` when a manifest has them (another tool's
    file, another format: a problem) and reads one without them (a record written by
    hand, as the audit's tests write one); of two copies of one removal, a readable one
    wins over a torn one."""
    runs = tmp_path / "runs"
    _write_manifest(runs, "r-plain", {"removal_id": "r-plain", "cells": []})
    _write_manifest(runs, "r-kind", {"kind": "something.else", "cells": []})
    _write_manifest(runs, "r-schema", {"kind": tool.KIND, "schema": 2, "cells": []})
    _write_manifest(runs, "r-cells", {"kind": tool.KIND, "schema": 1, "cells": {}})
    found = {m.removal_id: m.problems for m in tool.load_manifests(runs)}
    assert found["r-plain"] == []
    assert "kind 'something.else'" in found["r-kind"][0]
    assert "schema 2" in found["r-schema"][0]
    assert "not a list" in found["r-cells"][0]
    view = tmp_path / "view"
    _write_manifest(view, "r-1", {}).write_text("{torn")
    good = _write_manifest(
        view / "_hosts" / "cloud", "r-1", {"removal_id": "r-1", "cells": []}
    )
    (only,) = tool.load_manifests(view)
    assert only.path == good and only.problems == []


def test_a_manifest_row_without_its_costs_is_a_warning_not_a_crash(
    removed_runs, capsys
):
    """A manifest row that cannot stand in for its cell (its costs lost) is left out of
    the removed block and named in a warning; the manifest's other rows are listed and
    its calls still count. The learning summary does the same."""
    assert _run(removed_runs, "--yes", capsys=capsys)[0] == tool.EXIT_OK
    intact = report.build_report(removed_runs, frozen=FP)
    gone = sorted(c["cell_id"] for c in intact["removed"]["cells"] if not c["on_disk"])
    path, manifest = _manifest(removed_runs)
    broken = manifest["cells"][0]
    del broken["costs"]
    path.write_text(json.dumps(manifest))
    assert tool.cell_row_problem(broken) == f"{broken['cell_id']}: no costs.total_cost"
    data = report.build_report(removed_runs, frozen=FP)
    (warning,) = [
        w for w in data["warnings"] if w["check"] == "removed_manifest_unreadable"
    ]
    assert f"{broken['cell_id']}: no costs.total_cost" in warning["detail"]
    listed = sorted(c["cell_id"] for c in data["removed"]["cells"] if not c["on_disk"])
    assert listed == [c for c in gone if c != broken["cell_id"]]
    assert data["spend"] == intact["spend"]
    assert "- **Warning (removed_manifest_unreadable):**" in report.render_markdown(
        data
    )
    summary = summarize_learning.summarize(removed_runs, split_mode=None)
    assert broken["cell_id"] not in [c["cell_id"] for c in summary["removed"]]
    assert summary["removed_manifest_problems"] == [
        f"{path}: {broken['cell_id']}: no costs.total_cost"
    ]
