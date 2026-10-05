"""tools/aggregate_replicates.py on cells written by hand in run.py's layout
(``test_report.make_cell``), so every replicate value, mean, standard deviation and flag
is known in advance. The tests write a passing audit for every runs directory
(:func:`audited`) unless they check what its absence does."""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import report, study
from evals.skill_bench.cell_records import (
    NOTE_RERUN_ACKNOWLEDGED,
    CellPaths,
    records_digest,
    write_json_atomic,
)
from evals.skill_bench.tests.test_report import _row, binary, make_cell
from evals.skill_bench.tools import aggregate_replicates as agg

METHODS = ("mirix", "reasoningbank", "awm", "none")


def shop(prefix: str, rewards: list[float]) -> list[dict[str, Any]]:
    """WebShop test rows: the reward is the primary score, success is reward == 1."""
    return [
        _row(
            f"{prefix}{i:03d}",
            r,
            {"success": float(r == 1.0), "success_within_15": float(r == 1.0)},
        )
        for i, r in enumerate(rewards)
    ]


#: Synthetic ALFWorld successes per (method, replicate), four test items each.
ALFWORLD = {
    "mirix": ([1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 1]),
    "reasoningbank": ([1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]),
    "awm": ([1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]),
    "none": ([0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]),
}
#: Synthetic WebShop rewards per (method, replicate), four test items each.
WEBSHOP = {
    "mirix": ([1.0, 0.9, 0.9, 0.0], [1.0, 1.0, 0.9, 0.9], [1.0, 0.9, 0.9, 0.9]),
    "reasoningbank": ([1.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0]),
    "awm": ([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]),
    "none": ([0.0, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
}
#: A timestamp for the records the tests write.
AT = "2026-01-02T00:00:00+00:00"


def audited(*dirs: Path) -> None:
    """A passing audit (audit.json schema 2) of every cell directory of each runs
    directory, on the records now on disk (``report.audit_status``: passed)."""
    for runs in dirs:
        cells = []
        for root in report.discover_cells(runs):
            cell = report.load_cell(root, runs)
            cells.append(
                {
                    "cell_id": cell.cell_id,
                    "root": str(root),
                    "audited": True,
                    "attempt": cell.reported,
                    "violations": 0,
                    "info": {"records_sha256": records_digest(CellPaths(root))},
                }
            )
        write_json_atomic(
            runs / "_audit" / "audit.json",
            {
                "generated_at": AT,
                "passed": True,
                "validity_checks": [],
                "findings": [],
                "cells": cells,
            },
        )


def three_replicates(tmp_path: Path, *, audit: bool = True) -> list[Path]:
    dirs = [tmp_path / "runs" / f"phase16-r{n}" for n in (1, 2, 3)]
    for index, runs in enumerate(dirs):
        for method in METHODS:
            make_cell(
                runs,
                "alfworld",
                method,
                test=binary("a", ALFWORLD[method][index]),
                instance=f"i{index}{method[:3]}",
            )
            make_cell(
                runs,
                "webshop",
                method,
                test=shop("w", WEBSHOP[method][index]),
                instance=f"w{index}{method[:3]}",
            )
    if audit:
        audited(*dirs)
    return dirs


def remove_cell(runs: Path, cell_id: str) -> None:
    """Delete one cell directory (a replicate that never ran the cell)."""
    root = runs / cell_id
    for path in sorted(root.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    root.rmdir()


def row_of(data: dict[str, Any], cell_id: str) -> dict[str, Any]:
    return next(r for r in data["cells"] if r["cell_id"] == cell_id)


def flags_of(data: dict[str, Any], cell_id: str) -> dict[str, dict[str, Any]]:
    return {f["flag"]: f for f in row_of(data, cell_id)["flags"]}


def set_attempt(runs: Path, cell_id: str, **fields: Any) -> None:
    """Add ``fields`` to the cell's reported attempt.json (attempt 1)."""
    path = CellPaths(runs / cell_id).attempt(1).state
    state = json.loads(path.read_text())
    state.update(fields)
    write_json_atomic(path, state)


def test_each_cell_over_three_replicates(tmp_path):
    dirs = three_replicates(tmp_path)
    data = agg.aggregate(dirs)
    assert data["schema"] == agg.SCHEMA_VERSION
    assert data["expect_replicates"] == 3 and data["split_mode"] == "protocol"
    assert data["warnings"] == [] and data["missing_cells"] == []
    assert [r["label"] for r in data["replicates"]] == [
        "phase16-r1",
        "phase16-r2",
        "phase16-r3",
    ]
    assert all(r["complete"] == r["cells"] == 8 for r in data["replicates"])
    assert all(r["audit"] for r in data["replicates"])
    assert len(data["cells"]) == 8 and data["flags"] == []
    mirix = row_of(data, "alfworld-luna-mirix")
    assert (mirix["benchmark"], mirix["model_key"], mirix["method"]) == (
        "alfworld",
        "luna",
        "mirix",
    )
    assert mirix["main_metric"] == "success_rate" and mirix["group"] == "alfworld-luna"
    stats = mirix["metrics"]["success_rate"]
    assert stats["main"] is True and stats["values"] == {
        "phase16-r1": 0.75,
        "phase16-r2": 1.0,
        "phase16-r3": 0.75,
    }
    assert stats["n"] == 3 and stats["mean"] == pytest.approx(2.5 / 3)
    assert stats["std"] == pytest.approx(statistics.stdev([0.75, 1.0, 0.75]))
    assert (stats["min"], stats["max"]) == (0.75, 1.0)
    entry = mirix["replicates"]["phase16-r2"]
    assert entry["complete"] and entry["items"] == 4 and entry["reported_attempt"] == 1
    assert entry["superseded_attempts"] == [] and entry["audit"] == report.AUDIT_PASSED
    assert entry["test_items_sha256"] == agg.test_items_digest(
        report.load_cell(dirs[1] / "alfworld-luna-mirix", dirs[1])
    )
    # make_cell records no code or agent context (an attempt from before them).
    assert (entry["harness"], entry["mirix"], entry["agent_context_sha256"]) == (
        None,
        None,
        None,
    )
    assert entry["reruns"] is False
    # WebShop: every metric report.py reports, the success rate first (main).
    shop_row = row_of(data, "webshop-luna-mirix")
    assert list(shop_row["metrics"]) == ["success_rate", "score", "success_within_15"]
    assert shop_row["main_metric"] == "success_rate"
    assert shop_row["metrics"]["score"]["main"] is False
    assert shop_row["metrics"]["success_rate"]["values"]["phase16-r1"] == 0.25
    assert shop_row["metrics"]["score"]["mean"] == pytest.approx(
        statistics.fmean([0.7, 0.95, 0.925])
    )
    # Launch order: benchmark, then MIRIX and the baselines.
    assert [r["cell_id"] for r in data["cells"]][:4] == [
        "alfworld-luna-mirix",
        "alfworld-luna-reasoningbank",
        "alfworld-luna-awm",
        "alfworld-luna-none",
    ]


def test_the_summary_gives_mean_std_and_n_per_benchmark_model_and_method(tmp_path):
    data = agg.aggregate(three_replicates(tmp_path))
    markdown = agg.render_markdown(data)
    summary = markdown.split("## Summary")[1].split("## Cells")[0]
    assert "### alfworld" in summary and "### webshop" in summary
    assert "| Group | Model | Method | success rate (main) |" in summary
    mirix = f"{2.5 / 3:.3f} ± {statistics.stdev([0.75, 1.0, 0.75]):.3f} (n=3)"
    assert f"| alfworld-luna | luna | mirix | {mirix} |" in summary
    # One value has no spread; no value is a dash.
    assert agg.mean_std_text({"n": 1, "mean": 0.5, "std": None}) == "0.500 (n=1)"
    assert agg.mean_std_text({"n": 0, "mean": None, "std": None}) == "-"
    shop_line = next(
        line for line in summary.splitlines() if line.startswith("| webshop-luna |")
    )
    assert shop_line.count("(n=3)") == 3  # the success rate, the score, within 15


def test_without_an_audit_the_numbers_stand_and_say_so(tmp_path):
    """No runs directory was audited: every replicate's audit status says so, no cell
    is flagged for it (only an audit that exists and does not vouch is), and the
    Markdown names the runs directories without one."""
    data = agg.aggregate(three_replicates(tmp_path, audit=False))
    assert not any(r["audit"] for r in data["replicates"])
    entry = row_of(data, "alfworld-luna-mirix")["replicates"]["phase16-r1"]
    assert entry["audit"] == report.AUDIT_MISSING
    assert data["flags"] == []
    assert row_of(data, "alfworld-luna-mirix")["metrics"]["success_rate"]["n"] == 3
    assert "complete, no audit)" in agg.render_markdown(data)


def test_mixed_agent_contexts_and_mirix_code_across_replicates_are_flagged(tmp_path):
    """A cell whose replicates carried different agent contexts or ran different MIRIX
    code is flagged, with the digests of each replicate."""
    dirs = three_replicates(tmp_path, audit=False)
    for n, runs in enumerate(dirs):
        set_attempt(
            runs,
            "alfworld-luna-mirix",
            code={"harness": "1" * 64, "mirix": ("a" if n < 2 else "b") * 64},
            agent_context={"sha256": ("c" if n == 0 else "d") * 64, "chars": 9},
        )
    audited(*dirs)
    data = agg.aggregate(dirs)
    flags = flags_of(data, "alfworld-luna-mirix")
    assert set(flags) == {agg.FLAG_MIXED_AGENT_CONTEXT, agg.FLAG_MIXED_MIRIX}
    assert flags[agg.FLAG_MIXED_AGENT_CONTEXT]["detail"] == (
        "2 agent contexts: phase16-r1 cccccccccccc, phase16-r2 dddddddddddd, "
        "phase16-r3 dddddddddddd"
    )
    assert flags[agg.FLAG_MIXED_MIRIX]["replicates"] == [
        "phase16-r1",
        "phase16-r2",
        "phase16-r3",
    ]
    # The numbers are kept: a flag never leaves a replicate out.
    assert row_of(data, "alfworld-luna-mirix")["metrics"]["success_rate"]["n"] == 3


def test_flags_say_over_which_replicates_the_numbers_are(tmp_path):
    dirs = three_replicates(tmp_path, audit=False)
    r1, r2, r3 = dirs
    # r3 lacks a cell; r2's is still testing; r1's AWM cell is a violation.
    remove_cell(r3, "alfworld-luna-none")
    write_json_atomic(r2 / "alfworld-luna-none" / "status.json", {"status": "testing"})
    write_json_atomic(r1 / "alfworld-luna-awm" / "status.json", {"status": "violation"})
    # r1's MIRIX cell superseded a tested attempt; r2's ran under another protocol.
    make_cell(
        r1,
        "alfworld",
        "mirix",
        test=binary("a", ALFWORLD["mirix"][0]),
        instance="i0mir",
        attempts=[
            (
                "invalid",
                [],
                {
                    "invalidation": "superseded",
                    "previous_state": "complete",
                    "superseded_by": 2,
                },
            ),
            ("complete", [], {}),
        ],
    )
    make_cell(
        r2,
        "alfworld",
        "mirix",
        test=binary("a", ALFWORLD["mirix"][1]),
        instance="i1mir",
        fingerprint="e" * 64,
    )
    # A second directory of one cell id in r3 (the older one is set aside).
    make_cell(
        r3,
        "alfworld",
        "reasoningbank",
        test=binary("a", ALFWORLD["reasoningbank"][2]),
        started_at=AT,
        instance="i2rea",
    )
    make_cell(
        r3,
        "alfworld",
        "reasoningbank",
        test=binary("a", [0, 0, 0, 0]),
        directory="alfworld-luna-reasoningbank-old",
        started_at="2026-01-01T00:00:00+00:00",
        instance="oldrb",
    )
    # r1 and r3 pass their audits; an audit file r2's cells cannot be vouched for by.
    audited(r1, r3)
    (r2 / "_audit").mkdir()
    (r2 / "_audit" / "audit.json").write_text("{")
    data = agg.aggregate(dirs)
    none = flags_of(data, "alfworld-luna-none")
    assert none[agg.FLAG_MISSING]["replicates"] == ["phase16-r3"]
    assert none[agg.FLAG_MISSING]["detail"] == "no alfworld-luna-none in phase16-r3"
    assert none[agg.FLAG_INCOMPLETE]["replicates"] == ["phase16-r2"]
    assert "phase16-r2: testing" in none[agg.FLAG_INCOMPLETE]["detail"]
    stats = row_of(data, "alfworld-luna-none")["metrics"]["success_rate"]
    assert stats["n"] == 1 and stats["values"]["phase16-r2"] is None
    assert stats["values"]["phase16-r3"] is None and stats["std"] is None
    awm = flags_of(data, "alfworld-luna-awm")
    assert awm[agg.FLAG_BLOCKED]["replicates"] == ["phase16-r1"]
    assert agg.FLAG_INCOMPLETE not in awm  # blocked, not merely incomplete
    mirix = flags_of(data, "alfworld-luna-mirix")
    assert mirix[agg.FLAG_SUPERSEDED]["replicates"] == ["phase16-r1"]
    assert "attempt(s) 1" in mirix[agg.FLAG_SUPERSEDED]["detail"]
    assert set(mirix[agg.FLAG_MIXED_PROTOCOL]["replicates"]) == {
        "phase16-r1",
        "phase16-r2",
        "phase16-r3",
    }
    assert mirix[agg.FLAG_AUDIT]["replicates"] == ["phase16-r2"]
    rb = flags_of(data, "alfworld-luna-reasoningbank")
    assert rb[agg.FLAG_DUPLICATES]["replicates"] == ["phase16-r3"]
    assert (
        "alfworld-luna-reasoningbank-old (superseded) not used"
        in (rb[agg.FLAG_DUPLICATES]["detail"])
    )
    # The newest directory is the one counted.
    assert row_of(data, "alfworld-luna-reasoningbank")["metrics"]["success_rate"][
        "values"
    ]["phase16-r3"] == pytest.approx(0.5)
    assert set(data["flags"]) >= {
        agg.FLAG_MISSING,
        agg.FLAG_INCOMPLETE,
        agg.FLAG_BLOCKED,
        agg.FLAG_SUPERSEDED,
        agg.FLAG_MIXED_PROTOCOL,
        agg.FLAG_DUPLICATES,
        agg.FLAG_AUDIT,
    }
    markdown = agg.render_markdown(data)
    assert (
        "`alfworld-luna-none` missing_replicate: no alfworld-luna-none in phase16-r3"
        in (markdown)
    )
    summary = markdown.split("## Summary")[1].split("## Cells")[0]
    assert "| alfworld-luna | luna | none | 0.250 (n=1) |" in summary


def test_a_rerun_after_test_items_is_flagged(tmp_path):
    """An operator note (run.py --acknowledge-rerun) of an earlier run that scored test
    items: the cell ran more than once, and its row says so."""
    dirs = three_replicates(tmp_path, audit=False)
    path = dirs[0] / "alfworld-luna-mirix" / "cell.json"
    manifest = json.loads(path.read_text())
    manifest["operator_notes"] = [
        {
            "kind": NOTE_RERUN_ACKNOWLEDGED,
            "file": "operator_notes/0123456789ab-rerun.md",
            "sha256": "0" * 64,
            "at": AT,
            "superseded": [
                {
                    "run_id": "alfworld-luna-mirix-a01-0dd0dd01",
                    "instance": "0dd0dd01",
                    "test_actor_calls": 20,
                }
            ],
        }
    ]
    write_json_atomic(path, manifest)
    audited(*dirs)
    data = agg.aggregate(dirs)
    (flag,) = row_of(data, "alfworld-luna-mirix")["flags"]
    assert flag["flag"] == agg.FLAG_RERUN and flag["replicates"] == ["phase16-r1"]
    assert data["flags"] == [agg.FLAG_RERUN]
    assert "- `alfworld-luna-mirix` rerun_after_test: " in agg.render_markdown(data)


def test_another_number_of_runs_directories_than_expected_is_a_warning(tmp_path):
    dirs = three_replicates(tmp_path)
    data = agg.aggregate(dirs[:2], expect=3)
    assert data["warnings"] == [
        "2 runs directories named, 3 replicates expected: each cell's numbers are "
        "over the runs directories named"
    ]
    assert row_of(data, "alfworld-luna-mirix")["metrics"]["success_rate"]["n"] == 2
    assert agg.aggregate(dirs, expect=2)["warnings"][0].startswith(
        "3 runs directories named, 2 replicates expected"
    )
    with pytest.raises(ValueError, match="at least 1"):
        agg.aggregate(dirs, expect=0)
    with pytest.raises(ValueError, match="named twice"):
        agg.aggregate([dirs[0], dirs[0] / "."])
    with pytest.raises(ValueError, match="at least one"):
        agg.aggregate([])


def test_study_phase16_expects_its_three_replicates(tmp_path, capsys):
    """``--study phase16`` over two runs directories: the study's replicate count is
    expected whatever is named, with a warning."""
    assert study.STUDIES["phase16"].replicates == 3
    assert study.STUDIES["phase16-dev"].replicates == 1
    assert study.STUDIES["phase16-pilot"].replicates == 1
    dirs = three_replicates(tmp_path)
    data = agg.aggregate(dirs[:2], study_name="phase16")
    assert data["expect_replicates"] == 3
    assert data["warnings"] == [
        "2 runs directories named, 3 replicates expected (--study phase16): each "
        "cell's numbers are over the runs directories named"
    ]
    # --expect still overrides it.
    assert agg.aggregate(dirs[:2], study_name="phase16", expect=2)["warnings"] == []
    out = tmp_path / "agg"
    assert agg.main([*map(str, dirs[:2]), "--study", "phase16", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "warning: 2 runs directories named, 3 replicates expected" in captured.err
    assert json.loads(captured.out)["complete"] == 16


def test_a_study_keeps_its_cells_and_lists_the_missing_ones(tmp_path):
    """``--study phase16``: the study's cells only (a cell of another study's split is
    left out), and every (benchmark, model, method) of the study that no runs
    directory holds: luna56 never ran here."""
    dirs = three_replicates(tmp_path)
    make_cell(
        dirs[0],
        "alfworld",
        "mirix",
        test=binary("v", [1, 0]),
        split_mode="smoke",
        n_train=39,
        n_test=2,
    )
    remove_cell(dirs[2], "webshop-luna-awm")
    audited(*dirs)
    data = agg.aggregate(dirs, study_name="phase16")
    assert data["study"] == "phase16" and data["split_mode"] == "protocol"
    assert {r["cell_id"] for r in data["cells"]} == {
        f"{b}-luna-{m}" for b in ("alfworld", "webshop") for m in METHODS
    }
    assert data["missing_cells"] == [
        {"benchmark": b, "model_key": "luna56", "method": m}
        for b in ("alfworld", "webshop")
        for m in METHODS
    ]
    # A cell some runs directory holds is not missing: its row flags the replicate.
    assert flags_of(data, "webshop-luna-awm")[agg.FLAG_MISSING]["replicates"] == [
        "phase16-r3"
    ]
    markdown = agg.render_markdown(data)
    assert markdown.splitlines()[2].startswith("Study `phase16`, split mode `protocol`")
    assert (
        "- **Missing cells** (the study's, in no runs directory): "
        "alfworld/luna56/mirix, alfworld/luna56/reasoningbank" in markdown
    )
    # Without a study, nothing is expected beyond the cells found.
    assert agg.aggregate(dirs)["missing_cells"] == []
    with pytest.raises(ValueError, match="unknown study"):
        agg.aggregate(dirs, study_name="no-such-study")
    with pytest.raises(SystemExit):
        agg.main([str(dirs[0]), "--study", "phase16", "--split-mode", "smoke"])


def test_the_spend_union_of_replicates_that_share_a_budget_start(tmp_path):
    dirs = three_replicates(tmp_path)
    start = {"start_usage_usd": 5.0, "started_at": AT}
    for n, runs in enumerate(dirs):
        rows = [
            {
                "ts": f"2026-01-02T0{n + 1}:00:00+00:00",
                "usage": 5.0 + 2.0 * (n + 1),
                "budget": {**start, "spent_usd": 2.0 * (n + 1)},
            }
        ]
        (runs / "_gateway").mkdir(parents=True, exist_ok=True)
        (runs / "_gateway" / "balance.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    spend = agg.aggregate(dirs)["spend"]
    assert [s["replicate"] for s in spend["directories"]] == [
        "phase16-r1",
        "phase16-r2",
        "phase16-r3",
    ]
    assert spend["directories"][0]["key_usage_delta_usd"] == 2.0
    union = spend["union"]
    assert union["shared"] is True and union["key_usage_delta_usd"] == 6.0
    # Another start: no union can be computed, and it says so.
    (dirs[2] / "_gateway" / "balance.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-01-02T04:00:00+00:00",
                "usage": 20.0,
                "budget": {
                    "start_usage_usd": 9.0,
                    "started_at": "x",
                    "spent_usd": 11.0,
                },
            }
        )
        + "\n"
    )
    assert agg.aggregate(dirs)["spend"]["union"]["shared"] is False


def test_the_spend_lines_say_unattributed_spend_is_any_other_use_of_the_key(
    tmp_path,
):
    """Development runs on the same key fall into the unattributed spend, and the
    Markdown says so."""
    data = agg.aggregate(three_replicates(tmp_path))
    assert "No balance record in any runs directory." in agg.render_markdown(data)
    data["spend"] = {
        "directories": [
            {
                "replicate": "phase16-r1",
                "key_usage_delta_usd": 3.0,
                "ledger_cost_usd": 1.25,
                "unattributed_usd": 1.75,
            }
        ],
        "union": {
            "shared": True,
            "key_usage_delta_usd": 3.0,
            "ledger_cost_usd": 1.25,
            "unattributed_usd": 1.75,
        },
    }
    spend = agg.render_markdown(data).split("## Spend")[1]
    assert (
        "- `phase16-r1`: key usage +$3.00, ledgers $1.25, unattributed $1.75" in spend
    )
    assert spend.count("development runs") == 2
    assert "any other use of the same key" in spend


def test_rows_follow_the_schedulers_method_order():
    from evals.skill_bench import schedule

    assert agg.METHOD_ORDER == schedule.METHOD_ORDER


def test_the_cli_writes_json_and_markdown(tmp_path, capsys):
    dirs = three_replicates(tmp_path)
    out = tmp_path / "agg"
    assert agg.main([*map(str, dirs), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "cells": 8,
        "complete": 24,
        "flags": [],
        "missing_cells": 0,
        "json": str(out / agg.REPLICATES_JSON),
        "markdown": str(out / agg.REPLICATES_MD),
    }
    data = json.loads((out / agg.REPLICATES_JSON).read_text())
    assert data["schema"] == agg.SCHEMA_VERSION and len(data["cells"]) == 8
    markdown = (out / agg.REPLICATES_MD).read_text()
    assert markdown.startswith("# Replicates\n")
    for section in ("## Summary", "## Cells", "## Spend"):
        assert section in markdown
    assert (
        "| alfworld-luna-mirix | success rate (main) | 0.750 | 1.000 | 0.750 | 3 |"
        in (markdown)
    )
    # Default output beside the runs directories; no complete cell exits 2.
    assert agg.main([str(dirs[0]), str(dirs[1])]) == 0
    assert (dirs[0].parent / agg.OUT_DIR_NAME / agg.REPLICATES_JSON).is_file()
    capsys.readouterr()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert agg.main([str(empty), "--out", str(tmp_path / "e")]) == 2
    assert "No cell in any runs directory." in (
        (tmp_path / "e" / agg.REPLICATES_MD).read_text()
    )
    with pytest.raises(SystemExit):
        agg.main([str(dirs[0]), str(dirs[0])])


def test_pilot_and_smoke_cells_are_aggregated_by_their_split(tmp_path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    for runs in (r1, r2):
        make_cell(
            runs,
            "alfworld",
            "mirix",
            test=binary("v", [1, 0]),
            split_mode="smoke",
            n_train=39,
            n_test=2,
        )
        make_cell(runs, "alfworld", "none", test=binary("t", [1, 1]))
    smoke = agg.aggregate([r1, r2], split_mode="smoke")
    assert [r["cell_id"] for r in smoke["cells"]] == [
        "alfworld-luna-mirix-smoke-l39-t2"
    ]
    assert smoke["cells"][0]["group"] == "alfworld-luna-smoke-t2"
    everything = agg.aggregate([r1, r2], split_mode=None)
    assert len(everything["cells"]) == 2 and everything["split_mode"] == "all"


@pytest.mark.asyncio
async def test_the_async_entry_reads_in_a_thread(tmp_path):
    dirs = three_replicates(tmp_path)
    data = await agg.aggregate_async(dirs[:1])
    assert data["expect_replicates"] == 1 and len(data["cells"]) == 8
    assert asyncio.get_running_loop() is not None
