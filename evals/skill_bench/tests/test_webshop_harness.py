"""WebShop in the harness around the environment: the report's metrics, diagnostics and
caveat, the pilot cells, the judge's input, the
INFRA failure code, the study tools' cell ids and the resource estimate. The environment
and its server are ``test_env_webshop.py`` and ``test_webshop_server.py``; the audit's
branches ``test_audit.py``; the protocol's per-benchmark diff ``test_protocol.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol, report, study
from evals.skill_bench.cell_records import CellPaths
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.infra import failures, resources
from evals.skill_bench.methods import judge
from evals.skill_bench.run import PILOT_METHOD, CellSpec, select_items
from evals.skill_bench.tests.method_contract import contract_items
from evals.skill_bench.tools import remove_method_data

WITHIN = f"success_within_{protocol.WEBSHOP_DIAGNOSTIC_STEPS}"


def row(raw_id: str, reward: float, purchased: bool = True, steps: int = 5, **metrics):
    success = 1.0 if purchased and reward == 1.0 else 0.0
    return {
        "raw_id": raw_id,
        "score": {
            "primary": reward,
            "metrics": {
                "success": success,
                WITHIN: success if steps <= 15 else 0.0,
                **metrics,
            },
            "answered": purchased,
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_webshop_metrics_are_success_score_and_the_diagnostic():
    success, score, within = report.METRICS["webshop"]
    assert (success.name, success.key, success.binary, success.main) == (
        "success_rate",
        "success",
        True,
        True,
    )
    assert (score.name, score.key, score.binary, score.main) == (
        "score",
        None,
        False,
        False,
    )
    assert (within.name, within.key, within.binary, within.main) == (
        WITHIN,
        WITHIN,
        True,
        False,
    )
    partial = row("fixed_1", 0.75)
    assert report.metric_value(partial, success) == 0.0
    assert report.metric_value(partial, score) == 0.75
    won_late = row("fixed_2", 1.0, steps=20)
    assert (
        report.metric_value(won_late, success),
        report.metric_value(won_late, within),
    ) == (
        1.0,
        0.0,
    )


def test_the_caveat_names_the_score_feedback():
    assert report.CAVEAT_WEBSHOP in report.CAVEATS
    assert "learning phase only" in report.CAVEAT_WEBSHOP


def test_item_diagnostics_read_the_best_possible_reward():
    rows = [
        row("fixed_1", 1.0, best_reward=1.0),
        row("fixed_2", 0.5, best_reward=0.5),
        row("fixed_3", 0.0, purchased=False, best_reward=1.0),
    ]
    assert report.item_diagnostics(rows) == {
        "best_reward_mean": pytest.approx(2.5 / 3),
        "unsolvable_items": 1,
    }
    assert report.item_diagnostics([row("fixed_1", 1.0)]) == {}  # not recorded: none
    assert report.item_diagnostics([{"raw_id": "x", "score": None}]) == {}
    # The rollout zeroes every metric of a provider-unserved row, the
    # item's best reward included; such a row says nothing about the item.
    unserved = {
        **row("fixed_4", 0.0, purchased=False, best_reward=0.0),
        "provider_unserved": True,
    }
    assert report.item_diagnostics([*rows, unserved]) == report.item_diagnostics(rows)


def test_j_is_measured_against_a_score_of_one():
    assert report._truths(row("fixed_1", 1.0)) == {"success": True}
    assert report._truths(row("fixed_1", 0.99)) == {"success": False}
    assert report._truths(row("fixed_1", 0.0, purchased=False)) == {"success": False}


def _cell(spec: CellSpec, tmp_path: Path) -> report.CellData:
    return report.CellData(
        cell_id=spec.cell_id,
        paths=CellPaths(tmp_path / spec.cell_id),
        definition=spec.to_dict(),
        manifest={},
        status={},
    )


def test_a_pilot_cell_is_a_group_of_its_own(tmp_path):
    pilot = _cell(CellSpec("webshop", "luna", "none", "pilot"), tmp_path)
    formal = _cell(CellSpec("webshop", "luna", "none"), tmp_path)
    assert pilot.split_mode == "pilot" and pilot.group == "webshop-luna-pilot"
    assert formal.group == "webshop-luna"


# ---------------------------------------------------------------------------
# Pilot cells
# ---------------------------------------------------------------------------


def test_pilot_cells_are_webshops_no_memory_cells_on_the_pilot_split():
    spec = CellSpec("webshop", "luna", PILOT_METHOD, "pilot")
    assert spec.cell_id == "webshop-luna-none-pilot" and spec.test_split == "pilot"
    assert not spec.learns and spec.pilot and not spec.smoke
    for bad in (
        dict(benchmark="webshop", method="awm"),
        dict(benchmark="alfworld", method="none"),
        dict(benchmark="webshop", method="none", n_test=3),
    ):
        args: dict[str, Any] = {"model": "luna", "split_mode": "pilot", **bad}
        with pytest.raises(ValueError):
            CellSpec(**args)
    env = ws.WebShopEnv()
    learning, test = select_items(env, spec)
    assert (
        learning == [] and len(test) == protocol.EXTRA_SPLIT_SIZES["webshop"]["pilot"]
    )
    assert {t.split for t in test} == {"pilot"}
    formal = {t.raw_id for t in env.load_tasks("test")} | {
        t.raw_id for t in env.load_tasks("val")
    }
    assert not formal & {t.raw_id for t in test}  # disjoint from val (and test)


def test_the_study_tools_read_pilot_cell_ids():
    assert study.cell_id_method("webshop-luna-none-pilot") == "none"
    assert study.cell_id_method("webshop-luna56-awm") == "awm"
    assert study.cell_id_method("webshop-luna-none-pilotx") is None
    pattern = remove_method_data._method_run_pattern("none")
    assert pattern.match("webshop-luna-none-pilot-a01")
    assert pattern.match("webshop-luna-none-a02-0a8507e4")
    assert pattern.match("webshop-luna56-none-pilot-a01")
    assert pattern.match("alfworld-luna56-none-a03-0a8507e4")
    assert not pattern.match("alfworld-luna5-none-a01")


def test_the_schedule_builds_pilot_cells():
    from evals.skill_bench import schedule

    (cell,) = schedule.matrix(["webshop"], ["luna"], ["none"], split_mode="pilot")
    assert cell.cell_id == "webshop-luna-none-pilot"
    with pytest.raises(ValueError):
        schedule.matrix(["webshop"], ["luna"], ["awm"], split_mode="pilot")


def test_a_pilot_cell_runs_audits_and_reports(tmp_path):
    """End to end: the no-memory pilot cell over the image's real server code (the
    scripted shop), through ``run.py``'s runner on a real gateway over the fake upstream,
    then the audit (H9's pilot items) and the report (its own group)."""
    import asyncio

    from evals.skill_bench import audit
    from evals.skill_bench.cell_records import read_jsonl
    from evals.skill_bench.tests.e2e_support import E2E_MODEL, E2EModel, run_cells
    from evals.skill_bench.tests.fakes import FakeUpstream, serve_gateway
    from evals.skill_bench.tests.webshop_support import make_shop

    env = make_shop(tmp_path / "shop").env()
    spec = CellSpec("webshop", E2E_MODEL, "none", "pilot")
    model = E2EModel()
    fake = FakeUpstream(responder=model)
    runs = tmp_path / "runs"
    with serve_gateway(
        tmp_path, fake, backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S
    ) as (
        _gw,
        url,
    ):
        outcomes = asyncio.run(run_cells(runs, url, {"webshop": env}, [spec]))
    assert outcomes[spec.cell_id].status == "complete", outcomes[spec.cell_id].detail
    rows = read_jsonl(CellPaths(runs / spec.cell_id).results)
    pilot = {t.raw_id for t in env.load_tasks("pilot")}
    assert {r["raw_id"] for r in rows} == pilot
    assert {r["score"]["primary"] for r in rows} <= {0.0, 0.5, 1.0}
    assert any(r["score"]["metrics"]["success"] == 1.0 for r in rows)
    result = audit.audit_runs(
        runs,
        env_factory=lambda _b: env,
        expected_upstream="http://upstream",
        split_mode="pilot",
    )
    problems = [
        f"{f.invariant}:{f.check} {f.detail}"
        for cell in result.cells
        for f in cell.findings
        if f.severity == audit.VIOLATION
    ]
    assert result.passed and not problems, problems
    assert [c.cell_id for c in result.cells] == [spec.cell_id]
    data = report.build_report(runs, split_mode="pilot")
    (cell,) = data["cells"]
    assert cell["group"] == "webshop-luna-pilot" and cell["test"]["items"] == 50
    assert set(cell["test"]["metrics"]) == {m.name for m in report.METRICS["webshop"]}
    assert data["comparisons"] == []  # nothing to compare a pilot with
    assert report.build_report(runs)["cells"] == []


# ---------------------------------------------------------------------------
# Methods, failures, resources
# ---------------------------------------------------------------------------


def test_the_judge_reads_a_webshop_trajectory_without_a_final_answer():
    """J sees the trajectory (the score line included, learning phase only); no final
    answer is extracted for a text-action benchmark, and nothing WebShop-specific is
    added to any method: no tips, no success threshold."""
    item = contract_items("webshop")[0]
    text = judge.build_judge_input(item.trajectory(), "webshop").messages[1]["content"]
    assert judge.judge_format()["no_response"] in text
    from evals.skill_bench.methods import awm, mirix, reasoningbank

    for module in (awm, reasoningbank, mirix):
        assert "webshop" not in Path(module.__file__ or "").read_text().lower()
    prompts = protocol.SKILL_BENCH_DIR / "methods" / "prompts"
    for path in prompts.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            assert "webshop" not in path.read_text().lower(), path


def test_server_errors_are_infra_and_env_errors_are_harness():
    infra = failures.classify_exception(ws.WebShopServerError("HTTP 503: loading"))
    assert (infra.kind, infra.code) == (
        failures.FailureKind.INFRA,
        failures.WEBSHOP_SERVER,
    )
    harness = failures.classify_exception(ws.WebShopEnvError("bad request"))
    assert harness.kind == failures.FailureKind.HARNESS
    drift = failures.classify_exception(protocol.ProtocolDriftError("manifest"))
    assert drift.code == failures.PROTOCOL_DRIFT


def test_webshop_cells_have_a_memory_estimate():
    for method in protocol.METHODS:
        assert (
            resources.estimate_gb("webshop", method)
            == resources.CELL_MEMORY_GB["webshop"][method]
        )
    assert resources.estimate_gb("webshop", "mirix") <= resources.DEFAULT_CELL_MEMORY_GB
