"""report.py on constructed cell records with known statistics.

Every cell here is written by hand in ``run.py``'s layout (``cell_records.py``), with a
gateway ledger per attempt, so each number the report prints is known in advance: the
McNemar p-values, the bootstrap intervals, the learning-curve windows, the cost split by
side, role and attempt, the provider shares, the injection distribution and J's
confusion matrix.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol, report, study
from evals.skill_bench.cell_records import (
    CellPaths,
    records_digest,
    write_json_atomic,
)
from evals.skill_bench.tests.conftest import REMOVED_METHOD

FP = "f" * 64
LUNA = protocol.MODELS["luna"]
LUNA56 = protocol.MODELS["luna56"]
CALL_IDS = itertools.count(1)
WITHIN = f"success_within_{protocol.WEBSHOP_DIAGNOSTIC_STEPS}"


# --- building cells --------------------------------------------------------------------------


def _row(
    raw_id: str, primary: float, metrics: Mapping[str, float] | None = None, **kw: Any
):
    return {
        "raw_id": raw_id,
        "task_uid": "t" + raw_id.encode().hex()[:20].ljust(20, "0"),
        "score": {"primary": primary, "metrics": dict(metrics or {}), "answered": True},
        "primary": primary,
        "steps": 2,
        "actor_calls": 2,
        "model_failures": kw.pop("model_failures", 0),
        "format_errors": 0,
        **kw,
    }


def _call(
    role: str,
    phase: str,
    cost: float,
    *,
    provider: str = "P1",
    item: str | None = None,
    model: str = LUNA,
    served: str | None = None,
) -> dict[str, Any]:
    return {
        "event": "call",
        "call_id": f"c{next(CALL_IDS):06d}",
        "role": role,
        "phase": phase,
        "item": item,
        "model": protocol.EMBEDDING_MODEL if role == "embedding" else model,
        "model_served": served
        or (protocol.EMBEDDING_MODEL if role == "embedding" else model),
        "provider": provider,
        "status": "ok",
        "cost_total": cost,
        "billed_attempts": 1,
        "retries": 0,
        "input_tokens_estimated": 100,
        "input_tokens_actual": 110,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "latency_s": 0.1,
        "protocol_conformant": True,
    }


def make_cell(
    runs: Path,
    benchmark: str,
    method: str,
    *,
    test: Sequence[dict[str, Any]],
    learning: Sequence[dict[str, Any]] = (),
    calls: Sequence[dict[str, Any]] = (),
    invalid_calls: Sequence[dict[str, Any]] | None = None,
    injections: Sequence[dict[str, Any]] = (),
    judge: Sequence[dict[str, Any]] = (),
    snapshot: Mapping[str, Any] | None = None,
    status: str = "complete",
    fingerprint: str = FP,
    model_key: str = "luna",
    split_mode: str = "protocol",
    n_train: int = 2,
    n_test: int = 2,
    directory: str | None = None,
    retrievals: Sequence[dict[str, Any]] = (),
    started_at: str | None = None,
    created_at: str | None = None,
    attempts: Sequence[tuple[str, Sequence[dict[str, Any]], Mapping[str, Any]]]
    | None = None,
    row_attempt: bool = False,
    instance: str | None = None,
) -> Path:
    """One cell in run.py's layout; ``invalid_calls`` adds an invalidated attempt 1 first.

    ``attempts`` (state, ledger calls, extra attempt.json fields) replaces that: the last
    one is the cell's current attempt. ``started_at`` goes into the last attempt's
    attempt.json, ``created_at`` and ``instance`` into cell.json; ``row_attempt`` stamps
    the published rows with the last attempt's number (as run.py does)."""
    cell_id = f"{benchmark}-{model_key}-{method}"
    if split_mode == "smoke":
        cell_id += f"-smoke-l{n_train}-t{n_test}"
    root = runs / (directory or cell_id)
    if attempts is None:
        attempts = []
        if invalid_calls is not None:
            attempts.append(("invalid", invalid_calls, {}))
        attempts.append(("complete" if status == "complete" else "testing", calls, {}))
    for number, (state, ledger, extra) in enumerate(attempts, start=1):
        run_id = f"{cell_id}-a{number:02d}" + (f"-{instance}" if instance else "")
        ledger_path = runs / run_id / "gateway.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in ledger))
        attempt = root / f"attempt-{number:02d}"
        write_json_atomic(
            attempt / "attempt.json",
            {
                "attempt": number,
                "run_id": run_id,
                "state": state,
                "ledger": str(ledger_path),
                "test_reruns": [],
                **(
                    {"started_at": started_at}
                    if started_at and number == len(attempts)
                    else {}
                ),
                **extra,
            },
        )
    current = root / f"attempt-{len(attempts):02d}"
    if row_attempt:
        test = [{**r, "attempt": len(attempts)} for r in test]
        learning = [{**r, "attempt": len(attempts)} for r in learning]
    (current / "injections.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in injections)
    )
    if judge:
        (current / "method").mkdir(parents=True, exist_ok=True)
        (current / "method" / "judge.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in judge)
        )
    if retrievals:
        (current / "method").mkdir(parents=True, exist_ok=True)
        (current / "method" / "mirix_retrievals.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in retrievals)
        )
    write_json_atomic(
        current / "snapshots" / "frozen.json",
        {"sha256": "x", "snapshot": dict(snapshot or {"method": method})},
    )
    write_json_atomic(
        root / "cell.json",
        {
            "cell": {
                "cell_id": cell_id,
                "benchmark": benchmark,
                "model": protocol.MODELS[model_key],
                "model_key": model_key,
                "method": method,
                "split_mode": split_mode,
                "n_train": n_train if split_mode == "smoke" else None,
                "n_test": n_test if split_mode == "smoke" else None,
                "protocol_fingerprint": fingerprint,
                "test_items": [r["raw_id"] for r in test],
            },
            "current_attempt": len(attempts),
            **({"created_at": created_at} if created_at else {}),
            **({"instance": instance} if instance else {}),
        },
    )
    write_json_atomic(root / "status.json", {"status": status})
    if status == "complete":
        (root / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))
        (root / "learning.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in learning)
        )
        # The attempt's own rows (run.py publishes the cell's from them).
        (current / "results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in test)
        )
    return root


def binary(prefix: str, values: Sequence[int], **kw: Any) -> list[dict[str, Any]]:
    return [_row(f"{prefix}{i:03d}", float(v), **kw) for i, v in enumerate(values)]


def shop(prefix: str, scores: Sequence[float], **kw: Any) -> list[dict[str, Any]]:
    """WebShop test rows: the score is the primary one, success is a score of 1 (within
    15 steps too)."""
    return [
        _row(
            f"{prefix}{i:03d}",
            float(s),
            {"success": float(s == 1.0), WITHIN: float(s == 1.0)},
            **kw,
        )
        for i, s in enumerate(scores)
    ]


# --- statistics -------------------------------------------------------------------------------


def test_exact_mcnemar_on_known_counts():
    # n = 12 discordant pairs, min = 2: 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 158 / 4096.
    assert report.mcnemar_exact(10, 2) == pytest.approx(158 / 4096, abs=1e-15)
    assert report.mcnemar_exact(2, 10) == report.mcnemar_exact(10, 2)
    assert report.mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert report.mcnemar_exact(20, 0) == pytest.approx(2 / 2**20)
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(5, 5) == 1.0  # capped
    assert report.mcnemar_exact(1, 0) == 1.0
    # Large n stays exact (big integers) and matches the binomial CDF.
    b, c = 200, 160
    tail = sum(math.comb(b + c, i) for i in range(c + 1)) / 2 ** (b + c)
    assert report.mcnemar_exact(b, c) == pytest.approx(2 * tail, rel=1e-12)
    with pytest.raises(ValueError):
        report.mcnemar_exact(-1, 2)


def test_mcnemar_agrees_with_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    for b, c in ((10, 2), (7, 3), (30, 18), (0, 9), (4, 4)):
        expected = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        assert report.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-9)


def test_paired_bootstrap_on_known_differences():
    constant = report.paired_bootstrap([0.25] * 40)
    assert constant["delta"] == pytest.approx(0.25)
    assert (constant["ci_low"], constant["ci_high"]) == pytest.approx((0.25, 0.25))
    assert constant["resamples"] == protocol.BOOTSTRAP_RESAMPLES == 10_000
    assert constant["level"] == protocol.CI_LEVEL == 0.95
    assert constant["seed"] == protocol.SEED and constant["ci_method"] == "percentile"
    # Half +1, half 0 over 400 items: mean 0.5, standard error 0.025, so the 95% interval
    # is about 0.5 -/+ 1.96 * 0.025.
    half = report.paired_bootstrap([1.0, 0.0] * 200)
    assert half["delta"] == pytest.approx(0.5)
    assert half["ci_low"] == pytest.approx(0.5 - 1.96 * 0.025, abs=0.006)
    assert half["ci_high"] == pytest.approx(0.5 + 1.96 * 0.025, abs=0.006)
    assert half["excludes_zero"] is True
    # Reproducible, and centred differences give an interval around zero.
    assert report.paired_bootstrap([1.0, 0.0] * 200) == half
    centred = report.paired_bootstrap([1.0, -1.0] * 100)
    assert centred["ci_low"] < 0 < centred["ci_high"] and not centred["excludes_zero"]
    small = report.paired_bootstrap([0.1, 0.4, -0.2], resamples=500, seed=7)
    assert small["resamples"] == 500 and small["n"] == 3
    assert -0.2 - 1e-9 <= small["ci_low"] <= small["ci_high"] <= 0.4 + 1e-9
    with pytest.raises(ValueError):
        report.paired_bootstrap([])


def test_distribution_and_agreement():
    stats = report.distribution([0, 10, 20, 30, 40])
    assert stats == {
        "n": 5,
        "mean": 20.0,
        "median": 20.0,
        "p10": 4.0,
        "p90": 36.0,
        "min": 0.0,
        "max": 40.0,
        "total": 100.0,
    }
    assert report.distribution([]) == {"n": 0}
    pairs = (
        [(True, True)] * 3 + [(True, False)] + [(False, True)] + [(False, False)] * 3
    )
    a = report.agreement(pairs)
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (3, 1, 1, 3)
    assert a["accuracy"] == 0.75 and a["precision"] == 0.75 and a["recall"] == 0.75
    assert a["kappa"] == pytest.approx(0.5)  # expected agreement 0.5
    assert report.agreement([])["n"] == 0


# --- a constructed matrix -------------------------------------------------------------------------


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    # ALFWorld: 20 test items. The baseline wins items 0-5; ReasoningBank wins 0-3 and 6-15:
    # it loses 2 the baseline won (4, 5) and wins 10 the baseline lost (6-15): b=10, c=2.
    none_alf = [1] * 6 + [0] * 14
    rb_alf = [1] * 4 + [0, 0] + [1] * 10 + [0] * 4
    task_types = ["pick"] * 10 + ["heat"] * 10
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[
            _row(f"a{i:03d}", float(v), task_type=task_types[i])
            for i, v in enumerate(none_alf)
        ],
        calls=[_call("actor", "test", 0.01, item=f"u{i}") for i in range(4)],
        injections=[
            {
                "phase": "test",
                "raw_id": f"a{i:03d}",
                "tokens": 0,
                "injected_tokens_total": 0,
            }
            for i in range(20)
        ],
    )
    learning = binary("l", [1] * 10 + [0] * 10 + [1, 1, 1, 1, 0])
    for position, row in enumerate(learning):
        row["position"] = position
    make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=[
            _row(f"a{i:03d}", float(v), task_type=task_types[i])
            for i, v in enumerate(rb_alf)
        ],
        learning=learning,
        calls=[
            _call("embedding", "setup", 0.0, provider="Google"),
            _call("actor", "learning", 0.01, item="l1"),
            _call("actor", "learning", 0.01, item="l2"),
            _call("actor", "learning", 0.01, item="l3", provider="P2"),
            _call("judge", "learning", 0.02, item="l1"),
            _call("judge", "learning", 0.02, item="l2"),
            _call("actor", "test", 0.01, item="u1"),
            _call("actor", "test", 0.03, item="u2", provider="P2"),
            _call("embedding", "test", 0.001, provider="Google", item="u1"),
        ],
        invalid_calls=[
            _call("actor", "learning", 0.03, item="l1"),
            _call("judge", "learning", 0.03, item="l1"),
        ],
        injections=[
            {
                "phase": "test",
                "raw_id": f"a{i:03d}",
                "tokens": t,
                "injected_tokens_total": 3 * t,
            }
            for i, t in enumerate([0, 10, 20, 30, 40])
        ]
        + [
            {
                "phase": "learning",
                "raw_id": f"l{i:03d}",
                "tokens": 5,
                "injected_tokens_total": 5,
            }
            for i in range(3)
        ],
        judge=[
            # J against the truth of the first 8 learning items (truth: 1 for items 0-9):
            # tp 3, fp 1 (on a failed item), fn 1, tn 3 with the last items.
            {
                "event": "verdict",
                "task_uid": learning[i]["task_uid"],
                "success": s,
                "parsed": True,
            }
            for i, s in ((0, True), (1, True), (2, True), (3, False))
        ]
        + [
            {
                "event": "verdict",
                "task_uid": learning[i]["task_uid"],
                "success": s,
                "parsed": p,
            }
            for i, s, p in (
                (10, True, True),
                (11, False, True),
                (12, False, False),
                (13, False, True),
            )
        ],
        snapshot={
            "method": "reasoningbank",
            "counters": {"entries": 7},
            "judge": {"calls": 8},
        },
    )
    # WebShop: every score difference is +0.25 (the bootstrap interval is that point);
    # success (a score of 1): none 0 of 4, AWM 2 of 4, b=2, c=0.
    make_cell(runs, "webshop", "none", test=shop("w", [0.5, 0.25, 0.75, 0.75]))
    make_cell(runs, "webshop", "awm", test=shop("w", [0.75, 0.5, 1.0, 1.0]))
    # luna56: a method cell without a complete baseline, and a baseline still testing.
    make_cell(runs, "webshop", "awm", model_key="luna56", test=shop("o", [1.0, 0.0]))
    make_cell(
        runs, "webshop", "none", model_key="luna56", test=[], status="test_incomplete"
    )
    # A smoke cell, reported only on request, under another fingerprint.
    make_cell(
        runs,
        "webshop",
        "none",
        test=shop("s", [1.0]),
        split_mode="smoke",
        fingerprint="0" * 64,
    )
    return runs


def by_id(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["cell_id"]: c for c in data["cells"]}


def test_main_metrics_and_comparisons(runs):
    data = report.build_report(runs, frozen=FP)
    cells = by_id(data)
    assert sorted(cells) == [
        "alfworld-luna-none",
        "alfworld-luna-reasoningbank",
        "webshop-luna-awm",
        "webshop-luna-none",
        "webshop-luna56-awm",
        "webshop-luna56-none",
    ]  # the smoke cell only with split_mode None / "smoke"
    assert cells["alfworld-luna-none"]["test"]["metrics"] == {"success_rate": 0.3}
    assert cells["alfworld-luna-reasoningbank"]["test"]["metrics"] == {
        "success_rate": 0.7
    }
    assert cells["webshop-luna-none"]["test"]["metrics"] == {
        "success_rate": 0.0,
        "score": 0.5625,
        WITHIN: 0.0,
    }
    assert cells["webshop-luna-awm"]["test"]["metrics"] == {
        "success_rate": 0.5,
        "score": 0.8125,
        WITHIN: 0.5,
    }
    assert cells["alfworld-luna-none"]["test"]["unanswered"] == 0
    assert cells["webshop-luna56-none"]["complete"] is False
    assert "test" not in cells["webshop-luna56-none"]
    compared = {
        (c["group"], c["method"], c.get("metric")): c for c in data["comparisons"]
    }
    alf = compared[("alfworld-luna", "reasoningbank", "success_rate")]
    assert (alf["b"], alf["c"], alf["items"]) == (10, 2, 20)
    assert alf["p_value"] == pytest.approx(158 / 4096)
    assert alf["delta"] == pytest.approx(0.4) and alf["test"] == "mcnemar_exact"
    # McNemar on the successes: w002 and w003 right only with the method (b=2), none
    # the other way: p = 2 * C(2, 0) / 2^2.
    web = compared[("webshop-luna", "awm", "success_rate")]
    assert web["test"] == "mcnemar_exact"
    assert (web["b"], web["c"], web["p_value"], web["items"]) == (2, 0, 0.5, 4)
    assert (web["method_value"], web["none_value"]) == (0.5, 0.0)
    assert web["delta"] == pytest.approx(0.5)
    # The mean score: every paired difference is +0.25, so is every resample's mean.
    score = compared[("webshop-luna", "awm", "score")]
    assert score["test"] == "paired_bootstrap" and score["delta"] == pytest.approx(0.25)
    assert (score["ci_low"], score["ci_high"]) == pytest.approx((0.25, 0.25))
    assert [k for k in compared if k[0] == "webshop-luna"] == [
        ("webshop-luna", "awm", "success_rate"),
        ("webshop-luna", "awm", "score"),
        ("webshop-luna", "awm", WITHIN),
    ]
    assert compared[("webshop-luna56", "awm", None)]["skipped"].startswith(
        "no complete no-memory cell"
    )


def test_learning_curve_windows_of_ten(runs):
    cells = by_id(report.build_report(runs, frozen=FP))
    curve = cells["alfworld-luna-reasoningbank"]["learning_curve"]
    assert [
        (w["window"], w["first_position"], w["last_position"], w["items"])
        for w in curve
    ] == [
        (1, 0, 9, 10),
        (2, 10, 19, 10),
        (3, 20, 24, 5),
    ]
    assert [w["metrics"]["success_rate"] for w in curve] == [1.0, 0.0, 0.8]
    assert cells["alfworld-luna-none"]["learning_curve"] == []


def test_costs_by_side_role_and_attempt(runs):
    data = report.build_report(runs, frozen=FP)
    costs = by_id(data)["alfworld-luna-reasoningbank"]["costs"]
    assert costs["reported_attempt"] == 2
    assert costs["learning"]["cost"] == pytest.approx(0.07)  # setup + learning phases
    assert costs["learning"]["by_role"] == {
        "actor": {"cost": pytest.approx(0.03), "calls": 3, "unknown_cost_calls": 0},
        "embedding": {"cost": 0.0, "calls": 1, "unknown_cost_calls": 0},
        "judge": {"cost": pytest.approx(0.04), "calls": 2, "unknown_cost_calls": 0},
    }
    assert costs["test"]["cost"] == pytest.approx(0.041)
    assert costs["test"]["calls"] == 3
    assert costs["other_attempts"] == {
        "attempts": [1],
        "states": {"1": "invalid"},
        "cost": pytest.approx(0.06),
        "calls": 2,
        # The harness invalidated it: no --new-attempt --note superseded it.
        "superseded": [],
        "superseded_cost": 0.0,
        "invalid_cost": pytest.approx(0.06),
    }
    assert costs["total_cost"] == pytest.approx(0.171)
    assert costs["actor_cost_per_test_item"] == pytest.approx(0.02)  # (0.01 + 0.03) / 2
    assert costs["learning"]["input_tokens_actual"] == 6 * 110
    matrix = data["costs"]
    assert matrix["total"] == pytest.approx(0.171 + 0.04)
    assert matrix["invalidated_attempts"] == pytest.approx(0.06)
    assert matrix["by_side"] == {
        "learning": pytest.approx(0.07),
        "test": pytest.approx(0.081),
    }
    assert matrix["by_role"]["judge"] == pytest.approx(0.04)
    assert matrix["by_benchmark"]["alfworld"] == pytest.approx(0.211)


def test_providers_injections_judge_and_task_types(runs):
    cells = by_id(report.build_report(runs, frozen=FP))
    rb = cells["alfworld-luna-reasoningbank"]
    providers = rb["providers"]
    assert providers["overall"] == {"P1": 5, "P2": 2, "Google": 2}
    assert providers["shares"]["P2"] == pytest.approx(2 / 9)
    assert providers["by_role"]["judge"] == {"P1": 2}
    assert providers["by_role"]["actor"] == {"P1": 3, "P2": 2}
    assert providers["model_served"][LUNA] == {LUNA: 7}
    test_inj = rb["injections"]["test"]
    assert (test_inj["items"], test_inj["empty"]) == (5, 1)
    assert test_inj["tokens"]["mean"] == 20.0 and test_inj["tokens"]["p90"] == 36.0
    assert test_inj["tokens_times_actor_calls"]["mean"] == 60.0
    assert rb["injections"]["learning"]["tokens"]["mean"] == 5.0
    judge = rb["judge_agreement"]
    assert (judge["verdicts"], judge["joined"], judge["unparseable"]) == (8, 8, 1)
    success = judge["by_truth"]["success"]
    assert (success["tp"], success["fp"], success["fn"], success["tn"]) == (3, 1, 1, 3)
    assert success["kappa"] == pytest.approx(0.5)
    assert cells["alfworld-luna-none"]["judge_agreement"] is None
    assert rb["task_types"] == {
        "heat": {"items": 10, "success_rate": 0.6},
        "pick": {"items": 10, "success_rate": 0.8},
    }
    assert cells["webshop-luna-none"]["task_types"] is None
    assert rb["method_counters"] == {"counters": {"entries": 7}, "judge": {"calls": 8}}


def test_fingerprints_markdown_and_files(runs, tmp_path):
    data = report.build_report(runs, split_mode=None, frozen=FP)
    assert data["protocol"]["frozen_fingerprint"] == FP
    assert data["protocol"]["cells_not_frozen"] == ["webshop-luna-none-smoke-l2-t2"]
    assert "webshop-luna-none-smoke-l2-t2" in by_id(data)
    assert report.CAVEAT_SINGLE_RUN in data["caveats"]
    smoke = report.build_report(runs, split_mode="smoke", frozen=FP)
    assert [c["cell_id"] for c in smoke["cells"]] == ["webshop-luna-none-smoke-l2-t2"]
    json_path, md_path = report.write_report(data, tmp_path / "out")
    assert json.loads(json_path.read_text())["protocol"]["frozen_fingerprint"] == FP
    md = md_path.read_text()
    assert FP in md and report.CAVEAT_SINGLE_RUN in md and report.CAVEAT_JUDGE in md
    assert "McNemar b=10 c=2 p=0.03857" in md
    assert (
        "| webshop-luna | awm | success_rate | 4 | 0.500 | 0.000 | "
        "+0.500 | McNemar b=2 c=0 p=0.5 |"
    ) in md
    assert (
        "| webshop-luna | awm | score | 4 | 0.812 | 0.562 | +0.250 | "
        "CI [+0.250, +0.250] |"
    ) in md
    # The intro names McNemar on the environment outcomes and the mean score's bootstrap.
    intro = _section(md, "Comparison with the no-memory method")[2]
    assert "on the two cells' environment outcomes" in intro
    assert "paired bootstrap interval" in intro
    assert (
        "| `alfworld-luna-reasoningbank` | success | 8 | 3 | 1 | 1 | 3 | 0.750 | 0.500 |"
        in md
    )
    # The smoke cell's fingerprint is no known version (an earlier one would only be
    # marked "(earlier)").
    assert data["protocol"]["cells_unknown_version"] == [
        "webshop-luna-none-smoke-l2-t2"
    ]
    assert data["protocol"]["cells_earlier_version"] == []
    # Compared and marked, never refused (schema 3).
    assert "**Cells under a protocol version no freeze registered**" in md
    assert f"`{'0' * 12}` (**unknown**)" in md
    assert "heat" in md and "0.600 (n=10)" in md


def test_the_cli_writes_both_files(runs, tmp_path, capsys):
    out = tmp_path / "rep"
    assert (
        report.main(["--runs-dir", str(runs), "--out", str(out), "--resamples", "200"])
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["complete"] == 5 and printed["cells"] == 6
    assert (out / "report.json").is_file() and (out / "report.md").is_file()
    data = json.loads((out / "report.json").read_text())
    assert data["statistics"]["resamples"] == 200
    empty = tmp_path / "empty"
    empty.mkdir()
    assert report.main(["--runs-dir", str(empty), "--out", str(tmp_path / "e")]) == 2


def test_a_metric_that_is_not_binary_is_refused(tmp_path):
    runs = tmp_path / "runs"
    make_cell(runs, "alfworld", "none", test=[_row("a1", 0.5)])
    with pytest.raises(ValueError, match="not 0/1"):
        report.build_report(runs, frozen=FP)


# --- audit status, reruns and versions -----------------------------------------------------


def _audit_entry(runs: Path, cell_id: str, **kw: Any) -> dict[str, Any]:
    """audit.py's record of one cell that passed on the records now on disk."""
    from evals.skill_bench.cell_records import CellPaths, records_digest

    entry = {
        "cell_id": cell_id,
        "root": str(runs / cell_id),
        "audited": True,
        "attempt": 2 if cell_id == "alfworld-luna-reasoningbank" else 1,
        "violations": 0,
        "info": {"records_sha256": records_digest(CellPaths(runs / cell_id))},
    }
    entry.update(kw)
    return entry


def _write_audit(runs: Path, cells: Sequence[dict[str, Any]], **kw: Any) -> Path:
    path = runs / "_audit" / "audit.json"
    write_json_atomic(
        path,
        {
            "generated_at": "2026-09-28T00:00:00+00:00",
            "passed": not any(c.get("violations") for c in cells),
            "findings": kw.get("findings", []),
            "cells": list(cells),
        },
    )
    return path


def test_every_cell_is_marked_with_its_audit_status(runs, tmp_path, capsys):
    """A number is vouched for only by an audit that passed on the very records the
    report prints; --require-audit leaves every other cell out of the comparisons."""
    data = report.build_report(runs, frozen=FP)
    assert {c["audit"]["status"] for c in data["cells"]} == {report.AUDIT_MISSING}
    assert data["audit"]["found"] is False
    assert "**No audit**" in report.render_markdown(data)
    _write_audit(
        runs,
        [
            _audit_entry(runs, "alfworld-luna-none"),
            _audit_entry(runs, "alfworld-luna-reasoningbank", violations=2),
            _audit_entry(runs, "webshop-luna-none"),
            _audit_entry(runs, "webshop-luna-awm", attempt=2),
            _audit_entry(runs, "webshop-luna56-none", audited=False, reason="testing"),
        ],
    )
    # webshop-luna-none changed after its audit: its row edited by hand.
    results = runs / "webshop-luna-none" / "results.jsonl"
    results.write_text(results.read_text().replace('"primary": 0.5', '"primary": 1.0'))
    statuses = {
        c["cell_id"]: c["audit"]["status"]
        for c in report.build_report(runs, frozen=FP)["cells"]
    }
    assert statuses == {
        "alfworld-luna-none": report.AUDIT_PASSED,
        "alfworld-luna-reasoningbank": report.AUDIT_FAILED,
        "webshop-luna-none": report.AUDIT_STALE,
        "webshop-luna-awm": report.AUDIT_STALE,
        "webshop-luna56-awm": report.AUDIT_NOT_AUDITED,
        "webshop-luna56-none": report.AUDIT_NOT_AUDITED,
    }
    marked = report.build_report(runs, frozen=FP)
    assert any(c.get("method_value") is not None for c in marked["comparisons"])
    required = report.build_report(runs, frozen=FP, require_audit=True)
    skipped = {(c["group"], c["method"]): c["skipped"] for c in required["comparisons"]}
    assert skipped[("alfworld-luna", "reasoningbank")].startswith("no passing audit")
    assert (
        "alfworld-luna-reasoningbank (failed)"
        in skipped[("alfworld-luna", "reasoningbank")]
    )
    assert skipped[("webshop-luna", "awm")].startswith("no passing audit")
    md = report.render_markdown(required)
    assert (
        "**failed**" in md and "**stale**" in md and "left out of the comparisons" in md
    )
    out = tmp_path / "rep"
    argv = ["--runs-dir", str(runs), "--out", str(out), "--resamples", "100"]
    assert report.main(argv) == 0
    assert report.main([*argv, "--require-audit"]) == 3
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "alfworld-luna-reasoningbank" in printed[-1]["not_audited"]
    # A violation across cells fails every cell's mark.
    _write_audit(
        runs,
        [_audit_entry(runs, "alfworld-luna-none")],
        findings=[
            {"severity": "violation", "check": "first_prompt_differs_across_cells"}
        ],
    )
    cells = by_id(report.build_report(runs, frozen=FP))
    assert cells["alfworld-luna-none"]["audit"]["status"] == report.AUDIT_FAILED
    # --audit reads another file.
    other = _write_audit(tmp_path / "elsewhere", [])
    cells = by_id(report.build_report(runs, frozen=FP, audit_path=other))
    assert cells["alfworld-luna-none"]["audit"]["status"] == report.AUDIT_NOT_AUDITED


def test_cells_run_more_than_once_are_listed_with_their_notes(runs):
    """The single-run caveat does not hold for a cell run again from a new
    directory. The report lists the operator note (run.py --acknowledge-rerun) and the
    earlier tested runs the audit found, and names the cells in the caveats, whatever
    the audit status (an acknowledged re-run passes it)."""
    data = report.build_report(runs, frozen=FP)
    assert data["cells_run_more_than_once"] == []
    assert data["caveats"] == list(report.CAVEATS)
    assert "Runs and attempts" not in report.render_markdown(data)

    manifest_path = runs / "alfworld-luna-reasoningbank" / "cell.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["operator_notes"] = [
        {
            "kind": "rerun_acknowledged",
            "file": "operator_notes/0123456789ab-rerun.md",
            "sha256": "0" * 64,
            "at": "2026-09-28T10:00:00+00:00",
            "superseded": [
                {
                    "run_id": "alfworld-luna-reasoningbank-a01-0dd0dd01",
                    "instance": "0dd0dd01",
                    "test_actor_calls": 20,
                    "calls": 90,
                }
            ],
        },
        {"kind": "something_else", "file": "x"},
    ]
    write_json_atomic(manifest_path, manifest)
    earlier = {
        "run_id": "alfworld-luna-none-a01-0dd0dd01",
        "instance": "0dd0dd01",
        "attempt": 1,
        "test_phase": True,
        "test_actor_calls": 20,
        "first_test_actor_at": "2026-09-27T09:00:00+00:00",
        "calls": 20,
    }
    untested = {**earlier, "run_id": "alfworld-luna-none-a02-0dd0dd01"}
    untested.update(test_actor_calls=0, first_test_actor_at=None)
    _write_audit(
        runs,
        [
            _audit_entry(
                runs, "alfworld-luna-none", violations=1
            ),  # cell_rerun_after_test
            _audit_entry(runs, "alfworld-luna-reasoningbank"),
        ],
    )
    audit_path = runs / "_audit" / "audit.json"
    audit_data = json.loads(audit_path.read_text())
    audit_data["cells"][0]["info"]["other_gateway_runs"] = [earlier, untested]
    audit_path.write_text(json.dumps(audit_data))

    data = report.build_report(runs, frozen=FP)
    assert data["cells_run_more_than_once"] == [
        "alfworld-luna-none",
        "alfworld-luna-reasoningbank",
    ]
    cells = by_id(data)
    none_runs = cells["alfworld-luna-none"]["reruns"]
    assert none_runs["operator_notes"] == []
    assert [r["run_id"] for r in none_runs["earlier_tested_runs"]] == [
        earlier["run_id"]
    ]
    (note,) = cells["alfworld-luna-reasoningbank"]["reruns"]["operator_notes"]
    assert note["at"] == "2026-09-28T10:00:00+00:00"
    assert note["superseded"] == [
        {
            "run_id": "alfworld-luna-reasoningbank-a01-0dd0dd01",
            "instance": "0dd0dd01",
            "test_actor_calls": 20,
        }
    ]
    assert cells["webshop-luna-none"]["reruns"] is None
    # A note over earlier runs that never ran a test item: started over, run once.
    other = runs / "webshop-luna-awm" / "cell.json"
    restarted = json.loads(other.read_text())
    restarted["operator_notes"] = [
        {
            **manifest["operator_notes"][0],
            "superseded": [
                {
                    "run_id": "webshop-luna-awm-a01-0dd0dd01",
                    "instance": "0dd0dd01",
                    "test_actor_calls": 0,
                }
            ],
        }
    ]
    write_json_atomic(other, restarted)
    again = report.build_report(runs, frozen=FP)
    assert by_id(again)["webshop-luna-awm"]["reruns"] is None
    assert "webshop-luna-awm" not in again["cells_run_more_than_once"]
    caveat = data["caveats"][-1]
    assert caveat.startswith("Not every cell ran once")
    assert "alfworld-luna-none, alfworld-luna-reasoningbank" in caveat
    md = report.render_markdown(data)
    assert "## Runs and attempts" in md
    assert f"`{earlier['run_id']}` (20 test actor calls, from 2026-09-27" in md
    assert "`operator_notes/0123456789ab-rerun.md` (2026-09-28T10:00:00+00:00" in md
    assert "**none**" in md  # alfworld-luna-none: no operator note
    # The comparison section repeats the exception next to the single-run caveat.
    comparison = md.split("## Comparison with the no-memory method")[1]
    assert "Not every cell ran once" in comparison.split("##")[0]


EARLIER = "1" * 64
HARNESS = "a" * 64


def _set_code(cell: Path, harness: str | None, commit: str = "c0ffee0") -> None:
    """Record the code and commit of the cell's last attempt (the reported one here)."""
    number = max(int(p.name[8:]) for p in cell.glob("attempt-*"))
    _set_state(
        cell,
        number,
        code={"harness": harness, "harness_files": 31},
        git={"commit": commit},
    )


def test_cells_of_other_versions_are_compared_and_flagged(runs):
    """Versions are recorded and reported, not enforced across cells (a baseline fixed
    after its run re-runs only its own cells). Cells of different known protocol
    versions or code are compared (the audit gate stays) and every such row says so. A
    cell under a fingerprint no freeze registered is compared too and marked
    ``same_protocol`` = ``unknown`` (schema 3: record, don't block; the audit warns)."""
    manifest = runs / "alfworld-luna-reasoningbank" / "cell.json"
    data = json.loads(manifest.read_text())
    data["cell"]["protocol_fingerprint"] = EARLIER
    write_json_atomic(manifest, data)
    for name in ("webshop-luna-none", "webshop-luna-awm"):
        _set_code(runs / name, HARNESS)
    _set_code(runs / "alfworld-luna-none", HARNESS)
    _set_code(runs / "alfworld-luna-reasoningbank", "b" * 64, commit="decafbad")
    known_earlier = {
        FP: {"fingerprint": FP, "current": True},
        EARLIER: {"fingerprint": EARLIER, "current": False},
    }

    # EARLIER as a known earlier version: compared, and flagged.
    flagged = report.build_report(runs, frozen=FP, versions=known_earlier)
    alf = [c for c in flagged["comparisons"] if c["group"] == "alfworld-luna"]
    assert alf and all("skipped" not in c for c in alf)
    (row,) = alf
    assert (row["same_protocol"], row["same_code"]) == (False, False)
    assert row["method_versions"] == {
        "protocol": EARLIER,
        "protocol_current": False,
        "harness": "b" * 64,
        "mirix": None,
        "commit": "decafbad",
        # One version over the whole attempt (a legacy attempt: one segment).
        "mixed": False,
        "protocol_fingerprints": [EARLIER],
    }
    assert (row["protocol_unknown"], row["mixed_version"]) == ([], [])
    assert row["none_versions"]["protocol"] == FP
    assert row["none_versions"]["protocol_current"] is True
    table = _section(
        report.render_markdown(flagged), "Comparison with the no-memory method"
    )
    assert any(
        "| alfworld-luna | reasoningbank |" in line
        and line.endswith("| **differ: protocol, code** |")
        for line in table
    )

    # EARLIER as a fingerprint no freeze registered: compared, marked unknown.
    result = report.build_report(runs, frozen=FP)
    compared = result["comparisons"]
    assert _skips(result)[("alfworld-luna", "reasoningbank")] is None
    (unknown_row,) = [
        c for c in compared if c["group"] == "alfworld-luna" and "skipped" not in c
    ]
    assert unknown_row["same_protocol"] == "unknown"
    assert unknown_row["protocol_unknown"] == ["alfworld-luna-reasoningbank"]
    assert [
        w["cells"]
        for w in result["warnings"]
        if w["check"] == "unknown_protocol_version"
    ] == [["alfworld-luna-reasoningbank"]]
    web = [c for c in compared if c["group"] == "webshop-luna"]
    assert len(web) == 3 and all("skipped" not in c for c in web)
    assert {(c["same_protocol"], c["same_code"]) for c in web} == {(True, True)}
    # A benchmark with more than one version gets the caveat, naming it.
    groups = result["protocol"]["versions"]
    assert groups["alfworld"]["protocol"] == {
        EARLIER: ["alfworld-luna-reasoningbank"],
        FP: ["alfworld-luna-none"],
    }
    assert groups["alfworld"]["harness"] == {
        HARNESS: ["alfworld-luna-none"],
        "b" * 64: ["alfworld-luna-reasoningbank"],
    }
    assert groups["webshop"]["protocol"] == {
        FP: [
            "webshop-luna-awm",
            "webshop-luna-none",
            "webshop-luna56-awm",
            "webshop-luna56-none",
        ]
    }
    caveat = result["caveats"][-1]
    assert caveat.startswith(report.CAVEAT_MIXED_VERSIONS)
    assert caveat.endswith("Mixed here: alfworld.")
    cells = by_id(result)
    assert cells["alfworld-luna-reasoningbank"]["protocol"] == {
        "fingerprint": EARLIER,
        "current": False,
        "known": False,
    }
    assert cells["alfworld-luna-reasoningbank"]["code"] == {
        "harness": "b" * 64,
        "mirix": None,
        "commit": "decafbad",
    }
    assert result["protocol"]["cells_unknown_version"] == [
        "alfworld-luna-reasoningbank"
    ]
    md = report.render_markdown(result)
    table = _section(md, "Comparison with the no-memory method")
    assert any(
        "| alfworld-luna | reasoningbank |" in line
        and line.endswith(
            "| **protocol not registered: alfworld-luna-reasoningbank**; "
            "**differ: code** |"
        )
        for line in table
    )
    assert any(
        "| webshop-luna | awm |" in line and line.endswith("| same |") for line in table
    )
    cells_md = _section(md, "Cells")
    assert any(
        "`alfworld-luna-reasoningbank`" in line
        and f"`{EARLIER[:12]}` (**unknown**)" in line
        and f"`{'b' * 12}`" in line
        for line in cells_md
    )
    assert any(
        "`webshop-luna-none`" in line and f"| `{FP[:12]}` | `{HARNESS[:12]}` |" in line
        for line in cells_md
    )

    # A known earlier version: "(earlier)" with its freeze reason in the header.
    known = {
        FP: {"fingerprint": FP, "current": True},
        EARLIER: {
            "fingerprint": EARLIER,
            "current": False,
            "frozen_at": "2026-09-29T15:43:01+00:00",
            "reason": "an earlier freeze",
        },
    }
    earlier = report.build_report(runs, frozen=FP, versions=known)
    assert by_id(earlier)["alfworld-luna-reasoningbank"]["protocol"]["known"] is True
    assert earlier["protocol"]["cells_earlier_version"] == [
        "alfworld-luna-reasoningbank"
    ]
    assert earlier["protocol"]["cells_unknown_version"] == []
    assert earlier["protocol"]["earlier_versions"] == {
        EARLIER: {
            "frozen_at": "2026-09-29T15:43:01+00:00",
            "reason": "an earlier freeze",
            "replaced_by": None,
            "differs_from_frozen": protocol.describe_version_diff(None),
        }
    }
    md = report.render_markdown(earlier)
    assert f"`{EARLIER[:12]}` (earlier)" in md
    assert "an earlier freeze" in md and "Cells under an earlier protocol" in md
    # With both freezes' times and file maps recorded, the header also says which
    # freeze replaced the earlier version and what differs.
    awm = "methods/prompts/awm/format.json"
    mapped = {
        FP: {
            "fingerprint": FP,
            "current": True,
            "frozen_at": "2026-09-30T04:00:00+00:00",
            "reason": "a prompt change",
            "files": {awm: "new", "config/models.json": "same"},
            "constants_sha256": "c",
        },
        EARLIER: dict(known[EARLIER], files={awm: "old", "config/models.json": "same"})
        | {"constants_sha256": "c"},
    }
    replaced = report.build_report(runs, frozen=FP, versions=mapped)
    version = replaced["protocol"]["earlier_versions"][EARLIER]
    assert version["replaced_by"] == {
        "fingerprint": FP,
        "frozen_at": "2026-09-30T04:00:00+00:00",
        "reason": "a prompt change",
    }
    assert version["differs_from_frozen"] == f"changed: {awm}"
    header = report._version_header_lines(replaced["protocol"])
    assert (
        f"    - replaced by `{FP[:12]}` (frozen 2026-09-30T04:00:00+00:00): a prompt change"
        in header
    )
    assert f"    - differs from the frozen value: changed: {awm}" in header
    # Rendered from report.json (sorted keys) the version columns are the same.
    stored = json.loads(json.dumps(earlier, sort_keys=True))
    for title in ("Cells", "Comparison with the no-memory method"):
        assert _section(report.render_markdown(stored), title) == _section(md, title)
    # A report.json written before the cells carried versions still renders.
    assert report._protocol_text(
        {"protocol_fingerprint": FP}, {"frozen_fingerprint": FP}
    ) == (f"`{FP[:12]}`")
    assert report._protocol_text(
        {"protocol_fingerprint": EARLIER}, {"frozen_fingerprint": FP}
    ) == (f"`{EARLIER[:12]}` (**unknown**)")
    assert report._comparison_versions_text({"metric": "score"}) == "-"
    assert report._version_header_lines({"cells_not_frozen": ["x"]}) == [
        "- **Cells whose fingerprint is not the frozen one (H10.6):** `x`"
    ]
    # Another frozen value that keeps FP as an earlier version: compared, none current.
    other = report.build_report(
        runs,
        frozen="2" * 64,
        versions={FP: {"fingerprint": FP, "current": False}},
    )["comparisons"]
    assert all("skipped" not in c for c in other if c["group"] == "webshop-luna")
    assert {
        c["method_versions"]["protocol_current"]
        for c in other
        if c["group"] == "webshop-luna"
    } == {False}
    # One that does not know FP at all: every pair is compared, marked unknown.
    unknown = report.build_report(runs, frozen="2" * 64)["comparisons"]
    paired = [c for c in unknown if c["group"] in ("webshop-luna", "alfworld-luna")]
    assert paired and all("skipped" not in c for c in paired)
    assert {c["same_protocol"] for c in paired} == {"unknown"}


def test_an_older_cell_reports_the_manifests_protocol_and_no_code_is_unknown(
    tmp_path,
):
    """A cell written before attempts recorded their protocol reports cell.json's (the
    run.py of that time refused any other), or the protocol_history entry that lists its
    attempt once the cell adopted another one; an attempt without a code record makes
    the comparison's ``same_code`` unknown, not false."""
    runs = tmp_path / "runs"
    base = [_row("a1", 1.0)]
    none = make_cell(runs, "alfworld", "none", test=base)
    ours = make_cell(runs, "alfworld", "awm", test=base)
    _set_code(ours, HARNESS)
    data = report.build_report(runs, frozen=FP)
    assert by_id(data)["alfworld-luna-none"]["protocol"]["fingerprint"] == FP
    (row,) = [c for c in data["comparisons"] if c["metric"] == "success_rate"]
    assert (row["same_protocol"], row["same_code"]) == (True, None)
    table = _section(report.render_markdown(data), "Comparison with the no-memory")
    assert any(line.endswith("| not recorded: code |") for line in table)
    assert report.CAVEAT_MIXED_VERSIONS not in " ".join(data["caveats"])

    # The cell adopted another protocol after its attempt ran: the history, which lists
    # the attempt, says what it ran under; once the attempt records its own, that wins.
    manifest = json.loads((none / "cell.json").read_text())
    manifest["cell"]["protocol_fingerprint"] = "3" * 64
    manifest["protocol_history"] = [
        {"fingerprint": FP, "constants_sha256": "c", "attempts": [1]}
    ]
    write_json_atomic(none / "cell.json", manifest)
    listed = by_id(report.build_report(runs, frozen=FP))["alfworld-luna-none"]
    assert listed["protocol"]["fingerprint"] == FP
    _set_state(none, protocol={"fingerprint": "3" * 64, "constants_sha256": "d"})
    data = report.build_report(runs, frozen=FP)
    assert by_id(data)["alfworld-luna-none"]["protocol_fingerprint"] == "3" * 64
    # "3" * 64 is no freeze's here: compared all the same, marked unknown.
    assert _skips(data)[("alfworld-luna", "awm")] is None
    (row,) = [c for c in data["comparisons"] if c["metric"] == "success_rate"]
    assert (row["same_protocol"], row["protocol_unknown"]) == (
        "unknown",
        ["alfworld-luna-none"],
    )
    # Once a freeze produced it (a later re-freeze), the pair is compared and flagged.
    data = report.build_report(
        runs,
        frozen=FP,
        versions={
            FP: {"fingerprint": FP, "current": True},
            "3" * 64: {"fingerprint": "3" * 64, "current": False},
        },
    )
    (row,) = [c for c in data["comparisons"] if c["metric"] == "success_rate"]
    assert (
        row["same_protocol"] is False and row["none_versions"]["protocol"] == "3" * 64
    )
    assert data["caveats"][-1].endswith("Mixed here: alfworld.")


def test_smoke_groups_ignore_n_train_and_duplicate_cells_compare_the_newest(tmp_path):
    """Smoke test items depend only on n_test, so a no-memory cell run with
    --n-train 0 pairs with a learning cell. Two complete cells of one (group, method)
    are no error (record, don't block): the one whose
    reported attempt started last is compared, the other gets a skipped row naming it,
    the header warns, and the CLI exits 0."""
    runs = tmp_path / "runs"
    base = [_row("s1", 0.0), _row("s2", 1.0)]
    better = [_row("s1", 1.0), _row("s2", 1.0)]
    make_cell(
        runs,
        "alfworld",
        "none",
        test=base,
        split_mode="smoke",
        n_train=0,
        started_at="2026-09-29T10:00:00+00:00",
    )
    make_cell(runs, "alfworld", "awm", test=better, split_mode="smoke")
    data = report.build_report(runs, split_mode="smoke", frozen=FP)
    assert {c["group"] for c in data["cells"]} == {"alfworld-luna-smoke-t2"}
    assert data["errors"] == []
    assert "duplicate_group_method" not in {w["check"] for w in data["warnings"]}
    compared = next(c for c in data["comparisons"] if c.get("metric") == "success_rate")
    assert compared["delta"] == pytest.approx(0.5)
    assert (compared["b"], compared["c"]) == (1, 0)
    # A second no-memory cell (n_train 2), started later: every answer right.
    make_cell(
        runs,
        "alfworld",
        "none",
        test=better,
        split_mode="smoke",
        n_train=2,
        started_at="2026-09-30T10:00:00+00:00",
    )
    data = report.build_report(runs, split_mode="smoke", frozen=FP)
    assert data["errors"] == []
    (warning,) = [w for w in data["warnings"] if w["check"] == "duplicate_group_method"]
    assert warning["cells"] == [
        "alfworld-luna-none-smoke-l0-t2",
        "alfworld-luna-none-smoke-l2-t2",
    ]
    compared = [c for c in data["comparisons"] if "skipped" not in c]
    (row,) = compared
    assert (row["none_value"], row["delta"]) == (1.0, 0.0)  # the newer none cell's
    assert [c["skipped"] for c in data["comparisons"] if "skipped" in c] == [
        "a newer complete cell of this group and method is compared: "
        "alfworld-luna-none-smoke-l2-t2"
    ]
    md = report.render_markdown(data)
    assert "**Error:**" not in md
    assert "- **Warning (duplicate_group_method):** 2 complete none cells" in md
    argv = [
        "--runs-dir",
        str(runs),
        "--split-mode",
        "smoke",
        "--out",
        str(tmp_path / "o"),
    ]
    assert report.main(argv) == 0


def test_costs_the_ledgers_cannot_count_are_listed(tmp_path):
    """An attempt without a ledger and a call without its final line (a gateway
    stopped mid-call) have an unknown cost: listed next to the totals, not hidden."""
    runs = tmp_path / "runs"
    root = make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=[_row("a1", 1.0)],
        calls=[
            _call("actor", "test", 0.01, item="u1"),
            {"event": "call_started", "call_id": "cut-off", "role": "actor"},
        ],
        invalid_calls=[_call("actor", "learning", 0.03, item="l1")],
    )
    (runs / f"{root.name}-a01" / "gateway.jsonl").unlink()
    costs = by_id(report.build_report(runs, frozen=FP))[root.name]["costs"]
    assert costs["ledgers_missing"] == [1] and costs["unfinished_calls"] == 1
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert "**Cost not in the totals:**" in md
    assert "no ledger for attempt(s) 1; 1 unfinished call(s)" in md


def test_mirix_diagnostics_of_a_mirix_cell(tmp_path):
    """MIRIX's learning runs, usage, skills and retrievals, from its frozen snapshot
    and its retrieval records (the shape the stub MIRIX and MirixMethod write)."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "mirix",
        test=[_row("a1", 1.0), _row("a2", 0.0)],
        snapshot={
            "method": "mirix",
            "learning_runs": [
                {
                    "id": "r1",
                    "status": "succeeded",
                    "trigger": "threshold",
                    "session_ids": ["s1", "s2"],
                    "llm_usage": {
                        "calls": 2,
                        "prompt_tokens": 80,
                        "completion_tokens": 30,
                    },
                    "retry_events": [{"attempt": 1}],
                },
                {
                    "id": "r2",
                    "status": "failed",
                    "trigger": "flush",
                    "session_ids": ["s3"],
                    "error": "gateway 504",
                    "llm_usage": {
                        "calls": 1,
                        "prompt_tokens": 40,
                        "completion_tokens": 0,
                    },
                },
            ],
            "skills": [{"id": "k1"}, {"id": "k2"}],
            "skill_history": {"k1": {"versions": [1, 2]}, "k2": [1]},
        },
        retrievals=[
            {"event": "retrieval", "frozen": False, "item_ids": ["k1"]},
            {"event": "retrieval", "frozen": True, "item_ids": ["k1", "k2"]},
            {
                "event": "retrieval",
                "frozen": True,
                "item_ids": [],
                "evidence": {"extraction_status": "fallback"},
                "failure": None,
            },
            {  # refused: the item was re-run, not scored on it
                "event": "retrieval",
                "frozen": True,
                "item_ids": ["k1"],
                "failure": {
                    "kind": "infra",
                    "code": "mirix_retrieval",
                    "role": "mirix",
                },
            },
        ],
    )
    cell = report.build_report(runs, frozen=FP)["cells"][0]
    mirix = cell["mirix"]
    assert mirix["learning_runs"] == 2
    assert mirix["runs_by_status"] == {"succeeded": 1, "failed": 1}
    assert mirix["runs_by_trigger"] == {"threshold": 1, "flush": 1}
    assert mirix["sessions_per_run"]["mean"] == 1.5
    assert mirix["run_errors"] == ["gateway 504"] and mirix["retry_events"] == 1
    assert mirix["llm_usage"] == {
        "calls": 3,
        "prompt_tokens": 120,
        "completion_tokens": 30,
    }
    assert mirix["skills"] == 2 and mirix["skill_versions"] == 3
    assert mirix["retrievals"]["learning"]["count"] == 1
    assert (
        mirix["retrievals"]["test"]["count"],
        mirix["retrievals"]["test"]["empty"],
        mirix["retrievals"]["test"]["extraction_fallback"],
        mirix["retrievals"]["test"]["refused"],
        mirix["retrievals"]["test"]["refused_by_role"],
    ) == (2, 1, 1, 1, {"mirix": 1})
    assert mirix["retrievals"]["learning"]["refused"] == 0
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert "| `alfworld-luna-mirix` | 2 | failed 1, succeeded 1 | 2 | 2 (1) | 1 |" in md


def pool_batch(**kw: Any) -> dict[str, Any]:
    """A settled MIRIX curator batch that reports the experience pool: the
    keys ``report`` and ``methods.mirix.experience_pool`` read, zero unless given."""
    batch: dict[str, Any] = {
        "batch_id": kw.pop("batch_id", "r:b0"),
        "status": "settled",
        "history_pushed": 0,
        "history_skipped": "empty_pool",
        "history": {},
        "uncited_writes": 0,
        "triage_calls": 0,
        "dispositions": {},
        "history_cited": 0,
        "history_covered": 0,
        "history_not_worth": 0,
        "history_ignored": 0,
        "recurrences": 0,
        "retired": {},
        "pool_active_after": 0,
    }
    batch.update(kw)
    return batch


POOL_RUNS = [
    {
        "id": "r1",
        "created_at": "2026-01-01T00:00:01",
        "status": "succeeded",
        "curator_metrics": [
            pool_batch(
                batch_id="r1:b0",
                triage_calls=1,
                dispositions={"used": 1, "not_worth": 1, "deferred": 3, "defaulted": 2},
                pool_active_after=4,
            )
        ],
    },
    {
        "id": "r2",
        "created_at": "2026-01-01T00:00:02",
        "status": "succeeded",
        "curator_metrics": [
            pool_batch(
                batch_id="r2:b0",
                history_pushed=4,
                history_skipped=None,
                dispositions={"used": 1, "deferred": 2, "defaulted": 1},
                history_cited=1,
                history_covered=1,
                history_not_worth=1,
                history_ignored=1,
                recurrences=4,
                retired={"used": 1, "covered": 1},
                pool_active_after=4,
            )
        ],
    },
]


def _mirix_pool_cell(runs: Path, snapshot: Mapping[str, Any], **kw: Any) -> Path:
    return make_cell(
        runs,
        "alfworld",
        "mirix",
        test=[_row("a1", 1.0), _row("a2", 0.0)],
        snapshot={"method": "mirix", "skills": [{"id": "k1"}], **snapshot},
        **kw,
    )


def test_mirix_diagnostics_report_the_experience_pool(tmp_path):
    """The pool the MIRIX method froze (``experience_pool``), in the JSON and
    in its own Markdown table."""
    from evals.skill_bench.methods.mirix import experience_pool

    runs = tmp_path / "runs"
    frozen = experience_pool(POOL_RUNS)
    _mirix_pool_cell(runs, {"learning_runs": POOL_RUNS, "experience_pool": frozen})
    data = report.build_report(runs, frozen=FP)
    pool = data["cells"][0]["mirix"]["experience_pool"]
    assert pool == frozen
    assert (pool["active_at_freeze"], pool["entered"], pool["unaccounted"]) == (4, 6, 0)
    md = report.render_markdown(data)
    assert "#### MIRIX experience pool" in md
    assert (
        "| `alfworld-luna-mirix` | 4 (6) | used 2, covered 0, not worth 1, deferred 5 "
        "(3 by default), unaddressed 0 | 4: cited 1, covered 1, not worth 1, ignored 1 "
        "| 4 | used 1, covered 1, ignored 0, merged 0, unembeddable 0 | empty_pool 1 |"
    ) in md
    assert "Batches whose earlier lessons were lost" not in md
    assert "do not add up" not in md


def test_the_report_computes_the_pool_of_a_snapshot_without_it(tmp_path):
    """A snapshot frozen before the method exported the pool: the same summary, from its
    learning runs; a MIRIX before the pool (no batch reports it): "not reported"."""
    from evals.skill_bench.methods.mirix import experience_pool

    runs = tmp_path / "runs"
    _mirix_pool_cell(runs, {"learning_runs": POOL_RUNS})
    _mirix_pool_cell(
        runs,
        {"learning_runs": [{"id": "r0", "curator_metrics": [{"writes": 1}]}]},
        model_key="luna56",
    )
    data = report.build_report(runs, frozen=FP)
    pools = {c["cell_id"]: c["mirix"]["experience_pool"] for c in data["cells"]}
    assert pools == {
        "alfworld-luna-mirix": experience_pool(POOL_RUNS),
        "alfworld-luna56-mirix": None,
    }
    md = report.render_markdown(data)
    assert "| `alfworld-luna56-mirix` | not reported | - | - | - | - | - |" in md


def test_the_report_names_lost_earlier_lessons_and_pool_metrics_that_do_not_add_up(
    tmp_path,
):
    from evals.skill_bench.methods.mirix import experience_pool

    broken = [
        {
            **POOL_RUNS[0],
            "curator_metrics": [
                pool_batch(
                    batch_id="r1:b0",
                    history_skipped="embedding_failed",
                    dispositions={"deferred": 2, "defaulted": 2},
                    pool_active_after=1,  # 2 entered, none retired: 1 unaccounted
                ),
                pool_batch(
                    batch_id="r1:b1", history_skipped="disabled", pool_active_after=1
                ),
            ],
        }
    ]
    runs = tmp_path / "runs"
    _mirix_pool_cell(
        runs, {"learning_runs": broken, "experience_pool": experience_pool(broken)}
    )
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert (
        "**Batches whose earlier lessons were lost or switched off:** "
        "`alfworld-luna-mirix` disabled x1, `alfworld-luna-mirix` embedding_failed x1"
    ) in md
    assert (
        "**Pool metrics that do not add up** (entered - retired - active): "
        "`alfworld-luna-mirix` 1 (entered 2, retired 0, active 1)."
    ) in md


# --- MIRIX session context and skill feedback ------------------------------------------------

#: The agent context's length in :func:`context_runs` (what the attempt sent).
CONTEXT_CHARS = 5210


def _distilled(session_id: str, **kw: Any) -> dict[str, Any]:
    """A learned session's ``distill_metrics`` entry of a MIRIX with session context: the keys
    ``methods.mirix.learning_context`` reads, nothing served unless given."""
    entry: dict[str, Any] = {
        "session_id": session_id,
        "status": "ok",
        "messages": 6,
        "experiences": 1,
        "served_skills": 0,
        "served_skills_hidden": 0,
        "agent_context_chars": CONTEXT_CHARS,
        "context_tokens": 2100,
        "context_truncated_tokens": 0,
        "skill_feedback": None,
    }
    entry.update(kw)
    return entry


def _feedback(*rows: tuple[str, str, str, str], **counts: int) -> dict[str, Any]:
    """A session's ``skill_feedback`` with one stored row per ``(skill_id, name,
    followed, effect)``."""
    return {
        "served": counts.get("served", len(rows)),
        "not_shown": counts.get("not_shown", 0),
        "given": counts.get("given", len(rows)),
        "followed_unverified": counts.get("followed_unverified", 0),
        "effect_unverified": counts.get("effect_unverified", 0),
        "skills": [
            {
                "skill_id": skill_id,
                "name": name,
                "version_no": 1,
                "followed": followed,
                "effect": effect,
            }
            for skill_id, name, followed, effect in rows
        ],
    }


def _usage(shown=(), served=(), **kw: Any) -> dict[str, Any]:
    """A curator batch's ``skill_usage``."""
    return {
        "skill_ids": list(shown),
        "served_ids": list(served),
        "tokens": kw.get("tokens", 0),
        "compact": kw.get("compact", False),
        "left_out_ids": list(kw.get("left_out", ())),
        "skipped": kw.get("skipped", None if shown else "none"),
    }


def context_runs() -> list[dict[str, Any]]:
    """Two learning runs of a MIRIX with session context and skill usage, listed out of
    order. r1: s1 (no skill served yet), s2 failed (learned again in r2); one curator
    batch without skill usage (no usage section). r2: s2 (two skills served, its context
    cut by 120 tokens), s3 (one served, one stated value without a verifying quote); one
    batch showed two usage records (two citations without a verified quote), one had
    none to show."""
    return [
        {
            "id": "r2",
            "created_at": "2026-01-01T00:00:02",
            "status": "succeeded",
            "distill_metrics": [
                _distilled(
                    "s2",
                    served_skills=2,
                    context_truncated_tokens=120,
                    skill_feedback=_feedback(
                        ("proc-a", "Find then take", "yes", "helped"),
                        ("proc-b", "Check the price", "unclear", "unclear"),
                    ),
                ),
                _distilled(
                    "s3",
                    served_skills=1,
                    skill_feedback=_feedback(
                        ("proc-a", "Find then take", "no", "unclear"),
                        followed_unverified=0,
                        effect_unverified=1,
                    ),
                ),
            ],
            "curator_metrics": [
                pool_batch(
                    batch_id="r2:b0",
                    skill_usage=_usage(
                        ["proc-a", "proc-b"], ["proc-a"], tokens=310, compact=True
                    ),
                    citations_unverified=2,
                ),
                pool_batch(batch_id="r2:b1", skill_usage=_usage()),
            ],
        },
        {
            "id": "r1",
            "created_at": "2026-01-01T00:00:01",
            "status": "succeeded",
            "distill_metrics": [
                _distilled("s1"),
                {"session_id": "s2", "status": "failed", "agent_context_chars": 0},
            ],
            "curator_metrics": [pool_batch(batch_id="r1:b0")],
        },
    ]


#: :func:`methods.mirix.learning_context` of :func:`context_runs`.
CONTEXT_SUMMARY = {
    "sessions": 3,
    "with_agent_context": 3,
    "without_agent_context": [],
    "agent_context_chars": [CONTEXT_CHARS],
    "context_cut": 1,
    "context_truncated_tokens": 120,
    "served_sessions": 2,
    "served_skills": 3,
    "served_per_session": {"0": 1, "1": 1, "2": 1},
    "served_skills_hidden": 0,
    "feedback": {
        "served": 3,
        "not_shown": 0,
        "given": 3,
        "followed_unverified": 0,
        "effect_unverified": 1,
        "followed": {"no": 1, "unclear": 1, "yes": 1},
        "effect": {"helped": 1, "unclear": 2},
    },
    "skills_recorded": True,
    "skills": [
        {
            "skill_id": "proc-a",
            "name": "Find then take",
            "served": 2,
            "followed": {"no": 1, "yes": 1},
            "effect": {"helped": 1, "unclear": 1},
        },
        {
            "skill_id": "proc-b",
            "name": "Check the price",
            "served": 1,
            "followed": {"unclear": 1},
            "effect": {"unclear": 1},
        },
    ],
    "usage": {
        "batches": 2,
        "shown": 1,
        "records": 2,
        "served": 1,
        "compact": 1,
        "left_out": 0,
        "tokens": 310,
        "skipped": {"none": 1},
    },
    "citations_unverified": 2,
}


def test_mirix_diagnostics_report_what_the_distiller_was_shown_and_said(tmp_path):
    """The report's MIRIX diagnostics carry the session context (the agent context
    reached the distiller), the served skills' stored feedback, the usage records the
    curator saw and its citations without a verified quote, in the JSON and in their
    own Markdown table; a MIRIX without session context: "not reported"."""
    runs = tmp_path / "runs"
    _mirix_pool_cell(runs, {"learning_runs": context_runs()})
    _mirix_pool_cell(runs, {"learning_runs": POOL_RUNS}, model_key="luna56")
    data = report.build_report(runs, frozen=FP)
    contexts = {c["cell_id"]: c["mirix"]["learning_context"] for c in data["cells"]}
    assert contexts == {
        "alfworld-luna-mirix": CONTEXT_SUMMARY,
        "alfworld-luna56-mirix": None,
    }
    pools = {c["cell_id"]: c["mirix"]["experience_pool"] for c in data["cells"]}
    assert pools["alfworld-luna-mirix"]["citations_unverified"] == 2
    assert pools["alfworld-luna56-mirix"]["citations_unverified"] == 0
    md = report.render_markdown(data)
    assert "#### MIRIX session context and skill feedback" in md
    assert (
        f"| `alfworld-luna-mirix` | 3/3 ({CONTEXT_CHARS}) | 3 (2) | 3 (3) | no 1, "
        "unclear 1, yes 1 | helped 1, unclear 2 | 0, 1 | 2 (1/2) | 2 |"
    ) in md
    assert (
        "| `alfworld-luna56-mirix` | not reported | - | - | - | - | - | - | - |" in md
    )


# --- price tiers and latency ----------------------------------------------------------------


def test_the_list_price_of_a_call():
    """``config/models.json`` pricing: luna $0.10/$0.50 per M, cached input at $0.01,
    and $0.20/$0.75 once the prompt reaches 272K tokens (reasoning bills as output)."""
    from evals.skill_bench.cell_records import list_price_usd

    listed, over = list_price_usd(LUNA, 100_000, 0, 10_000)
    assert listed == pytest.approx(100_000 * 0.10e-6 + 10_000 * 0.50e-6) and not over
    listed, _ = list_price_usd(LUNA, 100_000, 40_000, 10_000)
    assert listed == pytest.approx(60_000 * 0.10e-6 + 40_000 * 0.01e-6 + 5e-3)
    listed, over = list_price_usd(LUNA, 300_000, 0, 10_000)
    assert listed == pytest.approx(300_000 * 0.20e-6 + 10_000 * 0.75e-6) and over
    listed, _ = list_price_usd(LUNA56, 100_000, 0, 0)
    assert listed == pytest.approx(100_000 * 0.20e-6)
    assert list_price_usd("nobody/else", 10, 0, 10) == (None, False)


def _priced_call(
    provider: str,
    ratio: float,
    *,
    prompt: int = 100_000,
    completion: int = 10_000,
    latency_s: float = 3.0,
) -> dict[str, Any]:
    """A luna call billed at ``ratio`` times its list price."""
    from evals.skill_bench.cell_records import list_price_usd

    listed, _ = list_price_usd(LUNA, prompt, 0, completion)
    assert listed is not None
    row = _call("actor", "test", listed * ratio, provider=provider, item="u1")
    row.update(
        cost=listed * ratio,
        input_tokens_actual=prompt,
        output_tokens=completion,
        cached_tokens=0,
        latency_s=latency_s + 0.5,
        attempts=[{"latency_s": latency_s, "outcome": "ok"}],
    )
    return row


def test_the_report_shows_the_tier_each_provider_billed(tmp_path):
    """OpenRouter names OpenAI's flex, standard and fast tiers alike
    ("OpenAI"); the billed cost against the list price tells them apart, and the report
    names a provider billed outside the band, with the latency of its answers and the
    calls that timed out on the model's own attempt."""
    runs = tmp_path / "runs"
    timeout = _call("actor", "test", 0.0, item="u2")
    timeout.update(
        status="deadline_exceeded",
        provider=None,
        model_served=None,
        error={"type": "deadline_exceeded", "blame": "model"},
    )
    make_cell(
        runs,
        "webshop",
        "none",
        test=shop("o", [1.0]),
        calls=[
            _priced_call("OpenAI", 1.0, latency_s=3.0),
            _priced_call("OpenAI", 0.5, latency_s=250.0),  # flex, and slow
            _priced_call("Azure", 1.1, latency_s=4.0),
            _priced_call("OpenAI", 1.0, prompt=300_000, latency_s=8.0),  # long input
            timeout,
        ],
    )
    make_cell(
        runs,
        "webshop",
        "awm",
        test=shop("o", [1.0]),
        calls=[_priced_call("OpenAI", 0.5, latency_s=40.0)],
    )
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["webshop-luna-none"]["providers"]
    openai = cell["pricing"][LUNA]["OpenAI"]
    assert (openai["calls"], openai["priced_calls"], openai["over_tier_calls"]) == (
        3,
        3,
        1,
    )
    assert (openai["ratio_min"], openai["ratio_max"]) == (0.5, 1.0)
    assert openai["latency_s_max"] == 250.0
    assert cell["timeouts_model_blamed"] == {LUNA: 1}
    pricing = data["pricing"]
    merged = pricing["by_model"][LUNA]["OpenAI"]
    assert merged["calls"] == 4 and merged["ratio_min"] == 0.5
    assert merged["cost_usd"] / merged["list_cost_usd"] == pytest.approx(
        merged["ratio"], abs=1e-4
    )
    assert 0.5 < merged["ratio"] < 0.9  # half its list cost was billed at flex
    assert merged["latency_p50_le_s"] == 10.0 and merged["latency_p90_le_s"] == 300.0
    assert pricing["by_model"][LUNA]["Azure"]["ratio"] == pytest.approx(1.1)
    assert pricing["outside_band"] == [f"{LUNA} / OpenAI"]
    assert pricing["timeouts_model_blamed"] == {LUNA: 1}
    md = report.render_markdown(data)
    assert report.CAVEAT_PRICE_TIERS in md
    assert f"**Billed outside 0.9-1.15 times the list price:** {LUNA} / OpenAI" in md
    assert f"deadline on the model's own attempt: {LUNA} 1" in md


# --- unknown costs, unattributed spend -------------------------------------------------------


def _deadline_call(
    role: str, phase: str, *, item: str | None = None, legacy: bool = False
) -> dict[str, Any]:
    """A call the deadline cut off: OpenRouter billed it, the ledger never learned how
    much. ``legacy``: a line written before the gateway counted such attempts (only the
    attempt's ``outcome`` says so)."""
    row = _call(role, phase, 0.0, item=item)
    row.update(
        status="deadline_exceeded",
        http_status=504,
        provider=None,
        model_served=None,
        cost_total=None,
        billed_attempts=0,
        attempts=[{"n": 1, "outcome": "deadline", "latency_s": 600.0}],
        error={"type": "deadline_exceeded", "blame": "model"},
    )
    if not legacy:
        row["attempts"][0]["billed"] = "unknown"
        row["unbilled_deadline_attempts"] = 1
    return row


def test_deadline_calls_are_listed_as_cost_unknown(tmp_path):
    """A deadline call was billed at a cost no ledger holds. The totals cannot
    count it, so the cell names it per side and role, beside its test items, in the
    'not counted' column, and older ledger lines are recognised by their attempts."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=[_row("a1", 1.0), _row("a2", 0.0)],
        calls=[
            _call("actor", "learning", 0.01, item="l1"),
            _deadline_call("learner", "learning", legacy=True),
            _call("actor", "test", 0.02, item="u1"),
            _deadline_call("actor", "test", item="u1"),
            _deadline_call("actor", "test", item="u2"),
        ],
    )
    data = report.build_report(runs, frozen=FP)
    costs = by_id(data)["alfworld-luna-reasoningbank"]["costs"]
    assert costs["unknown_cost_calls"] == 3
    assert costs["learning"]["unknown_cost_calls"] == 1
    assert costs["learning"]["by_role"]["learner"] == {
        "cost": 0.0,
        "calls": 1,
        "unknown_cost_calls": 1,
    }
    assert costs["test"]["by_role"]["actor"]["unknown_cost_calls"] == 2
    assert costs["test_items_unknown_cost"] == 2
    assert costs["total_cost"] == pytest.approx(0.03)  # the known part only
    assert data["costs"]["unknown_cost_calls"] == 3
    assert report.uncounted_costs(costs) == "3 cut-off call(s), cost unknown"
    md = report.render_markdown(data)
    assert "| 3 cut-off call(s), cost unknown |" in md
    assert (
        "**Cost not in the totals:** `alfworld-luna-reasoningbank` 3 cut-off call(s), "
        "cost unknown. " + report.CAVEAT_UNKNOWN_COST
    ) in md


def _timed(row: dict[str, Any], ts: str, **kw: Any) -> dict[str, Any]:
    return {**row, "ts": ts, **kw}


def _write_balance(gateway_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = gateway_dir / "_gateway" / "balance.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_unattributed_spend_is_the_key_usage_the_ledgers_miss(tmp_path):
    """The key's usage growth over the budget's window against ``cost_total`` of every
    ledger in the gateway runs directory, whichever cell or split mode it served, over
    the calls that ended in that window (``completed_at``, else ``ts`` + latency)."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[
            _timed(
                _call("actor", "test", 0.10, item="u1"),
                "2026-09-29T06:00:00+00:00",
                completed_at="2026-09-29T06:00:05+00:00",
            ),
            _timed(  # older line: ends at ts + latency
                _deadline_call("actor", "test", item="u1", legacy=True),
                "2026-09-29T06:10:00+00:00",
                latency_s=600.0,
            ),
            _timed(  # ends after the last balance poll: outside the window
                _call("actor", "test", 0.05, item="u1"),
                "2026-09-29T07:59:00+00:00",
                completed_at="2026-09-29T08:01:00+00:00",
            ),
        ],
    )
    # A smoke cell's ledger in the same gateway runs directory: not reported (split
    # mode), but the same key paid for it.
    make_cell(
        runs,
        "webshop",
        "none",
        test=shop("s", [1.0]),
        split_mode="smoke",
        calls=[
            _timed(
                _call("actor", "test", 0.20, item="s1"),
                "2026-09-29T06:30:00+00:00",
                completed_at="2026-09-29T06:30:02+00:00",
            )
        ],
    )
    start = "2026-09-29T05:00:00+00:00"
    budget = {"budget_usd": 12.0, "started_at": start, "start_usage_usd": 100.0}
    _write_balance(
        runs,
        [
            {"ts": start, "usage": 100.0, "budget": {**budget, "spent_usd": 0.0}},
            {
                "ts": "2026-09-29T08:00:00+00:00",
                "usage": 100.5,
                "budget": {**budget, "spent_usd": 0.5},
            },
        ],
    )
    data = report.build_report(runs, frozen=FP)
    (spend,) = data["spend"]
    assert spend["gateway_dir"] == str(runs)
    assert spend["source"] == "budget.spent_usd"
    assert spend["key_usage_delta_usd"] == 0.5
    assert spend["ledger_cost_usd"] == pytest.approx(0.30)
    assert spend["ledger_calls"] == 3 and spend["unknown_cost_calls"] == 1
    assert spend["unattributed_usd"] == pytest.approx(0.20)
    assert spend["calls_outside_window"] == 1
    assert spend["cost_outside_window_usd"] == pytest.approx(0.05)
    md = report.render_markdown(data)
    assert "**unattributed spend $0.2000** (1 cut-off call(s) of unknown cost" in md
    assert report.CAVEAT_UNATTRIBUTED in md
    # Without a spend budget: the usage growth since the first balance poll.
    _write_balance(
        runs,
        [
            {"ts": "2026-09-29T05:30:00+00:00", "usage": 100.0},
            {"ts": "2026-09-29T08:00:00+00:00", "usage": 100.45},
        ],
    )
    (spend,) = report.build_report(runs, frozen=FP)["spend"]
    assert spend["source"] == "usage" and spend["key_usage_delta_usd"] == 0.45
    assert spend["unattributed_usd"] == pytest.approx(0.15)
    (runs / "_gateway" / "balance.jsonl").unlink()
    assert report.build_report(runs, frozen=FP)["spend"] == []


def test_one_gateway_directory_is_one_spend_however_it_is_spelled(
    tmp_path, monkeypatch
):
    """The report's ``--runs-dir`` given relative while the cells record their
    ledgers by absolute path (as run.py does), or the other way round, names one
    directory: its spend is listed once, never summed twice."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[
            _timed(
                _call("actor", "test", 0.10, item="u1"),
                "2026-09-29T06:00:00+00:00",
                completed_at="2026-09-29T06:00:05+00:00",
            )
        ],
    )
    _write_balance(
        runs,
        [
            {"ts": "2026-09-29T05:30:00+00:00", "usage": 100.0},
            {"ts": "2026-09-29T08:00:00+00:00", "usage": 100.45},
        ],
    )
    monkeypatch.chdir(tmp_path)
    for spelling in (Path("runs"), Path("./runs/../runs"), runs):
        (spend,) = report.build_report(spelling, frozen=FP)["spend"]
        assert spend["gateway_dir"] == str(runs)
        assert spend["ledger_calls"] == 1
    # The ledger paths recorded relative (an older run.py), the report absolute.
    cell_dir = runs / "alfworld-luna-none"
    rewritten = 0
    for attempt_json in cell_dir.glob("attempt-*/attempt.json"):
        record = json.loads(attempt_json.read_text())
        record["ledger"] = os.path.relpath(record["ledger"], tmp_path)
        attempt_json.write_text(json.dumps(record))
        rewritten += 1
    assert rewritten == 1
    data = report.build_report(runs, frozen=FP)
    (spend,) = data["spend"]
    assert spend["gateway_dir"] == str(runs) and spend["ledger_calls"] == 1
    assert report.render_markdown(data).count("unattributed spend") == 1


# --- price tiers: cache writes, reasoning, slow calls ------------------------------------------


def _luna_cache_write_row(
    prompt: int, cached: int, written: int, out: int, cost: float
):
    """A ledger row of an OpenAI call of gpt-6-luna, billed at list price once cache
    writes are priced at ``input_cache_write``."""
    row = _call("actor", "test", cost, provider="OpenAI", item="u1")
    row.update(
        cost=cost,
        input_tokens_actual=prompt,
        cached_tokens=cached,
        output_tokens=out,
        usage={
            "prompt_tokens": prompt,
            "completion_tokens": out,
            "cost": cost,
            "prompt_tokens_details": {
                "cached_tokens": cached,
                "cache_write_tokens": written,
            },
        },
        attempts=[{"latency_s": 2.4, "outcome": "ok"}],
    )
    return row


def test_cache_writes_are_priced_so_a_standard_tier_reads_one(tmp_path):
    """Cache writes (1.25x the prompt price) are priced, so luna served at OpenAI's
    standard tier reads a ratio of 1.000 against the list price."""
    from evals.skill_bench.cell_records import list_price_usd

    listed, _ = list_price_usd(LUNA, 1010, 0, 100, 1000)
    assert listed == pytest.approx(0.000176, rel=1e-9)
    listed, _ = list_price_usd(LUNA, 4010, 1000, 500, 3000)
    assert listed == pytest.approx(0.000636, rel=1e-9)
    # The embedding model lists no cache-write price: written tokens count at the
    # prompt price.
    embedding = protocol.EMBEDDING_MODEL
    assert list_price_usd(embedding, 1000, 0, 0, 400) == list_price_usd(
        embedding, 1000, 0, 0
    )
    # Never more written than the uncached prompt.
    assert list_price_usd(LUNA, 100, 60, 0, 90)[0] == pytest.approx(
        60 * 0.01e-6 + 40 * 0.125e-6
    )
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "webshop",
        "none",
        test=shop("o", [1.0]),
        calls=[
            _luna_cache_write_row(1010, 0, 1000, 100, 0.000176),
            _luna_cache_write_row(4010, 1000, 3000, 500, 0.000636),
        ],
    )
    entry = report.build_report(runs, frozen=FP)["pricing"]["by_model"][LUNA]["OpenAI"]
    assert entry["ratio"] == 1.0
    assert (entry["ratio_min"], entry["ratio_max"]) == (1.0, 1.0)


def _luna56_call(provider: str, reasoning: int | None, latency_s: float = 3.0):
    row = _call("actor", "test", 0.001, provider=provider, model=LUNA56)
    row.update(
        reasoning_tokens=reasoning,
        attempts=[{"latency_s": latency_s, "outcome": "ok"}],
    )
    return row


def test_reasoning_and_slow_calls_per_provider(tmp_path):
    """A model's reasoning can depend on the provider OpenRouter picks. The price
    tier table shows each provider's reasoning tokens (mean, p95) and its ok calls past
    half the deadline, and names a provider reasoning 10x the model's median one."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "webshop",
        "none",
        model_key="luna56",
        test=shop("o", [1.0]),
        calls=[
            *[_luna56_call("Modal", 50) for _ in range(4)],
            *[_luna56_call("Together", 40) for _ in range(3)],
            _luna56_call("Together", None),  # no reasoning count: not in the mean
            *[_luna56_call("Parasail", 60) for _ in range(2)],
            _luna56_call("Wafer", 5_000, latency_s=150.0),
            _luna56_call("Wafer", 25_000, latency_s=400.0),  # past 300 s
            _luna56_call("Tiny", 0),
        ],
    )
    pricing = report.build_report(runs, frozen=FP)["pricing"]
    assert pricing["slow_call_s"] == protocol.CALL_DEADLINE_S / 2
    by_provider = pricing["by_model"][LUNA56]
    wafer = by_provider["Wafer"]
    assert wafer["reasoning_calls"] == 2 and wafer["reasoning_mean"] == 15_000.0
    assert wafer["reasoning_tokens_max"] == 25_000
    assert wafer["reasoning_p95_le"] == 50_000.0  # the bucket's upper edge
    assert wafer["slow_calls"] == 1
    together = by_provider["Together"]
    assert together["calls"] == 4 and together["reasoning_calls"] == 3
    assert together["reasoning_mean"] == 40.0 and together["reasoning_p95_le"] == 50.0
    assert by_provider["Modal"]["slow_calls"] == 0
    assert by_provider["Tiny"]["reasoning_p95_le"] == 0.0
    # The median of the other providers' means (0, 40, 50, 60) is 45: Wafer is 333x it.
    assert pricing["reasoning_outliers"] == [
        {
            "model": LUNA56,
            "provider": "Wafer",
            "reasoning_mean": 15_000.0,
            "median_provider_mean": 45.0,
        }
    ]
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert "| ok calls answered after > 300 s | reasoning tokens, mean |" in md
    assert f"| {LUNA56} | Wafer | 2 |" in md and "| 1 | 15000.0 | <= 50000 |" in md
    assert (
        "**Providers reasoning more than 10 times the model's median other provider:** "
        f"{LUNA56} / Wafer (mean 15000 against 45)"
    ) in md


def test_a_small_mean_over_a_zero_median_is_not_an_outlier(tmp_path):
    """With most providers at 0 reasoning tokens, a provider averaging a few tokens is
    noise, not a routing effect: a provider is named only from 100 tokens on. With two
    providers, the outlier is compared with the other one, not with their average."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "webshop",
        "none",
        model_key="luna56",
        test=shop("o", [1.0]),
        calls=[
            _luna56_call("A", 0),
            _luna56_call("B", 0),
            _luna56_call("C", 30),
            _luna56_call("D", 400),
        ],
    )
    outliers = report.build_report(runs, frozen=FP)["pricing"]["reasoning_outliers"]
    assert [o["provider"] for o in outliers] == ["D"]
    two = tmp_path / "two"
    make_cell(
        two,
        "webshop",
        "none",
        model_key="luna56",
        test=shop("o", [1.0]),
        calls=[_luna56_call("A", 30), _luna56_call("B", 5_000)],
    )
    outliers = report.build_report(two, frozen=FP)["pricing"]["reasoning_outliers"]
    assert [(o["provider"], o["median_provider_mean"]) for o in outliers] == [
        ("B", 30.0)
    ]


# --- rendering ---------------------------------------------------------------------------------


def _section(md: str, title: str) -> list[str]:
    start = md.index(f"## {title}")
    end = md.find("\n## ", start + 1)
    return md[start : end if end != -1 else None].splitlines()


def test_injected_tokens_rows_render_from_the_report_and_from_its_json(runs):
    """Every complete cell's per-phase injection row appears under its header, both
    when rendered from the report and from report.json, whose keys are sorted: the
    Markdown never depends on the order of the JSON's dicts."""
    data = report.build_report(runs, frozen=FP)
    md = report.render_markdown(data)
    injected = _section(md, "Injected tokens")
    rows = [line for line in injected if line.startswith("| `")]
    assert rows == [
        "| `alfworld-luna-none` | test | 20 | 20 | 0.0 | 0.0 | 0.0 | 0 | 0.0 |",
        "| `alfworld-luna-reasoningbank` | learning | 3 | 0 | 5.0 | 5.0 | 5.0 | 5 | 5.0 |",
        "| `alfworld-luna-reasoningbank` | test | 5 | 1 | 20.0 | 20.0 | 36.0 | 40 | 60.0 |",
    ]
    reloaded = json.loads(json.dumps(data, sort_keys=True))
    assert report.render_markdown(reloaded) == md
    main = _section(md, "Main results (test items)")
    # WebShop's three metrics, each under its label.
    assert (
        "| group | method | items | success rate (score = 1) | mean score | "
        "success within 15 steps |" in main
    )
    assert "| webshop-luna56 | awm | 2 | 0.500 | 0.500 | 0.500 |" in main
    assert "| alfworld-luna | none | 20 | 0.300 |" in main


def test_the_header_and_the_matrix_total_name_cells_that_are_not_complete(runs):
    """A cell that is not complete (running, stopped) is in no comparison, audit or not;
    its ledger so far is in the matrix total, and the report says both."""
    data = report.build_report(runs, frozen=FP)
    assert data["costs"]["incomplete"] == {"webshop-luna56-none": 0.0}
    md = report.render_markdown(data)
    header = next(
        line for line in md.splitlines() if "Cells without a passing audit" in line
    )
    assert "`webshop-luna56-none` (not complete: in no comparison)" in header
    compared, _, not_compared = header.partition(" (still compared;")
    assert "`alfworld-luna-none`" in compared and "webshop-luna56-none" not in compared
    assert "webshop-luna56-none" in not_compared
    assert (
        "Matrix total $0.2110 (every cell listed, including the ledgers so far of 1 "
        "cell(s) not complete: `webshop-luna56-none` $0.0000)"
    ) in md


# --- resilience: resumes, hosts, the provider policy, two hosts ---------------------------


def _state_of(cell: Path, number: int = 1) -> dict[str, Any]:
    return json.loads((cell / f"attempt-{number:02d}" / "attempt.json").read_text())


def _set_state(cell: Path, number: int = 1, **fields: Any) -> None:
    state = _state_of(cell, number)
    state.update(fields)
    write_json_atomic(cell / f"attempt-{number:02d}" / "attempt.json", state)


def test_a_resumed_learning_phase_shows_its_discarded_cost(tmp_path):
    """The calls of a rolled-back try are part of the learning side, shown apart
    (``learning_discarded``), and each resume is listed: the checkpoint it restored, what
    was in flight, and what its calls cost as the ledger's final lines say (a call in
    flight at the crash gets its final line after the row)."""
    runs = tmp_path / "runs"
    judge = _call("judge", "learning", 0.03, item="u1")
    learner = _call("learner", "learning", 0.05, item="u1")
    late = _call("learner", "learning", 0.07, item="u1")
    redo = _call("judge", "learning", 0.03, item="u1")
    cell = make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=[_row("a1", 1.0)],
        learning=[_row("l1", 1.0, position=0)],
        calls=[
            _call("actor", "learning", 0.10, item="u1"),
            judge,
            learner,
            late,
            redo,
            _call("actor", "test", 0.20, item="t1"),
        ],
    )
    segment = {
        "segment": 1,
        "attempt": 1,
        "checkpoint": {"id": "c0000-played", "offset": 10},
        "reason": "gateway_unreachable",
        "failure": {"code": "gateway_unreachable", "kind": "infra"},
        "in_flight": {"position": 0, "stage": "after_task"},
        "moved": {"learning_positions": []},
        "calls": [
            {"call_id": judge["call_id"], "role": "judge", "cost_total": 0.03},
            {"call_id": learner["call_id"], "role": "learner", "cost_total": 0.05},
            # In flight at the crash: the row saw no final line (cost 0 then).
            {
                "call_id": late["call_id"],
                "role": "learner",
                "final_line": False,
                "cost_total": 0.0,
            },
        ],
    }
    (cell / "attempt-01" / "discarded.jsonl").write_text(json.dumps(segment) + "\n")
    data = report.build_report(runs, frozen=FP)
    (section,) = data["cells"]
    costs = section["costs"]
    assert costs["learning"]["cost"] == pytest.approx(0.28)
    assert costs["learning_discarded"]["cost"] == pytest.approx(0.15)
    assert costs["learning_discarded"]["calls"] == 3
    assert costs["learning_discarded"]["by_role"] == {"judge": 1, "learner": 2}
    assert costs["total_cost"] == pytest.approx(0.48)  # counted once, in learning
    assert data["costs"]["learning_discarded"] == pytest.approx(0.15)
    (resume,) = section["resumes"]
    assert resume["checkpoint"] == "c0000-played"
    assert resume["failure_code"] == "gateway_unreachable"
    assert (resume["in_flight_position"], resume["in_flight_stage"]) == (
        0,
        "after_task",
    )
    assert resume["cost"] == pytest.approx(0.15) and resume["calls"] == 3
    md = report.render_markdown(data)
    lines = _section(md, "Resumed learning")
    assert any("`c0000-played`" in line and "$0.1500" in line for line in lines)
    assert "0 (after_task)" in "\n".join(lines)
    cost_lines = _section(md, "Cost")
    assert any("$0.1500 (3 calls)" in line for line in cost_lines)
    # A cell without resumes has no such section.
    (cell / "attempt-01" / "discarded.jsonl").unlink()
    data = report.build_report(runs, frozen=FP)
    assert data["cells"][0]["resumes"] == []
    assert data["cells"][0]["costs"]["learning_discarded"]["calls"] == 0
    assert "## Resumed learning" not in report.render_markdown(data)


def test_every_cell_names_its_host(tmp_path):
    runs = tmp_path / "runs"
    cell = make_cell(runs, "webshop", "none", test=shop("w", [1.0]))
    host = {
        "hostname": "mac-1",
        "system": "Darwin",
        "machine": "arm64",
        "python": "3.11.13",
    }
    _set_state(cell, host=host)
    data = report.build_report(runs, frozen=FP)
    # A host recorded before its machine id was has none.
    assert data["cells"][0]["host"] == {
        "hostname": "mac-1",
        "machine_id": None,
        "system": "Darwin",
        "machine": "arm64",
    }
    cells = _section(report.render_markdown(data), "Cells")
    assert any("mac-1 (Darwin arm64)" in line for line in cells)
    _set_state(cell, host={**host, "machine_id": "0123456789abcdef" * 2})
    data = report.build_report(runs, frozen=FP)
    assert data["cells"][0]["host"]["machine_id"] == "0123456789abcdef" * 2
    cells = _section(report.render_markdown(data), "Cells")
    assert any("mac-1 (Darwin arm64, id 01234567)" in line for line in cells)
    _set_state(cell, host=None)
    assert report.build_report(runs, frozen=FP)["cells"][0]["host"] is None


#: A provider policy for luna56, for these tests only (``protocol.PROVIDER_POLICY``
#: names no model): the routing its calls carry.
LUNA56_POLICY = {"ignore": ["open-inference", "wafer"]}


@pytest.fixture
def luna56_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The protocol's provider policy with :data:`LUNA56_POLICY` for luna56."""
    monkeypatch.setattr(protocol, "PROVIDER_POLICY", {LUNA56: LUNA56_POLICY})


def test_the_header_names_the_provider_policy(runs, monkeypatch):
    data = report.build_report(runs, frozen=FP)
    assert data["protocol"]["provider_policy"] == {}
    assert (
        "- Provider policy: openai/gpt-6-luna unrestricted; "
        "openai/gpt-5.6-luna unrestricted"
    ) in report.render_markdown(data)
    monkeypatch.setattr(protocol, "PROVIDER_POLICY", {LUNA56: LUNA56_POLICY})
    data = report.build_report(runs, frozen=FP)
    assert data["protocol"]["provider_policy"] == {LUNA56: LUNA56_POLICY}
    md = report.render_markdown(data)
    assert (
        "- Provider policy: openai/gpt-5.6-luna ignores open-inference, "
        "wafer; openai/gpt-6-luna unrestricted"
    ) in md


def _sent(row: dict[str, Any], provider: Any = None, **kw: Any) -> dict[str, Any]:
    """A call line as the gateway writes it once the request went out: its hash and the
    ``wire`` fields, ``provider`` only when the call carried a routing."""
    wire = {"model": row["model"], "stream": False}
    if provider is not None:
        wire["provider"] = provider
    return {**row, "request_sha256": "a" * 64, "wire": wire, **kw}


def test_the_header_states_the_routing_the_ledgers_show(tmp_path, luna56_policy):
    """The header states the routing the reported attempts' calls carried,
    per chat model, next to the protocol's policy; they agree here, so nothing is
    flagged. An invalidated attempt's calls are not the report's."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "none",
        model_key="luna56",
        test=[_row("a1", 1.0)],
        calls=[
            _sent(_call("actor", "test", 0.01, item="t1", model=LUNA56), LUNA56_POLICY),
            _sent(_call("actor", "test", 0.01, item="t2", model=LUNA56), LUNA56_POLICY),
            _sent(_call("embedding", "setup", 0.0)),
        ],
        invalid_calls=[_sent(_call("actor", "test", 0.01, item="t1", model=LUNA56))],
    )
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[_sent(_call("actor", "test", 0.01, item="t1"))],
    )
    data = report.build_report(runs, frozen=FP)
    assert data["protocol"]["provider_routing_observed"] == {
        LUNA: {"routings": {"none": 1}, "unrecorded": 0},
        LUNA56: {
            "routings": {json.dumps(LUNA56_POLICY, sort_keys=True): 2},
            "unrecorded": 0,
        },
    }
    md = report.render_markdown(data)
    assert (
        "- Provider routing the ledgers show: openai/gpt-6-luna unrestricted (1 calls); "
        "openai/gpt-5.6-luna ignores open-inference, wafer (2 calls)\n"
    ) in md
    assert "not the protocol's policy" not in md


def test_a_mixed_or_unrecorded_routing_is_flagged(tmp_path, luna56_policy):
    """Calls that went out without the policy (data from before it) make the
    model's routing ``mixed``; a line without ``wire`` is ``not recorded``; either way the
    header says the data is not the protocol's policy throughout."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "none",
        model_key="luna56",
        test=[_row("a1", 1.0)],
        calls=[
            _sent(_call("actor", "test", 0.01, item="t1", model=LUNA56), LUNA56_POLICY),
            _sent(_call("actor", "test", 0.01, item="t2", model=LUNA56)),
        ],
    )
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[{**_call("actor", "test", 0.01, item="t1"), "request_sha256": "b" * 64}],
    )
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    line = next(x for x in md.splitlines() if "Provider routing the ledgers show" in x)
    assert line == (
        "- Provider routing the ledgers show: openai/gpt-6-luna not recorded (1 calls); "
        "openai/gpt-5.6-luna mixed: unrestricted (1 calls), ignores open-inference, "
        "wafer (1 calls) (**not the protocol's policy throughout**)"
    )


def test_calls_without_the_policy_are_flagged_even_when_uniform(
    tmp_path, luna56_policy
):
    """Every luna56 call without the routing (a run before the policy)
    is one routing, not mixed, and still not the protocol's."""
    runs = tmp_path / "runs"
    make_cell(
        runs,
        "alfworld",
        "none",
        model_key="luna56",
        test=[_row("a1", 1.0)],
        calls=[_sent(_call("actor", "test", 0.01, item="t1", model=LUNA56))],
    )
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert (
        "- Provider routing the ledgers show: openai/gpt-6-luna: no calls sent; "
        "openai/gpt-5.6-luna unrestricted (1 calls) (**not the protocol's policy "
        "throughout**)"
    ) in md


def test_an_older_report_without_the_routing_renders(runs):
    data = report.build_report(runs, frozen=FP)
    data["protocol"].pop("provider_routing_observed")
    md = report.render_markdown(data)
    assert "Provider routing the ledgers show" not in md
    assert "- Provider policy:" in md


def test_a_copied_runs_directory_reads_its_own_ledgers(tmp_path):
    """The attempts' recorded absolute ledger paths name another host's tree: the report
    reads the ledgers relative to the runs directory it is given."""
    runs = tmp_path / "runs"
    cell = make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[_call("actor", "test", 0.10, item="t1")],
    )
    state = _state_of(cell)
    stale = tmp_path / "other-host" / state["run_id"] / "gateway.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(_call("actor", "test", 9.0, item="t1")) + "\n")
    _set_state(cell, ledger=str(stale))
    data = report.build_report(runs, frozen=FP)
    assert data["cells"][0]["costs"]["total_cost"] == pytest.approx(0.10)
    # Only without a relative ledger is the recorded one read (its directory is the run's).
    (runs / state["run_id"] / "gateway.jsonl").unlink()
    data = report.build_report(runs, frozen=FP)
    assert data["cells"][0]["costs"]["total_cost"] == pytest.approx(9.0)


def _host_runs(
    runs: Path, host: str, benchmark: str, cost: float, ts: str, end: str, budget: dict
) -> None:
    """One host's runs directory mirrored under ``runs/_hosts/<host>`` with one cell,
    linked into the merged view like ``tools/sync_runs.py`` does."""
    mirror = runs / "_hosts" / host
    make_cell(
        mirror,
        benchmark,
        "none",
        test=[_row(f"{host}1", 1.0, {"success": 1.0, WITHIN: 1.0})],
        calls=[
            _timed(
                _call("actor", "test", cost, item="u1"),
                ts,
                completed_at=ts,
            )
        ],
    )
    for child in mirror.iterdir():
        if not child.name.startswith("_"):
            (runs / child.name).symlink_to(os.path.relpath(child, runs))
    spent = budget.pop("spent_usd")
    usage = budget.pop("usage")
    _write_balance(
        mirror,
        [
            {
                "ts": budget["started_at"],
                "usage": budget["start_usage_usd"],
                "budget": {**budget, "spent_usd": 0.0},
            },
            {"ts": end, "usage": usage, "budget": {**budget, "spent_usd": spent}},
        ],
    )


def test_two_hosts_on_one_key_report_the_spend_over_their_union(tmp_path):
    """Each host's gateway counts the key's whole usage, so each host's unattributed
    spend includes the other host's calls. With one shared budget start the report also
    computes it once over both hosts' ledgers (the latest usage less the start)."""
    runs = tmp_path / "runs"
    start = "2026-09-29T05:00:00+00:00"
    shared = {"budget_usd": 20.0, "started_at": start, "start_usage_usd": 100.0}
    _host_runs(
        runs,
        "cloud",
        "alfworld",
        0.30,
        "2026-09-29T06:00:00+00:00",
        "2026-09-29T07:00:00+00:00",
        {**shared, "spent_usd": 0.40, "usage": 100.40},
    )
    _host_runs(
        runs,
        "local",
        "webshop",
        0.20,
        "2026-09-29T06:30:00+00:00",
        "2026-09-29T08:00:00+00:00",
        {**shared, "spent_usd": 0.60, "usage": 100.60},
    )
    data = report.build_report(runs, frozen=FP)
    assert {c["cell_id"] for c in data["cells"]} == {
        "alfworld-luna-none",
        "webshop-luna-none",
    }
    spends = {Path(s["gateway_dir"]).name: s for s in data["spend"]}
    assert set(spends) == {"cloud", "local"}  # the merged view itself has no balance
    assert spends["cloud"]["ledger_cost_usd"] == pytest.approx(0.30)
    assert spends["local"]["unattributed_usd"] == pytest.approx(0.40)
    union = data["spend_union"]
    assert (
        union["shared"] is True and union["window_end"] == "2026-09-29T08:00:00+00:00"
    )
    assert union["key_usage_delta_usd"] == pytest.approx(0.60)
    assert union["ledger_cost_usd"] == pytest.approx(0.50)
    assert union["unattributed_usd"] == pytest.approx(0.10)
    md = report.render_markdown(data)
    assert "**Over all hosts**" in md and "**unattributed spend $0.1000**" in md
    # Hosts that did not share a start: no union, and the report says why.
    balance = runs / "_hosts" / "local" / "_gateway" / "balance.jsonl"
    rows = [json.loads(line) for line in balance.read_text().splitlines()]
    for row in rows:
        row["budget"]["start_usage_usd"] = 100.25
    balance.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = report.build_report(runs, frozen=FP)
    assert data["spend_union"]["shared"] is False
    assert "not computed" in report.render_markdown(data)


def test_one_host_has_no_union(tmp_path):
    runs = tmp_path / "runs"
    make_cell(runs, "alfworld", "none", test=[_row("a1", 1.0)])
    _write_balance(
        runs,
        [
            {"ts": "2026-09-29T05:30:00+00:00", "usage": 100.0},
            {"ts": "2026-09-29T08:00:00+00:00", "usage": 100.45},
        ],
    )
    data = report.build_report(runs, frozen=FP)
    assert data["spend_union"] is None
    assert "Over all hosts" not in report.render_markdown(data)


def test_the_report_says_when_a_merged_view_is_not_final(tmp_path):
    """Over a merged view the header shows what its _merged.json says
    (when the view was built and each host synced) and why it is not final (a
    --mirror-only sync since it was built, each conflict); no number changes."""
    runs = tmp_path / "runs"
    make_cell(runs, "alfworld", "none", test=[_row("a1", 1.0)])
    plain = report.build_report(runs, frozen=FP)
    assert plain["merged_view"] is None
    assert "Merged view" not in report.render_markdown(plain)
    write_json_atomic(
        runs / "_merged.json",
        {
            "schema": 1,
            "sources": {
                "cloud": {"synced_at": "2026-09-29T12:00:00+00:00", "rsync_exit": 0},
                "local": {"synced_at": "2026-09-29T10:00:00+00:00", "rsync_exit": 0},
            },
            "view_built_at": "2026-09-29T10:00:05+00:00",
            "last_sync": {"at": "2026-09-29T12:00:00+00:00", "mode": "mirror-only"},
            "conflicts": [
                {
                    "entry": "webshop-luna-none",
                    "hosts": ["cloud", "local"],
                    "reason": "different content",
                }
            ],
        },
    )
    data = report.build_report(runs, frozen=FP)
    merged = data["merged_view"]
    assert merged["final"] is False and merged["view_built_at"] == (
        "2026-09-29T10:00:05+00:00"
    )
    assert [w["check"] for w in merged["warnings"]] == [
        "merged_view_not_final",
        "merged_view_conflict",
    ]
    md = report.render_markdown(data)
    assert "- Merged view (`_merged.json`): 2 host(s) (cloud synced" in md
    assert "- **Merged view not final:** cloud synced after the view was built" in md
    assert "a --mirror-only sync" in md
    assert (
        "- **Merged view conflict:** conflict: webshop-luna-none from cloud, local "
        "(different content)" in md
    )
    for key in ("cells", "comparisons", "costs", "caveats", "errors"):
        assert data[key] == plain[key], key
    # An unreadable record is said so, too.
    (runs / "_merged.json").write_text("{")
    md = report.render_markdown(report.build_report(runs, frozen=FP))
    assert "- **Merged view unreadable:**" in md


def _skips(data: Mapping[str, Any]) -> dict[tuple[str, str], str | None]:
    return {(c["group"], c["method"]): c.get("skipped") for c in data["comparisons"]}


# --- removed methods (study.REMOVED_METHODS) -----------------------------------------------

#: The method conftest's ``removed_method`` fixture records as removed from the study;
#: its cells stay on disk.
DC = REMOVED_METHOD
DC_WEB = f"webshop-luna-{DC}"
DC_ALF56 = f"alfworld-luna56-{DC}"
DC_ALF = f"alfworld-luna-{DC}"
#: Every section a removed method's cell must not appear in.
STUDY_SECTIONS = (
    "Caveats",
    "Cells",
    "Main results (test items)",
    "Comparison with the no-memory method",
    "Learning curves",
    "Providers",
    "Injected tokens",
    "Diagnostics",
)


def _leftover_cells(runs: Path) -> dict[str, Path]:
    """The removed method's cells a runs directory still holds: a complete
    WebShop cell, a complete ALFWorld cell of luna56, and an ALFWorld cell stopped
    mid-learning. Their ledgers: $0.05, $0.01 and $0.02."""
    web = make_cell(
        runs,
        "webshop",
        DC,
        test=shop("w", [1.0, 0.0, 1.0, 0.0]),
        calls=[
            _call("actor", "learning", 0.02, item="l1"),
            _call("learner", "learning", 0.02, item="l1"),
            _call("actor", "test", 0.01, item="u1"),
        ],
        snapshot={"method": DC, "notes": ""},
    )
    alf56 = make_cell(
        runs,
        "alfworld",
        DC,
        model_key="luna56",
        test=[_row("a1", 1.0)],
        calls=[_call("actor", "test", 0.01, item="u1", model=LUNA56)],
    )
    alf = make_cell(
        runs,
        "alfworld",
        DC,
        test=[],
        status="interrupted",
        calls=[_call("actor", "learning", 0.02, item="l1")],
    )
    return {DC_WEB: web, DC_ALF56: alf56, DC_ALF: alf}


def test_the_cells_of_a_removed_method_are_set_aside_but_their_money_counts(
    runs, removed_method
):
    """The removed method's cells left in a runs directory are set aside right after
    they are read. Every study number is what it is without them;
    they are listed under ``removed`` (the decision, each cell's status and costs), and
    their ledgers are counted apart from the matrix total."""
    before = report.build_report(runs, frozen=FP)
    _leftover_cells(runs)
    data = report.build_report(runs, frozen=FP)
    for key in (
        "cells",
        "comparisons",
        "errors",
        "protocol",
        "audit",
        "caveats",
        "pricing",
        "cells_run_more_than_once",
        "spend",
        "spend_union",
    ):
        assert data[key] == before[key], key
    for key, value in before["costs"].items():
        if key not in ("removed", "all_cells_total"):
            assert data["costs"][key] == value, key
    # The removed cells' money: apart from the matrix total, and in all cells' total.
    removed_costs = data["costs"]["removed"]
    assert removed_costs["total"] == pytest.approx(0.08)
    assert removed_costs["by_method"] == {DC: pytest.approx(0.08)}
    assert removed_costs["by_benchmark"] == {
        "alfworld": pytest.approx(0.03),
        "webshop": pytest.approx(0.05),
    }
    assert removed_costs["incomplete"] == {DC_ALF: pytest.approx(0.02)}
    assert data["costs"]["all_cells_total"] == pytest.approx(
        before["costs"]["total"] + 0.08
    )
    removed = data["removed"]
    assert removed["methods"] == {
        DC: {
            "decision": study.REMOVED_METHODS[DC].decision,
            "reason": study.REMOVED_METHODS[DC].reason,
            "cells": [DC_ALF, DC_ALF56, DC_WEB],
        }
    }
    listed = {c["cell_id"]: c for c in removed["cells"]}
    assert [c["cell_id"] for c in removed["cells"]] == [DC_ALF, DC_ALF56, DC_WEB]
    assert {c: (e["status"], e["complete"]) for c, e in listed.items()} == {
        DC_ALF: ("interrupted", False),
        DC_ALF56: ("complete", True),
        DC_WEB: ("complete", True),
    }
    assert listed[DC_WEB]["costs"]["total_cost"] == pytest.approx(0.05)
    assert listed[DC_WEB]["costs"]["learning"]["by_role"]["learner"]["cost"] == (
        pytest.approx(0.02)
    )
    assert listed[DC_ALF]["cell_dir"] == DC_ALF and listed[DC_ALF]["not_counted"] == "-"
    assert removed["cost_usd"] == pytest.approx(0.08)
    # Location independent: a merged view and the runs directory report one block.
    assert str(runs) not in json.dumps(removed) + json.dumps(data["costs"])

    md = report.render_markdown(data)
    for title in STUDY_SECTIONS:
        assert DC not in "\n".join(_section(md, title)), title
    assert not [
        line for line in _section(md, "Cost") if line.startswith("| `") and DC in line
    ]
    header = md.split("## Caveats")[0]
    assert (
        f"- **Removed methods:** `{DC}` ({study.REMOVED_METHODS[DC].decision}; 3 "
        "cell(s)). Their cells are set aside"
    ) in header
    assert "their ledgers' $0.0800 is counted apart from the matrix total" in header
    section = _section(md, "Removed methods")
    assert f"- `{DC}`, {study.REMOVED_METHODS[DC].decision}: " in "\n".join(section)
    assert study.REMOVED_METHODS[DC].reason in "\n".join(section)
    rows = [line for line in section if line.startswith("| `")]
    # The last column: the data is on disk (tools/remove_method_data.py not run).
    assert rows == [
        f"| `{DC_ALF}` | interrupted | no | 1 | 1 | $0.0200 | $0.0000 | $0.0000 | "
        "$0.0200 | - | yes |",
        f"| `{DC_ALF56}` | complete | yes | 1 | 1 | $0.0000 | $0.0100 | $0.0000 | "
        "$0.0100 | - | yes |",
        f"| `{DC_WEB}` | complete | yes | 1 | 1 | $0.0400 | $0.0100 | $0.0000 | "
        "$0.0500 | - | yes |",
    ]
    assert {c["on_disk"] for c in removed["cells"]} == {True}
    assert {c["deleted"] for c in removed["cells"]} == {None}
    assert removed["deleted"] == {"cells": 0, "cost_usd": 0.0, "removal_ids": []}
    cost = "\n".join(_section(md, "Cost"))
    matrix = before["costs"]["total"]
    assert f"Matrix total {report._usd(matrix)}" in cost
    assert (
        "**Removed methods' cells** (see 'Removed methods'): their ledgers recorded "
        f"$0.0800 more, not in the matrix total ({DC} $0.0800; 1 of the cells not "
        f"complete: `{DC_ALF}` $0.0200; of which $0.0000 in invalidated attempts). "
        f"**All cells: {report._usd(matrix + 0.08)}**"
    ) in cost
    # Rendered from report.json (sorted keys) the same.
    assert report.render_markdown(json.loads(json.dumps(data, sort_keys=True))) == md


def test_a_runs_directory_without_removed_cells_renders_as_before(runs, removed_method):
    """Without a removed method's cell the new keys add nothing to the Markdown: the
    report renders exactly as a report.json without them does."""
    data = json.loads(json.dumps(report.build_report(runs, frozen=FP)))
    assert data["removed"]["cells"] == [] and data["removed"]["cost_usd"] == 0.0
    assert data["removed"]["methods"][DC]["cells"] == []
    assert data["costs"]["removed"]["total"] == 0.0
    assert data["costs"]["all_cells_total"] == data["costs"]["total"]
    md = report.render_markdown(data)
    assert "Removed methods" not in md and "All cells" not in md
    older = json.loads(json.dumps(data))
    older.pop("removed")
    older["costs"].pop("removed")
    older["costs"].pop("all_cells_total")
    assert report.render_markdown(older) == md


def test_a_report_json_written_before_the_removal_renders_as_written(
    runs, removed_method
):
    """A stored schema-2 report.json from before the removal lists the removed method's
    cell among its cells and comparisons: it renders them as ordinary rows, as when it
    was written (the renderer never filters by method), with no 'Removed methods'."""
    assert report.SCHEMA_VERSION == 4
    data = json.loads(json.dumps(report.build_report(runs, frozen=FP)))
    data.pop("removed")
    data["costs"].pop("removed")
    data["costs"].pop("all_cells_total")
    awm = next(c for c in data["cells"] if c["cell_id"] == "webshop-luna-awm")
    # A schema-2 cell row names no directory.
    stored = {k: v for k, v in awm.items() if k != "cell_dir"}
    data["cells"].append({**stored, "cell_id": DC_WEB, "method": DC})
    row = next(r for r in data["comparisons"] if r["method"] == "awm" and "b" in r)
    data["comparisons"].append({**row, "method": DC})
    md = report.render_markdown(data)
    assert f"| `{DC_WEB}` | complete |" in "\n".join(_section(md, "Cells"))
    assert f"| webshop-luna | {DC} | 4 | 0.500 | 0.812 | 0.500 |" in md
    assert f"| webshop-luna | {DC} | success_rate | 4 |" in md
    assert "## Removed methods" not in md and "Removed methods:" not in md
    assert "All cells" not in md


def test_a_removed_cell_never_changes_the_exit_status(
    runs, tmp_path, capsys, removed_method
):
    """A complete removed cell with a failing or missing audit is set aside: the CLI
    exits as if it were absent (0, not 3) and names it. A runs directory holding only
    removed cells has no study cell: exit 2."""
    _leftover_cells(runs)
    argv = ["--runs-dir", str(runs), "--out", str(tmp_path / "o"), "--resamples", "50"]

    def run(args: list[str]) -> tuple[int, dict[str, Any]]:
        code = report.main(args)
        return code, json.loads(capsys.readouterr().out)

    code, printed = run(argv)
    assert code == 0
    # No audit yet: every complete cell of the study is unaudited, and only those.
    assert DC not in json.dumps(printed["not_audited"])
    assert len(printed["not_audited"]) == 5
    assert printed["removed"] == [DC_ALF, DC_ALF56, DC_WEB]
    assert printed["cells"] == 6 and printed["complete"] == 5
    # --require-audit: the study's complete cells passed; the removed ones did not.
    study_cells = [
        "alfworld-luna-none",
        "alfworld-luna-reasoningbank",
        "webshop-luna-none",
        "webshop-luna-awm",
        "webshop-luna56-awm",
    ]
    _write_audit(
        runs,
        [_audit_entry(runs, c) for c in study_cells]
        + [
            _audit_entry(runs, DC_WEB, violations=7),
            _audit_entry(runs, DC_ALF56, audited=False, reason="testing"),
        ],
    )
    code, printed = run([*argv, "--require-audit"])
    assert (code, printed["not_audited"]) == (0, [])
    data = json.loads((tmp_path / "o" / "report.json").read_text())
    assert DC not in json.dumps(data["audit"]["cells_not_passed"])
    # Only removed cells: nothing of the study to report.
    only = tmp_path / "only"
    _leftover_cells(only)
    code, printed = run(["--runs-dir", str(only), "--out", str(tmp_path / "p")])
    assert (code, printed["cells"], printed["complete"]) == (2, 0, 0)
    assert printed["removed"] == [DC_ALF, DC_ALF56, DC_WEB]


def test_two_directories_of_one_removed_cell_are_no_error(runs, removed_method):
    """Duplicate complete cells of a study group are an error nothing is compared past;
    a removed method's two directories of one cell id are two listed entries."""
    old = make_cell(
        runs,
        "webshop",
        DC,
        test=shop("w", [1.0]),
        calls=[_call("actor", "test", 0.03, item="u1")],
        directory=f"{DC_WEB}.old",
    )
    run_id = f"{DC_WEB}-a01-0dd0dd01"  # the old directory's instance: its own run
    (runs / f"{DC_WEB}-a01").rename(runs / run_id)
    _set_state(old, run_id=run_id, ledger=str(runs / run_id / "gateway.jsonl"))
    _leftover_cells(runs)
    data = report.build_report(runs, frozen=FP)
    assert data["errors"] == []
    entries = [(c["cell_id"], c["cell_dir"]) for c in data["removed"]["cells"]]
    assert (DC_WEB, DC_WEB) in entries and (DC_WEB, f"{DC_WEB}.old") in entries
    assert data["removed"]["methods"][DC]["cells"] == [DC_ALF, DC_ALF56, DC_WEB]
    assert data["costs"]["removed"]["total"] == pytest.approx(0.11)
    md = report.render_markdown(data)
    assert f"| `{DC_WEB}` (directory `{DC_WEB}.old`) | complete |" in md
    assert "**Error:**" not in md


def test_the_spend_windows_count_the_removed_cells_ledgers(tmp_path, removed_method):
    """The key paid for a removed cell's calls: a spend window sums every ledger of its
    directory, theirs included (never unattributed spend), and a gateway runs directory
    only a removed cell's ledgers name is still found."""
    runs = tmp_path / "runs"
    at = "2026-09-29T06:00:00+00:00"
    make_cell(
        runs,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[_timed(_call("actor", "test", 0.10, item="u1"), at, latency_s=1.0)],
    )
    make_cell(
        runs,
        "alfworld",
        DC,
        test=[_row("a1", 0.0)],
        calls=[_timed(_call("learner", "learning", 0.20), at, latency_s=1.0)],
    )
    start = "2026-09-29T05:00:00+00:00"
    budget = {"budget_usd": 12.0, "started_at": start, "start_usage_usd": 100.0}
    rows = [
        {"ts": start, "usage": 100.0, "budget": {**budget, "spent_usd": 0.0}},
        {
            "ts": "2026-09-29T08:00:00+00:00",
            "usage": 100.3,
            "budget": {**budget, "spent_usd": 0.3},
        },
    ]
    _write_balance(runs, rows)
    data = report.build_report(runs, frozen=FP)
    (spend,) = data["spend"]
    assert spend["ledger_cost_usd"] == pytest.approx(0.30)
    assert spend["unattributed_usd"] == pytest.approx(0.0)
    assert data["costs"]["total"] == pytest.approx(0.10)
    assert data["costs"]["removed"]["total"] == pytest.approx(0.20)
    assert data["costs"]["all_cells_total"] == pytest.approx(spend["ledger_cost_usd"])
    # The removed cell's gateway run in a directory of its own (another gateway runs
    # directory, as a second host's), with that directory's balance record.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run_id = f"alfworld-luna-{DC}-a01"
    (runs / run_id).rename(elsewhere / run_id)
    _set_state(
        runs / f"alfworld-luna-{DC}", ledger=str(elsewhere / run_id / "gateway.jsonl")
    )
    _write_balance(elsewhere, rows)
    data = report.build_report(runs, frozen=FP)
    by_dir = {s["gateway_dir"]: s for s in data["spend"]}
    assert set(by_dir) == {str(runs), str(elsewhere)}
    assert by_dir[str(elsewhere)]["ledger_cost_usd"] == pytest.approx(0.20)
    assert data["costs"]["removed"]["total"] == pytest.approx(0.20)


def test_comparisons_never_pair_a_removed_methods_cell(runs, removed_method):
    """Defence in depth: comparisons of cells read without the split (the loader's raw
    output) pair the study's methods only."""
    _leftover_cells(runs)
    cells = report.load_cells(runs)
    assert DC in {c.method for c in cells}
    assert not [r for r in report.comparisons(cells) if r["method"] == DC]


def test_the_removed_cells_calls_of_unknown_cost_are_named_in_the_cost_section(
    runs, removed_method
):
    """The removed cells' calls of unknown cost (cut off, unfinished) belong in the Cost
    section, whose spend windows count them. 'Cost not in the totals' names them, marked
    as a removed method's, and the removed cells' paragraph says the all-cells total
    leaves them out."""
    before = report.build_report(runs, frozen=FP)
    make_cell(
        runs,
        "alfworld",
        DC,
        test=[],
        status="interrupted",
        calls=[
            _call("actor", "learning", 0.02, item="l1"),
            _deadline_call("learner", "learning"),
            _deadline_call("learner", "learning"),
            {"event": "call_started", "call_id": "cut-mid-call", "role": "learner"},
        ],
    )
    data = report.build_report(runs, frozen=FP)
    removed = data["costs"]["removed"]
    assert (removed["unknown_cost_calls"], removed["unfinished_calls"]) == (2, 1)
    # The study's counts are what they were.
    for key in ("unknown_cost_calls", "unfinished_calls", "total"):
        assert data["costs"][key] == before["costs"][key], key
    (listed,) = data["removed"]["cells"]
    not_counted = "1 unfinished call(s); 2 cut-off call(s), cost unknown"
    assert listed["not_counted"] == not_counted
    md = report.render_markdown(data)
    cost = "\n".join(_section(md, "Cost"))
    assert "**Cost not in the totals:** " in cost
    assert f"`{DC_ALF}` (removed method) {not_counted}" in cost
    assert (
        "of which $0.0000 in invalidated attempts; not counting their 2 cut-off "
        "and 1 unfinished call(s) of unknown cost, named under 'Cost not in the "
        "totals'). **All cells: "
    ) in cost
    assert report.render_markdown(json.loads(json.dumps(data, sort_keys=True))) == md
    # A report JSON written before the count (no unfinished_calls): only the cut-off.
    older = json.loads(json.dumps(data))
    older["costs"]["removed"].pop("unfinished_calls")
    assert "not counting their 2 cut-off call(s) of unknown cost" in (
        report.render_markdown(older)
    )


def test_a_removed_cell_whose_cell_json_names_no_method_is_set_aside_by_its_id(
    runs, removed_method
):
    """A removed method's cell whose cell.json lost its method must not crash the
    report (KeyError 'method'). Its id names the method (``study.cell_method``): it is
    set aside and listed as when the method was there."""
    cells = _leftover_cells(runs)
    before = report.build_report(runs, frozen=FP)
    manifest = cells[DC_WEB] / "cell.json"
    record = json.loads(manifest.read_text())
    del record["cell"]["method"]
    manifest.write_text(json.dumps(record))
    data = report.build_report(runs, frozen=FP)
    for key in ("removed", "cells", "comparisons", "errors"):
        assert data[key] == before[key], key
    assert data["costs"] == before["costs"]
    assert {c["cell_id"]: c["method"] for c in data["removed"]["cells"]}[DC_WEB] == DC


# --- record, don't block (schema 3) -------------------------------------------------------

NOTE = "rerun after a code fix"
EARLY = "2026-09-29T10:00:00+00:00"
LATE = "2026-09-30T10:00:00+00:00"


def _superseded_state(by: int | None, *, scored: int = 4) -> dict[str, Any]:
    """attempt.json fields of an attempt a ``--new-attempt --note`` replaced after it
    completed (state ``invalid``, invalidation ``superseded``)."""
    return {
        "invalidation": report.INVALIDATION_SUPERSEDED,
        "previous_state": "complete",
        "invalid_reason": f"superseded by a new attempt (--new-attempt --note): {NOTE}",
        "started_at": EARLY,
        "superseded": {
            "note": NOTE,
            "at": LATE,
            "launch": "l2",
            "scored_items": scored,
            "test_actor_calls": scored,
            "cell_records_sha256": "e" * 64,
            "by_attempt": by,
        },
    }


def _new_attempt_state(supersedes: int) -> dict[str, Any]:
    return {
        "note": {"text": NOTE, "at": LATE, "launch": "l2", "new_attempt": True},
        "supersedes": supersedes,
    }


def _write_rows(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rerun_cell(runs: Path, second_state: str = "complete", **kw: Any) -> Path:
    """An ALFWorld ReasoningBank cell run twice in one directory: attempt 1 complete
    (success 0.75) and superseded, attempt 2 (success 0.25) ``second_state``."""
    first = binary("a", [1, 1, 1, 0])
    second = binary("a", [1, 0, 0, 0])
    make_cell(runs, "alfworld", "none", test=binary("a", [0, 0, 0, 0]))
    root = make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=second if second_state == "complete" else first,
        status="complete" if second_state == "complete" else "running",
        row_attempt=second_state == "complete",
        started_at=LATE,
        attempts=[
            (
                "invalid",
                [_call("actor", "test", 0.05, item="u1")],
                _superseded_state(2),
            ),
            (
                second_state,
                [_call("actor", "test", 0.02, item="u1")],
                _new_attempt_state(1),
            ),
        ],
        **kw,
    )
    _write_rows(root, "attempt-01/results.jsonl", [{**r, "attempt": 1} for r in first])
    return root


def test_a_rerun_in_the_same_directory_reports_the_newest_complete_attempt_and_never_mixes_them(
    tmp_path,
):
    """The reported attempt is the newest complete one; the superseded attempt is
    listed with its own score and cost (Runs and attempts), in no other table, and no
    number combines the two."""
    runs = tmp_path / "runs"
    _rerun_cell(runs)
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-reasoningbank"]
    assert (cell["reported_attempt"], cell["complete"]) == (2, True)
    assert cell["test"]["metrics"] == {"success_rate": 0.25}
    assert cell["newer_attempt"] is None and cell["record_problems"] == []
    other = cell["costs"]["other_attempts"]
    assert other["superseded"] == [1]
    assert other["superseded_cost"] == pytest.approx(0.05)
    assert other["invalid_cost"] == 0.0
    assert data["costs"]["superseded_attempts"] == pytest.approx(0.05)
    assert data["costs"]["total"] == pytest.approx(0.05 + 0.02)
    assert cell["failures"]["superseded_attempts"] == [1]
    detail = cell["attempts_detail"]
    assert [(a["attempt"], a["reported"], a["metrics"]) for a in detail] == [
        (1, False, {"success_rate": 0.75}),
        (2, True, {"success_rate": 0.25}),
    ]
    assert detail[0]["superseded"]["note"] == NOTE
    assert (detail[0]["state"], detail[0]["previous_state"]) == ("invalid", "complete")
    assert (detail[1]["note"], detail[1]["supersedes"]) == (NOTE, 1)
    assert detail[0]["test_items_scored"] == 4 and detail[0]["cost"] == pytest.approx(
        0.05
    )
    # It ran twice: listed, with the attempt that scored test items.
    assert data["cells_run_more_than_once"] == ["alfworld-luna-reasoningbank"]
    assert [a["attempt"] for a in cell["reruns"]["attempts_tested"]] == [1]
    (row,) = [c for c in data["comparisons"] if "skipped" not in c]
    assert (row["method_value"], row["none_value"]) == (0.25, 0.0)
    md = report.render_markdown(data)
    runs_table = _section(md, "Runs and attempts")
    assert any(
        "| 1 | invalid (was complete) | superseded by attempt 2: " + NOTE in line
        and "success_rate 0.750" in line
        and line.endswith("| no |")
        for line in runs_table
    )
    assert any(
        "success_rate 0.250" in line and line.endswith("| **yes** |")
        for line in runs_table
    )
    # The superseded score appears in no other table.
    outside = md.replace("\n".join(runs_table), "")
    assert "0.750" not in outside
    assert "other attempts (invalidated or superseded)" in md


def test_every_attempt_lists_the_versions_it_ran_under(tmp_path):
    """A rerun to compare old and fixed MIRIX code: every attempt (the superseded one
    too) names its protocol, its code (harness and MIRIX digests, commit) and how many
    version segments it ran, in the JSON and in the Markdown attempts table, so which
    code produced a superseded number is in the report, not only in attempt.json."""
    old_fp, old_harness, old_mirix, old_commit = "1" * 64, "a" * 64, "b" * 64, "c" * 40
    new_harness, new_mirix, new_commit = "d" * 64, "e" * 64, "9" * 40
    segment = {"index": 0, "why": "launch"}
    runs = tmp_path / "runs"
    make_cell(runs, "alfworld", "none", test=binary("a", [0, 0, 0, 0]))
    root = make_cell(
        runs,
        "alfworld",
        "mirix",
        test=binary("a", [1, 0, 0, 0]),
        row_attempt=True,
        started_at=LATE,
        attempts=[
            (
                "invalid",
                [_call("actor", "test", 0.05, item="u1")],
                {
                    **_superseded_state(2),
                    "protocol": {"fingerprint": old_fp, "constants_sha256": "2" * 64},
                    "code": {"harness": old_harness, "mirix": old_mirix},
                    "git": {"commit": old_commit, "dirty": True},
                    "version_segments": [segment, {**segment, "index": 1}],
                    "versions_mixed": {"protocol": False, "code": True, "commit": True},
                },
            ),
            (
                "complete",
                [_call("actor", "test", 0.02, item="u1")],
                {
                    **_new_attempt_state(1),
                    "code": {"harness": new_harness, "mirix": new_mirix},
                    "git": {"commit": new_commit, "dirty": False},
                },
            ),
        ],
    )
    first = binary("a", [1, 1, 1, 0])
    _write_rows(root, "attempt-01/results.jsonl", [{**r, "attempt": 1} for r in first])
    data = report.build_report(runs, frozen=FP)
    one, two = by_id(data)["alfworld-luna-mirix"]["attempts_detail"]
    assert one["protocol"] == {
        "fingerprint": old_fp,
        "constants_sha256": "2" * 64,
        "source": "attempt",
    }
    assert one["code"] == {
        "harness": old_harness,
        "mirix": old_mirix,
        "commit": old_commit,
        "dirty": True,
    }
    assert one["version_segments"] == 2
    assert one["versions_mixed"] == {"protocol": False, "code": True, "commit": True}
    # Attempt 2 records no protocol of its own: cell.json's (source "cell").
    assert two["protocol"]["fingerprint"] == FP
    assert two["protocol"]["source"] == "cell"
    assert two["code"] == {
        "harness": new_harness,
        "mirix": new_mirix,
        "commit": new_commit,
        "dirty": False,
    }
    assert (two["version_segments"], two["versions_mixed"]) == (0, None)
    md = report.render_markdown(data)
    runs_table = _section(md, "Runs and attempts")
    (header,) = [line for line in runs_table if line.startswith("| cell |")]
    assert "| host | protocol | code | segments | items scored |" in header
    (row1,) = [line for line in runs_table if "superseded by attempt 2" in line]
    assert f"`{old_fp[:12]}`" in row1
    assert f"harness `{old_harness[:12]}`" in row1
    assert f"mirix `{old_mirix[:12]}`" in row1
    assert f"commit `{old_commit[:12]}` (dirty)" in row1
    assert "| 2 | 4 |" in row1  # two segments, four items scored
    (row2,) = [line for line in runs_table if line.endswith("| **yes** |")]
    assert f"commit `{new_commit[:12]}`" in row2 and "(dirty)" not in row2
    assert f"mirix `{new_mirix[:12]}`" in row2


def test_while_a_rerun_runs_the_cell_is_not_complete(tmp_path):
    """While the newer attempt of a rerun runs, the cell is not complete (no metric,
    no comparison) and the header warns; the superseded attempt's score stays listed and
    its money counts."""
    runs = tmp_path / "runs"
    root = _rerun_cell(runs, second_state="learning")
    # The cell records attempt 1 published are still on disk (run.py rewrites them only
    # when attempt 2 completes).
    _write_rows(root, "results.jsonl", binary("a", [1, 1, 1, 0]))
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-reasoningbank"]
    assert cell["complete"] is False and "test" not in cell
    assert cell["reported_attempt"] == 2
    assert cell["newer_attempt"] == {
        "number": 2,
        "state": "learning",
        "started_at": LATE,
        "supersedes": 1,
    }
    (warning,) = [w for w in data["warnings"] if w["check"] == "newer_attempt_running"]
    assert warning["cells"] == ["alfworld-luna-reasoningbank"]
    assert "attempt 2 (learning)" in warning["detail"]
    assert [(a["attempt"], a["metrics"]) for a in cell["attempts_detail"]] == [
        (1, {"success_rate": 0.75}),
        (2, None),
    ]
    assert not [c for c in data["comparisons"] if c["method"] == "reasoningbank"]
    assert data["costs"]["incomplete"] == {
        "alfworld-luna-reasoningbank": pytest.approx(0.07)
    }
    assert "- **Warning (newer_attempt_running):**" in report.render_markdown(data)


def test_cell_records_of_another_attempt_are_not_reported(tmp_path):
    """The published rows must be the reported attempt's. Rows of another attempt
    (a rerun whose records were not rewritten) make the cell not complete, with a
    warning, never a mix of two attempts."""
    runs = tmp_path / "runs"
    root = _rerun_cell(runs)
    _write_rows(
        root, "results.jsonl", [{**r, "attempt": 1} for r in binary("a", [1, 1, 1, 0])]
    )
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-reasoningbank"]
    assert cell["complete"] is False and "test" not in cell
    assert cell["record_problems"] == [
        "results.jsonl rows belong to attempt 1, not the reported attempt 2"
    ]
    (warning,) = [
        w for w in data["warnings"] if w["check"] == "records_not_of_reported_attempt"
    ]
    assert warning["cells"] == ["alfworld-luna-reasoningbank"]


def _numbered(values: Sequence[int], **kw: Any) -> list[dict[str, Any]]:
    return [_row(f"d{i + 1}", float(v), **kw) for i, v in enumerate(values)]


def test_two_directories_of_one_cell_report_the_newest_complete_and_count_both_costs(
    tmp_path,
):
    """Two directories of one cell id (a MIRIX rerun on a second host, beside the first
    host's complete cell). The newest complete one is reported; the other is listed
    with its score, audit status and money, which counts in the matrix total."""
    runs = tmp_path / "runs"
    make_cell(runs, "alfworld", "none", test=_numbered([0, 0, 0, 0]))
    old = make_cell(
        runs,
        "alfworld",
        "mirix",
        test=_numbered([1, 0, 0, 0]),
        calls=[_call("actor", "test", 0.03, item="u1")],
        started_at=EARLY,
        instance="aaaa1111",
    )
    new = make_cell(
        runs,
        "alfworld",
        "mirix",
        test=_numbered([1, 1, 1, 0]),
        calls=[_call("actor", "test", 0.02, item="u1")],
        status="testing",
        started_at=LATE,
        directory="alfworld-luna-mirix@cloud",
        instance="bbbb2222",
    )
    data = report.build_report(runs, frozen=FP)
    (cell,) = [c for c in data["cells"] if c["cell_id"] == "alfworld-luna-mirix"]
    assert cell["cell_dir"] == old.name and cell["complete"] is True
    assert cell["test"]["metrics"] == {"success_rate": 0.25}
    (dup,) = data["duplicates"]
    (row,) = dup["not_used"]
    assert (row["cell_dir"], row["role"], row["complete"]) == (
        new.name,
        report.ROLE_NEWER_INCOMPLETE,
        False,
    )
    assert data["costs"]["total"] == pytest.approx(0.03 + 0.02)
    assert data["costs"]["superseded_directories"]["total"] == pytest.approx(0.02)
    assert data["costs"]["incomplete"] == {
        "alfworld-luna-mirix (directory alfworld-luna-mirix@cloud)": pytest.approx(0.02)
    }
    assert [w["check"] for w in data["warnings"]] == ["duplicate_directories"]
    # The newer, incomplete directory is not used and reports no attempt; the chosen
    # directory's attempt is reported and used.
    assert [(a["reported"], a["directory_used"]) for a in row["attempts_detail"]] == [
        (False, False)
    ]
    assert [(a["reported"], a["directory_used"]) for a in cell["attempts_detail"]] == [
        (True, True)
    ]
    runs_table = _section(report.render_markdown(data), "Runs and attempts")
    assert not [
        line
        for line in runs_table
        if line.startswith(f"| `alfworld-luna-mirix` | `{new.name}` |")
        and not line.endswith("| no |")
    ]
    # The rerun completes: it is reported, the first host's directory is listed with its
    # score.
    new = make_cell(
        runs,
        "alfworld",
        "mirix",
        test=_numbered([1, 1, 1, 0]),
        calls=[_call("actor", "test", 0.02, item="u1")],
        started_at=LATE,
        directory="alfworld-luna-mirix@cloud",
        instance="bbbb2222",
    )
    audit_entry = {
        "cell_id": "alfworld-luna-mirix",
        "root": str(new),
        "audited": True,
        "attempt": 1,
        "violations": 0,
        "not_checked": 0,
        "findings": [],
        "info": {"records_sha256": records_digest(CellPaths(new))},
    }
    _write_audit(runs, [audit_entry])
    audit_path = runs / "_audit" / "audit.json"
    audit_data = json.loads(audit_path.read_text())
    audit_data.update(
        schema=2,
        validity_checks=["H10.1:label_key"],
        duplicates=[
            {
                "cell_id": "alfworld-luna-mirix",
                "root": str(old),
                "role": "superseded",
                "chosen_root": str(new),
                "started_at": EARLY,
                "status": "complete",
                "complete": True,
            }
        ],
    )
    audit_path.write_text(json.dumps(audit_data))
    data = report.build_report(runs, frozen=FP)
    (cell,) = [c for c in data["cells"] if c["cell_id"] == "alfworld-luna-mirix"]
    assert cell["cell_dir"] == new.name
    assert cell["test"]["metrics"] == {"success_rate": 0.75}
    assert cell["audit"]["status"] == report.AUDIT_PASSED
    (row,) = data["duplicates"][0]["not_used"]
    assert (row["cell_dir"], row["role"]) == (old.name, report.ROLE_SUPERSEDED)
    assert row["test"]["metrics"] == {"success_rate": 0.25}
    assert row["audit"]["status"] == report.AUDIT_SUPERSEDED
    # JSON and Markdown agree: the old directory's own reported attempt is not used.
    assert [(a["reported"], a["directory_used"]) for a in row["attempts_detail"]] == [
        (True, False)
    ]
    assert [(a["reported"], a["directory_used"]) for a in cell["attempts_detail"]] == [
        (True, True)
    ]
    assert data["costs"]["total"] == pytest.approx(0.05)
    assert data["costs"]["by_method"]["mirix"] == pytest.approx(0.05)
    # Compared once, with the reported directory.
    (compared,) = [
        c for c in data["comparisons"] if c["method"] == "mirix" and "skipped" not in c
    ]
    assert compared["method_value"] == 0.75
    md = report.render_markdown(data)
    table = _section(md, "Duplicate cell directories")
    assert any(
        f"| `{new.name}` |" in line and line.endswith("| **yes** |") for line in table
    )
    assert any(
        f"| `{old.name}` |" in line
        and "success_rate 0.250" in line
        and line.endswith("| no (superseded) |")
        for line in table
    )
    assert f"`alfworld-luna-mirix` (directory `{old.name}`, not reported)" in md
    # Runs and attempts: only the reported directory's attempt says **yes**; the
    # directory not used names why its own reported attempt is not the cell's.
    runs_table = _section(md, "Runs and attempts")
    (new_row,) = [
        line
        for line in runs_table
        if line.startswith(f"| `alfworld-luna-mirix` | `{new.name}` |")
    ]
    (old_row,) = [
        line
        for line in runs_table
        if line.startswith(f"| `alfworld-luna-mirix` | `{old.name}` |")
    ]
    assert new_row.endswith("| **yes** |")
    assert old_row.endswith("| no (directory not used) |")
    assert "- **Warning (duplicate_directories):** 2 directories" in md
    out = tmp_path / "o"
    assert report.main(["--runs-dir", str(runs), "--out", str(out)]) == 0


def _learning_rows(values: Sequence[int]) -> list[dict[str, Any]]:
    rows = [_row(f"l{i}", float(v)) for i, v in enumerate(values)]
    for position, row in enumerate(rows):
        row["position"] = position
    return rows


def _audit_v2(runs: Path, cells: Sequence[dict[str, Any]], **kw: Any) -> Path:
    """An audit.json of schema 2 (it declares its validity checks)."""
    path = _write_audit(runs, cells, **kw)
    data = json.loads(path.read_text())
    data.update(
        schema=2,
        validity_checks=["H10.1:label_key", "H10.2:first_prompt_differs_across_cells"],
        unattributed_violations=kw.get("unattributed", 0),
        duplicates=kw.get("duplicates", []),
    )
    path.write_text(json.dumps(data))
    return path


COMPLETE = (
    "alfworld-luna-none",
    "alfworld-luna-reasoningbank",
    "webshop-luna-none",
    "webshop-luna-awm",
    "webshop-luna56-awm",
)


def test_require_audit_counts_the_validity_violations_and_only_named_cells_fail_on_runs_level_ones(
    runs, tmp_path, capsys
):
    """With audit.json schema 2 a run-level violation fails only the cells it
    names (the audit counts it in their entries); one that names no cell still fails
    every cell (fail closed). Warnings never fail a cell; a cell the auditing machine
    could not check is 'not checked', which --require-audit refuses (exit 3)."""
    entries = [_audit_entry(runs, c, not_checked=0, findings=[]) for c in COMPLETE]
    for entry in entries:
        if entry["cell_id"] == "alfworld-luna-reasoningbank":
            entry["violations"] = 1  # the copy of the across-cells violation
        if entry["cell_id"] == "webshop-luna-awm":
            entry["findings"] = [
                {
                    "invariant": "H10.3",
                    "check": "repeat_of_method",
                    "severity": "warning",
                }
            ]
    named = {
        "invariant": "H10.2",
        "check": "first_prompt_differs_across_cells",
        "severity": "violation",
        "cells": ["alfworld-luna-reasoningbank"],
    }
    _audit_v2(runs, entries, findings=[named])
    statuses = {
        c["cell_id"]: c["audit"]["status"]
        for c in report.build_report(runs, frozen=FP)["cells"]
        if c["complete"]
    }
    assert statuses == {
        **{c: report.AUDIT_PASSED for c in COMPLETE},
        "alfworld-luna-reasoningbank": report.AUDIT_FAILED,
    }
    argv = ["--runs-dir", str(runs), "--out", str(tmp_path / "o"), "--require-audit"]
    assert report.main(argv) == 3
    assert json.loads(capsys.readouterr().out)["not_audited"] == [
        "alfworld-luna-reasoningbank"
    ]
    # A run-level violation that names no cell fails every cell.
    _audit_v2(runs, entries, findings=[{**named, "cells": []}])
    cells = report.build_report(runs, frozen=FP)["cells"]
    assert {c["audit"]["status"] for c in cells if c["complete"]} == {
        report.AUDIT_FAILED
    }
    assert (
        "naming no cell"
        in by_id({"cells": cells})["alfworld-luna-none"]["audit"]["detail"]
    )
    # Not checked on the auditing machine: not passed.
    for entry in entries:
        entry["violations"] = 0
        if entry["cell_id"] == "webshop-luna56-awm":
            entry["not_checked"] = 2
    _audit_v2(runs, entries)
    data = report.build_report(runs, frozen=FP)
    assert (
        by_id(data)["webshop-luna56-awm"]["audit"]["status"] == report.AUDIT_NOT_CHECKED
    )
    assert by_id(data)["webshop-luna-awm"]["audit"]["status"] == report.AUDIT_PASSED
    assert "legacy_audit" not in {w["check"] for w in data["warnings"]}
    assert report.main(argv) == 3
    assert json.loads(capsys.readouterr().out)["not_audited"] == ["webshop-luna56-awm"]
    # A schema-1 audit.json: every run-level violation fails every cell, as before.
    _write_audit(runs, entries, findings=[named])
    data = report.build_report(runs, frozen=FP)
    assert {c["audit"]["status"] for c in data["cells"] if c["complete"]} == {
        report.AUDIT_FAILED
    }
    assert "schema 1" in by_id(data)["webshop-luna-awm"]["audit"]["detail"]
    assert "legacy_audit" in {w["check"] for w in data["warnings"]}


LINUX = {
    "hostname": "ip-10-0-0-5",
    "machine_id": "aws00001",
    "system": "Linux",
    "machine": "x86_64",
    "platform": "Linux-6.8-x86_64",
}
MAC = {
    "hostname": "mac",
    "machine_id": "mac00001",
    "system": "Darwin",
    "machine": "arm64",
    "platform": "macOS-26-arm64",
}


def _segment(index: int, fingerprint: str, harness: str, host: Mapping[str, Any], **kw):
    return {
        "index": index,
        "legacy": False,
        "why": "first attempt" if index == 0 else "learning resume of attempt 2",
        "phase": "learning",
        "first_launch": f"l{index}",
        "started_at": f"2026-09-30T0{index}:00:00+00:00",
        "launches": [f"l{index}"],
        "learning_from": {
            "position": 41 if index else 0,
            "stage": "after_task",
            "checkpoint": None,
        },
        "test_from": None,
        "protocol": {"fingerprint": fingerprint, "constants_sha256": "c"},
        "code": {"harness": harness, "harness_files": 31},
        "git": {"commit": "c0ffee0", "dirty": False},
        "host": dict(host),
        "tree": None,
        "version_file": None,
        "auto_freeze": {"status": "already_frozen"},
        "known_version": True,
        "note": None,
        "changed": None,
        **kw,
    }


def test_version_segments_mark_a_mixed_version_cell_and_its_comparisons(runs):
    """A learning resume under another protocol and code on another host goes on and
    is recorded as a second segment: the report lists
    both, marks the cell mixed, lists it under both versions, and flags its comparisons
    (same_protocol false, mixed_version, same_platform)."""
    root = runs / "alfworld-luna-reasoningbank"
    changed = {
        "protocol": True,
        "protocol_diff": {
            "from": FP,
            "to": EARLIER,
            "constants_changed": True,
            "constants_changed_keys": ["reasoning_effort"],
            "inconsistent": [],
            "changed": ["methods/prompts/awm/format.json"],
            "added": [],
            "removed": [],
        },
        "code": ["harness"],
        "code_files": {
            "harness": {"changed": ["methods/awm.py"], "added": [], "removed": []}
        },
        "commit": False,
        "tree": False,
    }
    _set_state(
        root,
        2,
        protocol={"fingerprint": FP, "constants_sha256": "c"},
        host=MAC,
        version_segments=[
            _segment(0, FP, HARNESS, MAC),
            _segment(1, EARLIER, "b" * 64, LINUX, changed=changed),
        ],
        versions_mixed={"protocol": True, "code": True, "commit": False},
    )
    _set_state(runs / "alfworld-luna-none", 1, host=MAC)
    known = {
        FP: {"fingerprint": FP, "current": True},
        EARLIER: {"fingerprint": EARLIER, "current": False},
    }
    data = report.build_report(runs, frozen=FP, versions=known)
    cell = by_id(data)["alfworld-luna-reasoningbank"]
    versions = cell["versions"]
    assert versions["mixed"] is True
    assert versions["protocol_fingerprints"] == sorted([FP, EARLIER])
    assert [s["segment"] for s in versions["segments"]] == [0, 1]
    second = versions["segments"][1]
    assert second["from"] == {"position": 41, "stage": "after_task", "checkpoint": None}
    assert second["changed"]["constants_changed_keys"] == ["reasoning_effort"]
    assert second["changed"]["protocol_files"]["changed"] == [
        "methods/prompts/awm/format.json"
    ]
    assert [h["platform"] for h in cell["hosts"]] == [
        MAC["platform"],
        LINUX["platform"],
    ]
    assert [h["segments"] for h in cell["hosts"]] == [[0], [1]]
    # The attempt-start protocol stays the cell's; the segments add the rest.
    assert cell["protocol"]["fingerprint"] == FP
    groups = data["protocol"]["versions"]["alfworld"]["protocol"]
    assert "alfworld-luna-reasoningbank" in groups[FP]
    assert "alfworld-luna-reasoningbank" in groups[EARLIER]
    assert data["protocol"]["cells_mixed_version"] == ["alfworld-luna-reasoningbank"]
    (row,) = [c for c in data["comparisons"] if c["group"] == "alfworld-luna"]
    assert row["same_protocol"] is False
    assert row["mixed_version"] == ["alfworld-luna-reasoningbank"]
    assert row["same_platform"] is False
    assert [w["cells"] for w in data["warnings"] if w["check"] == "mixed_version"] == [
        ["alfworld-luna-reasoningbank"]
    ]
    md = report.render_markdown(data)
    cells_table = _section(md, "Cells")
    assert any(
        "`alfworld-luna-reasoningbank`" in line
        and "(mixed: 2 versions)" in line
        and "ip-10-0-0-5" in line
        for line in cells_table
    )
    versions_table = _section(md, "Versions per cell")
    assert any(
        "| `alfworld-luna-reasoningbank` | 1 | learning checkpoint None, position 41, "
        "stage after_task |"
        in line
        and "constants reasoning_effort" in line
        and "harness changed methods/awm.py" in line
        for line in versions_table
    )
    assert "mixed version: alfworld-luna-reasoningbank" in "\n".join(
        _section(md, "Comparison with the no-memory method")
    )


def test_a_cell_without_segments_reports_one_implicit_segment(runs):
    """An attempt written before version segments: one segment from its attempt.json
    (protocol.attempt_versions synthesizes it), never mixed."""
    _set_state(runs / "alfworld-luna-none", 1, host=MAC, started_at=EARLY)
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-none"]
    (segment,) = cell["versions"]["segments"]
    assert (segment["segment"], segment["legacy"]) == (0, True)
    assert segment["protocol_fingerprint"] == FP
    assert cell["versions"]["mixed"] is False
    assert cell["versions"]["unknown_fingerprints"] == []
    assert [h["hostname"] for h in cell["hosts"]] == ["mac"]
    assert data["protocol"]["cells_mixed_version"] == []
    assert "## Versions per cell" not in report.render_markdown(data)


def test_checks_not_reverified_and_audit_warnings_are_listed_per_cell(runs):
    """The checks the audit could not re-verify under a cell's earlier version,
    its recorded conditions and its warnings are listed per cell; a schema-1 audit's
    protocol_version_unauditable warnings count as not re-verified."""
    entry = _audit_entry(
        runs,
        "webshop-luna-awm",
        attempt=1,
        not_checked=0,
        findings=[
            {"invariant": "H10.3", "check": "repeat_of_method", "severity": "warning"},
            {"invariant": "H10.3", "check": "repeat_of_method", "severity": "warning"},
            {"invariant": "H10.6", "check": "host_changed", "severity": "warning"},
        ],
    )
    entry["info"].update(
        not_reverified=[
            {
                "check": "H10.1:label_key",
                "inputs": [
                    "protocol.py constant reasoning_effort 20ff7aef -> 69c87149"
                ],
                "findings": 0,
            },
            {"check": "H10.4:snapshot", "inputs": ["methods/awm.py"], "findings": 1},
        ],
        recorded_counts={"H10.3:failure_counted_and_continued": 3},
    )
    _audit_v2(runs, [entry])
    data = report.build_report(runs, frozen=FP)
    detail = by_id(data)["webshop-luna-awm"]["audit_detail"]
    assert [c["check"] for c in detail["not_reverified"]] == [
        "H10.1:label_key",
        "H10.4:snapshot",
    ]
    assert detail["warnings_by_check"] == {
        "H10.3:repeat_of_method": 2,
        "H10.6:host_changed": 1,
    }
    assert detail["recorded_counts"] == {"H10.3:failure_counted_and_continued": 3}
    assert by_id(data)["webshop-luna-none"]["audit_detail"] is None
    md = report.render_markdown(data)
    assert any(
        line.startswith("| `webshop-luna-awm` |") and "| 2 |" in line
        for line in _section(md, "Cells")
    )
    versions = "\n".join(_section(md, "Versions per cell"))
    assert "| `webshop-luna-awm` | not re-verified | H10.1:label_key |" in versions
    warnings = "\n".join(_section(md, "Diagnostics"))
    assert "| `webshop-luna-awm` | H10.3:repeat_of_method | warning | 2 |" in warnings
    assert (
        "| `webshop-luna-awm` | H10.3:failure_counted_and_continued | recorded | 3 |"
        in warnings
    )
    # Schema 1: the unauditable-version warnings are the checks not re-verified.
    legacy = _audit_entry(
        runs,
        "webshop-luna-awm",
        attempt=1,
        findings=[
            {
                "invariant": "H10.6",
                "check": "protocol_version_unauditable",
                "severity": "warning",
                "detail": "reasoning_effort changed",
            }
        ],
    )
    _write_audit(runs, [legacy])
    detail = by_id(report.build_report(runs, frozen=FP))["webshop-luna-awm"][
        "audit_detail"
    ]
    assert detail["not_reverified"] == [
        {
            "check": "H10.6:protocol_version_unauditable",
            "inputs": ["reasoning_effort changed"],
            "findings": 1,
        }
    ]


def _listing(tmp_path: Path, endpoints: Sequence[tuple[str, str]]) -> Path:
    path = tmp_path / "endpoints.json"
    path.write_text(
        json.dumps(
            {
                "_source": "GET .../endpoints, 2026-09-29",
                "data": {
                    "id": LUNA56,
                    "endpoints": [
                        {"provider_name": p, "quantization": q} for p, q in endpoints
                    ],
                },
            }
        )
    )
    return path


def _luna56_test_call(provider: str, item: str, status: str = "ok") -> dict[str, Any]:
    return {
        **_call("actor", "test", 0.001, item=item, provider=provider, model=LUNA56),
        "status": status,
    }


def test_provider_and_quantization_distribution_per_cell(tmp_path):
    """Each cell's test actor calls by provider and by the endpoint listing's
    quantization label (a provider with two endpoints is 'mixed', one the listing does
    not name 'not listed'), offline, from --provider-endpoints."""
    runs = tmp_path / "runs"
    calls = (
        [_luna56_test_call("A", "u1")] * 3
        + [_luna56_test_call("B", "u2")] * 2
        + [_luna56_test_call("C", "u3")]
    )
    make_cell(
        runs,
        "alfworld",
        "none",
        model_key="luna56",
        test=[_row("a1", 1.0), _row("a2", 0.0), _row("a3", 1.0)],
        calls=calls,
    )
    listing = _listing(
        tmp_path, [("A", "fp8"), ("B", "fp4"), ("C", "fp8"), ("C", "unknown")]
    )
    data = report.build_report(runs, frozen=FP, provider_endpoints=[listing])
    mix = by_id(data)["alfworld-luna56-none"]["provider_mix"]
    assert mix["test_actor_calls"] == {"A": 3, "B": 2, "C": 1}
    assert mix["quantization_calls"] == {"fp8": 3, "fp4": 2, "mixed(fp8/unknown)": 1}
    assert data["provider_endpoints"]["by_model"][LUNA56]["C"] == ["fp8", "unknown"]
    assert data["provider_endpoints"]["sources"][0]["_source"].startswith("GET")
    assert "quantization_unlisted" not in {w["check"] for w in data["warnings"]}
    md = report.render_markdown(data)
    assert "fp8 3, fp4 2, mixed(fp8/unknown) 1" in md
    # Without a listing: every provider is 'not listed', and the header says so.
    plain = report.build_report(runs, frozen=FP)
    mix = by_id(plain)["alfworld-luna56-none"]["provider_mix"]
    assert mix["quantization_calls"] == {report.NOT_LISTED: 6}
    (warning,) = [w for w in plain["warnings"] if w["check"] == "quantization_unlisted"]
    assert "no endpoint listing was given" in warning["detail"]
    # The CLI reads the listing too.
    out = tmp_path / "o"
    argv = [
        "--runs-dir",
        str(runs),
        "--out",
        str(out),
        "--provider-endpoints",
        str(listing),
    ]
    assert report.main(argv) == 0
    stored = json.loads((out / "report.json").read_text())
    assert stored["provider_endpoints"]["files"] == [str(listing.absolute())]


def test_luna56_items_grouped_by_main_provider(tmp_path):
    """Every test item's main provider (the provider of most of its ok actor calls,
    from the row's gateway_call_ids) and the accuracy per provider, pooled over the
    cells of one model and benchmark; an ALFWorld item served by A, B, B is B's (2/3,
    mixed)."""
    runs = tmp_path / "runs"
    shop_calls = {m: [] for m in ("none", "awm")}
    rows = {}
    plan = {"none": ["A", "A", "B", "B"], "awm": ["A", "B", "B", "B"]}
    values = {"none": [1, 0, 1, 1], "awm": [1, 1, 0, 1]}
    for method in ("none", "awm"):
        rows[method] = []
        for i, provider in enumerate(plan[method]):
            call = _luna56_test_call(provider, f"t{i}")
            shop_calls[method].append(call)
            row = shop("w", [values[method][i]])[0]
            row.update(raw_id=f"w{i + 1}", gateway_call_ids=[call["call_id"]])
            rows[method].append(row)
        make_cell(
            runs,
            "webshop",
            method,
            model_key="luna56",
            test=rows[method],
            calls=shop_calls[method],
        )
    alf_calls = [
        _luna56_test_call("A", "x"),
        _luna56_test_call("B", "x"),
        _luna56_test_call("B", "x"),
    ]
    alf_calls.append(_luna56_test_call("C", "x", status="upstream_error"))
    alf_row = _row("a1", 1.0, gateway_call_ids=[c["call_id"] for c in alf_calls])
    make_cell(
        runs, "alfworld", "none", model_key="luna56", test=[alf_row], calls=alf_calls
    )
    listing = _listing(tmp_path, [("A", "fp8"), ("B", "fp4")])
    data = report.build_report(runs, frozen=FP, provider_endpoints=[listing])
    alf = by_id(data)["alfworld-luna56-none"]
    assert alf["accuracy_by_main_provider"] == {
        "B": {"items": 1, "correct": 1.0, "mixed_items": 1}
    }
    assert alf["provider_mix"]["mixed_provider_items"] == 1
    cells = by_id(data)
    assert cells["webshop-luna56-none"]["accuracy_by_main_provider"] == {
        "A": {"items": 2, "correct": 1.0, "mixed_items": 0},
        "B": {"items": 2, "correct": 2.0, "mixed_items": 0},
    }
    assert cells["webshop-luna56-awm"]["accuracy_by_main_provider"] == {
        "A": {"items": 1, "correct": 1.0, "mixed_items": 0},
        "B": {"items": 3, "correct": 2.0, "mixed_items": 0},
    }
    pooled = data["provider_accuracy"][LUNA56]["webshop"]
    assert pooled["A"]["items"] == 3 and pooled["A"]["accuracy"] == pytest.approx(2 / 3)
    assert pooled["B"]["items"] == 5 and pooled["B"]["accuracy"] == pytest.approx(4 / 5)
    assert pooled["B"]["share_of_items"] == pytest.approx(5 / 8)
    assert (pooled["A"]["quantization"], pooled["B"]["quantization"]) == ("fp8", "fp4")
    assert pooled["B"]["cells"] == ["webshop-luna56-awm", "webshop-luna56-none"]
    md = report.render_markdown(data)
    section = "\n".join(_section(md, "Providers"))
    assert "### Accuracy by main provider" in section
    assert f"#### {LUNA56}" in section
    assert "| webshop | B | fp4 | 5 | 0 | 0.800 | 62% |" in section
    assert "| alfworld | B | fp4 | 1 | 1 | 1.000 | 100% |" in section
    assert report.CAVEAT_PROVIDER_ACCURACY in section


def test_rows_without_call_ids_fall_back_to_the_ledger_item_uid(tmp_path):
    """A row written without gateway_call_ids finds its calls by its task uid in
    the ledger's test-phase actor calls."""
    runs = tmp_path / "runs"
    row = _row("a1", 1.0)
    calls = [
        _luna56_test_call("A", row["task_uid"]),
        _luna56_test_call("A", row["task_uid"]),
        _luna56_test_call("B", row["task_uid"]),
        _luna56_test_call("B", "another item"),
    ]
    make_cell(runs, "alfworld", "none", model_key="luna56", test=[row], calls=calls)
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna56-none"]
    assert cell["accuracy_by_main_provider"] == {
        "A": {"items": 1, "correct": 1.0, "mixed_items": 1}
    }
    assert cell["provider_mix"]["items_by_main_provider"] == {"A": 1}
    assert cell["provider_mix"]["items_without_provider"] == 0


def test_items_scored_under_degraded_conditions_are_listed_with_an_accuracy_without_unserved_items(
    tmp_path,
):
    """Rows flagged provider_unserved (scored unanswered after the provider
    refused twice) and retrieval_degraded (MIRIX injected a degraded retrieval) are
    listed per cell, with the accuracy without the provider-unserved items."""
    runs = tmp_path / "runs"
    rows = _numbered([1, 0, 1, 0])
    rows[0]["retrieval_degraded"] = True
    rows[1].update(
        provider_unserved=True,
        end_reason="provider_unserved",
        failure={
            "kind": "harness",
            "code": "provider_unserved",
            "role": "actor",
            "status": 404,
            "detail": "No endpoints found",
            "retries": 0,
        },
        score={"primary": 0.0, "answered": False, "metrics": {}},
    )
    learning = _learning_rows([1, 1])
    learning[1]["retrieval_degraded"] = True
    make_cell(runs, "alfworld", "mirix", test=rows, learning=learning)
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-mirix"]
    assert cell["degraded"] == {
        "test": {"provider_unserved": ["d2"], "retrieval_degraded": ["d1"]},
        "learning": {"retrieval_degraded": 1},
    }
    assert cell["test"]["metrics"] == {"success_rate": 0.5}
    assert cell["test"]["metrics_without_provider_unserved"] == {
        "success_rate": pytest.approx(2 / 3)
    }
    assert cell["test"]["unanswered"] == 1
    assert cell["tolerated"]["test"]["provider_unserved"] == ["d2"]
    (warning,) = [w for w in data["warnings"] if w["check"] == "degraded_items"]
    assert warning["cells"] == ["alfworld-luna-mirix"]
    md = report.render_markdown(data)
    section = "\n".join(_section(md, "Items scored under degraded conditions"))
    assert (
        "| `alfworld-luna-mirix` | provider_unserved (test) | 1: `d2` | success_rate 0.667 |"
        in section
    )
    assert (
        "| `alfworld-luna-mirix` | retrieval_degraded (learning) | 1 | - |" in section
    )
    # Without a flagged row: no metric without them, no section.
    plain = by_id(report.build_report(runs, frozen=FP))["alfworld-luna-mirix"]
    assert plain["test"]["metrics_without_provider_unserved"] is not None


def test_the_report_counts_unserved_rows_by_the_audits_predicate(tmp_path):
    """The report reads a provider-unserved row as the audit does
    (``infra/failures.provider_unserved_row``): the flag, or the ``end_reason`` or failure
    code of a row written without it; and it says when an item is scored so (a test item
    after two consecutive no-endpoint 404 refusals, a learning item after one refusal)."""
    runs = tmp_path / "runs"
    rows = _numbered([1, 0, 1, 0])
    unanswered = {"primary": 0.0, "answered": False, "metrics": {}}
    rows[1].update(end_reason="provider_unserved", score=unanswered)
    rows[3].update(
        failure={
            "kind": "harness",
            "code": "provider_unserved",
            "role": "actor",
            "status": 404,
            "detail": "No endpoints found",
            "retries": 0,
        },
        score=unanswered,
    )
    make_cell(runs, "alfworld", "mirix", test=rows, learning=_learning_rows([1, 1]))
    data = report.build_report(runs, frozen=FP)
    cell = by_id(data)["alfworld-luna-mirix"]
    assert cell["degraded"]["test"] == {"provider_unserved": ["d2", "d4"]}
    assert cell["test"]["metrics_without_provider_unserved"] == {
        "success_rate": pytest.approx(1.0)
    }
    md = report.render_markdown(data)
    section = "\n".join(_section(md, "Items scored under degraded conditions"))
    assert "two consecutive no-endpoint 404 refusals" in section
    assert "learning items: one refusal" in section
    assert "refused the request twice (HTTP 400/404/413)" not in md


def test_learning_refusals_of_awm_and_reasoningbank_are_counted(tmp_path):
    """The learning calls a method's provider refused are counted and gone on
    from (AWM keeps its previous workflow). A missing counter reads 0."""
    runs = tmp_path / "runs"
    rejected = {
        "kind": "harness",
        "code": "upstream_rejected",
        "role": "learner",
        "status": 413,
        "detail": "request too large",
        "retries": 0,
    }
    make_cell(
        runs,
        "alfworld",
        "awm",
        test=[_row("a1", 1.0)],
        snapshot={
            "method": "awm",
            "counters": {
                "inductions": 10,
                "induction_rejected": 2,
                "induction_retries": 1,
            },
            "inductions": [
                {
                    "index": 0,
                    "trigger": "t0",
                    "request_bytes": 5_000_000,
                    "rejected": rejected,
                    "retry": {"refusal": "payload", "used": 3, "dropped": 4},
                    "provider": "P1",
                },
                {"index": 1, "trigger": "t1", "wrote": True},
                {"index": 2, "trigger": "t2", "rejected": {**rejected, "status": 400}},
            ],
        },
    )
    make_cell(
        runs,
        "alfworld",
        "reasoningbank",
        test=[_row("a1", 1.0)],
        snapshot={"method": "reasoningbank", "counters": {"entries": 1}},
    )
    data = report.build_report(runs, frozen=FP)
    awm = by_id(data)["alfworld-luna-awm"]["method_refusals"]
    assert awm["counters"] == {
        "induction_rejected": 2,
        "induction_retries": 1,
        "judge_rejected": 0,
    }
    assert awm["inductions"] == 3
    assert awm["rejected_by_status"] == {"400": 1, "413": 1}
    first = awm["rejected_examples"][0]
    assert (first["status"], first["provider"], first["retried"]) == (413, "P1", True)
    rb = by_id(data)["alfworld-luna-reasoningbank"]["method_refusals"]
    assert rb["counters"] == {"extraction_rejected": 0, "judge_rejected": 0}
    (warning,) = [w for w in data["warnings"] if w["check"] == "method_refusals"]
    assert warning["cells"] == ["alfworld-luna-awm"]
    md = report.render_markdown(data)
    assert (
        "| `alfworld-luna-awm` | awm | induction_rejected 2, induction_retries 1, "
        "judge_rejected 0 | 400 x1, 413 x1 | 3 |"
    ) in md


def test_mirix_lost_learning_calls_are_counted_from_the_ledger(tmp_path):
    """The learning side's MIRIX and embedding calls that did not end ok and that
    no resume discarded are counted from the ledger (exact, whatever the method
    recorded), next to what the method accepted (snapshot tolerated) and the batches
    accepted after rollbacks (attempt.json accepted_learning_calls)."""
    runs = tmp_path / "runs"
    lost = {**_call("mirix", "learning", 0.0), "status": "upstream_error"}
    discarded = {**_call("mirix", "learning", 0.0), "status": "upstream_error"}
    embedding = {**_call("embedding", "setup", 0.0), "status": "deadline_exceeded"}
    test_failure = {**_call("mirix", "test", 0.0), "status": "upstream_error"}
    root = make_cell(
        runs,
        "alfworld",
        "mirix",
        test=[_row("a1", 1.0)],
        calls=[
            lost,
            discarded,
            embedding,
            test_failure,
            _call("mirix", "learning", 0.01),
        ],
        snapshot={
            "method": "mirix",
            "learning_runs": [],
            "tolerated": {
                "counts": {"gateway_refusal": 1, "mirix_learning_failed": 1},
                "problems": [
                    {"reason": "gateway_refusal", "text": "400", "rollbacks": 0}
                ],
            },
        },
        retrievals=[
            {
                "frozen": True,
                "item_ids": ["s1"],
                "failure": None,
                "degraded": {"code": "x"},
            },
            {"frozen": True, "item_ids": [], "failure": None},
            {
                "frozen": True,
                "failure": {"role": "embedding", "code": "mirix_retrieval"},
            },
        ],
    )
    _write_rows(
        root,
        "attempt-01/discarded.jsonl",
        [{"segment": 1, "calls": [{"call_id": discarded["call_id"]}]}],
    )
    _set_state(
        root, 1, accepted_learning_calls=[{"call_id": lost["call_id"], "role": "mirix"}]
    )
    data = report.build_report(runs, frozen=FP)
    mirix = by_id(data)["alfworld-luna-mirix"]["mirix"]
    assert mirix["learning_calls_lost"] == 2
    assert mirix["learning_calls_lost_by"] == {
        "embedding/deadline_exceeded": 1,
        "mirix/upstream_error": 1,
    }
    assert mirix["tolerated"] == {"gateway_refusal": 1, "mirix_learning_failed": 1}
    assert (mirix["tolerated_problems"], mirix["accepted_learning_calls"]) == (1, 1)
    test = mirix["retrievals"]["test"]
    assert (test["count"], test["empty"], test["degraded"], test["refused"]) == (
        2,
        1,
        1,
        1,
    )
    (warning,) = [w for w in data["warnings"] if w["check"] == "mirix_learning_lost"]
    assert (
        "2 lost call(s) (method: gateway_refusal 1, mirix_learning_failed 1)"
        in (warning["detail"])
    )
    md = report.render_markdown(data)
    assert (
        "| 1 (degraded, injected: 1) | 2 (gateway_refusal 1, mirix_learning_failed 1) |"
        in md
    )


def test_a_report_json_without_warnings_renders(runs):
    """A stored schema-2 report.json (no warnings, duplicates, attempts or versions per
    segment) renders, and its errors still show."""
    data = json.loads(json.dumps(report.build_report(runs, frozen=FP)))
    for key in ("warnings", "duplicates", "provider_accuracy", "provider_endpoints"):
        data.pop(key)
    for cell in data["cells"]:
        for key in (
            "attempts_detail",
            "versions",
            "hosts",
            "audit_detail",
            "provider_mix",
            "degraded",
            "method_refusals",
            "cell_dir",
        ):
            cell.pop(key, None)
    data["errors"] = [
        {"detail": "2 complete none cells in group g", "cells": ["/r/a", "/r/b"]}
    ]
    md = report.render_markdown(data)
    assert "- **Error:** 2 complete none cells in group g: /r/a, /r/b" in md
    assert "Warning (" not in md and "## Duplicate cell directories" not in md


def test_the_report_never_combines_two_attempts_or_directories(tmp_path):
    """Every published number comes from exactly one attempt's rows, for a
    rerun in the same directory and for two directories of one cell id."""
    runs = tmp_path / "runs"
    _rerun_cell(runs)  # attempt 1: 0.75 (superseded), attempt 2: 0.25
    other = tmp_path / "runs2"
    make_cell(other, "alfworld", "none", test=binary("a", [0, 0, 0, 0]))
    make_cell(
        other,
        "alfworld",
        "reasoningbank",
        test=binary("a", [1, 1, 1, 0]),
        started_at=EARLY,
    )
    make_cell(
        other,
        "alfworld",
        "reasoningbank",
        test=binary("a", [1, 0, 0, 0]),
        started_at=LATE,
        directory="alfworld-luna-reasoningbank@cloud",
    )
    for directory in (runs, other):
        data = report.build_report(directory, frozen=FP)
        cell = by_id(data)["alfworld-luna-reasoningbank"]
        published = [
            cell["test"]["metrics"]["success_rate"],
            *[
                c["method_value"]
                for c in data["comparisons"]
                if c["method"] == "reasoningbank" and "skipped" not in c
            ],
        ]
        assert published == [0.25, 0.25], directory
        md = report.render_markdown(data)
        main = "\n".join(_section(md, "Main results (test items)"))
        assert "| alfworld-luna | reasoningbank | 4 | 0.250 |" in main
        assert "0.750" not in main and "0.500" not in main


# --- one directory rule in the audit and the report (integration seam) ------------------


_T0 = "2026-09-29T05:00:00+00:00"
_T1 = "2026-09-30T05:00:00+00:00"


@pytest.mark.parametrize(
    ("plain", "linked"),
    [
        # (status, started_at of the dated attempt, cell.json created_at) per directory:
        # a host's alfworld-luna-mirix and its rerun on another host, linked as
        # '...@cloud'.
        (("complete", _T0, _T0), ("complete", _T1, _T1)),  # the newer complete one
        (("complete", _T1, _T1), ("complete", _T0, _T0)),  # the plain name is newer
        (("complete", _T0, _T0), ("complete", _T0, _T1)),  # created_at decides
        (("complete", _T0, _T0), ("complete", _T0, _T0)),  # the entry name decides
        (("complete", _T0, _T0), ("testing", _T1, _T1)),  # newer, still running
        (("complete", _T1, _T1), ("testing", _T0, _T0)),  # older and never finished
        (("testing", _T0, _T0), ("testing", _T1, _T1)),  # neither complete
        (
            ("complete", None, None),
            ("complete", _T0, None),
        ),  # missing times sort lowest
    ],
)
def test_the_audit_and_the_report_choose_one_directory_alike(tmp_path, plain, linked):
    """The directory rule is defined once (the audit's ``choose_directory``) and applied
    by the report
    (``select_reported``): for two directories of one cell id both agree on which one is
    read and what the other is."""
    from evals.skill_bench import audit

    runs = tmp_path / "runs"
    names = ("alfworld-luna-mirix", "alfworld-luna-mirix@cloud")
    for name, (status, started, created) in zip(names, (plain, linked)):
        make_cell(
            runs,
            "alfworld",
            "mirix",
            test=[_row("a1", 1.0)],
            status=status,
            started_at=started,
            created_at=created,
            directory=name,
        )
    cells = report.load_cells(runs, split_mode=None)
    assert sorted(c.paths.root.name for c in cells) == sorted(names)
    kept, not_used = report.select_reported(cells)
    (reported,) = kept
    chosen, others = audit.choose_directory([runs / n for n in names])
    assert Path(chosen.root).name == reported.paths.root.name
    assert [(Path(f.root).name, role) for f, role in others] == [
        (cell.paths.root.name, role) for cell, _winner, role in not_used
    ]
