"""Tests for results_aggregator (PRD Testing Decisions).

Fixture-based: no external services. Tests verify external behavior of
aggregate() — table rendering, missing-arm handling, deviation block.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.metaclaw_aligned.aggregator import aggregate


def _make_fixture_run(tmp: Path, name: str, *, accuracy: float, correct: int,
                     total: int, per_test: dict[str, dict]) -> Path:
    """Construct a runs/<name>/ directory with the paper-bench-shaped report
    + run.meta.json that aggregator expects."""
    rd = tmp / name
    (rd / "infer" / "run_001").mkdir(parents=True, exist_ok=True)
    report = {
        "summary": {
            "total_questions": total,
            "correct": correct,
            "accuracy": accuracy,
            "metrics": {"passed": accuracy, "f1": accuracy},
        },
        "by_task": per_test,
    }
    (rd / "infer" / "run_001" / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    meta = {
        "arm": name, "model": "openai/gpt-5.2",
        "started_at": "20260520-100000", "finished_at": "20260520-101500",
        "wall_seconds": 900.0, "exit_code": 0,
        "mirix_base_url": "http://127.0.0.1:8531",
        "mirix_user_id": "eval-aligned-gating",
    }
    (rd / "run.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rd


def test_four_arm_table_populated(tmp_path):
    runs = {
        "A": _make_fixture_run(tmp_path, "armA-mirix-skills", accuracy=0.6, correct=9, total=15,
                               per_test={
                                   "day01": {"accuracy": 0.6, "questions": [{}] * 5},
                                   "day02": {"accuracy": 0.6, "questions": [{}] * 5},
                                   "day03": {"accuracy": 0.6, "questions": [{}] * 5},
                               }),
        "B": _make_fixture_run(tmp_path, "armB-mirix-legacy", accuracy=0.5, correct=8, total=15,
                               per_test={
                                   "day01": {"accuracy": 0.4, "questions": [{}] * 5},
                                   "day02": {"accuracy": 0.6, "questions": [{}] * 5},
                                   "day03": {"accuracy": 0.5, "questions": [{}] * 5},
                               }),
        "C": _make_fixture_run(tmp_path, "armC-baseline", accuracy=0.33, correct=5, total=15,
                               per_test={
                                   "day01": {"accuracy": 0.4, "questions": [{}] * 5},
                                   "day02": {"accuracy": 0.2, "questions": [{}] * 5},
                                   "day03": {"accuracy": 0.4, "questions": [{}] * 5},
                               }),
        "D": _make_fixture_run(tmp_path, "armD-paper-native", accuracy=0.7, correct=11, total=15,
                               per_test={
                                   "day01": {"accuracy": 0.8, "questions": [{}] * 5},
                                   "day02": {"accuracy": 0.6, "questions": [{}] * 5},
                                   "day03": {"accuracy": 0.7, "questions": [{}] * 5},
                               }),
    }
    md = aggregate(runs, phase="gating", bench="small")

    # Smoke checks
    assert "Four-arm comparison" in md
    assert "armA-mirix-skills" in md
    assert "armB-mirix-legacy" in md
    assert "armC-baseline" in md
    assert "armD-paper-native" in md
    # Per-test rows
    assert "| day01 |" in md and "| day02 |" in md and "| day03 |" in md
    # Aggregate row
    assert "**mean**" in md
    assert "0.600" in md  # arm A mean
    assert "0.700" in md  # arm D mean
    # Pairwise deltas — A - D = -0.1, with sign
    assert "+0.100" in md or "-0.100" in md
    # Known deviations
    assert "Known deviations from paper" in md
    assert "Single seed" in md
    assert "skills_only" in md


def test_missing_arm_handled_gracefully(tmp_path):
    """Aggregator must not crash if one arm didn't run."""
    runs = {
        "A": _make_fixture_run(tmp_path, "armA", accuracy=0.5, correct=5, total=10,
                               per_test={"day01": {"accuracy": 0.5, "questions": [{}] * 10}}),
        "B": None,  # missing
        "C": _make_fixture_run(tmp_path, "armC", accuracy=0.3, correct=3, total=10,
                               per_test={"day01": {"accuracy": 0.3, "questions": [{}] * 10}}),
        "D": None,  # missing
    }
    md = aggregate(runs, phase="gating")

    # Missing arms show markers
    assert "**missing**" in md
    # Surviving arms still render normally
    assert "0.500" in md
    assert "0.300" in md
    # Deltas with missing arm should be "—"
    assert "| A − B | — |" in md
    assert "| A − D | — |" in md


def test_zero_pass_renders_valid_md(tmp_path):
    """All-zero accuracy still produces well-formed markdown."""
    runs = {
        "A": _make_fixture_run(tmp_path, "armA", accuracy=0.0, correct=0, total=15,
                               per_test={
                                   "day01": {"accuracy": 0.0, "questions": [{}] * 5},
                                   "day02": {"accuracy": 0.0, "questions": [{}] * 5},
                                   "day03": {"accuracy": 0.0, "questions": [{}] * 5},
                               }),
        "B": None, "C": None, "D": None,
    }
    md = aggregate(runs)
    # No crashes; mean row present
    assert "**mean**" in md
    assert "0.000" in md
    assert "Known deviations" in md


def test_deviations_section_canonical_bullets(tmp_path):
    """PRD D12: deviations section must list the 4 canonical bullets."""
    runs = {"A": None, "B": None, "C": None, "D": None}
    md = aggregate(runs, phase="gating", bench="small")
    assert "Skill backend" in md
    assert "Single seed" in md
    assert "Mode coverage" in md
    assert "Subset" in md  # added when bench=small


def test_run_metadata_section_present(tmp_path):
    runs = {
        "A": _make_fixture_run(tmp_path, "armA", accuracy=0.5, correct=5, total=10,
                               per_test={"day01": {"accuracy": 0.5, "questions": [{}] * 10}}),
        "B": None, "C": None, "D": None,
    }
    md = aggregate(runs)
    assert "Run metadata" in md
    assert "openai/gpt-5.2" in md
    assert "20260520-100000" in md


def test_aggregate_picks_latest_report(tmp_path):
    """Aggregator finds report.json under infer/<latest>/ even when multiple
    run-id subdirs exist (paper bench creates run_YYYYMMDD_HHMMSS)."""
    rd = tmp_path / "armA"
    for run_id in ("run_20260520_080000", "run_20260520_120000", "run_20260520_100000"):
        (rd / "infer" / run_id).mkdir(parents=True, exist_ok=True)
        report = {
            "summary": {"total_questions": 5, "correct": 3, "accuracy": 0.6},
            "by_task": {"day01": {"accuracy": 0.6, "questions": [{}] * 5}},
        }
        (rd / "infer" / run_id / "report.json").write_text(json.dumps(report), encoding="utf-8")

    meta = {"model": "openai/gpt-5.2", "wall_seconds": 100, "exit_code": 0}
    (rd / "run.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    md = aggregate({"A": rd, "B": None, "C": None, "D": None})
    # picked the lexicographically latest run id (120000)
    assert "run_20260520_120000" in md
