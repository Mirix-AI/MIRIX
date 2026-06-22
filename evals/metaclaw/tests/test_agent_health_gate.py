"""H2 — Sanity gate also asserts AGENT-EXECUTION health.

The FIX7 sanity gate verified the distill/evolve/records pipeline but NEVER
checked that the *agent itself* produced answers. A network-degenerate run
(every day04+ round errored with a GatewayClientRequestError /
EmbeddedAttemptSessionTakeoverError / connect failure, pass-rate ~0) would
FALSE-PASS the gate and its 0% delta would be trusted.

``evals.metaclaw.agent_health_gate`` scans the bench_output infer_result.json
files and FAILS when:
  * the round error-rate exceeds a threshold, OR
  * a run of consecutive days collapses to ~0 pass-rate (the day04+ signature).
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.metaclaw import agent_health_gate as gate


# --------------------------------------------------------------------------- #
# Helpers to synthesize a bench_output tree                                    #
# --------------------------------------------------------------------------- #


def _write_round(
    root: Path, day: str, rnd: str, *, status: str, passed: bool | None, error: str | None = None
) -> None:
    rdir = root / "bench_output" / "run_x" / day / day / rnd
    rdir.mkdir(parents=True, exist_ok=True)
    obj: dict = {
        "test_id": day,
        "group_id": day,
        "round_id": rnd,
        "status": status,
    }
    if passed is not None:
        obj["inline_score"] = {"passed": passed}
    if error is not None:
        obj["error"] = error
    (rdir / "infer_result.json").write_text(json.dumps(obj), encoding="utf-8")


def _healthy_run(root: Path, days: int = 5, rounds: int = 10) -> None:
    for d in range(1, days + 1):
        for r in range(1, rounds + 1):
            _write_round(root, f"day{d:02d}", f"r{r}", status="success", passed=(r % 2 == 0))


# --------------------------------------------------------------------------- #
# Healthy run passes                                                           #
# --------------------------------------------------------------------------- #


def test_healthy_run_passes(tmp_path: Path):
    _healthy_run(tmp_path, days=6, rounds=10)
    report = gate.evaluate_agent_health(tmp_path)
    assert report.ok is True, report.failures
    assert report.total_rounds == 60
    assert report.errored_rounds == 0


# --------------------------------------------------------------------------- #
# The day04+ network-degenerate signature is caught                           #
# --------------------------------------------------------------------------- #


def test_network_degenerate_run_fails(tmp_path: Path):
    """day01-03 healthy, day04+ every round errors with a gateway/connect error
    and passes nothing → both the error-rate AND the consecutive-zero-day check
    should fire."""
    # Healthy first three days
    for d in range(1, 4):
        for r in range(1, 11):
            _write_round(tmp_path, f"day{d:02d}", f"r{r}", status="success", passed=(r % 2 == 0))
    # day04..day08 collapse: every round errors, nothing passes
    for d in range(4, 9):
        for r in range(1, 11):
            _write_round(
                tmp_path,
                f"day{d:02d}",
                f"r{r}",
                status="error",
                passed=False,
                error="GatewayClientRequestError: EmbeddedAttemptSessionTakeoverError",
            )
    report = gate.evaluate_agent_health(tmp_path)
    assert report.ok is False
    # Both signatures should be present in the failure reasons.
    joined = " ".join(report.failures).lower()
    assert "error" in joined
    assert "consecutive" in joined or "collapse" in joined or "zero" in joined


def test_high_error_rate_alone_fails(tmp_path: Path):
    """Errors scattered across days (no full-day collapse) still fail purely on
    the aggregate error-rate threshold."""
    for d in range(1, 6):
        for r in range(1, 11):
            # 6/10 rounds errored every day → 60% error rate, well over threshold
            errored = r <= 6
            _write_round(
                tmp_path,
                f"day{d:02d}",
                f"r{r}",
                status="error" if errored else "success",
                passed=(not errored) and (r % 2 == 0),
                error="httpx.ConnectError: All connection attempts failed" if errored else None,
            )
    report = gate.evaluate_agent_health(tmp_path)
    assert report.ok is False
    assert report.errored_rounds == 30
    assert any("error rate" in f.lower() for f in report.failures)


def test_consecutive_zero_pass_days_fails_even_without_errors(tmp_path: Path):
    """Even if rounds don't carry an explicit error, a run of consecutive days
    with ~0 pass-rate is the collapse signature and must fail."""
    # 2 healthy days then 4 days where the agent answered but passed nothing.
    for d in range(1, 3):
        for r in range(1, 11):
            _write_round(tmp_path, f"day{d:02d}", f"r{r}", status="success", passed=True)
    for d in range(3, 7):
        for r in range(1, 11):
            _write_round(tmp_path, f"day{d:02d}", f"r{r}", status="success", passed=False)
    report = gate.evaluate_agent_health(tmp_path)
    assert report.ok is False
    assert any("consecutive" in f.lower() or "collapse" in f.lower() for f in report.failures)


# --------------------------------------------------------------------------- #
# Empty / missing bench_output is itself a failure (never silently pass)       #
# --------------------------------------------------------------------------- #


def test_non_adjacent_collapsed_days_do_not_count_as_consecutive(tmp_path: Path):
    """codex P1: collapsed day01, day03, day05 with healthy day02/day04 between
    them must NOT satisfy the '3 consecutive collapsed days' rule. A numeric gap
    OR a healthy day in between breaks the run."""
    # Alternate collapsed / healthy: 01 dead, 02 ok, 03 dead, 04 ok, 05 dead.
    for d in range(1, 6):
        dead = d % 2 == 1
        for r in range(1, 11):
            _write_round(
                tmp_path, f"day{d:02d}", f"r{r}", status="success", passed=(not dead)
            )
    report = gate.evaluate_agent_health(tmp_path)
    # Error rate is 0 (all status=success) and no 3-in-a-row collapse → passes.
    assert report.ok is True, report.failures


def test_collapsed_days_with_numeric_gap_break_the_run(tmp_path: Path):
    """codex P1: collapsed day01, day02 then a MISSING day03 (no rounds) then
    collapsed day04, day05 — the numeric gap at day03 must break the consecutive
    run so neither side reaches the >=3 threshold."""
    for d in (1, 2, 4, 5):
        for r in range(1, 11):
            _write_round(tmp_path, f"day{d:02d}", f"r{r}", status="success", passed=False)
    report = gate.evaluate_agent_health(tmp_path)
    # Two runs of length 2 separated by the day03 gap → no >=3 collapse.
    assert not any(
        "consecutive" in f.lower() or "collapse" in f.lower() for f in report.failures
    ), report.failures


def test_no_rounds_found_fails(tmp_path: Path):
    (tmp_path / "bench_output").mkdir()
    report = gate.evaluate_agent_health(tmp_path)
    assert report.ok is False
    assert any("no" in f.lower() and "round" in f.lower() for f in report.failures)


# --------------------------------------------------------------------------- #
# CLI entrypoint exits non-zero on a degenerate run                            #
# --------------------------------------------------------------------------- #


def test_cli_main_exit_codes(tmp_path: Path):
    _healthy_run(tmp_path, days=6, rounds=10)
    assert gate.main([str(tmp_path)]) == 0

    bad = tmp_path / "bad"
    bad.mkdir()
    for d in range(1, 6):
        for r in range(1, 11):
            _write_round(bad, f"day{d:02d}", f"r{r}", status="error", passed=False,
                         error="connect failed")
    assert gate.main([str(bad)]) == 1
