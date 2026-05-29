"""Tests for :mod:`evals.metaclaw.cli` — the budget gate, --yes guard, and
estimate plumbing introduced in slice #6 (issue 06).

These tests stay offline by stubbing :func:`run_arm` / :func:`run_both` and
:mod:`sys.stdin` so the gate logic is exercised in isolation from the real
benchmark subprocesses.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from evals.metaclaw import cli
from evals.metaclaw.runner import RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_runners(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list]:
    """Patch run_arm and run_both with capturing stubs that return rc=0.

    Returns a dict with ``arm_calls`` / ``both_calls`` lists the tests can
    inspect to assert what kwargs were forwarded (notably ``extra_meta``).
    """
    captured: dict[str, list] = {"arm": [], "both": []}

    def _fake_run_arm(**kwargs: Any) -> RunResult:
        captured["arm"].append(kwargs)
        out = Path(kwargs.get("out_dir") or tmp_path / "default")
        out.mkdir(parents=True, exist_ok=True)
        return RunResult(arm=kwargs["arm"], exit_code=0, output_dir=out)

    def _fake_run_both(**kwargs: Any) -> tuple[RunResult, RunResult]:
        captured["both"].append(kwargs)
        out = Path(kwargs.get("out_dir") or tmp_path / "default-both")
        out.mkdir(parents=True, exist_ok=True)
        return (
            RunResult(arm="metaclaw", exit_code=0, output_dir=out),
            RunResult(arm="mirix", exit_code=0, output_dir=out),
        )

    monkeypatch.setattr(cli, "run_arm", _fake_run_arm)
    monkeypatch.setattr(cli, "run_both", _fake_run_both)
    return captured


def _make_stdin(is_tty: bool, response: str = "") -> io.StringIO:
    """Build a stdin-like stub whose ``isatty()`` returns *is_tty*.

    When *is_tty* is True we also queue *response* so :func:`input` works.
    """
    buf = io.StringIO(
        response + ("\n" if response and not response.endswith("\n") else "")
    )
    buf.isatty = lambda: is_tty  # type: ignore[method-assign]
    return buf


# ---------------------------------------------------------------------------
# _format_seconds_human
# ---------------------------------------------------------------------------


def test_format_seconds_human_sub_minute() -> None:
    assert cli._format_seconds_human(0) == "~0s"
    assert cli._format_seconds_human(45) == "~45s"


def test_format_seconds_human_minutes() -> None:
    assert cli._format_seconds_human(60) == "~1m"
    assert cli._format_seconds_human(180) == "~3m"


def test_format_seconds_human_hours_and_minutes() -> None:
    assert cli._format_seconds_human(3600) == "~1h0m"
    assert cli._format_seconds_human(3600 + 1800) == "~1h30m"
    assert cli._format_seconds_human(10800) == "~3h0m"


# ---------------------------------------------------------------------------
# Days < 5: no gate
# ---------------------------------------------------------------------------


def test_days_below_threshold_skips_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured = _stub_runners(monkeypatch, tmp_path)
    # Non-TTY without --yes — would normally fail at >=5 days, but days=2 skips.
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))
    rc = cli.main(
        ["--arm", "metaclaw", "--days", "2", "--output-dir", str(tmp_path / "r1")]
    )
    assert rc == 0
    assert len(captured["arm"]) == 1
    out = capsys.readouterr().out
    # No "Selected:" banner for sub-threshold runs.
    assert "Selected:" not in out


# ---------------------------------------------------------------------------
# Acceptance #2 / #3 / #4: --days >= 5 gate logic
# ---------------------------------------------------------------------------


def test_days_five_non_tty_without_yes_returns_rc3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Acceptance: --days 5 with no TTY and no --yes -> rc=3 + structured msg."""
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))
    rc = cli.main(
        ["--arm", "metaclaw", "--days", "5", "--output-dir", str(tmp_path / "r1")]
    )
    assert rc == cli.RC_BUDGET_NEEDS_YES == 3
    # run_arm must NOT have been called.
    assert captured["arm"] == []
    out = capsys.readouterr().out
    assert "Selected: --arm metaclaw --days 5" in out
    assert "Estimated wall-time:" in out
    assert "pass --yes for non-interactive runs >= 5 days" in out


def test_days_five_with_yes_runs_without_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Acceptance: --days 5 --yes -> skip prompt, still print summary."""
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))

    # Sentinel: input() must not be called.
    def _no_input(*a: Any, **kw: Any) -> str:  # noqa: ARG001
        raise AssertionError("input() should not be called when --yes is set")

    monkeypatch.setattr("builtins.input", _no_input)
    rc = cli.main(
        [
            "--arm",
            "metaclaw",
            "--days",
            "5",
            "--yes",
            "--output-dir",
            str(tmp_path / "r1"),
        ]
    )
    assert rc == 0
    assert len(captured["arm"]) == 1
    out = capsys.readouterr().out
    assert "Selected: --arm metaclaw --days 5" in out


def test_days_five_tty_y_response_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Acceptance: TTY + user types 'y' -> proceeds to runner."""
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=True))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    rc = cli.main(
        ["--arm", "metaclaw", "--days", "5", "--output-dir", str(tmp_path / "r1")]
    )
    assert rc == 0
    assert len(captured["arm"]) == 1


def test_days_five_tty_n_response_aborts_with_rc130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Acceptance: TTY + user types 'n' (or empty) -> rc=130 (user abort)."""
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=True))
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    rc = cli.main(
        ["--arm", "metaclaw", "--days", "5", "--output-dir", str(tmp_path / "r1")]
    )
    assert rc == cli.RC_USER_ABORTED == 130
    assert captured["arm"] == []


# ---------------------------------------------------------------------------
# Acceptance #5: estimates land in run.meta.json via extra_meta
# ---------------------------------------------------------------------------


def test_estimates_flow_to_run_arm_extra_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Acceptance: estimated_wallclock_seconds_{min,max} reach run_arm."""
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))
    rc = cli.main(
        [
            "--arm",
            "metaclaw",
            "--days",
            "5",
            "--yes",
            "--output-dir",
            str(tmp_path / "r1"),
        ]
    )
    assert rc == 0
    em = captured["arm"][0]["extra_meta"]
    assert "estimated_wallclock_seconds_min" in em
    assert "estimated_wallclock_seconds_max" in em
    assert "estimated_rounds" in em
    assert em["estimated_wallclock_seconds_min"] > 0
    assert (
        em["estimated_wallclock_seconds_max"] >= em["estimated_wallclock_seconds_min"]
    )
    # 5 days * (10+11+12+10+13)=56 rounds * n=3 retry; default min=20s -> 3360s
    expected_rounds = 10 + 11 + 12 + 10 + 13  # canonical day01..day05 round counts
    assert em["estimated_rounds"] == expected_rounds
    assert em["estimated_wallclock_seconds_min"] == expected_rounds * 3 * 20
    assert em["estimated_wallclock_seconds_max"] == expected_rounds * 3 * 60


def test_estimates_flow_to_run_both_extra_meta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))
    rc = cli.main(
        ["--arm", "both", "--days", "5", "--yes", "--output-dir", str(tmp_path / "r1")]
    )
    assert rc == 0
    em = captured["both"][0]["extra_meta"]
    assert em["estimated_wallclock_seconds_min"] > 0
    assert (
        em["estimated_wallclock_seconds_max"] >= em["estimated_wallclock_seconds_min"]
    )


# ---------------------------------------------------------------------------
# Negative-days hard-fail (defensive)
# ---------------------------------------------------------------------------


def test_negative_days_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_runners(monkeypatch, tmp_path)
    monkeypatch.setattr(cli.sys, "stdin", _make_stdin(is_tty=False))
    with pytest.raises(SystemExit) as ei:
        cli.main(["--arm", "metaclaw", "--days", "-1"])
    assert ei.value.code != 0


# The wiring from `cli.main` → `run_arm(extra_meta=...)` is covered by
# `test_estimates_flow_to_run_arm_extra_meta` above.  The bytes-on-disk
# verification is covered by every `test_runner.py` test that asserts on
# `run.meta.json`, which already exercises the same write path.  A
# `cli.main`-driven end-to-end variant cannot stub the proxy via
# `monkeypatch.setattr(runner, "_start_proxy", ...)` because `run_arm`
# binds the default-arg reference at def-time; the patched module attr is
# never consulted and the real `_start_proxy` spawns clawdbot.  Use the
# explicit `proxy_starter=` kwarg pattern (see test_runner.py) instead.
