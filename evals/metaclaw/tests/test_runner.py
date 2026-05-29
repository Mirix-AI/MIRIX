"""Tests for :mod:`evals.metaclaw.runner` with subprocess hooks stubbed.

These tests exercise only externally observable behaviour: the env vars
passed to the (stubbed) proxy + bench, the structure of the output dir,
the parsed RunResult, and that ``finally:`` cleanup fires on both success
and exception.

Real subprocess invocation lives in the smoke run gated on the integration
marker; these tests stay offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals.metaclaw import runner
from evals.metaclaw.runner import RunResult, run_arm


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to a known API key so _resolve_benchmark_env doesn't raise."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    # Force _load_env_file to return {} so we don't pick up real .env values.
    monkeypatch.setattr(runner, "_load_env_file", lambda: {})


def _fake_proxy_starter() -> tuple[MagicMock, list[dict]]:
    """Build a proxy starter stub that records its env dict."""
    captured: list[dict] = []

    def _starter(yaml: Path, port: int, log: Path, env: dict):  # noqa: ARG001
        captured.append({"yaml": yaml, "port": port, "log": log, "env": dict(env)})
        proc = MagicMock()
        proc.pid = 12345
        proc.poll = MagicMock(return_value=None)
        return proc

    return _starter, captured


def _fake_proxy_stopper() -> tuple[MagicMock, list[Any]]:
    stopped: list[Any] = []

    def _stopper(proc):
        stopped.append(proc)

    return _stopper, stopped


def _fake_bench(rc: int = 0):
    captured: dict = {}

    def _runner(tests_used: Path, out_dir: Path, env: dict, **kw):  # noqa: ARG001
        captured["tests_used"] = tests_used
        captured["out_dir"] = out_dir
        captured["env"] = dict(env)
        captured["kwargs"] = kw
        # Drop a minimal report.json so parsing exercises real shape.
        report_dir = out_dir / "infer_x"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": {
                "total_questions": 5,
                "correct": 3,
                "accuracy": 0.6,
                "tokens": {
                    "agent": {"total_input": 100, "input": 80, "output": 40},
                    "compaction": {"total_input": 10, "input": 5, "output": 5},
                },
            }
        }
        (report_dir / "report.json").write_text(json.dumps(report))
        return rc

    return _runner, captured


def test_run_arm_metaclaw_happy_path(tmp_path: Path) -> None:
    starter, captured_proxy = _fake_proxy_starter()
    stopper, stopped = _fake_proxy_stopper()
    bench, captured_bench = _fake_bench(rc=0)

    result = run_arm(
        arm="metaclaw",
        days=1,
        out_dir=tmp_path / "run1",
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
    )

    assert isinstance(result, RunResult)
    assert result.arm == "metaclaw"
    assert result.exit_code == 0
    assert result.accuracy == pytest.approx(0.6)
    assert result.total_tokens == 240  # 100+80+40 + 10+5+5
    # finally: stopper invoked
    assert len(stopped) == 1
    # proxy received the BENCHMARK_* env triple
    env = captured_proxy[0]["env"]
    assert env["BENCHMARK_API_KEY"] == "sk-test-fake"
    assert env["BENCHMARK_BASE_URL"] == runner.DEFAULT_BENCHMARK_BASE_URL
    assert env["BENCHMARK_MODEL"] == runner.DEFAULT_BENCHMARK_MODEL
    assert env["METACLAW_SKILLS_PROVIDER"] == "metaclaw"
    assert "METACLAW_PROXY_PORT" in env
    assert "METACLAW_ROOT" in env
    # bench received the same env triple
    assert captured_bench["env"]["METACLAW_PROXY_PORT"] == env["METACLAW_PROXY_PORT"]
    # Slice produced and was passed to bench
    assert captured_bench["tests_used"].exists()
    sliced = json.loads(captured_bench["tests_used"].read_text())
    assert len(sliced["test"]) == 1
    # run.meta.json populated
    meta = json.loads((tmp_path / "run1" / "run.meta.json").read_text())
    assert meta["arm"] == "metaclaw"
    assert meta["days"] == 1
    assert meta["mirix_url"] is None
    assert meta["exit_code"] == 0
    assert meta["accuracy"] == pytest.approx(0.6)
    assert meta["total_tokens"] == 240
    assert meta["vendor_sha"]  # non-empty
    assert meta["started_at"]
    assert meta["finished_at"]


def test_run_arm_cleanup_on_bench_exception(tmp_path: Path) -> None:
    starter, _ = _fake_proxy_starter()
    stopper, stopped = _fake_proxy_stopper()

    def _exploding_bench(*a, **kw):  # noqa: ARG001
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_arm(
            arm="metaclaw",
            days=1,
            out_dir=tmp_path / "run2",
            proxy_starter=starter,
            proxy_stopper=stopper,
            bench_runner=_exploding_bench,
        )
    assert len(stopped) == 1, "finally: must stop proxy even on bench exception"


def test_run_arm_mirix_health_check_fails_returns_rc2(tmp_path: Path) -> None:
    """Slice #3 wires the mirix arm.  When the server is unreachable, the
    runner returns rc=2 with an informative error rather than raising."""
    result = run_arm(
        arm="mirix",
        days=1,
        out_dir=tmp_path / "mirix-bad",
        mirix_url="http://127.0.0.1:1",  # guaranteed unreachable
    )
    assert result.exit_code == 2
    assert result.report_summary.get("error") == "mirix_unreachable"


def test_run_arm_rejects_both_arm_directly() -> None:
    """``--arm both`` is dispatched via :func:`run_both` (slice #5); calling
    :func:`run_arm` with ``arm='both'`` is a programmer error."""
    with pytest.raises(ValueError, match="run_both"):
        run_arm(arm="both", days=1)


def test_run_arm_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm(arm="bogus", days=1)


def test_run_arm_rejects_negative_days() -> None:
    with pytest.raises(ValueError, match="days must be >= 0"):
        run_arm(arm="metaclaw", days=-1)


def test_run_arm_raises_when_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_API_KEY", raising=False)
    starter, _ = _fake_proxy_starter()
    stopper, _ = _fake_proxy_stopper()
    bench, _ = _fake_bench(rc=0)
    with pytest.raises(RuntimeError, match="BENCHMARK_API_KEY"):
        run_arm(
            arm="metaclaw",
            days=1,
            out_dir=tmp_path / "no_key",
            proxy_starter=starter,
            proxy_stopper=stopper,
            bench_runner=bench,
        )


def test_run_arm_propagates_extra_env(tmp_path: Path) -> None:
    starter, captured = _fake_proxy_starter()
    stopper, _ = _fake_proxy_stopper()
    bench, _ = _fake_bench(rc=0)
    run_arm(
        arm="metaclaw",
        days=1,
        out_dir=tmp_path / "envrun",
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
        extra_env={"BENCHMARK_MODEL": "fakemodel/v1"},
    )
    assert captured[0]["env"]["BENCHMARK_MODEL"] == "fakemodel/v1"


def test_run_arm_preserves_dirs_on_failure(tmp_path: Path) -> None:
    """When bench exit code != 0, scratch dirs survive for debugging."""
    starter, _ = _fake_proxy_starter()
    stopper, _ = _fake_proxy_stopper()
    bench, captured = _fake_bench(rc=2)
    result = run_arm(
        arm="metaclaw",
        days=1,
        out_dir=tmp_path / "fail_run",
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
    )
    assert result.exit_code == 2
    # The output dir always survives — that's where run.meta.json lives.
    assert (tmp_path / "fail_run" / "run.meta.json").exists()


def test_run_arm_writes_proxy_yaml_with_skills_only(tmp_path: Path) -> None:
    starter, captured = _fake_proxy_starter()
    stopper, _ = _fake_proxy_stopper()
    bench, _ = _fake_bench(rc=0)
    run_arm(
        arm="metaclaw",
        days=1,
        out_dir=tmp_path / "yaml_check",
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
    )
    yaml_path = captured[0]["yaml"]
    body = yaml_path.read_text()
    assert "mode: skills_only" in body
    assert "auto_evolve: true" in body
    assert "retrieval_mode: template" in body
    assert "enabled: false" in body  # rl/scheduler/memory


# ---------------------------------------------------------------------------
# Slice #6: MIRIX health-check fail-fast formatting + reset endpoint
# ---------------------------------------------------------------------------


def test_mirix_health_diagnose_unreachable_returns_no_status() -> None:
    """Connection-level failures must surface as (False, None, <detail>).

    Total wall-time stays well under the 6s acceptance budget because the
    socket connect to 127.0.0.1:1 fails fast.
    """
    import time as _time

    t0 = _time.monotonic()
    ok, status, detail = runner._mirix_health_diagnose(
        "http://127.0.0.1:1", timeout_s=2
    )
    elapsed = _time.monotonic() - t0
    assert ok is False
    assert status is None
    assert "connection" in detail.lower() or "refused" in detail.lower()
    assert elapsed < 6.0, f"diagnose took {elapsed:.2f}s — must stay under 6s budget"


def test_format_mirix_unreachable_has_required_fields() -> None:
    """Acceptance #1: error message contains URL, status, and start hint."""
    msg = runner._format_mirix_unreachable(
        "http://127.0.0.1:8531", None, "connection refused"
    )
    assert "ERROR: MIRIX server not reachable at http://127.0.0.1:8531" in msg
    assert "status: no response" in msg
    assert "detail: connection refused" in msg
    assert "Start it with: python scripts/start_server.py --port 8531" in msg


def test_format_mirix_unreachable_renders_numeric_status() -> None:
    msg = runner._format_mirix_unreachable(
        "http://x", 503, "HTTP 503 Service Unavailable"
    )
    assert "status: 503" in msg


def test_run_arm_mirix_bad_url_returns_rc2_under_6s(tmp_path: Path) -> None:
    """Acceptance #1: bad MIRIX URL -> rc=2, structured msg, <6s total."""
    import time as _time

    t0 = _time.monotonic()
    result = run_arm(
        arm="mirix",
        days=1,
        out_dir=tmp_path / "fast_fail",
        mirix_url="http://127.0.0.1:1",  # guaranteed unreachable
    )
    elapsed = _time.monotonic() - t0
    assert result.exit_code == 2
    assert result.report_summary.get("error") == "mirix_unreachable"
    assert result.report_summary.get("url") == "http://127.0.0.1:1"
    # status may be None for connection-level failures, but the key must exist.
    assert "status" in result.report_summary
    assert "detail" in result.report_summary
    assert elapsed < 6.0, f"mirix health-check fail took {elapsed:.2f}s; must be <6s"


def test_mirix_reset_user_skills_handles_404(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """404 from /v1/skills/reset must be logged + swallowed (issue-06 §3)."""
    import urllib.error
    import urllib.request

    def _fake_urlopen(*a: Any, **kw: Any):  # noqa: ARG001
        raise urllib.error.HTTPError(
            url="http://x/v1/skills/reset",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    runner._mirix_reset_user_skills("http://127.0.0.1:8531", "user-xyz")
    out = capsys.readouterr().out
    assert "MIRIX /v1/skills/reset endpoint not available" in out
    assert "minting fresh user_id is sufficient" in out


def test_mirix_reset_user_skills_handles_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """2xx responses log success and return without raising."""
    import urllib.request

    class _R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a, **kw):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _R())
    runner._mirix_reset_user_skills("http://127.0.0.1:8531", "user-xyz")
    out = capsys.readouterr().out
    assert "MIRIX /v1/skills/reset OK" in out


def test_mirix_reset_user_skills_handles_connection_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Connection errors are also swallowed (best-effort contract)."""
    import urllib.error
    import urllib.request

    def _boom(*a: Any, **kw: Any):  # noqa: ARG001
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    runner._mirix_reset_user_skills("http://127.0.0.1:8531", "user-xyz")
    out = capsys.readouterr().out
    assert "MIRIX /v1/skills/reset failed" in out
    assert "continuing" in out


# ---------------------------------------------------------------------------
# Slice #6: rounds counting + wallclock estimator
# ---------------------------------------------------------------------------


def test_count_rounds_per_day_first_three_days_match_dataset() -> None:
    """The canonical dataset has 10/11/12 rounds for day01..day03 today.

    If you re-vendor the dataset and these counts change, this test will
    fail — that's intentional: the wall-time estimator hard-codes a
    seconds-per-round constant, so the rounds-per-day shape is part of
    our reproducibility contract.
    """
    counts = runner.count_rounds_per_day(3)
    assert counts == [10, 11, 12]


def test_count_rounds_per_day_zero_returns_all_days() -> None:
    counts = runner.count_rounds_per_day(0)
    # The dataset is 30 days; total rounds = 346 (sanity-checked at vendor-time).
    assert len(counts) == 30
    assert sum(counts) == 346


def test_count_rounds_per_day_rejects_negative() -> None:
    with pytest.raises(ValueError, match="n_days must be >= 0"):
        runner.count_rounds_per_day(-1)


def test_estimate_wallclock_seconds_uses_real_rounds() -> None:
    """For days=5 retry=3 with the 20s/60s default bounds we expect the
    exact integers below - they pin the bench-side contract."""
    expected_rounds = 10 + 11 + 12 + 10 + 13  # day01..day05
    total, lo, hi = runner.estimate_wallclock_seconds(5, 3)
    assert total == expected_rounds
    assert lo == expected_rounds * 3 * 20
    assert hi == expected_rounds * 3 * 60


def test_estimate_wallclock_seconds_floors_retry_at_one() -> None:
    """retry=0 would otherwise zero the estimate; we floor it at 1."""
    total, lo, hi = runner.estimate_wallclock_seconds(1, 0)
    assert total > 0
    assert lo > 0
    assert hi > 0


# ---------------------------------------------------------------------------
# Slice #6: extra_meta plumbing reaches run.meta.json
# ---------------------------------------------------------------------------


def test_extra_meta_lands_in_run_meta_json(tmp_path: Path) -> None:
    """Acceptance #5: extra_meta dict is merged into run.meta.json without
    overwriting canonical keys."""
    starter, _ = _fake_proxy_starter()
    stopper, _ = _fake_proxy_stopper()
    bench, _ = _fake_bench(rc=0)
    run_arm(
        arm="metaclaw",
        days=1,
        out_dir=tmp_path / "meta_extra",
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
        extra_meta={
            "estimated_rounds": 42,
            "estimated_wallclock_seconds_min": 600,
            "estimated_wallclock_seconds_max": 1800,
            "arm": "WONT_OVERWRITE",  # canonical key — extra_meta must lose
        },
    )
    meta = json.loads((tmp_path / "meta_extra" / "run.meta.json").read_text())
    assert meta["estimated_rounds"] == 42
    assert meta["estimated_wallclock_seconds_min"] == 600
    assert meta["estimated_wallclock_seconds_max"] == 1800
    # Canonical key not overwritten.
    assert meta["arm"] == "metaclaw"
