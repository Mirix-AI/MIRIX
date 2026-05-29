"""Offline end-to-end smoke tests for the MetaClaw eval plumbing.

These tests exercise the *runner* end-to-end against:

  * a stub LLM server (``stub_llm`` fixture) that returns deterministic
    OpenAI-format chat completions,
  * a stub MIRIX server (``stub_mirix`` fixture) that returns minimal but
    structurally-correct responses on the endpoints the adapters touch,
  * a stub ``proxy_starter`` that does NOT launch the real metaclaw proxy
    (and therefore does NOT spawn ``clawdbot`` / ``openclaw``),
  * a stub ``bench_runner`` that writes a deterministic ``report.json`` so
    the runner's parsing path is exercised end-to-end.

What they assert: the plumbing (env-var composition, dataset slicing, output
tree shape, ``run.meta.json`` shape, ``reports.md`` rendering for ``--arm
both``).  Accuracy numbers are deterministic constants from the stub bench —
the tests do NOT assert on benchmark correctness.

Quarantined behind ``@pytest.mark.integration``; the project's ``pytest.ini``
default-skips this marker.  Run explicitly:

    pytest -m integration evals/metaclaw/tests/test_smoke.py -v

CRITICAL OPERATIONAL NOTE: these tests run safely alongside a live MetaClaw
eval (which spawns its own real ``clawdbot``) because the stub
``proxy_starter`` and ``bench_runner`` never invoke ``openclaw`` /
``clawdbot`` / ``python -m src.cli`` / ``python -m metaclaw``.  Verify with
``ps aux | grep -E 'clawdbot|openclaw'`` after a smoke-test run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from evals.metaclaw.runner import RunResult, run_arm, run_both


# ---------------------------------------------------------------------------
# Stub subprocess hooks (offline — no clawdbot / openclaw / vendored bench)
# ---------------------------------------------------------------------------


def _make_stub_proxy_starter():
    """Return a proxy_starter that captures inputs and returns a fake Popen."""
    captured: list[dict] = []

    def _starter(yaml: Path, port: int, log: Path, env: dict):
        captured.append({"yaml": yaml, "port": port, "log": log, "env": dict(env)})
        # Drop a token line into the log so the runner's tail-on-failure path
        # has something benign to print if anything later trips.
        try:
            log.write_text("[stub] proxy not launched (smoke test)\n", encoding="utf-8")
        except OSError:
            pass
        proc = MagicMock()
        proc.pid = 99999
        proc.poll = MagicMock(return_value=None)
        return proc

    return _starter, captured


def _make_stub_proxy_stopper():
    stopped: list[Any] = []

    def _stopper(proc):
        stopped.append(proc)

    return _stopper, stopped


def _make_stub_bench_runner(rc: int = 0):
    """Return a bench_runner that writes a deterministic ``report.json``."""
    captured: dict = {}

    def _runner(
        tests_used: Path,
        out_dir: Path,
        env: dict,
        retry: int = 1,
        workers: int = 1,
        max_rounds: Optional[int] = None,
    ) -> int:
        captured["tests_used"] = tests_used
        captured["out_dir"] = out_dir
        captured["env"] = dict(env)
        captured["retry"] = retry
        captured["workers"] = workers
        captured["max_rounds"] = max_rounds
        # Mirror the real bench's per-day shape: one nested dir, one report.json.
        report_dir = out_dir / "infer_stub"
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "summary": {
                "total_questions": 4,
                "correct": 2,
                "accuracy": 0.5,
                "tokens": {
                    "agent": {"total_input": 100, "input": 80, "output": 20},
                },
            }
        }
        (report_dir / "report.json").write_text(json.dumps(report))
        return rc

    return _runner, captured


# ---------------------------------------------------------------------------
# The three required smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_metaclaw_arm_smoke(stub_llm: str, tmp_path: Path) -> None:
    """``--arm metaclaw --days 1`` runs against stub LLM + stub proxy/bench,
    produces ``report.json`` and a parseable :class:`RunResult`."""
    starter, captured_proxy = _make_stub_proxy_starter()
    stopper, stopped = _make_stub_proxy_stopper()
    bench, captured_bench = _make_stub_bench_runner(rc=0)

    out_dir = tmp_path / "metaclaw-run"
    result = run_arm(
        arm="metaclaw",
        days=1,
        out_dir=out_dir,
        max_rounds=2,
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
        extra_env={
            "OPENROUTER_API_KEY": "sk-stub",
            "BENCHMARK_API_KEY": "sk-stub",
            "BENCHMARK_BASE_URL": stub_llm + "/v1",
            "BENCHMARK_MODEL": "stub/model",
        },
    )

    assert isinstance(result, RunResult)
    assert result.exit_code == 0
    assert result.arm == "metaclaw"
    assert isinstance(result.accuracy, float)
    assert result.accuracy == pytest.approx(0.5)

    # report.json exists somewhere under bench_output.
    report = next((out_dir / "bench_output").rglob("report.json"), None)
    assert report is not None, "stub bench should have produced a report.json"

    # run.meta.json shape sanity.
    meta = json.loads((out_dir / "run.meta.json").read_text())
    assert meta["arm"] == "metaclaw"
    assert meta["exit_code"] == 0
    assert meta["accuracy"] == pytest.approx(0.5)
    assert meta["base_url"] == stub_llm + "/v1"

    # Proxy stub was called, proxy stopper fired in finally.
    assert len(captured_proxy) == 1
    assert len(stopped) == 1

    # Bench env contains BENCHMARK_* overrides pointing at the stub LLM.
    assert captured_bench["env"]["BENCHMARK_BASE_URL"] == stub_llm + "/v1"
    assert captured_bench["env"]["BENCHMARK_API_KEY"] == "sk-stub"
    assert captured_bench["env"]["METACLAW_SKILLS_PROVIDER"] == "metaclaw"


@pytest.mark.integration
def test_mirix_arm_smoke(stub_llm: str, stub_mirix: str, tmp_path: Path) -> None:
    """``--arm mirix --days 1`` runs against stub LLM + stub MIRIX, produces
    ``report.json`` and routes the runner through the MIRIX prelude (health
    probe, user creation, skills reset) against the stub server."""
    starter, captured_proxy = _make_stub_proxy_starter()
    stopper, stopped = _make_stub_proxy_stopper()
    bench, captured_bench = _make_stub_bench_runner(rc=0)

    out_dir = tmp_path / "mirix-run"
    result = run_arm(
        arm="mirix",
        days=1,
        out_dir=out_dir,
        max_rounds=2,
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
        mirix_url=stub_mirix,
        extra_env={
            "OPENROUTER_API_KEY": "sk-stub",
            "BENCHMARK_API_KEY": "sk-stub",
            "BENCHMARK_BASE_URL": stub_llm + "/v1",
            "BENCHMARK_MODEL": "stub/model",
        },
    )

    assert isinstance(result, RunResult)
    assert result.exit_code == 0
    assert result.arm == "mirix"
    assert result.accuracy == pytest.approx(0.5)

    report = next((out_dir / "bench_output").rglob("report.json"), None)
    assert report is not None, "stub bench should have produced a report.json"

    # MIRIX prelude propagated correct env vars to the proxy / bench.
    bench_env = captured_bench["env"]
    assert bench_env["METACLAW_SKILLS_PROVIDER"] == "mirix"
    assert bench_env["METACLAW_EVOLVER_PROVIDER"] == "mirix"
    assert bench_env["METACLAW_MIRIX_BASE_URL"] == stub_mirix
    assert bench_env["METACLAW_MIRIX_USER_ID"].startswith("eval-metaclaw-")

    # run.meta.json records the mirix-arm fields.
    meta = json.loads((out_dir / "run.meta.json").read_text())
    assert meta["arm"] == "mirix"
    assert meta["mirix_url"] == stub_mirix
    assert meta["mirix_user_id"] is not None
    assert meta["exit_code"] == 0

    # Proxy + stopper still ran.
    assert len(captured_proxy) == 1
    assert len(stopped) == 1


@pytest.mark.integration
def test_both_arm_smoke(stub_llm: str, stub_mirix: str, tmp_path: Path) -> None:
    """``--arm both --days 1`` runs metaclaw then mirix against the SAME sliced
    dataset and produces a combined ``reports.md``."""
    starter, captured_proxy = _make_stub_proxy_starter()
    stopper, _stopped = _make_stub_proxy_stopper()
    bench, _captured_bench = _make_stub_bench_runner(rc=0)

    out_dir = tmp_path / "both-run"
    metaclaw_result, mirix_result = run_both(
        days=1,
        out_dir=out_dir,
        max_rounds=2,
        proxy_starter=starter,
        proxy_stopper=stopper,
        bench_runner=bench,
        mirix_url=stub_mirix,
        extra_env={
            "OPENROUTER_API_KEY": "sk-stub",
            "BENCHMARK_API_KEY": "sk-stub",
            "BENCHMARK_BASE_URL": stub_llm + "/v1",
            "BENCHMARK_MODEL": "stub/model",
        },
    )

    # Both arms succeeded with the stub bench rc=0.
    assert metaclaw_result.exit_code == 0
    assert mirix_result.exit_code == 0
    assert metaclaw_result.arm == "metaclaw"
    assert mirix_result.arm == "mirix"

    # reports.md exists at the parent run directory.
    reports_md = out_dir / "reports.md"
    assert reports_md.exists(), "--arm both must produce reports.md at parent"
    text = reports_md.read_text()
    assert "metaclaw" in text.lower()
    assert "mirix" in text.lower()

    # The shared sliced dataset lives at parent/all_tests_used.json.
    shared = out_dir / "all_tests_used.json"
    assert shared.exists()
    # Both arms each have their own per-arm dir with a copy of the slice.
    arm_dirs = [d for d in out_dir.iterdir() if d.is_dir()]
    assert any(d.name.startswith("metaclaw-") for d in arm_dirs)
    assert any(d.name.startswith("mirix-") for d in arm_dirs)

    # Each arm produced a report.json.
    for arm_dir in arm_dirs:
        report = next((arm_dir / "bench_output").rglob("report.json"), None)
        assert report is not None, f"missing report.json under {arm_dir}"

    # Proxy started twice (once per arm) — confirms run_both invoked run_arm twice.
    assert len(captured_proxy) == 2


# ---------------------------------------------------------------------------
# Belt-and-suspenders: confirm no clawdbot/openclaw process was spawned by
# this test module.  This is a soft check (psutil-free) using `pgrep` if
# available; absence of pgrep is not a failure.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_clawdbot_spawned_by_smoke_suite() -> None:
    """Sanity belt: after the smoke fixtures run, no clawdbot/openclaw process
    should be attributable to this pytest session.

    This is intentionally a soft check — it only fails when a child process
    whose ppid matches the test runner's pid is found running clawdbot or
    openclaw.  We deliberately do NOT scan the whole machine because a
    parallel live eval may legitimately be running its own real clawdbot.
    """
    import subprocess

    my_pid = os.getpid()
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,ppid=,comm=", "-A"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        pytest.skip("ps not available / not permitted; cannot verify spawn check")

    offenders: list[str] = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            _, ppid_s, comm = parts[0], parts[1], parts[2]
            ppid = int(ppid_s)
        except ValueError:
            continue
        if ppid != my_pid:
            continue
        lc = comm.lower()
        if "clawdbot" in lc or "openclaw" in lc:
            offenders.append(line.strip())

    assert not offenders, (
        f"smoke suite must not spawn clawdbot/openclaw children; found: {offenders}"
    )
