#!/usr/bin/env python3
"""
baseline_run.py - Run the MetaClaw benchmark without any MetaClaw proxy.

The agent communicates directly with the target LLM API (BENCHMARK_BASE_URL /
BENCHMARK_MODEL) using the plain openclaw agent, with no memory, skills, or RL
enhancements.  This establishes the baseline score for comparison.

Set BENCHMARK_BASE_URL, BENCHMARK_API_KEY, and BENCHMARK_MODEL before running.
See scripts/_env_arg_example.sh for a template.
"""

import json
import os
import pty
import select
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR    = Path(__file__).resolve().parent
_METACLAW_ROOT = Path(os.environ.get("METACLAW_ROOT") or _SCRIPT_DIR.parent.parent)


# ===================== Configuration =====================
# All paths are derived from METACLAW_ROOT automatically.
# Edit the tuneable parameters (BENCH_WORKERS, BENCH_COUNT) or set
# METACLAW_ROOT / METACLAW_BENCH_BIN in the environment to override defaults.
class cfg:
    LOG_FILE      = str(_METACLAW_ROOT / "benchmark" / "logs" / "baseline_run" / "bench_run.log")
    BENCH_BIN     = os.environ.get("METACLAW_BENCH_BIN", "metaclaw-bench")
    BENCH_INPUT   = os.environ.get(
        "METACLAW_BENCH_INPUT",
        str(_METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench" / "all_tests.json"),
    )
    BENCH_OUTPUT  = str(_METACLAW_ROOT / "benchmark" / "results")
    BENCH_WORKERS = 15   # -w  concurrent workers
    BENCH_COUNT   = 3    # -n  retries per failed question
    # Set METACLAW_API_KEY_SCRIPT to a shell script that exports BENCHMARK_*
    # variables. Leave unset (None) to rely on the current environment.
    API_KEY_SCRIPT = os.environ.get("METACLAW_API_KEY_SCRIPT")
# =========================================================


def resolve_log_path(log_file: str) -> Path:
    """Return a unique log path, appending _1/_2/... if the file already exists."""
    p = Path(log_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    n = 1
    while True:
        candidate = p.parent / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def load_env_from_shell(script_path: str) -> dict:
    """Source a shell script and return the resulting environment as a dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["bash", "-c",
             f"source {script_path} && "
             "python3 -c 'import os,json; json.dump(dict(os.environ),open(os.environ[\"__TMP_ENV\"],\"w\"))'"],
            env={**os.environ, "__TMP_ENV": tmp_path},
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to load env script: {script_path}")
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        os.unlink(tmp_path)


def run_command(cmd: list, log_path: Path, env: dict = None) -> int:
    """Run *cmd* via a pseudo-terminal so output is line-buffered in real time.

    Output is written to both the terminal and *log_path*.
    Returns the subprocess exit code.
    """
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    with open(log_path, "a", encoding="utf-8") as log_f:
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 1.0)
                if r:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    log_f.write(text)
                    log_f.flush()
                elif proc.poll() is not None:
                    break
            except OSError:
                break

    os.close(master_fd)
    proc.wait()
    return proc.returncode


def append_timing(log_path: Path, start: datetime, end: datetime):
    """Append a timing summary to the log file and print it to the terminal."""
    elapsed = (end - start).total_seconds()
    lines = [
        "",
        "----------------------------------------",
        "Done.",
        f"Start:   {start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"End:     {end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed: {elapsed:.3f}s",
        "========================================",
        "",
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("----------------------------------------")
    print(f"Done. Elapsed: {elapsed:.3f}s")
    print(f"Output logged to: {log_path}")


def main():
    os.system("clear")

    log_path = resolve_log_path(cfg.LOG_FILE)

    env = None
    if cfg.API_KEY_SCRIPT:
        env = load_env_from_shell(cfg.API_KEY_SCRIPT)

    start = datetime.now()

    run_cmd = [
        cfg.BENCH_BIN, "run",
        "-i", cfg.BENCH_INPUT,
        "-o", cfg.BENCH_OUTPUT,
        "-w", str(cfg.BENCH_WORKERS),
        "-n", str(cfg.BENCH_COUNT),
    ]
    run_command(run_cmd, log_path, env=env)

    end = datetime.now()
    append_timing(log_path, start, end)


if __name__ == "__main__":
    main()
