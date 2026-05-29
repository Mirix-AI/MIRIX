#!/usr/bin/env python3
"""
proxy_passthrough_run.py - Run the benchmark through the MetaClaw proxy with all
enhancements disabled (no skills, no memory, no RL).

Unlike baseline_run.py (which bypasses the proxy entirely), this script starts a
MetaClaw proxy in passthrough mode and routes agent requests through it.  Use
this to validate the proxy plumbing or measure its raw overhead without any
enhancement features active.

Set BENCHMARK_BASE_URL, BENCHMARK_API_KEY, and BENCHMARK_MODEL before running.
See scripts/_env_arg_example.sh for a template.
"""

import json
import os
import pty
import re
import select
import shutil
import signal
import socket
import sys
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR    = Path(__file__).resolve().parent
_METACLAW_ROOT = Path(os.environ.get("METACLAW_ROOT") or _SCRIPT_DIR.parent.parent)


# ===================== Configuration =====================
# All paths are derived from METACLAW_ROOT automatically.
# Edit BENCH_WORKERS / BENCH_COUNT as needed, or set environment variables
# listed in docs/scripts.md to override defaults.
class cfg:
    LOG_FILE      = str(_METACLAW_ROOT / "benchmark" / "logs" / "proxy_passthrough_run" / "bench_run.log")
    BENCH_BIN     = os.environ.get("METACLAW_BENCH_BIN", "metaclaw-bench")
    BENCH_INPUT   = os.environ.get(
        "METACLAW_BENCH_INPUT",
        str(_METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench" / "all_tests_metaclaw.json"),
    )
    BENCH_OUTPUT  = str(_METACLAW_ROOT / "benchmark" / "results")
    BENCH_WORKERS = 15   # -w  concurrent workers
    BENCH_COUNT   = 3    # -n  retries per failed question
    API_KEY_SCRIPT = os.environ.get("METACLAW_API_KEY_SCRIPT")
    PROXY_SCRIPT  = str(_SCRIPT_DIR / "proxy_run.py")
    PROXY_CONFIG  = str(_SCRIPT_DIR / "config" / "dummy.yaml")
    # Skills dir is required by the proxy config even when skills are disabled.
    # Set METACLAW_SKILLS_DIR to point to your skills directory; defaults to
    # <METACLAW_ROOT>/memory_data/skills.
    ORIGINAL_SKILL_DIR = os.environ.get(
        "METACLAW_SKILLS_DIR", str(_METACLAW_ROOT / "memory_data" / "skills")
    )
# =========================================================


def find_free_port() -> int:
    """Return an available TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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


def write_proxy_config(env: dict, temp_skill_dir: str, port: int) -> str:
    """Expand ${VAR} placeholders in PROXY_CONFIG and override skills.dir / proxy.port.

    Writes the result to a temporary file and returns its path.
    """
    import yaml as _yaml

    with open(cfg.PROXY_CONFIG, encoding="utf-8") as f:
        content = f.read()

    def replace(m):
        var = m.group(1)
        val = env.get(var, "")
        if not val:
            print(f"[config] WARNING: env var {var} not set, substituting empty string")
        return val

    content = re.sub(r'\$\{(\w+)\}', replace, content)

    data = _yaml.safe_load(content) or {}
    data.setdefault("skills", {})["dir"] = temp_skill_dir
    data.setdefault("proxy", {})["port"] = port

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="metaclaw_cfg_",
        delete=False, encoding="utf-8",
    )
    _yaml.dump(data, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    print(f"[config] Temporary config written to {tmp.name} (port={port})")
    return tmp.name


def start_proxy(env: dict, config_path: str, port: int) -> subprocess.Popen:
    """Start PROXY_SCRIPT in the background and wait until /healthz returns 200."""
    print(f"[proxy] Starting proxy in passthrough mode (port={port})...")
    proxy_env = dict(env) if env else dict(os.environ)
    proxy_env["METACLAW_CONFIG_FILE"] = config_path
    proc = subprocess.Popen(
        [sys.executable, cfg.PROXY_SCRIPT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=proxy_env,
        start_new_session=True,
    )

    url = f"http://localhost:{port}/healthz"
    print("[proxy] Waiting for proxy to become ready...")
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"[proxy] Process exited unexpectedly (exit code {proc.returncode})"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print("[proxy] Proxy ready, starting benchmark...")
                    return proc
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError("[proxy] Timed out (120s): no health-check response")


def stop_proxy(proc: subprocess.Popen):
    """Send SIGTERM to the proxy process group; SIGKILL after 10s if needed."""
    if proc.poll() is not None:
        return
    print("[proxy] Stopping proxy...")
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
    print("[proxy] Proxy stopped.")


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

    # Allocate a free port for this proxy instance.
    port = find_free_port()
    print(f"[port] Allocated free port: {port}")

    # Copy ORIGINAL_SKILL_DIR to a temp location so each run starts from the
    # same initial skill state (required by the proxy config even though skills
    # are disabled in passthrough mode).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_skill_dir = str(Path(tempfile.gettempdir()) / f"metaclaw_skills_{ts}")
    shutil.copytree(cfg.ORIGINAL_SKILL_DIR, temp_skill_dir)
    print(f"[skills] Copied initial skill dir to: {temp_skill_dir}")

    # Write the proxy config with expanded env vars, overriding port and skill dir.
    tmp_config_path = write_proxy_config(env or os.environ.copy(), temp_skill_dir, port)

    # Start the proxy and wait for it to be ready.
    proxy_proc = start_proxy(env or os.environ.copy(), tmp_config_path, port)

    # Inject the dynamic port so the openclaw config can resolve ${METACLAW_PROXY_PORT}.
    bench_env = dict(env) if env else dict(os.environ)
    bench_env["METACLAW_PROXY_PORT"] = str(port)

    start = datetime.now()
    try:
        run_cmd = [
            cfg.BENCH_BIN, "run",
            "-i", cfg.BENCH_INPUT,
            "-o", cfg.BENCH_OUTPUT,
            "-w", str(cfg.BENCH_WORKERS),
            "-n", str(cfg.BENCH_COUNT),
        ]
        run_command(run_cmd, log_path, env=bench_env)

    finally:
        stop_proxy(proxy_proc)
        if os.path.isdir(temp_skill_dir):
            shutil.rmtree(temp_skill_dir)
            print(f"[skills] Cleaned up temporary skill dir: {temp_skill_dir}")
        if os.path.isfile(tmp_config_path):
            os.unlink(tmp_config_path)
            print(f"[config] Cleaned up temporary config: {tmp_config_path}")

    end = datetime.now()
    append_timing(log_path, start, end)


if __name__ == "__main__":
    main()
