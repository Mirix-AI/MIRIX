#!/usr/bin/env python3
"""
buffer_memory_run.py - Run the benchmark with incremental per-turn memory ingestion.

Uses the buffer_turn / flush_session API path introduced in the memory-check branch:
  - After every round  : POST /buffer_turn  (sidecar auto-flushes every flush_every turns)
  - After every scene  : POST /flush_session(final=True)  (emits working_summary + clears state)
  - No skills, no RL.

This is the comparison target for memory_run.py (batch ingest).  Run both to
determine whether intra-scene incremental memory retrieval improves benchmark scores.

Set BENCHMARK_BASE_URL, BENCHMARK_API_KEY, and BENCHMARK_MODEL before running.
See scripts/_env_arg_example.sh for a template.
"""

import json
import os
import pty
import re
import select
import signal
import socket
import sys
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR    = Path(__file__).resolve().parent
_METACLAW_ROOT = Path(os.environ.get("METACLAW_ROOT") or _SCRIPT_DIR.parent.parent)


# ===================== Configuration =====================
class cfg:
    LOG_FILE      = str(_METACLAW_ROOT / "benchmark" / "logs" / "buffer_memory_run" / "bench_run.log")
    BENCH_BIN     = os.environ.get("METACLAW_BENCH_BIN", "metaclaw-bench")
    BENCH_INPUT   = os.environ.get(
        "METACLAW_BENCH_INPUT",
        str(_METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench" / "all_tests_metaclaw.json"),
    )
    BENCH_OUTPUT  = str(_METACLAW_ROOT / "benchmark" / "results")
    BENCH_COUNT   = 3    # -n  retries per failed question
    API_KEY_SCRIPT = os.environ.get("METACLAW_API_KEY_SCRIPT")
    PROXY_SCRIPT  = str(_SCRIPT_DIR / "proxy_run.py")
    PROXY_CONFIG  = str(_SCRIPT_DIR / "config" / "buffer-memory.yaml")
    # Per-run memory snapshots are stored under MEMORY_STORE_BASE/<timestamp>/.
    MEMORY_STORE_BASE = str(_METACLAW_ROOT / "benchmark" / "memory_runs")
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


def create_memory_run_dir() -> Path:
    """Create a timestamped subdirectory under MEMORY_STORE_BASE and write a
    benchmark-optimised policy.json preset."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.MEMORY_STORE_BASE) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[memory] Memory store for this run: {run_dir}")

    policy = {
        "version": 1,
        "retrieval_mode": "hybrid",
        "max_injected_units": 10,
        "max_injected_tokens": 1500,
        "keyword_weight": 1.0,
        "metadata_weight": 0.6,
        "importance_weight": 0.7,
        "recency_weight": 0.1,
        "recent_bonus_hours": 168,
        "type_boosts": {
            "semantic": 1.3,
            "preference": 1.2,
            "project_state": 1.1,
            "procedural_observation": 1.1,
            "working_summary": 1.0,
            "episodic": 0.7,
        },
        "notes": ["bench-optimized: high importance, low recency, semantic-boosted"],
    }
    policy_path = run_dir / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[memory] Policy preset written: {policy_path}")
    return run_dir


def write_proxy_config(env: dict, memory_dir: Path, port: int) -> str:
    """Expand ${VAR} placeholders in PROXY_CONFIG and override memory.dir / proxy.port."""
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
    data.setdefault("memory", {})["dir"] = str(memory_dir)
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
    print(f"[proxy] Starting proxy in buffer-memory mode (port={port})...")
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
    """Run *cmd* via a pseudo-terminal so output is line-buffered in real time."""
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


# ---------------------------------------------------------------------------
# Memory stats collection and report generation
# ---------------------------------------------------------------------------

def _api_get(path: str, port: int) -> dict | None:
    """GET a path on the local proxy and return parsed JSON, or None on error."""
    url = f"http://localhost:{port}{path}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[memory] WARNING: request to {path} failed: {e}")
        return None


def _api_post(path: str, port: int) -> dict | None:
    """POST to a path on the local proxy and return parsed JSON, or None on error."""
    url = f"http://localhost:{port}{path}"
    try:
        req = urllib.request.Request(
            url, data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[memory] WARNING: request to {path} failed: {e}")
        return None


def collect_memory_stats(port: int) -> dict:
    """Query the running proxy for memory statistics."""
    print("[memory] Collecting memory stats from proxy...")
    data = {}
    data["stats"]            = _api_get("/v1/memory/stats",            port) or {}
    data["summary"]          = _api_get("/v1/memory/summary",          port) or {}
    data["health"]           = _api_get("/v1/memory/health",           port) or {}
    data["operator_report"]  = _api_get("/v1/memory/operator-report",  port) or {}
    data["feedback_analysis"]= _api_get("/v1/memory/feedback-analysis",port) or {}
    print("[memory] Memory stats collected.")
    return data


def write_memory_report(mem_data: dict, output_dir: str, elapsed_seconds: float):
    """Write memory_report.md and memory_report.json to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "memory_report.md"
    json_path   = out / "memory_report.json"

    stats    = mem_data.get("stats", {})
    summary  = mem_data.get("summary", {})
    health   = mem_data.get("health", {})
    operator = mem_data.get("operator_report", {})
    feedback = mem_data.get("feedback_analysis", {})

    lines = [
        "# Buffer-Memory Benchmark Report (Incremental Ingestion)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Benchmark duration: {elapsed_seconds:.1f}s ({elapsed_seconds / 60:.1f}min)",
        "",
        "## Ingestion Mode",
        "",
        "- **Path**: incremental (`buffer_turn` per round + `flush_session` per scene)",
        "- **flush_every**: 3  (sidecar auto-flushes every 3 buffered turns)",
        "- **Benefit**: memories from early rounds are available within the same scene",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Active memories | {stats.get('active', 0)} |",
        f"| Total memories  | {stats.get('total', 0)} |",
        f"| Scope           | {stats.get('scope_id', 'default')} |",
        "",
    ]

    active_by_type = stats.get("active_by_type", {})
    if active_by_type:
        lines += ["## Memory Type Distribution", "", "| Type | Count | Ratio |", "|------|-------|-------|"]
        total_active = stats.get("active", 1) or 1
        for mtype, count in sorted(active_by_type.items(), key=lambda x: -x[1]):
            lines.append(f"| {mtype} | {count} | {count / total_active:.1%} |")
        lines.append("")

    for section, title in [
        (health,   "Health Check"),
        (summary,  "System Summary"),
        (operator, "Operator Report"),
        (feedback, "Feedback Analysis"),
    ]:
        if not section or not isinstance(section, dict):
            continue
        lines += [f"## {title}", ""]
        for k, v in section.items():
            if isinstance(v, (str, int, float, bool)):
                lines.append(f"- **{k}**: {v}")
            elif isinstance(v, dict):
                lines.append(f"- **{k}**:")
                for sk, sv in v.items():
                    lines.append(f"  - {sk}: {sv}")
            elif isinstance(v, list) and len(v) <= 20:
                lines.append(f"- **{k}**:")
                for item in v:
                    if isinstance(item, dict):
                        lines.append(f"  - {json.dumps(item, ensure_ascii=False)[:200]}")
                    else:
                        lines.append(f"  - {item}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[memory] Written: {report_path}")

    json_path.write_text(json.dumps(mem_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[memory] Written: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.system("clear")

    log_path = resolve_log_path(cfg.LOG_FILE)

    env = None
    if cfg.API_KEY_SCRIPT:
        env = load_env_from_shell(cfg.API_KEY_SCRIPT)

    port = find_free_port()
    print(f"[port] Allocated free port: {port}")

    memory_dir = create_memory_run_dir()
    tmp_config_path = write_proxy_config(env or os.environ.copy(), memory_dir, port)

    proxy_proc = start_proxy(env or os.environ.copy(), tmp_config_path, port)

    bench_env = dict(env) if env else dict(os.environ)
    bench_env["METACLAW_PROXY_PORT"] = str(port)

    start = datetime.now()
    end = start
    try:
        run_cmd = [
            cfg.BENCH_BIN, "run",
            "-i", cfg.BENCH_INPUT,
            "-o", cfg.BENCH_OUTPUT,
            "-w", "1",
            "-n", str(cfg.BENCH_COUNT),
            "--buffer-turns",
            "--memory-proxy-port", str(port),
        ]
        run_command(run_cmd, log_path, env=bench_env)

        end = datetime.now()
        elapsed = (end - start).total_seconds()

        mem_data = collect_memory_stats(port)
        write_memory_report(mem_data, cfg.BENCH_OUTPUT, elapsed)

    finally:
        stop_proxy(proxy_proc)
        if os.path.isfile(tmp_config_path):
            os.unlink(tmp_config_path)
            print(f"[config] Cleaned up temporary config: {tmp_config_path}")
        end = datetime.now()

    append_timing(log_path, start, end)


if __name__ == "__main__":
    main()
