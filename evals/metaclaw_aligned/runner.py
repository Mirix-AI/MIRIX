"""Driver for arm A / arm B / arm C / arm D under paper's exact harness.

PRD D4: four arms over paper's metaclaw-bench infer pipeline. The only
variable across arms is the skill backend selection at proxy startup
(PRD D6 single injection point via METACLAW_SKILLS_PROVIDER env var).

Usage:

    # arm A: MIRIX-skill-evolve backend
    python -m evals.metaclaw_aligned.runner --arm A --bench small --max-tests 3

    # arm B: MIRIX-legacy backend
    python -m evals.metaclaw_aligned.runner --arm B --bench small --max-tests 3

    # arm C: paper baseline (no proxy)
    python -m evals.metaclaw_aligned.runner --arm C --bench small --max-tests 3

    # arm D: paper-native skills (no MIRIX)
    python -m evals.metaclaw_aligned.runner --arm D --bench small --max-tests 3

Required env (PRD D10):
    BENCHMARK_BASE_URL  e.g. https://openrouter.ai/api/v1
    BENCHMARK_API_KEY   OpenRouter / OpenAI key
    BENCHMARK_MODEL     openai/gpt-5.2  (paper's published model — pinned)

Output structure:
    evals/metaclaw_aligned/runs/<arm>-<bench>-<timestamp>/
        all_tests_used.json          # the (possibly truncated) test list
        infer/                        # paper metaclaw-bench infer output
            <test_id>/<group>/<round>/infer_result.json
        proxy.log                     # proxy stdout/stderr (arms A/B/D)
        run.meta.json                 # arm config, env, timing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METACLAW_ROOT = REPO_ROOT / "third_party" / "MetaClaw"


def _node_v22_path() -> str:
    """Return the node v22.x bin dir from nvm (openclaw requires v22.16+).
    Returns empty string if v22 not found — caller falls back to env PATH."""
    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    if not nvm_dir.exists():
        return ""
    v22_dirs = sorted(p for p in nvm_dir.iterdir() if p.name.startswith("v22."))
    if not v22_dirs:
        return ""
    return str(v22_dirs[-1] / "bin")  # latest v22.x

# Two bench JSONs per dataset:
#   *_metaclaw.json points at metaclaw.json openclaw_cfg (uses METACLAW_PROXY_PORT)
#   *.json (no metaclaw suffix) points at openclaw.json (talks directly to BENCHMARK_BASE_URL)
# arms A/B/D use the proxy-routed config; arm C bypasses.
BENCH_INPUTS = {
    ("small", True):  METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench-small" / "all_tests_metaclaw.json",
    ("small", False): METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench-small" / "all_tests.json",
    ("full",  True):  METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench"       / "all_tests_metaclaw.json",
    ("full",  False): METACLAW_ROOT / "benchmark" / "data" / "metaclaw-bench"       / "all_tests.json",
}
# arm D needs paper's pre-built skill bank
PAPER_NATIVE_SKILLS = METACLAW_ROOT / "memory_data" / "skills"

ARM_PROFILES = {
    "A": {"name": "armA-mirix-skills",  "needs_proxy": True,  "provider": "mirix-skill-evolve",
          "default_mirix_url": "http://127.0.0.1:8531"},
    "B": {"name": "armB-mirix-legacy",  "needs_proxy": True,  "provider": "mirix-legacy",
          "default_mirix_url": "http://127.0.0.1:8532"},
    "C": {"name": "armC-baseline",      "needs_proxy": False, "provider": None,
          "default_mirix_url": None},
    "D": {"name": "armD-paper-native",  "needs_proxy": True,  "provider": "paper",
          "default_mirix_url": None},
}


def _check_required_env() -> None:
    missing = [v for v in ("BENCHMARK_BASE_URL", "BENCHMARK_API_KEY", "BENCHMARK_MODEL")
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"Missing required env vars: {missing}. "
            f"PRD D10 requires BENCHMARK_MODEL=openai/gpt-5.2 against an "
            f"OpenAI-compatible endpoint."
        )


def _truncate_tests(src: Path, dst: Path, max_tests: int) -> int:
    """Copy *src* JSON to *dst*, keeping only the first *max_tests* tests.
    Returns the actual count kept. 0 → keep all."""
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    full_n = len(data.get("test", []))
    if max_tests > 0:
        data["test"] = data["test"][:max_tests]
    kept = len(data["test"])
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[runner] truncated tests {full_n} → {kept} ({src.name} → {dst})", flush=True)
    return kept


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_proxy_config(yaml_path: Path, skills_dir: str, port: int) -> None:
    """Write a minimal metaclaw config.yaml. We use mode=skills_only for arms
    A, B, D — the proxy still loads, but its SkillManager class is the one
    selected by METACLAW_SKILLS_PROVIDER env (PRD D6).

    metaclaw's config loader does NOT auto-substitute env vars at load time;
    we expand them here. paper's `skills_only_run.py` does the same expansion
    in write_proxy_config (see benchmark/scripts/skills_only_run.py).
    """
    base = os.environ.get("BENCHMARK_BASE_URL", "")
    key  = os.environ.get("BENCHMARK_API_KEY",  "")
    model = os.environ.get("BENCHMARK_MODEL",   "")
    yaml_path.write_text(
        f"""mode: skills_only
proxy:
  host: 127.0.0.1
  port: {port}
llm:
  api_base: {base}
  api_key:  {key}
  model_id: {model}
  provider: custom
skills:
  enabled: true
  dir: {skills_dir}
  auto_evolve: true
  retrieval_mode: template
  top_k: 6
  task_specific_top_k: 10
memory:
  enabled: false
rl:
  enabled: false
scheduler:
  enabled: false
  calendar:
    enabled: false
""",
        encoding="utf-8",
    )


def _start_proxy(config_path: Path, port: int, log_path: Path, env: dict) -> subprocess.Popen:
    print(f"[proxy] starting metaclaw on :{port}, config={config_path}", flush=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["metaclaw", "start", "--config", str(config_path)],
        stdout=log_fh, stderr=log_fh, env=env,
        start_new_session=True,
    )
    # Wait for /healthz
    deadline = time.time() + 120
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        if proc.poll() is not None:
            log_fh.close()
            with open(log_path, encoding="utf-8") as f:
                tail = f.read()[-2000:]
            raise RuntimeError(f"proxy exited prematurely (rc={proc.returncode})\n--- log tail ---\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"[proxy] healthy at {url}", flush=True)
                    return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    log_fh.close()
    raise RuntimeError(f"proxy /healthz never came up within 120s (log: {log_path})")


def _stop_proxy(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    print("[proxy] stopping", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=list(ARM_PROFILES), required=True,
                   help="A=MIRIX-skill-evolve, B=MIRIX-legacy, C=paper baseline (no proxy), "
                        "D=paper-native skills.")
    p.add_argument("--bench", choices=["small", "full"], default="small")
    p.add_argument("--max-tests", type=int, default=0,
                   help="Cap tests (0 = all). Gating uses --bench small --max-tests 3.")
    p.add_argument("--mirix-base-url", default=None,
                   help="Override MIRIX server URL (arms A/B only).")
    p.add_argument("--mirix-user-id", default="eval-metaclaw-aligned-gating",
                   help="MIRIX user_id partition (arms A/B only).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--retry", type=int, default=0,
                   help="Per-question retry count (paper's -n flag).")
    args = p.parse_args()

    _check_required_env()
    profile = ARM_PROFILES[args.arm]

    bench_input = BENCH_INPUTS[(args.bench, profile["needs_proxy"])]
    if not bench_input.exists():
        sys.exit(f"bench input not found: {bench_input}")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "evals" / "metaclaw_aligned" / "runs" / f"{profile['name']}-{args.bench}-{ts}"
    )
    (out / "infer").mkdir(parents=True, exist_ok=True)

    # 1. Truncate / copy tests file.
    tests_used = out / "all_tests_used.json"
    kept = _truncate_tests(bench_input, tests_used, args.max_tests)
    if kept == 0:
        sys.exit("no tests after truncate; bench input may be empty")

    # 2. Write run meta.
    meta = {
        "arm": profile["name"],
        "provider": profile["provider"],
        "bench": args.bench,
        "max_tests": args.max_tests,
        "tests_used_count": kept,
        "started_at": ts,
        "model": os.environ.get("BENCHMARK_MODEL"),
        "mirix_base_url": args.mirix_base_url or profile.get("default_mirix_url"),
        "mirix_user_id": args.mirix_user_id if profile["needs_proxy"] and profile["provider"] != "paper" else None,
    }
    (out / "run.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 3. Start proxy if needed.
    proxy_proc = None
    proxy_port = None
    cleanup_paths = []

    try:
        if profile["needs_proxy"]:
            proxy_port = _find_free_port()
            yaml_path = Path(tempfile.mkstemp(prefix="metaclaw_cfg_", suffix=".yaml")[1])
            cleanup_paths.append(yaml_path)

            if profile["provider"] == "paper":
                # arm D: paper's SkillManager.add_skill writes new SKILL.md
                # files into the configured skills_dir during evolve. If we
                # mounted PAPER_NATIVE_SKILLS directly, this run would
                # mutate paper's reference bank — every subsequent arm-D
                # run would start from a different (contaminated) state.
                # Paper's own `skills_only_run.py` solves this by copying
                # to a tempdir and cleaning up. We do the same (codex parity
                # review confound #1).
                if not PAPER_NATIVE_SKILLS.exists():
                    sys.exit(f"arm D requires paper skills at {PAPER_NATIVE_SKILLS}, not found")
                skills_dir = tempfile.mkdtemp(prefix="metaclaw_paper_native_skills_")
                shutil.copytree(PAPER_NATIVE_SKILLS, skills_dir, dirs_exist_ok=True)
                cleanup_paths.append(Path(skills_dir))
                print(f"[runner] arm D skills isolated: {PAPER_NATIVE_SKILLS} -> {skills_dir}",
                      flush=True)
            else:
                # arms A, B: empty skill dir (MIRIX adapter ignores it)
                skills_dir = tempfile.mkdtemp(prefix="metaclaw_empty_skills_")
                cleanup_paths.append(Path(skills_dir))

            _write_proxy_config(yaml_path, skills_dir, proxy_port)

            proxy_env = {
                **os.environ,
                "METACLAW_ROOT": str(METACLAW_ROOT),
                # Propagate REPO_ROOT to subprocess Python path so the forked
                # metaclaw launcher can import evals.metaclaw_aligned.*
                "PYTHONPATH": f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
            }
            if profile["provider"] in ("mirix-skill-evolve", "mirix-legacy"):
                proxy_env["METACLAW_SKILLS_PROVIDER"] = profile["provider"]
                proxy_env["METACLAW_MIRIX_BASE_URL"] = (
                    args.mirix_base_url or profile["default_mirix_url"]
                )
                proxy_env["METACLAW_MIRIX_USER_ID"] = args.mirix_user_id
            else:
                proxy_env["METACLAW_SKILLS_PROVIDER"] = "paper"

            proxy_proc = _start_proxy(yaml_path, proxy_port, out / "proxy.log", proxy_env)
            print(f"[runner] proxy up on :{proxy_port}", flush=True)

        # 4. Run metaclaw-bench infer (and run = infer + scoring + report).
        # arm C bypasses proxy: agent talks directly to BENCHMARK_BASE_URL.
        bench_env = {**os.environ}

        # openclaw requires Node v22.16+; prepend the nvm v22 bin so the
        # `#!/usr/bin/env node` shebang resolves correctly even when the
        # parent shell has v20 on PATH.
        v22_bin = _node_v22_path()
        if v22_bin:
            bench_env["PATH"] = v22_bin + ":" + bench_env.get("PATH", "")

        if profile["needs_proxy"]:
            # Paper's openclaw_cfg/metaclaw.json substitutes ${METACLAW_PROXY_PORT}.
            # We point it at the proxy we just started; openclaw subprocess will
            # then send completions to http://127.0.0.1:{port}/v1.
            bench_env["METACLAW_PROXY_PORT"] = str(proxy_port)

        # Use `run` subcommand: infer → scoring → report.
        cmd = [
            "metaclaw-bench", "run",
            "-i", str(tests_used),
            "-o", str(out / "infer"),
            "-w", "1",
            "-n", str(args.retry),
        ]
        print(f"[runner] cmd: {' '.join(cmd)}", flush=True)
        start = time.time()
        rc = subprocess.call(cmd, env=bench_env)
        elapsed = time.time() - start

        meta["exit_code"] = rc
        meta["wall_seconds"] = round(elapsed, 1)
        meta["finished_at"] = datetime.now().strftime("%Y%m%d-%H%M%S")
        (out / "run.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[runner] {profile['name']} exit={rc} in {elapsed:.1f}s", flush=True)
        return rc

    finally:
        if proxy_proc is not None:
            _stop_proxy(proxy_proc)
        for p_ in cleanup_paths:
            if isinstance(p_, Path) and p_.is_file():
                p_.unlink(missing_ok=True)
            elif isinstance(p_, Path) and p_.is_dir():
                shutil.rmtree(p_, ignore_errors=True)
            elif isinstance(p_, str) and os.path.isdir(p_):
                shutil.rmtree(p_, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
