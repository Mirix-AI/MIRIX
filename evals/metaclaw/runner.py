"""Runner for the MetaClaw benchmark.

This is slice #1 — only ``arm="metaclaw"`` is wired end-to-end.  The runner

1.  slices ``all_tests_metaclaw.json`` to N days,
2.  copies ``workspaces/shared/`` to a per-run isolated location,
3.  stitches a synthetic ``METACLAW_ROOT`` layout (symlinks) so the vendored
    dataset resolves the same way upstream's
    ``benchmark/data/metaclaw-bench/...`` paths do,
4.  writes a ``skills_only`` proxy YAML and launches the vendored MetaClaw
    proxy in a fresh process group,
5.  waits for the proxy ``/healthz`` to come up,
6.  invokes the vendored bench (``python -m src.cli run …``) against the
    truncated dataset, and
7.  parses ``report.json`` into a :class:`RunResult`.

All subprocess + tempdir cleanup goes through ``try/finally`` and the proxy
is killed by process group so no orphaned uvicorns survive a crash.

MIRIX is exposed only through the production memory path: ``--arm mirix-generic``
ingests turns via MIRIX's generic memory API and retrieves procedural memory via
the unified search API.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .dataset_slice import slice_tests
from .mirix_adapters.generic_adapter import DEFAULT_EXPECTED_TRIGGER_SESSIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_METACLAW_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = EVALS_METACLAW_ROOT / "vendor"
VENDOR_METACLAW_PKG_PARENT = VENDOR_ROOT  # contains the `metaclaw/` package
VENDOR_BENCH_DIR = (
    VENDOR_ROOT / "benchmark"
)  # contains `src/` and `openclaw_customize/`
DATA_DIR = EVALS_METACLAW_ROOT / "data"
RUNS_DIR = EVALS_METACLAW_ROOT / "runs"
METACLAW_VERSION_FILE = EVALS_METACLAW_ROOT / "METACLAW_VERSION"

PROXY_HEALTH_TIMEOUT_S = 120
PROXY_STOP_TIMEOUT_S = 10
DEFAULT_BENCH_WORKERS = 1
DEFAULT_BENCH_RETRY = 3

# MIRIX server defaults — used by the ``mirix`` arm (slice #3+).
DEFAULT_MIRIX_BASE_URL = "http://127.0.0.1:8531"
MIRIX_HEALTH_TIMEOUT_S = 5
MIRIX_CLIENT_ID = "client-00000000-0000-4000-8000-000000000000"
MIRIX_ORG_ID = "org-00000000-0000-4000-8000-000000000000"
# Initializing a meta agent spins up 6 sub-agents server-side, which can take a
# while on a cold DB — give it a generous ceiling (mirrors init_meta_agent.py).
MIRIX_META_INIT_TIMEOUT_S = 180

# Env-var contract for the OpenAI-compatible upstream the proxy forwards to.
# We read OPENROUTER_API_KEY out of the user's .env (via python-dotenv) and
# remap to MetaClaw's BENCHMARK_* names so the vendored bench config picks
# them up unchanged.
DEFAULT_BENCHMARK_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_BENCHMARK_MODEL = "openai/gpt-5.2"


# ---------------------------------------------------------------------------
# Arm taxonomy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """How one arm configures the proxy + bench. Resolved by :func:`_resolve_arm`."""

    # The skills/evolver provider env value (`metaclaw` | `mirix`); None when
    # skills are off entirely (no-skills floor).
    skills_provider: Optional[str]
    # Whether this arm needs the MIRIX server prelude (health-check + user mint).
    needs_mirix: bool
    skills_enabled: bool
    auto_evolve: bool
    # "raw_transcript" for native MetaClaw, or "generic_memory" for MIRIX.
    evolution_mode: str
    # Pass the per-round memory-ingest callback to the bench.
    memory_rounds: bool
    # Human label for run.meta.json / logs.
    label: str


# Canonical arm names.
ARM_METACLAW = "metaclaw"  # native MetaClaw skill backend (cross-system anchor)
ARM_NO_SKILLS = "no-skills"  # floor: skills disabled
ARM_NATIVE = "native"  # alias of metaclaw, for P1-10 naming clarity
# MIRIX generic production memory path: ingest each turn via /memory/add_sync and
# let MIRIX's own automatic procedural trigger evolve skills.
ARM_MIRIX_GENERIC = "mirix-generic"

_SINGLE_ARMS = (
    ARM_METACLAW,
    ARM_MIRIX_GENERIC,
    ARM_NO_SKILLS,
    ARM_NATIVE,
)


def _resolve_arm(arm: str) -> ArmSpec:
    """Map an arm name to its :class:`ArmSpec`.

    ``metaclaw``/``native`` use the vendored MetaClaw skill backend.
    ``mirix-generic`` uses MIRIX through the production memory interface.
    """
    if arm in (ARM_METACLAW, ARM_NATIVE):
        return ArmSpec(
            skills_provider="metaclaw",
            needs_mirix=False,
            skills_enabled=True,
            auto_evolve=True,
            evolution_mode="raw_transcript",
            memory_rounds=False,
            label="native (vendored MetaClaw skill backend)",
        )
    if arm == ARM_MIRIX_GENERIC:
        return ArmSpec(
            skills_provider="mirix",
            needs_mirix=True,
            skills_enabled=True,
            auto_evolve=True,
            evolution_mode="generic_memory",
            memory_rounds=True,
            label="mirix-generic (production memory path: /memory/add_sync + automatic procedural trigger)",
        )
    if arm == ARM_NO_SKILLS:
        return ArmSpec(
            skills_provider=None,
            needs_mirix=False,
            skills_enabled=False,
            auto_evolve=False,
            evolution_mode="raw_transcript",
            memory_rounds=False,
            label="no-skills floor",
        )
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """The externally observable outcome of a single arm run."""

    arm: str
    exit_code: int
    output_dir: Path
    accuracy: Optional[float] = None
    total_tokens: Optional[int] = None
    report_summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — public so tests can patch them via DI hooks below
# ---------------------------------------------------------------------------


def _read_vendor_sha() -> str:
    """Return the pinned upstream SHA from METACLAW_VERSION."""
    if not METACLAW_VERSION_FILE.exists():
        return ""
    for line in METACLAW_VERSION_FILE.read_text().splitlines():
        if line.startswith("metaclaw_sha="):
            return line.split("=", 1)[1].strip()
    return ""


def _node_v22_bin() -> str:
    """Return the latest node v22.x bin directory from nvm.

    OpenClaw 2026.5.18 requires Node >=22.16.  We prepend this to ``PATH`` for
    every subprocess so it resolves even if the parent shell has v20 active.
    """
    nvm = Path.home() / ".nvm" / "versions" / "node"
    if not nvm.exists():
        return ""
    candidates = sorted(p for p in nvm.iterdir() if p.name.startswith("v22."))
    if not candidates:
        return ""
    return str(candidates[-1] / "bin")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load_env_file() -> dict:
    """Read MIRIX's ``.env`` and return its mapping.  Empty dict on failure."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return {}
    raw = dotenv_values(env_path)
    return {k: v for k, v in raw.items() if v is not None}


def _resolve_benchmark_env(extra_env: Optional[dict] = None) -> dict:
    """Compose the ``BENCHMARK_*`` env triple, with .env taking precedence over
    the parent shell, and *extra_env* overriding both (used by tests)."""
    parent = dict(os.environ)
    dotenv = _load_env_file()
    overrides = extra_env or {}

    base_url = (
        overrides.get("BENCHMARK_BASE_URL")
        or dotenv.get("BENCHMARK_BASE_URL")
        or parent.get("BENCHMARK_BASE_URL")
        or DEFAULT_BENCHMARK_BASE_URL
    )
    model = (
        overrides.get("BENCHMARK_MODEL")
        or dotenv.get("BENCHMARK_MODEL")
        or parent.get("BENCHMARK_MODEL")
        or DEFAULT_BENCHMARK_MODEL
    )
    api_key = (
        overrides.get("BENCHMARK_API_KEY")
        or dotenv.get("BENCHMARK_API_KEY")
        or parent.get("BENCHMARK_API_KEY")
        or dotenv.get("OPENROUTER_API_KEY")
        or parent.get("OPENROUTER_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "No BENCHMARK_API_KEY / OPENROUTER_API_KEY found in environment "
            f"or {REPO_ROOT / '.env'!s}. The proxy cannot reach the upstream LLM."
        )
    return {
        "BENCHMARK_BASE_URL": base_url,
        "BENCHMARK_API_KEY": api_key,
        "BENCHMARK_MODEL": model,
    }


# ---------------------------------------------------------------------------
# MIRIX server prelude helpers (mirix arm only)
# ---------------------------------------------------------------------------


def _mirix_health_ok(base_url: str, timeout_s: float = MIRIX_HEALTH_TIMEOUT_S) -> bool:
    """Probe ``<base_url>/health``.  Returns True iff HTTP 200 inside *timeout_s*.

    Kept for back-compat with slice-3 callers; new code should prefer
    :func:`_mirix_health_diagnose` which returns a structured (ok, status, detail)
    triple so callers can format the fail-fast error message exactly per the
    issue-06 contract.
    """
    ok, _status, _detail = _mirix_health_diagnose(base_url, timeout_s)
    return ok


def _mirix_health_diagnose(
    base_url: str, timeout_s: float = MIRIX_HEALTH_TIMEOUT_S
) -> tuple[bool, Optional[int], str]:
    """Probe ``<base_url>/health`` and return ``(ok, status_code, detail)``.

    *ok* is True iff HTTP 200 within *timeout_s*.  *status_code* is the integer
    HTTP status when one was received (even non-2xx), or ``None`` for
    connection-level failures.  *detail* is a short human-readable string
    suitable for the runner's fail-fast error message.

    Total wall-time is bounded by *timeout_s* (default 5s) so the whole
    pre-flight stays well under the 6s budget required by the issue-06
    acceptance test.
    """
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            status = int(getattr(r, "status", 0) or 0)
            if status == 200:
                return True, 200, "ok"
            return False, status, f"unexpected HTTP {status}"
    except urllib.error.HTTPError as e:
        return False, int(getattr(e, "code", 0) or 0), f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, None, f"connection failed: {e.reason}"
    except (TimeoutError, socket.timeout) as e:
        return False, None, f"timeout after {timeout_s}s: {e}"
    except (ConnectionError, OSError) as e:
        return False, None, f"connection error: {e}"


def _format_mirix_unreachable(base_url: str, status: Optional[int], detail: str) -> str:
    """Build the multi-line fail-fast error message for an unreachable MIRIX.

    Format (issue-06 acceptance #1):

        ERROR: MIRIX server not reachable at <url>
          status: <code-or-"no response">
          detail: <detail>
        Start it with: python scripts/start_server.py --port 8531
    """
    status_str = str(status) if status is not None else "no response"
    return (
        f"ERROR: MIRIX server not reachable at {base_url}\n"
        f"  status: {status_str}\n"
        f"  detail: {detail}\n"
        f"Start it with: python scripts/start_server.py --port 8531\n"
    )


def count_rounds_per_day(n_days: int) -> list[int]:
    """Return the ``rounds[]`` count per day for the first *n_days* of the dataset.

    Reads ``data/eval/dayNN/questions.json`` directly.  ``n_days=0`` means
    "all 30 days".  Used by the CLI's wallclock-budget estimator and exposed
    here so tests can patch the data path via DATA_DIR.

    Missing ``questions.json`` files are skipped silently (the day simply
    contributes 0 rounds to the estimate).
    """
    if n_days < 0:
        raise ValueError(f"n_days must be >= 0, got {n_days}")
    eval_dir = DATA_DIR / "eval"
    if not eval_dir.exists():
        return []
    counts: list[int] = []
    # Enumerate dayNN/ in numeric order; n_days=0 means "all available days".
    day_dirs = sorted(
        p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("day")
    )
    if n_days:
        day_dirs = day_dirs[:n_days]
    for d in day_dirs:
        q = d / "questions.json"
        if not q.exists():
            counts.append(0)
            continue
        try:
            with q.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            rounds = data.get("rounds", []) if isinstance(data, dict) else []
            counts.append(len(rounds) if isinstance(rounds, list) else 0)
        except (OSError, json.JSONDecodeError):
            counts.append(0)
    return counts


def estimate_wallclock_seconds(
    days: int,
    retry: int,
    *,
    seconds_per_round_min: float = 20.0,
    seconds_per_round_max: float = 60.0,
) -> tuple[int, int, int]:
    """Estimate the wall-time budget for a run as ``(rounds, min_s, max_s)``.

    ``rounds = sum(count_rounds_per_day(days))``.

    ``min_s = rounds * retry * seconds_per_round_min`` (optimistic)
    ``max_s = rounds * retry * seconds_per_round_max`` (pessimistic)

    Both bounds are rounded to int.  ``retry`` is floored at 1 so the
    estimate is never zero when there is real work to do.
    """
    retry = max(int(retry), 1)
    total_rounds = sum(count_rounds_per_day(days))
    min_s = int(round(total_rounds * retry * seconds_per_round_min))
    max_s = int(round(total_rounds * retry * seconds_per_round_max))
    return total_rounds, min_s, max_s


def _mirix_create_or_get_user(base_url: str, user_id: str) -> Optional[str]:
    """POST /users/create_or_get with ``{user_id, name: user_id}``.

    Returns the server-resolved user id on success, ``None`` on failure.

    Failing loud here is intentional: an eval harness that reports a
    comparable accuracy number must not silently degrade to "MIRIX
    arm with no usable user".  The runner's mirix prelude turns a
    ``None`` return into rc=2 so the failure surfaces in
    run.meta.json instead of being buried in proxy.log warnings.
    (slice-3 follow-up to codex review finding #2)
    """
    url = base_url.rstrip("/") + "/users/create_or_get"
    body = json.dumps({"user_id": user_id, "name": user_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Client-Id": MIRIX_CLIENT_ID,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            if not (200 <= r.status < 300):
                print(
                    f"[runner] /users/create_or_get returned HTTP {r.status} for {user_id}",
                    flush=True,
                )
                return None
            payload = json.loads(r.read().decode("utf-8"))
            resolved = payload.get("id") if isinstance(payload, dict) else None
            if not resolved:
                print(
                    f"[runner] /users/create_or_get returned no 'id' field for "
                    f"{user_id}: {payload!r}",
                    flush=True,
                )
                return None
            print(
                f"[runner] minted MIRIX user_id={user_id} (server id={resolved})",
                flush=True,
            )
            return resolved
    except (urllib.error.URLError, ConnectionError, OSError, ValueError) as e:
        print(
            f"[runner] /users/create_or_get failed for {user_id}: {e}",
            flush=True,
        )
        return None


def _mirix_ensure_client_write_scope(
    base_url: str,
    write_scope: str = "admin",
    timeout_s: float = 10.0,
) -> bool:
    """Idempotently ensure ``MIRIX_CLIENT_ID`` has a non-null ``write_scope``.

    Why this is REQUIRED (not a nicety): the hardcoded eval client
    ``MIRIX_CLIENT_ID`` exists on the server but ships with
    ``write_scope=None`` (read-only). MIRIX gates meta-agent creation on a
    non-null write_scope: ``POST /agents/meta/initialize`` returns ``200 null``
    for a read-only client (rest_api.py:1972-1974) and creates NO agent. With no
    meta agent there is no procedural-memory sub-agent, so production ingestion
    and retrieval cannot work reliably: ``/memory/add_sync`` has no valid owner
    and ``GET /memory/search?memory_type=procedural`` cannot resolve the
    procedural agent. Without this step ``_mirix_ensure_meta_agent`` refuses to
    run rather than allowing a degenerate no-skills eval.

    We use ``PATCH /clients/{client_id}`` rather than ``POST
    /clients/create_or_get`` because create_or_get is a NO-OP for an
    already-existing client (rest_api.py:1618-1628) — it returns the existing
    read-only client UNCHANGED and never updates its write_scope. PATCH is the
    only path that actually mutates the existing client's write_scope. We fall
    back to create_or_get only if the client does not exist yet (fresh DB), where
    create_or_get DOES create it with the requested write_scope.

    ``write_scope="admin"`` is chosen deliberately: it is the dashboard/first-user
    convention for unrestricted access (rest_api.py:6897), it is non-null (the
    only property meta-agent creation actually checks, rest_api.py:1973), and it
    is self-consistent for the production memory pipeline: add_sync writes turns
    and retrieval reads procedural memory under the SAME client identity. Returns
    True on success, False on any failure (the caller turns False into rc=2 so a
    degenerate run cannot masquerade as healthy).
    """
    patch_url = base_url.rstrip("/") + f"/clients/{MIRIX_CLIENT_ID}"
    patch_body = json.dumps(
        {"id": MIRIX_CLIENT_ID, "write_scope": write_scope}
    ).encode("utf-8")
    patch_req = urllib.request.Request(
        patch_url,
        data=patch_body,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "X-Client-Id": MIRIX_CLIENT_ID,
            "X-Org-Id": MIRIX_ORG_ID,
        },
    )
    try:
        with urllib.request.urlopen(patch_req, timeout=timeout_s) as r:
            status = int(getattr(r, "status", 0) or 0)
            raw = r.read().decode("utf-8", "replace")
            if 200 <= status < 300:
                try:
                    payload = json.loads(raw) if raw.strip() else None
                except ValueError:
                    payload = None
                got_scope = (
                    payload.get("write_scope") if isinstance(payload, dict) else None
                )
                if got_scope:
                    print(
                        f"[runner] MIRIX client {MIRIX_CLIENT_ID} write_scope ensured "
                        f"(PATCH /clients -> HTTP {status}, write_scope={got_scope!r})",
                        flush=True,
                    )
                    return True
                print(
                    f"[runner] PATCH /clients/{MIRIX_CLIENT_ID} returned HTTP {status} "
                    f"but write_scope is still null (body={raw[:300]!r})",
                    flush=True,
                )
                return False
    except urllib.error.HTTPError as e:
        # 404 -> client doesn't exist yet (fresh DB); fall through to create_or_get.
        # Any other HTTPError on PATCH is a real failure.
        if e.code != 404:
            print(
                f"[runner] PATCH /clients/{MIRIX_CLIENT_ID} HTTP {e.code} {e.reason}: "
                f"{e.read().decode('utf-8', 'replace')[:300]}",
                flush=True,
            )
            return False
        print(
            f"[runner] PATCH /clients/{MIRIX_CLIENT_ID} -> 404 (client absent); "
            f"creating it via /clients/create_or_get",
            flush=True,
        )
    except (urllib.error.URLError, ConnectionError, OSError, ValueError) as e:
        print(
            f"[runner] PATCH /clients/{MIRIX_CLIENT_ID} failed: {e}",
            flush=True,
        )
        return False

    # Fallback: client absent -> create it WITH the write_scope in one shot.
    create_url = base_url.rstrip("/") + "/clients/create_or_get"
    create_body = json.dumps(
        {
            "client_id": MIRIX_CLIENT_ID,
            "org_id": MIRIX_ORG_ID,
            "write_scope": write_scope,
        }
    ).encode("utf-8")
    create_req = urllib.request.Request(
        create_url,
        data=create_body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Client-Id": MIRIX_CLIENT_ID,
            "X-Org-Id": MIRIX_ORG_ID,
        },
    )
    try:
        with urllib.request.urlopen(create_req, timeout=timeout_s) as r:
            status = int(getattr(r, "status", 0) or 0)
            raw = r.read().decode("utf-8", "replace")
            if not (200 <= status < 300):
                print(
                    f"[runner] /clients/create_or_get returned HTTP {status}: "
                    f"{raw[:300]}",
                    flush=True,
                )
                return False
            try:
                payload = json.loads(raw) if raw.strip() else None
            except ValueError:
                payload = None
            got_scope = (
                payload.get("write_scope") if isinstance(payload, dict) else None
            )
            if not got_scope:
                # create_or_get is a no-op if the client somehow already existed
                # read-only — surface that rather than silently proceeding.
                print(
                    f"[runner] /clients/create_or_get returned HTTP {status} but "
                    f"write_scope still null (body={raw[:300]!r}); the client likely "
                    f"pre-existed read-only and create_or_get does not update it",
                    flush=True,
                )
                return False
            print(
                f"[runner] MIRIX client {MIRIX_CLIENT_ID} created with "
                f"write_scope={got_scope!r} (POST /clients/create_or_get -> HTTP {status})",
                flush=True,
            )
            return True
    except urllib.error.HTTPError as e:
        print(
            f"[runner] /clients/create_or_get HTTP {e.code} {e.reason}: "
            f"{e.read().decode('utf-8', 'replace')[:300]}",
            flush=True,
        )
        return False
    except (urllib.error.URLError, ConnectionError, OSError, ValueError) as e:
        print(
            f"[runner] /clients/create_or_get failed: {e}",
            flush=True,
        )
        return False


def _mirix_ensure_meta_agent(
    base_url: str,
    bench_env: dict,
    timeout_s: float = MIRIX_META_INIT_TIMEOUT_S,
) -> bool:
    """Idempotently ensure a meta agent (with a procedural-memory sub-agent) exists.

    Production ``/memory/add_sync`` needs a meta agent to own the write, and
    procedural retrieval walks the existing agent tree to find the procedural
    child. Minting a fresh user is not enough: with no meta agent under
    ``MIRIX_CLIENT_ID`` ingestion/retrieval degenerates into a no-skills run.

    This POSTs ``/agents/meta/initialize`` under the SAME ``X-Client-Id`` the
    adapters use, so the agent it creates is exactly the one those endpoints
    resolve. The endpoint enforces "one meta agent per client", so a second call
    is a no-op.

    The LLM + embedding config mirrors ``init_meta_agent.py`` but is sourced from the
    runner's already-resolved bench env (``BENCHMARK_MODEL`` / ``BENCHMARK_BASE_URL`` /
    ``BENCHMARK_API_KEY``) so the eval's chosen model + key flow through unchanged
    instead of being hardcoded. Returns True on success, False on any failure (the
    caller decides whether a False is fatal).
    """
    config = {
        "llm_config": {
            "model": bench_env["BENCHMARK_MODEL"],
            # OpenRouter's /chat/completions is OpenAI-compatible and this MIRIX
            # branch has no dedicated openrouter chat client, so use the "openai"
            # endpoint type pointed at the bench base_url (same trick the agent
            # side uses via BENCHMARK_BASE_URL).
            "model_endpoint_type": "openai",
            "model_endpoint": bench_env["BENCHMARK_BASE_URL"],
            "context_window": 128000,
            "api_key": bench_env["BENCHMARK_API_KEY"],
        },
        "embedding_config": {
            "embedding_model": "gemini-embedding-001",
            "embedding_endpoint_type": "openrouter",
            "embedding_endpoint": bench_env["BENCHMARK_BASE_URL"],
            # Pad/truncate to MAX_EMBEDDING_DIM (4096) so vectors match the
            # VECTOR(4096) ORM columns (mirix/constants.py: MAX_EMBEDDING_DIM).
            "embedding_dim": 4096,
            "embedding_chunk_size": 2048,
            "api_key": bench_env["BENCHMARK_API_KEY"],
        },
    }
    body = json.dumps({"config": config, "update_agents": True}).encode("utf-8")
    url = base_url.rstrip("/") + "/agents/meta/initialize"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Client-Id": MIRIX_CLIENT_ID,
            "X-Org-Id": MIRIX_ORG_ID,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            status = int(getattr(r, "status", 0) or 0)
            raw = r.read().decode("utf-8", "replace")
            if not (200 <= status < 300):
                print(
                    f"[runner] /agents/meta/initialize returned HTTP {status}; "
                    f"memory ingest/retrieve may fail: {raw[:500]}",
                    flush=True,
                )
                return False
            # A 2xx is NECESSARY but not SUFFICIENT: the endpoint returns
            # ``200 null`` for a read-only client (no write_scope -> no agents),
            # which would still break ingest/retrieve.
            # Require a non-null agent object with an id so a degenerate run can't
            # masquerade as a healthy one (the caller turns a False into rc=2).
            try:
                payload = json.loads(raw) if raw.strip() else None
            except ValueError:
                payload = None
            agent_id = payload.get("id") if isinstance(payload, dict) else None
            if not agent_id:
                print(
                    "[runner] /agents/meta/initialize returned HTTP "
                    f"{status} but NO agent (body={raw[:300]!r}); the client likely "
                    f"has no write_scope -> memory ingest/retrieve will fail",
                    flush=True,
                )
                return False
            print(
                "[runner] MIRIX meta agent ensured "
                f"(POST /agents/meta/initialize -> HTTP {status}, agent_id={agent_id})",
                flush=True,
            )
            return True
    except urllib.error.HTTPError as e:
        print(
            f"[runner] /agents/meta/initialize HTTP {e.code} {e.reason}: "
            f"{e.read().decode('utf-8', 'replace')[:500]}",
            flush=True,
        )
        return False
    except (urllib.error.URLError, ConnectionError, OSError, ValueError) as e:
        print(
            f"[runner] /agents/meta/initialize failed: {e}",
            flush=True,
        )
        return False


# ---------------------------------------------------------------------------
# METACLAW_ROOT layout stitching
# ---------------------------------------------------------------------------


def _build_metaclaw_root(per_run_workspace: Path, scratch_root: Path) -> Path:
    """Materialise a synthetic ``METACLAW_ROOT`` under *scratch_root*.

    The vendored dataset's ``all_tests_metaclaw.json`` carries upstream paths
    like ``./benchmark/data/metaclaw-bench/openclaw_state``.  Rather than
    rewrite every entry (which would also require rewriting the
    ``openclaw_cfg/metaclaw.json`` interpolations), we materialise a tree of
    symlinks that satisfies those upstream-shaped relative paths against our
    vendored copies.

    Layout produced:

        <scratch_root>/
            benchmark/
                openclaw_customize/         -> evals/metaclaw/vendor/benchmark/openclaw_customize
                data/metaclaw-bench/
                    eval/                   -> evals/metaclaw/data/eval
                    openclaw_cfg/           -> evals/metaclaw/data/openclaw_cfg
                    openclaw_state/         -> evals/metaclaw/data/openclaw_state
                    workspaces/shared/      -> *per-run isolated copy*
    """
    bench_link = scratch_root / "benchmark"
    (bench_link / "data" / "metaclaw-bench" / "workspaces").mkdir(
        parents=True, exist_ok=True
    )

    # Top-level openclaw_customize plugin dir
    customize_link = bench_link / "openclaw_customize"
    if not customize_link.exists():
        customize_link.symlink_to(VENDOR_BENCH_DIR / "openclaw_customize")

    # 30-day dataset subdirs
    mcl_bench = bench_link / "data" / "metaclaw-bench"
    for name in ("eval", "openclaw_cfg", "openclaw_state"):
        link = mcl_bench / name
        if not link.exists():
            link.symlink_to(DATA_DIR / name)

    # Per-run isolated workspace (real copy, not symlink — bench mutates it).
    ws_dst = mcl_bench / "workspaces" / "shared"
    if ws_dst.exists() or ws_dst.is_symlink():
        if ws_dst.is_symlink() or ws_dst.is_file():
            ws_dst.unlink()
        else:
            shutil.rmtree(ws_dst)
    ws_dst.symlink_to(per_run_workspace)
    return scratch_root


# ---------------------------------------------------------------------------
# Proxy lifecycle
# ---------------------------------------------------------------------------


def _write_proxy_yaml(
    yaml_path: Path,
    skills_dir: Path,
    port: int,
    bench_env: dict,
    skill_top_k: int = 6,
    *,
    skills_enabled: bool = True,
    auto_evolve: bool = True,
    evolution_mode: str = "raw_transcript",
    evolution_every_n_rounds: int = 5,
) -> None:
    """Write a minimal ``skills_only`` proxy config.

    The proxy's launcher loads this YAML; the LLM credentials are baked into
    the YAML (MetaClaw's loader doesn't auto-substitute env vars).  We disable
    RL + scheduler + memory because slice #1 only exercises the skill pipeline.

    ``skill_top_k`` controls ``skills.top_k`` — the number of skills
    ``_inject_skills`` requests per turn (``self.config.skill_top_k``).  The
    metaclaw arm keeps paper's default 6 (its three-bucket retrieve still adds
    task-specific + mistakes on top); the mirix arm raises it so MIRIX's single
    flat bucket injects a comparable count (mirix only returns this many total).

      * ``skills_enabled``: False for the ``no-skills`` floor arm.
      * ``auto_evolve``: False to retrieve-only (no evolution at all).
      * ``evolution_mode``: ``"raw_transcript"`` for vendored MetaClaw or
        ``"generic_memory"`` for MIRIX production memory ingestion.
      * ``evolution_every_n_rounds``: retained for the vendored config shape;
        MIRIX production cadence lives on the server.
    """
    skill_lines = [
        "skills:",
        f"  enabled: {str(skills_enabled).lower()}",
        f"  dir: {skills_dir}",
        f"  auto_evolve: {str(auto_evolve).lower()}",
        "  retrieval_mode: template",
        f"  top_k: {skill_top_k}",
        "  task_specific_top_k: 10",
    ]
    if evolution_mode != "raw_transcript":
        skill_lines.append(f"  evolution_mode: {evolution_mode}")
        skill_lines.append(f"  evolution_every_n_rounds: {evolution_every_n_rounds}")
    contents = (
        "mode: skills_only\n"
        "proxy:\n"
        "  host: 127.0.0.1\n"
        f"  port: {port}\n"
        "llm:\n"
        "  provider: custom\n"
        "  auth_method: api_key\n"
        f"  api_base: {bench_env['BENCHMARK_BASE_URL']}\n"
        f"  api_key:  {bench_env['BENCHMARK_API_KEY']}\n"
        f"  model_id: {bench_env['BENCHMARK_MODEL']}\n" + "\n".join(skill_lines) + "\n"
        "memory:\n"
        "  enabled: false\n"
        "rl:\n"
        "  enabled: false\n"
        "scheduler:\n"
        "  enabled: false\n"
        "  calendar:\n"
        "    enabled: false\n"
        "wechat:\n"
        "  enabled: false\n"
    )
    yaml_path.write_text(contents, encoding="utf-8")


def _prewarm_openclaw() -> None:
    """Warm up the openclaw CLI (clawdbot) before starting the proxy.

    The first invocation of any ``openclaw config …`` sub-command after a
    fresh boot can take ~25s as ``clawdbot`` cold-starts; the proxy
    launcher applies a hard 15s timeout to its internal config calls, so
    a cold ``openclaw`` reliably trips that timeout.  Two cheap probe
    calls bring subsequent invocations under 10s.

    Failures are swallowed — if ``openclaw`` is missing entirely the
    proxy startup will surface the real error.  This is an opportunistic
    speed-up, not a correctness gate.
    """
    for i in range(2):
        try:
            subprocess.run(
                ["openclaw", "config", "get", "models"],
                timeout=60,
                capture_output=True,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            print(
                f"[runner] _prewarm_openclaw iter={i} skipped: {e}",
                flush=True,
            )
            return
    print("[runner] _prewarm_openclaw done", flush=True)


def _start_proxy(
    config_path: Path,
    port: int,
    log_path: Path,
    env: dict,
) -> subprocess.Popen:
    """Launch the vendored MetaClaw proxy via ``python -m metaclaw start``.

    Uses ``start_new_session=True`` so the proxy lives in its own process
    group; ``_stop_proxy`` kills the whole group on shutdown.
    """
    log_fh = open(log_path, "w", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "metaclaw",
        "start",
        "--config",
        str(config_path),
        "--port",
        str(port),
    ]
    print(f"[proxy] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        env=env,
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]  # keep handle alive

    deadline = time.time() + PROXY_HEALTH_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = _tail(log_path, 4096)
            raise RuntimeError(
                f"proxy exited prematurely (rc={proc.returncode})\n--- proxy.log tail ---\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    print(f"[proxy] healthy at {url}", flush=True)
                    return proc
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)

    _stop_proxy(proc)
    tail = _tail(log_path, 4096)
    raise RuntimeError(
        f"proxy /healthz never came up within {PROXY_HEALTH_TIMEOUT_S}s "
        f"(log: {log_path})\n--- proxy.log tail ---\n{tail}"
    )


def _stop_proxy(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        _close_proxy_log(proc)
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=PROXY_STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _close_proxy_log(proc)


def _close_proxy_log(proc: subprocess.Popen) -> None:
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _tail(path: Path, max_bytes: int) -> str:
    if not path.exists():
        return "(no log file)"
    size = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        return f.read()


# ---------------------------------------------------------------------------
# Bench invocation
# ---------------------------------------------------------------------------


def _run_bench(
    tests_used: Path,
    out_dir: Path,
    env: dict,
    retry: int = DEFAULT_BENCH_RETRY,
    workers: int = DEFAULT_BENCH_WORKERS,
    max_rounds: Optional[int] = None,  # noqa: ARG001 — slice #1 passes through; vendored bench has no such flag
    memory_rounds: bool = False,
    proxy_port: Optional[int] = None,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "run",
        "-i",
        str(tests_used),
        "-o",
        str(out_dir),
        "-w",
        str(workers),
        "-n",
        str(retry),
    ]
    if memory_rounds:
        # Fail loud instead of silently degrading. Without a live proxy port the
        # bench's per-round memory callback falls back to the dead default :30000
        # and every ingest is refused.
        if proxy_port is None:
            print(
                "[bench] FATAL: memory_rounds=True but proxy_port is None — "
                "refusing to run a MIRIX memory arm that would POST rounds to "
                "the dead default :30000. (rc=2)",
                flush=True,
            )
            return 2
        # The vendored bench flag name is historical; our proxy route maps it to
        # MIRIX production memory ingestion.
        cmd.append("--skill-records")
        cmd += ["--memory-proxy-port", str(proxy_port)]
    print(f"[bench] launching: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env, cwd=str(VENDOR_BENCH_DIR))


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------


def _find_report_json(bench_out: Path) -> Optional[Path]:
    candidates = sorted(bench_out.rglob("report.json"))
    return candidates[0] if candidates else None


def _parse_report(report_path: Path) -> tuple[Optional[float], Optional[int], dict]:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, {}
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    acc = summary.get("accuracy")
    if not isinstance(acc, (int, float)):
        acc = None
    tok = summary.get("tokens", {}) or {}
    total = 0
    found = False
    for bucket in tok.values():
        if not isinstance(bucket, dict):
            continue
        for k in ("total_input", "input", "output"):
            v = bucket.get(k)
            if isinstance(v, (int, float)):
                total += int(v)
                found = True
    return (
        (float(acc) if acc is not None else None),
        (total if found else None),
        summary,
    )


# ---------------------------------------------------------------------------
# Subprocess hooks (DI for tests)
# ---------------------------------------------------------------------------

ProxyStarter = Callable[[Path, int, Path, dict], subprocess.Popen]
ProxyStopper = Callable[[subprocess.Popen], None]
BenchRunner = Callable[..., int]


def _isoformat(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_arm(
    arm: str,
    days: int,
    out_dir: Optional[Path] = None,
    max_rounds: Optional[int] = None,
    retry: int = DEFAULT_BENCH_RETRY,
    *,
    proxy_starter: ProxyStarter = _start_proxy,
    proxy_stopper: ProxyStopper = _stop_proxy,
    bench_runner: BenchRunner = _run_bench,
    extra_env: Optional[dict] = None,
    mirix_url: str = DEFAULT_MIRIX_BASE_URL,
    pre_sliced_tests: Optional[Path] = None,
    extra_meta: Optional[dict] = None,
) -> RunResult:
    """Run a single arm of the MetaClaw eval end-to-end.

    Args:
        arm: one of:
            * ``"metaclaw"`` / ``"native"`` — vendored MetaClaw skill backend
              (cross-system anchor).
            * ``"mirix-generic"`` — MIRIX production memory path.
            * ``"no-skills"`` — floor (skills disabled).
        days: Number of dataset days to include (``0`` = full 30).
        out_dir: Directory for run artifacts.  Defaults to
            ``evals/metaclaw/runs/<arm>-<utc-ts>/``.
        max_rounds: Reserved — passes through to the bench in future slices.
        retry: Per-question retry count (``-n`` flag on metaclaw-bench).
        proxy_starter / proxy_stopper / bench_runner: DI hooks for tests.
        extra_env: Optional env overrides; useful in tests.
        mirix_url: Base URL of the MIRIX REST server (mirix-generic only).

    Returns:
        RunResult: see :class:`RunResult`.
    """
    if arm not in _SINGLE_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    spec = _resolve_arm(arm)
    if days < 0:
        raise ValueError(f"days must be >= 0, got {days}")

    bench_env_overrides = _resolve_benchmark_env(extra_env)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = RUNS_DIR / f"{arm}-{ts}"
    # Resolve to an ABSOLUTE path: the bench subprocess runs with cwd set to a
    # temp metaclaw_root (not the repo root), so a relative --output-dir (e.g.
    # the runner script passes evals/metaclaw/runs/v1-<arm>) would make the
    # bench's -i/-o resolve under that scratch cwd -> FileNotFoundError on
    # all_tests_used.json. Absolute paths make tests_used/-i/-o cwd-independent.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_out_dir = out_dir / "bench_output"
    bench_out_dir.mkdir(exist_ok=True)
    proxy_log = out_dir / "proxy.log"
    tests_used = out_dir / "all_tests_used.json"
    proxy_yaml = out_dir / "proxy.yaml"

    src_tests = DATA_DIR / "all_tests_metaclaw.json"
    if not src_tests.exists():
        raise FileNotFoundError(
            f"vendored 30-day dataset not found at {src_tests} — re-run vendoring"
        )

    if pre_sliced_tests is not None:
        # Test/repro hook: callers can provide an already-sliced dataset. We
        # still copy it into the per-arm dir so the output tree matches a normal
        # invocation.
        pre_sliced_tests = Path(pre_sliced_tests)
        if not pre_sliced_tests.exists():
            raise FileNotFoundError(
                f"pre_sliced_tests path does not exist: {pre_sliced_tests}"
            )
        shutil.copyfile(pre_sliced_tests, tests_used)
        # Compute kept from the JSON itself rather than the (potentially
        # stale) days argument; this is what the bench actually sees.
        try:
            with tests_used.open("r", encoding="utf-8") as fh:
                kept = len(json.load(fh).get("test") or [])
        except (OSError, json.JSONDecodeError):
            kept = 0
        print(
            f"[runner] reused pre-sliced dataset from {pre_sliced_tests} "
            f"({kept} day(s)) into {tests_used}",
            flush=True,
        )
    else:
        kept = slice_tests(src_tests, days, tests_used)
        print(f"[runner] sliced {kept} day(s) into {tests_used}", flush=True)

    # ---- MIRIX prelude: health-check server + mint a fresh user ----
    mirix_env: dict = {}
    mirix_user_id: Optional[str] = None
    if spec.needs_mirix:
        ok, status, detail = _mirix_health_diagnose(mirix_url)
        if not ok:
            msg = _format_mirix_unreachable(mirix_url, status, detail)
            print(msg, flush=True)
            return RunResult(
                arm=arm,
                exit_code=2,
                output_dir=out_dir,
                accuracy=None,
                total_tokens=None,
                report_summary={
                    "error": "mirix_unreachable",
                    "url": mirix_url,
                    "status": status,
                    "detail": detail,
                },
            )
        mirix_user_id = f"eval-metaclaw-{ts}-{arm}"
        resolved = _mirix_create_or_get_user(mirix_url, mirix_user_id)
        if resolved is None:
            msg = (
                f"ERROR: MIRIX /users/create_or_get failed for {mirix_user_id} "
                f"at {mirix_url}. Refusing to run the {arm} arm with a broken "
                f"backend user — that would silently produce a no-skills run "
                f"that is not comparable to the metaclaw arm.\n"
            )
            print(msg, flush=True)
            return RunResult(
                arm=arm,
                exit_code=2,
                output_dir=out_dir,
                accuracy=None,
                total_tokens=None,
                report_summary={
                    "error": "mirix_user_create_failed",
                    "url": mirix_url,
                    "user_id": mirix_user_id,
                },
            )
        mirix_env = {
            "METACLAW_SKILLS_PROVIDER": "mirix",
            "METACLAW_EVOLVER_PROVIDER": "mirix-generic",
            "METACLAW_MIRIX_BASE_URL": mirix_url,
            "METACLAW_MIRIX_USER_ID": mirix_user_id,
        }

        # Ensure a meta agent (with a procedural-memory child) exists under our
        # client. Production add_sync writes to the meta agent, and procedural
        # search resolves the same agent tree.
        if spec.skills_provider == "mirix":
            # The hardcoded eval client may exist read-only. Grant write_scope
            # first so meta initialization and add_sync can write.
            scope_ok = _mirix_ensure_client_write_scope(mirix_url)
            if not scope_ok:
                msg = (
                    f"ERROR: could not grant write_scope to MIRIX client "
                    f"{MIRIX_CLIENT_ID} at {mirix_url} for the {arm} arm. Without a "
                    f"write_scope the meta agent is never created (meta/initialize "
                    f"returns 200-null) and memory ingest/search would be "
                    f"degenerate. Refusing "
                    f"to proceed.\n"
                )
                print(msg, flush=True)
                return RunResult(
                    arm=arm,
                    exit_code=2,
                    output_dir=out_dir,
                    accuracy=None,
                    total_tokens=None,
                    report_summary={
                        "error": "mirix_eval_client_write_scope_failed",
                        "url": mirix_url,
                        "client_id": MIRIX_CLIENT_ID,
                    },
                )
            agent_ok = _mirix_ensure_meta_agent(mirix_url, bench_env_overrides)
            if not agent_ok:
                msg = (
                    f"ERROR: could not ensure a MIRIX meta agent at {mirix_url} for "
                    f"the {arm} arm. Without a procedural-memory agent the "
                    f"production memory path would be degenerate "
                    f"(0 skills retrieved/evolved). "
                    f"Refusing to proceed.\n"
                )
                print(msg, flush=True)
                return RunResult(
                    arm=arm,
                    exit_code=2,
                    output_dir=out_dir,
                    accuracy=None,
                    total_tokens=None,
                    report_summary={
                        "error": "mirix_meta_agent_ensure_failed",
                        "url": mirix_url,
                        "user_id": mirix_user_id,
                    },
                )

    started_at_ts = time.time()
    meta = {
        "arm": arm,
        "arm_label": spec.label,
        "days": days,
        "kept_tests": kept,
        "vendor_sha": _read_vendor_sha(),
        "mirix_url": mirix_url if spec.needs_mirix else None,
        "mirix_user_id": mirix_user_id,
        "evolution_mode": spec.evolution_mode,
        "memory_rounds": spec.memory_rounds,
        "started_at": _isoformat(started_at_ts),
        "finished_at": None,
        "exit_code": None,
        "accuracy": None,
        "total_tokens": None,
        "model": bench_env_overrides["BENCHMARK_MODEL"],
        "base_url": bench_env_overrides["BENCHMARK_BASE_URL"],
    }
    if extra_meta:
        # CLI passes estimated_wallclock_seconds_{min,max} here so the
        # budget estimate is durably attached to the run for postmortems.
        # Existing keys win — we never let extra_meta overwrite the
        # canonical fields above.
        for k, v in extra_meta.items():
            meta.setdefault(k, v)
    (out_dir / "run.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Per-run isolated workspace + synthetic METACLAW_ROOT layout.
    scratch_root = Path(tempfile.mkdtemp(prefix=f"metaclaw_root_{arm}_"))
    per_run_workspace = Path(tempfile.mkdtemp(prefix=f"metaclaw_ws_{arm}_"))
    skills_dir = Path(tempfile.mkdtemp(prefix=f"metaclaw_skills_{arm}_"))
    cleanup_dirs: list[Path] = [scratch_root, per_run_workspace, skills_dir]
    proxy_proc: Optional[subprocess.Popen] = None
    exit_code = 1
    run_succeeded = False

    try:
        # Copy workspaces/shared into the per-run isolated copy.
        shared_src = DATA_DIR / "workspaces" / "shared"
        if not shared_src.exists():
            raise FileNotFoundError(
                f"shared workspace not found at {shared_src} — re-run vendoring"
            )
        shutil.copytree(shared_src, per_run_workspace, dirs_exist_ok=True)
        _build_metaclaw_root(per_run_workspace, scratch_root)

        # Allocate proxy port + write YAML.
        port = _find_free_port()
        # mirix arms return a single flat bucket, so raise top_k to 10 to match
        # the count metaclaw's three-bucket retrieve injects (~8.5 avg). metaclaw
        # keeps paper's default 6 (it still adds task-specific + mistakes on top).
        _skill_top_k = 10 if spec.needs_mirix else 6
        _write_proxy_yaml(
            proxy_yaml,
            skills_dir,
            port,
            bench_env_overrides,
            skill_top_k=_skill_top_k,
            skills_enabled=spec.skills_enabled,
            auto_evolve=spec.auto_evolve,
            evolution_mode=spec.evolution_mode,
            evolution_every_n_rounds=DEFAULT_EXPECTED_TRIGGER_SESSIONS,
        )

        # Compose subprocess env.
        subprocess_env = _compose_subprocess_env(
            metaclaw_root=scratch_root,
            proxy_port=port,
            bench_env=bench_env_overrides,
            mirix_env=mirix_env,
        )

        # Launch proxy (skipped if bench_runner doesn't actually need a live proxy —
        # tests pass a stub bench_runner; we still build everything so the env-var
        # construction and cleanup paths get exercised).
        # Slice #1: bench_runner is `_run_bench`, so the proxy is required.
        if proxy_starter is _start_proxy:
            # Prewarm openclaw / clawdbot only when actually launching the real
            # proxy — tests use injected proxy_starter stubs and don't need it.
            _prewarm_openclaw()
            proxy_proc = proxy_starter(proxy_yaml, port, proxy_log, subprocess_env)
        else:
            # DI hook: still call so tests can assert it was called.
            proxy_proc = proxy_starter(proxy_yaml, port, proxy_log, subprocess_env)

        # Run bench. The MIRIX memory arm passes the vendored per-round callback
        # flag so the proxy can submit each visible {query, answer} turn to
        # MIRIX /memory/add_sync.
        exit_code = bench_runner(
            tests_used,
            bench_out_dir,
            subprocess_env,
            retry=retry,
            workers=DEFAULT_BENCH_WORKERS,
            max_rounds=max_rounds,
            memory_rounds=spec.memory_rounds,
            # The dynamic proxy port the bench must POST memory rounds to.
            proxy_port=port,
        )
        run_succeeded = exit_code == 0

    finally:
        if proxy_proc is not None:
            try:
                proxy_stopper(proxy_proc)
            except Exception as e:
                print(f"[runner] proxy_stopper raised: {e}", flush=True)
        # Preserve workspace + scratch on failure for debugging; only clean on success.
        if run_succeeded:
            for d in cleanup_dirs:
                shutil.rmtree(d, ignore_errors=True)
        else:
            print(
                f"[runner] preserving scratch dirs for debug: {[str(d) for d in cleanup_dirs]}",
                flush=True,
            )

    # Parse report.
    accuracy: Optional[float] = None
    total_tokens: Optional[int] = None
    report_summary: dict = {}
    report = _find_report_json(bench_out_dir)
    if report is not None:
        accuracy, total_tokens, report_summary = _parse_report(report)

    finished_at_ts = time.time()
    meta["exit_code"] = exit_code
    meta["finished_at"] = _isoformat(finished_at_ts)
    meta["accuracy"] = accuracy
    meta["total_tokens"] = total_tokens
    meta["wall_seconds"] = round(finished_at_ts - started_at_ts, 1)
    (out_dir / "run.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = RunResult(
        arm=arm,
        exit_code=exit_code,
        output_dir=out_dir,
        accuracy=accuracy,
        total_tokens=total_tokens,
        report_summary=report_summary,
    )
    print(
        f"[runner] arm={arm} rc={exit_code} accuracy={accuracy} "
        f"tokens={total_tokens} out={out_dir}",
        flush=True,
    )
    return result


def _ensure_python_shim(path_value: str) -> Optional[Path]:
    """Guarantee a ``python`` executable resolves for the checker subprocesses.

    The vendored ``file_check`` checkers run ``python scripts/check_*.py`` via
    ``/bin/sh``.  On hosts that only ship ``python3`` (e.g. macOS without a
    Homebrew/pyenv ``python`` shim), that command exits 127 ("python: command
    not found"), so EVERY file_check silently scores 0 and the file-check
    completion metric (Compl) collapses to 0% with no error ever surfaced to the
    runner — while multiple-choice questions (scored inline, no ``python`` call)
    keep working, which makes the failure easy to miss.

    If ``python`` already resolves on ``path_value`` we return None (no shim
    needed).  Otherwise we create a stable shim dir containing ``python`` ->
    the current interpreter (``sys.executable``) and return it so the caller can
    prepend it to PATH.
    """
    import shutil

    if shutil.which("python", path=path_value):
        return None  # `python` already available — no shim needed
    target = sys.executable
    if not target:
        return None
    shim_dir = EVALS_METACLAW_ROOT / ".pyshim"
    shim = shim_dir / "python"

    def _points_at_target() -> bool:
        return shim.is_symlink() and os.path.realpath(str(shim)) == os.path.realpath(
            target
        )

    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
        if _points_at_target():
            return shim_dir  # already correct — idempotent fast path
        # Atomically (re)point the shim so concurrent runs can't
        # trip over each other: build a pid-unique temp symlink, then os.replace
        # it onto `python` (atomic + overwrites). Avoids the unlink+symlink TOCTOU.
        tmp = shim_dir / f".python.{os.getpid()}.tmp"
        try:
            if tmp.is_symlink() or tmp.exists():
                tmp.unlink()
            tmp.symlink_to(target)
            os.replace(str(tmp), str(shim))
        finally:
            if tmp.is_symlink() or tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    except OSError:
        # A concurrent run may have already installed a correct shim — use it.
        return shim_dir if _points_at_target() else None
    return shim_dir


def _compose_subprocess_env(
    *,
    metaclaw_root: Path,
    proxy_port: int,
    bench_env: dict,
    mirix_env: Optional[dict] = None,
) -> dict:
    """Build the env dict for the proxy + bench subprocesses.

    Importantly:
    - ``PYTHONPATH`` is prepended with the vendor dirs so ``python -m metaclaw``
      and ``python -m src.cli`` resolve to our vendored copies (no runtime
      dependency on ``third_party/``).
    - ``PATH`` is prepended with the Node v22 nvm bin so ``openclaw`` resolves
      even when the parent shell uses a different node version.
    - ``METACLAW_ROOT`` and ``METACLAW_PROXY_PORT`` are the two upstream-known
      knobs the vendored bench and openclaw_cfg interpolate.
    """
    base = dict(os.environ)
    pythonpath_parts = [
        str(VENDOR_METACLAW_PKG_PARENT),  # so `import metaclaw` resolves
        str(VENDOR_BENCH_DIR),  # so `import src.*` resolves
        str(REPO_ROOT),  # so `import evals.metaclaw.*` resolves (slice #2+)
        base.get("PYTHONPATH", ""),
    ]
    base["PYTHONPATH"] = ":".join(p for p in pythonpath_parts if p)

    node_bin = _node_v22_bin()
    if node_bin:
        base["PATH"] = node_bin + ":" + base.get("PATH", "")

    # The vendored file_check checkers invoke `python scripts/check_*.py` via
    # /bin/sh; without a `python` on PATH that exits 127 and Compl silently → 0.
    python_shim = _ensure_python_shim(base.get("PATH", ""))
    if python_shim:
        base["PATH"] = str(python_shim) + ":" + base["PATH"]

    base["METACLAW_ROOT"] = str(metaclaw_root)
    base["METACLAW_PROXY_PORT"] = str(proxy_port)
    # Default backend; D6 (slice #2) dispatches on this env var.
    # "metaclaw" preserves upstream byte-identical behavior.  Callers
    # (tests, future slices) may override via extra_env.
    base.setdefault("METACLAW_SKILLS_PROVIDER", "metaclaw")

    # The vendored bench's openclaw_cfg references ${BENCHMARK_WORKSPACE_DIR};
    # bench's internal _patch_agent_workspace fills the workspace per-test.
    base.update(bench_env)
    # mirix arm: route the proxy's D6 dispatch to MirixSkillsAdapter (overrides
    # the "metaclaw" default set above).
    if mirix_env:
        base.update(mirix_env)
    return base


__all__ = ["RunResult", "run_arm"]
