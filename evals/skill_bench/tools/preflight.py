"""Check, before a paid run and without spending anything, that it can start.

Usage (from the repository root)::

    python -m evals.skill_bench.tools.preflight --env-file /path/to/.env
    python -m evals.skill_bench.tools.preflight --env-file /path/to/.env \\
        --benchmark alfworld webshop --method none mirix --split-mode smoke --budget 20
    python -m evals.skill_bench.tools.preflight --offline      # no network at all
    python -m evals.skill_bench.tools.preflight --env-file /path/to/.env \\
        --split-mode protocol --benchmark alfworld --budget 100 \\
        --runs-dir evals/skill_bench/runs/my-run \\
        --min-free-memory-gb 4 --min-free-disk-gb 10
    python -m evals.skill_bench.tools.preflight --env-file /path/to/.env \\
        --study phase16 --runs-dir evals/skill_bench/runs/phase16-r1 --budget 100

No model is called. The only requests that leave the machine are OpenRouter's key-info
endpoint (``GET /api/v1/key``, and ``/api/v1/credits`` for a key without its own limit),
which reads the balance and costs nothing, and the public metadata endpoints (no key
sent): the two ``tools/fetch_models.py`` reads and each provider-policy model's
``/models/<model>/endpoints``. ``--offline`` skips them all.

Checks, one line each:

* ``key``: ``OPENROUTER_API_KEY`` is in ``--env-file`` (or, without it, the environment),
  as the gateway reads it (``infra/gateway.load_api_key``). Its value is never printed:
  every line of output is scrubbed of it.
* ``balance``: what the key may still spend (``limit_remaining``, else the account's
  credits) covers the gateway's pause floor (``--balance-floor``, default the gateway's
  10 USD) plus ``--budget`` (the run's cost estimate, default 0), or, when
  ``--runs-dir`` holds a shared budget record, plus what is left of it (the ``budget``
  check's spend: a host that shares the record may start after another host has spent
  some of it). An unknown balance fails: the scheduler launches nothing on one.
* ``models``: ``config/models.json`` agrees with what ``fetch_models`` builds from the
  live metadata (``fetch_models --check``) in every field the protocol reads
  (``fetch_models.protocol_fields``: the token limits, modalities, ``seed``, reasoning
  efforts), so the frozen limits still hold. A difference is a warning, never a
  failure (recorded, not blocking): the line names each changed field with its frozen
  and its live value (``info.changes``: ``[{field, frozen, live}]``), and cells keep
  using ``config/models.json`` (the frozen limits and the provider-window clamp) until
  a refresh with ``fetch_models`` and the next freeze. A change only to what the
  protocol does not read (``raw_sha256`` of the raw entry, pricing ...) is a warning too
  and needs no refresh. Metadata that cannot be read or used at all fails. The line
  names the model every short key resolves to in this tree (``luna=openai/gpt-6-luna
  ...``, with ``--offline`` too), so a run from a stale checkout shows the model it
  would pay for before it starts.
* ``provider_policy``: for every model of ``protocol.PROVIDER_POLICY`` (none at
  present), the public endpoint list. The audit must recognise the ignored endpoints'
  provider names as ignored and no other endpoint's (``protocol.ignored_provider``),
  else it fails. An ignored slug that is the provider part of no endpoint's tag (a
  typo, or a provider OpenRouter delisted) is a warning: it excludes nothing, and
  editing the frozen policy mid-run would split the protocol across hosts. Fewer than
  :data:`MIN_POLICY_ENDPOINTS` other endpoints up (status 0) is a warning.
* ``protocol``: the tree's fingerprint equals the frozen record
  (``config/protocol_frozen.json``, ``freeze_protocol --check``). A drift, a missing
  record or a corrupt one is a warning: ``run.py`` freezes the tree automatically at
  the next attempt or resume and records why; a missing or corrupt record is never
  recreated (repair it with ``tools/freeze_protocol``). The warning names what changed
  (``info``: the fingerprints, the changed, added and removed files and the changed
  constants).
* ``code``: the commit and the code identity ``run.py`` records
  (``protocol.code_fingerprint``: the harness digest, and MIRIX's when a MIRIX cell
  runs). Two hosts that split a matrix must print the same line. Uncommitted changes
  to tracked files are a warning (the audit's ``dirty_tree``).
* ``host``: this host as every attempt records it (``cell_records.host_info``), with its
  machine id; a warning when no machine id can be read, since resumes then compare the
  hostname, which on a Mac follows the network.
* ``budget`` (with ``--runs-dir``): the spend budget's starting point in
  ``<runs-dir>/_gateway/budget.json`` (a second host copies the first host's record
  there before its first launch, so both count from one start; with
  ``--budget-start-from`` and no record there yet, that runs directory's record). With
  the key's usage (the ``balance`` check) it says what is spent since and what
  ``--budget`` leaves; a spent budget, a usage below the start (another key) or an
  unreadable record fails (the gateway would pause at once). A merged view
  (``_merged.json``) fails: cells never run there.
* ``resources``: available memory, swap and free disk (``infra/resources.sample``);
  below ``--min-free-memory-gb`` / ``--min-free-disk-gb`` (the scheduler's launch gates)
  is a warning.
* ``data:<benchmark>``: the items of the splits the run uses load, and ``run.py``'s own
  environment preflight passes: ALFWorld's game files and engine against the pinned
  hashes; WebShop's final manifest and its server (``envs/webshop.py`` ``preflight``: the
  server is ready, verified exactly this manifest and its price table, and runs the
  pinned upstream commit, data, spaCy model and fuzzy backend). WebShop is checked when
  its manifest is final (``tools/prepare_webshop.py freeze``), or when ``--benchmark``
  names it; otherwise it is skipped.
* ``mirix`` (when a MIRIX cell runs): this checkout's MIRIX client and renderer import;
  no ``.env`` is visible to MIRIX's ``load_dotenv()`` and no ``MIRIX_API_KEY`` is in the
  environment (the method refuses one at setup); the compose Postgres and its template
  database answer; MIRIX reads ``MIRIX_LLM_MAX_RETRIES=0``,
  ``MIRIX_LLM_TIMEOUT_SECONDS`` and ``MIRIX_EMBEDDING_TIMEOUT_SECONDS`` (``run.py``'s
  MIRIX preflight).

``--study`` checks what the named study (``study.STUDIES``, the scheduler's ``--study``)
runs: its benchmarks, methods and split mode; ``--benchmark`` and ``--method`` narrow
it. ``--json`` prints the checks as JSON instead of one line each.

Exit status: 0 no check failed (warnings and skips allowed), 1 at least one failed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evals.skill_bench import protocol, study

OK = "ok"
WARN = "warn"
SKIP = "skip"
FAIL = "FAIL"
#: OpenRouter's API root: ``/key``, ``/credits``, ``/models``, ``/embeddings/models``.
OPENROUTER_API = "https://openrouter.ai/api/v1"
HTTP_TIMEOUT_S = 30.0
#: The splits a cell of each split mode reads (``run.select_items``).
SPLITS = {
    "smoke": ("train", "val"),
    "protocol": ("train", "test"),
    # The no-memory pilot plays only its own split (WebShop's).
    "pilot": ("pilot",),
}
REDACTED = "<redacted>"
#: A shorter "key" is not a key (and replacing it would garble the output).
MIN_REDACTED_CHARS = 8
#: A provider policy leaving fewer endpoints up than this is a warning.
MIN_POLICY_ENDPOINTS = 5


@dataclass
class Check:
    name: str
    status: str
    detail: str
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# What the checks call (tests replace these)
# ---------------------------------------------------------------------------


def _default_env(benchmark: str) -> Any:
    from evals.skill_bench.run import default_env

    return default_env(benchmark)


async def _default_env_preflight(
    env: Any, benchmark: str, splits: Sequence[str]
) -> Mapping[str, Any]:
    from evals.skill_bench.run import preflight_env

    return await preflight_env(env, benchmark, splits)


async def _default_mirix(cell_dir: Path) -> Mapping[str, Any]:
    from evals.skill_bench.methods.mirix import load_mirix
    from evals.skill_bench.run import mirix_preflight

    await load_mirix()
    return await mirix_preflight("skill-bench-preflight", cell_dir)


def _webshop_prepared() -> bool:
    from evals.skill_bench.envs.webshop import MANIFEST_FINAL, load_split_manifest

    try:
        return load_split_manifest()["status"] == MANIFEST_FINAL
    except Exception:  # noqa: BLE001 - not prepared
        return False


def _default_host() -> Mapping[str, Any]:
    from evals.skill_bench.cell_records import host_info

    return host_info()


def _default_git() -> Mapping[str, Any]:
    from evals.skill_bench.cell_records import git_info

    return git_info()


def _default_resources(where: Path) -> Mapping[str, Any]:
    from evals.skill_bench.infra.resources import sample

    return sample(where).to_dict()


@dataclass
class Deps:
    """The pieces each check calls; the defaults are the real ones."""

    transport: httpx.AsyncBaseTransport | None = None
    env_factory: Callable[[str], Any] = _default_env
    env_preflight: Callable[[Any, str, Sequence[str]], Awaitable[Mapping[str, Any]]] = (
        _default_env_preflight
    )
    mirix: Callable[[Path], Awaitable[Mapping[str, Any]]] = _default_mirix
    protocol_drift: Callable[[], Mapping[str, Any]] = protocol.protocol_drift
    webshop_prepared: Callable[[], bool] = _webshop_prepared
    models_json: Path = protocol.MODELS_JSON
    code_fingerprint: Callable[..., Mapping[str, Any]] = protocol.code_fingerprint
    git: Callable[[], Mapping[str, Any]] = _default_git
    host: Callable[[], Mapping[str, Any]] = _default_host
    resources: Callable[[Path], Mapping[str, Any]] = _default_resources


@dataclass
class Options:
    env_file: Path | None = None
    benchmarks: tuple[str, ...] | None = None  # None: alfworld (+ webshop if prepared)
    methods: tuple[str, ...] = protocol.METHODS
    split_mode: str = "smoke"
    offline: bool = False
    balance_floor: float | None = None
    budget: float = 0.0
    openrouter: str = OPENROUTER_API
    runs_dir: Path | None = None
    min_free_memory_gb: float | None = None
    min_free_disk_gb: float | None = None
    #: ``schedule.py --budget-start-from``: the runs directory whose budget record this
    #: one will copy at its first launch (replicates that share one budget start).
    budget_start_from: Path | None = None


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def read_key(env_file: Path | None) -> tuple[str | None, str]:
    """The key and where it came from, as the gateway reads it (never printed)."""
    from evals.skill_bench.infra.gateway import KEY_ENV

    if env_file is not None:
        from dotenv import dotenv_values

        if not env_file.is_file():
            return None, f"--env-file {env_file} (no such file)"
        value = dotenv_values(env_file).get(KEY_ENV)
        source = f"--env-file {env_file}"
    else:
        value = os.environ.get(KEY_ENV)
        source = f"${KEY_ENV}"
    value = value.strip() if value else None
    return value or None, source


def check_key(key: str | None, source: str) -> Check:
    from evals.skill_bench.infra.gateway import KEY_ENV

    if key is None:
        return Check("key", FAIL, f"{KEY_ENV} not found in {source}")
    return Check("key", OK, f"{KEY_ENV} present in {source} (value not shown)")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


async def check_balance(
    client: httpx.AsyncClient, key: str | None, floor: float, budget: float
) -> Check:
    """``GET /key`` (key info, free); a key without a limit falls back to ``/credits``."""
    if key is None:
        return Check("balance", SKIP, "no key")
    headers = {"Authorization": f"Bearer {key}"}
    try:
        response = await client.get("/key", headers=headers)
    except httpx.HTTPError as exc:
        return Check("balance", FAIL, f"OpenRouter unreachable: {type(exc).__name__}")
    if response.status_code in (401, 403):
        return Check(
            "balance",
            FAIL,
            f"OpenRouter rejected the key (HTTP {response.status_code})",
        )
    if response.status_code != 200:
        return Check("balance", FAIL, f"GET /key answered HTTP {response.status_code}")
    try:
        data = dict(response.json()["data"])
    except (ValueError, KeyError, TypeError):
        return Check("balance", FAIL, "GET /key answered an unexpected body")
    remaining, source = _number(data.get("limit_remaining")), "key limit"
    if remaining is None:
        source = "account credits"
        try:
            credits = await client.get("/credits", headers=headers)
            body = credits.json()["data"] if credits.status_code == 200 else {}
            total, used = (
                _number(body.get("total_credits")),
                _number(body.get("total_usage")),
            )
            remaining = total - used if total is not None and used is not None else None
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            remaining = None
    info = {
        "remaining_usd": remaining,
        "source": source if remaining is not None else None,
        "usage_usd": _number(data.get("usage")),
        "limit_usd": _number(data.get("limit")),
        "is_free_tier": data.get("is_free_tier"),
        "floor_usd": floor,
        "budget_usd": budget,
    }
    if remaining is None:
        return Check(
            "balance",
            FAIL,
            "balance unknown (no key limit, no readable credits): the scheduler "
            "launches nothing on an unknown balance",
            info,
        )
    return judge_balance(Check("balance", OK, "", info), floor, budget, None)


def judge_balance(
    check: Check, floor: float, budget: float, spent: float | None
) -> Check:
    """The balance check's verdict: the key must hold the floor plus what is left of the
    budget. ``spent``: what the runs directory's shared budget has spent since its start
    (``check_budget``: a host that shares the budget may start after another host has
    spent some of it, so the whole budget is not needed), None: nothing yet."""
    info = check.info
    remaining = _number(info.get("remaining_usd"))
    if check.status == SKIP or remaining is None:
        return check
    left = max(0.0, budget - max(0.0, spent)) if spent is not None else budget
    needed = floor + left
    info["needed_usd"] = round(needed, 6)
    budget_text = (
        f"budget ${budget:.2f} - ${max(0.0, spent or 0.0):.2f} already spent"
        if spent is not None
        else f"budget ${budget:.2f}"
    )
    text = (
        f"${remaining:.2f} left ({info.get('source')}); needs ${needed:.2f} "
        f"(floor ${floor:.2f} + {budget_text})"
    )
    return Check("balance", FAIL if remaining < needed else OK, text, info)


def resolved_models() -> dict[str, str]:
    """What each short key of this tree resolves to (``luna`` -> ``openai/gpt-6-luna``)."""
    return {**protocol.MODELS, "embedding": protocol.EMBEDDING_MODEL}


def _resolved_text() -> str:
    return " ".join(f"{key}={model}" for key, model in resolved_models().items())


async def check_models(client: httpx.AsyncClient, models_json: Path) -> Check:
    """``fetch_models --check``: the live public metadata builds ``config/models.json``.

    Every outcome names the models this tree runs (:func:`resolved_models`)."""
    check = await _check_models(client, models_json)
    check.detail = f"{_resolved_text()}; {check.detail}"
    check.info["resolved"] = resolved_models()
    return check


async def _check_models(client: httpx.AsyncClient, models_json: Path) -> Check:
    from evals.skill_bench.tools import fetch_models

    try:
        chat = await client.get("/models")
        embeddings = await client.get("/embeddings/models")
        chat.raise_for_status()
        embeddings.raise_for_status()
        payloads = chat.json(), embeddings.json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check(
            "models", FAIL, f"model metadata unavailable: {type(exc).__name__}"
        )
    try:
        frozen = json.loads(models_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check("models", FAIL, f"{models_json} unreadable: {exc}")
    try:
        live = fetch_models.build_models_config(
            *payloads,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except (KeyError, ValueError) as exc:  # a model missing, or its limits unusable
        reason = exc.args[0] if exc.args else type(exc).__name__
        return Check("models", FAIL, f"the live metadata is unusable: {reason}")
    info: dict[str, Any] = {"frozen_at": frozen.get("fetched_at")}
    differences, unread = fetch_models.compare_models_config(frozen, live)
    if differences:
        changes = _field_changes(frozen, live)
        info["differences"] = differences
        info["unread_differences"] = unread
        info["changes"] = changes
        shown = "; ".join(
            f"{change['field']} {_shown(change['frozen'])} -> {_shown(change['live'])}"
            for change in changes
        ) or ", ".join(differences)
        return Check(
            "models",
            WARN,
            f"{models_json.name} (fetched {frozen.get('fetched_at')}) differs from the live "
            f"metadata in what the protocol reads: {shown}. {MODELS_CHANGE_RECORDED}",
            info,
        )
    if unread:
        info["unread_differences"] = unread
        return Check(
            "models",
            WARN,
            f"{models_json.name} (fetched {frozen.get('fetched_at')}) matches the live "
            "metadata in every field the protocol reads; OpenRouter changed only "
            f"{', '.join(unread)}, which the protocol does not read: no refresh, no "
            "re-freeze",
            info,
        )
    return Check(
        "models",
        OK,
        f"{models_json.name} matches the live metadata (fetched {frozen.get('fetched_at')})",
        info,
    )


#: What a ``models`` warning for a change the protocol reads says.
MODELS_CHANGE_RECORDED = (
    "Recorded, not blocking: cells keep using "
    "config/models.json (the frozen limits and the provider-window clamp); refresh with "
    "fetch_models before the next freeze if the change should reach the protocol"
)
#: A value longer than this is cut in a check's detail line (never in its ``info``).
SHOWN_VALUE_CHARS = 200


def _shown(value: Any) -> str:
    """``value`` as one detail-line token: JSON, cut to :data:`SHOWN_VALUE_CHARS`."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) > SHOWN_VALUE_CHARS:
        return text[: SHOWN_VALUE_CHARS - 3] + "..."
    return text


def _field_changes(
    frozen: Mapping[str, Any], live: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Each field the protocol reads that differs, with its frozen and its live value.

    The same fields as ``fetch_models.compare_models_config``'s protocol drift, in its
    order: the top-level keys of ``fetch_models.PROTOCOL_TOP_LEVEL`` (sorted), then per
    model (sorted) each field of ``fetch_models.protocol_fields``; a model only one side
    lists is ``{"field": "models.<id>", "frozen": "present" | None, "live": ...}``.
    """
    from evals.skill_bench.tools import fetch_models

    changes: list[dict[str, Any]] = []
    for key in sorted(fetch_models.PROTOCOL_TOP_LEVEL):
        if frozen.get(key) != live.get(key):
            changes.append(
                {"field": key, "frozen": frozen.get(key), "live": live.get(key)}
            )
    models_before = frozen.get("models")
    models_after = live.get("models")
    before = models_before if isinstance(models_before, Mapping) else {}
    after = models_after if isinstance(models_after, Mapping) else {}
    for model in sorted(set(before) | set(after)):
        if model not in before or model not in after:
            changes.append(
                {
                    "field": f"models.{model}",
                    "frozen": "present" if model in before else None,
                    "live": "present" if model in after else None,
                }
            )
            continue
        old = fetch_models.protocol_fields(before[model])
        new = fetch_models.protocol_fields(after[model])
        changes.extend(
            {"field": f"models.{model}.{name}", "frozen": old[name], "live": new[name]}
            for name in old
            if old[name] != new[name]
        )
    return changes


def _tag_provider(tag: Any) -> str:
    """The provider part of an endpoint tag (``open-inference/fp4`` -> ``open-inference``)."""
    return str(tag or "").split("/", 1)[0]


async def check_provider_policy(
    client: httpx.AsyncClient,
    policy: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> Check:
    """The provider policy against each policy model's public endpoint list (no key)."""
    policy = protocol.PROVIDER_POLICY if policy is None else policy
    if not policy:
        return Check(
            "provider_policy", OK, "no provider policy: every model unrestricted"
        )
    failures: list[str] = []
    warnings: list[str] = []
    parts: list[str] = []
    info: dict[str, Any] = {}
    for model, rule in sorted(policy.items()):
        ignored = [str(s) for s in rule.get("ignore", ())]
        try:
            response = await client.get(f"/models/{model}/endpoints")
            response.raise_for_status()
            endpoints = [
                e
                for e in response.json()["data"]["endpoints"]
                if isinstance(e, Mapping)
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                f"{model}: endpoint list unavailable ({type(exc).__name__})"
            )
            continue
        matched = {
            slug: [e for e in endpoints if _tag_provider(e.get("tag")) == slug]
            for slug in ignored
        }
        others = [
            e for e in endpoints if _tag_provider(e.get("tag")) not in set(ignored)
        ]
        up = [e for e in others if e.get("status") == 0]
        for slug, found in matched.items():
            if not found:
                # An ignore entry that matches nothing excludes nothing: harmless, and
                # "fixing" it by editing the policy mid-run would split the protocol
                # across hosts. A provider renamed under the same name is
                # still caught below, as an allowed provider the audit would flag.
                warnings.append(
                    f"{model}: ignored slug {slug!r} names no endpoint now (tags: "
                    f"{', '.join(sorted({_tag_provider(e.get('tag')) for e in endpoints}))}"
                    "): it excludes nothing; leave the policy as frozen"
                )
            for endpoint in found:
                name = endpoint.get("provider_name")
                if protocol.ignored_provider(model, name) != slug:
                    failures.append(
                        f"{model}: the audit would not recognise {name!r} (tag "
                        f"{endpoint.get('tag')}) as the ignored {slug!r}"
                    )
        flagged = sorted(
            {
                str(e.get("provider_name"))
                for e in others
                if protocol.ignored_provider(model, e.get("provider_name"))
            }
        )
        if flagged:
            failures.append(
                f"{model}: the audit would flag allowed providers as ignored: "
                + ", ".join(flagged)
            )
        if len(up) < MIN_POLICY_ENDPOINTS:
            warnings.append(
                f"{model}: only {len(up)} other endpoints are up (status 0), fewer than "
                f"{MIN_POLICY_ENDPOINTS}"
            )
        info[model] = {
            "ignore": ignored,
            "ignored_endpoints": {
                slug: [str(e.get("tag")) for e in found]
                for slug, found in matched.items()
            },
            "other_endpoints": len(others),
            "other_endpoints_up": len(up),
        }
        parts.append(
            f"{model} ignores "
            + ", ".join(
                f"{slug} ({len(found)} endpoints)" for slug, found in matched.items()
            )
            + f"; {len(others)} other endpoints, {len(up)} up"
        )
    if failures:
        return Check("provider_policy", FAIL, "; ".join(failures), info)
    if warnings:
        return Check("provider_policy", WARN, "; ".join(parts + warnings), info)
    return Check("provider_policy", OK, "; ".join(parts), info)


def check_code(deps: Deps, mirix: bool) -> Check:
    """The commit and the code identity this host would record."""
    try:
        code = dict(deps.code_fingerprint(mirix=mirix))
    except Exception as exc:
        return Check(
            "code", FAIL, f"code fingerprint unreadable: {type(exc).__name__}: {exc}"
        )
    git = dict(deps.git())
    commit = str(git.get("commit") or "")
    text = (
        f"commit {commit[:12] or 'unknown'}"
        + (f" ({git['branch']})" if git.get("branch") else "")
        + f"; harness {str(code.get('harness'))[:16]} ({code.get('harness_files')} files)"
    )
    if mirix:
        text += (
            f", mirix {str(code.get('mirix'))[:16]} ({code.get('mirix_files')} files)"
        )
    info = {"git": git, "code": code}
    if git.get("dirty"):
        return Check(
            "code",
            WARN,
            f"{text}; the tracked files differ from the commit (the audit's dirty_tree): "
            "run the committed code, the same on every host",
            info,
        )
    return Check("code", OK, text, info)


def check_host(deps: Deps) -> Check:
    try:
        host = dict(deps.host())
    except Exception as exc:
        return Check("host", WARN, f"host unreadable: {type(exc).__name__}: {exc}")
    text = (
        f"{host.get('hostname')} ({host.get('system')} {host.get('release')} "
        f"{host.get('machine')}), Python {host.get('python')}"
    )
    if not host.get("machine_id"):
        # Without a machine id the hostname names the host, and a Mac's follows its
        # network: a new name mid-run refuses every learning resume.
        return Check(
            "host",
            WARN,
            f"{text}; no machine id could be read (ioreg's IOPlatformUUID on macOS, "
            "/etc/machine-id on Linux), so resumes compare the hostname: fix it first "
            "(macOS: sudo scutil --set HostName <name>)",
            host,
        )
    return Check("host", OK, f"{text}, machine id {host['machine_id'][:8]}", host)


def check_budget(
    runs_dir: Path | None,
    budget: float,
    usage: float | None,
    start_from: Path | None = None,
) -> Check:
    """The runs directory's spend-budget record (``_gateway/budget.json``); with
    ``start_from`` (``schedule.py --budget-start-from``) and no record here yet, the
    record of that runs directory, which the first launch here copies."""
    from evals.skill_bench.cell_records import MERGED_MARKER
    from evals.skill_bench.infra.gateway import (
        BUDGET_FILE,
        BUDGET_USAGE_TOLERANCE_USD,
    )

    if runs_dir is None:
        return Check("budget", SKIP, "no --runs-dir")
    if (runs_dir / MERGED_MARKER).is_file():
        return Check(
            "budget",
            FAIL,
            f"{runs_dir} is a merged view of several hosts ({MERGED_MARKER}): cells "
            "never run there",
        )
    path = runs_dir / "_gateway" / BUDGET_FILE
    shared_from: Path | None = None
    if not path.is_file() and start_from is not None:
        source = start_from / "_gateway" / BUDGET_FILE
        if not source.is_file():
            return Check(
                "budget",
                WARN,
                f"no budget record in {path.parent} nor in {source.parent}: with "
                f"--budget-start-from the scheduler is refused (and retried by the "
                f"supervisor) until {start_from}'s gateway has written its record; start "
                "that runs directory first",
                {"file": str(path), "record": None, "start_from": str(source)},
            )
        shared_from, path = source, source
    if not path.is_file():
        return Check(
            "budget",
            OK,
            f"no budget record in {path.parent}: the first budgeted launch starts one at "
            "the key's usage then (a second host copies the first host's budget.json "
            "here before its first launch)",
            {"file": str(path), "record": None},
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        start = record["start_usage_usd"]
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError("start_usage_usd is not a number")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return Check(
            "budget",
            FAIL,
            f"{path} is unreadable ({type(exc).__name__}): the gateway would pause "
            "(budget_unknown)",
            {"file": str(path)},
        )
    amounts = [
        b.get("budget_usd") for b in record.get("budgets") or () if isinstance(b, dict)
    ]
    info: dict[str, Any] = {
        "file": str(path),
        "start_usage_usd": float(start),
        "started_at": record.get("started_at"),
        "last_budget_usd": amounts[-1] if amounts else None,
    }
    text = f"start ${float(start):.2f} at {record.get('started_at')}" + (
        f" (last budget ${amounts[-1]:.2f})"
        if amounts and isinstance(amounts[-1], (int, float))
        else ""
    )
    if shared_from is not None:
        info["start_from"] = str(shared_from)
        text += f", shared from {shared_from} (copied at the first launch here)"
    if usage is None:
        return Check("budget", OK, f"{text}; the key's usage is unknown here", info)
    spent = usage - float(start)
    info["spent_usd"] = round(spent, 6)
    if spent < -BUDGET_USAGE_TOLERANCE_USD:
        return Check(
            "budget",
            FAIL,
            f"{text}; the key's usage ${usage:.2f} is below it: another key? The gateway "
            "would pause (budget_invalid)",
            info,
        )
    text += f"; ${spent:.2f} spent since"
    if budget > 0:
        info["left_usd"] = round(budget - spent, 6)
        text += f", ${budget - spent:.2f} left of --budget ${budget:.2f}"
        if spent >= budget:
            return Check(
                "budget",
                FAIL,
                f"{text}: the gateway would pause at once (budget_spent)",
                info,
            )
    return Check("budget", OK, text, info)


def check_resources(
    deps: Deps, where: Path, min_memory_gb: float | None, min_disk_gb: float | None
) -> Check:
    """Available memory and free disk against the scheduler's launch gates."""
    try:
        reading = dict(deps.resources(where))
    except Exception as exc:
        return Check("resources", WARN, f"unreadable: {type(exc).__name__}: {exc}")
    memory = reading.get("mem_available_gb")
    total = reading.get("mem_total_gb")
    disk = reading.get("disk_free_gb")
    swap = reading.get("swap_used_gb")

    def gb(value: Any) -> str:
        return "?" if value is None else f"{float(value):.1f} GB"

    text = (
        f"memory {gb(memory)} available of {gb(total)}"
        + (f", swap used {gb(swap)}" if swap is not None else "")
        + f"; disk {gb(disk)} free under {where}"
    )
    info = {
        **reading,
        "min_free_memory_gb": min_memory_gb,
        "min_free_disk_gb": min_disk_gb,
    }
    problems = []
    if memory is None or disk is None:
        problems.append(f"unreadable: {', '.join(reading.get('errors') or ()) or '?'}")
    if min_memory_gb is not None and memory is not None and memory < min_memory_gb:
        problems.append(
            f"available memory below --min-free-memory-gb {min_memory_gb:g}: the "
            "scheduler would launch nothing"
        )
    if min_disk_gb is not None and disk is not None and disk < min_disk_gb:
        problems.append(
            f"free disk below --min-free-disk-gb {min_disk_gb:g}: the scheduler would "
            "launch nothing"
        )
    if problems:
        return Check("resources", WARN, f"{text}; " + "; ".join(problems), info)
    return Check("resources", OK, text, info)


#: What every ``protocol`` warning adds (``run.py`` auto-freezes).
AUTO_FREEZE_TEXT = (
    "run.py freezes this tree automatically at the next attempt or resume (a "
    "missing/corrupt record is never recreated: repair with tools/freeze_protocol)"
)


def check_protocol(drift_of: Callable[[], Mapping[str, Any]]) -> Check:
    """The tree against the frozen protocol record: equal is OK; a drift, a missing or a
    corrupt record is a warning, never a failure (``run.py`` auto-freezes at the next
    attempt or resume and records the change as a version)."""
    try:
        drift = drift_of()
    except Exception as exc:  # a missing or corrupt frozen record
        return Check(
            "protocol",
            WARN,
            f"no usable frozen record: {type(exc).__name__}: {exc}; {AUTO_FREEZE_TEXT}",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
    if drift.get("drifted"):
        # The manual re-freeze hint is replaced by the auto-freeze text.
        hint = getattr(protocol, "REFREEZE_HINT", None)
        lines = [
            line.strip()
            for line in protocol.describe_drift(drift).splitlines()
            if line.strip() and line.strip() != hint
        ]
        info = {
            key: drift.get(key)
            for key in (
                "current",
                "frozen",
                "changed",
                "added",
                "removed",
                "constants_changed_keys",
                "record_problems",
            )
        }
        return Check("protocol", WARN, "; ".join([*lines, AUTO_FREEZE_TEXT]), info)
    return Check(
        "protocol",
        OK,
        f"fingerprint {str(drift.get('current'))[:16]} equals the frozen value",
        {"fingerprint": drift.get("current")},
    )


async def check_data(benchmark: str, splits: Sequence[str], deps: Deps) -> Check:
    name = f"data:{benchmark}"
    try:
        env = await asyncio.to_thread(deps.env_factory, benchmark)
        items = {}
        for split in splits:
            items[split] = len(await asyncio.to_thread(env.load_tasks, split))
        info = dict(await deps.env_preflight(env, benchmark, splits))
    except Exception as exc:
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}")
    shown = ", ".join(f"{s} {n}" for s, n in items.items())
    return Check(name, OK, f"items load ({shown}); environment preflight passed", info)


async def check_mirix(deps: Deps) -> Check:
    try:
        with tempfile.TemporaryDirectory(prefix="skill-bench-preflight-") as scratch:
            info = dict(await deps.mirix(Path(scratch)))
    except Exception as exc:
        return Check("mirix", FAIL, f"{type(exc).__name__}: {exc}")
    contract = info.get("llm_contract") or {}
    return Check(
        "mirix",
        OK,
        f"Postgres {info.get('postgres')} with {info.get('template')}; MIRIX retries "
        f"{contract.get('llm_max_retries')}, timeout {contract.get('llm_timeout_seconds')} s"
        f", embedding timeout {contract.get('embedding_timeout_seconds')} s",
        info,
    )


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------


def protocol_policy_text() -> str:
    """The provider policy in words (``report.provider_policy_text``'s spelling)."""
    return (
        "; ".join(
            f"{model} ignores {', '.join(rule.get('ignore', ()))}"
            for model, rule in sorted(protocol.PROVIDER_POLICY.items())
        )
        or "no provider policy"
    )


def _floor(value: float | None) -> float:
    if value is not None:
        return value
    from evals.skill_bench.infra.gateway import DEFAULT_BALANCE_FLOOR_USD

    return DEFAULT_BALANCE_FLOOR_USD


async def run_checks(options: Options, deps: Deps | None = None) -> list[Check]:
    deps = deps or Deps()
    checks: list[Check] = []
    key, source = read_key(options.env_file)
    checks.append(check_key(key, source))
    if options.offline:
        checks.append(Check("balance", SKIP, "--offline"))
        checks.append(
            Check(
                "models",
                SKIP,
                f"{_resolved_text()}; --offline: not compared with the live metadata",
                {"resolved": resolved_models()},
            )
        )
        checks.append(
            Check(
                "provider_policy",
                SKIP,
                f"--offline: {protocol_policy_text()} not compared with the live "
                "endpoints",
                {"policy": protocol.provider_policy_settings()},
            )
        )
    else:
        async with httpx.AsyncClient(
            base_url=options.openrouter.rstrip("/"),
            timeout=HTTP_TIMEOUT_S,
            transport=deps.transport,
            headers={"Accept": "application/json"},
        ) as client:
            checks.append(
                await check_balance(
                    client, key, _floor(options.balance_floor), options.budget
                )
            )
            checks.append(await check_models(client, deps.models_json))
            checks.append(await check_provider_policy(client))
    checks.append(await asyncio.to_thread(check_protocol, deps.protocol_drift))
    checks.append(await asyncio.to_thread(check_code, deps, "mirix" in options.methods))
    checks.append(await asyncio.to_thread(check_host, deps))
    balance = next((c for c in checks if c.name == "balance"), None)
    usage = _number((balance.info if balance else {}).get("usage_usd"))
    budget_check = await asyncio.to_thread(
        check_budget,
        options.runs_dir,
        options.budget,
        usage,
        options.budget_start_from,
    )
    checks.append(budget_check)
    spent = _number(budget_check.info.get("spent_usd"))
    if balance is not None and spent is not None:
        # The shared budget has spent some already: the key needs only what is left.
        judged = judge_balance(
            balance, _floor(options.balance_floor), options.budget, spent
        )
        checks[checks.index(balance)] = judged
    checks.append(
        await asyncio.to_thread(
            check_resources,
            deps,
            options.runs_dir or Path.cwd(),
            options.min_free_memory_gb,
            options.min_free_disk_gb,
        )
    )
    requested = options.benchmarks
    prepared = {"webshop": await asyncio.to_thread(deps.webshop_prepared)}
    unprepared = {
        "webshop": "the manifest is provisional (python -m "
        "evals.skill_bench.tools.prepare_webshop freeze, on the server image)",
    }
    benchmarks = (
        list(requested)
        if requested is not None
        else [
            b
            for b in protocol.BENCHMARKS
            if prepared.get(b, True)
            and (
                options.split_mode != "pilot"
                or "pilot" in protocol.EXTRA_SPLIT_SIZES.get(b, {})
            )
        ]
    )
    splits = SPLITS[options.split_mode]
    for benchmark in protocol.BENCHMARKS:
        if benchmark in benchmarks:
            checks.append(await check_data(benchmark, splits, deps))
        elif benchmark in unprepared and requested is None:
            checks.append(Check(f"data:{benchmark}", SKIP, unprepared[benchmark]))
    if "mirix" in options.methods:
        checks.append(await check_mirix(deps))
    else:
        checks.append(Check("mirix", SKIP, "no MIRIX cell"))
    if key and len(key) >= MIN_REDACTED_CHARS:
        for check in checks:  # no line of output carries the key
            check.detail = check.detail.replace(key, REDACTED)
            check.info = json.loads(
                json.dumps(check.info, default=str).replace(key, REDACTED)
            )
    return checks


def render(checks: Sequence[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = [f"[{c.status:<4}] {c.name:<{width}}  {c.detail}" for c in checks]
    failed = [c.name for c in checks if c.status == FAIL]
    lines.append(
        f"preflight: {len(failed)} failed ({', '.join(failed)})"
        if failed
        else "preflight: ready (no check failed)"
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, deps: Deps | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env-file", type=Path, help="dotenv file with the key")
    parser.add_argument(
        "--study",
        type=study.study_argument,
        help="check what the named study (study.STUDIES, the scheduler's --study) "
        "runs: its benchmarks, methods and split mode; --benchmark and --method narrow it",
    )
    parser.add_argument("--benchmark", nargs="*", choices=protocol.BENCHMARKS)
    parser.add_argument(
        "--method", nargs="*", type=study.method_argument, choices=protocol.METHODS
    )
    parser.add_argument(
        "--split-mode",
        choices=tuple(SPLITS),
        help="default: smoke (or the study's)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the network checks (balance, model metadata, provider endpoints)",
    )
    parser.add_argument(
        "--balance-floor", type=float, help="USD (default: the gateway's floor)"
    )
    parser.add_argument(
        "--budget", type=float, default=0.0, help="USD the run is expected to spend"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="the runs directory the run will use (the budget line reads its "
        "_gateway/budget.json; the resources line its disk)",
    )
    parser.add_argument(
        "--budget-start-from",
        type=Path,
        metavar="RUNS_DIR",
        help="the scheduler's --budget-start-from: without a budget record in "
        "--runs-dir yet, the budget line reads RUNS_DIR's (the record the first launch "
        "copies)",
    )
    parser.add_argument(
        "--min-free-memory-gb",
        type=float,
        help="warn below this available memory (the scheduler's launch gate)",
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        help="warn below this free disk (the scheduler's launch gate)",
    )
    parser.add_argument("--openrouter", default=OPENROUTER_API, help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="print JSON, not text")
    args = parser.parse_args(argv)
    benchmarks = tuple(args.benchmark) if args.benchmark else None
    methods = tuple(args.method) if args.method else protocol.METHODS
    split_mode = args.split_mode or "smoke"
    if args.study:
        entry = study.STUDIES[args.study]
        if args.split_mode is not None and args.split_mode != entry.split_mode:
            parser.error(
                f"--study {entry.name} runs --split-mode {entry.split_mode}, not "
                f"{args.split_mode}"
            )
        try:
            benchmarks, _models, methods = study.narrowed(
                entry, benchmarks=args.benchmark, methods=args.method
            )
        except ValueError as exc:
            parser.error(str(exc))
        split_mode = entry.split_mode
    options = Options(
        env_file=args.env_file,
        benchmarks=benchmarks,
        methods=methods,
        split_mode=split_mode,
        offline=args.offline,
        balance_floor=args.balance_floor,
        budget=args.budget,
        openrouter=args.openrouter,
        runs_dir=args.runs_dir,
        min_free_memory_gb=args.min_free_memory_gb,
        min_free_disk_gb=args.min_free_disk_gb,
        budget_start_from=args.budget_start_from,
    )
    # What the libraries print while they are checked (MIRIX logs to stdout when its
    # client is imported) goes to stderr: stdout carries the report only.
    with contextlib.redirect_stdout(sys.stderr):
        checks = asyncio.run(run_checks(options, deps))
    if args.json:
        print(
            json.dumps(
                [asdict(c) for c in checks], indent=2, sort_keys=True, default=str
            )
        )
    else:
        sys.stdout.write(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
