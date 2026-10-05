"""Metrics, statistics, costs and diagnostics of a skill_bench run.

Usage (from the repository root)::

    python -m evals.skill_bench.report --runs-dir evals/skill_bench/runs
    python -m evals.skill_bench.report --split-mode smoke --out /tmp/smoke-report

It reads the scorer-side records ``run.py`` writes (``cell_records.py`` has the layout)
and the gateway ledgers, never an environment and never a model, and writes
``report.json`` and ``report.md`` (default ``<runs-dir>/_report/``), both stamped with
the frozen protocol fingerprint and every cell's own versions. The JSON is the source of
truth (``schema`` 4: :data:`SCHEMA_VERSION` lists what changed); the Markdown is rendered
from it.

**Record, don't block.** A condition that could stop the report or leave a cell out
is reported and warned about instead (``warnings``: one row per check, the cells it
concerns and what to do; a "Warning" line at the top of the Markdown): two directories
of one cell id, a cell under a protocol version no freeze registered, a reported attempt
that ran under more than one version, a legacy audit, degraded items, refused learning
calls, MIRIX learning calls lost. The CLI exits 0 whenever the report was written,
2 when no cell is complete, and 3 only with ``--require-audit`` when a complete cell is
not ``passed``.

**Which run is reported.** A cell's reported attempt is its highest-numbered
attempt whose state is ``complete``; an attempt a ``run.py --new-attempt --note``
superseded (state ``invalid``, ``invalidation`` ``superseded``) is listed with its note,
its scored items and its cost, and a newer attempt still running is named
(``newer_attempt``). A directory counts as complete when its ``status.json`` says so,
some attempt completed and its ``results.jsonl`` exists; rows of another attempt in the
reported records are named (``record_problems``) and the cell is not complete. Of several
directories of one cell id, the newest complete one is reported (by the reported
attempt's ``started_at``, then ``cell.json`` ``created_at``, then the name); the others
are listed under ``duplicates`` (``superseded`` or ``newer_incomplete``) and their money
stays counted (``costs.superseded_directories``, in ``costs.total``).

**Removed methods** (``study.REMOVED_METHODS``). The cells of a method the study
removed are set aside right after they are read (:func:`build_report`): they are in no
table, comparison or audit count. They are listed under ``removed`` (each method's
decision and reason; each cell with its status and its ledgers' costs) and in a
"Removed methods" section. Their money stays counted: the Cost section shows the matrix
total (the study's cells), the removed cells' share (``costs.removed``) and both
together (``costs.all_cells_total``); a gateway spend window sums every ledger of its
directory, theirs included, and a gateway runs directory only their ledgers name is
still found. Once their data was deleted (``tools/remove_method_data.py``), the removal
manifests (``_removed/<method>/<removal_id>/``, host mirrors included) stand in for
them: each deleted cell is listed from its manifest (``on_disk`` false, ``deleted``: the
removal, when and where), with the costs the manifest recorded, and a spend window adds
the manifest's copy of the deleted ledgers' calls, so every total is what it was before
the deletion; an unreadable manifest, or one whose calls do not match it, is warned
about.

What it reports:

* **Main metrics** of every complete cell on its test items, the environment's own
  outcome (:data:`METRICS`): ALFWorld's success rate; WebShop's success rate, mean score
  and success within 15 steps.
* **Comparison with the no-memory method** of the same benchmark, model and split, paired
  item by item: binary metrics (ALFWorld's and WebShop's success and WebShop's success
  within 15 steps) with the exact two-sided McNemar test; a metric
  that is not binary (WebShop's mean score) with a 95% paired bootstrap interval of the
  mean difference (10,000 resamples, seed 42, percentile method). Every cell ran once,
  so these only reflect which items were drawn (the single-run caveat is printed
  with every table). Of two complete cells with the same group and method the newest is
  compared and the other is named (a warning).
* **Versions** (recorded and reported, not enforced across cells: a baseline fixed
  after its run re-runs only its own cells). Every cell names the protocol its
  reported attempt ran under (``protocol``: its fingerprint, ``current``
  when it is the frozen value, else an earlier known one or ``known: false``; read with
  ``protocol.attempt_protocol``) and its code (``code``: the harness and MIRIX digests
  and the commit, from its ``attempt.json``). The reported attempt's version segments
  (``protocol.attempt_versions``: where a relaunch ran other code or another protocol)
  are listed per cell (``versions``: ``segments``, ``mixed``), with the hosts each ran
  on. Cells are compared across versions: each comparison row carries both cells'
  versions (``method_versions``, ``none_versions``), ``same_protocol`` (``"unknown"``
  when a side ran under a fingerprint no freeze registered: compared and marked, never
  left out), ``same_code``, ``mixed_version`` and ``same_platform``, and the Markdown
  marks a comparison across versions. ``protocol.versions`` lists each benchmark's cells
  per protocol and code version, and a benchmark with more than one gets a caveat
  (:data:`CAVEAT_MIXED_VERSIONS`). The audit still fails a cell under an unregistered
  fingerprint (``--require-audit`` leaves it out).
* **Runs and attempts**, the exception to the one-run rule: a cell whose ``cell.json``
  holds an operator note (``run.py --acknowledge-rerun``: the note, when it was recorded, and the
  earlier runs it covers), a cell with a superseded attempt (its note; the scores of both
  attempts), a cell with more than one directory, or one for which the
  audit found an earlier gateway run of the cell id that ran test items. They are
  listed, and named in the caveats. Every attempt listed names the versions it ran
  under (``attempts_detail[*]``: ``protocol``, ``code`` with the harness and MIRIX
  digests and the commit, ``version_segments``, ``versions_mixed``), so a superseded
  attempt's code is in the report.
* **Audit status** of every cell, from ``<runs-dir>/_audit/audit.json`` (``--audit``;
  schema 2 read by its ``validity_checks``, a legacy file marked so): ``passed`` only
  when the audit covered the reported attempt, found no violation, and saw the very
  ``results.jsonl``/``learning.jsonl`` bytes reported here; ``not checked`` when the
  auditing machine could not check it; a directory not reported is ``superseded`` or
  ``newer, not complete``. ``--require-audit`` leaves every other cell out of the
  comparisons (exit 3).
* **Learning curves**: the learning items' outcomes in stream order, per window of 10
  (``protocol.LEARNING_CURVE_WINDOW``), from the scorer. No method saw them.
* **Cost** from the gateway ledgers (``cost_total``, which includes billed retries):
  per cell, role and phase, learning side (``setup`` + ``learning``) apart from test, the
  reported attempt apart from the invalidated attempts. Attempts without a ledger,
  calls without a final ledger line and cut-off calls (an attempt cut off after its
  request was sent, by the deadline or a dropped connection: billed by OpenRouter for
  what the provider generated) have an unknown cost: they are
  listed, not guessed. Per gateway runs directory, the key's usage growth
  (``_gateway/balance.jsonl``) against every ledger's ``cost_total`` in the same window
  is the **unattributed spend**. The matrix total includes the ledgers so far
  of cells that are not complete. The learning side includes the calls of tries a
  learning resume rolled back (``learning_discarded``, from ``discarded.jsonl``),
  and **Resumed learning** lists every resume: the checkpoint
  it restored, what was in flight, and the discarded calls and their cost.
* **Two hosts**: every cell names the host its reported attempt ran on. Over a merged
  view of several hosts (``tools/sync_runs.py``: cells and runs linked in from
  ``_hosts/<name>/``) each host's gateway runs directory has its own unattributed
  spend, which counts the other hosts' spend too (one key); when the hosts share one
  budget start, the spend is also computed once over the union of their ledgers
  (``spend_union``). Ledgers are found relative to the runs directory first
  (``cell_records.resolve_ledger_path``), so a copied runs directory reads its own. The
  header shows what a merged view's ``_merged.json`` says (``merged_view``): when the
  view was built and each host synced, and why it is not final (a ``--mirror-only``
  sync, or one stopped by conflicts or an rsync failure, since it was built; a host's
  last rsync that did not exit 0; each conflict). It changes no number.
* **Provider policy**: the header names the routing the protocol sets
  (``protocol.PROVIDER_POLICY``) and the routing the reported attempts' ledgers show
  their calls carried (``wire.provider``, per model), and says so when they differ:
  ``mixed``, ``none`` or ``not recorded`` (a report over data from before the policy,
  or over a mix, must not claim the policy).
* **Providers** OpenRouter chose per cell and role, and ``model_served`` (a source of
  variance); per model and provider, the billed cost against the list price (the service
  tier), the answering attempts' latency and the calls past half the deadline, and the
  reasoning tokens (mean and p95), naming a provider that reasons more than 10 times
  the median of the model's other providers. With ``--provider-endpoints``
  (saved OpenRouter endpoint listings) each provider's quantization is named, and each
  cell's accuracy is split by the provider that served most of an item's test actor
  calls (``accuracy_by_main_provider``, ``provider_accuracy``).
* **Injected tokens** per item and phase: distribution, empty injections, and the
  total over an episode's actor calls (ALFWorld injects on every step).
* **Diagnostics**: MIRIX learning runs, experience pool and retrievals; J's
  agreement with the true outcome (the scorer's success: ALFWorld's outcome, WebShop's
  score of 1), computed here and only here, after the experiment; ALFWorld results per
  task type, a scorer-side label that appears in this report only; failures, re-runs and the
  methods' own counters. What a cell went on despite (record, don't stop;
  ``cell_records.tolerated_failures``) is listed: items scored or learned
  with the provider refusing the actor (``provider_unserved``) or a degraded MIRIX
  retrieval (``degraded``; the metrics without the unserved items beside them), the
  learning calls AWM and ReasoningBank counted as refused (``method_refusals``), and the
  MIRIX learning calls lost and accepted (``tolerated``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evals.skill_bench import cell_records, protocol, study
from evals.skill_bench.cell_records import (
    ATTEMPT_COMPLETE,
    ATTEMPT_INVALID,
    HOSTS_DIR,
    LATENCY_EDGES_S,
    LAUNCH_WORK,
    NON_CELL_DIRS,
    NOTE_RERUN_ACKNOWLEDGED,
    REASONING_EDGES,
    SLOW_CALL_DEADLINE_FRACTION,
    STATUS_COMPLETE,
    CellPaths,
    call_completed_at,
    discarded_call_ids,
    host_key,
    host_text,
    read_discarded,
    read_json,
    read_jsonl,
    records_digest,
    resolve_ledger_path,
    summarize_ledger,
    unknown_cost_attempts,
)
from evals.skill_bench.infra.failures import provider_unserved_row
from evals.skill_bench.tools.remove_method_data import (
    REMOVED_DIR,
    cell_row_problem,
    load_manifests,
    manifest_calls,
)
from evals.skill_bench.tools.sync_runs import merged_view_status

#: The report JSON's schema version.
#:
#: 2: LLM-graded metrics (``llm_accuracy``, ``grading`` blocks, ``statistics.graded``)
#: and the removed methods' keys (``removed``, ``costs.removed``,
#: ``costs.all_cells_total``).
#:
#: 3 (record, don't block): ``comparisons[*].same_protocol`` may
#: be the string ``"unknown"`` (a cell under a protocol version no freeze registered is
#: compared and marked); ``errors`` is always ``[]`` (two directories of one cell or two
#: cells of one group and method are warnings: the newest complete one is reported);
#: new top-level keys ``warnings``, ``duplicates``, ``provider_accuracy`` and
#: ``provider_endpoints``; new per-cell keys ``attempts_detail``, ``newer_attempt``,
#: ``record_problems``, ``versions``, ``hosts``, ``audit_detail``, ``provider_mix``,
#: ``accuracy_by_main_provider``, ``degraded``, ``method_refusals``, ``tolerated``,
#: ``sandbox_failures`` and ``learning_curve_complete``; ``test.items_graded`` /
#: ``items_ungraded`` / ``metrics_without_provider_unserved``; ``grading.
#: cells_partly_graded`` / ``rubric_current`` / ``cells_other_rubric``;
#: ``costs.superseded_directories``; ``removed.deleted`` and
#: ``removed.cells[*].on_disk`` / ``deleted``; ``attempts_detail[*].protocol`` /
#: ``code`` / ``version_segments`` / ``versions_mixed`` (each attempt's versions).
#: ``cells`` and ``costs.total`` hold the study's cells (one directory per cell id;
#: ``costs.total`` counts every directory).
#:
#: 4: no LLM-graded metrics: every key only they used is gone (``grading`` blocks and
#: costs, graded-item counts, ``sandbox_failures``, ``learning_curve_complete``, the
#: comparisons' and attempts' grading fields).
SCHEMA_VERSION = 4
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
#: Where ``audit.py`` writes its verdict by default (``<runs-dir>/_audit/audit.json``).
AUDIT_FILE = Path("_audit") / "audit.json"
#: A cell's audit status: only ``passed`` means audit.json vouches for the very
#: records this report prints.
AUDIT_PASSED = "passed"
AUDIT_FAILED = "failed"
AUDIT_STALE = "stale"
AUDIT_NOT_AUDITED = "not audited"
AUDIT_MISSING = "no audit"
#: audit.json schema 2: the auditing machine could not check the cell (its items or its
#: environment are missing there). Not ``passed``: ``--require-audit`` fails it.
AUDIT_NOT_CHECKED = "not checked"
#: A directory the report does not report (another directory of its cell id is), as the
#: audit's ``duplicates`` list names it (audit.json schema 2).
AUDIT_SUPERSEDED = "superseded (not audited)"
AUDIT_NEWER_INCOMPLETE = "newer, not complete"
#: ``attempt.json`` ``invalidation`` of an attempt a ``run.py --new-attempt --note``
#: replaced after it ran test items (``run.INVALIDATION_SUPERSEDED``; a literal here: the
#: report never imports run.py). Its state is ``invalid`` and its ``previous_state`` the
#: state it had.
INVALIDATION_SUPERSEDED = "superseded"
#: The roles of a directory that is not reported although it holds a cell id another
#: directory reports (the audit's rule, audit.json ``duplicates``): below the chosen one
#: by (reported attempt's ``started_at``, ``cell.json`` ``created_at``, name), or above it
#: but not complete (a newer run still going).
ROLE_SUPERSEDED = "superseded"
ROLE_NEWER_INCOMPLETE = "newer_incomplete"
#: Ledger phases on the learning side of a cell (the method's setup belongs to learning).
LEARNING_PHASES = frozenset({"setup", "learning"})
#: The gateway's record of the key's usage, one row per balance poll (``infra/gateway.py``),
#: below a gateway runs directory.
BALANCE_FILE = Path("_gateway") / "balance.jsonl"
#: A provider whose mean reasoning tokens exceed this many times the median of the
#: model's other providers (and :data:`REASONING_OUTLIER_MIN_MEAN`) is named.
REASONING_OUTLIER_FACTOR = 10
REASONING_OUTLIER_MIN_MEAN = 100

CAVEAT_SINGLE_RUN = (
    "Every cell ran once. The McNemar p-values and bootstrap intervals only "
    "reflect which test items were drawn; they contain no run-to-run variation (the "
    "model's sampling, OpenRouter's provider choice), which one run cannot measure."
)
CAVEAT_JUDGE = (
    "J's agreement with the true outcome is computed here, after the experiment, from "
    "scorer-side records. No method and no judge call ever saw the true outcome."
)
CAVEAT_TASK_TYPES = (
    "ALFWorld task types are scorer-side labels; they appear in this report only."
)
CAVEAT_PROVIDERS = (
    "OpenRouter chose the provider of every call (no pinning); the provider mix of "
    "a cell is a source of variance."
)
CAVEAT_PRICE_TIERS = (
    "OpenRouter's provider name does not say which service tier served a call: OpenAI "
    "lists gpt-6-luna at $0.05/$0.25 per million tokens (flex), $0.10/$0.50 and "
    "$0.20/$1.00 (fast), Azure and Bedrock at up to 1.1 times the list price. The billed "
    "cost against the list price of config/models.json tells them apart (about 0.5, 1 "
    "or 2). Latency is the answering attempt's, bucketed; a quantile is shown as the "
    "bucket's upper edge, and so is the reasoning tokens' p95 (the mean is exact, over "
    "the ok calls that report reasoning tokens)."
)
CAVEAT_UNKNOWN_COST = (
    "A missing ledger, a call without its final ledger line (a gateway stopped "
    "mid-call) and a cut-off call (an attempt cut off after its request was sent, by the "
    "deadline or a dropped connection: OpenRouter bills what the provider generated, and "
    "the ledger never learns the amount) have an unknown cost, counted here as $0."
)
CAVEAT_UNATTRIBUTED = (
    "The key's usage counts every call made with the key: the cut-off calls' unknown "
    "cost, and any other user of the key in the window (another runs directory's "
    "gateway included). OpenRouter updates the usage a little after each call, so the "
    "calls that ended just before the last balance poll may be missing from it."
)
#: A provider whose billed cost is this far from the list price is named in the report.
PRICE_RATIO_BAND = (0.9, 1.15)
CAVEAT_ALFWORLD = (
    "On ALFWorld every method (and J) can tell a won episode from its length, since a game "
    "has no losing state; the protocol accepts this termination signal."
)
CAVEAT_WEBSHOP = (
    "On WebShop every method (and J) sees the purchase score after Buy Now, in the "
    "learning phase only: it mirrors real user feedback and is the same for every "
    'method; methods see "S0 + score" there. Success is a score of 1; no success '
    "threshold is put in J or MIRIX's distiller, and J's agreement with a score of 1 "
    "is measured (judge_agreement)."
)
CAVEAT_MIXED_VERSIONS = (
    "Not every cell of a benchmark ran one protocol and one code version (a baseline "
    "fixed after its run re-runs only its own cells, so versions are recorded and "
    "reported per cell, not enforced across cells). A comparison across versions is "
    "still made and is marked (same_protocol / same_code); the Cells table names each "
    "cell's version."
)
CAVEATS: tuple[str, ...] = (
    CAVEAT_SINGLE_RUN,
    CAVEAT_JUDGE,
    CAVEAT_TASK_TYPES,
    CAVEAT_PROVIDERS,
    CAVEAT_ALFWORLD,
    CAVEAT_WEBSHOP,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDef:
    """One reported metric, read from the scorer's record: ``key`` None reads
    ``score.primary``, else ``score.metrics[key]``."""

    name: str
    label: str
    key: str | None
    binary: bool
    main: bool


METRICS: Mapping[str, tuple[MetricDef, ...]] = {
    "alfworld": (MetricDef("success_rate", "success rate", None, True, True),),
    # Success (reward == 1) first, the mean score second, success within 15 steps as a
    # diagnostic (equal to the success rate while the step cap is 15). The environment's
    # reward.
    "webshop": (
        MetricDef("success_rate", "success rate (score = 1)", "success", True, True),
        MetricDef("score", "mean score", None, False, False),
        MetricDef(
            f"success_within_{protocol.WEBSHOP_DIAGNOSTIC_STEPS}",
            f"success within {protocol.WEBSHOP_DIAGNOSTIC_STEPS} steps",
            f"success_within_{protocol.WEBSHOP_DIAGNOSTIC_STEPS}",
            True,
            False,
        ),
    ),
}
if set(METRICS) != set(protocol.BENCHMARKS):  # never an empty metric list
    raise RuntimeError(
        f"report.METRICS names {sorted(METRICS)}, protocol.BENCHMARKS is "
        f"{sorted(protocol.BENCHMARKS)}: every benchmark needs its metrics"
    )


def metric_value(row: Mapping[str, Any], metric: MetricDef) -> float:
    """The metric of one scored result row; binary metrics must be 0 or 1."""
    score = row.get("score")
    if not isinstance(score, Mapping):
        raise ValueError(f"item {row.get('raw_id')} has no score")
    if metric.key is None:
        value = score.get("primary")
    else:
        value = (score.get("metrics") or {}).get(metric.key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"item {row.get('raw_id')} has no numeric {metric.name}")
    value = float(value)
    if metric.binary and value not in (0.0, 1.0):
        raise ValueError(f"item {row.get('raw_id')}: {metric.name} {value} is not 0/1")
    return value


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    ``b`` = items the method gets right and the baseline wrong, ``c`` = the reverse. Under
    the null each discordant pair is a fair coin, so ``p = min(1, 2 * P(X <= min(b, c)))``
    with ``X ~ Binomial(b + c, 1/2)``; no discordant pair gives 1.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, (2 * tail) / (2**n))


def paired_bootstrap(
    differences: Sequence[float],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    level: float = protocol.CI_LEVEL,
    seed: int = protocol.SEED,
) -> dict[str, Any]:
    """Percentile bootstrap interval of the mean of paired differences (method - baseline).

    Items are resampled with replacement ``resamples`` times from a generator seeded with
    ``seed`` (numpy PCG64), so the interval is reproducible.
    """
    d = np.asarray(list(differences), dtype=float)
    n = int(d.size)
    if n == 0:
        raise ValueError("no paired items")
    if resamples < 1 or not 0 < level < 1:
        raise ValueError("resamples must be >= 1 and level in (0, 1)")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    chunk = max(1, 2_000_000 // n)  # bounded memory: chunk x n indices at a time
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start : start + size] = d[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "delta": float(d.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "level": level,
        "resamples": resamples,
        "seed": seed,
        "ci_method": "percentile",
        "n": n,
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def distribution(values: Sequence[float]) -> dict[str, Any]:
    """Count, mean, median, 10th/90th percentile (linear), min, max and total."""
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    p10, p90 = np.quantile(array, [0.1, 0.9])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(statistics.median(values)),
        "p10": float(p10),
        "p90": float(p90),
        "min": float(array.min()),
        "max": float(array.max()),
        "total": float(array.sum()),
    }


def agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Confusion matrix of (J says success, truly success) pairs, accuracy and Cohen's kappa."""
    tp = sum(1 for j, t in pairs if j and t)
    fp = sum(1 for j, t in pairs if j and not t)
    fn = sum(1 for j, t in pairs if not j and t)
    tn = sum(1 for j, t in pairs if not j and not t)
    n = len(pairs)
    out: dict[str, Any] = {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    if not n:
        return out
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    out.update(
        accuracy=observed,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        kappa=(observed - expected) / (1 - expected) if expected < 1 else None,
    )
    return out


# ---------------------------------------------------------------------------
# Loading cells
# ---------------------------------------------------------------------------


@dataclass
class AttemptData:
    number: int
    state: dict[str, Any]
    ledger: dict[str, Any] | None
    #: ``discarded.jsonl``: the segments a learning resume rolled back.
    discarded: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CellData:
    """Everything the report reads about one cell."""

    cell_id: str
    paths: CellPaths
    definition: dict[str, Any]
    manifest: dict[str, Any]
    status: dict[str, Any]
    attempts: list[AttemptData] = field(default_factory=list)
    reported: int | None = None
    complete: bool = False
    results: list[dict[str, Any]] = field(default_factory=list)
    learning: list[dict[str, Any]] = field(default_factory=list)
    injections: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    frozen: dict[str, Any] | None = None
    #: The attempt run after the last one that completed, while it has not completed
    #: itself (a ``--new-attempt --note`` rerun going on): ``{number, state, started_at,
    #: supersedes}``. The cell is then not complete: no metric until it finishes.
    newer_attempt: dict[str, Any] | None = None
    #: Why the cell's published records were not read (rows of another attempt than the
    #: reported one): the cell is then not complete.
    record_problems: list[str] = field(default_factory=list)
    #: Read once, on first use: the reported ledger's final call lines
    #: (:func:`_ledger_rows`) and ``cell_records.tolerated_failures``.
    ledger_calls: list[dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )
    tolerated_cache: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def benchmark(self) -> str:
        return str(self.definition["benchmark"])

    @property
    def method(self) -> str:
        return str(self.definition["method"])

    @property
    def model(self) -> str:
        return str(self.definition["model"])

    @property
    def model_key(self) -> str:
        return str(self.definition.get("model_key") or self.model)

    @property
    def split_mode(self) -> str:
        return str(self.definition.get("split_mode") or "protocol")

    @property
    def group(self) -> str:
        """The cells compared with each other: same benchmark, model and test items. A
        smoke cell's test items depend only on ``n_test`` (the head of the val order), so
        a no-memory smoke cell run with ``--n-train 0`` pairs with every learning smoke
        cell of the same ``n_test``. A pilot cell plays other items than the protocol
        cells: a group of its own."""
        base = f"{self.benchmark}-{self.model_key}"
        if self.split_mode == "smoke":
            base += f"-smoke-t{self.definition.get('n_test')}"
        elif self.split_mode == "pilot":
            base += "-pilot"
        return base

    @property
    def method_dir(self) -> Path | None:
        if self.reported is None:
            return None
        return self.paths.attempt(self.reported).method_dir

    def attempt(self, number: int | None) -> AttemptData | None:
        return next((a for a in self.attempts if a.number == number), None)


def _ledger_path(
    paths: CellPaths,
    state: Mapping[str, Any],
    runs_dir: Path,
    gateway_runs_dirs: Sequence[Path] = (),
) -> Path:
    """An attempt's ledger, relative to the runs directory first: a runs directory copied
    from another host reads its own ledgers (``cell_records.resolve_ledger_path``)."""
    return resolve_ledger_path(
        state, runs_dir=runs_dir, gateway_runs_dirs=gateway_runs_dirs
    )


def discover_cells(runs_dir: Path) -> list[Path]:
    """Cell directories (with a ``cell.json``) directly below ``runs_dir``, sorted. The
    removed methods' data (``_removed/``) and hidden entries (``.name``, such as a
    deletion's trash) are never cells."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        child
        for child in runs_dir.iterdir()
        if child.is_dir()
        and child.name not in NON_CELL_DIRS
        and child.name != REMOVED_DIR
        and not child.name.startswith(".")
        and (child / "cell.json").is_file()
    )


def attempt_was_complete(state: Mapping[str, Any]) -> bool:
    """Whether an attempt completed: its state is ``complete``, or it was superseded by a
    later ``--new-attempt --note`` after it completed (state ``invalid``, invalidation
    ``superseded``, ``previous_state`` ``complete``). A superseded attempt is never
    reported; this only tells the newer attempt it is being replaced by."""
    if state.get("state") == ATTEMPT_COMPLETE:
        return True
    return (
        state.get("invalidation") == INVALIDATION_SUPERSEDED
        and state.get("previous_state") == ATTEMPT_COMPLETE
    )


def attempt_superseded(state: Mapping[str, Any]) -> bool:
    """Whether a ``--new-attempt --note`` replaced the attempt after it ran test items
    (``invalidation`` ``superseded``)."""
    return (
        state.get("state") == ATTEMPT_INVALID
        and state.get("invalidation") == INVALIDATION_SUPERSEDED
    )


def _row_attempt_problem(
    rows: Sequence[Mapping[str, Any]], name: str, reported: int | None
) -> str | None:
    """A problem when a published row names another attempt than the reported one (rows
    without an integer ``attempt``, older records, are accepted)."""
    others = sorted(
        {
            r["attempt"]
            for r in rows
            if isinstance(r.get("attempt"), int)
            and not isinstance(r.get("attempt"), bool)
            and r["attempt"] != reported
        }
    )
    if not others:
        return None
    return (
        f"{name} rows belong to attempt {', '.join(str(n) for n in others)}, not the "
        f"reported attempt {reported}"
    )


def load_cell(
    root: Path, runs_dir: Path | None = None, gateway_runs_dirs: Sequence[Path] = ()
) -> CellData:
    """Read one cell's records (blocking)."""
    paths = CellPaths(root)
    runs_dir = runs_dir or root.parent
    manifest = read_json(paths.manifest) or {}
    definition = dict(manifest.get("cell") or {})
    status = read_json(paths.status) or {}
    cell = CellData(
        cell_id=str(definition.get("cell_id") or root.name),
        paths=paths,
        definition=definition,
        manifest=manifest,
        status=status if isinstance(status, dict) else {},
    )
    for number in paths.attempt_numbers():
        state = read_json(paths.attempt(number).state) or {}
        ledger_path = _ledger_path(paths, state, runs_dir, gateway_runs_dirs)
        discarded = discarded_call_ids(paths.attempt(number))
        ledger = (
            summarize_ledger(ledger_path, discarded=discarded)
            if ledger_path.is_file()
            else None
        )
        segments = _discarded_segments(paths, number, ledger_path) if discarded else []
        cell.attempts.append(AttemptData(number, state, ledger, segments))
    # The reported attempt is the highest-numbered one whose state is complete (a
    # superseded attempt is ``invalid``, so never reported); the directory is complete
    # when status.json says so, some attempt is complete and results.jsonl exists. A
    # cell whose rerun is running is therefore not complete.
    complete = [a for a in cell.attempts if a.state.get("state") == ATTEMPT_COMPLETE]
    current = manifest.get("current_attempt")
    if complete:
        cell.reported = complete[-1].number
    elif isinstance(current, int):
        cell.reported = current
    elif cell.attempts:
        cell.reported = cell.attempts[-1].number
    cell.complete = (
        bool(complete)
        and cell.status.get("status") == STATUS_COMPLETE
        and paths.results.is_file()
    )
    finished = [a.number for a in cell.attempts if attempt_was_complete(a.state)]
    last = cell.attempts[-1] if cell.attempts else None
    if (
        last is not None
        and finished
        and last.number > max(finished)
        and last.state.get("state") != ATTEMPT_COMPLETE
    ):
        cell.newer_attempt = {
            "number": last.number,
            "state": last.state.get("state"),
            "started_at": last.state.get("started_at"),
            "supersedes": last.state.get("supersedes"),
        }
    if cell.reported is not None:
        attempt = paths.attempt(cell.reported)
        cell.injections = read_jsonl(attempt.injections)
        cell.failures = read_jsonl(attempt.failures)
        frozen = read_json(attempt.frozen_snapshot)
        cell.frozen = frozen if isinstance(frozen, dict) else None
    if cell.complete:
        cell.results = read_jsonl(paths.results)
        cell.learning = read_jsonl(paths.learning)
        # Every published number comes from one attempt's rows: rows written by
        # another attempt make the cell not complete, never a mix.
        problems = [
            p
            for p in (
                _row_attempt_problem(cell.results, "results.jsonl", cell.reported),
                _row_attempt_problem(cell.learning, "learning.jsonl", cell.reported),
            )
            if p
        ]
        if problems:
            cell.record_problems = problems
            cell.complete = False
            cell.results, cell.learning = [], []
    return cell


def _discarded_segments(
    paths: CellPaths, number: int, ledger_path: Path
) -> list[dict[str, Any]]:
    """The learning resumes of attempt ``number`` (``discarded.jsonl``), each with what
    its discarded calls cost as the ledger's final lines record it (a call in flight when
    the harness died gets its final line later than the row; blocking)."""
    rows = read_discarded(paths.attempt(number))
    finals = {
        str(r.get("call_id")): r
        for r in read_jsonl(ledger_path)
        if r.get("event") == "call" and r.get("call_id")
    }
    out = []
    for row in rows:
        calls = [c for c in row.get("calls") or () if isinstance(c, Mapping)]
        roles: Counter[str] = Counter(str(c.get("role")) for c in calls)
        cost = 0.0
        unknown = 0
        for c in calls:
            final = finals.get(str(c.get("call_id")))
            if final is not None:
                cost += float(_number(final.get("cost_total")) or 0.0)
                unknown += int(unknown_cost_attempts(final) > 0)
        checkpoint = (
            row.get("checkpoint") if isinstance(row.get("checkpoint"), Mapping) else {}
        )
        in_flight = (
            row.get("in_flight") if isinstance(row.get("in_flight"), Mapping) else {}
        )
        moved = row.get("moved") if isinstance(row.get("moved"), Mapping) else {}
        out.append(
            {
                "segment": row.get("segment"),
                "checkpoint": checkpoint.get("id"),
                "reason": row.get("reason"),
                "failure_code": (row.get("failure") or {}).get("code")
                if isinstance(row.get("failure"), Mapping)
                else None,
                "operator_resume": bool(row.get("operator_resume")),
                "at": row.get("at"),
                "in_flight_position": in_flight.get("position"),
                "in_flight_stage": in_flight.get("stage"),
                "rows_moved": list(moved.get("learning_positions") or ()),
                "calls": len(calls),
                "calls_by_role": dict(sorted(roles.items())),
                "cost": round(cost, 10),
                "unknown_cost_calls": unknown,
            }
        )
    return out


def load_cells(
    runs_dir: Path,
    *,
    split_mode: str | None = "protocol",
    gateway_runs_dirs: Sequence[Path] = (),
) -> list[CellData]:
    """Every cell of ``runs_dir`` (``split_mode`` None: protocol and smoke cells alike)."""
    cells = [
        load_cell(root, runs_dir, gateway_runs_dirs)
        for root in discover_cells(runs_dir)
    ]
    if split_mode is not None:
        cells = [c for c in cells if c.split_mode == split_mode]
    return cells


# ---------------------------------------------------------------------------
# Per-cell sections
# ---------------------------------------------------------------------------


#: Row flags ``run.py`` writes, only when true: the actor's provider refused the
#: request and the item was scored unanswered (``provider_unserved``: a test item after
#: two consecutive no-endpoint 404 refusals, a learning item after one refusal, HTTP
#: 400/404/413); MIRIX's retrieval lost its topic extraction or query vector and was
#: injected anyway (``retrieval_degraded``). A row is provider-unserved by the audit's own
#: predicate (``infra/failures.provider_unserved_row``: the flag, or the ``end_reason``
#: or failure code of a row written without it).
ROW_PROVIDER_UNSERVED = "provider_unserved"
ROW_RETRIEVAL_DEGRADED = "retrieval_degraded"
DEGRADED_FLAGS: tuple[str, ...] = (ROW_PROVIDER_UNSERVED, ROW_RETRIEVAL_DEGRADED)
#: What provider_unserved means, as the report prints it.
UNSERVED_TEXT = (
    "the actor's provider refused the request and the item was scored unanswered "
    "(test items: two consecutive no-endpoint 404 refusals; learning items: one "
    "refusal, HTTP 400/404/413)"
)


def _row_flagged(row: Mapping[str, Any], flag: str) -> bool:
    """Whether ``row`` carries degraded condition ``flag`` (:data:`DEGRADED_FLAGS`)."""
    if flag == ROW_PROVIDER_UNSERVED:
        return provider_unserved_row(row)
    return row.get(flag) is True


def _metric_means(
    rows: Sequence[Mapping[str, Any]], metrics: Iterable[MetricDef]
) -> dict[str, float | None]:
    return {m.name: mean([metric_value(r, m) for r in rows]) for m in metrics}


def test_metrics(cell: CellData) -> dict[str, Any]:
    """The cell's test metrics, and how many items went unanswered (``unanswered``, from
    the scorer's record). With any test row scored ``provider_unserved`` (two
    consecutive no-endpoint refusals, scored unanswered),
    ``metrics_without_provider_unserved`` is the same metrics over the other rows (None
    without such a row)."""
    rows = cell.results
    defs = METRICS.get(cell.benchmark, ())
    metrics = _metric_means(rows, defs)
    unserved = [r for r in rows if provider_unserved_row(r)]
    served = [r for r in rows if not provider_unserved_row(r)]
    answered = [
        bool((r.get("score") or {}).get("answered", True))
        for r in rows
        if r.get("score") is not None
    ]
    return {
        "items": len(rows),
        "metrics": metrics,
        "unanswered": sum(1 for a in answered if not a),
        **item_diagnostics(rows),
        "metrics_without_provider_unserved": _metric_means(served, defs)
        if unserved
        else None,
        "model_failures": sum(int(r.get("model_failures") or 0) for r in rows),
        "format_errors": sum(int(r.get("format_errors") or 0) for r in rows),
        "steps_mean": mean([float(r.get("steps") or 0) for r in rows]),
        "actor_calls_mean": mean([float(r.get("actor_calls") or 0) for r in rows]),
    }


def item_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the items allow, where the scorer records it (WebShop: each item's
    best possible reward, buying the gold product with the goal's options): the mean and
    how many items cannot reach a score of 1 (``unsolvable_items``). Empty for a benchmark
    that records none. A ``provider_unserved`` row is left out: the rollout zeroes every
    metric of such a row, this item constant included."""
    best = [
        float(value)
        for r in rows
        if not provider_unserved_row(r)
        and isinstance(r.get("score"), Mapping)
        and isinstance(
            value := (r["score"].get("metrics") or {}).get("best_reward"), (int, float)
        )
        and not isinstance(value, bool)
    ]
    if not best:
        return {}
    return {
        "best_reward_mean": mean(best),
        "unsolvable_items": sum(1 for b in best if b < 1.0),
    }


def learning_curve(
    cell: CellData, window: int = protocol.LEARNING_CURVE_WINDOW
) -> list[dict[str, Any]]:
    """The learning stream's outcomes in stream order, one row per window of ``window``
    items: the scored learning rows (a row without a score is skipped)."""
    rows = sorted(
        (r for r in cell.learning if r.get("score") is not None),
        key=lambda r: int(r.get("position") or 0),
    )
    out = []
    metrics = METRICS.get(cell.benchmark, ())
    for start in range(0, len(rows), window):
        chunk = rows[start : start + window]
        out.append(
            {
                "window": start // window + 1,
                "first_position": int(chunk[0].get("position") or 0),
                "last_position": int(chunk[-1].get("position") or 0),
                "items": len(chunk),
                "metrics": _metric_means(chunk, metrics),
            }
        )
    return out


def _split_role_bucket(
    by_phase_role: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Learning side vs test vs other, each with totals and per-role numbers."""
    sides: dict[str, dict[str, Any]] = {}
    for key, bucket in by_phase_role.items():
        phase, _, role = key.partition("/")
        side = (
            "learning"
            if phase in LEARNING_PHASES
            else "test"
            if phase == "test"
            else "other"
        )
        entry = sides.setdefault(
            side,
            {
                "cost": 0.0,
                "calls": 0,
                "unknown_cost_calls": 0,
                "input_tokens_actual": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "by_role": {},
            },
        )
        cost = float(bucket.get("cost_total") or 0.0)
        unknown = int(bucket.get("unknown_cost_calls") or 0)
        entry["cost"] += cost
        entry["calls"] += int(bucket.get("calls") or 0)
        entry["unknown_cost_calls"] += unknown
        for name in ("input_tokens_actual", "output_tokens", "reasoning_tokens"):
            entry[name] += int(bucket.get(name) or 0)
        role_entry = entry["by_role"].setdefault(
            role, {"cost": 0.0, "calls": 0, "unknown_cost_calls": 0}
        )
        role_entry["cost"] += cost
        role_entry["calls"] += int(bucket.get("calls") or 0)
        role_entry["unknown_cost_calls"] += unknown
    for entry in sides.values():
        entry["cost"] = round(entry["cost"], 10)
        for role_entry in entry["by_role"].values():
            role_entry["cost"] = round(role_entry["cost"], 10)
    return sides


def cell_costs(cell: CellData) -> dict[str, Any]:
    """Costs from the gateway ledgers: the reported attempt by side and role, and the
    invalidated attempts (their spend is part of the experiment's cost)."""
    reported = cell.attempt(cell.reported)
    sides = (
        _split_role_bucket(reported.ledger["by_phase_role"])
        if reported and reported.ledger
        else {}
    )
    others = [a for a in cell.attempts if a.number != cell.reported]

    def spent(attempt: AttemptData) -> float:
        return float((attempt.ledger or {}).get("totals", {}).get("cost_total") or 0.0)

    other_cost = sum(spent(a) for a in others)
    # A superseded attempt (--new-attempt --note after it ran test items) apart from the
    # attempts the harness or an operator invalidated otherwise; both were billed.
    superseded_cost = sum(spent(a) for a in others if attempt_superseded(a.state))
    other_calls = sum(
        int((a.ledger or {}).get("totals", {}).get("calls") or 0) for a in others
    )
    item_costs = (
        [
            float(costs.get("test") or 0.0)
            for costs in ((reported.ledger or {}).get("items") or {}).values()
            if "test" in costs
        ]
        if reported
        else []
    )
    unknown_items = (
        (reported.ledger or {}).get("items_unknown_cost") or {} if reported else {}
    )
    total = sum(s["cost"] for s in sides.values()) + other_cost
    empty = {"cost": 0.0, "calls": 0, "unknown_cost_calls": 0, "by_role": {}}
    lost = ((reported.ledger or {}).get("discarded") if reported else None) or {}
    return {
        "reported_attempt": cell.reported,
        "learning": sides.get("learning", dict(empty)),
        # Part of the learning side: the calls of tries a learning resume rolled back
        # (discarded.jsonl), made and billed once more by the redo.
        "learning_discarded": {
            "cost": round(float(lost.get("cost_total") or 0.0), 10),
            "calls": int(lost.get("calls") or 0),
            "unknown_cost_calls": int(lost.get("unknown_cost_calls") or 0),
            "by_role": dict(lost.get("by_role") or {}),
        },
        "test": sides.get("test", dict(empty)),
        "other_phases": sides.get("other"),
        "other_attempts": {
            "attempts": [a.number for a in others],
            "states": {str(a.number): a.state.get("state") for a in others},
            "cost": round(other_cost, 10),
            "calls": other_calls,
            "superseded": [a.number for a in others if attempt_superseded(a.state)],
            "superseded_cost": round(float(superseded_cost), 10),
            "invalid_cost": round(
                float(
                    sum(
                        spent(a)
                        for a in others
                        if a.state.get("state") == ATTEMPT_INVALID
                        and not attempt_superseded(a.state)
                    )
                ),
                10,
            ),
        },
        "total_cost": round(total, 10),
        "actor_cost_per_test_item": mean(item_costs),
        # Test items with an actor call of unknown cost: the mean above undercounts them.
        "test_items_unknown_cost": sum(
            1 for phases in unknown_items.values() if "test" in phases
        ),
        "ledgers_missing": [a.number for a in cell.attempts if a.ledger is None],
        # Calls without a final ledger line (a gateway stopped mid-call): not in the
        # totals above, since their cost is unknown.
        "unfinished_calls": sum(
            int((a.ledger or {}).get("unfinished_calls") or 0) for a in cell.attempts
        ),
        # Calls with an attempt the deadline cut off, in every attempt: billed, at a cost
        # the ledger never learns, so not in the totals above either.
        "unknown_cost_calls": sum(
            int(((a.ledger or {}).get("totals") or {}).get("unknown_cost_calls") or 0)
            for a in cell.attempts
        ),
    }


def cell_providers(cell: CellData) -> dict[str, Any]:
    """The providers OpenRouter used per role (all phases), and ``model_served``."""
    reported = cell.attempt(cell.reported)
    ledger = (reported.ledger if reported else None) or {}
    by_role: dict[str, Counter[str]] = defaultdict(Counter)
    overall: Counter[str] = Counter()
    for key, providers in (ledger.get("providers") or {}).items():
        role = key.partition("/")[2]
        for provider, count in providers.items():
            by_role[role][provider] += int(count)
            overall[provider] += int(count)
    total = sum(overall.values())
    return {
        "overall": dict(overall.most_common()),
        "shares": {p: n / total for p, n in overall.most_common()} if total else {},
        "by_role": {r: dict(c.most_common()) for r, c in sorted(by_role.items())},
        "model_served": ledger.get("model_served") or {},
        "pricing": ledger.get("pricing") or {},
        "timeouts_model_blamed": ledger.get("timeouts_model_blamed") or {},
    }


def cell_injections(cell: CellData) -> dict[str, Any]:
    """Tokens injected per item (o200k, counted by the framework), per phase."""
    out: dict[str, Any] = {}
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell.injections:
        by_phase[str(row.get("phase"))].append(row)
    for phase, rows in sorted(by_phase.items()):
        latest = {str(r.get("raw_id")): r for r in rows}  # a re-run item: last row wins
        tokens = [float(r.get("tokens") or 0) for r in latest.values()]
        totals = [float(r.get("injected_tokens_total") or 0) for r in latest.values()]
        out[phase] = {
            "items": len(latest),
            "empty": sum(1 for t in tokens if t == 0),
            "tokens": distribution(tokens),
            "tokens_times_actor_calls": distribution(totals),
        }
    return out


def _truths(row: Mapping[str, Any]) -> dict[str, bool]:
    """The true outcome of a learning item: ALFWorld's scorer-side success, WebShop's
    score of 1 (J's agreement with reward == 1 is measured)."""
    score = row.get("score") or {}
    return {"success": float(score.get("primary") or 0.0) >= 1.0}


def judge_agreement(cell: CellData) -> dict[str, Any] | None:
    """J's verdicts (``judge.jsonl``) against the true outcome of each learning item
    (:func:`_truths`)."""
    method_dir = cell.method_dir
    if method_dir is None or not (method_dir / "judge.jsonl").is_file():
        return None
    judged: dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(method_dir / "judge.jsonl"):
        if row.get("event", "verdict") == "verdict" and row.get("task_uid"):
            judged[str(row["task_uid"])] = row
    truth_rows = {
        str(r.get("task_uid")): r for r in cell.learning if r.get("score") is not None
    }
    joined = [(judged[u], truth_rows[u]) for u in truth_rows if u in judged]
    out: dict[str, Any] = {
        "verdicts": len(judged),
        "joined": len(joined),
        "unparseable": sum(1 for v in judged.values() if v.get("parsed") is False),
        "model_failures": sum(1 for v in judged.values() if v.get("failure")),
        "by_truth": {},
    }
    names = sorted({name for _, row in joined for name in _truths(row)})
    for name in names:
        pairs = [(bool(v.get("success")), _truths(row)[name]) for v, row in joined]
        out["by_truth"][name] = agreement(pairs)
    return out


#: ``history_skipped`` reasons of a MIRIX curator batch that the report names: the
#: pool's embedding failed (the attempt should have restarted), or the pool or MIRIX's
#: embeddings were off (not MIRIX's default).
POOL_SKIPS_NAMED = ("embedding_failed", "disabled", "embeddings_disabled")


def mirix_pool(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    """The experience pool of a MIRIX frozen snapshot: what the method froze
    (``experience_pool``), or, in a snapshot frozen before the method exported it, the
    same summary computed from the learning runs
    (:func:`evals.skill_bench.methods.mirix.experience_pool`). None: no curator batch
    reports the pool (a MIRIX without the experience pool)."""
    if "experience_pool" in snapshot:
        pool = snapshot["experience_pool"]
        return dict(pool) if isinstance(pool, Mapping) else None
    from evals.skill_bench.methods.mirix import experience_pool

    runs = [r for r in snapshot.get("learning_runs") or () if isinstance(r, Mapping)]
    return experience_pool(runs)


def mirix_diagnostics(cell: CellData) -> dict[str, Any] | None:
    """MIRIX's learning runs and experience pool (frozen snapshot: :func:`mirix_pool`),
    what its distiller was shown and said of the served skills and the usage records
    its curator saw (``learning_context``: ``methods.mirix.learning_context``, None for
    a MIRIX that records no session context) and retrievals (the method's records).

    A retrieval the method refused because it lost its topic extraction or query vector
    to the infrastructure (the item was re-run, not scored on it) is counted
    under ``refused`` and left out of the other retrieval numbers; ``extraction_fallback``
    counts the injected retrievals whose extraction MIRIX itself fell back from (the
    model's unusable answer: MIRIX's own behaviour, scored)."""
    if cell.method != "mirix":
        return None
    from evals.skill_bench.methods.mirix import learning_context

    snapshot = (cell.frozen or {}).get("snapshot") or {}
    runs = list(snapshot.get("learning_runs") or [])
    usage: Counter[str] = Counter()
    for run in runs:
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens"):
            value = (run.get("llm_usage") or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] += int(value)
    retrievals = (
        read_jsonl(cell.method_dir / "mirix_retrievals.jsonl")
        if cell.method_dir is not None
        else []
    )
    by_phase: dict[str, list[int]] = defaultdict(list)
    refused: Counter[str] = Counter()
    refused_roles: dict[str, Counter[str]] = defaultdict(Counter)
    fallback: Counter[str] = Counter()
    degraded: Counter[str] = Counter()
    for row in retrievals:
        phase = "test" if row.get("frozen") else "learning"
        failure = row.get("failure")
        if isinstance(failure, Mapping):
            refused[phase] += 1
            refused_roles[phase][str(failure.get("role"))] += 1
            continue
        # An accepted degraded retrieval (``degraded`` set, ``failure`` null) was injected:
        # counted with the injected ones, and on its own.
        if isinstance(row.get("degraded"), Mapping):
            degraded[phase] += 1
        by_phase[phase].append(len(row.get("item_ids") or []))
        evidence = row.get("evidence")
        if (
            isinstance(evidence, Mapping)
            and evidence.get("extraction_status") == "fallback"
        ):
            fallback[phase] += 1
    history = snapshot.get("skill_history") or {}
    tolerated = cell_tolerated(cell) or {}
    tolerated_mirix = ((tolerated.get("method") or {}).get("mirix")) or {}
    lost = mirix_lost_learning_calls(cell)
    return {
        "learning_runs": len(runs),
        "runs_by_status": dict(Counter(str(r.get("status")) for r in runs)),
        "runs_by_trigger": dict(Counter(str(r.get("trigger")) for r in runs)),
        "sessions_per_run": distribution(
            [float(len(r.get("session_ids") or [])) for r in runs]
        ),
        "run_errors": [str(r.get("error")) for r in runs if r.get("error")][:20],
        "retry_events": sum(len(r.get("retry_events") or []) for r in runs),
        "llm_usage": dict(usage),
        "experience_pool": mirix_pool(snapshot),
        "learning_context": learning_context(
            [r for r in runs if isinstance(r, Mapping)]
        ),
        "skills": len(snapshot.get("skills") or []),
        "skill_versions": sum(
            len((v.get("versions") if isinstance(v, Mapping) else v) or [])
            for v in history.values()
        )
        if isinstance(history, Mapping)
        else None,
        "retrievals": {
            phase: {
                "count": len(by_phase.get(phase, [])),
                "empty": sum(1 for s in by_phase.get(phase, []) if s == 0),
                "skills_returned": distribution(
                    [float(s) for s in by_phase.get(phase, [])]
                ),
                "extraction_fallback": fallback[phase],
                "refused": refused[phase],
                "refused_by_role": dict(refused_roles[phase]),
                "degraded": degraded[phase],
            }
            for phase in sorted(set(by_phase) | set(refused))
        },
        # The ledger's count (exact, whatever the method recorded): the learning side's
        # MIRIX and embedding calls that did not end ok and no learning resume rolled
        # back (MIRIX retries none, so each was lost to its batch).
        "learning_calls_lost": lost["calls"],
        "learning_calls_lost_by": lost["by"],
        # What the method itself accepted and went on with (snapshot ``tolerated``;
        # record, don't stop): counts by reason, the problems.
        "tolerated": dict(
            (tolerated_mirix.get("counts") or {})
            if isinstance(tolerated_mirix, Mapping)
            else {}
        ),
        "tolerated_problems": len(
            (tolerated_mirix.get("problems") or ())
            if isinstance(tolerated_mirix, Mapping)
            else ()
        ),
        # attempt.json accepted_learning_calls: batches accepted after 3 rollbacks.
        "accepted_learning_calls": len(tolerated.get("accepted_learning_calls") or ()),
    }


def _ledger_rows(cell: CellData) -> list[dict[str, Any]]:
    """The reported attempt's final ``call`` ledger lines (read once per cell; blocking).
    Empty without a ledger."""
    if cell.ledger_calls is None:
        reported = cell.attempt(cell.reported)
        path = ((reported.ledger if reported else None) or {}).get("ledger")
        cell.ledger_calls = (
            [r for r in read_jsonl(Path(str(path))) if r.get("event") == "call"]
            if path
            else []
        )
    return cell.ledger_calls


def _call_phase(row: Mapping[str, Any]) -> str:
    """A call's phase: the run's phase ``run.py`` set, else the caller's tag."""
    return str(row.get("run_phase") or row.get("phase"))


#: The MIRIX side's roles: its own model calls and its embeddings.
MIRIX_ROLES = frozenset({"mirix", "embedding"})


def mirix_lost_learning_calls(cell: CellData) -> dict[str, Any]:
    """The reported attempt's learning-side (``setup`` and ``learning``) MIRIX and
    embedding calls whose final ledger line is not ``ok`` and that no learning resume
    discarded (``discarded.jsonl``): ``{calls, by: {"role/status": n}}``."""
    dropped = (
        discarded_call_ids(cell.paths.attempt(cell.reported))
        if cell.reported is not None
        else frozenset()
    )
    by: Counter[str] = Counter()
    for row in _ledger_rows(cell):
        if (
            _call_phase(row) in LEARNING_PHASES
            and row.get("role") in MIRIX_ROLES
            and row.get("status") != "ok"
            and str(row.get("call_id")) not in dropped
        ):
            by[f"{row.get('role')}/{row.get('status')}"] += 1
    return {"calls": sum(by.values()), "by": dict(sorted(by.items()))}


def cell_tolerated(cell: CellData) -> dict[str, Any] | None:
    """What the reported attempt was scored or learned despite
    (``cell_records.tolerated_failures``: the one reader of the rows' flags, the methods'
    refusal counters, MIRIX's tolerated block, costs.json refusals and attempt.json
    accepted_learning_calls); None without a reported attempt (blocking)."""
    if cell.reported is None:
        return None
    if cell.tolerated_cache is None:
        cell.tolerated_cache = cell_records.tolerated_failures(
            cell.paths.attempt(cell.reported)
        )
    return cell.tolerated_cache


def task_type_breakdown(cell: CellData) -> dict[str, Any] | None:
    """ALFWorld success per task type (scorer-side label; this report only)."""
    if cell.benchmark != "alfworld":
        return None
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in cell.results:
        by_type[str(row.get("task_type"))].append(
            metric_value(row, METRICS["alfworld"][0])
        )
    return {
        task_type: {"items": len(values), "success_rate": mean(values)}
        for task_type, values in sorted(by_type.items())
    }


def failure_summary(cell: CellData) -> dict[str, Any]:
    reported = cell.attempt(cell.reported)
    counts = Counter(
        (str(f.get("phase")), str(f.get("kind")), str(f.get("code")))
        for f in cell.failures
    )
    ledger = (reported.ledger if reported else None) or {}
    return {
        "failures": [
            {"phase": p, "kind": k, "code": c, "count": n}
            for (p, k, c), n in sorted(counts.items())
        ],
        "test_reruns": len(
            (reported.state if reported else {}).get("test_reruns") or []
        ),
        "invalid_attempts": sum(
            1 for a in cell.attempts if a.state.get("state") == "invalid"
        ),
        "superseded_attempts": [
            a.number for a in cell.attempts if attempt_superseded(a.state)
        ],
        "repeat_of": ledger.get("repeat_of") or {},
        "gateway_statuses": ledger.get("statuses") or {},
    }


#: The refusal counters each learning method writes to its frozen snapshot, only
#: when above 0: a missing one reads 0.
METHOD_REFUSAL_COUNTERS: Mapping[str, tuple[str, ...]] = {
    "awm": ("induction_rejected", "induction_retries", "judge_rejected"),
    "reasoningbank": ("extraction_rejected", "judge_rejected"),
}
#: How many refused inductions the report names per cell.
REFUSAL_EXAMPLES = 5


def method_refusals(cell: CellData) -> dict[str, Any] | None:
    """A learning method's calls the provider refused (HTTP 400/404/413), counted and
    gone on from (record, don't stop): the frozen snapshot's
    counters through ``cell_records.tolerated_failures`` (a missing counter is 0), and
    for AWM the refused inductions (``snapshot.inductions[*].rejected``, a failure whose
    HTTP code is ``status``: the previous workflow was kept). None for a method without
    such counters."""
    keys = METHOD_REFUSAL_COUNTERS.get(cell.method)
    if keys is None:
        return None
    tolerated = cell_tolerated(cell) or {}
    counters = (tolerated.get("method") or {}).get("counters") or {}
    snapshot = (cell.frozen or {}).get("snapshot") or {}
    inductions = [i for i in snapshot.get("inductions") or () if isinstance(i, Mapping)]
    rejected = [i for i in inductions if isinstance(i.get("rejected"), Mapping)]
    return {
        "counters": {k: _count(counters.get(k)) for k in keys},
        "inductions": len(inductions) if cell.method == "awm" else None,
        "rejected_by_status": dict(
            sorted(Counter(str(i["rejected"].get("status")) for i in rejected).items())
        ),
        "rejected_examples": [
            {
                "index": i.get("index"),
                "trigger": i.get("trigger"),
                "status": i["rejected"].get("status"),
                "code": i["rejected"].get("code"),
                "detail": str(i["rejected"].get("detail") or "")[:200],
                "provider": i.get("provider"),
                "request_bytes": i.get("request_bytes"),
                "retried": isinstance(i.get("retry"), Mapping),
            }
            for i in rejected[:REFUSAL_EXAMPLES]
        ],
    }


def degraded_items(cell: CellData) -> dict[str, Any]:
    """The items scored under a degraded condition, from the published rows' flags:
    ``provider_unserved`` (the actor's provider refused the request: a test item after
    two consecutive no-endpoint 404 refusals, a learning item after one refusal; scored
    unanswered) and ``retrieval_degraded`` (MIRIX's retrieval lost its topic extraction
    or query vector and was injected anyway). ``test``: the raw ids per flag; ``learning``:
    the count per flag. Empty mappings when none."""
    test = {
        flag: sorted(
            str(r.get("raw_id")) for r in cell.results if _row_flagged(r, flag)
        )
        for flag in DEGRADED_FLAGS
    }
    learning = {
        flag: sum(1 for r in cell.learning if _row_flagged(r, flag))
        for flag in DEGRADED_FLAGS
    }
    return {
        "test": {k: v for k, v in test.items() if v},
        "learning": {k: v for k, v in learning.items() if v},
    }


def method_counters(cell: CellData) -> dict[str, Any] | None:
    snapshot = (cell.frozen or {}).get("snapshot") or {}
    out = {
        key: snapshot[key]
        for key in ("counters", "judge")
        if isinstance(snapshot.get(key), Mapping)
    }
    return out or None


def cell_host(cell: CellData) -> dict[str, Any] | None:
    """The host the reported attempt ran on (``attempt.json`` ``host``), None when it
    records none (an attempt started before hosts were recorded)."""
    reported = cell.attempt(cell.reported)
    host = (reported.state if reported else {}).get("host")
    if not isinstance(host, Mapping):
        return None
    return {k: host.get(k) for k in ("hostname", "machine_id", "system", "machine")}


def cell_code(cell: CellData) -> dict[str, Any]:
    """The code the reported attempt ran (``attempt.json`` ``code`` and ``git``, like
    :func:`cell_host`): the harness and MIRIX digests (``protocol.code_fingerprint``) and
    the commit; None where it records none (versions are reported per cell)."""
    reported = cell.attempt(cell.reported)
    state = reported.state if reported else {}
    code = state.get("code") if isinstance(state.get("code"), Mapping) else {}
    git = state.get("git") if isinstance(state.get("git"), Mapping) else {}
    return {
        "harness": code.get("harness"),
        "mirix": code.get("mirix"),
        "commit": git.get("commit"),
    }


def cell_agent_context(cell: CellData) -> dict[str, Any] | None:
    """The agent context the reported attempt recorded (``attempt.json``
    ``agent_context``: its ``sha256`` and ``chars``, ``envs/agent_context.
    context_record``), None for an attempt that records none."""
    reported = cell.attempt(cell.reported)
    state = reported.state if reported else {}
    record = state.get("agent_context")
    if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str):
        return None
    return {"sha256": record["sha256"], "chars": record.get("chars")}


def cell_protocol(
    cell: CellData,
    *,
    frozen: str | None = None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The protocol the reported attempt ran under (``protocol.attempt_protocol``: its
    ``attempt.json`` ``protocol``, else what ``cell.json`` attributes to it):
    ``fingerprint``, ``current`` (it is the frozen value) and ``known`` (a freeze produced
    it: the frozen value or an earlier one of ``versions``)."""
    reported = cell.attempt(cell.reported)
    recorded = protocol.attempt_protocol(
        cell.manifest, reported.state if reported else None, cell.reported
    )
    fingerprint = recorded["fingerprint"]
    current = frozen is not None and fingerprint == frozen
    return {
        "fingerprint": fingerprint,
        "current": current,
        "known": current or (bool(fingerprint) and fingerprint in (versions or {})),
    }


#: The host fields a version segment's host is shown with (``cell_records.host_info``).
SEGMENT_HOST_FIELDS: tuple[str, ...] = (
    "hostname",
    "machine_id",
    "system",
    "machine",
    "platform",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _file_changes(value: Any) -> dict[str, list[str]] | None:
    """``{changed, added, removed}`` of a recorded file comparison, None without one."""
    if not isinstance(value, Mapping):
        return None
    return {
        k: [str(p) for p in value.get(k) or ()] for k in ("changed", "added", "removed")
    }


def _trim_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    """One ``attempt.json`` ``version_segments`` entry as the report lists it."""
    proto = _mapping(segment.get("protocol"))
    code = _mapping(segment.get("code"))
    git = _mapping(segment.get("git"))
    host = segment.get("host")
    changed = segment.get("changed")
    trimmed_changed: dict[str, Any] | None = None
    if isinstance(changed, Mapping):
        diff = _mapping(changed.get("protocol_diff"))
        code_files = _mapping(changed.get("code_files"))
        trimmed_changed = {
            "protocol": bool(changed.get("protocol")),
            "code": [str(k) for k in changed.get("code") or ()],
            "commit": bool(changed.get("commit")),
            "tree": bool(changed.get("tree")),
            "protocol_files": _file_changes(diff) if diff else None,
            "constants_changed_keys": list(diff.get("constants_changed_keys") or ())
            if diff.get("constants_changed_keys") is not None
            else None,
            "code_files": {
                str(part): _file_changes(files)
                for part, files in sorted(code_files.items())
            }
            if code_files
            else None,
        }
    auto_freeze = segment.get("auto_freeze")
    return {
        "segment": segment.get("index"),
        "legacy": bool(segment.get("legacy")),
        "reason": segment.get("why"),
        "phase": segment.get("phase"),
        "at": segment.get("started_at"),
        "from": segment.get("learning_from") or segment.get("test_from"),
        "protocol_fingerprint": proto.get("fingerprint"),
        "constants_sha256": proto.get("constants_sha256"),
        "harness": code.get("harness"),
        "mirix": code.get("mirix"),
        "commit": git.get("commit"),
        "dirty": git.get("dirty"),
        "host": {k: host.get(k) for k in SEGMENT_HOST_FIELDS}
        if isinstance(host, Mapping)
        else None,
        "known_version": segment.get("known_version"),
        "version_file": segment.get("version_file"),
        "auto_freeze": auto_freeze.get("status")
        if isinstance(auto_freeze, Mapping)
        else auto_freeze,
        "note": segment.get("note"),
        "changed": trimmed_changed,
    }


def cell_segments(cell: CellData) -> list[dict[str, Any]]:
    """The versions the reported attempt ran under, one entry per segment
    (``protocol.attempt_versions``: ``attempt.json`` ``version_segments``, or segments
    synthesized from the attempt's work lines for an attempt written before them;
    blocking). Empty without a reported attempt."""
    reported = cell.attempt(cell.reported)
    if reported is None:
        return []
    work = [r for r in read_jsonl(cell.paths.launches) if r.get("event") == LAUNCH_WORK]
    segments = protocol.attempt_versions(
        cell.manifest, reported.state, reported.number, work
    )
    return [_trim_segment(s) for s in segments if isinstance(s, Mapping)]


def _segment_known(
    cell: CellData,
    segment: Mapping[str, Any],
    frozen: str | None,
    versions: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Whether a freeze registered a segment's protocol: the frozen value, a known
    version, the segment's own record (``known_version``), or the cell's
    ``version.json`` when it reproduces its fingerprint (``protocol.recorded_version``: a
    version auto-frozen on another host)."""
    fingerprint = segment.get("protocol_fingerprint")
    if not fingerprint:
        return False
    if fingerprint == frozen or fingerprint in versions or segment.get("known_version"):
        return True
    relative = segment.get("version_file")
    if not relative:
        return False
    try:
        document = read_json(cell.paths.root / str(relative))
    except (OSError, ValueError):
        return False
    if not isinstance(document, Mapping):
        return False
    recorded = protocol.recorded_version(document)
    return isinstance(recorded, Mapping) and recorded.get("fingerprint") == fingerprint


def cell_version_record(
    cell: CellData,
    *,
    frozen: str | None = None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """A cell's ``versions``: its segments (:func:`cell_segments`), whether it ran more
    than one protocol or code version (``mixed``: ``attempt.json`` ``versions_mixed``
    protocol or code; a commit or tree change alone is listed but not mixed), the
    distinct protocol fingerprints and harness digests, and the fingerprints no freeze
    registered (``unknown_fingerprints``)."""
    segments = cell_segments(cell)
    reported = cell.attempt(cell.reported)
    recorded = _mapping((reported.state if reported else {}).get("versions_mixed"))
    if recorded:
        mixed_detail = {
            k: bool(recorded.get(k)) for k in ("protocol", "code", "commit")
        }
    else:
        changes = [_mapping(s.get("changed")) for s in segments]
        mixed_detail = {
            "protocol": any(c.get("protocol") for c in changes),
            "code": any(c.get("code") for c in changes),
            "commit": any(c.get("commit") or c.get("tree") for c in changes),
        }
    fingerprints = sorted(
        {str(s["protocol_fingerprint"]) for s in segments if s["protocol_fingerprint"]}
    )
    known = versions or {}
    checked = frozen is not None or versions is not None
    return {
        "segments": segments,
        "mixed": bool(mixed_detail["protocol"] or mixed_detail["code"]),
        "mixed_detail": mixed_detail,
        "protocol_fingerprints": fingerprints,
        "harnesses": sorted({str(s["harness"]) for s in segments if s["harness"]}),
        "unknown_fingerprints": sorted(
            {
                str(s["protocol_fingerprint"])
                for s in segments
                if s["protocol_fingerprint"]
                and not _segment_known(cell, s, frozen, known)
            }
        )
        if checked
        else [],
    }


def cell_hosts(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The distinct hosts the reported attempt ran on, over its segments, each with its
    platform string and the segments it ran."""
    found: dict[str, dict[str, Any]] = {}
    for seg in segments:
        host = seg.get("host")
        if not isinstance(host, Mapping):
            continue
        key = host_key(host)
        entry = found.setdefault(key, {**host, "segments": []})
        entry["segments"].append(seg.get("segment"))
    return list(found.values())


def cell_versions(
    cell: CellData,
    *,
    frozen: str | None = None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A cell's protocol and code in one mapping, for the comparison rows: the
    attempt-start protocol and code, and over its segments whether it is ``mixed`` and
    every protocol fingerprint it ran (``record``: :func:`cell_version_record`, computed
    when not given)."""
    recorded = cell_protocol(cell, frozen=frozen, versions=versions)
    record = (
        record
        if record is not None
        else cell_version_record(cell, frozen=frozen, versions=versions)
    )
    return {
        "protocol": recorded["fingerprint"],
        "protocol_current": recorded["current"],
        **cell_code(cell),
        "mixed": bool(record.get("mixed")),
        "protocol_fingerprints": list(record.get("protocol_fingerprints") or ()),
    }


def _same(a: Any, b: Any) -> bool | None:
    """Whether two recorded versions are equal; None when either is not recorded."""
    return None if a is None or b is None else a == b


def cell_resumes(cell: CellData) -> list[dict[str, Any]]:
    """The learning resumes of the reported attempt, one entry per discarded segment
    (``AttemptData.discarded``)."""
    reported = cell.attempt(cell.reported)
    return list(reported.discarded) if reported else []


def cell_section(
    cell: CellData,
    *,
    frozen: str | None = None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    recorded = cell_protocol(cell, frozen=frozen, versions=versions)
    record = cell_version_record(cell, frozen=frozen, versions=versions)
    section: dict[str, Any] = {
        "cell_id": cell.cell_id,
        # The directory's entry name under the runs directory (a merged view's link
        # name, '<entry>@<host>', for a second copy of one cell id): never resolved.
        "cell_dir": cell.paths.root.name,
        "benchmark": cell.benchmark,
        "model": cell.model,
        "model_key": cell.model_key,
        "method": cell.method,
        "split_mode": cell.split_mode,
        "group": cell.group,
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "attempts": len(cell.attempts),
        "newer_attempt": cell.newer_attempt,
        "record_problems": list(cell.record_problems),
        # The reported attempt's own versions: the protocol it started under, and every
        # version segment it ran.
        "protocol_fingerprint": recorded["fingerprint"],
        "protocol": recorded,
        "code": cell_code(cell),
        "versions": record,
        "host": cell_host(cell),
        "hosts": cell_hosts(record["segments"]),
        "resumes": cell_resumes(cell),
        "costs": cell_costs(cell),
        "providers": cell_providers(cell),
        "failures": failure_summary(cell),
        "tolerated": cell_tolerated(cell),
    }
    if cell.complete:
        section.update(
            test=test_metrics(cell),
            learning_items=len(cell.learning),
            learning_curve=learning_curve(cell),
            injections=cell_injections(cell),
            judge_agreement=judge_agreement(cell),
            mirix=mirix_diagnostics(cell),
            task_types=task_type_breakdown(cell),
            method_counters=method_counters(cell),
            method_refusals=method_refusals(cell),
            degraded=degraded_items(cell),
        )
    return section


# ---------------------------------------------------------------------------
# Comparisons with the no-memory method
# ---------------------------------------------------------------------------


def compare(
    method: CellData,
    baseline: CellData,
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.SEED,
) -> list[dict[str, Any]]:
    """Paired comparison of ``method`` with ``baseline`` on every metric of the
    benchmark."""
    ours = {str(r["raw_id"]): r for r in method.results}
    theirs = {str(r["raw_id"]): r for r in baseline.results}
    if set(ours) != set(theirs):
        return [
            {
                "benchmark": method.benchmark,
                "group": method.group,
                "method": method.method,
                "skipped": "the two cells tested different items",
            }
        ]
    items = sorted(ours)
    out = []
    for metric in METRICS.get(method.benchmark, ()):
        if not items:
            out.append(
                {
                    "benchmark": method.benchmark,
                    "group": method.group,
                    "method": method.method,
                    "metric": metric.name,
                    "skipped": "neither cell scored a test item",
                }
            )
            continue
        a = [metric_value(ours[i], metric) for i in items]
        b = [metric_value(theirs[i], metric) for i in items]
        entry: dict[str, Any] = {
            "benchmark": method.benchmark,
            "model": method.model,
            "group": method.group,
            "method": method.method,
            "metric": metric.name,
            "main": metric.main,
            "items": len(items),
            "method_value": mean(a),
            "none_value": mean(b),
            "delta": (mean(a) or 0.0) - (mean(b) or 0.0),
        }
        if metric.binary:
            wins = sum(1 for x, y in zip(a, b) if x == 1.0 and y == 0.0)
            losses = sum(1 for x, y in zip(a, b) if x == 0.0 and y == 1.0)
            entry.update(
                test="mcnemar_exact",
                b=wins,
                c=losses,
                p_value=mcnemar_exact(wins, losses),
            )
        else:
            entry.update(
                test="paired_bootstrap",
                **{
                    k: v
                    for k, v in paired_bootstrap(
                        [x - y for x, y in zip(a, b)], resamples=resamples, seed=seed
                    ).items()
                    if k not in ("delta", "n")
                },
            )
        out.append(entry)
    return out


def _parse_time(value: Any) -> datetime:
    """An ISO timestamp as an aware datetime; the earliest time when missing or
    unreadable (missing times sort lowest)."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _key_attempt(cell: CellData) -> AttemptData | None:
    """The attempt a directory is dated by: the reported one of a complete
    directory, else its last attempt."""
    if cell.complete:
        return cell.attempt(cell.reported)
    return cell.attempts[-1] if cell.attempts else None


def directory_started_at(cell: CellData) -> str | None:
    """The ``started_at`` of the attempt a directory is dated by (:func:`_key_attempt`)."""
    attempt = _key_attempt(cell)
    value = attempt.state.get("started_at") if attempt else None
    return str(value) if value else None


def _directory_rank(cell: CellData) -> tuple[datetime, datetime, str]:
    """(started, created, name): the order of directories of one cell id after
    completeness."""
    return (
        _parse_time(directory_started_at(cell)),
        _parse_time(cell.manifest.get("created_at")),
        cell.paths.root.name,
    )


def select_reported(
    cells: Sequence[CellData],
) -> tuple[list[CellData], list[tuple[CellData, CellData, str]]]:
    """One directory per cell id (the audit's rule): among the directories holding
    one ``cell.json`` cell id, the chosen one is the max by (complete, the dated
    attempt's ``started_at``, ``cell.json`` ``created_at``, directory name), missing
    times lowest. Returns ``(chosen, not_used)``; ``not_used`` holds
    ``(directory, chosen directory, role)``: ``superseded`` when it sorts below the chosen
    one by (started, created, name), else ``newer_incomplete`` (a newer run still going).
    Both keep the input order."""
    by_id: dict[str, list[CellData]] = defaultdict(list)
    for cell in cells:
        by_id[cell.cell_id].append(cell)
    chosen: dict[str, CellData] = {
        cell_id: max(found, key=lambda c: (c.complete, *_directory_rank(c)))
        for cell_id, found in by_id.items()
    }
    kept = [c for c in cells if chosen[c.cell_id] is c]
    others = []
    for cell in cells:
        winner = chosen[cell.cell_id]
        if winner is cell:
            continue
        role = (
            ROLE_SUPERSEDED
            if _directory_rank(cell) < _directory_rank(winner)
            else ROLE_NEWER_INCOMPLETE
        )
        others.append((cell, winner, role))
    return kept, others


def duplicate_cells(cells: Sequence[CellData]) -> list[dict[str, Any]]:
    """Complete cells of distinct cell ids that claim one (group, method) (smoke cells
    of different ``n_train``): a warning each; :func:`comparisons` compares the one
    whose reported attempt started last and names the others."""
    by_key: dict[tuple[str, str], list[CellData]] = defaultdict(list)
    for cell in cells:
        if cell.complete:
            by_key[(cell.group, cell.method)].append(cell)
    out = []
    for (group, method), found in sorted(by_key.items()):
        if len(found) < 2:
            continue
        used = _group_choice(found)
        out.append(
            {
                "check": "duplicate_group_method",
                "group": group,
                "method": method,
                "used": used.cell_id,
                "cells": sorted(c.cell_id for c in found),
                "detail": f"{len(found)} complete {method} cells in group {group} "
                f"(same benchmark, model and test items): {used.cell_id} is compared "
                "(its reported attempt started last); "
                + ", ".join(sorted(c.cell_id for c in found if c is not used))
                + " not compared",
            }
        )
    return out


def _group_choice(found: Sequence[CellData]) -> CellData:
    """The cell compared among complete cells of one (group, method): the newest
    reported attempt, then the cell id (deterministic)."""
    return max(found, key=lambda c: (*_directory_rank(c)[:2], c.cell_id))


def _not_comparable(
    method: CellData,
    baseline: CellData,
    *,
    audits: Mapping[str, Mapping[str, Any]] | None,
) -> str | None:
    """Why ``method`` must not be compared with ``baseline``, or None, every reason
    that applies, joined. Different protocol or code versions are no reason, not even a
    version no freeze registered (such a comparison is made and marked,
    ``same_protocol`` ``unknown``; :func:`comparisons`). With ``audits``
    (``--require-audit``) only cells whose audit passed."""
    reasons = []
    if audits is not None:
        failing = [
            f"{c.cell_id} ({audits.get(c.cell_id, {}).get('status', AUDIT_MISSING)})"
            for c in (method, baseline)
            if audits.get(c.cell_id, {}).get("status") != AUDIT_PASSED
        ]
        if failing:
            reasons.append("no passing audit: " + ", ".join(failing))
    return "; ".join(reasons) or None


def _platforms(record: Mapping[str, Any]) -> set[tuple[Any, Any]]:
    return {
        (s["host"].get("system"), s["host"].get("machine"))
        for s in record.get("segments") or ()
        if isinstance(s.get("host"), Mapping)
    }


def _same_protocol(
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    unknown: Sequence[str],
) -> bool | str | None:
    """``unknown`` when a freeze registered neither version of a cell's protocol; else
    True only when each cell ran one protocol fingerprint and they are equal (None: one
    not recorded)."""
    if unknown:
        return "unknown"
    a = list(ours.get("protocol_fingerprints") or ()) or (
        [ours["protocol"]] if ours.get("protocol") else []
    )
    b = list(theirs.get("protocol_fingerprints") or ()) or (
        [theirs["protocol"]] if theirs.get("protocol") else []
    )
    if not a or not b:
        return None
    return len(a) == 1 and a == b


def comparisons(
    cells: Sequence[CellData],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.SEED,
    frozen: str | None = None,
    audits: Mapping[str, Mapping[str, Any]] | None = None,
    versions: Mapping[str, Mapping[str, Any]] | None = None,
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Paired comparisons with the no-memory cell of each group; with ``audits``
    (``--require-audit``), only of cells whose audit passed on these very records.
    ``cells`` holds one directory per cell id (:func:`select_reported`). Among complete
    cells of distinct ids that claim one (group, method), the one whose reported attempt
    started last is compared and each other gets a skipped row naming it.

    Cells of different protocol or code versions are compared:
    every row of a pair carries ``method_versions`` and ``none_versions``
    (:func:`cell_versions`) and ``same_protocol`` (``unknown`` when a freeze registered
    neither of a cell's protocol versions, with ``frozen`` or ``versions`` given;
    ``protocol_unknown`` names the cells), ``same_code`` (the harness digest; None when
    one is not recorded), ``mixed_version`` (the cells that ran more than one version)
    and ``same_platform`` (the hosts' system and machine; None when unrecorded).
    ``records`` maps a cell id to its :func:`cell_version_record` (computed when
    missing)."""
    by_key: dict[tuple[str, str], list[CellData]] = defaultdict(list)
    for cell in cells:
        if cell.complete:
            by_key[(cell.group, cell.method)].append(cell)
    by_group: dict[str, dict[str, CellData]] = defaultdict(dict)
    shadowed: list[tuple[str, str, CellData, CellData]] = []
    for (group, method), found in by_key.items():
        used = _group_choice(found)
        by_group[group][method] = used
        shadowed += [(group, method, c, used) for c in found if c is not used]
    checked = frozen is not None or versions is not None

    def record_of(cell: CellData) -> Mapping[str, Any]:
        if records is not None and cell.cell_id in records:
            return records[cell.cell_id]
        return cell_version_record(cell, frozen=frozen, versions=versions)

    out: list[dict[str, Any]] = []
    for group in sorted(by_group):
        methods = by_group[group]
        baseline = methods.get("none")
        for name in protocol.METHODS:
            if name == "none" or name not in methods:
                continue
            if baseline is None:
                out.append(
                    {
                        "group": group,
                        "method": name,
                        "skipped": "no complete no-memory cell in this group",
                    }
                )
                continue
            why = _not_comparable(methods[name], baseline, audits=audits)
            if why is not None:
                out.append({"group": group, "method": name, "skipped": why})
                continue
            pair = (methods[name], baseline)
            recs = [record_of(c) for c in pair]
            ours = cell_versions(
                methods[name], frozen=frozen, versions=versions, record=recs[0]
            )
            theirs = cell_versions(
                baseline, frozen=frozen, versions=versions, record=recs[1]
            )
            unknown = (
                [
                    c.cell_id
                    for c, rec in zip(pair, recs)
                    if not cell_protocol(c, frozen=frozen, versions=versions)["known"]
                    or rec.get("unknown_fingerprints")
                ]
                if checked
                else []
            )
            platforms = [_platforms(rec) for rec in recs]
            flags = {
                "method_versions": ours,
                "none_versions": theirs,
                "same_protocol": _same_protocol(ours, theirs, unknown),
                "protocol_unknown": unknown,
                "same_code": _same(ours["harness"], theirs["harness"]),
                "mixed_version": [
                    c.cell_id for c, rec in zip(pair, recs) if rec.get("mixed")
                ],
                "same_platform": None
                if not platforms[0] or not platforms[1]
                else (len(platforms[0]) == 1 and platforms[0] == platforms[1]),
            }
            out.extend(
                {**row, **flags}
                for row in compare(
                    methods[name], baseline, resamples=resamples, seed=seed
                )
            )
    for group, method, cell, used in sorted(
        shadowed, key=lambda s: (s[0], s[1], s[2].cell_id)
    ):
        out.append(
            {
                "group": group,
                "method": method,
                "cell_id": cell.cell_id,
                "skipped": "a newer complete cell of this group and method is "
                f"compared: {used.cell_id}",
            }
        )
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _frozen_fingerprint() -> str | None:
    try:
        return protocol.frozen_fingerprint()
    except protocol.ProtocolDriftError:
        return None


def _known_versions() -> dict[str, dict[str, Any]]:
    try:
        return protocol.known_versions()
    except protocol.ProtocolDriftError:
        return {}


def version_groups(sections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each benchmark's cells per version: ``protocol``
    (fingerprint -> cells), ``harness`` and ``mirix`` (digest -> cells; a cell that
    records none is left out). A cell that ran more than one version (its version
    segments) is listed under each."""
    groups: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: {
            "protocol": defaultdict(set),
            "harness": defaultdict(set),
            "mirix": defaultdict(set),
        }
    )
    for section in sections:
        benchmark = str(section["benchmark"])
        cell_id = str(section["cell_id"])
        values: dict[str, set[Any]] = {
            "protocol": {(section.get("protocol") or {}).get("fingerprint")},
            "harness": {(section.get("code") or {}).get("harness")},
            "mirix": {(section.get("code") or {}).get("mirix")},
        }
        for segment in (section.get("versions") or {}).get("segments") or ():
            values["protocol"].add(segment.get("protocol_fingerprint"))
            values["harness"].add(segment.get("harness"))
            values["mirix"].add(segment.get("mirix"))
        for key, found in values.items():
            for value in found:
                if value:
                    groups[benchmark][key][str(value)].add(cell_id)
    return {
        benchmark: {
            key: {v: sorted(c) for v, c in sorted(variants.items())}
            for key, variants in kinds.items()
        }
        for benchmark, kinds in sorted(groups.items())
    }


def mixed_version_benchmarks(groups: Mapping[str, Any]) -> list[str]:
    """The benchmarks whose cells ran more than one protocol or code version."""
    return [
        benchmark
        for benchmark, kinds in groups.items()
        if any(len(variants) > 1 for variants in kinds.values())
    ]


def caveat_mixed_versions(benchmarks: Sequence[str]) -> str:
    """:data:`CAVEAT_MIXED_VERSIONS`, naming the benchmarks it concerns."""
    return f"{CAVEAT_MIXED_VERSIONS} Mixed here: {', '.join(benchmarks)}."


def load_audit(path: Path) -> dict[str, Any] | None:
    """``audit.json`` as ``audit.py`` wrote it, None when there is none (blocking)."""
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        return {"unreadable": f"{type(exc).__name__}: {exc}"}
    if data is None:
        return None
    return data if isinstance(data, dict) else {"unreadable": "not a JSON object"}


def caveat_reruns(cell_ids: Sequence[str]) -> str:
    """The single-run caveat's exception for cells that ran more than once."""
    return (
        "Not every cell ran once: "
        + ", ".join(cell_ids)
        + " ran again after an earlier run of the cell had run test items (see 'Runs "
        "and attempts'). Each reports its newest complete run; the other runs are "
        "listed with their scores in Runs and attempts, and no score combines two runs. "
        "The earlier run's scores could be seen before the newer one."
    )


def _audit_entries(
    cell: CellData, audit: Mapping[str, Any] | None
) -> list[Mapping[str, Any]]:
    """The audit's entries for this very cell directory (cell id and the root's name)."""
    return [
        e
        for e in (audit or {}).get("cells") or ()
        if isinstance(e, Mapping)
        and e.get("cell_id") == cell.cell_id
        and Path(str(e.get("root") or "")).name == cell.paths.root.name
    ]


def _audit_duplicate(
    cell: CellData, audit: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """The audit's ``duplicates`` entry of this directory (audit.json schema 2: a
    directory of a cell id another directory reports), None when it lists none."""
    for entry in (audit or {}).get("duplicates") or ():
        if (
            isinstance(entry, Mapping)
            and entry.get("cell_id") == cell.cell_id
            and Path(str(entry.get("root") or "")).name == cell.paths.root.name
        ):
            return entry
    return None


def audit_schema2(audit: Mapping[str, Any] | None) -> bool:
    """Whether ``audit`` is an audit.json of schema 2 or later (it declares its
    ``validity_checks``): run-level violations then name the cells they fail."""
    return isinstance(audit, Mapping) and "validity_checks" in audit


def cell_reruns(
    cell: CellData,
    audit: Mapping[str, Any] | None,
    *,
    attempts: Sequence[Mapping[str, Any]] = (),
    directories: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Whether the cell ran more than once (the single-run exception), None when nothing says
    so: the operator notes in ``cell.json`` (``--acknowledge-rerun``), the earlier
    gateway runs of the cell id that ran test items as the audit found them, the
    attempts of this directory that scored test items and are not reported
    (``attempts``: :func:`cell_attempts` entries; a superseded attempt), and the other
    directories of the cell id (``directories``). A cell started over before any
    earlier run reached a test item ran once, note or not."""
    notes = [
        {
            "file": n.get("file"),
            "sha256": n.get("sha256"),
            "at": n.get("at"),
            "superseded": [
                {k: r.get(k) for k in ("run_id", "instance", "test_actor_calls")}
                for r in n.get("superseded") or ()
                if isinstance(r, Mapping)
            ],
        }
        for n in cell.manifest.get("operator_notes") or ()
        if isinstance(n, Mapping) and n.get("kind") == NOTE_RERUN_ACKNOWLEDGED
    ]
    entries = _audit_entries(cell, audit)
    info = (entries[0].get("info") or {}) if len(entries) == 1 else {}
    earlier = [
        {
            k: r.get(k)
            for k in ("run_id", "instance", "test_actor_calls", "first_test_actor_at")
        }
        for r in info.get("other_gateway_runs") or ()
        if isinstance(r, Mapping) and _count(r.get("test_actor_calls")) > 0
    ]
    noted = any(
        _count(r.get("test_actor_calls")) > 0 for n in notes for r in n["superseded"]
    )
    tested = [
        dict(a)
        for a in attempts
        if not a.get("reported") and _count(a.get("test_items_scored")) > 0
    ]
    if not noted and not earlier and not tested and not directories:
        return None
    return {
        "operator_notes": notes,
        "earlier_tested_runs": earlier,
        "attempts_tested": tested,
        "directories": [dict(d) for d in directories],
    }


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def audit_status(cell: CellData, audit: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether ``audit`` vouches for the records this report prints for ``cell``:

    * ``passed``: the audit audited the cell's reported attempt, found no validity
      violation in it nor across cells, checked it, and saw the very
      ``results.jsonl``/``learning.jsonl`` bytes read here (``records_sha256``);
    * ``failed``: it found validity violations in the cell (with audit.json schema 2
      its own and those across cells that name it), or a run-level violation that names
      no cell (it fails every cell: fail closed). An audit.json of schema 1 counts every
      violation and every run-level violation fails every cell, as before;
    * ``not checked`` (schema 2): the auditing machine could not check it (no
      violation found);
    * ``stale``: it audited another attempt or other record bytes (the cell changed since);
    * ``superseded (not audited)`` / ``newer, not complete``: the audit set this
      directory aside for another directory of its cell id (its ``duplicates``);
    * ``not audited`` / ``no audit``: the audit skipped the cell, or there is none.

    Only ``passed`` passes ``--require-audit``."""
    if audit is None:
        return {"status": AUDIT_MISSING}
    if audit.get("unreadable"):
        return {"status": AUDIT_MISSING, "detail": f"audit.json: {audit['unreadable']}"}
    schema2 = audit_schema2(audit)
    entries = _audit_entries(cell, audit)
    if not entries:
        duplicate = _audit_duplicate(cell, audit)
        if duplicate is not None:
            chosen = Path(str(duplicate.get("chosen_root") or "")).name
            return {
                "status": AUDIT_SUPERSEDED
                if duplicate.get("role") == ROLE_SUPERSEDED
                else AUDIT_NEWER_INCOMPLETE,
                "detail": f"the audit reports the directory {chosen}",
            }
    if len(entries) != 1 or not entries[0].get("audited"):
        reason = entries[0].get("reason") if len(entries) == 1 else None
        return {"status": AUDIT_NOT_AUDITED, "detail": reason}
    entry = entries[0]
    legacy = (
        ""
        if schema2
        else " (an audit.json of schema 1: every violation counts; re-run audit.py)"
    )
    if entry.get("violations"):
        return {
            "status": AUDIT_FAILED,
            "detail": f"{entry['violations']} violation(s) in the cell{legacy}",
        }
    runs_level = sum(
        1
        for f in audit.get("findings") or ()
        if isinstance(f, Mapping)
        and f.get("severity") == "violation"
        and not (schema2 and f.get("cells"))
    )
    if runs_level:
        return {
            "status": AUDIT_FAILED,
            "detail": f"{runs_level} runs-level violation(s)"
            + (" naming no cell" if schema2 else "")
            + f" (see audit.json){legacy}",
        }
    if schema2 and _count(entry.get("not_checked")) > 0:
        return {
            "status": AUDIT_NOT_CHECKED,
            "detail": f"{_count(entry.get('not_checked'))} check(s) the auditing "
            "machine could not run (its items or environment are missing there)",
        }
    if not cell.complete or entry.get("attempt") != cell.reported:
        return {
            "status": AUDIT_STALE,
            "detail": f"the audit saw attempt {entry.get('attempt')}, the report "
            f"reads attempt {cell.reported}",
        }
    if (entry.get("info") or {}).get("records_sha256") != records_digest(cell.paths):
        return {
            "status": AUDIT_STALE,
            "detail": "the cell's results.jsonl/learning.jsonl changed after the audit",
        }
    return {"status": AUDIT_PASSED}


def _finding_key(finding: Mapping[str, Any]) -> str:
    return f"{finding.get('invariant')}:{finding.get('check')}"


def audit_detail(
    cell: CellData, audit: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """What the audit recorded about the cell besides its status (audit.json schema 2,
    ``info``): the checks it could not re-verify under an earlier version
    (``not_reverified``: ``{check, inputs, findings}``), the conditions it recorded
    instead of failing (``recorded_counts``: ``{"INV:check": n}``), its warnings per
    check (``warnings_by_check``, from the entry's findings), the version segments it
    saw after the first and the superseded attempts it listed. A schema-1 audit.json
    names its unauditable protocol versions as warnings: those are listed as not
    re-verified. None without an entry for this directory."""
    entries = _audit_entries(cell, audit)
    if len(entries) != 1:
        return None
    entry = entries[0]
    info = _mapping(entry.get("info"))
    findings = [f for f in entry.get("findings") or () if isinstance(f, Mapping)]
    not_reverified = [
        {
            "check": str(item.get("check")),
            "inputs": [str(i) for i in item.get("inputs") or ()],
            "findings": _count(item.get("findings")),
        }
        for item in info.get("not_reverified") or ()
        if isinstance(item, Mapping)
    ]
    if not audit_schema2(audit) and not not_reverified:
        not_reverified = [
            {"check": _finding_key(f), "inputs": [str(f.get("detail"))], "findings": 1}
            for f in findings
            if f.get("severity") == "warning"
            and f.get("check") == "protocol_version_unauditable"
        ]
    warnings = Counter(
        _finding_key(f) for f in findings if f.get("severity") == "warning"
    )
    return {
        "not_reverified": not_reverified,
        "recorded_counts": {
            str(k): _count(v)
            for k, v in sorted(_mapping(info.get("recorded_counts")).items())
        },
        "warnings_by_check": dict(sorted(warnings.items())),
        "not_checked": _count(entry.get("not_checked")),
        "version_segments": [
            dict(s)
            for s in info.get("version_segments") or ()
            if isinstance(s, Mapping)
        ],
        "superseded_attempts": [
            dict(s)
            for s in info.get("superseded_attempts") or ()
            if isinstance(s, Mapping)
        ],
    }


def matrix_costs(sections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Every reported cell's cost, complete or not: ``incomplete`` names the cells whose
    ledgers so far are part of the total (a running cell, one that stopped)."""
    by_role: defaultdict[str, float] = defaultdict(float)
    by_side: defaultdict[str, float] = defaultdict(float)
    by_benchmark: defaultdict[str, float] = defaultdict(float)
    by_method: defaultdict[str, float] = defaultdict(float)
    incomplete: dict[str, float] = {}
    total = 0.0
    invalid = 0.0
    superseded = 0.0
    discarded = 0.0
    unknown = unfinished = 0
    for section in sections:
        costs = section["costs"]
        total += float(costs["total_cost"])
        invalid += float(costs["other_attempts"]["cost"])
        superseded += float(costs["other_attempts"].get("superseded_cost") or 0.0)
        discarded += float((costs.get("learning_discarded") or {}).get("cost") or 0.0)
        unknown += int(costs.get("unknown_cost_calls") or 0)
        unfinished += int(costs.get("unfinished_calls") or 0)
        if not section.get("complete"):
            # A directory that is not reported is named with its directory
            # (``cost_label``): two directories of one cell id never collide here.
            label = str(section.get("cost_label") or section["cell_id"])
            incomplete[label] = round(float(costs["total_cost"]), 10)
        by_benchmark[section["benchmark"]] += float(costs["total_cost"])
        by_method[section["method"]] += float(costs["total_cost"])
        for side in ("learning", "test"):
            by_side[side] += float(costs[side]["cost"])
            for role, entry in costs[side]["by_role"].items():
                by_role[role] += float(entry["cost"])
    rounded = lambda c: {k: round(v, 10) for k, v in sorted(c.items())}  # noqa: E731
    return {
        "total": round(total, 10),
        "invalidated_attempts": round(invalid, 10),
        # Of which attempts superseded by a --new-attempt --note rerun.
        "superseded_attempts": round(superseded, 10),
        # Part of the learning side: tries a learning resume rolled back and redid.
        "learning_discarded": round(discarded, 10),
        "incomplete": dict(sorted(incomplete.items())),
        "unknown_cost_calls": unknown,
        # Calls without a final ledger line, cost unknown: not in the total either.
        "unfinished_calls": unfinished,
        "by_side": rounded(by_side),
        "by_role": rounded(by_role),
        "by_benchmark": rounded(by_benchmark),
        "by_method": rounded(by_method),
    }


def _number(value: Any) -> float | None:
    """A JSON number as float; None for anything else (booleans included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _ledger_cost_in_window(
    gateway_dir: Path,
    start: datetime | None,
    end: datetime | None,
    *,
    seen_removals: set[str] | None = None,
) -> dict[str, Any]:
    """``cost_total`` over every final call line of every ledger in ``gateway_dir`` (every
    run: invalidated attempts, cells of any split mode) that ended inside
    ``start``..``end`` (``completed_at``, else ``ts`` + ``latency_s``), and the rest.

    The ledgers of a removed method deleted from ``gateway_dir``
    (``tools/remove_method_data.py``: its ``_removed/`` manifests' ``calls.jsonl``) count
    as they did on disk; a run still on disk counts once, from its ledger (an
    interrupted deletion). ``seen_removals`` (a set the caller shares across
    directories) counts each removal once. Hidden entries (``.name``) are skipped."""
    ledger_cost = outside_cost = 0.0
    calls = outside = unknown_calls = 0
    live: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        nonlocal ledger_cost, outside_cost, calls, outside, unknown_calls
        cost = _number(row.get("cost_total")) or 0.0
        ended = call_completed_at(row)
        inside = (
            ended is not None
            and (start is None or ended >= start)
            and (end is None or ended <= end)
        )
        if not inside:
            outside += 1
            outside_cost += cost
            return
        calls += 1
        ledger_cost += cost
        unknown_calls += int(unknown_cost_attempts(row) > 0)

    for child in sorted(gateway_dir.iterdir()) if gateway_dir.is_dir() else ():
        ledger = child / "gateway.jsonl"
        if (
            child.name in NON_CELL_DIRS
            or child.name.startswith(".")
            or not ledger.is_file()
        ):
            continue
        live.add(child.name)
        for row in read_jsonl(ledger):
            if row.get("event") == "call":
                add(row)
    for manifest in load_manifests(gateway_dir, hosts=False):
        if manifest.problems:
            continue  # unreadable: named by the report's warnings, its money unknown
        if seen_removals is not None:
            if manifest.removal_id in seen_removals:
                continue
            seen_removals.add(manifest.removal_id)
        for row in manifest_calls(manifest):
            if (
                row.get("event", "call") == "call"
                and str(row.get("run_id")) not in live
            ):
                add(row)
    return {
        "ledger_cost_usd": ledger_cost,
        "ledger_calls": calls,
        "unknown_cost_calls": unknown_calls,
        "calls_outside_window": outside,
        "cost_outside_window_usd": outside_cost,
    }


def _balance_rows(gateway_dir: Path) -> list[dict[str, Any]]:
    return [r for r in read_jsonl(gateway_dir / BALANCE_FILE) if isinstance(r, dict)]


def gateway_spend(gateway_dir: Path) -> dict[str, Any] | None:
    """What the key spent while a gateway served ``gateway_dir``, against what its
    ledgers recorded (blocking). None without a balance record.

    * ``key_usage_delta_usd``: the key's usage growth, from the last balance row: its
      ``budget.spent_usd`` (since the runs directory's budget started,
      ``budget.started_at``); without a budget, its ``usage`` less the first row's;
    * ``ledger_cost_usd``: ``cost_total`` over every final call line of every ledger in
      ``gateway_dir`` (every run: invalidated attempts, cells of any split mode) that
      ended inside that window (``completed_at``, else ``ts`` + ``latency_s``);
      ``calls_outside_window`` / ``cost_outside_window_usd`` are the rest;
    * ``unattributed_usd``: the difference, and ``unknown_cost_calls``: the calls in the
      window whose cost is unknown (:func:`~cell_records.unknown_cost_attempts`).
      See :data:`CAVEAT_UNATTRIBUTED`.

    ``budget_start`` (``start_usage_usd``, ``started_at``) is kept, so hosts that share
    one budget start (:func:`union_spend`) can be told apart from hosts that do not.
    """
    rows = _balance_rows(gateway_dir)
    if not rows:
        return None
    last = rows[-1]
    budget = last.get("budget") if isinstance(last.get("budget"), Mapping) else {}
    spent = _number(budget.get("spent_usd"))
    source = "budget.spent_usd"
    start_ts = budget.get("started_at")
    if spent is None:
        first, final = _number(rows[0].get("usage")), _number(last.get("usage"))
        if first is None or final is None:
            return None
        spent, source, start_ts = final - first, "usage", rows[0].get("ts")
    start = call_completed_at({"ts": start_ts}) if start_ts else None
    end = call_completed_at({"ts": last.get("ts")})
    window = _ledger_cost_in_window(gateway_dir, start, end)
    return {
        "gateway_dir": str(gateway_dir),
        "source": source,
        "window_start": start_ts,
        "window_end": last.get("ts"),
        "key_usage_delta_usd": round(spent, 6),
        "ledger_cost_usd": round(window["ledger_cost_usd"], 10),
        "ledger_calls": window["ledger_calls"],
        "unattributed_usd": round(spent - window["ledger_cost_usd"], 6),
        "unknown_cost_calls": window["unknown_cost_calls"],
        "calls_outside_window": window["calls_outside_window"],
        "cost_outside_window_usd": round(window["cost_outside_window_usd"], 10),
        "budget_start": {
            "start_usage_usd": _number(budget.get("start_usage_usd")),
            "started_at": budget.get("started_at"),
        }
        if source == "budget.spent_usd"
        else None,
        "usage_usd": _number(last.get("usage")),
    }


def union_spend(spends: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The unattributed spend over several gateway runs directories that use one key
    (two hosts; blocking). None for fewer than two.

    Each directory's own figure (:func:`gateway_spend`) counts the key's whole usage
    growth, so it includes what the other directories' gateways spent. With one shared
    budget start (the second host started from a copy of the first host's
    ``_gateway/budget.json``: equal ``start_usage_usd`` and ``started_at``), the union is
    computed once: the key's usage at the latest balance row of any directory less the
    shared start, against the ledgers of every directory over the calls that ended from
    the start to that row (``shared: true``). Without one, no union can be computed
    (``shared: false``, and each directory's figure includes the others' spend)."""
    if len(spends) < 2:
        return None
    starts = {
        (s["budget_start"]["start_usage_usd"], s["budget_start"]["started_at"])
        if isinstance(s.get("budget_start"), Mapping)
        else None
        for s in spends
    }
    dirs = [str(s["gateway_dir"]) for s in spends]
    if len(starts) != 1 or None in starts:
        return {
            "shared": False,
            "gateway_dirs": dirs,
            "detail": "the directories do not share one budget start: each one's "
            "unattributed spend includes the others' spend",
        }
    ((start_usage, started_at),) = starts
    latest = max(
        spends,
        key=lambda s: call_completed_at({"ts": s["window_end"]})
        or datetime.min.replace(tzinfo=timezone.utc),
    )
    usage = _number(latest.get("usage_usd"))
    if usage is None or start_usage is None:
        return {
            "shared": True,
            "gateway_dirs": dirs,
            "detail": "the latest balance row carries no usage",
        }
    start = call_completed_at({"ts": started_at}) if started_at else None
    end = call_completed_at({"ts": latest["window_end"]})
    seen: set[str] = set()
    windows = [
        _ledger_cost_in_window(Path(d), start, end, seen_removals=seen) for d in dirs
    ]
    ledger_cost = sum(w["ledger_cost_usd"] for w in windows)
    spent = usage - start_usage
    return {
        "shared": True,
        "gateway_dirs": dirs,
        "window_start": started_at,
        "window_end": latest["window_end"],
        "start_usage_usd": start_usage,
        "key_usage_delta_usd": round(spent, 6),
        "ledger_cost_usd": round(ledger_cost, 10),
        "ledger_calls": sum(w["ledger_calls"] for w in windows),
        "unattributed_usd": round(spent - ledger_cost, 6),
        "unknown_cost_calls": sum(w["unknown_cost_calls"] for w in windows),
    }


def _same_dir_key(path: Path) -> str:
    """One spelling per directory: absolute (against the working directory) and
    normalised, without resolving symlinks (``run.py`` records absolute, unresolved
    paths; older runs recorded relative ones)."""
    return os.path.normpath(Path(path).absolute())


def runs_spend(
    runs_dir: Path, cells: Iterable[CellData], extra_dirs: Iterable[Path] = ()
) -> list[dict[str, Any]]:
    """:func:`gateway_spend` for ``runs_dir``, every host mirror of a merged view
    (``_hosts/<name>/``, ``tools/sync_runs.py``), every other gateway runs directory
    the cells' ledgers live in (``<gateway dir>/<run id>/gateway.jsonl``; a run linked
    into a merged view counts where it really is) and ``extra_dirs`` (the runs
    directories of the removal manifests: a directory only a deleted cell's ledgers named
    is still one), in path order. Each directory counts once however its paths are
    spelled (a relative ``--runs-dir`` and the absolute ledger paths of the
    same runs would otherwise list, and sum, the key's spend twice)."""
    dirs: dict[str, Path] = {_same_dir_key(runs_dir): Path(_same_dir_key(runs_dir))}
    hosts = runs_dir / HOSTS_DIR
    for mirror in sorted(hosts.iterdir()) if hosts.is_dir() else ():
        if mirror.is_dir():
            dirs.setdefault(_same_dir_key(mirror), Path(_same_dir_key(mirror)))
    for extra in extra_dirs:
        dirs.setdefault(_same_dir_key(extra), Path(_same_dir_key(extra)))
    for cell in cells:
        for attempt in cell.attempts:
            ledger = (attempt.ledger or {}).get("ledger")
            if ledger:
                run_dir = Path(str(ledger)).parent
                if run_dir.is_symlink():
                    # Where the link points, spelled like every other path here (not
                    # resolved: /var and /private/var would count one directory twice).
                    run_dir = Path(
                        os.path.normpath(run_dir.parent / os.readlink(run_dir))
                    )
                directory = run_dir.parent
                dirs.setdefault(
                    _same_dir_key(directory), Path(_same_dir_key(directory))
                )
    found = []
    for directory in sorted(dirs.values(), key=str):
        spend = gateway_spend(directory)
        if spend is not None:
            found.append(spend)
    return found


def _runs_relative(path: str | Path | None, runs_dir: Path) -> str | None:
    """``path`` spelled relative to the runs directory when it lies inside it (a removal
    manifest under ``_removed/``), else absolute and normalised. The same cells read
    from a merged view or a copied runs directory then report the same ``removed``
    block, as they report the same costs (``test_sync_runs.py`` compares the
    two reports' cells)."""
    if path is None:
        return None
    absolute, base = _same_dir_key(Path(path)), _same_dir_key(runs_dir)
    if absolute == base or absolute.startswith(base.rstrip(os.sep) + os.sep):
        return Path(os.path.relpath(absolute, base)).as_posix()
    return absolute


# ---------------------------------------------------------------------------
# Removed methods (study.REMOVED_METHODS)
# ---------------------------------------------------------------------------


#: The keys of a ``removed.cells`` row, whether the cell is on disk or was deleted
#: (``tools/remove_method_data.py`` keeps them in its manifest, computed the same way).
REMOVED_ROW_KEYS: tuple[str, ...] = (
    "cell_id",
    "cell_dir",
    "method",
    "benchmark",
    "model_key",
    "split_mode",
    "status",
    "complete",
    "reported_attempt",
    "attempts",
    "costs",
    "not_counted",
)


def removed_manifest_rows(
    manifests: Sequence[Any],
    live: Sequence[CellData],
    split_mode: str | None,
    runs_dir: Path,
) -> list[dict[str, Any]]:
    """The deleted cells of the removal manifests (``tools/remove_method_data.py``), as
    :func:`removed_section` rows (``on_disk`` false, ``deleted`` naming the removal), of
    the report's split mode. A cell still on disk (the same directory and instance: an
    interrupted deletion) is read from disk and not from its manifest, so it counts
    once. A manifest with problems is left out, and so is a row that cannot stand in for
    its cell (``remove_method_data.cell_row_problem``); the report warns about both
    (:func:`manifest_row_problems`)."""
    on_disk = {(c.paths.root.name, c.manifest.get("instance")) for c in live}
    rows = []
    for manifest in manifests:
        if manifest.problems:
            continue
        data = manifest.data
        for cell in data.get("cells") or ():
            if cell_row_problem(cell) is not None:
                continue
            if split_mode is not None and cell.get("split_mode") != split_mode:
                continue
            if (cell.get("cell_dir"), cell.get("instance")) in on_disk:
                continue
            rows.append(
                {
                    **{k: cell.get(k) for k in REMOVED_ROW_KEYS},
                    "on_disk": False,
                    "deleted": {
                        "removal_id": manifest.removal_id,
                        "manifest": _runs_relative(manifest.path, runs_dir),
                        "at": data.get("completed_at") or data.get("created_at"),
                        "state": data.get("state"),
                        "host": _mapping(data.get("host")).get("hostname"),
                    },
                }
            )
    return rows


def manifest_row_problems(manifests: Sequence[Any]) -> list[str]:
    """The cell rows of readable removal manifests that :func:`removed_manifest_rows`
    leaves out (``remove_method_data.cell_row_problem``), one text per manifest."""
    out = []
    for manifest in manifests:
        if manifest.problems:
            continue
        found = [
            p
            for p in (cell_row_problem(r) for r in manifest.data.get("cells") or ())
            if p is not None
        ]
        if found:
            out.append(
                f"{manifest.path}: {len(found)} cell row(s) cannot be listed ("
                + "; ".join(found[:3])
                + ("; ..." if len(found) > 3 else "")
                + "): those cells and their costs are not in the removed block; the "
                "spend windows still count the manifest's calls"
            )
    return out


def removed_section(
    cells: Sequence[CellData],
    manifest_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The report's ``removed`` block: the cells of the methods the study removed
    (``study.REMOVED_METHODS``), which :func:`build_report` set
    aside (JSON-ready, without an absolute path: a merged view and the runs directory
    it links report the same block).

    * ``methods``: every removed method's decision and reason, and its cells' ids;
    * ``cells``: each cell (``cell_dir``: its directory's name; two directories of one
      cell id are two entries) with its status and reported attempt, its ledgers'
      costs exactly as a study cell shows them (``costs``: :func:`cell_costs`;
      ``not_counted``: :func:`uncounted_costs`); ``on_disk`` false and ``deleted`` for
      a cell whose data was deleted, read from its
      removal manifest (``manifest_rows``);
    * ``deleted``: how many cells were deleted, their ledgers' cost and the removals;
    * ``cost_usd``: the cells' ledgers in all (``costs.removed`` has the breakdown)."""
    rows = []
    for cell in cells:
        costs = cell_costs(cell)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "cell_dir": cell.paths.root.name,
                # As split_cells read it: a definition without a method is no crash.
                "method": study.method_of(cell),
                "benchmark": cell.benchmark,
                "model_key": cell.model_key,
                "split_mode": cell.split_mode,
                "status": cell.status.get("status"),
                "complete": cell.complete,
                "reported_attempt": cell.reported,
                "attempts": len(cell.attempts),
                "costs": costs,
                "not_counted": uncounted_costs(costs),
                "on_disk": True,
                "deleted": None,
            }
        )
    rows += [dict(r) for r in manifest_rows]
    rows.sort(key=lambda r: (str(r["cell_id"]), str(r["cell_dir"])))
    deleted = [r for r in rows if not r["on_disk"]]
    return {
        "methods": study.removed_methods_record(
            (str(r["method"]), str(r["cell_id"])) for r in rows
        ),
        "cells": rows,
        "deleted": {
            "cells": len(deleted),
            "cost_usd": round(
                sum(float(r["costs"]["total_cost"]) for r in deleted), 10
            ),
            "removal_ids": sorted({str(r["deleted"]["removal_id"]) for r in deleted}),
        },
        "cost_usd": round(sum(float(r["costs"]["total_cost"]) for r in rows), 10),
    }


def _bucket_quantile(
    buckets: Sequence[int], q: float, edges: Sequence[float] = LATENCY_EDGES_S
) -> float | None:
    """The upper edge of the bucket (of ``edges``: latency by default) that holds
    quantile ``q`` (None: no call; ``inf`` for the open last bucket)."""
    total = sum(buckets)
    if not total:
        return None
    seen = 0
    for index, count in enumerate(buckets):
        seen += count
        if seen >= q * total:
            return float(edges[index]) if index < len(edges) else float("inf")
    return float("inf")


def matrix_pricing(sections: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Per model and provider over every reported cell: ok calls, billed cost against the
    list price (``ratio``, and the per-call extremes), calls in the long-input price tier,
    the answering attempts' latency and ``slow_calls`` (past
    :data:`~cell_records.SLOW_CALL_DEADLINE_FRACTION` of the deadline), and the reasoning
    tokens (``reasoning_mean``, ``reasoning_p95_le``); per model, the calls that timed out
    on the model's own attempt. ``outside_band`` names the providers whose
    ratio lies outside :data:`PRICE_RATIO_BAND`; ``reasoning_outliers`` those whose mean
    reasoning exceeds :data:`REASONING_OUTLIER_FACTOR` times the median of the model's
    other providers' means (and :data:`REASONING_OUTLIER_MIN_MEAN`)."""
    merged: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    timeouts: Counter[str] = Counter()
    for section in sections:
        providers = section.get("providers") or {}
        for model, by_provider in (providers.get("pricing") or {}).items():
            for provider, entry in by_provider.items():
                into = merged[model].setdefault(
                    provider,
                    {
                        "calls": 0,
                        "priced_calls": 0,
                        "cost_usd": 0.0,
                        "list_cost_usd": 0.0,
                        "ratio_min": None,
                        "ratio_max": None,
                        "over_tier_calls": 0,
                        "latency_s_max": 0.0,
                        "latency_buckets": [0] * (len(LATENCY_EDGES_S) + 1),
                        "slow_calls": 0,
                        "reasoning_calls": 0,
                        "reasoning_tokens": 0,
                        "reasoning_tokens_max": 0,
                        "reasoning_buckets": [0] * (len(REASONING_EDGES) + 1),
                    },
                )
                for key in (
                    "calls",
                    "priced_calls",
                    "over_tier_calls",
                    "slow_calls",
                    "reasoning_calls",
                    "reasoning_tokens",
                ):
                    into[key] += int(entry.get(key) or 0)
                into["reasoning_tokens_max"] = max(
                    into["reasoning_tokens_max"],
                    int(entry.get("reasoning_tokens_max") or 0),
                )
                for index, count in enumerate(entry.get("reasoning_buckets") or ()):
                    if index < len(into["reasoning_buckets"]):
                        into["reasoning_buckets"][index] += int(count)
                for key in ("cost_usd", "list_cost_usd"):
                    into[key] = round(into[key] + float(entry.get(key) or 0.0), 10)
                for key, pick in (("ratio_min", min), ("ratio_max", max)):
                    value = entry.get(key)
                    if value is not None:
                        into[key] = (
                            value if into[key] is None else pick(into[key], value)
                        )
                into["latency_s_max"] = max(
                    into["latency_s_max"], float(entry.get("latency_s_max") or 0.0)
                )
                for index, count in enumerate(entry.get("latency_buckets") or ()):
                    if index < len(into["latency_buckets"]):
                        into["latency_buckets"][index] += int(count)
        for model, count in (providers.get("timeouts_model_blamed") or {}).items():
            timeouts[model] += int(count)
    low, high = PRICE_RATIO_BAND
    outside = []
    outliers = []
    for model, by_provider in merged.items():
        for provider, entry in by_provider.items():
            listed = entry["list_cost_usd"]
            entry["ratio"] = round(entry["cost_usd"] / listed, 4) if listed else None
            entry["latency_p50_le_s"] = _bucket_quantile(entry["latency_buckets"], 0.5)
            entry["latency_p90_le_s"] = _bucket_quantile(entry["latency_buckets"], 0.9)
            reasoned = entry["reasoning_calls"]
            entry["reasoning_mean"] = (
                round(entry["reasoning_tokens"] / reasoned, 1) if reasoned else None
            )
            entry["reasoning_p95_le"] = _bucket_quantile(
                entry["reasoning_buckets"], 0.95, REASONING_EDGES
            )
            if entry["ratio"] is not None and not low <= entry["ratio"] <= high:
                outside.append(f"{model} / {provider}")
        means = {
            p: e["reasoning_mean"]
            for p, e in by_provider.items()
            if e["reasoning_mean"] is not None
        }
        for provider, value in means.items():
            # The median of the model's other providers: a provider is never its own
            # reference, so one outlier among two providers is still named.
            others = [v for p, v in means.items() if p != provider]
            if not others:
                continue
            median = statistics.median(others)
            if (
                value >= REASONING_OUTLIER_MIN_MEAN
                and value > REASONING_OUTLIER_FACTOR * median
            ):
                outliers.append(
                    {
                        "model": model,
                        "provider": provider,
                        "reasoning_mean": value,
                        "median_provider_mean": median,
                    }
                )
    return {
        "latency_edges_s": list(LATENCY_EDGES_S),
        "reasoning_edges": list(REASONING_EDGES),
        "slow_call_s": protocol.CALL_DEADLINE_S * SLOW_CALL_DEADLINE_FRACTION,
        "band": list(PRICE_RATIO_BAND),
        "by_model": {
            m: {p: merged[m][p] for p in sorted(merged[m])} for m in sorted(merged)
        },
        "timeouts_model_blamed": dict(sorted(timeouts.items())),
        "outside_band": sorted(outside),
        "reasoning_outliers": sorted(
            outliers, key=lambda o: (o["model"], o["provider"])
        ),
    }


# ---------------------------------------------------------------------------
# Every attempt and directory of a cell (the newest complete one is reported, the
# others listed with their scores; no score combines two runs)
# ---------------------------------------------------------------------------


def _attempt_test_rows(cell: CellData, number: int) -> list[dict[str, Any]]:
    """An attempt's own scored test rows: the last scored row of each test item (as
    ``run.py`` publishes a complete attempt's), in test order."""
    rows = read_jsonl(cell.paths.attempt(number).results)
    latest = {str(r.get("raw_id")): r for r in rows if r.get("score") is not None}
    order = [str(i) for i in cell.definition.get("test_items") or ()]
    ordered = [latest[i] for i in order if i in latest]
    return ordered + [r for i, r in latest.items() if i not in set(order)]


def _attempt_metrics(
    cell: CellData, rows: Sequence[Mapping[str, Any]]
) -> dict[str, float | None] | None:
    """The metrics of an attempt that is not reported, on its own rows only; None
    without a row, or when a row has no value of a metric."""
    if not rows:
        return None
    try:
        return {
            m.name: mean([metric_value(r, m) for r in rows])
            for m in METRICS.get(cell.benchmark, ())
        }
    except ValueError:
        return None


def cell_attempts(
    cell: CellData,
    test: Mapping[str, Any] | None = None,
    *,
    directory_used: bool = True,
) -> list[dict[str, Any]]:
    """Every attempt of the directory: its state (a superseded attempt's ``superseded``
    record, note and ``previous_state``), when and where it ran, whether it is the
    reported one (``reported``: within this directory; ``directory_used``: False for a
    directory of the cell id the report does not use, :func:`not_used_row`, so its own
    reported attempt is not the cell's), how many test items it scored, its metrics on its own rows
    (:func:`_attempt_metrics`; the reported attempt copies ``test``, the section's) and
    its ledger's cost. Each attempt also names the versions it ran under (a
    superseded attempt's code is in the report, not only in its ``attempt.json``):
    ``protocol`` (``protocol.attempt_protocol``: the protocol it started under),
    ``code`` (its harness and MIRIX digests, commit and whether the tree was dirty;
    each None where it records none), ``version_segments`` (how many version
    segments it recorded; 0 before segments were recorded) and ``versions_mixed``.
    Scores of attempts that are not reported appear only in 'Runs and attempts', never
    in a table or comparison."""
    out = []
    for attempt in cell.attempts:
        state = attempt.state
        code = _mapping(state.get("code"))
        git = _mapping(state.get("git"))
        segments = state.get("version_segments")
        reported = cell.complete and attempt.number == cell.reported
        rows = _attempt_test_rows(cell, attempt.number)
        metrics = (
            dict(test.get("metrics") or {})
            if reported and test is not None
            else _attempt_metrics(cell, rows)
        )
        host = state.get("host")
        note = state.get("note")
        out.append(
            {
                "attempt": attempt.number,
                "state": state.get("state"),
                "invalidation": state.get("invalidation"),
                "previous_state": state.get("previous_state"),
                "superseded": dict(state["superseded"])
                if isinstance(state.get("superseded"), Mapping)
                else None,
                "note": note.get("text") if isinstance(note, Mapping) else None,
                "supersedes": state.get("supersedes"),
                "started_at": state.get("started_at"),
                "completed_at": state.get("completed_at"),
                "host": {
                    k: host.get(k)
                    for k in ("hostname", "machine_id", "system", "machine")
                }
                if isinstance(host, Mapping)
                else None,
                "reported": reported,
                "directory_used": directory_used,
                "protocol": protocol.attempt_protocol(
                    cell.manifest, state, attempt.number
                ),
                "code": {
                    "harness": code.get("harness"),
                    "mirix": code.get("mirix"),
                    "commit": git.get("commit"),
                    "dirty": git.get("dirty"),
                },
                "version_segments": len(segments)
                if isinstance(segments, (list, tuple))
                else 0,
                "versions_mixed": dict(state["versions_mixed"])
                if isinstance(state.get("versions_mixed"), Mapping)
                else None,
                "test_items_scored": len(rows),
                "metrics": metrics,
                "cost": (
                    ((attempt.ledger or {}).get("totals") or {}).get("cost_total")
                    if attempt.ledger
                    else None
                ),
            }
        )
    return out


def not_used_row(
    cell: CellData,
    chosen: CellData,
    role: str,
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A directory of a cell id that another directory reports: what it is, its
    scores (never in any other table), audit status and money (counted in
    ``costs.total`` and ``costs.superseded_directories``)."""
    costs = cell_costs(cell)
    test = test_metrics(cell) if cell.complete else None
    return {
        "cell_id": cell.cell_id,
        "cell_dir": cell.paths.root.name,
        "role": role,
        "chosen_dir": chosen.paths.root.name,
        "benchmark": cell.benchmark,
        "method": cell.method,
        "model_key": cell.model_key,
        "group": cell.group,
        "host": cell_host(cell),
        "status": cell.status.get("status"),
        "complete": cell.complete,
        "reported_attempt": cell.reported,
        "started_at": directory_started_at(cell),
        "test": test,
        "audit": audit_status(cell, audit),
        "costs": costs,
        "not_counted": uncounted_costs(costs),
        "attempts_detail": cell_attempts(cell, test, directory_used=False),
        "cost_label": f"{cell.cell_id} (directory {cell.paths.root.name})",
    }


# ---------------------------------------------------------------------------
# Providers and quantization (from the records, no rerun)
# ---------------------------------------------------------------------------

#: A provider the endpoint listing does not name.
NOT_LISTED = "not listed"
CAVEAT_QUANTIZATION = (
    "Quantization is the endpoint listing's label for the provider name (see "
    "provider_endpoints: file and source), not a per-call record: the gateway ledger "
    "stores the provider name only, and a provider can serve several endpoints or "
    "change its quantization."
)
CAVEAT_PROVIDER_ACCURACY = (
    "Accuracy by main provider is observational: OpenRouter picked each call's provider, "
    "so the groups are not random samples; an item's main provider is the one that "
    "served most of its actor calls (ALFWorld items take up to 50 calls, possibly from "
    "several providers: 'of which mixed'), and every cell ran once."
)


def load_provider_endpoints(paths: Sequence[Path]) -> dict[str, Any]:
    """OpenRouter endpoint listings (``GET /api/v1/models/<id>/endpoints`` JSON;
    read-only files, no network):
    ``by_model`` {model id: {provider name: [quantizations]}}, ``sources`` [{file, model,
    _source}] and ``problems`` (unreadable files; blocking)."""
    by_model: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sources = []
    problems = []
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            body = data["data"]
            model = str(body["id"])
            endpoints = list(body.get("endpoints") or ())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping) or not endpoint.get("provider_name"):
                continue
            by_model[model][str(endpoint["provider_name"])].add(
                str(endpoint.get("quantization") or "unknown")
            )
        sources.append(
            {
                "file": str(path),
                "model": model,
                "_source": data.get("_source") if isinstance(data, Mapping) else None,
            }
        )
    return {
        "files": [str(p) for p in paths],
        "sources": sources,
        "by_model": {
            m: {p: sorted(q) for p, q in sorted(providers.items())}
            for m, providers in sorted(by_model.items())
        },
        "problems": problems,
    }


def quantization_label(
    endpoints: Mapping[str, Any] | None, model: str, provider: str
) -> str:
    """The listing's quantization of ``provider`` for ``model``: its one value,
    ``mixed(a/b)`` for several endpoints, else :data:`NOT_LISTED`."""
    listed = ((endpoints or {}).get("by_model") or {}).get(model, {}).get(provider)
    if not listed:
        return NOT_LISTED
    return listed[0] if len(listed) == 1 else f"mixed({'/'.join(listed)})"


def item_main_providers(cell: CellData) -> dict[str, dict[str, Any]]:
    """Each published test row's main provider: the provider of most of its actor
    calls that ended ok (the row's ``gateway_call_ids``, else the ledger's test-phase
    actor calls of its ``task_uid``); a tie goes to the provider of its last ok call.
    ``{raw_id: {main, share, mixed, calls}}``; a row without an ok call is left out."""
    calls = _ledger_rows(cell)
    by_id = {str(r.get("call_id")): r for r in calls if r.get("call_id")}
    by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in calls:
        if (
            _call_phase(row) == "test"
            and row.get("role") == "actor"
            and row.get("item")
        ):
            by_item[str(row["item"])].append(row)
    out = {}
    for row in cell.results:
        ids = [str(c) for c in row.get("gateway_call_ids") or () if c]
        found = (
            [by_id[c] for c in ids if c in by_id]
            if ids
            else by_item.get(str(row.get("task_uid")), [])
        )
        ok = [c for c in found if c.get("status") == "ok" and c.get("provider")]
        if not ok:
            continue
        counts = Counter(str(c["provider"]) for c in ok)
        top = max(counts.values())
        tied = {p for p, n in counts.items() if n == top}
        main = next(str(c["provider"]) for c in reversed(ok) if c["provider"] in tied)
        out[str(row.get("raw_id"))] = {
            "main": main,
            "share": counts[main] / len(ok),
            "mixed": len(counts) > 1,
            "calls": len(ok),
        }
    return out


def accuracy_by_main_provider(cell: CellData) -> dict[str, dict[str, Any]]:
    """Per main provider (:func:`item_main_providers`): items, correct (the cell's main
    metric) and the items whose calls several providers served."""
    metric = next((m for m in METRICS.get(cell.benchmark, ()) if m.main), None)
    if metric is None:
        return {}
    mains = item_main_providers(cell)
    out: dict[str, dict[str, Any]] = {}
    for row in cell.results:
        main = mains.get(str(row.get("raw_id")))
        if main is None:
            continue
        entry = out.setdefault(
            main["main"], {"items": 0, "correct": 0.0, "mixed_items": 0}
        )
        entry["items"] += 1
        entry["correct"] += metric_value(row, metric)
        entry["mixed_items"] += int(main["mixed"])
    return dict(sorted(out.items()))


def cell_provider_mix(
    cell: CellData, endpoints: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The test phase's actor calls by provider and by the listing's quantization
    label, and the items by main provider."""
    reported = cell.attempt(cell.reported)
    ledger = (reported.ledger if reported else None) or {}
    calls = dict(ledger.get("providers", {}).get("test/actor") or {})
    quantization: Counter[str] = Counter()
    for provider, n in calls.items():
        quantization[quantization_label(endpoints, cell.model, provider)] += int(n)
    mains = item_main_providers(cell) if cell.complete else {}
    return {
        "test_actor_calls": dict(sorted(calls.items())),
        "quantization_calls": dict(sorted(quantization.items())),
        "items_by_main_provider": dict(
            sorted(Counter(m["main"] for m in mains.values()).items())
        ),
        "mixed_provider_items": sum(1 for m in mains.values() if m["mixed"]),
        "items_without_provider": len(cell.results) - len(mains)
        if cell.complete
        else None,
    }


def provider_accuracy(
    sections: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Any] | None
) -> dict[str, Any]:
    """{model: {benchmark: {provider: {items, correct, accuracy, share_of_items,
    mixed_items, quantization, cells}}}} pooled over the complete reported cells (every
    method) of each chat model and benchmark."""
    pooled: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for section in sections:
        for provider, entry in (section.get("accuracy_by_main_provider") or {}).items():
            into = pooled[str(section["model"])][str(section["benchmark"])].setdefault(
                provider,
                {"items": 0, "correct": 0.0, "mixed_items": 0, "cells": []},
            )
            into["items"] += int(entry["items"])
            into["correct"] += float(entry["correct"])
            into["mixed_items"] += int(entry["mixed_items"])
            into["cells"].append(str(section["cell_id"]))
    out: dict[str, Any] = {}
    for model, benchmarks in sorted(pooled.items()):
        out[model] = {}
        for benchmark, providers in sorted(benchmarks.items()):
            total = sum(e["items"] for e in providers.values())
            out[model][benchmark] = {
                p: {
                    **e,
                    "cells": sorted(e["cells"]),
                    "accuracy": e["correct"] / e["items"] if e["items"] else None,
                    "share_of_items": e["items"] / total if total else None,
                    "quantization": quantization_label(endpoints, model, p),
                }
                for p, e in sorted(providers.items())
            }
    return out


def _removal_manifests(
    runs_dir: Path, gateway_runs_dirs: Sequence[Path] = ()
) -> list[Any]:
    """Every removal manifest of the runs directory, its host mirrors and the other
    gateway runs directories, one per removal id (``remove_method_data.load_manifests``;
    blocking)."""
    seen: set[str] = set()
    out = []
    for directory in [runs_dir, *gateway_runs_dirs]:
        for manifest in load_manifests(directory):
            if manifest.removal_id not in seen:
                seen.add(manifest.removal_id)
                out.append(manifest)
    return out


def _names(cells: Iterable[str], limit: int = 20) -> str:
    cells = list(cells)
    shown = ", ".join(f"`{c}`" for c in cells[:limit])
    return shown + (f" and {len(cells) - limit} more" if len(cells) > limit else "")


def _warning(check: str, detail: str, cells: Iterable[str] = ()) -> dict[str, Any]:
    return {"check": check, "detail": detail, "cells": sorted(set(cells))}


def collect_warnings(
    *,
    sections: Sequence[Mapping[str, Any]],
    duplicates: Sequence[Mapping[str, Any]],
    group_duplicates: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any] | None,
    removed_problems: Sequence[tuple[str, str]],
    endpoints: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """The report's warnings (record, don't block): each is a
    condition the reader must know about, written down and shown in the header; none
    changes the exit status (only ``--require-audit`` and nothing complete do)."""
    out: list[dict[str, Any]] = []
    for entry in duplicates:
        used = entry["used"]
        others = "; ".join(
            f"`{row['cell_dir']}` ({row['role']}, {row['status']}, "
            f"started {row.get('started_at') or '?'}"
            + (
                ", score "
                + ", ".join(
                    f"{k} {_fmt(v)}"
                    for k, v in ((row.get("test") or {}).get("metrics") or {}).items()
                )
                if row.get("complete")
                else ""
            )
            + f", {_usd(row['costs']['total_cost'])})"
            for row in entry["not_used"]
        )
        out.append(
            _warning(
                "duplicate_directories",
                f"{1 + len(entry['not_used'])} directories of cell `{entry['cell_id']}`: "
                f"reported `{used['cell_dir']}` ({entry['rule']}, started "
                f"{used.get('started_at') or '?'}); not used: {others}. Both "
                "directories' money counts.",
                [entry["cell_id"]],
            )
        )
    for entry in group_duplicates:
        out.append(
            _warning("duplicate_group_method", str(entry["detail"]), entry["cells"])
        )
    unknown = [
        s["cell_id"]
        for s in sections
        if not s["protocol"]["known"]
        or (s.get("versions") or {}).get("unknown_fingerprints")
    ]
    if unknown:
        out.append(
            _warning(
                "unknown_protocol_version",
                "under a protocol version no freeze registered (compared, marked "
                "same_protocol = unknown; the audit warns): " + _names(unknown),
                unknown,
            )
        )
    mixed = [s["cell_id"] for s in sections if (s.get("versions") or {}).get("mixed")]
    if mixed:
        out.append(
            _warning(
                "mixed_version",
                "ran more than one protocol or code version in their reported attempt "
                "(see 'Versions per cell'; compared, marked mixed_version): "
                + _names(mixed),
                mixed,
            )
        )
    if audit is not None and not audit.get("unreadable") and not audit_schema2(audit):
        out.append(
            _warning(
                "legacy_audit",
                "audit.json is of schema 1 (no validity_checks): every violation "
                "counts and a run-level violation fails every cell; re-run audit.py",
            )
        )
    running = [s for s in sections if s.get("newer_attempt")]
    if running:
        out.append(
            _warning(
                "newer_attempt_running",
                "a newer attempt is running (not complete until it finishes: no metric "
                "until then; the earlier attempt's score is in 'Runs and attempts'): "
                + "; ".join(
                    f"`{s['cell_id']}` attempt {s['newer_attempt']['number']} "
                    f"({s['newer_attempt']['state']})"
                    for s in running
                ),
                [s["cell_id"] for s in running],
            )
        )
    problems = [s for s in sections if s.get("record_problems")]
    if problems:
        out.append(
            _warning(
                "records_not_of_reported_attempt",
                "published records of another attempt (the cell is not complete, no "
                "number of it is shown): "
                + "; ".join(
                    f"`{s['cell_id']}`: {', '.join(s['record_problems'])}"
                    for s in problems
                ),
                [s["cell_id"] for s in problems],
            )
        )
    degraded = [s for s in sections if (s.get("degraded") or {}).get("test")]
    degraded_learning = [
        s for s in sections if (s.get("degraded") or {}).get("learning")
    ]
    if degraded or degraded_learning:
        parts = [
            f"`{s['cell_id']}` "
            + ", ".join(
                f"{flag} {len(ids)} test item(s)"
                for flag, ids in s["degraded"]["test"].items()
            )
            for s in degraded
        ] + [
            f"`{s['cell_id']}` "
            + ", ".join(
                f"{flag} {n} learning item(s)"
                for flag, n in s["degraded"]["learning"].items()
            )
            for s in degraded_learning
        ]
        out.append(
            _warning(
                "degraded_items",
                "items scored under a degraded condition (see 'Items scored under "
                "degraded conditions'): " + "; ".join(parts),
                [s["cell_id"] for s in degraded + degraded_learning],
            )
        )
    refused = [
        s
        for s in sections
        if any((s.get("method_refusals") or {}).get("counters", {}).values())
    ]
    if refused:
        out.append(
            _warning(
                "method_refusals",
                "learning calls the provider refused, counted and gone on from (AWM kept "
                "its previous workflow): "
                + "; ".join(
                    f"`{s['cell_id']}` "
                    + ", ".join(
                        f"{k} {v}"
                        for k, v in s["method_refusals"]["counters"].items()
                        if v
                    )
                    for s in refused
                ),
                [s["cell_id"] for s in refused],
            )
        )
    lost = [
        s
        for s in sections
        if (s.get("mirix") or {}).get("learning_calls_lost")
        or (s.get("mirix") or {}).get("tolerated")
    ]
    if lost:
        out.append(
            _warning(
                "mirix_learning_lost",
                "MIRIX learning went on past lost calls: "
                + "; ".join(
                    f"`{s['cell_id']}` {s['mirix']['learning_calls_lost']} lost call(s)"
                    + (
                        " (method: "
                        + ", ".join(
                            f"{k} {v}"
                            for k, v in sorted(s["mirix"]["tolerated"].items())
                        )
                        + ")"
                        if s["mirix"].get("tolerated")
                        else ""
                    )
                    for s in lost
                ),
                [s["cell_id"] for s in lost],
            )
        )
    # Where the quantization can differ: a cell whose test actor calls went to more
    # than one provider, some of them not in the listing.
    unlisted = [
        s["cell_id"]
        for s in sections
        if s.get("complete")
        and len((s.get("provider_mix") or {}).get("test_actor_calls") or {}) > 1
        and (s.get("provider_mix") or {}).get("quantization_calls", {}).get(NOT_LISTED)
    ]
    if unlisted:
        out.append(
            _warning(
                "quantization_unlisted",
                (
                    "no endpoint listing was given (--provider-endpoints): "
                    if not endpoints.get("sources")
                    else "providers the endpoint listing does not name: "
                )
                + "their calls' quantization is 'not listed' in "
                + _names(unlisted),
                unlisted,
            )
        )
    for check, detail in removed_problems:
        out.append(_warning(check, detail))
    return out


def build_report(
    runs_dir: Path,
    *,
    split_mode: str | None = "protocol",
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.SEED,
    frozen: str | None = None,
    audit_path: Path | None = None,
    require_audit: bool = False,
    gateway_runs_dirs: Sequence[Path] = (),
    versions: Mapping[str, Mapping[str, Any]] | None = None,
    provider_endpoints: Sequence[Path] = (),
) -> dict[str, Any]:
    """The whole report as JSON-ready data (blocking). Every cell is marked with its audit
    status (``audit_path``, default ``<runs-dir>/_audit/audit.json``); ``require_audit``
    also leaves cells without a passing audit out of the comparisons.

    One directory per cell id is reported (:func:`select_reported`: the newest
    complete one); the others are listed under ``duplicates`` with their scores, and
    their money counts in ``costs.total``. Within a directory the reported attempt is
    the newest complete one; every attempt is listed with its own score
    (``attempts_detail``), and no number combines two attempts or two directories.

    ``runs_dir`` may be a merged view of several hosts (``tools/sync_runs.py``);
    ``gateway_runs_dirs`` names other directories that hold ledgers. ``versions`` are the
    known protocol versions (``protocol.known_versions``): by default those of the
    frozen record when ``frozen`` is read from it too, and only ``frozen`` itself when
    the caller names it. ``provider_endpoints`` are OpenRouter endpoint listings
    (:func:`load_provider_endpoints`) that label the providers' quantization.

    The cells of a method the study removed (``study.REMOVED_METHODS``) are set aside
    right after they are read: everything below reads the study's cells only, except
    the money. Their deleted data (``tools/remove_method_data.py``) is read from the
    removal manifests, so the money counts the same before and after the deletion. The
    gateway spend windows sum every ledger of their directory and the deleted ledgers'
    calls; the removed cells' ledgers are ``costs.removed`` and in
    ``costs.all_cells_total``, and are listed under ``removed`` (:func:`removed_section`)."""
    loaded = load_cells(
        runs_dir, split_mode=split_mode, gateway_runs_dirs=gateway_runs_dirs
    )
    study_cells, removed = study.split_cells(loaded)
    cells, set_apart = select_reported(study_cells)
    if frozen is None:
        frozen = _frozen_fingerprint()
        if versions is None:
            versions = _known_versions()
    known: Mapping[str, Mapping[str, Any]] = versions or {}
    endpoints = load_provider_endpoints(list(provider_endpoints))
    sections = [cell_section(c, frozen=frozen, versions=known) for c in cells]
    records = {s["cell_id"]: s["versions"] for s in sections}
    fingerprints = {s["cell_id"]: s["protocol_fingerprint"] for s in sections}
    groups = version_groups(sections)
    mixed = mixed_version_benchmarks(groups)
    audit_file = audit_path if audit_path is not None else runs_dir / AUDIT_FILE
    audit = load_audit(audit_file)
    audits = {c.cell_id: audit_status(c, audit) for c in cells}
    usable = audit if audit is not None and not audit.get("unreadable") else None
    not_used = [not_used_row(c, chosen, role, audit) for c, chosen, role in set_apart]
    for cell, section in zip(cells, sections):
        section["audit"] = audits[section["cell_id"]]
        section["audit_detail"] = audit_detail(cell, usable)
        section["attempts_detail"] = cell_attempts(cell, section.get("test"))
        directories = [
            {
                k: row[k]
                for k in (
                    "cell_dir",
                    "role",
                    "status",
                    "complete",
                    "started_at",
                    "host",
                )
            }
            for row in not_used
            if row["cell_id"] == cell.cell_id
        ]
        section["reruns"] = cell_reruns(
            cell, usable, attempts=section["attempts_detail"], directories=directories
        )
        section["provider_mix"] = cell_provider_mix(cell, endpoints)
        section["accuracy_by_main_provider"] = (
            accuracy_by_main_provider(cell) if cell.complete else {}
        )
    reran = sorted(s["cell_id"] for s in sections if s["reruns"])
    manifests = _removal_manifests(runs_dir, gateway_runs_dirs)
    removed_problems: list[tuple[str, str]] = [
        ("removed_manifest_unreadable", f"{'; '.join(m.problems)} (its money unknown)")
        for m in manifests
        if m.problems
    ]
    removed_problems += [
        ("removed_manifest_unreadable", text)
        for text in manifest_row_problems(manifests)
    ]
    manifest_rows = removed_manifest_rows(
        [m for m in manifests if not m.problems], removed, split_mode, runs_dir
    )
    for manifest in manifests:
        if manifest.problems:
            continue
        manifest_calls(manifest)
        if manifest.problems:
            removed_problems.append(
                (
                    "removed_manifest_mismatch",
                    "; ".join(manifest.problems) + " (its rows are counted)",
                )
            )
    # Every cell's ledgers name the gateway directories, the removed cells' too: their
    # calls were billed, and a window without them would call that spend unattributed.
    spend = runs_spend(runs_dir, loaded, extra_dirs=[m.runs_dir for m in manifests])
    union = union_spend(spend)
    # Every directory's money: the reported ones and those another directory of their
    # cell id replaced (their calls were billed too).
    costs = matrix_costs([*sections, *not_used])
    costs["superseded_directories"] = matrix_costs(not_used)
    set_aside = removed_section(removed, manifest_rows)
    costs["removed"] = matrix_costs(set_aside["cells"])
    costs["all_cells_total"] = round(costs["total"] + costs["removed"]["total"], 10)
    by_id = {s["cell_id"]: s for s in sections}
    duplicates = [
        {
            "cell_id": cell_id,
            "group": by_id[cell_id]["group"],
            "method": by_id[cell_id]["method"],
            "rule": "newest complete (the reported attempt's started_at, then "
            "cell.json created_at, then the directory name)",
            "used": {
                "cell_dir": by_id[cell_id]["cell_dir"],
                "host": by_id[cell_id]["host"],
                "started_at": directory_started_at(
                    next(c for c in cells if c.cell_id == cell_id)
                ),
                "attempt": by_id[cell_id]["reported_attempt"],
                "complete": by_id[cell_id]["complete"],
            },
            "not_used": [row for row in not_used if row["cell_id"] == cell_id],
        }
        for cell_id in sorted({row["cell_id"] for row in not_used})
    ]
    group_duplicates = duplicate_cells(cells)
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_dir": str(runs_dir),
        "split_mode": split_mode or "all",
        "protocol": {
            "protocol_id": protocol.PROTOCOL_ID,
            "frozen_fingerprint": frozen,
            "report_tree_fingerprint": protocol.protocol_fingerprint(),
            "cells": fingerprints,
            "cells_not_frozen": sorted(
                c for c, fp in fingerprints.items() if fp != frozen
            ),
            # Versions are recorded and reported, not enforced across cells. An earlier
            # version is a known one; an unknown one is compared and marked
            # (same_protocol = unknown).
            "cells_earlier_version": sorted(
                s["cell_id"]
                for s in sections
                if s["protocol"]["known"] and not s["protocol"]["current"]
            ),
            "cells_unknown_version": sorted(
                s["cell_id"]
                for s in sections
                if not s["protocol"]["known"]
                or s["versions"].get("unknown_fingerprints")
            ),
            "cells_mixed_version": sorted(
                s["cell_id"] for s in sections if s["versions"].get("mixed")
            ),
            "versions": groups,
            "earlier_versions": {
                fp: {
                    "frozen_at": known[fp].get("frozen_at"),
                    "reason": known[fp].get("reason"),
                    # Why it is earlier: the freeze that replaced it, and what differs
                    # from the frozen value (as the audit names them too).
                    "replaced_by": _replaced_by(fp, known),
                    "differs_from_frozen": protocol.describe_version_diff(
                        protocol.version_diff(fp, frozen, known)
                    ),
                }
                for fp in sorted(
                    {str(fp) for fp in fingerprints.values() if fp}
                    | {
                        str(f)
                        for s in sections
                        for f in s["versions"].get("protocol_fingerprints") or ()
                    }
                )
                if fp != frozen and fp in known
            },
            # The provider policy: OpenRouter's provider routing per chat model; a model
            # without an entry is unrestricted.
            "provider_policy": protocol.provider_policy_settings(),
            # What the reported attempts' calls carried.
            "provider_routing_observed": observed_routing(cells),
        },
        "statistics": {
            "binary": "exact two-sided McNemar test on the discordant pairs",
            "non_binary": "paired percentile bootstrap of the mean difference",
            "resamples": resamples,
            "level": protocol.CI_LEVEL,
            "seed": seed,
            "learning_curve_window": protocol.LEARNING_CURVE_WINDOW,
        },
        "audit": {
            "path": str(audit_file),
            "found": audit is not None and not audit.get("unreadable"),
            "schema": (audit or {}).get("schema"),
            "generated_at": (audit or {}).get("generated_at"),
            "passed": (audit or {}).get("passed"),
            "required": require_audit,
            "cells_not_passed": sorted(
                c for c, a in audits.items() if a["status"] != AUDIT_PASSED
            ),
        },
        # A merged view of several hosts: when it was built, and why it is not final;
        # None for a host's own runs directory.
        "merged_view": merged_view_status(runs_dir),
        "caveats": list(CAVEATS)
        + ([caveat_reruns(reran)] if reran else [])
        + ([caveat_mixed_versions(mixed)] if mixed else []),
        "cells_run_more_than_once": reran,
        # Schema 3: nothing is an error any more (record, don't block); what was one is
        # in ``warnings``.
        "errors": [],
        "warnings": collect_warnings(
            sections=sections,
            duplicates=duplicates,
            group_duplicates=group_duplicates,
            audit=audit,
            removed_problems=removed_problems,
            endpoints=endpoints,
        ),
        "cells": sections,
        "duplicates": duplicates,
        # The cells of the methods the study removed, set aside.
        "removed": set_aside,
        "comparisons": comparisons(
            cells,
            resamples=resamples,
            seed=seed,
            frozen=frozen,
            audits=audits if require_audit else None,
            versions=known,
            records=records,
        ),
        "costs": costs,
        "spend": spend,
        "spend_union": union,
        "pricing": matrix_pricing(sections),
        "provider_endpoints": {
            k: endpoints[k] for k in ("files", "sources", "by_model", "problems")
        },
        "provider_accuracy": provider_accuracy(sections, endpoints),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _usd(value: Any) -> str:
    return "-" if value is None else f"${float(value):.4f}"


def _p(value: float) -> str:
    return f"{value:.4g}"


def uncounted_costs(costs: Mapping[str, Any]) -> str:
    """The cost a cell's totals leave out: attempts without a ledger, unfinished calls,
    cut-off calls (billed, at a cost the ledger never learns)."""
    parts = []
    if costs.get("ledgers_missing"):
        parts.append(
            "no ledger for attempt(s) "
            + ", ".join(str(n) for n in costs["ledgers_missing"])
        )
    if costs.get("unfinished_calls"):
        parts.append(f"{costs['unfinished_calls']} unfinished call(s)")
    if costs.get("unknown_cost_calls"):
        parts.append(f"{costs['unknown_cost_calls']} cut-off call(s), cost unknown")
    return "; ".join(parts) or "-"


def _metric_names(benchmark: Any, present: Iterable[str]) -> list[str]:
    """``present`` in the benchmark's metric order (main metric first), then the rest
    sorted: report.json is written with sorted keys, so a Markdown rendered from it must
    not take the order of its dicts."""
    present = list(present)
    known = [m.name for m in METRICS.get(str(benchmark), ()) if m.name in present]
    return known + sorted(n for n in present if n not in known)


def _by_count(counts: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Counts largest first, then by name (independent of the dict's order)."""
    return sorted(counts.items(), key=lambda kv: (-float(kv[1] or 0), kv[0]))


#: The order MIRIX's LLM usage counters are printed in.
LLM_USAGE_ORDER = {
    "calls": 0,
    "prompt_tokens": 1,
    "completion_tokens": 2,
    "total_tokens": 3,
}


def _phases(by_phase: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Per-phase entries in stream order: learning, then test, then any other."""
    order = {"setup": 0, "learning": 1, "test": 2}
    return sorted(by_phase.items(), key=lambda kv: (order.get(kv[0], 3), kv[0]))


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def provider_policy_text(policy: Mapping[str, Any] | None) -> str:
    """The report header's provider-policy line: the protocol's."""
    policy = policy or {}
    parts = [
        f"{model} ignores {', '.join(policy[model].get('ignore') or ())}"
        for model in sorted(policy)
    ]
    parts += [f"{m} unrestricted" for m in protocol.CHAT_MODELS if m not in policy]
    return "Provider policy: " + "; ".join(parts)


def observed_routing(cells: Sequence[CellData]) -> dict[str, dict[str, Any]]:
    """Per chat model, the provider routings the reported attempts' calls carried
    (``routing``: canonical JSON or ``"none"`` -> calls) and the calls whose line does
    not record one (``unrecorded``), summed over every cell."""
    out: dict[str, dict[str, Any]] = {
        m: {"routings": {}, "unrecorded": 0} for m in protocol.CHAT_MODELS
    }
    for cell in cells:
        reported = cell.attempt(cell.reported)
        ledger = (reported.ledger if reported else None) or {}
        for model, counts in (ledger.get("routing") or {}).items():
            if model not in out:
                continue
            routings = out[model]["routings"]
            for key, n in counts.items():
                routings[key] = routings.get(key, 0) + int(n)
        for model, n in (ledger.get("routing_unrecorded") or {}).items():
            if model in out:
                out[model]["unrecorded"] += int(n)
    for entry in out.values():
        entry["routings"] = dict(sorted(entry["routings"].items()))
    return out


def _routing_text(key: str) -> str:
    if key == "none":
        return "unrestricted"
    try:
        routing = json.loads(key)
    except ValueError:
        return key
    if isinstance(routing, Mapping) and set(routing) == {"ignore"}:
        return "ignores " + ", ".join(str(s) for s in routing["ignore"])
    return json.dumps(routing, sort_keys=True)


def provider_routing_text(
    observed: Mapping[str, Any] | None, policy: Mapping[str, Any] | None
) -> str | None:
    """The header's line on the routing the ledgers show, None for a report
    that has no such record (an older report's data)."""
    if not observed:
        return None
    policy = policy or {}
    parts: list[str] = []
    differs = False
    for model in protocol.CHAT_MODELS:
        entry = observed.get(model) or {}
        routings = entry.get("routings") or {}
        unrecorded = int(entry.get("unrecorded") or 0)
        expected = (
            json.dumps(policy[model], sort_keys=True) if model in policy else "none"
        )
        seen = [
            f"{_routing_text(key)} ({n} calls)" for key, n in sorted(routings.items())
        ]
        if unrecorded:
            seen.append(f"not recorded ({unrecorded} calls)")
        if not seen:
            parts.append(f"{model}: no calls sent")
            continue
        if len(seen) > 1:
            parts.append(f"{model} mixed: " + ", ".join(seen))
            differs = True
            continue
        parts.append(f"{model} {seen[0]}")
        if unrecorded or set(routings) != {expected}:
            differs = True
    text = "Provider routing the ledgers show: " + "; ".join(parts)
    if differs:
        text += " (**not the protocol's policy throughout**)"
    return text


def _discarded_cost(costs: Mapping[str, Any]) -> str:
    lost = costs.get("learning_discarded") or {}
    if not lost.get("calls"):
        return "-"
    return f"{_usd(lost.get('cost'))} ({lost['calls']} calls)"


def _union_text(union: Mapping[str, Any]) -> str:
    """The unattributed spend over several hosts' gateway runs directories."""
    dirs = ", ".join(f"`{d}`" for d in union.get("gateway_dirs") or ())
    if not union.get("shared") or "unattributed_usd" not in union:
        return (
            f"Over all hosts ({dirs}): not computed: {union.get('detail')}. Each "
            "figure above counts the key's whole usage, the other hosts' spend included."
        )
    return (
        f"**Over all hosts** ({dirs}; one budget start at "
        f"${float(union['start_usage_usd']):.2f}, {union['window_start']} to "
        f"{union['window_end']}): the key's usage grew "
        f"{_usd(union['key_usage_delta_usd'])}; the ledgers of every host recorded "
        f"{_usd(union['ledger_cost_usd'])} over {union['ledger_calls']} call(s); "
        f"**unattributed spend {_usd(union['unattributed_usd'])}** "
        f"({union['unknown_cost_calls']} cut-off call(s) "
        "of unknown cost). The per-host figures above each count the other hosts' "
        "spend as unattributed."
    )


def _resumed_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """'Resumed learning': every learning resume of the reported attempts: the checkpoint
    restored, what was in flight, and the discarded calls and their cost."""
    lines = [
        "## Resumed learning",
        "",
        "An interrupted learning phase resumed from its latest checkpoint: what came "
        "after the checkpoint was rolled back and done again (`discarded.jsonl`). The "
        "discarded calls were made and billed; their cost is part of the learning side.",
        "",
    ]
    rows = []
    for cell in cells:
        for seg in cell.get("resumes") or ():
            redone = seg.get("in_flight_position")
            rows.append(
                (
                    f"`{cell['cell_id']}`",
                    seg.get("segment"),
                    f"`{seg.get('checkpoint')}`" if seg.get("checkpoint") else "setup",
                    seg.get("failure_code") or seg.get("reason") or "-",
                    "-"
                    if redone is None
                    else f"{redone} ({seg.get('in_flight_stage') or '-'})",
                    ", ".join(str(p) for p in seg.get("rows_moved") or ()) or "-",
                    ", ".join(
                        f"{r} {n}" for r, n in _by_count(seg.get("calls_by_role") or {})
                    )
                    or "-",
                    _usd(seg.get("cost"))
                    + (
                        f" (+{seg['unknown_cost_calls']} cut off)"
                        if seg.get("unknown_cost_calls")
                        else ""
                    ),
                )
            )
    lines += _table(
        [
            "cell",
            "segment",
            "restored",
            "why",
            "redone from position (stage)",
            "rows rolled back",
            "discarded calls",
            "cost",
        ],
        rows,
    )
    return lines + [""]


def _host(host: Mapping[str, Any] | None) -> str:
    return host_text(host) if host else "-"


#: The header's label of each merged-view warning.
_MERGED_VIEW_LABELS = {
    "merged_view_not_final": "Merged view not final",
    "merged_view_conflict": "Merged view conflict",
    "merged_view_unreadable": "Merged view unreadable",
}


def _protocol_text(cell: Mapping[str, Any], proto: Mapping[str, Any]) -> str:
    """A cell's protocol in the Cells table: its fingerprint's first 12 hex, marked
    "(earlier)" for a known earlier version and "(**unknown**)" for one no freeze
    produced. A report JSON from before the cells recorded
    ``protocol`` is read from its ``protocol_fingerprint``."""
    recorded = cell.get("protocol")
    if not isinstance(recorded, Mapping):
        fingerprint = cell.get("protocol_fingerprint")
        current = fingerprint == proto.get("frozen_fingerprint")
        recorded = {"fingerprint": fingerprint, "current": current, "known": current}
    fingerprint = recorded.get("fingerprint")
    if not fingerprint:
        return "-"
    text = f"`{str(fingerprint)[:12]}`"
    if recorded.get("current"):
        return text
    return text + (" (earlier)" if recorded.get("known") else " (**unknown**)")


def _code_text(code: Mapping[str, Any] | None) -> str:
    """A cell's harness digest (first 12 hex) in the Cells table."""
    harness = (code or {}).get("harness")
    return f"`{str(harness)[:12]}`" if harness else "-"


def _comparison_versions_text(row: Mapping[str, Any]) -> str:
    """The comparisons table's version marker: ``same``, or what differs between the
    two cells (compared, and flagged); ``-`` for a row of a report
    JSON from before rows carried versions."""
    if "same_protocol" not in row and "same_code" not in row:
        return "-"
    differ = [
        name
        for name, key in (("protocol", "same_protocol"), ("code", "same_code"))
        if row.get(key) is False
    ]
    unknown = [
        name
        for name, key in (("protocol", "same_protocol"), ("code", "same_code"))
        if key in row and row.get(key) is None
    ]
    parts = []
    if row.get("same_protocol") == "unknown":
        parts.append(
            "**protocol not registered: "
            + ", ".join(row.get("protocol_unknown") or ())
            + "**"
        )
    if row.get("mixed_version"):
        parts.append("mixed version: " + ", ".join(row["mixed_version"]))
    if row.get("same_platform") is False:
        parts.append("**another platform**")
    if differ:
        parts.append(f"**differ: {', '.join(differ)}**")
    if unknown:
        parts.append(f"not recorded: {', '.join(unknown)}")
    return "; ".join(parts) or "same"


def _metric_label(benchmark: Any, name: str) -> str:
    """A metric's column label: its :data:`METRICS` label, else its name (a report JSON
    of another schema, whose metrics this code no longer defines)."""
    return next(
        (m.label for m in METRICS.get(str(benchmark), ()) if m.name == name), name
    )


def _removed_cells(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The cells a report set aside as removed methods' (``removed.cells``); none for a
    report JSON from before the removal (its cells, a removed method's included, are
    ordinary rows, rendered as they were written)."""
    removed = report.get("removed")
    if not isinstance(removed, Mapping):
        return []
    return [c for c in removed.get("cells") or () if isinstance(c, Mapping)]


def _removed_methods_named(report: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """``(method, entry)`` of each removed method with a cell in the report."""
    removed = report.get("removed")
    methods = removed.get("methods") if isinstance(removed, Mapping) else None
    return [
        (method, entry)
        for method, entry in sorted((methods or {}).items())
        if isinstance(entry, Mapping) and entry.get("cells")
    ]


def _removed_header_lines(report: Mapping[str, Any]) -> list[str]:
    """The header's line on the removed methods' cells, when the
    runs directory holds any."""
    cells = _removed_cells(report)
    if not cells:
        return []
    named = ", ".join(
        f"`{method}` ({entry.get('decision')}; {len(entry['cells'])} cell(s))"
        for method, entry in _removed_methods_named(report)
    )
    removed = report["removed"]
    deleted = (
        removed.get("deleted") if isinstance(removed.get("deleted"), Mapping) else {}
    )
    return [
        f"- **Removed methods:** {named}. Their cells are set aside: not compared, "
        "audited or summarized, and in no table below but 'Removed methods'; "
        f"their ledgers' {_usd(removed.get('cost_usd'))} is counted apart from the "
        "matrix total (see 'Cost')."
        + (
            f" The data of {deleted.get('cells')} cell(s) was deleted; their money is "
            "kept from the removal "
            f"manifest(s) ({', '.join(deleted.get('removal_ids') or ())})."
            if deleted.get("cells")
            else ""
        )
    ]


def _removed_section(report: Mapping[str, Any]) -> list[str]:
    """'Removed methods': why each method left the study, and its cells as they are on
    disk, with what their ledgers cost."""
    cells = _removed_cells(report)
    if not cells:
        return []
    lines = [
        "",
        "## Removed methods",
        "",
        "Methods removed from the study (`study.REMOVED_METHODS`). Their cells stay on "
        "disk with their gateway runs, ledgers and blobs until the operator deletes them "
        "(`tools/remove_method_data.py`, which first writes their spend to a "
        "manifest), but none is compared, audited or summarized, and none counts as a "
        "cell not complete or not audited. Their calls were billed: their costs are "
        "below and in 'Cost', and every spend window sums their ledgers (a deleted "
        "ledger's calls from its manifest).",
        "",
    ]
    for method, entry in _removed_methods_named(report):
        lines.append(f"- `{method}`, {entry.get('decision')}: {entry.get('reason')}")
    lines.append("")
    lines += _table(
        [
            "cell",
            "status",
            "complete",
            "attempt",
            "attempts",
            "learning",
            "test",
            "invalidated attempts",
            "total",
            "not counted",
            *(["on disk"] if any("on_disk" in c for c in cells) else []),
        ],
        [
            (
                f"`{c.get('cell_id')}`"
                + (
                    f" (directory `{c.get('cell_dir')}`)"
                    if c.get("cell_dir") not in (None, c.get("cell_id"))
                    else ""
                ),
                c.get("status") or "-",
                _fmt(c.get("complete")),
                _fmt(c.get("reported_attempt")),
                _fmt(c.get("attempts")),
                _usd(((c.get("costs") or {}).get("learning") or {}).get("cost")),
                _usd(((c.get("costs") or {}).get("test") or {}).get("cost")),
                _usd(((c.get("costs") or {}).get("other_attempts") or {}).get("cost")),
                _usd((c.get("costs") or {}).get("total_cost")),
                c.get("not_counted") or "-",
                *([_on_disk_text(c)] if any("on_disk" in c for c in cells) else []),
            )
            for c in cells
        ],
    )
    return lines


def _on_disk_text(cell: Mapping[str, Any]) -> str:
    """A removed cell's data: on disk, or deleted (when, where, which manifest)."""
    if cell.get("on_disk", True):
        return "yes"
    deleted = cell.get("deleted") or {}
    return (
        f"deleted {deleted.get('at') or '?'} on {deleted.get('host') or '?'}, "
        f"manifest `{deleted.get('manifest')}`"
    )


def _removed_cost_lines(costs: Mapping[str, Any]) -> list[str]:
    """The Cost section's paragraph on the removed methods' cells: their ledgers' share,
    apart from the matrix total, and every cell together (the money stays counted)."""
    removed = costs.get("removed")
    if not isinstance(removed, Mapping) or not (
        removed.get("by_method") or removed.get("total")
    ):
        return []
    incomplete = removed.get("incomplete") or {}
    # The removed cells' calls of unknown cost are named here too (a report JSON written
    # before the count has no unfinished_calls: none named).
    unknown = [
        f"{n} {what}"
        for n, what in (
            (int(removed.get("unknown_cost_calls") or 0), "cut-off"),
            (int(removed.get("unfinished_calls") or 0), "unfinished"),
        )
        if n
    ]
    return [
        f"**Removed methods' cells** (see 'Removed methods'): their ledgers recorded "
        f"{_usd(removed.get('total'))} more, not in the matrix total ("
        + ", ".join(
            f"{m} {_usd(v)}" for m, v in (removed.get("by_method") or {}).items()
        )
        + (
            f"; {len(incomplete)} of the cells not complete: "
            + ", ".join(f"`{c}` {_usd(v)}" for c, v in incomplete.items())
            if incomplete
            else ""
        )
        + f"; of which {_usd(removed.get('invalidated_attempts'))} in invalidated "
        "attempts"
        + (
            f"; not counting their {' and '.join(unknown)} call(s) of unknown cost, "
            "named under 'Cost not in the totals'"
            if unknown
            else ""
        )
        + f"). **All cells: {_usd(costs.get('all_cells_total'))}** (the matrix total "
        "and the removed methods' cells).",
        "",
    ]


def _replaced_by(
    fingerprint: str, known: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The known version frozen next after ``fingerprint`` (by ``frozen_at``), as
    ``{fingerprint, frozen_at, reason}``: the freeze whose reason says why ``fingerprint``
    became an earlier version (the audit's ``_replaced_by``). None when a time is missing."""

    def at(version: Mapping[str, Any]) -> datetime | None:
        try:
            when = datetime.fromisoformat(str(version.get("frozen_at")))
        except ValueError:
            return None
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)

    own = at(known.get(fingerprint) or {})
    if own is None:
        return None
    later = [
        (when, fp, version)
        for fp, version in known.items()
        if (when := at(version)) is not None and when > own
    ]
    if not later:
        return None
    _when, fp, version = min(later, key=lambda item: item[0])
    return {
        "fingerprint": fp,
        "frozen_at": version.get("frozen_at"),
        "reason": version.get("reason"),
    }


def _version_header_lines(proto: Mapping[str, Any]) -> list[str]:
    """The header's version lines: the cells under an earlier
    protocol version (reported, not re-run; each version's freeze reason), and those
    under a fingerprint no freeze produced (the audit fails them). A report JSON from
    before the versions were split lists ``cells_not_frozen`` alone."""
    if "cells_earlier_version" not in proto:
        if not proto.get("cells_not_frozen"):
            return []
        return [
            "- **Cells whose fingerprint is not the frozen one (H10.6):** "
            + ", ".join(f"`{c}`" for c in proto["cells_not_frozen"])
        ]
    lines = []
    if proto.get("cells_earlier_version"):
        lines.append(
            "- Cells under an earlier protocol version (reported, not re-run): "
            + ", ".join(f"`{c}`" for c in proto["cells_earlier_version"])
        )
        for fingerprint, version in sorted(
            (proto.get("earlier_versions") or {}).items()
        ):
            lines.append(
                f"  - `{fingerprint[:12]}` (frozen {version.get('frozen_at')}): "
                f"{version.get('reason')}"
            )
            replaced = version.get("replaced_by")
            if replaced:
                lines.append(
                    f"    - replaced by `{str(replaced.get('fingerprint'))[:12]}` "
                    f"(frozen {replaced.get('frozen_at')}): {replaced.get('reason')}"
                )
            if version.get("differs_from_frozen"):
                lines.append(
                    f"    - differs from the frozen value: {version['differs_from_frozen']}"
                )
    if proto.get("cells_unknown_version"):
        lines.append(
            "- **Cells under a protocol version no freeze registered** (compared, marked "
            "same_protocol = unknown; the audit warns): "
            + ", ".join(f"`{c}`" for c in proto["cells_unknown_version"])
        )
    if proto.get("cells_mixed_version"):
        lines.append(
            "- Cells that ran more than one protocol or code version (see 'Versions per "
            "cell'; compared, marked mixed): "
            + ", ".join(f"`{c}`" for c in proto["cells_mixed_version"])
        )
    return lines


#: Warnings the header already shows on a line of their own.
_WARNINGS_SHOWN_ELSEWHERE = frozenset({"unknown_protocol_version"})


def _warning_lines(report: Mapping[str, Any]) -> list[str]:
    """The header's warnings (schema 3: recorded, not blocking):
    one line each, except those the header shows on their own line."""
    return [
        f"- **Warning ({w.get('check')}):** {w.get('detail')}"
        for w in report.get("warnings") or ()
        if isinstance(w, Mapping) and w.get("check") not in _WARNINGS_SHOWN_ELSEWHERE
    ]


def _hosts_text(cell: Mapping[str, Any]) -> str:
    """The Cells table's host: every host the reported attempt ran on when more than one
    (its version segments), else the attempt's host."""
    hosts = [h for h in cell.get("hosts") or () if isinstance(h, Mapping)]
    if len(hosts) > 1:
        return "; ".join(
            f"{host_text(h)} [{h.get('platform') or '-'}] (segments "
            + ", ".join(str(n) for n in h.get("segments") or ())
            + ")"
            for h in hosts
        )
    return _host(cell.get("host"))


def _versions_of(cell: Mapping[str, Any]) -> set[tuple[Any, Any, Any]]:
    return {
        (s.get("protocol_fingerprint"), s.get("harness"), s.get("mirix"))
        for s in (cell.get("versions") or {}).get("segments") or ()
        if isinstance(s, Mapping)
    }


def _mixed_text(cell: Mapping[str, Any]) -> str:
    """`` (mixed: N versions)`` for a cell whose reported attempt ran more than one
    protocol or code version, else empty."""
    if not (cell.get("versions") or {}).get("mixed"):
        return ""
    return f" (mixed: {len(_versions_of(cell))} versions)"


def _not_reverified_count(cell: Mapping[str, Any]) -> str:
    detail = cell.get("audit_detail")
    count = len((detail or {}).get("not_reverified") or ()) if detail else 0
    return str(count) if count else "-"


def _score_text(entry: Mapping[str, Any]) -> str:
    """An attempt's or a directory's score: its metrics."""
    metrics = entry.get("metrics")
    if metrics is None and isinstance(entry.get("test"), Mapping):
        metrics = entry["test"].get("metrics")
    if not metrics:
        return "-"
    return ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(metrics.items()))


def _why_text(attempt: Mapping[str, Any]) -> str:
    superseded = attempt.get("superseded")
    if isinstance(superseded, Mapping):
        return (
            f"superseded by attempt {superseded.get('by_attempt') or '?'}: "
            f"{superseded.get('note') or '-'}"
        )
    if attempt.get("note"):
        return f"--note: {attempt['note']}"
    if attempt.get("invalidation"):
        return f"invalidated ({attempt['invalidation']})"
    return "-"


def _attempt_protocol_text(attempt: Mapping[str, Any]) -> str:
    """An attempt's protocol in 'Runs and attempts': its fingerprint's first 12 hex,
    ``-`` without one (or in a report JSON from before attempts listed versions)."""
    fingerprint = _mapping(attempt.get("protocol")).get("fingerprint")
    return f"`{str(fingerprint)[:12]}`" if fingerprint else "-"


def _attempt_code_text(attempt: Mapping[str, Any]) -> str:
    """An attempt's code in 'Runs and attempts': the harness and MIRIX digests and the
    commit (first 12 hex each; the commit marked ``(dirty)`` for a dirty tree), ``-``
    when it records none."""
    code = _mapping(attempt.get("code"))
    parts = [
        f"{name} `{str(code[name])[:12]}`"
        for name in ("harness", "mirix", "commit")
        if code.get(name)
    ]
    if parts and code.get("commit") and code.get("dirty"):
        parts[-1] += " (dirty)"
    return ", ".join(parts) or "-"


def _attempt_rows(
    cell_id: str, directory: str, attempts: Sequence[Mapping[str, Any]]
) -> list[tuple[Any, ...]]:
    """One table row per attempt. ``reported`` is per directory, so a directory the
    report does not use (``directory_used`` False: another directory of the cell id was
    chosen) never says **yes**: its own reported attempt says ``no (directory not
    used)``."""
    return [
        (
            f"`{cell_id}`",
            f"`{directory}`",
            a.get("attempt"),
            str(a.get("state"))
            + (
                f" (was {a['previous_state']})"
                if a.get("previous_state") and a.get("state") == ATTEMPT_INVALID
                else ""
            ),
            _why_text(a),
            a.get("started_at") or "-",
            _host(a.get("host")),
            _attempt_protocol_text(a),
            _attempt_code_text(a),
            a.get("version_segments", "-"),
            a.get("test_items_scored"),
            _score_text(a),
            (
                (
                    "**yes**"
                    if a.get("directory_used", True)
                    else "no (directory not used)"
                )
                if a.get("reported")
                else "no"
            ),
        )
        for a in attempts
    ]


def _runs_section(
    report: Mapping[str, Any], reran: Sequence[Mapping[str, Any]]
) -> list[str]:
    """'Runs and attempts': the cells run more than once (the single-run exception) with every
    attempt and directory, each with its own score; only the reported one's is in any
    other table. A report JSON without ``attempts_detail`` renders its earlier table."""
    new = any("attempts_detail" in c for c in reran)
    lines = [
        "",
        "## Runs and attempts" if new else "## Cells run more than once",
        "",
        "The exception to the single-run caveat: these cells ran more than once. Each "
        "reports its newest "
        "complete run; the other runs are listed with their own scores, and no score "
        "combines two runs. Operator notes are `run.py --acknowledge-rerun`; the "
        "earlier gateway runs are the audit's; a superseded attempt is a `run.py "
        "--new-attempt --note` rerun.",
        "",
    ]
    noted = [
        c
        for c in reran
        if c["reruns"].get("earlier_tested_runs")
        or c["reruns"].get("operator_notes")
        or not new
    ]
    if noted:
        lines += _table(
            ["cell", "earlier runs that ran test items", "operator notes"],
            [
                (
                    f"`{c['cell_id']}`",
                    "; ".join(
                        f"`{r['run_id']}` ({r['test_actor_calls']} test actor calls"
                        + (
                            f", from {r['first_test_actor_at']}"
                            if r.get("first_test_actor_at")
                            else ""
                        )
                        + ")"
                        for r in c["reruns"]["earlier_tested_runs"]
                    )
                    or "-",
                    "; ".join(
                        f"`{n['file']}` ({n['at']}; covers "
                        + (
                            ", ".join(f"`{r['run_id']}`" for r in n["superseded"])
                            or "-"
                        )
                        + ")"
                        for n in c["reruns"]["operator_notes"]
                    )
                    or "**none**",
                )
                for c in noted
            ],
        )
        lines.append("")
    if not new:
        return lines
    others = {
        d["cell_id"]: d["not_used"]
        for d in report.get("duplicates") or ()
        if isinstance(d, Mapping)
    }
    rows = []
    for c in reran:
        rows += _attempt_rows(
            c["cell_id"], str(c.get("cell_dir") or c["cell_id"]), c["attempts_detail"]
        )
        for row in others.get(c["cell_id"], ()):
            rows += _attempt_rows(
                c["cell_id"], str(row["cell_dir"]), row.get("attempts_detail") or ()
            )
    lines += _table(
        [
            "cell",
            "directory",
            "attempt",
            "state",
            "why",
            "started",
            "host",
            "protocol",
            "code",
            "segments",
            "items scored",
            "score",
            "reported",
        ],
        rows,
    )
    return lines


def _duplicates_section(report: Mapping[str, Any]) -> list[str]:
    """'Duplicate cell directories': every cell id held by more than one directory (the
    newest complete one is reported, the others listed with their scores; the money
    of each counts)."""
    duplicates = [d for d in report.get("duplicates") or () if isinstance(d, Mapping)]
    if not duplicates:
        return []
    by_id = {c["cell_id"]: c for c in report.get("cells") or ()}
    rows = []
    for entry in duplicates:
        used = by_id.get(entry["cell_id"]) or {}
        rows.append(
            (
                f"`{entry['cell_id']}`",
                f"`{entry['used'].get('cell_dir')}`",
                _host(used.get("host")),
                used.get("status") or "-",
                entry["used"].get("started_at") or "-",
                _score_text(used) if used.get("complete") else "-",
                _usd((used.get("costs") or {}).get("total_cost")),
                (used.get("audit") or {}).get("status", "-"),
                "**yes**",
            )
        )
        for row in entry.get("not_used") or ():
            rows.append(
                (
                    f"`{row['cell_id']}`",
                    f"`{row['cell_dir']}`",
                    _host(row.get("host")),
                    row.get("status") or "-",
                    row.get("started_at") or "-",
                    _score_text(row) if row.get("complete") else "-",
                    _usd((row.get("costs") or {}).get("total_cost")),
                    (row.get("audit") or {}).get("status", "-"),
                    f"no ({row.get('role')})",
                )
            )
    return [
        "",
        "## Duplicate cell directories",
        "",
        "More than one directory holds these cell ids (a rerun in a new directory, or a "
        "cell run on two hosts). The newest complete one is reported (its reported "
        "attempt's started_at, then cell.json created_at, then the name); the others "
        "are listed with their scores, compared with nothing, and their money counts in "
        "the matrix total.",
        "",
        *_table(
            [
                "cell",
                "directory",
                "host",
                "status",
                "started",
                "score",
                "cost",
                "audit",
                "used",
            ],
            rows,
        ),
    ]


#: How many raw ids a degraded-items row names before it only counts.
DEGRADED_SHOWN = 20


def _degraded_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """'Items scored under degraded conditions' (scored and recorded): the test items
    whose provider refused the actor (two consecutive no-endpoint refusals; scored
    unanswered) or whose MIRIX retrieval was degraded, and the accuracy without the
    provider-unserved items."""
    rows = []
    for c in cells:
        degraded = c.get("degraded") or {}
        without = (c.get("test") or {}).get("metrics_without_provider_unserved")
        for flag, ids in sorted((degraded.get("test") or {}).items()):
            shown = ", ".join(f"`{i}`" for i in ids[:DEGRADED_SHOWN])
            more = len(ids) - DEGRADED_SHOWN
            rows.append(
                (
                    f"`{c['cell_id']}`",
                    f"{flag} (test)",
                    f"{len(ids)}: {shown}" + (f" and {more} more" if more > 0 else ""),
                    ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(without.items()))
                    if flag == ROW_PROVIDER_UNSERVED and without
                    else "-",
                )
            )
        for flag, n in sorted((degraded.get("learning") or {}).items()):
            rows.append((f"`{c['cell_id']}`", f"{flag} (learning)", n, "-"))
    if not rows:
        return []
    return [
        "## Items scored under degraded conditions",
        "",
        f"provider_unserved: {UNSERVED_TEXT}; retrieval_degraded: MIRIX's "
        "retrieval lost its topic extraction or query vector and was injected anyway. "
        "They count in every number above; the last column is the accuracy without "
        "the provider-unserved items.",
        "",
        *_table(
            ["cell", "condition", "items", "without provider-unserved items"], rows
        ),
        "",
    ]


def _provider_accuracy_section(report: Mapping[str, Any]) -> list[str]:
    """'Accuracy by main provider': per chat model served by at least two
    providers, each benchmark's items by the provider that served most of their actor
    calls, and the accuracy of each group (from the records, no rerun)."""
    accuracy = report.get("provider_accuracy")
    if not isinstance(accuracy, Mapping):
        return []
    endpoints = report.get("provider_endpoints") or {}
    sources = "; ".join(
        f"`{s.get('file')}` ({s.get('_source') or 'no source recorded'})"
        for s in endpoints.get("sources") or ()
    )
    lines: list[str] = []
    for model, benchmarks in sorted(accuracy.items()):
        providers = {p for by in benchmarks.values() for p in by}
        if len(providers) < 2:
            continue
        if not lines:
            lines = [
                "",
                "### Accuracy by main provider",
                "",
                CAVEAT_PROVIDER_ACCURACY,
                "",
                CAVEAT_QUANTIZATION
                + (f" Listing: {sources}." if sources else " No listing was given."),
                "",
            ]
        lines += [f"#### {model}", ""]
        lines += _table(
            [
                "benchmark",
                "provider",
                "quantization",
                "items",
                "of which mixed",
                "accuracy",
                "share of items",
            ],
            [
                (
                    benchmark,
                    provider,
                    e.get("quantization"),
                    e.get("items"),
                    e.get("mixed_items"),
                    _fmt(e.get("accuracy")),
                    f"{float(e['share_of_items']):.0%}"
                    if e.get("share_of_items") is not None
                    else "-",
                )
                for benchmark, by in sorted(benchmarks.items())
                for provider, e in sorted(
                    by.items(), key=lambda kv: (-int(kv[1].get("items") or 0), kv[0])
                )
            ],
        )
        lines.append("")
    return lines


def _lost_text(mirix: Mapping[str, Any]) -> str:
    lost = mirix.get("learning_calls_lost")
    if lost is None:
        return "-"
    tolerated = mirix.get("tolerated") or {}
    method = ", ".join(f"{k} {v}" for k, v in sorted(tolerated.items()))
    return f"{lost}" + (f" ({method})" if method else "")


def _refusals_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """'Refused learning calls': the learning methods' calls the provider refused,
    counted and gone on from, with the refused inductions' HTTP
    codes (AWM kept its previous workflow)."""
    rows = [
        (
            f"`{c['cell_id']}`",
            c["method"],
            ", ".join(f"{k} {v}" for k, v in c["method_refusals"]["counters"].items()),
            ", ".join(
                f"{k} x{v}"
                for k, v in (
                    c["method_refusals"].get("rejected_by_status") or {}
                ).items()
            )
            or "-",
            "-"
            if c["method_refusals"].get("inductions") is None
            else c["method_refusals"]["inductions"],
        )
        for c in cells
        if c.get("method_refusals") and any(c["method_refusals"]["counters"].values())
    ]
    if not rows:
        return []
    return [
        "",
        "### Refused learning calls (counted and continued)",
        "",
        *_table(
            [
                "cell",
                "method",
                "counters",
                "refused inductions by HTTP status",
                "inductions",
            ],
            rows,
        ),
    ]


def _audit_warnings_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """'Audit warnings': each cell's audit warnings and recorded conditions per check."""
    rows = []
    for c in cells:
        detail = c.get("audit_detail") or {}
        for check, n in sorted((detail.get("warnings_by_check") or {}).items()):
            rows.append((f"`{c['cell_id']}`", check, "warning", n))
        for check, n in sorted((detail.get("recorded_counts") or {}).items()):
            rows.append((f"`{c['cell_id']}`", check, "recorded", n))
    if not rows:
        return []
    return [
        "",
        "### Audit warnings",
        "",
        "What the audit wrote down instead of failing the cell (record, don't block).",
        "",
        *_table(["cell", "check", "kind", "count"], rows),
    ]


def _versions_section(cells: Sequence[Mapping[str, Any]]) -> list[str]:
    """'Versions per cell': every version segment of the cells that ran more than one
    version, have checks the audit could not re-verify, or have audit warnings."""
    shown = [
        c
        for c in cells
        if (c.get("versions") or {}).get("mixed")
        or len((c.get("versions") or {}).get("segments") or ()) > 1
        or (c.get("audit_detail") or {}).get("not_reverified")
        or (c.get("audit_detail") or {}).get("warnings_by_check")
    ]
    if not shown:
        return []
    rows = []
    for c in shown:
        for seg in (c.get("versions") or {}).get("segments") or ():
            changed = seg.get("changed") or {}
            files = changed.get("protocol_files") or {}
            code_files = changed.get("code_files") or {}
            what = [
                f"{kind} {', '.join(paths)}"
                for kind, paths in sorted(files.items())
                if paths
            ] + [
                f"{part} {kind} {', '.join(paths)}"
                for part, by in sorted(code_files.items())
                if by
                for kind, paths in sorted(by.items())
                if paths
            ]
            if changed.get("constants_changed_keys"):
                what.append("constants " + ", ".join(changed["constants_changed_keys"]))
            start = seg.get("from") or {}
            rows.append(
                (
                    f"`{c['cell_id']}`",
                    seg.get("segment"),
                    f"{seg.get('phase') or '-'} "
                    + (
                        ", ".join(f"{k} {v}" for k, v in sorted(start.items()))
                        if isinstance(start, Mapping) and start
                        else "-"
                    ),
                    f"`{str(seg.get('protocol_fingerprint'))[:12]}`"
                    if seg.get("protocol_fingerprint")
                    else "-",
                    _code_text(seg),
                    _host(seg.get("host")),
                    "; ".join(what) or ("-" if not changed else "commit or tree only"),
                )
            )
        for check in (c.get("audit_detail") or {}).get("not_reverified") or ():
            rows.append(
                (
                    f"`{c['cell_id']}`",
                    "not re-verified",
                    check.get("check"),
                    "-",
                    "-",
                    "-",
                    "; ".join(check.get("inputs") or ()) or "-",
                )
            )
    return [
        "",
        "## Versions per cell",
        "",
        "The versions each reported attempt ran under (attempt.json version_segments; "
        "a resume under another version goes on and is recorded), "
        "and the checks the audit could not re-verify under an earlier version.",
        "",
        *_table(
            ["cell", "segment", "from", "protocol", "code", "host", "changed"], rows
        ),
    ]


def _invalid_text(failures: Mapping[str, Any]) -> str:
    superseded = failures.get("superseded_attempts") or ()
    return str(failures.get("invalid_attempts")) + (
        f" (superseded: {', '.join(str(n) for n in superseded)})" if superseded else ""
    )


def render_markdown(report: Mapping[str, Any]) -> str:  # noqa: C901 - one section each
    proto = report["protocol"]
    stats = report["statistics"]
    cells = list(report["cells"])
    lines = [
        "# skill_bench report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Runs: `{report['runs_dir']}` (split mode: {report['split_mode']})",
        f"- Protocol: {proto['protocol_id']}, frozen fingerprint "
        f"`{proto['frozen_fingerprint']}`",
        f"- Tree the report ran on: `{proto['report_tree_fingerprint']}`",
        f"- {provider_policy_text(proto.get('provider_policy'))}",
    ]
    routing = provider_routing_text(
        proto.get("provider_routing_observed"), proto.get("provider_policy")
    )
    if routing is not None:
        lines.append(f"- {routing}")
    lines += _version_header_lines(proto)
    audit = report.get("audit") or {}
    if not audit.get("found"):
        lines.append(
            f"- **No audit** (`{audit.get('path')}`): no number below is vouched for by "
            "`audit.py`; run it before publishing."
        )
    else:
        lines.append(
            f"- Audit `{audit['path']}` ({audit.get('generated_at')}): "
            + ("passed" if audit.get("passed") else "**not passed**")
        )
    if audit.get("cells_not_passed"):
        # Only complete cells are ever compared: a running or stopped one is in no
        # comparison, audit or not.
        is_complete = {c["cell_id"]: bool(c.get("complete")) for c in cells}
        unaudited = [c for c in audit["cells_not_passed"] if is_complete.get(c)]
        not_complete = [c for c in audit["cells_not_passed"] if not is_complete.get(c)]
        parts = []
        if unaudited:
            parts.append(
                ", ".join(f"`{c}`" for c in unaudited)
                + (
                    " (left out of the comparisons)"
                    if audit.get("required")
                    else " (still compared; `--require-audit` leaves them out)"
                )
            )
        if not_complete:
            parts.append(
                ", ".join(f"`{c}`" for c in not_complete)
                + " (not complete: in no comparison)"
            )
        lines.append(
            "- **Cells without a passing audit of these records:** " + "; ".join(parts)
        )
    lines += _removed_header_lines(report)
    lines += _warning_lines(report)
    merged = report.get("merged_view")
    if merged:
        lines.append(f"- Merged view (`_merged.json`): {merged.get('summary')}")
        for warning in merged.get("warnings") or ():
            label = _MERGED_VIEW_LABELS.get(warning.get("check"), "Merged view")
            lines.append(f"- **{label}:** {warning.get('detail')}")
    for error in report.get("errors") or ():
        lines.append(f"- **Error:** {error['detail']}: " + ", ".join(error["cells"]))
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in report["caveats"]]

    lines += ["", "## Cells", ""]
    lines += _table(
        [
            "cell",
            "status",
            "attempt",
            "attempts",
            "host",
            "protocol",
            "code",
            "audit",
            "not re-verified",
        ],
        [
            (
                f"`{c['cell_id']}`"
                + (
                    f" (directory `{c['cell_dir']}`)"
                    if c.get("cell_dir") not in (None, c["cell_id"])
                    else ""
                ),
                c["status"],
                _fmt(c["reported_attempt"]),
                c["attempts"],
                _hosts_text(c),
                _protocol_text(c, proto) + _mixed_text(c),
                _code_text(c.get("code")),
                c["audit"]["status"]
                if c["audit"]["status"] == AUDIT_PASSED
                else f"**{c['audit']['status']}**",
                _not_reverified_count(c),
            )
            for c in cells
        ],
    )

    reran = [c for c in cells if c.get("reruns")]
    if reran:
        lines += _runs_section(report, reran)
    lines += _duplicates_section(report)
    lines += _versions_section(cells)
    lines += _removed_section(report)

    complete = [c for c in cells if c["complete"]]
    lines += ["", "## Main results (test items)", ""]
    for benchmark in protocol.BENCHMARKS:
        rows = [c for c in complete if c["benchmark"] == benchmark]
        if not rows:
            continue
        # The metrics the cells report, labelled from METRICS.
        names = _metric_names(
            benchmark, {n for c in rows for n in c["test"].get("metrics") or {}}
        )
        lines += [f"### {benchmark}", ""]
        lines += _table(
            [
                "group",
                "method",
                "items",
                *[_metric_label(benchmark, n) for n in names],
            ],
            [
                (
                    c["group"],
                    c["method"],
                    c["test"]["items"],
                    *[_fmt(c["test"]["metrics"].get(n)) for n in names],
                )
                for c in sorted(rows, key=lambda c: (c["group"], c["method"]))
            ],
        )
        lines.append("")
    lines += _degraded_section(complete)

    compared = [c for c in report["comparisons"] if "skipped" not in c]
    bootstrapped = any(c.get("test") == "paired_bootstrap" for c in compared)
    lines += [
        "## Comparison with the no-memory method",
        "",
        "Binary metrics: exact two-sided McNemar test (b = the method right and "
        "no-memory wrong, c = the reverse) on the two cells' environment outcomes, "
        "paired item by item."
        + (
            f" Other metrics: {stats['level']:.0%} paired bootstrap interval of the "
            f"mean difference ({stats['resamples']:,} resamples, seed {stats['seed']})."
            if bootstrapped
            else ""
        )
        + f" {CAVEAT_SINGLE_RUN}"
        + (
            " " + caveat_reruns(report["cells_run_more_than_once"])
            if report.get("cells_run_more_than_once")
            else ""
        ),
        "",
    ]
    if compared:
        lines += _table(
            [
                "group",
                "method",
                "metric",
                "items",
                "method",
                "none",
                "delta",
                "test",
                "versions",
            ],
            [
                (
                    c["group"],
                    c["method"],
                    c["metric"],
                    c["items"],
                    _fmt(c["method_value"]),
                    _fmt(c["none_value"]),
                    f"{c['delta']:+.3f}",
                    f"McNemar b={c['b']} c={c['c']} p={_p(c['p_value'])}"
                    if c["test"] == "mcnemar_exact"
                    else f"CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                    _comparison_versions_text(c),
                )
                for c in compared
            ],
        )
    for c in report["comparisons"]:
        if "skipped" in c:
            lines.append(
                f"- {c['group']} / {c['method']}: not compared ({c['skipped']})"
            )

    lines += [
        "",
        "## Learning curves",
        "",
        f"Outcomes of the learning items in stream order, per window of "
        f"{stats['learning_curve_window']} (the scorer's records; no method saw them).",
        "",
    ]
    for c in complete:
        curve = c.get("learning_curve") or []
        if not curve:
            continue
        names = _metric_names(c["benchmark"], curve[0]["metrics"])
        lines += [f"### `{c['cell_id']}`", ""]
        lines += _table(
            ["window", "positions", "items", *names],
            [
                (
                    w["window"],
                    f"{w['first_position']}-{w['last_position']}",
                    w["items"],
                    *[_fmt(w["metrics"][n]) for n in names],
                )
                for w in curve
            ],
        )
        lines.append("")

    costs = report["costs"]
    incomplete = costs.get("incomplete") or {}
    lines += [
        "## Cost",
        "",
        f"From the gateway ledgers (`cost_total`, billed retries included). Matrix total "
        f"{_usd(costs['total'])}"
        + (
            f" (every cell listed, including the ledgers so far of {len(incomplete)} "
            "cell(s) not complete: "
            + ", ".join(f"`{c}` {_usd(v)}" for c, v in incomplete.items())
            + ")"
            if incomplete
            else ""
        )
        + f", of which {_usd(costs['invalidated_attempts'])} in "
        "attempts invalidated or superseded"
        + (
            f" ({_usd(costs['superseded_attempts'])} superseded by a rerun)"
            if costs.get("superseded_attempts")
            else ""
        )
        + (
            f" and {_usd((costs.get('superseded_directories') or {}).get('total'))} in "
            "directories not reported (see 'Duplicate cell directories')"
            if (costs.get("superseded_directories") or {}).get("total")
            else ""
        )
        + "; learning side "
        f"{_usd(costs['by_side'].get('learning'))} (of which "
        f"{_usd(costs.get('learning_discarded') or 0.0)} in tries learning resumes "
        f"rolled back), test {_usd(costs['by_side'].get('test'))}.",
        "",
    ]
    lines += _removed_cost_lines(costs)
    lines += _table(
        [
            "cell",
            "learning",
            "of which discarded",
            "test",
            "other attempts (invalidated or superseded)",
            "total",
            "by role",
            "not counted",
        ],
        [
            (
                label,
                _usd(c["costs"]["learning"]["cost"]),
                _discarded_cost(c["costs"]),
                _usd(c["costs"]["test"]["cost"]),
                _usd(c["costs"]["other_attempts"]["cost"]),
                _usd(c["costs"]["total_cost"]),
                ", ".join(
                    f"{side}/{role} {_usd(v['cost'])}"
                    for side in ("learning", "test")
                    for role, v in sorted(c["costs"][side]["by_role"].items())
                ),
                uncounted_costs(c["costs"]),
            )
            for label, c in [(f"`{c['cell_id']}`", c) for c in cells]
            + [
                (
                    f"`{row['cell_id']}` (directory `{row['cell_dir']}`, not reported)",
                    row,
                )
                for d in report.get("duplicates") or ()
                for row in d.get("not_used") or ()
            ]
        ],
    )
    # The removed methods' cells too (their cut-off and unfinished calls belong in this
    # section, whose spend windows count them): 'All cells' leaves
    # them out as the matrix total leaves the study's out.
    uncounted = [(f"`{c['cell_id']}`", uncounted_costs(c["costs"])) for c in cells] + [
        (
            f"`{c.get('cell_id')}` (removed method"
            + (
                f", directory `{c.get('cell_dir')}`"
                if c.get("cell_dir") not in (None, c.get("cell_id"))
                else ""
            )
            + ")",
            uncounted_costs(c.get("costs") or {}),
        )
        for c in _removed_cells(report)
    ]
    uncounted = [(name, text) for name, text in uncounted if text != "-"]
    if uncounted:
        lines += [
            "",
            "**Cost not in the totals:** "
            + "; ".join(f"{name} {text}" for name, text in uncounted)
            + ". "
            + CAVEAT_UNKNOWN_COST,
        ]
    lines += [
        "",
        "By role: "
        + ", ".join(f"{role} {_usd(v)}" for role, v in costs["by_role"].items()),
        "",
    ]
    for spend in report.get("spend") or ():
        lines += [
            f"Key usage against the ledgers (`{spend['gateway_dir']}`, "
            f"{spend['window_start']} to {spend['window_end']}): the key's usage grew "
            f"{_usd(spend['key_usage_delta_usd'])} (`{spend['source']}`); the ledgers "
            f"recorded {_usd(spend['ledger_cost_usd'])} over "
            f"{spend['ledger_calls']} call(s) that ended in that window; "
            f"**unattributed spend {_usd(spend['unattributed_usd'])}** "
            f"({spend['unknown_cost_calls']} cut-off call(s) of unknown cost in the window"
            + (
                f"; {spend['calls_outside_window']} call(s) "
                f"({_usd(spend['cost_outside_window_usd'])}) ended outside it"
                if spend["calls_outside_window"]
                else ""
            )
            + "). "
            + CAVEAT_UNATTRIBUTED,
            "",
        ]
    union = report.get("spend_union")
    if union:
        lines += [_union_text(union), ""]
    resumed = [c for c in cells if c.get("resumes")]
    if resumed:
        lines += _resumed_section(resumed)

    lines += ["## Providers", "", CAVEAT_PROVIDERS, ""]
    quantized = any("provider_mix" in c for c in cells)
    lines += _table(
        [
            "cell",
            "providers (calls)",
            "model served",
            *(["test actor calls by quantization"] if quantized else []),
        ],
        [
            (
                f"`{c['cell_id']}`",
                ", ".join(
                    f"{p} {n} ({c['providers']['shares'][p]:.0%})"
                    for p, n in _by_count(c["providers"]["overall"])
                )
                or "-",
                "; ".join(
                    f"{m}: " + ", ".join(f"{s} {n}" for s, n in _by_count(served))
                    for m, served in sorted(c["providers"]["model_served"].items())
                )
                or "-",
                *(
                    [
                        ", ".join(
                            f"{q} {n}"
                            for q, n in _by_count(
                                (c.get("provider_mix") or {}).get("quantization_calls")
                                or {}
                            )
                        )
                        or "-"
                    ]
                    if quantized
                    else []
                ),
            )
            for c in cells
        ],
    )
    lines += _provider_accuracy_section(report)

    pricing = report.get("pricing") or {}
    slow_s = float(
        pricing.get("slow_call_s")
        or protocol.CALL_DEADLINE_S * SLOW_CALL_DEADLINE_FRACTION
    )
    lines += ["", "### Price tiers and latency", "", CAVEAT_PRICE_TIERS, ""]
    rows = []
    for model, by_provider in (pricing.get("by_model") or {}).items():
        for provider, e in by_provider.items():
            edge = lambda v, unit="": "-" if v is None else f"<= {v:g}{unit}"  # noqa: E731
            rows.append(
                (
                    model,
                    provider,
                    e["calls"],
                    f"{_fmt(e.get('ratio'))} ({_fmt(e.get('ratio_min'))}-"
                    f"{_fmt(e.get('ratio_max'))})",
                    e["over_tier_calls"],
                    edge(e.get("latency_p50_le_s"), " s"),
                    edge(e.get("latency_p90_le_s"), " s"),
                    f"{float(e['latency_s_max']):.1f} s",
                    e.get("slow_calls", "-"),
                    _fmt(e.get("reasoning_mean"), 1),
                    edge(e.get("reasoning_p95_le")),
                )
            )
    lines += _table(
        [
            "model",
            "provider",
            "ok calls",
            "billed / list price (per call min-max)",
            "calls in the long-input tier",
            "latency p50",
            "latency p90",
            "latency max",
            f"ok calls answered after > {slow_s:g} s",
            "reasoning tokens, mean",
            "reasoning tokens, p95",
        ],
        rows,
    )
    timed_out = pricing.get("timeouts_model_blamed") or {}
    lines += [
        "",
        "Calls that ran past the deadline on the model's own attempt: "
        + (", ".join(f"{m} {n}" for m, n in timed_out.items()) or "none"),
    ]
    if pricing.get("outside_band"):
        low, high = pricing.get("band") or PRICE_RATIO_BAND
        lines += [
            "",
            f"**Billed outside {low}-{high} times the list price:** "
            + ", ".join(pricing["outside_band"])
            + ". Check which service tier served them.",
        ]
    if pricing.get("reasoning_outliers"):
        lines += [
            "",
            f"**Providers reasoning more than {REASONING_OUTLIER_FACTOR} times the "
            "model's median other provider:** "
            + ", ".join(
                f"{o['model']} / {o['provider']} (mean {o['reasoning_mean']:g} "
                f"against {o['median_provider_mean']:g})"
                for o in pricing["reasoning_outliers"]
            )
            + ". The reasoning setting a call gets depends on the provider OpenRouter "
            "picks.",
        ]

    lines += ["", "## Injected tokens", ""]
    lines += _table(
        [
            "cell",
            "phase",
            "items",
            "empty",
            "mean",
            "median",
            "p90",
            "max",
            "x actor calls, mean",
        ],
        [
            (
                f"`{c['cell_id']}`",
                phase,
                v["items"],
                v["empty"],
                _fmt(v["tokens"].get("mean"), 1),
                _fmt(v["tokens"].get("median"), 1),
                _fmt(v["tokens"].get("p90"), 1),
                _fmt(v["tokens"].get("max"), 0),
                _fmt(v["tokens_times_actor_calls"].get("mean"), 1),
            )
            for c in complete
            for phase, v in _phases(c.get("injections") or {})
        ],
    )

    lines += ["", "## Diagnostics", "", "### J against the true outcome", ""]
    lines += [CAVEAT_JUDGE, ""]
    judged = [c for c in complete if c.get("judge_agreement")]
    if judged:
        lines += _table(
            ["cell", "truth", "n", "tp", "fp", "fn", "tn", "accuracy", "kappa"],
            [
                (
                    f"`{c['cell_id']}`",
                    truth,
                    a["n"],
                    a["tp"],
                    a["fp"],
                    a["fn"],
                    a["tn"],
                    _fmt(a.get("accuracy")),
                    _fmt(a.get("kappa")),
                )
                for c in judged
                for truth, a in c["judge_agreement"]["by_truth"].items()
            ],
        )
        untrue = [
            f"- `{c['cell_id']}`: no true outcome to compare "
            f"({c['judge_agreement']['truth_unavailable']})."
            for c in judged
            if c["judge_agreement"].get("truth_unavailable")
        ]
        if untrue:
            lines += ["", *untrue]
    else:
        lines.append("No cell with J.")

    lines += ["", "### MIRIX", ""]
    mirix = [c for c in complete if c.get("mirix")]
    if mirix:
        lines += _table(
            [
                "cell",
                "runs",
                "by status",
                "skills",
                "retrievals (test, empty)",
                "refused",
                "learning calls lost (method: accepted)",
                "LLM usage",
            ],
            [
                (
                    f"`{c['cell_id']}`",
                    c["mirix"]["learning_runs"],
                    ", ".join(
                        f"{k} {v}"
                        for k, v in sorted(c["mirix"]["runs_by_status"].items())
                    ),
                    c["mirix"]["skills"],
                    "{} ({})".format(
                        c["mirix"]["retrievals"].get("test", {}).get("count", 0),
                        c["mirix"]["retrievals"].get("test", {}).get("empty", 0),
                    ),
                    str(c["mirix"]["retrievals"].get("test", {}).get("refused", 0))
                    + (
                        f" (degraded, injected: "
                        f"{c['mirix']['retrievals'].get('test', {}).get('degraded')})"
                        if c["mirix"]["retrievals"].get("test", {}).get("degraded")
                        else ""
                    ),
                    _lost_text(c["mirix"]),
                    ", ".join(
                        f"{k} {v}"
                        for k, v in sorted(
                            c["mirix"]["llm_usage"].items(),
                            key=lambda kv: (LLM_USAGE_ORDER.get(kv[0], 99), kv[0]),
                        )
                    )
                    or "-",
                )
                for c in mirix
            ],
        )
        lines += ["", "#### MIRIX experience pool", ""]
        lines += _table(
            [
                "cell",
                "active at freeze (entered)",
                "dispositions",
                "earlier lessons shown",
                "recurrences",
                "retired",
                "batches without earlier lessons",
            ],
            [_pool_row(c) for c in mirix],
        )
        lines += ["", "#### MIRIX session context and skill feedback", ""]
        lines += _table(
            [
                "cell",
                "learned sessions with the agent context (chars)",
                "skills served (sessions)",
                "feedback entries (served)",
                "followed",
                "effect",
                "stated, unverified (followed, effect)",
                "usage records shown (batches)",
                "citations unverified",
            ],
            [_context_row(c) for c in mirix],
        )
        named = [
            (c["cell_id"], reason, n)
            for c in mirix
            for reason, n in sorted(
                (
                    (c["mirix"].get("experience_pool") or {}).get("history_skipped")
                    or {}
                ).items()
            )
            if reason in POOL_SKIPS_NAMED
        ]
        # A failed embedding that only left pool experiences out (the batch still
        # showed the rest) is counted in the pool's history, not as a skip reason.
        named += [
            (c["cell_id"], "pool_embedding_failed", n)
            for c in mirix
            for n in [
                _count(
                    (
                        (c["mirix"].get("experience_pool") or {}).get("history") or {}
                    ).get("embedding_failed")
                )
            ]
            if n
            and not (
                (c["mirix"].get("experience_pool") or {}).get("history_skipped") or {}
            ).get("embedding_failed")
        ]
        if named:
            lines.append("")
            lines.append(
                "**Batches whose earlier lessons were lost or switched off:** "
                + ", ".join(f"`{c}` {reason} x{n}" for c, reason, n in named)
                + " (embedding_failed: the attempt should have restarted; "
                "pool_embedding_failed: failed embeddings that left pool experiences "
                "out, likewise; disabled / embeddings_disabled: not MIRIX's default)."
            )
        unbalanced = [
            (c["cell_id"], c["mirix"]["experience_pool"])
            for c in mirix
            if (c["mirix"].get("experience_pool") or {}).get("unaccounted")
            not in (0, None)
        ]
        if unbalanced:
            lines.append("")
            lines.append(
                "**Pool metrics that do not add up** (entered - retired - active): "
                + ", ".join(
                    f"`{c}` {p['unaccounted']} (entered {p['entered']}, retired "
                    f"{sum((p.get('retired') or {}).values())}, active "
                    f"{p['active_at_freeze']})"
                    for c, p in unbalanced
                )
                + "."
            )
    else:
        lines.append("No MIRIX cell.")

    lines += ["", "### ALFWorld per task type", "", CAVEAT_TASK_TYPES, ""]
    typed = [c for c in complete if c.get("task_types")]
    if typed:
        types = sorted({t for c in typed for t in c["task_types"]})
        lines += _table(
            ["cell", *types],
            [
                (
                    f"`{c['cell_id']}`",
                    *[
                        "{} (n={})".format(
                            _fmt(c["task_types"][t]["success_rate"]),
                            c["task_types"][t]["items"],
                        )
                        if t in c["task_types"]
                        else "-"
                        for t in types
                    ],
                )
                for c in typed
            ],
        )
    else:
        lines.append("No ALFWorld cell.")

    lines += _refusals_section(complete)
    lines += ["", "### Failures and re-runs", ""]
    lines += _table(
        [
            "cell",
            "invalid attempts",
            "test re-runs",
            "failures",
            "repeat_of",
        ],
        [
            (
                f"`{c['cell_id']}`",
                _invalid_text(c["failures"]),
                c["failures"]["test_reruns"],
                ", ".join(
                    f"{f['phase']}/{f['kind']}/{f['code']} x{f['count']}"
                    for f in c["failures"]["failures"]
                )
                or "-",
                ", ".join(f"{r} {n}" for r, n in c["failures"]["repeat_of"].items())
                or "-",
            )
            for c in cells
        ],
    )
    lines += _audit_warnings_section(cells)
    counters = [c for c in complete if c.get("method_counters")]
    if counters:
        lines += ["", "### Method counters (frozen snapshot)", ""]
        for c in counters:
            lines.append(
                f"- `{c['cell_id']}`: "
                + json.dumps(
                    c["method_counters"], sort_keys=True, separators=(",", ":")
                )
            )
    return "\n".join(lines) + "\n"


def _pool_row(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    """One MIRIX cell's row of the experience-pool table ("not reported" for a MIRIX
    without the experience pool)."""
    pool = cell["mirix"].get("experience_pool")
    if not pool:
        return (f"`{cell['cell_id']}`", "not reported", "-", "-", "-", "-", "-")
    d = pool.get("dispositions") or {}
    retired = pool.get("retired") or {}
    return (
        f"`{cell['cell_id']}`",
        "{} ({})".format(
            "?" if pool.get("active_at_freeze") is None else pool["active_at_freeze"],
            pool.get("entered"),
        ),
        f"used {d.get('used', 0)}, covered {d.get('covered', 0)}, not worth "
        f"{d.get('not_worth', 0)}, deferred {d.get('deferred', 0)} "
        f"({d.get('defaulted', 0)} by default), unaddressed {d.get('unaddressed', 0)}",
        f"{pool.get('history_pushed', 0)}: cited {pool.get('history_cited', 0)}, "
        f"covered {pool.get('history_covered', 0)}, not worth "
        f"{pool.get('history_not_worth', 0)}, ignored {pool.get('history_ignored', 0)}",
        pool.get("recurrences", 0),
        f"used {retired.get('used', 0)}, covered {retired.get('covered', 0)}, "
        f"ignored {retired.get('ignored', 0)}, merged {retired.get('merged', 0)}, "
        f"unembeddable {retired.get('unembeddable', 0)}",
        ", ".join(f"{k} {v}" for k, v in _by_count(pool.get("history_skipped") or {}))
        or "-",
    )


def _context_row(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    """One MIRIX cell's row of the session-context table ("not reported" for a MIRIX
    that records no session context)."""
    context = cell["mirix"].get("learning_context")
    if not context:
        return (f"`{cell['cell_id']}`", "not reported", *["-"] * 7)
    feedback = context.get("feedback") or {}
    usage = context.get("usage") or {}
    return (
        f"`{cell['cell_id']}`",
        "{}/{} ({})".format(
            context.get("with_agent_context", 0),
            context.get("sessions", 0),
            ", ".join(str(n) for n in context.get("agent_context_chars") or ()) or "-",
        ),
        f"{context.get('served_skills', 0)} ({context.get('served_sessions', 0)})",
        f"{feedback.get('given', 0)} ({feedback.get('served', 0)})",
        ", ".join(f"{k} {v}" for k, v in (feedback.get("followed") or {}).items())
        or "-",
        ", ".join(f"{k} {v}" for k, v in (feedback.get("effect") or {}).items()) or "-",
        f"{feedback.get('followed_unverified', 0)}, "
        f"{feedback.get('effect_unverified', 0)}",
        f"{usage.get('records', 0)} ({usage.get('shown', 0)}/{usage.get('batches', 0)})",
        context.get("citations_unverified", 0),
    )


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / REPORT_JSON, out_dir / REPORT_MD
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs-dir", type=Path, default=protocol.RUNS_DIR)
    parser.add_argument("--out", type=Path, help="default <runs-dir>/_report")
    parser.add_argument(
        "--split-mode",
        default="protocol",
        choices=("protocol", "smoke", "pilot", "all"),
        help="which cells to report (default: protocol cells only)",
    )
    parser.add_argument("--resamples", type=int, default=protocol.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=protocol.SEED)
    parser.add_argument(
        "--audit",
        type=Path,
        help="the audit.json to mark cells with (default <runs-dir>/_audit/audit.json)",
    )
    parser.add_argument(
        "--require-audit",
        action="store_true",
        help="compare only cells whose audit passed on the records reported; exit 3 "
        "only when a complete reported cell has no passing audit of its validity "
        "checks (the cells of a removed method, study.REMOVED_METHODS, and directories "
        "another directory of their cell id replaced are set aside and never counted)",
    )
    parser.add_argument(
        "--gateway-runs-dir",
        action="append",
        type=Path,
        default=[],
        help="another directory holding gateway runs (<dir>/<run_id>/gateway.jsonl), "
        "searched after the runs directory (repeatable)",
    )
    parser.add_argument(
        "--provider-endpoints",
        action="append",
        type=Path,
        default=[],
        help="an OpenRouter endpoint listing (GET /api/v1/models/<id>/endpoints JSON) "
        "that labels each provider's quantization (repeatable; read offline)",
    )
    args = parser.parse_args(argv)
    # Absolute, as run.py and schedule.py record every path.
    args.runs_dir = args.runs_dir.absolute()
    report = build_report(
        args.runs_dir,
        split_mode=None if args.split_mode == "all" else args.split_mode,
        resamples=args.resamples,
        seed=args.seed,
        audit_path=args.audit,
        require_audit=args.require_audit,
        gateway_runs_dirs=[d.absolute() for d in args.gateway_runs_dir],
        provider_endpoints=[p.absolute() for p in args.provider_endpoints],
    )
    json_path, md_path = write_report(report, args.out or args.runs_dir / "_report")
    complete = sum(1 for c in report["cells"] if c["complete"])
    unaudited = [
        c["cell_id"]
        for c in report["cells"]
        if c["complete"] and c["audit"]["status"] != AUDIT_PASSED
    ]
    print(
        json.dumps(
            {
                "cells": len(report["cells"]),
                "complete": complete,
                "not_audited": unaudited,
                "errors": len(report["errors"]),
                "warnings": sorted({w["check"] for w in report["warnings"]}),
                "duplicates_not_used": [
                    row["cell_dir"]
                    for d in report["duplicates"]
                    for row in d["not_used"]
                ],
                # Set aside (study.REMOVED_METHODS): never counted in the exit status.
                "removed": [c["cell_id"] for c in report["removed"]["cells"]],
                "json": str(json_path),
                "markdown": str(md_path),
            }
        )
    )
    if not complete:
        return 2
    if args.require_audit and unaudited:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
