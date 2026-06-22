"""H2 — Agent-execution health gate for a MetaClaw eval run.

The FIX7 sanity gate in ``run_v1_eval.sh`` checks the distill/evolve/records
*pipeline* but NOT whether the agent actually produced answers. A network-
degenerate run (the day02->day03 boundary OpenRouter drop) collapses day04+ to
~0% with every round erroring on a ``GatewayClientRequestError`` /
``EmbeddedAttemptSessionTakeoverError`` / connect failure — yet still FALSE-
PASSES the pipeline checks. Its 0% delta would then be trusted.

This module scans the bench ``infer_result.json`` files and FAILS (LOUD, non-
zero exit) when agent execution collapsed. It is invoked by ``sanity_gate`` in
run_v1_eval.sh as an additional, independent assertion.

Two independent signatures, either of which invalidates the run:

  1. **Aggregate error-rate** — fraction of rounds whose ``status`` is not
     ``success`` OR whose ``error`` text matches a transport/gateway-failure
     marker. A momentary blip is tolerated; a storm is not.

  2. **Consecutive-day pass-rate collapse** — a run of ``>= N`` consecutive
     days whose per-day pass-rate is ~0. This is the precise day04+ signature:
     the agent ran but every answer is wrong/empty because the gateway was
     wedged. Caught even when rounds don't carry an explicit ``error`` field.

All thresholds are module-level constants with rationale comments so they are
easy to audit and tune.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Thresholds (audit me) — deliberately conservative: tolerate isolated blips,  #
# trip hard on a sustained collapse.                                           #
# --------------------------------------------------------------------------- #

# Max tolerated fraction of rounds that errored before the WHOLE run is invalid.
# A handful of stray gateway hiccups across a 15-day run is normal; >15% means a
# systemic transport failure, not noise. (Observed degenerate run: ~80%+.)
MAX_ERROR_RATE = 0.15

# A day is considered "collapsed" if its pass-rate is at or below this. We use a
# small positive epsilon (not exactly 0) so a day that happens to land one lucky
# round doesn't mask an otherwise-dead day. The degenerate signature is whole
# days at literally 0.
DAY_COLLAPSE_PASS_RATE = 0.05

# Number of CONSECUTIVE collapsed days that invalidates the run. The real
# incident collapsed day04..dayNN (>= a dozen) — but as few as 3 consecutive
# dead days is already a clear systemic failure, never normal variance.
MAX_CONSECUTIVE_COLLAPSED_DAYS = 3

# Substrings (case-insensitive) in a round's ``error`` text that mark the
# transport/gateway failure mode this gate exists to catch. ``status != success``
# already counts a round as errored; these additionally flag a *successful*-
# status round whose error text leaked the failure (defensive).
_TRANSPORT_ERROR_MARKERS = (
    "connecterror",
    "all connection attempts failed",
    "gatewayclientrequesterror",
    "embeddedattemptsessiontakeovererror",
    "session takeover",
    "upstream_unreachable",
    "upstream llm endpoint unreachable",
    "connection refused",
    "connect error",
    "connect failed",
)


@dataclass
class DayStats:
    day: str
    total: int = 0
    passed: int = 0
    errored: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def collapsed(self) -> bool:
        # A day with rounds but ~0 pass-rate is collapsed. An empty day is not
        # counted (no rounds → nothing to assert).
        return self.total > 0 and self.pass_rate <= DAY_COLLAPSE_PASS_RATE


@dataclass
class AgentHealthReport:
    ok: bool
    total_rounds: int
    errored_rounds: int
    error_rate: float
    days: list[DayStats] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rounds={self.total_rounds} errored={self.errored_rounds} "
            f"error_rate={self.error_rate:.1%}",
        ]
        for d in self.days:
            flag = " COLLAPSED" if d.collapsed else ""
            lines.append(
                f"  {d.day}: pass={d.passed}/{d.total} "
                f"({d.pass_rate:.0%}) errored={d.errored}{flag}"
            )
        return "\n".join(lines)


def _round_errored(result: dict) -> bool:
    """True if this round failed (non-success status or transport-error text)."""
    if str(result.get("status", "")).lower() != "success":
        return True
    err = result.get("error")
    if err:
        low = str(err).lower()
        if any(m in low for m in _TRANSPORT_ERROR_MARKERS):
            return True
    return False


def _round_passed(result: dict) -> bool:
    score = result.get("inline_score")
    if isinstance(score, dict):
        return bool(score.get("passed"))
    return False


def _day_number(day: str) -> int | None:
    """Extract the numeric index from a ``dayNN`` label, else None."""
    if day.lower().startswith("day") and day[3:].isdigit():
        return int(day[3:])
    return None


def _day_sort_key(day: str) -> tuple:
    # "day04" -> (4,) so days order numerically, not lexically; fall back to the
    # raw string for anything that isn't dayNN.
    if day.lower().startswith("day"):
        tail = day[3:]
        if tail.isdigit():
            return (0, int(tail))
    return (1, day)


def evaluate_agent_health(run_dir: Path | str) -> AgentHealthReport:
    """Scan ``<run_dir>/bench_output/**/infer_result.json`` and grade health."""
    run_dir = Path(run_dir)
    search_root = run_dir / "bench_output"
    if not search_root.exists():
        # The bench_output dir may BE the run_dir in some layouts.
        search_root = run_dir

    day_stats: dict[str, DayStats] = {}
    total = errored = 0

    for result_path in search_root.rglob("infer_result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            # A truncated/garbage result is itself a failed round.
            result = {"status": "error", "error": "unparseable infer_result.json"}
            # Best-effort day from the path: .../dayNN/<group>/<round>/infer_result.json
            day = _day_from_path(result_path)
            result["test_id"] = day
        day = str(result.get("test_id") or _day_from_path(result_path) or "unknown")
        ds = day_stats.setdefault(day, DayStats(day=day))
        ds.total += 1
        total += 1
        if _round_errored(result):
            ds.errored += 1
            errored += 1
        if _round_passed(result):
            ds.passed += 1

    days = [day_stats[d] for d in sorted(day_stats, key=_day_sort_key)]
    error_rate = errored / total if total else 0.0

    failures: list[str] = []

    if total == 0:
        failures.append(
            "no rounds found — bench_output had no infer_result.json files "
            "(the bench produced nothing; run is invalid)"
        )

    if total > 0 and error_rate > MAX_ERROR_RATE:
        failures.append(
            f"agent error rate {error_rate:.1%} exceeds max {MAX_ERROR_RATE:.0%} "
            f"({errored}/{total} rounds errored — gateway/connect storm signature)"
        )

    # Longest run of *truly consecutive* collapsed days. A run is broken by a
    # non-collapsed day OR by a numeric gap between adjacent observed days
    # (codex P1): collapsed day01, day03, day05 — with day02/day04 either healthy
    # or entirely absent (no rounds) — must NOT count as 3-in-a-row, since the
    # day04+ incident was a contiguous block of dead days.
    run_len = 0
    worst_run = 0
    worst_span: list[str] = []
    cur_span: list[str] = []
    prev_num: int | None = None
    for ds in days:
        num = _day_number(ds.day)
        # A gap in the numeric day sequence breaks any in-progress run: the
        # missing day(s) had no collapse evidence, so we cannot treat the
        # collapsed days on either side as contiguous.
        gapped = (
            prev_num is not None and num is not None and num != prev_num + 1
        )
        if ds.collapsed and not gapped:
            run_len += 1
            cur_span.append(ds.day)
        elif ds.collapsed and gapped:
            # Start a fresh run AT this collapsed day (the gap severed the prior).
            run_len = 1
            cur_span = [ds.day]
        else:
            run_len = 0
            cur_span = []
        if run_len > worst_run:
            worst_run = run_len
            worst_span = list(cur_span)
        prev_num = num
    if worst_run >= MAX_CONSECUTIVE_COLLAPSED_DAYS:
        failures.append(
            f"{worst_run} consecutive days collapsed to ~0 pass-rate "
            f"({worst_span[0]}..{worst_span[-1]}) — agent-execution collapse "
            f"signature (network-degenerate run)"
        )

    return AgentHealthReport(
        ok=not failures,
        total_rounds=total,
        errored_rounds=errored,
        error_rate=error_rate,
        days=days,
        failures=failures,
    )


def _day_from_path(result_path: Path) -> str | None:
    for part in result_path.parts:
        if part.lower().startswith("day") and part[3:].isdigit():
            return part
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m evals.metaclaw.agent_health_gate <run_dir>``.

    Exit 0 if agent execution was healthy, 1 (LOUD) otherwise.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: agent_health_gate <run_dir>", file=sys.stderr)
        return 2
    run_dir = argv[0]
    report = evaluate_agent_health(run_dir)
    print(report.summary())
    if report.ok:
        print(f"  [agent-health] ok: error_rate={report.error_rate:.1%}, no day collapse")
        return 0
    for f in report.failures:
        print(f"  [agent-health] FAIL: {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
