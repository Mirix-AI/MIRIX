"""Argparse shim over :func:`evals.metaclaw.runner.run_arm` and :func:`run_both`.

Usage:

    python -m evals.metaclaw --arm metaclaw --days 1

For ``--days >= 5`` the CLI prints a wall-clock budget summary and either
prompts (TTY) or refuses (non-TTY without ``--yes``) per the issue-06 contract.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .runner import (
    DEFAULT_MIRIX_BASE_URL,
    RUNS_DIR,
    RunResult,
    estimate_wallclock_seconds,
    run_arm,
    run_both,
)

# Exit codes — module-level constants so tests can assert without magic numbers.
RC_BUDGET_NEEDS_YES = 3  # non-TTY run with --days >= 5 and no --yes
RC_USER_ABORTED = 130  # 128 + SIGINT, mirrors `^C` convention
RC_BOTH_FAILED = 1  # --arm both: both arms failed (slice-5 contract)


def _format_seconds_human(seconds: int) -> str:
    """Render seconds as ``"~XhYm"`` (or ``"~Ym"`` for sub-hour budgets).

    The leading ``~`` is part of the issue-06 spec so users perceive it as
    an estimate, not a contract.
    """
    if seconds < 60:
        return f"~{max(seconds, 0)}s"
    minutes_total = (seconds + 30) // 60  # round to nearest minute
    if minutes_total < 60:
        return f"~{minutes_total}m"
    h, m = divmod(minutes_total, 60)
    return f"~{h}h{m}m"


def _format_budget_summary(
    arm: str,
    days: int,
    retry: int,
    output_dir: Path,
    total_rounds: int,
    min_s: int,
    max_s: int,
) -> str:
    """Render the multi-line budget banner per issue-06 acceptance #2.

    Example::

        Selected: --arm metaclaw --days 5  (56 rounds total, n=3 retry)
        Estimated wall-time: ~56m - ~2h48m
        Output: evals/metaclaw/runs/metaclaw-20260528T120000Z/
    """
    return (
        f"Selected: --arm {arm} --days {days}  "
        f"({total_rounds} rounds total, n={retry} retry)\n"
        f"Estimated wall-time: {_format_seconds_human(min_s)} - {_format_seconds_human(max_s)}\n"
        f"Output: {output_dir}\n"
    )


def _default_output_dir(arm: str) -> Path:
    """Return the directory the runner will mint when ``--output-dir`` is unset.

    Mirrors :mod:`runner`'s default exactly so the summary banner is honest
    about where files will land (issue-06 acceptance #2 requires showing the
    real path).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RUNS_DIR / f"{arm}-{ts}"


def _maybe_confirm_budget(args: argparse.Namespace) -> Optional[int]:
    """Apply the ``--days >= 5`` budget gate.  Returns ``None`` to proceed, or
    an exit code to terminate immediately.

    - ``days < 5``                     -> no gate, no summary, returns None.
    - TTY + interactive prompt         -> blocks on ``[y/N]``; ``n`` -> 130.
    - Non-TTY without ``--yes``        -> rc=3 with structured instructions.
    - ``--yes`` flag set               -> prints summary, returns None.

    Estimates are also stashed on ``args`` for the caller to forward into
    ``run.meta.json`` via ``extra_meta=``.
    """
    total_rounds, min_s, max_s = estimate_wallclock_seconds(args.days, args.retry)
    args._estimate_total_rounds = total_rounds  # type: ignore[attr-defined]
    args._estimate_min_s = min_s  # type: ignore[attr-defined]
    args._estimate_max_s = max_s  # type: ignore[attr-defined]

    if args.days < 5:
        return None

    output_dir = args.output_dir or _default_output_dir(args.arm)
    summary = _format_budget_summary(
        arm=args.arm,
        days=args.days,
        retry=args.retry,
        output_dir=output_dir,
        total_rounds=total_rounds,
        min_s=min_s,
        max_s=max_s,
    )
    print(summary, flush=True)

    if args.yes:
        return None

    # Use sys.stdin.isatty() so tests can monkeypatch a stub stdin.
    if not sys.stdin.isatty():
        print(
            "ERROR: --days >= 5 requires confirmation.\n"
            "  pass --yes for non-interactive runs >= 5 days\n",
            flush=True,
        )
        return RC_BUDGET_NEEDS_YES

    try:
        resp = input("Continue? [y/N] ")
    except EOFError:
        resp = ""
    if resp.strip().lower() not in ("y", "yes"):
        print("[cli] aborted by user", flush=True)
        return RC_USER_ABORTED

    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m evals.metaclaw",
        description="Run the MetaClaw 30-day benchmark against the vendored harness.",
    )
    p.add_argument(
        "--arm",
        required=True,
        choices=("metaclaw", "mirix", "mirix-records", "mirix-generic", "no-skills", "native", "both"),
        help=(
            "Skill backend under test. 'metaclaw'/'native' = vendored skills_dir; "
            "'mirix' = MIRIX old harness (raw-transcript every-10-turn evolve, "
            "the regression baseline); 'mirix-records' = MIRIX NEW harness (C5: "
            "per-round distill + records evolution every 5 rounds); "
            "'mirix-generic' = MIRIX generic production memory path "
            "(/memory/add_sync + /memory/auto_dream every 5 turns); "
            "'no-skills' = floor (skills disabled); "
            "'both' = run metaclaw then mirix against a SHARED dataset slice and "
            "write a combined reports.md at the parent dir. Load-bearing delta = "
            "mirix-records - mirix."
        ),
    )
    p.add_argument(
        "--days",
        type=int,
        default=0,
        help="Number of dataset days to include (0 = full 30).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Cap rounds per test (reserved for future slices; passes through).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory (default: evals/metaclaw/runs/<arm>-<ts>/).",
    )
    p.add_argument(
        "--retry",
        type=int,
        default=3,
        help="Per-question retry count forwarded to metaclaw-bench (-n).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the interactive confirmation prompt on long runs "
            "(required for non-interactive runs with --days >= 5)."
        ),
    )
    p.add_argument(
        "--mirix-url",
        default=DEFAULT_MIRIX_BASE_URL,
        help=(
            "Base URL of the MIRIX REST server (only used when --arm mirix "
            f"or --arm both). Default: {DEFAULT_MIRIX_BASE_URL}."
        ),
    )
    args = p.parse_args(argv)

    if args.days < 0:
        p.error(f"--days must be >= 0, got {args.days}")

    gate_rc = _maybe_confirm_budget(args)
    if gate_rc is not None:
        return gate_rc

    extra_meta = {
        "estimated_rounds": getattr(args, "_estimate_total_rounds", 0),
        "estimated_wallclock_seconds_min": getattr(args, "_estimate_min_s", 0),
        "estimated_wallclock_seconds_max": getattr(args, "_estimate_max_s", 0),
    }

    if args.arm == "both":
        metaclaw_res, mirix_res = run_both(
            days=args.days,
            out_dir=args.output_dir,
            max_rounds=args.max_rounds,
            retry=args.retry,
            mirix_url=args.mirix_url,
            extra_meta=extra_meta,
        )
        # Reports.md is always written.  Exit non-zero only when BOTH arms
        # failed — partial success still surfaces the working arm's data.
        if metaclaw_res.exit_code != 0 and mirix_res.exit_code != 0:
            return RC_BOTH_FAILED
        return 0

    result: RunResult = run_arm(
        arm=args.arm,
        days=args.days,
        out_dir=args.output_dir,
        max_rounds=args.max_rounds,
        retry=args.retry,
        mirix_url=args.mirix_url,
        extra_meta=extra_meta,
    )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
