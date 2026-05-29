"""Aggregate 4 arm reports into a single four-arm-comparison.md.

PRD D11 + D12. Reads paper-bench-generated report.json files from each arm's
runs/ directory and produces a markdown document with:

  - Per-arm × per-test accuracy table
  - Aggregate mean per arm
  - Pairwise deltas (A−B, A−C, A−D, B−C, B−D, C−D)
  - Canonical known-deviations section (PRD D12)
  - Run metadata (timestamps, model, MIRIX user_ids, bench subset)

Missing arms are handled gracefully — partial table with "—" markers, no
exception. See test fixtures for the expected output shape.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

ARM_TITLES = {
    "A": "armA-mirix-skills",
    "B": "armB-mirix-legacy",
    "C": "armC-baseline",
    "D": "armD-paper-native",
}
ARM_DESCRIPTIONS = {
    "A": "paper proxy + MIRIX-skill-evolve backend (:8531)",
    "B": "paper proxy + MIRIX-legacy backend (:8532)",
    "C": "paper baseline (no proxy, agent → LLM direct)",
    "D": "paper proxy + paper-native SkillManager (anchor)",
}


def _latest_report(run_dir: Path) -> Path | None:
    """Find the latest `report.json` under run_dir/infer/<run-id>/."""
    infer = run_dir / "infer"
    if not infer.exists():
        return None
    candidates = sorted(infer.glob("*/report.json"), key=lambda p: p.parent.name, reverse=True)
    return candidates[0] if candidates else None


def _load_arm_report(run_dir: Path) -> dict | None:
    """Return parsed report.json + run.meta.json merged, or None if missing."""
    rpt_path = _latest_report(run_dir)
    if not rpt_path:
        return None
    try:
        with open(rpt_path, encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    meta_path = run_dir / "run.meta.json"
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"report": report, "meta": meta, "report_path": str(rpt_path)}


def _per_test_accuracy(report: dict) -> dict[str, float]:
    """Extract {test_id: accuracy} from report.by_task."""
    by_task = report.get("by_task", {}) or {}
    out: dict[str, float] = {}
    for tid, td in by_task.items():
        acc = td.get("accuracy")
        if acc is not None:
            out[tid] = float(acc)
    return out


def _per_test_question_count(report: dict) -> dict[str, int]:
    by_task = report.get("by_task", {}) or {}
    return {tid: len(td.get("questions", []) or []) for tid, td in by_task.items()}


def aggregate(
    arm_runs: dict[str, Path | None],
    *,
    phase: str = "gating",
    bench: str = "small",
) -> str:
    """Render a four-arm-comparison.md given each arm's run directory.

    arm_runs: {"A": Path | None, "B": ..., "C": ..., "D": ...}
              None = arm didn't run (will show "—" in the table).
    """
    loaded: dict[str, dict | None] = {}
    for k, p in arm_runs.items():
        loaded[k] = _load_arm_report(p) if p else None

    # Union of test ids across all arms (sorted)
    all_tests: set[str] = set()
    for entry in loaded.values():
        if entry:
            all_tests.update(_per_test_accuracy(entry["report"]).keys())
    test_order = sorted(all_tests)

    lines: list[str] = []
    lines.append(f"# MetaClaw paper-aligned {phase} — Four-arm comparison")
    lines.append("")
    lines.append(f"**Date**: {dt.date.today().isoformat()}")
    lines.append(f"**Bench**: `metaclaw-bench-{bench}` first {len(test_order)} test(s)")
    lines.append(f"**Phase**: {phase}")
    lines.append("**Harness**: paper original (openclaw subprocess + metaclaw-bench infer)")
    model = next(
        (loaded[a]["meta"].get("model") for a in ("A", "B", "C", "D")
         if loaded.get(a) and loaded[a].get("meta", {}).get("model")),
        "(unknown)",
    )
    lines.append(f"**Model**: `{model}` (PRD D10 — paper's published model)")
    lines.append("")

    # Arms run table
    lines.append("## Arms")
    lines.append("")
    lines.append("| Arm | Description | Status | Output |")
    lines.append("|---|---|---|---|")
    for k in ("A", "B", "C", "D"):
        entry = loaded.get(k)
        if entry is None:
            status = "**missing**"
            out_path = "—"
        else:
            status = "ok"
            out_path = f"`{entry['report_path']}`"
        lines.append(f"| {k} ({ARM_TITLES[k]}) | {ARM_DESCRIPTIONS[k]} | {status} | {out_path} |")
    lines.append("")

    # Per-test accuracy table
    lines.append("## Per-test accuracy")
    lines.append("")
    header = "| Test | A | B | C | D |"
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    arm_accs: dict[str, dict[str, float]] = {
        k: (_per_test_accuracy(loaded[k]["report"]) if loaded.get(k) else {})
        for k in ("A", "B", "C", "D")
    }
    arm_counts: dict[str, dict[str, int]] = {
        k: (_per_test_question_count(loaded[k]["report"]) if loaded.get(k) else {})
        for k in ("A", "B", "C", "D")
    }
    for tid in test_order:
        row = [tid]
        for k in ("A", "B", "C", "D"):
            acc = arm_accs[k].get(tid)
            cnt = arm_counts[k].get(tid, 0)
            row.append(f"{acc:.2f} ({int(acc * cnt)}/{cnt})" if acc is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Aggregate row
    agg_row = ["**mean**"]
    arm_means: dict[str, float | None] = {}
    arm_totals: dict[str, tuple[int, int]] = {}  # (correct, total)
    for k in ("A", "B", "C", "D"):
        entry = loaded.get(k)
        if entry is None:
            agg_row.append("—")
            arm_means[k] = None
            arm_totals[k] = (0, 0)
            continue
        summary = entry["report"].get("summary", {})
        total = int(summary.get("total_questions", 0))
        correct = float(summary.get("correct", 0.0))
        mean = float(summary.get("accuracy", 0.0)) if total else 0.0
        arm_means[k] = mean
        arm_totals[k] = (int(round(correct)), total)
        agg_row.append(f"**{mean:.3f}** ({int(round(correct))}/{total})")
    lines.append("| " + " | ".join(agg_row) + " |")
    lines.append("")

    # Pairwise deltas
    lines.append("## Pairwise deltas (mean)")
    lines.append("")
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    lines.append("| Pair | Δ |")
    lines.append("|---|---|")
    for x, y in pairs:
        mx = arm_means.get(x)
        my = arm_means.get(y)
        if mx is None or my is None:
            lines.append(f"| {x} − {y} | — |")
        else:
            lines.append(f"| {x} − {y} | {mx - my:+.3f} |")
    lines.append("")

    # Known deviations (PRD D12)
    lines.append("## Known deviations from paper")
    lines.append("")
    lines.append("1. **Skill backend** (arms A, B only): MIRIX serves the skill backend "
                 "in arms A and B, where paper uses its own "
                 "`metaclaw/skill_manager.SkillManager`. arms C and D are paper original.")
    lines.append("2. **Single seed**: no confidence intervals; all numbers are point estimates.")
    lines.append("3. **Mode coverage**: only `skills_only` and `baseline` modes. paper's "
                 "other modes (memory_run, buffer_memory_run, madmax_memory_run, rl_*, "
                 "proxy_passthrough_run) are out of scope (PRD).")
    if bench == "small":
        lines.append(f"4. **Subset**: paper-small first {len(test_order)} of 12 tests "
                     f"(gating phase per PRD Q5.b option G1). Main run will use full 30 tests.")
    lines.append("")

    # Run metadata
    lines.append("## Run metadata")
    lines.append("")
    lines.append("| Arm | Started | Finished | Wall sec | exit | MIRIX URL | MIRIX user_id |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in ("A", "B", "C", "D"):
        entry = loaded.get(k)
        if entry is None:
            lines.append(f"| {k} | — | — | — | — | — | — |")
            continue
        m = entry.get("meta", {})
        lines.append(
            f"| {k} | {m.get('started_at', '—')} | {m.get('finished_at', '—')} | "
            f"{m.get('wall_seconds', '—')} | {m.get('exit_code', '—')} | "
            f"{m.get('mirix_base_url') or '—'} | {m.get('mirix_user_id') or '—'} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm-a", type=Path, default=None, help="arm A run dir")
    p.add_argument("--arm-b", type=Path, default=None)
    p.add_argument("--arm-c", type=Path, default=None)
    p.add_argument("--arm-d", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="output markdown path")
    p.add_argument("--phase", default="gating", choices=["gating", "30day"])
    p.add_argument("--bench", default="small", choices=["small", "full"])
    args = p.parse_args()

    arm_runs = {"A": args.arm_a, "B": args.arm_b, "C": args.arm_c, "D": args.arm_d}
    md = aggregate(arm_runs, phase=args.phase, bench=args.bench)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"[aggregator] wrote {args.out} ({len(md)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
