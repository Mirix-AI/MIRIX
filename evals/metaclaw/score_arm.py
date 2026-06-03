#!/usr/bin/env python3
"""Score a single MetaClaw arm run directory.

Walks every rN/infer_result.json under bench_output and computes the two paper
metrics plus diagnostics:
  - Acc   : mean(inline_score.passed) over ALL questions      (paper "Acc.")
  - Compl : mean(inline_score.passed) over file_check only     (paper "Compl.")
  - MC    : mean(inline_score.passed) over multi_choice only   (our diagnostic)
Also dumps the file_check exit_code histogram so we can prove the python shim
held for the whole run (must contain NO 127).
"""
import json
import sys
from collections import Counter
from pathlib import Path


def score(run_dir: Path) -> dict:
    bo = run_dir / "bench_output"
    runs = sorted([p for p in bo.iterdir() if p.is_dir()]) if bo.is_dir() else []
    base = runs[0] if runs else bo
    results = list(base.rglob("infer_result.json"))

    all_passed = []
    fc_passed = []          # file_check
    mc_passed = []          # multi_choice
    fc_exit = Counter()
    status_ctr = Counter()
    n_err = 0

    for f in results:
        try:
            d = json.loads(f.read_text())
        except Exception:
            n_err += 1
            continue
        status_ctr[d.get("status", "?")] += 1
        qt = d.get("question_type", "?")
        ins = d.get("inline_score") or {}
        passed = bool(ins.get("passed", False))
        all_passed.append(passed)
        if qt == "file_check":
            fc_passed.append(passed)
            ec = ins.get("exit_code")
            if ec is not None:
                fc_exit[ec] += 1
        elif qt == "multi_choice":
            mc_passed.append(passed)

    def pct(xs):
        return 100.0 * sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_total": len(results),
        "n_parse_err": n_err,
        "acc": pct(all_passed),
        "compl": pct(fc_passed),
        "mc": pct(mc_passed),
        "n_all": len(all_passed),
        "n_fc": len(fc_passed),
        "n_mc": len(mc_passed),
        "fc_exit": dict(sorted(fc_exit.items())),
        "status": dict(status_ctr),
    }


def main():
    if len(sys.argv) < 2:
        print("usage: score_arm.py <run_dir> [label]", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else run_dir.name
    r = score(run_dir)
    print(f"[{label}]")
    print(f"  questions scored : {r['n_total']}  (all={r['n_all']} fc={r['n_fc']} mc={r['n_mc']} parse_err={r['n_parse_err']})")
    print(f"  Acc.   (all)     : {r['acc']:.1f}%   (n={r['n_all']})")
    print(f"  Compl. (file_chk): {r['compl']:.1f}%   (n={r['n_fc']})")
    print(f"  MC-only          : {r['mc']:.1f}%   (n={r['n_mc']})")
    print(f"  file_check exit  : {r['fc_exit']}   <- must have NO 127")
    print(f"  status           : {r['status']}")


if __name__ == "__main__":
    main()
