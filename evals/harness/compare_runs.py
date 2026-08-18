"""Side-by-side comparison of eval runs, overall AND per category.

organize_results.py already computes accuracy_by_category; nothing ever printed it,
so every experiment was judged on a single noisy total. A change that fixes one
question type is visible here even when the total moves less than the noise floor.

    python compare_runs.py locomo lc26_arm4_graph_rr v712
    python compare_runs.py longmem lm60_ctrl lm60_v2fix
    python compare_runs.py locomo --repeats v712_qa1 v712_qa2 v712_qa3

--repeats treats the named runs as repeats of ONE configuration and reports
mean +/- sd instead of a comparison, which is how you find the noise floor.
"""
import argparse
import json
import math
import sys
from pathlib import Path

EVALS = Path("/home/lj/code/MIRIX/evals/results")

# LoCoMo categories are bare integers in the dataset.
LOCOMO_CATS = {
    "1": "1 multi-hop", "2": "2 temporal", "3": "3 open-domain",
    "4": "4 single-hop", "5": "5 adversarial",
}


def load(bench: str, name: str) -> dict:
    p = EVALS / bench / name / "metrics.json"
    if not p.exists():
        sys.exit(f"missing: {p}")
    with p.open() as fh:
        return json.load(fh)


def rows(m: dict) -> dict:
    out = {}
    for k, v in (m.get("accuracy_by_category") or {}).items():
        out[LOCOMO_CATS.get(str(k), str(k))] = (
            float(v.get("total_correct") or 0), int(v.get("total_judged") or 0))
    mm = m.get("metrics", {})
    out["TOTAL"] = (float(mm.get("total_correct") or 0), int(mm.get("total_judged") or 0))
    return out


def fmt(correct: float, judged: int) -> str:
    if not judged:
        return "    -    "
    return f"{correct:5.1f}/{judged:<3d} {correct/judged:.3f}"


def compare(bench: str, names: list) -> None:
    data = {n: rows(load(bench, n)) for n in names}
    cats = []
    for n in names:
        for c in data[n]:
            if c not in cats:
                cats.append(c)
    cats = [c for c in cats if c != "TOTAL"] + ["TOTAL"]

    w = max(len(c) for c in cats) + 2
    print(f"\n{'category':<{w}}" + "".join(f"{n[:17]:>19s}" for n in names)
          + ("      delta" if len(names) == 2 else ""))
    print("-" * (w + 19 * len(names) + (11 if len(names) == 2 else 0)))
    for c in cats:
        line = f"{c:<{w}}"
        for n in names:
            line += f"{fmt(*data[n].get(c, (0, 0))):>19s}"
        if len(names) == 2:
            a, b = data[names[0]].get(c, (0, 0)), data[names[1]].get(c, (0, 0))
            if a[1] and b[1]:
                d = b[0] - a[0]
                # questions, not percentage points: on 152 questions the repeat noise
                # is about +/-3 questions, so a raw count is the honest unit.
                line += f"   {d:+5.1f} q"
            else:
                line += "        -"
        if c == "TOTAL":
            print("-" * (w + 19 * len(names) + (11 if len(names) == 2 else 0)))
        print(line)
    print()


def repeats(bench: str, names: list) -> None:
    data = [rows(load(bench, n)) for n in names]
    cats = []
    for d in data:
        for c in d:
            if c not in cats:
                cats.append(c)
    cats = [c for c in cats if c != "TOTAL"] + ["TOTAL"]

    w = max(len(c) for c in cats) + 2
    print(f"\n{len(names)} repeats of the same configuration: {', '.join(names)}")
    print(f"\n{'category':<{w}}{'mean':>10s}{'sd':>8s}{'min':>7s}{'max':>7s}"
          f"{'n':>5s}   detectable difference")
    print("-" * (w + 37 + 24))
    for c in cats:
        vals = [d[c][0] for d in data if c in d and d[c][1]]
        judged = next((d[c][1] for d in data if c in d and d[c][1]), 0)
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        # 95% two-sample threshold: 1.96 * sd * sqrt(2/n)
        det = 1.96 * sd * math.sqrt(2 / len(vals)) if sd else 0.0
        if c == "TOTAL":
            print("-" * (w + 37 + 24))
        print(f"{c:<{w}}{mean:10.2f}{sd:8.2f}{min(vals):7.0f}{max(vals):7.0f}"
              f"{judged:5d}   > {det:.1f} q ({det/judged*100:.1f} pts)" if judged else "")
    print("\nA change smaller than the 'detectable difference' cannot be distinguished")
    print("from run-to-run noise at this number of repeats.")
    print("CAUTION: a per-category sd of 0.00 at n=3 is often coincidence, not stability.")
    print("Measured case: LoCoMo cat-1 totalled 29 in all three v7.12 repeats while four")
    print("borderline questions flipped differently in each -- identical totals, different")
    print("composition. Check which QUESTIONS moved before reading a category as stable.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=("locomo", "longmem"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--repeats", action="store_true",
                    help="treat the runs as repeats of one config; report mean/sd")
    a = ap.parse_args()
    (repeats if a.repeats else compare)(a.bench, a.runs)


if __name__ == "__main__":
    main()
