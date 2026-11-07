"""Where does the supporting row RANK, for the errors that already retrieved it?

Bucket C from error_partition.py — 90 of 169 wrong answers — is "the gold-supporting
text was in the returned rows". That is usually read as an answering failure, but it
does not distinguish two very different situations:

  gold row at rank 1-3, still answered wrong   -> the model had it and did not use it
  gold row at rank 12 among 14 distractors     -> retrieval returned it and buried it

The second is a RANKING failure, which is retrieval work, and it is invisible to any
recall metric because recall only asks about membership. This splits C on that line so
the two get different treatment.

Reported per position, not just as a mean, because a bimodal distribution (top-ranked
or buried, nothing between) means something different from a flat one.

    python rank_audit.py
"""
import asyncio
import collections
import json
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

# Which run to analyse, and the user_id prefix its store was written under. Defaults
# are the contaminated pre-36d8d40 run so old invocations keep reproducing; pass --run
# and --prefix for the decontaminated ones.
DEFAULT_RUN = "v724_full_qaonly_v723graph_r1"
DEFAULT_PREFIX = "v723full-r1__"
RESULTS = f"/home/lj/code/MIRIX/evals/results/locomo/{DEFAULT_RUN}/metrics.json"
PREFIX = DEFAULT_PREFIX
SUPPORT = 0.6
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


def _apply_overrides(run, prefix):
    global RESULTS, PREFIX
    if run:
        RESULTS = f"/home/lj/code/MIRIX/evals/results/locomo/{run}/metrics.json"
    if prefix:
        PREFIX = prefix


def build():
    from mirix_memory_system import MirixMemorySystem
    wrong = [r for r in json.load(open(RESULTS))["llm_judge_results"]
             if r.get("score") == 0]
    right = [r for r in json.load(open(RESULTS))["llm_judge_results"]
             if r.get("score") == 1]
    convs = sorted({r["sample_id"] for r in wrong + right})
    clients = {c: MirixMemorySystem(
        user_id=PREFIX + c,
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")
        for c in convs}
    return clients, wrong, right


async def main(clients, wrong, right) -> None:
    sem = asyncio.Semaphore(6)

    async def ranks_for(r):
        """1-based positions of the returned rows that support the gold answer."""
        gold = content(r["expected_answer"])
        if not gold:
            return None
        async with sem:
            res = await clients[r["sample_id"]].client.search(
                user_id=PREFIX + r["sample_id"], query=r["question"],
                memory_type="all", search_method="embedding", limit=15)
        rows = [x for x in (res.get("results") or []) if isinstance(x, dict)]
        hits = [i for i, x in enumerate(rows, start=1)
                if len(gold & content(f"{x.get('summary') or ''} "
                                      f"{x.get('details') or ''} "
                                      f"{x.get('name') or ''}")) / len(gold) >= SUPPORT]
        return (min(hits) if hits else None), len(rows)

    wrong_r = await asyncio.gather(*[ranks_for(r) for r in wrong])
    # a control: when the model answers CORRECTLY, where does the support sit?
    sample_right = right[::max(1, len(right) // 200)][:200]
    right_r = await asyncio.gather(*[ranks_for(r) for r in sample_right])

    def dist(pairs, label):
        got = [p[0] for p in pairs if p and p[0] is not None]
        n = len([p for p in pairs if p])
        buckets = collections.Counter()
        for k in got:
            buckets["1" if k == 1 else
                    "2-3" if k <= 3 else
                    "4-7" if k <= 7 else
                    "8-15"] += 1
        print(f"\n{label}: support found in {len(got)}/{n}")
        for key in ("1", "2-3", "4-7", "8-15"):
            v = buckets[key]
            bar = "#" * int(40 * v / max(len(got), 1))
            print(f"    rank {key:>5s}  {v:4d}  ({v/max(len(got),1):5.1%})  {bar}")
        if got:
            got_sorted = sorted(got)
            print(f"    median rank {got_sorted[len(got_sorted)//2]}, "
                  f"top-3 share {sum(1 for k in got if k <= 3)/len(got):.1%}")
        return buckets, len(got)

    wb, wn = dist(wrong_r, "WRONG answers")
    rb, rn = dist(right_r, f"CORRECT answers (sample of {len(sample_right)})")

    top3_wrong = (wb["1"] + wb["2-3"]) / max(wn, 1)
    top3_right = (rb["1"] + rb["2-3"]) / max(rn, 1)
    print(f"\n=== is C a ranking failure? ===")
    print(f"  support in top-3 when WRONG   {top3_wrong:.1%}")
    print(f"  support in top-3 when CORRECT {top3_right:.1%}")
    gap = top3_right - top3_wrong
    print(f"  gap {gap:+.1%}")
    if gap > 0.15:
        print("  -> ranking matters: wrong answers get their evidence lower down.")
        print("     A re-ranking change is retrieval work and can move bucket C.")
    else:
        print("  -> ranking does NOT separate them: the evidence sits in the same")
        print("     place whether the answer is right or wrong, so bucket C is not")
        print("     a ranking problem and re-ranking cannot move it.")


import argparse as _argp
_parser = _argp.ArgumentParser()
_parser.add_argument("--run", default="")
_parser.add_argument("--prefix", default="")
_args = _parser.parse_args()
_apply_overrides(_args.run, _args.prefix)

_c, _w, _r = build()
asyncio.run(main(_c, _w, _r))
