"""Does giving the answerer FEWER rows help the questions whose evidence is buried?

Bucket C of the error partition splits cleanly by where the supporting row sits:

    sub-bucket        n   median rank   rank 1   rank 8-15
    C4 discursive     9       16          0%       78%
    C2 counting       4       11          0%       50%      } buried, 27 questions
    C1 abstained     14        6         21%       43%
    C3 temporal      20        1         55%       20%      } already on top, 46
    C5 wrong specific 26       2         38%       19%

The buried group is not a recall failure — the evidence came back — but it arrives at
position 6 to 16 of about thirty rows. The hypothesis is precision, not recall: past
versions raised the evidence budget "for better coverage" while recall was already
saturated at 0.940, and the extra rows are noise the answerer has to swim through.

THE CONTROL MATTERS MORE THAN THE TREATMENT. Testing only the buried questions would
regress to the mean by construction, so three groups run at every budget:

    buried   the 27 the hypothesis is about
    shallow  the 46 whose support is already at rank 1-2 — should not change
    correct  a sample the system currently gets RIGHT — must not degrade

A budget cut that lifts `buried` while sinking `correct` is a trade, not a fix, and
only the three-group view shows which it is.

    python evidence_budget.py --budgets 15,8,5
"""
import argparse
import asyncio
import collections
import json
import os
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RESULTS = ("/home/lj/code/MIRIX/evals/results/locomo/"
           "v724_full_qaonly_v723graph_r1/metrics.json")
PREFIX = "v723full-r1__"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())
ABST = re.compile(r"no (specific |explicit )?(information|mention|record|detail)|"
                  r"not (explicitly |specifically )?(mentioned|stated|specified|provided)|"
                  r"there is no|cannot (be )?determine|unable to", re.I)
HOWMANY = re.compile(r"^how (many|much|long|often)", re.I)


def content(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 3 or x.isdigit()) and x not in STOP}


def build(args):
    from mirix_memory_system import MirixMemorySystem
    from task_agent import TaskAgent
    judged = json.load(open(RESULTS))["llm_judge_results"]
    wrong = [r for r in judged if r.get("score") == 0]
    right = [r for r in judged if r.get("score") == 1]
    control = right[:: max(1, len(right) // args.controls)][: args.controls]
    convs = sorted({r["sample_id"] for r in wrong + control})
    clients = {c: MirixMemorySystem(
        user_id=PREFIX + c,
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")
        for c in convs}
    agent = TaskAgent(
        mirix_config_path="/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml")
    return clients, agent, wrong, control


async def classify(clients, wrong):
    """Split the wrong answers into buried / shallow by support rank."""
    sem = asyncio.Semaphore(6)
    buried, shallow = [], []

    async def one(r):
        gold = content(r["expected_answer"])
        if not gold:
            return
        async with sem:
            res = await clients[r["sample_id"]].client.search(
                user_id=PREFIX + r["sample_id"], query=r["question"],
                memory_type="all", search_method="embedding", limit=15)
        rows = [x for x in (res.get("results") or []) if isinstance(x, dict)]
        rank = None
        for i, x in enumerate(rows, 1):
            if len(gold & content(f"{x.get('summary') or ''} {x.get('details') or ''} "
                                  f"{x.get('name') or ''}")) / len(gold) >= 0.6:
                rank = i
                break
        if rank is None:
            return
        (buried if rank >= 4 else shallow).append(r)

    await asyncio.gather(*[one(r) for r in wrong])
    return buried, shallow


def score_group(clients, agent, group, judge_fn):
    """Synchronous on purpose.

    wrap_user_prompt and TaskAgent.answer each call asyncio.run internally, which
    creates AND CLOSES an event loop. Driving them from asyncio.to_thread leaves the
    client's httpx connections bound to a loop that has since closed, and the run dies
    with "Event loop is closed" partway through. Sequential is slower and correct;
    the arms are compared on the same questions so the wall clock is the only cost.
    """
    correct = 0
    for idx, r in enumerate(group, 1):
        msgs = clients[r["sample_id"]].wrap_user_prompt(r["question"])
        trace = agent.answer(msgs, PREFIX + r["sample_id"])
        if judge_fn(r, str(trace.get("answer") or "")):
            correct += 1
        if idx % 20 == 0:
            print(f"      .. {idx}/{len(group)}", flush=True)
    return correct, len(group)


def lenient_judge(r, pred):
    """Token-containment stand-in for the LLM judge.

    Used ONLY to compare budgets against each other on the same questions, never as an
    accuracy figure. It is biased low on abstractive gold, identically across arms, so
    differences between arms remain interpretable while absolute values do not.
    """
    gold = content(r["expected_answer"])
    return bool(gold) and len(gold & content(pred)) / len(gold) >= 0.6


def main(clients, agent, buried, shallow, control, budgets):
    print(f"buried (support rank >= 4): {len(buried)}   "
          f"shallow (rank 1-3): {len(shallow)}   control (currently correct): {len(control)}",
          flush=True)
    groups = {"buried": buried, "shallow": shallow, "control": control}
    print(f"\n{'budget':>7s}" + "".join(f"{g:>20s}" for g in groups))
    for b in budgets:
        os.environ["MIRIX_SEARCH_LIMIT"] = str(b)
        line = f"{b:>7d}"
        for g, rows in groups.items():
            c, t = score_group(clients, agent, rows, lenient_judge)
            line += f"{c:>10d}/{t:<9d}"
        print(line, flush=True)
    print("\n  Lenient token-containment judge: comparable ACROSS arms, not an accuracy.")
    print("  A cut that lifts `buried` while sinking `control` is a trade, not a fix.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="15,8,5")
    ap.add_argument("--controls", type=int, default=50)
    a = ap.parse_args()
    # Phase 1: classify with the async client, in its own loop.
    _c, _ag, _w, _ctl = build(a)
    _buried, _shallow = asyncio.run(classify(_c, _w))
    # Phase 2: rebuild the clients so their connections belong to a live loop, then
    # score synchronously.
    _c2, _ag2, _, _ = build(a)
    main(_c2, _ag2, _buried, _shallow, _ctl, [int(x) for x in a.budgets.split(",")])
