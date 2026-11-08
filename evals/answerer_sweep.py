"""How much of the remaining error is the ANSWERER's capability rather than the memory?

Everything is held fixed except the model that reads the retrieved rows: same store,
same graph, same retrieval, same prompt, same tools. If a stronger answerer converts a
large share of the current failures, then a benchmark score at a fixed weak answerer is
measuring that answerer's ceiling and not the memory system, and every version
comparison run this way inherits that confound.

Three groups, because the treatment alone would be uninterpretable:

    wrong    a sample of what the system currently gets wrong
    control  a sample of what it currently gets RIGHT — a stronger model must not
             break these, and a weaker one tells you how much of "correct" was luck

The judge is token containment: comparable ACROSS arms on the same questions, biased
low on abstractive gold identically in every arm, and never an accuracy figure.

    python answerer_sweep.py --models gpt-4.1-mini,gpt-4.1 --wrong 90 --controls 40
"""
import argparse
import asyncio
import json
import random
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RESULTS = ("/home/lj/code/MIRIX/evals/results/locomo/"
           "v724_full_qaonly_v723graph_r1/metrics.json")
PREFIX = "v723full-r1__"
CONFIG = "/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did what when where who how why which their her "
           "his its she he they them about into over under after before not".split())


def content(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 3 or x.isdigit()) and x not in STOP}


def judge(r, pred):
    gold = content(r["expected_answer"])
    return bool(gold) and len(gold & content(pred)) / len(gold) >= 0.6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-4.1-mini,gpt-4.1")
    ap.add_argument("--wrong", type=int, default=90)
    ap.add_argument("--controls", type=int, default=40)
    a = ap.parse_args()

    from mirix_memory_system import MirixMemorySystem
    from task_agent import TaskAgent

    judged = json.load(open(RESULTS))["llm_judge_results"]
    wrong_all = [r for r in judged if r.get("score") == 0]
    right_all = [r for r in judged if r.get("score") == 1]
    random.seed(7)
    wrong = random.sample(wrong_all, min(a.wrong, len(wrong_all)))
    control = random.sample(right_all, min(a.controls, len(right_all)))
    convs = sorted({r["sample_id"] for r in wrong + control})
    clients = {c: MirixMemorySystem(user_id=PREFIX + c, mirix_config_path=CONFIG)
               for c in convs}
    print(f"wrong sample {len(wrong)} of {len(wrong_all)}, "
          f"control {len(control)} of {len(right_all)}", flush=True)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"\n{'model':>16s}{'wrong -> now right':>22s}{'control kept':>16s}")
    for m in models:
        agent = TaskAgent(mirix_config_path=CONFIG, model=m)
        rescued = sum(
            1 for r in wrong
            if judge(r, str(agent.answer(
                clients[r["sample_id"]].wrap_user_prompt(r["question"]),
                PREFIX + r["sample_id"]).get("answer") or "")))
        kept = sum(
            1 for r in control
            if judge(r, str(agent.answer(
                clients[r["sample_id"]].wrap_user_prompt(r["question"]),
                PREFIX + r["sample_id"]).get("answer") or "")))
        print(f"{m:>16s}{rescued:>13d}/{len(wrong):<8d}{kept:>9d}/{len(control):<6d}",
              flush=True)
    print("\n  Containment judge: comparable across arms, not an accuracy figure.")
    print("  'wrong -> now right' on a sample the CURRENT model got wrong is an upper")
    print("  bound with a regression-to-the-mean component; the control column is what")
    print("  says whether a gain is real or a trade.")


if __name__ == "__main__":
    main()
