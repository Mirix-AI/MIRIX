"""Does separating the working-out from the final answer move bucket C?

The answerer's prompt contains two instructions that cannot both hold:

    "4. Be VERY CONCISE in your response, only output the answer and nothing else."
    "- Then write out EVERY distinct matching instance as an explicit numbered list."

Measured compliance with the second: 7% on "how many" questions, 0% on "what items /
activities" questions, and 0 of the 169 wrong answers contained an enumeration at all.
The concise instruction wins, and the instruction it suppresses is the one aimed at the
largest measured failure modes — set questions answered with a single member, and
undercounted totals.

MIRIX_ANSWER_STYLE=scratch lets both stand: enumerate inside <scratch>...</scratch>,
then state the answer. The block is stripped before anything scores it.

THE JUDGE MUST BE THE REAL ONE. The scratch arm produces longer, list-shaped answers,
so a token-containment proxy would score it higher for purely mechanical reasons — more
tokens, more chances to contain the gold. That would manufacture the result this is
supposed to test. evaluate_llm_judge is the same judge the benchmark uses.

Two groups, because the treatment alone regresses to the mean by construction:
    target   the bucket-C questions the change is aimed at
    control  questions currently answered correctly — these must not break

    python answer_style_ab.py --target 60 --controls 40
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RESULTS = ("/home/lj/code/MIRIX/evals/results/locomo/"
           "clean_v724_qaonly/metrics.json")
PREFIX = "clean-r1__"
CONFIG = "/home/lj/code/MIRIX/evals/configs/0201c_v6.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--controls", type=int, default=40)
    ap.add_argument("--styles", default="default,scratch")
    a = ap.parse_args()

    from mirix_memory_system import MirixMemorySystem
    from task_agent import TaskAgent
    from llm_judge import evaluate_llm_judge

    judged = json.load(open(RESULTS))["llm_judge_results"]
    wrong = [r for r in judged if r.get("score") == 0]
    right = [r for r in judged if r.get("score") == 1]
    random.seed(11)
    target = random.sample(wrong, min(a.target, len(wrong)))
    control = random.sample(right, min(a.controls, len(right)))
    convs = sorted({r["sample_id"] for r in target + control})
    clients = {c: MirixMemorySystem(user_id=PREFIX + c, mirix_config_path=CONFIG)
               for c in convs}
    print(f"target {len(target)} (currently wrong), control {len(control)} "
          f"(currently right)", flush=True)

    def score(agent, rows, tag):
        ok = 0
        for i, r in enumerate(rows, 1):
            ans = str(agent.answer(
                clients[r["sample_id"]].wrap_user_prompt(r["question"]),
                PREFIX + r["sample_id"]).get("answer") or "")
            ok += int(evaluate_llm_judge(r["question"], r["expected_answer"], ans) or 0)
            if i % 20 == 0:
                print(f"      {tag} {i}/{len(rows)}", flush=True)
        return ok

    print(f"\n{'style':>10s}{'target -> now right':>22s}{'control kept':>16s}")
    for style in [s.strip() for s in a.styles.split(",") if s.strip()]:
        os.environ["MIRIX_ANSWER_STYLE"] = "" if style == "default" else style
        import importlib
        import task_agent as _ta
        importlib.reload(_ta)                     # the style is read at prompt build
        agent = _ta.TaskAgent(mirix_config_path=CONFIG)
        t = score(agent, target, f"{style}/target")
        c = score(agent, control, f"{style}/control")
        print(f"{style:>10s}{t:>13d}/{len(target):<8d}{c:>9d}/{len(control):<6d}",
              flush=True)
    print("\n  Judged by evaluate_llm_judge (gpt-4o-mini), the benchmark's own judge.")
    print("  A style that lifts `target` while sinking `control` is a trade, not a fix.")


if __name__ == "__main__":
    main()
