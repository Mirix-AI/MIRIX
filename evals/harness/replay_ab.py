"""The gold was in the context 72% of the time. Can a different instruction find it?

Measured independently: of 166 non-adversarial wrong answers, 70 had EVERY content token of
the gold somewhere in the message list actually sent to the answerer, and another 50 had at
least half. So "the sentence was never written" — my framing yesterday — is wrong on this
store. The sentence is there. What is missing is the predicate.

    "What is Evan's favorite food?"  gold "Ginger snaps"
    "ginger snaps" appears SIX times in the context that was sent. Every appearance frames
    it as a restriction: "despite liking ginger snaps", "limiting himself to two ginger
    snaps a day". Lasagna appears as the headline of the memory. The answerer said lasagna.

That is a reading failure with a specific shape — a fact asserted inside a subordinate or
negated clause loses to a fact asserted as a headline — and it is testable without touching
ingest at all.

This REPLAYS the stored context byte-for-byte. No retrieval runs, so retrieval cannot
differ between arms; the only thing that changes is the final instruction. That is what
makes it a clean test of the answerer, unlike the two global answerer swaps already
measured (gpt-5-mini -11, HyperMem-style CoT -4), both of which also changed how many
searches were issued.

    target   wrong answers whose gold is >=50% present in their own stored context
    control  currently-correct questions, replayed the same way. Sized against the
             lesson that cost the gpt-5-mini experiment: a 60-question control could not
             see a 5.4% regression rate, and 5.4% of 1361 is 73 questions.

    python replay_ab.py --controls 250
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RUN = "/home/lj/code/MIRIX/evals/results/locomo/clean_v724_qaonly"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did his her their about they them".split())

ARMS = {
    "control": "",
    "clause": (
        "\nA fact can be asserted inside a subordinate or negated clause — "
        "\"despite liking X\", \"limiting himself to two X a day\", \"used to hate X\". "
        "Such a clause still asserts that X is true of the person. Do not prefer a fact "
        "merely because it is the headline or title of a memory; weigh what each memory "
        "says, not how prominently it says it."
    ),
    "listall": (
        "\nIf the question asks WHICH or WHAT things (books, cities, items, activities, "
        "sports, dishes), the answer is EVERY one the memories support, not the clearest "
        "one. Give them as a comma-separated list. A single item is the correct answer "
        "only when the memories support exactly one — otherwise a one-item answer is "
        "wrong even if that item is right."
    ),
    "quote": (
        "\nA fact can be asserted inside a subordinate or negated clause — "
        "\"despite liking X\", \"limiting himself to two X a day\". Such a clause still "
        "asserts that X is true of the person. Do not prefer a fact merely because it is "
        "the headline of a memory.\n"
        "Before answering, quote the exact span you are relying on, on one line beginning "
        "EVIDENCE:. Then give the answer on a line beginning FINAL ANSWER:. Only the "
        "FINAL ANSWER line is graded."
    ),
}


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


def load():
    """Stored records, keyed so the judged score can be joined on."""
    judged = {(r["sample_id"], r["question"]): r
              for r in json.load(open(RUN + "/metrics.json"))["llm_judge_results"]
              if str(r.get("category")) != "5"}
    target, control = [], []
    for f in glob.glob(RUN + "/conv-*.json"):
        for r in json.load(open(f)).get("records", {}).values():
            k = (r["sample_id"], r["question"])
            j = judged.get(k)
            if not j or not r.get("messages"):
                continue
            gold = content(r["expected_answer"])
            if not gold:
                continue
            ctx = " ".join(str(m.get("content") or "") for m in r["messages"])
            cov = len(gold & content(ctx)) / len(gold)
            if j.get("score") == 0 and cov >= 0.5:
                target.append(r)
            elif j.get("score") == 1:
                control.append(r)
    return target, control


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=int, default=250)
    ap.add_argument("--model", default="gpt-4.1-mini")
    a = ap.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv
    from llm_judge import evaluate_llm_judge
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    target, control_all = load()
    random.seed(11)
    control = random.sample(control_all, min(a.controls, len(control_all)))
    print(f"target {len(target)} (wrong, gold >=50% present in their own context), "
          f"control {len(control)} of {len(control_all)}", flush=True)

    _FINAL = re.compile(r"final answer\s*:\s*(.*)", re.I | re.S)

    def replay(rec, extra):
        # Everything the answerer saw, minus its own final answer, plus one instruction.
        msgs = [m for m in rec["messages"] if m.get("role") != "assistant"
                or m.get("tool_calls")]
        msgs = [dict(m) for m in rec["messages"]]
        while msgs and msgs[-1].get("role") == "assistant" and not msgs[-1].get("tool_calls"):
            msgs.pop()
        if extra:
            msgs.append({"role": "system", "content": extra})
        try:
            r = client.chat.completions.create(
                model=a.model, messages=msgs, temperature=0, seed=42,
                max_completion_tokens=600 if extra else 128)
        except Exception as e:  # noqa: BLE001
            return ""
        out = (r.choices[0].message.content or "").strip()
        m = _FINAL.search(out)
        return m.group(1).strip() if m and m.group(1).strip() else out

    def score(rows, extra, tag):
        ok = 0
        for i, rec in enumerate(rows, 1):
            ans = replay(rec, extra)
            ok += int(evaluate_llm_judge(
                rec["question"], rec["expected_answer"], ans) or 0)
            if i % 100 == 0:
                print(f"      {tag} {i}/{len(rows)}", flush=True)
        return ok

    print(f"\n{'arm':>9s}{'target -> right':>18s}{'control kept':>16s}")
    for name, extra in ARMS.items():
        t = score(target, extra, f"{name}/t")
        c = score(control, extra, f"{name}/c")
        print(f"{name:>9s}{t:>11d}/{len(target):<6d}{c:>11d}/{len(control):<4d}", flush=True)

    print("\n  control arm is NOT zero: replaying an unchanged context still flips some")
    print("  answers, and the judge relabels ~9% of what it called wrong. Beat control,")
    print("  not zero.")


if __name__ == "__main__":
    main()
