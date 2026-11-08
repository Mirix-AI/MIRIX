"""Can the answerer be made to pick the RIGHT fact when several similar ones are in view?

83 of 169 errors on ab_graph_r1 are "everything was present and the answer is still wrong",
and reading them shows one shape almost every time. The answer is not vague and it does not
hedge — it is a different, true, confidently stated fact:

    languages besides German   gold Spanish        answered French
    favourite movie genres     gold action, sci-fi answered fantasy, sci-fi
    what donations bought      gold a fire truck   answered canned food
    event in June 2023         gold live music     answered arts and crafts

The context holds thirty-odd rows about the same person, several of them adjacent to the
question, and the wrong neighbour gets picked. That is a discrimination failure, not a
knowledge or formatting one — which is why four earlier arms, all of them rewordings of the
format and completeness rules, moved the target set and the control set by the same amount
and netted nothing.

So these arms change the MECHANISM rather than the wording:

    discriminate  name the failure and force a comparison against the question's exact terms
    cite          the answer must carry the verbatim memory line it came from
    twostage      one call selects the relevant lines, a second answers from only those
    selfconsist   answer three times and take the majority

Replay only: the stored context is re-sent unchanged, so retrieval variance is excluded and
only the answering step differs. Every arm is scored on the same target set AND on a control
sample of questions that are currently right, because an instruction that lifts the target
by breaking the control is worth nothing — the four earlier arms all failed exactly there.

    python replay_disc.py --controls 250
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RUN = "/home/lj/code/MIRIX/evals/results/locomo/ab_graph_r1"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did his her their about they them".split())

DISCRIMINATE = (
    "\nThe memories below contain SEVERAL facts about this person that are close to the "
    "question but answer a different one. Before answering, re-read the question and note "
    "its exact terms — which person, which category of thing, which date. Then find the "
    "memory that matches ALL of those terms. A memory that matches the person and the "
    "topic but not the specific category or date is the WRONG memory, however confident it "
    "sounds. If two memories both match, prefer the one that uses the question's own words."
)

CITE = (
    "\nAnswer in exactly two lines.\n"
    "EVIDENCE: copy, verbatim, the single sentence from the memories that contains the "
    "answer. Copy it exactly; do not paraphrase. If no sentence contains the answer, write "
    "EVIDENCE: none.\n"
    "FINAL ANSWER: the minimal answer, taken only from the sentence you just copied.\n"
    "Only the FINAL ANSWER line is graded."
)

SELECT = (
    "\nDo not answer the question yet. List the memories that could answer it, most likely "
    "first, at most three, each as one verbatim line copied from the memories above. "
    "Output nothing else."
)

ANSWER_FROM = (
    "\nAnswer the question using ONLY the lines below. If they do not contain the answer, "
    "say so rather than recalling anything else.\n\n{sel}\n\nQuestion: {q}\n"
    "Answer with the minimal direct answer and nothing else."
)


def content(text) -> set:
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {w for w in words if (len(w) > 3 or w.isdigit()) and w not in STOP}


def load():
    judged = {(r["sample_id"], r["question"]): r
              for r in json.load(open(RUN + "/metrics.json"))["llm_judge_results"]
              if str(r.get("category")) != "5"}
    target, control = [], []
    for f in glob.glob(RUN + "/conv-*.json"):
        if f.endswith("_memories.json"):
            continue
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
    ap.add_argument("--arms", default="")
    a = ap.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv
    from llm_judge import evaluate_llm_judge
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    target, control_all = load()
    random.seed(11)
    control = random.sample(control_all, min(a.controls, len(control_all)))
    print(f"target {len(target)} (wrong, gold >=50% in own context), "
          f"control {len(control)} of {len(control_all)}", flush=True)

    FINAL = re.compile(r"final answer\s*:\s*(.*)", re.I | re.S)

    def base_msgs(rec):
        msgs = [dict(m) for m in rec["messages"]]
        while msgs and msgs[-1].get("role") == "assistant" and not msgs[-1].get("tool_calls"):
            msgs.pop()
        return msgs

    def call(msgs, temp=0.0, cap=600):
        try:
            r = client.chat.completions.create(model=a.model, messages=msgs,
                                               temperature=temp, seed=42,
                                               max_completion_tokens=cap)
            return (r.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def run_arm(rec, arm):
        msgs = base_msgs(rec)
        if arm == "control":
            return call(msgs, cap=128)
        if arm == "discriminate":
            return call(msgs + [{"role": "system", "content": DISCRIMINATE}], cap=200)
        if arm == "cite":
            out = call(msgs + [{"role": "system", "content": CITE}])
            m = FINAL.search(out)
            return (m.group(1).strip() if m else out).split("\n")[0]
        if arm == "twostage":
            sel = call(msgs + [{"role": "system", "content": SELECT}], cap=400)
            if not sel.strip():
                return call(msgs, cap=128)
            q = rec["question"]
            return call([{"role": "user",
                          "content": ANSWER_FROM.format(sel=sel[:4000], q=q)}], cap=128)
        if arm == "verifyretry":
            # Every unconditional instruction so far lifts the target and costs more on the
            # control, because the control set is twelve times larger — cite gained 9 and
            # lost 23, twostage gained 4 and lost 35. The only arm that ever survived that
            # arithmetic, countfirst, worked because it fired on a detectable subset.
            # So: answer normally first, which preserves the control by construction, then
            # spend the discrimination pass ONLY where a check says the answer is not
            # supported. Control damage is bounded by the verifier's false-alarm rate.
            first = call(msgs, cap=128)
            if not first:
                return ""
            chk = call(msgs + [{"role": "system", "content":
                "\nA candidate answer to the question is given below. Is it explicitly "
                "supported by the memories above, matching the question's exact person, "
                "category and date? Reply with one word, SUPPORTED or UNSUPPORTED.\n\n"
                "Candidate: " + first}], cap=10)
            if "UNSUPPORTED" not in chk.upper():
                return first
            second = call(msgs + [{"role": "system", "content": DISCRIMINATE}], cap=200)
            return second or first
        if arm == "selfconsist":
            outs = [call(msgs, temp=0.8, cap=128) for _ in range(3)]
            outs = [o for o in outs if o]
            if not outs:
                return ""
            # majority by content-word overlap: the answer closest to the other two
            best, bs = outs[0], -1.0
            for o in outs:
                s = sum(len(content(o) & content(p)) / max(len(content(o) | content(p)), 1)
                        for p in outs if p is not o)
                if s > bs:
                    best, bs = o, s
            return best
        raise ValueError(arm)

    ARMS = a.arms.split(",") if a.arms else [
        "control", "discriminate", "cite", "twostage", "selfconsist", "verifyretry"]
    res = collections.defaultdict(lambda: {"t": 0, "c": 0})
    for arm in ARMS:
        for label, rows in (("t", target), ("c", control)):
            ok = 0
            for rec in rows:
                ans = run_arm(rec, arm)
                if not ans:
                    continue
                try:
                    s = evaluate_llm_judge(rec["question"], rec["expected_answer"], ans)
                    ok += int(bool(s))
                except Exception:  # noqa: BLE001
                    pass
            res[arm][label] = ok
        print(f"{arm:>14s} {res[arm]['t']:>4d}/{len(target)}   "
              f"{res[arm]['c']:>4d}/{len(control)}", flush=True)

    print("\n  target = currently wrong with the gold present in context;")
    print("  control = currently right. An arm only counts if target rises and control holds.")


if __name__ == "__main__":
    main()
