"""Does the answerer do better if the evidence is a table instead of prose?

Ten attempts to fix the answering step have now failed, all of them instructions: reword the
format rules, name the failure mode, force a citation, split select-then-answer, vote over
three samples, verify-then-retry. Every one lifted the 110 target questions by 4-9 and cost
more than that on the 1335 that are already right. Verify-then-retry moved the target by
zero — the model calls its own wrong answer supported, so it cannot detect the error either.

What those all have in common is that they tell the model to try harder on the same input.
The one intervention that ever gained anything (+10) reshaped the input instead. And the
failure mode says why that might matter: the wrong answers are confident, specific, and
adjacent — French for Spanish, arts-and-crafts for live music, canned food for a fire truck.
The right row and three plausible neighbours are all in view as undifferentiated prose.

Worth noting that ingest DOES know the difference. The meta agent routes each memory to a
typed store, with an actor and a resolved occurred_at. By the time the answerer sees it, all
of that is flattened into a paragraph. These arms hand some of it back:

    fielded    every memory rendered as WHO | WHEN | WHAT on one line
    grouped    same, but sorted by subject then date, so neighbours sit together
    dated      prose kept as-is, with an explicit resolved date prefix per row

    python replay_format.py --controls 200
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


def content(text) -> set:
    w = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return {x for x in w if (len(x) > 3 or x.isdigit()) and x not in STOP}


def load():
    judged = {(r["sample_id"], r["question"]): r
              for r in json.load(open(RUN + "/metrics.json"))["llm_judge_results"]
              if str(r.get("category")) != "5"}
    target, control = [], []
    for f in glob.glob(RUN + "/conv-*.json"):
        if f.endswith("_memories.json"):
            continue
        for r in json.load(open(f)).get("records", {}).values():
            j = judged.get((r["sample_id"], r["question"]))
            if not j or not r.get("messages"):
                continue
            gold = content(r["expected_answer"])
            if not gold:
                continue
            ctx = " ".join(str(m.get("content") or "") for m in r["messages"])
            if j.get("score") == 0 and len(gold & content(ctx)) / len(gold) >= 0.5:
                target.append(r)
            elif j.get("score") == 1:
                control.append(r)
    return target, control


DATE = re.compile(r"\b(\d{1,2} (?:January|February|March|April|May|June|July|August|"
                  r"September|October|November|December) \d{4}|\d{4}-\d{2}-\d{2})\b")


def rows_from(rec):
    """Pull the individual memory rows out of the tool results the answerer received."""
    out = []
    for m in rec.get("messages") or []:
        if m.get("role") != "tool":
            continue
        try:
            payload = json.loads(m.get("content") or "{}")
        except Exception:  # noqa: BLE001
            continue
        cands = []
        if isinstance(payload, dict):
            for k in ("new_evidence", "results", "memories", "rows"):
                v = payload.get(k)
                if isinstance(v, list):
                    cands.extend(v)
        elif isinstance(payload, list):
            cands = payload
        for c in cands:
            if not isinstance(c, dict):
                continue
            text = " ".join(str(c.get(k) or "") for k in
                            ("summary", "details", "name", "content") if c.get(k))
            if not text.strip():
                continue
            who = str(c.get("actor") or c.get("subject") or "").strip()
            when = str(c.get("timestamp") or c.get("occurred_at")
                       or c.get("mentioned_at") or "").strip()[:10]
            if not when:
                m2 = DATE.search(text)
                when = m2.group(1) if m2 else ""
            out.append((who, when, " ".join(text.split())[:300]))
    return out


def render(rows, mode):
    if mode == "grouped":
        rows = sorted(rows, key=lambda r: (r[0].lower(), r[1]))
    lines = []
    for i, (who, when, text) in enumerate(rows, 1):
        if mode == "dated":
            lines.append(f"[{i}] ({when or 'undated'}) {text}")
        else:
            lines.append(f"[{i}] WHO: {who or '-'} | WHEN: {when or '-'} | WHAT: {text}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=int, default=200)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--arms", default="control,fielded,grouped,dated")
    a = ap.parse_args()

    from openai import OpenAI
    from dotenv import load_dotenv
    from llm_judge import evaluate_llm_judge
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    target, control_all = load()
    random.seed(11)
    control = random.sample(control_all, min(a.controls, len(control_all)))
    print(f"target {len(target)}, control {len(control)} of {len(control_all)}", flush=True)

    def call(msgs, cap=128):
        try:
            r = client.chat.completions.create(model=a.model, messages=msgs,
                                               temperature=0, seed=42,
                                               max_completion_tokens=cap)
            return (r.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def run(rec, arm):
        msgs = [dict(m) for m in rec["messages"]]
        while msgs and msgs[-1].get("role") == "assistant" and not msgs[-1].get("tool_calls"):
            msgs.pop()
        if arm == "control":
            return call(msgs)
        rows = rows_from(rec)
        if len(rows) < 2:
            return call(msgs)          # nothing to reshape; fall back rather than fabricate
        # Rebuild only the tool results, never the conversation. The first version of this
        # experiment collapsed everything into one user message, and all three format arms
        # then lost the same 18-19 controls while differing from each other by one question
        # — the damage tracked the rebuild, not the formatting, and buried whatever the
        # formatting was worth. Here the message sequence, the system prompt and the tool
        # call structure stay byte-identical; only the CONTENT of each tool result is
        # re-rendered, so the arm measures formatting and nothing else.
        table = render(rows, arm)
        first_tool = True
        new = []
        for m in msgs:
            if m.get("role") == "tool" and first_tool:
                mm = dict(m)
                mm["content"] = json.dumps({"evidence": table}, ensure_ascii=False)
                new.append(mm)
                first_tool = False
            elif m.get("role") == "tool":
                mm = dict(m)
                mm["content"] = json.dumps({"evidence": "(folded into the first result)"})
                new.append(mm)
            else:
                new.append(m)
        return call(new)

    for arm in a.arms.split(","):
        got = {}
        for label, rows in (("t", target), ("c", control)):
            ok = 0
            for rec in rows:
                ans = run(rec, arm)
                if not ans:
                    continue
                try:
                    ok += int(bool(evaluate_llm_judge(
                        rec["question"], rec["expected_answer"], ans)))
                except Exception:  # noqa: BLE001
                    pass
            got[label] = ok
        print(f"{arm:>10s} {got['t']:>4d}/{len(target)}   {got['c']:>4d}/{len(control)}",
              flush=True)


if __name__ == "__main__":
    main()
