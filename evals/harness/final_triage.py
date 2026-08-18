"""Where does each wrong answer actually break? The definitive pass.

Two earlier attempts at this got it wrong in opposite directions, both because they matched
the gold answer by token overlap:

  the first  called 83 of 169 errors "the answerer had everything and still failed", because
             the gold's words appeared SOMEWHERE among thirty rows of prose. A reader model
             looking at the same contexts put only 12% of them in that bucket.
  the second tried to check the store the same way and asked a model to find a fact inside
             150k characters. Its positive control — questions we answer CORRECTLY — came
             back 93% ABSENT, so the method was measuring context length, not ingest.

What survived is the context audit: on questions we get right it says SUPPORTED 94% of the
time, on questions we get wrong 12%. That separation is what makes it usable.

This applies that audit to every error, and answers the store question the way the failed
attempt should have — by narrowing the haystack with keyword search before asking, so the
model reads forty candidate rows rather than the whole store. Three votes per verdict,
majority wins, because a single call from this model flips on borderline items.

Each stage prints its own positive control on correctly-answered questions. A stage whose
control does not separate is reported as unusable rather than quietly believed.

    python final_triage.py
"""
import asyncio
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys

RUN = "/home/lj/code/MIRIX/evals/results/locomo/ab_graph_r1"
PREFIX = "clean-r1__"
DB = "mirix_locomo_clean_r1"
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which who how "
           "why are does".split())

SUFFICIENCY = """You are auditing whether a memory system RETRIEVED what was needed, not
whether it answered well.

Decide whether the evidence below contains what a careful reader would need to produce the
correct answer. Judge only the evidence. Do not use outside knowledge.

SUPPORTED  - a statement in the evidence gives the answer, directly or by one obvious step.
PARTIAL    - the subject or topic appears, but the specific fact needed is not stated.
ABSENT     - nothing in the evidence bears on the question.

Question: {q}
Correct answer: {gold}

Evidence:
{ctx}

Reply with exactly one word: SUPPORTED, PARTIAL or ABSENT."""

DERIVABLE = """Below is a conversation excerpt and a question with its official answer.

Decide whether that answer is actually derivable from the excerpt, or whether it requires
information or judgement the excerpt does not contain.

DERIVABLE   - the excerpt states or clearly implies the answer.
INFERENCE   - the answer is a plausible interpretation, but the excerpt does not establish it.
CONTRADICTS - the excerpt says something incompatible with the answer.

Question: {q}
Official answer: {gold}

Excerpt:
{ctx}

Reply with exactly one word: DERIVABLE, INFERENCE or CONTRADICTS."""


def content(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 2 or x.isdigit()) and x not in STOP}


def main() -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    turns, gold_ev = {}, {}
    for sid, x in lc.items():
        for k, v in x["conversation"].items():
            if isinstance(v, list):
                for t in v:
                    if t.get("dia_id"):
                        turns[(sid, t["dia_id"])] = (
                            f"{t.get('speaker')}: {t.get('text','')} "
                            f"{t.get('blip_caption') or ''}")
        for qa in x.get("qa", []):
            gold_ev[(sid, qa["question"])] = qa.get("evidence") or []

    # store rows, kept individually so a question can be given a narrowed haystack
    rows = collections.defaultdict(list)
    q = ("SELECT user_id, coalesce(summary,'')||' — '||coalesce(details,'') "
         "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
         "UNION ALL SELECT user_id, coalesce(name,'')||' — '||coalesce(details,'') "
         "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%';")
    out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                          "-d", DB, "-At", "-F", "\t", "-c", q],
                         capture_output=True, text=True,
                         env={**os.environ, "PGPASSWORD": "mirix"},
                         cwd="/home/lj/MIRIX_eval").stdout
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[1].strip():
            rows[p[0]].append(" ".join(p[1].split()))
    print(f"store: {sum(len(v) for v in rows.values())} rows", flush=True)

    rec = {}
    for f in glob.glob(RUN + "/conv-*.json"):
        if f.endswith("_memories.json"):
            continue
        for r in json.load(open(f)).get("records", {}).values():
            rec[(r["sample_id"], r["question"])] = r

    J = [x for x in json.load(open(RUN + "/metrics.json"))["llm_judge_results"]
         if str(x.get("category")) != "5"]
    errs = [x for x in J if x.get("score") == 0]
    rights = [x for x in J if x.get("score") == 1]
    random.seed(7)
    ctrl = random.sample([x for x in rights if (x["sample_id"], x["question"]) in rec],
                         min(80, len(rights)))

    def ctx_of(x):
        r = rec.get((x["sample_id"], x["question"]))
        if not r:
            return ""
        return "\n".join(str(m.get("content") or "") for m in (r.get("messages") or []))

    def haystack(x, k=40):
        """The store rows most likely to bear on this question, by word overlap."""
        want = content(x["question"]) | content(x["expected_answer"])
        scored = []
        for r in rows.get(PREFIX + x["sample_id"], []):
            c = content(r)
            if not c:
                continue
            scored.append((len(want & c) / (len(want) ** 0.5 + 1), r))
        scored.sort(reverse=True)
        return "\n".join(f"- {r}" for _, r in scored[:k])

    def evidence_text(x):
        ev = gold_ev.get((x["sample_id"], x["question"])) or []
        got = [turns.get((x["sample_id"], t), "") for t in ev]
        return "\n".join(t for t in got if t)

    sem = asyncio.Semaphore(10)

    async def vote(prompt, keys):
        async with sem:
            async def one():
                for a in range(4):
                    try:
                        r = await client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0, max_completion_tokens=8)
                        t = (r.choices[0].message.content or "").strip().upper()
                        for k in keys:
                            if k in t:
                                return k
                        return keys[-1]
                    except Exception:  # noqa: BLE001
                        await asyncio.sleep(2 ** a)
                return keys[-1]
            v = await asyncio.gather(*[one() for _ in range(3)])
            return collections.Counter(v).most_common(1)[0][0]

    SUF = ["SUPPORTED", "PARTIAL", "ABSENT"]
    DER = ["DERIVABLE", "INFERENCE", "CONTRADICTS"]

    async def classify(x):
        gold, qq = x["expected_answer"], x["question"]
        # 1. is the gold even derivable from the turns the benchmark cites?
        ev = evidence_text(x)
        der = await vote(DERIVABLE.format(q=qq, gold=gold, ctx=ev), DER) if ev else "DERIVABLE"
        if der in ("INFERENCE", "CONTRADICTS"):
            return f"GOLD — {der.lower()}"
        # 2. did the evidence reach the answerer?
        suf = await vote(SUFFICIENCY.format(q=qq, gold=gold, ctx=ctx_of(x)[:400000]), SUF)
        if suf == "SUPPORTED":
            return "ANSWERING — evidence was there"
        # 3. is it in the store at all, judged over a narrowed haystack?
        st = await vote(SUFFICIENCY.format(q=qq, gold=gold, ctx=haystack(x)), SUF)
        return ("RETRIEVAL — in store, not retrieved" if st == "SUPPORTED"
                else "INGEST — never written")

    async def run():
        v = await asyncio.gather(*[classify(x) for x in errs])
        # positive controls: the same three stages on questions we get RIGHT
        cd = await asyncio.gather(*[
            vote(DERIVABLE.format(q=x["question"], gold=x["expected_answer"],
                                  ctx=evidence_text(x)), DER)
            for x in ctrl if evidence_text(x)])
        cs = await asyncio.gather(*[
            vote(SUFFICIENCY.format(q=x["question"], gold=x["expected_answer"],
                                    ctx=ctx_of(x)[:400000]), SUF) for x in ctrl])
        ch = await asyncio.gather(*[
            vote(SUFFICIENCY.format(q=x["question"], gold=x["expected_answer"],
                                    ctx=haystack(x)), SUF) for x in ctrl])
        return v, cd, cs, ch

    verdicts, cd, cs, ch = asyncio.run(run())

    cnt = collections.Counter(verdicts)
    tot = len(verdicts)
    print(f"\n{RUN.split('/')[-1]}: {len(J)} questions, {tot} wrong\n")
    print(f"{'bucket':40s}{'n':>5s}{'of errors':>11s}{'of all':>9s}")
    for k, v in cnt.most_common():
        print(f"  {k:38s}{v:>5d}{v/tot:>11.0%}{v/len(J):>9.1%}")

    print("\npositive controls, same stages on 80 questions we answer CORRECTLY:")
    print(f"  gold derivable from cited turns : {collections.Counter(cd)['DERIVABLE']}/{len(cd)}")
    print(f"  context sufficient              : {collections.Counter(cs)['SUPPORTED']}/{len(cs)}")
    print(f"  narrowed store sufficient       : {collections.Counter(ch)['SUPPORTED']}/{len(ch)}")
    print("  a stage only means something if its control is high; the whole-store version of")
    print("  stage 3 scored 7% here and was discarded.")

    bycat = collections.defaultdict(collections.Counter)
    C = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}
    for x, v in zip(errs, verdicts):
        bycat[C.get(str(x.get("category")), "?")][v] += 1
    ks = [k for k, _ in cnt.most_common()]
    print(f"\n{'category':13s}" + "".join(f"{k.split(' ')[0][:11]:>13s}" for k in ks))
    for c in ("single-hop", "multi-hop", "temporal", "open-domain"):
        print(f"{c:13s}" + "".join(f"{bycat[c][k]:>13d}" for k in ks))

    json.dump([{"sample_id": x["sample_id"], "question": x["question"],
                "gold": x["expected_answer"], "answer": x.get("predicted_answer"),
                "category": C.get(str(x.get("category"))), "bucket": v}
               for x, v in zip(errs, verdicts)],
              open("/home/lj/MIRIX_eval/final_triage.json", "w"),
              ensure_ascii=False, indent=1)
    print("\nper-question verdicts -> /home/lj/MIRIX_eval/final_triage.json")


if __name__ == "__main__":
    main()
