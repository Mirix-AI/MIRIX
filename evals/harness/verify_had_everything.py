"""Was the evidence really in the context, or did the gold's words merely appear somewhere?

The triage put 83 of 169 errors in "had everything, answered wrong", and that bucket now
carries the whole argument that the answering step is the bottleneck. It was decided by token
overlap: the gold answer's content words appearing anywhere across the thirty-odd retrieved
rows. On a one- or two-word gold that is a weak test. If the gold is "Spanish" and any row
mentions Spanish for an unrelated reason, the question counts as fully supported even when
the sentence that would answer it was never retrieved.

So the bucket is an upper bound, and the part of it that is really a retrieval failure is
unknown. This asks a model instead, once per question, with the actual context in front of
it and the gold withheld from the judgement of sufficiency:

    SUPPORTED    a reader given only this context could answer correctly
    PARTIAL      the topic is present but the specific fact is not
    ABSENT       nothing here answers it

Only SUPPORTED is genuinely the answerer's failure. PARTIAL and ABSENT belong to retrieval,
and would move the headline accordingly.

    python verify_had_everything.py
"""
import asyncio
import collections
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "/home/lj/code/MIRIX/evals")

RUN = "/home/lj/code/MIRIX/evals/results/locomo/ab_graph_r1"
PREFIX = "clean-r1__"
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which who "
           "how why".split())

PROMPT = """You are auditing a memory system's retrieval, not its answer.

Below is the evidence that was retrieved for a question, and the correct answer. Decide
whether the evidence CONTAINS what a careful reader would need to produce that answer.

Judge only sufficiency of the evidence. Do not judge whether the answer is well phrased,
and do not use knowledge beyond the evidence shown.

SUPPORTED  - a specific statement in the evidence gives the answer, directly or by an
             obvious one-step inference from what is stated.
PARTIAL    - the subject or topic appears, but the specific fact needed is not stated.
ABSENT     - nothing in the evidence bears on the question.

Question: {q}
Correct answer: {gold}

Evidence:
{ctx}

Reply with exactly one word: SUPPORTED, PARTIAL or ABSENT."""


def content(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 2 or x.isdigit()) and x not in STOP}


async def main() -> None:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    src = {}
    for sid, x in lc.items():
        p = []
        for k, v in x["conversation"].items():
            if isinstance(v, list):
                for t in v:
                    p.append(f"{t.get('text','')} {t.get('blip_caption') or ''}")
        src[sid] = content(" ".join(p))

    q = ("SELECT user_id, string_agg(t,' ') FROM ("
         "SELECT user_id, coalesce(summary,'')||' '||coalesce(details,'') AS t "
         "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
         "UNION ALL SELECT user_id, coalesce(name,'')||' '||coalesce(details,'') "
         "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%'"
         ") z GROUP BY user_id;")
    out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                          "-d", "mirix_locomo_clean_r1", "-At", "-F", "\t", "-c", q],
                         capture_output=True, text=True,
                         env={**os.environ, "PGPASSWORD": "mirix"},
                         cwd="/home/lj/MIRIX_eval").stdout
    store = {p.split("\t")[0]: content(p.split("\t")[1])
             for p in out.splitlines() if len(p.split("\t")) >= 2}

    rec = {}
    for f in glob.glob(RUN + "/conv-*.json"):
        if f.endswith("_memories.json"):
            continue
        for r in json.load(open(f)).get("records", {}).values():
            rec[(r["sample_id"], r["question"])] = r

    J = [x for x in json.load(open(RUN + "/metrics.json"))["llm_judge_results"]
         if str(x.get("category")) != "5" and x.get("score") == 0]

    # reproduce exactly the bucket under audit
    had = []
    for x in J:
        sid, qq = x["sample_id"], x["question"]
        g = content(x["expected_answer"])
        if not g:
            continue
        r = rec.get((sid, qq))
        if not r:
            continue
        ctx_txt = "\n".join(str(m.get("content") or "") for m in (r.get("messages") or []))
        c = content(ctx_txt)
        if len(g & src[sid]) / len(g) < 0.5:
            continue
        if len(g & store.get(PREFIX + sid, set())) / len(g) < 0.5:
            continue
        if len(g & c) / len(g) < 0.5:
            continue
        if len(g & content(x.get("predicted_answer"))) / len(g) >= 0.6:
            continue
        had.append((x, ctx_txt))
    print(f"bucket under audit: {len(had)} questions", flush=True)

    sem = asyncio.Semaphore(8)

    async def one(x, ctx):
        async with sem:
            # 60000 was too small: the episodic half of a single conversation's store runs to
            # 73k characters on conv-48, so truncating there asked "is it in the first half"
            # and answered 93% ABSENT for reasons that had nothing to do with ingest.
            body = PROMPT.format(q=x["question"], gold=x["expected_answer"], ctx=ctx[:600000])
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[{"role": "user", "content": body}],
                        temperature=0, max_completion_tokens=8)
                    t = (r.choices[0].message.content or "").strip().upper()
                    for k in ("SUPPORTED", "PARTIAL", "ABSENT"):
                        if k in t:
                            return k
                    return "?"
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2 ** attempt)
            return "?"

    verdicts = await asyncio.gather(*[one(x, c) for x, c in had])

    # For everything the audit demoted, ask the same question of the WHOLE store. If the
    # store can answer it, the fact was written and simply not retrieved; if the store
    # cannot, it was never written. Those two need opposite fixes, and the aggregate
    # numbers so far cannot tell them apart.
    demoted = [(x, c, v) for (x, c), v in zip(had, verdicts) if v in ("PARTIAL", "ABSENT")]
    store_txt = {}
    q2 = ("SELECT user_id, string_agg(t, E'\n') FROM ("
          "SELECT user_id, coalesce(summary,'')||' '||coalesce(details,'') AS t "
          "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
          "UNION ALL SELECT user_id, coalesce(name,'')||' '||coalesce(details,'') "
          "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%'"
          ") z GROUP BY user_id;")
    o2 = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                         "-d", "mirix_locomo_clean_r1", "-At", "-F", "\t", "-c", q2],
                        capture_output=True, text=True,
                        env={**os.environ, "PGPASSWORD": "mirix"},
                        cwd="/home/lj/MIRIX_eval").stdout
    for line in o2.splitlines():
        pp = line.split("\t")
        if len(pp) >= 2:
            store_txt[pp[0]] = pp[1]
    sv = await asyncio.gather(*[
        one(x, store_txt.get(PREFIX + x["sample_id"], "")) for x, _, _ in demoted])
    c2 = collections.Counter(sv)
    print(f"\n=== of the {len(demoted)} demoted, asked against the WHOLE store ===")
    for k in ("SUPPORTED", "PARTIAL", "ABSENT", "?"):
        if c2[k]:
            print(f"  {k:11s}{c2[k]:>4d}  {c2[k]/len(demoted):5.0%}")
    print("  SUPPORTED here = written but not retrieved (a retrieval fix)")
    print("  PARTIAL/ABSENT = never written (an ingest fix)")
    cnt = collections.Counter(verdicts)
    print()
    for k in ("SUPPORTED", "PARTIAL", "ABSENT", "?"):
        if cnt[k]:
            print(f"  {k:11s}{cnt[k]:>4d}  {cnt[k]/len(had):5.0%}")
    print(f"\n  token-overlap said all {len(had)} had everything.")
    print(f"  a reader says {cnt['SUPPORTED']} did.")
    print("\n=== examples the audit demoted ===")
    shown = 0
    for (x, _), v in zip(had, verdicts):
        if v in ("PARTIAL", "ABSENT") and shown < 5:
            print(f"  [{v}] {x['question'][:58]}")
            print(f"        gold: {str(x['expected_answer'])[:42]}")
            shown += 1


asyncio.run(main())
