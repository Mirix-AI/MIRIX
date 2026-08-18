"""Should the cold-fact lane be run at all, and at what threshold?

Three hours of QA arms are only worth spending if this index can actually surface the turn
that answers a question we currently get wrong, without firing on the questions we already
get right. Both halves matter: every intervention that failed this week gained on the 169
wrong and lost more on the 1371 right, because the right set is eight times larger.

So sweep the cosine threshold and report, per threshold:

  target rank   for errors triaged as INGEST or RETRIEVAL, where the turn containing the
                gold answer sits in the cold-fact ranking. Rank 1-3 is reachable; the lane
                takes k=3.
  fire rate     the share of currently-correct questions where ANY cold fact clears the
                threshold. Every one of those is a chance to displace a working answer.

The existing call site uses thresh=0.82, tuned for 129 LLM-written facts on LongMemEval.
This index has 5832 raw turns, a different distribution entirely, so the old number carries
no information here.

KILL CRITERION, fixed before looking: if no threshold puts the gold turn in the top 3 for at
least a quarter of the target questions while firing on under 25% of controls, do not run
the arms.

    python coldfact_gate.py
"""
import collections
import json
import os
import random
import re

STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which who "
           "how why".split())


def content(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return {x for x in w if (len(x) > 2 or x.isdigit()) and x not in STOP}


def main() -> None:
    from openai import OpenAI
    from dotenv import load_dotenv
    import numpy as np
    load_dotenv("/home/lj/MIRIX_eval/.env")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    idx = {}
    for sid in [f"conv-{n}" for n in (26, 30, 41, 42, 43, 44, 47, 48, 49, 50)]:
        p = os.path.expanduser(f"~/MIRIX_eval/coldfacts_clean-r1__{sid}.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p, encoding="utf-8"))
        idx[sid] = (rows, np.array([r["emb"] for r in rows], dtype="float32"))
    print(f"index: {sum(len(v[0]) for v in idx.values())} turns over {len(idx)} conversations",
          flush=True)

    tri = json.load(open("/home/lj/MIRIX_eval/final_triage.json", encoding="utf-8"))
    target = [x for x in tri if x["bucket"].split(" —")[0] in ("INGEST", "RETRIEVAL")]
    J = [x for x in json.load(open(
        "/home/lj/code/MIRIX/evals/results/locomo/ab_graph_r1/metrics.json",
        encoding="utf-8"))["llm_judge_results"]
        if str(x.get("category")) != "5" and x.get("score") == 1]
    random.seed(11)
    control = random.sample(J, 200)
    print(f"target {len(target)} (INGEST+RETRIEVAL errors), control {len(control)}\n",
          flush=True)

    def embed(texts):
        out = []
        for i in range(0, len(texts), 256):
            r = client.embeddings.create(model="text-embedding-ada-002",
                                         input=[t[:400] for t in texts[i:i + 256]])
            out.extend(d.embedding for d in r.data)
        a = np.array(out, dtype="float32")
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)

    tq = embed([x["question"] for x in target])
    cq = embed([x["question"] for x in control])

    # where does the gold answer's own wording sit in the ranking?
    ranks, sims_at_gold = [], []
    for x, q in zip(target, tq):
        rows, mat = idx.get(x["sample_id"], (None, None))
        if rows is None:
            continue
        s = mat @ q
        order = np.argsort(-s)
        gold = content(x["gold"])
        if not gold:
            continue
        hit = None
        for rank, i in enumerate(order[:50], 1):
            if len(gold & content(rows[i]["fact"])) / len(gold) >= 0.6:
                hit = (rank, float(s[i]))
                break
        ranks.append(hit[0] if hit else 999)
        sims_at_gold.append(hit[1] if hit else 0.0)

    r = collections.Counter()
    for x in ranks:
        r["1" if x == 1 else "2-3" if x <= 3 else "4-10" if x <= 10 else
          "11-50" if x <= 50 else "not in top 50"] += 1
    print("rank of the turn containing the gold answer:")
    for k in ("1", "2-3", "4-10", "11-50", "not in top 50"):
        print(f"   {k:14s}{r[k]:>4d}  {r[k]/max(len(ranks),1):5.0%}")

    print(f"\n{'thresh':>8s}{'target top-3':>14s}{'control fire':>14s}{'verdict':>10s}")
    for th in (0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78):
        top3 = sum(1 for rank, s in zip(ranks, sims_at_gold)
                   if rank <= 3 and s >= th)
        fire = 0
        for x, q in zip(control, cq):
            rows, mat = idx.get(x["sample_id"], (None, None))
            if rows is None:
                continue
            if float((mat @ q).max()) >= th:
                fire += 1
        t3 = top3 / max(len(ranks), 1)
        fr = fire / len(control)
        ok = "PASS" if (t3 >= 0.25 and fr <= 0.25) else ""
        print(f"{th:>8.2f}{top3:>8d} {t3:>5.0%}{fire:>8d} {fr:>5.0%}{ok:>10s}")

    print("\n  gate: some threshold must reach >=25% of targets at top-3 while firing on")
    print("  <=25% of controls. No PASS row means do not spend the QA hours.")


if __name__ == "__main__":
    main()
