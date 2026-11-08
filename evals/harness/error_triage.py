"""Every wrong answer, sorted into the three places it can fail.

Three stages can lose an answer, and they need completely different fixes, so guessing
between them is expensive. For each error, follow the gold answer through the pipeline:

  STORE      is the answer in Postgres at all, in any wording?
  RETRIEVAL  did it reach the context the answerer actually read?
  ANSWERING  it was in the context and the answer is still wrong.

Two things this deliberately does NOT do. It does not use the gold evidence turn ids — the
earlier pass keyed on those and their token overlap put 61% of multi-hop errors in "not
stored", which a closer look showed was almost entirely its own 0.5 cutoff talking. And it
does not treat a low score as proof of a defect: golds needing inference ("Somewhat, but not
extremely religious") are flagged separately rather than counted against the store, because
their words were never going to appear anywhere.

Matching is by content-word overlap against the gold answer, which is crude on one- or
two-word golds. Every bucket is therefore reported with its example questions so the reader
can judge, and the counts should be read as a shape, not a measurement.

    python error_triage.py --run ab_graph_r1
"""
import argparse
import collections
import json
import glob
import os
import re
import subprocess

PREFIX = "clean-r1__"
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which who "
           "how why does is are be to of in on at for with".split())
CAT = {"1": "multi-hop", "2": "temporal", "3": "open-domain", "4": "single-hop"}


def toks(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return frozenset(x for x in w if (len(x) > 2 or x.isdigit()) and x not in STOP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ab_graph_r1")
    ap.add_argument("--db", default="mirix_locomo_clean_r1")
    a = ap.parse_args()
    D = f"/home/lj/code/MIRIX/evals/results/locomo/{a.run}"

    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    src = {}
    for sid, x in lc.items():
        parts = []
        for k, v in x["conversation"].items():
            if isinstance(v, list):
                for t in v:
                    parts.append(f"{t.get('text','')} {t.get('blip_caption') or ''}")
        src[sid] = toks(" ".join(parts))

    store = {}
    q = ("SELECT user_id, string_agg(t, ' ') FROM ("
         "SELECT user_id, coalesce(summary,'')||' '||coalesce(details,'') AS t "
         "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
         "UNION ALL SELECT user_id, coalesce(name,'')||' '||coalesce(details,'') "
         "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%'"
         ") z GROUP BY user_id;")
    out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                          "-d", a.db, "-At", "-F", "\t", "-c", q],
                         capture_output=True, text=True,
                         env={**os.environ, "PGPASSWORD": "mirix"},
                         cwd="/home/lj/MIRIX_eval").stdout
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            store[p[0]] = toks(p[1])

    rec = {}
    for f in glob.glob(f"{D}/conv-*.json"):
        if f.endswith("_memories.json"):
            continue
        for r in json.load(open(f)).get("records", {}).values():
            rec[(r["sample_id"], r["question"])] = r

    J = [x for x in json.load(open(f"{D}/metrics.json"))["llm_judge_results"]
         if str(x.get("category")) != "5"]
    errs = [x for x in J if x.get("score") == 0]
    print(f"{a.run}: {len(J)} questions, {len(errs)} wrong\n", flush=True)

    stat = collections.Counter()
    bycat = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list)
    for x in errs:
        sid, qq = x["sample_id"], x["question"]
        g = toks(x["expected_answer"])
        if not g:
            stat["unscorable (gold has no content words)"] += 1
            continue
        r = rec.get((sid, qq))
        ctx = toks(" ".join(str(m.get("content") or "") for m in (r.get("messages") or []))) if r else frozenset()
        in_src = len(g & src[sid]) / len(g)
        in_store = len(g & store.get(PREFIX + sid, frozenset())) / len(g)
        in_ctx = len(g & ctx) / len(g)
        in_ans = len(g & toks(x.get("predicted_answer"))) / len(g)

        if in_src < 0.5:
            k = "GOLD needs inference / not in transcript"
        elif in_store < 0.5:
            k = "STORE — lost at ingest"
        elif in_ctx < 0.5:
            k = "RETRIEVAL — in store, never retrieved"
        elif in_ans >= 0.6:
            k = "ANSWERING — answer contains gold, judged wrong anyway"
        else:
            k = "ANSWERING — had everything, answered wrong"
        stat[k] += 1
        bycat[CAT.get(str(x.get("category")), "?")][k] += 1
        ex[k].append((sid, qq, x["expected_answer"], x.get("predicted_answer")))

    tot = sum(stat.values())
    print(f"{'bucket':46s}{'n':>5s}{'share':>8s}")
    for k, v in stat.most_common():
        print(f"  {k:44s}{v:>5d}{v/tot:>8.0%}")

    print(f"\n{'category':13s}" + "".join(f"{k[:11]:>13s}" for k in stat))
    for c in ("single-hop", "multi-hop", "temporal", "open-domain"):
        print(f"{c:13s}" + "".join(f"{bycat[c][k]:>13d}" for k in stat))

    for k in stat:
        if not ex[k]:
            continue
        print(f"\n=== {k} ===")
        for sid, qq, gold, pred in ex[k][:3]:
            print(f"  [{sid}] {qq[:58]}")
            print(f"     gold: {str(gold)[:44]}")
            print(f"     ans : {str(pred)[:64]}")


if __name__ == "__main__":
    main()
