"""For a wrong multi-hop answer, is the evidence really absent from memory, or just worded
differently?

The merge simulation put 19 of 31 multi-hop errors in "no memory describes this evidence
turn", which would make ingest recall the dominant failure. But that bucket was decided by
token overlap at 0.5, so a memory that captured the turn faithfully in its own words falls
into it too. The difference matters a lot: "never stored" is an extraction problem, "stored
in different words" is a retrieval problem, and they need opposite fixes.

So check each supposedly-missing turn three ways, cheapest first:
  distinctive content  do the turn's rare words (names, numbers, quoted titles) appear
                       anywhere in this conversation's store?
  best overlap         what is the highest token overlap any single memory achieves?
  gold answer          does the gold answer's own text appear in the store?

A turn whose rare words are absent AND whose best overlap is low is genuinely missing.
A turn with high best-overlap that merely fell under 0.5 is a threshold artifact.
"""
import collections
import json
import os
import re
import subprocess

PREFIX = "clean-r1__"
RUN = "ab_graph_r1"
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them what when which".split())


def toks(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return frozenset(x for x in w if len(x) > 2 and x not in STOP)


def main() -> None:
    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    turn_text, turn_raw = {}, {}
    for sid, x in lc.items():
        for k, v in x["conversation"].items():
            if isinstance(v, list):
                for t in v:
                    if t.get("dia_id"):
                        raw = f"{t.get('text','')} {t.get('blip_caption') or ''}"
                        turn_raw[(sid, t["dia_id"])] = raw
                        turn_text[(sid, t["dia_id"])] = toks(raw)
    gold_ev, gold_ans = {}, {}
    for sid, x in lc.items():
        for q in x.get("qa", []):
            gold_ev[(sid, q["question"])] = q.get("evidence") or []
            gold_ans[(sid, q["question"])] = q.get("answer")

    # whole-store text per conversation, plus per-memory texts for best-overlap
    store_all, store_mem = {}, collections.defaultdict(list)
    q = ("SELECT user_id, coalesce(summary,'')||' '||coalesce(details,'') "
         "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
         "UNION ALL SELECT user_id, coalesce(name,'')||' '||coalesce(details,'') "
         "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%';")
    out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                          "-d", "mirix_locomo_clean_r1", "-At", "-F", "\t", "-c", q],
                         capture_output=True, text=True,
                         env={**os.environ, "PGPASSWORD": "mirix"},
                         cwd="/home/lj/MIRIX_eval").stdout
    for line in out.splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            store_mem[p[0]].append(toks(p[1]))
            store_all[p[0]] = store_all.get(p[0], "") + " " + p[1].lower()

    J = [x for x in json.load(open(
        f"/home/lj/code/MIRIX/evals/results/locomo/{RUN}/metrics.json"))["llm_judge_results"]
        if str(x.get("category")) == "1" and x.get("score") == 0]

    RARE = re.compile(r'"[^"]{3,40}"|\b[A-Z][a-z]{2,}\b|\b\d{2,4}\b')
    stat = collections.Counter()
    examples = collections.defaultdict(list)
    for x in J:
        sid, qq = x["sample_id"], x["question"]
        u = PREFIX + sid
        miss = []
        for turn in gold_ev.get((sid, qq)) or []:
            tt = turn_text.get((sid, turn))
            if not tt or len(tt) < 3:
                continue
            best = max((len(tt & m) / len(tt) for m in store_mem[u]), default=0.0)
            if best >= 0.5:
                continue
            raw = turn_raw[(sid, turn)]
            rare = {r.strip('"').lower() for r in RARE.findall(raw)}
            rare = {r for r in rare if len(r) > 3}
            present = sum(1 for r in rare if r in store_all.get(u, ""))
            miss.append((turn, best, len(rare), present, raw))
        if not miss:
            stat["evidence all present"] += 1
            continue
        turn, best, nrare, present, raw = min(miss, key=lambda m: m[1])
        ga = toks(gold_ans.get((sid, qq)))
        gold_in_store = (len(ga & toks(store_all.get(u, ""))) / len(ga)) if ga else 0
        if nrare and present == 0:
            k = "genuinely missing (rare words absent too)"
        elif best >= 0.35:
            k = "threshold artifact (worded differently)"
        elif gold_in_store >= 0.8:
            k = "turn missing but gold answer IS in store"
        else:
            k = "partly missing (some detail survived)"
        stat[k] += 1
        examples[k].append((sid, qq, turn, best, nrare, present, raw[:110]))

    tot = sum(stat.values())
    print(f"{RUN}: {tot} multi-hop errors\n")
    for k, v in stat.most_common():
        print(f"  {k:44s}{v:>4d}  {v/tot:5.0%}")
    for k in examples:
        print(f"\n=== {k} ===")
        for sid, qq, turn, best, nrare, present, raw in examples[k][:3]:
            print(f"  [{sid}] {qq[:56]}")
            print(f"     turn {turn}  best-overlap {best:.2f}  rare words {present}/{nrare} in store")
            print(f"     source: {raw}")


if __name__ == "__main__":
    main()
