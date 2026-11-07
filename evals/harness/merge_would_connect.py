"""Would resolving anchors connect the evidence we currently fail to join?

The graph is worth +63 questions, but +43 of those are single-hop and only +11.5 are
multi-hop — it earns its keep as an index, not as a structure. Separately, the structure is
measurably degenerate: 65% of anchors touch one fact or none, 58% overlap heavily with
another anchor of the same user, and the hubs are nothing but speaker names.

The tempting story is that the second fact explains the first. Twice now that kind of
inference has been wrong here — "the repair pass fragments the graph so it hurts multi-hop"
reversed under clean prompts, and "the graph displaces flat rows from the context" died when
running both retrievals changed nothing. So this measures instead of inferring.

For each multi-hop question we get wrong, take the gold evidence turns, find the memories
citing them, and ask whether those memories currently share an anchor. If they do, joining
was possible and something else failed. If they do not, ask whether merging near-duplicate
anchors would make them share one. Only that second group is evidence that fragmentation
costs answers.

Nothing is written. Merging is simulated in memory, and names are never merged with names —
the hubs are already too connected.

    python merge_would_connect.py
"""
import collections
import itertools
import json
import os
import re

PREFIX = "clean-r1__"
RUN = "ab_graph_r1"
STOP = set("the a an and or of to in on at for with from that this it is was were be been "
           "have has had do does did his her their about they them".split())
HARD = re.compile(r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b|"
                  r"\b(january|february|march|april|may|june|july|august|september|"
                  r"october|november|december)\b", re.I)


def toks(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return frozenset(x for x in w if len(x) > 2 and x not in STOP)


def hard(t):
    return frozenset(m.group(0).lower() for m in HARD.finditer(str(t or "")))


def main() -> None:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv("/home/lj/MIRIX_eval/.env")

    lc = {x["sample_id"]: x for x in json.load(
        open("/home/lj/code/MIRIX/evals/data/locomo10.json"))}
    gold_ev = {}
    for sid, x in lc.items():
        for q in x.get("qa", []):
            gold_ev[(sid, q["question"])] = q.get("evidence") or []

    J = [x for x in json.load(open(
        f"/home/lj/code/MIRIX/evals/results/locomo/{RUN}/metrics.json"))["llm_judge_results"]
        if str(x.get("category")) == "1" and x.get("score") == 0]
    print(f"{RUN}: {len(J)} multi-hop errors", flush=True)

    d = GraphDatabase.driver(os.environ["MIRIX_NEO4J_URI"],
                             auth=(os.environ["MIRIX_NEO4J_USER"],
                                   os.environ["MIRIX_NEO4J_PASSWORD"]))
    # V7Fact stores memory_ids (not citation_memory_ids) and carries no source_refs, and
    # Postgres records provenance as session_id/turn_id rather than the dia_id the gold
    # uses. Rather than reconcile two indexing schemes, map a gold evidence turn to the
    # memories that describe it by text overlap — the same mapping used earlier to trace
    # where an answer is lost, and robust to how provenance happens to be encoded.
    mem_anchors = collections.defaultdict(set)
    anchors = {}
    deg = collections.Counter()
    with d.session() as s:
        for r in s.run(
                "MATCH (f:V7Fact)-[:V7_FACT_ARG]->(a:V7Anchor) "
                "WHERE f.user_id STARTS WITH $p "
                "RETURN f.user_id AS u, a.id AS aid, a.name AS name, "
                "a.anchor_type AS t, coalesce(f.memory_ids, []) AS mids", p=PREFIX):
            anchors[r["aid"]] = (r["name"], r["t"], r["u"])
            deg[r["aid"]] += 1
            for m in r["mids"]:
                mem_anchors[(r["u"], m)].add(r["aid"])
    d.close()

    # memory id -> its text, from Postgres
    import subprocess
    mem_text = {}
    q = ("SELECT user_id, id, coalesce(summary,'')||' '||coalesce(details,'') "
         "FROM episodic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%' "
         "UNION ALL SELECT user_id, id, coalesce(name,'')||' '||coalesce(details,'') "
         "FROM semantic_memory WHERE NOT is_deleted AND user_id LIKE 'clean-r1__%';")
    out = subprocess.run(["pgenv/bin/psql", "-w", "-h", "localhost", "-U", "mirix",
                          "-d", "mirix_locomo_clean_r1", "-At", "-F", "\t", "-c", q],
                         capture_output=True, text=True,
                         env={**os.environ, "PGPASSWORD": "mirix"},
                         cwd="/home/lj/MIRIX_eval").stdout
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            mem_text[(parts[0], parts[1])] = toks(parts[2])
    print(f"loaded {len(mem_text)} memory texts, {len(anchors)} anchors", flush=True)

    turn_text = {}
    for sid, x in lc.items():
        for k, v in x["conversation"].items():
            if isinstance(v, list):
                for t in v:
                    if t.get("dia_id"):
                        turn_text[(sid, t["dia_id"])] = toks(
                            f"{t.get('text','')} {t.get('blip_caption') or ''}")

    names = {a for a, (n, t, _) in anchors.items()
             if len(str(n or "").split()) == 1 and deg[a] >= 50}

    # simulate the merge, per conversation, at the conservative threshold
    by_user = collections.defaultdict(list)
    for a, (n, t, u) in anchors.items():
        if a not in names:
            by_user[u].append((a, toks(n), hard(n), str(n or "").lower()))
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    merged = 0
    for u, rows in by_user.items():
        for (a1, t1, h1, n1), (a2, t2, h2, n2) in itertools.combinations(rows, 2):
            if not t1 or not t2 or h1 != h2:
                continue
            # possessive guard: "Melanie" vs "Melanie's art" are not the same thing
            if n1.startswith(n2 + "'s ") or n2.startswith(n1 + "'s "):
                continue
            if len(t1 & t2) / len(t1 | t2) < 0.6:
                continue
            r1, r2 = find(a1), find(a2)
            if r1 != r2:
                parent[r2] = r1
                merged += 1
    print(f"simulated {merged} anchor merges (jaccard>=0.6, no names, no possessives)\n",
          flush=True)

    stat = collections.Counter()
    for x in J:
        sid, q = x["sample_id"], x["question"]
        ev = gold_ev.get((sid, q)) or []
        if len(ev) < 2:
            stat["gold cites <2 turns"] += 1
            continue
        u = PREFIX + sid
        # memories citing each evidence turn
        groups = []
        for turn in ev:
            tt = turn_text.get((sid, turn))
            if not tt or len(tt) < 3:
                continue
            ms = [m for (uu, m), mt in mem_text.items()
                  if uu == u and mt and len(tt & mt) / len(tt) >= 0.5]
            if ms:
                groups.append(set().union(*[mem_anchors[(u, m)] for m in ms]))
        if len(groups) < 2:
            stat["evidence turn has no memory"] += 1
            continue
        raw = set.intersection(*groups) if groups else set()
        mgd = [set(find(a) for a in g) for g in groups]
        after = set.intersection(*mgd) if mgd else set()
        if raw - names:
            stat["already share an anchor"] += 1
        elif after - names:
            stat["MERGE WOULD CONNECT"] += 1
        else:
            stat["still unconnected"] += 1

    tot = sum(stat.values())
    print(f"{'multi-hop errors':34s}{tot:>5d}")
    for k, v in stat.most_common():
        print(f"  {k:32s}{v:>5d}  {v/tot:5.0%}")
    print("\n  'MERGE WOULD CONNECT' is the only bucket where anchor resolution could have")
    print("  helped. 'already share an anchor' means the join was available and unused —")
    print("  a retrieval or answerer failure, not a graph-structure one.")


if __name__ == "__main__":
    main()
