"""If anchors were resolved, how many facts would become connected that are not now?

Measured on the live v7.23 graph (10 LoCoMo conversations, prefix clean-r1__):
    10,269 anchors, 13,541 facts
    6,674 anchors (65%) participate in one fact or none
    5,957 anchors (58%) overlap heavily with another anchor of the same user
    the high-degree end is nothing but speaker names — Jolene 791, John 757, James 697

The graph is bimodal and neither mode supports a hop. You either land on a name that
connects to everything, or on a unique phrase that connects to one thing. Five separate
anchors for one concept —

    education and career options in counseling
    education and career options in mental health
    career options in counseling and mental health
    mental health careers
    counseling careers

— mean a question hitting one of them cannot reach the facts hanging off the other four.
That is a multi-hop failure caused at write time and invisible to any traversal policy,
which is consistent with twelve versions of traversal and reranking work moving the
benchmark by 1.0 point in total.

This simulates the merge OFFLINE. Nothing is written; no re-ingest is paid for. It answers
one question: after merging, how many PAIRS OF FACTS acquire a shared anchor that they do
not share today? That number is the traversal the graph cannot currently perform.

Two guards, because a merge that is too eager invents connections rather than restoring
them:
  * names are never merged with each other — the hubs are already too connected, and
    fusing "John" with "Jon" would be a different experiment
  * a numeric or month-name difference blocks a merge, the same rule the write-side dedup
    uses, because "two dogs" and "three dogs" are different anchors however similar

Reported against a threshold sweep, since the honest version of this result is a curve and
the operating point is a choice about false merges.

    python anchor_merge_sim.py
"""
import collections
import itertools
import os
import re
import sys

PREFIX = "clean-r1__"
STOP = set("the a an and or of to in on at for with from that this it is was were be "
           "been have has had do does did his her their about they them".split())
HARD = re.compile(
    r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)\b|"
    r"\b(january|february|march|april|may|june|july|august|september|october|november|"
    r"december)\b", re.I)


def toks(t):
    w = re.sub(r"[^a-z0-9 ]", " ", str(t or "").lower()).split()
    return frozenset(x for x in w if len(x) > 2 and x not in STOP)


def hard(t):
    return frozenset(m.group(0).lower() for m in HARD.finditer(str(t or "")))


def main() -> None:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv("/home/lj/MIRIX_eval/.env")
    d = GraphDatabase.driver(os.environ["MIRIX_NEO4J_URI"],
                             auth=(os.environ["MIRIX_NEO4J_USER"],
                                   os.environ["MIRIX_NEO4J_PASSWORD"]))
    anchors, facts = {}, collections.defaultdict(set)
    with d.session() as s:
        for r in s.run("MATCH (a:V7Anchor) WHERE a.user_id STARTS WITH $p "
                       "RETURN a.id AS id, a.name AS name, a.user_id AS u", p=PREFIX):
            anchors[r["id"]] = (r["name"], r["u"])
        for r in s.run("MATCH (f:V7Fact)-[:V7_FACT_ARG]->(a:V7Anchor) "
                       "WHERE f.user_id STARTS WITH $p "
                       "RETURN f.id AS f, a.id AS a", p=PREFIX):
            facts[r["f"]].add(r["a"])
    d.close()
    print(f"{len(anchors)} anchors, {len(facts)} facts", flush=True)

    # Speaker names: the hubs. Identified by being one word and very high degree, so the
    # rule does not depend on a hand-written list of this benchmark's characters.
    deg = collections.Counter()
    for a_ids in facts.values():
        for a in a_ids:
            deg[a] += 1
    names = {a for a, (n, _) in anchors.items()
             if len(str(n or "").split()) == 1 and deg[a] >= 50}
    print(f"treating {len(names)} high-degree single-word anchors as names (never merged)")

    by_user = collections.defaultdict(list)
    for a, (n, u) in anchors.items():
        by_user[u].append((a, toks(n), hard(n)))

    def pairs_sharing(assign):
        """Fact pairs that share at least one (merged) anchor."""
        bucket = collections.defaultdict(set)
        for f, a_ids in facts.items():
            for a in a_ids:
                bucket[assign.get(a, a)].add(f)
        seen = set()
        for k, fs in bucket.items():
            if k in names or len(fs) > 200:      # a hub connects everything; not a hop
                continue
            for x, y in itertools.combinations(sorted(fs), 2):
                seen.add((x, y))
        return seen

    base = pairs_sharing({})
    print(f"fact pairs sharing a non-hub anchor today: {len(base)}\n")
    print(f"{'threshold':>10s}{'merges':>9s}{'anchors left':>14s}{'new fact pairs':>16s}")
    for th in (0.9, 0.8, 0.7, 0.6, 0.5):
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        merges = 0
        for u, rows in by_user.items():
            rows = [r for r in rows if r[0] not in names and r[1]]
            for (a1, t1, h1), (a2, t2, h2) in itertools.combinations(rows, 2):
                if h1 != h2:
                    continue
                if len(t1 & t2) / len(t1 | t2) < th:
                    continue
                r1, r2 = find(a1), find(a2)
                if r1 != r2:
                    parent[r2] = r1
                    merges += 1
        assign = {a: find(a) for a in anchors}
        after = pairs_sharing(assign)
        left = len(set(assign.values()))
        print(f"{th:>10.1f}{merges:>9d}{left:>14d}{len(after - base):>16d}", flush=True)

    print("\n  'new fact pairs' is what the graph cannot traverse today and could after")
    print("  merging. Hub anchors and any merged group joining >200 facts are excluded —")
    print("  connecting everything to everything is not a hop, it is noise.")


if __name__ == "__main__":
    main()
