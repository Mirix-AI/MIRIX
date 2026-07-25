"""Refine the v7.10 hypergraph: remove redundancy without losing information.

Four passes (dry-run by default; pass --apply to write):
  1. TAUTOLOGY   drop facts whose subject == object ("french -include-> french")
  2. DEDUPE      collapse identical (subject, predicate, object) into ONE V7Fact,
                 keeping every source as an extra V7_FACT_FROM edge — provenance
                 is preserved, only the duplicated node goes away. This is the
                 correct hypergraph semantics: a fact is one fact, cited N times.
  3. PREDICATE   canonicalize surface variants (includes->include, is located in
                 -> located in) so the same relation is one relation
  4. PRUNE       delete anchors no fact touches AND that link <=1 memory (dead
                 weight: they can neither answer nor bridge)
"""
import re
import sys
from collections import defaultdict

from neo4j import GraphDatabase

APPLY = "--apply" in sys.argv
U = "longmem_s_0"
d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "mirix_neo4j_dev"))

AUX = re.compile(r"^(is|are|was|were|be|been|being)\s+", re.I)


def canon_pred(p: str) -> str:
    """Canonical predicate: lowercase, drop leading copula, singularize the verb."""
    if not p:
        return p
    p = AUX.sub("", p.lower().strip())
    toks = p.split()
    if toks:
        w = toks[0]
        # includes->include, offers->offer, provides->provide; keep has/is/ss-words
        if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "as")):
            toks[0] = w[:-1]
    return " ".join(toks)


def stats(s, tag):
    f = s.run("MATCH (f:V7Fact {user_id:$u}) RETURN count(f) AS c", u=U).single()["c"]
    a = s.run("MATCH (a:V7Anchor {user_id:$u}) RETURN count(a) AS c", u=U).single()["c"]
    p = s.run("MATCH (f:V7Fact {user_id:$u}) WHERE f.predicate IS NOT NULL "
              "RETURN count(DISTINCT f.predicate) AS c", u=U).single()["c"]
    e = s.run("MATCH (n {user_id:$u})-[r]->() RETURN count(r) AS c", u=U).single()["c"]
    print(f"  [{tag}] facts={f}  anchors={a}  distinct_predicates={p}  edges={e}")
    return f, a, p, e


with d.session() as s:
    print("=== BEFORE ===")
    stats(s, "before")

    # ---- 1. tautologies ----
    taut = s.run("""MATCH (x)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id:$u})-[:V7_FACT_OBJECT]->(y)
                    WHERE toLower(x.name)=toLower(y.name) RETURN count(f) AS c""", u=U).single()["c"]
    print(f"\n1. TAUTOLOGY  self-loop facts (subject==object): {taut}")
    if APPLY and taut:
        s.run("""MATCH (x)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id:$u})-[:V7_FACT_OBJECT]->(y)
                 WHERE toLower(x.name)=toLower(y.name) DETACH DELETE f""", u=U)

    # ---- 2. dedupe identical triples, keeping provenance ----
    dup_rows = s.run("""
        MATCH (su)<-[:V7_FACT_SUBJECT]-(f:V7Fact {user_id:$u})-[:V7_FACT_OBJECT]->(o)
        WITH toLower(su.name)+'|'+toLower(coalesce(f.predicate,''))+'|'+toLower(o.name) AS k,
             collect(f.id) AS ids
        WHERE size(ids) > 1 RETURN k, ids""", u=U).data()
    dropped = sum(len(r["ids"]) - 1 for r in dup_rows)
    print(f"2. DEDUPE     duplicate triples: {len(dup_rows)} groups, {dropped} redundant fact nodes "
          f"(sources re-attached to the survivor)")
    if APPLY:
        for r in dup_rows:
            keep, rest = r["ids"][0], r["ids"][1:]
            s.run("""
                MATCH (keep:V7Fact {id:$keep})
                UNWIND $rest AS rid
                MATCH (dupe:V7Fact {id:rid})-[:V7_FACT_FROM]->(m:V7MemoryRef)
                MERGE (keep)-[:V7_FACT_FROM]->(m)""", keep=keep, rest=rest)
            s.run("UNWIND $rest AS rid MATCH (dupe:V7Fact {id:rid}) DETACH DELETE dupe", rest=rest)

    # ---- 3. canonical predicates ----
    preds = [r["p"] for r in s.run("MATCH (f:V7Fact {user_id:$u}) WHERE f.predicate IS NOT NULL "
                                   "RETURN DISTINCT f.predicate AS p", u=U)]
    mapping = {p: canon_pred(p) for p in preds if canon_pred(p) != p}
    groups = defaultdict(list)
    for p in preds:
        groups[canon_pred(p)].append(p)
    merged = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    print(f"3. PREDICATE  {len(preds)} distinct -> {len(groups)} canonical ({merged} variants merged)")
    for k, v in sorted(groups.items(), key=lambda x: -len(x[1]))[:5]:
        if len(v) > 1:
            print(f"      '{k}' <- {v[:4]}")
    if APPLY and mapping:
        for old, new in mapping.items():
            s.run("MATCH (f:V7Fact {user_id:$u}) WHERE f.predicate=$old SET f.predicate=$new",
                  u=U, old=old, new=new)

    # ---- 4. prune dead-weight anchors ----
    dead = s.run("""MATCH (a:V7Anchor {user_id:$u})
                    WHERE NOT (a)<-[:V7_FACT_SUBJECT|V7_FACT_OBJECT]-(:V7Fact)
                    WITH a, size([(a)-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(:V7MemoryRef)|1]) AS deg
                    WHERE deg <= 1 RETURN count(a) AS c""", u=U).single()["c"]
    print(f"4. PRUNE      dead-weight anchors (no fact, <=1 memory): {dead}")
    if APPLY and dead:
        s.run("""MATCH (a:V7Anchor {user_id:$u})
                 WHERE NOT (a)<-[:V7_FACT_SUBJECT|V7_FACT_OBJECT]-(:V7Fact)
                 WITH a, size([(a)-[:V7_APPEARS_IN|V7_DESCRIBED_BY]->(:V7MemoryRef)|1]) AS deg
                 WHERE deg <= 1 DETACH DELETE a""", u=U)

    if APPLY:
        print("\n=== AFTER ===")
        stats(s, "after")
    else:
        print("\n(dry run — rerun with --apply to write)")
d.close()
