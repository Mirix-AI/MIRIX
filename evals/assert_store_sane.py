"""Pre-flight assertions for an evaluation arm. Fail loudly instead of scoring nothing.

Two silent failures happened this week, and both produced a plausible-looking number:

  * a run wrote every row under the bare "conv-43" namespace instead of its prefixed one,
    landing on top of an older run's 959 anchors in the shared Neo4j. rc=0, accuracy 0.882.
  * a hand-rolled `uvicorn mirix.server.server:app` died with "Attribute app not found",
    the wait loop only retried and never asserted, and 18 questions were answered against
    no memory at all before anyone noticed.

Earlier there was a third: Neo4j went down mid-run, all 1540 questions were answered from an
empty memory, judged, and reported rc=0 at 0.3461.

A score is not evidence that an arm ran. These checks are, and they cost a second.

    python assert_store_sane.py --db mirix_locomo_clean_r1 --prefix clean-r1__ [--graph]
"""
import argparse
import os
import subprocess
import sys


def psql(db, sql):
    r = subprocess.run(["/home/lj/MIRIX_eval/pgenv/bin/psql", "-w", "-h", "localhost",
                        "-U", "mirix", "-d", db, "-At", "-c", sql],
                       capture_output=True, text=True,
                       env={**os.environ, "PGPASSWORD": "mirix"})
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--graph", action="store_true",
                    help="also assert the Neo4j side is populated and prefix-pure")
    ap.add_argument("--min-rows", type=int, default=500)
    a = ap.parse_args()
    bad = []

    n = psql(a.db, f"SELECT count(*) FROM episodic_memory WHERE NOT is_deleted "
                   f"AND user_id LIKE '{a.prefix}%';")
    stray = psql(a.db, f"SELECT count(*) FROM episodic_memory WHERE NOT is_deleted "
                       f"AND user_id NOT LIKE '{a.prefix}%';")
    print(f"postgres: {n} rows under {a.prefix!r}, {stray} outside it")
    if not n.isdigit() or int(n) < a.min_rows:
        bad.append(f"only {n} prefixed episodic rows (want >= {a.min_rows}) — "
                   f"wrong database, wrong prefix, or an ingest that never ran")
    if stray.isdigit() and int(stray) > 0:
        bad.append(f"{stray} episodic rows sit OUTSIDE the prefix in this database — "
                   f"a previous run wrote to the bare namespace")

    # A live server is not the same as a live server pointed at this store.
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8531/health", timeout=5) as r:
            print(f"server: /health {r.status}")
    except Exception as e:  # noqa: BLE001
        bad.append(f"no server on 8531 ({e}) — every question would be answered "
                   f"from an empty memory and still return rc=0")

    if a.graph:
        try:
            from neo4j import GraphDatabase
            d = GraphDatabase.driver(os.environ["MIRIX_NEO4J_URI"],
                                     auth=(os.environ["MIRIX_NEO4J_USER"],
                                           os.environ["MIRIX_NEO4J_PASSWORD"]))
            with d.session() as s:
                mine = s.run("MATCH (a:V7Anchor) WHERE a.user_id STARTS WITH $p "
                             "RETURN count(*) AS c", p=a.prefix).single()["c"]
                orphan = s.run(
                    "MATCH (f:V7Fact) WHERE f.user_id STARTS WITH $p "
                    "AND NOT (f)-[:V7_FACT_ARG]->() RETURN count(*) AS c",
                    p=a.prefix).single()["c"]
            d.close()
            print(f"neo4j: {mine} anchors under {a.prefix!r}, {orphan} facts with no anchor")
            if mine == 0:
                bad.append("zero anchors under this prefix — the graph arm would silently "
                           "degrade to whatever the fallback path returns")
        except Exception as e:  # noqa: BLE001
            bad.append(f"neo4j unreachable ({e})")

    if bad:
        print("\nFAILED:")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
