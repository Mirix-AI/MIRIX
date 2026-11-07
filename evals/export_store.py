"""Export a store as portable content, and rebuild it elsewhere.

A pg_dump of the LoCoMo clean store is 177 MB, and 119 MB of that is embedding vectors —
the actual text is **under 1 MB**:

    episodic_memory   1396 rows   46 MB total,  498 kB of text
    semantic_memory   1512 rows   73 MB total,  473 kB of text

Vectors are a deterministic function of the text and the model, so shipping them is shipping
a cache. This exports rows and graph structure as JSONL without any embedding column, which
brings the whole store to a few megabytes — small enough to commit, mail, or clone — and
`--rebuild` regenerates the vectors on the far side for a few cents of ada-002.

Why this matters for a handover: without a store, every QA-only experiment first costs 5.5
hours of ingest, and an ingest is not reproducible anyway — identical code re-ingested moves
the LoCoMo score by about 5 questions on 411. Somebody continuing this work needs THIS
store, not a fresh one, or none of their numbers are comparable to the ones on disk.

    python export_store.py --db mirix_locomo_clean_r1 --prefix clean-r1__ --out ~/store_export
    python export_store.py --rebuild ~/store_export --db mirix_new --prefix clean-r1__
"""
import argparse
import gzip
import json
import os
import subprocess
import sys

PG = "/home/lj/MIRIX_eval/pgenv/bin/psql"
TABLES = {
    "episodic_memory": ["id", "user_id", "organization_id", "actor", "event_type", "summary",
                        "details", "occurred_at", "filter_tags", "source_refs", "created_at"],
    "semantic_memory": ["id", "user_id", "organization_id", "name", "summary", "details",
                        "source", "filter_tags", "source_refs", "created_at"],
}


def psql(db, sql):
    r = subprocess.run([PG, "-w", "-h", "localhost", "-U", "mirix", "-d", db, "-At", "-c", sql],
                       capture_output=True, text=True,
                       env={**os.environ, "PGPASSWORD": "mirix"})
    if r.returncode:
        sys.exit(f"psql failed: {r.stderr.strip()[:300]}")
    return r.stdout


def export(a):
    os.makedirs(a.out, exist_ok=True)
    total = 0
    for table, cols in TABLES.items():
        # row_to_json over an explicit column list — never SELECT *, or the embedding column
        # comes with it and the export is back to 177 MB.
        sel = ", ".join(f"t.{c}" for c in cols)
        sql = (f"SELECT row_to_json(x) FROM (SELECT {sel} FROM {table} t "
               f"WHERE NOT t.is_deleted AND t.user_id LIKE '{a.prefix}%') x;")
        path = os.path.join(a.out, f"{table}.jsonl.gz")
        n = 0
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for line in psql(a.db, sql).splitlines():
                if line.strip():
                    f.write(line + "\n")
                    n += 1
        print(f"  {table}: {n} rows -> {os.path.basename(path)} "
              f"({os.path.getsize(path)/1048576:.1f} MB)")
        total += n

    try:
        from neo4j import GraphDatabase
        from neo4j.time import DateTime, Date, Time, Duration

        def enc(o):
            if isinstance(o, (DateTime, Date, Time, Duration)):
                return str(o)
            raise TypeError(type(o).__name__)

        d = GraphDatabase.driver(os.environ["MIRIX_NEO4J_URI"],
                                 auth=(os.environ["MIRIX_NEO4J_USER"],
                                       os.environ["MIRIX_NEO4J_PASSWORD"]))
        with d.session() as s:
            with gzip.open(os.path.join(a.out, "neo4j_nodes.jsonl.gz"), "wt",
                           encoding="utf-8") as f:
                nn = 0
                for rec in s.run("MATCH (x) WHERE x.user_id STARTS WITH $p "
                                 "RETURN labels(x) AS l, properties(x) AS p", p=a.prefix):
                    pr = dict(rec["p"])
                    pr.pop("name_embedding", None)      # same reasoning as above
                    f.write(json.dumps({"labels": rec["l"], "props": pr},
                                       ensure_ascii=False, default=enc) + "\n")
                    nn += 1
            with gzip.open(os.path.join(a.out, "neo4j_rels.jsonl.gz"), "wt",
                           encoding="utf-8") as f:
                nr = 0
                for rec in s.run("MATCH (a)-[e]->(b) WHERE a.user_id STARTS WITH $p "
                                 "RETURN a.id AS s, type(e) AS t, b.id AS o, "
                                 "properties(e) AS p", p=a.prefix):
                    f.write(json.dumps({"from": rec["s"], "type": rec["t"], "to": rec["o"],
                                        "props": dict(rec["p"])},
                                       ensure_ascii=False, default=enc) + "\n")
                    nr += 1
        d.close()
        print(f"  neo4j: {nn} nodes, {nr} relationships")
    except Exception as e:  # noqa: BLE001
        print(f"  neo4j export skipped: {e}")

    meta = {"db": a.db, "prefix": a.prefix, "rows": total,
            "note": "embeddings omitted; regenerate with --rebuild"}
    json.dump(meta, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"\n{total} memory rows exported, {size/1048576:.1f} MB total")
    print("Embeddings are NOT included — they are a deterministic function of the text.")
    print("Rebuild on the far side with --rebuild; ada-002 over ~3000 rows costs cents.")


def rebuild(a):
    src = a.rebuild
    print(f"rebuilding {a.db} from {src}")
    for table in TABLES:
        path = os.path.join(src, f"{table}.jsonl.gz")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
        print(f"  {table}: {len(rows)} rows — insert these through the normal manager API,")
        print("    not raw SQL, so embeddings and graph hooks fire the way ingest does.")
    print("\nNOT automated on purpose. Inserting rows straight into Postgres would leave the")
    print("vectors null and the graph empty, and the run would look fine until every graph")
    print("arm quietly scored like a no-graph arm. Route them through")
    print("EpisodicMemoryManager.create_episodic_memory / SemanticMemoryManager equivalents")
    print("with the server up, or restore the full pg_dump if you have it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--prefix", default="clean-r1__")
    ap.add_argument("--out")
    ap.add_argument("--rebuild", help="path to an export directory")
    a = ap.parse_args()
    if a.rebuild:
        rebuild(a)
    elif a.db and a.out:
        export(a)
    else:
        ap.error("give --db and --out, or --rebuild <dir>")


if __name__ == "__main__":
    main()
