"""Save / load Mirix memory snapshots so an expensive ingest can be reused.

Ingesting one LongMemEval-S conversation takes ~2h (no-graph) to ~9h (graph).
This script captures the post-ingest state — PostgreSQL flat-memory tables and
the Neo4j graph — into a named snapshot, and restores it later so a QA-only
re-run can skip ingest entirely.

Usage:
    python memory_snapshot.py save <name> [--agents]
    python memory_snapshot.py load <name>
    python memory_snapshot.py list
    python memory_snapshot.py delete <name>

A snapshot lives in evals/snapshots/<name>/ and contains:
    pg_memory.dump   — pg_dump of the six memory tables (+ agents/messages
                       when --agents is given, needed for a true QA-only run)
    neo4j_graph.json — every node + relationship, exported via Cypher

Notes
-----
* `load` REPLACES current memory: it truncates the same tables / wipes the
  Neo4j graph before restoring, so the snapshot is the exact state afterwards.
* PG connection + Neo4j auth are read from the same env / settings the server
  uses (.env in repo root). Defaults match docker/env.example.
* A snapshot is mode-specific: a no-graph snapshot has an empty graph, a graph
  snapshot has both. Restoring a no-graph snapshot then running graph-mode QA
  will NOT magically produce a graph.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Load .env the same way the server does.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

SNAP_ROOT = Path(__file__).resolve().parent / "snapshots"

# Mirix memory tables. Core memory lives in `block`; `raw_memory` holds the
# verbatim ingested chunks. agents/messages are optional (only needed for a
# fully ingest-free QA run that also skips meta-agent recreation).
# Names verified against the live schema (pg_tables).
MEMORY_TABLES = [
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    "resource_memory",
    "knowledge_vault",
    "block",
    "raw_memory",
]
AGENT_TABLES = ["agents", "messages"]

# --- PG / Neo4j connection (env first, then docker/env.example defaults) ---
PG = {
    "host": os.environ.get("MIRIX_PG_HOST", "localhost"),
    "port": os.environ.get("MIRIX_PG_PORT", "5432"),
    "user": os.environ.get("MIRIX_PG_USER", "mirix"),
    "password": os.environ.get("MIRIX_PG_PASSWORD", "mirix"),
    "db": os.environ.get("MIRIX_PG_DB", "mirix"),
}
PG_BIN = os.environ.get("MIRIX_PG_BIN", "/usr/local/opt/postgresql@17/bin")
NEO4J = {
    "uri": os.environ.get("MIRIX_NEO4J_URI", "bolt://localhost:7687"),
    "user": os.environ.get("MIRIX_NEO4J_USER", "neo4j"),
    "password": os.environ.get("MIRIX_NEO4J_PASSWORD", "mirix_neo4j_dev"),
    "database": os.environ.get("MIRIX_NEO4J_DATABASE", "neo4j"),
}


def _pg_env():
    return {**os.environ, "PGPASSWORD": PG["password"]}


def _pg_args(tool: str):
    return [
        f"{PG_BIN}/{tool}",
        "-h", PG["host"], "-p", PG["port"], "-U", PG["user"], "-d", PG["db"],
    ]


def _table_exists(table: str) -> bool:
    out = subprocess.run(
        _pg_args("psql") + ["-tAc", f"SELECT to_regclass('public.{table}');"],
        capture_output=True, text=True, env=_pg_env(), timeout=30,
    )
    return out.stdout.strip() not in ("", "NULL")


# ---------------------------------------------------------------- save ----
def save(name: str, include_agents: bool) -> None:
    snap = SNAP_ROOT / name
    snap.mkdir(parents=True, exist_ok=True)

    tables = [t for t in MEMORY_TABLES if _table_exists(t)]
    if include_agents:
        tables += [t for t in AGENT_TABLES if _table_exists(t)]
    if not tables:
        raise SystemExit("No memory tables found in the database.")

    # --- PostgreSQL: pg_dump the selected tables ---
    dump_path = snap / "pg_memory.dump"
    table_flags = []
    for t in tables:
        table_flags += ["-t", t]
    print(f"[snapshot] pg_dump {len(tables)} table(s) -> {dump_path}")
    res = subprocess.run(
        _pg_args("pg_dump") + ["-Fc", "--data-only", "-f", str(dump_path)] + table_flags,
        capture_output=True, text=True, env=_pg_env(), timeout=600,
    )
    if res.returncode != 0:
        raise SystemExit(f"pg_dump failed: {res.stderr}")

    # --- Neo4j: export all nodes + relationships via Cypher ---
    graph = _neo4j_export()
    graph_path = snap / "neo4j_graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False))
    print(f"[snapshot] neo4j export: {len(graph['nodes'])} nodes, "
          f"{len(graph['relationships'])} relationships -> {graph_path}")

    meta = {
        "name": name,
        "pg_tables": tables,
        "includes_agents": include_agents,
        "neo4j_nodes": len(graph["nodes"]),
        "neo4j_relationships": len(graph["relationships"]),
    }
    (snap / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[snapshot] saved '{name}' -> {snap}")


def _neo4j_export() -> dict:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[snapshot] neo4j driver not installed; graph export skipped")
        return {"nodes": [], "relationships": []}
    try:
        driver = GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))
        nodes, rels = [], []
        with driver.session(database=NEO4J["database"]) as ses:
            for rec in ses.run(
                "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props"
            ):
                nodes.append({"id": rec["id"], "labels": rec["labels"], "props": dict(rec["props"])})
            for rec in ses.run(
                "MATCH (a)-[r]->(b) RETURN id(a) AS src, id(b) AS dst, "
                "type(r) AS type, properties(r) AS props"
            ):
                rels.append({"src": rec["src"], "dst": rec["dst"],
                             "type": rec["type"], "props": dict(rec["props"])})
        driver.close()
        return {"nodes": nodes, "relationships": rels}
    except Exception as exc:
        print(f"[snapshot] neo4j export failed ({exc}); graph snapshot empty")
        return {"nodes": [], "relationships": []}


# ---------------------------------------------------------------- load ----
def load(name: str) -> None:
    snap = SNAP_ROOT / name
    if not snap.exists():
        raise SystemExit(f"Snapshot '{name}' not found at {snap}")
    meta = json.loads((snap / "meta.json").read_text())
    tables = meta["pg_tables"]

    # --- PostgreSQL: truncate the snapshot's tables, then pg_restore data ---
    print(f"[snapshot] truncating {len(tables)} table(s) before restore")
    subprocess.run(
        _pg_args("psql") + ["-c", f"TRUNCATE {', '.join(tables)} CASCADE;"],
        capture_output=True, text=True, env=_pg_env(), timeout=120, check=True,
    )
    print(f"[snapshot] pg_restore <- {snap / 'pg_memory.dump'}")
    res = subprocess.run(
        _pg_args("pg_restore") + ["--data-only", "--disable-triggers",
                                  "-d", PG["db"], str(snap / "pg_memory.dump")],
        capture_output=True, text=True, env=_pg_env(), timeout=600,
    )
    # pg_restore returns non-zero on benign warnings; surface stderr but continue
    if res.returncode != 0 and res.stderr.strip():
        print(f"[snapshot] pg_restore warnings:\n{res.stderr.strip()[:500]}")

    # --- Neo4j: wipe graph, recreate nodes + relationships ---
    _neo4j_import(json.loads((snap / "neo4j_graph.json").read_text()))
    print(f"[snapshot] loaded '{name}'")


def _neo4j_import(graph: dict) -> None:
    if not graph["nodes"]:
        return
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[snapshot] neo4j driver not installed; graph restore skipped")
        return
    driver = GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))
    with driver.session(database=NEO4J["database"]) as ses:
        ses.run("MATCH (n) DETACH DELETE n")
        # Recreate nodes; keep the original id() under _snap_id to rewire edges.
        for n in graph["nodes"]:
            labels = ":".join(n["labels"]) if n["labels"] else "Node"
            ses.run(
                f"CREATE (x:{labels}) SET x = $props, x._snap_id = $sid",
                props=n["props"], sid=n["id"],
            )
        for r in graph["relationships"]:
            ses.run(
                f"MATCH (a {{_snap_id: $src}}), (b {{_snap_id: $dst}}) "
                f"CREATE (a)-[r:{r['type']}]->(b) SET r = $props",
                src=r["src"], dst=r["dst"], props=r["props"],
            )
        ses.run("MATCH (n) REMOVE n._snap_id")
    driver.close()
    print(f"[snapshot] neo4j restored: {len(graph['nodes'])} nodes, "
          f"{len(graph['relationships'])} relationships")


# --------------------------------------------------------------- list/rm --
def list_snapshots() -> None:
    if not SNAP_ROOT.exists():
        print("(no snapshots)")
        return
    for snap in sorted(SNAP_ROOT.iterdir()):
        meta_file = snap / "meta.json"
        if meta_file.exists():
            m = json.loads(meta_file.read_text())
            print(f"  {m['name']:30s} pg_tables={len(m['pg_tables'])} "
                  f"graph={m['neo4j_nodes']}n/{m['neo4j_relationships']}r "
                  f"agents={'yes' if m.get('includes_agents') else 'no'}")


def delete(name: str) -> None:
    import shutil
    snap = SNAP_ROOT / name
    if snap.exists():
        shutil.rmtree(snap)
        print(f"[snapshot] deleted '{name}'")
    else:
        print(f"[snapshot] '{name}' not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save/load Mirix memory snapshots.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save", help="Snapshot current memory state.")
    p_save.add_argument("name")
    p_save.add_argument("--agents", action="store_true",
                        help="Also snapshot agents/messages (for fully ingest-free QA).")

    p_load = sub.add_parser("load", help="Restore a snapshot (REPLACES current memory).")
    p_load.add_argument("name")

    sub.add_parser("list", help="List snapshots.")

    p_del = sub.add_parser("delete", help="Delete a snapshot.")
    p_del.add_argument("name")

    args = parser.parse_args()
    if args.cmd == "save":
        save(args.name, include_agents=args.agents)
    elif args.cmd == "load":
        load(args.name)
    elif args.cmd == "list":
        list_snapshots()
    elif args.cmd == "delete":
        delete(args.name)


if __name__ == "__main__":
    main()
