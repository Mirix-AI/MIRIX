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
* `load` REPLACES current memory. PostgreSQL truncation/data restore and Neo4j
  graph replacement each use a transaction. They are separate transactions:
  a later Neo4j failure is reported but cannot roll back the committed PG restore.
  Stop concurrent writers while saving or restoring.
* PG connection + Neo4j auth are read from .env in the repo root. Passwords have
  no shared defaults. PostgreSQL tools must be on PATH or in MIRIX_PG_BIN.
* A snapshot is mode-specific: a no-graph snapshot records that no graph was
  captured; a configured graph's export failure aborts the save. A graph
  snapshot has both. Restoring a no-graph snapshot then running graph-mode QA
  will NOT magically produce a graph.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    "password": os.environ.get("MIRIX_PG_PASSWORD", os.environ.get("PGPASSWORD", "")),
    "db": os.environ.get("MIRIX_PG_DB", "mirix"),
}
PG_BIN = os.environ.get("MIRIX_PG_BIN")
NEO4J = {
    "uri": os.environ.get("MIRIX_NEO4J_URI", "bolt://localhost:7687"),
    "user": os.environ.get("MIRIX_NEO4J_USER", "neo4j"),
    "password": os.environ.get("MIRIX_NEO4J_PASSWORD", ""),
    "database": os.environ.get("MIRIX_NEO4J_DATABASE", "neo4j"),
}


def _pg_env():
    return {**os.environ, "PGPASSWORD": PG["password"]}


def _pg_args(tool: str):
    return [
        str(Path(PG_BIN) / tool) if PG_BIN else tool,
        "-h", PG["host"], "-p", PG["port"], "-U", PG["user"], "-d", PG["db"],
    ]


def _table_exists(table: str) -> bool:
    out = _run_checked(
        _pg_args("psql") + ["-tAc", f"SELECT to_regclass('public.{table}');"],
        timeout=30,
    )
    return out.stdout.strip() not in ("", "NULL")


def _run_checked(args, *, timeout):
    result = subprocess.run(
        args, capture_output=True, text=True, env=_pg_env(), timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr.strip()}")
    return result


def _snapshot_path(name: str) -> Path:
    """Snapshot names are single directory names, never arbitrary paths."""
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("Snapshot name must be a single directory name")
    root = SNAP_ROOT.resolve()
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Snapshot path must remain inside the snapshot directory")
    return path


_TYPE_TAG = "__mirix_snapshot_type__"


def _encode_snapshot(value):
    # Neo4j Duration and Point inherit tuple, so json.dumps(default=...) alone
    # would silently serialize them as arrays and lose their database types.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _encode_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not type(value).__module__.startswith("neo4j"):
        return [_encode_snapshot(item) for item in value]
    return _json_default(value)


def _json_default(value):
    """Losslessly encode temporal properties, including Neo4j nanoseconds."""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {_TYPE_TAG: f"python.{type(value).__name__}", "value": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {_TYPE_TAG: "python.timedelta", "value": [value.days, value.seconds, value.microseconds]}
    from neo4j import time as neo_time
    for name in ("DateTime", "Date", "Time", "Duration"):
        if isinstance(value, getattr(neo_time, name)):
            return {_TYPE_TAG: f"neo4j.{name}", "value": value.iso_format()}
    from neo4j.spatial import CartesianPoint, WGS84Point
    for cls in (CartesianPoint, WGS84Point):
        if isinstance(value, cls):
            return {_TYPE_TAG: f"neo4j.{cls.__name__}", "value": list(value)}
    raise TypeError(f"Unsupported snapshot property type: {type(value).__name__}")


def _json_object_hook(value):
    if set(value) != {_TYPE_TAG, "value"}:
        return value
    kind, raw = value[_TYPE_TAG], value["value"]
    native = {"python.datetime": dt.datetime, "python.date": dt.date, "python.time": dt.time}
    if kind in native:
        return native[kind].fromisoformat(raw)
    if kind == "python.timedelta":
        return dt.timedelta(days=raw[0], seconds=raw[1], microseconds=raw[2])
    from neo4j import time as neo_time
    for name in ("DateTime", "Date", "Time", "Duration"):
        if kind == f"neo4j.{name}":
            return getattr(neo_time, name).from_iso_format(raw)
    from neo4j.spatial import CartesianPoint, WGS84Point
    for cls in (CartesianPoint, WGS84Point):
        if kind == f"neo4j.{cls.__name__}":
            return cls(raw)
    raise ValueError(f"Unknown snapshot property type: {kind}")


def _validate_graph(graph: dict) -> None:
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(
        graph.get("relationships"), list
    ):
        raise ValueError("Invalid graph snapshot")
    ids = set()
    for node in graph["nodes"]:
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("id"), (int, str))
            or node["id"] in ids
            or not isinstance(node.get("labels"), list)
            or not all(isinstance(label, str) and label for label in node["labels"])
            or not isinstance(node.get("props"), dict)
        ):
            raise ValueError("Invalid graph node")
        ids.add(node["id"])
    for rel in graph["relationships"]:
        if (
            not isinstance(rel, dict)
            or rel.get("src") not in ids or rel.get("dst") not in ids
            or not isinstance(rel.get("type"), str) or not rel["type"]
            or not isinstance(rel.get("props"), dict)
        ):
            raise ValueError("Invalid graph relationship")


def _graph_configured() -> bool:
    return os.environ.get("MIRIX_ENABLE_GRAPH_MEMORY", "").lower() in {"true", "1"} or bool(NEO4J["password"])


# ---------------------------------------------------------------- save ----
def save(name: str, include_agents: bool) -> None:
    snap = _snapshot_path(name)
    if snap.exists():
        raise ValueError(f"Snapshot '{name}' already exists; choose a new name")
    snap.parent.mkdir(parents=True, exist_ok=True)

    tables = [t for t in MEMORY_TABLES if _table_exists(t)]
    if include_agents:
        tables += [t for t in AGENT_TABLES if _table_exists(t)]
    if not tables:
        raise SystemExit("No memory tables found in the database.")

    table_flags = []
    for t in tables:
        table_flags += ["-t", t]
    # Publish only a complete snapshot. A failed export never overwrites an
    # earlier good snapshot or leaves metadata advertising an incomplete one.
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=snap.parent) as directory:
        staging = Path(directory)
        dump_path = staging / "pg_memory.dump"
        print(f"[snapshot] pg_dump {len(tables)} table(s)")
        _run_checked(
            _pg_args("pg_dump") + ["-Fc", "--data-only", "-f", str(dump_path)] + table_flags,
            timeout=600,
        )
        graph_captured = _graph_configured()
        graph = _neo4j_export() if graph_captured else {"nodes": [], "relationships": []}
        _validate_graph(graph)
        (staging / "neo4j_graph.json").write_text(
            json.dumps(_encode_snapshot(graph), ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "format_version": 2,
            "name": name,
            "pg_tables": tables,
            "includes_agents": include_agents,
            "neo4j_captured": graph_captured,
            "neo4j_nodes": len(graph["nodes"]),
            "neo4j_relationships": len(graph["relationships"]),
        }
        (staging / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        staging.rename(snap)
    print(f"[snapshot] saved '{name}' -> {snap}")


def _neo4j_export() -> dict:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))
    try:
        with driver.session(database=NEO4J["database"]) as ses:
            def export(tx):
                nodes = [
                    {"id": rec["id"], "labels": rec["labels"], "props": dict(rec["props"])}
                    for rec in tx.run(
                        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
                    )
                ]
                rels = [
                    {"src": rec["src"], "dst": rec["dst"], "type": rec["type"], "props": dict(rec["props"])}
                    for rec in tx.run(
                        "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, "
                        "type(r) AS type, properties(r) AS props"
                    )
                ]
                return {"nodes": nodes, "relationships": rels}
            return ses.execute_read(export)
    finally:
        driver.close()


# ---------------------------------------------------------------- load ----
def load(name: str) -> None:
    snap = _snapshot_path(name)
    if not snap.exists():
        raise SystemExit(f"Snapshot '{name}' not found at {snap}")
    meta = json.loads((snap / "meta.json").read_text())
    tables = meta["pg_tables"]
    if (
        not isinstance(tables, list) or not tables
        or not all(isinstance(table, str) and table in MEMORY_TABLES + AGENT_TABLES for table in tables)
        or len(set(tables)) != len(tables)
    ):
        raise ValueError("Snapshot contains invalid table names")
    dump_path = snap / "pg_memory.dump"
    if not dump_path.is_file():
        raise FileNotFoundError(dump_path)
    # Parse and validate every input before touching either live store.
    graph = json.loads(
        (snap / "neo4j_graph.json").read_text(encoding="utf-8"), object_hook=_json_object_hook
    )
    _validate_graph(graph)

    # Render the archive first, then execute TRUNCATE + restore in ONE PG
    # transaction. ON_ERROR_STOP causes any SQL failure to roll it all back.
    with tempfile.TemporaryDirectory(prefix="mirix-restore-") as directory:
        sql_path = Path(directory) / "data.sql"
        pg_restore = str(Path(PG_BIN) / "pg_restore") if PG_BIN else "pg_restore"
        _run_checked(
            [pg_restore, "--data-only", "--disable-triggers", "-f", str(sql_path), str(dump_path)],
            timeout=600,
        )
        restore_path = Path(directory) / "restore.sql"
        with restore_path.open("wb") as target, sql_path.open("rb") as source:
            names = ", ".join(f'public."{table}"' for table in tables)
            target.write(f"TRUNCATE {names} CASCADE;\n".encode())
            shutil.copyfileobj(source, target)
        print(f"[snapshot] restoring {len(tables)} table(s) in one transaction")
        _run_checked(
            _pg_args("psql") + ["-X", "--set=ON_ERROR_STOP=1", "--single-transaction", "-f", str(restore_path)],
            timeout=600,
        )
    _invalidate_redis_cache(tables)
    # Legacy snapshots always captured a graph (including an empty one).
    # A deliberately graph-free snapshot also clears a configured graph target.
    if meta.get("neo4j_captured", True) or graph["nodes"] or _graph_configured():
        _neo4j_import(graph)
    print(f"[snapshot] loaded '{name}'")


def _invalidate_redis_cache(tables: list[str]) -> None:
    """Restored IDs may have older content than Redis; clear only MIRIX caches."""
    if os.environ.get("MIRIX_REDIS_ENABLED", "").lower() not in {"true", "1"}:
        return
    from redis import Redis
    from mirix.database.redis_client import RedisMemoryClient as Cache

    prefixes = {
        "episodic_memory": Cache.EPISODIC_PREFIX, "semantic_memory": Cache.SEMANTIC_PREFIX,
        "procedural_memory": Cache.PROCEDURAL_PREFIX, "resource_memory": Cache.RESOURCE_PREFIX,
        "knowledge_vault": Cache.KNOWLEDGE_PREFIX, "raw_memory": Cache.RAW_MEMORY_PREFIX,
        "block": Cache.BLOCK_PREFIX, "agents": Cache.AGENT_PREFIX, "messages": Cache.MESSAGE_PREFIX,
    }
    uri = os.environ.get("MIRIX_REDIS_URI")
    client = Redis.from_url(uri, socket_timeout=10) if uri else Redis(
        host=os.environ.get("MIRIX_REDIS_HOST", "localhost"),
        port=int(os.environ.get("MIRIX_REDIS_PORT", "6379")),
        db=int(os.environ.get("MIRIX_REDIS_DB", "0")),
        password=os.environ.get("MIRIX_REDIS_PASSWORD") or None, socket_timeout=10,
    )
    try:
        for table in tables:
            batch = []
            for key in client.scan_iter(match=prefixes[table] + "*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    client.delete(*batch)
                    batch.clear()
            if batch:
                client.delete(*batch)
    finally:
        client.close()


def _neo4j_import(graph: dict) -> None:
    _validate_graph(graph)
    from neo4j import GraphDatabase

    def identifier(value):
        return "`" + value.replace("`", "``") + "`"

    def restore(tx):
        # Even an empty snapshot must remove the previous graph.
        tx.run("MATCH (n) DETACH DELETE n").consume()
        ids = {}
        for node in graph["nodes"]:
            labels = "".join(":" + identifier(label) for label in node["labels"])
            record = tx.run(
                f"CREATE (x{labels}) SET x = $props RETURN elementId(x) AS id",
                props=node["props"],
            ).single(strict=True)
            ids[node["id"]] = record["id"]
        for rel in graph["relationships"]:
            tx.run(
                "MATCH (a), (b) WHERE elementId(a) = $src AND elementId(b) = $dst "
                f"CREATE (a)-[r:{identifier(rel['type'])}]->(b) SET r = $props",
                src=ids[rel["src"]], dst=ids[rel["dst"]], props=rel["props"],
            ).consume()

    driver = GraphDatabase.driver(NEO4J["uri"], auth=(NEO4J["user"], NEO4J["password"]))
    try:
        with driver.session(database=NEO4J["database"]) as ses:
            ses.execute_write(restore)
    finally:
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
    snap = _snapshot_path(name)
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
