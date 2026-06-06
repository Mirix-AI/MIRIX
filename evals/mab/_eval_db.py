"""Direct-PG helpers for eval scripts.

Two helpers shared by ruler_eval / longmem_eval / main_eval:

- ``measure_memory_size(sample_id)`` — total stored chars+tokens for a
  user, plus per-table row counts; reads ``MIRIX_PG_DB`` env so it tracks
  whatever DB the runner set (e.g. ``mirix_isolate``) instead of being
  pinned to ``mirix``.

- ``dump_memories(sample_id)`` — full per-user dump of episodic +
  semantic rows, straight from Postgres. Bypasses
  ``/memory/components``, which caps each memory type at 50 by default
  and 200 even with an explicit ``limit`` parameter — so it's the only
  way to get a ``_memories.json`` snapshot whose token count reflects
  the real store on conversations with thousands of rows.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List


def _pg_conn() -> Dict[str, str]:
    return {
        "bin":  shutil.which("psql") or "/usr/local/opt/postgresql@17/bin/psql",
        "host": os.environ.get("MIRIX_PG_HOST", "localhost"),
        "user": os.environ.get("MIRIX_PG_USER", "mirix"),
        "db":   os.environ.get("MIRIX_PG_DB",   "mirix"),
        "pwd":  os.environ.get("MIRIX_PG_PASSWORD", "mirix"),
    }


def _pg_scalar(sql: str, timeout: int = 30) -> str:
    c = _pg_conn()
    try:
        out = subprocess.run(
            [c["bin"], "-h", c["host"], "-U", c["user"], "-d", c["db"], "-tAc", sql],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PGPASSWORD": c["pwd"]},
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _pg_rows(sql: str, timeout: int = 300) -> List[List[str]]:
    """Run SQL with control-byte separators so embedded newlines in
    summary/details don't get mistaken for row boundaries.

    SOH (``\\x01``) separates columns, STX (``\\x02``) separates records.
    Neither appears in normal memory text.
    """
    c = _pg_conn()
    out = subprocess.run(
        [c["bin"], "-h", c["host"], "-U", c["user"], "-d", c["db"],
         "-A", "-F", "\x01", "-R", "\x02", "-t", "-c", sql],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PGPASSWORD": c["pwd"]},
    )
    rows: List[List[str]] = []
    for chunk in out.stdout.split("\x02"):
        if not chunk:
            continue
        rows.append(chunk.split("\x01"))
    return rows


def _esc(s: str) -> str:
    return s.replace("'", "''")


def measure_memory_size(sample_id: str) -> Dict:
    """Total stored chars + tokens for ``sample_id``.

    PG (flat) and Neo4j (graph) backends measured on the same yardstick
    by concatenating every stored text field.
    """
    stats: Dict = {"unit": "chars+tokens"}
    uid = _esc(sample_id)

    flat: Dict = {}
    flat_chars = 0
    for table, cols in (
        ("episodic_memory", "coalesce(summary,'')||coalesce(details,'')"),
        ("semantic_memory", "coalesce(name,'')||coalesce(summary,'')||coalesce(details,'')"),
    ):
        row_n = _pg_scalar(f"SELECT count(*) FROM {table} WHERE user_id='{uid}';")
        chars = _pg_scalar(
            f"SELECT coalesce(sum(length({cols})),0) FROM {table} WHERE user_id='{uid}';"
        )
        n = int(row_n) if row_n.isdigit() else 0
        c = int(chars) if chars.lstrip('-').isdigit() else 0
        flat[table] = {"rows": n, "chars": c}
        flat_chars += c
    stats["flat"] = flat
    stats["flat_total_chars"] = flat_chars

    try:
        from neo4j import GraphDatabase
        from mirix.settings import settings
        driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        with driver.session(database=settings.neo4j_database) as ses:
            rec = ses.run(
                """
                MATCH (n) WHERE n.user_id=$uid
                WITH count(n) AS nodes,
                     sum(size(coalesce(n.description,'')) + size(coalesce(n.summary,''))) AS node_chars
                RETURN nodes, node_chars
                """, uid=sample_id
            ).single()
            erec = ses.run(
                """
                MATCH (a)-[r]->(b) WHERE a.user_id=$uid
                RETURN count(r) AS edges,
                       sum(size(coalesce(r.description,''))) AS edge_chars
                """, uid=sample_id
            ).single()
        driver.close()
        node_chars = (rec["node_chars"] or 0) if rec else 0
        edge_chars = (erec["edge_chars"] or 0) if erec else 0
        stats["graph"] = {
            "nodes": (rec["nodes"] or 0) if rec else 0,
            "edges": (erec["edges"] or 0) if erec else 0,
            "node_chars": node_chars,
            "edge_chars": edge_chars,
        }
        stats["graph_total_chars"] = node_chars + edge_chars
    except Exception as exc:
        stats["graph"] = {"error": str(exc)}
        stats["graph_total_chars"] = 0

    total_chars = max(stats["flat_total_chars"], stats["graph_total_chars"])
    stats["total_chars"] = total_chars
    stats["total_tokens"] = total_chars // 4
    return stats


def dump_memories(sample_id: str) -> Dict:
    """Full per-user dump of episodic + semantic rows for ``sample_id``.

    Shape matches what ``organize_results.count_memory_tokens`` expects:
    ``{"memories": {"episodic": {"total_count": N, "items": [...]},
                    "semantic": {...}}}``.
    """
    uid = _esc(sample_id)
    epi_rows = _pg_rows(
        "SELECT id, "
        "coalesce(to_char(occurred_at,'YYYY-MM-DD\"T\"HH24:MI:SSOF'),''), "
        "coalesce(event_type,''), coalesce(actor,''), "
        "coalesce(summary,''), coalesce(details,'') "
        f"FROM episodic_memory WHERE user_id='{uid}' "
        "ORDER BY occurred_at NULLS LAST, created_at"
    )
    sem_rows = _pg_rows(
        "SELECT id, coalesce(name,''), coalesce(summary,''), "
        "coalesce(details,''), coalesce(source,'') "
        f"FROM semantic_memory WHERE user_id='{uid}' ORDER BY created_at"
    )
    return {
        "user_id": sample_id,
        "memories": {
            "episodic": {
                "total_count": len(epi_rows),
                "items": [
                    {"id": r[0], "occurred_at": r[1], "event_type": r[2],
                     "actor": r[3], "summary": r[4], "details": r[5]}
                    for r in epi_rows
                ],
            },
            "semantic": {
                "total_count": len(sem_rows),
                "items": [
                    {"id": r[0], "name": r[1], "summary": r[2],
                     "details": r[3], "source": r[4]}
                    for r in sem_rows
                ],
            },
        },
    }
