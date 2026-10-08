"""What a real MIRIX cell holds in its database, table by table (for "the test phase
wrote nothing", H1.6 / H10.4, checked below MIRIX's API).

:func:`database_digest` connects to one cell attempt's database on the compose Postgres
(``infra/mirix_server.PostgresAdmin``) and returns, for every table of its ``public``
schema, the row count and an md5 over the rows' text. Two digests are equal exactly when
no row was added, changed or removed. The retrieval log (M7.4) is left out: a retrieval
writes its row by design, and that is the only write a frozen test phase may make. So is
one column MIRIX's server rewrites whenever it starts: it upserts its base tools
(``mirix/server/server.py``, ``tool_manager.upsert_base_tools``), which bumps
``tools.updated_at``; the tools are MIRIX's own code, not memory, and a resumed test
phase starts the server again (:data:`REWRITTEN_AT_STARTUP`).
"""

from __future__ import annotations

import json
from typing import Any

from evals.skill_bench.infra.mirix_server import PostgresAdmin
from evals.skill_bench.infra.mirix_server import (
    database_digest as infra_database_digest,
)

#: Tables a frozen test phase writes by design: one retrieval log row per retrieval.
WRITTEN_BY_RETRIEVAL = frozenset({"retrieval_log"})
#: Columns every MIRIX server start rewrites (the base tools' upsert), per table.
REWRITTEN_AT_STARTUP: dict[str, frozenset[str]] = {"tools": frozenset({"updated_at"})}


async def database_digest(
    db_name: str, pg: PostgresAdmin | None = None
) -> dict[str, tuple[int, str]]:
    """``{table: (rows, md5 of the sorted row texts)}`` for every table but the
    retrieval log, without the columns a server start rewrites
    (``infra/mirix_server.database_digest``, which learning checkpoints use whole)."""
    digest = await infra_database_digest(
        pg or PostgresAdmin(),
        db_name,
        include_retrieval_log=False,
        drop_columns=REWRITTEN_AT_STARTUP,
    )
    return {
        table: (int(rows), str(md5))
        for table, (rows, md5) in digest.items()
        if table not in WRITTEN_BY_RETRIEVAL
    }


async def retrieval_log_rows(db_name: str, pg: PostgresAdmin | None = None) -> int:
    """How many retrieval log rows the cell database holds."""
    import asyncpg

    pg = pg or PostgresAdmin()
    conn: Any = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=db_name,
        timeout=30,
    )
    try:
        return int(await conn.fetchval("SELECT count(*) FROM retrieval_log"))
    finally:
        await conn.close()


async def _fetch(
    db_name: str, query: str, pg: PostgresAdmin | None = None, *args: Any
) -> list[Any]:
    import asyncpg

    pg = pg or PostgresAdmin()
    conn: Any = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=db_name,
        timeout=30,
    )
    try:
        return list(await conn.fetch(query, *args))
    finally:
        await conn.close()


async def session_agent_contexts(
    db_name: str, pg: PostgresAdmin | None = None
) -> dict[str, str | None]:
    """``{session_id: agent_context}`` of every conversation session MIRIX stored
    (the agent context a client attached to the session)."""
    rows = await _fetch(
        db_name, "SELECT session_id, agent_context FROM conversation_session", pg
    )
    return {str(r["session_id"]): r["agent_context"] for r in rows}


async def skill_feedback_rows(
    db_name: str, pg: PostgresAdmin | None = None
) -> list[dict[str, Any]]:
    """MIRIX's per-skill feedback rows (one per distilled session and skill
    served in it), oldest first: the skill, the stored values, where their quotes were
    found (``*_source``) and the values stated but not verified (``stated_*``)."""
    rows = await _fetch(
        db_name,
        "SELECT session_id, skill_id, skill_name, skill_version_no, served_count, "
        "session_outcome, followed, followed_source, stated_followed, effect, "
        "effect_source, stated_effect FROM skill_session_feedback "
        "ORDER BY created_at, id",
        pg,
    )
    return [dict(r) for r in rows]


async def experience_rows(
    db_name: str, ids: list[str], pg: PostgresAdmin | None = None
) -> dict[str, dict[str, Any]]:
    """``{id: row}`` of MIRIX's distilled experiences ``ids``: what the curator did
    with each (``disposition``), the skills a counted citation recorded
    (``cited_skill_ids``, B12: none for a citation without a verified quote) and
    whether it left the experience pool (``pool_retire_reason``)."""
    rows = await _fetch(
        db_name,
        "SELECT id, status, disposition, cited_skill_ids, pool_retire_reason "
        "FROM skill_experience WHERE id = ANY($1::text[])",
        pg,
        list(ids),
    )
    found = {str(r["id"]): dict(r) for r in rows}
    for row in found.values():  # a JSON column reads back as its text
        if isinstance(row["cited_skill_ids"], str):
            row["cited_skill_ids"] = json.loads(row["cited_skill_ids"])
    return found
