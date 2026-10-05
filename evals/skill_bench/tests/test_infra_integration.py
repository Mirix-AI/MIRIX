"""Real Postgres + real MIRIX server isolation. No model is ever called:
the checks use MIRIX endpoints that only touch the database (organizations, clients).

Requires the skill_bench compose Postgres::

    docker compose -f evals/skill_bench/infra/compose.yaml up -d
    pytest -m integration evals/skill_bench/tests/test_infra_integration.py
"""

from __future__ import annotations

import asyncio
import re
import secrets
import socket
from pathlib import Path

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.infra import mirix_server as ms
from evals.skill_bench.infra.processes import find_free_port
from evals.skill_bench.methods.base import ITEM_HEADER

pytestmark = pytest.mark.integration


@pytest.fixture
def pg() -> ms.PostgresAdmin:
    admin = ms.PostgresAdmin()
    try:
        with socket.create_connection((admin.host, admin.port), timeout=1):
            pass
    except OSError:
        pytest.skip(f"skill_bench Postgres not reachable on {admin.host}:{admin.port}")
    return admin


async def _connect(pg: ms.PostgresAdmin, database: str):
    import asyncpg

    return await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=database,
    )


def _snapshot(directory: Path) -> dict[str, float]:
    if not directory.exists():
        return {}
    return {
        str(p.relative_to(directory)): p.stat().st_mtime for p in directory.rglob("*")
    }


@pytest.mark.asyncio
async def test_cell_databases_are_isolated_template_clones(pg):
    suffix = secrets.token_hex(3)
    names = [ms.cell_db_name(f"it-{i}-{suffix}", 0) for i in range(6)]
    try:
        # Concurrent creation from one template is serialised by the advisory lock.
        created = await asyncio.gather(
            *(ms.ensure_cell_database(pg, n, must_be_new=True) for n in names)
        )
        assert created == [True] * len(names)
        with pytest.raises(ms.MirixServerError):
            await ms.ensure_cell_database(pg, names[0], must_be_new=True)
        assert await ms.ensure_cell_database(pg, names[0]) is False
        a, b = await _connect(pg, names[0]), await _connect(pg, names[1])
        try:
            extensions = {
                r["extname"] for r in await a.fetch("SELECT extname FROM pg_extension")
            }
            assert {"vector", "pg_trgm", "uuid-ossp"} <= extensions
            await a.execute("CREATE TABLE only_in_a (id int)")
            assert await b.fetchval("SELECT to_regclass('only_in_a')") is None
        finally:
            await a.close()
            await b.close()
    finally:
        for name in names:
            await ms.drop_cell_database(pg, name)


@pytest.mark.asyncio
async def test_real_mirix_server_runs_in_its_own_home_and_database(pg, tmp_path):
    cell = ms.CellServerSpec(
        f"it-server-{secrets.token_hex(3)}",
        0,
        tmp_path / "cell",
        find_free_port(),
        pg=pg,
    )
    real_mirix = Path.home() / ".mirix"
    before = _snapshot(real_mirix)
    server = ms.MirixServer(cell, mode="fresh", startup_timeout_s=300)
    try:
        await server.start()
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            health = (await client.get(server.base_url + "/health")).json()
        assert health["status"] == "healthy"
        conn = await _connect(pg, cell.db_name)
        try:
            tables = {
                r["tablename"]
                for r in await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
            }
        finally:
            await conn.close()
        assert {"organizations", "users"} <= tables  # MIRIX used the cell database
        await server.ensure_alive()
    finally:
        await server.stop()
        await ms.drop_cell_database(pg, cell.db_name)
    assert (cell.home / ".mirix").is_dir()
    assert _snapshot(real_mirix) == before
    assert cell.log_path.stat().st_size > 0


async def _table_rows(pg: ms.PostgresAdmin, database: str, query: str) -> set[str]:
    conn = await _connect(pg, database)
    try:
        return {r[0] for r in await conn.fetch(query)}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_two_real_mirix_cells_running_at_once_cannot_see_each_other(pg, tmp_path):
    """Plan 9.4 (W22 / W38): two live MIRIX servers, each only in its own database and HOME."""
    suffix = secrets.token_hex(3)
    cells = [
        ms.CellServerSpec(
            f"it-pair-{name}-{suffix}", 0, tmp_path / name, find_free_port(), pg=pg
        )
        for name in ("left", "right")
    ]
    assert cells[0].port != cells[1].port
    real_mirix = Path.home() / ".mirix"
    before = _snapshot(real_mirix)
    servers = [
        ms.MirixServer(cell, mode="fresh", startup_timeout_s=300) for cell in cells
    ]
    orgs = [f"org-{name}-{suffix}" for name in ("left", "right")]
    clients = [f"client-{name}-{suffix}" for name in ("left", "right")]
    try:
        await asyncio.gather(*(server.start() for server in servers))
        async with httpx.AsyncClient(trust_env=False, timeout=60) as http:

            async def write(server: ms.MirixServer, org: str, client: str) -> None:
                created = await http.post(
                    server.base_url + "/organizations/create_or_get",
                    json={"org_id": org, "name": org},
                )
                assert created.status_code in (200, 201), created.text
                made = await http.post(
                    server.base_url + "/clients/create_or_get",
                    json={"client_id": client, "name": client, "org_id": org},
                )
                assert made.status_code in (200, 201), made.text

            # Both servers write at the same time.
            await asyncio.gather(
                *(write(s, o, c) for s, o, c in zip(servers, orgs, clients))
            )
            for i, server in enumerate(servers):
                listed = await http.get(
                    server.base_url + "/organizations", params={"limit": 1000}
                )
                seen = {org["id"] for org in listed.json()}
                assert orgs[i] in seen and orgs[1 - i] not in seen
                other = await http.get(
                    server.base_url + f"/organizations/{orgs[1 - i]}"
                )
                assert other.status_code != 200 or other.json()["id"] != orgs[1 - i]
        for i, cell in enumerate(cells):
            org_rows = await _table_rows(
                pg, cell.db_name, "SELECT id FROM organizations"
            )
            client_rows = await _table_rows(pg, cell.db_name, "SELECT id FROM clients")
            assert orgs[i] in org_rows and orgs[1 - i] not in org_rows
            assert clients[i] in client_rows and clients[1 - i] not in client_rows
        for server in servers:
            await server.ensure_alive()
    finally:
        await asyncio.gather(*(server.stop() for server in servers))
        for cell in cells:
            await ms.drop_cell_database(pg, cell.db_name)
    for cell in cells:
        assert (cell.home / ".mirix").is_dir()  # each server wrote its own ~/.mirix
        assert cell.log_path.stat().st_size > 0
    assert cells[0].home != cells[1].home
    assert _snapshot(real_mirix) == before  # the harness HOME's ~/.mirix is untouched


@pytest.mark.asyncio
async def test_contract_probe_reads_the_real_mirix_settings(pg, tmp_path):
    """W9: the probe imports this checkout's mirix.settings under the cell's environment.

    This MIRIX reads MIRIX_LLM_MAX_RETRIES and
    MIRIX_LLM_TIMEOUT_SECONDS, so the real server starts and records what it checked. A
    MIRIX that did not would be refused before any database or HOME is created
    (``test_mirix_server.test_start_refuses_a_mirix_that_would_retry_on_its_own``).

    The cell is built as run.py builds it, with the B32 session header setting
    (``MIRIX_LLM_SESSION_HEADER`` = the gateway's item header): the probe records, for
    the record only, the header this MIRIX reads from it (``llm_session_header``, never
    a contract check: a MIRIX without the setting just leaves the ledger's item null).
    """
    cell = ms.CellServerSpec(
        f"it-probe-{secrets.token_hex(3)}",
        0,
        tmp_path / "cell",
        find_free_port(),
        pg=pg,
        extra_env={ms.MIRIX_SESSION_HEADER_ENV: ITEM_HEADER},
    )
    env = ms.build_server_env(cell)
    values = await ms.probe_llm_contract(env)
    assert set(values) == set(protocol.MIRIX_SETTINGS_CONTRACT) | set(
        protocol.MIRIX_CONSTANTS_CONTRACT
    ) | {
        "image_token_multiplier",
        "skill_experience_pool",
        "llm_session_header",
    }
    assert values["llm_session_header"] == ITEM_HEADER
    assert ms.check_llm_contract(values) == []
    server = ms.MirixServer(cell, mode="fresh", startup_timeout_s=300)
    assert server.verify_llm_contract  # the real command is always checked
    try:
        await server.start()
        assert server.llm_contract == values
    finally:
        await server.stop()
        await ms.drop_cell_database(pg, cell.db_name)


# --- the template's build path carries MIRIX's current schema ---------------------------------

#: The migration that adds the experience pool's columns to an existing MIRIX database.
POOL_MIGRATION = protocol.REPO_ROOT / "scripts" / "migrate_skill_experience_pool.sql"
_ADD_COLUMN = re.compile(
    r"^ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+) (\w+(?:\(\d+\))?)(.*);$",
    re.MULTILINE,
)
_CREATE_INDEX = re.compile(r"CREATE INDEX IF NOT EXISTS (\w+)\s+ON (\w+)")
#: A migration's SQL type as Postgres reports it (``pg_type.typname``, and the
#: dimensions of a pgvector column, its ``atttypmod``).
_PG_TYPES = {
    "VARCHAR": "varchar",
    "JSON": "json",
    "INTEGER": "int4",
    "TIMESTAMPTZ": "timestamptz",
}


def pool_migration() -> tuple[list[tuple[str, str, str, bool]], list[tuple[str, str]]]:
    """``(table, column, pg type, NOT NULL)`` of every column the pool migration adds,
    and ``(index, table)`` of its indexes."""
    text = POOL_MIGRATION.read_text(encoding="utf-8")
    columns = []
    for table, column, sql_type, rest in _ADD_COLUMN.findall(text):
        vector = re.fullmatch(r"vector\((\d+)\)", sql_type)
        pg_type = f"vector({vector.group(1)})" if vector else _PG_TYPES[sql_type]
        columns.append((table, column, pg_type, "NOT NULL" in rest))
    return columns, _CREATE_INDEX.findall(text)


async def _columns(conn, table: str) -> dict[str, tuple[str, bool]]:
    """``column -> (pg type, NOT NULL)`` of one table, pgvector's dimensions included."""
    rows = await conn.fetch(
        "SELECT a.attname, t.typname, a.atttypmod, a.attnotnull "
        "FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid "
        "WHERE a.attrelid = $1::regclass AND a.attnum > 0 AND NOT a.attisdropped",
        table,
    )
    return {
        r["attname"]: (
            f"vector({r['atttypmod']})" if r["typname"] == "vector" else r["typname"],
            bool(r["attnotnull"]),
        )
        for r in rows
    }


def test_the_pool_migration_is_parsed_whole():
    columns, indexes = pool_migration()
    assert len(columns) == 12 and {t for t, *_ in columns} == {"skill_experience"}
    assert ("skill_experience", "pool_embedding", "vector(4096)", False) in columns
    assert ("skill_experience", "pool_embedding_failures", "int4", True) in columns
    assert ("skill_experience", "recurrence_count", "int4", True) in columns
    assert indexes == [("ix_skill_experience_agent_disposition", "skill_experience")]


@pytest.mark.asyncio
async def test_a_new_cell_database_has_the_pool_schema_and_an_old_one_is_refused(
    pg, tmp_path
):
    """``skill_bench_template`` holds only the extensions (``init.sql``); a cell attempt's
    database is a clone of it, and MIRIX builds its tables (``create_all``) when the
    cell's server first starts. So a MIRIX schema change, like the experience pool's,
    reaches every new attempt with no template rebuild: every column and index
    ``scripts/migrate_skill_experience_pool.sql`` adds is there, with its type. A cell
    database from before the change (the migration's columns missing) is refused by
    MIRIX's startup schema check, which names the script; once migrated it starts."""
    columns, indexes = pool_migration()
    cell = ms.CellServerSpec(
        f"it-schema-{secrets.token_hex(3)}",
        0,
        tmp_path / "cell",
        find_free_port(),
        pg=pg,
    )
    try:
        server = ms.MirixServer(cell, mode="fresh", startup_timeout_s=300)
        try:
            await server.start()
        finally:
            await server.stop()
        conn = await _connect(pg, cell.db_name)
        try:
            for table in {t for t, *_ in columns}:
                have = await _columns(conn, table)
                for _, column, pg_type, not_null in (
                    c for c in columns if c[0] == table
                ):
                    assert have.get(column) == (pg_type, not_null), column
            for index, table in indexes:
                assert await conn.fetchval(
                    "SELECT 1 FROM pg_indexes WHERE indexname = $1 AND tablename = $2",
                    index,
                    table,
                ), index
            # The same database as a cell of the previous MIRIX: no pool columns.
            for table, column, *_ in columns:
                await conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
        finally:
            await conn.close()

        refused = ms.MirixServer(cell, mode="resume", startup_timeout_s=300)
        with pytest.raises(ms.MirixServerError):
            try:
                await refused.start()
            finally:
                await refused.stop()
        log = cell.log_path.read_text(encoding="utf-8", errors="replace")
        assert "Database schema is out of date" in log
        assert "scripts/migrate_skill_experience_pool.sql" in log

        conn = await _connect(pg, cell.db_name)
        try:
            await conn.execute(POOL_MIGRATION.read_text(encoding="utf-8"))
            await conn.execute(POOL_MIGRATION.read_text(encoding="utf-8"))  # idempotent
        finally:
            await conn.close()
        resumed = ms.MirixServer(cell, mode="resume", startup_timeout_s=300)
        try:
            await resumed.start()
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                health = (await client.get(resumed.base_url + "/health")).json()
            assert health["status"] == "healthy"
        finally:
            await resumed.stop()
    finally:
        await ms.drop_cell_database(pg, cell.db_name)
