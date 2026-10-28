"""Smoke test for the Postgres integration harness.

Run it through the fresh-database runner, which creates a throwaway database,
exports MIRIX_PG_URI before MIRIX is imported, and drops the database after:

    scripts/run_pg_integration_tests.sh tests/test_pg_integration_smoke.py

It proves three things every later PG integration test relies on: MIRIX is in
Postgres mode, the database is a fresh per-run one with the required
extensions, and MIRIX's own startup DDL (create_all + drift guard) succeeds on
it.
"""

import pytest
from sqlalchemy import text

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


async def test_connects_to_a_fresh_database_with_required_extensions():
    from mirix.server.server import engine

    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
        database = (await conn.execute(text("SELECT current_database()"))).scalar_one()
        extensions = set(
            (await conn.execute(text("SELECT extname FROM pg_extension")))
            .scalars()
            .all()
        )
    assert database.startswith("mirix_it_"), (
        "run through scripts/run_pg_integration_tests.sh"
    )
    assert {"vector", "pg_trgm", "uuid-ossp"} <= extensions


async def test_startup_ddl_succeeds_on_the_fresh_database():
    from mirix.server.server import engine, ensure_tables_created

    await ensure_tables_created()
    async with engine.connect() as conn:
        tables = set(
            (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert {
        "organizations",
        "users",
        "procedural_memory",
        "conversation_message",
    } <= tables
    await engine.dispose()
