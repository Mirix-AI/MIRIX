"""
Lightweight startup migrations.

MIRIX has no Alembic — schema is created via ``Base.metadata.create_all`` in
``ensure_tables_created``. This module runs idempotent DROP/ALTER statements
that need to happen *before* ``create_all`` so the new ORM state takes hold.

Add a migration by appending to ``MIGRATIONS`` below. Each migration is a
``(name, sql_statements)`` tuple. Each statement runs in its own transaction;
failures are logged but do not raise (so a partial drop on dev DBs doesn't
brick startup).
"""

from typing import List, Tuple

from sqlalchemy.ext.asyncio import AsyncEngine

from mirix.log import get_logger

logger = get_logger(__name__)


# (migration_name, [statements])
MIGRATIONS: List[Tuple[str, List[str]]] = [
    (
        # v2 graph memory tables — replaced by Neo4j-backed implementation
        # (see lightrag_graph_manager). Drop in dependency order: edges that
        # FK into entity_nodes / episode_nodes must go first.
        "drop_v2_graph_memory_tables",
        [
            "DROP TABLE IF EXISTS involves_edges CASCADE",
            "DROP TABLE IF EXISTS entity_edges CASCADE",
            "DROP TABLE IF EXISTS episode_nodes CASCADE",
            "DROP TABLE IF EXISTS entity_nodes CASCADE",
        ],
    ),
]


async def run_startup_migrations(engine: AsyncEngine) -> None:
    """Run all pending migrations against ``engine``. Safe to call repeatedly."""
    for name, statements in MIGRATIONS:
        logger.info("Startup migration: %s", name)
        for stmt in statements:
            try:
                async with engine.begin() as conn:
                    from sqlalchemy import text as sa_text
                    await conn.execute(sa_text(stmt))
            except Exception as e:
                # Most likely on fresh DBs: table never existed. That's fine.
                logger.debug("Migration stmt skipped (%s): %s", stmt, e)
