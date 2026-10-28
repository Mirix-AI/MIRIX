"""Postgres integration tests for the skill-learning data layer and its migrations.

Run through the fresh-database runner (MIRIX_PG_URI must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_skill_data_layer_pg.py

Two paths must converge on the same schema:

* fresh — ``create_all`` with the current models (what a new deployment gets);
* upgrade — the pre-change ("main") shape of the changed tables, with data in
  it, migrated by the six scripts. The main-era models are not importable here,
  so the old shape is produced by building the current schema and applying the
  exact pre-change DDL (new tables absent; old procedural vector columns and
  full unique constraint; no content_parts / original_text_length / files.user_id
  / curation_attempts). Each script is then run TWICE.

Checked: Postgres-only DDL (generated tsvector + GIN, partial unique index,
CHECK / FK / cascade), the conversation_session backfill, that re-running the
scripts changes nothing, that the drift guard flags then clears the old schema,
that the migrated schema is identical to the fresh one, and that the managers
work on the migrated schema (instructions keyword fallback, vector search, re-creating
a soft-deleted skill's name).
"""

import datetime as dt
import json
import re
import uuid
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
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

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
# Run order: columns on existing tables first, then the new tables (skill
# history references procedural_memory).
MIGRATION_SCRIPTS = [
    "migrate_conversation_message_parts.sql",
    "migrate_skill_experience_attempts.sql",
    "migrate_skill_retrieval_fields.sql",
    "migrate_add_conversation_session.sql",
    "migrate_add_skill_history.sql",
    "migrate_add_learning_run_and_retrieval_log.sql",
    "migrate_skill_experience_pool.sql",
    "migrate_p16_agent_context_and_skill_feedback.sql",
]
NEW_TABLES = [
    "conversation_session",
    "skill_object",
    "skill_version",
    "skill_learning_run",
    "retrieval_log",
    "skill_session_feedback",
]
CHANGED_TABLES = [
    "conversation_message",
    "files",
    "skill_experience",
    "procedural_memory",
]

# The experience pool's columns and index, absent before it.
POOL_COLUMNS = [
    "disposition",
    "cited_skill_ids",
    "pool_scope",
    "pool_embedding",
    "pool_embedding_model",
    "recurrence_count",
    "ignored_pushes",
    "pushed_count",
    "last_pushed_at",
    "pool_retired_at",
    "pool_retire_reason",
]
POOL_DOWNGRADE_SQL = [
    "DROP INDEX IF EXISTS ix_skill_experience_agent_disposition",
    *(f"ALTER TABLE skill_experience DROP COLUMN {column}" for column in POOL_COLUMNS),
]

# The pre-change ("main" 8cb06a6) shape of the changed tables, applied on top of
# a fresh current schema. triggers/examples become JSONB like a database that
# went through scripts/migrate_procedural_to_skill.sql.
DOWNGRADE_TO_MAIN_SQL = [
    *(f"DROP TABLE IF EXISTS {table} CASCADE" for table in NEW_TABLES),
    "ALTER TABLE conversation_message DROP COLUMN content_parts",
    "ALTER TABLE conversation_message DROP COLUMN original_text_length",
    "ALTER TABLE files DROP COLUMN user_id",
    "ALTER TABLE skill_experience DROP COLUMN curation_attempts",
    *POOL_DOWNGRADE_SQL,
    "DROP INDEX uq_procedural_memory_org_user_name_active",
    "ALTER TABLE procedural_memory DROP COLUMN retrieval_tsv",
    "ALTER TABLE procedural_memory RENAME COLUMN retrieval_embedding TO description_embedding",
    "ALTER TABLE procedural_memory ADD COLUMN instructions_embedding vector(4096)",
    "ALTER TABLE procedural_memory ADD CONSTRAINT uq_procedural_memory_org_user_name "
    "UNIQUE (organization_id, user_id, name)",
    "ALTER TABLE procedural_memory ALTER COLUMN triggers TYPE JSONB USING triggers::jsonb",
    "ALTER TABLE procedural_memory ALTER COLUMN examples TYPE JSONB USING examples::jsonb",
]

NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
IDLE = dt.timedelta(seconds=172800)


def _asyncpg_dsn() -> str:
    uri = settings.mirix_pg_uri_no_default
    for prefix in (
        "postgresql+pg8000://",
        "postgresql+asyncpg://",
        "postgresql+psycopg2://",
    ):
        if uri.startswith(prefix):
            return "postgresql://" + uri[len(prefix) :]
    return uri


async def _run_script(name: str, prelude: str = "") -> None:
    """Execute a migration file as psql would (simple protocol, multi-statement)."""
    sql = (SCRIPTS_DIR / name).read_text()
    conn = await asyncpg.connect(_asyncpg_dsn())
    try:
        await conn.execute(prelude + sql)
    finally:
        await conn.close()


async def _run_all_scripts(prelude: str = "") -> None:
    for name in MIGRATION_SCRIPTS:
        await _run_script(name, prelude)


async def _execute(*statements: str, params: dict = None) -> None:
    from mirix.server.server import engine

    async with engine.begin() as conn:
        for statement in statements:
            await conn.execute(text(statement), params or {})


async def _fetch(sql: str, **params):
    from mirix.server.server import engine

    async with engine.connect() as conn:
        return (await conn.execute(text(sql), params)).mappings().all()


async def _reset_to_fresh_schema() -> None:
    """Drop everything MIRIX owns, then create_all the current models."""
    from mirix.orm.base import Base
    from mirix.server.server import engine

    async with engine.begin() as conn:
        for table in NEW_TABLES:
            await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def _build_main_schema() -> None:
    await _reset_to_fresh_schema()
    await _execute(*DOWNGRADE_TO_MAIN_SQL)


async def _schema_snapshot(tables) -> dict:
    """Columns, indexes and constraints of `tables`, keyed by name (order-free)."""
    columns = await _fetch(
        """
        SELECT table_name, column_name, data_type, udt_name, is_nullable,
               column_default, is_generated, generation_expression,
               character_maximum_length
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = ANY(:tables)
        """,
        tables=list(tables),
    )
    indexes = await _fetch(
        """
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE schemaname = current_schema() AND tablename = ANY(:tables)
        """,
        tables=list(tables),
    )
    constraints = await _fetch(
        """
        SELECT c.conrelid::regclass::text AS table_name, c.conname,
               pg_get_constraintdef(c.oid) AS definition
        FROM pg_constraint c
        WHERE c.connamespace = current_schema()::regnamespace
          AND c.conrelid::regclass::text = ANY(:tables)
        """,
        tables=list(tables),
    )
    return {
        "columns": {(r["table_name"], r["column_name"]): dict(r) for r in columns},
        "indexes": {(r["tablename"], r["indexname"]): r["indexdef"] for r in indexes},
        "constraints": {
            (r["table_name"], r["conname"]): r["definition"] for r in constraints
        },
    }


async def _data_snapshot() -> dict:
    return {
        "sessions": [
            dict(r)
            for r in await _fetch(
                "SELECT * FROM conversation_session ORDER BY user_id, session_id"
            )
        ],
        "skills": [
            dict(r)
            for r in await _fetch(
                "SELECT id, name, is_deleted, retrieval_embedding IS NULL AS no_vector, "
                "retrieval_tsv::text AS tsv FROM procedural_memory ORDER BY id"
            )
        ],
        "experiences": [
            dict(r)
            for r in await _fetch(
                "SELECT id, curation_attempts FROM skill_experience ORDER BY id"
            )
        ],
        "messages": [
            dict(r)
            for r in await _fetch(
                "SELECT id, content_parts::text AS parts, original_text_length FROM conversation_message ORDER BY id"
            )
        ],
        "files": [
            dict(r) for r in await _fetch("SELECT id, user_id FROM files ORDER BY id")
        ],
    }


# --------------------------------------------------------------------------- #
# Seed data for the upgrade path                                              #
# --------------------------------------------------------------------------- #

ORG = "org-it"
USER_A = "user-it-a"
USER_B = "user-it-b"
META_AGENT = "agent-it-meta"
SUB_AGENT = "agent-it-sub"


async def _seed_main_era_data() -> dict:
    """Rows as they would exist on main before the migration."""
    t = lambda delta: NOW - delta  # noqa: E731
    await _execute(
        f"INSERT INTO organizations (id, name) VALUES ('{ORG}', '{ORG}')",
        f"INSERT INTO users (id, name, organization_id, status, timezone, is_admin) VALUES "
        f"('{USER_A}', 'a', '{ORG}', 'active', 'UTC', FALSE), ('{USER_B}', 'b', '{ORG}', 'active', 'UTC', FALSE)",
        f"INSERT INTO agents (id, organization_id, agent_type) VALUES "
        f"('{META_AGENT}', '{ORG}', 'meta_memory_agent'), ('{SUB_AGENT}', '{ORG}', 'episodic_memory_agent')",
    )

    def cm(row_id, user, session, created, tag, deleted=False):
        return (
            "INSERT INTO conversation_message (id, organization_id, user_id, session_id, role, content, "
            "session_tag, created_at, is_deleted) VALUES "
            f"('{row_id}', '{ORG}', '{user}', '{session}', 'user', 'x', "
            f"{'NULL' if tag is None else repr(tag)}, '{created.isoformat()}', {str(deleted).upper()})"
        )

    old_first = t(dt.timedelta(days=5))
    old_last = t(dt.timedelta(days=5) - dt.timedelta(minutes=3))
    recent_first = t(dt.timedelta(minutes=30))
    recent_last = t(dt.timedelta(minutes=10))
    legacy_at = t(dt.timedelta(days=3))
    await _execute(
        # old_task: first row tagged 'task', later rows 'conversation' -> 'task'.
        cm("convmsg-1", USER_A, "old_task", old_first, "task"),
        cm(
            "convmsg-2",
            USER_A,
            "old_task",
            old_first + dt.timedelta(minutes=1),
            "conversation",
        ),
        cm("convmsg-3", USER_A, "old_task", old_last, "conversation"),
        # recent_conv: open; one deleted row that must not count.
        cm("convmsg-4", USER_A, "recent_conv", recent_first, "conversation"),
        cm("convmsg-5", USER_A, "recent_conv", recent_last, "conversation"),
        cm(
            "convmsg-6",
            USER_A,
            "recent_conv",
            recent_last + dt.timedelta(minutes=1),
            "conversation",
            deleted=True,
        ),
        # legacy_null: untagged legacy rows, no agent-loop history.
        cm("convmsg-7", USER_B, "legacy_null", legacy_at, None),
        # odd_tag: a non-standard tag (misconfigured default) becomes NULL.
        cm("convmsg-10", USER_B, "odd_tag", legacy_at, "chat"),
        # the same session_id under two users = two sessions.
        cm("convmsg-8", USER_A, "shared", recent_first, "task"),
        cm("convmsg-9", USER_B, "shared", legacy_at, "task"),
        # Agent-loop history: the META agent saw old_task (a sub-agent row with
        # an earlier timestamp must not win).
        "INSERT INTO messages (id, organization_id, user_id, agent_id, role, tool_calls, session_id, "
        "created_at) VALUES "
        f"('message-1', '{ORG}', '{USER_A}', '{SUB_AGENT}', 'user', '[]', 'old_task', "
        f"'{(old_first - dt.timedelta(minutes=5)).isoformat()}'), "
        f"('message-2', '{ORG}', '{USER_A}', '{META_AGENT}', 'user', '[]', 'old_task', '{old_first.isoformat()}')",
        # Skills: a live one with an old description vector, and a soft-deleted one.
        "INSERT INTO procedural_memory (id, organization_id, user_id, name, entry_type, description, "
        "instructions, triggers, examples, version, last_modify, description_embedding, "
        "instructions_embedding, is_deleted) VALUES "
        f"('proc-live', '{ORG}', '{USER_A}', 'deploy-service', 'workflow', 'Ship the service to production', "
        """'Run kubectl apply', '["release day", "ship it"]', '[]', '0.1.0', '{}', """
        "array_fill(0.5, ARRAY[4096])::vector, array_fill(0.25, ARRAY[4096])::vector, FALSE), "
        f"('proc-gone', '{ORG}', '{USER_A}', 'retired-skill', 'workflow', 'old', 'old', '[]', '[]', "
        "'0.1.0', '{}', NULL, NULL, TRUE)",
        "INSERT INTO skill_experience (id, organization_id, user_id, agent_id, session_id, "
        "experience_type, title, content, importance, credibility, evidence, status) VALUES "
        f"('sexp-1', '{ORG}', '{USER_A}', '{META_AGENT}', 'old_task', 'worth_learning', 't', 'c', 0.5, 0.5, "
        "'', 'pending')",
        f"INSERT INTO files (id, organization_id, file_name) VALUES ('file-1', '{ORG}', 'a.png')",
    )
    return {
        "old_first": old_first,
        "old_last": old_last,
        "recent_first": recent_first,
        "recent_last": recent_last,
        "legacy_at": legacy_at,
    }


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def engine_lifecycle():
    from mirix.server.server import engine

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def fresh_snapshot(engine_lifecycle):
    """Schema of every new/changed table as create_all builds it."""
    await _reset_to_fresh_schema()
    return await _schema_snapshot(NEW_TABLES + CHANGED_TABLES)


# --------------------------------------------------------------------------- #
# Fresh path                                                                  #
# --------------------------------------------------------------------------- #


async def test_fresh_schema_has_the_postgres_only_ddl(fresh_snapshot):
    columns, indexes, constraints = (
        fresh_snapshot["columns"],
        fresh_snapshot["indexes"],
        fresh_snapshot["constraints"],
    )
    tsv = columns[("procedural_memory", "retrieval_tsv")]
    assert tsv["udt_name"] == "tsvector"
    assert tsv["is_generated"] == "ALWAYS"
    assert "setweight" in tsv["generation_expression"]
    assert ("procedural_memory", "description_embedding") not in columns
    assert ("procedural_memory", "instructions_embedding") not in columns
    assert columns[("procedural_memory", "retrieval_embedding")]["udt_name"] == "vector"

    assert (
        "USING gin (retrieval_tsv)"
        in indexes[("procedural_memory", "ix_procedural_memory_retrieval_tsv")]
    )
    partial = indexes[
        ("procedural_memory", "uq_procedural_memory_org_user_name_active")
    ]
    assert partial.startswith("CREATE UNIQUE INDEX")
    assert "WHERE (NOT is_deleted)" in partial
    assert (
        "procedural_memory",
        "uq_procedural_memory_org_user_name",
    ) not in constraints

    assert (
        "UNIQUE (organization_id, user_id, session_id)"
        in constraints[
            ("conversation_session", "uq_conversation_session_org_user_session")
        ]
    )
    assert (
        "session_id"
        in constraints[
            ("conversation_session", "ck_conversation_session_session_id_format")
        ]
    )
    assert (
        constraints[("skill_object", "skill_object_pkey")]
        == "PRIMARY KEY (organization_id, sha256)"
    )
    assert (
        "UNIQUE (skill_id, version_no)"
        in constraints[("skill_version", "uq_skill_version_skill_version_no")]
    )
    assert (
        "ON DELETE CASCADE"
        in constraints[("skill_version", "skill_version_skill_id_fkey")]
    )
    assert "ON DELETE SET NULL" in constraints[("files", "files_user_id_fkey")]


async def test_scripts_are_no_ops_on_a_fresh_schema(fresh_snapshot):
    await _reset_to_fresh_schema()
    for _ in range(2):
        await _run_all_scripts()
        assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == fresh_snapshot


async def test_fresh_schema_passes_the_drift_guard(fresh_snapshot):
    from mirix.server.server import ensure_tables_created

    await _reset_to_fresh_schema()
    await ensure_tables_created()


# --------------------------------------------------------------------------- #
# Upgrade path                                                                #
# --------------------------------------------------------------------------- #


async def test_drift_guard_flags_the_main_schema_with_per_column_hints(fresh_snapshot):
    from mirix.server.server import (
        _describe_schema_drift,
        _find_missing_columns,
        engine,
    )

    await _build_main_schema()
    async with engine.connect() as conn:
        missing = await conn.run_sync(_find_missing_columns)
    assert missing == {
        "conversation_message": ["content_parts", "original_text_length"],
        "files": ["user_id"],
        "procedural_memory": ["retrieval_embedding", "retrieval_tsv"],
        "skill_experience": ["curation_attempts", *POOL_COLUMNS],
    }
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert (
        "'procedural_memory' is missing columns ['retrieval_embedding', 'retrieval_tsv']; run scripts/migrate_skill_retrieval_fields.sql"
        in report
    )
    assert (
        "'files' is missing columns ['user_id']; run scripts/migrate_conversation_message_parts.sql"
        in report
    )
    assert (
        "run scripts/migrate_skill_experience_attempts.sql, then "
        "scripts/migrate_skill_experience_pool.sql" in report
    )
    assert "no registered script" not in report


async def test_the_experience_pool_upgrade_converges_on_the_fresh_schema(
    fresh_snapshot,
):
    """The experience pool on a database from before it: the drift guard names
    the pool script, which adds exactly the create_all columns (a 4,096-wide
    vector, the counters' defaults on existing rows) and the index, twice
    without change, and backfills nothing."""
    from mirix.server.server import (
        _describe_schema_drift,
        _find_missing_columns,
        engine,
    )

    await _reset_to_fresh_schema()
    await _execute(*POOL_DOWNGRADE_SQL)
    await _execute(
        f"INSERT INTO organizations (id, name) VALUES ('{ORG}', '{ORG}')",
        f"INSERT INTO users (id, name, organization_id, status, timezone, is_admin) VALUES "
        f"('{USER_A}', 'a', '{ORG}', 'active', 'UTC', FALSE)",
        f"INSERT INTO agents (id, organization_id, agent_type) VALUES "
        f"('{META_AGENT}', '{ORG}', 'meta_memory_agent')",
        "INSERT INTO skill_experience (id, organization_id, user_id, agent_id, session_id, "
        "experience_type, title, content, importance, credibility, evidence, status, "
        "consumed_by) VALUES "
        f"('sexp-old', '{ORG}', '{USER_A}', '{META_AGENT}', 's', 'worth_learning', 't', "
        "'c', 0.5, 0.5, '', 'consumed', 'run:b0')",
    )
    async with engine.connect() as conn:
        missing = await conn.run_sync(_find_missing_columns)
    assert missing == {"skill_experience": POOL_COLUMNS}
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert "run scripts/migrate_skill_experience_pool.sql" in report
    for _ in range(2):
        await _run_all_scripts()
        assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == fresh_snapshot
    [width] = await _fetch(
        "SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute "
        "WHERE attrelid = 'skill_experience'::regclass AND attname = 'pool_embedding'"
    )
    assert width["t"] == "vector(4096)"
    [old] = await _fetch(
        "SELECT disposition, pool_scope, recurrence_count, ignored_pushes, "
        "pushed_count, pool_retired_at FROM skill_experience WHERE id = 'sexp-old'"
    )
    # Not backfilled into the pool: consumed before it, disposition unknown.
    assert dict(old) == {
        "disposition": None,
        "pool_scope": None,
        "recurrence_count": 1,
        "ignored_pushes": 0,
        "pushed_count": 0,
        "pool_retired_at": None,
    }
    async with engine.connect() as conn:
        assert await conn.run_sync(_find_missing_columns) == {}


async def test_an_earlier_revision_of_the_new_tables_is_brought_up_to_date(
    fresh_snapshot,
):
    """Retry bookkeeping columns (and the retrieval log's chunked
    embedding calls) were added to tables an earlier revision of these scripts
    created. The drift guard names the scripts, and re-running them yields
    exactly the create_all schema."""
    from mirix.server.server import (
        _describe_schema_drift,
        _find_missing_columns,
        engine,
    )

    await _reset_to_fresh_schema()
    await _execute(
        "ALTER TABLE conversation_session DROP COLUMN distill_infra_failures",
        "ALTER TABLE skill_learning_run DROP COLUMN retry_events",
        "ALTER TABLE retrieval_log DROP COLUMN retry_events",
        "ALTER TABLE retrieval_log DROP COLUMN embedding_chunked_calls",
    )
    async with engine.connect() as conn:
        missing = await conn.run_sync(_find_missing_columns)
    assert {table: sorted(columns) for table, columns in missing.items()} == {
        "conversation_session": ["distill_infra_failures"],
        "retrieval_log": ["embedding_chunked_calls", "retry_events"],
        "skill_learning_run": ["retry_events"],
    }
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert "run scripts/migrate_add_conversation_session.sql" in report
    assert "run scripts/migrate_add_learning_run_and_retrieval_log.sql" in report
    assert "no registered script" not in report
    for _ in range(2):
        await _run_all_scripts()
        assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == fresh_snapshot


async def test_upgrade_migrates_backfills_and_is_idempotent(fresh_snapshot):
    from mirix.server.server import _find_missing_columns, engine, ensure_tables_created

    await _build_main_schema()
    seeded = await _seed_main_era_data()

    await _run_all_scripts()
    first_schema = await _schema_snapshot(NEW_TABLES + CHANGED_TABLES)
    first_data = await _data_snapshot()

    # --- converges on the fresh schema (JSONB triggers/examples aside) ------
    tsv_key = ("procedural_memory", "retrieval_tsv")

    def comparable(snapshot):
        out = {kind: dict(items) for kind, items in snapshot.items()}
        for column in ("triggers", "examples"):
            out["columns"].pop(("procedural_memory", column))
        # The same expression, `triggers::jsonb::text`: Postgres stores it
        # as ((triggers)::jsonb)::text on a JSON column and (triggers)::text
        # on a JSONB one (the cast is a no-op there).
        tsv = dict(out["columns"][tsv_key])
        tsv["generation_expression"] = tsv["generation_expression"].replace(
            "((triggers)::jsonb)::text", "(triggers)::text"
        )
        out["columns"][tsv_key] = tsv
        return out

    assert comparable(first_schema) == comparable(fresh_snapshot)
    assert (
        first_schema["columns"][("procedural_memory", "triggers")]["udt_name"]
        == "jsonb"
    )
    assert (
        "((triggers)::jsonb)::text"
        in (fresh_snapshot["columns"][tsv_key]["generation_expression"])
    )

    # --- procedural retrieval fields ----------------------------------------
    skills = {s["id"]: s for s in first_data["skills"]}
    assert skills["proc-live"]["no_vector"], (
        "the old description vector must be cleared"
    )
    matches = await _fetch(
        "SELECT id FROM procedural_memory WHERE retrieval_tsv @@ to_tsquery('english', :q) ORDER BY id",
        q="release",
    )
    assert [m["id"] for m in matches] == ["proc-live"]  # a trigger word is indexed
    weights = skills["proc-live"]["tsv"]
    assert re.search(r"'deploy':\d+A", weights)  # name at weight A
    assert re.search(r"'releas':\d+A", weights)  # triggers at weight A
    assert re.search(r"'product':\d+B", weights)  # description at weight B
    assert (
        "kubectl" not in weights
    )  # instructions are not part of the retrieval document

    # A soft-deleted name can be reused; a live name cannot.
    insert_skill = (
        "INSERT INTO procedural_memory (id, organization_id, user_id, name, entry_type, description, "
        "instructions, triggers, examples, version, last_modify) VALUES "
        f"(:id, '{ORG}', '{USER_A}', :name, 'workflow', 'd', 'i', '[]', '[]', '0.1.0', '{{}}')"
    )
    await _execute(insert_skill, params={"id": "proc-reuse", "name": "retired-skill"})
    with pytest.raises(Exception) as exc:
        await _execute(
            insert_skill, params={"id": "proc-dup", "name": "deploy-service"}
        )
    assert "uq_procedural_memory_org_user_name_active" in str(exc.value)
    await _execute("DELETE FROM procedural_memory WHERE id = 'proc-reuse'")

    # --- conversation_session backfill --------------------------------
    sessions = {(s["user_id"], s["session_id"]): s for s in first_data["sessions"]}
    assert set(sessions) == {
        (USER_A, "old_task"),
        (USER_A, "recent_conv"),
        (USER_A, "shared"),
        (USER_B, "legacy_null"),
        (USER_B, "odd_tag"),
        (USER_B, "shared"),
    }
    assert sessions[(USER_B, "odd_tag")]["session_tag"] is None
    old = sessions[(USER_A, "old_task")]
    assert old["id"].startswith("convsess-")
    assert old["organization_id"] == ORG
    assert old["session_tag"] == "task"  # the earliest row's tag
    assert (
        old["agent_id"] == META_AGENT
    )  # the meta agent, not the earlier sub-agent row
    assert (old["first_message_at"], old["last_message_at"], old["message_count"]) == (
        seeded["old_first"],
        seeded["old_last"],
        3,
    )
    assert old["closed_at"] == seeded["old_last"] + IDLE
    assert old["close_reason"] == "idle"
    for counter in (
        "reopen_count",
        "distill_failures",
        "truncated_message_count",
        "truncated_chars",
    ):
        assert old[counter] == 0
    assert old["distill_failed_at"] is None and old["distill_retry_after"] is None

    recent = sessions[(USER_A, "recent_conv")]
    assert recent["message_count"] == 2  # the deleted row is not counted
    assert recent["last_message_at"] == seeded["recent_last"]
    assert recent["closed_at"] is None and recent["close_reason"] is None
    # No agent-loop history and no recorded writer: the organization's only
    # live meta agent (an episodic sub-agent is not a candidate).
    assert recent["agent_id"] == META_AGENT

    legacy = sessions[(USER_B, "legacy_null")]
    assert legacy["session_tag"] is None
    assert legacy["closed_at"] == seeded["legacy_at"] + IDLE
    assert legacy["agent_id"] == META_AGENT

    # --- other changed tables ----------------------------------------------
    assert first_data["experiences"] == [{"id": "sexp-1", "curation_attempts": 0}]
    assert all(
        m["parts"] is None and m["original_text_length"] is None
        for m in first_data["messages"]
    )
    assert first_data["files"] == [{"id": "file-1", "user_id": None}]

    # --- idempotent: a second run changes nothing ---------------------------
    tsv_attnum = "SELECT attnum FROM pg_attribute WHERE attrelid = 'procedural_memory'::regclass AND attname = 'retrieval_tsv'"
    before_rerun = await _fetch(tsv_attnum)
    await _run_all_scripts()
    assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == first_schema
    assert await _data_snapshot() == first_data
    # Not even rebuilt (a DROP + ADD would renumber it): on the JSONB column
    # the tsv is right, and rebuilding it would rewrite the table every run.
    assert await _fetch(tsv_attnum) == before_rerun

    # --- the drift guard now passes -----------------------------------------
    async with engine.connect() as conn:
        assert await conn.run_sync(_find_missing_columns) == {}
    await ensure_tables_created()


async def test_backfill_attributes_sessions_without_agent_history(fresh_snapshot):
    """The backfill attributes a session whose agent-loop history is
    gone to the only live meta agent of the only client that wrote it — and
    leaves every ambiguous session unattributed. The runtime rule
    (adopt_unattributed_sessions) agrees with it."""
    from mirix.services.conversation_session_manager import (
        ConversationSessionManager,
    )

    await _build_main_schema()
    org, user = "org-attr", "user-attr"
    at = (NOW - dt.timedelta(days=4)).isoformat()
    await _execute(
        f"INSERT INTO organizations (id, name) VALUES ('{org}', '{org}')",
        f"INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
        f"VALUES ('{user}', 'u', '{org}', 'active', 'UTC', FALSE)",
        "INSERT INTO clients (id, organization_id, name, status, write_scope, read_scopes) VALUES "
        + ", ".join(
            f"('{c}', '{org}', '{c}', 'active', 'test', '[\"test\"]')"
            for c in ("client-1", "client-2", "client-3")
        ),
        "INSERT INTO agents (id, organization_id, agent_type, _created_by_id, is_deleted) VALUES "
        f"('meta-1', '{org}', 'meta_memory_agent', 'client-1', FALSE), "
        f"('meta-1-old', '{org}', 'meta_memory_agent', 'client-1', TRUE), "
        f"('meta-2', '{org}', 'meta_memory_agent', 'client-2', FALSE), "
        f"('sub-3', '{org}', 'episodic_memory_agent', 'client-3', FALSE)",
    )

    def row(row_id, session, writer):
        writer_sql = "NULL" if writer is None else f"'{writer}'"
        return (
            "INSERT INTO conversation_message (id, organization_id, user_id, session_id, "
            "role, content, session_tag, created_at, is_deleted, _created_by_id) VALUES "
            f"('{row_id}', '{org}', '{user}', '{session}', 'user', 'x', 'task', '{at}', "
            f"FALSE, {writer_sql})"
        )

    await _execute(
        row("a-1", "by-client-1", "client-1"),
        row("a-2", "by-client-1", "client-1"),
        row("b-1", "two-writers", "client-1"),
        row("b-2", "two-writers", "client-2"),
        row("c-1", "no-writer", None),
        row("d-1", "writer-without-meta", "client-3"),
        row("e-1", "history-wins", "client-1"),
        "INSERT INTO messages (id, organization_id, user_id, agent_id, role, tool_calls, "
        f"session_id, created_at) VALUES ('message-e', '{org}', '{user}', 'meta-2', 'user', "
        f"'[]', 'history-wins', '{at}')",
    )
    await _run_script("migrate_add_conversation_session.sql")
    rows = await _fetch(
        "SELECT session_id, agent_id FROM conversation_session WHERE user_id = :u",
        u=user,
    )
    attributed = {r["session_id"]: r["agent_id"] for r in rows}
    assert attributed == {
        "by-client-1": "meta-1",  # the deleted meta-1-old is not a candidate
        "two-writers": None,
        "no-writer": None,  # two live meta agents in the organization
        "writer-without-meta": None,
        "history-wins": "meta-2",
    }
    assert (
        await ConversationSessionManager().adopt_unattributed_sessions(
            organization_id=org
        )
        == 0
    )

    # client-1 deletes and re-creates its meta agent. Its sessions
    # are orphaned, and the runtime rule hands them to the new meta agent (the
    # ambiguous ones stay as they are).
    await _execute(
        "DELETE FROM agents WHERE id = 'meta-1'",
        "INSERT INTO agents (id, organization_id, agent_type, _created_by_id, is_deleted) "
        f"VALUES ('meta-1-new', '{org}', 'meta_memory_agent', 'client-1', FALSE)",
    )
    manager = ConversationSessionManager()
    assert not await manager.agent_is_live("meta-1")
    assert await manager.adopt_unattributed_sessions(organization_id=org) == 1
    rows = await _fetch(
        "SELECT session_id, agent_id FROM conversation_session WHERE user_id = :u",
        u=user,
    )
    assert {r["session_id"]: r["agent_id"] for r in rows} == {
        "by-client-1": "meta-1-new",
        "two-writers": None,
        "no-writer": None,
        "writer-without-meta": None,
        "history-wins": "meta-2",
    }


async def test_following_the_drift_guard_makes_legacy_sessions_learnable(
    fresh_snapshot,
):
    """An operator who upgrades a main-era database and runs exactly
    the scripts the startup guard names ends up with every legacy pending
    session claimable (the guard names the conversation_session backfill)."""
    from mirix.server.server import ensure_tables_created
    from mirix.services.agent_trigger_state_manager import AgentTriggerStateManager
    from mirix.services.conversation_session_manager import (
        ConversationSessionManager,
    )

    await _build_main_schema()
    await _seed_main_era_data()
    await _execute(
        "INSERT INTO clients (id, organization_id, name, status, write_scope, read_scopes) "
        f"VALUES ('client-it', '{ORG}', 'c', 'active', 'test', '[\"test\"]')",
        "UPDATE agents SET _created_by_id = 'client-it'",
    )

    with pytest.raises(RuntimeError) as exc:
        await ensure_tables_created()
    report = str(exc.value)
    named = list(dict.fromkeys(re.findall(r"scripts/[\w.-]+\.sql", report)))
    assert set(named) == {
        "scripts/migrate_conversation_message_parts.sql",
        "scripts/migrate_skill_experience_attempts.sql",
        "scripts/migrate_skill_retrieval_fields.sql",
        "scripts/migrate_add_conversation_session.sql",
        "scripts/migrate_skill_experience_pool.sql",
    }
    assert "data: sessions written before this version" in report

    for script in named:
        await _run_script(script.split("/", 1)[1])
    await ensure_tables_created()  # the guard is satisfied

    scopes = await ConversationSessionManager().list_pending_scopes()
    assert (ORG, USER_A, META_AGENT) in scopes and (ORG, USER_B, META_AGENT) in scopes
    trigger = AgentTriggerStateManager()
    claim_a = await trigger.check_and_claim_fire(
        agent_id=META_AGENT, user_id=USER_A, organization_id=ORG, force=True
    )
    assert claim_a.session_ids == ["old_task"]
    claim_b = await trigger.check_and_claim_fire(
        agent_id=META_AGENT, user_id=USER_B, organization_id=ORG, force=True
    )
    assert sorted(claim_b.session_ids) == ["legacy_null", "odd_tag", "shared"]


async def test_a_null_org_turn_does_not_abort_startup_after_the_backfill(
    fresh_snapshot,
):
    """A live turn whose organization_id is NULL gets a session row
    from the backfill (its NOT EXISTS is NULL-safe); the startup data check
    must match it NULL-safely too, or every startup aborts asking for a script
    that cannot help."""
    from mirix.server.server import (
        _conversation_sessions_need_backfill,
        engine,
        ensure_tables_created,
    )

    await _build_main_schema()
    await _seed_main_era_data()
    at = (NOW - dt.timedelta(days=4)).isoformat()
    await _execute(
        "INSERT INTO conversation_message (id, organization_id, user_id, session_id, "
        "role, content, session_tag, created_at, is_deleted) VALUES "
        f"('orgless-1', NULL, '{USER_A}', 'orgless', 'user', 'x', 'task', '{at}', FALSE)"
    )
    await _run_all_scripts()
    rows = await _fetch(
        "SELECT organization_id FROM conversation_session WHERE session_id = 'orgless'"
    )
    assert [r["organization_id"] for r in rows] == [None]
    async with engine.connect() as conn:
        assert await conn.run_sync(_conversation_sessions_need_backfill) is False
    await ensure_tables_created()  # does not abort
    # And a re-run of the backfill does not duplicate the NULL-org row.
    await _run_script("migrate_add_conversation_session.sql")
    assert (
        len(
            await _fetch(
                "SELECT id FROM conversation_session WHERE session_id = 'orgless'"
            )
        )
        == 1
    )


async def test_backfill_honors_the_idle_timeout_setting(fresh_snapshot):
    await _build_main_schema()
    await _seed_main_era_data()
    await _run_all_scripts(prelude="SET mirix.session_idle_timeout_seconds = '60';\n")
    rows = await _fetch(
        "SELECT closed_at, last_message_at FROM conversation_session WHERE user_id = :u AND session_id = 'recent_conv'",
        u=USER_A,
    )
    assert rows[0]["closed_at"] == rows[0]["last_message_at"] + dt.timedelta(seconds=60)


async def test_retrieval_rename_never_clears_recomputed_vectors(fresh_snapshot):
    """A re-run after vectors were recomputed must keep them (the clear runs once)."""
    await _build_main_schema()
    await _seed_main_era_data()
    await _run_script("migrate_skill_retrieval_fields.sql")
    await _execute(
        "UPDATE procedural_memory SET retrieval_embedding = array_fill(0.1, ARRAY[4096])::vector"
    )
    await _run_script("migrate_skill_retrieval_fields.sql")
    rows = await _fetch(
        "SELECT count(*) AS n FROM procedural_memory WHERE retrieval_embedding IS NULL"
    )
    assert rows[0]["n"] == 0


async def _pending_data_scripts():
    """The scripts the startup guard's data checks name on this database."""
    from mirix.server.server import _find_pending_data_migrations, engine

    async with engine.connect() as conn:
        pending = await conn.run_sync(_find_pending_data_migrations)
    return [script for _, script in pending]


async def test_an_escaped_tsv_of_an_earlier_script_is_rebuilt_once(fresh_snapshot):
    """An earlier revision generated retrieval_tsv from `triggers::text`,
    whose backslash-u escapes (json.dumps' ensure_ascii) reached the index as
    'u201ccool'. The script rebuilds such a column once, then leaves it."""
    await _reset_to_fresh_schema()
    await _execute(
        "ALTER TABLE procedural_memory DROP COLUMN retrieval_tsv",
        "ALTER TABLE procedural_memory ADD COLUMN retrieval_tsv tsvector GENERATED "
        "ALWAYS AS (setweight(to_tsvector('english', coalesce(name, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(triggers::text, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(description, '')), 'B')) STORED",
        "CREATE INDEX ix_procedural_memory_retrieval_tsv ON procedural_memory "
        "USING gin (retrieval_tsv)",
    )
    await _execute(
        "INSERT INTO organizations (id, name) VALUES (:o, :o) ON CONFLICT DO NOTHING",
        params={"o": ORG},
    )
    await _execute(
        "INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
        "VALUES ('user-escaped', 'u', :o, 'active', 'UTC', FALSE) "
        "ON CONFLICT DO NOTHING",
        params={"o": ORG},
    )
    await _execute(
        "INSERT INTO procedural_memory (id, organization_id, user_id, name, "
        "entry_type, description, instructions, triggers, examples, version, "
        "last_modify, is_deleted) VALUES ('proc-escaped', :o, 'user-escaped', "
        "'n', 'workflow', 'd', 'i', CAST(:tr AS json), '[]', '0.1.0', '{}', FALSE)",
        params={"o": ORG, "tr": json.dumps(["\u201ccool\u201d the apple\u2014then"])},
    )

    async def lexemes():
        rows = await _fetch(
            "SELECT array_to_string(tsvector_to_array(retrieval_tsv), ' ') AS l "
            "FROM procedural_memory WHERE id = 'proc-escaped'"
        )
        return set(rows[0]["l"].split())

    assert {"u201ccool", "u2014then"} <= await lexemes()  # the bug being fixed
    assert await _pending_data_scripts() == ["scripts/migrate_skill_retrieval_fields.sql"]
    await _run_script("migrate_skill_retrieval_fields.sql")
    assert await _pending_data_scripts() == []  # The guard is satisfied
    assert {"cool", "appl"} <= await lexemes()  # ("then" is a stop word)
    assert not any(lexeme.startswith("u20") for lexeme in await lexemes())
    assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == fresh_snapshot
    attnum = (
        "SELECT attnum FROM pg_attribute WHERE attrelid = "
        "'procedural_memory'::regclass AND attname = 'retrieval_tsv'"
    )
    rebuilt = await _fetch(attnum)
    await _run_script("migrate_skill_retrieval_fields.sql")  # a no-op now
    assert await _schema_snapshot(NEW_TABLES + CHANGED_TABLES) == fresh_snapshot
    assert await _fetch(attnum) == rebuilt  # not rebuilt a second time
    await _execute(
        "DELETE FROM procedural_memory WHERE id = 'proc-escaped'",
        "DELETE FROM users WHERE id = 'user-escaped'",
    )


@pytest.mark.parametrize("column_type", ["json", "jsonb"])
async def test_a_jsonb_text_tsv_is_rebuilt_once_without_escape_junk(
    fresh_snapshot, column_type
):
    """An earlier expression, `triggers::jsonb::text`, still
    indexed jsonb's escapes of control characters and quotes — a trigger
    'line1<newline>line2' became the lexeme 'nline2', '<tab>' glued a 't' to
    the next word — on a JSON column and on an upgraded JSONB one alike. The
    script rebuilds such a column once (on both column types), after which
    the words are words, and a re-run leaves it alone."""
    await _reset_to_fresh_schema()
    await _execute("ALTER TABLE procedural_memory DROP COLUMN retrieval_tsv")
    if column_type == "jsonb":
        await _execute(
            "ALTER TABLE procedural_memory ALTER COLUMN triggers TYPE JSONB "
            "USING triggers::jsonb"
        )
    await _execute(
        "ALTER TABLE procedural_memory ADD COLUMN retrieval_tsv tsvector GENERATED "
        "ALWAYS AS (setweight(to_tsvector('english', coalesce(name, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(triggers::jsonb::text, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(description, '')), 'B')) STORED",
        "CREATE INDEX ix_procedural_memory_retrieval_tsv ON procedural_memory "
        "USING gin (retrieval_tsv)",
        "INSERT INTO organizations (id, name) VALUES ('org-ctl', 'org-ctl')",
        "INSERT INTO users (id, name, organization_id, status, timezone, is_admin) "
        "VALUES ('user-ctl', 'u', 'org-ctl', 'active', 'UTC', FALSE)",
    )
    await _execute(
        "INSERT INTO procedural_memory (id, organization_id, user_id, name, "
        "entry_type, description, instructions, triggers, examples, version, "
        "last_modify, is_deleted) VALUES ('proc-ctl', 'org-ctl', 'user-ctl', "
        f"'n', 'workflow', 'd', 'i', CAST(:tr AS {column_type}), '[]', '0.1.0', "
        "'{}', FALSE)",
        params={
            "tr": json.dumps(
                ["unpack boxes\nlabel shelves", "tab\tbudget", 'say "hello"']
            )
        },
    )

    async def lexemes():
        rows = await _fetch(
            "SELECT array_to_string(tsvector_to_array(retrieval_tsv), ' ') AS l "
            "FROM procedural_memory WHERE id = 'proc-ctl'"
        )
        return set(rows[0]["l"].split())

    assert {"nlabel", "tbudget"} <= await lexemes()  # the bug being fixed
    attnum = (
        "SELECT attnum FROM pg_attribute WHERE attrelid = "
        "'procedural_memory'::regclass AND attname = 'retrieval_tsv'"
    )
    before = await _fetch(attnum)
    # Startup names the script (the column exists, so only the
    # data check can tell).
    assert await _pending_data_scripts() == ["scripts/migrate_skill_retrieval_fields.sql"]
    await _run_script("migrate_skill_retrieval_fields.sql")
    assert await _pending_data_scripts() == []
    words = await lexemes()
    assert {"unpack", "box", "label", "shelv", "tab", "budget", "say", "hello"} <= words
    assert not ({"nlabel", "tbudget"} & words), words
    rebuilt = await _fetch(attnum)
    assert rebuilt != before  # dropped and added again
    await _run_script("migrate_skill_retrieval_fields.sql")  # a no-op now
    assert await _fetch(attnum) == rebuilt
    snapshot = await _schema_snapshot(NEW_TABLES + CHANGED_TABLES)
    if column_type == "json":
        assert snapshot == fresh_snapshot
    else:
        tsv = snapshot["columns"][("procedural_memory", "retrieval_tsv")]
        assert "regexp_replace((triggers)::text" in tsv["generation_expression"]


# --------------------------------------------------------------------------- #
# The application on the migrated schema                                      #
# --------------------------------------------------------------------------- #


async def test_managers_work_on_the_migrated_schema(fresh_snapshot):
    from fakes.fake_llm_server import FakeLLMServer

    import mirix.services.procedural_memory_manager as pmm
    from mirix.schemas.agent import AgentState, AgentType
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.embedding_config import EmbeddingConfig
    from mirix.schemas.llm_config import LLMConfig
    from mirix.schemas.user import User as PydanticUser

    await _build_main_schema()
    await _seed_main_era_data()
    await _run_all_scripts()
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    await _execute(
        "INSERT INTO clients (id, organization_id, name, status, write_scope, read_scopes) "
        f"VALUES ('{client_id}', '{ORG}', 'c', 'active', 'test', '[\"test\"]')"
    )

    # Every embedding call goes to the local fake (token-hash vectors, so texts
    # sharing words are similar); nothing can reach a real provider.
    fake = FakeLLMServer(embedding_mode="token_hash", embedding_dim=64).start()
    agent_state = AgentState(
        id=META_AGENT,
        name="meta",
        system="s",
        agent_type=AgentType.meta_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="hugging-face",
            embedding_endpoint=fake.base_url,
            embedding_model="fake-embedding",
            embedding_dim=64,
        ),
        tools=[],
    )
    user = PydanticUser(id=USER_A, name="a", organization_id=ORG, timezone="UTC")
    actor = PydanticClient(
        id=client_id,
        organization_id=ORG,
        name="c",
        write_scope="test",
        read_scopes=["test"],
    )
    mgr = pmm.ProceduralMemoryManager()
    try:
        await _exercise_managers(mgr, agent_state, user, actor, fake)
    finally:
        fake.stop()


async def _exercise_managers(mgr, agent_state, user, actor, fake):
    created = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=META_AGENT,
        name="kube-restart",
        description="Restart kube workloads",
        instructions="Use rollout restart, then check readiness probes",
        entry_type="workflow",
        actor=actor,
        organization_id=ORG,
        user_id=USER_A,
        triggers=["pods stuck"],
        use_cache=False,
    )
    stored = await _fetch(
        "SELECT retrieval_embedding IS NOT NULL AS has_vector, retrieval_tsv::text AS tsv "
        "FROM procedural_memory WHERE id = :id",
        id=created.id,
    )
    assert stored[0]["has_vector"]
    assert "'stuck'" in stored[0]["tsv"]  # triggers are in the generated document
    # The vector embeds the retrieval text (name, triggers, description),
    # computed by the fake.
    assert [r.json["input"] for r in fake.embedding_requests()] == [
        "kube-restart\npods stuck\nRestart kube workloads"
    ]

    # On Postgres an embedding search on instructions is full-text search.
    by_instructions = await mgr.list_procedures(
        agent_state=agent_state,
        user=user,
        query="readiness probes",
        search_field="instructions",
        search_method="embedding",
        use_cache=False,
    )
    assert [s.id for s in by_instructions] == [created.id]
    assert (
        len(fake.embedding_requests()) == 1
    )  # the instructions search embedded nothing

    # Vector search runs on retrieval_embedding (pgvector cosine).
    by_vector = await mgr.list_procedures(
        agent_state=agent_state,
        user=user,
        query="kube",
        search_field="description",
        search_method="embedding",
        use_cache=False,
    )
    assert by_vector[0].id == created.id
    assert fake.embedding_requests()[-1].json["input"] == "kube"

    # Name reuse through the manager: soft-delete, then re-create the same name.
    await mgr.soft_delete_procedure_by_id(created.id, actor=actor, user=user)
    again = await mgr.insert_procedure(
        agent_state=agent_state,
        agent_id=META_AGENT,
        name="kube-restart",
        description="Restart kube workloads, v2",
        instructions="i",
        entry_type="workflow",
        actor=actor,
        organization_id=ORG,
        user_id=USER_A,
        use_cache=False,
    )
    assert again.id != created.id
