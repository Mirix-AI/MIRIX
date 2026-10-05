"""Tests for the startup schema-drift guard.

create_all only creates missing tables — it never ALTERs existing ones — so an
operator who upgrades without running the manual migration scripts would get
UndefinedColumn errors on every request. ensure_tables_created() detects that
drift at startup and aborts with the exact scripts to run.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

import mirix.orm  # noqa: F401 -- register every table on Base.metadata
from mirix.orm.base import Base
from mirix.server.server import (
    _MIGRATION_HINTS,
    _describe_schema_drift,
    _find_missing_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def sqlite_conn():
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        yield conn
    engine.dispose()


def test_no_drift_on_empty_database(sqlite_conn):
    """Tables that don't exist yet are create_all's job, not drift."""
    assert _find_missing_columns(sqlite_conn) == {}


def test_detects_missing_session_id_on_legacy_messages_table(sqlite_conn):
    """A pre-upgrade messages table (no session_id) must be reported."""
    sqlite_conn.execute(text("CREATE TABLE messages (id VARCHAR PRIMARY KEY)"))
    sqlite_conn.commit()

    missing = _find_missing_columns(sqlite_conn)

    assert "messages" in missing
    assert "session_id" in missing["messages"]


def test_detects_missing_skill_columns_on_legacy_procedural_table(sqlite_conn):
    """A pre-skill procedural_memory table (summary/steps era) must be reported."""
    sqlite_conn.execute(
        text(
            "CREATE TABLE procedural_memory ("
            "id VARCHAR PRIMARY KEY, entry_type VARCHAR, summary VARCHAR, steps JSON)"
        )
    )
    sqlite_conn.commit()

    missing = _find_missing_columns(sqlite_conn)

    assert "procedural_memory" in missing
    for column in ("name", "description", "instructions", "version"):
        assert column in missing["procedural_memory"]


def test_detects_pre_retrieval_procedural_table(sqlite_conn):
    """A skill-era procedural_memory with the old per-field vectors is drift."""
    sqlite_conn.execute(
        text(
            "CREATE TABLE procedural_memory (id VARCHAR PRIMARY KEY, name VARCHAR, entry_type VARCHAR, "
            "description VARCHAR, instructions VARCHAR, triggers JSON, examples JSON, version VARCHAR, "
            "description_embedding BLOB, instructions_embedding BLOB)"
        )
    )
    sqlite_conn.commit()

    missing = _find_missing_columns(sqlite_conn)

    assert "retrieval_embedding" in missing["procedural_memory"]
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert "run scripts/migrate_skill_retrieval_fields.sql" in report
    assert "migrate_procedural_to_skill.sql" not in report  # skill columns are present


def test_pre_139_conversation_message_points_at_the_session_tag_script(sqlite_conn):
    """#139 added session_tag without a drift hint; it has one now."""
    sqlite_conn.execute(
        text(
            "CREATE TABLE conversation_message (id VARCHAR PRIMARY KEY, session_id VARCHAR, role VARCHAR, "
            "content TEXT, distilled_at TIMESTAMP, user_id VARCHAR, organization_id VARCHAR, "
            "created_at TIMESTAMP, updated_at TIMESTAMP, is_deleted BOOLEAN, "
            "_created_by_id VARCHAR, _last_updated_by_id VARCHAR)"
        )
    )
    sqlite_conn.commit()

    missing = _find_missing_columns(sqlite_conn)

    assert missing["conversation_message"] == ["session_tag", "content_parts", "original_text_length"]
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert (
        "run scripts/migrate_add_conversation_message_session_tag.sql, "
        "then scripts/migrate_conversation_message_parts.sql" in report
    )


def test_every_hint_names_an_existing_script_that_adds_the_column():
    """(table, column) -> script: the script exists and mentions both names."""
    for (table_name, column_name), hint in _MIGRATION_HINTS.items():
        script = REPO_ROOT / hint.split(" ", 1)[0]
        assert script.is_file(), f"{table_name}.{column_name}: missing {script}"
        sql = script.read_text()
        assert table_name in sql, f"{script.name} never mentions table {table_name}"
        assert column_name in sql, f"{script.name} never mentions column {column_name}"


def test_every_hint_is_an_orm_column():
    """No stale hints: each registered (table, column) exists on the ORM.

    retrieval_tsv is Postgres-only, so it is absent from the SQLite-mode metadata
    this unit suite imports (it is asserted by the Postgres integration test).
    """
    for table_name, column_name in _MIGRATION_HINTS:
        if (table_name, column_name) == ("procedural_memory", "retrieval_tsv"):
            continue
        assert table_name in Base.metadata.tables, table_name
        assert column_name in Base.metadata.tables[table_name].c, f"{table_name}.{column_name}"


def test_hints_cover_every_column_added_to_an_existing_table():
    """Each column a migration ADDs (or renames into place) on an existing table has a hint."""
    for key in [
        ("messages", "session_id"),
        ("agent_trigger_state", "last_fired_tied_session_ids"),
        ("procedural_memory", "name"),
        ("procedural_memory", "description"),
        ("procedural_memory", "instructions"),
        ("procedural_memory", "version"),
        ("procedural_memory", "retrieval_embedding"),
        ("procedural_memory", "retrieval_tsv"),
        ("conversation_message", "session_tag"),
        ("conversation_message", "content_parts"),
        ("conversation_message", "original_text_length"),
        ("files", "user_id"),
        ("skill_experience", "curation_attempts"),
        # The experience pool.
        ("skill_experience", "disposition"),
        ("skill_experience", "cited_skill_ids"),
        ("skill_experience", "pool_scope"),
        ("skill_experience", "pool_embedding"),
        ("skill_experience", "pool_embedding_model"),
        ("skill_experience", "pool_embedding_failures"),
        ("skill_experience", "recurrence_count"),
        ("skill_experience", "ignored_pushes"),
        ("skill_experience", "pushed_count"),
        ("skill_experience", "last_pushed_at"),
        ("skill_experience", "pool_retired_at"),
        ("skill_experience", "pool_retire_reason"),
    ]:
        assert key in _MIGRATION_HINTS, key


def test_the_pool_script_comes_after_the_earlier_skill_experience_script():
    """A database missing both runs the attempts script first (the hints'
    insertion order is the order the drift report names the scripts in)."""
    lines = _describe_schema_drift(
        {"skill_experience": ["pool_scope", "curation_attempts", "disposition"]},
        "postgresql",
    )
    assert lines == [
        "  - table 'skill_experience' is missing columns ['pool_scope', "
        "'curation_attempts', 'disposition']; run "
        "scripts/migrate_skill_experience_attempts.sql, then "
        "scripts/migrate_skill_experience_pool.sql"
    ]


def test_drift_report_orders_scripts_and_flags_unregistered_columns():
    lines = _describe_schema_drift(
        {"procedural_memory": ["retrieval_embedding", "name", "mystery"]},
        "postgresql",
    )
    assert lines == [
        "  - table 'procedural_memory' is missing columns ['retrieval_embedding', 'name', 'mystery']; "
        "run scripts/migrate_procedural_to_skill.sql, then scripts/migrate_skill_retrieval_fields.sql; "
        "no registered script for ['mystery']: see scripts/migrate_*.sql"
    ]


def test_drift_report_adds_a_sqlite_note():
    lines = _describe_schema_drift({"files": ["user_id"]}, "sqlite")
    assert lines[0].endswith("run scripts/migrate_conversation_message_parts.sql")
    assert "PostgreSQL" in lines[1] and "SQLite" in lines[1]
    assert len(_describe_schema_drift({"files": ["user_id"]}, "postgresql")) == 1


@pytest.mark.asyncio
async def test_ensure_tables_created_raises_with_the_per_column_report(monkeypatch, tmp_path):
    """End-to-end on SQLite: an old files table aborts startup with the hint."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import mirix.server.server as server_module

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'old.db'}")
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE files (id VARCHAR PRIMARY KEY, organization_id VARCHAR)"))
    monkeypatch.setattr(server_module, "engine", engine)
    monkeypatch.setattr(server_module, "USE_PGLITE", False)
    try:
        with pytest.raises(RuntimeError) as exc:
            await server_module.ensure_tables_created()
    finally:
        await engine.dispose()
    message = str(exc.value)
    assert "table 'files' is missing columns" in message
    assert "scripts/migrate_conversation_message_parts.sql" in message


# --------------------------------------------------------------------------- #
# Data backfills and the script registry                                       #
# --------------------------------------------------------------------------- #

# Scripts of the pre-branch base (main 8cb06a6) that create tables create_all
# also builds, or index/backfill passes an operator runs once by hand; none of
# them adds a column the ORM would miss, so none can be drift.
_BASE_SCRIPTS_WITHOUT_A_DRIFT_SIGNAL = {
    "scripts/migrate_add_conversation_message.sql",  # CREATE TABLE
    "scripts/migrate_add_conversation_message_phase2.sql",  # CONCURRENTLY indexes
    "scripts/migrate_add_skill_experience.sql",  # CREATE TABLE
    "scripts/migrate_backfill_procedural_scope.sql",  # filter_tags backfill
    "scripts/migrate_add_message_session_id_phase2.sql",  # named in the hint text
}


def _registered_scripts():
    from mirix.server.server import _DATA_MIGRATION_CHECKS, _TABLE_MIGRATION_SCRIPTS

    scripts = {hint.split(" ", 1)[0] for hint in _MIGRATION_HINTS.values()}
    scripts.update(_TABLE_MIGRATION_SCRIPTS.values())
    scripts.update(script for _, script, _ in _DATA_MIGRATION_CHECKS)
    return scripts


def test_every_migration_script_is_registered():
    """Each scripts/migrate_*.sql is known to the guard (a column
    hint, a table it creates, or a data backfill it performs)."""
    on_disk = {
        f"scripts/{p.name}" for p in (REPO_ROOT / "scripts").glob("migrate_*.sql")
    }
    unregistered = on_disk - _registered_scripts() - _BASE_SCRIPTS_WITHOUT_A_DRIFT_SIGNAL
    assert unregistered == set()
    for script in _registered_scripts():
        assert (REPO_ROOT / script).is_file(), script


def test_every_registered_table_is_an_orm_table_its_script_creates():
    from mirix.server.server import _TABLE_MIGRATION_SCRIPTS

    for table_name, script in _TABLE_MIGRATION_SCRIPTS.items():
        assert table_name in Base.metadata.tables, table_name
        sql = (REPO_ROOT / script).read_text()
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in sql, (table_name, script)


def test_the_conversation_session_backfill_is_a_data_check():
    from mirix.server.server import _DATA_MIGRATION_CHECKS

    scripts = [script for _, script, _ in _DATA_MIGRATION_CHECKS]
    assert "scripts/migrate_add_conversation_session.sql" in scripts


def test_an_outdated_retrieval_tsv_is_a_data_check_postgres_only(sqlite_conn):
    """The column exists, so only a data check can name the script
    that rebuilds a retrieval_tsv of an earlier expression (the PG side is in
    tests/test_skill_data_layer_pg.py); SQLite has no such column."""
    from mirix.server.server import _DATA_MIGRATION_CHECKS, _retrieval_tsv_is_outdated

    [(description, script)] = [
        (d, s) for d, s, probe in _DATA_MIGRATION_CHECKS if probe is _retrieval_tsv_is_outdated
    ]
    assert script == "scripts/migrate_skill_retrieval_fields.sql"
    assert "retrieval_tsv" in description
    sql = (REPO_ROOT / script).read_text()
    assert "NOT LIKE '%regexp_replace%'" in sql  # the same predicate as the probe
    Base.metadata.create_all(sqlite_conn)
    assert _retrieval_tsv_is_outdated(sqlite_conn) is False


def _add_tenant(conn):
    from sqlalchemy.orm import Session

    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.orm.user import User as UserORM

    with Session(bind=conn) as session:
        session.add(OrganizationORM(id="org-1", name="org-1"))
        for user_id in ("user-1", "user-2"):
            session.add(
                UserORM(
                    id=user_id,
                    name=user_id,
                    organization_id="org-1",
                    status="active",
                    timezone="UTC",
                )
            )
        session.flush()


def _add_turn(conn, *, session_id, is_deleted=False, user_id="user-1"):
    conn.execute(
        text(
            "INSERT INTO conversation_message (id, organization_id, user_id, "
            "session_id, role, content, created_at, updated_at, is_deleted) "
            "VALUES (:id, 'org-1', :user_id, :sid, 'user', 'hi', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :deleted)"
        ),
        {
            "id": f"cm-{session_id}-{is_deleted}",
            "user_id": user_id,
            "sid": session_id,
            "deleted": is_deleted,
        },
    )


def _add_session_row(conn, *, session_id, user_id="user-1"):
    conn.execute(
        text(
            "INSERT INTO conversation_session (id, organization_id, user_id, "
            "session_id, message_count, reopen_count, distill_failures, "
            "first_message_at, last_message_at, created_at, updated_at, "
            "is_deleted) VALUES (:id, 'org-1', :user_id, :sid, 1, 0, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, "
            "CURRENT_TIMESTAMP, 0)"
        ),
        {"id": f"cs-{user_id}-{session_id}", "user_id": user_id, "sid": session_id},
    )


def test_pending_conversation_session_backfill_is_detected(sqlite_conn):
    from mirix.server.server import (
        _conversation_sessions_need_backfill,
        _find_pending_data_migrations,
    )

    Base.metadata.create_all(sqlite_conn)
    _add_tenant(sqlite_conn)
    assert _find_pending_data_migrations(sqlite_conn) == []
    # Only a soft-deleted turn: nothing the backfill would create.
    _add_turn(sqlite_conn, session_id="gone", is_deleted=True)
    assert _conversation_sessions_need_backfill(sqlite_conn) is False
    # A live pre-upgrade turn without its session row: the backfill is pending.
    _add_turn(sqlite_conn, session_id="old-task")
    [(description, script)] = _find_pending_data_migrations(sqlite_conn)
    assert script == "scripts/migrate_add_conversation_session.sql"
    assert "conversation_session" in description
    # Another user's row for the same session_id does not count.
    _add_session_row(sqlite_conn, session_id="old-task", user_id="user-2")
    assert _conversation_sessions_need_backfill(sqlite_conn) is True
    _add_session_row(sqlite_conn, session_id="old-task")
    assert _find_pending_data_migrations(sqlite_conn) == []


def _add_null_org_turn(conn, *, session_id):
    conn.execute(
        text(
            "INSERT INTO conversation_message (id, organization_id, user_id, "
            "session_id, role, content, created_at, updated_at, is_deleted) "
            "VALUES (:id, NULL, 'user-1', :sid, 'user', 'hi', "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0)"
        ),
        {"id": f"cm-null-{session_id}", "sid": session_id},
    )


def _add_null_org_session_row(conn, *, session_id):
    conn.execute(
        text(
            "INSERT INTO conversation_session (id, organization_id, user_id, "
            "session_id, message_count, reopen_count, distill_failures, "
            "first_message_at, last_message_at, created_at, updated_at, "
            "is_deleted) VALUES (:id, NULL, 'user-1', :sid, 1, 0, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, "
            "CURRENT_TIMESTAMP, 0)"
        ),
        {"id": f"cs-null-{session_id}", "sid": session_id},
    )


def test_a_null_org_turn_is_matched_null_safely(sqlite_conn):
    """The probe matches organization_id with IS NOT DISTINCT FROM,
    like the backfill script — a NULL-org live turn whose session row the
    backfill created must not abort every startup."""
    from mirix.server.server import _conversation_sessions_need_backfill

    Base.metadata.create_all(sqlite_conn)
    _add_tenant(sqlite_conn)
    _add_null_org_turn(sqlite_conn, session_id="orgless")
    # No session row yet: the backfill is genuinely pending.
    assert _conversation_sessions_need_backfill(sqlite_conn) is True
    # A row of ANOTHER organization is not this session's row.
    _add_session_row(sqlite_conn, session_id="orgless")
    assert _conversation_sessions_need_backfill(sqlite_conn) is True
    # The row the backfill writes for it (organization_id NULL) satisfies it.
    _add_null_org_session_row(sqlite_conn, session_id="orgless")
    assert _conversation_sessions_need_backfill(sqlite_conn) is False


def test_no_data_check_before_the_tables_exist(sqlite_conn):
    from mirix.server.server import _find_pending_data_migrations

    assert _find_pending_data_migrations(sqlite_conn) == []


def test_drift_report_lists_pending_data_after_the_columns():
    lines = _describe_schema_drift(
        {"files": ["user_id"]},
        "postgresql",
        [("sessions lack rows", "scripts/migrate_add_conversation_session.sql")],
    )
    assert lines == [
        "  - table 'files' is missing columns ['user_id']; "
        "run scripts/migrate_conversation_message_parts.sql",
        "  - data: sessions lack rows; run scripts/migrate_add_conversation_session.sql",
    ]
    assert _describe_schema_drift({}, "postgresql", []) == []


@pytest.mark.asyncio
async def test_ensure_tables_created_aborts_on_a_pending_backfill(monkeypatch, tmp_path):
    """End-to-end on SQLite: the current schema, but pre-upgrade turns without
    session rows — startup names the backfill script instead of starting with
    every legacy session unlearnable."""
    from sqlalchemy.ext.asyncio import create_async_engine

    import mirix.server.server as server_module

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'upgraded.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_tenant)
        await conn.run_sync(lambda c: _add_turn(c, session_id="legacy"))
    monkeypatch.setattr(server_module, "engine", engine)
    monkeypatch.setattr(server_module, "USE_PGLITE", False)
    try:
        with pytest.raises(RuntimeError) as exc:
            await server_module.ensure_tables_created()
        message = str(exc.value)
        assert "data:" in message
        assert "scripts/migrate_add_conversation_session.sql" in message
        assert "missing columns" not in message
        # Once the backfill ran, startup passes.
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: _add_session_row(c, session_id="legacy"))
        await server_module.ensure_tables_created()
    finally:
        await engine.dispose()


def test_a_session_without_its_agent_context_column_is_drift(sqlite_conn):
    """conversation_session.agent_context is added by a script;
    skill_session_feedback is a table it creates (create_all builds it)."""
    from mirix.server.server import _TABLE_MIGRATION_SCRIPTS

    sqlite_conn.execute(
        text("CREATE TABLE conversation_session (id VARCHAR PRIMARY KEY, session_id VARCHAR)")
    )
    sqlite_conn.commit()
    missing = _find_missing_columns(sqlite_conn)
    assert "agent_context" in missing["conversation_session"]
    report = "\n".join(_describe_schema_drift(missing, "postgresql"))
    assert "scripts/migrate_p16_agent_context_and_skill_feedback.sql" in report
    assert (
        _TABLE_MIGRATION_SCRIPTS["skill_session_feedback"]
        == "scripts/migrate_p16_agent_context_and_skill_feedback.sql"
    )
