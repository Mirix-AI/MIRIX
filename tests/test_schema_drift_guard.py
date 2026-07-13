"""Tests for the startup schema-drift guard.

create_all only creates missing tables — it never ALTERs existing ones — so an
operator who upgrades without running the manual migration scripts would get
UndefinedColumn errors on every request. ensure_tables_created() detects that
drift at startup and aborts with the exact scripts to run.
"""

import pytest
from sqlalchemy import create_engine, text

from mirix.server.server import _MIGRATION_HINTS, _find_missing_columns


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


def test_migration_hints_cover_all_migration_scripted_tables():
    """Every table with a manual migration script has an actionable hint."""
    for table_name in (
        "messages",
        "procedural_memory",
        "conversation_message",
        "skill_experience",
        "agent_trigger_state",
    ):
        assert table_name in _MIGRATION_HINTS
        assert "scripts/migrate_" in _MIGRATION_HINTS[table_name]
