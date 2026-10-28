"""SQL-text tests for the skill-learning migration scripts.

Pattern of tests/test_agent_trigger_state.py::TestMigrationSql: no database,
just the script text. Every script must be idempotent (IF [NOT] EXISTS or a
guarded DO block for every DDL statement), transactional, free of CONCURRENTLY
(which cannot run inside the transaction), and in lock-step with the ORM:
new tables declare exactly the ORM's columns and index names. Behavior against
a real Postgres (both paths, run twice) is in tests/test_skill_data_layer_pg.py.
The scripts the server names at startup exist, and that Postgres test upgrades
with every new script.
"""

import re
from pathlib import Path

import pytest

import mirix.orm  # noqa: F401 -- register every table on Base.metadata
from mirix.orm.base import Base
from mirix.orm.procedural_memory import RETRIEVAL_TSV_EXPRESSION, UNIQUE_LIVE_NAME_INDEX
from mirix.schemas.skill_version import SKILL_VERSIONED_FIELDS

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

SESSION_SQL = "migrate_add_conversation_session.sql"
PARTS_SQL = "migrate_conversation_message_parts.sql"
RETRIEVAL_SQL = "migrate_skill_retrieval_fields.sql"
HISTORY_SQL = "migrate_add_skill_history.sql"
ATTEMPTS_SQL = "migrate_skill_experience_attempts.sql"
RUN_LOG_SQL = "migrate_add_learning_run_and_retrieval_log.sql"
POOL_SQL = "migrate_skill_experience_pool.sql"
P16_SQL = "migrate_p16_agent_context_and_skill_feedback.sql"
ALL_SCRIPTS = [
    SESSION_SQL,
    PARTS_SQL,
    RETRIEVAL_SQL,
    HISTORY_SQL,
    ATTEMPTS_SQL,
    RUN_LOG_SQL,
    POOL_SQL,
    P16_SQL,
]


def _sql(name: str) -> str:
    return (SCRIPTS / name).read_text()


def _code(name: str) -> str:
    """The script with `--` comments stripped (what Postgres executes)."""
    return "\n".join(line.split("--", 1)[0] for line in _sql(name).splitlines())


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def _create_table_columns(code: str, table: str) -> set:
    match = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", code, re.S)
    assert match, f"no CREATE TABLE IF NOT EXISTS {table}"
    columns = set()
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.upper().startswith(("PRIMARY KEY", "CONSTRAINT", "UNIQUE")):
            continue
        columns.add(line.split()[0].strip('"'))
    return columns


def _orm_index_names(table: str) -> set:
    return {ix.name for ix in Base.metadata.tables[table].indexes}


# --------------------------------------------------------------------------- #
# Every script                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ALL_SCRIPTS)
class TestEveryScript:
    def test_exists_and_is_one_transaction(self, name):
        code = _code(name)
        assert re.search(r"^BEGIN;", code, re.M)
        assert code.rstrip().endswith("COMMIT;")

    def test_no_concurrently_inside_the_transaction(self, name):
        assert "CONCURRENTLY" not in _code(name)

    def test_ddl_is_idempotent(self, name):
        code = _norm(_code(name))
        for statement in re.findall(
            r"CREATE (?:UNIQUE )?(?:INDEX|TABLE) \S+ \S+ \S+", code
        ):
            assert "IF NOT EXISTS" in statement, statement
        for statement in re.findall(r"ADD COLUMN \S+ \S+ \S+", code):
            assert statement.startswith("ADD COLUMN IF NOT EXISTS"), statement
        for statement in re.findall(r"DROP (?:COLUMN|INDEX|TABLE) \S+ \S+", code):
            assert "IF EXISTS" in statement, statement
        # Constraints/renames have no IF NOT EXISTS: each must sit in a guarded DO block.
        for keyword in ("ADD CONSTRAINT", "DROP CONSTRAINT", "RENAME COLUMN"):
            for match in re.finditer(keyword, code):
                block_start = code.rfind("DO $$", 0, match.start())
                block_end = code.find("END$$", match.start())
                assert block_start != -1 and block_end != -1, (
                    f"{keyword} outside a DO block"
                )
                assert re.search(
                    r"IF (NOT )?EXISTS", code[block_start : match.start()]
                ), keyword

    def test_header_documents_idempotence(self, name):
        header = _sql(name).split("BEGIN;", 1)[0]
        assert "Idempotent" in header
        assert "create_all" in header  # explains the fresh-database path


# --------------------------------------------------------------------------- #
# migrate_add_conversation_session.sql                                        #
# --------------------------------------------------------------------------- #


class TestConversationSessionSql:
    def test_infra_failure_counter_is_created_and_added(self):
        """A table from an earlier revision of the script gets it too."""
        code = _norm(_code(SESSION_SQL))
        assert "distill_infra_failures INTEGER NOT NULL DEFAULT 0," in code
        assert (
            "ALTER TABLE conversation_session ADD COLUMN IF NOT EXISTS "
            "distill_infra_failures INTEGER NOT NULL DEFAULT 0;"
        ) in code

    def test_table_matches_the_orm(self):
        orm_columns = {
            c.name for c in Base.metadata.tables["conversation_session"].columns
        }
        # agent_context came later: migrate_p16_agent_context_and_skill_feedback.sql.
        added_later = set(
            re.findall(
                r"ALTER TABLE conversation_session ADD COLUMN IF NOT EXISTS (\w+)",
                _code(P16_SQL),
            )
        )
        assert added_later == {"agent_context"}
        assert (
            _create_table_columns(_code(SESSION_SQL), "conversation_session")
            | added_later
            == orm_columns
        )

    def test_constraints_and_indexes_match_the_orm(self):
        code = _code(SESSION_SQL)
        assert "uq_conversation_session_org_user_session" in code
        assert "UNIQUE (organization_id, user_id, session_id)" in code
        assert "ck_conversation_session_session_id_format" in code
        assert "session_id ~ '^[A-Za-z0-9_-]{1,64}$'" in code
        for index in _orm_index_names("conversation_session"):
            assert f"CREATE INDEX IF NOT EXISTS {index}" in code, index

    def test_backfill_groups_live_rows_per_org_user_session(self):
        code = _norm(_code(SESSION_SQL))
        assert "FROM conversation_message cm WHERE cm.is_deleted IS NOT TRUE" in code
        assert "GROUP BY cm.organization_id, cm.user_id, cm.session_id" in code
        assert (
            "MIN(cm.created_at)" in code
            and "MAX(cm.created_at)" in code
            and "COUNT(*)" in code
        )

    def test_backfill_takes_the_earliest_rows_tag(self):
        code = _norm(_code(SESSION_SQL))
        assert "(array_agg(cm.session_tag ORDER BY cm.created_at, cm.id))[1]" in code

    def test_backfill_closes_idle_sessions_with_the_default_timeout(self):
        code = _norm(_code(SESSION_SQL))
        assert "current_setting('mirix.session_idle_timeout_seconds', true)" in code
        assert "172800" in code
        assert "THEN s.last_at + cfg.idle END" in code
        assert "THEN 'idle' END" in code

    def test_backfill_attributes_sessions_to_the_meta_agent(self):
        code = _norm(_code(SESSION_SQL))
        assert "a.agent_type = 'meta_memory_agent'" in code
        # One set-based pass over messages, restricted to backfilled sessions.
        assert "SELECT DISTINCT ON (m.user_id, m.session_id)" in code
        assert (
            "(m.user_id, m.session_id) IN (SELECT user_id, session_id FROM sessions)"
            in code
        )
        assert (
            "LEFT JOIN meta_agents ma ON ma.user_id = s.user_id AND ma.session_id = s.session_id"
            in code
        )

    def test_backfill_falls_back_to_the_sole_writers_meta_agent(self):
        """Without agent-loop history, the only live meta agent of
        the only writing client — or, with no recorded writer, of the
        organization; otherwise NULL (the same rule as the runtime adoption)."""
        code = _norm(_code(SESSION_SQL))
        assert (
            "COALESCE( ma.agent_id, CASE WHEN s.n_writers = 1 THEN oa.agent_id END, "
            "CASE WHEN s.n_writers = 0 THEN ga.agent_id END )" in code
        )
        assert "LEFT JOIN owner_agent oa ON oa.owner = s.writer" in code
        assert (
            "LEFT JOIN org_agent ga ON ga.organization_id = s.organization_id" in code
        )
        for cte in ("live_meta", "owner_agent", "org_agent"):
            assert f"{cte} AS (" in code, cte

    def test_backfill_keeps_only_valid_tags(self):
        code = _norm(_code(SESSION_SQL))
        assert (
            "CASE WHEN s.first_tag IN ('task', 'conversation') THEN s.first_tag END"
            in code
        )

    def test_backfill_is_rerunnable(self):
        code = _norm(_code(SESSION_SQL))
        assert "WHERE NOT EXISTS ( SELECT 1 FROM conversation_session cs" in code
        assert "cs.organization_id IS NOT DISTINCT FROM s.organization_id" in code
        assert "ON CONFLICT (organization_id, user_id, session_id) DO NOTHING" in code


# --------------------------------------------------------------------------- #
# migrate_conversation_message_parts.sql                                      #
# --------------------------------------------------------------------------- #


class TestConversationMessagePartsSql:
    def test_adds_the_two_message_columns(self):
        code = _norm(_code(PARTS_SQL))
        assert (
            "ALTER TABLE conversation_message ADD COLUMN IF NOT EXISTS content_parts JSON NULL"
            in code
        )
        assert (
            "ALTER TABLE conversation_message ADD COLUMN IF NOT EXISTS original_text_length INTEGER NULL"
            in code
        )

    def test_adds_files_user_id_with_set_null_fk_and_index(self):
        code = _norm(_code(PARTS_SQL))
        assert (
            "ALTER TABLE files ADD COLUMN IF NOT EXISTS user_id VARCHAR NULL REFERENCES users(id) ON DELETE SET NULL"
            in code
        )
        assert "CREATE INDEX IF NOT EXISTS ix_files_user_id ON files (user_id)" in code
        assert "ix_files_user_id" in _orm_index_names("files")


# --------------------------------------------------------------------------- #
# migrate_skill_retrieval_fields.sql                                          #
# --------------------------------------------------------------------------- #


class TestSkillRetrievalFieldsSql:
    def _rename_block(self) -> str:
        code = _code(RETRIEVAL_SQL)
        start = code.index("DO $$")
        return code[start : code.index("END$$", start)]

    def test_rename_and_clear_happen_once_in_one_guarded_block(self):
        block = _norm(self._rename_block())
        assert "column_name = 'description_embedding'" in block
        assert (
            "AND NOT EXISTS" in block and "column_name = 'retrieval_embedding'" in block
        )
        assert "RENAME COLUMN description_embedding TO retrieval_embedding" in block
        assert "UPDATE procedural_memory SET retrieval_embedding = NULL" in block
        # The clear must never run outside the guard (it would wipe recomputed vectors).
        assert _code(RETRIEVAL_SQL).count("SET retrieval_embedding = NULL") == 1

    def test_drops_the_per_field_vectors(self):
        code = _norm(_code(RETRIEVAL_SQL))
        assert "DROP COLUMN IF EXISTS instructions_embedding" in code
        assert "DROP COLUMN IF EXISTS description_embedding" in code
        assert "ADD COLUMN IF NOT EXISTS retrieval_embedding vector(4096)" in code

    def test_generated_column_matches_the_orm_expression(self):
        code = _norm(_code(RETRIEVAL_SQL))
        match = re.search(
            r"ADD COLUMN IF NOT EXISTS retrieval_tsv tsvector GENERATED ALWAYS AS \( (.*?) \) STORED",
            code,
        )
        assert match, "retrieval_tsv generated column missing"
        assert _norm(match.group(1)) == _norm(RETRIEVAL_TSV_EXPRESSION)

    def test_the_triggers_are_indexed_through_jsonb(self):
        # The text of a JSON column is the stored json.dumps output: its
        # \uXXXX escapes would reach the index instead of the words;
        # jsonb's text still escapes control characters and quotes, which are
        # replaced by spaces.
        code = _norm(_code(RETRIEVAL_SQL))
        assert "coalesce(regexp_replace(triggers::jsonb::text, " in code
        assert "coalesce(triggers::text, '')" not in code
        assert "coalesce(triggers::jsonb::text, '')" not in code

    def test_a_tsv_of_the_escaped_expression_is_rebuilt_once(self):
        # Any earlier expression — `triggers::text` or
        # `triggers::jsonb::text`, on a JSON or JSONB column — lacks
        # regexp_replace; the current one has it, so a re-run keeps it.
        code = _norm(_code(RETRIEVAL_SQL))
        assert "t.atttypid = 'json'::regtype" not in code
        guard = code.index(
            "pg_get_expr(d.adbin, d.adrelid) NOT LIKE '%regexp_replace%'"
        )
        drop = code.index(
            "ALTER TABLE procedural_memory DROP COLUMN IF EXISTS retrieval_tsv;"
        )
        add = code.index("ADD COLUMN IF NOT EXISTS retrieval_tsv tsvector")
        index = code.index(
            "CREATE INDEX IF NOT EXISTS ix_procedural_memory_retrieval_tsv"
        )
        # Dropped only while it is the old expression, then added and indexed.
        assert guard < drop < add < index
        assert "a.attname = 'retrieval_tsv'" in code

    def test_gin_index(self):
        code = _norm(_code(RETRIEVAL_SQL))
        assert (
            "CREATE INDEX IF NOT EXISTS ix_procedural_memory_retrieval_tsv ON procedural_memory USING gin (retrieval_tsv)"
            in code
        )

    def test_full_unique_constraint_becomes_a_partial_index(self):
        code = _norm(_code(RETRIEVAL_SQL))
        assert (
            "ALTER TABLE procedural_memory DROP CONSTRAINT uq_procedural_memory_org_user_name;"
            in code
        )
        assert "DROP INDEX IF EXISTS uq_procedural_memory_org_user_name;" in code
        assert (
            f"CREATE UNIQUE INDEX IF NOT EXISTS {UNIQUE_LIVE_NAME_INDEX} ON procedural_memory "
            "(organization_id, user_id, name) WHERE NOT is_deleted;" in code
        )

    def test_runs_after_the_skill_migration(self):
        assert "AFTER scripts/migrate_procedural_to_skill.sql" in _sql(RETRIEVAL_SQL)


# --------------------------------------------------------------------------- #
# migrate_add_skill_history.sql                                               #
# --------------------------------------------------------------------------- #


class TestSkillHistorySql:
    def test_tables_match_the_orm(self):
        code = _code(HISTORY_SQL)
        for table in ("skill_object", "skill_version"):
            orm_columns = {c.name for c in Base.metadata.tables[table].columns}
            assert _create_table_columns(code, table) == orm_columns, table

    def test_object_key_is_org_and_sha(self):
        assert "PRIMARY KEY (organization_id, sha256)" in _code(HISTORY_SQL)
        assert "content_zlib        BYTEA" in _code(HISTORY_SQL)

    def test_version_has_the_seven_field_hashes_and_no_experience_ids(self):
        code = _code(HISTORY_SQL)
        for field in SKILL_VERSIONED_FIELDS:
            assert re.search(rf"\b{field}_sha256\s+VARCHAR\(64\)\s+NOT NULL", code), (
                field
            )
        assert "experience_ids" not in code

    def test_version_number_is_unique_per_skill_and_history_cascades(self):
        code = _norm(_code(HISTORY_SQL))
        assert "uq_skill_version_skill_version_no UNIQUE (skill_id, version_no)" in code
        assert "REFERENCES procedural_memory(id) ON DELETE CASCADE" in code
        for index in _orm_index_names("skill_version"):
            assert f"CREATE INDEX IF NOT EXISTS {index}" in code, index


# --------------------------------------------------------------------------- #
# migrate_skill_experience_attempts.sql                                       #
# --------------------------------------------------------------------------- #


class TestSkillExperienceAttemptsSql:
    def test_adds_curation_attempts_defaulting_to_zero(self):
        code = _norm(_code(ATTEMPTS_SQL))
        assert (
            "ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS curation_attempts INTEGER NOT NULL DEFAULT 0"
            in code
        )

    def test_adds_no_vector(self):
        assert "embedding" not in _code(ATTEMPTS_SQL)


# --------------------------------------------------------------------------- #
# migrate_add_learning_run_and_retrieval_log.sql                              #
# --------------------------------------------------------------------------- #


class TestLearningRunAndRetrievalLogSql:
    @pytest.mark.parametrize("table", ["skill_learning_run", "retrieval_log"])
    def test_tables_match_the_orm(self, table):
        orm_columns = {c.name for c in Base.metadata.tables[table].columns}
        assert _create_table_columns(_code(RUN_LOG_SQL), table) == orm_columns

    @pytest.mark.parametrize("table", ["skill_learning_run", "retrieval_log"])
    def test_indexes_match_the_orm(self, table):
        code = _code(RUN_LOG_SQL)
        for index in _orm_index_names(table):
            assert f"CREATE INDEX IF NOT EXISTS {index}" in code, index

    def test_learning_run_defaults_and_agent_cascade(self):
        code = _norm(_code(RUN_LOG_SQL))
        assert '"trigger" VARCHAR NOT NULL' in code
        assert "status VARCHAR NOT NULL DEFAULT 'queued'" in code
        assert "session_ids JSON NOT NULL DEFAULT '[]'" in code
        assert "session_watermarks JSON NOT NULL DEFAULT '{}'" in code
        assert "embedding_chunked_calls INTEGER NOT NULL DEFAULT 0" in code
        assert "REFERENCES agents(id) ON DELETE CASCADE" in code

    def test_retrieval_log_chunked_calls_column_is_created_and_added(self):
        """A retrieval_log from an earlier revision of the script gets
        embedding_chunked_calls too."""
        code = _norm(_code(RUN_LOG_SQL))
        assert "embedding_chunked_calls INTEGER NOT NULL DEFAULT 0" in code
        assert (
            "ALTER TABLE retrieval_log ADD COLUMN IF NOT EXISTS "
            "embedding_chunked_calls INTEGER NOT NULL DEFAULT 0;"
        ) in code

    @pytest.mark.parametrize("table", ["skill_learning_run", "retrieval_log"])
    def test_retry_events_column_is_created_and_added(self, table):
        """A table from an earlier revision of the script gets it too."""
        code = _norm(_code(RUN_LOG_SQL))
        assert "retry_events JSON NOT NULL DEFAULT '[]'" in code
        assert (
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS retry_events "
            "JSON NOT NULL DEFAULT '[]';"
        ) in code


class TestSkillExperiencePoolSql:
    """The experience pool's columns on skill_experience."""

    COLUMNS = {
        "disposition": "VARCHAR NULL",
        "cited_skill_ids": "JSON NULL",
        "pool_scope": "VARCHAR NULL",
        "pool_embedding": "vector(4096) NULL",
        "pool_embedding_model": "VARCHAR NULL",
        "pool_embedding_failures": "INTEGER NOT NULL DEFAULT 0",
        "recurrence_count": "INTEGER NOT NULL DEFAULT 1",
        "ignored_pushes": "INTEGER NOT NULL DEFAULT 0",
        "pushed_count": "INTEGER NOT NULL DEFAULT 0",
        "last_pushed_at": "TIMESTAMPTZ NULL",
        "pool_retired_at": "TIMESTAMPTZ NULL",
        "pool_retire_reason": "VARCHAR NULL",
    }

    def test_adds_every_pool_column_with_its_type_and_default(self):
        code = _norm(_code(POOL_SQL))
        for column, definition in self.COLUMNS.items():
            assert (
                f"ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS {column} {definition};"
            ) in code, column

    def test_the_vector_is_as_wide_as_every_stored_vector(self):
        from mirix.constants import MAX_EMBEDDING_DIM

        assert f"vector({MAX_EMBEDDING_DIM})" in _code(POOL_SQL)

    def test_the_scripts_columns_are_the_orms_new_columns(self):
        added = set(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", _code(POOL_SQL)))
        orm = {c.name for c in Base.metadata.tables["skill_experience"].columns}
        assert added == set(self.COLUMNS) and added <= orm
        from mirix.server import server

        hinted = {
            column
            for (table, column), script in server._MIGRATION_HINTS.items()
            if table == "skill_experience" and script.endswith(POOL_SQL)
        }
        assert hinted == added

    def test_creates_the_pool_index(self):
        code = _norm(_code(POOL_SQL))
        assert (
            "CREATE INDEX IF NOT EXISTS ix_skill_experience_agent_disposition "
            "ON skill_experience (agent_id, disposition);"
        ) in code
        assert "ix_skill_experience_agent_disposition" in _orm_index_names(
            "skill_experience"
        )

    def test_backfills_nothing(self):
        code = _code(POOL_SQL).upper()
        for keyword in ("UPDATE ", "INSERT ", "DELETE "):
            assert keyword not in code, keyword


class TestP16AgentContextAndSkillFeedbackSql:
    """conversation_session.agent_context and the
    skill_session_feedback table."""

    def test_adds_the_agent_context_column(self):
        code = _norm(_code(P16_SQL))
        assert (
            "ALTER TABLE conversation_session ADD COLUMN IF NOT EXISTS "
            "agent_context TEXT NULL;"
        ) in code
        from mirix.server import server

        assert server._MIGRATION_HINTS[
            ("conversation_session", "agent_context")
        ].endswith(P16_SQL)

    def test_the_feedback_table_matches_the_orm(self):
        orm = {c.name for c in Base.metadata.tables["skill_session_feedback"].columns}
        assert _create_table_columns(_code(P16_SQL), "skill_session_feedback") == orm
        code = _code(P16_SQL)
        for index in _orm_index_names("skill_session_feedback"):
            assert f"CREATE INDEX IF NOT EXISTS {index}" in code, index
        assert "REFERENCES procedural_memory (id) ON DELETE CASCADE" in code
        from mirix.server import server

        assert server._TABLE_MIGRATION_SCRIPTS["skill_session_feedback"].endswith(
            P16_SQL
        )

    def test_backfills_nothing(self):
        code = _code(P16_SQL).upper().replace("ON DELETE CASCADE", "")
        for keyword in ("UPDATE ", "INSERT ", "DELETE "):
            assert keyword not in code, keyword


class TestUpgradeScripts:
    """The scripts the server names at startup exist, and the Postgres upgrade
    test runs every new script."""

    def test_every_script_the_server_asks_for_exists(self):
        from mirix.server import server

        asked = set(server._MIGRATION_HINTS.values())
        asked |= set(server._TABLE_MIGRATION_SCRIPTS.values())
        asked |= {script for _desc, script, _probe in server._DATA_MIGRATION_CHECKS}
        for hint in asked:
            script = re.match(r"scripts/([\w.-]+\.sql)", hint).group(1)
            assert (SCRIPTS / script).is_file(), hint

    def test_the_postgres_upgrade_test_runs_every_new_script(self):
        from test_skill_data_layer_pg import MIGRATION_SCRIPTS

        assert len(MIGRATION_SCRIPTS) == len(set(MIGRATION_SCRIPTS))
        assert sorted(MIGRATION_SCRIPTS) == sorted(ALL_SCRIPTS)

    def test_the_python_upgrade_scripts_exist(self):
        for script in (
            "backfill_skill_baseline_versions.py",
            "reembed_skill_retrieval.py",
        ):
            assert (SCRIPTS / script).is_file()
