-- Phase 2 of the session_id migration — run AFTER migrate_add_message_session_id.sql.
-- MUST run outside a transaction (no BEGIN/COMMIT here, and do NOT run via `psql -1`).
-- Use e.g.:
--     psql "$DATABASE_URL" -f scripts/migrate_add_message_session_id_phase2.sql
--
-- CONCURRENTLY avoids the write-blocking ACCESS EXCLUSIVE lock on large tables.
-- If the build fails midway, Postgres leaves an INVALID index behind; drop it and retry:
--     DROP INDEX CONCURRENTLY IF EXISTS ix_messages_agent_session_created_at;

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_messages_agent_session_created_at
    ON messages (agent_id, session_id, created_at);
