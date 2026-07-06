-- Migration PHASE 2 for conversation_message: build the composite indexes
-- CONCURRENTLY. Run this AFTER scripts/migrate_add_conversation_message.sql.
--
-- IMPORTANT: run OUTSIDE a transaction — plain `psql -f`, NOT
-- `psql -1`/--single-transaction. CREATE INDEX CONCURRENTLY cannot run inside a
-- transaction block. CONCURRENTLY avoids taking a write-blocking ACCESS
-- EXCLUSIVE lock on the table while the index builds.
--
-- New databases get these indexes via SQLAlchemy create_all at server startup
-- (the ORM model in mirix/orm/conversation_message.py declares them), so only
-- existing databases being migrated in place need this file.
--
-- If a build fails midway, Postgres leaves an INVALID index behind: drop it
-- (DROP INDEX <name>) and re-run.

-- Primary access pattern: list/seal/order this (org, user)'s sessions by
-- first-appearance time.
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    ix_conversation_message_org_user_session_created
    ON conversation_message (organization_id, user_id, session_id, created_at);

-- Accelerates the per-session ascending fetch in list_turns_for_session.
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    ix_conversation_message_session_created
    ON conversation_message (session_id, created_at);

-- Single-column organization index (mirrors the ORM's pg-only org index).
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    ix_conversation_message_organization_id
    ON conversation_message (organization_id);
