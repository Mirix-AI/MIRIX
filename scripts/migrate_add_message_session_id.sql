-- Migration: add top-level session_id column + index on messages.
-- Run once on existing databases. New databases get this via SQLAlchemy create_all.
--
-- This migration is split into two phases because CREATE INDEX CONCURRENTLY
-- cannot run inside a transaction block.
--
-- PHASE 1 (transactional): add the column and a CHECK constraint so any writer
-- is immediately bound by the format rules. Runs quickly; acquires a short
-- ACCESS EXCLUSIVE lock only for the DDL itself.
--
-- PHASE 2 (must run OUTSIDE a transaction, not inside psql -1):
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_messages_agent_session_created_at
--       ON messages (agent_id, session_id, created_at);
--
-- CONCURRENTLY avoids the write-blocking ACCESS EXCLUSIVE lock on large tables.
-- If the index build fails midway, Postgres leaves an INVALID index behind; drop
-- it and retry.

-- Phase 1 is idempotent: safe to re-run (e.g. if phase 2 failed and the whole
-- migration is replayed). Column/constraint additions all guard against
-- existing state.

BEGIN;

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS session_id VARCHAR(64);

-- CHECK constraint matches mirix.schemas.message._validate_session_id.
-- The constraint is NOT VALID first so adding it on a large table does not
-- rewrite existing rows; legacy rows all have NULL session_id and satisfy it.
-- Guarded so a re-run (after a partial failure) is a no-op.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_messages_session_id_format'
          AND conrelid = 'messages'::regclass
    ) THEN
        ALTER TABLE messages
            ADD CONSTRAINT ck_messages_session_id_format
            CHECK (session_id IS NULL OR session_id ~ '^[A-Za-z0-9_-]{1,64}$')
            NOT VALID;
    END IF;
END$$;

-- Validate the constraint in a separate step so it only takes SHARE UPDATE
-- EXCLUSIVE (reads + writes continue; no concurrent DDL only).
-- VALIDATE is a no-op if the constraint is already validated.
ALTER TABLE messages VALIDATE CONSTRAINT ck_messages_session_id_format;

COMMIT;

-- PHASE 2 — run scripts/migrate_add_message_session_id_phase2.sql separately,
-- NOT inside a transaction (do not combine with this file via `psql -1`).
