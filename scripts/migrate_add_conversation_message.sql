-- Migration: create the conversation_message table + indexes.
-- Run once on existing databases. New databases get this via SQLAlchemy
-- create_all at server startup (the ORM model is registered in mirix/orm/__init__.py).
--
-- The conversation_message store holds ONLY external conversation turns that
-- arrived through the memory-add API carrying a session_id, with their real
-- user/assistant roles preserved. It is the single source the procedural-memory
-- (skill) distiller reads — separate from the agent-loop `messages` table.
--
-- This migration is split into two phases because CREATE INDEX CONCURRENTLY
-- cannot run inside a transaction block.
--
-- PHASE 1 (transactional): create the table with its columns, the CHECK
-- constraint on session_id (matching mirix.schemas.message), and the
-- non-concurrent indexes implied by column-level `index=True`.
--
-- PHASE 2 (must run OUTSIDE a transaction, not inside psql -1): build the
-- composite indexes CONCURRENTLY so they do not take a write-blocking ACCESS
-- EXCLUSIVE lock on a large table.
--
-- Phase 1 is idempotent: safe to re-run (e.g. if phase 2 failed and the whole
-- migration is replayed). All object creation guards against existing state.

BEGIN;

CREATE TABLE IF NOT EXISTS conversation_message (
    id               VARCHAR PRIMARY KEY,
    session_id       VARCHAR(64) NOT NULL,
    role             VARCHAR     NOT NULL,
    content          TEXT        NOT NULL DEFAULT '',
    distilled_at     TIMESTAMPTZ,
    user_id          VARCHAR     NOT NULL REFERENCES users(id),
    organization_id  VARCHAR     REFERENCES organizations(id),
    -- CommonSqlalchemyMetaMixins columns, matching sibling tables.
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now(),
    is_deleted       BOOLEAN     DEFAULT FALSE,
    _created_by_id   VARCHAR,
    _last_updated_by_id VARCHAR
);

-- CHECK constraint matches mirix.schemas.message._validate_session_id. Unlike
-- the `messages` table, session_id here is NOT NULL (this store only holds
-- session'd turns), so the NULL branch is omitted. Guarded so a re-run is a
-- no-op.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_conversation_message_session_id_format'
          AND conrelid = 'conversation_message'::regclass
    ) THEN
        ALTER TABLE conversation_message
            ADD CONSTRAINT ck_conversation_message_session_id_format
            CHECK (session_id ~ '^[A-Za-z0-9_-]{1,64}$');
    END IF;
END$$;

-- Non-concurrent single-column index from the ORM column-level `index=True` on
-- session_id. Small/cheap; kept in phase 1.
CREATE INDEX IF NOT EXISTS ix_conversation_message_session_id
    ON conversation_message (session_id);

COMMIT;

-- PHASE 2 — the composite indexes are built CONCURRENTLY in a SEPARATE,
-- EXECUTABLE file: scripts/migrate_add_conversation_message_phase2.sql. They
-- live there (rather than commented out here) because CREATE INDEX CONCURRENTLY
-- cannot run inside a transaction block, so they must NOT share this file's
-- BEGIN/COMMIT. Run that file next, OUTSIDE a transaction (plain `psql -f`, NOT
-- `psql -1`/--single-transaction). New databases get these indexes via
-- SQLAlchemy create_all and need neither migration file.
