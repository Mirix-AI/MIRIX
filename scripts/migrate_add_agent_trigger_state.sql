-- Migration: create agent_trigger_state table.
--
-- Run once on existing databases. New databases get this via SQLAlchemy
-- create_all at server startup, so there is no separate "phase 2" here.
--
-- Purpose: persist per-(agent, user, trigger_type) cursors for interval-
-- driven memory triggers (e.g. "fire procedural extraction every N sessions").
-- Only the last-fire cursor is stored; the "how many sessions since" counter
-- is derived from the messages table at read time.
--
-- Entire migration is idempotent and safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS agent_trigger_state (
    id                          VARCHAR        PRIMARY KEY,
    organization_id             VARCHAR        REFERENCES organizations(id),
    user_id                     VARCHAR        NOT NULL REFERENCES users(id),
    agent_id                    VARCHAR        NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    trigger_type                VARCHAR(64)    NOT NULL,
    last_fired_at               TIMESTAMPTZ    NULL,
    last_fired_session_id       VARCHAR(64)    NULL,
    -- Session_ids tied at the watermark timestamp. Stored so the next window
    -- can use `created_at >= last_fired_at` with session-level tie-break,
    -- instead of `created_at > last_fired_at` which silently drops any row
    -- that committed at the exact same microsecond as our SELECT.
    last_fired_tied_session_ids JSON           NULL,
    created_at                  TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    is_deleted                  BOOLEAN        NOT NULL DEFAULT FALSE,
    _created_by_id              VARCHAR        NULL,
    _last_updated_by_id         VARCHAR        NULL
);

-- Back-compat: if an older deployment created this table before the tied
-- set was added, fill it in. Idempotent.
ALTER TABLE agent_trigger_state
    ADD COLUMN IF NOT EXISTS last_fired_tied_session_ids JSON NULL;

-- One live cursor per (agent, user, trigger_type). The app always addresses
-- a row by this triple, so the uniqueness closes an insert-race between
-- concurrent workers.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_agent_trigger_state_agent_user_type'
          AND conrelid = 'agent_trigger_state'::regclass
    ) THEN
        ALTER TABLE agent_trigger_state
            ADD CONSTRAINT uq_agent_trigger_state_agent_user_type
            UNIQUE (agent_id, user_id, trigger_type);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS ix_agent_trigger_state_agent_user_type
    ON agent_trigger_state (agent_id, user_id, trigger_type);

COMMIT;
