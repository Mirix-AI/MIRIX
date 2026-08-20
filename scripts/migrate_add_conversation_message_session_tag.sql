-- Migration: add nullable session_tag column + index on conversation_message.
-- Run once on existing databases. New databases get this via SQLAlchemy create_all.
--
-- session_tag drives adaptive per-session memory routing:
--   'task'         -> procedural/skill distillation (auto_dream mode="procedural")
--   'conversation' -> inline episodic/semantic extraction; NO skill distillation
--   NULL (legacy)  -> treated as eligible by the distiller (fail-open), so this
--                     backfill-free migration never silently drops skill-building
--                     on pre-existing task data.
--
-- The column is nullable with no default: existing rows keep NULL, and the
-- distiller gate (list_sealed_undistilled_sessions) excludes only rows
-- EXPLICITLY tagged 'conversation'. No row rewrite, no backfill required.

-- Idempotent: safe to re-run.
BEGIN;

ALTER TABLE conversation_message
    ADD COLUMN IF NOT EXISTS session_tag VARCHAR;

COMMIT;

-- Index the routing tag so the distiller's sealed-session enumeration can filter
-- on it cheaply. CREATE INDEX CONCURRENTLY cannot run inside a transaction, so
-- run this line OUTSIDE a transaction block (not via psql -1):
--
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_conversation_message_session_tag
--       ON conversation_message (session_tag);
--
-- On a fresh/small database the transactional form is fine:
CREATE INDEX IF NOT EXISTS ix_conversation_message_session_tag
    ON conversation_message (session_tag);
