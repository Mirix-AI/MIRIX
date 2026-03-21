-- Migration 001: Additive schema changes (run BEFORE merging the code)
-- Safe to run on a live database — all changes are backward-compatible.
-- After running this script, merge the code PR.

-- 1. Add message_set_retention_count to clients table
ALTER TABLE clients
    ADD COLUMN IF NOT EXISTS message_set_retention_count INTEGER DEFAULT 0;

-- 2. Add composite index for efficient retention queries on messages
--    Supports: ORDER BY created_at DESC, id DESC WHERE agent_id=? AND user_id=?
CREATE INDEX IF NOT EXISTS ix_messages_agent_user_created_at
    ON messages (agent_id, user_id, created_at, id);
