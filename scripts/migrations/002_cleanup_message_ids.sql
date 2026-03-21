-- Migration 002: Cleanup (run AFTER the code has been deployed and verified)
-- WARNING: Destructive — drops the message_ids column and deletes legacy system messages.
-- Ensure the new code is running correctly before executing this script.

-- 1. Delete legacy system messages stored as Message rows
--    (system prompt now lives exclusively in agent_state.system)
DELETE FROM messages WHERE role = 'system';

-- 2. Drop the message_ids column from agents table
ALTER TABLE agents DROP COLUMN IF EXISTS message_ids;
