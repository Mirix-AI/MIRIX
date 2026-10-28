-- Migration: multimodal conversation-message columns + files.user_id.
--
-- Run once on existing PostgreSQL databases. New databases get these columns via
-- SQLAlchemy create_all at server startup.
--
--   conversation_message.content_parts         JSON, NULL
--       Ordered text parts and image references, written ONLY for messages that
--       carry an image; `content` then holds a `[image: file-...]`
--       placeholder instead of base64. Text-only messages leave it NULL.
--   conversation_message.original_text_length  INTEGER, NULL
--       Text length before the storage cap truncated the message; NULL when
--       stored untruncated.
--   files.user_id                               VARCHAR, NULL -> users(id) ON DELETE SET NULL
--       Whose data a stored file (e.g. a conversation image) is, so deleting a
--       user's data can remove their images. SET NULL rather than
--       CASCADE: a cascaded delete would orphan the file on disk with no record.
--
-- All three are nullable with no default: adding them is a metadata-only change
-- (no table rewrite) and existing rows keep NULL, which means exactly what it
-- means for new rows (text-only / untruncated / not attributable to a user).
--
-- Idempotent: ADD COLUMN IF NOT EXISTS, CREATE INDEX IF NOT EXISTS.

BEGIN;

ALTER TABLE conversation_message
    ADD COLUMN IF NOT EXISTS content_parts JSON NULL;

ALTER TABLE conversation_message
    ADD COLUMN IF NOT EXISTS original_text_length INTEGER NULL;

-- The inline REFERENCES creates files_user_id_fkey together with the column, so
-- a re-run (column present) skips both.
ALTER TABLE files
    ADD COLUMN IF NOT EXISTS user_id VARCHAR NULL REFERENCES users(id) ON DELETE SET NULL;

-- "All files of this user" for user-data deletion. On a very large files table,
-- build it CONCURRENTLY instead, OUTSIDE a transaction:
--   CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_files_user_id ON files (user_id);
CREATE INDEX IF NOT EXISTS ix_files_user_id ON files (user_id);

COMMIT;
