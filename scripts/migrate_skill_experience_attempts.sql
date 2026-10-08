-- Migration: add skill_experience.curation_attempts.
--
-- Run once on existing PostgreSQL databases. New databases get the column via
-- SQLAlchemy create_all (mirix/orm/skill_experience.py).
--
-- curation_attempts counts the curator batches that ended without addressing an
-- experience (stop reason context_limit / step_cap, or a non-infrastructure
-- curator failure): the batch returns the experience to 'pending' and adds 1;
-- after 3 attempts it is closed out as consumed (unaddressed) instead of looping
-- forever. Existing experiences start at 0.
--
-- No vector column is added: experiences are no longer grouped by embedding.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS. On PostgreSQL 11+ a constant DEFAULT is
-- metadata-only, so this does not rewrite the table.

BEGIN;

ALTER TABLE skill_experience
    ADD COLUMN IF NOT EXISTS curation_attempts INTEGER NOT NULL DEFAULT 0;

COMMIT;
