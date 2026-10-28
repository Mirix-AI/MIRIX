-- Migration: the skill-experience pool.
--
-- Run once on existing PostgreSQL databases. New databases get the columns via
-- SQLAlchemy create_all (mirix/orm/skill_experience.py).
--
-- A curator batch that finishes now records what became of each of its
-- experiences (disposition: used | covered | not_worth | deferred, or
-- unaddressed for one closed out after its third attempt) and which skills a
-- write cited it for (cited_skill_ids; influenced_skill_ids stays the batch
-- lineage). Consumed experiences that are deferred or not worth a skill form
-- the experience pool of their (organization, user, meta agent) and write
-- scope (pool_scope): a later batch shows the most similar of them again as
-- "earlier lessons" (pool_embedding, from pool_embedding_model: the model and
-- its dimension), counts how often a lesson recurs (recurrence_count), and
-- retires one that was used, covered, ignored too often (ignored_pushes,
-- pushed_count, last_pushed_at), merged into an earlier lesson it recurred,
-- or that the embedding model kept refusing (pool_embedding_failures):
-- pool_retired_at, pool_retire_reason.
--
-- Existing consumed experiences are NOT added to the pool: their lineage is
-- per batch, so which of them a skill used cannot be told apart, and the
-- scope they were learned under is unknown. They keep disposition NULL.
--
-- pool_embedding is vector(4096) (MAX_EMBEDDING_DIM, zero-padded like
-- procedural_memory.retrieval_embedding); the vector extension is already
-- required by that column.
--
-- Idempotent: ADD COLUMN / CREATE INDEX IF NOT EXISTS. On PostgreSQL 11+ a
-- constant DEFAULT is metadata-only, so this does not rewrite the table.

BEGIN;

ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS disposition VARCHAR NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS cited_skill_ids JSON NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_scope VARCHAR NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_embedding vector(4096) NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_embedding_model VARCHAR NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_embedding_failures INTEGER NOT NULL DEFAULT 0;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS recurrence_count INTEGER NOT NULL DEFAULT 1;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS ignored_pushes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pushed_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS last_pushed_at TIMESTAMPTZ NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_retired_at TIMESTAMPTZ NULL;
ALTER TABLE skill_experience ADD COLUMN IF NOT EXISTS pool_retire_reason VARCHAR NULL;

CREATE INDEX IF NOT EXISTS ix_skill_experience_agent_disposition
    ON skill_experience (agent_id, disposition);

COMMIT;
