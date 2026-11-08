-- Migration: add the graph-memory pipeline's provenance columns.
--
-- Run once on existing PostgreSQL databases. New databases get the columns via
-- SQLAlchemy create_all (mirix/orm/episodic_memory.py, mirix/orm/semantic_memory.py,
-- mirix/orm/user.py).
--
-- episodic_memory.source_refs and semantic_memory.source_refs list the ingests a
-- row came from (turn_id, chunk_id, occurred_at); semantic_memory.prior_values
-- keeps the values conflict resolution superseded; users.turn_counter and
-- users.chunk_counter number a user's ingests. They are written only when graph
-- memory is on (MIRIX_ENABLE_GRAPH_MEMORY=true); otherwise they keep their defaults.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS. On PostgreSQL 11+ a constant DEFAULT is
-- metadata-only, so this does not rewrite the tables.

BEGIN;

ALTER TABLE episodic_memory
    ADD COLUMN IF NOT EXISTS source_refs JSON NOT NULL DEFAULT '[]';

ALTER TABLE semantic_memory
    ADD COLUMN IF NOT EXISTS source_refs JSON NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS prior_values JSON NOT NULL DEFAULT '[]';

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS turn_counter INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS chunk_counter INTEGER NOT NULL DEFAULT 0;

COMMIT;
