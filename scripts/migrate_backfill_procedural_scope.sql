-- Migration: backfill procedural_memory.filter_tags 'scope' from the owning client.
--
-- Run ONCE on existing Postgres databases, AFTER migrate_procedural_to_skill.sql.
-- Postgres-only (jsonb operators). SQLite deployments get their schema from
-- SQLAlchemy create_all and have no multi-client scope to backfill (see header note).
--
-- Problem: scope-based read authorization filters procedural skills on
-- filter_tags->>'scope' (mirix/database/filter_tags_query.py). Rows written before
-- scope-tagging existed -- or created internally by skill-evolve without the REST
-- scope-injection -- have no 'scope' key, so they are invisible to every scoped
-- reader. This reconstructs the scope from the row's owning client
-- (procedural_memory.client_id -> clients.write_scope), which is exactly the value
-- the REST layer stamps into filter_tags['scope'] on create.
--
-- Data shapes handled: the ORM column is generic JSON with SQLAlchemy's default
-- none_as_null=False, so a Python None is stored as the JSON scalar 'null' (NOT
-- SQL NULL). jsonb_set() raises "cannot set path in scalar" on such rows, so any
-- non-object value (SQL NULL, JSON null, or another scalar) is replaced with '{}'
-- before setting the key. Verified against live rows where every unscoped row was
-- the JSON scalar 'null'.
--
-- Idempotent: only touches rows whose filter_tags has no 'scope' key, so re-running
-- is a no-op. Runs in one transaction (pure UPDATE, no DDL, no CONCURRENTLY).

BEGIN;

-- Fixable rows: filter_tags has no 'scope' key AND the owning client has a non-null
-- write_scope to copy. jsonb_set(..., create_if_missing => true) preserves any other
-- keys an object row already has; non-object rows are normalized to '{}' first,
-- casting json->jsonb->json around the mutation.
--
-- Deliberately LEFT UNTOUCHED (reported below, never guessed):
--   * rows with client_id IS NULL          -- no client to derive a scope from
--   * rows whose client.write_scope IS NULL -- read-only client, no scope to donate
-- Stamping a guessed scope would be a silent authorization change, so unresolved rows
-- stay unscoped (invisible) rather than mis-scoped (leaked into the wrong scope).
UPDATE procedural_memory pm
SET filter_tags = jsonb_set(
        CASE
            WHEN pm.filter_tags IS NULL THEN '{}'::jsonb
            WHEN jsonb_typeof(pm.filter_tags::jsonb) <> 'object' THEN '{}'::jsonb
            ELSE pm.filter_tags::jsonb
        END,
        '{scope}',
        to_jsonb(c.write_scope),
        true
    )::json
FROM clients c
WHERE pm.client_id = c.id
  AND c.write_scope IS NOT NULL
  AND (pm.filter_tags IS NULL
       OR jsonb_typeof(pm.filter_tags::jsonb) <> 'object'
       OR NOT jsonb_exists(pm.filter_tags::jsonb, 'scope'));

-- Report rows still lacking a scope after the backfill (NULL client_id or read-only
-- client). Operators must scope these by hand if they need to be readable.
DO $$
DECLARE
    unresolved INTEGER;
BEGIN
    SELECT COUNT(*) INTO unresolved
    FROM procedural_memory pm
    WHERE pm.filter_tags IS NULL
       OR jsonb_typeof(pm.filter_tags::jsonb) <> 'object'
       OR NOT jsonb_exists(pm.filter_tags::jsonb, 'scope');
    RAISE NOTICE 'procedural_memory rows still without a scope tag after backfill: %', unresolved;
END $$;

COMMIT;
