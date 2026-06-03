-- Migration: Convert procedural memory from flat format to skill-based format
-- Run this ONCE on existing databases before deploying the skill-evolve branch.
-- For new databases, the schema is created automatically by SQLAlchemy create_all.

BEGIN;

-- Step 1: Add new columns with safe defaults
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS name VARCHAR NOT NULL DEFAULT '';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS triggers JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS examples JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS version VARCHAR NOT NULL DEFAULT '0.1.0';

-- Step 2: Generate name from summary (slugify: lowercase, strip punctuation, spaces to hyphens, truncate).
-- Guard against NULL/empty summary by falling back to "skill-<id-suffix>" so the
-- NOT NULL column constraint is never violated. Rows whose slug would otherwise
-- be empty (e.g. summary='   ' or summary='!!!') also get the fallback.
UPDATE procedural_memory
SET name = NULLIF(
    LOWER(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LEFT(TRIM(COALESCE(summary, '')), 60),
                '[^a-zA-Z0-9\s-]', '', 'g'
            ),
            '\s+', '-', 'g'
        )
    ),
    ''
)
WHERE name = '';

UPDATE procedural_memory
SET name = 'skill-' || RIGHT(id, 8)
WHERE name IS NULL OR name = '';

-- Step 2b: Resolve slug collisions deterministically by suffixing -N, using
-- created_at (then id) as the tie-breaker. The first occurrence keeps the
-- bare slug; subsequent occurrences get -2, -3, etc. This runs before the
-- unique index so the unique constraint creation cannot fail on legacy data.
WITH ranked AS (
    SELECT
        id,
        name,
        organization_id,
        user_id,
        ROW_NUMBER() OVER (
            PARTITION BY organization_id, user_id, name
            ORDER BY created_at, id
        ) AS occurrence
    FROM procedural_memory
)
UPDATE procedural_memory pm
SET name = pm.name || '-' || ranked.occurrence
FROM ranked
WHERE pm.id = ranked.id
  AND ranked.occurrence > 1;

-- Step 3: Convert steps from JSON array to plain text (newline-joined)
UPDATE procedural_memory
SET steps = ARRAY_TO_STRING(
    ARRAY(SELECT jsonb_array_elements_text(steps::jsonb)),
    E'\n'
)
WHERE steps IS NOT NULL AND steps::text LIKE '[%';

-- Step 4: Rename columns
ALTER TABLE procedural_memory RENAME COLUMN summary TO description;
ALTER TABLE procedural_memory RENAME COLUMN steps TO instructions;
ALTER TABLE procedural_memory RENAME COLUMN summary_embedding TO description_embedding;
ALTER TABLE procedural_memory RENAME COLUMN steps_embedding TO instructions_embedding;

-- Step 5: Change instructions column type from JSON to TEXT
ALTER TABLE procedural_memory ALTER COLUMN instructions TYPE TEXT USING instructions::TEXT;

-- Step 6: Drop legacy non-unique lookup index (replaced by unique constraint below).
DROP INDEX IF EXISTS ix_procedural_memory_org_user_name;

-- Step 7: Enforce per-user skill-name uniqueness at the DB level. Matches the
-- UniqueConstraint on the ORM model; without it, concurrent skill_create
-- calls race past the application-level pre-check.
ALTER TABLE procedural_memory
    ADD CONSTRAINT uq_procedural_memory_org_user_name
    UNIQUE (organization_id, user_id, name);

COMMIT;
