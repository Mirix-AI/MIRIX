-- Migration: Convert procedural memory from flat format to skill-based format
-- Run this ONCE on existing databases before deploying the skill-evolve branch.
-- For new databases, the schema is created automatically by SQLAlchemy create_all.

BEGIN;

-- Step 1: Add new columns with safe defaults
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS name VARCHAR NOT NULL DEFAULT '';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS triggers JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS examples JSONB DEFAULT '[]';
ALTER TABLE procedural_memory ADD COLUMN IF NOT EXISTS version VARCHAR NOT NULL DEFAULT '0.1.0';

-- Step 2: Generate name from summary (slugify: lowercase, strip punctuation, spaces to hyphens, truncate)
UPDATE procedural_memory
SET name = LOWER(
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            LEFT(TRIM(summary), 60),
            '[^a-zA-Z0-9\s-]', '', 'g'
        ),
        '\s+', '-', 'g'
    )
)
WHERE name = '';

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

-- Step 6: Add new index for name-based dedup lookups
CREATE INDEX IF NOT EXISTS ix_procedural_memory_org_user_name
ON procedural_memory (organization_id, user_id, name);

COMMIT;
