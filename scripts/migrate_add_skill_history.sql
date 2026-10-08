-- Migration: create the skill version-history tables skill_object and skill_version
--.
--
-- Run once on existing PostgreSQL databases, after
-- scripts/migrate_skill_retrieval_fields.sql. New databases get both tables via
-- SQLAlchemy create_all (mirix/orm/skill_object.py, mirix/orm/skill_version.py).
--
-- A git-style content-addressed store:
--   skill_object   one zlib-compressed canonical skill-field value per
--                  (organization_id, sha256 of the uncompressed bytes). Scoped
--                  per organization so deduplication never crosses tenants.
--                  Written with INSERT ... ON CONFLICT (organization_id, sha256)
--                  DO UPDATE SET last_referenced_at = now().
--   skill_version  one "commit" per change of a skill: version_no (1, 2, ...;
--                  unique per skill), parent version, change type
--                  (baseline | create | edit | delete | restore), provenance, and
--                  the sha256 of the seven versioned fields (name, entry_type,
--                  description, instructions, triggers, examples, filter_tags).
--                  Embeddings are never versioned.
-- Existing skills get their 'baseline' version lazily, on their first change,
-- or eagerly from scripts/backfill_skill_baseline_versions.py.
--
-- Idempotent: CREATE TABLE / INDEX IF NOT EXISTS, guarded constraints.

BEGIN;

CREATE TABLE IF NOT EXISTS skill_object (
    organization_id     VARCHAR      NOT NULL REFERENCES organizations(id),
    sha256              VARCHAR(64)  NOT NULL,
    content_zlib        BYTEA        NOT NULL,
    raw_length          INTEGER      NOT NULL,
    created_at          TIMESTAMPTZ  DEFAULT now(),
    last_referenced_at  TIMESTAMPTZ  DEFAULT now(),
    PRIMARY KEY (organization_id, sha256)
);

CREATE TABLE IF NOT EXISTS skill_version (
    id                   VARCHAR      PRIMARY KEY,
    organization_id      VARCHAR      REFERENCES organizations(id),
    user_id              VARCHAR      NOT NULL REFERENCES users(id),
    -- A hard-deleted skill takes its whole history with it (erasure);
    -- soft-deleted skills keep theirs and can be restored.
    skill_id             VARCHAR      NOT NULL REFERENCES procedural_memory(id) ON DELETE CASCADE,
    version_no           INTEGER      NOT NULL,
    parent_version_id    VARCHAR      NULL,
    change_type          VARCHAR      NOT NULL,
    name_sha256          VARCHAR(64)  NOT NULL,
    entry_type_sha256    VARCHAR(64)  NOT NULL,
    description_sha256   VARCHAR(64)  NOT NULL,
    instructions_sha256  VARCHAR(64)  NOT NULL,
    triggers_sha256      VARCHAR(64)  NOT NULL,
    examples_sha256      VARCHAR(64)  NOT NULL,
    filter_tags_sha256   VARCHAR(64)  NOT NULL,
    run_id               VARCHAR      NULL,
    actor_kind           VARCHAR      NULL,
    reason               TEXT         NULL,
    created_at           TIMESTAMPTZ  DEFAULT now(),
    updated_at           TIMESTAMPTZ  DEFAULT now(),
    is_deleted           BOOLEAN      NOT NULL DEFAULT FALSE,
    _created_by_id       VARCHAR      NULL,
    _last_updated_by_id  VARCHAR      NULL
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_skill_version_skill_version_no'
          AND conrelid = 'skill_version'::regclass
    ) THEN
        ALTER TABLE skill_version
            ADD CONSTRAINT uq_skill_version_skill_version_no
            UNIQUE (skill_id, version_no);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS ix_skill_version_org_skill
    ON skill_version (organization_id, skill_id);

-- "Which versions did this curator batch write" (run_id = `{learning_run_id}:b{i}`).
CREATE INDEX IF NOT EXISTS ix_skill_version_run_id
    ON skill_version (run_id);

COMMIT;
