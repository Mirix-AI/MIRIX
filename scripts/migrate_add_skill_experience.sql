-- Migration: create skill_experience table.
--
-- Run once on existing databases. New databases get this via SQLAlchemy
-- create_all at server startup (the ORM class is imported in
-- mirix/orm/__init__.py), so there is no separate "phase 2" here.
--
-- Purpose: durable, general store for transferable EXPERIENCES distilled from
-- a single work session's transcript (Goal-2). Each experience is either
-- 'worth_learning' or 'worth_avoiding', scored by importance/credibility in
-- [0,1]. Consumed every N sessions by the Goal-3 skill-evolution run, ordered
-- by importance*credibility. Experiences flow pending -> consumed | superseded.
--
-- Entire migration is idempotent and safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS skill_experience (
    id                    VARCHAR     PRIMARY KEY,
    organization_id       VARCHAR     REFERENCES organizations(id),
    user_id               VARCHAR     NOT NULL REFERENCES users(id),
    agent_id              VARCHAR     NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    session_id            VARCHAR     NOT NULL,
    -- experience_type / status are plain strings (NOT pg ENUMs); their value
    -- spaces are validated in mirix/schemas/skill_experience.py, so adding a
    -- value never requires a DB migration.
    experience_type       VARCHAR     NOT NULL,
    title                 VARCHAR     NOT NULL,
    -- content/evidence/importance/credibility are NOT NULL with defaults to
    -- match the ORM + the pydantic full schema, which treat them as required.
    -- A NULL here would break to_pydantic().
    content               TEXT        NOT NULL DEFAULT '',
    importance            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    credibility           DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    evidence              TEXT        NOT NULL DEFAULT '',
    status                VARCHAR     NOT NULL DEFAULT 'pending',
    consumed_by           VARCHAR     NULL,
    influenced_skill_ids  JSON        NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted            BOOLEAN     NOT NULL DEFAULT FALSE,
    _created_by_id        VARCHAR     NULL,
    _last_updated_by_id   VARCHAR     NULL
);

-- Primary access pattern: list_experiences(agent_id, status='pending').
CREATE INDEX IF NOT EXISTS ix_skill_experience_agent_status
    ON skill_experience (agent_id, status);

-- Organization-level query optimization (mirrors procedural_memory).
CREATE INDEX IF NOT EXISTS ix_skill_experience_organization_id
    ON skill_experience (organization_id);

COMMIT;
