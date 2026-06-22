-- Migration: create skill_evolution_record table.
--
-- Run once on existing databases. New databases get this via SQLAlchemy
-- create_all at server startup (the ORM class is imported in
-- mirix/orm/__init__.py), so there is no separate "phase 2" here.
--
-- Purpose: durable store for the distilled per-round success/failure records
-- (C2) produced by the C1 distiller and consumed every N rounds by the C3
-- curator. Records flow pending -> consumed | superseded.
--
-- Entire migration is idempotent and safe to re-run.

BEGIN;

CREATE TABLE IF NOT EXISTS skill_evolution_record (
    id                    VARCHAR     PRIMARY KEY,
    organization_id       VARCHAR     REFERENCES organizations(id),
    user_id               VARCHAR     NOT NULL REFERENCES users(id),
    agent_id              VARCHAR     NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    day                   VARCHAR     NOT NULL,
    round_id              VARCHAR     NOT NULL,
    round_index           INTEGER     NOT NULL,
    -- record_type / status are plain strings (NOT pg ENUMs); their value
    -- spaces are validated in mirix/schemas/skill_evolution_record.py, so
    -- adding a value never requires a DB migration.
    record_type           VARCHAR     NOT NULL,
    title                 VARCHAR     NOT NULL,
    -- description/detail/evidence_round_ids/quality_score/generality are NOT
    -- NULL with defaults to match the ORM + the pydantic full schema, which
    -- treat them as required. A NULL here would break to_pydantic().
    description           TEXT        NOT NULL DEFAULT '',
    detail                TEXT        NOT NULL DEFAULT '',
    evidence_round_ids    JSON        NOT NULL DEFAULT '[]',
    quality_score         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    generality            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    status                VARCHAR     NOT NULL DEFAULT 'pending',
    consumed_by           VARCHAR     NULL,
    influenced_skill_ids  JSON        NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted            BOOLEAN     NOT NULL DEFAULT FALSE,
    _created_by_id        VARCHAR     NULL,
    _last_updated_by_id   VARCHAR     NULL
);

-- Primary access pattern: list_pending(agent_id, status='pending', round_index).
CREATE INDEX IF NOT EXISTS ix_skill_evolution_record_agent_status
    ON skill_evolution_record (agent_id, status, round_index);

-- Organization-level query optimization (mirrors procedural_memory).
CREATE INDEX IF NOT EXISTS ix_skill_evolution_record_organization_id
    ON skill_evolution_record (organization_id);

COMMIT;
