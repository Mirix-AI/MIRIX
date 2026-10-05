-- Migration: create skill_learning_run and retrieval_log.
--
-- Run once on existing PostgreSQL databases. New databases get both tables via
-- SQLAlchemy create_all (mirix/orm/skill_learning_run.py, mirix/orm/retrieval_log.py).
--
--   skill_learning_run  one row per skill-learner execution (distill -> curate)
--                       for one (organization, user, meta agent) scope — and
--                       one per auto_dream run of any other mode: trigger
--                       (threshold | close | sweep | flush | manual), mode
--                       (procedural | core | episodic | semantic | resource |
--                       knowledge | experience), status
--                       (queued | running | succeeded | failed), the claimed
--                       session_ids with their watermarks, the lease heartbeat,
--                       per-session distill and per-batch curator metrics, LLM
--                       usage, chunked embedding calls and every retry
--                       (retry_events).
--   retrieval_log       one row per memory retrieval (its id is returned as
--                       retrieval_id): raw query, extracted topics and status,
--                       query-embedding status, per-memory-type ranking trace,
--                       usage, latency, every retry (retry_events) and the
--                       chunked embedding calls (embedding_chunked_calls). Kept MIRIX_RETRIEVAL_LOG_RETENTION_DAYS
--                       (default 30) and deleted with the user's data.
--
-- Idempotent: CREATE TABLE / INDEX IF NOT EXISTS.

BEGIN;

CREATE TABLE IF NOT EXISTS skill_learning_run (
    id                       VARCHAR      PRIMARY KEY,
    organization_id          VARCHAR      REFERENCES organizations(id),
    user_id                  VARCHAR      NOT NULL REFERENCES users(id),
    agent_id                 VARCHAR      NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    -- trigger / mode / status are plain strings (NOT pg ENUMs), validated in
    -- mirix/schemas/skill_learning_run.py.
    "trigger"                VARCHAR      NOT NULL,
    mode                     VARCHAR      NOT NULL DEFAULT 'procedural',
    status                   VARCHAR      NOT NULL DEFAULT 'queued',
    session_ids              JSON         NOT NULL DEFAULT '[]',
    session_watermarks       JSON         NOT NULL DEFAULT '{}',
    heartbeat_at             TIMESTAMPTZ  NULL,
    started_at               TIMESTAMPTZ  NULL,
    finished_at              TIMESTAMPTZ  NULL,
    error                    TEXT         NULL,
    distill_metrics          JSON         NULL,
    curator_metrics          JSON         NULL,
    llm_usage                JSON         NULL,
    embedding_chunked_calls  INTEGER      NOT NULL DEFAULT 0,
    retry_events             JSON         NOT NULL DEFAULT '[]',
    created_at               TIMESTAMPTZ  DEFAULT now(),
    updated_at               TIMESTAMPTZ  DEFAULT now(),
    is_deleted               BOOLEAN      NOT NULL DEFAULT FALSE,
    _created_by_id           VARCHAR      NULL,
    _last_updated_by_id      VARCHAR      NULL
);

-- A table created by an earlier revision of this script lacks `mode` and
-- `retry_events`.
ALTER TABLE skill_learning_run
    ADD COLUMN IF NOT EXISTS mode VARCHAR NOT NULL DEFAULT 'procedural';
ALTER TABLE skill_learning_run
    ADD COLUMN IF NOT EXISTS retry_events JSON NOT NULL DEFAULT '[]';

-- Active runs of one scope (claim exclusion, wait-until-idle).
CREATE INDEX IF NOT EXISTS ix_skill_learning_run_scope_status
    ON skill_learning_run (organization_id, user_id, agent_id, status);

-- Lease reaper: queued/running runs whose heartbeat is stale.
CREATE INDEX IF NOT EXISTS ix_skill_learning_run_status_heartbeat
    ON skill_learning_run (status, heartbeat_at);

-- GET /memory/learning_runs, newest first.
CREATE INDEX IF NOT EXISTS ix_skill_learning_run_org_user_created
    ON skill_learning_run (organization_id, user_id, created_at);

CREATE TABLE IF NOT EXISTS retrieval_log (
    id                   VARCHAR      PRIMARY KEY,
    organization_id      VARCHAR      REFERENCES organizations(id),
    user_id              VARCHAR      NOT NULL REFERENCES users(id),
    -- Plain strings (no FK): a log row must not block deleting a client.
    client_id            VARCHAR      NULL,
    session_id           VARCHAR(64)  NULL,
    memory_types         JSON         NULL,
    query_text           TEXT         NULL,
    extracted_topics     TEXT         NULL,
    extraction_status    VARCHAR      NULL,
    embedding_status     VARCHAR      NULL,
    per_type             JSON         NULL,
    llm_usage            JSON         NULL,
    latency_ms           INTEGER      NULL,
    retry_events         JSON         NOT NULL DEFAULT '[]',
    embedding_chunked_calls INTEGER   NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ  DEFAULT now(),
    updated_at           TIMESTAMPTZ  DEFAULT now(),
    is_deleted           BOOLEAN      NOT NULL DEFAULT FALSE,
    _created_by_id       VARCHAR      NULL,
    _last_updated_by_id  VARCHAR      NULL
);

-- A table created by an earlier revision of this script lacks `retry_events`.
ALTER TABLE retrieval_log
    ADD COLUMN IF NOT EXISTS retry_events JSON NOT NULL DEFAULT '[]';

-- ... and `embedding_chunked_calls` (chunked embedding calls of a
-- retrieval are counted on its log row).
ALTER TABLE retrieval_log
    ADD COLUMN IF NOT EXISTS embedding_chunked_calls INTEGER NOT NULL DEFAULT 0;

-- GET /memory/retrieval_logs for one user, newest first.
CREATE INDEX IF NOT EXISTS ix_retrieval_log_org_user_created
    ON retrieval_log (organization_id, user_id, created_at);

-- GET /memory/retrieval_logs lists one client's rows, newest first; a client's
-- erasure deletes them by client_id.
CREATE INDEX IF NOT EXISTS ix_retrieval_log_org_client_created
    ON retrieval_log (organization_id, client_id, created_at);

CREATE INDEX IF NOT EXISTS ix_retrieval_log_session_id
    ON retrieval_log (session_id);

-- Retention sweep: DELETE ... WHERE created_at < cutoff.
CREATE INDEX IF NOT EXISTS ix_retrieval_log_created_at
    ON retrieval_log (created_at);

COMMIT;
