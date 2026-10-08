-- Migration: create conversation_session and backfill it from conversation_message.
--
-- Run once on existing PostgreSQL databases (plain `psql -f`, or `psql -1`).
-- New databases get the table via SQLAlchemy create_all at server startup (the
-- ORM model is mirix/orm/conversation_session.py); on such a database this
-- script is a no-op apart from backfilling any sessions that lack a row.
--
-- conversation_session is the lifecycle row of every (organization, user,
-- session_id) in the Conversation Message Store: first/last message
-- time, message count, explicit or idle close, reopen count, the meta agent that
-- wrote it, distill failure / backoff bookkeeping (distill_retry_after and the
-- consecutive infrastructure failures that double it) and storage-cap
-- truncation totals. Whether a session still needs distilling stays per message row
-- (conversation_message.distilled_at IS NULL).
--
-- Backfill, one row per (organization_id, user_id, session_id) of
-- live conversation_message rows:
--   * first/last_message_at = MIN/MAX(created_at), message_count = COUNT(*);
--   * session_tag = the tag of the session's EARLIEST row (by created_at, id),
--     NULL included — the first write decides the tag; a value other than
--     'task' / 'conversation' is stored as NULL (legacy, treated as eligible);
--   * agent_id = the meta memory agent that received the session's messages in
--     the agent-loop `messages` table (earliest one). When that history is gone
--     (message retention keeps only the last few sessions per agent): the only
--     live meta agent of the only client that wrote the session's rows
--     (conversation_message._created_by_id); with no recorded writer at all,
--     the organization's only live meta agent. Otherwise NULL — the server
--     applies the same rule at runtime (ConversationSessionManager.
--     adopt_unattributed_sessions), and a session nobody can be shown to own is
--     never learned from;
--   * sessions whose last message is older than the idle timeout are written as
--     already closed: closed_at = last_message_at + timeout, close_reason='idle'.
--     The timeout defaults to 172800 s (MIRIX_SESSION_IDLE_TIMEOUT_SECONDS'
--     default). If you run the server with another value, set it first:
--         SET mirix.session_idle_timeout_seconds = '3600';
--
-- Idempotent: every object is created IF NOT EXISTS / guarded, and the backfill
-- only inserts sessions that have no row yet, so re-running changes nothing.
-- gen_random_uuid() is built in from PostgreSQL 13.

BEGIN;

CREATE TABLE IF NOT EXISTS conversation_session (
    id                       VARCHAR      PRIMARY KEY,
    organization_id          VARCHAR      REFERENCES organizations(id),
    user_id                  VARCHAR      NOT NULL REFERENCES users(id),
    -- Plain string, no FK: backfilled sessions may not be attributable to an
    -- agent, and the column must survive agent re-creation.
    agent_id                 VARCHAR      NULL,
    session_id               VARCHAR(64)  NOT NULL,
    session_tag              VARCHAR      NULL,
    first_message_at         TIMESTAMPTZ  NOT NULL,
    last_message_at          TIMESTAMPTZ  NOT NULL,
    message_count            INTEGER      NOT NULL DEFAULT 0,
    closed_at                TIMESTAMPTZ  NULL,
    close_reason             VARCHAR      NULL,
    reopen_count             INTEGER      NOT NULL DEFAULT 0,
    distill_failures         INTEGER      NOT NULL DEFAULT 0,
    distill_failed_at        TIMESTAMPTZ  NULL,
    distill_retry_after      TIMESTAMPTZ  NULL,
    distill_infra_failures   INTEGER      NOT NULL DEFAULT 0,
    truncated_message_count  INTEGER      NOT NULL DEFAULT 0,
    truncated_chars          BIGINT       NOT NULL DEFAULT 0,
    -- CommonSqlalchemyMetaMixins columns, matching sibling tables.
    created_at               TIMESTAMPTZ  DEFAULT now(),
    updated_at               TIMESTAMPTZ  DEFAULT now(),
    is_deleted               BOOLEAN      NOT NULL DEFAULT FALSE,
    _created_by_id           VARCHAR      NULL,
    _last_updated_by_id      VARCHAR      NULL
);

-- A table created by an earlier revision of this script lacks the counter
-- that doubles the infrastructure-error backoff.
ALTER TABLE conversation_session
    ADD COLUMN IF NOT EXISTS distill_infra_failures INTEGER NOT NULL DEFAULT 0;

-- One lifecycle row per (organization, user, session). Guarded so a re-run is a
-- no-op; mirrors the ORM UniqueConstraint.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_conversation_session_org_user_session'
          AND conrelid = 'conversation_session'::regclass
    ) THEN
        ALTER TABLE conversation_session
            ADD CONSTRAINT uq_conversation_session_org_user_session
            UNIQUE (organization_id, user_id, session_id);
    END IF;
END$$;

-- Same session_id format rule as conversation_message (mirix.schemas.message).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_conversation_session_session_id_format'
          AND conrelid = 'conversation_session'::regclass
    ) THEN
        ALTER TABLE conversation_session
            ADD CONSTRAINT ck_conversation_session_session_id_format
            CHECK (session_id ~ '^[A-Za-z0-9_-]{1,64}$');
    END IF;
END$$;

-- Trigger counting / flush claim: pending task sessions of one scope.
CREATE INDEX IF NOT EXISTS ix_conversation_session_scope
    ON conversation_session (organization_id, user_id, agent_id, session_tag);

-- Sweep: open sessions whose last message is older than the idle cutoff.
CREATE INDEX IF NOT EXISTS ix_conversation_session_closed_last_message
    ON conversation_session (closed_at, last_message_at);

-- Backfill.
WITH cfg AS (
    SELECT make_interval(secs => COALESCE(
        NULLIF(current_setting('mirix.session_idle_timeout_seconds', true), '')::double precision,
        172800
    )) AS idle
),
sessions AS (
    SELECT
        cm.organization_id,
        cm.user_id,
        cm.session_id,
        MIN(cm.created_at) AS first_at,
        MAX(cm.created_at) AS last_at,
        COUNT(*)           AS n,
        (array_agg(cm.session_tag ORDER BY cm.created_at, cm.id))[1] AS first_tag,
        -- Writing clients (non-NULL _created_by_id): the fallback attribution.
        COUNT(DISTINCT cm._created_by_id) AS n_writers,
        MIN(cm._created_by_id)            AS writer
    FROM conversation_message cm
    WHERE cm.is_deleted IS NOT TRUE
    GROUP BY cm.organization_id, cm.user_id, cm.session_id
),
-- Live meta agents per owning client, and organizations with exactly one.
live_meta AS (
    SELECT a.id, a.organization_id, a._created_by_id AS owner
    FROM agents a
    WHERE a.agent_type = 'meta_memory_agent' AND a.is_deleted IS NOT TRUE
),
owner_agent AS (
    SELECT owner, MIN(id) AS agent_id FROM live_meta
    WHERE owner IS NOT NULL
    GROUP BY owner HAVING COUNT(*) = 1
),
org_agent AS (
    SELECT organization_id, MIN(id) AS agent_id FROM live_meta
    GROUP BY organization_id HAVING COUNT(*) = 1
),
-- One set-based pass (messages has no index led by session_id, so a per-session
-- correlated lookup would rescan it for every session): the earliest META
-- agent message of each backfilled (user, session).
meta_agents AS (
    SELECT DISTINCT ON (m.user_id, m.session_id)
        m.user_id, m.session_id, m.agent_id
    FROM messages m
    JOIN agents a ON a.id = m.agent_id
    WHERE a.agent_type = 'meta_memory_agent'
      AND (m.user_id, m.session_id) IN (SELECT user_id, session_id FROM sessions)
    ORDER BY m.user_id, m.session_id, m.created_at, m.id
)
INSERT INTO conversation_session (
    id, organization_id, user_id, agent_id, session_id, session_tag,
    first_message_at, last_message_at, message_count,
    closed_at, close_reason,
    reopen_count, distill_failures, truncated_message_count, truncated_chars,
    created_at, updated_at, is_deleted
)
SELECT
    'convsess-' || gen_random_uuid()::text,
    s.organization_id,
    s.user_id,
    COALESCE(
        ma.agent_id,
        CASE WHEN s.n_writers = 1 THEN oa.agent_id END,
        CASE WHEN s.n_writers = 0 THEN ga.agent_id END
    ),
    s.session_id,
    -- Only the two valid tags survive; anything else (only possible through a
    -- misconfigured MIRIX_DEFAULT_SESSION_TAG) becomes legacy NULL, which the
    -- skill learner treats as eligible — the same fail-open as before.
    CASE WHEN s.first_tag IN ('task', 'conversation') THEN s.first_tag END,
    s.first_at,
    s.last_at,
    s.n,
    CASE WHEN s.last_at < now() - cfg.idle THEN s.last_at + cfg.idle END,
    CASE WHEN s.last_at < now() - cfg.idle THEN 'idle' END,
    0, 0, 0, 0,
    now(), now(), FALSE
FROM sessions s
CROSS JOIN cfg
LEFT JOIN meta_agents ma ON ma.user_id = s.user_id AND ma.session_id = s.session_id
LEFT JOIN owner_agent oa ON oa.owner = s.writer
LEFT JOIN org_agent ga ON ga.organization_id = s.organization_id
-- NOT EXISTS (not only ON CONFLICT): unique constraints treat NULLs as
-- distinct, so a NULL organization_id would otherwise be re-inserted on every
-- run.
WHERE NOT EXISTS (
    SELECT 1 FROM conversation_session cs
    WHERE cs.organization_id IS NOT DISTINCT FROM s.organization_id
      AND cs.user_id = s.user_id
      AND cs.session_id = s.session_id
)
ON CONFLICT (organization_id, user_id, session_id) DO NOTHING;

COMMIT;
