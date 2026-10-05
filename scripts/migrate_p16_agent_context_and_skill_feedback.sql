-- Migration: a session's agent context, and per-session feedback on served
-- skills.
--
-- Run once on existing PostgreSQL databases. New databases get both via
-- SQLAlchemy create_all (mirix/orm/conversation_session.py,
-- mirix/orm/skill_session_feedback.py).
--
-- conversation_session.agent_context: how the agent's prompt was built (its
-- prompt template / setup) as the client attached it to the session with an
-- add of its turns (optional; the latest one given wins). It is not a
-- conversation turn: the skill distiller shows it beside the transcript.
-- Existing sessions have none (NULL) and behave as before.
--
-- skill_session_feedback: one row per (distilled session, served skill) —
-- whether the agent followed the skill and with what effect, as the distiller
-- judged it from the transcript (each with its quote), the version served and
-- the session's outcome. The curator reads it as each skill's usage record.
-- Nothing is backfilled: earlier sessions were distilled without it.
--
-- Idempotent: ADD COLUMN / CREATE TABLE / CREATE INDEX IF NOT EXISTS.

BEGIN;

ALTER TABLE conversation_session ADD COLUMN IF NOT EXISTS agent_context TEXT NULL;

CREATE TABLE IF NOT EXISTS skill_session_feedback (
    id VARCHAR NOT NULL PRIMARY KEY,
    organization_id VARCHAR NULL REFERENCES organizations (id),
    user_id VARCHAR NOT NULL REFERENCES users (id),
    agent_id VARCHAR NULL,
    session_id VARCHAR(64) NOT NULL,
    learning_run_id VARCHAR NULL,
    skill_id VARCHAR NOT NULL REFERENCES procedural_memory (id) ON DELETE CASCADE,
    skill_name VARCHAR NULL,
    skill_version_no INTEGER NULL,
    served_count INTEGER NOT NULL DEFAULT 1,
    session_outcome VARCHAR NULL,
    followed VARCHAR NOT NULL DEFAULT 'unclear',
    followed_evidence TEXT NULL,
    followed_source VARCHAR NULL,
    stated_followed VARCHAR NULL,
    effect VARCHAR NOT NULL DEFAULT 'unclear',
    effect_evidence TEXT NULL,
    effect_source VARCHAR NULL,
    stated_effect VARCHAR NULL,
    created_at TIMESTAMPTZ NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NULL DEFAULT now(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    _created_by_id VARCHAR NULL,
    _last_updated_by_id VARCHAR NULL
);

CREATE INDEX IF NOT EXISTS ix_skill_session_feedback_owner_skill
    ON skill_session_feedback (organization_id, user_id, skill_id);
CREATE INDEX IF NOT EXISTS ix_skill_session_feedback_owner_session
    ON skill_session_feedback (organization_id, user_id, session_id);

COMMIT;
