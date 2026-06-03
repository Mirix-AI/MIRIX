"""Unit tests for the session-based procedural-memory trigger.

Covers:
- Pydantic schema validation for AgentTriggerState and its trigger_type rules.
- ORM shape: table name, columns, unique constraint, index.
- Manager surface: exposes the expected async methods.
- memory_tools.trigger_memory_update wires to the new manager/threshold.
- Constants: SKILL_TRIGGER_SESSION_THRESHOLD is defined and env-overridable.
- Migration SQL creates the expected table.

These are DB-free unit tests. DB-level UPSERT behavior is covered indirectly by
reading the manager source; an integration test that requires a running Postgres
is intentionally omitted here to keep this file in the fast lane.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from sqlalchemy import inspect as sa_inspect

from mirix import constants
from mirix.orm.agent_trigger_state import AgentTriggerState as AgentTriggerStateORM
from mirix.schemas.agent_trigger_state import (
    KNOWN_TRIGGER_TYPES,
    TRIGGER_TYPE_MAX_LEN,
    TRIGGER_TYPE_PATTERN,
    TRIGGER_TYPE_PROCEDURAL_SKILL,
    AgentTriggerState as PydanticAgentTriggerState,
    _validate_trigger_type,
)
from mirix.services.agent_trigger_state_manager import (
    AgentTriggerStateManager,
    ClaimFireResult,
)


# ----------------------------- Schema validation -------------------------


class TestTriggerTypeValidation:
    def test_procedural_skill_is_registered(self):
        assert TRIGGER_TYPE_PROCEDURAL_SKILL in KNOWN_TRIGGER_TYPES

    def test_accepts_known_trigger_type(self):
        assert _validate_trigger_type(TRIGGER_TYPE_PROCEDURAL_SKILL) == TRIGGER_TYPE_PROCEDURAL_SKILL

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_trigger_type("")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError):
            _validate_trigger_type("a" * (TRIGGER_TYPE_MAX_LEN + 1))

    def test_rejects_uppercase(self):
        # Pattern is lowercase + digits + underscores; reject uppercase to
        # keep the namespace tidy and avoid dupes by case.
        with pytest.raises(ValueError):
            _validate_trigger_type("Procedural_Skill")

    def test_rejects_hyphens(self):
        with pytest.raises(ValueError):
            _validate_trigger_type("procedural-skill")

    def test_pattern_matches_spec(self):
        # Sanity-check the regex itself rejects a leading digit.
        assert re.fullmatch(TRIGGER_TYPE_PATTERN, "1bad") is None
        assert re.fullmatch(TRIGGER_TYPE_PATTERN, "a") is not None

    def test_rejects_unregistered_but_syntactically_valid(self):
        # KNOWN_TRIGGER_TYPES is the source of truth; a syntactically valid
        # but unregistered name must be rejected so typos cannot create a
        # parallel bookkeeping row that no one watches.
        assert "future_trigger" not in KNOWN_TRIGGER_TYPES
        assert re.fullmatch(TRIGGER_TYPE_PATTERN, "future_trigger") is not None
        with pytest.raises(ValueError):
            _validate_trigger_type("future_trigger")


class TestPydanticAgentTriggerState:
    def test_accepts_minimal_valid(self):
        s = PydanticAgentTriggerState(
            agent_id="agent-1",
            user_id="user-1",
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
        )
        assert s.agent_id == "agent-1"
        assert s.last_fired_at is None
        assert s.last_fired_session_id is None

    def test_optional_cursor_fields(self):
        from datetime import datetime, timezone

        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s = PydanticAgentTriggerState(
            agent_id="agent-1",
            user_id="user-1",
            trigger_type=TRIGGER_TYPE_PROCEDURAL_SKILL,
            last_fired_at=ts,
            last_fired_session_id="sess-abc",
        )
        assert s.last_fired_at == ts
        assert s.last_fired_session_id == "sess-abc"

    def test_rejects_invalid_trigger_type(self):
        with pytest.raises(ValueError):
            PydanticAgentTriggerState(
                agent_id="agent-1",
                user_id="user-1",
                trigger_type="BAD",
            )


# ----------------------------- ORM shape ---------------------------------


class TestAgentTriggerStateORM:
    def _column_names(self):
        mapper = sa_inspect(AgentTriggerStateORM)
        return {col.key for col in mapper.columns}

    def test_table_name(self):
        assert AgentTriggerStateORM.__tablename__ == "agent_trigger_state"

    def test_has_required_columns(self):
        cols = self._column_names()
        expected = {
            "id",
            "organization_id",
            "user_id",
            "agent_id",
            "trigger_type",
            "last_fired_at",
            "last_fired_session_id",
            "last_fired_tied_session_ids",
            "created_at",
            "updated_at",
            "is_deleted",
        }
        assert expected.issubset(cols), f"missing columns: {expected - cols}"

    def test_trigger_type_column_is_bounded(self):
        col = AgentTriggerStateORM.__table__.c.trigger_type
        assert col.type.length == TRIGGER_TYPE_MAX_LEN

    def test_unique_constraint_covers_triple(self):
        uqs = [
            c for c in AgentTriggerStateORM.__table__.constraints
            if c.__class__.__name__ == "UniqueConstraint"
        ]
        names = {u.name for u in uqs}
        assert "uq_agent_trigger_state_agent_user_type" in names
        target = next(u for u in uqs if u.name == "uq_agent_trigger_state_agent_user_type")
        covered = {c.name for c in target.columns}
        assert covered == {"agent_id", "user_id", "trigger_type"}

    def test_has_composite_index(self):
        idx_names = {i.name for i in AgentTriggerStateORM.__table__.indexes}
        assert "ix_agent_trigger_state_agent_user_type" in idx_names

    def test_is_registered_in_orm_package(self):
        import mirix.orm as orm_pkg

        assert "AgentTriggerState" in orm_pkg.__all__
        assert orm_pkg.AgentTriggerState is AgentTriggerStateORM

    def test_pydantic_model_link(self):
        # SqlalchemyBase.to_pydantic() dispatches via __pydantic_model__, so
        # this link must point at the pydantic class to avoid runtime errors.
        assert AgentTriggerStateORM.__pydantic_model__ is PydanticAgentTriggerState


# ----------------------------- Manager surface ---------------------------


class TestAgentTriggerStateManagerSurface:
    """Ensure the manager exposes the contract memory_tools depends on.

    Wire-level correctness (UPSERT etc.) needs a real DB and is covered by
    integration tests; here we just guard the public shape.
    """

    def test_exposes_async_methods(self):
        # The manager methods are decorated by @enforce_types, whose wrapper
        # is sync; unwrap to inspect the underlying async def.
        mgr = AgentTriggerStateManager
        for name in (
            "get_state",
            "count_distinct_sessions_since",
            "check_and_claim_fire",
        ):
            assert hasattr(mgr, name), f"manager missing {name}"
            method = getattr(mgr, name)
            unwrapped = inspect.unwrap(method)
            assert inspect.iscoroutinefunction(unwrapped), (
                f"{name} must be async under its decorators"
            )

    def test_check_and_claim_fire_signature(self):
        sig = inspect.signature(AgentTriggerStateManager.check_and_claim_fire)
        params = sig.parameters
        for name in (
            "agent_id",
            "user_id",
            "trigger_type",
            "threshold",
            "organization_id",
            "current_session_id",
        ):
            assert name in params, f"check_and_claim_fire missing kwarg {name}"

    def test_claim_result_shape(self):
        # ClaimFireResult is the contract with memory_tools: if a field
        # renames silently, the fire-path branches will read stale data.
        fields = {f.name for f in __import__("dataclasses").fields(ClaimFireResult)}
        assert fields == {"fired", "sessions_since", "just_installed", "state"}

    def test_count_signature_accepts_expected_kwargs(self):
        sig = inspect.signature(AgentTriggerStateManager.count_distinct_sessions_since)
        params = sig.parameters
        for name in (
            "agent_id",
            "user_id",
            "since",
            "tied_session_ids",
        ):
            assert name in params, f"count_distinct_sessions_since missing kwarg {name}"
        # exclude_session_id was only meaningful under MAX windowing; MIN
        # filter subsumes it (the in-progress session's MIN is <= cursor).
        assert "exclude_session_id" not in params

    def test_uses_select_for_update(self):
        # The whole point of check_and_claim_fire is serializing two workers
        # on the same cursor row — regress-guard against the lock hint being
        # dropped during refactors.
        src = inspect.getsource(AgentTriggerStateManager.check_and_claim_fire)
        assert ".with_for_update(" in src, (
            "check_and_claim_fire must lock the cursor row with SELECT FOR UPDATE"
        )

    def test_excludes_soft_deleted_messages(self):
        # Messages carry an is_deleted flag from CommonSqlalchemyMetaMixins.
        # Leaving them in the count lets a user's deleted history keep
        # firing procedural extraction — mirror message_manager's
        # `is_deleted == False` convention.
        agg_src = inspect.getsource(AgentTriggerStateManager._aggregate_window)
        assert "MessageModel.is_deleted.is_(False)" in agg_src, (
            "aggregate must exclude soft-deleted messages"
        )

    def test_uses_min_semantics_not_max(self):
        # MIN-based windowing is what prevents double-counting a session
        # whose last message happens to fall in a future window. Each
        # session's MIN(created_at) is immutable once the first message is
        # inserted, so the HAVING filter `MIN > cursor` matches exactly
        # one window. Reverting to MAX reintroduces that bug.
        agg_src = inspect.getsource(AgentTriggerStateManager._aggregate_window)
        assert "func.min(MessageModel.created_at)" in agg_src
        assert "func.max(MessageModel.created_at)" not in agg_src
        # Sanity: the filter clamps per-session MIN against the cursor via
        # HAVING, not rows via WHERE — MIN is an aggregate.
        assert ".having(" in agg_src

    def test_count_and_tied_ids_from_single_query(self):
        # count, watermark, and tied_ids MUST all come from one aggregate
        # query. Running them as separate queries under READ COMMITTED
        # allows a message that commits between them to land in tied_ids
        # without being counted, which would drop its session forever.
        agg_src = inspect.getsource(AgentTriggerStateManager._aggregate_window)
        assert "group_by(MessageModel.session_id)" in agg_src, (
            "tied-set and count must be derived from one GROUP BY query"
        )
        # And there must be no separate tied-ids query hanging around.
        mgr_src = inspect.getsource(AgentTriggerStateManager)
        assert "_tied_session_ids_at" not in mgr_src

    def test_window_uses_tie_breaker(self):
        # Guards against regressing to the naive `MIN > cursor` filter,
        # which silently drops the rare case of a session whose first
        # message has the exact watermark timestamp but committed after
        # our SELECT.
        agg_src = inspect.getsource(AgentTriggerStateManager._aggregate_window)
        # Strictly-newer sessions are captured via `>`.
        assert "per_session_min > since" in agg_src
        # Sessions with MIN == cursor are captured via the tie-breaker,
        # OR via the `>=` shortcut when no tied set exists.
        assert "per_session_min >= since" in agg_src
        assert "per_session_min == since" in agg_src
        assert "session_id.notin_(tied_session_ids)" in agg_src

    def test_advance_records_tied_set(self):
        # When a fire claims, the new cursor must record both the watermark
        # timestamp AND the session_ids tied to it. Missing tied set =>
        # next window silently loses any concurrent insert at the same ts.
        src = inspect.getsource(AgentTriggerStateManager.check_and_claim_fire)
        assert "last_fired_tied_session_ids = tied_ids" in src
        # Tied ids come from the same aggregate query as the count, so
        # they are mutually consistent with the counted set.
        agg_src = inspect.getsource(AgentTriggerStateManager._aggregate_window)
        assert "row.first_ts == watermark" in agg_src


# ----------------------------- memory_tools wiring -----------------------


class TestMemoryToolsWiring:
    """Guard the integration point inside trigger_memory_update.

    We inspect source text so the test stays DB-free. If the wiring ever
    shifts to a different symbol, these assertions will flag it.
    """

    @staticmethod
    def _source() -> str:
        from mirix.functions.function_sets import memory_tools

        return inspect.getsource(memory_tools.trigger_memory_update)

    def test_uses_session_threshold(self):
        src = self._source()
        assert "SKILL_TRIGGER_SESSION_THRESHOLD" in src
        # The old per-chunk message-count heuristic must be gone.
        assert "SKILL_TRIGGER_MESSAGE_THRESHOLD" not in src
        assert "[USER]" not in src and "[ASSISTANT]" not in src

    def test_uses_new_manager_and_trigger_type(self):
        src = self._source()
        assert "AgentTriggerStateManager" in src
        assert "TRIGGER_TYPE_PROCEDURAL_SKILL" in src

    def test_uses_atomic_check_and_claim_fire(self):
        # The fire path must go through the single atomic entry point — not
        # a separate read-then-write, which reintroduces the double-fire race.
        src = self._source()
        assert "check_and_claim_fire" in src
        assert "record_fire" not in src
        assert "count_distinct_sessions_since" not in src

    def test_reads_current_session_id_from_agent(self):
        # The fire event must record the session in progress so the next
        # count excludes it (no double counting of the open session).
        src = self._source()
        assert "_current_step_session_id" in src

    def test_skips_when_no_user(self):
        # Counter is keyed per-user; without a user we must not bookkeep a
        # fire event, or users would share a single cursor unexpectedly.
        src = self._source()
        assert "Skipping session-based procedural trigger" in src


# ----------------------------- Constants ---------------------------------


class TestThresholdConstant:
    def test_default_is_five(self, monkeypatch):
        # The env var may be set on the developer's machine; reimport under
        # a pristine environment to check the default.
        monkeypatch.delenv("SKILL_TRIGGER_SESSION_THRESHOLD", raising=False)
        import importlib

        mod = importlib.reload(constants)
        assert mod.SKILL_TRIGGER_SESSION_THRESHOLD == 5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SKILL_TRIGGER_SESSION_THRESHOLD", "12")
        import importlib

        mod = importlib.reload(constants)
        try:
            assert mod.SKILL_TRIGGER_SESSION_THRESHOLD == 12
        finally:
            monkeypatch.delenv("SKILL_TRIGGER_SESSION_THRESHOLD", raising=False)
            importlib.reload(constants)


# ----------------------------- Migration SQL -----------------------------


class TestMigrationSql:
    SQL_PATH = Path("scripts/migrate_add_agent_trigger_state.sql")

    def _sql(self) -> str:
        return self.SQL_PATH.read_text()

    def test_creates_table(self):
        sql = self._sql()
        assert "CREATE TABLE IF NOT EXISTS agent_trigger_state" in sql

    def test_declares_unique_and_index(self):
        sql = self._sql()
        assert "uq_agent_trigger_state_agent_user_type" in sql
        assert "UNIQUE (agent_id, user_id, trigger_type)" in sql
        assert "ix_agent_trigger_state_agent_user_type" in sql

    def test_has_tied_session_ids_column(self):
        sql = self._sql()
        # Present both in the CREATE TABLE ... and the back-compat ALTER
        # so deployments that created the table before this fix get it too.
        assert "last_fired_tied_session_ids" in sql
        assert "ADD COLUMN IF NOT EXISTS last_fired_tied_session_ids" in sql

    def test_trigger_type_column_length_matches_schema(self):
        sql = self._sql()
        # Tolerate varying column alignment; only check that the SQL
        # declares trigger_type with the shared max length.
        assert re.search(
            rf"\btrigger_type\s+VARCHAR\({TRIGGER_TYPE_MAX_LEN}\)", sql
        ), "SQL column length must mirror TRIGGER_TYPE_MAX_LEN"

    def test_agent_fk_cascades(self):
        # Deleting an agent must not leave orphan trigger-state rows behind.
        sql = self._sql()
        assert "REFERENCES agents(id) ON DELETE CASCADE" in sql
