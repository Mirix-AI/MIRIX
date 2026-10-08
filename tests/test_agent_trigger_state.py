"""Unit tests for the skill-learning trigger's cursor row and claim surface.

Covers:
- Pydantic schema validation for AgentTriggerState and its trigger_type rules.
- ORM shape: table name, columns, unique constraint, index.
- Manager surface: the claim contract; the old sealing machinery is gone.
- memory_tools.trigger_memory_update no longer triggers skill learning.
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
    """The trigger-claim contract the ingestion seam, close endpoint, sweeper
    and flush depend on. Behavior is covered DB-backed in
    tests/test_skill_learning_lifecycle.py (SQLite) and
    tests/test_agent_trigger_state_integration.py (Postgres)."""

    def test_exposes_async_methods(self):
        mgr = AgentTriggerStateManager
        for name in ("count_pending_sessions", "check_and_claim_fire"):
            assert hasattr(mgr, name), f"manager missing {name}"
            unwrapped = inspect.unwrap(getattr(mgr, name))
            assert inspect.iscoroutinefunction(unwrapped), f"{name} must be async"

    def test_check_and_claim_fire_signature(self):
        params = inspect.signature(AgentTriggerStateManager.check_and_claim_fire).parameters
        for name in (
            "agent_id",
            "user_id",
            "organization_id",
            "trigger",
            "threshold",
            "force",
            "allow_empty",
            "limit",
            "trigger_type",
        ):
            assert name in params, f"check_and_claim_fire missing kwarg {name}"

    def test_claim_result_shape(self):
        # just_installed is gone: the first check of a scope evaluates the
        # threshold like any other.
        fields = {f.name for f in __import__("dataclasses").fields(ClaimFireResult)}
        assert fields == {
            "fired",
            "run_id",
            "session_ids",
            "pending_count",
            "pending_experiences",  # What a flush found for the curator
            "state",
        }

    def test_uses_select_for_update(self):
        # Concurrent checks of one scope serialize on the cursor row, and the
        # claimed session rows are locked against a concurrent write.
        src = inspect.getsource(AgentTriggerStateManager._claim_once)
        assert ".with_for_update(" in src
        assert "for_update=True" in src

    def test_sealing_machinery_is_gone(self):
        # No "a newer session seals the older one" rule anywhere.
        import mirix.services.agent_trigger_state_manager as module

        src = inspect.getsource(module)
        for gone in (
            "_aggregate_sealed_undistilled",
            "count_sealed_undistilled_sessions",
            "grouped[:-1]",
            "just_installed",
            "sealed",
        ):
            assert gone not in src, f"obsolete sealing fragment {gone!r} present"

    def test_claim_excludes_all_live_runs_and_records_watermarks(self):
        src = inspect.getsource(AgentTriggerStateManager._claim_once)
        assert "exclude_claimed=True" in src
        assert "session_watermarks" in src
        assert "create_run" in src


class TestClaimRetry:
    """A claim transaction that hits a transient database conflict (deadlock,
    serialization failure, SQLite lock) is retried from scratch."""

    @staticmethod
    def _db_error(message):
        from sqlalchemy.exc import DBAPIError

        return DBAPIError("SELECT 1", {}, Exception(message))

    @staticmethod
    def _manager():
        from contextlib import asynccontextmanager

        class _NoLock:
            @asynccontextmanager
            async def acquire(self, key):
                yield

        mgr = AgentTriggerStateManager(session_maker=object(), scope_locks=_NoLock())

        async def no_cursor(**kwargs):
            return None

        mgr._ensure_cursor = no_cursor
        return mgr

    @pytest.mark.asyncio
    async def test_transient_conflict_is_retried(self, monkeypatch):
        import mirix.services.agent_trigger_state_manager as module

        monkeypatch.setattr(module.asyncio, "sleep", _instant_sleep)
        mgr = self._manager()
        attempts = []

        async def flaky(**kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise self._db_error("deadlock detected")
            return ClaimFireResult(fired=True, run_id="slrun-1")

        mgr._claim_once = flaky
        claim = await mgr.check_and_claim_fire(agent_id="a", user_id="u", organization_id="o")
        assert claim.run_id == "slrun-1" and len(attempts) == 3

    @pytest.mark.asyncio
    async def test_other_errors_are_not_retried(self):
        mgr = self._manager()
        attempts = []

        async def broken(**kwargs):
            attempts.append(1)
            raise self._db_error("relation does not exist")

        mgr._claim_once = broken
        with pytest.raises(Exception):
            await mgr.check_and_claim_fire(agent_id="a", user_id="u", organization_id="o")
        assert len(attempts) == 1

    @pytest.mark.asyncio
    async def test_retries_are_bounded(self, monkeypatch):
        import mirix.services.agent_trigger_state_manager as module

        monkeypatch.setattr(module.asyncio, "sleep", _instant_sleep)
        mgr = self._manager()
        attempts = []

        async def always_locked(**kwargs):
            attempts.append(1)
            raise self._db_error("database is locked")

        mgr._claim_once = always_locked
        with pytest.raises(Exception):
            await mgr.check_and_claim_fire(agent_id="a", user_id="u", organization_id="o")
        assert len(attempts) == module._CLAIM_ATTEMPTS


async def _instant_sleep(seconds):
    return None


# ----------------------------- memory_tools wiring -----------------------


class TestMemoryToolsWiring:
    """The skill-learning trigger no longer lives in the meta agent's
    ``trigger_memory_update`` tool: it runs deterministically after
    task writes, explicit closes and in the sweeper. Conversation dispatch is
    unchanged."""

    @staticmethod
    def _source() -> str:
        from mirix.functions.function_sets import memory_tools

        return inspect.getsource(memory_tools.trigger_memory_update)

    def test_no_trigger_in_the_tool(self):
        src = self._source()
        for gone in (
            "check_and_claim_fire",
            "AgentTriggerStateManager",
            "SKILL_TRIGGER_SESSION_THRESHOLD",
            "_schedule_experience_auto_dream",
            "_current_step_session_id",
        ):
            assert gone not in src, f"task-session trigger remnant {gone!r} present"

    def test_fire_and_forget_scheduler_is_gone(self):
        from mirix.functions.function_sets import memory_tools

        # The unreferenced asyncio.create_task scheduler is deleted; runs
        # are scheduled with a retained reference by skill_learning_trigger.
        assert not hasattr(memory_tools, "_schedule_experience_auto_dream")
        assert "asyncio.create_task(_run_dream" not in inspect.getsource(memory_tools)

    def test_procedural_still_filtered_and_no_task_branch(self):
        # Task sessions never reach the meta agent (the add path stores them in
        # the conversation store only; the queue worker drops any that is
        # queued), so the tool has no per-tag branch left — only the procedural
        # filter.
        src = self._source()
        assert 'mt != "procedural"' in src
        assert "SESSION_TAG_TASK" not in src
        assert "session_tag" not in src


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
