"""
Tests for the Conversation Message Store — the single source the procedural
memory (skill) distiller reads.

Layered exactly like the session_id tests:

  * DB-free shape/contract tests (mirroring tests/test_session_id.py): the ORM
    column set, the Pydantic schema shape, role/session_id validation, and the
    Postgres-only CHECK-constraint dialect gating. No DB, no running server.
  * Postgres round-trip tests (marked @pytest.mark.integration, mirroring
    tests/test_session_id_integration.py): record_turns ordering + accumulation,
    count_distinct_sessions, the OLDEST-first sealed-only selection,
    mark_sessions_distilled idempotency, and per-(user, org) scoping isolation.

The DB-free run (`pytest -m "not integration"`) executes only the shape tests;
everything that needs Postgres carries the integration marker.
"""
from __future__ import annotations

import pytest


# ============================ DB-free: ORM shape ============================


class TestConversationMessageOrm:
    """The table/column contract the migration + create_all must satisfy."""

    def test_table_name(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        assert CM.__tablename__ == "conversation_message"

    def test_key_columns_present(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        cols = {c.name for c in CM.__table__.columns}
        # The learnable-turn columns plus the owner scoping + barrier marker.
        for expected in {
            "id",
            "session_id",
            "role",
            "content",
            "distilled_at",
            "user_id",
            "organization_id",
            "created_at",
        }:
            assert expected in cols, f"missing column {expected!r}"

    def test_session_id_is_not_null(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        # This store only holds session'd turns — session_id is NOT NULL.
        assert CM.__table__.c.session_id.nullable is False

    def test_role_and_content_not_null(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        assert CM.__table__.c.role.nullable is False
        assert CM.__table__.c.content.nullable is False

    def test_distilled_at_is_nullable(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        # NULL distilled_at == not yet consumed by a distill round.
        assert CM.__table__.c.distilled_at.nullable is True

    def test_session_id_column_length_matches_constant(self):
        from mirix.orm.conversation_message import ConversationMessage as CM
        from mirix.schemas.message import SESSION_ID_MAX_LEN

        # One source of truth for the session_id length across schema + ORM.
        assert CM.__table__.c.session_id.type.length == SESSION_ID_MAX_LEN

    def test_session_id_is_indexed(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        covered = any(
            any(c.name == "session_id" for c in idx.columns)
            for idx in CM.__table__.indexes
        )
        assert covered, "session_id should be indexed"

    def test_id_default_carries_convmsg_prefix(self):
        from mirix.orm.conversation_message import ConversationMessage as CM

        default = CM.__table__.c.id.default
        # The PK uses a `convmsg-` prefixed callable default (sexp-/proc- convention).
        assert default is not None and default.is_callable
        assert default.arg(None).startswith("convmsg-")


class TestConversationMessageCheckConstraintGating:
    """The session_id CHECK uses Postgres' `~` regex operator, which SQLite
    cannot parse. It must be emitted ONLY for Postgres so SQLite create_all works.
    Mirrors tests/test_session_id.py::TestCheckConstraintDialectGating.
    """

    def test_check_compiles_for_postgresql(self):
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        from mirix.orm.conversation_message import ConversationMessage as CM

        ddl = str(CreateTable(CM.__table__).compile(dialect=postgresql.dialect()))
        assert "ck_conversation_message_session_id_format" in ddl
        assert "~" in ddl  # Postgres regex operator

    def test_check_is_suppressed_for_sqlite(self):
        from sqlalchemy.dialects import sqlite
        from sqlalchemy.schema import CreateTable

        from mirix.orm.conversation_message import ConversationMessage as CM

        ddl = str(CreateTable(CM.__table__).compile(dialect=sqlite.dialect()))
        assert "ck_conversation_message_session_id_format" not in ddl
        assert "session_id ~" not in ddl

    def test_check_constraint_uses_shared_pattern(self):
        from mirix.orm.conversation_message import ConversationMessage as CM
        from mirix.schemas.message import SESSION_ID_SQL_PATTERN

        ck_texts = [
            str(c.sqltext)
            for c in CM.__table__.constraints
            if getattr(c, "name", None) == "ck_conversation_message_session_id_format"
        ]
        assert ck_texts, "ck_conversation_message_session_id_format not found"
        assert SESSION_ID_SQL_PATTERN in ck_texts[0]


# ============================ DB-free: Pydantic shape ======================


class TestConversationMessageSchema:
    def test_full_schema_field_shape(self):
        from mirix.schemas.conversation_message import ConversationMessage

        fields = ConversationMessage.model_fields
        for expected in {
            "id",
            "session_id",
            "role",
            "content",
            "user_id",
            "organization_id",
            "distilled_at",
            "created_at",
            "updated_at",
        }:
            assert expected in fields, f"missing schema field {expected!r}"

    def test_create_carries_owner_ids(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        c = ConversationMessageCreate(
            session_id="sess-1",
            role="user",
            content="hi",
            user_id="user-1",
            organization_id="org-1",
        )
        assert c.role == "user"
        assert c.user_id == "user-1"
        assert c.organization_id == "org-1"

    def test_response_is_full_schema_alias(self):
        from mirix.schemas.conversation_message import (
            ConversationMessage,
            ConversationMessageResponse,
        )

        # Response must expose the same fields as the full schema.
        assert (
            set(ConversationMessageResponse.model_fields)
            == set(ConversationMessage.model_fields)
        )

    def test_role_accepts_user_and_assistant(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        for role in ("user", "assistant"):
            c = ConversationMessageCreate(
                session_id="sess-1",
                role=role,
                content="x",
                user_id="u",
                organization_id="o",
            )
            assert c.role == role

    def test_role_rejects_other_values(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        # 'tool'/'system' are role-collapsed away; only real turn roles persist.
        for bad in ("tool", "system", "", "User"):
            with pytest.raises(ValueError):
                ConversationMessageCreate(
                    session_id="sess-1",
                    role=bad,
                    content="x",
                    user_id="u",
                    organization_id="o",
                )

    def test_session_id_required_rejects_none_and_empty(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        # This store never holds a session-less turn.
        with pytest.raises(ValueError):
            ConversationMessageCreate(
                session_id="",
                role="user",
                content="x",
                user_id="u",
                organization_id="o",
            )
        with pytest.raises(ValueError):
            ConversationMessageCreate(
                session_id=None,
                role="user",
                content="x",
                user_id="u",
                organization_id="o",
            )

    def test_session_id_charset_and_length_enforced(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate
        from mirix.schemas.message import SESSION_ID_MAX_LEN

        with pytest.raises(ValueError):
            ConversationMessageCreate(
                session_id="bad/chars",
                role="user",
                content="x",
                user_id="u",
                organization_id="o",
            )
        with pytest.raises(ValueError):
            ConversationMessageCreate(
                session_id="a" * (SESSION_ID_MAX_LEN + 1),
                role="user",
                content="x",
                user_id="u",
                organization_id="o",
            )
        # Allowed charset passes.
        ok = ConversationMessageCreate(
            session_id="sess_Abc-123",
            role="assistant",
            content="x",
            user_id="u",
            organization_id="o",
        )
        assert ok.session_id == "sess_Abc-123"

    def test_content_length_cap_enforced(self):
        from mirix.schemas.conversation_message import (
            CONVERSATION_MESSAGE_MAX_CONTENT_LEN,
            ConversationMessageCreate,
        )

        with pytest.raises(ValueError):
            ConversationMessageCreate(
                session_id="sess-1",
                role="user",
                content="c" * (CONVERSATION_MESSAGE_MAX_CONTENT_LEN + 1),
                user_id="u",
                organization_id="o",
            )

    def test_content_defaults_to_empty(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        c = ConversationMessageCreate(
            session_id="sess-1",
            role="user",
            user_id="u",
            organization_id="o",
        )
        assert c.content == ""


class TestConversationMessageManagerContract:
    """DB-free: the five-method async contract downstream agents depend on.

    No DB here — we only assert the methods exist, are coroutines, and expose
    the keyword-only signature the ingestion seam / trigger / distiller call.
    """

    def test_manager_exposes_five_async_methods(self):
        import inspect

        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        for name in (
            "record_turns",
            "count_distinct_sessions",
            "list_sealed_undistilled_sessions",
            "list_turns_for_session",
            "mark_sessions_distilled",
        ):
            fn = getattr(ConversationMessageManager, name, None)
            assert fn is not None, f"manager missing {name}"
            # Unwrap the @enforce_types decorator (a plain-def wrapper that
            # returns the coroutine) to inspect the real `async def` underneath.
            assert inspect.iscoroutinefunction(
                inspect.unwrap(fn)
            ), f"{name} must be async"

    def test_record_turns_signature_is_keyword_only(self):
        import inspect

        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        sig = inspect.signature(ConversationMessageManager.record_turns)
        params = sig.parameters
        for expected in (
            "session_id",
            "user_id",
            "organization_id",
            "turns",
            "actor",
        ):
            assert expected in params, f"record_turns missing {expected}"
            assert (
                params[expected].kind is inspect.Parameter.KEYWORD_ONLY
            ), f"{expected} must be keyword-only"

    def test_count_distinct_sessions_has_only_undistilled_flag(self):
        import inspect

        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        sig = inspect.signature(
            ConversationMessageManager.count_distinct_sessions
        )
        assert "only_undistilled" in sig.parameters
        assert sig.parameters["only_undistilled"].default is False


# ============================ Integration: Postgres ========================
#
# Everything below needs a live Postgres (docker-compose). It is marked with the
# integration marker so the default `-m "not integration"` CI subset skips it.

import asyncio  # noqa: E402
import sys  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest_asyncio  # noqa: E402

pytestmark = []  # module-level markers applied per-class below instead

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest_asyncio.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _make_actor():
    """Create a fresh org + client (a distinct owner). Returns the client/actor.

    Each call mints its OWN organization so two actors are guaranteed to live in
    different organizations — exactly what the per-(user, org) scoping tests need.
    """
    from mirix.schemas.client import Client as PydanticClient
    from mirix.schemas.organization import Organization as PydanticOrganization
    from mirix.services.client_manager import ClientManager
    from mirix.services.organization_manager import OrganizationManager

    org_mgr = OrganizationManager()
    client_mgr = ClientManager()

    org_id = f"test-convmsg-org-{uuid.uuid4().hex[:8]}"
    try:
        await org_mgr.get_organization_by_id(org_id)
    except Exception:
        await org_mgr.create_organization(
            PydanticOrganization(id=org_id, name="ConvMsg Test Org")
        )

    client_id = f"test-convmsg-client-{uuid.uuid4().hex[:8]}"
    try:
        return await client_mgr.get_client_by_id(client_id)
    except Exception:
        return await client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="ConvMsg Test Client",
                write_scope="test-convmsg",
                read_scopes=["test-convmsg"],
            )
        )


@pytest_asyncio.fixture(scope="module")
async def cm_actor():
    return await _make_actor()


@pytest_asyncio.fixture(scope="module")
async def cm_actor_org_b():
    """A second actor in a DIFFERENT organization, for cross-org isolation."""
    return await _make_actor()


async def _make_user(org_id: str):
    from mirix.schemas.user import User as PydanticUser
    from mirix.services.user_manager import UserManager

    user_mgr = UserManager()
    user_id = f"test-convmsg-user-{uuid.uuid4().hex[:8]}"
    try:
        return await user_mgr.get_user_by_id(user_id)
    except Exception:
        return await user_mgr.create_user(
            PydanticUser(
                id=user_id,
                name="ConvMsg Test User",
                organization_id=org_id,
                timezone="UTC",
            )
        )


@pytest_asyncio.fixture(scope="module")
async def cm_user(cm_actor):
    return await _make_user(cm_actor.organization_id)


@pytest.fixture
def manager():
    from mirix.services.conversation_message_manager import (
        ConversationMessageManager,
    )

    return ConversationMessageManager()


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestRecordTurns:
    async def test_records_turns_in_order_with_real_roles(
        self, manager, cm_actor, cm_user
    ):
        sid = f"rt-{uuid.uuid4().hex[:8]}"
        out = await manager.record_turns(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            turns=[
                {"role": "user", "content": "Q: 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            actor=cm_actor,
        )
        assert [t.role for t in out] == ["user", "assistant"]
        assert [t.content for t in out] == ["Q: 2+2?", "4"]
        # The STRICTLY increasing created_at is the documented contract guarantee
        # (turns share one transaction, so func.now() would collapse them to one
        # timestamp and destroy order). It is what later sealing/ordering relies
        # on, so we pin it as behavior — not merely the already-checked ascending
        # retrieval below.
        assert out[0].created_at < out[1].created_at

        got = await manager.list_turns_for_session(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert [t.content for t in got] == ["Q: 2+2?", "4"]

    async def test_multiple_calls_same_session_accumulate_in_order(
        self, manager, cm_actor, cm_user
    ):
        sid = f"acc-{uuid.uuid4().hex[:8]}"
        await manager.record_turns(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "user", "content": "first"}],
            actor=cm_actor,
        )
        await manager.record_turns(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "assistant", "content": "second"}],
            actor=cm_actor,
        )
        got = await manager.list_turns_for_session(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        # A later call's turns sort AFTER an earlier call's — one ordered unit.
        assert [t.content for t in got] == ["first", "second"]

    async def test_empty_turns_is_noop(self, manager, cm_actor, cm_user):
        sid = f"empty-{uuid.uuid4().hex[:8]}"
        out = await manager.record_turns(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            turns=[],
            actor=cm_actor,
        )
        assert out == []

    async def test_bad_role_rejected_before_any_write(
        self, manager, cm_actor, cm_user
    ):
        sid = f"badrole-{uuid.uuid4().hex[:8]}"
        with pytest.raises(ValueError):
            await manager.record_turns(
                session_id=sid,
                user_id=cm_user.id,
                organization_id=cm_actor.organization_id,
                turns=[
                    {"role": "user", "content": "ok"},
                    {"role": "system", "content": "bad"},  # invalid role
                ],
                actor=cm_actor,
            )
        # The whole batch is validated up front: NOTHING was written.
        got = await manager.list_turns_for_session(
            session_id=sid,
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert got == []


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestCountDistinctSessions:
    async def test_counts_distinct_sessions_for_owner(
        self, manager, cm_actor, cm_user
    ):
        user = await _make_user(cm_actor.organization_id)
        for sid in ("c1", "c2", "c3"):
            await manager.record_turns(
                session_id=f"{sid}-{uuid.uuid4().hex[:6]}",
                user_id=user.id,
                organization_id=cm_actor.organization_id,
                turns=[{"role": "user", "content": "x"}],
                actor=cm_actor,
            )
        n = await manager.count_distinct_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert n == 3

    async def test_only_undistilled_excludes_distilled_sessions(
        self, manager, cm_actor, cm_user
    ):
        user = await _make_user(cm_actor.organization_id)
        sids = []
        for _ in range(3):
            sid = f"u-{uuid.uuid4().hex[:8]}"
            sids.append(sid)
            await manager.record_turns(
                session_id=sid,
                user_id=user.id,
                organization_id=cm_actor.organization_id,
                turns=[{"role": "user", "content": "x"}],
                actor=cm_actor,
            )
        # Distill one session.
        await manager.mark_sessions_distilled(
            session_ids=[sids[0]],
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        total = await manager.count_distinct_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        undistilled = await manager.count_distinct_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
            only_undistilled=True,
        )
        assert total == 3
        assert undistilled == 2


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestSealedUndistilledSelection:
    async def test_returns_oldest_first_sealed_only(
        self, manager, cm_actor, cm_user
    ):
        user = await _make_user(cm_actor.organization_id)
        # Create 6 sessions, each a clearly-later MIN(created_at) than the last.
        ordered = []
        for i in range(6):
            sid = f"seal-{i}-{uuid.uuid4().hex[:6]}"
            ordered.append(sid)
            await manager.record_turns(
                session_id=sid,
                user_id=user.id,
                organization_id=cm_actor.organization_id,
                turns=[{"role": "user", "content": f"turn {i}"}],
                actor=cm_actor,
            )
        sealed = await manager.list_sealed_undistilled_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
            limit=5,
        )
        # 6 sessions → 5 oldest sealed, oldest-FIRST; the newest is the open head.
        assert sealed == ordered[:5]
        assert ordered[5] not in sealed  # open head never returned

    async def test_single_session_is_never_sealed(
        self, manager, cm_actor, cm_user
    ):
        user = await _make_user(cm_actor.organization_id)
        await manager.record_turns(
            session_id=f"solo-{uuid.uuid4().hex[:8]}",
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "user", "content": "alone"}],
            actor=cm_actor,
        )
        sealed = await manager.list_sealed_undistilled_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
            limit=5,
        )
        assert sealed == []

    async def test_distilled_sessions_drop_out_of_sealed_window(
        self, manager, cm_actor, cm_user
    ):
        user = await _make_user(cm_actor.organization_id)
        ordered = []
        for i in range(4):
            sid = f"drop-{i}-{uuid.uuid4().hex[:6]}"
            ordered.append(sid)
            await manager.record_turns(
                session_id=sid,
                user_id=user.id,
                organization_id=cm_actor.organization_id,
                turns=[{"role": "user", "content": f"t{i}"}],
                actor=cm_actor,
            )
        # Distill the two oldest; they must drop out of the sealed window.
        await manager.mark_sessions_distilled(
            session_ids=ordered[:2],
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        sealed = await manager.list_sealed_undistilled_sessions(
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
            limit=5,
        )
        # 4 sessions, newest is open head (ordered[3]); ordered[0..1] distilled →
        # only ordered[2] remains sealed-and-undistilled.
        assert sealed == [ordered[2]]


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestMarkSessionsDistilled:
    async def test_idempotent_rerun_marks_zero(self, manager, cm_actor, cm_user):
        user = await _make_user(cm_actor.organization_id)
        sid = f"idem-{uuid.uuid4().hex[:8]}"
        await manager.record_turns(
            session_id=sid,
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            turns=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
            actor=cm_actor,
        )
        first = await manager.mark_sessions_distilled(
            session_ids=[sid],
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        second = await manager.mark_sessions_distilled(
            session_ids=[sid],
            user_id=user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert first == 2  # both turns stamped
        assert second == 0  # re-run is a no-op

    async def test_empty_session_ids_is_noop(self, manager, cm_actor, cm_user):
        n = await manager.mark_sessions_distilled(
            session_ids=[],
            user_id=cm_user.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert n == 0


@pytest.mark.integration
@pytest.mark.asyncio(loop_scope="module")
class TestPerUserOrgScoping:
    async def test_sessions_isolated_per_user(self, manager, cm_actor):
        user_a = await _make_user(cm_actor.organization_id)
        user_b = await _make_user(cm_actor.organization_id)
        sid_a = f"iso-a-{uuid.uuid4().hex[:6]}"
        sid_b = f"iso-b-{uuid.uuid4().hex[:6]}"

        await manager.record_turns(
            session_id=sid_a,
            user_id=user_a.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "user", "content": "from A"}],
            actor=cm_actor,
        )
        await manager.record_turns(
            session_id=sid_b,
            user_id=user_b.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "user", "content": "from B"}],
            actor=cm_actor,
        )

        # Each user counts only their own session.
        assert (
            await manager.count_distinct_sessions(
                user_id=user_a.id,
                organization_id=cm_actor.organization_id,
                actor=cm_actor,
            )
            == 1
        )
        assert (
            await manager.count_distinct_sessions(
                user_id=user_b.id,
                organization_id=cm_actor.organization_id,
                actor=cm_actor,
            )
            == 1
        )
        # User A cannot see B's turns.
        a_turns = await manager.list_turns_for_session(
            session_id=sid_b,
            user_id=user_a.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert a_turns == []

    async def test_mark_distilled_cannot_cross_user(self, manager, cm_actor):
        user_a = await _make_user(cm_actor.organization_id)
        user_b = await _make_user(cm_actor.organization_id)
        sid = f"xuser-{uuid.uuid4().hex[:6]}"

        await manager.record_turns(
            session_id=sid,
            user_id=user_a.id,
            organization_id=cm_actor.organization_id,
            turns=[{"role": "user", "content": "A owns this"}],
            actor=cm_actor,
        )
        # User B attempting to mark A's session distills nothing.
        crossed = await manager.mark_sessions_distilled(
            session_ids=[sid],
            user_id=user_b.id,
            organization_id=cm_actor.organization_id,
            actor=cm_actor,
        )
        assert crossed == 0
        # A's session is still undistilled.
        assert (
            await manager.count_distinct_sessions(
                user_id=user_a.id,
                organization_id=cm_actor.organization_id,
                actor=cm_actor,
                only_undistilled=True,
            )
            == 1
        )

    async def test_sessions_isolated_per_organization(
        self, manager, cm_actor, cm_actor_org_b
    ):
        """A session recorded in org A must be invisible to org B's queries.

        The decisive move is the CROSS query: we ask for org A's user under org
        B's id. A manager that filtered only by user_id (ignoring
        organization_id) would still return org A's session here; correct
        per-(user, org) scoping returns nothing. (A real `users` row is tied to
        exactly one org by FK, so two orgs cannot share a user PK — the cross
        query, not a shared id, is what proves the org dimension is enforced.)
        """
        org_a = cm_actor.organization_id
        org_b = cm_actor_org_b.organization_id
        assert org_a != org_b  # the fixtures mint distinct orgs

        user_a = await _make_user(org_a)
        sid_a = f"org-a-{uuid.uuid4().hex[:6]}"
        await manager.record_turns(
            session_id=sid_a,
            user_id=user_a.id,
            organization_id=org_a,
            turns=[{"role": "user", "content": "belongs to org A"}],
            actor=cm_actor,
        )

        # Correct, same-org query sees the session…
        assert (
            await manager.count_distinct_sessions(
                user_id=user_a.id,
                organization_id=org_a,
                actor=cm_actor,
            )
            == 1
        )
        # …but the SAME user_id under org B's organization_id sees NOTHING.
        assert (
            await manager.count_distinct_sessions(
                user_id=user_a.id,
                organization_id=org_b,
                actor=cm_actor_org_b,
            )
            == 0
        )
        # And org B cannot read org A's turns even with A's user_id.
        assert (
            await manager.list_turns_for_session(
                session_id=sid_a,
                user_id=user_a.id,
                organization_id=org_b,
                actor=cm_actor_org_b,
            )
            == []
        )

    async def test_mark_distilled_cannot_cross_organization(
        self, manager, cm_actor, cm_actor_org_b
    ):
        """mark_sessions_distilled is org-scoped: passing org B's
        organization_id (with org A's user_id and session) distills nothing."""
        org_a = cm_actor.organization_id
        org_b = cm_actor_org_b.organization_id
        user_a = await _make_user(org_a)
        sid = f"xorg-{uuid.uuid4().hex[:6]}"
        await manager.record_turns(
            session_id=sid,
            user_id=user_a.id,
            organization_id=org_a,
            turns=[{"role": "user", "content": "A owns this"}],
            actor=cm_actor,
        )
        # Marking under org B's organization_id must be a no-op.
        crossed = await manager.mark_sessions_distilled(
            session_ids=[sid],
            user_id=user_a.id,
            organization_id=org_b,
            actor=cm_actor_org_b,
        )
        assert crossed == 0
        # A's session remains undistilled in its own org.
        assert (
            await manager.count_distinct_sessions(
                user_id=user_a.id,
                organization_id=org_a,
                actor=cm_actor,
                only_undistilled=True,
            )
            == 1
        )
