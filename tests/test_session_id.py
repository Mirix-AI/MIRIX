"""
Unit tests for the top-level `session_id` field on messages.

Scope:
- Pydantic schema: MessageCreate / MessageUpdate / Message accept optional session_id.
- Validation: length/charset rules on session_id.
- Helper: prepare_input_message_create propagates session_id to the Message.
- REST schemas: SendMessageRequest / AddMemoryRequest accept session_id.
- Queue serialization: put_messages path sets session_id on proto, worker restores it.
- ORM column exists and is indexed.

These are unit tests only; no DB, no running server.
"""

from __future__ import annotations

import pytest

from mirix.helpers.message_helpers import prepare_input_message_create
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import Message, MessageCreate, MessageUpdate


# ----------------------------- Schema ------------------------------------


class TestMessageCreateSessionId:
    def test_accepts_session_id(self):
        m = MessageCreate(role=MessageRole.user, content="hi", session_id="sess-abc")
        assert m.session_id == "sess-abc"

    def test_session_id_optional(self):
        m = MessageCreate(role=MessageRole.user, content="hi")
        assert m.session_id is None

    def test_rejects_empty_string_session_id(self):
        # Empty string is ambiguous; require None or a non-empty string.
        with pytest.raises(ValueError):
            MessageCreate(role=MessageRole.user, content="hi", session_id="")

    def test_rejects_too_long_session_id(self):
        with pytest.raises(ValueError):
            MessageCreate(role=MessageRole.user, content="hi", session_id="a" * 65)

    def test_rejects_invalid_chars(self):
        with pytest.raises(ValueError):
            MessageCreate(role=MessageRole.user, content="hi", session_id="sess/abc")

    def test_accepts_allowed_chars(self):
        m = MessageCreate(
            role=MessageRole.user,
            content="hi",
            session_id="sess_Abc-123",
        )
        assert m.session_id == "sess_Abc-123"


class TestMessageSchemaSessionId:
    def test_accepts_session_id(self):
        m = Message(
            agent_id="agent-1",
            role=MessageRole.user,
            session_id="sess-xyz",
        )
        assert m.session_id == "sess-xyz"

    def test_defaults_to_none(self):
        m = Message(agent_id="agent-1", role=MessageRole.user)
        assert m.session_id is None


class TestMessageUpdateSessionId:
    def test_accepts_session_id(self):
        u = MessageUpdate(session_id="sess-upd")
        assert u.session_id == "sess-upd"

    def test_defaults_to_none(self):
        u = MessageUpdate()
        assert u.session_id is None


# ----------------------------- Helper ------------------------------------


class TestPrepareInputMessageCreate:
    def test_propagates_session_id(self):
        create = MessageCreate(role=MessageRole.user, content="hi", session_id="sess-1")
        msg = prepare_input_message_create(create, agent_id="agent-1")
        assert msg.session_id == "sess-1"

    def test_missing_session_id_is_none(self):
        create = MessageCreate(role=MessageRole.user, content="hi")
        msg = prepare_input_message_create(create, agent_id="agent-1")
        assert msg.session_id is None


class TestDictToMessageSessionId:
    """Codex review C1: internal agent-synthesized messages must inherit session_id."""

    def test_passes_session_id_through_kwarg(self):
        msg = Message.dict_to_message(
            agent_id="agent-1",
            openai_message_dict={"role": "user", "content": "hi"},
            session_id="sess-internal",
        )
        assert msg.session_id == "sess-internal"

    def test_defaults_to_none(self):
        msg = Message.dict_to_message(
            agent_id="agent-1",
            openai_message_dict={"role": "user", "content": "hi"},
        )
        assert msg.session_id is None

    def test_tool_role_message_carries_session_id(self):
        msg = Message.dict_to_message(
            agent_id="agent-1",
            openai_message_dict={
                "role": "tool",
                "content": "ok",
                "tool_call_id": "tc-1",
                "name": "do_stuff",
            },
            session_id="sess-internal",
        )
        assert msg.session_id == "sess-internal"


# ----------------------------- REST request schemas ----------------------


class TestRestRequestSchemasSessionId:
    def test_send_message_request_accepts_session_id(self):
        from mirix.server.rest_api import SendMessageRequest

        req = SendMessageRequest(message="hi", role="user", session_id="sess-req")
        assert req.session_id == "sess-req"

    def test_send_message_request_session_id_optional(self):
        from mirix.server.rest_api import SendMessageRequest

        req = SendMessageRequest(message="hi", role="user")
        assert req.session_id is None

    def test_add_memory_request_accepts_session_id(self):
        from mirix.server.rest_api import AddMemoryRequest

        req = AddMemoryRequest(
            meta_agent_id="meta-1",
            messages=[{"role": "user", "content": "hi"}],
            session_id="sess-mem",
        )
        assert req.session_id == "sess-mem"

    def test_add_memory_request_session_id_optional(self):
        from mirix.server.rest_api import AddMemoryRequest

        req = AddMemoryRequest(
            meta_agent_id="meta-1",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.session_id is None

    # Validation at the REST boundary (Codex review: C2 + I5).
    # Invalid input should raise at model construction, not be queued.
    def test_send_message_request_rejects_invalid_session_id(self):
        from mirix.server.rest_api import SendMessageRequest

        with pytest.raises(ValueError):
            SendMessageRequest(message="hi", role="user", session_id="bad/chars")
        with pytest.raises(ValueError):
            SendMessageRequest(message="hi", role="user", session_id="")
        with pytest.raises(ValueError):
            SendMessageRequest(message="hi", role="user", session_id="a" * 65)

    def test_add_memory_request_rejects_invalid_session_id(self):
        from mirix.server.rest_api import AddMemoryRequest

        with pytest.raises(ValueError):
            AddMemoryRequest(
                meta_agent_id="meta-1",
                messages=[{"role": "user", "content": "hi"}],
                session_id="bad chars",
            )
        with pytest.raises(ValueError):
            AddMemoryRequest(
                meta_agent_id="meta-1",
                messages=[{"role": "user", "content": "hi"}],
                session_id="",
            )


# ----------------------------- Queue proto -------------------------------


class TestAddMemoryMismatchRejection:
    """Codex review I1: top-level session_id and filter_tags.session_id must agree."""

    def test_matching_values_are_allowed(self):
        from mirix.server.rest_api import AddMemoryRequest

        # Same value in both places is fine.
        req = AddMemoryRequest(
            meta_agent_id="meta-1",
            messages=[{"role": "user", "content": "hi"}],
            session_id="sess-1",
            filter_tags={"session_id": "sess-1"},
        )
        assert req.session_id == "sess-1"
        assert req.filter_tags["session_id"] == "sess-1"

    def test_mismatch_is_rejected_at_request_model(self):
        from mirix.server.rest_api import AddMemoryRequest

        with pytest.raises(ValueError):
            AddMemoryRequest(
                meta_agent_id="meta-1",
                messages=[{"role": "user", "content": "hi"}],
                session_id="sess-1",
                filter_tags={"session_id": "sess-2"},
            )


class TestQueueProtoSessionId:
    """Ensure the proto round-trips session_id through MessageCreate."""

    def test_proto_message_create_has_session_id_field(self):
        from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate

        field_names = {f.name for f in ProtoMessageCreate.DESCRIPTOR.fields}
        assert "session_id" in field_names

    def test_put_messages_serializes_session_id(self, monkeypatch):
        """put_messages should copy MessageCreate.session_id onto proto."""
        import asyncio

        from mirix.queue import queue_util
        from mirix.schemas.client import Client

        saved = {}

        class FakeQueue:
            async def save(self, msg):
                saved["msg"] = msg

        monkeypatch.setattr(queue_util, "queue", FakeQueue())

        async def run():
            await queue_util.put_messages(
                actor=Client(
                    id="client-1",
                    organization_id="org-1",
                    name="c",
                    write_scope="w",
                    read_scopes=["w"],
                ),
                agent_id="agent-1",
                input_messages=[
                    MessageCreate(
                        role=MessageRole.user,
                        content="hi",
                        session_id="sess-q",
                    )
                ],
            )

        asyncio.run(run())

        msg = saved["msg"]
        assert len(msg.input_messages) == 1
        proto = msg.input_messages[0]
        assert proto.HasField("session_id")
        assert proto.session_id == "sess-q"

    def test_worker_restores_session_id_from_proto(self):
        from mirix.queue.message_pb2 import MessageCreate as ProtoMessageCreate
        from mirix.queue.worker import QueueWorker

        proto = ProtoMessageCreate()
        proto.role = ProtoMessageCreate.ROLE_USER
        proto.text_content = "hi"
        proto.session_id = "sess-w"

        worker = QueueWorker.__new__(QueueWorker)
        out = worker._convert_proto_message_to_pydantic(proto)
        assert out.session_id == "sess-w"


# ----------------------------- ORM --------------------------------------


class TestAgentStepPropagation:
    """Codex v2 review: heartbeat / warning / meta / summary messages inside Agent.step
    must inherit session_id from the triggering input so session-scoped list queries
    return the whole conversation, not just the initial user turn."""

    def test_step_heartbeat_sites_include_session_id(self):
        """Sanity check: each heartbeat dict_to_message site in Agent.step passes a session_id kwarg.

        We scan mirix/agent/agent.py for every `Message.dict_to_message(` opener and
        require a `session_id=` kwarg before the matching outer `)`. A regex with
        lazy-matched parens trips on nested calls (e.g. `get_contine_chaining(...)`
        spread across lines after ruff reformat), so we walk parens explicitly.
        """
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        opener = "Message.dict_to_message("
        i = 0
        blocks = []
        while True:
            j = src.find(opener, i)
            if j == -1:
                break
            depth = 0
            k = j + len(opener) - 1  # position of the opening `(`
            for k in range(j + len(opener) - 1, len(src)):
                ch = src[k]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            blocks.append(src[j : k + 1])
            i = k + 1

        assert blocks, "expected at least one Message.dict_to_message call in agent.py"
        missing = [b for b in blocks if "session_id=" not in b]
        assert not missing, (
            "These dict_to_message sites do not pass session_id:\n"
            + "\n---\n".join(missing)
        )

    def test_persisted_message_create_sites_include_session_id(self):
        """Codex v3 follow-up: MessageCreate(...) sites that are persisted (turned
        into a Message via prepare_input_message_create) must also inherit session_id.

        agent.py has two relevant MessageCreate() call sites: the meta-memory
        bootstrap (persisted via prepare_input_message_create) and topic-extraction
        scratch prompts (sent straight to the LLM, not persisted). We lint the
        meta-bootstrap path specifically.
        """
        import re
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        # The meta-bootstrap MessageCreate is inside the `if ... meta_memory_agent ...` branch.
        meta_block_match = re.search(
            r"meta_message\s*=\s*prepare_input_message_create\(\n"
            r"\s*MessageCreate\([\s\S]*?\)\s*,\n",
            src,
        )
        assert meta_block_match, "meta-bootstrap MessageCreate block not found"
        assert "session_id=step_session_id" in meta_block_match.group(0), (
            "meta-memory bootstrap MessageCreate must carry session_id=step_session_id, "
            f"got:\n{meta_block_match.group(0)}"
        )

    def test_step_seeds_and_restores_stash(self):
        """Codex v4 review: Agent.step() also stashes _current_step_session_id on
        self for the duration of the step. Without a try/finally, an exception or a
        successful return would leave the prior caller's value clobbered, so a
        later summarize_messages_inplace() outside any step() would read a stale
        session. Lock in the save/restore-with-finally pattern via source inspection
        (a real step() requires LLM + DB so behavioral test is impractical here)."""
        import inspect

        from mirix.agent import agent as agent_mod

        src = inspect.getsource(agent_mod.Agent.step)
        assert (
            'prev_session_id = getattr(self, "_current_step_session_id", None)' in src
        ), (
            "step() must capture the prior _current_step_session_id before overwriting it"
        )
        assert "self._current_step_session_id = step_session_id" in src
        assert "try:" in src and "finally:" in src, "step() must use try/finally"
        assert "self._current_step_session_id = prev_session_id" in src, (
            "step() must restore prev_session_id in its finally block"
        )


class TestOrmSessionId:
    def test_orm_has_session_id_column(self):
        from mirix.orm.message import Message as MessageORM

        cols = {c.name for c in MessageORM.__table__.columns}
        assert "session_id" in cols

    def test_orm_session_id_is_indexed(self):
        from mirix.orm.message import Message as MessageORM

        # At least one composite or single index covers session_id.
        covered = any(
            any(c.name == "session_id" for c in idx.columns)
            for idx in MessageORM.__table__.indexes
        )
        assert covered, "session_id should be indexed"


class TestStepUserMessageSessionContext:
    """Codex review v3: step_user_message() bypasses step() and must still seed
    _current_step_session_id for any pre-persist summary, then restore the prior value.
    We inspect the source rather than run the coroutine (full step needs LLM + DB)."""

    def test_step_user_message_seeds_and_restores_stash(self):
        import inspect

        from mirix.agent import agent as agent_mod

        src = inspect.getsource(agent_mod.Agent.step_user_message)
        # Seeds current step session context for summarizer.
        assert "self._current_step_session_id = session_id" in src
        # Uses try/finally to restore the prior value — no leaks across calls.
        assert "prev_session_id" in src
        assert "finally" in src
        assert "self._current_step_session_id = prev_session_id" in src

    def test_summarize_messages_inplace_accepts_explicit_session_id(self):
        import inspect

        from mirix.agent import agent as agent_mod

        sig = inspect.signature(agent_mod.Agent.summarize_messages_inplace)
        assert "session_id" in sig.parameters
        # Default is optional None so existing callers don't break.
        assert sig.parameters["session_id"].default is None


class TestCheckConstraintDialectGating:
    """The CHECK uses PG's `~` regex operator, which SQLite doesn't understand.
    Verify the constraint is only emitted for Postgres so SQLite create_all works."""

    def test_check_compiles_for_postgresql(self):
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        from mirix.orm.message import Message as MessageORM

        ddl = str(
            CreateTable(MessageORM.__table__).compile(dialect=postgresql.dialect())
        )
        assert "ck_messages_session_id_format" in ddl
        assert "~" in ddl  # Postgres regex operator

    def test_check_is_suppressed_for_sqlite(self):
        from sqlalchemy.dialects import sqlite
        from sqlalchemy.schema import CreateTable

        from mirix.orm.message import Message as MessageORM

        ddl = str(CreateTable(MessageORM.__table__).compile(dialect=sqlite.dialect()))
        # Constraint must not appear on SQLite — its `~` regex would break CREATE TABLE.
        assert "ck_messages_session_id_format" not in ddl
        assert "session_id ~" not in ddl


class TestSessionIdConstantsInSync:
    """Codex v2 nit: one source of truth for pattern/length across Python, ORM CheckConstraint, and SQL."""

    def test_orm_column_length_matches_constant(self):
        from mirix.orm.message import Message as MessageORM
        from mirix.schemas.message import SESSION_ID_MAX_LEN

        col = MessageORM.__table__.c.session_id
        assert col.type.length == SESSION_ID_MAX_LEN

    def test_orm_check_constraint_uses_shared_pattern(self):
        from mirix.orm.message import Message as MessageORM
        from mirix.schemas.message import SESSION_ID_SQL_PATTERN

        ck_texts = [
            str(c.sqltext)
            for c in MessageORM.__table__.constraints
            if getattr(c, "name", None) == "ck_messages_session_id_format"
        ]
        assert ck_texts, "ck_messages_session_id_format not found"
        assert SESSION_ID_SQL_PATTERN in ck_texts[0]

    def test_migration_sql_uses_shared_pattern_and_length(self):
        from pathlib import Path

        from mirix.schemas.message import (
            SESSION_ID_MAX_LEN,
            SESSION_ID_SQL_PATTERN,
        )

        phase1 = Path("scripts/migrate_add_message_session_id.sql").read_text()
        phase2 = Path("scripts/migrate_add_message_session_id_phase2.sql").read_text()

        # Column length must match.
        assert f"VARCHAR({SESSION_ID_MAX_LEN})" in phase1
        # CHECK regex must match the shared SQL pattern exactly.
        assert SESSION_ID_SQL_PATTERN in phase1
        # Phase 2 must use CONCURRENTLY (online, write-safe).
        assert "CREATE INDEX CONCURRENTLY" in phase2
