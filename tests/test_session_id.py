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
            MessageCreate(
                role=MessageRole.user, content="hi", session_id="a" * 65
            )

    def test_rejects_invalid_chars(self):
        with pytest.raises(ValueError):
            MessageCreate(
                role=MessageRole.user, content="hi", session_id="sess/abc"
            )

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
        create = MessageCreate(
            role=MessageRole.user, content="hi", session_id="sess-1"
        )
        msg = prepare_input_message_create(create, agent_id="agent-1")
        assert msg.session_id == "sess-1"

    def test_missing_session_id_is_none(self):
        create = MessageCreate(role=MessageRole.user, content="hi")
        msg = prepare_input_message_create(create, agent_id="agent-1")
        assert msg.session_id is None


class TestDictToMessageSessionId:
    """dict_to_message must FAITHFULLY pass through whatever session_id it is
    given (the chat_agent path supplies one; non-chat callers pass None). The
    decoupling is enforced at the CALLER (see TestAgentStepPropagation, which
    pins that non-chat callers pass None) — here we only pin the mechanism: the
    kwarg round-trips and defaults to None when omitted."""

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

        req = SendMessageRequest(
            message="hi", role="user", session_id="sess-req"
        )
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
            SendMessageRequest(
                message="hi", role="user", session_id="bad/chars"
            )
        with pytest.raises(ValueError):
            SendMessageRequest(message="hi", role="user", session_id="")
        with pytest.raises(ValueError):
            SendMessageRequest(
                message="hi", role="user", session_id="a" * 65
            )

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
    """Decoupling lint (PRD: procedural-memory-distillation-decoupling).

    session_id identifies an EXTERNAL conversation, never the memory-production
    machinery. The decoupling INVERTS the old rule: a non-chat agent
    (meta_memory_agent + the memory sub-agents) must NOT stamp session_id on the
    messages it synthesizes inside Agent.step — those are transient bookkeeping
    that gets discarded after each extraction. Only the chat_agent path may keep
    inheriting it.

    The synthesized-message sites bind session_id to two derived locals —
    `input_session_id` (in the response-processing path) and `step_session_id`
    (in the step loop / meta-bootstrap / summary path). The decoupling lives in
    how those two locals are DERIVED: each is gated `... if <chat_agent> else
    None`. We lint that derivation rather than every call site, so the test is
    robust to whitespace and to call-site churn while still pinning the
    behavior: for a non-chat agent the value is None, so nothing it synthesizes
    carries a session_id.
    """

    # Tolerant whitespace between tokens so the lint never breaks on reformatting.
    _WS = r"\s*"

    def _gated_to_none_for_non_chat(self, src: str, local: str, attr_obj: str) -> bool:
        """True iff `<local> = getattr(<attr_obj>, "session_id", None) if
        <is chat_agent> else None` appears in `src`, whitespace-insensitively.

        This is the load-bearing decoupling: the value is the input's session_id
        ONLY for the chat_agent, and explicitly `None` for every other agent.
        """
        import re

        ws = self._WS
        # The assignment is wrapped in parens spanning several lines:
        #   <local> = (
        #       getattr(<attr_obj>, "session_id", None)
        #       if self.agent_state.is_type(AgentType.chat_agent)
        #       else None
        #   )
        # `\s*` (in self._WS) spans newlines, so allow an optional opening paren
        # after `=` and tolerate arbitrary whitespace between every token.
        pattern = (
            rf"{re.escape(local)}{ws}={ws}\(?{ws}"
            rf"getattr\({ws}{re.escape(attr_obj)}{ws},{ws}"
            rf"[\"']session_id[\"']{ws},{ws}None{ws}\){ws}"
            rf"if{ws}self\.agent_state\.is_type\({ws}AgentType\.chat_agent{ws}\){ws}"
            rf"else{ws}None"
        )
        return re.search(pattern, src) is not None

    def test_input_session_id_is_none_for_non_chat_agents(self):
        """The response-path local `input_session_id` must resolve to None for
        every non-chat agent, so the assistant/tool messages it stamps carry no
        session_id."""
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        assert self._gated_to_none_for_non_chat(
            src, "input_session_id", "input_message"
        ), (
            "input_session_id must be gated `... if chat_agent else None` so "
            "non-chat synthesized messages carry NO session_id"
        )

    def test_step_session_id_is_none_for_non_chat_agents(self):
        """The step-loop local `step_session_id` (heartbeats, meta-bootstrap,
        summaries) must resolve to None for every non-chat agent."""
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        assert self._gated_to_none_for_non_chat(
            src, "step_session_id", "first_input_message"
        ), (
            "step_session_id must be gated `... if chat_agent else None` so "
            "non-chat synthesized messages carry NO session_id"
        )

    def test_synthesized_sites_never_bind_raw_input_session_id(self):
        """Defense-in-depth: no synthesized message may read session_id straight
        off the RAW triggering input, which would bypass the chat-agent gate and
        leak the conversation id onto a non-chat agent's bookkeeping. The raw
        inputs are `input_message` (response path) and `first_input_message`
        (step path); their `.session_id` must only ever be read INSIDE the gated
        derivation, never as a `session_id=` kwarg on a synthesized message.
        """
        import re
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        ws = self._WS
        for raw in ("input_message", "first_input_message"):
            # A `session_id=` kwarg that reads the raw input's session_id — in
            # EITHER form, attribute access `<raw>.session_id` or
            # `getattr(<raw>, "session_id", ...)` — would bypass the chat-agent
            # gate and leak the conversation id. The legitimate gated derivation
            # assigns getattr(...) to the LOCAL (`input_session_id = ...`), never
            # to a `session_id=` kwarg, so it is not matched here.
            attr_leak = re.search(
                r"session_id" + ws + r"=" + ws + re.escape(raw) + r"\.session_id",
                src,
            )
            getattr_leak = re.search(
                r"session_id" + ws + r"=" + ws
                + r"getattr\(" + ws + re.escape(raw) + ws + r",",
                src,
            )
            assert attr_leak is None and getattr_leak is None, (
                f"synthesized session_id must never bind {raw}'s session_id "
                "directly (attribute or getattr) — it must go through the "
                "chat-agent-gated local"
            )

    def test_meta_bootstrap_does_not_inherit_session_id_for_non_chat(self):
        """The meta-memory bootstrap MessageCreate is persisted (via
        prepare_input_message_create) on the meta_memory_agent — a NON-chat
        agent. It must bind the gated `step_session_id` (which is None for the
        meta agent), NOT a raw input attribute, so the bootstrap carries no
        session_id.
        """
        import re
        from pathlib import Path

        src = Path("mirix/agent/agent.py").read_text()
        ws = self._WS
        meta_block_match = re.search(
            r"meta_message" + ws + r"=" + ws + r"prepare_input_message_create\("
            r"[\s\S]*?MessageCreate\([\s\S]*?\)" + ws + r",",
            src,
        )
        assert meta_block_match, "meta-bootstrap MessageCreate block not found"
        block = meta_block_match.group(0)
        # It binds the gated local (whitespace/newline tolerant)…
        assert re.search(
            r"session_id" + ws + r"=" + ws + r"step_session_id", block
        ), (
            "meta-memory bootstrap must bind session_id=step_session_id "
            f"(gated to None for the meta agent), got:\n{block}"
        )
        # …and never a raw input attribute (attribute or getattr form).
        for raw in ("input_message", "first_input_message"):
            assert not re.search(re.escape(raw) + r"\.session_id", block)
            assert not re.search(r"getattr\(" + ws + re.escape(raw) + ws + r",", block)


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
    """step_user_message() bypasses step() but is the SAME decoupling site: it
    synthesizes a user Message and seeds _current_step_session_id for any
    pre-persist summary, then restores the prior value. Per the
    procedural-memory-distillation-decoupling PRD this site is also GATED — a
    non-chat agent must NOT inherit the conversation's session_id here either, so
    the seeded value is gated `... if chat_agent else None` (not the raw
    session_id argument). We inspect the source rather than run the coroutine
    (full step needs LLM + DB)."""

    def test_step_user_message_gates_session_id_for_non_chat(self):
        import inspect
        import re

        from mirix.agent import agent as agent_mod

        src = inspect.getsource(agent_mod.Agent.step_user_message)
        ws = r"\s*"
        # The gate must bind to `effective_session_id` SPECIFICALLY — i.e.
        #   effective_session_id = (session_id if chat_agent else None)
        # Binding the regex to the actual local that flows into the synthesized
        # message defeats a decoy like `unused = session_id if chat_agent else
        # None; effective_session_id = session_id`, which an unbound `... if
        # chat_agent else None` match would wrongly accept.
        gate = re.search(
            r"effective_session_id" + ws + r"=" + ws + r"\(?" + ws
            + r"session_id" + ws
            + r"if" + ws + r"self\.agent_state\.is_type\(" + ws
            + r"AgentType\.chat_agent" + ws + r"\)" + ws + r"else" + ws + r"None",
            src,
        )
        assert gate is not None, (
            "step_user_message must gate `effective_session_id = (session_id if "
            "chat_agent else None)` so a non-chat agent's synthesized message "
            "carries NO session_id"
        )
        # The synthesized message binds the GATED value, never the raw argument.
        assert re.search(r"session_id" + ws + r"=" + ws + r"effective_session_id", src), (
            "step_user_message's dict_to_message must bind the gated "
            "effective_session_id, not the raw session_id argument"
        )
        # And it must NOT bind the raw `session_id` argument to the message.
        assert not re.search(
            r"dict_to_message\([\s\S]*?session_id" + ws + r"=" + ws + r"session_id\b",
            src,
        ), "step_user_message must not bind the RAW session_id argument"

    def test_step_user_message_seeds_gated_value_and_restores_stash(self):
        import inspect
        import re

        from mirix.agent import agent as agent_mod

        src = inspect.getsource(agent_mod.Agent.step_user_message)
        ws = r"\s*"
        # Seeds the GATED session context for the summarizer (not the raw arg)…
        assert re.search(
            r"self\._current_step_session_id" + ws + r"=" + ws + r"effective_session_id",
            src,
        ), "must seed _current_step_session_id with the gated effective_session_id"
        # …and uses try/finally to restore the prior value — no leaks across calls.
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

        ddl = str(
            CreateTable(MessageORM.__table__).compile(dialect=sqlite.dialect())
        )
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
            str(c.sqltext) for c in MessageORM.__table__.constraints
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
