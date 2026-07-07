"""Tests for conversation-turn extraction into the Conversation Message Store.

The distiller's prompt has always treated tool errors/retries as strong
learning signals (`signal_type: tool_error`), but the ingestion seam used to
drop every non-user/assistant message — blinding the distiller to exactly that
signal. These tests pin the repaired pipeline: tool results and assistant
tool_calls survive extraction, content parts are flattened to clean text, and
foreign roles are still dropped.
"""

from types import SimpleNamespace

import pytest

from mirix.server.rest_api import (
    _extract_conversation_turns,
    _ingest_session_turns,
    _serialize_tool_calls,
)


class TestExtractConversationTurns:
    def test_user_and_assistant_kept(self):
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "deploy the app"},
                {"role": "assistant", "content": "done"},
            ]
        )
        assert turns == [
            {"role": "user", "content": "deploy the app"},
            {"role": "assistant", "content": "done"},
        ]

    def test_tool_role_is_kept_with_name_prefix(self):
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "run the tests"},
                {"role": "tool", "name": "run_tests", "content": "2 failed, 10 passed"},
            ]
        )
        assert turns[1] == {
            "role": "tool",
            "content": "[run_tests] 2 failed, 10 passed",
        }

    def test_function_role_maps_to_tool(self):
        """OpenAI's legacy 'function' role is the same concept as 'tool'."""
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "q"},
                {"role": "function", "name": "search", "content": "no results"},
            ]
        )
        assert turns[1]["role"] == "tool"
        assert "no results" in turns[1]["content"]

    def test_assistant_tool_calls_are_serialized(self):
        """A pure tool-call assistant turn (empty content) must still yield a
        turn — the call itself is the work-process signal."""
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "deploy"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "kubectl_apply",
                                "arguments": '{"file": "deploy.yaml"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "kubectl_apply",
                    "content": "error: forbidden",
                },
            ]
        )
        assert len(turns) == 3
        assert turns[1]["role"] == "assistant"
        assert 'kubectl_apply({"file": "deploy.yaml"})' in turns[1]["content"]
        assert turns[2] == {
            "role": "tool",
            "content": "[kubectl_apply] error: forbidden",
        }

    def test_assistant_content_and_tool_calls_both_kept(self):
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [{"name": "search", "arguments": "query"}],
                },
            ]
        )
        assert turns[1]["content"].startswith("let me check")
        assert "[tool_call] search(query)" in turns[1]["content"]

    def test_content_parts_flatten_to_text(self):
        """The SDK's [{"type":"text","text":...}] parts must store clean text,
        not dict reprs."""
        turns = _extract_conversation_turns(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert turns[0]["content"] == "hello\nworld"

    def test_foreign_roles_are_dropped(self):
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "q"},
                {"role": "system", "content": "scaffolding — not a learnable turn"},
                {"role": "assistant", "content": "a"},
            ]
        )
        assert [t["role"] for t in turns] == ["user", "assistant"]

    def test_non_role_bearing_payload_yields_nothing(self):
        assert (
            _extract_conversation_turns([{"type": "text", "text": "screenshot"}]) == []
        )
        assert _extract_conversation_turns([]) == []

    def test_oversized_turn_is_truncated_to_store_cap(self):
        """One huge tool result must fail softly (truncate), not blow the whole
        batch's Pydantic validation in record_turns."""
        from mirix.schemas.conversation_message import (
            CONVERSATION_MESSAGE_MAX_CONTENT_LEN,
            ConversationMessageCreate,
        )

        huge = "x" * (CONVERSATION_MESSAGE_MAX_CONTENT_LEN + 1000)
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "name": "dump", "content": huge},
            ]
        )
        assert len(turns[1]["content"]) <= CONVERSATION_MESSAGE_MAX_CONTENT_LEN
        assert turns[1]["content"].endswith("…[truncated]")
        # And the truncated turn passes the store schema it will be validated by.
        ConversationMessageCreate(
            session_id="sess-1",
            user_id="u",
            organization_id="o",
            role=turns[1]["role"],
            content=turns[1]["content"],
        )

    def test_malformed_tool_calls_do_not_raise(self):
        """Extraction must never abort the add path on odd shapes."""
        turns = _extract_conversation_turns(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": None, "tool_calls": "not-a-list"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [42, {"function": "not-a-dict"}],
                },
                {"role": "tool", "content": {"weird": "dict"}},
            ]
        )
        assert [t["role"] for t in turns] == ["user", "assistant", "assistant", "tool"]


class TestSerializeToolCalls:
    def test_openai_shape(self):
        rendered = _serialize_tool_calls(
            [{"function": {"name": "grep", "arguments": '{"q": "x"}'}}]
        )
        assert rendered == '[tool_call] grep({"q": "x"})'

    def test_flat_shape_and_dict_arguments(self):
        rendered = _serialize_tool_calls(
            [{"name": "read", "arguments": {"path": "a.py"}}]
        )
        assert rendered == '[tool_call] read({"path": "a.py"})'

    def test_unrecognized_falls_back_to_str(self):
        assert "[tool_call]" in _serialize_tool_calls(["opaque"])


@pytest.mark.asyncio
class TestIngestSessionTurnsIsolation:
    """The store write is ADDITIVE: a store or extraction failure must degrade
    to a missed session for skill distillation, never abort the primary
    /memory/add(_sync) ingestion."""

    @staticmethod
    def _request(session_id, messages):
        return SimpleNamespace(session_id=session_id, messages=messages)

    @staticmethod
    def _client():
        return SimpleNamespace(organization_id="org-1")

    async def test_record_turns_failure_does_not_propagate(self, monkeypatch):
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        async def boom(self, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(ConversationMessageManager, "record_turns", boom)

        request = self._request(
            "sess-iso-1",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        )
        # Must not raise — the primary memory add would proceed.
        await _ingest_session_turns(request, [], self._client(), "user-1")

    async def test_extraction_failure_does_not_propagate(self, monkeypatch):
        """Extraction runs INSIDE the guard: a message whose content cannot be
        rendered must not reach the store, and must not raise."""
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        calls = []

        async def recorder(self, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(ConversationMessageManager, "record_turns", recorder)

        class Unrenderable:
            def __str__(self):
                raise ValueError("unrenderable content part")

        request = self._request(
            "sess-iso-2", [{"role": "user", "content": [Unrenderable()]}]
        )
        await _ingest_session_turns(request, [], self._client(), "user-1")
        assert calls == []

    async def test_no_session_id_skips_store_entirely(self, monkeypatch):
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        calls = []

        async def recorder(self, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(ConversationMessageManager, "record_turns", recorder)

        input_msg = SimpleNamespace(session_id=None)
        request = self._request(None, [{"role": "user", "content": "q"}])
        await _ingest_session_turns(request, [input_msg], self._client(), "user-1")
        assert calls == []
        assert input_msg.session_id is None  # no stamping without a session

    async def test_session_id_stamped_and_turns_recorded(self, monkeypatch):
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        calls = []

        async def recorder(self, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(ConversationMessageManager, "record_turns", recorder)

        unstamped = SimpleNamespace(session_id=None)
        prestamped = SimpleNamespace(session_id="sess-own")
        request = self._request(
            "sess-batch",
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "name": "grep", "content": "no match"},
                {"role": "assistant", "content": "a"},
            ],
        )
        await _ingest_session_turns(
            request, [unstamped, prestamped], self._client(), "user-1"
        )
        assert unstamped.session_id == "sess-batch"
        assert prestamped.session_id == "sess-own"  # own id wins over the batch id
        assert len(calls) == 1
        assert calls[0]["session_id"] == "sess-batch"
        assert calls[0]["user_id"] == "user-1"
        assert calls[0]["organization_id"] == "org-1"
        assert [t["role"] for t in calls[0]["turns"]] == ["user", "tool", "assistant"]


class TestToolTurnsRoundTripThroughSchema:
    """The store schema must accept what extraction now produces."""

    def test_tool_role_valid_in_schema(self):
        from mirix.schemas.conversation_message import ConversationMessageCreate

        msg = ConversationMessageCreate(
            session_id="sess-tool-1",
            user_id="user-1",
            organization_id="org-1",
            role="tool",
            content="[run_tests] 2 failed",
        )
        assert msg.role == "tool"

    def test_distiller_renders_tool_turns(self):
        from types import SimpleNamespace

        from mirix.services.session_experience_distiller import (
            SessionExperienceDistiller,
        )

        transcript = SessionExperienceDistiller._render_transcript(
            [
                SimpleNamespace(role="user", content="deploy"),
                SimpleNamespace(role="tool", content="[kubectl] error: forbidden"),
                SimpleNamespace(role="assistant", content="fixed the RBAC and retried"),
            ]
        )
        assert "tool: [kubectl] error: forbidden" in transcript
