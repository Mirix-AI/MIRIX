"""Tests for conversation-turn extraction into the Conversation Message Store.

The distiller's prompt has always treated tool errors/retries as strong
learning signals (`signal_type: tool_error`), but the ingestion seam used to
drop every non-user/assistant message — blinding the distiller to exactly that
signal. These tests pin the repaired pipeline: tool results and assistant
tool_calls survive extraction, content parts are flattened to clean text, and
foreign roles are still dropped.
"""

from mirix.server.rest_api import _extract_conversation_turns, _serialize_tool_calls


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
        assert turns[1] == {"role": "tool", "content": "[run_tests] 2 failed, 10 passed"}

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
                {"role": "tool", "name": "kubectl_apply", "content": "error: forbidden"},
            ]
        )
        assert len(turns) == 3
        assert turns[1]["role"] == "assistant"
        assert 'kubectl_apply({"file": "deploy.yaml"})' in turns[1]["content"]
        assert turns[2] == {"role": "tool", "content": "[kubectl_apply] error: forbidden"}

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
                    "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
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
        assert _extract_conversation_turns([{"type": "text", "text": "screenshot"}]) == []
        assert _extract_conversation_turns([]) == []


class TestSerializeToolCalls:
    def test_openai_shape(self):
        rendered = _serialize_tool_calls(
            [{"function": {"name": "grep", "arguments": '{"q": "x"}'}}]
        )
        assert rendered == '[tool_call] grep({"q": "x"})'

    def test_flat_shape_and_dict_arguments(self):
        rendered = _serialize_tool_calls([{"name": "read", "arguments": {"path": "a.py"}}])
        assert rendered == '[tool_call] read({"path": "a.py"})'

    def test_unrecognized_falls_back_to_str(self):
        assert "[tool_call]" in _serialize_tool_calls(["opaque"])


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
