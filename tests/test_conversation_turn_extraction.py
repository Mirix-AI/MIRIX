"""The conversation-store ingestion seam (`rest_api._ingest_session_turns`).

The distiller's prompt treats tool errors/retries as strong learning signals,
so the store keeps tool results and assistant tool_calls with their REAL roles.
Turn extraction now goes through the one parser both consumers share
(`mirix.services.conversation_content`, covered in test_conversation_content);
these tests pin the seam around it:

* the batch session_id and the real roles reach `record_turns` (this test
  used to fail on every run: the request had no `session_tag` attribute and the
  AttributeError was swallowed);
* conversation sessions (non-strict): a store or parse failure degrades to a
  missed session and never aborts the primary memory-add;
* task sessions (strict): a store failure propagates, and any image already
  stored for the failed write is discarded;
* the storage cap and image refs are handed to `record_turns`.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mirix.server.rest_api import _ingest_session_turns
from mirix.services import image_store
from mirix.services.conversation_message_manager import ConversationMessageManager


def _request(session_id, messages, *, tag=None):
    return SimpleNamespace(
        session_id=session_id,
        messages=messages,
        session_tag=tag,
        meta_agent_id="agent-meta",
    )


CLIENT = SimpleNamespace(id="client-1", organization_id="org-1")


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    async def record(self, **kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(ConversationMessageManager, "record_turns", record)
    return calls


@pytest.mark.asyncio
class TestIngestSessionTurns:
    async def test_session_id_stamped_and_turns_recorded(self, recorder):
        """The batch session_id and the REAL roles reach the store."""
        result = await _ingest_session_turns(
            _request(
                "sess-batch",
                [
                    {"role": "user", "content": "q"},
                    {"role": "tool", "name": "grep", "content": "no match"},
                    {"role": "assistant", "content": "a"},
                ],
                tag="conversation",
            ),
            CLIENT,
            "user-1",
            max_text_chars=1234,
        )
        assert result is not None and result.stored_count == 0
        [call] = recorder
        assert call["session_id"] == "sess-batch"
        assert call["user_id"] == "user-1"
        assert call["organization_id"] == "org-1"
        assert call["session_tag"] == "conversation"
        assert call["agent_id"] == "agent-meta"
        assert call["max_text_chars"] == 1234
        assert call["turns"] == [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "[grep] no match"},
            {"role": "assistant", "content": "a"},
        ]

    async def test_no_session_id_skips_the_store(self, recorder):
        assert (
            await _ingest_session_turns(
                _request(None, [{"role": "user", "content": "q"}]), CLIENT, "u"
            )
            is None
        )
        assert recorder == []

    async def test_only_foreign_roles_write_nothing(self, recorder):
        result = await _ingest_session_turns(
            _request("sess-sys", [{"role": "system", "content": "scaffolding"}]),
            CLIENT,
            "u",
        )
        assert result is None and recorder == []

    async def test_store_failure_does_not_propagate_for_conversation_sessions(
        self, monkeypatch
    ):
        async def boom(self, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(ConversationMessageManager, "record_turns", boom)
        result = await _ingest_session_turns(
            _request("sess-iso-1", [{"role": "user", "content": "q"}]), CLIENT, "user-1"
        )
        assert result is None

    async def test_parse_failure_does_not_propagate_for_conversation_sessions(
        self, recorder
    ):
        result = await _ingest_session_turns(
            _request(
                "sess-iso-2", [{"role": "user", "content": [{"type": "input_audio"}]}]
            ),
            CLIENT,
            "user-1",
        )
        assert result is None and recorder == []

    async def test_store_failure_propagates_for_task_sessions(self, monkeypatch):
        async def boom(self, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(ConversationMessageManager, "record_turns", boom)
        with pytest.raises(RuntimeError, match="store down"):
            await _ingest_session_turns(
                _request("sess-task", [{"role": "user", "content": "q"}], tag="task"),
                CLIENT,
                "user-1",
                strict=True,
            )

    async def test_strict_parse_failure_is_a_422(self, recorder):
        with pytest.raises(HTTPException) as exc:
            await _ingest_session_turns(
                _request(
                    "sess-task",
                    [{"role": "user", "content": [{"type": "input_audio"}]}],
                    tag="task",
                ),
                CLIENT,
                "user-1",
                strict=True,
            )
        assert exc.value.status_code == 422

    async def test_images_are_stored_first_and_discarded_when_the_write_fails(
        self, monkeypatch
    ):
        stored, discarded = [], []

        async def fake_store(
            part, *, organization_id, user_id, file_manager=None, images_dir=None
        ):
            ref = {
                "type": "image_ref",
                "file_id": f"file-{len(stored)}",
                "url": part.url,
            }
            stored.append((ref, organization_id, user_id))
            return ref

        async def fake_discard(refs, *, organization_id, session_maker=None):
            discarded.append((list(refs), organization_id))

        seen = []

        async def record(self, **kwargs):
            seen.append(kwargs["turns"])
            raise RuntimeError("store down")

        monkeypatch.setattr(image_store, "store_image_part", fake_store)
        monkeypatch.setattr(image_store, "discard_image_refs", fake_discard)
        monkeypatch.setattr(ConversationMessageManager, "record_turns", record)
        request = _request(
            "sess-img",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/1.png"},
                        },
                    ],
                }
            ],
            tag="task",
        )
        with pytest.raises(RuntimeError):
            await _ingest_session_turns(request, CLIENT, "user-1", strict=True)
        [turns] = seen
        assert turns == [
            {
                "role": "user",
                "content_parts": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_ref",
                        "file_id": "file-0",
                        "url": "https://x.test/1.png",
                    },
                ],
            }
        ]
        assert stored[0][1:] == ("org-1", "user-1")
        assert discarded == [([stored[0][0]], "org-1")]


class TestToolTurnsRoundTripThroughSchema:
    """The store schema must accept what extraction produces."""

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
        from mirix.services.context_budget import TokenCounter
        from mirix.services.distill_transcript import TranscriptPlanner, build_messages

        counter = TokenCounter()
        messages, _ = build_messages(
            [
                SimpleNamespace(role="user", content="deploy"),
                SimpleNamespace(role="tool", content="[kubectl] error: forbidden"),
                SimpleNamespace(role="assistant", content="fixed the RBAC and retried"),
            ],
            counter=counter,
        )
        transcript = TranscriptPlanner(
            messages, agent_id="a", session_id="s", counter=counter
        ).whole_chunk().text()
        assert "tool: [kubectl] error: forbidden" in transcript
