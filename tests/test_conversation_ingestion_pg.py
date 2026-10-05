"""Postgres integration tests for memory-add ingestion.

Postgres enforces what SQLite silently ignores here — the foreign keys from
``conversation_message`` and ``files`` to ``users`` — and stores
``content_parts`` as real JSON. Run through the fresh-database runner (MIRIX_PG_URI
must be set before import):

    scripts/run_pg_integration_tests.sh tests/test_conversation_ingestion_pg.py

The FastAPI app runs in-process (httpx ASGI transport) with the real user,
conversation, session, file and trigger managers on Postgres; only the meta
agent dispatch (``send_messages`` / ``put_messages``) is a spy, so no model is
ever called. Covered: a first-contact task write with images and an over-cap
message; the FK failure a task write propagates while a
conversation write stays additive; a conversation session's images reaching the
meta agent; prune and user erasure deleting images; a NUL character in
a task session's tool output, which Postgres cannot store, written as U+FFFD
with a warning instead of a 500.
"""

import base64
import datetime as dt
import io
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import select

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


def _png(color="red", size=(8, 6)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode()}"


def _sid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tables():
    """Schema, plus an engine pool bound to THIS module's event loop."""
    from mirix.server import server

    await server.engine.dispose(close=False)
    await server.ensure_tables_created()
    yield
    await server.engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def api(tables, monkeypatch, tmp_path):
    from mirix.orm.agent import Agent as AgentORM
    from mirix.orm.organization import Organization as OrganizationORM
    from mirix.schemas.agent import AgentType
    from mirix.schemas.client import Client as PydanticClient
    from mirix.server import rest_api
    from mirix.server.server import db_context
    from mirix.services import skill_learning_trigger
    from mirix.services.user_manager import UserManager

    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    monkeypatch.delenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", raising=False)

    org_id = f"it-org-{uuid.uuid4().hex[:8]}"
    client_id = f"it-client-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    async with db_context() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        await db.commit()
    async with db_context() as db:
        agent = AgentORM(
            id=agent_id,
            organization_id=org_id,
            agent_type="meta_memory_agent",
            name="meta",
        )
        agent._created_by_id = client_id
        db.add(agent)
        await db.commit()

    client = PydanticClient(
        id=client_id,
        organization_id=org_id,
        name=client_id,
        write_scope="it",
        read_scopes=["it"],
    )
    state = SimpleNamespace(
        org=org_id,
        client=client,
        agent_id=agent_id,
        context_window=250,
        sent=[],
        queued=[],
        scheduled=[],
    )

    class FakeClientManager:
        async def get_client_by_id(self, cid):
            return client if cid == client_id else None

    class FakeAgentManager:
        async def get_agent_by_id(self, aid, actor):
            return SimpleNamespace(
                id=agent_id,
                agent_type=AgentType.meta_memory_agent,
                llm_config=SimpleNamespace(
                    model="m", context_window=state.context_window
                ),
            )

    async def fake_send(**kwargs):
        state.sent.append(kwargs)

    async def fake_put(**kwargs):
        state.queued.append(kwargs)

    monkeypatch.setattr(
        rest_api,
        "get_server",
        lambda: SimpleNamespace(
            client_manager=FakeClientManager(),
            agent_manager=FakeAgentManager(),
            user_manager=UserManager(),
            send_messages=fake_send,
        ),
    )
    monkeypatch.setattr(rest_api, "put_messages", fake_put)
    monkeypatch.setattr(
        skill_learning_trigger,
        "schedule_learning_run",
        lambda run_id, **kw: state.scheduled.append(run_id),
    )

    transport = httpx.ASGITransport(app=rest_api.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:

        async def post(endpoint, **body):
            body.setdefault("meta_agent_id", agent_id)
            return await http.post(
                endpoint,
                json=body,
                headers={"x-client-id": client_id, "x-org-id": org_id},
            )

        state.post = post
        yield state


async def _rows(session_id):
    from mirix.orm.conversation_message import ConversationMessage as MessageORM
    from mirix.server.server import db_context

    async with db_context() as db:
        result = await db.execute(
            select(MessageORM)
            .where(MessageORM.session_id == session_id)
            .order_by(MessageORM.created_at)
        )
        return list(result.scalars().all())


async def _session(session_id):
    from mirix.orm.conversation_session import ConversationSession as SessionORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (
            await db.execute(
                select(SessionORM).where(SessionORM.session_id == session_id)
            )
        ).scalar_one_or_none()


async def _files(**filters):
    from mirix.orm.file import FileMetadata as FileORM
    from mirix.server.server import db_context

    async with db_context() as db:
        stmt = select(FileORM)
        for key, value in filters.items():
            stmt = stmt.where(getattr(FileORM, key) == value)
        return list((await db.execute(stmt)).scalars().all())


async def _user_exists(user_id) -> bool:
    from mirix.orm.user import User as UserORM
    from mirix.server.server import db_context

    async with db_context() as db:
        return (await db.get(UserORM, user_id)) is not None


async def test_the_store_enforces_the_user_foreign_key(api):
    """Why the user is created first: on Postgres a turn of a missing user is rejected."""
    from mirix.services.conversation_message_manager import ConversationMessageManager

    with pytest.raises(Exception):
        await ConversationMessageManager().record_turns(
            session_id=_sid("fk"),
            user_id=f"it-ghost-{uuid.uuid4().hex[:8]}",
            organization_id=api.org,
            turns=[{"role": "user", "content": "q"}],
            actor=api.client,
            session_tag="task",
            agent_id=api.agent_id,
        )


async def test_first_contact_task_write_with_images_and_an_over_cap_message(api):
    user_id = f"it-new-{uuid.uuid4().hex[:8]}"
    assert not await _user_exists(user_id)
    png = _png("navy", (12, 7))
    sid = _sid("pg-task")
    long_result = "BEGIN" + "r" * 6000 + "END"
    resp = await api.post(
        "/memory/add",
        user_id=user_id,
        session_id=sid,
        session_tag="task",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Your task is to: read the chart."},
                    {"type": "image_url", "image_url": {"url": _data_url(png)}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "ocr", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "name": "ocr", "content": long_result},
            {"role": "assistant", "content": "The chart shows 42."},
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "stored" and body["message_count"] == 4
    assert api.sent == [] and api.queued == []
    assert await _user_exists(user_id)  # Created before the write, FK satisfied

    user_row, call_row, tool_row, answer_row = await _rows(sid)
    # Images: JSON content_parts with a stored image ref; placeholders, no base64.
    parts = user_row.content_parts
    assert [p["type"] for p in parts] == ["text", "image_ref"]
    ref = parts[1]
    assert (ref["mime_type"], ref["width"], ref["height"], ref["bytes"]) == (
        "image/png",
        12,
        7,
        len(png),
    )
    assert (
        user_row.content
        == f"Your task is to: read the chart.\n[image: {ref['file_id']}]"
    )
    assert base64.b64encode(png).decode()[:40] not in user_row.content
    [file_row] = await _files(id=ref["file_id"])
    assert file_row.user_id == user_id and file_row.organization_id == api.org
    assert Path(file_row.file_path).read_bytes() == png
    # Text-only turns keep content_parts NULL.
    assert all(r.content_parts is None for r in (call_row, tool_row, answer_row))
    # Context window 250 -> cap 800 chars; head + tail + exact marker.
    stored_tool_text = "[ocr] " + long_result
    assert len(tool_row.content) <= 800
    assert tool_row.content.startswith("[ocr] BEGIN") and tool_row.content.endswith(
        "END"
    )
    assert tool_row.original_text_length == len(stored_tool_text)
    [removed] = [
        int(n) for n in re.findall(r"…\[truncated (\d+) chars\]…", tool_row.content)
    ]
    session = await _session(sid)
    assert session.session_tag == "task"
    assert (session.truncated_message_count, session.truncated_chars) == (1, removed)
    assert session.message_count == 4


async def test_task_storage_errors_propagate_conversation_ones_do_not(api, monkeypatch):
    """A real FK violation: with the user-creation step skipped, the write fails."""
    from mirix.server import rest_api

    async def skip_user_creation(server, client, user_id):
        return None

    monkeypatch.setattr(rest_api, "_ensure_user_exists", skip_user_creation)
    task_sid = _sid("pg-fk-task")
    resp = await api.post(
        "/memory/add",
        user_id=f"it-ghost-{uuid.uuid4().hex[:8]}",
        session_id=task_sid,
        session_tag="task",
        messages=[{"role": "user", "content": "q"}],
    )
    assert resp.status_code == 500
    assert await _rows(task_sid) == [] and await _session(task_sid) is None
    assert api.queued == []

    chat_sid = _sid("pg-fk-chat")
    resp = await api.post(
        "/memory/add",
        user_id=f"it-ghost-{uuid.uuid4().hex[:8]}",
        session_id=chat_sid,
        messages=[{"role": "user", "content": "q"}],
    )
    assert resp.status_code == 200 and resp.json()["status"] == "queued"
    assert len(api.queued) == 1
    assert await _rows(chat_sid) == []


async def test_a_nul_in_a_task_tool_message_is_stored_as_u_fffd(api):
    """Postgres rejects U+0000 in text and jsonb: without the replacement this
    write is a 500 (MirixServerError, INFRA, for the harness) on every retry."""
    user_id = f"it-nul-{uuid.uuid4().hex[:8]}"
    sid = _sid("pg-nul")
    resp = await api.post(
        "/memory/add",
        user_id=user_id,
        session_id=sid,
        session_tag="task",
        messages=[
            {"role": "user", "content": "Your task is to: read data.bin"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "read", "arguments": '{"p": "\u0000"}'},
                    }
                ],
            },
            {"role": "tool", "name": "read", "content": "PK\x03\x04\x00\x00ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "page\x00"},
                    {"type": "image_url", "image_url": {"url": _data_url(_png())}},
                ],
            },
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "stored" and body["message_count"] == 4
    assert body["warnings"] == [
        "4 NUL character(s) (U+0000) in 3 message(s) were stored as U+FFFD: "
        "the conversation store cannot hold U+0000"
    ]
    _, call_row, tool_row, image_row = await _rows(sid)
    assert call_row.content == '[tool_call] read({"p": "\ufffd"})'
    assert tool_row.content == "[read] PK\x03\x04\ufffd\ufffdok"
    assert image_row.content_parts[0] == {"type": "text", "text": "page\ufffd"}
    assert image_row.content.startswith("page\ufffd\n[image: ")
    session = await _session(sid)
    assert session.message_count == 4 and session.session_tag == "task"


async def test_conversation_images_reach_the_meta_agent(api):
    user_id = f"it-chat-{uuid.uuid4().hex[:8]}"
    sid = _sid("pg-chat")
    resp = await api.post(
        "/memory/add_sync",
        user_id=user_id,
        session_id=sid,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_data",
                        "image_data": {
                            "data": base64.b64encode(_png("green")).decode()
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "a green square"}],
            },
        ],
    )
    assert resp.status_code == 200, resp.text
    [call] = api.sent
    [message] = call["input_messages"]
    assert [c.type.value for c in message.content] == ["text", "image_url", "text"]
    assert message.content[2].text == "[ASSISTANT] a green square"
    [meta_file] = await _files(id=message.content[1].image_id)
    assert meta_file.user_id == user_id
    user_row, _ = await _rows(sid)
    assert user_row.content_parts[1]["file_id"] != meta_file.id
    assert user_row.session_tag == "conversation"


async def test_prune_and_erasure_delete_images(api):
    from mirix.services.conversation_message_manager import ConversationMessageManager
    from mirix.services.user_manager import UserManager

    user_id = f"it-k24-{uuid.uuid4().hex[:8]}"

    async def write(color, endpoint="/memory/add", tag="task"):
        sid = _sid("pg-k24")
        resp = await api.post(
            endpoint,
            user_id=user_id,
            session_id=sid,
            session_tag=tag,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png(color))},
                        }
                    ],
                }
            ],
        )
        assert resp.status_code == 200, resp.text
        [row] = await _rows(sid)
        [file_row] = await _files(id=row.content_parts[0]["file_id"])
        return sid, file_row.id, Path(file_row.file_path)

    pruned_sid, pruned_id, pruned_path = await write("maroon")
    kept_sid, kept_id, kept_path = await write("olive")
    await ConversationMessageManager().mark_sessions_distilled(
        session_watermarks={
            pruned_sid: dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)
        },
        user_id=user_id,
        organization_id=api.org,
        actor=api.client,
    )
    future = ConversationMessageManager(
        clock=lambda: dt.datetime(2100, 6, 1, tzinfo=dt.timezone.utc)
    )
    assert (
        await future.prune_distilled_turns(
            user_id=user_id, organization_id=api.org, actor=api.client, retention_days=1
        )
        == 1
    )
    assert await _files(id=pruned_id) == [] and not pruned_path.exists()
    assert len(await _files(id=kept_id)) == 1 and kept_path.exists()

    # A conversation write leaves a meta-agent copy too; full erasure removes all.
    _, chat_store_id, chat_path = await write(
        "teal", endpoint="/memory/add_sync", tag="conversation"
    )
    assert len(await _files(user_id=user_id)) == 3  # kept + chat store copy + meta copy
    await UserManager().delete_memories_by_user_id(user_id)
    assert await _files(user_id=user_id) == []
    assert not kept_path.exists() and not chat_path.exists()
    assert await _rows(kept_sid) == []
