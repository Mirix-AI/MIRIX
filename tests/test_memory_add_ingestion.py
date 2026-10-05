"""/memory/add and /memory/add_sync ingestion.

The FastAPI app runs in-process over httpx's ASGI transport against a throwaway
SQLite database and a tmp images directory; the server object is a fake whose
``send_messages`` and the module's ``put_messages`` are spies, so nothing ever
reaches a model. The real managers (users, conversation store, sessions,
files, trigger) run against the database.

Covers:
* a task session is stored only: it never reaches ``send_messages`` or
  ``put_messages``; 422 without a session_id or without storable turns;
  storage errors propagate; the trigger check runs and its run id is echoed;
* the first write fixes the tag; a different tag only warns;
* a first-contact user is created before the store write;
* empty lists, bad shapes and bad images are 422 (nothing written);
  list-of-parts and image payloads no longer 500 on either endpoint;
* images land as ``content_parts`` image refs (no base64 in the store),
  text-only turns keep ``content_parts`` NULL, the meta agent gets its own
  image copies;
* head + tail truncation from the meta agent's context window, the env
  override, ``original_text_length`` and the session's truncation counters;
* prune and erasure remove the images only the deleted turns referenced;
* images survive the queue; the worker drops task-tagged messages;
* the client's ``add(session_tag=)``; the content-part converter fixes;
* a NUL (U+0000) in a turn is stored as U+FFFD (Postgres cannot hold
  it), counted in the response's ``warnings``, on both consumers.
"""

import base64
import datetime as dt
import io
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.conversation_message import ConversationMessage as MessageORM
from mirix.orm.conversation_session import ConversationSession as SessionORM
from mirix.orm.file import FileMetadata as FileORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.server import rest_api
from mirix.services import skill_learning_trigger
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_session_manager import ConversationSessionManager

pytestmark = pytest.mark.asyncio(loop_scope="module")

ENDPOINTS = ["/memory/add", "/memory/add_sync"]


def _png_bytes(size=(5, 4), color="red", fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


def _data_url(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("memory_add_ingestion") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    local = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def _ctx():
        async with local() as session:
            try:
                yield session
            finally:
                await session.close()

    yield _ctx
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def api(sm, monkeypatch, tmp_path):
    import mirix.server.server as server_module
    from mirix.services.user_manager import UserManager
    from mirix.settings import settings

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(settings, "images_dir", tmp_path / "images")
    monkeypatch.delenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", raising=False)

    org_id = f"org-{uuid.uuid4().hex[:8]}"
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    # A second client of the same organization, with its own meta agent.
    other_client_id = f"client-{uuid.uuid4().hex[:8]}"
    other_agent_id = f"agent-{uuid.uuid4()}"
    # A memory sub-agent of the first client (not a meta agent).
    sub_agent_id = f"agent-{uuid.uuid4()}"
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        for cid in (client_id, other_client_id):
            db.add(
                ClientORM(
                    id=cid,
                    organization_id=org_id,
                    name=cid,
                    status="active",
                    write_scope="test",
                    read_scopes=["test"],
                )
            )
        sub_agent = AgentORM(
            id=sub_agent_id,
            organization_id=org_id,
            agent_type="episodic_memory_agent",
            name="episodic",
        )
        sub_agent._created_by_id = client_id
        db.add(sub_agent)
        other_agent = AgentORM(
            id=other_agent_id,
            organization_id=org_id,
            agent_type="meta_memory_agent",
            name="meta-2",
        )
        other_agent._created_by_id = other_client_id
        db.add(other_agent)
        db.add(
            UserORM(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
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
        write_scope="test",
        read_scopes=["test"],
    )
    other_client = PydanticClient(
        id=other_client_id,
        organization_id=org_id,
        name=other_client_id,
        write_scope="other",
        read_scopes=["other"],
    )
    state = SimpleNamespace(
        client=client,
        other_client=other_client,
        other_agent_id=other_agent_id,
        sub_agent_id=sub_agent_id,
        org=org_id,
        user=user_id,
        agent_id=agent_id,
        context_window=1_000_000,
        sent=[],
        queued=[],
        scheduled=[],
        sm=sm,
        images=tmp_path / "images",
    )

    class FakeClientManager:
        async def get_client_by_id(self, cid):
            return {client_id: client, other_client_id: other_client}.get(cid)

    class FakeAgentManager:
        async def get_agent_by_id(self, aid, actor):
            owned = {
                agent_id: (client_id, AgentType.meta_memory_agent),
                sub_agent_id: (client_id, AgentType.episodic_memory_agent),
                other_agent_id: (other_client_id, AgentType.meta_memory_agent),
            }
            owner, agent_type = owned[aid]
            assert actor.id == owner
            return SimpleNamespace(
                id=aid,
                agent_type=agent_type,
                llm_config=SimpleNamespace(
                    model="m", context_window=state.context_window
                ),
            )

    async def fake_send_messages(**kwargs):
        state.sent.append(kwargs)

    async def fake_put_messages(**kwargs):
        state.queued.append(kwargs)

    def fake_schedule(run_id, **context):
        state.scheduled.append(run_id)

    fake_server = SimpleNamespace(
        client_manager=FakeClientManager(),
        agent_manager=FakeAgentManager(),
        user_manager=UserManager(),
        send_messages=fake_send_messages,
    )
    monkeypatch.setattr(rest_api, "get_server", lambda: fake_server)
    monkeypatch.setattr(rest_api, "put_messages", fake_put_messages)
    monkeypatch.setattr(skill_learning_trigger, "schedule_learning_run", fake_schedule)

    transport = httpx.ASGITransport(app=rest_api.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        state.http = http

        async def post(endpoint, **body):
            body.setdefault("meta_agent_id", agent_id)
            body.setdefault("user_id", user_id)
            return await http.post(
                endpoint,
                json=body,
                headers={"x-client-id": client_id, "x-org-id": org_id},
            )

        async def post_as_other(endpoint, **body):
            body.setdefault("meta_agent_id", other_agent_id)
            body.setdefault("user_id", user_id)
            return await http.post(
                endpoint,
                json=body,
                headers={"x-client-id": other_client_id, "x-org-id": org_id},
            )

        state.post = post
        state.post_as_other = post_as_other
        yield state


def _sid(prefix="s") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _rows(sm, session_id):
    async with sm() as db:
        result = await db.execute(
            select(MessageORM)
            .where(MessageORM.session_id == session_id)
            .order_by(MessageORM.created_at)
        )
        return list(result.scalars().all())


async def _session(sm, session_id):
    async with sm() as db:
        result = await db.execute(
            select(SessionORM).where(SessionORM.session_id == session_id)
        )
        return result.scalar_one_or_none()


async def _files(sm, **filters):
    async with sm() as db:
        stmt = select(FileORM)
        for key, value in filters.items():
            stmt = stmt.where(getattr(FileORM, key) == value)
        return list((await db.execute(stmt)).scalars().all())


TASK_TURNS = [
    {"role": "user", "content": "Your task is to: put a mug in the sink."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "function": {"name": "act", "arguments": '{"a": "go to sink 1"}'},
            }
        ],
    },
    {"role": "tool", "name": "act", "content": "You arrive at sink 1."},
    {"role": "assistant", "content": "done"},
]


# --------------------------------------------------------------------------- #
# Task sessions are stored only                                                #
# --------------------------------------------------------------------------- #


class TestTaskSessions:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_task_write_is_stored_and_never_reaches_the_meta_agent(
        self, api, endpoint
    ):
        sid = _sid("task")
        resp = await api.post(
            endpoint, messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "stored"
        assert body["session_id"] == sid
        assert body["learning_run_id"] is None
        assert body["warnings"] == []
        assert body["message_count"] == 4
        # The spies prove the task session is stored only: no meta dispatch, no queue.
        assert api.sent == [] and api.queued == []

        rows = await _rows(api.sm, sid)
        assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
        assert rows[1].content == '[tool_call] act({"a": "go to sink 1"})'
        assert rows[2].content == "[act] You arrive at sink 1."
        assert all(r.session_tag == "task" and r.content_parts is None for r in rows)
        session = await _session(api.sm, sid)
        assert session.session_tag == "task" and session.agent_id == api.agent_id

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_task_without_session_id_is_422(self, api, endpoint):
        resp = await api.post(endpoint, messages=TASK_TURNS, session_tag="task")
        assert resp.status_code == 422
        assert "session_id" in resp.json()["detail"]
        assert api.sent == [] and api.queued == []

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_task_without_storable_turns_is_422(self, api, endpoint):
        sid = _sid("task-empty")
        resp = await api.post(
            endpoint,
            messages=[{"type": "text", "text": "a bare content part, no roles"}],
            session_id=sid,
            session_tag="task",
        )
        assert resp.status_code == 422
        assert await _rows(api.sm, sid) == [] and await _session(api.sm, sid) is None

    async def test_default_tag_task_without_session_id_is_422(self, api, monkeypatch):
        monkeypatch.setattr("mirix.constants.DEFAULT_SESSION_TAG", "task")
        resp = await api.post("/memory/add", messages=TASK_TURNS)
        assert resp.status_code == 422

    async def test_first_contact_user_is_created_before_the_store_write(
        self, api, monkeypatch
    ):
        """A direct REST caller's new user must exist before its first turns
        are written (conversation_message.user_id and files.user_id are FKs)."""
        new_user = f"user-new-{uuid.uuid4().hex[:8]}"
        seen = []
        real_record = ConversationMessageManager.record_turns

        async def spy_record(self, **kwargs):
            async with api.sm() as db:
                seen.append(await db.get(UserORM, kwargs["user_id"]))
            return await real_record(self, **kwargs)

        monkeypatch.setattr(ConversationMessageManager, "record_turns", spy_record)
        sid = _sid("first")
        resp = await api.post(
            "/memory/add",
            user_id=new_user,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes())},
                        },
                    ],
                }
            ],
            session_id=sid,
            session_tag="task",
        )
        assert resp.status_code == 200, resp.text
        assert len(seen) == 1 and seen[0] is not None
        assert seen[0].organization_id == api.org
        [row] = await _rows(api.sm, sid)
        [image] = await _files(api.sm, id=row.content_parts[1]["file_id"])
        assert image.user_id == new_user

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_task_storage_errors_propagate_and_images_are_discarded(
        self, api, monkeypatch, endpoint
    ):
        async def broken(self, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(ConversationMessageManager, "record_turns", broken)
        before = len(await _files(api.sm, organization_id=api.org))
        sid = _sid("broken")
        resp = await api.post(
            endpoint,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes(color="teal"))},
                        }
                    ],
                }
            ],
            session_id=sid,
            session_tag="task",
        )
        assert resp.status_code == 500
        assert api.sent == [] and api.queued == []
        assert len(await _files(api.sm, organization_id=api.org)) == before

    async def test_trigger_check_runs_after_the_write_and_its_run_is_returned(
        self, api
    ):
        """Five ended task sessions are pending; the next task write's trigger
        check claims them and the response carries the learning run id."""
        for _ in range(5):
            sid = _sid("ended")
            await ConversationMessageManager().record_turns(
                session_id=sid,
                user_id=api.user,
                organization_id=api.org,
                turns=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                actor=api.client,
                session_tag="task",
                agent_id=api.agent_id,
            )
            await ConversationSessionManager().close_session(
                organization_id=api.org, user_id=api.user, session_id=sid
            )
        resp = await api.post(
            "/memory/add_sync",
            messages=TASK_TURNS,
            session_id=_sid("live"),
            session_tag="task",
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["learning_run_id"]
        assert run_id and api.scheduled == [run_id]
        assert api.sent == []

    async def test_conversation_writes_never_run_the_trigger(self, api, monkeypatch):
        calls = []

        async def spy_trigger(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(
            skill_learning_trigger, "trigger_skill_learning", spy_trigger
        )
        resp = await api.post(
            "/memory/add", messages=TASK_TURNS, session_id=_sid("chat")
        )
        assert resp.status_code == 200 and calls == []


class TestTagResolution:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_task_session_stays_task_and_warns(self, api, endpoint):
        sid = _sid("q2")
        first = await api.post(
            endpoint, messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert first.json()["status"] == "stored"
        second = await api.post(
            endpoint, messages=TASK_TURNS, session_id=sid, session_tag="conversation"
        )
        assert second.status_code == 200
        body = second.json()
        assert body["status"] == "stored"
        assert body["warnings"] and body["warnings"][0].startswith(
            "session_tag mismatch"
        )
        # No tag at all keeps the stored one silently.
        third = await api.post(endpoint, messages=TASK_TURNS, session_id=sid)
        assert third.json()["status"] == "stored" and third.json()["warnings"] == []
        assert api.sent == [] and api.queued == []
        assert len(await _rows(api.sm, sid)) == 12

    async def test_conversation_session_stays_conversation_and_warns(self, api):
        sid = _sid("q2c")
        first = await api.post(
            "/memory/add",
            messages=TASK_TURNS,
            session_id=sid,
            session_tag="conversation",
        )
        assert first.json()["status"] == "queued"
        second = await api.post(
            "/memory/add", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert second.status_code == 200
        assert second.json()["status"] == "queued"
        assert second.json()["warnings"][0].startswith("session_tag mismatch")
        assert len(api.queued) == 2
        assert api.queued[1]["filter_tags"]["session_tag"] == "conversation"


# --------------------------------------------------------------------------- #
# Bad payloads are 422, nothing is written                                     #
# --------------------------------------------------------------------------- #


class TestRejectedPayloads:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_empty_messages_is_422(self, api, endpoint):
        sid = _sid("empty")
        resp = await api.post(endpoint, messages=[], session_id=sid)
        assert resp.status_code == 422
        assert await _session(api.sm, sid) is None

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": [{"type": "input_audio", "data": "x"}]}],
            [{"role": "user", "content": "a"}, {"type": "text", "text": "b"}],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}
                    ],
                }
            ],
        ],
    )
    async def test_malformed_payloads_are_422(self, api, endpoint, messages):
        sid = _sid("bad")
        resp = await api.post(endpoint, messages=messages, session_id=sid)
        assert resp.status_code == 422
        assert await _session(api.sm, sid) is None
        assert api.sent == [] and api.queued == []

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    @pytest.mark.parametrize("tag", ["task", "conversation"])
    async def test_undecodable_image_is_422_before_any_write(self, api, endpoint, tag):
        before = len(await _files(api.sm, organization_id=api.org))
        sid = _sid("badimg")
        resp = await api.post(
            endpoint,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes(color="lime"))},
                        },
                        {
                            "type": "image_data",
                            "image_data": {
                                "data": base64.b64encode(b"not an image").decode()
                            },
                        },
                    ],
                }
            ],
            session_id=sid,
            session_tag=tag,
        )
        assert resp.status_code == 422
        assert "image" in resp.json()["detail"]
        assert await _session(api.sm, sid) is None
        assert len(await _files(api.sm, organization_id=api.org)) == before
        assert api.sent == [] and api.queued == []


# --------------------------------------------------------------------------- #
# Conversation path                                                            #
# --------------------------------------------------------------------------- #


class TestConversationPath:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_list_of_text_parts_no_longer_500s(self, api, endpoint):
        """The SDK-recommended parts format crashed the role collapse."""
        resp = await api.post(
            endpoint,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            ],
        )
        assert resp.status_code == 200, resp.text
        [call] = api.sent if endpoint.endswith("_sync") else api.queued
        [message] = call["input_messages"]
        assert [c.text for c in message.content] == [
            "[USER] hello\nworld",
            "[ASSISTANT] hi",
        ]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_images_are_stored_as_refs_and_reach_the_meta_agent(
        self, api, endpoint
    ):
        png = _png_bytes(size=(9, 7), color="navy")
        jpeg = _png_bytes(size=(3, 2), color="gold", fmt="JPEG")
        png_b64 = base64.b64encode(png).decode()
        jpeg_b64 = base64.b64encode(jpeg).decode()
        sid = _sid("img")
        resp = await api.post(
            endpoint,
            session_id=sid,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what are these?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(png), "detail": "high"},
                        },
                        {"type": "image_data", "image_data": {"data": jpeg_b64}},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/remote.png"},
                        },
                    ],
                },
                {"role": "assistant", "content": "a chart, a photo and a link"},
            ],
        )
        assert resp.status_code == 200, resp.text

        user_row, assistant_row = await _rows(api.sm, sid)
        assert assistant_row.content_parts is None
        assert assistant_row.content == "a chart, a photo and a link"
        parts = user_row.content_parts
        assert [p["type"] for p in parts] == [
            "text",
            "image_ref",
            "image_ref",
            "image_ref",
        ]
        assert parts[1]["mime_type"] == "image/png" and (
            parts[1]["width"],
            parts[1]["height"],
        ) == (9, 7)
        assert parts[1]["detail"] == "high" and parts[1]["bytes"] == len(png)
        assert parts[2]["mime_type"] == "image/jpeg" and (
            parts[2]["width"],
            parts[2]["height"],
        ) == (3, 2)
        assert parts[3]["url"] == "https://x.test/remote.png"
        assert user_row.content == "\n".join(
            ["what are these?"] + [f"[image: {p['file_id']}]" for p in parts[1:]]
        )
        # No base64 anywhere in the store.
        for blob in (user_row.content, str(parts)):
            assert (
                png_b64[:40] not in blob
                and jpeg_b64[:40] not in blob
                and "base64" not in blob
            )

        store_ids = {p["file_id"] for p in parts[1:]}
        for fid in store_ids:
            [row] = await _files(api.sm, id=fid)
            assert row.user_id == api.user
        [store_png] = await _files(api.sm, id=parts[1]["file_id"])
        assert Path(store_png.file_path).read_bytes() == png

        # The meta agent got the images too — through its OWN files rows.
        [call] = api.sent if endpoint.endswith("_sync") else api.queued
        [message] = call["input_messages"]
        assert message.session_id == sid
        kinds = [c.type.value for c in message.content]
        assert kinds == ["text", "image_url", "image_url", "image_url", "text"]
        meta_ids = {c.image_id for c in message.content if c.type.value == "image_url"}
        assert len(meta_ids) == 3 and not (meta_ids & store_ids)
        assert message.content[0].text == "[USER] what are these?"
        assert message.content[1].detail == "high"
        [meta_png] = await _files(api.sm, id=message.content[1].image_id)
        assert meta_png.file_path == store_png.file_path  # same bytes, one disk file
        assert call["filter_tags"]["session_tag"] == "conversation"
        assert call["filter_tags"]["session_id"] == sid

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_single_image_content_only_payload(self, api, endpoint):
        """A lone image part (legacy content-only format) reaches the agent as a
        one-part list; the user-message assertion no longer fires downstream."""
        resp = await api.post(
            endpoint,
            messages=[
                {"type": "image_url", "image_url": {"url": _data_url(_png_bytes())}}
            ],
        )
        assert resp.status_code == 200, resp.text
        [call] = api.sent if endpoint.endswith("_sync") else api.queued
        [message] = call["input_messages"]
        assert [c.type.value for c in message.content] == ["image_url"]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_unreadable_legacy_file_part_is_422(self, api, endpoint, tmp_path):
        resp = await api.post(
            endpoint,
            messages=[{"type": "file_uri", "file_uri": str(tmp_path / "missing.pdf")}],
        )
        assert resp.status_code == 422
        assert api.sent == [] and api.queued == []

    async def test_without_session_id_nothing_is_stored(self, api):
        before = await _files(api.sm, organization_id=api.org)
        resp = await api.post(
            "/memory/add", messages=[{"role": "user", "content": "hi"}]
        )
        assert resp.status_code == 200
        assert (
            api.queued and api.queued[0]["filter_tags"]["session_tag"] == "conversation"
        )
        assert "session_id" not in api.queued[0]["filter_tags"]
        assert len(await _files(api.sm, organization_id=api.org)) == len(before)

    async def test_conversation_store_failure_is_additive(self, api, monkeypatch):
        async def broken(self, **kwargs):
            raise RuntimeError("store down")

        monkeypatch.setattr(ConversationMessageManager, "record_turns", broken)
        resp = await api.post(
            "/memory/add_sync", messages=TASK_TURNS, session_id=_sid("chat-broken")
        )
        assert resp.status_code == 200
        assert len(api.sent) == 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_client_without_write_scope_writes_nothing(self, api, endpoint):
        api.client.write_scope = None
        try:
            sid = _sid("noscope")
            resp = await api.post(
                endpoint, messages=TASK_TURNS, session_id=sid, session_tag="task"
            )
            assert resp.status_code == 403
            assert await _session(api.sm, sid) is None
        finally:
            api.client.write_scope = "test"


# --------------------------------------------------------------------------- #
# Storage cap                                                                  #
# --------------------------------------------------------------------------- #


class TestStorageCap:
    async def test_over_the_cap_keeps_head_and_tail_and_counts(self, api):
        api.context_window = 250  # cap = floor(250 * 0.8) * 4 = 800
        sid = _sid("cap")
        long_text = "HEAD" + "m" * 5000 + "TAIL"
        resp = await api.post(
            "/memory/add",
            session_id=sid,
            session_tag="task",
            messages=[
                {"role": "tool", "name": "dump", "content": long_text},
                {"role": "assistant", "content": "short"},
            ],
        )
        assert resp.status_code == 200, resp.text
        tool_row, short_row = await _rows(api.sm, sid)
        stored_text = "[dump] " + long_text
        assert len(tool_row.content) <= 800
        assert tool_row.content.startswith("[dump] HEAD") and tool_row.content.endswith(
            "TAIL"
        )
        assert tool_row.original_text_length == len(stored_text)
        [n] = [
            int(m) for m in re.findall(r"…\[truncated (\d+) chars\]…", tool_row.content)
        ]
        assert short_row.original_text_length is None and short_row.content == "short"
        session = await _session(api.sm, sid)
        assert session.truncated_message_count == 1
        assert session.truncated_chars == n
        assert len(tool_row.content) - len(f"…[truncated {n} chars]…") + n == len(
            stored_text
        )

        # A second over-long write accumulates on the session.
        resp = await api.post(
            "/memory/add",
            session_id=sid,
            messages=[{"role": "user", "content": "x" * 2000}],
        )
        assert resp.json()["status"] == "stored"
        session = await _session(api.sm, sid)
        assert session.truncated_message_count == 2
        assert session.truncated_chars > n

    async def test_cap_follows_the_context_window(self, api):
        text = "y" * 3000
        api.context_window = 250  # cap 800
        small = _sid("cw-small")
        await api.post(
            "/memory/add",
            session_id=small,
            session_tag="task",
            messages=[{"role": "user", "content": text}],
        )
        api.context_window = 1000  # cap 3200
        big = _sid("cw-big")
        await api.post(
            "/memory/add",
            session_id=big,
            session_tag="task",
            messages=[{"role": "user", "content": text}],
        )
        [small_row] = await _rows(api.sm, small)
        [big_row] = await _rows(api.sm, big)
        assert len(small_row.content) <= 800 and small_row.original_text_length == 3000
        assert big_row.content == text and big_row.original_text_length is None

    async def test_env_override_lowers_the_cap(self, api, monkeypatch):
        api.context_window = 1_000_000
        monkeypatch.setenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", "300")
        sid = _sid("cap-env")
        await api.post(
            "/memory/add",
            session_id=sid,
            session_tag="task",
            messages=[{"role": "user", "content": "z" * 1000}],
        )
        [row] = await _rows(api.sm, sid)
        assert len(row.content) <= 300 and row.original_text_length == 1000

    async def test_image_message_caps_text_parts_only(self, api):
        api.context_window = 250  # cap 800
        sid = _sid("cap-img")
        resp = await api.post(
            "/memory/add",
            session_id=sid,
            session_tag="task",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "short caption"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes())},
                        },
                        {"type": "text", "text": "Q" * 4000},
                    ],
                }
            ],
        )
        assert resp.status_code == 200, resp.text
        [row] = await _rows(api.sm, sid)
        texts = [p["text"] for p in row.content_parts if p["type"] == "text"]
        assert texts[0] == "short caption"
        assert sum(len(t) for t in texts) <= 800
        assert "truncated" in texts[1]
        assert row.content_parts[1]["type"] == "image_ref"
        assert row.original_text_length == len("short caption") + 4000
        assert f"[image: {row.content_parts[1]['file_id']}]" in row.content


# --------------------------------------------------------------------------- #
# NUL characters are stored as U+FFFD                                          #
# --------------------------------------------------------------------------- #


class TestNulCharacters:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_a_task_turn_with_nul_is_stored_with_u_fffd_and_a_warning(
        self, api, endpoint
    ):
        sid = _sid("nul")
        resp = await api.post(
            endpoint,
            session_id=sid,
            session_tag="task",
            messages=[
                {"role": "user", "content": "read the file"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "python", "arguments": '"a\u0000b"'},
                        }
                    ],
                },
                {"role": "tool", "name": "python", "content": "x\x00y\x00z"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "caption\x00"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes())},
                        },
                    ],
                },
            ],
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "stored" and body["message_count"] == 4
        [warning] = body["warnings"]
        assert "4 NUL character(s)" in warning and "in 3 message(s)" in warning
        _, call_row, tool_row, image_row = await _rows(api.sm, sid)
        assert call_row.content == '[tool_call] python("a\ufffdb")'
        assert tool_row.content == "[python] x\ufffdy\ufffdz"
        assert image_row.content_parts[0]["text"] == "caption\ufffd"
        assert image_row.content.startswith("caption\ufffd\n[image: ")
        assert all("\x00" not in r.content for r in (call_row, tool_row, image_row))

    async def test_nul_in_a_conversation_reaches_the_meta_agent_replaced(self, api):
        sid = _sid("nul-chat")
        resp = await api.post(
            "/memory/add_sync",
            session_id=sid,
            messages=[{"role": "user", "content": "a\x00b"}],
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["warnings"] == [
            "1 NUL character(s) (U+0000) in 1 message(s) were stored as U+FFFD: "
            "the conversation store cannot hold U+0000"
        ]
        [call] = api.sent
        [message] = call["input_messages"]
        assert [c.text for c in message.content] == ["[USER] a\ufffdb"]
        [row] = await _rows(api.sm, sid)
        assert row.content == "a\ufffdb"

    async def test_text_without_nul_gets_no_warning(self, api):
        sid = _sid("no-nul")
        resp = await api.post(
            "/memory/add",
            session_id=sid,
            session_tag="task",
            messages=[{"role": "user", "content": "plain \\x00 text"}],
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["warnings"] == []
        [row] = await _rows(api.sm, sid)
        assert row.content == "plain \\x00 text"


# --------------------------------------------------------------------------- #
# Prune and erasure remove images                                              #
# --------------------------------------------------------------------------- #


async def _task_session_with_image(api, color):
    sid = _sid("k24")
    resp = await api.post(
        "/memory/add",
        session_id=sid,
        session_tag="task",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(_png_bytes(color=color))},
                    }
                ],
            },
            {"role": "assistant", "content": "seen"},
        ],
    )
    assert resp.status_code == 200, resp.text
    rows = await _rows(api.sm, sid)
    file_id = rows[0].content_parts[0]["file_id"]
    [file_row] = await _files(api.sm, id=file_id)
    return sid, file_id, Path(file_row.file_path)


class TestImageLifecycle:
    async def test_prune_deletes_images_of_pruned_turns(self, api):
        sid, file_id, path = await _task_session_with_image(api, "maroon")
        keep_sid, keep_id, keep_path = await _task_session_with_image(api, "olive")
        manager = ConversationMessageManager()
        await manager.mark_sessions_distilled(
            session_watermarks={sid: dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)},
            user_id=api.user,
            organization_id=api.org,
            actor=api.client,
        )
        future = ConversationMessageManager(
            clock=lambda: dt.datetime(2100, 6, 1, tzinfo=dt.timezone.utc)
        )
        pruned = await future.prune_distilled_turns(
            user_id=api.user,
            organization_id=api.org,
            actor=api.client,
            retention_days=1,
        )
        assert pruned >= 2
        assert await _rows(api.sm, sid) == []
        assert await _files(api.sm, id=file_id) == [] and not path.exists()
        # The undistilled session keeps its image.
        assert len(await _files(api.sm, id=keep_id)) == 1 and keep_path.exists()

    async def test_prune_keeps_an_image_a_remaining_turn_still_references(self, api):
        sid, file_id, path = await _task_session_with_image(api, "silver")
        [first, _] = await _rows(api.sm, sid)
        # A later (undistilled) turn of the same user referencing the same file.
        other_sid = _sid("k24-ref")
        await ConversationMessageManager().record_turns(
            session_id=other_sid,
            user_id=api.user,
            organization_id=api.org,
            turns=[{"role": "user", "content_parts": first.content_parts}],
            actor=api.client,
            session_tag="task",
            agent_id=api.agent_id,
        )
        await ConversationMessageManager().mark_sessions_distilled(
            session_watermarks={sid: dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)},
            user_id=api.user,
            organization_id=api.org,
            actor=api.client,
        )
        future = ConversationMessageManager(
            clock=lambda: dt.datetime(2100, 6, 1, tzinfo=dt.timezone.utc)
        )
        await future.prune_distilled_turns(
            user_id=api.user,
            organization_id=api.org,
            actor=api.client,
            retention_days=1,
        )
        assert await _rows(api.sm, sid) == []
        assert len(await _files(api.sm, id=file_id)) == 1 and path.exists()

    async def test_user_erasure_deletes_store_images(self, api):
        sid, file_id, path = await _task_session_with_image(api, "coral")
        deleted = await ConversationMessageManager().delete_by_user_id(api.user)
        assert deleted >= 2
        assert await _rows(api.sm, sid) == []
        assert await _files(api.sm, id=file_id) == [] and not path.exists()

    async def test_client_erasure_deletes_store_images(self, api):
        sid, file_id, path = await _task_session_with_image(api, "plum")
        await ConversationMessageManager().delete_by_client_id(api.client)
        assert await _rows(api.sm, sid) == []
        assert await _files(api.sm, id=file_id) == [] and not path.exists()

    async def test_full_user_erasure_also_removes_meta_agent_copies(self, api):
        """DELETE /users/{id}/memories deletes the user's messages, so the meta
        agent's image copies are unreferenced too and go with them."""
        from mirix.services.user_manager import UserManager

        sid = _sid("erase-all")
        resp = await api.post(
            "/memory/add_sync",
            session_id=sid,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes(color="khaki"))},
                        }
                    ],
                }
            ],
        )
        assert resp.status_code == 200, resp.text
        [message] = api.sent[0]["input_messages"]
        meta_id = message.content[1].image_id
        [meta_row] = await _files(api.sm, id=meta_id)
        assert meta_row.user_id == api.user
        await UserManager().delete_memories_by_user_id(api.user)
        assert await _files(api.sm, user_id=api.user) == []
        assert not Path(meta_row.file_path).exists()


# --------------------------------------------------------------------------- #
# Queue carries images; the worker drops task sessions                         #
# --------------------------------------------------------------------------- #


class TestQueue:
    async def _queue(self, monkeypatch, messages, filter_tags=None):
        from mirix.queue import queue_util

        saved = []

        class FakeQueue:
            async def save(self, msg):
                saved.append(msg)

        monkeypatch.setattr(queue_util, "queue", FakeQueue())
        await queue_util.put_messages(
            actor=PydanticClient(
                id="client-q",
                organization_id="org-q",
                name="c",
                write_scope="w",
                read_scopes=["w"],
            ),
            agent_id="agent-q",
            input_messages=messages,
            filter_tags=filter_tags,
        )
        return saved[0]

    async def test_image_parts_round_trip_through_the_queue(self, monkeypatch):
        from mirix.queue.queue_util import (
            deserialize_queue_message,
            serialize_queue_message,
        )
        from mirix.queue.worker import QueueWorker
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.mirix_message_content import (
            CloudFileContent,
            FileContent,
            ImageContent,
            TextContent,
        )

        content = [
            TextContent(text="[USER] look"),
            ImageContent(image_id="file-img", detail="high"),
            ImageContent(image_id="file-img2"),
            FileContent(file_id="file-doc"),
            CloudFileContent(cloud_file_uri="file-cloud"),
            TextContent(text="[ASSISTANT] ok"),
        ]
        queued = await self._queue(
            monkeypatch,
            [
                MessageCreate(
                    role=MessageRole.user, content=content, session_id="sess-q"
                )
            ],
        )
        proto = queued.input_messages[0]
        assert proto.HasField("structured_content") and not proto.HasField(
            "text_content"
        )
        worker = QueueWorker.__new__(QueueWorker)
        for fmt in ("protobuf", "json"):
            restored_msg = deserialize_queue_message(
                serialize_queue_message(queued, fmt), fmt
            )
            restored = worker._convert_proto_message_to_pydantic(
                restored_msg.input_messages[0]
            )
            assert restored.session_id == "sess-q"
            assert [c.model_dump() for c in restored.content] == [
                c.model_dump() for c in content
            ]

    async def test_text_only_lists_keep_the_flat_text_content(self, monkeypatch):
        from mirix.queue.worker import QueueWorker
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate
        from mirix.schemas.mirix_message_content import TextContent

        queued = await self._queue(
            monkeypatch,
            [
                MessageCreate(
                    role=MessageRole.user,
                    content=[TextContent(text="a"), TextContent(text="b")],
                )
            ],
        )
        proto = queued.input_messages[0]
        assert proto.HasField("text_content") and proto.text_content == "a\nb"
        restored = QueueWorker.__new__(QueueWorker)._convert_proto_message_to_pydantic(
            proto
        )
        assert restored.content == "a\nb"

    @pytest.mark.parametrize("tag, dropped", [("task", True), ("conversation", False)])
    async def test_worker_drops_task_tagged_messages(self, monkeypatch, tag, dropped):
        from mirix.queue.worker import QueueWorker
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import MessageCreate

        queued = await self._queue(
            monkeypatch,
            [MessageCreate(role=MessageRole.user, content="hi")],
            filter_tags={"session_tag": tag, "scope": "w"},
        )
        lookups = []

        class Stop(Exception):
            pass

        class FakeClientManager:
            async def get_client_by_id(self, client_id):
                lookups.append(client_id)
                raise Stop()  # past the guard: stop before any real work

        async def never_send(**kwargs):
            raise AssertionError("send_messages must not run in this test")

        worker = QueueWorker.__new__(QueueWorker)
        worker._server = SimpleNamespace(
            client_manager=FakeClientManager(), send_messages=never_send
        )
        await worker._process_message_async(queued)
        assert (lookups == []) is dropped


# --------------------------------------------------------------------------- #
# Client and content-part converter                                           #
# --------------------------------------------------------------------------- #


class TestClientAdd:
    def _client(self, monkeypatch):
        from mirix.client.remote_client import MirixClient

        client = MirixClient(api_key="test-key", base_url="http://test")
        client._meta_agent = SimpleNamespace(id="agent-c")
        calls = []

        async def fake_request(method, endpoint, json=None, params=None, headers=None):
            calls.append({"endpoint": endpoint, "json": json})
            return {"status": "stored"}

        async def no_user(user_id, headers=None):
            return None

        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "_ensure_user_exists", no_user)
        return client, calls

    async def test_session_tag_is_sent(self, monkeypatch):
        client, calls = self._client(monkeypatch)
        await client.add(
            user_id="u", messages=TASK_TURNS, session_id="s-1", session_tag="task"
        )
        assert calls[0]["json"]["session_tag"] == "task"
        assert calls[0]["json"]["session_id"] == "s-1"

    async def test_session_tag_omitted_by_default(self, monkeypatch):
        client, calls = self._client(monkeypatch)
        await client.add(
            user_id="u", messages=TASK_TURNS, session_id="s-1", async_add=False
        )
        assert "session_tag" not in calls[0]["json"]
        assert calls[0]["endpoint"] == "/memory/add_sync"

    async def test_invalid_session_tags_fail_fast(self, monkeypatch):
        client, calls = self._client(monkeypatch)
        with pytest.raises(ValueError):
            await client.add(
                user_id="u", messages=TASK_TURNS, session_id="s-1", session_tag="chat"
            )
        with pytest.raises(ValueError):
            await client.add(user_id="u", messages=TASK_TURNS, session_tag="task")
        assert calls == []


class TestK23SingleImagePart:
    def test_single_image_message_converts_for_openai_and_anthropic(self):
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import Message
        from mirix.schemas.mirix_message_content import ImageContent, ReasoningContent

        msg = Message(
            role=MessageRole.user,
            content=[ImageContent(image_id="file-1", detail="low")],
        )
        openai_dict = msg.to_openai_dict()
        assert openai_dict["role"] == "user"
        assert openai_dict["content"] == [
            {"type": "image_url", "image_id": "file-1", "detail": "low"}
        ]
        anthropic_dict = msg.to_anthropic_dict()
        assert anthropic_dict["content"] == [
            {"type": "image_url", "image_id": "file-1"}
        ]
        # A lone non-media part keeps its previous handling (content=None).
        reasoning = Message(
            role=MessageRole.assistant,
            content=[ReasoningContent(reasoning="r", is_native=True)],
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        )
        assert reasoning.to_openai_dict()["content"] is None


class TestK2Converter:
    async def test_images_are_awaited_and_attributed(self, api):
        from mirix.utils import convert_message_to_mirix_message

        [msg] = await convert_message_to_mirix_message(
            [
                {"type": "text", "text": "hi"},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(_png_bytes(color="pink"))},
                },
                {
                    "type": "image_data",
                    "image_data": {
                        "data": base64.b64encode(_png_bytes(color="pink")).decode()
                    },
                },
                {"type": "image_url", "image_url": {"url": "https://x.test/r.png"}},
            ],
            org_id=api.org,
            user_id=api.user,
        )
        ids = [c.image_id for c in msg.content[1:]]
        assert len(set(ids)) == 3
        for fid in ids:
            [row] = await _files(api.sm, id=fid)
            assert row.user_id == api.user and row.organization_id == api.org

    async def test_file_parts_need_an_org(self, api):
        from mirix.utils import convert_message_to_mirix_message

        with pytest.raises(ValueError, match="org_id"):
            await convert_message_to_mirix_message(
                [{"type": "image_url", "image_url": {"url": "https://x.test/r.png"}}]
            )
        [msg] = await convert_message_to_mirix_message(
            [{"type": "text", "text": "plain"}]
        )
        assert msg.content[0].text == "plain"

    async def test_local_file_parts(self, api, tmp_path):
        from mirix.utils import convert_message_to_mirix_message

        image_file = tmp_path / "shot.png"
        image_file.write_bytes(_png_bytes(color="brown"))
        doc = tmp_path / "notes.txt"
        doc.write_text("notes")
        [msg] = await convert_message_to_mirix_message(
            [
                {"type": "file_uri", "file_uri": f"file://{image_file}"},
                {"type": "file_uri", "file_uri": str(doc)},
                {
                    "type": "google_cloud_file_uri",
                    "google_cloud_file_uri": "gs://bucket/pic.png",
                },
            ],
            org_id=api.org,
            user_id=api.user,
        )
        image, file_part, cloud = msg.content
        [image_row] = await _files(api.sm, id=image.image_id)
        assert Path(image_row.file_path).read_bytes() == image_file.read_bytes()
        assert Path(image_row.file_path) != image_file  # copied into the store
        [doc_row] = await _files(api.sm, id=file_part.file_id)
        assert doc_row.file_path == str(doc) and doc_row.file_size == 5
        [cloud_row] = await _files(api.sm, id=cloud.cloud_file_uri)
        assert cloud_row.google_cloud_url == "gs://bucket/pic.png"


# --------------------------------------------------------------------------- #
# Ingest hardening                                                             #
# --------------------------------------------------------------------------- #


def _noisy_png(size=(64, 64)) -> bytes:
    """A PNG that does not compress away (for byte-cap tests)."""
    import os as _os

    buf = io.BytesIO()
    Image.frombytes("RGB", size, _os.urandom(size[0] * size[1] * 3)).save(buf, "PNG")
    return buf.getvalue()


def _image_message(url_or_parts):
    parts = url_or_parts if isinstance(url_or_parts, list) else [url_or_parts]
    return [{"role": "user", "content": parts}]


async def _insert_file(sm, *, org_id, user_id=None, file_id=None):
    file_id = file_id or f"file-{uuid.uuid4()}"
    async with sm() as db:
        db.add(
            FileORM(
                id=file_id,
                organization_id=org_id,
                user_id=user_id,
                file_name="x.png",
                file_path=None,
                source_url="https://img.example/x.png",
                file_type="image/png",
            )
        )
        await db.commit()
    return file_id


class TestServerLocalFileParts:
    """file_uri and google_cloud_file_uri name a file on the SERVER.

    Over the API they are refused up front — never read, never stat'ed — with
    the same answer whether or not the path exists (no existence oracle)."""

    @pytest.fixture
    def no_disk(self, monkeypatch):
        import mirix.utils as utils

        def forbidden(*args, **kwargs):
            raise AssertionError("the server must not touch a caller-named path")

        monkeypatch.setattr(utils, "_read_local_file", forbidden)
        monkeypatch.setattr(utils, "_local_file_metadata", forbidden)
        monkeypatch.setattr(utils, "_save_image_from_google_cloud_uri", forbidden)

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    @pytest.mark.parametrize("role_bearing", [True, False])
    async def test_file_uri_is_refused_and_never_read(
        self, api, endpoint, role_bearing, tmp_path, no_disk
    ):
        secret = tmp_path / "server_secret.png"
        secret.write_bytes(_png_bytes(color="gold"))
        files_before = len(await _files(api.sm, organization_id=api.org))
        details = []
        for path in (str(secret), str(tmp_path / "does_not_exist.png")):
            part = {"type": "file_uri", "file_uri": path}
            messages = _image_message(part) if role_bearing else [part]
            sid = _sid("fileuri")
            resp = await api.post(endpoint, messages=messages, session_id=sid)
            assert resp.status_code == 422, resp.text
            detail = resp.json()["detail"]
            assert "not accepted" in detail and path not in detail
            details.append(detail)
            assert await _rows(api.sm, sid) == [] and await _session(api.sm, sid) is None
        # An existing and a missing path get the very same answer.
        assert details[0] == details[1]
        assert len(await _files(api.sm, organization_id=api.org)) == files_before
        assert api.sent == [] and api.queued == []

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_google_cloud_file_uri_is_refused(self, api, endpoint, no_disk):
        resp = await api.post(
            endpoint,
            messages=[
                {
                    "type": "google_cloud_file_uri",
                    "google_cloud_file_uri": "gs://internal-bucket/x.png",
                }
            ],
        )
        assert resp.status_code == 422
        assert "not accepted" in resp.json()["detail"]
        assert api.sent == [] and api.queued == []


class TestFileReferenceOwnership:
    """database_* parts may only name files the caller owns."""

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_own_and_org_wide_files_are_accepted(self, api, endpoint):
        own = await _insert_file(api.sm, org_id=api.org, user_id=api.user)
        legacy = await _insert_file(api.sm, org_id=api.org, user_id=None)
        resp = await api.post(
            endpoint,
            messages=[
                {"type": "database_image_id", "image_id": own},
                {"type": "database_image_id", "image_id": legacy},
            ],
        )
        assert resp.status_code == 200, resp.text
        [call] = api.sent if endpoint.endswith("_sync") else api.queued
        [message] = call["input_messages"]
        assert [c.image_id for c in message.content] == [own, legacy]

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_foreign_and_missing_files_get_the_same_422(self, api, endpoint):
        other_user = f"user-{uuid.uuid4().hex[:8]}"
        other_org = f"org-{uuid.uuid4().hex[:8]}"
        async with api.sm() as db:
            db.add(OrganizationORM(id=other_org, name=other_org))
            db.add(
                UserORM(
                    id=other_user,
                    name=other_user,
                    organization_id=api.org,
                    status="active",
                    timezone="UTC",
                )
            )
            await db.commit()
        foreign_user = await _insert_file(api.sm, org_id=api.org, user_id=other_user)
        foreign_org = await _insert_file(api.sm, org_id=other_org, user_id=None)
        missing = f"file-{uuid.uuid4()}"
        answers = set()
        for file_id in (foreign_user, foreign_org, missing):
            sid = _sid("fref")
            resp = await api.post(
                endpoint,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            "see",
                            {"type": "database_image_id", "image_id": file_id},
                        ],
                    }
                ],
                session_id=sid,
            )
            assert resp.status_code == 422, resp.text
            answers.add(resp.json()["detail"].replace(file_id, "<id>"))
            assert await _rows(api.sm, sid) == []
        assert len(answers) == 1
        assert api.sent == [] and api.queued == []


class TestNothingWrittenBeforeA422:
    """Every 422 is decided before the store write."""

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_image_plus_refused_part_writes_nothing(self, api, endpoint):
        files_before = len(await _files(api.sm, organization_id=api.org))
        sid = _sid("f33")
        resp = await api.post(
            endpoint,
            messages=_image_message(
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(_png_bytes(color="teal"))},
                    },
                    {"type": "database_image_id", "image_id": "file-unknown"},
                ]
            ),
            session_id=sid,
        )
        assert resp.status_code == 422
        assert await _rows(api.sm, sid) == [] and await _session(api.sm, sid) is None
        assert len(await _files(api.sm, organization_id=api.org)) == files_before

    async def test_a_failed_queue_put_discards_the_meta_copies(self, api, monkeypatch):
        async def broken_put(**kwargs):
            raise RuntimeError("queue down")

        monkeypatch.setattr(rest_api, "put_messages", broken_put)
        sid = _sid("f33-queue")
        resp = await api.post(
            "/memory/add",
            messages=_image_message(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(_png_bytes(color="navy"))},
                }
            ),
            session_id=sid,
        )
        assert resp.status_code == 500
        # The conversation store kept its turn (additive) and its own copy; the
        # meta agent's copy — never queued, never referenced — is gone.
        [row] = await _rows(api.sm, sid)
        store_file = row.content_parts[0]["file_id"]
        user_files = await _files(api.sm, user_id=api.user)
        assert store_file in {f.id for f in user_files}
        sha = row.content_parts[0]["sha256"]
        same_bytes = [f for f in user_files if f.file_path and sha in f.file_path]
        assert [f.id for f in same_bytes] == [store_file]


class TestImageBounds:
    """Only provider-supported formats, bounded sizes, a valid detail."""

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    @pytest.mark.parametrize("fmt,mime", [("TIFF", "image/tiff"), ("BMP", "image/bmp")])
    async def test_unsupported_formats_are_422(self, api, endpoint, fmt, mime):
        resp = await api.post(
            endpoint,
            messages=_image_message(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(_png_bytes(fmt=fmt), mime)},
                }
            ),
            session_id=_sid("fmt"),
        )
        assert resp.status_code == 422
        assert "unsupported image format" in resp.json()["detail"]
        assert api.sent == [] and api.queued == []

    async def test_an_oversized_image_is_422(self, api, monkeypatch):
        import mirix.constants as constants

        monkeypatch.setattr(constants, "MAX_IMAGE_BYTES", 1000)
        resp = await api.post(
            "/memory/add_sync",
            messages=_image_message(
                {"type": "image_url", "image_url": {"url": _data_url(_noisy_png())}}
            ),
        )
        assert resp.status_code == 422
        assert "limit is 1000 bytes" in resp.json()["detail"]

    async def test_the_request_total_is_capped(self, api, monkeypatch):
        import mirix.constants as constants

        one = _png_bytes(color="red")
        monkeypatch.setattr(constants, "MAX_IMAGE_BYTES_PER_REQUEST", len(one) + 10)
        resp = await api.post(
            "/memory/add_sync",
            messages=_image_message(
                [
                    {"type": "image_url", "image_url": {"url": _data_url(one)}},
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url(_png_bytes(color="blue"))},
                    },
                ]
            ),
        )
        assert resp.status_code == 422
        assert "in total" in resp.json()["detail"]
        assert api.sent == []

    @pytest.mark.parametrize("detail", ["ultra", 3])
    async def test_an_invalid_detail_is_422(self, api, detail):
        resp = await api.post(
            "/memory/add_sync",
            messages=_image_message(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(_png_bytes()), "detail": detail},
                }
            ),
        )
        assert resp.status_code == 422
        assert "detail must be one of" in resp.json()["detail"]


class TestMetaAgentRequired:
    """The add endpoints only feed META memory agents."""

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_a_sub_agent_is_refused(self, api, endpoint):
        sid = _sid("sub")
        resp = await api.post(
            endpoint,
            meta_agent_id=api.sub_agent_id,
            messages=TASK_TURNS,
            session_id=sid,
            session_tag="task",
        )
        assert resp.status_code == 400
        assert "not a meta_memory_agent" in resp.json()["detail"]
        assert await _session(api.sm, sid) is None


class TestSessionOwnership:
    """A session belongs to one meta agent; another client's write
    to it is a 409 with nothing stored (and never adopts its tag)."""

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_another_meta_agents_write_is_409(self, api, endpoint):
        sid = _sid("owned")
        first = await api.post(
            endpoint, messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert first.status_code == 200, first.text
        # Client B, conversation tag: without the check it would have been
        # "stored" into A's task session (A's tag adopted).
        second = await api.post_as_other(
            endpoint,
            messages=[{"role": "user", "content": "client B's turn"}],
            session_id=sid,
        )
        assert second.status_code == 409, second.text
        assert "another meta agent" in second.json()["detail"]
        rows = await _rows(api.sm, sid)
        assert len(rows) == 4 and all("client B" not in r.content for r in rows)
        session = await _session(api.sm, sid)
        assert session.agent_id == api.agent_id and session.message_count == 4
        assert api.sent == [] and api.queued == []

    async def test_legacy_session_written_by_another_client_is_409(self, api):
        sid = _sid("legacy")
        async with api.sm() as db:
            db.add(
                SessionORM(
                    id=f"convsess-{uuid.uuid4()}",
                    organization_id=api.org,
                    user_id=api.user,
                    agent_id=None,
                    session_id=sid,
                    session_tag="task",
                    first_message_at=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
                    last_message_at=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
                    message_count=1,
                )
            )
            row = MessageORM(
                session_id=sid,
                user_id=api.user,
                organization_id=api.org,
                role="user",
                content="written by client A before the upgrade",
                session_tag="task",
                created_at=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
            )
            row._set_created_and_updated_by_fields(api.client.id)
            db.add(row)
            await db.commit()
        other = await api.post_as_other(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert other.status_code == 409
        assert (await _session(api.sm, sid)).agent_id is None
        mine = await api.post(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert mine.status_code == 200, mine.text
        assert (await _session(api.sm, sid)).agent_id == api.agent_id


class TestOrphanedSessionAdoption:
    """A client that deletes and re-creates its meta agent gets its
    sessions back. The old owner no longer exists, so the session is orphaned:
    the client's new meta agent adopts it (on write, on close, in the sweep)
    instead of a 409 "belongs to another meta agent" forever — and another
    client still cannot take it."""

    async def _recreate_meta_agent(self, api, *, hard: bool) -> str:
        new_agent_id = f"agent-{uuid.uuid4()}"
        async with api.sm() as db:
            if hard:
                await db.execute(delete(AgentORM).where(AgentORM.id == api.agent_id))
            else:
                await db.execute(
                    update(AgentORM)
                    .where(AgentORM.id == api.agent_id)
                    .values(is_deleted=True)
                )
            agent = AgentORM(
                id=new_agent_id,
                organization_id=api.org,
                agent_type="meta_memory_agent",
                name="meta-recreated",
            )
            agent._created_by_id = api.client.id
            db.add(agent)
            await db.commit()
        manager = rest_api.get_server().agent_manager
        original = manager.get_agent_by_id

        async def get_agent_by_id(aid, actor):
            if aid != new_agent_id:
                return await original(aid, actor)
            assert actor.id == api.client.id
            return SimpleNamespace(
                id=aid,
                agent_type=AgentType.meta_memory_agent,
                llm_config=SimpleNamespace(
                    model="m", context_window=api.context_window
                ),
            )

        manager.get_agent_by_id = get_agent_by_id
        return new_agent_id

    async def _write_with_old_agent(self, api, endpoint="/memory/add_sync"):
        sid = _sid("orphan")
        first = await api.post(
            endpoint, messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert first.status_code == 200, first.text
        return sid

    @pytest.mark.parametrize("hard", [True, False], ids=["hard-delete", "soft-delete"])
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    async def test_the_new_meta_agent_adopts_on_write(self, api, endpoint, hard):
        sid = await self._write_with_old_agent(api, endpoint)
        new_agent = await self._recreate_meta_agent(api, hard=hard)

        again = await api.post(
            endpoint,
            meta_agent_id=new_agent,
            messages=TASK_TURNS,
            session_id=sid,
            session_tag="task",
        )

        assert again.status_code == 200, again.text
        session = await _session(api.sm, sid)
        assert session.agent_id == new_agent
        assert session.message_count == 8 and session.session_tag == "task"
        assert len(await _rows(api.sm, sid)) == 8

    async def test_another_client_cannot_adopt_an_orphan(self, api):
        sid = await self._write_with_old_agent(api)
        await self._recreate_meta_agent(api, hard=True)

        other = await api.post_as_other(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )

        assert other.status_code == 409, other.text
        session = await _session(api.sm, sid)
        assert session.agent_id == api.agent_id  # still orphaned, not taken
        assert session.message_count == 4

    async def test_a_live_owner_still_conflicts(self, api):
        """Only a deleted owner makes a session adoptable."""
        sid = await self._write_with_old_agent(api)
        other = await api.post_as_other(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert other.status_code == 409
        assert (await _session(api.sm, sid)).agent_id == api.agent_id

    async def test_the_sweep_reattributes_an_orphan(self, api):
        sid = await self._write_with_old_agent(api)
        new_agent = await self._recreate_meta_agent(api, hard=True)
        sessions = ConversationSessionManager(session_maker=api.sm)

        # A claim of another scope never adopts into itself.
        assert (
            await sessions.adopt_unattributed_sessions(
                organization_id=api.org, user_id=api.user, agent_id=api.other_agent_id
            )
            == 0
        )
        # (Scoped to this test's organization: the module database is shared.)
        assert await sessions.adopt_unattributed_sessions(organization_id=api.org) == 1
        assert (await _session(api.sm, sid)).agent_id == new_agent
        assert (
            await sessions.adopt_unattributed_sessions(organization_id=api.org) == 0
        )  # settled

    async def test_an_orphan_without_a_successor_stays_unlearnable(self, api):
        sid = await self._write_with_old_agent(api)
        async with api.sm() as db:
            await db.execute(delete(AgentORM).where(AgentORM.id == api.agent_id))
            await db.commit()
        sessions = ConversationSessionManager(session_maker=api.sm)
        assert await sessions.adopt_unattributed_sessions(organization_id=api.org) == 0
        assert (await _session(api.sm, sid)).agent_id == api.agent_id
        assert not await sessions.agent_is_live(api.agent_id)

    async def test_close_adopts_an_orphan(self, api):
        sid = await self._write_with_old_agent(api)
        new_agent = await self._recreate_meta_agent(api, hard=True)

        resp = await api.http.post(
            f"/memory/sessions/{sid}/close",
            json={"user_id": api.user},
            headers={"x-client-id": api.client.id, "x-org-id": api.org},
        )

        assert resp.status_code == 200, resp.text
        session = await _session(api.sm, sid)
        assert session.agent_id == new_agent and session.closed_at is not None


class TestOrphanOwnershipAndLearning:
    """An orphan whose rows record no writer (backfilled rows) is never
    attributed by the organization-wide fallback — only to its owning client
    (the deleted agent's creator, else the session row's creator). Its
    pending learning moves with it."""

    _recreate = TestOrphanedSessionAdoption._recreate_meta_agent

    async def _orphan_without_writers(self, api, *, hard, keep_session_creator):
        sid = _sid("orphan-nw")
        first = await api.post(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert first.status_code == 200, first.text
        async with api.sm() as db:
            await db.execute(
                update(MessageORM)
                .where(MessageORM.session_id == sid)
                .values(_created_by_id=None)
            )
            if not keep_session_creator:
                await db.execute(
                    update(SessionORM)
                    .where(SessionORM.session_id == sid)
                    .values(_created_by_id=None)
                )
            await db.commit()
        return sid

    async def _delete_old_agent(self, api, *, hard):
        async with api.sm() as db:
            if hard:
                await db.execute(delete(AgentORM).where(AgentORM.id == api.agent_id))
            else:
                await db.execute(
                    update(AgentORM)
                    .where(AgentORM.id == api.agent_id)
                    .values(is_deleted=True)
                )
            await db.commit()

    async def test_an_orphan_with_no_known_owner_goes_to_nobody(self, api):
        sid = await self._orphan_without_writers(
            api, hard=True, keep_session_creator=False
        )
        # The other client's meta agent is now the organization's ONLY live
        # one — the legacy fallback would hand it this transcript.
        await self._delete_old_agent(api, hard=True)
        sessions = ConversationSessionManager(session_maker=api.sm)
        assert await sessions.adopt_unattributed_sessions(organization_id=api.org) == 0
        other = await api.post_as_other(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert other.status_code == 409, other.text
        assert (await _session(api.sm, sid)).agent_id == api.agent_id

    @pytest.mark.parametrize("hard", [True, False], ids=["hard-delete", "soft-delete"])
    async def test_an_orphan_goes_only_to_its_owning_client(self, api, hard):
        # Hard delete: the agent row is gone; the session row's creator names
        # the owner. Soft delete: the deleted agent's creator does.
        sid = await self._orphan_without_writers(
            api, hard=hard, keep_session_creator=hard
        )
        new_agent = await self._recreate(api, hard=hard)
        other = await api.post_as_other(
            "/memory/add_sync", messages=TASK_TURNS, session_id=sid, session_tag="task"
        )
        assert other.status_code == 409, other.text
        sessions = ConversationSessionManager(session_maker=api.sm)
        assert await sessions.adopt_unattributed_sessions(organization_id=api.org) == 1
        assert (await _session(api.sm, sid)).agent_id == new_agent

    async def test_the_owning_client_adopts_it_on_write(self, api):
        sid = await self._orphan_without_writers(
            api, hard=True, keep_session_creator=True
        )
        new_agent = await self._recreate(api, hard=True)
        mine = await api.post(
            "/memory/add_sync",
            meta_agent_id=new_agent,
            messages=TASK_TURNS,
            session_id=sid,
            session_tag="task",
        )
        assert mine.status_code == 200, mine.text
        assert (await _session(api.sm, sid)).agent_id == new_agent

    async def _experience(self, api, sid, *, status):
        from mirix.orm.skill_experience import SkillExperience as ExperienceORM

        exp_id = f"sexp-{uuid.uuid4().hex[:8]}"
        async with api.sm() as db:
            db.add(
                ExperienceORM(
                    id=exp_id,
                    organization_id=api.org,
                    user_id=api.user,
                    agent_id=api.agent_id,
                    session_id=sid,
                    experience_type="worth_learning",
                    title="t",
                    content="c",
                    evidence="",
                    status=status,
                )
            )
            await db.commit()
        return exp_id

    async def _experience_agents(self, api, *ids):
        from mirix.orm.skill_experience import SkillExperience as ExperienceORM

        async with api.sm() as db:
            rows = (
                await db.execute(
                    select(ExperienceORM.id, ExperienceORM.agent_id).where(
                        ExperienceORM.id.in_(ids)
                    )
                )
            ).all()
        return dict(rows)

    @pytest.mark.parametrize("path", ["write", "sweep"])
    async def test_pending_experiences_move_with_an_adopted_orphan(self, api, path):
        sid = await TestOrphanedSessionAdoption()._write_with_old_agent(api)
        pending = await self._experience(api, sid, status="pending")
        consumed = await self._experience(api, sid, status="consumed")
        new_agent = await self._recreate(api, hard=False)
        if path == "write":
            again = await api.post(
                "/memory/add_sync",
                meta_agent_id=new_agent,
                messages=TASK_TURNS,
                session_id=sid,
                session_tag="task",
            )
            assert again.status_code == 200, again.text
        else:
            sessions = ConversationSessionManager(session_maker=api.sm)
            assert (
                await sessions.adopt_unattributed_sessions(organization_id=api.org)
                == 1
            )
        assert await self._experience_agents(api, pending, consumed) == {
            pending: new_agent,  # the new scope's curator will see it
            consumed: api.agent_id,  # history stays as it was
        }

    async def test_a_hard_deleted_meta_agent_releases_its_pending_learning(
        self, api, monkeypatch
    ):
        import datetime as _dt

        import mirix.server.server as server_module
        from mirix.services.agent_manager import AgentManager

        sid = await TestOrphanedSessionAdoption()._write_with_old_agent(api)
        learned = await TestOrphanedSessionAdoption()._write_with_old_agent(api)
        async with api.sm() as db:
            await db.execute(
                update(MessageORM)
                .where(MessageORM.session_id.in_([sid, learned]))
                .values(distilled_at=_dt.datetime.now(_dt.timezone.utc))
            )
            await db.commit()
        await self._experience(api, sid, status="pending")
        await self._experience(api, learned, status="consumed")
        monkeypatch.setattr(server_module, "db_context", api.sm)
        await AgentManager().delete_agent(api.agent_id, actor=api.client)

        # The session whose learning was still pending is distilled again by
        # its next owner; a fully curated one is not.
        assert all(r.distilled_at is None for r in await _rows(api.sm, sid))
        assert all(r.distilled_at is not None for r in await _rows(api.sm, learned))
        async with api.sm() as db:
            gone = (
                await db.execute(select(AgentORM.id).where(AgentORM.id == api.agent_id))
            ).first()
        assert gone is None


class TestMessageImageRelease:
    """The meta agent's image copies go with the agent messages that
    referenced them (message cleanup, client erasure) unless one still does."""

    async def _meta_copy(self, api, color):
        resp = await api.post(
            "/memory/add_sync",
            messages=_image_message(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(_png_bytes(color=color))},
                }
            ),
        )
        assert resp.status_code == 200, resp.text
        [message] = api.sent[-1]["input_messages"]
        image_id = message.content[1].image_id
        [row] = await _files(api.sm, id=image_id)
        return image_id, Path(row.file_path)

    async def _agent_message(self, api, agent_id, image_id):
        from mirix.orm.message import Message as AgentMessageORM
        from mirix.schemas.mirix_message_content import ImageContent, TextContent

        message_id = f"message-{uuid.uuid4()}"
        async with api.sm() as db:
            db.add(
                AgentMessageORM(
                    id=message_id,
                    role="user",
                    content=[TextContent(text="[USER]"), ImageContent(image_id=image_id)],
                    organization_id=api.org,
                    user_id=api.user,
                    agent_id=agent_id,
                    client_id=api.client.id,
                    tool_calls=[],
                )
            )
            await db.commit()
        return message_id

    async def _ensure_agent(self, api):
        async with api.sm() as db:
            agent = (
                await db.execute(select(AgentORM).where(AgentORM.id == api.agent_id))
            ).scalar_one()
            return agent

    async def test_client_erasure_releases_meta_copies(self, api):
        from mirix.services.message_manager import MessageManager

        image_id, path = await self._meta_copy(api, "salmon")
        await self._agent_message(api, api.agent_id, image_id)
        assert path.exists()
        await MessageManager().delete_by_client_id(api.client)
        assert await _files(api.sm, id=image_id) == [] and not path.exists()

    async def test_an_image_another_message_still_references_is_kept(self, api):
        from mirix.services.message_manager import MessageManager

        image_id, path = await self._meta_copy(api, "wheat")
        first = await self._agent_message(api, api.agent_id, image_id)
        await self._agent_message(api, api.sub_agent_id, image_id)  # forwarded
        await MessageManager().delete_message_by_id(first, actor=api.client)
        assert len(await _files(api.sm, id=image_id)) == 1 and path.exists()

    async def _turn_copy(self, api, color):
        """A task turn's stored image: (file id, path) of the conversation copy."""
        from mirix.orm.conversation_message import ConversationMessage

        sid = _sid("turn-img")
        resp = await api.post(
            "/memory/add_sync",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _data_url(_png_bytes(color=color))},
                        },
                    ],
                }
            ],
            session_id=sid,
            session_tag="task",
        )
        assert resp.status_code == 200, resp.text
        async with api.sm() as db:
            [parts] = (
                await db.execute(
                    select(ConversationMessage.content_parts).where(
                        ConversationMessage.session_id == sid
                    )
                )
            ).scalars().all()
        [file_id] = [p["file_id"] for p in parts if p.get("type") == "image_ref"]
        [row] = await _files(api.sm, id=file_id)
        return file_id, Path(row.file_path)

    async def test_a_message_never_releases_a_stored_turns_image(self, api):
        """A caller may hand the meta agent a stored image by id; deleting that
        message must not pull the image out from under the transcript."""
        from mirix.services.message_manager import MessageManager

        file_id, path = await self._turn_copy(api, "khaki")
        message_id = await self._agent_message(api, api.agent_id, file_id)
        await MessageManager().delete_message_by_id(message_id, actor=api.client)
        assert len(await _files(api.sm, id=file_id)) == 1 and path.exists()

    async def test_a_turn_never_releases_an_image_a_message_references(self, api):
        """The converse: erasing the client's turns keeps a stored image the
        agent's message history still points at."""
        from mirix.services.conversation_message_manager import (
            ConversationMessageManager,
        )

        file_id, path = await self._turn_copy(api, "navy")
        await self._agent_message(api, api.agent_id, file_id)
        await ConversationMessageManager(session_maker=api.sm).delete_by_client_id(
            api.client
        )
        assert len(await _files(api.sm, id=file_id)) == 1 and path.exists()

    async def test_detached_message_cleanup_releases_meta_copies(self, api):
        from mirix.services.message_manager import MessageManager

        image_id, path = await self._meta_copy(api, "orchid")
        message_id = await self._agent_message(api, api.agent_id, image_id)
        agent = await self._ensure_agent(api)
        assert message_id not in (agent.message_ids or [])  # detached
        await MessageManager().delete_detached_messages_for_agent(
            agent_id=api.agent_id, actor=api.client
        )
        assert await _files(api.sm, id=image_id) == [] and not path.exists()
