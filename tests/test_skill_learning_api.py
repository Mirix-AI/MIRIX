"""REST endpoints and client methods for sessions and skill learning.

The FastAPI app is driven in-process through httpx's ASGI transport (no
lifespan, no server, no network). Authentication is replaced by a switchable
fake returning one of two clients, the database is a throwaway SQLite file, and
scheduling a learning run is recorded instead of executed. Covers:

* POST /memory/sessions/{session_id}/close — idempotent, 404, 422, client
  isolation, returns the learning run the close started;
* POST /memory/skills/flush — fires below the threshold, null when nothing is
  pending, close_open_sessions;
* GET /memory/learning_runs[/{id}] — status filter, client isolation, 404;
* POST /memory/auto_dream procedural with wait=false — returns run_ids at once;
* the add-path ingestion seam — tag-mismatch warning, trigger after a task write;
* the MirixClient methods' request shapes.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix.client.remote_client import MirixClient
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.server import rest_api
from mirix.services import skill_learning_trigger
from mirix.services.conversation_message_manager import ConversationMessageManager
from mirix.services.conversation_session_manager import ConversationSessionManager

pytestmark = pytest.mark.asyncio(loop_scope="module")

# The real authentication, kept before the fixtures replace it.
_REAL_AUTH = rest_api.get_client_from_jwt_or_api_key


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_learning_api") / "test.db"
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


async def _tenant(sm, org_id):
    """A client with its own meta agent and user."""
    client_id = f"client-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    agent_id = f"agent-{uuid.uuid4()}"
    async with sm() as db:
        db.add(
            ClientORM(
                id=client_id,
                organization_id=org_id,
                name=client_id,
                status="active",
                write_scope="test",
                read_scopes=["test"],
            )
        )
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
    return SimpleNamespace(
        client=PydanticClient(
            id=client_id,
            organization_id=org_id,
            name=client_id,
            write_scope="test",
            read_scopes=["test"],
        ),
        user=PydanticUser(
            id=user_id, name=user_id, organization_id=org_id, timezone="UTC"
        ),
        meta=SimpleNamespace(
            id=agent_id,
            agent_type=AgentType.meta_memory_agent,
            llm_config=SimpleNamespace(model="m"),
        ),
    )


@pytest_asyncio.fixture(loop_scope="module")
async def api(sm, monkeypatch):
    import mirix.server.server as server_module

    org_id = f"org-{uuid.uuid4().hex[:8]}"
    foreign_org_id = f"org-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org_id, name=org_id))
        db.add(OrganizationORM(id=foreign_org_id, name=foreign_org_id))
        await db.commit()
    a = await _tenant(sm, org_id)
    b = await _tenant(sm, org_id)
    # A client (with its own meta agent and user) of ANOTHER organization.
    c = await _tenant(sm, foreign_org_id)
    state = SimpleNamespace(
        current=a, scheduled=[], scheduled_context=[], a=a, b=b, c=c
    )

    async def fake_auth(authorization=None, http_request=None):
        return state.current.client, "api_key"

    class FakeAgentManager:
        async def get_agent_by_id(self, agent_id, actor):
            for tenant in (a, b, c):
                if tenant.meta.id == agent_id and tenant.client.id == actor.id:
                    return tenant.meta
            raise LookupError(agent_id)

        async def list_agents(self, actor):
            return [t.meta for t in (a, b, c) if t.client.id == actor.id]

    class FakeUserManager:
        async def get_user_by_id(self, user_id=None):
            for tenant in (a, b, c):
                if tenant.user.id == user_id:
                    return tenant.user
            raise NoResultFound(user_id)

    fake_server = SimpleNamespace(
        agent_manager=FakeAgentManager(), user_manager=FakeUserManager()
    )
    state.server = fake_server

    def fake_schedule(run_id, **context):
        state.scheduled.append(run_id)
        state.scheduled_context.append(context)

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", fake_auth)
    monkeypatch.setattr(rest_api, "get_server", lambda: fake_server)
    monkeypatch.setattr(skill_learning_trigger, "schedule_learning_run", fake_schedule)

    transport = httpx.ASGITransport(app=rest_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        state.http = http
        yield state


async def _write(tenant, sid, *, tag="task"):
    return await ConversationMessageManager().record_turns(
        session_id=sid,
        user_id=tenant.user.id,
        organization_id=tenant.client.organization_id,
        turns=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        actor=tenant.client,
        session_tag=tag,
        agent_id=tenant.meta.id,
    )


async def _ended(tenant, count, prefix):
    sids = []
    for i in range(count):
        sid = f"{prefix}-{i}-{uuid.uuid4().hex[:6]}"
        await _write(tenant, sid)
        await ConversationSessionManager().close_session(
            organization_id=tenant.client.organization_id,
            user_id=tenant.user.id,
            session_id=sid,
        )
        sids.append(sid)
    return sids


class TestCloseEndpoint:
    async def test_close_then_close_again(self, api):
        await _write(api.a, "close-api")
        first = await api.http.post(
            "/memory/sessions/close-api/close", json={"user_id": api.a.user.id}
        )
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["session_id"] == "close-api"
        assert body["already_closed"] is False
        assert body["close_reason"] == "explicit" and body["closed_at"]
        assert body["learning_run_id"] is None  # 1 < SKILL_TRIGGER_SESSION_THRESHOLD
        second = await api.http.post(
            "/memory/sessions/close-api/close", json={"user_id": api.a.user.id}
        )
        assert second.json()["already_closed"] is True
        assert second.json()["closed_at"] == body["closed_at"]

    async def test_the_close_that_completes_the_threshold_starts_a_run(self, api):
        earlier = await _ended(api.a, 4, "thr")
        await _write(api.a, "fifth")
        response = await api.http.post(
            "/memory/sessions/fifth/close", json={"user_id": api.a.user.id}
        )
        run_id = response.json()["learning_run_id"]
        assert run_id and api.scheduled == [run_id]
        # Closing again never triggers a second run.
        again = await api.http.post(
            "/memory/sessions/fifth/close", json={"user_id": api.a.user.id}
        )
        assert again.json()["learning_run_id"] is None
        run = (await api.http.get(f"/memory/learning_runs/{run_id}")).json()
        assert run["trigger"] == "close"
        assert sorted(run["session_ids"]) == sorted(earlier + ["fifth"])

    async def test_unknown_session_is_404(self, api):
        response = await api.http.post(
            "/memory/sessions/never-seen/close", json={"user_id": api.a.user.id}
        )
        assert response.status_code == 404

    async def test_malformed_session_id_is_422(self, api):
        response = await api.http.post(
            "/memory/sessions/bad%20id/close", json={"user_id": api.a.user.id}
        )
        assert response.status_code == 422

    async def test_another_clients_session_is_404(self, api):
        await _write(api.a, "a-owned")
        api.current = api.b
        response = await api.http.post(
            "/memory/sessions/a-owned/close", json={"user_id": api.a.user.id}
        )
        assert response.status_code == 404
        api.current = api.a
        session = await ConversationSessionManager().get_session(
            organization_id=api.a.client.organization_id,
            user_id=api.a.user.id,
            session_id="a-owned",
        )
        assert session.closed_at is None


class TestFlushEndpoint:
    async def test_flush_with_three_sessions_fires(self, api):
        sids = await _ended(api.a, 3, "flush")
        response = await api.http.post(
            "/memory/skills/flush", json={"user_id": api.a.user.id}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_count"] == 3 and body["learning_run_id"]
        assert api.scheduled == [body["learning_run_id"]]
        run = (
            await api.http.get(f"/memory/learning_runs/{body['learning_run_id']}")
        ).json()
        assert run["trigger"] == "flush" and sorted(run["session_ids"]) == sorted(sids)

    async def test_flush_with_nothing_pending_returns_null(self, api):
        response = await api.http.post(
            "/memory/skills/flush", json={"user_id": api.a.user.id}
        )
        assert response.json() == {
            "learning_run_id": None,
            "session_count": 0,
            "closed_session_ids": [],
            "pending_experiences": 0,
        }

    async def test_close_open_sessions_first(self, api):
        await _write(api.a, "open-1")
        response = await api.http.post(
            "/memory/skills/flush",
            json={"user_id": api.a.user.id, "close_open_sessions": True},
        )
        body = response.json()
        assert body["closed_session_ids"] == ["open-1"]
        assert body["session_count"] == 1 and body["learning_run_id"]

    async def test_unknown_meta_agent_is_404(self, api):
        response = await api.http.post(
            "/memory/skills/flush",
            json={"user_id": api.a.user.id, "meta_agent_id": api.b.meta.id},
        )
        assert response.status_code == 404


class TestLearningRunEndpoints:
    async def test_list_filter_and_isolation(self, api):
        await _ended(api.a, 2, "runs")
        run_id = (
            await api.http.post("/memory/skills/flush", json={"user_id": api.a.user.id})
        ).json()["learning_run_id"]

        listed = await api.http.get(
            "/memory/learning_runs", params={"user_id": api.a.user.id}
        )
        assert listed.status_code == 200
        assert [r["id"] for r in listed.json()["learning_runs"]] == [run_id]
        active = await api.http.get(
            "/memory/learning_runs", params={"status": "active"}
        )
        assert [r["id"] for r in active.json()["learning_runs"]] == [run_id]
        done = await api.http.get(
            "/memory/learning_runs", params={"status": "succeeded"}
        )
        assert done.json()["learning_runs"] == []
        bad = await api.http.get("/memory/learning_runs", params={"status": "bogus"})
        assert bad.status_code == 400
        too_many = await api.http.get("/memory/learning_runs", params={"limit": 0})
        assert too_many.status_code == 422

        api.current = api.b
        assert (await api.http.get("/memory/learning_runs")).json()[
            "learning_runs"
        ] == []
        assert (
            await api.http.get(f"/memory/learning_runs/{run_id}")
        ).status_code == 404
        api.current = api.a
        one = await api.http.get(f"/memory/learning_runs/{run_id}")
        assert one.status_code == 200
        assert one.json()["status"] == "queued" and one.json()["id"] == run_id

    async def test_unknown_run_is_404(self, api):
        assert (
            await api.http.get("/memory/learning_runs/slrun-nope")
        ).status_code == 404


class TestAutoDreamProcedural:
    async def test_wait_false_returns_run_ids_immediately(self, api):
        await _ended(api.a, 1, "dream")
        response = await api.http.post(
            "/memory/auto_dream",
            params={"user_id": api.a.user.id},
            json={"mode": "procedural", "wait": False},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["run_ids"]) == 1 and body["dry_run"] is False
        assert api.scheduled == body["run_ids"]
        run = (await api.http.get(f"/memory/learning_runs/{body['run_ids'][0]}")).json()
        assert run["trigger"] == "manual" and run["status"] == "queued"


class TestIngestionSeam:
    async def test_tag_mismatch_warning_and_trigger_after_task_write(
        self, api, monkeypatch
    ):
        checks = []

        async def fake_trigger(**kwargs):
            checks.append(kwargs)
            return skill_learning_trigger.ClaimFireResult(fired=True, run_id="slrun-t")

        monkeypatch.setattr(
            skill_learning_trigger, "trigger_skill_learning", fake_trigger
        )

        def request(tag):
            return SimpleNamespace(
                session_id="seam",
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
                session_tag=tag,
                meta_agent_id=api.a.meta.id,
            )

        first = await rest_api._ingest_session_turns(
            request("task"), api.a.client, api.a.user.id
        )
        assert first.effective_tag == "task" and first.warnings == []
        assert first.learning_run_id == "slrun-t"
        assert checks[0]["trigger"] == "threshold"
        assert checks[0]["agent_id"] == api.a.meta.id

        second = await rest_api._ingest_session_turns(
            request("conversation"), api.a.client, api.a.user.id
        )
        assert second.effective_tag == "task"
        assert second.warnings and second.warnings[0].startswith("session_tag mismatch")
        assert rest_api._resolved_session_tag(request("conversation"), second) == "task"

    async def test_conversation_writes_do_not_trigger(self, api, monkeypatch):
        checks = []

        async def fake_trigger(**kwargs):
            checks.append(kwargs)

        monkeypatch.setattr(
            skill_learning_trigger, "trigger_skill_learning", fake_trigger
        )
        result = await rest_api._ingest_session_turns(
            SimpleNamespace(
                session_id="chat-seam",
                messages=[{"role": "user", "content": "hi"}],
                session_tag=None,
                meta_agent_id=api.a.meta.id,
            ),
            api.a.client,
            api.a.user.id,
        )
        assert result.effective_tag == "conversation"
        assert checks == []

    async def test_trigger_failure_does_not_fail_the_write(self, api, monkeypatch):
        async def broken_trigger(**kwargs):
            raise RuntimeError("lock timeout")

        monkeypatch.setattr(
            skill_learning_trigger, "trigger_skill_learning", broken_trigger
        )
        result = await rest_api._ingest_session_turns(
            SimpleNamespace(
                session_id="robust-seam",
                messages=[{"role": "user", "content": "q"}],
                session_tag="task",
                meta_agent_id=api.a.meta.id,
            ),
            api.a.client,
            api.a.user.id,
        )
        assert result.effective_tag == "task" and result.learning_run_id is None


# --------------------------------------------------------------------------- #
# Client methods                                                               #
# --------------------------------------------------------------------------- #


def _recording_client(monkeypatch, response):
    client = MirixClient(api_key="test-key", base_url="http://test")
    calls = []

    async def fake_request(method, endpoint, json=None, params=None, headers=None):
        calls.append(
            {"method": method, "endpoint": endpoint, "json": json, "params": params}
        )
        return response

    monkeypatch.setattr(client, "_request", fake_request)
    return client, calls


class TestClientMethods:
    async def test_close_session(self, monkeypatch):
        client, calls = _recording_client(monkeypatch, {"already_closed": False})
        try:
            await client.close_session("sess-1", user_id="u1")
            await client.close_session("sess-2")
            with pytest.raises(ValueError):
                await client.close_session("bad id")
        finally:
            await client.close()
        assert calls == [
            {
                "method": "POST",
                "endpoint": "/memory/sessions/sess-1/close",
                "json": {"user_id": "u1"},
                "params": None,
            },
            {
                "method": "POST",
                "endpoint": "/memory/sessions/sess-2/close",
                "json": {},
                "params": None,
            },
        ]

    async def test_flush_skills_sends_the_meta_agent(self, monkeypatch):
        client, calls = _recording_client(monkeypatch, {"learning_run_id": None})
        client._meta_agent = SimpleNamespace(id="agent-meta-1")
        try:
            await client.flush_skills(user_id="u1", close_open_sessions=True)
        finally:
            await client.close()
        assert calls == [
            {
                "method": "POST",
                "endpoint": "/memory/skills/flush",
                "json": {
                    "close_open_sessions": True,
                    "user_id": "u1",
                    "meta_agent_id": "agent-meta-1",
                },
                "params": None,
            }
        ]

    async def test_list_and_get_learning_runs(self, monkeypatch):
        client, calls = _recording_client(
            monkeypatch, {"learning_runs": [{"id": "slrun-1"}]}
        )
        try:
            runs = await client.list_learning_runs(
                user_id="u1", status="active", limit=5
            )
            await client.get_learning_run("slrun-1")
        finally:
            await client.close()
        assert runs == [{"id": "slrun-1"}]
        assert calls[0] == {
            "method": "GET",
            "endpoint": "/memory/learning_runs",
            "json": None,
            "params": {"limit": 5, "user_id": "u1", "status": "active"},
        }
        assert calls[1]["endpoint"] == "/memory/learning_runs/slrun-1"

    async def test_auto_dream_sends_wait_only_when_false(self, monkeypatch):
        client, calls = _recording_client(monkeypatch, {"run_ids": []})

        async def no_user(user_id=None, headers=None):
            return None

        monkeypatch.setattr(client, "_ensure_user_exists", no_user)
        try:
            await client.auto_dream(user_id="u1", mode="procedural", wait=False)
            await client.auto_dream(user_id="u1", mode="procedural")
        finally:
            await client.close()
        assert calls[0]["json"] == {
            "mode": "procedural",
            "dry_run": False,
            "wait": False,
        }
        assert calls[1]["json"] == {"mode": "procedural", "dry_run": False}

    async def test_auto_dream_keeps_the_positional_headers(self, monkeypatch):
        """Headers stays the 7th positional parameter (as on main); the
        new wait flag is keyword-only, so a positional headers dict can never
        become wait (and go out unauthenticated)."""
        import inspect

        client = MirixClient(api_key="test-key", base_url="http://test")
        seen = []

        async def fake_request(method, endpoint, json=None, params=None, headers=None):
            seen.append((json, headers))
            return {"run_ids": []}

        async def no_user(user_id=None, headers=None):
            return None

        monkeypatch.setattr(client, "_request", fake_request)
        monkeypatch.setattr(client, "_ensure_user_exists", no_user)
        try:
            await client.auto_dream(
                "u1", "procedural", False, None, None, 2, {"Authorization": "Bearer t"}
            )
            with pytest.raises(TypeError):
                await client.auto_dream(
                    "u1", "procedural", False, None, None, 2, None, False
                )
        finally:
            await client.close()
        [(body, headers)] = seen
        assert headers == {"Authorization": "Bearer t"}
        assert body == {"mode": "procedural", "dry_run": False, "last_n_sessions": 2}
        params = inspect.signature(MirixClient.auto_dream).parameters
        assert list(params)[:8] == [
            "self",
            "user_id",
            "mode",
            "dry_run",
            "model",
            "meta_agent_id",
            "last_n_sessions",
            "headers",
        ]
        assert params["wait"].kind is inspect.Parameter.KEYWORD_ONLY


class TestLifespanSweeper:
    async def test_lifespan_helpers_start_and_stop_the_sweeper(self, monkeypatch):
        events = []

        class FakeSweeper:
            def start(self):
                events.append("start")

            async def stop(self):
                events.append("stop")

        async def fake_cancel(timeout=10.0):
            events.append("cancel-runs")
            return 0

        monkeypatch.setattr(skill_learning_trigger, "SessionSweeper", FakeSweeper)
        monkeypatch.setattr(
            skill_learning_trigger, "cancel_background_learning_runs", fake_cancel
        )
        sweeper = await rest_api.start_session_sweeper()
        assert isinstance(sweeper, FakeSweeper)
        await rest_api.stop_session_sweeper(sweeper)
        assert events == ["start", "stop", "cancel-runs"]

    async def test_a_failing_sweeper_never_blocks_startup_or_shutdown(
        self, monkeypatch
    ):
        class BrokenSweeper:
            def start(self):
                raise RuntimeError("no loop")

            async def stop(self):
                raise RuntimeError("already gone")

        async def fake_cancel(timeout=10.0):
            raise RuntimeError("cancel failed")

        monkeypatch.setattr(skill_learning_trigger, "SessionSweeper", BrokenSweeper)
        monkeypatch.setattr(
            skill_learning_trigger, "cancel_background_learning_runs", fake_cancel
        )
        sweeper = await rest_api.start_session_sweeper()
        await rest_api.stop_session_sweeper(sweeper)  # logs, does not raise

    async def test_real_sweeper_is_cancelled_cleanly(self, monkeypatch):
        passes = []

        async def fake_run_once(self):
            passes.append(1)

        monkeypatch.setattr(
            skill_learning_trigger.SessionSweeper, "run_once", fake_run_once
        )
        sweeper = await rest_api.start_session_sweeper()
        while not passes:
            await asyncio.sleep(0.01)
        task = sweeper._task
        await rest_api.stop_session_sweeper(sweeper)
        assert task.done() and not sweeper.running



# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


async def _set_session(tenant, sid, **values):
    import mirix.server.server as server_module
    from sqlalchemy import update

    from mirix.orm.conversation_session import ConversationSession as SessionORM

    async with server_module.db_context() as db:
        await db.execute(
            update(SessionORM)
            .where(SessionORM.user_id == tenant.user.id, SessionORM.session_id == sid)
            .values(**values)
        )
        await db.commit()


class TestCloseEndpointReview:
    async def test_closing_an_idle_session_triggers(self, api):
        """The close that records an idle end runs the check."""
        import datetime as dt

        from mirix.constants import SESSION_IDLE_TIMEOUT_SECONDS

        earlier = await _ended(api.a, 4, "idle-thr")
        await _write(api.a, "idle-fifth")
        long_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
            seconds=SESSION_IDLE_TIMEOUT_SECONDS + 60
        )
        await _set_session(
            api.a, "idle-fifth", last_message_at=long_ago, first_message_at=long_ago
        )
        response = await api.http.post(
            "/memory/sessions/idle-fifth/close", json={"user_id": api.a.user.id}
        )
        body = response.json()
        assert response.status_code == 200, response.text
        assert body["already_closed"] is True and body["close_reason"] == "idle"
        assert body["learning_run_id"] and api.scheduled == [body["learning_run_id"]]
        run = (await api.http.get(f"/memory/learning_runs/{body['learning_run_id']}")).json()
        assert sorted(run["session_ids"]) == sorted(earlier + ["idle-fifth"])

    async def test_a_legacy_session_is_closed_only_by_its_writer(self, api):
        """A session without a recorded meta agent is attributed to the
        writing client's meta agent; another client still gets 404."""
        await _write(api.a, "legacy-close")
        await _set_session(api.a, "legacy-close", agent_id=None)
        api.current = api.b
        denied = await api.http.post(
            "/memory/sessions/legacy-close/close", json={"user_id": api.a.user.id}
        )
        api.current = api.a
        assert denied.status_code == 404
        allowed = await api.http.post(
            "/memory/sessions/legacy-close/close", json={"user_id": api.a.user.id}
        )
        assert allowed.status_code == 200, allowed.text
        session = await ConversationSessionManager().get_session(
            organization_id=api.a.client.organization_id,
            user_id=api.a.user.id,
            session_id="legacy-close",
        )
        assert session.agent_id == api.a.meta.id and session.closed_at is not None


class TestFlushEndpointReview:
    async def test_unknown_user_is_404(self, api):
        """Not a 500."""
        response = await api.http.post(
            "/memory/skills/flush", json={"user_id": "user-does-not-exist"}
        )
        assert response.status_code == 404
        assert "user-does-not-exist" in response.json()["detail"]


class TestTenantGuardOnWrites:
    """A client never closes sessions, starts learning runs or writes
    memories for a user of another organization."""

    async def test_flush_for_another_organizations_user_is_404(self, api):
        await _ended(api.a, 3, "tg-flush")
        api.current = api.c
        response = await api.http.post(
            "/memory/skills/flush", json={"user_id": api.a.user.id}
        )
        api.current = api.a
        assert response.status_code == 404
        assert response.json()["detail"] == f"User {api.a.user.id} not found"
        assert api.scheduled == []  # no learning run for the foreign user

    async def test_close_for_another_organizations_user_is_404(self, api):
        await _write(api.a, "tg-close")
        api.current = api.c
        response = await api.http.post(
            "/memory/sessions/tg-close/close", json={"user_id": api.a.user.id}
        )
        api.current = api.a
        assert response.status_code == 404
        session = await ConversationSessionManager().get_session(
            organization_id=api.a.client.organization_id,
            user_id=api.a.user.id,
            session_id="tg-close",
        )
        assert session.closed_at is None and api.scheduled == []

    async def test_an_add_for_another_organizations_user_is_404(self, api):
        created = []

        async def create_user(pydantic_user):
            created.append(pydantic_user)
            return pydantic_user

        api.server.user_manager.create_user = create_user
        with pytest.raises(HTTPException) as refused:
            await rest_api._ensure_user_exists(api.server, api.c.client, api.a.user.id)
        assert refused.value.status_code == 404
        # Its own user is returned; a new user is created in ITS organization.
        mine = await rest_api._ensure_user_exists(
            api.server, api.c.client, api.c.user.id
        )
        assert mine.id == api.c.user.id and created == []
        new = await rest_api._ensure_user_exists(
            api.server, api.c.client, "user-brand-new"
        )
        assert new.organization_id == api.c.client.organization_id
        assert [u.id for u in created] == ["user-brand-new"]


class TestRealAuthentication:
    """The new endpoints run the real authentication; an anonymous
    or badly authenticated call is refused before anything else happens."""

    @pytest.fixture
    def real_auth(self, api, monkeypatch):
        from mirix.services.client_manager import ClientManager

        monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", _REAL_AUTH)
        fake_server = rest_api.get_server()
        monkeypatch.setattr(
            fake_server, "client_manager", ClientManager(), raising=False
        )
        monkeypatch.setattr(
            fake_server,
            "organization_manager",
            SimpleNamespace(DEFAULT_ORG_ID="org-default"),
            raising=False,
        )
        return api

    ENDPOINTS = [
        ("POST", "/memory/sessions/some-session/close", {"user_id": "u"}),
        ("POST", "/memory/sessions/bad%20id/close", {"user_id": "u"}),
        ("POST", "/memory/skills/flush", {"user_id": "u"}),
        ("GET", "/memory/learning_runs", None),
        ("GET", "/memory/learning_runs/slrun-anything", None),
    ]
    CREDENTIALS = [
        {},
        {"Authorization": "Bearer not-a-jwt"},
        {"X-API-Key": "mirix-bogus-key"},
    ]

    @pytest.mark.parametrize("method,path,body", ENDPOINTS)
    @pytest.mark.parametrize("headers", CREDENTIALS)
    async def test_refused(self, real_auth, method, path, body, headers):
        response = await real_auth.http.request(
            method, path, json=body, headers=headers
        )
        assert response.status_code in (401, 403), (path, headers, response.text)

    async def test_nothing_changes_for_a_refused_close(self, real_auth, api):
        # Written with the fake identity before the real check is in force.
        await _write(api.a, "anon-close")
        response = await real_auth.http.post(
            "/memory/sessions/anon-close/close", json={"user_id": api.a.user.id}
        )
        assert response.status_code == 401
        session = await ConversationSessionManager().get_session(
            organization_id=api.a.client.organization_id,
            user_id=api.a.user.id,
            session_id="anon-close",
        )
        assert session.closed_at is None


class TestLifespanWiring:
    async def test_the_lifespan_runs_the_sweeper_and_stops_it(self, monkeypatch):
        """The app's lifespan itself starts the sweeper after initialize
        and stops it (and in-flight runs) before cleanup. It also starts
        the skill re-embedding pass in the background, and cancels it first on
        shutdown. The retrieval log's retention purge starts last and
        stops first."""
        from mirix.services import retrieval_log_manager, skill_retrieval_reembed

        events = []
        loops = {"sweep": 0, "lease": 0}

        async def fake_purge_pass(self):
            if not self.passes:
                events.append("purge-started")
            self.passes += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("purge-cancelled")
                raise

        monkeypatch.setattr(
            retrieval_log_manager.RetrievalLogPurger, "run_once", fake_purge_pass
        )

        async def endless_reembed_pass(**kwargs):
            events.append("reembed-started")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("reembed-cancelled")
                raise

        monkeypatch.setattr(
            skill_retrieval_reembed,
            "reembed_missing_retrieval_embeddings",
            endless_reembed_pass,
        )

        async def fake_initialize():
            events.append("initialize")

        async def fake_cleanup():
            events.append("cleanup")

        async def fake_run_once(self):
            loops["sweep"] += 1

        async def fake_recover(self):
            loops["lease"] += 1
            return []

        real_cancel = skill_learning_trigger.cancel_background_learning_runs

        async def recording_cancel(timeout=10.0):
            events.append("cancel-runs")
            return await real_cancel(timeout)

        monkeypatch.setattr(rest_api, "initialize", fake_initialize)
        monkeypatch.setattr(rest_api, "cleanup", fake_cleanup)
        monkeypatch.setattr(
            skill_learning_trigger.SessionSweeper, "run_once", fake_run_once
        )
        monkeypatch.setattr(
            skill_learning_trigger.SessionSweeper, "recover_leases_once", fake_recover
        )
        monkeypatch.setattr(
            skill_learning_trigger, "cancel_background_learning_runs", recording_cancel
        )
        app = SimpleNamespace(state=SimpleNamespace())
        async with rest_api.lifespan(app):
            sweeper = app.state.session_sweeper
            reembedder = app.state.skill_reembedder
            purger = app.state.retrieval_log_purger
            while not (loops["sweep"] and loops["lease"] and purger.passes):
                await asyncio.sleep(0.01)
            assert sorted(events) == ["initialize", "purge-started", "reembed-started"]
            assert events[0] == "initialize"
            assert sweeper.running and reembedder.running and purger.running
            tasks = [t for t in (sweeper._task, sweeper._lease_task) if t is not None]
            assert tasks and not any(t.done() for t in tasks)
            purge_task = purger.task
        assert events[3:] == [
            "purge-cancelled",
            "reembed-cancelled",
            "cancel-runs",
            "cleanup",
        ]
        assert not sweeper.running and all(t.done() for t in tasks)
        assert not reembedder.running and reembedder.task.cancelled()
        assert not purger.running and purge_task.cancelled()
