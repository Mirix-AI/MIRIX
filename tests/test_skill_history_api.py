"""REST endpoints and client methods for skill version history.

* ``GET  /memory/skills/{id}/history`` — every version, oldest first, also for
  a soft-deleted skill;
* ``GET  /memory/skills/{id}/versions/{n}`` — version n rebuilt exactly;
* ``POST /memory/skills/{id}/restore`` ``{version_no}`` — restore (un-deletes
  a deleted skill); 409 when the name is taken by another live skill;
* ``DELETE /memory/procedural/{id}`` — hard delete that erases the history,
  with its user-ownership check;
* the ``MirixClient`` methods, driven end to end against the app.

Every endpoint enforces the organization and the read scope; a skill outside
them is a 404. The owner (``user_id``) is enforced when given; the writes
(restore, hard delete) default it to the client's admin user, the reads
(history, version) look a skill up by its id alone — the harness's export call.
The app runs in-process over httpx's ASGI transport with a throwaway SQLite
database and a switchable fake authentication (the real one is exercised in
TestRealAuthentication). No network, no model provider.
"""

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
import mirix.services.procedural_memory_manager as pmm
from mirix.client.remote_client import MirixClient
from mirix.orm.agent import Agent as AgentORM
from mirix.orm.base import Base
from mirix.orm.client import Client as ClientORM
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.skill_object import SkillObject as SkillObjectORM
from mirix.orm.skill_version import SkillVersion as SkillVersionORM
from mirix.orm.user import User as UserORM
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.server import rest_api
from mirix.services import skill_history as history
from mirix.services.procedural_memory_manager import ProceduralMemoryManager
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

_REAL_AUTH = rest_api.get_client_from_jwt_or_api_key


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("skill_history_api") / "test.db"
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


class _FakeEmbedModel:
    async def get_text_embedding(self, text):
        return [0.2] * 8


def _pyd_client(client_id, org, *, write_scope="test", read_scopes=("test",)):
    return PydanticClient(
        id=client_id,
        organization_id=org,
        name=client_id,
        write_scope=write_scope,
        read_scopes=list(read_scopes),
    )


@pytest_asyncio.fixture(loop_scope="module")
async def api(sm, monkeypatch):
    import mirix.server.server as server_module
    from mirix.services.admin_user_manager import ClientAuthManager

    org = f"org-{uuid.uuid4().hex[:8]}"
    other_org = f"org-{uuid.uuid4().hex[:8]}"
    c1 = f"client-{uuid.uuid4().hex[:8]}"
    c_other_org = f"client-{uuid.uuid4().hex[:8]}"
    owner = ClientAuthManager.get_admin_user_id_for_client(c1)
    stranger = f"user-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        for o in (org, other_org):
            db.add(OrganizationORM(id=o, name=o))
        await db.flush()
        for uid in (owner, stranger):
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org,
                    status="active",
                    timezone="UTC",
                )
            )
        for cid, o in ((c1, org), (c_other_org, other_org)):
            db.add(
                ClientORM(
                    id=cid,
                    organization_id=o,
                    name=cid,
                    status="active",
                    write_scope="test",
                    read_scopes=["test"],
                )
            )
        db.add(AgentORM(id=f"agent-{uuid.uuid4()}", organization_id=org))
        await db.commit()

    async def fake_embedding_model(config, *args, **kwargs):
        return _FakeEmbedModel()

    manager = ProceduralMemoryManager(session_maker=sm)
    state = SimpleNamespace(
        org=org,
        other_org=other_org,
        owner=owner,
        stranger=stranger,
        client=_pyd_client(c1, org),
        other_org_client=_pyd_client(c_other_org, other_org),
        blind_client=_pyd_client(c1, org, read_scopes=("elsewhere",)),
        read_only_client=_pyd_client(c1, org, write_scope=None),
        manager=manager,
        sm=sm,
    )
    state.current = state.client

    async def fake_auth(authorization=None, http_request=None):
        return state.current, "api_key"

    monkeypatch.setattr(server_module, "db_context", sm)
    monkeypatch.setattr(pmm, "embedding_model", fake_embedding_model)
    monkeypatch.setattr(rest_api, "get_client_from_jwt_or_api_key", fake_auth)
    fake_server = SimpleNamespace(procedural_memory_manager=manager)
    monkeypatch.setattr(rest_api, "get_server", lambda: fake_server)
    transport = httpx.ASGITransport(app=rest_api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        state.http = http
        yield state


async def _skill(api, name="deploy", *, user_id=None, **overrides):
    fields = dict(
        name=name,
        entry_type="workflow",
        description=f"How to {name} — ✓",
        instructions=f"1. {name}\n2. verify",
        triggers=[f"when {name}"],
        examples=[{"output": {"b": 2, "a": 1}, "input": "go"}],
        filter_tags={"scope": "test"},
        organization_id=api.org,
    )
    fields.update(overrides)
    result = await api.manager.apply_skill_change(
        "create", actor=api.client, user_id=user_id or api.owner, fields=fields
    )
    return result.skill


async def _edit(api, skill_id, **fields):
    return await api.manager.apply_skill_change(
        "edit", actor=api.client, user_id=api.owner, skill_id=skill_id, fields=fields
    )


async def _delete(api, skill_id, reason=None):
    return await api.manager.apply_skill_change(
        "delete", actor=api.client, user_id=api.owner, skill_id=skill_id, reason=reason
    )


class TestHistory:
    async def test_lists_every_version_with_changed_fields(self, api):
        skill = await _skill(api, "hist")
        await _edit(api, skill.id, instructions="1. better")
        await _delete(api, skill.id, reason="obsolete")
        response = await api.http.get(f"/memory/skills/{skill.id}/history")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["skill_id"] == skill.id and body["name"] == "hist"
        assert body["is_deleted"] is True
        assert body["delete_reason"] == "obsolete"
        assert body["deleted_at"]
        versions = body["versions"]
        assert [(v["version_no"], v["change_type"]) for v in versions] == [
            (1, "create"),
            (2, "edit"),
            (3, "delete"),
        ]
        assert versions[1]["changed_fields"] == ["instructions"]
        assert versions[2]["changed_fields"] == []
        assert versions[2]["reason"] == "obsolete"
        assert body["skill"]["user_id"] == api.owner

    async def test_says_whether_the_skill_has_its_retrieval_vector(self, api):
        """A skill without its vector (its embedding failed, or none is
        configured, as here) is invisible to gated retrieval until re-embedded;
        the history says which it is."""
        from mirix.constants import MAX_EMBEDDING_DIM
        from mirix.orm.procedural_memory import ProceduralMemoryItem

        skill = await _skill(api, "vectorless")
        path = f"/memory/skills/{skill.id}/history"
        assert (await api.http.get(path)).json()["has_retrieval_embedding"] is False
        async with api.sm() as db:
            row = await db.get(ProceduralMemoryItem, skill.id)
            row.retrieval_embedding = [0.1] * MAX_EMBEDDING_DIM
            await db.commit()
        assert (await api.http.get(path)).json()["has_retrieval_embedding"] is True

    async def test_404_outside_owner_organization_or_scope(self, api):
        skill = await _skill(api, "guarded")
        path = f"/memory/skills/{skill.id}/history"
        assert (
            await api.http.get(path, params={"user_id": api.stranger})
        ).status_code == 404
        api.current = api.other_org_client
        assert (
            await api.http.get(path, params={"user_id": api.owner})
        ).status_code == 404
        api.current = api.blind_client
        assert (await api.http.get(path)).status_code == 404
        api.current = api.client
        assert (
            await api.http.get("/memory/skills/proc-nope/history")
        ).status_code == 404

    async def test_without_user_id_any_user_of_the_organization(self, api):
        """The eval harness exports a skill's history by its id alone
        (``skill_history(skill_id)``): a skill of a user other than the
        client's admin user is found; organization and scope still hold."""
        skill = await _skill(api, "someone-elses", user_id=api.stranger)
        await api.manager.apply_skill_change(
            "edit",
            actor=api.client,
            user_id=api.stranger,
            skill_id=skill.id,
            fields={"instructions": "1. other\n2. verify"},
        )
        path = f"/memory/skills/{skill.id}/history"
        response = await api.http.get(path)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["skill"]["user_id"] == api.stranger
        assert [v["change_type"] for v in body["versions"]] == ["create", "edit"]
        version = await api.http.get(f"/memory/skills/{skill.id}/versions/1")
        assert version.status_code == 200, version.text
        assert version.json()["content"]["name"] == "someone-elses"
        # Naming a user still restricts to that user.
        assert (
            await api.http.get(path, params={"user_id": api.owner})
        ).status_code == 404
        assert (
            await api.http.get(
                f"/memory/skills/{skill.id}/versions/1", params={"user_id": api.owner}
            )
        ).status_code == 404
        # The tenant boundary: another organization, a scope the client cannot read.
        for client in (api.other_org_client, api.blind_client):
            api.current = client
            assert (await api.http.get(path)).status_code == 404
            assert (
                await api.http.get(f"/memory/skills/{skill.id}/versions/1")
            ).status_code == 404
        api.current = api.client
        # The writes keep their default owner (the client's admin user).
        restore = await api.http.post(
            f"/memory/skills/{skill.id}/restore", json={"version_no": 1}
        )
        assert restore.status_code == 404


class TestVersion:
    async def test_rebuilds_the_version_exactly(self, api):
        skill = await _skill(api, "exact")
        original = history.content_of(skill)
        await _edit(api, skill.id, description="changed", examples=[])
        response = await api.http.get(f"/memory/skills/{skill.id}/versions/1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["version"]["version_no"] == 1
        assert body["content"] == original
        # Byte-identical canonical form after the JSON round trip.
        for field, value in body["content"].items():
            assert history.canonical_bytes(field, value) == history.canonical_bytes(
                field, original[field]
            )

    async def test_unknown_version_is_404(self, api):
        skill = await _skill(api, "short-history")
        response = await api.http.get(f"/memory/skills/{skill.id}/versions/7")
        assert response.status_code == 404


class TestRestore:
    async def test_restores_and_undeletes(self, api):
        skill = await _skill(api, "back")
        await _edit(api, skill.id, instructions="1. wrong turn")
        await _delete(api, skill.id)
        response = await api.http.post(
            f"/memory/skills/{skill.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["changed"] is True and body["restored_version_no"] == 1
        assert body["version"]["change_type"] == "restore"
        assert body["version"]["version_no"] == 4
        assert body["skill"]["instructions"] == "1. back\n2. verify"
        found = await api.manager.get_item_by_id(
            item_id=skill.id,
            user=PydanticUser(
                id=api.owner, name="o", organization_id=api.org, timezone="UTC"
            ),
            timezone_str="UTC",
            actor=api.client,
        )
        assert found.instructions == "1. back\n2. verify"

    async def test_name_conflict_is_409(self, api):
        skill = await _skill(api, "clash")
        await _delete(api, skill.id)
        await _skill(api, "clash", description="the new one")
        response = await api.http.post(
            f"/memory/skills/{skill.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 409, response.text

    async def test_404s_403_and_422(self, api):
        skill = await _skill(api, "restricted")
        path = f"/memory/skills/{skill.id}/restore"
        assert (await api.http.post(path, json={"version_no": 5})).status_code == 404
        assert (
            await api.http.post(path, json={"version_no": 1, "user_id": api.stranger})
        ).status_code == 404
        assert (await api.http.post(path, json={"version_no": 0})).status_code == 422
        api.current = api.other_org_client
        assert (
            await api.http.post(path, json={"version_no": 1, "user_id": api.owner})
        ).status_code == 404
        api.current = api.blind_client
        assert (await api.http.post(path, json={"version_no": 1})).status_code == 404
        api.current = api.read_only_client
        assert (await api.http.post(path, json={"version_no": 1})).status_code == 403
        api.current = api.client
        async with api.sm() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(SkillVersionORM)
                    .where(SkillVersionORM.skill_id == skill.id)
                )
            ).scalar_one()
        assert count == 1  # nothing was written by the refused calls

    async def test_a_restore_needs_the_write_scope_not_only_a_read_scope(self, api):
        # A client that can READ scope "other" (but writes "test")
        # cannot rewrite or un-delete a skill of "other" through a restore.
        foreign = await _skill(api, "other-scope", filter_tags={"scope": "other"})
        await _edit(api, foreign.id, instructions="1. edited in other")
        await _delete(api, foreign.id)
        api.current = _pyd_client(
            api.client.id, api.org, write_scope="test", read_scopes=("test", "other")
        )
        response = await api.http.post(
            f"/memory/skills/{foreign.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 403, response.text
        # A skill that is in "other" now, whatever scope the version had.
        left = await _skill(api, "moved-out")  # scope "test"
        await _edit(api, left.id, filter_tags={"scope": "other"})
        response = await api.http.post(
            f"/memory/skills/{left.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 403, response.text
        # Nor can it move a skill of its own scope back into another scope.
        moved = await _skill(api, "moved-in", filter_tags={"scope": "other"})
        await _edit(api, moved.id, filter_tags={"scope": "test"})
        response = await api.http.post(
            f"/memory/skills/{moved.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 403, response.text
        api.current = api.client
        async with api.sm() as db:
            counts = {
                skill_id: (
                    await db.execute(
                        select(func.count())
                        .select_from(SkillVersionORM)
                        .where(SkillVersionORM.skill_id == skill_id)
                    )
                ).scalar_one()
                for skill_id in (foreign.id, left.id, moved.id)
            }
        assert counts == {foreign.id: 3, left.id: 2, moved.id: 2}  # nothing written

    async def test_the_409_never_names_the_holder(self, api):
        # The holder of the name may be of a scope the caller cannot read.
        mine = await _skill(api, "held-name")
        await _delete(api, mine.id)
        hidden = await _skill(
            api, "held-name", filter_tags={"scope": "hidden"}, description="hidden"
        )
        response = await api.http.post(
            f"/memory/skills/{mine.id}/restore", json={"version_no": 1}
        )
        assert response.status_code == 409
        assert hidden.id not in response.text
        assert "held-name" in response.json()["detail"]


class TestHardDelete:
    async def test_other_users_skill_is_404_and_intact(self, api):
        skill = await _skill(api, "not-yours")
        response = await api.http.delete(
            f"/memory/procedural/{skill.id}", params={"user_id": api.stranger}
        )
        assert response.status_code == 404
        assert (
            await api.http.get(f"/memory/skills/{skill.id}/history")
        ).status_code == 200

    async def test_owner_delete_erases_the_history(self, api):
        skill = await _skill(api, "erase-me", description="unique erase-me text")
        await _edit(api, skill.id, instructions="unique erase-me steps")
        response = await api.http.delete(f"/memory/procedural/{skill.id}")
        assert response.status_code == 200, response.text
        assert (
            await api.http.get(f"/memory/skills/{skill.id}/history")
        ).status_code == 404
        async with api.sm() as db:
            versions = (
                await db.execute(
                    select(func.count())
                    .select_from(SkillVersionORM)
                    .where(SkillVersionORM.skill_id == skill.id)
                )
            ).scalar_one()
            unique_objects = (
                await db.execute(
                    select(func.count())
                    .select_from(SkillObjectORM)
                    .where(
                        SkillObjectORM.sha256.in_(
                            [
                                history.sha256_hex(b"unique erase-me text"),
                                history.sha256_hex(b"unique erase-me steps"),
                            ]
                        )
                    )
                )
            ).scalar_one()
        assert versions == 0 and unique_objects == 0

    @pytest.mark.parametrize(
        "caller,scope,status",
        [
            # Writes "test" but cannot read it: it cannot erase by an id it
            # could only have guessed.
            ("blind_client", "test", 404),
            ("read_only_client", "test", 403),  # no write_scope at all
            ("wide_reader", "other", 404),  # reads "other", writes only "test"
        ],
    )
    async def test_only_the_skills_write_scope_can_erase_it(
        self, api, caller, scope, status
    ):
        # The erasure is a write — the skill must be of the caller's
        # write_scope, which it can also read (checked in the purge
        # transaction); a skill of another scope is not found, whether the
        # caller can read it or not.
        skill = await _skill(api, f"erase-{caller}", filter_tags={"scope": scope})
        if caller == "wide_reader":
            api.current = _pyd_client(
                api.client.id,
                api.org,
                write_scope="test",
                read_scopes=("test", "other"),
            )
        else:
            api.current = getattr(api, caller)
        response = await api.http.delete(f"/memory/procedural/{skill.id}")
        assert response.status_code == status, response.text
        api.current = api.client
        async with api.sm() as db:
            versions = (
                await db.execute(
                    select(func.count())
                    .select_from(SkillVersionORM)
                    .where(SkillVersionORM.skill_id == skill.id)
                )
            ).scalar_one()
        assert versions == 1  # the skill and its history are intact


class TestRealAuthentication:
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

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("GET", "/memory/skills/proc-x/history", None),
            ("GET", "/memory/skills/proc-x/versions/1", None),
            ("POST", "/memory/skills/proc-x/restore", {"version_no": 1}),
            ("DELETE", "/memory/procedural/proc-x", None),
        ],
    )
    @pytest.mark.parametrize(
        "headers", [{}, {"Authorization": "Bearer not-a-jwt"}, {"X-API-Key": "bogus"}]
    )
    async def test_refused(self, real_auth, method, path, body, headers):
        response = await real_auth.http.request(
            method, path, json=body, headers=headers
        )
        assert response.status_code in (401, 403), (path, headers, response.text)


class TestClientMethods:
    """The MirixClient methods, end to end against the app."""

    @pytest_asyncio.fixture(loop_scope="module")
    async def client(self, api):
        client = MirixClient(api_key="test-key", base_url="http://test")
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rest_api.app), base_url="http://test"
        )
        try:
            yield client
        finally:
            await client.close()

    async def test_history_version_restore_delete(self, api, client):
        skill = await _skill(api, "via-client")
        await _edit(api, skill.id, instructions="1. edited")
        await _delete(api, skill.id, reason="retired")

        hist = await client.get_skill_history(skill.id)
        assert [v["change_type"] for v in hist["versions"]] == [
            "create",
            "edit",
            "delete",
        ]
        assert hist["delete_reason"] == "retired"

        v1 = await client.get_skill_version(skill.id, 1, user_id=api.owner)
        assert v1["content"]["instructions"] == "1. via-client\n2. verify"

        restored = await client.restore_skill(skill.id, 1)
        assert restored["changed"] is True
        assert restored["skill"]["instructions"] == "1. via-client\n2. verify"

        with pytest.raises(Exception):
            await client.restore_skill(skill.id, 99)

        await client.delete_skill(skill.id)
        with pytest.raises(Exception):
            await client.get_skill_history(skill.id)

    async def test_request_shapes(self, monkeypatch):
        client = MirixClient(api_key="test-key", base_url="http://test")
        calls = []

        async def fake_request(method, endpoint, json=None, params=None, headers=None):
            calls.append((method, endpoint, json, params))
            return {}

        monkeypatch.setattr(client, "_request", fake_request)
        try:
            await client.get_skill_history("proc-1")
            await client.get_skill_history("proc-1", user_id="u1")
            await client.get_skill_version("proc-1", 2)
            await client.restore_skill("proc-1", 3, user_id="u1")
            await client.delete_skill("proc-1", user_id="u1")
        finally:
            await client.close()
        assert calls == [
            ("GET", "/memory/skills/proc-1/history", None, None),
            ("GET", "/memory/skills/proc-1/history", None, {"user_id": "u1"}),
            ("GET", "/memory/skills/proc-1/versions/2", None, None),
            (
                "POST",
                "/memory/skills/proc-1/restore",
                {"version_no": 3, "user_id": "u1"},
                None,
            ),
            ("DELETE", "/memory/procedural/proc-1", None, {"user_id": "u1"}),
        ]
