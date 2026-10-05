"""The retrieval log: writing, listing, retention,
erasure, and the background purge.

SQLite throwaway database; a clock is injected where time matters. The
Postgres behaviour of the same manager is in tests/test_retrieval_log_pg.py.
"""

import asyncio
import datetime as dt
import math
import time
import uuid
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix import constants
from mirix.orm.base import Base
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.retrieval_log import RetrievalLog as RetrievalLogORM
from mirix.orm.user import User as UserORM
from mirix.services.retrieval_log_manager import (
    MAX_LIST_LIMIT,
    RetrievalLogManager,
    RetrievalLogPurger,
    json_safe,
    retrieval_log_enabled,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

T0 = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **delta):
        self.now = self.now + dt.timedelta(**delta)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("retrieval_log") / "test.db"
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
async def scope(sm):
    org = f"org-{uuid.uuid4().hex[:8]}"
    users = [f"user-{uuid.uuid4().hex[:8]}" for _ in range(2)]
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        for uid in users:
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org,
                    status="active",
                    timezone="UTC",
                )
            )
        await db.commit()
    return org, users


def _log_kwargs(org, user, **overrides):
    values = dict(
        organization_id=org,
        user_id=user,
        client_id="client-a",
        session_id="task-1",
        memory_types=["procedural"],
        query_text="USER: sum column B",
        extracted_topics="sum column B",
        extraction_status="ok",
        embedding_status="ok",
        per_type={"procedural": {"returned_ids": ["s1"], "candidates": []}},
        llm_usage={"topic_extraction": {"prompt_tokens": 10}},
        latency_ms=12,
        retry_events=[{"kind": "infra", "attempt": 1, "waited_s": 0.5, "at": "x"}],
        embedding_chunked_calls=2,
    )
    values.update(overrides)
    return values


async def _all(sm, **where):
    async with sm() as db:
        stmt = select(RetrievalLogORM)
        for key, value in where.items():
            stmt = stmt.where(getattr(RetrievalLogORM, key) == value)
        return list((await db.execute(stmt)).scalars().all())


class TestWrite:
    async def test_every_m7_4_field_round_trips(self, sm, scope):
        org, (user, _) = scope
        manager = RetrievalLogManager(session_maker=sm, clock=Clock())
        log_id = await manager.create_log(**_log_kwargs(org, user))
        assert log_id.startswith("rlog-")
        [row] = await _all(sm, id=log_id)
        assert row.organization_id == org and row.user_id == user
        assert row.client_id == "client-a" and row.session_id == "task-1"
        assert row.memory_types == ["procedural"]
        assert row.query_text == "USER: sum column B"
        assert row.extracted_topics == "sum column B"
        assert row.extraction_status == "ok" and row.embedding_status == "ok"
        assert row.per_type["procedural"]["returned_ids"] == ["s1"]
        assert row.llm_usage == {"topic_extraction": {"prompt_tokens": 10}}
        assert row.latency_ms == 12 and row.embedding_chunked_calls == 2
        assert row.retry_events[0]["kind"] == "infra"
        assert row.created_at.replace(tzinfo=dt.timezone.utc) == T0

    async def test_values_are_validated_before_writing(self, sm, scope):
        org, (user, _) = scope
        manager = RetrievalLogManager(session_maker=sm)
        for bad in (
            {"memory_types": ["skills"]},
            {"extraction_status": "maybe"},
            {"embedding_status": "partial"},
            {"session_id": "has space"},
        ):
            with pytest.raises(ValueError):
                await manager.create_log(**_log_kwargs(org, user, **bad))
        assert await _all(sm, user_id=user, extraction_status="maybe") == []

    async def test_non_finite_numbers_become_null(self, sm, scope):
        """Postgres rejects NaN in a JSON column."""
        org, (user, _) = scope
        manager = RetrievalLogManager(session_maker=sm)
        per_type = {
            "procedural": {"candidates": [{"cosine": math.nan, "bm25": math.inf}]}
        }
        log_id = await manager.create_log(**_log_kwargs(org, user, per_type=per_type))
        [row] = await _all(sm, id=log_id)
        assert row.per_type["procedural"]["candidates"] == [
            {"cosine": None, "bm25": None}
        ]

    def test_json_safe(self):
        import numpy as np

        assert json_safe(
            {"a": (1, 2.5, np.float64(0.25)), 3: {dt.date(2026, 1, 2)}}
        ) == {
            "a": [1, 2.5, 0.25],
            "3": ["2026-01-02"],
        }
        assert json_safe(float("-inf")) is None and json_safe(object()).startswith("<")

    async def test_a_transient_database_error_is_retried(self, sm, scope, monkeypatch):
        from sqlalchemy.exc import OperationalError

        from mirix.services import retry_policy

        org, (user, _) = scope
        failures = []

        @asynccontextmanager
        async def flaky():
            if not failures:
                failures.append(1)
                raise OperationalError("INSERT", {}, Exception("database is locked"))
            async with sm() as session:
                yield session

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(retry_policy.asyncio, "sleep", no_sleep)
        log_id = await RetrievalLogManager(session_maker=flaky).create_log(
            **_log_kwargs(org, user)
        )
        assert failures == [1] and len(await _all(sm, id=log_id)) == 1

    def test_the_switch_is_read_at_call_time(self, monkeypatch):
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_ENABLED", False)
        assert retrieval_log_enabled() is False
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_ENABLED", True)
        assert retrieval_log_enabled() is True


class TestList:
    async def test_scoped_to_the_client_newest_first(self, sm, scope):
        org, (user, other_user) = scope
        clock = Clock()
        manager = RetrievalLogManager(session_maker=sm, clock=clock)
        client = f"client-{uuid.uuid4().hex[:6]}"
        ids = []
        for session in ("s-a", "s-b", "s-a"):
            clock.advance(seconds=1)
            ids.append(
                await manager.create_log(
                    **_log_kwargs(org, user, client_id=client, session_id=session)
                )
            )
        clock.advance(seconds=1)
        other_user_log = await manager.create_log(
            **_log_kwargs(org, other_user, client_id=client, session_id="s-a")
        )
        # Another client's and another organization's rows are never visible.
        await manager.create_log(**_log_kwargs(org, user, client_id="client-other"))

        listed = await manager.list_logs(organization_id=org, client_id=client)
        assert [log.id for log in listed] == [other_user_log, ids[2], ids[1], ids[0]]
        by_session = await manager.list_logs(
            organization_id=org, client_id=client, session_id="s-a"
        )
        assert [log.id for log in by_session] == [other_user_log, ids[2], ids[0]]
        by_user = await manager.list_logs(
            organization_id=org, client_id=client, user_id=user, session_id="s-a"
        )
        assert [log.id for log in by_user] == [ids[2], ids[0]]
        assert [
            log.id
            for log in await manager.list_logs(
                organization_id=org, client_id=client, limit=1
            )
        ] == [other_user_log]
        assert (
            await manager.list_logs(organization_id="org-none", client_id=client) == []
        )
        got = await manager.get_log(ids[0], organization_id=org)
        assert got is not None and got.session_id == "s-a"
        assert await manager.get_log(ids[0], organization_id="org-none") is None

    async def test_limit_is_capped(self, sm, scope, monkeypatch):
        org, (user, _) = scope
        manager = RetrievalLogManager(session_maker=sm)
        seen = {}
        real_select = __import__(
            "mirix.services.retrieval_log_manager", fromlist=["x"]
        ).select

        def spy_select(*args):
            stmt = real_select(*args)
            original_limit = stmt.limit

            def limit(n):
                seen["limit"] = n
                return original_limit(n)

            stmt.limit = limit  # type: ignore[method-assign]
            return stmt

        import mirix.services.retrieval_log_manager as rlm

        monkeypatch.setattr(rlm, "select", spy_select)
        await manager.list_logs(organization_id=org, client_id="c", limit=10_000)
        assert seen["limit"] == MAX_LIST_LIMIT
        await manager.list_logs(organization_id=org, client_id="c", limit=0)
        assert seen["limit"] == 1


class TestRetentionAndErasure:
    async def _logs_at(self, sm, org, user, clock, ages_days, client="client-r"):
        manager = RetrievalLogManager(session_maker=sm, clock=clock)
        ids = {}
        for age in ages_days:
            clock.now = T0 - dt.timedelta(days=age)
            ids[age] = await manager.create_log(
                **_log_kwargs(org, user, client_id=client)
            )
        clock.now = T0
        return manager, ids

    async def test_purge_deletes_rows_past_the_window(self, sm, scope):
        org, (user, _) = scope
        client = f"client-{uuid.uuid4().hex[:6]}"
        manager, ids = await self._logs_at(
            sm, org, user, Clock(), [40, 31, 29, 1], client
        )
        assert await manager.purge_expired(retention_days=30) == 2
        left = {row.id for row in await _all(sm, client_id=client)}
        assert left == {ids[29], ids[1]}

    async def test_default_window_and_keep_forever(self, sm, scope, monkeypatch):
        org, (user, _) = scope
        client = f"client-{uuid.uuid4().hex[:6]}"
        manager, ids = await self._logs_at(sm, org, user, Clock(), [400, 5], client)
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_RETENTION_DAYS", 0)
        assert await manager.purge_expired() == 0
        assert await manager.purge_expired(retention_days=-1) == 0
        assert len(await _all(sm, client_id=client)) == 2
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_RETENTION_DAYS", 30)
        assert await manager.purge_expired() >= 1
        assert {row.id for row in await _all(sm, client_id=client)} == {ids[5]}

    async def test_purge_works_in_batches(self, sm, scope, monkeypatch):
        org, (user, _) = scope
        client = f"client-{uuid.uuid4().hex[:6]}"
        manager, _ids = await self._logs_at(
            sm, org, user, Clock(), [90, 91, 92, 93, 94], client
        )
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_PURGE_BATCH", 2)
        assert await manager.purge_expired(retention_days=30) >= 5
        assert await _all(sm, client_id=client) == []

    async def test_user_and_client_erasure(self, sm, scope):
        org, (user, other_user) = scope
        manager = RetrievalLogManager(session_maker=sm)
        client = f"client-{uuid.uuid4().hex[:6]}"
        mine = await manager.create_log(**_log_kwargs(org, user, client_id=client))
        theirs = await manager.create_log(
            **_log_kwargs(org, other_user, client_id=client)
        )
        other_client = await manager.create_log(
            **_log_kwargs(org, other_user, client_id="client-keep")
        )
        assert await manager.delete_by_user_id(user) >= 1
        assert await _all(sm, id=mine) == [] and len(await _all(sm, id=theirs)) == 1
        assert await manager.delete_by_client_id(client) == 1
        assert await _all(sm, id=theirs) == []
        assert len(await _all(sm, id=other_client)) == 1


class TestErasureHooks:
    async def test_user_memory_purge_deletes_the_users_rows(
        self, sm, scope, monkeypatch
    ):
        """DELETE /users/{id}/memories erases the user's retrieval log."""
        import mirix.server.server as server_module
        from mirix.services.user_manager import UserManager

        org, (user, other_user) = scope
        monkeypatch.setattr(server_module, "db_context", sm)
        manager = RetrievalLogManager(session_maker=sm)
        mine = await manager.create_log(**_log_kwargs(org, user, client_id="c-erase"))
        theirs = await manager.create_log(
            **_log_kwargs(org, other_user, client_id="c-erase")
        )
        await UserManager().delete_memories_by_user_id(user)
        assert await _all(sm, id=mine) == []
        assert len(await _all(sm, id=theirs)) == 1

    async def test_client_memory_purge_deletes_the_clients_rows(
        self, sm, scope, monkeypatch
    ):
        """DELETE /clients/{id}/memories erases that client's retrieval log."""
        import mirix.server.server as server_module
        from mirix.orm.client import Client as ClientORM
        from mirix.services.client_manager import ClientManager

        org, (user, _) = scope
        monkeypatch.setattr(server_module, "db_context", sm)
        client_id = f"client-{uuid.uuid4().hex[:8]}"
        async with sm() as db:
            db.add(
                ClientORM(
                    id=client_id,
                    organization_id=org,
                    name=client_id,
                    status="active",
                    write_scope="s",
                    read_scopes=["s"],
                )
            )
            await db.commit()
        manager = RetrievalLogManager(session_maker=sm)
        mine = await manager.create_log(**_log_kwargs(org, user, client_id=client_id))
        kept = await manager.create_log(
            **_log_kwargs(org, user, client_id="client-keep-2")
        )
        await ClientManager().delete_memories_by_client_id(client_id)
        assert await _all(sm, id=mine) == []
        assert len(await _all(sm, id=kept)) == 1


class TestErasureRaces:
    """Deleting a user deletes its rows, and a retrieval in
    flight while its user (or client) is erased or deleted writes none."""

    async def _new_user(self, sm, org):
        uid = f"user-{uuid.uuid4().hex[:8]}"
        async with sm() as db:
            db.add(
                UserORM(
                    id=uid,
                    name=uid,
                    organization_id=org,
                    status="active",
                    timezone="UTC",
                )
            )
            await db.commit()
        return uid

    async def test_deleting_a_user_deletes_its_rows(self, sm, scope, monkeypatch):
        """DELETE /users/{id}: the raw queries go with the user."""
        import mirix.server.server as server_module
        from mirix.services.user_manager import UserManager

        org, (_, other_user) = scope
        monkeypatch.setattr(server_module, "db_context", sm)
        user = await self._new_user(sm, org)
        manager = RetrievalLogManager(session_maker=sm)
        mine = await manager.create_log(**_log_kwargs(org, user, client_id="c-del"))
        theirs = await manager.create_log(
            **_log_kwargs(org, other_user, client_id="c-del")
        )
        await UserManager().delete_user_by_id(user)
        assert await _all(sm, id=mine) == []
        assert len(await _all(sm, id=theirs)) == 1
        assert (
            await manager.list_logs(
                organization_id=org, client_id="c-del", user_id=user
            )
            == []
        )

    async def test_a_deleted_user_gets_no_new_row(self, sm, scope):
        """A retrieval that finishes after its user was (soft) deleted."""
        org, _ = scope
        user = await self._new_user(sm, org)
        async with sm() as db:
            row = (
                await db.execute(select(UserORM).where(UserORM.id == user))
            ).scalar_one()
            row.is_deleted = True
            await db.commit()
        manager = RetrievalLogManager(session_maker=sm)
        assert await manager.create_log(**_log_kwargs(org, user)) is None
        assert await _all(sm, user_id=user) == []
        # A user that does not exist at all gets none either.
        assert await manager.create_log(**_log_kwargs(org, "user-never")) is None

    @pytest.mark.parametrize("target", ["user", "client"])
    async def test_a_retrieval_racing_an_erasure_writes_nothing(
        self, sm, scope, target
    ):
        """The retrieval started, then the erasure ran (and deleted the rows);
        the retrieval's row, written after it, would undo the erasure."""
        from mirix.services.retrieval_log_manager import erasing_retrieval_logs

        org, _ = scope
        user = await self._new_user(sm, org)
        client = f"client-{uuid.uuid4().hex[:6]}"
        manager = RetrievalLogManager(session_maker=sm)
        erasure = (
            erasing_retrieval_logs(user)
            if target == "user"
            else erasing_retrieval_logs(client_id=client)
        )
        started = time.monotonic()
        async with erasure:  # while the erasure runs
            assert (
                await manager.create_log(
                    **_log_kwargs(org, user, client_id=client),
                    started_monotonic=started,
                )
                is None
            )
        # After it ended: the retrieval that started before it still writes
        # nothing, one that started after it does.
        assert (
            await manager.create_log(
                **_log_kwargs(org, user, client_id=client), started_monotonic=started
            )
            is None
        )
        later = await manager.create_log(
            **_log_kwargs(org, user, client_id=client),
            started_monotonic=time.monotonic(),
        )
        assert later is not None
        assert [r.id for r in await _all(sm, user_id=user)] == [later]

    async def test_an_erasure_starting_during_the_insert_removes_the_row(
        self, sm, scope, monkeypatch
    ):
        """The erasure's DELETE ran between the check and the COMMIT: the
        writer removes its own row."""
        from mirix.services import retrieval_log_manager as rlm

        org, _ = scope
        user = await self._new_user(sm, org)
        manager = RetrievalLogManager(session_maker=sm)
        real_live = rlm._user_is_live
        erasure = rlm.erasing_retrieval_logs(user)

        async def live_then_erase(db, uid):
            ok = await real_live(db, uid)
            await erasure.__aenter__()  # the erasure starts now
            return ok

        monkeypatch.setattr(rlm, "_user_is_live", live_then_erase)
        try:
            result = await manager.create_log(
                **_log_kwargs(org, user), started_monotonic=time.monotonic()
            )
        finally:
            await erasure.__aexit__(None, None, None)
        assert result is None and await _all(sm, user_id=user) == []

    async def test_the_erasure_bookkeeping_nests(self):
        from mirix.services import retrieval_log_manager as rlm

        uid = f"user-{uuid.uuid4().hex[:8]}"
        before = time.monotonic()
        async with rlm.erasing_retrieval_logs(uid):
            async with rlm.erasing_retrieval_logs(uid):
                pass
            assert rlm.erased_since(uid, time.monotonic())  # still running
        assert ("user", uid) not in rlm._ERASURES_RUNNING
        assert rlm.erased_since(uid, before)
        assert not rlm.erased_since(uid, time.monotonic())
        assert not rlm.erased_since(uid, None)
        async with rlm.erasing_retrieval_logs(None):  # a no-op
            pass


class TestLostCommit:
    async def test_a_retry_after_a_committed_insert_returns_the_id(
        self, sm, scope, monkeypatch
    ):
        """The COMMIT went through but its reply was lost; the retry's
        duplicate-key error means the row IS there."""
        from sqlalchemy.exc import OperationalError

        from mirix.services import retry_policy

        org, (user, _) = scope
        commits = []

        @asynccontextmanager
        async def lossy():
            async with sm() as session:
                real_commit = session.commit

                async def commit():
                    await real_commit()
                    if not commits:
                        commits.append(1)
                        raise OperationalError(
                            "COMMIT", {}, Exception("server closed the connection")
                        )

                session.commit = commit
                yield session

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(retry_policy.asyncio, "sleep", no_sleep)
        log_id = await RetrievalLogManager(session_maker=lossy).create_log(
            **_log_kwargs(org, user)
        )
        assert log_id is not None and commits == [1]
        assert len(await _all(sm, id=log_id)) == 1

    async def test_a_real_integrity_error_still_raises(self, sm, scope, monkeypatch):
        from sqlalchemy.exc import IntegrityError

        org, (user, _) = scope

        @asynccontextmanager
        async def broken():
            async with sm() as session:

                async def commit():
                    raise IntegrityError("INSERT", {}, Exception("check failed"))

                session.commit = commit
                yield session

        with pytest.raises(IntegrityError):
            await RetrievalLogManager(session_maker=broken).create_log(
                **_log_kwargs(org, user)
            )


class TestRowSize:
    async def test_a_huge_query_keeps_its_head_and_tail(self, sm, scope, monkeypatch):
        """The raw query is bounded, with a marker and the count."""
        monkeypatch.setattr(constants, "RETRIEVAL_LOG_QUERY_MAX_CHARS", 20)
        org, (user, _) = scope
        text = "HEAD" + "x" * 1000 + "TAIL"
        log_id = await RetrievalLogManager(session_maker=sm).create_log(
            **_log_kwargs(org, user, query_text=text)
        )
        [row] = await _all(sm, id=log_id)
        assert (
            row.query_text
            == "HEAD" + "x" * 6 + "\n[... 988 characters cut ...]\n" + "x" * 6 + "TAIL"
        )
        assert row.llm_usage["query_text_truncated_chars"] == 988
        assert row.llm_usage["topic_extraction"] == {"prompt_tokens": 10}

    async def test_nul_in_the_query_is_stored_as_u_fffd(self, sm, scope):
        """Postgres cannot store U+0000; the row keeps the query readable."""
        org, (user, _) = scope
        log_id = await RetrievalLogManager(session_maker=sm).create_log(
            **_log_kwargs(
                org, user, query_text="a\x00b\x00", extracted_topics="t\x00"
            )
        )
        [row] = await _all(sm, id=log_id)
        assert row.query_text == "a\ufffdb\ufffd"
        assert row.extracted_topics == "t\ufffd"
        assert row.llm_usage["nul_chars_replaced"] == 3
        assert row.llm_usage["topic_extraction"] == {"prompt_tokens": 10}

    def test_cut_query_text(self):
        from mirix.services.retrieval_log_manager import cut_query_text

        assert cut_query_text(None, 5) == (None, 0)
        assert cut_query_text("short", 5) == ("short", 0)
        assert cut_query_text("abcdefgh", 0) == ("abcdefgh", 0)  # no limit
        assert cut_query_text("abcdefgh", 3) == ("ab\n[... 5 characters cut ...]\nh", 5)
        assert cut_query_text("abcdefgh", 1) == ("a\n[... 7 characters cut ...]\n", 7)

    async def test_listing_without_the_trace(self, sm, scope):
        org, (user, _) = scope
        client = f"client-{uuid.uuid4().hex[:6]}"
        manager = RetrievalLogManager(session_maker=sm)
        log_id = await manager.create_log(**_log_kwargs(org, user, client_id=client))
        [full] = await manager.list_logs(organization_id=org, client_id=client)
        [light] = await manager.list_logs(
            organization_id=org, client_id=client, include_trace=False
        )
        assert full.id == light.id == log_id
        assert full.per_type == {
            "procedural": {"returned_ids": ["s1"], "candidates": []}
        }
        assert light.per_type is None
        assert light.model_dump(exclude={"per_type"}) == full.model_dump(
            exclude={"per_type"}
        )

    def test_the_client_listing_is_indexed(self):
        from mirix.orm.retrieval_log import RetrievalLog

        indexes = {
            index.name: [c.name for c in index.columns]
            for index in RetrievalLog.__table__.indexes
        }
        assert indexes["ix_retrieval_log_org_client_created"] == [
            "organization_id",
            "client_id",
            "created_at",
        ]


class TestPurger:
    async def test_runs_at_start_then_every_interval_and_stops(self):
        naps = []
        calls = []

        class Manager:
            async def purge_expired(self):
                calls.append(1)
                return 0

        async def fake_sleep(seconds):
            naps.append(seconds)
            await asyncio.sleep(0)

        purger = RetrievalLogPurger(
            interval_seconds=3600, manager=Manager(), sleep=fake_sleep
        )
        task = purger.start()
        assert purger.start() is task  # idempotent
        while purger.passes < 3:
            await asyncio.sleep(0.01)
        assert purger.running and set(naps) == {3600}
        await purger.stop()
        assert task.done() and not purger.running
        await purger.stop()  # twice is fine

    async def test_a_failing_pass_does_not_stop_the_loop(self):
        calls = []

        class Manager:
            async def purge_expired(self):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("database hiccup")
                return 0

        async def fake_sleep(_seconds):
            await asyncio.sleep(0)

        purger = RetrievalLogPurger(
            interval_seconds=1, manager=Manager(), sleep=fake_sleep
        )
        purger.start()
        while len(calls) < 3:
            await asyncio.sleep(0.01)
        await purger.stop()

    async def test_disabled_interval_does_not_start(self):
        purger = RetrievalLogPurger(interval_seconds=0, manager=object())
        assert purger.start() is None and not purger.running

    async def test_purges_the_real_table(self, sm, scope):
        org, (user, _) = scope
        client = f"client-{uuid.uuid4().hex[:6]}"
        clock = Clock(T0 - dt.timedelta(days=60))
        manager = RetrievalLogManager(session_maker=sm, clock=clock)
        old = await manager.create_log(**_log_kwargs(org, user, client_id=client))
        clock.now = T0
        purger = RetrievalLogPurger(interval_seconds=3600, manager=manager)
        assert await purger.run_once() >= 1
        assert await _all(sm, id=old) == []


class TestLifespanHelpers:
    async def test_a_failing_purger_never_blocks_startup_or_shutdown(self, monkeypatch):
        from mirix.server import rest_api
        from mirix.services import retrieval_log_manager

        class Broken:
            def start(self):
                raise RuntimeError("no loop")

        monkeypatch.setattr(retrieval_log_manager, "RetrievalLogPurger", Broken)
        assert rest_api.start_retrieval_log_purger() is None
        await rest_api.stop_retrieval_log_purger(None)

        class BrokenStop:
            async def stop(self):
                raise RuntimeError("already gone")

        await rest_api.stop_retrieval_log_purger(BrokenStop())  # logs, no raise
