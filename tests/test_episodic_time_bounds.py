"""Time bounds of episodic-memory reads and the time-range tool.

``episodic_memory.occurred_at`` is a naive ``DateTime`` holding UTC; only the
write path used to normalize to it. A read bound given as an AWARE datetime
(the time-range tool's converted arguments, the REST temporal filter's
``...Z`` dates) was bound as its wall-clock time: SQLite missed events by the
offset, and the SQLite BM25 fallback raised TypeError comparing naive with
aware. Every read bound is now normalized to naive UTC at the manager
boundary (:func:`mirix.utils.to_naive_utc`).

The tool ``list_memory_within_timerange`` accepted only one time format and
returned None for every memory type but episodic, "all" included; it now
parses ISO 8601 and bare dates too (:func:`mirix.utils.parse_datetime_arg`),
answers "all" with the episodic timeline (the only time-indexed memory) and
refuses the other types explicitly.

Offline: a throwaway SQLite database, no model, no network.
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import mirix.orm  # noqa: F401 -- register every ORM class before create_all
from mirix.orm.base import Base
from mirix.orm.episodic_memory import EpisodicEvent
from mirix.orm.organization import Organization as OrganizationORM
from mirix.orm.user import User as UserORM
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.user import User as PydanticUser
from mirix.settings import settings

pytestmark = pytest.mark.asyncio(loop_scope="module")

SHANGHAI = timezone(timedelta(hours=8))


@pytest.fixture(autouse=True)
def _force_sqlite_branch(monkeypatch):
    monkeypatch.setattr(
        type(settings), "mirix_pg_uri_no_default", property(lambda self: None)
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sm(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("episodic_time_bounds") / "test.db"
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


def _agent_state() -> AgentState:
    return AgentState(
        id=f"agent-{uuid.uuid4()}",
        name="episodic",
        system="s",
        agent_type=AgentType.episodic_memory_agent,
        llm_config=LLMConfig(
            model="x", model_endpoint_type="openai", context_window=8000
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="http://127.0.0.1:9/v1",
            embedding_model="x",
            embedding_dim=8,
        ),
        tools=[],
    )


@pytest_asyncio.fixture(loop_scope="module")
async def world(sm):
    from mirix.services.episodic_memory_manager import EpisodicMemoryManager

    org = f"org-{uuid.uuid4().hex[:8]}"
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    async with sm() as db:
        db.add(OrganizationORM(id=org, name=org))
        await db.flush()
        db.add(
            UserORM(
                id=user_id,
                name="u",
                organization_id=org,
                status="active",
                timezone="UTC",
            )
        )
        # Stored as the write path stores it: naive UTC.
        db.add(
            EpisodicEvent(
                id=f"ep-{uuid.uuid4().hex[:8]}",
                organization_id=org,
                user_id=user_id,
                occurred_at=datetime(2025, 1, 1, 12, 0, 0),
                actor="user",
                event_type="activity",
                summary="noon meeting about the budget",
                details="the budget meeting at noon UTC",
                last_modify={
                    "timestamp": "2025-01-01T12:00:00+00:00",
                    "operation": "created",
                },
            )
        )
        await db.commit()
    manager = EpisodicMemoryManager()
    manager.session_maker = sm
    user = PydanticUser(id=user_id, name="u", organization_id=org, timezone="UTC")
    return SimpleNamespace(manager=manager, user=user, agent_state=_agent_state())


class TestToNaiveUtc:
    def test_aware_values_become_naive_utc_and_naive_values_stay(self):
        from mirix.utils import to_naive_utc

        assert to_naive_utc(datetime(2025, 1, 1, 20, 0, tzinfo=SHANGHAI)) == datetime(
            2025, 1, 1, 12, 0
        )
        naive = datetime(2025, 1, 1, 12, 0)
        assert to_naive_utc(naive) is naive
        assert to_naive_utc(None) is None


class TestManagerBounds:
    async def test_an_aware_window_finds_the_event_it_contains(self, world):
        found = await world.manager.list_episodic_memory_around_timestamp(
            agent_state=world.agent_state,
            start_time=datetime(2025, 1, 1, 19, 0, tzinfo=SHANGHAI),  # 11:00 UTC
            end_time=datetime(2025, 1, 1, 21, 0, tzinfo=SHANGHAI),  # 13:00 UTC
            user=world.user,
        )
        assert [e.summary for e in found] == ["noon meeting about the budget"]
        missed = await world.manager.list_episodic_memory_around_timestamp(
            agent_state=world.agent_state,
            start_time=datetime(2025, 1, 1, 21, 30, tzinfo=SHANGHAI),
            end_time=datetime(2025, 1, 1, 23, 0, tzinfo=SHANGHAI),
            user=world.user,
        )
        assert missed == []

    async def test_the_statement_is_bound_with_naive_datetimes(
        self, world, monkeypatch
    ):
        captured = []

        class _Result:
            def scalars(self):
                return SimpleNamespace(all=lambda: [])

        class _Session:
            async def execute(self, statement):
                captured.append(statement.compile().params)
                return _Result()

        @asynccontextmanager
        async def fake_session():
            yield _Session()

        monkeypatch.setattr(world.manager, "session_maker", fake_session)
        await world.manager.list_episodic_memory_around_timestamp(
            agent_state=world.agent_state,
            start_time=datetime(2025, 1, 1, 19, 0, tzinfo=SHANGHAI),
            end_time=datetime(2025, 1, 1, 21, 0, tzinfo=SHANGHAI),
            user=world.user,
        )
        [params] = captured
        bounds = [v for v in params.values() if isinstance(v, datetime)]
        assert bounds == [datetime(2025, 1, 1, 11, 0), datetime(2025, 1, 1, 13, 0)]

    @pytest.mark.parametrize(
        "method, query", [("bm25", "budget"), ("string_match", "budget"), ("bm25", "")]
    )
    async def test_list_episodic_memory_filters_on_aware_dates(
        self, world, method, query
    ):
        """The recent-events branch, the SQL branch and the SQLite BM25
        fallback's Python comparison (a TypeError before) agree."""
        inside = await world.manager.list_episodic_memory(
            agent_state=world.agent_state,
            user=world.user,
            query=query,
            search_field="details",
            search_method=method,
            use_cache=False,
            start_date=datetime(2025, 1, 1, 19, 0, tzinfo=SHANGHAI),
            end_date=datetime(2025, 1, 1, 21, 0, tzinfo=SHANGHAI),
        )
        assert [e.summary for e in inside] == ["noon meeting about the budget"]
        outside = await world.manager.list_episodic_memory(
            agent_state=world.agent_state,
            user=world.user,
            query=query,
            search_field="details",
            search_method=method,
            use_cache=False,
            start_date=datetime(2025, 1, 1, 21, 30, tzinfo=SHANGHAI),
            end_date=datetime(2025, 1, 2, 23, 0, tzinfo=SHANGHAI),
        )
        assert outside == []

    @pytest.mark.parametrize("query", ["", "budget"])
    async def test_the_cache_search_gets_naive_utc_bounds(
        self, world, monkeypatch, query
    ):
        """The Redis cache compares ``bound.timestamp()`` with
        ``occurred_at_ts``, written as ``occurred_at.timestamp()`` of the naive
        UTC column value (``SqlalchemyBase``): only a naive UTC bound maps to
        the same scale whatever the server's local timezone; an aware bound's
        epoch is off by that zone's offset. The bounds are normalized before
        the cache is asked, as they are before the SQL branches."""
        from mirix.database import redis_client as redis_module

        seen = []

        class _Cache:
            EPISODIC_INDEX = "episodic"

            async def _search(self, **kwargs):
                seen.append((kwargs.get("start_date"), kwargs.get("end_date")))
                return []  # a miss: the database answers

            search_recent = search_text = search_vector = _search

        monkeypatch.setattr(redis_module, "get_redis_client", lambda: _Cache())
        found = await world.manager.list_episodic_memory(
            agent_state=world.agent_state,
            user=world.user,
            query=query,
            search_field="details",
            search_method="bm25",
            start_date=datetime(2025, 1, 1, 19, 0, tzinfo=SHANGHAI),
            end_date=datetime(2025, 1, 1, 21, 0, tzinfo=SHANGHAI),
        )
        assert seen == [(datetime(2025, 1, 1, 11, 0), datetime(2025, 1, 1, 13, 0))]
        assert [e.summary for e in found] == ["noon meeting about the budget"]


class TestParseDatetimeArg:
    def test_the_accepted_forms(self):
        from mirix.utils import parse_datetime_arg

        assert parse_datetime_arg("2025-09-01T00:00:00Z", "UTC") == datetime(2025, 9, 1)
        assert parse_datetime_arg("2025-09-01T08:00:00+08:00", "UTC") == datetime(
            2025, 9, 1
        )
        # An explicit offset wins over the user's timezone.
        assert parse_datetime_arg(
            "2025-09-01T08:00:00+08:00", "America/New_York"
        ) == datetime(2025, 9, 1)
        assert parse_datetime_arg("2025-09-01", "UTC") == datetime(2025, 9, 1)
        assert parse_datetime_arg("2025-09-01", "UTC", end_of_day=True) == datetime(
            2025, 9, 1, 23, 59, 59, 999999
        )
        assert parse_datetime_arg(
            "2025-09-01 13:00:00", "Asia/Shanghai (UTC+08:00)"
        ) == datetime(2025, 9, 1, 5, 0)
        assert parse_datetime_arg(' "2025-09-01 13:00:00.250" ', None) == datetime(
            2025, 9, 1, 13, 0, 0, 250000
        )

    def test_iso_forms_parse_the_same_on_every_supported_python(self):
        """Python 3.10's fromisoformat rejects a short fraction, "+0800" and
        "+08"; pyproject allows 3.10. The value is made canonical first."""
        from mirix.utils import _canonical_iso, parse_datetime_arg

        assert parse_datetime_arg("2025-09-01T00:00:00.12Z", "UTC") == datetime(
            2025, 9, 1, 0, 0, 0, 120000
        )
        for offset in ("+0800", "+08", "+08:00"):
            assert parse_datetime_arg(
                f"2025-09-01T08:00:00{offset}", "UTC"
            ) == datetime(2025, 9, 1), offset
        assert parse_datetime_arg("2025-09-01T08:00-05:00", "UTC") == datetime(
            2025, 9, 1, 13, 0
        )
        assert parse_datetime_arg("2025-09-01 13:00", "UTC") == datetime(
            2025, 9, 1, 13, 0
        )
        assert _canonical_iso("2025-09-01T08:00:00.1234567Z") == (
            "2025-09-01T08:00:00.123456+00:00"
        )
        assert _canonical_iso("2025-09-01") is None
        assert _canonical_iso("2025-09-01T08:00:00 garbage") is None

    def test_a_zone_mirix_cannot_name_is_utc_not_an_error(self):
        """A user timezone such as "UTC+8" or "Beijing (UTC+08:00)" raised
        UnknownTimeZoneError on every time-range call and episodic insert."""
        from mirix.utils import convert_timezone_to_utc, parse_datetime_arg

        for zone in ("UTC+8", "Beijing (UTC+08:00)", "Not/AZone", ""):
            assert parse_datetime_arg("2025-09-01 13:00:00", zone) == datetime(
                2025, 9, 1, 13, 0
            ), zone
            assert convert_timezone_to_utc("2025-09-01 13:00:00", zone).hour == 13

    def test_a_sentinel_at_the_edge_of_the_range_is_that_end_of_the_range(self):
        """Models pass "0001-01-01 00:00:00" / "9999-12-31 23:59:59"
        as "all time" bounds. The zone's offset carried them past the datetime
        range and the call failed with an OverflowError, reported as MIRIX's
        own error. Such a bound is the end of the range it is next to: it
        bounds nothing, as it meant (as rest_api's date parser does)."""
        from mirix.utils import convert_timezone_to_utc, parse_datetime_arg

        # A start bound (a bare date is the start of its day) and an end bound
        # (a bare date is the end of its day), as list_memory_within_timerange
        # parses them.
        for value, zone, end_of_day, expected in (
            ("0001-01-01 00:00:00", "Asia/Shanghai (UTC+08:00)", False, datetime.min),
            ("0001-01-01", "Asia/Tokyo (UTC+09:00)", False, datetime.min),
            ("0001-01-01T00:00:00+05:00", None, False, datetime.min),
            ("9999-12-31 23:59:59", "America/New_York (UTC-05:00)", True, datetime.max),
            ("9999-12-31", "Etc/GMT+12", True, datetime.max),
            ("9999-12-31T23:00:00-02:00", "UTC", True, datetime.max),
        ):
            assert parse_datetime_arg(value, zone, end_of_day=end_of_day) == expected, (
                value
            )
        assert convert_timezone_to_utc(
            "9999-12-31 23:59:59", "America/New_York (UTC-05:00)"
        ) == datetime.max.replace(tzinfo=timezone.utc)
        # Near the edge without crossing it, the value is converted as usual.
        assert parse_datetime_arg(
            "0001-01-01 00:00:00", "America/New_York (UTC-05:00)"
        ) == datetime(1, 1, 1, 4, 56)
        assert parse_datetime_arg("9999-12-31 12:00:00", "Asia/Shanghai") == datetime(
            9999, 12, 31, 4, 0
        )

    def test_anything_else_names_the_accepted_forms(self):
        from mirix.utils import parse_datetime_arg

        for bad in ("string", '"%Y-%m-%d %H:%M:%S"', "", "yesterday"):
            with pytest.raises(ValueError) as raised:
                parse_datetime_arg(bad, "UTC")
            assert "YYYY-MM-DD HH:MM:SS" in str(raised.value) and "ISO 8601" in str(
                raised.value
            )

    def test_convert_timezone_to_utc_keeps_returning_aware_utc(self):
        from mirix.utils import convert_timezone_to_utc

        value = convert_timezone_to_utc(
            "2025-09-01 13:00:00", "Asia/Shanghai (UTC+08:00)"
        )
        assert value.tzinfo is not None
        assert value.replace(tzinfo=None) == datetime(2025, 9, 1, 5, 0)
        assert (
            convert_timezone_to_utc("2025-09-01T13:00:00Z", "Asia/Shanghai").hour == 13
        )


class _FakeEpisodic:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def list_episodic_memory_around_timestamp(self, **kwargs):
        self.calls.append(kwargs)
        return self.events


def _tool_agent(events):
    event = SimpleNamespace(
        id="ep-1",
        occurred_at=datetime(2025, 1, 1, 12, 0),
        event_type="activity",
        actor="user",
        summary="noon meeting",
    )
    return SimpleNamespace(
        user=SimpleNamespace(id="u"),
        agent_state=SimpleNamespace(id="a"),
        episodic_memory_manager=_FakeEpisodic([event] if events else []),
    )


class TestTimeRangeTool:
    async def test_all_and_episodic_return_the_episodic_timeline(self):
        from mirix.functions.function_sets.base import list_memory_within_timerange

        for memory_type in ("all", '"episodic"', " Episodic "):
            agent = _tool_agent(events=True)
            result = await list_memory_within_timerange(
                agent,
                memory_type,
                "2025-01-01T19:00:00+08:00",
                "2025-01-01 21:00:00",
                "Asia/Shanghai (UTC+08:00)",
            )
            assert isinstance(result, str), memory_type
            answer = json.loads(result)  # text the model reads, not a tuple repr
            assert answer["count"] == 1
            [item] = answer["results"]
            assert item["summary"] == "noon meeting"
            assert item["timestamp"] == "2025-01-01T12:00:00"
            assert "datetime.datetime" not in result
            [call] = agent.episodic_memory_manager.calls
            assert call["start_time"] == datetime(2025, 1, 1, 11, 0)
            assert call["end_time"] == datetime(2025, 1, 1, 13, 0)
            assert call["start_time"].tzinfo is None and call["end_time"].tzinfo is None

    async def test_other_types_and_bad_ranges_are_refused_explicitly(self):
        from mirix.functions.function_sets.base import list_memory_within_timerange

        agent = _tool_agent(events=True)
        with pytest.raises(ValueError, match="only episodic memory is time-indexed"):
            await list_memory_within_timerange(
                agent, "procedural", "2025-01-01", "2025-01-02", "UTC"
            )
        with pytest.raises(ValueError, match="after"):
            await list_memory_within_timerange(
                agent, "episodic", "2025-01-02", "2025-01-01", "UTC"
            )
        with pytest.raises(ValueError, match="ISO 8601"):
            await list_memory_within_timerange(
                agent, "episodic", '"%Y-%m-%d %H:%M:%S"', "2025-01-01", "UTC"
            )
        assert agent.episodic_memory_manager.calls == []

    async def test_search_in_memory_takes_quoted_or_padded_choices(self):
        """search_in_memory normalizes its choice arguments the same way
        (a model's '"Procedural"', ' BM25 ', 'NULL'); an empty search_field
        still reaches the manager as before (its own default field)."""
        from mirix.functions.function_sets.base import search_in_memory

        class _Procedural:
            def __init__(self):
                self.calls = []

            async def list_procedures(self, **kwargs):
                self.calls.append(kwargs)
                return []

        for search_field, expected in (("NULL", "description"), ("", "")):
            procedural = _Procedural()
            agent = SimpleNamespace(
                user=SimpleNamespace(id="u"),
                agent_state=SimpleNamespace(id="a"),
                procedural_memory_manager=procedural,
            )
            result = await search_in_memory(
                agent, '"Procedural"', "open a file", search_field, " BM25 ", "UTC"
            )
            assert result == ([], 0)
            [call] = procedural.calls
            assert call["search_method"] == "bm25"
            assert call["search_field"] == expected

    async def test_a_bare_end_date_covers_the_whole_day(self):
        from mirix.functions.function_sets.base import list_memory_within_timerange

        agent = _tool_agent(events=False)
        assert (
            await list_memory_within_timerange(
                agent, "episodic", "2025-01-01", "2025-01-01", "UTC"
            )
            == "No results found."
        )
        [call] = agent.episodic_memory_manager.calls
        assert call["end_time"] == datetime(2025, 1, 1, 23, 59, 59, 999999)


class TestTimeRangeToolTimestamps:
    """The tool's docstring (what the model reads) said its
    timestamps are UTC, but ``@update_timezone`` shows them in the user's
    timezone, with its offset; a user timezone pytz cannot name raised
    ``UnknownTimeZoneError`` there, while ``parse_datetime_arg`` reads the
    same string as UTC."""

    @staticmethod
    def _agent(world):
        return SimpleNamespace(
            user=world.user,
            agent_state=world.agent_state,
            episodic_memory_manager=world.manager,
        )

    async def test_the_timestamps_are_in_the_users_timezone_as_documented(self, world):
        from mirix.functions.function_sets.base import list_memory_within_timerange

        result = await list_memory_within_timerange(
            self._agent(world),
            "episodic",
            "2025-01-01 19:00:00",
            "2025-01-01 21:00:00",
            "Asia/Shanghai (UTC+08:00)",
        )
        [item] = json.loads(result)["results"]
        # Noon UTC, shown as local time with its offset.
        assert item["timestamp"] == "2025-01-01T20:00:00+08:00"
        doc = " ".join(list_memory_within_timerange.__doc__.split())
        assert "timestamps in UTC" not in doc
        assert "timestamps in your timezone, with its UTC offset" in doc

    @staticmethod
    def _warnings(monkeypatch):
        """The warnings of ``mirix.services.utils`` (the MIRIX logger does not
        propagate to caplog), with a fresh warn-once memory."""
        from mirix.services import utils

        warnings = []
        utils.display_timezone.cache_clear()
        monkeypatch.setattr(
            utils.logger, "warning", lambda msg, *args: warnings.append(msg % args)
        )
        return warnings

    async def test_an_unknown_user_timezone_is_utc_with_a_warning(
        self, world, monkeypatch
    ):
        from mirix.functions.function_sets.base import list_memory_within_timerange

        warnings = self._warnings(monkeypatch)
        result = await list_memory_within_timerange(
            self._agent(world),
            "episodic",
            "2025-01-01 11:00:00",
            "2025-01-01 13:00:00",
            "Beijing (UTC+08:00)",
        )
        [item] = json.loads(result)["results"]
        # The range was read as UTC (parse_datetime_arg) and so is the answer.
        assert item["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert len(warnings) == 1 and "'Beijing (UTC+08:00)'" in warnings[0]
        assert "UTC" in warnings[0]

    def test_localize_timestamps_falls_back_to_utc_and_warns_once(self, monkeypatch):
        from mirix.services.utils import localize_timestamps

        warnings = self._warnings(monkeypatch)

        def _row():
            return SimpleNamespace(
                occurred_at=datetime(2025, 1, 1, 12, 0),
                created_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                updated_at=None,
                last_modify={"timestamp": "2025-01-01T12:00:00Z"},
            )

        zone = "Nowhere/Land (UTC+05:00)"
        [row] = localize_timestamps([_row()], zone)
        localize_timestamps([_row()], zone)
        assert row.occurred_at == datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        assert row.occurred_at.utcoffset() == timedelta(0)
        assert row.last_modify["timestamp"].utcoffset() == timedelta(0)
        assert len(warnings) == 1 and "Nowhere/Land" in warnings[0]
        # A zone pytz names, in MIRIX's "Region (UTC+hh:mm)" form, still converts.
        [row] = localize_timestamps([_row()], "Asia/Shanghai (UTC+08:00)")
        assert row.occurred_at.isoformat() == "2025-01-01T20:00:00+08:00"

    def test_a_time_at_the_edge_of_the_range_is_shown_in_utc(self):
        """An "all time" sentinel is stored as the end of the
        datetime range (parse_datetime_arg). Shown in a zone on the far side
        of UTC it would pass that end, and the OverflowError failed the whole
        read; such a time is shown in UTC, the others still convert."""
        from mirix.services.utils import localize_timestamps

        edge = SimpleNamespace(
            occurred_at=datetime.max,
            created_at=datetime.min.replace(tzinfo=timezone.utc),
            updated_at=datetime(2025, 1, 1, 12, 0),
            last_modify={"timestamp": "9999-12-31T23:59:59Z"},
        )
        [row] = localize_timestamps([edge], "Asia/Shanghai (UTC+08:00)")
        assert row.occurred_at == datetime.max.replace(tzinfo=timezone.utc)
        assert row.occurred_at.utcoffset() == timedelta(0)
        assert row.last_modify["timestamp"].utcoffset() == timedelta(0)
        assert row.updated_at.isoformat() == "2025-01-01T20:00:00+08:00"
        # West of UTC the minimum passes its end instead.
        [row] = localize_timestamps(
            [SimpleNamespace(occurred_at=datetime.min, created_at=datetime.max)],
            "America/New_York (UTC-05:00)",
        )
        assert row.occurred_at == datetime.min.replace(tzinfo=timezone.utc)
        assert row.created_at.year == 9999 and row.created_at.hour == 18
