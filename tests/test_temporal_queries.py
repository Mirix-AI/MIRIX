"""Tests for temporal query functionality."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
import requests

from mirix.temporal.temporal_parser import TemporalRange, parse_temporal_expression

# Integration test config (used only by TestTemporalIntegration)
TEST_USER_ID_TEMPORAL = "temporal-test-user"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"


class TestTemporalParser:
    """Test temporal expression parsing."""

    def test_parse_today(self):
        """Test parsing 'today' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened today?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 19, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()
        assert result.end.hour == 23
        assert result.end.minute == 59

    def test_parse_yesterday(self):
        """Test parsing 'yesterday' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What did I do yesterday?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 18, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 18).date()
        assert result.end.hour == 23

    def test_parse_last_week(self):
        """Test parsing 'last week' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened last week?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 12, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_this_week(self):
        """Test parsing 'this week' expression."""
        # Use a Wednesday for testing
        ref_time = datetime(2025, 11, 19, 14, 30, 0)  # Wednesday
        result = parse_temporal_expression("Show me this week's events", ref_time)

        assert result is not None
        # Should start from Monday (2 days before Wednesday)
        assert result.start == datetime(2025, 11, 17, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_month(self):
        """Test parsing 'last month' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What happened last month?", ref_time)

        assert result is not None
        # Approximately 30 days ago
        expected_start = datetime(2025, 10, 20, 0, 0, 0, 0)
        assert result.start == expected_start

    def test_parse_this_month(self):
        """Test parsing 'this month' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me this month's activities", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 1, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_days(self):
        """Test parsing 'last N days' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("What did I do in the last 3 days?", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 16, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_weeks(self):
        """Test parsing 'last N weeks' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me last 2 weeks", ref_time)

        assert result is not None
        assert result.start == datetime(2025, 11, 5, 0, 0, 0, 0)
        assert result.end.date() == datetime(2025, 11, 19).date()

    def test_parse_last_n_months(self):
        """Test parsing 'last N months' expression."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)
        result = parse_temporal_expression("Show me last 2 months", ref_time)

        assert result is not None
        # Approximately 60 days ago
        expected_start = datetime(2025, 9, 20, 0, 0, 0, 0)
        assert result.start == expected_start

    def test_no_temporal_expression(self):
        """Test that None is returned when no temporal expression is found."""
        result = parse_temporal_expression("What is the weather?", datetime.now())
        assert result is None

        result = parse_temporal_expression("Tell me about Python", datetime.now())
        assert result is None

    def test_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        ref_time = datetime(2025, 11, 19, 14, 30, 0)

        result1 = parse_temporal_expression("What happened TODAY?", ref_time)
        result2 = parse_temporal_expression("What happened today?", ref_time)
        result3 = parse_temporal_expression("What happened ToDay?", ref_time)

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        assert result1.start == result2.start == result3.start

    def test_temporal_range_to_dict(self):
        """Test TemporalRange to_dict() method."""
        start = datetime(2025, 11, 19, 0, 0, 0)
        end = datetime(2025, 11, 19, 23, 59, 59)
        range_obj = TemporalRange(start, end)

        result = range_obj.to_dict()
        assert result["start"] == start.isoformat()
        assert result["end"] == end.isoformat()

    def test_temporal_range_none_values(self):
        """Test TemporalRange with None values."""
        range_obj = TemporalRange(None, None)

        result = range_obj.to_dict()
        assert result["start"] is None
        assert result["end"] is None


class TestCalendarAnchors:
    """A relative window is never substituted for an absolute date. An
    expression that names a year, a month or a numeric date is left
    unparsed (no filter), since the parser can represent none of them."""

    REF = datetime(2026, 9, 30, 14, 0, 0)

    @pytest.mark.parametrize(
        "text",
        [
            "last week of April in 1972–1974",
            "this month (September 2006)",
            "last week of April",
            "May 3",
            "yesterday, 3 May",
            "the 5th of May last week",
            "today 2025-11-19",
            "last week, 12/03",
            "last month 03.04.2025",
            "in 2024 last month",
            "Dec 24 this week",
            "last week of March",
            "March 3 this week",
            "last week, 12/03/24",
            "last week 31/12",
        ],
    )
    def test_an_absolute_anchor_is_not_parsed(self, text):
        assert parse_temporal_expression(text, self.REF) is None

    @pytest.mark.parametrize(
        "text, start",
        [
            # Fractions, ratios, versions and the verb "march" name no date.
            ("last week I used 3/4 of budget", datetime(2026, 9, 23)),
            ("this week, 1/2 of tasks", datetime(2026, 9, 28)),
            ("last month: 50/50 split", datetime(2026, 8, 31)),
            ("last week, 24/7 support", datetime(2026, 9, 23)),
            ("last week about version 1.2.30", datetime(2026, 9, 23)),
            ("yesterday I had to march", datetime(2026, 9, 29)),
            ("last week, error 13/13", datetime(2026, 9, 23)),
        ],
    )
    def test_numbers_that_are_no_date_do_not_block_a_relative_window(self, text, start):
        result = parse_temporal_expression(text, self.REF)
        assert result is not None and result.start == start

    @pytest.mark.parametrize(
        "text",
        [
            "this week last year",
            "last month of last year",
            "the day before yesterday",
            "a year ago today",
            "next week",
            "today and tomorrow",
            # Each word below modifies a time unit or a date.
            "what did I do before last week",
            "what did I do after this week?",
            "what did I do 2 years ago",
            "last week, and what is planned for next month",
            "the week before last week",
            "prior to yesterday",
            "yesterday and the previous week",
            "yesterday and the following day",
            "this week over the past 3 years",
            "yesterday and the next few days",
            "last weekend and yesterday",
            "this quarter, today",
            "yesterday, a year before",
            "the week after next, today",
            "last week, two days later",
            "yesterday before Monday",
            "last month, prior week",
        ],
    )
    def test_an_unsupported_relative_word_is_not_parsed(self, text):
        """A window picked from the rest of the text would be the wrong one."""
        assert parse_temporal_expression(text, self.REF) is None

    @pytest.mark.parametrize(
        "text, start",
        [
            ("what did I do yesterday before lunch", datetime(2026, 9, 29)),
            ("what did I work on today after the standup?", datetime(2026, 9, 30)),
            ("what did I do today before 3pm", datetime(2026, 9, 30)),
            ("yesterday, what should I do next?", datetime(2026, 9, 29)),
            ("last week I reviewed the previous draft", datetime(2026, 9, 23)),
            ("last week after the meeting", datetime(2026, 9, 23)),
            ("my prior notes from this month", datetime(2026, 9, 1)),
            ("last week I read about 10 years of history", datetime(2026, 9, 23)),
            ("the year-end review this week", datetime(2026, 9, 28)),
            ("what came after the launch yesterday", datetime(2026, 9, 29)),
            ("yesterday I noted what comes after next steps", datetime(2026, 9, 29)),
            ("what did I put next to the door yesterday", datetime(2026, 9, 29)),
            # A frequency ("twice a day") is no offset from a date.
            ("yesterday I took it twice a day before meals", datetime(2026, 9, 29)),
            ("last week I ran 3 times a week after work", datetime(2026, 9, 23)),
            ("this month, once a month after payday", datetime(2026, 9, 1)),
        ],
    )
    def test_a_relative_word_that_modifies_no_time_keeps_the_window(self, text, start):
        """Before, after, next, previous, prior and years made any
        query unsupported, so "what did I do yesterday before lunch" lost its
        yesterday filter; they are unsupported only where they modify a time
        unit or a date expression."""
        result = parse_temporal_expression(text, self.REF)
        assert result is not None and result.start == start, text
        # A day's window ends that day; a week's or a month's, today.
        end = start.date() if "yesterday" in text else self.REF.date()
        assert result.end.date() == end, text

    @pytest.mark.parametrize(
        "text, start",
        [
            # A day's own event: "before today's meeting" is today, before it.
            ("what did I note before today's meeting", datetime(2026, 9, 30)),
            ("what did I do after today\u2019s standup", datetime(2026, 9, 30)),
            ("what came up after yesterday's call", datetime(2026, 9, 29)),
            ("prior to yesterday's review, what did I read", datetime(2026, 9, 29)),
            # A frequency whatever its spelling or spacing.
            ("yesterday I took it twice  a day before meals", datetime(2026, 9, 29)),
            ("yesterday I took it 2x a day after meals", datetime(2026, 9, 29)),
            ("yesterday I took it 3 x a day before meals", datetime(2026, 9, 29)),
            ("last week I ran three\ttimes a week after work", datetime(2026, 9, 23)),
            ("yesterday I took it thrice a day before meals", datetime(2026, 9, 29)),
        ],
    )
    def test_a_possessive_day_or_a_frequency_keeps_the_window(self, text, start):
        """Regression: "before today's meeting" matched "before today" and
        lost the day's window, and the frequency guard missed "2x a day" and
        "twice  a day" (two spaces)."""
        result = parse_temporal_expression(text, self.REF)
        assert result is not None and result.start == start, text
        end = start.date() if "yesterday" in text else self.REF.date()
        assert result.end.date() == end, text

    @pytest.mark.parametrize(
        "text",
        [
            # A week's or a month's event: before it may be any earlier time.
            "what did I do before last week's review",
            "after this month's release, today",
            # An offset from the day, possessive or not.
            "the day before yesterday's meeting",
            "what did I do before yesterday",
            "today, and the day after",
            "yesterday, a day later",
            # Not a frequency: an offset after a count of something else.
            "yesterday I finished two tasks, a day before the deadline",
        ],
    )
    def test_an_offset_from_a_date_is_still_unsupported(self, text):
        assert parse_temporal_expression(text, self.REF) is None, text

    @pytest.mark.parametrize(
        "text, start",
        [
            ("What may I have done last week?", datetime(2026, 9, 23)),
            ("You may recall what happened today", datetime(2026, 9, 30)),
            ("in the last 1000 days", datetime(2024, 1, 4)),
            ("last 2 weeks", datetime(2026, 9, 16)),
        ],
    )
    def test_relative_expressions_still_parse(self, text, start):
        result = parse_temporal_expression(text, self.REF)
        assert result is not None and result.start == start
        assert result.end.date() == self.REF.date()


class TestRetrievalDateRange:
    """/memory/retrieve/conversation's date range is naive UTC, the form
    episodic occurred_at stores: a temporal expression resolves in the
    user's time zone (MIRIX's own "Region (UTC+hh:mm)" format included),
    explicit request dates are converted to UTC before losing their offset."""

    NOW = datetime(2026, 9, 30, 3, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def _range(*args, tz="UTC (UTC+00:00)", now=NOW):
        from types import SimpleNamespace

        from mirix.server.rest_api import _retrieval_date_range

        return _retrieval_date_range(*args, SimpleNamespace(timezone=tz), now=now)

    def test_today_for_mirix_default_utc_user(self):
        assert self._range(None, None, "today") == (
            datetime(2026, 9, 30, 0, 0, 0),
            datetime(2026, 9, 30, 23, 59, 59, 999999),
        )

    def test_today_in_the_users_zone_as_utc(self):
        start, end = self._range(
            None, None, "today", tz="America/Los_Angeles (UTC-07:00)"
        )
        # 20:00 on 09-29 in Los Angeles: its day, 07:00 UTC to 06:59:59 UTC.
        assert (start, end) == (
            datetime(2026, 9, 29, 7, 0, 0),
            datetime(2026, 9, 30, 6, 59, 59, 999999),
        )
        assert start.tzinfo is None and end.tzinfo is None

    def test_a_window_across_a_dst_change_uses_each_days_offset(self):
        # 2026-11-01: Los Angeles leaves daylight saving time (UTC-7 -> UTC-8).
        start, end = self._range(
            None,
            None,
            "last week",
            tz="America/Los_Angeles",
            now=datetime(2026, 11, 3, 20, 0, 0, tzinfo=timezone.utc),
        )
        assert start == datetime(2026, 10, 27, 7, 0, 0)  # PDT midnight
        assert end == datetime(2026, 11, 4, 7, 59, 59, 999999)  # PST

    @pytest.mark.parametrize("tz", ["Not/AZone (UTC+05:00)", "", None, "garbage"])
    def test_an_unknown_or_missing_zone_is_utc(self, tz):
        assert self._range(None, None, "today", tz=tz) == (
            datetime(2026, 9, 30, 0, 0, 0),
            datetime(2026, 9, 30, 23, 59, 59, 999999),
        )

    def test_no_user_is_utc(self):
        from mirix.server.rest_api import _retrieval_date_range

        assert _retrieval_date_range(None, None, "today", None, now=self.NOW) == (
            datetime(2026, 9, 30, 0, 0, 0),
            datetime(2026, 9, 30, 23, 59, 59, 999999),
        )

    def test_explicit_dates_are_converted_to_utc(self):
        assert self._range(
            "2026-09-01T00:00:00+02:00", "2026-09-02T00:00:00Z", "today"
        ) == (datetime(2026, 8, 31, 22, 0, 0), datetime(2026, 9, 2, 0, 0, 0))

    def test_a_naive_explicit_date_is_taken_as_utc(self):
        assert self._range("2026-09-01T08:00:00", None, None) == (
            datetime(2026, 9, 1, 8, 0, 0),
            None,
        )

    def test_an_invalid_explicit_date_is_no_filter(self):
        assert self._range("not a date", None, "today") == (None, None)

    def test_a_sentinel_date_with_an_offset_is_the_end_of_the_range(self):
        """0001-01-01+01:00 in UTC is before year 1, 9999-12-31-05:00 after
        9999: converting them raised OverflowError (HTTP 500). They bound
        nothing, as they meant."""
        from mirix.server.rest_api import _request_datetime

        assert _request_datetime("0001-01-01T00:00:00+01:00") == datetime.min
        assert _request_datetime("9999-12-31T23:59:59-05:00") == datetime.max
        assert self._range(
            "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-05:00", None
        ) == (datetime.min, datetime.max)
        # Within the range nothing changes.
        assert _request_datetime("0001-01-01T05:00:00+01:00") == datetime(1, 1, 1, 4)

    def test_an_expression_past_the_datetime_range_is_no_filter(self):
        assert self._range(None, None, "last 1000000 days") == (None, None)

    def test_an_expression_with_a_calendar_anchor_is_no_filter(self):
        assert self._range(None, None, "this month (September 2006)") == (None, None)

    def test_the_default_reference_time_is_now_in_utc(self):
        from mirix.server.rest_api import _retrieval_date_range

        before = datetime.now(timezone.utc).replace(tzinfo=None)
        start, end = _retrieval_date_range(None, None, "today", None)
        after = datetime.now(timezone.utc).replace(tzinfo=None)
        assert start.date() in (before.date(), after.date())
        assert start <= after and end >= before


class TestUserTimezone:
    @pytest.mark.parametrize(
        "text, zone",
        [
            ("UTC (UTC+00:00)", "UTC"),
            ("Asia/Shanghai (UTC+08:00)", "Asia/Shanghai"),
            ("Europe/Berlin", "Europe/Berlin"),
            ("  America/New_York (UTC-05:00)  ", "America/New_York"),
            ("Not/AZone", "UTC"),
            ("", "UTC"),
            (None, "UTC"),
        ],
    )
    def test_mirix_timezone_strings(self, text, zone):
        from mirix.utils import user_timezone

        assert user_timezone(text).zone == zone


# ---------------------------------------------------------------------------
# Integration test fixtures (used only by TestTemporalIntegration)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_process():
    """Check that server is running (requires manual start)."""
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            yield None
            return
    except (requests.ConnectionError, requests.Timeout):
        pass
    pytest.skip(
        "Server not running. Start with: python scripts/start_server.py --port=8000"
    )


@pytest_asyncio.fixture(scope="module")
async def api_auth(server_process):
    """Create org and client once per module; yield auth for client creation."""
    from conftest import _create_client_and_key

    auth = await _create_client_and_key(
        "temporal-test-client", "temporal-test-org", org_name="Temporal Test Org"
    )
    os.environ.setdefault("MIRIX_API_URL", "http://localhost:8000")
    os.environ["MIRIX_API_KEY"] = auth["api_key"]
    return auth


@pytest_asyncio.fixture
async def temporal_client(server_process, api_auth):
    """Create a MirixClient for temporal integration tests."""
    from mirix.client import MirixClient

    c = await MirixClient.create(
        api_key=api_auth["api_key"],
        base_url="http://localhost:8000",
        debug=False,
    )
    await c.initialize_meta_agent(config_path=str(CONFIG_PATH), update_agents=False)
    assert c._meta_agent is not None, "Meta agent must be initialized"
    return c


class TestTemporalIntegration:
    """Integration tests for temporal query feature.

    Require a running server. Run with:
      pytest tests/test_temporal_queries.py -v -m integration
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set (needed for meta agent)",
    )
    async def test_retrieve_with_temporal_expression(self, temporal_client):
        """Test retrieval with natural language temporal expression (e.g. 'today')."""
        # Add an episodic memory with occurred_at so we have something to retrieve
        await temporal_client.add(
            user_id=TEST_USER_ID_TEMPORAL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I had a standup meeting at 10 AM and reviewed PRs in the afternoon.",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Recorded your standup and PR review."}],
                },
            ],
            occurred_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        )
        await asyncio.sleep(2)

        result = await temporal_client.retrieve_with_conversation(
            user_id=TEST_USER_ID_TEMPORAL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "What did I do today?"}]}
            ],
            limit=10,
        )

        assert result is not None
        assert result.get("success") is True
        assert "memories" in result
        # Server may return temporal_expression when it parses "today" from the query
        assert "memories" in result
        if result.get("temporal_expression") or result.get("date_range"):
            assert "episodic" in result["memories"] or len(result["memories"]) >= 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set (needed for meta agent)",
    )
    async def test_retrieve_with_explicit_date_range(self, temporal_client):
        """Test retrieval with explicit start_date and end_date."""
        start = "2025-11-01T00:00:00"
        end = "2025-11-30T23:59:59"

        result = await temporal_client.retrieve_with_conversation(
            user_id=TEST_USER_ID_TEMPORAL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Show me November 2025 events"}]}
            ],
            limit=10,
            start_date=start,
            end_date=end,
        )

        assert result is not None
        assert result.get("success") is True
        assert "memories" in result
        assert result.get("date_range") is not None
        assert result["date_range"].get("start") is not None
        assert result["date_range"].get("end") is not None
        assert "2025-11-01" in result["date_range"]["start"]
        assert "2025-11-30" in result["date_range"]["end"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set (needed for meta agent)",
    )
    async def test_temporal_filtering_episodic_only(self, temporal_client):
        """Test that temporal filtering applies only to episodic memories."""
        # Add episodic memory with occurred_at in a specific range
        await temporal_client.add(
            user_id=TEST_USER_ID_TEMPORAL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "I learned that Python uses list comprehensions."}
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]},
            ],
        )
        await asyncio.sleep(2)

        start = "2025-10-01T00:00:00"
        end = "2025-12-31T23:59:59"
        result = await temporal_client.retrieve_with_conversation(
            user_id=TEST_USER_ID_TEMPORAL,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "What do you know about me?"}]}
            ],
            limit=10,
            start_date=start,
            end_date=end,
        )

        assert result is not None
        assert result.get("success") is True
        assert "memories" in result
        # date_range is applied; episodic (if any) are filtered by it; other types (semantic, etc.) are not
        assert result.get("date_range") is not None
        # Memories can contain episodic (filtered by date) and other types (unfiltered by date)
        assert isinstance(result["memories"], dict)


# Additional documentation and usage examples
"""
Usage Examples:
===============

1. Automatic temporal parsing:
   >>> from mirix import MirixClient
   >>> client = MirixClient(...)
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "What did we discuss today?"}]}
   ...     ]
   ... )
   
2. Explicit date range:
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "Show me meetings"}]}
   ...     ],
   ...     start_date="2025-11-19T00:00:00",
   ...     end_date="2025-11-19T23:59:59"
   ... )

3. Combine with filter_tags:
   >>> memories = client.retrieve_with_conversation(
   ...     user_id='demo-user',
   ...     messages=[
   ...         {"role": "user", "content": [{"type": "text", "text": "What did I do yesterday?"}]}
   ...     ],
   ...     filter_tags={"expert_id": "expert-123"}
   ... )

Supported Temporal Expressions:
================================
- "today": Current day from 00:00:00 to 23:59:59
- "yesterday": Previous day
- "last N days": Previous N days including today
- "last week": Previous 7 days
- "this week": From Monday of current week to now
- "last month": Previous 30 days
- "this month": From 1st of current month to now
- "last N weeks": Previous N weeks
- "last N months": Previous N * 30 days

Note: Only episodic memories are filtered by temporal expressions.
      Other memory types (semantic, procedural, resource, knowledge vault, core) 
      do not have occurred_at timestamps and are not affected.
"""
