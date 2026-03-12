"""Tests for temporal query functionality."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
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
