"""Tests for temporal query functionality."""

import pytest
from datetime import datetime, timedelta
from mirix.temporal.temporal_parser import parse_temporal_expression, TemporalRange


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


class TestTemporalIntegration:
    """Integration tests for temporal query feature."""
    
    # Note: These tests require a running server and database
    # They are marked with @pytest.mark.integration to skip in unit test runs
    
    @pytest.mark.integration
    def test_retrieve_with_temporal_expression(self):
        """Test retrieval with natural language temporal expression."""
        # This would test the full flow from client to database
        # Skip for now as it requires full setup
        pytest.skip("Integration test - requires running server and full setup")
    
    @pytest.mark.integration
    def test_retrieve_with_explicit_date_range(self):
        """Test retrieval with explicit start_date and end_date."""
        pytest.skip("Integration test - requires running server and full setup")
    
    @pytest.mark.integration
    def test_temporal_filtering_episodic_only(self):
        """Test that temporal filtering only affects episodic memories."""
        pytest.skip("Integration test - requires running server and full setup")


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

