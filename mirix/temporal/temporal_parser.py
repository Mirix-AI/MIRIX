"""
Temporal expression parser for converting natural language dates to date ranges.

This module provides functionality to parse temporal expressions from natural language
queries (e.g., "today", "yesterday", "last week") and convert them to concrete datetime ranges.
"""

import re
from datetime import datetime, timedelta
from typing import Optional


class TemporalRange:
    """Represents a date/time range for filtering."""

    def __init__(self, start: Optional[datetime] = None, end: Optional[datetime] = None):
        """
        Initialize a temporal range.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
        """
        self.start = start
        self.end = end

    def __repr__(self):
        return f"TemporalRange(start={self.start}, end={self.end})"

    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }


def parse_temporal_expression(text: str, reference_time: Optional[datetime] = None) -> Optional[TemporalRange]:
    """
    Parse temporal expressions from natural language text.

    This function identifies temporal expressions like "today", "yesterday", "last week"
    and converts them to concrete datetime ranges. The parsing is case-insensitive and
    looks for whole word matches.

    Args:
        text: Input text containing temporal expressions
        reference_time: Reference datetime for relative expressions (defaults to current time)

    Returns:
        TemporalRange object or None if no temporal expression found

    Examples:
        >>> parse_temporal_expression("What happened today?")
        TemporalRange(start=2025-11-19 00:00:00, end=2025-11-19 23:59:59)

        >>> parse_temporal_expression("Show me yesterday's events")
        TemporalRange(start=2025-11-18 00:00:00, end=2025-11-18 23:59:59)

        >>> parse_temporal_expression("What did I do last week?")
        TemporalRange(start=2025-11-12 00:00:00, end=2025-11-19 23:59:59)

    Supported expressions:
        - "today": Current day from 00:00:00 to 23:59:59
        - "yesterday": Previous day
        - "last N days": Previous N days including today
        - "last week": Previous 7 days
        - "this week": From Monday of current week to now
        - "last month": Previous 30 days
        - "this month": From 1st of current month to now
    """
    if reference_time is None:
        reference_time = datetime.now()

    text_lower = text.lower()

    # Today
    if re.search(r"\btoday\b", text_lower):
        start = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Yesterday
    if re.search(r"\byesterday\b", text_lower):
        yesterday = reference_time - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Last N days
    match = re.search(r"\blast\s+(\d+)\s+days?\b", text_lower)
    if match:
        days = int(match.group(1))
        start = (reference_time - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Last week
    if re.search(r"\blast\s+week\b", text_lower):
        start = (reference_time - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # This week (from Monday of current week to now)
    if re.search(r"\bthis\s+week\b", text_lower):
        days_since_monday = reference_time.weekday()
        start = (reference_time - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Last month (approximate: 30 days)
    if re.search(r"\blast\s+month\b", text_lower):
        start = (reference_time - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # This month (from 1st of current month to now)
    if re.search(r"\bthis\s+month\b", text_lower):
        start = reference_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Last N weeks
    match = re.search(r"\blast\s+(\d+)\s+weeks?\b", text_lower)
    if match:
        weeks = int(match.group(1))
        start = (reference_time - timedelta(weeks=weeks)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # Last N months (approximate: N * 30 days)
    match = re.search(r"\blast\s+(\d+)\s+months?\b", text_lower)
    if match:
        months = int(match.group(1))
        start = (reference_time - timedelta(days=months * 30)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = reference_time.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TemporalRange(start, end)

    # No temporal expression found
    return None
