"""
Temporal expression parser for converting natural language dates to date ranges.

This module provides functionality to parse temporal expressions from natural language
queries (e.g., "today", "yesterday", "last week") and convert them to concrete datetime ranges.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

# Calendar anchors: parts of an expression that name an absolute date, which
# the relative patterns below cannot represent. "last week of April in
# 1972-1974" or "this month (September 2006)" name a period of their own; a
# relative window around now would filter by the wrong dates.
_MONTHS = (
    "january february april june july august september october november december"
).split()
# Month names that are also common words ("may" a modal verb, "march" a verb):
# a month only next to a day or a year, or after a preposition of time.
_AMBIGUOUS_MONTHS = ("may", "march")
_MONTH_ABBREVIATIONS = "jan feb mar apr jun jul aug sep sept oct nov dec".split()
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_YEAR = r"[12]\d{3}"
# A day and a month as numbers, in either order (d/m or m/d): 1-31 and 1-12.
_DAY_NUMBER = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH_NUMBER = r"(?:0?[1-9]|1[0-2])"
_DAY_MONTH = rf"(?:{_DAY_NUMBER}/{_MONTH_NUMBER}|{_MONTH_NUMBER}/{_DAY_NUMBER})"
_SHORT_MONTHS = "|".join(_AMBIGUOUS_MONTHS + tuple(_MONTH_ABBREVIATIONS))
_CALENDAR_ANCHOR = re.compile(
    "|".join(
        [
            # A year (not the N of "last N days/weeks/months").
            rf"\b{_YEAR}\b(?!\s*(?:days?|weeks?|months?)\b)",
            # A full month name.
            r"\b(?:" + "|".join(_MONTHS) + r")\b",
            # "may", "march", or an abbreviated month name next to a day or a
            # year, or after a preposition of time ("last week of March").
            rf"\b(?:{_SHORT_MONTHS})\.?\s+(?:{_DAY}|{_YEAR})\b",
            rf"\b{_DAY}\s+(?:of\s+)?(?:{_SHORT_MONTHS})\b",
            r"\b(?:of|in|during|since|until|till|through|from|by|early|mid|late)\s+"
            rf"(?:{_SHORT_MONTHS})\b",
            # A numeric date: d/m[/y] (valid day and month numbers; not a
            # fraction or ratio: "3/4 of the budget", "24/7"), d.m.yyyy,
            # d-m-yyyy or yyyy-m-d (a version "1.2.30" is no date).
            rf"(?<![\d/.]){_DAY_MONTH}(?:/(?:\d{{4}}|\d{{2}}))?(?![\d/])(?!\s+of\b)",
            r"\b\d{1,2}[.-]\d{1,2}[.-]\d{4}\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        ]
    )
)
# Round-the-clock "24/7" is no date.
_NOT_A_DATE = re.compile(r"(?<![\d/])24/7(?![\d/])")
# Relative time the patterns below cannot represent: a window they would pick
# from the rest of the text ("this week last year", "the day before
# yesterday", "a year ago today") would be the wrong one. A relative word
# counts only where it modifies a time unit or a date ("before last week",
# "2 years ago", "next month"): "yesterday before lunch", "what should I do
# next", "the previous draft" or "10 years of history" keep their window.
_WEEKDAY = r"(?:mon|tues|wednes|thurs|fri|satur|sun)days?"
# Units of a day or longer, and weekdays.
_TIME_UNIT = (
    r"(?:days?|nights?|weeks?|weekends?|fortnights?|months?|years?|quarters?|"
    rf"decades?|centur(?:y|ies)|{_WEEKDAY})"
)
# Units none of the patterns has a window for ("this year", "last weekend").
_UNSUPPORTED_UNIT = (
    rf"(?:weekends?|fortnights?|years?|quarters?|decades?|centur(?:y|ies)|{_WEEKDAY})"
)
_COUNT = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|several|"
    r"couple(?:\s+of)?)"
)
# Determiners that make a unit relative to now, none of which a pattern reads.
_RELATIVE_DETERMINER = r"(?:next|previous|prior|past|coming|following|recent)"
_DATE_EXPRESSION = (
    rf"(?:the\s+)?(?:today|tonight|yesterday|tomorrow|{_WEEKDAY}|"
    rf"(?:last|this|{_RELATIVE_DETERMINER})\s+(?:{_COUNT}\s+)?{_TIME_UNIT})"
)
_UNSUPPORTED_RELATIVE = re.compile(
    "|".join(
        [
            # Relative by themselves.
            r"\b(?:ago|tomorrow)\b",
            # "next month", "the previous week", "the past 3 years".
            rf"\b{_RELATIVE_DETERMINER}\s+(?:{_COUNT}\s+)?{_TIME_UNIT}\b",
            # "last year", "this quarter", "last weekend", "this Friday".
            rf"\b(?:last|this)\s+(?:{_COUNT}\s+)?{_UNSUPPORTED_UNIT}\b",
            # "before last week", "the day before yesterday", "prior to today".
            # Not a day's own event: "before today's meeting" is today, before
            # the meeting, as "yesterday before lunch" is (a week's or a
            # month's event may lie anywhere in it: "before last week's review"
            # stays unsupported).
            r"\b(?:before|after|prior\s+to)\s+"
            r"(?!(?:the\s+)?(?:today|tonight|yesterday)['\u2019]s\b)"
            rf"{_DATE_EXPRESSION}\b",
            # "the week after next" (not "after next steps").
            rf"\b{_TIME_UNIT}\s+after\s+next\b",
        ]
    )
)
# An offset from a date: "the day before", "two days later" — unless a
# frequency counts it (:data:`_FREQUENCY_BEFORE`: "twice a day before meals",
# "3 times a week after work", "2x a day after meals").
_DATE_OFFSET = re.compile(
    rf"\b(?:{_COUNT}|the)\s+{_TIME_UNIT}\s+(?:before|after|prior|earlier|later)\b"
)
# A frequency word ending the text right before an offset's match, and how
# far back it is sought (the words and their spacing are short; a bounded
# look keeps the check linear).
_FREQUENCY_BEFORE = re.compile(r"(?<!\w)(?:once|twice|thrice|times|\d+\s*x)\s+$")
_FREQUENCY_LOOKBACK_CHARS = 40


def has_calendar_anchor(text: str) -> bool:
    """True when ``text`` names an absolute date: a four-digit year, a month
    name ("may", "march" and abbreviations only next to a day number or a
    year, or after a preposition of time) or a numeric date (a day and a
    month that exist; not a fraction, a ratio, "24/7" or a version number)."""
    return bool(_CALENDAR_ANCHOR.search(_NOT_A_DATE.sub(" ", text.lower())))


def has_unsupported_relative(text: str) -> bool:
    """True when ``text`` holds relative time the parser cannot represent:
    "ago" or "tomorrow"; "next", "previous", "prior", "past", "coming",
    "following" or "recent" before a time unit or a weekday ("next month");
    "last" / "this" before a unit no pattern covers ("last year", "this
    quarter", "last weekend"); "before", "after" or "prior to" a date
    ("before last week", "the day before yesterday"); an offset such as "the
    day before" or "two days later" (not a frequency: "twice a day before
    meals", "2x a day after meals"). The words elsewhere ("yesterday before
    lunch", "before today's meeting", "the previous draft", "10 years of
    history") are not."""
    lower = text.lower()
    if _UNSUPPORTED_RELATIVE.search(lower):
        return True
    return any(
        not _FREQUENCY_BEFORE.search(
            lower, max(0, match.start() - _FREQUENCY_LOOKBACK_CHARS), match.start()
        )
        for match in _DATE_OFFSET.finditer(lower)
    )


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

    A relative window is never substituted for an absolute date or for a
    relative one it cannot represent: text that names a year, a month or a
    numeric date (:func:`has_calendar_anchor`, e.g. "last week of April in
    1972-1974", "this month (September 2006)"), or holds relative time none
    of the patterns covers (:func:`has_unsupported_relative`, e.g. "this week
    last year", "the day before yesterday"; not "yesterday before lunch")
    returns None, and no filter is the safe degrade.
    """
    if reference_time is None:
        reference_time = datetime.now()

    if has_calendar_anchor(text) or has_unsupported_relative(text):
        return None

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
