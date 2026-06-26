"""Deprecated. Use ``mirix.observability.timed.timedspan`` instead.

Kept as a thin re-export so any lingering ``from ...timed_spans import timed_span``
keeps working during/after the VEPAGE-1313 migration.
"""

from mirix.observability.timed import timedspan as timed_span  # noqa: F401
