"""Unit tests for the backward-compatible AutoDreamResponse schema extension.

The generic-arm driver health-gates off structured evolution counts rather than
parsing the human-readable `message`, so AutoDreamResponse gained `skills_changed`
and `changes`. These tests pin the contract:

  * both new fields are OPTIONAL with sane defaults (0 / {}), so every existing
    caller (and non-procedural modes) keeps working unchanged;
  * when populated they round-trip through model_dump.

No live server / API key needed.
"""

import datetime as dt

from mirix.schemas.auto_dream import AutoDreamResponse, MemoryTypeStats


def _now() -> dt.datetime:
    return dt.datetime(2026, 6, 23, 12, 0, 0)


def test_defaults_are_zero_and_empty():
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={},
        last_dream_at=_now(),
        dry_run=False,
    )
    assert resp.skills_changed == 0
    assert resp.changes == {}


def test_populated_fields_round_trip():
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={"procedural": MemoryTypeStats(total=4)},
        last_dream_at=_now(),
        dry_run=False,
        skills_changed=2,
        changes={"created": ["s1"], "edited": ["s2"], "deleted": []},
    )
    assert resp.skills_changed == 2
    assert resp.changes == {"created": ["s1"], "edited": ["s2"], "deleted": []}

    dumped = resp.model_dump()
    assert dumped["skills_changed"] == 2
    assert dumped["changes"]["created"] == ["s1"]
    assert dumped["processed"]["procedural"]["total"] == 4

    # Re-parse the dump → identical structured counts (wire-stable for the driver).
    reparsed = AutoDreamResponse.model_validate(dumped)
    assert reparsed.skills_changed == 2
    assert reparsed.changes == resp.changes


def test_existing_caller_without_new_fields_unaffected():
    # Mirrors the exact shape pre-existing callers build (no skills_changed/changes).
    resp = AutoDreamResponse(
        start_date=None,
        end_date=None,
        processed={"procedural": MemoryTypeStats(total=0)},
        last_dream_at=_now(),
        dry_run=True,
        message="Dry run — distilled 0 experience(s).",
    )
    assert resp.message.startswith("Dry run")
    assert resp.skills_changed == 0
    assert resp.changes == {}
