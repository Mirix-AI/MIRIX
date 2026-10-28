#!/usr/bin/env python
"""Write the ``baseline`` version of every skill that has no history yet.

Skill version history (``skill_version`` + ``skill_object``) starts with the
first change of a skill made after the history tables exist: that change first
records the skill's content as it was, as a ``baseline`` version, then its own
version. Skills nobody has changed since therefore have no history — and a
soft-deleted one cannot be restored, because restoring needs a version to
restore to. This script records the baseline of every such skill (live AND
soft-deleted) now.

Safe to run at any time, as often as you like, while the server runs:

* each skill is handled in its own transaction under the same locks every
  skill write takes (Postgres: the organization's shared advisory lock and the
  skill row's ``FOR UPDATE``), so it never races a concurrent change;
* a skill that already has a version is skipped, so a second run writes
  nothing (idempotent), and an interrupted run simply continues where it
  stopped.

Usage (from the repository root, with the server's database settings, e.g.
``MIRIX_PG_URI``, in the environment):

    python scripts/backfill_skill_baseline_versions.py
    python scripts/backfill_skill_baseline_versions.py --dry-run
    python scripts/backfill_skill_baseline_versions.py --organization-id org-... --batch-size 200

Run it after scripts/migrate_add_skill_history.sql. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class BackfillReport:
    """What a backfill run did."""

    #: Skills without history that were looked at.
    scanned: int = 0
    #: Baseline versions written (in a dry run: that would be written).
    written: int = 0
    #: Skills that got a version from someone else meanwhile, or vanished.
    skipped: int = 0
    dry_run: bool = False

    def summary(self) -> str:
        verb = "would write" if self.dry_run else "wrote"
        return (
            f"scanned {self.scanned} skill(s) without history; {verb} "
            f"{self.written} baseline version(s); skipped {self.skipped}"
        )


async def backfill_baseline_versions(
    *,
    session_maker=None,
    organization_id: Optional[str] = None,
    batch_size: int = 500,
    dry_run: bool = False,
    manager=None,
) -> BackfillReport:
    """Record a ``baseline`` version for every skill that has none (async).

    ``session_maker`` defaults to the server's database (``db_context``);
    ``manager`` (a ``ProceduralMemoryManager``) may be passed instead.
    ``organization_id`` limits the run to one organization.
    """
    from mirix.services import skill_history as history
    from mirix.services.procedural_memory_manager import ProceduralMemoryManager

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    manager = manager or ProceduralMemoryManager(session_maker=session_maker)
    report = BackfillReport(dry_run=dry_run)
    after_id: Optional[str] = None
    while True:
        async with manager.session_maker() as db:
            skill_ids: List[str] = await history.skill_ids_without_history(
                db,
                organization_id=organization_id,
                after_id=after_id,
                limit=batch_size,
            )
        if not skill_ids:
            break
        for skill_id in skill_ids:
            report.scanned += 1
            if dry_run:
                report.written += 1
                continue
            version = await manager.ensure_baseline_version(skill_id)
            if version is None:
                report.skipped += 1
            else:
                report.written += 1
        after_id = skill_ids[-1]
    return report


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the baseline version of every skill that has no history yet."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help="Only skills of this organization (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Skills read per query (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count the skills that would get a baseline.",
    )
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> BackfillReport:
    from mirix.server.server import engine

    try:
        return await backfill_baseline_versions(
            organization_id=args.organization_id,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    finally:
        await engine.dispose()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(_main_async(args))
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
