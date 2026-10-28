#!/usr/bin/env python
"""Compute the retrieval vector of every live skill that has none.

A skill's ``retrieval_embedding`` is the vector of its retrieval text — name,
triggers (one per line), description. It is NULL after
scripts/migrate_skill_retrieval_fields.sql (the migration clears the
renamed column, whose old content was the description's vector), and when an
embedding call still failed after its retries. Until it has one, the skill
fails every vector gate of retrieval. The server re-embeds such skills in the
background at startup; this script does the same on demand:

* each skill is embedded with its OWN stored ``embedding_config`` and written
  through the one skill write path (``apply_skill_change``) under an
  optimistic check that its retrieval text did not change meanwhile — no
  version is written, and the skill's modification stamp is left alone;
* a skill that already has a vector is skipped, so the script is idempotent:
  run it as often as you like, while the server runs; an interrupted run
  simply continues where it stopped;
* soft-deleted skills are not re-embedded (nothing retrieves them).

It calls the embedding provider of each skill's ``embedding_config`` (this
costs what those embedding calls cost); ``--dry-run`` only counts the skills.

Usage (from the repository root, with the server's database settings, e.g.
``MIRIX_PG_URI``, and the embedding provider's key in the environment):

    python scripts/reembed_skill_retrieval.py --dry-run
    python scripts/reembed_skill_retrieval.py
    python scripts/reembed_skill_retrieval.py --organization-id org-... --concurrency 2

Exit code 0 when every skill found got its vector (or needed none); 2 when
some embedding calls still failed (run it again later); 3 when none failed
but some skills have no embedding config at all — they can never get a
vector, so the operator must fix or re-create them (the summary names the
count); 1 on an error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the retrieval vector of every live skill that has none."
    )
    parser.add_argument(
        "--organization-id",
        default=None,
        help="Only skills of this organization (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Skills read per query (default: 100).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Skills embedded at once (default: MIRIX_SKILL_REEMBED_CONCURRENCY, 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count the skills without a retrieval vector.",
    )
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace):
    from mirix.server.server import engine
    from mirix.services.skill_retrieval_reembed import (
        reembed_missing_retrieval_embeddings,
    )

    try:
        return await reembed_missing_retrieval_embeddings(
            organization_id=args.organization_id,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
    finally:
        await engine.dispose()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = asyncio.run(_main_async(args))
    print(report.summary())
    if report.failed:
        return 2
    if report.no_config:
        print(
            f"{report.no_config} skill(s) have no embedding config and can never "
            "get a retrieval vector: no gated retrieval returns them"
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
