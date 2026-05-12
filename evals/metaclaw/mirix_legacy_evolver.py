"""LegacyMirixEvolver — feeds each round's serialised message to the
original-mirix `meta_agent` via POST /memory/add_sync. Subclasses
metaclaw.skill_evolver.SkillEvolver and overrides evolve() to match the
driver's call signature (failed_samples, current_skills=None) -> list[dict].

Returns [] because /memory/add_sync does not surface created
procedural_memory rows in its response, and querying for the diff would
be racy. Retrieval of newly stored procedures happens via
LegacyMirixManager.retrieve_async in subsequent rounds.

Per-round POST errors are caught and logged, not raised — one bad round
must not nuke evolution for an entire day's batch (see design spec §6).
"""
from __future__ import annotations

import logging
from typing import Iterable

from metaclaw.skill_evolver import SkillEvolver

from evals.metaclaw.format_adapter import RoundResult, round_to_message
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient

logger = logging.getLogger(__name__)


class LegacyMirixEvolver(SkillEvolver):
    """Subclass that delegates evolve() to MIRIX's /memory/add_sync endpoint."""

    def __init__(self, mirix: LegacyMirixClient) -> None:
        # Bypass parent __init__: it expects OpenAI env vars we don't need
        # because LLM work happens inside MIRIX, not this class.
        # update_history / history_path are kept as no-op state because
        # inherited get_update_summary() reads update_history.
        self.mirix = mirix
        self.update_history: list[dict] = []
        self.history_path = None

    def should_evolve(self, batch, threshold: float = 0.0) -> bool:  # noqa: ARG002
        # Driver-driven: always allow.
        return True

    async def evolve(
        self,
        failed_samples: Iterable[RoundResult],
        current_skills: dict | None = None,    # noqa: ARG002 (signature parity)
    ) -> list[dict]:
        """POST each round to /memory/add_sync. Returns [] (see module docstring)."""
        for r in failed_samples:
            try:
                await self.mirix.add_memory(round_to_message(r))
            except Exception as e:
                logger.warning(
                    "legacy evolve POST failed for round %s: %s", r.round_id, e
                )
        return []
