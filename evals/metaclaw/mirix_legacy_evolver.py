"""LegacyMirixEvolver — feeds each round's serialised message to the
original-mirix `meta_agent` via POST /memory/add_sync. Subclasses
metaclaw.skill_evolver.SkillEvolver for interface parity but bypasses
parent __init__ since we don't need any of its state.
"""
from __future__ import annotations

from typing import Any

from metaclaw.skill_evolver import SkillEvolver

from evals.metaclaw.format_adapter import RoundResult, round_to_message
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient


class LegacyMirixEvolver(SkillEvolver):
    def __init__(self, mirix: LegacyMirixClient):
        # Skip parent __init__ — we don't need its file-system / LLM state.
        self.mirix = mirix

    def should_evolve(self, *args: Any, **kwargs: Any) -> bool:
        return True

    async def evolve_async(self, rounds: list[RoundResult]) -> dict[str, Any]:
        for r in rounds:
            await self.mirix.add_memory(round_to_message(r))
        return {"sent": len(rounds)}
