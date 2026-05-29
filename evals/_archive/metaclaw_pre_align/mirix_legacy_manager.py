"""LegacyMirixManager — retrieves procedural memories from main-branch
MIRIX via GET /memory/search and maps them into the metaclaw skill-shape
that round_runner expects. Mirrors MirixSkillManager:
  - bypasses parent __init__ (which scans a skills_dir)
  - keeps no-op state for parent class / dashboard call paths
  - exposes retrieve_async, awaited directly by the driver
"""
from __future__ import annotations

from typing import Any

from metaclaw.skill_manager import SkillManager

from evals.metaclaw.format_adapter import legacy_procedural_to_metaclaw
from evals.metaclaw.mirix_legacy_client import LegacyMirixClient

# Single source of truth for the retrieval top-k default.
DEFAULT_TOP_K = 6


class LegacyMirixManager(SkillManager):
    """Subclass whose retrieve_async() delegates to MIRIX REST /memory/search."""

    def __init__(self, mirix: LegacyMirixClient) -> None:
        # Bypass parent __init__ (which requires a skills_dir to scan).
        # No-op state mirrors MirixSkillManager — parent class and dashboard
        # code may read self.skills / self.generation.
        self.mirix = mirix
        self.skills: dict[str, Any] = {
            "general_skills": [], "task_specific_skills": {}, "common_mistakes": []
        }
        self.generation: int = 0

    async def retrieve_async(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """Async retrieval path used directly by the driver."""
        rows = await self.mirix.search_procedural(query=query, limit=top_k)
        return [legacy_procedural_to_metaclaw(r) for r in rows]
