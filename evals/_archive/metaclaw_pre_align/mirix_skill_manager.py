"""MIRIX-backed replacement for MetaClaw's SkillManager.

The parent class scans a directory of SKILL.md files; this subclass bypasses
that scan and retrieves skills from MIRIX (BM25 over description) every time
retrieve() is called. The parent's retrieval interface is sync, so we wrap
the async MIRIX client via asyncio. Public API parity is enough for the
metaclaw-side code paths that consume `.retrieve()`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from metaclaw.skill_manager import SkillManager

from evals.metaclaw.format_adapter import mirix_to_metaclaw
from evals.metaclaw.mirix_client import MirixClient


# Single source of truth for the retrieval top-k default; the driver
# imports this so its CLI default and the manager's runtime default cannot
# drift out of sync.
DEFAULT_TOP_K = 6


class MirixSkillManager(SkillManager):
    """Subclass whose retrieve() delegates to MIRIX REST."""

    def __init__(self, mirix_client: MirixClient) -> None:
        # Bypass parent __init__ (which requires a skills_dir to scan).
        # No **kwargs forwarding: this subclass does not honor the parent's
        # constructor knobs (skills_dir, retrieval_mode, embedding_model_path,
        # task_specific_top_k); accepting them silently would be a footgun.
        self.mirix = mirix_client
        self.skills: dict[str, Any] = {
            "general_skills": [], "task_specific_skills": {}, "common_mistakes": []
        }
        self.generation: int = 0

    async def retrieve_async(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> list[dict[str, Any]]:
        """Async path used by the eval driver: avoids the
        sync-bridge-into-a-thread-loop bug that caused httpx.AsyncClient
        objects bound to the main loop to be re-awaited from a worker
        loop. Driver code should `await` this directly.
        """
        skills = await self.mirix.search_skills(query=query, limit=top_k)
        return [mirix_to_metaclaw(s) for s in skills]

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Sync path for parity with metaclaw's parent SkillManager.retrieve.

        Routes through asyncio.run when no loop is active. NOT safe to
        call from inside an already-running event loop while sharing a
        MirixClient that was first awaited on the main loop — use
        retrieve_async() in that case.
        """
        return asyncio.run(self._retrieve_coro(query, top_k))

    async def _retrieve_coro(self, query: str, top_k: int):
        return await self.retrieve_async(query, top_k)
