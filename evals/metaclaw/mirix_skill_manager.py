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

    def retrieve(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        skills = _run_sync(self.mirix.search_skills(query=query, limit=top_k))
        return [mirix_to_metaclaw(s) for s in skills]


def _run_sync(coro):
    """Run an async coroutine from a sync caller. Works whether or not an
    event loop is already running in the current thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We are inside an async context already — bounce onto a fresh loop
        # in a thread. Acceptable here because retrieve() is on the bench
        # path, not on a tight inner loop.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
