"""No-op stub adapters that satisfy the MetaClaw SkillManager / SkillEvolver
duck-type surfaces.

The launcher's D6 dispatch (``METACLAW_SKILLS_PROVIDER=stub`` /
``METACLAW_EVOLVER_PROVIDER=stub``) wires these in instead of the paper
implementations.  They exist purely to *prove* the dispatch path works without
needing a live MIRIX server — every method logs at INFO level and returns the
empty/zero value of the correct type.

Surfaces covered (cross-checked against
``evals/metaclaw/vendor/metaclaw/api_server.py``):

SkillManager-shaped:
    - ``retrieve(task_description, top_k=...)``           → ``[]``
    - ``retrieve_relevant(task_description, top_k=...)``  → ``[]``
    - ``format_for_conversation(skills)``                 → ``""``
    - ``add_skill(skill)``                                → ``False``
    - ``add_skills(new_skills, category="general")``      → ``0``
    - ``get_skill_count()``                               → all-zero dict
    - ``reload()`` / ``save(path=None)``                  → no-op
    - ``.skills``       → empty paper-shape dict
    - ``.generation``   → 0

SkillEvolver-shaped:
    - ``evolve(failed_samples, current_skills)`` (async)  → ``[]``
    - ``should_evolve(batch, threshold=0.4)``             → ``True``
    - ``.update_history``                                 → ``[]``
    - ``.history_path``                                   → ``None``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_EMPTY_SKILLS_BANK: Dict[str, Any] = {
    "general_skills": [],
    "task_specific_skills": {},
    "common_mistakes": [],
}


class StubSkillsAdapter:
    """No-op SkillManager-shape used when ``METACLAW_SKILLS_PROVIDER=stub``."""

    def __init__(self) -> None:
        # paper-shape default so api_server's ``existing = self.skill_manager.skills``
        # read returns a structurally valid dict.
        self.skills: Dict[str, Any] = dict(_EMPTY_SKILLS_BANK)
        self.skills["task_specific_skills"] = {}
        self.generation: int = 0
        logger.info("[stub] StubSkillsAdapter initialised")

    # -- retrieval --------------------------------------------------------- #
    def retrieve(self, task_description: str, top_k: int = 6) -> List[dict]:
        logger.info("[stub] retrieve(q=%r, k=%d)", task_description[:80], top_k)
        return []

    def retrieve_relevant(self, task_description: str, top_k: int = 6) -> List[dict]:
        logger.info(
            "[stub] retrieve_relevant(q=%r, k=%d)", task_description[:80], top_k
        )
        return []

    def format_for_conversation(self, skills: List[dict]) -> str:
        logger.info("[stub] format_for_conversation(n=%d)", len(skills))
        return ""

    # -- mutation ---------------------------------------------------------- #
    def add_skill(self, skill: dict) -> bool:
        logger.info(
            "[stub] add_skill(name=%r)",
            skill.get("name") if isinstance(skill, dict) else None,
        )
        return False

    def add_skills(self, new_skills: List[dict], category: str = "general") -> int:
        logger.info("[stub] add_skills(n=%d, category=%r)", len(new_skills), category)
        return 0

    # -- introspection / lifecycle ---------------------------------------- #
    def get_skill_count(self) -> Dict[str, int]:
        # api_server only reads it for logging; return the same shape paper does.
        return {
            "general_skills": 0,
            "task_specific_skills": 0,
            "common_mistakes": 0,
            "total": 0,
        }

    def reload(self) -> None:
        logger.info("[stub] reload()")

    def save(self, path: Optional[str] = None) -> None:
        logger.info("[stub] save(path=%r)", path)


class StubEvolverAdapter:
    """No-op SkillEvolver-shape used when ``METACLAW_EVOLVER_PROVIDER=stub``."""

    def __init__(self) -> None:
        self.update_history: List[dict] = []
        self.history_path: Optional[str] = None
        logger.info("[stub] StubEvolverAdapter initialised")

    async def evolve(
        self, failed_samples: list, current_skills: Dict[str, Any]
    ) -> List[dict]:
        logger.info(
            "[stub] evolve(n_samples=%d, n_existing=%d)",
            len(failed_samples),
            sum(
                len(v) if isinstance(v, list) else len(v)
                for v in (current_skills or {}).values()
                if isinstance(v, (list, dict))
            ),
        )
        return []

    def should_evolve(self, batch: list, threshold: float = 0.4) -> bool:
        logger.info("[stub] should_evolve(n=%d, threshold=%.2f)", len(batch), threshold)
        return True


__all__ = ["StubSkillsAdapter", "StubEvolverAdapter"]
