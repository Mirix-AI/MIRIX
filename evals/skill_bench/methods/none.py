"""No-memory method: ``before_task`` injects nothing, every other hook does nothing.

It runs only the test phase (H1.6) and is the baseline every method is compared with (H9).
With it, the actor prompt is exactly SkillOpt's template with an empty skill slot.
"""

from __future__ import annotations

from typing import Any

from evals.skill_bench.methods.base import (
    Injection,
    MemoryMethodBase,
    SignalLevel,
    TaskView,
)


class NoMemory(MemoryMethodBase):
    name = "none"
    signal_level: SignalLevel = "none"

    async def before_task(self, task: TaskView) -> Injection:
        return Injection()

    async def snapshot(self) -> dict[str, Any]:
        return {"method": self.name, "memory": None}
