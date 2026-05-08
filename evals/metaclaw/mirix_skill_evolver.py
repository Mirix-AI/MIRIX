"""MIRIX-backed replacement for MetaClaw's SkillEvolver.

The evolve trigger logic (success-rate threshold) is bypassed: the driver
calls evolve() exactly once per day at end-of-day. Within evolve(), each
RoundResult is serialized to one MIRIX evolve-message (success and failure
both included), MIRIX's ProceduralMemoryAgent decides what to create / edit
/ delete, and the resulting skills are returned in MetaClaw's
{name, description, content, category} format.
"""
from __future__ import annotations

from typing import Any, Iterable

from metaclaw.skill_evolver import SkillEvolver

from evals.metaclaw.format_adapter import (
    RoundResult,
    mirix_to_metaclaw,
    round_to_message,
)
from evals.metaclaw.mirix_client import MirixClient


class MirixSkillEvolver(SkillEvolver):
    """Subclass that delegates evolve() to MIRIX's REST endpoint."""

    def __init__(self, mirix_client: MirixClient | None) -> None:
        # Bypass parent __init__: it expects OpenAI env vars we don't need
        # here because the LLM call happens inside MIRIX, not in this class.
        # No **kwargs forwarding: this subclass does not honor the parent's
        # constructor knobs (max_new_skills, azure_deployment,
        # max_completion_tokens, llm_client, history_path); accepting them
        # silently would be a footgun. update_history / history_path are
        # kept as no-op state because inherited get_update_summary() reads
        # update_history.
        self.mirix = mirix_client
        self.update_history: list[dict] = []
        self.history_path = None

    def should_evolve(self, batch, threshold: float = 0.0) -> bool:  # noqa: ARG002
        # Driver-driven: always allow; the driver decides timing.
        return True

    async def evolve(
        self,
        failed_samples: Iterable[RoundResult],
        current_skills: dict | None = None,    # noqa: ARG002 (signature parity)
    ) -> list[dict]:
        rounds = list(failed_samples)
        if not rounds:
            return []
        messages = [round_to_message(r) for r in rounds]
        diff = await self.mirix.evolve(messages)
        produced = list(diff.get("created", [])) + list(diff.get("edited", []))
        return [mirix_to_metaclaw(s) for s in produced]
