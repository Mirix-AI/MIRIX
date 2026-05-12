"""Schema bridge between MIRIX skill objects and MetaClaw skill dicts,
plus serialization of one bench round into a single evolve-message string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundResult:
    """One bench round outcome — what we send to MIRIX evolve."""
    round_id: str
    round_type: str            # "file_check" | "multi_choice"
    question: str
    final_answer: str          # agent's final tool output or bbox answer
    reward: float              # 1.0 pass, 0.0 fail
    eval_outcome: str          # "pass" | "fail"
    feedback: str              # bench's correct/incorrect feedback string
    transcript: list[dict] = field(default_factory=list)
    error: str | None = None


def mirix_to_metaclaw(skill: dict[str, Any]) -> dict[str, Any]:
    """MIRIX skill dict (id/name/description/instructions/entry_type/version)
    → MetaClaw skill dict (name/description/content/category)."""
    return {
        "name":        skill["name"],
        "description": skill["description"],
        "content":     skill["instructions"],
        "category":    skill.get("entry_type") or "general",
    }


def round_to_message(r: RoundResult) -> str:
    status = "PASS" if r.reward >= 1.0 else "FAIL"
    parts = [
        f"### Round {r.round_id}  [{r.round_type}]  outcome={status}",
        "",
        "**Question:**",
        r.question.strip(),
        "",
        "**Agent final answer:**",
        r.final_answer.strip() or "(empty)",
        "",
        "**Bench feedback:**",
        r.feedback.strip() or "(no feedback)",
    ]
    if r.error:
        parts += ["", f"**Error:** {r.error}"]
    return "\n".join(parts)
