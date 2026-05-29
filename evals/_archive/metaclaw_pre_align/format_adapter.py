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


def legacy_procedural_to_metaclaw(row: dict) -> dict:
    """Map a main-branch procedural_memory row to the metaclaw skill-shape.

    Old schema fields: summary, steps, entry_type.
    Target shape (same as mirix_to_metaclaw): name, description, content, category.

    Note: in MIRIX's procedural_memory schema/ORM, `steps` is a `list[str]`
    (see mirix/schemas/procedural_memory.py and mirix/orm/procedural_memory.py).
    Consumers like round_runner.build_system_prompt() call `.strip()` on
    `content`, so we must collapse list-shape steps into a newline-joined
    string here. Some test fixtures and historical rows store steps as a
    bare string — we accept either shape.
    """
    entry_type = row.get("entry_type") or "procedure"
    steps = row.get("steps")
    if isinstance(steps, list):
        # Stringify each element defensively (steps is typed List[str] but
        # serialized JSON might round-trip with stray non-strings).
        content = "\n".join(str(s) for s in steps if s is not None)
    else:
        content = steps or ""
    return {
        "name": entry_type,
        "description": row.get("summary") or "",
        "content": content,
        "category": entry_type,
    }
