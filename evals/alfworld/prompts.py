"""SkillOpt-aligned prompt rendering for ALFWorld rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."

ROLLOUT_NO_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

ROLLOUT_WITH_HISTORY = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def format_admissible_actions(actions: Sequence[str]) -> str:
    """Render admissible actions exactly like SkillOpt's ALFWorld manager."""

    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def format_action_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Render recent observation/action pairs for the with-history prompt."""

    if not history:
        return ""
    chunks: list[str] = []
    for idx, step in enumerate(history, start=1):
        observation = str(step.get("observation", "")).strip()
        action = str(step.get("action", "")).strip()
        chunks.append(f"\n[{idx}] Observation: {observation}\n[{idx}] Action: {action}")
    return "".join(chunks)


def format_skill_knowledge(skills: Sequence[Mapping[str, Any]] | str | None) -> str:
    """Render MIRIX procedural memories as SkillOpt's skill-knowledge prelude."""

    if skills is None:
        return ""
    if isinstance(skills, str):
        content = skills.strip()
    else:
        blocks: list[str] = []
        for idx, skill in enumerate(skills, start=1):
            name = str(skill.get("name") or f"skill-{idx}").strip()
            description = str(skill.get("description") or "").strip()
            instructions = str(
                skill.get("instructions") or skill.get("content") or ""
            ).strip()
            if not (name or description or instructions):
                continue
            parts = [f"### {name}"]
            if description:
                parts.append(f"Description: {description}")
            if instructions:
                parts.append(f"Instructions:\n{instructions}")
            blocks.append("\n".join(parts))
        content = "\n\n".join(blocks).strip()

    if not content:
        return ""
    return (
        "## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{content}\n"
    )


def render_rollout_prompt(
    *,
    current_observation: str,
    admissible_actions: Sequence[str],
    task_description: str = "",
    step_count: int = 0,
    history: Sequence[Mapping[str, Any]] = (),
    history_length: int = 2,
    skills: Sequence[Mapping[str, Any]] | str | None = None,
) -> str:
    """Build the user prompt sent to the target model for one ALFWorld step."""

    recent_history = list(history)[-history_length:] if history_length > 0 else []
    rendered_actions = format_admissible_actions(admissible_actions)
    if not recent_history:
        body = ROLLOUT_NO_HISTORY.format(
            current_observation=current_observation,
            admissible_actions=rendered_actions,
        )
    else:
        body = ROLLOUT_WITH_HISTORY.format(
            task_description=task_description,
            step_count=step_count,
            history_length=len(recent_history),
            action_history=format_action_history(recent_history),
            current_step=step_count + 1,
            current_observation=current_observation,
            admissible_actions=rendered_actions,
        )

    skill_block = format_skill_knowledge(skills)
    if skill_block:
        return f"{skill_block}\n{body}"
    return body
