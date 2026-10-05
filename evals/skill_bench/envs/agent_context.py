"""The executor's prompt as a static text for memory methods (the agent context).

Every method's learner may receive the executor's prompt template as a generic "agent
context". An environment that offers one (ALFWorld and WebShop both do) builds it here
from its frozen actor templates, the very strings its ``render_actor_messages`` fills,
so the text is exactly what the actor is sent with the per-step values left as visible
named placeholders (``{current_observation}``, ``{memory}`` ...). It is built from the
templates alone, never from a task: it is the same for every item of a benchmark and
holds no label (the audit's H10.1 checks search it like the rest of the method-visible
text, ``audit.py``).

The rollout puts it into :attr:`~evals.skill_bench.methods.base.TaskView.agent_context`
in both phases; a method decides whether to use it (MIRIX attaches it to its learning
sessions through its agent-context API; ReasoningBank and AWM keep their published
pipelines and ignore it).

The framing words (the intro, the section titles and the placeholder descriptions) are
harness code that describes what the rendering code fills in; they are recorded with the
harness code fingerprint of every invocation. The template text inside the
sections is the fingerprinted protocol text itself. No file is added under
``envs/templates``, so the fingerprinted templates stay byte-identical.

This is the one model-visible text outside the protocol fingerprint (``protocol_asset``
says every prompt is fingerprinted; the exception is recorded, not moved). So that a
change of it between cells or replicates cannot pass unseen, every attempt records the
text's digest (:func:`context_record`, ``attempt.json`` ``agent_context``), and
``tools/aggregate_replicates.py`` flags a cell whose replicates carried different texts
(``mixed_agent_context``).
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The marks around a section's title (``=== user message, step 1 ===``).
SECTION_MARK = "==="
#: The intro: what the text is (MIRIX's distiller shows it in its ``<agent_context>``
#: block, "how the agent's prompt was built").
INTRO = (
    "How the agent's prompt is built. At every step the agent receives the messages "
    "below, built afresh from the current state, and replies once; it sees nothing of "
    "earlier steps beyond what these messages show. Text in braces is filled in at "
    "each step:"
)
#: ``{memory}``: the Q21 slot (``insert_memory``: the slot and the joiner go in front of
#: the prompt only when the method's text is not blank).
MEMORY_DESCRIPTION = (
    "the memory text given to the agent for this task, the same at every step; when "
    "there is none, {memory} and the blank line after it are left out"
)
STEP_COUNT_DESCRIPTION = "the number of steps taken so far"
CURRENT_STEP_DESCRIPTION = "this step's number ({step_count} + 1)"


@dataclass(frozen=True)
class PromptSection:
    """One message of the actor's prompt: its ``role``, the steps it is sent at (from
    ``first_step`` to ``last_step``; None: every later step too) and its ``template``,
    the frozen text with every per-step value as ``{name}``."""

    role: str
    first_step: int
    last_step: int | None
    template: str

    @property
    def title(self) -> str:
        if self.first_step == 1 and self.last_step is None:
            when = "every step"
        elif self.last_step is None:
            when = f"step {self.first_step} and later"
        elif self.last_step == self.first_step:
            when = f"step {self.first_step}"
        else:
            when = f"steps {self.first_step} to {self.last_step}"
        return f"{self.role} message, {when}"

    def sent_at(self, step: int) -> bool:
        """Whether the actor's call of step ``step`` (1-based) uses this section."""
        return step >= self.first_step and (
            self.last_step is None or step <= self.last_step
        )


def fields_of(template: str) -> tuple[str, ...]:
    """The placeholder names of ``template``, in order of first use."""
    names: list[str] = []
    for _literal, name, _spec, _conversion in string.Formatter().parse(template):
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def fill_fields(template: str, **values: object) -> str:
    """``template`` with the named placeholders filled in and every other one kept as
    ``{name}``, so the result still shows where the per-step values go.

    Only plain ``{name}`` placeholders are accepted: a format spec, a conversion or a
    literal brace would not survive being shown and filled again (the actor templates
    have none; ``ValueError`` otherwise)."""
    out: list[str] = []
    for literal, name, spec, conversion in string.Formatter().parse(template):
        if "{" in literal or "}" in literal:
            raise ValueError("a template shown as the agent context has literal braces")
        out.append(literal)
        if name is None:
            continue
        if spec or conversion or not name.isidentifier():
            raise ValueError(f"placeholder {{{name}}} is not a plain name")
        out.append(str(values[name]) if name in values else "{" + name + "}")
    return "".join(out)


def history_description(history_length: int, entry: str) -> str:
    """``{action_history}``: what ``format_action_history`` lists, ``entry`` being the
    frozen per-step entry template (``history_entry``)."""
    shown = fill_fields(entry, step_num="N", obs="…", act="…")
    return (
        f"the {history_length} most recent earlier steps (fewer before step "
        f"{history_length + 1}), oldest first, one per line, each as {shown}"
    )


def single_turn_sections(
    role: str, first: str, later: str, history_length: int
) -> list[PromptSection]:
    """The messages of a single-turn prompt (ALFWorld, WebShop) that shows the last
    ``history_length`` steps: ``first`` on step 1, then ``later`` once per step while
    the history is still filling (step 2 shows one earlier step ...) and once for every
    step after that, each with its ``{history_length}`` filled in as
    ``format_action_history`` fills it. Without a history every step uses ``first``."""
    if history_length < 1:
        return [PromptSection(role, 1, None, first)]
    sections = [PromptSection(role, 1, 1, first)]
    sections += [
        PromptSection(
            role, shown + 1, shown + 1, fill_fields(later, history_length=shown)
        )
        for shown in range(1, history_length)
    ]
    sections.append(
        PromptSection(
            role,
            history_length + 1,
            None,
            fill_fields(later, history_length=history_length),
        )
    )
    return sections


def render(
    sections: Sequence[PromptSection],
    placeholders: Mapping[str, str],
    notes: Sequence[str] = (),
) -> str:
    """The agent context: the intro, one line per placeholder, the notes, then each
    section under its title. Every placeholder the sections use must be described and
    every described one used (``ValueError`` otherwise): the text never names a slot
    the prompt does not have, nor leaves one unexplained."""
    used: list[str] = []
    for section in sections:
        used += [name for name in fields_of(section.template) if name not in used]
    missing = [name for name in used if name not in placeholders]
    unused = [name for name in placeholders if name not in used]
    if missing or unused:
        raise ValueError(
            f"placeholders without a description: {missing}; described but not in "
            f"the prompt: {unused}"
        )
    lines = [INTRO]
    lines += [f"- {{{name}}}: {placeholders[name]}" for name in used]
    lines += list(notes)
    head = "\n".join(lines)
    body = "\n\n".join(
        f"{SECTION_MARK} {section.title} {SECTION_MARK}\n{section.template}"
        for section in sections
    )
    return f"{head}\n\n{body}"


def env_agent_context(env: Any) -> str:
    """An environment's static prompt template for methods (``TaskView.agent_context``):
    its ``agent_context()``, built from its frozen templates alone, so every item of the
    benchmark gets the same text in both phases; "" for an environment that offers none.
    The rollout sets it, the audit compares the records with it."""
    offer = getattr(env, "agent_context", None)
    if offer is None:
        return ""
    text = offer()
    if not isinstance(text, str):
        raise TypeError(
            f"{type(env).__name__}.agent_context() returned {type(text).__name__}"
        )
    return text


def context_record(text: str) -> dict[str, Any]:
    """What an attempt records of the agent context its items' task views carry
    (``attempt.json`` ``agent_context``): the text's ``sha256`` (UTF-8) and its length in
    ``chars`` (0 for a benchmark that offers none). Pure."""
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chars": len(text),
    }
