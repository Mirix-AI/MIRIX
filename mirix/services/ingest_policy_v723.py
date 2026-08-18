"""Source-fidelity prompt policy for graph v7.23.

The policy is intentionally independent of LoCoMo.  It tells the two memory
writers to preserve deterministic source details before the graph extractor ever
sees their rows.  It does not create a second memory store, inspect benchmark
answers, or change episodic identity.
"""

from __future__ import annotations

from typing import Any


_MARKER = "V7.23 SOURCE-FIDELITY POLICY"

_COMMON = f"""

{_MARKER}:
- Treat the supplied message, transcript, document excerpt, or image caption as
  source evidence. Do not replace a concrete source detail with a broader gist.
- Preserve every explicit date/time expression, number, named entity, quoted
  phrase/title, enumerated item, and concrete visual attribute that is relevant
  to the memory being written.
- Keep exact source wording for signs, labels, titles, names, colors, counts, and
  other short identifying details. A concise summary may be broad, but `details`
  must retain those specifics.
- Do not invent an observation. Distinguish source assertions from reasonable
  inferences, and omit unsupported inferences from stored facts.
"""

_EPISODIC = """
- Keep separately timed real-world occurrences as separate episodic items even
  when their topics are similar. Never blend dates from different occurrences.
- Record the event time separately from the time at which it was mentioned, and
  preserve an uncertain or relative time expression when it cannot be resolved.
"""

_SEMANTIC = """
- Semantic memory may summarize stable knowledge, but it must not erase source
  qualifiers, historical values, or the citations needed to recover them.
- Do not turn repeated descriptions of an event into an additional occurrence.
"""


def source_fidelity_suffix(memory_kind: str) -> str:
    kind = (memory_kind or "").strip().lower()
    if kind == "episodic":
        return _COMMON + _EPISODIC
    if kind == "semantic":
        return _COMMON + _SEMANTIC
    return _COMMON


def apply_source_fidelity_prompt(agent_state: Any, memory_kind: str) -> bool:
    """Append the v7.23 policy once to an in-memory AgentState.

    Agent states are loaded repeatedly.  The marker makes the operation
    idempotent and avoids persisting a mutated prompt back to the database.
    """

    current = str(getattr(agent_state, "system", "") or "")
    if _MARKER in current:
        return False
    try:
        agent_state.system = current + source_fidelity_suffix(memory_kind)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


__all__ = ["apply_source_fidelity_prompt", "source_fidelity_suffix"]
