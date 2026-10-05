"""Render a memory retrieval as prompt text.

:func:`render_retrieved_memories` turns the answer of
``POST /memory/retrieve/conversation`` (or ``GET /memory/retrieve/topic``) into
the text an agent puts in its context. ``MirixClient.render_memories`` is the
same function. It is pure: no I/O, no network, the same text for the same
response.

Procedural memory (the skills) renders as::

    ## Relevant skills from memory
    ### {name}
    {description}
    When to use: {trigger 1}; {trigger 2}; ...
    Instructions:
    {instructions}
    Examples:
    - {key}: {value}; {key}: {value}
    - ...

one ``### {name}`` block per skill, in the retrieval's order. The name and
the instructions are filled in literally. A field with no content is left
out rather than printed empty: the description line when it holds no
letter or digit (a placeholder such as "..."), a trigger with none, the
``When to use:`` line when no trigger is left, and the ``Examples:`` block
when no example renders. Each example is one ``- `` line: a mapping as its
``key: value`` pairs in its own key order (text as it is, any other value as
compact JSON), anything else as its text. A pair without a value is left out
(None, blank text, an elision such as "..." or "…", an empty container); an
example none of whose values holds a letter or a digit is left out, while a
punctuation-only value of an example that does ("output: ==") is kept. A
multi-line value keeps its lines (a code or a table example): an example
with one gets a line per pair, and the value's lines follow its key's line,
indented under it with their relative indentation::

    - input: Sum column B
      output:
        ```python
        df["B"].sum()
        ```

so no line of an example starts a line of the skill's block ("### ...").
The other memory types:

* **core** — ``## Core memory``, then ``### {label}`` and the block's value for
  every non-empty block (``### {label} (scope: {scope})`` when the blocks come
  from more than one scope);
* **episodic** — ``## Relevant events from memory`` (the events the query
  matched) and ``## Recent events from memory`` (the most recent ones not
  already listed): ``### {timestamp} {summary}`` and the details;
* **semantic** — ``## Relevant knowledge from memory``: ``### {name}``, the
  summary and the details;
* **resource** — ``## Relevant resources from memory``: ``### {title}
  ({resource_type})`` and the summary;
* **knowledge_vault** — ``## Relevant knowledge vault entries``: one
  ``- {caption}`` line per entry (a retrieval never returns the secret values).

Sections come in the order of ``memory_types`` (default: core, episodic,
semantic, procedural, resource, knowledge_vault), separated by a blank line;
empty ones are left out. Nothing to render is the empty string.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

SKILLS_HEADING = "## Relevant skills from memory"
CORE_HEADING = "## Core memory"
RELEVANT_EVENTS_HEADING = "## Relevant events from memory"
RECENT_EVENTS_HEADING = "## Recent events from memory"
SEMANTIC_HEADING = "## Relevant knowledge from memory"
RESOURCE_HEADING = "## Relevant resources from memory"
KNOWLEDGE_VAULT_HEADING = "## Relevant knowledge vault entries"

#: Default section order.
RENDER_ORDER = (
    "core",
    "episodic",
    "semantic",
    "procedural",
    "resource",
    "knowledge_vault",
)


def _text(value: Any) -> str:
    """A field as text: None is empty, anything else its ``str``, stripped."""
    if value is None:
        return ""
    return str(value).strip()


def _items(section: Any, key: str = "items") -> List[Mapping[str, Any]]:
    if not isinstance(section, Mapping):
        return []
    items = section.get(key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _raw(value: Any) -> str:
    """A skill field as the template writes it: None is empty, anything
    else its ``str`` — NOT stripped (the template is filled literally)."""
    return "" if value is None else str(value)


# A letter or a digit (any script): text without one carries no content.
_CONTENT = re.compile(r"[^\W_]")


def _has_content(text: str) -> bool:
    return bool(_CONTENT.search(text))


def _triggers(value: Any) -> List[str]:
    """The triggers with content, as they are."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [t for t in (_raw(v) for v in value) if _has_content(t)]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


# Text that only elides ("...", ". . .", "…"): a placeholder, not a value.
_ELISION = re.compile(r"[\s.\u2026]*")
#: One step of an example's indentation: its lines after the first sit under
#: its first line ("- " is two characters), and a multi-line value's lines one
#: step more, under their key.
_INDENT = "  "


def _empty(value: Any) -> bool:
    """Whether an example value holds nothing: None, blank text, an elision,
    or a container whose items (a mapping's keys and values) all hold
    nothing (``[]``, ``{}``, ``[[], None]``). A number or a boolean is a
    value (0, false)."""
    if value is None:
        return True
    if isinstance(value, str):
        return _ELISION.fullmatch(value) is not None
    if isinstance(value, Mapping):
        return all(_empty(k) and _empty(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_empty(v) for v in value)
    return False


def _value_lines(value: Any) -> List[str]:
    """An example value's lines: text with its surrounding blank lines
    dropped, trailing spaces removed, runs of blank lines kept as one and the
    indentation all its lines share removed (their relative indentation, a
    code block's, stays); anything else as one line of compact JSON. Every
    line break ``str.splitlines`` knows ends a line. [] when it holds nothing
    (:func:`_empty`)."""
    if _empty(value):
        return []
    if not isinstance(value, str):
        return [_compact_json(value)]
    lines: List[str] = []
    for line in value.splitlines():
        line = line.rstrip()
        if line or (lines and lines[-1]):
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    # The leading whitespace every non-blank line starts with, character for
    # character: a tab and two spaces share none (a count of leading
    # characters would strip one from each).
    margin = len(
        os.path.commonprefix(
            [line[: len(line) - len(line.lstrip())] for line in lines if line]
        )
    )
    return [line[margin:] for line in lines]


def _one_line(text: Any) -> str:
    """A mapping key on one line: its whitespace, line breaks included, collapsed."""
    return " ".join(str(text).split())


def _example_block(example: Any) -> List[str]:
    """One example's lines, the first without its ``- `` (see the module
    docstring); [] when it holds nothing or says nothing (no value with a
    letter or a digit)."""
    if isinstance(example, Mapping):
        pairs = [(_one_line(k), _value_lines(v)) for k, v in example.items()]
        pairs = [(key, lines) for key, lines in pairs if lines]
    else:
        lines = _value_lines(example)
        pairs = [(None, lines)] if lines else []
    if not any(_has_content(line) for _, lines in pairs for line in lines):
        return []
    if all(len(lines) == 1 for _, lines in pairs):
        return [
            "; ".join(
                lines[0] if key is None else f"{key}: {lines[0]}"
                for key, lines in pairs
            )
        ]
    if pairs[0][0] is None:  # one multi-line text: its lines under the "- "
        first, *rest = pairs[0][1]
        return [first, *_indented(rest)]
    block: List[str] = []
    for key, lines in pairs:
        if len(lines) == 1:
            block.append(f"{key}: {lines[0]}")
        else:
            block += [f"{key}:", *_indented(lines)]
    return [block[0], *_indented(block[1:])]


def _indented(lines: List[str]) -> List[str]:
    """``lines`` one :data:`_INDENT` step deeper (a blank line stays empty)."""
    return [_INDENT + line if line else "" for line in lines]


def _examples(value: Any) -> List[str]:
    """The lines of a skill's examples (one mapping is one example): each
    example's first line after ``- ``, its other lines indented under it."""
    if not value:
        return []
    if isinstance(value, (Mapping, str)) or not isinstance(value, (list, tuple)):
        value = [value]
    out: List[str] = []
    for example in value:
        block = _example_block(example)
        if block:
            out += ["- " + block[0], *block[1:]]
    return out


def render_skill(skill: Mapping[str, Any]) -> str:
    """One skill's block, without the section heading. The name
    and instructions are filled in literally (nothing stripped); a
    description without content, the ``When to use:`` line without a trigger
    and an empty ``Examples:`` block are left out."""
    lines = ["### " + _raw(skill.get("name"))]
    description = _raw(skill.get("description"))
    if _has_content(description):
        lines.append(description)
    triggers = _triggers(skill.get("triggers"))
    if triggers:
        lines.append("When to use: " + "; ".join(triggers))
    lines += ["Instructions:", _raw(skill.get("instructions"))]
    examples = _examples(skill.get("examples"))
    if examples:
        lines += ["Examples:", *examples]
    return "\n".join(lines)


def render_procedural(section: Any) -> str:
    skills = _items(section)
    if not skills:
        return ""
    return "\n".join([SKILLS_HEADING, *(render_skill(s) for s in skills)])


def _block(title: str, *body: str) -> str:
    return "\n".join([title, *(b for b in body if b)])


def _event(event: Mapping[str, Any]) -> str:
    title = " ".join(
        t for t in (_text(event.get("timestamp")), _text(event.get("summary"))) if t
    )
    return _block(f"### {title}".rstrip(), _text(event.get("details")))


def render_episodic(section: Any) -> str:
    relevant = _items(section, "relevant")
    shown = {item.get("id") for item in relevant if item.get("id") is not None}
    recent = [
        item
        for item in _items(section, "recent")
        if item.get("id") is None or item.get("id") not in shown
    ]
    parts = []
    if relevant:
        parts.append(
            "\n".join([RELEVANT_EVENTS_HEADING, *(_event(e) for e in relevant)])
        )
    if recent:
        parts.append("\n".join([RECENT_EVENTS_HEADING, *(_event(e) for e in recent)]))
    return "\n\n".join(parts)


def render_semantic(section: Any) -> str:
    items = _items(section)
    if not items:
        return ""
    blocks = [
        _block(
            f"### {_text(item.get('name'))}".rstrip(),
            _text(item.get("summary")),
            _text(item.get("details")),
        )
        for item in items
    ]
    return "\n".join([SEMANTIC_HEADING, *blocks])


def render_resource(section: Any) -> str:
    items = _items(section)
    if not items:
        return ""
    blocks = []
    for item in items:
        title = _text(item.get("title"))
        kind = _text(item.get("resource_type"))
        heading = f"### {title} ({kind})" if kind else f"### {title}"
        blocks.append(_block(heading.rstrip(), _text(item.get("summary"))))
    return "\n".join([RESOURCE_HEADING, *blocks])


def render_knowledge_vault(section: Any) -> str:
    captions = [
        c for c in (_text(item.get("caption")) for item in _items(section)) if c
    ]
    if not captions:
        return ""
    return "\n".join([KNOWLEDGE_VAULT_HEADING, *(f"- {c}" for c in captions)])


def render_core(section: Any) -> str:
    scopes = section.get("scopes") if isinstance(section, Mapping) else None
    if not isinstance(scopes, Mapping):
        return ""
    several = len(scopes) > 1
    blocks = []
    for scope, entry in scopes.items():
        for item in _items(entry):
            value = _text(item.get("value"))
            if not value:
                continue
            label = _text(item.get("label"))
            heading = f"### {label} (scope: {scope})" if several else f"### {label}"
            blocks.append(_block(heading.rstrip(), value))
    if not blocks:
        return ""
    return "\n".join([CORE_HEADING, *blocks])


_RENDERERS: Dict[str, Callable[[Any], str]] = {
    "core": render_core,
    "episodic": render_episodic,
    "semantic": render_semantic,
    "procedural": render_procedural,
    "resource": render_resource,
    "knowledge_vault": render_knowledge_vault,
}


def _types(memory_types: Optional[Iterable[str]]) -> Sequence[str]:
    if memory_types is None:
        return RENDER_ORDER
    if isinstance(memory_types, str):
        memory_types = [memory_types]
    wanted = list(dict.fromkeys(memory_types))
    unknown = [t for t in wanted if t not in _RENDERERS]
    if unknown:
        raise ValueError(
            f"unknown memory_types {unknown}; allowed: {list(RENDER_ORDER)}"
        )
    return wanted


def render_retrieved_memories(
    response: Mapping[str, Any], memory_types: Optional[Iterable[str]] = None
) -> str:
    """The retrieved memories of ``response`` as prompt text.

    Args:
        response: the JSON answer of ``/memory/retrieve/conversation`` (or
            ``/memory/retrieve/topic``): its ``memories`` are rendered.
        memory_types: render only these types, in this order (default: every
            type in the response, in :data:`RENDER_ORDER`). An unknown name is
            a ``ValueError``.

    Returns:
        The rendered text; ``""`` when there is nothing to render (no
        memories, or the retrieval found none — e.g. no skill passed the
        similarity gate).
    """
    types = _types(memory_types)
    memories = response.get("memories") if isinstance(response, Mapping) else None
    if not isinstance(memories, Mapping):
        return ""
    sections = []
    for memory_type in types:
        text = _RENDERERS[memory_type](memories.get(memory_type))
        if text:
            sections.append(text)
    return "\n\n".join(sections)


__all__ = [
    "RENDER_ORDER",
    "render_retrieved_memories",
    "render_skill",
]
