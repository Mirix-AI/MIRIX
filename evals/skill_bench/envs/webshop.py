"""WebShop environment.

The shop is princeton-nlp/WebShop's text environment (``observation_mode="text"``, the full
1.18M products, human-written instructions) served by one long-lived process in a Docker
container on the run host (``envs/templates/webshop_server/``: ``server.py`` wraps
upstream's ``SimServer`` / ``WebAgentTextEnv`` with one global lock, a session prefix per
episode, a readiness gate and the frozen-manifest check). This module is the harness side:
an async HTTP client (``httpx.AsyncClient``) per episode, the actor prompt and the
method-visible messages.

The actor sees its whole episode as one append-only conversation (verl-agent instead
re-renders one user message per step with the last two observation/action pairs). Step
1's request is one user message, verl-agent 20bd331's ``rollout_no_history.md``
(``templates/webshop/``, provenance in its ``format.json``); the request of step k >= 2
is step k-1's followed by the model's reply of step k-1 as an assistant message and
``rollout_turn.md`` (verl-agent's ``rollout_with_history.md`` without its persona and
history lines) with step k's page and actions as a user message. Earlier messages are
never re-rendered, truncated or dropped, and there is no 13,000-character fallback. Why:
a page never shows which option is selected, so a two-step window loses a selection once
the agent opens a detail page after it, while the whole conversation keeps every earlier
step; provider prefix caching reads every earlier turn. The memory goes at the start of
the first user message only, as for ALFWorld. The prompt never states the 15-step cap
(verl-agent's WebShop cap).

Who sees what:

* ``Reset.messages`` / ``StepResult.messages`` (Q20, what methods receive): the first
  message is the raw search page, which states the instruction; every later one is the
  page without the instruction (verl-agent's ``format_obs``), each followed by the
  available actions. After Buy Now the only message is
  ``Thank you for shopping with us! Your score (min 0.0, max 1.0): <reward>``: the
  server never returns the rendered done page, whose hidden blocks keep the goal. The
  score is shown to every method in the learning phase only (methods receive messages
  only while learning). An episode the
  step cap ends without a purchase ends with the last page and then
  ``The episode ended at the step limit without a purchase. Your score (min 0.0, max
  1.0): 0.00`` (upstream ends it silently), shown to methods in the same way.
* ``fields`` (actor template inputs): the formatted page, the available actions, the
  instruction, the actor's conversation so far (``conversation``, without the memory;
  it never holds the score or cap line) and, after a step, the executed action, whether
  it was a format error, the projection's validity, whether the shop acted on it, why an
  action was not sent or not processed (``rejected``: a special search, an action that
  is not valid UTF-8, or one the shop raised on, with the server's ``shop_error``) and
  whether the step cap ended the episode without a purchase.
* ``private`` (scorer only): ``reward`` (None until a purchase), ``purchased`` and
  ``steps``. ``Task.raw`` holds the frozen goal (ASIN, attributes, options, price limit,
  category, ...) and never leaves the scorer.

``Episode.step`` takes the model's raw response. A response without an
``<action>...</action>`` pair (in any case, as the projection reads it), or with an empty
one, is a format error and a frozen no-op: the page stays as it is, nothing reaches the
server, and the step counts (verl-agent would send the reply's last 20 characters).
Otherwise verl-agent's projection (``vendor/webshop_projection.py``) extracts the action;
a search whose first word is a special engine command (``<r>``, ``<a>``, ``<c>``, ``<q>``)
is refused here, as an action the shop does not accept, and again by the server, and so
is an action that is not valid UTF-8 (a lone surrogate). Any other action goes to the
server; one upstream does not accept, or cannot process (its page code raises; the server
restores the session), leaves the page unchanged, with no extra text. The episode ends at
Buy Now or after :data:`protocol.WEBSHOP_MAX_STEPS` steps (then unanswered, score 0,
and the learners' last message is the cap line), and refuses every step after that.

Scoring: ``Score.primary`` is the shop's reward (0..1); ``metrics["success"]`` is
reward == 1, the main metric; ``success_within_15`` and the item's ``best_reward`` (the
reward of buying the gold product with the goal's options, from the frozen manifest)
are diagnostics (with the 15-step cap, ``success_within_15`` equals ``success``). WebShop
is not LLM-graded.

The environment works without a server (the audit host): its constructor, ``load_tasks``,
``score``, ``label_strings`` and the leak check read only the frozen files. The server
and its data are checked by :meth:`WebShopEnv.preflight` before a cell starts: a server
that is unreachable or still loading is INFRA (the cell runs again later); one that
failed its own checks, or differs from the frozen manifest, the protocol or this
checkout's Docker context, needs a person.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import socket
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.envs.agent_context import SECTION_MARK, fields_of
from evals.skill_bench.envs.base import (
    Action,
    Message,
    Reset,
    Score,
    Split,
    StepResult,
    Task,
)
from evals.skill_bench.envs.vendor.webshop_projection import webshop_projection

BENCHMARK = "webshop"
TEMPLATE_DIR = "envs/templates/webshop"
SERVER_DIR = "envs/templates/webshop_server"
SPLIT_DIR = "envs/splits/webshop"
VENDOR_FILES = ("envs/vendor/webshop_projection.py",)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
MANIFEST_FILE = f"{SPLIT_DIR}/split_manifest.json"
SOURCE_FILE = f"{SPLIT_DIR}/source.json"
#: The goal fields a final manifest pins per item (upstream ``goal.py:56-66``); the server
#: refuses to serve when any differs from the goals it built.
MANIFEST_FIELDS: tuple[str, ...] = (
    "asin",
    "instruction_text",
    "attributes",
    "goal_options",
    "price_upper",
    "name",
    "query",
    "category",
    "product_category",
)
MANIFEST_FINAL = "final"
MANIFEST_PROVISIONAL = "provisional"
ITEM_ID_PREFIX = "fixed_"  # never a bare number (the audit matches raw ids as words)

#: Where the server listens (loopback only). Not a protocol value.
SERVER_URL_ENV = "SKILL_BENCH_WEBSHOP_URL"
DEFAULT_SERVER_URL = "http://127.0.0.1:18560"
#: The server's HTTP API version (``server.py`` ``API_VERSION``).
SERVER_API_VERSION = 1
#: Longest wait for one server reply. A search runs over 1.18M products, and requests of
#: up to 24 concurrent episodes queue behind one global lock.
REQUEST_TIMEOUT_S = 120.0
READY_TIMEOUT_S = 10.0
#: The action of a model call without a reply (:meth:`WebShopEnv.call_error_response`).
NO_REPLY_KEY = "no_reply"
#: ``format.json`` ``history``: the actor's prompt is the whole episode, one append-only
#: conversation; the only mode this module renders.
FULL_CONVERSATION = "full_conversation"
#: Why an action left the page as it was without the shop's own refusal (``rejected``):
#: never sent (a special search, an action that is not valid UTF-8), or sent and not
#: processed (the shop raised on it; the server's ``shop_error`` says how).
REJECTED_SPECIAL_SEARCH = "special_search"
REJECTED_UNENCODABLE = "unencodable"
REJECTED_SHOP_ERROR = "shop_error"
SUCCESS_WITHIN_KEY = f"success_within_{protocol.WEBSHOP_DIAGNOSTIC_STEPS}"

#: Docker names. Only containers with this label are ever stopped.
IMAGE_REPOSITORY = "skill-bench-webshop"
CONTAINER_NAME = "skill-bench-webshop"
PREPARE_CONTAINER_NAME = "skill-bench-webshop-prepare"
CONTAINER_LABEL_KEY = "skill-bench.service"
CONTAINER_LABEL_VALUE = "webshop"
DOCKER_TIMEOUT_S = 60.0
#: ``docker run --restart``: a server that dies (an OOM kill at its memory cap) comes back
#: by itself, seeded and checked again; the gate holds WebShop cells until it is ready,
#: and the episodes it lost are re-run (INFRA). It also comes back after a Docker daemon
#: restart or a reboot of the host, which ``on-failure`` never did (a study runs
#: unattended for days); ``--stop`` removes it. Docker spaces repeated crashes
#: out with a growing delay; nothing in the harness relied on ``on-failure:3``'s cap of
#: three, and a cell whose episodes keep failing reaches the scheduler's resume caps (a
#: person looks). The run record keeps the container's restart count (preflight).
RESTART_POLICY = "unless-stopped"
#: The build argument (``server.py`` ``CONTEXT_ENV``) carrying :func:`context_digest`.
CONTEXT_BUILD_ARG = "SKILL_BENCH_CONTEXT"

_T = TypeVar("_T")


class WebShopEnvError(RuntimeError):
    """The environment, its frozen files or a request it sent are wrong (a harness
    failure, never a model failure)."""


class WebShopServerError(RuntimeError):
    """The WebShop server is unreachable, not ready, timed out or failed a request.
    ``infra/failures.py`` classifies it as INFRA ``webshop_server``: the item is re-run
    (test phase) or the learning phase resumes (H8), never scored."""


# ---------------------------------------------------------------------------
# Frozen templates and their rendering (verl-agent 20bd331)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebShopTemplates:
    """The frozen actor-prompt pieces and message formats (``envs/templates/webshop``).

    ``no_history`` is the conversation's first user message, ``turn`` the user message of
    every later step and ``no_reply_turn`` the assistant message of a step without a
    reply. ``with_history`` (verl-agent's with-history template, which ``turn`` is derived
    from) is not sent."""

    no_history: str
    with_history: str
    turn: str
    history: str
    no_reply_turn: str
    observation_separator: str
    instruction_marker: str
    observation_part: str
    available_search: str
    available_click: str
    available_excluded_clickables: tuple[str, ...]
    available_item: str
    available_joiner: str
    memory_slot: str
    memory_joiner: str
    action_tag_pattern: str
    action_parse_pattern: str
    special_search_pattern: str
    noop_action: str
    observation_message: str
    observation_message_joiner: str
    final_message: str
    cap_message: str
    leak_markers: tuple[tuple[str, ...], ...]

    @classmethod
    def load(cls, root: Path = protocol.SKILL_BENCH_DIR) -> WebShopTemplates:
        protocol.verify_frozen_assets(TEMPLATE_DIR + "/", root)
        spec = json.loads(
            protocol.read_protocol_text(f"{TEMPLATE_DIR}/format.json", root)
        )["templates"]
        if spec.get("history") != FULL_CONVERSATION:
            raise WebShopEnvError(
                f"format.json's history is {spec.get('history')!r}: this environment "
                f"renders only {FULL_CONVERSATION!r}"
            )
        return cls(
            no_history=protocol.read_protocol_text(
                f"{TEMPLATE_DIR}/{spec['no_history_file']}", root
            ),
            with_history=protocol.read_protocol_text(
                f"{TEMPLATE_DIR}/{spec['with_history_file']}", root
            ),
            turn=protocol.read_protocol_text(
                f"{TEMPLATE_DIR}/{spec['turn_file']}", root
            ),
            history=spec["history"],
            no_reply_turn=spec["no_reply_turn"],
            observation_separator=spec["observation_separator"],
            instruction_marker=spec["instruction_marker"],
            observation_part=spec["observation_part"],
            available_search=spec["available_search"],
            available_click=spec["available_click"],
            available_excluded_clickables=tuple(spec["available_excluded_clickables"]),
            available_item=spec["available_item"],
            available_joiner=spec["available_joiner"],
            memory_slot=spec["memory_slot"],
            memory_joiner=spec["memory_joiner"],
            action_tag_pattern=spec["action_tag_pattern"],
            action_parse_pattern=spec["action_parse_pattern"],
            special_search_pattern=spec["special_search_pattern"],
            noop_action=spec["noop_action"],
            observation_message=spec["observation_message"],
            observation_message_joiner=spec["observation_message_joiner"],
            final_message=spec["final_message"],
            cap_message=spec["cap_message"],
            leak_markers=tuple(tuple(m) for m in spec["leak_markers"]),
        )


def list_available_actions(
    t: WebShopTemplates, available: Mapping[str, Any]
) -> tuple[str, ...]:
    """verl-agent ``format_avail_actions`` (``env_manager.py:449-462``), without
    ``click[search]``: upstream refuses it."""
    unknown = set(available) - {"has_search_bar", "clickables"}
    if unknown:
        raise WebShopEnvError(
            f"unknown keys in the available actions: {sorted(unknown)}"
        )
    clickables = available.get("clickables")
    if not isinstance(clickables, (list, tuple)) or not all(
        isinstance(c, str) for c in clickables
    ):
        raise WebShopEnvError("the server's clickables are not a list of strings")
    actions = [t.available_search] if available.get("has_search_bar") else []
    actions += [
        t.available_click.format(clickable=c)
        for c in clickables
        if c not in t.available_excluded_clickables
    ]
    return tuple(actions)


def format_available_actions(t: WebShopTemplates, actions: Sequence[str]) -> str:
    """verl-agent ``env_manager.py:478``: each action quoted with a comma, one per line."""
    return t.available_joiner.join(t.available_item.format(action=a) for a in actions)


def extract_task(t: WebShopTemplates, observation: str) -> str:
    """verl-agent ``extract_task`` (``env_manager.py:426-432``): the part after
    ``Instruction:`` of the first page (the item's instruction)."""
    parts = observation.split(t.observation_separator)
    if len(parts) < 3 or parts[1] != t.instruction_marker:
        raise WebShopEnvError("the first page does not state the instruction")
    return parts[2]


def format_observation(t: WebShopTemplates, observation: str, task: str) -> str:
    """verl-agent ``format_obs`` (``env_manager.py:434-447``): the parts after the
    instruction, each quoted; a page without the instruction stays as it is."""
    parts = observation.split(t.observation_separator)
    try:
        index = parts.index(task)
    except ValueError:
        return observation
    return t.observation_separator.join(
        t.observation_part.format(part=p) for p in parts[index + 1 :]
    )


def insert_memory(t: WebShopTemplates, prompt: str, memory: str) -> str:
    """The memory slot at the start of the user prompt (as for ALFWorld). An
    empty or whitespace-only memory injects nothing."""
    if not memory or not memory.strip():
        return prompt
    return t.memory_slot.format(memory=memory) + t.memory_joiner + prompt


def first_prompt(
    t: WebShopTemplates, task_description: str, observation: str, actions: Sequence[str]
) -> str:
    """The conversation's first user message, without the memory: verl-agent's
    no-history prompt (``build_text_obs``, ``env_manager.py:480-485``)."""
    return t.no_history.format(
        task_description=task_description,
        current_observation=observation,
        available_actions=format_available_actions(t, actions),
    )


def turn_prompt(
    t: WebShopTemplates,
    task_description: str,
    step: int,
    observation: str,
    actions: Sequence[str],
) -> str:
    """The user message appended for step ``step`` >= 2: ``rollout_turn.md`` with that
    step's page and actions."""
    return t.turn.format(
        task_description=task_description,
        current_step=step,
        current_observation=observation,
        available_actions=format_available_actions(t, actions),
    )


def reply_turn(t: WebShopTemplates, action: Action) -> str:
    """The assistant message of a step: the reply exactly as :meth:`WebShopEpisode.step`
    received it, with U+FFFD in place of each lone surrogate; ``no_reply_turn`` for a
    call without a reply and for a reply that is empty or whitespace only."""
    if isinstance(action, str) and action.strip():
        return without_surrogates(action)
    return t.no_reply_turn


def conversation_of(fields: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The episode's conversation in ``fields``, checked: user and assistant messages in
    turn, from the first user message to the current step's (2 * ``step_count`` + 1
    messages). Fields that lost it, or part of it, raise :class:`WebShopEnvError`
    rather than send the actor a shorter episode; so do the fields of an episode that is
    over (its conversation ends with the last reply: no call follows)."""
    conversation = fields.get("conversation")
    steps = fields.get("step_count")
    if not isinstance(conversation, (tuple, list)) or not conversation:
        raise WebShopEnvError(
            "the fields hold no conversation: the actor's prompt is the whole episode, "
            "and these fields cannot rebuild it"
        )
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise WebShopEnvError("the fields hold no step_count")
    for index, message in enumerate(conversation):
        role = "assistant" if index % 2 else "user"
        if (
            not isinstance(message, Mapping)
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
        ):
            raise WebShopEnvError(
                f"conversation message {index} should be the {role}'s text"
            )
    if len(conversation) == 2 * steps:
        raise WebShopEnvError(
            "the episode is over: no actor call follows its last step"
        )
    if len(conversation) != 2 * steps + 1:
        raise WebShopEnvError(
            f"the conversation holds {len(conversation)} messages, not the "
            f"{2 * steps + 1} of {steps} step(s): part of the episode is lost"
        )
    return tuple(conversation)


def render_actor_messages(
    t: WebShopTemplates, fields: Mapping[str, Any], memory: str = ""
) -> list[Message]:
    """The actor call of the current step: the episode's whole
    conversation (:func:`conversation_of`), with the memory in the slot of its first
    message only. Step 1's call is one user message, as verl-agent sends its prompt (it
    has no system message); each later call is the one before followed by that step's
    reply and this step's user message."""
    first, *later = conversation_of(fields)
    return [
        {"role": "user", "content": insert_memory(t, first["content"], memory)},
        *({"role": m["role"], "content": m["content"]} for m in later),
    ]


@dataclass(frozen=True)
class ConversationSection:
    """One kind of message of the actor's conversation in the agent context: its role,
    its ``title`` and its ``template``, the frozen text with every per-step value as
    ``{name}``."""

    role: str
    title: str
    template: str


#: The intro of WebShop's agent context. ``agent_context.INTRO`` (ALFWorld's) describes
#: messages built afresh at each step; WebShop's prompt is one conversation that grows.
CONVERSATION_INTRO = (
    "How the agent's prompt is built. The prompt is one conversation that grows by one "
    "exchange per step: at step 1 it is the first user message below; after each step "
    "the agent's reply is appended as an assistant message, then the next step's user "
    "message. Earlier messages are never changed or dropped, so at step N the agent "
    "sees all N-1 earlier steps: each page, its admissible actions and the agent's own "
    "reply, verbatim. Text in braces is filled in:"
)


def agent_context_sections(t: WebShopTemplates) -> tuple[ConversationSection, ...]:
    """The conversation's three kinds of message as :func:`render_actor_messages` sends
    them: the first user message (``rollout_no_history.md`` behind the memory slot), the
    assistant message of each step (its reply) and the user message appended at step 2
    and later (``rollout_turn.md``). verl-agent sends no system message."""
    return (
        ConversationSection(
            "user",
            "user message, step 1",
            t.memory_slot + t.memory_joiner + t.no_history,
        ),
        ConversationSection(
            "assistant", "assistant message, the agent's reply at each step", "{reply}"
        ),
        ConversationSection(
            "user", "user message, appended at step 2 and later", t.turn
        ),
    )


def agent_context(t: WebShopTemplates) -> str:
    """The actor's prompt as a static, label-free text for memory methods
    (``envs/agent_context.py``), laid out as ``agent_context.render`` lays out
    ALFWorld's, under :data:`CONVERSATION_INTRO`. Every placeholder the sections use is
    described, in order of first use, and every one described is used (``ValueError``
    otherwise). The step cap is not in it: the prompt never states it."""
    sections = agent_context_sections(t)
    placeholders = {
        "memory": "the memory text given to the agent for this task, in the first "
        "message only; when there is none, {memory} and the blank line after it are "
        "left out",
        "task_description": "the shopping instruction",
        "current_observation": "the page of that step as text",
        "available_actions": "the actions available on that page, one per line, each "
        "in single quotes and followed by a comma",
        "reply": "the agent's reply at that step, exactly as it was given, except that "
        "a character that cannot be sent shows as U+FFFD; a step without a reply, or "
        f"with an empty or whitespace-only one, shows {t.no_reply_turn}",
        "current_step": "the number of the step the message opens (2, 3, ...)",
    }
    used: list[str] = []
    for section in sections:
        used += [name for name in fields_of(section.template) if name not in used]
    if sorted(used) != sorted(placeholders):
        raise ValueError(
            f"placeholders in the prompt {used} are not the described {list(placeholders)}"
        )
    head = "\n".join(
        [CONVERSATION_INTRO, *(f"- {{{name}}}: {placeholders[name]}" for name in used)]
    )
    body = "\n\n".join(
        f"{SECTION_MARK} {section.title} {SECTION_MARK}\n{section.template}"
        for section in sections
    )
    return f"{head}\n\n{body}"


def observation_message(
    t: WebShopTemplates, observation: str, actions: Sequence[str]
) -> Message:
    """The method-visible user message: the page and the available actions."""
    return {
        "role": "user",
        "content": t.observation_message.format(
            observation=observation,
            available_actions=t.observation_message_joiner.join(actions),
        ),
    }


def final_message(t: WebShopTemplates, reward: float) -> Message:
    """The only message after Buy Now: the score line, nothing else."""
    return {"role": "user", "content": t.final_message.format(reward=reward)}


def final_message_pattern(t: WebShopTemplates) -> re.Pattern[str]:
    """What a final message is, exactly: the template with a two-decimal score in 0..1."""
    head, _, tail = t.final_message.partition("{reward:.2f}")
    return re.compile(re.escape(head) + r"(?:0\.\d\d|1\.00)" + re.escape(tail))


def cap_message(t: WebShopTemplates) -> Message:
    """The last message of an episode the step cap ends without a purchase: the cap line
    with the score such an episode gets, 0.00, as the score line closes an episode that
    bought."""
    return {"role": "user", "content": t.cap_message.format(reward=0.0)}


def page_parts(t: WebShopTemplates, text: str) -> list[str]:
    """The non-empty parts of a page or message as upstream's text mode separates them
    (``observation_separator``; a message's own lines too), each stripped of whitespace
    and of the quotes ``format_obs`` puts around a part."""
    parts = []
    for chunk in text.split(t.observation_separator):
        for line in chunk.split("\n"):
            part = line.strip()
            if len(part) >= 2 and part[0] == part[-1] == "'":
                part = part[1:-1].strip()
            if part:
                parts.append(part)
    return parts


def _holds(parts: Sequence[str], marker: Sequence[str]) -> bool:
    size = len(marker)
    return any(
        tuple(parts[i : i + size]) == tuple(marker)
        for i in range(len(parts) - size + 1)
    )


def leak_problems(
    t: WebShopTemplates, messages: Sequence[Mapping[str, Any]]
) -> list[tuple[int, str]]:
    """The structural leak check (the audit's ``H10.1:structural_leak``):
    ``(message index, problem)`` for every environment message (the model's own turns are
    not the environment's) that holds done-page text or the score line without being
    exactly the final score line (or the cap line, :func:`cap_message`), and for a final
    score line or cap line that is not the last message. Labels cannot be searched for on WebShop pages (the target's ASIN, title and
    options appear on them legitimately), so this is the check.

    Done-page text is one of ``leak_markers``: done-page labels, each a run of whole parts
    of the page (:func:`page_parts`), as upstream's text mode renders the done page. A
    seller's own text that merely contains such words (a description ending "Thank you
    for shopping with us!", a "best price upper arm monitor") is part of a longer part,
    or a part followed by other parts, and is never a leak."""
    pattern = final_message_pattern(t)
    cap_line = cap_message(t)["content"]
    found: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        if pattern.fullmatch(text) or text == cap_line:
            if message.get("role") != "user":
                found.append((index, "the score line is not a user message"))
            elif index != len(messages) - 1:
                found.append((index, "the score line is not the last message"))
            continue
        if pattern.search(text) or cap_line in text:
            found.append((index, "it holds the score line among other text"))
            continue
        parts = page_parts(t, text)
        for marker in t.leak_markers:
            if _holds(parts, marker):
                shown = t.observation_separator.join(marker)
                found.append((index, f"it holds the done page's {shown!r}"))
                break
    return found


# ---------------------------------------------------------------------------
# Responses -> actions
# ---------------------------------------------------------------------------


def parse_action(t: WebShopTemplates, command: str) -> tuple[str, str | None]:
    """upstream ``engine.parse_action`` (``engine.py:117-128``)."""
    match = re.match(t.action_parse_pattern, command)
    if match is None:
        return command, None
    name, arg = match.groups()
    return name, arg


def refused_search(t: WebShopTemplates, command: str) -> bool:
    """A search whose first word is a special engine command (``<r>``, ``<a>``, ``<c>``,
    ``<q>``; any ``<x>``): refused before the server sees it."""
    name, arg = parse_action(t, command)
    if name != "search" or arg is None:
        return False
    words = arg.lower().split()
    return bool(words) and re.fullmatch(t.special_search_pattern, words[0]) is not None


@dataclass(frozen=True)
class ParsedResponse:
    """How one raw model response becomes a shop action.

    ``command`` is the action sent to the server, or None for the frozen no-op (a format
    error, or a call without a reply); ``format_error`` is True when the response had no
    ``<action>`` pair or an empty one; ``action_valid`` is the projection's validity
    (``<think>`` present, no CJK), kept as a diagnostic only, as for ALFWorld.
    ``rejected`` is :data:`REJECTED_UNENCODABLE` for an action that is not valid UTF-8
    (a lone surrogate, which no JSON request can carry): it is never sent, and
    ``command`` holds it with U+FFFD in place of each surrogate, for the history.
    """

    command: str | None
    format_error: bool
    action_valid: bool
    rejected: str | None = None


def without_surrogates(text: str) -> str:
    """``text`` with U+FFFD in place of each lone surrogate (the only code points a str
    holds that UTF-8 cannot encode)."""
    return "".join("\ufffd" if 0xD800 <= ord(c) <= 0xDFFF else c for c in text)


def parse_response(t: WebShopTemplates, raw: str | None) -> ParsedResponse:
    """The response's action, or the frozen no-op. The action-tag check is the one
    ALFWorld uses (SkillOpt ``rollout.py:38-40``), case-insensitive as verl-agent's
    projection reads the tags (``action_tag_pattern``); the projection only ever sees a
    response that has them, so its fallback (the reply's last 20 characters) never
    runs."""
    response = (raw or "").strip()
    if not response or re.search(t.action_tag_pattern, response, re.DOTALL) is None:
        return ParsedResponse(None, True, False)
    commands, valids = webshop_projection([response])
    command = commands[0].strip()
    if not command:
        return ParsedResponse(None, True, False)
    try:
        command.encode("utf-8")
    except UnicodeEncodeError:
        return ParsedResponse(
            without_surrogates(command), False, bool(valids[0]), REJECTED_UNENCODABLE
        )
    return ParsedResponse(command, False, bool(valids[0]))


# ---------------------------------------------------------------------------
# Frozen splits
# ---------------------------------------------------------------------------


def item_id(goal_index: int) -> str:
    return f"{ITEM_ID_PREFIX}{goal_index}"


def load_split_manifest(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, Any]:
    """The released split counts and SHA-256 checksums."""
    protocol.verify_frozen_assets(MANIFEST_FILE, root)
    data = json.loads(protocol.read_protocol_text(MANIFEST_FILE, root))
    if not isinstance(data, dict) or data.get("status") not in (
        MANIFEST_FINAL,
        MANIFEST_PROVISIONAL,
    ):
        raise WebShopEnvError(f"{MANIFEST_FILE} has no valid status")
    return data


def load_split(
    split: str, root: Path = protocol.SKILL_BENCH_DIR
) -> list[dict[str, Any]]:
    """Rows of ``envs/splits/webshop/<split>.json`` (scorer-side: they hold the gold)."""
    if split not in SPLIT_NAMES:
        raise ValueError(
            f"unknown WebShop split {split!r}; expected one of {SPLIT_NAMES}"
        )
    final = load_split_manifest(root)["status"] == MANIFEST_FINAL
    protocol.verify_frozen_assets(f"{SPLIT_DIR}/{split}.json", root)
    rows = json.loads(protocol.read_protocol_text(f"{SPLIT_DIR}/{split}.json", root))
    if not isinstance(rows, list):
        raise WebShopEnvError(f"{SPLIT_DIR}/{split}.json must be a JSON list")
    items: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        index = row.get("goal_index") if isinstance(row, Mapping) else None
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or row.get("id") != item_id(index)
            or not isinstance(row.get("asin"), str)
            or not isinstance(row.get("instruction"), str)
        ):
            raise WebShopEnvError(
                f"{split} item {position} needs goal_index, id 'fixed_<goal_index>', "
                "asin and instruction"
            )
        if final and not all(name in row for name in MANIFEST_FIELDS):
            raise WebShopEnvError(
                f"{split} item {row['id']} lacks a field of the final manifest "
                f"({', '.join(n for n in MANIFEST_FIELDS if n not in row)})"
            )
        items.append(dict(row))
    if len({row["id"] for row in items}) != len(items):
        raise WebShopEnvError(f"duplicate ids in the {split} manifest")
    return items


def split_files_sha256(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, str]:
    """sha256 of every manifest file, by name: what the server reports it verified."""
    directory = protocol.protocol_asset(f"{SPLIT_DIR}/split_manifest.json", root).parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.json"))
    }


# ---------------------------------------------------------------------------
# The server client (async; one per episode)
# ---------------------------------------------------------------------------

#: ``factory(base_url, timeout_s) -> httpx.AsyncClient``; tests pass a MockTransport client.
ClientFactory = Callable[[str, float], httpx.AsyncClient]


def default_client(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, timeout=timeout_s, trust_env=False)


def default_server_url() -> str:
    return os.environ.get(SERVER_URL_ENV) or DEFAULT_SERVER_URL


class ServerClient:
    """The WebShop server's HTTP API (``server.py``). Transport errors, timeouts, 5xx and
    a lost episode raise :class:`WebShopServerError` (INFRA); any other refusal is a
    request this harness should not have sent and raises :class:`WebShopEnvError`."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = REQUEST_TIMEOUT_S,
        factory: ClientFactory = default_client,
    ) -> None:
        self.base_url = base_url
        self._client = factory(base_url, timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        try:
            response = await self._client.request(
                method, path, json=dict(body) if body is not None else None
            )
        except httpx.HTTPError as exc:
            raise WebShopServerError(
                f"WebShop server {self.base_url}: {method} {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return response.status_code, payload if isinstance(payload, dict) else {}

    def _raise(self, method: str, path: str, status: int, payload: Mapping[str, Any]):
        detail = (
            f"WebShop server {method} {path}: HTTP {status}: {payload.get('error')}"
        )
        if status >= 500 or (status == 404 and path.endswith("/step")):
            # Down, not ready, an upstream error of one request, or an episode the
            # server no longer knows (it restarted, or reaped it as idle).
            raise WebShopServerError(detail)
        raise WebShopEnvError(detail)

    async def call(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        status, payload = await self._request(method, path, body)
        if status != 200:
            self._raise(method, path, status, payload)
        return payload

    async def ready(self) -> tuple[int, dict[str, Any]]:
        """``GET /ready``: its status code and payload (503 while loading)."""
        return await self._request("GET", "/ready")

    async def delete(self, episode: str) -> None:
        """``DELETE /episodes/<id>``; an episode the server no longer has is fine."""
        status, payload = await self._request("DELETE", f"/episodes/{episode}")
        if status not in (200, 404):
            self._raise("DELETE", f"/episodes/{episode}", status, payload)


def _owner() -> str:
    return f"{os.getpid()}@{socket.gethostname()}"


# ---------------------------------------------------------------------------
# Episode and environment
# ---------------------------------------------------------------------------


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class WebShopEpisode:
    """One shopping episode: ``reset`` once, then ``step`` with each raw model response
    until ``done`` (Buy Now or the step cap)."""

    def __init__(
        self,
        *,
        templates: WebShopTemplates,
        goal_index: int,
        instruction_text: str | None,
        server_url: str,
        client_factory: ClientFactory,
        max_steps: int,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._t = templates
        self._goal_index = goal_index
        self._expected_instruction = instruction_text
        self._server_url = server_url
        self._factory = client_factory
        self._timeout = timeout_s
        self._max_steps = max_steps
        self._lock = asyncio.Lock()
        self._inflight: asyncio.Future[Any] | None = None
        self._client: ServerClient | None = None
        self._episode: str | None = None
        self._started = False
        self._closed = False
        self._broken = False
        self._done = False
        self._purchased = False
        self._reward: float | None = None
        self._steps = 0
        self._format_errors = 0
        self._task = ""
        self._page = ""  # the raw page, as the server returned it
        self._actions: tuple[str, ...] = ()
        #: The actor's conversation without the memory: append-only, each message frozen
        #: as it was first sent.
        self._conversation: list[Mapping[str, Any]] = []

    # --- read-only state (harness side) ------------------------------------------

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def format_errors(self) -> int:
        return self._format_errors

    @property
    def done(self) -> bool:
        return self._done

    @property
    def session_prefix(self) -> str | None:
        return self._episode

    # --- server calls -------------------------------------------------------------

    async def _call(self, call: Coroutine[Any, Any, _T]) -> _T:
        """One server request, shielded: a cancelled or failed request leaves the
        server-side episode in an unknown state (a step may have run), so the episode
        refuses further calls (the item is re-run, H8)."""
        if self._broken:
            call.close()  # never sent
            raise WebShopServerError(
                "a server call of this episode was interrupted or failed"
            )
        future: asyncio.Future[_T] = asyncio.ensure_future(call)
        # A request whose caller was cancelled may still fail later: its error is read
        # here, so it is never logged as "Future exception was never retrieved".
        future.add_done_callback(_retrieve_exception)
        self._inflight = future
        try:
            result = await asyncio.shield(future)
        except BaseException:
            self._broken = True
            raise
        if self._closed:
            raise WebShopEnvError("the episode was closed during a server call")
        return result

    def _read_page(self, reply: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
        page, available = reply.get("observation"), reply.get("available_actions")
        if not isinstance(page, str) or not isinstance(available, Mapping):
            self._broken = True
            raise WebShopEnvError("the server's page or available actions are missing")
        return page, list_available_actions(self._t, available)

    def _formatted(self) -> str:
        return format_observation(self._t, self._page, self._task)

    def _fields(self, **extra: Any) -> Mapping[str, Any]:
        return _freeze(
            {
                "observation": self._formatted(),
                "available_actions": self._actions,
                "task_description": self._task,
                "conversation": tuple(self._conversation),
                "step_count": self._steps,
                **extra,
            }
        )

    def _private(self) -> Mapping[str, Any]:
        return _freeze(
            {
                "reward": self._reward,
                "purchased": self._purchased,
                "steps": self._steps,
            }
        )

    # --- Episode protocol ------------------------------------------------------------

    async def reset(self) -> Reset:
        async with self._lock:
            if self._closed:
                raise RuntimeError("episode is closed")
            if self._started:
                raise RuntimeError("reset() may be called only once per episode")
            self._started = True
            self._client = ServerClient(
                self._server_url, timeout_s=self._timeout, factory=self._factory
            )
            prefix = f"{uuid.uuid4().hex}-"
            self._episode = prefix
            reply = await self._call(
                self._client.call(
                    "POST",
                    "/episodes",
                    {
                        "goal_index": self._goal_index,
                        "session_prefix": prefix,
                        "owner": _owner(),
                    },
                )
            )
            if reply.get("episode") != prefix:
                self._broken = True
                raise WebShopEnvError("the server answered for another episode")
            self._page, self._actions = self._read_page(reply)
            try:
                self._task = extract_task(self._t, self._page)
            except WebShopEnvError:
                self._broken = True
                raise
            if (
                self._expected_instruction is not None
                and self._task != self._expected_instruction
            ):
                self._broken = True
                raise protocol.ProtocolDriftError(
                    f"goal {self._goal_index}'s instruction on the server differs from "
                    "the frozen manifest"
                )
            first = first_prompt(self._t, self._task, self._formatted(), self._actions)
            self._conversation = [_freeze({"role": "user", "content": first})]
            return Reset(
                instruction=self._task,
                # The first message is the raw page, which states the instruction.
                messages=(observation_message(self._t, self._page, self._actions),),
                fields=self._fields(),
                private=self._private(),
            )

    async def step(self, action: Action) -> StepResult:
        if isinstance(action, Mapping) and action.get(NO_REPLY_KEY) is True:
            # A model call without a reply (rollout: call_error_response): the no-op.
            parsed = ParsedResponse(None, False, False)
        elif isinstance(action, str):
            parsed = parse_response(self._t, action)
        else:
            raise TypeError("a WebShop action is the model's raw response text")
        said = reply_turn(self._t, action)  # what the actor's conversation keeps
        async with self._lock:
            if self._closed:
                raise RuntimeError("episode is closed")
            if self._client is None or self._episode is None:
                raise RuntimeError("call reset() before step()")
            if self._done:
                raise RuntimeError("the episode is over")
            accepted = False
            command = parsed.command
            rejected = parsed.rejected
            shop_error: str | None = None
            if (
                command is not None
                and rejected is None
                and refused_search(self._t, command)
            ):
                rejected = REJECTED_SPECIAL_SEARCH
            if command is not None and rejected is None:
                reply = await self._call(
                    self._client.call(
                        "POST", f"/episodes/{self._episode}/step", {"action": command}
                    )
                )
                accepted = bool(reply.get("accepted"))
                if reply.get("shop_error") is not None:
                    # The shop raised on the action and restored the session: the page
                    # stays (the reply carries it), as for an action it does not accept.
                    shop_error = str(reply["shop_error"])
                    rejected = REJECTED_SHOP_ERROR
                    if accepted or reply.get("done") is not False:
                        self._broken = True
                        raise WebShopEnvError(
                            "the server reported a shop error on an accepted action"
                        )
                if reply.get("done") is True:
                    reward = reply.get("reward")
                    if not isinstance(reward, (int, float)) or not 0.0 <= reward <= 1.0:
                        self._broken = True
                        raise WebShopEnvError("the server's reward is not in [0, 1]")
                    self._purchased, self._reward = True, float(reward)
                else:
                    self._page, self._actions = self._read_page(reply)
            self._steps += 1
            self._format_errors += int(parsed.format_error)
            capped = self._steps >= self._max_steps
            truncated = capped and not self._purchased
            self._done = self._purchased or capped
            if self._purchased:
                assert self._reward is not None
                messages: tuple[Message, ...] = (final_message(self._t, self._reward),)
            else:
                messages = (
                    observation_message(self._t, self._formatted(), self._actions),
                )
                if truncated:
                    # The cap's 0.00 closes the episode for the learners, as the score
                    # line closes one that bought.
                    messages = (*messages, cap_message(self._t))
            # The actor's conversation grows by this step's reply and, unless the
            # episode is over, the next step's page; never the score or cap line.
            turns = [_freeze({"role": "assistant", "content": said})]
            if not self._done:
                following = turn_prompt(
                    self._t,
                    self._task,
                    self._steps + 1,
                    self._formatted(),
                    self._actions,
                )
                turns.append(_freeze({"role": "user", "content": following}))
            self._conversation += turns
            return StepResult(
                messages=messages,
                done=self._done,
                fields=self._fields(
                    action=command if command is not None else self._t.noop_action,
                    format_error=parsed.format_error,
                    action_valid=parsed.action_valid,
                    accepted=accepted,
                    rejected=rejected,
                    shop_error=shop_error,
                    truncated=truncated,
                ),
                private=self._private(),
            )

    async def close(self) -> None:
        """Delete the server-side episode (every session of its prefix, the
        constructor's included) and close the client. Idempotent; a server that is gone
        is not an error here: it drops idle sessions by itself."""
        self._closed = True
        inflight = self._inflight
        if inflight is not None and not inflight.done():
            await asyncio.wait([inflight])
        client, self._client = self._client, None
        if client is None:
            return
        try:
            if self._episode is not None:
                await client.delete(self._episode)
        except (WebShopServerError, WebShopEnvError):
            pass
        finally:
            await client.aclose()

    async def __aenter__(self) -> WebShopEpisode:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class WebShopEnv:
    """WebShop tasks (40/20/140), episodes and scoring."""

    benchmark = BENCHMARK

    def __init__(
        self,
        *,
        server_url: str | None = None,
        client_factory: ClientFactory | None = None,
        max_steps: int = protocol.WEBSHOP_MAX_STEPS,
        root: Path = protocol.SKILL_BENCH_DIR,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        if not 0 < max_steps <= protocol.WEBSHOP_MAX_STEPS:
            raise ValueError(f"max_steps must be in 1..{protocol.WEBSHOP_MAX_STEPS}")
        self.root = root
        self.templates = WebShopTemplates.load(root)
        protocol.verify_frozen_assets(SPLIT_DIR + "/", root)
        protocol.verify_frozen_assets(SERVER_DIR + "/", root)
        for vendored in VENDOR_FILES:  # the projection decides what is executed
            protocol.verify_frozen_assets(vendored, root)
        self.max_steps = max_steps
        self.server_url = server_url or default_server_url()
        self._factory: ClientFactory = client_factory or default_client
        self._timeout = timeout_s

    # --- tasks and episodes ------------------------------------------------------------

    def manifest_status(self) -> str:
        return str(load_split_manifest(self.root)["status"])

    def load_tasks(self, split: Split) -> list[Task]:
        rows = load_split(split, self.root)
        expected = protocol.SPLIT_SIZES[BENCHMARK][split]
        if len(rows) != expected:
            raise protocol.ProtocolDriftError(
                f"WebShop {split} has {len(rows)} items, protocol says {expected}"
            )
        return [Task(BENCHMARK, split, row["id"], _freeze(row)) for row in rows]

    def episode(self, task: Task) -> WebShopEpisode:
        if task.benchmark != BENCHMARK:
            raise ValueError(f"not a WebShop task: {task.benchmark}")
        expected = task.raw.get("instruction_text")
        # Only the goal index reaches the episode (and the frozen instruction it must
        # see); the gold stays with the scorer. Upstream's text mode strips every page
        # text, and an instruction without a price clause can end in a space.
        return WebShopEpisode(
            templates=self.templates,
            goal_index=int(task.raw["goal_index"]),
            instruction_text=expected.strip() if isinstance(expected, str) else None,
            server_url=self.server_url,
            client_factory=self._factory,
            max_steps=self.max_steps,
            timeout_s=self._timeout,
        )

    def call_error_response(self) -> Mapping[str, Any]:
        """The action of a model call without a reply: the frozen no-op (no format
        error; the model failure is recorded by the rollout)."""
        return MappingProxyType({NO_REPLY_KEY: True})

    def actor_messages(
        self, fields: Mapping[str, Any], memory: str = ""
    ) -> list[Message]:
        """The actor call for the current state, with ``memory`` in the Q21 slot."""
        return render_actor_messages(self.templates, fields, memory)

    def agent_context(self) -> str:
        """The actor's prompt template, the same for every item: what the rollout
        hands methods as ``TaskView.agent_context``."""
        return agent_context(self.templates)

    # --- scorer side -----------------------------------------------------------------

    def score(
        self, task: Task, final_answer: str | None, private: Mapping[str, Any]
    ) -> Score:
        """The shop's reward after Buy Now; an episode the step cap ended without a
        purchase is unanswered and scores 0. ``final_answer`` is unused."""
        purchased, reward, steps = (
            private.get("purchased"),
            private.get("reward"),
            private.get("steps"),
        )
        if not isinstance(purchased, bool) or not isinstance(steps, int):
            raise ValueError(
                "WebShop scoring needs the final step's private['purchased'] (bool) and "
                "private['steps'] (int)"
            )
        metrics: dict[str, float] = {}
        best = task.raw.get("best_reward")
        if isinstance(best, (int, float)) and not isinstance(best, bool):
            metrics["best_reward"] = float(best)
        if not purchased:
            return Score(
                primary=0.0,
                metrics=_freeze({"success": 0.0, SUCCESS_WITHIN_KEY: 0.0, **metrics}),
                answered=False,
            )
        if not isinstance(reward, (int, float)) or not 0.0 <= reward <= 1.0:
            raise ValueError("a purchase needs private['reward'] in [0, 1]")
        success = 1.0 if reward == 1.0 else 0.0
        within = success if steps <= protocol.WEBSHOP_DIAGNOSTIC_STEPS else 0.0
        return Score(
            primary=float(reward),
            metrics=_freeze(
                {"success": success, SUCCESS_WITHIN_KEY: within, **metrics}
            ),
            answered=True,
        )

    def label_strings(self, task: Task) -> list[str]:
        """Nothing beyond the raw id (which the audit adds itself): the
        target's ASIN, title and options appear on the pages legitimately, and its query
        and category collide with ordinary title words. WebShop's leak check is
        structural instead (:meth:`leak_problems`)."""
        return []

    def visible_by_design(self, task: Task) -> list[str]:
        """The instruction: methods receive it in the first message."""
        text = task.raw.get("instruction_text") or task.raw.get("instruction")
        return [str(text).strip()] if text else []

    def taxonomy_labels(self, task: Task) -> list[str]:
        """The goal's product category: shared by many items, not an identifier."""
        category = task.raw.get("category")
        return [str(category)] if category else []

    def answers(self, task: Task) -> list[str]:
        """The gold ASIN, for the audit's X26 check of method outputs."""
        return [str(task.raw["asin"])]

    def leak_problems(
        self, messages: Sequence[Mapping[str, Any]]
    ) -> list[tuple[int, str]]:
        """The structural leak check of one episode's method-visible messages."""
        return leak_problems(self.templates, messages)

    # --- preflight ---------------------------------------------------------------------

    def expected_server(self) -> dict[str, Any]:
        """What ``/ready`` must report: the frozen manifest's files, its price table and
        Lucene index, the seed, the engine, the data and the Docker context of this
        checkout's server directory."""
        source = json.loads(protocol.read_protocol_text(SOURCE_FILE, self.root))
        return {
            "api_version": SERVER_API_VERSION,
            "mode": "serve",
            "seed": protocol.SEED,
            "goal_count": protocol.WEBSHOP_GOAL_COUNT,
            "price_table_sha256": source.get("price_table_sha256"),
            "manifest_files": split_files_sha256(self.root),
            "upstream_commit": protocol.WEBSHOP_SERVER["upstream_commit"],
            "spacy_model": protocol.WEBSHOP_SERVER["spacy_model"],
            "fuzzy_backend": protocol.WEBSHOP_SERVER["fuzzy_backend"],
            "data": {
                name: pin["sha256"]
                for name, pin in protocol.WEBSHOP_DATA["files"].items()
            },
            "index_sha256": source.get("index_sha256"),
            "context": context_digest(self.root),
        }

    async def preflight(self) -> dict[str, Any]:
        """Before a cell starts: a final manifest, a ready server that verified exactly
        this manifest, and the engine, data, index and Docker context the protocol and
        this checkout pin. Returns what the run record keeps (with the server's image id
        under ``sandbox_image_id``, which the audit compares across cells, its build
        record, and the container's status and restart count, :func:`container_state`).

        Raises :class:`WebShopServerError` (INFRA: the cell runs again later) when the
        server is unreachable or still loading, :class:`WebShopEnvError` when it failed
        its own checks or is not a serving server (a person fixes and restarts it), and
        :class:`~evals.skill_bench.protocol.ProtocolDriftError` when it differs from the
        protocol, the frozen manifest or this checkout."""
        status = self.manifest_status()
        if status != MANIFEST_FINAL:
            raise WebShopEnvError(
                f"the WebShop manifest is {status}: restore the released final files "
                "and run tools/prepare_webshop.py check"
            )
        expected = self.expected_server()
        if not _is_sha256(expected["index_sha256"]):
            raise WebShopEnvError(
                f"{SOURCE_FILE} records no index_sha256: restore the released source file "
                "and run tools/prepare_webshop.py check"
            )
        client = ServerClient(
            self.server_url, timeout_s=READY_TIMEOUT_S, factory=self._factory
        )
        try:
            code, payload = await client.ready()  # unreachable: WebShopServerError
        finally:
            await client.aclose()
        state = payload.get("state")
        if code != 200 or state != "ready":
            detail = f"the WebShop server at {self.server_url} is {state!r}" + (
                f": {payload.get('problems')}" if payload.get("problems") else ""
            )
            if state == "loading":
                raise WebShopServerError(detail)
            raise WebShopEnvError(
                detail + " (fix it, then restart it: python -m "
                "evals.skill_bench.envs.webshop --stop, --start)"
            )
        info = payload.get("info") if isinstance(payload.get("info"), Mapping) else {}
        problems = server_problems(info, expected)
        if problems:
            raise protocol.ProtocolDriftError(
                "the WebShop server differs from the protocol: " + "; ".join(problems)
            )
        container = await container_state()
        return {
            "server_url": self.server_url,
            "sandbox_image_id": await container_image_id(),
            "manifest_status": status,
            "server": {
                key: info.get(key)
                for key in ("api_version", "seed", "goal_count", "price_table_sha256")
            }
            | {"build": info.get("build"), "load_s": info.get("load_s")},
            # How often Docker brought the server back so far (RESTART_POLICY; record,
            # don't block). None without the docker CLI or under another name.
            "container": None
            if container is None or not container["ours"]
            else {
                k: container[k] for k in ("status", "restart_count", "restart_policy")
            },
        }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def server_problems(info: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """How a ready server's ``/ready`` info differs from ``expected``
    (:meth:`WebShopEnv.expected_server`; pure)."""
    build = info.get("build") if isinstance(info.get("build"), Mapping) else {}
    manifest = info.get("manifest") if isinstance(info.get("manifest"), Mapping) else {}
    found = {
        "api_version": info.get("api_version"),
        "mode": info.get("mode"),
        "seed": info.get("seed"),
        "goal_count": info.get("goal_count"),
        "price_table_sha256": info.get("price_table_sha256"),
        "manifest_files": manifest.get("files"),
        "upstream_commit": build.get("upstream_commit"),
        "spacy_model": info.get("spacy_model"),
        "fuzzy_backend": info.get("fuzzy_backend"),
        "data": build.get("data"),
        "index_sha256": build.get("index_sha256"),
        "context": build.get("context"),
    }
    return [
        f"{key} is {found[key]!r}, expected {want!r}"
        for key, want in expected.items()
        if found.get(key) != want
    ]


# ---------------------------------------------------------------------------
# The server container: build, start, stop, status
# ---------------------------------------------------------------------------


def context_digest(root: Path = protocol.SKILL_BENCH_DIR) -> str:
    """sha256 over every file of the server's Docker context (names and bytes): the
    image's tag, and (a build argument kept in its ``build.json``) what preflight
    requires the serving image to have been built from."""
    context = root / SERVER_DIR
    digest = hashlib.sha256()
    for path in sorted(p for p in context.iterdir() if p.is_file()):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def image_name(root: Path = protocol.SKILL_BENCH_DIR) -> str:
    """``skill-bench-webshop:<tag>``, the tag derived from every file of the Docker
    context, so a change of the server, the allowlist or the data pins is a new image."""
    return f"{IMAGE_REPOSITORY}:{context_digest(root)[:16]}"


def dockerfile_base_image(root: Path = protocol.SKILL_BENCH_DIR) -> str:
    text = protocol.read_protocol_text(f"{SERVER_DIR}/Dockerfile", root)
    found = re.search(r"^FROM\s+(\S+)", text, re.MULTILINE)
    if found is None:
        raise WebShopEnvError("the server Dockerfile has no FROM line")
    return found.group(1)


def build_argv(root: Path = protocol.SKILL_BENCH_DIR) -> list[str]:
    if dockerfile_base_image(root) != protocol.WEBSHOP_SERVER["base_image"]:
        raise protocol.ProtocolDriftError(
            "the server Dockerfile is not built FROM protocol.WEBSHOP_SERVER['base_image']"
        )
    return [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "--label",
        f"{CONTAINER_LABEL_KEY}={CONTAINER_LABEL_VALUE}",
        "--build-arg",
        f"{CONTEXT_BUILD_ARG}={context_digest(root)}",
        "-t",
        image_name(root),
        str(root / SERVER_DIR),
    ]


def run_argv(
    *,
    port: int,
    prepare: bool = False,
    root: Path = protocol.SKILL_BENCH_DIR,
) -> list[str]:
    """``docker run`` of the server: loopback port only, one labelled container that
    restarts after a crash (:data:`RESTART_POLICY`), the memory cap, the protocol seed,
    and (serve mode) the manifest mounted read-only."""
    server = protocol.WEBSHOP_SERVER
    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        PREPARE_CONTAINER_NAME if prepare else CONTAINER_NAME,
        "--label",
        f"{CONTAINER_LABEL_KEY}={CONTAINER_LABEL_VALUE}",
        "--restart",
        RESTART_POLICY,
        "--memory",
        str(server["memory"]),
        "--memory-swap",
        str(server["memory"]),
        "--cpus",
        str(server["cpus"]),
        "-p",
        f"127.0.0.1:{port}:{server['container_port']}",
    ]
    if not prepare:
        manifest = (root / SPLIT_DIR).resolve()
        argv += ["--mount", f"type=bind,source={manifest},target=/manifest,readonly"]
    argv += [
        image_name(root),
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        str(server["container_port"]),
        "--seed",
        str(server["seed"]),
        "--session-idle-s",
        str(server["session_idle_s"]),
    ]
    argv += ["--prepare"] if prepare else ["--manifest-dir", "/manifest"]
    return argv


async def _docker(
    *argv: str, timeout_s: float = DOCKER_TIMEOUT_S, capture: bool = True
) -> tuple[int, str]:
    """Run the docker CLI (async); ``(exit code, stdout)``, ``(-1, "")`` without docker.
    ``capture=False`` (a build) shows docker's own output instead."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE if capture else None,
            stderr=asyncio.subprocess.DEVNULL if capture else None,
        )
    except OSError:
        return -1, ""
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return -1, ""
    return process.returncode or 0, (out or b"").decode("utf-8", "replace").strip()


async def container_image_id(name: str = CONTAINER_NAME) -> str | None:
    """The image id of the running server container, or None (no docker CLI here, or
    another container under that name)."""
    code, out = await _docker(
        "docker",
        "inspect",
        "--format",
        '{{index .Config.Labels "' + CONTAINER_LABEL_KEY + '"}} {{.Image}}',
        name,
    )
    if code != 0 or not out:
        return None
    label, _, image = out.partition(" ")
    return image or None if label == CONTAINER_LABEL_VALUE else None


async def stop_container(name: str = CONTAINER_NAME) -> bool:
    """Remove the server container, only when it carries this harness's label (never
    another container the host runs)."""
    code, out = await _docker(
        "docker",
        "inspect",
        "--format",
        '{{index .Config.Labels "' + CONTAINER_LABEL_KEY + '"}}',
        name,
    )
    if code != 0 or out != CONTAINER_LABEL_VALUE:
        return False
    code, _ = await _docker("docker", "rm", "-f", name)
    return code == 0


#: ``docker inspect`` fields :func:`container_state` reads, ``|``-separated.
_STATE_FORMAT = (
    '{{index .Config.Labels "'
    + CONTAINER_LABEL_KEY
    + '"}}|{{.State.Status}}|{{.RestartCount}}|{{.HostConfig.RestartPolicy.Name}}'
    + "|{{.Image}}"
)


async def container_state(name: str = CONTAINER_NAME) -> dict[str, Any] | None:
    """What Docker says of the container ``name``: ``ours`` (it carries this harness's
    label), ``status`` (``running``, ``exited`` ...), ``restart_count`` (how often its
    restart policy brought it back), ``restart_policy`` and ``image``; None when there is
    no such container (or no docker CLI)."""
    code, out = await _docker("docker", "inspect", "--format", _STATE_FORMAT, name)
    if code != 0 or not out:
        return None
    parts = out.split("|")
    if len(parts) != 5:
        return None
    label, status, restarts, policy, image = parts
    return {
        "ours": label == CONTAINER_LABEL_VALUE,
        "status": status,
        "restart_count": int(restarts) if restarts.isdigit() else None,
        "restart_policy": policy or None,
        "image": image or None,
    }


async def start_container(
    *, port: int, prepare: bool = False, root: Path = protocol.SKILL_BENCH_DIR
) -> tuple[bool, str]:
    """Start the server container, or keep the one that runs (``--start``: a relaunch
    after a reboot must not fail on the container left behind):

    * none under the name: ``docker run`` (:func:`run_argv`);
    * this harness's, running: left as it is (preflight checks it serves this
      checkout's build and manifest);
    * this harness's, not running (exited after a reboot under the former
      ``on-failure`` policy, or dead): removed and run again with this checkout's
      image, options and restart policy;
    * another container under the name: refused, never touched.

    Returns whether a server container now runs, and what was done."""
    name = PREPARE_CONTAINER_NAME if prepare else CONTAINER_NAME
    state = await container_state(name)
    if state is not None and not state["ours"]:
        return False, (
            f"a container named {name} that this harness did not start exists (no "
            f"{CONTAINER_LABEL_KEY}={CONTAINER_LABEL_VALUE} label): remove or rename it"
        )
    if state is not None and state["status"] == "running":
        return True, (
            f"{name} already runs (image {state['image']}, restart policy "
            f"{state['restart_policy']}, restarted {state['restart_count']} time(s)); "
            "left as it is"
        )
    replaced = ""
    if state is not None:
        if not await stop_container(name):
            return False, f"cannot remove the {state['status']} container {name}"
        replaced = (
            f"removed the {state['status']} container {name} (restart policy "
            f"{state['restart_policy']}); "
        )
    code, out = await _docker(*run_argv(port=port, prepare=prepare, root=root))
    if code != 0:
        return False, replaced + "docker run failed"
    return True, replaced + f"started {name} ({out})"


async def wait_ready(
    server_url: str,
    *,
    timeout_s: float,
    poll_s: float = 10.0,
    factory: ClientFactory = default_client,
) -> dict[str, Any]:
    """Poll ``/ready`` until the server is ``ready`` (or ``prepare``); raise on ``failed``
    or after ``timeout_s`` (loading the product set takes minutes)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        client = ServerClient(server_url, timeout_s=READY_TIMEOUT_S, factory=factory)
        try:
            _, payload = await client.ready()
        except WebShopServerError:
            payload = {"state": "unreachable"}
        finally:
            await client.aclose()
        state = payload.get("state")
        if state in ("ready", "prepare"):
            return payload
        if state == "failed":
            raise WebShopServerError(
                f"the WebShop server failed: {payload.get('problems')}"
            )
        if loop.time() >= deadline:
            raise WebShopServerError(
                f"the WebShop server is still {state!r} after {timeout_s:g} s"
            )
        await asyncio.sleep(poll_s)


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="The WebShop server container: build, start, stop, status."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="docker build (x86_64)")
    action.add_argument(
        "--start",
        action="store_true",
        help="start the server; one already running is left as it is, a stopped one "
        "(after a reboot) is replaced",
    )
    action.add_argument(
        "--start-prepare",
        action="store_true",
        help="start it in inspection mode (goals and diagnostics; no episodes)",
    )
    action.add_argument(
        "--stop", action="store_true", help="remove the serve container"
    )
    action.add_argument(
        "--stop-prepare", action="store_true", help="remove the prepare container"
    )
    action.add_argument("--status", action="store_true")
    action.add_argument("--print-argv", action="store_true", help="show the commands")
    parser.add_argument("--port", type=int, default=18560)
    parser.add_argument("--wait-s", type=float, default=0.0)
    args = parser.parse_args(argv)
    url = f"http://127.0.0.1:{args.port}"

    async def run() -> int:
        if args.print_argv:
            print(" ".join(build_argv()))
            print(" ".join(run_argv(port=args.port)))
            print(" ".join(run_argv(port=args.port, prepare=True)))
            return 0
        if args.build:
            code, _ = await _docker(*build_argv(), timeout_s=24 * 3600, capture=False)
            return 0 if code == 0 else 1
        if args.start or args.start_prepare:
            running, what = await start_container(
                port=args.port, prepare=args.start_prepare
            )
            print(what)
            if not running:
                return 1
            if args.wait_s > 0:
                payload = await wait_ready(url, timeout_s=args.wait_s)
                print(json.dumps(payload, indent=1, sort_keys=True))
            return 0
        if args.stop or args.stop_prepare:
            name = PREPARE_CONTAINER_NAME if args.stop_prepare else CONTAINER_NAME
            return 0 if await stop_container(name) else 1
        client = ServerClient(url, timeout_s=READY_TIMEOUT_S)
        try:
            code, payload = await client.ready()
        finally:
            await client.aclose()
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0 if code == 200 else 1

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(_main())
