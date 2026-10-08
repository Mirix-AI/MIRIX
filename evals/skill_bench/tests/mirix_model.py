"""The fake model MIRIX talks to in the harness's real-MIRIX tests (no provider is called).

A real MIRIX server reaches its model through the real gateway (roles ``mirix`` and
``embedding``), and the gateway forwards to ``tests/fakes.py``'s ``FakeUpstream``. The
responder here plays MIRIX's three model roles, telling them apart by the request, as
MIRIX's own offline e2e test does (``tests/test_skill_learning_e2e_pg.py``):

* **topic extraction** (M7.1, the forced tool ``update_topic_and_time``): a fixed topic;
  :attr:`MirixModel.fail_extraction_for` makes the call fail with an upstream 502 for
  every conversation that contains one of its strings (the gateway retries it and gives
  up: an infrastructure failure MIRIX falls back from, rule N1);
* **the distiller** (M4, system prompt ``You are the Experience Distiller``): one
  ``worth_learning`` experience per call, naming :data:`MIRIX_LEARNED_MARK`; each call's
  messages are kept (:attr:`MirixModel.distilled`: what the distiller was shown, its
  ``<skills_served>`` and ``<agent_context>`` blocks included). By default it
  answers in the legacy format (a bare list: no outcome, no skill feedback); with
  :attr:`MirixModel.p16_replies` it answers with the reply object real models send
  (``outcome``, ``outcome_evidence``, ``skill_feedback``, ``experiences``), its quotes
  copied from the transcript it was shown (:meth:`MirixModel._distill_p16`);
* **the curator** (M5, tools ``skill_create`` ... ``finish_memory_update``): creates
  :data:`SKILL` in its batch's first turn, then finishes; with
  :attr:`MirixModel.pool_curator` it cites and triages what its batch shows instead
  (the experience pool: :meth:`MirixModel._curate_pool`), and with
  :attr:`MirixModel.unverified_citation` one of its citations quotes text the write
  does not hold (B12). Every batch's payload is kept
  (:attr:`MirixModel.curator_payloads`);
* **embeddings**: the same unit vector for every text (3,072 dimensions, like
  gemini-embedding-001), so every skill passes MIRIX's similarity gate (M7.3) and is
  retrieved; :attr:`MirixModel.fail_embeddings` makes them fail with a 502.

Every other request (the actor's, J's) goes to ``fallback`` (the default reply when it
is None).
"""

from __future__ import annotations

import copy
import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from evals.skill_bench.tests.fakes import Reply, chat_body

EXTRACTION_TOOL = "update_topic_and_time"
DISTILLER_PROMPT_HEAD = "You are the Experience Distiller"
CURATOR_FINISH_TOOL = "finish_memory_update"
CURATOR_BATCH_HEAD = "## Skill curation"
#: The heading of a curator batch's usage records (MIRIX's
#: ``skill_experience_curator.SKILL_USAGE_HEADING``).
USAGE_SECTION_HEAD = "## Usage records of the skills these experiences concern"
SKILL_CREATE_TOOL = "skill_create"
TRIAGE_TOOL = "experience_triage"
EXTRACTED_TOPIC = "answer the task; check the source; verify the value"
EMBEDDING_DIM = 3072
#: Text only MIRIX's learned skill carries (its instructions); the e2e actor answers
#: better when its prompt has it (``e2e_support.LEARNED_MARKS``).
MIRIX_LEARNED_MARK = "Verify every value against its source before answering"
SKILL: Mapping[str, Any] = {
    "name": "verify-against-source",
    "description": "Check what the task asks for against its source before acting.",
    "instructions": (
        "1. Restate what the task asks for.\n"
        f"2. {MIRIX_LEARNED_MARK}.\n"
        "3. Give the answer in the requested format."
    ),
    "entry_type": "workflow",
    "triggers": ["a task that asks for a value from a source"],
    "examples": [],
}
#: The skill :attr:`MirixModel.pool_curator` creates in a batch that shows earlier
#: lessons (a batch without them creates :data:`SKILL`).
POOL_SKILL: Mapping[str, Any] = {
    "name": "read-the-labelled-field",
    "description": "Answer with the value next to the label the task names.",
    "instructions": (
        "1. Find the label the task names.\n"
        "2. Copy the value next to it verbatim.\n"
        "3. Give the answer in the requested format."
    ),
    "entry_type": "workflow",
    "triggers": ["a task that asks for a labelled value"],
    "examples": [],
}
UPSTREAM_502 = {"error": {"message": "fake upstream: bad gateway", "code": 502}}
#: The quote :attr:`MirixModel.unverified_citation` cites with: in no skill's text.
UNVERIFIED_QUOTE = "Ask the shopkeeper which shelf holds the item"
#: The labels MIRIX puts in front of a distilled transcript (``distill_transcript.
#: WHOLE_TRANSCRIPT_LABEL`` / ``CHUNK_TRANSCRIPT_LABEL``); its instruction follows the
#: transcript after a blank line and starts with :data:`_INSTRUCTION_HEAD`.
_TRANSCRIPT_LABELS = (
    "session_transcript (chronological):",
    "chunk_transcript (chronological):",
)
_INSTRUCTION_HEAD = "\n\nDistill "
#: A transcript row's first line (``role: text``).
_ROLE_LINE = re.compile(r"^(user|assistant|tool|system): ?(.*)$")
#: Letters and digits a quote needs: MIRIX checks quotes of 12 and more, squashed
#: (``session_experience_distiller.OUTCOME_QUOTE_MIN_CHARS``).
_QUOTE_MIN = 12


def _squash(text: str) -> str:
    """``text`` as MIRIX compares quotes: letters and digits only, case folded."""
    return "".join(c for c in text.casefold() if c.isalnum())


def transcript_messages(text: str) -> list[tuple[str, str]]:
    """The ``(role, text)`` messages of the transcript in a distiller request's user
    text (MIRIX renders each row as ``role: text``; the context blocks come before the
    transcript's label and the instruction after it)."""
    for label in _TRANSCRIPT_LABELS:
        if label in text:
            text = text.split(label, 1)[1]
            break
    cut = text.rfind(_INSTRUCTION_HEAD)
    if cut >= 0:
        text = text[:cut]
    messages: list[list[str]] = []
    for line in text.splitlines():
        match = _ROLE_LINE.match(line)
        if match:
            messages.append([match.group(1), match.group(2)])
        elif messages:
            messages[-1][1] += "\n" + line
    return [(role, body) for role, body in messages]


def _quotable(text: str) -> list[str]:
    """The lines of ``text`` long enough to quote and free of double quotes."""
    return [
        line.strip()
        for line in text.splitlines()
        if len(_squash(line)) >= _QUOTE_MIN and '"' not in line
    ]


def transcript_quotes(text: str) -> tuple[str | None, str | None]:
    """``(agent, environment)``: a line of the agent's first message, and the last line
    of a user or tool message after the task statement that neither the task statement
    nor any agent message holds (what the environment answered), from the transcript in
    a distiller request's user ``text``; None where there is none."""
    messages = transcript_messages(text)
    if not messages:
        return None, None
    task = _squash(messages[0][1]) if messages[0][0] == "user" else ""
    agent_texts = [body for role, body in messages if role == "assistant"]
    agent_squashed = [_squash(body) for body in agent_texts]
    agent = next((q for body in agent_texts for q in _quotable(body)), None)
    environment = None
    for role, body in messages[1:]:
        if role not in ("user", "tool"):
            continue
        for line in _quotable(body):
            squashed = _squash(line)
            if squashed in task or any(squashed in a for a in agent_squashed):
                continue
            environment = line
    return agent, environment


def _experience() -> dict[str, Any]:
    """The one experience every distiller reply gives (it names
    :data:`MIRIX_LEARNED_MARK`)."""
    return {
        "experience_type": "worth_learning",
        "title": "check the source before answering",
        "content": (
            "The task went well once the agent checked what was asked against its "
            f"source. {MIRIX_LEARNED_MARK}; then answer in the requested format."
        ),
        "importance": 0.8,
        "credibility": 0.9,
        "evidence": {
            "quote": "the answer matched",
            "signal_type": "user_confirmation",
        },
    }


def _citation_quote(skill: Mapping[str, Any]) -> str:
    """A quote of ``skill``'s instructions that applies its lesson: its second
    step, without the step number."""
    step = str(skill["instructions"]).split("\n")[1]
    return step.split(". ", 1)[1].rstrip(".")


#: The ids of a curator batch's experiences and earlier lessons (MIRIX's blocks,
#: ``render_experience`` / ``render_earlier_lesson``).
_EXPERIENCE_ID = re.compile(r'<experience id="([^"]+)">')
_LESSON_ID = re.compile(r'<earlier_lesson id="([^"]+)">')

Responder = Callable[[str, dict[str, Any]], "Reply | None"]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return ""


def _tool_names(body: Mapping[str, Any]) -> set[str]:
    return {
        str((tool.get("function") or {}).get("name"))
        for tool in body.get("tools") or ()
        if isinstance(tool, Mapping)
    }


def mirix_request_kind(body: Mapping[str, Any]) -> str | None:
    """``extraction``, ``distiller`` or ``curator`` for MIRIX's own chat requests."""
    tools = _tool_names(body)
    if EXTRACTION_TOOL in tools:
        return "extraction"
    if CURATOR_FINISH_TOOL in tools:
        return "curator"
    messages = body.get("messages") or []
    if messages and _text(messages[0].get("content")).startswith(DISTILLER_PROMPT_HEAD):
        return "distiller"
    return None


def _tool_reply(body: Mapping[str, Any], name: str, arguments: Any) -> Reply:
    return Reply(
        json=chat_body(
            model=str(body.get("model")),
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": f"call_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        )
    )


@dataclass
class MirixModel:
    """``FakeUpstream`` responder for MIRIX's model roles (see the module docstring)."""

    fallback: Responder | None = None
    #: Substrings of a conversation whose topic extraction fails (502, every attempt).
    fail_extraction_for: set[str] = field(default_factory=set)
    #: Every embedding request fails (502) while set.
    fail_embeddings: bool = False
    #: The curator cites and triages the ids its batch shows (:meth:`_curate_pool`).
    pool_curator: bool = False
    #: The pool curator's first write in a batch without earlier lessons also cites the
    #: batch's fourth experience, with :data:`UNVERIFIED_QUOTE` (B12: not counted, the
    #: experience stays pending).
    unverified_citation: bool = False
    #: The distiller answers with the skill-feedback reply object
    #: (:meth:`_distill_p16`).
    p16_replies: bool = False
    #: What each curator batch showed, in order (:attr:`pool_curator` only):
    #: ``{"experiences": [...], "lessons": [...]}``.
    curations: list[dict[str, list[str]]] = field(default_factory=list)
    #: The batch text of every curator batch (its user message with
    #: :data:`CURATOR_BATCH_HEAD`), once each, in order.
    curator_payloads: list[str] = field(default_factory=list)
    #: The skill-feedback replies given, in order: ``{"session_id", "served", "outcome",
    #: "agent_quote", "environment_quote"}`` (:meth:`_distill_p16`).
    p16_given: list[dict[str, Any]] = field(default_factory=list)
    kinds: Counter[str] = field(default_factory=Counter)
    #: The conversations topic extraction was asked about, in order.
    extractions: list[str] = field(default_factory=list)
    #: The messages of every distiller call, in order (:meth:`distiller_input`).
    distilled: list[list[dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Reply | None:
        with self._lock:
            if endpoint == "embeddings":
                return self._embeddings(body)
            kind = mirix_request_kind(body)
            if kind is None:
                return self.fallback(endpoint, body) if self.fallback else None
            self.kinds[kind] += 1
            if kind == "extraction":
                return self._extract(body)
            if kind == "distiller":
                return self._distill(body)
            return self._curate(body)

    def _embeddings(self, body: Mapping[str, Any]) -> Reply:
        self.kinds["embedding"] += 1
        if self.fail_embeddings:
            return Reply(status=502, json=UPSTREAM_502)
        raw = body.get("input")
        inputs = raw if isinstance(raw, list) else [raw]
        vector = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        return Reply(
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i in range(len(inputs))
                ],
                "model": body.get("model"),
                "provider": "Google",
                "usage": {
                    "prompt_tokens": 3 * len(inputs),
                    "total_tokens": 3 * len(inputs),
                    "cost": 0.000001,
                },
            }
        )

    def _extract(self, body: Mapping[str, Any]) -> Reply:
        conversation = "\n".join(
            _text(m.get("content"))
            for m in body.get("messages") or ()
            if m.get("role") == "user"
        )
        self.extractions.append(conversation)
        if any(marker in conversation for marker in self.fail_extraction_for):
            return Reply(status=502, json=UPSTREAM_502)
        return _tool_reply(body, EXTRACTION_TOOL, {"topic": EXTRACTED_TOPIC})

    def distiller_input(self, session_id: str) -> list[str]:
        """The user text of every distiller call about session ``session_id`` (MIRIX's
        transcript header names it: ``session_id: <id>``)."""
        found = []
        for messages in self.distilled:
            text = "\n".join(
                _text(m.get("content")) for m in messages if m.get("role") == "user"
            )
            if f"session_id: {session_id}\n" in text:
                found.append(text)
        return found

    def _distill(self, body: Mapping[str, Any]) -> Reply:
        self.distilled.append(copy.deepcopy(list(body.get("messages") or [])))
        if self.p16_replies:
            return self._distill_p16(body)
        return Reply(
            json=chat_body(
                model=str(body.get("model")), content=json.dumps([_experience()])
            )
        )

    def _distill_p16(self, body: Mapping[str, Any]) -> Reply:
        """The skill-feedback reply object, its quotes copied from the transcript shown
        (:func:`transcript_quotes`).

        A session served skills (``<skills_served>``): ``succeeded``, its evidence the
        environment's line; one ``skill_feedback`` entry per served skill (by its
        number), ``followed: yes`` quoting the agent's line and ``effect: helped``
        quoting the environment's: MIRIX stores both values. Any other session:
        ``failed`` quoting only the agent's own line, which MIRIX stores as unclear
        (S2: a failure needs a signal beyond the agent's claim). A transcript without
        the line a value needs (an episode whose model calls got no reply) gets
        ``unclear`` there, as a model with nothing to quote answers."""
        text = "\n".join(
            _text(m.get("content"))
            for m in body.get("messages") or ()
            if m.get("role") == "user"
        )
        session = re.search(r"^session_id: (\S+)$", text, re.MULTILINE)
        # Each served skill's header line in <skills_served> (``[1] name: ...``).
        served = len(re.findall(r"^\[\d+\] name: ", text, re.MULTILINE))
        agent, environment = transcript_quotes(text)

        def quoted(quote: str | None, value: str) -> tuple[str, str]:
            return (f'"{quote}"', value) if quote else ("", "unclear")

        if served:
            evidence, outcome = quoted(environment, "succeeded")
            followed_evidence, followed = quoted(agent, "yes")
            effect_evidence, effect = quoted(environment, "helped")
            reply: dict[str, Any] = {
                "outcome_evidence": evidence,
                "outcome": outcome,
                "skill_feedback": [
                    {
                        "skill": str(n),
                        "followed": followed,
                        "followed_evidence": followed_evidence,
                        "effect": effect,
                        "effect_evidence": effect_evidence,
                    }
                    for n in range(1, served + 1)
                ],
            }
        else:
            evidence, outcome = quoted(agent, "failed")
            reply = {"outcome_evidence": evidence, "outcome": outcome}
        reply["experiences"] = [_experience()]
        self.p16_given.append(
            {
                "session_id": session.group(1) if session else None,
                "served": served,
                "outcome": reply["outcome"],
                "agent_quote": agent,
                "environment_quote": environment,
            }
        )
        return Reply(
            json=chat_body(model=str(body.get("model")), content=json.dumps(reply))
        )

    def _curate(self, body: Mapping[str, Any]) -> Reply:
        messages = body.get("messages") or []
        starts = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and CURATOR_BATCH_HEAD in _text(m.get("content"))
        ]
        for i in starts:
            payload = _text(messages[i].get("content"))
            if payload not in self.curator_payloads:
                self.curator_payloads.append(payload)
        later = messages[starts[-1] + 1 :] if starts else []
        turn = sum(1 for m in later if m.get("role") == "assistant")
        if self.pool_curator:
            payload = _text(messages[starts[-1]].get("content")) if starts else ""
            return self._curate_pool(body, payload, turn)
        if not turn:
            return _tool_reply(body, SKILL_CREATE_TOOL, dict(SKILL))
        return _tool_reply(body, CURATOR_FINISH_TOOL, {})

    def _curate_pool(self, body: Mapping[str, Any], payload: str, turn: int) -> Reply:
        """A curator that uses the experience pool, one tool call a turn.

        A batch without earlier lessons (experiences ``e0, e1, ...``): ``skill_create``
        of :data:`SKILL` citing ``e0``, then ``experience_triage`` with ``e1`` not worth
        a skill and ``e2`` deferred (the rest are deferred by default), then finish.

        A batch with earlier lessons ``l0, l1, ...``: ``skill_create`` of
        :data:`POOL_SKILL` citing ``l0`` and ``e0``, then ``experience_triage`` with
        ``l1`` covered, ``l2`` not worth a skill and ``e1`` deferred, then finish.

        Each citation quotes the created skill's instructions (MIRIX counts a
        citation only with a quote of the text the write added); with
        :attr:`unverified_citation`, the first write of a batch without earlier lessons
        also cites ``e3`` with :data:`UNVERIFIED_QUOTE`, which MIRIX does not count
        (B12: ``citations_unverified``): nothing addressed ``e3``, so it stays pending
        for a later batch."""
        new = _EXPERIENCE_ID.findall(payload)
        lessons = _LESSON_ID.findall(payload)
        if turn == 0:
            self.curations.append({"experiences": new, "lessons": lessons})
            if lessons:
                skill, cited = POOL_SKILL, lessons[:1] + new[:1]
            else:
                skill, cited = SKILL, new[:1]
            quote = _citation_quote(skill)
            citations = [{"id": i, "quote": quote} for i in cited]
            if self.unverified_citation and not lessons and len(new) > 3:
                citations.append({"id": new[3], "quote": UNVERIFIED_QUOTE})
            return _tool_reply(
                body, SKILL_CREATE_TOOL, {**skill, "experience_ids": citations}
            )
        if turn == 1:
            if lessons:
                triage = {
                    "covered_ids": lessons[1:2],
                    "not_worth_ids": lessons[2:3],
                    "deferred_ids": new[1:2],
                }
            else:
                triage = {
                    "covered_ids": [],
                    "not_worth_ids": new[1:2],
                    "deferred_ids": new[2:3],
                }
            return _tool_reply(body, TRIAGE_TOOL, triage)
        return _tool_reply(body, CURATOR_FINISH_TOOL, {})
