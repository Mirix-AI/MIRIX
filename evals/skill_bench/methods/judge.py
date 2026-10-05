"""The shared label-free judge J, used by ReasoningBank and AWM (S1).

J decides whether one learning trajectory succeeded, from what a method may see and
nothing else: it takes a :class:`~evals.skill_bench.methods.base.TrajectoryView` (Q20) and
the cell's benchmark. It never receives gold answers, scores, ``won`` or any scorer-side
record: :meth:`Judge.judge` refuses anything but a ``TrajectoryView``.

**Prompt** (``methods/prompts/judge/``, fingerprinted; no model-visible literal here):

* system: the ReasoningBank paper's Figure 10 instruction, with only its three web task
  types rewritten into two classes (answering or looking up information;
  changing the state or taking an action). ``format.json`` records the one substitution,
  and a test diffs ``system.md`` against the verbatim original. The strict-judging
  paragraph the RB repo added later is left out (Q22: the paper wins).
* user: the RB repo's text-evaluation input (``autoeval/prompts.py`` at ed80611):

  - ``User Intent``: the Q28 ``instruction``;
  - ``Action History``: every assistant turn with an action, numbered, as ``<think>`` (the
    visible reasoning) and ``<action>`` (an ``<action>`` tag, or the ``<answer>`` element
    of a final answer). A turn without an action (no reply, no tag)
    is skipped before numbering, as RB's evaluator skips steps with an empty action
    (``evaluate_trajectory.extract_think_and_action``); the S1 learners read the same
    turns (:func:`episode_turns`, X1, X7);
  - the final state: the last <= 5 environment messages (``user``), each kept whole up
    to 40,000 characters, else head and tail with a marked cut;
  - ``Bot response to the user``: ``N/A`` (ALFWorld and WebShop change the state; they
    have no answer).

  If the whole input would not fit the model's input budget (with the M3.1 planning
  reserve left for the reply), the reasoning, action and observation texts are cut
  further, head and tail, and the verdict records it.

**Call**: gateway role ``judge``, the cell's model (H1.8), the phase ``learning`` and the
opaque task uid as the ledger item; the gateway sets everything else (no temperature:
RB's 0.0 is dropped by F8, see H7.3).

**Output**: ``Thoughts: ... / Status: success|failure``. The first line whose label is
``Status`` decides (RB's ``extract_content``), read case-insensitively and with markdown
emphasis and quotes stripped; any other value, no such line, or no reply at all makes
the verdict ``failure`` and counts as ``unparseable``. A MODEL failure of the call
(truncated or filtered reply: parsed like any reply; no reply: unparseable) is counted
too. INFRA and HARNESS failures raise
(:class:`~evals.skill_bench.agent.actor.ActorCallError`), which aborts the learning item
(H8).

Every verdict is appended to ``<work_dir>/judge.jsonl`` (task uid, status, how it was
parsed, the gateway call id, input size and cuts) so that J's agreement with the true
outcome can be computed after the experiment (H9), never during it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import Actor, ActorCallError, ActorReply
from evals.skill_bench.envs.base import Message
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import Failure, FailureKind
from evals.skill_bench.infra.processes import append_jsonl
from evals.skill_bench.methods.base import CellContext, TrajectoryView

JUDGE_ROLE = "judge"
PROMPT_DIR = "methods/prompts/judge"
FORMAT_FILE = f"{PROMPT_DIR}/format.json"
RECORD_FILE = "judge.jsonl"
#: The run phase J's calls belong to (J runs in ``after_task``, learning only).
JUDGE_PHASE = "learning"

Status = Literal["success", "failure"]
STATUSES: tuple[Status, ...] = ("success", "failure")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Frozen prompt text
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def judge_format() -> Mapping[str, Any]:
    """``format.json``: the model-visible pieces, limits, provenance and the substitution."""
    return MappingProxyType(json.loads(protocol.read_protocol_text(FORMAT_FILE)))


def _fmt(key: str) -> Any:
    return judge_format()[key]


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """J's system message (``system.md``, without the file's final newline)."""
    return protocol.read_protocol_text(f"{PROMPT_DIR}/{_fmt('system_file')}").rstrip(
        "\n"
    )


@lru_cache(maxsize=1)
def user_template() -> str:
    """J's user message template (``user.md``; RB's ``build_text_eval_prompt`` prompt)."""
    return protocol.read_protocol_text(f"{PROMPT_DIR}/{_fmt('user_file')}").rstrip("\n")


# ---------------------------------------------------------------------------
# Rendering a trajectory (pure functions)
# ---------------------------------------------------------------------------

_ACTION_TAG = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAGS = re.compile(r"</?think>", re.IGNORECASE)


def content_text(content: Any) -> str:
    """The text of a message's content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    return str(content)


@dataclass(frozen=True)
class Turn:
    """One assistant turn as J shows it: the visible reasoning and the action(s)."""

    think: str
    actions: tuple[str, ...]


def split_turn(message: Mapping[str, Any]) -> Turn:
    """Visible reasoning and actions of one assistant message (RB's think / action pair).

    * an ``<action>`` tag (ALFWorld, WebShop): the text before it, without ``<think>``
      tags, is the reasoning (RB ``evaluate_trajectory._extract_think_from_output``), the
      tag's body the action;
    * an ``<answer>`` element (a final answer): the element is the action, the rest of the
      text the reasoning;
    * anything else (no tag, or the empty turn of a call without a reply): the text is
      the reasoning and the action is empty.
    """
    text = content_text(message.get("content"))
    match = _ACTION_TAG.search(text)
    if match:
        think = _THINK_TAGS.sub("", text[: match.start()]).strip()
        return Turn(think, (match.group(1).strip(),))
    answer = _ANSWER_TAG.search(text)
    if answer:
        rest = text[: answer.start()] + text[answer.end() :]
        return Turn(_THINK_TAGS.sub("", rest).strip(), (answer.group(0),))
    return Turn(_THINK_TAGS.sub("", text).strip(), ("",))


def episode_turns(trajectory: TrajectoryView) -> list[Turn]:
    """Every assistant turn's think/action split (:func:`split_turn`) (pure). ALFWorld
    plays ``look`` for a turn without an ``<action>`` tag and WebShop a no-op, so a turn
    without a tag keeps an empty action, as does a turn without a reply."""
    return [split_turn(m) for m in trajectory.messages if m.get("role") == "assistant"]


def action_turns(trajectory: TrajectoryView) -> tuple[list[Turn], int]:
    """The turns that carry an action, and how many were skipped for having none (RB's
    evaluator and the RB / AWM learners skip a step without an action)."""
    turns = episode_turns(trajectory)
    kept = [t for t in turns if any(t.actions)]
    return kept, len(turns) - len(kept)


def cut_head_tail(text: str, max_chars: int | None) -> tuple[str, int]:
    """``text`` whole if it fits ``max_chars``; else head + marker + tail of that length.

    Returns the text and the number of characters cut. ``None`` never cuts.
    """
    total = len(text)
    if max_chars is None or total <= max_chars:
        return text, 0
    marker_format: str = _fmt("truncation_marker")
    omitted = total - max_chars
    marker = marker_format.format(omitted=omitted, total=total)
    head = tail = 0
    for _ in range(4):  # the marker's length depends on the digits of ``omitted``
        keep = max(max_chars - len(marker), 0)
        head = (keep + 1) // 2
        tail = keep - head
        if total - head - tail == omitted:
            break
        omitted = total - head - tail
        marker = marker_format.format(omitted=omitted, total=total)
    return text[:head] + marker + (text[total - tail :] if tail else ""), omitted


def render_actions(
    turns: Sequence[Turn], *, max_chars: int | None = None
) -> tuple[str, int]:
    """RB's numbered action history; returns it and the characters cut by ``max_chars``."""
    out: list[str] = []
    cut = 0
    for n, turn in enumerate(turns, start=1):
        items = []
        for action in turn.actions:
            kept, lost = cut_head_tail(action, max_chars)
            cut += lost
            items.append(_fmt("action_item").format(action=kept))
        actions = _fmt("action_joiner").join(items)
        think, lost = cut_head_tail(turn.think, max_chars)
        cut += lost
        if think:
            out.append(
                _fmt("action_with_think").format(n=n, think=think, actions=actions)
            )
        else:
            out.append(_fmt("action_without_think").format(n=n, actions=actions))
    return "".join(out), cut


def environment_messages(trajectory: TrajectoryView) -> list[Message]:
    """The environment's raw outputs in order (``user`` messages, Q20)."""
    return [m for m in trajectory.messages if m.get("role") == "user"]


def render_observations(
    trajectory: TrajectoryView, *, max_chars: int | None = None
) -> tuple[str, int, int]:
    """The last <= 5 observations as RB's page states; returns text, count and cut chars."""
    limit = int(_fmt("max_observations"))
    last = environment_messages(trajectory)[-limit:]
    cap = int(_fmt("observation_max_chars"))
    max_chars = cap if max_chars is None else min(cap, max_chars)
    parts = []
    cut = 0
    for i, message in enumerate(last, start=1):
        text, lost = cut_head_tail(content_text(message.get("content")), max_chars)
        cut += lost
        parts.append(_fmt("observation").format(i=i, n=len(last), text=text))
    return _fmt("observation_joiner").join(parts), len(last), cut


@dataclass(frozen=True)
class JudgeInput:
    """The messages of one J call and how they were built."""

    messages: list[Message]
    actions: int
    observations: int
    #: Characters cut: ``observations`` by the 40,000 limit or the budget, ``actions``
    #: only by the budget; ``cap`` is the per-text cap the budget forced (None: none).
    truncation: Mapping[str, Any]
    #: Assistant turns left out of the action history for having no action.
    skipped: int = 0


def build_judge_input(
    trajectory: TrajectoryView, benchmark: str, *, max_chars: int | None = None
) -> JudgeInput:
    """J's ``[system, user]`` messages for one trajectory (pure).

    ``max_chars`` further caps every reasoning, action and observation text (head and
    tail); :meth:`Judge.judge` sets it only when the input would not fit the budget.
    """
    if benchmark not in protocol.BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    turns, skipped = action_turns(trajectory)
    actions, actions_cut = render_actions(turns, max_chars=max_chars)
    observations, count, observations_cut = render_observations(
        trajectory, max_chars=max_chars
    )
    text = user_template().format(
        intent=trajectory.instruction,
        actions=actions,
        observations=observations,
        # ALFWorld and WebShop have no answer: they change the state (WebShop's Buy Now
        # is an action, and its score line is the last page state J reads).
        response=_fmt("no_response"),
    )
    return JudgeInput(
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": text},
        ],
        actions=len(turns),
        observations=count,
        truncation={
            "cap": max_chars,
            "actions_chars_cut": actions_cut,
            "observations_chars_cut": observations_cut,
        },
        skipped=skipped,
    )


def input_budget(limits: protocol.ModelLimits) -> int:
    """The largest judge input that leaves the M3.1 planning reserve for the reply."""
    return limits.max_input_for_output(limits.planning_reserve)


def fit_judge_input(
    trajectory: TrajectoryView, benchmark: str, limits: protocol.ModelLimits
) -> tuple[JudgeInput, int]:
    """:func:`build_judge_input` within :func:`input_budget` (blocking CPU work).

    Returns the input and its o200k estimate. When the full input does not fit, every
    text is capped at a halving length (never below ``min_truncated_chars``) until it
    does; an input that still does not fit is returned as is (the gateway refuses it,
    a harness failure: a judge input of that size means something else is wrong).
    """
    budget = input_budget(limits)
    built = build_judge_input(trajectory, benchmark)
    estimate = tokens.estimate_chat_input({"messages": built.messages}).total
    if estimate <= budget:
        return built, estimate
    longest = max(
        [len(content_text(m.get("content"))) for m in trajectory.messages] or [0]
    )
    cap = max(longest // 2, int(_fmt("min_truncated_chars")))
    floor = int(_fmt("min_truncated_chars"))
    while True:
        built = build_judge_input(trajectory, benchmark, max_chars=cap)
        estimate = tokens.estimate_chat_input({"messages": built.messages}).total
        if estimate <= budget or cap <= floor:
            return built, estimate
        cap = max(cap // 2, floor)


# ---------------------------------------------------------------------------
# Parsing the verdict
# ---------------------------------------------------------------------------

_STATUS_LINE = re.compile(r"^[\s>*_#`-]*status[\s*_`]*:(?P<value>.*)$", re.IGNORECASE)
_THOUGHTS_LINE = re.compile(
    r"^[\s>*_#`-]*thoughts[\s*_`]*:(?P<value>.*)$", re.IGNORECASE
)
_VALUE_STRIP = " \t\"'*_`.!;,"


@dataclass(frozen=True)
class Judgement:
    """What J's reply says: ``status`` None when no status could be read."""

    status: Status | None
    thoughts: str

    @property
    def parsed(self) -> bool:
        return self.status is not None


def parse_judgement(text: str | None) -> Judgement:
    """Read ``Thoughts: ... / Status: success|failure`` (RB ``extract_content``).

    The first line labelled ``Status`` decides, as in RB. The label is matched without
    regard to case or markdown emphasis (``**Status:**``), and the value must be exactly
    ``success`` or ``failure`` once quotes, emphasis and a final full stop are stripped
    (RB also strips quotes). ``Thoughts`` is everything from its label up to that line.
    """
    lines = (text or "").splitlines()
    status: Status | None = None
    status_at: int | None = None
    for i, line in enumerate(lines):
        match = _STATUS_LINE.match(line)
        if match:
            value = match.group("value").strip(_VALUE_STRIP).lower()
            status = (
                "success"
                if value == "success"
                else "failure"
                if value == "failure"
                else None
            )
            status_at = i
            break
    end = status_at if status_at is not None else len(lines)
    thoughts: list[str] = []
    started = False
    for line in lines[:end]:
        match = _THOUGHTS_LINE.match(line)
        if not started and match:
            started = True
            thoughts.append(match.group("value").strip(" \t*_`"))
        elif started:
            thoughts.append(line)
    return Judgement(status, "\n".join(thoughts).strip().rstrip('"').strip())


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeVerdict:
    """One J decision. ``success`` is what RB and AWM use; the rest is for audit (H9)."""

    task_uid: str
    success: bool
    status: Status
    parsed: bool
    thoughts: str
    raw: str | None
    #: The MODEL failure of the call, if any (truncated, filtered or no reply).
    failure: Mapping[str, Any] | None
    gateway_call_id: str | None
    http_status: int | None
    usage: Mapping[str, Any] | None
    cost: float | None
    latency_s: float | None
    input_tokens_estimated: int
    input_sha256: str
    actions: int
    observations: int
    truncation: Mapping[str, Any]
    #: Assistant turns left out of the action history for having no action.
    skipped_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeCounters:
    """Running counts for a method's snapshot (H7.8: unparseable verdicts are counted)."""

    calls: int = 0
    success: int = 0
    failure: int = 0
    unparseable: int = 0
    model_failures: int = 0
    #: Inputs whose texts were cut further to fit the model's input budget.
    truncated_inputs: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeCounters:
        """Counters a learning checkpoint recorded (``to_dict``); unknown keys refuse."""
        return cls(**{k: int(v) for k, v in dict(data).items()})


class Judge:
    """J for one cell attempt: :meth:`judge` a trajectory, get a :class:`JudgeVerdict`.

    Calls go to the gateway's ``judge`` role of ``cell.run_id`` with ``cell.model``.
    ``transport`` / ``client`` are for tests (in-process gateway). ``record_path``
    (default ``cell.work_dir / "judge.jsonl"``; None disables it) receives every verdict.
    Safe to use from concurrent tasks; call :meth:`aclose` (or use ``async with``).
    """

    def __init__(
        self,
        cell: CellContext,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
        record_path: Path | None | Literal["default"] = "default",
    ) -> None:
        self.cell = cell
        self.benchmark = cell.benchmark
        self.limits = cell.model_limits
        self.counters = JudgeCounters()
        self.record_path = (
            cell.work_dir / RECORD_FILE if record_path == "default" else record_path
        )
        self._client = Actor(
            gateway_base_url=cell.gateway_base_url,
            run_id=cell.run_id,
            model=cell.model,
            token=cell.gateway_token,
            transport=transport,
            client=client,
            role=JUDGE_ROLE,
        )
        self._calls = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    def restore(self, counters: Mapping[str, Any]) -> None:
        """Resume J's counts from a learning checkpoint (``JudgeCounters.to_dict``): the
        counters, and the index of its next call (one per verdict counted, the same as an
        uninterrupted run's). The counters object is updated in place, so a method that
        shares it (``Judge.counters``) sees the restored counts."""
        restored = JudgeCounters.from_dict(counters)
        for name, value in restored.to_dict().items():
            setattr(self.counters, name, value)
        self._calls = restored.calls

    async def __aenter__(self) -> Judge:
        await self._client.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def judge(self, trajectory: TrajectoryView) -> JudgeVerdict:
        """Judge one learning trajectory; raise only for INFRA / HARNESS failures."""
        if not isinstance(trajectory, TrajectoryView):
            raise TypeError(
                "J judges a method-visible TrajectoryView only, never a private record, "
                f"got {type(trajectory).__name__}"
            )
        await self._client.start()
        built, estimate = await asyncio.to_thread(
            fit_judge_input, trajectory, self.benchmark, self.limits
        )
        digest = hashlib.sha256(
            json.dumps(built.messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        index = self._calls
        self._calls += 1
        reply: ActorReply | None = None
        failure: Failure | None = None
        call = None
        try:
            reply = await self._client.complete(
                built.messages,
                index=index,
                item=trajectory.task_uid,
                phase=JUDGE_PHASE,
            )
            call = reply.call
            failure = reply.failure
        except ActorCallError as exc:
            if exc.failure.kind is not FailureKind.MODEL:
                raise  # INFRA / HARNESS: the learning item is aborted (H8)
            failure, call = exc.failure, exc.call
        raw = reply.content if reply is not None else None
        judgement = parse_judgement(raw)
        status: Status = judgement.status or "failure"
        verdict = JudgeVerdict(
            task_uid=trajectory.task_uid,
            success=status == "success",
            status=status,
            parsed=judgement.parsed,
            thoughts=judgement.thoughts,
            raw=raw,
            failure=failure.to_dict() if failure is not None else None,
            gateway_call_id=call.gateway_call_id if call is not None else None,
            http_status=call.http_status if call is not None else None,
            usage=call.usage if call is not None else None,
            cost=call.cost if call is not None else None,
            latency_s=call.latency_s if call is not None else None,
            input_tokens_estimated=estimate,
            input_sha256=digest,
            actions=built.actions,
            observations=built.observations,
            truncation=built.truncation,
            skipped_turns=built.skipped,
        )
        self._count(verdict)
        if self.record_path is not None:
            await asyncio.to_thread(
                append_jsonl,
                self.record_path,
                {"event": "verdict", "at": _now(), **verdict.to_dict()},
            )
        return verdict

    def _count(self, verdict: JudgeVerdict) -> None:
        c = self.counters
        c.calls += 1
        if verdict.success:
            c.success += 1
        else:
            c.failure += 1
        if not verdict.parsed:
            c.unparseable += 1
        if verdict.failure is not None:
            c.model_failures += 1
        if verdict.truncation.get("cap") is not None:
            c.truncated_inputs += 1
