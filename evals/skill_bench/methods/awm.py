"""Agent Workflow Memory, online mode.

Source: arXiv 2409.07429 and github.com/zorazrw/agent-workflow-memory at 8c0ff8c (Apache-2.0),
following the WebArena online pipeline (``webarena/pipeline.py``). Every model-visible text
and limit lives in ``methods/prompts/awm/`` (fingerprinted); ``format.json`` records its
provenance, and the tests diff the prompts against the verbatim originals there.

**Learning** (``after_task``, one learning item at a time):

1. J (``methods/judge.py``) judges the trajectory.
2. A failure is discarded. A success joins the pool, and only then (the pool changed) the
   workflows are induced again from the pool and **overwrite** the previous ones
   (``induce_prompt.py:112-161``). AWM's code re-induces after every task
   (``pipeline.py:34-40``); an unchanged pool only repeats the call.
3. The induction prompt is AWM's (``induce_prompt.py:78-99``), one user message:
   ``instruction.md`` (``prompt/instruction.txt`` with the two substitutions), then
   ``output_format.md`` in place of the WebArena one-shot example, then
   ``## Concrete Examples``, one ``Query: <instruction>\\nActions:\\n<trajectory>`` per
   example, and ``## Summary Workflows``. A trajectory is AWM's ``<think>`` / ``<action>``
   steps plus the environment's observations (head and tail kept).
4. AWM deduplicates by ``intent_template_id``, annotation metadata we do not have; instead
   the most recent successes that fit the F9 input budget and ``max_request_bytes``
   (29 MB) are used and the oldest are dropped and counted. The request cap is a
   transport limit the providers behind OpenRouter enforce (``format.json``
   ``max_request_bytes_rule``); every induction records its binding limit and
   ``request_bytes``.
5. The reply is the workflow text, stored verbatim, without the WebArena action-syntax
   lines ``induce_prompt.py:153`` appends. A reply without text keeps the previous
   workflows; model failures are counted, never retried (the gateway retries
   infrastructure failures; INFRA / HARNESS failures raise and abort the item, H8).
6. A J or learner call the provider refuses (``infra/failures.upstream_refusal``: a
   payload refused with HTTP 413, or 400 naming its size or images, a no-endpoint 404)
   is counted, not raised (rule harness-stop-learning): a refused verdict skips the item
   (``judge_rejected``: nothing joins the pool); a payload-refused induction is sent
   once more with its budgets divided by ``refused_retry_divisor`` (``format.json``
   ``refused_retry_rule``: the oldest successes are dropped first,
   ``induction_retries``) and, refused again or refused no-endpoint (never retried),
   keeps the previous workflow (``induction_rejected``). The same refusal on
   ``REFUSAL_STREAK_ITEMS`` (3) consecutive calls of J or of the learner raises
   ``upstream_refusal_streak`` (HARNESS: a person). The three counters appear in the
   snapshot only once they are non-zero, so a memory without refusals snapshots as it
   did before them. INFRA still raises (the learning phase resumes from its checkpoint).

**Injection** (``before_task``): the whole workflow text, unchanged and without a header
(``agents/legacy/agent.py:116-117``), into the SkillOpt skill slot (Q21). One workflow text
per cell; ALFWorld is not split by task type.

**Frozen test phase:** ``finalize`` writes ``frozen.json`` (the snapshot) next to
``workflow.md`` in ``work_dir``; after it only ``before_task`` runs, read-only and
concurrently. A test-phase ``setup`` restores that frozen memory or raises (H8).

**Learning checkpoints** (``methods/base.py`` contract item 6): every success that joins
the pool is also written once to ``objects/pool/<task_uid>.json`` (its rendered
trajectory), write-once files a checkpoint hard-links. :meth:`checkpoint` returns the counters, J's counts and the induction
summaries; a learning ``setup`` with ``cell.resume`` rebuilds the pool from the
``pool.jsonl`` success rows and those objects and the workflow from ``workflow.md``.

Model calls: J through the gateway role ``judge``, the induction through ``learner``, both
with the cell's model; the gateway sets every parameter (no temperature, F9 ``max_tokens``).

This module also holds the trajectory rendering both S1 baselines share: ReasoningBank's
``induce_memory.format_trajectory`` is AWM's ``induce_prompt.format_trajectory`` copied, so
``methods/reasoningbank.py`` uses :func:`trajectory_steps` and :func:`render_steps` from here.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import (
    Actor,
    ActorCall,
    ActorCallError,
    request_bytes,
)
from evals.skill_bench.envs.base import Message
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import (
    REFUSAL_PAYLOAD,
    Failure,
    FailureKind,
    GatewayCallError,
    RefusalStreaks,
    upstream_refusal,
)
from evals.skill_bench.infra.processes import append_jsonl
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods.base import (
    OBJECTS_DIR,
    CellContext,
    CheckpointMismatch,
    FrozenMemoryMismatch,
    Injection,
    MemoryMethodBase,
    SignalLevel,
    TaskView,
    TrajectoryView,
    check_resumed_snapshot,
)

PROMPT_DIR = "methods/prompts/awm"
FORMAT_FILE = f"{PROMPT_DIR}/format.json"
LEARNER_ROLE = "learner"
#: The run phase of every learning-side call (``after_task`` runs in the learning phase only).
LEARNING_PHASE = "learning"
TEST_PHASE = "test"

WORKFLOW_FILE = "workflow.md"
POOL_FILE = "pool.jsonl"
INDUCTIONS_FILE = "inductions.jsonl"
FROZEN_FILE = "frozen.json"
STATE_FILES = (
    WORKFLOW_FILE,
    POOL_FILE,
    INDUCTIONS_FILE,
    FROZEN_FILE,
    J.RECORD_FILE,
    OBJECTS_DIR,
)
#: Write-once pool objects under ``work_dir`` (contract item 6).
POOL_OBJECTS = f"{OBJECTS_DIR}/pool"

T = TypeVar("T")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen prompt text
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def awm_format() -> Mapping[str, Any]:
    """``format.json``: model-visible pieces, limits, provenance and the substitutions."""
    return MappingProxyType(json.loads(protocol.read_protocol_text(FORMAT_FILE)))


def _fmt(key: str) -> Any:
    return awm_format()[key]


@lru_cache(maxsize=1)
def instruction_text() -> str:
    """``instruction.md``: ``prompt/instruction.txt`` with the two substitutions."""
    return protocol.read_protocol_text(
        f"{PROMPT_DIR}/{_fmt('instruction_file')}"
    ).rstrip("\n")


@lru_cache(maxsize=1)
def output_format_text() -> str:
    """``output_format.md``: the output-format description replacing the one-shot example."""
    return protocol.read_protocol_text(
        f"{PROMPT_DIR}/{_fmt('output_format_file')}"
    ).rstrip("\n")


# ---------------------------------------------------------------------------
# Trajectory rendering shared by the S1 baselines (AWM format_trajectory + observations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One model turn with an action, and the environment's raw output that followed it."""

    think: str
    actions: tuple[str, ...]
    observations: tuple[str, ...]


@dataclass(frozen=True)
class StepTrajectory:
    """A Q20 trajectory as the baselines' learners read it.

    ``initial`` is the environment's output before the first model turn (the first
    messages); ``steps`` are the model turns that carry an action; ``skipped`` counts the
    turns without one (no reply, or no ``<action>`` / ``<answer>``), which RB and AWM skip
    with their observations.
    """

    initial: tuple[str, ...]
    steps: tuple[Step, ...]
    skipped: int

    @property
    def longest_text(self) -> int:
        """Length of the longest text a budget cap can cut (for :func:`fit_texts`)."""
        lengths = [len(t) for t in self.initial]
        for step in self.steps:
            lengths += [len(step.think), *map(len, step.actions)]
            lengths += [len(o) for o in step.observations]
        return max(lengths or [0])


def trajectory_steps(view: TrajectoryView) -> StepTrajectory:
    """Split a method-visible trajectory into observations and think/action steps (pure).

    The think/action split of a turn is J's (:func:`~evals.skill_bench.methods.judge.episode_turns`,
    RB's ``_extract_think_from_output``): the text before ``<action>`` without ``<think>``
    tags and the tag's body, or the ``<answer>`` element and the rest (a final answer).
    Every ``user`` message is an observation of the turn before it.
    """
    split = iter(J.episode_turns(view))
    initial: list[str] = []
    turns: list[tuple[J.Turn, list[str]]] = []
    for message in view.messages:
        role = message.get("role")
        if role == "assistant":
            turns.append((next(split), []))
        elif role == "user":
            text = J.content_text(message.get("content"))
            (turns[-1][1] if turns else initial).append(text)
    steps: list[Step] = []
    skipped = 0
    for turn, observations in turns:
        actions = tuple(a for a in turn.actions if a)
        if not actions:
            skipped += 1  # RB / AWM skip a step without an action
            continue
        steps.append(Step(turn.think, actions, tuple(observations)))
    return StepTrajectory(tuple(initial), tuple(steps), skipped)


def render_steps(
    trajectory: StepTrajectory,
    fmt: Mapping[str, Any],
    *,
    max_chars: int | None = None,
) -> tuple[str, int]:
    """The trajectory text of AWM's ``format_trajectory`` with the observations added.

    ``fmt`` is a method's ``format.json`` (``step``, ``action_joiner``, ``observation``,
    ``observation_joiner``, ``step_joiner``, ``observation_max_chars``). Every
    observation keeps its head and tail within ``observation_max_chars``; ``max_chars``
    (set only to fit a budget) caps every text further. Returns the text and the number of
    characters cut (J's :func:`~evals.skill_bench.methods.judge.cut_head_tail`).
    """
    cap = int(fmt["observation_max_chars"])
    observation_cap = cap if max_chars is None else min(cap, max_chars)
    cut = 0

    def keep(text: str, limit: int | None) -> str:
        nonlocal cut
        kept, lost = J.cut_head_tail(text, limit)
        cut += lost
        return kept

    def observations(texts: Sequence[str]) -> str:
        return fmt["observation_joiner"].join(
            fmt["observation"].format(text=keep(t, observation_cap)) for t in texts
        )

    blocks: list[str] = []
    if trajectory.initial:
        blocks.append(observations(trajectory.initial))
    for step in trajectory.steps:
        block = fmt["step"].format(
            think=keep(step.think, max_chars),
            action=fmt["action_joiner"].join(keep(a, max_chars) for a in step.actions),
        )
        if step.observations:
            block += fmt["observation_joiner"] + observations(step.observations)
        blocks.append(block)
    return fmt["step_joiner"].join(blocks), cut


def estimate_messages(messages: Sequence[Message]) -> int:
    """o200k input estimate of a chat request, as the gateway counts it (CPU work)."""
    return tokens.estimate_chat_input({"messages": list(messages)}).total


def fit_texts(
    build: Callable[[int | None], T],
    estimate: Callable[[T], int],
    budget: int,
    longest: int,
    floor: int,
) -> tuple[T, int, int | None]:
    """``build(None)`` if it fits ``budget``; else ``build(cap)`` with a halving cap.

    J's rule (:func:`~evals.skill_bench.methods.judge.fit_judge_input`): the cap starts at
    half the longest text and never goes below ``floor``. Returns what was built, its
    estimate and the cap (None: nothing was cut). A result still over the budget at the
    floor is returned as is; the gateway refuses it (HARNESS).
    """
    built = build(None)
    value = estimate(built)
    if value <= budget:
        return built, value, None
    cap = max(longest // 2, floor)
    while True:
        built = build(cap)
        value = estimate(built)
        if value <= budget or cap <= floor:
            return built, value, cap
        cap = max(cap // 2, floor)


# ---------------------------------------------------------------------------
# Learner calls and small helpers shared by the S1 baselines
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerResult:
    """One learner call: the reply text (None without a reply), its MODEL failure, if any,
    and ``refused``: the failure of a call the provider refused
    (``infra/failures.upstream_refusal``)."""

    content: str | None
    failure: Failure | None
    call: ActorCall | None
    refused: Failure | None = None

    def record(self) -> dict[str, Any]:
        call = self.call
        return {
            "gateway_call_id": call.gateway_call_id if call else None,
            "http_status": call.http_status if call else None,
            "finish_reason": call.finish_reason if call else None,
            "usage": dict(call.usage) if call and call.usage else None,
            "cost": call.cost if call else None,
            "latency_s": call.latency_s if call else None,
            "failure": self.failure.to_dict() if self.failure else None,
            # Only on a refused call: every other row stays as it was.
            **({"refused": self.refused.to_dict()} if self.refused else {}),
        }


async def call_learner(
    client: Actor, messages: Sequence[Message], *, index: int, item: str
) -> LearnerResult:
    """One learner call; MODEL failures are returned (counted by the caller), and so is a
    request the provider refused (``refused``, recorded instead of stopping the learning
    phase); every other INFRA and HARNESS failure raises
    :class:`~evals.skill_bench.agent.actor.ActorCallError` (H8)."""
    try:
        reply = await client.complete(
            messages, index=index, item=item, phase=LEARNING_PHASE
        )
    except ActorCallError as exc:
        if exc.failure.kind is FailureKind.MODEL:
            return LearnerResult(None, exc.failure, exc.call)
        if upstream_refusal(exc.failure) is not None:
            return LearnerResult(None, None, exc.call, refused=exc.failure)
        raise
    return LearnerResult(reply.content, reply.failure, reply.call)


async def judge_or_refused(
    judge: J.Judge, trajectory: TrajectoryView
) -> tuple[J.JudgeVerdict | None, ActorCallError | None]:
    """J's verdict, or the error of a J call the provider refused
    (``infra/failures.upstream_refusal``): the caller skips the item and counts it.
    Every other failure raises as :meth:`J.Judge.judge` raises it."""
    try:
        return await judge.judge(trajectory), None
    except ActorCallError as exc:
        if upstream_refusal(exc.failure) is None:
            raise
        return None, exc


def judge_refused_record(task_uid: str, refused: ActorCallError) -> dict[str, Any]:
    """The ``judge_refused`` event a method appends to its learning log (AWM's
    ``inductions.jsonl``, RB's ``extractions.jsonl``)."""
    call = refused.call
    return {
        "event": "judge_refused",
        "at": now(),
        "task_uid": task_uid,
        "failure": refused.failure.to_dict(),
        "gateway_call_id": call.gateway_call_id if call else None,
        "http_status": call.http_status if call else None,
        "provider": call.provider if call else None,
    }


def learner_client(
    cell: CellContext, transport: httpx.AsyncBaseTransport | None
) -> Actor:
    """The recorded chat client for the gateway role ``learner`` with the cell's model."""
    return Actor(
        gateway_base_url=cell.gateway_base_url,
        run_id=cell.run_id,
        model=cell.model,
        token=cell.gateway_token,
        transport=transport,
        role=LEARNER_ROLE,
    )


class Busy:
    """Learning in flight, for ``wait_until_idle`` (the baselines learn inside ``after_task``)."""

    def __init__(self) -> None:
        self._count = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def __aenter__(self) -> None:
        self._count += 1
        self._idle.clear()

    async def __aexit__(self, *exc: object) -> None:
        self._count -= 1
        if self._count == 0:
            self._idle.set()

    async def wait(self, timeout_s: float) -> None:
        """Return once idle; raise :class:`TimeoutError` after ``timeout_s`` (INFRA, Q18)."""
        await asyncio.wait_for(self._idle.wait(), timeout_s)


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` durably (temporary file, fsync, rename, fsync dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes; a missing file counts as empty."""
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def refuse_existing_state(work_dir: Path, names: Sequence[str], method: str) -> None:
    """A learning attempt starts from empty memory (H8): no state of an earlier attempt."""
    found = [n for n in names if (work_dir / n).exists()]
    if found:
        raise RuntimeError(
            f"{method}: {work_dir} already holds memory files {found}; a learning attempt "
            "must start from an empty work_dir (H8)"
        )


def read_frozen(work_dir: Path, method: str) -> dict[str, Any]:
    """``frozen.json`` of a finished learning phase, or raise (never resume empty, H8)."""
    path = work_dir / FROZEN_FILE
    if not path.is_file():
        raise FrozenMemoryMismatch(
            f"{method}: no frozen memory in {work_dir} ({FROZEN_FILE} missing); a "
            "test-phase setup resumes only the memory finalize() froze (H8)"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise FrozenMemoryMismatch(f"{method}: {path} is not valid JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("snapshot"), dict):
        raise FrozenMemoryMismatch(f"{method}: {path} is not a frozen snapshot")
    return data


# ---------------------------------------------------------------------------
# Write-once pool objects (contract item 6; blocking file I/O)
# ---------------------------------------------------------------------------


def write_once(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` (temporary file, fsync, rename) unless it exists with
    exactly these bytes; an existing file with other bytes raises: an object under
    ``objects/`` is never modified (a checkpoint shares its inode)."""
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(
                f"{path} already exists with other content: objects are write-once"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def step_trajectory_record(trajectory: StepTrajectory) -> dict[str, Any]:
    """A pool entry's rendered trajectory as JSON."""
    return {
        "initial": list(trajectory.initial),
        "steps": [
            {
                "think": step.think,
                "actions": list(step.actions),
                "observations": list(step.observations),
            }
            for step in trajectory.steps
        ],
        "skipped": trajectory.skipped,
    }


def step_trajectory_from_record(record: Mapping[str, Any]) -> StepTrajectory:
    """Inverse of :func:`step_trajectory_record`."""
    return StepTrajectory(
        tuple(str(t) for t in record["initial"]),
        tuple(
            Step(
                str(step["think"]),
                tuple(str(a) for a in step["actions"]),
                tuple(str(o) for o in step["observations"]),
            )
            for step in record["steps"]
        ),
        int(record["skipped"]),
    )


def pool_object_path(work_dir: Path, task_uid: str) -> Path:
    return work_dir / POOL_OBJECTS / f"{task_uid}.json"


def write_pool_object(work_dir: Path, entry: PoolEntry) -> None:
    """One success of the pool, written once (``objects/pool/<task_uid>.json``)."""
    record = {
        "task_uid": entry.task_uid,
        "instruction": entry.instruction,
        "verdict_call_id": entry.verdict_call_id,
        "trajectory": step_trajectory_record(entry.trajectory),
    }
    data = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
    write_once(pool_object_path(work_dir, entry.task_uid), data)


def read_pool(work_dir: Path) -> list[PoolEntry]:
    """The pool of a learning checkpoint: the ``pool.jsonl`` success rows, in order, each
    rebuilt from its object (blocking). A missing object raises ``CheckpointMismatch``."""
    path = work_dir / POOL_FILE
    rows = (
        # "\n" only: a pool row may hold U+2028 (written with ensure_ascii=False), which
        # str.splitlines() would split into two unparsable halves.
        [
            json.loads(line)
            for line in path.read_text("utf-8").split("\n")
            if line.strip()
        ]
        if path.exists()
        else []
    )
    pool = []
    for row in rows:
        if row.get("event") != "success":
            continue
        uid = str(row["task_uid"])
        target = pool_object_path(work_dir, uid)
        if not target.is_file():
            raise CheckpointMismatch(f"awm: the pool object of {uid} is missing")
        record = json.loads(target.read_text("utf-8"))
        if record.get("task_uid") != uid:
            raise CheckpointMismatch(f"awm: {target} holds another task")
        pool.append(
            PoolEntry(
                uid,
                str(record["instruction"]),
                step_trajectory_from_record(record["trajectory"]),
                record.get("verdict_call_id"),
            )
        )
    return pool


# ---------------------------------------------------------------------------
# The induction prompt and the example selection (pure / CPU)
# ---------------------------------------------------------------------------


@dataclass
class PoolEntry:
    """One success in the pool: its query (the instruction) and its trajectory."""

    task_uid: str
    instruction: str
    trajectory: StepTrajectory
    verdict_call_id: str | None = None
    #: Marginal (tokens, bytes) of the example in the prompt. Cached in memory only (the
    #: entry never changes); no pool object, checkpoint or snapshot stores it, so a pool
    #: read back from ``objects/pool/`` always measures it afresh.
    cost: tuple[int, int] | None = field(default=None, repr=False)


def induction_messages(
    examples: Sequence[tuple[str, StepTrajectory]], *, max_chars: int | None = None
) -> tuple[list[Message], int]:
    """AWM's induction request for ``examples`` (query, trajectory), oldest first.

    ``'\\n\\n'.join([INSTRUCTION, ONE_SHOT, format_examples(examples)])`` of
    ``induce_prompt.py:78-99`` with the output-format description in place of ONE_SHOT, as
    one user message. Returns the messages and the characters cut.
    """
    fmt = awm_format()
    joiner = fmt["part_joiner"]
    text = joiner.join(
        [instruction_text(), output_format_text(), fmt["examples_header"]]
    )
    cut = 0
    for query, trajectory in examples:
        rendered, lost = render_steps(trajectory, fmt, max_chars=max_chars)
        cut += lost
        text += joiner + fmt["example"].format(query=query, trajectory=rendered)
    text += joiner + fmt["workflows_header"]
    return [{"role": "user", "content": text}], cut


@dataclass(frozen=True)
class Selection:
    """The examples one induction uses and why the others were left out."""

    messages: list[Message]
    used: tuple[str, ...]
    dropped: int
    #: ``tokens`` or ``bytes``: the limit that kept the oldest successes out (None: all
    #: fit), the first one exceeded in that order (:data:`LIMITS`). When even the newest
    #: success alone did not fit, the limit it exceeded (its texts were cut).
    binding: str | None
    #: The per-text cap when even the newest success alone did not fit (None: no cut).
    cap: int | None
    chars_cut: int
    input_tokens_estimated: int
    request_bytes: int


#: The request limits of one induction, in the order :func:`select_examples` checks them
#: and names the binding one: the F9 input budget, ``max_request_bytes``.
LIMITS = ("tokens", "bytes")


def _measure(messages: Sequence[Message], model: str) -> tuple[int, int]:
    """(o200k tokens, request bytes as the gateway receives them)."""
    size = len(request_bytes({"model": model, "messages": list(messages)}))
    return estimate_messages(messages), size


def _exceeded(measured: tuple[int, int], budgets: tuple[int, int]) -> str | None:
    """The first of :data:`LIMITS` that ``measured`` exceeds (None: it fits both)."""
    for name, value, budget in zip(LIMITS, measured, budgets, strict=True):
        if value > budget:
            return name
    return None


def select_examples(
    pool: Sequence[PoolEntry],
    *,
    model: str,
    budget_tokens: int,
    budget_bytes: int,
) -> Selection:
    """The most recent successes whose induction request fits both budgets (CPU work).

    Each entry's marginal cost (tokens, bytes) is measured once (cached on the entry).
    The longest suffix of the pool whose summed cost fits is then built and measured
    exactly; if it does not fit, one more of the oldest is dropped. If not even the
    newest fits alone, its texts are cut to ``budget_tokens`` (:func:`fit_texts`). The
    binding limit is the first one exceeded in the order of :data:`LIMITS`.
    """
    if not pool:
        raise ValueError("an induction needs at least one success")
    budgets = (budget_tokens, budget_bytes)
    base_tokens, base_bytes = _measure(induction_messages([])[0], model)
    for entry in pool:
        if entry.cost is None:
            one, _ = induction_messages([(entry.instruction, entry.trajectory)])
            t, b = _measure(one, model)
            entry.cost = (t - base_tokens, b - base_bytes)
    n = 0
    total_tokens, total_bytes = base_tokens, base_bytes
    binding: str | None = None
    for entry in reversed(pool):
        assert entry.cost is not None
        t, b = entry.cost
        if total_tokens + t > budget_tokens:
            binding = "tokens"
            break
        if total_bytes + b > budget_bytes:
            binding = "bytes"
            break
        total_tokens, total_bytes = total_tokens + t, total_bytes + b
        n += 1
    while n > 0:
        chosen = pool[len(pool) - n :]
        messages, cut = induction_messages(
            [(e.instruction, e.trajectory) for e in chosen]
        )
        estimate, size = _measure(messages, model)
        over = _exceeded((estimate, size), budgets)
        if over is None:
            return Selection(
                messages=messages,
                used=tuple(e.task_uid for e in chosen),
                dropped=len(pool) - n,
                binding=binding if n < len(pool) else None,
                cap=None,
                chars_cut=cut,
                input_tokens_estimated=estimate,
                request_bytes=size,
            )
        binding = over
        n -= 1
    newest = pool[-1]
    trajectory = newest.trajectory
    example = [(newest.instruction, trajectory)]
    (messages, cut), estimate, cap = fit_texts(
        lambda c: induction_messages(example, max_chars=c),
        lambda built: estimate_messages(built[0]),
        budget_tokens,
        trajectory.longest_text,
        int(_fmt("min_truncated_chars")),
    )
    _, size = _measure(messages, model)
    return Selection(
        messages=messages,
        used=(newest.task_uid,),
        dropped=len(pool) - 1,
        binding=binding or "tokens",
        cap=cap,
        chars_cut=cut,
        input_tokens_estimated=estimate,
        request_bytes=size,
    )


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------


@dataclass
class AwmCounters:
    """Learning-phase counts (snapshot; frozen with the memory)."""

    tasks_learned: int = 0
    judged_success: int = 0
    judged_failure: int = 0
    inductions: int = 0
    workflows_written: int = 0
    #: Induction replies that were truncated or filtered, or that never came (MODEL).
    induction_model_failures: int = 0
    #: Inductions without a reply or with an empty one: the previous workflow is kept.
    induction_no_text: int = 0
    #: Successes left out of inductions by the budget, summed over inductions.
    examples_dropped: int = 0
    max_examples_dropped: int = 0
    inductions_trimmed: int = 0
    inductions_text_cut: int = 0
    #: Turns without an action, skipped as RB / AWM skip them.
    steps_skipped: int = 0
    #: Refusals (:data:`REFUSAL_COUNTERS`): inductions whose final
    #: request the provider refused (the previous workflow kept), halved retries sent after
    #: a payload refusal, and items whose J call was refused (not learned from).
    induction_rejected: int = 0
    induction_retries: int = 0
    judge_rejected: int = 0

    #: Counters a restored checkpoint did not hold (:meth:`restore`): left out of
    #: :meth:`to_dict` while they are 0, like :data:`REFUSAL_COUNTERS`.
    _absent: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def restore(cls, data: Mapping[str, Any]) -> AwmCounters:
        """The counters a learning checkpoint recorded (``to_dict``); unknown keys refuse.
        A counter the checkpoint did not hold (it was written by code from before the
        counter existed) stays out of
        :meth:`to_dict` while it is 0, so the rebuilt memory hashes as the checkpoint
        recorded and the resume goes on under this code."""
        data = dict(data)
        counters = cls(**data)
        counters._absent = frozenset(f.name for f in fields(cls)) - set(data)
        return counters

    def to_dict(self) -> dict[str, int]:
        """Every counter, except a :data:`REFUSAL_COUNTERS` one while it is 0 (a memory
        without refusals snapshots byte for byte as before they were counted, so a
        checkpoint or frozen memory of the code before them still matches its hash) and
        a counter a restored checkpoint did not hold while it is 0 (:meth:`restore`)."""
        data = asdict(self)
        for key in (*REFUSAL_COUNTERS, *sorted(self._absent)):
            if key in data and not data[key]:
                del data[key]
        return data


#: The refusal counters of :class:`AwmCounters`, written only when non-zero.
REFUSAL_COUNTERS: tuple[str, ...] = (
    "induction_rejected",
    "induction_retries",
    "judge_rejected",
)


class AgentWorkflowMemory(MemoryMethodBase):
    """AWM online: J-judged successes are induced into one workflow text."""

    name = "awm"
    signal_level: SignalLevel = "S1"

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__()
        self._transport = transport
        self._judge: J.Judge | None = None
        self._judge_counters: J.JudgeCounters | None = None
        self._learner: Actor | None = None
        self._pool: list[PoolEntry] = []
        self._workflow = ""
        self._inductions: list[dict[str, Any]] = []
        self._counters = AwmCounters()
        self._frozen: dict[str, Any] | None = None
        self._busy = Busy()
        #: Consecutive refusals of J and of the learner (checkpointed).
        self._streaks = RefusalStreaks()
        #: The refusal-streak stop after_task found, raised by the next hook: never
        #: checkpointed, so a learning phase resumed from the item's ``done``
        #: checkpoint (a person's --force-resume) goes on.
        self._held_stop: GatewayCallError | None = None

    # --- setup and state ------------------------------------------------------------

    def _hold_stop(self, stop: Failure, cause: BaseException | None = None) -> None:
        """Keep a refusal-streak stop for :meth:`_raise_held_stop`. Raised
        inside after_task, before ``run.py``'s ``done`` checkpoint of the item, the stop
        would be replayed by a --force-resume from ``played`` (the same refused call, the
        same streak): a person could never go on. The item's records are written and
        after_task returns; the next item's before_task, or finalize after the last
        item, raises it. A crash in between loses it: the resumed phase counts on from
        the checkpointed streak, and the next refusal in a row stops it again."""
        if self._held_stop is None:
            held = GatewayCallError(stop)
            held.__cause__ = cause
            self._held_stop = held

    def _raise_held_stop(self) -> None:
        if self._held_stop is not None:
            raise self._held_stop

    def _cell(self) -> CellContext:
        if self.cell is None:
            raise RuntimeError("awm: setup() was not called")
        return self.cell

    @property
    def workflow(self) -> str:
        return self._workflow

    async def setup(self, cell: CellContext) -> None:
        await super().setup(cell)
        if cell.learning:
            if cell.resume is None:
                refuse_existing_state(cell.work_dir, STATE_FILES, self.name)
            cell.work_dir.mkdir(parents=True, exist_ok=True)
            self._judge = J.Judge(cell, transport=self._transport)
            self._judge_counters = self._judge.counters
            self._learner = learner_client(cell, self._transport)
            if cell.resume is not None:
                await asyncio.to_thread(self._restore_learning, cell)
                check_resumed_snapshot(self.name, await self.snapshot(), cell.resume)
            return
        frozen = await asyncio.to_thread(read_frozen, cell.work_dir, self.name)
        snapshot = frozen["snapshot"]
        path = cell.work_dir / WORKFLOW_FILE
        workflow = path.read_text(encoding="utf-8") if path.exists() else ""
        if sha256_text(workflow) != snapshot.get("workflow_sha256") or workflow != (
            snapshot.get("workflow")
        ):
            raise FrozenMemoryMismatch(
                f"awm: {path} does not match the frozen snapshot in {FROZEN_FILE}"
            )
        self._workflow = workflow
        self._frozen = snapshot

    async def checkpoint(self) -> dict[str, Any]:
        """Contract item 6: the counters, J's counts, the induction summaries and the
        refusal streaks (the pool and the workflow are in ``work_dir``)."""
        return {
            "counters": self._counters.to_dict(),
            "judge": self._judge_counters.to_dict() if self._judge_counters else {},
            "inductions": copy.deepcopy(self._inductions),
            "refusal_streaks": self._streaks.to_dict(),
        }

    def _restore_learning(self, cell: CellContext) -> None:
        """Rebuild the memory of a learning checkpoint (blocking I/O): the pool from the
        ``pool.jsonl`` success rows and their objects, the workflow from ``workflow.md``,
        the counters, J's counts and the inductions from the checkpoint's state."""
        resume = cell.resume
        assert resume is not None and self._judge is not None
        if (cell.work_dir / FROZEN_FILE).exists():
            raise CheckpointMismatch(
                f"awm: {cell.work_dir} holds {FROZEN_FILE}: a learning checkpoint is "
                "taken before the memory is frozen"
            )
        try:
            self._pool = read_pool(cell.work_dir)
            path = cell.work_dir / WORKFLOW_FILE
            self._workflow = path.read_text(encoding="utf-8") if path.exists() else ""
            self._counters = AwmCounters.restore(resume.state["counters"])
            self._judge.restore(resume.state["judge"])
            self._inductions = [dict(i) for i in resume.state["inductions"]]
            # Absent in a checkpoint of the code before refusals were counted.
            self._streaks = RefusalStreaks(resume.state.get("refusal_streaks"))
        except CheckpointMismatch:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise CheckpointMismatch(
                f"awm: checkpoint {resume.checkpoint_id} cannot be restored "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    # --- hooks -------------------------------------------------------------------------

    async def before_task(self, task: TaskView) -> Injection:
        self._cell()
        self._raise_held_stop()
        workflow = self._workflow
        if not workflow.strip():
            return Injection()
        return Injection(workflow, (f"workflow:{sha256_text(workflow)[:16]}",))

    async def after_task(self, trajectory: TrajectoryView) -> None:
        cell = self._cell()
        if self._frozen is not None or not cell.learning:
            raise RuntimeError(
                "awm: after_task in the frozen test phase (memory is read-only)"
            )
        assert self._judge is not None
        async with self._busy:
            verdict, refused = await judge_or_refused(self._judge, trajectory)
            stop = self._streaks.observe(
                J.JUDGE_ROLE, refused.failure if refused is not None else None
            )
            if refused is not None:
                # Not learned from: nothing joins the pool.
                self._counters.judge_rejected += 1
                await asyncio.to_thread(
                    append_jsonl,
                    cell.work_dir / INDUCTIONS_FILE,
                    judge_refused_record(trajectory.task_uid, refused),
                )
                if stop is not None:
                    self._hold_stop(stop, refused)
                return
            assert verdict is not None
            steps = trajectory_steps(trajectory)
            counters = self._counters
            counters.tasks_learned += 1
            counters.steps_skipped += steps.skipped
            if not verdict.success:
                counters.judged_failure += 1
                return  # failures are discarded
            counters.judged_success += 1
            entry = PoolEntry(
                trajectory.task_uid,
                trajectory.instruction,
                steps,
                verdict.gateway_call_id,
            )
            # The entry's trajectory, written once before its pool.jsonl row: a learning
            # checkpoint rebuilds the pool from both (contract item 6).
            await asyncio.to_thread(write_pool_object, cell.work_dir, entry)
            self._pool.append(entry)
            await asyncio.to_thread(
                append_jsonl,
                cell.work_dir / POOL_FILE,
                {
                    "event": "success",
                    "at": now(),
                    "task_uid": entry.task_uid,
                    "instruction": entry.instruction,
                    "verdict_call_id": entry.verdict_call_id,
                    "steps": len(steps.steps),
                    "skipped_steps": steps.skipped,
                },
            )
            await self._induce(trigger=trajectory.task_uid)

    async def _induce(self, *, trigger: str) -> None:
        """Re-induce the workflows from the pool and overwrite them (the pool just changed).

        The examples fit the F9 input budget and ``max_request_bytes``
        (:func:`select_examples`); the summary (``inductions.jsonl``, the checkpoint state
        and the snapshot) records the binding limit and ``request_bytes``. An induction
        summary of a checkpoint written before those existed is restored as recorded,
        never rewritten.

        A request the provider refuses as a payload is sent once more with both budgets
        divided by ``refused_retry_divisor`` (``format.json``
        ``refused_retry_rule``; the refused request's own measures), which drops the
        oldest successes first; the summary then describes the retry and records the
        refused request under ``retry``. Refused again, or refused no-endpoint (never
        retried), the previous workflow is kept (``rejected``); the same refusal on three
        consecutive inductions raises ``upstream_refusal_streak``.
        """
        cell = self._cell()
        assert self._learner is not None
        selection = await asyncio.to_thread(
            select_examples,
            list(self._pool),
            model=cell.model,
            budget_tokens=J.input_budget(cell.model_limits),
            budget_bytes=int(_fmt("max_request_bytes")),
        )
        index = len(self._inductions)
        result = await call_learner(
            self._learner, selection.messages, index=index, item=trigger
        )
        retry: dict[str, Any] | None = None
        first = result.refused
        if first is not None and upstream_refusal(first) == REFUSAL_PAYLOAD:
            divisor = int(_fmt("refused_retry_divisor"))
            refused_call = result.call
            retry = {
                "refusal": first.to_dict(),
                "used": list(selection.used),
                "dropped": selection.dropped,
                "request_bytes": selection.request_bytes,
                "input_tokens_estimated": selection.input_tokens_estimated,
                "gateway_call_id": refused_call.gateway_call_id
                if refused_call
                else None,
                "provider": refused_call.provider if refused_call else None,
            }
            selection = await asyncio.to_thread(
                select_examples,
                list(self._pool),
                model=cell.model,
                budget_tokens=max(1, selection.input_tokens_estimated // divisor),
                budget_bytes=max(1, selection.request_bytes // divisor),
            )
            # The same index: the retry is the same induction (it only names the call).
            result = await call_learner(
                self._learner, selection.messages, index=index, item=trigger
            )
        rejected = result.refused
        text = result.content
        wrote = rejected is None and bool(text and text.strip())
        if wrote:
            assert text is not None
            self._workflow = text
            await asyncio.to_thread(
                write_atomic, cell.work_dir / WORKFLOW_FILE, text.encode("utf-8")
            )
        summary = {
            "index": index,
            "trigger": trigger,
            "pool_size": len(self._pool),
            "used": list(selection.used),
            "dropped": selection.dropped,
            "binding": selection.binding,
            "cap": selection.cap,
            "chars_cut": selection.chars_cut,
            "input_tokens_estimated": selection.input_tokens_estimated,
            "request_bytes": selection.request_bytes,
            "wrote": wrote,
            "failure_code": result.failure.code if result.failure else None,
            "workflow_sha256": sha256_text(self._workflow),
        }
        # Only on refused inductions: every other summary stays as it was.
        if retry is not None:
            summary["retry"] = retry
        if rejected is not None:
            summary["rejected"] = rejected.to_dict()
            summary["provider"] = result.call.provider if result.call else None
        self._inductions.append(summary)
        await asyncio.to_thread(
            append_jsonl,
            cell.work_dir / INDUCTIONS_FILE,
            {
                "event": "induction",
                "at": now(),
                **summary,
                **result.record(),
                "reply": text,
            },
        )
        c = self._counters
        c.inductions += 1
        c.workflows_written += int(wrote)
        c.induction_model_failures += int(result.failure is not None)
        c.induction_no_text += int(not wrote and rejected is None)
        c.induction_retries += int(retry is not None)
        c.induction_rejected += int(rejected is not None)
        c.examples_dropped += selection.dropped
        c.max_examples_dropped = max(c.max_examples_dropped, selection.dropped)
        c.inductions_trimmed += int(selection.dropped > 0)
        c.inductions_text_cut += int(selection.cap is not None)
        stop = self._streaks.observe(LEARNER_ROLE, rejected)
        if stop is not None:
            self._hold_stop(stop)

    async def wait_until_idle(self, timeout_s: float) -> None:
        await self._busy.wait(timeout_s)

    async def finalize(self) -> None:
        cell = self._cell()
        if self._frozen is not None:
            return
        self._raise_held_stop()
        snapshot = self._build_snapshot(frozen=True)
        data = json.dumps(
            {"at": now(), "snapshot": snapshot}, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        await asyncio.to_thread(write_atomic, cell.work_dir / FROZEN_FILE, data)
        self._frozen = snapshot

    def _build_snapshot(self, *, frozen: bool) -> dict[str, Any]:
        return {
            "method": self.name,
            "signal_level": self.signal_level,
            "frozen": frozen,
            "workflow": self._workflow,
            "workflow_sha256": sha256_text(self._workflow),
            "pool": [
                {"task_uid": e.task_uid, "instruction": e.instruction}
                for e in self._pool
            ],
            "inductions": copy.deepcopy(self._inductions),
            "counters": self._counters.to_dict(),
            "judge": self._judge_counters.to_dict() if self._judge_counters else {},
        }

    async def snapshot(self) -> dict[str, Any]:
        if self._frozen is not None:
            return copy.deepcopy(self._frozen)
        return self._build_snapshot(frozen=False)

    async def close(self) -> None:
        for client in (self._judge, self._learner):
            if client is not None:
                await client.aclose()
        self._judge = self._learner = None
