"""ReasoningBank baseline.

Source: arXiv 2509.25140 and github.com/google-research/reasoning-bank at ed80611
(Apache-2.0); where the paper and the code differ, the paper wins. Every model-visible text
and limit lives in ``methods/prompts/reasoningbank/`` (fingerprinted); ``format.json``
records its provenance, and the tests diff the prompts against the verbatim originals there.

**Memory:** an append-only bank of entries ``{query, status, memory_items}``, one per
learned task, each item ``{title, description, content}``, at most 3 per task; never merged
or deleted. Each entry's vector is the embedding of its unwrapped query (the Q28
``instruction``) by gemini-embedding-001 through the gateway role ``embedding``.

**Learning** (``after_task``, one learning item at a time):

1. J (``methods/judge.py``) judges the trajectory: success or failure.
2. The paper's Figure 9 success or failure instruction (``success_system.md`` /
   ``failure_system.md``: the two substitutions only) extracts memory items from
   ``**Query:** <instruction>\\n\\n**Trajectory:**\\n<trajectory>`` (``induce_memory.py:143``);
   the trajectory is RB's ``<think>`` / ``<action>`` steps plus the environment's
   observations (head and tail kept). J's reasoning is **not** appended
   (``induce_memory.py:145-162`` does; the paper does not).
3. The reply is parsed into items; the first 3 complete ones (title and content) are kept.
   A task with at least one item becomes a bank entry, embedded unwrapped.

**Retrieval** (``before_task``): the instruction wrapped in RB's instruction
(``memory_management.py:204``, 'web navigation' -> 'task') is embedded; the cosine top
k = 1 prior task wins and all of its items are injected: the paper's sentence, then each
item's title and content (paper A.2), into the SkillOpt skill slot (Q21). An empty bank
injects nothing and makes no call.

**Frozen test phase:** ``finalize`` writes ``frozen.json`` (the snapshot and the hashes of
``bank.jsonl`` and ``embeddings.jsonl``); after it only ``before_task`` runs, read-only and
concurrently: RB's ``screening()`` appends every query vector to its cache
(``memory_management.py:189-197``), which is not done here. A test-phase ``setup`` restores
the frozen bank or raises (H8). No MaTTS.

**Learning checkpoints** (``methods/base.py`` contract item 6): :meth:`checkpoint`
returns the counters and J's counts; a learning ``setup`` with ``cell.resume`` rebuilds
the bank's entries and vectors from ``bank.jsonl`` / ``embeddings.jsonl`` in the restored
``work_dir`` and refuses (``CheckpointMismatch``) a bank whose snapshot differs from the
one the checkpoint recorded.

Model failures (truncated, filtered or missing replies, unparseable output) are counted,
never retried; INFRA and HARNESS failures of J, the learner or the embedding raise and
abort the item (H8), except a J or extraction call the provider refuses
(``infra/failures.upstream_refusal``: a payload refused with HTTP 413, or 400 naming
its size or images, a no-endpoint 404; rule harness-stop-learning): it is counted and
the item stores nothing (``judge_rejected``: not learned from; ``extraction_rejected``:
no entry), recorded in ``extractions.jsonl``, and the same refusal on three consecutive
calls of J or of the learner raises ``upstream_refusal_streak`` (HARNESS: a person). Both
counters appear in the snapshot only once they are non-zero. Embedding refusals still
raise. Calls go through the gateway with the cell's model; the gateway sets every
parameter (no temperature, F9 ``max_tokens``).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import Actor
from evals.skill_bench.envs.base import Message
from evals.skill_bench.infra.failures import (
    Failure,
    FailureKind,
    GatewayCallError,
    RefusalStreaks,
    classify_exception,
    classify_gateway_error,
)
from evals.skill_bench.infra.gateway import EMBEDDINGS, gateway_url
from evals.skill_bench.infra.processes import append_jsonl
from evals.skill_bench.methods import judge as J
from evals.skill_bench.methods.awm import (
    FROZEN_FILE,
    LEARNER_ROLE,
    LEARNING_PHASE,
    TEST_PHASE,
    Busy,
    StepTrajectory,
    call_learner,
    estimate_messages,
    file_sha256,
    fit_texts,
    judge_or_refused,
    judge_refused_record,
    learner_client,
    now,
    read_frozen,
    refuse_existing_state,
    render_steps,
    trajectory_steps,
    write_atomic,
)
from evals.skill_bench.methods.base import (
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

PROMPT_DIR = "methods/prompts/reasoningbank"
FORMAT_FILE = f"{PROMPT_DIR}/format.json"
EMBEDDING_ROLE = "embedding"
PHASE_HEADER = "x-skill-bench-phase"

BANK_FILE = "bank.jsonl"
EMBEDDINGS_FILE = "embeddings.jsonl"
EXTRACTIONS_FILE = "extractions.jsonl"
RETRIEVALS_FILE = "retrievals.jsonl"
STATE_FILES = (
    BANK_FILE,
    EMBEDDINGS_FILE,
    EXTRACTIONS_FILE,
    RETRIEVALS_FILE,
    FROZEN_FILE,
    J.RECORD_FILE,
)


# ---------------------------------------------------------------------------
# Frozen prompt text
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def rb_format() -> Mapping[str, Any]:
    """``format.json``: model-visible pieces, limits, provenance and the substitutions."""
    return MappingProxyType(json.loads(protocol.read_protocol_text(FORMAT_FILE)))


def _fmt(key: str) -> Any:
    return rb_format()[key]


@lru_cache(maxsize=2)
def system_prompt(status: str) -> str:
    """Figure 9's extraction instruction for a J ``success`` or ``failure`` (two substitutions)."""
    files = _fmt("system_prompts")
    name = files["success_file"] if status == "success" else files["failure_file"]
    return protocol.read_protocol_text(f"{PROMPT_DIR}/{name}").rstrip("\n")


def retrieval_query(query: str) -> str:
    """RB's instruction-wrapped query (``get_detailed_instruct``), 'web navigation' -> 'task'."""
    return _fmt("retrieval_query").format(task=_fmt("retrieval_task"), query=query)


# ---------------------------------------------------------------------------
# Memory items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryItem:
    """One memory item (paper A.2 schema ``{title, description, content}``)."""

    title: str
    description: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


#: An optional list marker before a label ("- ", "1. ").
_LIST = r"(?:[-*+]\s+|\d+[.)]\s+)?"
_ITEM_HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*[*_]*\s*memory\s+item\b.*"
    rf"|{_LIST}[*_]*\s*memory\s+item\s*(?:\d+|i)?\s*[*_:]*\s*)$",
    re.IGNORECASE,
)
_FIELD_HASHED = re.compile(
    r"^\s*#{1,6}\s*[*_]*\s*(?P<name>title|description|content)\b[*_]*\s*:?[*_]*\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_FIELD_LABELLED = re.compile(
    rf"^\s*{_LIST}[*_]*\s*(?P<name>title|description|content)\s*[*_]*\s*:[*_]*\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^\s*```")


@dataclass(frozen=True)
class ParsedItems:
    """The complete items of a reply, in order, and how many incomplete ones were seen."""

    items: tuple[MemoryItem, ...]
    incomplete: int


def parse_memory_items(text: str | None) -> ParsedItems:
    """Read Figure 9's Markdown items (``# Memory Item i`` / ``## Title`` / ``## Description``
    / ``## Content``) from an extraction reply.

    Tolerant of what models write around the format: code fences, ``**`` emphasis, a
    colon after the label, ``Title: ...`` lines without ``#`` (colon required, optionally
    after a list marker), ``**Memory Item 2**`` headers without ``#``, the value on the
    label's line or on the lines below it. A ``Title`` starts a new item when the
    current one already has fields; an item is complete with a title and a content (a
    missing description stays empty).
    """
    items: list[MemoryItem] = []
    incomplete = 0
    fields: dict[str, list[str]] = {}
    current: str | None = None

    def close() -> None:
        nonlocal fields, current, incomplete
        if fields:
            values = {k: "\n".join(v).strip() for k, v in fields.items()}
            if values.get("title") and values.get("content"):
                items.append(
                    MemoryItem(
                        values["title"],
                        values.get("description", ""),
                        values["content"],
                    )
                )
            else:
                incomplete += 1
        fields, current = {}, None

    for line in (text or "").splitlines():
        if _FENCE.match(line):
            continue
        if _ITEM_HEADER.match(line):
            close()
            continue
        match = _FIELD_HASHED.match(line) or _FIELD_LABELLED.match(line)
        if match:
            name = match.group("name").lower()
            if name == "title" and fields or name in fields:
                close()
            fields[name] = [match.group("rest").strip().strip("*_").strip()]
            current = name
        elif current is not None:
            fields[current].append(line)
    close()
    return ParsedItems(tuple(items), incomplete)


def render_injection(items: Sequence[MemoryItem]) -> str:
    """The paper's sentence, then each item as its title and content (A.2); "" without items."""
    if not items:
        return ""
    rendered = [
        _fmt("injection_item").format(i=i, title=item.title, content=item.content)
        for i, item in enumerate(items, start=1)
    ]
    return _fmt("injection_joiner").join([_fmt("injection_preamble"), *rendered])


# ---------------------------------------------------------------------------
# The extraction request
# ---------------------------------------------------------------------------


def extraction_messages(
    query: str,
    trajectory: StepTrajectory,
    status: str,
    *,
    max_chars: int | None = None,
) -> tuple[list[Message], int]:
    """RB's extraction request: Figure 9's instruction for ``status`` and the query and
    trajectory (``induce_memory.py:142-143``). Returns the messages and the characters
    cut."""
    rendered, cut = render_steps(trajectory, rb_format(), max_chars=max_chars)
    text = _fmt("user_template").format(query=query, trajectory=rendered)
    return [
        {"role": "system", "content": system_prompt(status)},
        {"role": "user", "content": text},
    ], cut


def fit_extraction(
    query: str,
    trajectory: StepTrajectory,
    status: str,
    limits: protocol.ModelLimits,
) -> tuple[list[Message], int, int | None, int]:
    """:func:`extraction_messages` within the input budget that leaves the M3.1 planning
    reserve for the reply (J's rule). Returns messages, estimate, cap and characters cut."""
    (messages, cut), estimate, cap = fit_texts(
        lambda c: extraction_messages(query, trajectory, status, max_chars=c),
        lambda built: estimate_messages(built[0]),
        J.input_budget(limits),
        trajectory.longest_text,
        int(_fmt("min_truncated_chars")),
    )
    return messages, estimate, cap, cut


# ---------------------------------------------------------------------------
# Embeddings through the gateway role ``embedding``
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingCall:
    """Record of one embedding call (the ledger has the rest under ``gateway_call_id``)."""

    gateway_call_id: str | None
    http_status: int | None
    latency_s: float
    inputs: int
    dims: int | None
    usage: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingCallError(GatewayCallError):
    """An embedding call without usable vectors (every kind raises: the item is aborted, H8)."""

    def __init__(self, failure: Failure, call: EmbeddingCall) -> None:
        super().__init__(failure)
        self.call = call


class Embedder:
    """gemini-embedding-001 through the gateway role ``embedding`` of one cell run.

    Sends only ``model`` and ``input`` (the gateway owns the rest); never retries (the
    gateway retries infrastructure failures within its deadline; H10.3 forbids a second
    request from the method).
    """

    def __init__(
        self,
        cell: CellContext,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.cell = cell
        self.url = (
            f"{gateway_url(cell.gateway_base_url, cell.run_id, EMBEDDING_ROLE)}"
            f"/{EMBEDDINGS}"
        )
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(protocol.CLIENT_TIMEOUT_S, connect=30.0),
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(
        self, texts: Sequence[str], *, item: str, phase: str
    ) -> tuple[list[list[float]], EmbeddingCall]:
        headers = {
            **self.cell.headers(item),
            PHASE_HEADER: phase,
            "Content-Type": "application/json",
        }
        body = {"model": protocol.EMBEDDING_MODEL, "input": list(texts)}
        started = time.monotonic()

        def record(**values: Any) -> EmbeddingCall:
            fields: dict[str, Any] = {
                "gateway_call_id": None,
                "http_status": None,
                "latency_s": round(time.monotonic() - started, 3),
                "inputs": len(texts),
                "dims": None,
                "usage": None,
            }
            fields.update(values)
            return EmbeddingCall(**fields)

        try:
            response = await self._client.post(
                self.url, content=json.dumps(body).encode("utf-8"), headers=headers
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # transport errors, client timeout
            failure = classify_exception(exc, EMBEDDING_ROLE)
            raise EmbeddingCallError(
                failure, record(failure=failure.to_dict())
            ) from exc
        call_id = response.headers.get("x-skill-bench-call-id")
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code != 200:
            failure = classify_gateway_error(response.status_code, data, EMBEDDING_ROLE)
            raise EmbeddingCallError(
                failure,
                record(
                    gateway_call_id=call_id,
                    http_status=response.status_code,
                    failure=failure.to_dict(),
                ),
            )
        vectors = _vectors_of(data, len(texts))
        if vectors is None:
            failure = Failure(
                FailureKind.HARNESS,
                "invalid_embedding_response",
                EMBEDDING_ROLE,
                200,
                "gateway returned 200 without one finite vector per input",
            )
            raise EmbeddingCallError(
                failure,
                record(
                    gateway_call_id=call_id, http_status=200, failure=failure.to_dict()
                ),
            )
        usage = data.get("usage") if isinstance(data, Mapping) else None
        return vectors, record(
            gateway_call_id=call_id,
            http_status=200,
            dims=len(vectors[0]),
            usage=dict(usage) if isinstance(usage, Mapping) else None,
        )


def _vectors_of(data: Any, count: int) -> list[list[float]] | None:
    rows = data.get("data") if isinstance(data, Mapping) else None
    if not isinstance(rows, list) or len(rows) != count:
        return None
    ordered = sorted(
        rows,
        key=lambda r: r.get("index", 0) if isinstance(r, Mapping) else 0,
    )
    vectors: list[list[float]] = []
    for row in ordered:
        vector = row.get("embedding") if isinstance(row, Mapping) else None
        if not isinstance(vector, list) or not vector:
            return None
        try:
            values = [float(v) for v in vector]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in values):
            return None
        vectors.append(values)
    if len({len(v) for v in vectors}) != 1:
        return None
    return vectors


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """RB's ``l2_normalize`` (a zero vector stays zero)."""
    norm = math.sqrt(sum(v * v for v in vector))
    return tuple(v / norm for v in vector) if norm else tuple(0.0 for _ in vector)


def cosine_scores(
    query: Sequence[float], bank: Sequence[Sequence[float]]
) -> list[float]:
    """Cosine of ``query`` with every (L2-normalised) bank vector (CPU work)."""
    q = l2_normalize(query)
    for vector in bank:
        if len(vector) != len(q):
            raise ValueError(
                f"embedding dimensions differ: query {len(q)}, bank {len(vector)}"
            )
    return [math.fsum(a * b for a, b in zip(q, vector)) for vector in bank]


def top_index(scores: Sequence[float]) -> int:
    """Index of the highest score; ties go to the earliest entry (RB's stable sort, k = 1)."""
    best = 0
    for i, score in enumerate(scores):
        if score > scores[best]:
            best = i
    return best


def vector_sha256(vector: Sequence[float]) -> str:
    return hashlib.sha256(json.dumps(list(vector)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankEntry:
    """One learned task: ``{query, status, memory_items}`` and its vector's hash."""

    task_uid: str
    query: str
    status: str
    memory_items: tuple[MemoryItem, ...]
    embedding_sha256: str
    embedding_dims: int

    def item_id(self, i: int) -> str:
        return f"{self.task_uid}#{i}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_uid": self.task_uid,
            "query": self.query,
            "status": self.status,
            "memory_items": [
                {"id": self.item_id(i), **item.to_dict()}
                for i, item in enumerate(self.memory_items, start=1)
            ],
            "embedding_sha256": self.embedding_sha256,
            "embedding_dims": self.embedding_dims,
        }


@dataclass
class RbCounters:
    """Learning-phase counts (snapshot; frozen with the memory)."""

    tasks_learned: int = 0
    judged_success: int = 0
    judged_failure: int = 0
    extractions: int = 0
    #: Extraction replies that were truncated or filtered, or that never came (MODEL).
    extraction_model_failures: int = 0
    extraction_no_reply: int = 0
    #: Extraction inputs whose texts were cut further to fit the input budget.
    extraction_inputs_cut: int = 0
    #: Replies without one complete item (no entry is stored for the task).
    unparseable_extractions: int = 0
    incomplete_items: int = 0
    items_stored: int = 0
    #: Complete items beyond the first 3 of a reply (at most 3 per task).
    items_dropped_over_limit: int = 0
    entries: int = 0
    #: Tasks with an empty instruction: nothing to embed, so no retrieval and no entry.
    empty_queries: int = 0
    steps_skipped: int = 0
    #: Learning-phase ``before_task`` calls, and those that injected memory.
    retrievals: int = 0
    retrievals_with_memory: int = 0
    #: Refusals (:data:`REFUSAL_COUNTERS`): extractions the
    #: provider refused (no entry stored) and items whose J call was refused (not
    #: learned from).
    extraction_rejected: int = 0
    judge_rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        """Every counter, except a :data:`REFUSAL_COUNTERS` one while it is 0: a bank
        without refusals snapshots byte for byte as before they were counted, so a
        checkpoint or frozen bank of the code before them still matches its hash."""
        data = asdict(self)
        for key in REFUSAL_COUNTERS:
            if not data[key]:
                del data[key]
        return data


#: The refusal counters of :class:`RbCounters`, written only when non-zero.
REFUSAL_COUNTERS: tuple[str, ...] = ("extraction_rejected", "judge_rejected")


class ReasoningBank(MemoryMethodBase):
    """ReasoningBank: J-judged memory items, k = 1 embedding retrieval."""

    name = "reasoningbank"
    signal_level: SignalLevel = "S1"

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__()
        self._transport = transport
        self._judge: J.Judge | None = None
        self._judge_counters: J.JudgeCounters | None = None
        self._learner: Actor | None = None
        self._embedder: Embedder | None = None
        self._entries: list[BankEntry] = []
        self._vectors: list[tuple[float, ...]] = []
        self._counters = RbCounters()
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
        """Keep a refusal-streak stop for :meth:`_raise_held_stop`, as AWM
        does: raised inside after_task, before ``run.py``'s ``done`` checkpoint of the
        item, a --force-resume from ``played`` would replay the same refused call and stop
        again, so a person could never go on. The item's records are written and
        after_task returns; the next item's before_task, or finalize after the last item,
        raises it. A crash in between loses it: the resumed phase counts on from the
        checkpointed streak, and the next refusal in a row stops it again."""
        if self._held_stop is None:
            held = GatewayCallError(stop)
            held.__cause__ = cause
            self._held_stop = held

    def _raise_held_stop(self) -> None:
        if self._held_stop is not None:
            raise self._held_stop

    def _cell(self) -> CellContext:
        if self.cell is None:
            raise RuntimeError("reasoningbank: setup() was not called")
        return self.cell

    @property
    def entries(self) -> tuple[BankEntry, ...]:
        return tuple(self._entries)

    def _phase(self) -> str:
        return TEST_PHASE if self._frozen is not None else LEARNING_PHASE

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
        else:
            frozen = await asyncio.to_thread(read_frozen, cell.work_dir, self.name)
            await asyncio.to_thread(self._restore, cell.work_dir, frozen)
        self._embedder = Embedder(cell, transport=self._transport)
        if cell.learning and cell.resume is not None:
            check_resumed_snapshot(self.name, await self.snapshot(), cell.resume)

    async def checkpoint(self) -> dict[str, Any]:
        """Contract item 6: the counters, J's counts and the refusal streaks (the bank is
        in ``work_dir``)."""
        return {
            "counters": self._counters.to_dict(),
            "judge": self._judge_counters.to_dict() if self._judge_counters else {},
            "refusal_streaks": self._streaks.to_dict(),
        }

    def _restore_learning(self, cell: CellContext) -> None:
        """Rebuild the bank of a learning checkpoint (blocking I/O): the entries and
        vectors from ``bank.jsonl`` / ``embeddings.jsonl``, the counters and J's counts
        from the checkpoint's state."""
        resume = cell.resume
        assert resume is not None and self._judge is not None
        if (cell.work_dir / FROZEN_FILE).exists():
            raise CheckpointMismatch(
                f"reasoningbank: {cell.work_dir} holds {FROZEN_FILE}: a learning "
                "checkpoint is taken before the memory is frozen"
            )
        try:
            self._load_bank(cell.work_dir)
            self._counters = RbCounters(**dict(resume.state["counters"]))
            self._judge.restore(resume.state["judge"])
            # Absent in a checkpoint of the code before refusals were counted.
            self._streaks = RefusalStreaks(resume.state.get("refusal_streaks"))
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointMismatch(
                f"reasoningbank: checkpoint {resume.checkpoint_id} cannot be restored "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    def _load_bank(self, work_dir: Path) -> None:
        """The entries and their L2-normalised vectors from the bank files (blocking)."""
        vectors: dict[str, list[float]] = {}
        path = work_dir / EMBEDDINGS_FILE
        if path.exists():
            # "\n" only: a row may hold U+2028 (ensure_ascii=False).
            for line in path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                vectors[row["id"]] = row["embedding"]
        path = work_dir / BANK_FILE
        self._entries, self._vectors = [], []
        if path.exists():
            for line in path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                vector = vectors[row["task_uid"]]
                self._entries.append(_entry_from_row(row))
                self._vectors.append(l2_normalize(vector))

    def _restore(self, work_dir: Path, frozen: Mapping[str, Any]) -> None:
        """Load the frozen bank and check it against ``frozen.json`` (blocking I/O)."""
        files = frozen.get("files") or {}
        for name in (BANK_FILE, EMBEDDINGS_FILE):
            if file_sha256(work_dir / name) != files.get(name):
                raise FrozenMemoryMismatch(
                    f"reasoningbank: {work_dir / name} does not match {FROZEN_FILE}"
                )
        self._load_bank(work_dir)
        snapshot = dict(frozen["snapshot"])
        if [e.to_dict() for e in self._entries] != snapshot.get("entries"):
            raise FrozenMemoryMismatch(
                f"reasoningbank: the bank in {work_dir} differs from its frozen snapshot"
            )
        self._frozen = snapshot

    # --- retrieval -------------------------------------------------------------------

    async def before_task(self, task: TaskView) -> Injection:
        cell = self._cell()
        self._raise_held_stop()
        learning = self._frozen is None
        entries, vectors = list(self._entries), list(self._vectors)
        record: dict[str, Any] = {
            "event": "retrieval",
            "at": now(),
            "phase": self._phase(),
            "task_uid": task.task_uid,
            "bank_size": len(entries),
        }
        injection = Injection()
        if entries and task.instruction.strip():
            assert self._embedder is not None
            [vector], call = await self._embedder.embed(
                [retrieval_query(task.instruction)],
                item=task.task_uid,
                phase=self._phase(),
            )
            scores = await asyncio.to_thread(cosine_scores, vector, vectors)
            best = top_index(scores)
            entry = entries[best]
            injection = Injection(
                render_injection(entry.memory_items),
                tuple(entry.item_id(i) for i in range(1, len(entry.memory_items) + 1)),
            )
            record.update(
                embedding=call.to_dict(),
                top_task_uid=entry.task_uid,
                top_score=scores[best],
                item_ids=list(injection.item_ids),
            )
        if learning:
            self._counters.retrievals += 1
            self._counters.retrievals_with_memory += int(not injection.empty)
        await asyncio.to_thread(append_jsonl, cell.work_dir / RETRIEVALS_FILE, record)
        return injection

    # --- learning --------------------------------------------------------------------

    async def after_task(self, trajectory: TrajectoryView) -> None:
        cell = self._cell()
        if self._frozen is not None or not cell.learning:
            raise RuntimeError(
                "reasoningbank: after_task in the frozen test phase (memory is read-only)"
            )
        assert self._judge is not None and self._learner is not None
        async with self._busy:
            await self._learn(cell, trajectory)

    async def _learn(self, cell: CellContext, trajectory: TrajectoryView) -> None:
        assert self._judge is not None and self._learner is not None
        c = self._counters
        verdict, refused = await judge_or_refused(self._judge, trajectory)
        stop = self._streaks.observe(
            J.JUDGE_ROLE, refused.failure if refused is not None else None
        )
        if refused is not None:
            # Not learned from: no extraction, no entry.
            c.judge_rejected += 1
            await asyncio.to_thread(
                append_jsonl,
                cell.work_dir / EXTRACTIONS_FILE,
                judge_refused_record(trajectory.task_uid, refused),
            )
            if stop is not None:
                self._hold_stop(stop, refused)
            return
        assert verdict is not None
        c.tasks_learned += 1
        c.judged_success += int(verdict.success)
        c.judged_failure += int(not verdict.success)
        steps = trajectory_steps(trajectory)
        c.steps_skipped += steps.skipped
        query = trajectory.instruction
        messages, estimate, cap, cut = await asyncio.to_thread(
            fit_extraction, query, steps, verdict.status, cell.model_limits
        )
        index = c.extractions
        c.extractions += 1
        c.extraction_inputs_cut += int(cap is not None)
        result = await call_learner(
            self._learner, messages, index=index, item=trajectory.task_uid
        )
        # A refused extraction has no reply: nothing is parsed and nothing stored.
        stop = self._streaks.observe(LEARNER_ROLE, result.refused)
        c.extraction_rejected += int(result.refused is not None)
        c.extraction_model_failures += int(result.failure is not None)
        c.extraction_no_reply += int(result.content is None and result.refused is None)
        parsed = parse_memory_items(result.content)
        limit = int(_fmt("max_items_per_task"))
        kept = parsed.items[:limit]
        c.incomplete_items += parsed.incomplete
        c.items_dropped_over_limit += len(parsed.items) - len(kept)
        if result.content is not None and not kept:
            c.unparseable_extractions += 1
        stored: BankEntry | None = None
        embedding: dict[str, Any] | None = None
        if kept and not query.strip():
            c.empty_queries += 1
        elif kept:
            assert self._embedder is not None
            [vector], call = await self._embedder.embed(
                [query], item=trajectory.task_uid, phase=LEARNING_PHASE
            )
            embedding = call.to_dict()
            stored = BankEntry(
                trajectory.task_uid,
                query,
                verdict.status,
                tuple(kept),
                vector_sha256(vector),
                len(vector),
            )
            await asyncio.to_thread(
                self._append_entry, cell.work_dir, stored, vector, verdict, result
            )
            self._entries.append(stored)
            self._vectors.append(l2_normalize(vector))
            c.entries += 1
            c.items_stored += len(kept)
        await asyncio.to_thread(
            append_jsonl,
            cell.work_dir / EXTRACTIONS_FILE,
            {
                "event": "extraction",
                "at": now(),
                "index": index,
                "task_uid": trajectory.task_uid,
                "status": verdict.status,
                "verdict_call_id": verdict.gateway_call_id,
                "input_tokens_estimated": estimate,
                "cap": cap,
                "chars_cut": cut,
                "steps": len(steps.steps),
                "skipped_steps": steps.skipped,
                **result.record(),
                "reply": result.content,
                "items_parsed": len(parsed.items),
                "items_incomplete": parsed.incomplete,
                "items_kept": len(kept),
                "stored": stored is not None,
                "embedding": embedding,
            },
        )
        if stop is not None:
            self._hold_stop(stop)

    @staticmethod
    def _append_entry(
        work_dir: Path,
        entry: BankEntry,
        vector: Sequence[float],
        verdict: J.JudgeVerdict,
        result: Any,
    ) -> None:
        # RB keeps the vectors in their own file next to the bank (paper A.2 storage).
        append_jsonl(
            work_dir / EMBEDDINGS_FILE,
            {"id": entry.task_uid, "text": entry.query, "embedding": list(vector)},
        )
        append_jsonl(
            work_dir / BANK_FILE,
            {
                **entry.to_dict(),
                "verdict_call_id": verdict.gateway_call_id,
                "extraction_call_id": result.call.gateway_call_id
                if result.call
                else None,
            },
        )

    async def wait_until_idle(self, timeout_s: float) -> None:
        await self._busy.wait(timeout_s)

    async def finalize(self) -> None:
        cell = self._cell()
        if self._frozen is not None:
            return
        self._raise_held_stop()
        snapshot = self._build_snapshot(frozen=True)

        def write() -> None:
            files = {
                name: file_sha256(cell.work_dir / name)
                for name in (BANK_FILE, EMBEDDINGS_FILE)
            }
            data = {"at": now(), "files": files, "snapshot": snapshot}
            write_atomic(
                cell.work_dir / FROZEN_FILE,
                json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            )

        await asyncio.to_thread(write)
        self._frozen = snapshot

    def _build_snapshot(self, *, frozen: bool) -> dict[str, Any]:
        return {
            "method": self.name,
            "signal_level": self.signal_level,
            "frozen": frozen,
            "entries": [e.to_dict() for e in self._entries],
            "counters": self._counters.to_dict(),
            "judge": self._judge_counters.to_dict() if self._judge_counters else {},
        }

    async def snapshot(self) -> dict[str, Any]:
        if self._frozen is not None:
            return copy.deepcopy(self._frozen)
        return self._build_snapshot(frozen=False)

    async def close(self) -> None:
        for client in (self._judge, self._learner, self._embedder):
            if client is not None:
                await client.aclose()
        self._judge = self._learner = self._embedder = None


def _entry_from_row(row: Mapping[str, Any]) -> BankEntry:
    return BankEntry(
        task_uid=row["task_uid"],
        query=row["query"],
        status=row["status"],
        memory_items=tuple(
            MemoryItem(i["title"], i["description"], i["content"])
            for i in row["memory_items"]
        ),
        embedding_sha256=row["embedding_sha256"],
        embedding_dims=row["embedding_dims"],
    )
