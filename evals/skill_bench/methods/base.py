"""The memory-method interface every method implements.

This module is deliberately import-light (standard library and ``protocol`` only) so a
method, a test double or the audit can import it without the gateway, httpx or MIRIX.

What a method receives, and nothing else (H1.7, H5, Q20):

* :class:`CellContext` from :meth:`MemoryMethod.setup`: where the cell lives, which model
  and limits it uses, how to reach the gateway, and the phase it starts in.
* :class:`TaskView` from :meth:`MemoryMethod.before_task`: the opaque task uid, the
  ``first_messages`` of Q20 (the environment's raw output before the model's first reply),
  the ``instruction`` of Q28 (the agent-visible task statement) and the
  ``agent_context``: the actor's prompt template as a static text
  (``envs/agent_context.py``), the same for every item of a benchmark, empty for a
  benchmark that offers none.
* each trajectory message through :meth:`MemoryMethod.on_message`, in order, as it is
  produced (learning phase only): the first messages, then per step the model's raw
  output (``role="assistant"``) and the environment's raw output (``role="user"``).
* :class:`TrajectoryView` from :meth:`MemoryMethod.after_task` (learning phase only):
  the same messages as one tuple.

Never passed to a method: ``won``, scores, gold answers, task type, ALFWorld ``info``,
SkillOpt ``category`` / ``topic``, raw item ids, or any prompt actually sent (its system
message, its memory slot's content); the template the prompts are rendered from reaches
a method only as ``TaskView.agent_context`` (every method's learner may receive it;
MIRIX attaches it to its learning sessions, ReasoningBank and AWM keep their published
pipelines and ignore it). ``agent/trajectory.py`` builds these views; methods only read
them.

One thing reaches methods by design of Q20; it is accepted and documented, not hidden:

* **ALFWorld termination.** A game has no losing state: it ends when it is won or at the
  50-step cap. A trajectory shorter than the cap therefore reveals ``won`` to every
  method, so on ALFWorld every method effectively sees "S0 + termination" (and the judge
  J sees it too). It is an inherent property of the raw environment trajectory that Q20
  hands to methods; the protocol keeps it. No hook receives the step cap or a message
  count.

What a method returns: an :class:`Injection`. Its ``text`` goes, whole and unchanged, into
the fixed SkillOpt skill slot of the actor prompt (Q21). Methods leave ``tokens`` empty;
the framework counts it with ``o200k_base`` (E10).

Every model call a method makes goes through the gateway with role ``learner``, ``judge``,
``embedding`` (or ``mirix`` for the MIRIX server), using :meth:`CellContext.role_url` and
:meth:`CellContext.headers`, with the cell's model (H1.8).

The method contract (checked for every method by ``tests/method_contract.py``):

1. **Messages.** ``on_message`` and ``after_task`` accept every Q20 message shape
   :func:`evals.skill_bench.agent.trajectory.visible_message` produces, and never raise on
   one. In particular:

   * an assistant turn with **empty content** ``{"role": "assistant", "content": ""}``:
     ALFWorld emits it for a step whose model call got no reply (the environment then
     plays SkillOpt's ``look``), and a filtered reply can be empty on any benchmark.

2. **Phases.** ``setup`` with ``cell.phase == "learning"`` starts from empty memory (every
   learning attempt is new, H8), unless ``cell.resume`` names the learning checkpoint it
   restores (item 6). ``setup`` with ``cell.phase == "test"`` resumes the
   memory frozen at the end of this attempt's learning phase (a harness restarted in the
   test phase, H8): methods persist it under ``cell.work_dir`` by the end of
   ``finalize`` (MIRIX keeps it in the cell database), and a method that cannot find it
   raises rather than start empty. A method that finds its frozen memory missing or
   different from what ``finalize`` froze raises :class:`FrozenMemoryMismatch`, which
   ``run.py`` records as an H10.4 violation, not as a transient failure (X20).
3. **Frozen test phase.** In the test phase only ``before_task`` is called, possibly for
   several items at once: it must be read-only and safe to run concurrently.
   ``snapshot()`` is JSON-serialisable and identical before and after the test phase
   (H10.4).
4. **Idle.** ``wait_until_idle(timeout_s)`` returns once no learning is in flight, or
   raises (``TimeoutError``) after ``timeout_s``; the rollout classifies that as INFRA.
5. **Failures.** A method handles its own model failures (truncated or filtered replies,
   unparseable output): it counts them and carries on. Anything it raises aborts the
   item and is classified by ``infra/failures.py`` (an infrastructure failure in the
   learning phase resumes the attempt from its latest checkpoint, item 6; a harness
   failure stops it for a person, H8).
6. **Checkpoints.** A method that learns can be checkpointed
   between learning items and between an item's last ``on_message`` and its
   ``after_task``: ``run.py`` then copies ``work_dir`` and records
   :meth:`MemoryMethodBase.checkpoint` (a JSON object: what the method holds in memory
   and not in ``work_dir``) and the sha256 of :meth:`snapshot`. ``setup`` with
   ``cell.resume`` (:class:`LearningResume`, learning phase only) on that ``work_dir``
   copy and that state restores the memory instead of starting empty, and raises
   :class:`CheckpointMismatch` when the snapshot it rebuilt does not hash to
   ``resume.snapshot_sha256``. Continuing from there with the same model replies gives
   the same final frozen snapshot as a run that was never interrupted. Bulk data a
   method writes once goes under ``work_dir/objects/`` (:data:`OBJECTS_DIR`) and is
   never modified afterwards (the checkpoint hard-links it); every other file is copied,
   so files rewritten atomically (temporary file and rename) and append-only JSONL
   files are safe.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from evals.skill_bench import protocol
from evals.skill_bench.envs.base import Message

SignalLevel = Literal["none", "S0", "S1"]
SIGNAL_LEVELS: tuple[SignalLevel, ...] = ("none", "S0", "S1")

#: The phase a cell attempt is in when a method is set up (H1.4-H1.6).
CellPhase = Literal["learning", "test"]
CELL_PHASES: tuple[CellPhase, ...] = ("learning", "test")

#: Gateway roles a method may call (``actor`` belongs to the framework).
METHOD_ROLES: tuple[str, ...] = ("learner", "judge", "embedding", "mirix")

#: The gateway's local credential (``infra.gateway.LOCAL_TOKEN``; not a provider key).
#: Repeated here so this module stays import-light; a test keeps the two equal.
GATEWAY_LOCAL_TOKEN = "skill-bench-local-credential-not-a-provider-key"
#: Request header that tags a gateway call with the opaque task uid (ledger ``item``).
ITEM_HEADER = "x-skill-bench-item"

#: The H6 hooks, in the order the framework calls them for a learning item.
HOOKS: tuple[str, ...] = (
    "setup",
    "before_task",
    "on_message",
    "after_task",
    "wait_until_idle",
    "finalize",
    "snapshot",
)


#: Where a method keeps bulk data it writes once and never modifies (contract item 6):
#: a learning checkpoint hard-links ``work_dir/objects/**`` instead of copying it.
OBJECTS_DIR = "objects"

#: Learning checkpoint kinds (``cell_records.CHECKPOINT_KINDS``).
CheckpointKind = Literal["init", "played", "done"]
CHECKPOINT_KINDS: tuple[CheckpointKind, ...] = ("init", "played", "done")


def _empty() -> Mapping[str, Any]:
    return MappingProxyType({})


class FrozenMemoryMismatch(RuntimeError):
    """A test-phase ``setup`` found the frozen memory missing or changed since ``finalize``
    froze it (H8, H10.4): the attempt's memory cannot be trusted any more. ``run.py``
    turns it into a violation of the attempt, never into a pending item (X20)."""


class CheckpointMismatch(RuntimeError):
    """A learning-phase ``setup`` with ``cell.resume`` could not rebuild the memory the
    checkpoint recorded: a file is missing or changed, or the rebuilt snapshot does not
    hash to ``resume.snapshot_sha256`` (contract item 6). ``run.py`` classifies it as a
    HARNESS failure (``checkpoint_mismatch``): the attempt stops for a person."""


@dataclass(frozen=True)
class LearningResume:
    """The learning checkpoint a resumed learning phase starts from (contract item 6).

    ``state`` is what the method's :meth:`MemoryMethodBase.checkpoint` returned then;
    ``work_dir`` already holds that checkpoint's copy of the method's files.
    ``position`` is the learning item of a ``played`` or ``done`` checkpoint (0 for
    ``init``).

    ``rollbacks`` is how many learning resumes in a row, this one included, restored this
    same checkpoint after an INFRA ``mirix_learning`` failure
    (``cell_records.consecutive_rollbacks``, computed by ``run.py``). Only MIRIX reads it:
    the failure of the batch it replays is accepted once ``rollbacks`` reaches
    ``infra/failures.MIRIX_BATCH_ROLLBACKS``."""

    checkpoint_id: str
    kind: CheckpointKind
    position: int
    state: Mapping[str, Any]
    snapshot_sha256: str
    rollbacks: int = 0

    def __post_init__(self) -> None:
        if self.kind not in CHECKPOINT_KINDS:
            raise ValueError(f"checkpoint kind must be one of {CHECKPOINT_KINDS}")
        if self.position < 0:
            raise ValueError("a checkpoint position is never negative")
        if (
            not isinstance(self.rollbacks, int)
            or isinstance(self.rollbacks, bool)
            or self.rollbacks < 0
        ):
            raise ValueError("rollbacks counts resumes: an integer, never negative")
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))


def _plain_json(value: Any) -> Any:
    """Plain JSON types, as ``agent/trajectory.to_jsonable`` makes them (a method never
    imports the scorer side, X10): dataclasses (their ``to_dict``), mappings, tuples,
    paths, and NaN as null."""
    if is_dataclass(value) and not isinstance(value, type):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _plain_json(to_dict())
        return {f.name: _plain_json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value != value:
        return None
    return value


def snapshot_sha256(snapshot: Any) -> str:
    """sha256 of a snapshot's canonical JSON (sorted keys, no spaces, UTF-8): the hash
    ``run.py`` records for every snapshot (``cell_records.canonical_sha256``)."""
    data = json.dumps(
        _plain_json(snapshot), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def check_resumed_snapshot(method: str, snapshot: Any, resume: LearningResume) -> None:
    """Raise :class:`CheckpointMismatch` unless a method's rebuilt ``snapshot`` hashes to
    the one its checkpoint recorded."""
    actual = snapshot_sha256(snapshot)
    if actual != resume.snapshot_sha256:
        raise CheckpointMismatch(
            f"{method}: the memory rebuilt from checkpoint {resume.checkpoint_id} "
            f"hashes to {actual}, not to the recorded {resume.snapshot_sha256}"
        )


def checkpoint_state_json(state: Any) -> dict[str, Any]:
    """A method's :meth:`MemoryMethodBase.checkpoint` value as a plain JSON object."""
    if not isinstance(state, Mapping):
        raise TypeError(
            f"checkpoint() must return a JSON object, got {type(state).__name__}"
        )
    return json.loads(json.dumps(dict(state), ensure_ascii=False))


@dataclass(frozen=True)
class CellContext:
    """Everything a method needs to set itself up for one cell attempt (H1.3, H6, H8).

    * ``cell_dir`` is the cell's directory (shared by its attempts; the MIRIX server's
      ``CellServerSpec.cell_dir``). ``work_dir`` is the method's private directory for this
      attempt: a fresh one on every learning restart (H8), the same one when the test
      phase of the attempt resumes. ``cell_dir`` defaults to ``work_dir``.
    * ``phase`` is the phase the attempt is in when the method is set up: ``"learning"``
      (start empty) or ``"test"`` (resume the frozen memory; see the module docstring).
    * ``limits`` are the model's F9 limits from ``config/models.json``
      (:class:`~evals.skill_bench.protocol.ModelLimits`, with the gateway's tokenizer
      margin). Left out, they are loaded from ``models.json``. ``context_length`` and
      ``max_output_tokens`` mirror them (MIRIX's ``context_window`` and
      ``max_output_tokens``, H7.2); if given, they must agree with ``models.json``.
    * The gateway: :meth:`role_url` gives the OpenAI-compatible base URL of one role for
      this run, :meth:`headers` the local credential (never a provider key).
    * ``options`` holds method-specific settings chosen by ``run.py`` (for example the
      MIRIX server port).
    * ``resume`` (learning phase only): the checkpoint a resumed learning phase restores
      (contract item 6); None starts a learning phase from empty memory.
    """

    cell_id: str
    attempt: int
    benchmark: str
    method: str
    model: str
    gateway_base_url: str
    run_id: str
    gateway_token: str = field(repr=False)
    work_dir: Path
    context_length: int | None = None
    max_output_tokens: int | None = None
    embedding_model: str = protocol.EMBEDDING_MODEL
    seed: int = protocol.SEED
    options: Mapping[str, Any] = field(default_factory=_empty)
    cell_dir: Path | None = None
    phase: CellPhase = "learning"
    limits: protocol.ModelLimits | None = None
    resume: LearningResume | None = None

    def __post_init__(self) -> None:
        if self.benchmark not in protocol.BENCHMARKS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}")
        if self.model not in protocol.CHAT_MODELS:
            raise ValueError(f"model {self.model!r} is not a protocol chat model")
        if self.attempt < 1:
            raise ValueError("attempt numbers start at 1")
        if self.phase not in CELL_PHASES:
            raise ValueError(f"phase must be one of {CELL_PHASES}, not {self.phase!r}")
        if self.resume is not None:
            if self.phase != "learning":
                raise ValueError(
                    "resume is a learning checkpoint: a test-phase setup resumes the "
                    "frozen memory, never a checkpoint"
                )
            if not isinstance(self.resume, LearningResume):
                raise TypeError("resume must be a LearningResume")
        if self.embedding_model != protocol.EMBEDDING_MODEL:
            raise ValueError(
                f"the protocol embedding model is {protocol.EMBEDDING_MODEL}"
            )
        limits = self.limits or protocol.model_limits(self.model)
        if limits.model != self.model:
            raise ValueError(f"limits are for {limits.model}, not {self.model}")
        for name, value, frozen in (
            ("context_length", self.context_length, limits.context_length),
            ("max_output_tokens", self.max_output_tokens, limits.max_completion_tokens),
        ):
            if value is not None and value != frozen:
                raise ValueError(
                    f"{name}={value} disagrees with config/models.json ({frozen}) for "
                    f"{self.model}"
                )
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "context_length", limits.context_length)
        object.__setattr__(self, "max_output_tokens", limits.max_completion_tokens)
        object.__setattr__(self, "work_dir", Path(self.work_dir))
        object.__setattr__(
            self, "cell_dir", Path(self.cell_dir) if self.cell_dir else self.work_dir
        )
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def model_limits(self) -> protocol.ModelLimits:
        """``limits``, typed as always set (``__post_init__`` fills it)."""
        assert self.limits is not None
        return self.limits

    @property
    def learning(self) -> bool:
        return self.phase == "learning"

    def role_url(self, role: str) -> str:
        """OpenAI-compatible base URL of one gateway role for this cell (``.../runs/<run>/<role>/v1``)."""
        if role not in METHOD_ROLES:
            raise ValueError(f"methods may call roles {METHOD_ROLES}, not {role!r}")
        return f"{self.gateway_base_url.rstrip('/')}/runs/{self.run_id}/{role}/v1"

    @property
    def role_urls(self) -> Mapping[str, str]:
        """:meth:`role_url` for every role a method may call."""
        return MappingProxyType({role: self.role_url(role) for role in METHOD_ROLES})

    def headers(self, item: str | None = None) -> dict[str, str]:
        """Request headers for a gateway call: the local credential and, optionally, the
        opaque task uid the call belongs to (the ledger's ``item``)."""
        headers = {"Authorization": f"Bearer {self.gateway_token}"}
        if item is not None:
            headers[ITEM_HEADER] = item
        return headers


@dataclass(frozen=True)
class TaskView:
    """What ``before_task`` receives (H6): opaque uid, Q20 ``first_messages``, Q28
    ``instruction``, and the ``agent_context``: the actor's prompt template as a static
    text with its per-step values as named placeholders (the environment's
    ``agent_context()``, ``envs/agent_context.py``), the same for every item of the
    benchmark and in both phases; empty when the benchmark offers none. A method may use
    it or ignore it."""

    task_uid: str
    instruction: str
    first_messages: tuple[Message, ...]
    agent_context: str = ""


@dataclass(frozen=True)
class Injection:
    """What ``before_task`` returns (H6, E10).

    ``text`` is the method's whole memory text, including its own framing words; the
    framework puts it into the SkillOpt skill slot unchanged (Q21). An empty or
    whitespace-only text injects nothing. ``item_ids`` names the memory entries it came
    from (for diagnostics). ``tokens`` is filled in by the framework (``o200k_base``);
    a value set by the method is ignored and overwritten.

    ``degraded`` is set by a method that injects a retrieval it knows lost part of its
    input to a failure that was not the model's (MIRIX, rule N1). It is never
    model-visible: the scorer side flags the item's
    row (``retrieval_degraded``) and its ``injections.jsonl`` row (``degraded``). Records
    carry the key only when it is set (:meth:`to_dict`).
    """

    text: str = ""
    item_ids: tuple[str, ...] = ()
    tokens: int | None = None
    degraded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(
                f"Injection.text must be str, got {type(self.text).__name__}"
            )
        ids = tuple(self.item_ids)
        if not all(isinstance(i, str) for i in ids):
            raise TypeError("Injection.item_ids must be strings")
        if not isinstance(self.degraded, bool):
            raise TypeError(
                f"Injection.degraded must be bool, got {type(self.degraded).__name__}"
            )
        object.__setattr__(self, "item_ids", ids)

    def to_dict(self) -> dict[str, Any]:
        """The record of the injection: ``degraded`` only when set, so every record of an
        injection that is not degraded stays as it was before the flag existed."""
        data: dict[str, Any] = {
            "text": self.text,
            "item_ids": list(self.item_ids),
            "tokens": self.tokens,
        }
        if self.degraded:
            data["degraded"] = True
        return data

    @property
    def empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class TrajectoryView:
    """What ``after_task`` receives (H6, Q20): the raw messages of one episode, nothing else.

    ``messages`` is the whole trajectory in order and starts with ``first_messages``.
    Model reasoning returned separately by the provider is not included (audit only).
    """

    task_uid: str
    instruction: str
    first_messages: tuple[Message, ...]
    messages: tuple[Message, ...]


@runtime_checkable
class MemoryMethod(Protocol):
    """The framework calls, per learning item (H1.4):
    ``before_task`` -> ``on_message`` for each message -> ``after_task`` -> ``wait_until_idle``;
    at the end of the learning stream ``finalize`` -> ``wait_until_idle`` -> ``snapshot``;
    per test item only ``before_task`` (memory is frozen, items may run concurrently).

    See the module docstring for the contract every method keeps (message shapes,
    phases and resume, the frozen test phase, the idle wait, failures).
    """

    name: str
    signal_level: SignalLevel

    async def setup(self, cell: CellContext) -> None: ...

    async def before_task(self, task: TaskView) -> Injection: ...

    async def on_message(self, task_uid: str, message: Message) -> None: ...

    async def after_task(self, trajectory: TrajectoryView) -> None: ...

    async def wait_until_idle(self, timeout_s: float) -> None: ...

    async def finalize(self) -> None: ...

    async def snapshot(self) -> dict[str, Any]: ...


class MemoryMethodBase:
    """No-op implementation of :class:`MemoryMethod` for methods to subclass.

    :meth:`close` is not part of H6: ``run.py`` calls it once when the cell attempt ends
    (the MIRIX method stops its server there).
    """

    name: str = "base"
    signal_level: SignalLevel = "none"

    def __init__(self) -> None:
        self.cell: CellContext | None = None

    async def setup(self, cell: CellContext) -> None:
        self.cell = cell

    async def before_task(self, task: TaskView) -> Injection:
        return Injection()

    async def on_message(self, task_uid: str, message: Message) -> None:
        return None

    async def after_task(self, trajectory: TrajectoryView) -> None:
        return None

    async def wait_until_idle(self, timeout_s: float) -> None:
        return None

    async def finalize(self) -> None:
        return None

    async def snapshot(self) -> dict[str, Any]:
        return {"method": self.name}

    async def checkpoint(self) -> dict[str, Any]:
        """What the method holds in memory and not in ``work_dir``, as a JSON object
        (contract item 6): a learning ``setup`` with ``cell.resume`` gets it back as
        ``cell.resume.state``. The base method keeps everything it has on disk (or has
        nothing to keep, like the no-memory method)."""
        return {}

    async def close(self) -> None:
        return None


def method_class(name: str) -> type | None:
    """The class implementing method ``name`` (one of ``protocol.METHODS``).

    It is the one class defined in ``evals.skill_bench.methods.<name>`` that has every H6
    hook and whose ``name`` attribute is ``name``. Returns None while that module is still
    a placeholder (it defines no such class at all). Raises if the name is unknown, if
    several classes qualify, or if the module defines method classes none of which is
    named ``name`` (a misnamed method must not pass for a placeholder).
    """
    if name not in protocol.METHODS:
        raise ValueError(
            f"unknown method {name!r}; protocol methods: {protocol.METHODS}"
        )
    module = importlib.import_module(f"evals.skill_bench.methods.{name}")
    candidates = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and obj.__module__ == module.__name__
        and all(callable(getattr(obj, hook, None)) for hook in HOOKS)
    ]
    found = [obj for obj in candidates if getattr(obj, "name", None) == name]
    if len(found) > 1:
        raise ValueError(
            f"{module.__name__} defines several classes named {name!r}: "
            + ", ".join(cls.__name__ for cls in found)
        )
    if not found and candidates:
        raise ValueError(
            f"{module.__name__} defines method classes "
            + ", ".join(
                f"{c.__name__} (name={getattr(c, 'name', None)!r})" for c in candidates
            )
            + f" but none is named {name!r}"
        )
    return found[0] if found else None
