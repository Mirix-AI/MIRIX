"""Shared environment contract for every benchmark.

An episode is seen by three parties, and this module keeps their views apart:

* the **scorer** (``run.py`` / ``report.py``) holds the whole :class:`Task`,
  including ``Task.raw`` (gold answers, task type, source pages ...) and every
  ``private`` mapping, and produces the :class:`EpisodeResult`;
* the **actor** (``agent/``) builds prompts from ``messages`` and ``fields`` of
  :class:`Reset` / :class:`StepResult`;
* **memory methods** only ever receive the method-visible view built by
  ``agent/trajectory.py``: the raw ``messages`` (Q20), ``instruction`` (Q28)
  and the opaque task uid from :func:`opaque_task_uid` (H5).

Rules for environment implementers:

1. ``messages`` are OpenAI chat messages holding the environment's *raw* output
   only (observation text, admissible actions). No system prompt, no template
   text, no memory slot.
2. ``fields`` carry the same agent-visible information in structured form so a
   frozen prompt template can be filled (e.g. ``observation`` and
   ``admissible_actions`` for ALFWorld). Nothing scorer-only may go here.
3. Scorer-only data (``won``, ALFWorld ``info`` with ``facts`` / ``expert_plan``,
   gold answers, task type, source files, page numbers, SkillOpt
   ``category`` / ``topic``) lives only in ``Task.raw`` or ``private``.
4. ``Task.raw_id`` is private too; methods see :func:`opaque_task_uid` instead.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from evals.skill_bench.protocol import OPAQUE_UID_HEX_CHARS, OPAQUE_UID_PREFIX

Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")

#: One OpenAI chat-completions message (``{"role": ..., "content": ...}``).
Message = dict[str, Any]

#: What the rollout hands to :meth:`Episode.step`: the model's raw output (or an
#: environment's own no-reply action, ``call_error_response``).
Action = str | Mapping[str, Any]


def _empty() -> Mapping[str, Any]:
    return MappingProxyType({})


def new_cell_salt() -> bytes:
    """Return a fresh random salt; one per cell, stored with the cell's private state."""
    return secrets.token_bytes(32)


def opaque_task_uid(salt: bytes, raw_id: str) -> str:
    """Return the task id methods see: ``"t"`` + first 20 hex chars of HMAC-SHA256 (H5).

    The result matches MIRIX's ``session_id`` rule (``[A-Za-z0-9_-]{1,64}``) and hides
    the manifest order, which would otherwise leak the ALFWorld task type.
    """
    if not salt:
        raise ValueError("salt must be non-empty")
    digest = hmac.new(salt, raw_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return OPAQUE_UID_PREFIX + digest[:OPAQUE_UID_HEX_CHARS]


@dataclass(frozen=True)
class Task:
    """One benchmark item as loaded from a split manifest (scorer-side object).

    ``raw`` keeps the full dataset record, including labels. It is excluded from
    ``repr`` and equality so it cannot leak through logging by accident.
    """

    benchmark: str
    split: Split
    raw_id: str
    raw: Mapping[str, Any] = field(default_factory=_empty, repr=False, compare=False)

    def uid(self, salt: bytes) -> str:
        """Opaque, method-visible id of this task for one cell (see :func:`opaque_task_uid`)."""
        return opaque_task_uid(salt, self.raw_id)


@dataclass(frozen=True)
class Reset:
    """Start of an episode.

    ``instruction`` is the agent-visible task statement (Q28): ALFWorld's
    "Your task is to: ..." sentence, or WebShop's instruction.
    ``messages`` are the ``first_messages`` of Q20: everything the environment
    shows before the model's first reply.
    """

    instruction: str
    messages: tuple[Message, ...]
    fields: Mapping[str, Any] = field(default_factory=_empty)
    private: Mapping[str, Any] = field(default_factory=_empty, repr=False)


@dataclass(frozen=True)
class StepResult:
    """Environment output after one action.

    ``messages`` is the raw output appended to the trajectory. ``done`` ends the
    episode; ``private`` of the last
    step is what :meth:`Env.score` receives (e.g. ALFWorld ``won``).
    """

    messages: tuple[Message, ...]
    done: bool
    fields: Mapping[str, Any] = field(default_factory=_empty)
    private: Mapping[str, Any] = field(default_factory=_empty, repr=False)


@dataclass(frozen=True)
class Score:
    """Scorer output for one item (H9).

    ``primary`` is the main metric: ALFWorld success (0/1), WebShop's score.
    ``metrics`` holds the secondary ones. ``answered`` is False when the item
    ended unanswered (a WebShop episode without a purchase, a provider refusal);
    such items score 0.
    """

    primary: float
    metrics: Mapping[str, float] = field(default_factory=_empty)
    answered: bool = True


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one episode, written by the scorer and never shown to methods.

    ``score`` is None exactly when the episode was not scored because of an
    infrastructure or harness failure (``failure`` then says which, see
    ``infra/failures.py``). A model failure (format error, wrong answer,
    timeout) is scored normally; ``failure`` holds one that ended the episode, and
    ``model_failures`` counts every call-level model failure (truncated, filtered or
    timed-out replies), including those the episode continued after.
    """

    benchmark: str
    split: Split
    raw_id: str
    task_uid: str
    score: Score | None
    final_answer: str | None = None
    steps: int = 0
    format_errors: int = 0
    model_failures: int = 0
    injected_tokens: int = 0
    elapsed_s: float = 0.0
    failure: Mapping[str, Any] | None = None

    @property
    def scored(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (mapping proxies become plain dicts)."""
        return _plain(self)


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@runtime_checkable
class Episode(Protocol):
    """One live episode. Environments create a fresh object per task so test items can run concurrently."""

    async def reset(self) -> Reset: ...

    async def step(self, action: Action) -> StepResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class Env(Protocol):
    """A benchmark: task loading, episode factory and scoring.

    ``episode()`` must be cheap; heavy setup (game files, sandbox container)
    belongs in :meth:`Episode.reset`. ``score()`` is only called by the scorer.
    """

    benchmark: str

    def load_tasks(self, split: Split) -> list[Task]: ...

    def episode(self, task: Task) -> Episode: ...

    def score(
        self, task: Task, final_answer: str | None, private: Mapping[str, Any]
    ) -> Score: ...
