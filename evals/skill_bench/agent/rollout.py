"""The rollout loops, the fixed memory slot and the method hooks.

:func:`run_episode` plays one task with one memory method::

    reset = episode.reset()
    injection = method.before_task(TaskView)            # both phases (Q19); its
                                                        # agent_context is the env's
                                                        # static template
    [learning] method.on_message(uid, m) for each first message
    loop:
        actor call with injection.text in the SkillOpt skill slot (Q21)
        [learning] method.on_message(uid, assistant message)
        step = episode.step(...)
        [learning] method.on_message(uid, m) for each environment message
    score (env.score, scorer side; never shown to the method)
    [learning] on_played(EpisodeRun)                    # run.py: records + checkpoint
    [learning] method.after_task(TrajectoryView); method.wait_until_idle(timeout)

In the test phase only ``before_task`` is called (H1.6): memory is frozen and items may
run concurrently.

``on_played`` (learning phase only; stage ``played``) is the caller's hook between the
score and ``after_task``: ``run.py`` writes the item's trajectory, learning row and
injection row there and takes the item's ``played`` checkpoint, so a learning phase
interrupted during ``after_task`` or the idle wait resumes with the item already played
and scored. :func:`finish_learning` runs the same two learning
hooks for such a resumed item, on the view rebuilt from its recorded trajectory.

The two loops (one driver each):

* **ALFWorld** (text loop): each call is SkillOpt's single-turn prompt rendered from the
  environment fields (``env.actor_messages(fields, memory)``); the raw response goes to
  ``episode.step``, which applies SkillOpt's ``look`` fallbacks. At most 50 steps.
* **WebShop** (the same text loop): verl-agent's single-turn prompt; the raw response goes
  to ``episode.step``, which plays a frozen no-op for a response without an action tag.
  At most ``protocol.WEBSHOP_MAX_STEPS`` (15) steps; the episode ends at Buy Now, whose
  score line is the last message methods see (learning phase only).

The memory text is placed only by the environment's ``actor_messages`` in the fixed slot
(Q21): the start of the ALFWorld user prompt, WebShop's memory section. Nothing else in the prompt depends on the method (H10.2); the record keeps
the sha256 of the first prompt rendered without memory so ``audit.py`` can compare methods.

Failures (``infra/failures.py``). Model failures are scored normally (H8), the way
SkillOpt 79124b3 treats the same event (H4):

* A reply that arrived is used exactly as SkillOpt uses any reply, whatever its
  ``finish_reason``: SkillOpt's backends return ``message.content`` without looking at it.
  A ``length`` (truncated) or ``content_filter`` reply is recorded as a MODEL failure on
  the step (``StepRecord.fields["model_failure"]``, ``Trajectory.model_failures``) and the
  episode goes on: ALFWorld parses it (usually the missing-action ``look``), WebShop plays
  its action or its frozen no-op. Methods see it as the model's raw output (Q20).
* A call without a reply that is the model's fault (a single attempt that had the whole
  deadline, an upstream moderation refusal): ALFWorld plays SkillOpt's call-error fallback
  (``rollout.py:244-245``: ``<think>error</think><action>look</action>``, frozen in
  ``templates/alfworld/format.json``) and goes on, the method seeing an empty assistant
  turn; WebShop plays its frozen no-op the same way.
* An actor call the provider refuses (``infra/failures.upstream_refusal``: a payload
  refused with HTTP 413, or 400 naming its size or images, a no-endpoint 404) ends the
  item unanswered, scored 0,
  when the caller names its class in ``unserved`` (rules harness-stop-learning and
  gw-404-no-endpoint): ``run.py`` passes every class in the
  learning phase, and ``no_endpoint`` for a test item's try after a no-endpoint refusal.
  The trajectory ends ``provider_unserved`` (:data:`END_PROVIDER_UNSERVED`) with the
  failure ``provider_unserved`` (HARNESS kind, never a model failure); no fallback is
  played (retrying a refused request cannot help), and in the learning phase the method
  sees the unanswered trajectory as it sees a model's no-reply. By default nothing is
  tolerated and the refusal raises as below.
* Anything else (INFRA, HARNESS, environment or method errors) raises
  :class:`RolloutError` with the classified failure, the stage, the partial trajectory and
  an unscored result (or the scored one if only ``on_played``, ``after_task`` or the idle
  wait failed). ``run.py`` applies H8: resume the learning phase from its checkpoint (a
  harness failure waits for a person), re-run the item in the test phase. Cancellation
  propagates unchanged. The episode is always closed.

Known label channel, accepted (inherent to handing methods the raw messages): an
ALFWorld game has no losing state, so an episode that ends before the 50-step cap was
won. Every method (and the judge J) can read that from the number of messages, so on
ALFWorld every method effectively sees "S0 + termination". The protocol keeps it (the
raw environment trajectory is what Q20 hands to methods); nothing here hands methods the
step cap or a message count. On WebShop the score itself is the last learning-phase
message after Buy Now (it mirrors real user feedback and is the same for every method):
"S0 + score" there.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from evals.skill_bench import protocol
from evals.skill_bench.agent.actor import Actor, ActorCallError, ActorReply
from evals.skill_bench.agent.trajectory import (
    StepRecord,
    Trajectory,
    canonical_json,
    sha256_hex,
    visible_message,
    visible_messages,
)
from evals.skill_bench.envs.agent_context import env_agent_context
from evals.skill_bench.envs.base import (
    Action,
    Env,
    EpisodeResult,
    Message,
    Reset,
    Score,
    StepResult,
    Task,
)
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import (
    PROVIDER_UNSERVED,
    REFUSAL_CLASSES,
    Failure,
    FailureKind,
    classify_exception,
    upstream_refusal,
)
from evals.skill_bench.methods.base import Injection, MemoryMethod, TrajectoryView

RolloutPhase = Literal["learning", "test"]
PHASES: tuple[RolloutPhase, ...] = ("learning", "test")

#: Extra time allowed on top of a method's own ``wait_until_idle`` timeout.
IDLE_WAIT_GRACE_S = 30.0

# Failure stages (``RolloutError.stage`` / ``Trajectory.failure_stage``).
STAGE_RESET = "reset"
STAGE_BEFORE_TASK = "before_task"
STAGE_ON_MESSAGE = "on_message"
STAGE_ACTOR = "actor"
STAGE_ENV = "env_step"
STAGE_CLOSE = "close"
STAGE_SCORE = "score"
#: The caller's ``on_played`` hook (learning only): records and the ``played`` checkpoint.
STAGE_PLAYED = "played"
STAGE_AFTER_TASK = "after_task"
STAGE_IDLE_WAIT = "wait_until_idle"
STAGE_HARNESS = "harness"  # an error outside every tagged stage (a harness bug)

# ``Trajectory.end_reason`` values.
END_DONE = "done"
#: The provider refused an actor call of a class the caller tolerates (``unserved``): the
#: item ended unanswered, with a score of 0.
END_PROVIDER_UNSERVED = "provider_unserved"


# ---------------------------------------------------------------------------
# Results and errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRun:
    """A finished episode: the private record and the scored result."""

    trajectory: Trajectory
    result: EpisodeResult


class RolloutError(Exception):
    """An item that could not be completed and scored normally (INFRA or HARNESS, H8)."""

    def __init__(
        self,
        failure: Failure,
        *,
        stage: str,
        trajectory: Trajectory,
        result: EpisodeResult,
    ) -> None:
        super().__init__(
            f"{stage}: {failure.kind.value}/{failure.code}: {failure.detail}"
        )
        self.failure = failure
        self.stage = stage
        self.trajectory = trajectory
        self.result = result


#: ``run_episode``'s ``on_played`` hook: called with the scored episode, learning only.
OnPlayed = Callable[[EpisodeRun], Awaitable[None]]


class LearningHookError(Exception):
    """:func:`finish_learning` failed: ``after_task`` or the idle wait of a resumed played
    item (the same stages and classification as in :func:`run_episode`)."""

    def __init__(self, failure: Failure, *, stage: str) -> None:
        super().__init__(
            f"{stage}: {failure.kind.value}/{failure.code}: {failure.detail}"
        )
        self.failure = failure
        self.stage = stage


class _Stage(Exception):
    """Internal: an exception tagged with the stage it came from."""

    def __init__(self, stage: str, error: BaseException) -> None:
        super().__init__(stage)
        self.stage = stage
        self.error = error


class _EnvDidNotFinish(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Drivers: the benchmark-specific part of each loop
# ---------------------------------------------------------------------------


class _PromptEnv(Protocol):
    benchmark: str

    def actor_messages(
        self, fields: Mapping[str, Any], memory: str = ""
    ) -> list[Message]: ...


@dataclass(frozen=True)
class _Request:
    messages: list[Message]

    def fingerprint(self) -> str:
        # The hashed document keeps the ``tools`` and ``tool_choice`` keys (always null
        # here), so every fingerprint equals the one recorded before they were dropped
        # (``first_prompt_sha256``; ``audit.request_fingerprint`` hashes the same keys).
        return sha256_hex(
            canonical_json(
                {"messages": self.messages, "tools": None, "tool_choice": None}
            )
        )


class _Driver:
    """One benchmark loop: builds each actor request, turns replies into actions."""

    max_calls: int
    #: Step fields worth keeping in the private record.
    record_fields: tuple[str, ...] = ()

    def request(self, memory: str) -> _Request:
        raise NotImplementedError

    def without_reply(self) -> tuple[Message, Action]:
        """(method-visible assistant turn, action) when a call had no reply."""
        raise NotImplementedError

    def advance(self, assistant: Message, step: StepResult) -> None:
        return None


class _AlfWorldDriver(_Driver):
    record_fields = ("action", "format_error", "action_valid", "truncated")

    def __init__(self, env: Any, reset: Reset) -> None:
        self.env = cast(_PromptEnv, env)
        self.fields: Mapping[str, Any] = reset.fields
        self.max_calls = int(getattr(env, "max_steps", protocol.ALFWORLD_MAX_STEPS))

    def request(self, memory: str) -> _Request:
        return _Request(self.env.actor_messages(self.fields, memory))

    def without_reply(self) -> tuple[Message, Action]:
        # SkillOpt rollout.py:244-245: a failed call plays its error fallback ('look').
        # No model output exists, so the method sees an empty assistant turn.
        fallback = cast(Any, self.env).call_error_response()
        return {"role": "assistant", "content": ""}, fallback

    def advance(self, assistant: Message, step: StepResult) -> None:
        self.fields = step.fields


class _WebShopDriver(_AlfWorldDriver):
    """WebShop's text loop: ALFWorld's, with WebShop's own step cap
    (``protocol.WEBSHOP_MAX_STEPS``, never ALFWorld's 50) and record fields. A call
    without a reply plays the environment's frozen no-op (``call_error_response``)."""

    record_fields = (
        "action",
        "format_error",
        "action_valid",
        "accepted",
        "rejected",
        "shop_error",
        "truncated",
    )

    def __init__(self, env: Any, reset: Reset) -> None:
        super().__init__(env, reset)
        self.max_calls = int(getattr(env, "max_steps", protocol.WEBSHOP_MAX_STEPS))


_DRIVERS: Mapping[str, type[_Driver]] = {
    "alfworld": _AlfWorldDriver,
    "webshop": _WebShopDriver,
}
if set(_DRIVERS) != set(
    protocol.BENCHMARKS
):  # a benchmark without a loop, or a stale one
    raise RuntimeError(
        f"rollout._DRIVERS {sorted(_DRIVERS)} are not protocol.BENCHMARKS "
        f"{sorted(protocol.BENCHMARKS)}"
    )


def _driver(env: Any, reset: Reset) -> _Driver:
    try:
        cls = _DRIVERS[env.benchmark]
    except KeyError:
        raise ValueError(f"no rollout loop for benchmark {env.benchmark!r}") from None
    return cls(env, reset)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# The episode
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def measure_injection(injection: Any) -> Injection:
    """Validate a method's injection and set its ``o200k_base`` token count (E10).

    A blank text injects nothing and counts 0. Counting runs in a worker thread (the
    tokenizer may load on first use; long memories are CPU work).
    """
    if not isinstance(injection, Injection):
        raise TypeError(
            f"before_task must return an Injection, got {type(injection).__name__}"
        )
    count = (
        await asyncio.to_thread(tokens.count_text, injection.text)
        if not injection.empty
        else 0
    )
    return replace(injection, tokens=count)


def _pick(mapping: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {k: mapping[k] for k in keys if k in mapping}


def _unscored(trajectory: Trajectory, task: Task, failure: Failure) -> EpisodeResult:
    return EpisodeResult(
        benchmark=task.benchmark,
        split=task.split,
        raw_id=task.raw_id,
        task_uid=trajectory.task_uid,
        score=None,
        final_answer=trajectory.final_answer,
        steps=len(trajectory.steps),
        model_failures=len(trajectory.model_failures),
        injected_tokens=trajectory.injected_tokens,
        elapsed_s=trajectory.timings.get("episode_s", 0.0),
        failure=failure.to_dict(),
    )


class _Run:
    """State of one :func:`run_episode` call."""

    def __init__(
        self,
        env: Env,
        task: Task,
        *,
        method: MemoryMethod,
        actor: Actor,
        task_uid: str,
        phase: RolloutPhase,
        unserved: frozenset[str] = frozenset(),
    ) -> None:
        self.env = env
        self.task = task
        self.method = method
        self.actor = actor
        self.phase = phase
        self.learning = phase == "learning"
        #: The refusal classes that end the item unanswered instead of raising.
        self.unserved = unserved
        self.trajectory = Trajectory(
            benchmark=task.benchmark,
            split=task.split,
            raw_id=task.raw_id,
            task_uid=task_uid,
            phase=phase,
            method=str(getattr(method, "name", type(method).__name__)),
            model=actor.model,
            run_id=actor.run_id,
            started_at=_now(),
        )

    async def emit(self, message: Mapping[str, Any]) -> None:
        """Append one Q20 message to the trajectory and, when learning, hand it to the method."""
        try:
            visible = visible_message(message)
        except Exception as exc:  # an environment produced a message outside Q20
            raise _Stage(STAGE_ENV, exc) from exc
        self.trajectory.messages.append(visible)
        if self.learning:
            try:
                await self.method.on_message(
                    self.trajectory.task_uid, visible_message(visible)
                )
            except Exception as exc:
                raise _Stage(STAGE_ON_MESSAGE, exc) from exc

    async def call(self, request: _Request) -> ActorReply | Failure:
        """One actor call: the reply, or the failure of a call that has none: the MODEL
        failure, or ``provider_unserved`` for a refusal of a class in ``unserved``.

        Anything else raises (``_Stage``).
        """
        injection = self.trajectory.injection
        assert injection is not None
        index = len(self.trajectory.calls)
        try:
            reply = await self.actor.complete(
                request.messages,
                index=index,
                item=self.trajectory.task_uid,
                phase=self.phase,
                memory=injection.text,
                memory_tokens=injection.tokens or 0,
            )
        except ActorCallError as exc:
            self.trajectory.calls.append(exc.call)
            if exc.failure.kind is FailureKind.MODEL:  # the model's own doing (H8)
                return exc.failure
            refused = exc.failure
            if refused.role == "actor" and upstream_refusal(refused) in self.unserved:
                # The refused call stays in ``calls`` (H10.2 still rebuilds it).
                return Failure(
                    refused.kind,
                    PROVIDER_UNSERVED,
                    "actor",
                    refused.status,
                    refused.detail,
                    refused.retries,
                )
            raise _Stage(STAGE_ACTOR, exc) from exc
        except Exception as exc:
            raise _Stage(STAGE_ACTOR, exc) from exc
        self.trajectory.calls.append(reply.call)
        return reply

    def _record_model_failure(self, failure: Failure, call_index: int) -> None:
        self.trajectory.model_failures.append(
            {**failure.to_dict(), "call_index": call_index}
        )

    async def play(self, episode: Any, reset: Reset) -> Mapping[str, Any]:
        """The loop; returns the last step's (or the reset's) scorer-side ``private``."""
        driver = _driver(self.env, reset)
        injection = self.trajectory.injection
        assert injection is not None
        private: Mapping[str, Any] = reset.private
        for n in range(driver.max_calls):
            request = driver.request(injection.text)
            if n == 0:  # H10.2 audit inputs (hashed off the loop)
                plain = driver.request("")
                self.trajectory.first_prompt_sha256 = await asyncio.to_thread(
                    request.fingerprint
                )
                self.trajectory.first_prompt_no_memory_sha256 = await asyncio.to_thread(
                    plain.fingerprint
                )
            outcome = await self.call(request)
            call_index = len(self.trajectory.calls) - 1
            reasoning: str | None = None
            model_failure: Failure | None = None
            if isinstance(outcome, Failure) and outcome.code == PROVIDER_UNSERVED:
                # The provider refused the request: no fallback, the item ends unanswered.
                self.trajectory.end_reason = END_PROVIDER_UNSERVED
                self.trajectory.failure = outcome.to_dict()
                self.trajectory.failure_stage = STAGE_ACTOR
                return private
            if isinstance(outcome, Failure):  # no reply at all
                model_failure = outcome
                self._record_model_failure(outcome, call_index)
                assistant, action = driver.without_reply()
            else:
                reply = outcome
                reasoning = reply.reasoning
                if (
                    reply.failure is not None
                ):  # truncated or filtered: used as SkillOpt does
                    model_failure = reply.failure
                    self._record_model_failure(reply.failure, call_index)
                assistant = {"role": "assistant", "content": reply.content}
                action = reply.content
            await self.emit(assistant)
            try:
                step: StepResult = await episode.step(action)
            except Exception as exc:
                raise _Stage(STAGE_ENV, exc) from exc
            for message in step.messages:
                await self.emit(message)
            driver.advance(assistant, step)
            private = step.private
            fields = _pick(step.fields, driver.record_fields)
            if model_failure is not None:
                fields["model_failure"] = model_failure.code
            self.trajectory.steps.append(
                StepRecord(
                    index=len(self.trajectory.steps),
                    call_index=call_index,
                    assistant=visible_message(assistant),
                    reasoning=reasoning,
                    env_messages=visible_messages(step.messages),
                    done=step.done,
                    fields=fields,
                    private=dict(step.private),
                )
            )
            if step.done:
                self.trajectory.done = True
                self.trajectory.end_reason = END_DONE
                return private
        raise _Stage(
            STAGE_ENV,
            _EnvDidNotFinish(
                f"{self.env.benchmark} episode not done after {driver.max_calls} actor calls"
            ),
        )


async def run_episode(
    env: Env,
    task: Task,
    *,
    method: MemoryMethod,
    actor: Actor,
    salt: bytes,
    phase: RolloutPhase,
    idle_timeout_s: float | None = None,
    on_played: OnPlayed | None = None,
    unserved: Collection[str] = (),
) -> EpisodeRun:
    """Play ``task`` with ``method``'s memory and score it (see the module docstring).

    ``salt`` is the cell's secret salt for the opaque task uid (H5). In the learning phase
    ``idle_timeout_s`` is required: it bounds ``method.wait_until_idle`` after ``after_task``
    (Q18, H1.4), and the wait is INFRA-classified when it times out. Skipping the wait
    would let background learning race the next item.

    ``on_played`` (learning phase only) is awaited with the scored episode after the score
    and before ``after_task``; an exception from it raises :class:`RolloutError` at stage
    ``played`` (classified like any other: INFRA for an unreachable gateway, else what
    ``infra/failures.classify_exception`` says), and ``after_task`` is not called.
    Cancellation propagates unchanged.

    ``unserved`` names the refusal classes (``infra/failures.REFUSAL_CLASSES``) of an actor
    call that end the item unanswered and scored instead of raising (``provider_unserved``,
    see the module docstring); the default, none, raises as before.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    if isinstance(unserved, str) or not set(unserved) <= set(REFUSAL_CLASSES):
        raise ValueError(
            f"unserved must be a collection of {REFUSAL_CLASSES}, not {unserved!r}"
        )
    if phase == "learning" and (idle_timeout_s is None or idle_timeout_s <= 0):
        raise ValueError(
            "the learning phase needs idle_timeout_s > 0: every learning item ends with "
            "the Q18 wait_until_idle"
        )
    run = _Run(
        env,
        task,
        method=method,
        actor=actor,
        task_uid=task.uid(salt),
        phase=phase,
        unserved=frozenset(unserved),
    )
    trajectory = run.trajectory
    started = time.monotonic()
    result: EpisodeResult | None = None
    try:
        result = await _play_and_score(run)
        if run.learning:
            trajectory.timings["played_s"] = round(time.monotonic() - started, 3)
            if on_played is not None:
                try:
                    await on_played(EpisodeRun(trajectory, result))
                except Exception as exc:
                    raise _Stage(STAGE_PLAYED, exc) from exc
            await _learn(run, idle_timeout_s)
    except Exception as exc:  # CancelledError / KeyboardInterrupt propagate unchanged
        stage, error = (
            (exc.stage, exc.error) if isinstance(exc, _Stage) else (STAGE_HARNESS, exc)
        )
        failure = classify_exception(error, "actor" if stage == STAGE_ACTOR else None)
        trajectory.failure = failure.to_dict()
        trajectory.failure_stage = stage
        trajectory.timings["total_s"] = round(time.monotonic() - started, 3)
        failed = result if result is not None else _unscored(trajectory, task, failure)
        trajectory.result = failed
        raise RolloutError(
            failure, stage=stage, trajectory=trajectory, result=failed
        ) from error
    trajectory.timings["total_s"] = round(time.monotonic() - started, 3)
    return EpisodeRun(trajectory, result)


async def _play_and_score(run: _Run) -> EpisodeResult:
    env, task, trajectory = run.env, run.task, run.trajectory
    try:
        # The agent context is static (built from the environment's templates, never
        # from a task): it is set before the episode starts, so a trajectory whose reset
        # fails records the same text as every other item, not an empty one.
        trajectory.agent_context = env_agent_context(env)
        episode = env.episode(task)
    except Exception as exc:
        raise _Stage(STAGE_RESET, exc) from exc
    try:
        try:
            reset = await episode.reset()
            trajectory.instruction = reset.instruction
            trajectory.first_messages = visible_messages(reset.messages)
        except Exception as exc:
            raise _Stage(STAGE_RESET, exc) from exc
        t0 = time.monotonic()
        try:
            injection = await run.method.before_task(trajectory.task_view())
            trajectory.injection = await measure_injection(injection)
            trajectory.injection_encoding = tokens.encoding_name()
        except Exception as exc:
            raise _Stage(STAGE_BEFORE_TASK, exc) from exc
        trajectory.timings["before_task_s"] = round(time.monotonic() - t0, 3)
        for message in reset.messages:
            await run.emit(message)
        t0 = time.monotonic()
        private = await run.play(episode, reset)
        trajectory.timings["episode_s"] = round(time.monotonic() - t0, 3)
        trajectory.final_private = dict(private)
    except BaseException:
        with contextlib.suppress(Exception):  # the original error wins
            await episode.close()
        raise
    try:
        await episode.close()
    except Exception as exc:
        raise _Stage(STAGE_CLOSE, exc) from exc

    try:
        score: Score = env.score(task, trajectory.final_answer, private)
    except Exception as exc:
        raise _Stage(STAGE_SCORE, exc) from exc
    if trajectory.end_reason == END_PROVIDER_UNSERVED:
        # Scored 0 as unanswered, whatever the scorer makes of the unfinished episode
        # (S1): ALFWorld's scorer knows no unanswered game and would call it a lost
        # one (answered), which the audit's H9 check rejects.
        score = replace(
            score,
            primary=0.0,
            metrics=MappingProxyType({name: 0.0 for name in score.metrics}),
            answered=False,
        )
    result = EpisodeResult(
        benchmark=task.benchmark,
        split=task.split,
        raw_id=task.raw_id,
        task_uid=trajectory.task_uid,
        score=score,
        final_answer=trajectory.final_answer,
        steps=len(trajectory.steps),
        format_errors=int(getattr(episode, "format_errors", 0) or 0),
        model_failures=len(trajectory.model_failures),
        injected_tokens=trajectory.injected_tokens,
        elapsed_s=trajectory.timings.get("episode_s", 0.0),
        failure=trajectory.failure,
    )
    trajectory.result = result
    return result


async def _learn(run: _Run, idle_timeout_s: float | None) -> None:
    """Learning phase only: ``after_task`` with the view, then the Q18 idle wait."""
    assert idle_timeout_s is not None  # checked by run_episode
    await _after_played(
        run.method, run.trajectory.view(), idle_timeout_s, run.trajectory.timings
    )


async def _after_played(
    method: MemoryMethod,
    view: TrajectoryView,
    idle_timeout_s: float,
    timings: dict[str, float],
) -> None:
    """``after_task(view)``, then the Q18 idle wait (raises :class:`_Stage`)."""
    t0 = time.monotonic()
    try:
        await method.after_task(view)
    except Exception as exc:
        raise _Stage(STAGE_AFTER_TASK, exc) from exc
    timings["after_task_s"] = round(time.monotonic() - t0, 3)
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(
            method.wait_until_idle(idle_timeout_s),
            idle_timeout_s + IDLE_WAIT_GRACE_S,
        )
    except Exception as exc:
        raise _Stage(STAGE_IDLE_WAIT, exc) from exc
    timings["idle_wait_s"] = round(time.monotonic() - t0, 3)


async def finish_learning(
    method: MemoryMethod, view: TrajectoryView, *, idle_timeout_s: float
) -> dict[str, float]:
    """The learning hooks of an item that was already played and scored: ``after_task``
    with its view, then the Q18 idle wait, exactly as :func:`run_episode` runs them
    (``run.py`` resuming from the item's ``played`` checkpoint). Returns their timings;
    a failure raises :class:`LearningHookError` with the stage (``after_task`` or
    ``wait_until_idle``) and the classified failure. Cancellation propagates."""
    if idle_timeout_s <= 0:
        raise ValueError("finish_learning needs idle_timeout_s > 0 (Q18)")
    timings: dict[str, float] = {}
    try:
        await _after_played(method, view, idle_timeout_s, timings)
    except _Stage as exc:
        raise LearningHookError(
            classify_exception(exc.error), stage=exc.stage
        ) from exc.error
    return timings
