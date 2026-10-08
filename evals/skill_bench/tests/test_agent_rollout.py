"""The rollout loops through the gateway.

Every test runs the real gateway in-process over the scriptable fake upstream
(``tests/fakes.py``); nothing reaches a provider. SkillOpt parity tests load SkillOpt
79124b3's own code from the read-only clone and skip without it; real ALFWorld games use
``$ALFWORLD_DATA`` (or ``~/.cache/alfworld``) and skip without it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent import rollout as ro
from evals.skill_bench.agent.actor import Actor
from evals.skill_bench.agent.rollout import EpisodeRun, RolloutError, run_episode
from evals.skill_bench.agent.trajectory import (
    find_label_leaks,
    read_trajectory,
    write_trajectory,
)
from evals.skill_bench.envs import alfworld as alf
from evals.skill_bench.envs.base import Task
from evals.skill_bench.infra import tokens
from evals.skill_bench.infra.failures import (
    GATEWAY_REJECTED,
    GATEWAY_UNREACHABLE,
    IDLE_WAIT_TIMEOUT,
    MIRIX_SERVER,
    MODEL_TIMEOUT,
    TRUNCATED,
    UNEXPECTED_EXCEPTION,
    UPSTREAM_INFRA,
    UPSTREAM_REJECTED,
    FailureKind,
)
from evals.skill_bench.infra.mirix_server import MirixServerError
from evals.skill_bench.methods.base import TaskView
from evals.skill_bench.methods.none import NoMemory
from evals.skill_bench.tests.fakes import FakeUpstream, Reply
from evals.skill_bench.tests.rollout_support import (
    SALT,
    RecordingMethod,
    Script,
    gateway_actor,
    ledger_calls,
    slot_insertions,
    text_reply,
)
from evals.skill_bench.tests.skillopt_ref import requires_skillopt
from evals.skill_bench.tests.test_env_alfworld import (
    ALFWORLD_DATA,
    TRICKY_COMMANDS,
    TRICKY_FORMAT_ERRORS,
    TRICKY_RESPONSES,
    WINNING_RESPONSES,
    Games,
    ScriptedGame,
    act,
    requires_alfworld_data,
    run_skillopt,
)

MEMORY = "§MEM§ Lessons learned:\n- check the {receptacle} first & 'quote' \"x\"\n§END§"


def wire(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def upstream_bodies(fake: FakeUpstream) -> list[dict[str, Any]]:
    return [call.json for call in fake.calls("chat/completions")]


# --- environments -------------------------------------------------------------------------


def alfworld_env(
    tmp_path: Path, games: Games | None = None, **kw: Any
) -> alf.AlfWorldEnv:
    games = games or Games()
    return alf.AlfWorldEnv(data_root=tmp_path, backend_factory=games.factory(**kw))


def alfworld_task() -> Task:
    return alf.AlfWorldEnv(backend_factory=Games().factory()).load_tasks("train")[0]


async def play(
    tmp_path: Path,
    env: Any,
    task: Task,
    replies: list[Reply],
    *,
    method: Any = None,
    phase: ro.RolloutPhase = "test",
    unserved: tuple[str, ...] = (),
    **gateway: Any,
) -> tuple[EpisodeRun, FakeUpstream]:
    fake = FakeUpstream(responder=Script(**{env.benchmark: replies}))
    async with gateway_actor(tmp_path, fake, **gateway) as (_gateway, actor):
        run = await run_episode(
            env,
            task,
            method=method or NoMemory(),
            actor=actor,
            salt=SALT,
            phase=phase,
            idle_timeout_s=5.0 if phase == "learning" else None,
            unserved=unserved,
        )
    return run, fake


# --- SkillOpt parity of the actor prompt (H10.2) -------------------------------------------


@requires_skillopt
@pytest.mark.asyncio
async def test_alfworld_actor_calls_are_byte_identical_to_skillopt(tmp_path):
    games = Games()
    env = alfworld_env(tmp_path, games)
    task = alfworld_task()
    run, fake = await play(
        tmp_path, env, task, [text_reply(r) for r in TRICKY_RESPONSES]
    )
    calls, results = run_skillopt(
        ScriptedGame(Path("x"), 0, "train"),
        TRICKY_RESPONSES,
        out_root=str(tmp_path / "so"),
    )
    skillopt = [
        [
            {"role": "system", "content": c["system"]},
            {"role": "user", "content": c["user"]},
        ]
        for c in calls
    ]
    sent = [body["messages"] for body in upstream_bodies(fake)]
    assert len(sent) == len(skillopt) == len(TRICKY_RESPONSES)
    assert wire(sent[0]) == wire(skillopt[0])  # the first prompt, byte for byte
    assert [wire(m) for m in sent] == [wire(m) for m in skillopt]  # and every later one
    assert [c.request["messages"] for c in run.trajectory.calls] == skillopt
    assert games.made[0].commands == TRICKY_COMMANDS
    assert run.result.score is not None and run.result.score.primary == 1.0
    assert results[0]["hard"] == 1
    assert run.result.format_errors == TRICKY_FORMAT_ERRORS
    assert run.result.steps == len(TRICKY_RESPONSES)


# --- memory only in the Q21 slot -------------------------------------------------------------


async def first_requests(
    tmp_path: Path, make: Any, replies: Any, memory: str | None
) -> Any:
    env, task = make()
    method = RecordingMethod(memory) if memory is not None else NoMemory()
    run, fake = await play(tmp_path, env, task, replies(), method=method)
    return run, upstream_bodies(fake)


@pytest.mark.asyncio
async def test_memory_changes_only_the_skill_slot(tmp_path):
    def make() -> tuple[Any, Task]:
        return alfworld_env(tmp_path), alfworld_task()

    def replies() -> list[Reply]:
        return [text_reply(act("look")), text_reply(act("win"))]

    # ALFWorld: start of the user prompt with SkillOpt's joiners ("{memory}\n" + "\n").
    slot = MEMORY + "\n" + "\n"
    plain_run, plain = await first_requests(tmp_path / "a", make, replies, None)
    mem_run, with_memory = await first_requests(tmp_path / "b", make, replies, MEMORY)
    blank_run, blank = await first_requests(tmp_path / "c", make, replies, " \n\t ")
    assert len(plain) == len(with_memory) == len(blank) >= 1
    for a, b, c in zip(plain, with_memory, blank):
        # Every call, whole body: one insertion, exactly the method's text in the slot.
        assert slot_insertions(a, b) == [slot]
        assert wire(a) == wire(c)  # a blank memory injects nothing
    # H10.2: the prompt without memory is identical across methods.
    assert (
        plain_run.trajectory.first_prompt_no_memory_sha256
        == mem_run.trajectory.first_prompt_no_memory_sha256
        == blank_run.trajectory.first_prompt_sha256
        == plain_run.trajectory.first_prompt_sha256
    )
    assert (
        mem_run.trajectory.first_prompt_sha256
        != plain_run.trajectory.first_prompt_sha256
    )
    # E10: the framework counts the injection (the method's own number is ignored).
    assert mem_run.result.injected_tokens == tokens.count_text(MEMORY) > 0
    assert mem_run.trajectory.injection is not None
    assert mem_run.trajectory.injection.tokens == tokens.count_text(MEMORY)
    assert blank_run.result.injected_tokens == 0 == plain_run.result.injected_tokens
    assert all(
        c.memory_tokens == tokens.count_text(MEMORY) for c in mem_run.trajectory.calls
    )
    assert mem_run.trajectory.injected_tokens_total == tokens.count_text(MEMORY) * len(
        with_memory
    )


# --- what methods receive (Q19, Q20, Q28, H10.1) -------------------------------------------


def labels_of(task: Task, env: Any) -> list[str]:
    """What H10.1 forbids for ``task``, from the environment's own scorer-side list."""
    return list(env.label_strings(task))


async def learning_run(tmp_path: Path) -> Any:
    env, task = alfworld_env(tmp_path), alfworld_task()
    replies = [text_reply(act("look"), reasoning="r1"), text_reply(act("win"))]
    instruction = "put a clean cloth in countertop."
    method = RecordingMethod(MEMORY, item_ids=["m1", "m2"])
    run, _fake = await play(
        tmp_path, env, task, replies, method=method, phase="learning"
    )
    return run, method, task, instruction, env


@pytest.mark.asyncio
async def test_learning_hooks_receive_only_raw_messages(tmp_path):
    run, method, task, instruction, env = await learning_run(tmp_path)
    trajectory = run.trajectory
    uid = task.uid(SALT)
    messages = trajectory.messages
    # H1.4 order: before_task, one on_message per message, after_task, wait_until_idle.
    assert method.hooks() == [
        "before_task",
        *["on_message"] * len(messages),
        "after_task",
        "wait_until_idle",
    ]
    task_view = method.events[0][1]
    assert (task_view.task_uid, task_view.instruction) == (uid, instruction)
    assert task_view.first_messages == trajectory.first_messages
    assert messages[: len(trajectory.first_messages)] == list(trajectory.first_messages)
    streamed = [payload for name, payload in method.events if name == "on_message"]
    assert [u for u, _ in streamed] == [uid] * len(messages)
    assert [m for _, m in streamed] == messages
    view = method.events[-2][1]
    assert view.task_uid == uid and list(view.messages) == messages
    assert "MUTATED BY METHOD" not in json.dumps(messages)  # methods get copies
    # Q20 shape: environment output and the model's raw output, nothing else.
    roles = [m["role"] for m in messages]
    assert "system" not in roles and roles[0] == "user"
    for message in messages:
        assert set(message) <= {"role", "content"}
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert [m["content"] for m in assistant] == [
        (c.response or {})["choices"][0]["message"]["content"] or ""
        for c in trajectory.calls
    ]
    # Memory text never reaches the method, nor the prompts actually sent; the template
    # they are rendered from only as the task view's agent context (the env's static
    # template, ALFWorld's). Reasoning stays in the record.
    assert task_view.agent_context == env.agent_context()
    assert "Your admissible actions" in task_view.agent_context
    assert "{current_observation}" in task_view.agent_context
    assert trajectory.agent_context == task_view.agent_context
    without_context = [
        replace(p, agent_context="") if isinstance(p, TaskView) else p
        for p in method.received()
    ]
    everything = json.dumps([repr(p) for p in without_context])
    assert "§MEM§" not in everything and "Your admissible actions" not in everything
    assert "expert" not in everything.lower()
    assert "§MEM§" not in task_view.agent_context
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert trajectory.steps[0].reasoning == "r1" and "r1" not in everything
    # H10.1: no label key anywhere, no label string in harness-authored text.
    assert (
        find_label_leaks(method.received(), forbidden_strings=labels_of(task, env))
        == []
    )
    assert task.raw_id not in json.dumps([uid, messages, instruction])


@pytest.mark.asyncio
async def test_test_phase_calls_only_before_task(tmp_path):
    run, method, task, _, _ = await learning_run(tmp_path / "learn")
    env: Any = alfworld_env(tmp_path)
    replies = [text_reply(act("look")), text_reply(act("win"))]
    frozen = RecordingMethod(MEMORY)
    test_run, fake = await play(
        tmp_path, env, task, replies, method=frozen, phase="test"
    )
    assert frozen.hooks() == ["before_task"]
    assert test_run.trajectory.messages == run.trajectory.messages
    assert {row["phase"] for row in ledger_calls(tmp_path)} == {"test"}
    assert all(c.phase == "test" for c in test_run.trajectory.calls)
    assert len(upstream_bodies(fake)) == len(test_run.trajectory.calls)


@pytest.mark.asyncio
async def test_opaque_uid_is_the_only_item_id_on_the_wire(tmp_path):
    env, task = alfworld_env(tmp_path), alfworld_task()
    run, _ = await play(
        tmp_path, env, task, [text_reply(act("look")), text_reply(act("win"))]
    )
    uid = task.uid(SALT)
    assert run.result.task_uid == uid == run.trajectory.task_uid
    assert run.result.raw_id == task.raw_id  # scorer side only
    ledger = ledger_calls(tmp_path)
    assert [row["item"] for row in ledger] == [uid, uid]
    assert task.raw_id not in json.dumps(ledger)


# --- records -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trajectory_record_has_every_field(tmp_path):
    env, task = alfworld_env(tmp_path), alfworld_task()
    run, _ = await play(
        tmp_path,
        env,
        task,
        [text_reply(act("look")), text_reply(act("win"))],
        method=RecordingMethod(MEMORY),
    )
    trajectory = run.trajectory
    data = trajectory.to_dict()
    json.dumps(data)
    assert data["result"]["score"]["primary"] == 1.0 and data["done"] is True
    assert data["end_reason"] == "done" and data["failure"] is None
    assert data["final_answer"] is None
    assert data["final_private"] == {"won": True}
    assert data["injection"]["text"] == MEMORY
    assert data["injection"]["item_ids"] == []
    assert data["injection_encoding"] in ("o200k_base", "fallback_chars_div_3")
    assert set(data["timings"]) >= {"before_task_s", "episode_s", "total_s"}
    assert len(data["calls"]) == 2 == len(data["steps"])
    for index, call in enumerate(trajectory.calls):
        assert call.index == index and call.http_status == 200
        assert call.gateway_call_id and call.response and call.usage
        assert len(call.request_sha256) == 64 and call.max_tokens_expected > 0
        assert call.memory_tokens == tokens.count_text(MEMORY)
        assert call.item == task.uid(SALT) and call.phase == "test"
    assert trajectory.steps[0].fields == {
        "action": "look",
        "format_error": False,
        "action_valid": True,
        "truncated": False,
    }
    assert trajectory.steps[0].private == {"won": False}
    assert [s.call_index for s in trajectory.steps] == [0, 1]
    ledger = {row["call_id"]: row for row in ledger_calls(tmp_path)}
    assert set(ledger) == {c.gateway_call_id for c in trajectory.calls}


# --- model failures are scored as SkillOpt scores them (F9, H8, H4) ----------------------------


@pytest.mark.asyncio
async def test_alfworld_actor_timeout_plays_skillopts_look_and_goes_on(tmp_path):
    """W2/W27: a single-attempt timeout is a MODEL failure; SkillOpt plays 'look' and continues."""
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    method = RecordingMethod()
    run, _ = await play(
        tmp_path,
        env,
        task,
        [text_reply(act("look")), Reply(delay_s=3.0), text_reply(act("win"))],
        method=method,
        phase="learning",
        deadline_s=1.0,
    )
    assert run.result.score is not None and run.result.score.primary == 1.0
    assert run.trajectory.end_reason == "done" and run.trajectory.done
    assert run.result.failure is None and run.result.model_failures == 1
    assert [f["code"] for f in run.trajectory.model_failures] == [MODEL_TIMEOUT]
    assert run.trajectory.model_failures[0]["call_index"] == 1
    assert run.trajectory.calls[1].http_status == 504
    timed_out = run.trajectory.steps[1]
    assert timed_out.fields["model_failure"] == MODEL_TIMEOUT
    assert (
        timed_out.fields["action"] == "look"
        and timed_out.fields["format_error"] is False
    )
    assert timed_out.assistant == {"role": "assistant", "content": ""}
    assert games.made[0].commands == ["look", "look", "win"]
    assert run.result.steps == 3 and run.result.format_errors == 0
    roles = [m["role"] for m in run.trajectory.messages]
    assert roles == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert method.hooks()[-2:] == ["after_task", "wait_until_idle"]
    assert games.made[0].closed


@pytest.mark.asyncio
async def test_alfworld_truncated_reply_is_parsed_as_skillopt_parses_any_reply(
    tmp_path,
):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    looping = "<think>" + "I should think more. " * 50  # cut off before any action tag
    kept = (
        act("go to cabinet 1") + " and then I keep writing"
    )  # action complete, then cut
    run, _ = await play(
        tmp_path,
        env,
        task,
        [
            text_reply(looping, finish_reason="length"),
            text_reply(kept, finish_reason="length"),
            text_reply(act("win")),
        ],
    )
    assert run.result.score is not None and run.result.score.primary == 1.0
    assert games.made[0].commands == ["look", "go to cabinet 1", "win"]
    assert [s.fields.get("model_failure") for s in run.trajectory.steps] == [
        TRUNCATED,
        TRUNCATED,
        None,
    ]
    assert [s.fields["format_error"] for s in run.trajectory.steps] == [
        True,
        False,
        False,
    ]
    assert run.result.model_failures == 2 and run.result.format_errors == 1
    # Q20: the method-visible assistant turn is the model's raw output, as sent.
    assert run.trajectory.messages[1]["content"] == looping


@pytest.mark.asyncio
async def test_first_call_over_budget_is_still_a_harness_failure(tmp_path, monkeypatch):
    env = alfworld_env(tmp_path)
    monkeypatch.setattr(tokens, "count_text", lambda text: 10**7)
    with pytest.raises(RolloutError) as caught:
        await play(tmp_path, env, alfworld_task(), [])
    assert caught.value.failure.kind is FailureKind.HARNESS
    assert caught.value.failure.code == GATEWAY_REJECTED


@pytest.mark.asyncio
async def test_learning_phase_requires_the_q18_idle_wait(tmp_path):
    for bad in (None, 0.0):
        with pytest.raises(ValueError, match="idle_timeout_s"):
            await run_episode(
                alfworld_env(tmp_path),
                alfworld_task(),
                method=NoMemory(),
                actor=None,  # type: ignore[arg-type]
                salt=SALT,
                phase="learning",
                idle_timeout_s=bad,
            )


# --- infrastructure and harness failures abort the item (H8) -----------------------------------


@pytest.mark.asyncio
async def test_exhausted_gateway_retries_raise_an_unscored_infra_failure(tmp_path):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    method = RecordingMethod()
    with pytest.raises(RolloutError) as caught:
        await play(
            tmp_path,
            env,
            task,
            [text_reply(act("look")), Reply(status=503), Reply(status=502)],
            method=method,
            phase="learning",
            max_retries=1,
        )
    error = caught.value
    assert error.stage == "actor"
    assert (error.failure.kind, error.failure.code) == (
        FailureKind.INFRA,
        UPSTREAM_INFRA,
    )
    assert error.result.score is None and not error.result.scored
    assert error.result.failure == error.failure.to_dict()
    assert error.trajectory.failure_stage == "actor"
    assert (
        len(error.trajectory.calls) == 2
        and error.trajectory.calls[1].http_status == 502  # the last attempt
    )
    assert "after_task" not in method.hooks()
    assert games.made[0].closed


@pytest.mark.asyncio
async def test_unreachable_gateway_is_infra(tmp_path):
    async def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    async with Actor(
        gateway_base_url="http://gw",
        run_id="cell-a1",
        model=protocol.MODELS["luna"],
        transport=httpx.MockTransport(refuse),
    ) as actor:
        with pytest.raises(RolloutError) as caught:
            await run_episode(
                env, task, method=NoMemory(), actor=actor, salt=SALT, phase="test"
            )
    assert caught.value.failure.code == GATEWAY_UNREACHABLE
    assert caught.value.failure.kind is FailureKind.INFRA
    assert games.made[0].closed


class _Hang(RecordingMethod):
    async def wait_until_idle(self, timeout_s: float) -> None:
        self.events.append(("wait_until_idle", timeout_s))
        await asyncio.sleep(3600)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "stage", "kind", "code"),
    [
        (
            RecordingMethod(fail={"before_task": RuntimeError("bug")}),
            "before_task",
            FailureKind.HARNESS,
            UNEXPECTED_EXCEPTION,
        ),
        (
            RecordingMethod(inject=lambda view: "not an Injection"),
            "before_task",
            FailureKind.HARNESS,
            UNEXPECTED_EXCEPTION,
        ),
        (
            RecordingMethod(fail={"on_message": MirixServerError("server died")}),
            "on_message",
            FailureKind.INFRA,
            MIRIX_SERVER,
        ),
        (
            RecordingMethod(fail={"after_task": MirixServerError("server died")}),
            "after_task",
            FailureKind.INFRA,
            MIRIX_SERVER,
        ),
        (_Hang(), "wait_until_idle", FailureKind.INFRA, IDLE_WAIT_TIMEOUT),
    ],
)
async def test_method_failures_abort_the_item(
    tmp_path, monkeypatch, method, stage, kind, code
):
    monkeypatch.setattr(ro, "IDLE_WAIT_GRACE_S", 0.05)
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    fake = FakeUpstream(
        responder=Script(alfworld=[text_reply(act("look")), text_reply(act("win"))])
    )
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        with pytest.raises(RolloutError) as caught:
            await run_episode(
                env,
                task,
                method=method,
                actor=actor,
                salt=SALT,
                phase="learning",
                idle_timeout_s=0.05,
            )
    error = caught.value
    assert (error.stage, error.failure.kind, error.failure.code) == (stage, kind, code)
    assert games.made[0].closed
    if stage in ("after_task", "wait_until_idle"):
        # The episode itself finished: its scored result is kept.
        assert error.result.score is not None and error.result.score.primary == 1.0
    else:
        assert error.result.score is None


@pytest.mark.asyncio
async def test_environment_failures_are_classified(tmp_path):
    # An expert-enabled game leaks info keys: the environment refuses it at reset.
    games = Games()
    env = alfworld_env(tmp_path / "a", games, extra_info={"facts": ["(on x y)"]})
    with pytest.raises(RolloutError) as caught:
        await play(tmp_path / "a", env, alfworld_task(), [text_reply(act("look"))])
    assert caught.value.stage == "reset"
    assert caught.value.failure.kind is FailureKind.HARNESS
    assert caught.value.trajectory.calls == [] and games.made[0].closed


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_the_episode(tmp_path):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    fake = FakeUpstream(responder=Script(alfworld=[Reply(delay_s=30.0)]))
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        job = asyncio.create_task(
            run_episode(
                env, task, method=NoMemory(), actor=actor, salt=SALT, phase="test"
            )
        )
        while not fake.calls("chat/completions"):
            await asyncio.sleep(0.01)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
    assert games.made[0].closed


@pytest.mark.asyncio
async def test_invalid_phase_is_refused(tmp_path):
    with pytest.raises(ValueError):
        await run_episode(
            alfworld_env(tmp_path),
            alfworld_task(),
            method=NoMemory(),
            actor=None,  # type: ignore[arg-type]
            salt=SALT,
            phase="training",  # type: ignore[arg-type]
        )


# --- episode limits and broken environments -------------------------------------------------------


@pytest.mark.asyncio
async def test_alfworld_step_limit_ends_the_loop(tmp_path):
    games = Games()
    env = alf.AlfWorldEnv(
        data_root=tmp_path, backend_factory=games.factory(), max_steps=3
    )
    run, fake = await play(
        tmp_path, env, alfworld_task(), [text_reply(act("look"))] * 3
    )
    assert len(upstream_bodies(fake)) == 3 == run.result.steps
    assert run.trajectory.done and run.trajectory.steps[-1].fields["truncated"] is True
    assert run.result.score is not None and run.result.score.primary == 0.0
    assert run.result.failure is None


class _StubEpisode:
    def __init__(self, step_messages: tuple[dict[str, Any], ...], done: bool) -> None:
        self.step_messages, self.done_flag, self.closed = step_messages, done, False

    async def reset(self) -> Any:
        from evals.skill_bench.envs.base import Reset

        return Reset("q?", ({"role": "user", "content": "q?"},), {"question": "q?"})

    async def step(self, action: Any) -> Any:
        from evals.skill_bench.envs.base import StepResult

        return StepResult(self.step_messages, self.done_flag)

    async def close(self) -> None:
        self.closed = True


class _StubEnv:
    """An ALFWorld-shaped environment (one step) that misbehaves in a chosen way."""

    benchmark = "alfworld"
    max_steps = 1

    def __init__(self, episode: _StubEpisode) -> None:
        self.stub = episode

    def load_tasks(self, split: str) -> list[Task]:
        return []

    def episode(self, task: Task) -> _StubEpisode:
        return self.stub

    def score(self, task: Task, final_answer: Any, private: Any) -> Any:
        raise AssertionError("never scored")

    def actor_messages(self, fields: Any, memory: str = "") -> list[dict[str, Any]]:
        return [{"role": "user", "content": fields["question"]}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "done"),
    [
        ((), False),  # never finishes within its max_steps
        (({"role": "system", "content": "template text"},), True),  # not a Q20 message
    ],
)
async def test_misbehaving_environment_is_a_harness_failure(tmp_path, messages, done):
    stub = _StubEpisode(messages, done)
    task = Task("alfworld", "test", "q1")
    fake = FakeUpstream(
        responder=lambda endpoint, body: text_reply("<answer>a</answer>")
    )
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        with pytest.raises(RolloutError) as caught:
            await run_episode(
                _StubEnv(stub),  # type: ignore[arg-type]
                task,
                method=NoMemory(),
                actor=actor,
                salt=SALT,
                phase="test",
            )
    assert caught.value.stage == "env_step"
    assert caught.value.failure.kind is FailureKind.HARNESS
    assert caught.value.result.score is None and stub.closed


# --- concurrency (H1.6) -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_items_run_concurrently_on_one_actor(tmp_path):
    env = alfworld_env(tmp_path)
    tasks = env.load_tasks("train")[:6]
    fake = FakeUpstream(responder=lambda endpoint, body: text_reply(act("win")))
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        runs = await asyncio.gather(
            *(
                run_episode(
                    env, t, method=NoMemory(), actor=actor, salt=SALT, phase="test"
                )
                for t in tasks
            )
        )
    assert [r.result.score.primary for r in runs if r.result.score] == [1.0] * 6
    assert sorted(row["item"] for row in ledger_calls(tmp_path)) == sorted(
        t.uid(SALT) for t in tasks
    )


# --- offline end-to-end -------------------------------------------------------------------------


@requires_alfworld_data
@pytest.mark.asyncio
async def test_offline_end_to_end_none_method(tmp_path):
    """One real ALFWorld game with the no-memory method, through the real gateway over the
    fake upstream."""
    alf_env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA)
    alf_task = alf_env.load_tasks("val")[0]  # wins with WINNING_RESPONSES
    script = Script(alfworld=[text_reply(r) for r in WINNING_RESPONSES])
    fake = FakeUpstream(responder=script)
    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        alf_run = await run_episode(
            alf_env, alf_task, method=NoMemory(), actor=actor, salt=SALT, phase="test"
        )
    assert alf_run.result.score is not None and alf_run.result.score.primary == 1.0
    assert alf_run.result.failure is None
    assert alf_run.result.steps == 5 and alf_run.result.format_errors == 1
    assert len(alf_run.trajectory.messages) == 1 + 2 * 5
    assert alf_run.trajectory.instruction == "look at alarmclock under the desklamp."
    assert [s.fields["action"] for s in alf_run.trajectory.steps] == [
        "look",
        "go to sidetable 1",
        "take alarmclock 3 from sidetable 1",
        "go to sidetable 2",
        "use desklamp 2",
    ]
    assert alf_run.trajectory.final_private == {"won": True}
    # Every call went through the gateway's actor role with the opaque item id.
    ledger = ledger_calls(tmp_path)
    assert len(ledger) == 5 == len(alf_run.trajectory.calls)
    assert {row["role"] for row in ledger} == {"actor"}
    assert {row["status"] for row in ledger} == {"ok"}
    assert {row["item"] for row in ledger} == {alf_run.trajectory.task_uid}
    assert {row["call_id"] for row in ledger} == {
        c.gateway_call_id for c in alf_run.trajectory.calls
    }
    # No memory: the first prompt is the template with an empty slot.
    assert (
        alf_run.trajectory.first_prompt_sha256
        == alf_run.trajectory.first_prompt_no_memory_sha256
    )
    assert alf_run.result.injected_tokens == 0
    # The record round-trips through disk.
    blobs = tmp_path / "blobs"
    path = tmp_path / "records" / f"{alf_run.trajectory.task_uid}.json"
    write_trajectory(path, alf_run.trajectory, blob_dir=blobs)
    assert read_trajectory(path, blob_dir=blobs) == alf_run.trajectory.to_dict()


# --- on_played and finish_learning (learning checkpoints) -----------------------------------


@pytest.mark.asyncio
async def test_on_played_runs_after_the_score_and_before_after_task(tmp_path):
    """The caller's hook sees the scored episode after the last on_message and before
    after_task (run.py records the item and takes its played checkpoint there)."""
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    fake = FakeUpstream(
        responder=Script(alfworld=[text_reply(act("look")), text_reply(act("win"))])
    )
    method = RecordingMethod()
    seen: list[Any] = []

    async def on_played(run: EpisodeRun) -> None:
        seen.append((list(method.hooks()), run.result.score, run.trajectory.task_uid))

    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        run = await run_episode(
            env,
            task,
            method=method,
            actor=actor,
            salt=SALT,
            phase="learning",
            idle_timeout_s=5.0,
            on_played=on_played,
        )
    assert len(seen) == 1
    hooks, score, uid = seen[0]
    assert hooks[0] == "before_task" and hooks[-1] == "on_message"
    assert "after_task" not in hooks
    assert score is not None and score.primary == 1.0
    assert uid == run.trajectory.task_uid
    assert method.hooks()[-2:] == ["after_task", "wait_until_idle"]
    assert run.trajectory.timings["played_s"] <= run.trajectory.timings["total_s"]


@pytest.mark.asyncio
async def test_on_played_is_never_called_in_the_test_phase(tmp_path):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    fake = FakeUpstream(
        responder=Script(alfworld=[text_reply(act("look")), text_reply(act("win"))])
    )
    called: list[Any] = []

    async def on_played(run: EpisodeRun) -> None:
        called.append(run)

    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        await run_episode(
            env,
            task,
            method=NoMemory(),
            actor=actor,
            salt=SALT,
            phase="test",
            on_played=on_played,
        )
    assert called == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "kind", "code"),
    [
        (httpx.ConnectError("gateway gone"), FailureKind.INFRA, GATEWAY_UNREACHABLE),
        (RuntimeError("a bug"), FailureKind.HARNESS, UNEXPECTED_EXCEPTION),
    ],
)
async def test_an_on_played_failure_is_the_played_stage_and_skips_after_task(
    tmp_path, error, kind, code
):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    fake = FakeUpstream(
        responder=Script(alfworld=[text_reply(act("look")), text_reply(act("win"))])
    )
    method = RecordingMethod()

    async def on_played(run: EpisodeRun) -> None:
        raise error

    async with gateway_actor(tmp_path, fake) as (_gateway, actor):
        with pytest.raises(RolloutError) as caught:
            await run_episode(
                env,
                task,
                method=method,
                actor=actor,
                salt=SALT,
                phase="learning",
                idle_timeout_s=5.0,
                on_played=on_played,
            )
    failure = caught.value
    assert (failure.stage, failure.failure.kind, failure.failure.code) == (
        ro.STAGE_PLAYED,
        kind,
        code,
    )
    assert failure.__cause__ is error
    assert "after_task" not in method.hooks()
    assert failure.result.score is not None  # played and scored before the hook
    assert games.made[0].closed


@pytest.mark.asyncio
async def test_finish_learning_runs_the_learning_hooks_of_a_played_item(
    tmp_path, monkeypatch
):
    """A resume from an item's played checkpoint: after_task with the rebuilt view, then
    the Q18 wait, classified by stage like in run_episode."""
    monkeypatch.setattr(ro, "IDLE_WAIT_GRACE_S", 0.05)
    view = ro.TrajectoryView(
        task_uid="t0", instruction="do it", first_messages=(), messages=()
    )
    method = RecordingMethod()
    timings = await ro.finish_learning(method, view, idle_timeout_s=1.0)
    assert method.hooks() == ["after_task", "wait_until_idle"]
    assert method.events[0][1] is view
    assert set(timings) == {"after_task_s", "idle_wait_s"}
    failing = RecordingMethod(fail={"after_task": httpx.ConnectError("down")})
    with pytest.raises(ro.LearningHookError) as caught:
        await ro.finish_learning(failing, view, idle_timeout_s=1.0)
    assert caught.value.stage == "after_task"
    assert caught.value.failure.kind is FailureKind.INFRA
    with pytest.raises(ro.LearningHookError) as caught:
        await ro.finish_learning(_Hang(), view, idle_timeout_s=0.05)
    assert (caught.value.stage, caught.value.failure.code) == (
        "wait_until_idle",
        IDLE_WAIT_TIMEOUT,
    )
    with pytest.raises(ValueError):
        await ro.finish_learning(method, view, idle_timeout_s=0)


def _view_round_trip(tmp_path: Path, run: EpisodeRun) -> Any:
    """Write the trajectory as run.py does (with its blob store), rebuild the view
    from the record, and require it byte-identical (canonical JSON) to the original."""
    from evals.skill_bench.agent.trajectory import (
        canonical_json,
        trajectory_view_from_record,
    )

    blobs = tmp_path / "blobs"
    path = tmp_path / "trajectory.json"
    write_trajectory(path, run.trajectory, blob_dir=blobs)
    record = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = trajectory_view_from_record(record, blobs)
    assert canonical_json(rebuilt) == canonical_json(run.trajectory.view())
    assert isinstance(rebuilt.messages, tuple)
    return record


@pytest.mark.asyncio
async def test_a_played_view_is_rebuilt_byte_for_byte_alfworld(tmp_path):
    """ALFWorld, with an empty assistant turn (SkillOpt plays "look" for it)."""
    env = alfworld_env(tmp_path, Games())
    run, _fake = await play(
        tmp_path,
        env,
        alfworld_task(),
        [text_reply(""), text_reply(act("look")), text_reply(act("win"))],
        method=RecordingMethod(),
        phase="learning",
    )
    record = _view_round_trip(tmp_path, run)
    assert any(
        m.get("role") == "assistant" and not m.get("content")
        for m in run.trajectory.view().messages
    ), record["messages"]


# --- provider refusals the caller tolerates end the item unanswered ---------------------------

PAYLOAD_413 = Reply(
    413,
    {"error": {"code": 413, "message": "Request payload too large"}},
)
NO_TOOL_ENDPOINT = Reply(
    404,
    {
        "error": {
            "code": 404,
            "message": "No endpoints found that support tool use. Try disabling tools.",
        }
    },
)


@pytest.mark.asyncio
async def test_a_learning_actor_refusal_ends_the_item_unanswered(tmp_path):
    from evals.skill_bench.infra.failures import PROVIDER_UNSERVED, REFUSAL_CLASSES

    env, task = alfworld_env(tmp_path), alfworld_task()
    method = RecordingMethod()
    run, fake = await play(
        tmp_path,
        env,
        task,
        [PAYLOAD_413],
        method=method,
        phase="learning",
        unserved=REFUSAL_CLASSES,
    )
    score = run.result.score
    assert score is not None and score.primary == 0.0 and not score.answered
    failure = run.result.failure
    assert failure is not None
    assert (failure["kind"], failure["code"], failure["role"], failure["status"]) == (
        "harness",
        PROVIDER_UNSERVED,
        "actor",
        413,
    )
    assert "payload too large" in failure["detail"]
    assert run.trajectory.end_reason == ro.END_PROVIDER_UNSERVED == "provider_unserved"
    assert run.trajectory.failure == failure
    assert run.trajectory.failure_stage == ro.STAGE_ACTOR
    assert run.trajectory.model_failures == [] and run.result.model_failures == 0
    # The refused call is recorded (H10.2 rebuilds it), exactly once: no retry, no fallback.
    assert [c.http_status for c in run.trajectory.calls] == [413]
    assert len(upstream_bodies(fake)) == 1
    assert run.result.final_answer is None and run.trajectory.steps == []
    # The method sees the unanswered trajectory, as for a model's no-reply.
    assert method.hooks().count("after_task") == 1
    assert method.hooks()[-2:] == ["after_task", "wait_until_idle"]
    assert [m["role"] for m in run.trajectory.messages] == ["user"]


@pytest.mark.asyncio
async def test_a_test_item_is_scored_unanswered_only_when_run_py_says_so(tmp_path):
    from evals.skill_bench.infra.failures import PROVIDER_UNSERVED

    env, task = alfworld_env(tmp_path), alfworld_task()
    # By default (the item's first try) the refusal raises: the item stays pending.
    with pytest.raises(RolloutError) as caught:
        await play(tmp_path / "first", env, task, [NO_TOOL_ENDPOINT])
    assert (caught.value.failure.code, caught.value.failure.status) == (
        UPSTREAM_REJECTED,
        404,
    )
    assert caught.value.result.score is None
    # The try after a no-endpoint refusal: scored unanswered and flagged.
    run, _ = await play(
        tmp_path / "second", env, task, [NO_TOOL_ENDPOINT], unserved=("no_endpoint",)
    )
    assert run.result.score is not None and run.result.score.primary == 0.0
    assert not run.result.score.answered
    assert run.result.failure is not None
    assert (run.result.failure["code"], run.result.failure["status"]) == (
        PROVIDER_UNSERVED,
        404,
    )
    assert run.trajectory.end_reason == ro.END_PROVIDER_UNSERVED
    # A payload refusal is not a class the test phase tolerates: it still raises.
    with pytest.raises(RolloutError) as caught:
        await play(
            tmp_path / "payload",
            env,
            task,
            [PAYLOAD_413],
            unserved=("no_endpoint",),
        )
    assert (caught.value.failure.code, caught.value.failure.status) == (
        UPSTREAM_REJECTED,
        413,
    )


@pytest.mark.asyncio
async def test_an_account_refusal_and_other_errors_are_never_tolerated(tmp_path):
    from evals.skill_bench.infra.failures import REFUSAL_CLASSES

    env, task = alfworld_env(tmp_path), alfworld_task()
    policy = Reply(
        404,
        {
            "error": {
                "code": 404,
                "message": "No endpoints found matching your data policy "
                "(Free model publication).",
            }
        },
    )
    plain = Reply(403, {"error": {"code": 403, "message": "Forbidden"}})
    for n, reply in enumerate((policy, plain)):
        with pytest.raises(RolloutError) as caught:
            await play(
                tmp_path / str(n),
                env,
                task,
                [reply],
                method=RecordingMethod(),
                phase="learning",
                unserved=REFUSAL_CLASSES,
            )
        assert caught.value.failure.code == UPSTREAM_REJECTED
        assert caught.value.result.score is None


@pytest.mark.asyncio
async def test_an_alfworld_refusal_ends_the_game_without_the_look_fallback(tmp_path):
    games = Games()
    env, task = alfworld_env(tmp_path, games), alfworld_task()
    run, fake = await play(
        tmp_path,
        env,
        task,
        [text_reply(act("look")), NO_TOOL_ENDPOINT],
        unserved=("no_endpoint",),
    )
    assert run.result.score is not None and run.result.score.primary == 0.0
    # Scored as unanswered (S1), not as a game the actor lost.
    assert run.result.score.answered is False
    assert run.trajectory.end_reason == ro.END_PROVIDER_UNSERVED
    assert games.made[0].commands == ["look"]  # no 'look' fallback for the refused call
    assert len(upstream_bodies(fake)) == 2 and len(run.trajectory.steps) == 1
    assert games.made[0].closed


@pytest.mark.asyncio
async def test_unserved_must_be_refusal_classes(tmp_path):
    env, task = alfworld_env(tmp_path), alfworld_task()
    actor = Actor(
        gateway_base_url="http://gateway", run_id="r", model=protocol.MODELS["luna"]
    )
    async with actor:
        for bad in (("gateway",), ("payload", "x"), "payload"):
            with pytest.raises(ValueError, match="unserved"):
                await run_episode(
                    env,
                    task,
                    method=NoMemory(),
                    actor=actor,
                    salt=SALT,
                    phase="test",
                    unserved=bad,  # type: ignore[arg-type]
                )
