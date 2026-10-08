"""The memory-method contract, reusable by every method's tests.

``methods/base.py`` states the contract; this module checks it. The method tests call
it with their own method and cell (a gateway over the fake upstream where the
method calls models), and ``test_method_contract.py`` refuses a method implemented in
``methods/`` that is not registered there (:func:`unregistered_methods`).

* :func:`contract_items` builds Q20 trajectories that carry every message shape a method
  must accept: the **empty assistant turn** ALFWorld emits for a step whose model call got
  no reply, a reply without an action tag and WebShop's score line.
* :func:`exercise_method` drives a method through a whole cell as ``run.py`` will:
  ``setup`` -> per learning item ``before_task``, ``on_message`` per message,
  ``after_task``, ``wait_until_idle`` -> ``finalize``, ``wait_until_idle``,
  ``snapshot`` -> concurrent test-phase ``before_task`` with the snapshot unchanged
  (H10.4) -> a fresh instance set up in the test phase resumes the same snapshot (H8).
* :func:`play_alfworld_no_reply_step` runs the real rollout, with a model call that times
  out, so the method receives the empty assistant turn exactly as production sends it.
* :func:`exercise_checkpoint_resume` checks contract item 6 (learning checkpoints): a
  method checkpointed after an item (``done``) or between its
  last ``on_message`` and ``after_task`` (``played``), then "killed" after doing more,
  restored from the checkpoint's copy of ``work_dir`` and state (``setup`` with
  ``cell.resume``) rebuilds the recorded snapshot, and continuing with the same model
  replies freezes the memory an uninterrupted run freezes; files under ``objects/`` are
  never modified once written.
* :func:`contract_cell` builds a :class:`CellContext` for a test.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.agent.trajectory import find_label_leaks, visible_message
from evals.skill_bench.envs.base import Message
from evals.skill_bench.methods.base import (
    GATEWAY_LOCAL_TOKEN,
    OBJECTS_DIR,
    CellContext,
    CellPhase,
    Injection,
    LearningResume,
    MemoryMethod,
    TaskView,
    TrajectoryView,
    checkpoint_state_json,
    snapshot_sha256,
)

LUNA = protocol.MODELS["luna"]
#: The turn ALFWorld's rollout emits for a step whose model call got no reply
#: (``agent/rollout.py:_AlfWorldDriver.without_reply``).
EMPTY_ASSISTANT: Message = {"role": "assistant", "content": ""}
#: Extra time on top of a method's own idle timeout (as ``rollout.IDLE_WAIT_GRACE_S``).
IDLE_GRACE_S = 30.0

MethodFactory = Callable[[], MemoryMethod]


def contract_cell(
    work_dir: Path,
    *,
    benchmark: str,
    method: str,
    gateway_base_url: str = "http://127.0.0.1:9",
    model: str = LUNA,
    phase: CellPhase = "learning",
    run_id: str | None = None,
    attempt: int = 1,
    **options: Any,
) -> CellContext:
    """A cell attempt for tests: ``work_dir`` is the method's directory (and the cell dir)."""
    return CellContext(
        cell_id=f"{benchmark}-test-{method}",
        attempt=attempt,
        benchmark=benchmark,
        method=method,
        model=model,
        gateway_base_url=gateway_base_url,
        run_id=run_id or f"{benchmark}-test-{method}-a{attempt}".replace("_", "-"),
        gateway_token=GATEWAY_LOCAL_TOKEN,
        work_dir=work_dir,
        phase=phase,
        options=options,
    )


# ---------------------------------------------------------------------------
# Q20 trajectories with every message shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractItem:
    """One learning item: the view ``before_task`` gets and the whole trajectory."""

    view: TaskView
    messages: tuple[Message, ...]

    def trajectory(self) -> TrajectoryView:
        return TrajectoryView(
            task_uid=self.view.task_uid,
            instruction=self.view.instruction,
            first_messages=copy.deepcopy(self.view.first_messages),
            messages=copy.deepcopy(self.messages),
        )


def _item(
    uid: str, instruction: str, messages: Sequence[Message], first: int = 1
) -> ContractItem:
    shaped = tuple(visible_message(m) for m in messages)
    assert shaped == tuple(messages), "contract messages must already be Q20-shaped"
    return ContractItem(TaskView(uid, instruction, shaped[:first]), shaped)


def contract_items(benchmark: str) -> list[ContractItem]:
    """Two or three learning items for ``benchmark`` whose trajectories cover every shape."""
    if benchmark == "alfworld":
        first = (
            "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
            "Looking quickly around you, you see a cabinet 1 and a countertop 1.\n\n"
            "Your task is to: put a mug in cabinet 1.\n\nAdmissible actions:\n"
            "'go to cabinet 1'\n 'go to countertop 1'\n 'look'"
        )
        obs = "You arrive at countertop 1. On the countertop 1, you see a mug 1.\n\nAdmissible actions:\n'take mug 1 from countertop 1'\n 'look'"
        return [
            _item(
                "t" + "a" * 20,
                "put a mug in cabinet 1.",
                [
                    {"role": "user", "content": first},
                    {
                        "role": "assistant",
                        "content": "<think>The mug may be on the countertop.</think><action>go to countertop 1</action>",
                    },
                    {"role": "user", "content": obs},
                    EMPTY_ASSISTANT,  # the model call got no reply; the env played 'look'
                    {"role": "user", "content": obs},
                    {"role": "assistant", "content": "I would take the mug."},  # no tag
                    {"role": "user", "content": obs},
                    {
                        "role": "assistant",
                        "content": "<think>…</think><action>take mug 1 from countertop 1</action>",
                    },
                    {
                        "role": "user",
                        "content": "You pick up the mug 1 from the countertop 1.",
                    },
                ],
            ),
            _item(
                "t" + "b" * 20,
                "put a mug in cabinet 1.",
                [
                    {"role": "user", "content": first},
                    EMPTY_ASSISTANT,  # an episode may even start with a no-reply step
                    {"role": "user", "content": first},
                ],
            ),
        ]
    if benchmark == "webshop":
        return _webshop_items()
    raise ValueError(f"unknown benchmark {benchmark!r}")


def _webshop_items() -> list[ContractItem]:
    """WebShop's Q20 shapes, rendered with the frozen templates as the
    environment renders them: the first page raw, with the instruction; later pages as
    upstream's text mode serves them (each starting with the instruction) through
    ``format_observation`` (the instruction cut, each part quoted); the empty assistant
    turn of a call without a reply, a reply without the action tag, and the score line
    after Buy Now."""
    from evals.skill_bench.envs import webshop as ws

    t = ws.WebShopTemplates.load()
    instruction = (
        "i need a long clip-in hair extension which is natural looking, and price "
        "lower than 40.00 dollars"
    )
    header = f"WebShop [SEP] Instruction: [SEP] {instruction}"
    search_page = f"{header} [SEP] Search"
    search_actions = ws.list_available_actions(
        t, {"has_search_bar": True, "clickables": ["search"]}
    )
    results = ws.format_observation(
        t,
        f"{header} [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > "
        "[SEP] B09QQLDJ93 [SEP] Clip-in Hair Extensions, 22 inch, Natural Black [SEP] "
        "$25.99",
        instruction,
    )
    result_actions = ws.list_available_actions(
        t,
        {
            "has_search_bar": False,
            "clickables": ["back to search", "next >", "b09qqldj93"],
        },
    )
    item_page = ws.format_observation(
        t,
        f"{header} [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] natural black "
        "[SEP] Clip-in Hair Extensions, 22 inch [SEP] Price: $25.99 [SEP] Buy Now",
        instruction,
    )
    item_actions = ws.list_available_actions(
        t,
        {
            "has_search_bar": False,
            "clickables": ["back to search", "< prev", "natural black", "buy now"],
        },
    )
    first = ws.observation_message(t, search_page, search_actions)
    on_results = ws.observation_message(t, results, result_actions)
    return [
        _item(
            "t" + "g" * 20,
            instruction,
            [
                first,
                {
                    "role": "assistant",
                    "content": "<think>Search first.</think>"
                    "<action>search[clip-in hair extension natural]</action>",
                },
                on_results,
                EMPTY_ASSISTANT,  # the model call got no reply: the frozen no-op
                on_results,
                {"role": "assistant", "content": "I would open the first result."},
                on_results,  # no action tag: a format error, the page stays
                {
                    "role": "assistant",
                    "content": "<think>…</think><action>click[b09qqldj93]</action>",
                },
                ws.observation_message(t, item_page, item_actions),
                {
                    "role": "assistant",
                    "content": "<think>It fits.</think><action>click[buy now]</action>",
                },
                ws.final_message(t, 0.75),  # D11: the score line only
            ],
        ),
        _item(
            "t" + "h" * 20,
            instruction,
            [
                first,
                EMPTY_ASSISTANT,
                ws.observation_message(
                    t,
                    ws.format_observation(t, search_page, instruction),
                    search_actions,
                ),
                ws.cap_message(t),  # the step cap's 0.00 line
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# The whole-cell exercise
# ---------------------------------------------------------------------------


@dataclass
class ContractReport:
    """What :func:`exercise_method` observed (for method-specific assertions)."""

    learning_injections: list[Injection] = field(default_factory=list)
    test_injections: list[Injection] = field(default_factory=list)
    snapshot: dict[str, Any] = field(default_factory=dict)
    resumed_snapshot: dict[str, Any] | None = None


async def _timed(awaitable: Awaitable[Any], timeout_s: float, what: str) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout_s)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"{what} did not finish within {timeout_s:.0f} s"
        ) from None


async def exercise_method(
    factory: MethodFactory,
    cell: CellContext,
    *,
    items: Sequence[ContractItem] | None = None,
    idle_timeout_s: float = 30.0,
    hook_timeout_s: float = 120.0,
    check_resume: bool = True,
    forbidden_strings: Sequence[str] = (),
) -> ContractReport:
    """Run ``factory()`` through a whole cell attempt and check the H6 contract.

    ``cell`` must be a learning-phase context. ``forbidden_strings`` (gold answers, raw
    ids ...) must never appear in an injection. ``check_resume`` sets up a second instance
    in the test phase with the same ``work_dir`` and requires the same snapshot (H8).
    """
    assert cell.phase == "learning", "exercise_method starts with the learning phase"
    items = list(items if items is not None else contract_items(cell.benchmark))
    report = ContractReport()
    method = factory()
    assert isinstance(method, MemoryMethod), type(method)
    assert method.signal_level in ("none", "S0", "S1")
    try:
        await _timed(method.setup(cell), hook_timeout_s, "setup")
        for item in items:
            injection = await _timed(
                method.before_task(copy.deepcopy(item.view)),
                hook_timeout_s,
                "before_task",
            )
            _check_injection(injection, forbidden_strings)
            report.learning_injections.append(injection)
            for message in item.messages:
                await _timed(
                    method.on_message(item.view.task_uid, copy.deepcopy(message)),
                    hook_timeout_s,
                    f"on_message({message.get('role')})",
                )
            await _timed(
                method.after_task(item.trajectory()), hook_timeout_s, "after_task"
            )
            await _timed(
                method.wait_until_idle(idle_timeout_s),
                idle_timeout_s + IDLE_GRACE_S,
                "wait_until_idle",
            )
        await _timed(method.finalize(), hook_timeout_s, "finalize")
        await _timed(
            method.wait_until_idle(idle_timeout_s),
            idle_timeout_s + IDLE_GRACE_S,
            "wait_until_idle after finalize",
        )
        report.snapshot = await _timed(method.snapshot(), hook_timeout_s, "snapshot")
        _check_snapshot(report.snapshot)
        # Test phase: before_task only, concurrently, memory frozen (H1.6, H10.4).
        tests = await asyncio.gather(
            *(
                _timed(
                    method.before_task(copy.deepcopy(item.view)),
                    hook_timeout_s,
                    "test before_task",
                )
                for item in items
            )
        )
        for injection in tests:
            _check_injection(injection, forbidden_strings)
        report.test_injections = list(tests)
        after = await _timed(
            method.snapshot(), hook_timeout_s, "snapshot after the test phase"
        )
        assert after == report.snapshot, (
            "the test phase changed the frozen memory (H10.4)"
        )
    finally:
        await _close(method)
    if check_resume:
        resumed = factory()
        try:
            await _timed(
                resumed.setup(replace(cell, phase="test")),
                hook_timeout_s,
                "resume setup",
            )
            report.resumed_snapshot = await _timed(
                resumed.snapshot(), hook_timeout_s, "resume snapshot"
            )
            assert report.resumed_snapshot == report.snapshot, (
                "a test-phase restart did not resume the frozen memory (H8)"
            )
            again = await _timed(
                resumed.before_task(copy.deepcopy(items[0].view)),
                hook_timeout_s,
                "resume before_task",
            )
            _check_injection(again, forbidden_strings)
        finally:
            await _close(resumed)
    return report


def _check_injection(injection: Any, forbidden_strings: Sequence[str]) -> None:
    assert isinstance(injection, Injection), (
        f"before_task returned {type(injection).__name__}"
    )
    leaks = find_label_leaks(
        {"role": "user", "content": injection.text}, forbidden_strings=forbidden_strings
    )
    assert leaks == [], leaks


def _check_snapshot(snapshot: Any) -> None:
    assert isinstance(snapshot, dict), f"snapshot() returned {type(snapshot).__name__}"
    json.dumps(snapshot)  # JSON-serialisable (run.py stores it)


async def _close(method: Any) -> None:
    close = getattr(method, "close", None)
    if callable(close):
        await close()


# ---------------------------------------------------------------------------
# Contract item 6: learning checkpoints
# ---------------------------------------------------------------------------


@dataclass
class CheckpointHooks:
    """What a method keeps outside ``work_dir`` (MIRIX: its cell database), for
    :func:`exercise_checkpoint_resume`: ``capture(cell)`` at the checkpoint and
    ``restore(cell, captured)`` before the resumed ``setup``, as ``run.py`` dumps and
    restores the database; ``normalize(snapshot)`` for comparing the final memory with
    an uninterrupted run's (MIRIX: the timestamps a run records)."""

    capture: Callable[[CellContext], Any] | None = None
    restore: Callable[[CellContext, Any], None] | None = None
    normalize: Callable[[Any], Any] = lambda snapshot: snapshot


class ObjectWatch:
    """sha256 of every file under ``work_dir/objects`` when first seen: a later change is
    a contract violation (item 6: write-once; a checkpoint hard-links them)."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.seen: dict[str, str] = {}

    def check(self) -> int:
        import hashlib

        root = self.work_dir / OBJECTS_DIR
        if not root.is_dir():
            return len(self.seen)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(self.work_dir).as_posix()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            before = self.seen.setdefault(rel, sha)
            assert before == sha, f"{rel} was modified after it was written (item 6)"
        return len(self.seen)


async def _learn_item(
    method: MemoryMethod,
    item: ContractItem,
    *,
    upto: str,
    idle_timeout_s: float,
    hook_timeout_s: float,
) -> None:
    """One learning item's hooks, up to ``played`` (the last on_message) or ``done``."""
    await _timed(
        method.before_task(copy.deepcopy(item.view)), hook_timeout_s, "before_task"
    )
    for message in item.messages:
        await _timed(
            method.on_message(item.view.task_uid, copy.deepcopy(message)),
            hook_timeout_s,
            "on_message",
        )
    if upto == "played":
        return
    await _finish_item(
        method, item, idle_timeout_s=idle_timeout_s, hook_timeout_s=hook_timeout_s
    )


async def _finish_item(
    method: MemoryMethod,
    item: ContractItem,
    *,
    idle_timeout_s: float,
    hook_timeout_s: float,
) -> None:
    await _timed(method.after_task(item.trajectory()), hook_timeout_s, "after_task")
    await _timed(
        method.wait_until_idle(idle_timeout_s),
        idle_timeout_s + IDLE_GRACE_S,
        "wait_until_idle",
    )


async def _freeze(
    method: MemoryMethod, *, idle_timeout_s: float, hook_timeout_s: float
) -> dict[str, Any]:
    await _timed(method.finalize(), hook_timeout_s, "finalize")
    await _timed(
        method.wait_until_idle(idle_timeout_s),
        idle_timeout_s + IDLE_GRACE_S,
        "wait_until_idle after finalize",
    )
    return await _timed(method.snapshot(), hook_timeout_s, "snapshot")


async def exercise_checkpoint_resume(
    factory: MethodFactory,
    cell: CellContext,
    reference_cell: CellContext,
    *,
    cut: tuple[str, int],
    items: Sequence[ContractItem] | None = None,
    hooks: CheckpointHooks | None = None,
    idle_timeout_s: float = 30.0,
    hook_timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Contract item 6 for ``factory`` (see the module docstring).

    ``cut`` is where the checkpoint is taken: ``("init", 0)`` after ``setup``,
    ``("done", k)`` after item k's idle wait, ``("played", k)`` after item k's last
    ``on_message``. After it the method goes on (item k's ``after_task``, or the next
    item) and is then dropped without ``finalize``, as a killed harness drops it; its
    ``work_dir`` is replaced by the checkpoint's copy (``objects/**`` hard-linked, as
    ``run.py`` does) and a new instance is set up with ``cell.resume``. Its snapshot must
    hash to the checkpoint's, and continuing must freeze the same memory (after
    ``hooks.normalize``) as an uninterrupted run on ``reference_cell``. Returns the
    reference and final snapshots and the checkpoint's state."""
    import shutil
    import tempfile

    from evals.skill_bench.run import _copy_method_dir

    kind, position = cut
    items = list(items if items is not None else contract_items(cell.benchmark))
    assert 0 <= position < len(items) or kind == "init"
    hooks = hooks or CheckpointHooks()
    timing = {"idle_timeout_s": idle_timeout_s, "hook_timeout_s": hook_timeout_s}

    reference = factory()
    try:
        await _timed(reference.setup(reference_cell), hook_timeout_s, "setup")
        for item in items:
            await _learn_item(reference, item, upto="done", **timing)
        expected = await _freeze(reference, **timing)
    finally:
        await _close(reference)

    watch = ObjectWatch(cell.work_dir)
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="checkpoint-"))
    first = factory()
    try:
        await _timed(first.setup(cell), hook_timeout_s, "setup")
        start = 0
        if kind != "init":
            for index, item in enumerate(items[: position + 1]):
                upto = "played" if (kind, index) == ("played", position) else "done"
                await _learn_item(first, item, upto=upto, **timing)
                watch.check()
            start = position + 1
        state = checkpoint_state_json(await first.checkpoint())
        recorded = await _timed(first.snapshot(), hook_timeout_s, "snapshot")
        sha = snapshot_sha256(recorded)
        _copy_method_dir(cell.work_dir, checkpoint_dir / "method")
        captured = hooks.capture(cell) if hooks.capture else None
        # The discarded continuation: the method goes on, then dies.
        if kind == "played":
            await _finish_item(first, items[position], **timing)
        elif start < len(items):
            await _learn_item(first, items[start], upto="done", **timing)
        watch.check()
    finally:
        await _close(first)

    shutil.rmtree(cell.work_dir, ignore_errors=True)
    _copy_method_dir(checkpoint_dir / "method", cell.work_dir)
    shutil.rmtree(checkpoint_dir)
    if hooks.restore:
        hooks.restore(cell, captured)
    resume = LearningResume(
        checkpoint_id=f"c{position:04d}-{kind}",
        kind=kind,  # type: ignore[arg-type]
        position=position,
        state=state,
        snapshot_sha256=sha,
    )
    resumed = factory()
    try:
        await _timed(
            resumed.setup(replace(cell, resume=resume)), hook_timeout_s, "resume setup"
        )
        rebuilt = await _timed(resumed.snapshot(), hook_timeout_s, "resumed snapshot")
        assert snapshot_sha256(rebuilt) == sha, (
            "the memory rebuilt from the checkpoint differs from the recorded one "
            "(contract item 6)"
        )
        if kind == "played":
            await _finish_item(resumed, items[position], **timing)
        for item in items[start:]:
            await _learn_item(resumed, item, upto="done", **timing)
            watch.check()
        final = await _freeze(resumed, **timing)
    finally:
        await _close(resumed)
    watch.check()
    assert hooks.normalize(final) == hooks.normalize(expected), (
        "the memory frozen after a resume differs from an uninterrupted run's "
        "(contract item 6)"
    )
    return {
        "reference": expected,
        "final": final,
        "state": state,
        "objects": watch.seen,
    }


# ---------------------------------------------------------------------------
# The empty assistant turn through the real rollout
# ---------------------------------------------------------------------------


async def play_alfworld_no_reply_step(
    tmp_path: Path,
    factory: MethodFactory,
    *,
    method_name: str,
    replies: Callable[[str, dict[str, Any]], Any] | None = None,
) -> tuple[Any, MemoryMethod]:
    """Play one ALFWorld learning episode whose second model call times out.

    The actor and the method share one gateway (on a loopback port) over the fake
    upstream; the rollout hands the method the empty assistant turn. ``replies`` answers
    the method's own model calls (default: the fake's generic reply). Returns the
    :class:`~evals.skill_bench.agent.rollout.EpisodeRun` and the method.
    """
    import evals.skill_bench.envs.alfworld as alf
    from evals.skill_bench.agent.actor import Actor
    from evals.skill_bench.agent.rollout import run_episode
    from evals.skill_bench.tests.fakes import FakeUpstream, Reply, serve_gateway
    from evals.skill_bench.tests.rollout_support import SALT, benchmark_of, text_reply
    from evals.skill_bench.tests.test_env_alfworld import Games, act

    actor_replies = [
        text_reply(act("look")),
        Reply(delay_s=3.0),
        text_reply(act("win")),
    ]

    def responder(endpoint: str, body: dict[str, Any]) -> Any:
        if endpoint == "chat/completions" and benchmark_of(body) == "alfworld":
            return actor_replies.pop(0)
        return replies(endpoint, body) if replies is not None else None

    games = Games()
    env = alf.AlfWorldEnv(data_root=tmp_path, backend_factory=games.factory())
    task = alf.AlfWorldEnv(backend_factory=Games().factory()).load_tasks("train")[0]
    fake = FakeUpstream(responder=responder)
    with serve_gateway(tmp_path, fake, deadline_s=1.0) as (_gateway, base_url):
        cell = contract_cell(
            tmp_path / "method",
            benchmark="alfworld",
            method=method_name,
            gateway_base_url=base_url,
        )
        method = factory()
        await method.setup(cell)
        try:
            async with Actor(
                gateway_base_url=base_url, run_id=cell.run_id, model=cell.model
            ) as actor:
                run = await run_episode(
                    env,
                    task,
                    method=method,
                    actor=actor,
                    salt=SALT,
                    phase="learning",
                    idle_timeout_s=30.0,
                )
        finally:
            await _close(method)
    assert EMPTY_ASSISTANT in run.trajectory.messages, (
        "the rollout emitted no empty turn"
    )
    return run, method


def assert_no_label_leaks(received: Any, task_strings: Sequence[str]) -> None:
    """Every object a method received is free of label keys and label strings (H10.1)."""
    leaks = find_label_leaks(received, forbidden_strings=task_strings)
    assert leaks == [], leaks


def unregistered_methods(factories: Mapping[str, Any]) -> list[str]:
    """Protocol methods implemented in ``methods/`` (placeholders excluded) without a factory."""
    from evals.skill_bench.methods.base import method_class

    return [
        name
        for name in protocol.METHODS
        if method_class(name) is not None and name not in factories
    ]
