"""Test doubles for ``run.py`` and ``schedule.py`` tests (offline: fake upstream, real gateway).

* :class:`TinyEnv` - a one-step benchmark in ALFWorld's loop with a handful of items per
  split, so a whole cell (learning, freeze, test) runs in a second through the real
  rollout, actor and gateway. The fake upstream answers with :func:`tiny_responder`.
* :class:`StoringMethod` - a trivial learning method that stores something: the
  instruction of every learning item. It persists its memory under ``work_dir`` in
  ``finalize`` and restores it in a test-phase ``setup`` (the H6 contract), is
  checkpointed through :meth:`StoringMethod.checkpoint` and restored by a learning
  ``setup`` with ``cell.resume`` (contract item 6), and can be told to fail or to
  misbehave at a given point.
* :func:`conformant_gateway` - a real loopback gateway over a fake upstream with the
  protocol's settings (``protocol_conformant: true``).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.envs.base import Reset, Score, StepResult, Task
from evals.skill_bench.infra.gateway import Gateway
from evals.skill_bench.infra.processes import _registry_rows, process_start_time
from evals.skill_bench.methods.base import (
    CellContext,
    Injection,
    MemoryMethodBase,
    SignalLevel,
    TaskView,
    TrajectoryView,
    check_resumed_snapshot,
)
from evals.skill_bench.cell_records import (
    ATTEMPT_COMPLETE,
    CellPaths,
    read_json,
    read_ledger_lines,
    segment_calls,
)
from evals.skill_bench.run import CellRunner, CellSpec, RunOptions, seeded_order
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, chat_body, serve_gateway

TINY_SIZES: Mapping[str, int] = {"train": 6, "val": 4, "test": 5}
FIXED_GIT = {"commit": "c0ffee0", "dirty": False, "branch": "test"}


# ---------------------------------------------------------------------------
# A tiny benchmark
# ---------------------------------------------------------------------------


class TinyEpisode:
    def __init__(self, env: TinyEnv, task: Task) -> None:
        self.env = env
        self.task = task

    async def reset(self) -> Reset:
        question = str(self.task.raw["question"])
        if question in self.env.fail_reset:
            self.env.fail_reset.discard(question)  # once: a re-run succeeds
            raise InfraFailure("the environment's backend is unreachable")
        return Reset(
            instruction=question,
            messages=({"role": "user", "content": question},),
            fields={"question": question},
        )

    async def step(self, action: Any) -> StepResult:
        # One answer ends the episode; the scorer reads it from ``private``.
        return StepResult(messages=(), done=True, private={"answer": action})

    async def close(self) -> None:
        return None


class TinyEnv:
    """A one-step env in ALFWorld's loop: one question, one answer, scored by exact
    containment."""

    benchmark = "alfworld"

    def __init__(
        self,
        sizes: Mapping[str, int] = TINY_SIZES,
        *,
        fail_reset: Iterable[str] = (),
    ) -> None:
        self.sizes = dict(sizes)
        self.loaded: list[str] = []
        self.preflights: list[str] = []
        #: Questions whose episode fails once at reset with an INFRA error.
        self.fail_reset = set(fail_reset)

    def load_tasks(self, split: str) -> list[Task]:
        self.loaded.append(split)
        return [
            Task(
                self.benchmark,
                split,  # type: ignore[arg-type]
                f"{split}:{i:02d}",
                MappingProxyType(
                    {"question": f"Q {split} {i}?", "answer": f"A-{split}-{i}"}
                ),
            )
            for i in range(self.sizes[split])
        ]

    def episode(self, task: Task) -> TinyEpisode:
        return TinyEpisode(self, task)

    def score(
        self, task: Task, final_answer: str | None, private: Mapping[str, Any]
    ) -> Score:
        answer = private.get("answer")
        ok = isinstance(answer, str) and str(task.raw["answer"]) in answer
        return Score(primary=1.0 if ok else 0.0)

    def call_error_response(self) -> str:
        """The action the loop plays for a model call without a reply: no answer."""
        return ""

    def actor_messages(
        self, fields: Mapping[str, Any], memory: str = ""
    ) -> list[dict[str, Any]]:
        system = "TINY SYSTEM" + (f"\n{memory}" if memory else "")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": str(fields["question"])},
        ]


QUESTION_RE = re.compile(r"^Q (train|val|test) (\d+)\?$")


def tiny_answer(split: str, index: int) -> bool:
    """Which items the fake model answers correctly (odd items are wrong)."""
    return index % 2 == 0


def tiny_responder(endpoint: str, body: dict[str, Any]) -> Reply | None:
    """Answers TinyEnv questions (correctly for even items); other calls get the default."""
    if endpoint != "chat/completions":
        return None
    messages = body.get("messages") or []
    question = messages[-1].get("content") if messages else None
    match = QUESTION_RE.match(question) if isinstance(question, str) else None
    if match is None:
        return None
    split, index = match.group(1), int(match.group(2))
    answer = f"A-{split}-{index}" if tiny_answer(split, index) else "no idea"
    return Reply(
        json=chat_body(
            model=body.get("model", ""), content=f"<answer>{answer}</answer>"
        )
    )


def actor_questions(fake: FakeUpstream) -> list[str]:
    """The TinyEnv questions the actor asked upstream, in order."""
    out = []
    for call in fake.calls("chat/completions"):
        messages = call.json.get("messages") or []
        content = messages[-1].get("content") if messages else None
        if isinstance(content, str) and QUESTION_RE.match(content):
            out.append(content)
    return out


# ---------------------------------------------------------------------------
# A method that stores something
# ---------------------------------------------------------------------------


class InfraFailure(httpx.ConnectError):
    """An error ``infra/failures.py`` classifies as INFRA (gateway unreachable)."""

    def __init__(self, message: str = "injected infrastructure failure") -> None:
        super().__init__(message)


class StoringMethod(MemoryMethodBase):
    """Stores each learning item's instruction; injects how many it holds.

    ``events`` is shared by every instance of a test, so a test sees the hooks of a
    restarted harness too. Faults:

    * ``fail_after_task_at``: ``after_task`` of that learning item raises INFRA;
    * ``fail_test_questions``: ``before_task`` raises INFRA for those test questions (the
      set is shared and consumed, so a re-run of the item succeeds);
    * ``write_in_test``: the test phase changes the memory (breaks H10.4);
    * ``tamper_on_resume``: a test-phase ``setup`` restores something else;
    * ``on_before_task``: a coroutine run inside every ``before_task``;
    * ``fail_once``: ``{hook: {instruction, ...}}`` for the learning hooks
      ``before_task``, ``after_task`` and ``wait_until_idle`` (the item whose
      ``after_task`` ran last) and ``finalize`` (any key): the hook raises INFRA for that
      item once (the set is shared and consumed, so a resume goes through);
    * ``harness_once``: the same, raising a harness failure (a plain ``RuntimeError``);
    * ``tamper_checkpoint``: a learning ``setup`` with ``cell.resume`` restores
      something else (a :class:`~evals.skill_bench.methods.base.CheckpointMismatch`).

    Its memory lives in memory until ``finalize``: :meth:`checkpoint` returns it, and a
    learning ``setup`` with ``cell.resume`` puts it back (contract item 6).
    """

    name = "reasoningbank"
    signal_level: SignalLevel = "S1"

    def __init__(
        self,
        events: list[tuple[Any, ...]],
        *,
        name: str | None = None,
        fail_after_task_at: int | None = None,
        fail_test_questions: set[str] | None = None,
        write_in_test: bool = False,
        tamper_on_resume: bool = False,
        on_before_task: Callable[[TaskView], Any] | None = None,
        fail_once: dict[str, set[str]] | None = None,
        harness_once: dict[str, set[str]] | None = None,
        tamper_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if name is not None:
            self.name = name
        self.events = events
        self.memory: list[str] = []
        self.frozen = False
        self.fail_after_task_at = fail_after_task_at
        self.fail_test_questions = (
            fail_test_questions if fail_test_questions is not None else set()
        )
        self.write_in_test = write_in_test
        self.tamper_on_resume = tamper_on_resume
        self.on_before_task = on_before_task
        self.fail_once = fail_once if fail_once is not None else {}
        self.harness_once = harness_once if harness_once is not None else {}
        self.tamper_checkpoint = tamper_checkpoint
        self.learned = 0
        self.last_learned: str | None = None

    @property
    def memory_file(self) -> Path:
        assert self.cell is not None
        return self.cell.work_dir / "memory.json"

    def _fault(self, hook: str, key: str | None) -> None:
        for faults, error in (
            (
                self.fail_once,
                InfraFailure(f"injected infrastructure failure in {hook}"),
            ),
            (self.harness_once, RuntimeError(f"injected harness failure in {hook}")),
        ):
            keys = faults.get(hook)
            if keys and (key in keys or "*" in keys):
                keys.discard(key if key in keys else "*")
                raise error

    async def setup(self, cell: CellContext) -> None:
        await super().setup(cell)
        self.events.append(
            ("setup", cell.phase, cell.attempt, cell.run_id)
            + ((cell.resume.checkpoint_id,) if cell.resume is not None else ())
        )
        if cell.phase == "test":
            data = json.loads(self.memory_file.read_text())  # raises if never frozen
            self.memory = list(data) + (["TAMPERED"] if self.tamper_on_resume else [])
            self.frozen = True
        elif cell.resume is not None:
            self.memory = list(cell.resume.state["memory"])
            self.learned = int(cell.resume.state["learned"])
            if self.tamper_checkpoint:
                self.memory.append("TAMPERED")
            check_resumed_snapshot(self.name, self._snapshot(), cell.resume)
        else:
            assert not any(cell.work_dir.iterdir()), "a learning attempt starts empty"

    async def checkpoint(self) -> dict[str, Any]:
        self.events.append(("checkpoint",))
        return {"memory": list(self.memory), "learned": self.learned}

    async def before_task(self, task: TaskView) -> Injection:
        phase = "test" if self.frozen else "learning"
        self.events.append(("before_task", phase, task.instruction))
        if self.on_before_task is not None:
            await self.on_before_task(task)
        if not self.frozen:
            self._fault("before_task", task.instruction)
        if self.frozen and task.instruction in self.fail_test_questions:
            self.fail_test_questions.discard(task.instruction)
            raise InfraFailure()
        if self.frozen and self.write_in_test:
            self.memory.append("written in the test phase")
        text = f"MEMORY holds {len(self.memory)} lessons" if self.memory else ""
        return Injection(text, tuple(f"m{i}" for i in range(len(self.memory))))

    async def on_message(self, task_uid: str, message: dict[str, Any]) -> None:
        self.events.append(("on_message", message.get("role")))

    async def after_task(self, trajectory: TrajectoryView) -> None:
        self.events.append(("after_task", trajectory.instruction))
        if self.fail_after_task_at is not None and (
            self.learned == self.fail_after_task_at
        ):
            raise InfraFailure("the learner's store is down")
        self._fault("after_task", trajectory.instruction)
        self.memory.append(trajectory.instruction)
        self.learned += 1
        self.last_learned = trajectory.instruction

    async def wait_until_idle(self, timeout_s: float) -> None:
        self.events.append(("wait_until_idle",))
        if not self.frozen:
            self._fault("wait_until_idle", self.last_learned)

    async def finalize(self) -> None:
        self.events.append(("finalize",))
        self._fault("finalize", "*")
        self.memory_file.write_text(json.dumps(self.memory))
        # The memory is final: every later hook is the test phase's.
        self.frozen = True

    def _snapshot(self) -> dict[str, Any]:
        return {"method": self.name, "memory": list(self.memory)}

    async def snapshot(self) -> dict[str, Any]:
        self.events.append(("snapshot",))
        return self._snapshot()

    async def close(self) -> None:
        self.events.append(("close",))


# ---------------------------------------------------------------------------
# Gateway and runner
# ---------------------------------------------------------------------------


@contextmanager
def conformant_gateway(
    tmp_path: Path, fake: FakeUpstream, **overrides: Any
) -> Iterator[tuple[Gateway, str]]:
    """A real loopback gateway over ``fake`` with the protocol settings (conformant)."""
    overrides.setdefault("backoff_base_s", protocol.GATEWAY_BACKOFF_BASE_S)
    with serve_gateway(tmp_path, fake, **overrides) as served:
        yield served


def uncaptured_tree(dest: Path, **_kw: Any) -> dict[str, Any]:
    """``run.capture_tree`` for the tests' runners: a test that says the tree is dirty
    (``git={..., "dirty": True}``) never diffs the real checkout; it passes its own
    ``capture`` to see a tree diff recorded."""
    return {
        "dirty": True,
        "diff": None,
        "diff_sha256": None,
        "diff_bytes": 0,
        "untracked": 0,
        "untracked_skipped": [],
        "error": "not captured (tests/run_support.runner)",
    }


def runner(
    tmp_path: Path,
    base_url: str,
    *,
    spec: CellSpec | None = None,
    env: Any = None,
    factory: Callable[[], Any] | None = None,
    git: Mapping[str, Any] = FIXED_GIT,
    capture: Callable[..., Mapping[str, Any]] = uncaptured_tree,
    **options: Any,
) -> CellRunner:
    """A :class:`CellRunner` of ``spec`` (default the ALFWorld reasoningbank cell) on the
    tiny benchmark, attached to the gateway at ``base_url``, with a fixed git record and
    ``capture`` in place of ``run.capture_tree``; ``options`` go to :class:`RunOptions`
    (``note=`` included)."""
    spec = spec or CellSpec("alfworld", "luna", "reasoningbank")
    options.setdefault("preflight", False)
    options.setdefault("idle_timeout_s", 30.0)
    run_options = RunOptions(
        out_dir=options.pop("out_dir", tmp_path / "runs" / spec.cell_id),
        gateway_url=base_url,
        gateway_runs_dir=tmp_path / "runs",
        **options,
    )
    tiny = env if env is not None else TinyEnv()
    cell = CellRunner(
        spec,
        run_options,
        env_factory=lambda _benchmark: tiny,
        method_factory=factory,
        git=lambda: dict(git),
    )
    cell.capture_tree = capture
    return cell


# ---------------------------------------------------------------------------
# The learning-resume invariant
# ---------------------------------------------------------------------------


def order(sizes: Mapping[str, int] = TINY_SIZES) -> list[str]:
    """The raw ids of a TinyEnv cell's learning items, in the seed-42 order."""
    return [t.raw_id for t in seeded_order(TinyEnv(sizes).load_tasks("train"))]


def question(raw_id: str) -> str:
    split, index = raw_id.split(":")
    return f"Q {split} {int(index)}?"


def uninterrupted_memory() -> list[str]:
    """What :class:`StoringMethod` freezes after an uninterrupted learning phase."""
    return [question(r) for r in order()]


def recomputed_segments(ledger_path: Path, paths: CellPaths) -> list[list[str]]:
    """Each discarded segment's calls recomputed from the ledger markers."""
    lines = read_ledger_lines(ledger_path)
    out: list[list[str]] = []
    seen: set[str] = set()
    for row in read_rows(paths.attempt(1).discarded_log):
        calls = segment_calls(
            lines,
            checkpoint_offset=row["checkpoint"]["offset"],
            resume_offset=row["resume_marker_offset"],
            exclude=seen,
        )
        ids = [c["call_id"] for c in calls]
        seen.update(ids)
        out.append(ids)
    return out


def resumed_cleanly(
    ledger_path: Path, paths: CellPaths, *, storing: bool = True
) -> None:
    """The invariant of a resumed learning phase (attempt 1 of the cell at ``paths``, whose
    run's ledger is ``ledger_path``): positions 0..n-1 once each in the live
    learning.jsonl, the frozen memory of an uninterrupted run (``storing``: StoringMethod's),
    discarded rows equal to the markers' segments, a verified resumed snapshot, and at
    most two checkpoints kept."""
    attempt = paths.attempt(1)
    rows = read_rows(attempt.learning)
    assert [r["position"] for r in rows] == list(range(len(order())))
    assert [r["raw_id"] for r in rows] == order()
    assert all(r["score"] is not None for r in rows)
    if storing:  # StoringMethod's memory: every instruction once, in order
        frozen = read_json(attempt.frozen_snapshot)
        assert frozen["snapshot"]["memory"] == uninterrupted_memory()
    discarded = read_rows(attempt.discarded_log)
    assert [[c["call_id"] for c in r["calls"]] for r in discarded] == (
        recomputed_segments(ledger_path, paths)
    )
    assert [r["segment"] for r in discarded] == list(range(1, len(discarded) + 1))
    kept = [p for p in attempt.checkpoints.iterdir() if p.name != "LATEST"]
    assert len(kept) <= 2 and not [p for p in kept if p.name.endswith(".partial")]
    state = read_json(attempt.state)
    assert state["state"] == ATTEMPT_COMPLETE
    assert len(state["resumes"]) == len(discarded)
    for check in (
        c for c in state["snapshot_checks"] if c["kind"] == "learning_resume"
    ):
        assert check["equal"] is True


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ledger(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    return read_rows(tmp_path / "runs" / run_id / "gateway.jsonl")


async def eventually(check: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Processes a test leaves behind
# ---------------------------------------------------------------------------


def _same_start(pid: int, started: Any) -> bool:
    """``pid`` is still the process recorded with start time ``started``. A child of
    this process that has exited (a zombie, which ``ps`` still lists) is reaped first
    and counts as gone."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError:  # not our child, or already reaped
        pass
    current = process_start_time(pid)
    return current is not None and current == " ".join(str(started or "").split())


def leftover_processes(root: Path) -> dict[int, str]:
    """``{pid: start time}`` of what a test started under ``root`` and is still running:
    every unsettled ``started`` row of an ``owned_processes.jsonl`` (a launched gateway,
    a MIRIX server) and every ``process.json`` (a fake ``run.py``) whose pid still has
    the recorded start time (never a reused pid). Never this process. Blocking."""
    found: dict[int, str] = {}
    for registry in root.rglob("owned_processes.jsonl"):
        for pid, row in _registry_rows(registry).items():
            if _same_start(pid, row.get("started")):
                found[pid] = str(row.get("started"))
    for record in root.rglob("process.json"):
        try:
            info = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = info.get("pid") if isinstance(info, dict) else None
        if isinstance(pid, int) and not isinstance(pid, bool):
            if _same_start(pid, info.get("started")):
                found[pid] = str(info.get("started"))
    for own in (os.getpid(), os.getpgrp(), os.getppid(), 0, 1):
        found.pop(own, None)
    return found


def _signal_test_process(pid: int, sig: int) -> None:
    """``sig`` to ``pid``'s group when it leads one (a gateway, a worker: each started
    in a session of its own), else to ``pid`` alone; never to this process's group."""
    try:
        if os.getpgid(pid) == pid and pid != os.getpgrp():
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_test_processes(root: Path, *, grace_s: float = 5.0) -> list[int]:
    """Stop what a test (or a whole session, ``root`` its basetemp) started under
    ``root`` and left running: what its records prove (:func:`leftover_processes`: a
    gateway its scheduler only released, the workers of a test that failed or timed
    out) and every process whose command line names a path under ``root``
    (:func:`processes_naming`: a fake scheduler, a gateway with no registry row yet).
    ``SIGTERM``, then ``SIGKILL`` after ``grace_s`` to what is provably still the same
    process. Returns the pids that were running. Blocking; for a fixture's teardown and
    :func:`start_orphan_reaper`."""
    running = {**processes_naming(root), **leftover_processes(root)}
    return _stop_all(running, grace_s)


def _stop_all(running: Mapping[int, str], grace_s: float) -> list[int]:
    """``SIGTERM`` every ``{pid: start time}``, then ``SIGKILL`` after ``grace_s`` what is
    provably still the same process; returns the pids. Blocking."""
    for pid in running:
        _signal_test_process(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and any(
        _same_start(pid, started) for pid, started in running.items()
    ):
        time.sleep(0.05)
    for pid, started in running.items():
        if _same_start(pid, started):
            _signal_test_process(pid, signal.SIGKILL)
    return sorted(running)


def processes_naming(root: Path) -> dict[int, str]:
    """``{pid: start time}`` of every process whose command line names a path under
    ``root`` (a gateway's ``--runs-dir``, a worker's cell, a MIRIX stub's data), never
    this process or its parent. ``ps`` in the C locale. Blocking."""
    marks = {str(root) + os.sep, str(root.resolve()) + os.sep}
    result = subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,lstart=,command="],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C", "LANG": "C", "LC_TIME": "C"},
    )
    found: dict[int, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split(
            None, 6
        )  # pid, lstart ("Wed Sep 30 16:53:56 2026"), command
        if len(fields) < 7 or not fields[0].isdigit():
            continue
        if any(mark in fields[6] for mark in marks):
            found[int(fields[0])] = " ".join(fields[1:6])
    for own in (os.getpid(), os.getppid(), 0, 1):
        found.pop(own, None)
    return found


#: The watchdog :func:`start_orphan_reaper` runs: it waits, on the standard library
#: alone, until the pytest that started it is gone (its parent pid changes), then
#: imports this module and stops what that session left under its basetemp.
_REAPER = """
import os, sys, time
parent = int(sys.argv[1])
while os.getppid() == parent:
    time.sleep(0.5)
from pathlib import Path
from evals.skill_bench.tests.run_support import stop_test_processes
stop_test_processes(Path(sys.argv[2]))
"""

_reapers: dict[str, subprocess.Popen[bytes]] = {}


def start_orphan_reaper(basetemp: Path) -> subprocess.Popen[bytes]:
    """Start, once per ``basetemp`` in this process, a watchdog in a session of its own
    that outlives this pytest however it ends (a normal exit, or a ``SIGKILL`` from a
    test runner's timeout, which no fixture teardown survives) and then stops
    everything the session's tests left running under ``basetemp``
    (:func:`stop_test_processes`): a launched gateway runs in its own session with no
    parent-death check, so it would otherwise run on. The watchdog
    holds none of this process's file descriptors (a pipe reading pytest's output
    still ends with pytest)."""
    key = str(basetemp.resolve())
    reaper = _reapers.get(key)
    if reaper is not None and reaper.poll() is None:
        return reaper
    reaper = subprocess.Popen(
        [sys.executable, "-c", _REAPER, str(os.getpid()), key],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    _reapers[key] = reaper
    return reaper
