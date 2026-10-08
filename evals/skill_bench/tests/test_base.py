"""Shared env contract (envs/base.py) and the importable H2 skeleton."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs.base import (
    Action,
    Env,
    Episode,
    EpisodeResult,
    Reset,
    Score,
    Split,
    StepResult,
    Task,
    new_cell_salt,
    opaque_task_uid,
)

H2_MODULES = [
    "evals.skill_bench.protocol",
    "evals.skill_bench.envs.base",
    "evals.skill_bench.envs.alfworld",
    "evals.skill_bench.agent.actor",
    "evals.skill_bench.agent.rollout",
    "evals.skill_bench.agent.trajectory",
    "evals.skill_bench.methods.base",
    "evals.skill_bench.methods.none",
    "evals.skill_bench.methods.mirix",
    "evals.skill_bench.methods.reasoningbank",
    "evals.skill_bench.methods.awm",
    "evals.skill_bench.methods.judge",
    "evals.skill_bench.infra.gateway",
    "evals.skill_bench.infra.mirix_server",
    "evals.skill_bench.infra.failures",
    "evals.skill_bench.run",
    "evals.skill_bench.schedule",
    "evals.skill_bench.report",
    "evals.skill_bench.audit",
    "evals.skill_bench.study",
]


@pytest.mark.parametrize("module", H2_MODULES)
def test_h2_skeleton_imports(module):
    importlib.import_module(module)


def test_h2_non_python_files_exist():
    root = protocol.SKILL_BENCH_DIR
    for rel in ("config/models.json", "infra/compose.yaml", "infra/init.sql"):
        assert (root / rel).is_file(), rel


def test_opaque_uid_format_and_mirix_session_id_rule():
    from mirix.schemas.message import SESSION_ID_MAX_LEN, SESSION_ID_PATTERN

    salt = new_cell_salt()
    uid = opaque_task_uid(salt, "train:0000")
    assert re.fullmatch(r"t[0-9a-f]{20}", uid)
    assert re.fullmatch(SESSION_ID_PATTERN, uid) and len(uid) <= SESSION_ID_MAX_LEN
    assert opaque_task_uid(salt, "train:0000") == uid
    assert opaque_task_uid(salt, "train:0001") != uid
    assert opaque_task_uid(new_cell_salt(), "train:0000") != uid
    with pytest.raises(ValueError):
        opaque_task_uid(b"", "train:0000")


def test_task_hides_raw_item_from_repr_and_equality():
    gold = {"answer": "42", "task_type": "pick_and_place", "category": "x"}
    task = Task("alfworld", "train", "train:0007", gold)
    assert "42" not in repr(task) and "pick_and_place" not in repr(task)
    assert task == Task("alfworld", "train", "train:0007", {"answer": "other"})
    assert task.uid(b"salt") == opaque_task_uid(b"salt", "train:0007")
    reset = Reset(
        "Your task is to: put a mug in the sink.",
        ({"role": "user", "content": "obs"},),
        private={"won": 0},
    )
    step = StepResult(
        ({"role": "user", "content": "obs2"},),
        True,
        private={"won": True, "info": {"facts": []}},
    )
    assert "won" not in repr(reset) and "facts" not in repr(step)


def test_episode_result_serializes():
    result = EpisodeResult(
        "webshop",
        "test",
        "test:0001",
        "tabc",
        Score(1.0, {"success": 0.0}),
        final_answer="12.5",
        steps=3,
        failure={"kind": "model", "code": "format_error"},
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert data["score"] == {
        "primary": 1.0,
        "metrics": {"success": 0.0},
        "answered": True,
    }
    assert result.scored is True
    defaults = EpisodeResult("webshop", "test", "x", "t", Score(0.5)).to_dict()
    assert defaults["score"] == {"primary": 0.5, "metrics": {}, "answered": True}
    assert json.loads(json.dumps(defaults))["failure"] is None
    assert EpisodeResult("webshop", "test", "x", "t", None).scored is False


class _Echo:
    def __init__(self, task: Task) -> None:
        self.task = task

    async def reset(self) -> Reset:
        return Reset("q?", ({"role": "user", "content": "q?"},))

    async def step(self, action: Action) -> StepResult:
        return StepResult(
            ({"role": "user", "content": str(action)},),
            True,
            private={"last": str(action)},
        )

    async def close(self) -> None:
        return None


class _EchoEnv:
    benchmark = "echo"

    def load_tasks(self, split: Split) -> list[Task]:
        return [Task("echo", split, f"{split}:0")]

    def episode(self, task: Task) -> Episode:
        return _Echo(task)

    def score(
        self, task: Task, final_answer: str | None, private: Mapping[str, Any]
    ) -> Score:
        return Score(float(final_answer == "a"), answered=final_answer is not None)


@pytest.mark.asyncio
async def test_protocols_are_structural():
    env = _EchoEnv()
    assert isinstance(env, Env)
    task = env.load_tasks("val")[0]
    episode = env.episode(task)
    assert isinstance(episode, Episode)
    start = await episode.reset()
    result = await episode.step("a")
    await episode.close()
    assert start.instruction == "q?" and result.done
    assert env.score(task, "a", result.private).primary == 1.0


# --- what method code may reach ------------------------------------------------------------

#: Harness modules a method must never import: the scorer-side environments, the cell
#: records and the tools that read them.
FORBIDDEN_METHOD_IMPORTS = (
    "evals.skill_bench.cell_records",
    "evals.skill_bench.run",
    "evals.skill_bench.audit",
    "evals.skill_bench.report",
    "evals.skill_bench.schedule",
    "evals.skill_bench.study",
    "evals.skill_bench.agent.rollout",
    "evals.skill_bench.agent.trajectory",
)
#: What of an environment a method may import.
ALLOWED_ENV_IMPORTS = {
    "evals.skill_bench.envs.base": None,  # types only
}
#: The scorer-side files next to a method's work_dir (``cell_records.AttemptPaths``).
SCORER_RECORD_NAMES = (
    "results.jsonl",
    "learning.jsonl",
    "injections.jsonl",
    "failures.jsonl",
    "trajectories",
    "attempt.json",
    "cell.json",
    "salt",
)


def test_method_code_does_not_reach_for_the_scorers_records():
    """A guard, not isolation: a method runs in the harness process, so no directory
    layout can hide the scorer's records from it; the audit detects what a method could
    only have read (H10.1 label and gold-answer scans of its outputs) and this test
    keeps the methods' code from importing the scorer side or walking out of work_dir."""
    import ast
    from pathlib import Path

    methods_dir = Path(__file__).resolve().parents[1] / "methods"
    problems = []
    for path in sorted(methods_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                imported = {a.name for a in node.names}
                for name in names:
                    if name.startswith(FORBIDDEN_METHOD_IMPORTS):
                        problems.append(f"{path.name}:{node.lineno} imports {name}")
                    elif name.startswith("evals.skill_bench.envs"):
                        allowed = ALLOWED_ENV_IMPORTS.get(name, set())
                        if allowed is not None and (
                            isinstance(node, ast.Import) or not imported <= allowed
                        ):
                            problems.append(
                                f"{path.name}:{node.lineno} imports {name} "
                                f"{sorted(imported)}"
                            )
            elif isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
                target = node.value
                owner = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else getattr(target, "id", "")
                )
                if owner in ("work_dir", "cell_dir"):
                    problems.append(f"{path.name}:{node.lineno} leaves {owner}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in SCORER_RECORD_NAMES:
                    problems.append(f"{path.name}:{node.lineno} names {node.value!r}")
    assert problems == []


def test_an_injection_records_degraded_only_when_set():
    """``Injection.degraded``: the record of an injection
    that is not degraded stays byte for byte what it was before the flag existed."""
    from dataclasses import replace

    from evals.skill_bench.agent.trajectory import canonical_json, to_jsonable
    from evals.skill_bench.methods.base import Injection

    plain = Injection("x", ("a",), 3)
    assert to_jsonable(plain) == {"text": "x", "item_ids": ["a"], "tokens": 3}
    assert canonical_json(plain) == b'{"item_ids":["a"],"text":"x","tokens":3}'
    degraded = Injection("x", ("a",), 3, degraded=True)
    assert to_jsonable(degraded) == {
        "text": "x",
        "item_ids": ["a"],
        "tokens": 3,
        "degraded": True,
    }
    assert degraded != plain and not Injection().degraded
    with pytest.raises(TypeError, match="degraded"):
        Injection("x", degraded="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="degraded"):
        Injection("x", degraded=1)  # type: ignore[arg-type]
    # The rollout's token count (dataclasses.replace) keeps the flag.
    assert replace(degraded, tokens=5).degraded is True
    assert replace(degraded, tokens=5).to_dict()["tokens"] == 5


def test_a_learning_resume_counts_its_rollbacks():
    from evals.skill_bench.methods.base import LearningResume

    resume = LearningResume("c0003-played", "played", 3, {}, "0" * 64)
    assert resume.rollbacks == 0
    assert LearningResume("c0003-played", "played", 3, {}, "0" * 64, 3).rollbacks == 3
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="rollbacks"):
            LearningResume("c0003-played", "played", 3, {}, "0" * 64, rollbacks=bad)  # type: ignore[arg-type]
