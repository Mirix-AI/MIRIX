"""ALFWorld environment.

Parity tests run SkillOpt 79124b3's own ``run_alfworld_batch`` and
``AlfWorldEnvironmentManager`` from the read-only clone (``skillopt_ref``) with a
recording ``chat_target`` and compare every actor call byte for byte. Real-game tests
use ``$ALFWORLD_DATA`` (or ``~/.cache/alfworld``) and skip without it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs import alfworld as alf
from evals.skill_bench.envs.base import Env, Episode, Reset, Split, StepResult, Task
from evals.skill_bench.tests.skillopt_ref import (
    requires_skillopt,
    skillopt_available,
    skillopt_modules,
    skillopt_path,
)

SKILL_KNOWLEDGE_HEADER = (
    "\n\n## Skill Knowledge\n"
    "Below is a skill document with learned strategies. "
    "Use these guidelines to inform your decisions:\n\n"
)
FALLBACK_EMPTY = "<think>empty model response</think><action>look</action>"
FALLBACK_MISSING = "<think>missing action tag</think><action>look</action>"
FORBIDDEN_KEYS = {
    "won",
    "info",
    "infos",
    "facts",
    "expert_plan",
    "extra.expert_plan",
    "extra.gamefile",
    "gamefile",
    "task_type",
    "policy_commands",
}


def act(action: str) -> str:
    return f"<think>t</think><action>{action}</action>"


# Responses that exercise every branch of SkillOpt's fallbacks and the projection.
TRICKY_RESPONSES = [
    act("go to countertop 1"),
    "",  # empty -> look, format error
    "   \n ",  # empty after strip -> look, format error
    "I would go to the cabinet.",  # no action tag -> look, format error
    "<think>x</think><ACTION>Look</ACTION>",  # regex is case-sensitive -> format error
    "<action>Go To Cabinet 1</action>",  # no <think>: executed, projection-invalid
    "<think>这是</think><action>inventory</action>",  # CJK: executed, projection-invalid
    "<think>a</think><action>  examine {thing} 7  </action> then <action>look</action>",
    "</action> oops <action>look</action>",  # projection slices to ""
    "\n\n<think>ok</think>\n<action>\nLOOK\n</action>\n",
    act("win"),
]
TRICKY_FORMAT_ERRORS = 4
TRICKY_COMMANDS = [
    "go to countertop 1",
    "look",
    "look",
    "look",
    "look",
    "go to cabinet 1",
    "inventory",
    "examine {thing} 7",
    "",
    "look",
    "win",
]

INITIAL_OBSERVATION = (
    "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking "
    "quickly around you, you see a cabinet 1, a countertop 1, and a {sinkbasin} 'one'."
    "\n\nYour task is to: put a clean cloth in countertop."
)


class ScriptedGame:
    """A deterministic stand-in for one TextWorld game (``alf.GameBackend``)."""

    def __init__(
        self,
        gamefile: Path,
        seed: int,
        train_eval: str,
        *,
        extra_info: Mapping[str, Any] | None = None,
        step_delay_s: float = 0.0,
        gate: threading.Event | None = None,
    ) -> None:
        self.gamefile, self.seed, self.train_eval = gamefile, seed, train_eval
        self.extra = dict(extra_info or {})
        self.delay = step_delay_s
        self.gate = gate
        self.commands: list[str] = []
        self.closed = False
        self.thread_ids: list[int] = []

    def _admissible(self) -> list[str]:
        n = len(self.commands)
        return [
            "go to cabinet 1",
            "go to countertop 1",
            "help",
            "inventory",
            "look",
            f"examine {{thing}} {n}",
        ]

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        self.thread_ids.append(threading.get_ident())
        info = {
            "won": False,
            "admissible_commands": self._admissible(),
            "extra.gamefile": str(self.gamefile),
        }
        return INITIAL_OBSERVATION, {**info, **self.extra}

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        self.thread_ids.append(threading.get_ident())
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.delay:
            time.sleep(self.delay)
        self.commands.append(command)
        won = command == "win"
        obs = (
            "You win."
            if won
            else f"You try '{command}' {{step {len(self.commands)}}}. Nothing happens."
        )
        return (
            obs,
            won,
            {"won": won, "admissible_commands": self._admissible(), **self.extra},
        )

    def close(self) -> None:
        self.closed = True


class Games:
    def __init__(self) -> None:
        self.made: list[Any] = []

    def factory(
        self, cls: Callable[..., Any] = ScriptedGame, **kwargs: Any
    ) -> alf.BackendFactory:
        def make(path: Path, seed: int, train_eval: str) -> Any:
            game = cls(path, seed, train_eval, **kwargs)
            self.made.append(game)
            return game

        return make


@pytest.fixture
def games() -> Games:
    return Games()


@pytest.fixture
def train_task() -> Task:
    return alf.AlfWorldEnv(backend_factory=Games().factory()).load_tasks("train")[0]


def make_env(games: Games, tmp_path: Path, **kwargs: Any) -> alf.AlfWorldEnv:
    factory_kwargs = {
        k: kwargs.pop(k) for k in ("extra_info", "step_delay_s", "gate") if k in kwargs
    }
    return alf.AlfWorldEnv(
        data_root=tmp_path, backend_factory=games.factory(**factory_kwargs), **kwargs
    )


async def play(
    env: alf.AlfWorldEnv, task: Task, responses: Sequence[str], memory: str = ""
) -> SimpleNamespace:
    """Run one episode like the phase-11 rollout will: render, call, step."""
    prompts: list[list[dict[str, Any]]] = []
    steps: list[StepResult] = []
    async with env.episode(task) as episode:
        reset = await episode.reset()
        fields, done = reset.fields, False
        for response in responses:
            if done:
                break
            prompts.append(env.actor_messages(fields, memory))
            step = await episode.step(response)
            steps.append(step)
            fields, done = step.fields, step.done
    return SimpleNamespace(reset=reset, steps=steps, prompts=prompts, episode=episode)


def as_calls(prompts: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, str]]:
    calls = []
    for messages in prompts:
        assert [m["role"] for m in messages] == ["system", "user"]
        calls.append({"system": messages[0]["content"], "user": messages[1]["content"]})
    return calls


# --- SkillOpt reference run ------------------------------------------------------------


class VectorAdapter:
    """SkillOpt's vectorised ``AlfworldEnvs`` interface over one game backend."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.prev: list[Any] = [None]

    def reset(self) -> tuple[list[str], None, list[dict[str, Any]]]:
        obs, info = self.backend.reset()
        info = dict(info)
        self.prev = [list(info["admissible_commands"])]
        return [obs], None, [info]

    def step(
        self, actions: list[str]
    ) -> tuple[list[str], None, list[float], list[bool], list[dict[str, Any]]]:
        obs, done, info = self.backend.step(actions[0])
        info = dict(info)
        self.prev = [list(info["admissible_commands"])]
        return [obs], None, [10.0 * float(info["won"])], [done], [info]

    @property
    def get_admissible_commands(self) -> list[Any]:
        return self.prev

    def close(self) -> None:
        self.backend.close()


def run_skillopt(
    backend: Any,
    responses: Sequence[str],
    *,
    skill_content: str = "",
    max_steps: int = 50,
    out_root: str = "",
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """SkillOpt's ``run_alfworld_batch`` with a recording ``chat_target``."""
    calls: list[dict[str, str]] = []
    pending = list(responses)

    def chat_target(
        system: str, user: str, **_kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        calls.append({"system": system, "user": user})
        return pending.pop(0), {}

    with skillopt_modules(chat_target=chat_target) as load:
        rollout = load("skillopt.envs.alfworld.rollout")
        manager_module = load("skillopt.envs.alfworld.vendor.env_manager")
        projection = load("skillopt.envs.alfworld.vendor.alfworld_projection")
        manager = manager_module.AlfWorldEnvironmentManager(
            VectorAdapter(backend),
            projection.alfworld_projection,
            SimpleNamespace(env=SimpleNamespace(history_length=2)),
        )
        results = rollout.run_alfworld_batch(
            manager,
            skill_content,
            max_steps=max_steps,
            out_root=out_root,
            max_api_workers=1,
            result_ids=["item"],
        )
    return calls, results


# --- frozen templates -------------------------------------------------------------------


def test_frozen_assets_match_protocol_records():
    checked = protocol.verify_frozen_assets("envs/templates/alfworld/")
    assert checked == [
        "envs/templates/alfworld/config_tw.yaml",
        "envs/templates/alfworld/format.json",
        "envs/templates/alfworld/rollout_no_history.md",
        "envs/templates/alfworld/rollout_with_history.md",
    ]
    splits = protocol.verify_frozen_assets("envs/splits/alfworld/")
    assert "envs/splits/alfworld/gamefiles.sha256.json" in splits
    assert protocol.verify_frozen_assets("envs/vendor/alfworld_projection.py")
    manifest = protocol.fingerprint_manifest()["files"]
    for rel in checked:
        assert manifest[rel] == protocol.FROZEN_ASSET_SHA256[rel]


def test_tampered_template_is_refused(tmp_path):
    root = tmp_path / "skill_bench"
    shutil.copytree(
        protocol.SKILL_BENCH_DIR / "envs",
        root / "envs",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(protocol.SKILL_BENCH_DIR / "config", root / "config")
    alf.AlfWorldTemplates.load(root)  # pristine copy loads
    target = root / "envs/templates/alfworld/rollout_with_history.md"
    target.write_text(target.read_text() + " ", encoding="utf-8")
    with pytest.raises(protocol.ProtocolDriftError, match="rollout_with_history.md"):
        alf.AlfWorldTemplates.load(root)
    config = root / "envs/templates/alfworld/config_tw.yaml"
    config.write_text(config.read_text().replace("'dqn'", "'dagger'"), encoding="utf-8")
    with pytest.raises(protocol.ProtocolDriftError):
        alf.load_config(root)
    split = root / "envs/splits/alfworld/test.json"
    split.write_text(
        split.read_text().replace("test:0000", "test:9999"), encoding="utf-8"
    )
    with pytest.raises(protocol.ProtocolDriftError, match="test.json"):
        protocol.verify_frozen_assets("envs/splits/alfworld/", root)


@requires_skillopt
def test_template_files_are_byte_exact_copies_of_skillopt():
    for name in ("rollout_no_history.md", "rollout_with_history.md"):
        ours = (
            protocol.SKILL_BENCH_DIR / "envs/templates/alfworld" / name
        ).read_bytes()
        assert (
            ours == skillopt_path(f"skillopt/envs/alfworld/prompts/{name}").read_bytes()
        )
    for split in ("train", "val", "test"):
        ours = (
            protocol.SKILL_BENCH_DIR / f"envs/splits/alfworld/{split}.json"
        ).read_bytes()
        assert (
            ours
            == skillopt_path(
                f"data/alfworld_path_split/{split}/items.json"
            ).read_bytes()
        )
    assert (
        protocol.SKILL_BENCH_DIR / "envs/splits/alfworld/split_manifest.json"
    ).read_bytes() == skillopt_path(
        "data/alfworld_path_split/split_manifest.json"
    ).read_bytes()


@requires_skillopt
def test_format_json_excerpts_are_verbatim_skillopt_lines():
    spec = json.loads(
        (protocol.SKILL_BENCH_DIR / "envs/templates/alfworld/format.json").read_text()
    )
    templates = spec["templates"]
    for key, source in spec["provenance"].items():
        if source.get("file") is None:
            continue
        data = skillopt_path(source["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == source["file_sha256"], key
        first, last = source["lines"]
        lines = data.decode("utf-8").splitlines(keepends=True)
        assert "".join(lines[first - 1 : last]) == source["excerpt"], key
    prov = spec["provenance"]
    # Each frozen value is the literal SkillOpt wrote in the quoted lines.
    assert f'system="{templates["system_prompt"]}"' in prov["system_prompt"]["excerpt"]
    assert '"history_length": 2' in prov["history_length"]["excerpt"]
    assert f'f"{templates["history_entry"]}"' in prov["history_entry"]["excerpt"]
    assert (
        'memory_contexts.append("\\n".join(lines))' in prov["history_entry"]["excerpt"]
    )
    assert (
        "\"\\n \".join(\n                f\"'{s}'\" for s in admissible_actions[i] if s != 'help'"
        in (prov["admissible_item"]["excerpt"])
    )
    assert f"obs.find('{templates['task_marker']}')" in prov["task_marker"]["excerpt"]
    assert 'f"{skill_content}\\n"' in prov["memory_slot"]["excerpt"]
    assert 'prompt = skill_prompt + "\\n" + prompt' in prov["memory_joiner"]["excerpt"]
    assert (
        f'return idx, "{templates["fallback_empty_response"]}"'
        in prov["fallbacks"]["excerpt"]
    )
    assert (
        f'return idx, "{templates["fallback_missing_action"]}"'
        in prov["fallbacks"]["excerpt"]
    )
    assert (  # W2: a call that raised plays 'look' (rollout.py:244-245)
        f'            except Exception:\n                return idx, "{templates["fallback_call_error"]}"'
        in prov["fallbacks"]["excerpt"]
    )
    assert (
        f'r"{templates["action_tag_pattern"]}"' in prov["action_tag_pattern"]["excerpt"]
    )


@requires_skillopt
def test_config_differs_from_skillopt_only_in_the_expert_switch():
    ours = (protocol.SKILL_BENCH_DIR / alf.CONFIG_FILE).read_text().splitlines()
    theirs = (
        skillopt_path("skillopt/envs/alfworld/vendor/config_tw.yaml")
        .read_text()
        .splitlines()
    )
    header_len = next(i for i, line in enumerate(ours) if line.startswith("dataset:"))
    assert all(line.startswith("#") or not line for line in ours[:header_len])
    body = ours[header_len:]
    assert len(body) == len(theirs)
    diffs = [(a, b) for a, b in zip(body, theirs) if a != b]
    assert len(diffs) == 1
    assert (
        "training_method: 'dqn'" in diffs[0][0]
        and "training_method: 'dagger'" in diffs[0][1]
    )
    config = alf.load_config()
    assert config["general"]["training_method"] == "dqn"
    assert (
        config["rl"]["training"]["max_nb_steps_per_episode"]
        == protocol.ALFWORLD_MAX_STEPS
        == 50
    )


# --- rendering: golden strings (no clone needed) -----------------------------------------


@pytest.mark.asyncio
async def test_rendered_prompts_golden(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    run = await play(
        env,
        train_task,
        [act("go to countertop 1"), act("look"), act("inventory"), act("win")],
    )
    system = "You are an expert agent operating in the ALFRED Embodied Environment."
    admissible0 = "'go to cabinet 1'\n 'go to countertop 1'\n 'inventory'\n 'look'\n 'examine {thing} 0'"
    assert run.prompts[0] == [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"\n{system}\nYour current observation is: {INITIAL_OBSERVATION}\n"
                f"Your admissible actions of the current situation are: [{admissible0}].\n\n"
                "Now it's your turn to take an action.\n"
                "You should first reason step-by-step about the current situation. This reasoning "
                "process MUST be enclosed within <think> </think> tags.\n"
                "Once you've finished your reasoning, you should choose an admissible action for "
                "current step and present it within <action> </action> tags.\n"
            ),
        },
    ]
    admissible3 = "'go to cabinet 1'\n 'go to countertop 1'\n 'inventory'\n 'look'\n 'examine {thing} 3'"
    history = (
        "[Observation 2: 'You try 'go to countertop 1' {step 1}. Nothing happens.', Action 2: 'look']\n"
        "[Observation 3: 'You try 'look' {step 2}. Nothing happens.', Action 3: 'inventory']"
    )
    assert run.prompts[3][1]["content"] == (
        f"\n{system} Your task is to: put a clean cloth in countertop.\n"
        "Prior to this step, you have already taken 3 step(s). Below are the most recent 2 "
        f"observations and the corresponding actions you took: {history}\n"
        "You are now at step 4 and your current observation is: "
        "You try 'inventory' {step 3}. Nothing happens.\n"
        f"Your admissible actions of the current situation are: [{admissible3}].\n\n"
        "Now it's your turn to take an action.\n"
        "You should first reason step-by-step about the current situation. This reasoning "
        "process MUST be enclosed within <think> </think> tags.\n"
        "Once you've finished your reasoning, you should choose an admissible action for "
        "current step and present it within <action> </action> tags.\n"
    )


@pytest.mark.asyncio
async def test_memory_goes_in_the_slot_and_only_there(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    responses = [act("look"), act("inventory"), act("win")]
    plain = await play(env, train_task, responses)
    memory = "## Lessons\n- Check the countertop first.\n- {braces} stay literal."
    with_memory = await play(env, train_task, responses, memory=memory)
    for a, b in zip(plain.prompts, with_memory.prompts):
        assert a[0] == b[0]  # the system message never changes
        assert b[1]["content"] == memory + "\n" + "\n" + a[1]["content"]
    blank = await play(env, train_task, responses, memory=" \n\t")
    assert blank.prompts == plain.prompts


# --- parity with SkillOpt's own code -------------------------------------------------------


@requires_skillopt
@pytest.mark.asyncio
async def test_prompts_are_byte_identical_to_skillopt(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    ours = await play(env, train_task, TRICKY_RESPONSES)
    reference = ScriptedGame(Path("x"), 0, "train")
    calls, results = run_skillopt(
        reference, TRICKY_RESPONSES, out_root=str(tmp_path / "so")
    )
    assert as_calls(ours.prompts) == calls
    assert len(calls) == len(TRICKY_RESPONSES)
    # Same commands reach the game, so the fallbacks and the projection agree too.
    assert games.made[0].commands == reference.commands == TRICKY_COMMANDS
    assert results[0]["hard"] == 1
    assert env.score(train_task, None, ours.steps[-1].private).primary == 1.0
    # SkillOpt's recorded responses show which ones fell back to 'look'.
    conversation = json.loads(
        (tmp_path / "so/predictions/item/conversation.json").read_text()
    )
    fallbacks = [
        s["model_response"] in (FALLBACK_EMPTY, FALLBACK_MISSING) for s in conversation
    ]
    assert sum(fallbacks) == TRICKY_FORMAT_ERRORS == ours.episode.format_errors
    assert [s.fields["format_error"] for s in ours.steps] == fallbacks


@requires_skillopt
@pytest.mark.asyncio
async def test_prompt_with_memory_equals_skillopt_minus_its_skill_header(
    games, tmp_path, train_task
):
    memory = "Remember: {x} and 'quotes'.\n\nSecond paragraph."
    env = make_env(games, tmp_path)
    responses = [act("look"), "", act("win")]
    ours = await play(env, train_task, responses, memory=memory)
    calls, _ = run_skillopt(
        ScriptedGame(Path("x"), 0, "train"), responses, skill_content=memory
    )
    for mine, theirs in zip(as_calls(ours.prompts), calls):
        assert mine["system"] == theirs["system"]
        assert theirs["user"].startswith(SKILL_KNOWLEDGE_HEADER)
        assert mine["user"] == theirs["user"][len(SKILL_KNOWLEDGE_HEADER) :]


@requires_skillopt
@pytest.mark.asyncio
async def test_step_limit_matches_skillopt(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    responses = [act("look")] * 60
    ours = await play(env, train_task, responses)
    calls, results = run_skillopt(ScriptedGame(Path("x"), 0, "train"), responses)
    assert len(ours.prompts) == len(calls) == protocol.ALFWORLD_MAX_STEPS
    assert as_calls(ours.prompts) == calls
    last = ours.steps[-1]
    assert last.done and last.fields["truncated"] and last.private["won"] is False
    assert results[0]["hard"] == 0
    assert env.score(train_task, None, last.private).primary == 0.0


@requires_skillopt
def test_vendored_projection_matches_skillopt():
    corpus = [
        *TRICKY_RESPONSES,
        FALLBACK_EMPTY,
        FALLBACK_MISSING,
        "<think></think><action></action>",
        "<action>a</action><think>b</think>",
        "x" * 100,
        "<Think>A</Think><action>B</action>",
        "<think>a</think><action>put 中 in 1</action>",
    ]
    from evals.skill_bench.envs.vendor.alfworld_projection import alfworld_projection

    with skillopt_modules() as load:
        original = load(
            "skillopt.envs.alfworld.vendor.alfworld_projection"
        ).alfworld_projection
        assert original(list(corpus), [[]] * len(corpus)) == alfworld_projection(
            list(corpus), [[]] * len(corpus)
        )


def test_parse_response_branches():
    t = alf.AlfWorldTemplates.load()
    cases = {
        None: ("look", True, True),
        "": ("look", True, True),
        "no tags": ("look", True, True),
        "<think>x</think><ACTION>go</ACTION>": ("look", True, True),
        act("Go To Desk 1"): ("go to desk 1", False, True),
        "<action>go</action>": ("go", False, False),
        "<think>中</think><action>go</action>": ("go", False, False),
    }
    for raw, (command, format_error, valid) in cases.items():
        parsed = alf.parse_response(t, raw)
        assert (parsed.command, parsed.format_error, parsed.action_valid) == (
            command,
            format_error,
            valid,
        ), raw


# --- what leaves the environment -------------------------------------------------------------


def _walk(value: Any, path: str = "") -> list[tuple[str, Any]]:
    out = [(path, value)]
    if isinstance(value, Mapping):
        for key, item in value.items():
            out.append((f"{path}.{key}#key", key))
            out.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.extend(_walk(item, f"{path}[{i}]"))
    return out


def assert_agent_view_clean(result: Reset | StepResult, task: Task) -> None:
    view = {"messages": result.messages, "fields": result.fields}
    if isinstance(result, Reset):
        view["instruction"] = result.instruction
    for where, value in _walk(view):
        if where.endswith("#key"):
            assert value not in FORBIDDEN_KEYS, where
        if isinstance(value, bool) and not where.endswith(
            ("format_error", "action_valid", "truncated")
        ):
            raise AssertionError(f"unexpected boolean at {where}")
        if isinstance(value, str):
            assert task.raw["task_type"] not in value, where
            assert task.raw_id not in value, where
            assert "Episode result" not in value and "Task type" not in value, where
    assert set(result.private) == {"won"}
    for message in result.messages:
        assert message["role"] == "user" and isinstance(message["content"], str)


@pytest.mark.asyncio
async def test_reset_and_step_expose_no_info(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    run = await play(env, train_task, TRICKY_RESPONSES)
    assert_agent_view_clean(run.reset, train_task)
    for step in run.steps:
        assert_agent_view_clean(step, train_task)
    assert run.reset.instruction == "put a clean cloth in countertop."
    assert run.reset.messages[0]["content"] == (
        INITIAL_OBSERVATION + "\n\nAdmissible actions:\n"
        "go to cabinet 1\ngo to countertop 1\ninventory\nlook\nexamine {thing} 0"
    )
    assert [s.private["won"] for s in run.steps] == [False] * (
        len(TRICKY_RESPONSES) - 1
    ) + [True]


@pytest.mark.parametrize(
    "key", ["facts", "extra.expert_plan", "policy_commands", "score"]
)
@pytest.mark.asyncio
async def test_unexpected_info_keys_are_refused(games, tmp_path, train_task, key):
    env = make_env(games, tmp_path, extra_info={key: ["anything"]})
    episode = env.episode(train_task)
    with pytest.raises(alf.AlfWorldEnvError, match="unexpected TextWorld info keys"):
        await episode.reset()
    await episode.close()
    assert games.made[0].closed


def test_episode_receives_only_the_gamefile(games, tmp_path, train_task):
    # The gamefile path itself names the task type ("look_at_obj_in_light-..."), so it
    # must stay inside the episode; the agent-view checks above prove it never leaves.
    episode = make_env(games, tmp_path).episode(train_task)
    state = vars(episode)
    held = [v for v in state.values() if isinstance(v, (str, Path))]
    assert {str(v) for v in held if v} == {
        train_task.raw["gamefile"],
        str(tmp_path / train_task.raw["gamefile"]),
    }
    assert not any(isinstance(v, Mapping) for v in state.values())
    assert train_task.raw_id not in " ".join(str(v) for v in held)


# --- seeds, splits, data ------------------------------------------------------------------------


def test_game_seed_is_derived_from_the_gamefile_path():
    gamefile = "json_2.1.1/train/look_at_obj_in_light-AlarmClock-None-DeskLamp-305/trial_T20190908_082736_108723/game.tw-pddl"
    assert alf.game_seed(gamefile) == 126214300
    assert alf.game_seed(gamefile) == 42 + int(
        hashlib.sha256(gamefile.encode()).hexdigest()[:7], 16
    )
    seeds = {
        alf.game_seed(t.raw["gamefile"])
        for s in ("train", "val", "test")
        for t in alf.AlfWorldEnv().load_tasks(s)
    }
    assert len(seeds) == 191 and all(0 <= s < 2**32 for s in seeds)


@pytest.mark.asyncio
async def test_backend_gets_seed_and_train_eval_from_the_gamefile(games, tmp_path):
    env = make_env(games, tmp_path)
    expected: list[tuple[Split, str]] = [
        ("train", "train"),
        ("val", "eval_in_distribution"),
        ("test", "eval_out_of_distribution"),
    ]
    for split, train_eval in expected:
        task = env.load_tasks(split)[3]
        async with env.episode(task) as episode:
            await episode.reset()
            assert episode.seed == alf.game_seed(task.raw["gamefile"])
        game = games.made[-1]
        assert (game.seed, game.train_eval) == (
            alf.game_seed(task.raw["gamefile"]),
            train_eval,
        )
        assert game.gamefile == tmp_path / task.raw["gamefile"]


def test_splits_follow_skillopt_and_the_protocol():
    env = alf.AlfWorldEnv()
    tasks = {s: env.load_tasks(s) for s in ("train", "val", "test")}
    assert {s: len(t) for s, t in tasks.items()} == {
        "train": 39,
        "val": 18,
        "test": 134,
    }
    ids = [t.raw_id for ts in tasks.values() for t in ts]
    assert len(set(ids)) == len(ids)
    gamefiles = [t.raw["gamefile"] for ts in tasks.values() for t in ts]
    assert len(set(gamefiles)) == len(gamefiles)
    assert {alf.eval_dataset_for(t.raw["gamefile"]) for t in tasks["train"]} == {
        "train"
    }
    assert {alf.eval_dataset_for(t.raw["gamefile"]) for t in tasks["val"]} == {
        "eval_in_distribution"
    }
    assert {alf.eval_dataset_for(t.raw["gamefile"]) for t in tasks["test"]} == {
        "eval_out_of_distribution"
    }
    assert "task_type" not in repr(tasks["train"][0])


def test_resolve_gamefile_and_default_data_root(monkeypatch, tmp_path):
    assert (
        alf.resolve_gamefile("a/b.tw-pddl", tmp_path)
        == (tmp_path / "a/b.tw-pddl").resolve()
    )
    assert (
        alf.resolve_gamefile(str(tmp_path / "x"), "/elsewhere")
        == (tmp_path / "x").resolve()
    )
    monkeypatch.setenv("ALFWORLD_DATA", str(tmp_path))
    assert alf.default_data_root() == tmp_path
    monkeypatch.delenv("ALFWORLD_DATA")
    assert alf.default_data_root() == Path(os.path.expanduser("~/.cache/alfworld"))


def test_score_uses_only_the_private_won_flag(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    assert env.score(train_task, None, {"won": True}).primary == 1.0
    assert env.score(train_task, "ignored", {"won": False}).primary == 0.0
    with pytest.raises(ValueError):
        env.score(train_task, None, {})


def test_env_satisfies_the_protocols(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    assert isinstance(env, Env)
    assert isinstance(env.episode(train_task), Episode)
    with pytest.raises(ValueError):
        alf.AlfWorldEnv(max_steps=51)


# --- episode life cycle and threading ------------------------------------------------------------


@pytest.mark.asyncio
async def test_game_calls_run_off_the_event_loop(games, tmp_path, train_task):
    env = make_env(games, tmp_path, step_delay_s=0.3)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await play(env, train_task, [act("look"), act("win")])
    finally:
        task.cancel()
    assert ticks >= 20  # the loop kept running during two 0.3 s blocking steps
    assert threading.get_ident() not in games.made[0].thread_ids


@pytest.mark.asyncio
async def test_episode_order_and_reuse_rules(games, tmp_path, train_task):
    env = make_env(games, tmp_path)
    episode = env.episode(train_task)
    with pytest.raises(RuntimeError, match="reset"):
        await episode.step(act("look"))
    await episode.reset()
    with pytest.raises(RuntimeError, match="once"):
        await episode.reset()
    with pytest.raises(TypeError):
        await episode.step({"tool": "x"})
    step = await episode.step(act("win"))
    assert step.done
    with pytest.raises(RuntimeError, match="over"):
        await episode.step(act("look"))
    await episode.close()
    assert games.made[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        await episode.step(act("look"))


@pytest.mark.asyncio
async def test_cancelled_step_breaks_the_episode_and_close_waits(
    games, tmp_path, train_task
):
    gate = threading.Event()
    env = make_env(games, tmp_path, gate=gate)
    episode = env.episode(train_task)
    await episode.reset()
    step = asyncio.create_task(episode.step(act("look")))
    await asyncio.sleep(0.05)
    step.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step
    with pytest.raises(alf.AlfWorldEnvError, match="interrupted"):
        await episode.step(act("look"))
    closing = asyncio.create_task(episode.close())
    await asyncio.sleep(0.05)
    assert (
        not closing.done() and not games.made[0].closed
    )  # waits for the running game call
    gate.set()
    await closing
    assert games.made[0].closed and games.made[0].commands == ["look"]


class CappedGame(ScriptedGame):
    """Ends the game at the step cap the way real TextWorld does.

    The pinned config sets ``rl.training.max_nb_steps_per_episode`` to 50, so TextWorld
    itself reports ``done`` on the 50th step whether or not the task was won.
    """

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        obs, done, info = super().step(command)
        return obs, done or len(self.commands) >= protocol.ALFWORLD_MAX_STEPS, info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("last", "won"), [(act("look"), False), (act("win"), True)], ids=["cap", "win"]
)
async def test_truncated_marks_the_cap_even_when_the_game_ends_it(
    games, tmp_path, train_task, last, won
):
    env = alf.AlfWorldEnv(data_root=tmp_path, backend_factory=games.factory(CappedGame))
    responses = [act("look")] * (protocol.ALFWORLD_MAX_STEPS - 1) + [last]
    run = await play(env, train_task, responses)
    assert len(run.steps) == protocol.ALFWORLD_MAX_STEPS
    assert not any(s.done or s.fields["truncated"] for s in run.steps[:-1])
    end = run.steps[-1]
    assert end.done and end.private["won"] is won
    assert end.fields["truncated"] is (not won)  # a win on the last step is no cap hit
    assert env.score(train_task, None, end.private).primary == float(won)


class UnendingGame(ScriptedGame):
    """Reports ``won`` but never ``done``: only the harness's cap ends its episode."""

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        obs, _done, info = super().step(command)
        return obs, False, info


@pytest.mark.asyncio
@pytest.mark.parametrize("game", [ScriptedGame, UnendingGame])
@pytest.mark.parametrize(("last", "won"), [("look", False), ("win", True)])
async def test_truncated_at_a_harness_cap_below_textworlds(
    games, tmp_path, train_task, game, last, won
):
    env = alf.AlfWorldEnv(
        data_root=tmp_path, backend_factory=games.factory(game), max_steps=3
    )
    run = await play(env, train_task, [act("look"), act("look"), act(last)])
    end = run.steps[-1]
    assert len(run.steps) == 3 and end.private["won"] is won
    assert end.done  # the cap ends the episode, won or not
    assert end.fields["truncated"] is (not won)


class FailingGame(ScriptedGame):
    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        if command == "explode":
            raise alf.AlfWorldEnvError("ALFWorld game process died (exit code -9)")
        return super().step(command)


@pytest.mark.asyncio
async def test_failed_game_call_breaks_the_episode(games, tmp_path, train_task):
    env = alf.AlfWorldEnv(
        data_root=tmp_path, backend_factory=games.factory(FailingGame)
    )
    async with env.episode(train_task) as episode:
        await episode.reset()
        await episode.step(act("look"))
        with pytest.raises(alf.AlfWorldEnvError, match="died"):
            await episode.step(act("explode"))
        with pytest.raises(alf.AlfWorldEnvError, match="interrupted or failed"):
            await episode.step(act("look"))
    assert games.made[0].closed and games.made[0].commands == ["look"]


@pytest.mark.asyncio
async def test_close_during_a_game_call(games, tmp_path, train_task):
    gate = threading.Event()
    env = make_env(games, tmp_path, gate=gate)
    episode = env.episode(train_task)
    await episode.reset()
    step = asyncio.create_task(episode.step(act("look")))
    await asyncio.sleep(0.05)
    closing = asyncio.create_task(episode.close())
    await asyncio.sleep(0.05)
    gate.set()
    await closing
    with pytest.raises(alf.AlfWorldEnvError, match="closed during a game call"):
        await step
    assert games.made[0].closed


# --- pinned game data (W35) ----------------------------------------------------------------------


def _fake_release(root: Path, env: alf.AlfWorldEnv, task: Task) -> Path:
    """A data root holding one manifest game with exactly its pinned bytes' hashes swapped in."""
    rel = task.raw["gamefile"]
    game = root / rel
    game.parent.mkdir(parents=True)
    game.write_text("game", encoding="utf-8")
    (game.parent / "traj_data.json").write_text("{}", encoding="utf-8")
    env.gamefile_hashes[rel] = {
        "game": hashlib.sha256(b"game").hexdigest(),
        "traj_data": hashlib.sha256(b"{}").hexdigest(),
    }
    return game.resolve()


def test_every_manifest_game_has_pinned_hashes():
    env = alf.AlfWorldEnv(backend_factory=Games().factory())
    manifest = {
        task.raw["gamefile"]
        for split in ("train", "val", "test")
        for task in env.load_tasks(split)
    }
    assert set(env.gamefile_hashes) == manifest and len(manifest) == 191
    for hashes in env.gamefile_hashes.values():
        assert set(hashes) == {"game", "traj_data"}
        assert all(len(v) == 64 for v in hashes.values())
    assert set(env.shared_hashes) == {"logic/alfred.pddl", "logic/alfred.twl2"}


def test_gamefile_problems_names_missing_and_changed_files(tmp_path):
    game = tmp_path / "g" / "game.tw-pddl"
    game.parent.mkdir()
    game.write_text("game")
    pinned = {"game": hashlib.sha256(b"game").hexdigest(), "traj_data": "0" * 64}
    assert alf.gamefile_problems(game, None) == [f"{game} is not a manifest game"]
    assert alf.gamefile_problems(game, pinned) == [
        f"{game.parent / 'traj_data.json'} missing"
    ]
    (game.parent / "traj_data.json").write_text("{}")
    assert alf.gamefile_problems(game, pinned) == [
        f"{game.parent / 'traj_data.json'} differs from the pinned sha256"
    ]
    pinned["traj_data"] = hashlib.sha256(b"{}").hexdigest()
    assert alf.gamefile_problems(game, pinned) == []


@pytest.mark.asyncio
async def test_the_default_backend_refuses_a_changed_game(
    tmp_path, train_task, monkeypatch
):
    """A re-generated or edited game must never be played under the pinned manifest."""
    started: list[Path] = []

    class Spawned(ScriptedGame):
        def __init__(self, gamefile: Path, *args: Any, **kwargs: Any) -> None:
            started.append(gamefile)
            super().__init__(gamefile, *args)

    monkeypatch.setattr(alf, "ProcessBackend", Spawned)
    env = alf.AlfWorldEnv(data_root=tmp_path)
    game = _fake_release(tmp_path, env, train_task)
    async with env.episode(train_task) as episode:
        await episode.reset()  # identical bytes: the game starts
    assert started == [game]
    game.write_text("game with a different goal", encoding="utf-8")
    with pytest.raises(alf.AlfWorldEnvError, match="differs from the pinned sha256"):
        async with env.episode(train_task) as episode:
            await episode.reset()
    assert started == [game]  # the changed game never reached a game process
    with pytest.raises(alf.AlfWorldEnvError, match="not a manifest game"):
        env._factory(tmp_path / "other" / "game.tw-pddl", 1, "train")


def test_the_game_engine_packages_are_pinned(monkeypatch):
    """W37: TextWorld decides observations and admissible commands; a drift is refused."""
    from importlib import metadata

    assert alf.package_problems() == []  # this venv holds the pinned versions
    pyproject = (protocol.SKILL_BENCH_DIR.parents[1] / "pyproject.toml").read_text()
    for name, version in protocol.ALFWORLD_PACKAGES.items():
        assert f'"{name}=={version}"' in pyproject
    real = metadata.version

    def drifted(name: str) -> str:
        if name == "textworld":
            return "1.6.2"
        if name == "alfworld":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", drifted)
    assert alf.package_problems() == [
        "alfworld is not installed (protocol pins 0.4.2)",
        "textworld 1.6.2 is installed, protocol pins 1.7.0",
    ]
    env = alf.AlfWorldEnv(backend_factory=Games().factory())
    with pytest.raises(alf.AlfWorldEnvError, match="textworld 1.6.2"):
        env._verified_backend(Path("/nonexistent/game.tw-pddl"), 1, "train")


def test_verify_gamefiles_checks_the_shared_logic_files(tmp_path, train_task):
    env = alf.AlfWorldEnv(data_root=tmp_path, backend_factory=Games().factory())
    problems = env.verify_gamefiles("train")
    assert f"{tmp_path / 'logic/alfred.pddl'} missing" in problems
    assert len(problems) == 2 + 2 * 39  # both logic files, every game and traj_data
    (tmp_path / "logic").mkdir()
    for rel in env.shared_hashes:
        (tmp_path / rel).write_text("tampered")
    assert all("differs" in p for p in env.verify_gamefiles("train") if "/logic/" in p)


def _alfworld_data() -> Path | None:
    candidates = [
        os.environ.get("ALFWORLD_DATA"),
        str(Path.home() / ".cache/alfworld"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "json_2.1.1").is_dir():
            return Path(candidate)
    return None


def _alfworld_engine_installed() -> bool:
    """The ALFWorld game engine (``alfworld``, ``textworld``: the optional
    ``[skill-bench]`` extra) is importable."""
    import importlib.util

    try:
        return all(importlib.util.find_spec(n) for n in ("alfworld", "textworld"))
    except (ImportError, ValueError):  # a module hidden or half-installed
        return False


ALFWORLD_DATA = _alfworld_data()
requires_alfworld_engine = pytest.mark.skipif(
    not _alfworld_engine_installed(),
    reason="the ALFWorld game engine (the [skill-bench] extra) is not installed",
)
requires_alfworld_data = pytest.mark.skipif(
    ALFWORLD_DATA is None or not _alfworld_engine_installed(),
    reason="ALFWorld data or its game engine (the [skill-bench] extra) not found",
)


@requires_alfworld_data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_installed_alfworld_data_is_the_pinned_release(split):
    """W35: the local json_2.1.1 games are byte-identical to the ones the split pins."""
    env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA)
    assert env.verify_gamefiles(split) == []


# val:0000 (valid_seen look_at_obj_in_light-AlarmClock-None-DeskLamp-323): the first
# response is empty, so SkillOpt's fallback plays 'look'; the rest wins in 5 steps.
WINNING_RESPONSES = [
    "",
    act("go to sidetable 1"),
    act("take alarmclock 3 from sidetable 1"),
    act("go to sidetable 2"),
    act("use desklamp 2"),
]


class RecordingBackend:
    """Wraps a real backend and keeps what it returned, for a SkillOpt replay."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.reset_result: tuple[str, dict[str, Any]] | None = None
        self.transitions: list[tuple[str, tuple[str, bool, dict[str, Any]]]] = []

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        obs, info = self.inner.reset()
        self.reset_result = (obs, dict(info))
        return obs, info

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        obs, done, info = self.inner.step(command)
        self.transitions.append((command, (obs, done, dict(info))))
        return obs, done, info

    def close(self) -> None:
        self.inner.close()


class ReplayBackend:
    def __init__(self, recording: RecordingBackend) -> None:
        self.recording = recording
        self.i = 0

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        assert self.recording.reset_result is not None
        return self.recording.reset_result

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        expected, result = self.recording.transitions[self.i]
        assert command == expected
        self.i += 1
        return result

    def close(self) -> None:
        pass


def _gone(pid: int) -> bool:
    """True once the (reaped) process no longer exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@requires_alfworld_data
@pytest.mark.asyncio
async def test_real_game_smoke_via_worker_threads():
    started = time.monotonic()
    recordings: list[RecordingBackend] = []
    config = alf.load_config()

    def factory(path: Path, seed: int, train_eval: str) -> RecordingBackend:
        recording = RecordingBackend(
            alf.ProcessBackend(path, seed, train_eval, config=config)
        )
        recordings.append(recording)
        return recording

    env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA, backend_factory=factory)
    task = env.load_tasks("val")[0]
    run = await play(env, task, WINNING_RESPONSES)
    assert run.reset.instruction == "look at alarmclock under the desklamp."
    assert [s.fields["action"] for s in run.steps] == [
        "look",
        "go to sidetable 1",
        "take alarmclock 3 from sidetable 1",
        "go to sidetable 2",
        "use desklamp 2",
    ]
    assert run.steps[-1].done and run.steps[-1].private["won"] is True
    assert run.episode.steps == 5 and run.episode.format_errors == 1
    assert env.score(task, None, run.steps[-1].private).primary == 1.0
    assert_agent_view_clean(run.reset, task)
    for step in run.steps:
        assert_agent_view_clean(step, task)
    # No expert: TextWorld returned only the requested infos.
    recording = recordings[0]
    assert recording.reset_result is not None
    assert set(recording.reset_result[1]) == {
        "won",
        "admissible_commands",
        "extra.gamefile",
    }
    assert all(
        set(info) <= alf.ALLOWED_INFO_KEYS for _, (_, _, info) in recording.transitions
    )
    game_process = recording.inner
    assert isinstance(game_process, alf.ProcessBackend)
    assert _gone(game_process.pid)  # the game process ended with the episode
    elapsed = time.monotonic() - started
    assert elapsed < 20, f"real-game smoke took {elapsed:.1f} s; mark it slow"

    if skillopt_available():
        calls, results = run_skillopt(ReplayBackend(recording), WINNING_RESPONSES)
        assert as_calls(run.prompts) == calls
        assert results[0]["hard"] == 1


@requires_alfworld_data
@pytest.mark.asyncio
async def test_real_games_are_deterministic_and_run_concurrently():
    # Several games load at once. In one process this corrupts TextWorld's global
    # parsers (IndexError inside tatsu); each episode's own game process avoids it.
    env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA)
    tests = env.load_tasks("test")
    task = tests[0]
    responses = [act("look"), act("inventory"), "", act("go to desk 1")]
    first, second, *others = await asyncio.gather(
        play(env, task, responses),
        play(env, task, responses),
        *(play(env, t, responses) for t in tests[1:5]),
    )
    assert first.reset.messages == second.reset.messages
    assert [s.messages for s in first.steps] == [s.messages for s in second.steps]
    assert first.prompts == second.prompts
    assert (
        first.episode.seed == second.episode.seed == alf.game_seed(task.raw["gamefile"])
    )
    assert all(len(run.steps) == len(responses) for run in others)
    for task_, concurrent in zip(tests[1:5], others):  # same as when played alone
        assert (await play(env, task_, responses)).prompts == concurrent.prompts


@requires_alfworld_data
@pytest.mark.asyncio
async def test_real_expert_config_would_leak_and_is_refused():
    config = alf.load_config()
    dagger = {**config, "general": {**config["general"], "training_method": "dagger"}}
    env = alf.AlfWorldEnv(
        data_root=ALFWORLD_DATA,
        backend_factory=partial(alf.ProcessBackend, config=dagger),
    )
    episode = env.episode(env.load_tasks("train")[0])
    with pytest.raises(alf.AlfWorldEnvError, match="expert"):
        await episode.reset()
    await episode.close()


@requires_alfworld_data
def test_every_manifest_game_is_present_and_solvable():
    env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA)
    for split in ("train", "val", "test"):
        assert env.missing_gamefiles(split) == []
        for task in env.load_tasks(split):
            alf._check_gamefile(env.gamefile_path(task))


def test_missing_game_is_a_clear_error(tmp_path):
    env = alf.AlfWorldEnv(data_root=tmp_path)
    task = env.load_tasks("train")[0]
    with pytest.raises(alf.AlfWorldEnvError, match="missing ALFWorld game"):
        alf._check_gamefile(env.gamefile_path(task))
    assert len(env.missing_gamefiles("val")) == 18


# --- the per-game process -----------------------------------------------------------------------


def _fake_game_dir(tmp_path: Path, content: Mapping[str, Any]) -> Path:
    game_dir = tmp_path / "json_2.1.1/train/pick_and_place_simple-X-None-Y-1/trial_T0"
    game_dir.mkdir(parents=True)
    (game_dir / "traj_data.json").write_text("{}", encoding="utf-8")
    gamefile = game_dir / "game.tw-pddl"
    gamefile.write_text(json.dumps(content), encoding="utf-8")
    return gamefile


@requires_alfworld_engine
def test_game_process_reports_errors_from_the_child(tmp_path):
    gamefile = _fake_game_dir(tmp_path, {"solvable": True, "pddl_domain": "(not pddl"})
    backend = alf.ProcessBackend(gamefile, 1, "train", config=alf.load_config())
    try:
        with pytest.raises(alf.AlfWorldEnvError, match="game process failed"):
            backend.reset()
    finally:
        backend.close()
    assert _gone(backend.pid)
    with pytest.raises(alf.AlfWorldEnvError, match="closed"):
        backend.reset()
    unsolvable = _fake_game_dir(tmp_path / "u", {"solvable": False})
    with pytest.raises(alf.AlfWorldEnvError, match="not marked solvable"):
        alf.ProcessBackend(unsolvable, 1, "train", config=alf.load_config())


def test_silent_game_process_is_killed(tmp_path, monkeypatch):
    started: list[Any] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    gamefile = _fake_game_dir(tmp_path, {"solvable": True})
    with pytest.raises(alf.AlfWorldEnvError, match="did not answer"):
        alf.ProcessBackend(
            gamefile, 1, "train", config=alf.load_config(), call_timeout_s=0.001
        )
    assert len(started) == 1 and started[0].poll() is not None and _gone(started[0].pid)


def test_game_process_gets_no_credentials(tmp_path, monkeypatch):
    env = {
        "PATH": "/bin",
        "HOME": "/h",
        "OPENROUTER_API_KEY": "x",
        "GITHUB_TOKEN": "y",
        "DB_PASSWORD": "z",
        "AWS_SECRET_ACCESS_KEY": "w",
        "GOOGLE_APPLICATION_CREDENTIALS": "v",
    }
    assert alf.game_process_env(env) == {"PATH": "/bin", "HOME": "/h"}
    monkeypatch.setenv("SOME_API_KEY", "leak")
    assert "SOME_API_KEY" not in alf.game_process_env()


@requires_alfworld_engine
def test_game_process_exits_when_the_harness_goes_away(tmp_path):
    gamefile = _fake_game_dir(tmp_path, {"solvable": True})
    backend = alf.ProcessBackend(gamefile, 1, "train", config=alf.load_config())
    process = backend._process
    assert process.stdin is not None
    process.stdin.close()  # what a killed parent looks like to the child
    assert process.wait(timeout=30) == 0
    backend.close()


@requires_alfworld_data
def test_games_run_from_a_stdin_script():
    """A spawn-based worker would re-run the launcher's __main__ and fail here."""
    script = (
        "import asyncio\n"
        "from evals.skill_bench.envs import alfworld as alf\n"
        "async def main():\n"
        f"    env = alf.AlfWorldEnv(data_root={str(ALFWORLD_DATA)!r})\n"
        "    async with env.episode(env.load_tasks('val')[0]) as ep:\n"
        "        reset = await ep.reset()\n"
        "        step = await ep.step('<think>t</think><action>look</action>')\n"
        "    print(reset.instruction, step.fields['action'])\n"
        "asyncio.run(main())\n"
    )
    done = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        text=True,
        cwd=protocol.REPO_ROOT,
        env=alf.game_process_env(),
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    # Only the script's own line: TextWorld's planner output never reaches stdout.
    assert done.stdout == "look at alarmclock under the desklamp. look\n"


@requires_alfworld_data
def test_crashed_game_process_is_an_env_error():
    assert ALFWORLD_DATA is not None
    env = alf.AlfWorldEnv(data_root=ALFWORLD_DATA)
    task = env.load_tasks("val")[1]
    backend = alf.ProcessBackend(
        env.gamefile_path(task), 7, "eval_in_distribution", config=env.config
    )
    try:
        backend.reset()
        assert backend.pid is not None
        os.kill(backend.pid, 9)
        with pytest.raises(alf.AlfWorldEnvError, match="died|gone"):
            backend.step("look")
    finally:
        backend.close()
