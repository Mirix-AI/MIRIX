"""ALFWorld environment.

The actor sees exactly SkillOpt 79124b3's single-turn ALFWorld prompt: a fixed system
message and, per step, ``rollout_no_history.md`` (first step) or
``rollout_with_history.md`` (the last two observation/action pairs). The frozen pieces
live in ``envs/templates/alfworld/`` (``format.json`` records the SkillOpt source lines
each piece was copied from) and are checked against ``protocol.FROZEN_ASSET_SHA256``.

Who sees what:

* ``Reset.messages`` / ``StepResult.messages`` (Q20, what methods receive through
  ``agent/trajectory.py``): one user message per environment output, holding the raw
  observation and the admissible actions. The assistant message (the model's raw
  output) is added by the rollout.
* ``fields`` (the actor template's inputs): observation, admissible actions, task
  description, the observation/action history and, after a step, the executed action
  and whether a format fallback was used. :meth:`AlfWorldEnv.actor_messages` renders
  them with the memory slot of Q21.
* ``private`` (scorer only): ``{"won": bool}``. Nothing else of TextWorld's ``info``
  leaves this module; the game is built without AlfredExpert, so ``facts`` and
  ``expert_plan`` do not exist (the ported ``config_tw.yaml`` sets
  ``training_method: dqn``), and any unexpected ``info`` key is an error.

``Episode.step`` takes the model's raw response. It applies SkillOpt's fallbacks
(``rollout.py:238-243``: an empty response or one without ``<action>...</action>``
becomes ``look`` and counts as a format error) and SkillRL's projection, then steps
the game. Infrastructure errors of a model call are not turned into ``look``: the gateway
retries them and they surface as infrastructure failures (H4.1, H8). A MODEL failure
without a reply (a timeout of an attempt that had the whole deadline, a moderation
refusal) is played as SkillOpt ``rollout.py:244-245`` plays any failed call: the rollout
sends :meth:`AlfWorldEnv.call_error_response` (``<think>error</think><action>look</action>``).

Game files: ``splits/alfworld/gamefiles.sha256.json`` pins the sha256 of every manifest
game and its ``traj_data.json``. The default backend refuses a game whose files differ
(another ``alfworld-download`` release, a corrupted cache); :meth:`AlfWorldEnv.verify_gamefiles`
checks a whole split up front.

TextWorld is synchronous, so every game call runs in a worker thread
(``asyncio.to_thread``). Each episode owns one game, built with ALFWorld's own
``AlfredTWEnv.init_env`` for that single gamefile and seeded from the gamefile path
(:func:`game_seed`). As in SkillOpt, each game lives in its own process
(:class:`ProcessBackend`): TextWorld's parsers are module globals, so two games in one
process cannot load concurrently.

Left out on purpose: no retrieval query built from the gold task type, no "Episode
result" line, no sentinel sessions, no hard-coded ``context_window`` / ``top_k``, no
SIGALRM deadlines, no streaming early return. Memory methods receive raw messages only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

import yaml

from evals.skill_bench import protocol
from evals.skill_bench.envs.agent_context import (
    CURRENT_STEP_DESCRIPTION,
    MEMORY_DESCRIPTION,
    STEP_COUNT_DESCRIPTION,
    PromptSection,
    history_description,
    single_turn_sections,
)
from evals.skill_bench.envs.agent_context import render as render_agent_context
from evals.skill_bench.envs.base import (
    SPLITS,
    Action,
    Message,
    Reset,
    Score,
    Split,
    StepResult,
    Task,
)
from evals.skill_bench.envs.vendor.alfworld_projection import alfworld_projection

BENCHMARK = "alfworld"
TEMPLATE_DIR = "envs/templates/alfworld"
SPLIT_DIR = "envs/splits/alfworld"
CONFIG_FILE = f"{TEMPLATE_DIR}/config_tw.yaml"
GAMEFILE_HASHES = f"{SPLIT_DIR}/gamefiles.sha256.json"
VENDOR_FILES = ("envs/vendor/alfworld_projection.py",)
DATA_ENV_VAR = "ALFWORLD_DATA"
DEFAULT_DATA_ROOT = "~/.cache/alfworld"  # alfworld-download's default location

#: TextWorld ``info`` keys this module may receive. ``won`` goes to the scorer,
#: ``admissible_commands`` to the agent, ``extra.gamefile`` is dropped. Any other key
#: (for example ``facts`` or ``extra.expert_plan`` from AlfredExpert) is refused.
ALLOWED_INFO_KEYS = frozenset({"won", "admissible_commands", "extra.gamefile"})

_T = TypeVar("_T")


class AlfWorldEnvError(RuntimeError):
    """The game or its data is unusable (a harness failure, never a model failure)."""


# ---------------------------------------------------------------------------
# Frozen templates and their rendering (SkillOpt 79124b3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlfWorldTemplates:
    """The frozen actor-prompt pieces (``envs/templates/alfworld``)."""

    system_prompt: str
    no_history: str
    with_history: str
    history_length: int
    history_entry: str
    history_joiner: str
    admissible_item: str
    admissible_joiner: str
    admissible_excluded: tuple[str, ...]
    task_marker: str
    memory_slot: str
    memory_joiner: str
    action_tag_pattern: str
    fallback_empty_response: str
    fallback_missing_action: str
    fallback_call_error: str
    observation_message: str
    observation_message_joiner: str

    @classmethod
    def load(cls, root: Path = protocol.SKILL_BENCH_DIR) -> AlfWorldTemplates:
        protocol.verify_frozen_assets(TEMPLATE_DIR + "/", root)
        spec = json.loads(
            protocol.read_protocol_text(f"{TEMPLATE_DIR}/format.json", root)
        )["templates"]
        return cls(
            system_prompt=spec["system_prompt"],
            no_history=protocol.read_protocol_text(
                f"{TEMPLATE_DIR}/{spec['no_history_file']}", root
            ),
            with_history=protocol.read_protocol_text(
                f"{TEMPLATE_DIR}/{spec['with_history_file']}", root
            ),
            history_length=int(spec["history_length"]),
            history_entry=spec["history_entry"],
            history_joiner=spec["history_joiner"],
            admissible_item=spec["admissible_item"],
            admissible_joiner=spec["admissible_joiner"],
            admissible_excluded=tuple(spec["admissible_excluded"]),
            task_marker=spec["task_marker"],
            memory_slot=spec["memory_slot"],
            memory_joiner=spec["memory_joiner"],
            action_tag_pattern=spec["action_tag_pattern"],
            fallback_empty_response=spec["fallback_empty_response"],
            fallback_missing_action=spec["fallback_missing_action"],
            fallback_call_error=spec["fallback_call_error"],
            observation_message=spec["observation_message"],
            observation_message_joiner=spec["observation_message_joiner"],
        )


def format_admissible_actions(t: AlfWorldTemplates, actions: Sequence[str]) -> str:
    """SkillOpt ``env_manager.py:92-94``: ``'a'`` items joined by ``"\\n "``, without ``help``."""
    return t.admissible_joiner.join(
        t.admissible_item.format(action=a)
        for a in actions
        if a not in t.admissible_excluded
    )


def format_action_history(
    t: AlfWorldTemplates, history: Sequence[Mapping[str, str]]
) -> tuple[str, int]:
    """SkillOpt ``memory.py:73-84``: the last ``history_length`` pairs, numbered from 1."""
    recent = list(history[-t.history_length :]) if t.history_length > 0 else []
    start = len(history) - len(recent)
    lines = [
        t.history_entry.format(
            step_num=start + j + 1, obs=entry["observation"], act=entry["action"]
        )
        for j, entry in enumerate(recent)
    ]
    return t.history_joiner.join(lines), len(recent)


def insert_memory(t: AlfWorldTemplates, prompt: str, memory: str) -> str:
    """Q21 memory slot at the start of the user prompt (SkillOpt ``rollout.py:48-57, 218-220``).

    The method's text replaces SkillOpt's whole skill document, without SkillOpt's
    ``## Skill Knowledge`` header. An empty or whitespace-only memory injects nothing.
    """
    if not memory or not memory.strip():
        return prompt
    return t.memory_slot.format(memory=memory) + t.memory_joiner + prompt


def render_user_prompt(
    t: AlfWorldTemplates, fields: Mapping[str, Any], memory: str = ""
) -> str:
    """The actor's user prompt for the current state (SkillOpt ``env_manager.py:82-112``)."""
    history: Sequence[Mapping[str, str]] = fields["history"]
    admissible = format_admissible_actions(t, fields["admissible_actions"])
    if not history or t.history_length <= 0:
        body = t.no_history.format(
            current_observation=fields["observation"],
            admissible_actions=admissible,
        )
    else:
        action_history, valid_len = format_action_history(t, history)
        body = t.with_history.format(
            task_description=fields["task_description"],
            step_count=len(history),
            history_length=valid_len,
            action_history=action_history,
            current_step=len(history) + 1,
            current_observation=fields["observation"],
            admissible_actions=admissible,
        )
    return insert_memory(t, body, memory)


def render_actor_messages(
    t: AlfWorldTemplates, fields: Mapping[str, Any], memory: str = ""
) -> list[Message]:
    """``[system, user]`` exactly as SkillOpt's ``chat_target(system=..., user=...)`` sends them."""
    return [
        {"role": "system", "content": t.system_prompt},
        {"role": "user", "content": render_user_prompt(t, fields, memory)},
    ]


def agent_context_sections(t: AlfWorldTemplates) -> tuple[PromptSection, ...]:
    """The actor's messages as :func:`render_actor_messages` builds them, each with its
    per-step values as ``{name}``: the system message, then the user message of
    step 1 (``rollout_no_history.md``) and of the later steps
    (``rollout_with_history.md``, ``{history_length}`` filled in), each behind the Q21
    memory slot."""
    slot = t.memory_slot + t.memory_joiner
    return (
        PromptSection("system", 1, None, t.system_prompt),
        *single_turn_sections(
            "user",
            slot + t.no_history,
            slot + t.with_history,
            t.history_length,
        ),
    )


def agent_context(t: AlfWorldTemplates) -> str:
    """The actor's prompt as a static, label-free text for memory methods
    (``envs/agent_context.py``)."""
    return render_agent_context(
        agent_context_sections(t),
        {
            "memory": MEMORY_DESCRIPTION,
            "current_observation": "the environment's latest observation",
            "admissible_actions": "the actions allowed at this step, each in single "
            "quotes",
            "task_description": "the task the first observation states",
            "step_count": STEP_COUNT_DESCRIPTION,
            "action_history": history_description(t.history_length, t.history_entry),
            "current_step": CURRENT_STEP_DESCRIPTION,
        },
    )


def observation_message(
    t: AlfWorldTemplates, observation: str, admissible: Sequence[str]
) -> Message:
    """Q20 method-visible user message: raw observation plus the admissible actions."""
    listed = t.observation_message_joiner.join(
        a for a in admissible if a not in t.admissible_excluded
    )
    return {
        "role": "user",
        "content": t.observation_message.format(
            observation=observation, admissible_actions=listed
        ),
    }


def extract_task_description(t: AlfWorldTemplates, observation: str) -> str:
    """SkillOpt ``env_manager.py:74-80``: the stripped text after ``Your task is to: ``."""
    start = observation.find(t.task_marker)
    if start == -1:
        raise AlfWorldEnvError("Task description not found in text observation.")
    return observation[start + len(t.task_marker) :].strip()


@dataclass(frozen=True)
class ParsedResponse:
    """How one raw model response becomes a game command.

    ``response`` is what SkillOpt hands to the projection (the stripped response, or
    a ``look`` fallback); ``command`` is the executed action; ``format_error`` is True
    when a fallback was used; ``action_valid`` is SkillRL's projection validity
    (``<think>`` present, no CJK), kept as a diagnostic only, as in SkillOpt.
    """

    response: str
    command: str
    format_error: bool
    action_valid: bool


def parse_response(t: AlfWorldTemplates, raw: str | None) -> ParsedResponse:
    """SkillOpt ``rollout.py:238-243`` followed by ``alfworld_projection``."""
    response = (raw or "").strip()
    format_error = True
    if not response:
        response = t.fallback_empty_response
    elif re.search(t.action_tag_pattern, response, re.DOTALL) is None:
        response = t.fallback_missing_action
    else:
        format_error = False
    commands, valids = alfworld_projection([response], [[]])
    return ParsedResponse(response, commands[0], format_error, bool(valids[0]))


# ---------------------------------------------------------------------------
# Split manifests, data paths, seeds (as in SkillOpt)
# ---------------------------------------------------------------------------


def load_split(
    split: Split, root: Path = protocol.SKILL_BENCH_DIR
) -> list[dict[str, str]]:
    """Rows of SkillOpt's ``alfworld_path_split/<split>/items.json`` (byte-exact copy)."""
    if split not in SPLITS:
        raise ValueError(f"unknown ALFWorld split {split!r}; expected one of {SPLITS}")
    rows = json.loads(protocol.read_protocol_text(f"{SPLIT_DIR}/{split}.json", root))
    if not isinstance(rows, list):
        raise AlfWorldEnvError(f"{SPLIT_DIR}/{split}.json must be a JSON list")
    items: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not all(
            isinstance(row.get(k), str) and row[k]
            for k in ("id", "gamefile", "task_type")
        ):
            raise AlfWorldEnvError(
                f"{split} item {index} needs non-empty id, gamefile and task_type"
            )
        items.append({k: row[k] for k in ("id", "gamefile", "task_type")})
    if len({row["id"] for row in items}) != len(items):
        raise AlfWorldEnvError(f"duplicate ids in the {split} manifest")
    return items


def default_data_root() -> Path:
    """``$ALFWORLD_DATA``, else alfworld-download's default ``~/.cache/alfworld``."""
    raw = os.environ.get(DATA_ENV_VAR) or DEFAULT_DATA_ROOT
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def resolve_gamefile(gamefile: str, data_root: str | os.PathLike[str]) -> Path:
    """Manifest paths are relative to ``$ALFWORLD_DATA``."""
    path = Path(os.path.expandvars(os.path.expanduser(gamefile)))
    if not path.is_absolute():
        path = Path(os.path.expandvars(os.path.expanduser(os.fspath(data_root)))) / path
    return path.resolve(strict=False)


def eval_dataset_for(gamefile: str) -> str:
    """ALFWorld ``train_eval`` of a game (SkillOpt ``dataloader.py:44-52``)."""
    if "/valid_seen/" in gamefile:
        return "eval_in_distribution"
    if "/valid_unseen/" in gamefile:
        return "eval_out_of_distribution"
    return "train"


def game_seed(gamefile: str) -> int:
    """Per-game TextWorld seed: ``SEED + int(sha256(gamefile)[:7], 16)``.

    ``gamefile`` is the manifest path (relative to ``$ALFWORLD_DATA``), so the seed does
    not depend on the machine, the split order, the batch or the method.
    """
    return protocol.SEED + int(
        hashlib.sha256(gamefile.encode("utf-8")).hexdigest()[:7], 16
    )


def load_gamefile_hashes(
    root: Path = protocol.SKILL_BENCH_DIR,
) -> dict[str, dict[str, str]]:
    """The pinned sha256 of every manifest game (``game`` and ``traj_data``), by manifest path."""
    protocol.verify_frozen_assets(GAMEFILE_HASHES, root)
    data = json.loads(protocol.read_protocol_text(GAMEFILE_HASHES, root))
    return {str(k): dict(v) for k, v in data["games"].items()}


def load_shared_hashes(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, str]:
    """The pinned sha256 of the data release's shared files (``logic/``), by relative path."""
    protocol.verify_frozen_assets(GAMEFILE_HASHES, root)
    data = json.loads(protocol.read_protocol_text(GAMEFILE_HASHES, root))
    return {str(k): str(v) for k, v in data["shared"].items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gamefile_problems(path: Path, expected: Mapping[str, str] | None) -> list[str]:
    """Why the game at ``path`` differs from its pinned hashes (empty list = identical)."""
    if expected is None:
        return [f"{path} is not a manifest game"]
    problems = []
    for key, target in (("game", path), ("traj_data", path.parent / "traj_data.json")):
        if not target.is_file():
            problems.append(f"{target} missing")
        elif _file_sha256(target) != expected.get(key):
            problems.append(f"{target} differs from the pinned sha256")
    return problems


def package_problems() -> list[str]:
    """Installed game-engine packages that differ from ``protocol.ALFWORLD_PACKAGES``."""
    from importlib import metadata

    problems = []
    for name, pinned in sorted(protocol.ALFWORLD_PACKAGES.items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed (protocol pins {pinned})")
            continue
        if installed != pinned:
            problems.append(f"{name} {installed} is installed, protocol pins {pinned}")
    return problems


def load_config(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, Any]:
    """The frozen ``config_tw.yaml``, checked for the settings this module relies on."""
    protocol.verify_frozen_assets(CONFIG_FILE, root)
    config = yaml.safe_load(protocol.read_protocol_text(CONFIG_FILE, root))
    problems = []
    if config["env"]["type"] != "AlfredTWEnv":
        problems.append("env.type must be AlfredTWEnv")
    if config["env"]["domain_randomization"]:
        problems.append("env.domain_randomization must be False")
    if config["general"]["training_method"] != "dqn":
        problems.append("general.training_method must be 'dqn' (no AlfredExpert)")
    if (
        config["rl"]["training"]["max_nb_steps_per_episode"]
        != protocol.ALFWORLD_MAX_STEPS
    ):
        problems.append(
            f"rl.training.max_nb_steps_per_episode must be {protocol.ALFWORLD_MAX_STEPS}"
        )
    if problems:
        raise protocol.ProtocolDriftError(f"{CONFIG_FILE}: " + "; ".join(problems))
    return config


# ---------------------------------------------------------------------------
# Game backend (TextWorld, synchronous; always called from a worker thread)
# ---------------------------------------------------------------------------


class GameBackend(Protocol):
    """One loaded game. All methods block; the episode calls them via ``asyncio.to_thread``."""

    def reset(self) -> tuple[str, Mapping[str, Any]]: ...

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]: ...

    def close(self) -> None: ...


#: ``factory(gamefile_path, seed, train_eval) -> GameBackend`` (runs in a worker thread).
BackendFactory = Callable[[Path, int, str], GameBackend]


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, (list, tuple)) else value


class TextWorldBackend:
    """One ALFWorld game in this process, through ALFWorld's own ``AlfredTWEnv.init_env``.

    SkillOpt's worker builds ``AlfredTWEnv(config, train_eval)``, which walks and parses
    every game of the split, and then overrides ``game_files`` with the one gamefile.
    This backend skips the walk and keeps everything ``init_env`` does: the demangler
    and info wrappers, ``EnvInfos(won, admissible_commands, extras=["gamefile"])`` and
    the config's step limit. With ``training_method: dqn`` no AlfredExpert wrapper is
    added.

    Not thread-safe across games: with ``batch_size=1`` TextWorld plays in-process
    (``SyncBatchEnv``) and parses games with module-global parsers, so two games loading
    at once in one process corrupt each other. Episodes therefore use
    :class:`ProcessBackend`, which runs this class in a process of its own.
    """

    def __init__(
        self, gamefile: Path, seed: int, train_eval: str, *, config: Mapping[str, Any]
    ) -> None:
        _check_gamefile(gamefile)
        try:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise AlfWorldEnvError(
                "ALFWorld is not installed; install the 'skill-bench' extra"
            ) from exc
        base = AlfredTWEnv.__new__(AlfredTWEnv)
        base.config = dict(config)
        base.train_eval = train_eval
        base.game_files = [str(gamefile)]
        base.num_games = 1
        self._env: Any = base.init_env(batch_size=1)
        self._env.seed(seed)

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        obs, infos = self._env.reset()
        return str(_first(obs)), {k: _first(v) for k, v in infos.items()}

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        obs, _scores, dones, infos = self._env.step([command])
        return (
            str(_first(obs)),
            bool(_first(dones)),
            {k: _first(v) for k, v in infos.items()},
        )

    def close(self) -> None:
        env, self._env = self._env, None
        if env is not None:
            env.close()


#: Longest wait for one reply of a game process (start-up, reset or step). TextWorld
#: answers in milliseconds to a few seconds; a silent process is killed.
GAME_CALL_TIMEOUT_S = 120.0
GAME_CLOSE_TIMEOUT_S = 10.0
GAME_SERVER_FLAG = "--serve-game"
_CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def game_process_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The game process's environment: the parent's, minus anything credential-like."""
    source = os.environ if base is None else base
    return {
        k: v
        for k, v in source.items()
        if not any(marker in k.upper() for marker in _CREDENTIAL_MARKERS)
    }


def serve_game() -> int:
    """Game-process main loop (``python -m evals.skill_bench.envs.alfworld --serve-game``).

    The port of SkillOpt's ``vendor/alfworld_envs.py:_worker_loop``, speaking JSON lines:
    the first line on stdin names the game, then each request line gets exactly one reply
    line. File descriptor 1 is pointed at ``/dev/null`` first, because TextWorld and its
    planner print to stdout; replies go to a private duplicate of the original stdout.
    """
    import sys

    channel = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    requests = sys.stdin.buffer

    def reply(message: Mapping[str, Any]) -> None:
        # default=repr: unexpected info (e.g. an expert's facts) still reaches the
        # parent, which refuses it by key.
        channel.write(json.dumps(message, default=repr).encode("utf-8") + b"\n")

    backend: TextWorldBackend | None = None
    try:
        header = json.loads(requests.readline())
        backend = TextWorldBackend(
            Path(header["gamefile"]),
            int(header["seed"]),
            str(header["train_eval"]),
            config=header["config"],
        )
    except BaseException as exc:
        reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    reply({"ok": True})
    try:
        for line in requests:  # EOF when the parent goes away
            request = json.loads(line)
            if request["cmd"] == "close":
                reply({"ok": True})
                return 0
            try:
                if request["cmd"] == "reset":
                    obs, info = backend.reset()
                    result: list[Any] = [obs, dict(info)]
                elif request["cmd"] == "step":
                    obs, done, info = backend.step(str(request["arg"]))
                    result = [obs, done, dict(info)]
                else:
                    raise ValueError(f"unknown game command {request['cmd']!r}")
            except BaseException as exc:
                reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            else:
                reply({"ok": True, "result": result})
        return 0
    finally:
        try:
            backend.close()
        except Exception:
            pass


class ProcessBackend:
    """One game in a child Python process, like SkillOpt's ``_ProcessWorker``.

    Each game gets its own interpreter, so concurrent episodes never share TextWorld's
    module-global state and a crashing game cannot take the harness down. The child is
    started with ``subprocess`` (``python -m evals.skill_bench.envs.alfworld
    --serve-game``) rather than ``multiprocessing``: a ``spawn`` child re-imports the
    launcher's ``__main__`` module (the whole harness, or nothing at all when it runs
    from stdin) and ``fork`` is unsafe from the worker threads that start games. The
    child gets no credential-like variables and exits when its stdin closes, so a killed
    harness leaves no games behind. Every call blocks; episodes run them through
    ``asyncio.to_thread``.
    """

    def __init__(
        self,
        gamefile: Path,
        seed: int,
        train_eval: str,
        *,
        config: Mapping[str, Any],
        call_timeout_s: float = GAME_CALL_TIMEOUT_S,
    ) -> None:
        import subprocess
        import sys

        _check_gamefile(gamefile)
        self._timeout = call_timeout_s
        self._closed = False
        self._buffer = bytearray()
        self._process = subprocess.Popen(
            [sys.executable, "-m", "evals.skill_bench.envs.alfworld", GAME_SERVER_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(protocol.REPO_ROOT),
            env=game_process_env(),
        )
        header = {
            "gamefile": str(gamefile),
            "seed": seed,
            "train_eval": train_eval,
            "config": dict(config),
        }
        try:
            self._send(header)
            self._receive()  # the game is loaded
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int:
        return self._process.pid

    def _send(self, message: Mapping[str, Any]) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            stdin.flush()
        except (OSError, ValueError) as exc:
            self._kill()
            raise AlfWorldEnvError(
                f"ALFWorld game process is gone (exit code {self._process.poll()})"
            ) from exc

    def _receive(self, timeout_s: float | None = None) -> Any:
        import select
        import time

        stdout = self._process.stdout
        assert stdout is not None
        timeout = self._timeout if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise AlfWorldEnvError(
                    f"ALFWorld game process did not answer within {timeout:g} s"
                )
            ready, _, _ = select.select([stdout], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(stdout.fileno(), 1 << 16)
            if not chunk:
                self._kill()
                raise AlfWorldEnvError(
                    f"ALFWorld game process died (exit code {self._process.poll()})"
                )
            self._buffer += chunk
        line, _, rest = bytes(self._buffer).partition(b"\n")
        self._buffer = bytearray(rest)
        message = json.loads(line)
        if not message.get("ok"):
            raise AlfWorldEnvError(
                f"ALFWorld game process failed: {message.get('error')}"
            )
        return message.get("result")

    def _request(self, command: str, argument: Any = None) -> Any:
        if self._closed:
            raise AlfWorldEnvError("the game process is closed")
        self._send({"cmd": command, "arg": argument})
        return self._receive()

    def reset(self) -> tuple[str, Mapping[str, Any]]:
        obs, info = self._request("reset")
        return obs, info

    def step(self, command: str) -> tuple[str, bool, Mapping[str, Any]]:
        obs, done, info = self._request("step", command)
        return obs, bool(done), info

    def _kill(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        try:
            self._process.wait(timeout=GAME_CLOSE_TIMEOUT_S)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.poll() is None:
                self._send({"cmd": "close"})
                self._receive(GAME_CLOSE_TIMEOUT_S)
        except Exception:
            pass
        for stream in (self._process.stdin, self._process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            self._process.wait(timeout=GAME_CLOSE_TIMEOUT_S)
        except Exception:
            self._kill()


def _check_gamefile(gamefile: Path) -> None:
    """The checks ``AlfredTWEnv.collect_game_files`` applies to a game directory."""
    if not gamefile.is_file():
        raise AlfWorldEnvError(
            f"missing ALFWorld game {gamefile}; run alfworld-download and set {DATA_ENV_VAR}"
        )
    if not (gamefile.parent / "traj_data.json").is_file():
        raise AlfWorldEnvError(f"missing traj_data.json next to {gamefile}")
    try:
        solvable = json.loads(gamefile.read_text(encoding="utf-8")).get("solvable")
    except (OSError, ValueError) as exc:
        raise AlfWorldEnvError(f"unreadable ALFWorld game {gamefile}: {exc}") from exc
    if solvable is not True:
        raise AlfWorldEnvError(f"{gamefile} is not marked solvable")


# ---------------------------------------------------------------------------
# Episode and environment
# ---------------------------------------------------------------------------


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


class AlfWorldEpisode:
    """One game. ``reset`` once, then ``step`` with each raw model response until ``done``."""

    def __init__(
        self,
        *,
        templates: AlfWorldTemplates,
        gamefile: str,
        gamefile_path: Path,
        backend_factory: BackendFactory,
        max_steps: int,
    ) -> None:
        self._t = templates
        self._gamefile = gamefile
        self._path = gamefile_path
        self._factory = backend_factory
        self._max_steps = max_steps
        self._lock = asyncio.Lock()
        self._inflight: asyncio.Future[Any] | None = None
        self._backend: GameBackend | None = None
        self._started = False
        self._closed = False
        self._broken = False
        self._done = False
        self._won = False
        self._steps = 0
        self._format_errors = 0
        self._task_description = ""
        self._observation = ""
        self._admissible: tuple[str, ...] = ()
        self._history: list[Mapping[str, str]] = []

    # --- read-only state (harness side) ------------------------------------------

    @property
    def seed(self) -> int:
        return game_seed(self._gamefile)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def format_errors(self) -> int:
        return self._format_errors

    @property
    def done(self) -> bool:
        return self._done

    # --- worker-thread plumbing -------------------------------------------------

    async def _call(self, fn: Callable[..., _T], *args: Any) -> _T:
        """Run a blocking game call in a worker thread.

        The call is shielded: if the caller is cancelled, the thread still finishes and
        :meth:`close` waits for it before closing the game. A cancelled or failed call
        leaves the game in an unknown state, so the episode refuses further calls (the
        item is re-run, H8).
        """
        if self._broken:
            raise AlfWorldEnvError(
                "a game call of this episode was interrupted or failed"
            )
        future: asyncio.Future[_T] = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        self._inflight = future
        try:
            result = await asyncio.shield(future)
        except BaseException:
            self._broken = True
            raise
        if self._closed:
            raise AlfWorldEnvError("the episode was closed during a game call")
        return result

    def _checked(self, read: Callable[[], _T]) -> _T:
        """Validate a game reply; a bad reply also breaks the episode."""
        try:
            return read()
        except BaseException:
            self._broken = True
            raise

    def _build_backend(self) -> None:
        backend = self._factory(
            self._path, game_seed(self._gamefile), eval_dataset_for(self._gamefile)
        )
        # Set in the worker thread, so close() sees the game even after a cancellation.
        self._backend = backend

    def _read_info(self, info: Mapping[str, Any]) -> tuple[tuple[str, ...], bool]:
        unexpected = set(info) - ALLOWED_INFO_KEYS
        if unexpected:
            raise AlfWorldEnvError(
                f"unexpected TextWorld info keys {sorted(unexpected)} (is AlfredExpert enabled?)"
            )
        commands = info.get("admissible_commands")
        if not isinstance(commands, (list, tuple)):
            raise AlfWorldEnvError("TextWorld returned no admissible_commands")
        return tuple(str(c) for c in commands), bool(info.get("won", False))

    def _fields(self, **extra: Any) -> Mapping[str, Any]:
        return _freeze(
            {
                "observation": self._observation,
                "admissible_actions": self._admissible,
                "task_description": self._task_description,
                "history": tuple(self._history),
                "step_count": self._steps,
                **extra,
            }
        )

    # --- Episode protocol ------------------------------------------------------------

    async def reset(self) -> Reset:
        async with self._lock:
            if self._closed:
                raise RuntimeError("episode is closed")
            if self._started:
                raise RuntimeError("reset() may be called only once per episode")
            self._started = True
            await self._call(self._build_backend)
            backend = self._backend
            if backend is None:  # pragma: no cover - the factory returned nothing
                raise AlfWorldEnvError("the game backend factory returned no game")
            observation, info = await self._call(backend.reset)
            self._admissible, self._won = self._checked(lambda: self._read_info(info))
            self._task_description = self._checked(
                lambda: extract_task_description(self._t, observation)
            )
            self._observation = observation
            return Reset(
                instruction=self._task_description,
                messages=(observation_message(self._t, observation, self._admissible),),
                fields=self._fields(),
                private=_freeze({"won": self._won}),
            )

    async def step(self, action: Action) -> StepResult:
        if not isinstance(action, str):
            raise TypeError("an ALFWorld action is the model's raw response text")
        async with self._lock:
            if self._closed:
                raise RuntimeError("episode is closed")
            if self._backend is None:
                raise RuntimeError("call reset() before step()")
            if self._done:
                raise RuntimeError("the episode is over")
            parsed = parse_response(self._t, action)
            observation, game_done, info = await self._call(
                self._backend.step, parsed.command
            )
            admissible, won = self._checked(lambda: self._read_info(info))
            self._history.append(
                _freeze({"observation": self._observation, "action": parsed.command})
            )
            self._steps += 1
            self._format_errors += int(parsed.format_error)
            capped = self._steps >= self._max_steps
            # The step cap was reached without a win, whichever layer ended the game:
            # TextWorld's own max_nb_steps_per_episode (50) reports game_done at the cap.
            truncated = capped and not won
            self._done = game_done or capped
            self._observation, self._admissible, self._won = (
                observation,
                admissible,
                won,
            )
            return StepResult(
                messages=(observation_message(self._t, observation, admissible),),
                done=self._done,
                fields=self._fields(
                    action=parsed.command,
                    format_error=parsed.format_error,
                    action_valid=parsed.action_valid,
                    truncated=truncated,
                ),
                private=_freeze({"won": won}),
            )

    async def close(self) -> None:
        self._closed = True
        inflight = self._inflight
        if inflight is not None and not inflight.done():
            await asyncio.wait([inflight])
        backend, self._backend = self._backend, None
        if backend is not None:
            await asyncio.to_thread(backend.close)

    async def __aenter__(self) -> AlfWorldEpisode:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class AlfWorldEnv:
    """ALFWorld tasks (SkillOpt split 39/18/134), episodes and scoring."""

    benchmark = BENCHMARK

    def __init__(
        self,
        *,
        data_root: str | os.PathLike[str] | None = None,
        backend_factory: BackendFactory | None = None,
        max_steps: int = protocol.ALFWORLD_MAX_STEPS,
        root: Path = protocol.SKILL_BENCH_DIR,
    ) -> None:
        if not 0 < max_steps <= protocol.ALFWORLD_MAX_STEPS:
            raise ValueError(f"max_steps must be in 1..{protocol.ALFWORLD_MAX_STEPS}")
        self.root = root
        self.templates = AlfWorldTemplates.load(root)
        self.config = load_config(root)
        protocol.verify_frozen_assets(SPLIT_DIR + "/", root)
        for vendored in VENDOR_FILES:  # the action projection decides what is executed
            protocol.verify_frozen_assets(vendored, root)
        self.data_root = (
            Path(data_root) if data_root is not None else default_data_root()
        )
        self.max_steps = max_steps
        self.gamefile_hashes = load_gamefile_hashes(root)
        self.shared_hashes = load_shared_hashes(root)
        self._factory: BackendFactory = backend_factory or self._verified_backend

    def _verified_backend(
        self, gamefile: Path, seed: int, train_eval: str
    ) -> GameBackend:
        """The default backend: refuses a game whose files differ from the pinned hashes."""
        expected = self._expected_hashes().get(gamefile)
        problems = package_problems() + gamefile_problems(gamefile, expected)
        if problems:
            raise AlfWorldEnvError(
                "ALFWorld game data or engine differ from the pinned ones "
                "(splits/alfworld/gamefiles.sha256.json, protocol.ALFWORLD_PACKAGES): "
                + "; ".join(problems)
            )
        return ProcessBackend(gamefile, seed, train_eval, config=self.config)

    def _expected_hashes(self) -> dict[Path, Mapping[str, str]]:
        return {
            resolve_gamefile(rel, self.data_root): hashes
            for rel, hashes in self.gamefile_hashes.items()
        }

    def verify_gamefiles(self, split: Split) -> list[str]:
        """Problems with a split's game files, the shared ``logic/`` files and the game
        engine packages against the pinned ones (blocking; preflight). Empty = pinned."""
        problems = package_problems()
        for rel, digest in sorted(self.shared_hashes.items()):
            target = self.data_root / rel
            if not target.is_file():
                problems.append(f"{target} missing")
            elif _file_sha256(target) != digest:
                problems.append(f"{target} differs from the pinned sha256")
        for task in self.load_tasks(split):
            path = self.gamefile_path(task)
            problems += gamefile_problems(
                path, self.gamefile_hashes.get(task.raw["gamefile"])
            )
        return problems

    def call_error_response(self) -> str:
        """SkillOpt's response for a model call that failed (``rollout.py:244-245``): ``look``."""
        return self.templates.fallback_call_error

    def load_tasks(self, split: Split) -> list[Task]:
        rows = load_split(split, self.root)
        expected = protocol.SPLIT_SIZES[BENCHMARK][split]
        if len(rows) != expected:
            raise protocol.ProtocolDriftError(
                f"ALFWorld {split} has {len(rows)} items, protocol says {expected}"
            )
        return [Task(BENCHMARK, split, row["id"], _freeze(row)) for row in rows]

    def gamefile_path(self, task: Task) -> Path:
        return resolve_gamefile(task.raw["gamefile"], self.data_root)

    def missing_gamefiles(self, split: Split) -> list[str]:
        """Manifest gamefiles that are not present under the data root (a preflight check)."""
        return [
            task.raw["gamefile"]
            for task in self.load_tasks(split)
            if not self.gamefile_path(task).is_file()
        ]

    def episode(self, task: Task) -> AlfWorldEpisode:
        if task.benchmark != BENCHMARK:
            raise ValueError(f"not an ALFWorld task: {task.benchmark}")
        # Only the gamefile reaches the episode; the task type stays with the scorer.
        return AlfWorldEpisode(
            templates=self.templates,
            gamefile=task.raw["gamefile"],
            gamefile_path=self.gamefile_path(task),
            backend_factory=self._factory,
            max_steps=self.max_steps,
        )

    def label_strings(self, task: Task) -> list[str]:
        """Scorer-side strings no method may ever receive for ``task`` (H1.7, H10.1).

        ``won`` itself is never passed, but see the module docstring of
        ``agent/rollout.py``: an episode shorter than 50 steps was won (no losing state).
        """
        gamefile = str(task.raw["gamefile"])
        return [
            task.raw_id,
            str(task.raw["task_type"]),
            gamefile,
            Path(gamefile).parent.name,
            Path(gamefile).parent.parent.name,
        ]

    def taxonomy_labels(self, task: Task) -> list[str]:
        """The item's ``task_type``: a category shared by many items, not an identifier of
        this one (X2)."""
        return [str(task.raw["task_type"])]

    def score(
        self, task: Task, final_answer: str | None, private: Mapping[str, Any]
    ) -> Score:
        """Success (0/1) from the last step's ``private["won"]``; ``final_answer`` is unused."""
        won = private.get("won")
        if not isinstance(won, bool):
            raise ValueError(
                "ALFWorld scoring needs the final step's private['won'] (bool)"
            )
        return Score(primary=1.0 if won else 0.0)

    def actor_messages(
        self, fields: Mapping[str, Any], memory: str = ""
    ) -> list[Message]:
        """The actor call for the current state, with ``memory`` in the Q21 slot."""
        return render_actor_messages(self.templates, fields, memory)

    def agent_context(self) -> str:
        """The actor's prompt template, the same for every item: what the rollout
        hands methods as ``TaskView.agent_context``."""
        return agent_context(self.templates)


if __name__ == "__main__":  # the game process of ProcessBackend
    import sys

    if sys.argv[1:] != [GAME_SERVER_FLAG]:
        raise SystemExit(
            f"usage: python -m evals.skill_bench.envs.alfworld {GAME_SERVER_FLAG}"
        )
    raise SystemExit(serve_game())
