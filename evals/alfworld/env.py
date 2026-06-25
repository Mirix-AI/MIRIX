"""Thin ALFWorld TextWorld wrapper used by the MIRIX eval runner."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import AlfWorldItem, resolve_item_gamefile


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "vendor" / "config_tw.yaml"


class ALFWorldDependencyError(RuntimeError):
    """Raised when the optional ALFWorld runtime is not installed."""


@dataclass(frozen=True)
class ALFWorldObservation:
    observation: str
    admissible_actions: list[str]
    info: dict[str, Any]
    task_description: str


@dataclass(frozen=True)
class ALFWorldStep:
    observation: str
    admissible_actions: list[str]
    reward: float
    done: bool
    info: dict[str, Any]
    won: bool


def train_eval_for_split(split: str) -> str:
    """Map the SkillOpt manifest split name to ALFWorld's dataset selector."""

    if split == "train":
        return "train"
    if split == "val":
        return "eval_in_distribution"
    if split == "test":
        return "eval_out_of_distribution"
    return "train"


class ALFWorldTextEnv:
    """Single-game ALFWorld TextWorld environment.

    This intentionally avoids SkillOpt's vector environment manager. It uses the
    same official ALFWorld ``AlfredTWEnv`` backend and points it at a specific
    manifest game file.
    """

    def __init__(
        self,
        *,
        gamefile: str | os.PathLike[str],
        split: str,
        seed: int = 42,
        config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.gamefile = str(Path(gamefile).resolve(strict=False))
        self.split = split
        self.seed = seed
        self.config_path = Path(config_path)
        self._env = None
        self._task_description = ""
        self._build()

    @classmethod
    def from_item(
        cls,
        item: AlfWorldItem,
        *,
        alfworld_data: str | os.PathLike[str] | None = None,
        seed: int = 42,
        config_path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH,
    ) -> "ALFWorldTextEnv":
        return cls(
            gamefile=resolve_item_gamefile(item, alfworld_data, must_exist=True),
            split=item.split,
            seed=seed,
            config_path=config_path,
        )

    def _build(self) -> None:
        try:
            import yaml
            from alfworld.agents.environment import get_environment
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise ALFWorldDependencyError(
                "ALFWorld runtime is not installed. Install eval dependencies with "
                '`pip install -e ".[eval,alfworld]"`, then run `alfworld-download` '
                "and set ALFWORLD_DATA."
            ) from exc

        with self.config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        config = _expand_env_vars(config)

        env_type = config["env"]["type"]
        train_eval = train_eval_for_split(self.split)
        base_env = get_environment(env_type)(config, train_eval=train_eval)
        base_env.game_files = [self.gamefile]
        if hasattr(base_env, "num_games"):
            base_env.num_games = 1
        self._env = base_env.init_env(batch_size=1)
        if hasattr(self._env, "seed"):
            self._env.seed(self.seed)

    def reset(self) -> ALFWorldObservation:
        assert self._env is not None
        obs, infos = self._env.reset()
        observation = _first(obs)
        info = _unwrap_info(infos)
        self._task_description = _extract_task_description(observation)
        return ALFWorldObservation(
            observation=observation,
            admissible_actions=_admissible_actions(info),
            info=info,
            task_description=self._task_description,
        )

    def step(self, action: str) -> ALFWorldStep:
        assert self._env is not None
        obs, scores, dones, infos = self._env.step([action])
        observation = _first(obs)
        info = _unwrap_info(infos)
        return ALFWorldStep(
            observation=observation,
            admissible_actions=_admissible_actions(info),
            reward=float(_first(scores) or 0.0),
            done=bool(_first(dones)),
            info=info,
            won=bool(info.get("won", False)),
        )

    def close(self) -> None:
        env = self._env
        self._env = None
        if env is not None and hasattr(env, "close"):
            env.close()


def _extract_task_description(observation: str) -> str:
    marker = "Your task is to: "
    start = observation.find(marker)
    if start == -1:
        return ""
    return observation[start + len(marker) :].strip()


def _admissible_actions(info: dict[str, Any]) -> list[str]:
    actions = info.get("admissible_commands") or []
    return [str(action) for action in actions]


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value[0] if len(value) else None
    except Exception:  # pragma: no cover - numpy optional in bare test env
        pass
    return value


def _unwrap_info(infos: Any) -> dict[str, Any]:
    if isinstance(infos, dict):
        return {key: _first(value) for key, value in infos.items()}
    if isinstance(infos, (list, tuple)) and infos:
        info = infos[0]
        if isinstance(info, dict):
            return dict(info)
    return {}


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value
