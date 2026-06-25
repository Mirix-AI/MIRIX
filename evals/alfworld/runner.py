"""Command-line runner for MIRIX on ALFWorld.

Default experiment shape matches the current plan:

* source manifest: SkillOpt's train split
* episodes: 10
* procedural consolidation: every 5 episodes, for 2 total consolidations
* action format: SkillOpt-compatible ``<think>`` + ``<action>``
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .actions import parse_action
from .data import AlfWorldItem, load_manifest, summarize_manifest
from .env import ALFWorldDependencyError, ALFWorldTextEnv, DEFAULT_CONFIG_PATH
from .mirix_adapter import (
    DEFAULT_CLIENT_ID,
    DEFAULT_MIRIX_URL,
    DEFAULT_ORG_ID,
    OPENROUTER_BASE_URL,
    MirixALFWorldAdapter,
    MirixAdapterError,
    build_episode_session_id,
    read_first_env_key,
    render_episode_for_memory,
)
from .prompts import SYSTEM_PROMPT, render_rollout_prompt


DEFAULT_MANIFEST_ROOT = Path(__file__).resolve().parent / "data" / "alfworld_path_split"
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"


class ChatModelError(RuntimeError):
    """Raised when the target model client cannot be initialized or called."""


class OpenAICompatibleChatModel:
    """Minimal OpenAI-compatible chat client for ALFWorld action generation."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_completion_tokens: int,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional in bare env
            raise ChatModelError(
                "openai is required for target model calls. Install eval dependencies "
                'with `pip install -e ".[eval]"`.'
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, system: str, user: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
        }
        try:
            response = self._client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        return (content or "").strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or DEFAULT_RUNS_DIR / f"{args.arm}-{safe_id(run_id)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = select_items(
        manifest_root=Path(args.manifest_root),
        split=args.split,
        episodes=args.episodes,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    run_config = vars(args) | {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "selected_items": [asdict(item) for item in items],
    }
    write_json(output_dir / "config.json", run_config)

    if args.dry_run:
        summary = {
            "dry_run": True,
            "run_id": run_id,
            "output_dir": str(output_dir),
            "items": [asdict(item) for item in items],
            "manifest_summary": summarize_manifest(load_items_for_summary(args)),
        }
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    model_api_key = read_first_env_key(args.model_api_key_env, "OPENAI_API_KEY")
    if not model_api_key:
        raise SystemExit(
            f"No API key found in {args.model_api_key_env}/OPENAI_API_KEY or .env"
        )
    model = OpenAICompatibleChatModel(
        model=args.model,
        base_url=args.model_base_url,
        api_key=model_api_key,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    mirix: MirixALFWorldAdapter | None = None
    if args.arm == "mirix":
        user_id = args.mirix_user_id or f"alfworld-{run_id}"
        mirix = MirixALFWorldAdapter(
            base_url=args.mirix_url,
            user_id=user_id,
            client_id=args.client_id,
            org_id=args.org_id,
            meta_agent_id=args.meta_agent_id,
        )
        if not args.skip_mirix_prepare:
            mirix.prepare(
                init_meta_agent=args.init_meta_agent,
                init_api_key=read_first_env_key("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
                init_model=args.consolidation_model or args.model,
            )

    episodes: list[dict[str, Any]] = []
    consolidations: list[dict[str, Any]] = []
    try:
        for episode_index, item in enumerate(items, start=1):
            episode = run_episode(
                item=item,
                episode_index=episode_index,
                run_id=run_id,
                args=args,
                model=model,
                mirix=mirix,
                output_dir=output_dir,
            )
            episodes.append(episode)
            append_jsonl(output_dir / "episodes.jsonl", episode)
            print(
                f"[ALFWorld] episode {episode_index}/{len(items)} "
                f"{'success' if episode['success'] else 'failure'} "
                f"steps={episode['n_steps']} id={item.id}",
                flush=True,
            )

            if mirix is not None and episode_index % args.consolidate_every == 0:
                event = consolidate(
                    mirix=mirix,
                    run_id=run_id,
                    after_episode=episode_index,
                    last_n_sessions=args.consolidate_every,
                    seal_before=args.seal_before_consolidation,
                    model=args.consolidation_model,
                )
                consolidations.append(event)
                append_jsonl(output_dir / "consolidations.jsonl", event)
                print(
                    f"[ALFWorld] procedural consolidation after episode "
                    f"{episode_index}: skills_changed={event.get('skills_changed')}",
                    flush=True,
                )
    finally:
        if mirix is not None:
            mirix.close()

    summary = build_summary(
        run_id=run_id,
        output_dir=output_dir,
        args=args,
        episodes=episodes,
        consolidations=consolidations,
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_episode(
    *,
    item: AlfWorldItem,
    episode_index: int,
    run_id: str,
    args: argparse.Namespace,
    model: OpenAICompatibleChatModel,
    mirix: MirixALFWorldAdapter | None,
    output_dir: Path,
) -> dict[str, Any]:
    env = ALFWorldTextEnv.from_item(
        item,
        alfworld_data=args.alfworld_data,
        seed=args.seed + episode_index,
        config_path=args.alfworld_config,
    )
    try:
        state = env.reset()
        task_description = state.task_description
        current_observation = state.observation
        admissible_actions = state.admissible_actions

        fixed_skills: list[dict[str, Any]] = []
        retrieval_errors: list[str] = []
        if mirix is not None:
            fixed_skills, error = retrieve_skills(
                mirix,
                build_retrieval_query(item, task_description, current_observation),
                top_k=args.top_k,
            )
            if error:
                retrieval_errors.append(error)

        history: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        success = False
        done = False
        fail_reason = ""

        for step_idx in range(args.max_steps):
            skills = fixed_skills
            if mirix is not None and args.retrieve_every_step:
                skills, error = retrieve_skills(
                    mirix,
                    build_retrieval_query(item, task_description, current_observation),
                    top_k=args.top_k,
                )
                if error:
                    retrieval_errors.append(error)

            prompt = render_rollout_prompt(
                current_observation=current_observation,
                admissible_actions=admissible_actions,
                task_description=task_description,
                step_count=len(history),
                history=history,
                history_length=args.history_length,
                skills=skills,
            )
            try:
                response = model.complete(system=SYSTEM_PROMPT, user=prompt)
            except Exception as exc:
                response = f"<think>model call failed: {exc}</think><action>look</action>"

            parsed = parse_action(response, allow_json=args.allow_json_actions)
            action = parsed.action
            if args.strict_admissible and action not in admissible_actions:
                action = "look"

            next_state = env.step(action)
            step_record = {
                "step": step_idx,
                "observation": current_observation,
                "admissible_actions": admissible_actions,
                "model_response": response,
                "thought": parsed.thought,
                "action": action,
                "parsed_action": parsed.action,
                "parser_source": parsed.source,
                "format_valid": parsed.format_valid,
                "used_fallback": parsed.used_fallback,
                "next_observation": next_state.observation,
                "reward": next_state.reward,
                "done": next_state.done,
                "won": next_state.won,
            }
            steps.append(step_record)
            history.append({"observation": current_observation, "action": action})

            current_observation = next_state.observation
            admissible_actions = next_state.admissible_actions
            done = next_state.done
            success = next_state.won
            if done:
                break

        if not done:
            fail_reason = f"Timeout after {args.max_steps} steps"
        elif not success:
            fail_reason = "Episode ended without completing the task"

        episode = {
            "id": item.id,
            "session_id": build_episode_session_id(run_id, episode_index),
            "episode_index": episode_index,
            "split": item.split,
            "task_type": item.task_type,
            "gamefile": item.gamefile,
            "task_description": task_description,
            "success": success,
            "hard": 1 if success else 0,
            "soft": 1.0 if success else 0.0,
            "n_steps": len(steps),
            "done": done,
            "fail_reason": fail_reason,
            "retrieved_skill_count": len(fixed_skills),
            "retrieval_errors": retrieval_errors,
            "steps": steps,
        }

        prediction_dir = output_dir / "predictions" / safe_id(item.id)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        write_json(prediction_dir / "conversation.json", steps)

        if mirix is not None:
            user_content, assistant_content = render_episode_for_memory(episode)
            ingest_result = mirix.ingest_session(
                session_id=episode["session_id"],
                user_content=user_content,
                assistant_content=assistant_content,
            )
            episode["mirix_ingest"] = ingest_result

        return episode
    finally:
        env.close()


def retrieve_skills(
    mirix: MirixALFWorldAdapter,
    query: str,
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return mirix.search_skills(query, top_k=top_k), None
    except MirixAdapterError as exc:
        return [], str(exc)


def consolidate(
    *,
    mirix: MirixALFWorldAdapter,
    run_id: str,
    after_episode: int,
    last_n_sessions: int,
    seal_before: bool,
    model: str | None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "after_episode": after_episode,
        "last_n_sessions": last_n_sessions,
        "sealed": False,
    }
    if seal_before:
        event["seal_result"] = mirix.seal_for_consolidation(run_id=run_id)
        event["sealed"] = True
    result = mirix.auto_dream(last_n_sessions=last_n_sessions, model=model)
    event["auto_dream"] = result
    if isinstance(result, dict):
        event["skills_changed"] = result.get("skills_changed")
        event["message"] = result.get("message")
    return event


def build_retrieval_query(
    item: AlfWorldItem,
    task_description: str,
    observation: str,
) -> str:
    return (
        f"Task type: {item.task_type}\n"
        f"Task: {task_description}\n"
        f"Current observation: {observation}"
    )


def select_items(
    *,
    manifest_root: Path,
    split: str,
    episodes: int,
    offset: int,
    shuffle: bool,
    seed: int,
) -> list[AlfWorldItem]:
    splits = ("train", "val", "test") if split == "all" else (split,)
    items = load_manifest(manifest_root, splits=splits, require_all=True)
    if shuffle:
        rng = random.Random(seed)
        items = list(items)
        rng.shuffle(items)
    selected = items[offset : offset + episodes]
    if len(selected) < episodes:
        raise SystemExit(
            f"Requested {episodes} episodes from {manifest_root} split={split}, "
            f"but only {len(selected)} items are available after offset={offset}."
        )
    return selected


def load_items_for_summary(args: argparse.Namespace) -> list[AlfWorldItem]:
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    return load_manifest(Path(args.manifest_root), splits=splits, require_all=True)


def build_summary(
    *,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
    episodes: Sequence[dict[str, Any]],
    consolidations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    success_count = sum(1 for episode in episodes if episode.get("success"))
    by_task: dict[str, dict[str, int]] = {}
    for episode in episodes:
        bucket = by_task.setdefault(str(episode.get("task_type", "")), {"total": 0, "success": 0})
        bucket["total"] += 1
        bucket["success"] += 1 if episode.get("success") else 0
    return {
        "run_id": run_id,
        "arm": args.arm,
        "output_dir": str(output_dir),
        "episodes": len(episodes),
        "success_count": success_count,
        "success_rate": success_count / len(episodes) if episodes else 0.0,
        "consolidations": len(consolidations),
        "consolidate_every": args.consolidate_every,
        "by_task_type": by_task,
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Run MIRIX on ALFWorld.")
    parser.add_argument("--arm", choices=("mirix", "baseline"), default="mirix")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--consolidate-every", type=int, default=5)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    parser.add_argument("--manifest-root", default=str(DEFAULT_MANIFEST_ROOT))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--alfworld-data", default=os.environ.get("ALFWORLD_DATA"))
    parser.add_argument("--alfworld-config", default=str(DEFAULT_CONFIG_PATH))

    parser.add_argument("--model", default=os.environ.get("ALFWORLD_MODEL", "openai/gpt-5.2"))
    parser.add_argument(
        "--model-base-url",
        default=os.environ.get("OPENAI_API_BASE", OPENROUTER_BASE_URL),
    )
    parser.add_argument("--model-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument(
        "--allow-json-actions",
        action="store_true",
        help="Allow JSON action fallback in addition to SkillOpt XML tags.",
    )
    parser.add_argument(
        "--strict-admissible",
        action="store_true",
        help="Replace actions outside the admissible list with `look`.",
    )

    parser.add_argument("--mirix-url", default=os.environ.get("MIRIX_URL", DEFAULT_MIRIX_URL))
    parser.add_argument("--mirix-user-id")
    parser.add_argument(
        "--client-id",
        default=os.environ.get("MIRIX_ALFWORLD_CLIENT_ID", DEFAULT_CLIENT_ID),
    )
    parser.add_argument("--org-id", default=DEFAULT_ORG_ID)
    parser.add_argument("--meta-agent-id")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--retrieve-every-step", action="store_true")
    parser.add_argument("--skip-mirix-prepare", action="store_true")
    parser.add_argument("--init-meta-agent", action="store_true")
    parser.add_argument("--consolidation-model")
    parser.add_argument(
        "--no-seal-before-consolidation",
        dest="seal_before_consolidation",
        action="store_false",
    )
    parser.set_defaults(seal_before_consolidation=True)

    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.consolidate_every <= 0:
        parser.error("--consolidate-every must be positive")
    return args


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(Path.cwd() / ".env", override=False)


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:160]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ALFWorldDependencyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
