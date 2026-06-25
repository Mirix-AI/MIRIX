"""Tests for MIRIX ALFWorld adapter payload helpers."""

from __future__ import annotations

import re

from evals.alfworld.mirix_adapter import (
    build_add_sync_payload,
    build_consolidation_session_id,
    build_episode_session_id,
    render_episode_for_memory,
)


def test_build_add_sync_payload_uses_session_id_without_filter_tags() -> None:
    payload = build_add_sync_payload(
        meta_agent_id="agent-meta",
        user_id="eval-user",
        session_id="alfworld-run-ep-0001",
        user_content="trajectory",
        assistant_content="result",
    )

    assert payload["meta_agent_id"] == "agent-meta"
    assert payload["user_id"] == "eval-user"
    assert payload["session_id"] == "alfworld-run-ep-0001"
    assert payload["chaining"] is True
    assert payload["use_cache"] is True
    assert payload["messages"] == [
        {"role": "user", "content": "trajectory"},
        {"role": "assistant", "content": "result"},
    ]
    assert "filter_tags" not in payload


def test_mirix_session_ids_match_server_validator() -> None:
    episode_id = build_episode_session_id("run:with/slashes and 汉字", 12)
    boundary_id = build_consolidation_session_id("run:with/slashes and 汉字")

    assert episode_id == "alfworld-run_with_slashes_and-ep-0012"
    assert boundary_id == "alfworld-run_with_slashes_and-boundary"
    assert re.fullmatch(r"[A-Za-z0-9_-]+", episode_id)
    assert re.fullmatch(r"[A-Za-z0-9_-]+", boundary_id)
    assert len(episode_id) <= 64
    assert len(boundary_id) <= 64


def test_render_episode_for_memory_includes_outcome_after_episode() -> None:
    user_content, assistant_content = render_episode_for_memory(
        {
            "id": "train:0001",
            "task_type": "look_at_obj_in_light",
            "task_description": "examine the alarmclock with the desklamp",
            "gamefile": "json_2.1.1/train/foo/game.tw-pddl",
            "success": True,
            "n_steps": 1,
            "steps": [
                {
                    "step": 0,
                    "observation": "You are in a room.",
                    "admissible_actions": ["look", "use desklamp"],
                    "model_response": "<think>x</think><action>look</action>",
                    "action": "look",
                    "next_observation": "You see a desk.",
                    "reward": 0.0,
                    "done": False,
                }
            ],
        }
    )

    assert "Benchmark: ALFWorld" not in user_content
    assert "Episode id: train:0001" in user_content
    assert "Executed action: look" in user_content
    assert "Episode result: success" in assistant_content


def test_render_episode_for_memory_compacts_long_trajectories() -> None:
    steps = [
        {
            "step": idx,
            "observation": "obs " * 400,
            "admissible_actions": ["look", "go to desk 1"],
            "model_response": "<think>" + ("reason " * 400) + "</think><action>look</action>",
            "action": "look",
            "next_observation": "next " * 400,
            "reward": 0.0,
            "done": idx == 49,
            "won": False,
        }
        for idx in range(50)
    ]

    user_content, _ = render_episode_for_memory(
        {
            "id": "train:0002",
            "task_type": "look_at_obj_in_light",
            "task_description": "examine the cd with the desklamp",
            "gamefile": "json_2.1.1/train/foo/game.tw-pddl",
            "success": False,
            "n_steps": 50,
            "steps": steps,
        }
    )

    assert "Action sequence:" in user_content
    assert "0:look" in user_content
    assert "49:look[done]" in user_content
    assert "[... skipped middle steps; total steps: 50 ...]" in user_content
    assert "Step 25:" not in user_content
    assert len(user_content) < 20000
