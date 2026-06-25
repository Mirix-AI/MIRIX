"""Tests for ALFWorld eval data helpers."""

from __future__ import annotations

import json
from pathlib import Path

from evals.alfworld.data import load_split, resolve_gamefile, summarize_splits


def _write_skillopt_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "valid_seen.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "alfworld-skillopt-valid-seen",
                "format": "skillopt-manifest-v1",
                "split": "valid_seen",
                "tasks": [
                    {
                        "id": "valid_seen/pick_and_place/001",
                        "task_type": "pick_and_place",
                        "gamefile": "json_2.1.1/valid_seen/pick_and_place/task_001/game.tw-pddl",
                        "goal": "put a clean mug in the cabinet",
                    },
                    {
                        "id": "valid_seen/look_at_obj/002",
                        "task_type": "look_at_obj",
                        "gamefile": "json_2.1.1/valid_seen/look_at_obj/task_002/game.tw-pddl",
                        "goal": "examine a desk lamp",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def test_load_split_reads_skillopt_manifest_sample(tmp_path: Path) -> None:
    manifest = _write_skillopt_manifest(tmp_path)

    tasks = load_split(manifest)

    assert len(tasks) == 2
    assert tasks[0]["id"] == "valid_seen/pick_and_place/001"
    assert tasks[0]["task_type"] == "pick_and_place"
    assert (
        tasks[0]["gamefile"]
        == "json_2.1.1/valid_seen/pick_and_place/task_001/game.tw-pddl"
    )
    assert tasks[0]["goal"] == "put a clean mug in the cabinet"
    assert tasks[1]["task_type"] == "look_at_obj"


def test_summarize_splits_counts_task_types() -> None:
    splits = {
        "train": [
            {"task_type": "pick_and_place"},
            {"task_type": "pick_and_place"},
            {"task_type": "heat_then_place"},
        ],
        "valid_seen": [
            {"task_type": "look_at_obj"},
            {"task_type": "pick_and_place"},
        ],
        "valid_unseen": [],
    }

    summary = summarize_splits(splits)

    assert summary == {
        "train": {
            "total": 3,
            "task_types": {
                "heat_then_place": 1,
                "pick_and_place": 2,
            },
        },
        "valid_seen": {
            "total": 2,
            "task_types": {
                "look_at_obj": 1,
                "pick_and_place": 1,
            },
        },
        "valid_unseen": {
            "total": 0,
            "task_types": {},
        },
    }


def test_resolve_gamefile_uses_alfworld_data_env(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "alfworld-data"
    gamefile = (
        data_root
        / "json_2.1.1"
        / "valid_seen"
        / "pick_and_place"
        / "task_001"
        / "game.tw-pddl"
    )
    gamefile.parent.mkdir(parents=True)
    gamefile.write_text("dummy game", encoding="utf-8")
    monkeypatch.setenv("ALFWORLD_DATA", str(data_root))

    resolved = resolve_gamefile(
        "json_2.1.1/valid_seen/pick_and_place/task_001/game.tw-pddl"
    )

    assert resolved == gamefile


def test_resolve_gamefile_accepts_absolute_paths(tmp_path: Path) -> None:
    gamefile = tmp_path / "game.tw-pddl"
    gamefile.write_text("dummy game", encoding="utf-8")

    resolved = resolve_gamefile(gamefile)

    assert resolved == gamefile
