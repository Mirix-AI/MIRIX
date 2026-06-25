"""Tests for ALFWorld runner control-flow helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from evals.alfworld.runner import (
    select_items,
    should_consolidate,
    should_consolidate_final_remainder,
    should_ingest,
)


def _write_split(root: Path, split: str, count: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"{split}:{idx:04d}",
            "gamefile": f"json_2.1.1/{split}/game-{idx}/game.tw-pddl",
            "task_type": "pick_and_place_simple",
        }
        for idx in range(count)
    ]
    (split_dir / "items.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )


def test_select_items_zero_uses_full_remaining_split(tmp_path: Path) -> None:
    _write_split(tmp_path, "test", 3)

    items = select_items(
        manifest_root=tmp_path,
        split="test",
        episodes=0,
        offset=1,
        shuffle=False,
        seed=42,
    )

    assert [item.id for item in items] == ["test:0001", "test:0002"]


def test_frozen_memory_does_not_ingest_or_consolidate() -> None:
    args = Namespace(memory_mode="frozen", consolidate_every=5)
    mirix = object()

    assert should_ingest(args, mirix) is False
    assert should_consolidate(args, mirix, 5) is False


def test_online_memory_ingests_and_consolidates_on_boundary() -> None:
    args = Namespace(
        memory_mode="online",
        consolidate_every=5,
        consolidate_final_remainder=False,
    )
    mirix = object()

    assert should_ingest(args, mirix) is True
    assert should_consolidate(args, mirix, 4) is False
    assert should_consolidate(args, mirix, 5) is True
    assert should_consolidate_final_remainder(args, mirix, 9) is False


def test_final_remainder_consolidates_only_partial_online_batch() -> None:
    args = Namespace(
        memory_mode="online",
        consolidate_every=5,
        consolidate_final_remainder=True,
    )
    mirix = object()

    assert should_consolidate_final_remainder(args, mirix, 10) is False
    assert should_consolidate_final_remainder(args, mirix, 14) is True

    frozen_args = Namespace(
        memory_mode="frozen",
        consolidate_every=5,
        consolidate_final_remainder=True,
    )
    assert should_consolidate_final_remainder(frozen_args, mirix, 14) is False
