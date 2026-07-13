"""Tests for ALFWorld runner control-flow helpers."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from evals.alfworld.runner import (
    consolidate,
    select_items,
    should_consolidate,
    should_consolidate_final_remainder,
    should_ingest,
)


class _FakeAdapter:
    """Captures seal/auto_dream calls the way runner.consolidate drives them."""

    def __init__(self) -> None:
        self.seal_calls: list[dict] = []
        self.auto_dream_calls: list[dict] = []

    def seal_for_consolidation(self, *, run_id: str, after_episode: int) -> dict:
        self.seal_calls.append({"run_id": run_id, "after_episode": after_episode})
        return {"success": True}

    def auto_dream(self, *, last_n_sessions: int, model: str | None) -> dict:
        self.auto_dream_calls.append(
            {"last_n_sessions": last_n_sessions, "model": model}
        )
        return {"skills_changed": 1, "message": "ok"}


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


def test_consolidate_seals_per_boundary_and_adds_sentinel_slack() -> None:
    """The previous round's sentinel session occupies one slot in this round's
    distillation batch, so sealing must widen last_n_sessions by one."""
    fake = _FakeAdapter()

    event = consolidate(
        mirix=fake,
        run_id="run-x",
        after_episode=10,
        last_n_sessions=5,
        seal_before=True,
        model=None,
    )

    assert fake.seal_calls == [{"run_id": "run-x", "after_episode": 10}]
    assert fake.auto_dream_calls == [{"last_n_sessions": 6, "model": None}]
    assert event["sealed"] is True
    assert event["last_n_sessions"] == 6
    assert event["requested_last_n_sessions"] == 5
    assert event["skills_changed"] == 1


def test_consolidate_without_sealing_keeps_requested_batch_size() -> None:
    fake = _FakeAdapter()

    event = consolidate(
        mirix=fake,
        run_id="run-x",
        after_episode=5,
        last_n_sessions=5,
        seal_before=False,
        model="openai/gpt-5.2",
    )

    assert fake.seal_calls == []
    assert fake.auto_dream_calls == [{"last_n_sessions": 5, "model": "openai/gpt-5.2"}]
    assert event["sealed"] is False
    assert event["last_n_sessions"] == 5
