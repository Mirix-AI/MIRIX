"""Data helpers for the ALFWorld eval harness.

The harness uses the same split manifest shape as SkillOpt:

    data/alfworld_path_split/{train,val,test}/items.json

Each item contains ``id``, ``gamefile``, and ``task_type``.  Gamefile values are
stored relative to ``$ALFWORLD_DATA`` unless already absolute.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class AlfWorldItem:
    """One ALFWorld manifest row."""

    id: str
    gamefile: str
    task_type: str
    split: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], split: str) -> "AlfWorldItem":
        missing = [name for name in ("id", "gamefile", "task_type") if name not in data]
        if missing:
            raise ValueError(f"{split} item missing required fields: {', '.join(missing)}")

        return cls(
            id=str(data["id"]),
            gamefile=str(data["gamefile"]),
            task_type=str(data["task_type"]),
            split=split,
        )


def resolve_gamefile(
    gamefile: str | os.PathLike[str],
    alfworld_data: str | os.PathLike[str] | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a manifest ``gamefile`` against ``$ALFWORLD_DATA``.

    Absolute paths are returned after user/env expansion.  Relative paths use
    ``alfworld_data`` when supplied, otherwise the ``ALFWORLD_DATA`` environment
    variable.  ``must_exist`` can be enabled by callers that want an early
    dataset installation check.
    """

    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(gamefile))))
    if not path.is_absolute():
        root = alfworld_data if alfworld_data is not None else os.environ.get("ALFWORLD_DATA")
        if not root:
            raise ValueError(
                "ALFWORLD_DATA is required to resolve relative ALFWorld gamefile paths"
            )
        path = Path(os.path.expandvars(os.path.expanduser(os.fspath(root)))) / path

    resolved = path.resolve(strict=False)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_item_gamefile(
    item: AlfWorldItem,
    alfworld_data: str | os.PathLike[str] | None = None,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``item.gamefile`` against ``$ALFWORLD_DATA``."""

    return resolve_gamefile(item.gamefile, alfworld_data, must_exist=must_exist)


def load_split_manifest(
    manifest_root: str | os.PathLike[str],
    split: str,
    *,
    filename: str = "items.json",
) -> list[AlfWorldItem]:
    """Load one SkillOpt-compatible split manifest."""

    if split not in SPLITS:
        raise ValueError(f"unknown ALFWorld split {split!r}; expected one of {SPLITS}")

    path = Path(manifest_root) / split / filename
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = _coerce_items_payload(payload, path)
    return [AlfWorldItem.from_mapping(row, split) for row in rows]


def load_split(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load a single manifest file as dictionaries.

    This is a compatibility helper for callers that already have the concrete
    JSON file path.  It accepts the SkillOpt ``items.json`` shape and the older
    single-file ``{"tasks": [...]}`` sample shape, preserving extra fields.
    """

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = _coerce_items_payload(payload, manifest_path)
    return [dict(row) for row in rows]


def load_manifest(
    manifest_root: str | os.PathLike[str],
    splits: Sequence[str] = SPLITS,
    *,
    require_all: bool = False,
) -> list[AlfWorldItem]:
    """Load multiple ALFWorld split manifests.

    Missing split files are skipped by default so small local fixtures can load
    a subset.  Set ``require_all=True`` for production manifest validation.
    """

    items: list[AlfWorldItem] = []
    root = Path(manifest_root)
    for split in splits:
        path = root / split / "items.json"
        if not path.exists():
            if require_all:
                raise FileNotFoundError(path)
            continue
        items.extend(load_split_manifest(root, split))
    return items


def summarize_manifest(items: Iterable[AlfWorldItem]) -> dict[str, Any]:
    """Return split and task-type distributions for loaded manifest items."""

    by_split: Counter[str] = Counter()
    by_task_type: Counter[str] = Counter()
    by_split_task_type: dict[str, Counter[str]] = defaultdict(Counter)

    total = 0
    for item in items:
        total += 1
        by_split[item.split] += 1
        by_task_type[item.task_type] += 1
        by_split_task_type[item.split][item.task_type] += 1

    return {
        "total": total,
        "by_split": dict(sorted(by_split.items())),
        "by_task_type": dict(sorted(by_task_type.items())),
        "by_split_task_type": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(by_split_task_type.items())
        },
    }


def summarize_splits(splits: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize a mapping of split name to raw manifest rows."""

    summary: dict[str, Any] = {}
    for split, rows in splits.items():
        task_types: Counter[str] = Counter()
        total = 0
        for row in rows:
            total += 1
            task_type = row.get("task_type")
            if task_type is not None:
                task_types[str(task_type)] += 1

        summary[split] = {
            "total": total,
            "task_types": dict(sorted(task_types.items())),
        }
    return summary


def _coerce_items_payload(payload: Any, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        rows = payload["items"]
    elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    else:
        raise ValueError(
            f"{path} must be a JSON list or an object with an items/tasks list"
        )

    bad_index = next(
        (idx for idx, row in enumerate(rows) if not isinstance(row, Mapping)),
        None,
    )
    if bad_index is not None:
        raise ValueError(f"{path} item {bad_index} must be a JSON object")
    return rows
