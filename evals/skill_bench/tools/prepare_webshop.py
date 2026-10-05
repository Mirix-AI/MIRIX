"""Prepare the WebShop split manifests.

Three steps:

``select`` (offline, any host)
    Draws the splits from upstream's own goal files: ``baseline_models/data/human_goals.json``
    (the 12,087 human goals in the environment's goal order, each lower-cased, without
    quotes and without its price clause: upstream ``train_choice_il.py:95-101``) and
    ``baseline_models/data/items_human_ins.json`` (each ASIN's instructions with their
    attributes and options). Both are checked against ``protocol.WEBSHOP_DATA``. The result
    is **provisional**: the goal order and each goal's ASIN are inferred, the price limits
    are not known, and the environment refuses to run on it (``WebShopEnv.preflight``)::

        python -m evals.skill_bench.tools.prepare_webshop select --upstream <WebShop clone>

``freeze`` (an x86_64 Linux host with Docker; the server image started in prepare mode)
    Reads the real goals from the server (``GET /goals``; built after ``random.seed(42)``),
    checks that there are 12,087 and that their order is ``human_goals.json``'s, draws
    the splits again from the real goal table (the selection a provisional manifest made
    must come out again, or ``--allow-selection-change`` is needed), adds every pinned
    goal field (instruction with its price limit, ASIN, attributes, options, price limit,
    name, query, category, product category), the best possible reward of each item
    (buying the gold product with the goal's options) and the gold product's rank when
    the instruction is searched, and writes the **final** manifest with the price
    table's sha256 and the image's build record (``source.json``)::

        python -m evals.skill_bench.envs.webshop --start-prepare --wait-s 1800
        python -m evals.skill_bench.tools.prepare_webshop freeze --upstream <clone> \\
            --server-url http://127.0.0.1:18560

``check``
    The frozen files are consistent: sizes, ids, draw rules, statuses (offline). With
    ``--upstream``, the draw is run again on upstream's goal files and must give the
    manifest's selection (unless ``freeze`` recorded a changed selection)::

        python -m evals.skill_bench.tools.prepare_webshop check --upstream <clone>

``preview-sizes``
    What drawing with another split size would move (offline, changes nothing): the
    test split is drawn first and every later split skips the gold ASINs and
    instructions it holds, so a larger test split can redraw val, pilot and train, and
    can turn an item development used into a test item. The test size is fixed before
    any development run.
    ``select`` and ``freeze`` warn loudly with the same record (``selection_moves`` in
    ``split_manifest.json``) whenever their selection differs from the manifest on
    disk::

        python -m evals.skill_bench.tools.prepare_webshop preview-sizes \\
            --upstream <clone> --size test=200

``pin``
    Rewrites ``protocol.WEBSHOP_ASSET_SHA256`` (the block between its ``BEGIN`` / ``END``
    comments in ``protocol.py``) from the WebShop files on disk, after ``freeze`` (or any
    change of a WebShop template, the server's Docker context or the projection);
    ``pin --check`` only says whether it would change anything. The protocol itself is
    re-frozen separately (``tools/freeze_protocol``), not here::

        python -m evals.skill_bench.tools.prepare_webshop pin

The draw (``protocol.WEBSHOP_SPLIT_DRAWS``, in ``WEBSHOP_DRAW_ORDER``): each split is
``random.Random(seed).shuffle(list(range(first, stop)))`` taken in order, skipping a goal
whose instruction names more than one ASIN, whose ASIN or normalized instruction an earlier
split holds, and, in ``unique_asin`` splits, whose ASIN the split already holds. No gold ASIN
or normalized instruction is therefore shared across splits. Items are ``fixed_<goal
index>``, sorted by goal index. Items whose best possible reward is below 1 are kept;
more than 10% of the test split is flagged in ``split_manifest.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws

#: upstream ``process_goal`` (``baseline_models/train_choice_il.py:95-101``) cuts here.
PRICE_CLAUSE = ", and price lower than"
#: Flag the selection when more of the test split than this is unsolvable.
UNSOLVABLE_FLAG_FRACTION = 0.10
DIAGNOSTICS_BATCH = 10
DIAGNOSTICS_TIMEOUT_S = 1800.0
GOALS_TIMEOUT_S = 600.0


class PrepareError(RuntimeError):
    """The inputs are not the pinned ones, or the real data contradicts the selection."""


@dataclass(frozen=True)
class Goal:
    """One goal of the table the draw runs on: its index, its gold ASIN (None when its
    instruction text names more than one ASIN: the draw skips it), and its instruction
    in upstream's ``human_goals.json`` form (``key``) and whitespace-collapsed (``norm``,
    the cross-split rule)."""

    index: int
    asin: str | None
    key: str
    norm: str


def goal_key(instruction_text: str) -> str:
    """upstream ``process_goal`` on an instruction: lower case, no quotes, stripped, cut
    at the price clause (``human_goals.json``'s form)."""
    text = instruction_text.lower().replace('"', "").replace("'", "").strip()
    if PRICE_CLAUSE in text:
        text = text.split(PRICE_CLAUSE)[0]
    return text


def normalized(key: str) -> str:
    return " ".join(key.split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream_files(upstream: Path) -> tuple[list[str], dict[str, Any]]:
    """``human_goals.json`` and ``items_human_ins.json`` of an upstream clone, refused
    unless their sha256 is the pinned one (``protocol.WEBSHOP_DATA``)."""
    human = upstream / str(protocol.WEBSHOP_DATA["human_goals"]["path"])
    instructions = upstream / "baseline_models" / "data" / "items_human_ins.json"
    for path, want in (
        (human, protocol.WEBSHOP_DATA["human_goals"]["sha256"]),
        (
            instructions,
            protocol.WEBSHOP_DATA["files"]["items_human_ins.json"]["sha256"],
        ),
    ):
        if not path.is_file():
            raise PrepareError(
                f"{path} is missing (an upstream clone at the pinned commit)"
            )
        if (found := _sha256(path)) != want:
            raise PrepareError(f"{path}: sha256 {found}, pinned {want}")
    goals = json.loads(human.read_text(encoding="utf-8"))
    items = json.loads(instructions.read_text(encoding="utf-8"))
    if not isinstance(goals, list) or not isinstance(items, dict):
        raise PrepareError("the upstream goal files do not have the expected shape")
    return goals, items


def _instruction_keys(instruction: str) -> set[str]:
    """The ``human_goals.json`` forms a human instruction can take: upstream appends the
    price clause to ``instruction.strip('.')`` (``goal.py:57``) before ``process_goal``
    strips the whole text, so the instruction's own end is stripped only when no price
    clause was added."""
    base = instruction.strip(".").lower().replace('"', "").replace("'", "")
    return {base.lstrip(), base.strip()}


def offline_table(
    human_goals: Sequence[str], items_human_ins: Mapping[str, Any]
) -> list[Goal]:
    """The goal table from upstream's files: each goal's ASIN is the one instruction of
    ``items_human_ins.json`` (with attributes, as upstream keeps, ``goal.py:31-34``) whose
    ``human_goals.json`` form it is; a text several ASINs share is ambiguous."""
    asins: dict[str, set[str]] = defaultdict(set)
    for asin, instructions in items_human_ins.items():
        for entry in instructions:
            if entry.get("instruction_attributes"):
                for key in _instruction_keys(str(entry["instruction"])):
                    asins[key].add(str(asin))
    table = []
    for index, key in enumerate(human_goals):
        found = asins.get(key)
        if not found:
            raise PrepareError(f"goal {index} matches no human instruction: {key!r}")
        table.append(
            Goal(
                index,
                next(iter(found)) if len(found) == 1 else None,
                key,
                normalized(key),
            )
        )
    return table


def real_table(goals: Sequence[Mapping[str, Any]]) -> list[Goal]:
    """The goal table from the server's goals: their own ASINs; a text (``human_goals.json``
    form) that goals of several ASINs share is ambiguous, as offline."""
    keys = [goal_key(str(g["instruction_text"])) for g in goals]
    asins: dict[str, set[str]] = defaultdict(set)
    for key, goal in zip(keys, goals, strict=True):
        asins[key].add(str(goal["asin"]))
    return [
        Goal(
            index,
            str(goal["asin"]) if len(asins[key]) == 1 else None,
            key,
            normalized(key),
        )
        for index, (key, goal) in enumerate(zip(keys, goals, strict=True))
    ]


def protocol_sizes() -> dict[str, int]:
    """Each WebShop split's size as the protocol fixes it (test, val, train, pilot)."""
    return {
        split: int(size)
        for split, size in {
            **protocol.SPLIT_SIZES["webshop"],
            **protocol.EXTRA_SPLIT_SIZES["webshop"],
        }.items()
    }


def draw_splits(
    table: Sequence[Goal], sizes: Mapping[str, int] | None = None
) -> dict[str, list[int]]:
    """The selection (see the module docstring), each split sorted by goal index.
    ``sizes`` overrides some splits' sizes (a preview of another test size,
    :func:`size_change`); by default the protocol's."""
    taken_asins: set[str] = set()
    taken_norms: set[str] = set()
    selection: dict[str, list[int]] = {}
    sizes = {**protocol_sizes(), **(sizes or {})}
    for split in protocol.WEBSHOP_DRAW_ORDER:
        draw = protocol.WEBSHOP_SPLIT_DRAWS[split]
        order = list(range(int(draw["first"]), min(int(draw["stop"]), len(table))))
        random.Random(int(draw["seed"])).shuffle(order)
        picked: list[int] = []
        own: set[str] = set()
        for index in order:
            goal = table[index]
            if (
                goal.asin is None
                or goal.asin in taken_asins
                or goal.norm in taken_norms
            ):
                continue
            if draw["unique_asin"] and goal.asin in own:
                continue
            picked.append(index)
            own.add(goal.asin)
            if len(picked) == sizes[split]:
                break
        if len(picked) != sizes[split]:
            raise PrepareError(
                f"only {len(picked)} {split} goals satisfy the draw rules, the protocol "
                f"needs {sizes[split]}"
            )
        taken_asins |= {str(table[i].asin) for i in picked}
        taken_norms |= {table[i].norm for i in picked}
        selection[split] = sorted(picked)
    return selection


#: The splits development reads (development uses train and val; the pilot is played
#: before it). A test item that shares a gold ASIN or an instruction with one of theirs
#: is an item development may have seen.
DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "val", "pilot")


def selection_moves(
    before: Mapping[str, Sequence[int]],
    after: Mapping[str, Sequence[int]],
    table: Sequence[Goal],
) -> dict[str, Any]:
    """What a new selection does to an earlier one (pure):
    ``moved`` per split whose goals differ, the goals that left (``out``) and entered
    (``in``); ``development_items_in_test``, each test goal the new selection adds whose
    gold ASIN or normalized instruction a train, val or pilot goal of the earlier
    selection held (development may have seen it), with that goal and what they
    share; and ``development_splits_moved``, the train, val and pilot splits that
    changed (a test-size change redraws them, ``WEBSHOP_DRAW_ORDER``)."""
    moved = {
        split: {
            "out": sorted(set(before.get(split, ())) - set(after.get(split, ()))),
            "in": sorted(set(after.get(split, ())) - set(before.get(split, ()))),
        }
        for split in ws.SPLIT_NAMES
        if set(before.get(split, ())) != set(after.get(split, ()))
    }
    seen = [
        (split, index)
        for split in DEVELOPMENT_SPLITS
        for index in before.get(split, ())
        if 0 <= index < len(table)
    ]
    shared = []
    for index in (moved.get("test") or {}).get("in", ()):
        goal = table[index]
        for split, other in seen:
            held = table[other]
            same_asin = goal.asin is not None and goal.asin == held.asin
            same_instruction = goal.norm == held.norm
            if same_asin or same_instruction:
                shared.append(
                    {
                        "test_goal": index,
                        "split": split,
                        "goal": other,
                        "same_asin": same_asin,
                        "same_instruction": same_instruction,
                    }
                )
    return {
        "moved": moved,
        "development_splits_moved": [s for s in DEVELOPMENT_SPLITS if s in moved],
        "development_items_in_test": shared,
    }


def moves_warning(moves: Mapping[str, Any]) -> str | None:
    """The loud warning for :func:`selection_moves` (None when nothing moved): record,
    do not block; the test size is fixed before any development run."""
    moved = moves.get("moved") or {}
    if not moved:
        return None
    parts = [
        f"{split}: {len(m['out'])} out {m['out'][:5]}, {len(m['in'])} in {m['in'][:5]}"
        for split, m in moved.items()
    ]
    text = "WARNING: the selection changed (" + "; ".join(parts) + ")."
    dev = moves.get("development_splits_moved") or []
    if dev:
        text += (
            f" Splits development reads changed ({', '.join(dev)}): a cell that "
            "developed or learned on the earlier items did not on these."
        )
    shared = moves.get("development_items_in_test") or []
    if shared:
        text += (
            f" {len(shared)} new test item(s) share a gold ASIN or instruction with a "
            "train, val or pilot item of the earlier selection (development may have "
            "seen them): "
            + ", ".join(
                f"test goal {s['test_goal']} ~ {s['split']} goal {s['goal']}"
                for s in shared[:5]
            )
            + "."
        )
    return (
        text + " The test size is fixed before any development run: keep the earlier "
        "selection unless the operator decides otherwise."
    )


def size_change(table: Sequence[Goal], sizes: Mapping[str, int]) -> dict[str, Any]:
    """What drawing with ``sizes`` (e.g. another test size) instead of the
    protocol's would move (:func:`selection_moves`), with both selections' counts."""
    before = draw_splits(table)
    after = draw_splits(table, sizes)
    return {
        "sizes_before": protocol_sizes(),
        "sizes_after": {**protocol_sizes(), **dict(sizes)},
        **selection_moves(before, after, table),
    }


def order_problems(
    goals: Sequence[Mapping[str, Any]], human_goals: Sequence[str]
) -> list[str]:
    """How the server's goals differ from the count and order the protocol assumes."""
    problems = []
    if len(goals) != protocol.WEBSHOP_GOAL_COUNT:
        problems.append(
            f"the server built {len(goals)} goals, the protocol expects "
            f"{protocol.WEBSHOP_GOAL_COUNT}"
        )
    differ = [
        i
        for i, (goal, key) in enumerate(zip(goals, human_goals, strict=False))
        if goal_key(str(goal["instruction_text"])) != key
    ]
    if differ:
        problems.append(
            f"{len(differ)} goals are not in human_goals.json's order (first: goal "
            f"{differ[0]})"
        )
    return problems


def _provisional_row(goal: Goal, items_human_ins: Mapping[str, Any]) -> dict[str, Any]:
    entry = next(
        e
        for e in items_human_ins[str(goal.asin)]
        if e.get("instruction_attributes")
        and goal.key in _instruction_keys(str(e["instruction"]))
    )
    return {
        "id": ws.item_id(goal.index),
        "goal_index": goal.index,
        "asin": goal.asin,
        "instruction": goal.key,
        "attributes": list(entry["instruction_attributes"]),
        "goal_options": list(entry["instruction_options"]),
    }


def _final_row(
    goal: Goal, real: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "id": ws.item_id(goal.index),
        "goal_index": goal.index,
        "asin": goal.asin,
        "instruction": goal.key,
        **{name: real.get(name) for name in ws.MANIFEST_FIELDS if name != "asin"},
        "best_reward": diagnostics.get("best_reward"),
        "options_available": diagnostics.get("options_available"),
        "gold_search_rank": diagnostics.get("gold_search_rank"),
    }


def _split_manifest(
    status: str,
    selection: Mapping[str, Sequence[int]],
    table: Sequence[Goal],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": "WebShop",
        "status": status,
        "source_repo": protocol.WEBSHOP_SERVER["upstream_repo"],
        "source_commit": protocol.WEBSHOP_SERVER["upstream_commit"],
        "split_method": (
            "Each split is random.Random(seed).shuffle of its goal "
            "range taken in order, skipping a goal whose instruction names more than one "
            "ASIN, whose ASIN or normalized instruction an earlier split holds, and in "
            "unique_asin splits one whose ASIN the split holds (tools/prepare_webshop.py)"
        ),
        "draws": {k: dict(v) for k, v in protocol.WEBSHOP_SPLIT_DRAWS.items()},
        "draw_order": list(protocol.WEBSHOP_DRAW_ORDER),
        "counts": {split: len(selection[split]) for split in ws.SPLIT_NAMES},
        "ambiguous_goals_skipped": sum(1 for g in table if g.asin is None),
        "item_fields": [
            "id",
            "goal_index",
            "asin",
            "instruction",
            *[n for n in ws.MANIFEST_FIELDS if n != "asin"],
            "best_reward",
            "options_available",
            "gold_search_rank",
        ],
        **extra,
    }


def write_manifest(
    out_dir: Path,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ws.SPLIT_NAMES:
        _write_json(out_dir / f"{split}.json", list(rows[split]))
    _write_json(out_dir / "split_manifest.json", dict(manifest))
    _write_json(out_dir / "source.json", dict(source))


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def select(upstream: Path, out_dir: Path) -> dict[str, list[int]]:
    """The provisional manifest from upstream's goal files (offline)."""
    human_goals, items = load_upstream_files(upstream)
    if len(human_goals) != protocol.WEBSHOP_GOAL_COUNT:
        raise PrepareError(
            f"human_goals.json holds {len(human_goals)} goals, the protocol expects "
            f"{protocol.WEBSHOP_GOAL_COUNT}"
        )
    table = offline_table(human_goals, items)
    selection = draw_splits(table)
    before = current_selection(out_dir)
    moves = (
        selection_moves(before, selection, table)
        if before is not None and before != selection
        else None
    )
    warning = moves_warning(moves) if moves else None
    if warning:
        print(warning, file=sys.stderr)
    rows = {
        split: [_provisional_row(table[i], items) for i in selection[split]]
        for split in ws.SPLIT_NAMES
    }
    manifest = _split_manifest(
        ws.MANIFEST_PROVISIONAL,
        selection,
        table,
        {
            **({"selection_moves": moves} if moves else {}),
            "notes": [
                "Provisional: drawn offline from upstream's human_goals.json (goal order) "
                "and items_human_ins.json (ASINs, attributes, options). The price limits "
                "and the product fields are added, and the selection checked against the "
                "real data, by tools/prepare_webshop.py freeze on the server image.",
                "The environment refuses to run on a provisional manifest.",
            ],
        },
    )
    source = {
        "comment": "Where the WebShop items come from. A provisional manifest "
        "records upstream's goal files only; freeze adds the real data's pins.",
        "status": ws.MANIFEST_PROVISIONAL,
        "upstream_commit": protocol.WEBSHOP_SERVER["upstream_commit"],
        "human_goals_sha256": protocol.WEBSHOP_DATA["human_goals"]["sha256"],
        "items_human_ins_sha256": protocol.WEBSHOP_DATA["files"][
            "items_human_ins.json"
        ]["sha256"],
        "goal_count": len(human_goals),
    }
    write_manifest(out_dir, rows, manifest, source)
    return selection


def current_selection(out_dir: Path) -> dict[str, list[int]] | None:
    """The goal indices the manifest in ``out_dir`` holds, None without one."""
    try:
        return {
            split: [
                int(row["goal_index"])
                for row in json.loads((out_dir / f"{split}.json").read_text())
            ]
            for split in ws.SPLIT_NAMES
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def fetch_goals(
    server_url: str, factory: ws.ClientFactory = ws.default_client
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The prepare-mode server's ``/ready`` info and its goals (async)."""
    client = ws.ServerClient(server_url, timeout_s=GOALS_TIMEOUT_S, factory=factory)
    try:
        code, ready = await client.ready()
        if ready.get("state") != "prepare":
            raise PrepareError(
                f"the server at {server_url} is {ready.get('state')!r}, not 'prepare': "
                "start it with python -m evals.skill_bench.envs.webshop --start-prepare"
            )
        goals = (await client.call("GET", "/goals")).get("goals")
    finally:
        await client.aclose()
    if not isinstance(goals, list):
        raise PrepareError("the server's /goals holds no goal list")
    return dict(ready.get("info") or {}), goals


async def fetch_diagnostics(
    server_url: str,
    indices: Sequence[int],
    factory: ws.ClientFactory = ws.default_client,
) -> dict[int, dict[str, Any]]:
    """Each goal's best possible reward and the gold product's search rank."""
    client = ws.ServerClient(
        server_url, timeout_s=DIAGNOSTICS_TIMEOUT_S, factory=factory
    )
    found: dict[int, dict[str, Any]] = {}
    try:
        for start in range(0, len(indices), DIAGNOSTICS_BATCH):
            batch = list(indices[start : start + DIAGNOSTICS_BATCH])
            reply = await client.call("POST", "/diagnostics", {"indices": batch})
            for key, value in (reply.get("items") or {}).items():
                found[int(key)] = dict(value)
    finally:
        await client.aclose()
    missing = sorted(set(indices) - set(found))
    if missing:
        raise PrepareError(f"no diagnostics for goals {missing[:5]}")
    return found


async def freeze(
    upstream: Path,
    out_dir: Path,
    server_url: str,
    *,
    allow_selection_change: bool = False,
    factory: ws.ClientFactory = ws.default_client,
) -> dict[str, Any]:
    """The final manifest from the real goals (see the module docstring); returns the
    split manifest it wrote."""
    human_goals, _items = load_upstream_files(upstream)
    info, goals = await fetch_goals(server_url, factory)
    problems = order_problems(goals, human_goals)
    if info.get("seed") != protocol.SEED:
        problems.append(
            f"the server's seed is {info.get('seed')}, protocol {protocol.SEED}"
        )
    build = info.get("build") if isinstance(info.get("build"), Mapping) else {}
    if build.get("upstream_commit") != protocol.WEBSHOP_SERVER["upstream_commit"]:
        problems.append(f"the server runs upstream {build.get('upstream_commit')!r}")
    want_data = {n: p["sha256"] for n, p in protocol.WEBSHOP_DATA["files"].items()}
    if build.get("data") != want_data:
        problems.append("the server's data files are not the pinned ones")
    index = build.get("index_sha256")
    if not (isinstance(index, str) and re.fullmatch(r"[0-9a-f]{64}", index)):
        problems.append(f"the server's build record has no index hash ({index!r})")
    if build.get("context") != ws.context_digest():
        problems.append(
            "the server image was not built from this checkout's server directory "
            f"(context {build.get('context')!r}): python -m evals.skill_bench.envs.webshop "
            "--build"
        )
    if problems:
        raise PrepareError("; ".join(problems))
    table = real_table(goals)
    selection = draw_splits(table)
    before = current_selection(out_dir)
    changed = before is not None and before != selection
    moves = selection_moves(before, selection, table) if before and changed else None
    if changed and not allow_selection_change:
        moved = {
            split: sorted(set(selection[split]) ^ set(before[split]))
            for split in ws.SPLIT_NAMES
            if before and set(selection[split]) != set(before[split])
        }
        raise PrepareError(
            "the real data draws another selection than the manifest on disk "
            f"(goals in or out: {moved}); look at why, then pass --allow-selection-change"
            + (f". {moves_warning(moves)}" if moves else "")
        )
    if moves:
        print(moves_warning(moves), file=sys.stderr)
    indices = sorted({i for split in ws.SPLIT_NAMES for i in selection[split]})
    diagnostics = await fetch_diagnostics(server_url, indices, factory)
    rows = {
        split: [
            _final_row(table[i], goals[i], diagnostics[i]) for i in selection[split]
        ]
        for split in ws.SPLIT_NAMES
    }
    test_best = [diagnostics[i].get("best_reward") for i in selection["test"]]
    unsolvable = sum(1 for r in test_best if not isinstance(r, (int, float)) or r < 1.0)
    fraction = unsolvable / max(1, len(test_best))
    manifest = _split_manifest(
        ws.MANIFEST_FINAL,
        selection,
        table,
        {
            "selection_changed_from_provisional": changed,
            **({"selection_moves": moves} if moves else {}),
            "unsolvable": {
                split: sum(
                    1
                    for i in selection[split]
                    if not isinstance(diagnostics[i].get("best_reward"), (int, float))
                    or diagnostics[i]["best_reward"] < 1.0
                )
                for split in ws.SPLIT_NAMES
            },
            "unsolvable_test_fraction": round(fraction, 4),
            # Flagged for a person to review when more than 10% of test is unsolvable.
            "unsolvable_test_flag": fraction > UNSOLVABLE_FLAG_FRACTION,
            "gold_not_found_by_search": {
                split: sum(
                    1
                    for i in selection[split]
                    if diagnostics[i].get("gold_search_rank") is None
                )
                for split in ws.SPLIT_NAMES
            },
            "notes": [
                "Final: checked against the real goals of the server image (goal count, "
                "order equal to human_goals.json, seed 42 before SimServer), with every "
                "pinned goal field; the server refuses to serve when any differs.",
                "best_reward is the reward of buying the gold product with the goal's "
                "options; gold_search_rank is its 0-based rank among the top 50 when the "
                "instruction is searched (null: not found). Items are not filtered.",
            ],
        },
    )
    source = {
        "comment": "Where the WebShop items come from: the real data's pins, "
        "the price table the server must rebuild and the image's build record.",
        "status": ws.MANIFEST_FINAL,
        "upstream_commit": build.get("upstream_commit"),
        "human_goals_sha256": protocol.WEBSHOP_DATA["human_goals"]["sha256"],
        "items_human_ins_sha256": protocol.WEBSHOP_DATA["files"][
            "items_human_ins.json"
        ]["sha256"],
        "goal_count": len(goals),
        "seed": info.get("seed"),
        "price_table_sha256": info.get("price_table_sha256"),
        "data": build.get("data"),
        "index_sha256": build.get("index_sha256"),
        "spacy_model": info.get("spacy_model"),
        "fuzzy_backend": info.get("fuzzy_backend"),
        "products": info.get("products"),
    }
    write_manifest(out_dir, rows, manifest, source)
    return manifest


#: The WebShop files ``protocol.WEBSHOP_ASSET_SHA256`` pins (``pin``).
PIN_DIRS: tuple[str, ...] = (ws.SPLIT_DIR, ws.TEMPLATE_DIR, ws.SERVER_DIR)
PIN_BLOCK = re.compile(
    r"# BEGIN WEBSHOP_ASSET_SHA256\n.*?# END WEBSHOP_ASSET_SHA256", re.DOTALL
)


def asset_hashes(root: Path = protocol.SKILL_BENCH_DIR) -> dict[str, str]:
    """sha256 of every fingerprinted WebShop file under ``root``, by protocol path."""
    found: dict[str, str] = {}
    for directory in PIN_DIRS:
        for path in sorted((root / directory).iterdir()):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and protocol.is_fingerprinted(rel):
                found[rel] = _sha256(path)
    for rel in ws.VENDOR_FILES:
        found[rel] = _sha256(root / rel)
    return found


def pinned_block(hashes: dict[str, str]) -> str:
    lines = [f'    "{rel}": "{digest}",' for rel, digest in hashes.items()]
    return "\n".join(
        [
            "# BEGIN WEBSHOP_ASSET_SHA256",
            "WEBSHOP_ASSET_SHA256: Mapping[str, str] = {",
            *lines,
            "}",
            "# END WEBSHOP_ASSET_SHA256",
        ]
    )


def pin(
    root: Path = protocol.SKILL_BENCH_DIR, *, write: bool = True
) -> dict[str, tuple[str | None, str | None]]:
    """Re-pin ``protocol.WEBSHOP_ASSET_SHA256`` in ``root/protocol.py`` from the files on
    disk; returns what changed (path -> (pinned, on disk)), and writes only with
    ``write``."""
    source = root / "protocol.py"
    text = source.read_text(encoding="utf-8")
    match = PIN_BLOCK.search(text)
    if match is None:
        raise PrepareError(f"{source} has no WEBSHOP_ASSET_SHA256 block to rewrite")
    before = dict(re.findall(r'"([^"]+)": "([0-9a-f]{64})"', match.group(0)))
    after = asset_hashes(root)
    changed = {
        rel: (before.get(rel), after.get(rel))
        for rel in sorted(set(before) | set(after))
        if before.get(rel) != after.get(rel)
    }
    if changed and write:
        updated = text[: match.start()] + pinned_block(after) + text[match.end() :]
        tmp = source.with_suffix(".py.tmp")
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(source)
    return changed


def check(out_dir: Path, upstream: Path | None = None) -> list[str]:
    """Why the manifest in ``out_dir`` is not a consistent one (empty: it is). With
    ``upstream`` (a clone), the draw also runs again on its goal files and must give the
    manifest's selection, unless ``freeze`` recorded that the real data drew another
    (``selection_changed_from_provisional``)."""
    problems: list[str] = []
    try:
        manifest = json.loads((out_dir / "split_manifest.json").read_text())
        source = json.loads((out_dir / "source.json").read_text())
        rows = {
            split: json.loads((out_dir / f"{split}.json").read_text())
            for split in ws.SPLIT_NAMES
        }
    except (OSError, ValueError) as exc:
        return [f"unreadable manifest: {exc}"]
    status = manifest.get("status")
    if (
        status not in (ws.MANIFEST_FINAL, ws.MANIFEST_PROVISIONAL)
        or source.get("status") != status
    ):
        problems.append(f"status {status!r} / source {source.get('status')!r}")
    sizes = {**protocol.SPLIT_SIZES["webshop"], **protocol.EXTRA_SPLIT_SIZES["webshop"]}
    seen_asins: dict[str, str] = {}
    seen_norms: dict[str, str] = {}
    for split in ws.SPLIT_NAMES:
        items = rows[split]
        if len(items) != sizes[split]:
            problems.append(f"{split}: {len(items)} items, protocol {sizes[split]}")
        draw = protocol.WEBSHOP_SPLIT_DRAWS[split]
        own: set[str] = set()
        for row in items:
            index, asin = row.get("goal_index"), row.get("asin")
            if row.get("id") != ws.item_id(index if isinstance(index, int) else -1):
                problems.append(f"{split}: bad id {row.get('id')!r}")
                continue
            if not int(draw["first"]) <= index < int(draw["stop"]):
                problems.append(f"{split}: goal {index} outside its range")
            norm = normalized(str(row.get("instruction", "")))
            for seen, value, what in (
                (seen_asins, str(asin), "ASIN"),
                (seen_norms, norm, "instruction"),
            ):
                other = seen.get(value)
                if other is not None and other != split:
                    problems.append(
                        f"{split}: goal {index}'s {what} is also in {other}"
                    )
            if draw["unique_asin"] and str(asin) in own:
                problems.append(f"{split}: ASIN {asin} twice")
            own.add(str(asin))
            if status == ws.MANIFEST_FINAL and not all(
                n in row for n in ws.MANIFEST_FIELDS
            ):
                problems.append(f"{split}: goal {index} lacks a final field")
        for row in items:
            seen_asins.setdefault(str(row.get("asin")), split)
            seen_norms.setdefault(normalized(str(row.get("instruction", ""))), split)
    if upstream is not None and not manifest.get("selection_changed_from_provisional"):
        try:
            human_goals, instructions = load_upstream_files(upstream)
            drawn = draw_splits(offline_table(human_goals, instructions))
        except PrepareError as exc:
            problems.append(f"the draw cannot be run again: {exc}")
        else:
            for split in ws.SPLIT_NAMES:
                held = [row.get("goal_index") for row in rows[split]]
                if held != drawn[split]:
                    moved = sorted(set(map(str, held)) ^ set(map(str, drawn[split])))
                    problems.append(
                        f"{split}: the draw gives other goals (in or out: {moved[:10]})"
                    )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the WebShop split manifests.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "freeze"):
        step = sub.add_parser(name)
        step.add_argument(
            "--upstream",
            type=Path,
            required=True,
            help="a clone of princeton-nlp/WebShop at protocol.WEBSHOP_SERVER's commit",
        )
        step.add_argument(
            "--out", type=Path, default=protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR
        )
        if name == "freeze":
            step.add_argument("--server-url", default=ws.default_server_url())
            step.add_argument("--allow-selection-change", action="store_true")
    checker = sub.add_parser("check")
    checker.add_argument(
        "--out", type=Path, default=protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR
    )
    checker.add_argument(
        "--upstream",
        type=Path,
        help="a WebShop clone: run the draw again and compare the selection",
    )
    preview = sub.add_parser(
        "preview-sizes",
        help="what another split size (e.g. the test size) would move between splits, "
        "drawn offline from upstream's goal files (changes nothing)",
    )
    preview.add_argument("--upstream", type=Path, required=True)
    preview.add_argument(
        "--size",
        action="append",
        default=[],
        metavar="SPLIT=N",
        help="a split's size to preview, e.g. test=200 (repeatable)",
    )
    pinner = sub.add_parser("pin", help="re-pin protocol.WEBSHOP_ASSET_SHA256")
    pinner.add_argument("--check", action="store_true", help="only report a change")
    pinner.add_argument("--root", type=Path, default=protocol.SKILL_BENCH_DIR)
    args = parser.parse_args(argv)
    if args.command == "pin":
        try:
            changed = pin(args.root, write=not args.check)
        except PrepareError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for rel, (old, new) in changed.items():
            print(f"{rel}: {old or '-'} -> {new or '-'}")
        if not changed:
            print("protocol.WEBSHOP_ASSET_SHA256 is current")
        return 1 if changed and args.check else 0
    if args.command == "preview-sizes":
        try:
            sizes = _sizes_argument(args.size)
            human_goals, instructions = load_upstream_files(args.upstream)
            preview_data = size_change(offline_table(human_goals, instructions), sizes)
        except (PrepareError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(preview_data, indent=1, sort_keys=True))
        warning = moves_warning(preview_data)
        if warning:
            print(warning, file=sys.stderr)
        return 0
    try:
        if args.command == "select":
            selection = select(args.upstream, args.out)
            print(
                "provisional manifest: "
                + ", ".join(f"{s} {len(v)}" for s, v in selection.items())
            )
        elif args.command == "freeze":
            manifest = asyncio.run(
                freeze(
                    args.upstream,
                    args.out,
                    args.server_url,
                    allow_selection_change=args.allow_selection_change,
                )
            )
            print(
                json.dumps(
                    {
                        k: manifest[k]
                        for k in ("counts", "unsolvable", "unsolvable_test_flag")
                    }
                )
            )
            if manifest["unsolvable_test_flag"]:
                print(
                    "warning: more than 10% of the test items are unsolvable: review "
                    "the selection before any run",
                    file=sys.stderr,
                )
            print(
                "next: python -m evals.skill_bench.tools.prepare_webshop pin (the "
                "protocol is re-frozen separately)"
            )
        problems = check(
            args.out, args.upstream if args.command in ("select", "check") else None
        )
    except (PrepareError, ws.WebShopServerError, ws.WebShopEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


def _sizes_argument(values: Sequence[str]) -> dict[str, int]:
    """``--size SPLIT=N`` options as ``{split: N}`` (``ValueError`` on a bad one)."""
    sizes: dict[str, int] = {}
    for value in values:
        split, sep, number = value.partition("=")
        if not sep or split not in ws.SPLIT_NAMES or not number.isdigit():
            raise ValueError(
                f"--size {value!r}: give SPLIT=N, a split of {ws.SPLIT_NAMES}"
            )
        sizes[split] = int(number)
    if not sizes:
        raise ValueError("name at least one --size SPLIT=N")
    return sizes


if __name__ == "__main__":
    raise SystemExit(main())
