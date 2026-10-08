"""Verify the released, fixed WebShop task manifests.

``check`` validates SHA-256, counts, task fields and split isolation. Optional
``--upstream`` checks the selected goal indices against pinned upstream source files;
it never selects or replaces tasks. ``pin`` maintains asset hashes after reviewed
changes to the server or prompts. Neither command redraws or writes task manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws

PRICE_CLAUSE = ", and price lower than"


class PrepareError(RuntimeError):
    """The inputs differ from the released protocol."""


@dataclass(frozen=True)
class Goal:
    """One upstream goal used to verify a released task's source."""

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
    """Check the fixed release inputs; return problems without modifying any file."""
    problems: list[str] = []
    prefix = ws.SPLIT_DIR + "/"
    expected = {
        rel.removeprefix(prefix): digest
        for rel, digest in protocol.WEBSHOP_ASSET_SHA256.items()
        if rel.startswith(prefix)
    }
    found = {p.name for p in out_dir.glob("*.json")}
    for name in sorted(found - set(expected)):
        problems.append(f"unexpected manifest file: {name}")
    for name, digest in expected.items():
        path = out_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif _sha256(path) != digest:
            problems.append(f"{name}: SHA-256 differs from the released checksum")
    if problems:
        return problems
    try:
        manifest = json.loads((out_dir / "split_manifest.json").read_text())
        source = json.loads((out_dir / "source.json").read_text())
        rows = {
            split: json.loads((out_dir / f"{split}.json").read_text())
            for split in ws.SPLIT_NAMES
        }
    except (OSError, ValueError) as exc:
        return [f"unreadable manifest: {exc}"]
    if not isinstance(manifest, dict) or not isinstance(source, dict):
        return ["manifest and source must be JSON objects"]
    if (
        manifest.get("status") != ws.MANIFEST_FINAL
        or source.get("status") != ws.MANIFEST_FINAL
    ):
        problems.append("the released manifests must be final")
    sizes = dict(protocol.SPLIT_SIZES["webshop"])
    if manifest.get("counts") != sizes:
        problems.append("manifest counts differ from the protocol")
    split_hashes = {
        f"{split}.json": expected[f"{split}.json"] for split in ws.SPLIT_NAMES
    }
    if manifest.get("files") != split_hashes:
        problems.append("manifest checksums differ from the protocol")
    seen_asins: dict[str, str] = {}
    seen_norms: dict[str, str] = {}
    ranges = {
        "train": (1500, protocol.WEBSHOP_GOAL_COUNT),
        "val": (500, 1500),
        "test": (100, 500),
    }
    for split, items in rows.items():
        if not isinstance(items, list):
            problems.append(f"{split}: expected a JSON list")
            continue
        if len(items) != sizes[split]:
            problems.append(f"{split}: {len(items)} items, protocol {sizes[split]}")
        own_asins: set[str] = set()
        own_ids: set[str] = set()
        for row in items:
            if not isinstance(row, dict):
                problems.append(f"{split}: expected task objects")
                continue
            index, asin = row.get("goal_index"), row.get("asin")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or row.get("id") != ws.item_id(index)
            ):
                problems.append(f"{split}: bad id {row.get('id')!r}")
                continue
            if row["id"] in own_ids:
                problems.append(f"{split}: duplicate id {row['id']}")
            own_ids.add(row["id"])
            first, stop = ranges[split]
            if not first <= index < stop:
                problems.append(f"{split}: goal {index} outside its range")
            if not isinstance(asin, str) or not isinstance(row.get("instruction"), str):
                problems.append(f"{split}: goal {index} needs ASIN and instruction")
                continue
            norm = normalized(row["instruction"])
            for seen, value, what in (
                (seen_asins, asin, "ASIN"),
                (seen_norms, norm, "instruction"),
            ):
                other = seen.setdefault(value, split)
                if other != split:
                    problems.append(
                        f"{split}: goal {index}'s {what} is also in {other}"
                    )
            if split in ("train", "val") and asin in own_asins:
                problems.append(f"{split}: ASIN {asin} twice")
            own_asins.add(asin)
            if not all(n in row for n in ws.MANIFEST_FIELDS):
                problems.append(f"{split}: goal {index} lacks a final field")
    if problems or upstream is None:
        return problems
    try:
        human_goals, instructions = load_upstream_files(upstream)
        if len(human_goals) != protocol.WEBSHOP_GOAL_COUNT:
            raise PrepareError("upstream goal count differs from the protocol")
        table = offline_table(human_goals, instructions)
    except PrepareError as exc:
        return [f"cannot verify upstream source: {exc}"]
    for split, items in rows.items():
        for row in items:
            goal = table[row["goal_index"]]
            if goal.asin != row["asin"] or goal.key != row["instruction"]:
                problems.append(
                    f"{split}: goal {goal.index} differs from its upstream source"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    checker = sub.add_parser("check", help="verify fixed manifests without resampling")
    checker.add_argument(
        "--out", type=Path, default=protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR
    )
    checker.add_argument(
        "--upstream", type=Path, help="also check pinned upstream goal data"
    )
    pinner = sub.add_parser("pin", help="maintain WebShop asset checksums")
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
    problems = check(args.out, args.upstream)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    if not problems:
        print("Fixed WebShop manifests verified: train=40, val=20, test=140")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
