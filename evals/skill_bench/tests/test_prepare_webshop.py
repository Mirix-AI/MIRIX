"""``tools/prepare_webshop.py`` on fixtures.

The fixture is a synthetic upstream clone of the protocol's size (12,087 goals in
``baseline_models/data/human_goals.json``, their instructions in ``items_human_ins.json``,
with ambiguous texts, ASINs several goals share and an instruction two splits could
draw), whose pins replace ``protocol.WEBSHOP_DATA``'s for the test, and a prepare-mode
server (the image's real ``server.py`` service over the scripted backend) whose goals
are those instructions with a price limit. ``select`` -> ``freeze`` -> a serve-mode
server that verifies the frozen manifest is the round trip the remote host runs.

With ``WEBSHOP_UPSTREAM_DIR`` (princeton-nlp/WebShop at 64fa2a5), ``select`` must
reproduce the repository's provisional manifest byte for byte.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.tests.webshop_support import build_record, make_shop
from evals.skill_bench.tools import prepare_webshop as P

N = protocol.WEBSHOP_GOAL_COUNT
#: Goals whose instruction text two ASINs share: ambiguous, never drawn.
AMBIGUOUS = (101, 7001)
#: One ASIN for several goals (val and train draw one goal per ASIN).
SHARED_ASIN = {
    501: "BSHARED001",
    502: "BSHARED001",
    1600: "BSHARED002",
    1601: "BSHARED002",
}
#: The same instruction (and ASIN) in the test range and the train range.
TWIN = (150, 2000)


def instruction(index: int) -> str:
    if index in AMBIGUOUS:
        return "I want the shared thing"
    if index == TWIN[1]:
        return instruction(TWIN[0])
    return f'I want "fake" product {index}'


def asin(index: int) -> str:
    if index in SHARED_ASIN:
        return SHARED_ASIN[index]
    if index == TWIN[1]:
        return asin(TWIN[0])
    return f"B{index:09d}"


def server_goal(index: int) -> dict[str, Any]:
    return {
        "asin": asin(index),
        "instruction_text": instruction(index) + ", and price lower than 50.00 dollars",
        "attributes": ["fake attribute"],
        "goal_options": ["red"],
        "price_upper": 50.0,
        "name": f"product {index}",
        "query": "fake query",
        "category": "beauty",
        "product_category": "Beauty › Fake",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("webshop_upstream")
    data = root / "baseline_models" / "data"
    data.mkdir(parents=True)
    human = [P.goal_key(server_goal(i)["instruction_text"]) for i in range(N)]
    items: dict[str, list[dict[str, Any]]] = {}
    for i in range(N):
        if i == TWIN[1]:
            continue  # the twin is the same instruction of the same product
        items.setdefault(asin(i), []).append(
            {
                "instruction": instruction(i) + ".",
                "instruction_attributes": ["fake attribute"],
                "instruction_options": ["red"],
            }
        )
    # An instruction without attributes is not a goal upstream (goal.py:31-34).
    items["B000000003"].append(
        {
            "instruction": "Ignored",
            "instruction_attributes": [],
            "instruction_options": [],
        }
    )
    (data / "human_goals.json").write_text(json.dumps(human))
    (data / "items_human_ins.json").write_text(json.dumps(items))
    return root


@pytest.fixture
def pinned(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The fixture's files as the pinned ones."""
    data = upstream / "baseline_models" / "data"
    monkeypatch.setitem(
        protocol.WEBSHOP_DATA["human_goals"],
        "sha256",
        _sha256(data / "human_goals.json"),
    )
    monkeypatch.setitem(
        protocol.WEBSHOP_DATA["files"]["items_human_ins.json"],
        "sha256",
        _sha256(data / "items_human_ins.json"),
    )
    return upstream


def prepare_shop(tmp_path: Path, goals: list[dict[str, Any]] | None = None, **kw: Any):
    return make_shop(
        tmp_path,
        goals=goals if goals is not None else [server_goal(i) for i in range(N)],
        prepare=True,
        **kw,
    )


def freeze(out: Path, shop: Any, **kw: Any) -> dict[str, Any]:
    return asyncio.run(
        P.freeze(
            kw.pop("upstream"),
            out,
            "http://webshop.test",
            factory=shop.client_factory(),
            **kw,
        )
    )


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def test_goal_key_is_upstreams_process_goal():
    assert P.goal_key(' I need "Red" shoes, and price lower than 40.00 dollars') == (
        "i need red shoes"
    )
    assert P.goal_key("It's here") == "its here"
    assert P._instruction_keys("I want x. ") == {"i want x. ", "i want x."}
    assert P._instruction_keys("I want x.") == {"i want x"}


def test_the_offline_table_marks_ambiguous_texts_and_refuses_unknown_ones():
    items = {
        "B1": [{"instruction": "Same.", "instruction_attributes": ["a"]}],
        "B2": [{"instruction": "same", "instruction_attributes": ["a"]}],
        "B3": [{"instruction": "Own", "instruction_attributes": ["a"]}],
        "B4": [{"instruction": "No attributes", "instruction_attributes": []}],
    }
    table = P.offline_table(["same", "own"], items)
    assert [(g.index, g.asin) for g in table] == [(0, None), (1, "B3")]
    with pytest.raises(P.PrepareError, match="matches no human instruction"):
        P.offline_table(["no attributes"], items)


def _table() -> list[P.Goal]:
    return [
        P.Goal(
            i,
            asin(i) if i not in AMBIGUOUS else None,
            P.goal_key(instruction(i)),
            P.normalized(P.goal_key(instruction(i))),
        )
        for i in range(N)
    ]


def test_the_draw_follows_d2():
    selection = P.draw_splits(_table())
    assert selection == P.draw_splits(_table())  # deterministic
    sizes = {**protocol.SPLIT_SIZES["webshop"], **protocol.EXTRA_SPLIT_SIZES["webshop"]}
    assert {s: len(v) for s, v in selection.items()} == sizes
    for split, indices in selection.items():
        draw = protocol.WEBSHOP_SPLIT_DRAWS[split]
        assert indices == sorted(indices)
        assert all(draw["first"] <= i < draw["stop"] for i in indices)
        assert not set(indices) & set(AMBIGUOUS)
        if draw["unique_asin"]:
            assert len({asin(i) for i in indices}) == len(indices)
    splits = list(selection)
    for a in splits:
        for b in splits:
            if a < b:
                assert not {asin(i) for i in selection[a]} & {
                    asin(i) for i in selection[b]
                }
                assert not {P.goal_key(instruction(i)) for i in selection[a]} & {
                    P.goal_key(instruction(i)) for i in selection[b]
                }
    # The draw is a seeded shuffle of the range taken in order: test_p16's first pick is
    # its shuffle's first goal; test, drawn last, takes the first goal of its
    # own shuffle that no earlier split holds.
    import random

    order = list(range(100, 500))
    random.Random(42).shuffle(order)
    assert order[0] in selection["test_p16"]
    order = list(range(100, 500))
    random.Random(44).shuffle(order)
    earlier = {
        i for split in ("test_p16", "val", "pilot", "train") for i in selection[split]
    }
    taken = {asin(i) for i in earlier}
    first = next(i for i in order if i not in AMBIGUOUS and asin(i) not in taken)
    assert first in selection["test"]


def test_too_few_drawable_goals_is_an_error():
    table = [P.Goal(i, None, f"k{i}", f"k{i}") for i in range(N)]
    with pytest.raises(P.PrepareError, match="only 0 test_p16 goals"):
        P.draw_splits(table)


def _twin_table() -> list[P.Goal]:
    """Every goal of val's range 500..899 has the gold ASIN and instruction of a goal of
    the test range 100..499 (``i`` and ``i + 400``): a larger test split draws twins of
    val's goals and moves them (as goals 282 and 1115 of the real data do)."""
    table = []
    for i in range(N):
        twin = i - 400 if 500 <= i < 900 else i
        key = f"i want product {twin}"
        table.append(P.Goal(i, f"B{twin:09d}", key, key))
    return table


def test_a_larger_test_split_redraws_development_items_into_test():
    """The draw order (test_p16, val, pilot, train, test): test is
    drawn last, so a change of its size never moves a split development reads and never
    turns a development item into a test item; test_p16, drawn first, still can move
    val, and ``size_change`` says so (record, do not block) with a loud warning."""
    table = _twin_table()
    same = P.size_change(table, {"test": protocol.SPLIT_SIZES["webshop"]["test"]})
    assert same["moved"] == {} and P.moves_warning(same) is None
    larger = P.size_change(table, {"test": 200})
    assert larger["sizes_after"]["test"] == 200
    assert set(larger["moved"]) == {"test"}
    assert not larger["development_splits_moved"]
    assert not larger["development_items_in_test"]
    with pytest.raises(P.PrepareError, match="test goals satisfy the draw rules"):
        P.size_change(table, {"test": 400})  # what earlier splits hold is never drawn
    preview = P.size_change(table, {"test_p16": 200})
    assert preview["sizes_before"] == P.protocol_sizes()
    assert "val" in preview["development_splits_moved"]
    warning = P.moves_warning(preview)
    assert warning is not None and warning.startswith("WARNING: the selection changed")
    assert "Splits development reads changed (val" in warning
    assert "The test size is fixed before any development run" in warning


def test_select_over_a_manifest_of_another_size_warns_and_records(
    pinned, tmp_path, capsys, monkeypatch
):
    out = tmp_path / "splits"
    P.select(pinned, out)
    assert "selection_moves" not in json.loads(
        (out / "split_manifest.json").read_text()
    )
    capsys.readouterr()
    sizes = {**protocol.SPLIT_SIZES, "webshop": {**protocol.SPLIT_SIZES["webshop"]}}
    sizes["webshop"]["test"] = 150
    monkeypatch.setattr(protocol, "SPLIT_SIZES", sizes)
    P.select(pinned, out)
    assert "WARNING: the selection changed (test: 0 out" in capsys.readouterr().err
    record = json.loads((out / "split_manifest.json").read_text())["selection_moves"]
    assert record["moved"]["test"]["out"] == []
    assert len(record["moved"]["test"]["in"]) == 10


def test_preview_sizes_cli(pinned, capsys):
    assert (
        P.main(["preview-sizes", "--upstream", str(pinned), "--size", "test=150"]) == 0
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["sizes_after"]["test"] == 150 and len(data["moved"]["test"]["in"]) == 10
    assert "WARNING: the selection changed" in captured.err
    assert P.main(["preview-sizes", "--upstream", str(pinned), "--size", "dev=3"]) == 2
    assert P.main(["preview-sizes", "--upstream", str(pinned)]) == 2


@pytest.mark.skipif(
    not os.environ.get("WEBSHOP_UPSTREAM_DIR"),
    reason="WEBSHOP_UPSTREAM_DIR (princeton-nlp/WebShop clone at 64fa2a5) not set",
)
def test_the_real_goals_at_test_200_would_turn_a_val_item_into_a_test_item():
    """On the real goal files: at test=200, val goal 1115 leaves val and test
    gains goal 282, with the same gold ASIN and instruction; 100..160 move nothing."""
    human, instructions = P.load_upstream_files(
        Path(os.environ["WEBSHOP_UPSTREAM_DIR"])
    )
    table = P.offline_table(human, instructions)
    for size in (100, 160):
        assert P.size_change(table, {"test": size})["development_splits_moved"] == []
    preview = P.size_change(table, {"test": 200})
    assert preview["moved"]["val"]["out"] == [1115]
    assert {
        (s["test_goal"], s["split"], s["goal"], s["same_asin"], s["same_instruction"])
        for s in preview["development_items_in_test"]
    } == {(282, "val", 1115, True, True)}


def test_order_problems():
    goals = [server_goal(i) for i in range(N)]
    human = [P.goal_key(g["instruction_text"]) for g in goals]
    assert P.order_problems(goals, human) == []
    swapped = [*goals[:5], goals[6], goals[5], *goals[7:]]
    assert P.order_problems(swapped, human) == [
        "2 goals are not in human_goals.json's order (first: goal 5)"
    ]
    assert "built 10 goals" in P.order_problems(goals[:10], human)[0]


# ---------------------------------------------------------------------------
# select -> freeze -> serve
# ---------------------------------------------------------------------------


def test_select_writes_a_consistent_provisional_manifest(pinned, tmp_path):
    out = tmp_path / "splits"
    selection = P.select(pinned, out)
    assert P.check(out) == []
    manifest = json.loads((out / "split_manifest.json").read_text())
    assert manifest["status"] == ws.MANIFEST_PROVISIONAL
    assert manifest["ambiguous_goals_skipped"] == len(AMBIGUOUS)
    assert manifest["counts"] == {s: len(v) for s, v in selection.items()}
    row = json.loads((out / "test.json").read_text())[0]
    assert set(row) == {
        "id",
        "goal_index",
        "asin",
        "instruction",
        "attributes",
        "goal_options",
    }
    assert row["id"] == f"fixed_{row['goal_index']}"
    source = json.loads((out / "source.json").read_text())
    assert source["status"] == ws.MANIFEST_PROVISIONAL and source["goal_count"] == N
    # A provisional manifest lacks what only the real data has: the env refuses it.
    assert "price_upper" not in row and "instruction_text" not in row


def test_select_refuses_files_that_are_not_the_pinned_ones(upstream, tmp_path):
    with pytest.raises(P.PrepareError, match="pinned"):
        P.select(upstream, tmp_path / "splits")
    with pytest.raises(P.PrepareError, match="missing"):
        P.select(tmp_path / "nothing", tmp_path / "splits")


def test_freeze_writes_a_final_manifest_the_server_then_verifies(pinned, tmp_path):
    out = tmp_path / "splits"
    selection = P.select(pinned, out)
    shop = prepare_shop(tmp_path / "prepare")
    manifest = freeze(out, shop, upstream=pinned)
    assert manifest["status"] == ws.MANIFEST_FINAL
    assert manifest["selection_changed_from_provisional"] is False
    assert manifest["unsolvable"] == {s: 0 for s in ws.SPLIT_NAMES}
    assert manifest["unsolvable_test_flag"] is False
    assert P.check(out) == []
    assert P.current_selection(out) == selection
    row = json.loads((out / "test.json").read_text())[0]
    goal = shop.backend.goals[row["goal_index"]]
    for name in ws.MANIFEST_FIELDS:
        assert row[name] == goal[name], name
    assert row["instruction"] == P.goal_key(goal["instruction_text"])
    assert (row["best_reward"], row["gold_search_rank"]) == (1.0, 0)
    source = json.loads((out / "source.json").read_text())
    info = shop.service.readiness()[1]["info"]
    assert source["price_table_sha256"] == info["price_table_sha256"]
    assert source["seed"] == protocol.SEED and source["goal_count"] == N
    assert source["data"] == {
        n: p["sha256"] for n, p in protocol.WEBSHOP_DATA["files"].items()
    }
    # The round trip: a serve-mode server over the same goals accepts exactly this
    # manifest, and refuses it once a goal changed.
    served = make_shop(tmp_path / "serve", goals=shop.backend.goals, manifest_dir=out)
    assert served.service.state == "ready", served.service.problems
    changed = [dict(g) for g in shop.backend.goals]
    changed[row["goal_index"]]["price_upper"] = 40.0
    refused = make_shop(tmp_path / "refuse", goals=changed, manifest_dir=out)
    assert refused.service.state == "failed"
    assert any("price_upper" in p for p in refused.service.problems)
    # The index frozen with the manifest is the index served.
    assert source["index_sha256"] == build_record()["index_sha256"]
    rebuilt = make_shop(
        tmp_path / "rebuilt",
        goals=shop.backend.goals,
        manifest_dir=out,
        record={**build_record(), "index_sha256": "f" * 64},
    )
    assert rebuilt.service.state == "failed"
    assert any("index_sha256" in p for p in rebuilt.service.problems)


def test_freeze_refuses_a_server_that_is_not_the_protocols(pinned, tmp_path):
    out = tmp_path / "splits"
    P.select(pinned, out)
    goals = [server_goal(i) for i in range(N)]
    swapped = [*goals[:300], goals[301], goals[300], *goals[302:]]
    with pytest.raises(P.PrepareError, match="order"):
        freeze(out, prepare_shop(tmp_path / "a", swapped), upstream=pinned)
    with pytest.raises(P.PrepareError, match=f"expects {N}"):
        freeze(out, prepare_shop(tmp_path / "b", goals[:-1]), upstream=pinned)
    record = {**build_record(), "upstream_commit": "abc", "data": {}}
    with pytest.raises(P.PrepareError, match="upstream 'abc'.*data files"):
        freeze(out, prepare_shop(tmp_path / "c", record=record), upstream=pinned)
    no_index = {**build_record(), "index_sha256": None}
    with pytest.raises(P.PrepareError, match="no index hash"):
        freeze(out, prepare_shop(tmp_path / "e", record=no_index), upstream=pinned)
    stale = {**build_record(), "context": "0" * 64}
    with pytest.raises(P.PrepareError, match="not built from this checkout"):
        freeze(out, prepare_shop(tmp_path / "f", record=stale), upstream=pinned)
    serving = make_shop(tmp_path / "d", goals=goals[:10], goal_count=10, splits={})
    with pytest.raises(P.PrepareError, match="not 'prepare'"):
        freeze(out, serving, upstream=pinned)
    # Nothing was written by a refused freeze.
    assert json.loads((out / "split_manifest.json").read_text())["status"] == (
        ws.MANIFEST_PROVISIONAL
    )


def test_a_selection_change_needs_the_flag(pinned, tmp_path):
    out = tmp_path / "splits"
    P.select(pinned, out)
    rows = json.loads((out / "test.json").read_text())
    rows[0] = {**rows[0], "goal_index": 499, "id": "fixed_499"}
    (out / "test.json").write_text(json.dumps(rows))
    shop = prepare_shop(tmp_path / "prepare")
    with pytest.raises(P.PrepareError, match="another selection"):
        freeze(out, shop, upstream=pinned)
    manifest = freeze(out, shop, upstream=pinned, allow_selection_change=True)
    assert manifest["selection_changed_from_provisional"] is True
    assert P.check(out) == []


def test_unsolvable_items_are_kept_and_flagged(pinned, tmp_path):
    out = tmp_path / "splits"
    P.select(pinned, out)
    shop = prepare_shop(tmp_path / "prepare")
    shop.backend.best_reward = lambda goal: {  # type: ignore[method-assign]
        "best_reward": 0.5,
        "reward_info": {},
        "options_available": False,
    }
    shop.backend.search_rank = lambda query, gold: None  # type: ignore[method-assign]
    manifest = freeze(out, shop, upstream=pinned)
    assert manifest["counts"]["test"] == 140  # not filtered
    assert manifest["unsolvable"]["test"] == 140
    assert manifest["unsolvable_test_flag"] is True
    assert manifest["gold_not_found_by_search"]["train"] == 40


def test_check_names_what_is_inconsistent(pinned, tmp_path):
    out = tmp_path / "splits"
    P.select(pinned, out)
    train = json.loads((out / "train.json").read_text())
    test = json.loads((out / "test.json").read_text())
    train[0] = {**train[0], "asin": test[0]["asin"]}
    train[1] = {**train[1], "id": "1"}
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps([]))
    problems = P.check(out)
    assert "val: 0 items, protocol 20" in problems
    assert "train: bad id '1'" in problems
    assert any("'s ASIN is also in train" in p for p in problems)
    (out / "source.json").write_text("{")
    assert P.check(out)[0].startswith("unreadable manifest")


def test_check_runs_the_draw_again_with_the_upstream_files(pinned, tmp_path):
    """A hand-edited selection that keeps every range and rule passes
    the offline check; with the upstream files the draw runs again and names it."""
    out = tmp_path / "splits"
    P.select(pinned, out)
    assert P.check(out, pinned) == []
    rows = json.loads((out / "test.json").read_text())
    held = {row["goal_index"] for row in rows}
    others = {
        asin(row["goal_index"])
        for split in ("test_p16", "val", "pilot", "train")
        for row in json.loads((out / f"{split}.json").read_text())
    }
    swap = next(
        i
        for i in range(100, 500)
        if i not in held and i not in AMBIGUOUS and asin(i) not in others
    )
    rows[0] = {
        **rows[0],
        "goal_index": swap,
        "id": f"fixed_{swap}",
        "asin": asin(swap),
        "instruction": P.goal_key(instruction(swap)),
    }
    (out / "test.json").write_text(json.dumps(rows))
    assert P.check(out) == []  # every rule still holds
    (problem,) = P.check(out, pinned)
    assert (
        problem.startswith("test: the draw gives other goals") and str(swap) in problem
    )
    assert P.main(["check", "--out", str(out), "--upstream", str(pinned)]) == 1
    # A selection freeze recorded as changed is not drawn again.
    manifest = json.loads((out / "split_manifest.json").read_text())
    manifest["selection_changed_from_provisional"] = True
    (out / "split_manifest.json").write_text(json.dumps(manifest))
    assert P.check(out, pinned) == []
    manifest["selection_changed_from_provisional"] = False
    (out / "split_manifest.json").write_text(json.dumps(manifest))
    problems = P.check(out, tmp_path / "no clone")
    assert any("the draw cannot be run again" in p for p in problems)


def test_the_cli(pinned, tmp_path, capsys):
    out = tmp_path / "splits"
    assert P.main(["select", "--upstream", str(pinned), "--out", str(out)]) == 0
    assert (
        "provisional manifest: test_p16 140, val 20, pilot 50, train 40, test 140"
        in (capsys.readouterr().out)
    )
    assert P.main(["check", "--out", str(out)]) == 0
    assert P.main(["select", "--upstream", str(tmp_path / "x"), "--out", str(out)]) == 2
    (out / "val.json").write_text("[]")
    assert P.main(["check", "--out", str(out)]) == 1


def test_pin_rewrites_only_the_webshop_block(tmp_path, capsys):
    root = tmp_path / "skill_bench"
    for rel in (*P.PIN_DIRS, *ws.VENDOR_FILES):
        source = protocol.SKILL_BENCH_DIR / rel
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copyfile(source, target)
    original = (protocol.SKILL_BENCH_DIR / "protocol.py").read_text()
    (root / "protocol.py").write_text(original)
    assert P.pin(root) == {}  # the repository is pinned
    assert P.asset_hashes(root) == dict(protocol.WEBSHOP_ASSET_SHA256)
    manifest = root / ws.SPLIT_DIR / "split_manifest.json"
    manifest.write_text(manifest.read_text().replace("provisional", "final", 1))
    rel = f"{ws.SPLIT_DIR}/split_manifest.json"
    changed = P.pin(root, write=False)
    assert (
        list(changed) == [rel] and changed[rel][0] == protocol.WEBSHOP_ASSET_SHA256[rel]
    )
    assert (root / "protocol.py").read_text() == original
    assert P.main(["pin", "--check", "--root", str(root)]) == 1
    assert P.main(["pin", "--root", str(root)]) == 0
    assert f"{rel}: " in capsys.readouterr().out
    text = (root / "protocol.py").read_text()
    head, tail = (
        original.split("# BEGIN WEBSHOP_ASSET_SHA256")[0],
        original.split("# END WEBSHOP_ASSET_SHA256")[1],
    )
    assert text.startswith(head) and text.endswith(tail)
    assert P.pin(root) == {} and P.main(["pin", "--check", "--root", str(root)]) == 0
    (root / "protocol.py").write_text("no block here\n")
    assert P.main(["pin", "--root", str(root)]) == 2


def test_the_repository_manifest_is_consistent():
    assert P.check(protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR) == []


def test_the_repository_manifest_is_frozen_against_the_real_goals():
    """The repository's manifest is ``freeze``'s, against the real goals of the server
    image: the selection is the provisional one, every row carries the real goal's
    fields and its diagnostics, and the manifest's counts are the rows' own."""
    frozen = protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR
    manifest = json.loads((frozen / "split_manifest.json").read_text())
    source = json.loads((frozen / "source.json").read_text())
    assert manifest["status"] == source["status"] == ws.MANIFEST_FINAL
    assert manifest["selection_changed_from_provisional"] is False
    assert source["goal_count"] == protocol.WEBSHOP_GOAL_COUNT
    assert source["seed"] == protocol.SEED
    assert source["upstream_commit"] == protocol.WEBSHOP_SERVER["upstream_commit"]
    assert source["data"] == {
        name: pin["sha256"] for name, pin in protocol.WEBSHOP_DATA["files"].items()
    }
    for key in ("price_table_sha256", "index_sha256"):
        assert len(source[key]) == 64 and int(source[key], 16) >= 0, key
    rows = {s: json.loads((frozen / f"{s}.json").read_text()) for s in ws.SPLIT_NAMES}
    unsolvable: dict[str, int] = {}
    not_found: dict[str, int] = {}
    for split, items in rows.items():
        for row in items:
            # The instruction is upstream's process_goal of the real instruction text;
            # the text ends with the price clause unless upstream had no price for the
            # product (its 1,000,000 price limit, no clause).
            assert P.goal_key(row["instruction_text"]) == row["instruction"], row["id"]
            clause = f"{P.PRICE_CLAUSE} {row['price_upper']:.2f} dollars"
            assert row["instruction_text"].endswith(clause) or (
                row["price_upper"] == 1_000_000
                and P.PRICE_CLAUSE not in row["instruction_text"]
            ), row["id"]
            assert (
                isinstance(row["best_reward"], float) and 0 <= row["best_reward"] <= 1
            )
            rank = row["gold_search_rank"]
            assert rank is None or (isinstance(rank, int) and 0 <= rank < 50), row["id"]
            assert isinstance(row["options_available"], bool)
        unsolvable[split] = sum(1 for r in items if r["best_reward"] < 1)
        not_found[split] = sum(1 for r in items if r["gold_search_rank"] is None)
    assert manifest["unsolvable"] == unsolvable
    assert manifest["gold_not_found_by_search"] == not_found
    fraction = unsolvable["test"] / len(rows["test"])
    assert manifest["unsolvable_test_fraction"] == round(fraction, 4)
    assert manifest["unsolvable_test_flag"] is (fraction > P.UNSOLVABLE_FLAG_FRACTION)


@pytest.mark.skipif(
    not os.environ.get("WEBSHOP_UPSTREAM_DIR"),
    reason="WEBSHOP_UPSTREAM_DIR (princeton-nlp/WebShop clone at 64fa2a5) not set",
)
def test_select_reproduces_the_repository_manifest(tmp_path):
    out = tmp_path / "splits"
    P.select(Path(os.environ["WEBSHOP_UPSTREAM_DIR"]), out)
    frozen = protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR
    if ws.load_split_manifest()["status"] == ws.MANIFEST_FINAL:
        assert P.current_selection(out) == P.current_selection(frozen)
        return
    for name in sorted(p.name for p in frozen.glob("*.json")):
        assert (out / name).read_bytes() == (frozen / name).read_bytes(), name
