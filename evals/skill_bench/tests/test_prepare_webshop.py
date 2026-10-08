"""Fixed WebShop manifests: exact inputs, tamper rejection and source verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.envs import webshop as ws
from evals.skill_bench.run import CellSpec, select_items
from evals.skill_bench.tools import prepare_webshop as P

FROZEN = protocol.SKILL_BENCH_DIR / ws.SPLIT_DIR


@pytest.fixture
def copied(tmp_path):
    return Path(shutil.copytree(FROZEN, tmp_path / "splits"))


def test_release_contains_only_final_inputs_and_their_checksums():
    assert P.check(FROZEN) == []
    assert {p.name for p in FROZEN.iterdir()} == {
        "train.json",
        "val.json",
        "test.json",
        "source.json",
        "split_manifest.json",
    }
    manifest = json.loads((FROZEN / "split_manifest.json").read_text())
    assert manifest["counts"] == {"train": 40, "val": 20, "test": 140}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() == digest
        assert protocol.WEBSHOP_ASSET_SHA256[f"{ws.SPLIT_DIR}/{name}"] == digest


@pytest.mark.parametrize(
    "change",
    ["reorder", "replace", "instruction", "missing", "metadata", "checksum", "extra"],
)
def test_check_rejects_changed_inputs_without_rewriting_them(copied, change):
    path = copied / "test.json"
    rows = json.loads(path.read_text())
    if change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
        path.write_text(json.dumps(rows))
    elif change == "replace":
        rows[0] = rows[1]
        path.write_text(json.dumps(rows))
    elif change == "instruction":
        rows[0]["instruction_text"] += " changed"
        path.write_text(json.dumps(rows))
    elif change == "missing":
        path.unlink()
    elif change == "metadata":
        (copied / "source.json").write_text("{}")
    elif change == "checksum":
        manifest = json.loads((copied / "split_manifest.json").read_text())
        manifest["files"]["test.json"] = "0" * 64
        (copied / "split_manifest.json").write_text(json.dumps(manifest))
    else:
        (copied / "extra.json").write_text("[]")
    before = {p.name: p.read_bytes() for p in copied.iterdir()}
    assert P.check(copied)
    assert {p.name: p.read_bytes() for p in copied.iterdir()} == before


def test_check_cli_is_read_only(copied, capsys):
    before = {p.name: p.read_bytes() for p in copied.iterdir()}
    assert P.main(["check", "--out", str(copied)]) == 0
    assert "train=40, val=20, test=140" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in copied.iterdir()} == before
    (copied / "train.json").write_text("[]")
    assert P.main(["check", "--out", str(copied)]) == 1
    assert "SHA-256" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["select", "freeze", "preview-sizes"])
def test_cli_has_no_resampling_or_manifest_replacement_command(command):
    with pytest.raises(SystemExit) as exc:
        P.main([command])
    assert exc.value.code == 2


def test_loading_and_selecting_final_tasks_needs_only_release_inputs():
    env = ws.WebShopEnv()
    train, test = select_items(env, CellSpec("webshop", "luna", "mirix"))
    assert len(train) == 40 and len(test) == 140
    assert {t.raw_id for t in train} == {r["id"] for r in ws.load_split("train")}
    assert {t.raw_id for t in test} == {r["id"] for r in ws.load_split("test")}


def test_runtime_loader_rejects_a_reordered_task_file(tmp_path):
    root = tmp_path / "skill_bench"
    out = root / ws.SPLIT_DIR
    shutil.copytree(FROZEN, out)
    assert len(ws.load_split("test", root)) == 140
    rows = json.loads((out / "test.json").read_text())
    rows.reverse()
    (out / "test.json").write_text(json.dumps(rows))
    with pytest.raises(protocol.ProtocolDriftError, match="test.json: sha256"):
        ws.load_split("test", root)


def test_optional_upstream_check_verifies_membership_without_a_draw(monkeypatch):
    items = {s: ws.load_split(s) for s in ws.SPLIT_NAMES}
    table = [P.Goal(i, None, "", "") for i in range(protocol.WEBSHOP_GOAL_COUNT)]
    for rows in items.values():
        for row in rows:
            i = row["goal_index"]
            table[i] = P.Goal(
                i, row["asin"], row["instruction"], P.normalized(row["instruction"])
            )
    monkeypatch.setattr(P, "load_upstream_files", lambda _p: ([""] * len(table), {}))
    monkeypatch.setattr(P, "offline_table", lambda *_args: table)
    assert P.check(FROZEN, Path("source")) == []
    i = items["test"][0]["goal_index"]
    table[i] = P.Goal(i, "another-product", table[i].key, table[i].norm)
    assert P.check(FROZEN, Path("source")) == [
        f"test: goal {i} differs from its upstream source"
    ]


@pytest.mark.skipif(
    not os.environ.get("WEBSHOP_UPSTREAM_DIR"),
    reason="pinned upstream clone not configured",
)
def test_released_tasks_match_pinned_upstream_files():
    assert P.check(FROZEN, Path(os.environ["WEBSHOP_UPSTREAM_DIR"])) == []


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
    manifest.write_text(manifest.read_text() + " ")
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
