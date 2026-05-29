"""Tests for :mod:`evals.metaclaw.dataset_slice`.

Pure-function tests: no mocks, no fixtures with state.  Assert only on
return values and on the *content* of dst_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.metaclaw.dataset_slice import slice_tests


def _make_src(tmp_path: Path, n_tests: int = 30) -> Path:
    """Write a minimal MetaClaw-shaped ``all_tests_metaclaw.json``."""
    src = tmp_path / "all_tests_metaclaw.json"
    payload = {
        "name": "MetaClaw-Evolution-Bench",
        "openclaw_state_dir": "./openclaw_state",
        "openclaw_config_file": "./openclaw_cfg/metaclaw.json",
        "eval_dir": "./eval",
        "workspace_src": "./workspaces/shared",
        "test": [
            {"id": f"day{i + 1:02d}", "desc": f"Day {i + 1}"} for i in range(n_tests)
        ],
    }
    src.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return src


def test_slice_keeps_first_n(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"

    kept = slice_tests(src, 3, dst)

    assert kept == 3
    out = json.loads(dst.read_text())
    assert len(out["test"]) == 3
    assert [t["id"] for t in out["test"]] == ["day01", "day02", "day03"]


def test_slice_zero_keeps_all(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"

    kept = slice_tests(src, 0, dst)

    assert kept == 30
    out = json.loads(dst.read_text())
    assert len(out["test"]) == 30


def test_slice_more_than_available_keeps_all(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"

    kept = slice_tests(src, 999, dst)

    assert kept == 30
    out = json.loads(dst.read_text())
    assert len(out["test"]) == 30


def test_slice_preserves_top_level_keys(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"

    slice_tests(src, 5, dst)

    src_data = json.loads(src.read_text())
    dst_data = json.loads(dst.read_text())
    # All non-'test' keys identical and byte-equal.
    for k in src_data:
        if k == "test":
            continue
        assert dst_data[k] == src_data[k]
    # No extra keys leaked in.
    assert set(dst_data.keys()) == set(src_data.keys())


def test_slice_does_not_mutate_source(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"
    src_bytes_before = src.read_bytes()

    slice_tests(src, 7, dst)

    assert src.read_bytes() == src_bytes_before


def test_slice_rejects_negative_n(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "out.json"

    with pytest.raises(ValueError):
        slice_tests(src, -1, dst)


def test_slice_rejects_non_metaclaw_json(tmp_path: Path) -> None:
    src = tmp_path / "bad.json"
    src.write_text('{"oops": []}', encoding="utf-8")
    dst = tmp_path / "out.json"

    with pytest.raises(ValueError):
        slice_tests(src, 3, dst)


def test_slice_creates_parent_dirs(tmp_path: Path) -> None:
    src = _make_src(tmp_path, 30)
    dst = tmp_path / "deeper" / "subdir" / "out.json"

    kept = slice_tests(src, 1, dst)

    assert kept == 1
    assert dst.exists()


def test_slice_on_real_vendored_dataset() -> None:
    """Smoke check: the real 30-day vendored dataset slices cleanly to 3."""
    import tempfile

    from evals.metaclaw.runner import DATA_DIR

    src = DATA_DIR / "all_tests_metaclaw.json"
    if not src.exists():
        pytest.skip(f"vendored dataset not present at {src}")

    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "sliced.json"
        kept = slice_tests(src, 3, dst)
        assert kept == 3
        out = json.loads(dst.read_text())
        assert len(out["test"]) == 3
        # Top-level keys carried verbatim
        full = json.loads(src.read_text())
        for k in ("name", "openclaw_state_dir", "eval_dir", "workspace_src"):
            assert out[k] == full[k]
