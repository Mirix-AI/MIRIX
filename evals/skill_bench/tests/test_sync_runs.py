"""tools/sync_runs.py: two hosts' runs directories mirrored and merged into one read-only
view, with local-path sources and the machine's own rsync (no ssh).

The last test is the one that matters most: the offline matrix split into two "hosts" by
benchmark, each with its own gateway directory and absolute paths that name a tree which
does not exist here, then merged: ``audit.py``, ``report.py`` and ``summarize_learning``
give over the merged view exactly what they give over the unsplit directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import audit, report
from evals.skill_bench.cell_records import (
    MERGED_MARKER,
    CellPaths,
    read_json,
    write_json_atomic,
)
from evals.skill_bench.tests.conftest import REMOVED_METHOD
from evals.skill_bench.tests.e2e_support import (
    TEST_UPSTREAM,
    Matrix,
    copy_cells,
)
from evals.skill_bench.tools import summarize_learning
from evals.skill_bench.tools import sync_runs as S

RSYNC = shutil.which("rsync")
pytestmark = pytest.mark.skipif(RSYNC is None, reason="rsync is not installed")

BLOB = hashlib.sha256(b"page").hexdigest()


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def host_runs(
    root: Path, cells: tuple[str, ...], *, blobs: dict[str, str] | None = None
):
    """A host's runs directory with ``cells``, each with its gateway run directory and
    what a live run also leaves there (locks, temporaries, partial checkpoints)."""
    root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        c = root / cell
        _write(c / "cell.json", json.dumps({"cell": {"cell_id": cell}}))
        _write(c / "status.json", '{"status": "complete"}')
        _write(c / "cell.lock", "")
        _write(c / f".status.json.{'a' * 32}.tmp", "{")
        a = c / "attempt-01"
        _write(a / "attempt.json", json.dumps({"run_id": f"{cell}-a01-0001"}))
        _write(a / "learning.jsonl", '{"position": 0}\n')
        _write(a / "home" / ".mirix" / "db.sqlite", "home")
        _write(a / "checkpoints" / "LATEST", "{}")
        _write(a / "checkpoints" / "c0000-init" / "checkpoint.json", "{}")
        _write(a / "checkpoints" / "c0001-played.partial" / "snapshot.json", "{")
        _write(a / "discarded.jsonl", '{"segment": 1}\n')
        _write(a / "discarded" / "seg-01" / "learning.tail.jsonl", "{}\n")
        _write(a / "discarded" / "seg-01" / "home" / "new.txt", "home")
        _write(a / "discarded" / "seg-02.partial" / "learning.tail.jsonl", "{")
        _write(root / f"{cell}-a01-0001" / "gateway.jsonl", f'{{"cell": "{cell}"}}\n')
    _write(root / "_gateway" / "starts.jsonl", "{}\n")
    _write(root / "_gateway" / "budget.json", "{}")
    _write(root / "_schedule" / "state.json", "{}")
    _write(root / "_schedule" / "scheduler.lock", "")
    _write(root / "_audit" / "audit.json", '{"host": "remote"}')
    _write(root / "_report" / "report.md", "# remote")
    for name, text in (blobs if blobs is not None else {BLOB: "page"}).items():
        _write(root / "_blobs" / name, text)
    _write(root / "_blobs" / f".{BLOB}.{'b' * 32}.tmp", "pa")
    return root


def sync(tmp_path: Path, *sources: str, into: Path | None = None, **kw: Any) -> int:
    into = into or tmp_path / "merged"
    argv = [a for s in sources for a in ("--source", s)] + ["--into", str(into)]
    for flag, on in kw.items():
        if on is True:
            argv.append("--" + flag.replace("_", "-"))
        elif on:
            argv += ["--" + flag.replace("_", "-"), str(on)]
    return S.main(argv)


def files(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    }


@pytest.fixture()
def two_hosts(tmp_path: Path) -> tuple[Path, Path]:
    cloud = host_runs(
        tmp_path / "cloud" / "runs",
        ("alfworld-luna-none", "alfworld-luna56-none"),
        blobs={BLOB: "page", hashlib.sha256(b"a").hexdigest(): "a"},
    )
    local = host_runs(
        tmp_path / "local" / "runs",
        ("webshop-luna-none",),
        blobs={BLOB: "page", hashlib.sha256(b"d").hexdigest(): "d"},
    )
    return cloud, local


# --- the mirror ---------------------------------------------------------------------------------


def test_the_mirror_holds_the_records_only(tmp_path, capsys):
    source = host_runs(tmp_path / "local", ("webshop-luna-none",))
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    mirror = tmp_path / "merged" / "_hosts" / "local"
    got = files(mirror)
    cell = "webshop-luna-none"
    assert f"{cell}/cell.json" in got and f"{cell}/attempt-01/learning.jsonl" in got
    assert f"{cell}/attempt-01/discarded/seg-01/learning.tail.jsonl" in got
    assert f"{cell}-a01-0001/gateway.jsonl" in got
    assert {"_gateway/starts.jsonl", "_schedule/state.json", f"_blobs/{BLOB}"} <= got
    left_out = [
        f"{cell}/cell.lock",
        f"{cell}/.status.json.{'a' * 32}.tmp",
        f"{cell}/attempt-01/home/.mirix/db.sqlite",
        f"{cell}/attempt-01/checkpoints/LATEST",
        f"{cell}/attempt-01/checkpoints/c0001-played.partial/snapshot.json",
        f"{cell}/attempt-01/discarded/seg-01/home/new.txt",
        f"{cell}/attempt-01/discarded/seg-02.partial/learning.tail.jsonl",
        "_schedule/scheduler.lock",
        "_audit/audit.json",
        "_report/report.md",
        f"_blobs/.{BLOB}.{'b' * 32}.tmp",
    ]
    assert not got & set(left_out)
    assert "local: rsync exit 0; transferred" in capsys.readouterr().out
    # --with-home and --with-checkpoints bring those in (never a partial one).
    assert (
        sync(tmp_path, f"local={source}", with_home=True, with_checkpoints=True)
        == S.EXIT_OK
    )
    got = files(mirror)
    assert {
        f"{cell}/attempt-01/home/.mirix/db.sqlite",
        f"{cell}/attempt-01/checkpoints/LATEST",
        f"{cell}/attempt-01/checkpoints/c0000-init/checkpoint.json",
        f"{cell}/attempt-01/discarded/seg-01/home/new.txt",
    } <= got
    assert (
        f"{cell}/attempt-01/checkpoints/c0001-played.partial/snapshot.json" not in got
    )
    record = read_json(tmp_path / "merged" / MERGED_MARKER)["sources"]["local"]
    assert record["with_home"] is True and record["with_checkpoints"] is True


def test_the_mirror_follows_a_resume_that_moved_and_restored_files(tmp_path):
    """A learning resume moves a trajectory to discarded/ and restores the method's files
    from a checkpoint with their older times: the mirror drops the moved file and takes
    the restored one (no --update, which would keep the newer, discarded copy)."""
    source = host_runs(tmp_path / "local", ("webshop-luna-none",))
    attempt = source / "webshop-luna-none" / "attempt-01"
    trajectory = _write(attempt / "trajectories" / "learning" / "p0001.json", "{1}")
    state = _write(attempt / "method" / "state.json", "after")
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    mirror = (
        tmp_path / "merged" / "_hosts" / "local" / "webshop-luna-none" / "attempt-01"
    )
    assert (mirror / "method" / "state.json").read_text() == "after"
    shutil.move(trajectory, _write(attempt / "discarded" / "seg-02" / "x").parent)
    state.write_text("befor")  # restored from the checkpoint: same size, older time
    old = (mirror / "method" / "state.json").stat().st_mtime - 3600
    os.utime(state, (old, old))
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert not (mirror / "trajectories" / "learning" / "p0001.json").exists()
    assert (mirror / "discarded" / "seg-02" / "p0001.json").is_file()
    assert (mirror / "method" / "state.json").read_text() == "befor"
    record = read_json(tmp_path / "merged" / MERGED_MARKER)["sources"]["local"]
    assert record["deleted"] >= 1 and record["transferred"] >= 2


def test_a_local_audit_in_a_mirror_survives_and_the_sources_is_never_copied(tmp_path):
    source = host_runs(tmp_path / "local", ("webshop-luna-none",))
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    mirror = tmp_path / "merged" / "_hosts" / "local"
    mine = _write(mirror / "_audit" / "audit.json", '{"host": "here"}')
    _write(tmp_path / "merged" / "_audit" / "audit.json", "{}")
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert mine.read_text() == '{"host": "here"}'
    assert (tmp_path / "merged" / "_audit" / "audit.json").is_file()
    assert not (mirror / "_report").exists()


# --- the merged view ---------------------------------------------------------------------------


def test_the_merged_view_links_every_hosts_cells_runs_and_blobs(
    tmp_path, two_hosts, capsys
):
    cloud, local = two_hosts
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    links = {p.name: os.readlink(p) for p in into.iterdir() if p.is_symlink()}
    assert links == {
        "alfworld-luna-none": "_hosts/cloud/alfworld-luna-none",
        "alfworld-luna-none-a01-0001": "_hosts/cloud/alfworld-luna-none-a01-0001",
        "alfworld-luna56-none": "_hosts/cloud/alfworld-luna56-none",
        "alfworld-luna56-none-a01-0001": "_hosts/cloud/alfworld-luna56-none-a01-0001",
        "webshop-luna-none": "_hosts/local/webshop-luna-none",
        "webshop-luna-none-a01-0001": "_hosts/local/webshop-luna-none-a01-0001",
    }
    assert (into / "webshop-luna-none" / "cell.json").is_file()  # the links resolve
    for name in ("_gateway", "_schedule"):
        assert not (into / name).exists()
    blobs = into / "_blobs"
    assert blobs.is_dir() and not blobs.is_symlink()
    blob_links = {p.name: os.readlink(p) for p in blobs.iterdir()}
    assert blob_links == {
        BLOB: f"../_hosts/cloud/_blobs/{BLOB}",
        hashlib.sha256(
            b"a"
        ).hexdigest(): f"../_hosts/cloud/_blobs/{hashlib.sha256(b'a').hexdigest()}",
        hashlib.sha256(
            b"d"
        ).hexdigest(): f"../_hosts/local/_blobs/{hashlib.sha256(b'd').hexdigest()}",
    }
    assert (blobs / BLOB).read_text() == "page"
    merged = read_json(into / MERGED_MARKER)
    assert set(merged["sources"]) == {"cloud", "local"}
    assert merged["sources"]["cloud"]["spec"] == str(cloud)
    assert merged["sources"]["cloud"]["rsync_exit"] == 0
    assert (
        merged["sources"]["cloud"]["files"] > 0
        and merged["sources"]["cloud"]["bytes"] > 0
    )
    assert merged["sources"]["cloud"]["rsync_version"]
    assert merged["entries"]["webshop-luna-none"] == "local"
    assert merged["conflicts"] == [] and merged["blobs"] == 3
    out = capsys.readouterr().out
    assert "3 cells, 3 gateway runs, 3 blobs from 2 hosts; 9 links made" in out
    # Again: nothing to link, nothing removed.
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    assert "0 links made, 0 removed" in capsys.readouterr().out
    # The view is built from every mirror, not only the sources of this call.
    assert sync(tmp_path, f"cloud={cloud}") == S.EXIT_OK
    assert (into / "webshop-luna-none").is_symlink()
    # The readers see the cells and the gateway runs.
    assert [p.name for p in report.discover_cells(into)] == [
        "alfworld-luna-none",
        "alfworld-luna56-none",
        "webshop-luna-none",
    ]


def test_the_cells_of_a_removed_method_keep_syncing(tmp_path, capsys, removed_method):
    """A method the study removed keeps its run data: its cells (complete, or stopped
    mid-learning) and gateway runs are mirrored and linked into the view like any other;
    the study tools set them aside."""
    removed = (f"webshop-luna56-{removed_method}", f"webshop-luna-{removed_method}")
    local = host_runs(tmp_path / "local" / "runs", ("webshop-luna-none", *removed))
    _write(local / removed[0] / "status.json", '{"status": "interrupted"}')
    assert sync(tmp_path, f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    for cell in removed:
        assert os.readlink(into / cell) == f"_hosts/local/{cell}"
        assert os.readlink(into / f"{cell}-a01-0001") == (
            f"_hosts/local/{cell}-a01-0001"
        )
        assert (into / cell / "attempt-01" / "learning.jsonl").is_file()
    assert read_json(into / removed[0] / "status.json") == {"status": "interrupted"}
    assert "3 cells, 3 gateway runs" in capsys.readouterr().out
    # Again: nothing deleted, nothing refused.
    assert sync(tmp_path, f"local={local}") == S.EXIT_OK
    assert "0 links made, 0 removed" in capsys.readouterr().out


def test_the_readers_set_a_removed_methods_cells_aside_over_a_merged_view(
    tmp_path, removed_method
):
    """Over a merged view, the report, the learning summary and the audit set the
    removed method's cells aside exactly as each host's own runs directory does: the
    same listing (no path of either tree in it) and the same money, summed."""
    from evals.skill_bench.tests.test_report import FP, _call, _row, make_cell

    dc = removed_method
    cloud, local = tmp_path / "cloud" / "runs", tmp_path / "local" / "runs"
    make_cell(
        cloud,
        "alfworld",
        "none",
        test=[_row("a1", 1.0)],
        calls=[_call("actor", "test", 0.10, item="u1")],
    )
    make_cell(
        cloud,
        "alfworld",
        dc,
        test=[_row("a1", 0.0)],
        calls=[_call("learner", "learning", 0.20)],
    )
    make_cell(
        local,
        "webshop",
        dc,
        test=[],
        status="interrupted",
        calls=[_call("actor", "learning", 0.05, item="u1")],
    )
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    removed = [f"alfworld-luna-{dc}", f"webshop-luna-{dc}"]
    merged = report.build_report(into, frozen=FP)
    hosts = [report.build_report(runs, frozen=FP) for runs in (cloud, local)]
    assert [c["cell_id"] for c in merged["removed"]["cells"]] == removed
    assert merged["removed"]["cells"] == [
        c for host in hosts for c in host["removed"]["cells"]
    ]
    assert merged["costs"]["removed"]["total"] == pytest.approx(0.25)
    assert merged["costs"]["total"] == pytest.approx(0.10)
    assert merged["costs"]["all_cells_total"] == pytest.approx(0.35)
    summary = summarize_learning.summarize(into)
    assert summary["removed"] == [
        c
        for runs in (cloud, local)
        for c in summarize_learning.summarize(runs)["removed"]
    ]
    assert summary["removed_cost"] == pytest.approx(0.25)
    listed = audit.audit_runs(into, cells=removed)
    assert [(r.cell_id, r.status) for r in listed.removed] == [
        (removed[0], "complete"),
        (removed[1], "interrupted"),
    ]
    assert listed.cells == [] and listed.findings == []


def test_an_entry_gone_from_its_host_leaves_the_view(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    shutil.rmtree(cloud / "alfworld-luna56-none")
    shutil.rmtree(cloud / "alfworld-luna56-none-a01-0001")
    (cloud / "_blobs" / hashlib.sha256(b"a").hexdigest()).unlink()
    into = tmp_path / "merged"
    _write(into / "notes.txt", "mine")  # not a link into _hosts/: never touched
    os.symlink("elsewhere", into / "my-link")
    # Whole entries gone from a host: only the operator says so.
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_REFUSED
    assert (into / "alfworld-luna56-none").is_symlink()
    assert (
        sync(tmp_path, f"cloud={cloud}", f"local={local}", allow_deletes=True)
        == S.EXIT_OK
    )
    assert not (into / "alfworld-luna56-none").is_symlink()
    assert not (into / "alfworld-luna56-none-a01-0001").is_symlink()
    assert not (into / "_blobs" / hashlib.sha256(b"a").hexdigest()).is_symlink()
    assert (into / "notes.txt").read_text() == "mine" and (
        into / "my-link"
    ).is_symlink()
    assert "3 removed" in capsys.readouterr().out
    assert "alfworld-luna56-none" not in read_json(into / MERGED_MARKER)["entries"]


def test_the_same_cell_from_two_hosts(tmp_path, two_hosts, capsys):
    """Identical trees are one entry (recorded as duplicates); different ones stop the
    tool before it touches the view."""
    cloud, local = two_hosts
    shutil.copytree(cloud / "alfworld-luna-none", local / "alfworld-luna-none")
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    merged = read_json(into / MERGED_MARKER)
    assert merged["duplicates"] == {"alfworld-luna-none": ["cloud", "local"]}
    assert os.readlink(into / "alfworld-luna-none") == "_hosts/cloud/alfworld-luna-none"
    before = {p.name: os.readlink(p) for p in into.iterdir() if p.is_symlink()}
    capsys.readouterr()
    _write(local / "alfworld-luna-none" / "attempt-01" / "learning.jsonl", "{}\n{}\n")
    _write(local / "_blobs" / BLOB, "another page")
    shutil.rmtree(cloud / "webshop-luna-none", ignore_errors=True)
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_CONFLICT
    err = capsys.readouterr().err
    assert "conflict: alfworld-luna-none from cloud, local (different content)" in err
    assert f"conflict: _blobs/{BLOB} from cloud, local (different sizes)" in err
    assert "the merged view was not changed" in err
    after = {p.name: os.readlink(p) for p in into.iterdir() if p.is_symlink()}
    assert after == before
    conflicts = read_json(into / MERGED_MARKER)["conflicts"]
    assert {c["entry"] for c in conflicts} == {"alfworld-luna-none", f"_blobs/{BLOB}"}


# --- modes -------------------------------------------------------------------------------------


def test_a_dry_run_changes_nothing(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}", dry_run=True) == S.EXIT_OK
    assert not (tmp_path / "merged").exists()
    out = capsys.readouterr().out
    assert (
        "cloud: rsync exit 0; would transfer" in out
        and "dry run: nothing changed" in out
    )


def test_mirror_only_updates_the_mirror_not_the_view(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    first = read_json(into / MERGED_MARKER)
    host_runs(cloud, ("alfworld-luna-awm",))
    assert sync(tmp_path, f"cloud={cloud}", mirror_only=True) == S.EXIT_OK
    assert (into / "_hosts" / "cloud" / "alfworld-luna-awm" / "cell.json").is_file()
    assert not (into / "alfworld-luna-awm").exists()
    merged = read_json(into / MERGED_MARKER)
    assert merged["entries"] == first["entries"]
    assert merged["sources"]["local"] == first["sources"]["local"]
    assert (
        merged["sources"]["cloud"]["synced_at"]
        >= first["sources"]["cloud"]["synced_at"]
    )
    assert "mirror only" in capsys.readouterr().out
    # Monitoring from scratch: a mirror-only call makes a view directory that a later
    # full sync accepts.
    fresh = tmp_path / "fresh"
    assert sync(tmp_path, f"cloud={cloud}", into=fresh, mirror_only=True) == S.EXIT_OK
    assert (fresh / MERGED_MARKER).is_file()
    assert sync(tmp_path, f"cloud={cloud}", into=fresh) == S.EXIT_OK
    assert (fresh / "alfworld-luna-awm").is_symlink()


def _fake_rsync(tmp_path: Path, code: int) -> str:
    """An rsync that does the transfer and then exits ``code``."""
    script = tmp_path / f"rsync-{code}"
    script.write_text(
        "#!/bin/sh\n"
        f'"{RSYNC}" "$@"\n'
        'if [ "$1" = "--version" ]; then exit 0; fi\n'
        f"exit {code}\n"
    )
    script.chmod(0o755)
    return str(script)


def test_an_rsync_failure_leaves_the_view(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts
    into = tmp_path / "merged"
    assert (
        sync(
            tmp_path,
            f"cloud={cloud}",
            f"local={local}",
            rsync=_fake_rsync(tmp_path, 12),
        )
        == S.EXIT_RSYNC
    )
    assert "error: cloud: rsync failed (exit 12)" in capsys.readouterr().err
    assert not any(p.is_symlink() for p in into.iterdir())
    assert read_json(into / MERGED_MARKER)["sources"]["cloud"]["rsync_exit"] == 12
    # Exit 24 (files vanished while a run wrote) is a warning: the view is built.
    assert (
        sync(
            tmp_path,
            f"cloud={cloud}",
            f"local={local}",
            rsync=_fake_rsync(tmp_path, 24),
        )
        == S.EXIT_OK
    )
    assert "files vanished during the transfer" in capsys.readouterr().err
    assert (into / "webshop-luna-none").is_symlink()


# --- refusals ----------------------------------------------------------------------------------


def test_refusals(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts

    def refused(*argv: str) -> str:
        capsys.readouterr()
        assert S.main(list(argv)) == S.EXIT_REFUSED
        return capsys.readouterr().err

    into = str(tmp_path / "merged")
    assert "letters, digits" in refused("--source", f"a b={cloud}", "--into", into)
    assert "expected NAME=PATH" in refused("--source", str(cloud), "--into", into)
    assert "may not start with '-'" in refused(
        "--source", "x=-oProxyCommand=evil:runs", "--into", into
    )
    assert "more than once" in refused(
        "--source", f"cloud={cloud}", "--source", f"cloud={local}", "--into", into
    )
    assert "is not a directory" in refused(
        "--source", f"x={tmp_path / 'missing'}", "--into", into
    )
    # A host's runs directory as the view, or a view inside a source.
    assert "neither empty nor a merged view" in refused(
        "--source", f"local={local}", "--into", str(cloud)
    )
    assert "overlaps" in refused(
        "--source", f"cloud={cloud}", "--into", str(cloud / "merged")
    )
    # A merged view as a source.
    assert sync(tmp_path, f"cloud={cloud}") == S.EXIT_OK
    assert "is a merged view" in refused(
        "--source", f"m={tmp_path / 'merged'}", "--into", str(tmp_path / "other")
    )
    # A host's mirror is read-only too: never a source, never a place to run cells.
    mirror = tmp_path / "merged" / "_hosts" / "cloud"
    assert read_json(mirror / MERGED_MARKER)["mirror_of"] == str(cloud)
    assert "is a merged view" in refused(
        "--source", f"m={mirror}", "--into", str(tmp_path / "other")
    )
    # A view where a cell ran for real.
    _write(tmp_path / "merged" / "webshop-luna-awm" / "cell.json", "{}")
    assert "real cell or gateway directory" in refused(
        "--source", f"cloud={cloud}", "--into", into
    )
    assert not list(tmp_path.glob("other"))
    assert (
        S.main(["--source", f"cloud={cloud}", "--into", into, "--rsync", "no-rsync"])
        == 2
    )


def test_a_sync_never_empties_or_replaces_a_mirror_by_itself(tmp_path, capsys):
    """A mirror may be the only copy of a gone host's cells. An emptied or re-created
    source, another runs directory, or a sync deleting most of the mirror is refused
    (exit 2) and the mirror is left as it was; --allow-deletes lets the deletions
    through (and only them, below)."""
    source = host_runs(
        tmp_path / "cloud" / "runs",
        tuple(f"alfworld-luna-c{i}" for i in range(12)),
    )
    _write(source / "_gateway" / "starts.jsonl", '{"start_id": "gw-first"}\n')
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    record = read_json(merged / MERGED_MARKER)
    assert record["sources"]["cloud"]["first_start"] == "gw-first"
    before = files(mirror)
    backup = tmp_path / "backup"
    shutil.copytree(source, backup, symlinks=True)

    def refused(*extra: str) -> str:
        capsys.readouterr()
        assert (
            S.main(["--source", f"cloud={source}", "--into", str(merged), *extra])
            == S.EXIT_REFUSED
        )
        assert files(mirror) == before  # untouched
        return capsys.readouterr().err

    # An emptied (or re-created, not yet launched) runs directory.
    shutil.rmtree(source)
    source.mkdir()
    assert "has no _gateway/starts.jsonl" in refused()
    # Another runs directory (a mistyped path, a re-provisioned host).
    shutil.rmtree(source)
    host_runs(source, ("webshop-luna-none",))
    _write(source / "_gateway" / "starts.jsonl", '{"start_id": "gw-other"}\n')
    assert "is another runs directory" in refused()
    # The same runs directory, most of whose cells are gone.
    shutil.rmtree(source)
    shutil.copytree(backup, source, symlinks=True)
    for i in range(1, 12):
        shutil.rmtree(source / f"alfworld-luna-c{i}")
        shutil.rmtree(source / f"alfworld-luna-c{i}-a01-0001")
    assert "would delete" in refused()
    # The operator says so: the mirror follows its source.
    assert sync(tmp_path, f"cloud={source}", allow_deletes=True) == S.EXIT_OK
    assert not (mirror / "alfworld-luna-c5").exists()
    assert (mirror / "alfworld-luna-c0" / "cell.json").is_file()
    # A few deletions (a resume moving files aside) need no flag.
    shutil.rmtree(source / "alfworld-luna-c0" / "attempt-01" / "discarded")
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK


def test_a_remote_source_whose_identity_cannot_be_read_syncs_nothing(tmp_path, capsys):
    """The identity of a remote source is read with one rsync of its
    starts file over ssh; when ssh fails, nothing is synced (exit 3)."""
    calls = tmp_path / "calls.txt"
    script = tmp_path / "rsync-ssh-down"
    script.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "rsync  version 3"; exit 0; fi\n'
        f'echo "$@" >> "{calls}"\n'
        "echo 'ssh: connect to host remote port 22: Connection refused' >&2\n"
        "exit 255\n"
    )
    script.chmod(0o755)
    assert (
        sync(tmp_path, "cloud=user@remote:runs/study", rsync=str(script))
        == S.EXIT_RSYNC
    )
    err = capsys.readouterr().err
    assert "cannot read its _gateway/starts.jsonl (rsync exit 255)" in err
    (call,) = calls.read_text().splitlines()
    assert "user@remote:runs/study/_gateway/starts.jsonl" in call
    assert "-e ssh -o BatchMode=yes" in call
    assert not (tmp_path / "merged" / "_hosts" / "cloud").exists()


def test_the_identity_of_a_runs_directory():
    assert S.starts_identity(
        '\n{"start_id": "abc", "ts": "t"}\n{"start_id": "d"}\n'
    ) == ("abc")
    assert S.starts_identity("{}\n").startswith("sha256:")
    assert S.starts_identity("") is None
    local = S.parse_source("local=/nonexistent/runs")
    assert S.source_identity(local) == S.Identity(None)


def test_sources_and_the_rsync_command():
    remote = S.parse_source("cloud=203.0.113.10:checkout/evals/skill_bench/runs/study")
    assert (remote.host, remote.path) == (
        "203.0.113.10",
        "checkout/evals/skill_bench/runs/study",
    )
    assert remote.rsync_arg == "203.0.113.10:checkout/evals/skill_bench/runs/study/"
    assert (
        S.parse_source("u=user@remote:/home/user/checkout/runs/").rsync_arg
        == "user@remote:/home/user/checkout/runs/"
    )
    local = S.parse_source("local=evals/skill_bench/runs/study-local")
    assert local.host is None and local.rsync_arg.endswith("study-local/")
    assert S.parse_source("odd=./a:b").host is None
    argv = S.rsync_argv(remote, Path("/m/_hosts/cloud"))
    assert argv[-2:] == [
        "203.0.113.10:checkout/evals/skill_bench/runs/study/",
        "/m/_hosts/cloud/",
    ]
    assert (
        argv[argv.index("-e") + 1] == "ssh -o BatchMode=yes -o ServerAliveInterval=30"
    )
    assert "-z" in argv and "--timeout=600" in argv
    local_argv = S.rsync_argv(local, Path("/m/_hosts/local"))
    assert "-z" not in local_argv and "-e" not in local_argv
    assert "--exclude=attempt-*/home/" in local_argv
    full = S.rsync_argv(
        local,
        Path("/m"),
        dry_run=True,
        with_home=True,
        with_checkpoints=True,
        checksum=True,
    )
    assert {"-n", "-c", "-H"} <= set(full)
    assert "--exclude=attempt-*/home/" not in full
    assert "--exclude=attempt-*/checkpoints/" not in full
    assert "--exclude=*.partial/" in full


def _option(arg: str) -> str:
    return arg.split("=", 1)[0]


def test_every_rsync_option_is_one_openrsync_accepts():
    """macOS ships openrsync (protocol 29): the command uses no option it lacks, never
    --update (a restored older file must replace the mirror's) and never --partial (a
    cut transfer must not leave a truncated record)."""
    source = S.parse_source("cloud=host:runs")
    argv = S.rsync_argv(
        source, Path("/m"), dry_run=True, with_checkpoints=True, checksum=True
    )
    options = [a for a in argv[1:-2] if a.startswith("-")]
    assert {_option(a) for a in options} <= S.OPENRSYNC_OPTIONS
    assert not {"-u", "--update", "--partial", "--inplace", "-P"} & set(options)
    # openrsync's usage (macOS 15): its short flags, and "[-e program]".
    openrsync_short = set("0468BCDEFHIKLOPRSTWVabcdghiklnopqrtuvxyz") | {"e"}
    for option in S.OPENRSYNC_OPTIONS:
        if len(option) == 2:
            assert option[1] in openrsync_short, option
    # And the local rsync, whichever it is, lists every long option.
    help_text = subprocess.run([str(RSYNC), "--help"], capture_output=True, text=True)
    text = help_text.stdout + help_text.stderr
    for option in S.OPENRSYNC_OPTIONS:
        if option.startswith("--"):
            assert option in text, option


def test_itemized_output_of_both_rsyncs():
    items = S.parse_itemized(
        "cd+++++++ cell/\n"
        ">f+++++++ cell/cell.json\n"
        ">f.st.... run/gateway.jsonl\n"
        ">f....... cell/new.json\n"  # openrsync's dry run
        ".d..t.... cell/attempt-01/\n"
        "*deleting cell/attempt-01/trajectories/p0001.json\n"
    )
    assert items.transferred == ["cell/cell.json", "run/gateway.jsonl", "cell/new.json"]
    assert items.deleted == ["cell/attempt-01/trajectories/p0001.json"]
    # Directories, and what a sync cut off left behind, are no deleted files.
    # openrsync ("*deleting path"; it deletes its own --delay-updates staging directory
    # and a temporary file of "." + name + "." + 10 letters and digits) ...
    openrsync = S.parse_itemized(
        "*deleting cell/attempt-01/.learning.jsonl.aB3dE9xY7q\n"
        "*deleting cell/.~tmp~/cell.json\n"
        "*deleting cell/.~tmp~/\n"
        "*deleting cell/.cell.json.JGwmQBbyr0\n"
        "*deleting gone/sub/f\n"
        "*deleting gone/sub/\n"
        "*deleting gone/\n"
    )
    assert openrsync.deleted == ["gone/sub/f"]
    assert openrsync.deleted_dirs == ["gone/sub/", "gone/"]
    assert openrsync.leftovers == [
        "cell/attempt-01/.learning.jsonl.aB3dE9xY7q",
        "cell/.~tmp~/cell.json",
        "cell/.~tmp~/",
        "cell/.cell.json.JGwmQBbyr0",
    ]
    # ... and GNU rsync ("*deleting   path"; its temporary file has 6 random characters,
    # and it never deletes .~tmp~/, which it protects).
    gnu = S.parse_itemized(
        "cd+++++++++ cell/\n"
        ">f+++++++++ cell/cell.json\n"
        "*deleting   cell/attempt-01/trajectories/p0001.json\n"
        "*deleting   cell/attempt-01/trajectories/\n"
        "*deleting   cell/.status.json.aB3dE9\n"
        "*deleting   gone-cell/\n"
    )
    assert gnu.transferred == ["cell/cell.json"]
    assert gnu.deleted == ["cell/attempt-01/trajectories/p0001.json"]
    assert gnu.deleted_dirs == ["cell/attempt-01/trajectories/", "gone-cell/"]
    assert gnu.leftovers == ["cell/.status.json.aB3dE9"]
    # The harness's own names are never taken for leftovers.
    for path in (
        f"cell/.status.json.{'a' * 32}.tmp",
        f".merged.{'b' * 32}.link",
        "cell/attempt-01/learning.jsonl",
        f"_blobs/{BLOB}",
        ".DS_Store",
        "cell/attempt-01/home/.mirix/",
    ):
        assert not S.is_rsync_leftover(path), path


# --- an interrupted sync, whole entries, linked mirrors, what readers are told ------------------


def _twelve_cells(tmp_path: Path) -> Path:
    source = host_runs(
        tmp_path / "cloud" / "runs", tuple(f"alfworld-luna-c{i}" for i in range(12))
    )
    _write(source / "_gateway" / "starts.jsonl", '{"start_id": "gw-first"}\n')
    return source


def _interrupt(mirror: Path) -> list[Path]:
    """What a sync cut off (Ctrl-C, a kill, a dropped connection) leaves in a mirror, as
    openrsync leaves it: a ``.~tmp~/`` staging directory (``--delay-updates``)
    in every directory it was updating, holding the files received so far, and the
    temporary file of the one being received ("." + its name + "." + 10 letters and digits;
    GNU rsync's has 6)."""
    made = []
    for directory in sorted(p for p in mirror.rglob("*") if p.is_dir()):
        received = [f for f in directory.iterdir() if f.is_file()]
        if received:
            (directory / ".~tmp~").mkdir()
            for f in received:
                made.append(Path(shutil.copy2(f, directory / ".~tmp~" / f.name)))
    made.append(_write(mirror / "alfworld-luna-c1" / ".cell.json.JGwmQBbyr0", "{"))
    made.append(
        _write(mirror / "alfworld-luna-c2" / "attempt-01" / ".learning.jsonl.aB3dE9")
    )
    return made


def _leftovers(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if ".~tmp~" in p.parts or p.name.endswith((".JGwmQBbyr0", ".aB3dE9"))
    )


def test_an_interrupted_sync_does_not_stop_the_next_one(tmp_path, capsys):
    """A sync cut off mid-transfer leaves openrsync's staging directories and temporary
    files in the mirror. Counted as deleted files (with each directory), they would get
    the next sync refused, and only --allow-deletes, which also lifts the identity guard,
    would get past it. They are no data: left out of the count and the deletions, and
    removed."""
    source = _twelve_cells(tmp_path)
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    real = read_json(merged / MERGED_MARKER)["sources"]["cloud"]["files"]
    before = files(mirror)
    leftovers = _interrupt(mirror)
    staged = len({p.parent for p in leftovers if p.parent.name == ".~tmp~"})
    assert len(leftovers) > real and staged > 40  # more than the mirror really holds
    capsys.readouterr()
    assert sync(tmp_path, f"cloud={source}", dry_run=True) == S.EXIT_OK
    out = capsys.readouterr().out
    assert "would delete 0" in out and _leftovers(mirror)  # a dry run changes nothing
    assert S.tree_size(mirror)[0] == real
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    out = capsys.readouterr().out
    assert f"removed {staged} staging directories an interrupted sync left" in out
    assert _leftovers(mirror) == [] and files(mirror) == before
    record = read_json(merged / MERGED_MARKER)["sources"]["cloud"]
    assert record["files"] == real and record["deleted"] == 0
    assert record["leftovers_removed"] == staged
    # A cell holding leftovers is still identical to its copy on another host.
    digest = S.tree_digest(mirror / "alfworld-luna-c2")
    _interrupt(mirror)
    assert S.tree_digest(mirror / "alfworld-luna-c2") == digest


def test_the_guard_reads_gnu_rsyncs_deletions(tmp_path, capsys):
    """The deletion guard with GNU rsync's itemized dry run ("*deleting   path", its
    6-character temporary files): its leftovers are not counted, a whole cell is."""
    source = _twelve_cells(tmp_path)
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    lines = tmp_path / "dry-run.txt"
    script = tmp_path / "rsync-gnu"
    script.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        f'  if [ "$a" = "-n" ]; then cat "{lines}"; exit 0; fi\n'
        "done\n"
        f'exec "{RSYNC}" "$@"\n'
    )
    script.chmod(0o755)
    temporaries = "".join(
        f"*deleting   alfworld-luna-c{i % 12}/.status.json.a{i:05d}\n"
        for i in range(80)
    )
    lines.write_text(temporaries)
    assert sync(tmp_path, f"cloud={source}", rsync=str(script)) == S.EXIT_OK
    lines.write_text(
        temporaries
        + "*deleting   alfworld-luna-c7/status.json\n"
        + "*deleting   alfworld-luna-c7/\n"
    )
    capsys.readouterr()
    assert sync(tmp_path, f"cloud={source}", rsync=str(script)) == S.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "would delete 1 whole entry (alfworld-luna-c7)" in err
    assert "--allow-deletes" in err


def test_a_sync_deleting_a_whole_entry_or_a_record_needs_allow_deletes(
    tmp_path, capsys
):
    """A mistyped ``rm`` on a host takes a few whole cells, far fewer
    files than half of the mirror, which may be their only copy. Deleting a top-level
    entry (a cell, a gateway run, _schedule/ ...) or a record (cell.json, status.json,
    results.jsonl, learning.jsonl, attempt.json, gateway.jsonl) anywhere is refused
    without --allow-deletes; what a resume moves aside is not."""
    source = _twelve_cells(tmp_path)
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    before = files(mirror)
    backup = tmp_path / "backup"
    shutil.copytree(source, backup, symlinks=True)

    def refused() -> str:
        capsys.readouterr()
        assert S.main(["--source", f"cloud={source}", "--into", str(merged)]) == (
            S.EXIT_REFUSED
        )
        assert files(mirror) == before  # untouched
        shutil.rmtree(source)
        shutil.copytree(backup, source, symlinks=True)
        return capsys.readouterr().err

    shutil.rmtree(source / "alfworld-luna-c3")
    shutil.rmtree(source / "alfworld-luna-c4-a01-0001")
    err = refused()
    assert (
        "would delete 2 whole entries (alfworld-luna-c3, alfworld-luna-c4-a01-0001)"
        in err
    )
    assert "pass --allow-deletes" in err
    (source / "alfworld-luna-c5" / "attempt-01" / "learning.jsonl").unlink()
    assert "1 record (alfworld-luna-c5/attempt-01/learning.jsonl)" in refused()
    (source / "alfworld-luna-c6" / "status.json").unlink()
    assert "1 record (alfworld-luna-c6/status.json)" in refused()
    shutil.rmtree(source / "_schedule")
    assert "1 whole entry (_schedule)" in refused()
    # What a learning resume moves aside is no record: no flag needed.
    shutil.rmtree(source / "alfworld-luna-c0" / "attempt-01" / "discarded")
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    # The operator says so: the mirror follows.
    shutil.rmtree(source / "alfworld-luna-c3")
    (source / "alfworld-luna-c5" / "attempt-01" / "learning.jsonl").unlink()
    assert sync(tmp_path, f"cloud={source}", allow_deletes=True) == S.EXIT_OK
    assert not (mirror / "alfworld-luna-c3").exists()
    assert not (mirror / "alfworld-luna-c5" / "attempt-01" / "learning.jsonl").exists()


def test_the_deletion_guard_fails_closed_when_its_dry_run_fails(tmp_path, capsys):
    """A dry run cut off by a flaky link (ssh's 255) says nothing about what the sync
    would delete: letting the sync through unchecked, the real transfer that follows
    could delete whole cells from their only copy. A failed dry run refuses the sync."""
    source = _twelve_cells(tmp_path)
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    before = files(mirror)
    flaky = tmp_path / "rsync-flaky-dry-run"
    flaky.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do if [ "$a" = "-n" ]; then '
        'echo "ssh: connection reset" >&2; exit 255; fi; done\n'
        f'exec "{RSYNC}" "$@"\n'
    )
    flaky.chmod(0o755)
    shutil.rmtree(source / "alfworld-luna-c3")  # a mistyped rm on the host
    capsys.readouterr()
    code = S.main(
        ["--source", f"cloud={source}", "--into", str(merged), "--rsync", str(flaky)]
    )
    assert code == S.EXIT_REFUSED  # not EXIT_OK with alfworld-luna-c3 gone
    err = capsys.readouterr().err
    assert "failed (rsync exit 255: ssh: connection reset)" in err
    assert files(mirror) == before
    # A good link: the guard reads the deletion and refuses it as before.
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_REFUSED
    assert "1 whole entry (alfworld-luna-c3)" in capsys.readouterr().err


def test_a_dry_run_exit_23_is_read_not_refused(tmp_path, capsys):
    """macOS openrsync exits 23 for a file that vanished or cannot be read during a
    live run; failing closed on it would refuse every sync of a host with one unreadable
    file. The listing is read: a deletion in it is still refused,
    and without one the guard lets the sync go."""
    source = _twelve_cells(tmp_path)
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    cell = source / "alfworld-luna-c1"
    locked = cell / "worker.log.1"
    locked.write_text("x")
    locked.chmod(0)
    try:
        capsys.readouterr()
        assert sync(tmp_path, f"cloud={source}") != S.EXIT_REFUSED
        assert "the dry run that checks" not in capsys.readouterr().err
        shutil.rmtree(source / "alfworld-luna-c3")  # a mistyped rm on the host
        assert sync(tmp_path, f"cloud={source}") == S.EXIT_REFUSED
        assert "1 whole entry (alfworld-luna-c3)" in capsys.readouterr().err
    finally:
        locked.chmod(0o644)


def test_an_io_error_in_the_dry_run_never_hides_a_deletion(tmp_path, capsys):
    """After an I/O error (exit 23) both rsyncs skip every deletion unless
    --ignore-errors, so a dry run that met one (a directory unreadable for that moment)
    would list none, the guard would pass, and the real run, with no error then, could
    delete a whole cell. The guard's dry run ignores errors."""
    source = _twelve_cells(tmp_path)
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    flaky = tmp_path / "rsync-unreadable-during-the-dry-run"
    target = source / "alfworld-luna-c1"
    flaky.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do if [ "$a" = "-n" ]; then '
        f'chmod 000 "{target}"; "{RSYNC}" "$@"; code=$?; chmod 755 "{target}"; '
        "exit $code; fi; done\n"
        f'exec "{RSYNC}" "$@"\n'
    )
    flaky.chmod(0o755)
    shutil.rmtree(source / "alfworld-luna-c3")  # a mistyped rm on the host
    capsys.readouterr()
    try:
        code = S.main(
            [
                "--source",
                f"cloud={source}",
                "--into",
                str(merged),
                "--rsync",
                str(flaky),
            ]
        )
    finally:
        target.chmod(0o755)
    assert code == S.EXIT_REFUSED  # not EXIT_OK with alfworld-luna-c3 gone
    err = capsys.readouterr().err
    assert "alfworld-luna-c3" in err
    assert "the dry run met an I/O error, rsync exit 23" in err
    assert (mirror / "alfworld-luna-c3").is_dir()


def test_allow_deletes_never_lifts_the_identity_guard(tmp_path, capsys):
    """--allow-deletes lets a mirror follow what its source deleted, never
    follow another runs directory: that takes --allow-other-source, and what following it
    deletes still takes --allow-deletes."""
    source = _twelve_cells(tmp_path)
    merged = tmp_path / "merged"
    mirror = merged / "_hosts" / "cloud"
    assert sync(tmp_path, f"cloud={source}") == S.EXIT_OK
    before = files(mirror)

    def refused(*extra: str) -> str:
        capsys.readouterr()
        assert (
            S.main(["--source", f"cloud={source}", "--into", str(merged), *extra])
            == S.EXIT_REFUSED
        )
        assert files(mirror) == before
        return capsys.readouterr().err

    shutil.rmtree(source)
    host_runs(source, ("webshop-luna-none",))
    _write(source / "_gateway" / "starts.jsonl", '{"start_id": "gw-other"}\n')
    err = refused("--allow-deletes")
    assert "is another runs directory" in err and "--allow-other-source" in err
    err = refused("--allow-other-source")
    assert "would delete" in err and "--allow-deletes" in err
    assert (
        sync(tmp_path, f"cloud={source}", allow_deletes=True, allow_other_source=True)
        == S.EXIT_OK
    )
    assert read_json(merged / MERGED_MARKER)["sources"]["cloud"]["first_start"] == (
        "gw-other"
    )
    assert (mirror / "webshop-luna-none" / "cell.json").is_file()
    # An emptied runs directory: the same.
    before = files(mirror)
    shutil.rmtree(source)
    source.mkdir()
    assert "has no _gateway/starts.jsonl" in refused("--allow-deletes")


def test_a_mirror_is_never_a_link_into_a_runs_directory(tmp_path, two_hosts, capsys):
    """A mirror symlinked into a live runs directory ("to save a copy") would get
    _merged.json written into it, and run.py would then refuse every cell there. The
    tool makes _hosts/ and each mirror itself: a symlink there, or an --into inside a
    runs directory, is refused before anything is written."""
    cloud, local = two_hosts
    into = tmp_path / "merged"

    def refused(*argv: str) -> str:
        capsys.readouterr()
        assert S.main(list(argv)) == S.EXIT_REFUSED
        assert not (local / MERGED_MARKER).exists()
        return capsys.readouterr().err

    (into / "_hosts").mkdir(parents=True)
    (into / "_hosts" / "local").symlink_to(local)
    err = refused("--source", f"local={local}", "--into", str(into))
    assert "_hosts/local is a symlink" in err
    # Whichever sources this call syncs.
    assert "is a symlink" in refused("--source", f"cloud={cloud}", "--into", str(into))
    assert not (into / "_hosts" / "cloud").exists()
    # _hosts itself.
    shutil.rmtree(into)
    into.mkdir()
    (into / "_hosts").symlink_to(local.parent)
    assert "_hosts is a symlink" in refused(
        "--source", f"local={local}", "--into", str(into)
    )
    # An --into inside a cell, or inside another view's mirror, even of a runs
    # directory this call does not sync.
    cell = local / "webshop-luna-none"
    err = refused(
        "--source", f"cloud={cloud}", "--mirror-only", "--into", str(cell / "v")
    )
    assert f"is inside {cell}, which holds cell.json" in err
    assert not (cell / "v").exists()
    shutil.rmtree(into)
    assert sync(tmp_path, f"cloud={cloud}", mirror_only=True) == S.EXIT_OK
    nested = into / "_hosts" / "cloud" / "view"
    err = refused("--source", f"local={local}", "--into", str(nested))
    assert "which holds _merged.json" in err and not nested.exists()
    # An --into that is a symlink to (or into) a runs directory this call syncs.
    link = tmp_path / "link"
    link.symlink_to(local)
    assert "overlaps" in refused("--source", f"local={local}", "--into", str(link))
    # Runs directories nest by design: a view beside them in a directory that has a
    # _gateway/ of its own (a gateway started with the default --runs-dir) is fine.
    runs = tmp_path / "runs"
    _write(runs / "_gateway" / "starts.jsonl", '{"start_id": "gw-default"}\n')
    assert sync(tmp_path, f"cloud={cloud}", into=runs / "view") == S.EXIT_OK


def test_one_sync_at_a_time(tmp_path, two_hosts, capsys):
    """Removing an interrupted sync's staging directories is safe only
    while no other sync of the view runs, so a second one is refused (exit 2); a dry
    run only reads, and runs."""
    cloud, local = two_hosts
    into = tmp_path / "merged"
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    with open(into / S.LOCK_FILE, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        capsys.readouterr()
        assert sync(tmp_path, f"cloud={cloud}") == S.EXIT_REFUSED
        assert "another sync of" in capsys.readouterr().err
        assert sync(tmp_path, f"cloud={cloud}", dry_run=True) == S.EXIT_OK
    assert sync(tmp_path, f"cloud={cloud}") == S.EXIT_OK
    # The lock file is no entry of the view.
    assert S.plan_view(into).conflicts == []
    assert S.LOCK_FILE not in read_json(into / MERGED_MARKER)["entries"]


def test_what_the_readers_are_told_about_a_view(tmp_path, two_hosts):
    """audit.py and report.py show what _merged.json says: when the view
    was built and each host synced, and why it is not final (a --mirror-only sync, or a
    sync stopped by a conflict or an rsync failure, since it was built; a host whose last
    rsync did not exit 0; each conflict)."""
    cloud, local = two_hosts
    into = tmp_path / "merged"
    assert S.merged_view_status(into) is None
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    status = S.merged_view_status(into)
    assert status["kind"] == "view" and status["final"] is True
    assert status["warnings"] == [] and set(status["hosts"]) == {"cloud", "local"}
    assert 0 <= status["age_hours"] < 1
    built = datetime.fromisoformat(status["view_built_at"])
    later = S.merged_view_status(into, at=built + timedelta(hours=5))
    assert later["age_hours"] == pytest.approx(5.0, abs=0.01)
    assert "view built" in status["summary"] and "cloud" in status["summary"]

    def checks(status: dict[str, Any]) -> list[str]:
        return [w["check"] for w in status["warnings"]]

    # A --mirror-only sync since: not final.
    assert sync(tmp_path, f"cloud={cloud}", mirror_only=True) == S.EXIT_OK
    status = S.merged_view_status(into)
    assert status["final"] is False and checks(status) == ["merged_view_not_final"]
    assert "--mirror-only" in status["warnings"][0]["detail"]
    assert "cloud" in status["warnings"][0]["detail"]
    # A full sync makes it final again.
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    assert S.merged_view_status(into)["final"] is True
    # A conflict: each is named, and the view is not final.
    shutil.copytree(cloud / "alfworld-luna-none", local / "alfworld-luna-none")
    _write(local / "alfworld-luna-none" / "attempt-01" / "learning.jsonl", "{}\n{}\n")
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_CONFLICT
    status = S.merged_view_status(into)
    assert checks(status) == ["merged_view_not_final", "merged_view_conflict"]
    assert "stopped by conflicts" in status["warnings"][0]["detail"]
    assert (
        "alfworld-luna-none from cloud, local (different content)"
        in (status["warnings"][1]["detail"])
    )
    assert status["conflicts"][0]["entry"] == "alfworld-luna-none"
    # Resolved (kept on one host only; the mirror drops the other copy on the
    # operator's word), with an rsync that saw files vanish (a run writing).
    shutil.rmtree(local / "alfworld-luna-none")
    assert (
        sync(
            tmp_path,
            f"cloud={cloud}",
            f"local={local}",
            rsync=_fake_rsync(tmp_path, 24),
            allow_deletes=True,
        )
        == S.EXIT_OK
    )
    status = S.merged_view_status(into)
    assert checks(status) == ["merged_view_not_final", "merged_view_not_final"]
    assert "exited 24" in " ".join(w["detail"] for w in status["warnings"])
    # A view never built (--mirror-only from scratch), a host's mirror, and an
    # unreadable record.
    fresh = tmp_path / "fresh"
    assert sync(tmp_path, f"cloud={cloud}", into=fresh, mirror_only=True) == S.EXIT_OK
    status = S.merged_view_status(fresh)
    assert status["final"] is False and "never built" in status["warnings"][0]["detail"]
    mirror = S.merged_view_status(fresh / "_hosts" / "cloud")
    assert mirror["kind"] == "mirror" and mirror["warnings"] == []
    assert str(cloud) in mirror["summary"]
    (fresh / MERGED_MARKER).write_text("{")
    assert checks(S.merged_view_status(fresh)) == ["merged_view_unreadable"]


# --- _gateway/starts.jsonl copied last -------------------------------------------------------


def test_starts_jsonl_is_copied_after_everything_else(tmp_path, monkeypatch):
    """A gateway that starts on the host while the main rsync runs appends its start
    record there: the starts file is copied once more, last, so the mirror holds the
    start of every ledger line it holds (the audit's gateway_start_missing)."""
    source = host_runs(tmp_path / "local", ("webshop-luna-none",))
    real_run = subprocess.run
    calls: list[list[str]] = []

    def run(argv, *args, **kwargs):
        calls.append(list(argv))
        result = real_run(argv, *args, **kwargs)
        if "--delete-after" in argv and "-n" not in argv:
            # A gateway starts on the host during the main rsync: a new run's ledger
            # line lands after its start record, both after rsync read the directory.
            with (source / "_gateway" / "starts.jsonl").open("a") as handle:
                handle.write('{"start_id": "gw-late"}\n')
        return result

    monkeypatch.setattr(S.subprocess, "run", run)
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    mirror = tmp_path / "merged" / "_hosts" / "local"
    syncs = [c for c in calls if c[1:2] == ["-a"]]
    assert syncs[-1] == S.starts_argv(S.parse_source(f"local={source}"), mirror)
    assert "--delete-after" in syncs[-2]
    assert (mirror / "_gateway" / "starts.jsonl").read_text() == (
        source / "_gateway" / "starts.jsonl"
    ).read_text()
    assert "gw-late" in (mirror / "_gateway" / "starts.jsonl").read_text()
    record = read_json(tmp_path / "merged" / MERGED_MARKER)["sources"]["local"]
    assert record["starts_exit"] == 0 and record["starts_synced_at"]
    assert record["rsync_exit"] == 0


def test_the_starts_copy_of_a_remote_source_goes_over_ssh():
    remote = S.parse_source("cloud=user@remote:runs/study")
    argv = S.starts_argv(remote, Path("/m/_hosts/cloud"))
    assert argv[-2:] == [
        "user@remote:runs/study/_gateway/starts.jsonl",
        "/m/_hosts/cloud/_gateway/",
    ]
    assert argv[argv.index("-e") + 1] == S.SSH_COMMAND
    assert "-z" in argv and "--timeout=600" in argv
    local = S.starts_argv(S.parse_source("local=/r"), Path("/m/_hosts/local"))
    assert local == [
        "rsync",
        "-a",
        "/r/_gateway/starts.jsonl",
        "/m/_hosts/local/_gateway/",
    ]


def _rsync_failing_on(tmp_path: Path, needle: str, code: int) -> str:
    """An rsync that exits ``code`` (doing nothing) when an argument contains
    ``needle``, and is the real one otherwise."""
    script = tmp_path / f"rsync-fail-{code}"
    script.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in *' + needle + "*) "
        f"echo 'failed on purpose' >&2; exit {code};; esac; done\n"
        f'exec "{RSYNC}" "$@"\n'
    )
    script.chmod(0o755)
    return str(script)


def test_a_failed_starts_copy_leaves_the_view(tmp_path, two_hosts, capsys):
    cloud, local = two_hosts
    into = tmp_path / "merged"
    rsync = _rsync_failing_on(tmp_path, "_gateway/starts.jsonl", 12)
    assert (
        sync(tmp_path, f"cloud={cloud}", f"local={local}", rsync=rsync) == S.EXIT_RSYNC
    )
    err = capsys.readouterr().err
    assert "error: cloud: rsync failed (exit 12)" in err
    assert "copy of _gateway/starts.jsonl failed (exit 12)" in err
    assert not any(p.is_symlink() for p in into.iterdir())
    record = read_json(into / MERGED_MARKER)
    assert record["sources"]["cloud"]["rsync_exit"] == 0
    assert record["sources"]["cloud"]["starts_exit"] == 12
    assert record["sources"]["cloud"]["starts_synced_at"] is None
    # The main rsync brought the cells: the mirror is marked read-only all the same.
    assert read_json(into / "_hosts" / "cloud" / MERGED_MARKER)["mirror_of"] == str(
        cloud
    )
    status = S.merged_view_status(into)
    assert status["final"] is False
    assert any("starts.jsonl exited 12" in w["detail"] for w in status["warnings"])
    # The next sync, with a working rsync, builds the view.
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    assert S.merged_view_status(into)["final"] is True


def test_a_source_whose_gateway_never_started_syncs(tmp_path, capsys):
    """No starts file on the host yet (rsync exit 23 for that one file): fine."""
    source = host_runs(tmp_path / "local", ("webshop-luna-none",))
    (source / "_gateway" / "starts.jsonl").unlink()
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    record = read_json(tmp_path / "merged" / MERGED_MARKER)["sources"]["local"]
    assert record["starts_exit"] == S.RSYNC_PARTIAL
    assert record["starts_synced_at"] is None
    assert (tmp_path / "merged" / "webshop-luna-none").is_symlink()


# --- entries removed on purpose ----------------------------------------------------------------


def _fake_removed_entries_from_index(
    text: str, *, finished_only: bool = False
) -> frozenset[str]:
    """The index reader, as tools/remove_method_data.py specifies it."""
    record = json.loads(text)
    return frozenset(
        entry
        for removal in record.get("removals") or ()
        if removal.get("state")
        in (("complete",) if finished_only else ("deleting", "complete"))
        for entry in removal.get("entries") or ()
    )


@pytest.fixture()
def removal_tool(monkeypatch, removed_method):
    """tools/remove_method_data.py, or a stand-in with its index reader when it cannot
    be imported; :data:`REMOVED_METHOD` is recorded as removed from the study."""
    try:
        from evals.skill_bench.tools import remove_method_data
    except ImportError:
        from evals.skill_bench import tools

        remove_method_data = types.ModuleType(
            "evals.skill_bench.tools.remove_method_data"
        )
        remove_method_data.REMOVED_DIR = "_removed"
        remove_method_data.INDEX_FILE = "_removed/index.json"
        remove_method_data.removed_entries_from_index = _fake_removed_entries_from_index
        monkeypatch.setitem(
            sys.modules, remove_method_data.__name__, remove_method_data
        )
        monkeypatch.setattr(
            tools, "remove_method_data", remove_method_data, raising=False
        )
    assert S.REMOVED_DIR == remove_method_data.REMOVED_DIR
    assert S.REMOVED_INDEX == remove_method_data.INDEX_FILE
    return remove_method_data


def _removal(
    runs: Path,
    entries: list[str],
    *,
    state: str = "complete",
    rid: str = "20261001T000000Z-ab12",
) -> str:
    """What remove_method_data leaves: a manifest and the index listing ``entries``."""
    manifest = f"{REMOVED_METHOD}/{rid}/manifest.json"
    _write(
        runs / "_removed" / manifest,
        json.dumps(
            {"schema": 1, "kind": "skill_bench.removed_method_data", "state": state}
        ),
    )
    index = runs / "_removed" / "index.json"
    removals = read_json(index)["removals"] if index.is_file() else []
    removals = [r for r in removals if r["removal_id"] != rid]
    removals.append(
        {
            "removal_id": rid,
            "method": REMOVED_METHOD,
            "state": state,
            "manifest": manifest,
            "entries": entries,
            "created_at": "2026-10-01T00:00:00+00:00",
            "completed_at": "2026-10-01T00:01:00+00:00"
            if state == "complete"
            else None,
        }
    )
    _write(index, json.dumps({"schema": 1, "removals": removals}))
    return manifest


DC = f"webshop-luna-{REMOVED_METHOD}"


def test_entries_removed_on_purpose_leave_the_mirror_and_the_view_without_allow_deletes(
    tmp_path, capsys, removal_tool
):
    source = host_runs(tmp_path / "local", ("webshop-luna-none", DC))
    into = tmp_path / "merged"
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert (into / DC).is_symlink() and (into / f"{DC}-a01-0001").is_symlink()
    # The removal tool on the host: manifest and index first, then the deletions.
    manifest = _removal(source, [DC, f"{DC}-a01-0001"])
    shutil.rmtree(source / DC)
    shutil.rmtree(source / f"{DC}-a01-0001")
    capsys.readouterr()
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK  # no --allow-deletes
    out = capsys.readouterr().out
    assert f"local: 2 entries removed on purpose ({manifest})" in out
    assert "2 deleted from the mirror now" in out
    mirror = into / "_hosts" / "local"
    assert not (mirror / DC).exists() and not (mirror / f"{DC}-a01-0001").exists()
    assert not (into / DC).is_symlink() and not (into / f"{DC}-a01-0001").is_symlink()
    assert (mirror / "webshop-luna-none" / "cell.json").is_file()
    assert (mirror / "_blobs" / BLOB).is_file() and (
        into / "_blobs" / BLOB
    ).is_symlink()
    # The manifests are mirrored (the report reads them there), never linked.
    assert (mirror / "_removed" / manifest).is_file()
    assert not (into / "_removed").exists()
    record = read_json(into / MERGED_MARKER)
    assert record["removed_on_purpose"] == {
        "local": {DC: manifest, f"{DC}-a01-0001": manifest}
    }
    assert record["sources"]["local"]["removed_from_mirror"] == [DC, f"{DC}-a01-0001"]
    assert record["sources"]["local"]["removed_manifests"] == [manifest]
    assert record["sources"]["local"]["removed_problems"] == []
    assert DC not in record["entries"]
    status = S.merged_view_status(into)
    assert status["final"] is True
    assert status["removed_on_purpose"] == record["removed_on_purpose"]
    first_synced = record["sources"]["local"]["synced_at"]
    deleted_at = {DC: first_synced, f"{DC}-a01-0001": first_synced}
    assert record["sources"]["local"]["removed_from_mirror_at"] == deleted_at
    # Again: nothing left to remove, nothing refused. This sync deleted nothing, and the
    # record still says which entries an earlier sync deleted, and when.
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    record = read_json(into / MERGED_MARKER)
    assert record["sources"]["local"]["removed_from_mirror"] == []
    assert record["sources"]["local"]["removed_from_mirror_at"] == deleted_at
    assert "0 deleted from the mirror now" in capsys.readouterr().out
    # A failed sync keeps them too.
    rsync = _rsync_failing_on(tmp_path, "_gateway/starts.jsonl", 12)
    assert sync(tmp_path, f"local={source}", rsync=rsync) == S.EXIT_RSYNC
    record = read_json(into / MERGED_MARKER)
    assert record["sources"]["local"]["removed_from_mirror_at"] == deleted_at


def test_removed_from_mirror_at_carries_a_record_written_before_it_existed():
    # A _merged.json written before removed_from_mirror_at existed has only this
    # sync's list: it is carried
    # at that record's own synced_at; an entry deleted again keeps its first time.
    old = {"removed_from_mirror": ["a", "b"], "synced_at": "T1"}
    assert S.removed_from_mirror_at(old, ["b", "c"], "T2") == {
        "a": "T1",
        "b": "T1",
        "c": "T2",
    }
    new = {"removed_from_mirror_at": {"a": "T0"}, "removed_from_mirror": []}
    assert S.removed_from_mirror_at(new, [], "T3") == {"a": "T0"}
    assert S.removed_from_mirror_at({}, [], "T3") == {}
    bad = {"removed_from_mirror_at": ["not", "a", "map"], "removed_from_mirror": "x"}
    assert S.removed_from_mirror_at(bad, ["d"], "T4") == {"d": "T4"}


def test_an_unfinished_removal_excludes_but_keeps_the_mirror_copy(
    tmp_path, capsys, removal_tool
):
    source = host_runs(tmp_path / "local", ("webshop-luna-none", DC))
    into = tmp_path / "merged"
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    _removal(source, [DC, f"{DC}-a01-0001"], state="deleting")
    # The tool moved the cell into its trash (a .*.tmp name, never mirrored) and was
    # cut off before the gateway run.
    trash = source / "_removed" / REMOVED_METHOD / "x" / ".trash.tmp"
    trash.mkdir(parents=True)
    os.rename(source / DC, trash / DC)
    capsys.readouterr()
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    captured = capsys.readouterr()
    assert "0 deleted from the mirror now" in captured.out
    assert "listed by a removal not finished on the host" in captured.err
    mirror = into / "_hosts" / "local"
    assert (mirror / DC / "cell.json").is_file()  # kept: the removal may not finish
    assert (mirror / f"{DC}-a01-0001" / "gateway.jsonl").is_file()
    assert not (into / DC).is_symlink() and not (into / f"{DC}-a01-0001").is_symlink()
    assert not (mirror / "_removed" / REMOVED_METHOD / "x" / ".trash.tmp").exists()
    # The removal completes on the host: the mirror follows.
    shutil.rmtree(trash)
    shutil.rmtree(source / f"{DC}-a01-0001")
    _removal(source, [DC, f"{DC}-a01-0001"], state="complete")
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert not (mirror / DC).exists() and not (mirror / f"{DC}-a01-0001").exists()


def test_a_listed_entry_the_host_still_holds_is_not_copied_back(
    tmp_path, capsys, removal_tool
):
    source = host_runs(tmp_path / "local", ("webshop-luna-none", DC))
    _removal(source, [DC])  # the host still holds (part of) it
    fresh = tmp_path / "fresh"
    assert sync(tmp_path, f"local={source}", into=fresh) == S.EXIT_OK
    mirror = fresh / "_hosts" / "local"
    assert not (mirror / DC).exists() and not (fresh / DC).exists()
    # Its gateway run is not listed: copied and linked like any other.
    assert (fresh / f"{DC}-a01-0001").is_symlink()
    # A sync of an existing mirror that holds it deletes it from the mirror only.
    into = tmp_path / "merged"
    (source / "_removed" / "index.json").rename(tmp_path / "index.json")
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert (into / "_hosts" / "local" / DC).is_dir()
    (tmp_path / "index.json").rename(source / "_removed" / "index.json")
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    assert not (into / "_hosts" / "local" / DC).exists() and not (into / DC).exists()
    assert (source / DC / "cell.json").is_file()  # the host's copy is never touched


def test_an_unlisted_deletion_still_needs_allow_deletes(tmp_path, capsys, removal_tool):
    source = host_runs(
        tmp_path / "local", ("webshop-luna-none", "webshop-luna-awm", DC)
    )
    into = tmp_path / "merged"
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    _removal(source, [DC, f"{DC}-a01-0001"])
    for entry in (DC, f"{DC}-a01-0001", "webshop-luna-awm"):
        shutil.rmtree(source / entry)
    capsys.readouterr()
    assert sync(tmp_path, f"local={source}") == S.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "would delete 1 whole entry (webshop-luna-awm)" in err
    assert DC not in err.split("would delete", 1)[1].split("from its mirror")[0]
    assert (into / "_hosts" / "local" / DC).is_dir()  # nothing changed
    assert (into / DC).is_symlink()
    assert sync(tmp_path, f"local={source}", allow_deletes=True) == S.EXIT_OK
    assert not (into / "webshop-luna-awm").is_symlink() and not (into / DC).is_symlink()


def test_a_bad_index_entry_is_ignored_with_a_warning(tmp_path, capsys, removal_tool):
    source = host_runs(tmp_path / "local", ("webshop-luna-none", DC))
    into = tmp_path / "merged"
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    bad = ["_blobs", "_gateway", ".hidden", "a/b", "..", "webshop-luna-none"]
    _removal(source, [*bad, DC])
    shutil.rmtree(source / DC)
    capsys.readouterr()
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    captured = capsys.readouterr()
    for name in bad:
        assert f"'{name}' ignored" in captured.err
    mirror = into / "_hosts" / "local"
    assert (mirror / "_blobs" / BLOB).is_file() and (mirror / "_gateway").is_dir()
    assert not (mirror / DC).exists()
    # The study cell a wrong index named stays mirrored and linked.
    assert (mirror / "webshop-luna-none" / "cell.json").is_file()
    assert (into / "webshop-luna-none").is_symlink()
    record = read_json(into / MERGED_MARKER)
    assert len(record["sources"]["local"]["removed_problems"]) == len(bad)
    assert set(record["removed_on_purpose"]["local"]) == {DC}
    # And its deletion on the host is guarded as always.
    shutil.rmtree(source / "webshop-luna-none")
    assert sync(tmp_path, f"local={source}") == S.EXIT_REFUSED
    assert "(webshop-luna-none)" in capsys.readouterr().err


def test_an_unreadable_index_excludes_nothing(tmp_path, capsys, removal_tool):
    source = host_runs(tmp_path / "local", ("webshop-luna-none", DC))
    assert sync(tmp_path, f"local={source}") == S.EXIT_OK
    _write(source / "_removed" / "index.json", "{not json")
    shutil.rmtree(source / DC)
    capsys.readouterr()
    # The guard sees the deletion as it would without any index.
    assert sync(tmp_path, f"local={source}") == S.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "_removed/index.json unreadable" in err and f"({DC})" in err


def test_the_index_reader(removed_method):
    dc2 = f"alfworld-luna56-{removed_method}-smoke-l2-t3"
    text = json.dumps(
        {
            "schema": 1,
            "removals": [
                {
                    "removal_id": "r1",
                    "method": REMOVED_METHOD,
                    "state": "complete",
                    "manifest": f"{REMOVED_METHOD}/r1/manifest.json",
                    "entries": [DC, f"{DC}-a01-0f3e", f"{DC}-a02"],
                },
                {
                    "removal_id": "r2",
                    "method": REMOVED_METHOD,
                    "state": "deleting",
                    "entries": [dc2, "webshop-luna-none", "webshop-luna-none-a01-0001"],
                },
            ],
        }
    )

    fake = types.ModuleType("evals.skill_bench.tools.remove_method_data")
    fake.removed_entries_from_index = _fake_removed_entries_from_index
    saved = sys.modules.get(fake.__name__)
    sys.modules[fake.__name__] = fake
    try:
        removed = S.removed_from_index(text)
    finally:
        if saved is None:
            sys.modules.pop(fake.__name__)
        else:
            sys.modules[fake.__name__] = saved
    r1, r2 = (
        f"{REMOVED_METHOD}/r1/manifest.json",
        f"{REMOVED_METHOD}/r2/manifest.json",
    )
    assert removed.entries == {DC: r1, f"{DC}-a01-0f3e": r1, f"{DC}-a02": r1, dc2: r2}
    assert removed.finished == {DC, f"{DC}-a01-0f3e", f"{DC}-a02"}
    assert removed.manifests == [r1, r2]
    # A study cell and its gateway run are never excluded nor deleted, whatever an
    # index says: the removal tool removes a removed method's data only.
    assert [p.split(" ignored")[0].rsplit(": ", 1)[1] for p in removed.problems] == [
        "'webshop-luna-none'",
        "'webshop-luna-none-a01-0001'",
    ]
    assert all(
        "not a cell or gateway run of a removed method" in p for p in removed.problems
    )
    argv = S.rsync_argv(S.parse_source("l=/r"), Path("/m"), extra_excludes=["c2", "c1"])
    assert argv[-4:-2] == ["--exclude=/c1/", "--exclude=/c2/"]
    for bad in ("a*", "b?", "[x]", " lead", "a\\b"):
        assert S._entry_problem(bad) == "not a top-level entry name"


def _ssh_rsync(tmp_path: Path, root: Path, *, index_exit: int | None = None) -> str:
    """An rsync for the remote source ``host:<path>``: ``host:`` paths are read under
    ``root`` and ssh options dropped; ``index_exit`` fails the index read (ssh down)."""
    script = tmp_path / "rsync-ssh"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['--version']:\n"
        f"    os.execv({RSYNC!r}, [{RSYNC!r}, '--version'])\n"
        f"if {index_exit!r} is not None and any(a.endswith('_removed/index.json') for a in args):\n"
        "    sys.stderr.write('ssh: connect to host host port 22: Connection refused\\n')\n"
        f"    sys.exit({index_exit!r})\n"
        "out, skip = [], False\n"
        "for a in args:\n"
        "    if skip:\n"
        "        skip = False\n"
        "        continue\n"
        "    if a == '-e':\n"
        "        skip = True\n"
        "        continue\n"
        "    if a.startswith('host:'):\n"
        f"        a = os.path.join({str(root)!r}, a[len('host:'):])\n"
        "    out.append(a)\n"
        f"sys.exit(subprocess.run([{RSYNC!r}, *out]).returncode)\n"
    )
    script.chmod(0o755)
    return str(script)


def test_a_remote_sources_removals_are_read_with_one_rsync(
    tmp_path, capsys, removal_tool
):
    root = tmp_path / "remote-home"
    source = host_runs(root / "runs", ("webshop-luna-none", DC))
    rsync = _ssh_rsync(tmp_path, root)
    into = tmp_path / "merged"
    assert sync(tmp_path, "cloud=host:runs", rsync=rsync) == S.EXIT_OK
    assert (into / DC).is_symlink()
    manifest = _removal(source, [DC])
    shutil.rmtree(source / DC)
    assert sync(tmp_path, "cloud=host:runs", rsync=rsync) == S.EXIT_OK
    assert (
        not (into / "_hosts" / "cloud" / DC).exists() and not (into / DC).is_symlink()
    )
    assert read_json(into / MERGED_MARKER)["removed_on_purpose"] == {
        "cloud": {DC: manifest}
    }
    # The index cannot be read (ssh down): nothing is synced.
    capsys.readouterr()
    down = _ssh_rsync(tmp_path, root, index_exit=255)
    assert sync(tmp_path, "cloud=host:runs", rsync=down) == S.EXIT_RSYNC
    err = capsys.readouterr().err
    assert "cannot read its _removed/index.json (rsync exit 255)" in err
    assert "nothing was synced" in err


def test_a_remote_source_without_removals(tmp_path, removal_tool):
    root = tmp_path / "remote-home"
    host_runs(root / "runs", ("webshop-luna-none",))
    assert (
        sync(tmp_path, "cloud=host:runs", rsync=_ssh_rsync(tmp_path, root)) == S.EXIT_OK
    )
    record = read_json(tmp_path / "merged" / MERGED_MARKER)
    assert record["removed_on_purpose"] == {}
    assert record["sources"]["cloud"]["starts_exit"] == 0


# --- two runs of one cell id on two hosts ------------------------------------------------------


def _independent_cell(
    runs: Path, cell: str, *, instance: str, created_at: str | None, run: str
) -> None:
    host_runs(runs, (cell,))
    shutil.rmtree(runs / f"{cell}-a01-0001")
    _write(runs / run / "gateway.jsonl", f'{{"cell": "{cell}"}}\n')
    manifest: dict[str, Any] = {
        "schema": 1,
        "cell": {"cell_id": cell},
        "instance": instance,
    }
    if created_at is not None:
        manifest["created_at"] = created_at
    _write(runs / cell / "cell.json", json.dumps(manifest))
    _write(runs / cell / "attempt-01" / "attempt.json", json.dumps({"run_id": run}))


MIRIX = "webshop-luna-mirix"


def test_two_runs_of_one_cell_id_on_two_hosts_are_both_linked(tmp_path, capsys):
    cloud, local = tmp_path / "cloud" / "runs", tmp_path / "local" / "runs"
    _independent_cell(
        local,
        MIRIX,
        instance="i-mac",
        created_at="2026-09-28T01:00:00+00:00",
        run=f"{MIRIX}-a01-1111",
    )
    _independent_cell(
        cloud,
        MIRIX,
        instance="i-cloud",
        created_at="2026-10-02T01:00:00+00:00",
        run=f"{MIRIX}-a01-2222",
    )
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    into = tmp_path / "merged"
    assert os.readlink(into / MIRIX) == f"_hosts/local/{MIRIX}"  # the earliest
    assert os.readlink(into / f"{MIRIX}@cloud") == f"_hosts/cloud/{MIRIX}"
    assert os.readlink(into / f"{MIRIX}-a01-1111") == f"_hosts/local/{MIRIX}-a01-1111"
    assert os.readlink(into / f"{MIRIX}-a01-2222") == f"_hosts/cloud/{MIRIX}-a01-2222"
    out = capsys.readouterr().out
    assert f"{MIRIX}: independent cells of one id on several hosts, all linked" in out
    assert "2 cells, 2 gateway runs" in out
    record = read_json(into / MERGED_MARKER)
    assert (
        record["entries"][MIRIX] == "local"
        and record["entries"][f"{MIRIX}@cloud"] == "cloud"
    )
    assert record["links"] == {f"{MIRIX}@cloud": MIRIX}
    assert record["cells_on_several_hosts"] == {
        MIRIX: {"cloud": f"{MIRIX}@cloud", "local": MIRIX}
    }
    assert record["conflicts"] == []
    status = S.merged_view_status(into)
    assert status["final"] is True and status["warnings"] == []
    assert status["cells_on_several_hosts"] == record["cells_on_several_hosts"]
    # The readers see two directories of one cell id.
    assert [
        (p.name, read_json(p / "cell.json")["cell"]["cell_id"])
        for p in report.discover_cells(into)
    ] == [
        (MIRIX, MIRIX),
        (f"{MIRIX}@cloud", MIRIX),
    ]
    # Again: stable.
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    assert "0 links made, 0 removed" in capsys.readouterr().out


def test_the_plain_name_stays_with_the_host_it_already_links(tmp_path, capsys):
    cloud, local = tmp_path / "cloud" / "runs", tmp_path / "local" / "runs"
    _independent_cell(
        cloud,
        MIRIX,
        instance="i-cloud",
        created_at="2026-10-02T01:00:00+00:00",
        run=f"{MIRIX}-a01-2222",
    )
    assert sync(tmp_path, f"cloud={cloud}") == S.EXIT_OK
    into = tmp_path / "merged"
    assert os.readlink(into / MIRIX) == f"_hosts/cloud/{MIRIX}"
    # An older cell of that id appears on the other host: the link does not move.
    _independent_cell(
        local,
        MIRIX,
        instance="i-mac",
        created_at="2026-09-28T01:00:00+00:00",
        run=f"{MIRIX}-a01-1111",
    )
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_OK
    assert os.readlink(into / MIRIX) == f"_hosts/cloud/{MIRIX}"
    assert os.readlink(into / f"{MIRIX}@local") == f"_hosts/local/{MIRIX}"
    # Ties and missing times: a missing time sorts last, then the host name.
    created = {"b": None, "a": None, "c": S._time("2026-01-01T00:00:00+00:00")}
    assert S._plain_name_host(tmp_path / "nowhere", MIRIX, created) == "c"
    assert (
        S._plain_name_host(tmp_path / "nowhere", MIRIX, {"b": None, "a": None}) == "a"
    )


@pytest.mark.parametrize("broken", ["unreadable", "no_instance", "same_instance"])
def test_cells_of_one_name_that_cannot_be_told_apart_are_a_conflict(
    tmp_path, capsys, broken
):
    """sync-faithful: only readable, different instances are independent cells; a
    copied cell that diverged, or one whose cell.json cannot say, stops the tool."""
    cloud, local = tmp_path / "cloud" / "runs", tmp_path / "local" / "runs"
    _independent_cell(
        local,
        MIRIX,
        instance="i-mac",
        created_at="2026-09-28T01:00:00+00:00",
        run=f"{MIRIX}-a01-1111",
    )
    _independent_cell(
        cloud,
        MIRIX,
        instance="i-cloud",
        created_at="2026-10-02T01:00:00+00:00",
        run=f"{MIRIX}-a01-2222",
    )
    if broken == "unreadable":
        _write(cloud / MIRIX / "cell.json", "{")
    elif broken == "no_instance":
        _write(cloud / MIRIX / "cell.json", json.dumps({"cell": {"cell_id": MIRIX}}))
    else:
        manifest = read_json(local / MIRIX / "cell.json")
        _write(cloud / MIRIX / "cell.json", json.dumps(manifest))
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_CONFLICT
    assert (
        f"conflict: {MIRIX} from cloud, local (different content)"
        in capsys.readouterr().err
    )
    assert not (tmp_path / "merged" / MIRIX).exists()


def test_a_link_name_an_entry_already_has_is_a_conflict(tmp_path, capsys):
    cloud, local = tmp_path / "cloud" / "runs", tmp_path / "local" / "runs"
    _independent_cell(
        local,
        MIRIX,
        instance="i-mac",
        created_at="2026-09-28T01:00:00+00:00",
        run=f"{MIRIX}-a01-1111",
    )
    _independent_cell(
        cloud,
        MIRIX,
        instance="i-cloud",
        created_at="2026-10-02T01:00:00+00:00",
        run=f"{MIRIX}-a01-2222",
    )
    _independent_cell(
        local, f"{MIRIX}@cloud", instance="i-odd", created_at=None, run="odd-a01-3333"
    )
    assert sync(tmp_path, f"cloud={cloud}", f"local={local}") == S.EXIT_CONFLICT
    assert f"conflict: {MIRIX} from cloud, local" in capsys.readouterr().err


# --- the readers over a merged view --------------------------------------------------------------


def _split_by_benchmark(matrix: Matrix, root: Path) -> dict[str, Path]:
    """The matrix as two hosts run it (ALFWorld on a remote host, WebShop here): each
    with its own gateway directory and blobs, and the absolute paths its records hold
    naming that host's tree, which does not exist on this one."""
    cells = sorted(matrix.outcomes)
    local_benchmarks = {"webshop"}
    split = {
        "cloud": [c for c in cells if c.split("-")[0] not in local_benchmarks],
        "local": [c for c in cells if c.split("-")[0] in local_benchmarks],
    }
    hosts = {}
    for host, host_cells in split.items():
        runs = copy_cells(matrix, root / host, host_cells)
        if (matrix.runs / "_blobs").is_dir():
            shutil.copytree(matrix.runs / "_blobs", runs / "_blobs")
        elsewhere = Path(f"/nonexistent-{host}/checkout/runs")
        for cell_id in host_cells:
            paths = CellPaths(runs / cell_id)
            manifest = read_json(paths.manifest)
            manifest["blob_dir"] = str(elsewhere / "_blobs")
            write_json_atomic(paths.manifest, manifest)
            for number in paths.attempt_numbers():
                state = read_json(paths.attempt(number).state)
                state["ledger"] = str(elsewhere / state["run_id"] / "gateway.jsonl")
                write_json_atomic(paths.attempt(number).state, state)
        hosts[host] = runs
    return hosts


def _audit_view(result: audit.AuditReport) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "violations": result.violations,
        "findings": sorted((f.invariant, f.check, f.severity) for f in result.findings),
        "cells": {
            c.cell_id: {
                "audited": c.audited,
                "violations": c.violations,
                "warnings": c.warnings,
                "checked": c.checked,
                "findings": sorted(
                    (f.invariant, f.check, f.severity, f.item or "") for f in c.findings
                ),
            }
            for c in result.cells
        },
    }


def _report_view(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "cells": data["cells"],
        "costs": data["costs"],
        "comparisons": {k: v for k, v in data.items() if k not in _REPORT_LOCAL_KEYS},
    }


#: What legitimately differs between two runs directories holding the same cells: where
#: they are, when the report was made, the gateway spend (one gateway directory, or one
#: per host), and what a merged view's _merged.json says.
_REPORT_LOCAL_KEYS = frozenset(
    {
        "cells",
        "costs",
        "runs_dir",
        "generated_at",
        "spend",
        "spend_union",
        "audit",
        "merged_view",
    }
)


def test_the_readers_give_the_same_answers_over_a_merged_view(e2e_matrix, tmp_path):
    hosts = _split_by_benchmark(e2e_matrix, tmp_path / "hosts")
    into = tmp_path / "view"
    assert (
        S.main(
            [
                "--source",
                f"cloud={hosts['cloud']}",
                "--source",
                f"local={hosts['local']}",
                "--into",
                str(into),
            ]
        )
        == S.EXIT_OK
    )
    assert {p.name for p in report.discover_cells(into)} == set(e2e_matrix.outcomes)

    def audited(runs: Path) -> audit.AuditReport:
        return audit.audit_runs(
            runs,
            env_factory=e2e_matrix.env_factory,
            split_mode=None,
            expected_upstream=TEST_UPSTREAM,
        )

    merged, unsplit = audited(into), audited(e2e_matrix.runs)
    assert merged.passed, merged.to_markdown()
    assert _audit_view(merged) == _audit_view(unsplit)
    assert all(
        str(Path("/nonexistent-cloud")) not in json.dumps(c.info, default=str)
        for c in merged.cells
    )
    merged_report = report.build_report(into, split_mode=None)
    unsplit_report = report.build_report(e2e_matrix.runs, split_mode=None)
    assert _report_view(merged_report) == _report_view(unsplit_report)
    assert {s["gateway_dir"] for s in merged_report["spend"]} <= {
        str(into / "_hosts" / "cloud"),
        str(into / "_hosts" / "local"),
    }

    def summary(runs: Path) -> list[dict[str, Any]]:
        return summarize_learning.summarize(runs)["cells"]

    assert summary(into) == summary(e2e_matrix.runs)
    # The readers never write into a host's mirror.
    assert not (into / "_hosts" / "cloud" / "_audit").exists()


def test_the_readers_give_the_same_answers_over_a_copied_runs_directory(
    e2e_matrix, tmp_path
):
    """The same without the tool: a whole runs directory copied to another host, its
    records' absolute paths naming the tree it came from (gone here)."""
    runs = tmp_path / "elsewhere" / "runs"
    shutil.copytree(e2e_matrix.runs, runs, symlinks=True)
    gone = Path("/nonexistent-origin/checkout/runs")
    for cell_id in e2e_matrix.outcomes:
        paths = CellPaths(runs / cell_id)
        manifest = read_json(paths.manifest)
        manifest["blob_dir"] = str(gone / "_blobs")
        write_json_atomic(paths.manifest, manifest)
        for number in paths.attempt_numbers():
            state = read_json(paths.attempt(number).state)
            state["ledger"] = str(gone / state["run_id"] / "gateway.jsonl")
            write_json_atomic(paths.attempt(number).state, state)

    def audited(where: Path) -> audit.AuditReport:
        return audit.audit_runs(
            where,
            env_factory=e2e_matrix.env_factory,
            split_mode=None,
            expected_upstream=TEST_UPSTREAM,
        )

    copied = audited(runs)
    assert copied.passed, copied.to_markdown()
    assert _audit_view(copied) == _audit_view(audited(e2e_matrix.runs))
    assert _report_view(report.build_report(runs, split_mode=None)) == _report_view(
        report.build_report(e2e_matrix.runs, split_mode=None)
    )
    assert (
        summarize_learning.summarize(runs)["cells"]
        == summarize_learning.summarize(e2e_matrix.runs)["cells"]
    )
