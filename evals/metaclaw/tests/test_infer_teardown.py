"""H3 — Per-test work-copy cleanup + gateway process-group reaping.

``infer_cmd`` creates ``data/work/openclaw_state_<uuid>/`` per test and never
removed it → 323 accumulated dirs from a long run. It also started the gateway
without its own session/process-group, so detached children (reparented to
init) orphaned. These tests pin the teardown helper:

  * the test's work-copy dir is ``shutil.rmtree``'d on teardown,
  * the gateway is started with ``start_new_session=True`` and the WHOLE process
    group is reaped (``os.killpg``) so no child orphans,
  * teardown is robust: a missing dir / already-dead process does not raise.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_vendored_src():
    bench = str(REPO_ROOT / "evals" / "metaclaw" / "vendor" / "benchmark")
    cached = (
        importlib.import_module("src.infer.infer_cmd")
        if "src.infer.infer_cmd" in sys.modules
        else None
    )
    if cached is not None and getattr(cached, "__file__", "").startswith(bench):
        return cached
    for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    sys.path.insert(0, bench)
    return importlib.import_module("src.infer.infer_cmd")


@pytest.fixture
def vendored_src_sandbox():
    saved_modules = {
        n: m for n, m in sys.modules.items() if n == "src" or n.startswith("src.")
    }
    saved_path = list(sys.path)
    try:
        yield
    finally:
        for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


@pytest.fixture
def infer_cmd(vendored_src_sandbox):
    return _import_vendored_src()


# --------------------------------------------------------------------------- #
# H3a — work-copy cleanup                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_teardown_rmtrees_work_copy(infer_cmd, tmp_path):
    work_copy = tmp_path / "work" / "openclaw_state_abc123"
    (work_copy / "agents").mkdir(parents=True)
    (work_copy / "openclaw.json").write_text("{}", encoding="utf-8")
    assert work_copy.exists()

    proc = _DummyProc(returncode=0)  # already exited cleanly
    await infer_cmd._teardown_gateway(proc, work_copy, gateway_port=12345)

    assert not work_copy.exists(), "work copy must be removed on teardown"


@pytest.mark.asyncio
async def test_teardown_missing_work_copy_does_not_raise(infer_cmd, tmp_path):
    missing = tmp_path / "work" / "gone"
    proc = _DummyProc(returncode=0)
    # Must not raise even though the dir was never created.
    await infer_cmd._teardown_gateway(proc, missing, gateway_port=1)


# --------------------------------------------------------------------------- #
# H3b — gateway started with its own session (so the group can be reaped)      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gateway_started_with_new_session(infer_cmd, tmp_path, monkeypatch):
    """``_start_work_gateway`` must pass ``start_new_session=True`` so the
    gateway and any detached children form a reapable process group."""
    state_dir = tmp_path / "openclaw_state_x"
    state_dir.mkdir()
    (state_dir / "openclaw.json").write_text("{}", encoding="utf-8")

    captured: dict = {}

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _DummyProc(returncode=None)

    monkeypatch.setattr(
        infer_cmd.asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    await infer_cmd._start_work_gateway(state_dir, 23456)
    assert captured["kwargs"].get("start_new_session") is True


# --------------------------------------------------------------------------- #
# H3c — the whole process group is reaped on teardown                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_teardown_reaps_process_group(infer_cmd, tmp_path, monkeypatch):
    """A still-running gateway → teardown terminates it AND killpg's its group."""
    killed_pgids: list[int] = []

    def _fake_getpgid(pid):
        return pid  # 1:1 mapping for the test

    def _fake_killpg(pgid, sig):
        killed_pgids.append(pgid)

    monkeypatch.setattr(infer_cmd.os, "getpgid", _fake_getpgid)
    monkeypatch.setattr(infer_cmd.os, "killpg", _fake_killpg)

    work_copy = tmp_path / "work" / "openclaw_state_live"
    work_copy.mkdir(parents=True)

    proc = _DummyProc(returncode=None, pid=4242)  # still running
    await infer_cmd._teardown_gateway(proc, work_copy, gateway_port=999)

    assert proc.terminated or proc.killed, "gateway proc must be stopped"
    assert 4242 in killed_pgids, "process group must be reaped via killpg"
    assert not work_copy.exists()


@pytest.mark.asyncio
async def test_teardown_reaps_group_even_when_leader_already_dead(infer_cmd, tmp_path, monkeypatch):
    """codex P1: if the gateway leader already exited, os.getpgid(pid) raises
    ProcessLookupError. We must still reap the group using pid-as-pgid (the
    process was started with start_new_session=True, so pgid == pid), so detached
    children don't survive."""
    killed_pgids: list[int] = []

    def _getpgid_dead(pid):
        raise ProcessLookupError()  # leader already reaped

    def _fake_killpg(pgid, sig):
        killed_pgids.append(pgid)

    monkeypatch.setattr(infer_cmd.os, "getpgid", _getpgid_dead)
    monkeypatch.setattr(infer_cmd.os, "killpg", _fake_killpg)

    work_copy = tmp_path / "work" / "openclaw_state_deadleader"
    work_copy.mkdir(parents=True)
    proc = _DummyProc(returncode=0, pid=5151)  # already exited
    await infer_cmd._teardown_gateway(proc, work_copy, gateway_port=1)

    # Fallback: reap pid-as-pgid since the session leader's pgid == its pid.
    assert 5151 in killed_pgids, "must fall back to pid-as-pgid when leader is dead"
    assert not work_copy.exists()


@pytest.mark.asyncio
async def test_teardown_killpg_processlookup_is_swallowed(infer_cmd, tmp_path, monkeypatch):
    """If the group is already gone, killpg raising ProcessLookupError must not
    propagate (the teardown is best-effort)."""

    def _raise_lookup(pid):
        raise ProcessLookupError()

    def _killpg_raise_lookup(pgid, sig):
        # Fallback pid-as-pgid path also finds the group already gone.
        raise ProcessLookupError()

    monkeypatch.setattr(infer_cmd.os, "getpgid", _raise_lookup)
    monkeypatch.setattr(infer_cmd.os, "killpg", _killpg_raise_lookup)

    work_copy = tmp_path / "work" / "openclaw_state_z"
    work_copy.mkdir(parents=True)
    proc = _DummyProc(returncode=None, pid=7)
    # Both getpgid and the fallback killpg find the group gone — must not raise.
    await infer_cmd._teardown_gateway(proc, work_copy, gateway_port=1)
    assert not work_copy.exists()


# --------------------------------------------------------------------------- #
# Dummy process                                                               #
# --------------------------------------------------------------------------- #


class _DummyProc:
    def __init__(self, returncode=None, pid=1234):
        self.returncode = returncode
        self.pid = pid
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True
        # Simulate the process exiting after terminate().
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode
