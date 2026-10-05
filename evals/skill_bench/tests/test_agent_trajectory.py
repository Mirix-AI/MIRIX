"""Trajectories, method-visible views, opaque ids and the method interface."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evals.skill_bench import protocol
from evals.skill_bench.agent import trajectory as tj
from evals.skill_bench.agent.trajectory import (
    FORBIDDEN_LABEL_KEYS,
    StepRecord,
    Trajectory,
    externalize_blobs,
    find_label_leaks,
    internalize_blobs,
    opaque_task_uid,
    read_trajectory,
    visible_message,
    write_trajectory,
)
from evals.skill_bench.envs.base import EpisodeResult, Score, Task, new_cell_salt
from evals.skill_bench.methods.base import (
    CellContext,
    Injection,
    MemoryMethod,
    MemoryMethodBase,
    TaskView,
    TrajectoryView,
)
from evals.skill_bench.methods.none import NoMemory

# --- opaque task ids (H5) ----------------------------------------------------------

RAW_IDS = [
    "train:0000",
    "test:0133",
    "val:0017",
    "fixed_1794",  # WebShop goal id
    "id with spaces / slashes: and ünïcode",
    "",
]


def test_opaque_uid_passes_mirix_session_id_validation():
    from mirix.schemas.message import (
        SESSION_ID_MAX_LEN,
        MessageCreate,
        _validate_session_id,
    )

    salt = new_cell_salt()
    for raw in RAW_IDS:
        uid = opaque_task_uid(salt, raw)
        assert re.fullmatch(r"t[0-9a-f]{20}", uid)
        assert len(uid) == 1 + protocol.OPAQUE_UID_HEX_CHARS <= SESSION_ID_MAX_LEN
        assert _validate_session_id(uid) == uid  # AddMemoryRequest uses the same rule
        assert MessageCreate(role="user", content="x", session_id=uid).session_id == uid  # type: ignore[arg-type]
    # The raw ids themselves are rejected by MIRIX (the reason for H5).
    with pytest.raises(ValueError):
        _validate_session_id("train:0000")


def test_opaque_uid_is_stable_per_salt_and_hides_the_raw_id():
    salt_a, salt_b = b"a" * 32, b"b" * 32
    uids_a = [opaque_task_uid(salt_a, raw) for raw in RAW_IDS]
    assert uids_a == [opaque_task_uid(salt_a, raw) for raw in RAW_IDS]  # stable
    assert len(set(uids_a)) == len(RAW_IDS)  # distinct items, distinct uids
    uids_b = [opaque_task_uid(salt_b, raw) for raw in RAW_IDS]
    assert not set(uids_a) & set(uids_b)  # another cell cannot link items
    # Golden value: HMAC-SHA256(salt, raw id), first 20 hex chars, "t" prefix.
    import hashlib
    import hmac

    digest = hmac.new(salt_a, b"train:0000", hashlib.sha256).hexdigest()
    assert opaque_task_uid(salt_a, "train:0000") == "t" + digest[:20]
    # Manifest order does not survive: sorting uids does not recover the raw order.
    ordered = [opaque_task_uid(salt_a, f"train:{i:04d}") for i in range(39)]
    assert ordered != sorted(ordered)
    task = Task("alfworld", "train", "train:0000")
    assert task.uid(salt_a) == opaque_task_uid(salt_a, "train:0000")
    with pytest.raises(ValueError):
        opaque_task_uid(b"", "train:0000")
    assert tj.opaque_task_uid is opaque_task_uid  # one implementation (envs/base.py)


# --- method-visible messages (Q20) ------------------------------------------------------


def test_visible_message_allowlists_keys_per_role():
    assistant = {
        "role": "assistant",
        "content": None,
        "reasoning": "private chain of thought",
        "refusal": None,
        "annotations": [{"x": 1}],
    }
    assert visible_message(assistant) == {"role": "assistant", "content": ""}
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Q?", "cache_control": {"type": "ephemeral"}},
        ],
        "name": "someone",
    }
    assert visible_message(user) == {
        "role": "user",
        "content": [{"type": "text", "text": "Q?"}],
    }
    for bad in (
        {"role": "system", "content": "template text"},
        {"role": "developer", "content": "x"},
        {"content": "no role"},
        {"role": "user", "content": [{"type": "input_audio", "data": "..."}]},
        {"role": "user", "content": 42},
    ):
        with pytest.raises(ValueError):
            visible_message(bad)


def test_visible_message_is_a_deep_copy():
    original = {
        "role": "user",
        "content": [{"type": "text", "text": "obs"}],
    }
    copy_ = visible_message(original)
    copy_["content"][0]["text"] = "changed"
    assert original["content"][0]["text"] == "obs"


# --- label scan (H10.1) ------------------------------------------------------------------


def test_find_label_leaks_checks_keys_everywhere_and_strings_in_harness_text():
    view = TrajectoryView(
        task_uid="t" + "0" * 20,
        instruction="What is the total?",
        first_messages=({"role": "user", "content": "What is the total?"},),
        messages=(
            {"role": "user", "content": "What is the total?"},
            {"role": "assistant", "content": "<answer>GOLD 42</answer>"},
        ),
    )
    # Model output may contain the answer; harness text may not.
    assert find_label_leaks(view, forbidden_strings=["gold 42"]) == []
    leaky = TrajectoryView(
        view.task_uid,
        "What is the total? (answer: GOLD   42)",
        view.first_messages,
        view.messages,
    )
    assert find_label_leaks(leaky, forbidden_strings=["gold 42"]) == [
        "$.instruction: contains 'gold 42'"
    ]
    nested = {"messages": [{"role": "user", "content": "x", "meta": {"won": True}}]}
    assert find_label_leaks(nested) == ["$.messages[0].meta: forbidden key 'won'"]
    assert {"won", "task_type", "facts", "expert_plan", "answers", "source_docs"} <= (
        FORBIDDEN_LABEL_KEYS
    )


# --- the method interface (H6) ------------------------------------------------------------


def test_injection_validation():
    injection = Injection("memory", ["a", "b"])  # type: ignore[arg-type]
    assert injection.item_ids == ("a", "b") and injection.tokens is None
    assert Injection().empty and Injection(" \n\t").empty and not injection.empty
    with pytest.raises(TypeError):
        Injection(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Injection("x", (1,))  # type: ignore[arg-type]


def cell(**overrides) -> CellContext:
    values = dict(
        cell_id="alfworld-luna-none",
        attempt=1,
        benchmark="alfworld",
        method="none",
        model=protocol.MODELS["luna"],
        gateway_base_url="http://127.0.0.1:18540/",
        run_id="alfworld-luna-none-a1",
        gateway_token="local-token",
        work_dir=Path("/tmp/x"),
        context_length=1_050_000,
        max_output_tokens=128_000,
    )
    values.update(overrides)
    return CellContext(**values)  # type: ignore[arg-type]


def test_cell_context_role_urls_and_validation():
    context = cell()
    assert (
        context.role_url("learner")
        == "http://127.0.0.1:18540/runs/alfworld-luna-none-a1/learner/v1"
    )
    assert context.role_url("embedding").endswith("/embedding/v1")
    with pytest.raises(ValueError):
        context.role_url("actor")  # the actor role belongs to the framework
    assert "local-token" not in repr(context)
    for bad in (
        {"benchmark": "webarena"},
        {"model": "openai/gpt-4o"},
        {"attempt": 0},
    ):
        with pytest.raises(ValueError):
            cell(**bad)


@pytest.mark.asyncio
async def test_no_memory_method():
    method = NoMemory()
    assert isinstance(method, MemoryMethod)
    assert isinstance(MemoryMethodBase(), MemoryMethod)
    assert method.name == "none" and method.signal_level == "none"
    await method.setup(cell())
    view = TaskView("t" + "1" * 20, "do it", ({"role": "user", "content": "obs"},))
    injection = await method.before_task(view)
    assert injection == Injection() and injection.empty
    await method.on_message(view.task_uid, {"role": "user", "content": "obs"})
    await method.after_task(TrajectoryView(view.task_uid, "do it", (), ()))
    await method.wait_until_idle(1.0)
    await method.finalize()
    assert await method.snapshot() == {"method": "none", "memory": None}
    await method.close()


def test_method_interface_is_import_light():
    code = (
        "import sys\n"
        "import evals.skill_bench.methods.base, evals.skill_bench.methods.none\n"
        "heavy = sorted(m for m in ('httpx', 'fastapi', 'mirix', 'openai', 'tiktoken',"
        " 'pydantic', 'docker', 'alfworld', 'PIL') if m in sys.modules)\n"
        "print(','.join(heavy))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=protocol.REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == ""


# --- the private record -----------------------------------------------------------------


def sample_trajectory() -> Trajectory:
    image = "data:image/png;base64," + "A" * 6000
    first = (
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": image}}],
        },
    )
    trajectory = Trajectory(
        benchmark="alfworld",
        split="test",
        raw_id="12345",
        task_uid="t" + "a" * 20,
        phase="test",
        method="none",
        model=protocol.MODELS["luna"],
        run_id="cell-a1",
        instruction="What?",
        first_messages=first,
        messages=[*first, {"role": "assistant", "content": "<answer>x</answer>"}],
        injection=Injection("mem", ("m1",), tokens=1),
    )
    trajectory.steps.append(
        StepRecord(
            0,
            0,
            {"role": "assistant", "content": "<answer>x</answer>"},
            "reasoning",
            (),
            True,
            {"answer": "x"},
            {},
        )
    )
    trajectory.result = EpisodeResult(
        "alfworld", "test", "12345", trajectory.task_uid, Score(0.0)
    )
    return trajectory


def test_views_are_fresh_deep_copies():
    trajectory = sample_trajectory()
    view = trajectory.view()
    assert view.messages == tuple(trajectory.messages)
    view.messages[1]["content"] = "tampered"
    assert trajectory.messages[1]["content"] == "<answer>x</answer>"
    task_view = trajectory.task_view()
    assert task_view.first_messages == trajectory.first_messages
    assert task_view.first_messages[0] is not trajectory.first_messages[0]
    assert set(vars(view)) == {"task_uid", "instruction", "first_messages", "messages"}
    # The task view also carries the benchmark's static agent context
    # (empty here: none was set).
    assert set(vars(task_view)) == {
        "task_uid",
        "instruction",
        "first_messages",
        "agent_context",
    }
    assert task_view.agent_context == trajectory.agent_context == ""
    trajectory.agent_context = "the actor's prompt: {current_observation}"
    assert trajectory.task_view().agent_context == trajectory.agent_context
    assert trajectory.to_dict()["agent_context"] == trajectory.agent_context


def test_trajectory_serialises_with_content_addressed_blobs(tmp_path):
    trajectory = sample_trajectory()
    data = trajectory.to_dict()
    json.dumps(data)  # plain JSON
    assert data["injected_tokens"] == 1 and data["result"]["score"]["primary"] == 0.0
    blobs = tmp_path / "blobs"
    write_trajectory(tmp_path / "a.json", trajectory, blob_dir=blobs)
    write_trajectory(tmp_path / "b.json", trajectory, blob_dir=blobs)
    assert len(list(blobs.iterdir())) == 1  # the image is stored once
    stored = json.loads((tmp_path / "a.json").read_text())
    assert "AAAA" not in json.dumps(stored)
    assert read_trajectory(tmp_path / "a.json", blob_dir=blobs) == data
    # Small strings and non-data URLs stay inline; tampering is detected.
    assert externalize_blobs({"x": "data:short"}, blobs) == {"x": "data:short"}
    blob = next(blobs.iterdir())
    blob.write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        internalize_blobs(stored, blobs)


# --- durable blobs -----------------------------------------------------------------------


def _record_fsyncs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Every ``os.fsync`` of the trajectory module, as the (device, inode) it synced."""
    import os

    synced: list[tuple[int, int]] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        stat = os.fstat(fd)
        synced.append((stat.st_dev, stat.st_ino))
        real(fd)

    monkeypatch.setattr(tj.os, "fsync", fsync)
    return synced


def _inode(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino)


def test_a_blob_is_on_disk_before_the_record_that_references_it(tmp_path, monkeypatch):
    """The image blob is written to a temporary file, fsync'd, renamed and
    its directory fsync'd (with every directory the write created) before the
    trajectory JSON is fsync'd and renamed, whose directory is fsync'd last: a played
    checkpoint taken after write_trajectory returns never references a blob that a power
    loss leaves zero bytes long (ext4's delayed allocation)."""
    synced = _record_fsyncs(monkeypatch)
    runs = tmp_path / "runs"
    runs.mkdir()
    blobs = runs / "_blobs"  # created by the write
    path = runs / "cell" / "attempt-01" / "trajectories" / "learning" / "t1.json"
    write_trajectory(path, sample_trajectory(), blob_dir=blobs)
    (blob,) = blobs.iterdir()
    order = {key: synced.index(key) for key in synced}
    for needed in (blob, blobs, runs, path, path.parent, runs / "cell"):
        assert _inode(needed) in order, f"{needed} was never fsync'd"
    # The blob and its directory entry first, then the record, then its entry.
    assert order[_inode(blob)] < order[_inode(blobs)] < order[_inode(path)]
    assert order[_inode(path)] < synced.index(_inode(path.parent), order[_inode(path)])
    assert not [p for p in blobs.iterdir() if p.name.endswith(".tmp")]
    # A blob already there (another cell's page) is fsync'd too, never trusted blind.
    synced.clear()
    write_trajectory(tmp_path / "again.json", sample_trajectory(), blob_dir=blobs)
    assert _inode(blob) in synced and _inode(blobs) in synced


@pytest.mark.parametrize("damage", ["emptied", "torn", "corrupt"])
def test_a_blob_a_crash_damaged_is_written_again(tmp_path, damage):
    """A blob left empty (a crash before its data reached the disk), torn
    or corrupt by an earlier writer is not taken for the image because a file of its name
    exists: the next write of a record that references it puts the image back."""
    trajectory = sample_trajectory()
    blobs = tmp_path / "blobs"
    write_trajectory(tmp_path / "a.json", trajectory, blob_dir=blobs)
    (blob,) = blobs.iterdir()
    good = blob.read_bytes()
    blob.write_bytes(
        {"emptied": b"", "torn": good[:100], "corrupt": b"X" * len(good)}[damage]
    )
    with pytest.raises((ValueError, OSError)):
        read_trajectory(tmp_path / "a.json", blob_dir=blobs)
    write_trajectory(tmp_path / "b.json", trajectory, blob_dir=blobs)
    assert blob.read_bytes() == good
    assert read_trajectory(tmp_path / "a.json", blob_dir=blobs) == trajectory.to_dict()


def test_a_failed_blob_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    """A blob whose fsync fails (a full disk) raises, and its temporary file is gone."""
    import errno

    def full(_fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(tj.os, "fsync", full)
    blobs = tmp_path / "blobs"
    with pytest.raises(OSError, match="No space"):
        write_trajectory(tmp_path / "a.json", sample_trajectory(), blob_dir=blobs)
    assert list(blobs.iterdir()) == []


def test_unreadable_blobs_names_the_missing_and_the_corrupt_ones(tmp_path):
    """What ``run.py`` re-creates before it rebuilds a played item's view: every
    referenced blob that is missing or does not hash to its name, once."""
    blobs = tmp_path / "blobs"
    images = ["data:image/png;base64," + c * 5000 for c in "ABC"]
    stored = externalize_blobs(
        {"first": [images[0], images[1]], "calls": [{"x": images[0]}, images[2]]},
        blobs,
    )
    digests = [tj.sha256_hex(i) for i in images]
    assert tj.unreadable_blobs(stored, blobs) == []
    (blobs / digests[1]).unlink()
    (blobs / digests[2]).write_bytes(b"")
    assert tj.unreadable_blobs(stored, blobs) == sorted([digests[1], digests[2]])
    assert tj.unreadable_blobs({"no": "blobs"}, blobs) == []
