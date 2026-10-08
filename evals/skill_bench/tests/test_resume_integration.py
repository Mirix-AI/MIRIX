"""Kill a learning cell for real and resume it.

Every test runs the cell in a process of its own (``tests/resume_worker.py``: run.py's
``run_cell``, the CLI's signal handling, on a TinyEnv cell) against a real loopback
gateway in the test process over the fake upstream. The upstream holds one chosen call
in flight; the test kills the process there (SIGKILL: nothing of the harness runs; SIGTERM:
the CLI's handler), runs it again and checks the invariant: every learning
position once in the live records, the frozen memory an uninterrupted cell with the same
replies freezes, one discarded segment whose calls the ledger markers recompute, a
verified restore and a complete test phase. Nothing reaches a provider.

* StoringMethod and the file methods (AWM, ReasoningBank) run without
  Postgres;
* MIRIX (``-m integration``) is this checkout's real server on the compose Postgres, with
  cell databases of the test's own (``sb_alfworld_luna_mirix_rt<hex>...``, dropped
  afterwards; never a smoke or real cell's); the upstream plays its model
  (``tests/mirix_model.py``). The dead harness's MIRIX server survives the SIGKILL (its
  own process group) and holds the cell database until the resume stops it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from evals.skill_bench import protocol, run
from evals.skill_bench.cell_records import (
    ATTEMPT_LEARNING,
    ATTEMPT_LEARNING_INTERRUPTED,
    CellPaths,
    read_json,
    read_ledger_lines,
)
from evals.skill_bench.infra.mirix_server import (
    EXPORT_FILE,
    PostgresAdmin,
    drop_cell_database,
)
from evals.skill_bench.infra.processes import cleanup_owned
from evals.skill_bench.methods import awm as A
from evals.skill_bench.run import EXIT_CODES, STATUS_INTERRUPTED
from evals.skill_bench.tests.fakes import FakeUpstream, Reply, serve_gateway
from evals.skill_bench.tests.mirix_model import MirixModel, mirix_request_kind
from evals.skill_bench.tests.rollout_support import text_reply
from evals.skill_bench.tests.run_support import (
    QUESTION_RE,
    order,
    question,
    read_rows,
    resumed_cleanly,
    tiny_responder,
)
from evals.skill_bench.tests.s1_support import call_kind, items_reply, judged

REPO = Path(__file__).resolve().parents[3]
SALT = "a fixed test salt, 32 bytes long!"
#: The learning order of the worker's TinyEnv (6 train items): even raw ids are answered
#: right, so the items at positions 2, 3 and 4 succeed.
TRAIN = order()
WORKER_TIMEOUT_S = 300.0
WORKFLOW = (
    "## Workflow: answer a question about a page\n"
    "1. Find the field the question names.\n"
    "2. Answer with its value, in the requested format."
)
Predicate = Callable[[str, dict[str, Any]], bool]


# ---------------------------------------------------------------------------
# The upstream: deterministic replies, one call held in flight
# ---------------------------------------------------------------------------


def _request_text(body: Mapping[str, Any]) -> str:
    texts = []
    for message in body.get("messages") or ():
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [str(p.get("text") or "") for p in content if isinstance(p, dict)]
    return "\n".join(texts)


def learners(endpoint: str, body: dict[str, Any]) -> Reply | None:
    """Replies that depend on the request only (so a replayed item gets the reply the
    killed one got): the TinyEnv actor, J (a wrong answer is a failure), ReasoningBank's
    and AWM's learners; embeddings are the fake's hash vectors."""
    if endpoint != "chat/completions":
        return None
    actor = tiny_responder(endpoint, body)
    if actor is not None:
        return actor
    kind = call_kind(body)
    if kind == "judge":
        verdict = "failure" if "no idea" in _request_text(body) else "success"
        return text_reply(judged(verdict), model=body["model"])
    if kind == "learner":
        if len(body["messages"]) == 1:  # AWM's induction
            return text_reply(WORKFLOW, model=body["model"])
        return text_reply(
            items_reply("check the source first", "keep the answer format"),
            model=body["model"],
        )
    return None


class HeldUpstream:
    """The fake upstream's responder: ``inner``'s replies, except that the first call the
    predicate of :meth:`hold` matches waits upstream until :attr:`release` is set."""

    def __init__(self, inner: Callable[[str, dict[str, Any]], Reply | None]) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self._predicate: Predicate | None = None
        self.reached = threading.Event()
        self.release = threading.Event()

    def hold(self, predicate: Predicate) -> None:
        """Hold the next call ``predicate`` matches (fresh events: a helper still
        waiting to release an earlier call cannot release this one)."""
        with self._lock:
            self._predicate = predicate
            self.reached = threading.Event()
            self.release = threading.Event()

    def __call__(self, endpoint: str, body: dict[str, Any]) -> Reply | None:
        with self._lock:
            held = self._predicate is not None and self._predicate(endpoint, body)
            if held:
                self._predicate = None
            reached, release = self.reached, self.release
        reply = self.inner(endpoint, body)
        if not held:
            return reply
        reached.set()
        return dataclasses.replace(reply or Reply(), hold=release)


def actor_call(text: str) -> Predicate:
    def match(endpoint: str, body: dict[str, Any]) -> bool:
        messages = body.get("messages") or []
        last = messages[-1].get("content") if messages else None
        return (
            endpoint == "chat/completions"
            and isinstance(last, str)
            and last == text
            and QUESTION_RE.match(last) is not None
        )

    return match


def judge_call(text: str) -> Predicate:
    return lambda endpoint, body: (
        endpoint == "chat/completions"
        and call_kind(body) == "judge"
        and text in _request_text(body)
    )


def induction_call(text: str) -> Predicate:
    return lambda endpoint, body: (
        endpoint == "chat/completions"
        and len(body.get("messages") or ()) == 1
        and _request_text(body).startswith(A.instruction_text())
        and text in _request_text(body)
    )


def curator_call(endpoint: str, body: dict[str, Any]) -> bool:
    return endpoint == "chat/completions" and mirix_request_kind(body) == "curator"


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Bench:
    """A gateway over the held upstream, and the cells the tests run against it."""

    root: Path
    url: str
    upstream: HeldUpstream
    tag: str = ""
    procs: list[subprocess.Popen[bytes]] = dataclasses.field(default_factory=list)

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def start(self, cell: Path, method: str, *extra: str) -> subprocess.Popen[bytes]:
        """One invocation of the worker for ``cell`` (stdout/stderr in files beside it)."""
        n = len(list(cell.parent.glob(f"{cell.name}.*.out")))
        stdout = open(cell.parent / f"{cell.name}.{n}.out", "wb")  # noqa: SIM115
        stderr = open(cell.parent / f"{cell.name}.{n}.err", "wb")  # noqa: SIM115
        command = [
            sys.executable,
            "-m",
            "evals.skill_bench.tests.resume_worker",
            "--method",
            method,
            "--out",
            str(cell),
            "--gateway-url",
            self.url,
            "--gateway-runs-dir",
            str(self.runs),
            "--salt",
            SALT,
            "--code-file",
            str(self.root / "code.json"),
            *extra,
        ]
        if method == "mirix":
            command += ["--db-tag", f"{self.tag}{cell.name.replace('-', '')}"[:20]]
        try:
            proc = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr)
        finally:
            stdout.close()
            stderr.close()
        self.procs.append(proc)
        return proc

    def stop(self) -> None:
        """Kill any worker a failed test left running, and release the held call."""
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
        self.upstream.release.set()

    def finish(
        self, proc: subprocess.Popen[bytes], cell: Path
    ) -> tuple[int, dict[str, Any]]:
        """Wait for the worker; its exit code and outcome line."""
        try:
            code = proc.wait(timeout=WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise AssertionError(f"the worker did not finish:\n{worker_output(cell)}")
        return code, outcome_line(cell)

    def run(self, cell: Path, method: str, *extra: str) -> dict[str, Any]:
        """An invocation to its end, which must complete the cell."""
        code, outcome = self.finish(self.start(cell, method, *extra), cell)
        assert (code, outcome.get("status")) == (0, "complete"), worker_output(cell)
        return outcome


def worker_output(cell: Path) -> str:
    out = []
    for path in sorted(cell.parent.glob(f"{cell.name}.*.*")):
        text = path.read_text(errors="replace")
        out.append(f"--- {path.name}\n{text[-4000:]}")
    return "\n".join(out)


def outcome_line(cell: Path) -> dict[str, Any]:
    """The last invocation's outcome (the last JSON line of its stdout)."""
    outs = sorted(
        cell.parent.glob(f"{cell.name}.*.out"), key=lambda p: int(p.name.split(".")[-2])
    )
    for line in reversed(outs[-1].read_text(errors="replace").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def latest(cell: Path) -> str | None:
    try:
        return read_json(CellPaths(cell).attempt(1).latest_checkpoint)["id"]
    except (OSError, TypeError, KeyError, ValueError):
        return None


def kill_when(
    proc: subprocess.Popen[bytes],
    cell: Path,
    ready: Callable[[], bool],
    sig: int = signal.SIGKILL,
) -> int:
    """Send ``sig`` once ``ready()``; the worker's exit code."""
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    while not ready():
        if proc.poll() is not None:
            raise AssertionError(
                f"the worker ended before the kill:\n{worker_output(cell)}"
            )
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError(
                f"the kill point was never reached:\n{worker_output(cell)}"
            )
        time.sleep(0.02)
    os.kill(proc.pid, sig)
    return proc.wait(timeout=WORKER_TIMEOUT_S)


def held_at(bench: Bench, cell: Path, checkpoint: str) -> Callable[[], bool]:
    return lambda: bench.upstream.reached.is_set() and latest(cell) == checkpoint


def release_after_resume(bench: Bench, ledger: Path) -> threading.Thread:
    """Let the held call finish only once the resume marker is in the ledger (a zombie
    call whose final line comes after the marker)."""
    release = bench.upstream.release

    def wait() -> None:
        deadline = time.monotonic() + WORKER_TIMEOUT_S
        while time.monotonic() < deadline and not release.is_set():
            if any(
                line.row.get("event") == "resume" for line in read_ledger_lines(ledger)
            ):
                break
            time.sleep(0.02)
        release.set()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return thread


def run_ledger(bench: Bench, cell: Path) -> Path:
    run_id = read_json(CellPaths(cell).attempt(1).state)["run_id"]
    return bench.runs / run_id / "gateway.jsonl"


def the_row(cell: Path) -> dict[str, Any]:
    (row,) = read_rows(CellPaths(cell).attempt(1).discarded_log)
    assert row["restore"]["files_equal"] is True, row["restore"]
    assert (
        row["restore"]["snapshot_sha256_actual"]
        == (row["restore"]["snapshot_sha256_expected"])
    )
    return row


def frozen(cell: Path) -> dict[str, Any]:
    return read_json(CellPaths(cell).attempt(1).frozen_snapshot)["snapshot"]


def _without_times(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _without_times(v)
            for k, v in value.items()
            if not (k.endswith("_at") or k in ("at", "ts", "frozen_snapshot_sha256"))
        }
    if isinstance(value, list):
        return [_without_times(v) for v in value]
    return value


def assert_zombie_after_the_marker(ledger: Path, row: Mapping[str, Any]) -> str:
    """The held call was in flight at the resume (``final_line`` False in the row) and
    the gateway wrote its final line after the resume marker; its id."""
    (zombie,) = [c for c in row["calls"] if not c["final_line"]]
    finals = [
        line
        for line in read_ledger_lines(ledger)
        if line.row.get("event") == "call"
        and line.row.get("call_id") == zombie["call_id"]
    ]
    deadline = time.monotonic() + 30
    while not finals and time.monotonic() < deadline:  # the gateway finishes it
        time.sleep(0.05)
        finals = [
            line
            for line in read_ledger_lines(ledger)
            if line.row.get("event") == "call"
            and line.row.get("call_id") == zombie["call_id"]
        ]
    assert finals and finals[0].offset > row["resume_marker_offset"]
    return str(zombie["call_id"])


def assert_calls_once(
    paths: CellPaths, reference: CellPaths, row: Mapping[str, Any]
) -> None:
    """The ledger side of "every item once": the cell's calls less the discarded segment's
    are the uninterrupted cell's, role by role (``costs.json``), and the discarded bucket
    is the segment's."""
    costs = read_json(paths.attempt(1).costs)
    expected = read_json(reference.attempt(1).costs)["by_phase_role"]
    discarded = Counter(c["role"] for c in row["calls"])
    assert costs["discarded"]["calls"] == len(row["calls"])
    assert costs["discarded"]["by_role"] == dict(discarded)
    live = {
        key: value["calls"]
        - (discarded[key.split("/", 1)[1]] if key.startswith("learning/") else 0)
        for key, value in costs["by_phase_role"].items()
    }
    assert live == {key: value["calls"] for key, value in expected.items()}


@pytest.fixture(autouse=True)
def _frozen_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This test's own copy of the frozen record, frozen to this tree: the workers (which
    inherit the environment) freeze automatically when the tree is not the frozen value
    and must never write the repository's record."""
    from evals.skill_bench.tools import freeze_protocol

    copy = tmp_path / "_frozen_record" / "protocol_frozen.json"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(protocol.FROZEN_PROTOCOL_JSON, copy)
    freeze_protocol.freeze(
        reason="the tests baseline",
        path=copy,
        git=lambda: {"commit": "c0ffee0", "dirty": False, "branch": "test"},
    )
    monkeypatch.setenv(protocol.FROZEN_RECORD_ENV, str(copy))
    return copy


# ---------------------------------------------------------------------------
# StoringMethod and the file methods (no Postgres)
# ---------------------------------------------------------------------------


@pytest.fixture
def bench(tmp_path: Path) -> Iterator[Bench]:
    upstream = HeldUpstream(learners)
    with serve_gateway(
        tmp_path,
        FakeUpstream(responder=upstream),
        backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S,
    ) as (_gw, url):
        bench = Bench(tmp_path, url, upstream)
        try:
            yield bench
        finally:
            bench.stop()


FILE_CASES = {
    # name: (method, worker options, what is held, LATEST at the kill, stage, position)
    "awm-rollout": (
        "awm",
        (),
        actor_call(question(TRAIN[3])),
        "c0002-done",
        "rollout",
        3,
    ),
    "awm-induction": (
        "awm",
        (),
        induction_call(question(TRAIN[3])),
        "c0003-played",
        "after_task",
        3,
    ),
    "reasoningbank-judge": (
        "reasoningbank",
        (),
        judge_call(question(TRAIN[3])),
        "c0003-played",
        "after_task",
        3,
    ),
}


@pytest.mark.parametrize("case", list(FILE_CASES))
def test_a_file_method_killed_mid_item_resumes_to_the_uninterrupted_memory(bench, case):
    """SIGKILL in the middle of a learning item (the actor's call of a rollout, J's call,
    AWM's induction in ``after_task``), then the same command again: the cell resumes
    from the last checkpoint and freezes the memory an uninterrupted cell freezes."""
    method, extra, hold, checkpoint, stage, position = FILE_CASES[case]
    reference = bench.root / "reference"
    bench.run(reference, method, *extra)

    cell = bench.root / "killed"
    bench.upstream.hold(hold)
    killed = kill_when(
        bench.start(cell, method, *extra), cell, held_at(bench, cell, checkpoint)
    )
    assert killed == -signal.SIGKILL
    paths = CellPaths(cell)
    assert read_json(paths.attempt(1).state)["state"] == ATTEMPT_LEARNING
    ledger = run_ledger(bench, cell)
    release_after_resume(bench, ledger)
    bench.run(cell, method, *extra)

    row = the_row(cell)
    assert row["reason"] == "process_killed" and row["from_state"] == ATTEMPT_LEARNING
    assert row["checkpoint"]["id"] == checkpoint
    assert (row["in_flight"]["position"], row["in_flight"]["stage"]) == (
        position,
        stage,
    )
    assert_zombie_after_the_marker(ledger, row)
    assert_calls_once(paths, CellPaths(reference), row)
    resumed_cleanly(ledger, paths, storing=False)
    assert _without_times(frozen(cell)) == _without_times(frozen(reference))


def test_a_sigterm_in_the_learning_phase_leaves_the_cell_resumable(bench):
    """SIGTERM (the CLI's handler: the task is cancelled) while an actor call is in
    flight: the invocation ends ``interrupted`` (exit 130) with the attempt
    ``learning_interrupted``, and the next invocation resumes it from the last
    checkpoint. The abandoned call stays in flight upstream through the whole resumed
    run: its checkpoints do not wait for it (it is the discarded segment's)."""
    cell = bench.root / "terminated"
    bench.upstream.hold(actor_call(question(TRAIN[2])))
    proc = bench.start(cell, "storing")
    code = kill_when(proc, cell, held_at(bench, cell, "c0001-done"), signal.SIGTERM)
    assert code == EXIT_CODES[STATUS_INTERRUPTED] == 130, worker_output(cell)
    assert outcome_line(cell)["status"] == STATUS_INTERRUPTED
    paths = CellPaths(cell)
    state = read_json(paths.attempt(1).state)
    assert state["state"] == ATTEMPT_LEARNING_INTERRUPTED
    assert state["interruption"]["reason"] == "interrupted"
    assert state["learning"]["checkpoint"] == "c0001-done"
    ledger = run_ledger(bench, cell)
    started = time.monotonic()
    bench.run(cell, "storing")
    assert time.monotonic() - started < run.CHECKPOINT_QUIESCE_S
    assert not bench.upstream.release.is_set()  # still in flight
    row = the_row(cell)
    assert (row["reason"], row["from_state"]) == (
        "interrupted",
        ATTEMPT_LEARNING_INTERRUPTED,
    )
    assert row["checkpoint"]["id"] == "c0001-done"
    assert [(c["role"], c["final_line"]) for c in row["calls"]] == [("actor", False)]
    resumed_cleanly(ledger, paths)
    bench.upstream.release.set()
    assert_zombie_after_the_marker(ledger, row)


@pytest.mark.parametrize("point", ["partial", "renamed"])
def test_a_harness_killed_while_writing_a_checkpoint_resumes_from_the_one_before(
    bench, point
):
    """SIGKILL inside the capture of ``c0002-done``: with everything in its ``.partial``
    directory but its record, or with the directory renamed into place but ``LATEST``
    not naming it yet. ``LATEST`` is the commit point: the resume restores
    ``c0002-played`` (the one LATEST names), removes the unfinished checkpoint and
    redoes the item's ``after_task``."""
    cell = bench.root / "stalled"
    flag = bench.root / "stalled.flag"
    proc = bench.start(
        cell,
        "storing",
        "--stall-checkpoint",
        "c0002-done",
        "--stall-point",
        point,
        "--stall-flag",
        str(flag),
    )
    assert kill_when(proc, cell, flag.exists) == -signal.SIGKILL
    paths = CellPaths(cell)
    checkpoints = paths.attempt(1).checkpoints
    assert latest(cell) == "c0002-played"
    unfinished = "c0002-done.partial" if point == "partial" else "c0002-done"
    assert (checkpoints / unfinished).is_dir()
    bench.run(cell, "storing")
    row = the_row(cell)
    assert row["checkpoint"]["id"] == "c0002-played"
    assert row["in_flight"]["position"] == 2
    assert row["in_flight"]["stage"] == "after_task"
    assert not (checkpoints / "c0002-done.partial").exists()
    resumed_cleanly(run_ledger(bench, cell), paths)


def test_a_cell_killed_twice_resumes_twice(bench):
    """Two SIGKILLs in two invocations (the second in the resumed segment): two
    discarded segments, each with its own calls, and the memory of an uninterrupted
    run."""
    cell = bench.root / "twice"
    bench.upstream.hold(actor_call(question(TRAIN[1])))
    first = bench.start(cell, "storing")
    assert kill_when(first, cell, held_at(bench, cell, "c0000-done")) == -signal.SIGKILL
    bench.upstream.release.set()
    bench.upstream.hold(actor_call(question(TRAIN[4])))
    second = bench.start(cell, "storing")
    assert (
        kill_when(second, cell, held_at(bench, cell, "c0003-done")) == -signal.SIGKILL
    )
    bench.upstream.release.set()
    bench.run(cell, "storing")
    paths = CellPaths(cell)
    rows = read_rows(paths.attempt(1).discarded_log)
    assert [(r["segment"], r["checkpoint"]["id"]) for r in rows] == [
        (1, "c0000-done"),
        (2, "c0003-done"),
    ]
    assert [[c["role"] for c in r["calls"]] for r in rows] == [["actor"], ["actor"]]
    resumed_cleanly(run_ledger(bench, cell), paths)


# ---------------------------------------------------------------------------
# A real MIRIX (compose Postgres)
# ---------------------------------------------------------------------------


def _postgres_up() -> bool:
    admin = PostgresAdmin()
    try:
        with socket.create_connection((admin.host, admin.port), timeout=1):
            return True
    except OSError:
        return False


async def _database_names(prefix: str) -> list[str]:
    import asyncpg

    pg = PostgresAdmin()
    conn = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=pg.maintenance_db,
    )
    try:
        rows = await conn.fetch("SELECT datname FROM pg_database")
    finally:
        await conn.close()
    return sorted(r["datname"] for r in rows if r["datname"].startswith(prefix))


async def _rows_with_id(
    db_name: str, table: str, row_id: str | None, session_id: str | None = None
) -> int:
    """How many rows of ``table`` have that ``id`` (or that ``session_id``)."""
    import asyncpg

    pg = PostgresAdmin()
    conn = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=db_name,
    )
    try:
        if row_id is not None:
            query, value = f'SELECT count(*) FROM "{table}" WHERE id = $1', row_id
        else:
            query, value = (
                f'SELECT count(*) FROM "{table}" WHERE session_id = $1',
                session_id,
            )
        return int(await conn.fetchval(query, value))
    finally:
        await conn.close()


async def _table_counts(db_name: str) -> dict[str, int]:
    """The row count of every table of a cell database."""
    import asyncpg

    pg = PostgresAdmin()
    conn = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=db_name,
    )
    try:
        tables = [
            r["tablename"]
            for r in await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "ORDER BY tablename"
            )
        ]
        return {
            t: int(await conn.fetchval(f'SELECT count(*) FROM "{t}"')) for t in tables
        }
    finally:
        await conn.close()


#: MIRIX's random ids, and wall-clock times inside values (keys ending ``_at`` are
#: dropped): two cells with the same inputs differ in these only.
_MIRIX_IDS = re.compile(
    r"\b(?:(proc)_[A-Za-z0-9]+|(slrun|sexp|skver|agent)-[0-9a-f]{8}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


#: A curator batch's token estimates of prompts that embed those ids and times (a few
#: tokens apart between two cells; the counts of what the batch saw are compared).
_ID_BEARING_COUNTS = frozenset(
    {"context_peak_tokens", "history_tokens", "payload_tokens"}
)
#: The same inside one block, by the block's key: the batch's usage-record section
#: (``skill_usage``: its ``tokens`` count a text that names skill ids and
#: session ids, the cell's opaque task uids); its other keys (which skills, served
#: where, compact, left out) are compared. ``tokens`` is too common a key to drop
#: everywhere.
_ID_BEARING_NESTED_COUNTS: dict[str, frozenset[str]] = {
    "skill_usage": frozenset({"tokens"})
}
#: Snapshot lists the method sorts by MIRIX's random ids: compared as sorted lists.
_ID_ORDERED_LISTS = frozenset(
    {"learning_runs", "skills", "deleted_skills", "skills_without_vector"}
)


def normalized(value: Any) -> Any:
    """A MIRIX snapshot without its random ids (each replaced by its kind), its times and
    the token estimates of prompts that embed them."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_at") or key in ("at", "ts", "frozen_snapshot_sha256"):
                continue
            if key in _ID_BEARING_COUNTS:
                continue
            if key in _ID_BEARING_NESTED_COUNTS and isinstance(item, dict):
                item = {
                    k: v
                    for k, v in item.items()
                    if k not in _ID_BEARING_NESTED_COUNTS[key]
                }
            name = normalized(key)
            assert name not in out, f"two keys normalize to {name!r}"
            out[name] = normalized(item)
            if key in _ID_ORDERED_LISTS and isinstance(out[name], list):
                out[name] = sorted(
                    out[name], key=lambda v: json.dumps(v, sort_keys=True)
                )
        return out
    if isinstance(value, list):
        return [normalized(v) for v in value]
    if isinstance(value, str):
        if _TIMESTAMP.match(value):
            return "<time>"
        return _MIRIX_IDS.sub(lambda m: f"<{m.group(1) or m.group(2)}>", value)
    return value


def test_normalized_drops_the_id_bearing_token_counts_only():
    """The usage-record section's ``tokens`` (it counts a text naming skill and session
    ids) is dropped inside ``skill_usage`` only; its other keys, and a ``tokens`` key
    anywhere else, are compared."""
    batch = {
        "history_tokens": 120,
        "skill_usage": {
            "skill_ids": ["proc_abc123"],
            "served_ids": [],
            "tokens": 87,
            "compact": False,
            "left_out_ids": [],
            "skipped": None,
        },
        "usage": {"tokens": 5},
        "started_at": "2026-10-03T04:00:00Z",
    }
    assert normalized(batch) == {
        "skill_usage": {
            "skill_ids": ["<proc>"],
            "served_ids": [],
            "compact": False,
            "left_out_ids": [],
            "skipped": None,
        },
        "usage": {"tokens": 5},
    }
    other = json.loads(json.dumps(batch))
    other["skill_usage"]["tokens"] = 91  # another cell's ids: a few tokens apart
    assert normalized(other) == normalized(batch)
    other["skill_usage"]["compact"] = True
    assert normalized(other) != normalized(batch)


def _owned_server_pid(cell: Path) -> int:
    rows = read_rows(cell / "owned_processes.jsonl")
    started = [r for r in rows if r.get("event") == "started"]
    stopped = {r.get("pid") for r in rows if r.get("event") == "stopped"}
    (pid,) = [int(r["pid"]) for r in started if r["pid"] not in stopped]
    return pid


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child of nobody we wait for still answers kill(0)
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return True
    return bool(state) and not state.startswith("Z")


@dataclasses.dataclass
class MirixBench:
    bench: Bench
    reference: Path
    prefix: str


@pytest.fixture(scope="module")
def mirix(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MirixBench]:
    """A gateway whose upstream plays MIRIX's model, and a reference cell that learned
    uninterrupted; the tests' cell databases and MIRIX servers are removed afterwards."""
    if not _postgres_up():
        pytest.skip(
            "skill_bench Postgres not reachable "
            "(docker compose -f evals/skill_bench/infra/compose.yaml up -d)"
        )
    root = tmp_path_factory.mktemp("mirix")
    tag = f"rt{secrets.token_hex(3)}"
    prefix = f"sb_alfworld_luna_mirix_{tag}"
    upstream = HeldUpstream(MirixModel(fallback=learners))
    try:
        with serve_gateway(
            root,
            FakeUpstream(responder=upstream),
            backoff_base_s=protocol.GATEWAY_BACKOFF_BASE_S,
        ) as (_gw, url):
            bench = Bench(root, url, upstream, tag)
            try:
                reference = root / "reference"
                bench.run(reference, "mirix")
                yield MirixBench(bench, reference, prefix)
            finally:
                bench.stop()
    finally:
        for registry in root.glob("*/owned_processes.jsonl"):
            cleanup_owned(registry)
        for name in asyncio.run(_database_names(prefix)):
            assert "smoke" not in name and name.startswith(prefix), name
            asyncio.run(drop_cell_database(PostgresAdmin(), name))


def _holding(cell: Path, checkpoint: str, predicate: Predicate) -> Predicate:
    """``predicate``, for a call made while ``checkpoint`` is the cell's LATEST."""
    return lambda endpoint, body: latest(cell) == checkpoint and predicate(
        endpoint, body
    )


MIRIX_CASES = {
    # name: (what is held, LATEST at the kill, in-flight stage, position)
    "mid-item": (
        lambda cell: actor_call(question(TRAIN[3])),
        "c0002-done",
        "rollout",
        3,
    ),
    "in-batch": (
        lambda cell: _holding(cell, "c0004-played", curator_call),
        "c0004-played",
        "after_task",
        4,
    ),
    "in-freeze": (
        lambda cell: _holding(cell, "c0005-done", curator_call),
        "c0005-done",
        "freeze",
        None,
    ),
}


@pytest.mark.integration
@pytest.mark.parametrize("case", list(MIRIX_CASES))
def test_a_real_mirix_cell_killed_in_learning_resumes_to_the_uninterrupted_memory(
    mirix, case
):
    """A real MIRIX cell, SIGKILLed with a call in flight: in the middle of a learning item (the actor's call), while the
    curator batch that the fifth close started runs, or in the freeze's flush. The dead
    harness's MIRIX server survives it and still holds the cell database; the resume
    stops it, moves the post-crash database aside, exports what only it holds (the
    killed item's retrieval and session, the stopped learning run), drops it, restores
    the checkpoint's database and HOME, verifies both, and replays. The live database
    then holds what an uninterrupted cell's does (table by table), and the frozen memory is the uninterrupted one up to MIRIX's random ids and times."""
    make_hold, checkpoint, stage, position = MIRIX_CASES[case]
    bench = mirix.bench
    cell = bench.root / f"killed-{case}"
    bench.upstream.hold(make_hold(cell))
    killed = kill_when(
        bench.start(cell, "mirix"), cell, held_at(bench, cell, checkpoint)
    )
    assert killed == -signal.SIGKILL
    paths = CellPaths(cell)
    state = read_json(paths.attempt(1).state)
    assert state["state"] == ATTEMPT_LEARNING
    db_name = state["mirix"]["db_name"]
    assert db_name.startswith(mirix.prefix)
    orphan = _owned_server_pid(cell)
    assert _alive(orphan)  # the dead harness's MIRIX server outlives it

    ledger = run_ledger(bench, cell)
    release_after_resume(bench, ledger)
    bench.run(cell, "mirix")
    assert not _alive(orphan)

    row = the_row(cell)
    assert (row["reason"], row["checkpoint"]["id"]) == ("process_killed", checkpoint)
    assert (row["in_flight"]["position"], row["in_flight"]["stage"]) == (
        position,
        stage,
    )
    assert row["restore"]["db_digest_equal"] is True
    assert row["mirix"]["post_crash_db"] == "dropped"
    assert asyncio.run(_database_names(db_name)) == [db_name]  # no <db>_dNN is left
    export = read_json(paths.attempt(1).discarded_dir(1) / EXPORT_FILE)
    assert export is not None
    roles = Counter(c["role"] for c in row["calls"])
    if case == "mid-item":
        uid = row["in_flight"]["task_uid"]
        assert roles["actor"] == 1 and roles["mirix"] >= 1  # the retrieval's extraction
        (retrieval,) = row["mirix"]["retrievals"]
        assert retrieval["session_id"] == uid
        # The killed try's retrieval-log row is in the export, not in the live database.
        exported = export["rows"]["retrieval_log"]
        assert [r["id"] for r in exported] == [retrieval["id"]]
        assert not asyncio.run(_rows_with_id(db_name, "retrieval_log", retrieval["id"]))
        assert asyncio.run(_rows_with_id(db_name, "conversation_session", None, uid))
        (session,) = [s for s in row["mirix"]["sessions"] if s["session_id"] == uid]
        assert session["closed"] is False and session["messages"] >= 1
        assert not row["mirix"]["learning_runs"]
    else:
        assert "actor" not in roles and set(roles) <= {"mirix", "embedding"}
        (stopped,) = row["mirix"]["learning_runs"]
        assert stopped["trigger"] == ("close" if case == "in-batch" else "flush")
        assert stopped["status"] in ("queued", "running")
    assert_zombie_after_the_marker(ledger, row)
    assert_calls_once(paths, CellPaths(mirix.reference), row)
    resumed_cleanly(ledger, paths, storing=False)

    reference_db = read_json(CellPaths(mirix.reference).attempt(1).state)["mirix"][
        "db_name"
    ]
    assert asyncio.run(_table_counts(db_name)) == asyncio.run(
        _table_counts(reference_db)
    )
    assert normalized(frozen(cell)) == normalized(frozen(mirix.reference))
