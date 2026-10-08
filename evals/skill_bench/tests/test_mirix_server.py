"""Per-cell MIRIX isolation with a stub server and a fake database connection."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from evals.skill_bench import protocol
from evals.skill_bench.infra import mirix_server as ms
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    cleanup_owned,
    find_free_port,
    process_start_time,
    registry_in_use,
    sweep_owned,
)
from evals.skill_bench.tests.fakes import write_stub_server

PROVIDER_ENV = {
    "OPENAI_API_KEY": "sk-should-not-leak-1",
    "OPENROUTER_API_KEY": "sk-or-should-not-leak",
    "ANTHROPIC_API_KEY": "sk-ant-should-not-leak",
    "GEMINI_API_KEY": "g-should-not-leak",
    "GOOGLE_API_KEY": "g2-should-not-leak",
    "AWS_SECRET_ACCESS_KEY": "aws-should-not-leak",
    "LANGFUSE_SECRET_KEY": "lf-should-not-leak",
    "AZURE_OPENAI_ENDPOINT": "https://example.invalid",
}


def spec(
    tmp_path: Path, cell_id: str, attempt: int = 0, port: int | None = None, **kw: Any
) -> ms.CellServerSpec:
    return ms.CellServerSpec(
        cell_id, attempt, tmp_path / "cells" / cell_id, port or find_free_port(), **kw
    )


def snapshot(directory: Path) -> dict[str, float]:
    if not directory.exists():
        return {}
    return {
        str(p.relative_to(directory)): p.stat().st_mtime for p in directory.rglob("*")
    }


def test_cells_get_distinct_databases_homes_and_logs(tmp_path):
    a = spec(tmp_path, "alfworld-luna-mirix")
    b = spec(tmp_path, "alfworld-luna56-mirix")
    a1 = spec(tmp_path, "alfworld-luna-mirix", attempt=1)
    names = {a.db_name, b.db_name, a1.db_name}
    assert len(names) == 3
    assert all(ms.DB_NAME_RE.fullmatch(n) and len(n) <= 63 for n in names)
    assert (
        len({a.home, b.home, a1.home}) == 3
        and len({a.log_path, b.log_path, a1.log_path}) == 3
    )
    assert a.db_name == ms.cell_db_name(
        "alfworld-luna-mirix", 0
    )  # deterministic: test-phase resume reuses it
    assert a.pg_uri.endswith("/" + a.db_name) and a.pg_uri.startswith(
        "postgresql+asyncpg://"
    )
    assert ms.cell_db_name("a-b", 0) != ms.cell_db_name("a_b", 0)
    assert len(ms.cell_db_name("x" * 200, 12)) <= 63


def test_child_env_has_isolation_settings_and_no_provider_keys(tmp_path):
    cell = spec(
        tmp_path,
        "webshop-luna56-mirix",
        extra_env={"SKILL_TRIGGER_SESSION_THRESHOLD": "5"},
    )
    base = {
        **PROVIDER_ENV,
        "PATH": "/usr/bin",
        "HTTP_PROXY": "http://proxy:1",
        "PYTHONPATH": "/x",
        "LANG": "C",
    }
    env = ms.build_server_env(cell, base)
    assert not any(ms.is_secret_env_name(k) for k in env)
    assert not set(PROVIDER_ENV) & set(env)
    assert "HTTP_PROXY" not in env and "PYTHONPATH" not in env
    assert env["PATH"] == "/usr/bin" and env["LANG"] == "C"
    assert env["HOME"] == str(cell.home)
    assert env["MIRIX_PG_URI"] == cell.pg_uri
    assert env["MIRIX_REDIS_ENABLED"] == "false"
    assert (
        env["MIRIX_LLM_MAX_RETRIES"] == "0"
        and env["MIRIX_LLM_TIMEOUT_SECONDS"] == "630"
        and env["MIRIX_EMBEDDING_TIMEOUT_SECONDS"] == "630"
    )
    assert env["SKILL_TRIGGER_SESSION_THRESHOLD"] == "5"
    # B32: run.py asks MIRIX to name the session it serves on its gateway calls, in
    # the gateway's own item header (no credential, not a protected setting).
    from evals.skill_bench.methods.base import ITEM_HEADER

    tagged = spec(tmp_path, "c", extra_env={ms.MIRIX_SESSION_HEADER_ENV: ITEM_HEADER})
    assert ms.build_server_env(tagged, base)["MIRIX_LLM_SESSION_HEADER"] == (
        "x-skill-bench-item"
    )
    for bad in (
        {"OPENAI_API_KEY": "x"},
        {"HOME": "/tmp"},
        {"MIRIX_LLM_MAX_RETRIES": "2"},
        {"MY_SERVICE_TOKEN": "x"},
    ):
        with pytest.raises(ms.MirixServerError):
            ms.build_server_env(spec(tmp_path, "c", extra_env=bad), base)


@pytest.mark.asyncio
async def test_two_stub_cells_run_isolated_without_touching_home(tmp_path, monkeypatch):
    for name, value in PROVIDER_ENV.items():
        monkeypatch.setenv(name, value)
    stub = write_stub_server(tmp_path)
    real_mirix = Path.home() / ".mirix"
    before = snapshot(real_mirix)
    cells = [
        spec(tmp_path, "alfworld-luna-mirix"),
        spec(tmp_path, "demo-luna-mirix"),
    ]
    servers = [
        ms.MirixServer(
            c,
            command=[sys.executable, str(stub), "--port", str(c.port)],
            database=False,
            mode="fresh",
        )
        for c in cells
    ]
    envs = []
    try:
        for server in servers:
            await server.start()
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            for server in servers:
                envs.append((await client.get(server.base_url + "/env")).json())
    finally:
        for server in servers:
            await server.stop()
    for cell, env in zip(cells, envs, strict=True):
        assert env["HOME"] == str(cell.home)
        assert env["MIRIX_PG_URI"] == cell.pg_uri
        assert not any(ms.is_secret_env_name(k) for k in env), sorted(env)
        assert not any(v in PROVIDER_ENV.values() for v in env.values())
        assert (cell.home / ".mirix" / "stub_marker").is_file()
        assert "stub mirix server starting" in cell.log_path.read_text()
        events = [
            json.loads(line)["event"] for line in cell.registry.read_text().splitlines()
        ]
        assert events == ["started", "stopped"]
    assert envs[0]["MIRIX_PG_URI"] != envs[1]["MIRIX_PG_URI"]
    assert snapshot(real_mirix) == before
    assert all(s.process is not None and not s.process.alive() for s in servers)


@pytest.mark.asyncio
async def test_fresh_attempt_refuses_used_home(tmp_path):
    cell = spec(tmp_path, "c1")
    cell.home.mkdir(parents=True)
    (cell.home / "leftover").write_text("state from a failed attempt")
    server = ms.MirixServer(
        cell, command=[sys.executable, "-c", "pass"], database=False, mode="fresh"
    )
    with pytest.raises(ms.MirixServerError, match="fresh HOME"):
        await server.start()


@pytest.mark.asyncio
async def test_crashing_server_is_reported_as_mirix_server_error(tmp_path):
    cell = spec(tmp_path, "c2")
    server = ms.MirixServer(
        cell,
        command=[sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
        database=False,
        mode="fresh",
        startup_timeout_s=20,
    )
    with pytest.raises(ms.MirixServerError, match="exited with code 3") as excinfo:
        await server.start()
    assert "boom" in str(excinfo.value)
    with pytest.raises(ms.MirixServerError):
        await server.ensure_alive()


@pytest.mark.asyncio
async def test_port_in_use_is_rejected(tmp_path):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        cell = spec(tmp_path, "c3", port=sock.getsockname()[1])
        server = ms.MirixServer(
            cell, command=[sys.executable, "-c", "pass"], database=False, mode="fresh"
        )
        with pytest.raises(ms.MirixServerError, match="in use"):
            await server.start()


def _pg_ident(name: str) -> str:
    """Postgres's ``format('%I')``: a plain lower-case name stays bare."""
    if re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _pg_literal(text: str) -> str:
    """Postgres's ``format('%L')`` (``quote_literal``)."""
    if "\\" in text:
        return "E'" + text.replace("\\", "\\\\").replace("'", "''") + "'"
    return "'" + text.replace("'", "''") + "'"


_COMMENT_SQL = re.compile(
    r'COMMENT ON DATABASE ("(?:[^"]|"")+"|[a-z_][a-z0-9_]*) IS (E?)\'(.*)\'', re.S
)


class FakeConn:
    """A fake asyncpg connection to the compose Postgres: the databases that exist
    (``existing``) and their comments (``comments``, set by ``COMMENT ON DATABASE``),
    answering the statements ``infra/mirix_server.py`` sends by their text. Every
    statement is logged as ``(database connected to, sql)``."""

    def __init__(
        self,
        log: list[tuple[str, str]],
        database: str,
        existing: set[str],
        comments: dict[str, str] | None = None,
    ) -> None:
        self.log, self.database, self.existing = log, database, existing
        self.comments = comments if comments is not None else {}

    async def execute(self, sql: str, *args: Any) -> None:
        self.log.append((self.database, sql))
        if sql.startswith("CREATE DATABASE"):
            self.existing.add(sql.split('"')[1])
        elif sql.startswith("COMMENT ON DATABASE") and sql.endswith(" IS NULL"):
            name = sql[len("COMMENT ON DATABASE ") : -len(" IS NULL")]
            if name.startswith('"'):
                name = name[1:-1].replace('""', '"')
            assert name in self.existing, f"COMMENT on a missing database {name}"
            self.comments.pop(name, None)
        elif sql.startswith("COMMENT ON DATABASE"):
            match = _COMMENT_SQL.fullmatch(sql)
            assert match, sql
            name = match[1]
            if name.startswith('"'):
                name = name[1:-1].replace('""', '"')
            text = match[3].replace("''", "'")
            if match[2]:
                text = text.replace("\\\\", "\\")
            assert name in self.existing, f"COMMENT on a missing database {name}"
            self.comments[name] = text
        elif sql.startswith("ALTER DATABASE"):
            old, new = sql.split('"')[1], sql.split('"')[3]
            assert old in self.existing and new not in self.existing, sql
            self.existing.discard(old)
            self.existing.add(new)
            if old in self.comments:
                self.comments[new] = self.comments.pop(old)
        elif sql.startswith("DROP DATABASE"):
            name = sql.split('"')[1]
            self.existing.discard(name)
            self.comments.pop(name, None)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.log.append((self.database, sql))
        if sql.startswith("SELECT format('COMMENT ON DATABASE %I IS %L'"):
            return f"COMMENT ON DATABASE {_pg_ident(args[0])} IS {_pg_literal(args[1])}"
        if sql.startswith("SELECT format('COMMENT ON DATABASE %I IS NULL'"):
            return f"COMMENT ON DATABASE {_pg_ident(args[0])} IS NULL"
        if "pg_stat_activity" in sql:
            return 0
        assert sql == "SELECT 1 FROM pg_database WHERE datname = $1", sql
        return 1 if args[0] in self.existing else None

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.log.append((self.database, sql))
        assert "shobj_description(oid, 'pg_database')" in sql, sql
        pattern = re.compile(args[1]) if len(args) > 1 else None
        return [
            {"datname": name, "comment": self.comments.get(name)}
            for name in sorted(self.existing)
            if name == args[0] or (pattern is not None and pattern.search(name))
        ]

    async def close(self) -> None:
        self.log.append((self.database, "CLOSE"))


def fake_connect(
    log: list[tuple[str, str]],
    existing: set[str],
    comments: dict[str, str] | None = None,
):
    shared = comments if comments is not None else {}

    async def connect(**kwargs: Any) -> FakeConn:
        assert (
            kwargs["password"] == "skill_bench_local" and kwargs["host"] == "127.0.0.1"
        )
        return FakeConn(log, kwargs["database"], existing, shared)

    return connect


def changes(log: list[tuple[str, str]]) -> list[str]:
    """The statements of ``log`` that change a database or its marker."""
    return [
        sql
        for _, sql in log
        if sql.startswith(("CREATE DATABASE", "COMMENT", "ALTER", "DROP"))
    ]


#: One cell id in two runs directories (``drill`` and ``drill2``, B02).
DRILL_CELL = "webshop-luna-mirix-dev-l8-t2"
#: The name both runs directories computed before names carried the instance.
SHARED_LEGACY_NAME = ms.cell_db_name(DRILL_CELL, 1)


def owner(instance: str = "164c9dcf") -> Any:
    """The owner of attempt 1 of the drill cell in the cell directory ``instance``."""
    return ms.DatabaseOwner(f"{DRILL_CELL}-a01-{instance}", DRILL_CELL, 1, instance)


@pytest.mark.asyncio
async def test_ensure_cell_database_clones_template_under_lock():
    pg = ms.PostgresAdmin()
    name = ms.cell_db_name("webshop-luna-awm", 0)
    log: list[tuple[str, str]] = []
    existing: set[str] = set()
    assert (
        await ms.ensure_cell_database(
            pg, name, must_be_new=True, connect=fake_connect(log, existing)
        )
        is True
    )
    statements = [sql for _, sql in log]
    assert statements[0].startswith("SELECT pg_advisory_lock")
    assert f'CREATE DATABASE "{name}" TEMPLATE "skill_bench_template"' in statements
    assert statements.index("SELECT pg_advisory_unlock($1)") > statements.index(
        f'CREATE DATABASE "{name}" TEMPLATE "skill_bench_template"'
    )
    in_cell = [sql for db, sql in log if db == name and sql != "CLOSE"]
    assert in_cell == [
        f'CREATE EXTENSION IF NOT EXISTS "{e}"'
        for e in ("vector", "pg_trgm", "uuid-ossp")
    ]

    log.clear()
    assert (
        await ms.ensure_cell_database(pg, name, connect=fake_connect(log, existing))
        is False
    )  # resume reuses
    assert not any(sql.startswith("CREATE DATABASE") for _, sql in log)

    log.clear()
    with pytest.raises(ms.MirixServerError, match="fresh"):
        await ms.ensure_cell_database(
            pg, name, must_be_new=True, connect=fake_connect(log, existing)
        )
    assert ("postgres", "SELECT pg_advisory_unlock($1)") in log and (
        "postgres",
        "CLOSE",
    ) in log

    with pytest.raises(ValueError):
        await ms.ensure_cell_database(
            pg, 'x"; DROP DATABASE postgres; --', connect=fake_connect(log, existing)
        )
    with pytest.raises(ValueError):
        await ms.drop_cell_database(pg, "postgres", connect=fake_connect(log, existing))


def test_dotenv_guard_finds_files_mirix_would_load(tmp_path):
    package = tmp_path / "repo" / "mirix"
    package.mkdir(parents=True)
    dotenv = tmp_path / "repo" / ".env"
    assert dotenv not in ms.dotenv_files_visible_to_mirix(package)
    dotenv.write_text("NAME=value\n")
    assert dotenv in ms.dotenv_files_visible_to_mirix(package)


def test_default_command_runs_repo_start_server(tmp_path):
    cell = spec(tmp_path, "c4", port=18999)
    command = ms.default_command(cell)
    assert command[1] == str(protocol.REPO_ROOT / "scripts" / "start_server.py")
    assert command[-6:] == [
        "--host",
        "127.0.0.1",
        "--port",
        "18999",
        "--log-level",
        "warning",
    ]


def test_cleanup_kills_only_registered_groups(tmp_path):
    registry = tmp_path / "owned.jsonl"
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    owned = OwnedProcess.spawn(
        "owned",
        sleeper,
        env=dict(os.environ),
        cwd=tmp_path,
        log_path=tmp_path / "o.log",
        registry=registry,
    )
    other = subprocess.Popen(sleeper, start_new_session=True)
    try:
        assert cleanup_owned(registry) == [
            owned.pid
        ]  # harness "crashed": owned was never stopped
        assert owned.popen.wait(timeout=10) == -signal.SIGKILL
        assert other.poll() is None
        assert cleanup_owned(registry) == []  # idempotent
        # A registered PID that now belongs to another process (e.g. after a reboot) is
        # never signalled; the row is closed as pid_reused so the cell is not blocked.
        with registry.open("a") as fh:
            row = {
                "event": "started",
                "name": "x",
                "pid": other.pid,
                "pgid": other.pid,
                "started": "Thu Jan  1 00:00:00 1970",
            }
            fh.write(json.dumps(row) + "\n")
        assert cleanup_owned(registry) == []
        assert other.poll() is None
        last = json.loads(registry.read_text().splitlines()[-1])
        assert (last["event"], last["pid"], last["reason"]) == (
            "cleaned",
            other.pid,
            "pid_reused",
        )
        assert cleanup_owned(registry) == []  # the stale row no longer blocks anything
    finally:
        for proc in (owned.popen, other):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


@pytest.mark.asyncio
async def test_owned_process_stop_escalates_to_sigkill(tmp_path):
    stubborn = [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)",
    ]
    proc = OwnedProcess.spawn(
        "stubborn",
        stubborn,
        env=dict(os.environ),
        cwd=tmp_path,
        log_path=tmp_path / "s.log",
        registry=tmp_path / "r.jsonl",
    )
    deadline = time.monotonic() + 10
    while (
        "ready" not in (tmp_path / "s.log").read_text() and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    started = time.monotonic()
    code = await proc.stop(grace_s=0.5)
    assert code == -signal.SIGKILL and time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_one_live_server_per_cell(tmp_path):
    stub = write_stub_server(tmp_path)
    cell = spec(tmp_path, "locked-cell")
    command = [sys.executable, str(stub), "--port", str(cell.port)]
    first = ms.MirixServer(cell, command=command, database=False, mode="fresh")
    second = ms.MirixServer(cell, command=command, database=False, mode="resume")
    await first.start()
    try:
        with pytest.raises(ms.MirixServerError, match="owned by another"):
            await second.start()
        assert (
            first.process is not None and first.process.alive()
        )  # not killed by cleanup
    finally:
        await first.stop()
    await second.start()  # the lock is released on stop
    await second.stop()


def test_process_start_time_does_not_depend_on_the_locale(monkeypatch):
    pid = os.getpid()
    c_locale = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        check=True,
    ).stdout
    expected = " ".join(c_locale.split())
    assert process_start_time(pid) == expected
    # A harness resumed under another locale ("lun. 28 sept." instead of "Mon Sep 28")
    # must still recognise its own processes.
    for name, value in (("LC_ALL", "fr_FR.UTF-8"), ("LANG", "fr_FR.UTF-8")):
        monkeypatch.setenv(name, value)
    assert process_start_time(pid) == expected


def test_stale_registry_after_a_reboot_never_blocks_the_cell(tmp_path):
    """Registry rows from before a reboot: the PIDs are gone or belong to strangers."""
    registry = tmp_path / "cell" / "owned_processes.jsonl"
    registry.parent.mkdir()
    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=10)
    try:
        rows = [
            {"event": "started", "name": "mirix", "pid": pid, "pgid": pid, "started": s}
            for pid, s in (
                (stranger.pid, "Mon Jan  1 00:00:00 2024"),  # reused PID
                (gone.pid, "Mon Jan  1 00:00:00 2024"),  # nothing runs there now
            )
        ]
        registry.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert cleanup_owned(registry) == []
        assert stranger.poll() is None  # never signalled
        reasons = {
            row["pid"]: row["reason"]
            for row in map(json.loads, registry.read_text().splitlines())
            if row["event"] == "cleaned"
        }
        assert reasons == {stranger.pid: "pid_reused", gone.pid: "gone"}
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


@pytest.mark.asyncio
async def test_a_stale_registry_does_not_stop_a_cell_from_starting(tmp_path):
    stub = write_stub_server(tmp_path)
    cell = spec(tmp_path, "rebooted-cell")
    cell.registry.parent.mkdir(parents=True)
    cell.registry.write_text(
        json.dumps(
            {
                "event": "started",
                "name": "mirix",
                "pid": os.getpid(),  # "reused" by the test process itself
                "pgid": os.getpid(),
                "started": "Mon Jan  1 00:00:00 2024",
            }
        )
        + "\n"
    )
    server = ms.MirixServer(
        cell,
        command=[sys.executable, str(stub), "--port", str(cell.port)],
        database=False,
        mode="fresh",
    )
    await server.start()
    await server.stop()


def test_sweep_cleans_every_registry_except_live_cells(tmp_path):
    runs = tmp_path / "runs"
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    procs = {}
    for cell in ("cell-a", "cell-b", "_gateway"):
        procs[cell] = OwnedProcess.spawn(
            cell,
            sleeper,
            env=dict(os.environ),
            cwd=tmp_path,
            log_path=tmp_path / f"{cell}.log",
            registry=runs / cell / "owned_processes.jsonl",
        )
    import fcntl

    lock = (runs / "cell-b" / "server.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # a live harness owns cell-b
    try:
        assert registry_in_use(runs / "cell-b" / "owned_processes.jsonl")
        killed = sweep_owned(runs)
        assert killed == {
            "_gateway/owned_processes.jsonl": [procs["_gateway"].pid],
            "cell-a/owned_processes.jsonl": [procs["cell-a"].pid],
        }
        assert procs["cell-b"].alive()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
        for proc in procs.values():
            if proc.popen.poll() is None:
                proc.popen.kill()
            proc.popen.wait(timeout=10)


@pytest.mark.asyncio
async def test_resume_refuses_a_missing_database_or_home(tmp_path):
    log: list[tuple[str, str]] = []
    existing: set[str] = set()
    cell = spec(tmp_path, "resumed-cell")
    command = [sys.executable, "-c", "pass"]
    server = ms.MirixServer(
        cell,
        command=command,
        mode="resume",
        connect=fake_connect(log, existing),
    )
    with pytest.raises(ms.MirixServerError, match="missing or empty"):
        await server.start()  # the attempt's HOME is gone
    # A HOME with other entries, or an empty ~/.mirix, holds no MIRIX state either.
    cell.home.mkdir(parents=True)
    (cell.home / ".cache").mkdir()
    (cell.home / ".mirix").mkdir()
    server = ms.MirixServer(
        cell, command=command, mode="resume", connect=fake_connect(log, existing)
    )
    with pytest.raises(ms.MirixServerError, match="missing or empty") as refused:
        await server.start()
    assert "link-nested-home" not in str(refused.value)  # nothing nested to point at
    (cell.home / ".mirix" / "config").write_text("[defaults]\n")  # what MIRIX writes
    server = ms.MirixServer(
        cell,
        command=command,
        mode="resume",
        connect=fake_connect(log, existing),
    )
    with pytest.raises(ms.MirixServerError, match="does not exist"):
        await server.start()  # e.g. `docker compose down -v` dropped the database
    assert not any(sql.startswith("CREATE DATABASE") for _, sql in log)
    with pytest.raises(ValueError):
        ms.MirixServer(cell, command=command, mode="restart")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ensure_cell_database_must_exist():
    pg = ms.PostgresAdmin()
    name = ms.cell_db_name("webshop-luna56-mirix", 1)
    log: list[tuple[str, str]] = []
    existing: set[str] = set()
    with pytest.raises(ms.MirixServerError, match="does not exist"):
        await ms.ensure_cell_database(
            pg, name, must_exist=True, connect=fake_connect(log, existing)
        )
    assert ("postgres", "SELECT pg_advisory_unlock($1)") in log
    existing.add(name)
    assert (
        await ms.ensure_cell_database(
            pg, name, must_exist=True, connect=fake_connect(log, existing)
        )
        is False
    )
    with pytest.raises(ValueError):
        await ms.ensure_cell_database(
            pg,
            name,
            must_be_new=True,
            must_exist=True,
            connect=fake_connect(log, existing),
        )


IDLE = protocol.MIRIX_SESSION_IDLE_TIMEOUT_S


def test_llm_contract_check():
    good = {
        "llm_max_retries": 0,
        "llm_timeout_seconds": 630.0,
        "embedding_timeout_seconds": 630.0,
        "SESSION_IDLE_TIMEOUT_SECONDS": IDLE,
    }
    assert ms.check_llm_contract(good) == []
    missing = ms.check_llm_contract({**good, "llm_max_retries": {"missing": True}})
    assert missing == [
        "mirix.settings has no 'llm_max_retries' (does not read $MIRIX_LLM_MAX_RETRIES)"
    ]
    wrong = ms.check_llm_contract(
        {
            "llm_max_retries": 2,
            "llm_timeout_seconds": 600,
            "embedding_timeout_seconds": 60,
            "SESSION_IDLE_TIMEOUT_SECONDS": IDLE,
        }
    )
    assert len(wrong) == 3 and "MIRIX_LLM_MAX_RETRIES=0" in wrong[0]
    # A MIRIX whose embedding calls give up on the gateway after 60 s.
    assert "MIRIX_EMBEDDING_TIMEOUT_SECONDS=630" in wrong[2]
    assert ms.check_llm_contract(
        {**good, "embedding_timeout_seconds": {"missing": True}}
    ) == [
        "mirix.settings has no 'embedding_timeout_seconds' (does not read "
        "$MIRIX_EMBEDDING_TIMEOUT_SECONDS)"
    ]
    # A MIRIX whose task sessions end by idling after its default 48 h.
    assert ms.check_llm_contract({**good, "SESSION_IDLE_TIMEOUT_SECONDS": 172800}) == [
        f"constants.SESSION_IDLE_TIMEOUT_SECONDS = 172800, but the cell sets "
        f"MIRIX_SESSION_IDLE_TIMEOUT_SECONDS={IDLE}"
    ]
    assert ms.check_llm_contract(
        {**good, "SESSION_IDLE_TIMEOUT_SECONDS": {"missing": True}}
    ) == [
        "mirix.constants has no 'SESSION_IDLE_TIMEOUT_SECONDS' (does not read "
        "$MIRIX_SESSION_IDLE_TIMEOUT_SECONDS)"
    ]


@pytest.mark.asyncio
async def test_llm_contract_probe_reads_mirix_settings_in_a_scratch_home(tmp_path):
    fake_repo = tmp_path / "repo"
    (fake_repo / "mirix").mkdir(parents=True)
    (fake_repo / "mirix" / "__init__.py").write_text("")
    (fake_repo / "mirix" / "settings.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.path.expanduser('~'), 'probe_was_here').write_text('x')\n"
        "class S:\n"
        "    llm_max_retries = int(os.environ['MIRIX_LLM_MAX_RETRIES'])\n"
        "    llm_timeout_seconds = float(os.environ['MIRIX_LLM_TIMEOUT_SECONDS'])\n"
        "    embedding_timeout_seconds = float(\n"
        "        os.environ['MIRIX_EMBEDDING_TIMEOUT_SECONDS']\n"
        "    )\n"
        "settings = S()\n"
    )
    (fake_repo / "mirix" / "constants.py").write_text(
        "import os\n"
        "SKILL_EXPERIENCE_POOL_ENABLED = True\nSKILL_EXPERIENCE_POOL_DECAY = 0.5\n"
        "SESSION_IDLE_TIMEOUT_SECONDS = int(\n"
        "    os.environ['MIRIX_SESSION_IDLE_TIMEOUT_SECONDS']\n"
        ")\n"
    )
    cell = spec(tmp_path, "probe-cell")
    env = ms.build_server_env(cell, {"PATH": os.environ["PATH"]})
    values = await ms.probe_llm_contract(env, repo=fake_repo)
    assert ms.check_llm_contract(values) == []
    # B32 (informational, never checked): a MIRIX without the session header setting.
    assert values["llm_session_header"] == {"missing": True}
    # The probe records the pool settings the cell runs with (None: this
    # MIRIX has no such constant).
    assert set(values["skill_experience_pool"]) == set(ms.POOL_SETTINGS_PROBED)
    assert values["skill_experience_pool"]["pool_enabled"] is True
    assert values["skill_experience_pool"]["pool_decay"] == 0.5
    assert values["skill_experience_pool"]["history_top_k"] is None
    assert not cell.home.exists()  # the probe ran in a throw-away HOME
    (fake_repo / "mirix" / "settings.py").write_text("settings = object()\n")
    values = await ms.probe_llm_contract(env, repo=fake_repo)
    assert len(ms.check_llm_contract(values)) == 3


@pytest.mark.asyncio
async def test_this_checkouts_mirix_honours_the_retry_timeout_contract(tmp_path):
    """This checkout's ``mirix.settings`` reads ``MIRIX_LLM_MAX_RETRIES``,
    ``MIRIX_LLM_TIMEOUT_SECONDS`` and ``MIRIX_EMBEDDING_TIMEOUT_SECONDS``, so a real
    MIRIX cell starts. The probe imports it in a child process with a throw-away HOME;
    nothing connects anywhere."""
    cell = spec(tmp_path, "probe-this-checkout")
    env = ms.build_server_env(cell)
    values = await ms.probe_llm_contract(env)
    # B32: off unless the cell asks for it; run.py's cells do (informational).
    assert values.pop("llm_session_header") is None
    tagged = replace(
        cell, extra_env={ms.MIRIX_SESSION_HEADER_ENV: "x-skill-bench-item"}
    )
    tagged_values = await ms.probe_llm_contract(ms.build_server_env(tagged))
    assert tagged_values["llm_session_header"] == "x-skill-bench-item"
    pool = values.pop("skill_experience_pool")
    assert set(pool) == set(ms.POOL_SETTINGS_PROBED)
    assert all(value is not None for value in pool.values())
    # MIRIX's task sessions never end by idling in a cell.
    assert values.pop("SESSION_IDLE_TIMEOUT_SECONDS") == int(
        protocol.MIRIX_SERVER_ENV["MIRIX_SESSION_IDLE_TIMEOUT_SECONDS"]
    )
    assert values == {
        "llm_max_retries": int(protocol.MIRIX_SERVER_ENV["MIRIX_LLM_MAX_RETRIES"]),
        "llm_timeout_seconds": float(
            protocol.MIRIX_SERVER_ENV["MIRIX_LLM_TIMEOUT_SECONDS"]
        ),
        "embedding_timeout_seconds": float(
            protocol.MIRIX_SERVER_ENV["MIRIX_EMBEDDING_TIMEOUT_SECONDS"]
        ),
    }
    assert values == {
        "llm_max_retries": 0,
        "llm_timeout_seconds": protocol.CLIENT_TIMEOUT_S,
        "embedding_timeout_seconds": protocol.CLIENT_TIMEOUT_S,
    }
    assert protocol.CLIENT_TIMEOUT_S > protocol.CALL_DEADLINE_S
    assert (
        ms.check_llm_contract(
            {
                **values,
                "SESSION_IDLE_TIMEOUT_SECONDS": protocol.MIRIX_SESSION_IDLE_TIMEOUT_S,
            }
        )
        == []
    )
    assert not cell.home.exists()


def test_a_cells_task_session_never_ends_by_idling(tmp_path):
    """A learning resume from a played checkpoint restores session p open, with its last
    message at the original time. With MIRIX's default 48 h idle timeout, a resume later
    than that (a person's decision after budget_spent, a weekend) would find p ended at
    the resumed server's startup sweep, and after_task's close would answer
    already_closed (a HARNESS stop). Every cell's MIRIX runs with the protocol's
    timeout, which extra_env cannot shorten, and under it a session a year old is still
    open."""
    import datetime as dt
    from types import SimpleNamespace

    from mirix.constants import DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS
    from mirix.services.conversation_session_manager import (
        ConversationSessionManager,
    )

    cell = spec(tmp_path, "idle-cell")
    env = ms.build_server_env(cell)
    timeout = int(env["MIRIX_SESSION_IDLE_TIMEOUT_SECONDS"])
    assert timeout == protocol.MIRIX_SESSION_IDLE_TIMEOUT_S >= 10 * 365 * 86_400
    with pytest.raises(ms.MirixServerError, match="MIRIX_SESSION_IDLE_TIMEOUT"):
        ms.build_server_env(
            replace(cell, extra_env={"MIRIX_SESSION_IDLE_TIMEOUT_SECONDS": "3"})
        )
    now = dt.datetime(2026, 9, 29, tzinfo=dt.timezone.utc)
    session = SimpleNamespace(
        closed_at=None, last_message_at=now - dt.timedelta(days=365)
    )
    cell_manager = ConversationSessionManager(
        session_maker=object(), idle_timeout_seconds=timeout
    )
    assert not cell_manager.is_ended(session, now)
    # The closing time a materialisation would compute stays a valid datetime.
    assert session.last_message_at + cell_manager.idle_timeout > now
    default = ConversationSessionManager(
        session_maker=object(),
        idle_timeout_seconds=DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    )
    assert default.is_ended(
        SimpleNamespace(closed_at=None, last_message_at=now - dt.timedelta(days=3)),
        now,
    )


@pytest.mark.asyncio
async def test_start_refuses_a_mirix_that_would_retry_on_its_own(tmp_path, monkeypatch):
    async def probe(env, **kwargs):
        return {"llm_max_retries": {"missing": True}, "llm_timeout_seconds": 600}

    monkeypatch.setattr(ms, "probe_llm_contract", probe)
    stub = write_stub_server(tmp_path)
    cell = spec(tmp_path, "contract-cell")
    server = ms.MirixServer(
        cell,
        command=[sys.executable, str(stub), "--port", str(cell.port)],
        database=False,
        mode="fresh",
        verify_llm_contract=True,
    )
    with pytest.raises(ms.MirixServerError, match="retry/timeout contract"):
        await server.start()
    assert server.process is None  # nothing was spawned
    default = ms.MirixServer(cell, mode="fresh", database=False)
    assert (
        default.verify_llm_contract is True
    )  # the real MIRIX command is always probed


def test_a_sweep_probing_the_cell_lock_does_not_fail_a_starting_server(tmp_path):
    """X23: a scheduler's sweep holds the cell's server lock for a moment to see whether it
    is held; a server starting then waits for it instead of failing on INFRA."""
    import threading

    from evals.skill_bench.infra.processes import OwnerLock

    cell = spec(tmp_path, "probed-cell")
    probe = OwnerLock.acquire(cell.cell_dir / "server.lock")
    timer = threading.Timer(0.3, probe.release)
    timer.start()
    server = ms.MirixServer(cell, mode="fresh", command=[sys.executable, "-c", "pass"])
    try:
        server._acquire_cell_lock()
        assert server._lock is not None and server._lock.held
        with pytest.raises(ms.MirixServerError, match="already started"):
            server._acquire_cell_lock()
    finally:
        timer.join()
        server._release_cell_lock()
    # A lock another live harness keeps is still refused (after the wait).
    owner = OwnerLock.acquire(cell.cell_dir / "server.lock")
    try:
        other = ms.MirixServer(
            cell, mode="fresh", command=[sys.executable, "-c", "pass"]
        )
        with pytest.raises(ms.MirixServerError, match="another live harness"):
            other._acquire_cell_lock()
    finally:
        owner.release()


# --- a relative cell directory ---------------------------------------------------------------


def legacy_mirix_state(attempt_dir: Path, relative_home: str) -> Path:
    """What a server started with a relative HOME wrote before the fix: MIRIX's
    ``expanduser("~")`` returns the relative HOME, resolved against the server's working
    directory, the attempt directory."""
    attempt_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import os\n"
        "state = os.path.join(os.path.expanduser('~'), '.mirix')\n"
        "os.makedirs(os.path.join(state, 'images', 'org'))\n"
        "open(os.path.join(state, 'config'), 'w').write('[defaults]\\n')\n"
        "open(os.path.join(state, 'images', 'org', 'page.png'), 'wb').write(b'png')\n"
        "print(os.path.join(state, 'images', 'org', 'page.png'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=attempt_dir,
        env={"PATH": os.environ["PATH"], "HOME": relative_home},
        capture_output=True,
        text=True,
        check=True,
    )
    image_row = out.stdout.strip()  # the file_path MIRIX would store: relative
    assert not Path(image_row).is_absolute()
    return attempt_dir / relative_home / ".mirix"


@pytest.mark.asyncio
async def test_a_relative_cell_dir_puts_mirix_state_in_the_documented_home(
    tmp_path, monkeypatch
):
    """A relative layout: the harness runs with a relative ``--out`` from another
    working directory than the server's (its attempt directory). HOME, the guards, the
    log and the registry all name one absolute directory, and a resume after a fresh
    start passes the guard."""
    stub = write_stub_server(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative = Path("runs") / "dev" / "webshop-luna-mirix"
    cell = ms.CellServerSpec("webshop-luna-mirix", 1, relative, find_free_port())
    assert cell.cell_dir == Path.cwd() / relative and cell.cell_dir.is_absolute()
    assert cell.home == Path.cwd() / relative / "attempt-01" / "home"
    env = ms.build_server_env(cell, {"PATH": os.environ["PATH"]})
    assert env["HOME"] == str(cell.home) and Path(env["HOME"]).is_absolute()
    assert replace(cell, cell_id="renamed").cell_dir == cell.cell_dir

    async def start(mode: str) -> dict[str, str]:
        spec = ms.CellServerSpec("webshop-luna-mirix", 1, relative, find_free_port())
        server = ms.MirixServer(
            spec,
            command=[sys.executable, str(stub), "--port", str(spec.port)],
            database=False,
            mode=mode,  # type: ignore[arg-type]
        )
        await server.start()
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                return (await client.get(server.base_url + "/env")).json()
        finally:
            await server.stop()

    fresh = await start("fresh")
    assert fresh["HOME"] == str(cell.home)
    assert (cell.home / ".mirix" / "stub_marker").is_file()
    assert not (cell.attempt_dir / "runs").exists()  # nothing nested below the attempt
    resumed = await start("resume")  # the guard sees the state the fresh start wrote
    assert resumed["HOME"] == str(cell.home)
    with pytest.raises(ms.MirixServerError, match="fresh HOME"):
        await start("fresh")
    assert "stub mirix server starting" in cell.log_path.read_text()
    assert cell.registry.is_file()


def test_nested_mirix_state_is_found_and_linked_never_moved(tmp_path, capsys):
    """Attempts that ran before the fix: ``link-nested-home`` gives the documented HOME
    a relative symlink to the nested state, which stays where MIRIX's file rows (paths
    relative to the attempt directory) point."""
    runs = tmp_path / "runs" / "dev"
    cell = ms.CellServerSpec(
        "webshop-luna56-mirix", 1, runs / "webshop-luna56-mirix", 0
    )
    relative_home = "evals/skill_bench/runs/dev/webshop-luna56-mirix/attempt-01/home"
    nested = legacy_mirix_state(cell.attempt_dir, relative_home)
    cell.home.mkdir()  # the harness created the documented HOME; nothing wrote there
    (runs / "webshop-luna56-none" / "attempt-01" / "home").mkdir(
        parents=True
    )  # no MIRIX
    (runs / "_schedule").mkdir()
    assert ms.nested_mirix_dirs(cell.attempt_dir) == [nested]
    assert ms.attempt_dirs(runs) == [
        cell.attempt_dir,
        runs / "webshop-luna56-none" / "attempt-01",
    ]
    assert ms.attempt_dirs(cell.cell_dir) == ms.attempt_dirs(cell.attempt_dir)
    with pytest.raises(ms.MirixServerError, match="link-nested-home") as refused:
        ms.check_home(cell, "resume")
    assert str(nested) in str(refused.value)

    assert ms.main(["link-nested-home", "--dry-run", str(runs)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["attempt"], r["status"]) for r in rows] == [
        (str(cell.attempt_dir), "would_link")
    ]
    assert not cell.mirix_dir.is_symlink()  # a dry run changes nothing

    assert ms.main(["link-nested-home", str(runs)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["status"] == "linked" and row["nested"] == [relative_home + "/.mirix"]
    link = os.readlink(cell.mirix_dir)
    assert link == row["link"] == "../" + relative_home + "/.mirix"  # relative
    assert cell.mirix_dir.resolve() == nested.resolve()
    assert (nested / "config").is_file()  # not moved
    assert (cell.attempt_dir / relative_home / ".mirix/images/org/page.png").is_file()
    assert (cell.mirix_dir / "images" / "org" / "page.png").read_bytes() == b"png"
    ms.check_home(cell, "resume")  # the guard now finds the state
    with pytest.raises(ms.MirixServerError, match="fresh HOME"):
        ms.check_home(cell, "fresh")

    # Idempotent: a second run leaves the link alone and says so.
    assert ms.link_nested_mirix_home(cell.attempt_dir) is None
    assert ms.main(["link-nested-home", str(cell.attempt_dir)]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["status"] == "home_has_mirix" and again["link"] == link


def test_several_nested_mirix_states_are_never_guessed(tmp_path, capsys):
    cell = ms.CellServerSpec("c", 2, tmp_path / "runs" / "c", 0)
    first = legacy_mirix_state(cell.attempt_dir, "runs/c/attempt-02/home")
    second = legacy_mirix_state(cell.attempt_dir, "other/runs/c/attempt-02/home")
    assert ms.nested_mirix_dirs(cell.attempt_dir) == sorted([first, second])
    with pytest.raises(ms.MirixServerError, match="several"):
        ms.link_nested_mirix_home(cell.attempt_dir)
    assert ms.main(["link-nested-home", str(cell.attempt_dir)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "ambiguous"
    assert not cell.mirix_dir.exists() and not cell.mirix_dir.is_symlink()
    with pytest.raises(ms.MirixServerError, match="missing or empty") as refused:
        ms.check_home(cell, "resume")
    assert "link-nested-home" not in str(refused.value)  # no single one to name


def test_a_linked_mirix_state_must_stay_inside_its_attempt(tmp_path):
    """A ``.mirix`` link to another attempt's state, or to a real home's ``~/.mirix``,
    would share or leak MIRIX state across cells (H8): a resume refuses it."""
    other = ms.CellServerSpec("other", 1, tmp_path / "runs" / "other", 0)
    other.mirix_dir.mkdir(parents=True)
    (other.mirix_dir / "config").write_text("[defaults]\n")
    cell = ms.CellServerSpec("cell", 1, tmp_path / "runs" / "cell", 0)
    cell.home.mkdir(parents=True)
    cell.mirix_dir.symlink_to(other.mirix_dir, target_is_directory=True)
    with pytest.raises(ms.MirixServerError, match="never leaves its attempt"):
        ms.check_home(cell, "resume")
    cell.mirix_dir.unlink()
    inside = cell.attempt_dir / "kept" / ".mirix"
    inside.mkdir(parents=True)
    (inside / "config").write_text("[defaults]\n")
    cell.mirix_dir.symlink_to(os.path.relpath(inside, cell.home))
    ms.check_home(cell, "resume")  # a link inside the attempt is fine


def test_a_linked_home_must_stay_inside_its_attempt_too(tmp_path):
    """Not only a linked ``.mirix``: a HOME that is itself a link to another cell's HOME
    would lead a resume onto that cell's MIRIX state, and so could a fresh start. Any
    link on the way counts."""
    other = ms.CellServerSpec("a", 1, tmp_path / "runs" / "a", 0)
    other.mirix_dir.mkdir(parents=True)
    (other.mirix_dir / "config").write_text("[defaults]\n")
    cell = ms.CellServerSpec("b", 1, tmp_path / "runs" / "b", 0)
    cell.attempt_dir.mkdir(parents=True)
    cell.home.symlink_to(os.path.relpath(other.home, cell.attempt_dir))
    assert not cell.mirix_dir.is_symlink() and ms.holds_mirix_state(cell.home)
    for mode in ("resume", "fresh"):
        with pytest.raises(ms.MirixServerError, match="never leaves its attempt"):
            ms.check_home(cell, mode)
    # A HOME linked inside its own attempt is fine.
    cell.home.unlink()
    kept = cell.attempt_dir / "kept-home"
    (kept / ".mirix").mkdir(parents=True)
    (kept / ".mirix" / "config").write_text("[defaults]\n")
    cell.home.symlink_to("kept-home")
    ms.check_home(cell, "resume")


# --- learning checkpoints: HOME -----------------------------------------------------------


def test_home_manifest_capture_restore_and_move_aside(tmp_path):
    home = tmp_path / "home"
    (home / ".mirix" / "images" / "ab").mkdir(parents=True)
    (home / ".mirix" / "tmp").mkdir()
    (home / ".mirix" / "config").write_text("config v1")
    (home / ".mirix" / "images" / "ab" / "page.png").write_bytes(b"png bytes")
    (home / ".mirix" / "tmp" / "scratch").write_text("ignored")
    manifest = ms.home_manifest(home)
    assert set(manifest["files"]) == {".mirix/config", ".mirix/images/ab/page.png"}
    assert manifest["files"][".mirix/images/ab/page.png"]["kind"] == "image"
    assert manifest["files"][".mirix/config"]["kind"] == "file"
    out = tmp_path / "checkpoint"
    ms.capture_home(home, manifest, out)
    image = ".mirix/images/ab/page.png"
    assert (out / "home_files" / image).stat().st_ino == (home / image).stat().st_ino
    assert (out / "home_files" / ".mirix/config").stat().st_ino != (
        home / ".mirix/config"
    ).stat().st_ino
    # An unchanged file keeps its hash without being read again.
    again = ms.home_manifest(home, previous=manifest)
    assert again == manifest

    # After the checkpoint: a new image, a changed config, a new scratch file.
    (home / ".mirix" / "images" / "ab" / "late.png").write_bytes(b"late")
    (home / ".mirix" / "config").write_text("config v2")
    moved = ms.move_new_home_files(home, manifest, tmp_path / "aside")
    assert sorted(moved) == [
        ".mirix/config",
        ".mirix/images/ab/late.png",
        ".mirix/tmp/scratch",
    ]
    assert (tmp_path / "aside" / ".mirix" / "config").read_text() == "config v2"
    assert ms.restore_home(home, out, manifest) == []
    assert (home / ".mirix" / "config").read_text() == "config v1"
    assert not (home / ".mirix" / "images" / "ab" / "late.png").exists()
    assert (
        ms.move_new_home_files(home, manifest, tmp_path / "aside2") == []
    )  # repeatable
    # A changed image (write-once, so never expected) is detected, a missing one restored.
    (home / image).unlink()
    assert ms.restore_home(home, out, manifest) == []
    assert (home / image).read_bytes() == b"png bytes"
    (home / image).unlink()
    (home / image).write_bytes(b"other")
    assert ms.restore_home(home, out, manifest) == [
        f"{image}: sha256 differs from the checkpoint's"
    ]
    # No manifest (a rollback to before the first checkpoint): HOME is emptied.
    everything = ms.move_new_home_files(home, None, tmp_path / "aside3")
    assert ".mirix/config" in everything and not any(home.iterdir())


def test_the_postgres_container_can_be_named(monkeypatch):
    import asyncio

    monkeypatch.setenv(ms.PG_CONTAINER_ENV, "my-postgres")
    assert asyncio.run(ms.postgres_container()) == "my-postgres"


def test_discarded_database_names_stay_postgres_names():
    name = ms.cell_db_name("x" * 200, 12)
    assert ms.discarded_db_name(name, 7) == name + "_d07"
    assert ms.DB_NAME_RE.fullmatch(ms.discarded_db_name(name, 99))
    with pytest.raises(ValueError):
        ms.discarded_db_name(name, 0)


def test_a_server_without_a_database_has_nothing_to_dump(tmp_path):
    import asyncio

    server = ms.MirixServer(spec(tmp_path, "c"), mode="create", database=False)
    store = server.checkpoint_store()
    assert isinstance(store, ms.NoDatabase)
    assert (
        asyncio.run(store.check_owner(claim_unmarked=True, segment=1))["states"] == {}
    )
    summary = asyncio.run(
        store.rollback(1, None, tmp_path / "export.json", claim_unmarked=True)
    )
    assert summary["post_crash_db"] == "none" and summary["retrievals"] == []
    assert json.loads((tmp_path / "export.json").read_text())["rows"] == {}
    real = ms.MirixServer(spec(tmp_path, "d"), mode="create")
    assert isinstance(real.checkpoint_store(), ms.CellDatabase)
    assert real.checkpoint_store().name == real.spec.db_name


# --- cell database identity and ownership (B02) --------------------------------------------------


def test_database_names_bind_the_cell_directory_instance(tmp_path):
    """Two runs directories with one cell id (runs/drill and runs/drill2) must not
    compute one database name, or the second one's resume would drop the first one's
    database (B02). A name carries the cell directory's instance id, as its gateway run
    ids do."""
    drill = ms.cell_db_name(DRILL_CELL, 1, "164c9dcf")
    drill2 = ms.cell_db_name(DRILL_CELL, 1, "0a1b2c3d")
    assert drill != drill2
    assert drill == "sb_webshop_luna_mirix_dev_l8_t2_a1_i164c9dcf"
    # Without an instance the name has the legacy shape, byte for byte.
    assert (
        ms.cell_db_name("webshop-luna-mirix", 1) == "sb_webshop_luna_mirix_a1_fd022b49"
    )
    assert (
        ms.cell_db_name("alfworld-luna-mirix", 1)
        == "sb_alfworld_luna_mirix_a1_938b2ade"
    )
    # The new shape never looks like the old one ("i" is no hex digit), even for an
    # instance id that is one of the old names' digests.
    legacy = re.compile(r"_a\d+_[0-9a-f]{8}$")
    assert ms.cell_db_name("webshop-luna-mirix", 1, "fd022b49") != ms.cell_db_name(
        "webshop-luna-mirix", 1
    )
    for cell_id, attempt, instance in (
        ("x" * 200, 999, "z" * 12),
        ("webshop-luna-mirix", 1, "fd022b49"),
        ("a", 0, "0"),
    ):
        name = ms.cell_db_name(cell_id, attempt, instance)
        assert not legacy.search(name), name
        # Room for a rollback's <db>_dNN aside, whatever the cell id and attempt.
        assert ms.DB_NAME_RE.fullmatch(ms.discarded_db_name(name, 99)), name
    for bad in ("", "UPPER", "a-b", "x" * 13, "../x"):
        with pytest.raises(ValueError):
            ms.cell_db_name("c", 1, bad)

    bound = spec(
        tmp_path, DRILL_CELL, attempt=1, instance="164c9dcf", owner=owner().run_id
    )
    assert bound.db_name == drill and bound.name_bound_to_instance
    assert bound.database_owner == owner()
    assert bound.pg_uri.endswith("/" + drill)
    # A recorded name always wins: legacy names, and the test seams that
    # tag the cell id (replace() keeps the recorded name).
    recorded = replace(bound, database="sb_webshop_luna_mirix_a1_fd022b49")
    assert recorded.db_name == "sb_webshop_luna_mirix_a1_fd022b49"
    assert not recorded.name_bound_to_instance
    assert replace(recorded, cell_id="tagged").db_name == recorded.db_name
    # A recorded name that is the instance's own is still bound to the instance.
    assert replace(bound, database=drill).name_bound_to_instance
    tagged = replace(bound, cell_id=f"{DRILL_CELL}-e2e1")
    assert tagged.db_name == ms.cell_db_name(f"{DRILL_CELL}-e2e1", 1, "164c9dcf")
    for bad_name in ("postgres", "skill_bench_template", 'sb_x"; drop'):
        with pytest.raises(ValueError):
            spec(tmp_path, "c", database=bad_name)
    with pytest.raises(ValueError):
        spec(tmp_path, "c", instance="NOT-AN-ID")
    plain = spec(tmp_path, "c")  # the method's own server (development)
    assert plain.db_name == ms.cell_db_name("c", 0)
    assert plain.database_owner is None and not plain.name_bound_to_instance


@pytest.mark.asyncio
async def test_a_fresh_attempt_refuses_an_existing_database_as_a_conflict():
    """A new learning attempt that finds its database already there was INFRA
    ``mirix_server`` (a transient outage), so the next invocation resumed it and the
    resume dropped the database it found (B02). It is a HARNESS conflict now: a person
    decides, nothing resumes it."""
    from evals.skill_bench import run
    from evals.skill_bench.infra.failures import FailureKind, classify_exception

    pg = ms.PostgresAdmin()
    for comments in ({}, {SHARED_LEGACY_NAME: owner("0a1b2c3d").comment()}):
        log: list[tuple[str, str]] = []
        existing = {SHARED_LEGACY_NAME}
        with pytest.raises(ms.CellDatabaseConflict, match="already exists") as refused:
            await ms.ensure_cell_database(
                pg,
                SHARED_LEGACY_NAME,
                must_be_new=True,
                owner=owner(),
                connect=fake_connect(log, existing, dict(comments)),
            )
        assert isinstance(refused.value, ms.MirixServerError)
        assert ("postgres", "SELECT pg_advisory_unlock($1)") in log
        assert changes(log) == []
        failure = classify_exception(refused.value)
        assert (failure.kind, failure.code) == (
            FailureKind.HARNESS,
            "mirix_database_conflict",
        )
        assert run.learning_recovery(failure) != run.RECOVERY_RESUME_FROM_CHECKPOINT
    # The refusal names the run that owns the database.
    assert owner("0a1b2c3d").run_id in str(refused.value)
    # Any other MIRIX server failure stays INFRA mirix_server.
    other = classify_exception(ms.MirixServerError("port 1 is already in use"))
    assert (other.kind, other.code) == (FailureKind.INFRA, "mirix_server")


@pytest.mark.asyncio
async def test_a_new_database_is_marked_with_its_owner_under_the_lock():
    pg = ms.PostgresAdmin()
    mine = owner()
    name = ms.cell_db_name(DRILL_CELL, 1, mine.instance)
    log: list[tuple[str, str]] = []
    existing: set[str] = set()
    comments: dict[str, str] = {}
    connect = fake_connect(log, existing, comments)
    created = await ms.ensure_cell_database(
        pg, name, must_be_new=True, owner=mine, connect=connect
    )
    assert created is True
    admin = [sql for db, sql in log if db == "postgres"]
    create = admin.index(f'CREATE DATABASE "{name}" TEMPLATE "skill_bench_template"')
    comment = next(i for i, s in enumerate(admin) if s.startswith("COMMENT"))
    unlock = admin.index("SELECT pg_advisory_unlock($1)")
    assert create < comment < unlock
    # Postgres quotes the marker itself (format %I / %L); the harness never does.
    assert admin[comment - 1].startswith("SELECT format(")
    assert json.loads(comments[name]) == {"skill_bench_owner": mine.marker()}
    assert mine.marker() == {
        "run_id": mine.run_id,
        "cell_id": DRILL_CELL,
        "attempt": 1,
        "instance": "164c9dcf",
    }
    assert await ms.database_owners(pg, name, mine, connect=connect) == {
        name: ms.OWNER_OURS
    }
    assert await ms.database_owners(pg, name, owner("0a1b2c3d"), connect=connect) == {
        name: ms.OWNER_FOREIGN
    }
    assert await ms.database_owners(pg, name + "x", mine, connect=connect) == {}
    # A server started without an owner (tests, development) marks nothing.
    log.clear()
    plain = ms.cell_db_name("c", 0)
    assert await ms.ensure_cell_database(pg, plain, connect=connect) is True
    assert plain not in comments and not any(
        s.startswith("COMMENT") for s in changes(log)
    )
    assert await ms.database_owners(pg, plain, mine, connect=connect) == {
        plain: ms.OWNER_UNMARKED
    }
    # A marker with quotes, a backslash and non-ASCII text survives Postgres's quoting.
    odd = ms.DatabaseOwner("r'un\\1", "cell 'é'", 2, None)
    await ms.mark_database(pg, plain, odd, connect=connect)
    assert json.loads(comments[plain]) == {"skill_bench_owner": odd.marker()}
    assert await ms.database_owners(pg, plain, odd, connect=connect) == {
        plain: ms.OWNER_OURS
    }


@pytest.mark.asyncio
async def test_resume_refuses_a_database_another_run_marked_and_accepts_an_unmarked_one(
    tmp_path,
):
    pg = ms.PostgresAdmin()
    mine, other = owner(), owner("0a1b2c3d")
    name = SHARED_LEGACY_NAME  # a legacy name an older attempt recorded (no marker yet)
    log: list[tuple[str, str]] = []
    existing = {name}
    comments: dict[str, str] = {}
    connect = fake_connect(log, existing, comments)
    # A database from before owner markers: a test-phase resume goes on, unchanged.
    assert (
        await ms.ensure_cell_database(
            pg, name, must_exist=True, owner=mine, connect=connect
        )
        is False
    )
    assert name not in comments and changes(log) == []
    comments[name] = mine.comment()
    assert (
        await ms.ensure_cell_database(
            pg, name, must_exist=True, owner=mine, connect=connect
        )
        is False
    )
    # Another run's database is refused in every mode, and so is one whose comment is
    # no skill_bench marker (its owner cannot be proven).
    for comment, match in ((other.comment(), other.run_id), ("made by hand", "marker")):
        comments[name] = comment
        for mode in ({"must_exist": True}, {}):
            with pytest.raises(ms.CellDatabaseConflict, match=match):
                await ms.ensure_cell_database(
                    pg, name, owner=mine, connect=connect, **mode
                )
    assert comments[name] == "made by hand" and changes(log) == []

    # Through MirixServer: the spec's owner reaches the check, before any spawn.
    comments[name] = other.comment()
    cell = spec(
        tmp_path,
        DRILL_CELL,
        attempt=1,
        instance=mine.instance,
        owner=mine.run_id,
        database=name,
    )
    (cell.home / ".mirix").mkdir(parents=True)
    (cell.home / ".mirix" / "config").write_text("[defaults]\n")
    server = ms.MirixServer(
        cell, command=[sys.executable, "-c", "pass"], mode="resume", connect=connect
    )
    with pytest.raises(ms.CellDatabaseConflict, match=other.run_id):
        await server.start()
    assert not cell.registry.exists()  # no server was spawned
    del comments[name]
    server = ms.MirixServer(
        cell, command=[sys.executable, "-c", "pass"], mode="resume", connect=connect
    )
    with pytest.raises(ms.MirixServerError, match="exited") as started:
        await server.start()  # the database passed; the stand-in exits at once
    assert not isinstance(started.value, ms.CellDatabaseConflict)
    assert name not in comments  # a resume never marks a pre-fix database


@pytest.mark.asyncio
async def test_rollback_never_renames_or_drops_a_database_another_run_owns(tmp_path):
    pg = ms.PostgresAdmin()
    mine, other = owner(), owner("0a1b2c3d")
    name = SHARED_LEGACY_NAME
    aside = ms.discarded_db_name(name, 1)
    for present in (
        {name: other.comment()},  # drill's database, found by drill2's resume
        {name: mine.comment(), aside: other.comment()},  # another run's kept aside
    ):
        log: list[tuple[str, str]] = []
        existing = set(present)
        comments = dict(present)
        store = ms.CellDatabase(
            pg, name, owner=mine, connect=fake_connect(log, existing, comments)
        )
        with pytest.raises(ms.CellDatabaseConflict, match=other.run_id):
            await store.check_owner(segment=1)
        # Not even a person's claim (--force-resume) takes another run's database.
        export = tmp_path / "seg-01" / ms.EXPORT_FILE
        with pytest.raises(ms.CellDatabaseConflict):
            await store.rollback(1, None, export, claim_unmarked=True)
        assert changes(log) == []
        assert existing == set(present) and comments == present
        assert not export.exists()


@pytest.mark.asyncio
async def test_rollback_of_an_unmarked_database_needs_a_claim(tmp_path, monkeypatch):
    exported: list[tuple[str, str | None]] = []

    async def export_rows(
        pg: Any, discarded_db: str, live_db: str | None, **kw: Any
    ) -> dict[str, Any]:
        exported.append((discarded_db, live_db))
        return {
            "discarded_db": discarded_db,
            "live_db": live_db,
            "tables": {},
            "rows": {},
        }

    monkeypatch.setattr(ms, "export_discarded_rows", export_rows)
    pg = ms.PostgresAdmin()
    mine = owner()
    name = SHARED_LEGACY_NAME
    aside = ms.discarded_db_name(name, 1)
    log: list[tuple[str, str]] = []
    existing = {name}
    comments: dict[str, str] = {}
    connect = fake_connect(log, existing, comments)
    store = ms.CellDatabase(pg, name, owner=mine, connect=connect)
    # Unmarked under a name another runs directory may compute: never touched unasked.
    with pytest.raises(ms.CellDatabaseConflict, match="--force-resume"):
        await store.check_owner(segment=1)
    with pytest.raises(ms.CellDatabaseConflict):
        await store.rollback(1, None, tmp_path / "e0.json")
    assert changes(log) == [] and existing == {name}
    # A person's claim: the database is marked ours first, then moved aside, exported
    # and dropped.
    summary = await store.rollback(1, None, tmp_path / "e1.json", claim_unmarked=True)
    assert changes(log) == [
        f"COMMENT ON DATABASE {name} IS {_pg_literal(mine.comment())}",
        f'ALTER DATABASE "{name}" RENAME TO "{aside}"',
        f'DROP DATABASE IF EXISTS "{aside}" WITH (FORCE)',
    ]
    assert exported == [(aside, None)] and existing == set()
    assert summary["post_crash_db"] == "dropped"
    assert summary["owner"] == {
        "database": name,
        "owner": mine.run_id,
        "states": {name: ms.OWNER_UNMARKED},
        "claimed": [name],
    }

    # A name bound to the attempt's own cell directory: nobody else computes it, so an
    # unmarked one (a crash between CREATE and COMMENT) is the attempt's own.
    bound = spec(
        tmp_path, DRILL_CELL, attempt=1, instance=mine.instance, owner=mine.run_id
    )
    existing.add(bound.db_name)
    server = ms.MirixServer(bound, mode="create", connect=connect)
    bound_store = server.checkpoint_store()
    assert isinstance(bound_store, ms.CellDatabase)
    assert bound_store.claim_unmarked and bound_store.owner == mine
    summary = await bound_store.rollback(1, None, tmp_path / "e2.json")
    assert summary["post_crash_db"] == "dropped" and bound.db_name not in existing
    recorded = ms.MirixServer(
        replace(bound, database=name), mode="create", connect=connect
    ).checkpoint_store()
    assert isinstance(recorded, ms.CellDatabase) and not recorded.claim_unmarked

    # A live database next to this segment's aside marked ours is the rollback's own
    # half-made restore (a crash between CREATE and COMMENT): accepted unclaimed.
    log.clear()
    existing.update({name, aside})
    comments[aside] = mine.comment()
    summary = await store.rollback(1, None, tmp_path / "e3.json")
    assert changes(log) == [
        f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)',
        f'DROP DATABASE IF EXISTS "{aside}" WITH (FORCE)',
    ]
    assert summary["owner"]["claimed"] == [] and existing == set()
    # ... but only next to its own segment's aside.
    existing.update({name, aside})
    comments[aside] = mine.comment()
    with pytest.raises(ms.CellDatabaseConflict):
        await store.check_owner(segment=2)


@pytest.mark.asyncio
async def test_an_unmarked_aside_of_the_segment_next_to_the_live_database_is_never_claimed(
    tmp_path, monkeypatch
):
    """The drill/drill2 shape with --keep-discarded-db: the live database and an
    aside of the same segment (another runs directory's kept copy) carry no marker.
    A --force-resume that claimed both would let the rollback take the live database
    for its own half-made restore, drop it WITHOUT exporting it, export the other
    directory's aside instead and drop that too."""
    exported: list[str] = []

    async def export_rows(pg: Any, discarded_db: str, live_db: Any, **kw: Any) -> Any:
        exported.append(discarded_db)
        return {
            "discarded_db": discarded_db,
            "live_db": live_db,
            "tables": {},
            "rows": {},
        }

    monkeypatch.setattr(ms, "export_discarded_rows", export_rows)
    pg = ms.PostgresAdmin()
    mine = owner()
    name = SHARED_LEGACY_NAME
    aside = ms.discarded_db_name(name, 1)
    for present in ({}, {name: mine.comment()}):
        log: list[tuple[str, str]] = []
        existing = {name, aside}
        comments = dict(present)
        store = ms.CellDatabase(
            pg, name, owner=mine, connect=fake_connect(log, existing, comments)
        )
        with pytest.raises(ms.CellDatabaseConflict) as refused:
            await store.rollback(1, None, tmp_path / "e.json", claim_unmarked=True)
        text = str(refused.value)
        assert f"{aside} carries no owner marker" in text
        assert "drop it without exporting it" in text
        assert "Not even --force-resume claims it" in text
        assert changes(log) == [] and existing == {name, aside}
        assert comments == present and exported == []
        # A store whose name is bound to its cell directory claims unmarked
        # databases on its own: this one it refuses too.
        bound = ms.CellDatabase(
            pg,
            name,
            owner=mine,
            claim_unmarked=True,
            connect=fake_connect(log, existing, comments),
        )
        with pytest.raises(ms.CellDatabaseConflict):
            await bound.check_owner(segment=1)
    # Without a live database the unmarked aside is the post-crash database a
    # rollback left (a crash between its rename and its restore): a claim takes it.
    log = []
    existing = {aside}
    comments = {}
    store = ms.CellDatabase(
        pg, name, owner=mine, connect=fake_connect(log, existing, comments)
    )
    summary = await store.rollback(1, None, tmp_path / "e2.json", claim_unmarked=True)
    assert exported == [aside] and summary["owner"]["claimed"] == [aside]


@pytest.mark.asyncio
async def test_restore_re_marks_the_restored_database(tmp_path, monkeypatch):
    """``pg_dump`` without ``--create`` carries no database comment, so a restored
    database is marked again right after it is created, before ``pg_restore``."""
    log: list[tuple[str, str]] = []

    async def docker(*args: str, **kw: Any) -> tuple[int, bytes, bytes]:
        log.append(("docker", " ".join(a for a in args if not a.startswith("PGPASS"))))
        return 0, b"", b""

    async def export_rows(pg: Any, discarded_db: str, live_db: Any, **kw: Any) -> Any:
        return {
            "discarded_db": discarded_db,
            "live_db": live_db,
            "tables": {},
            "rows": {},
        }

    monkeypatch.setattr(ms, "_docker", docker)
    monkeypatch.setattr(ms, "export_discarded_rows", export_rows)
    pg = ms.PostgresAdmin()
    mine = owner()
    name = ms.cell_db_name(DRILL_CELL, 1, mine.instance)
    existing: set[str] = set()
    comments: dict[str, str] = {}
    connect = fake_connect(log, existing, comments)
    dump = tmp_path / "c0001-done" / ms.DUMP_FILE
    dump.parent.mkdir()
    dump.write_bytes(b"PGDMP")
    await ms.restore_database(
        pg, name, dump, container="pg", owner=mine, connect=connect
    )
    steps = [sql for _, sql in log if not sql.startswith(("SELECT 1", "CLOSE"))]
    create = steps.index(f'CREATE DATABASE "{name}" TEMPLATE template0')
    comment = next(i for i, s in enumerate(steps) if s.startswith("COMMENT"))
    restore = next(i for i, s in enumerate(steps) if "pg_restore" in s)
    assert create < comment < restore
    assert json.loads(comments[name]) == {"skill_bench_owner": mine.marker()}

    # Through a rollback: the restored database is the attempt's again.
    store = ms.CellDatabase(pg, name, owner=mine, container="pg", connect=connect)
    summary = await store.rollback(1, dump.parent, tmp_path / "export.json")
    assert summary["post_crash_db"] == "dropped"
    assert await ms.database_owners(pg, name, mine, asides=True, connect=connect) == {
        name: ms.OWNER_OURS
    }


#: The host ``mark_owners`` runs on in these tests (``cell_records.host_info``'s shape).
HERE = {
    "hostname": "lab-mac",
    "machine_id": "5f1c0e2a-host-here",
    "system": "Darwin",
    "machine": "arm64",
}
#: Another machine, whose runs directory was synced here.
ELSEWHERE = {**HERE, "hostname": "lab-linux", "machine_id": "77aa-host-there"}


def _attempt(
    root: Path,
    cell_id: str,
    instance: str,
    db_name: str | None,
    number: int = 1,
    host: Mapping[str, Any] | None = HERE,
) -> Path:
    path = root / cell_id / f"attempt-{number:02d}"
    path.mkdir(parents=True)
    state: dict[str, Any] = {
        "attempt": number,
        "run_id": f"{cell_id}-a{number:02d}-{instance}",
        "instance": instance,
    }
    if host is not None:
        state["host"] = dict(host)
    if db_name is not None:
        state["mirix"] = {"db_name": db_name}
    (path / "attempt.json").write_text(json.dumps(state))
    return path


@pytest.mark.asyncio
async def test_mark_owners_stamps_only_databases_one_attempt_records(tmp_path):
    """The operator's tool for pre-fix databases: a database is marked with the one
    attempt that records its name; a name two runs directories record (the drill /
    drill2 shape) or one another run marked is left for a person, and nothing is changed
    with ``--dry-run``."""
    other = owner("0a1b2c3d")
    single = ms.cell_db_name("webshop-luna-mirix", 1)
    marked = ms.cell_db_name("alfworld-luna-mirix", 1)
    foreign = ms.cell_db_name("demo-luna-mirix", 1)
    gone = ms.cell_db_name("demo-luna56-mirix", 1)
    drill, drill2, phase = tmp_path / "drill", tmp_path / "drill2", tmp_path / "earlier"
    _attempt(drill, DRILL_CELL, "aaaa0001", SHARED_LEGACY_NAME)
    _attempt(drill2, DRILL_CELL, "bbbb0002", SHARED_LEGACY_NAME)
    only = _attempt(phase, "webshop-luna-mirix", "164c9dcf", single)
    _attempt(phase, "alfworld-luna-mirix", "164c9dcf", marked)
    _attempt(phase, "demo-luna-mirix", "164c9dcf", foreign)
    _attempt(phase, "demo-luna56-mirix", "164c9dcf", gone)
    _attempt(phase, "webshop-luna-awm", "164c9dcf", None)  # no MIRIX database
    ours_marker = ms.DatabaseOwner(
        "alfworld-luna-mirix-a01-164c9dcf", "alfworld-luna-mirix", 1, "164c9dcf"
    )
    log: list[tuple[str, str]] = []
    existing = {SHARED_LEGACY_NAME, single, marked, foreign}
    comments = {marked: ours_marker.comment(), foreign: other.comment()}
    connect = fake_connect(log, existing, comments)
    dry = await ms.mark_owners(
        [drill, drill2, phase], dry_run=True, host=HERE, connect=connect
    )
    assert changes(log) == []
    status = {row["db_name"]: row["status"] for row in dry}
    assert status == {
        SHARED_LEGACY_NAME: "ambiguous",
        single: "would_mark",
        marked: "already_marked",
        foreign: "foreign",
        gone: "absent",
    }
    rows = await ms.mark_owners([drill, drill2, phase], host=HERE, connect=connect)
    assert {row["db_name"]: row["status"] for row in rows} == {
        **status,
        single: "marked",
    }
    assert json.loads(comments[single]) == {
        "skill_bench_owner": {
            "run_id": "webshop-luna-mirix-a01-164c9dcf",
            "cell_id": "webshop-luna-mirix",
            "attempt": 1,
            "instance": "164c9dcf",
        }
    }
    assert SHARED_LEGACY_NAME not in comments and comments[foreign] == other.comment()
    (row,) = [r for r in rows if r["db_name"] == single]
    assert row["attempt"] == str(only)
    (ambiguous,) = [r for r in rows if r["db_name"] == SHARED_LEGACY_NAME]
    assert len(ambiguous["attempts"]) == 2


@pytest.mark.asyncio
async def test_mark_owners_marks_the_kept_asides_and_leaves_other_hosts_alone(tmp_path):
    """H5: a rollback run with --keep-discarded-db before owner markers left
    ``<db>_dNN`` asides without a marker, and the next resume refuses them; mark-owners
    marks them with the attempt that records the live name. An attempt that ran on
    another host (a runs directory synced from another machine) or that recorded no
    host is never marked on this Postgres: the name here may be a local attempt's."""
    single = ms.cell_db_name("webshop-luna-mirix", 1)
    aside1, aside2 = ms.discarded_db_name(single, 1), ms.discarded_db_name(single, 2)
    remote = ms.cell_db_name("alfworld-luna-mirix", 1)
    unrecorded = ms.cell_db_name("demo-luna-mirix", 1)
    phase, synced = tmp_path / "earlier", tmp_path / "synced"
    _attempt(phase, "webshop-luna-mirix", "164c9dcf", single)
    _attempt(synced, "alfworld-luna-mirix", "0a1b2c3d", remote, host=ELSEWHERE)
    _attempt(synced, "demo-luna-mirix", "0a1b2c3d", unrecorded, host=None)
    other = owner("0a1b2c3d")
    log: list[tuple[str, str]] = []
    existing = {single, aside1, aside2, remote, unrecorded}
    comments = {aside2: other.comment()}
    connect = fake_connect(log, existing, comments)
    dry = await ms.mark_owners(
        [phase, synced], dry_run=True, host=HERE, connect=connect
    )
    assert changes(log) == []
    assert {(r["db_name"], r["status"], r.get("aside_of")) for r in dry} == {
        (single, "would_mark", None),
        (aside1, "would_mark", single),
        (aside2, "foreign", single),
        (remote, "other_host", None),
        (unrecorded, "other_host", None),
    }
    (elsewhere,) = [r for r in dry if r["db_name"] == remote]
    assert "lab-linux" in elsewhere["found"]
    rows = await ms.mark_owners([phase, synced], host=HERE, connect=connect)
    assert {r["db_name"]: r["status"] for r in rows} == {
        single: "marked",
        aside1: "marked",
        aside2: "foreign",
        remote: "other_host",
        unrecorded: "other_host",
    }
    run_id = "webshop-luna-mirix-a01-164c9dcf"
    assert ms.marker_of(comments[aside1])["run_id"] == run_id
    assert ms.marker_of(comments[single])["run_id"] == run_id
    assert comments[aside2] == other.comment()
    assert remote not in comments and unrecorded not in comments
    # The marked aside is what the resume's owner check now accepts.
    mine = ms.DatabaseOwner(run_id, "webshop-luna-mirix", 1, "164c9dcf")
    states = await ms.database_owners(
        ms.PostgresAdmin(), single, mine, asides=True, connect=connect
    )
    assert states == {
        single: ms.OWNER_OURS,
        aside1: ms.OWNER_OURS,
        aside2: ms.OWNER_FOREIGN,
    }
    # The CLI exits 1 while a name is left for a person.
    assert any(r["status"] == "other_host" for r in rows)


@pytest.mark.asyncio
async def test_mark_owners_leaves_an_aside_another_attempt_records_to_its_own_row(
    tmp_path,
):
    """A name that matches another recorded name's aside pattern is that attempt's
    own database, never marked as the other's aside."""
    base = "sb_cell_a1_ixyz"
    lookalike = ms.discarded_db_name(base, 1)
    root = tmp_path / "runs"
    _attempt(root, "cell", "xyz", base)
    _attempt(root, "cell-two", "abc", lookalike)
    log: list[tuple[str, str]] = []
    existing = {base, lookalike}
    comments: dict[str, str] = {}
    rows = await ms.mark_owners(
        [root], host=HERE, connect=fake_connect(log, existing, comments)
    )
    assert sorted((r["db_name"], r["run_id"], r.get("aside_of")) for r in rows) == [
        (base, "cell-a01-xyz", None),
        (lookalike, "cell-two-a01-abc", None),
    ]
    assert ms.marker_of(comments[lookalike])["run_id"] == "cell-two-a01-abc"


@pytest.mark.asyncio
async def test_unmark_owner_removes_only_the_named_runs_marker():
    """A wrong mark-owners is undone with ``unmark-owner``: the marker goes only when it
    names the given run; another run's marker, a hand-written comment, a database
    without one and a missing database are left as they are."""
    mine, other = owner(), owner("0a1b2c3d")
    name = ms.cell_db_name(DRILL_CELL, 1, mine.instance)
    theirs = ms.cell_db_name(DRILL_CELL, 1, other.instance)
    by_hand, bare = "sb_by_hand_a1_ihand", "sb_bare_a1_ibare"
    log: list[tuple[str, str]] = []
    existing = {name, theirs, by_hand, bare}
    comments = {
        name: mine.comment(),
        theirs: other.comment(),
        by_hand: "made by hand",
    }
    connect = fake_connect(log, existing, comments)
    pg = ms.PostgresAdmin()
    dry = await ms.unmark_owner(name, mine.run_id, dry_run=True, pg=pg, connect=connect)
    assert dry["status"] == "would_unmark" and changes(log) == []
    for db, status in (
        (theirs, "foreign"),
        (by_hand, "foreign"),
        (bare, "unmarked_already"),
        ("sb_missing_a1_inone", "absent"),
    ):
        row = await ms.unmark_owner(db, mine.run_id, pg=pg, connect=connect)
        assert row["status"] == status, db
    assert changes(log) == []
    row = await ms.unmark_owner(name, mine.run_id, pg=pg, connect=connect)
    assert row["status"] == "unmarked"
    assert name not in comments
    assert comments[theirs] == other.comment() and comments[by_hand] == "made by hand"
    assert changes(log) == [f"COMMENT ON DATABASE {name} IS NULL"]


def test_unmark_owner_cli_refuses_a_name_that_is_no_cell_database(capsys):
    with pytest.raises(SystemExit) as exited:
        ms.main(["unmark-owner", 'x"; DROP DATABASE y', "--run-id", "r"])
    assert exited.value.code == 2
    assert "not a cell database name" in capsys.readouterr().err


# --- learning checkpoints: the cell database (integration: the compose Postgres) ---------------


@pytest.fixture
def pg() -> ms.PostgresAdmin:
    import socket

    admin = ms.PostgresAdmin()
    try:
        with socket.create_connection((admin.host, admin.port), timeout=1):
            pass
    except OSError:
        pytest.skip(f"skill_bench Postgres not reachable on {admin.host}:{admin.port}")
    return admin


async def _sql(pg: ms.PostgresAdmin, database: str, *statements: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=database,
    )
    try:
        for statement in statements:
            await conn.execute(statement)
    finally:
        await conn.close()


SCHEMA = (
    "CREATE TABLE retrieval_log (id text PRIMARY KEY, session_id text, "
    "extraction_status text, embedding_status text, per_type jsonb, "
    "created_at timestamptz DEFAULT now())",
    "CREATE TABLE skill_learning_run (id text PRIMARY KEY, status text, trigger text, "
    "session_ids jsonb, started_at timestamptz, finished_at timestamptz, error text)",
    "CREATE TABLE conversation_message (id text PRIMARY KEY, session_id text, "
    "role text, content text)",
    "CREATE TABLE plain (n int)",  # no id column: counted by the digest only
    "INSERT INTO retrieval_log (id, session_id, extraction_status, embedding_status, "
    "per_type) VALUES ('r1', 's1', 'ok', 'ok', '{\"procedural\": 3}')",
    "INSERT INTO skill_learning_run VALUES ('run1', 'succeeded', 'threshold', "
    "'[\"s1\"]', now(), now(), NULL)",
    "INSERT INTO conversation_message VALUES ('m1', 's1', 'user', 'hello')",
    "INSERT INTO plain VALUES (1), (2)",
)


#: The attempt the integration tests' cell databases belong to (B02: a rollback touches
#: only databases marked as the attempt's).
IT_OWNER = ms.DatabaseOwner("it-ckpt-a01-0000feed", "it-ckpt", 1, "0000feed")


@pytest_asyncio.fixture
async def cell_db(pg):
    import secrets

    name = ms.cell_db_name(f"it-ckpt-{secrets.token_hex(3)}", 1)
    await ms.ensure_cell_database(pg, name, must_be_new=True, owner=IT_OWNER)
    await _sql(pg, name, *SCHEMA)
    try:
        yield name
    finally:
        for candidate in (name, *(ms.discarded_db_name(name, s) for s in range(1, 4))):
            await ms.drop_cell_database(pg, candidate)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_compose_postgres_container_is_found_by_its_labels(pg):
    container = await ms.postgres_container()
    proc = subprocess.run(
        ["docker", "inspect", "--format", "{{.Name}}", container],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "mirix-skill-bench" in proc.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_dump_restores_to_the_digest_it_was_taken_with(pg, cell_db, tmp_path):
    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    await store.check_available()
    out = tmp_path / "c0001-done"
    out.mkdir()
    taken = await store.checkpoint(out)
    assert taken["active_runs"] == 0 and taken["db_name"] == cell_db
    assert taken["dump_bytes"] == (out / ms.DUMP_FILE).stat().st_size > 0
    assert taken["db_digest"] == await store.digest()
    assert taken["db_digest"]["retrieval_log"][0] == 1
    # The post-crash database: rows added, changed and removed after the checkpoint.
    await _sql(
        pg,
        cell_db,
        "INSERT INTO retrieval_log (id, session_id, extraction_status, "
        "embedding_status) VALUES ('r2', 's2', 'fallback', 'failed')",
        "INSERT INTO skill_learning_run VALUES ('run2', 'running', 'sweep', "
        "'[\"s2\"]', now(), NULL, NULL)",
        "INSERT INTO conversation_message VALUES ('m2', 's2', 'user', 'wörld')",
        "UPDATE conversation_message SET content = 'changed' WHERE id = 'm1'",
        "DELETE FROM plain WHERE n = 2",
    )
    assert await store.active_runs() == 1
    assert await store.digest() != taken["db_digest"]
    export_path = tmp_path / "seg-01" / ms.EXPORT_FILE
    summary = await store.rollback(1, out, export_path)
    assert await store.digest() == taken["db_digest"]
    assert await store.active_runs() == 0
    assert summary["post_crash_db"] == "dropped"
    assert not await ms.database_exists(pg, ms.discarded_db_name(cell_db, 1))
    assert [r["id"] for r in summary["retrievals"]] == ["r2"]
    assert summary["retrievals"][0]["extraction_status"] == "fallback"
    assert [(r["id"], r["status"]) for r in summary["learning_runs"]] == [
        ("run2", "running")
    ]
    assert summary["changed"] == {"conversation_message": ["m1"]}
    export = json.loads(export_path.read_text())
    message = export["rows"]["conversation_message"][0]
    assert message["id"] == "m2" and message["content_bytes"] == len("wörld".encode())
    assert "content" not in message and len(message["content_sha256"]) == 64
    assert export["tables"]["retrieval_log"] == {
        "added": 1,
        "removed": 0,
        "changed": [],
    }
    # The restored database is the attempt's again (pg_dump carries no comment).
    assert await ms.database_owners(pg, cell_db, IT_OWNER, asides=True) == {
        cell_db: ms.OWNER_OURS
    }
    # Repeating the finished rollback changes nothing and keeps its export.
    again = await store.rollback(1, out, export_path)
    assert again["retrievals"] == summary["retrievals"]
    assert await store.digest() == taken["db_digest"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_rollback_interrupted_after_the_rename_is_finished(
    pg, cell_db, tmp_path
):
    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    out = tmp_path / "c"
    out.mkdir()
    taken = await store.checkpoint(out)
    await _sql(pg, cell_db, "INSERT INTO plain VALUES (3)")
    aside = ms.discarded_db_name(cell_db, 2)
    await ms.rename_database(pg, cell_db, aside)  # the dead resume got this far
    summary = await store.rollback(
        2, out, tmp_path / "export.json", keep_discarded=True
    )
    assert await store.digest() == taken["db_digest"]
    assert summary["post_crash_db"] == f"kept:{aside}"
    assert await ms.database_exists(pg, aside)
    # A restore done twice: the live database is dropped and restored again.
    await _sql(pg, cell_db, "INSERT INTO plain VALUES (4)")
    await store.rollback(2, out, tmp_path / "export2.json")
    assert await store.digest() == taken["db_digest"]
    assert not await ms.database_exists(pg, aside)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_rollback_to_the_empty_state_drops_the_database(pg, cell_db, tmp_path):
    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    summary = await store.rollback(1, None, tmp_path / "export.json")
    assert not await ms.database_exists(pg, cell_db)
    assert [r["id"] for r in summary["retrievals"]] == [
        "r1"
    ]  # everything was discarded
    assert summary["post_crash_db"] == "dropped"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_dump_and_the_digest_share_one_snapshot_under_a_writer(
    pg, cell_db, tmp_path
):
    """A writer inserting rows while the checkpoint runs: the digest recorded with the
    dump describes exactly the dumped rows (both read the exported snapshot)."""
    import asyncio

    import asyncpg

    stop = asyncio.Event()

    async def writer() -> int:
        conn = await asyncpg.connect(
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            database=cell_db,
        )
        n = 0
        try:
            while not stop.is_set():
                n += 1
                await conn.execute(
                    "INSERT INTO conversation_message VALUES ($1, 's9', 'user', 'x')",
                    f"w{n}",
                )
                await asyncio.sleep(0.001)
        finally:
            await conn.close()
        return n

    task = asyncio.ensure_future(writer())
    await asyncio.sleep(0.2)
    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    out = tmp_path / "c"
    out.mkdir()
    taken = await store.checkpoint(out)
    stop.set()
    written = await task
    assert written > taken["db_digest"]["conversation_message"][0] - 1 > 0
    # Restore the dump elsewhere: its digest is the recorded one.
    other = ms.discarded_db_name(cell_db, 3)
    await ms.restore_database(pg, other, out / ms.DUMP_FILE)
    assert await ms.database_digest(pg, other) == taken["db_digest"]
    # A restore never overwrites an existing database.
    with pytest.raises(ms.MirixServerError, match="exists"):
        await ms.restore_database(pg, other, out / ms.DUMP_FILE)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_database_still_in_use_is_not_rolled_back(pg, cell_db, tmp_path):
    import asyncpg

    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    out = tmp_path / "c"
    out.mkdir()
    await store.checkpoint(out)
    conn = await asyncpg.connect(
        host=pg.host, port=pg.port, user=pg.user, password=pg.password, database=cell_db
    )
    try:
        assert await store.connections() == 1
        with pytest.raises(asyncpg.ObjectInUseError):
            await store.rollback(1, out, tmp_path / "export.json")
        assert await ms.database_exists(pg, cell_db)  # nothing was changed
    finally:
        await conn.close()
    assert await store.connections() == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_an_active_learning_run_is_seen_in_the_dumped_snapshot(
    pg, cell_db, tmp_path
):
    await _sql(
        pg,
        cell_db,
        "INSERT INTO skill_learning_run VALUES ('run3', 'queued', 'flush', '[]', NULL, "
        "NULL, NULL)",
    )
    out = tmp_path / "c"
    out.mkdir()
    taken = await ms.CellDatabase(pg, cell_db, owner=IT_OWNER).checkpoint(out)
    assert taken["active_runs"] == 1  # run.py refuses such a checkpoint


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_database_another_run_marked_is_never_rolled_back_or_started(
    pg, cell_db, tmp_path
):
    """B02 on the real Postgres: the marker is the database's comment, and another
    run's marker stops a rollback (even a claimed one) and a server start."""
    other = ms.DatabaseOwner("it-other-a01-0a1b2c3d", "it-ckpt", 1, "0a1b2c3d")
    assert await ms.database_owners(pg, cell_db, IT_OWNER) == {cell_db: ms.OWNER_OURS}
    await ms.mark_database(pg, cell_db, other)
    store = ms.CellDatabase(pg, cell_db, owner=IT_OWNER)
    before = await store.digest()
    with pytest.raises(ms.CellDatabaseConflict, match=other.run_id):
        await store.rollback(1, None, tmp_path / "export.json", claim_unmarked=True)
    with pytest.raises(ms.CellDatabaseConflict, match=other.run_id):
        await ms.ensure_cell_database(pg, cell_db, must_exist=True, owner=IT_OWNER)
    assert await store.digest() == before
    assert await ms.database_owners(pg, cell_db, other, asides=True) == {
        cell_db: ms.OWNER_OURS
    }
    assert not (tmp_path / "export.json").exists()
