"""One isolated MIRIX server per cell attempt.

Isolation per cell attempt:

* its own Postgres database, cloned from ``skill_bench_template`` (``infra/init.sql``,
  extensions vector / pg_trgm / uuid-ossp preinstalled);
* its own ``HOME`` (``<cell_dir>/attempt-NN/home``, always absolute), so ``~/.mirix``
  (``MIRIX_DIR``, images, config) never touches the real home directory;
* its own port and an owned process group, logged to ``attempt-NN/logs/mirix_server.log``;
* **no provider keys**: the child environment is built from an allowlist, and MIRIX
  reaches models only through the gateway roles ``mirix`` / ``embedding`` that the
  MIRIX method configures per agent.

H8: a learning-phase failure restarts the cell as a new attempt (new database, new
HOME); a test-phase resume reuses the attempt's database and HOME. The caller states
which (:data:`ServerMode`): ``"fresh"`` refuses an existing database or a used HOME,
``"resume"`` refuses a missing database or a HOME without MIRIX's state (``~/.mirix``
missing or empty; so a dropped database or HOME is never silently replaced by an empty
one), ``"create"`` (tests and development only) accepts either.

The cell directory is made absolute once, in :class:`CellServerSpec`:
the server runs in its attempt directory, so a relative ``HOME`` would name a directory
nested under it, not the one the guards check. Attempts that ran before that fix keep
their MIRIX state in such a nested directory; :func:`link_nested_mirix_home` (CLI
``python -m evals.skill_bench.infra.mirix_server link-nested-home``) links it to the
documented ``home/.mirix`` so their test phase can resume.

Learning checkpoints (``run.py``): a learning phase resumes at
item boundaries, so a checkpoint must capture the cell database and HOME exactly and a
resume must restore them before MIRIX starts again. :class:`CellDatabase` does it with
the compose Postgres: :meth:`CellDatabase.checkpoint` exports a snapshot of a
``REPEATABLE READ`` transaction, takes the table digest (:func:`database_digest`) and
checks that no learning run is active inside it, and streams ``pg_dump -Fc
--snapshot=<id>`` from the Postgres container (:func:`postgres_container`, found by its
compose labels; ``docker exec``) into the checkpoint; :meth:`CellDatabase.rollback`
renames the post-crash database aside, restores the dump into a fresh database from
``template0`` (``pg_restore --single-transaction``), exports the rows the restored
database no longer has (:func:`export_discarded_rows`) and drops the post-crash one.
HOME is recorded by :func:`home_manifest` (sizes and sha256; ``.mirix/tmp`` left out) and
restored by :func:`restore_home`; :func:`move_new_home_files` moves aside what a
discarded segment added.

Before starting the real MIRIX server, :func:`probe_llm_contract` checks that this
MIRIX reads ``MIRIX_LLM_MAX_RETRIES`` / ``MIRIX_LLM_TIMEOUT_SECONDS`` /
``MIRIX_EMBEDDING_TIMEOUT_SECONDS`` (``protocol.MIRIX_SETTINGS_CONTRACT``): otherwise its
SDK clients would retry on their own (2 retries x 600 s by default), which H3 / M3.2
forbid, and its embedding calls would give up on the gateway after 60 s and be sent
again, and that ``mirix.constants`` reads ``MIRIX_SESSION_IDLE_TIMEOUT_SECONDS``
(``protocol.MIRIX_CONSTANTS_CONTRACT``: a task session must never end by idling, or a
learning resume long after its checkpoint would find it closed). It also records MIRIX's
experience-pool settings. The gateway additionally flags repeated identical requests in
its ledger (``repeat_of``).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol
from urllib.parse import quote

import httpx

from evals.skill_bench import protocol
from evals.skill_bench.infra.processes import (
    OwnedProcess,
    OwnedProcessError,
    OwnerLock,
    cleanup_owned,
    port_is_free,
)

START_SERVER = protocol.REPO_ROOT / "scripts" / "start_server.py"
MIRIX_PACKAGE_DIR = protocol.REPO_ROOT / "mirix"
EXTENSIONS = ("vector", "pg_trgm", "uuid-ossp")
DB_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_DB_CREATE_LOCK = (
    0x5B_BE_4C_00  # pg_advisory_lock key serialising CREATE DATABASE ... TEMPLATE
)

#: Variables copied from the harness environment; everything else is dropped.
PASSTHROUGH_ENV = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)
_SECRET_NAME_RE = re.compile(
    r"(API_KEY|API_TOKEN|_TOKEN|SECRET|ACCESS_KEY|ACCESS_KEY_ID|CREDENTIALS)$"
    r"|^(OPENAI|ANTHROPIC|GEMINI|GOOGLE|OPENROUTER|AZURE|GROQ|MISTRAL|COHERE|TOGETHER|DEEPSEEK|XAI|AWS|LANGFUSE|COMPOSIO)_"
)
_PROTECTED_ENV = frozenset({"HOME", "MIRIX_PG_URI", *protocol.MIRIX_SERVER_ENV})


#: How a cell attempt's database and HOME are treated (see the module docstring).
ServerMode = Literal["fresh", "resume", "create"]
SERVER_MODES: tuple[str, ...] = ("fresh", "resume", "create")
PROBE_MARKER = "SKILL_BENCH_PROBE "
PROBE_TIMEOUT_S = 120.0
#: MIRIX's state directory under its HOME (``MIRIX_DIR``: config, images, tmp).
MIRIX_STATE_DIR = ".mirix"
#: MIRIX's opt-in setting naming the request header in which it sends the conversation
#: session each LLM and embedding call serves (``mirix/observability/context.py``, B32).
#: ``run.py`` sets it to the gateway's item header, so MIRIX's calls are ledgered with the
#: task uid they serve; a MIRIX without the setting ignores it (item stays null).
MIRIX_SESSION_HEADER_ENV = "MIRIX_LLM_SESSION_HEADER"
ATTEMPT_DIR_RE = re.compile(r"^attempt-\d{2,}$")


class MirixServerError(RuntimeError):
    """A cell's MIRIX server or database could not be prepared, started or kept healthy."""


class CellDatabaseConflict(MirixServerError):
    """The database under a cell attempt's name cannot be proven the attempt's: a new
    attempt found it already there, its owner marker names another run, or a rollback
    would move aside and drop one without a marker (B02). HARNESS
    ``mirix_database_conflict`` (``infra/failures.py``): never retried or resumed
    automatically, since whatever holds the name may be another run's memory; a person
    decides."""


def is_secret_env_name(name: str) -> bool:
    return bool(_SECRET_NAME_RE.search(name.upper()))


@dataclass(frozen=True)
class PostgresAdmin:
    """Connection to the compose Postgres (``infra/compose.yaml``); local-only credentials."""

    host: str = "127.0.0.1"
    port: int = int(os.environ.get("SKILL_BENCH_PG_PORT", "55432"))
    user: str = "skill_bench"
    password: str = field(default="skill_bench_local", repr=False)
    maintenance_db: str = "postgres"
    template: str = "skill_bench_template"

    def uri(self, database: str, driver: str = "postgresql+asyncpg") -> str:
        return f"{driver}://{quote(self.user)}:{quote(self.password)}@{self.host}:{self.port}/{database}"


#: A cell directory's instance id (``cell.json`` ``instance``, ``run.new_instance_id``).
INSTANCE_RE = re.compile(r"^[a-z0-9]{1,12}$")
#: What a cell database name leaves room for: a rollback's ``<db>_dNN`` aside.
_ASIDE_SUFFIX = "_d99"
_POSTGRES_NAME_MAX = 63


def cell_db_name(cell_id: str, attempt: int, instance: str | None = None) -> str:
    """The database name of one cell attempt (<= 63 chars).

    With the cell directory's ``instance`` id: ``sb_<slug>_a<attempt>_i<instance>``, so
    two runs directories holding one cell id never share a database, just as their
    gateway run ids differ (X14, B02: a resume in one runs directory must never drop
    another's database of the same name). The slug is shortened so a rollback's
    ``<db>_dNN`` aside still fits. Without one, the name of the harness before that,
    ``sb_<slug>_a<attempt>_<sha256(cell_id, attempt)[:8]>``, byte for byte: the attempts
    that recorded it and the method's own development server keep finding their
    databases. The ``i`` (no hex digit) keeps the two shapes apart.
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    slug = re.sub(r"[^a-z0-9]+", "_", cell_id.lower()).strip("_")[:40] or "cell"
    if instance is None:
        digest = hashlib.sha256(f"{cell_id}\0{attempt}".encode()).hexdigest()[:8]
        name = f"sb_{slug}_a{attempt}_{digest}"
        assert DB_NAME_RE.fullmatch(name), name
        return name
    if not INSTANCE_RE.fullmatch(instance):
        raise ValueError(f"unsafe instance id {instance!r}")
    tail = f"_a{attempt}_i{instance}"
    room = _POSTGRES_NAME_MAX - len(_ASIDE_SUFFIX) - len("sb_") - len(tail)
    if room < 1:
        raise ValueError(f"attempt {attempt} leaves no room for a database name")
    name = f"sb_{slug[:room]}{tail}"
    assert DB_NAME_RE.fullmatch(name + _ASIDE_SUFFIX), name
    return name


# ---------------------------------------------------------------------------
# Cell database ownership (B02)
# ---------------------------------------------------------------------------

#: The key of a cell database's owner marker, its ``COMMENT ON DATABASE`` (JSON).
OWNER_MARKER_KEY = "skill_bench_owner"
#: What a database's marker says (:func:`owner_state`): none (a database from before
#: owner markers, or of a server run without an owner), this attempt's, or another's (an
#: unparseable comment, or one that is no skill_bench marker, is another's: its owner
#: cannot be proven).
OWNER_UNMARKED = "unmarked"
OWNER_OURS = "ours"
OWNER_FOREIGN = "foreign"


@dataclass(frozen=True)
class DatabaseOwner:
    """The attempt a cell database belongs to: its gateway run id
    (``<cell_id>-aNN-<instance>``, one per cell directory and attempt, X14), with the
    cell id, attempt and instance for the person who reads the marker. A database is an
    attempt's when its marker names the attempt's run id."""

    run_id: str
    cell_id: str
    attempt: int
    instance: str | None = None

    def marker(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "attempt": self.attempt,
            "instance": self.instance,
        }

    def comment(self) -> str:
        """The ``COMMENT ON DATABASE`` text that marks a database as this attempt's."""
        return json.dumps({OWNER_MARKER_KEY: self.marker()}, sort_keys=True)


def marker_of(comment: str | None) -> dict[str, Any] | None:
    """The owner a database comment names (its marker's body), or None when the comment
    is missing or no skill_bench owner marker."""
    if comment is None:
        return None
    try:
        data = json.loads(comment)
    except ValueError:
        return None
    marker = data.get(OWNER_MARKER_KEY) if isinstance(data, dict) else None
    return marker if isinstance(marker, dict) else None


def owner_state(comment: str | None, owner: DatabaseOwner | None) -> str:
    """:data:`OWNER_UNMARKED`, :data:`OWNER_OURS` or :data:`OWNER_FOREIGN` for a database
    whose comment is ``comment``, as seen by ``owner`` (None: nobody's database is ours)."""
    if comment is None:
        return OWNER_UNMARKED
    marker = marker_of(comment)
    if (
        marker is not None
        and owner is not None
        and marker.get("run_id") == owner.run_id
    ):
        return OWNER_OURS
    return OWNER_FOREIGN


def describe_owner(comment: str | None) -> str:
    """Who a database comment says owns the database, for a person."""
    if comment is None:
        return "no owner marker: made before owner markers, or by a server without one"
    marker = marker_of(comment)
    if marker is None:
        return f"its comment is no skill_bench owner marker: {comment[:120]!r}"
    return (
        f"owned by run {marker.get('run_id')} (cell {marker.get('cell_id')}, attempt "
        f"{marker.get('attempt')}, instance {marker.get('instance')})"
    )


@dataclass(frozen=True)
class CellServerSpec:
    """One cell attempt's MIRIX server. ``instance`` (the cell directory's instance id)
    names the database (:func:`cell_db_name`) unless ``database`` gives the name the
    attempt recorded, which always wins (older attempts keep their names); ``owner``
    (the attempt's gateway run id) marks the database it creates as the attempt's and
    proves a database is the attempt's before a rollback touches it (B02)."""

    cell_id: str
    attempt: int
    cell_dir: Path
    port: int
    pg: PostgresAdmin = field(default_factory=PostgresAdmin)
    host: str = "127.0.0.1"
    extra_env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    instance: str | None = None
    owner: str | None = None
    database: str | None = None

    def __post_init__(self) -> None:
        # Absolute once, here: the server runs with its attempt
        # directory as working directory, so a relative HOME would put ~/.mirix in a
        # directory nested under it, while the guards, the log and attempt.json name
        # the documented one. absolute(), not resolve(): a symlinked runs directory
        # (/tmp -> /private/tmp) keeps the spelling the caller gave.
        object.__setattr__(self, "cell_dir", Path(self.cell_dir).absolute())
        if self.instance is not None and not INSTANCE_RE.fullmatch(self.instance):
            raise ValueError(f"unsafe instance id {self.instance!r}")
        if self.database is not None and not (
            DB_NAME_RE.fullmatch(self.database) and self.database.startswith("sb_")
        ):
            raise ValueError(f"{self.database!r} is no cell database name")

    @property
    def attempt_dir(self) -> Path:
        return self.cell_dir / f"attempt-{self.attempt:02d}"

    @property
    def home(self) -> Path:
        return self.attempt_dir / "home"

    @property
    def mirix_dir(self) -> Path:
        """MIRIX's ``~/.mirix`` in this attempt's HOME."""
        return self.home / MIRIX_STATE_DIR

    @property
    def log_path(self) -> Path:
        return self.attempt_dir / "logs" / "mirix_server.log"

    @property
    def registry(self) -> Path:
        return self.cell_dir / "owned_processes.jsonl"

    @property
    def db_name(self) -> str:
        return self.database or cell_db_name(self.cell_id, self.attempt, self.instance)

    @property
    def name_bound_to_instance(self) -> bool:
        """True when the database name is this cell directory's own (it carries the
        instance id): no other runs directory computes it, so an unmarked database of
        this name (a crash between its CREATE and its marker) is this attempt's."""
        return self.instance is not None and self.db_name == cell_db_name(
            self.cell_id, self.attempt, self.instance
        )

    @property
    def database_owner(self) -> DatabaseOwner | None:
        if self.owner is None:
            return None
        return DatabaseOwner(self.owner, self.cell_id, self.attempt, self.instance)

    @property
    def pg_uri(self) -> str:
        return self.pg.uri(self.db_name)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def tiktoken_cache_dir() -> str:
    """The harness's tiktoken cache, shared read-mostly so cells do not re-download the vocab."""
    return (
        os.environ.get("TIKTOKEN_CACHE_DIR")
        or os.environ.get("DATA_GYM_CACHE_DIR")
        or os.path.join(tempfile.gettempdir(), "data-gym-cache")
    )


def build_server_env(
    spec: CellServerSpec, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Child environment: allowlisted basics + cell isolation + protocol settings, no provider keys."""
    base = os.environ if base_env is None else base_env
    env = {k: base[k] for k in PASSTHROUGH_ENV if k in base}
    loopback = "127.0.0.1,localhost,::1"
    env.update(
        {
            "HOME": str(spec.home),
            "MIRIX_PG_URI": spec.pg_uri,
            **protocol.MIRIX_SERVER_ENV,
            "NO_PROXY": loopback,
            "no_proxy": loopback,
            "PYTHONUNBUFFERED": "1",
            "TIKTOKEN_CACHE_DIR": tiktoken_cache_dir(),
            "SKILL_BENCH_CELL_ID": spec.cell_id,
            "SKILL_BENCH_ATTEMPT": str(spec.attempt),
        }
    )
    for name, value in spec.extra_env.items():
        if name in _PROTECTED_ENV:
            raise MirixServerError(f"extra_env may not override {name}")
        if is_secret_env_name(name):
            raise MirixServerError(f"extra_env may not carry credentials ({name})")
        env[name] = value
    leaked = sorted(name for name in env if is_secret_env_name(name))
    if leaked:
        raise MirixServerError(
            f"refusing to pass credential-like variables to MIRIX: {leaked}"
        )
    return env


def dotenv_files_visible_to_mirix(package_dir: Path = MIRIX_PACKAGE_DIR) -> list[Path]:
    """``.env`` files that ``load_dotenv()`` in ``mirix/settings.py`` would find (it walks up from there).

    Such a file could inject provider keys into the MIRIX process behind the gateway's back.
    Only existence is checked; the file is never read.
    """
    found = []
    for directory in (package_dir, *package_dir.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            found.append(candidate)
    return found


def holds_mirix_state(home: Path) -> bool:
    """True if ``home`` has MIRIX's state: a non-empty ``~/.mirix`` (blocking).

    MIRIX writes ``~/.mirix/config`` at every server start, and its images and ``tmp``
    there; the check follows a symlinked ``.mirix`` (:func:`link_nested_mirix_home`)."""
    state = home / MIRIX_STATE_DIR
    return state.is_dir() and any(state.iterdir())


def nested_mirix_dirs(attempt_dir: Path) -> list[Path]:
    """MIRIX state directories left below ``attempt_dir`` by a relative HOME (blocking).

    Before :class:`CellServerSpec` made ``cell_dir`` absolute, a
    relative ``--out`` gave the server ``HOME=<relative cell dir>/attempt-NN/home``,
    which it resolved against its working directory, the attempt directory: MIRIX's
    state landed in ``attempt-NN/<relative cell dir>/attempt-NN/home/.mirix`` and the
    documented ``attempt-NN/home`` stayed empty. The attempt's own ``home`` and symlinks
    are never searched.
    """
    attempt_dir = Path(attempt_dir).absolute()
    found = []
    for dirpath, dirnames, _ in os.walk(attempt_dir):
        here = Path(dirpath)
        if here == attempt_dir:
            dirnames[:] = [d for d in dirnames if d != "home"]
            continue
        if (
            here.name == MIRIX_STATE_DIR
            and here.parent.name == "home"
            and here.parent.parent.name == attempt_dir.name
            and here.parent.parent.parent.name == attempt_dir.parent.name
        ):
            found.append(here)
            dirnames[:] = []  # MIRIX's own files: nothing nested further down
    return sorted(found)


def link_nested_mirix_home(attempt_dir: Path, *, dry_run: bool = False) -> Path | None:
    """Link an attempt's nested MIRIX state (:func:`nested_mirix_dirs`) to its documented
    ``home/.mirix`` (blocking); return the nested directory linked, or None.

    A relative symlink, never a move: MIRIX's file rows hold the image paths it wrote,
    relative to its working directory (the attempt directory), so those files must stay
    where they are; through the link, a resumed server (absolute HOME, same working
    directory) finds its config and images at both paths. Nothing is done when the HOME
    already has a ``.mirix`` or no nested one exists; several nested ones are refused.
    """
    attempt_dir = Path(attempt_dir).absolute()
    home = attempt_dir / "home"
    link = home / MIRIX_STATE_DIR
    if link.is_symlink() or link.exists():
        return None
    nested = nested_mirix_dirs(attempt_dir)
    if not nested:
        return None
    if len(nested) > 1:
        raise MirixServerError(
            f"{attempt_dir} holds several nested MIRIX state directories "
            f"({', '.join(map(str, nested))}); link the right one by hand"
        )
    if not dry_run:
        home.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(nested[0], home), target_is_directory=True)
    return nested[0]


def attempt_dirs(path: Path) -> list[Path]:
    """The attempt directories ``path`` names: itself (``attempt-NN``), a cell's, or the
    cells' of a runs directory (blocking)."""
    path = Path(path).absolute()
    if ATTEMPT_DIR_RE.fullmatch(path.name):
        return [path] if path.is_dir() else []

    def own(directory: Path) -> list[Path]:
        return sorted(
            child
            for child in directory.iterdir()
            if child.is_dir() and ATTEMPT_DIR_RE.fullmatch(child.name)
        )

    if not path.is_dir():
        return []
    found = own(path)
    if found:
        return found
    return [
        attempt
        for child in sorted(path.iterdir())
        if child.is_dir() and not child.is_symlink()
        for attempt in own(child)
    ]


def check_home(spec: CellServerSpec, mode: ServerMode) -> None:
    """The HOME guards of ``mode`` (blocking): the MIRIX state (``home/.mirix``) must
    resolve inside the attempt directory whatever links lead to it — a linked ``home``
    as much as a linked ``.mirix`` — so it never is another attempt's state
    or the real ``~/.mirix``; ``fresh`` needs an unused HOME, ``resume`` one that holds
    MIRIX's state."""
    state = spec.mirix_dir
    attempt = spec.attempt_dir.resolve()
    if not state.resolve().is_relative_to(attempt):
        raise MirixServerError(
            f"{state} leads to {state.resolve()}, outside {spec.attempt_dir}: an "
            "attempt's MIRIX state never leaves its attempt directory (H8)"
        )
    if mode == "fresh" and spec.home.is_dir() and any(spec.home.iterdir()):
        raise MirixServerError(
            f"{spec.home} is not empty; a new learning attempt needs a fresh HOME"
        )
    if mode != "resume":
        return
    if not holds_mirix_state(spec.home):
        nested = nested_mirix_dirs(spec.attempt_dir)
        hint = (
            f". This attempt's MIRIX state is at {nested[0]} (a server started with a "
            "relative HOME, before cell directories were made absolute); link it with "
            "`python -m "
            f"evals.skill_bench.infra.mirix_server link-nested-home {spec.attempt_dir}`"
            if len(nested) == 1
            else ""
        )
        raise MirixServerError(
            f"{spec.mirix_dir} is missing or empty; a resumed attempt needs its own HOME "
            "(~/.mirix holds images and state), never a new empty one" + hint
        )


def default_command(spec: CellServerSpec, python: str = sys.executable) -> list[str]:
    return [
        python,
        str(START_SERVER),
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--log-level",
        "warning",
    ]


Connect = Callable[..., Awaitable[Any]]


async def _asyncpg_connect(**kwargs: Any) -> Any:
    import asyncpg

    return await asyncpg.connect(**kwargs)


_OWNER_QUERY = (
    "SELECT datname, shobj_description(oid, 'pg_database') AS comment "
    "FROM pg_database WHERE datname = $1"
)


async def _comments_on(
    conn: Any, db_name: str, *, asides: bool = False
) -> dict[str, str | None]:
    """``{name: comment}`` of ``db_name`` (and, with ``asides``, of its rollbacks'
    ``<db>_dNN`` asides) as far as they exist. ``db_name`` passed :data:`DB_NAME_RE`, so
    it holds no regular-expression character."""
    if asides:
        rows = await conn.fetch(
            _OWNER_QUERY + " OR datname ~ $2", db_name, f"^{db_name}_d[0-9]{{2,}}$"
        )
    else:
        rows = await conn.fetch(_OWNER_QUERY, db_name)
    return {str(r["datname"]): r["comment"] for r in rows}


async def _mark_on(conn: Any, db_name: str, owner: DatabaseOwner) -> None:
    """Mark ``db_name`` as ``owner``'s. Postgres quotes the name and the marker itself
    (``format('%I', '%L')``); the statement is never assembled from text here."""
    sql = await conn.fetchval(
        "SELECT format('COMMENT ON DATABASE %I IS %L', $1::text, $2::text)",
        db_name,
        owner.comment(),
    )
    await conn.execute(sql)


async def mark_database(
    pg: PostgresAdmin,
    db_name: str,
    owner: DatabaseOwner,
    *,
    connect: Connect = _asyncpg_connect,
) -> None:
    """Mark the existing database ``db_name`` as ``owner``'s (its ``COMMENT``)."""
    _check_db_name(db_name)
    conn = await _admin(pg, connect)
    try:
        await _mark_on(conn, db_name, owner)
    finally:
        await conn.close()


async def database_owners(
    pg: PostgresAdmin,
    db_name: str,
    owner: DatabaseOwner | None,
    *,
    asides: bool = False,
    connect: Connect = _asyncpg_connect,
) -> dict[str, str]:
    """``{name: owner state}`` (:func:`owner_state`) of ``db_name`` and, with ``asides``,
    its ``<db>_dNN`` asides, as ``owner`` sees them; a missing database is left out.
    Read-only."""
    _check_db_name(db_name)
    conn = await _admin(pg, connect)
    try:
        comments = await _comments_on(conn, db_name, asides=asides)
    finally:
        await conn.close()
    return {name: owner_state(comment, owner) for name, comment in comments.items()}


async def ensure_cell_database(
    pg: PostgresAdmin,
    db_name: str,
    *,
    must_be_new: bool = False,
    must_exist: bool = False,
    owner: DatabaseOwner | None = None,
    connect: Connect = _asyncpg_connect,
) -> bool:
    """Create ``db_name`` from the template if missing; return True if it was created.

    ``must_be_new`` (a new learning attempt) refuses an existing database instead of reusing it;
    ``must_exist`` (a test-phase resume) refuses to create a missing one, which would hold no
    learned memory. Creation is serialised with an advisory lock because concurrent clones
    of one template can fail.

    Ownership (B02): a database this call creates is marked as ``owner``'s under the lock,
    right after its ``CREATE``. An existing database whose marker names another run (or
    that carries a comment that is no marker, or any marker when ``owner`` is None) is
    refused in every mode, and a new attempt refuses any existing one, both as
    :class:`CellDatabaseConflict`. A database without a marker (made before markers)
    is accepted where an existing database is: a test-phase resume of an older attempt
    goes on, and nothing is marked on the way.
    """
    if must_be_new and must_exist:
        raise ValueError("must_be_new and must_exist exclude each other")
    for name in (db_name, pg.template):
        if not DB_NAME_RE.fullmatch(name):
            raise ValueError(f"unsafe database name {name!r}")
    auth = {
        "host": pg.host,
        "port": pg.port,
        "user": pg.user,
        "password": pg.password,
        "timeout": 30,
    }
    conn = await connect(database=pg.maintenance_db, **auth)
    try:
        await conn.execute("SELECT pg_advisory_lock($1)", _DB_CREATE_LOCK)
        try:
            comments = await _comments_on(conn, db_name)
            exists = db_name in comments
            if exists and must_be_new:
                raise CellDatabaseConflict(
                    f"database {db_name} already exists "
                    f"({describe_owner(comments[db_name])}); a new learning attempt "
                    "needs a fresh one. Its name may belong to another runs directory "
                    "or attempt, so nothing resumes or drops it automatically: find out "
                    "whose it is before anything else"
                )
            if exists and owner_state(comments[db_name], owner) == OWNER_FOREIGN:
                raise CellDatabaseConflict(
                    f"database {db_name} is not this attempt's "
                    f"({describe_owner(comments[db_name])}; this attempt is run "
                    f"{owner.run_id if owner else '(none)'}): MIRIX never starts on "
                    "another run's memory"
                )
            if not exists and must_exist:
                raise MirixServerError(
                    f"database {db_name} does not exist; a resumed attempt needs its own "
                    "database (it holds the learned memory), never a new empty one"
                )
            if not exists:
                await conn.execute(
                    f'CREATE DATABASE "{db_name}" TEMPLATE "{pg.template}"'
                )
                if owner is not None:
                    await _mark_on(conn, db_name, owner)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _DB_CREATE_LOCK)
    finally:
        await conn.close()
    conn = await connect(database=db_name, **auth)
    try:
        for extension in EXTENSIONS:
            await conn.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}"')
    finally:
        await conn.close()
    return not exists


async def drop_cell_database(
    pg: PostgresAdmin, db_name: str, *, connect: Connect = _asyncpg_connect
) -> None:
    """Drop one cell database (manual cleanup; failed attempts are kept for forensics by default)."""
    if not DB_NAME_RE.fullmatch(db_name) or not db_name.startswith("sb_"):
        raise ValueError(f"refusing to drop {db_name!r}")
    conn = await connect(
        database=pg.maintenance_db,
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        timeout=30,
    )
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    finally:
        await conn.close()


_PROBE_CODE = """
import json, sys
sys.path.insert(0, {repo!r})
from mirix.settings import settings
out = {{}}
for name in {names!r}:
    out[name] = getattr(settings, name) if hasattr(settings, name) else {{"missing": True}}
try:
    from mirix import constants
    for name in {constants_names!r}:
        out[name] = getattr(constants, name) if hasattr(constants, name) else {{"missing": True}}
except Exception as exc:
    for name in {constants_names!r}:
        out[name] = {{"missing": True, "error": repr(exc)[:300]}}
out["llm_session_header"] = (
    getattr(settings, "llm_session_header")
    if hasattr(settings, "llm_session_header")
    else {{"missing": True}}
)
try:
    from mirix import constants
    out["skill_experience_pool"] = {{
        name: getattr(constants, "SKILL_EXPERIENCE_" + name.upper(), None)
        for name in {pool_settings!r}
    }}
except Exception as exc:
    out["skill_experience_pool"] = {{"missing": True, "error": repr(exc)[:300]}}
print({marker!r} + json.dumps(out, default=str), flush=True)
"""

#: MIRIX's experience-pool settings the probe records (the harness sets none of them, so
#: a cell runs MIRIX's defaults; the record shows which defaults a cell ran with).
POOL_SETTINGS_PROBED = (
    "pool_enabled",
    "history_min_similarity",
    "recurrence_similarity",
    "history_top_k",
    "pool_scan_limit",
    "pool_decay",
    "pool_retire_ignored",
    "not_worth_penalty",
    "not_worth_start",
    "pool_embed_max_failures",
)


def check_llm_contract(
    values: Mapping[str, Any], env: Mapping[str, str] = protocol.MIRIX_SERVER_ENV
) -> list[str]:
    """Problems with MIRIX's effective LLM settings (empty list = contract honoured):
    the retry/timeout settings the cell sets and the constants it sets
    (``protocol.MIRIX_CONSTANTS_CONTRACT``: the session idle timeout)."""
    problems = []
    contracts = [
        ("settings", name, variable)
        for name, variable in protocol.MIRIX_SETTINGS_CONTRACT.items()
    ] + [
        ("constants", name, variable)
        for name, variable in protocol.MIRIX_CONSTANTS_CONTRACT.items()
    ]
    for module, name, variable in contracts:
        value = values.get(name)
        if isinstance(value, Mapping) and value.get("missing"):
            problems.append(
                f"mirix.{module} has no {name!r} (does not read ${variable})"
            )
            continue
        try:
            same = float(value) == float(env[variable])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            same = False
        if not same:
            problems.append(
                f"{module}.{name} = {value!r}, but the cell sets {variable}={env[variable]}"
            )
    return problems


async def probe_llm_contract(
    env: Mapping[str, str],
    *,
    python: str = sys.executable,
    repo: Path = protocol.REPO_ROOT,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> dict[str, Any]:
    """Import ``mirix.settings`` under the cell's environment and read the contract fields,
    MIRIX's experience-pool settings (``skill_experience_pool``, :data:`POOL_SETTINGS_PROBED`)
    and, for the record only, the session header it sends (``llm_session_header``,
    :data:`MIRIX_SESSION_HEADER_ENV`; ``{"missing": True}`` for a MIRIX without it).

    Runs in a child process (the harness never imports MIRIX) with a throw-away HOME and
    working directory, so the probe leaves nothing in the cell; returns the values.
    """
    code = _PROBE_CODE.format(
        repo=str(repo),
        names=list(protocol.MIRIX_SETTINGS_CONTRACT),
        constants_names=list(protocol.MIRIX_CONSTANTS_CONTRACT),
        pool_settings=list(POOL_SETTINGS_PROBED),
        marker=PROBE_MARKER,
    )
    with tempfile.TemporaryDirectory(prefix="skill-bench-probe-") as scratch:
        proc = await asyncio.create_subprocess_exec(
            python,
            "-c",
            code,
            cwd=scratch,
            env={**env, "HOME": scratch},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise MirixServerError(
                f"MIRIX settings probe did not finish within {timeout_s:g} s"
            ) from exc
    text = out.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(PROBE_MARKER):
            return json.loads(line[len(PROBE_MARKER) :])
    raise MirixServerError(
        f"MIRIX settings probe failed (exit code {proc.returncode}): {text[-2000:]}"
    )


def _healthy(response: httpx.Response) -> bool:
    if response.status_code != 200:
        return False
    try:
        return response.json().get("status") == "healthy"
    except ValueError:
        return False


class MirixServer:
    """Start/stop the MIRIX server of one cell attempt.

    ``mode`` is required: ``"fresh"`` for a new learning attempt (new database, empty
    HOME), ``"resume"`` to restart an attempt's server for the test phase (its database
    must exist and its HOME hold MIRIX's state), ``"create"`` for tests and development
    only. ``command`` replaces ``scripts/start_server.py`` (tests use a stub). With
    ``database=True`` the cell database is prepared first. ``verify_llm_contract`` (default: only for the real MIRIX
    command) runs :func:`probe_llm_contract` and refuses to start a MIRIX that would retry
    on its own.
    """

    def __init__(
        self,
        spec: CellServerSpec,
        *,
        mode: ServerMode,
        command: Sequence[str] | None = None,
        base_env: Mapping[str, str] | None = None,
        database: bool = True,
        verify_llm_contract: bool | None = None,
        startup_timeout_s: float = 180.0,
        connect: Connect = _asyncpg_connect,
    ) -> None:
        if mode not in SERVER_MODES:
            raise ValueError(f"mode must be one of {SERVER_MODES}, not {mode!r}")
        self.spec = spec
        self.mode: ServerMode = mode
        self.command = list(command) if command is not None else default_command(spec)
        self.env = build_server_env(spec, base_env)
        self.database = database
        self.verify_llm_contract = (
            command is None if verify_llm_contract is None else verify_llm_contract
        )
        self.llm_contract: dict[str, Any] | None = None
        self.startup_timeout_s = startup_timeout_s
        self._connect = connect
        self.process: OwnedProcess | None = None
        self._lock: OwnerLock | None = None

    @property
    def fresh(self) -> bool:
        return self.mode == "fresh"

    @property
    def base_url(self) -> str:
        return self.spec.base_url

    async def start(self) -> None:
        """Lock the cell, prepare database and HOME, spawn the server and wait for /health."""
        await asyncio.to_thread(self._acquire_cell_lock)
        try:
            await self._start_locked()
        except BaseException:
            self._release_cell_lock()
            raise

    async def _start_locked(self) -> None:
        spec = self.spec
        leaks = dotenv_files_visible_to_mirix()
        if leaks:
            raise MirixServerError(
                f"MIRIX would load {leaks[0]} via load_dotenv(); move it out of the way"
            )
        await asyncio.to_thread(check_home, spec, self.mode)
        # Safe under the cell lock: any registered, still-running server of this cell
        # belongs to a harness process that has died.
        await asyncio.to_thread(cleanup_owned, spec.registry)
        if not port_is_free(spec.port, spec.host):
            raise MirixServerError(f"port {spec.port} is already in use")
        if self.verify_llm_contract:
            values = await probe_llm_contract(self.env)
            problems = check_llm_contract(values)
            if problems:
                raise MirixServerError(
                    "this MIRIX does not honour the retry/timeout contract (H3, M3.2): "
                    + "; ".join(problems)
                    + ". Its SDK clients would retry on their own inside the gateway's "
                    "deadline; use a MIRIX that reads these settings before running MIRIX "
                    "cells."
                )
            self.llm_contract = values
        if self.database:
            await ensure_cell_database(
                spec.pg,
                spec.db_name,
                must_be_new=self.mode == "fresh",
                must_exist=self.mode == "resume",
                owner=spec.database_owner,
                connect=self._connect,
            )
        await asyncio.to_thread(spec.home.mkdir, parents=True, exist_ok=True)
        self.process = await asyncio.to_thread(
            OwnedProcess.spawn,
            f"mirix-{spec.cell_id}-a{spec.attempt}",
            self.command,
            env=self.env,
            cwd=spec.attempt_dir,
            log_path=spec.log_path,
            registry=spec.registry,
        )
        try:
            await self.process.wait_healthy(
                spec.base_url + "/health",
                timeout_s=self.startup_timeout_s,
                accept=_healthy,
            )
        except OwnedProcessError as exc:
            await self.process.stop()
            raise MirixServerError(str(exc)) from exc

    def _acquire_cell_lock(self) -> None:
        """Exclusive, crash-released lock: one live server per cell across all processes
        (blocking). Retries for ``OWNER_LOCK_WAIT_S``: a scheduler's sweep probes the
        lock for a moment, which must not look like another live harness (X23)."""
        if self._lock is not None:
            raise MirixServerError(f"server of {self.spec.cell_id} already started")
        try:
            self._lock = OwnerLock.acquire(self.spec.cell_dir / "server.lock")
        except OwnedProcessError:
            raise MirixServerError(
                f"cell {self.spec.cell_id} is owned by another live harness process"
            ) from None

    def _release_cell_lock(self) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()

    async def ensure_alive(self) -> None:
        """Raise :class:`MirixServerError` (an infra failure) if the server died."""
        if self.process is None or not self.process.alive():
            code = None if self.process is None else self.process.popen.returncode
            raise MirixServerError(
                f"MIRIX server of {self.spec.cell_id} is not running (exit code {code})"
            )

    async def stop(self, grace_s: float = 20.0) -> int | None:
        try:
            if self.process is None:
                return None
            return await self.process.stop(grace_s)
        finally:
            self._release_cell_lock()

    def checkpoint_store(self) -> CheckpointStore:
        """The cell database, as learning checkpoints dump and restore it: the compose
        Postgres (:class:`CellDatabase`), or :class:`NoDatabase` for a server run without
        one (``database=False``: tests with a stub server)."""
        if not self.database:
            return NoDatabase(self.spec.db_name)
        return CellDatabase.of(self.spec, connect=self._connect)

    async def __aenter__(self) -> MirixServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# Learning checkpoints: the cell database
# ---------------------------------------------------------------------------

#: The compose project and service of the skill_bench Postgres (``infra/compose.yaml``).
COMPOSE_PROJECT = "mirix-skill-bench"
COMPOSE_SERVICE = "postgres"
#: Names one Postgres container explicitly (a Postgres outside this compose project).
PG_CONTAINER_ENV = "SKILL_BENCH_PG_CONTAINER"
#: The database dump inside a checkpoint directory (``pg_dump -Fc``).
DUMP_FILE = "mirix_db.dump"
#: What :meth:`CellDatabase.rollback` writes: the rows only the post-crash database had.
EXPORT_FILE = "mirix_export.json"
#: MIRIX's learning-run statuses of a run that is not finished (M9).
ACTIVE_RUN_STATUSES = ("queued", "running")
#: Bound of one ``pg_dump`` / ``pg_restore`` through ``docker exec``.
DUMP_TIMEOUT_S = 900.0
#: The retrieval log, which a frozen test phase writes by design (one row per retrieval).
RETRIEVAL_LOG_TABLE = "retrieval_log"
#: The tables whose added rows a rollback exports one by one (the others are counted),
#: with the columns it keeps of each (those the table has); :data:`SUMMARISED_COLUMNS`
#: are kept as their size and sha256.
EXPORTED_TABLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "retrieval_log": (
            "id",
            "session_id",
            "extraction_status",
            "embedding_status",
            "created_at",
        ),
        "skill_learning_run": (
            "id",
            "status",
            "trigger",
            "mode",
            "session_ids",
            "started_at",
            "finished_at",
            "error",
        ),
        "conversation_session": (
            "id",
            "session_id",
            "message_count",
            "closed_at",
            "close_reason",
            "distill_failures",
        ),
        "conversation_message": ("id", "session_id", "role", "content"),
        "skill_experience": ("id", "session_id", "status", "title", "content"),
    }
)
SUMMARISED_COLUMNS = frozenset({"content"})


def _check_db_name(name: str) -> str:
    if not DB_NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe database name {name!r}")
    return name


def discarded_db_name(db_name: str, segment: int) -> str:
    """The name a rollback gives the post-crash database of discarded segment ``segment``
    (``<db>_dSS``) while it exports what only that database has."""
    if segment < 1:
        raise ValueError("segments are numbered from 1")
    return _check_db_name(f"{db_name}_d{segment:02d}")


async def _docker(
    *args: str,
    docker_bin: str = "docker",
    stdin: Any = None,
    stdout: Any = None,
    timeout_s: float = 120.0,
) -> tuple[int, bytes, bytes]:
    """Run ``docker <args>`` (async, never blocking the loop); returns the exit code, the
    captured stdout (empty when ``stdout`` is a file) and stderr."""
    proc = await asyncio.create_subprocess_exec(
        docker_bin,
        *args,
        stdin=stdin if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=stdout if stdout is not None else asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out or b"", err or b""


async def postgres_container(
    *,
    docker_bin: str = "docker",
    project: str = COMPOSE_PROJECT,
    service: str = COMPOSE_SERVICE,
) -> str:
    """The id of the skill_bench Postgres container: ``$SKILL_BENCH_PG_CONTAINER``, else
    the one running container with the compose labels of ``infra/compose.yaml``. None or
    several is a :class:`MirixServerError` (the checkpoint cannot know which one holds
    the cell databases)."""
    named = os.environ.get(PG_CONTAINER_ENV)
    if named:
        return named
    try:
        code, out, err = await _docker(
            "ps",
            "-q",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--filter",
            f"label=com.docker.compose.service={service}",
            docker_bin=docker_bin,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        raise MirixServerError(
            f"docker ps failed: {type(exc).__name__}: {exc}"
        ) from exc
    if code != 0:
        raise MirixServerError(
            f"docker ps failed: {err.decode(errors='replace')[:500]}"
        )
    ids = [line.strip() for line in out.decode().splitlines() if line.strip()]
    if len(ids) != 1:
        raise MirixServerError(
            f"{len(ids)} running containers carry the labels of compose project "
            f"{project!r} service {service!r}; a learning checkpoint needs exactly one "
            f"(or ${PG_CONTAINER_ENV})"
        )
    return ids[0]


async def _admin(
    pg: PostgresAdmin, connect: Connect, database: str | None = None
) -> Any:
    return await connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=database or pg.maintenance_db,
        timeout=30,
    )


async def database_exists(
    pg: PostgresAdmin, db_name: str, *, connect: Connect = _asyncpg_connect
) -> bool:
    conn = await _admin(pg, connect)
    try:
        return bool(
            await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        )
    finally:
        await conn.close()


async def db_connections(
    pg: PostgresAdmin, db_name: str, *, connect: Connect = _asyncpg_connect
) -> int:
    """How many backends other than this one are connected to ``db_name``."""
    conn = await _admin(pg, connect)
    try:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = $1 "
                "AND pid <> pg_backend_pid()",
                db_name,
            )
        )
    finally:
        await conn.close()


async def rename_database(
    pg: PostgresAdmin, old: str, new: str, *, connect: Connect = _asyncpg_connect
) -> None:
    """``ALTER DATABASE old RENAME TO new`` (Postgres refuses while anyone is connected)."""
    _check_db_name(old)
    _check_db_name(new)
    conn = await _admin(pg, connect)
    try:
        await conn.execute(f'ALTER DATABASE "{old}" RENAME TO "{new}"')
    finally:
        await conn.close()


async def _digest_on(
    conn: Any,
    *,
    include_retrieval_log: bool = True,
    drop_columns: Mapping[str, Collection[str]] | None = None,
) -> dict[str, list[Any]]:
    tables = [
        r["tablename"]
        for r in await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "ORDER BY tablename"
        )
    ]
    digest: dict[str, list[Any]] = {}
    for table in tables:
        if table == RETRIEVAL_LOG_TABLE and not include_retrieval_log:
            continue
        quoted = '"' + table.replace('"', '""') + '"'
        dropped = "".join(
            " - '" + column.replace("'", "''") + "'"
            for column in sorted((drop_columns or {}).get(table, ()))
        )
        text = f"(to_jsonb(t){dropped})::text"
        row = await conn.fetchrow(
            f"SELECT count(*) AS n, "
            f"md5(coalesce(string_agg(md5({text}), ',' ORDER BY md5({text})), '')) "
            f"AS h FROM {quoted} t"
        )
        digest[table] = [int(row["n"]), str(row["h"])]
    return digest


async def database_digest(
    pg: PostgresAdmin,
    db_name: str,
    *,
    include_retrieval_log: bool = True,
    drop_columns: Mapping[str, Collection[str]] | None = None,
    connect: Connect = _asyncpg_connect,
) -> dict[str, list[Any]]:
    """``{table: [rows, md5 of the sorted row texts]}`` for every table of the ``public``
    schema of ``db_name``: equal exactly when no row was added, changed or removed.

    A learning checkpoint keeps everything (the retrieval log included, no column
    dropped). The test-phase checks (``tests/mirix_db.py``) leave out the retrieval log,
    which a frozen test phase writes by design, and the columns a server start rewrites."""
    conn = await _admin(pg, connect, db_name)
    try:
        return await _digest_on(
            conn, include_retrieval_log=include_retrieval_log, drop_columns=drop_columns
        )
    finally:
        await conn.close()


async def _active_runs_on(conn: Any) -> int:
    exists = await conn.fetchval("SELECT to_regclass('public.skill_learning_run')")
    if exists is None:
        return 0
    return int(
        await conn.fetchval(
            "SELECT count(*) FROM skill_learning_run WHERE status = ANY($1::text[])",
            list(ACTIVE_RUN_STATUSES),
        )
    )


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _open_for_write(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


async def checkpoint_database(
    pg: PostgresAdmin,
    db_name: str,
    out_path: Path,
    *,
    container: str | None = None,
    docker_bin: str = "docker",
    connect: Connect = _asyncpg_connect,
) -> dict[str, Any]:
    """Dump ``db_name`` into ``out_path`` (``pg_dump -Fc``) as one consistent snapshot.

    On one connection, a ``REPEATABLE READ READ ONLY`` transaction exports its snapshot
    (``pg_export_snapshot()``), takes the table digest (:func:`database_digest`, the
    retrieval log included) and counts the active learning runs inside it, and stays
    open while ``docker exec <container> pg_dump --snapshot=<id>`` streams the same
    snapshot into ``out_path``. So the digest describes exactly the dump, whatever
    writes the database while it runs. Returns ``db_name``, ``dump_sha256``,
    ``dump_bytes``, ``db_digest``, ``active_runs`` and ``snapshot_id``; the caller refuses
    a checkpoint with an active run."""
    _check_db_name(db_name)
    container = container or await postgres_container(docker_bin=docker_bin)
    conn = await _admin(pg, connect, db_name)
    try:
        transaction = conn.transaction(isolation="repeatable_read", readonly=True)
        await transaction.start()
        try:
            snapshot = str(await conn.fetchval("SELECT pg_export_snapshot()"))
            digest = await _digest_on(conn)
            active = await _active_runs_on(conn)
            handle = await asyncio.to_thread(_open_for_write, out_path)
            try:
                code, _out, err = await _docker(
                    "exec",
                    "-e",
                    f"PGPASSWORD={pg.password}",
                    container,
                    "pg_dump",
                    "-U",
                    pg.user,
                    "-d",
                    db_name,
                    "-Fc",
                    f"--snapshot={snapshot}",
                    docker_bin=docker_bin,
                    stdout=handle,
                    timeout_s=DUMP_TIMEOUT_S,
                )
            finally:
                await asyncio.to_thread(handle.close)
            if code != 0:
                raise MirixServerError(
                    f"pg_dump of {db_name} failed ({code}): "
                    f"{err.decode(errors='replace')[:1000]}"
                )
        finally:
            await transaction.rollback()
    finally:
        await conn.close()
    await asyncio.to_thread(_fsync_file, out_path)
    sha, size = await asyncio.to_thread(_file_sha256, out_path)
    return {
        "db_name": db_name,
        "dump_sha256": sha,
        "dump_bytes": size,
        "db_digest": digest,
        "active_runs": active,
        "snapshot_id": snapshot,
    }


async def restore_database(
    pg: PostgresAdmin,
    db_name: str,
    dump_path: Path,
    *,
    container: str | None = None,
    docker_bin: str = "docker",
    owner: DatabaseOwner | None = None,
    connect: Connect = _asyncpg_connect,
) -> None:
    """Create ``db_name`` from ``template0`` and restore ``dump_path`` into it in one
    transaction (``pg_restore --single-transaction --exit-on-error``). An existing
    ``db_name`` is refused (the caller moves the post-crash database aside first); a
    failed restore drops the half-made database again. The new database is marked as
    ``owner``'s before ``pg_restore`` runs: a dump without ``--create`` carries no
    database comment, so the restored database would otherwise lose its marker (B02)."""
    _check_db_name(db_name)
    container = container or await postgres_container(docker_bin=docker_bin)
    if await database_exists(pg, db_name, connect=connect):
        raise MirixServerError(
            f"database {db_name} exists: a restore creates it (move it aside first)"
        )
    conn = await _admin(pg, connect)
    try:
        await conn.execute(f'CREATE DATABASE "{db_name}" TEMPLATE template0')
    finally:
        await conn.close()
    try:
        if owner is not None:
            await mark_database(pg, db_name, owner, connect=connect)
        handle = await asyncio.to_thread(dump_path.open, "rb")
        try:
            code, _out, err = await _docker(
                "exec",
                "-i",
                "-e",
                f"PGPASSWORD={pg.password}",
                container,
                "pg_restore",
                "-U",
                pg.user,
                "-d",
                db_name,
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                "--single-transaction",
                docker_bin=docker_bin,
                stdin=handle,
                timeout_s=DUMP_TIMEOUT_S,
            )
        finally:
            await asyncio.to_thread(handle.close)
        if code != 0:
            raise MirixServerError(
                f"pg_restore into {db_name} failed ({code}): "
                f"{err.decode(errors='replace')[:1000]}"
            )
    except BaseException:
        await drop_cell_database(pg, db_name, connect=connect)
        raise


def _plain(value: Any) -> Any:
    """A database value as JSON (timestamps as ISO text, JSON columns parsed, anything
    else that JSON cannot hold as text)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        if value[:1] in "[{":
            try:
                return json.loads(value)
            except ValueError:
                return value
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


async def _columns(conn: Any, table: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1",
        table,
    )
    return {str(r["column_name"]) for r in rows}


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _export_select(table: str, available: set[str]) -> str:
    parts = []
    for column in EXPORTED_TABLES[table]:
        if column not in available:
            continue
        quoted = _quote_ident(column)
        if column in SUMMARISED_COLUMNS:
            parts.append(
                f"octet_length({quoted}::text) AS {_quote_ident(column + '_bytes')}"
            )
            parts.append(
                f"encode(sha256(convert_to({quoted}::text, 'UTF8')), 'hex') "
                f"AS {_quote_ident(column + '_sha256')}"
            )
        else:
            parts.append(quoted)
    return ", ".join(parts)


async def _row_hashes(conn: Any, table: str) -> dict[str, str] | None:
    """``{id: md5 of the row}`` of a table with an ``id`` column (None without one)."""
    has_id = await conn.fetchval(
        "SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' "
        "AND table_name = $1 AND column_name = 'id'",
        table,
    )
    if not has_id:
        return None
    rows = await conn.fetch(
        f"SELECT id::text AS id, md5(to_jsonb(t)::text) AS h FROM {_quote_ident(table)} t"
    )
    return {str(r["id"]): str(r["h"]) for r in rows}


async def export_discarded_rows(
    pg: PostgresAdmin,
    discarded_db: str,
    live_db: str | None,
    *,
    connect: Connect = _asyncpg_connect,
) -> dict[str, Any]:
    """What the post-crash database ``discarded_db`` holds that the restored ``live_db``
    does not (None: nothing was restored, a rollback to before the first checkpoint).

    ``tables``: per table with an ``id`` column, the ids only the post-crash database has
    (``added``), those only the live one has (``removed``) and those whose row differs
    (``changed``). ``rows``: for :data:`EXPORTED_TABLES` the added rows' key columns
    (retrieval-log rows, learning runs, sessions, messages by size and sha256,
    experiences). Blocking reads of both databases, never a write."""
    lost = await _admin(pg, connect, discarded_db)
    live = await _admin(pg, connect, live_db) if live_db else None
    try:
        tables = [
            r["tablename"]
            for r in await lost.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "ORDER BY tablename"
            )
        ]
        summary: dict[str, Any] = {}
        rows: dict[str, list[dict[str, Any]]] = {}
        for table in tables:
            before = await _row_hashes(lost, table)
            if before is None:
                continue
            after = (await _row_hashes(live, table) or {}) if live is not None else {}
            added = sorted(set(before) - set(after))
            summary[table] = {
                "added": len(added),
                "removed": len(set(after) - set(before)),
                "changed": sorted(
                    i for i in set(before) & set(after) if before[i] != after[i]
                ),
            }
            if table in EXPORTED_TABLES and added:
                fetched = await lost.fetch(
                    f"SELECT {_export_select(table, await _columns(lost, table))} "
                    f"FROM {_quote_ident(table)} "
                    "WHERE id::text = ANY($1::text[]) ORDER BY id",
                    added,
                )
                rows[table] = [
                    {key: _plain(value) for key, value in dict(r).items()}
                    for r in fetched
                ]
        return {
            "discarded_db": discarded_db,
            "live_db": live_db,
            "tables": summary,
            "rows": rows,
        }
    finally:
        await lost.close()
        if live is not None:
            await live.close()


class CheckpointStore(Protocol):
    """What ``run.py`` needs of a MIRIX cell's memory store for learning checkpoints: the
    compose Postgres (:class:`CellDatabase`), or a test's stub MIRIX."""

    @property
    def name(self) -> str: ...

    async def check_available(self) -> None: ...

    async def check_owner(
        self, *, claim_unmarked: bool = False, segment: int | None = None
    ) -> dict[str, Any]: ...

    async def connections(self) -> int: ...

    async def active_runs(self) -> int: ...

    async def checkpoint(self, out_dir: Path) -> dict[str, Any]: ...

    async def digest(self) -> dict[str, Any]: ...

    async def rollback(
        self,
        segment: int,
        checkpoint_dir: Path | None,
        export_path: Path,
        *,
        keep_discarded: bool = False,
        claim_unmarked: bool = False,
    ) -> dict[str, Any]: ...


def export_summary(export: Mapping[str, Any]) -> dict[str, Any]:
    """The ``discarded.jsonl`` ``mirix`` summary of an export: the discarded retrievals,
    learning runs and sessions (with their message counts)."""
    rows = export.get("rows") or {}
    messages: dict[str, int] = {}
    for message in rows.get("conversation_message") or ():
        session = str(message.get("session_id"))
        messages[session] = messages.get(session, 0) + 1
    sessions = [
        {
            "session_id": s.get("session_id"),
            "messages": s.get("message_count"),
            "closed": s.get("closed_at") is not None,
        }
        for s in rows.get("conversation_session") or ()
    ]
    known = {s["session_id"] for s in sessions}
    sessions += [
        {"session_id": sid, "messages": n, "closed": None, "new_messages_only": True}
        for sid, n in sorted(messages.items())
        if sid not in known
    ]
    return {
        "retrievals": [
            {
                "id": r.get("id"),
                "session_id": r.get("session_id"),
                "extraction_status": r.get("extraction_status"),
                "embedding_status": r.get("embedding_status"),
            }
            for r in rows.get("retrieval_log") or ()
        ],
        "learning_runs": [
            {
                "id": r.get("id"),
                "status": r.get("status"),
                "trigger": r.get("trigger"),
                "session_ids": r.get("session_ids"),
                "started_at": r.get("started_at"),
                "finished_at": r.get("finished_at"),
            }
            for r in rows.get("skill_learning_run") or ()
        ],
        "sessions": sessions,
        "changed": {
            table: entry.get("changed")
            for table, entry in (export.get("tables") or {}).items()
            if entry.get("changed")
        },
    }


class CellDatabase:
    """A MIRIX cell attempt's database on the compose Postgres, as a
    :class:`CheckpointStore`.

    ``owner`` is the attempt the database belongs to: a rollback renames or drops only
    databases :meth:`check_owner` proves are its own, and marks the one it restores.
    ``claim_unmarked`` takes a database without a marker as the attempt's without a
    person's word: right only when no other runs directory computes the name (it carries
    the cell directory's instance id, :attr:`CellServerSpec.name_bound_to_instance`)."""

    def __init__(
        self,
        pg: PostgresAdmin,
        db_name: str,
        *,
        owner: DatabaseOwner | None = None,
        claim_unmarked: bool = False,
        docker_bin: str = "docker",
        container: str | None = None,
        connect: Connect = _asyncpg_connect,
    ) -> None:
        self.pg = pg
        self.db_name = _check_db_name(db_name)
        self.owner = owner
        self.claim_unmarked = claim_unmarked
        self.docker_bin = docker_bin
        self._container = container
        self._connect = connect

    @classmethod
    def of(
        cls, spec: CellServerSpec, *, connect: Connect = _asyncpg_connect
    ) -> CellDatabase:
        return cls(
            spec.pg,
            spec.db_name,
            owner=spec.database_owner,
            claim_unmarked=spec.name_bound_to_instance,
            connect=connect,
        )

    @property
    def name(self) -> str:
        return self.db_name

    async def container(self) -> str:
        if self._container is None:
            self._container = await postgres_container(docker_bin=self.docker_bin)
        return self._container

    async def check_available(self) -> None:
        """Postgres answers and its container is found (else :class:`MirixServerError`)."""
        try:
            await database_exists(self.pg, self.db_name, connect=self._connect)
        except Exception as exc:
            raise MirixServerError(
                f"the compose Postgres on {self.pg.host}:{self.pg.port} does not answer: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        await self.container()

    async def check_owner(
        self, *, claim_unmarked: bool = False, segment: int | None = None
    ) -> dict[str, Any]:
        """Prove that the live database and its ``<db>_dNN`` asides are this attempt's,
        before a rollback renames or drops any of them (read-only; B02). Returns
        ``{"database", "owner", "states": {name: state}, "claimed": [names]}``.

        A database another run marked (or one with a comment that is no marker) is a
        :class:`CellDatabaseConflict`, whatever ``claim_unmarked`` says. One without a
        marker is one too unless it is claimed (``claim_unmarked`` here or on the store:
        a person passed ``--force-resume``, or the name is the cell directory's own); the
        claimed ones are listed in ``claimed``. The exception is the live database next
        to segment ``segment``'s aside marked ours: that rollback's own half-made
        restore (a crash between its ``CREATE`` and its marker).

        Segment ``segment``'s aside without a marker next to a live database is refused
        even when claimed: the rollback would take the live database for its own
        half-made restore and drop it unexported, and export the aside (perhaps another
        runs directory's kept post-crash copy) in its place. Which of the two holds this
        attempt's memory only a person can tell."""
        conn = await _admin(self.pg, self._connect)
        try:
            comments = await _comments_on(conn, self.db_name, asides=True)
        finally:
            await conn.close()
        states = {
            name: owner_state(comment, self.owner) for name, comment in comments.items()
        }
        run_id = self.owner.run_id if self.owner is not None else None
        foreign = sorted(n for n, state in states.items() if state == OWNER_FOREIGN)
        if foreign:
            found = "; ".join(f"{n}: {describe_owner(comments[n])}" for n in foreign)
            raise CellDatabaseConflict(
                f"{found}. This attempt is run {run_id}: a learning resume never renames, "
                "restores over or drops another run's database, even with --force-resume. "
                "Two runs directories share this cell database name; start the cell over "
                "in a new attempt (--new-attempt), whose database has a name of its own"
            )
        aside = (
            discarded_db_name(self.db_name, segment) if segment is not None else None
        )
        if (
            aside is not None
            and states.get(aside) == OWNER_UNMARKED
            and self.db_name in states
        ):
            raise CellDatabaseConflict(
                f"{aside} carries no owner marker and has the name this rollback moves "
                f"the live database {self.db_name} to; with both present the rollback "
                f"would take {self.db_name} for its own half-made restore and drop it "
                f"without exporting it, and export {aside} in its place. This attempt "
                f"is run {run_id}. Not even --force-resume claims it: a person checks "
                f"whose {aside} is. Another runs directory's kept post-crash copy "
                "(--keep-discarded-db) is renamed or dropped by hand; this attempt's "
                "own, from a harness before owner markers, is compared with "
                f"{self.db_name} and the one that is not the post-crash database is "
                "dropped by hand; then the resume is started again"
            )
        half_made = (
            aside is not None
            and states.get(self.db_name) == OWNER_UNMARKED
            and states.get(aside) == OWNER_OURS
        )
        unproven = sorted(
            n
            for n, state in states.items()
            if state == OWNER_UNMARKED and not (half_made and n == self.db_name)
        )
        if unproven and not (claim_unmarked or self.claim_unmarked):
            raise CellDatabaseConflict(
                f"{', '.join(unproven)} carries no owner marker (made before owner "
                "markers) under a name another runs directory of this cell id computes "
                f"as well; this attempt is run {run_id}. A resume would move it aside and "
                "drop it, so a person first checks that it is this attempt's database "
                "(its sessions are this attempt's learning items), then passes "
                "--force-resume; otherwise --new-attempt"
            )
        return {
            "database": self.db_name,
            "owner": run_id,
            "states": dict(sorted(states.items())),
            "claimed": unproven,
        }

    async def connections(self) -> int:
        return await db_connections(self.pg, self.db_name, connect=self._connect)

    async def active_runs(self) -> int:
        if not await database_exists(self.pg, self.db_name, connect=self._connect):
            return 0
        conn = await _admin(self.pg, self._connect, self.db_name)
        try:
            return await _active_runs_on(conn)
        finally:
            await conn.close()

    async def checkpoint(self, out_dir: Path) -> dict[str, Any]:
        return await checkpoint_database(
            self.pg,
            self.db_name,
            out_dir / DUMP_FILE,
            container=await self.container(),
            docker_bin=self.docker_bin,
            connect=self._connect,
        )

    async def digest(self) -> dict[str, Any]:
        return await database_digest(self.pg, self.db_name, connect=self._connect)

    async def rollback(
        self,
        segment: int,
        checkpoint_dir: Path | None,
        export_path: Path,
        *,
        keep_discarded: bool = False,
        claim_unmarked: bool = False,
    ) -> dict[str, Any]:
        """Move the post-crash database aside (``<db>_dSS``), restore the checkpoint's
        dump as ``<db>`` (none before the first checkpoint), export what only the
        post-crash database held into ``export_path``, then drop it (unless
        ``keep_discarded``). Safe to repeat after a crash at any point: a restore found
        next to a post-crash database is dropped and done again, and an export already
        written is kept once the post-crash database is gone.

        Nothing is renamed or dropped before :meth:`check_owner` proved every database
        involved the attempt's (B02); a claimed database without a marker is marked
        before it is renamed, and the restored one right after its ``CREATE``. The
        ownership check is returned as ``owner``."""
        pg, connect = self.pg, self._connect
        owners = await self.check_owner(claim_unmarked=claim_unmarked, segment=segment)
        if self.owner is not None:
            for name in owners["claimed"]:
                await mark_database(pg, name, self.owner, connect=connect)
        aside = discarded_db_name(self.db_name, segment)
        live = self.db_name in owners["states"]
        moved = aside in owners["states"]
        done = await asyncio.to_thread(_read_export, export_path)
        if (
            live
            and not moved
            and done is not None
            and done.get("discarded_db") in (aside, None)
            and (done.get("live_db") == self.db_name) == (checkpoint_dir is not None)
        ):
            # An earlier try of this rollback finished (the post-crash database was
            # exported and dropped) and died before its segment was sealed: the live
            # database is the restored one.
            return {
                **export_summary(done),
                "post_crash_db": "dropped",
                "owner": owners,
            }
        if live and moved:
            await drop_cell_database(pg, self.db_name, connect=connect)
        elif live:
            await rename_database(pg, self.db_name, aside, connect=connect)
            moved = True
        if checkpoint_dir is not None:
            await restore_database(
                pg,
                self.db_name,
                checkpoint_dir / DUMP_FILE,
                container=await self.container(),
                docker_bin=self.docker_bin,
                owner=self.owner,
                connect=connect,
            )
        if moved:
            export = await export_discarded_rows(
                pg,
                aside,
                self.db_name if checkpoint_dir is not None else None,
                connect=connect,
            )
            await asyncio.to_thread(_write_json_file, export_path, export)
        elif await asyncio.to_thread(export_path.is_file):
            export = json.loads(await asyncio.to_thread(export_path.read_text, "utf-8"))
        else:
            export = {"discarded_db": None, "live_db": None, "tables": {}, "rows": {}}
            await asyncio.to_thread(_write_json_file, export_path, export)
        post_crash = "none"
        if moved:
            if keep_discarded:
                post_crash = f"kept:{aside}"
            else:
                await drop_cell_database(pg, aside, connect=connect)
                post_crash = "dropped"
        elif export.get("discarded_db"):
            post_crash = "dropped"
        return {**export_summary(export), "post_crash_db": post_crash, "owner": owners}


def _read_export(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class NoDatabase:
    """The :class:`CheckpointStore` of a MIRIX server run without a cell database
    (``MirixServer(database=False)``, tests with a stub server): nothing to dump."""

    def __init__(self, db_name: str) -> None:
        self._name = db_name

    @property
    def name(self) -> str:
        return self._name

    async def check_available(self) -> None:
        return None

    async def check_owner(
        self, *, claim_unmarked: bool = False, segment: int | None = None
    ) -> dict[str, Any]:
        return {"database": None, "owner": None, "states": {}, "claimed": []}

    async def connections(self) -> int:
        return 0

    async def active_runs(self) -> int:
        return 0

    async def checkpoint(self, out_dir: Path) -> dict[str, Any]:
        return {
            "db_name": None,
            "dump_sha256": None,
            "dump_bytes": 0,
            "db_digest": {},
            "active_runs": 0,
            "snapshot_id": None,
        }

    async def digest(self) -> dict[str, Any]:
        return {}

    async def rollback(
        self,
        segment: int,
        checkpoint_dir: Path | None,
        export_path: Path,
        *,
        keep_discarded: bool = False,
        claim_unmarked: bool = False,
    ) -> dict[str, Any]:
        export = {"discarded_db": None, "live_db": None, "tables": {}, "rows": {}}
        await asyncio.to_thread(_write_json_file, export_path, export)
        return {**export_summary(export), "post_crash_db": "none"}


def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, sort_keys=True, indent=1, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Learning checkpoints: HOME
# ---------------------------------------------------------------------------

#: HOME files MIRIX writes once and never changes (its stored images, named by their
#: sha256): a checkpoint hard-links them. Every other file (``.mirix/config``) is copied.
HOME_IMAGE_DIRS: tuple[str, ...] = (f"{MIRIX_STATE_DIR}/images",)
#: HOME directories a checkpoint leaves out (MIRIX's scratch files).
HOME_IGNORED_DIRS: tuple[str, ...] = (f"{MIRIX_STATE_DIR}/tmp",)
HOME_MANIFEST_FILE = "home_manifest.json"
HOME_FILES_DIR = "home_files"


def _under(rel: str, prefixes: Sequence[str]) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in prefixes)


def home_files(home: Path) -> list[str]:
    """Every file under ``home`` (relative POSIX paths, sorted), symlinked directories
    followed (a linked ``.mirix``, :func:`link_nested_mirix_home`)."""
    found = []
    if not home.is_dir():
        return found
    for dirpath, _dirs, files in os.walk(home, followlinks=True):
        for name in files:
            found.append((Path(dirpath) / name).relative_to(home).as_posix())
    return sorted(found)


def home_manifest(
    home: Path, *, previous: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """HOME's files but ``.mirix/tmp``: ``{"files": {rel: {"size", "mtime_ns", "sha256",
    "kind"}}}`` (``kind`` ``image`` for MIRIX's stored images, else ``file``). A file
    whose size and mtime equal its entry in ``previous`` keeps that sha256, so only new
    or changed files are read (blocking)."""
    known = (previous or {}).get("files") or {}
    files: dict[str, dict[str, Any]] = {}
    for rel in home_files(home):
        if _under(rel, HOME_IGNORED_DIRS):
            continue
        stat = (home / rel).stat()
        entry = known.get(rel)
        if (
            isinstance(entry, Mapping)
            and entry.get("size") == stat.st_size
            and entry.get("mtime_ns") == stat.st_mtime_ns
        ):
            sha = str(entry["sha256"])
        else:
            sha = _file_sha256(home / rel)[0]
        files[rel] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha,
            "kind": "image" if _under(rel, HOME_IMAGE_DIRS) else "file",
        }
    return {"files": files}


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def capture_home(home: Path, manifest: Mapping[str, Any], out_dir: Path) -> None:
    """Keep HOME's files in a checkpoint (``out_dir/home_files``): images hard-linked
    (write-once), every other file copied (blocking)."""
    for rel, entry in (manifest.get("files") or {}).items():
        dest = out_dir / HOME_FILES_DIR / rel
        if entry.get("kind") == "image":
            _link_or_copy(home / rel, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(home / rel, dest)


def _home_file_matches(path: Path, entry: Mapping[str, Any]) -> bool:
    size = path.stat().st_size
    if size != entry.get("size"):
        return False
    if entry.get("kind") == "image":
        return True  # write-once, named by its sha256
    return _file_sha256(path)[0] == entry.get("sha256")


def move_new_home_files(
    home: Path, manifest: Mapping[str, Any] | None, dest: Path
) -> list[str]:
    """Move every HOME file the checkpoint's ``manifest`` does not list, or lists with
    other content (``.mirix/tmp`` included), to ``dest`` (same relative paths), so the
    discarded segment keeps what it wrote; with no manifest (a rollback to before the
    first checkpoint), HOME's whole content, directories too, so a fresh MIRIX finds it
    empty. Returns the moved files (blocking; safe to repeat)."""
    moved: list[str] = []
    if not home.is_dir():
        return moved
    if manifest is None:
        for child in sorted(home.iterdir()):
            target = dest / child.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target = dest / f"{child.name}.{uuid.uuid4().hex[:8]}"
            if child.is_dir():
                moved += [f"{child.name}/{rel}" for rel in home_files(child)]
            else:
                moved.append(child.name)
            shutil.move(str(child), str(target))
        return moved
    listed = manifest.get("files") or {}
    for rel in home_files(home):
        entry = listed.get(rel)
        if entry is not None and _home_file_matches(home / rel, entry):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(home / rel), str(target))
        moved.append(rel)
    return moved


def restore_home(
    home: Path, checkpoint_dir: Path, manifest: Mapping[str, Any]
) -> list[str]:
    """Put HOME back as ``manifest`` recorded it from the checkpoint's ``home_files``:
    every file copied back (an image linked back when it is missing), then checked
    against its sha256. Returns the problems (empty: restored exactly; blocking)."""
    problems: list[str] = []
    for rel, entry in sorted((manifest.get("files") or {}).items()):
        target = home / rel
        source = checkpoint_dir / HOME_FILES_DIR / rel
        if entry.get("kind") == "image" and target.is_file():
            pass  # write-once: checked below
        elif source.is_file():
            if target.exists() or target.is_symlink():
                target.unlink()
            if entry.get("kind") == "image":
                _link_or_copy(source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        if not target.is_file():
            problems.append(f"{rel}: missing")
            continue
        if _file_sha256(target)[0] != entry.get("sha256"):
            problems.append(f"{rel}: sha256 differs from the checkpoint's")
    return problems


# ---------------------------------------------------------------------------
# CLI: link the nested MIRIX state of attempts that ran with a relative HOME
# ---------------------------------------------------------------------------


def link_nested_homes(
    paths: Sequence[Path], *, dry_run: bool = False
) -> list[dict[str, Any]]:
    """:func:`link_nested_mirix_home` for every attempt ``paths`` name (blocking): one row
    per attempt with nested MIRIX state, whose ``status`` is ``linked`` (``would_link``
    with ``dry_run``), ``home_has_mirix`` (left alone) or ``ambiguous`` (several nested
    directories: link the right one by hand)."""
    rows: list[dict[str, Any]] = []
    for attempt in (a for path in paths for a in attempt_dirs(path)):
        nested = nested_mirix_dirs(attempt)
        if not nested:
            continue
        link = attempt / "home" / MIRIX_STATE_DIR
        row: dict[str, Any] = {
            "attempt": str(attempt),
            "nested": [os.path.relpath(n, attempt) for n in nested],
        }
        if link.is_symlink() or link.exists():
            row["status"] = "home_has_mirix"
            row["link"] = os.readlink(link) if link.is_symlink() else None
        elif len(nested) > 1:
            row["status"] = "ambiguous"
        else:
            link_nested_mirix_home(attempt, dry_run=dry_run)
            row["status"] = "would_link" if dry_run else "linked"
            row["link"] = os.path.relpath(nested[0], link.parent)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI: mark the cell databases of attempts from before owner markers (B02)
# ---------------------------------------------------------------------------


def _recorded_databases(
    paths: Sequence[Path],
) -> dict[str, list[tuple[Path, dict[str, Any], str]]]:
    """``{db_name: [(attempt dir, attempt.json, cell id)]}`` of every attempt ``paths``
    name that recorded a MIRIX database (blocking)."""
    found: dict[str, list[tuple[Path, dict[str, Any], str]]] = {}
    seen: set[Path] = set()
    for attempt in (a for path in paths for a in attempt_dirs(path)):
        if attempt.resolve() in seen:
            continue
        seen.add(attempt.resolve())
        try:
            state = json.loads((attempt / "attempt.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        mirix = state.get("mirix") if isinstance(state, dict) else None
        db_name = mirix.get("db_name") if isinstance(mirix, dict) else None
        if not isinstance(db_name, str) or not db_name:
            continue
        cell_id = attempt.parent.name
        try:
            manifest = json.loads(
                (attempt.parent / "cell.json").read_text(encoding="utf-8")
            )
            cell_id = str(manifest["cell"]["cell_id"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        found.setdefault(db_name, []).append((attempt, state, cell_id))
    return found


async def mark_owners(
    paths: Sequence[Path],
    *,
    dry_run: bool = False,
    host: Mapping[str, Any] | None = None,
    pg: PostgresAdmin | None = None,
    connect: Connect = _asyncpg_connect,
) -> list[dict[str, Any]]:
    """Mark the databases the attempts in ``paths`` (runs, cell or attempt directories)
    recorded, and their rollbacks' kept ``<db>_dNN`` asides, for databases made before
    owner markers (B02). One row per recorded name (and per aside found, with
    ``aside_of``):

    * ``marked`` (``would_mark`` with ``dry_run``): exactly one attempt across ``paths``
      records the name, it ran on this host, and the database carries no marker; it is
      marked as that attempt's (its run id, cell id, attempt and instance);
    * ``ambiguous``: several attempts record the name (``attempts``; the drill/drill2
      shape): left for a person, who knows which runs directory made it;
    * ``other_host``: the attempt ran on another host (``attempt.json`` ``host``; e.g.
      a runs directory copied or merged from another machine, whose names this
      Postgres may hold for an attempt of its own) or recorded none: left alone;
    * ``already_marked``, ``foreign`` (another run's marker, or a comment that is no
      marker: left alone), ``absent`` (no such database), ``unreadable`` (the attempt
      records no run id).

    ``host`` is this host (:func:`~evals.skill_bench.cell_records.host_info` by
    default). Only a person runs this, on the runs directories that share the
    Postgres; nothing marks an older database automatically. A wrong mark is removed
    with :func:`unmark_owner` (``unmark-owner``)."""
    from evals.skill_bench.cell_records import host_info, host_text, same_host

    pg = pg or PostgresAdmin()
    here = host if host is not None else await asyncio.to_thread(host_info)
    recorded = await asyncio.to_thread(_recorded_databases, paths)
    rows: list[dict[str, Any]] = []
    for db_name, attempts in sorted(recorded.items()):
        row: dict[str, Any] = {"db_name": db_name}
        if len(attempts) > 1:
            rows.append(
                {
                    **row,
                    "status": "ambiguous",
                    "attempts": [str(a) for a, _, _ in attempts],
                }
            )
            continue
        ((attempt, state, cell_id),) = attempts
        row["attempt"] = str(attempt)
        run_id, number = state.get("run_id"), state.get("attempt")
        instance = state.get("instance")
        if not isinstance(run_id, str) or not run_id or not isinstance(number, int):
            rows.append({**row, "status": "unreadable"})
            continue
        owner = DatabaseOwner(
            run_id, cell_id, number, instance if isinstance(instance, str) else None
        )
        row["run_id"] = run_id
        if not DB_NAME_RE.fullmatch(db_name):
            rows.append({**row, "status": "unreadable"})
            continue
        if not same_host(state.get("host"), here):
            rows.append(
                {
                    **row,
                    "status": "other_host",
                    "found": host_text(state.get("host")),
                }
            )
            continue
        conn = await _admin(pg, connect)
        try:
            comments = await _comments_on(conn, db_name, asides=True)
        finally:
            await conn.close()
        if db_name not in comments:
            rows.append({**row, "status": "absent"})
        for name in sorted(comments):
            if name != db_name and name in recorded:
                continue  # an attempt records it under its own name: its own row
            entry = {**row, "db_name": name}
            if name != db_name:
                entry["aside_of"] = db_name
            state_now = owner_state(comments[name], owner)
            if state_now == OWNER_OURS:
                rows.append({**entry, "status": "already_marked"})
            elif state_now == OWNER_FOREIGN:
                rows.append(
                    {
                        **entry,
                        "status": "foreign",
                        "found": describe_owner(comments[name]),
                    }
                )
            elif dry_run:
                rows.append({**entry, "status": "would_mark"})
            else:
                await mark_database(pg, name, owner, connect=connect)
                rows.append({**entry, "status": "marked"})
    return rows


async def unmark_owner(
    db_name: str,
    run_id: str,
    *,
    dry_run: bool = False,
    pg: PostgresAdmin | None = None,
    connect: Connect = _asyncpg_connect,
) -> dict[str, Any]:
    """Remove the owner marker of ``db_name`` when it names ``run_id`` (a person's
    correction of a wrong ``mark-owners``): ``unmarked`` (``would_unmark``), or
    ``absent``, ``unmarked_already`` or ``foreign`` (the marker names another run, or
    the comment is no marker: left alone)."""
    _check_db_name(db_name)
    pg = pg or PostgresAdmin()
    conn = await _admin(pg, connect)
    try:
        comments = await _comments_on(conn, db_name)
        row: dict[str, Any] = {"db_name": db_name, "run_id": run_id}
        if db_name not in comments:
            return {**row, "status": "absent"}
        if comments[db_name] is None:
            return {**row, "status": "unmarked_already"}
        marker = marker_of(comments[db_name])
        if marker is None or marker.get("run_id") != run_id:
            return {
                **row,
                "status": "foreign",
                "found": describe_owner(comments[db_name]),
            }
        if dry_run:
            return {**row, "status": "would_unmark"}
        sql = await conn.fetchval(
            "SELECT format('COMMENT ON DATABASE %I IS NULL', $1::text)", db_name
        )
        await conn.execute(sql)
        return {**row, "status": "unmarked"}
    finally:
        await conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.skill_bench.infra.mirix_server",
        description="Per-cell MIRIX server tools.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    mark = commands.add_parser(
        "mark-owners",
        help="mark the cell databases that attempts recorded before owner markers "
        "existed (B02) with their attempt, where exactly one attempt records the name",
    )
    mark.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="every runs directory (or cell or attempt directory) using this Postgres",
    )
    mark.add_argument(
        "--dry-run", action="store_true", help="show what would be marked"
    )
    unmark = commands.add_parser(
        "unmark-owner",
        help="remove the owner marker of a cell database when it names the given run "
        "(a person's correction of a wrong mark-owners)",
    )
    unmark.add_argument("db_name", help="the database whose marker is removed")
    unmark.add_argument("--run-id", required=True, help="the run its marker must name")
    unmark.add_argument(
        "--dry-run", action="store_true", help="show what would be unmarked"
    )
    link = commands.add_parser(
        "link-nested-home",
        help="link the MIRIX state an attempt wrote under a relative HOME (before cell "
        "directories were made absolute) to its documented home/.mirix, so its test "
        "phase can resume",
    )
    link.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="runs directories, cell directories or attempt directories",
    )
    link.add_argument(
        "--dry-run", action="store_true", help="show what would be linked"
    )
    args = parser.parse_args(argv)
    if args.command == "unmark-owner":
        if not DB_NAME_RE.fullmatch(args.db_name):
            parser.error(f"not a cell database name: {args.db_name!r}")
        # A command-line tool of its own (never inside a running server's loop).
        row = asyncio.run(unmark_owner(args.db_name, args.run_id, dry_run=args.dry_run))
        print(json.dumps(row))
        return 0 if row["status"] in ("unmarked", "would_unmark") else 1
    missing = [str(p) for p in args.paths if not p.is_dir()]
    if missing:
        parser.error(f"not a directory: {', '.join(missing)}")
    if args.command == "mark-owners":
        # A command-line tool of its own (never inside a running server's loop).
        rows = asyncio.run(mark_owners(args.paths, dry_run=args.dry_run))
        for row in rows:
            print(json.dumps(row))
        left = ("ambiguous", "foreign", "unreadable", "other_host")
        return 1 if any(row["status"] in left for row in rows) else 0
    rows = link_nested_homes(args.paths, dry_run=args.dry_run)
    for row in rows:
        print(json.dumps(row))
    return 1 if any(row["status"] == "ambiguous" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
