"""Keep unit tests off the developer's real MIRIX database and off the network.

MIRIX binds one global async engine when ``mirix.server.server`` is imported:
Postgres when ``MIRIX_PG_URI`` is set, otherwise SQLite at
``$HOME/.mirix/sqlite.db``. A unit run with an isolated HOME (``HOME=$(mktemp -d)
pytest ...``, with the model-provider API keys unset so no test can make a paid
call) therefore gets a throwaway database — but a developer who runs
``pytest -m "not integration"`` from a normal shell would point the tests at
their REAL ``~/.mirix/sqlite.db``. The helpers here make that impossible:

* :func:`refuse_real_mirix_database` skips a test module before it runs DDL or
  writes against the invoking user's real ``~/.mirix`` database (resolved from
  the password database, so an isolated ``$HOME`` is recognized as isolated);
* :func:`isolated_database` points MIRIX's global engine at a fresh SQLite file
  for a module and initializes it the way server startup does
  (``ensure_tables_created`` + ``AsyncServer.ensure_defaults``); with a
  Postgres configured it uses it only when it is a throwaway database the PG
  runner created (:func:`refuse_configured_postgres`);
* :func:`block_non_loopback_network` makes any connection (or DNS lookup) to a
  host other than loopback fail loudly, so an "offline" suite cannot silently
  reach a model provider.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pytest

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def real_mirix_dir() -> Path:
    """The invoking user's real ``~/.mirix``, independent of ``$HOME``."""
    home: Optional[str] = None
    try:
        import pwd

        home = pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError, AttributeError):  # non-POSIX: best effort
        home = None
    return Path(home or os.path.expanduser("~"), ".mirix").resolve()


def sqlite_database_path(engine: Any) -> Optional[Path]:
    """The resolved file of a SQLite engine; None for other dialects/in-memory."""
    if getattr(getattr(engine, "dialect", None), "name", None) != "sqlite":
        return None
    database = getattr(getattr(engine, "url", None), "database", None)
    if not database or database == ":memory:":
        return None
    return Path(database).expanduser().resolve()


def is_real_mirix_database(engine: Any) -> bool:
    """True when ``engine`` is a SQLite database inside the real ``~/.mirix``."""
    path = sqlite_database_path(engine)
    if path is None:
        return False
    real = real_mirix_dir()
    return path == real or real in path.parents


#: The database name prefix of the throwaway Postgres databases
#: ``scripts/run_pg_integration_tests.sh`` creates (and drops) per run.
THROWAWAY_PG_DATABASE_PREFIX = "mirix_it_"
#: Opt-in for a module to run on another configured Postgres anyway.
ALLOW_CONFIGURED_PG = "MIRIX_TEST_ALLOW_CONFIGURED_PG"


def configured_pg_database() -> Optional[str]:
    """The database name of the Postgres MIRIX is configured for (``MIRIX_PG_URI``
    or the ``MIRIX_PG_*`` parts, which a ``.env`` MIRIX loads on import may set),
    None without one."""
    from sqlalchemy.engine import make_url

    from mirix.settings import settings

    uri = settings.mirix_pg_uri_no_default
    if not uri:
        return None
    return make_url(uri).database or ""


def refuse_configured_postgres(*, what: str = "DDL and test writes") -> None:
    """Skip the calling test (module) when MIRIX is configured for a Postgres that
    is not a throwaway one (its name starts with ``mirix_it_``): a developer's
    database, named by a ``.env`` copied from ``docker/env.example``, must not
    get test DDL, default rows and fake memories. Set
    ``MIRIX_TEST_ALLOW_CONFIGURED_PG=1`` to run on it deliberately."""
    name = configured_pg_database()
    if name is None or name.startswith(THROWAWAY_PG_DATABASE_PREFIX):
        return
    if os.environ.get(ALLOW_CONFIGURED_PG) == "1":
        return
    pytest.skip(
        f"refusing to run {what} against the configured Postgres database "
        f"{name!r}: use scripts/run_pg_integration_tests.sh (a throwaway "
        f"{THROWAWAY_PG_DATABASE_PREFIX}* database), unset MIRIX_PG_URI / "
        f"MIRIX_PG_* (a .env file may set them), or set {ALLOW_CONFIGURED_PG}=1"
    )


def refuse_real_mirix_database(engine: Any, *, what: str = "DDL and writes") -> None:
    """Skip the calling test (module, from a module fixture) on the real database."""
    if is_real_mirix_database(engine):
        pytest.skip(
            f"refusing to run {what} against the real MIRIX database "
            f"{sqlite_database_path(engine)}; run the unit suite with an isolated "
            "HOME (HOME=$(mktemp -d) pytest ...) and the model-provider API "
            "keys unset"
        )


async def create_schema_unless_real(engine: Any) -> None:
    """``create_all`` on ``engine`` — after refusing (skipping) the real database.

    For modules that run on the global engine (the isolated ``$HOME`` database
    of a documented unit run): the guard comes first, so not even the DDL
    reaches a developer's real ``~/.mirix/sqlite.db``.
    """
    from mirix.orm.base import Base

    refuse_real_mirix_database(engine, what="DDL and test writes")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def isolated_database(directory: Path) -> AsyncIterator[Any]:
    """A fresh, fully initialized MIRIX database for the duration of the block.

    Without ``MIRIX_PG_URI`` the global engine (and the session factory every
    manager's ``db_context`` uses) is swapped for a new SQLite file under
    ``directory`` and restored afterwards, so the block never touches whatever
    database ``$HOME`` points at. With ``MIRIX_PG_URI`` set (the ORM then has
    Postgres-only columns, so SQLite cannot host it) the configured database is
    used — the PG runner script creates a fresh one per run — and only such a
    throwaway one (:func:`refuse_configured_postgres` skips the module
    otherwise).

    Either way the database is initialized like server startup: tables are
    created with the schema-drift guard (``ensure_tables_created``) and the
    default organization, admin user, default client and base tools exist
    (``AsyncServer.ensure_defaults``).
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    import mirix.server.server as server_module
    from mirix.settings import settings

    refuse_configured_postgres()
    swapped = None
    if not settings.mirix_pg_uri_no_default:
        directory.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{directory / 'mirix.db'}",
            connect_args={"timeout": 30},
        )
        swapped = (server_module.engine, server_module.AsyncSessionLocal)
        server_module.engine = engine
        server_module.AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    else:
        engine = server_module.engine
    try:
        await server_module.ensure_tables_created()
        await server_module.AsyncServer().ensure_defaults()
        yield engine
    finally:
        if swapped is not None:
            server_module.engine, server_module.AsyncSessionLocal = swapped
            await engine.dispose()


def _is_loopback_host(host: Any) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port): the local wildcard/loopback
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0].lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def block_non_loopback_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any socket connect or DNS lookup that is not to loopback.

    Unix sockets and loopback (the fake LLM server) keep working; anything else
    raises ``ConnectionRefusedError`` naming the host, so a test that would
    reach a real model provider fails loudly instead of spending money.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _check(address: Any) -> None:
        if isinstance(address, (str, bytes)):
            return  # AF_UNIX path
        if isinstance(address, tuple) and address and _is_loopback_host(address[0]):
            return
        raise ConnectionRefusedError(
            f"offline test: refusing a network connection to {address!r}"
        )

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        _check(address)
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        _check(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_loopback_host(host):
            raise socket.gaierror(f"offline test: refusing to resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
