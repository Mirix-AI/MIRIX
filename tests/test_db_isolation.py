"""The unit-test isolation guards themselves (tests/db_isolation.py).

* the real-database guard recognizes the invoking user's REAL ``~/.mirix``
  (from the password database, not ``$HOME``) and refuses DDL there before any
  statement runs (the guard test_multi_scope_access relies on);
* ``isolated_database`` initializes a fresh database like server startup and
  restores MIRIX's global engine afterwards (test_local_client);
* the network guard lets loopback through and refuses everything else;
* ``isolated_database`` refuses a configured Postgres that is not a throwaway
  ``mirix_it_*`` database the PG runner created, e.g. a developer database a
  ``.env`` names.
"""

import socket
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

import db_isolation
from db_isolation import (
    ALLOW_CONFIGURED_PG,
    block_non_loopback_network,
    configured_pg_database,
    create_schema_unless_real,
    is_real_mirix_database,
    isolated_database,
    real_mirix_dir,
    refuse_configured_postgres,
    refuse_real_mirix_database,
)


def _fake_engine(dialect: str, database):
    return SimpleNamespace(
        dialect=SimpleNamespace(name=dialect), url=SimpleNamespace(database=database)
    )


class TestRealDatabaseRecognition:
    def test_real_dir_ignores_home(self, monkeypatch, tmp_path):
        before = real_mirix_dir()
        monkeypatch.setenv("HOME", str(tmp_path))
        assert real_mirix_dir() == before
        assert tmp_path not in real_mirix_dir().parents

    def test_sqlite_inside_the_real_dir_is_real(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db_isolation, "real_mirix_dir", lambda: tmp_path)
        assert is_real_mirix_database(
            _fake_engine("sqlite", str(tmp_path / "sqlite.db"))
        )
        assert is_real_mirix_database(
            _fake_engine("sqlite", str(tmp_path / "nested" / "x.db"))
        )

    @pytest.mark.parametrize(
        "engine",
        [
            _fake_engine("sqlite", "/somewhere/else/sqlite.db"),
            _fake_engine("sqlite", ":memory:"),
            _fake_engine("sqlite", None),
            _fake_engine("postgresql", "mirix"),
        ],
    )
    def test_everything_else_is_not(self, monkeypatch, tmp_path, engine):
        monkeypatch.setattr(db_isolation, "real_mirix_dir", lambda: tmp_path)
        assert not is_real_mirix_database(engine)

    def test_refusal_is_a_skip_naming_the_database(self, monkeypatch, tmp_path):
        monkeypatch.setattr(db_isolation, "real_mirix_dir", lambda: tmp_path)
        with pytest.raises(pytest.skip.Exception, match="real MIRIX database"):
            refuse_real_mirix_database(
                _fake_engine("sqlite", str(tmp_path / "sqlite.db"))
            )
        refuse_real_mirix_database(_fake_engine("sqlite", "/elsewhere/sqlite.db"))


@pytest.mark.asyncio
class TestSchemaGuard:
    async def test_no_ddl_reaches_the_real_database(self, monkeypatch, tmp_path):
        """The multi-scope suite's schema fixture refuses BEFORE create_all."""
        monkeypatch.setattr(db_isolation, "real_mirix_dir", lambda: tmp_path)
        db_file = tmp_path / "sqlite.db"
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_file}")
        try:
            with pytest.raises(pytest.skip.Exception):
                await create_schema_unless_real(engine)
        finally:
            await engine.dispose()
        assert (
            not db_file.exists()
            or not sqlite3.connect(db_file)
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        )

    async def test_an_isolated_database_gets_the_schema(self, tmp_path):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'iso.db'}")
        try:
            await create_schema_unless_real(engine)
        finally:
            await engine.dispose()
        tables = {
            row[0]
            for row in sqlite3.connect(tmp_path / "iso.db").execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"users", "organizations", "conversation_session"} <= tables


@pytest.mark.asyncio(loop_scope="function")
async def test_isolated_database_initializes_like_startup_and_restores(tmp_path):
    import mirix.server.server as server_module
    from mirix.services.organization_manager import OrganizationManager
    from mirix.services.user_manager import UserManager

    saved_engine = server_module.engine
    saved_factory = server_module.AsyncSessionLocal
    async with isolated_database(tmp_path / "db") as engine:
        assert server_module.engine is engine is not saved_engine
        assert Path(engine.url.database).parent == tmp_path / "db"
        # The defaults server startup creates exist in the fresh database.
        org = await OrganizationManager().get_organization_by_id(
            OrganizationManager.DEFAULT_ORG_ID
        )
        assert org.id == OrganizationManager.DEFAULT_ORG_ID
        admin = await UserManager().get_user_by_id(UserManager.ADMIN_USER_ID)
        assert admin.id == UserManager.ADMIN_USER_ID
    assert server_module.engine is saved_engine
    assert server_module.AsyncSessionLocal is saved_factory


class TestNetworkGuard:
    def test_loopback_is_allowed_and_everything_else_refused(self, monkeypatch):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        accepted = []
        thread = threading.Thread(target=lambda: accepted.append(listener.accept()))
        thread.start()
        block_non_loopback_network(monkeypatch)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                pass
            thread.join(timeout=5)
            assert accepted
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                with pytest.raises(ConnectionRefusedError, match="offline test"):
                    probe.connect(("10.255.255.1", 443))
                with pytest.raises(ConnectionRefusedError, match="offline test"):
                    probe.connect_ex(("10.255.255.1", 443))
            finally:
                probe.close()
            with pytest.raises(socket.gaierror, match="offline test"):
                socket.getaddrinfo("api.openai.com", 443)
            assert socket.getaddrinfo("localhost", port)
        finally:
            for conn, _ in accepted:
                conn.close()
            listener.close()


class TestConfiguredPostgresGuard:
    @pytest.fixture
    def configured(self, monkeypatch):
        """Configure MIRIX's settings for a Postgres by URI or by parts."""
        from mirix.settings import settings

        monkeypatch.delenv(ALLOW_CONFIGURED_PG, raising=False)
        for field in (
            "pg_uri",
            "pg_db",
            "pg_user",
            "pg_password",
            "pg_host",
            "pg_port",
        ):
            monkeypatch.setattr(settings, field, None)

        def configure(**fields):
            for field, value in fields.items():
                monkeypatch.setattr(settings, field, value)

        return configure

    def test_no_postgres_configured_is_allowed(self, configured):
        configured()
        assert configured_pg_database() is None
        refuse_configured_postgres()

    def test_the_pg_runners_throwaway_database_is_allowed(self, configured):
        configured(
            pg_uri="postgresql+pg8000://test:test@127.0.0.1:5433/mirix_it_0a1b2c3d4e5f"
        )
        assert configured_pg_database() == "mirix_it_0a1b2c3d4e5f"
        refuse_configured_postgres()

    @pytest.mark.parametrize(
        "fields",
        [
            # docker/env.example's database, by parts as a copied .env sets them.
            {
                "pg_db": "mirix",
                "pg_user": "mirix",
                "pg_password": "mirix",
                "pg_host": "localhost",
                "pg_port": 5432,
            },
            {"pg_uri": "postgresql+pg8000://u:p@127.0.0.1:5432/mirix"},
            {"pg_uri": "postgresql+pg8000://u:p@127.0.0.1:5432/"},
        ],
    )
    def test_any_other_configured_database_is_refused(self, configured, fields):
        configured(**fields)
        with pytest.raises(pytest.skip.Exception, match="refusing to run"):
            refuse_configured_postgres()

    def test_an_explicit_opt_in_allows_the_configured_database(
        self, configured, monkeypatch
    ):
        configured(pg_uri="postgresql+pg8000://u:p@127.0.0.1:5432/mirix")
        monkeypatch.setenv(ALLOW_CONFIGURED_PG, "1")
        refuse_configured_postgres()

    @pytest.mark.asyncio(loop_scope="function")
    async def test_isolated_database_refuses_before_touching_the_database(
        self, configured, tmp_path
    ):
        import mirix.server.server as server_module

        configured(pg_uri="postgresql+pg8000://u:p@127.0.0.1:1/mirix")
        saved_engine = server_module.engine
        with pytest.raises(pytest.skip.Exception, match="'mirix'"):
            async with isolated_database(tmp_path / "db"):
                pytest.fail("isolated_database ran on the configured database")
        assert server_module.engine is saved_engine
        assert not (tmp_path / "db").exists()
