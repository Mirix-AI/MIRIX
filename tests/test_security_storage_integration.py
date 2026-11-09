"""Destructive tests for a DISPOSABLE database named mirix_security_test.

Set MIRIX_SECURITY_TEST_PG_URI (asyncpg), MIRIX_SECURITY_TEST_PG_BIN and
MIRIX_SECURITY_TEST_NEO4J_URI/PASSWORD to run; no production settings are used.
"""

import json
import os
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


pytestmark = [pytest.mark.integration]


@pytest_asyncio.fixture
async def pg(monkeypatch):
    uri = os.environ.get("MIRIX_SECURITY_TEST_PG_URI")
    if not uri:
        pytest.skip("Disposable PostgreSQL is not configured")
    url = make_url(uri)
    assert url.database == "mirix_security_test"
    engine = create_async_engine(uri)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for table in ("episodic_memory", "semantic_memory"):
            await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            await conn.execute(text(f"""
                CREATE TABLE {table} (
                    id text PRIMARY KEY, user_id text, organization_id text,
                    summary text, details text, actor text, name text, source text,
                    source_refs jsonb NOT NULL DEFAULT '[]', occurred_at timestamp,
                    created_at timestamp DEFAULT now(), is_deleted boolean DEFAULT false,
                    summary_embedding vector
                )
            """))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def context():
        async with factory() as session:
            yield session

    from mirix.server import server
    monkeypatch.setattr(server, "db_context", context)
    yield engine, context, url
    await engine.dispose()


@pytest.fixture
def graph():
    uri = os.environ.get("MIRIX_SECURITY_TEST_NEO4J_URI")
    password = os.environ.get("MIRIX_SECURITY_TEST_NEO4J_PASSWORD")
    if not uri or not password:
        pytest.skip("Disposable Neo4j is not configured")
    GraphDatabase = pytest.importorskip("neo4j").GraphDatabase
    driver = GraphDatabase.driver(uri, auth=("neo4j", password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    yield driver
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
    driver.close()


@pytest.fixture
def snapshot_env(pg, graph, tmp_path, monkeypatch):
    from evals import memory_snapshot as snapshot

    _, _, url = pg
    monkeypatch.setattr(snapshot, "SNAP_ROOT", tmp_path / "snapshots")
    monkeypatch.setattr(snapshot.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(snapshot, "PG_BIN", os.environ["MIRIX_SECURITY_TEST_PG_BIN"])
    monkeypatch.setattr(snapshot, "PG", {
        "host": url.host, "port": str(url.port), "user": url.username,
        "password": url.password, "db": url.database,
    })
    monkeypatch.setattr(snapshot, "NEO4J", {
        "uri": os.environ["MIRIX_SECURITY_TEST_NEO4J_URI"],
        "user": "neo4j", "password": os.environ["MIRIX_SECURITY_TEST_NEO4J_PASSWORD"],
        "database": "neo4j",
    })
    return snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["episodic", "semantic"])
@pytest.mark.parametrize("rerank", [False, True])
async def test_graph_fetch_hides_deleted_pg_rows(pg, kind, rerank):
    from mirix.constants import MAX_EMBEDDING_DIM
    from mirix.services.graph_retriever_v7 import V7Retriever

    engine, _, _ = pg
    embedding = json.dumps([1.0] + [0.0] * (MAX_EMBEDDING_DIM - 1))
    async with engine.begin() as conn:
        for mid, deleted in (("live", False), ("deleted", True)):
            await conn.execute(text(f"""
                INSERT INTO {kind}_memory (id, user_id, summary, details, is_deleted, summary_embedding)
                VALUES (:id, 'u', :summary, 'details', :deleted, CAST(:embedding AS vector))
            """), {"id": mid, "summary": mid + "-text", "deleted": deleted, "embedding": embedding})
    retriever = V7Retriever()
    fetch = retriever._fetch_episodic if kind == "episodic" else retriever._fetch_semantic
    rows = await fetch("u", ["live", "deleted"], q_emb=[1.0] if rerank else None)
    assert [row.id for row in rows] == ["live"]
    assert await retriever._live_memory_ids("u", ["live", "deleted"]) == {"live"}
    assert await retriever._live_memory_ids("another-user", ["live", "deleted"]) == set()


@pytest.mark.asyncio
async def test_real_snapshot_temporal_properties_round_trip(pg, graph, snapshot_env):
    from neo4j.time import DateTime, Duration

    engine, _, _ = pg
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO semantic_memory(id, user_id, summary) VALUES ('one', 'u', 'original')"
        ))
    with graph.session() as session:
        session.run(
            "CREATE (a:SnapshotTest {name:'original', changed: datetime('2026-11-09T12:30:01.123456789Z'), "
            "elapsed: duration('P2M3DT4.000000005S')}) "
            "CREATE (a)-[:`SEEN``AT` {day: date('2026-11-09')}]->(:`Odd``Label` {name:'other'})"
        ).consume()
        expected = dict(session.run("MATCH (n:SnapshotTest) RETURN properties(n) AS p").single()["p"])
    snapshot_env.save("complete", False)
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE semantic_memory SET summary='changed'"))
    with graph.session() as session:
        session.run("MATCH (n) SET n.name='changed'").consume()
    snapshot_env.load("complete")
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT summary FROM semantic_memory"))).scalar_one() == "original"
    with graph.session() as session:
        restored = dict(session.run("MATCH (n:SnapshotTest) RETURN properties(n) AS p").single()["p"])
    assert restored == expected
    assert isinstance(restored["changed"], DateTime)
    assert isinstance(restored["elapsed"], Duration)
    with graph.session() as session:
        edge = session.run(
            "MATCH (:SnapshotTest)-[r]->(b) RETURN type(r) AS type, r.day AS day, labels(b) AS labels"
        ).single()
    assert edge["type"] == "SEEN`AT" and edge["labels"] == ["Odd`Label"]
    assert str(edge["day"]) == "2026-11-09"


@pytest.mark.asyncio
async def test_real_empty_graph_snapshot_removes_previous_nodes(graph, snapshot_env):
    snapshot_env.save("empty", False)
    with graph.session() as session:
        session.run("CREATE (:SnapshotTest {name:'stale'})").consume()
    snapshot_env.load("empty")
    with graph.session() as session:
        assert session.run("MATCH (n) RETURN count(n) AS n").single()["n"] == 0


@pytest.mark.asyncio
async def test_pg_restore_constraint_failure_rolls_back_truncation(pg, graph, snapshot_env, capsys):
    engine, _, _ = pg
    async with engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO semantic_memory(id, user_id, summary) VALUES ('one', 'u', 'snapshot-value')"
        ))
    snapshot_env.save("before", False)
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE semantic_memory SET summary='live-value'"))
        await conn.execute(text(
            "ALTER TABLE semantic_memory ADD CONSTRAINT reject_snapshot "
            "CHECK (summary <> 'snapshot-value') NOT VALID"
        ))
    with graph.session() as session:
        session.run("CREATE (:SnapshotTest {name:'untouched'})").consume()
    with pytest.raises(RuntimeError, match="psql failed"):
        snapshot_env.load("before")
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT summary FROM semantic_memory"))).scalar_one() == "live-value"
    with graph.session() as session:
        assert session.run("MATCH (n) RETURN n.name AS name").single()["name"] == "untouched"
    assert "[snapshot] loaded" not in capsys.readouterr().out


def test_neo4j_failure_rolls_back_graph_clear(graph):
    from evals import memory_snapshot as snapshot
    from neo4j.exceptions import ConstraintError

    with graph.session() as session:
        session.run(
            "CREATE CONSTRAINT snapshot_unique IF NOT EXISTS "
            "FOR (n:SnapshotUnique) REQUIRE n.name IS UNIQUE"
        ).consume()
        session.run("CREATE (:SnapshotUnique {name:'original'})").consume()
    original = dict(snapshot.NEO4J)
    snapshot.NEO4J.update(
        uri=os.environ["MIRIX_SECURITY_TEST_NEO4J_URI"],
        user="neo4j", password=os.environ["MIRIX_SECURITY_TEST_NEO4J_PASSWORD"],
    )
    try:
        with pytest.raises(ConstraintError):
            snapshot._neo4j_import({
                "nodes": [
                    {"id": 1, "labels": ["SnapshotUnique"], "props": {"name": "duplicate"}},
                    {"id": 2, "labels": ["SnapshotUnique"], "props": {"name": "duplicate"}},
                ],
                "relationships": [],
            })
    finally:
        snapshot.NEO4J.update(original)
        with graph.session() as session:
            session.run("DROP CONSTRAINT snapshot_unique IF EXISTS").consume()
    with graph.session() as session:
        assert session.run("MATCH (n:SnapshotUnique) RETURN n.name AS name").single()["name"] == "original"


@pytest.mark.asyncio
async def test_graph_facts_cannot_cite_deleted_memories(pg, graph):
    from neo4j import AsyncGraphDatabase
    from mirix.services.graph_retriever_v7 import V7AnchorHit, V7Retriever

    engine, _, _ = pg
    async with engine.begin() as connection:
        await connection.execute(text(
            "INSERT INTO semantic_memory(id, user_id, is_deleted) "
            "VALUES ('deleted', 'u', true), ('live', 'u', false)"
        ))
    with graph.session() as session:
        session.run("""
            CREATE (a:V7Anchor {id:'alice', name:'Alice', user_id:'u', anchor_type:'Person'})
            CREATE (b:V7Anchor {id:'guitar', name:'Guitar', user_id:'u', anchor_type:'Object'})
            CREATE (dead:V7Fact {id:'dead-fact', user_id:'u', predicate:'play',
                                memory_ids:['deleted'], lit_keys:[], lit_vals:[]})
            CREATE (live:V7Fact {id:'live-fact', user_id:'u', predicate:'play',
                                memory_ids:['deleted','live'], lit_keys:[], lit_vals:[]})
            CREATE (dead)-[:V7_FACT_ARG {role:'subject'}]->(a)
            CREATE (dead)-[:V7_FACT_ARG {role:'object'}]->(b)
            CREATE (live)-[:V7_FACT_ARG {role:'subject'}]->(a)
            CREATE (live)-[:V7_FACT_ARG {role:'object'}]->(b)
        """).consume()
    driver = AsyncGraphDatabase.driver(
        os.environ["MIRIX_SECURITY_TEST_NEO4J_URI"],
        auth=("neo4j", os.environ["MIRIX_SECURITY_TEST_NEO4J_PASSWORD"]),
    )
    try:
        rows, citations = await V7Retriever()._collect_relation_facts(
            driver, user_id="u", query="What instrument does Alice play?",
            exact_anchors=[V7AnchorHit("alice", "Alice", "Person", 1.0)],
        )
    finally:
        await driver.close()
    assert [row.id for row in rows] == ["live-fact"]
    assert citations == ["live"]
