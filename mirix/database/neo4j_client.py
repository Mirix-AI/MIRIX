"""
Neo4j async client for MIRIX graph memory (v4: two independent graphs).

Used only when ``settings.enable_graph_memory`` is True. Provides:
- Singleton AsyncDriver wrapping the bolt connection
- Schema bootstrap (constraints + vector indexes for both graphs)
- Health check

Two independent graphs, dispatched by which MIRIX memory layer wrote the data:

**G_episodic** (written by EpisodicMemoryManager.insert_event):
- (:Episode {id, user_id, organization_id, summary, occurred_at})
- (:EpisodicEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
- (:Episode)-[:NEXT]->(:Episode)                       # auto, per user, by occurred_at
- (:Episode)-[:CAUSED_BY]->(:Episode)                  # optional, LLM-judged
- (:Episode)-[:MENTIONS {role}]->(:EpisodicEntity)
- (:EpisodicEntity)-[:EP_RELATES {id, keywords, description, weight,
                                   source_episode_ids, valid_at, invalid_at,
                                   expired_at, keywords_embedding}]
                                   ->(:EpisodicEntity)

**G_semantic** (written by SemanticMemoryManager.insert_semantic_item):
- (:Concept {id, user_id, organization_id, name, summary, created_at})
- (:SemanticEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
- (:Concept)-[:CONCEPT_RELATES {keywords, description, weight}]->(:Concept)
- (:Concept)-[:MENTIONS]->(:SemanticEntity)
- (:SemanticEntity)-[:SEM_RELATES {id, keywords, description, weight,
                                    source_concept_ids, keywords_embedding}]
                                    ->(:SemanticEntity)

Two independent graphs with disjoint labels AND disjoint edge types — vector
indexes can be queried per-graph without endpoint-label post-filtering.
EpisodicEntity and SemanticEntity are independent even when they share a name
("Apple" as a fruit Caroline ate vs "Apple" the company concept are distinct).
"""

from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from mirix.log import get_logger
from mirix.settings import settings

logger = get_logger(__name__)

_neo4j_driver: Optional[AsyncDriver] = None


# Constraint + non-vector index DDL. Idempotent.
_SCHEMA_STATEMENTS = [
    # G_episodic constraints
    "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS "
    "FOR (e:Episode) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT episodic_entity_id_unique IF NOT EXISTS "
    "FOR (e:EpisodicEntity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT episodic_entity_user_name_unique IF NOT EXISTS "
    "FOR (e:EpisodicEntity) REQUIRE (e.user_id, e.name_lower) IS UNIQUE",
    "CREATE INDEX episode_user_time IF NOT EXISTS "
    "FOR (e:Episode) ON (e.user_id, e.occurred_at)",

    # G_semantic constraints
    "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT semantic_entity_id_unique IF NOT EXISTS "
    "FOR (e:SemanticEntity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT semantic_entity_user_name_unique IF NOT EXISTS "
    "FOR (e:SemanticEntity) REQUIRE (e.user_id, e.name_lower) IS UNIQUE",
    "CREATE INDEX concept_user_created IF NOT EXISTS "
    "FOR (c:Concept) ON (c.user_id, c.created_at)",

    # G_v6 (lean entity index) — orthogonal to v5 labels so both can coexist.
    # Only built when graph_version=v6; harmless when graph_version=v5.
    "CREATE CONSTRAINT v6_entity_id_unique IF NOT EXISTS "
    "FOR (e:V6Entity) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT v6_entity_user_name_unique IF NOT EXISTS "
    "FOR (e:V6Entity) REQUIRE (e.user_id, e.name_lower) IS UNIQUE",

    # G_v7 (minimal semantic+episodic linkage graph). Separate labels so v7
    # can be compared against v5/v6 without deleting earlier experiments.
    "CREATE CONSTRAINT v7_anchor_id_unique IF NOT EXISTS "
    "FOR (a:V7Anchor) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT v7_anchor_user_name_unique IF NOT EXISTS "
    "FOR (a:V7Anchor) REQUIRE (a.user_id, a.name_lower) IS UNIQUE",
    "CREATE CONSTRAINT v7_memory_ref_id_unique IF NOT EXISTS "
    "FOR (m:V7MemoryRef) REQUIRE m.id IS UNIQUE",
    "CREATE INDEX v7_memory_ref_user_source IF NOT EXISTS "
    "FOR (m:V7MemoryRef) ON (m.user_id, m.source_key)",
]


# Migration: drop v3 schema if it exists (old single-graph design used
# :Entity, :Event, and shared :RELATES). Also drops any old v4-pre-confirmation
# shared :RELATES vector index. Safe to run on a fresh DB — all OPTIONAL.
_V3_CLEANUP_STATEMENTS = [
    "MATCH (n:Entity) DETACH DELETE n",
    "MATCH (n:Event) DETACH DELETE n",
    "DROP CONSTRAINT entity_id_unique IF EXISTS",
    "DROP CONSTRAINT event_id_unique IF EXISTS",
    "DROP CONSTRAINT entity_user_name_unique IF EXISTS",
    "DROP INDEX event_user_time IF EXISTS",
    "DROP INDEX rel_expired IF EXISTS",
    "DROP INDEX entity_name_emb IF EXISTS",
    # Old v4-draft used a shared :RELATES vector index — drop in favor of
    # ep_rel_kw_emb / sem_rel_kw_emb / concept_rel_kw_emb.
    "DROP INDEX rel_kw_emb IF EXISTS",
]


def _vector_index_statement(name: str, label_or_rel: str, prop: str, dim: int, is_rel: bool) -> str:
    """Build a CREATE VECTOR INDEX statement for nodes or relationships."""
    target = f"()-[r:{label_or_rel}]-()" if is_rel else f"(n:{label_or_rel})"
    var = "r" if is_rel else "n"
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        f"FOR {target} ON {var}.{prop} "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {dim}, "
        f"`vector.similarity_function`: 'cosine'"
        f"}}}}"
    )


# Vector indexes — one per (graph, target) pair. Disjoint relationship types
# (:EP_RELATES vs :SEM_RELATES) let us keep separate vector indexes so
# queryRelationships returns episodic-only or semantic-only hits without any
# post-filtering by endpoint label. Concept-Concept edges also get their own
# vector index in case we want to do hl-style retrieval on concept relations
# in the future (P3 may or may not use it).
def _vector_indexes(dim: int) -> list[str]:
    return [
        # G_episodic — entity nodes + entity-entity relations
        _vector_index_statement("ep_entity_name_emb", "EpisodicEntity", "name_embedding", dim, is_rel=False),
        _vector_index_statement("ep_rel_kw_emb", "EP_RELATES", "keywords_embedding", dim, is_rel=True),

        # G_semantic — entity nodes + entity-entity relations + concept-concept relations
        _vector_index_statement("sem_entity_name_emb", "SemanticEntity", "name_embedding", dim, is_rel=False),
        _vector_index_statement("sem_rel_kw_emb", "SEM_RELATES", "keywords_embedding", dim, is_rel=True),
        _vector_index_statement("concept_rel_kw_emb", "CONCEPT_RELATES", "keywords_embedding", dim, is_rel=True),

        # G_v6 — lean entity index. Single vector index on V6Entity.name_embedding.
        # No edge index — V6_COOCCUR carries only a count, no embedding.
        _vector_index_statement("v6_entity_name_emb", "V6Entity", "name_embedding", dim, is_rel=False),

        # G_v7 — minimal linkage graph. Only anchors are vector searched;
        # memory refs point back to PG flat memory rows for details.
        _vector_index_statement("v7_anchor_name_emb", "V7Anchor", "name_embedding", dim, is_rel=False),
    ]


async def init_neo4j_client() -> Optional[AsyncDriver]:
    """Initialize the Neo4j async driver and bootstrap v4 schema.

    Returns the driver, or ``None`` if graph memory is disabled or connection
    fails. Failures are logged but do not raise — the rest of MIRIX must
    continue working without graph memory.
    """
    global _neo4j_driver
    if not settings.enable_graph_memory:
        logger.debug("Graph memory disabled; skipping Neo4j init")
        return None

    if _neo4j_driver is not None:
        return _neo4j_driver

    try:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await driver.verify_connectivity()
        logger.info("Neo4j async driver connected at %s", settings.neo4j_uri)

        await _bootstrap_schema(driver, settings.neo4j_vector_dim, settings.neo4j_database)

        _neo4j_driver = driver
        return driver
    except Exception as e:
        logger.error("Neo4j init failed: %s — graph memory will be unavailable", e)
        _neo4j_driver = None
        return None


async def _bootstrap_schema(driver: AsyncDriver, vector_dim: int, database: str) -> None:
    """Run DDL in order: v3 cleanup → v4 constraints/indexes → vector indexes."""
    async with driver.session(database=database) as session:
        # Step 1: clean up v3 schema (no-op on fresh DB)
        for stmt in _V3_CLEANUP_STATEMENTS:
            try:
                await session.run(stmt)
            except Exception as e:
                logger.debug("v3 cleanup stmt skipped (%s): %s", stmt[:50], e)

        # Step 2: v4 constraints + plain indexes
        for stmt in _SCHEMA_STATEMENTS:
            try:
                await session.run(stmt)
            except Exception as e:
                logger.warning("v4 schema stmt failed (%s): %s", stmt[:60], e)

        # Step 3: vector indexes
        for stmt in _vector_indexes(vector_dim):
            try:
                await session.run(stmt)
            except Exception as e:
                logger.warning("vector index stmt failed (%s): %s", stmt[:60], e)

    logger.info("Neo4j v4 schema bootstrap complete (vector_dim=%d)", vector_dim)


def get_neo4j_driver() -> Optional[AsyncDriver]:
    """Get the global Neo4j driver. Returns None if not initialized."""
    return _neo4j_driver


async def close_neo4j_driver() -> None:
    """Close the global driver. Safe to call when not initialized."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        try:
            await _neo4j_driver.close()
        except Exception as e:
            logger.warning("Error closing Neo4j driver: %s", e)
        _neo4j_driver = None


async def neo4j_healthcheck() -> bool:
    """Return True iff a trivial Cypher round-trip succeeds."""
    driver = _neo4j_driver
    if driver is None:
        return False
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return record is not None and record["ok"] == 1
    except Exception as e:
        logger.warning("Neo4j healthcheck failed: %s", e)
        return False
