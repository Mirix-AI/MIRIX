# v4 Graph Memory — Code Changes

Per-file changelog of the dual-graph LightRAG patch. Companion to [README.md](README.md) (overall design) and [v4_graph_memory.patch](v4_graph_memory.patch) (machine-applicable form).

**Conventions**
- New files: full source in a ```python``` fence
- Modified files: minimal hunk in a ```diff``` fence
- Deleted files: brief note (full deletion captured in the .patch)

## Table of contents

**New files (14)**
- [mirix/database/neo4j_client.py](#mirixdatabaseneo4jclientpy)
- [mirix/database/startup_migrations.py](#mirixdatabasestartupmigrationspy)
- [mirix/database/token_tracker.py](#mirixdatabasetokentrackerpy)
- [mirix/prompts/lightrag_prompts.py](#mirixpromptslightragpromptspy)
- [mirix/services/_graph_common.py](#mirixservicesgraphcommonpy)
- [mirix/services/_graph_retriever_base.py](#mirixservicesgraphretrieverbasepy)
- [mirix/services/episodic_graph_manager.py](#mirixservicesepisodicgraphmanagerpy)
- [mirix/services/episodic_graph_retriever.py](#mirixservicesepisodicgraphretrieverpy)
- [mirix/services/graph_retriever_dispatcher.py](#mirixservicesgraphretrieverdispatcherpy)
- [mirix/services/lightrag_extractor.py](#mirixserviceslightragextractorpy)
- [mirix/services/lightrag_keyword_extractor.py](#mirixserviceslightragkeywordextractorpy)
- [mirix/services/lightrag_merger.py](#mirixserviceslightragmergerpy)
- [mirix/services/semantic_graph_manager.py](#mirixservicessemanticgraphmanagerpy)
- [mirix/services/semantic_graph_retriever.py](#mirixservicessemanticgraphretrieverpy)

**Modified files (12)**
- [docker-compose.yml](#docker-composeyml)
- [evals/main_eval.py](#evalsmainevalpy)
- [evals/mirix_memory_system.py](#evalsmirixmemorysystempy)
- [evals/task_agent.py](#evalstaskagentpy)
- [mirix/llm_api/openai.py](#mirixllmapiopenaipy)
- [mirix/server/rest_api.py](#mirixserverrestapipy)
- [mirix/server/server.py](#mirixserverserverpy)
- [mirix/services/episodic_memory_manager.py](#mirixservicesepisodicmemorymanagerpy)
- [mirix/services/semantic_memory_manager.py](#mirixservicessemanticmemorymanagerpy)
- [mirix/orm/__init__.py](#mirixorminitpy)
- [mirix/settings.py](#mirixsettingspy)
- [requirements.txt](#requirementstxt)

**Deleted files (2)**
- `mirix/orm/graph_memory.py`
- `mirix/services/graph_memory_manager.py`

---

## New files

### `mirix/database/neo4j_client.py`

_Async Neo4j driver + schema bootstrap (6 constraints, 5 vector indexes)_

```python
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
```

### `mirix/database/startup_migrations.py`

_Idempotent PG migration framework run before create_all (drops v2 graph tables)_

```python
"""
Lightweight startup migrations.

MIRIX has no Alembic — schema is created via ``Base.metadata.create_all`` in
``ensure_tables_created``. This module runs idempotent DROP/ALTER statements
that need to happen *before* ``create_all`` so the new ORM state takes hold.

Add a migration by appending to ``MIGRATIONS`` below. Each migration is a
``(name, sql_statements)`` tuple. Each statement runs in its own transaction;
failures are logged but do not raise (so a partial drop on dev DBs doesn't
brick startup).
"""

from typing import List, Tuple

from sqlalchemy.ext.asyncio import AsyncEngine

from mirix.log import get_logger

logger = get_logger(__name__)


# (migration_name, [statements])
MIGRATIONS: List[Tuple[str, List[str]]] = [
    (
        # v2 graph memory tables — replaced by Neo4j-backed implementation
        # (see lightrag_graph_manager). Drop in dependency order: edges that
        # FK into entity_nodes / episode_nodes must go first.
        "drop_v2_graph_memory_tables",
        [
            "DROP TABLE IF EXISTS involves_edges CASCADE",
            "DROP TABLE IF EXISTS entity_edges CASCADE",
            "DROP TABLE IF EXISTS episode_nodes CASCADE",
            "DROP TABLE IF EXISTS entity_nodes CASCADE",
        ],
    ),
]


async def run_startup_migrations(engine: AsyncEngine) -> None:
    """Run all pending migrations against ``engine``. Safe to call repeatedly."""
    for name, statements in MIGRATIONS:
        logger.info("Startup migration: %s", name)
        for stmt in statements:
            try:
                async with engine.begin() as conn:
                    from sqlalchemy import text as sa_text
                    await conn.execute(sa_text(stmt))
            except Exception as e:
                # Most likely on fresh DBs: table never existed. That's fine.
                logger.debug("Migration stmt skipped (%s): %s", stmt, e)
```

### `mirix/database/token_tracker.py`

_Opt-in process-global LLM token usage counter (default off)_

```python
"""
Global token-usage tracker for instrumenting MIRIX's LLM calls.

Designed so call-sites can blindly call ``record(...)`` and external code (evals,
benchmarks) decides what counts as "build" vs "query" via context-managed phases.

Why a tracker module instead of LangFuse:
  - LangFuse is heavy (network round-trips, project setup, env vars).
  - For evals we just want a per-run integer total. A process-global dict
    that records by ``(phase, user_id)`` is enough.

Usage in eval:

    from mirix.database.token_tracker import set_phase, snapshot, reset

    reset()  # at process start, optional
    with set_phase("build"):
        await client.add(...)   # all server LLM calls recorded under "build"
    with set_phase("query"):
        await task_agent.answer(...)
    stats = snapshot()  # {(phase, user_id): {prompt, completion, total, calls}}

Usage in call-sites (one-liner):

    from mirix.database.token_tracker import record
    record(prompt_tokens=..., completion_tokens=...)

Thread-safe via a single lock; concurrency-safe via contextvars for ``_phase_var``.
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import Optional

# Process-wide enable flag. Default OFF so the tracker is a true no-op for
# anyone not running an eval. Flip via enable()/disable() — typically called
# from an eval harness (see evals/main_eval.py) or from the
# /debug/token_stats/* REST endpoints.
_enabled: bool = False

# Current logical phase, propagated through asyncio tasks via contextvar.
# When tracker is enabled and no explicit phase is set, falls back to "server".
# Evals call set_phase("build") or set_phase("query") to bucket more finely.
_phase_var: ContextVar[Optional[str]] = ContextVar("mirix_token_phase", default=None)

# Stable buckets keyed by (phase, user_id). user_id is optional — calls from
# server endpoints that don't know the user just bucket as user_id="*".
_lock = threading.Lock()
_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
    lambda: {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
)


def enable() -> None:
    """Turn the tracker on. ``record()`` becomes a real write after this."""
    global _enabled
    _enabled = True


def disable() -> None:
    """Turn the tracker off. ``record()`` becomes a no-op."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def record(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: Optional[int] = None,
    user_id: str = "*",
) -> None:
    """Add one OpenAI/Anthropic ``usage`` payload to the active phase bucket.

    No-op unless ``enable()`` has been called. Phase defaults to "server"
    when enabled but no ``set_phase`` context is active.
    Robust to ``None`` / negative inputs.
    """
    if not _enabled:
        return
    phase = _phase_var.get() or "server"
    p = max(int(prompt_tokens or 0), 0)
    c = max(int(completion_tokens or 0), 0)
    t = int(total_tokens) if total_tokens is not None else p + c
    with _lock:
        bucket = _stats[(phase, user_id)]
        bucket["prompt"] += p
        bucket["completion"] += c
        bucket["total"] += t
        bucket["calls"] += 1


@contextlib.contextmanager
def set_phase(phase: str):
    """Context manager that sets ``_phase_var`` for the duration of the block.

    Nested calls are supported — inner phase wins, restored on exit. Cross-task
    propagation works because ``_phase_var`` is a contextvar (each asyncio Task
    inherits the calling task's context).
    """
    token = _phase_var.set(phase)
    try:
        yield
    finally:
        _phase_var.reset(token)


def snapshot() -> dict[str, dict[str, int]]:
    """Return a copy of current stats keyed by ``"phase|user_id"`` strings."""
    with _lock:
        return {f"{phase}|{uid}": dict(v) for (phase, uid), v in _stats.items()}


def reset() -> None:
    """Wipe all buckets. Use at the start of a fresh eval run."""
    with _lock:
        _stats.clear()
```

### `mirix/prompts/lightrag_prompts.py`

_LightRAG entity-extraction + keyword-extraction + summarize prompts_

```python
"""
LightRAG prompt templates, adapted for MIRIX graph memory.

Source: https://github.com/HKUDS/LightRAG (MIT License) — see prompt.py.
The structure (delimiters, system/user split, examples format) is preserved
verbatim so output parsers can be shared. Entity types are tuned for
MIRIX's conversational corpus (added "Date", "Quantity"; dropped
"NaturalObject" / "Artifact" which rarely appear in chat).
"""

# Delimiters used inside extracted tuples. Must match the parser in
# lightrag_extractor._parse_extraction_output.
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# Default entity types — tuned for personal-assistant conversation memory.
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Date",
    "Quantity",
    "Other",
]


ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
        *   `relationship_strength`: A floating point value between 0.0 and 1.0 estimating how strong/important this relationship is.
    *   **Output Format - Relationships:** Output a total of 6 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description{tuple_delimiter}relationship_strength`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}Location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""


ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""


# A single conversational example to keep the system prompt small. Adding more
# examples helps consistency but inflates the prompt cost on every chunk.
ENTITY_EXTRACTION_EXAMPLES = [
    """<Entity_types>
["Person","Organization","Location","Event","Concept","Method","Content","Date","Quantity","Other"]

<Input Text>
```
Caroline mentioned that her cousin Melanie just moved to Berlin to start a job at SAP last month. They used to live together in Munich while Caroline was finishing her PhD on quantum optics.
```

<Output>
entity{tuple_delimiter}Caroline{tuple_delimiter}Person{tuple_delimiter}Caroline is the speaker; she previously lived in Munich while pursuing a PhD on quantum optics.
entity{tuple_delimiter}Melanie{tuple_delimiter}Person{tuple_delimiter}Melanie is Caroline's cousin who recently moved to Berlin to start a job at SAP.
entity{tuple_delimiter}Berlin{tuple_delimiter}Location{tuple_delimiter}Berlin is the city Melanie moved to for her new job at SAP.
entity{tuple_delimiter}Munich{tuple_delimiter}Location{tuple_delimiter}Munich is the city where Caroline and Melanie used to live together while Caroline was a PhD student.
entity{tuple_delimiter}SAP{tuple_delimiter}Organization{tuple_delimiter}SAP is the organization where Melanie recently started working.
entity{tuple_delimiter}Quantum Optics{tuple_delimiter}Concept{tuple_delimiter}Quantum optics is the subject of Caroline's PhD research.
relation{tuple_delimiter}Caroline{tuple_delimiter}Melanie{tuple_delimiter}family relation, cohabitation{tuple_delimiter}Caroline and Melanie are cousins who previously lived together in Munich.{tuple_delimiter}0.9
relation{tuple_delimiter}Melanie{tuple_delimiter}Berlin{tuple_delimiter}relocation, residence{tuple_delimiter}Melanie recently moved to Berlin.{tuple_delimiter}0.8
relation{tuple_delimiter}Melanie{tuple_delimiter}SAP{tuple_delimiter}employment, new job{tuple_delimiter}Melanie started a job at SAP.{tuple_delimiter}0.85
relation{tuple_delimiter}Caroline{tuple_delimiter}Munich{tuple_delimiter}past residence, education{tuple_delimiter}Caroline lived in Munich while completing her PhD.{tuple_delimiter}0.7
relation{tuple_delimiter}Caroline{tuple_delimiter}Quantum Optics{tuple_delimiter}academic research, PhD topic{tuple_delimiter}Caroline pursued a PhD on quantum optics.{tuple_delimiter}0.8
{completion_delimiter}
""",
]


# Used by the description merge step in lightrag_merger when an entity or
# relation accumulates more than FORCE_LLM_SUMMARY_ON_MERGE descriptions.
SUMMARIZE_DESCRIPTIONS_PROMPT = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given {description_type} into a single, comprehensive, and cohesive summary.

---Instructions---
1. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
2. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the {description_type} at the beginning of the summary to ensure immediate clarity and context.
3. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, attempt to reconcile them or present both viewpoints with noted uncertainty.
4. Length Constraint: The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
5. Output: Plain text, no markdown fences, no preamble.

---Input---
{description_type} Name: {description_name}

Description List:
```
{description_list}
```

---Output---
"""


KEYWORDS_EXTRACTION_PROMPT = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
Example 1:
Query: "How does international trade influence global economic stability?"
Output:
{{"high_level_keywords": ["International trade", "Global economic stability", "Economic impact"], "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]}}

Example 2:
Query: "Where did Caroline live during her PhD?"
Output:
{{"high_level_keywords": ["Past residence", "Academic life"], "low_level_keywords": ["Caroline", "PhD", "Munich"]}}

Example 3:
Query: "What is the role of education in reducing poverty?"
Output:
{{"high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"], "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]}}

---Real Data---
User Query: {query}

---Output---
"""


def render_keywords_extraction_prompt(query: str, language: str = "English") -> str:
    return KEYWORDS_EXTRACTION_PROMPT.format(query=query, language=language)


def render_extraction_system_prompt(
    entity_types: list[str] | None = None,
    language: str = "English",
) -> str:
    """Render the system prompt with entity types and example bodies inlined."""
    types = entity_types or DEFAULT_ENTITY_TYPES
    types_str = ", ".join(types)
    example_ctx = {
        "tuple_delimiter": TUPLE_DELIMITER,
        "completion_delimiter": COMPLETION_DELIMITER,
    }
    examples = "\n".join(ex.format(**example_ctx) for ex in ENTITY_EXTRACTION_EXAMPLES)
    return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types=types_str,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        language=language,
        examples=examples,
    )


def render_extraction_user_prompt(
    input_text: str,
    entity_types: list[str] | None = None,
    language: str = "English",
) -> str:
    types = entity_types or DEFAULT_ENTITY_TYPES
    return ENTITY_EXTRACTION_USER_PROMPT.format(
        entity_types=", ".join(types),
        completion_delimiter=COMPLETION_DELIMITER,
        language=language,
        input_text=input_text,
    )


def render_summarize_descriptions_prompt(
    description_type: str,
    description_name: str,
    description_list: list[str],
    summary_length: int = 500,
) -> str:
    return SUMMARIZE_DESCRIPTIONS_PROMPT.format(
        description_type=description_type,
        description_name=description_name,
        description_list="\n".join(f"- {d}" for d in description_list),
        summary_length=summary_length,
    )
```

### `mirix/services/_graph_common.py`

_Shared helpers: gen_id, normalize_name, iso, embed_batch_

```python
"""
Shared helpers for v4 graph managers (episodic + semantic).

Both managers do roughly the same things but write into disjoint Neo4j labels:
  - episodic: (:Episode), (:EpisodicEntity), [:EP_RELATES], [:MENTIONS], [:NEXT]
  - semantic: (:Concept), (:SemanticEntity), [:SEM_RELATES], [:MENTIONS],
              [:CONCEPT_RELATES]

This module hosts the parts that don't care which label set is in play:
helpers for id generation, name normalization, embedding batching, and the
LLM model resolution from an AgentState.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from mirix.embeddings import embedding_model
from mirix.log import get_logger
from mirix.schemas.agent import AgentState

logger = get_logger(__name__)


def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def iso(ts: datetime) -> str:
    """Neo4j datetime properties want ISO-8601 strings (with tz)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def llm_model_from_agent(agent_state: AgentState, default: str = "gpt-4.1-mini") -> str:
    """Pull LLM model name from agent_state, falling back to default."""
    try:
        cfg = getattr(agent_state, "llm_config", None)
        if cfg is not None and getattr(cfg, "model", None):
            return cfg.model
    except Exception:
        pass
    return default


async def embed_batch(
    texts: list[str], agent_state: AgentState, *, max_concurrency: int = 8
) -> list[Optional[list[float]]]:
    """
    Compute embeddings for many short strings via the agent's configured model.

    MIRIX's embedding adapter is single-text only, so we fan out with bounded
    concurrency. Returns ``None`` for failed entries so callers can decide
    whether to drop the row or store without a vector.
    """
    if not texts:
        return []
    try:
        model = await embedding_model(agent_state.embedding_config)
    except Exception as e:
        logger.warning("Embedding model init failed: %s", e)
        return [None] * len(texts)

    sem = asyncio.Semaphore(max_concurrency)

    async def one(t: str) -> Optional[list[float]]:
        async with sem:
            try:
                return await model.get_text_embedding(t)
            except Exception as e:
                logger.debug("Embed failed for '%s...': %s", (t or "")[:40], e)
                return None

    return await asyncio.gather(*(one(t) for t in texts))
```

### `mirix/services/_graph_retriever_base.py`

_Symmetric base for episodic+semantic retrievers (ll/hl Cypher + budget)_

```python
"""
Shared retriever base for v4 graph readers.

EpisodicRetriever and SemanticRetriever are structurally identical — they
walk one graph (entities → relations → items), dedup, and score. The only
differences are which labels/types they MATCH, which vector indexes they
query, and what extra item-item edges they expand (NEXT for episodes,
CONCEPT_RELATES for concepts). This base factors out the shared algorithm
and lets subclasses fill in label/type strings.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import tiktoken

from mirix.log import get_logger

logger = get_logger(__name__)


# Token budgets and ranking constants. Tuned for chat-memory scale.
DEFAULT_TOP_K = 30
DEFAULT_CHUNK_TOP_K = 10
RECENCY_HALF_LIFE_DAYS = 30.0
COSINE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3


_tokenizer = None


def count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    if not text:
        return 0
    return len(_tokenizer.encode(text))


def recency_decay(ts: Optional[Any]) -> float:
    """Exponential decay over days; missing ts → 0.5."""
    if ts is None:
        return 0.5
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
        if age_days < 0:
            return 1.0
        return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5


def final_score(cosine: float, ts: Optional[Any]) -> float:
    return COSINE_WEIGHT * float(cosine or 0.0) + RECENCY_WEIGHT * recency_decay(ts)


def fmt_date(ts: Optional[Any]) -> str:
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return ts.strftime("%d %B %Y")
    except Exception:
        return str(ts)


# ────────────────────────────────────────────────────────────── dataclasses

@dataclass
class EntityHit:
    id: str
    name: str
    entity_type: str
    description: str
    rank: int
    cosine: float
    updated_at: Optional[Any]
    score: float = 0.0


@dataclass
class RelationHit:
    id: str
    src_name: str
    tgt_name: str
    keywords: str
    description: str
    weight: float
    cosine: float
    valid_at: Optional[Any]
    score: float = 0.0


@dataclass
class ItemHit:
    """Episode or Concept — same shape so format/budget code is shared."""
    id: str
    label: str           # 'Episode' or 'Concept'
    summary: str
    detail: str          # episode.details or concept.summary (fallback)
    timestamp: Optional[Any]   # occurred_at for Episode, created_at for Concept
    cosine: float
    score: float = 0.0
    source: str = "mentions"  # 'mentions', 'one_hop', etc — for debugging


@dataclass
class GraphSearchResult:
    entities: list[EntityHit] = field(default_factory=list)
    relations: list[RelationHit] = field(default_factory=list)
    items: list[ItemHit] = field(default_factory=list)


# ────────────────────────────────────────── round-robin merge helpers

def round_robin_merge_entities(
    locals_: list[EntityHit], globals_: list[EntityHit]
) -> list[EntityHit]:
    merged: list[EntityHit] = []
    seen: set[str] = set()
    n = max(len(locals_), len(globals_))
    for i in range(n):
        for source in (locals_, globals_):
            if i < len(source):
                hit = source[i]
                key = (hit.name or "").lower() or hit.id
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
    return merged


def round_robin_merge_relations(
    locals_: list[RelationHit], globals_: list[RelationHit]
) -> list[RelationHit]:
    merged: list[RelationHit] = []
    seen: set[tuple[str, str]] = set()
    n = max(len(locals_), len(globals_))
    for i in range(n):
        for source in (locals_, globals_):
            if i < len(source):
                hit = source[i]
                key = tuple(sorted([(hit.src_name or "").lower(), (hit.tgt_name or "").lower()]))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
    return merged


# ────────────────────────────────────────── token budget application

def apply_budget_to_search(
    search: GraphSearchResult,
    *,
    max_entity_tokens: int,
    max_relation_tokens: int,
    max_item_tokens: int,
) -> GraphSearchResult:
    def _trim(items: list, render, budget: int) -> list:
        kept = []
        used = 0
        for it in items:
            t = count_tokens(render(it))
            if used + t > budget:
                break
            kept.append(it)
            used += t
        return kept

    kept_e = _trim(
        search.entities,
        lambda e: f"- {e.name} ({e.entity_type}, rank={e.rank}): {e.description}",
        max_entity_tokens,
    )
    kept_r = _trim(
        search.relations,
        lambda r: f"- {r.src_name} <-> {r.tgt_name} [{r.keywords}]: {r.description}",
        max_relation_tokens,
    )
    kept_i = _trim(
        search.items,
        lambda it: f"- [{fmt_date(it.timestamp)}] {it.summary} {it.detail[:200]}",
        max_item_tokens,
    )
    return GraphSearchResult(entities=kept_e, relations=kept_r, items=kept_i)


# ────────────────────────────────────────── retriever base

class GraphRetrieverBase:
    """Subclasses set these as class attributes."""

    # Labels
    ENTITY_LABEL: str = ""        # 'EpisodicEntity' or 'SemanticEntity'
    ITEM_LABEL: str = ""          # 'Episode' or 'Concept'
    REL_TYPE: str = ""            # 'EP_RELATES' or 'SEM_RELATES'

    # Vector index names
    ENTITY_VECTOR_INDEX: str = "" # 'ep_entity_name_emb' or 'sem_entity_name_emb'
    REL_VECTOR_INDEX: str = ""    # 'ep_rel_kw_emb' or 'sem_rel_kw_emb'

    # Title used in markdown output
    SECTION_TITLE: str = ""       # 'Episodic' or 'Semantic'

    # ──────────────────────────────────────────────────────── low-level path

    def _ll_cypher(self) -> str:
        # Vector search on entity names → seed entities. For each seed, pull
        # one-hop neighbor entities via REL_TYPE (no expired_at on semantic
        # rels — but the predicate is harmless: missing property tests as
        # NULL which evaluates the WHERE filter as "IS NULL = true").
        return f"""
        CALL db.index.vector.queryNodes('{self.ENTITY_VECTOR_INDEX}', $top_k, $emb)
        YIELD node AS e, score AS sim
        WHERE e.user_id = $user_id
        WITH e, sim
        ORDER BY sim DESC LIMIT $top_k
        WITH collect({{e: e, sim: sim}}) AS seeds
        UNWIND seeds AS s
        WITH s.e AS seed, s.sim AS sim
        OPTIONAL MATCH (seed)-[r:{self.REL_TYPE}]-(other:{self.ENTITY_LABEL} {{user_id: $user_id}})
        WHERE coalesce(r.expired_at, datetime('9999-01-01')) > datetime()
        RETURN
            seed.id AS sid, seed.name AS sname, seed.entity_type AS stype,
            seed.description AS sdesc, coalesce(seed.rank, 0) AS srank,
            seed.updated_at AS supdated, sim AS sim,
            r.id AS rid, r.description AS rdesc, r.keywords AS rkw,
            coalesce(r.weight, 0.5) AS rweight, r.valid_at AS rvalid,
            other.name AS oname
        """

    def _hl_cypher(self) -> str:
        # Vector search on relation keywords → seed relations. Pull both
        # endpoint entities. We don't have to filter by endpoint label since
        # REL_TYPE alone identifies the graph (EP_RELATES vs SEM_RELATES).
        return f"""
        CALL db.index.vector.queryRelationships('{self.REL_VECTOR_INDEX}', $top_k, $emb)
        YIELD relationship AS r, score AS sim
        WHERE coalesce(r.expired_at, datetime('9999-01-01')) > datetime()
        MATCH (a:{self.ENTITY_LABEL})-[r]-(b:{self.ENTITY_LABEL})
        WHERE a.user_id = $user_id AND b.user_id = $user_id
        WITH r, sim, a, b
        ORDER BY sim DESC LIMIT $top_k
        RETURN
            r.id AS rid, r.description AS rdesc, r.keywords AS rkw,
            coalesce(r.weight, 0.5) AS rweight, r.valid_at AS rvalid, sim AS sim,
            a.id AS aid, a.name AS aname, a.entity_type AS atype,
            a.description AS adesc, coalesce(a.rank, 0) AS arank, a.updated_at AS aupdated,
            b.id AS bid, b.name AS bname, b.entity_type AS btype,
            b.description AS bdesc, coalesce(b.rank, 0) AS brank, b.updated_at AS bupdated
        """

    # ─────────────────────────────────────── high-level orchestration

    async def search(
        self,
        *,
        driver,
        user_id: str,
        ll_embedding: Optional[list[float]],
        hl_embedding: Optional[list[float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[list[EntityHit], list[RelationHit]]:
        """Run ll + hl in parallel where embeddings are available."""
        from mirix.settings import settings

        tasks: dict[str, asyncio.Task] = {}
        if ll_embedding is not None:
            tasks["ll"] = asyncio.create_task(
                self._run_ll(driver, user_id, ll_embedding, top_k, settings.neo4j_database)
            )
        if hl_embedding is not None:
            tasks["hl"] = asyncio.create_task(
                self._run_hl(driver, user_id, hl_embedding, top_k, settings.neo4j_database)
            )
        if not tasks:
            return [], []
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        local_e, local_r, global_e, global_r = [], [], [], []
        for purpose, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                logger.warning("%s retrieval branch %s failed: %s", self.SECTION_TITLE, purpose, res)
                continue
            if purpose == "ll":
                local_e, local_r = res
            elif purpose == "hl":
                global_e, global_r = res

        final_e = round_robin_merge_entities(local_e, global_e)
        final_r = round_robin_merge_relations(local_r, global_r)
        for e in final_e:
            e.score = final_score(e.cosine, e.updated_at)
        for r in final_r:
            r.score = final_score(r.cosine, r.valid_at)
        final_e.sort(key=lambda x: x.score, reverse=True)
        final_r.sort(key=lambda x: x.score, reverse=True)
        return final_e, final_r

    async def _run_ll(self, driver, user_id, emb, top_k, database):
        entities: dict[str, EntityHit] = {}
        relations: dict[str, RelationHit] = {}
        async with driver.session(database=database) as session:
            result = await session.run(self._ll_cypher(), user_id=user_id, emb=emb, top_k=top_k)
            async for rec in result:
                sid = rec["sid"]
                if sid and sid not in entities:
                    entities[sid] = EntityHit(
                        id=sid, name=rec["sname"],
                        entity_type=rec["stype"] or "Other",
                        description=rec["sdesc"] or "",
                        rank=int(rec["srank"] or 0),
                        cosine=float(rec["sim"] or 0.0),
                        updated_at=rec["supdated"],
                    )
                rid = rec["rid"]
                if rid and rid not in relations:
                    relations[rid] = RelationHit(
                        id=rid,
                        src_name=rec["sname"], tgt_name=rec["oname"] or "",
                        keywords=rec["rkw"] or "", description=rec["rdesc"] or "",
                        weight=float(rec["rweight"] or 0.5),
                        cosine=float(rec["sim"] or 0.0),
                        valid_at=rec["rvalid"],
                    )
        return list(entities.values()), list(relations.values())

    async def _run_hl(self, driver, user_id, emb, top_k, database):
        entities: dict[str, EntityHit] = {}
        relations: dict[str, RelationHit] = {}
        async with driver.session(database=database) as session:
            result = await session.run(self._hl_cypher(), user_id=user_id, emb=emb, top_k=top_k)
            async for rec in result:
                rid = rec["rid"]
                if rid and rid not in relations:
                    relations[rid] = RelationHit(
                        id=rid,
                        src_name=rec["aname"] or "", tgt_name=rec["bname"] or "",
                        keywords=rec["rkw"] or "", description=rec["rdesc"] or "",
                        weight=float(rec["rweight"] or 0.5),
                        cosine=float(rec["sim"] or 0.0),
                        valid_at=rec["rvalid"],
                    )
                for prefix in ("a", "b"):
                    eid = rec[f"{prefix}id"]
                    if not eid or eid in entities:
                        continue
                    entities[eid] = EntityHit(
                        id=eid, name=rec[f"{prefix}name"],
                        entity_type=rec[f"{prefix}type"] or "Other",
                        description=rec[f"{prefix}desc"] or "",
                        rank=int(rec[f"{prefix}rank"] or 0),
                        cosine=float(rec["sim"] or 0.0),
                        updated_at=rec[f"{prefix}updated"],
                    )
        return list(entities.values()), list(relations.values())
```

### `mirix/services/episodic_graph_manager.py`

_Writes G_episodic: Episode, EpisodicEntity, NEXT, EP_RELATES, MENTIONS_

```python
"""
Episodic graph manager (v4) — writes G_episodic in Neo4j.

Hooked from EpisodicMemoryManager.insert_event after the PG row has been
committed. Failures are non-fatal: PG remains the source of truth.

Graph elements written here:
  (:Episode {id, user_id, organization_id, summary, occurred_at})
  (:EpisodicEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
  (:Episode)-[:NEXT]->(:Episode)
  (:Episode)-[:MENTIONS {role}]->(:EpisodicEntity)
  (:EpisodicEntity)-[:EP_RELATES {id, keywords, description, weight,
                                  source_episode_ids, valid_at, invalid_at,
                                  expired_at, keywords_embedding}]
                                  ->(:EpisodicEntity)

CAUSED_BY edges are reserved for a future optional LLM step (P2 leaves them
unused — write path stays at ~1 LLM call/insert).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import (
    embed_batch,
    gen_id,
    iso,
    llm_model_from_agent,
    normalize_name,
)
from mirix.services.lightrag_extractor import (
    ExtractedEntity,
    ExtractedRelation,
    extract_entities_and_relations,
)
from mirix.services.lightrag_merger import merge_descriptions
from mirix.settings import settings

logger = get_logger(__name__)


class EpisodicGraphManager:
    """Stateless coordinator. Construct one per call."""

    async def process_episode(
        self,
        *,
        episode_id: str,
        summary: str,
        details: str,
        occurred_at: datetime,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Run the full episodic write path. Never raises."""
        if not settings.enable_graph_memory:
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        text = (summary or "") + ("\n" + details if details else "")

        # W2: extract first so we know what to embed/upsert
        extraction = await extract_entities_and_relations(
            text=text, llm_model=llm_model_from_agent(agent_state)
        )

        # W1: always create the Episode node, even when extraction is empty
        await self._upsert_episode(
            driver,
            episode_id=episode_id,
            summary=summary,
            occurred_at=occurred_at,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W6: connect Episode to previous Episode by occurred_at (auto NEXT)
        await self._link_next(driver, user_id=user_id, episode_id=episode_id, occurred_at=occurred_at)

        if not extraction.entities and not extraction.relations:
            return {"entities": 0, "relations": 0}

        # W3: upsert EpisodicEntity nodes
        merged_entities = await self._upsert_entities(
            driver,
            entities=extraction.entities,
            episode_id=episode_id,
            agent_state=agent_state,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W4: upsert EP_RELATES edges
        merged_relations = await self._upsert_relations(
            driver,
            relations=extraction.relations,
            episode_id=episode_id,
            occurred_at=occurred_at,
            agent_state=agent_state,
            user_id=user_id,
            llm_model=llm_model_from_agent(agent_state),
        )

        # W7: refresh rank (degree)
        touched_names = sorted({e.name for e in extraction.entities})
        await self._refresh_ranks(driver, names=touched_names, user_id=user_id)

        return {
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
            "merged_entities": merged_entities,
            "merged_relations": merged_relations,
        }

    # --------------------------------------------------------- W1: Episode

    async def _upsert_episode(
        self,
        driver,
        *,
        episode_id: str,
        summary: str,
        occurred_at: datetime,
        user_id: str,
        organization_id: str,
    ) -> None:
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MERGE (e:Episode {id: $id})
                ON CREATE SET e.user_id = $user_id,
                              e.organization_id = $org_id,
                              e.summary = $summary,
                              e.occurred_at = $occurred_at
                ON MATCH SET e.summary = $summary,
                             e.occurred_at = $occurred_at
                """,
                id=episode_id,
                user_id=user_id,
                org_id=organization_id,
                summary=summary or "",
                occurred_at=iso(occurred_at),
            )

    # ------------------------------------------------------- W6: NEXT edges

    async def _link_next(
        self, driver, *, user_id: str, episode_id: str, occurred_at: datetime
    ) -> None:
        """
        Connect the new episode to the most recent prior episode (same user)
        with a :NEXT edge. Idempotent: if a NEXT edge from the same prior
        episode already exists, MERGE keeps it.
        """
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MATCH (current:Episode {id: $id})
                OPTIONAL MATCH (prev:Episode {user_id: $user_id})
                WHERE prev.id <> $id AND prev.occurred_at < $occurred_at
                WITH current, prev
                ORDER BY prev.occurred_at DESC
                LIMIT 1
                FOREACH (_ IN CASE WHEN prev IS NULL THEN [] ELSE [1] END |
                    MERGE (prev)-[:NEXT]->(current)
                )
                """,
                id=episode_id,
                user_id=user_id,
                occurred_at=iso(occurred_at),
            )

    # -------------------------------------------------------- W3: Entities

    async def _upsert_entities(
        self,
        driver,
        *,
        entities: list[ExtractedEntity],
        episode_id: str,
        agent_state: AgentState,
        user_id: str,
        organization_id: str,
    ) -> int:
        if not entities:
            return 0

        # Fetch existing entities by (user_id, name_lower) in one round-trip
        name_lowers = [normalize_name(e.name) for e in entities]
        existing = await self._fetch_existing_entities(driver, user_id, name_lowers)

        # Embed names for entities not yet in the graph
        new_entities = [e for e in entities if normalize_name(e.name) not in existing]
        new_embeddings = await embed_batch([e.name for e in new_entities], agent_state)
        new_emb_map: dict[str, Optional[list[float]]] = {
            normalize_name(e.name): emb for e, emb in zip(new_entities, new_embeddings)
        }

        now = iso(datetime.now(timezone.utc))
        merged_count = 0
        llm_model = llm_model_from_agent(agent_state)

        new_rows: list[dict[str, Any]] = []
        for e in new_entities:
            nl = normalize_name(e.name)
            new_rows.append({
                "id": gen_id("epent"),
                "name": e.name,
                "name_lower": nl,
                "entity_type": e.entity_type,
                "description": e.description,
                "name_embedding": new_emb_map.get(nl),
                "user_id": user_id,
                "organization_id": organization_id,
                "created_at": now,
                "updated_at": now,
            })

        # Update path: merge descriptions for entities that already exist
        update_rows: list[dict[str, Any]] = []
        for e in entities:
            nl = normalize_name(e.name)
            existing_row = existing.get(nl)
            if existing_row is None:
                continue
            old_desc = existing_row.get("description") or ""
            new_desc = e.description or ""
            if not new_desc.strip() or new_desc.strip() == old_desc.strip():
                continue
            merged, llm_used = await merge_descriptions(
                description_type="episodic entity",
                name=existing_row["name"],
                descriptions=[old_desc, new_desc] if old_desc else [new_desc],
                llm_model=llm_model,
            )
            if llm_used:
                merged_count += 1
            update_rows.append({"id": existing_row["id"], "description": merged, "updated_at": now})

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (e:EpisodicEntity {
                        id: row.id,
                        name: row.name,
                        name_lower: row.name_lower,
                        entity_type: row.entity_type,
                        description: row.description,
                        rank: 0,
                        user_id: row.user_id,
                        organization_id: row.organization_id,
                        created_at: row.created_at,
                        updated_at: row.updated_at
                    })
                    WITH e, row
                    CALL {
                        WITH e, row
                        WITH e, row WHERE row.name_embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'name_embedding', row.name_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(e) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:EpisodicEntity {id: row.id})
                    SET e.description = row.description, e.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

            # MENTIONS edges from Episode → EpisodicEntity (covers both new + existing)
            mention_rows = [
                {"episode_id": episode_id, "name_lower": normalize_name(e.name), "user_id": user_id}
                for e in entities
            ]
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (ep:Episode {id: row.episode_id})
                MATCH (e:EpisodicEntity {user_id: row.user_id, name_lower: row.name_lower})
                MERGE (ep)-[m:MENTIONS]->(e)
                ON CREATE SET m.role = 'MENTIONED'
                """,
                rows=mention_rows,
            )

        return merged_count

    async def _fetch_existing_entities(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:EpisodicEntity {user_id: $user_id, name_lower: nl})
                RETURN e.id AS id, e.name AS name, e.name_lower AS name_lower,
                       e.description AS description, e.entity_type AS entity_type
                """,
                names=name_lowers,
                user_id=user_id,
            )
            out: dict[str, dict[str, Any]] = {}
            async for rec in result:
                out[rec["name_lower"]] = dict(rec)
        return out

    # -------------------------------------------------------- W4: EP_RELATES

    async def _upsert_relations(
        self,
        driver,
        *,
        relations: list[ExtractedRelation],
        episode_id: str,
        occurred_at: datetime,
        agent_state: AgentState,
        user_id: str,
        llm_model: str,
    ) -> int:
        if not relations:
            return 0

        kw_embeddings = await embed_batch(
            [r.keywords or r.description for r in relations], agent_state
        )

        pairs = [(normalize_name(r.src), normalize_name(r.tgt)) for r in relations]
        existing_edges = await self._fetch_existing_edges(driver, user_id, pairs)

        now = iso(datetime.now(timezone.utc))
        valid_at = iso(occurred_at)
        merged_count = 0

        new_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        for r, kw_emb in zip(relations, kw_embeddings):
            a = normalize_name(r.src)
            b = normalize_name(r.tgt)
            key = tuple(sorted([a, b]))
            existing = existing_edges.get(key)
            if existing is None:
                new_rows.append({
                    "id": gen_id("eprel"),
                    "src_lower": a,
                    "tgt_lower": b,
                    "user_id": user_id,
                    "keywords": r.keywords,
                    "description": r.description,
                    "weight": float(r.weight),
                    "valid_at": valid_at,
                    "created_at": now,
                    "source_episode_ids": [episode_id],
                    "keywords_embedding": kw_emb,
                })
                continue

            # Merge description, average weight, accumulate source_episode_ids
            old_desc = existing.get("description") or ""
            new_desc = r.description or ""
            if old_desc.strip() and new_desc.strip() and old_desc.strip() != new_desc.strip():
                merged_desc, llm_used = await merge_descriptions(
                    description_type="episodic relation",
                    name=f"{r.src} <-> {r.tgt}",
                    descriptions=[old_desc, new_desc],
                    llm_model=llm_model,
                )
                if llm_used:
                    merged_count += 1
            else:
                merged_desc = new_desc or old_desc

            old_weight = float(existing.get("weight") or 0.5)
            new_weight = (old_weight + float(r.weight)) / 2.0
            old_sources: list[str] = list(existing.get("source_episode_ids") or [])
            if episode_id not in old_sources:
                old_sources.append(episode_id)
            update_rows.append({
                "id": existing["id"],
                "description": merged_desc,
                "weight": new_weight,
                "source_episode_ids": old_sources,
                "updated_at": now,
            })

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:EpisodicEntity {user_id: row.user_id, name_lower: row.src_lower})
                    MATCH (b:EpisodicEntity {user_id: row.user_id, name_lower: row.tgt_lower})
                    CREATE (a)-[r:EP_RELATES {
                        id: row.id,
                        keywords: row.keywords,
                        description: row.description,
                        weight: row.weight,
                        valid_at: row.valid_at,
                        created_at: row.created_at,
                        source_episode_ids: row.source_episode_ids
                    }]->(b)
                    WITH r, row
                    CALL {
                        WITH r, row
                        WITH r, row WHERE row.keywords_embedding IS NOT NULL
                        CALL db.create.setRelationshipVectorProperty(r, 'keywords_embedding', row.keywords_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(r) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH ()-[r:EP_RELATES {id: row.id}]->()
                    SET r.description = row.description,
                        r.weight = row.weight,
                        r.source_episode_ids = row.source_episode_ids,
                        r.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

        return merged_count

    async def _fetch_existing_edges(
        self, driver, user_id: str, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not pairs:
            return {}
        rows = [{"a": a, "b": b} for a, b in pairs]
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $rows AS row
                MATCH (x:EpisodicEntity {user_id: $user_id, name_lower: row.a})
                MATCH (y:EpisodicEntity {user_id: $user_id, name_lower: row.b})
                MATCH (x)-[r:EP_RELATES]-(y)
                WHERE r.expired_at IS NULL
                RETURN row.a AS a, row.b AS b,
                       r.id AS id, r.description AS description,
                       r.weight AS weight, r.source_episode_ids AS source_episode_ids
                """,
                rows=rows,
                user_id=user_id,
            )
            out: dict[tuple[str, str], dict[str, Any]] = {}
            async for rec in result:
                key = tuple(sorted([rec["a"], rec["b"]]))
                out.setdefault(key, {
                    "id": rec["id"],
                    "description": rec["description"],
                    "weight": rec["weight"],
                    "source_episode_ids": rec["source_episode_ids"],
                })
        return out

    # -------------------------------------------------------- W7: ranks

    async def _refresh_ranks(self, driver, *, names: list[str], user_id: str) -> None:
        if not names:
            return
        name_lowers = [normalize_name(n) for n in names]
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:EpisodicEntity {user_id: $user_id, name_lower: nl})
                OPTIONAL MATCH (e)-[r:EP_RELATES]-()
                WHERE r.expired_at IS NULL
                WITH e, count(r) AS deg
                SET e.rank = deg
                """,
                names=name_lowers,
                user_id=user_id,
            )
```

### `mirix/services/episodic_graph_retriever.py`

_Reads G_episodic: dual-level + MENTIONS reverse + NEXT one-hop_

```python
"""
Episodic graph retriever (v4) — reads G_episodic in Neo4j.

Pipeline:
  1. ll embedding → ep_entity_name_emb vector → seed EpisodicEntities + 1-hop EP_RELATES
  2. hl embedding → ep_rel_kw_emb vector → seed EP_RELATES + both endpoints
  3. Round-robin merge entities, round-robin merge relations
  4. MENTIONS reverse: entities → Episodes that mention them
  5. NEXT one-hop expansion: each Episode → ±1 temporal neighbors
  6. Score + dedup items
  7. PG fetch full episode details (summary + details + occurred_at)
"""

from __future__ import annotations

from typing import Optional

from mirix.log import get_logger
from mirix.services._graph_retriever_base import (
    DEFAULT_TOP_K,
    GraphRetrieverBase,
    GraphSearchResult,
    ItemHit,
    final_score,
)

logger = get_logger(__name__)


class EpisodicRetriever(GraphRetrieverBase):
    ENTITY_LABEL = "EpisodicEntity"
    ITEM_LABEL = "Episode"
    REL_TYPE = "EP_RELATES"
    ENTITY_VECTOR_INDEX = "ep_entity_name_emb"
    REL_VECTOR_INDEX = "ep_rel_kw_emb"
    SECTION_TITLE = "Episodic"

    async def retrieve(
        self,
        *,
        driver,
        user_id: str,
        ll_embedding: Optional[list[float]],
        hl_embedding: Optional[list[float]],
        top_k: int = DEFAULT_TOP_K,
        item_top_k: int = 15,
    ) -> GraphSearchResult:
        """Full episodic pipeline: search → MENTIONS reverse → NEXT one-hop → PG fetch."""
        entities, relations = await self.search(
            driver=driver,
            user_id=user_id,
            ll_embedding=ll_embedding,
            hl_embedding=hl_embedding,
            top_k=top_k,
        )

        # Reverse MENTIONS: get Episodes that mention the surviving entities
        entity_ids = [e.id for e in entities]
        episodes_via_mentions = await self._fetch_episodes_via_mentions(
            driver, user_id=user_id, entity_ids=entity_ids, limit=item_top_k * 2,
        )

        # NEXT one-hop expansion
        episode_ids = [it.id for it in episodes_via_mentions]
        episodes_via_one_hop = await self._fetch_episodes_one_hop(
            driver, user_id=user_id, episode_ids=episode_ids, limit=item_top_k,
        )

        # Merge + dedup
        seen_ids: set[str] = set()
        merged_items: list[ItemHit] = []
        for it in episodes_via_mentions + episodes_via_one_hop:
            if it.id in seen_ids:
                continue
            seen_ids.add(it.id)
            merged_items.append(it)

        # Score & sort by recency-aware score
        for it in merged_items:
            it.score = final_score(it.cosine, it.timestamp)
        merged_items.sort(key=lambda x: x.score, reverse=True)
        merged_items = merged_items[:item_top_k]

        # PG fetch full details for the kept items (summary already in graph;
        # PG has details). Best-effort — degrade gracefully if PG miss.
        await self._enrich_with_pg(merged_items, user_id=user_id)

        return GraphSearchResult(entities=entities, relations=relations, items=merged_items)

    async def _fetch_episodes_via_mentions(
        self, driver, *, user_id: str, entity_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not entity_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (e:EpisodicEntity {id: eid})<-[:MENTIONS]-(ep:Episode {user_id: $user_id})
                WITH DISTINCT ep
                ORDER BY ep.occurred_at DESC
                LIMIT $limit
                RETURN ep.id AS id, ep.summary AS summary, ep.occurred_at AS occurred_at
                """,
                eids=entity_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Episode",
                    summary=rec["summary"] or "", detail="",
                    timestamp=rec["occurred_at"], cosine=0.5, source="mentions",
                )
                async for rec in result
            ]

    async def _fetch_episodes_one_hop(
        self, driver, *, user_id: str, episode_ids: list[str], limit: int
    ) -> list[ItemHit]:
        """For each Episode, fetch its NEXT predecessor/successor (±1 hop)."""
        if not episode_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (ep:Episode {id: eid})
                OPTIONAL MATCH (ep)-[:NEXT]->(next:Episode {user_id: $user_id})
                OPTIONAL MATCH (prev:Episode {user_id: $user_id})-[:NEXT]->(ep)
                WITH collect(DISTINCT next) + collect(DISTINCT prev) AS neighbors
                UNWIND neighbors AS n
                WITH n WHERE n IS NOT NULL
                RETURN DISTINCT n.id AS id, n.summary AS summary, n.occurred_at AS occurred_at
                LIMIT $limit
                """,
                eids=episode_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Episode",
                    summary=rec["summary"] or "", detail="",
                    timestamp=rec["occurred_at"], cosine=0.3, source="one_hop",
                )
                async for rec in result
            ]

    async def _enrich_with_pg(self, items: list[ItemHit], *, user_id: str) -> None:
        """Pull full episodic_memory.details for kept items. Graceful on miss."""
        if not items:
            return
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        ids = [it.id for it in items]
        try:
            async with db_context() as session:
                result = await session.execute(
                    sa_text(
                        "SELECT id, details FROM episodic_memory "
                        "WHERE user_id = :u AND id = ANY(:ids)"
                    ),
                    {"u": user_id, "ids": ids},
                )
                detail_map = {row[0]: (row[1] or "") for row in result.fetchall()}
        except Exception as e:
            logger.debug("PG enrich for episodic failed: %s", e)
            return

        for it in items:
            if it.id in detail_map:
                it.detail = detail_map[it.id]
```

### `mirix/services/graph_retriever_dispatcher.py`

_Parallel dispatch to both retrievers + 50/50 budget split + combined markdown_

```python
"""
Top-level dispatcher that runs both graph retrievers in parallel.

Entry point from rest_api.retrieve_memories_by_keywords. Owns:
- keyword extraction (1 LLM call, cached, shared between graphs)
- batch embed [ll_kw, hl_kw] (1 API call)
- parallel dispatch to EpisodicRetriever + SemanticRetriever
- token-budget split (50/50 between graphs)
- combined markdown formatting

Returns an empty string when graph memory is disabled, when Neo4j is down,
or when no hits across either graph. Callers treat empty as "no graph context".
"""

from __future__ import annotations

import asyncio
from typing import Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import embed_batch, llm_model_from_agent
from mirix.services._graph_retriever_base import (
    GraphSearchResult,
    apply_budget_to_search,
    fmt_date,
)
from mirix.services.episodic_graph_retriever import EpisodicRetriever
from mirix.services.lightrag_keyword_extractor import extract_keywords
from mirix.services.semantic_graph_retriever import SemanticRetriever
from mirix.settings import settings

logger = get_logger(__name__)


# Total token budget across both graphs (split 50/50 per Q2 decision).
DEFAULT_MAX_TOTAL_TOKENS = 12000


class GraphRetrieverDispatcher:
    """Stateless. Create one per request."""

    async def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        agent_state: AgentState,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        top_k: int = 30,
        item_top_k: int = 15,
    ) -> str:
        """Full v4 retrieval. Returns markdown context string."""
        if not settings.enable_graph_memory:
            return ""

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return ""

        # ─── Step 1: keyword extraction (1 LLM call, cached) ───────────────
        llm_model = llm_model_from_agent(agent_state)
        kw = await extract_keywords(query or "", user_id=user_id, llm_model=llm_model)

        # ─── Step 2: batch embed [ll, hl] ──────────────────────────────────
        ll_str = ", ".join(kw.low_level) if kw.low_level else ""
        hl_str = ", ".join(kw.high_level) if kw.high_level else ""
        texts: list[str] = []
        purposes: list[str] = []
        if ll_str:
            texts.append(ll_str); purposes.append("ll")
        if hl_str:
            texts.append(hl_str); purposes.append("hl")

        emb_by_purpose: dict[str, Optional[list[float]]] = {"ll": None, "hl": None}
        if texts:
            embeddings = await embed_batch(texts, agent_state)
            for p, e in zip(purposes, embeddings):
                emb_by_purpose[p] = e

        ll_emb = emb_by_purpose["ll"]
        hl_emb = emb_by_purpose["hl"]

        if ll_emb is None and hl_emb is None:
            logger.info("Graph retrieve: no embeddings → empty context")
            return ""

        # ─── Step 3: dispatch both retrievers in parallel ──────────────────
        ep_task = asyncio.create_task(
            EpisodicRetriever().retrieve(
                driver=driver, user_id=user_id,
                ll_embedding=ll_emb, hl_embedding=hl_emb,
                top_k=top_k, item_top_k=item_top_k,
            )
        )
        sem_task = asyncio.create_task(
            SemanticRetriever().retrieve(
                driver=driver, user_id=user_id,
                ll_embedding=ll_emb, hl_embedding=hl_emb,
                top_k=top_k, item_top_k=item_top_k,
            )
        )
        ep_result, sem_result = await asyncio.gather(ep_task, sem_task, return_exceptions=True)

        if isinstance(ep_result, Exception):
            logger.warning("Episodic retrieve failed: %s", ep_result)
            ep_result = GraphSearchResult()
        if isinstance(sem_result, Exception):
            logger.warning("Semantic retrieve failed: %s", sem_result)
            sem_result = GraphSearchResult()

        # ─── Step 4: token budget split 50/50, then format ─────────────────
        per_graph_budget = max_total_tokens // 2
        # Within each graph, split: 30% entity, 35% relations, 35% items
        e_budget = int(per_graph_budget * 0.30)
        r_budget = int(per_graph_budget * 0.35)
        i_budget = per_graph_budget - e_budget - r_budget

        ep_trim = apply_budget_to_search(
            ep_result, max_entity_tokens=e_budget,
            max_relation_tokens=r_budget, max_item_tokens=i_budget,
        )
        sem_trim = apply_budget_to_search(
            sem_result, max_entity_tokens=e_budget,
            max_relation_tokens=r_budget, max_item_tokens=i_budget,
        )

        ep_md = _format_section(ep_trim, "Episodic")
        sem_md = _format_section(sem_trim, "Semantic")

        parts = []
        if ep_md:
            parts.append(ep_md)
        if sem_md:
            parts.append(sem_md)
        ctx = "\n\n".join(parts)
        logger.info(
            "Graph retrieve: ep[%dE/%dR/%dI] sem[%dE/%dR/%dI] total %d chars",
            len(ep_trim.entities), len(ep_trim.relations), len(ep_trim.items),
            len(sem_trim.entities), len(sem_trim.relations), len(sem_trim.items),
            len(ctx),
        )
        return ctx


def _format_section(s: GraphSearchResult, title: str) -> str:
    if not (s.entities or s.relations or s.items):
        return ""
    lines = [f"## {title} Knowledge Graph"]
    if s.entities:
        lines.append("### Entities")
        for e in s.entities:
            lines.append(f"- {e.name} ({e.entity_type}, rank={e.rank}): {e.description}")
    if s.relations:
        lines.append("\n### Relationships")
        for r in s.relations:
            validity = f" (on/since {fmt_date(r.valid_at)})" if r.valid_at else ""
            lines.append(
                f"- {r.src_name} <-> {r.tgt_name} [{r.keywords}]: {r.description}{validity}"
            )
    if s.items:
        item_label = "Episodes" if title == "Episodic" else "Concepts"
        lines.append(f"\n### Related {item_label}")
        for it in s.items:
            ts = fmt_date(it.timestamp) if it.timestamp else ""
            ts_part = f"[{ts}] " if ts else ""
            head = f"- {ts_part}{it.summary}".rstrip()
            lines.append(head)
            if it.detail and it.detail != it.summary:
                lines.append(f"  {it.detail[:400]}")
    return "\n".join(lines)
```

### `mirix/services/lightrag_extractor.py`

_LLM-driven entity/relation extraction with delimiter parsing_

```python
"""
LightRAG-style entity & relation extractor (W2 of the write path).

Adapted from LightRAG operate.py:extract_entities. One LLM call per event;
output is delimiter-separated tuples that are parsed into structured dicts.
Optional gleaning pass (default off) re-prompts the LLM to catch misses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import (
    COMPLETION_DELIMITER,
    DEFAULT_ENTITY_TYPES,
    TUPLE_DELIMITER,
    render_extraction_system_prompt,
    render_extraction_user_prompt,
)

logger = get_logger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str


@dataclass
class ExtractedRelation:
    src: str
    tgt: str
    keywords: str
    description: str
    weight: float


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
        return s[1:-1].strip()
    return s


def _coerce_weight(raw: str) -> float:
    """Parse the trailing relationship_strength field. Defaults to 0.5 on bad input."""
    try:
        v = float(_strip_quotes(raw))
        if 0.0 <= v <= 1.0:
            return v
        # Some models emit 0..10 or 0..100. Normalize.
        if 1.0 < v <= 10.0:
            return v / 10.0
        if 10.0 < v <= 100.0:
            return v / 100.0
    except (ValueError, TypeError):
        pass
    return 0.5


def parse_extraction_output(raw: str) -> ExtractionResult:
    """
    Parse LightRAG-style delimiter output into structured entities & relations.

    Each line should look like:
        entity<|#|>NAME<|#|>TYPE<|#|>DESCRIPTION
        relation<|#|>SRC<|#|>TGT<|#|>KEYWORDS<|#|>DESCRIPTION<|#|>STRENGTH
    Lines that do not parse cleanly are logged and skipped.
    """
    result = ExtractionResult()
    if not raw:
        return result

    # Stop at the completion delimiter if the model emitted it
    cut = raw.find(COMPLETION_DELIMITER)
    if cut >= 0:
        raw = raw[:cut]

    seen_entity_names: set[str] = set()
    seen_relation_keys: set[tuple[str, str]] = set()

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or TUPLE_DELIMITER not in line:
            continue
        parts = [p.strip() for p in line.split(TUPLE_DELIMITER)]
        kind = parts[0].lower().strip("()`* ")
        if kind == "entity" and len(parts) >= 4:
            name = _strip_quotes(parts[1])
            entity_type = _strip_quotes(parts[2]) or "Other"
            description = _strip_quotes(parts[3])
            if not name or name in seen_entity_names:
                continue
            seen_entity_names.add(name)
            result.entities.append(
                ExtractedEntity(name=name, entity_type=entity_type, description=description)
            )
        elif kind == "relation" and len(parts) >= 5:
            src = _strip_quotes(parts[1])
            tgt = _strip_quotes(parts[2])
            keywords = _strip_quotes(parts[3])
            description = _strip_quotes(parts[4])
            weight = _coerce_weight(parts[5]) if len(parts) >= 6 else 0.5
            if not src or not tgt or src == tgt:
                continue
            # Treat undirected; dedup on sorted endpoints
            key = tuple(sorted([src.lower(), tgt.lower()]))
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            result.relations.append(
                ExtractedRelation(
                    src=src,
                    tgt=tgt,
                    keywords=keywords,
                    description=description,
                    weight=weight,
                )
            )
        else:
            # Unknown leading token — skip silently to avoid log spam on
            # benign formatting variations.
            continue

    return result


async def call_openai_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Bare-metal OpenAI chat completion. Mirrors v2 graph_memory_manager."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    endpoint = f"{api_base.rstrip('/')}/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # Record token usage if a phase is active (no-op outside instrumented evals)
    try:
        from mirix.database.token_tracker import record as _record_tokens
        usage = (data.get("usage") or {})
        _record_tokens(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
        )
    except Exception:
        pass

    return data["choices"][0]["message"]["content"]


async def extract_entities_and_relations(
    text: str,
    *,
    llm_model: str = "gpt-4.1-mini",
    entity_types: Optional[list[str]] = None,
    language: str = "English",
    max_input_chars: int = 12000,
) -> ExtractionResult:
    """
    Run a single LLM extraction pass over ``text`` and parse the result.

    Returns an empty ``ExtractionResult`` on error so the caller can carry on.
    """
    if not text or not text.strip():
        return ExtractionResult()

    types = entity_types or DEFAULT_ENTITY_TYPES
    system_prompt = render_extraction_system_prompt(entity_types=types, language=language)
    user_prompt = render_extraction_user_prompt(
        input_text=text[:max_input_chars],
        entity_types=types,
        language=language,
    )

    try:
        raw = await call_openai_chat(system_prompt, user_prompt, model=llm_model)
    except Exception as e:
        logger.warning("LightRAG extraction LLM call failed: %s", e)
        return ExtractionResult()

    parsed = parse_extraction_output(raw)
    logger.info(
        "LightRAG extraction: %d entities, %d relations from %d chars",
        len(parsed.entities),
        len(parsed.relations),
        len(text),
    )
    return parsed
```

### `mirix/services/lightrag_keyword_extractor.py`

_Query keyword extraction (ll/hl), Redis-cached_

```python
"""
LightRAG-style query keyword extractor (high-level / low-level split).

One LLM call per unique query, cached in Redis (or skipped if Redis is not
available — the system still works, just pays the extraction cost each time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import render_keywords_extraction_prompt
from mirix.services.lightrag_extractor import call_openai_chat

logger = get_logger(__name__)


# Match LightRAG defaults (24h is long enough for typical chat sessions).
KEYWORD_CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class Keywords:
    high_level: list[str]
    low_level: list[str]


def _cache_key(user_id: str, query: str, language: str) -> str:
    h = hashlib.sha1(f"{language}|{query}".encode("utf-8")).hexdigest()[:24]
    return f"mirix:lightrag:kw:{user_id}:{h}"


def _parse_json_loose(raw: str) -> Optional[dict]:
    """Try strict JSON first; if that fails, strip code fences and retry."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip ``` fences
    if "```" in raw:
        try:
            body = raw.split("```json")[-1] if "```json" in raw else raw.split("```")[1]
            body = body.split("```")[0]
            return json.loads(body.strip())
        except (json.JSONDecodeError, IndexError):
            pass
    return None


async def _cache_get(key: str) -> Optional[Keywords]:
    try:
        from mirix.database.cache_provider import get_cache_provider

        provider = get_cache_provider()
        if provider is None:
            return None
        data = await provider.get_json(key)
        if not data:
            return None
        return Keywords(
            high_level=list(data.get("high_level", []) or []),
            low_level=list(data.get("low_level", []) or []),
        )
    except Exception as e:
        logger.debug("Keyword cache get failed: %s", e)
        return None


async def _cache_set(key: str, kw: Keywords) -> None:
    try:
        from mirix.database.cache_provider import get_cache_provider

        provider = get_cache_provider()
        if provider is None:
            return
        await provider.set_json(
            key,
            {"high_level": kw.high_level, "low_level": kw.low_level},
            ttl=KEYWORD_CACHE_TTL_SECONDS,
        )
    except Exception as e:
        logger.debug("Keyword cache set failed: %s", e)


def _fallback_keywords(query: str) -> Keywords:
    """When the LLM returns nothing useful, treat the query itself as ll keyword.

    Mirrors LightRAG operate.py:get_keywords_from_query short-query fallback.
    """
    q = (query or "").strip()
    if not q:
        return Keywords(high_level=[], low_level=[])
    if len(q) < 50:
        return Keywords(high_level=[], low_level=[q])
    # Long but empty parse: keep first few content words as best-effort.
    words = [w for w in q.split() if len(w) > 3][:6]
    return Keywords(high_level=[], low_level=words or [q[:80]])


async def extract_keywords(
    query: str,
    *,
    user_id: str,
    llm_model: str = "gpt-4.1-mini",
    language: str = "English",
    use_cache: bool = True,
) -> Keywords:
    """
    Return (high_level, low_level) keyword lists for ``query``.

    On any failure or empty model output, falls back to using the query itself
    as a single low-level keyword (short queries) or splitting into content
    words (long queries). Never raises.
    """
    if not query or not query.strip():
        return Keywords(high_level=[], low_level=[])

    cache_key = _cache_key(user_id, query, language)
    if use_cache:
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

    prompt = render_keywords_extraction_prompt(query=query, language=language)

    try:
        raw = await call_openai_chat(
            system_prompt="You are a precise keyword extractor. Output JSON only.",
            user_prompt=prompt,
            model=llm_model,
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        logger.warning("Keyword extraction LLM call failed: %s", e)
        return _fallback_keywords(query)

    parsed = _parse_json_loose(raw)
    if not parsed:
        logger.warning("Keyword extraction returned unparsable output: %s", (raw or "")[:120])
        return _fallback_keywords(query)

    kw = Keywords(
        high_level=[k.strip() for k in parsed.get("high_level_keywords", []) if k and k.strip()],
        low_level=[k.strip() for k in parsed.get("low_level_keywords", []) if k and k.strip()],
    )
    if not kw.high_level and not kw.low_level:
        kw = _fallback_keywords(query)

    if use_cache:
        await _cache_set(cache_key, kw)
    return kw
```

### `mirix/services/lightrag_merger.py`

_Map-reduce description merge (cheap path → LLM summary → recursive)_

```python
"""
Description merging for entities and relations (W3/W4 helper).

Adapted from LightRAG operate.py:_handle_entity_relation_summary. The strategy:

1. If the descriptions, joined, fit within ``summary_context_size`` tokens AND
   there are fewer than ``force_llm_summary_on_merge`` of them → just join with
   a separator. No LLM call.
2. If the joined text fits within ``summary_max_tokens`` → ask the LLM for a
   single summary. 1 LLM call.
3. Otherwise → split into chunks, summarize each, recurse on the summaries.

Token counts are estimated with tiktoken (cl100k_base) for cheap accuracy.
"""

from __future__ import annotations

from typing import Optional

import tiktoken

from mirix.log import get_logger
from mirix.prompts.lightrag_prompts import render_summarize_descriptions_prompt
from mirix.services.lightrag_extractor import call_openai_chat

logger = get_logger(__name__)


# Defaults align with LightRAG's recommended values. Tuned smaller to keep
# write-path cost low (MIRIX writes much more often than LightRAG ingests docs).
DEFAULT_SUMMARY_CONTEXT_SIZE = 1000  # tokens — when joined desc still fits, no summary
DEFAULT_SUMMARY_MAX_TOKENS = 500     # tokens — target output length
DEFAULT_FORCE_LLM_MERGE_AT = 6       # description count threshold
DEFAULT_SEPARATOR = " | "

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


async def merge_descriptions(
    description_type: str,
    name: str,
    descriptions: list[str],
    *,
    llm_model: str = "gpt-4.1-mini",
    summary_context_size: int = DEFAULT_SUMMARY_CONTEXT_SIZE,
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    force_llm_merge_at: int = DEFAULT_FORCE_LLM_MERGE_AT,
    separator: str = DEFAULT_SEPARATOR,
    max_recursion: int = 4,
) -> tuple[str, bool]:
    """
    Merge a list of descriptions for a single entity or relation.

    Returns ``(merged_text, llm_used)``. ``llm_used`` lets the caller decide
    whether to bump cache invalidation timestamps.
    """
    descs = [d.strip() for d in descriptions if d and d.strip()]
    if not descs:
        return "", False
    if len(descs) == 1:
        return descs[0], False

    # Phase 1: cheap path — no LLM if small enough and few enough.
    joined = separator.join(descs)
    total_tokens = _count_tokens(joined)
    if total_tokens <= summary_context_size and len(descs) < force_llm_merge_at:
        return joined, False

    # Phase 2: single LLM summary if it all fits as a prompt.
    if total_tokens <= summary_max_tokens * 4:  # rough budget for prompt+output
        summary = await _summarize_via_llm(
            description_type=description_type,
            name=name,
            descriptions=descs,
            llm_model=llm_model,
            summary_max_tokens=summary_max_tokens,
        )
        return summary or joined[: summary_max_tokens * 4], True

    # Phase 3: map-reduce. Chunk descs into groups whose joined size fits, then
    # summarize each chunk, then recurse on the chunk summaries.
    if max_recursion <= 0:
        # Hard stop: just truncate the joined text. Avoids unbounded recursion
        # on pathological input.
        return joined[: summary_max_tokens * 4], False

    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for d in descs:
        d_tokens = _count_tokens(d)
        if current and current_tokens + d_tokens > summary_context_size:
            chunks.append(current)
            current, current_tokens = [d], d_tokens
        else:
            current.append(d)
            current_tokens += d_tokens
    if current:
        chunks.append(current)

    chunk_summaries: list[str] = []
    llm_used = False
    for ch in chunks:
        if len(ch) == 1:
            chunk_summaries.append(ch[0])
            continue
        s = await _summarize_via_llm(
            description_type=description_type,
            name=name,
            descriptions=ch,
            llm_model=llm_model,
            summary_max_tokens=summary_max_tokens,
        )
        if s:
            chunk_summaries.append(s)
            llm_used = True
        else:
            # Fallback: keep raw join of this chunk
            chunk_summaries.append(separator.join(ch))

    # Recurse on the chunk summaries (now fewer items, each smaller).
    final, recurse_used = await merge_descriptions(
        description_type=description_type,
        name=name,
        descriptions=chunk_summaries,
        llm_model=llm_model,
        summary_context_size=summary_context_size,
        summary_max_tokens=summary_max_tokens,
        force_llm_merge_at=force_llm_merge_at,
        separator=separator,
        max_recursion=max_recursion - 1,
    )
    return final, llm_used or recurse_used


async def _summarize_via_llm(
    description_type: str,
    name: str,
    descriptions: list[str],
    llm_model: str,
    summary_max_tokens: int,
) -> Optional[str]:
    """One LLM call to merge ``descriptions`` into a single paragraph."""
    prompt = render_summarize_descriptions_prompt(
        description_type=description_type,
        description_name=name,
        description_list=descriptions,
        summary_length=summary_max_tokens,
    )
    try:
        # Use a tiny system prompt; the user prompt carries the full template.
        return (
            await call_openai_chat(
                system_prompt="You are a precise summarizer.",
                user_prompt=prompt,
                model=llm_model,
                max_tokens=summary_max_tokens + 200,
            )
        ).strip()
    except Exception as e:
        logger.warning("Description merge LLM call failed for %s '%s': %s", description_type, name, e)
        return None
```

### `mirix/services/semantic_graph_manager.py`

_Writes G_semantic: Concept, SemanticEntity, CONCEPT_RELATES, SEM_RELATES_

```python
"""
Semantic graph manager (v4) — writes G_semantic in Neo4j.

Hooked from SemanticMemoryManager.insert_semantic_item after the PG row has
been committed. Failures are non-fatal.

Graph elements written here:
  (:Concept {id, user_id, organization_id, name, summary, created_at})
  (:SemanticEntity {id, user_id, organization_id, name, name_lower,
                    entity_type, description, rank, name_embedding,
                    created_at, updated_at})
  (:Concept)-[:CONCEPT_RELATES {keywords, description, weight,
                                 keywords_embedding}]->(:Concept)
  (:Concept)-[:MENTIONS]->(:SemanticEntity)
  (:SemanticEntity)-[:SEM_RELATES {id, keywords, description, weight,
                                    source_concept_ids, keywords_embedding}]
                                    ->(:SemanticEntity)

Concept-Concept edges are LLM-judged: when a new Concept is inserted, the
top-K most similar existing Concepts (by name embedding) are candidates;
one LLM call decides which actually have a meaningful relationship.

Cost per insert: ~1 LLM call (entity extraction) + ~0.3-1 LLM call (description
merging + concept relation judgement). Heavier than episodic by design — the
semantic graph is small and dense, so investing in good edges pays off.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from mirix.log import get_logger
from mirix.schemas.agent import AgentState
from mirix.services._graph_common import (
    embed_batch,
    gen_id,
    iso,
    llm_model_from_agent,
    normalize_name,
)
from mirix.services.lightrag_extractor import (
    ExtractedEntity,
    ExtractedRelation,
    call_openai_chat,
    extract_entities_and_relations,
)
from mirix.services.lightrag_merger import merge_descriptions
from mirix.settings import settings

logger = get_logger(__name__)


# Concept-concept relation candidate pool size. Top-K nearest concepts (by
# name embedding) get sent to the LLM for relation judgement in one batch.
DEFAULT_CONCEPT_REL_TOP_K = 5

# Concept-concept relation judgement prompt. Asks the LLM to return JSON for
# which candidates actually relate to the new concept and how.
_CONCEPT_REL_PROMPT = """You are a knowledge graph editor. A new concept has been added to the user's semantic memory. Decide which of the candidate concepts have a meaningful relationship with the new one.

New concept:
  name: {new_name}
  summary: {new_summary}

Candidate concepts (existing in the graph):
{candidates_block}

For each candidate that genuinely relates to the new concept (e.g. IS_A, PART_OF, RELATES_TO, CONTRADICTS, ENABLES, CAUSED_BY), output one JSON object per line. Skip candidates that are unrelated or duplicates. Output strict JSON, one object per line, no markdown fences. If nothing relates, output nothing.

Each object must have:
  "candidate_name": str  // exact name from the list above
  "keywords": str        // short phrase summarizing the relation type (e.g. "subclass", "part of", "contradicts")
  "description": str     // one sentence explaining the relationship
  "weight": float        // 0.0-1.0 strength
"""


class SemanticGraphManager:
    """Stateless coordinator. Construct one per call."""

    async def process_concept(
        self,
        *,
        concept_id: str,
        name: str,
        summary: str,
        details: str,
        agent_state: AgentState,
        organization_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Run the full semantic write path. Never raises."""
        if not settings.enable_graph_memory:
            return {"skipped": "disabled"}

        from mirix.database.neo4j_client import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver is None:
            return {"skipped": "no_driver"}

        text = f"{name}: {summary}\n{details or ''}"
        llm_model = llm_model_from_agent(agent_state)

        # W2: extract entities + entity-entity relations from the concept text
        extraction = await extract_entities_and_relations(text=text, llm_model=llm_model)

        # W1: always create the Concept node
        concept_name_emb = (await embed_batch([name], agent_state))[0]
        await self._upsert_concept(
            driver,
            concept_id=concept_id,
            name=name,
            summary=summary,
            user_id=user_id,
            organization_id=organization_id,
            name_embedding=concept_name_emb,
        )

        # W6: concept-concept relation discovery
        concept_rels_added = await self._discover_concept_relations(
            driver,
            new_concept_id=concept_id,
            new_concept_name=name,
            new_concept_summary=summary,
            new_concept_name_emb=concept_name_emb,
            user_id=user_id,
            agent_state=agent_state,
            llm_model=llm_model,
        )

        if not extraction.entities and not extraction.relations:
            return {"entities": 0, "relations": 0, "concept_rels": concept_rels_added}

        # W3: upsert SemanticEntity nodes
        merged_entities = await self._upsert_entities(
            driver,
            entities=extraction.entities,
            concept_id=concept_id,
            agent_state=agent_state,
            user_id=user_id,
            organization_id=organization_id,
        )

        # W4: upsert SEM_RELATES edges
        merged_relations = await self._upsert_relations(
            driver,
            relations=extraction.relations,
            concept_id=concept_id,
            agent_state=agent_state,
            user_id=user_id,
            llm_model=llm_model,
        )

        # W7: refresh rank
        await self._refresh_ranks(
            driver,
            names=sorted({e.name for e in extraction.entities}),
            user_id=user_id,
        )

        return {
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
            "concept_rels": concept_rels_added,
            "merged_entities": merged_entities,
            "merged_relations": merged_relations,
        }

    # --------------------------------------------------------- W1: Concept

    async def _upsert_concept(
        self,
        driver,
        *,
        concept_id: str,
        name: str,
        summary: str,
        user_id: str,
        organization_id: str,
        name_embedding: Optional[list[float]],
    ) -> None:
        now = iso(datetime.now(timezone.utc))
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                MERGE (c:Concept {id: $id})
                ON CREATE SET c.user_id = $user_id,
                              c.organization_id = $org_id,
                              c.name = $name,
                              c.summary = $summary,
                              c.created_at = $now
                ON MATCH SET c.name = $name, c.summary = $summary
                """,
                id=concept_id,
                user_id=user_id,
                org_id=organization_id,
                name=name,
                summary=summary or "",
                now=now,
            )
            if name_embedding:
                # Attach name embedding to the concept itself so we can find
                # similar concepts via vector search later.
                await session.run(
                    """
                    MATCH (c:Concept {id: $id})
                    CALL db.create.setNodeVectorProperty(c, 'name_embedding', $emb)
                    RETURN count(*) AS _
                    """,
                    id=concept_id,
                    emb=name_embedding,
                )

    # ----------------------------------------- W6: concept-concept relations

    async def _discover_concept_relations(
        self,
        driver,
        *,
        new_concept_id: str,
        new_concept_name: str,
        new_concept_summary: str,
        new_concept_name_emb: Optional[list[float]],
        user_id: str,
        agent_state: AgentState,
        llm_model: str,
        top_k: int = DEFAULT_CONCEPT_REL_TOP_K,
    ) -> int:
        """Find candidate concepts by name vector, ask LLM which actually relate."""
        if new_concept_name_emb is None:
            return 0

        # Find top-K similar concepts via raw cypher (Concept nodes don't yet
        # have a dedicated vector index by design — we use cosine over the
        # property we set above; small graphs make this affordable). For
        # larger deployments switch to a dedicated index on Concept.
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (c:Concept {user_id: $user_id})
                WHERE c.id <> $id AND c.name_embedding IS NOT NULL
                WITH c, vector.similarity.cosine(c.name_embedding, $emb) AS sim
                ORDER BY sim DESC LIMIT $k
                RETURN c.id AS id, c.name AS name, c.summary AS summary, sim
                """,
                user_id=user_id,
                id=new_concept_id,
                emb=new_concept_name_emb,
                k=top_k,
            )
            candidates = [dict(rec) async for rec in result]

        if not candidates:
            return 0

        candidates_block = "\n".join(
            f"  - {c['name']}: {(c.get('summary') or '')[:200]}" for c in candidates
        )
        prompt = _CONCEPT_REL_PROMPT.format(
            new_name=new_concept_name,
            new_summary=(new_concept_summary or "")[:300],
            candidates_block=candidates_block,
        )

        try:
            raw = await call_openai_chat(
                system_prompt="You are a precise knowledge graph editor. Output JSON only.",
                user_prompt=prompt,
                model=llm_model,
                temperature=0.0,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("Concept relation LLM call failed: %s", e)
            return 0

        # Parse line-delimited JSON, tolerant to LLM noise
        relations: list[dict[str, Any]] = []
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-").strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cand_name = (obj.get("candidate_name") or "").strip()
            if not cand_name:
                continue
            # Find candidate id by name (case-insensitive)
            cand = next((c for c in candidates if c["name"].lower() == cand_name.lower()), None)
            if cand is None:
                continue
            relations.append({
                "src_id": new_concept_id,
                "tgt_id": cand["id"],
                "keywords": (obj.get("keywords") or "")[:120],
                "description": (obj.get("description") or "")[:500],
                "weight": float(obj.get("weight") or 0.5),
            })

        if not relations:
            return 0

        # Embed keywords for the new edges
        kw_embs = await embed_batch([r["keywords"] or r["description"] for r in relations], agent_state)
        for r, emb in zip(relations, kw_embs):
            r["keywords_embedding"] = emb

        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Concept {id: row.src_id})
                MATCH (b:Concept {id: row.tgt_id})
                MERGE (a)-[r:CONCEPT_RELATES]->(b)
                ON CREATE SET r.keywords = row.keywords,
                              r.description = row.description,
                              r.weight = row.weight
                ON MATCH SET r.keywords = row.keywords,
                             r.description = row.description,
                             r.weight = (coalesce(r.weight, 0.5) + row.weight) / 2.0
                WITH r, row
                CALL {
                    WITH r, row
                    WITH r, row WHERE row.keywords_embedding IS NOT NULL
                    CALL db.create.setRelationshipVectorProperty(r, 'keywords_embedding', row.keywords_embedding)
                    RETURN count(*) AS _
                }
                RETURN count(r) AS created
                """,
                rows=relations,
            )

        return len(relations)

    # --------------------------------------------------- W3: SemanticEntity

    async def _upsert_entities(
        self,
        driver,
        *,
        entities: list[ExtractedEntity],
        concept_id: str,
        agent_state: AgentState,
        user_id: str,
        organization_id: str,
    ) -> int:
        if not entities:
            return 0

        name_lowers = [normalize_name(e.name) for e in entities]
        existing = await self._fetch_existing_entities(driver, user_id, name_lowers)

        new_entities = [e for e in entities if normalize_name(e.name) not in existing]
        new_embeddings = await embed_batch([e.name for e in new_entities], agent_state)
        new_emb_map: dict[str, Optional[list[float]]] = {
            normalize_name(e.name): emb for e, emb in zip(new_entities, new_embeddings)
        }

        now = iso(datetime.now(timezone.utc))
        merged_count = 0
        llm_model = llm_model_from_agent(agent_state)

        new_rows: list[dict[str, Any]] = []
        for e in new_entities:
            nl = normalize_name(e.name)
            new_rows.append({
                "id": gen_id("sement"),
                "name": e.name,
                "name_lower": nl,
                "entity_type": e.entity_type,
                "description": e.description,
                "name_embedding": new_emb_map.get(nl),
                "user_id": user_id,
                "organization_id": organization_id,
                "created_at": now,
                "updated_at": now,
            })

        update_rows: list[dict[str, Any]] = []
        for e in entities:
            nl = normalize_name(e.name)
            existing_row = existing.get(nl)
            if existing_row is None:
                continue
            old_desc = existing_row.get("description") or ""
            new_desc = e.description or ""
            if not new_desc.strip() or new_desc.strip() == old_desc.strip():
                continue
            merged, llm_used = await merge_descriptions(
                description_type="semantic entity",
                name=existing_row["name"],
                descriptions=[old_desc, new_desc] if old_desc else [new_desc],
                llm_model=llm_model,
            )
            if llm_used:
                merged_count += 1
            update_rows.append({"id": existing_row["id"], "description": merged, "updated_at": now})

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (e:SemanticEntity {
                        id: row.id,
                        name: row.name,
                        name_lower: row.name_lower,
                        entity_type: row.entity_type,
                        description: row.description,
                        rank: 0,
                        user_id: row.user_id,
                        organization_id: row.organization_id,
                        created_at: row.created_at,
                        updated_at: row.updated_at
                    })
                    WITH e, row
                    CALL {
                        WITH e, row
                        WITH e, row WHERE row.name_embedding IS NOT NULL
                        CALL db.create.setNodeVectorProperty(e, 'name_embedding', row.name_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(e) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:SemanticEntity {id: row.id})
                    SET e.description = row.description, e.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

            # MENTIONS edges from Concept → SemanticEntity
            mention_rows = [
                {"concept_id": concept_id, "name_lower": normalize_name(e.name), "user_id": user_id}
                for e in entities
            ]
            await session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Concept {id: row.concept_id})
                MATCH (e:SemanticEntity {user_id: row.user_id, name_lower: row.name_lower})
                MERGE (c)-[m:MENTIONS]->(e)
                """,
                rows=mention_rows,
            )

        return merged_count

    async def _fetch_existing_entities(
        self, driver, user_id: str, name_lowers: list[str]
    ) -> dict[str, dict[str, Any]]:
        if not name_lowers:
            return {}
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:SemanticEntity {user_id: $user_id, name_lower: nl})
                RETURN e.id AS id, e.name AS name, e.name_lower AS name_lower,
                       e.description AS description, e.entity_type AS entity_type
                """,
                names=name_lowers,
                user_id=user_id,
            )
            out: dict[str, dict[str, Any]] = {}
            async for rec in result:
                out[rec["name_lower"]] = dict(rec)
        return out

    # -------------------------------------------------------- W4: SEM_RELATES

    async def _upsert_relations(
        self,
        driver,
        *,
        relations: list[ExtractedRelation],
        concept_id: str,
        agent_state: AgentState,
        user_id: str,
        llm_model: str,
    ) -> int:
        if not relations:
            return 0

        kw_embeddings = await embed_batch(
            [r.keywords or r.description for r in relations], agent_state
        )
        pairs = [(normalize_name(r.src), normalize_name(r.tgt)) for r in relations]
        existing_edges = await self._fetch_existing_edges(driver, user_id, pairs)

        now = iso(datetime.now(timezone.utc))
        merged_count = 0

        new_rows: list[dict[str, Any]] = []
        update_rows: list[dict[str, Any]] = []

        for r, kw_emb in zip(relations, kw_embeddings):
            a, b = normalize_name(r.src), normalize_name(r.tgt)
            key = tuple(sorted([a, b]))
            existing = existing_edges.get(key)
            if existing is None:
                new_rows.append({
                    "id": gen_id("semrel"),
                    "src_lower": a,
                    "tgt_lower": b,
                    "user_id": user_id,
                    "keywords": r.keywords,
                    "description": r.description,
                    "weight": float(r.weight),
                    "created_at": now,
                    "source_concept_ids": [concept_id],
                    "keywords_embedding": kw_emb,
                })
                continue

            old_desc = existing.get("description") or ""
            new_desc = r.description or ""
            if old_desc.strip() and new_desc.strip() and old_desc.strip() != new_desc.strip():
                merged_desc, llm_used = await merge_descriptions(
                    description_type="semantic relation",
                    name=f"{r.src} <-> {r.tgt}",
                    descriptions=[old_desc, new_desc],
                    llm_model=llm_model,
                )
                if llm_used:
                    merged_count += 1
            else:
                merged_desc = new_desc or old_desc

            old_weight = float(existing.get("weight") or 0.5)
            new_weight = (old_weight + float(r.weight)) / 2.0
            old_sources: list[str] = list(existing.get("source_concept_ids") or [])
            if concept_id not in old_sources:
                old_sources.append(concept_id)
            update_rows.append({
                "id": existing["id"],
                "description": merged_desc,
                "weight": new_weight,
                "source_concept_ids": old_sources,
                "updated_at": now,
            })

        async with driver.session(database=settings.neo4j_database) as session:
            if new_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:SemanticEntity {user_id: row.user_id, name_lower: row.src_lower})
                    MATCH (b:SemanticEntity {user_id: row.user_id, name_lower: row.tgt_lower})
                    CREATE (a)-[r:SEM_RELATES {
                        id: row.id,
                        keywords: row.keywords,
                        description: row.description,
                        weight: row.weight,
                        created_at: row.created_at,
                        source_concept_ids: row.source_concept_ids
                    }]->(b)
                    WITH r, row
                    CALL {
                        WITH r, row
                        WITH r, row WHERE row.keywords_embedding IS NOT NULL
                        CALL db.create.setRelationshipVectorProperty(r, 'keywords_embedding', row.keywords_embedding)
                        RETURN count(*) AS _
                    }
                    RETURN count(r) AS created
                    """,
                    rows=new_rows,
                )
            if update_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MATCH ()-[r:SEM_RELATES {id: row.id}]->()
                    SET r.description = row.description,
                        r.weight = row.weight,
                        r.source_concept_ids = row.source_concept_ids,
                        r.updated_at = row.updated_at
                    """,
                    rows=update_rows,
                )

        return merged_count

    async def _fetch_existing_edges(
        self, driver, user_id: str, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not pairs:
            return {}
        rows = [{"a": a, "b": b} for a, b in pairs]
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $rows AS row
                MATCH (x:SemanticEntity {user_id: $user_id, name_lower: row.a})
                MATCH (y:SemanticEntity {user_id: $user_id, name_lower: row.b})
                MATCH (x)-[r:SEM_RELATES]-(y)
                RETURN row.a AS a, row.b AS b,
                       r.id AS id, r.description AS description,
                       r.weight AS weight, r.source_concept_ids AS source_concept_ids
                """,
                rows=rows,
                user_id=user_id,
            )
            out: dict[tuple[str, str], dict[str, Any]] = {}
            async for rec in result:
                key = tuple(sorted([rec["a"], rec["b"]]))
                out.setdefault(key, {
                    "id": rec["id"],
                    "description": rec["description"],
                    "weight": rec["weight"],
                    "source_concept_ids": rec["source_concept_ids"],
                })
        return out

    # -------------------------------------------------------- W7: ranks

    async def _refresh_ranks(self, driver, *, names: list[str], user_id: str) -> None:
        if not names:
            return
        name_lowers = [normalize_name(n) for n in names]
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(
                """
                UNWIND $names AS nl
                MATCH (e:SemanticEntity {user_id: $user_id, name_lower: nl})
                OPTIONAL MATCH (e)-[r:SEM_RELATES]-()
                WITH e, count(r) AS deg
                SET e.rank = deg
                """,
                names=name_lowers,
                user_id=user_id,
            )
```

### `mirix/services/semantic_graph_retriever.py`

_Reads G_semantic: dual-level + MENTIONS reverse + CONCEPT_RELATES one-hop_

```python
"""
Semantic graph retriever (v4) — reads G_semantic in Neo4j.

Pipeline:
  1. ll embedding → sem_entity_name_emb vector → seed SemanticEntities + 1-hop SEM_RELATES
  2. hl embedding → sem_rel_kw_emb vector → seed SEM_RELATES + endpoints
  3. Round-robin merge
  4. MENTIONS reverse: entities → Concepts that mention them
  5. CONCEPT_RELATES one-hop: each Concept → adjacent Concepts
  6. Score + dedup
  7. PG fetch full concept details

Unlike episodic, there is no timestamp ordering — concepts are ordered by
cosine score (recency_decay defaults to 0.5 when timestamp is missing).
"""

from __future__ import annotations

from typing import Optional

from mirix.log import get_logger
from mirix.services._graph_retriever_base import (
    DEFAULT_TOP_K,
    GraphRetrieverBase,
    GraphSearchResult,
    ItemHit,
    final_score,
)

logger = get_logger(__name__)


class SemanticRetriever(GraphRetrieverBase):
    ENTITY_LABEL = "SemanticEntity"
    ITEM_LABEL = "Concept"
    REL_TYPE = "SEM_RELATES"
    ENTITY_VECTOR_INDEX = "sem_entity_name_emb"
    REL_VECTOR_INDEX = "sem_rel_kw_emb"
    SECTION_TITLE = "Semantic"

    async def retrieve(
        self,
        *,
        driver,
        user_id: str,
        ll_embedding: Optional[list[float]],
        hl_embedding: Optional[list[float]],
        top_k: int = DEFAULT_TOP_K,
        item_top_k: int = 15,
    ) -> GraphSearchResult:
        entities, relations = await self.search(
            driver=driver,
            user_id=user_id,
            ll_embedding=ll_embedding,
            hl_embedding=hl_embedding,
            top_k=top_k,
        )

        entity_ids = [e.id for e in entities]
        concepts_via_mentions = await self._fetch_concepts_via_mentions(
            driver, user_id=user_id, entity_ids=entity_ids, limit=item_top_k * 2,
        )

        concept_ids = [it.id for it in concepts_via_mentions]
        concepts_via_one_hop = await self._fetch_concepts_one_hop(
            driver, user_id=user_id, concept_ids=concept_ids, limit=item_top_k,
        )

        seen: set[str] = set()
        merged: list[ItemHit] = []
        for it in concepts_via_mentions + concepts_via_one_hop:
            if it.id in seen:
                continue
            seen.add(it.id)
            merged.append(it)

        for it in merged:
            it.score = final_score(it.cosine, it.timestamp)
        merged.sort(key=lambda x: x.score, reverse=True)
        merged = merged[:item_top_k]

        await self._enrich_with_pg(merged, user_id=user_id)

        return GraphSearchResult(entities=entities, relations=relations, items=merged)

    async def _fetch_concepts_via_mentions(
        self, driver, *, user_id: str, entity_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not entity_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $eids AS eid
                MATCH (e:SemanticEntity {id: eid})<-[:MENTIONS]-(c:Concept {user_id: $user_id})
                WITH DISTINCT c
                ORDER BY c.created_at DESC
                LIMIT $limit
                RETURN c.id AS id, c.name AS name, c.summary AS summary, c.created_at AS created_at
                """,
                eids=entity_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Concept",
                    summary=rec["name"] or "",          # concept "summary" line uses name
                    detail=rec["summary"] or "",        # detail line uses summary
                    timestamp=rec["created_at"], cosine=0.5, source="mentions",
                )
                async for rec in result
            ]

    async def _fetch_concepts_one_hop(
        self, driver, *, user_id: str, concept_ids: list[str], limit: int
    ) -> list[ItemHit]:
        if not concept_ids:
            return []
        from mirix.settings import settings

        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                UNWIND $cids AS cid
                MATCH (c:Concept {id: cid})
                OPTIONAL MATCH (c)-[:CONCEPT_RELATES]-(n:Concept {user_id: $user_id})
                WITH n WHERE n IS NOT NULL
                RETURN DISTINCT n.id AS id, n.name AS name, n.summary AS summary, n.created_at AS created_at
                LIMIT $limit
                """,
                cids=concept_ids, user_id=user_id, limit=limit,
            )
            return [
                ItemHit(
                    id=rec["id"], label="Concept",
                    summary=rec["name"] or "",
                    detail=rec["summary"] or "",
                    timestamp=rec["created_at"], cosine=0.3, source="one_hop",
                )
                async for rec in result
            ]

    async def _enrich_with_pg(self, items: list[ItemHit], *, user_id: str) -> None:
        """Pull full semantic_memory.details. Best-effort; graph summary covers basics."""
        if not items:
            return
        from sqlalchemy import text as sa_text
        from mirix.server.server import db_context

        ids = [it.id for it in items]
        try:
            async with db_context() as session:
                result = await session.execute(
                    sa_text(
                        "SELECT id, details FROM semantic_memory "
                        "WHERE user_id = :u AND id = ANY(:ids)"
                    ),
                    {"u": user_id, "ids": ids},
                )
                detail_map = {row[0]: (row[1] or "") for row in result.fetchall()}
        except Exception as e:
            logger.debug("PG enrich for semantic failed: %s", e)
            return

        for it in items:
            if it.id in detail_map:
                # If PG details is more informative than graph summary, use it
                pg_detail = detail_map[it.id]
                if pg_detail and pg_detail != it.detail:
                    it.detail = pg_detail
```

---

## Modified files

### `docker-compose.yml`

_Adds neo4j:5.20-community (profile-gated: graph) + MIRIX_NEO4J_* env wiring_

```diff
diff --git a/docker-compose.yml b/docker-compose.yml
index 2a87d123..44667d0d 100644
--- a/docker-compose.yml
+++ b/docker-compose.yml
@@ -71,6 +71,40 @@ services:
       retries: 5
       start_period: 5s
 
+  # ==========================================================================
+  # Neo4j (graph memory backend, only used when MIRIX_ENABLE_GRAPH_MEMORY=true)
+  # ==========================================================================
+  neo4j:
+    image: neo4j:5.20-community
+    container_name: mirix_neo4j
+    restart: unless-stopped
+    # Only starts when explicitly requested via:
+    #   docker compose --profile graph up -d
+    # Without the profile, `docker compose up` skips this service entirely,
+    # so users who don't enable graph memory pay zero overhead.
+    profiles: ["graph"]
+    networks:
+      default:
+        aliases:
+          - mirix-neo4j
+    ports:
+      - "7474:7474"
+      - "7687:7687"
+    environment:
+      - NEO4J_AUTH=${MIRIX_NEO4J_USER:-neo4j}/${MIRIX_NEO4J_PASSWORD:-mirix_neo4j_dev}
+      - NEO4J_PLUGINS=["apoc"]
+      - NEO4J_dbms_memory_heap_max__size=2G
+      - NEO4J_dbms_memory_pagecache_size=1G
+    volumes:
+      - ./.persist/neo4j-data:/data
+      - ./.persist/neo4j-logs:/logs
+    healthcheck:
+      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:7474 || exit 1"]
+      interval: 10s
+      timeout: 5s
+      retries: 10
+      start_period: 30s
+
   # ==========================================================================
   # Mirix API Backend
   # ==========================================================================
@@ -99,6 +133,13 @@ services:
         condition: service_healthy
       redis:
         condition: service_healthy
+      # neo4j is profile-gated ("graph"). required: false means mirix_api
+      # starts even when the neo4j service isn't included in the compose run.
+      # When graph memory is enabled, bring it up with:
+      #   docker compose --profile graph up -d
+      neo4j:
+        condition: service_healthy
+        required: false
     networks:
       default:
         aliases:
@@ -131,6 +172,15 @@ services:
       - MIRIX_REDIS_ENABLED=true
       - MIRIX_REDIS_HOST=redis
       - MIRIX_REDIS_PORT=6379
+
+      # =======================================================================
+      # Neo4j Configuration (graph memory)
+      # =======================================================================
+      # Used when MIRIX_ENABLE_GRAPH_MEMORY=true.
+      - MIRIX_ENABLE_GRAPH_MEMORY=${MIRIX_ENABLE_GRAPH_MEMORY:-false}
+      - MIRIX_NEO4J_URI=bolt://neo4j:7687
+      - MIRIX_NEO4J_USER=${MIRIX_NEO4J_USER:-neo4j}
+      - MIRIX_NEO4J_PASSWORD=${MIRIX_NEO4J_PASSWORD:-mirix_neo4j_dev}
       # - MIRIX_REDIS_PASSWORD=           # Set if Redis requires auth
       # - MIRIX_REDIS_DB=0                # Redis database number
       # Alternative: Full Redis URI (overrides individual settings)
```

### `evals/main_eval.py`

_Per-sample token tracker reset/snapshot, writes token_stats into result JSON_

```diff
diff --git a/evals/main_eval.py b/evals/main_eval.py
index 44229f88..9afab12e 100644
--- a/evals/main_eval.py
+++ b/evals/main_eval.py
@@ -141,8 +141,13 @@ def main() -> None:
     parser.add_argument(
         "--output_path",
         type=Path,
-        default=Path("results"),
-        help="Output folder for per-sample JSON results.",
+        default=Path("locomo_run"),
+        help=(
+            "Output sub-folder name. The path is resolved relative to "
+            "<repo>/evals/results/locomo/, so passing 'foo' writes to "
+            "evals/results/locomo/foo. Absolute paths are still honored "
+            "but warned about, since they bypass the locomo namespace."
+        ),
     )
     parser.add_argument(
         "--mirix_config_path",
@@ -159,8 +164,43 @@ def main() -> None:
     mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
     mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")
 
-    output_path = args.output_path
+    # Force every main_eval run into the LoCoMo namespace so MAB and LoCoMo
+    # outputs cannot bleed into each other. The user can still pass an
+    # absolute path to break out (e.g. for one-off experiments), but a warning
+    # makes the divergence explicit.
+    locomo_root = Path(__file__).resolve().parent / "results" / "locomo"
+    if args.output_path.is_absolute():
+        print(
+            f"[main_eval] WARNING: --output_path is absolute ({args.output_path}); "
+            f"writing outside evals/results/locomo/ namespace.",
+        )
+        output_path = args.output_path
+    else:
+        output_path = locomo_root / args.output_path
     output_path.mkdir(parents=True, exist_ok=True)
+    print(f"[main_eval] writing per-sample results to {output_path}")
+
+    # Server-side token tracker is always-on (see mirix/database/token_tracker.py).
+    # We just need to (a) reset before each sample's ingest, (b) snapshot after
+    # ingest to get "build" tokens, (c) snapshot after QA to get "query" tokens.
+    import httpx
+    server_base = "http://127.0.0.1:8531"
+    def _reset_tokens():
+        try:
+            httpx.post(f"{server_base}/debug/token_stats/reset", timeout=10)
+        except Exception:
+            pass
+    def _snapshot_tokens():
+        try:
+            r = httpx.get(f"{server_base}/debug/token_stats", timeout=10)
+            return r.json().get("stats", {})
+        except Exception:
+            return {}
+    def _sum_tokens(stats):
+        s = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
+        for v in stats.values():
+            for k in s: s[k] += v.get(k, 0)
+        return s
 
     for item in items:
         sample_id = item.get("sample_id")
@@ -186,6 +226,9 @@ def main() -> None:
                     mirix_config_path=str(args.mirix_config_path),
                     client=task_agent.mirix_client)
 
+        # Reset server-side token counter so build_tokens reflects only this sample's ingest
+        _reset_tokens()
+
         conversation = item.get("conversation", {})
         for idx, session in enumerate(iter_sessions(conversation), start=1):
             idx_key = str(idx)
@@ -215,6 +258,11 @@ def main() -> None:
             sample_result["timings"]["add_chunk"][idx_key] = elapsed
             save_sample_result(sample_path, sample_result)
 
+        # Snapshot build tokens (everything since reset, before any QA runs)
+        build_stats = _snapshot_tokens()
+        sample_result["token_stats"] = {"build_raw": build_stats, "build_sum": _sum_tokens(build_stats)}
+        save_sample_result(sample_path, sample_result)
+
         qa_list = item.get("qa", [])
         if args.max_questions is not None:
             qa_list = qa_list[: args.max_questions]
@@ -300,6 +348,22 @@ def main() -> None:
         with memories_path.open("w", encoding="utf-8") as handle:
             json.dump(all_memories, handle, ensure_ascii=False, indent=2)
 
+        # Snapshot post-QA total tokens. "query_tokens" is server-side retrieval
+        # cost only (keyword extraction + LightRAG sub-calls). The actual QA
+        # answer LLM call goes through task_agent (client-side OpenAI), tracked
+        # separately in records[*].usage_total.
+        post_qa_stats = _snapshot_tokens()
+        post_qa_sum = _sum_tokens(post_qa_stats)
+        build_sum = sample_result.get("token_stats", {}).get("build_sum", {})
+        query_sum = {
+            k: max(post_qa_sum.get(k, 0) - build_sum.get(k, 0), 0)
+            for k in ("prompt", "completion", "total", "calls")
+        }
+        sample_result.setdefault("token_stats", {})
+        sample_result["token_stats"]["query_raw"] = post_qa_stats
+        sample_result["token_stats"]["query_sum"] = query_sum
+        save_sample_result(sample_path, sample_result)
+
 
 if __name__ == "__main__":
     main()
```

### `evals/mirix_memory_system.py`

_Client timeout 60s → 600s (v4 ingest is slow). Fixes content list shape for retrieve_

```diff
diff --git a/evals/mirix_memory_system.py b/evals/mirix_memory_system.py
index 4de1ebe9..cffdf906 100644
--- a/evals/mirix_memory_system.py
+++ b/evals/mirix_memory_system.py
@@ -48,7 +48,10 @@ class MirixMemorySystem:
 
     def __init__(self, user_id: Optional[str] = None, mirix_config_path: Optional[str] = None, client_id: Optional[str] = None, org_id: Optional[str] = None, client: Optional[MirixClient] = None):
         if client is None:
-            self.client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write")
+            # Long timeout: v4 graph hooks add per-chunk LLM extraction + Neo4j writes
+            # for both episodic and semantic graphs, easily pushing single-chunk processing
+            # past the 60s default. 600s gives headroom; LightRAG retrievals are still fast.
+            self.client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write", timeout=600)
             config_path = Path(mirix_config_path) if mirix_config_path else Path(__file__).with_name("mirix_openai.yaml")
             with config_path.open("r", encoding="utf-8") as handle:
                 config = yaml.safe_load(handle) or {}
@@ -75,10 +78,14 @@ class MirixMemorySystem:
         return response
 
     def wrap_user_prompt(self, prompt: str):
+        # The retrieve endpoint's topic-extraction step iterates msg["content"]
+        # expecting a list of {type, text} dicts (multimodal format). Passing a
+        # bare string here silently degrades to topics="" → LightRAG retrieve
+        # gets an empty query → empty graph context.
         memories = asyncio.run(self.client.retrieve_with_conversation(
             user_id=self.user_id,
             messages=[
-                {'role': 'user', 'content': prompt}
+                {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}
             ]
         ))
```

### `evals/task_agent.py`

_Same client timeout bump_

```diff
diff --git a/evals/task_agent.py b/evals/task_agent.py
index be34a1bd..4f38a677 100644
--- a/evals/task_agent.py
+++ b/evals/task_agent.py
@@ -32,7 +32,7 @@ class TaskAgent:
         self.model = model
         self.user_id = user_id
         self.max_tool_rounds = max_tool_rounds
-        self.mirix_client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write")
+        self.mirix_client = MirixClient(client_id=client_id, org_id=org_id, base_url="http://127.0.0.1:8531", write_scope="read_write", timeout=600)
         self.user_id = user_id if user_id is not None else str(uuid.uuid4())
         config_path = Path(mirix_config_path)
         with config_path.open("r", encoding="utf-8") as handle:
```

### `mirix/llm_api/openai.py`

_Records token usage to tracker after each chat completion (no-op if tracker disabled)_

```diff
diff --git a/mirix/llm_api/openai.py b/mirix/llm_api/openai.py
index 30c148fc..7fffe64c 100755
--- a/mirix/llm_api/openai.py
+++ b/mirix/llm_api/openai.py
@@ -536,6 +536,18 @@ async def openai_chat_completions_request(
 
     response_json = await make_post_request(url, headers, data)
 
+    # Record token usage for instrumented eval runs (no-op outside)
+    try:
+        from mirix.database.token_tracker import record as _record_tokens
+        usage = (response_json.get("usage") or {})
+        _record_tokens(
+            prompt_tokens=usage.get("prompt_tokens", 0),
+            completion_tokens=usage.get("completion_tokens", 0),
+            total_tokens=usage.get("total_tokens"),
+        )
+    except Exception:
+        pass
+
     return ChatCompletionResponse(**response_json)
```

### `mirix/server/rest_api.py`

_Neo4j init in lifespan, /debug/token_stats endpoints, dispatcher hook_

```diff
diff --git a/mirix/server/rest_api.py b/mirix/server/rest_api.py
index b592a6b3..e0ca3d58 100644
--- a/mirix/server/rest_api.py
+++ b/mirix/server/rest_api.py
@@ -108,6 +108,14 @@ async def initialize():
     except Exception as e:
         logger.warning("Redis async init failed: %s", e)
 
+    # Initialize Neo4j driver if graph memory is enabled. No-op otherwise.
+    try:
+        from mirix.database.neo4j_client import init_neo4j_client
+
+        await init_neo4j_client()
+    except Exception as e:
+        logger.warning("Neo4j init failed: %s — graph memory will be unavailable", e)
+
     # Initialize AsyncServer (singleton) and create default org/user/client
     server = get_server()
     await server.ensure_defaults()
@@ -154,6 +162,14 @@ async def cleanup():
     await queue_manager.cleanup()
     logger.info("Queue service stopped")
 
+    # Close Neo4j driver if initialized
+    try:
+        from mirix.database.neo4j_client import close_neo4j_driver
+
+        await close_neo4j_driver()
+    except Exception as e:
+        logger.warning("Error closing Neo4j driver: %s", e)
+
 
 @asynccontextmanager
 async def lifespan(app: FastAPI):
@@ -681,6 +697,34 @@ async def health_check():
     return {"status": "healthy", "service": "mirix-api"}
 
 
+@router.get("/debug/token_stats")
+async def debug_token_stats():
+    """Return cumulative LLM token usage recorded server-side since last reset.
+
+    Tracker is off by default; only counts data after a POST to
+    /debug/token_stats/reset (which enables it).
+    """
+    from mirix.database.token_tracker import is_enabled, snapshot
+    return {"enabled": is_enabled(), "stats": snapshot()}
+
+
+@router.post("/debug/token_stats/reset")
+async def debug_token_stats_reset():
+    """Wipe counters and enable the tracker. Idempotent."""
+    from mirix.database.token_tracker import enable, reset
+    reset()
+    enable()
+    return {"status": "reset", "enabled": True}
+
+
+@router.post("/debug/token_stats/disable")
+async def debug_token_stats_disable():
+    """Turn the tracker off (recording becomes a no-op again)."""
+    from mirix.database.token_tracker import disable
+    disable()
+    return {"status": "disabled"}
+
+
 # ============================================================================
 # Agent Endpoints
 # ============================================================================
@@ -1944,6 +1988,10 @@ async def initialize_meta_agent(
             create_params["agents"] = meta_config["agents"]
         if "system_prompts" in meta_config:
             create_params["system_prompts"] = meta_config["system_prompts"]
+        if "enable_conflict_resolution" in meta_config:
+            create_params["enable_conflict_resolution"] = bool(
+                meta_config["enable_conflict_resolution"]
+            )
 
     # Check if meta agent already exists for this client
     # list_agents now automatically filters by client (organization_id + _created_by_id)
@@ -1985,6 +2033,68 @@ async def initialize_meta_agent(
     return meta_agent
 
 
+async def _augment_source_meta_with_server_fallbacks(
+    filter_tags: dict,
+    user_id: str,
+    n_turns: int,
+    request_occurred_at: Optional[str],
+    server: AsyncServer,
+) -> None:
+    """Mutate ``filter_tags`` in place so it carries a ``source_meta`` dict
+    with at least ``turn_id``, ``chunk_id``, and ``occurred_at`` set.
+
+    Policy:
+
+    1. Anything the client already put in ``filter_tags["source_meta"]``
+       wins. This lets callers with domain knowledge (e.g. the MAB
+       adapter which knows the serial range of a chunk) carry their
+       fields through unchanged.
+    2. Fields the client did NOT set get filled from the server:
+         - ``turn_id``  : next per-user counter (one per input message)
+         - ``chunk_id`` : next per-user counter (one per /memory/add call)
+         - ``occurred_at`` : the request's ``occurred_at`` if provided,
+                             else server wall-clock ISO 8601.
+    3. ``serial`` is never auto-filled. It is a domain-specific signal
+       (e.g. FactConsolidation's numbered fact list) and only present
+       when the caller explicitly set it.
+
+    This is the single point that makes conflict resolution + source
+    provenance general: every ``/memory/add`` (sync or async) ends up
+    with the same ``source_meta`` contract, regardless of which client
+    sent it.
+    """
+    from datetime import timezone as _dt_tz
+
+    existing = filter_tags.get("source_meta")
+    if not isinstance(existing, dict):
+        existing = {}
+    else:
+        existing = dict(existing)  # don't mutate the caller's dict
+
+    needs_turn = "turn_id" not in existing
+    needs_chunk = "chunk_id" not in existing
+    if needs_turn or needs_chunk:
+        reserved = await server.user_manager.reserve_source_ids(
+            user_id=user_id, n_turns=max(n_turns, 1)
+        )
+        if needs_turn:
+            # For a multi-message batch we record the *first* turn_id of
+            # the batch; the agent is free to walk the message list if it
+            # needs per-message granularity. Single-message ingests are
+            # the common case and this is exact.
+            existing["turn_id"] = reserved["turn_id_start"]
+        if needs_chunk:
+            existing["chunk_id"] = reserved["chunk_id"]
+
+    if "occurred_at" not in existing:
+        if request_occurred_at:
+            existing["occurred_at"] = request_occurred_at
+        else:
+            existing["occurred_at"] = datetime.now(_dt_tz.utc).isoformat()
+
+    filter_tags["source_meta"] = existing
+
+
 class AddMemoryRequest(BaseModel):
     """Request model for adding memory."""
 
@@ -2100,6 +2210,20 @@ async def add_memory(
         raise HTTPException(status_code=403, detail="Client has no write_scope - cannot create memories")
     filter_tags["scope"] = client.write_scope
 
+    # Merge client-provided source_meta with server-side fallbacks (turn_id,
+    # chunk_id, occurred_at). This is what makes conflict resolution +
+    # source provenance general: clients with their own source knowledge
+    # (e.g. the MAB adapter knows the chunk's serial range) keep what they
+    # passed; clients that pass nothing still get turn_id / chunk_id /
+    # occurred_at auto-filled from the server.
+    await _augment_source_meta_with_server_fallbacks(
+        filter_tags=filter_tags,
+        user_id=user_id,
+        n_turns=len(input_messages),
+        request_occurred_at=request.occurred_at,
+        server=server,
+    )
+
     # Queue for async processing instead of synchronous execution
     # Note: actor is Client for org-level access control
     #       user_id represents the actual end-user (or admin user if not provided)
@@ -2193,6 +2317,16 @@ async def add_memory_sync(
         raise HTTPException(status_code=403, detail="Client has no write_scope - cannot create memories")
     filter_tags["scope"] = client.write_scope
 
+    # Same server-side source_meta fallback as the async path; see helper
+    # docstring for details.
+    await _augment_source_meta_with_server_fallbacks(
+        filter_tags=filter_tags,
+        user_id=user_id,
+        n_turns=len(input_messages),
+        request_occurred_at=request.occurred_at,
+        server=server,
+    )
+
     from mirix.services.user_manager import UserManager
 
     user_manager = UserManager()
@@ -2301,19 +2435,27 @@ async def retrieve_memories_by_keywords(
         timezone_str = "UTC"
     memories = {}
 
-    # Graph memory retrieval (supplements flat retrieval when enabled)
+    # LightRAG-style dual-level graph retrieval (P3). Supplements flat memory
+    # retrieval with KG entities/relations + episodic chunks. Returns an empty
+    # context string when no hits — caller is robust to that.
     if settings.enable_graph_memory:
         try:
-            graph_context = await server.graph_memory_manager.retrieve_graph_context(
+            from mirix.services.graph_retriever_dispatcher import GraphRetrieverDispatcher
+
+            logger.info(
+                "Graph retrieve: user_id=%s, key_words=%r (len=%d)",
+                user_id, (key_words or "")[:120], len(key_words or ""),
+            )
+            graph_context = await GraphRetrieverDispatcher().retrieve(
                 query=key_words,
-                agent_state=agent_state,
-                organization_id=client.organization_id,
                 user_id=user_id,
+                agent_state=agent_state,
             )
+            logger.info("Graph retrieve result: ctx_len=%d", len(graph_context or ""))
             if graph_context:
                 memories["graph"] = {"context": graph_context}
         except Exception as e:
-            logger.error("Graph memory retrieval failed: %s", e)
+            logger.error("Graph retrieval failed: %s", e, exc_info=True)
 
     # Get episodic memories (recent + relevant) with optional temporal filtering
     try:
```

### `mirix/server/server.py`

_Calls run_startup_migrations before Base.metadata.create_all_

```diff
diff --git a/mirix/server/server.py b/mirix/server/server.py
index 91ddc920..ce13e101 100644
--- a/mirix/server/server.py
+++ b/mirix/server/server.py
@@ -85,7 +85,6 @@ from mirix.services.organization_manager import OrganizationManager
 from mirix.services.per_agent_lock_manager import PerAgentLockManager
 from mirix.services.procedural_memory_manager import ProceduralMemoryManager
 from mirix.services.provider_manager import ProviderManager
-from mirix.services.graph_memory_manager import GraphMemoryManager
 from mirix.services.raw_memory_manager import RawMemoryManager
 from mirix.services.resource_memory_manager import ResourceMemoryManager
 from mirix.services.semantic_memory_manager import SemanticMemoryManager
@@ -454,9 +453,15 @@ else:
 
 
 async def ensure_tables_created():
-    """Create all tables on the async engine. Call from FastAPI lifespan startup."""
+    """Create all tables on the async engine. Call from FastAPI lifespan startup.
+
+    Order matters: startup migrations (e.g. dropping retired tables) must run
+    *before* ``create_all`` so the new ORM state is what gets materialized.
+    """
     if USE_PGLITE:
         return
+    from mirix.database.startup_migrations import run_startup_migrations
+    await run_startup_migrations(engine)
     async with engine.begin() as conn:
         await conn.run_sync(Base.metadata.create_all)
 
@@ -521,7 +526,6 @@ class AsyncServer(Server):
         self.raw_memory_manager = RawMemoryManager()
         self.resource_memory_manager = ResourceMemoryManager()
         self.semantic_memory_manager = SemanticMemoryManager()
-        self.graph_memory_manager = GraphMemoryManager()
 
         # Provider Manager
         self.provider_manager = ProviderManager()
```

### `mirix/services/episodic_memory_manager.py`

_Sync hook after PG insert → EpisodicGraphManager.process_episode_

```diff
diff --git a/mirix/services/episodic_memory_manager.py b/mirix/services/episodic_memory_manager.py
index c5c6f7eb..0bee714a 100755
--- a/mirix/services/episodic_memory_manager.py
+++ b/mirix/services/episodic_memory_manager.py
@@ -565,6 +565,16 @@ class EpisodicMemoryManager:
                 summary_embedding = None
                 embedding_config = None
 
+            # Source provenance: when the /memory/add caller (or the
+            # server-side fallback in rest_api._augment_source_meta_with_
+            # server_fallbacks) attaches a ``source_meta`` dict to
+            # filter_tags, copy it onto the episodic event's
+            # ``source_refs``. We do not strip it from filter_tags so that
+            # downstream filtering / debug still sees it.
+            source_refs_for_event: list = []
+            if filter_tags and isinstance(filter_tags.get("source_meta"), dict):
+                source_refs_for_event = [dict(filter_tags["source_meta"])]
+
             event = await self.create_episodic_memory(
                 PydanticEpisodicEvent(
                     occurred_at=timestamp,
@@ -580,6 +590,7 @@ class EpisodicMemoryManager:
                     details_embedding=details_embedding,
                     embedding_config=embedding_config,
                     filter_tags=filter_tags,
+                    source_refs=source_refs_for_event,
                     last_modify={
                         "timestamp": datetime.now(dt.timezone.utc).isoformat(),
                         "operation": "created",
@@ -591,22 +602,24 @@ class EpisodicMemoryManager:
                 use_cache=use_cache,
             )
 
-            # Graph memory: create episode node + involves edges (async, non-blocking)
+            # Graph memory write path (v4): writes to G_episodic in Neo4j.
+            # Sync hook — failures logged but do not affect the PG insert that
+            # already completed.
             if settings.enable_graph_memory:
                 try:
-                    from mirix.services.graph_memory_manager import GraphMemoryManager
-                    gm = GraphMemoryManager()
-                    await gm.process_for_graph(
-                        text=f"{summary}\n{details}",
+                    from mirix.services.episodic_graph_manager import EpisodicGraphManager
+
+                    await EpisodicGraphManager().process_episode(
+                        episode_id=event.id,
                         summary=summary,
                         details=details,
-                        event_time=timestamp,
+                        occurred_at=timestamp,
                         agent_state=agent_state,
                         organization_id=organization_id,
                         user_id=user_id or "unknown",
                     )
                 except Exception as graph_err:
-                    logger.warning("Graph memory processing failed (non-fatal): %s", graph_err)
+                    logger.warning("Episodic graph write failed (non-fatal): %s", graph_err)
 
             return event
 
@@ -1287,6 +1300,7 @@ class EpisodicMemoryManager:
         actor: PydanticClient = None,
         agent_state: AgentState = None,
         update_mode: str = "append",
+        additional_source_ref: Optional[Dict[str, Any]] = None,
     ):
         """
         Update the selected events
@@ -1299,6 +1313,11 @@ class EpisodicMemoryManager:
             agent_state: Agent state containing embedding configuration (needed for embedding regeneration)
             update_mode: How to handle new_details - "append" (default) appends to existing,
                         "replace" overwrites existing details entirely
+            additional_source_ref: Optional source-provenance dict from the
+                current ingest call (turn_id / chunk_id / serial /
+                occurred_at). When supplied, it is appended to the event's
+                ``source_refs`` list so a merged event still carries the
+                trail of every ingest that contributed to it.
         """
 
         async with self.session_maker() as session:
@@ -1332,6 +1351,15 @@ class EpisodicMemoryManager:
                 )
                 selected_event.embedding_config = agent_state.embedding_config
 
+            # Append the current ingest's source_ref to the event's
+            # provenance trail (if provided). Late-arriving ingests that
+            # merge into an existing event keep their pointer in the list
+            # rather than being lost.
+            if additional_source_ref:
+                existing_refs = list(selected_event.source_refs or [])
+                existing_refs.append(dict(additional_source_ref))
+                selected_event.source_refs = existing_refs
+
             # Update last_modify field with timestamp and operation info
             selected_event.last_modify = {
                 "timestamp": datetime.now(dt.timezone.utc).isoformat(),
```

### `mirix/services/semantic_memory_manager.py`

_Sync hook after PG insert → SemanticGraphManager.process_concept_

```diff
diff --git a/mirix/services/semantic_memory_manager.py b/mirix/services/semantic_memory_manager.py
index 89d9bc2a..47edec47 100755
--- a/mirix/services/semantic_memory_manager.py
+++ b/mirix/services/semantic_memory_manager.py
@@ -959,7 +959,40 @@ class SemanticMemoryManager:
     ) -> PydanticSemanticMemoryItem:
         """
         Create a new semantic memory entry using provided parameters.
+
+        Auto-route: when ``filter_tags`` contains a ``source_meta`` dict
+        (chunk_id / serial / occurred_at) AND ``name`` is shaped like
+        ``"<entity> / <relation>"``, the call is forwarded to
+        ``upsert_with_conflict_resolution`` for deterministic merge with
+        ``prior_values`` history. Otherwise the legacy free-form path is
+        used unchanged.
         """
+        # ---- conflict-resolution auto-route ------------------------------
+        source_meta = (filter_tags or {}).get("source_meta")
+        if source_meta and isinstance(name, str) and " / " in name:
+            entity, _, relation = name.partition(" / ")
+            entity, relation = entity.strip(), relation.strip()
+            if entity and relation:
+                # The ``source_meta`` dict is the per-ingest payload sent by
+                # the client. Carry every field through as the source_ref
+                # so the ordering tuple (occurred_at > serial > created_at)
+                # in the manager can use whichever fields are present.
+                return await self.upsert_with_conflict_resolution(
+                    actor=actor,
+                    agent_state=agent_state,
+                    agent_id=agent_id,
+                    entity=entity,
+                    relation=relation,
+                    value=summary,
+                    source_ref=dict(source_meta),
+                    organization_id=organization_id,
+                    extra_filter_tags={
+                        k: v for k, v in (filter_tags or {}).items() if k != "source_meta"
+                    },
+                    use_cache=use_cache,
+                    client_id=client_id,
+                    user_id=user_id,
+                )
         try:
             # Set defaults for required fields
             from mirix.services.user_manager import UserManager
@@ -1008,27 +1041,246 @@ class SemanticMemoryManager:
 
             # Note: Item is already added to clustering tree in create_item()
 
-            # Graph memory: extract entities/relations from semantic item (async, non-blocking)
+            # Graph memory write path (v4): writes to G_semantic in Neo4j.
+            # Each semantic item becomes a (:Concept) node; LightRAG-style
+            # extraction adds (:SemanticEntity) + [:SEM_RELATES], and an LLM
+            # judgement step builds (:Concept)-[:CONCEPT_RELATES]->(:Concept)
+            # edges to existing top-K similar concepts. Sync hook — failures
+            # logged but do not affect the PG insert that already completed.
             if settings.enable_graph_memory:
                 try:
-                    from mirix.services.graph_memory_manager import GraphMemoryManager
-                    gm = GraphMemoryManager()
-                    await gm.process_for_graph(
-                        text=f"{name}: {summary}\n{details or ''}",
+                    from mirix.services.semantic_graph_manager import SemanticGraphManager
+
+                    await SemanticGraphManager().process_concept(
+                        concept_id=semantic_item.id,
+                        name=name,
                         summary=summary,
-                        details=details,
-                        event_time=datetime.now(timezone.utc),
+                        details=details or "",
                         agent_state=agent_state,
                         organization_id=organization_id,
                         user_id=user_id or "unknown",
                     )
                 except Exception as graph_err:
-                    logger.warning("Graph memory processing failed (non-fatal): %s", graph_err)
+                    logger.warning("Semantic graph write failed (non-fatal): %s", graph_err)
 
             return semantic_item
         except Exception as e:
             raise e
 
+    @staticmethod
+    def _build_cr_filter_tags(
+        entity: str,
+        relation: str,
+        existing: Optional[Dict[str, Any]] = None,
+    ) -> Dict[str, Any]:
+        """Merge the conflict-resolution lookup keys into a filter_tags dict.
+
+        ``cr_entity`` and ``cr_relation`` are the index used by
+        ``upsert_with_conflict_resolution`` to find the canonical item for a
+        given (entity, relation) pair within a user_id. Other tags
+        (scope, project_id, ...) are preserved.
+        """
+        out: Dict[str, Any] = dict(existing or {})
+        out["cr_entity"] = entity
+        out["cr_relation"] = relation
+        return out
+
+    @staticmethod
+    def _source_ref_key(source_ref: Optional[Dict[str, Any]]) -> tuple:
+        """Total ordering for source refs.
+
+        Priority: occurred_at > serial > created_at (caller fills created_at
+        when nothing else is available). All missing → very small key, so
+        the caller's new ref wins ties via the explicit ``-1`` fallback.
+        """
+        if not source_ref:
+            return (0, "", -1, "")
+        # occurred_at: ISO 8601 strings compare lexicographically when in UTC.
+        occurred = source_ref.get("occurred_at") or ""
+        serial = source_ref.get("serial")
+        created = source_ref.get("created_at") or ""
+        # Each tier becomes its own sort key; "" sorts before any real value.
+        return (
+            1 if occurred else 0, occurred,
+            1 if serial is not None else 0, serial if serial is not None else -1,
+            1 if created else 0, created,
+        )
+
+    async def _find_by_entity_relation(
+        self,
+        entity: str,
+        relation: str,
+        user_id: str,
+        actor: PydanticClient,
+    ) -> Optional[SemanticMemoryItem]:
+        """Lookup the existing canonical item for (entity, relation) under
+        this user, or None. Uses the ``cr_entity`` / ``cr_relation`` keys
+        the upsert path writes into ``filter_tags``.
+        """
+        async with self.session_maker() as session:
+            # Postgres: filter_tags is JSONB; use ->> operator. SQLite path
+            # falls back to a Python-side filter for the small subset that
+            # already matches user_id.
+            if settings.mirix_pg_uri_no_default:
+                stmt = (
+                    select(SemanticMemoryItem)
+                    .where(SemanticMemoryItem.user_id == user_id)
+                    .where(text("(filter_tags->>'cr_entity') = :ent"))
+                    .where(text("(filter_tags->>'cr_relation') = :rel"))
+                    .params(ent=entity, rel=relation)
+                    .limit(1)
+                )
+                result = await session.execute(stmt)
+                row = result.scalar_one_or_none()
+                return row
+            # SQLite fallback
+            stmt = select(SemanticMemoryItem).where(
+                SemanticMemoryItem.user_id == user_id
+            )
+            result = await session.execute(stmt)
+            for row in result.scalars().all():
+                ft = row.filter_tags or {}
+                if ft.get("cr_entity") == entity and ft.get("cr_relation") == relation:
+                    return row
+            return None
+
+    async def upsert_with_conflict_resolution(
+        self,
+        actor: PydanticClient,
+        agent_state: AgentState,
+        agent_id: str,
+        entity: str,
+        relation: str,
+        value: str,
+        source_ref: Dict[str, Any],
+        organization_id: str,
+        status: str = "asserted",
+        extra_filter_tags: Optional[Dict[str, Any]] = None,
+        use_cache: bool = True,
+        client_id: Optional[str] = None,
+        user_id: Optional[str] = None,
+    ) -> PydanticSemanticMemoryItem:
+        """Deterministic upsert of a (entity, relation, value) fact.
+
+        Lookup the existing canonical item for this (entity, relation) and:
+
+        - If no existing item, insert a new one with ``name = "<entity> /
+          <relation>"``, ``summary = value``, ``source_refs = [source_ref]``,
+          and ``filter_tags`` carrying ``cr_entity``/``cr_relation``.
+        - If the new source_ref has a strictly larger sort key than the
+          existing canonical's most recent ref, replace the canonical:
+          old summary/source_refs move into ``prior_values`` with status
+          ``"superseded"``; new value becomes the current ``summary``.
+        - Otherwise append the new ref to ``prior_values`` as a late-arriving
+          older version (so the audit trail is preserved without changing
+          the current canonical).
+        - ``status="corrected"`` forces a replace and marks the displaced
+          version with ``status="corrected"`` regardless of source_ref order.
+
+        Returns the canonical item after the upsert.
+
+        No LLM is involved in this method — the merge is deterministic on
+        the contents of ``source_ref``.
+        """
+        from mirix.services.user_manager import UserManager
+
+        if client_id is None:
+            client_id = actor.id
+        if user_id is None:
+            user_id = UserManager.ADMIN_USER_ID
+
+        existing = await self._find_by_entity_relation(entity, relation, user_id, actor)
+        merged_tags = self._build_cr_filter_tags(entity, relation, extra_filter_tags)
+
+        if existing is None:
+            # Cold path: behave like a regular insert, but seed source_refs
+            # and stash the cr_entity/cr_relation in filter_tags.
+            name = f"{entity} / {relation}"
+            details = f"Current value: {value}"
+            item = await self.insert_semantic_item(
+                actor=actor,
+                agent_state=agent_state,
+                agent_id=agent_id,
+                name=name,
+                summary=value,
+                details=details,
+                source=str(source_ref) if source_ref else "",
+                organization_id=organization_id,
+                filter_tags=merged_tags,
+                use_cache=use_cache,
+                client_id=client_id,
+                user_id=user_id,
+            )
+            # Patch source_refs onto the row in-place; insert_semantic_item
+            # doesn't take it as a parameter to keep the legacy surface stable.
+            async with self.session_maker() as session:
+                db_row = await SemanticMemoryItem.read(
+                    db_session=session, identifier=item.id, actor=actor
+                )
+                db_row.source_refs = [source_ref] if source_ref else []
+                await session.commit()
+                await session.refresh(db_row)
+                return db_row.to_pydantic()
+
+        # Hot path: an existing canonical exists. Compare source_refs and
+        # decide whether the new ref supersedes it.
+        existing_refs: List[Dict[str, Any]] = list(existing.source_refs or [])
+        # The "most recent" existing ref is the max under our ordering.
+        existing_top = max(existing_refs, key=self._source_ref_key) if existing_refs else None
+        new_wins = (
+            status == "corrected"
+            or existing_top is None
+            or self._source_ref_key(source_ref) > self._source_ref_key(existing_top)
+        )
+
+        async with self.session_maker() as session:
+            db_row = await SemanticMemoryItem.read(
+                db_session=session, identifier=existing.id, actor=actor
+            )
+            now_iso = datetime.now(timezone.utc).isoformat()
+
+            if new_wins:
+                # Move the current value into prior_values, swap in the new.
+                prior_entry = {
+                    "value": db_row.summary,
+                    "source_refs": list(db_row.source_refs or []),
+                    "status": "corrected" if status == "corrected" else "superseded",
+                    "moved_at": now_iso,
+                }
+                db_row.prior_values = list(db_row.prior_values or []) + [prior_entry]
+                db_row.summary = value
+                db_row.details = f"Current value: {value}"
+                db_row.source_refs = [source_ref] if source_ref else []
+                # Keep the cr_entity/cr_relation tags intact while merging
+                # any extra tags from this update.
+                merged_existing = self._build_cr_filter_tags(
+                    entity, relation, {**(db_row.filter_tags or {}), **(extra_filter_tags or {})}
+                )
+                db_row.filter_tags = merged_existing
+                db_row.last_modify = {
+                    "timestamp": now_iso,
+                    "operation": "cr_supersede" if status != "corrected" else "cr_correct",
+                }
+            else:
+                # Late-arriving older fact — record in prior_values, don't
+                # touch the canonical.
+                prior_entry = {
+                    "value": value,
+                    "source_refs": [source_ref] if source_ref else [],
+                    "status": "superseded",
+                    "moved_at": now_iso,
+                    "note": "late-arrived older fact",
+                }
+                db_row.prior_values = list(db_row.prior_values or []) + [prior_entry]
+                db_row.last_modify = {
+                    "timestamp": now_iso,
+                    "operation": "cr_record_late",
+                }
+
+            await session.commit()
+            await session.refresh(db_row)
+            return db_row.to_pydantic()
+
     async def delete_semantic_item_by_id(self, semantic_memory_id: str, actor: PydanticClient) -> None:
         """Delete a semantic memory item by ID (removes from cache)."""
         async with self.session_maker() as session:
```

### `mirix/orm/__init__.py`

_Drops v2 ORM imports (EntityNode, EntityEdge, EpisodeNode, InvolvesEdge)_

```diff
diff --git a/mirix/orm/__init__.py b/mirix/orm/__init__.py
index aaa885e8..62007389 100755
--- a/mirix/orm/__init__.py
+++ b/mirix/orm/__init__.py
@@ -6,7 +6,6 @@ from mirix.orm.client_api_key import ClientApiKey
 from mirix.orm.cloud_file_mapping import CloudFileMapping
 from mirix.orm.episodic_memory import EpisodicEvent
 from mirix.orm.file import FileMetadata
-from mirix.orm.graph_memory import EntityEdge, EntityNode, EpisodeNode, InvolvesEdge
 from mirix.orm.knowledge_vault import KnowledgeVaultItem
 from mirix.orm.message import Message
 from mirix.orm.organization import Organization
@@ -26,12 +25,8 @@ __all__ = [
     "Client",
     "ClientApiKey",
     "CloudFileMapping",
-    "EntityEdge",
-    "EntityNode",
-    "EpisodeNode",
     "EpisodicEvent",
     "FileMetadata",
-    "InvolvesEdge",
     "KnowledgeVaultItem",
     "Message",
     "Organization",
```

### `mirix/settings.py`

_Adds neo4j_uri, neo4j_user, neo4j_password, neo4j_database, neo4j_vector_dim_

```diff
diff --git a/mirix/settings.py b/mirix/settings.py
index 5d513c19..dbd18fc3 100755
--- a/mirix/settings.py
+++ b/mirix/settings.py
@@ -226,8 +226,18 @@ class Settings(BaseSettings):
     llm_retry_backoff_factor: float = Field(0.5, env="MIRIX_LLM_RETRY_BACKOFF_FACTOR")  # Exponential backoff multiplier
     llm_retry_max_delay: float = Field(10.0, env="MIRIX_LLM_RETRY_MAX_DELAY")  # Max delay between retries (seconds)
 
-    # Graph memory: temporal knowledge graph for episodic + semantic
+    # Graph memory: LightRAG-style dual-level retrieval over Neo4j.
+    # When enabled, episodic event inserts also extract entities/relations into
+    # Neo4j; retrieval supplements flat memory with graph context.
     enable_graph_memory: bool = Field(False, env="MIRIX_ENABLE_GRAPH_MEMORY")
+    neo4j_uri: str = Field("bolt://localhost:7687", env="MIRIX_NEO4J_URI")
+    neo4j_user: str = Field("neo4j", env="MIRIX_NEO4J_USER")
+    neo4j_password: str = Field("mirix_neo4j_dev", env="MIRIX_NEO4J_PASSWORD")
+    neo4j_database: str = Field("neo4j", env="MIRIX_NEO4J_DATABASE")
+    # Dimension of embeddings used for entity/relation vector indexes in Neo4j.
+    # Must match the embedding model in use (1536 for text-embedding-3-small,
+    # 3072 for text-embedding-3-large).
+    neo4j_vector_dim: int = Field(1536, env="MIRIX_NEO4J_VECTOR_DIM")
 
     # cron job parameters
     enable_batch_job_polling: bool = False
```

### `requirements.txt`

_Adds neo4j>=5.20.0,<6.0.0_

```diff
diff --git a/requirements.txt b/requirements.txt
index 90b1f708..81a875a4 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -48,6 +48,7 @@ httpx
 ipdb
 protobuf>=5.0.0,<6.0.0
 redis[hiredis]>=7.0.1,<8.0.0
+neo4j>=5.20.0,<6.0.0
 aiokafka>=0.13.0,<0.14.0
 asyncddgs
 google-auth
```

---

## Deleted files

These v2 single-graph artifacts are removed in favor of the v4 dual-graph design. Full deletion diffs are in `v4_graph_memory.patch`.

- `mirix/orm/graph_memory.py`
- `mirix/services/graph_memory_manager.py`
