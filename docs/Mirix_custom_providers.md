# MIRIX Custom Providers: Cache, Relational DB, and Search

> Scope: This document is a code-derived companion to
> `docs/Mirix_API_flow_document.md`. It describes the three pluggable
> backend providers that let MIRIX (and ECMS, Enterprise Context and
> Memory Service, as a consumer) swap out Postgres + Redis for an
> alternate stack at runtime, what each provider is responsible for,
> every code touch-point, the data / logic deltas vs. the default
> Postgres + Redis backend, and how to wire them in.

---

## 0. The Big Picture

MIRIX has three independent provider registries. Each is a tiny
duck-typed plug-in point in `mirix/database/`:

| Layer | Module | Default backend | Purpose |
|-------|--------|-----------------|---------|
| Cache | `mirix/database/cache_provider.py` | Redis (`RedisCacheProvider` over `RedisMemoryClient`) | Read-through / write-through KV cache for hot rows; strict best-effort, never authoritative. |
| Relational DB | `mirix/database/relational_provider.py` | SQLAlchemy + Postgres (no provider, ORM path) | CRUD + named-query authoritative store; replaces the SQLAlchemy ORM session path when registered. |
| Search | `mirix/database/search_provider.py` | Postgres pgvector + `pg_bm25` via SQLAlchemy (no provider) | Vector / BM25 / string-match retrieval and per-table counts; replaces the Redis-as-search and SQLAlchemy `list_*` paths. |

They share the same registration model:

```python
_providers: Dict[str, Any] = {}
_active_provider_name: Optional[str] = None

def register_X_provider(name, provider): ...     # last writer wins
def get_X_provider() -> Optional[Any]: ...        # None => default backend
def unregister_X_provider(name): ...
```

Because each registry is *purely* additive, MIRIX has four legal runtime
shapes:

1. **No providers** — original behavior. Postgres for CRUD + search,
   no caching.
2. **Cache only** — Postgres for CRUD + search, cache provider
   (typically Redis) speeds up hot reads.
3. **Cache + Relational + Search** — full ECMS / Provider shape: external
   relational store + external search index, with optional cache.
4. **Relational + Search (no cache)** — supported; cache is
   independent of the other two.

`mirix/database/provider_validation.py::validate_provider_pairing_or_raise()`
exists to enforce that **Relational and Search must be registered
together or not at all** (cache is independent). Note: it is currently
called only from tests; ECMS-style consumers that register both at
startup never hit the failure case in production. The function should be
called explicitly during the startup of any consumer that registers
either provider.

The agent / manager code never branches on the *kind* of provider,
only on whether one is registered:

```python
provider = get_relational_provider()
if provider:
    return await provider.read("agents", agent_id)
# else: SQLAlchemy ORM fallback
async with self.session_maker() as session:
    ...
```

This is the central pattern. The rest of this document walks each
provider through:

- a) the touch-points where it is consulted in the API flow,
- b) where the custom path diverges from the Postgres-only path,
- c) what to set / call to run with a custom implementation.

---

## 1. How the Cache Works

This section is about caching as a **system behavior** in MIRIX —
where it gets read, where it gets written, what gets cached, and how
the absence of a cache is handled. The next section (#2) covers the
provider abstraction itself.

### 1.1 Cache topology

The cache is keyed by entity type with a per-type prefix and a
type-appropriate Redis representation:

| Entity / Table | Prefix attribute | Redis representation | TTL setting |
|---|---|---|---|
| `organizations` | `ORGANIZATION_PREFIX` | Hash | `redis_ttl_organizations` (12h default) |
| `users` | `USER_PREFIX` | Hash | `redis_ttl_users` (12h) |
| `clients` | `CLIENT_PREFIX` | Hash | `redis_ttl_clients` (12h) |
| `agents` | `AGENT_PREFIX` | Hash (with denormalized `tool_ids`, `memory_block_ids`, `children_ids`) | `redis_ttl_agents` (12h) |
| `tools` | `TOOL_PREFIX` | Hash (`json_schema` and `tags` flattened to JSON strings) | `redis_ttl_tools` (12h) |
| `messages` | `MESSAGE_PREFIX` | Hash | `redis_ttl_messages` (2h) |
| `block` (core memory) | `BLOCK_PREFIX` | Hash | `redis_ttl_blocks` (2h) |
| `episodic_memory` | `EPISODIC_PREFIX` | JSON + RediSearch vector indexes (`details_embedding`, `summary_embedding`) | `redis_ttl_default` (1h) |
| `semantic_memory` | `SEMANTIC_PREFIX` | JSON + 3 vector indexes | `redis_ttl_default` (1h) |
| `procedural_memory` | `PROCEDURAL_PREFIX` | JSON + 2 vector indexes | `redis_ttl_default` (1h) |
| `resource_memory` | `RESOURCE_PREFIX` | JSON + 1 vector index | `redis_ttl_default` (1h) |
| `knowledge_vault` | `KNOWLEDGE_PREFIX` | JSON + 1 vector index | `redis_ttl_default` (1h) |
| `raw_memory` | `RAW_MEMORY_PREFIX` | JSON | `redis_ttl_default` (1h) |
| `memory_sources` | `MEMORY_SOURCE_PREFIX` | JSON | `redis_ttl_default` (1h) |
| `memory_citations` | `MEMORY_CITATION_PREFIX` | JSON (used for citation-exists short-circuit) | `redis_ttl_default` (1h) |

The constants are defined twice — on `RedisMemoryClient`
(`mirix/database/redis_client.py`) for the underlying client and on
`RedisCacheProvider` (`mirix/database/redis_cache_provider.py`) for
the provider interface. Custom providers must expose the same set of
prefix constants on their class because manager code reads them
directly:

```37:42:mirix/database/cache_provider.py
Key prefix constants (all providers should define these):
    BLOCK_PREFIX, MESSAGE_PREFIX, EPISODIC_PREFIX, SEMANTIC_PREFIX,
    PROCEDURAL_PREFIX, RESOURCE_PREFIX, KNOWLEDGE_PREFIX, RAW_MEMORY_PREFIX,
    ORGANIZATION_PREFIX, USER_PREFIX, CLIENT_PREFIX, AGENT_PREFIX, TOOL_PREFIX
```

### 1.2 Cache invariants

1. **Cache is best-effort.** Every read/write site is wrapped in
   `try/except`, logs a warning, and falls through to the
   authoritative store on failure. A Redis outage never fails an HTTP
   request.
2. **Read-through.** Every `get_*_by_id` first checks the cache, then
   the DB / Relational provider, then populates the cache on a miss.
3. **Write-through invalidation.** Every write path (create / update
   / delete / soft-delete) deletes or refreshes the cache key for the
   touched entity *after* the authoritative write. Soft deletes
   delete the key; sets get `set_hash` / `set_json` written.
4. **Memory tables also write `*_ts` numeric companions** for
   `created_at` / `occurred_at` so RediSearch can do numeric range
   queries. `RedisMemoryClient.clean_redis_fields()` strips these
   before Pydantic validation on the read path.
5. **`use_cache=False` is honored.** Read-side helpers (e.g.
   `MessageManager.get_message_by_id`, episodic search) accept a
   `use_cache` flag; when `False` they skip the cache lookup entirely
   and go straight to the authoritative store.
6. **Cascade deletes invalidate caches in batches.** Bulk delete
   methods (`delete_by_client_id`, `delete_memories_by_user_id`)
   collect the row IDs first and then delete the corresponding cache
   keys in a loop — see e.g.
   `MessageManager.delete_by_client_id`.

### 1.3 Touch-points in the API flow

The map below complements §3 of `Mirix_API_flow_document.md`. For
each route group, the column lists every place where the cache is
consulted on the path between the HTTP handler and the authoritative
store.

| API surface | Read-through (cache.get) | Invalidate / refresh (cache.delete or set) |
|---|---|---|
| `GET /agents/{id}` | `AgentManager.get_agent_by_id` (Hash) | `AgentManager.update_*`, `delete_agent`, `_invalidate_parent_cache_for_child` |
| `GET /agents` | (per-row read-through inside list expansion) | `update_agent` invalidates the row + parent caches |
| `POST /agents`, `POST /agents/meta/initialize` | — | `SqlalchemyBase.create_with_redis` writes Hash + denormalized children/tools |
| `PATCH /agents/{id}/system` | — | `agent_manager.update_system_prompt` invalidates AGENT_PREFIX |
| `GET /tools/{id}` | `ToolManager.get_tool_by_id` (Hash; rehydrates `json_schema`/`tags` from JSON strings) | `ToolManager.delete_tool_by_id` deletes; `create_tool` writes via ORM hook |
| `GET /blocks`, `GET /blocks/{id}` | `BlockManager.get_block_by_id` (Hash) | `update_block`, `delete_block`, `create_or_update_block` invalidate BLOCK_PREFIX |
| `GET /memory/episodic/{id}`, `…/semantic/{id}`, `…/procedural/{id}`, `…/resource/{id}`, `…/knowledge_vault/{id}` | per-manager `get_*_by_id` (JSON) | corresponding `update_*` / `delete_*` invalidate EPISODIC/SEMANTIC/… prefixes |
| `POST /memory/raw`, `GET /memory/raw/{id}`, `PATCH …`, `DELETE …` | `RawMemoryManager.get_raw_memory_by_id` (JSON) | every CRUD method invalidates RAW_MEMORY_PREFIX |
| `GET /memory-sources/{id}` | `MemorySourceManager.get_by_id` (JSON; strips unknown fields before Pydantic) | populated on miss |
| Memory citation existence (internal, used during write) | `MemoryCitationManager._exists_cache_key` (JSON `{"exists": True}`) | written when a citation row is inserted |
| `GET /organizations/{id}` | `OrganizationManager.get_organization_by_id` (Hash) | `delete_organization` invalidates ORGANIZATION_PREFIX |
| `GET /users/{id}` | `UserManager.get_user_by_id` (Hash) | user update / soft-delete invalidate USER_PREFIX |
| `GET /clients/{id}` | `ClientManager.get_client_by_id` (Hash) | `update_client`, `delete_client_by_id` invalidate CLIENT_PREFIX |
| `GET /agents/{id}/messages` (and `/memory/add` retention) | `MessageManager.get_message_by_id` (Hash) | `delete_message`, `delete_by_client_id` invalidate MESSAGE_PREFIX |

Two important things this table does *not* show:

- The **shared invalidation helper** in
  `mirix/services/memory_manager_helpers.py::invalidate_memory_cache(table, ids)`
  centralises bulk invalidation for the seven memory tables (block,
  raw_memory, episodic_memory, semantic_memory, procedural_memory,
  resource_memory, knowledge_vault) and silently no-ops when no
  cache provider is registered.
- The **ORM-level cache hook** in
  `mirix/orm/sqlalchemy_base.py::SqlalchemyBase._update_redis_cache(operation)`
  is the single place that knows how to serialise each table for the
  cache (Hash vs JSON, denormalisation rules for `agents`/`tools`).
  It runs on every `create_with_redis` / `update_with_redis` /
  `soft_delete_with_redis` call from the SQLAlchemy fallback path.
  Custom relational providers must provide the equivalent serialised
  payloads if they want a populated cache.

### 1.4 Custom-provider deltas vs default Postgres backend

When the cache is absent, MIRIX simply talks to Postgres on every
read. The functional behaviour is identical; only latency changes.
When a custom (non-Redis) cache provider is wired in, the only
visible deltas are:

- The provider must support `get_hash` / `set_hash` *and* `get_json`
  / `set_json` semantics — the manager code chooses one or the
  other depending on the entity type. A Redis-shaped backend
  satisfies this naturally; an opaque KV store should serialise
  both shapes to JSON internally and just expose the dual API.
- A custom cache cannot do RediSearch. The Redis cache provider
  sneaks an extra index-based search path into hot read flows
  (e.g. `RedisMemoryClient.search_text` / `search_vector`) that
  custom providers do not. For non-Redis caches, all retrieval must
  go through the **search provider**, not the cache provider —
  see §4. (The cache provider interface deliberately exposes only
  KV operations.)
- TTLs are passed in seconds; the provider is free to ignore them.
  The default Redis provider honours them; an in-memory LRU cache
  can ignore them and rely on size-based eviction.
- Cache absence is not the same as "no cache miss penalty". If you
  register a degenerate provider that always returns `None`, every
  read still falls through to the authoritative store. The cleaner
  way to disable caching is to leave the registry empty
  (`MIRIX_REDIS_ENABLED=false`).

### 1.5 Sample configuration & command

Two-line setup with the default Redis cache:

```bash
export MIRIX_REDIS_ENABLED=true
export MIRIX_REDIS_HOST=localhost      # or set MIRIX_REDIS_URI=redis://...
export MIRIX_REDIS_PORT=6379
export MIRIX_REDIS_TTL_DEFAULT=3600
python scripts/start_server.py --port 8531
```

`mirix/database/redis_client.py::initialize_redis_client()` reads
those settings, instantiates `RedisMemoryClient`, then
**auto-registers** a `RedisCacheProvider` against the cache registry
so service managers can pick it up via `get_cache_provider()`.

To run with no cache (single-process dev):

```bash
export MIRIX_REDIS_ENABLED=false
python scripts/start_server.py --port 8531
```

To run with a custom cache (sketch, no startup hook is required —
just call `register_cache_provider` early enough that it precedes
the first `get_cache_provider()` call):

```python
from mirix.database.cache_provider import register_cache_provider

class MyMemoryCache:
    BLOCK_PREFIX = "block:"
    MESSAGE_PREFIX = "msg:"
    EPISODIC_PREFIX = "episodic:"
    SEMANTIC_PREFIX = "semantic:"
    PROCEDURAL_PREFIX = "procedural:"
    RESOURCE_PREFIX = "resource:"
    KNOWLEDGE_PREFIX = "knowledge:"
    RAW_MEMORY_PREFIX = "raw_memory:"
    ORGANIZATION_PREFIX = "org:"
    USER_PREFIX = "user:"
    CLIENT_PREFIX = "client:"
    AGENT_PREFIX = "agent:"
    TOOL_PREFIX = "tool:"
    MEMORY_SOURCE_PREFIX = "memory_source:"
    MEMORY_CITATION_PREFIX = "memory_citation:"

    async def get(self, key): ...
    async def set(self, key, data, ttl=None): ...
    async def delete(self, key): ...
    async def get_hash(self, key): ...
    async def set_hash(self, key, data, ttl=None): ...
    async def get_json(self, key): ...
    async def set_json(self, key, data, ttl=None): ...

register_cache_provider("my_cache", MyMemoryCache())
```

In ECMS, this is what `common.ips_provider_setup.register_ips_providers`
does for the Intuit cache backend at startup.

---

## 2. How the Cache Provider Works

Section 1 described the *system* behaviour. This section describes
the *abstraction* — the contract a cache backend has to satisfy and
how it is registered, looked up, and consumed.

### 2.1 Interface (duck-typed, all async)

```9:16:mirix/database/cache_provider.py
Expected methods (duck typing, all async):
    - async get(key: str) -> Optional[Dict[str, Any]]
    - async set(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - async delete(key: str) -> bool
    - async get_hash(key: str) -> Optional[Dict[str, Any]]
    - async set_hash(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
    - async get_json(key: str) -> Optional[Dict[str, Any]]
    - async set_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool
```

There is no abstract base class. The provider can be any object that
exposes these coroutine methods plus the prefix constants from §1.1.
This mirrors the pattern used for `mirix.llm_api.auth_provider`.

The split between `get` / `get_hash` / `get_json` is deliberate:

- `get` / `set` are the generic "string or JSON" path used by
  `MemoryCitationManager` for tiny existence checks.
- `get_hash` / `set_hash` are used for flat, embedding-free entities
  (organizations, users, clients, agents, tools, blocks, messages).
- `get_json` / `set_json` are used for entities that have nested or
  vector-bearing fields (the six memory tables, `memory_sources`,
  `memory_citations`).

The cache provider does **not** expose the search APIs that
`RedisMemoryClient` defines (`search_text`, `search_vector`,
`search_recent`, …). Those are intentionally kept inside the Redis
client and reachable only through the **search provider** registry.

### 2.2 Registry semantics

```46:91:mirix/database/cache_provider.py
def register_cache_provider(name: str, provider: Any) -> None:
    ...
    _cache_providers[name] = provider
    _active_provider_name = name
    logger.info("Registered cache provider: %s", name)


def get_cache_provider() -> Optional[Any]:
    ...
    if _active_provider_name and _active_provider_name in _cache_providers:
        return _cache_providers[_active_provider_name]
    return None
```

- Last writer wins: every `register_cache_provider` updates
  `_active_provider_name` to the just-registered name. There is no
  composite / fallback chain.
- `get_cache_provider()` returns `None` when nothing is registered;
  every consumer is required to handle that case.
- `unregister_cache_provider(name)` clears the active name when
  unregistering the active provider — primarily a test fixture.

### 2.3 Touch-points in the API flow

The diagram below traces the cache provider through the request
lifecycle for a single `GET /memory/episodic/{id}` (typical
read-through example) and a single `POST /memory/raw` (typical
write-through example):

```
GET /memory/episodic/{id}
  → EpisodicMemoryManager.get_episodic_event_by_id(id)
      ├── get_cache_provider() ── if cache_provider:
      │     cache_key = f"{cache_provider.EPISODIC_PREFIX}{id}"
      │     cached = await cache_provider.get_json(cache_key)
      │     if cached: return PydanticEpisodicEvent(**clean_redis_fields([cached])[0])
      ├── (cache miss) get_relational_provider() ── if rp:
      │     rp.read("episodic_memory", id, actor=actor)
      │     [populate cache ad hoc here in some managers]
      └── (else) async with session_maker() as session:
            EpisodicEventModel.read(...)            # ORM path
            (auto-populates cache via _update_redis_cache)
```

```
POST /memory/raw
  → RawMemoryManager.create_raw_memory(...)
      ├── if get_relational_provider(): rp.create("raw_memory", payload)
      └── else: async with session_maker(): event.create_with_redis(session)
                  └── SqlalchemyBase._update_redis_cache("update")
                        └── cache_provider.set_json(RAW_MEMORY_PREFIX+id, data, ttl)
```

These two patterns repeat across every `*_manager.py`. The exhaustive
list of files that import `get_cache_provider`:

- `mirix/orm/sqlalchemy_base.py` (the canonical write-through hook)
- `mirix/services/agent_manager.py`
- `mirix/services/block_manager.py`
- `mirix/services/client_manager.py`
- `mirix/services/episodic_memory_manager.py`
- `mirix/services/knowledge_vault_manager.py`
- `mirix/services/memory_citation_manager.py`
- `mirix/services/memory_manager_helpers.py`
- `mirix/services/memory_source_manager.py`
- `mirix/services/message_manager.py`
- `mirix/services/organization_manager.py`
- `mirix/services/procedural_memory_manager.py`
- `mirix/services/raw_memory_manager.py`
- `mirix/services/resource_memory_manager.py`
- `mirix/services/semantic_memory_manager.py`
- `mirix/services/tool_manager.py`
- `mirix/services/user_manager.py`

### 2.4 Custom cache provider deltas vs Redis default

| Concern | Default `RedisCacheProvider` | Custom provider expectations |
|---|---|---|
| Storage shape | Hash for flat tables, JSON for embedding-bearing tables | Implementer chooses; must accept dict and round-trip cleanly |
| TTL | Honoured per-set call (Redis EXPIRE) | Optional; ignoring is acceptable. Manager code never reads its own TTL back. |
| Embedding payload | Embeddings live in the JSON value alongside the row | Same. Caches do not need RediSearch — they only ever round-trip the row. |
| Cache poisoning | Soft delete writes `is_deleted=True` to the cache too; reads must respect it | Same. The manager reads the JSON and then constructs Pydantic; it does *not* re-check `is_deleted` on cache hits, so the cache MUST reflect the latest write. |
| `*_ts` numeric companions | Written by `_update_redis_cache` for memory tables; stripped by `clean_redis_fields` on read | A custom cache provider that does not need these fields can simply round-trip them; nothing else depends on them. |
| Failure mode | Exceptions are swallowed; `logger.warning(...)` and return `None` / `False` | Required. Throwing must not propagate to the request. |
| Auto-registration | `initialize_redis_client()` registers `"redis"` if `MIRIX_REDIS_ENABLED=true` | Custom providers must register themselves explicitly (no MIRIX-side hook) |

### 2.5 Sample configuration & command

Default Redis (zero-code):

```bash
cp docker/env.example .env
# in .env: MIRIX_REDIS_ENABLED=true (already in template)
docker-compose up -d
```

Custom cache wired in by an external consumer (ECMS-style):

```python
# anywhere that runs before the first request
from common.intuit_cache import IntuitCache         # external impl
from mirix.database.cache_provider import register_cache_provider

cache = IntuitCache.from_env()
await cache.start()
register_cache_provider("intuit_cache", cache)
```

If you also want the cache without ever touching Postgres for read
shape (i.e. you are running the *full* ECMS stack), see §5 — the
relational and search providers must both be registered too.

---

## 3. How the Relational Database Provider Works

The relational provider replaces the **authoritative store** for
all CRUD operations on the operational tables (organizations, users,
clients, agents, tools, blocks, messages, raw_memory, the six memory
tables, memory_sources, source_messages, memory_citations, steps).
When registered, it **fully bypasses** the SQLAlchemy ORM session
path; when not registered, the manager falls back to
`session_maker()` over Postgres.

### 3.1 Interface

```9:60:mirix/database/relational_provider.py
Expected methods (duck typing, all async):

    async create(table: str, data: dict, event_context=None) -> dict
    async read(table: str, identifier: str, include_relationships=None) -> Optional[dict]
    async list(
        table: str, *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        time_range: Optional[Dict[str, Optional[datetime]]] = None,
        include_relationships: Optional[list] = None,
        **kwargs,
    ) -> list[dict]
    async update(table: str, identifier: str, data: dict, event_context=None) -> dict
    async delete(table: str, identifier: str, soft: bool = True, event_context=None) -> bool
    async hard_delete(table: str, identifier: str, event_context=None) -> bool
    async size(
        table: str, *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        scopes: Optional[List[str]] = None,
    ) -> int
    async find_using_filter(
        table: str, filter_item: Any, pageable_filter: Any = None,
        limit: Optional[int] = None, sort: Optional[str] = None,
    ) -> list[dict]
    async bulk_delete(...)
    async bulk_upsert(...)
    async find_using_named_query(
        table: str, query_name: str, *,
        params: Optional[Dict[str, Any]] = None,
        ...
    ) -> list
    async mutate_using_named_query(...)
```

The interface is intentionally larger than the cache provider's:
named queries, soft/hard delete distinction, scoping arguments, and
the `event_context` hook are all required for parity with what the
ORM path can do.

Important interface contracts:

- **Soft vs hard delete.** `delete` defaults to `soft=True` —
  flips `is_deleted=True` on the row but leaves it physically.
  `hard_delete` is the destructive variant. Cascade semantics
  (e.g. `delete_user_by_id` soft-deleting all memory rows for a
  user) are implemented in the *manager*, not the provider — the
  manager calls `mutate_using_named_query` per cascade target.
- **`event_context`** lets the relational provider emit a domain
  event onto the search provider's pipeline (memory tables only).
  The default ORM path does not have this; instead the cache and
  Postgres are kept consistent by `_update_redis_cache`.
- **Named queries** correspond to YAML-defined SQL fragments on
  the relational provider side. MIRIX never embeds SQL itself;
  every named-query call passes a query name and a params dict. The
  query names follow the convention `<manager>.<verb>` (e.g.
  `agent_manager.list_agents_asc`).
- **`include_relationships=["tools"]`** is the provider equivalent
  of SQLAlchemy's `joinedload` — used by `agent_manager` to fetch
  agents with their tools in one round-trip.
- **`filter_tags`** is passed through to the relational provider's
  filter-tag DSL via the shared
  `mirix/database/filter_tags_query.py` helpers (`apply_filter_tags`
  on the SQL side, `build_filter_tags_redis` on the Redis search
  side, plus `can_redis_handle` to gate operator support).
- **`scopes`** maps to `filter_tags.scope = ANY(:scopes)` predicate
  on the SQL side; combined with `apply_access_predicate` it
  enforces the read-scope guard described in §2.6 of
  `Mirix_API_flow_document.md`.
- **`time_range`** keys are `{field}__gte` / `{field}__lte` (e.g.
  `created_at__gte`) — the same shape `RawMemoryManager.search_raw_memories`
  accepts at the HTTP layer.
- **`actor=`** is allowed as a kwarg on `read` (and `delete` /
  `update` in some manager call sites). The provider uses it to
  enforce `organization_id` and `_created_by_id` (a.k.a.
  `ipsr_entity_owner`) match without the manager building a SQL
  predicate.

### 3.2 Registry semantics

Same as cache: last-writer-wins, single active provider, returns
`None` when unregistered.

```101:130:mirix/database/relational_provider.py
def register_relational_provider(name: str, provider: Any) -> None:
    ...
    _relational_providers[name] = provider
    _active_provider_name = name


def get_relational_provider() -> Optional[Any]:
    ...
    if _active_provider_name and _active_provider_name in _relational_providers:
        return _relational_providers[_active_provider_name]
    return None
```

### 3.3 Touch-points in the API flow

The relational provider is consulted from every CUD-side manager
method. The pattern is uniform:

```python
from mirix.database.relational_provider import get_relational_provider

provider = get_relational_provider()
if provider:
    # provider branch — fully replaces the ORM path
    result = await provider.create("agents", data_dict)
    return PydanticAgentState(**result)

# fallback: SQLAlchemy ORM
async with self.session_maker() as session:
    new_agent = AgentModel(**data)
    await new_agent.create_with_redis(session, actor=actor)
    return new_agent.to_pydantic()
```

Endpoint-level summary (extends §3 of `Mirix_API_flow_document.md`):

| Endpoint | Manager method that hits the relational provider | Provider call(s) |
|---|---|---|
| `POST /agents/meta/initialize`, `POST /agents` | `AgentManager.create_agent` | `provider.create("agents", ...)` (organisation/parent invariants on the manager side) |
| `GET /agents/{id}` | `AgentManager.get_agent_by_id` | `find_using_named_query("agents","agent_manager.get_agent_by_id", include_relationships=["tools"])` |
| `GET /agents` | `AgentManager.list_agents` | `find_using_named_query("agents","agent_manager.list_agents_{asc,desc}")` or `…_by_query_text` |
| `PATCH /agents/{id}/system`, MCP / tool updates | `AgentManager.update_agent` | `provider.update("agents", id, scalar_fields)` + cache invalidation |
| `DELETE /agents/{id}` | `AgentManager.delete_agent` | `provider.delete("agents", id, soft=True)` |
| `POST /tools` | `ToolManager.create_tool` (provider path used when registered) | `provider.create("tools", ...)` (with UPSERT semantics on `name` per org) |
| `GET /tools/{id}` | `ToolManager.get_tool_by_id` | named query then cache populate |
| `POST /blocks`, `PATCH /blocks/*` | `BlockManager.create_or_update_block`, `update_block` | `provider.bulk_upsert("block", ...)` and `provider.update("block", id, ...)` |
| `DELETE /clients/{id}/memories`, `DELETE /users/{id}/memories` | `ClientManager.delete_memories_by_client_id`, `UserManager.delete_memories_by_user_id` | `mutate_using_named_query("messages","message_manager.update_by_client_id", …)` etc. (one named query per cascade target) |
| `POST /memory/add` (eventual) | `EpisodicMemoryManager.create_event`, `SemanticMemoryManager.create_item`, … via the meta-agent's tool calls | `provider.create("episodic_memory", row_dict)` with `_created_by_id` and `organization_id` |
| `PATCH /memory/<type>/{id}` | `<Type>MemoryManager.update_item` | `provider.update("<type>_memory", id, diff_dict)` |
| `DELETE /memory/<type>/{id}` | `<Type>MemoryManager.delete_*_by_id` | `provider.hard_delete("<type>_memory", id)` |
| `POST/GET/PATCH/DELETE /memory/raw` | `RawMemoryManager.*` | `provider.create/read/update/hard_delete("raw_memory", ...)` |
| `GET /memory-sources/{id}[/messages]` (write side) | `MemorySourceManager.create`, `SourceMessageManager.bulk_insert` | `provider.create` / `bulk_upsert` |
| Admin / dashboard auth | `ClientAuthManager` → `ClientManager` | provider used for all CRUD on `clients` |
| `POST /memory/add` provenance write | `MemoryCitationManager.write` | `provider.create("memory_citations", ...)` (uniqueness of `(memory_id, memory_source_id)` enforced by L3 cache) |

The agent runtime (`mirix/agent/agent.py`) only consults the
relational provider in one place: `_fetch_recent_indexing_lag_window`
in the dedup hybrid-read helper (see §5 below).

### 3.4 Logical / data deltas vs Postgres-only path

The custom relational provider isn't a drop-in *clone* of Postgres
— several semantic shifts come with it:

1. **No row-level locks.** The Postgres path uses
   `SELECT … FOR UPDATE` on `raw_memory.update` to serialise
   concurrent append/merge. The relational provider has no such
   primitive exposed; updates are last-write-wins. This is
   documented in `RawMemoryManager.update_raw_memory`.
2. **Embeddings are not computed in the manager.** When a relational
   provider is registered, memory managers skip the local
   `embedding_model(...).get_text_embedding(...)` call and pass the
   raw text fields through; the search provider computes embeddings
   on its side. See `RawMemoryManager.create_raw_memory`:
   `if provider: raw_memory.context_embedding = None`.
3. **Pre-generated UUIDs.** `agent_manager.create_agent` writes
   `data_dict["id"] = str(uuid.uuid4())` without the typical
   `agent-...` / `tool-...` prefix; the relational provider's
   storage schema requires raw UUIDs. The natural-key prefixed ID
   is stored separately as the entity_key.
4. **`_created_by_id` becomes `ipsr_entity_owner`.** The provider
   maps `_created_by_id` from MIRIX schemas to its
   ownership column (`ipsr_entity_owner`); the manager forwards
   `actor.id` and lets the provider do the translation.
5. **Named-query routing.** Operations that the SQLAlchemy path
   builds via the query builder (sort + filter + pagination) become
   pre-defined named queries on the provider side. The manager
   calls `find_using_named_query("agents","agent_manager.list_agents_asc", params={...})`
   instead of building a query. This means: any new sort / filter
   the dashboard wants must have a corresponding named query
   defined on the relational provider — there is no ad-hoc
   query path in the provider branch.
6. **Soft-delete cascades go through `mutate_using_named_query`.**
   Where the ORM path uses
   `MessageModel.update_by_user_id` (a class method on the ORM),
   the provider path passes the same intent as a named query:
   `mutate_using_named_query("messages","message_manager.update_by_user_id", params={"userId":..., "isDeleted":True})`.
7. **`size` is not always exact.** The relational provider's `size`
   reflects rows it sees; if a write from another writer is
   pending, the count lags behind. For a count that must reconcile
   with the search index, callers use the search provider's `count`
   (`SemanticMemoryManager.size_semantic_items` documents this).
8. **Engine tables (`messages`, `steps`) vs domain tables.** Engine
   tables are modelled directly; domain tables (the seven memory
   tables, `block`, `agents`, `tools`, etc.) get domain-event
   propagation when `event_context` is supplied. Engine-table
   manager calls deliberately omit `event_context`; the inline
   comments in `user_manager.py` note this:
   `"engine table — no domain events needed"`.
9. **`include_relationships=["tools"]` ≈ `joinedload`.** SQLAlchemy
   uses lazy-joined relations on the agent → tools edge; the
   provider returns the full nested structure in one round-trip.

### 3.5 Sample configuration & command

The relational provider is **not** auto-wired by MIRIX. A consumer
must register it explicitly. ECMS does this in
`common.ips_provider_setup.register_ips_providers`, called from
`context-and-memory-service`'s startup hook. Assume that ECMS implements
IPSRelational, IPSSearch, and IPSCache providers in the following
samples.

Conceptual sketch:

```python
# In your application's startup (before first request)
from mirix.database.relational_provider import register_relational_provider
from mirix.database.search_provider import register_search_provider
from mirix.database.cache_provider import register_cache_provider
from mirix.database.provider_validation import validate_provider_pairing_or_raise

# Build provider instances against your stack
relational = AsyncIPSRelationalProvider(config)
await relational.initialize()           # opens connection pool, loads named queries

search = IPSSearchProvider(config)
await search.initialize()

cache = IPSCacheProvider(config)
await cache.start()

register_cache_provider("ips_cache", cache)
register_relational_provider("ips_relational", relational)
register_search_provider("ips_search", search)

# Coupled validation: relational and search must come together
validate_provider_pairing_or_raise()
```

Run command (when consuming MIRIX as a library, e.g. ECMS):

```bash
# ECMS owns its own entrypoint; MIRIX's start_server.py is bypassed
cd context-and-memory-service
poetry run uvicorn ecms.app:app --port 8000
```

To run the *standalone* MIRIX server with a custom relational
provider, you would have to register it before
`mirix.server.rest_api.initialize()` is invoked. The simplest way
is a small wrapper module:

```python
# bootstrap_with_custom_providers.py
import asyncio
from mirix.database.relational_provider import register_relational_provider
from mirix.database.search_provider import register_search_provider
from my_providers import MyRel, MySearch
from mirix.server.rest_api import app

async def boot():
    rel = MyRel(...);  await rel.initialize();   register_relational_provider("my_rel", rel)
    sch = MySearch(...); await sch.initialize();  register_search_provider("my_search", sch)

asyncio.get_event_loop().run_until_complete(boot())
# now run uvicorn against `app`
```

Reminder from §0: registering a relational provider without a
matching search provider (or vice versa) is **invalid**;
`validate_provider_pairing_or_raise()` will raise.

---

## 4. How the Search Provider Works

The search provider is the registry for **retrieval, count, and
single-row-by-id reads** against the search index — the seven
memory tables (`block`, `episodic_memory`, `semantic_memory`,
`procedural_memory`, `resource_memory`, `knowledge_vault`,
`raw_memory`). It replaces both:

- the SQLAlchemy `list_*` paths (BM25 via `pg_bm25`, vector via
  `pgvector` cosine), and
- the Redis-as-search path (`RedisMemoryClient.search_text` /
  `search_vector` / `search_recent`).

When a search provider is registered, neither of those paths is
consulted; the request is routed straight to it.

### 4.1 Interface

```9:43:mirix/database/search_provider.py
async def search(
    table: str, *,
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    search_method: str = "embedding",         # 'embedding' | 'bm25' | 'string_match' | 'fuzzy_match'
    search_field: Optional[str] = None,       # 'summary' | 'details' | 'caption' | …
    user_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    filter_tags: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[str]] = None,
    limit: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    similarity_threshold: Optional[float] = None,
    sort: Optional[str] = None,
    cursor: Optional[str] = None,
    time_range: Optional[Dict[str, Optional[datetime]]] = None,
    **kwargs,                                  # e.g. sensitivity=…
) -> Tuple[List[Dict], Optional[str]]

async def count(
    table: str, *,
    user_id: Optional[str] = None, organization_id: Optional[str] = None,
    filter_tags: Optional[Dict[str, Any]] = None,
    scopes: Optional[List[str]] = None,
) -> int

async def get_by_id(
    table: str, identifier: str, *, user_id: Optional[str] = None,
) -> Optional[Dict]
```

The single unified `search()` signature is deliberate: it covers
every list/search call shape in the manager layer.

| Caller (manager method) | What it asks for |
|---|---|
| Memory list / search (six tables) | `query_text`, `query_embedding`, `search_method`, `search_field`, `similarity_threshold` |
| `RawMemoryManager.search_raw_memories` | `filter_tags`, `sort`, `cursor`, `time_range` |
| `BlockManager.get_blocks` | `filter_tags`, `scopes`, `limit`, exact-match shortcuts via `**kwargs` (`label`, `id`) |
| `KnowledgeVaultManager.list_knowledge` | `sensitivity` via `**kwargs` |
| Org-wide endpoints (`/memory/search_all_users`) | `organization_id` only, no `user_id` |
| Cursor pagination (raw_memory) | `sort` + `cursor`; provider returns `(rows, next_cursor)` |

`count()` mirrors `get_total_number_of_items` shape with the same
scoping arguments; this is what the dashboard uses for "how many
episodic events does this user have?". It is the search-provider
view of the count, which lags behind the relational provider by the
indexing window.

`get_by_id()` is used for hybrid merge dedup when the same memory
needs to be fetched from the search index by ID rather than from the
authoritative store. **Single-row reads by ID for the canonical row
go through the relational provider's `read()`**, not this method —
documented in the module docstring.

### 4.2 Registry semantics

Identical to the other two registries: last-writer-wins, single
active provider, returns `None` when unregistered. See
`mirix/database/search_provider.py`.

### 4.3 Touch-points in the API flow

| Endpoint | Manager method | Provider call(s) |
|---|---|---|
| `POST /memory/retrieve/conversation`, `GET /memory/retrieve/topic` | per-type `list_*` (`list_episodic_events`, `list_semantic_items`, …) | `search_provider.search(table, query_text, search_method='bm25', search_field='details'/'summary'/...)` |
| `GET /memory/search` (per-type and `memory_type='all'`) | per-type `list_*` again, called from `asyncio.gather` | `search_provider.search(...)` with `query_embedding` if precomputed; falls through to the same per-type code paths |
| `GET /memory/search_all_users` | iterates users, otherwise same | same |
| `GET /blocks` | `BlockManager.get_blocks` | `search_provider.search("block", query_text="", search_method="string_match", filter_tags=..., scopes=...)` |
| `POST /memory/search_raw` | `RawMemoryManager.search_raw_memories` | `search_provider.search("raw_memory", filter_tags=..., scopes=..., sort=..., cursor=..., time_range=...)`; returns `(rows, next_cursor)` |
| Memory size / count UIs | `<Type>MemoryManager.size_*` | `search_provider.count(table, user_id, organization_id)` |
| Hybrid dedup at write time | `Agent._fetch_recent_indexing_lag_window` → `fetch_and_dedup_candidates` | `search_provider.search(...)` in parallel with `relational_provider.list(...)` |
| Per-row search index read by id (rare) | internal merge dedup | `search_provider.get_by_id(table, id, user_id=...)` |

Every per-memory-type manager runs the same routing pattern:

```python
from mirix.database.search_provider import get_search_provider

search_provider = get_search_provider()
if search_provider:
    results, _next = await search_provider.search(
        "episodic_memory",
        query_text=query, query_embedding=embedded_text,
        search_method=search_method, search_field=search_field,
        user_id=user.id, organization_id=user.organization_id,
        filter_tags=filter_tags, scopes=scopes,
        limit=limit, start_date=start_date, end_date=end_date,
        similarity_threshold=similarity_threshold,
    )
    return [PydanticEpisodicEvent(**r) for r in results]

# else: try Redis-as-search
redis_client = get_redis_client()
if use_cache and redis_client:
    ...search via redis...

# else: SQLAlchemy ORM (BM25 via pg_bm25 or pgvector cosine)
async with session_maker() as session:
    return await EpisodicEventModel.list(...)
```

Note the **three-level fallback** — when the search provider is
registered, the Redis-as-search and ORM list paths are completely
skipped. This is the central reason cache and search are decoupled
from each other in the registry: the cache is still consulted for
single-row reads, but search routing is a separate decision.

### 4.4 Logical / data deltas vs Postgres-only path

1. **No `pg_bm25` dependency.** The Postgres path uses
   `pg_bm25.bm25_score(...)` for `search_method='bm25'` — see the
   `EpisodicEventModel.list` docstring. The search provider is
   responsible for delivering equivalent ranking; it can use any
   indexing technology (Lucene-class engines, Vespa, MIRIX's
   in-memory BM25 fallback for SQLite, …).
2. **Embeddings are computed elsewhere.** When a search provider is
   registered, the manager passes `query_embedding=embedded_text`
   only when the caller already has it; otherwise it passes
   `query_text` and lets the provider embed on its side. This is
   how the indexing-lag-aware ECMS pipeline avoids double embedding.
3. **`count()` is eventually consistent.** The manager docstring
   for `SemanticMemoryManager.size_semantic_items` is explicit:
   *"this count is the Search-provider-visible row count, which
   lags behind the source-of-truth Relational provider by the
   provider's indexing window."* Callers needing exact counts must
   use the relational provider's `size`.
4. **Filter-tag operator support is provider-defined.** The shared
   `mirix/database/filter_tags_query.py::can_redis_handle()` gate is
   a Redis-search-specific guard; the search provider may support
   richer operators (`$gte`, `$in`, `$or`) than the Redis fallback.
   When a complex filter doesn't fit one backend, that backend is
   skipped — the manager simply asks the search provider, which
   knows what its own backend supports.
5. **Cursor pagination is provider-shaped.** The manager forwards
   `cursor` and `sort` opaquely; the provider returns `next_cursor`
   in the second tuple element. The Postgres path uses keyset
   pagination on `(updated_at, id)` (or `created_at` /
   `occurred_at`); a custom search provider can use any cursor
   format as long as it round-trips.
6. **`scopes` is the only mandatory predicate.** Every manager
   passes `scopes=client.read_scopes` (or `any_scopes` for
   `BlockManager.get_blocks`). The provider MUST honour this; a
   search provider that returned rows outside the read scope would
   constitute a data-leak bug.
7. **`block` is searched too.** The block (core memory) table is in
   the search provider's domain; `BlockManager.get_blocks` calls
   `search_provider.search("block", query_text="", search_method="string_match", filter_tags=..., scopes=...)`.
   This is true even though the block table has no embedding —
   the provider must support `search_method='string_match'` /
   exact-match shortcuts for it.
8. **`memory_sources` and `memory_citations` are NOT in the search
   provider's domain.** Those sidecar tables go through the
   relational provider only; the search provider does not need to
   index them. (The cache, however, does cache `memory_sources`.)
9. **Org-wide search variant.** When `user_id=None` and
   `organization_id` is set, the provider scopes to the org;
   `/memory/search_all_users` and `BlockManager.get_blocks(user=None,
   organization_id=..., any_scopes=[...])` rely on this.

### 4.5 Sample configuration & command

The search provider must be registered alongside the relational
provider — `validate_provider_pairing_or_raise` enforces this. There
is no MIRIX-side auto-registration; the consumer wires it.

```python
from mirix.database.search_provider import register_search_provider
from mirix.database.relational_provider import register_relational_provider
from mirix.database.provider_validation import validate_provider_pairing_or_raise

relational = MyRelationalProvider(...); await relational.initialize()
search = MySearchProvider(...); await search.initialize()
register_relational_provider("my_rel", relational)
register_search_provider("my_search", search)
validate_provider_pairing_or_raise()
```

To run with **only** the search provider but no custom relational
backing (i.e. keep Postgres for CRUD, swap search alone): not
supported. The validation call above raises, and even if you skip
the validation, the hybrid-read helper uses both.

To **disable** the search provider entirely (Postgres + pgvector +
pg_bm25 default), simply do not register one and do not register the
relational one. With Postgres as the search backend, BM25 still
works via `pg_bm25` and vector search via pgvector cosine; this is
the default `docker-compose up -d` path.

To tune the **hybrid indexing-lag window** (used by the dedup helper):

```python
from mirix.services.hybrid_search_helper import set_hybrid_window_seconds
set_hybrid_window_seconds(5)   # default; configure from your settings
```

ECMS sets this from `ips_hybrid_read_window_seconds` at startup.

---

## 5. How All the Custom Providers Work Together

This section ties §1–§4 together: how the three registries interact
on a real request, where they share state, and where they're
deliberately decoupled.

### 5.1 Decision matrix per request stage

For any service-manager call (`MyManager.do_thing`), the routing is:

```
1. Cache?          ── get_cache_provider() → if hit and not stale, return.
2. Relational?     ── get_relational_provider() → if registered, do CRUD here.
3. Search?         ── get_search_provider() → if registered, do retrieval here.
4. Default         ── ORM session_maker() (Postgres) for both CRUD and retrieval,
                      with optional Redis cache around the read.
```

Each level is independent of the others. Concretely:

|                          | Cache | Relational | Search | Default Postgres |
|---|:-:|:-:|:-:|:-:|
| `GET /<entity>/{id}` (single-row read by ID) | ✓ | ✓ | — | ✓ (ORM `read`) |
| `GET /memory/search` (multi-row retrieval) | — | — | ✓ | ✓ (`pg_bm25` / `pgvector`) |
| `GET /memory/raw/{id}` (single-row read) | ✓ | ✓ | — | ✓ |
| `POST /memory/raw` (write) | invalidate | ✓ | — | ORM + cache hook |
| `POST /memory/add` derived path (write) | invalidate | ✓ | — | ORM + cache hook |
| Hybrid dedup at write time | — | ✓ (recent window) | ✓ (relevant) | gather() over ORM |
| `GET /memory-sources/{id}` | ✓ (JSON) | ✓ | — | ORM |

Two notes:

- **Search is never called for single-row reads by ID.** Those go
  through cache → relational → ORM. `search_provider.get_by_id` is
  reserved for index-side merge operations (advanced internal
  callers).
- **Cache is never authoritative.** Even when present, every cache
  hit is followed by a Pydantic validation that uses the cached
  data; the underlying store is unaffected. A poisoned cache
  resolves itself as soon as the cached entry expires or the row
  is updated (which invalidates the cache).

### 5.2 Hybrid read at write time

The save-flow dedup helper is the one place where two providers
must speak in lockstep. From `mirix/services/hybrid_search_helper.py`:

```python
async def fetch_and_dedup_candidates(table, search_provider, relational_provider, *, ...):
    # Search-side: ranked relevant candidates
    search_coro = search_provider.search(table, ..., limit=effective_limit)

    # Relational-side: rows written within the indexing-lag window
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    recent_coro = relational_provider.list(
        table,
        time_range={"updated_at__gte": cutoff.isoformat(),
                    "created_at__gte": cutoff.isoformat()},
        time_range_or_null_updated=True,
        limit=recent_cap,
    )

    (search_results, _next), recent_records = await asyncio.gather(search_coro, recent_coro)
    return DedupCandidates(relevant=..., recent=...)
```

Both are awaited concurrently via `asyncio.gather`. The labelled
buckets — `relevant` (ranked from search) and `recent` (write-window
from relational) — are passed to the LLM in two separate sections so
it can detect duplicate writes that the search index has not yet
indexed. The window is configurable via `set_hybrid_window_seconds`;
default 5s.

Fail-closed: if either provider call raises, the exception
propagates. We do not return partial buckets.

### 5.3 Coupled startup validation

```13:49:mirix/database/provider_validation.py
def validate_provider_pairing_or_raise() -> None:
    """
    Valid states:
        - Both relational and search providers registered.
        - Neither relational nor search provider registered.

    Invalid states (raises RuntimeError):
        - Relational provider registered but search provider is not.
        - Search provider registered but relational provider is not.

    Cache provider state is independent and not checked here.
    """
```

Run this immediately after `register_*` in your startup hook to
fail fast on misconfiguration. Cache is intentionally excluded from
the check because cache-only and cache-with-providers are both
legal shapes.

### 5.4 Example: `POST /memory/add` write path with all three providers

```
HTTP POST /memory/add
   |
   |  REST handler (rest_api.py :: add_memory)
   |  - get_client_and_org()
   |  - put_messages(...) → QueueMessage
   v
Queue (Kafka or in-memory)
   |
   |  QueueWorker pulls
   v
AsyncServer.send_messages → Agent.step (MetaMemoryAgent)
   |
   |  Agent._persist_memory_source:
   |     ├── memory_source_manager.create(...)   # relational.create("memory_sources", ...)
   |     │                                        # → cache.set_json(MEMORY_SOURCE_PREFIX+id, ...)
   |     └── source_message_manager.bulk_insert(...)  # relational.bulk_upsert(...)
   |
   |  inner_step → meta agent dispatches sub-agents (semantic, episodic, …)
   |
   |  Each sub-agent loop:
   |     ├── build_system_prompt_with_memories:
   |     │     for owning agent type → fetch_and_dedup_candidates(
   |     │                                  search.search(...), relational.list(time_range=...)
   |     │                                ) with asyncio.gather
   |     ├── LLM tool call → memory_tool (e.g. semantic_memory_insert)
   |     ├── memory tool calls SemanticMemoryManager.create_item:
   |     │     → relational.create("semantic_memory", row)         # source of truth
   |     │     → cache.set_json(SEMANTIC_PREFIX+id, row, ttl)       # best-effort
   |     │     [search index is updated by relational provider's domain-event hook]
   |     └── _write_citation:
   |           → cache.get_json(citation-exists) (L3 dedup)
   |           → relational.create("memory_citations", row)
   |           → cache.set_json(citation-exists, ...)
   |
   |  memory_source_manager.mark_processing_complete:
   |     → relational.update("memory_sources", id, {"processing_complete": True})
   |     → cache.set_json(MEMORY_SOURCE_PREFIX+id, refreshed_data, ttl)
   v
Return MirixUsageStatistics
```

A few invariants visible above:

- The relational provider is hit on **every** persistence step;
  the cache is best-effort everywhere.
- The search provider only appears in the **read** half of the dedup
  helper; the relational provider's `event_context` is what propagates
  the new memory to the search provider's index.
- The L3 citation-dedup uses the cache as a fast-path (the
  `_exists_cache_key` JSON write), but the canonical uniqueness
  guarantee is the L1 source-level uniqueness on
  `memory_sources` enforced by the relational provider.

### 5.5 Example: `GET /memory/search` read path with all three providers

```
HTTP GET /memory/search?query=...&memory_type=all
   |
   v
REST handler (rest_api.py :: search_memory)
   - get_client_from_jwt_or_api_key()
   - precompute_embedding_for_search(query) → embedded_text
   - asyncio.gather(
        search_episodic(query, embedded_text, ...),    # SemanticMemoryManager.list_*
        search_resource(...),
        search_procedural(...),
        search_knowledge(...),
        search_semantic(...),
        [search_core(...) optional],
     )
   |
   |  Each search_* delegates to its manager's list_*, which:
   |     ├── search_provider.search(table, query_text, query_embedding,
   |     │                          search_method, search_field,
   |     │                          user_id, organization_id,
   |     │                          filter_tags, scopes=client.read_scopes,
   |     │                          start_date, end_date, similarity_threshold)
   |     │       returns (rows, next_cursor)
   |     └── [if include_citations] _attach_citations_to_results:
   |               for memory_id batch:
   |                   relational.list("memory_citations", filter_tags={memory_ids}, ...)
   |                   relational.read("memory_sources", source_id)  # batched
   |
   v
JSON response
```

Notes:

- The cache provider is **not** consulted here. Search is owned by
  the search provider; per-row caches do not help when the route
  returns N freshly-ranked rows.
- Reads of `memory_sources` for citation enrichment go through the
  relational provider, but those are individually small and the
  read-through cache (set on previous reads) absorbs most repeats.
- `client.read_scopes` is the access-control predicate; the search
  provider MUST honour it.

### 5.6 Sample combined configuration & command

The full ECMS / Provider stack: cache + relational + search, with
Postgres + Redis disabled.

```bash
# .env
MIRIX_REDIS_ENABLED=false                 # MIRIX-side Redis off — IPS cache replaces it
MIRIX_PG_URI=                             # leave blank or set to a no-op DSN
GEMINI_API_KEY=...
MIRIX_LANGFUSE_ENABLED=true
MIRIX_LANGFUSE_PUBLIC_KEY=...
MIRIX_LANGFUSE_SECRET_KEY=...
```

```python
# context-and-memory-service/ecms/startup.py
import asyncio
from common.ips_provider_setup import register_ips_providers
from mirix.database.provider_validation import validate_provider_pairing_or_raise

async def on_startup():
    # register_ips_providers() does:
    #   1. construct AsyncIPSRelationalProvider, IPSSearchProvider, IPSCacheProvider
    #   2. await each provider.initialize()
    #   3. register_cache_provider("ips_cache", cache)
    #   4. register_relational_provider("ips_relational", relational)
    #   5. register_search_provider("ips_search", search)
    #   6. set_hybrid_window_seconds(config.ips_hybrid_read_window_seconds)
    await register_ips_providers()
    validate_provider_pairing_or_raise()

asyncio.run(on_startup())
```

Run command (ECMS owns the entrypoint, MIRIX is imported as a library):

```bash
cd context-and-memory-service
poetry install
poetry run uvicorn ecms.app:app --host 0.0.0.0 --port 8000
```

Run command for **MIRIX standalone with default Postgres + Redis**
(no custom providers):

```bash
cp docker/env.example .env
# edit .env to set GEMINI_API_KEY / OPENAI_API_KEY etc.
docker-compose up -d
# or, without Docker:
pip install -e .
python scripts/start_server.py --port 8531
```

Run command for **MIRIX standalone with custom providers**: write a
small bootstrap script that registers your providers before importing
`mirix.server.rest_api`. Example skeleton:

```python
# bootstrap.py
import asyncio
import uvicorn

async def register():
    from mirix.database.cache_provider import register_cache_provider
    from mirix.database.relational_provider import register_relational_provider
    from mirix.database.search_provider import register_search_provider
    from mirix.database.provider_validation import validate_provider_pairing_or_raise
    from my_providers import MyCache, MyRel, MySearch

    cache = MyCache()
    rel = MyRel(); await rel.initialize()
    search = MySearch(); await search.initialize()

    register_cache_provider("my_cache", cache)
    register_relational_provider("my_rel", rel)
    register_search_provider("my_search", search)
    validate_provider_pairing_or_raise()

asyncio.get_event_loop().run_until_complete(register())

from mirix.server.rest_api import app   # imports after registration
uvicorn.run(app, host="0.0.0.0", port=8531)
```

```bash
python bootstrap.py
```

### 5.7 Failure modes and observability

| Symptom | Likely cause | Where to look |
|---|---|---|
| `RuntimeError: Relational provider is registered but Search provider is not` (or the inverse) | Pairing rule violated | `validate_provider_pairing_or_raise()` should be called immediately after registration |
| Cache hit returning stale data | Write-through invalidation not run because exception swallowed | `mirix/orm/sqlalchemy_base.py::_update_redis_cache` and per-manager `cache_provider.delete` calls are wrapped in `try/except`; check WARNING-level logs |
| Search returning fewer results than expected | Index lag on the search provider; recent writes not yet visible | The dedup helper exists exactly to expose this lag at write time. For reads, callers may force a re-query after a TTL or ask the relational provider for an exact count. |
| Domain-event lost between relational and search providers | `event_context` not threaded from manager to provider | Memory-table CUD calls in the manager pass `event_context`; engine tables (`messages`, `steps`) deliberately don't |
| Provider call failing on `block` searches | `search_method='string_match'` not implemented | Block search uses `search_method='string_match'` with `filter_tags` + `scopes`; provider must accept this |

LangFuse spans are emitted around every memory-update step (see
`mirix/observability/skip_spans.py`), so misconfigured providers
typically surface as spans without children, or with the
"Idempotency Skip" span fired for the wrong reason.

---

## 6. Related Reading

- `docs/Mirix_API_flow_document.md` — endpoint catalog and the
  request-to-storage trace that this document refines.
- `docs/Mirix_async_native_changes.md` — async-native rationale and
  the five approved sync touch-points.
- `docs/ARCHITECTURE.md` — the source-of-truth diagram and table
  layer responsibilities.
- `mirix/database/cache_provider.py`,
  `mirix/database/relational_provider.py`,
  `mirix/database/search_provider.py` — the registry modules
  themselves; tiny and worth reading top-to-bottom.
- `mirix/database/redis_client.py` and
  `mirix/database/redis_cache_provider.py` — the default cache
  implementation; reference shape for custom providers.
- `mirix/database/provider_validation.py` — the coupled pairing
  check.
- `mirix/services/hybrid_search_helper.py` — the
  search + relational hybrid-read helper and the configurable
  indexing-lag window.
- `mirix/services/memory_manager_helpers.py` — the shared cache
  invalidation + provider list helpers used across the seven memory
  tables.
- `mirix/orm/sqlalchemy_base.py::_update_redis_cache` — the
  ORM-side write-through hook; reference for the per-table
  serialization rules.
- `tests/test_provider_registration.py`,
  `tests/test_provider_validation.py`,
  `tests/test_manager_delegation.py`,
  `tests/test_cache_provider.py` — the contract tests for the
  provider abstractions.
