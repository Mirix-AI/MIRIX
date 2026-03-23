# MIRIX Async-Native Rewrite

## 1. Why Async-Native

MIRIX is a multi-agent system where every user request fans out into
database queries, Redis lookups, LLM API calls, embedding computations,
and Kafka messages -- all I/O-bound. The previous sync codebase serialized
these operations: each blocked thread sat idle waiting for a network
response, and concurrency was limited to the thread-pool size.

Rewriting the stack to be async-native delivers several concrete benefits:

**Higher throughput on the same hardware.**
A single event loop multiplexes thousands of in-flight I/O operations
without dedicating a thread to each one. Connection pools (asyncpg, Redis,
httpx) are shared across all coroutines, so the server handles more
concurrent users with fewer file descriptors, less memory, and less
context-switching overhead.

**End-to-end consistency with FastAPI / Uvicorn.**
FastAPI is async-first. When route handlers are `async def` and directly
`await` the server/manager/agent/LLM chain, there is no implicit offload
to a thread-pool executor. This removes an entire class of subtle bugs
(thread-safety of shared state, session leaks across threads) and makes
the call stack easy to reason about.

**Natural streaming and SSE.**
LLM token streaming and Server-Sent Events map directly to async
generators. No background thread is needed to feed an SSE response; the
generator yields tokens as they arrive from the LLM provider.

**In-process background workers.**
Queue consumers (memory extraction, cleanup) run as `asyncio.Task`s in
the same process. This simplifies deployment (one container, one process)
while keeping workers non-blocking.

**Lower tail latency.**
`asyncio.sleep`-based retries and exponential back-off do not occupy a
thread during the wait, freeing the loop to serve other requests.

---

## 2. High-Level Changes

### 2.1 External Library Migrations

| Layer | Sync (before) | Async (after) |
|-------|---------------|---------------|
| **Database driver** | `pg8000` / `psycopg2-binary` | **asyncpg** (PostgreSQL), **aiosqlite** (SQLite) |
| **SQLAlchemy** | `create_engine`, `sessionmaker`, `Session` | `create_async_engine`, `async_sessionmaker`, `AsyncSession` |
| **Redis** | `redis.Redis` | **redis.asyncio.Redis** with `hiredis` |
| **HTTP client** | `requests` | **httpx.AsyncClient** |
| **OpenAI** | `openai.OpenAI` | `openai.AsyncOpenAI` |
| **Anthropic** | `anthropic.Anthropic` | `anthropic.AsyncAnthropic` |
| **Azure OpenAI** | `AzureOpenAI` | `AsyncAzureOpenAI` |
| **Google AI** | sync `genai` calls | async `genai` + `httpx.AsyncClient` |
| **Kafka** | sync kafka-python (if used) | **aiokafka** (`AIOKafkaProducer`, `AIOKafkaConsumer`) |
| **Web search** | `duckduckgo_search` | **asyncddgs** |
| **Google APIs** | sync `google-api-python-client` | **aiogoogle** |
| **Test runner** | sync pytest | **pytest-asyncio** (`asyncio_mode = "auto"`) |

### 2.2 Application-Layer Changes

**ORM base** (`mirix/orm/sqlalchemy_base.py`)
All CRUD methods (`create`, `read`, `update`, `delete`, `list`) are
`async def`. Sessions are used via `async with session`. Retry decorators
use `asyncio.sleep()`.

**Service managers** (`mirix/services/`)
All 16 managers are async:

| # | Manager | File |
|---|---------|------|
| 1 | UserManager | `user_manager.py` |
| 2 | ClientManager | `client_manager.py` |
| 3 | ToolManager | `tool_manager.py` |
| 4 | AdminUserManager | `admin_user_manager.py` |
| 5 | OrganizationManager | `organization_manager.py` |
| 6 | BlockManager | `block_manager.py` |
| 7 | MessageManager | `message_manager.py` |
| 8 | CloudFileMappingManager | `cloud_file_mapping_manager.py` |
| 9 | StepManager | `step_manager.py` |
| 10 | AgentManager | `agent_manager.py` |
| 11 | RawMemoryManager | `raw_memory_manager.py` |
| 12 | EpisodicMemoryManager | `episodic_memory_manager.py` |
| 13 | SemanticMemoryManager | `semantic_memory_manager.py` |
| 14 | ProceduralMemoryManager | `procedural_memory_manager.py` |
| 15 | ResourceMemoryManager | `resource_memory_manager.py` |
| 16 | KnowledgeVaultManager | `knowledge_vault_manager.py` |

Every manager method uses `async with self.session_maker()` and `await`
for all database operations.

**LLM API layer** (`mirix/llm_api/`)
`LLMClientBase.send_llm_request()` and `request()` are async. All
provider clients (OpenAI, Anthropic, Azure, Google, Cohere, Mistral, AWS
Bedrock) use their respective async SDK classes. Streaming responses are
`AsyncGenerator`. `retry_with_exponential_backoff()` uses
`asyncio.sleep()`.

**Agent execution** (`mirix/agent/agent.py`)
`step()`, `inner_step()`, `_get_ai_reply()`, and `_handle_ai_response()`
are all async.
Built-in tools (core, memory, extras) are async. User-defined tools
execute in `ToolExecutionSandbox` via `asyncio.create_subprocess_exec()`
(no thread pool).

**MetaAgent** (`mirix/agent/meta_agent.py`)
`MetaAgent.step()`, `initialize()`, and sub-agent orchestration are async.
`MessageQueue` uses `asyncio.Lock` instead of `threading.Lock`.

**Queue system** (`mirix/queue/`)
- `MemoryQueue` wraps `asyncio.Queue`.
- `KafkaQueue` uses `aiokafka` (fully async producer/consumer).
- `QueueWorker` runs as an `asyncio.Task` in the main event loop.

**Server** (`mirix/server/server.py`)
`AsyncServer` (renamed from the former `SyncServer`) exposes async
methods: `send_messages()`, `_step()`, `load_agent()`, `create_agent()`.
A backward-compatible alias `SyncServer = AsyncServer` is retained for
external callers that have not yet updated.

**REST API** (`mirix/server/rest_api.py`)
All route handlers are `async def` and directly `await` server methods.
Zero `asyncio.to_thread` wrappers on the request path. SSE streaming uses
`sse_async_generator()`.

**Client SDK** (`mirix/client/remote_client.py`)
`MirixClient` uses `httpx.AsyncClient` with `RetryTransport`. All public
methods (`add`, `send_message`, `create_agent`, etc.) are async.
`MirixClient.create()` is an async factory for initialization.

**Observability** (`mirix/observability/langfuse_client.py`)
Singleton initialization uses `asyncio.Lock` for coroutine-safe
double-checked locking. The sync LangFuse SDK is called via
`asyncio.to_thread` (see Section 3.1).

**Tests** (`tests/`, `pyproject.toml`)
`pytest-asyncio` with `asyncio_mode = "auto"`. Fixtures in `conftest.py`
are async. `asyncio_default_fixture_loop_scope = "session"`.

---

## 3. Remaining Synchronous Code

The request-serving hot path is fully async. The items below are the only
remaining synchronous touch-points. Each is intentional.

### 3.1 LangFuse SDK

| | |
|---|---|
| **Where** | `mirix/observability/langfuse_client.py` |
| **What** | `Langfuse()` init, `.flush()`, `.shutdown()` are sync SDK calls, wrapped with `await asyncio.to_thread(...)`. |
| **Why** | No official async LangFuse client exists. |
| **Impact** | **Low.** Observability is off the hot path. `to_thread` borrows a thread from the default executor briefly; it does not block the event loop or limit request concurrency. |

### 3.2 Gmail OAuth

| | |
|---|---|
| **Where** | `mirix/functions/mcp_client/gmail_client.py` |
| **What** | `authenticate_gmail_local()` blocks waiting for a browser OAuth redirect. Called via `await asyncio.to_thread(...)`. |
| **Why** | The OAuth flow is inherently blocking (human in the loop). |
| **Impact** | **Low.** One-time auth; not on the per-request path. |

### 3.3 SQLAlchemy DDL at Startup

| | |
|---|---|
| **Where** | `mirix/server/server.py`, `ensure_tables_created()` |
| **What** | `await conn.run_sync(Base.metadata.create_all)` |
| **Why** | SQLAlchemy's DDL/metadata API is sync-only; `run_sync` is the documented pattern for async engines. |
| **Impact** | **None at runtime.** Runs once during application startup. |

### 3.4 Cleanup Job Entry Point

| | |
|---|---|
| **Where** | `mirix/jobs/cleanup_raw_memories.py` |
| **What** | `asyncio.run(delete_stale_raw_memories_async(threshold))` in `__main__`. |
| **Why** | Standard pattern for a standalone script invoked by cron; it bootstraps its own event loop. |
| **Impact** | **None.** Separate process; does not affect the API server. |

### 3.5 Pure CPU Helpers -- Intentionally Sync

| | |
|---|---|
| **Where** | `mirix/utils.py`, `mirix/services/utils.py`, and private helpers in memory managers (`_clean_text_for_search`, `_parse_embedding_field`, `_count_word_matches`, `_preprocess_text_for_bm25`). |
| **What** | String manipulation, regex, JSON parsing, token counting, date formatting, UUID generation. Zero I/O. |
| **Why** | Adding `async def` to a function that never `await`s provides no concurrency benefit. The event loop only yields at `await` points, so an `async def` body with no awaits runs identically to a plain `def` -- but with extra coroutine-object overhead. A function should be `async def` if and only if it performs I/O. (Note: `mirix/services/utils.py::build_query` *is* correctly `async def` because it awaits `embedding_model()`.) |
| **Impact** | **None.** These run in microseconds. If a future helper became CPU-heavy, the correct fix would be `asyncio.to_thread` (offload to a thread), not `async def`. |

### 3.6 Server Class Naming (Resolved)

The class formerly named `SyncServer` has been renamed to `AsyncServer`
as part of this change set. All imports, type hints, docstrings, and tests
have been updated. A backward-compatible alias `SyncServer = AsyncServer`
is retained in `mirix/server/server.py`.

---

## 4. Summary

The MIRIX application is async-native from the HTTP boundary through the
server, agents, service managers, ORM, database, Redis, Kafka, and
LLM/embedding clients. The only remaining sync touch-points are:

1. **LangFuse** -- sync SDK wrapped in `asyncio.to_thread`; low impact.
2. **Gmail OAuth** -- blocking by design; wrapped in `to_thread`; rare.
3. **Startup DDL** -- one-time `run_sync`; no runtime impact.
4. **Cleanup script** -- `asyncio.run()` in `__main__`; separate process.
5. **Pure CPU helpers** -- no I/O; `async def` would add overhead, not benefit.
6. **Server naming** -- `SyncServer` renamed to `AsyncServer`; alias kept.

None of these limit MIRIX's ability to scale request throughput or
concurrent users. The critical path is fully async.
