# MIRIX Walk-Through: Architecture, Runtime Flow, and Endpoint-to-Storage Trace

> Scope: This is a code-reading-derived companion to `docs/ARCHITECTURE.md` and
> `CLAUDE.md`. It is meant to be read top-to-bottom: the diagram and the
> startup/runtime sections give you the mental model; the endpoint catalog
> traces every public HTTP route down to the SQL row(s) it touches and the
> manipulations applied in between.

---

## 1. High-Level System Architecture

### 1.1 Component diagram

```
                                ┌──────────────────────────────────────────┐
                                │                CLIENTS                   │
                                │  - Dashboard (React/Vite, port 5173)     │
                                │  - MirixClient SDK (mirix-client PyPI)   │
                                │  - ECMS (imports mirix as a library)     │
                                └────────────────────┬─────────────────────┘
                                                     │  HTTPS
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI REST LAYER (port 8531)                          │
│                          mirix/server/rest_api.py                                │
│                                                                                  │
│  Middleware:   store_request_context  →  CORS  →  with_langfuse_tracing          │
│  Auth helpers: get_client_and_org()   /  get_client_from_jwt_or_api_key()        │
│                                                                                  │
│  Route groups:                                                                   │
│    /health         /agents/*          /agents/meta/initialize                    │
│    /memory/add     /memory/retrieve/* /memory/search*  /memory/raw/*             │
│    /memory/{episodic,semantic,procedural,resource,knowledge_vault}/{id}          │
│    /memory-sources/{id}[/messages]                                               │
│    /tools/*        /blocks/*          /config/{llm,embedding}                    │
│    /users/*        /organizations/*   /clients/*                                 │
│    /admin/auth/*   /admin/dashboard-clients                                      │
└────────────────────────────────────────────┬─────────────────────────────────────┘
                                             │  in-process calls
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       AsyncServer  (singleton, sync-from-init)                   │
│                       mirix/server/server.py :: AsyncServer                      │
│                                                                                  │
│  - Owns one instance of each Manager (services/*_manager.py)                     │
│  - load_agent()  →  builds the runtime Agent for an agent_id                     │
│  - _step() / send_messages() →  invokes Agent.step() with full provenance        │
│  - run_command(), construct_system_message(), list_{llm,embedding}_models(), …   │
└────────────────────────────────────────────┬─────────────────────────────────────┘
                                             │
            ┌────────────────────────────────┼─────────────────────────────────┐
            │                                │                                 │
            ▼                                ▼                                 ▼
  ┌───────────────────────┐      ┌──────────────────────────────┐  ┌─────────────────────┐
  │  Queue (Kafka or in-  │      │   Service Managers           │  │   Observability     │
  │  memory) async writer │      │   mirix/services/*_manager.py│  │   mirix/            │
  │  mirix/queue/         │      │                              │  │   observability/    │
  │   ├─ manager.py       │      │  block, agent, message,      │  │                     │
  │   ├─ worker.py        │      │  episodic_memory,            │  │  LangFuse traces +  │
  │   ├─ kafka_queue.py   │      │  semantic_memory,            │  │  skip spans         │
  │   ├─ memory_queue.py  │      │  procedural_memory,          │  │                     │
  │   └─ message_pb2.py   │      │  resource_memory,            │  │                     │
  │                       │      │  knowledge_vault,            │  └─────────────────────┘
  │  protobuf payload:    │      │  raw_memory,                 │
  │  QueueMessage         │      │  memory_source,              │
  └───────────┬───────────┘      │  memory_citation,            │
              │                  │  source_message,             │
              │ (consumed)       │  tool, client, user, org,    │
              ▼                  │  provider, step              │
  ┌──────────────────────┐       └──────────────┬───────────────┘
  │  QueueWorker (asyncio│                      │
  │  task) → server.     │                      │  SQLAlchemy 2.x async ORM
  │  send_messages(...)  │                      ▼
  │                      │       ┌──────────────────────────────┐
  │  reroutes back into  │       │      mirix/orm/*.py          │
  │  AsyncServer._step() │       │  (declarative tables w/      │
  └──────────┬───────────┘       │   OrganizationMixin, UserMixin│
             │                   │   AgentMixin via Base+        │
             │                   │   CommonSqlalchemyMetaMixins) │
             │                   └──────────────┬───────────────┘
             ▼                                  │
  ┌────────────────────────────────────────────────────────────────────────┐
  │                          Agent / MetaAgent                             │
  │                  mirix/agent/{agent,meta_agent}.py                     │
  │                                                                        │
  │   MetaAgent  ──┬─► CoreMemoryAgent          (persona / human blocks)   │
  │                ├─► EpisodicMemoryAgent      (timestamped events)       │
  │                ├─► SemanticMemoryAgent      (facts / concepts)         │
  │                ├─► ProceduralMemoryAgent    (how-tos / steps)          │
  │                ├─► ResourceMemoryAgent      (files / links / assets)   │
  │                ├─► KnowledgeVaultAgent      (sensitive facts)          │
  │                ├─► ReflexionAgent           (self-reflection)          │
  │                ├─► BackgroundAgent          (background tasks)         │
  │                └─► MetaMemoryAgent          (orchestrator agent type)  │
  │                                                                        │
  │   Plus a "Summary Agent" task dispatched via asyncio.create_task in    │
  │   parallel with sub-agents (LangFuse "Summary Agent" span; not a       │
  │   registered AgentType).                                               │
  └──────────────────────────────────┬─────────────────────────────────────┘
                                     │  LLMClient.create(llm_config)
                                     ▼
                        ┌────────────────────────────┐
                        │  mirix/llm_api/*           │
                        │  OpenAI / Anthropic /      │
                        │  Google / Azure /          │
                        │  Bedrock / Ollama / vLLM   │
                        └────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA & INFRASTRUCTURE                           │
│  PostgreSQL + pgvector  (5432)   — primary store, BM25, vector search    │
│  Redis Stack            (6379)   — cache (RedisJSON, RediSearch)         │
│  Kafka (aiokafka)               — async memory extraction queue          │
│  LangFuse                       — distributed tracing & spans            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer responsibility (canonical table)

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Client SDK | `mirix/client/remote_client.py` | Async HTTP wrapper (70+ methods) |
| REST API | `mirix/server/rest_api.py` | Auth, routing, request shaping |
| AsyncServer | `mirix/server/server.py` | Singleton orchestrator; owns managers |
| Queue | `mirix/queue/{manager,worker,kafka_queue,memory_queue}.py` | Async fan-out of `/memory/add` jobs |
| MetaAgent | `mirix/agent/meta_agent.py` | Coordinates the six memory sub-agents |
| Sub-Agents | `mirix/agent/*_memory_agent.py`, `reflexion_agent.py`, `background_agent.py` | LLM-driven extraction per memory type |
| Tool Sandbox | `mirix/services/tool_execution_sandbox.py` | Isolated execution for `USER_DEFINED` tools |
| Managers | `mirix/services/*_manager.py` | Business logic, DB access |
| Schemas | `mirix/schemas/*.py` | Pydantic request/response validation |
| ORM | `mirix/orm/*.py` | SQLAlchemy tables (`OrganizationMixin`, `UserMixin`, `AgentMixin`) |
| LLM clients | `mirix/llm_api/*.py` | OpenAI/Anthropic/Google/Azure/Bedrock/Ollama/vLLM |
| Observability | `mirix/observability/*.py` | LangFuse spans + skip span emission |

### 1.3 Memory types and provenance sidecar

| Type | ORM | Manager | Agent / Tool |
|------|-----|---------|--------------|
| Core | `mirix/orm/block.py` | `block_manager.py` | `core_memory_agent` |
| Episodic | `mirix/orm/episodic_memory.py` | `episodic_memory_manager.py` | `episodic_memory_agent` |
| Semantic | `mirix/orm/semantic_memory.py` | `semantic_memory_manager.py` | `semantic_memory_agent` |
| Procedural | `mirix/orm/procedural_memory.py` | `procedural_memory_manager.py` | `procedural_memory_agent` |
| Resource | `mirix/orm/resource_memory.py` | `resource_memory_manager.py` | `resource_memory_agent` |
| Knowledge Vault | `mirix/orm/knowledge_vault.py` | `knowledge_vault_manager.py` | `knowledge_vault_memory_agent` |
| Raw | `mirix/orm/raw_memory.py` | `raw_memory_manager.py` | (no agent — direct HTTP only) |

**Provenance sidecar** (VEPAGE-760), written alongside every memory:

| Table | ORM | Manager | Purpose |
|-------|-----|---------|---------|
| `memory_sources` | `mirix/orm/memory_source.py` | `memory_source_manager.py` | One row per `/memory/add`; holds `external_id`, `batch_hash`, `source_type/system/metadata`, `summary`, `processing_complete` |
| `memory_citations` | `mirix/orm/memory_citation.py` | `memory_citation_manager.py` | `(memory_type, memory_id) → memory_source_id`; unique on `(source, memory_type, memory_id)` |
| `source_messages` | `mirix/orm/source_message.py` | `source_message_manager.py` | Raw per-turn dicts for the derived path |

---

## 2. Step-by-Step: How MIRIX Works

### 2.1 Process startup

`mirix/server/rest_api.py::lifespan()` runs once when the FastAPI app boots
(it also runs when ECMS, Enterprise Context and Memory Service, imports
MIRIX-as-a-library and explicitly calls `initialize()`):

1. **`ensure_tables_created()`** — opens an async SQLAlchemy engine to PostgreSQL
   and runs `Base.metadata.create_all` (one of the five approved sync touch-points,
   wrapped in `asyncio.to_thread`).
2. **Redis async client** — pings, creates RediSearch indexes, logs stats.
3. **`get_server()`** — creates the `AsyncServer` singleton:
   - Builds every Manager: `agent_manager`, `block_manager`, `client_manager`,
     `cloud_file_mapping_manager`, `episodic_memory_manager`,
     `knowledge_vault_manager`, `message_manager`, `organization_manager`,
     `procedural_memory_manager`, `provider_manager`, `raw_memory_manager`,
     `resource_memory_manager`, `semantic_memory_manager`, `step_manager`,
     `tool_manager`, `user_manager`.
   - `await server.ensure_defaults()` — guarantees a default organization /
     admin client / admin user exist.
   - Loads provider API keys from `mirix/settings.py` (Pydantic `BaseSettings`,
     env prefix `MIRIX_`).
4. **`initialize_queue(server)`** — creates the `QueueManager` singleton,
   instantiates the queue backend (`MemoryQueue`, `PartitionedMemoryQueue`,
   or `KafkaQueue` depending on `QUEUE_TYPE`), and starts N `QueueWorker`s as
   `asyncio.Task`s. Each worker holds a reference to `AsyncServer`.
5. **`initialize_langfuse()`** — async observability bootstrap (best-effort).

On shutdown the inverse runs: flush LangFuse → stop queue workers → close
SQLAlchemy engine.

### 2.2 Meta-agent bootstrap (one-time per client)

```
Client → POST /agents/meta/initialize
       → AgentManager.list_agents(actor=client)
       → 0 existing? AgentManager.create_meta_agent(CreateMetaAgent)
                       creates "meta_memory_agent" parent +
                       child agents (episodic, semantic, procedural,
                       resource, knowledge_vault, core, reflexion, background)
       → 1 existing & update_agents=True? update_meta_agent(...)
       → returns AgentState of the meta agent (or null for read-only clients)
```

ECMS (Enterprise Context and Memory Service) calls this once; subsequent saves
reuse the returned `meta_agent_id`.

### 2.3 Write path — derived (LLM-driven memory extraction)

```
[1] HTTP                            POST /memory/add
    ───────────────────────────────────────────────────────────────
    Body: { meta_agent_id, messages[], user_id?, filter_tags?,
            block_filter_tags?, block_filter_tags_update_mode?,
            external_id?, external_thread_id?, source_type?,
            source_system?, source_metadata?, summary?, summarize?,
            occurred_at?, chaining?, verbose?, use_cache?,
            direct_writes? }
    Mutual exclusion: when `direct_writes` is set, `messages`
    must be empty and `summary` / `summarize` must be unset
    (validated by `AddMemoryRequest._validate_direct_writes_exclusivity`).

[2] REST handler (rest_api.py :: add_memory)
    ───────────────────────────────────────────────────────────────
    a. get_client_and_org() → resolves Client
    b. ensure meta agent exists (404 otherwise)
    c. resolve user_id (admin user if not provided)
    d. FLATTEN messages → packed MessageCreate with [USER]/[ASSISTANT]
       text markers (per-message identity is lost — that's why we also
       carry `original_messages` separately)
    e. inject filter_tags["scope"] = client.write_scope (403 if missing)
    f. pre-generate memory_source_id = "src-<uuid4>"
    g. queue_util.put_messages(...) builds a QueueMessage protobuf with:
         - input_messages = [packed MessageCreate]
         - source_messages = original per-turn dicts
         - direct_writes = None
         - all source provenance fields
         - LangFuse trace context attached
       and queue_manager.save(queue_msg)
    h. RETURN 200 { success, status:"queued", memory_source_id, ... }
       (the actual processing happens async)

[3] QueueWorker._consume_loop (mirix/queue/worker.py)
    ───────────────────────────────────────────────────────────────
    Pulls a QueueMessage. Restores LangFuse trace context.
    Resolves actor (Client) and user (auto-creates user if missing).
    Opens "Meta Agent" LangFuse span.
    Calls server.send_messages(actor, agent_id=meta_agent_id, ...).

[4] AsyncServer.send_messages → _step → Agent.step
    ───────────────────────────────────────────────────────────────
    load_agent() builds a MetaMemoryAgent runtime instance.
    Attaches provenance fields to the agent instance:
        memory_source_id, external_id, external_thread_id,
        source_type/system/metadata, summary, summarize,
        source_messages, direct_writes (None here)

[5] Agent.step (mirix/agent/agent.py)
    ───────────────────────────────────────────────────────────────
    Branch: agent.is_type(meta_memory_agent) && memory_source_id set
      → _persist_memory_source(memory_source_id, raw_input_messages):
          - normalize messages
          - compute external_id (derive from external_message_ids if all set)
              else compute batch_hash = SHA-256 of (thread_id, occurred_at, messages)
          - memory_source_manager.create(...) — INSERT ... ON CONFLICT DO NOTHING
                If conflict (L1 dedup): set _source_deduped=True
          - source_message_manager.bulk_insert(...) — DB UNIQUE on
              (memory_source_id, external_message_id) (L1)

      → If _source_deduped → emit "Idempotency Skip: source deduped" span; return
      → If memory_source.processing_complete → "processing complete" skip (L2)

      → If summarize=True and no client summary:
          summary_task = asyncio.create_task(_generate_source_summary_traced())
          (runs LLM in parallel; result later written to memory_sources.summary)

    [Direct write branch is skipped for derived path]

    [Topic extraction] _extract_topics_from_messages → kwargs["topics"]

    [Chaining loop] while True:
        step_response = await inner_step(...)
          ├─ Build LLM messages (system, retained, current)
          ├─ Call LLMClient.send_llm_request(...) → tool calls
          ├─ Execute each tool call (memory tool / send_message /
          │      finish_memory_update / continue_chaining)
          ├─ Persist Step audit row (StepManager.create — write-only)
          └─ Returns continue_chaining + function_failed + usage

        Heartbeat injection:
          - continue_chaining → "[System] returning control" user message
          - function_failed   → "[System] function call failed"
          - max_chaining_steps reached → "[System] please call finish_memory_update"
        Break when:
          - chaining=False and last call succeeded, OR
          - finish_memory_update was called (memory agents terminate on this),
          - max_chaining_steps exceeded.

        ECMS (Enterprise Context and Memory Service)-deployment note:
        CHAINING_FOR_META_AGENT=false means this loop runs exactly once.

    [Sub-agent dispatch happens inside meta_memory_agent's tool calls]
    The `trigger_memory_update(memory_types=[...])` tool fans out to
    each owning sub-agent concurrently via `asyncio.gather`. Each
    sub-agent runs the same Agent.step loop, but their tools write
    rows to their memory table AND call _write_citation() to insert a
    row in memory_citations referencing this memory_source_id
    (L3 dedup on (source, memory_type, memory_id)).

    Owning sub-agents (the one whose memory type is being updated)
    consult `_fetch_recent_indexing_lag_window(...)` →
    `mirix/services/hybrid_search_helper.py::fetch_and_dedup_candidates`
    to get a `recent` bucket (rows in the last
    `set_hybrid_window_seconds` written via the relational provider but
    not yet visible in the search index) alongside the `relevant`
    bucket from the search provider. The buckets are surfaced as
    separate sections in the system prompt so the LLM can detect
    near-duplicate writes that just landed.

    Sub-agent fan-out error handling
    (`mirix/functions/function_sets/memory_tools.py::
    _decide_step_outcome_from_sub_agent_results`):
    - If any sub-agent failure classifies as Permanent
      (`error_policy.classify`), that exception is re-raised unwrapped
      so the outer policy wrapper sees the original `LLMError` type.
    - Otherwise the first Transient exception is re-raised so the
      whole-step retry can succeed.
    Wrapping the exception would erase the LLM error type and break
    the 422 cascade fix (VEPAGE-1091).

    [After the loop]
    - await summary_task (raises → Kafka redelivery → safe full retry;
      VEPAGE-1157 logs the cause chain via
      `mirix/queue/error_policy.py::format_exc_chain`)
    - memory_source_manager.mark_processing_complete(memory_source_id)
       → flips memory_sources.processing_complete = True
    - return MirixUsageStatistics

[6] Postgres state after a successful save
    ───────────────────────────────────────────────────────────────
    memory_sources               1 row  (processing_complete=True)
    source_messages              N rows (one per conversation turn)
    episodic_memory              0..M rows (with embeddings)
    semantic_memory              0..M rows
    procedural_memory            0..M rows
    resource_memory              0..M rows
    knowledge_vault              0..M rows
    block (core)                 0..M rows updated
    memory_citations             Σ rows  (one per memory write, linked back)
    steps                        N rows  (one per inner_step LLM call)
    messages                     0..N rows (only if MetaAgent retention > 0)
```

### 2.4 Write path — direct (caller-authored, no LLM)

```
POST /memory/add
    body.direct_writes = [
       {"memory_type": "episodic",   "payload": {"items": [...]}},
       {"memory_type": "semantic",   "payload": {"items": [...]}},
       ...
    ]
    body.messages = []                  (validator requires this)
    body.summary  = None                (validator)
    body.summarize = False              (validator)

→ Same enqueue path. QueueMessage.direct_writes is set; source_messages is
  empty (placeholder MessageCreate makes the queue happy).

→ Worker → AsyncServer._step → Agent.step (MetaMemoryAgent)

→ Agent.step:
   1. _persist_memory_source — same as derived (still gets the memory_sources
      row), but source_messages stays empty because direct_writes is set.
   2. Direct-write branch runs BEFORE the chaining loop:
        _apply_direct_writes_traced():
          for write in direct_writes:
              handler = DIRECT_WRITE_HANDLERS[write["memory_type"]]
              await handler(self, **write["payload"])
          Each handler IS the corresponding memory-tool function
          (episodic_memory_insert, semantic_memory_insert, ...).
          The tool writes the memory row via its manager AND calls
          _write_citation() exactly like the LLM path.
   3. mark_processing_complete(memory_source_id)
   4. return — no inner_step / no LLM call

Mutual exclusion: `messages`, `summary`, `summarize`, and `direct_writes`
cannot coexist on the same request (rejected by `AddMemoryRequest`'s
`_validate_direct_writes_exclusivity` model validator at HTTP time).
```

### 2.5 Read path — retrieval

```
POST /memory/retrieve/conversation                       (LLM topic extraction)
   ├─ extract_topics_with_local_model() (optional Ollama)
   ├─ extract_topics_and_temporal_info() (default LLM)
   ├─ parse_temporal_expression() → start_date/end_date
   └─ retrieve_memories_by_keywords(...)
         ├─ episodic_memory_manager.list_episodic_memory(..., search_method="bm25")
         ├─ semantic_memory_manager.list_semantic_items(...)
         ├─ resource_memory_manager.list_resources(...)
         ├─ procedural_memory_manager.list_procedures(...)
         ├─ knowledge_vault_manager.list_knowledge(...)
         └─ block_manager.get_blocks(... any_scopes=client.read_scopes)
   └─ if include_citations: _attach_citations_to_memories_dict()

GET /memory/retrieve/topic                               (skip LLM, search directly)
   └─ retrieve_memories_by_keywords(key_words=topic, ...)

GET /memory/search                                       (one or all memory types)
   ├─ _precompute_embedding_for_search() (once, padded for episodic)
   ├─ asyncio.gather(search_episodic, search_resource,
   │                 search_procedural, search_knowledge,
   │                 search_semantic [+ search_core if requested])
   ├─ Each search uses BM25 (pg_bm25) or pgvector cosine distance
   ├─ if include_citations: _attach_citations_to_results()
   └─ Merge & truncate to `limit`

GET /memory/search_all_users                             (admin/dashboard variant)
   └─ Same as /memory/search but iterates across users in the org

GET /memory-sources/{id}                                 (provenance metadata)
   └─ memory_source_manager.get_by_id(...) + scope filter against read_scopes

GET /memory-sources/{id}/messages                        (raw turns, paginated)
   └─ source_message_manager.get_messages_by_source_id(..., limit, cursor)
```

### 2.6 Authentication & access control

- **API key**: `X-API-Key: <key>` — checked by middleware against
  `client_api_keys` table (hashed at rest in `mirix/orm/client_api_key.py`,
  helpers in `mirix/security/api_keys.py`).
- **JWT (dashboard)**: `Authorization: Bearer <token>` — signed/verified by
  `ClientAuthManager` in `mirix/services/admin_user_manager.py`.
- **Headers** `x-client-id` / `x-org-id` — explicit-tenant fallback for
  service-to-service callers (resolved via `get_client_and_org`).
- **Access predicate**: every Manager call goes through
  `apply_access_predicate(actor=client, ...)` which filters on
  `organization_id` and `_created_by_id` (and `is_deleted=False`).
- **Read scope vs write scope**:
  - `client.write_scope` (single string) is stamped onto
    `filter_tags->>'scope'` on every memory row at write time.
  - `client.read_scopes` (list of strings) gates every read — the manager
    appends `filter_tags->>'scope' = ANY(:read_scopes)` to its query.

### 2.7 Idempotency, temporal guard, and tracing

- **L1 source-level**: `(client_id, user_id, external_id)` or
  `batch_hash` partial unique index on `memory_sources`. Replay returns the
  existing `memory_source_id`, no re-ingest.
- **L2 processing short-circuit**: `memory_source.processing_complete=True`
  → meta-agent skips everything before sub-agent dispatch.
- **L3 citation-level**: `(memory_id, memory_source_id)` unique inside
  `_write_citation()` — same memory never cited twice by the same source.
- **Temporal guard**: For derived (non-episodic) writes, `_should_update_memory`
  computes `MAX(occurred_at)` across existing citations for the
  `(user, memory_type, external_thread_id)` tuple. Older incoming
  `occurred_at` → drop. Episodic appends only, so unaffected.
- Every skip path emits a LangFuse span via
  `mirix/observability/skip_spans.py::emit_idempotency_skip_span`.

### 2.8 Error policy and bounded retries

The whole save-flow runs under
`mirix/queue/error_policy.py::process_with_policy`, which classifies
exceptions into two buckets and decides ack-vs-redeliver behavior at
the queue boundary (see `mirix/queue/__init__.py::process_external_message`).

| Bucket | Examples | Behavior |
|--------|----------|----------|
| **Permanent** | `LLMUnprocessableEntityError` (422), `LLMBadRequestError` (400), `LLMAuthenticationError` (401), `LLMPermissionDeniedError` (403) | First occurrence propagates immediately. The `_mark_permanent` callback flips `memory_source.processing_complete=True` so a redelivery hits the L2 short-circuit and never re-fires the LLM. The consumer ACKs the message. |
| **Transient** | `LLMRateLimitError` (429), `LLMServerError` (5xx), `LLMConnectionError` | Retried inside `_get_ai_reply` (inline LLM retry, settings: `MIRIX_LLM_INLINE_RETRY_*`) and again at whole-step level (settings: `MIRIX_WHOLE_STEP_RETRY_*`) with exponential backoff + full jitter. Exhaustion re-raises so the consumer redelivers (Kafka path); the L1/L2 idempotency layers make the second attempt a safe full retry. |

Notable contracts (all are tested in `tests/test_error_policy.py` and
`tests/test_agent_get_ai_reply_retry.py`):

- `classify()` walks `exc.__cause__` so an `LLMUnprocessableEntityError`
  wrapped in `RuntimeError(...)` from `... from llm_err` still classifies
  as Permanent (VEPAGE-1091 fix; otherwise the wrapper turns 422 into a
  Transient retry cascade).
- Sub-agent fan-out (`trigger_memory_update`) re-raises the original
  exception unwrapped so the cause chain reaches the policy wrapper
  with its real type.
- Unknown exception classes default to Transient with a one-shot warning
  log so the explicit bucket lists can be updated.
- `format_exc_chain(exc)` renders the full cause chain
  (`Outer: msg <- Inner: msg`) on a single log line — used by VEPAGE-1157
  cause-chain instrumentation in `agent.py` and `sqlalchemy_base.py` so
  wrapped-exception cascades stay debuggable in Splunk.

---

## 3. Endpoint-by-Endpoint Walk-Through

All endpoints are declared on either `router` (most) or `app` (one — the
`/agents/{id}/messages` route). `app.include_router(router)` is the last
line of `rest_api.py`. The decorator `@with_langfuse_tracing` is attached
to the agentic routes — it pulls the current FastAPI `Request` from a
`ContextVar` so every traced route gets a trace ID without changing its
signature.

Legend: **A→B** means handler A calls method B; the column under each
section shows the SQL tables touched (R=read, W=write).

### 3.1 Health

| Method | Path | Handler |
|--------|------|---------|
| GET | `/health` | `health_check` |

- Logic: returns `{"status": "healthy", "service": "mirix-api"}`. No I/O.
- Storage: **none**.

### 3.2 Agents

| Method | Path | Handler | Manager call(s) | Tables |
|--------|------|---------|-----------------|--------|
| GET    | `/agents` | `list_agents` | `agent_manager.list_agents` | R `agents`, `tools_agents`, `block` (via lazy=joined relations) |
| POST   | `/agents` | `create_agent` | `block_manager.create_or_update_block` per memory block, then `server.create_agent(CreateAgent, client)` → `agent_manager.create_agent` | W `agents`, `block`, `tools_agents` |
| GET    | `/agents/{id}` | `get_agent` | `agent_manager.get_agent_by_id` | R `agents` |
| DELETE | `/agents/{id}` | `delete_agent` | `agent_manager.delete_agent` | W `agents` (soft delete: `is_deleted=True`) |
| PATCH  | `/agents/{id}` | `update_agent` | (not yet implemented — returns 501) | — |
| PATCH  | `/agents/{id}/system` | `update_agent_system_prompt` | `agent_manager.update_system_prompt` | W `agents` (sets `system` column) + Redis cache invalidate |
| PATCH  | `/agents/by-name/{name}/system` | `update_agent_system_prompt_by_name` | Lists agents, resolves "episodic"/"semantic"/… short names by stripping `meta_memory_agent_` prefix, then same as above | R `agents`, W `agents` |
| POST   | `/agents/meta/initialize` | `initialize_meta_agent` | `agent_manager.list_agents` → either `create_meta_agent(CreateMetaAgent)` (creates meta + 8 children) or `update_meta_agent(UpdateMetaAgent)` (when `update_agents=True`) | W `agents` (×9), `tools_agents`, `block` (template-derived) |
| POST   | `/agents/{id}/messages` | `send_message_to_agent` | Wraps message in `MessageCreate`, calls `queue_util.put_messages(...)` (chaining=True) | enqueue only (eventual W) |

Notes on `POST /agents/meta/initialize`:
- Reads `request.config.{llm_config, embedding_config, meta_agent_config.agents,
  meta_agent_config.system_prompts}`.
- `assert len(existing_meta_agents) <= 1` — one meta per client.
- Read-only clients (`client.write_scope is None`) get `None` returned.

### 3.3 Memory — write/process

#### `POST /memory/add` — **the central write endpoint**

| Stage | What happens |
|-------|--------------|
| Auth | `get_client_and_org()`; if client is the well-known `ClientManager.DEFAULT_CLIENT_ID` and missing, auto-create via `create_default_client(org_id)`. |
| Resolve agent | `agent_manager.get_agent_by_id(meta_agent_id, client)` (404 if missing/inaccessible). |
| Resolve user | If `user_id` absent → `ClientAuthManager.get_admin_user_id_for_client(client.id)`. |
| Shape messages | If list looks like `[{role, content}, ...]`, flatten to one content list with `[USER]`/`[ASSISTANT]` markers, then `convert_message_to_mirix_message(...)` packs into a single `MessageCreate`. **Preserves the original list as `original_messages`** for source_message persistence. |
| Filter tags | Reject if `block_filter_tags` non-dict; pop `scope` from caller-supplied `block_filter_tags`; inject `filter_tags["scope"] = client.write_scope` (403 if missing). Validate `block_filter_tags_update_mode in {"merge","replace"}`. |
| Source ID | Generates `memory_source_id = f"src-{uuid.uuid4()}"`. |
| Direct-write input validation | `DirectWriteInput._validate_payload_for_memory_type` coerces every item through the per-type Pydantic schema (`EpisodicEventForLLM`, `SemanticMemoryItemBase`, `ProceduralMemoryItemBase`, `ResourceMemoryItemBase`, `KnowledgeVaultItemBase`) — `core` is **not** a supported direct-write type. Missing or wrong-typed fields surface as 422 ValidationError at the HTTP boundary. |
| Validator | `AddMemoryRequest._validate_direct_writes_exclusivity` → 422 if both `direct_writes` and (`messages` or `summary` or `summarize`) provided. |
| Enqueue | `put_messages(actor=client, agent_id=meta_agent_id, input_messages, chaining, user_id, ..., memory_source_id, external_id, external_thread_id, source_type, source_system, source_metadata, summary, summarize, source_messages=original_messages, direct_writes=...)`. |
| Response | `{ success, message:"Memory queued for processing", status:"queued", agent_id, message_count, memory_source_id }`. **HTTP returns before the memory is processed.** |

Tables eventually written: `memory_sources`, `source_messages`, one or more
of (`episodic_memory`, `semantic_memory`, `procedural_memory`,
`resource_memory`, `knowledge_vault`, `block`), `memory_citations`, `steps`,
optionally `messages`. (See §2.3.)

#### `POST /memory/retrieve/conversation`

| Stage | What happens |
|-------|--------------|
| Auth | `get_client_and_org()` → `Client`. |
| Resolve agent | Picks `all_agents[0]` for the client (for embedding/LLM config). |
| Topic extraction | If `local_model_for_retrieval` is set, tries Ollama via `extract_topics_with_local_model`; otherwise `extract_topics_and_temporal_info` (LLM call). |
| Temporal | Explicit `start_date/end_date` from request win; else parse extracted `temporal_expression` against the user's timezone. |
| Retrieve | `retrieve_memories_by_keywords(...)` (BM25, default `limit=10` per type). |
| Citations | If `include_citations=True`, `_attach_citations_to_memories_dict(...)` → batched fetch via `MemoryCitationManager.get_citations_for_memories(...)`. |
| Response | `{ success, topics, temporal_expression, date_range, memories }`. |

Tables read: `episodic_memory` (BM25 on `details`), `semantic_memory`
(BM25 on `details`), `resource_memory` (BM25 on `summary`),
`procedural_memory` (BM25 on `summary`), `knowledge_vault` (BM25 on
`caption`), `block` (filtered by `read_scopes`), and `memory_citations` +
`memory_sources` if citations were requested.

#### `GET /memory/retrieve/topic`

| Stage | What happens |
|-------|--------------|
| Auth | Same as above. |
| Filter | `filter_tags` accepted as a JSON string and parsed. |
| Retrieve | Skips LLM topic extraction; just calls `retrieve_memories_by_keywords(key_words=topic, ...)` over BM25. |
| Citations | Optional `include_citations`. |

#### `GET /memory/search`

| Stage | What happens |
|-------|--------------|
| Auth | Supports BOTH `Authorization: Bearer <jwt>` and `X-API-Key`/header pair. |
| Validate | Reject `embedding` for `resource.content` and `knowledge_vault.secret_value` (not supported); force `search_field="null"` when `memory_type="all"`. |
| Embedding | `_precompute_embedding_for_search(...)` runs `embedding_model(agent_state.embedding_config).get_text_embedding(query)` once and pads to `MAX_EMBEDDING_DIM` for episodic. |
| Search | For `memory_type="all"`, runs the six per-type async search functions concurrently with `asyncio.gather`. Each calls its manager's `list_*` (BM25 or pgvector). |
| Filter | `filter_tags` JSON-parsed; `scopes = client.read_scopes`; `start_date`/`end_date` applied to episodic only. |
| Citations | Optional `include_citations` (batched citation fetch). |

#### `GET /memory/search_all_users`

Same as `/memory/search` but iterates users within the caller's org —
intended for the dashboard. Honors `read_scopes` per query.

#### `GET /memory/components` / `GET /memory/fields`

Static introspection endpoints returning the supported memory component
labels and per-component searchable fields used by the dashboard search UI.
Pure-Python; no DB I/O.

### 3.4 Memory — single-row mutation endpoints

All accept either JWT or API key (via `get_client_from_jwt_or_api_key`) and
fall back to the admin user when `user_id` is omitted.

| Method | Path | Manager call | Storage manipulation |
|--------|------|--------------|----------------------|
| PATCH  | `/memory/episodic/{id}` | `episodic_memory_manager.update_event` | UPDATE `episodic_memory` set `summary`/`details`; bumps `last_modify={timestamp, operation:"updated"}` JSON. |
| DELETE | `/memory/episodic/{id}` | `episodic_memory_manager.delete_event_by_id` | Hard DELETE from `episodic_memory` (cascade to `memory_citations` rows via FK is **not** applied — citations remain by design as audit trail). |
| PATCH  | `/memory/semantic/{id}` | `semantic_memory_manager.update_item` | UPDATE `semantic_memory.{name,summary,details}`. |
| DELETE | `/memory/semantic/{id}` | `semantic_memory_manager.delete_semantic_item_by_id` | DELETE. |
| PATCH  | `/memory/procedural/{id}` | `procedural_memory_manager.update_item` | UPDATE `procedural_memory.{summary,steps}` (steps stored as JSON array). |
| DELETE | `/memory/procedural/{id}` | `procedural_memory_manager.delete_procedure_by_id` | DELETE. |
| PATCH  | `/memory/resource/{id}` | `resource_memory_manager.update_item` | UPDATE `resource_memory.{title,summary,content}`. |
| DELETE | `/memory/resource/{id}` | `resource_memory_manager.delete_resource_by_id` | DELETE. |
| DELETE | `/memory/knowledge_vault/{id}` | `knowledge_vault_manager.delete_knowledge_by_id` | DELETE. |

Each manager update path:
1. Fetches the row with access predicate (org + scope guard).
2. Applies the diff.
3. Re-embeds fields that changed (via the agent's `embedding_config`).
4. Writes the row + invalidates the Redis cache key for that memory.
5. Does **not** touch citations or memory_sources.

### 3.5 Raw Memory (ephemeral 14-day store)

Raw memory is task-context scratch space — no agent involvement, hard
14-day TTL via a nightly cleanup job (`mirix/jobs/cleanup_raw_memories.py`).

| Method | Path | Handler → Manager | Storage |
|--------|------|-------------------|---------|
| POST   | `/memory/raw` | `create_raw_memory_handler` → `raw_memory_manager.create_raw_memory` | INSERT `raw_memory` (with `context_embedding` if model available); Redis cache write |
| GET    | `/memory/raw/{id}` | `get_raw_memory_handler` → `raw_memory_manager.get_raw_memory_by_id` | R `raw_memory` (Redis-cached) |
| PATCH  | `/memory/raw/{id}` | `update_raw_memory_handler` → `raw_memory_manager.update_raw_memory` | UPDATE `raw_memory.{context,filter_tags}`; supports `context_update_type=append|replace` and `tags_update_type=merge|replace`; re-embed if context changed |
| DELETE | `/memory/raw/{id}` | `delete_raw_memory_handler` → `raw_memory_manager.delete_raw_memory` | HARD DELETE; Redis cache delete |
| POST   | `/memory/search_raw` | `search_raw_memory_handler` → `raw_memory_manager.search_raw_memories` | R `raw_memory` with `filter_tags` + scopes + `time_range` ({created_at,occurred_at,updated_at}_{gte,lte}) + cursor-based pagination + sort |
| POST   | `/memory/raw/cleanup` | `cleanup_raw_memories_handler` → `mirix/jobs/cleanup_raw_memories.delete_stale_raw_memories(days_threshold)` | Bulk DELETE where `updated_at < now() - threshold` (default 14d, 1–365). |

Request validation highlights:
- `time_range` datetimes are converted to UTC and stripped of timezone (DB
  stores naive datetimes assumed UTC).
- `sort` must be one of `{updated_at, -updated_at, created_at, -created_at,
  occurred_at, -occurred_at}`.
- Client must have an `organization_id`.

### 3.6 Memory Sources (provenance / progressive disclosure)

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET | `/memory-sources/{id}` | `MemorySourceManager.get_by_id` | R `memory_sources`; access check: `source.filter_tags->>'scope' ∈ client.read_scopes`, plus optional `user_id` match. |
| GET | `/memory-sources/{id}/messages` | `MemorySourceManager.get_by_id` for scope check, then `SourceMessageManager.get_messages_by_source_id(memory_source_id, limit, cursor)` | R `source_messages` ordered by `occurred_at` then `id`. |

Both 404 on missing scope rather than 403 to avoid leaking source existence.

### 3.7 Tools

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/tools` | `tool_manager.list_tools` | R `tools` filtered by client org |
| GET    | `/tools/{id}` | `tool_manager.get_tool_by_id` | R `tools` |
| POST   | `/tools` | `tool_manager.create_tool` | INSERT `tools` (UPSERT on `name` per org) |
| DELETE | `/tools/{id}` | `tool_manager.delete_tool_by_id` | Soft DELETE `tools` |

User-defined tools (Pydantic `Tool` with `tool_type=USER_DEFINED`) execute
inside `ToolExecutionSandbox` at agent runtime (`mirix/services/tool_execution_sandbox.py`).
Built-in `MIRIX_CORE`/`MIRIX_MEMORY_CORE`/`MIRIX_EXTRA` tools resolve via
`get_function_from_module(...)`; MCP tools pass arguments through unchanged.

### 3.8 Blocks (Core Memory)

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/blocks` | `block_manager.get_blocks(user, label, any_scopes=client.read_scopes, auto_create_from_default=False)` | R `block` filtered by scope |
| GET    | `/blocks/{id}` | `block_manager.get_block_by_id` | R `block` |
| POST   | `/blocks` | `block_manager.create_or_update_block` | UPSERT `block`; stamps `filter_tags->>'scope'` from `client.write_scope` |
| DELETE | `/blocks/{id}` | `block_manager.delete_block` | DELETE `block` (cascade to junction `blocks_agents` if present) |

The `core_memory_agent` reads blocks at the start of every `step()` via
`block_manager.get_blocks(user, any_scopes=self._block_scopes,
filter_tags_set_on_create=self.block_filter_tags)`. New blocks are created
from a default template when missing; existing blocks get
`filter_tags` merged/replaced based on `block_filter_tags_update_mode`.

### 3.9 Configuration

| Method | Path | AsyncServer call | Storage |
|--------|------|-------------------|---------|
| GET | `/config/llm` | `list_llm_models()` | Reads providers from `provider_manager` (DB) and `mirix/settings.py` env-backed defaults |
| GET | `/config/embedding` | `list_embedding_models()` | Same — embedding handles only |

### 3.10 Organizations

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/organizations` | `organization_manager.list_organizations(cursor, limit)` | R `organizations` |
| POST   | `/organizations` | `organization_manager.create_organization(Organization(name=...))` | INSERT `organizations` (id=`org-<uuid4hex8>`) |
| GET    | `/organizations/{id}` | `organization_manager.get_organization_by_id` → fallback `get_organization_or_default` | R `organizations` (returns default if missing) |
| POST   | `/organizations/create_or_get` | Same: try-get-then-create | R+W `organizations` |

### 3.11 Users

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/users` | `user_manager.list_users(cursor, limit, organization_id)` (admin JWT required) | R `users` |
| GET    | `/users/{id}` | `user_manager.get_user_by_id` | R `users` |
| POST   | `/users/create_or_get` | Try-get; on miss, `user_manager.create_user(PydanticUser(...))`. Accepts JWT or API key. | R+W `users` |
| DELETE | `/users/{id}` | `user_manager.delete_user_by_id` (**soft delete cascade**) | W `users.is_deleted=True` and same on `episodic_memory`, `semantic_memory`, `procedural_memory`, `resource_memory`, `knowledge_vault`, `messages`, `block` (filtered out of all subsequent reads) |
| DELETE | `/users/{id}/memories` | `user_manager.delete_memories_by_user_id` (**hard delete**) | DELETE rows in all six memory tables + `messages` + `block` for that user; user record preserved |

### 3.12 Clients

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/clients` | `client_manager.list_clients(cursor, limit)` (admin JWT) | R `clients` |
| GET    | `/clients/{id}` | `client_manager.get_client_by_id` (admin JWT) | R `clients` |
| POST   | `/clients/create_or_get` | Try-get; on miss, `client_manager.create_client(Client(...))`. `fail_if_exists=True` returns 409 on duplicate. | R+W `clients` |
| PATCH  | `/clients/{id}` | `client_manager.update_client(ClientUpdate)` | UPDATE `clients` |
| DELETE | `/clients/{id}` | `client_manager.delete_client_by_id` (**soft delete cascade**) | W `clients.is_deleted=True` and same on associated `agents`, `tools`, `block`. Memory records remain but are filtered via `client.is_deleted`. |
| DELETE | `/clients/{id}/memories` | `client_manager.delete_memories_by_client_id` (**hard delete**) | DELETE in `episodic_memory`, `semantic_memory`, `procedural_memory`, `resource_memory`, `knowledge_vault`, `messages`, `block` for the client. Preserves client + agents + tools. |

### 3.13 Client API Keys

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| POST   | `/clients/{id}/api-keys` | `generate_api_key()` (raw 32-byte key) → `client_manager.create_client_api_key(client_id, api_key, name, permission, user_id)` | INSERT `client_api_keys` storing **hashed** key; returns raw key once. |
| GET    | `/clients/{id}/api-keys` | `client_manager.list_client_api_keys(client_id)` | R `client_api_keys` (metadata only — never returns raw key). |
| DELETE | `/clients/{id}/api-keys/{api_key_id}` | `client_manager.delete_client_api_key(api_key_id)` | HARD DELETE `client_api_keys`; key becomes unusable immediately. |

Permission levels (`all`, `restricted`, `read_only`) are recorded on the
key and enforced inside `get_client_from_jwt_or_api_key()`. All three
endpoints require an admin JWT.

### 3.14 Admin / Dashboard Auth

| Method | Path | Manager call | Storage |
|--------|------|--------------|---------|
| GET    | `/admin/auth/check-setup` | `ClientAuthManager.is_first_dashboard_user()` | R `clients` (count of dashboard-enabled clients) |
| POST   | `/admin/auth/register` | `ClientAuthManager.register_client_for_dashboard(name, email, password, write_scope="admin")` then `create_access_token(client)` | INSERT `clients` (with `password_hash`, `email`, `write_scope="admin"`); returns JWT (24h) |
| POST   | `/admin/auth/login` | `ClientAuthManager.authenticate(email, password)` returns `(client, token, status)` | R `clients` by email; bcrypt verify; on success bump `last_login`; returns JWT |
| GET    | `/admin/auth/me` | `ClientAuthManager.get_client_by_id(payload["sub"])` | R `clients` |
| POST   | `/admin/auth/change-password` | `ClientAuthManager.change_password(client_id, current, new)` | UPDATE `clients.password_hash` (after bcrypt verify of current) |
| GET    | `/admin/dashboard-clients` | `ClientAuthManager.list_dashboard_clients(cursor, limit)` (requires `scope="admin"` in JWT) | R `clients` where `password_hash IS NOT NULL` |

### 3.15 Quick reference: who calls whom

```
HTTP route                                  ──►  Manager(s)                                  ──►  Tables (R/W)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
POST /memory/add                             queue_util.put_messages (eventual:                memory_sources W
                                              memory_source_manager, source_message_manager,    source_messages W
                                              episodic/semantic/procedural/resource/             (per-memory) W
                                              knowledge_vault/block managers,                    memory_citations W
                                              memory_citation_manager, step_manager,             steps W
                                              message_manager)                                   messages W (if retention)
POST /memory/retrieve/conversation           LLM(topics) + retrieve_memories_by_keywords →     episodic/semantic/.../block R
                                              (+ memory_citation_manager if include_citations)   memory_citations R, memory_sources R
GET  /memory/retrieve/topic                  retrieve_memories_by_keywords                     same as above (no LLM)
GET  /memory/search                          asyncio.gather over 5 list_* + optional core      same as above
GET  /memory/search_all_users                Same but iterates users                           same as above
GET  /memory/{components,fields}             —                                                 none
PATCH/DELETE /memory/<type>/{id}             *_manager.update_item / delete_*                  <type> W
POST/GET/PATCH/DELETE /memory/raw[/{id}]     raw_memory_manager.*                              raw_memory R/W + Redis
POST /memory/search_raw                      raw_memory_manager.search_raw_memories            raw_memory R
POST /memory/raw/cleanup                     jobs.cleanup_raw_memories                         raw_memory W (bulk DELETE)
GET  /memory-sources/{id}[/messages]         memory_source_manager.get_by_id                   memory_sources R
                                              source_message_manager.get_messages_by_source_id  source_messages R
POST /agents/meta/initialize                 agent_manager.create_meta_agent /                 agents W (×9), tools_agents W, block W
                                              update_meta_agent
GET/POST/GET/DELETE /agents[/{id}]           agent_manager.list/create/get/delete              agents R/W, tools_agents W, block W
PATCH /agents/{id}/system                    agent_manager.update_system_prompt                agents W
POST /agents/{id}/messages                   queue_util.put_messages                           enqueue only
GET/POST/GET/DELETE /tools[/{id}]            tool_manager.*                                    tools R/W
GET/POST/GET/DELETE /blocks[/{id}]           block_manager.*                                   block R/W
GET /config/{llm,embedding}                  server.list_{llm,embedding}_models                providers R + env
GET/POST /organizations[/{id,create_or_get}] organization_manager.*                            organizations R/W
GET/POST/DELETE /users[/{id}[/memories]]     user_manager.*                                    users R/W; memory tables W on cascade
GET/POST/PATCH/DELETE /clients[/{id}[/...]]  client_manager.*                                  clients R/W; agents/tools/block/memory W on cascade
POST/GET/DELETE /clients/{id}/api-keys[/{k}] client_manager.create/list/delete_client_api_key  client_api_keys R/W
POST /admin/auth/*                           ClientAuthManager.* (services/admin_user_manager) clients R/W
GET  /admin/dashboard-clients                ClientAuthManager.list_dashboard_clients          clients R
GET  /health                                 —                                                 none
```

---

## 4. Cross-Cutting Notes Worth Internalizing

1. **The HTTP boundary for `/memory/add` is always async** — the worker may
   redeliver. Anything that throws inside `Agent.step` after
   `_persist_memory_source` is a safe full retry because L1/L2 idempotency
   short-circuits the second attempt.
2. **There are two parallel "message" concepts**: queue/protobuf messages
   (transient, used to fan out work) and DB `messages` rows (persistent,
   only written when MetaAgent retention > 0). Don't conflate them.
3. **Steps are write-only audit logs** — never call `step_manager.get_step`
   from product code.
4. **Direct writes bypass the LLM but NOT provenance** — they still write
   `memory_sources` and `memory_citations`. Bypassing direct-write handlers
   to insert via the manager would break progressive disclosure.
5. **ECMS (Enterprise Context and Memory Service) deployment toggles**:
   `CHAINING_FOR_MEMORY_UPDATE=false` and `CHAINING_FOR_META_AGENT=false`.
   Each agent runs exactly once per save, simplifying the chaining loop
   in §2.3 to a single iteration.
6. **All I/O is async**: `asyncpg`, `redis.asyncio`, `aiokafka`,
   `httpx.AsyncClient`. The five intentional sync touch-points are listed
   in `docs/Mirix_async_native_changes.md`; don't add a sixth.
7. **Access control is data-shaped, not RBAC-shaped**: every memory row
   carries `filter_tags->>'scope'`, and every read query appends
   `scope = ANY(:read_scopes)`. Forget the scope on write, and the data is
   effectively orphaned.
8. **Default organization can be overridden by env**: `MIRIX_DEFAULT_ORG_ID`
   and `MIRIX_DEFAULT_ORG_NAME` (read in `mirix/constants.py` and
   `mirix/services/organization_manager.py`) let an embedding host
   (e.g. ECMS) pin a per-deployment org without code changes. Unset =
   the historical hardcoded `org-00000000-0000-4000-8000-000000000000` /
   `default_org`.
9. **Recent vs Relevant distinction at write time**: owning sub-agents
   call `_fetch_recent_indexing_lag_window(...)` to pull rows the
   relational provider just wrote (within `set_hybrid_window_seconds`)
   that the search provider may not yet have indexed. The bucket is
   surfaced in the prompt as `recent_*` separate from the search-ranked
   `relevant_*`, so the LLM can detect near-duplicate writes that just
   landed. PG-only deployments don't have an indexing-lag window, so the
   helper returns `[]` and the prompt only carries `relevant_*`.

---

## 5. Related Reading

- `docs/ARCHITECTURE.md` — the source-of-truth diagram and the endpoint
  table this document expands on.
- `docs/Mirix_async_native_changes.md` — the async refactor rationale and
  the list of approved sync touch-points.
- `docs/Mirix_Langfuse_logging.md` — span/trace shape per endpoint and the
  Langfuse PII masking pipeline (§1.7).
- `docs/Mirix_custom_providers.md` — Cache / Relational / Search provider
  contracts and the hybrid-search dedup helper.
- `CLAUDE.md` — ECMS (Enterprise Context and Memory Service) integration notes,
  idempotency layers, temporal guard, provenance sidecar.
- `mirix/server/rest_api.py` — every endpoint definition, in order, with
  their `BaseModel` request schemas inline.
- `mirix/agent/agent.py` — the chaining loop (`step` → `inner_step`), the
  direct-write dispatcher, and `_persist_memory_source`.
- `mirix/agent/meta_agent.py` — sub-agent orchestration and configuration.
- `mirix/queue/worker.py` and `mirix/queue/queue_util.py` — protobuf
  serialization, trace propagation, and the worker → `send_messages`
  bridge.
- `mirix/queue/error_policy.py` — `process_with_policy`, the
  Permanent/Transient classifier (`classify`), and `format_exc_chain` for
  cause-chain logging.
- `mirix/functions/function_sets/memory_tools.py::trigger_memory_update`
  — the meta-agent's sub-agent fan-out tool and the source of the
  per-sub-agent Langfuse `agent` spans.
- `mirix/services/hybrid_search_helper.py::fetch_and_dedup_candidates`
  — the relational+search hybrid-read helper that powers the
  `recent` vs `relevant` bucket split.
- `mirix/observability/pii_mask.py` and `mirix/pii.py` — ispy-pii
  integration for redacting PII out of Langfuse traces and LLM-error
  log lines respectively.
