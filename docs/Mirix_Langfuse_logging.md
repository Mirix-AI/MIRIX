# MIRIX LangFuse Logging

This document describes how MIRIX integrates with
[LangFuse](https://langfuse.com/) for observability and distributed tracing.
It covers:

1. How to enable / disable LangFuse at server start.
2. The categories of information that MIRIX emits to LangFuse, with
   representative samples.
3. The exact code-level touch points where LangFuse data is produced.

> Scope: this document only describes how MIRIX (this repo) talks to
> LangFuse. It does not describe LangFuse server administration or the
> LangFuse UI.

---

## 1. Enabling and Disabling LangFuse at Startup

### 1.1 Default behaviour

LangFuse is **disabled by default**. With no environment configuration,
`initialize_langfuse()` returns `None`, `is_langfuse_enabled()` returns
`False`, and every per-request tracing block becomes a no-op
context manager.

The relevant default lives in
`mirix/settings.py`:

```215:227:mirix/settings.py
    # LangFuse observability settings (for distributed tracing)
    langfuse_enabled: bool = Field(False, env="MIRIX_LANGFUSE_ENABLED")
    langfuse_public_key: Optional[str] = Field(None, env="MIRIX_LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(None, env="MIRIX_LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", env="MIRIX_LANGFUSE_HOST")
    langfuse_flush_interval: float = Field(1.0, env="MIRIX_LANGFUSE_FLUSH_INTERVAL")  # seconds
    langfuse_flush_at: int = Field(512, env="MIRIX_LANGFUSE_FLUSH_AT")  # spans per batch
    langfuse_debug: bool = Field(False, env="MIRIX_LANGFUSE_DEBUG")
    langfuse_flush_timeout: float = Field(10.0, env="MIRIX_LANGFUSE_FLUSH_TIMEOUT")  # seconds
    # Environment identifier for filtering traces in shared Langfuse projects
    # Common values: "dev", "e2e", "qal", "prf", "prod"
    # Must match regex: ^(?!langfuse)[a-z0-9-_]+$ with max 40 chars
    langfuse_environment: str = Field("dev", env="MIRIX_LANGFUSE_ENVIRONMENT")
```

### 1.2 Environment variables

All LangFuse configuration is driven by `MIRIX_LANGFUSE_*` environment
variables. The Pydantic `BaseSettings` loader maps them to fields on the
global `settings` object.

| Env var | Field | Default | Purpose |
|---|---|---|---|
| `MIRIX_LANGFUSE_ENABLED` | `langfuse_enabled` | `false` | Master switch. When `false`, no spans are created and no credentials are required. |
| `MIRIX_LANGFUSE_PUBLIC_KEY` | `langfuse_public_key` | `None` | LangFuse project public key. **Required when enabled.** |
| `MIRIX_LANGFUSE_SECRET_KEY` | `langfuse_secret_key` | `None` | LangFuse project secret key. **Required when enabled.** |
| `MIRIX_LANGFUSE_HOST` | `langfuse_host` | `https://cloud.langfuse.com` | LangFuse API base URL (cloud or self-hosted). |
| `MIRIX_LANGFUSE_ENVIRONMENT` | `langfuse_environment` | `dev` | Environment tag stamped on every trace (`dev`, `e2e`, `qal`, `prf`, `prod`). |
| `MIRIX_LANGFUSE_FLUSH_INTERVAL` | `langfuse_flush_interval` | `1.0` (s) | How often the SDK flushes batched spans to the LangFuse server. |
| `MIRIX_LANGFUSE_FLUSH_AT` | `langfuse_flush_at` | `512` | Max spans per batch before forcing a flush. |
| `MIRIX_LANGFUSE_FLUSH_TIMEOUT` | `langfuse_flush_timeout` | `10.0` (s) | Wait time when flushing on shutdown. |
| `MIRIX_LANGFUSE_DEBUG` | `langfuse_debug` | `false` | Enables verbose LangFuse SDK debug logs. |

### 1.3 Enabling LangFuse

To enable LangFuse on a MIRIX server, set the three required variables
in the process environment (or in `.env`) **before** the server starts:

```bash
# .env (or any other env source)
MIRIX_LANGFUSE_ENABLED=true
MIRIX_LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
MIRIX_LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
# optional overrides
MIRIX_LANGFUSE_HOST=https://cloud.langfuse.com
MIRIX_LANGFUSE_ENVIRONMENT=dev
```

Then start the server normally:

```bash
python scripts/start_server.py --port 8531
# or
docker-compose up -d
```

On startup, the FastAPI lifespan calls `initialize_langfuse()` via
`mirix/server/rest_api.py`:

```104:111:mirix/server/rest_api.py
    # Initialize LangFuse observability (async)
    try:
        from mirix.observability import initialize_langfuse

        logger.info("Initializing LangFuse observability...")
        await initialize_langfuse()
    except Exception as e:
        logger.warning(f"LangFuse initialization failed: {e}. Continuing without observability.")
```

Inside `initialize_langfuse()` (`mirix/observability/langfuse_client.py`)
MIRIX:

- Returns `None` immediately if `langfuse_enabled` is false.
- Emits a warning and returns `None` if `public_key` or `secret_key`
  are missing.
- Constructs the singleton `Langfuse(...)` SDK client (wrapped with
  `asyncio.to_thread` because the SDK is sync), passing through
  `public_key`, `secret_key`, `host`, `debug`, `flush_interval`,
  `flush_at`, a fresh `TracerProvider`, `environment`, **and a `mask=`
  callable returned by `mirix.observability.pii_mask.get_langfuse_mask()`**
  (see §1.7 below).
- Calls `_langfuse_client.flush()` once as a health check.
- Registers a process-`atexit` handler so traces are flushed even on
  abnormal shutdown.

A successful startup logs:

```
LangFuse observability initialized and verified successfully (environment: dev)
```

### 1.4 Disabling LangFuse

There are three equivalent ways to disable LangFuse:

1. Do not set `MIRIX_LANGFUSE_ENABLED` (it defaults to `false`).
2. Explicitly set `MIRIX_LANGFUSE_ENABLED=false`.
3. Set `MIRIX_LANGFUSE_ENABLED=true` but omit `MIRIX_LANGFUSE_PUBLIC_KEY`
   or `MIRIX_LANGFUSE_SECRET_KEY`. The initializer logs a warning and
   falls back to disabled.

When disabled, every code path that uses LangFuse follows the same
guard pattern:

```61:69:mirix/llm_api/llm_client_base.py
        langfuse = get_langfuse_client()
        trace_context = get_trace_context() if langfuse else {}
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None
        if langfuse and trace_id:
            return await self._execute_with_langfuse(langfuse, request_data, messages, tools, trace_id, parent_span_id)
        reason = "LangFuse client not available" if not langfuse else "No active trace_id in context"
        self.logger.debug(f"Sending LLM request without LangFuse tracing ({reason})")
        return await self._execute_without_langfuse(request_data, messages)
```

So a disabled LangFuse adds **zero runtime cost beyond an `if` check**;
no network calls are made.

### 1.5 Test environment

The dockerised test script forces LangFuse off so that unit/integration
tests don't accidentally ship traces to a real project:

```155:158:scripts/run_tests_with_docker.sh
export MIRIX_REDIS_PORT="6380"
export MIRIX_LANGFUSE_ENABLED="false"
```

### 1.6 Shutdown flow

`cleanup()` in `mirix/server/rest_api.py` flushes pending spans and
shuts the LangFuse client down so traces are not lost on container
exit:

```121:134:mirix/server/rest_api.py
    # Flush LangFuse traces before shutdown
    try:
        from mirix.observability import flush_langfuse, shutdown_langfuse

        logger.info("Flushing LangFuse traces...")
        flush_success = await flush_langfuse(timeout=10.0)
        if flush_success:
            logger.info("LangFuse traces flushed successfully")
        else:
            logger.warning("LangFuse trace flush completed with warnings")
        await shutdown_langfuse()
        logger.info("LangFuse observability shut down")
    except Exception as e:
        logger.warning(f"Error during LangFuse shutdown: {e}")
```

`flush_langfuse(timeout)` wraps `Langfuse.flush()` in
`asyncio.to_thread`, and `shutdown_langfuse()` additionally calls
`Langfuse.shutdown()` and resets the singleton flags.

### 1.7 PII masking on the Langfuse path (VEPAGE-983)

Every value Langfuse would export is run through a `mask=` callback
that forwards strings to ispy-pii (Intuit's PII redaction service) and
walks dicts / lists / tuples recursively. The callback is built by
`mirix/observability/pii_mask.py::build_langfuse_mask` and wired into
`Langfuse(...)` at initialization time.

Key design points (live in `mirix/observability/pii_mask.py` and the
shared helpers in `mirix/pii.py`):

- **Sync HTTP, by design.** The SDK invokes `mask` on its background
  flush thread, not on the request hot path, so a sync `httpx.Client`
  is acceptable here. The async log helper
  `mirix.pii.log_error_strip_pii` (used inside LLM-error catch blocks
  in `openai_client.py` / `anthropic_client.py`) uses
  `httpx.AsyncClient` instead — same endpoint, different I/O model.
- **Bounded LRU cache** (default 4096 entries, thread-safe via
  `OrderedDict + Lock`). Only successful redactions are cached so a
  transient ispy-pii outage doesn't poison the cache for the rest of
  the process lifetime.
- **Fail closed to a placeholder.** On any non-200 / timeout / network
  / malformed response after retries, the masker returns
  `REDACTED_PLACEHOLDER` (`"[REDACTED — PII masking unavailable]"`)
  rather than the original text. Failures must never raise — a logging
  path that re-enters its own exception handler would deadlock.
- **PrivateAuth header.** When `MIRIX_ISPY_PII_APPID` and
  `MIRIX_ISPY_PII_APP_SECRET` are set,
  `get_ispy_pii_auth_headers()` builds an
  `Authorization: Intuit_IAM_Authentication intuit_appid=...,intuit_app_secret=...`
  header plus `intuit_offeringid: <appid>`. Unset = no header is sent
  (standalone / OSS).
- **Pluggable.** Downstream consumers (notably ECMS) can register a
  different callable via
  `mirix.observability.set_langfuse_mask(fn)` *before* calling
  `initialize_langfuse()`; once the SDK is constructed the callback is
  closed over the singleton.

| Env var | Default | Purpose |
|---|---|---|
| `MIRIX_LANGFUSE_MASK_ENABLED` | `true` | Master kill-switch for the Langfuse mask. When `false`, the callable becomes a passthrough. |
| `MIRIX_ISPY_PII_ENABLED` | `true` | Same kill-switch for the **async** log helper in `mirix/pii.py` (used by LLM error handlers). |
| `MIRIX_ISPY_PII_ENDPOINT` | `https://ispypiis-e2e.api.intuit.com/v2/analyze` | ispy-pii v2/analyze URL (shared between mask and log helper). |
| `MIRIX_ISPY_PII_TIMEOUT_MS` | `200` | Per-request timeout. |
| `MIRIX_ISPY_PII_MAX_RETRIES` | `2` | Retries on `429 / 5xx` only. Total attempts = retries + 1. |
| `MIRIX_ISPY_PII_RETRY_BACKOFF_MS` | `50` | Base backoff; actual delays = `base * 2^(attempt-1) * jitter(0.8–1.2)`. |
| `MIRIX_ISPY_PII_APPID` | unset | PrivateAuth `intuit_appid`. Header is omitted when either appid or secret is missing. |
| `MIRIX_ISPY_PII_APP_SECRET` | unset | PrivateAuth `intuit_app_secret`. |

Test coverage lives in `tests/test_pii.py` and `tests/test_pii_mask.py`.

---

## 2. What MIRIX Sends to LangFuse

MIRIX uses the LangFuse v3 OpenTelemetry-style API. Every traced unit of
work is a single observation, either a `span`, an `agent`, a `tool`, a
`generation`, or an `embedding`. Observations are linked together with
the same `trace_id`. Trace metadata (`user_id`, `session_id`, request
attributes) is attached to the root via `update_current_trace(...)`.

### 2.1 Categories at a glance

| # | Category | Observation type | Originating module |
|---|---|---|---|
| 1 | HTTP request trace (root) | `span` | `mirix/server/rest_api.py::with_langfuse_tracing` |
| 2 | Meta Agent execution | `agent` | `mirix/queue/worker.py::_process_message_async` |
| 3 | Memory sub-agent execution | `agent` | `mirix/functions/function_sets/memory_tools.py::trigger_memory_update` (parallel fan-out via `asyncio.gather`) |
| 4 | Summary Agent execution | `agent` | `mirix/agent/agent.py::_generate_source_summary_traced` |
| 5 | Direct Writes (LLM-bypass) | `span` (+ child `span`s) | `mirix/agent/agent.py::_apply_direct_writes_traced` |
| 6 | LLM completion (generation) | `generation` | `mirix/llm_api/llm_client_base.py::_execute_with_langfuse` and `mirix/llm_api/llm_api_tools.py::create` |
| 7 | Embedding call | `embedding` | `mirix/embeddings.py::traced_embedding_with_retry` |
| 8 | Built-in tool execution | `tool` | `mirix/agent/agent.py::execute_tool_modifications_and_persist_agent_state` |
| 9 | User-defined tool sandbox | `tool` | `mirix/services/tool_execution_sandbox.py::run` |
| 10 | MCP tool invocation | `tool` | `mirix/functions/mcp_client/base_client.py::execute_tool` |
| 11 | Idempotency skip | `span` | `mirix/observability/skip_spans.py::emit_idempotency_skip_span` |

The hierarchical shape of a typical `POST /memory/add` trace is:

```
POST /memory/add                    (span, root, trace_id=...)
└── Meta Agent                      (agent, queue worker)
    ├── llm_completion              (generation, meta-agent LLM call)
    ├── tool: trigger_memory_update (tool, MIRIX_MEMORY_CORE)
    ├── Episodic Memory Agent       (agent, child sub-agent — concurrent)
    │   ├── llm_completion          (generation)
    │   ├── tool: episodic_memory_insert (tool, MIRIX_MEMORY_CORE)
    │   └── embedding               (embedding)
    ├── Semantic Memory Agent       (agent — concurrent)
    │   └── ...
    ├── Procedural Memory Agent     (agent — concurrent)
    │   └── ...
    ├── Direct Writes               (span)   ← only if request carries direct_writes
    │   └── insert_episodic_memory  (span)
    ├── Summary Agent               (agent)  ← only if request asks for a summary;
    │   └── llm_completion          (generation)         dispatched in parallel via
    │                                                    asyncio.create_task
    └── Idempotency Skip: ...       (span)   ← only when L1/L2/L3 short-circuits fire
```

> **Concurrency note.** The "Memory Agent" children and the "Summary
> Agent" sibling all run concurrently (`asyncio.gather` for sub-agents,
> `asyncio.create_task` for the summary). Their spans capture the
> parent context at dispatch time, so they appear as siblings under
> "Meta Agent" rather than as a strict serial chain.

The sections below describe each category and show the input/output/
metadata structure that the corresponding `start_as_current_observation`
or `update(...)` call sends to LangFuse.

### 2.2 HTTP request trace (root)

Every agentic REST endpoint decorated with `@with_langfuse_tracing`
opens the root observation for the request. A fresh `trace_id` is
generated per call and stored in a `ContextVar` so async/await tasks
inherit it transparently.

```243:286:mirix/server/rest_api.py
        method = request.method
        path = request.url.path
        user_id = request.headers.get("x-user-id")
        client_id = request.headers.get("x-client-id")
        org_id = request.headers.get("x-org-id")
        session_id = f"{client_id}-{uuid.uuid4().hex[:8]}"

        # Generate a unique trace_id for each request
        trace_id = uuid.uuid4().hex

        # Langfuse returns a sync context manager (_AgnosticContextManager); use
        # "with" not "async with" to avoid "does not support the asynchronous
        # context manager protocol".
        with langfuse.start_as_current_observation(
            name=f"{method} {path}",
            as_type="span",
            trace_context=cast(TraceContext, {"trace_id": trace_id}),
        ) as span:
            try:
                observation_id = getattr(span, "id", None)
                logger.debug(
                    f"LangFuse trace created: trace_id={trace_id}, observation_id={observation_id}, path={path}"
                )

                langfuse.update_current_trace(
                    name=f"{method} {path}",
                    user_id=user_id or client_id,
                    session_id=session_id,
                    metadata={
                        "method": method,
                        "path": path,
                        "client_id": client_id,
                        "org_id": org_id,
                        "user_agent": request.headers.get("user-agent"),
                    },
                )
```

Endpoints decorated with `@with_langfuse_tracing` today:

- `POST /memory/add`
- `POST /memory/retrieve/conversation`
- `GET /memory/retrieve/topic`
- `GET /memory/search`
- `GET /memory/search_all_users`

Sample data sent (conceptual JSON):

```json
{
  "name": "POST /memory/add",
  "trace_id": "3f2c…f1a0",
  "user_id": "client-abc",
  "session_id": "client-abc-9c8a72d4",
  "metadata": {
    "method": "POST",
    "path": "/memory/add",
    "client_id": "client-abc",
    "org_id": "org-001",
    "user_agent": "ECMS/0.4.2 httpx/0.27.0"
  },
  "output": {"status": "completed"}
}
```

If the handler raises, the span output becomes `{"error": "<str(e)>"}`
with `level="ERROR"`.

### 2.3 Meta Agent observation (queue worker)

When the request's queue message is consumed,
`mirix/queue/worker.py` restores the trace context that was attached at
enqueue time and opens an `agent` observation:

```349:380:mirix/queue/worker.py
            if langfuse and trace_id:
                from typing import cast

                from langfuse.types import TraceContext

                from mirix.observability.context import set_trace_context

                trace_context_dict: dict = {"trace_id": trace_id}

                with langfuse.start_as_current_observation(
                    name="Meta Agent",
                    as_type="agent",
                    trace_context=cast(TraceContext, trace_context_dict),
                    metadata={
                        "agent_id": message.agent_id,
                        "message_count": len(input_messages),
                        "source": "queue_worker",
                    },
                ) as span:
                    mark_observation_as_child(span)

                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=trace_context.get("user_id"),
                            session_id=trace_context.get("session_id"),
                        )
                    usage = await _do_send_messages()
```

Sample:

```json
{
  "name": "Meta Agent",
  "as_type": "agent",
  "trace_id": "3f2c…f1a0",
  "metadata": {
    "agent_id": "meta-agent-uuid",
    "message_count": 3,
    "source": "queue_worker"
  }
}
```

> Trace propagation: the trace context arrives on the
> `QueueMessage` protobuf via the `langfuse_trace_id`,
> `langfuse_observation_id`, `langfuse_session_id`, and
> `langfuse_user_id` fields, populated by
> `mirix/observability/trace_propagation.py::add_trace_to_queue_message`.

### 2.4 Memory sub-agent observations

When the meta agent's `trigger_memory_update(memory_types=[...])` tool
fans out to the specialised memory agents (Episodic / Semantic /
Procedural / Resource / Knowledge Vault / Core), each child agent
execution is wrapped in its own `agent` observation with a Title-Case
name derived from the agent type. The fan-out is concurrent
(`asyncio.gather`), so each task captures the parent trace context at
dispatch time and re-installs it inside the coroutine before opening
its span.

```1063:1087:mirix/functions/function_sets/memory_tools.py
                with langfuse.start_as_current_observation(
                    name=span_name,
                    as_type="agent",
                    trace_context=cast(TraceContext, trace_context_dict),
                    metadata={
                        "memory_type": memory_type,
                        "agent_name": agent_state.name,
                    },
                ) as span:
                    mark_observation_as_child(span)

                    # Get this span's ID for child operations
                    span_observation_id = getattr(span, "id", None)
                    if span_observation_id:
                        logger.debug(
                            f"Child agent span created: agent={agent_type_str}, "
                            f"span_id={span_observation_id}, parent={parent_span_id}"
                        )
                        # Update ContextVar so child LLM calls use this span as parent
                        set_trace_context(
                            trace_id=trace_id,
                            observation_id=span_observation_id,
                            user_id=parent_trace_context.get("user_id"),
                            session_id=parent_trace_context.get("session_id"),
                        )
```

> Note: `trigger_memory_update_with_instruction(...)` is a separate
> single-target tool used by the chat-agent / reflexion-agent prompts.
> It dispatches via the local-client SDK (no concurrent fan-out and
> no per-call sub-agent span — only the LLM `llm_completion`
> generation surfaces in Langfuse for that path).

Span names you will see: `Episodic Memory Agent`,
`Semantic Memory Agent`, `Procedural Memory Agent`,
`Resource Memory Agent`, `Knowledge Vault Memory Agent`,
`Core Memory Agent`.

Sample:

```json
{
  "name": "Episodic Memory Agent",
  "as_type": "agent",
  "metadata": {
    "memory_type": "episodic",
    "agent_name": "episodic_memory_agent_admin@org-001"
  }
}
```

### 2.5 Summary Agent observation

When the caller asks MIRIX to generate a summary for the
`memory_source` (via `summarize=true` in `AddMemoryRequest`), the
background summarisation task wraps the work in its own `agent` span:

```1795:1813:mirix/agent/agent.py
        with langfuse.start_as_current_observation(
            name="Summary Agent",
            as_type="agent",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata={
                "memory_source_id": self.memory_source_id,
                "agent_name": self.agent_state.name,
            },
        ) as span:
            mark_observation_as_child(span)
            span_observation_id = getattr(span, "id", None)
            if span_observation_id:
                set_trace_context(
                    trace_id=trace_id,
                    observation_id=span_observation_id,
                    user_id=parent_trace_context.get("user_id"),
                    session_id=parent_trace_context.get("session_id"),
                )
            await self._generate_source_summary()
```

Sample:

```json
{
  "name": "Summary Agent",
  "as_type": "agent",
  "metadata": {
    "memory_source_id": "ms_01HV…",
    "agent_name": "meta_agent_admin@org-001"
  }
}
```

The LLM call inside `_generate_source_summary()` is then captured as a
nested `llm_completion` generation (see 2.7).

### 2.6 Direct Writes spans

For caller-authored memories that bypass the LLM pipeline
(`AddMemoryRequest.direct_writes`), the meta agent emits a parent
`Direct Writes` span plus one child `insert_<memory_type>_memory` span
per direct write:

```1579:1632:mirix/agent/agent.py
        with langfuse.start_as_current_observation(
            name="Direct Writes",
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata={
                "memory_source_id": self.memory_source_id,
                "agent_name": self.agent_state.name,
                "direct_write_count": len(self.direct_writes),
                "memory_types": [w["memory_type"] for w in self.direct_writes],
            },
        ) as span:
            mark_observation_as_child(span)
            span_observation_id = getattr(span, "id", None)
            if span_observation_id:
                set_trace_context(
                    trace_id=trace_id,
                    observation_id=span_observation_id,
                    user_id=parent_trace_context.get("user_id"),
                    session_id=parent_trace_context.get("session_id"),
                )

            for write in self.direct_writes:
                memory_type = write["memory_type"]
                payload = write["payload"]
                handler = DIRECT_WRITE_HANDLERS.get(memory_type)
                function_name = handler.__name__ if handler else str(memory_type)
                insert_span_name = f"insert_{memory_type}_memory"

                trace_input: dict = {
                    "memory_type": memory_type,
                    "function": function_name,
                    "payload_keys": list(payload.keys()),
                }
                items = payload.get("items")
                if isinstance(items, list):
                    trace_input["items_count"] = len(items)

                if not span_observation_id:
                    await self._apply_direct_write(memory_type, payload)
                    continue

                insert_trace_dict: dict = {"trace_id": trace_id, "parent_span_id": span_observation_id}
                try:
                    insert_observation_cm = langfuse.start_as_current_observation(
                        name=insert_span_name,
                        as_type="span",
                        trace_context=cast(TraceContext, insert_trace_dict),
                        input=trace_input,
                        metadata={
                            "memory_source_id": self.memory_source_id,
                            "memory_type": memory_type,
                            "function": function_name,
                        },
                    )
```

Sample parent span:

```json
{
  "name": "Direct Writes",
  "as_type": "span",
  "metadata": {
    "memory_source_id": "ms_01HV…",
    "agent_name": "meta_agent_admin@org-001",
    "direct_write_count": 2,
    "memory_types": ["episodic", "semantic"]
  }
}
```

Sample child span:

```json
{
  "name": "insert_episodic_memory",
  "as_type": "span",
  "input": {
    "memory_type": "episodic",
    "function": "episodic_memory_insert",
    "payload_keys": ["items", "filter_tags"],
    "items_count": 3
  },
  "metadata": {
    "memory_source_id": "ms_01HV…",
    "memory_type": "episodic",
    "function": "episodic_memory_insert"
  },
  "output": {"status": "completed"}
}
```

> **Direct-write handler functions** (registered in
> `mirix/functions/direct_write_handlers.py`) are the same memory-tool
> callables used by the LLM path: `episodic_memory_insert`,
> `semantic_memory_insert`, `procedural_memory_insert`,
> `resource_memory_insert`, and `knowledge_vault_insert`. **`core` is
> NOT a registered direct-write type** — block writes go through the
> meta-agent's tool path.

> Direct writes are explicitly **not** wrapped by an LLM generation
> because they bypass the model entirely.

### 2.7 LLM completion (generation)

This is the most data-rich category. Every LLM call goes through one
of two instrumentation points:

1. `mirix/llm_api/llm_client_base.py::_execute_with_langfuse` —
   the standard async path used by all `LLMClient.create(...)`-based
   calls.
2. `mirix/llm_api/llm_api_tools.py::create` — a legacy/back-compat
   wrapper that supports the same provider matrix (OpenAI, Azure,
   Anthropic, Google AI, Bedrock, Groq, ...).

The generation captures:

- `name = "llm_completion"`, `as_type = "generation"`.
- `model`: prefers `llm_config.langfuse_model` (the canonical model
  name for cost tracking) and falls back to `llm_config.model`.
- `input.messages`: each message as `{role, content[, tool_calls]}`.
  Reasoning content (Anthropic / DeepSeek style) is marked with a
  `[reasoning]` prefix.
- `input.tools`: the OpenAI-style tool schemas (when sent).
- `metadata.provider`: `openai`, `azure`, `anthropic`, `google_ai`,
  `bedrock`, etc.
- `metadata.tools_count` (client-base path) or
  `metadata.functions_count` + `metadata.function_names` +
  `metadata.endpoint` + `metadata.summarizing` + `metadata.max_tokens` +
  `metadata.image_count` (legacy path).
- `output`: the assistant message + any `tool_calls` it produced.
- `usage_details` / `usage`: `{input, output, total}` token counts.
- On error: `level="ERROR"`, `status_message=str(e)`,
  `metadata.error_type = type(e).__name__`.

```138:148:mirix/llm_api/llm_client_base.py
            observation_context = langfuse.start_as_current_observation(
                name="llm_completion",
                as_type="generation",
                trace_context=cast("TraceContext", trace_context_dict),
                model=self.llm_config.langfuse_model or self.llm_config.model,
                input=trace_input,
                metadata={
                    "provider": self.llm_config.model_endpoint_type,
                    "tools_count": len(tools) if tools else 0,
                },
            )
```

Sample (success):

```json
{
  "name": "llm_completion",
  "as_type": "generation",
  "model": "gpt-4.1",
  "input": {
    "messages": [
      {"role": "system", "content": "You are the meta memory agent..."},
      {"role": "user", "content": "[USER] What did we talk about yesterday?"}
    ],
    "tools": [
      {"name": "trigger_memory_update_with_instruction", "description": "..."},
      {"name": "send_message", "description": "..."}
    ]
  },
  "metadata": {
    "provider": "openai",
    "tools_count": 7
  },
  "output": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_abc",
        "type": "function",
        "function": {
          "name": "trigger_memory_update_with_instruction",
          "arguments": "{\"memory_types\":[\"episodic\",\"semantic\"]}"
        }
      }
    ]
  },
  "usage_details": {"input": 3142, "output": 187, "total": 3329}
}
```

Sample (error):

```json
{
  "name": "llm_completion",
  "as_type": "generation",
  "model": "gpt-4.1",
  "level": "ERROR",
  "status_message": "Rate limit exceeded",
  "metadata": {"error_type": "RateLimitError"}
}
```

> Token counts are reported per call. LangFuse can then attribute
> per-trace cost using the `model` field, which is why
> `llm_config.langfuse_model` exists — it lets you map a private
> deployment name (e.g. an Intuit Genie route) to the canonical
> `gpt-4.1` cost tier.

### 2.8 Embedding observations

When a sub-agent generates embeddings via the standard wrappers
(`AsyncOpenAI`, `EmbeddingEndpoint`, `AzureOpenAIEmbedding`,
`OllamaEmbeddings`, ...), each retry attempt is traced individually:

```75:106:mirix/embeddings.py
            observation_context = langfuse.start_as_current_observation(
                name="embedding",
                as_type="embedding",
                trace_context=cast(TraceContext, trace_context_dict),
                model=model,
                input={"text": text_preview},
                metadata={
                    "provider": provider,
                    "endpoint": endpoint,
                    "input_length": len(text),
                },
            )
        except Exception as e:
            logger.error(f"Langfuse failed to start observation. Continuing without tracing: {e}")
            return await embedding_func()

        # Langfuse returns a sync context manager; use "with" not "async with"
        with observation_context as generation:
            mark_observation_as_child(generation)

            try:
                result = await embedding_func()
                try:
                    generation.update(
                        output={"embedding_dim": len(result) if result else 0},
                        metadata={
                            "provider": provider,
                            "endpoint": endpoint,
                            "input_length": len(text),
                            "output_dim": len(result) if result else 0,
                        },
                    )
```

Sample:

```json
{
  "name": "embedding",
  "as_type": "embedding",
  "model": "text-embedding-3-small",
  "input": {"text": "User asked about Q3 expenses..."},
  "metadata": {
    "provider": "openai_custom_auth",
    "endpoint": "https://intuit-genie/embeddings",
    "input_length": 128,
    "output_dim": 1536
  },
  "output": {"embedding_dim": 1536}
}
```

On failure the span is closed with `level="ERROR"`,
`status_message=str(e)` and `metadata.error_type`.

### 2.9 Built-in tool execution

Tools dispatched directly from the agent loop
(`MIRIX_CORE`, `MIRIX_MEMORY_CORE`, `MIRIX_EXTRA`, and
`MIRIX_MCP`-routed) are wrapped in a `tool` observation:

```598:623:mirix/agent/agent.py
                with langfuse.start_as_current_observation(
                    name=f"tool: {function_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={"tool_name": function_name, "args": args_for_trace},
                    metadata={
                        "tool_type": str(target_mirix_tool.tool_type),
                        "tool_name": function_name,
                        "agent_name": self.agent_state.name,
                    },
                ) as span:
                    mark_observation_as_child(span)
                    function_response, is_error = await _execute_tool_inner()

                    span.update(
                        output={
                            "response": str(function_response),
                            "is_error": is_error,
                        },
                        metadata={
                            "tool_type": str(target_mirix_tool.tool_type),
                            "tool_name": function_name,
                            "is_error": is_error,
                        },
                        level="ERROR" if is_error else "DEFAULT",
                    )
```

Sample:

```json
{
  "name": "tool: check_episodic_memory",
  "as_type": "tool",
  "input": {
    "tool_name": "check_episodic_memory",
    "args": {
      "query": "Q3 expenses",
      "timezone_str": "America/Los_Angeles"
    }
  },
  "metadata": {
    "tool_type": "ToolType.MIRIX_MEMORY_CORE",
    "tool_name": "check_episodic_memory",
    "agent_name": "episodic_memory_agent_admin@org-001",
    "is_error": false
  },
  "output": {
    "response": "Found 4 matching memories...",
    "is_error": false
  }
}
```

### 2.10 User-defined tool execution (sandbox)

A second `tool` span is emitted from `ToolExecutionSandbox.run` for the
sandboxed (`USER_DEFINED`) path. It carries the sandbox type:

```112:136:mirix/services/tool_execution_sandbox.py
                with langfuse.start_as_current_observation(
                    name=f"tool_execution: {self.tool_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={"tool_name": self.tool_name, "args": args_for_trace},
                    metadata={
                        "sandbox_type": sandbox_type,
                        "tool_name": self.tool_name,
                    },
                ) as span:
                    result = await _execute_tool()

                    span.update(
                        output={
                            "status": result.status,
                            "has_stdout": bool(result.stdout),
                            "has_stderr": bool(result.stderr),
                        },
                        metadata={
                            "sandbox_type": sandbox_type,
                            "tool_name": self.tool_name,
                            "status": result.status,
                        },
                        level="ERROR" if result.status == "error" else "DEFAULT",
                    )
```

Sample:

```json
{
  "name": "tool_execution: custom_lookup",
  "as_type": "tool",
  "input": {
    "tool_name": "custom_lookup",
    "args": {"id": "42"}
  },
  "metadata": {
    "sandbox_type": "local",
    "tool_name": "custom_lookup",
    "status": "success"
  },
  "output": {
    "status": "success",
    "has_stdout": true,
    "has_stderr": false
  }
}
```

`sandbox_type` is either `"e2b"` (when `tool_settings.e2b_api_key` is
set) or `"local"`.

### 2.11 MCP tool invocation

Remote MCP tools are traced from `BaseAsyncMCPClient.execute_tool` with
the originating MCP server name attached:

```106:126:mirix/functions/mcp_client/base_client.py
                with langfuse.start_as_current_observation(
                    name=f"mcp_tool: {tool_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={
                        "tool_name": tool_name,
                        "server": self.server_config.server_name,
                        "args": args_for_trace,
                    },
                    metadata={
                        "mcp_server": self.server_config.server_name,
                        "tool_name": tool_name,
                    },
                ) as span:
                    final_content, is_error = await _do_execute()

                    span.update(
                        output={"response": final_content, "is_error": is_error},
                        level="ERROR" if is_error else "DEFAULT",
                    )
                    return final_content, is_error
```

Sample:

```json
{
  "name": "mcp_tool: get_user_profile",
  "as_type": "tool",
  "input": {
    "tool_name": "get_user_profile",
    "server": "user-intuit-capability-context",
    "args": {"user_id": "user-123"}
  },
  "metadata": {
    "mcp_server": "user-intuit-capability-context",
    "tool_name": "get_user_profile"
  },
  "output": {
    "response": "{\"profile\":{...}}",
    "is_error": false
  }
}
```

### 2.12 Idempotency skip spans

When the provenance pipeline short-circuits because of one of the three
idempotency layers (source-deduped, processing-complete, temporal
guard), a dedicated `span` is emitted so the trace shows the skip
reason instead of looking like processing stopped silently:

```44:58:mirix/observability/skip_spans.py
    span_metadata: Dict[str, Any] = {"skip_reason": reason}
    if metadata:
        span_metadata.update(metadata)

    trace_context_dict: Dict[str, Any] = {"trace_id": trace_id}
    if parent_span_id:
        trace_context_dict["parent_span_id"] = parent_span_id

    try:
        with langfuse.start_as_current_observation(
            name=name,
            as_type="span",
            trace_context=cast(TraceContext, trace_context_dict),
            metadata=span_metadata,
        ) as span:
            mark_observation_as_child(span)
```

Sample:

```json
{
  "name": "Idempotency Skip: source deduped",
  "as_type": "span",
  "metadata": {
    "skip_reason": "source-deduped",
    "memory_source_id": "ms_01HV…",
    "external_id": "ext_thread_42#turn_5"
  }
}
```

### 2.13 What is *not* sent to LangFuse

A few important non-emissions:

- **Database messages**: raw `messages` rows and `source_messages` rows
  are stored in PostgreSQL, not LangFuse. Only short previews (first
  500 chars of the first 10 messages, in `_execute_with_langfuse`'s
  legacy path) are embedded in the generation input.
- **Embeddings vectors**: only `embedding_dim` is reported; the raw
  vector is never sent.
- **Secrets / API keys / auth headers**: provider keys, auth-provider
  headers, OAuth refresh tokens, and dashboard JWTs are not added to
  any span input or metadata.
- **Server housekeeping**: `GET /health`, `/docs`, `/redoc`, Redis
  pings, SQLAlchemy DDL — none of these are decorated with
  `@with_langfuse_tracing`.

---

## 3. LangFuse Touch Points in the Code

This is the exhaustive list of every place in the MIRIX repo where
LangFuse code is invoked, grouped by role. Filenames are relative to
the repo root.

### 3.1 SDK plumbing

| File | Role |
|---|---|
| `pyproject.toml` (`langfuse (>=3.11.0,<4.0.0)`) | Pins the LangFuse Python SDK version. |
| `scripts/packaging/setup_server.py` (`"langfuse>=3.11.0,<4.0.0"`) | Same pin for the standalone installer. |
| `mirix/settings.py` (lines 215–227) | Defines all eight `langfuse_*` settings, read from `MIRIX_LANGFUSE_*` env vars. |

### 3.2 Singleton and lifecycle (`mirix/observability/`)

| File | Symbol | Role |
|---|---|---|
| `mirix/observability/langfuse_client.py` | `initialize_langfuse()` | Coroutine-safe singleton init (calls `Langfuse(...)` via `asyncio.to_thread`). |
| `mirix/observability/langfuse_client.py` | `get_langfuse_client()` | Returns the singleton (or `None` when disabled). |
| `mirix/observability/langfuse_client.py` | `is_langfuse_enabled()` | Boolean predicate used by all hot paths. |
| `mirix/observability/langfuse_client.py` | `flush_langfuse(timeout)` | `await asyncio.to_thread(client.flush)`. |
| `mirix/observability/langfuse_client.py` | `shutdown_langfuse()` | Flush + `client.shutdown()` + reset singleton. |
| `mirix/observability/langfuse_client.py` | `_sync_flush_for_atexit` | Sync flush registered via `atexit`. |
| `mirix/observability/langfuse_client.py` | `_reset_for_testing()` | Test-only singleton reset. |
| `mirix/observability/context.py` | `current_trace_id`, `current_observation_id`, `current_session_id`, `current_user_id` | `ContextVar`s that carry trace context across `await` boundaries. |
| `mirix/observability/context.py` | `set_trace_context`, `get_trace_context`, `clear_trace_context` | Helpers for ContextVar IO. |
| `mirix/observability/context.py` | `mark_observation_as_child(observation)` | Forces `LangfuseOtelSpanAttributes.AS_ROOT=False` so children don't become roots. |
| `mirix/observability/trace_propagation.py` | `add_trace_to_queue_message(msg)` / `restore_trace_from_queue_message(msg)` | Round-trip trace context through protobuf queue messages. |
| `mirix/observability/trace_propagation.py` | `serialize_trace_context()` / `deserialize_trace_context(msg)` / `add_trace_to_message(msg)` | Dict-based equivalents used by older Kafka payloads/tests. |
| `mirix/observability/skip_spans.py` | `emit_idempotency_skip_span(name, reason, metadata)` | Emits a no-op span when a memory operation short-circuits. |
| `mirix/observability/pii_mask.py` | `build_langfuse_mask()` / `get_langfuse_mask()` / `set_langfuse_mask(fn)` | Sync ispy-pii-backed callable plugged into `Langfuse(mask=...)`; consumers (e.g. ECMS) can swap in a custom callable before `initialize_langfuse()`. |
| `mirix/observability/__init__.py` | Re-exports above for the `from mirix.observability import …` style used everywhere. |

### 3.3 Server startup / shutdown (`mirix/server/`)

| File | Line | Touch point |
|---|---|---|
| `mirix/server/rest_api.py` | 104–111 | `initialize()` calls `await initialize_langfuse()`. |
| `mirix/server/rest_api.py` | 121–134 | `cleanup()` calls `flush_langfuse(timeout=10.0)` then `shutdown_langfuse()`. |
| `mirix/server/rest_api.py` | 142–154 | `lifespan()` context manager wires both into FastAPI. |
| `mirix/server/server.py` | 387–396 | Imports `initialize_langfuse`, `flush_langfuse`, `shutdown_langfuse` so the observability module is loadable from the library entry point. |

### 3.4 REST tracing decorator (`mirix/server/rest_api.py`)

| Line | Touch point |
|---|---|
| 209–309 | `@with_langfuse_tracing` decorator. Opens the root `span`, generates a fresh `trace_id`, populates trace metadata via `langfuse.update_current_trace(...)`, pushes context into ContextVars, updates `span.update(output=...)` on success, and `span.update(output={'error': str(e)}, level='ERROR')` on failure. |
| 2062 | `@with_langfuse_tracing` on `add_memory` (`POST /memory/add`). |
| 2536 | `@with_langfuse_tracing` on `retrieve_memory_with_conversation` (`POST /memory/retrieve/conversation`). |
| 2695 | `@with_langfuse_tracing` on `retrieve_memory_with_topic` (`GET /memory/retrieve/topic`). |
| 2917 | `@with_langfuse_tracing` on `search_memory` (`GET /memory/search`). |
| 3511 | `@with_langfuse_tracing` on `search_memory_all_users` (`GET /memory/search_all_users`). |

### 3.5 Queue worker (`mirix/queue/`)

| File | Line | Touch point |
|---|---|---|
| `mirix/queue/worker.py` | 13 | Imports `get_langfuse_client`, `mark_observation_as_child`, `restore_trace_from_queue_message`. |
| `mirix/queue/worker.py` | 164 | `restore_trace_from_queue_message(message)` restores the trace context produced by the API tier. |
| `mirix/queue/worker.py` | 178–182 | Reads `langfuse` + current trace context. |
| `mirix/queue/worker.py` | 349–378 | Opens the `"Meta Agent"` (`as_type="agent"`) observation and pushes the new observation id into ContextVars before calling `server.send_messages(...)`. |
| `mirix/queue/worker.py` | 396 | `clear_trace_context()` in `finally`. |
| `mirix/queue/queue_util.py` | 8, 275–276 | `add_trace_to_queue_message(queue_msg)` attaches the current trace context to the outgoing protobuf message. |
| `mirix/queue/message.proto` | 41–52 | Optional protobuf fields `langfuse_trace_id`, `langfuse_observation_id`, `langfuse_session_id`, `langfuse_user_id`. |
| `mirix/queue/message_pb2.py` / `message_pb2.pyi` | — | Generated Python bindings for the protobuf fields above. |

### 3.6 Agent runtime (`mirix/agent/`)

| File | Line | Touch point |
|---|---|---|
| `mirix/agent/agent.py` | 41 | `from mirix.observability.langfuse_client import get_langfuse_client`. |
| `mirix/agent/agent.py` | ~480–650 (around `start_as_current_observation` on line 598) | `execute_tool_modifications_and_persist_agent_state(...)` wraps each built-in tool call in a `"tool: <name>"` (`as_type="tool"`) observation with `input.tool_name`, `input.args`, and on completion sets `output.response`, `output.is_error`, `level=ERROR` on error. |
| `mirix/agent/agent.py` | 1341–1347 | Inside `Agent.step`: `summary_task = asyncio.create_task(self._generate_source_summary_traced())` is dispatched in parallel, and `await self._apply_direct_writes_traced()` runs before the chaining loop when `direct_writes` are present. |
| `mirix/agent/agent.py` | 1542–1750 | `_apply_direct_writes_traced(...)` opens the `"Direct Writes"` parent span (line 1579) and one `"insert_<memory_type>_memory"` child span per direct write (line 1622); manages ContextVar parenting so child handlers (and their embeddings) attach correctly. |
| `mirix/agent/agent.py` | 1769–1820 | `_generate_source_summary_traced(...)` opens the `"Summary Agent"` (`as_type="agent"`) span used by the parallel `asyncio.create_task` summary job (start_as_current_observation on line 1795). |

### 3.7 Sub-agent dispatch (`mirix/functions/function_sets/memory_tools.py`)

| Line | Touch point |
|---|---|
| 17 | Imports `get_langfuse_client`. |
| 879 | `trigger_memory_update(memory_types=[...])` — the entry point fanned out by the meta agent's tool call (NOT `trigger_memory_update_with_instruction`, which is used by chat / reflexion agents and dispatches via the local-client SDK). |
| 947–948 | Captures `parent_trace_context` + `langfuse` once before spawning sub-agent tasks (needed because `asyncio.gather` runs them concurrently and the ContextVar can't be inherited). |
| 953–959 | Each sub-agent task restores `parent_trace_context` into its own ContextVar so its LLM calls parent correctly. |
| 1063–1087 | Opens an `as_type="agent"` observation per memory sub-agent named after the agent type (`Episodic Memory Agent`, `Semantic Memory Agent`, ...), then calls `memory_agent.step(...)` with the new span as the active context. |
| 1109–1110 | `clear_trace_context()` in `finally` to prevent context leaks across the gather boundary. |

### 3.8 LLM API instrumentation (`mirix/llm_api/`)

| File | Line | Touch point |
|---|---|---|
| `mirix/llm_api/llm_client_base.py` | 6, 10 | Imports `Langfuse` SDK type + `get_langfuse_client`. |
| `mirix/llm_api/llm_client_base.py` | 38–69 | `send_llm_request(...)` branches on `langfuse and trace_id` to `_execute_with_langfuse` vs `_execute_without_langfuse`. |
| `mirix/llm_api/llm_client_base.py` | 83–190 | `_execute_with_langfuse(...)` builds message list (with `role`, `content`, `tool_calls`, `[reasoning]…` markers), opens `"llm_completion"` (`as_type="generation"`) with `model = langfuse_model or model`, then `generation.update(output=…, usage_details=…)` on success or `level="ERROR"`, `status_message`, `metadata.error_type` on failure. |
| `mirix/llm_api/llm_client_base.py` | 192–232 | `_build_output_message(...)` and `_build_usage_dict(...)` helpers. |
| `mirix/llm_api/llm_api_tools.py` | 27 | Imports `get_langfuse_client`. |
| `mirix/llm_api/llm_api_tools.py` | 99–117 | `_extract_generation_metadata(...)` populates `provider`, `endpoint`, `summarizing`, `functions_count`, `function_names`, `max_tokens`, `image_count`. |
| `mirix/llm_api/llm_api_tools.py` | 157–199 | Legacy `create(...)` path: opens `"llm_completion"` (`as_type="generation"`) and updates with output + usage per provider branch (OpenAI, Azure, Anthropic, Google AI, Bedrock, Groq, ...). |
| `mirix/schemas/llm_config.py` | 95–98 | `LLMConfig.langfuse_model` — canonical model name for cost tracking. Wins over `LLMConfig.model` whenever a generation is emitted. |
| `mirix/schemas/embedding_config.py` | 58 | `EmbeddingConfig.langfuse_model` — same idea for embeddings. |

### 3.9 Embeddings (`mirix/embeddings.py`)

| Line | Touch point |
|---|---|
| 15 | Imports `get_langfuse_client`. |
| 22–28 | `is_embedding_tracing_enabled()` predicate (requires both a client and an active `trace_id`). |
| 44–132 | `traced_embedding_with_retry(...)` opens an `"embedding"` (`as_type="embedding"`) observation per attempt with `input.text`, `model`, `metadata.provider`, `metadata.endpoint`, `metadata.input_length`, and updates `output.embedding_dim` / `metadata.output_dim` on success or `level="ERROR"` on failure. |
| 243, 321, 354, 403 | Per-provider embedding classes call `traced_embedding_with_retry(..., model=self.config.langfuse_model or self.model, provider="...")`. |
| 504, 512, 519 | Embedding factory passes `langfuse_model=config.langfuse_model` into each per-provider wrapper. |

### 3.10 Tools and sandboxes

| File | Line | Touch point |
|---|---|---|
| `mirix/services/tool_execution_sandbox.py` | 13, 85–141 | Wraps the e2b/local sandbox `run(...)` in `"tool_execution: <name>"` (`as_type="tool"`) with `metadata.sandbox_type ∈ {e2b, local}`. |
| `mirix/functions/mcp_client/base_client.py` | 13, 69–131 | Wraps `BaseAsyncMCPClient.execute_tool(...)` in `"mcp_tool: <name>"` (`as_type="tool"`) with `metadata.mcp_server` and `output.response` + `output.is_error`. |
| `mirix/functions/function_sets/memory_tools.py` | (see 3.7) | Sub-agent spans (the `agent` observations into which built-in tool spans nest). |

### 3.11 Tests that exercise LangFuse instrumentation

| File | What it covers |
|---|---|
| `tests/test_langfuse_integration.py` | `initialize_langfuse`, `is_langfuse_enabled`, `flush_langfuse`, `shutdown_langfuse`, trace-context serialisation, `add_trace_to_message` / `deserialize_trace_context`. Uses `_reset_for_testing()` between cases. |
| `tests/test_idempotency_skip_spans.py` | `emit_idempotency_skip_span(...)` for the L1/L2/L3 short-circuits. |
| `tests/test_citation_writes.py` | Verifies citation writes don't leak trace context across calls. |
| `tests/test_summary_generation.py` | Verifies the `"Summary Agent"` span behaviour. |
| `tests/test_manager_delegation.py` | Sanity-checks that manager delegation does not bypass the LangFuse-aware paths. |
| `tests/test_block_filter_tags_update_mode.py` | Uses LangFuse off as a test default. |
| `tests/test_convert_timezone_to_utc.py` | Same. |
| `scripts/run_tests_with_docker.sh` (line 157) | Sets `MIRIX_LANGFUSE_ENABLED="false"` for all dockerised test runs. |

### 3.12 Documentation cross-references

- `CLAUDE.md` — describes the `Summary Agent` span and the
  `skip_spans.py` idempotency contract.
- `docs/ARCHITECTURE.md` — mentions LangFuse as the observability
  backend.
- `docs/Mirix_async_native_changes.md` (sections 134–160) — explains
  the `asyncio.to_thread` wrapping of the sync LangFuse SDK as one of
  the five approved sync touch points.
- `docs/Mirix_custom_providers.md` (lines 1145–1244) — sample `.env`
  with `MIRIX_LANGFUSE_*` and the note that misconfigured providers
  show up as childless spans or wrong-reason `Idempotency Skip` spans.
- `docs/Mirix_API_flow_document.md` — endpoint catalogue that maps
  one-to-one with the `@with_langfuse_tracing`-decorated routes in
  section 3.4 above.
