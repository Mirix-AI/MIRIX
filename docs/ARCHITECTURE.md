# MIRIX Architecture

## System Overview

Three-tier architecture: Client SDK → REST API → AsyncServer

```
┌──────────────────────────────────┐
│  MirixClient                     │
│  mirix/client/remote_client.py   │
└─────────────┬────────────────────┘
              │ HTTP/HTTPS
              ▼
┌──────────────────────────────────┐
│  REST API (FastAPI, port 8531)   │
│  mirix/server/rest_api.py        │
│  get_server() → singleton        │
└─────────────┬────────────────────┘
              │ method calls
              ▼
┌──────────────────────────────────┐
│  AsyncServer                     │
│  mirix/server/server.py          │
└─────────────┬────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  MetaAgent                              │
│  mirix/agent/meta_agent.py              │
└──┬──────┬──────┬──────┬──────┬──────┬──┘
   ▼      ▼      ▼      ▼      ▼      ▼
 Core  Episodic Semantic Proced Resource KnowledgeVault
Agent  Agent   Agent    Agent  Agent    Agent
```

## Data Flow

### Write path (add memories)
```
Client → POST /memory/add
       → AsyncServer.add()
       → MetaAgent.step()
       → Agent chaining loop (see Agent Execution below)
       → Sub-agents extract typed memories via tools
       → Managers (mirix/services/) persist via ORM
       → PostgreSQL
```

### Read path (retrieve memories)
```
Client → POST /memory/retrieve/conversation
       → AsyncServer.retrieve()
       → Memory Managers query PostgreSQL
           - BM25 full-text search (pg_bm25)
           - Vector similarity search (pgvector)
           - Fuzzy match (rapidfuzz)
       → Results ranked and returned
```

## Layer Responsibilities

| Layer | Location | Responsibility |
|-------|----------|----------------|
| ORM Models | `mirix/orm/` | SQLAlchemy table definitions |
| Schemas | `mirix/schemas/` | Pydantic request/response validation |
| Managers | `mirix/services/` | Business logic, DB access |
| Agents | `mirix/agent/` | LLM orchestration, memory extraction |
| API | `mirix/server/rest_api.py` | HTTP routing, auth |
| Client SDK | `mirix/client/remote_client.py` | External API wrapper |

## Memory Types

| Type | ORM | Manager | Agent | Description |
|------|-----|---------|-------|-------------|
| Core | `mirix/orm/block.py` | `mirix/services/block_manager.py` | `core_memory_agent` | Persona + human profile blocks |
| Episodic | `mirix/orm/episodic_memory.py` | `mirix/services/episodic_memory_manager.py` | `episodic_memory_agent` | Time-stamped events |
| Semantic | `mirix/orm/semantic_memory.py` | `mirix/services/semantic_memory_manager.py` | `semantic_memory_agent` | Facts and concepts |
| Procedural | `mirix/orm/procedural_memory.py` | `mirix/services/procedural_memory_manager.py` | `procedural_memory_agent` | How-to procedures |
| Resource | `mirix/orm/resource_memory.py` | `mirix/services/resource_memory_manager.py` | `resource_memory_agent` | Files, links, assets |
| Knowledge Vault | `mirix/orm/knowledge_vault.py` | `mirix/services/knowledge_vault_manager.py` | `knowledge_vault_memory_agent` | Sensitive/private facts |

## Agent Types

| Agent | AgentType Enum | Purpose |
|-------|---------------|---------|
| `core_memory_agent` | `AgentType.core_memory_agent` | Maintains persona + human profile |
| `episodic_memory_agent` | `AgentType.episodic_memory_agent` | Extracts time-stamped events |
| `semantic_memory_agent` | `AgentType.semantic_memory_agent` | Extracts facts and concepts |
| `procedural_memory_agent` | `AgentType.procedural_memory_agent` | Extracts procedures |
| `resource_memory_agent` | `AgentType.resource_memory_agent` | Extracts files/links/assets |
| `knowledge_vault_memory_agent` | `AgentType.knowledge_vault_memory_agent` | Extracts sensitive facts |
| `meta_memory_agent` | `AgentType.meta_memory_agent` | Coordinates memory sub-agents |
| `reflexion_agent` | `AgentType.reflexion_agent` | Self-reflection and learning |
| `background_agent` | `AgentType.background_agent` | Background processing tasks |
| `chat_agent` | `AgentType.chat_agent` | User-facing conversation |
| `coder_agent` | `AgentType.coder_agent` | Specialized coding tasks |

## Agent Execution

### step() method
```python
async def step(
    self,
    input_messages,
    chaining: bool = True,       # enabled by default
    max_chaining_steps: Optional[int] = None,
    ...
) -> MirixUsageStatistics
```

### Chaining loop
Agents loop through `inner_step()` calls until a terminal tool is called:
- **Chat agents** terminate on `send_message()` — delivers response to user
- **Memory agents** terminate on `finish_memory_update()` — signals memory write complete

Between steps, heartbeat messages are injected:
- Normal continuation: `"[System] Function called using continue_chaining=true, returning control"`
- After failure: `"[System] Function call failed, returning control"`

When `max_chaining_steps` is reached, agents are directed to call their terminal tool immediately.

### Context window management
- **Warning threshold**: 75% of context window (`memory_warning_threshold = 0.75`)
- **Target after summarization**: 10% memory pressure (`desired_memory_token_pressure = 0.1`)
- Summarization preserves the last 5 messages (`keep_last_n_messages = 5`)
- Summary is prepended; old messages are trimmed

## Tool Types

| Type | Execution |
|------|-----------|
| `MIRIX_CORE` | `get_function_from_module(MIRIX_CORE_TOOL_MODULE_NAME, name)` |
| `MIRIX_MEMORY_CORE` | `get_function_from_module(MIRIX_MEMORY_TOOL_MODULE_NAME, name)` |
| `MIRIX_EXTRA` | `get_function_from_module(MIRIX_EXTRA_TOOL_MODULE_NAME, name)` |
| `USER_DEFINED` | `ToolExecutionSandbox(...).run()` — isolated sandbox |
| `MCP` | Model Context Protocol — arguments passed as-is |

## REST API Endpoints

### Health
| Method | Path |
|--------|------|
| GET | `/health` |

### Agents
| Method | Path |
|--------|------|
| GET | `/agents` |
| POST | `/agents` |
| GET | `/agents/{agent_id}` |
| DELETE | `/agents/{agent_id}` |
| PATCH | `/agents/{agent_id}` |
| PATCH | `/agents/{agent_id}/system` |
| PATCH | `/agents/by-name/{agent_name}/system` |
| POST | `/agents/meta/initialize` |
| POST | `/agents/{agent_id}/messages` |

### Memory
| Method | Path |
|--------|------|
| POST | `/memory/add` |
| POST | `/memory/retrieve/conversation` |
| GET | `/memory/retrieve/topic` |
| GET | `/memory/search` |
| GET | `/memory/search_all_users` |
| POST | `/memory/search_raw` |
| GET | `/memory/components` |
| GET | `/memory/fields` |
| POST | `/memory/raw` |
| GET | `/memory/raw/{memory_id}` |
| PATCH | `/memory/raw/{memory_id}` |
| DELETE | `/memory/raw/{memory_id}` |
| POST | `/memory/raw/cleanup` |
| PATCH | `/memory/episodic/{memory_id}` |
| DELETE | `/memory/episodic/{memory_id}` |
| PATCH | `/memory/semantic/{memory_id}` |
| DELETE | `/memory/semantic/{memory_id}` |
| PATCH | `/memory/procedural/{memory_id}` |
| DELETE | `/memory/procedural/{memory_id}` |
| PATCH | `/memory/resource/{memory_id}` |
| DELETE | `/memory/resource/{memory_id}` |
| DELETE | `/memory/knowledge_vault/{memory_id}` |

### Tools
| Method | Path |
|--------|------|
| GET | `/tools` |
| POST | `/tools` |
| GET | `/tools/{tool_id}` |
| DELETE | `/tools/{tool_id}` |

### Blocks
| Method | Path |
|--------|------|
| GET | `/blocks` |
| POST | `/blocks` |
| GET | `/blocks/{block_id}` |
| DELETE | `/blocks/{block_id}` |

### Users & Organizations
| Method | Path |
|--------|------|
| GET | `/users` |
| GET | `/users/{user_id}` |
| POST | `/users/create_or_get` |
| DELETE | `/users/{user_id}` |
| DELETE | `/users/{user_id}/memories` |
| GET | `/organizations` |
| POST | `/organizations` |
| GET | `/organizations/{org_id}` |
| POST | `/organizations/create_or_get` |

### Clients & API Keys
| Method | Path |
|--------|------|
| GET | `/clients` |
| POST | `/clients/create_or_get` |
| GET | `/clients/{client_id}` |
| PATCH | `/clients/{client_id}` |
| DELETE | `/clients/{client_id}` |
| POST | `/clients/{client_id}/api-keys` |
| GET | `/clients/{client_id}/api-keys` |
| DELETE | `/clients/{client_id}/api-keys/{api_key_id}` |
| DELETE | `/clients/{client_id}/memories` |

### Configuration
| Method | Path |
|--------|------|
| GET | `/config/llm` |
| GET | `/config/embedding` |

### Admin / Auth
| Method | Path |
|--------|------|
| POST | `/admin/auth/register` |
| POST | `/admin/auth/login` |
| GET | `/admin/auth/me` |
| POST | `/admin/auth/change-password` |
| GET | `/admin/auth/check-setup` |
| GET | `/admin/dashboard-clients` |

## Authentication

Two supported methods:
- **API Key**: `X-API-Key: <key>` header — for programmatic clients
- **Bearer JWT**: `Authorization: Bearer <token>` — for dashboard sessions

Client context passed via:
- `x-client-id` header — identifies the client
- `x-org-id` header — identifies the organization

Validated via `get_client_and_org()` helper in `rest_api.py`.

## Infrastructure Services

| Service | Image | Port | Used For |
|---------|-------|------|----------|
| PostgreSQL + pgvector | `ankane/pgvector:v0.5.1` | 5432 | Primary storage, BM25, vector search |
| Redis Stack | `redis/redis-stack-server` | 6379 | Caching, fast lookups |
| Kafka (via aiokafka) | — | — | Async memory extraction queue |

## Key Files

| File | Purpose |
|------|---------|
| `mirix/server/rest_api.py` | All HTTP endpoints (FastAPI) |
| `mirix/server/server.py` | AsyncServer — core request handling logic |
| `mirix/agent/meta_agent.py` | Orchestrates all memory sub-agents |
| `mirix/agent/agent.py` | Base async agent: `step()`, chaining, tool execution |
| `mirix/settings.py` | All env var configuration (Pydantic BaseSettings) |
| `mirix/constants.py` | Heartbeat messages, thresholds, shared constants |
| `mirix/queue/manager.py` | Queue orchestration |
| `mirix/llm_api/` | LLM provider clients (OpenAI, Anthropic, Google, etc.) |
| `mirix/client/remote_client.py` | Async HTTP client SDK (70+ methods) |

## Async Design
The full async rationale is documented in `docs/Mirix_async_native_changes.md`.
All I/O uses async drivers: `asyncpg`, `redis.asyncio`, `aiokafka`, `httpx.AsyncClient`.
Only 5 intentional sync touch-points exist — see that doc for details.
