---
name: message_ids refactor plan
overview: Overhaul MIRIX message management by removing the message_ids JSON array from Agent, eliminating redundant system message storage, and introducing in-memory message accumulation for agent steps. This resolves scaling bottlenecks, removes write contention, and eliminates wasteful write-then-delete churn.
todos:
  - id: in-memory-accumulator
    content: Refactor Agent.step() and inner_step() to accumulate messages in-memory during the step loop instead of persisting after each inner_step
    status: pending
  - id: system-message-from-agent
    content: Construct system message on the fly from agent_state.system instead of storing it as a Message row
    status: pending
  - id: orm-changes
    content: Remove message_ids from Agent ORM and change messages relationship to lazy=noload
    status: pending
  - id: schema-changes
    content: Update Pydantic schemas for Agent (remove message_ids)
    status: pending
  - id: agent-manager-rewrite
    content: Rewrite AgentManager message methods to use query-based retrieval ordered by created_at, id
    status: pending
  - id: message-manager-updates
    content: Add query-based fetch and bulk hard-delete methods for retention pruning; remove detached message cleanup
    status: pending
  - id: llm-api-layer
    content: Update Anthropic and other LLM clients that assume messages[0] is the system message
    status: pending
  - id: retention-config
    content: Add message_set_retention_count to Client ORM/schema; implement retention enforcement at end of step()
    status: pending
  - id: summarization-cleanup
    content: Remove unneeded summarization code paths/settings for memory extraction flow and fail directly on context overflow
    status: pending
  - id: cleanup-managers
    content: Replace message_ids manipulation in UserManager, ClientManager with bulk message hard-delete for retention pruning
    status: pending
  - id: api-client-sdk
    content: Update REST API, server, SDK, and client layers to remove message_ids references
    status: pending
  - id: migration
    content: Create SQL migration to add message_set_retention_count, indexes, remove legacy system messages, and drop message_ids
    status: pending
  - id: tests
    content: "Update and add tests for new message management patterns: unit tests (mocked, no infra) for granular method behavior + integration tests invoking REST API endpoints to verify end-to-end correctness. Rewrite test_message_handling.py and test_agent_prompt_update.py in-place. Run via: ./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO"
    status: pending
  - id: chat-agent-deprecation
    content: Make chat_agent fail loudly (NotImplementedError) when step() is invoked — it is broken by this refactor and will be fixed in a follow-up. Add deprecation notice to docs/ARCHITECTURE.md and a warning comment in agent.py at the chat_agent branch.
    status: pending
isProject: false
---

# Proposal: MIRIX Message Management Overhaul

## 1. Problem

MIRIX manages agent conversation history through a `message_ids` JSON column on the `agents` table. This is a flat array of message IDs representing the agent's "in-context memory." The design has several scaling and correctness problems:

**Scaling bottleneck.** One meta-agent (and its sub-agents) exists per client, shared across all end-users. The `message_ids` array on a single agent row accumulates message IDs for every user. Every message operation (append, trim, clear) requires a read-modify-write of this array on the agent row, creating write contention when multiple workers process messages for different users concurrently.

**Lost message state updates under concurrent load.** Because the `message_ids` array is read-modify-written as a whole, concurrent processing of messages for different users on the same agent causes lost updates. Worker A reads `message_ids`, Worker B reads the same `message_ids`, both append their respective message IDs, and whichever writes last silently overwrites the other's changes. This is a classic lost-update anomaly. Under production load — where a single agent processes messages for hundreds of millions of users simultaneously — this means message references are silently dropped, leading to missing conversation context, orphaned message rows, and non-deterministic agent behavior.

**Eager loading hazard.** The `messages` relationship on the Agent ORM uses `lazy="selectin"`, which means loading an agent eagerly loads *all* of its messages into memory. For an agent serving millions of users, this is a ticking time bomb.

**Redundant system message storage.** The system prompt is stored twice: once in `agent.system` (a column on the agent row) and once as a `Message` row at position 0 of `message_ids`. The code reads the system prompt from the message row, enriches it with memories, and sends it to the LLM. The `agent.system` column is the source of truth but the message row is what gets used at runtime.

**Write-then-delete churn.** For the ECMS memory extraction path (the production use case), every agent step persists messages to the database, then immediately deletes them when `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` fires. Sub-agents each store their own copies of the input messages, their LLM responses, tool results, and heartbeat messages -- all of which are deleted moments later. This is pure I/O overhead.

## 2. Changes

This proposal makes six interconnected changes:

### 2.1 Store Message History In-Memory only

Today, each call to `inner_step()` persists all new messages to the database via `append_to_in_context_messages`, then the next `inner_step()` reads them back via `get_in_context_messages` to build the LLM context. For memory extraction agents, these messages are written and then mostly deleted when `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` fires.

Each sub-agent (episodic, semantic, etc.) stores **its own redundant copy** of the input messages, LLM responses, tool results, heartbeat messages and message summaries in its `message_id` property.

After a memory agent finishes, the history clearing logic resets the `message_ids` to only include: the system message & the current message-set that just processed. The heartbeats, tool calls, and any previously processed messages are cleared.

**Change:**

The `step()` loop will maintain an in-memory message list. Each `inner_step()` appends to this list instead of writing to the database. Messages are only persisted at the end of the `step()` loop, and only when the client's retention policy calls for it (see section 2.4)

This eliminates:

- All write-then-delete churn for the ECMS path
- Redundant message copies across sub-agents (each sub-agent currently stores its own copy of the input)
- The `save_agent()` call that writes `message_ids` after every step
- The `delete_detached_messages_for_agent()` cleanup

Observability is not affected: the `steps` table and LangFuse traces still capture all LLM interactions.

For chaining (multiple steps in one `step()` call), the in-memory list grows across steps within the same invocation. The LLM sees the full conversation history without any database round-trips between steps.

### 2.2 Store System Message Exclusively in Agent State

Today, the system prompt is stored as a `Message` row with `role="system"` at position 0 of `message_ids`. The code in `inner_step()` reads it back, enriches it with retrieved memories, and mutates it in-memory before sending to the LLM. The `rebuild_system_prompt` method creates a new Message row and swaps `message_ids[0]` every time the prompt changes.

**Change:**

The system prompt will live exclusively in `agent_state.system`. When building the LLM message list, `inner_step()` constructs a system `Message` object on the fly from `agent_state.system`, enriches it with memories, and prepends it. No system message is stored in the messages table.

This eliminates:

- The duplicate storage of the system prompt
- The `rebuild_system_prompt` dance of creating a new message row and swapping array positions
- The convention that `messages[0]` is always the system message (a source of fragile assumptions across the codebase)
- The `get_system_message()` method (callers read `agent_state.system` directly)

### 2.3 Remove `message_ids` From Agent

The `message_ids` JSON column on the `agents` table will be removed entirely. This is the main goal of this plan. For agent types that persist messages (retention > 0), conversation history is retrieved by querying the `messages` table directly, scoped by `(agent_id, user_id)` and ordered by `created_at`.

**Ordering strategy:** Retrieval uses `ORDER BY created_at DESC, id DESC LIMIT N` to select the newest `N` retained sets, then reverses in-memory to chronological order (oldest -> newest) before prompt assembly.

- `created_at` reflects processing order, which is what the LLM actually saw. Kafka already guarantees in-order delivery per user (messages are partitioned by `user_id`), so processing order matches real-world order.
- `id` is the tiebreaker for deterministic ordering when timestamps match (a practically impossible edge-case).

The `messages` relationship on the Agent ORM also changes from `lazy="selectin"` to `lazy="noload"` to prevent accidental eager loading.

### 2.4 Configurable Message Retention Per Client

Today, the history clearing behavior is hardcoded: after memory extraction, keep the system message + one "last edited memory item" summary + the most recent input message-set. Different clients have different needs:

- A **batch client** (like ECMS) that sends an entire conversation thread as a single `save` call has no use for retained messages. It wants `N=0`.
- An **interactive agent** that processes messages one at a time may benefit from seeing what it did in the last few invocations. It wants `N=5` or similar.

**Change:** Add a `message_set_retention_count` field to the Client model. This integer controls how many recent **input message-sets** are retained in the database after processing.

A **message-set** is defined as the input conversation payload from a single `step()` invocation. In the current ECMS path, this is typically persisted as one `messages` row whose `content` contains a packed multi-turn sequence (e.g., `[USER]... [ASSISTANT]...`). It does not include the agent's internal working messages (tool calls, tool results, heartbeats, intermediate assistant/tool chain messages). Those exist only in the in-memory accumulator during the step and are not persisted.

- `message_set_retention_count = 0` -- No messages persisted. All agent work is in-memory only. (Default for memory extraction clients.)
- `message_set_retention_count = N` -- Keep the N most recent input message-sets per `(agent_id, user_id)`. Older sets are hard-deleted.

**Runtime contract for retention changes:** retrieval and pruning both enforce `N`.

- **Read path:** when loading retained context at the start of `step()`, query only the newest `N` sets using `ORDER BY created_at DESC, id DESC LIMIT N`, then **always reverse in-memory** before prompt assembly so the LLM sees chronological order (oldest -> newest).
- **Write path:** after persisting current input set(s), hard-delete rows older than the newest `N`.
- This guarantees that changing `message_set_retention_count` at runtime takes effect on the **very next save/step** even before background cleanup completes.

When retention >= 1, the start of a `step()` invocation loads the retained input message-sets from the DB into the in-memory accumulator, giving the agent context about what it processed recently. This DB load is capped with `LIMIT N` so only the newest retained sets are considered. Persistence at end-of-step writes only the current invocation's input message-set(s), then enforces retention by hard-deleting older sets.

This replaces:

- The `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` environment variable (a global boolean)
- The hardcoded "keep system message + last edited item" behavior
- The per-agent-type branching logic that builds the "last edited memory item" summary

**Note:** The MIRIX chat agent (`chat_agent` type) is a known casualty of this change. It requires retention of full step outputs (including assistant responses, tool calls, and tool results) for conversational continuity. This will be addressed in a follow-up.

**Schema change:** Add `message_set_retention_count` (nullable Integer, default 0) to `[mirix/orm/client.py](mirix/orm/client.py)` and `[mirix/schemas/client.py](mirix/schemas/client.py)`.

### 2.5 Remove In-Loop Summarization for Memory Extraction

The in-loop summarization path is removed from this refactor scope. For the memory extraction path, if the prompt exceeds the context window, the step should fail with an explicit context-overflow error and skip memory extraction for that message. This keeps behavior simple, removes extra LLM calls, and aligns with the low expected frequency of oversized inputs.

This is not only a behavior change: dead summarization branches used by this flow should be removed as part of the refactor (retry loops, summarizer-specific branching, and unused helper calls/settings in the memory extraction path).

### 2.6 Keep "Last Edited Memory Item" but only as Ephemeral Chaining Context

The current "last edited memory item" signal is useful context for follow-up reasoning, but it should no longer be persisted as a retained `messages` row.

**Change:**

- Preserve the behavior as an **in-memory-only** helper message when another chain step is about to run.
- Do **not** write this synthetic summary to the `messages` table.
- Do **not** include it in retained message-sets (`message_set_retention_count` controls persisted input sets only).

This keeps the useful self-awareness signal for chaining while avoiding storage churn and retention pollution.

Note: `occurred_at` is **not** added to the messages table. The real-world timestamp of a conversation is already stored where it matters — on the memory records themselves (episodic events, raw memories, etc.). Message ordering uses `created_at` (processing order), which is correct because Kafka guarantees in-order delivery per user and the LLM's context should reflect what it actually saw, not a reconstructed timeline.

## 3. How It Works End-to-End

### Batch Client (message_set_retention_count = 0)

```
1. Client save request → put_messages() → Kafka
2. Worker consumes → _process_message_async() → server.send_messages()
3. Agent step starts:
   - N=0 → skip retained-set load (nothing to retrieve)
   - Construct system message from `agent_state.system`
4. LLM execution:
   - Send [system_msg, new_input] to LLM
   - Accumulate assistant/tool/intermediate messages in-memory only (not persisted)
5. Memory tool fan-out (if triggered):
   - EpisodicMemoryAgent.step():
     - Constructs system message in-memory
     - Sends [system_msg, input_copy] to LLM
     - LLM returns episodic_memory_insert(...)
     - Tool executes → writes to episodic_events table (this IS persisted)
     - Accumulates messages in-memory (not persisted)
   - SemanticMemoryAgent.step(): same pattern
6. Retention write-back:
   - `message_set_retention_count = 0` → do not persist input message-sets
   - No retention prune needed
7. No agent row updates (`message_ids` removed)
8. Kafka offset handling unchanged in Phase 1 (existing auto-commit behavior remains)
```

### Real Time Client (message_set_retention_count = 3)

```
1. Client message/request enters `step()`
2. Agent step starts:
   - Load retained input sets with `ORDER BY created_at DESC, id DESC LIMIT N`
   - Reverse in-memory to chronological order (oldest -> newest)
   - Construct system message from `agent_state.system`
3. LLM execution:
   - Send [system_msg, retained_inputs..., new_input] to LLM
   - Accumulate assistant/tool/intermediate messages in-memory only (not persisted)
4. Memory tool fan-out (if triggered):
   - Same sub-agent behavior as batch flow (memory table writes persist; message churn does not)
5. Retention write-back:
   - Persist current invocation input message-set(s) to `messages` (single row)
6. Retention prune:
   - Hard-delete rows older than newest `N` for `(agent_id, user_id)`
7. No agent row updates (`message_ids` removed)
8. Step completes with bounded retained context for next invocation
```

### MIRIX Chat Agent (known broken — follow-up)

The chat agent requires retention of full step outputs (assistant responses, tool calls, tool results) for conversational continuity. This is not supported by the input-message-set-only retention model. The chat agent will be addressed in a follow-up change.

### Context Overflow Behavior (no summarization)

If the message-sequence exceeds the model context window, the step fails with a context-overflow error. No summarization retry is attempted. The worker records the failure and proceeds according to retry/DLQ policy in Phase 2.

## 4. Edge Cases and Special Considerations

**In-memory message loss on crash.** If a worker crashes mid-step, in-memory messages are lost. For memory agents this is acceptable because retained message-sets are ephemeral in this design. For chat agents, this is a behavior change: today a crash mid-step can leave partial messages in the DB. With this change, a crash can lose the entire step's in-memory message work. Note: this PR does **not** change Kafka offset commit semantics; existing auto-commit behavior remains. Manual commit/retry guarantees are deferred to Phase 2.

**Context overflow now hard-fails extraction.** With summarization removed from scope, oversized inputs can fail memory extraction for that message. This is an accepted trade-off for Phase 1; Phase 2 retry/DLQ handling will surface these failures operationally.

**Timestamp ties.** Two messages with the same `created_at` are disambiguated by `id`. In practice, messages within a step are created sequentially and differ by microseconds. The `id` tiebreaker gives a stable order for the rare tie case.

**Chat agent is broken.** The MIRIX chat agent requires retention of full step outputs (assistant responses, tool calls, tool results) for conversational continuity. The input-message-set-only retention model does not support this. This is a known, accepted trade-off — the chat agent will be fixed in a follow-up.

**Anthropic client assumption.** The Anthropic LLM client asserts `messages[0].role == "system"`. The caller (`inner_step`) will prepend the system message before passing to the LLM client, so this assertion continues to hold. The change is that the system message comes from `agent_state.system` rather than from a DB row.

**"Last edited memory item" becomes ephemeral.** Today, the history clearing code builds a per-agent-type summary message (e.g., "Last edited memory item: [Episodic Event ID]: ...") and persists it as retained history. With configurable retention, retained history is input message-sets only. Keep this summary as optional **in-memory chaining context** only; do not persist it.

## 5. Database Migration

SQL migration steps (no Alembic dependency assumed):

1. Add `message_set_retention_count` (nullable `Integer`, default `0`) to `clients`
2. Add/adjust composite index on `(agent_id, user_id, is_deleted, created_at, id)` to `messages`
3. Delete legacy system messages (`role = 'system'`)
4. Drop `message_ids` from `agents`
5. No eager backfill/prune is required for retention-size changes: read path `LIMIT N` guarantees immediate behavior after config change; write path pruning converges storage on subsequent saves

### Migration rollout strategy (explicit)

Important: updating ORM models does **not** alter existing tables by itself. `Base.metadata.create_all` only creates missing tables; it does not add/drop columns on existing tables. Use explicit SQL migration steps for column/index changes.

Compatibility note:

- Additive schema changes are backward-compatible for old code as long as new columns are nullable or have safe defaults.
- For this plan, adding `clients.message_set_retention_count DEFAULT 0` is intentionally safe for existing clients and existing code.
- Breaking changes are contract-phase changes (e.g., dropping `agents.message_ids`) and must happen only after code cutover.

Recommended sequence:

1. **Expand schema first**
  - Add `clients.message_set_retention_count` (default `0`)
  - Add/adjust read-path index for retention queries
  - Keep `agents.message_ids` in place temporarily during this phase
2. **Ship compatible code**
  - New read path uses query-based retrieval (`DESC + LIMIT N`, then in-memory reverse)
  - New write path uses retention hard-delete
  - Do not depend on `agents.message_ids`
3. **Data cleanup**
  - Delete legacy system-message rows from `messages`
  - (Optional) one-time cleanup SQL to remove obsolete/non-retained rows if desired
4. **Contract schema**
  - Drop `agents.message_ids` only after code no longer reads/writes it anywhere
5. **Validation**
  - Existing clients with no explicit setting behave as `N=0` (default), matching current memory-extraction expectations
  - Changing `N` at runtime takes effect on first subsequent step because read path enforces `LIMIT N`

## 6. How This Sets Up Phase 2 (Kafka Durability, Idempotency, Retries)

This refactor is Phase 1. Kafka offset semantics are unchanged here (existing auto-commit remains enabled). Phase 2 will add manual Kafka offset commit, retry limits, and a dead-letter queue. The changes in this refactor are specifically designed to make Phase 2 straightforward.

### 6.1 No Partial State Left Behind

**Today's problem.** If a worker crashes mid-step, you get partial state: some messages are persisted in the messages table, some aren't; `message_ids` on the agent row may or may not have been updated; some memory inserts (episodic events, etc.) may have succeeded, others not. The Kafka offset is already auto-committed, so the message won't be retried.

**After this refactor.** A crash mid-step leaves zero message state in the DB (for retention=0 clients). The only side effects are the actual memory writes (episodic events, semantic items, etc.). When Phase 2 switches to manual Kafka offset commit, a crash means the offset isn't committed, so the message gets redelivered. The retry sees a clean slate in the messages table — no partial message state to conflict with.

### 6.2 No Agent Row Contention

**Today's problem.** Every message operation does a read-modify-write on the agent row's `message_ids`. If two workers process messages for different users on the same agent concurrently, they race on the same row. With manual Kafka commit + retries, this gets worse — a retried message could interleave with a new message's processing.

**After this refactor.** The agent row is never updated during message processing. Workers operating on different users are completely independent — they only touch the messages table, scoped by `(agent_id, user_id)`. Retries don't conflict with concurrent processing.

### 6.3 Memory Writes Become the Idempotency Boundary

With messages out of the picture, the only persistent side effects of processing a Kafka message are the actual memory writes:

- `episodic_events` table inserts
- `semantic_memory_items` table inserts
- `resource_memory_items` table inserts
- `procedural_memory_items` table inserts
- `knowledge_vault_items` table inserts

For Phase 2, these are the operations that need idempotency keys. A natural key would be the Kafka message offset + partition, or a hash of `(user_id, input_content)`. If a retry attempts to insert a memory that already exists (same idempotency key), it's a no-op.

This refactor doesn't add idempotency keys yet, but it dramatically simplifies where they need to go. Instead of needing idempotency across messages table + agent row + memory tables, you only need it on the memory tables.

## 7. Files to Modify

### Agent Execution (`[mirix/agent/agent.py](mirix/agent/agent.py)`)

The heaviest changes. Key modifications:

- `inner_step()`: construct system message from `agent_state.system`; append new messages to an in-memory list instead of calling `append_to_in_context_messages`; load `in_context_messages` from the in-memory list (for chaining) or from DB query (for first step with retention > 0)
- `step()`: maintain the in-memory message accumulator; at end of loop, check client's `message_set_retention_count` to decide whether/how much to persist; enforce retention limit by hard-deleting excess message-sets
- `_handle_ai_response()`: remove the old persistence-oriented `should_clear_history` / `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` block. Keep per-agent "last edited memory item" generation only as  in-memory chaining context (non-persistent).
- `save_agent()`: remove `message_ids` write (this function may become a no-op or be removed)
- Remove summarizer retry/in-place compression path for memory extraction flow; context-overflow should raise and fail the step.
- Remove now-unused summarizer wiring in this flow (imports, settings checks, and helper branches that only supported summarize-and-retry behavior).

### Agent Manager (`[mirix/services/agent_manager.py](mirix/services/agent_manager.py)`)

Rewrite or remove message-related methods:

- `get_in_context_messages()` -- query `messages` table by `(agent_id, user_id)`, no system message; apply `ORDER BY created_at DESC, id DESC LIMIT N` based on client `message_set_retention_count`
- `get_system_message()` -- return `agent_state.system` directly, or remove
- `append_to_in_context_messages()` -- just create message rows (no agent row update). Only called for persistence at end of step.
- `prepend_to_in_context_messages()` -- remove (no summarizer prepend path needed after this refactor)
- `set_in_context_messages()` -- remove entirely
- `trim_older_in_context_messages()` -- hard-delete older messages via query
- `trim_all_in_context_messages_except_system()` -- rename to `clear_user_messages()`, hard-delete via query
- `reset_messages()` -- hard-delete user's messages directly
- `rebuild_system_prompt()` -- just update `agent.system` column
- `_generate_initial_message_sequence()` -- no longer creates a system message row
- Remove `message_ids` from `_update_agent()` and Redis cache serialization

### Message Manager (`[mirix/services/message_manager.py](mirix/services/message_manager.py)`)

- Add `get_messages_for_agent_user(agent_id, user_id, limit=None)` -- query by `(agent_id, user_id)` with deterministic ordering; when used for retained-context load, call with `ORDER BY created_at DESC, id DESC LIMIT N`
- Add `hard_delete_user_messages(agent_id, user_id)` -- bulk hard-delete for retention pruning
- Remove `delete_detached_messages_for_agent()` and `cleanup_all_detached_messages()`

### ORM Models

- `[mirix/orm/message.py](mirix/orm/message.py)` -- Add/adjust composite index for retained-message-set queries on `(agent_id, user_id, is_deleted, created_at, id)`
- `[mirix/orm/agent.py](mirix/orm/agent.py)` -- Remove `message_ids` column; change `messages` relationship to `lazy="noload"`
- `[mirix/orm/client.py](mirix/orm/client.py)` -- Add `message_set_retention_count` column (nullable Integer, default 0)
- `[mirix/orm/sqlalchemy_base.py](mirix/orm/sqlalchemy_base.py)` -- Remove `message_ids` from Redis cache serialization

### Pydantic Schemas

- `[mirix/schemas/agent.py](mirix/schemas/agent.py)` -- Remove `message_ids` from `AgentState` and `UpdateAgent`
- `[mirix/schemas/client.py](mirix/schemas/client.py)` -- Add `message_set_retention_count` field

### LLM API Layer

- `[mirix/llm_api/anthropic_client.py](mirix/llm_api/anthropic_client.py)` -- Currently asserts `messages[0].role == "system"` and extracts it to a top-level param. Update to handle system message prepended by the caller.
- `[mirix/llm_api/anthropic.py](mirix/llm_api/anthropic.py)` -- Same pattern.

### Cleanup Managers

- `[mirix/services/user_manager.py](mirix/services/user_manager.py)` -- Replace `agent.message_ids = [agent.message_ids[0]]` with bulk message hard-delete by `user_id` (retention path)
- `[mirix/services/client_manager.py](mirix/services/client_manager.py)` -- Same, by `client_id`

### API / Client / SDK

- `[mirix/server/rest_api.py](mirix/server/rest_api.py)` -- Remove `message_ids` from `UpdateAgentRequest`
- `[mirix/server/server.py](mirix/server/server.py)` -- Update if `save_agent` changes
- `[mirix/client/client.py](mirix/client/client.py)`, `[mirix/client/remote_client.py](mirix/client/remote_client.py)`, `[mirix/local_client/local_client.py](mirix/local_client/local_client.py)`, `[mirix/sdk.py](mirix/sdk.py)` -- Remove `message_ids` references

### Tests

**Test strategy:** Two layers — unit tests (mocked, no infra) for granular method behavior, and integration tests invoking the REST API to verify end-to-end correctness.

**Run tests via:**

```bash
./scripts/run_tests_with_docker.sh --podman -s -v --log-cli-level=INFO
```

**Format/lint before committing:**

```bash
poetry run black . && poetry run isort .
```

**Files to update (rewrite in-place):**

- `[tests/test_message_handling.py](tests/test_message_handling.py)` — rewrite entirely: current tests cover `get_messages_by_ids` and `message_ids`-based `get_in_context_messages`, both of which are removed. Replace with unit tests for the new query-based retrieval methods (`get_messages_for_agent_user`, `hard_delete_user_messages`, retention pruning logic).
- `[tests/test_agent_prompt_update.py](tests/test_agent_prompt_update.py)` — rewrite in-place: remove all `message_ids[0]` assertions (system message is no longer stored as a row). Replace with assertions that `agent_state.system` holds the updated prompt and that no system message row exists in the DB.

**New unit tests to add:**

- In-memory accumulator: messages accumulate across `inner_step()` calls without DB writes
- Retention count = 0: no message rows written after `step()` completes
- Retention count = N: exactly N input message-sets retained per `(agent_id, user_id)`; older sets hard-deleted
- Context overflow: `step()` raises hard error, no summarization retry attempted
- Ephemeral "last edited memory item": present in LLM prompt when chaining, absent from DB retention rows

**New integration tests to add (REST API level):**

- `PUT /agents/{id}` system prompt update: verify `agent_state.system` updated, no system message row created
- `POST /messages` (save flow): verify retention=0 client writes no message rows; retention=N client writes and prunes correctly
- Context overflow via API: verify error response, no partial state left in DB

**Remove:**

- All tests asserting summarizer retry/compression behavior in the memory extraction flow

## 8. Chat Agent Deprecation

The `chat_agent` agent type is **deprecated** as of this refactor. It requires retention of full step outputs (assistant responses, tool calls, tool results) for conversational continuity, which is incompatible with the input-message-set-only retention model introduced here.

**Changes required in this refactor:**

- In `mirix/agent/agent.py`: at the top of `step()`, check if `agent_state.agent_type == AgentType.chat_agent` and immediately raise `NotImplementedError` with a clear message pointing to the follow-up ticket.
- In `mirix/server/server.py`: where `chat_agent` is handled (line 662), add a deprecation warning log before the `NotImplementedError` propagates.
- In `docs/ARCHITECTURE.md`: add a **Deprecated** section or callout marking `chat_agent` as unsupported pending a follow-up redesign. Explain why (retention model incompatibility) and that it will be addressed in Phase 2.
- In `mirix/schemas/agent.py`: add a comment on the `chat_agent` enum value marking it as deprecated.

**Do not remove the `chat_agent` enum value** — it is needed for backward-compatible DB reads of existing agent rows.

## 9. Instructions for Developers

After merging this refactor, developers must reset their local databases before running the server or tests. This change removes legacy `agents.message_ids` behavior and introduces new retention semantics, so existing local DB state will be incompatible.

Run `python scripts/reset_database.py` to reset your local database.
