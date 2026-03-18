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
    content: Add message_type to Message ORM; remove message_ids from Agent ORM; change messages relationship to lazy=noload
    status: pending
  - id: schema-changes
    content: Update Pydantic schemas for Agent (remove message_ids) and Message (add message_type)
    status: pending
  - id: agent-manager-rewrite
    content: Rewrite AgentManager message methods to use query-based retrieval ordered by created_at, id
    status: pending
  - id: message-manager-updates
    content: Add query-based fetch and bulk soft-delete methods; remove detached message cleanup
    status: pending
  - id: llm-api-layer
    content: Update Anthropic and other LLM clients that assume messages[0] is the system message
    status: pending
  - id: retention-config
    content: Add message_retention_count to Client ORM/schema; implement retention enforcement at end of step()
    status: pending
  - id: cleanup-managers
    content: Replace message_ids manipulation in UserManager, ClientManager with bulk message soft-delete
    status: pending
  - id: api-client-sdk
    content: Update REST API, server, SDK, and client layers to remove message_ids references
    status: pending
  - id: ecms-sync
    content: Remove message_ids from ECMS IPSR agents entity
    status: pending
  - id: migration
    content: Create Alembic migration to add message_type, message_retention_count, indexes, soft-delete system messages, and drop message_ids
    status: pending
  - id: tests
    content: Update and add tests for new message management patterns
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

This proposal makes four interconnected changes:

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

**Ordering strategy:** `ORDER BY created_at, id`

- `created_at` reflects processing order, which is what the LLM actually saw. Kafka already guarantees in-order delivery per user (messages are partitioned by `user_id`), so processing order matches real-world order.
- `id` is the tiebreaker for deterministic ordering when timestamps match (a practically impossible edge-case).

The `messages` relationship on the Agent ORM also changes from `lazy="selectin"` to `lazy="noload"` to prevent accidental eager loading.



### 2.4 Configurable Message Retention Per Client

Today, the history clearing behavior is hardcoded: after memory extraction, keep the system message + one "last edited memory item" summary + the most recent input message-set. Different clients have different needs:

- A **batch client** (like ECMS) that sends an entire conversation thread as a single `save` call has no use for retained messages. It wants `N=0`.
- An **interactive agent** that processes messages one at a time may benefit from seeing what it did in the last few invocations. It wants `N=5` or similar.

**Change:** Add a `message_retention_count` field to the Client model. This integer controls how many recent **input message-sets** are retained in the database after processing.

A **message-set** is defined as the input messages from a single `step()` invocation — i.e., the conversation content that was sent to the agent for processing. It does not include the agent's internal working messages (assistant responses, tool calls, tool results, heartbeats). Those exist only in the in-memory accumulator during the step and are not persisted.

- `message_retention_count = 0` -- No messages persisted. All agent work is in-memory only. (Default for memory extraction clients.)
- `message_retention_count = N` -- Keep the N most recent input message-sets per `(agent_id, user_id)`. Older sets are soft-deleted.

When retention >= 1, the start of a `step()` invocation loads the retained input message-sets from the DB into the in-memory accumulator, giving the agent context about what it processed recently. If the accumulated in-memory context grows large (high N or large messages), summarization compresses older messages in the in-memory list. The post-summary state is what gets persisted at the end.

This replaces:

- The `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` environment variable (a global boolean)
- The hardcoded "keep system message + last edited item" behavior
- The per-agent-type branching logic that builds the "last edited memory item" summary

**Note:** The MIRIX chat agent (`chat_agent` type) is a known casualty of this change. It requires retention of full step outputs (including assistant responses, tool calls, and tool results) for conversational continuity. This will be addressed in a follow-up.

**Schema change:** Add `message_retention_count` (nullable Integer, default 0) to `[mirix/orm/client.py](mirix/orm/client.py)` and `[mirix/schemas/client.py](mirix/schemas/client.py)`.

### 2.5 Add `message_type` to Messages

A `message_type` column (`String`, nullable) is added to the `messages` table. It distinguishes message origins: `"original"` for normal messages, `"summary"` for messages created by the summarizer. This replaces the implicit convention of identifying summaries by their content.

Note: `occurred_at` is **not** added to the messages table. The real-world timestamp of a conversation is already stored where it matters — on the memory records themselves (episodic events, raw memories, etc.). Message ordering uses `created_at` (processing order), which is correct because Kafka guarantees in-order delivery per user and the LLM's context should reflect what it actually saw, not a reconstructed timeline.

## 3. How It Works End-to-End

### ECMS Memory Extraction (message_retention_count = 0)

```
1. POST /memory/add → put_messages() → Kafka
2. Worker consumes → _process_message_async() → server.send_messages()
3. MetaMemoryAgent.step():
   - Constructs system message in-memory from agent_state.system
   - Sends [system_msg, input_msg] to LLM
   - LLM returns trigger_memory_update(["episodic", "semantic"])
   - Accumulates assistant response + tool result in-memory (not persisted)
4. trigger_memory_update() runs sub-agents in parallel:
   - EpisodicMemoryAgent.step():
     - Constructs system message in-memory
     - Sends [system_msg, input_copy] to LLM
     - LLM returns episodic_memory_insert(...)
     - Tool executes → writes to episodic_events table (this IS persisted)
     - Accumulates messages in-memory (not persisted)
   - SemanticMemoryAgent.step(): same pattern
5. retention_count=0 → no messages written to the messages table
6. No agent row updates
7. Kafka offset committed
```

### Interactive Agent (message_retention_count = 3)

```
1. Agent processes a message via step()
2. Load retained input message-sets from DB into in-memory accumulator
   → agent sees up to 3 prior input message-sets as context
3. Construct system message from agent_state.system
4. Send [system_msg, retained_inputs..., new_input] to LLM
5. LLM responds, agent executes tools, all intermediate messages
   accumulated in-memory (not persisted)
6. At end of step(): persist the new input message-set to the messages table
7. Enforce retention: soft-delete input message-sets older than the 3 most recent
```

### MIRIX Chat Agent (known broken — follow-up)

The chat agent requires retention of full step outputs (assistant responses, tool calls, tool results) for conversational continuity. This is not supported by the input-message-set-only retention model. The chat agent will be addressed in a follow-up change.

### Summarization (retention >= 1 with high N)

When retention is high, the in-memory accumulator can grow large (retained input message-sets from prior invocations + new messages from the current step). If the token count exceeds the memory pressure threshold, summarization compresses the in-memory list:

1. Detect memory pressure after an `inner_step()` completes
2. Calculate cutoff using token counts on the in-memory list (same logic as today)
3. Send older messages from the in-memory list to the LLM summarizer
4. Replace the summarized messages in the in-memory list with a single summary message (`message_type="summary"`)
5. Continue the `step()` loop with the reduced in-memory list
6. At end of `step()`, persist the post-summary state to the DB (same as normal retention flow)

No mid-step DB writes are needed. The summarizer works entirely on the in-memory list, and the final state — including the summary message — is what gets persisted at the end.

For retention = 0, summarization never fires — there are no retained messages to accumulate, and memory extraction agents run a single step.

## 4. Edge Cases and Special Considerations

**In-memory message loss on crash.** If a worker crashes mid-step, in-memory messages are lost. For memory agents this is fine -- the messages were going to be deleted anyway, and the Kafka offset hasn't been committed (Phase 2 will add manual commit). For chat agents, this is a behavior change: today a crash mid-step leaves partial messages in the DB. With this change, a crash loses the entire step's messages. This is arguably better (no partial state) and aligns with the Kafka retry strategy in Phase 2.

**Timestamp ties.** Two messages with the same `created_at` are disambiguated by `id`. In practice, messages within a step are created sequentially and differ by microseconds. The `id` tiebreaker gives a stable order for the rare tie case.

**Chat agent is broken.** The MIRIX chat agent requires retention of full step outputs (assistant responses, tool calls, tool results) for conversational continuity. The input-message-set-only retention model does not support this. This is a known, accepted trade-off — the chat agent will be fixed in a follow-up.

**Anthropic client assumption.** The Anthropic LLM client asserts `messages[0].role == "system"`. The caller (`inner_step`) will prepend the system message before passing to the LLM client, so this assertion continues to hold. The change is that the system message comes from `agent_state.system` rather than from a DB row.

**Removal of "last edited memory item" summary.** Today, the history clearing code builds a per-agent-type summary message (e.g., "Last edited memory item: [Episodic Event ID]: ...") and keeps it as the sole retained message. With configurable retention, a client using `message_retention_count=1` retains the raw messages from the last invocation, which contain the same information in the tool results. The synthetic summary construction (lines 1273-1391 in `agent.py`) is removed. If the synthetic summary format is specifically needed, it can be reintroduced as an optional post-processing step, but the raw tool results are arguably more useful since they contain the full structured data.

## 5. How This Sets Up Phase 2 (Kafka Durability, Idempotency, Retries)

This refactor is Phase 1. Phase 2 will add manual Kafka offset commit, retry limits, and a dead-letter queue. The changes in this refactor are specifically designed to make Phase 2 straightforward.

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

## 6. Files to Modify

### Agent Execution (`[mirix/agent/agent.py](mirix/agent/agent.py)`)

The heaviest changes. Key modifications:

- `inner_step()`: construct system message from `agent_state.system`; append new messages to an in-memory list instead of calling `append_to_in_context_messages`; load `in_context_messages` from the in-memory list (for chaining) or from DB query (for first step with retention > 0)
- `step()`: maintain the in-memory message accumulator; at end of loop, check client's `message_retention_count` to decide whether/how much to persist; enforce retention limit by soft-deleting excess message-sets
- `_handle_ai_response()`: remove the entire `should_clear_history` / `CLEAR_HISTORY_AFTER_MEMORY_UPDATE` block and the per-agent-type "last edited memory item" logic. Retention is now handled uniformly at the end of `step()` based on the client config.
- `save_agent()`: remove `message_ids` write (this function may become a no-op or be removed)
- `summarize_messages_inplace()`: rewrite to operate on the in-memory message list directly instead of reading/writing the DB. Calculate cutoff, call LLM summarizer, replace old messages with summary message in the list. No DB operations mid-step.

### Agent Manager (`[mirix/services/agent_manager.py](mirix/services/agent_manager.py)`)

Rewrite or remove message-related methods:

- `get_in_context_messages()` -- query `messages` table by `(agent_id, user_id)`, no system message
- `get_system_message()` -- return `agent_state.system` directly, or remove
- `append_to_in_context_messages()` -- just create message rows (no agent row update). Only called for persistence at end of step.
- `prepend_to_in_context_messages()` -- remove (summarizer creates messages directly; no need to "prepend" when ordering is by `created_at`)
- `set_in_context_messages()` -- remove entirely
- `trim_older_in_context_messages()` -- soft-delete older messages via query
- `trim_all_in_context_messages_except_system()` -- rename to `clear_user_messages()`, soft-delete via query
- `reset_messages()` -- soft-delete user's messages directly
- `rebuild_system_prompt()` -- just update `agent.system` column
- `_generate_initial_message_sequence()` -- no longer creates a system message row
- Remove `message_ids` from `_update_agent()` and Redis cache serialization

### Message Manager (`[mirix/services/message_manager.py](mirix/services/message_manager.py)`)

- Add `get_messages_for_agent_user(agent_id, user_id)` -- query ordered by `created_at, id`
- Add `soft_delete_user_messages(agent_id, user_id)` -- bulk soft-delete
- Remove `delete_detached_messages_for_agent()` and `cleanup_all_detached_messages()`

### ORM Models

- `[mirix/orm/message.py](mirix/orm/message.py)` -- Add `message_type` column; add composite index on `(agent_id, user_id, is_deleted)`
- `[mirix/orm/agent.py](mirix/orm/agent.py)` -- Remove `message_ids` column; change `messages` relationship to `lazy="noload"`
- `[mirix/orm/client.py](mirix/orm/client.py)` -- Add `message_retention_count` column (nullable Integer, default 0)
- `[mirix/orm/sqlalchemy_base.py](mirix/orm/sqlalchemy_base.py)` -- Remove `message_ids` from Redis cache serialization

### Pydantic Schemas

- `[mirix/schemas/agent.py](mirix/schemas/agent.py)` -- Remove `message_ids` from `AgentState` and `UpdateAgent`
- `[mirix/schemas/message.py](mirix/schemas/message.py)` -- Add `message_type` field
- `[mirix/schemas/client.py](mirix/schemas/client.py)` -- Add `message_retention_count` field

### LLM API Layer

- `[mirix/llm_api/anthropic_client.py](mirix/llm_api/anthropic_client.py)` -- Currently asserts `messages[0].role == "system"` and extracts it to a top-level param. Update to handle system message prepended by the caller.
- `[mirix/llm_api/anthropic.py](mirix/llm_api/anthropic.py)` -- Same pattern.

### Cleanup Managers

- `[mirix/services/user_manager.py](mirix/services/user_manager.py)` -- Replace `agent.message_ids = [agent.message_ids[0]]` with bulk message soft-delete by `user_id`
- `[mirix/services/client_manager.py](mirix/services/client_manager.py)` -- Same, by `client_id`

### API / Client / SDK

- `[mirix/server/rest_api.py](mirix/server/rest_api.py)` -- Remove `message_ids` from `UpdateAgentRequest`
- `[mirix/server/server.py](mirix/server/server.py)` -- Update if `save_agent` changes
- `[mirix/client/client.py](mirix/client/client.py)`, `[mirix/client/remote_client.py](mirix/client/remote_client.py)`, `[mirix/local_client/local_client.py](mirix/local_client/local_client.py)`, `[mirix/sdk.py](mirix/sdk.py)` -- Remove `message_ids` references

### ECMS

- `[context-and-memory-service/common/ipsr/entities/agents.py](context-and-memory-service/common/ipsr/entities/agents.py)` -- Remove `message_ids` from IPSR agents entity

### Database Migration

Alembic migration:

1. Add `message_type` (nullable `String`) to `messages`
2. Add `message_retention_count` (nullable `Integer`, default `0`) to `clients`
3. Add composite index on `(agent_id, user_id, is_deleted)` to `messages`
4. Soft-delete existing system messages (`role = 'system'`)
5. Drop `message_ids` from `agents`

### Tests

- `[tests/test_message_handling.py](tests/test_message_handling.py)` -- Update for query-based retrieval
- `[tests/test_agent_prompt_update.py](tests/test_agent_prompt_update.py)` -- Update for system message from `agent_state.system`
- New tests for in-memory accumulation, in-memory summarization, and retention count behavior (0, N)

