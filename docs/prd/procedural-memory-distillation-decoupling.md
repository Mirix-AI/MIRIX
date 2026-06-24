# PRD: Decouple Procedural-Memory (Skill) Distillation from the Meta Agent Transcript

> Status: ready-for-agent
> Area: memory ingestion, procedural memory / skill evolution
> Ticket prefix: `[VEPEAGE-NNN]`

## Problem Statement

When a caller feeds a conversation into MIRIX, that conversation is injected **directly into the `meta_memory_agent`'s own message thread**. The meta agent then does its job — it emits a `[System Message]` bootstrap, calls `trigger_memory_update`, receives tool results, and emits `continue_chaining` heartbeats. Every one of those internally-generated messages inherits the **same `session_id`** as the real conversation, and lives under the **same `agent_id`**, in the **same `messages` table**.

Skill learning (the procedural-memory distillation we added) reads that thread back by `(agent_id, session_id)`. Because the real conversation and the meta agent's own bookkeeping are indistinguishable at the data layer, the distiller learns from MIRIX operating *itself*. The symptom the user caught: skills such as *"Trigger episodic memory update when the system requests meta memory management"* — self-referential noise that is not a real, transferable skill.

The root cause, in the user's words: **`session_id` is bound to the agent that *makes* the memory, when it should identify the *conversation being remembered*.** Two further smells fall out of the same root:

- Message **retention was bolted onto the meta agent** (retain last N sessions) purely so the distiller would have a transcript to read — giving the memory-production machinery a storage/retention burden it never should have had.
- There is **no clean gate**: a caller that does not supply a `session_id` still flows through the same path, silently producing nothing learnable, with no explicit contract.

The current mitigation is a Python heuristic (`_is_mirix_scaffolding`) that string-matches and drops scaffolding rows at distill time. It works today but is **convention-driven** — it depends on every external turn carrying `[USER]`/`[ASSISTANT]` tags and on recognizing memory-tool names. A future change to the agent loop can silently reintroduce the leak.

## Solution

Give the real external conversation **its own home**, separate from the memory-production machinery.

When a caller adds memory **with a `session_id`**, MIRIX persists that conversation — with its **real `user`/`assistant` roles preserved** — into a dedicated **Conversation Message Store**. That store is the **single source of truth for skill learning**. The procedural-memory distiller reads **only** that store, so it can never see the meta agent's internal messages.

The `meta_memory_agent` and its six sub-agents go back to being **pure, transient memory producers**: their working messages carry **no `session_id`** and are **discarded after each extraction** (the original MIRIX behavior, `retain=0`). The memory-production machinery owns no durable conversation record anymore.

Skills are distilled in **rolling, sealed batches**: a session is "sealed" once a newer distinct `session_id` has appeared after it. When the store holds **6 distinct sessions**, the **5 oldest (sealed)** sessions are each distilled — independently — into experiences and then skills, and marked as distilled. The 6th remains as the open head of the next window.

If a caller **omits the `session_id`**, the other **five memory components** (`core`, `episodic`, `semantic`, `resource`, `knowledge_vault`) still extract normally — but **no procedural/skill memory is produced**, because procedural memory now flows exclusively through the Conversation Message Store distillation path.

## User Stories

1. As a developer integrating MIRIX in production, I want to pass a `session_id` when I add a conversation, so that MIRIX records that conversation as a discrete, learnable unit.
2. As an integrating developer, I want the skills MIRIX learns to come only from the conversations I actually fed it, so that retrieved skills never describe MIRIX's own internal bookkeeping.
3. As an integrating developer, I want to omit `session_id` for lightweight ingestion and still get the five non-procedural memory components, so that I am not forced into skill learning for every call.
4. As an integrating developer, I want a clear contract that "no `session_id` means no procedural memory," so that the behavior of my integration is predictable.
5. As an integrating developer, I want the conversation stored with its real `user`/`assistant` roles, so that distilled skills reflect the true turn structure rather than a flattened blob.
6. As an integrating developer retrieving skills, I want them attributed to a coherent conversation, so that I can trace a skill back to where it was learned.
7. As a MIRIX maintainer, I want the meta agent's transcript to be transient again, so that the memory-production path carries no storage or retention burden.
8. As a MIRIX maintainer, I want skill distillation isolated behind a dedicated store, so that a future change to the agent loop cannot reintroduce scaffolding leakage.
9. As a MIRIX maintainer, I want to delete the `_is_mirix_scaffolding` heuristic, so that correctness comes from structure (which table we read) rather than from fragile string conventions.
10. As a MIRIX maintainer, I want `session_id` to identify a conversation and never be stamped on the meta agent's synthesized messages, so that the identifier means one thing.
11. As the system, I want to distill the five oldest *sealed* sessions when a sixth distinct session appears, so that only completed conversations are learned and the in-progress one is left alone.
12. As the system, I want each session distilled independently, so that experiences and skills attribute to a single coherent conversation.
13. As the system, I want already-distilled sessions marked so they are not re-distilled, so that the rolling barrier advances and does not reprocess history.
14. As the system, I want the conversation store scoped per `(user, organization)`, so that one user's sessions never count toward or leak into another's learning window.
15. As the system, when a caller omits `session_id`, I want to skip the conversation-store write and the procedural distillation entirely, so that no orphaned or NULL-keyed learning state is created.
16. As the system, when a caller omits `session_id`, I want the five non-procedural components to extract exactly as before, so that the core product is unaffected by the procedural gating.
17. As an eval engineer running the `mirix-generic` MetaClaw arm, I want learning to be driven purely by `session_id`-bearing add calls and a barrier, so that the arm mirrors production with no eval-only interface.
18. As an eval engineer, I want the barrier to distill sealed sessions from the store, so that the "every 5 turns" cadence maps cleanly onto "every 5 sealed sessions."
19. As a desktop chat user, I want my chat conversation to be retained as it is today, so that the refactor does not regress the desktop assistant.
20. As a MIRIX maintainer, I want the meta agent to dispatch to only the five non-procedural sub-agents during normal extraction, so that procedural memory has exactly one producer (the distiller).
21. As a MIRIX maintainer, I want the skill-experience store and evolution chain unchanged, so that the refactor swaps only the *source* of distillation, not the learning algorithm.
22. As a MIRIX maintainer, I want a clear test that the meta agent's messages carry no `session_id` and are not retained after a memory update, so that the decoupling is verifiable.
23. As a MIRIX maintainer, I want a clear test that the distiller never reads meta agent messages, so that the isolation is enforced by the test suite, not just by code review.
24. As an integrating developer, I want to add a conversation with the same `session_id` across multiple calls and have its turns accumulate into one session, so that multi-turn conversations are a single learnable unit.
25. As the system, I want the trigger/cadence logic to count distinct sealed sessions in the conversation store rather than in the messages table, so that the count reflects real conversations only.
26. As a MIRIX maintainer, I want the `messages.session_id` column kept as generic infrastructure, so that the refactor does not force a destructive schema revert.
27. As an integrating developer, I want skills produced after my 5th sealed conversation without me calling any special endpoint, so that learning is automatic in production.
28. As an eval engineer, I want an explicit barrier endpoint to force distillation of sealed sessions on demand, so that the blocking-barrier eval flow remains supported.
29. As a MIRIX maintainer, I want the procedural gating to be observable in `/health` or a similar surface, so that an operator can confirm whether procedural learning is active for a deployment.
30. As a MIRIX maintainer, I want a smoke test proving the "trigger memory update" class of self-referential noise no longer appears in produced skills, so that the original bug is provably closed.

## Implementation Decisions

### New domain concept: Conversation Message Store

- A new persisted entity — the **Conversation Message** — is the canonical, learnable record of an external conversation turn. It is **separate** from the `messages` table that backs the agent loop.
- It is written **only** when an add request carries a `session_id`. It is the **only** source the procedural-memory distiller reads.
- Real roles are preserved (`user` / `assistant`), not the `[USER]`/`[ASSISTANT]` role-collapsed form the meta agent receives.
- Scoped by `(session_id, user, organization)`. A "sealed/distilled" marker tracks which sessions have been consumed.

Schema shape (encodes the decision; final column set may be refined):

```
conversation_message
  id            (pk)
  session_id    (indexed, NOT NULL — this store only holds session'd turns)
  user_id       (indexed)
  organization_id
  role          ('user' | 'assistant')
  content       (text)
  created_at    (indexed; MIN(created_at) per session defines session order)
  distilled_at  (nullable; set when the session has been consumed by a distill round)
```

### Ingestion seam

- The memory-add API (async and sync) gains a single branch: **if `session_id` is present**, write the normalized external turns to the Conversation Message Store, **independently of** dispatching the content to the `meta_memory_agent`.
- The meta agent dispatch is **unchanged in mechanism** but **gated in scope**: it dispatches to the **five non-procedural** sub-agents always; procedural memory is no longer produced via the meta dispatch.

### Cadence / barrier

- The trigger counts **distinct sealed sessions** in the Conversation Message Store. "Sealed" = a strictly newer distinct `session_id` exists.
- When the store holds **6 distinct sessions**, the **5 oldest sealed** sessions are each distilled (rolling). Distilled sessions are marked; the window advances. The same logic backs both the automatic in-band trigger and the explicit barrier endpoint used by the eval arm.

### Meta agent reverts to original behavior

- Meta agent message retention returns to `retain=0` (revert the Goal-1 last-N-sessions retention for the meta agent).
- The meta agent and its sub-agents **stop stamping `session_id`** on synthesized messages (revert the Goal-1 inheritance for non-chat agents). Their working messages remain transient and are discarded after each extraction.
- Context reconstruction is unaffected: the agent loop reconstructs context by `message_ids`, never by `session_id`.

### Procedural memory has exactly one producer

- Procedural / skill memory is produced **exclusively** by the Conversation Message Store distillation path.
- Without a `session_id`, no conversation is stored, the distiller is not triggered, and **no procedural memory is produced** — while the other five components extract normally.

### Distiller source swap

- The procedural-memory distiller's session enumeration and per-session fetch read from the **Conversation Message Store**, not from the meta agent's `messages`.
- The `_is_mirix_scaffolding` heuristic is **removed** — it is dead once the source is a clean store.
- The skill-experience store, the experience→skill evolution, and procedural-memory retrieval are **unchanged**; only the input source changes.

### Retained infrastructure

- The `messages.session_id` column is **kept** (generic infrastructure; the chat path may still use it). The memory-learning path simply no longer depends on it.

## Testing Decisions

A good test here asserts **external behavior** — what skills get produced, what the gate does, what the meta agent leaves behind — never internal call counts or private method shapes.

### Primary seam (highest, preferred — ideally the only one)

- The **memory HTTP API** (`/memory/add`, `/memory/add_sync`) plus the **skills retrieval** (`/v1/skills`). This is the production-faithful seam and the same one the `mirix-generic` eval arm already drives.
- Behaviors to assert end-to-end:
  - Drive ≥6 sessions **with** `session_id` → skills are produced, and contain **zero** self-referential / scaffolding noise (assert the absence of the "trigger memory update / meta memory manager" class).
  - Drive add **without** `session_id` → the five non-procedural components extract, **no** procedural/skill memory appears, and nothing is written to the Conversation Message Store.
  - After a memory update, the meta agent's messages have **NULL `session_id`** and are **not retained**.
  - Multiple add calls sharing one `session_id` accumulate into a single distilled session.

### Supporting seam (only if the primary cannot cover it)

- The new **Conversation Message manager**, tested the way `session_id` infrastructure is already tested: a DB-free shape/contract test (mirroring the existing `session_id` unit tests) plus a Postgres round-trip integration test (mirroring the existing `session_id` integration test). Covers: create turns, count distinct sealed sessions, select the oldest-5 sealed, mark-distilled idempotency, and per-`(user, org)` scoping.

### Updated / removed tests

- The distillation tests are re-pointed to the new source. The scaffolding-filter tests are **removed** (the heuristic is gone) and replaced by a test asserting the distiller **never reads meta agent messages**.
- The `TestAgentStepPropagation` lint — which currently *enforces* that synthesized messages inherit `session_id` — is **inverted** to assert that non-chat agents' synthesized messages carry **no** `session_id`.

### Prior art

- `tests/test_memory_integration.py`, `tests/test_memory_server.py` (API-seam behavior).
- `tests/test_session_experience_distillation.py` (distiller behavior).
- `tests/test_session_id.py`, `tests/test_session_id_integration.py` (manager shape + Postgres round-trip patterns).

## Out of Scope

- **Backfilling** existing retained meta agent transcripts into the new Conversation Message Store. Old retained rows are left to age out / be cleaned up; historical skills are not regenerated.
- **Desktop `chat_agent` learning.** The chat agent is unaffected and is **not** a distillation source in this PRD; whether chat conversations should later feed skill learning is a separate decision.
- **Reworking the five non-procedural sub-agents** or their extraction quality.
- **Removing the `messages.session_id` column** or otherwise reverting the schema introduced for it.
- **Changing the experience→skill evolution algorithm**, scoring, or retrieval ranking.
- **Rewriting the `mirix-generic` eval arm trace documentation** (follow-up; the arm itself keeps working).

## Further Notes

- **Decision to confirm with the developer:** this PRD treats the legacy `procedural_memory_agent` sub-agent as **retired from the normal meta dispatch** — procedural memory becomes solely the distillation path. If instead the legacy procedural sub-agent should be *retained but ungated when `session_id` is absent*, that is a one-line scope change; flagged here because it materially changes "what produces procedural memory."
- **"Sealed" definition** (a newer distinct `session_id` exists) is what lets the rolling barrier avoid distilling an in-progress conversation. The eval arm's per-turn `session_id` makes every completed turn immediately sealed.
- Per `.claude/rules/agent_dev.md`, the implementation must be **reviewed by codex** after coding, with fixes applied by priority.
- Per the project async rules, all new manager methods are `async def`; no `asyncio.run()` inside the server; wrap any unavoidable sync calls with `asyncio.to_thread()`.
- Publication note: no project issue tracker is configured in this environment, so this PRD is filed as a repository document. It can be pushed to a tracker (with the `ready-for-agent` label) once a tracker is provided.
