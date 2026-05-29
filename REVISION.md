# isolate_revision

Clean upstream base (`origin/main`, `b45563a`) **+ targeted memory fixes**, with
**no dual-graph (v4/v5+) code**. 5 commits ahead of `origin/main`, 0 behind.

Intended as a minimal, reproducible test base for LoCoMo / LongMemEval memory
evaluation: upstream `main` plus only the fixes needed for correct episodic /
semantic ingest and accurate token accounting.

## What this branch adds

### 1. MAB conflict-resolution + source provenance
- ORM / schema: `episodic_memory.source_refs`, `semantic_memory.source_refs`
  and `prior_values`, `users.turn_counter` / `users.chunk_counter`.
- Deterministic semantic-insert conflict resolution, gated per meta-agent by
  `enable_conflict_resolution`; legacy free-form insert stays the default.
- Source provenance (turn_id / chunk_id / serial / occurred_at) flows from
  `/memory/add` through to stored records.
- `UserManager.reserve_source_ids`, `semantic_memory_upsert_fact` tool,
  MetaAgent prompt augmentation.
- Design docs: `docs/mab_conflict_resolution_and_provenance.md`,
  `docs/mab_raw_chunk_side_channel.md`, `docs/mab_user_id_isolation_fix.md`.

### 2. Three retrieval / ingest fixes
- **pgvector SELECT**: the episodic & semantic embedding-search built an
  explicit column list that omitted `source_refs` (+ `prior_values` on the
  semantic side). Those are non-nullable List fields, so `to_pydantic()`
  received `None` → every search threw a Pydantic `ValidationError` and
  silently returned no memories. Added them to the `select()` column lists.
  **Net effect on conv-26 / 0201c: 20.4% → 80.3%.**
- `semantic_memory_insert` indexed `item['source']` directly, so any LLM call
  that omitted `source` (common — it is the least essential field) crashed
  with `KeyError` and lost the whole item. Switched to `item.get('source', '')`.

### 3. `average_memory_tokens` "50-bug" (server + client)
- Server `list_memory_components`: old `max(1, min(limit, 200))` forced `0 → 1`
  and capped at 200, so token accounting only ever saw a 50/200-item sample.
  Now `limit <= 0` means "no limit".
- Client: old `if limit:` dropped `limit=0` as falsy, so it never reached the
  server. Now `if limit is not None`.
- Net: `average_memory_tokens` now reflects **all** memory items, not a sample.

### 4. `episodic_memory_merge` fix
- The MAB caller (`memory_tools.episodic_memory_merge`) passes
  `additional_source_ref`, but `update_event()` lacked that parameter on this
  base (it lived in the v4-graph commit, which is intentionally not included
  here) → the merge tool raised `TypeError` whenever the agent chose to merge.
  Ported the parameter + the `source_refs`-append handling from `main`.

## Verified (conv-26, no-graph, config `0201c`)
- Episodic count stable at **~113-122** across 3 runs (122 / 122 / 113);
  QA accuracy **84-88%**.
- `average_memory_tokens` uncapped (~16.7k-19.2k), counting every item.
