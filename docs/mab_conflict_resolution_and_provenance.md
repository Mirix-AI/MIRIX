# Patch note: conflict resolution + source provenance for semantic and episodic memory

**Status.** Opt-in at meta-agent **create time** for the conflict
resolution policy section; source provenance is **always-on** server-side
(no opt-in needed). Default behaviour is preserved: existing user flows
keep their semantics, existing items get `source_refs=[]` /
`prior_values=[]` after migration and stay opaque to the new paths.

**Scope.**

- Three new persisted columns (`semantic_memory.source_refs`,
  `semantic_memory.prior_values`, `episodic_memory.source_refs`).
- Two new per-user counter columns (`users.turn_counter`,
  `users.chunk_counter`).
- A new manager method `SemanticMemoryManager.upsert_with_conflict_resolution`,
  an auto-route in the existing `insert_semantic_item`, and an
  `additional_source_ref` parameter on `EpisodicMemoryManager.update_event`.
- A server-side helper `_augment_source_meta_with_server_fallbacks`
  invoked from both `/memory/add` entry points.
- A `UserManager.reserve_source_ids` helper that atomically bumps the
  per-user counters.
- One ~1 KB prompt section appended to the semantic agent's stored
  system prompt when `enable_conflict_resolution=True` is passed to
  `create_meta_agent`.

No new tool, no new validator, no `semantic_memory_*` tool list change,
no `update_meta_agent` rewiring.

## The problem

MIRIX's `semantic_memory_agent` resolves conflicts using LLM free-text
merge with no notion of recency, version, or provenance:

- Multiple conflicting facts about an entity collapse into one
  `summary` / `details` string. FactConsolidation entries like
  `0. Thomas Kyd was born in London` and
  `306. Thomas Kyd was born in Leeds` become
  `"Thomas Kyd was born in London, though some data says Leeds"`.
- The merging LLM uses its world-knowledge prior as a tie-breaker, often
  marking the dataset's authoritative value as
  `"conflicting"` / `"incorrectly attributed"` / `"erroneously"`.
- After delete-then-insert, the old value is gone — no audit trail.

Three concrete cases caught from `prompt_debug` in a prior run:

  - `Thomas Kyd born in` → MIRIX summary said `London`, suppressed `Leeds`.
  - `Japan official language` → kept `Japanese`, dropped `Swedish`.
  - `Microsoft CEO` → kept `Satya Nadella`, dropped `Steve Jobs`.

Correct behaviour for a personal assistant. Wrong for any system that
needs to honour the user's most recent statement when it contradicts
world knowledge, or to recall *when* a fact was first / last asserted.

## The change

Two independent but cooperating mechanisms:

### A. Source provenance (always-on, general)

Every `/memory/add` request — whether from the MAB adapter, a personal
assistant SDK, or any other caller — ends up with a `filter_tags["source_meta"]`
dict carrying at least `turn_id`, `chunk_id`, and `occurred_at`. The
fields the caller already supplied win; the server fills in the rest.

The pipeline:

1. **Client may pre-fill** any subset of
   `filter_tags["source_meta"] = {turn_id, chunk_id, serial, occurred_at}`.
   - The MAB adapter populates `chunk_id`, `serial_first`, `serial_last`
     because it knows the chunk's internal structure.
   - A personal-assistant SDK can leave it empty.
2. **Server `/memory/add` merges with fallbacks** via
   `_augment_source_meta_with_server_fallbacks`:
   - `turn_id` missing → call `UserManager.reserve_source_ids(n_turns)`
     and use `turn_id_start`.
   - `chunk_id` missing → use the reservation's `chunk_id`.
   - `occurred_at` missing → use the request's top-level `occurred_at`
     if present, else wall-clock UTC ISO 8601.
   - `serial` is never auto-filled; it stays present iff the caller put
     it there (FactConsolidation-style numbered input).
3. **Counters are persisted** in two new `users` columns
   (`turn_counter`, `chunk_counter`), bumped atomically by
   `reserve_source_ids`.
4. **`SemanticMemoryManager.insert_semantic_item` copies `source_meta`
   into `source_refs`** when the auto-route fires (see B below).
5. **`EpisodicMemoryManager.insert_event` copies `source_meta` into
   `source_refs`** unconditionally — every event gets a provenance trail,
   not just CR-eligible ones.
6. **`EpisodicMemoryManager.update_event` accepts
   `additional_source_ref`** so `episodic_memory_merge` can append the
   current ingest's pointer onto an already-existing event's
   `source_refs`. Multi-batch events therefore preserve every chunk
   that contributed.

### B. Conflict resolution (opt-in at create time)

A second path through the existing `semantic_memory_insert` call,
selected once at meta-agent create:

1. **Schema.** `semantic_memory.source_refs JSON NOT NULL DEFAULT '[]'`
   and `semantic_memory.prior_values JSON NOT NULL DEFAULT '[]'`.
   Legacy rows default to empty and stay opaque.
2. **Manager.** `SemanticMemoryManager.upsert_with_conflict_resolution(
   entity, relation, value, source_ref, ...)` does deterministic merge:
   priority `occurred_at > serial > created_at`, newer wins as
   `summary`, older goes into `prior_values` with status
   `superseded` (or `corrected` when the caller asks).
3. **Auto-route.** `SemanticMemoryManager.insert_semantic_item` checks
   the incoming `filter_tags["source_meta"]` dict. If it is present AND
   `name` is shaped like `"<entity> / <relation>"`, the call is
   forwarded to `upsert_with_conflict_resolution`. Otherwise the
   legacy free-form path runs unchanged.
4. **Prompt.** When `enable_conflict_resolution=True` is passed to
   `create_meta_agent`, a ~1 KB policy section is appended to the
   semantic agent's stored system prompt. The section tells the agent
   to write facts as `name="<entity> / <relation>", summary=<value>`,
   verbatim, no hedging.

The conflict-resolution path is selected **once**, at meta-agent
create. When off, the policy section is not in the prompt and the agent
never writes the triple-shape names, so the auto-route in
`insert_semantic_item` never fires.

## Files touched

| File | Change |
| --- | --- |
| `mirix/orm/user.py` | + `turn_counter INT NOT NULL DEFAULT 0`, + `chunk_counter INT NOT NULL DEFAULT 0` |
| `mirix/orm/semantic_memory.py` | + `source_refs JSON NOT NULL DEFAULT '[]'`, + `prior_values JSON NOT NULL DEFAULT '[]'` |
| `mirix/orm/episodic_memory.py` | + `source_refs JSON NOT NULL DEFAULT '[]'` |
| `mirix/schemas/user.py` | Surface `turn_counter` + `chunk_counter` on `User` |
| `mirix/schemas/semantic_memory.py` | Surface both new fields on `SemanticMemoryItem` + `SemanticMemoryItemUpdate` |
| `mirix/schemas/episodic_memory.py` | Surface `source_refs` on `EpisodicEvent` + `EpisodicEventUpdate` |
| `mirix/services/user_manager.py` | + `reserve_source_ids(user_id, n_turns)` — atomic counter bump used by the `/memory/add` fallback. |
| `mirix/services/semantic_memory_manager.py` | + `upsert_with_conflict_resolution(...)`, + `_find_by_entity_relation`, + `_build_cr_filter_tags`, + `_source_ref_key`; auto-route inside `insert_semantic_item`. |
| `mirix/services/episodic_memory_manager.py` | `insert_event` copies `filter_tags["source_meta"]` into the new `source_refs` column; `update_event` accepts `additional_source_ref` so merge appends the current ingest's pointer. |
| `mirix/agent/meta_agent.py` | + module-level `_CONFLICT_RESOLUTION_POLICY_PROMPT` (~1 KB). No flag plumbing through the class. |
| `mirix/schemas/agent.py` | + `enable_conflict_resolution: bool = False` on `CreateMetaAgent` only |
| `mirix/services/agent_manager.py` | `create_meta_agent`: when the flag is set, append the policy section to the semantic agent's stored system prompt at creation time |
| `mirix/server/rest_api.py` | + `_augment_source_meta_with_server_fallbacks(...)` helper, called from both `/memory/add` and `/memory/add_sync`. Pass `enable_conflict_resolution` from `meta_agent_config` into `CreateMetaAgent`. |
| `mirix/functions/function_sets/memory_tools.py` | `episodic_memory_merge` reads `self.filter_tags["source_meta"]` and forwards it to `update_event` as `additional_source_ref`. |
| `scripts/migrate_add_provenance_columns.py` | One-shot, idempotent ALTER TABLE for the five new columns. Safe to re-run. |
| `samples/memoryagentbench/mirix_adapter.py` | + `mirix_enable_conflict_resolution` YAML key; ingest sends `filter_tags={"source_meta": {chunk_id, serial_first, serial_last}}` and ISO-8601 `occurred_at`. `update_agents=False` — flag is set at create only. |
| `samples/memoryagentbench/run_bench.py` | + `--enable-conflict-resolution` / `--no-enable-conflict-resolution` CLI flag |
| `samples/memoryagentbench/run_ablation.py` | Forward `--enable-conflict-resolution` to every spawned `run_bench` |

Nothing was changed in `mirix/agent/tool_validators.py` or
`mirix/constants.py`'s `SEMANTIC_MEMORY_TOOLS`.

## Data model

`SemanticMemoryItem.source_refs : list[dict]` — provenance pointers for
the current value. Each entry is a small dict; any subset of
`{turn_id, chunk_id, serial, occurred_at}` may be present.

`SemanticMemoryItem.prior_values : list[dict]` — values that used to be
canonical. Shape:

```python
[
    {
        "value": str,                       # the prior canonical value
        "source_refs": list[dict],          # provenance for that prior value
        "status": "superseded"              # OR "corrected"
                | "corrected",
        "moved_at": "2026-05-15T01:00:00",  # when the demotion happened
        "note": Optional[str],              # e.g. "late-arrived older fact"
    },
]
```

`EpisodicEvent.source_refs : list[dict]` — same shape as on semantic.

All three columns are non-null with `default=list` / `DEFAULT '[]'`.
Legacy items written before this change get `[]` after the migration.

## Deterministic ordering

`SemanticMemoryManager._source_ref_key(source_ref) -> tuple` produces a
lexicographic sort key:

```python
return (
    1 if occurred_at else 0, occurred_at,
    1 if serial is not None else 0, serial if serial is not None else -1,
    1 if created_at else 0, created_at,
)
```

Priority: `occurred_at > serial > created_at`. The MAB adapter sets all
three where available (`occurred_at = now_iso8601()`, `serial =
serial_last` extracted from the chunk text, `created_at` fills in at
DB write).

## Auto-route inside `insert_semantic_item`

The behaviour change is fully contained in one method:

```python
async def insert_semantic_item(self, ..., name, summary, filter_tags=None, ...):
    source_meta = (filter_tags or {}).get("source_meta")
    if source_meta and isinstance(name, str) and " / " in name:
        entity, _, relation = name.partition(" / ")
        if entity.strip() and relation.strip():
            return await self.upsert_with_conflict_resolution(
                entity=entity.strip(),
                relation=relation.strip(),
                value=summary,
                source_ref=dict(source_meta),
                extra_filter_tags={k: v for k, v in (filter_tags or {}).items()
                                   if k != "source_meta"},
                ...
            )
    # legacy free-form path unchanged
    ...
```

Two conditions both need to hold to enter the conflict-resolution path:

1. The caller passed `filter_tags["source_meta"]` (the MAB adapter, the
   only caller that knows what source the input came from, does this
   when its `mirix_enable_conflict_resolution` flag is on).
2. The agent put `" / "` in `name` (the policy section in the system
   prompt teaches it to do this for triple-shaped facts).

If either condition is missing, the legacy `insert_semantic_item`
path runs as before. This means:

- Concept items the agent writes without the triple shape
  (`name="Crystal chandelier care"`) → legacy path, unchanged.
- Triple-shaped items written without source provenance (no adapter, no
  flag) → legacy path, unchanged.
- Both present → conflict-resolution path.

## Agent prompt section

When `enable_conflict_resolution=True` is passed to `create_meta_agent`,
`agent_manager.create_meta_agent` appends
`_CONFLICT_RESOLUTION_POLICY_PROMPT` to the semantic agent's stored
system prompt at creation time. The section, in full:

```
## Conflict resolution policy

When ingesting a fact that asserts a value for some (entity, relation)
already covered by an existing semantic item:

- DO NOT merge the new value into a hedging free-text summary
  ("X, though some data says Y", "incorrectly attributed",
  "according to some sources").
- DO NOT use your own world knowledge to choose which value is "correct".
- The user's most recent assertion is authoritative.

When you call semantic_memory_insert for a fact of this shape, write:

  - name: "<entity> / <relation>" (e.g. "Thomas Kyd / born in")
  - summary: the raw value, verbatim (e.g. "Leeds"). No paraphrasing.
  - details: short context only.

The manager will then preserve any prior canonical value with a
"superseded" marker in prior_values based on source ordering — you
do not have to hedge in summary to keep the old value safe.

For free-form concepts that do not fit a triple shape (multi-paragraph
how-tos, abstract topics), keep calling semantic_memory_insert
normally; the manager will route those down the legacy free-form path
unchanged.
```

When `enable_conflict_resolution=False`, this section is never emitted;
the semantic agent sees the unaltered base prompt and behaves exactly
as before. No tool changes, so the agent's tool list is identical in
both modes.

## How to enable

### Python client / SDK

```python
client = await MirixClient.create(...)
await client.initialize_meta_agent(
    config={
        "llm_config": {...},
        "embedding_config": {...},
        "meta_agent_config": {
            "agents": [...],
            "enable_conflict_resolution": True,   # <-- create-time only
        },
    },
)
```

If a meta-agent already exists for this client, you must delete and
re-create to switch the flag.

### MAB adapter (YAML)

```yaml
mirix_enable_conflict_resolution: true
```

### MAB adapter (CLI override)

```
python samples/memoryagentbench/run_bench.py ... --enable-conflict-resolution
python samples/memoryagentbench/run_ablation.py ... --enable-conflict-resolution
```

### Migration

```bash
python scripts/migrate_add_provenance_columns.py
```

Run once per Postgres deployment. Idempotent. Existing rows get
`source_refs=[]`, `prior_values=[]`. No backfill.

## Validation

Three things were verified end-to-end against the running server:

### 1. Server-side source_meta fallback (always-on path)

Two consecutive `/memory/add` calls for a fresh user with no
`filter_tags` on the request side at all. Result:

```
ep_TY7K  "User went hiking at Mt Rainier ..."
         source_refs = [{"chunk_id": 0, "turn_id": 0, "occurred_at": "...09:24:49..."}]

ep_OCEO  "User went to a yoga class downtown ..."
         source_refs = [{"chunk_id": 1, "turn_id": 1, "occurred_at": "...09:24:49..."}]
```

`user.turn_counter` and `user.chunk_counter` both advanced 0→1→2. No
client-side cooperation needed.

### 2. Client-provided source_meta (MAB path)

Sent `filter_tags={"source_meta": {"serial_last": 500, "chunk_id": 99}}`.
After the call:

- The event's `source_refs` carried `serial_last=500` (preserved) and
  `chunk_id=99` (preserved).
- `turn_id` and `occurred_at` were filled by the server fallback.

### 3. Conflict resolution policy (opt-in path)

With `enable_conflict_resolution=True` passed to
`initialize_meta_agent`, the semantic agent's stored system prompt
grew from 5 554 chars to 6 674 chars (= base + 1 118-char policy
section + 2 separator). On a FactConsolidation `sh_6k` smoke run, the
agent wrote items with names like `"Thomas Kyd / born in"` and the
manager's auto-route fired (`cr_entity` / `cr_relation` populated in
`filter_tags`). For that entity, the canonical value was `"Leeds"`
(the higher-serial value), not the world-knowledge `"London"`.

End-to-end EM numbers depend additionally on how many facts the agent
writes within its ingest token budget (separate concern from conflict
resolution itself) and remain a tuning question for individual
benchmarks.

## What this does *not* fix

- Multi-hop reasoning. Conflict resolution is per `(entity, relation)`
  pair — chained `(entity_A, rel_1, ?)` → `(?, rel_2, ?)` lookups are
  out of scope.
- Verbatim quote recall (LongMemEval `single-session-assistant`). The
  agent still paraphrases the assistant's prior wording when storing
  semantic items; only the *value* slot of triple-shaped facts gets the
  no-hedging guarantee.
- Per-request toggling. The flag is read once at create time and baked
  into the agent's stored prompt; changing it requires re-creating the
  meta-agent.
- The recommended ingest density. Triggering the conflict-resolution
  branch still depends on the agent writing the right set of facts;
  the policy nudges the *shape* and the *value choice*, not the
  *coverage* of what gets written.

## Known limitations

### Chunk-level source_ref is not fact-level

The MAB adapter's `source_meta` carries
``{chunk_id, serial_first, serial_last, occurred_at}`` for the whole
chunk. Inside one `client.add` call, every individual fact the
semantic agent extracts ends up with the **same** `source_ref` — there
is no per-fact serial yet.

Consequence: when the agent writes two competing items with the same
`name` (e.g. `"Thomas Kyd / born in" → "London"` from fact #0 and
`"Thomas Kyd / born in" → "Leeds"` from fact #306) in the **same**
ingest, the deterministic merge in
`upsert_with_conflict_resolution` cannot distinguish them on
`source_ref` alone:

- `occurred_at` is identical (same wall-clock instant).
- `serial_last` is identical (always `306` for the whole chunk).
- Only `created_at` differs → tie broken by **insert order**.

If the agent writes the higher-serial fact last, the right value wins.
If it writes them in any other order, the wrong value wins. Either
way, the outcome is not really deterministic on the input contents —
it is determined by the agent's traversal order. For real conversational
inputs (each new fact is a separate `client.add` call with its own
`occurred_at`), the gap does not exist: timestamps separate the
ingests cleanly.

Fix paths considered (not in this patch):

1. Agent extracts the per-fact serial from the chunk text and puts it
   on each item it writes (prompt-engineering only, but
   `semantic_memory_insert`'s `items[]` schema currently has no
   per-item `source_ref` field).
2. Add an optional per-item `source_ref` to `SemanticMemoryItemBase`
   so the agent can override the chunk-level one.
3. Server-side: `insert_semantic_item` regex-scans the original chunk
   text to recover the serial. Requires also persisting the raw chunk
   on the agent context, which we are otherwise trying to avoid.

### Agent does not always write both sides of a conflict

The new policy section tells the agent to write facts verbatim with no
hedging. Empirically the agent will sometimes write only the value it
considers most plausible (often the one matching world knowledge),
skipping the other side of the conflict entirely. When that happens
the deterministic merge has nothing to merge — `prior_values` stays
`[]` and the result reflects the agent's pick, not the data.

This is independent from the chunk-vs-fact source_ref limitation above.
It is a prompt-following gap; mitigations are prompt-engineering or
a finer-grained tool surface, neither of which is in scope here.

### `serial` is never auto-derived by the server

Only the caller can populate `source_meta["serial"]` (or
`serial_first` / `serial_last`). The server fallback in
`_augment_source_meta_with_server_fallbacks` only fills `turn_id`,
`chunk_id`, and `occurred_at`. This is deliberate — `serial` is a
domain-specific signal — but it does mean that benchmarks with
implicit numbered facts (FactConsolidation) require client-side
support to surface that signal.

### `prior_values` is not yet surfaced on retrieval

Items written via the conflict-resolution path persist their history
in `prior_values`, but `retrieve_with_conversation` currently only
returns the current `summary`. Time-travel queries
("Where did I used to live?") would need a separate retrieval path
that exposes `prior_values` and lets the LLM see the timeline. Not
in this patch.
