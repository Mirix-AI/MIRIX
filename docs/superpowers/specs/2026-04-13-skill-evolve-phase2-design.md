# Skill-Evolve Phase 2: Trigger Mechanism, CLI Tools, and External API

## Overview

Phase 2 of the skill-evolve work covers three changes:
1. **Fixed-interval trigger**: Procedural Agent is triggered every 10 messages instead of by MetaMemory LLM judgment
2. **CLI-style tools**: Replace `skill_insert`/`skill_update` with 5 granular tools (`skill_list`, `skill_read`, `skill_create`, `skill_edit`, `skill_delete`)
3. **External REST API**: Expose `/v1/skills` CRUD + `/v1/skills/evolve` for external consumers

Phase 1 (completed) upgraded the data model from `summary + steps` to `name + description + instructions + triggers + examples + version`. Phase 2 builds on that foundation.

## 1. Trigger Mechanism

### Current Flow

```
Messages accumulate → MetaMemory Agent → LLM decides which memory types → may or may not trigger procedural
```

### New Flow

```
Messages accumulate → MetaMemory Agent
  ├─ core/episodic/semantic/resource/knowledge_vault → still LLM-decided
  └─ procedural → fixed trigger every N messages (no LLM judgment)
```

### Implementation

- Add a per-user message counter in the MetaMemory Agent's trigger logic (in `trigger_memory_update` or the MetaMemory Agent's step function)
- When counter >= threshold: automatically include `"procedural"` in `memory_types`, regardless of MetaMemory's LLM output
- Reset counter to 0 after triggering
- Threshold is configurable via environment variable: `SKILL_TRIGGER_MESSAGE_THRESHOLD` (default: 10)
- MetaMemory Agent's system prompt: remove the Procedural Memory section from the decision framework (sections about "Does this explain HOW TO do something?"). MetaMemory no longer judges whether to trigger procedural — it happens automatically.
- Other 5 memory types remain LLM-decided by MetaMemory as before.

### Chaining

Procedural Agent needs multi-step chaining (list → read → create/edit). When MetaMemory triggers procedural, it must pass `chaining=True`. This is controlled by passing the flag in the `trigger_memory_update` function when calling the Procedural Agent.

## 2. CLI-Style Tools

### Tool Replacement

**Remove:**
- `skill_insert`
- `skill_update`
- `_bump_patch_version` helper (move version bump logic into `skill_edit`)

**Add:**

#### `skill_list(query?, limit?)`

List skills with optional search query.

- **Parameters**: `query` (str, optional — search term), `limit` (int, optional, default 50)
- **Returns**: List of skill summaries: `[{id, name, description, entry_type, version}]`
- **Implementation**: Calls `procedural_memory_manager.list_procedures()` with `search_field="description"`, `search_method="bm25"` when query provided; empty query returns all sorted by `created_at` desc.

#### `skill_read(name_or_id)`

Read complete skill content.

- **Parameters**: `name_or_id` (str — skill name or ID)
- **Returns**: Full skill object: `{id, name, description, instructions, entry_type, triggers, examples, version, created_at}`
- **Implementation**: If input looks like an ID (starts with "proc"), use `get_item_by_id()`. Otherwise, search by name using `list_procedures(query=name_or_id, search_field="name", limit=1)`.

#### `skill_create(name, description, instructions, entry_type, triggers?, examples?)`

Create a new skill.

- **Parameters**:
  - `name` (str, required) — kebab-case identifier
  - `description` (str, required) — what this skill does
  - `instructions` (str, required) — detailed instructions as plain text
  - `entry_type` (str, required) — "workflow" | "guide" | "script"
  - `triggers` (List[str], optional, default [])
  - `examples` (List[dict], optional, default [])
- **Dedup**: Before creating, check if a skill with the same `name` already exists. If yes, return an error: `"Skill '{name}' already exists (ID: {id}). Use skill_edit to modify it, or skill_delete to remove it first."`
- **Version**: Always starts at `"0.1.0"`.
- **Returns**: Created skill summary with ID.

#### `skill_edit(skill_id, field, old_text, new_text)` / `skill_edit(skill_id, field, value)`

Edit an existing skill. Two modes based on field type:

**Text patch mode** (for `instructions`, `description`, `name`):
- **Parameters**: `skill_id` (str), `field` (str), `old_text` (str), `new_text` (str)
- **Behavior**: Find `old_text` in the field's current value, replace with `new_text`. If `old_text` not found, return error with current field value so Agent can retry.
- Fuzzy matching: normalize whitespace before comparison (strip leading/trailing, collapse multiple spaces).

**Value replace mode** (for `triggers`, `examples`, `entry_type`):
- **Parameters**: `skill_id` (str), `field` (str), `value` (any)
- **Behavior**: Replace the entire field value.

**Both modes**:
- Auto-bump version (patch +1) on success.
- Update `last_modify` timestamp and operation="updated".
- Regenerate embeddings for `description`/`instructions` if those fields changed.
- Returns updated skill summary.

#### `skill_delete(skill_id)`

Delete a skill.

- **Parameters**: `skill_id` (str)
- **Returns**: Confirmation message.
- **Implementation**: Calls `procedural_memory_manager.delete_procedure_by_id()`.

### Validator Updates

Replace `validate_skill_insert`/`validate_skill_update` with validators for each new tool:

- `validate_skill_create`: `name`, `description`, `instructions` must be non-empty
- `validate_skill_edit`: `skill_id` and `field` must be non-empty; `field` must be one of the valid field names
- `validate_skill_delete`: `skill_id` must be non-empty
- `skill_list` and `skill_read`: no validation needed

### Constants Update

```python
# mirix/constants.py
SKILL_TOOLS = ["skill_list", "skill_read", "skill_create", "skill_edit", "skill_delete"]
```

### Agent Tool Set

Procedural Agent tools: `SKILL_TOOLS + UNIVERSAL_MEMORY_TOOLS`

`UNIVERSAL_MEMORY_TOOLS` = `["search_in_memory", "finish_memory_update", "list_memory_within_timerange"]`

## 3. Procedural Agent Prompt

Replace the current "single function call" prompt with a multi-step CLI workflow prompt.

Key changes:
- Remove "Make only ONE function call total" restriction
- Guide the agent through: `skill_list` → analyze → `skill_read` if needed → `skill_create` / `skill_edit` / `skill_delete` → `finish_memory_update`
- Emphasize: always check existing skills before creating (avoid duplicates)
- Emphasize: prefer `skill_edit` over delete+recreate when improving existing skills
- Keep `finish_memory_update` as the required termination signal

## 4. External REST API

### New Endpoints

All under `/v1/` prefix. Authentication: JWT or Client API Key (same as existing endpoints).

#### `GET /v1/skills`

List/search skills.

**Query Parameters:**
- `query` (str, optional) — search term
- `limit` (int, optional, default 50)
- `search_method` (str, optional, default "bm25") — "bm25" | "embedding" | "string_match"
- `search_field` (str, optional, default "description") — "description" | "instructions" | "name"
- `user_id` (str, optional)

**Response:**
```json
{
  "skills": [
    {"id": "proc-xxx", "name": "deploy-production", "description": "...", "entry_type": "workflow", "version": "0.1.2", "created_at": "..."}
  ],
  "total_count": 42
}
```

#### `GET /v1/skills/{skill_id}`

Get a single skill with full content.

**Response:**
```json
{
  "id": "proc-xxx",
  "name": "deploy-production",
  "description": "How to deploy to production",
  "instructions": "1. Run tests\n2. Build image\n...",
  "entry_type": "workflow",
  "triggers": ["user mentions deployment"],
  "examples": [{"input": "deploy the app", "output": "starting deploy..."}],
  "version": "0.1.2",
  "created_at": "...",
  "updated_at": "..."
}
```

#### `POST /v1/skills`

Create a new skill.

**Request Body:**
```json
{
  "name": "deploy-production",
  "description": "How to deploy to production",
  "instructions": "1. Run tests\n2. Build image\n...",
  "entry_type": "workflow",
  "triggers": ["user mentions deployment"],
  "examples": [],
  "user_id": "user-xxx"
}
```

**Response:** Created skill object with `id` and `version: "0.1.0"`.

Name dedup: if name already exists for this user, return 409 Conflict.

#### `PATCH /v1/skills/{skill_id}`

Update a skill. Supports partial update — only provided fields are changed.

**Request Body:**
```json
{
  "description": "Updated description",
  "instructions": "Updated instructions",
  "triggers": ["new trigger"],
  "user_id": "user-xxx"
}
```

Auto-bumps version. Regenerates embeddings for changed text fields.

**Response:** Updated skill object.

#### `DELETE /v1/skills/{skill_id}`

Delete a skill.

**Response:** `{"success": true, "message": "Skill proc-xxx deleted"}`

#### `POST /v1/skills/evolve`

Trigger Procedural Agent to extract/update skills from a batch of messages.

**Request Body:**
```json
{
  "messages": ["message text 1", "message text 2", "..."],
  "user_id": "user-xxx"
}
```

**Behavior:**
1. Instantiate Procedural Agent with the provided messages as input
2. Agent runs its full workflow (list → analyze → create/edit/delete → finish)
3. Return a summary of changes

**Response:**
```json
{
  "success": true,
  "changes": {
    "created": [{"id": "proc-xxx", "name": "new-skill"}],
    "edited": [{"id": "proc-yyy", "name": "existing-skill", "new_version": "0.1.3"}],
    "deleted": []
  }
}
```

### Backward Compatibility

Existing endpoints `PATCH /memory/procedural/{memory_id}` and `DELETE /memory/procedural/{memory_id}` are kept. They internally delegate to the same manager methods. No breaking changes for current consumers.

## 5. Files to Modify

| File | Change |
|------|--------|
| `mirix/constants.py` | Update `SKILL_TOOLS` list, add `SKILL_TRIGGER_MESSAGE_THRESHOLD` |
| `mirix/functions/function_sets/memory_tools.py` | Replace `skill_insert`/`skill_update` with 5 CLI tools, update `trigger_memory_update` for fixed-interval procedural trigger |
| `mirix/agent/tool_validators.py` | Replace validators for new tools |
| `mirix/services/agent_manager.py` | No change (already uses `SKILL_TOOLS` from constants) |
| `mirix/services/procedural_memory_manager.py` | Add `get_item_by_name()` helper for `skill_read` by name |
| `mirix/server/rest_api.py` | Add `/v1/skills` endpoints |
| `mirix/client/remote_client.py` | Add client methods for new API |
| `mirix/prompts/system/base/procedural_memory_agent.txt` | Rewrite for CLI workflow |
| `mirix/prompts/system/base/meta_memory_agent.txt` | Remove procedural from LLM decision (keep description for context, remove from classification rule) |
| Tests | Update existing, add new for CLI tools and API |

## Out of Scope

- "Dreaming" mechanism (separate team)
- Semantic dedup (description embedding similarity) — future phase
- Security scanning for skill content — future phase
- Progressive disclosure for Chat Agent (skill_view tool) — future phase
