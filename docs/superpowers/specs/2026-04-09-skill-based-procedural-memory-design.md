# Skill-Based Procedural Memory Design

## Overview

Upgrade MIRIX's procedural memory from a flat `entry_type + summary + steps` structure to a skill-based format. This is an in-place migration (Approach A): existing `procedural_memory` table is altered, existing CRUD/search/cache infrastructure is reused.

The goal is better extraction, organization, and deduplication of procedural knowledge — not changing MIRIX's architecture or adding new agent types.

## Data Model

### Field Changes

| Current Field | Change | New Field | Type |
|--------------|--------|-----------|------|
| `summary` | Rename | `description` | String |
| `steps` | Rename + type change | `instructions` | String (was JSON List[str]) |
| `summary_embedding` | Rename | `description_embedding` | Vector(4096) |
| `steps_embedding` | Rename | `instructions_embedding` | Vector(4096) |
| `entry_type` | Keep | `entry_type` | String |
| `filter_tags` | Keep | `filter_tags` | JSON |
| `last_modify` | Keep | `last_modify` | JSON |
| `embedding_config` | Keep | `embedding_config` | JSON |

### New Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String, required | — | Short identifier, e.g. "deploy-production" |
| `triggers` | JSON (List[str]) | `[]` | Conditions that indicate this skill is relevant |
| `examples` | JSON (List[dict]) | `[]` | Input/output examples, `[{"input": "...", "output": "..."}]` |
| `version` | String | `"0.1.0"` | Semver, incremented on update |

### Indexes

- Existing indexes on `organization_id`, `filter_tags` (GIN), `org_filter_scope` are retained.
- New index: `ix_procedural_memory_name` on `(organization_id, user_id, name)` for name-based dedup lookups.

## Tool Changes

### Renamed Tools

| Current | New |
|---------|-----|
| `procedural_memory_insert` | `skill_insert` |
| `procedural_memory_update` | `skill_update` |

### Constants

```python
# mirix/constants.py
SKILL_TOOLS = ["skill_insert", "skill_update"]  # replaces PROCEDURAL_MEMORY_TOOLS
```

Procedural Memory Agent's tool set becomes: `SKILL_TOOLS + UNIVERSAL_MEMORY_TOOLS`.

### `skill_insert` Parameters

```python
items: List[dict]
# Each item:
{
    "name": str,           # required
    "description": str,    # required
    "instructions": str,   # required
    "entry_type": str,     # required: "workflow" | "guide" | "script"
    "triggers": List[str], # optional, default []
    "examples": List[dict] # optional, default []
}
```

### `skill_update` Parameters

Unchanged interface: `old_ids: List[str], new_items: List[dict]`. The `new_items` use the same schema as `skill_insert`. Version auto-increments on update.

### Dedup Logic in `skill_insert`

Current: exact match on `summary + steps` → skip.

New two-tier dedup:
1. **Name match**: If an existing skill has the same `name`, treat as duplicate.
2. **Semantic match**: If no name match, compute cosine similarity on `description_embedding`. If similarity > 0.9 threshold, treat as duplicate.
3. **On duplicate**: Instead of skipping, call `skill_update` to merge/enhance the existing skill (delete old, insert new with version bump).

## Manager Changes

### `procedural_memory_manager.py`

- `insert_procedure()`: Parameters rename (`summary` → `description`, `steps` → `instructions`), add `name`, `triggers`, `examples`, `version`. No more `"\n".join(steps)` for embedding — `instructions` is already a string.
- `list_procedures()`: Search field references update (`summary` → `description`, `steps` → `instructions`). For BM25 full-text search, `instructions` is now a plain text column, simplifying the `regexp_replace(steps::text, ...)` workaround for JSON arrays.
- `_postgresql_fulltext_search()`: tsvector construction simplifies — `instructions` is text, not JSON. Weight mapping: `description='A'`, `instructions='B'`, `entry_type='C'` (same relative weights).
- Version management: On update, parse current version and increment patch (e.g. "0.1.0" → "0.1.1").

## Schema Changes

### `mirix/schemas/procedural_memory.py`

```python
class ProceduralMemoryItemBase(MirixBase):
    __id_prefix__ = "proc_item"
    name: str = Field(..., description="Short skill identifier")
    description: str = Field(..., description="What this skill does and when it's useful")
    instructions: str = Field(..., description="Detailed instructions as plain text")
    entry_type: str = Field(..., description="Category: 'workflow', 'guide', 'script'")
    triggers: List[str] = Field(default_factory=list, description="Trigger conditions")
    examples: List[dict] = Field(default_factory=list, description="Input/output examples")

class ProceduralMemoryItem(ProceduralMemoryItemBase):
    # ... existing db fields (id, agent_id, client_id, user_id, etc.)
    version: str = Field(default="0.1.0", description="Semver version")
    description_embedding: Optional[List[float]] = Field(None)
    instructions_embedding: Optional[List[float]] = Field(None)
    # ... rest unchanged
```

Embedding validator renamed: `pad_embeddings` covers `description_embedding` and `instructions_embedding`.

## ORM Changes

### `mirix/orm/procedural_memory.py`

- Column renames: `summary` → `description`, `steps` → `instructions` (type changes from `JSON` to `String`), embedding columns follow.
- New columns: `name` (String, not null), `triggers` (JSON, default `[]`), `examples` (JSON, default `[]`), `version` (String, default `"0.1.0"`).
- New index on `(organization_id, user_id, name)`.

## Prompt Changes

### `procedural_memory_agent.txt`

Replace current instructions with skill-oriented extraction guidance:
- Extract reusable skills with `name`, `description`, `instructions`, `triggers`, `examples`, `entry_type`
- `name` should be a concise kebab-case identifier
- `instructions` should be comprehensive plain text, not a numbered list
- `triggers` describe when this skill is applicable
- If a semantically similar skill already exists, use `skill_update` to merge/enhance rather than creating a duplicate

### `meta_memory_agent.txt`

Update the Procedural Memory section:
- Change "Step-by-Step Instructions & Processes" to "Reusable Skills & Procedures"
- Update description to mention skills with triggers, instructions, and examples
- Update the key question to "What reusable skill or procedure can be extracted?"
- Examples should reflect skill format

## API Changes

### `mirix/server/rest_api.py`

- `PATCH /memory/procedural/{memory_id}`: Request body updated — `summary` → `description`, `steps` → `instructions`, add `name`, `triggers`, `examples`.
- `DELETE /memory/procedural/{memory_id}`: No change needed.
- Response schemas auto-update from Pydantic model changes.

### `mirix/client/remote_client.py`

- Method signatures updated to match new field names.

## Validator Changes

### `mirix/agent/tool_validators.py`

- `validate_skill_insert`: Check `name` non-empty, `description` non-empty, `instructions` non-empty (replaces steps validation).
- `validate_skill_update`: Same field checks for `new_items`.

## Cache Changes

### `mirix/database/redis_client.py`

- Redis index field names updated: `summary` → `description`, `steps` → `instructions`, `summary_embedding` → `description_embedding`, `steps_embedding` → `instructions_embedding`.
- Additional indexed fields for `name` (TAG type for exact match).
- Index needs to be rebuilt after migration.

## Database Migration

Alembic migration script:

1. Add new columns: `name`, `triggers`, `examples`, `version` (with defaults so existing rows don't break).
2. Rename columns: `summary` → `description`, `steps` → `instructions`, `summary_embedding` → `description_embedding`, `steps_embedding` → `instructions_embedding`.
3. Data conversion:
   - `name`: Generate from old `summary` via slugify (lowercase, strip, replace spaces with hyphens, truncate to 60 chars).
   - `instructions`: `json.loads(steps)` → `"\n".join(steps_list)` to convert JSON array to plain text.
   - `triggers`: Default `[]`.
   - `examples`: Default `[]`.
   - `version`: Default `"0.1.0"`.
4. Change `instructions` column type from JSON to String (after data conversion).
5. Add new index on `(organization_id, user_id, name)`.

## Test Changes

All test files referencing procedural memory fields need updating:
- Field names in assertions and fixtures
- Tool names in mock calls
- Test data structures

## Out of Scope (Future Work)

- Skill evolution via replay-evaluate-mutate loop (AutoSkill pattern)
- Skill quality scoring and evaluation pipeline
- Skill sharing across users/organizations
- Chat Agent skill retrieval and execution feedback loop
- Security scanning for skill content
