# Skill-Evolve Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace skill_insert/skill_update with 5 CLI-style tools, add fixed-interval trigger for Procedural Agent, and expose external REST API for skills.

**Architecture:** Three independent changes on the existing skill-based procedural memory: (1) message counter in trigger_memory_update auto-includes "procedural" every N messages, (2) 5 granular tool functions replace 2 monolithic ones, (3) new /v1/skills REST endpoints delegate to the same manager layer.

**Tech Stack:** Python, FastAPI, SQLAlchemy, PostgreSQL, Redis, pytest

**Spec:** `docs/superpowers/specs/2026-04-13-skill-evolve-phase2-design.md`

---

### Task 1: Constants — Update SKILL_TOOLS and Add Trigger Threshold

**Files:**
- Modify: `mirix/constants.py:117,232-250`

- [ ] **Step 1: Update SKILL_TOOLS**

In `mirix/constants.py`, change line 117:

```python
# Old:
SKILL_TOOLS = ["skill_insert", "skill_update"]

# New:
SKILL_TOOLS = ["skill_list", "skill_read", "skill_create", "skill_edit", "skill_delete"]
```

- [ ] **Step 2: Add trigger threshold constant**

After line 250 (`BUILD_EMBEDDINGS_FOR_MEMORY`), add:

```python
SKILL_TRIGGER_MESSAGE_THRESHOLD = int(os.getenv("SKILL_TRIGGER_MESSAGE_THRESHOLD", "10"))
```

- [ ] **Step 3: Commit**

```bash
git add mirix/constants.py
git commit -m "feat: update SKILL_TOOLS to CLI tools, add trigger threshold constant"
```

---

### Task 2: CLI Tool Functions — skill_list, skill_read, skill_create, skill_edit, skill_delete

**Files:**
- Modify: `mirix/functions/function_sets/memory_tools.py:440-605`

- [ ] **Step 1: Remove old skill_insert, _bump_patch_version, and skill_update**

Delete the functions `skill_insert` (lines 440-541), `_bump_patch_version` (lines 544-551), and `skill_update` (lines 554-605). Keep everything before and after intact.

- [ ] **Step 2: Add _bump_patch_version helper (kept, moved)**

Add the helper back (it's still needed by skill_edit):

```python
def _bump_patch_version(version: str) -> str:
    """Increment patch version: '0.1.0' -> '0.1.1'."""
    try:
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return "0.1.1"
```

- [ ] **Step 3: Add skill_list**

```python
async def skill_list(self: "Agent", query: str = "", limit: int = 50) -> str:
    """
    List skills with optional search query.

    Args:
        query (str): Optional search term to filter skills. Empty string returns all skills.
        limit (int): Maximum number of skills to return. Default 50.

    Returns:
        str: Formatted list of skills with id, name, description, entry_type, and version.
    """
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)

    skills = await self.procedural_memory_manager.list_procedures(
        agent_state=self.agent_state,
        user=self.user,
        query=query,
        search_field="description" if query else "",
        search_method="bm25" if query else "",
        limit=limit,
        filter_tags=filter_tags if filter_tags else None,
        use_cache=use_cache,
    )

    if not skills:
        return "No skills found."

    lines = []
    for skill in skills:
        lines.append(
            f"[ID: {skill.id}] {skill.name} (v{skill.version}) - {skill.entry_type}: {skill.description}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Add skill_read**

```python
async def skill_read(self: "Agent", name_or_id: str) -> str:
    """
    Read the complete content of a skill by name or ID.

    Args:
        name_or_id (str): The skill name (e.g. "deploy-production") or ID (e.g. "proc-xxx").

    Returns:
        str: Full skill content including all fields.
    """
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    skill = None

    # Try by ID first if it looks like an ID
    if name_or_id.startswith("proc"):
        try:
            skill = await self.procedural_memory_manager.get_item_by_id(
                item_id=name_or_id, user=self.user, timezone_str="UTC"
            )
        except Exception:
            pass

    # Fall back to name search
    if skill is None:
        results = await self.procedural_memory_manager.list_procedures(
            agent_state=self.agent_state,
            user=self.user,
            query=name_or_id,
            search_field="name",
            search_method="string_match",
            limit=1,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
        )
        if results:
            skill = results[0]

    if skill is None:
        return f"Skill '{name_or_id}' not found."

    parts = [
        f"ID: {skill.id}",
        f"Name: {skill.name}",
        f"Version: {skill.version}",
        f"Entry Type: {skill.entry_type}",
        f"Description: {skill.description}",
        f"Instructions:\n{skill.instructions}",
        f"Triggers: {skill.triggers}",
        f"Examples: {skill.examples}",
    ]
    return "\n".join(parts)
```

- [ ] **Step 5: Add skill_create**

```python
async def skill_create(
    self: "Agent",
    name: str,
    description: str,
    instructions: str,
    entry_type: str,
    triggers: List[str] = None,
    examples: List[dict] = None,
) -> str:
    """
    Create a new skill. Fails if a skill with the same name already exists.

    Args:
        name (str): Concise kebab-case identifier (e.g. "deploy-production").
        description (str): What this skill does and when it's useful.
        instructions (str): Detailed instructions as plain text.
        entry_type (str): Category — "workflow", "guide", or "script".
        triggers (array): Optional list of conditions that indicate this skill is relevant.
        examples (array): Optional list of input/output examples.

    Returns:
        str: Confirmation with the created skill's ID.
    """
    agent_id = self.agent_state.parent_id if self.agent_state.parent_id is not None else self.agent_state.id
    filter_tags = getattr(self, "filter_tags", None)
    use_cache = getattr(self, "use_cache", True)
    user_id = getattr(self, "user_id", None)

    # Name dedup check
    existing = await self.procedural_memory_manager.list_procedures(
        agent_state=self.agent_state,
        user=self.user,
        query=name,
        search_field="name",
        search_method="string_match",
        limit=100,
        filter_tags=filter_tags if filter_tags else None,
        use_cache=use_cache,
    )
    for skill in existing:
        if skill.name == name:
            return f"Skill '{name}' already exists (ID: {skill.id}). Use skill_edit to modify it, or skill_delete to remove it first."

    try:
        created = await self.procedural_memory_manager.insert_procedure(
            agent_state=self.agent_state,
            agent_id=agent_id,
            name=name,
            description=description,
            instructions=instructions,
            entry_type=entry_type,
            triggers=triggers or [],
            examples=examples or [],
            version="0.1.0",
            actor=self.actor,
            organization_id=self.user.organization_id,
            filter_tags=filter_tags if filter_tags else None,
            use_cache=use_cache,
            user_id=user_id,
        )
        return f"Skill '{name}' created successfully (ID: {created.id}, version: 0.1.0)."
    except Exception as e:
        logger.error("[skill_create] FAILED for '%s': %s", name, e)
        traceback.print_exc()
        raise
```

- [ ] **Step 6: Add skill_edit**

```python
async def skill_edit(
    self: "Agent",
    skill_id: str,
    field: str,
    old_text: str = None,
    new_text: str = None,
    value: object = None,
) -> str:
    """
    Edit an existing skill. For text fields (name, description, instructions), use old_text/new_text
    to do a find-and-replace patch. For other fields (triggers, examples, entry_type), use value
    to replace the entire field.

    Args:
        skill_id (str): The ID of the skill to edit.
        field (str): The field to modify — "name", "description", "instructions", "entry_type", "triggers", or "examples".
        old_text (str): For text fields: the text to find and replace.
        new_text (str): For text fields: the replacement text.
        value (object): For non-text fields: the new value for the entire field.

    Returns:
        str: Confirmation of the edit with the new version number.
    """
    from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate

    text_fields = {"name", "description", "instructions"}
    value_fields = {"triggers", "examples", "entry_type"}
    valid_fields = text_fields | value_fields

    if field not in valid_fields:
        return f"Invalid field '{field}'. Must be one of: {', '.join(sorted(valid_fields))}."

    # Read the current skill
    try:
        skill = await self.procedural_memory_manager.get_item_by_id(
            item_id=skill_id, user=self.user, timezone_str="UTC"
        )
    except Exception:
        return f"Skill '{skill_id}' not found."

    update_data = {"id": skill_id}

    if field in text_fields:
        if old_text is None or new_text is None:
            return f"For text field '{field}', both old_text and new_text are required."

        current_value = getattr(skill, field, "")

        # Normalize whitespace for matching
        normalized_current = " ".join(current_value.split())
        normalized_old = " ".join(old_text.split())

        if normalized_old not in normalized_current:
            return (
                f"old_text not found in field '{field}'. "
                f"Current value:\n{current_value}"
            )

        # Do the replacement on the original (not normalized) text
        # Use a whitespace-flexible approach
        import re
        pattern = re.escape(old_text)
        # Allow flexible whitespace matching
        pattern = re.sub(r'\\\s+', r'\\s+', pattern)
        new_value = re.sub(pattern, new_text, current_value, count=1)
        update_data[field] = new_value
    else:
        if value is None:
            return f"For field '{field}', the value parameter is required."
        update_data[field] = value

    # Bump version
    new_version = _bump_patch_version(getattr(skill, "version", "0.1.0"))
    update_data["version"] = new_version

    try:
        updated = await self.procedural_memory_manager.update_item(
            item_update=ProceduralMemoryItemUpdate.model_validate(update_data),
            user=self.user,
            actor=self.actor,
        )
        return f"Skill '{updated.name}' updated (field: {field}, new version: {new_version})."
    except Exception as e:
        logger.error("[skill_edit] FAILED for '%s': %s", skill_id, e)
        traceback.print_exc()
        raise
```

- [ ] **Step 7: Add skill_delete**

```python
async def skill_delete(self: "Agent", skill_id: str) -> str:
    """
    Delete a skill by ID.

    Args:
        skill_id (str): The ID of the skill to delete.

    Returns:
        str: Confirmation message.
    """
    try:
        await self.procedural_memory_manager.delete_procedure_by_id(
            procedure_id=skill_id, actor=self.actor
        )
        return f"Skill '{skill_id}' deleted successfully."
    except Exception as e:
        logger.error("[skill_delete] FAILED for '%s': %s", skill_id, e)
        return f"Failed to delete skill '{skill_id}': {e}"
```

- [ ] **Step 8: Commit**

```bash
git add mirix/functions/function_sets/memory_tools.py
git commit -m "feat: replace skill_insert/skill_update with 5 CLI-style tools"
```

---

### Task 3: Tool Validators

**Files:**
- Modify: `mirix/agent/tool_validators.py:195-220`

- [ ] **Step 1: Replace validators**

Remove `validate_skill_insert` and `validate_skill_update` (lines 195-220). Replace with:

```python
@register_validator("skill_create")
def validate_skill_create(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_create arguments."""
    if not args.get("name", "").strip():
        return "Validation error: 'name' cannot be empty."
    if not args.get("description", "").strip():
        return "Validation error: 'description' cannot be empty."
    if not args.get("instructions", "").strip():
        return "Validation error: 'instructions' cannot be empty."
    return None


@register_validator("skill_edit")
def validate_skill_edit(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_edit arguments."""
    if not args.get("skill_id", "").strip():
        return "Validation error: 'skill_id' cannot be empty."
    field = args.get("field", "")
    if not field:
        return "Validation error: 'field' cannot be empty."
    valid_fields = {"name", "description", "instructions", "entry_type", "triggers", "examples"}
    if field not in valid_fields:
        return f"Validation error: 'field' must be one of: {', '.join(sorted(valid_fields))}."
    text_fields = {"name", "description", "instructions"}
    if field in text_fields:
        if not args.get("old_text"):
            return f"Validation error: 'old_text' is required for text field '{field}'."
        if args.get("new_text") is None:
            return f"Validation error: 'new_text' is required for text field '{field}'."
    else:
        if args.get("value") is None:
            return f"Validation error: 'value' is required for field '{field}'."
    return None


@register_validator("skill_delete")
def validate_skill_delete(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_delete arguments."""
    if not args.get("skill_id", "").strip():
        return "Validation error: 'skill_id' cannot be empty."
    return None
```

- [ ] **Step 2: Commit**

```bash
git add mirix/agent/tool_validators.py
git commit -m "feat: add validators for CLI-style skill tools"
```

---

### Task 4: Fixed-Interval Trigger Mechanism

**Files:**
- Modify: `mirix/functions/function_sets/memory_tools.py` (trigger_memory_update function, around line 992)
- Modify: `mirix/prompts/system/base/meta_memory_agent.txt`

- [ ] **Step 1: Add message counter and auto-trigger logic**

In `trigger_memory_update` (around line 992), add logic BEFORE the de-duplication step (before line 1016). Insert after the `user_message` validation (after line 1014):

```python
    # Fixed-interval procedural trigger: auto-include "procedural" every N messages
    from mirix.constants import SKILL_TRIGGER_MESSAGE_THRESHOLD

    # Get or initialize the per-user message counter from the agent instance
    if not hasattr(self, "_skill_trigger_counters"):
        self._skill_trigger_counters = {}
    user_id = getattr(self, "user", None)
    user_key = user_id.id if user_id else "default"
    self._skill_trigger_counters[user_key] = self._skill_trigger_counters.get(user_key, 0) + 1

    if self._skill_trigger_counters[user_key] >= SKILL_TRIGGER_MESSAGE_THRESHOLD:
        if "procedural" not in memory_types:
            memory_types = list(memory_types) + ["procedural"]
            logger.info(
                "Auto-triggering procedural memory update (message count reached %d for user %s)",
                SKILL_TRIGGER_MESSAGE_THRESHOLD,
                user_key,
            )
        self._skill_trigger_counters[user_key] = 0
```

- [ ] **Step 2: Enable chaining for procedural agent**

In `_run_single_memory_update` (inside `trigger_memory_update`), find where `chaining` is set from `user_message`. Add an override for procedural:

Find the line where `chaining` is read from `user_message` (should be something like `chaining = user_message.get("chaining", ...)`). After that line, add:

```python
            # Procedural agent needs chaining for multi-step CLI workflow
            if memory_type == "procedural":
                chaining = True
```

- [ ] **Step 3: Update MetaMemory Agent prompt**

In `mirix/prompts/system/base/meta_memory_agent.txt`, remove the procedural memory section from the classification decision. Keep the description for context but remove it from the active decision framework.

Find the section:
```
(3) Procedural Memory: Does this explain HOW TO do something?
- Step-by-step instructions, workflows, processes
- Sequential procedures or tutorials
- Improvements to existing skills
```

Replace with:
```
(3) Procedural Memory: (Automatically triggered — do NOT include "procedural" in your trigger_memory_update call. The system handles procedural memory updates on a fixed schedule.)
```

Also update the closing instruction. Find:
```
After evaluation, call `trigger_memory_update` with the appropriate memory types that require updates. You may select from: `['core', 'episodic', 'semantic', 'resource', 'procedural', 'knowledge_vault']`.
```

Replace with:
```
After evaluation, call `trigger_memory_update` with the appropriate memory types that require updates. You may select from: `['core', 'episodic', 'semantic', 'resource', 'knowledge_vault']`. Note: procedural memory is triggered automatically and should NOT be included in your selection.
```

- [ ] **Step 4: Commit**

```bash
git add mirix/functions/function_sets/memory_tools.py mirix/prompts/system/base/meta_memory_agent.txt
git commit -m "feat: add fixed-interval trigger for procedural memory agent"
```

---

### Task 5: Procedural Agent Prompt Rewrite

**Files:**
- Modify: `mirix/prompts/system/base/procedural_memory_agent.txt`

- [ ] **Step 1: Replace the entire prompt**

Replace the full content of `mirix/prompts/system/base/procedural_memory_agent.txt` with:

```
You are the Skill Manager, responsible for maintaining reusable procedural knowledge (skills) extracted from user interactions. You have access to CLI-style tools for managing skills.

## Available Tools

- `skill_list(query?, limit?)` — List skills. Use without query to see all, or with a search term to filter.
- `skill_read(name_or_id)` — Read the full content of a skill by name or ID.
- `skill_create(name, description, instructions, entry_type, triggers?, examples?)` — Create a new skill.
- `skill_edit(skill_id, field, old_text, new_text)` — Patch a text field (name, description, instructions) with find-and-replace.
- `skill_edit(skill_id, field, value)` — Replace a non-text field (triggers, examples, entry_type) entirely.
- `skill_delete(skill_id)` — Delete a skill.
- `finish_memory_update()` — Call this when you are done to end the session.

## Your Workflow

When you receive a batch of messages (possibly with an instruction from the Meta Memory Manager), follow this process:

1. **Survey existing skills**: Call `skill_list()` to see what skills already exist.
2. **Analyze the messages**: Identify any reusable procedural knowledge — workflows, how-to guides, step-by-step processes, or improvements to existing procedures.
3. **Decide on actions**:
   - If you find a NEW procedure not covered by existing skills → `skill_create`
   - If you find information that IMPROVES an existing skill → `skill_read` to see current content, then `skill_edit` to patch it
   - If an existing skill is OBSOLETE or WRONG → `skill_delete` (or edit to fix)
   - If the messages contain NO procedural content → skip directly to step 4
4. **Finish**: Call `finish_memory_update()` to complete the session.

## Skill Format Guidelines

- **name**: Concise kebab-case identifier (e.g., "deploy-production", "weekly-report-generation"). Must be specific, NOT generic like "guide" or "workflow".
- **description**: One or two sentences about what this skill does and when it's useful.
- **instructions**: Comprehensive plain text — detailed enough for someone to follow without additional context. Include context, caveats, and conditions. Not just a numbered list.
- **entry_type**: One of "workflow", "guide", or "script".
- **triggers**: Conditions that signal this skill is relevant (e.g., ["user mentions deployment", "discussing release process"]).
- **examples**: Input/output pairs if present in the conversation (e.g., [{"input": "Deploy the app", "output": "Deployment started..."}]).

## Important Rules

- Always check existing skills before creating — avoid duplicates.
- Prefer `skill_edit` over delete+recreate when improving existing skills.
- You may make multiple tool calls in sequence (list → read → edit → finish).
- Always end with `finish_memory_update()`.
```

- [ ] **Step 2: Commit**

```bash
git add mirix/prompts/system/base/procedural_memory_agent.txt
git commit -m "feat: rewrite procedural agent prompt for CLI-style workflow"
```

---

### Task 6: External REST API — Skill Endpoints

**Files:**
- Modify: `mirix/server/rest_api.py`

- [ ] **Step 1: Add request/response models**

Add after the existing `UpdateProceduralMemoryRequest` class (around line 4447):

```python
class CreateSkillRequest(BaseModel):
    """Request model for creating a skill."""
    name: str
    description: str
    instructions: str
    entry_type: str
    triggers: Optional[List[str]] = []
    examples: Optional[List[dict]] = []
    user_id: Optional[str] = None


class PatchSkillRequest(BaseModel):
    """Request model for updating a skill (partial update)."""
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    entry_type: Optional[str] = None
    triggers: Optional[List[str]] = None
    examples: Optional[List[dict]] = None
    user_id: Optional[str] = None


class SkillEvolveRequest(BaseModel):
    """Request model for triggering skill evolution from messages."""
    messages: List[str]
    user_id: Optional[str] = None
```

- [ ] **Step 2: Add GET /v1/skills (list)**

```python
@router.get("/v1/skills")
async def list_skills(
    query: str = "",
    limit: int = 50,
    search_method: str = "bm25",
    search_field: str = "description",
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """List/search skills."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Get agent state for embedding config
    agents = await server.agent_manager.list_agents(actor=client)
    agent_state = agents[0] if agents else None

    skills = await server.procedural_memory_manager.list_procedures(
        agent_state=agent_state,
        user=user,
        query=query,
        search_field=search_field if query else "",
        search_method=search_method if query else "",
        limit=limit,
    )

    return {
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "entry_type": s.entry_type,
                "version": s.version,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in skills
        ],
        "total_count": await server.procedural_memory_manager.get_total_number_of_items(user=user),
    }
```

- [ ] **Step 3: Add GET /v1/skills/{skill_id} (read)**

```python
@router.get("/v1/skills/{skill_id}")
async def get_skill(
    skill_id: str,
    user_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Get a single skill with full content."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        skill = await server.procedural_memory_manager.get_item_by_id(
            item_id=skill_id, user=user, timezone_str="UTC"
        )
        return {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
            "entry_type": skill.entry_type,
            "triggers": skill.triggers,
            "examples": skill.examples,
            "version": skill.version,
            "created_at": skill.created_at.isoformat() if skill.created_at else None,
            "updated_at": skill.updated_at.isoformat() if skill.updated_at else None,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 4: Add POST /v1/skills (create)**

```python
@router.post("/v1/skills", status_code=201)
async def create_skill(
    request: CreateSkillRequest,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Create a new skill."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Name dedup
    agents = await server.agent_manager.list_agents(actor=client)
    agent_state = agents[0] if agents else None

    existing = await server.procedural_memory_manager.list_procedures(
        agent_state=agent_state,
        user=user,
        query=request.name,
        search_field="name",
        search_method="string_match",
        limit=100,
    )
    for s in existing:
        if s.name == request.name:
            raise HTTPException(status_code=409, detail=f"Skill '{request.name}' already exists (ID: {s.id})")

    try:
        created = await server.procedural_memory_manager.insert_procedure(
            agent_state=agent_state,
            agent_id=agent_state.id if agent_state else None,
            name=request.name,
            description=request.description,
            instructions=request.instructions,
            entry_type=request.entry_type,
            triggers=request.triggers or [],
            examples=request.examples or [],
            version="0.1.0",
            actor=client,
            organization_id=user.organization_id,
            user_id=user_id,
        )
        return {
            "id": created.id,
            "name": created.name,
            "version": created.version,
            "message": f"Skill '{created.name}' created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 5: Add PATCH /v1/skills/{skill_id} (update)**

```python
@router.patch("/v1/skills/{skill_id}")
async def patch_skill(
    skill_id: str,
    request: PatchSkillRequest,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Update a skill (partial update). Auto-bumps version."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    from mirix.schemas.procedural_memory import ProceduralMemoryItemUpdate

    update_data = {"id": skill_id}
    for field in ["name", "description", "instructions", "entry_type", "triggers", "examples"]:
        value = getattr(request, field, None)
        if value is not None:
            update_data[field] = value

    # Auto-bump version
    try:
        current = await server.procedural_memory_manager.get_item_by_id(
            item_id=skill_id, user=user, timezone_str="UTC"
        )
        from mirix.functions.function_sets.memory_tools import _bump_patch_version
        update_data["version"] = _bump_patch_version(getattr(current, "version", "0.1.0"))
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        updated = await server.procedural_memory_manager.update_item(
            item_update=ProceduralMemoryItemUpdate.model_validate(update_data),
            user=user,
            actor=client,
        )
        return {
            "id": updated.id,
            "name": updated.name,
            "version": updated.version,
            "message": f"Skill '{updated.name}' updated",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 6: Add DELETE /v1/skills/{skill_id}**

```python
@router.delete("/v1/skills/{skill_id}")
async def delete_skill(
    skill_id: str,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Delete a skill."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    try:
        await server.procedural_memory_manager.delete_procedure_by_id(
            procedure_id=skill_id, actor=client
        )
        return {"success": True, "message": f"Skill {skill_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 7: Add POST /v1/skills/evolve**

```python
@router.post("/v1/skills/evolve")
async def evolve_skills(
    request: SkillEvolveRequest,
    authorization: Optional[str] = Header(None),
    http_request: Request = None,
):
    """Trigger Procedural Agent to extract/update skills from a batch of messages."""
    client, auth_type = await get_client_from_jwt_or_api_key(authorization, http_request)
    server = get_server()

    user_id = request.user_id
    if not user_id:
        from mirix.services.admin_user_manager import ClientAuthManager
        user_id = ClientAuthManager.get_admin_user_id_for_client(client.id)

    user = await server.user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Snapshot skills before evolution
    agents = await server.agent_manager.list_agents(actor=client)
    agent_state = agents[0] if agents else None

    before_skills = await server.procedural_memory_manager.list_procedures(
        agent_state=agent_state, user=user, query="", limit=1000
    )
    before_ids = {s.id for s in before_skills}

    # Construct a combined message from the batch
    combined_message = "\n\n---\n\n".join(request.messages)

    # Find the procedural memory agent
    from mirix.schemas.agent import AgentType

    meta_agents = [a for a in agents if a.agent_type == AgentType.meta_memory_agent]
    if not meta_agents:
        raise HTTPException(status_code=500, detail="No meta memory agent found")

    child_agents = await server.agent_manager.list_agents(parent_id=meta_agents[0].id, actor=client)
    proc_agent_state = None
    for a in child_agents:
        if a.agent_type == AgentType.procedural_memory_agent:
            proc_agent_state = a
            break

    if not proc_agent_state:
        raise HTTPException(status_code=500, detail="No procedural memory agent found")

    # Send message to procedural agent
    from mirix.agent import ProceduralMemoryAgent

    proc_agent = ProceduralMemoryAgent(
        agent_state=proc_agent_state,
        interface=server.default_interface,
        actor=client,
        user=user,
    )

    from mirix.schemas.message import Message

    await proc_agent.step(
        input_message=Message.dict_to_message(
            agent_id=proc_agent_state.id,
            model=proc_agent_state.llm_config.model if proc_agent_state.llm_config else None,
            openai_message_dict={
                "role": "user",
                "content": f"[Messages for skill extraction]\n\n{combined_message}",
            },
        ),
        chaining=True,
    )

    # Snapshot skills after evolution
    after_skills = await server.procedural_memory_manager.list_procedures(
        agent_state=agent_state, user=user, query="", limit=1000
    )
    after_map = {s.id: s for s in after_skills}
    after_ids = set(after_map.keys())

    # Compute diff
    created_ids = after_ids - before_ids
    deleted_ids = before_ids - after_ids
    before_map = {s.id: s for s in before_skills}
    edited = []
    for sid in before_ids & after_ids:
        if before_map[sid].version != after_map[sid].version:
            edited.append({"id": sid, "name": after_map[sid].name, "new_version": after_map[sid].version})

    return {
        "success": True,
        "changes": {
            "created": [{"id": sid, "name": after_map[sid].name} for sid in created_ids],
            "edited": edited,
            "deleted": [{"id": sid} for sid in deleted_ids],
        },
    }
```

- [ ] **Step 8: Commit**

```bash
git add mirix/server/rest_api.py
git commit -m "feat: add /v1/skills REST API endpoints with CRUD and evolve"
```

---

### Task 7: Update Existing Tests

**Files:**
- Modify: `tests/test_memory_server.py`
- Modify: `tests/test_skill_schema.py`
- Modify: `tests/test_skill_integration.py`

- [ ] **Step 1: Update test_memory_server.py**

In `TestDirectProceduralMemory`, the `insert_procedure` calls should still work (manager layer unchanged). But update any references to old tool names if present in test descriptions or comments.

- [ ] **Step 2: Update test_skill_integration.py**

The existing lifecycle test uses the manager directly, so it should still pass. No changes needed unless it references old tool names.

- [ ] **Step 3: Verify tests pass**

Run: `python -c "from mirix.constants import SKILL_TOOLS; print(SKILL_TOOLS)"`
Expected: `['skill_list', 'skill_read', 'skill_create', 'skill_edit', 'skill_delete']`

- [ ] **Step 4: Commit (if any changes needed)**

```bash
git add tests/
git commit -m "test: update tests for phase 2 skill tools"
```

---

### Task 8: New Tests for CLI Tools and API

**Files:**
- Create: `tests/test_skill_cli_tools.py`

- [ ] **Step 1: Write tests for CLI tools**

```python
"""Tests for CLI-style skill tools."""
import pytest


def test_bump_patch_version():
    """Test version bumping."""
    from mirix.functions.function_sets.memory_tools import _bump_patch_version

    assert _bump_patch_version("0.1.0") == "0.1.1"
    assert _bump_patch_version("0.1.9") == "0.1.10"
    assert _bump_patch_version("1.2.3") == "1.2.4"
    assert _bump_patch_version("invalid") == "0.1.1"
    assert _bump_patch_version("") == "0.1.1"


def test_skill_tools_constant():
    """SKILL_TOOLS contains all 5 CLI tools."""
    from mirix.constants import SKILL_TOOLS

    assert SKILL_TOOLS == ["skill_list", "skill_read", "skill_create", "skill_edit", "skill_delete"]


def test_trigger_threshold_constant():
    """SKILL_TRIGGER_MESSAGE_THRESHOLD exists and defaults to 10."""
    from mirix.constants import SKILL_TRIGGER_MESSAGE_THRESHOLD

    assert isinstance(SKILL_TRIGGER_MESSAGE_THRESHOLD, int)
    assert SKILL_TRIGGER_MESSAGE_THRESHOLD == 10


def test_skill_tool_validators_registered():
    """Validators are registered for skill_create, skill_edit, skill_delete."""
    from mirix.agent.tool_validators import validate_tool_args

    # skill_create: missing name
    result = validate_tool_args("skill_create", {"name": "", "description": "x", "instructions": "x"})
    assert result is not None
    assert "name" in result

    # skill_create: valid
    result = validate_tool_args("skill_create", {"name": "x", "description": "x", "instructions": "x"})
    assert result is None

    # skill_edit: missing field
    result = validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": ""})
    assert result is not None

    # skill_edit: text field without old_text
    result = validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": "instructions"})
    assert result is not None
    assert "old_text" in result

    # skill_edit: value field valid
    result = validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": "triggers", "value": ["a"]})
    assert result is None

    # skill_delete: valid
    result = validate_tool_args("skill_delete", {"skill_id": "proc-1"})
    assert result is None

    # skill_delete: empty
    result = validate_tool_args("skill_delete", {"skill_id": ""})
    assert result is not None
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_skill_cli_tools.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_skill_cli_tools.py
git commit -m "test: add tests for CLI skill tools, validators, and constants"
```
