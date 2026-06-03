"""
Tool argument validation registry.

Usage:
    @register_validator("episodic_memory_insert", "episodic_memory_replace")
    def validate_episodic_memory(function_name: str, args: dict) -> Optional[str]:
        '''Returns error message if invalid, None if valid.'''
        ...

    # In agent.py:
    error = validate_tool_args(function_name, function_args)
    if error:
        # handle validation failure
"""

from typing import Callable, Dict, Optional

# Registry: tool_name -> validator_function
_VALIDATORS: Dict[str, Callable[[str, dict], Optional[str]]] = {}


def register_validator(*tool_names: str):
    """
    Decorator to register a validation function for one or more tools.

    The validator function signature: (function_name: str, args: dict) -> Optional[str]
    Returns error message if validation fails, None if valid.
    """

    def decorator(func: Callable[[str, dict], Optional[str]]):
        for name in tool_names:
            _VALIDATORS[name] = func
        return func

    return decorator


def validate_tool_args(function_name: str, function_args: dict) -> Optional[str]:
    """
    Validate tool arguments using registered validator.
    Returns error message if validation fails, None if valid.
    """
    validator = _VALIDATORS.get(function_name)
    if validator:
        return validator(function_name, function_args)
    return None


# ============================================================
# Validators - Add new validators below using @register_validator
# ============================================================


@register_validator("episodic_memory_insert")
def validate_episodic_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in item {i} cannot be empty. "
                "Please provide a detailed description of the event."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a concise summary of the event."
            )
    return None


@register_validator("episodic_memory_replace")
def validate_episodic_memory_replace(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_replace arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in new_items[{i}] cannot be empty. "
                "Please provide a detailed description of the event."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a concise summary of the event."
            )
    return None


@register_validator("episodic_memory_merge")
def validate_episodic_memory_merge(function_name: str, args: dict) -> Optional[str]:
    """Validate episodic_memory_merge arguments."""
    if not args.get("event_id", "").strip():
        return "Validation error: 'event_id' cannot be empty. Please provide the ID of the event to merge into."
    return None


# ============================================================
# Semantic Memory Validators
# ============================================================


@register_validator("semantic_memory_insert")
def validate_semantic_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate semantic_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in item {i} cannot be empty. "
                "Please provide the name or main concept for this knowledge entry."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a concise summary of the concept."
            )
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in item {i} cannot be empty. "
                "Please provide detailed explanation or context for the concept."
            )
    return None


@register_validator("semantic_memory_update")
def validate_semantic_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate semantic_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("name", "").strip():
            return (
                f"Validation error: 'name' field in new_items[{i}] cannot be empty. "
                "Please provide the name or main concept for this knowledge entry."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a concise summary of the concept."
            )
        if not item.get("details", "").strip():
            return (
                f"Validation error: 'details' field in new_items[{i}] cannot be empty. "
                "Please provide detailed explanation or context for the concept."
            )
    return None


# ============================================================
# Resource Memory Validators
# ============================================================


@register_validator("resource_memory_insert")
def validate_resource_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate resource_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("title", "").strip():
            return (
                f"Validation error: 'title' field in item {i} cannot be empty. "
                "Please provide a title for this resource."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a summary of this resource."
            )
    return None


@register_validator("resource_memory_update")
def validate_resource_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate resource_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("title", "").strip():
            return (
                f"Validation error: 'title' field in new_items[{i}] cannot be empty. "
                "Please provide a title for this resource."
            )
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a summary of this resource."
            )
    return None


# ============================================================
# Skill Validators
# ============================================================


@register_validator("skill_create")
def validate_skill_create(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_create arguments."""
    from mirix.schemas.procedural_memory import (
        SKILL_ENTRY_TYPES,
        SKILL_MAX_DESCRIPTION_LEN,
        SKILL_MAX_INSTRUCTIONS_LEN,
        SKILL_MAX_NAME_LEN,
    )

    name = args.get("name", "")
    if not name.strip():
        return "Validation error: 'name' cannot be empty."
    if len(name) > SKILL_MAX_NAME_LEN:
        return f"Validation error: 'name' exceeds max length {SKILL_MAX_NAME_LEN}."
    description = args.get("description", "")
    if not description.strip():
        return "Validation error: 'description' cannot be empty."
    if len(description) > SKILL_MAX_DESCRIPTION_LEN:
        return f"Validation error: 'description' exceeds max length {SKILL_MAX_DESCRIPTION_LEN}."
    instructions = args.get("instructions", "")
    if not instructions.strip():
        return "Validation error: 'instructions' cannot be empty."
    if len(instructions) > SKILL_MAX_INSTRUCTIONS_LEN:
        return f"Validation error: 'instructions' exceeds max length {SKILL_MAX_INSTRUCTIONS_LEN}."
    entry_type = args.get("entry_type", "")
    if not entry_type:
        return "Validation error: 'entry_type' cannot be empty."
    if entry_type not in SKILL_ENTRY_TYPES:
        return (
            f"Validation error: 'entry_type' must be one of: {sorted(SKILL_ENTRY_TYPES)}."
        )
    return None


@register_validator("skill_edit")
def validate_skill_edit(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_edit arguments."""
    from mirix.schemas.procedural_memory import (
        SKILL_ENTRY_TYPES,
        SKILL_MAX_DESCRIPTION_LEN,
        SKILL_MAX_INSTRUCTIONS_LEN,
        SKILL_MAX_NAME_LEN,
    )

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
        new_text = args.get("new_text") or ""
        caps = {
            "name": SKILL_MAX_NAME_LEN,
            "description": SKILL_MAX_DESCRIPTION_LEN,
            "instructions": SKILL_MAX_INSTRUCTIONS_LEN,
        }
        if len(new_text) > caps[field]:
            return f"Validation error: 'new_text' for field '{field}' exceeds max length {caps[field]}."
    else:
        value = args.get("value")
        if value is None:
            return f"Validation error: 'value' is required for field '{field}'."
        if field == "entry_type" and value not in SKILL_ENTRY_TYPES:
            return (
                f"Validation error: 'entry_type' must be one of: {sorted(SKILL_ENTRY_TYPES)}."
            )
    return None


@register_validator("skill_delete")
def validate_skill_delete(function_name: str, args: dict) -> Optional[str]:
    """Validate skill_delete arguments."""
    if not args.get("skill_id", "").strip():
        return "Validation error: 'skill_id' cannot be empty."
    return None


# ============================================================
# Knowledge Vault Validators
# ============================================================


@register_validator("knowledge_vault_insert")
def validate_knowledge_vault_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate knowledge_vault_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("caption", "").strip():
            return (
                f"Validation error: 'caption' field in item {i} cannot be empty. "
                "Please provide a description for this knowledge vault entry."
            )
        if not item.get("secret_value", "").strip():
            return (
                f"Validation error: 'secret_value' field in item {i} cannot be empty. "
                "Please provide the credential or data value."
            )
    return None


@register_validator("knowledge_vault_update")
def validate_knowledge_vault_update(function_name: str, args: dict) -> Optional[str]:
    """Validate knowledge_vault_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("caption", "").strip():
            return (
                f"Validation error: 'caption' field in new_items[{i}] cannot be empty. "
                "Please provide a description for this knowledge vault entry."
            )
        if not item.get("secret_value", "").strip():
            return (
                f"Validation error: 'secret_value' field in new_items[{i}] cannot be empty. "
                "Please provide the credential or data value."
            )
    return None
