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
# Procedural Memory Validators
# ============================================================


@register_validator("procedural_memory_insert")
def validate_procedural_memory_insert(function_name: str, args: dict) -> Optional[str]:
    """Validate procedural_memory_insert arguments."""
    items = args.get("items", [])
    for i, item in enumerate(items):
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in item {i} cannot be empty. "
                "Please provide a descriptive summary of this procedure."
            )
        steps = item.get("steps", [])
        if not steps or all(not s.strip() for s in steps):
            return (
                f"Validation error: 'steps' field in item {i} cannot be empty. "
                "Please provide at least one non-empty step."
            )
    return None


@register_validator("procedural_memory_update")
def validate_procedural_memory_update(function_name: str, args: dict) -> Optional[str]:
    """Validate procedural_memory_update arguments."""
    items = args.get("new_items", [])
    for i, item in enumerate(items):
        if not item.get("summary", "").strip():
            return (
                f"Validation error: 'summary' field in new_items[{i}] cannot be empty. "
                "Please provide a descriptive summary of this procedure."
            )
        steps = item.get("steps", [])
        if not steps or all(not s.strip() for s in steps):
            return (
                f"Validation error: 'steps' field in new_items[{i}] cannot be empty. "
                "Please provide at least one non-empty step."
            )
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
