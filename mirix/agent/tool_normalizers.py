"""
Tool argument normalizer registry.

Mirrors tool_validators.py's shape: LLM-emitted item fields that MIRIX treats
as fixed constants (the prompt tells the model to always emit the same
value) sometimes get omitted from the tool call's JSON entirely, especially
by smaller models (see ECMS-534). A bare `item["field"]` access on a missing
key crashes the whole save instead of degrading gracefully.

Usage:
    @register_normalizer("episodic_memory_insert", "episodic_memory_replace")
    def normalize_episodic_memory(function_name: str, args: dict) -> None:
        '''Mutates args in place, filling in omitted constant fields.'''
        ...

    # In agent.py, before validate_tool_args:
    normalize_tool_args(function_name, function_args)
"""

from typing import Callable, Dict

from mirix.log import get_logger

logger = get_logger(__name__)

# Registry: tool_name -> normalizer_function
_NORMALIZERS: Dict[str, Callable[[str, dict], None]] = {}


def register_normalizer(*tool_names: str):
    """
    Decorator to register a normalizer function for one or more tools.

    The normalizer function signature: (function_name: str, args: dict) -> None
    Mutates ``args`` in place; does not return a value.
    """

    def decorator(func: Callable[[str, dict], None]):
        for name in tool_names:
            _NORMALIZERS[name] = func
        return func

    return decorator


def normalize_tool_args(function_name: str, function_args: dict) -> None:
    """
    Apply the registered normalizer (if any) for this tool, mutating
    ``function_args`` in place. No-op if no normalizer is registered.
    """
    normalizer = _NORMALIZERS.get(function_name)
    if normalizer:
        normalizer(function_name, function_args)


# ============================================================
# Constant-field defaults registry.
#
# Each entry maps a tool name to (items_key, {item_field: default_value}).
# The prompt mandates a fixed value for each of these fields (never a model
# judgment call) — see the intuit-tuned agent prompts in
# context-and-memory-service/configs/prompts/intuit-tuned/. Adding a new
# constant field to an existing tool, or a new tool with one, is a one-line
# addition to this table rather than a fresh .get() call site sweep.
# ============================================================

_CONSTANT_FIELD_DEFAULTS: Dict[str, tuple] = {
    "episodic_memory_insert": ("items", {"actor": "user", "event_type": "user_message"}),
    "episodic_memory_replace": ("new_items", {"actor": "user", "event_type": "user_message"}),
    "semantic_memory_insert": ("items", {"source": "user message"}),
    "semantic_memory_update": ("new_items", {"source": "user message"}),
    "knowledge_vault_insert": ("items", {"source": "user message"}),
    "knowledge_vault_update": ("new_items", {"source": "user message"}),
}


@register_normalizer(*_CONSTANT_FIELD_DEFAULTS.keys())
def normalize_constant_fields(function_name: str, args: dict) -> None:
    """Fill in omitted prompt-mandated constant fields with their default.

    Extraction-path clients (memory-ingest-client, sales-batch-ingest) feed
    raw record text as a synthetic user message with no real user/assistant
    turn, so the LLM sometimes has no natural value to supply and omits the
    key. Defaulting here (rather than at each memory_tools.py call site)
    keeps every constant-field default in one table (ECMS-534).
    """
    items_key, defaults = _CONSTANT_FIELD_DEFAULTS[function_name]
    items = args.get(items_key)
    if not items:
        return
    for item in items:
        for field, default in defaults.items():
            # setdefault only fills in a truly missing key (not an empty/
            # falsy value the LLM did supply), matching the .get(field,
            # default) semantics this replaced. Log only the omission case —
            # this is the metric that tells us whether prompt compliance is
            # holding up in prod, not a per-save no-op (ECMS-534).
            if field not in item:
                logger.warning(
                    "Defaulted omitted constant field (ECMS-534) "
                    "field=%s default_value=%r function_name=%s",
                    field,
                    default,
                    function_name,
                )
            item.setdefault(field, default)
