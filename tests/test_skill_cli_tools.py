"""Tests for CLI-style skill tools, validators, and constants."""
import pytest


def test_bump_patch_version():
    """Test version bumping."""
    # Import via importlib to avoid heavy dependency chain (rapidfuzz, etc.)
    import importlib.util, types, sys

    spec = importlib.util.spec_from_file_location(
        "_memory_tools_isolated",
        "mirix/functions/function_sets/memory_tools.py",
    )
    # We only need to extract the pure function source, not execute the whole module.
    # Read the function directly to avoid import side-effects.
    import ast, textwrap

    with open("mirix/functions/function_sets/memory_tools.py") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_bump_patch_version":
            func_source = ast.get_source_segment(source, node)
            break
    else:
        pytest.fail("_bump_patch_version not found in memory_tools.py")

    ns = {}
    exec(func_source, ns)
    _bump_patch_version = ns["_bump_patch_version"]

    assert _bump_patch_version("0.1.0") == "0.1.1"
    assert _bump_patch_version("0.1.9") == "0.1.10"
    assert _bump_patch_version("1.2.3") == "1.2.4"
    assert _bump_patch_version("invalid") == "0.1.1"
    assert _bump_patch_version("") == "0.1.1"


def test_skill_tools_constant():
    """SKILL_TOOLS contains all 5 CLI tools."""
    from mirix.constants import SKILL_TOOLS

    assert SKILL_TOOLS == ["skill_list", "skill_read", "skill_create", "skill_edit", "skill_delete"]


def test_skill_curation_tools_constant():
    """experience_triage is a procedural-agent tool of its own:
    not in SKILL_TOOLS (the auto-dream agent never curates a batch), but a
    registered tool."""
    from mirix.constants import ALL_TOOLS, AUTO_DREAM_TOOLS, SKILL_CURATION_TOOLS

    assert SKILL_CURATION_TOOLS == ["experience_triage"]
    assert "experience_triage" in ALL_TOOLS
    assert "experience_triage" not in AUTO_DREAM_TOOLS


def test_trigger_threshold_constant():
    """SKILL_TRIGGER_MESSAGE_THRESHOLD exists and defaults to 10."""
    from mirix.constants import SKILL_TRIGGER_MESSAGE_THRESHOLD

    assert isinstance(SKILL_TRIGGER_MESSAGE_THRESHOLD, int)
    assert SKILL_TRIGGER_MESSAGE_THRESHOLD == 10


def test_skill_validators_registered():
    """Validators are registered for skill_create, skill_edit, skill_delete."""
    # Import validator module directly to avoid heavy dependency chain
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tool_validators",
        "mirix/agent/tool_validators.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # skill_create: missing name
    result = mod.validate_tool_args(
        "skill_create",
        {"name": "", "description": "x", "instructions": "x", "entry_type": "workflow"},
    )
    assert result is not None
    assert "name" in result

    # skill_create: valid
    result = mod.validate_tool_args(
        "skill_create",
        {"name": "x", "description": "x", "instructions": "x", "entry_type": "workflow"},
    )
    assert result is None

    # skill_edit: missing field
    result = mod.validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": ""})
    assert result is not None

    # skill_edit: text field with neither a patch nor a whole-field value
    result = mod.validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": "instructions"})
    assert result is not None
    assert "old_text" in result and "value" in result

    # skill_edit: a whole-field rewrite of a text field is valid (no size gate)
    result = mod.validate_tool_args(
        "skill_edit", {"skill_id": "proc-1", "field": "instructions", "value": "x" * 5000}
    )
    assert result is None

    # skill_edit: malformed arguments are a validation error, not a crash
    result = mod.validate_tool_args("skill_edit", {"skill_id": None, "field": "name"})
    assert result is not None and "invalid arguments" in result

    # skill_edit: value field valid
    result = mod.validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": "triggers", "value": ["a"]})
    assert result is None

    # skill_edit: invalid field
    result = mod.validate_tool_args("skill_edit", {"skill_id": "proc-1", "field": "bogus", "value": "x"})
    assert result is not None
    assert "must be one of" in result

    # skill_delete: valid (a deletion always names its reason)
    result = mod.validate_tool_args(
        "skill_delete", {"skill_id": "proc-1", "reason": "wrong advice"}
    )
    assert result is None

    # skill_delete: no reason
    result = mod.validate_tool_args("skill_delete", {"skill_id": "proc-1"})
    assert result is not None and "reason" in result
    result = mod.validate_tool_args("skill_delete", {"skill_id": "proc-1", "reason": "  "})
    assert result is not None and "reason" in result

    # skill_delete: empty
    result = mod.validate_tool_args("skill_delete", {"skill_id": ""})
    assert result is not None
