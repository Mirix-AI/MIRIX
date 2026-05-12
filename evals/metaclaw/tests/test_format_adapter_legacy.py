"""Unit tests for legacy_procedural_to_metaclaw — the adapter that maps
main-branch procedural_memory rows ({summary, steps, entry_type}) to the
metaclaw skill-shape ({name, description, content, category}).
"""
from evals.metaclaw.format_adapter import legacy_procedural_to_metaclaw


def test_full_row_maps_all_fields():
    row = {
        "id": "proc-1",
        "summary": "Format dates as ISO 8601",
        "steps": "1. Identify date 2. Convert to YYYY-MM-DDTHH:MM:SSZ",
        "entry_type": "guide",
    }
    out = legacy_procedural_to_metaclaw(row)
    assert out == {
        "name": "guide",
        "description": "Format dates as ISO 8601",
        "content": "1. Identify date 2. Convert to YYYY-MM-DDTHH:MM:SSZ",
        "category": "guide",
    }


def test_missing_entry_type_defaults_to_procedure():
    out = legacy_procedural_to_metaclaw({"summary": "x", "steps": "y"})
    assert out["name"] == "procedure"
    assert out["category"] == "procedure"


def test_missing_summary_and_steps_yield_empty_strings():
    out = legacy_procedural_to_metaclaw({"entry_type": "workflow"})
    assert out == {
        "name": "workflow",
        "description": "",
        "content": "",
        "category": "workflow",
    }


def test_null_values_treated_as_missing():
    out = legacy_procedural_to_metaclaw(
        {"summary": None, "steps": None, "entry_type": None}
    )
    assert out == {
        "name": "procedure",
        "description": "",
        "content": "",
        "category": "procedure",
    }
