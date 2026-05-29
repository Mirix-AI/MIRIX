"""Tests for evals.metaclaw.format_adapter."""
from evals.metaclaw.format_adapter import (
    mirix_to_metaclaw,
    round_to_message,
    RoundResult,
)


def test_mirix_to_metaclaw_maps_required_fields():
    mirix_skill = {
        "id": "proc-abc",
        "name": "iso8601-with-cst-offset",
        "description": "Format datetimes as YYYY-MM-DDTHH:MM:SS+08:00.",
        "instructions": "When asked to record any datetime field...",
        "entry_type": "guide",
        "version": "0.1.0",
    }
    out = mirix_to_metaclaw(mirix_skill)
    assert out["name"] == "iso8601-with-cst-offset"
    assert out["description"].startswith("Format datetimes")
    assert out["content"].startswith("When asked")
    assert out["category"] == "guide"


def test_mirix_to_metaclaw_defaults_category_when_missing():
    out = mirix_to_metaclaw({
        "name": "x", "description": "d", "instructions": "i"
    })
    assert out["category"] == "general"


def test_round_to_message_includes_outcome_and_feedback():
    r = RoundResult(
        round_id="r1",
        round_type="file_check",
        question="Save standup notes...",
        final_answer="Wrote day01/standup.json",
        reward=0.0,
        eval_outcome="fail",
        feedback="Time fields must use full datetime with +08:00 offset.",
    )
    msg = round_to_message(r)
    assert "r1" in msg
    assert "FAIL" in msg or "fail" in msg.lower()
    assert "+08:00" in msg              # feedback content carried through
    assert "Save standup notes" in msg


def test_round_to_message_handles_multichoice():
    r = RoundResult(
        round_id="r3",
        round_type="multi_choice",
        question="Which formats are valid?",
        final_answer="\\bbox{A,E}",
        reward=1.0,
        eval_outcome="pass",
        feedback="Correct! A and E both valid.",
    )
    msg = round_to_message(r)
    assert "PASS" in msg or "pass" in msg.lower()
    assert "\\bbox{A,E}" in msg
