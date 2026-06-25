"""Tests for parsing ALFWorld model actions."""

from __future__ import annotations

from evals.alfworld.actions import parse_model_response
from evals.alfworld.actions import parse_action


def test_parse_model_response_extracts_think_and_action_tags() -> None:
    parsed = parse_model_response(
        "<think>I need to inspect the countertop first.</think>\n"
        "<action>look at countertop</action>"
    )

    assert parsed == {
        "thought": "I need to inspect the countertop first.",
        "action": "look at countertop",
    }


def test_parse_model_response_falls_back_when_action_tag_missing() -> None:
    parsed = parse_model_response(
        "I should open the fridge before taking anything.\n"
        "open fridge"
    )

    assert parsed == {
        "thought": "I should open the fridge before taking anything.",
        "action": "open fridge",
    }


def test_parse_model_response_supports_json_action_payload() -> None:
    parsed = parse_model_response(
        '{"thought": "The apple needs to be cooled.", "action": "put apple in fridge"}'
    )

    assert parsed == {
        "thought": "The apple needs to be cooled.",
        "action": "put apple in fridge",
    }


def test_parse_model_response_supports_json_action_only_payload() -> None:
    parsed = parse_model_response('{"action": "take mug from counter"}')

    assert parsed == {
        "thought": "",
        "action": "take mug from counter",
    }


def test_parse_action_defaults_to_skillopt_xml_contract() -> None:
    parsed = parse_action("<think>Inspect first.</think><action>Open Fridge</action>")

    assert parsed.action == "open fridge"
    assert parsed.format_valid is True
    assert parsed.source == "xml"
    assert parsed.used_fallback is False


def test_parse_action_rejects_missing_think_for_format_validity() -> None:
    parsed = parse_action("<action>open fridge</action>")

    assert parsed.action == "open fridge"
    assert parsed.format_valid is False


def test_parse_action_does_not_accept_json_by_default() -> None:
    parsed = parse_action('{"action": "open fridge"}')

    assert parsed.action == "look"
    assert parsed.used_fallback is True
