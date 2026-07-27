"""Unit tests for flatten_messages_for_agent's single-content-block packing.

The packed agent input must carry all consecutive text (role markers + turn
content) as one "\n"-joined TextContent so per-block downstream scanners see
a single block, while non-text items pass through unchanged as their own parts.
Pure function tests — no fixtures, DB, or server.
"""

import pytest

from mirix.schemas.mirix_message_content import ImageContent, TextContent
from mirix.utils import flatten_messages_for_agent


def test_all_string_turns_produce_single_text_block():
    messages = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 1
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]\nhi there\n[ASSISTANT]\nhello!"


def test_markers_ordering_and_role_ternary_preserved():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "assistant", "content": "third"},
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 1
    # Any non-"user" role (e.g. "system") is marked [ASSISTANT]; input order kept,
    # each marker immediately followed by its content line.
    assert content[0].text.split("\n") == [
        "[USER]",
        "first",
        "[ASSISTANT]",
        "second",
        "[ASSISTANT]",
        "third",
    ]


def test_list_content_all_text_coalesces_into_single_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "text", "text": "part two"},
            ],
        },
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 1
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]\npart one\npart two"


def test_non_text_item_survives_with_text_coalesced_around_it():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before image"},
                {"type": "database_image_id", "image_id": "img-1"},
                {"type": "text", "text": "after image"},
            ],
        },
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 3
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]\nbefore image"
    assert isinstance(content[1], ImageContent)
    assert content[1].image_id == "img-1"
    assert isinstance(content[2], TextContent)
    assert content[2].text == "after image"


def test_trailing_non_text_item_flushes_prior_text_only():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "some text"},
                {"type": "database_image_id", "image_id": "img-2"},
            ],
        },
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 2
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]\nsome text"
    assert isinstance(content[1], ImageContent)
    assert content[1].image_id == "img-2"


def test_text_item_missing_text_key_coalesces_to_empty_line():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text"},
                {"type": "text", "text": "real text"},
            ],
        },
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 1
    # A text item with no "text" key contributes an empty line instead of
    # raising — malformed items degrade to blank rather than dropping the turn.
    assert content[0].text == "[USER]\n\nreal text"


def test_leading_non_text_item_flushes_marker_alone():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "database_image_id", "image_id": "img-3"},
                {"type": "text", "text": "caption"},
            ],
        },
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 3
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]"
    assert isinstance(content[1], ImageContent)
    assert content[1].image_id == "img-3"
    assert isinstance(content[2], TextContent)
    assert content[2].text == "caption"


def test_text_recoalesces_across_turns_after_non_text_item():
    messages = [
        {"role": "user", "content": "look at this"},
        {
            "role": "user",
            "content": [{"type": "database_image_id", "image_id": "img-4"}],
        },
        {"role": "assistant", "content": "nice photo"},
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 3
    assert isinstance(content[0], TextContent)
    assert content[0].text == "[USER]\nlook at this\n[USER]"
    assert isinstance(content[1], ImageContent)
    assert content[1].image_id == "img-4"
    assert isinstance(content[2], TextContent)
    assert content[2].text == "[ASSISTANT]\nnice photo"


def test_empty_messages_returns_empty_list():
    assert flatten_messages_for_agent([]) == []


def test_empty_string_content_keeps_marker_line():
    messages = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "hello!"},
    ]

    result = flatten_messages_for_agent(messages)

    assert len(result) == 1
    content = result[0].content
    assert len(content) == 1
    # Empty turn contributes marker + empty line: consecutive newlines,
    # matching the legacy "\n".join(["[USER]", ""]) wire behavior.
    assert content[0].text == "[USER]\n\n[ASSISTANT]\nhello!"


def test_invalid_content_type_raises_value_error():
    with pytest.raises(ValueError, match="Invalid content type"):
        flatten_messages_for_agent([{"role": "user", "content": 42}])
