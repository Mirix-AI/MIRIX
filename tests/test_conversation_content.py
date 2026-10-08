"""Unified memory-add message parsing and the conversation-store text cap.

Pure, DB-free tests of
``mirix.services.conversation_content``: one parser for string / list-of-parts /
None content, feeding both the conversation store and the meta agent, and the
head + tail truncation that replaced the static 65,536-char schema cap.
"""

import base64
import io

import pytest
from PIL import Image

from mirix.services.conversation_content import (
    ImagePart,
    MessageParseError,
    PassthroughPart,
    TextPart,
    build_meta_input_messages,
    cap_text_parts,
    context_window_char_cap,
    effective_char_cap,
    iter_images,
    meta_agent_char_cap,
    parse_messages,
    serialize_tool_calls,
    store_turn_drafts,
    truncation_marker,
)


def _png_b64(width=3, height=2) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _drafts(messages):
    return store_turn_drafts(parse_messages(messages))


def _texts(messages):
    return [(d.role, d.to_turn({})["content"]) for d in _drafts(messages)]


# --------------------------------------------------------------------------- #
# parse_messages                                                               #
# --------------------------------------------------------------------------- #


class TestParseShapes:
    def test_string_content(self):
        [msg] = parse_messages([{"role": "user", "content": "hi"}])
        assert msg.role == "user" and msg.parts == [TextPart("hi")]

    def test_list_of_text_parts_and_bare_strings(self):
        [msg] = parse_messages(
            [{"role": "user", "content": [{"type": "text", "text": "a"}, "b"]}]
        )
        assert msg.parts == [TextPart("a"), TextPart("b")]

    def test_none_content_with_tool_calls(self):
        [msg] = parse_messages(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                }
            ]
        )
        assert msg.parts == [] and msg.tool_calls

    def test_inline_data_url_image(self):
        data = _png_b64()
        [msg] = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ]
        )
        image = msg.parts[1]
        assert isinstance(image, ImagePart) and not image.is_remote
        assert image.data_b64 == data
        assert image.declared_mime_type == "image/png"
        assert image.detail == "high"

    def test_image_data_part_bare_base64_and_data_url(self):
        data = _png_b64()
        [msg] = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_data", "image_data": {"data": data}},
                        {
                            "type": "image_data",
                            "image_data": {"data": f"data:image/png;base64,{data}"},
                        },
                    ],
                }
            ]
        )
        assert [p.data_b64 for p in msg.parts] == [data, data]
        assert msg.parts[0].declared_mime_type is None
        assert msg.parts[1].declared_mime_type == "image/png"

    def test_remote_image_url_is_recorded_not_inline(self):
        [msg] = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/a.png"},
                        }
                    ],
                }
            ]
        )
        assert msg.parts[0].is_remote and msg.parts[0].url == "https://x.test/a.png"

    def test_image_url_string_shorthand(self):
        [msg] = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "https://x.test/b.jpg"}
                    ],
                }
            ]
        )
        assert msg.parts[0].url == "https://x.test/b.jpg"

    def test_content_only_payload_is_one_roleless_message(self):
        parsed = parse_messages(
            [
                {"type": "text", "text": "screen"},
                {"type": "image_url", "image_url": {"url": "https://x.test/s.png"}},
            ]
        )
        assert len(parsed) == 1 and parsed[0].role is None
        assert len(parsed[0].parts) == 2

    def test_legacy_file_parts_pass_through(self):
        parsed = parse_messages(
            [{"type": "database_image_id", "image_id": "file-legacy"}]
        )
        assert isinstance(parsed[0].parts[0], PassthroughPart)
        assert parsed[0].parts[0].part_type == "database_image_id"

    def test_tool_results_that_are_json_become_json_text(self):
        parsed = parse_messages(
            [
                {
                    "role": "tool",
                    "name": "api",
                    "content": {"type": "error", "code": 7},
                },
                {"role": "tool", "content": [{"rows": 2}, 3, ["x"]]},
                {"role": "function", "content": 42},
            ]
        )
        assert parsed[0].parts == [TextPart('{"type": "error", "code": 7}')]
        assert parsed[1].parts == [
            TextPart('{"rows": 2}'),
            TextPart("3"),
            TextPart('["x"]'),
        ]
        assert parsed[2].parts == [TextPart("42")]

    def test_untyped_object_in_user_content_is_json_text(self):
        [msg] = parse_messages([{"role": "user", "content": [{"k": "v"}]}])
        assert msg.parts == [TextPart('{"k": "v"}')]

    def test_iter_images_in_order(self):
        parsed = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/1.png"},
                        }
                    ],
                },
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/2.png"},
                        }
                    ],
                },
            ]
        )
        assert [img.url for img in iter_images(parsed)] == [
            "https://x.test/1.png",
            "https://x.test/2.png",
        ]


class TestParseErrors:
    @pytest.mark.parametrize(
        "messages, fragment",
        [
            ([], "non-empty"),
            ("hello", "non-empty list"),
            (
                [{"role": "user", "content": "a"}, {"type": "text", "text": "b"}],
                "mixes",
            ),
            ([{"role": "", "content": "a"}], "role"),
            (
                [{"role": "user", "content": [{"type": "input_audio", "data": "x"}]}],
                "input_audio",
            ),
            (
                [{"role": "user", "content": [{"type": "text", "text": 5}]}],
                "string 'text'",
            ),
            (
                [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
                "url",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///etc/passwd"},
                            }
                        ],
                    }
                ],
                "http(s)",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/svg+xml;utf8,<svg/>"},
                            }
                        ],
                    }
                ],
                "base64",
            ),
            (
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_data", "image_data": {"data": ""}}],
                    }
                ],
                "empty",
            ),
            ([{"role": "user", "content": [{"type": "image_data"}]}], "image_data"),
            ([{"role": "user", "content": [["nested"]]}], "nested list"),
            ([{"type": "unknown_part"}], "unknown_part"),
            ([{"text": "a part without a type"}], "neither"),
            ([{"type": "database_image_id"}], "image_id"),
            (
                [{"role": "user", "content": [{"type": "file_uri", "file_uri": ""}]}],
                "file_uri",
            ),
            (["not-an-object"], "object"),
        ],
    )
    def test_rejected_with_a_clear_message(self, messages, fragment):
        with pytest.raises(MessageParseError) as exc:
            parse_messages(messages)
        assert fragment in str(exc.value)

    def test_parse_error_is_a_value_error(self):
        # Callers that only know ValueError still catch it.
        assert issubclass(MessageParseError, ValueError)


# --------------------------------------------------------------------------- #
# Conversation store drafts                                                    #
# --------------------------------------------------------------------------- #


class TestStoreTurnDrafts:
    def test_real_roles_kept_foreign_and_roleless_dropped(self):
        assert _texts(
            [
                {"role": "user", "content": "deploy"},
                {"role": "system", "content": "scaffolding"},
                {"role": "assistant", "content": "done"},
                {"role": "function", "name": "search", "content": "no results"},
            ]
        ) == [
            ("user", "deploy"),
            ("assistant", "done"),
            ("tool", "[search] no results"),
        ]
        assert _drafts([{"type": "text", "text": "screenshot"}]) == []

    def test_text_parts_join_with_newlines(self):
        assert _texts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        ) == [("user", "hello\nworld")]

    def test_tool_calls_serialized_onto_assistant_text(self):
        turns = _texts(
            [
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "grep", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"name": "read", "arguments": {"path": "a.py"}}],
                },
            ]
        )
        assert turns[0] == ("assistant", 'let me check\n[tool_call] grep({"q": "x"})')
        assert turns[1] == ("assistant", '[tool_call] read({"path": "a.py"})')

    def test_a_tool_result_takes_its_name_from_the_call_it_answers(self):
        """A tool result without a name is named after the assistant tool
        call its tool_call_id answers (in the same payload)."""
        turns = _texts(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "grep", "arguments": "{}"}},
                        {"id": "c2", "name": "read", "arguments": {}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c2", "content": "file text"},
                {"role": "tool", "tool_call_id": "c1", "content": "no matches"},
                {"role": "tool", "tool_call_id": "c9", "content": "unknown call"},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "name": "own",
                    "content": "named",
                },
            ]
        )
        assert turns[1:] == [
            ("tool", "[read] file text"),
            ("tool", "[grep] no matches"),
            ("tool", "unknown call"),
            ("tool", "[own] named"),
        ]
        parsed = parse_messages(
            [{"role": "tool", "tool_call_id": "c7", "content": "x"}]
        )
        assert parsed[0].tool_call_id == "c7" and parsed[0].name is None

    def test_malformed_tool_calls_do_not_raise(self):
        turns = _texts(
            [
                {"role": "assistant", "content": None, "tool_calls": "not-a-list"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [42, {"function": "not-a-dict"}],
                },
            ]
        )
        assert [role for role, _ in turns] == ["assistant", "assistant"]
        assert turns[0][1] == "not-a-list"

    def test_text_only_turn_has_no_content_parts(self):
        [draft] = _drafts(
            [{"role": "user", "content": [{"type": "text", "text": "a"}]}]
        )
        turn = draft.to_turn({})
        assert turn == {"role": "user", "content": "a"}
        assert "content_parts" not in turn

    def test_image_turn_keeps_part_order_and_decorations(self):
        [draft] = _drafts(
            [
                {
                    "role": "tool",
                    "name": "screenshot",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/1.png"},
                        },
                        {"type": "text", "text": "page 1"},
                    ],
                }
            ]
        )
        image = draft.images[0]
        ref = {"type": "image_ref", "file_id": "file-1", "url": "https://x.test/1.png"}
        turn = draft.to_turn({id(image): ref})
        assert turn["role"] == "tool"
        assert turn["content_parts"] == [
            {"type": "text", "text": "[screenshot]"},
            ref,
            {"type": "text", "text": "page 1"},
        ]

    def test_assistant_image_turn_appends_tool_calls(self):
        [draft] = _drafts(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/1.png"},
                        }
                    ],
                    "tool_calls": [{"name": "click", "arguments": "{}"}],
                }
            ]
        )
        ref = {"type": "image_ref", "file_id": "file-1"}
        parts = draft.to_turn({id(draft.images[0]): ref})["content_parts"]
        assert parts == [ref, {"type": "text", "text": "[tool_call] click({})"}]

    def test_passthrough_parts_become_placeholders_in_the_store(self):
        assert _texts(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "database_file_id", "file_id": "file-doc"},
                        "see file",
                    ],
                }
            ]
        ) == [("user", "[attachment: database_file_id]\nsee file")]

    def test_to_turn_rejects_a_non_image_ref(self):
        [draft] = _drafts(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x.test/1.png"},
                        }
                    ],
                }
            ]
        )
        with pytest.raises(ValueError):
            draft.to_turn({id(draft.images[0]): {"type": "text", "text": "nope"}})


def test_serialize_tool_calls_shapes():
    assert serialize_tool_calls(
        [{"function": {"name": "grep", "arguments": '{"q": "x"}'}}]
    ) == ('[tool_call] grep({"q": "x"})')
    assert serialize_tool_calls([{"name": "read", "arguments": {"path": "a.py"}}]) == (
        '[tool_call] read({"path": "a.py"})'
    )
    assert "[tool_call]" in serialize_tool_calls(["opaque"])


# --------------------------------------------------------------------------- #
# Meta agent input                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestMetaInputMessages:
    async def _build(self, messages, monkeypatch, session_id=None):
        stored = []

        async def fake_store(
            part, *, organization_id, user_id, file_manager=None, images_dir=None
        ):
            stored.append((part, organization_id, user_id))
            return {"type": "image_ref", "file_id": f"file-{len(stored)}"}

        import mirix.services.image_store as image_store

        monkeypatch.setattr(image_store, "store_image_part", fake_store)
        result = await build_meta_input_messages(
            parse_messages(messages),
            organization_id="org-1",
            user_id="user-1",
            session_id=session_id,
        )
        return result, stored

    async def test_list_of_text_parts_no_longer_crashes(self, monkeypatch):
        """`prefix + " " + part` with a dict part used to be a TypeError (500)."""
        [msg], stored = await self._build(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                },
                {"role": "assistant", "content": "hi"},
            ],
            monkeypatch,
            session_id="sess-meta",
        )
        assert [c.text for c in msg.content] == [
            "[USER] hello\nworld",
            "[ASSISTANT] hi",
        ]
        assert msg.session_id == "sess-meta"
        assert stored == []

    async def test_images_become_image_content(self, monkeypatch):
        [msg], stored = await self._build(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://x.test/1.png",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": "what is it?"},
                    ],
                },
                {"role": "tool", "name": "ocr", "content": "text: 42"},
            ],
            monkeypatch,
        )
        kinds = [
            (c.type.value, getattr(c, "text", None) or getattr(c, "image_id", None))
            for c in msg.content
        ]
        assert kinds == [
            ("text", "[USER]"),
            ("image_url", "file-1"),
            ("text", "what is it?"),
            ("text", "[TOOL] [ocr] text: 42"),
        ]
        assert msg.content[1].detail == "low"
        assert stored[0][1:] == ("org-1", "user-1")

    async def test_content_only_payload_keeps_unprefixed_parts(self, monkeypatch):
        [msg], _ = await self._build(
            [
                {"type": "text", "text": "screen"},
                {"type": "image_url", "image_url": {"url": "https://x.test/s.png"}},
            ],
            monkeypatch,
        )
        assert msg.content[0].text == "screen"
        assert msg.content[1].image_id == "file-1"

    async def test_legacy_database_ids_pass_through(self, monkeypatch):
        [msg], _ = await self._build(
            [
                {"type": "database_image_id", "image_id": "file-old"},
                {"type": "database_file_id", "file_id": "file-doc"},
            ],
            monkeypatch,
        )
        assert msg.content[0].image_id == "file-old"
        assert msg.content[1].file_id == "file-doc"


# --------------------------------------------------------------------------- #
# Storage cap                                                                  #
# --------------------------------------------------------------------------- #


class TestCapValues:
    def test_spec_table_values(self):
        # floor(window * 0.8) * 4 characters.
        assert context_window_char_cap(1_050_000) == 3_360_000
        assert context_window_char_cap(1_310_720) == 4_194_304
        assert context_window_char_cap(128_000) == 409_600

    def test_floor_is_taken_before_times_four(self):
        # floor(1001 * 0.8) * 4 = 800 * 4, not floor(3203.2)
        assert context_window_char_cap(1001) == 3200

    @pytest.mark.parametrize("bad", [None, 0, -5, "128000", 12.5, True])
    def test_unknown_context_window_means_no_model_cap(self, bad):
        assert context_window_char_cap(bad) is None

    def test_cap_follows_the_meta_agents_context_window(self):
        from types import SimpleNamespace

        small = SimpleNamespace(llm_config=SimpleNamespace(context_window=1000))
        big = SimpleNamespace(llm_config=SimpleNamespace(context_window=200_000))
        assert meta_agent_char_cap(small) == 3200
        assert meta_agent_char_cap(big) == 640_000
        assert meta_agent_char_cap(SimpleNamespace()) is None

    def test_env_override_takes_the_minimum(self, monkeypatch):
        monkeypatch.delenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", raising=False)
        assert effective_char_cap(3200) == 3200
        assert effective_char_cap(None) is None
        monkeypatch.setenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", "1000")
        assert effective_char_cap(3200) == 1000
        assert effective_char_cap(None) == 1000
        monkeypatch.setenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", "99999")
        assert effective_char_cap(3200) == 3200

    @pytest.mark.parametrize("raw", ["", "abc", "0", "-3"])
    def test_invalid_env_override_is_ignored(self, monkeypatch, raw):
        monkeypatch.setenv("MIRIX_CONVERSATION_MESSAGE_MAX_CHARS", raw)
        assert effective_char_cap(3200) == 3200


class TestCapTextParts:
    def test_under_the_cap_is_untouched(self):
        assert cap_text_parts(["abc", "de"], 5) == (["abc", "de"], 0)
        assert cap_text_parts(["abc"], None) == (["abc"], 0)

    def test_head_and_tail_with_exact_marker(self):
        text = "H" * 500 + "M" * 1000 + "T" * 500
        [out], removed = cap_text_parts([text], 1000)
        assert len(out) <= 1000
        assert out.startswith("H" * 400) and out.endswith("T" * 400)
        marker = truncation_marker(removed)
        assert marker in out
        assert removed == len(text) - (len(out) - len(marker))
        head, tail = out.split(marker)
        assert abs(len(head) - len(tail)) <= 1

    def test_longest_part_is_cut_first(self):
        short, long_ = "s" * 300, "L" * 3000
        out, removed = cap_text_parts([short, long_], 1500)
        assert out[0] == short
        assert len(out[0]) + len(out[1]) <= 1500
        assert truncation_marker(removed) in out[1]

    def test_next_longest_part_is_cut_when_still_over(self):
        out, removed = cap_text_parts(["a" * 1000, "b" * 900, "c" * 50], 200)
        assert sum(len(t) for t in out) <= 200
        assert out[2] == "c" * 50
        assert removed > 0

    def test_marker_counts_are_exact_across_parts(self):
        import re

        out, removed = cap_text_parts(["x" * 10_000, "y" * 9_000], 2_000)
        counted = sum(
            int(n) for n in re.findall(r"truncated (\d+) chars", "".join(out))
        )
        assert counted == removed

    def test_tiny_cap_degrades_gracefully(self):
        out, removed = cap_text_parts(["x" * 100], 5)
        assert out == [truncation_marker(100)]
        assert removed == 100



# --------------------------------------------------------------------------- #
# Edge cases                                                                   #
# --------------------------------------------------------------------------- #


class TestServerLocalPartsAndDetail:
    """Server-local file parts and the image detail, at the parser."""

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "file_uri", "file_uri": "/etc/hosts"},
            {"type": "file_uri", "file_uri": "file:///root/.ssh/id_rsa"},
            {"type": "google_cloud_file_uri", "google_cloud_file_uri": "gs://b/x.png"},
        ],
    )
    def test_server_local_parts_are_rejected(self, part):
        for messages in ([part], [{"role": "user", "content": [part]}]):
            with pytest.raises(MessageParseError) as exc:
                parse_messages(messages)
            assert "not accepted over the API" in str(exc.value)
            assert part.get("file_uri", "@@") not in str(exc.value)

    @pytest.mark.parametrize("detail", ["auto", "low", "high", None, ""])
    def test_valid_details(self, detail):
        spec = {"url": "https://x.test/a.png"}
        if detail is not None:
            spec["detail"] = detail
        [msg] = parse_messages([{"type": "image_url", "image_url": spec}])
        assert msg.parts[0].detail == (detail or None)

    @pytest.mark.parametrize("detail", ["ultra", 5, ["high"]])
    def test_invalid_details_are_rejected(self, detail):
        with pytest.raises(MessageParseError, match="detail must be one of"):
            parse_messages(
                [
                    {
                        "type": "image_data",
                        "image_data": {"data": "aGk=", "detail": detail},
                    }
                ]
            )

    def test_referenced_file_ids(self):
        from mirix.services.conversation_content import referenced_file_ids

        parsed = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "database_image_id", "image_id": "file-a"},
                        {"type": "database_file_id", "file_id": "file-b"},
                        {"type": "database_image_id", "image_id": "file-a"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "database_google_cloud_file_uri",
                            "cloud_file_uri": "file-c",
                        }
                    ],
                },
            ]
        )
        assert referenced_file_ids(parsed) == ["file-a", "file-b", "file-c"]


@pytest.mark.asyncio
class TestMetaBuildCleanup:
    async def test_images_stored_before_a_failure_are_discarded(self, monkeypatch):
        """If building the meta input fails part-way, the image copies it
        already stored are released instead of leaking."""
        import mirix.services.image_store as image_store

        stored, discarded = [], []

        async def fake_store(part, **kwargs):
            if stored:
                raise RuntimeError("disk full")
            ref = {"type": "image_ref", "file_id": f"file-{len(stored)}"}
            stored.append(ref)
            return ref

        async def fake_discard(refs, **kwargs):
            discarded.extend(refs)

        monkeypatch.setattr(image_store, "store_image_part", fake_store)
        monkeypatch.setattr(image_store, "discard_image_refs", fake_discard)
        parsed = parse_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://a.test/1.png"}},
                        {"type": "image_url", "image_url": {"url": "https://a.test/2.png"}},
                    ],
                }
            ]
        )
        collected = []
        with pytest.raises(RuntimeError, match="disk full"):
            await build_meta_input_messages(
                parsed, organization_id="org", user_id="u", stored_refs=collected
            )
        assert discarded == stored and collected == []
