"""Retrieval input and topic extraction.

Pure parts (message normalization, plain text, the extraction input and its
budget, the prompt) are tested directly; the extraction call itself runs the
real OpenAI-compatible client against the offline fake model server
(``fake_llm_server``) — nothing reaches a provider.
"""

import base64
import io
import json
import os
import re

import pytest
from PIL import Image

from mirix.schemas.llm_config import LLMConfig
from mirix.services import topic_extraction as te
from mirix.services.context_budget import (
    UNKNOWN_IMAGE_TOKENS,
    ModelLimits,
    TokenCounter,
    image_tokens,
)
from mirix.services.conversation_content import MessageParseError
from mirix.services.distill_transcript import (
    IMAGE_LABEL_TOKENS,
    ImagePiece,
    TextPiece,
    head_tail_pieces,
)
from mirix.services.image_store import prepare_images

pytestmark = pytest.mark.asyncio


class _CharCounter:
    """1 token per character: exact, predictable budgets."""

    def encode(self, text, **_kwargs):
        return list(text)


COUNTER = TokenCounter(encoding=_CharCounter())


def _png(size=(40, 30), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _data_url(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _image_part(raw: bytes, detail=None):
    spec = {"url": _data_url(raw)}
    if detail:
        spec["detail"] = detail
    return {"type": "image_url", "image_url": spec}


def _text(pieces):
    return "".join(p.text if isinstance(p, TextPiece) else "<IMG>" for p in pieces)


# --------------------------------------------------------------------------- #
# Message shapes                                                               #
# --------------------------------------------------------------------------- #


class TestNormalize:
    def test_string_content_is_text(self):
        """String content used to count as 'no text'."""
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": "find the invoice total"}]
        )
        assert te.plain_text(parsed) == "USER: find the invoice total"

    def test_list_of_parts_is_text(self):
        """A list of parts used to raise TypeError inside the extractor."""
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ]
        )
        assert te.plain_text(parsed) == "USER: part one part two"

    def test_none_content_with_tool_calls_is_kept(self):
        parsed = te.normalize_retrieval_messages(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "open_file", "arguments": '{"p": 1}'}}
                    ],
                },
                {"role": "tool", "name": "open_file", "content": "done"},
            ]
        )
        text = te.plain_text(parsed)
        assert 'ASSISTANT: [tool_call] open_file({"p": 1})' in text
        assert "TOOL: [open_file] done" in text

    def test_bare_parts_payload(self):
        parsed = te.normalize_retrieval_messages([{"type": "text", "text": "bare"}])
        assert te.plain_text(parsed) == "USER: bare"

    @pytest.mark.parametrize("messages", [[], None])
    def test_empty_is_an_input_error(self, messages):
        with pytest.raises(te.RetrievalInputError):
            te.normalize_retrieval_messages(messages)

    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": "   \n"}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": []}],
            [{"role": "user", "content": [{"type": "text", "text": "  "}]}],
        ],
    )
    def test_no_text_and_no_image_is_an_input_error(self, messages):
        """Nothing to retrieve with is a 422, never 'the most recent N'."""
        with pytest.raises(te.RetrievalInputError):
            te.normalize_retrieval_messages(messages)

    def test_an_image_alone_is_content(self):
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": [_image_part(_png())]}]
        )
        assert te.has_retrieval_content(parsed)
        assert te.plain_text(parsed) == ""  # images are not text

    def test_server_file_parts_are_rejected(self):
        with pytest.raises(MessageParseError):
            te.normalize_retrieval_messages(
                [
                    {
                        "role": "user",
                        "content": [{"type": "file_uri", "file_uri": "/etc/passwd"}],
                    }
                ]
            )

    def test_input_error_is_a_parse_error(self):
        """The REST layer maps every MessageParseError to 422."""
        assert issubclass(te.RetrievalInputError, MessageParseError)


class TestFallbackQuery:
    def test_short_text_unchanged(self):
        assert te.fallback_query("abc", 10) == ("abc", 0)

    def test_long_text_keeps_head_and_tail_without_a_marker(self):
        text = "a" * 60 + "b" * 40
        query, removed = te.fallback_query(text, 10)
        assert query == "aaaaa\nbbbbb" and removed == 90
        assert "truncated" not in query

    def test_zero_cap_keeps_everything(self):
        assert te.fallback_query("abc", 0) == ("abc", 0)


# --------------------------------------------------------------------------- #
# Extraction input: pieces, images, budget                                     #
# --------------------------------------------------------------------------- #


class TestConversationPieces:
    async def test_roles_prefixed_and_images_in_place(self):
        raw = _png((40, 30))
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        _image_part(raw, detail="low"),
                        {"type": "text", "text": "after"},
                    ],
                },
                {"role": "assistant", "content": "ok"},
            ]
        )
        await prepare_images(p for m in parsed for p in m.images)
        pieces, images = te.conversation_pieces(parsed)
        assert _text(pieces) == "[USER] before\n<IMG>\nafter\n[ASSISTANT] ok"
        [image] = [p for p in pieces if isinstance(p, ImagePiece)]
        assert image.detail == "low"
        assert image.tokens == image_tokens(40, 30)
        assert set(images) == {image.file_id}
        assert not images[image.file_id].stored
        assert image.file_id.startswith("retrieval-image-")

    def test_role_labels(self):
        parsed = te.normalize_retrieval_messages(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "tool", "content": "t"},
                {"role": "critic", "content": "c"},
            ]
        )
        pieces, _ = te.conversation_pieces(parsed)
        assert _text(pieces) == (
            "[SYSTEM] be brief\n[USER] u\n[ASSISTANT] a\n[TOOL] t\n[CRITIC] c"
        )

    def test_remote_image_costs_the_unknown_estimate(self):
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        }
                    ],
                }
            ]
        )
        pieces, images = te.conversation_pieces(parsed)
        [image] = [p for p in pieces if isinstance(p, ImagePiece)]
        assert image.tokens == UNKNOWN_IMAGE_TOKENS
        assert images[image.file_id].part.url == "https://example.com/a.png"

    def test_stored_references(self):
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "database_image_id", "image_id": "file-img"},
                        {"type": "database_file_id", "file_id": "file-doc"},
                    ],
                }
            ]
        )
        pieces, images = te.conversation_pieces(parsed)
        assert "[attachment: database_file_id]" in _text(pieces)
        assert images["file-img"].stored
        assert "file-doc" not in images


class TestHeadTailPieces:
    def test_fitting_content_is_unchanged(self):
        pieces = [TextPiece("hello"), TextPiece(" world")]
        cut = head_tail_pieces(pieces, 100, COUNTER)
        assert _text(cut.pieces) == "hello world" and not cut.truncated

    def test_text_keeps_head_and_tail_with_a_marker(self):
        text = "H" * 500 + "M" * 1000 + "T" * 500
        cut = head_tail_pieces([TextPiece(text)], 300, COUNTER)
        out = _text(cut.pieces)
        assert out.startswith("H") and out.endswith("T") and "M" * 10 not in out
        assert cut.tokens <= 300 and cut.tokens == len(out)
        assert cut.removed_chars == len(text) - out.count("H") - out.count("T")
        assert f"truncated {cut.removed_chars} chars" in out

    def test_images_are_never_split(self):
        """An image is kept whole or left out; the marker counts the dropped."""
        big = ImagePiece(file_id="big", detail=None, tokens=400)
        small = ImagePiece(file_id="small", detail=None, tokens=20)
        pieces = [
            TextPiece("A" * 100),
            big,
            TextPiece("B" * 400),
            small,
            TextPiece("C" * 100),
        ]
        cut = head_tail_pieces(pieces, 400, COUNTER)
        kept = [p for p in cut.pieces if isinstance(p, ImagePiece)]
        assert [p.file_id for p in kept] == ["small"]
        assert cut.dropped_images == 1
        assert "1 image(s)" in _text(cut.pieces)
        assert cut.tokens <= 400
        # The kept image is intact (same object: never re-cut).
        assert kept[0] is small

    def test_tiny_budget_is_the_marker_alone(self):
        cut = head_tail_pieces([TextPiece("x" * 1000)], 5, COUNTER)
        assert len(cut.pieces) == 1 and "truncated 1000 chars" in cut.pieces[0].text
        assert cut.removed_chars == 1000


class TestBuildExtractionInput:
    async def test_within_budget(self):
        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "hi"}])
        built = te.build_extraction_input(parsed, 1000, COUNTER)
        assert _text(built.pieces) == "[USER] hi"
        assert built.stats["truncated"] is False
        assert built.stats["estimated_tokens"] == built.stats["sent_tokens"] == 9
        assert built.stats["budget_tokens"] == 1000

    async def test_over_budget_keeps_head_tail_and_whole_images(self):
        raw = _png((64, 64))
        parsed = te.normalize_retrieval_messages(
            [
                {"role": "user", "content": "HEAD " + "x" * 2000},
                {"role": "user", "content": [_image_part(raw)]},
                {"role": "user", "content": "y" * 2000 + " TAIL"},
            ]
        )
        await prepare_images(p for m in parsed for p in m.images)
        image_cost = image_tokens(64, 64) + IMAGE_LABEL_TOKENS
        budget = 400
        assert image_cost > budget // 2  # the image cannot fit in either half
        built = te.build_extraction_input(parsed, budget, COUNTER)
        text = _text(built.pieces)
        assert text.startswith("[USER] HEAD") and text.endswith(" TAIL")
        assert "<IMG>" not in text and built.images == {}
        stats = built.stats
        assert stats["truncated"] and stats["dropped_images"] == 1
        assert stats["images"] == 1 and stats["image_tokens"] == image_cost
        assert stats["sent_tokens"] <= budget < stats["estimated_tokens"]

    async def test_an_image_that_fits_is_kept_and_sent(self):
        raw = _png((16, 16))
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [_image_part(raw), {"type": "text", "text": "q" * 3000}],
                },
            ]
        )
        await prepare_images(p for m in parsed for p in m.images)
        # Room for the image in the head half, not for all of the text (the
        # estimate follows context_budget.image_tokens, whatever its factor).
        image_cost = image_tokens(16, 16) + IMAGE_LABEL_TOKENS
        budget = 2 * image_cost + 400
        assert image_cost < budget // 2 and image_cost + 3000 > budget
        built = te.build_extraction_input(parsed, budget, COUNTER)
        text = _text(built.pieces)
        assert text.startswith("[USER] <IMG>\nqqq") and text.endswith("qqq")
        assert built.stats["truncated"] and built.stats["dropped_images"] == 0
        assert len(built.images) == 1


class TestBudget:
    def test_budget_is_input_budget_minus_the_prompts_and_tool(self):
        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=100_000,
            max_output_tokens=8_000,
        )
        limits = ModelLimits.from_llm_config(config)
        fixed = te.fixed_prompt_tokens(COUNTER)
        assert fixed == (
            len(te.EXTRACTION_SYSTEM_PROMPT)
            + len(te.CONVERSATION_HEADER)
            + len("\n\n" + te.EXTRACTION_INSTRUCTION)
            + COUNTER.count_tools([te.EXTRACTION_TOOL])
            + 8
        )
        # floor(100000 * 0.8) - fixed - min(floor(8000 * 0.8), 100000 // 10)
        assert te.extraction_budget(config, COUNTER) == 80_000 - fixed - 6_400
        assert limits.input_budget(fixed) == te.extraction_budget(config, COUNTER)

    def test_the_reserve_never_drops_below_the_clients_minimum_output(self):
        """A small window's planning reserve (window / 10) can be below the
        1,024 output tokens a client insists on: the budget keeps 1,024."""
        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=8_000,
            max_output_tokens=4_000,
        )
        fixed = te.fixed_prompt_tokens(COUNTER)
        assert ModelLimits.from_llm_config(config).planning_output_reserve == 800
        assert te.extraction_budget(config, COUNTER) == 6_400 - fixed - 1_024


# --------------------------------------------------------------------------- #
# A generic prompt                                                             #
# --------------------------------------------------------------------------- #

# Vocabulary of the three benchmarks and their task domains: none of it
# may appear in MIRIX's extraction prompt.
BENCHMARK_WORDS = [
    "alfworld",
    "officeqa",
    "docvqa",
    "skillopt",
    "benchmark",
    "household",
    "receptacle",
    "countertop",
    "cabinet",
    "fridge",
    "microwave",
    "sinkbasin",
    "treasury",
    "bulletin",
    "fiscal",
    "anls",
    "invoice",
    "scanned",
    "webshop",
    "go to",
    "pick up",
]


class TestGenericPrompt:
    def _prompt_text(self):
        return "\n".join(
            [
                te.EXTRACTION_SYSTEM_PROMPT,
                te.CONVERSATION_HEADER,
                te.EXTRACTION_INSTRUCTION,
                te.LOCAL_MODEL_SYSTEM_PROMPT,
                te.LOCAL_MODEL_INSTRUCTION,
                json.dumps(te.EXTRACTION_TOOL),
            ]
        ).lower()

    @pytest.mark.parametrize("word", BENCHMARK_WORDS)
    def test_no_benchmark_vocabulary(self, word):
        assert re.search(rf"\b{re.escape(word)}\b", self._prompt_text()) is None

    def test_asks_for_the_four_things(self):
        instruction = te.EXTRACTION_INSTRUCTION.lower()
        for needle in (
            "user or the agent wants to accomplish",
            "key objects",
            "operations",
            "temporal expression",
        ):
            assert needle in instruction
        assert "images" in instruction  # The images are part of the input

    def test_the_rest_api_module_uses_this_prompt(self):
        """The old benchmark-agnostic-but-vague prompt ('focusing on') is gone."""
        path = os.path.join(os.path.dirname(te.__file__), "..", "server", "rest_api.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        assert (
            "extracts topics and temporal information from the user's input"
            not in source
        )
        assert "Brief description of what the user is focusing on" not in source


def _answer(arguments):
    """A chat completion whose one tool call carries ``arguments``."""
    from types import SimpleNamespace

    call = SimpleNamespace(
        function=SimpleNamespace(
            name=te.EXTRACTION_FUNCTION_NAME, arguments=json.dumps(arguments)
        )
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))]
    )


class TestStructuredAnswer:
    """The extraction separates the kind of task (goal, operations) from
    the instance (key objects). The topics keep all three, in today's order,
    for the keyword-searched types; the procedural query is the task-level
    part only, since a skill's retrieval text describes a kind of task."""

    def test_the_parts_make_the_topics_and_the_procedural_query(self):
        topics, temporal, procedural = te.parse_extraction_response(
            _answer(
                {
                    "goal": " g ",
                    "operations": "ops",
                    "key_objects": "o",
                    "temporal_expression": " yesterday ",
                }
            )
        )
        assert topics == "g; o; ops"
        assert procedural == "g; ops"
        assert temporal == "yesterday"

    def test_a_legacy_topic_answer_still_parses(self):
        assert te.parse_extraction_response(
            _answer({"topic": "a; b", "temporal_expression": ""})
        ) == ("a; b", None, None)

    def test_key_objects_alone_are_topics_but_no_procedural_query(self):
        topics, _temporal, procedural = te.parse_extraction_response(
            _answer(
                {
                    "goal": " ",
                    "operations": "",
                    "key_objects": "Q3 report",
                    "temporal_expression": "",
                }
            )
        )
        assert topics == "Q3 report" and procedural is None

    def test_lists_and_stray_separators_are_normalized(self):
        topics, _temporal, procedural = te.parse_extraction_response(
            _answer(
                {
                    "goal": "total a column;",
                    "operations": ["open the file", " sum it "],
                    "key_objects": "; column B ;",
                    "temporal_expression": "",
                }
            )
        )
        assert topics == "total a column; column B; open the file; sum it"
        assert procedural == "total a column; open the file; sum it"

    @pytest.mark.parametrize(
        "arguments",
        [
            {
                "goal": "",
                "operations": " ",
                "key_objects": "",
                "temporal_expression": "",
            },
            {"goal": ";", "operations": [], "key_objects": None},
            {"temporal_expression": "today"},
        ],
    )
    def test_nothing_usable_is_an_error(self, arguments):
        with pytest.raises(te.TopicExtractionError, match="no usable topic"):
            te.parse_extraction_response(_answer(arguments))

    def test_the_schema_requires_the_four_fields(self):
        parameters = te.EXTRACTION_TOOL["parameters"]
        assert te.EXTRACTION_FUNCTION_NAME == "update_topic_and_time"
        assert (
            set(parameters["properties"])
            == set(te.EXTRACTION_FIELDS)
            == {
                "goal",
                "operations",
                "key_objects",
                "temporal_expression",
            }
        )
        assert set(parameters["required"]) == set(te.EXTRACTION_FIELDS)
        assert all(p["type"] == "string" for p in parameters["properties"].values())

    def test_the_goal_is_the_kind_of_task_without_instance_values(self):
        for text in (
            te.EXTRACTION_TOOL["parameters"]["properties"]["goal"]["description"],
            te.EXTRACTION_INSTRUCTION,
        ):
            lowered = text.lower()
            assert "kind of task" in lowered
            for value in ("names", "numbers", "dates"):
                assert value in lowered

    def test_the_operations_are_kinds_of_action_without_instance_values(self):
        """``operations`` is half of the skill query (goal; operations): a
        value there ("locate Johnson & Johnson ticker") puts the instance back
        into the skill query vector it was taken out of."""
        for text in (
            te.EXTRACTION_TOOL["parameters"]["properties"]["operations"]["description"],
            te.EXTRACTION_INSTRUCTION.split("3. Operations:", 1)[1].split("4.", 1)[0],
        ):
            lowered = text.lower()
            assert "kinds of action" in lowered
            for value in ("names", "numbers", "dates"):
                assert value in lowered
            assert "key objects" in lowered

    def test_the_temporal_expression_is_when_the_recalled_things_happened(self):
        """The old wording ("any phrase about time ... or a date")
        invited the dates of the conversation's subject matter."""
        for text in (
            te.EXTRACTION_TOOL["parameters"]["properties"]["temporal_expression"][
                "description"
            ],
            te.EXTRACTION_INSTRUCTION,
        ):
            lowered = text.lower()
            assert "or a date" not in lowered
            assert "when the things to recall happened" in lowered
            assert "only mentions dates as part of its subject" in lowered


# --------------------------------------------------------------------------- #
# The extraction call (fake model server)                                      #
# --------------------------------------------------------------------------- #


def _config(fake, **overrides):
    values = dict(
        model="fake-model",
        model_endpoint_type="openai",
        model_endpoint=fake.base_url,
        context_window=200_000,
        max_output_tokens=16_000,
        api_key="fake-key",
    )
    values.update(overrides)
    return LLMConfig(**values)


class _NoFiles:
    """A FileManager stand-in that fails the test if a stored file is read."""

    async def get_file_metadata_by_id(self, file_id, *args, **kwargs):
        raise AssertionError(f"read stored file {file_id}")


def _user_content(request):
    [system, user] = request.json["messages"]
    assert system["role"] in ("system", "developer")
    assert system["content"] == te.EXTRACTION_SYSTEM_PROMPT
    assert user["role"] == "user"
    return user["content"]


def _joined(content):
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if part["type"] == "text" else "<IMG>" for part in content
    )


class TestExtractTopics:
    async def test_text_extraction(self, fake_llm_server):
        fake_llm_server.queue_tool_call(
            te.EXTRACTION_FUNCTION_NAME,
            {
                "topic": " open the report; sum column B ",
                "temporal_expression": "yesterday",
            },
            usage={"prompt_tokens": 321, "completion_tokens": 12, "total_tokens": 333},
        )
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": "Sum column B of yesterday's report"}]
        )
        outcome = await te.extract_topics(parsed, _config(fake_llm_server))
        assert outcome.status == "ok" and outcome.error is None
        assert outcome.topics == "open the report; sum column B"
        assert outcome.temporal_expression == "yesterday"
        assert outcome.usage == {
            "model": "fake-model",
            "calls": 1,
            "prompt_tokens": 321,
            "completion_tokens": 12,
            "total_tokens": 333,
        }
        request = fake_llm_server.last_chat_request()
        text = _joined(_user_content(request))
        assert text.startswith(te.CONVERSATION_HEADER + "[USER] Sum column B")
        assert text.endswith(te.EXTRACTION_INSTRUCTION)
        assert (
            request.json["tool_choice"]["function"]["name"]
            == te.EXTRACTION_FUNCTION_NAME
        )
        [tool] = request.json["tools"]
        assert tool["function"]["name"] == te.EXTRACTION_FUNCTION_NAME
        entry = outcome.log_entry()
        assert entry["status"] == "ok" and entry["input"]["truncated"] is False
        # A legacy one-string answer has no task-level query.
        assert outcome.procedural_query is None
        assert "procedural_query" not in entry

    async def test_a_structured_answer_gives_a_task_level_query(self, fake_llm_server):
        """The goal and the operations (the kind of task) are the
        procedural query; the key objects join them in the topics."""
        fake_llm_server.queue_tool_call(
            te.EXTRACTION_FUNCTION_NAME,
            {
                "goal": "total a spreadsheet column",
                "operations": "open the file; sum the column",
                "key_objects": "Q3 report; column B",
                "temporal_expression": "",
            },
        )
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": "Sum column B of the Q3 report"}]
        )
        outcome = await te.extract_topics(parsed, _config(fake_llm_server))
        assert outcome.status == "ok"
        assert outcome.topics == (
            "total a spreadsheet column; Q3 report; column B; open the file; "
            "sum the column"
        )
        assert outcome.procedural_query == (
            "total a spreadsheet column; open the file; sum the column"
        )
        assert outcome.temporal_expression is None
        assert outcome.log_entry()["procedural_query"] == outcome.procedural_query
        [tool] = fake_llm_server.last_chat_request().json["tools"]
        parameters = tool["function"]["parameters"]
        assert set(parameters["required"]) == set(te.EXTRACTION_FIELDS)

    async def test_images_go_with_the_text_and_are_never_stored(
        self, fake_llm_server, monkeypatch, tmp_path
    ):
        """The extraction model sees the image in place; it is read from a
        temporary file that is gone afterwards, and no ``files`` row is read
        or written."""
        created = []
        real_mkdtemp = te.tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            kwargs.setdefault("dir", str(tmp_path))
            created.append(real_mkdtemp(*args, **kwargs))
            return created[-1]

        monkeypatch.setattr(te.tempfile, "mkdtemp", spy_mkdtemp)
        fake_llm_server.queue_tool_call(
            te.EXTRACTION_FUNCTION_NAME, {"topic": "read the chart"}
        )
        raw = _png((30, 20), color="blue")
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What does this show?"},
                        _image_part(raw, detail="high"),
                        {"type": "text", "text": "Answer briefly."},
                    ],
                }
            ]
        )
        await prepare_images(p for m in parsed for p in m.images)
        outcome = await te.extract_topics(
            parsed, _config(fake_llm_server), file_manager=_NoFiles()
        )
        assert outcome.status == "ok" and outcome.topics == "read the chart"
        content = _user_content(fake_llm_server.last_chat_request())
        images = [p for p in content if p["type"] == "image_url"]
        assert len(images) == 1
        url = images[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == raw
        assert images[0]["image_url"]["detail"] == "high"
        # In place: the question before it, the instruction after it.
        joined = _joined(content)
        assert (
            joined.index("What does this show?")
            < joined.index("<IMG>")
            < joined.index("Answer briefly.")
        )
        assert outcome.input_stats["images"] == 1
        assert created and not any(os.path.exists(d) for d in created)

    async def test_a_stored_image_reference_is_read_from_files(
        self, fake_llm_server, tmp_path
    ):
        from mirix.services.topic_extraction import EphemeralFile

        raw = _png((8, 8))
        path = tmp_path / "stored.png"
        path.write_bytes(raw)
        seen = []

        class Files:
            async def get_file_metadata_by_id(self, file_id, *args, **kwargs):
                seen.append(file_id)
                return EphemeralFile(
                    id=file_id, file_name="stored.png", file_path=str(path)
                )

        fake_llm_server.queue_tool_call(te.EXTRACTION_FUNCTION_NAME, {"topic": "x"})
        parsed = te.normalize_retrieval_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "database_image_id", "image_id": "file-stored"},
                    ],
                }
            ]
        )
        outcome = await te.extract_topics(
            parsed, _config(fake_llm_server), file_manager=Files()
        )
        assert outcome.status == "ok" and seen == ["file-stored"]
        content = _user_content(fake_llm_server.last_chat_request())
        assert any(p["type"] == "image_url" for p in content)

    async def test_an_oversize_input_is_cut_to_the_budget(self, fake_llm_server):
        """The cut input goes through the client's own budget check (at least
        1,024 output tokens left): the budget and the client count alike."""
        from mirix.services.context_budget import get_token_counter

        counter = get_token_counter()
        fake_llm_server.queue_tool_call(te.EXTRACTION_FUNCTION_NAME, {"topic": "t"})
        config = _config(fake_llm_server, context_window=8_000, max_output_tokens=4_000)
        budget = (
            te.extraction_budget(config, counter) - te.EXTRACTION_BUDGET_MARGIN_TOKENS
        )
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": "START " + "filler word " * 6000 + "FINISH"}]
        )
        outcome = await te.extract_topics(parsed, config)
        assert outcome.status == "ok", outcome.error
        assert outcome.input_stats["budget_shrinks"] == 0
        stats = outcome.input_stats
        assert stats["truncated"] and stats["budget_tokens"] == budget
        assert stats["sent_tokens"] <= budget < stats["estimated_tokens"]
        text = _joined(_user_content(fake_llm_server.last_chat_request()))
        assert "START" in text and "FINISH" in text and "…[truncated " in text

    async def test_a_client_refusal_cuts_the_input_again(self):
        """ContextBudgetExceeded is raised before anything is sent: the input
        is cut by the shortfall and sent again; a client that keeps refusing
        ends in the fallback."""
        from types import SimpleNamespace

        from mirix.errors import ContextBudgetExceeded

        sent = []

        class Refusing:
            file_manager = None

            def __init__(self, refusals):
                self.refusals = refusals

            async def send_llm_request(self, messages, **kwargs):
                sent.append(
                    sum(len(getattr(c, "text", "")) for c in messages[1].content)
                )
                if self.refusals:
                    self.refusals -= 1
                    raise ContextBudgetExceeded(
                        estimated_input_tokens=1,
                        output_cap=1000,
                        minimum_output_tokens=1024,
                    )
                call = SimpleNamespace(
                    function=SimpleNamespace(
                        name=te.EXTRACTION_FUNCTION_NAME, arguments='{"topic": "t"}'
                    )
                )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))
                    ],
                    usage=None,
                )

        config = LLMConfig(
            model="m",
            model_endpoint_type="openai",
            context_window=8_000,
            max_output_tokens=4_000,
        )
        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": "x" * 20_000}]
        )
        outcome = await te.extract_topics(
            parsed, config, counter=COUNTER, client_factory=lambda c: Refusing(1)
        )
        assert outcome.status == "ok" and outcome.input_stats["budget_shrinks"] == 1
        first, second = sent
        assert first - second == 24 + te.EXTRACTION_BUDGET_MARGIN_TOKENS
        sent.clear()
        outcome = await te.extract_topics(
            parsed, config, counter=COUNTER, client_factory=lambda c: Refusing(99)
        )
        assert (
            outcome.status == "fallback" and "CONTEXT_WINDOW_EXCEEDED" in outcome.error
        )
        assert len(sent) == te.MAX_BUDGET_SHRINKS + 1

    @pytest.mark.parametrize(
        "answer",
        [
            {"content": "I think the topic is reports."},  # no tool call
            {"tool": {"topic": "   "}},  # empty topic
            {"tool": "not json"},  # unparseable arguments
        ],
        ids=["no-tool-call", "empty-topic", "bad-json"],
    )
    async def test_an_unusable_answer_falls_back(self, fake_llm_server, answer):
        if "content" in answer:
            fake_llm_server.queue_text(answer["content"])
        else:
            fake_llm_server.queue_tool_call(te.EXTRACTION_FUNCTION_NAME, answer["tool"])
        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "q"}])
        outcome = await te.extract_topics(parsed, _config(fake_llm_server))
        assert outcome.status == "fallback" and outcome.topics is None
        assert outcome.error and outcome.log_entry()["error"] == outcome.error

    async def test_a_4xx_error_falls_back(self, fake_llm_server):
        from fakes.fake_llm_server import Fault

        fake_llm_server.inject_fault(Fault.http_error(400, "bad request"))
        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "q"}])
        outcome = await te.extract_topics(parsed, _config(fake_llm_server))
        assert outcome.status == "fallback" and "400" in (outcome.error or "")

    async def test_a_window_too_small_falls_back_without_a_call(self, fake_llm_server):
        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "q"}])
        outcome = await te.extract_topics(
            parsed, _config(fake_llm_server, context_window=500, max_output_tokens=50)
        )
        assert outcome.status == "fallback" and "too small" in outcome.error
        assert fake_llm_server.chat_requests() == []

    async def test_no_config_or_no_client_falls_back(self):
        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "q"}])
        assert (await te.extract_topics(parsed, None)).status == "fallback"
        config = LLMConfig(
            model="m", model_endpoint_type="openai", context_window=100_000
        )
        outcome = await te.extract_topics(parsed, config, client_factory=lambda c: None)
        assert outcome.status == "fallback" and "no LLM client" in outcome.error

    async def test_cancellation_propagates(self):
        import asyncio

        class Hanging:
            file_manager = None

            async def send_llm_request(self, **kwargs):
                await asyncio.sleep(3600)

        parsed = te.normalize_retrieval_messages([{"role": "user", "content": "q"}])
        config = LLMConfig(
            model="m", model_endpoint_type="openai", context_window=100_000
        )
        task = asyncio.ensure_future(
            te.extract_topics(parsed, config, client_factory=lambda c: Hanging())
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_temp_files_are_removed_when_the_call_fails(
        self, monkeypatch, tmp_path
    ):
        created = []
        real_mkdtemp = te.tempfile.mkdtemp

        def spy_mkdtemp(*args, **kwargs):
            kwargs.setdefault("dir", str(tmp_path))
            created.append(real_mkdtemp(*args, **kwargs))
            return created[-1]

        monkeypatch.setattr(te.tempfile, "mkdtemp", spy_mkdtemp)

        class Failing:
            file_manager = None

            async def send_llm_request(self, **kwargs):
                meta = await self.file_manager.get_file_metadata_by_id(
                    next(iter(self.file_manager._metadata))
                )
                assert os.path.exists(meta.file_path)
                raise RuntimeError("provider down")

        parsed = te.normalize_retrieval_messages(
            [{"role": "user", "content": [_image_part(_png())]}]
        )
        await prepare_images(p for m in parsed for p in m.images)
        config = LLMConfig(
            model="m", model_endpoint_type="openai", context_window=100_000
        )
        outcome = await te.extract_topics(
            parsed, config, client_factory=lambda c: Failing()
        )
        assert outcome.status == "fallback" and "provider down" in outcome.error
        assert created and not any(os.path.exists(d) for d in created)
