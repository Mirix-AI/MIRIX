"""The distiller's input: rendering and chunking.

Pure CPU and DB-free. Token counts use a deterministic stand-in encoding (one
token per 4 characters) so every boundary is exact; the distiller-level tests
force chunking with a small-window ``LLMConfig`` and check every chunk against
the budget the distiller derived from it.
"""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

from mirix.errors import ContextBudgetExceeded
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.mirix_message_content import ImageContent, TextContent
from mirix.services.context_budget import (
    IMAGE_TOKEN_MULTIPLIER,
    MESSAGE_OVERHEAD_TOKENS,
    UNKNOWN_IMAGE_TOKENS,
    ModelLimits,
    TokenCounter,
    image_formula_tokens,
    image_tokens,
)
from mirix.services.distill_transcript import (
    IMAGE_LABEL_TOKENS,
    ImagePiece,
    TextPiece,
    TranscriptPlanner,
    build_messages,
    head_tail_text,
    row_image_ids,
    split_rounds,
    truncate_message,
)
from mirix.services.session_experience_distiller import SessionExperienceDistiller


class QuarterEncoding:
    """One token per 4 characters (rounded up): exact, deterministic counts."""

    def encode(self, text, disallowed_special=()):
        return [0] * ((len(text) + 3) // 4)


COUNTER = TokenCounter(encoding=QuarterEncoding())


def row(role, content="", parts=None):
    return SimpleNamespace(role=role, content=content, content_parts=parts)


def text_part(text):
    return {"type": "text", "text": text}


def image_part(file_id, width=512, height=512, **extra):
    return {
        "type": "image_ref",
        "file_id": file_id,
        "width": width,
        "height": height,
        **extra,
    }


def planner_for(rows, **kwargs):
    messages, unavailable = build_messages(rows, counter=COUNTER, **kwargs)
    return TranscriptPlanner(
        messages,
        agent_id="agent-1",
        session_id="sess-1",
        counter=COUNTER,
        images_unavailable=unavailable,
    )


def alfworld_rows(rounds, obs_chars=400, act_chars=120):
    rows = [row("user", "Your task is to: put a clean mug in the coffee machine.")]
    rows.append(row("assistant", "think: find a mug " + "a" * act_chars))
    for i in range(1, rounds):
        rows.append(row("user", f"obs {i:03d} " + "o" * obs_chars))
        rows.append(row("assistant", f"act {i:03d} " + "a" * act_chars))
    return rows


def body_positions(planner, chunk):
    return chunk.new_positions


def chunk_content_tokens(chunk, image_costs=None):
    """Recount a chunk's user message independently of the planner: all its
    text at once, plus each image's cost (its tokens and the client's label)."""
    text = "".join(p.text for p in chunk.content if isinstance(p, TextContent))
    images = sum(
        (image_costs or {}).get(p.image_id, UNKNOWN_IMAGE_TOKENS + IMAGE_LABEL_TOKENS)
        for p in chunk.content
        if isinstance(p, ImageContent)
    )
    return COUNTER.count_text(text) + images


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #


class TestRendering:
    def test_images_are_placed_at_their_original_positions(self):
        rows = [
            row("user", "plain question"),
            row(
                "user",
                "placeholder text that must not be used",
                parts=[
                    text_part("look at this chart:"),
                    image_part("file-a"),
                    text_part("and this one:"),
                    image_part("file-b", width=100, height=50),
                    text_part("which is larger?"),
                ],
            ),
            row("assistant", "the first"),
            row(
                "tool", "", parts=[image_part("file-c"), text_part("[screenshot] done")]
            ),
        ]
        chunk = planner_for(rows).whole_chunk()
        kinds = [
            ("image", p.image_id) if isinstance(p, ImageContent) else ("text", p.text)
            for p in chunk.content
        ]
        assert [k for k in kinds if k[0] == "image"] == [
            ("image", "file-a"),
            ("image", "file-b"),
            ("image", "file-c"),
        ]
        # Text around each image is exactly what surrounded it in its message.
        text_before_a = kinds[0][1]
        assert text_before_a.endswith(
            "user: plain question\nuser: look at this chart:\n"
        )
        assert kinds[2] == ("text", "\nand this one:\n")
        assert kinds[4][1].startswith(
            "\nwhich is larger?\nassistant: the first\ntool: "
        )
        assert kinds[6][1].startswith("\n[screenshot] done")
        # content_parts rows never fall back to their placeholder `content`.
        assert "placeholder text" not in chunk.text()

    def test_one_multimodal_user_message_with_header_and_role_lines(self):
        chunk = planner_for(
            [
                row("user", "hi", parts=[text_part("hi"), image_part("file-x")]),
                row("assistant", "hello"),
            ]
        ).whole_chunk()
        text = chunk.text()
        assert text.startswith(
            "agent_id: agent-1\nsession_id: sess-1\nsession: 2 message(s)"
        )
        assert "1 image(s)" in text.split("\n")[2]
        assert (
            "session_transcript (chronological):\nuser: hi\n[image: file-x]\nassistant: hello"
            in text
        )
        assert text.endswith(
            "Distill the transferable experiences from this session as one JSON "
            "object: outcome_evidence, outcome, then experiences."
        )
        assert chunk.whole and chunk.images == 1

    def test_image_detail_defaults_to_auto_and_is_kept_when_given(self):
        chunk = planner_for(
            [row("user", "", parts=[image_part("f1"), image_part("f2", detail="low")])]
        ).whole_chunk()
        images = [p for p in chunk.content if isinstance(p, ImageContent)]
        assert [(i.image_id, i.detail) for i in images] == [
            ("f1", "auto"),
            ("f2", "low"),
        ]

    def test_an_unavailable_image_becomes_a_counted_placeholder(self):
        rows = [
            row(
                "user",
                "",
                parts=[text_part("see"), image_part("gone"), image_part("ok")],
            )
        ]
        planner = planner_for(rows, available_images={"ok"})
        chunk = planner.whole_chunk()
        assert "[image unavailable: gone]" in chunk.text()
        assert [p.image_id for p in chunk.content if isinstance(p, ImageContent)] == [
            "ok"
        ]
        assert planner.stats.images == 1 and planner.stats.images_unavailable == 1
        assert "(1 unavailable)" in planner.metadata_line()

    def test_row_image_ids_are_distinct_and_ordered(self):
        rows = [
            row("user", "", parts=[image_part("b"), image_part("a")]),
            row(
                "user",
                "",
                parts=[
                    image_part("b"),
                    {"type": "image_ref", "url": "https://x/y.png"},
                ],
            ),
        ]
        assert row_image_ids(rows) == ["b", "a"]

    def test_image_tokens_use_the_high_detail_formula(self):
        messages, _ = build_messages(
            [
                row("user", "", parts=[image_part("f", width=1024, height=1024)]),
                row("user", "", parts=[{"type": "image_ref", "file_id": "g"}]),
            ],
            counter=COUNTER,
        )
        [first] = [p for p in messages[0].pieces if isinstance(p, ImagePiece)]
        [second] = [p for p in messages[1].pieces if isinstance(p, ImagePiece)]
        # The high-detail formula (765 for 1024x1024) times the conservative
        # image factor.
        assert image_formula_tokens(1024, 1024) == 765
        expected = math.ceil(765 * IMAGE_TOKEN_MULTIPLIER)
        assert first.tokens == image_tokens(1024, 1024) == expected
        assert second.tokens == UNKNOWN_IMAGE_TOKENS
        assert first.cost == expected + IMAGE_LABEL_TOKENS

    def test_a_row_without_text_or_image_is_skipped(self):
        messages, _ = build_messages(
            [
                row("user", "   "),
                row("tool", "", parts=[text_part("  ")]),
                row("assistant", "x"),
            ],
            counter=COUNTER,
        )
        assert [m.role for m in messages] == ["assistant"]
        assert messages[0].position == 1


# --------------------------------------------------------------------------- #
# Rounds                                                                       #
# --------------------------------------------------------------------------- #


class TestRounds:
    def _rounds(self, roles):
        messages, _ = build_messages(
            [row(r, f"{r} {i}") for i, r in enumerate(roles)], counter=COUNTER
        )
        return [[m.role for m in r.messages] for r in split_rounds(messages)]

    def test_alfworld_round_is_observation_plus_action(self):
        assert self._rounds(["user", "assistant", "user", "assistant"]) == [
            ["user", "assistant"],
            ["user", "assistant"],
        ]

    def test_officeqa_round_is_tool_result_plus_next_call(self):
        roles = ["user", "assistant", "tool", "assistant", "tool", "tool", "assistant"]
        assert self._rounds(roles) == [
            ["user", "assistant"],
            ["tool", "assistant"],
            ["tool", "tool", "assistant"],
        ]

    def test_trailing_messages_form_the_last_round(self):
        assert self._rounds(["user", "assistant", "user", "tool"]) == [
            ["user", "assistant"],
            ["user", "tool"],
        ]

    def test_a_leading_assistant_message_is_a_round_of_its_own(self):
        assert self._rounds(["assistant", "user", "assistant"]) == [
            ["assistant"],
            ["user", "assistant"],
        ]


# --------------------------------------------------------------------------- #
# Tool-result status in the header                                             #
# --------------------------------------------------------------------------- #


class TestToolResultStatus:
    """The header's tool-error count used to search every word of every tool
    result for an error word, so a document read that mentions "error" counted
    and a search that found nothing did not. Only a result's leading status
    (its first line after the ``[name]`` prefix) counts now, and a result that
    found nothing is counted on its own."""

    def _stats(self, *tool_texts):
        rows = [row("user", "find the value")]
        for text in tool_texts:
            rows += [row("assistant", "[tool_call] grep(x)"), row("tool", text)]
        return planner_for(rows).stats

    def test_an_empty_result_is_counted_apart_from_errors(self):
        stats = self._stats(
            "[grep] [no matches]", "[read] [empty file]", "[bash] [no output]"
        )
        assert stats.empty_results == 3 and stats.tool_errors == 0

    def test_prose_below_the_first_line_is_not_an_error(self):
        document = "\n".join(
            ["[read] 1 | Annual report"]
            + [f"{i} | line {i}" for i in range(2, 40)]
            + ["40 | the error bars show the spread"]
        )
        stats = self._stats(document)
        assert stats.tool_errors == 0 and stats.empty_results == 0

    def test_a_leading_error_counts(self):
        stats = self._stats(
            "[bash] Traceback (most recent call last):\n  File x",
            "Error: permission denied",
        )
        assert stats.tool_errors == 2

    def test_the_header_names_both_counts(self):
        planner = planner_for(
            [
                row("user", "find it"),
                row("assistant", "[tool_call] grep(x)"),
                row("tool", "[grep] no matches found"),
                row("assistant", "[tool_call] read(y)"),
                row("tool", "[read] Error: no such file"),
            ]
        )
        assert "tool errors: 1; empty results: 1" in planner.metadata_line()

    def test_unnamed_results_take_the_names_of_the_calls_they_answer(self):
        messages, _ = build_messages(
            [
                row("assistant", "[tool_call] grep(a)\n[tool_call] read(b)"),
                row("tool", "no matches"),
                row("tool", "file text"),
                row("assistant", "[tool_call] grep(c)\n[tool_call] read(d)"),
                row("tool", "only one result"),
                row("assistant", "done"),
            ],
            counter=COUNTER,
        )
        texts = [m.plain_text() for m in messages]
        assert texts[1] == "tool: [grep] no matches"
        assert texts[2] == "tool: [read] file text"
        # Two calls, one result: which call it answers is unknown.
        assert texts[4] == "tool: only one result"
        assert messages[1].tokens == COUNTER.count_text(texts[1]) + 1
        stats = TranscriptPlanner(
            messages, agent_id="a", session_id="s", counter=COUNTER
        ).stats
        assert stats.tool_names == ["grep", "read"] and stats.empty_results == 1


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #


def check_every_boundary(planner, plan):
    """The chunking invariants, checked for every chunk of ``plan``."""
    budget = plan.budget
    frame_cap = budget // 10  # The frame is at most 10% of the budget
    all_positions = [m.position for m in planner.messages]
    round_of = {m.position: r.number for r in planner.rounds for m in r.messages}
    image_costs = {
        p.file_id: p.cost
        for m in planner.messages
        for p in m.pieces
        if isinstance(p, ImagePiece)
    }
    seen_new = []
    for i, chunk in enumerate(plan.chunks, start=1):
        assert chunk.index == i and chunk.total == len(plan.chunks)
        # Within the budget, the frame within 10% of it.
        assert chunk.tokens <= budget
        assert chunk.frame_tokens <= frame_cap
        # Independently recounted: the message's text + every image's cost.
        assert chunk_content_tokens(chunk, image_costs) <= budget
        # "chunk i/n covering rounds a-b" in the frame.
        text = chunk.text()
        assert (
            f"chunk {i}/{len(plan.chunks)} covering rounds {chunk.first_round}-{chunk.last_round}"
            in text
        )
        # New material starts on a round boundary unless its round was split.
        first_new = chunk.new_positions[0]
        start_of_round = min(p for p, r in round_of.items() if r == round_of[first_new])
        if first_new != start_of_round:
            assert (
                planner.rounds[round_of[first_new] - 1].tokens
                > budget - chunk.frame_tokens
            )
        seen_new.extend(chunk.new_positions)
        # Overlap: the (up to) 2 rounds right before the new material.
        if i > 1:
            overlap_positions = chunk.positions[
                : len(chunk.positions) - len(chunk.new_positions)
            ]
            if overlap_positions:
                assert overlap_positions[-1] == first_new - 1
                overlap_rounds = {round_of[p] for p in overlap_positions}
                assert len(overlap_rounds) <= 2
                # Whole rounds: only the oldest round of the overlap may
                # be cut (when it did not fit); every newer one is there from
                # its first message (up to the new material).
                for number in overlap_rounds - {min(overlap_rounds)}:
                    shown = [
                        p for p, r in round_of.items() if r == number and p < first_new
                    ]
                    assert set(shown) <= set(overlap_positions), (i, number)
    # Every message is new in exactly one chunk, in order.
    assert seen_new == all_positions


class TestChunking:
    def test_a_session_that_fits_is_sent_whole(self):
        planner = planner_for(alfworld_rows(5))
        plan = planner.plan(planner.whole_tokens())
        assert len(plan.chunks) == 1 and plan.chunks[0].whole and not plan.chunked
        assert planner.plan(None).chunks[0].whole

    def test_a_long_session_is_cut_on_round_boundaries(self):
        planner = planner_for(alfworld_rows(60))
        budget = planner.whole_tokens() // 4
        plan = planner.plan(budget)
        assert len(plan.chunks) >= 4 and plan.chunked
        check_every_boundary(planner, plan)
        # No round is split (every round fits), so every chunk starts a round.
        for chunk in plan.chunks:
            first = chunk.new_positions[0]
            assert any(r.messages[0].position == first for r in planner.rounds)
            last = chunk.new_positions[-1]
            assert any(r.messages[-1].position == last for r in planner.rounds)

    def test_consecutive_chunks_overlap_by_exactly_two_rounds(self):
        planner = planner_for(alfworld_rows(60))
        plan = planner.plan(planner.whole_tokens() // 4)
        for chunk in plan.chunks[1:]:
            assert chunk.overlap_rounds == [
                chunk.new_first_round - 2,
                chunk.new_first_round - 1,
            ]
            assert chunk.first_round == chunk.new_first_round - 2
            text = chunk.text()
            assert (
                f"(rounds {chunk.first_round}-{chunk.new_first_round - 1} repeat the end of the previous chunk)"
                in text
            )
            assert (
                f"--- new in this chunk: rounds {chunk.new_first_round}-{chunk.last_round} ---"
                in text
            )
        assert plan.chunks[0].overlap_rounds == []

    def test_the_frame_carries_the_task_statement_and_the_final_rounds(self):
        planner = planner_for(alfworld_rows(200))
        plan = planner.plan(planner.whole_tokens() // 4)
        first, middle, last = plan.chunks[0], plan.chunks[1], plan.chunks[-1]
        # Round 1 is in the first chunk's transcript: no separate task section.
        assert "task statement (round 1):" not in first.text()
        assert "final rounds (rounds 199-200):" in first.text()
        assert (
            "task statement (round 1):\nuser: Your task is to: put a clean mug in the "
            "coffee machine.\nassistant: think: find a mug"
        ) in middle.text()
        assert "final rounds (rounds 199-200):\nuser: obs 198" in middle.text()
        assert "\nassistant: act 199 " in middle.text().split("chunk_transcript", 1)[0]
        # The last chunk shows the final rounds itself.
        assert "final rounds (rounds" not in last.text()
        for chunk in plan.chunks:
            assert (
                "session: 400 message(s) (assistant 200, user 200), 200 round(s)"
                in chunk.text()
            )
        check_every_boundary(planner, plan)

    def test_small_frames_still_keep_what_fits_of_the_sections(self):
        planner = planner_for(alfworld_rows(60))
        plan = planner.plan(planner.whole_tokens() // 4)
        check_every_boundary(planner, plan)
        middle = plan.chunks[1].text()
        # A tight frame keeps the head and tail of the task statement.
        assert "task statement (round 1):\nuser: Your task" in middle
        assert "…[truncated " in middle.split("chunk_transcript", 1)[0]

    def test_a_frame_that_does_not_fit_is_cut_head_and_tail(self):
        rows = alfworld_rows(40)
        rows[0] = row("user", "Your task is to: " + "t" * 6000 + " END-OF-TASK")
        planner = planner_for(rows)
        plan = planner.plan(3000)
        middle = plan.chunks[1]
        assert middle.frame_tokens <= 3000 // 10
        task = middle.text().split("task statement (round 1):\n", 1)[1]
        assert task.startswith("user: Your task is to: ttt")
        assert "…[truncated " in task.split("\n\n", 1)[0]
        check_every_boundary(planner, plan)

    def test_images_are_never_split_and_each_is_new_in_one_chunk(self):
        rows = alfworld_rows(30)
        for i in range(3, 60, 7):
            rows[i] = row(
                rows[i].role,
                "",
                parts=[
                    text_part(f"frame {i} before"),
                    image_part(f"img-{i}", 256, 256),
                    text_part("after"),
                ],
            )
        planner = planner_for(rows)
        plan = planner.plan(planner.whole_tokens() // 3)
        check_every_boundary(planner, plan)
        new_images = []
        for chunk in plan.chunks:
            ids = [p.image_id for p in chunk.content if isinstance(p, ImageContent)]
            new = set()
            for m in planner.messages:
                if m.position in chunk.new_positions:
                    new.update(p.file_id for p in m.pieces if isinstance(p, ImagePiece))
            assert new <= set(ids)
            new_images.extend(sorted(new))
            # An image sits between its own message's text parts.
            for image_id in ids:
                index = next(
                    k
                    for k, p in enumerate(chunk.content)
                    if isinstance(p, ImageContent) and p.image_id == image_id
                )
                n = image_id.split("-")[1]
                assert chunk.content[index - 1].text.endswith(f"frame {n} before\n")
                assert chunk.content[index + 1].text.startswith("\nafter")
        assert sorted(new_images) == sorted(f"img-{i}" for i in range(3, 60, 7))

    def test_a_round_too_large_for_a_chunk_is_cut_between_its_messages(self):
        rows = alfworld_rows(10)
        # Round 5: three big tool results before the assistant message.
        big = [row("tool", f"[bash] part {k} " + "b" * 4000) for k in range(3)]
        rows = rows[:9] + big + rows[9:]
        planner = planner_for(rows)
        plan = planner.plan(3000)
        split = [
            c
            for c in plan.chunks
            if any(
                "[bash] part" in m.plain_text()
                for m in planner.messages
                if m.position in c.new_positions
            )
        ]
        assert len(split) >= 2  # round 5 spans chunks, cut between its messages
        check_every_boundary(planner, plan)
        assert plan.truncated_messages == 0  # every message fits on its own

    def test_the_overlap_is_two_whole_rounds_when_a_round_is_split(self):
        # A round too large for a chunk is split into one segment per
        # message. The next chunk's overlap is still "the 2 rounds before" —
        # the part of the split round already shown and the whole round before
        # it — not the 2 last messages of the split round.
        rows = alfworld_rows(8)
        tools = [row("tool", f"[tool] result {k} " + "t" * 120) for k in range(40)]
        rows = rows[:9] + tools + rows[9:]
        planner = planner_for(rows)
        plan = planner.plan(800)
        check_every_boundary(planner, plan)
        second = plan.chunks[1]
        first_new = second.new_positions[0]
        round_4, round_5 = planner.rounds[3], planner.rounds[4]
        assert first_new in [m.position for m in round_5.messages]
        assert first_new != round_5.messages[0].position  # it continues round 5
        overlap = second.positions[: len(second.positions) - len(second.new_positions)]
        shown_of_5 = [m.position for m in round_5.messages if m.position < first_new]
        assert len(shown_of_5) > 2
        assert overlap == [m.position for m in round_4.messages] + shown_of_5
        assert second.overlap_rounds == [4, 5]

    def test_an_oversize_message_keeps_its_head_and_tail_and_is_counted(self):
        rows = alfworld_rows(6)
        rows.insert(5, row("tool", "[bash] HEAD-MARK " + "x" * 30_000 + " TAIL-MARK"))
        planner = planner_for(rows)
        plan = planner.plan(2000)
        check_every_boundary(planner, plan)
        assert plan.truncated_messages == 1
        assert plan.truncated_tokens > 5000 and plan.truncated_chars > 20_000
        [cut] = [c for c in plan.chunks if "HEAD-MARK" in c.text()]
        text = cut.text()
        assert "TAIL-MARK" in text
        assert "…[truncated " in text and "chars (~" in text
        assert cut.tokens <= 2000

    def test_an_oversize_message_keeps_images_whole_or_drops_them_counted(self):
        parts = [text_part("HEAD " + "h" * 3000)]
        parts += [image_part(f"big-{k}", 2048, 2048) for k in range(4)]
        parts += [text_part("t" * 3000 + " TAIL")]
        messages, _ = build_messages([row("user", "", parts=parts)], counter=COUNTER)
        cut = truncate_message(messages[0], 1600, COUNTER)
        assert cut.tokens <= 1600
        kept = [p.file_id for p in cut.pieces if isinstance(p, ImagePiece)]
        assert len(kept) < 4 and cut.dropped_images == 4 - len(kept)
        text = "".join(p.text for p in cut.pieces if isinstance(p, TextPiece))
        assert text.startswith("user: HEAD ") and text.endswith(" TAIL")
        assert f"{cut.dropped_images} image(s)]" in text

    def test_the_overlap_shrinks_when_two_rounds_take_more_than_half_the_room(self):
        rows = [row("user", "task"), row("assistant", "a0")]
        for i in range(1, 8):
            rows.append(row("user", f"obs {i} " + "o" * 3200))
            rows.append(row("assistant", f"act {i}"))
        planner = planner_for(rows)
        plan = planner.plan(3000)
        check_every_boundary(planner, plan)
        assert any(len(c.overlap_rounds) < 2 for c in plan.chunks[1:])
        for chunk in plan.chunks[1:]:
            overlap_tokens = sum(
                r.tokens for r in planner.rounds if r.number in chunk.overlap_rounds
            )
            assert overlap_tokens <= 3000 // 2

    def test_a_budget_too_small_for_the_frame_is_refused(self):
        planner = planner_for(alfworld_rows(20))
        with pytest.raises(ContextBudgetExceeded):
            planner.plan(60)
        with pytest.raises(ContextBudgetExceeded):
            planner.plan(300)  # 30 tokens cannot hold even the compact frame

    def test_a_tight_budget_gets_the_compact_frame(self):
        planner = planner_for(alfworld_rows(20))
        plan = planner.plan(1200)  # 120 tokens: too small for the full frame
        assert plan.compact_frame
        check_every_boundary(planner, plan)
        text = plan.chunks[1].text()
        assert text.startswith("agent_id: agent-1\nsession_id: sess-1\nchunk 2/")
        assert "session: " not in text and "task statement" not in text
        big = planner_for(alfworld_rows(200))
        assert not big.plan(big.whole_tokens() // 2).compact_frame

    def test_head_tail_text(self):
        text = "HEAD" + "m" * 4000 + "TAIL"
        cut, chars, tokens = head_tail_text(text, 100, COUNTER)
        assert COUNTER.count_text(cut) <= 100
        assert cut.startswith("HEADmm") and cut.endswith("mmTAIL")
        assert chars > 3000 and tokens > 750
        assert head_tail_text("short", 100, COUNTER) == ("short", 0, 0)


# --------------------------------------------------------------------------- #
# Through the distiller: a small-window LLMConfig forces chunking              #
# --------------------------------------------------------------------------- #


class _Conv:
    def __init__(self, turns_by_session):
        self.turns = turns_by_session

    async def list_turns_for_session(self, *, session_id, **kwargs):
        return list(self.turns.get(session_id, []))


class _RecordingLLM:
    """Answers every call with one experience naming the chunk it saw."""

    def __init__(self, delay=0.0, reply=None):
        self.requests = []
        self.live = 0
        self.peak = 0
        self.delay = delay
        self.reply = reply

    async def send_llm_request(self, *, messages):
        self.requests.append(messages)
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            text = "".join(
                p.text for p in messages[1].content if isinstance(p, TextContent)
            )
            label = "whole"
            if "covering rounds" in text:
                label = text.split("\nchunk ", 1)[1].split(" covering", 1)[0]
            reply = self.reply or json.dumps(
                [{"experience_type": "worth_learning", "title": f"lesson from {label}"}]
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=reply), finish_reason="stop"
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100, completion_tokens=10, total_tokens=110
                ),
            )
        finally:
            self.live -= 1


class _Store:
    def __init__(self):
        self.created = []

    async def create_experience(self, **kwargs):
        kwargs.pop("created_by_id", None)
        record = SimpleNamespace(**kwargs)
        self.created.append(record)
        return record


SMALL_WINDOW = LLMConfig(
    model="fake-model",
    model_endpoint_type="openai",
    model_endpoint="http://127.0.0.1:9",
    context_window=12_000,
    max_output_tokens=2_000,
)


def _meta():
    return (
        SimpleNamespace(id="agent-1"),
        SimpleNamespace(id="user-1"),
        SimpleNamespace(id="client-1", organization_id="org-1"),
    )


def _distiller(llm, conv, **kwargs):
    return SessionExperienceDistiller(
        llm_client=llm,
        llm_config=SMALL_WINDOW,
        conversation_manager=conv,
        experience_manager=_Store(),
        token_counter=COUNTER,
        **kwargs,
    )


@pytest.mark.asyncio
class TestDistillerChunking:
    async def test_a_small_window_forces_chunks_that_all_fit_its_budget(self):
        llm = _RecordingLLM()
        conv = _Conv({"long": alfworld_rows(120), "short": alfworld_rows(3)})
        d = _distiller(llm, conv)
        limits = ModelLimits.from_llm_config(SMALL_WINDOW)
        system_tokens = (
            COUNTER.count_text(d._system_prompt) + 2 * MESSAGE_OVERHEAD_TOKENS
        )
        budget = limits.input_budget(system_tokens)
        assert budget == 9_600 - system_tokens - 1_200

        meta, user, actor = _meta()
        metrics = {}
        experiences, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["long", "short"],
            metrics=metrics,
        )
        assert processed == ["long", "short"]
        long_calls = [
            m for m in llm.requests if "session_id: long" in m[1].content[0].text
        ]
        n = metrics["long"]["chunks"]
        assert n >= 2 and len(long_calls) == n
        assert metrics["long"]["chunk_budget_tokens"] == budget
        assert (
            metrics["short"]["chunks"] == 1
            and metrics["short"]["chunk_budget_tokens"] is None
        )
        for call in long_calls:
            user_text = "".join(
                p.text for p in call[1].content if isinstance(p, TextContent)
            )
            assert COUNTER.count_text(user_text) <= budget
            assert (
                call[0].role == "system" and call[1].role == "user" and len(call) == 2
            )
        # Re-plan with the same budget and check every boundary.
        messages, _ = build_messages(alfworld_rows(120), counter=COUNTER)
        planner = TranscriptPlanner(
            messages, agent_id="agent-1", session_id="long", counter=COUNTER
        )
        check_every_boundary(planner, planner.plan(budget))
        # Every chunk's experiences are kept (no dedupe).
        titles = sorted(e.title for e in experiences if e.session_id == "long")
        assert titles == sorted(f"lesson from {i}/{n}" for i in range(1, n + 1))

    async def test_identical_experiences_from_two_chunks_are_both_kept(self):
        same = json.dumps(
            [{"experience_type": "worth_avoiding", "title": "same lesson"}]
        )
        llm = _RecordingLLM(reply=same)
        d = _distiller(llm, _Conv({"long": alfworld_rows(120)}))
        meta, user, actor = _meta()
        metrics = {}
        experiences, _ = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["long"],
            metrics=metrics,
        )
        assert len(experiences) == metrics["long"]["chunks"] >= 2
        assert {e.title for e in experiences} == {"same lesson"}

    async def test_concurrency_is_bounded_across_chunks_and_sessions(self):
        llm = _RecordingLLM(delay=0.02)
        sessions = {"long": alfworld_rows(160)}
        sessions.update({f"s{i}": alfworld_rows(3) for i in range(5)})
        d = _distiller(llm, _Conv(sessions), max_concurrency=3)
        meta, user, actor = _meta()
        metrics = {}
        _, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=list(sessions),
            metrics=metrics,
        )
        assert len(processed) == 6 and metrics["long"]["chunks"] >= 3
        assert llm.peak == 3

    async def test_the_bound_is_process_wide_across_concurrent_runs(self):
        # The total concurrency: over every learning run of the
        # process (runs of different users are not serialized), not per run.
        llm = _RecordingLLM(delay=0.02)
        runs = []
        for owner in ("a", "b", "c"):
            sessions = {f"{owner}{i}": alfworld_rows(3) for i in range(4)}
            runs.append((_distiller(llm, _Conv(sessions), max_concurrency=3), sessions))
        meta, user, actor = _meta()

        async def one(distiller, sessions):
            _, processed = await distiller.distill_sessions(
                meta_agent_state=meta,
                user=user,
                actor=actor,
                session_ids=list(sessions),
            )
            return processed

        results = await asyncio.gather(*(one(d, s) for d, s in runs))
        assert [len(p) for p in results] == [4, 4, 4]
        assert llm.peak == 3  # not 3 per run (9)

    async def test_one_long_session_uses_the_whole_concurrency_for_its_chunks(self):
        llm = _RecordingLLM(delay=0.02)
        d = _distiller(llm, _Conv({"long": alfworld_rows(200)}), max_concurrency=4)
        meta, user, actor = _meta()
        metrics = {}
        await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["long"],
            metrics=metrics,
        )
        assert metrics["long"]["chunks"] >= 4 and llm.peak == 4

    async def test_images_reach_the_model_in_place_in_one_user_message(self):
        rows = [
            row(
                "user",
                "",
                parts=[
                    text_part("Q: what does the sign say?"),
                    image_part("file-sign"),
                    text_part("(photo)"),
                ],
            ),
            row("assistant", "It says STOP."),
            row("user", "correct, thanks"),
        ]
        llm = _RecordingLLM()
        d = _distiller(llm, _Conv({"img": rows}), image_resolver=_all_available)
        meta, user, actor = _meta()
        metrics = {}
        await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["img"],
            metrics=metrics,
        )
        [call] = llm.requests
        content = call[1].content
        assert [type(p).__name__ for p in content] == [
            "TextContent",
            "ImageContent",
            "TextContent",
        ]
        assert content[0].text.endswith("user: Q: what does the sign say?\n")
        assert content[1].image_id == "file-sign"
        assert content[2].text.startswith(
            "\n(photo)\nassistant: It says STOP.\nuser: correct, thanks"
        )
        assert (
            metrics["img"]["images"] == 1 and metrics["img"]["images_unavailable"] == 0
        )

    async def test_a_missing_image_file_does_not_fail_the_session(self):
        rows = [
            row("user", "", parts=[text_part("see"), image_part("file-gone")]),
            row("assistant", "ok"),
        ]
        llm = _RecordingLLM()

        async def none_available(ids):
            return set()

        d = _distiller(llm, _Conv({"s": rows}), image_resolver=none_available)
        meta, user, actor = _meta()
        metrics = {}
        _, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["s"],
            metrics=metrics,
        )
        assert processed == ["s"]
        [call] = llm.requests
        assert not any(isinstance(p, ImageContent) for p in call[1].content)
        assert "[image unavailable: file-gone]" in call[1].content[0].text
        assert metrics["s"]["images_unavailable"] == 1

    async def test_the_metrics_count_tool_errors_and_empty_results(self):
        """The session's distill metrics carry both header counts, and the
        header the distiller reads names them."""
        rows = [
            row("user", "find the total"),
            row("assistant", "[tool_call] grep(total)"),
            row("tool", "[no matches]"),
            row("assistant", "[tool_call] read(report.txt)"),
            row("tool", "Error: no such file\nthe error bars show the spread"),
            row("assistant", "[tool_call] read(summary.txt)"),
            row("tool", "Summary\nno errors were found in the audit"),
            row("assistant", "The total is 12."),
        ]
        llm = _RecordingLLM()
        d = _distiller(llm, _Conv({"t": rows}))
        meta, user, actor = _meta()
        metrics = {}
        await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["t"],
            metrics=metrics,
        )
        assert metrics["t"]["tool_errors"] == 1
        assert metrics["t"]["empty_results"] == 1
        [call] = llm.requests
        text = "".join(
            p.text for m in call for p in m.content if isinstance(p, TextContent)
        )
        assert "tool errors: 1; empty results: 1" in text
        assert "tool: [grep] [no matches]" in text

    async def test_existing_skills_are_nowhere_in_the_prompt_or_the_message(self):
        """The distiller does not see the skill bank."""
        import inspect

        from mirix.services import auto_dream_manager, session_experience_distiller

        llm = _RecordingLLM()
        d = _distiller(llm, _Conv({"s": alfworld_rows(2)}))
        meta, user, actor = _meta()
        await d.distill_sessions(
            meta_agent_state=meta, user=user, actor=actor, session_ids=["s"]
        )
        [call] = llm.requests
        for message in call:
            for part in message.content:
                if isinstance(part, TextContent):
                    assert "existing_skills" not in part.text
        assert "existing_skills" not in d._system_prompt
        params = inspect.signature(
            SessionExperienceDistiller.distill_sessions
        ).parameters
        assert "existing_skills" not in params
        assert not hasattr(SessionExperienceDistiller, "_format_existing_skills")
        assert not hasattr(
            auto_dream_manager.AutoDreamManager, "_existing_skill_summaries"
        )
        assert "existing_skills" not in inspect.getsource(session_experience_distiller)
        assert "existing_skill" not in inspect.getsource(auto_dream_manager)


async def _all_available(ids):
    return set(ids)


@pytest.mark.asyncio
class TestImagesAreExpandedBeforeSending:
    async def test_the_openai_client_inlines_each_image_where_it_was(self, tmp_path):
        """The client (fill_image_content_in_messages) turns every
        ImageContent of the distiller's message into base64 in place."""
        import base64
        import io

        from PIL import Image

        from mirix.llm_api.openai_client import OpenAIClient
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import Message

        paths = {}
        for name, color in (("file-a", "red"), ("file-b", "blue")):
            buffer = io.BytesIO()
            Image.new("RGB", (4, 3), color).save(buffer, "PNG")
            path = tmp_path / f"{name}.png"
            path.write_bytes(buffer.getvalue())
            paths[name] = path

        class Files:
            async def get_file_metadata_by_id(self, file_id):
                return SimpleNamespace(source_url=None, file_path=str(paths[file_id]))

        chunk = planner_for(
            [
                row(
                    "user",
                    "",
                    parts=[
                        text_part("first"),
                        image_part("file-a"),
                        text_part("middle"),
                    ],
                ),
                row("tool", "", parts=[image_part("file-b"), text_part("last")]),
            ]
        ).whole_chunk()
        client = OpenAIClient(
            LLMConfig(
                model="fake-model",
                model_endpoint_type="openai",
                model_endpoint="http://127.0.0.1:9",
                context_window=100_000,
                api_key="k",
            )
        )
        client.file_manager = Files()
        request = await client.build_request_data(
            [
                Message(role=MessageRole.system, content=[TextContent(text="sys")]),
                Message(role=MessageRole.user, content=list(chunk.content)),
            ],
            client.llm_config,
        )
        parts = request["messages"][1]["content"]
        shapes = [(p["type"], p.get("text", "")[-12:]) for p in parts]
        assert [t for t, _ in shapes] == [
            "text",
            "text",
            "image_url",
            "text",
            "text",
            "image_url",
            "text",
        ]
        assert parts[0]["text"].endswith("user: first\n")
        assert parts[3]["text"].startswith("\nmiddle\ntool: ")
        assert parts[6]["text"].startswith("\nlast")
        for index, name in ((2, "file-a"), (5, "file-b")):
            url = parts[index]["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            assert base64.b64decode(url.split(",", 1)[1]) == paths[name].read_bytes()
            assert parts[index]["image_url"]["detail"] == "auto"

    async def _chunk_and_files(self, tmp_path):
        import io

        from PIL import Image

        paths = {}
        for name, color in (("file-a", "red"), ("file-b", "blue")):
            buffer = io.BytesIO()
            Image.new("RGB", (4, 3), color).save(buffer, "PNG")
            path = tmp_path / f"{name}.png"
            path.write_bytes(buffer.getvalue())
            paths[name] = path

        class Files:
            async def get_file_metadata_by_id(self, file_id):
                return SimpleNamespace(
                    source_url=None,
                    file_path=str(paths[file_id]),
                    file_type="image/png",
                )

        chunk = planner_for(
            [
                row(
                    "user",
                    "",
                    parts=[
                        text_part("first"),
                        image_part("file-a"),
                        text_part("middle"),
                    ],
                ),
                row("tool", "", parts=[image_part("file-b"), text_part("last")]),
            ]
        ).whole_chunk()
        return chunk, paths, Files()

    def _messages(self, chunk):
        from mirix.schemas.enums import MessageRole
        from mirix.schemas.message import Message

        return [
            Message(role=MessageRole.system, content=[TextContent(text="sys")]),
            Message(role=MessageRole.user, content=list(chunk.content)),
        ]

    async def test_the_anthropic_client_inlines_each_image_where_it_was(self, tmp_path):
        import base64

        from mirix.llm_api.anthropic_client import AnthropicClient

        chunk, paths, files = await self._chunk_and_files(tmp_path)
        config = LLMConfig(
            model="claude-fake",
            model_endpoint_type="anthropic",
            context_window=100_000,
            max_output_tokens=8_000,
            api_key="k",
        )
        client = AnthropicClient(config)
        client.file_manager = files
        request = await client.build_request_data(self._messages(chunk), config)
        parts = request["messages"][0]["content"]
        assert [p["type"] for p in parts] == ["text", "image", "text", "image", "text"]
        assert parts[0]["text"].endswith("user: first\n")
        assert parts[2]["text"].startswith("\nmiddle\ntool: ")
        assert parts[4]["text"].startswith("\nlast")
        for index, name in ((1, "file-a"), (3, "file-b")):
            source = parts[index]["source"]
            assert source["type"] == "base64" and source["media_type"] == "image/png"
            assert base64.b64decode(source["data"]) == paths[name].read_bytes()

    async def test_the_google_client_inlines_each_image_where_it_was(self, tmp_path):
        import base64

        from mirix.llm_api.google_ai_client import GoogleAIClient

        chunk, paths, files = await self._chunk_and_files(tmp_path)
        config = LLMConfig(
            model="gemini-fake",
            model_endpoint_type="google_ai",
            model_endpoint="https://example.invalid",
            context_window=100_000,
            max_output_tokens=8_000,
            api_key="k",
        )
        client = GoogleAIClient(config)
        client.file_manager = files
        request = await client.build_request_data(
            self._messages(chunk), config, tools=[]
        )
        # Gemini has no system role: the system prompt is the first user turn.
        system, user = [c for c in request["contents"] if c["role"] == "user"]
        assert system["parts"] == [{"text": "sys"}]
        parts = user["parts"]
        kinds = ["image" if "inline_data" in p else "text" for p in parts]
        assert kinds == ["text", "text", "image", "text", "text", "image", "text"]
        assert parts[0]["text"].endswith("user: first\n")
        assert parts[1]["text"] == "<image 0>"
        assert parts[3]["text"].startswith("\nmiddle\ntool: ")
        assert parts[6]["text"].startswith("\nlast")
        for index, name in ((2, "file-a"), (5, "file-b")):
            data = parts[index]["inline_data"]
            assert data["mime_type"] == "image/png"
            assert base64.b64decode(data["data"]) == paths[name].read_bytes()
