"""The distiller states each session's outcome before its lessons.

MIRIX has no external judge: it never receives labels, gold answers or scores.
Its session distiller therefore judges, from the transcript alone, whether the
session achieved its task — ``succeeded``, ``failed`` or ``unclear`` — and the
evidence for it, in the SAME call and BEFORE the lessons (the reply is one JSON
object whose first fields are ``outcome_evidence`` and ``outcome``). The outcome
and its evidence are stored with every experience of the session (inside the
experience's ``evidence`` JSON, so no migration is needed), where the curator
sees them. An ``unclear`` session still yields what can be learned, at low
credibility, instead of nothing.

DB-free and offline: a hand-rolled fake LLM client and in-memory managers.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirix.schemas.mirix_message_content import TextContent
from mirix.schemas.skill_experience import SESSION_OUTCOMES, SkillExperience
from mirix.services.context_budget import LARGE_TEXT_CHARS, ModelLimits, TokenCounter
from mirix.services.distill_transcript import TranscriptPlanner, build_messages
from mirix.services.session_experience_distiller import (
    INFERRED_CREDIBILITY_MAX,
    OUTCOME_EVIDENCE_MAX_CHARS,
    OUTCOME_QUOTE_MIN_CHARS,
    OUTCOME_SOURCE_AGENT,
    OUTCOME_SOURCE_EXTERNAL,
    OUTCOME_SOURCE_NOT_FOUND,
    REPAIR_PROMPT,
    SELF_REPORTED_OUTCOME_CREDIBILITY_MAX,
    DistillerReply,
    SessionExperienceDistiller,
    SessionVerdict,
    _load_distiller_prompt,
    _RunContext,
    check_reply_outcome,
    chunk_verdicts,
    normalize_signal_type,
    outcome_quotes,
    outcome_source,
    parse_distiller_reply,
    reconcile_verdicts,
    transcript_index,
)
from mirix.services.skill_experience_curator import render_evidence, render_experience

PROMPT_PATH = Path("mirix/prompts/system/base/auto_dream_agent/procedural.txt")
CURATOR_PROMPT_PATH = Path("mirix/prompts/system/base/procedural_memory_agent.txt")


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


class _QuarterEncoding:
    """One token per 4 characters: deterministic counts, no vocabulary."""

    def encode(self, text, disallowed_special=()):
        return [0] * ((len(text) + 3) // 4)


_COUNTER = TokenCounter(encoding=_QuarterEncoding())


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeLLM:
    """Replies with ``replies`` in order (the last one repeats) and records
    every request's messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    async def send_llm_request(self, *, messages):
        self.requests.append(messages)
        index = min(len(self.requests), len(self.replies)) - 1
        return _response(self.replies[index])


class _RecordingManager:
    """``SkillExperienceManager.create_experience`` without a DB, validating
    like the real one."""

    def __init__(self):
        self.created = []

    async def create_experience(self, **kwargs):
        from mirix.schemas.skill_experience import SkillExperienceCreate

        kwargs.pop("created_by_id", None)
        rec = SimpleNamespace(**SkillExperienceCreate(**kwargs).model_dump())
        self.created.append(rec)
        return rec


class _Conversations:
    def __init__(self, rows_by_session):
        self.rows_by_session = rows_by_session

    async def list_turns_for_session(self, *, session_id, **_):
        return list(self.rows_by_session.get(session_id, []))


def _row(role, content, **extra):
    return SimpleNamespace(
        role=role, content=content, session_id="s1", distilled_at=None, **extra
    )


def _owner():
    return (
        SimpleNamespace(id="agent-meta-1"),
        SimpleNamespace(id="user-1"),
        SimpleNamespace(organization_id="org-1"),
    )


def _texts(messages):
    """Every text of a request, one string per message."""
    out = []
    for message in messages:
        out.append(
            "\n".join(getattr(part, "text", "") or "" for part in message.content or [])
        )
    return out


async def _distill(reply_or_llm, rows=None, *, metrics=None):
    llm = reply_or_llm if isinstance(reply_or_llm, _FakeLLM) else _FakeLLM(reply_or_llm)
    manager = _RecordingManager()
    distiller = SessionExperienceDistiller(
        llm_client=llm,
        experience_manager=manager,
        conversation_manager=_Conversations(
            {
                "s1": rows
                or [
                    _row("user", "Put the report in the shared folder."),
                    _row("assistant", "[tool_call] move(report.pdf, /shared)"),
                    _row("tool", "[move] error: permission denied"),
                    _row("assistant", "I could not move it."),
                ]
            }
        ),
    )
    meta, user, actor = _owner()
    metrics = {} if metrics is None else metrics
    created, processed = await distiller.distill_sessions(
        meta_agent_state=meta,
        user=user,
        actor=actor,
        session_ids=["s1"],
        metrics=metrics,
    )
    return created, processed, metrics, llm


def _reply(outcome, experiences, evidence="the tool said: permission denied"):
    return json.dumps(
        {"outcome_evidence": evidence, "outcome": outcome, "experiences": experiences}
    )


def _lesson(title, kind="worth_avoiding", signal="tool_error", credibility=0.9):
    return {
        "experience_type": kind,
        "title": title,
        "content": f"lesson: {title}",
        "importance": 0.7,
        "credibility": credibility,
        "evidence": {"quote": "permission denied", "signal_type": signal},
    }


# --------------------------------------------------------------------------- #
# The request asks for the outcome and its evidence before the lessons         #
# --------------------------------------------------------------------------- #


class TestTheRequestAsksForTheOutcomeFirst:
    def _format_block(self) -> str:
        return _load_distiller_prompt().split("## Output Format", 1)[1]

    def test_the_output_format_puts_outcome_evidence_and_outcome_before_the_lessons(
        self,
    ):
        block = self._format_block()
        positions = [
            block.index('"outcome_evidence"'),
            block.index('"outcome"'),
            block.index('"experiences"'),
            block.index('"experience_type"'),
        ]
        assert positions == sorted(positions)

    def test_the_prompt_defines_the_three_outcomes_and_what_each_one_teaches(self):
        text = _load_distiller_prompt()
        for outcome in SESSION_OUTCOMES:
            assert f"`{outcome}`" in text
        low = text.lower()
        # The owner's decision reverses the old "do not label the session" rule.
        assert "do not label the whole session" not in low
        # Judged from the transcript alone, not by redoing the task.
        assert "transcript alone" in low
        assert "do not redo the task" in low
        # An unclear session still yields what can be learned, at low credibility.
        unclear = low.split("- `unclear` —", 2)[2]
        # The procedure the agent followed.
        assert "the procedure the agent followed" in unclear and "low credibility" in unclear

    def test_only_the_environment_a_tool_or_the_user_grounds_a_high_outcome(self):
        """A check the agent states in its own words is no HIGH outcome, and
        the prompt says the quotes are checked (the code enforces both)."""
        text = _load_distiller_prompt()
        succeeded = text.split("- `succeeded` —", 1)[1].split("\n", 1)[0]
        assert "the agent checked its result" not in succeeded
        assert "states only in its own words" in text
        assert "MEDIUM credibility at most" in text
        assert "verbatim in double quotes" in text
        assert "is stored as `unclear`" in text
        scoring = text.split("## Scoring", 1)[1].split("\n\n", 2)[1]
        assert "an outcome only the agent's own words show" in scoring

    def test_the_prompt_states_the_rules_the_quotes_are_checked_by(self):
        """What the code enforces (:func:`outcome_source`) is what the model
        is told: substantial quotes, every one of them verbatim, and not the
        task statement alone."""
        text = _load_distiller_prompt()
        step1 = text.split("## Step 1", 1)[1].split("## Step 2", 1)[0]
        assert "at least three or four words" in step1
        assert "not the task statement" in step1
        assert "when any quote is not in the transcript" in step1
        assert "the task statement is the only place they occur" in step1
        # A quote is copied from one message as it is, and a
        # tool or the user repeating the agent's words shows only its claim.
        assert "exactly as it appears in ONE message" in step1
        assert "only repeats words the agent wrote before" in step1
        evidence = text.split("## Evidence", 1)[1].split("## Output Format", 1)[0]
        assert "checked like the quotes of `outcome_evidence`" in evidence

    def test_the_output_format_asks_for_no_label_score_or_answer_key_field(self):
        keys = set(re.findall(r'"([a-z_]+)"\s*:', self._format_block()))
        assert keys == {
            "outcome_evidence",
            "outcome",
            # MIRIX's own judgement of each served skill, not a label.
            "skill_feedback",
            "skill",
            "followed",
            "followed_evidence",
            "effect",
            "effect_evidence",
            "experiences",
            "experience_type",
            "title",
            "content",
            "importance",
            "credibility",
            "evidence",
            "quote",
            "signal_type",
        }

    def test_the_repair_prompt_asks_for_the_same_object(self):
        text = REPAIR_PROMPT
        assert text.index('"outcome_evidence"') < text.index('"outcome"')
        assert text.index('"outcome"') < text.index('"experiences"')

    @pytest.mark.asyncio
    async def test_the_request_closes_by_asking_for_the_outcome_before_the_lessons(
        self,
    ):
        _, _, _, llm = await _distill(_reply("failed", []))
        [request] = llm.requests
        system, user = _texts(request)
        assert system.startswith("You are the Experience Distiller")
        closing = user.rsplit("\n\n", 1)[1]
        # "Distill the transferable experiences ...: first the outcome_evidence
        # and the outcome, then the experiences, as one JSON object."
        assert closing.index("outcome_evidence") < closing.rindex("experiences")
        assert "JSON object" in closing and "JSON array" not in closing

    @pytest.mark.asyncio
    async def test_nothing_but_the_transcript_reaches_the_request(self):
        # Whatever else a conversation row carries (a tag, a score, an answer
        # key a caller may have stored beside it), the distiller sends only
        # its role and content: MIRIX gains no path to labels or scores.
        rows = [
            _row(
                "user",
                "How many units were sold?",
                session_tag="GOLD-LABEL-7731",
                score=0.123456,
                gold_answer="ANSWER-KEY-9917",
                metadata_={"reward": "REWARD-5521"},
            ),
            _row("assistant", "412 units.", session_tag="GOLD-LABEL-7731"),
        ]
        _, _, _, llm = await _distill(_reply("unclear", []), rows)
        sent = "\n".join(_texts(llm.requests[0]))
        for marker in ("GOLD-LABEL-7731", "0.123456", "ANSWER-KEY-9917", "REWARD-5521"):
            assert marker not in sent
        assert "How many units were sold?" in sent and "412 units." in sent


# --------------------------------------------------------------------------- #
# Replies: the outcome is stored with every lesson of the session              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
class TestOutcomeReplies:
    async def test_a_failed_session_yields_a_failure_lesson_with_its_outcome(self):
        reply = _reply(
            "failed",
            [_lesson("Check folder permissions before moving files")],
            evidence='the move failed ("permission denied") and was never retried',
        )
        created, processed, metrics, llm = await _distill(reply)
        assert processed == ["s1"] and len(llm.requests) == 1
        [lesson] = created
        assert lesson.experience_type == "worth_avoiding"
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "failed"
        assert evidence["outcome_evidence"] == (
            'the move failed ("permission denied") and was never retried'
        )
        assert evidence["signal_type"] == "tool_error"
        assert evidence["quote"] == "permission denied"
        assert lesson.credibility == 0.9  # a tool error is not capped
        assert metrics["s1"]["outcomes"] == ["failed"]

    async def test_every_lesson_of_the_session_carries_the_session_outcome(self):
        rows = [
            _row("user", "Put the report in the shared folder."),
            _row("assistant", "[tool_call] move(report.pdf, /shared)"),
            _row("tool", "[move] moved to /shared"),
            _row("assistant", "The report is in the shared folder."),
        ]
        reply = _reply(
            "succeeded",
            [
                _lesson(
                    "Confirm the target first",
                    kind="worth_learning",
                    signal="task_outcome",
                )
                | {"evidence": {"quote": "moved to /shared", "signal_type": "outcome"}},
                # A per-lesson outcome never overrides the session's.
                {
                    **_lesson("Retry with the right path"),
                    "evidence": {
                        "quote": "moved",
                        "signal_type": "self_correction",
                        "session_outcome": "failed",
                    },
                },
            ],
            evidence='the tool answered "moved to /shared"',
        )
        created, _, _, _ = await _distill(reply, rows)
        stored = [json.loads(e.evidence) for e in created]
        assert [e["session_outcome"] for e in stored] == ["succeeded", "succeeded"]
        assert {e["outcome_evidence"] for e in stored} == {
            'the tool answered "moved to /shared"'
        }
        # task_outcome grounds a lesson of an evidenced (succeeded) session.
        assert stored[0]["signal_type"] == "task_outcome"
        assert created[0].credibility == 0.9

    async def test_an_unclear_session_keeps_its_procedural_lesson_at_low_credibility(
        self,
    ):
        reply = _reply(
            "unclear",
            [
                _lesson(
                    "Read the totals row before summing the columns yourself",
                    kind="worth_learning",
                    signal="inferred",
                    credibility=0.8,  # over-rated: inference is LOW
                )
            ],
            evidence="an answer was given; nothing afterwards confirms or refutes it",
        )
        created, processed, metrics, _ = await _distill(reply)
        assert processed == ["s1"]
        [lesson] = created  # kept, not dropped
        assert lesson.experience_type == "worth_learning"
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "unclear"
        assert evidence["signal_type"] == "inferred"
        assert metrics["s1"]["outcomes"] == ["unclear"]
        assert metrics["s1"]["credibility_capped"] == 1

    async def test_task_outcome_needs_an_evidenced_outcome(self):
        # An unclear session cannot ground a lesson on its outcome: the
        # signal falls back to inference, and its credibility with it.
        reply = _reply(
            "unclear",
            [
                _lesson(
                    "Answer from the totals row",
                    kind="worth_learning",
                    signal="task_outcome",
                    credibility=0.9,
                )
            ],
        )
        [lesson], _, _, _ = await _distill(reply)
        assert json.loads(lesson.evidence)["signal_type"] == "inferred"
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX

    async def test_a_low_inferred_credibility_is_kept_as_given(self):
        reply = _reply(
            "unclear",
            [_lesson("x", kind="worth_learning", signal="inferred", credibility=0.2)],
        )
        [lesson], _, metrics, _ = await _distill(reply)
        assert lesson.credibility == 0.2
        assert metrics["s1"]["credibility_capped"] == 0

    async def test_stated_outcome_with_no_lessons_is_nothing_to_learn(self):
        created, processed, metrics, llm = await _distill(_reply("succeeded", []))
        assert created == [] and processed == ["s1"]
        assert len(llm.requests) == 1  # an explicit [] needs no repair
        assert metrics["s1"]["status"] == "ok"
        assert metrics["s1"]["outcomes"] == ["succeeded"]

    async def test_a_legacy_array_reply_is_still_accepted_without_an_outcome(self):
        reply = json.dumps([_lesson("Check folder permissions")])
        [lesson], processed, metrics, llm = await _distill(reply)
        assert processed == ["s1"] and len(llm.requests) == 1
        evidence = json.loads(lesson.evidence)
        assert "session_outcome" not in evidence
        assert "outcome_evidence" not in evidence
        assert metrics["s1"]["outcomes"] == [None]

    async def test_an_object_without_an_experiences_list_is_repaired(self):
        llm = _FakeLLM(
            json.dumps({"outcome_evidence": "e", "outcome": "failed"}),
            _reply("failed", [_lesson("repaired")]),
        )
        [lesson], processed, metrics, _ = await _distill(llm)
        assert lesson.title == "repaired" and processed == ["s1"]
        assert metrics["s1"]["repair_prompts"] == 1
        repair = _texts(llm.requests[1])[-1]
        assert '"experiences"' in repair and '"outcome"' in repair

    async def test_a_broken_reply_object_is_repaired_and_keeps_its_outcome(self):
        broken = (
            '{"outcome_evidence": "the tool said "permission denied"", '
            '"outcome": "failed", "experiences": ['
            + json.dumps(_lesson("t", signal="task_outcome"))
            + "]}"
        )
        llm = _FakeLLM(
            broken,
            _reply(
                "failed",
                [_lesson("t", signal="task_outcome")],
                evidence='the tool said "permission denied"',
            ),
        )
        [lesson], processed, metrics, _ = await _distill(llm)
        assert processed == ["s1"] and len(llm.requests) == 2
        assert metrics["s1"]["repair_prompts"] == 1
        assert metrics["s1"]["failure_kinds"] == ["parse"]
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "failed"
        assert evidence["signal_type"] == "task_outcome" and lesson.credibility == 0.9

    async def test_a_misspelled_signal_type_is_normalized_not_demoted(self):
        for spelling in ("Tool_Error", "tool error", " TOOL-ERROR ", "tool_failure"):
            reply = _reply(
                "failed",
                [{**_lesson("t"), "evidence": {"quote": "q", "signal_type": spelling}}],
            )
            [lesson], _, _, _ = await _distill(reply)
            assert json.loads(lesson.evidence)["signal_type"] == "tool_error", spelling
            assert lesson.credibility == 0.9, spelling
        assert normalize_signal_type("made up") is None
        assert normalize_signal_type(None) is None

    async def test_an_overlong_outcome_evidence_keeps_its_head_and_tail(self):
        long_evidence = "A" * 50 + "x" * (3 * OUTCOME_EVIDENCE_MAX_CHARS) + "Z" * 50
        reply = _reply("failed", [_lesson("t")], evidence=long_evidence)
        [lesson], _, metrics, _ = await _distill(reply)
        stored = json.loads(lesson.evidence)["outcome_evidence"]
        assert len(stored) <= OUTCOME_EVIDENCE_MAX_CHARS
        assert stored.startswith("A" * 50) and stored.endswith("Z" * 50)
        assert "…[truncated " in stored
        assert metrics["s1"]["fields_truncated"] == 1

    async def _two_chunks(self, first, second):
        """``_distill_plan`` over two chunks whose calls answer ``first`` and
        ``second`` (the first finishes last); no transcript: not checked."""

        class _ByChunk:
            def __init__(self):
                self.requests = []

            async def send_llm_request(self, *, messages):
                self.requests.append(messages)
                text = _texts(messages)[1]
                if "CHUNK-A" in text:
                    await asyncio.sleep(0.02)  # finishes last
                    return _response(first)
                return _response(second)

        distiller = SessionExperienceDistiller(llm_client=_ByChunk())
        meta, user, actor = _owner()
        ctx = _RunContext(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            call_gate=asyncio.Semaphore(4),
            recorder=None,
            commit_session=None,
            record_failure=None,
        )
        plan = SimpleNamespace(
            budget=1000,
            chunks=[
                SimpleNamespace(content=[TextContent(text="CHUNK-A transcript")]),
                SimpleNamespace(content=[TextContent(text="CHUNK-B transcript")]),
            ],
        )
        metrics = {
            "llm_calls": 0,
            "llm_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "failure_kinds": [],
            "invalid_items": 0,
            "timeouts": 0,
        }
        items = await distiller._distill_plan(ctx, "s1", plan, metrics)
        return items, metrics

    async def test_the_chunks_of_a_session_store_one_reconciled_outcome(self):
        # A long session is distilled in chunks, each in its own call; each
        # chunk states the outcome it can see (kept in chunk order in the
        # metrics), and every lesson carries the session's one outcome: the
        # last chunk's, which covers the end.
        items, metrics = await self._two_chunks(
            _reply("unclear", [_lesson("from a")]),
            _reply("failed", [_lesson("from b")]),
        )
        assert metrics["outcomes"] == ["unclear", "failed"]
        assert metrics["session_outcome"] == "failed"
        assert [(i["title"], i["evidence"]["session_outcome"]) for i in items] == [
            ("from a", "failed"),
            ("from b", "failed"),
        ]

    async def test_a_failure_an_earlier_chunk_saw_is_overruled_by_the_end(self):
        """Chunk 1 sees an error the agent later recovered from; the last
        chunk sees the success: chunk 1's lessons are not stored as causes of
        failure of a session that succeeded."""
        items, metrics = await self._two_chunks(
            _reply("failed", [_lesson("from a")], evidence='"error" at first'),
            _reply(
                "succeeded",
                [_lesson("from b", kind="worth_learning")],
                evidence='"done"',
            ),
        )
        assert metrics["outcomes"] == ["failed", "succeeded"]
        assert {i["evidence"]["session_outcome"] for i in items} == {"succeeded"}
        assert {i["evidence"]["outcome_evidence"] for i in items} == {'"done"'}

    async def test_disagreeing_chunks_before_an_unclear_end_are_unclear(self):
        items, metrics = await self._two_chunks(
            _reply("failed", [_lesson("from a")], evidence="E1"),
            _reply("unclear", [_lesson("from b")], evidence="E2"),
        )
        assert metrics["session_outcome"] == "failed"  # one evidenced: kept
        verdict = reconcile_verdicts(
            [
                SessionVerdict("failed", "E1"),
                SessionVerdict("succeeded", "E2"),
                SessionVerdict("unclear", "E3"),
            ]
        )
        assert verdict.outcome == "unclear"
        assert verdict.evidence == (
            "The chunks of this session disagree: chunk 1 stated failed (E1); "
            "chunk 2 stated succeeded (E2)"
        )
        assert reconcile_verdicts([SessionVerdict(None), SessionVerdict(None)]) == (
            SessionVerdict(None)
        )
        assert reconcile_verdicts([]).outcome is None


# --------------------------------------------------------------------------- #
# The stated outcome is checked against the transcript                          #
# --------------------------------------------------------------------------- #


_SELF_CHECK_ROWS = [
    _row("user", "What was the total enrollment in the table?"),
    _row("assistant", "I double-checked the table: the total is 281,000."),
]


@pytest.mark.asyncio
class TestTheOutcomeIsCheckedAgainstTheTranscript:
    async def test_an_outcome_no_quote_shows_is_stored_as_unclear(self):
        """The distiller claimed a tool confirmed the (wrong) answer; the
        session has no tool message. The lesson is not stored at HIGH
        credibility on an outcome the transcript does not show."""
        reply = _reply(
            "succeeded",
            [
                _lesson(
                    "Verify totals against the table",
                    kind="worth_learning",
                    signal="task_outcome",
                    credibility=0.95,
                )
                | {
                    "evidence": {
                        "quote": "value verified",
                        "signal_type": "task_outcome",
                    }
                }
            ],
            evidence='the tool confirmed "value verified"',
        )
        [lesson], _, metrics, _ = await _distill(reply, _SELF_CHECK_ROWS)
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "unclear"
        assert evidence["stated_outcome"] == "succeeded"
        assert evidence["outcome_source"] == OUTCOME_SOURCE_NOT_FOUND
        assert evidence["signal_type"] == "inferred"
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX
        assert metrics["s1"]["outcomes"] == ["succeeded"]  # as stated
        assert metrics["s1"]["session_outcome"] == "unclear"
        assert metrics["s1"]["outcomes_unverified"] == 1

    async def test_an_outcome_only_the_agent_states_caps_its_lessons(self):
        reply = _reply(
            "succeeded",
            [
                _lesson(
                    "Check the table before answering",
                    kind="worth_learning",
                    signal="task_outcome",
                    credibility=0.95,
                )
                | {
                    "evidence": {
                        "quote": "I double-checked the table",
                        "signal_type": "task_outcome",
                    }
                }
            ],
            evidence='the agent wrote "I double-checked the table"',
        )
        [lesson], _, metrics, _ = await _distill(reply, _SELF_CHECK_ROWS)
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "succeeded"
        assert evidence["outcome_source"] == OUTCOME_SOURCE_AGENT
        assert evidence["signal_type"] == "task_outcome"
        assert lesson.credibility == SELF_REPORTED_OUTCOME_CREDIBILITY_MAX == 0.7
        assert metrics["s1"]["outcomes_self_reported"] == 1
        assert metrics["s1"]["credibility_capped"] == 1

    async def test_an_outcome_a_tool_shows_keeps_its_lessons(self):
        reply = _reply(
            "failed",
            [_lesson("Check permissions", signal="task_outcome", credibility=0.9)],
            evidence="the tool answered \u201cError: Permission DENIED\u201d",
        )
        [lesson], _, metrics, _ = await _distill(reply)
        evidence = json.loads(lesson.evidence)
        assert evidence["session_outcome"] == "failed"
        assert evidence["outcome_source"] == OUTCOME_SOURCE_EXTERNAL
        assert lesson.credibility == 0.9
        assert metrics["s1"]["outcomes_unverified"] == 0

    async def test_an_unclear_or_legacy_outcome_is_not_checked(self):
        [lesson], _, metrics, _ = await _distill(
            _reply("unclear", [_lesson("x")], evidence='"not in the transcript"')
        )
        evidence = json.loads(lesson.evidence)
        assert "outcome_source" not in evidence and "stated_outcome" not in evidence
        assert metrics["s1"]["outcomes_unverified"] == 0


# --------------------------------------------------------------------------- #
# Verification means something: every quote found, substantial, beyond the    #
# task statement, in what the chunk was shown                                  #
# --------------------------------------------------------------------------- #


# An OfficeQA-like session whose only tool call failed: nothing shows success.
_RECEIPTS_ROWS = [
    _row("user", "What were total receipts in 1941? Answer with one number."),
    _row("assistant", "[tool_call] read_file(receipts_1941.csv)"),
    _row("tool", "[read_file] Error: file not found"),
    _row("assistant", "The total receipts in 1941 were 42."),
]

# ALFWorld: the environment's observations are user messages after the first.
_ALFWORLD_ROWS = [
    _row(
        "user",
        "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. "
        "Looking quickly around you, you see a countertop 1, a fridge 1, and a "
        "sinkbasin 1.\n\nYour task is to: put a clean apple in fridge.\n"
        "Admissible actions: go to countertop 1, go to fridge 1, go to sinkbasin 1",
    ),
    _row("assistant", "go to countertop 1"),
    _row(
        "user",
        "You arrive at countertop 1. On the countertop 1, you see a apple 1.\n"
        "Admissible actions: take apple 1 from countertop 1, go to fridge 1",
    ),
    _row("assistant", "take apple 1 from countertop 1"),
    _row("user", "You pick up the apple 1 from the countertop 1."),
    _row("assistant", "go to sinkbasin 1"),
    _row("user", "You arrive at sinkbasin 1. On the sinkbasin 1, you see nothing."),
    _row("assistant", "clean apple 1 with sinkbasin 1"),
    _row("user", "You clean the apple 1 using the sinkbasin 1."),
    _row("assistant", "go to fridge 1"),
    _row("user", "You arrive at fridge 1. The fridge 1 is closed."),
    _row("assistant", "put apple 1 in/on fridge 1"),
    _row("user", "You put the apple 1 in/on the fridge 1."),
]

# OfficeQA: the answer comes from a tool's search result.
_OFFICEQA_ROWS = [
    _row(
        "user",
        "Using the Treasury bulletin, what were the total receipts in fiscal "
        "year 1941?",
    ),
    _row("assistant", "[tool_call] search_documents(query='total receipts 1941')"),
    _row(
        "tool",
        "treasury_1941.txt: Table 3. Budget receipts and expenditures\n"
        "Fiscal year 1941 | Total receipts | 7,607 million dollars",
    ),
    _row("assistant", "The total receipts in fiscal year 1941 were $7,607 million."),
]

# DocVQA-like: a page read by an OCR tool, and the user's reply.
_DOCVQA_ROWS = [
    _row("user", "What is the invoice total on this page?"),
    _row("assistant", "[tool_call] ocr_page(page=1)"),
    _row("tool", "INVOICE No. 4471\nSubtotal: $1,150.00\nTOTAL DUE:   $1,240.00"),
    _row("assistant", "The invoice total is $1,240.00."),
    _row("user", "Yes, that matches the invoice, thanks."),
]


_FOUND_MUG = "You find a mug 7 in the cabinet 30."
_PUT_MUG = "You put the mug 7 in/on the coffeemachine 1."


def _long_session():
    """60 ALFWorld-like rounds: a find in round 50 (neither round 1 nor a
    final round) and the success in the last round."""
    rows = [
        _row("user", "Your task is to: put a clean mug in the coffee machine."),
        _row("assistant", "think: find a mug"),
    ]
    for number in range(2, 61):
        observation = f"obs {number:03d} " + "o" * 400
        if number == 50:
            observation = _FOUND_MUG + " " + "o" * 400
        elif number == 60:
            observation = _PUT_MUG
        rows.append(_row("user", observation))
        rows.append(_row("assistant", f"act {number:03d} " + "a" * 120))
    return rows


def _outcome_lesson(quote, credibility=0.95):
    return _lesson(
        "Verify the result against its source",
        kind="worth_learning",
        signal="task_outcome",
        credibility=credibility,
    ) | {"evidence": {"quote": quote, "signal_type": "task_outcome"}}


def _assert_unverified(lesson, metrics, stated):
    evidence = json.loads(lesson.evidence)
    assert evidence["session_outcome"] == "unclear"
    assert evidence["stated_outcome"] == stated
    assert evidence["outcome_source"] == OUTCOME_SOURCE_NOT_FOUND
    assert metrics["s1"]["session_outcome"] == "unclear"
    assert metrics["s1"]["outcomes_unverified"] == 1


def _assert_grounded(lesson, metrics, outcome, credibility):
    evidence = json.loads(lesson.evidence)
    assert evidence["session_outcome"] == outcome
    assert evidence["outcome_source"] == OUTCOME_SOURCE_EXTERNAL
    assert "stated_outcome" not in evidence
    assert evidence["signal_type"] == "task_outcome"
    assert lesson.credibility == credibility
    assert metrics["s1"]["outcomes_unverified"] == 0
    assert metrics["s1"]["outcomes_self_reported"] == 0


@pytest.mark.asyncio
class TestTheOutcomeCheckMeansSomething:
    async def test_an_invented_tool_quote_is_not_rescued_by_a_word_of_the_task(self):
        """Scenario 1: the tool quote is made up ("Error: file not
        found" is all the tool said), and "answer" only occurs in the task.
        The stated success is not stored as grounded by the user or a tool."""
        reply = _reply(
            "succeeded",
            [_outcome_lesson("verified: total matches 42")],
            evidence=(
                'The tool returned "verified: total matches 42" and the user '
                'asked to "answer" it.'
            ),
        )
        [lesson], _, metrics, _ = await _distill(reply, _RECEIPTS_ROWS)
        _assert_unverified(lesson, metrics, "succeeded")
        assert json.loads(lesson.evidence)["signal_type"] == "inferred"
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX

    async def test_quoting_the_task_statement_alone_shows_no_outcome(self):
        """Scenario 2: the question the session was asked was there
        before the agent did anything; it shows no result."""
        reply = _reply(
            "succeeded",
            [_outcome_lesson("What were total receipts in 1941?")],
            evidence='the user asked "what were total receipts in 1941" and got 42',
        )
        [lesson], _, metrics, _ = await _distill(reply, _RECEIPTS_ROWS)
        _assert_unverified(lesson, metrics, "succeeded")
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX

    async def test_every_quote_of_the_evidence_must_be_in_the_transcript(self):
        reply = _reply(
            "failed",
            [_lesson("Check the path first", signal="tool_error", credibility=0.9)],
            evidence=(
                'the tool said "Error: file not found" and then "the mirror '
                'returned HTTP 503"'
            ),
        )
        [lesson], _, metrics, _ = await _distill(reply, _RECEIPTS_ROWS)
        _assert_unverified(lesson, metrics, "failed")
        # A tool_error lesson keeps its own grounding: the error is real.
        assert lesson.credibility == 0.9

    async def test_an_invented_quote_of_a_task_outcome_lesson_unverifies(self):
        reply = _reply(
            "failed",
            [_outcome_lesson("the mirror returned HTTP 503", credibility=0.9)],
            evidence='the tool said "Error: file not found"',
        )
        [lesson], _, metrics, _ = await _distill(reply, _RECEIPTS_ROWS)
        _assert_unverified(lesson, metrics, "failed")
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX

    async def test_quotes_too_short_to_say_anything_verify_nothing(self):
        rows = [
            _row("user", "Tidy the build folder."),
            _row("assistant", "[tool_call] clean(build)"),
            _row("tool", "[clean] OK. Done."),
            _row("assistant", "The build folder is clean."),
        ]
        reply = _reply(
            "succeeded",
            [_outcome_lesson("Done")],
            evidence='the tool said "OK" and "Done"',
        )
        [lesson], _, metrics, _ = await _distill(reply, rows)
        _assert_unverified(lesson, metrics, "succeeded")

    async def test_an_alfworld_observation_grounds_the_outcome(self):
        # Case, spacing and punctuation differ from the transcript; restating
        # the task beside the real quotes is harmless.
        reply = _reply(
            "succeeded",
            [_outcome_lesson("You clean the apple 1 using the sinkbasin 1", 0.9)],
            evidence=(
                'the task was to "put a clean apple in fridge"; after "you clean '
                'the apple 1 using the  sinkbasin 1" the environment answered '
                '"You put the Apple 1 in / on the fridge 1."'
            ),
        )
        [lesson], _, metrics, _ = await _distill(reply, _ALFWORLD_ROWS)
        _assert_grounded(lesson, metrics, "succeeded", 0.9)

    async def test_an_officeqa_tool_result_grounds_the_outcome(self):
        reply = _reply(
            "succeeded",
            [_outcome_lesson("Total receipts | 7,607 million dollars", 0.85)],
            evidence=(
                'the user asked for "the total receipts in fiscal year 1941"; '
                'the search result reads "Fiscal year 1941 | TOTAL RECEIPTS | '
                '7,607 million dollars" and the answer repeats it'
            ),
        )
        [lesson], _, metrics, _ = await _distill(reply, _OFFICEQA_ROWS)
        _assert_grounded(lesson, metrics, "succeeded", 0.85)

    async def test_a_docvqa_ocr_result_and_the_users_reply_ground_the_outcome(
        self,
    ):
        reply = _reply(
            "succeeded",
            [_outcome_lesson("yes, that matches the invoice", 0.9)],
            evidence=(
                'the OCR read "Total due: $1,240.00" and the user replied '
                "\u201cYes \u2014 that matches the invoice\u201d"
            ),
        )
        [lesson], _, metrics, _ = await _distill(reply, _DOCVQA_ROWS)
        _assert_grounded(lesson, metrics, "succeeded", 0.9)

    @pytest.mark.parametrize(
        "quoted",
        [
            "'Yes, that's exactly what I needed'",
            "‘Yes, that’s exactly what I needed’",
        ],
    )
    async def test_a_single_quoted_reply_with_an_apostrophe_grounds_it(self, quoted):
        """Single quotes save escaping inside a JSON string, and English puts
        apostrophes inside them: the whole quote is checked, not its head."""
        rows = [
            _row("user", "Summarize the memo in two lines."),
            _row("assistant", "[tool_call] read_file(memo.txt)"),
            _row("tool", "[read_file] The Q3 budget review moves to Friday."),
            _row("assistant", "The Q3 budget review moves to Friday."),
            _row("user", "Yes, that's exactly what I needed, thanks!"),
        ]
        reply = _reply(
            "succeeded",
            [_outcome_lesson(quoted, 0.9)],
            evidence=f"the user replied {quoted}",
        )
        [lesson], _, metrics, _ = await _distill(reply, rows)
        _assert_grounded(lesson, metrics, "succeeded", 0.9)
        assert outcome_quotes(f"the user replied {quoted}", []) == [
            ["yesthatsexactlywhatineeded"]
        ]

    async def test_the_agent_repeating_the_task_is_only_its_own_claim(self):
        rows = _RECEIPTS_ROWS[:3] + [
            _row("assistant", "You asked what were total receipts in 1941: 42.")
        ]
        reply = _reply(
            "succeeded",
            [_outcome_lesson("what were total receipts in 1941")],
            evidence='"what were total receipts in 1941"',
        )
        [lesson], _, metrics, _ = await _distill(reply, rows)
        evidence = json.loads(lesson.evidence)
        assert evidence["outcome_source"] == OUTCOME_SOURCE_AGENT
        assert lesson.credibility == SELF_REPORTED_OUTCOME_CREDIBILITY_MAX

    # ------------------------- chunked sessions --------------------------- #

    async def _distill_chunked(self, first_chunk_reply, other_reply):
        class _ByChunk:
            def __init__(self):
                self.shown = {}

            async def send_llm_request(self, *, messages):
                text = _texts(messages)[1]
                label = text.split("\nchunk ", 1)[1].split(" covering", 1)[0]
                self.shown[label] = text
                if label.startswith("1/"):
                    return _response(first_chunk_reply)
                if callable(other_reply):
                    return _response(other_reply(text))
                return _response(other_reply)

        llm = _ByChunk()
        distiller = SessionExperienceDistiller(
            llm_client=llm,
            experience_manager=_RecordingManager(),
            conversation_manager=_Conversations({"s1": _long_session()}),
            token_counter=_COUNTER,
            model_limits=ModelLimits(context_window=12_000, max_output_tokens=2_000),
        )
        meta, user, actor = _owner()
        metrics = {}
        created, processed = await distiller.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["s1"],
            metrics=metrics,
        )
        assert processed == ["s1"] and metrics["s1"]["chunks"] >= 2
        return created, metrics["s1"], llm.shown

    async def test_a_chunk_cannot_be_verified_by_a_message_it_was_not_shown(self):
        created, metrics, shown = await self._distill_chunked(
            _reply(
                "succeeded",
                [_outcome_lesson(_FOUND_MUG, 0.9)],
                evidence=f'the environment said "{_FOUND_MUG}"',
            ),
            _reply("unclear", []),
        )
        first = next(text for label, text in shown.items() if label.startswith("1/"))
        assert _FOUND_MUG not in first  # round 50 is in a later chunk
        assert metrics["outcomes"][0] == "succeeded"
        assert metrics["outcomes_unverified"] == 1
        assert metrics["session_outcome"] == "unclear"
        [lesson] = created
        assert json.loads(lesson.evidence)["session_outcome"] == "unclear"
        assert lesson.credibility == INFERRED_CREDIBILITY_MAX

    async def test_a_chunk_is_verified_by_the_final_rounds_its_frame_shows(self):
        created, metrics, shown = await self._distill_chunked(
            _reply(
                "succeeded",
                [_outcome_lesson(_PUT_MUG, 0.9)],
                evidence=f'the last observation is "{_PUT_MUG}"',
            ),
            _reply("unclear", []),
        )
        first = next(text for label, text in shown.items() if label.startswith("1/"))
        assert _PUT_MUG in first.split("chunk_transcript", 1)[0]  # the frame
        assert metrics["outcomes_unverified"] == 0
        assert metrics["session_outcome"] == "succeeded"
        [lesson] = created
        evidence = json.loads(lesson.evidence)
        assert evidence["outcome_source"] == OUTCOME_SOURCE_EXTERNAL
        assert lesson.credibility == 0.9

    async def test_the_chunk_that_shows_a_message_is_verified_by_it(self):
        # Only the chunk that shows the find quotes it: how many chunks the
        # session takes depends on the prompt's length, not on this check.
        def by_chunk(text):
            if _FOUND_MUG not in text:
                return _reply("unclear", [])
            return _reply(
                "failed",
                [_outcome_lesson(_FOUND_MUG, 0.9)],
                evidence=f'"{_FOUND_MUG}" came too late',
            )

        created, metrics, _ = await self._distill_chunked(
            _reply("unclear", []), by_chunk
        )
        assert metrics["outcomes_unverified"] == 0
        assert metrics["session_outcome"] == "failed"
        assert created and all(
            json.loads(e.evidence)["outcome_source"] == OUTCOME_SOURCE_EXTERNAL
            for e in created
        )

    async def test_a_large_sessions_outcome_is_checked_off_the_event_loop(
        self, monkeypatch
    ):
        import threading

        import mirix.services.session_experience_distiller as module

        threads = []
        real = module.chunk_verdicts

        def recording(*args):
            threads.append(threading.get_ident())
            return real(*args)

        monkeypatch.setattr(module, "chunk_verdicts", recording)
        rows = _OFFICEQA_ROWS[:2] + [
            _row(
                "tool",
                "x" * (LARGE_TEXT_CHARS + 1)
                + "\nFiscal year 1941 | Total receipts | 7,607 million dollars",
            ),
            _OFFICEQA_ROWS[3],
        ]
        reply = _reply(
            "succeeded",
            [_outcome_lesson("Total receipts | 7,607 million dollars", 0.85)],
            evidence='the search result reads "Total receipts | 7,607 million"',
        )
        [lesson], _, metrics, _ = await _distill(reply, rows)
        _assert_grounded(lesson, metrics, "succeeded", 0.85)
        assert threads and threads[0] != threading.get_ident()


class TestHowQuotesAreChecked:
    def test_the_quotes_and_where_they_occur(self):
        items = [
            {
                "evidence": {
                    "quote": "moved the report ... to /shared/team",
                    "signal_type": "task_outcome",
                }
            },
            {
                "evidence": {
                    "quote": "ignored: not an outcome",
                    "signal_type": "tool_error",
                }
            },
        ]
        quotes = outcome_quotes(
            'the tool said "OK" and \u2018You won the game!\u2019', items
        )
        # "OK" is too short to check; an ellipsis splits a quote into parts.
        assert quotes == [["youwonthegame"], ["movedthereport", "tosharedteam"]]
        messages, _ = build_messages(
            [
                _row("user", "Put the mug in the coffee machine."),
                _row("assistant", "I think you won the game."),
                _row("tool", "[move] moved the report file to /shared/team"),
                _row("user", "Now put the mug in the coffee machine, please."),
            ],
            counter=_COUNTER,
        )
        index = transcript_index(messages)
        assert [(m.position, m.role, m.task_statement) for m in index] == [
            (1, "user", True),
            (2, "assistant", False),
            (3, "tool", False),
            (4, "user", False),
        ]
        # The line as the distiller is shown it, its role label included.
        assert index[0].text == "userputthemuginthecoffeemachine"
        assert outcome_source([["youwonthegame"]], index) == OUTCOME_SOURCE_AGENT
        assert (
            outcome_source([["movedthereport", "tosharedteam"]], index)
            == OUTCOME_SOURCE_EXTERNAL
        )
        # Every part of a quote in ONE message.
        assert (
            outcome_source([["putthemug", "tosharedteam"]], index)
            == OUTCOME_SOURCE_NOT_FOUND
        )
        # The task statement grounds nothing; a later user message does.
        assert outcome_source([["putthemug"]], index[:3]) == OUTCOME_SOURCE_NOT_FOUND
        assert outcome_source([["putthemug"]], index) == OUTCOME_SOURCE_EXTERNAL
        # Every quote must be found; the best place any of them is found wins.
        assert (
            outcome_source([["youwonthegame"], ["notthere"]], index)
            == OUTCOME_SOURCE_NOT_FOUND
        )
        assert (
            outcome_source([["youwonthegame"], ["tosharedteam"]], index)
            == OUTCOME_SOURCE_EXTERNAL
        )
        assert outcome_source([], index) == OUTCOME_SOURCE_NOT_FOUND
        # Without quotation marks the whole text must be verbatim.
        assert outcome_quotes("the tool said: permission denied", []) == [
            ["thetoolsaidpermissiondenied"]
        ]

    def test_how_long_a_quote_must_be(self):
        assert OUTCOME_QUOTE_MIN_CHARS == 12
        # Single words and stock fragments occur in any transcript.
        assert outcome_quotes('"You won!" "Done" "correct" "wrong answer"', []) == []
        # The shortest real environment and tool reports are long enough.
        assert outcome_quotes('"Nothing happens."', []) == [["nothinghappens"]]
        assert outcome_quotes('"file not found"', []) == [["filenotfound"]]
        # Each part of an elided quote is judged on its own.
        assert outcome_quotes('"moved ... to /shared/reports"', []) == [
            ["tosharedreports"]
        ]
        # CJK: a character weighs three letters, so four are enough.
        assert outcome_quotes("\u300c\u4efb\u52a1\u5b8c\u6210\u300d", []) == [
            ["\u4efb\u52a1\u5b8c\u6210"]
        ]
        assert outcome_quotes("\u300c\u6210\u529f\u4e86\u300d", []) == []

    def test_a_single_quoted_span_is_bounded_and_found_in_linear_time(self):
        # A span that may run past apostrophes is bounded: without the bound
        # every opening quote of a reply full of them is sought to its end.
        assert outcome_quotes("'" + "a" * 300 + "'", []) == [["a" * 300]]
        whole = "'" + "a" * 301 + "'"
        assert outcome_quotes(whole, []) == [["a" * 301]]  # no span: whole text
        started = time.perf_counter()
        for text in (" 'a" * 16_000, "‘a’b" * 16_000):
            outcome_quotes(text, [])
        assert time.perf_counter() - started < 3.0  # quadratic: ~20 s

    def test_a_compact_frame_shows_no_final_rounds_to_verify_by(self):
        messages, _ = build_messages(_long_session(), counter=_COUNTER)
        planner = TranscriptPlanner(
            messages, agent_id="agent-1", session_id="s1", counter=_COUNTER
        )
        quoting_the_end = parse_distiller_reply(
            _reply(
                "succeeded",
                [_outcome_lesson(_PUT_MUG, 0.9)],
                evidence=f'the last observation is "{_PUT_MUG}"',
            )
        )
        full = planner.plan(6_000)
        assert not full.compact_frame
        compact = next(
            plan
            for plan in map(planner.plan, range(750, 3_000, 50))
            if plan.compact_frame and len(plan.chunks) > 2
        )
        for plan, source in (
            (full, OUTCOME_SOURCE_EXTERNAL),
            (compact, OUTCOME_SOURCE_NOT_FOUND),
        ):
            assert len(plan.chunks) >= 2
            replies = [quoting_the_end] + [
                parse_distiller_reply(_reply("unclear", []))
            ] * (len(plan.chunks) - 1)
            verdicts = chunk_verdicts(replies, plan, planner)
            assert verdicts[0].source == source
            assert [v.source for v in verdicts[1:]] == [None] * (len(plan.chunks) - 1)
        # Without a transcript nothing is checked.
        [unchecked] = chunk_verdicts([quoting_the_end], full, None)
        assert unchecked == SessionVerdict(
            "succeeded", quoting_the_end.outcome_evidence
        )


# The quoted-span syntax :func:`outcome_quotes` reads, as one regular
# expression: the oracle the linear scanner must agree with. As a regex it is
# quadratic on a text of opening marks that never close (every one is sought
# to the end of the text), so it only checks short texts here.
_QUOTED_SPAN_ORACLE = re.compile(
    r'"([^"]+)"|\u201c([^\u201d]+)\u201d'
    r"|\u2018((?:[^\u2019]|\u2019(?=[A-Za-z])){1,300})\u2019(?![A-Za-z])"
    r"|\u300c([^\u300d]+)\u300d|\u300e([^\u300f]+)\u300f"
    r"|(?<![\w'])'((?:[^']|'(?=\w)){1,300})'(?!\w)"
)


def _verdict(evidence, rows, outcome="succeeded", items=()):
    """The verdict of a reply stating ``outcome`` with ``evidence`` (and
    ``items``), checked against ``rows`` as the distiller was shown them."""
    messages, _ = build_messages(rows, counter=_COUNTER)
    reply = DistillerReply(
        outcome=outcome, outcome_evidence=evidence, items=list(items)
    )
    return check_reply_outcome(reply, transcript_index(messages))


class TestQuotesAsTheDistillerCopiesThem:
    """A quote copied faithfully from what the distiller was shown
    is found, whatever label, normalization or escaping it carries; the
    agent's own words a tool prints back are still only the agent's claim;
    and reading the quotes stays linear whatever marks the reply holds."""

    _PORT_ROWS = [
        _row("user", "Find the config file and fix the port."),
        _row("assistant", '[tool_call] grep({"pattern": "port"})'),
        _row("tool", "port = 8080 found in /etc/app.conf"),
        _row("assistant", "Changed the port."),
        _row("user", "Great, the service is reachable now, thanks!"),
    ]

    def test_a_line_quoted_with_its_role_label_is_found(self):
        """The distiller is shown ``role: text`` lines: a quote of the whole
        line, label included, is verbatim, as is one without the label."""
        for evidence in (
            'The user said "user: Great, the service is reachable now, thanks!"',
            'The user said "Great, the service is reachable now, thanks!"',
            'The tool returned "tool: [grep] port = 8080 found in /etc/app.conf"',
            'The tool returned "[grep] port = 8080 found in /etc/app.conf"',
        ):
            verdict = _verdict(evidence, self._PORT_ROWS)
            assert (verdict.outcome, verdict.source) == (
                "succeeded",
                OUTCOME_SOURCE_EXTERNAL,
            ), evidence
        # The label must be the message's own: a tool never said the user's words.
        verdict = _verdict(
            'The tool said "tool: Great, the service is reachable now"',
            self._PORT_ROWS,
        )
        assert (verdict.outcome, verdict.source) == (
            "unclear",
            OUTCOME_SOURCE_NOT_FOUND,
        )

    def test_unicode_forms_and_json_escapes_do_not_break_a_verbatim_quote(self):
        import unicodedata

        rows = [
            _row("user", "Find the cafe menu and the Q3 receipts."),
            _row("assistant", "[tool_call] search(menu)"),
            _row(
                "tool",
                unicodedata.normalize(
                    "NFD", "[search] R\u00e9sultat: menu du caf\u00e9 trouv\u00e9 "
                )
                + "\uff11\uff12\uff13 entries",
            ),
            _row("assistant", "[tool_call] read_json(receipts.json)"),
            _row(
                "tool",
                '[read_json] {"result": "Table 3.\\nTotal receipts\\n1941: 7,607 '
                'million", "note": "caf\\u00e9 prices in \\"francs\\"", '
                '"path": "C:\\\\new\\\\table.csv"}',
            ),
            _row("assistant", "[tool_call] read_text(notes.txt)"),
            _row("tool", "[read_text] Budget line\nTotal outlays 9,062 million"),
            _row("assistant", "Found both."),
        ]
        for quote in (
            # NFC against NFD, full-width digits against ASCII ones.
            "R\u00e9sultat: menu du caf\u00e9 trouv\u00e9",
            "menu du caf\u00e9 trouv\u00e9 123 entries",
            # JSON escapes the tool's text holds, quoted as what they stand for.
            "Total receipts 1941: 7,607 million",
            'caf\u00e9 prices in "francs"',
            # ... or as the transcript shows them (its escapes kept).
            "Table 3.\\nTotal receipts",
            # A path whose backslashes the reply's own JSON unescaped.
            "C:\\new\\table.csv",
            # Plain text quoted with an escape for its line break.
            "Budget line\\nTotal outlays 9,062 million",
        ):
            verdict = _verdict(f'the tool returned "{quote}"', rows)
            assert (verdict.outcome, verdict.source) == (
                "succeeded",
                OUTCOME_SOURCE_EXTERNAL,
            ), quote
        # Normalizing never makes an invented quote match.
        verdict = _verdict(
            'the tool returned "Total receipts 1941: 8,000 million"', rows
        )
        assert verdict.source == OUTCOME_SOURCE_NOT_FOUND

    def test_the_agents_words_a_tool_prints_back_are_its_own_claim(self):
        """A tool that echoes what the agent wrote before (a ``print`` of its
        own verdict) shows only the agent's claim: agent_only, not
        user_or_tool. A tool's output the agent repeats AFTER it is the
        tool's."""
        claim = "Verified: total receipts 1941 = 7,607 million (matches table)"
        rows = [
            _row("user", "What were total receipts in 1941? Files: /data/tb1941.txt"),
            _row(
                "assistant",
                "[tool_call] python("
                + json.dumps({"code": f"print({json.dumps(claim)})"})
                + ")",
            ),
            _row("tool", f"[python] {claim}"),
            _row("assistant", "<answer>7,607 million</answer>"),
        ]
        verdict = _verdict(f'The python tool returned "{claim}"', rows)
        assert (verdict.outcome, verdict.source) == ("succeeded", OUTCOME_SOURCE_AGENT)
        # The same echo by the user (the agent's words cited back) is the
        # agent's claim too.
        chat = [
            _row("user", "Convert the report to Excel."),
            _row("assistant", "Done: the report has 1,204 rows and 9 sheets."),
            _row("user", "You said the report has 1,204 rows and 9 sheets."),
        ]
        verdict = _verdict('"the report has 1,204 rows and 9 sheets"', chat)
        assert verdict.source == OUTCOME_SOURCE_AGENT
        # The tool said it first, the agent repeated it: the tool's words.
        verdict = _verdict(
            'the search result reads "Total receipts | 7,607 million dollars"',
            _OFFICEQA_ROWS
            + [_row("assistant", "Total receipts | 7,607 million dollars.")],
        )
        assert verdict.source == OUTCOME_SOURCE_EXTERNAL
        # One quote the agent echoed and one a tool showed on its own: the
        # tool grounds the outcome.
        verdict = _verdict(
            f'"{claim}" and the file reads "Fiscal year 1941 total receipts"',
            rows
            + [
                _row("assistant", "[tool_call] read_file(tb1941.txt)"),
                _row("tool", "[read_file] Fiscal year 1941 total receipts 7,607"),
            ],
        )
        assert verdict.source == OUTCOME_SOURCE_EXTERNAL

    def test_every_kind_of_quotation_mark_is_read_in_linear_time(self):
        """A curly or CJK opening mark with no closing one was
        sought to the end of the text from every opening mark (10 s for 20k
        curly openers; the evidence is capped only when stored)."""
        started = time.perf_counter()
        for text in (
            "\u201cthe agent said " * 20_000,
            "\u300c\u4ed6\u8bf4 " * 20_000,
            "\u300e\u4ed6\u8bf4 " * 20_000,
            "\u2018a " * 20_000,
            " 'a" * 20_000,
            '"' + "x" * 200_000,
        ):
            outcome_quotes(text, [])
        assert time.perf_counter() - started < 2.0  # quadratic: ~15 s
        # Unclosed marks do not hide the quotes after them.
        text = "\u201c" * 1_000 + ' then "You put the apple 1 in the fridge 1."'
        assert outcome_quotes(text, []) == [["youputtheapple1inthefridge1"]]

    def test_the_spans_are_those_of_the_quote_syntax(self):
        """The linear scanner finds exactly the spans the quote syntax
        (:data:`_QUOTED_SPAN_ORACLE`) does, on random texts of quotation
        marks, apostrophes, letters and CJK characters."""
        import random

        from mirix.services.session_experience_distiller import _quoted_spans

        alphabet = [
            '"',
            "\u201c",
            "\u201d",
            "\u2018",
            "\u2019",
            "\u300c",
            "\u300d",
            "\u300e",
            "\u300f",
            "'",
            "a",
            "b",
            "Z",
            "1",
            " ",
            " ",
            ".",
            "\u4efb",
            "\n",
        ]
        rng = random.Random(20260930)
        for _ in range(20_000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
            expected = [
                next(group for group in m.groups() if group)
                for m in _QUOTED_SPAN_ORACLE.finditer(text)
            ]
            expected = expected or ([text] if text.strip() else [])
            assert _quoted_spans(text) == expected, repr(text)


# --------------------------------------------------------------------------- #
# Parsing the reply                                                            #
# --------------------------------------------------------------------------- #


class TestParseDistillerReply:
    def test_the_object_shape(self):
        reply = parse_distiller_reply(_reply("failed", [_lesson("a")], evidence="e"))
        assert (reply.outcome, reply.outcome_evidence) == ("failed", "e")
        assert [i["title"] for i in reply.items] == ["a"]

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda s: f"```json\n{s}\n```",
            lambda s: f"Here is my analysis:\n{s}\nDone.",
            lambda s: f"[{s}]",  # the object inside an array
        ],
    )
    def test_fenced_prose_wrapped_or_enclosed_objects_parse(self, wrap):
        reply = parse_distiller_reply(wrap(_reply("succeeded", [_lesson("b")])))
        assert reply.outcome == "succeeded"
        assert [i["title"] for i in reply.items] == ["b"]

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Succeeded", "succeeded"),
            (" FAILED ", "failed"),
            ("success", "succeeded"),
            ("failure", "failed"),
            ("unknown", "unclear"),
            ("partially", None),
            (None, None),
            (3, None),
        ],
    )
    def test_the_outcome_is_normalized_or_left_unstated(self, raw, expected):
        text = json.dumps({"outcome_evidence": "e", "outcome": raw, "experiences": []})
        assert parse_distiller_reply(text).outcome == expected

    def test_a_single_experience_object_under_experiences_is_wrapped(self):
        text = json.dumps({"outcome": "failed", "experiences": _lesson("solo")})
        assert [i["title"] for i in parse_distiller_reply(text).items] == ["solo"]

    @pytest.mark.parametrize(
        "text",
        [
            json.dumps({"outcome": "failed", "experiences": None}),
            json.dumps({"outcome": "failed", "experiences": "none"}),
            json.dumps({"outcome": "failed", "experiences": [1, 2]}),
            "no json here",
            "",
            None,
        ],
    )
    def test_unusable_replies_are_none(self, text):
        assert parse_distiller_reply(text) is None

    def test_legacy_shapes_parse_without_an_outcome(self):
        for text in ("[]", json.dumps([_lesson("a")]), json.dumps(_lesson("solo"))):
            reply = parse_distiller_reply(text)
            assert reply is not None and reply.outcome is None
            assert reply.outcome_evidence == ""
        assert parse_distiller_reply("[]").items == []

    @pytest.mark.parametrize(
        "text",
        [
            # An unescaped quote inside outcome_evidence (it quotes the transcript).
            '{"outcome_evidence": "the tool said "permission denied" and the agent '
            'gave up", "outcome": "failed", "experiences": ['
            + json.dumps(_lesson("a"))
            + "]}",
            # A trailing comma, and a missing final brace.
            _reply("failed", [_lesson("a")])[:-1] + ",}",
            _reply("failed", [_lesson("a")])[:-1],
        ],
    )
    def test_a_broken_reply_object_is_repaired_not_taken_for_a_legacy_array(self, text):
        """The array inside a broken reply object used to be taken as a legacy
        answer: the outcome was silently dropped (no repair, no PARSE noted)
        and a task_outcome lesson fell to inferred."""
        assert parse_distiller_reply(text) is None

    def test_a_legacy_array_with_a_per_lesson_outcome_is_still_legacy(self):
        for lesson in (
            {**_lesson("a"), "outcome": "failed"},
            {**_lesson("a"), "evidence": {"quote": "q", "outcome": "failed"}},
        ):
            reply = parse_distiller_reply(json.dumps([lesson]))
            assert reply is not None and reply.outcome is None
            assert [i["title"] for i in reply.items] == ["a"]
        # An outcome object without its experiences is still unusable.
        assert parse_distiller_reply(json.dumps({"outcome": "failed"})) is None
        assert (
            parse_distiller_reply(
                json.dumps({"outcome": "failed"}) + "\n" + json.dumps([_lesson("a")])
            )
            is None
        )

    def test_non_dict_lessons_are_dropped(self):
        text = json.dumps({"outcome": "failed", "experiences": [_lesson("k"), 7, "x"]})
        assert [i["title"] for i in parse_distiller_reply(text).items] == ["k"]


# --------------------------------------------------------------------------- #
# The curator sees the outcome; legacy experiences still load                  #
# --------------------------------------------------------------------------- #


def _experience(evidence):
    return SkillExperience(
        id="sexp-1",
        agent_id="agent-meta-1",
        user_id="user-1",
        organization_id="org-1",
        session_id="s1",
        experience_type="worth_avoiding",
        title="Check folder permissions",
        content="lesson",
        importance=0.7,
        credibility=0.9,
        evidence=evidence,
    )


class TestTheCuratorSeesTheOutcome:
    def test_the_session_outcome_leads_the_rendered_evidence(self):
        text = render_evidence(
            json.dumps(
                {
                    "quote": "permission denied",
                    "signal_type": "tool_error",
                    "session_outcome": "failed",
                    "outcome_evidence": "the move was never retried",
                }
            )
        )
        assert text.split("\n") == [
            "- session_outcome: failed",
            "- outcome_evidence: the move was never retried",
            "- signal_type: tool_error",
            "- quote: permission denied",
        ]

    def test_a_legacy_experience_without_the_fields_still_loads_and_renders(self):
        from mirix.orm.skill_experience import SkillExperience as SkillExperienceORM

        legacy = _experience(
            json.dumps({"quote": "permission denied", "signal_type": "tool_error"})
        )
        text = render_experience(legacy)
        assert "- signal_type: tool_error" in text
        assert "session_outcome" not in text and "outcome_evidence" not in text
        # Stored inside the evidence JSON: no new column, so no migration.
        assert "session_outcome" not in SkillExperienceORM.__table__.c

    def test_the_curator_prompt_explains_how_to_weigh_the_outcome(self):
        text = CURATOR_PROMPT_PATH.read_text(encoding="utf-8")
        assert "session_outcome" in text and "outcome_evidence" in text
        weigh = text.split("**Weigh the evidence.**", 1)[1].split("\n", 1)[0]
        for outcome in SESSION_OUTCOMES:
            assert f"`{outcome}`" in weigh
        # Where the transcript shows the outcome, and an unshown claim.
        for key in (
            "outcome_source",
            "`user_or_tool`",
            "`agent_only`",
            "stated_outcome",
        ):
            assert key in text, key
        # A skill ends in the action that carries the task out,
        # with the closest choice offered rather than searching on.
        weigh = text.split("**Weigh the evidence.**", 1)[1].split("\n", 1)[0]
        assert "A skill leads to the action that carries the task out" in weigh
        assert "give it no branch that ends without that action" in weigh
        assert (
            "unless a succeeded session, a user's confirmation or correction, or a "
            "tool's error shows that ending without it was right" in weigh
        )
        assert "taking the closest choice offered, rather than searching on" in weigh
        # An approval step stays where the sessions asked for it.
        assert "keep a step that asks the user to approve an action" in weigh
        for name in ("alfworld", "webshop", "officeqa", "purchase", "buy"):
            assert name not in text.lower()

