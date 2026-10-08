"""Tests for general per-session experience distillation (DB-free).

Layered for the general Experience Distiller: no benchmark oracle and no
external success/failure label (the distiller states each session's outcome
itself, from the transcript alone: tests/test_distiller_outcome.py).

1. JSON-array parse robustness — bare array, fenced ```json, prose-wrapped,
   single-object fallback, empty `[]`, garbage -> [] (lenient) / None (strict:
   an unusable answer is never "no experiences").
2. Parsed-experience -> SkillExperience mapping (_persist_experiences via a stub
   manager, no DB): importance/credibility clamping, evidence normalization,
   bad-type / empty-title skips, head + tail caps (title 1,024, content and
   evidence 65,536; no silent 512-char quote cut), status='pending'.
3. Prioritization ordering invariant — importance*credibility is the priority
   the curator payload is built around.
4. Single session yields MULTIPLE mixed worth_learning + worth_avoiding from one
   fixed fake LLM reply (no external label needed).
5. Benchmark-decoupling assertion — the distiller prompt/flow carries no
   benchmark vocabulary and needs no external success/failure oracle.

The LLM client is a hand-rolled fake (no network). No Postgres: the experience
manager is replaced with an in-memory recorder so the per-session persistence
path is exercised without a DB.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirix.helpers.json_parsing import (
    parse_distiller_json_array,
    parse_json_array_or_none,
)
from mirix.schemas.skill_experience import (
    SKILL_EXPERIENCE_MAX_CONTENT_LEN,
    SKILL_EXPERIENCE_MAX_EVIDENCE_LEN,
    SKILL_EXPERIENCE_MAX_TITLE_LEN,
)
from mirix.services.session_experience_distiller import SessionExperienceDistiller

# ============================ JSON-array parse robustness ===================


class TestParseDistillerJsonArray:
    def test_bare_array(self):
        raw = '[{"experience_type":"worth_learning","title":"t"}]'
        out = parse_distiller_json_array(raw)
        assert len(out) == 1
        assert out[0]["title"] == "t"

    def test_empty_array_is_nothing_worth_remembering(self):
        assert parse_distiller_json_array("[]") == []

    def test_fenced_json_block(self):
        raw = '```json\n[{"experience_type":"worth_avoiding","title":"x"}]\n```'
        out = parse_distiller_json_array(raw)
        assert len(out) == 1
        assert out[0]["experience_type"] == "worth_avoiding"

    def test_plain_fence(self):
        raw = '```\n[{"title":"a"},{"title":"b"}]\n```'
        out = parse_distiller_json_array(raw)
        assert [o["title"] for o in out] == ["a", "b"]

    def test_prose_wrapped_array(self):
        raw = 'Here are the experiences:\n[{"title":"only"}]\nThanks!'
        out = parse_distiller_json_array(raw)
        assert len(out) == 1 and out[0]["title"] == "only"

    def test_single_object_fallback_is_wrapped(self):
        # A model that emits ONE experience without the enclosing array must not
        # be silently dropped.
        raw = '{"experience_type":"worth_learning","title":"solo"}'
        out = parse_distiller_json_array(raw)
        assert len(out) == 1 and out[0]["title"] == "solo"

    def test_non_dict_elements_dropped(self):
        raw = '[{"title":"keep"}, 7, "noise", null]'
        out = parse_distiller_json_array(raw)
        assert out == [{"title": "keep"}]

    def test_garbage_yields_empty(self):
        assert parse_distiller_json_array("not json at all") == []
        assert parse_distiller_json_array("") == []
        assert parse_distiller_json_array(None) == []


class TestStrictJsonArray:
    """The distiller's parser tells an explicit ``[]`` from an unusable answer."""

    def test_explicit_empty_array_is_empty(self):
        assert parse_json_array_or_none("[]") == []
        assert parse_json_array_or_none("```json\n[ ]\n```") == []

    @pytest.mark.parametrize(
        "raw",
        [None, "", "   ", "No experiences here.", '[{"title": "cut off', "[1, 2]"],
    )
    def test_unusable_answers_are_none(self, raw):
        assert parse_json_array_or_none(raw) is None

    def test_recoverable_shapes_still_parse(self):
        assert parse_json_array_or_none('Sure: [{"title": "a"}]') == [{"title": "a"}]
        assert parse_json_array_or_none('{"title": "solo"}') == [{"title": "solo"}]
        # An array of scalars before the real one is skipped, not taken as [].
        assert parse_json_array_or_none('scores [1, 2]; [{"title": "b"}]') == [
            {"title": "b"}
        ]


# ===================== Parsed-experience -> SkillExperience mapping =========


class _RecordingManager:
    """In-memory stand-in for SkillExperienceManager.create_experience.

    Validates exactly like the real manager (via SkillExperienceCreate) so the
    clamp / enum / length contracts are exercised, but never touches a DB.
    """

    def __init__(self):
        self.created = []

    async def create_experience(self, **kwargs):
        from mirix.schemas.skill_experience import SkillExperienceCreate

        # Mirror the real manager: created_by_id feeds the _created_by_id audit
        # column (client attribution for erasure), not SkillExperienceCreate.
        created_by_id = kwargs.pop("created_by_id", None)
        validated = SkillExperienceCreate(**kwargs)
        rec = SimpleNamespace(**validated.model_dump(), created_by_id=created_by_id)
        self.created.append(rec)
        return rec


def _distiller_with(manager):
    # llm_config=None / llm_client=None: we never call the LLM in these mapping
    # tests, we drive _persist_experiences directly.
    return SessionExperienceDistiller(experience_manager=manager)


def _meta_user_actor():
    meta = SimpleNamespace(id="agent-meta-1")
    user = SimpleNamespace(id="user-1")
    actor = SimpleNamespace(organization_id="org-1")
    return meta, user, actor


@pytest.mark.asyncio
class TestPersistMapping:
    async def test_basic_mapping_status_pending(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {
                "experience_type": "worth_learning",
                "title": "Cache the resolved path",
                "content": "When repeatedly resolving X, cache it.",
                "importance": 0.8,
                "credibility": 0.9,
                "evidence": {
                    "quote": "great, that worked",
                    "signal_type": "user_confirmation",
                },
            }
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="sess-1",
            parsed=parsed,
        )
        assert len(out) == 1
        rec = mgr.created[0]
        assert rec.experience_type == "worth_learning"
        assert rec.status == "pending"
        assert rec.session_id == "sess-1"
        assert rec.agent_id == "agent-meta-1"
        assert rec.user_id == "user-1"
        assert rec.organization_id == "org-1"
        ev = json.loads(rec.evidence)
        assert ev["signal_type"] == "user_confirmation"
        assert ev["quote"] == "great, that worked"

    async def test_clamps_importance_and_credibility(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {
                "experience_type": "worth_avoiding",
                "title": "t1",
                "importance": 5.0,
                "credibility": -3.0,
            },
            {
                "experience_type": "worth_learning",
                "title": "t2",
                "importance": "garbage",
                "credibility": None,
            },
        ]
        await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        assert mgr.created[0].importance == 1.0
        assert mgr.created[0].credibility == 0.0
        assert mgr.created[1].importance == 0.0  # garbage -> 0.0
        assert mgr.created[1].credibility == 0.0

    async def test_bad_type_is_skipped(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {"experience_type": "partial", "title": "bad"},  # bad enum
            {"experience_type": "worth_learning", "title": "good"},  # kept
            {"experience_type": None, "title": "alsobad"},  # missing
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        assert len(out) == 1
        assert mgr.created[0].title == "good"

    async def test_empty_title_is_skipped(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {"experience_type": "worth_learning", "title": "   "},
            {"experience_type": "worth_learning", "title": ""},
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        assert out == []
        assert mgr.created == []

    async def test_length_caps_enforced(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {
                "experience_type": "worth_avoiding",
                "title": "T" * (SKILL_EXPERIENCE_MAX_TITLE_LEN + 50),
                "content": "C" * (SKILL_EXPERIENCE_MAX_CONTENT_LEN + 100),
            }
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        assert len(out) == 1
        title, content = mgr.created[0].title, mgr.created[0].content
        assert len(title) <= SKILL_EXPERIENCE_MAX_TITLE_LEN
        assert len(content) <= SKILL_EXPERIENCE_MAX_CONTENT_LEN
        # Head + tail around an explicit marker, never a silent cut.
        assert title.startswith("T") and title.endswith("T")
        assert "…[truncated " in title and "…[truncated " in content

    def test_caps_are_the_m4_3_values(self):
        assert SKILL_EXPERIENCE_MAX_TITLE_LEN == 1024
        assert SKILL_EXPERIENCE_MAX_CONTENT_LEN == 65536
        assert SKILL_EXPERIENCE_MAX_EVIDENCE_LEN == 65536

    async def test_fields_within_the_caps_are_kept_whole(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        quote = "q" * 5000  # the old code cut every quote to 512 chars
        parsed = [
            {
                "experience_type": "worth_learning",
                "title": "T" * 1000,
                "content": "C" * 60000,
                "evidence": {"quote": quote, "signal_type": "tool_error"},
            }
        ]
        await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor, session_id="s", parsed=parsed
        )
        rec = mgr.created[0]
        assert rec.title == "T" * 1000 and rec.content == "C" * 60000
        assert json.loads(rec.evidence)["quote"] == quote

    async def test_an_oversize_quote_keeps_valid_json_within_the_cap(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        quote = 'say "hi"\n' * 20000  # escaping makes the JSON longer
        metrics = {"fields_truncated": 0, "field_chars_truncated": 0}
        payloads = d._experience_payloads(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=[
                {
                    "experience_type": "worth_avoiding",
                    "title": "t",
                    "evidence": {"quote": quote, "signal_type": "user_critique"},
                }
            ],
            metrics=metrics,
        )
        evidence = payloads[0]["evidence"]
        assert len(evidence) <= SKILL_EXPERIENCE_MAX_EVIDENCE_LEN
        parsed = json.loads(evidence)
        assert parsed["signal_type"] == "user_critique"
        assert "…[truncated " in parsed["quote"]
        assert parsed["quote"].startswith('say "hi"')
        assert metrics["fields_truncated"] == 1
        assert metrics["field_chars_truncated"] > 0

    async def test_evidence_normalized_when_missing_or_bad(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {"experience_type": "worth_learning", "title": "a"},  # no evidence
            {
                "experience_type": "worth_learning",
                "title": "b",
                "evidence": {"quote": "q", "signal_type": "bogus_signal"},
            },
            {
                "experience_type": "worth_learning",
                "title": "c",
                "evidence": "a raw string",
            },
        ]
        await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        for rec in mgr.created:
            ev = json.loads(rec.evidence)
            assert "quote" in ev and "signal_type" in ev
        # missing -> inferred
        assert json.loads(mgr.created[0].evidence)["signal_type"] == "inferred"
        # bad signal_type -> inferred
        assert json.loads(mgr.created[1].evidence)["signal_type"] == "inferred"
        # raw string -> inferred, quote preserved
        ev2 = json.loads(mgr.created[2].evidence)
        assert ev2["signal_type"] == "inferred"
        assert ev2["quote"] == "a raw string"

    async def test_one_bad_row_does_not_drop_the_rest(self):
        # A manager that raises on a specific title must not abort the batch.
        class _PartlyFailing(_RecordingManager):
            async def create_experience(self, **kwargs):
                if kwargs.get("title") == "boom":
                    raise RuntimeError("db blew up")
                return await super().create_experience(**kwargs)

        mgr = _PartlyFailing()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {"experience_type": "worth_learning", "title": "boom"},
            {"experience_type": "worth_learning", "title": "survivor"},
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_id="s",
            parsed=parsed,
        )
        assert [r.title for r in out] == ["survivor"]


# ============== Single session -> MULTIPLE mixed experiences (fake LLM) =====


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeLLMClient:
    """Returns a FIXED reply regardless of input — no network."""

    def __init__(self, reply: str):
        self._reply = reply
        self.calls = 0

    async def send_llm_request(self, *, messages):
        self.calls += 1
        return _FakeResponse(self._reply)


MIXED_REPLY = json.dumps(
    [
        {
            "experience_type": "worth_learning",
            "title": "Batch independent tool calls",
            "content": "When calls are independent, issue them together.",
            "importance": 0.7,
            "credibility": 0.85,
            "evidence": {
                "quote": "perfect, much faster",
                "signal_type": "user_confirmation",
            },
        },
        {
            "experience_type": "worth_avoiding",
            "title": "Do not assume the column exists",
            "content": "A query failed on a missing column; check schema first.",
            "importance": 0.9,
            "credibility": 0.95,
            "evidence": {"quote": "no such column: foo", "signal_type": "tool_error"},
        },
        {
            "experience_type": "worth_avoiding",
            "title": "Avoid wholesale rewrites",
            "content": "User said the rewrite lost context; prefer deltas.",
            "importance": 0.6,
            "credibility": 0.8,
            "evidence": {
                "quote": "this part isn't good enough",
                "signal_type": "user_critique",
            },
        },
    ]
)


@pytest.mark.asyncio
class TestSingleSessionMultipleMixed:
    async def test_one_session_yields_multiple_mixed_kinds(self):
        mgr = _RecordingManager()
        fake = _FakeLLMClient(MIXED_REPLY)
        conv = _RecordingConversationManager(
            turns_by_session={
                "sess-x": [
                    _conv_turn("user", "...", session_id="sess-x"),
                    _conv_turn("assistant", "...", session_id="sess-x"),
                ]
            }
        )
        d = SessionExperienceDistiller(
            llm_client=fake, experience_manager=mgr, conversation_manager=conv
        )
        meta, user, actor = _meta_user_actor()

        out, processed = await d.distill_sessions(
            meta_agent_state=meta, user=user, actor=actor, session_ids=["sess-x"]
        )
        assert processed == ["sess-x"]
        assert fake.calls == 1
        types = sorted(r.experience_type for r in out)
        # ONE session -> MULTIPLE experiences, a MIX of both kinds.
        assert types == ["worth_avoiding", "worth_avoiding", "worth_learning"]
        # No external success/failure label was needed or used.
        assert len(out) == 3


# ===================== Prioritization ordering (priority = imp*cred) ========


class TestCurationOrder:
    """The curator takes experiences in the order the manager hands them —
    oldest first, ``(created_at, id)`` (the DB-backed proof is
    test_skill_experience.py::TestDistillThenCurate) — and its batch planner and
    payload builder PRESERVE that order: neither re-sorts by priority nor by
    title."""

    def test_the_batch_payload_preserves_the_handed_order(self):
        import json as _json

        from mirix.services.context_budget import TokenCounter
        from mirix.services.skill_experience_curator import (
            build_batch_payload,
            plan_batches,
        )

        # The handed (oldest-first) order is neither the priority order
        # (Alpha > Zeta > Mu by importance*credibility) nor the lexical one.
        handed = [
            SimpleNamespace(
                id=f"sexp-{i}",
                experience_type=etype,
                title=title,
                content="c",
                importance=imp,
                credibility=cred,
                evidence=_json.dumps({"quote": "", "signal_type": "inferred"}),
                curation_attempts=0,
                influenced_skill_ids=None,
            )
            for i, (title, etype, imp, cred) in enumerate(
                [
                    ("Zeta", "worth_avoiding", 0.8, 0.5),
                    ("Mu", "worth_learning", 0.5, 0.6),
                    ("Alpha", "worth_avoiding", 0.9, 0.9),
                ]
            )
        ]
        [batch] = plan_batches(handed, capacity_tokens=100_000, counter=TokenCounter())
        assert batch.experience_ids == ["sexp-0", "sexp-1", "sexp-2"]
        payload = build_batch_payload(
            batch.blocks, batch_index=0, batch_count=1, budget=3, deletes_left=1
        )
        assert payload.index("Zeta") < payload.index("Mu") < payload.index("Alpha")


# ===================== Benchmark-decoupling assertion =======================


class TestNoBenchmarkVocabulary:
    PROMPT = Path("mirix/prompts/system/base/auto_dream_agent/procedural.txt")

    def _prompt_text(self) -> str:
        return self.PROMPT.read_text(encoding="utf-8")

    def test_prompt_has_no_metaclaw_vocabulary(self):
        text = self._prompt_text().lower()
        for banned in [
            "metaclaw",
            "previous feedback",
            "\\bbox",
            "bbox",
            "round_id",
            "round_index",
            "oracle",
            "quality_score",
            "openclaw",
        ]:
            assert banned not in text, f"benchmark vocab leaked into prompt: {banned!r}"

    def test_prompt_states_no_external_grader(self):
        text = self._prompt_text().lower()
        assert "no external grader" in text or "no external" in text

    def test_prompt_has_the_distiller_state_the_outcome_itself(self):
        # There is no external label: the distiller itself states, from the
        # transcript alone, whether the session succeeded, failed or is
        # unclear — before its lessons (tests/test_distiller_outcome.py).
        text = self._prompt_text().lower()
        assert "do not label" not in text
        assert "from the transcript alone" in text
        assert text.index("step 1 — the session's outcome") < text.index(
            "step 3 — the experiences"
        )

    def test_prompt_uses_our_experience_kinds(self):
        text = self._prompt_text()
        assert "worth_learning" in text
        assert "worth_avoiding" in text

    def test_distiller_module_has_no_metaclaw_terms(self):
        src = inspect.getsource(SessionExperienceDistiller)
        low = src.lower()
        for banned in ["metaclaw", "round_index", "quality_score", "oracle"]:
            assert banned not in low, (
                f"benchmark term leaked into distiller: {banned!r}"
            )


# ===== Source isolation — the distiller reads ONLY the Conversation Message Store =
# The scaffolding-filter heuristic (_is_mirix_scaffolding) is DELETED. Correctness
# now comes from STRUCTURE: the distiller's transcript source is the dedicated
# Conversation Message Store — external turns with their REAL user/assistant roles
# — never the meta agent's own `messages` thread. So MIRIX's memory-management
# scaffolding (the "[System Message] As the meta memory manager…" instruction, its
# trigger_memory_update tool calls + results, the continue_chaining control
# replies) can NEVER reach the distiller: those rows simply do not live in this
# store. These tests pin that structural isolation rather than a string heuristic.


def _conv_turn(role, content, *, session_id="sess-1"):
    """A ConversationMessage-shaped row (real role + plain-text content), the
    exact shape ConversationMessageManager.list_turns_for_session returns."""
    return SimpleNamespace(
        role=role, content=content, session_id=session_id, distilled_at=None
    )


class _RecordingConversationManager:
    """In-memory stand-in for ConversationMessageManager.

    Records every call so a test can prove the distiller pulled transcripts from
    THIS store (and, by construction, never from MessageManager). Returns the
    canned turns for a session; everything else is a no-op the distiller needs.
    """

    def __init__(self, turns_by_session=None):
        self._turns_by_session = turns_by_session or {}
        self.list_turns_calls = []
        self.list_turns_kwargs = []

    async def list_turns_for_session(
        self,
        *,
        session_id,
        user_id,
        organization_id,
        actor,
        watermark=None,
        undistilled_only=False,
    ):
        self.list_turns_calls.append(session_id)
        self.list_turns_kwargs.append(
            {"watermark": watermark, "undistilled_only": undistilled_only}
        )
        return list(self._turns_by_session.get(session_id, []))


def _render(turns) -> str:
    """The whole-session user message the distiller sends, as text."""
    from mirix.services.context_budget import TokenCounter
    from mirix.services.distill_transcript import TranscriptPlanner, build_messages

    counter = TokenCounter()
    messages, unavailable = build_messages(turns, counter=counter)
    planner = TranscriptPlanner(
        messages,
        agent_id="agent-meta-1",
        session_id="sess-1",
        counter=counter,
        images_unavailable=unavailable,
    )
    return planner.whole_chunk().text()


def _transcript_lines(text: str):
    body = text.split("session_transcript (chronological):\n", 1)[1]
    body = body.rsplit("\n\nDistill the transferable", 1)[0]
    return [line for line in body.splitlines() if line.strip()]


class TestRenderTranscriptFromConversationStore:
    """The distiller renders ConversationMessage rows with REAL roles as
    ``role: text`` lines — no scaffolding shapes, no character caps."""

    def test_real_roles_are_preserved(self):
        turns = [
            _conv_turn("user", "Q: which HTTP status means 'Not Found'?"),
            _conv_turn("assistant", "404"),
        ]
        # Exactly the two real turns, with their real roles — not [USER]/[ASSISTANT].
        assert _transcript_lines(_render(turns)) == [
            "user: Q: which HTTP status means 'Not Found'?",
            "assistant: 404",
        ]

    def test_no_scaffolding_filtering_remains(self):
        # The store never holds scaffolding, so the distiller does NOT strip it.
        # A row whose content merely RESEMBLES old scaffolding is rendered verbatim
        # (proving the heuristic is gone — the source is trusted by structure).
        turns = [
            _conv_turn(
                "user",
                "[System Message] As the meta memory manager, analyze the content.",
            ),
        ]
        assert "As the meta memory manager" in _render(turns)

    def test_empty_content_turn_is_dropped(self):
        turns = [_conv_turn("user", ""), _conv_turn("assistant", "real answer")]
        assert _transcript_lines(_render(turns)) == ["assistant: real answer"]

    def test_a_long_turn_is_not_cut(self):
        # The old 8,000-char per-turn cap is gone.
        huge = "x" * 50_000
        lines = _transcript_lines(_render([_conv_turn("tool", huge)]))
        assert lines == ["tool: " + huge]

    def test_a_long_session_is_not_elided(self):
        # The old 80,000-char head + tail elision is gone: a session that
        # fits the context budget is sent whole.
        turns = [
            _conv_turn("user", f"turn-{i:04d} " + "y" * 7_900) for i in range(40)
        ]
        text = _render(turns)
        assert "elided" not in text and "truncated" not in text
        assert all(f"turn-{i:04d}" in text for i in range(40))

    def test_header_metadata_precedes_the_transcript(self):
        text = _render(
            [
                _conv_turn("user", "deploy"),
                _conv_turn("assistant", "[tool_call] kubectl(apply)"),
                _conv_turn("tool", "[kubectl] error: forbidden"),
                _conv_turn("assistant", "fixed the RBAC and retried"),
            ]
        )
        header = text.split("session_transcript", 1)[0]
        assert "agent_id: agent-meta-1" in header and "session_id: sess-1" in header
        assert "4 message(s)" in header and "2 round(s)" in header
        assert "tools used: kubectl" in header and "tool errors: 1" in header

    def test_truncation_constants_are_deleted(self):
        import mirix.constants as constants
        from mirix.services import session_experience_distiller as sed

        for name in ("DISTILLER_MAX_TRANSCRIPT_CHARS", "DISTILLER_MAX_MESSAGE_CHARS"):
            assert not hasattr(constants, name)
        assert not hasattr(sed, "_MAX_TRANSCRIPT_CHARS")
        assert not hasattr(sed, "_MAX_MESSAGE_CHARS")
        assert not hasattr(SessionExperienceDistiller, "_render_transcript")

    def test_scaffolding_heuristic_is_deleted(self):
        # The fragile string heuristic must be GONE — correctness is structural.
        assert not hasattr(SessionExperienceDistiller, "_is_mirix_scaffolding")

    def test_distiller_module_does_not_reference_scaffolding_terms(self):
        src = inspect.getsource(SessionExperienceDistiller)
        for banned in (
            "_is_mirix_scaffolding",
            "trigger_memory_update",
            "META_MEMORY_TOOLS",
        ):
            assert banned not in src, (
                f"dead scaffolding reference left behind: {banned!r}"
            )


@pytest.mark.asyncio
class TestDistillerReadsConversationStoreNotMetaMessages:
    """The load-bearing isolation: the distiller's per-session transcript fetch
    goes through the injected ConversationMessageManager, NEVER through the meta
    agent's `messages` store (MessageManager)."""

    async def test_distill_one_pulls_turns_from_conversation_store(self, monkeypatch):
        # Hard guard: if the distiller ever touches MessageManager, blow up.
        import mirix.services.message_manager as mm

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError(
                "distiller must NOT read meta agent messages via MessageManager"
            )

        monkeypatch.setattr(mm.MessageManager, "list_messages_for_agent", _boom)

        conv = _RecordingConversationManager(
            turns_by_session={
                "sess-x": [
                    _conv_turn("user", "great, that worked", session_id="sess-x"),
                    _conv_turn("assistant", "glad to help", session_id="sess-x"),
                ]
            }
        )
        exp_mgr = _RecordingManager()
        fake = _FakeLLMClient(MIXED_REPLY)
        d = SessionExperienceDistiller(
            llm_client=fake, experience_manager=exp_mgr, conversation_manager=conv
        )
        meta, user, actor = _meta_user_actor()

        out, _ = await d.distill_sessions(
            meta_agent_state=meta, user=user, actor=actor, session_ids=["sess-x"]
        )
        # The transcript was fetched from the Conversation Message Store…
        assert conv.list_turns_calls == ["sess-x"]
        # …the LLM was called once, and experiences were produced.
        assert fake.calls == 1
        assert len(out) == 3

    async def test_empty_session_in_store_skips_llm(self):
        conv = _RecordingConversationManager(turns_by_session={"empty": []})
        fake = _FakeLLMClient(MIXED_REPLY)
        d = SessionExperienceDistiller(
            llm_client=fake,
            experience_manager=_RecordingManager(),
            conversation_manager=conv,
        )
        meta, user, actor = _meta_user_actor()
        out, processed = await d.distill_sessions(
            meta_agent_state=meta, user=user, actor=actor, session_ids=["empty"]
        )
        assert out == [] and processed == ["empty"]
        # No turns → no LLM call (the source store is the only thing consulted).
        assert conv.list_turns_calls == ["empty"]
        assert fake.calls == 0

    async def test_reads_only_undistilled_rows_up_to_the_watermark(self):
        """A run reads a session's not-yet-distilled rows at or before its
        claim-time watermark, and reports per-session metrics."""
        import datetime as dt

        conv = _RecordingConversationManager(
            turns_by_session={
                "s1": [
                    _conv_turn("user", "try again", session_id="s1"),
                    _conv_turn("assistant", "fixed it", session_id="s1"),
                ],
                "s2": [],
            }
        )
        d = SessionExperienceDistiller(
            llm_client=_FakeLLMClient(MIXED_REPLY),
            experience_manager=_RecordingManager(),
            conversation_manager=conv,
        )
        meta, user, actor = _meta_user_actor()
        wm = dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.timezone.utc)
        metrics = {}
        experiences, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["s1", "s2"],
            session_watermarks={"s1": wm},
            metrics=metrics,
        )
        by_session = dict(zip(conv.list_turns_calls, conv.list_turns_kwargs))
        assert by_session["s1"] == {"watermark": wm, "undistilled_only": True}
        assert by_session["s2"] == {"watermark": None, "undistilled_only": True}
        assert processed == ["s1", "s2"]
        assert len(experiences) == 3
        assert {k: metrics["s1"][k] for k in ("session_id", "messages", "status", "experiences")} == {
            "session_id": "s1",
            "messages": 2,
            "status": "ok",
            "experiences": 3,
        }
        assert metrics["s2"]["messages"] == 0 and metrics["s2"]["status"] == "ok"
        assert metrics["s2"]["llm_calls"] == 0

    async def test_failed_session_is_reported_in_metrics_and_not_processed(self):
        conv = _RecordingConversationManager(
            turns_by_session={"bad": [_conv_turn("user", "hello", session_id="bad")]}
        )

        class _Boom:
            calls = 0

            async def send_llm_request(self, messages):
                raise RuntimeError("provider down")

        d = SessionExperienceDistiller(
            llm_client=_Boom(),
            experience_manager=_RecordingManager(),
            conversation_manager=conv,
        )
        meta, user, actor = _meta_user_actor()
        metrics = {}
        experiences, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["bad"],
            metrics=metrics,
        )
        assert experiences == [] and processed == []
        assert metrics["bad"]["status"] == "failed"
        assert "provider down" in metrics["bad"]["error"]
        assert "config_deferral" not in metrics["bad"]

    async def test_a_distiller_without_an_llm_names_its_configuration_deferral(self):
        """A session deferred because the distiller has no LLM client
        or config is deferred the same way on every run. Its metrics name that
        configuration (``config_deferral``), so a reader tells it from a
        misconfigured deployment's 401/403/404 (``failure_kind`` config too),
        which a fix of the deployment ends."""
        from mirix.services.session_experience_distiller import (
            DISTILL_CONFIGURATION_DEFERRALS,
            DISTILL_DEFERRAL_NO_LLM_CLIENT,
        )

        conv = _RecordingConversationManager(
            turns_by_session={"s": [_conv_turn("user", "hello", session_id="s")]}
        )
        d = SessionExperienceDistiller(
            experience_manager=_RecordingManager(), conversation_manager=conv
        )
        meta, user, actor = _meta_user_actor()
        metrics = {}
        experiences, processed = await d.distill_sessions(
            meta_agent_state=meta,
            user=user,
            actor=actor,
            session_ids=["s"],
            metrics=metrics,
        )
        assert experiences == [] and processed == []
        entry = metrics["s"]
        assert (entry["status"], entry["failure_kind"]) == ("deferred", "config")
        assert entry["config_deferral"] == DISTILL_DEFERRAL_NO_LLM_CLIENT
        assert DISTILL_DEFERRAL_NO_LLM_CLIENT in DISTILL_CONFIGURATION_DEFERRALS


class TestDistillerSourceIsConversationStore:
    """Static guard that the distiller's source-of-truth is the Conversation
    Message Store and that it no longer enumerates the meta agent's messages."""

    def test_distiller_fetches_via_conversation_manager(self):
        src = inspect.getsource(SessionExperienceDistiller._distill_one)
        assert "list_turns_for_session" in src
        # Must NOT read the meta agent's message thread for distillation.
        assert "list_messages_for_agent" not in src

    def test_session_enumeration_is_gone(self):
        # Which sessions are distilled is the learning run's claim; the
        # distiller no longer enumerates "sealed" sessions itself.
        assert not hasattr(SessionExperienceDistiller, "enumerate_sealed_sessions")
        src = inspect.getsource(inspect.getmodule(SessionExperienceDistiller))
        assert "list_sealed_undistilled_sessions" not in src
        assert "sealed" not in src.lower()

    def test_distiller_constructor_injects_conversation_manager(self):
        sig = inspect.signature(SessionExperienceDistiller.__init__)
        assert "conversation_manager" in sig.parameters

    def test_distiller_does_not_import_message_manager(self):
        # The distiller's source must not pull in the meta-agent MessageManager,
        # nor query the meta `messages` store — that store is structurally out of
        # reach. (`ConversationMessageManager` legitimately contains the substring
        # "MessageManager", so we match the precise import / query forms instead.)
        src = inspect.getsource(inspect.getmodule(SessionExperienceDistiller))
        assert "from mirix.services.message_manager import" not in src
        assert "list_messages_for_agent" not in src
        # …and it DOES use the Conversation Message Store.
        assert "ConversationMessageManager" in src


@pytest.mark.asyncio
async def test_distillation_concurrency_is_capped():
    """A run claims every pending session, so the per-session fan-out
    is bounded by MIRIX_DISTILLER_MAX_CONCURRENCY."""
    import asyncio

    state = {"live": 0, "peak": 0}

    class _SlowLLM:
        async def send_llm_request(self, messages):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            await asyncio.sleep(0.01)
            state["live"] -= 1
            return _FakeResponse("[]")

    conv = _RecordingConversationManager(
        turns_by_session={
            f"s{i}": [_conv_turn("user", "hi", session_id=f"s{i}")] for i in range(9)
        }
    )
    d = SessionExperienceDistiller(
        llm_client=_SlowLLM(),
        experience_manager=_RecordingManager(),
        conversation_manager=conv,
        max_concurrency=2,
    )
    meta, user, actor = _meta_user_actor()
    _, processed = await d.distill_sessions(
        meta_agent_state=meta,
        user=user,
        actor=actor,
        session_ids=[f"s{i}" for i in range(9)],
    )
    assert len(processed) == 9
    assert state["peak"] == 2
