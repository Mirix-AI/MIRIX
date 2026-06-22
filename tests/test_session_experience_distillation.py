"""Goal-2 tests — general per-session experience distillation (DB-free).

Mirrors the layering of test_skill_session_distiller.py but for the GENERAL
Experience Distiller (no MetaClaw oracle, no success/failure label):

1. JSON-array parse robustness — bare array, fenced ```json, prose-wrapped,
   single-object fallback, empty `[]`, garbage -> [].
2. Parsed-experience -> SkillExperience mapping (_persist_experiences via a stub
   manager, no DB): importance/credibility clamping, evidence normalization,
   bad-type / empty-title skips, length caps, status='pending'.
3. Prioritization ordering invariant — importance*credibility is the priority
   the curator payload is built around.
4. Single session yields MULTIPLE mixed worth_learning + worth_avoiding from one
   fixed fake LLM reply (no external label needed).
5. De-MetaClaw HARD assertion — the distiller prompt/flow carries NO MetaClaw
   vocabulary and needs no external success/failure oracle.

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

from mirix.schemas.skill_experience import (
    SKILL_EXPERIENCE_MAX_CONTENT_LEN,
    SKILL_EXPERIENCE_MAX_TITLE_LEN,
)
from mirix.services.session_experience_distiller import SessionExperienceDistiller
from mirix.services.skill_session_distiller import parse_distiller_json_array


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
        raw = "```json\n[{\"experience_type\":\"worth_avoiding\",\"title\":\"x\"}]\n```"
        out = parse_distiller_json_array(raw)
        assert len(out) == 1
        assert out[0]["experience_type"] == "worth_avoiding"

    def test_plain_fence(self):
        raw = "```\n[{\"title\":\"a\"},{\"title\":\"b\"}]\n```"
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

        validated = SkillExperienceCreate(**kwargs)
        rec = SimpleNamespace(**validated.model_dump())
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
                "evidence": {"quote": "great, that worked", "signal_type": "user_confirmation"},
            }
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="sess-1", parsed=parsed,
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
            {"experience_type": "worth_avoiding", "title": "t1",
             "importance": 5.0, "credibility": -3.0},
            {"experience_type": "worth_learning", "title": "t2",
             "importance": "garbage", "credibility": None},
        ]
        await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
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
            {"experience_type": "partial", "title": "bad"},          # bad enum
            {"experience_type": "worth_learning", "title": "good"},  # kept
            {"experience_type": None, "title": "alsobad"},           # missing
        ]
        out = await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
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
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
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
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
        )
        assert len(out) == 1
        assert len(mgr.created[0].title) <= SKILL_EXPERIENCE_MAX_TITLE_LEN
        assert len(mgr.created[0].content) <= SKILL_EXPERIENCE_MAX_CONTENT_LEN

    async def test_evidence_normalized_when_missing_or_bad(self):
        mgr = _RecordingManager()
        d = _distiller_with(mgr)
        meta, user, actor = _meta_user_actor()
        parsed = [
            {"experience_type": "worth_learning", "title": "a"},  # no evidence
            {"experience_type": "worth_learning", "title": "b",
             "evidence": {"quote": "q", "signal_type": "bogus_signal"}},
            {"experience_type": "worth_learning", "title": "c",
             "evidence": "a raw string"},
        ]
        await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
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
            meta_agent_state=meta, user=user, actor=actor,
            session_id="s", parsed=parsed,
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


MIXED_REPLY = json.dumps([
    {
        "experience_type": "worth_learning",
        "title": "Batch independent tool calls",
        "content": "When calls are independent, issue them together.",
        "importance": 0.7,
        "credibility": 0.85,
        "evidence": {"quote": "perfect, much faster", "signal_type": "user_confirmation"},
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
        "evidence": {"quote": "this part isn't good enough", "signal_type": "user_critique"},
    },
])


@pytest.mark.asyncio
class TestSingleSessionMultipleMixed:
    async def test_one_session_yields_multiple_mixed_kinds(self):
        mgr = _RecordingManager()
        fake = _FakeLLMClient(MIXED_REPLY)
        d = SessionExperienceDistiller(llm_client=fake, experience_manager=mgr)
        meta, user, actor = _meta_user_actor()

        parsed = await d._call_llm(
            agent_id=meta.id, session_id="sess-x",
            transcript="user: ...\nassistant: ...", skills_block="(none)",
        )
        out = await d._persist_experiences(
            meta_agent_state=meta, user=user, actor=actor,
            session_id="sess-x", parsed=parsed,
        )
        assert fake.calls == 1
        types = sorted(r.experience_type for r in out)
        # ONE session -> MULTIPLE experiences, a MIX of both kinds.
        assert types == ["worth_avoiding", "worth_avoiding", "worth_learning"]
        # No external success/failure label was needed or used.
        assert len(out) == 3


# ===================== Prioritization ordering (priority = imp*cred) ========


class TestPriorityOrdering:
    """The importance*credibility DESC ordering is the load-bearing priority.

    The DB-backed proof that the MANAGER emits this ordering lives in
    test_skill_experience.py::TestCreateAndList::test_list_ordered_by_priority_desc
    (against the real SQL `ORDER BY importance*credibility DESC`). Here we assert
    the downstream guarantee the CURATOR relies on: build_experience_payload
    PRESERVES the order it is handed (so the highest-priority experience leads
    the prompt). This is non-tautological — it pins real curator behavior.
    """

    def test_payload_preserves_handed_priority_order(self):
        import json as _json

        from mirix.services.skill_experience_curator import build_experience_payload

        # Hand the payload builder experiences already in priority order (as the
        # manager would). The first block in the rendered prompt must be the
        # highest-priority one, the last the lowest.
        ordered = [
            SimpleNamespace(
                experience_type="worth_avoiding", title="HIGH",
                content="c", importance=0.8, credibility=0.9,
                evidence=_json.dumps({"quote": "", "signal_type": "inferred"}),
            ),
            SimpleNamespace(
                experience_type="worth_learning", title="MID",
                content="c", importance=0.5, credibility=0.6,
                evidence=_json.dumps({"quote": "", "signal_type": "inferred"}),
            ),
            SimpleNamespace(
                experience_type="worth_avoiding", title="LOW",
                content="c", importance=0.9, credibility=0.1,
                evidence=_json.dumps({"quote": "", "signal_type": "inferred"}),
            ),
        ]
        payload = build_experience_payload(ordered)
        assert payload.index("HIGH") < payload.index("MID") < payload.index("LOW")


# ===================== De-MetaClaw HARD assertion ===========================


class TestNoMetaClawVocabulary:
    PROMPT = Path(
        "mirix/prompts/system/base/auto_dream_agent/procedural.txt"
    )

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
            assert banned not in text, f"MetaClaw vocab leaked into prompt: {banned!r}"

    def test_prompt_states_no_external_grader(self):
        text = self._prompt_text().lower()
        assert "no external grader" in text or "no external" in text

    def test_prompt_does_not_label_session_success_or_failure(self):
        text = self._prompt_text().lower()
        # The prompt must instruct NOT to label the whole session success/failure.
        assert "do not label" in text and "success or a failure" in text

    def test_prompt_uses_our_experience_kinds(self):
        text = self._prompt_text()
        assert "worth_learning" in text
        assert "worth_avoiding" in text

    def test_distiller_module_has_no_metaclaw_terms(self):
        src = inspect.getsource(SessionExperienceDistiller)
        low = src.lower()
        for banned in ["metaclaw", "round_index", "quality_score", "oracle"]:
            assert banned not in low, f"MetaClaw term leaked into distiller: {banned!r}"
