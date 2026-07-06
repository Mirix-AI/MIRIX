"""Tests for C4 — bounded, count-driven, quality-aware edit budget + size/delete
gates on the skill tools.

Three layers, all DB-free / network-free:

1. Pure budget formula (`compute_edit_budget`): count-driven, clamped, B_min=0
   skip, hybrid `min(formula, autonomous)`. quality_score never drives it.
2. Pure size gate (`_edit_exceeds_size_gate`, `_instructions_over_ceiling`):
   char-delta + change-ratio boundaries, instructions ceiling.
3. Budget/size/delete GATE behavior on `skill_create` / `skill_edit` /
   `skill_delete`, exercised against a lightweight fake Agent whose
   `procedural_memory_manager` is a stub. The counter is the per-instance
   attribute `self._edit_budget_remaining`; we assert it is per-instance (no
   cross-agent / cross-user leak) and that an unset attr means "no limit".

The constants live in `mirix.constants` near the existing SKILL_* ones; the
gate logic lives in `mirix.functions.function_sets.memory_tools`.
"""

from __future__ import annotations

import asyncio
import difflib

from mirix.constants import (
    SKILL_DELETE_BUDGET_MAX,
    SKILL_EDIT_BUDGET_ALPHA_FAIL,
    SKILL_EDIT_BUDGET_ALPHA_SUCC,
    SKILL_EDIT_BUDGET_B0,
    SKILL_EDIT_BUDGET_MAX,
    SKILL_EDIT_BUDGET_MIN,
    SKILL_EDIT_MAJOR_RATIO,
    SKILL_MAX_EDIT_CHAR_DELTA,
    SKILL_MAX_INSTRUCTIONS_CHARS,
)
from mirix.functions.function_sets import memory_tools as mt


# ============================ Budget formula ==============================


class TestBudgetConstants:
    def test_constant_values_match_design(self):
        # DESIGN §C4: B0=1, alpha_f=1.0, alpha_s=0.5, B_min=0, B_max=6.
        assert SKILL_EDIT_BUDGET_B0 == 1
        assert SKILL_EDIT_BUDGET_ALPHA_FAIL == 1.0
        assert SKILL_EDIT_BUDGET_ALPHA_SUCC == 0.5
        assert SKILL_EDIT_BUDGET_MIN == 0
        assert SKILL_EDIT_BUDGET_MAX == 6
        assert SKILL_MAX_EDIT_CHAR_DELTA == 800
        assert SKILL_EDIT_MAJOR_RATIO == 0.4
        assert SKILL_MAX_INSTRUCTIONS_CHARS == 12000
        assert SKILL_DELETE_BUDGET_MAX == 1


class TestComputeEditBudget:
    def _agg(self, n_high_fail=0, n_high_succ=0, n=None, mean_q=0.0):
        if n is None:
            n = n_high_fail + n_high_succ
        return {
            "n": n,
            "n_high_fail": n_high_fail,
            "n_high_succ": n_high_succ,
            "mean_q": mean_q,
        }

    def test_all_failures_clamps_to_b_max(self):
        # raw = 1 + 1.0*5 + 0.5*0 = 6 -> clamp(6, 0, 6) = 6
        agg = self._agg(n_high_fail=5)
        assert mt.compute_edit_budget(agg) == 6

    def test_more_failures_still_clamps_to_b_max(self):
        # raw = 1 + 1.0*8 = 9 -> clamp to 6.
        agg = self._agg(n_high_fail=8)
        assert mt.compute_edit_budget(agg) == SKILL_EDIT_BUDGET_MAX

    def test_mixed_window_lands_mid(self):
        # raw = 1 + 1.0*2 + 0.5*2 = 4 -> 4
        agg = self._agg(n_high_fail=2, n_high_succ=2)
        assert mt.compute_edit_budget(agg) == 4

    def test_single_failure(self):
        # raw = 1 + 1.0*1 = 2
        agg = self._agg(n_high_fail=1)
        assert mt.compute_edit_budget(agg) == 2

    def test_successes_only_half_weight(self):
        # raw = 1 + 0.5*3 = 2.5 -> round(2.5) banker's? we want round-half-up=3? -3
        # DESIGN says clamp(round(raw)). Python round(2.5)=2 (banker's). The
        # implementation must use round-half-up so 2.5 -> 3 is deterministic and
        # matches "mixed => 2-4" intuition. Assert the documented value.
        agg = self._agg(n_high_succ=3)
        assert mt.compute_edit_budget(agg) == 3

    def test_noise_window_is_zero(self):
        # No structurally-gated records: n_high_* == 0. Even though B0=1, the
        # B_min=0 *skip* is handled by the curator (no aggregate -> no budget),
        # but compute_edit_budget on an all-noise aggregate must itself yield 0
        # so the curator's "skip when 0" decision keys off the same number.
        agg = self._agg(n_high_fail=0, n_high_succ=0, n=4)
        assert mt.compute_edit_budget(agg) == 0

    def test_quality_score_never_drives_budget(self):
        # Two aggregates with identical counts but wildly different mean_q must
        # produce the SAME budget: quality_score is ranking-only.
        low_q = self._agg(n_high_fail=2, n_high_succ=1, mean_q=0.01)
        high_q = self._agg(n_high_fail=2, n_high_succ=1, mean_q=0.99)
        assert mt.compute_edit_budget(low_q) == mt.compute_edit_budget(high_q)

    def test_hybrid_min_formula_and_autonomous(self):
        agg = self._agg(n_high_fail=5)  # formula = 6
        # autonomous may only REDUCE: final = min(6, 2) = 2.
        assert mt.compute_edit_budget(agg, autonomous=2) == 2
        # autonomous larger than formula cannot raise it: min(6, 10) = 6.
        assert mt.compute_edit_budget(agg, autonomous=10) == 6

    def test_hybrid_autonomous_clamped_to_band(self):
        agg = self._agg(n_high_fail=5)  # formula = 6
        # A negative autonomous is clamped to B_min then min'd: min(6, 0) = 0.
        assert mt.compute_edit_budget(agg, autonomous=-3) == 0


# ============================ Size gate (pure) ============================


class TestEditSizeGate:
    BASE = "x" * 1000

    def test_char_delta_799_allowed(self):
        # |len(new) - len(old)| = 799 < 800 -> allowed (no reason).
        old = self.BASE
        new = self.BASE + ("y" * 799)
        assert abs(len(new) - len(old)) == 799
        # keep ratio low so only the char-delta dimension is under test
        assert mt._edit_exceeds_size_gate(old, new) is None

    def test_char_delta_801_rejected(self):
        old = self.BASE
        new = self.BASE + ("y" * 801)
        assert abs(len(new) - len(old)) == 801
        reason = mt._edit_exceeds_size_gate(old, new)
        assert reason is not None
        assert "too large" in reason.lower()

    def test_char_delta_exactly_800_allowed(self):
        # Boundary: gate rejects only when delta > 800, so 800 is allowed.
        old = self.BASE
        new = self.BASE + ("y" * 800)
        assert abs(len(new) - len(old)) == 800
        assert mt._edit_exceeds_size_gate(old, new) is None

    def test_ratio_039_allowed(self):
        # Same-length old/new (char-delta 0) so ONLY the change-ratio gate is
        # under test. For old="a"*L, new="a"*k+"b"*(L-k), SequenceMatcher finds
        # the common run of `a`s (length k), so ratio = k/L and the change-ratio
        # is 1 - k/L. L=1000, k=610 -> change-ratio = 0.39 (< 0.40).
        old = "a" * 1000
        new = ("a" * 610) + ("b" * 390)
        ratio_change = 1 - difflib.SequenceMatcher(None, old, new).ratio()
        assert ratio_change < SKILL_EDIT_MAJOR_RATIO, ratio_change
        assert mt._edit_exceeds_size_gate(old, new) is None

    def test_ratio_041_rejected(self):
        # k=590 -> change-ratio = 0.41 (>= 0.40), so the gate trips.
        old = "a" * 1000
        new = ("a" * 590) + ("b" * 410)
        ratio_change = 1 - difflib.SequenceMatcher(None, old, new).ratio()
        assert ratio_change >= SKILL_EDIT_MAJOR_RATIO, ratio_change
        reason = mt._edit_exceeds_size_gate(old, new)
        assert reason is not None
        assert "too large" in reason.lower()

    def test_instructions_ceiling(self):
        under = "z" * (SKILL_MAX_INSTRUCTIONS_CHARS - 1)
        over = "z" * (SKILL_MAX_INSTRUCTIONS_CHARS + 1)
        assert mt._instructions_over_ceiling(under) is False
        assert mt._instructions_over_ceiling(over) is True


# ============================ Fake Agent harness ==========================


class _FakeSkill:
    def __init__(
        self,
        skill_id,
        name,
        instructions="short instructions",
        version="0.1.0",
        description="d",
        entry_type="guide",
    ):
        self.id = skill_id
        self.name = name
        self.instructions = instructions
        self.version = version
        self.description = description
        self.entry_type = entry_type
        self.triggers = []
        self.examples = []


class _FakeProcManager:
    """Stub manager: records mutations, never touches a DB or embeddings."""

    def __init__(self, skills=None):
        self._skills = {s.id: s for s in (skills or [])}
        self.created = []
        self.updated = []
        self.deleted = []

    async def list_procedures(self, **kwargs):
        # Used by skill_create's name-dedup pre-check. Return empty (no dup).
        return []

    async def insert_procedure(self, **kwargs):
        self.created.append(kwargs)
        sid = f"proc-new-{len(self.created)}"
        return _FakeSkill(sid, kwargs["name"], kwargs.get("instructions", ""))

    async def get_item_by_id(self, item_id, **kwargs):
        if item_id not in self._skills:
            raise KeyError(item_id)
        return self._skills[item_id]

    async def update_item(self, item_update, **kwargs):
        data = item_update.model_dump(exclude_unset=True)
        self.updated.append(data)
        skill = self._skills[data["id"]]
        for k, v in data.items():
            if k != "id":
                setattr(skill, k, v)
        return skill

    async def delete_procedure_by_id(self, procedure_id, **kwargs):
        self.deleted.append(procedure_id)
        self._skills.pop(procedure_id, None)


class _FakeUser:
    id = "user-1"
    organization_id = "org-1"


class _FakeActor:
    id = "client-1"
    organization_id = "org-1"


class _FakeAgentState:
    id = "agent-1"
    parent_id = None


class _FakeAgent:
    """Minimal `self` for the skill tools. Only the attributes the tools read."""

    def __init__(self, manager):
        self.procedural_memory_manager = manager
        self.user = _FakeUser()
        self.actor = _FakeActor()
        self.agent_state = _FakeAgentState()
        self.filter_tags = None
        self.use_cache = True
        self.user_id = "user-1"


def _run(coro):
    return asyncio.run(coro)


# ============================ Budget gate ================================


class TestBudgetGateExhaustion:
    def test_create_decrements_and_exhausts(self):
        agent = _FakeAgent(_FakeProcManager())
        agent._edit_budget_remaining = 2

        r1 = _run(mt.skill_create(agent, "skill-a", "d", "i", "guide"))
        assert "created" in r1.lower()
        assert agent._edit_budget_remaining == 1

        r2 = _run(mt.skill_create(agent, "skill-b", "d", "i", "guide"))
        assert "created" in r2.lower()
        assert agent._edit_budget_remaining == 0

        # Third create: budget exhausted -> no mutation, advisory message.
        r3 = _run(mt.skill_create(agent, "skill-c", "d", "i", "guide"))
        assert "exhausted" in r3.lower()
        assert "finish_memory_update" in r3
        assert len(agent.procedural_memory_manager.created) == 2  # only 2 created
        assert agent._edit_budget_remaining == 0

    def test_edit_decrements_same_counter_as_create(self):
        skill = _FakeSkill("proc-1", "skill-x", "the quick brown fox")
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 1

        # One edit consumes the only unit.
        r1 = _run(
            mt.skill_edit(
                agent, "proc-1", "instructions", old_text="quick", new_text="slow"
            )
        )
        assert "updated" in r1.lower()
        assert agent._edit_budget_remaining == 0

        # Now a CREATE must also see the shared budget exhausted.
        r2 = _run(mt.skill_create(agent, "skill-y", "d", "i", "guide"))
        assert "exhausted" in r2.lower()
        assert len(agent.procedural_memory_manager.created) == 0

    def test_exhausted_edit_does_not_mutate(self):
        skill = _FakeSkill("proc-1", "skill-x", "the quick brown fox")
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 0

        r = _run(
            mt.skill_edit(
                agent, "proc-1", "instructions", old_text="quick", new_text="slow"
            )
        )
        assert "exhausted" in r.lower()
        assert agent.procedural_memory_manager.updated == []

    def test_unset_budget_means_no_limit(self):
        agent = _FakeAgent(_FakeProcManager())
        # No _edit_budget_remaining attribute set at all.
        assert not hasattr(agent, "_edit_budget_remaining")
        for i in range(10):
            r = _run(mt.skill_create(agent, f"skill-{i}", "d", "i", "guide"))
            assert "created" in r.lower()
        assert len(agent.procedural_memory_manager.created) == 10


class TestBudgetCounterIsPerInstance:
    def test_two_agents_do_not_share_budget(self):
        a = _FakeAgent(_FakeProcManager())
        b = _FakeAgent(_FakeProcManager())
        a._edit_budget_remaining = 1
        b._edit_budget_remaining = 1

        # Exhaust agent a.
        _run(mt.skill_create(a, "skill-a1", "d", "i", "guide"))
        assert a._edit_budget_remaining == 0
        r = _run(mt.skill_create(a, "skill-a2", "d", "i", "guide"))
        assert "exhausted" in r.lower()

        # agent b is untouched: still has its own full budget.
        assert b._edit_budget_remaining == 1
        r = _run(mt.skill_create(b, "skill-b1", "d", "i", "guide"))
        assert "created" in r.lower()
        assert b._edit_budget_remaining == 0


# ============================ Size gate on edit =========================


class TestEditSizeGateOnTool:
    def test_oversized_edit_rejected_without_consuming_budget(self):
        big = "a" * 1000
        skill = _FakeSkill("proc-1", "skill-x", big)
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 3

        # new_text differs by > 800 chars from old_text.
        new = big + ("b" * 900)
        r = _run(
            mt.skill_edit(agent, "proc-1", "instructions", old_text=big, new_text=new)
        )
        assert "too large" in r.lower()
        # No mutation, and budget NOT consumed (rejection is free).
        assert agent.procedural_memory_manager.updated == []
        assert agent._edit_budget_remaining == 3

    def test_small_edit_passes_size_gate_and_consumes_budget(self):
        skill = _FakeSkill("proc-1", "skill-x", "the quick brown fox jumps")
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 3
        r = _run(
            mt.skill_edit(
                agent, "proc-1", "instructions", old_text="quick", new_text="speedy"
            )
        )
        assert "updated" in r.lower()
        assert agent._edit_budget_remaining == 2

    def test_over_ceiling_instructions_routes_to_create(self):
        # Editing instructions so the *resulting* text exceeds the hard ceiling
        # must be refused with a "use skill_create" style message and not mutate.
        base = "a" * (SKILL_MAX_INSTRUCTIONS_CHARS - 100)
        skill = _FakeSkill("proc-1", "skill-x", base)
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 3
        # Replace a short anchor with a huge block that pushes over the ceiling.
        anchor = base[:50]
        huge = anchor + ("b" * 500)  # net +450, but total > ceiling
        # Make the total exceed the ceiling regardless of size-gate by appending.
        skill.instructions = base + ("a" * 200)  # now total ~ ceiling+100
        r = _run(
            mt.skill_edit(
                agent, "proc-1", "instructions", old_text=anchor, new_text=huge
            )
        )
        reason = r.lower()
        assert (
            ("ceiling" in reason)
            or ("skill_create" in reason)
            or ("too large" in reason)
        )
        assert agent.procedural_memory_manager.updated == []
        assert agent._edit_budget_remaining == 3

    def test_size_gate_only_applies_to_text_fields(self):
        # A triggers (non-text) edit must not be size-gated.
        skill = _FakeSkill("proc-1", "skill-x", "i")
        agent = _FakeAgent(_FakeProcManager([skill]))
        agent._edit_budget_remaining = 3
        r = _run(mt.skill_edit(agent, "proc-1", "triggers", value='["a", "b"]'))
        assert "updated" in r.lower()
        assert agent._edit_budget_remaining == 2


# ============================ Delete gate ===============================


class _FakeSoftDeleteManager(_FakeProcManager):
    """Adds a soft-delete (exclude-from-retrieval) path the gate can prefer."""

    def __init__(self, skills=None):
        super().__init__(skills)
        self.soft_deleted = []

    async def soft_delete_procedure_by_id(self, procedure_id, **kwargs):
        # Soft delete = excluded from retrieval but row retained.
        self.soft_deleted.append(procedure_id)
        skill = self._skills.get(procedure_id)
        if skill is not None:
            skill.is_deleted = True

    async def list_procedures(self, **kwargs):
        # Retrieval excludes soft-deleted skills.
        return [s for s in self._skills.values() if not getattr(s, "is_deleted", False)]


class TestDeleteAuthorizationHelper:
    def test_record_names_skill_harmful_by_name(self):
        rec = {
            "detail": "root_cause: the deploy-prod skill is actively harmful, "
            "it deletes the wrong namespace",
            "record_type": "failure",
        }
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is True
        )

    def test_record_names_skill_harmful_by_id(self):
        rec = {
            "detail": "proc-9 caused the failure; it is harmful and should be removed",
            "record_type": "failure",
        }
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is True
        )

    def test_record_merely_redundant_does_not_authorize(self):
        rec = {
            "detail": "the deploy-prod skill is redundant with deploy-staging",
            "record_type": "failure",
        }
        # "redundant" is NOT "actively harmful" -> no delete authorization.
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is False
        )

    def test_success_record_never_authorizes_delete(self):
        rec = {"detail": "the deploy-prod skill is harmful", "record_type": "success"}
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is False
        )

    def test_harmful_marker_must_co_occur_with_named_skill(self):
        # The harmful marker and the skill mention are in DIFFERENT clauses:
        # "deploy-prod is redundant; deploy-staging was harmful". This must NOT
        # authorize deleting deploy-prod (the harmful marker refers to a DIFFERENT
        # skill). Proximity gate: marker + name must share a clause/sentence.
        rec = {
            "detail": "the deploy-prod skill is redundant; "
            "the deploy-staging skill was harmful and caused the failure",
            "record_type": "failure",
        }
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is False
        )
        # But deleting deploy-staging (the one the marker DOES refer to) is ok.
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-staging", skill_id="proc-10"
            )
            is True
        )

    def test_id_match_is_whole_token_not_prefix(self):
        # A record naming 'proc-90' harmful must NOT authorize deleting 'proc-9'
        # (substring/prefix cross-authorization guard).
        rec = {
            "detail": "proc-90 is actively harmful and caused the failure",
            "record_type": "failure",
        }
        assert (
            mt._record_authorizes_delete(rec, skill_name="other", skill_id="proc-9")
            is False
        )
        # The exact id still authorizes.
        assert (
            mt._record_authorizes_delete(rec, skill_name="other", skill_id="proc-90")
            is True
        )

    def test_record_not_naming_skill_does_not_authorize(self):
        rec = {"detail": "some other skill is harmful", "record_type": "failure"}
        assert (
            mt._record_authorizes_delete(
                rec, skill_name="deploy-prod", skill_id="proc-9"
            )
            is False
        )


class TestDeleteGateOnTool:
    def test_delete_rejected_without_authorization(self):
        skill = _FakeSkill("proc-1", "skill-x", "i")
        agent = _FakeAgent(_FakeProcManager([skill]))
        # No _delete_authorized_skill_ids set -> nothing is authorized.
        agent._delete_budget_remaining = 1
        r = _run(mt.skill_delete(agent, "proc-1"))
        assert "not authorized" in r.lower() or "no failure record" in r.lower()
        assert agent.procedural_memory_manager.deleted == []

    def test_delete_d_max_enforced(self):
        s1 = _FakeSkill("proc-1", "skill-1", "i")
        s2 = _FakeSkill("proc-2", "skill-2", "i")
        agent = _FakeAgent(_FakeProcManager([s1, s2]))
        agent._delete_budget_remaining = 1
        agent._delete_authorized_skill_ids = {"proc-1", "proc-2"}
        # Prefer hard delete here to keep the assertion on the hard path simple.
        agent._prefer_soft_delete = False

        r1 = _run(mt.skill_delete(agent, "proc-1"))
        assert "deleted" in r1.lower()
        assert agent._delete_budget_remaining == 0

        # Second authorized delete must be refused: D_max=1.
        r2 = _run(mt.skill_delete(agent, "proc-2"))
        assert (
            "budget" in r2.lower() or "d_max" in r2.lower() or "exhausted" in r2.lower()
        )
        assert agent.procedural_memory_manager.deleted == ["proc-1"]

    def test_soft_delete_preferred_and_excludes_from_retrieval(self):
        skill = _FakeSkill("proc-1", "skill-x", "i")
        mgr = _FakeSoftDeleteManager([skill])
        agent = _FakeAgent(mgr)
        agent._delete_budget_remaining = 1
        agent._delete_authorized_skill_ids = {"proc-1"}
        agent._prefer_soft_delete = True

        r = _run(mt.skill_delete(agent, "proc-1"))
        assert (
            "deleted" in r.lower()
            or "superseded" in r.lower()
            or "excluded" in r.lower()
        )
        # Soft-delete path taken (not a hard delete).
        assert mgr.soft_deleted == ["proc-1"]
        assert mgr.deleted == []
        # Excluded from retrieval afterwards.
        remaining = _run(mgr.list_procedures())
        assert all(s.id != "proc-1" for s in remaining)

    def test_delete_unset_budget_means_no_limit(self):
        # When no delete-budget attr is set, deletes are NOT capped, but they
        # still require authorization (the harmful-naming gate is independent of
        # the budget counter).
        s1 = _FakeSkill("proc-1", "skill-1", "i")
        agent = _FakeAgent(_FakeProcManager([s1]))
        agent._delete_authorized_skill_ids = {"proc-1"}
        agent._prefer_soft_delete = False
        r = _run(mt.skill_delete(agent, "proc-1"))
        assert "deleted" in r.lower()
        assert agent.procedural_memory_manager.deleted == ["proc-1"]
