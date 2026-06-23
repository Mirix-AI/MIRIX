"""Goal-3 tests — experiences -> procedural-skill self-evolution (DB-free).

Drives the injectable core `_run_experience_evolution_core` with a fake
experience manager, fake snapshot/run_step collaborators, and a fake agent — no
server, no DB, no LLM. Asserts the load-bearing contract:

* budget mapping: worth_avoiding -> n_high_fail, worth_learning -> n_high_succ,
  so the existing C4 formula (avoid weighted heavier than learn) applies; avoid
  yields a budget >= the same count of learns.
* B_min=0 early-exit: no pending experiences -> skipped, NO step, NO consume.
* ordering invariant: snapshot BEFORE -> step -> snapshot AFTER -> diff ->
  mark_consumed runs AFTER the diff (lineage can't be lost to a reset).
* no-delete gate: the agent gets a zero delete budget + empty auth set + soft
  preference set at step time.
* lineage: influenced_skill_ids = created|edited|deleted from the diff, passed
  to mark_consumed.

Also a DB-free budget-mapping check and a compact-payload rendering check.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from mirix.functions.function_sets.memory_tools import compute_edit_budget
from mirix.services.skill_experience_curator import (
    _run_experience_evolution_core,
    build_experience_payload,
)


# ============================ Budget mapping (DB-free) ======================


class TestBudgetMapping:
    def test_avoid_weighted_heavier_than_learn(self):
        # Same count: all-avoid budget >= all-learn budget (alpha_fail >= alpha_succ).
        b_avoid = compute_edit_budget({"n_high_fail": 3, "n_high_succ": 0})
        b_learn = compute_edit_budget({"n_high_fail": 0, "n_high_succ": 3})
        assert b_avoid >= b_learn

    def test_zero_experiences_is_min_budget(self):
        assert compute_edit_budget({"n_high_fail": 0, "n_high_succ": 0}) == 0


# ============================ Payload rendering (DB-free) ===================


class TestPayloadRendering:
    def test_compact_blocks_tagged_learn_avoid(self):
        import json

        exps = [
            SimpleNamespace(
                experience_type="worth_avoiding", title="Avoid X",
                content="do not X", importance=0.9, credibility=0.9,
                evidence=json.dumps({"quote": "no such column", "signal_type": "tool_error"}),
            ),
            SimpleNamespace(
                experience_type="worth_learning", title="Do Y",
                content="prefer Y", importance=0.5, credibility=0.5,
                evidence=json.dumps({"quote": "great", "signal_type": "user_confirmation"}),
            ),
        ]
        payload = build_experience_payload(exps)
        assert "[AVOID] Avoid X" in payload
        assert "[LEARN] Do Y" in payload
        assert "no such column" in payload  # evidence quote surfaced
        assert "importance=0.90" in payload

    def test_empty_list_renders_placeholder(self):
        payload = build_experience_payload([])
        assert "(no experiences)" in payload


# ============================ Fakes for the core ===========================


class _FakeExperienceManager:
    """In-memory experience manager exposing the 3 methods the core calls."""

    def __init__(self, experiences):
        self._experiences = experiences
        self.consumed_calls = []  # (ids, run_id, influenced)
        self.superseded_calls = []  # [ids]

    async def list_experiences(self, *, agent_id, user_id, status, limit, ids=None):
        assert status == "pending"
        # Mirror the real manager: ids=None -> no filter; ids=[] -> nothing;
        # otherwise scope to the given id set (this round's batch).
        if ids is not None:
            if not ids:
                return []
            wanted = set(ids)
            return [e for e in self._experiences if e.id in wanted][:limit]
        return list(self._experiences)[:limit]

    async def aggregate(self, *, ids):
        rows = [e for e in self._experiences if e.id in set(ids)]
        n_avoid = sum(1 for e in rows if e.experience_type == "worth_avoiding")
        n_learn = sum(1 for e in rows if e.experience_type == "worth_learning")
        return {
            "n": len(rows),
            "n_worth_learning": n_learn,
            "n_worth_avoiding": n_avoid,
            "sum_priority": sum(e.importance * e.credibility for e in rows),
        }

    async def mark_consumed(self, *, ids, run_id, influenced_skill_ids=None):
        self.consumed_calls.append((list(ids), run_id, influenced_skill_ids))
        return len(ids)

    async def mark_superseded(self, *, ids, agent_id=None, user_id=None):
        self.superseded_calls.append(list(ids))
        return len(ids)


class _Skill:
    def __init__(self, sid, version=1, instructions="i", description="d"):
        self.id = sid
        self.version = version
        self.instructions = instructions
        self.description = description


def _exp(eid, etype, imp, cred):
    return SimpleNamespace(
        id=eid, experience_type=etype, title=f"title-{eid}",
        content=f"content-{eid}", importance=imp, credibility=cred,
        evidence='{"quote":"q","signal_type":"inferred"}',
    )


class _OrderTracker:
    """Records the call order across snapshot/step/consume to assert ordering."""

    def __init__(self):
        self.events = []


@pytest.mark.asyncio
class TestEvolutionCore:
    async def test_b_min_skip_no_step_no_consume(self):
        mgr = _FakeExperienceManager([])  # nothing pending
        agent = SimpleNamespace()
        tracker = _OrderTracker()

        async def snapshot():
            tracker.events.append("snapshot")
            return []

        async def run_step(a, payload, budget):
            tracker.events.append("step")

        result = await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent,
            agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
            snapshot_skills=snapshot, run_step=run_step,
        )
        assert result["skipped"] is True
        assert result["budget"] == 0
        assert "step" not in tracker.events  # no step
        assert mgr.consumed_calls == []      # no consume

    async def test_ordering_snapshot_step_snapshot_diff_then_consume(self):
        exps = [
            _exp("sexp-a", "worth_avoiding", 0.9, 0.9),
            _exp("sexp-b", "worth_learning", 0.7, 0.8),
        ]
        mgr = _FakeExperienceManager(exps)
        agent = SimpleNamespace()
        tracker = _OrderTracker()

        # before snapshot: skill s1 v1. after: s1 v2 (edited) + s2 (created).
        snapshots = [
            [_Skill("s1", version=1)],
            [_Skill("s1", version=2), _Skill("s2", version=1)],
        ]
        snap_iter = iter(snapshots)

        async def snapshot():
            tracker.events.append("snapshot")
            return next(snap_iter)

        async def run_step(a, payload, budget):
            tracker.events.append("step")
            # The budget must be set on the agent BEFORE the step.
            assert a._edit_budget_remaining == budget

        # mark_consumed records its order via a tracker wrapper.
        orig_consume = mgr.mark_consumed

        async def tracked_consume(**kwargs):
            tracker.events.append("consume")
            return await orig_consume(**kwargs)

        mgr.mark_consumed = tracked_consume

        result = await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent,
            agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
            snapshot_skills=snapshot, run_step=run_step,
        )

        # Ordering invariant: snapshot, step, snapshot, then consume LAST.
        assert tracker.events == ["snapshot", "step", "snapshot", "consume"]
        # Diff computed BEFORE consume.
        assert set(result["changes"]["created"]) == {"s2"}
        assert set(result["changes"]["edited"]) == {"s1"}
        assert result["skills_changed"] == 2
        # Lineage = union of created|edited|deleted, passed to mark_consumed.
        ids, run_id, influenced = mgr.consumed_calls[0]
        assert set(ids) == {"sexp-a", "sexp-b"}
        assert run_id == result["run_id"]
        assert set(influenced) == {"s1", "s2"}
        assert set(result["influenced_skill_ids"]) == {"s1", "s2"}

    async def test_step_failure_leaves_experiences_pending(self):
        # If the step raises, the core must NOT consume the experiences (the
        # consume runs AFTER a successful step), so a retry can re-process them.
        exps = [_exp("sexp-a", "worth_avoiding", 0.9, 0.9)]
        mgr = _FakeExperienceManager(exps)
        agent = SimpleNamespace()

        async def snapshot():
            return []

        async def run_step(a, payload, budget):
            raise RuntimeError("step blew up")

        with pytest.raises(RuntimeError):
            await _run_experience_evolution_core(
                experience_manager=mgr, agent=agent,
                agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
                snapshot_skills=snapshot, run_step=run_step,
            )
        # mark_consumed must NOT have been called.
        assert mgr.consumed_calls == []

    async def test_no_delete_gate_set_at_step_time(self):
        exps = [_exp("sexp-a", "worth_avoiding", 0.9, 0.9)]
        mgr = _FakeExperienceManager(exps)
        agent = SimpleNamespace()
        observed = {}

        async def snapshot():
            return []

        async def run_step(a, payload, budget):
            observed["edit"] = a._edit_budget_remaining
            observed["delete"] = a._delete_budget_remaining
            observed["auth"] = a._delete_authorized_skill_ids
            observed["soft"] = a._prefer_soft_delete

        await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent,
            agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
            snapshot_skills=snapshot, run_step=run_step,
        )
        assert observed["edit"] >= 1
        assert observed["delete"] == 0
        assert observed["auth"] == set()
        assert observed["soft"] is True

    async def _budget_for(self, experiences):
        """Drive the core with a no-op step; return (result, observed budget)."""
        mgr = _FakeExperienceManager(experiences)
        agent = SimpleNamespace()
        observed = {}

        async def snapshot():
            return []

        async def run_step(a, payload, budget):
            observed["budget"] = budget

        result = await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent,
            agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
            snapshot_skills=snapshot, run_step=run_step,
        )
        return result, observed["budget"], mgr

    async def test_budget_reflects_avoid_count(self):
        # 3 worth_avoiding -> budget mapped via n_high_fail.
        exps = [_exp(f"sexp-{i}", "worth_avoiding", 0.9, 0.9) for i in range(3)]
        result, budget, mgr = await self._budget_for(exps)
        expected = compute_edit_budget({"n_high_fail": 3, "n_high_succ": 0})
        assert result["budget"] == expected
        assert budget == expected
        assert result["consumed_count"] == 3

    async def test_budget_maps_worth_learning_to_n_high_succ(self):
        # The core must map worth_learning -> n_high_succ (NOT n_high_fail and
        # NOT ignored). All-learn budget must equal the formula's all-succ value.
        exps = [_exp(f"sexp-{i}", "worth_learning", 0.8, 0.8) for i in range(3)]
        result, budget, _ = await self._budget_for(exps)
        expected_succ = compute_edit_budget({"n_high_fail": 0, "n_high_succ": 3})
        assert result["budget"] == expected_succ
        # And it must NOT be mistakenly mapped to the avoid weight.
        wrong_as_fail = compute_edit_budget({"n_high_fail": 3, "n_high_succ": 0})
        if wrong_as_fail != expected_succ:  # only meaningful when weights differ
            assert result["budget"] != wrong_as_fail
        # Learns must not be silently ignored: 3 learns >= budget for 0 learns.
        assert result["budget"] >= compute_edit_budget(
            {"n_high_fail": 0, "n_high_succ": 0}
        )

    async def test_budget_mixed_avoid_and_learn(self):
        # 2 avoid + 2 learn must map to the exact mixed formula value, proving
        # both counts flow through (not just one).
        exps = [
            _exp("sexp-a1", "worth_avoiding", 0.9, 0.9),
            _exp("sexp-a2", "worth_avoiding", 0.7, 0.7),
            _exp("sexp-l1", "worth_learning", 0.8, 0.8),
            _exp("sexp-l2", "worth_learning", 0.6, 0.6),
        ]
        result, budget, _ = await self._budget_for(exps)
        expected = compute_edit_budget({"n_high_fail": 2, "n_high_succ": 2})
        assert result["budget"] == expected
        assert budget == expected
        assert result["consumed_count"] == 4


# ===================== Round scoping (this window only) ====================


@pytest.mark.asyncio
class TestRoundScoping:
    """A scoped evolve sees ONLY the round's freshly-distilled experiences.

    The auto-dream procedural path passes the ids it just distilled so an evolve
    never re-surfaces earlier rounds' leftover pending experiences (which would
    dilute the curator prompt and hurt skill quality).
    """

    @staticmethod
    async def _run(mgr, *, experience_ids):
        agent = SimpleNamespace()
        seen = {}

        async def snapshot():
            return []

        async def run_step(a, payload, budget):
            seen["payload"] = payload
            seen["budget"] = budget

        result = await _run_experience_evolution_core(
            experience_manager=mgr, agent=agent,
            agent_id="proc-1", meta_agent_id="meta-1", user_id="user-1",
            snapshot_skills=snapshot, run_step=run_step,
            experience_ids=experience_ids,
        )
        return result, seen

    async def test_scopes_to_provided_ids_only(self):
        # Pool has 2 "old" + 2 "new" pending; scope to the 2 new ones only.
        old = [
            _exp("sexp-old1", "worth_avoiding", 0.9, 0.9),
            _exp("sexp-old2", "worth_learning", 0.9, 0.9),
        ]
        new = [
            _exp("sexp-new1", "worth_avoiding", 0.6, 0.6),
            _exp("sexp-new2", "worth_learning", 0.5, 0.5),
        ]
        mgr = _FakeExperienceManager(old + new)
        new_ids = [e.id for e in new]

        result, seen = await self._run(mgr, experience_ids=new_ids)

        assert result["skipped"] is False
        # Only the new batch is consumed — the old pending stays untouched.
        consumed_ids, _run, _infl = mgr.consumed_calls[0]
        assert set(consumed_ids) == {"sexp-new1", "sexp-new2"}
        assert result["consumed_count"] == 2
        # Budget reflects the scoped counts (1 avoid + 1 learn), NOT all four.
        assert result["budget"] == compute_edit_budget(
            {"n_high_fail": 1, "n_high_succ": 1}
        )
        # The old experiences' titles must NOT appear in the curator payload.
        assert "title-sexp-old1" not in seen["payload"]
        assert "title-sexp-old2" not in seen["payload"]
        assert "title-sexp-new1" in seen["payload"]
        # Whole scoped batch fit under the cap -> no overflow to supersede.
        assert mgr.superseded_calls == []

    async def test_overflow_beyond_cap_is_superseded_not_stranded(self):
        # A single round distilling MORE than the per-run cap must not strand its
        # lowest-priority tail as pending forever (future rounds pass only THEIR
        # fresh ids). The top _MAX_EXPERIENCES_PER_RUN are consumed; the rest of
        # THIS round's scoped ids are superseded (out of sight, still auditable).
        from mirix.services.skill_experience_curator import _MAX_EXPERIENCES_PER_RUN

        n = _MAX_EXPERIENCES_PER_RUN + 1
        exps = [_exp(f"sexp-{i:03d}", "worth_avoiding", 0.9, 0.9) for i in range(n)]
        mgr = _FakeExperienceManager(exps)
        all_ids = [e.id for e in exps]

        result, _seen = await self._run(mgr, experience_ids=all_ids)

        consumed_ids, _run, _infl = mgr.consumed_calls[0]
        assert len(consumed_ids) == _MAX_EXPERIENCES_PER_RUN
        assert result["consumed_count"] == _MAX_EXPERIENCES_PER_RUN
        # Exactly the overflow tail (those not consumed) is superseded — nothing
        # from this round is left pending/stranded.
        overflow = set(all_ids) - set(consumed_ids)
        assert len(overflow) == 1
        assert set(mgr.superseded_calls[0]) == overflow
        assert result["superseded_count"] == 1

    async def test_empty_ids_skips_even_with_pending_pool(self):
        # A round that distilled nothing (ids=[]) must NOT sweep the pending pool.
        mgr = _FakeExperienceManager([
            _exp("sexp-old1", "worth_avoiding", 0.9, 0.9),
        ])
        result, seen = await self._run(mgr, experience_ids=[])
        assert result["skipped"] is True
        assert result["budget"] == 0
        assert "payload" not in seen        # no step ran
        assert mgr.consumed_calls == []     # nothing consumed

    async def test_none_ids_evolves_whole_pool(self):
        # Backward-compat: ids=None means "no scope" -> the whole pending pool.
        mgr = _FakeExperienceManager([
            _exp("sexp-a", "worth_avoiding", 0.9, 0.9),
            _exp("sexp-b", "worth_learning", 0.7, 0.7),
        ])
        result, _seen = await self._run(mgr, experience_ids=None)
        consumed_ids, _run, _infl = mgr.consumed_calls[0]
        assert set(consumed_ids) == {"sexp-a", "sexp-b"}
        assert result["consumed_count"] == 2
        # Unscoped/global mode must NOT supersede leftovers — a later global
        # sweep is meant to pick them up.
        assert mgr.superseded_calls == []


# ============================ Module hygiene ===============================


class TestCuratorHygiene:
    def test_core_is_async(self):
        assert inspect.iscoroutinefunction(_run_experience_evolution_core)

    def test_no_asyncio_run(self):
        import mirix.services.skill_experience_curator as mod

        src = inspect.getsource(mod)
        assert "asyncio.run(" not in src

    def test_does_not_couple_to_metaclaw_record_store(self):
        # The general curator must NOT drive the MetaClaw SkillEvolutionRecord
        # store. Its docstring may CONTRAST against the records path (that's
        # documentation), but no executable line may reference the MetaClaw
        # manager/record types. Assert on code, stripping comments/docstrings.
        import mirix.services.skill_experience_curator as mod

        src = inspect.getsource(mod)
        code_only = _strip_comments_and_docstrings(src).lower()
        for banned in [
            "skillevolutionrecord",
            "run_records_evolution",
            "record_round_result",
            "round_index",
            "quality_score",
        ]:
            assert banned not in code_only, f"MetaClaw coupling leaked into code: {banned!r}"


def _strip_comments_and_docstrings(src: str) -> str:
    """Return ``src`` with comments and string/docstring literals removed.

    Tokenize-based so a MetaClaw term inside a contrastive docstring or comment
    does not trip the coupling assertion — only executable code is checked.
    """
    import io
    import tokenize

    out = []
    prev_type = tokenize.INDENT
    tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok_type, tok_str, _start, _end, _line in tokens:
        if tok_type in (tokenize.COMMENT, tokenize.STRING):
            continue
        if tok_type == tokenize.NL or tok_type == tokenize.NEWLINE:
            out.append("\n")
        else:
            out.append(tok_str + " ")
        prev_type = tok_type
    _ = prev_type
    return "".join(out)
