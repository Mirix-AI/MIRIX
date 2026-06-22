"""Goal 3 — general experience-driven skill self-evolution.

This is the records-based curator's GENERAL analog. Where
``skill_curator.run_records_evolution`` is hard-wired to the MetaClaw
``SkillEvolutionRecordManager`` (day / round_id / quality_score / failure-first
semantics), this drives the procedural skill agent (OUR paradigm:
``skill_list`` / ``skill_read`` / ``skill_create`` / ``skill_edit``) from the
GENERAL :class:`SkillExperience` store produced by Goal-2 distillation.

Flow (mirrors the load-bearing ordering of the records path so a context reset
can never wipe the bookkeeping):

    list_experiences(status="pending", priority DESC)
      -> aggregate -> compute_edit_budget (avoid weighted heavier than learn)
      -> B_min=0 early-exit when there is nothing structure-gated
      -> build compact payload (one block per experience, priority-ordered)
      -> set per-instance edit budget on the agent
      -> snapshot BEFORE -> run procedural agent step (reset before+after)
      -> snapshot AFTER -> diff skills -> compute influenced_skill_ids lineage
      -> mark_consumed(ids, run_id) + write lineage  (OUTSIDE the reset window)

Unlike the MetaClaw path we do NOT authorize destructive deletes from
experiences (a ``worth_avoiding`` lesson names a pitfall, not a harmful skill to
destroy); v1 is creates/edits only. The edit-budget gate, the per-mutation size
gate, and ``skill_create`` name-dedup (all already enforced inside the skill
tools) keep mutations delta/incremental — never a wholesale rewrite.

A per-agent :class:`asyncio.Lock` (reused from :mod:`skill_curator`) serializes
concurrent evolves on the same procedural agent, because the step is bracketed
by in-context resets that would otherwise corrupt a concurrent run.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from mirix.functions.function_sets.memory_tools import compute_edit_budget
from mirix.log import get_logger
from mirix.schemas.agent import AgentState, AgentType
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.skill_experience import (
    SKILL_EXPERIENCE_MAX_CONTENT_LEN as _CONTENT_CAP,
)
from mirix.schemas.user import User as PydanticUser
from mirix.services.skill_curator import _diff_skills, _lock_for_agent

logger = get_logger(__name__)

# How many pending experiences to consume in one evolution pass. Bounded so a
# huge backlog can't blow up the curator prompt; the highest-priority ones come
# first (importance*credibility DESC), so the cap drops only the weakest.
_MAX_EXPERIENCES_PER_RUN = 50


def build_experience_payload(experiences: List) -> str:
    """Render pending experiences as a COMPACT, priority-ordered curator prompt.

    ``experiences`` arrive already ordered by ``importance*credibility`` DESC
    (the manager's ``list_experiences`` ordering). Emits one block per
    experience as ``[LEARN|AVOID] title — content (importance=…, credibility=…;
    evidence: "quote")`` — far smaller than raw transcripts.
    """
    lines: List[str] = [
        "## Distilled experiences from recent sessions (highest-priority first)",
    ]
    if not experiences:
        lines.append("(no experiences)")
        return "\n".join(lines)

    for exp in experiences:
        etype = (_attr(exp, "experience_type") or "").lower()
        tag = "LEARN" if etype == "worth_learning" else "AVOID"
        title = _attr(exp, "title") or ""
        content = (_attr(exp, "content") or "")[:_CONTENT_CAP]
        importance = _attr(exp, "importance")
        credibility = _attr(exp, "credibility")
        quote = _evidence_quote(_attr(exp, "evidence"))
        ev_str = f'; evidence: "{quote}"' if quote else ""
        lines.append(
            f"- [{tag}] {title} — {content} "
            f"(importance={_fmt(importance)}, credibility={_fmt(credibility)}{ev_str})"
        )
    return "\n".join(lines)


def _fmt(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _evidence_quote(evidence) -> str:
    """Pull the short grounding quote out of the stored evidence JSON string."""
    if not evidence:
        return ""
    import json

    try:
        data = json.loads(evidence)
        return str(data.get("quote") or "")[:240]
    except (ValueError, TypeError):
        return str(evidence)[:240]


def _attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


async def _resolve_procedural_agent_state(server, actor, meta_agent_state):
    """Resolve the procedural memory agent child of the meta agent.

    Mirrors ``rest_api._find_procedural_agent_state`` but scoped to THIS meta
    agent's children, so each user/meta-agent drives its own procedural agent.
    """
    children = await server.agent_manager.list_agents(
        actor=actor, parent_id=meta_agent_state.id
    )
    for child in children:
        if child.agent_type == AgentType.procedural_memory_agent:
            return child
    # No type-only flat fallback: matching any procedural agent in the org would
    # break per-(meta-agent, user) scoping (it could evolve a DIFFERENT user's /
    # meta agent's skill bank). If this meta agent has no procedural child, skip
    # evolution — the caller treats None as "no agent, leave experiences pending".
    return None


async def run_experience_evolution(
    *,
    user: PydanticUser,
    actor: PydanticClient,
    meta_agent_state: AgentState,
    experience_manager=None,
) -> Dict:
    """Evolve skills from this (agent, user)'s PENDING experiences (Goal 3).

    This is the production wiring: it resolves the real procedural agent, builds
    the snapshot / step / lineage collaborators, and delegates the load-bearing
    ordering to :func:`_run_experience_evolution_core` (which is injectable for
    tests). Returns ``{skipped, budget, changes, influenced_skill_ids,
    consumed_count, skills_changed}``.

    Safe to call with no pending experiences (B_min=0 early-exit, no agent spawn).
    """
    from mirix.agent import ProceduralMemoryAgent
    from mirix.constants import SKILL_EVOLVE_MAX_CHAINING_STEPS
    from mirix.schemas.message import Message as PydanticMessage
    from mirix.schemas.mirix_message_content import TextContent
    from mirix.server.rest_api import (
        _reset_agent_in_context_to_system,
        get_server,
    )
    from mirix.services.skill_experience_manager import SkillExperienceManager

    server = get_server()
    exp_mgr = experience_manager or SkillExperienceManager()

    proc_agent_state = await _resolve_procedural_agent_state(
        server, actor, meta_agent_state
    )
    if proc_agent_state is None:
        logger.info(
            "[experience-curator] no procedural agent for meta=%s; skipping evolution",
            meta_agent_state.id,
        )
        return _empty_result(0, skipped=True)

    # The procedural child may have been created BEFORE the Experience-Based
    # Evolution prompt section landed (or its tool set predates skill_*). Refresh
    # its tools + system prompt from disk before we build the agent, so a manual
    # /memory/auto_dream or a direct AutoDreamManager.run(mode="procedural") never
    # runs a stale prompt. Best-effort: a failure here just uses the persisted
    # state (the prompt section is additive, not load-bearing for the tools).
    try:
        await server.agent_manager.update_agent_tools_and_system_prompts(
            proc_agent_state.id, actor=actor
        )
        proc_agent_state = await server.agent_manager.get_agent_by_id(
            agent_id=proc_agent_state.id, actor=actor
        )
    except Exception as refresh_err:  # noqa: BLE001
        logger.warning(
            "[experience-curator] could not refresh procedural agent prompt/tools: %s",
            refresh_err,
        )

    timezone_str = getattr(user, "timezone", None) or "UTC"

    proc_agent = ProceduralMemoryAgent(
        agent_state=proc_agent_state,
        interface=server.default_interface_factory(),
        actor=actor,
        user=user,
    )

    async def _snapshot():
        return await server.procedural_memory_manager.list_procedures(
            agent_state=proc_agent_state,
            user=user,
            query="",
            search_field="description",
            search_method="bm25",
            limit=1000,
            timezone_str=timezone_str,
            use_cache=False,
        )

    async def _run_step(agent, payload, budget):
        input_message = PydanticMessage(
            role="user",
            content=[TextContent(text=payload)],
            agent_id=proc_agent_state.id,
            name="user",
        )
        # Stateless reset discipline (identical to the records-evolve endpoint):
        # hard-reset BEFORE and (in finally) AFTER the step so each evolution is
        # context-isolated. The consume/lineage bookkeeping runs AFTER this
        # returns (in the core), so the post-step reset can never wipe it.
        await _reset_agent_in_context_to_system(server, proc_agent_state.id, actor)
        try:
            await agent.step(
                input_messages=[input_message],
                chaining=True,
                max_chaining_steps=SKILL_EVOLVE_MAX_CHAINING_STEPS,
                actor=actor,
                user=user,
            )
        finally:
            try:
                await _reset_agent_in_context_to_system(
                    server, proc_agent_state.id, actor
                )
            except Exception as cleanup_err:  # noqa: BLE001
                logger.warning(
                    "Post-experience-evolve procedural context reset failed: %s",
                    cleanup_err,
                )

    return await _run_experience_evolution_core(
        experience_manager=exp_mgr,
        agent=proc_agent,
        agent_id=proc_agent_state.id,
        meta_agent_id=meta_agent_state.id,
        user_id=user.id,
        snapshot_skills=_snapshot,
        run_step=_run_step,
    )


def _empty_result(budget: int, *, skipped: bool) -> Dict:
    return {
        "skipped": skipped,
        "budget": budget,
        "changes": {"created": [], "edited": [], "deleted": []},
        "influenced_skill_ids": [],
        "consumed_count": 0,
        "skills_changed": 0,
    }


async def _maybe_await(value):
    import asyncio

    if asyncio.iscoroutine(value):
        return await value
    return value


async def _run_experience_evolution_core(
    *,
    experience_manager,
    agent,
    agent_id: str,
    meta_agent_id: str,
    user_id: str,
    snapshot_skills,
    run_step,
    run_id: Optional[str] = None,
) -> Dict:
    """The injectable core — no server/agent assembly, fully unit-testable.

    Collaborators (``snapshot_skills``, ``run_step``) are passed in so a test can
    drive the exact load-bearing ordering with mocks. The PER-AGENT lock makes
    concurrent evolves on the same procedural agent safe.
    """
    run_id = run_id or f"xprun-{uuid.uuid4().hex[:12]}"
    lock = _lock_for_agent(agent_id)
    async with lock:
        return await _evolve_locked(
            experience_manager=experience_manager,
            agent=agent,
            agent_id=agent_id,
            meta_agent_id=meta_agent_id,
            user_id=user_id,
            snapshot_skills=snapshot_skills,
            run_step=run_step,
            run_id=run_id,
        )


async def _evolve_locked(
    *,
    experience_manager,
    agent,
    agent_id: str,
    meta_agent_id: str,
    user_id: str,
    snapshot_skills,
    run_step,
    run_id: str,
) -> Dict:
    # 1) Read this (meta-agent, user)'s pending experiences, priority-ordered
    #    (importance*credibility DESC) — the experiences are keyed by the META
    #    agent id (their provenance owner), not the procedural agent.
    experiences = await experience_manager.list_experiences(
        agent_id=meta_agent_id,
        user_id=user_id,
        status="pending",
        limit=_MAX_EXPERIENCES_PER_RUN,
    )
    exp_ids = [_attr(e, "id") for e in experiences]

    # 2) Aggregate -> count-driven budget. Map worth_avoiding -> n_high_fail and
    #    worth_learning -> n_high_succ so the existing C4 formula (avoid weighted
    #    heavier than learn) applies unchanged.
    agg = await experience_manager.aggregate(ids=exp_ids)
    n_avoid = int(agg.get("n_worth_avoiding", 0) or 0)
    n_learn = int(agg.get("n_worth_learning", 0) or 0)

    # B_min=0 early-exit: nothing to learn from -> no step, no LLM, no consume.
    if n_avoid + n_learn == 0:
        logger.info(
            "[experience-curator] B_min=0 skip: no pending experiences "
            "(meta=%s, agent=%s, run=%s)",
            meta_agent_id,
            agent_id,
            run_id,
        )
        return _empty_result(0, skipped=True)

    budget = compute_edit_budget(
        {"n_high_fail": n_avoid, "n_high_succ": n_learn}
    )

    # 3) Compact, priority-ordered payload.
    payload = build_experience_payload(experiences)

    # 4) Set the per-instance budgets BEFORE the step (plain ints on the
    #    instance: no cross-user / cross-run leak). Experiences are creates/edits
    #    ONLY — a lesson names a pitfall, not a skill to destroy. ENFORCE that
    #    here (the prompt's "No deletes" rule is guidance, not a gate): with a
    #    zero delete budget and an empty authorization set, skill_delete's gate
    #    (memory_tools._delete gate) rejects every delete this run. Also prefer
    #    soft-delete so even a gate change never hard-destroys a skill from here.
    agent._edit_budget_remaining = budget
    agent._delete_budget_remaining = 0
    agent._delete_authorized_skill_ids = set()
    agent._prefer_soft_delete = True

    # 5) Snapshot BEFORE, run the step, snapshot AFTER.
    before = await _maybe_await(snapshot_skills())
    await run_step(agent, payload, budget)
    after = await _maybe_await(snapshot_skills())

    # 6) Diff (reuses the records-path semantics: version/instructions/description).
    changes = _diff_skills(before, after)
    influenced = sorted(
        set(changes["created"]) | set(changes["edited"]) | set(changes["deleted"])
    )

    # 7) Bookkeeping OUTSIDE the step / reset window: consume the experiences and
    #    stamp the influenced-skill lineage in ONE load-modify-save (mark_consumed
    #    accepts influenced_skill_ids), so a reset can never lose the audit trail.
    consumed = 0
    if exp_ids:
        consumed = await experience_manager.mark_consumed(
            ids=exp_ids,
            run_id=run_id,
            influenced_skill_ids=influenced or None,
        )

    skills_changed = (
        len(changes["created"]) + len(changes["edited"]) + len(changes["deleted"])
    )

    logger.info(
        "[experience-curator] meta=%s agent=%s run=%s: budget=%d consumed=%d "
        "created=%d edited=%d deleted=%d",
        meta_agent_id,
        agent_id,
        run_id,
        budget,
        consumed,
        len(changes["created"]),
        len(changes["edited"]),
        len(changes["deleted"]),
    )

    return {
        "skipped": False,
        "budget": budget,
        "changes": changes,
        "influenced_skill_ids": influenced,
        "consumed_count": consumed,
        "skills_changed": skills_changed,
        "run_id": run_id,
    }
