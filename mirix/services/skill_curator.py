"""C3 — records-based skill curator (the new evolution path).

This is the orchestration CORE for the records-based evolution introduced in
DESIGN §C3/§C4. It is dependency-injected so it can be unit-tested with no
server, no network, and no real agent step:

* the curator reads the window's PENDING records from the C2 record store
  (failures first, ranked by quality_score) and formats them compactly as
  ``title + detail + evidence`` — NOT raw transcripts (the context-bloat fix);
* it computes the count-driven C4 budget from the C2 ``aggregate`` and, when the
  window has NO structure-gated records (``n_high_fail + n_high_succ == 0``),
  SKIPS the curator entirely (B_min=0 early exit — no agent spawn, no LLM);
* it sets the per-instance edit/delete budgets on the agent BEFORE the step;
* AFTER the step it diffs the before/after skill snapshots, computes the
  ``influenced_skill_ids`` lineage, and flips the consumed records — all OUTSIDE
  the agent's tool loop and the evolve endpoint's in-context reset window, so a
  reset can never wipe the bookkeeping (DESIGN §C3, fixes P1-7);
* a PER-AGENT asyncio lock serializes concurrent evolves on the same agent.

The REST endpoint (`/v1/skills/evolve-from-records`) is a thin wrapper that wires
the real collaborators (record manager, procedural agent step, skill snapshots,
lineage writer) into :func:`run_records_evolution`. The existing
`/v1/skills/evolve` raw-transcript path is left byte-identical.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

from mirix.constants import SKILL_DELETE_BUDGET_MAX
from mirix.functions.function_sets.memory_tools import (
    _record_authorizes_delete,
    compute_edit_budget,
)
from mirix.log import get_logger
from mirix.schemas.skill_evolution_record import (
    SKILL_RECORD_MAX_DETAIL_LEN,
)

logger = get_logger(__name__)


# Per-agent locks: a records-based evolve hard-resets the procedural agent's
# in-context history before and after the step, so two concurrent evolves on the
# SAME agent would corrupt each other (one reset deleting the other's mid-chain
# messages). Serialize them per agent id. Keyed by agent id; the dict only grows
# by the number of distinct procedural agents (bounded, tiny), and a lock object
# is cheap, so we never evict.
_evolve_locks: Dict[str, asyncio.Lock] = {}


def _lock_for_agent(agent_id: str) -> asyncio.Lock:
    lock = _evolve_locks.get(agent_id)
    if lock is None:
        lock = asyncio.Lock()
        _evolve_locks[agent_id] = lock
    return lock


def build_curator_payload(
    records: List,
    *,
    superseded_signatures: Optional[List[str]] = None,
) -> str:
    """Render the window's pending records as a COMPACT curator prompt.

    Records arrive already failures-first (C2 ``list_pending`` ordering); this
    preserves that order and emits one block per record as
    ``[FAILURE|SUCCESS] title — detail (evidence: r3, r4)`` — roughly one to two
    orders of magnitude smaller than the raw transcripts the old path joined,
    which is the context-bloat fix (DESIGN §C3).

    ``superseded_signatures`` (C4 anti-thrash, P1-5) is rendered as a leading
    "do NOT re-propose" block derived from recently superseded records' lineage,
    so the curator does not oscillate (re-introduce an edit it just reverted).
    """
    lines: List[str] = []

    sigs = superseded_signatures or []
    if sigs:
        lines.append("## Do NOT re-propose (recently superseded — avoid oscillation)")
        for sig in sigs:
            lines.append(f"- {sig}")
        lines.append("")

    lines.append("## Distilled records from this window (failures first)")
    if not records:
        lines.append("(no records)")
    for rec in records:
        rtype = (_attr(rec, "record_type") or "").upper()
        title = _attr(rec, "title") or ""
        detail = (_attr(rec, "detail") or "")[:SKILL_RECORD_MAX_DETAIL_LEN]
        evidence = _attr(rec, "evidence_round_ids") or []
        ev_str = ", ".join(str(e) for e in evidence)
        round_id = _attr(rec, "round_id") or ""
        lines.append(
            f"- [{rtype}] {title} — {detail} (round: {round_id}; evidence: {ev_str})"
        )

    return "\n".join(lines)


def _attr(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def build_superseded_signatures(superseded_records: List) -> List[str]:
    """Turn recently-superseded records into short anti-thrash signatures.

    Kept simple (DESIGN §C4): one line per superseded record naming its round +
    title so the curator recognizes a previously-rejected proposal.
    """
    sigs: List[str] = []
    for rec in superseded_records or []:
        round_id = _attr(rec, "round_id") or "?"
        title = _attr(rec, "title") or ""
        sigs.append(f"round {round_id}: {title}")
    return sigs


def compute_delete_authorizations(records: List, skills: List) -> set:
    """Skill ids a window's FAILURE records authorize for deletion (C4, P1-4).

    A skill is delete-authorized iff some failure record in the window names it
    actively harmful (see ``_record_authorizes_delete``). Successes and
    "redundant"-only mentions never authorize a delete.
    """
    authorized = set()
    for skill in skills or []:
        sid = _attr(skill, "id")
        sname = _attr(skill, "name") or ""
        for rec in records or []:
            if _record_authorizes_delete(rec, skill_name=sname, skill_id=sid):
                authorized.add(sid)
                break
    return authorized


async def run_records_evolution(
    *,
    record_manager,
    agent,
    agent_id: str,
    run_id: str,
    watermark: int,
    snapshot_skills: Callable[[], Awaitable[List]] | Callable[[], List],
    run_step: Callable[[object, str, int], Awaitable[None]],
    apply_lineage: Callable[[List[str], List[str]], Awaitable[None]],
    autonomous_budget_fn: Optional[Callable[[Dict], Awaitable[int]]] = None,
    use_autonomous_budget: bool = False,
    superseded_records: Optional[List] = None,
    skills_for_delete_auth: Optional[List] = None,
) -> Dict:
    """Run ONE records-based evolution for a procedural agent.

    Collaborators are injected so this is testable without a server/LLM:

    * ``record_manager`` — the C2 :class:`SkillEvolutionRecordManager`.
    * ``snapshot_skills`` — returns the current skill list (called once before
      and once after the step) for the before/after diff. May be sync or async.
    * ``run_step(agent, payload, budget)`` — runs the procedural agent over the
      curator payload (production: ``ProceduralMemoryAgent.step`` wrapped by the
      endpoint, which also performs the in-context resets). Awaitable.
    * ``apply_lineage(record_ids, influenced_skill_ids)`` — persists the lineage
      (sets ``influenced_skill_ids`` on the consumed records). Awaitable.
    * ``autonomous_budget_fn(aggregate)`` — optional cheap LLM that may only
      REDUCE the budget; gated behind ``use_autonomous_budget`` (default False).

    Returns a dict with ``skipped``, ``budget``, ``changes`` (created/edited/
    deleted id lists), ``influenced_skill_ids``, and ``consumed_count``.

    Bookkeeping ordering (the load-bearing invariant): ``list_pending`` -> build
    payload -> set budget -> ``run_step`` -> snapshot diff -> compute lineage ->
    ``mark_consumed`` + ``apply_lineage``. The consume/lineage happen here, in
    the caller's code, AFTER the diff — never as an agent tool side-effect that
    the endpoint's post-step reset would wipe.
    """
    lock = _lock_for_agent(agent_id)
    async with lock:
        return await _run_records_evolution_locked(
            record_manager=record_manager,
            agent=agent,
            agent_id=agent_id,
            run_id=run_id,
            watermark=watermark,
            snapshot_skills=snapshot_skills,
            run_step=run_step,
            apply_lineage=apply_lineage,
            autonomous_budget_fn=autonomous_budget_fn,
            use_autonomous_budget=use_autonomous_budget,
            superseded_records=superseded_records,
            skills_for_delete_auth=skills_for_delete_auth,
        )


def _empty_result(budget: int, *, skipped: bool) -> Dict:
    return {
        "skipped": skipped,
        "budget": budget,
        "changes": {"created": [], "edited": [], "deleted": []},
        "influenced_skill_ids": [],
        "consumed_count": 0,
    }


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _run_records_evolution_locked(
    *,
    record_manager,
    agent,
    agent_id: str,
    run_id: str,
    watermark: int,
    snapshot_skills,
    run_step,
    apply_lineage,
    autonomous_budget_fn,
    use_autonomous_budget: bool,
    superseded_records,
    skills_for_delete_auth,
) -> Dict:
    # 1) Read the window's pending records (failures first, watermark-bounded).
    records = await record_manager.list_pending(
        agent_id=agent_id, before_round_index=watermark
    )
    record_ids = [_attr(r, "id") for r in records]

    # 2) Structure-gated aggregate -> count-driven budget.
    agg = await record_manager.aggregate(ids=record_ids)
    gated = int(agg.get("n_high_fail", 0)) + int(agg.get("n_high_succ", 0))

    # B_min=0 early exit: no structure-gated records -> skip the curator entirely
    # (no step, no LLM, no consume). Return an empty diff.
    if gated == 0:
        logger.info(
            "[curator] B_min=0 skip: no structure-gated records "
            "(agent=%s, run=%s, n=%s)",
            agent_id,
            run_id,
            agg.get("n"),
        )
        return _empty_result(0, skipped=True)

    autonomous = None
    if use_autonomous_budget and autonomous_budget_fn is not None:
        try:
            autonomous = await _maybe_await(autonomous_budget_fn(agg))
        except Exception as e:  # noqa: BLE001 — autonomous is best-effort
            logger.warning("[curator] autonomous budget call failed: %s", e)
            autonomous = None

    budget = compute_edit_budget(agg, autonomous=autonomous)

    # 3) Build the compact, failure-first curator payload + anti-thrash block.
    sigs = build_superseded_signatures(superseded_records or [])
    payload = build_curator_payload(records, superseded_signatures=sigs)

    # 4) Set the per-instance budgets BEFORE the step (C4, P1-2). Plain ints on
    # the agent instance: no cross-user / cross-run leak.
    agent._edit_budget_remaining = budget
    agent._delete_budget_remaining = SKILL_DELETE_BUDGET_MAX
    agent._prefer_soft_delete = True
    agent._delete_authorized_skill_ids = compute_delete_authorizations(
        records, skills_for_delete_auth or []
    )

    # 5) Snapshot BEFORE, run the agent step, snapshot AFTER.
    before = await _maybe_await(snapshot_skills())
    await run_step(agent, payload, budget)
    after = await _maybe_await(snapshot_skills())

    # 6) Diff (identical semantics to the raw-transcript evolve path).
    changes = _diff_skills(before, after)
    influenced = sorted(
        set(changes["created"]) | set(changes["edited"]) | set(changes["deleted"])
    )

    # 7) Bookkeeping OUTSIDE the step / reset window: consume the records and
    # write the influenced-skill lineage. This is the whole point of doing it
    # here in the caller rather than as an agent tool side-effect.
    consumed = 0
    if record_ids:
        consumed = await record_manager.mark_consumed(ids=record_ids, run_id=run_id)
        await apply_lineage(record_ids, influenced)

    return {
        "skipped": False,
        "budget": budget,
        "changes": changes,
        "influenced_skill_ids": influenced,
        "consumed_count": consumed,
    }


def _diff_skills(before: List, after: List) -> Dict[str, List[str]]:
    """Compute created / edited / deleted skill ids between two snapshots.

    Mirrors the diff in the raw-transcript ``evolve_skills`` endpoint: a skill is
    EDITED when its version, instructions, or description changed.
    """
    before_map = {_attr(s, "id"): s for s in (before or [])}
    after_map = {_attr(s, "id"): s for s in (after or [])}
    before_ids = set(before_map)
    after_ids = set(after_map)

    created = sorted(after_ids - before_ids)
    deleted = sorted(before_ids - after_ids)
    edited = []
    for sid in before_ids & after_ids:
        b = before_map[sid]
        a = after_map[sid]
        if (
            _attr(b, "version") != _attr(a, "version")
            or _attr(b, "instructions") != _attr(a, "instructions")
            or _attr(b, "description") != _attr(a, "description")
        ):
            edited.append(sid)
    return {"created": created, "edited": sorted(edited), "deleted": deleted}
