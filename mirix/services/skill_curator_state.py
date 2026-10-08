"""The per-batch state a skill curator run hands to the skill tools.

A skill learning run curates its pending experiences in batches. For
each batch the curator attaches a :class:`CuratorBatchState` to the procedural
agent instance (``agent.curator_batch``, next to ``agent.skill_run_id`` and
``agent.learning_run_id``); the skill tools in
``mirix.functions.function_sets.memory_tools`` read it to enforce:

* the **edit budget**: one create or edit per experience in the
  batch. Only a write that went through costs 1 — a call rejected for any
  reason (invalid arguments, a duplicate name, an unknown skill, a result over
  the length ceiling, no change at all, an exhausted budget) costs nothing;
* the **delete limit**: at most one soft delete per learning run,
  all batches together (:class:`CuratorRunLimits` is shared by the run's
  batches), and every delete names its reason;
* the **write guard** (the curator stage's safety against a cancelled run or a
  lost lease): every skill write runs ``write_guard(db)`` inside its own
  transaction, which locks the learning run row and requires it to be still
  queued/running — a run that was reclaimed, erased or failed can no longer
  change a skill;
* **lineage**: each write stamps the skill on the batch's experiences
  (``influenced_skill_ids``) in the same transaction, so a batch that is
  interrupted and curated again later knows what was already changed;
* **citations**: each write names the experiences — and the earlier
  lessons the batch showed from the experience pool — it comes from
  (``experience_ids``); only ids the batch showed are accepted, and the
  write stamps itself on exactly those (``cited_skill_ids``);
* **triage**: ``experience_triage`` records which of the shown ids
  that no change cites are covered by an existing skill, not worth a skill,
  or deferred; the batch's settle turns that into each experience's
  disposition and the pool's decay;
* the **read cap**: ``skill_read`` / ``skill_list`` output is capped at
  ``max(60,000 chars, floor(window * 0.8) * 4 * 0.1)``, head + tail, counted;
* **retrieval-text follow-ups**: retrieval matches a skill by its name,
  description and triggers only, so after an instructions edit of a skill
  went through — in this batch, or in an earlier, returned attempt on its
  experiences — one change of each of those fields of that skill in the same
  batch costs no budget (keeping them in step never competes with a lesson)
  and records no citation (it carries no lesson of its own).

Outside a learning run (the in-flow procedural agent) there is no batch state:
no budget and no delete limit, as before.

Pure data and helpers: no database, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

#: Soft deletes allowed per learning run, all batches together.
MAX_SKILL_DELETES_PER_RUN = 1

#: The floor of the skill tools' output cap in characters.
SKILL_OUTPUT_MIN_CHARS = 60_000

#: Skills per ``skill_list`` page (the page is also cut to the output cap).
SKILL_LIST_PAGE_SIZE = 50

WriteGuard = Callable[[Any], Awaitable[None]]

# Rejection kinds (the curator metrics count them per kind).
REJECT_INVALID = "invalid_arguments"
REJECT_NOT_FOUND = "not_found"
REJECT_DUPLICATE = "duplicate_name"
REJECT_TOO_LONG = "too_long"
REJECT_NO_CHANGE = "no_change"
REJECT_BUDGET = "budget_exhausted"
REJECT_DELETE_LIMIT = "delete_limit"
# The skill changed between the tool's read and its write (an optimistic check).
REJECT_CHANGED = "changed_meanwhile"
# finish_memory_update refused while some of the batch's experiences are neither
# cited nor triaged.
REJECT_UNADDRESSED = "unaddressed_experiences"

#: The fields retrieval matches a skill by (its retrieval text).
RETRIEVAL_TEXT_FIELDS = ("name", "description", "triggers")


def skill_output_char_cap(llm_config: Any) -> int:
    """``max(60,000, floor(window * 0.8) * 4 * 0.1)`` characters: the
    most a ``skill_read`` (or one ``skill_list`` page) returns — about 10% of
    the usable window at 4 characters per token."""
    window = int(getattr(llm_config, "context_window", 0) or 0)
    usable = window * 4 // 5  # floor(window * 0.8)
    return max(SKILL_OUTPUT_MIN_CHARS, usable * 4 // 10)


def head_tail_chars(text: str, cap: int) -> Tuple[str, int]:
    """``text`` within ``cap`` characters: its head and tail around a marker
    naming how many characters were left out. Returns ``(text, omitted)``;
    ``text`` unchanged (and 0) when it fits."""
    if len(text) <= cap:
        return text, 0
    omitted = len(text) - cap
    for _ in range(8):  # the marker's own length shifts the split a little
        marker = f"\n…[{omitted} characters omitted]…\n"
        room = cap - len(marker)
        if room <= 0:
            return text[:cap], len(text) - cap
        head = room - room // 2
        tail = room // 2
        actual = len(text) - head - tail
        if actual == omitted:
            break
        omitted = actual
    marker = f"\n…[{omitted} characters omitted]…\n"
    room = max(0, cap - len(marker))
    head = room - room // 2
    tail = room // 2
    return text[:head] + marker + (text[len(text) - tail :] if tail else ""), (
        len(text) - head - tail
    )


@dataclass
class CuratorRunLimits:
    """Limits shared by every batch of one learning run."""

    run_id: str
    max_deletes: int = MAX_SKILL_DELETES_PER_RUN
    deletes_used: int = 0

    @property
    def deletes_left(self) -> int:
        return max(0, self.max_deletes - self.deletes_used)


@dataclass
class CuratorBatchState:
    """One curator batch (``{run_id}:b{i}``) as the skill tools see it."""

    run_id: str
    batch_id: str
    experience_ids: Sequence[str]
    budget: int
    run_limits: CuratorRunLimits
    write_guard: Optional[WriteGuard] = None
    output_char_cap: int = SKILL_OUTPUT_MIN_CHARS
    spent: int = 0
    created: List[str] = field(default_factory=list)
    edited: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    outputs_capped: int = 0
    output_chars_omitted: int = 0
    # The experience pool: the earlier lessons the batch shows, the
    # batch experiences an earlier (returned) attempt already cited, the
    # citations of this batch's writes and the curator's triage.
    history_ids: Sequence[str] = ()
    precited_ids: Sequence[str] = ()
    citations: Dict[str, List[str]] = field(default_factory=dict)
    triage: Dict[str, str] = field(default_factory=dict)
    triage_calls: int = 0
    # Creates / edits that went through citing nothing, and soft deletes that
    # did (a delete costs no budget, so it is not a write).
    uncited_writes: int = 0
    uncited_deletes: int = 0
    # finish_memory_update calls refused because experiences were unaddressed.
    finish_refusals: int = 0
    # Edits that went through without recording the ids they cited, because
    # they add no text (a deletion, a renumbering, whitespace only), or are a
    # free retrieval-text follow-up citing ids nothing cited yet.
    citations_dropped: int = 0
    # Ids a create or edit that went through cited without a quote of
    # the text it added that applies them (or with one that is not there):
    # id -> the skills of those writes. Not recorded as citations.
    unverified_citations: Dict[str, List[str]] = field(default_factory=dict)
    citations_unverified: int = 0
    # One entry per create / edit that went through: what it changed, how many
    # content words it added and removed, and whether its citations counted.
    write_log: List[Dict[str, Any]] = field(default_factory=list)
    # Retrieval-text edits that followed an instructions edit of their skill
    # and cost no budget, and the (skill id, field) pairs they used.
    retrieval_followups: int = 0
    followup_fields: List[Tuple[str, str]] = field(default_factory=list)
    # Skills whose instructions an earlier, returned attempt on this batch's
    # experiences created or changed (the curator reads the lineage): their
    # retrieval-text follow-up is free as after this attempt's own edit.
    prior_instructions_edits: Sequence[str] = ()

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    @property
    def shown_ids(self) -> Tuple[str, ...]:
        """The batch's experiences, then its earlier lessons: the ids a tool
        may cite or triage."""
        seen: Dict[str, None] = {}
        for experience_id in list(self.experience_ids) + list(self.history_ids):
            seen.setdefault(str(experience_id), None)
        return tuple(seen)

    def unknown_ids(self, ids: Sequence[str]) -> List[str]:
        """The ids of ``ids`` this batch did not show."""
        shown = set(self.shown_ids)
        return [i for i in ids if i not in shown]

    def record_citations(
        self, ids: Sequence[str], skill_id: str, *, delete: bool = False
    ) -> None:
        """A create or edit (``delete``: a soft delete) that went through cited
        ``ids``; with none, one more uncited write (uncited delete)."""
        if not ids:
            if delete:
                self.uncited_deletes += 1
            else:
                self.uncited_writes += 1
            return
        for experience_id in ids:
            skills = self.citations.setdefault(experience_id, [])
            if skill_id not in skills:
                skills.append(skill_id)

    def record_unverified_citations(
        self, unverified: Dict[str, str], skill_id: str
    ) -> None:
        """A write that went through cited ``unverified`` (id -> why) without
        a quote showing where it applies them: they stay unaddressed
        unless another change cites them with one, or the triage lists them."""
        for experience_id in unverified:
            skills = self.unverified_citations.setdefault(experience_id, [])
            if skill_id not in skills:
                skills.append(skill_id)
            self.citations_unverified += 1

    def record_triage(
        self,
        covered: Sequence[str],
        not_worth: Sequence[str],
        deferred: Sequence[str],
    ) -> None:
        """The curator's triage; a later call overrides an id's earlier list
        (ids it does not name keep theirs)."""
        for kind, ids in (
            ("covered", covered),
            ("not_worth", not_worth),
            ("deferred", deferred),
        ):
            for experience_id in ids:
                self.triage[experience_id] = kind

    def is_cited(self, experience_id: str) -> bool:
        return experience_id in self.citations or experience_id in set(
            self.precited_ids
        )

    def unaddressed_experiences(self) -> List[str]:
        """The batch's own experiences (not its earlier lessons) that no
        change cites (an earlier attempt's citation included) and the triage
        does not list, in batch order: what finish_memory_update would leave
        deferred by default."""
        return [
            str(i)
            for i in self.experience_ids
            if not self.is_cited(str(i)) and str(i) not in self.triage
        ]

    def record_write(self, entry: Dict[str, Any]) -> None:
        """A create or edit that went through (its ``write_log`` entry)."""
        self.write_log.append(dict(entry))

    def untriaged(self) -> List[str]:
        """Shown ids neither cited nor triaged, in shown order (they count
        as deferred)."""
        return [
            i for i in self.shown_ids if not self.is_cited(i) and i not in self.triage
        ]

    def triage_counts(self) -> Dict[str, int]:
        """How many shown ids the triage lists as covered, not worth and
        deferred — a cited id left out: a citation wins over its triage (it
        counts as used), so it would be counted twice."""
        counts = {"covered": 0, "not_worth": 0, "deferred": 0}
        for experience_id, kind in self.triage.items():
            if self.is_cited(experience_id):
                continue
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def reject(self, tool: str, kind: str, message: str) -> str:
        """Record a rejected tool call (it costs nothing) and return the
        message the model sees."""
        self.rejected.append({"tool": tool, "kind": kind})
        return message

    def budget_message(self) -> str:
        return (
            f"Edit budget exhausted: this batch allows {self.budget} create/edit "
            "operation(s) (one per experience) and all were used. Nothing was "
            "changed. List the experiences no change cites with "
            "experience_triage, then call finish_memory_update() alone to end "
            "the batch."
        )

    def charge(self, kind: str, skill_id: str) -> None:
        """A create or edit went through: it costs 1."""
        self.spent += 1
        target = self.created if kind == "create" else self.edited
        if skill_id not in target:
            target.append(skill_id)

    def is_free_followup(self, skill_id: str, field_name: str) -> bool:
        """Whether an edit of ``field_name`` of ``skill_id`` costs no budget:
        a retrieval-text field of a skill whose instructions an edit of this
        batch changed — or an earlier, returned attempt on its experiences
        (``prior_instructions_edits``) — not yet changed for free (once per
        field)."""
        if field_name not in RETRIEVAL_TEXT_FIELDS:
            return False
        if (skill_id, field_name) in self.followup_fields:
            return False
        if skill_id in self.prior_instructions_edits:
            return True
        return any(
            entry.get("tool") == "skill_edit"
            and entry.get("field") == "instructions"
            and entry.get("skill_id") == skill_id
            for entry in self.write_log
        )

    def record_free_followup(self, skill_id: str, field_name: str) -> None:
        """A free retrieval-text follow-up went through (no budget spent)."""
        self.retrieval_followups += 1
        self.followup_fields.append((skill_id, field_name))
        if skill_id not in self.edited:
            self.edited.append(skill_id)

    def record_delete(self, skill_id: str) -> None:
        self.run_limits.deletes_used += 1
        if skill_id not in self.deleted:
            self.deleted.append(skill_id)

    def record_capped_output(self, omitted: int) -> None:
        if omitted > 0:
            self.outputs_capped += 1
            self.output_chars_omitted += omitted

    def changes(self) -> Dict[str, List[str]]:
        return {
            "created": list(self.created),
            "edited": [s for s in self.edited if s not in self.created],
            "deleted": list(self.deleted),
        }

    def rejected_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.rejected:
            counts[item["kind"]] = counts.get(item["kind"], 0) + 1
        return counts
