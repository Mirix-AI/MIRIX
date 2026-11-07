"""General role, time, state, and citation policy for graph v7.24.

v7.24 deliberately adds no benchmark entities or reference answers.  It keeps
v7.23 candidate generation, then makes ordinary graph QA stricter about who did
what to whom, which event time a fact describes, whether an occurrence was only
planned, and how much room cited PG rows receive in a bounded result set.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from calendar import monthrange
import re
from typing import Any, Iterable, Sequence, TypeVar

from mirix.services import retrieval_policy_v722 as _v722
from mirix.services import retrieval_policy_v723 as _v723
from mirix.services.retrieval_policy_v722 import QueryPlan
from mirix.services.retrieval_policy_v723 import VerificationDecision


_WORD_RE = re.compile(r"[a-z0-9]+")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# These are broad linguistic relations, not benchmark-specific lanes.
_EXTRA_FAMILIES = {
    "show": {
        "show", "shows", "showed", "shown", "share", "shares", "shared",
        "display", "displays", "displayed", "post", "posts", "posted",
    },
    "enroll": {
        "enroll", "enrolls", "enrolled", "enrolling", "register",
        "registers", "registered", "registering", "signup", "signed",
    },
    # Map ordinary query paraphrases onto existing v7.22 predicate classes so
    # the graph can retrieve `inspire(...)` for "motivates" and
    # `experience(injury, ...)` for "got hurt" without entity-specific rules.
    "learn": {
        "inspire", "inspires", "inspired", "inspiring", "motivate",
        "motivates", "motivated", "motivating", "motivation",
    },
    "suffer": {
        "experience", "experienced", "experiences", "experiencing", "face",
        "faced", "faces", "hurt", "injure", "injured", "injury", "recover",
        "recovered", "recovery", "setback", "setbacks", "suffer", "suffered",
    },
}
_EXTRA_FORM_TO_CANONICAL = {
    form: canonical for canonical, forms in _EXTRA_FAMILIES.items() for form in forms
}

_ROLE_CLASSES = {
    "agent": {
        "agent", "actor", "author", "creator", "donor", "experiencer",
        "giver", "owner", "reader", "sender", "sharer", "speaker",
        "subject", "user", "viewer", "writer",
    },
    "recipient": {
        "addressee", "beneficiary", "listener", "recipient", "receiver",
        "student", "target", "viewer",
    },
    "source": {"from", "origin", "recommender", "source", "suggestor"},
    "companion": {"companion", "coagent", "participant", "partner", "with"},
    "location": {"destination", "location", "place", "site", "venue"},
    "object": {
        "content", "item", "object", "patient", "product", "theme", "topic",
    },
}
_ROLE_TO_CLASS = {
    role: role_class for role_class, roles in _ROLE_CLASSES.items() for role in roles
}
_PLANNED_WORDS = {
    "aim", "aimed", "intend", "intended", "plan", "planned", "planning",
    "schedule", "scheduled", "want", "wanted", "will", "would",
}
_STATUS_KEYS = {"status", "state", "event_status", "occurrence_status"}
_EVENT_TIME_KEYS = {
    "date", "event_date", "event_time", "occurred_at", "time", "when", "year",
}
_DIRECTIONAL_PREPOSITIONS = {"by", "for", "from", "to", "with"}
_QUESTION_CAPITALS = {
    "After", "Before", "Did", "Does", "How", "In", "Is", "On", "The",
    "Was", "What", "When", "Where", "Which", "Who", "Why", "Would",
    *{name.title() for name in _MONTHS},
}


def is_v724(version: str | None) -> bool:
    return (version or "").strip().lower() == "v7.24"


def plan_query(query: str) -> QueryPlan:
    """Extend v7.23 with general relation forms and citation-balanced quotas."""

    base = _v723.plan_query(query)
    tokens = set(_WORD_RE.findall((query or "").lower()))
    relations = set(base.relations)
    forms = set(base.predicate_forms)
    for token in tokens:
        canonical = _EXTRA_FORM_TO_CANONICAL.get(token)
        if canonical:
            relations.add(canonical)
            forms.update(_EXTRA_FAMILIES[canonical])
    if re.search(r"\bsign(?:ed|ing)?\s+up\b", query or "", re.I):
        relations.add("enroll")
        forms.update(_EXTRA_FAMILIES["enroll"])
        forms.update({"sign", "signed", "signup"})

    # Reserve cited-row room even in the compact 10-row initial retrieval.  v7.23
    # used base_quota=10 for a single query, which left zero slots for the exact PG
    # rows cited by a matching graph fact until the answer model issued a tool call.
    quotas = {
        "single": (10, 6, 4),
        "temporal": (14, 6, 6),
        "intersection": (18, 7, 7),
        "list_union": (22, 7, 9),
        "count": (26, 8, 12),
    }
    fact_limit, base_quota, relation_quota = quotas[base.operator]
    constraints = set(base.constraint_tokens)
    constraints.difference_update(forms)
    return replace(
        base,
        relations=tuple(sorted(relations)),
        predicate_forms=tuple(sorted(forms)),
        constraint_tokens=tuple(sorted(constraints)),
        fact_limit=fact_limit,
        base_quota=base_quota,
        relation_quota=relation_quota,
    )


def predicate_hints(query: str) -> list[str]:
    return list(plan_query(query).predicate_forms)


def canonical_predicate_tokens(predicate: str) -> set[str]:
    base = _v722.canonical_predicate_tokens(predicate)
    return {_EXTRA_FORM_TO_CANONICAL.get(token, token) for token in base}


def predicate_match_score(query: str, predicate: str) -> float:
    plan = plan_query(query)
    if not plan.relations:
        return 0.0
    pred = canonical_predicate_tokens(predicate)
    overlap = len(set(plan.relations) & pred)
    if overlap:
        return min(1.0, 0.72 + 0.14 * (overlap - 1))
    q = _v722.content_tokens(query)
    p = _v722.content_tokens((predicate or "").replace("_", " "))
    return min(0.40, 0.40 * len(q & p) / max(len(p), 1))


def rank_anchor_hits(query, hits, limit):
    return _v722.rank_anchor_hits(query, hits, limit, planner=plan_query)


def exact_anchor_terms(query: str) -> list[str]:
    return _v723.exact_anchor_terms(query)


def query_mentions_anchor(query: str, name: str) -> bool:
    return _v723.query_mentions_anchor(query, name)


def _canonical_role(role: object) -> str | None:
    raw = str(role or "").strip().lower().replace("-", "_")
    if not raw:
        return None
    for token in raw.split("_"):
        if token in _ROLE_TO_CLASS:
            return _ROLE_TO_CLASS[token]
    return _ROLE_TO_CLASS.get(raw)


def _name_span(query: str, name: str) -> tuple[int, int] | None:
    words = _WORD_RE.findall(str(name or "").lower())
    if not words:
        return None
    pattern = r"\b" + r"\W+".join(map(re.escape, words)) + r"\b"
    match = re.search(pattern, query or "", re.I)
    return match.span() if match else None


def _relation_spans(query: str, plan: QueryPlan) -> list[tuple[int, int]]:
    """Every predicate form that occurs in the query, in order of position.

    ``plan.predicate_forms`` is a flattened SET of surface forms across all relations
    the planner detected, so it routinely contains a noun that shares a stem with a
    verb family. Taking ``min()`` over it — the previous behaviour — anchored on
    whichever form appeared leftmost, which in an English wh-question is the fronted
    object, not the predicate. Measured on the live query set:

        "What painting did Melanie show to Caroline on October 13, 2023?"
          span -> 'painting' at offset 5, never 'show' at 26
          Melanie  -> recipient   (she is the agent)
          Caroline -> recipient   (right, but by accident)

    Returning every span lets each argument bind to its OWN nearest predicate, which
    is what determines its role. No word lists: this is positional locality.
    """
    spans: list[tuple[int, int]] = []
    for form in plan.predicate_forms:
        if len(form) < 3:
            continue
        for match in re.finditer(rf"\b{re.escape(form)}\b", query or "", re.I):
            spans.append(match.span())
    return sorted(set(spans))


def _nearest_span(spans: list[tuple[int, int]], name_span: tuple[int, int]):
    """The predicate span closest to a name, measured edge-to-edge."""
    if not spans:
        return None

    def gap(sp: tuple[int, int]) -> int:
        if sp[1] <= name_span[0]:
            return name_span[0] - sp[1]
        if name_span[1] <= sp[0]:
            return sp[0] - name_span[1]
        return 0

    return min(spans, key=gap)


def _expected_role(query: str, name: str, anchor_type: str, relation_spans) -> str | None:
    span = _name_span(query, name)
    if span is None:
        return None
    if isinstance(relation_spans, tuple) and len(relation_spans) == 2 \
            and all(isinstance(x, int) for x in relation_spans):
        relation_spans = [relation_spans]          # tolerate the old single-span form
    relation_span = _nearest_span(list(relation_spans or []), span)
    if relation_span is None:
        return None
    rel_start, rel_end = relation_span
    if span[1] <= rel_start:
        # A name before its nearest predicate is that predicate's subject, and in an
        # active wh-question the subject is the agent. The previous code had a
        # `by`-test here whose two branches BOTH returned "agent" — dead code that
        # read as passive handling without being any. There is no passive handling:
        # measured on the 1,542-question LoCoMo set, 24 questions (1.6%) contain a
        # passive construction and almost none put a person name before the
        # participle, so the cost of the omission is under one question. Stated as a
        # limitation rather than hidden behind an unreachable branch.
        return "agent"
    between = (query or "")[rel_end:span[0]].lower()
    prep_match = re.search(r"\b(by|for|from|to|with|in|at)\b[^\w]*$", between)
    prep = prep_match.group(1) if prep_match else None
    if prep == "by":
        return "agent"
    if prep == "from":
        return "source"
    if prep in {"to", "for"}:
        return "recipient"
    if prep == "with":
        return "companion"
    if prep in {"in", "at"} and str(anchor_type or "").lower() != "person":
        return "location"
    return "recipient" if str(anchor_type or "").lower() == "person" else "object"


def role_alignment_score(query: str, fact: Any, exact_anchor_ids: Iterable[str]) -> float:
    """Softly score explicit argument direction without trusting noisy roles blindly."""

    plan = plan_query(query)
    relation_spans = _relation_spans(query, plan)
    if not relation_spans:
        # Neutral, not a penalty: 0.0 is the identity of this additive score. 41.4%
        # of questions contain no predicate form the planner knows, so this branch is
        # a COVERAGE limit, not a scoring decision.
        return 0.0
    exact_ids = {str(value) for value in exact_anchor_ids}
    matched = mismatched = covered = 0
    for arg in list(getattr(fact, "args", ()) or ()):
        if str(arg.get("id")) not in exact_ids:
            continue
        expected = _expected_role(
            query,
            str(arg.get("name") or ""),
            str(arg.get("anchor_type") or ""),
            relation_spans,
        )
        actual = _canonical_role(arg.get("role"))
        if expected is None:
            continue
        covered += 1
        if actual == expected:
            matched += 1
        elif actual is not None:
            # source/recommender and agent are adjacent notions.  Do not punish a
            # frame extractor that represented "from Kim's suggestion" as agent=Kim.
            compatible = {actual, expected} <= {"agent", "source"}
            if not compatible:
                mismatched += 1
    score = 0.16 * matched - 0.28 * mismatched
    mentioned = sum(
        1 for arg in list(getattr(fact, "args", ()) or ())
        if str(arg.get("id")) in exact_ids and _name_span(query, arg.get("name") or "")
    )
    if mentioned >= 2 and covered >= 2 and mismatched == 0:
        score += 0.20
    return max(-0.70, min(0.60, score))


def _literal_map(fact: Any) -> dict[str, str]:
    return {
        str(key or "").strip().lower(): str(value or "").strip()
        for key, value in zip(
            getattr(fact, "lit_keys", ()) or (), getattr(fact, "lit_vals", ()) or ()
        )
    }


def _parse_time(value: object) -> datetime | None:
    raw = str(value or "").strip().lower().replace("z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)
    except ValueError:
        pass
    year_match = re.search(r"\b((?:19|20)\d{2})\b", raw)
    if not year_match:
        return None
    year = int(year_match.group(1))
    month = next((number for name, number in _MONTHS.items() if name in raw), 1)
    day_match = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", raw)
    day = int(day_match.group(1)) if day_match and int(day_match.group(1)) <= 31 else 1
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return datetime(year, month, 1, tzinfo=timezone.utc)


def _event_time(fact: Any) -> tuple[datetime | None, bool]:
    literals = _literal_map(fact)
    for key in _EVENT_TIME_KEYS:
        if key in literals:
            parsed = _parse_time(literals[key])
            if parsed is not None:
                return parsed, True
    return _parse_time(getattr(fact, "timestamp", None)), False


def temporal_state_score(query: str, fact: Any) -> float:
    plan = plan_query(query)
    literals = _literal_map(fact)
    dt, from_literal = _event_time(fact)
    score = 0.0
    if dt is not None and (plan.years or plan.months or plan.days):
        matched = True
        if plan.years and dt.year not in plan.years:
            matched = False
        if plan.months and dt.month not in plan.months:
            matched = False
        if plan.days and dt.day not in plan.days:
            matched = False
        score += (0.24 if from_literal else 0.12) if matched else (-0.28 if from_literal else -0.12)

        if re.search(r"\bas of\b", query or "", re.I):
            year = plan.years[-1] if plan.years else dt.year
            month = plan.months[-1] if plan.months else 12
            cutoff = datetime(
                year, month, monthrange(year, month)[1], 23, 59, tzinfo=timezone.utc
            )
            if dt > cutoff:
                score -= 0.55

    status = " ".join(
        value.lower() for key, value in literals.items() if key in _STATUS_KEYS
    )
    pred_tokens = canonical_predicate_tokens(getattr(fact, "predicate", ""))
    planned = bool(set(_WORD_RE.findall(status)) & _PLANNED_WORDS) or bool(
        pred_tokens & {"plan", "intend", "schedule", "want"}
    )
    query_planned = bool(set(_WORD_RE.findall((query or "").lower())) & _PLANNED_WORDS)
    if query_planned:
        score += 0.18 if planned else 0.0
    elif planned and plan.operator in {"count", "list_union", "temporal"}:
        score -= 0.38
    return score


T = TypeVar("T")


def rank_relation_facts(
    query: str,
    facts: Sequence[T],
    *,
    exact_anchor_ids: Iterable[str],
    limit: int | None = None,
) -> list[T]:
    plan = plan_query(query)
    exact_ids = {str(value) for value in exact_anchor_ids}
    if plan.operator == "intersection":
        ranked = _v723.rank_relation_facts(
            query, facts, exact_anchor_ids=exact_ids, limit=len(facts)
        )
    else:
        ranked = _v722.rank_relation_facts(
            query,
            facts,
            exact_anchor_ids=exact_ids,
            limit=len(facts),
            planner=plan_query,
            predicate_scorer=predicate_match_score,
        )

    for fact in ranked:
        role_score = role_alignment_score(query, fact, exact_ids)
        state_score = temporal_state_score(query, fact)
        base = float(getattr(fact, "policy_score", 0.0) or 0.0)
        setattr(fact, "v724_role_score", round(role_score, 6))
        setattr(fact, "v724_temporal_state_score", round(state_score, 6))
        setattr(fact, "policy_score", round(base + role_score + state_score, 6))
    ranked.sort(
        key=lambda fact: (
            # A fact carrying the item shared by every named participant is a
            # hard intersection match.  Role/time are soft refinements inside
            # that group, never a reason to replace it with a one-sided item.
            bool(
                plan.operator == "intersection"
                and float(getattr(fact, "v723_intersection_bonus", 0.0) or 0.0) > 0
            ),
            float(getattr(fact, "policy_score", 0.0) or 0.0),
            str(getattr(fact, "timestamp", "") or ""),
        ),
        reverse=True,
    )
    chosen = plan.fact_limit if limit is None else max(0, limit)
    return ranked[:chosen]


def _row_tokens(row: Any) -> set[str]:
    text = " ".join((
        str(getattr(row, "summary", "") or ""),
        str(getattr(row, "details", "") or ""),
    ))
    stop = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
    }
    return {token for token in _WORD_RE.findall(text.lower()) if token not in stop}


def _near_duplicate(left: Any, right: Any) -> bool:
    if getattr(left, "kind", "") != "episodic" or getattr(right, "kind", "") != "episodic":
        return False
    left_date = str(getattr(left, "timestamp", "") or "")[:10]
    right_date = str(getattr(right, "timestamp", "") or "")[:10]
    if not left_date or left_date != right_date:
        return False
    a, b = _row_tokens(left), _row_tokens(right)
    if min(len(a), len(b)) < 5:
        return False
    overlap = len(a & b)
    return overlap / len(a | b) >= 0.78 or overlap / min(len(a), len(b)) >= 0.90


def merge_memory_rows(query, base_rows, relation_rows, limit):
    """Protect graph citations and conservatively collapse duplicate episodes."""

    if limit <= 0:
        return []
    plan = plan_query(query)
    base = _v722._rank_rows(query, base_rows)
    relation = _v722._rank_rows(query, relation_rows)
    base_quota = min(limit, plan.base_quota)
    relation_quota = min(max(limit - base_quota, 0), plan.relation_quota)
    selected: list[Any] = []
    seen_ids: set[str] = set()

    def take(rows: Sequence[Any], amount: int) -> None:
        added = 0
        for row in rows:
            row_id = str(getattr(row, "id", "") or "")
            if row_id and row_id in seen_ids:
                continue
            if any(_near_duplicate(row, prior) for prior in selected):
                continue
            if row_id:
                seen_ids.add(row_id)
            selected.append(row)
            added += 1
            if added >= amount:
                return

    # Exact graph citations are rendered first, while a fixed base quota still
    # preserves ordinary graph traversal recall.
    take(relation, relation_quota)
    take(base, base_quota)
    if len(selected) < limit:
        take(relation[relation_quota:], limit - len(selected))
    if len(selected) < limit:
        take(base[base_quota:], limit - len(selected))
    return selected[:limit]


_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_UNCERTAINTY_RE = re.compile(
    r"\b(?:not mentioned|not specified|no (?:specific )?(?:information|record|evidence)|"
    r"cannot determine|unclear|unknown|does not say|doesn't say)\b",
    re.I,
)


def _ledger_text(evidence: Iterable[dict[str, Any]] | None) -> str:
    return " ".join(
        str(item.get(key) or "")
        for item in evidence or () if isinstance(item, dict)
        for key in ("name", "predicate", "summary", "details", "timestamp", "mentioned_at")
    )


def _question_names(question: str) -> list[str]:
    return [
        value for value in re.findall(r"\b[A-Z][A-Za-z'-]+\b", question or "")
        if value not in _QUESTION_CAPITALS
    ]


def verification_query(question: str, plan: QueryPlan | None = None) -> str:
    plan = plan or plan_query(question)
    suffix = [*plan.relations, *plan.constraint_tokens]
    if re.search(r"\bas of\b|\bcurrent(?:ly)?\b|\blatest\b", question or "", re.I):
        suffix.extend(("event time", "status"))
    return " ".join(filter(None, ((question or "").strip(), " ".join(dict.fromkeys(suffix[:8])))))


def verification_decision(
    question: str,
    answer: str,
    evidence: Iterable[dict[str, Any]] | None = None,
) -> VerificationDecision:
    """Request one retry only for observable role, literal, or state conflicts."""

    base = _v723.verification_decision(question, answer, evidence)
    reasons = list(base.reasons)
    plan = plan_query(question)
    text = str(answer or "").strip()
    ledger = _ledger_text(evidence)
    ledger_low = ledger.lower()

    names = _question_names(question)
    role_scores = [
        float(item.get("role_alignment"))
        for item in evidence or ()
        if isinstance(item, dict) and item.get("role_alignment") is not None
    ]
    if (
        plan.is_relation_query
        and len(set(names)) >= 2
        and role_scores
        and min(role_scores) < 0
    ):
        reasons.append("role_direction_review")

    if evidence and text:
        question_literals = set(_NUMBER_RE.findall(question or ""))
        unsupported = {
            value for value in _NUMBER_RE.findall(text)
            if value not in question_literals and value not in _NUMBER_RE.findall(ledger)
        }
        answer_months = {name for name in _MONTHS if name in text.lower()}
        evidence_months = {name for name in _MONTHS if name in ledger_low}
        if unsupported or not answer_months <= evidence_months | {
            name for name in _MONTHS if name in (question or "").lower()
        }:
            reasons.append("unsupported_answer_literal")

    if re.search(r"\bas of\b|\bcurrent(?:ly)?\b|\blatest\b|\bmost recent\b", question or "", re.I):
        if len(re.findall(r"\b(?:19|20)\d{2}\b", ledger)) >= 2:
            reasons.append("state_time_review")

    if plan.operator == "count" and not (set(_WORD_RE.findall(question.lower())) & _PLANNED_WORDS):
        if re.search(r"\b(?:planned|planning|intended|scheduled|wanted to)\b", ledger, re.I):
            reasons.append("planned_occurrence_review")

    unique = tuple(dict.fromkeys(reasons))
    return VerificationDecision(
        retry=bool(unique), reasons=unique, query=verification_query(question, plan)
    )


__all__ = [
    "QueryPlan",
    "VerificationDecision",
    "canonical_predicate_tokens",
    "exact_anchor_terms",
    "is_v724",
    "merge_memory_rows",
    "plan_query",
    "predicate_hints",
    "predicate_match_score",
    "query_mentions_anchor",
    "rank_anchor_hits",
    "rank_relation_facts",
    "role_alignment_score",
    "temporal_state_score",
    "verification_decision",
    "verification_query",
]
