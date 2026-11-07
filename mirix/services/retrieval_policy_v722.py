"""General query-planning and evidence-ranking policy for graph v7.22.

v7.22 is deliberately benchmark-agnostic.  It keeps v7.20 ingest and v7.19
AutoDream unchanged, then improves the v7.21 relation lane in three ways:

* query and stored predicates share one small linguistic normalizer;
* facts are ranked by relation, named entities, object/literal constraints and
  temporal qualifiers instead of relation alone;
* graph-traversal evidence and relation-cited evidence receive separate quotas,
  so the precision lane cannot erase the recall lane.

All inputs to this policy have already been reached through Neo4j.  Nothing in
this module performs or authorizes an unbounded PostgreSQL search.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence, TypeVar

from mirix.services.retrieval_policy_v720 import (
    anchor_policy_score,
    content_tokens,
    evidence_lexical_score,
)
from mirix.services.retrieval_policy_v721 import (
    exact_anchor_terms,
    query_mentions_anchor,
)


_WORD_RE = re.compile(r"[a-z0-9]+")
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_QUERY_GLUE = {
    "a", "an", "and", "are", "as", "at", "be", "been", "both", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "her",
    "his", "how", "in", "is", "it", "many", "much", "of", "on", "or",
    "she", "the", "their", "them", "they", "to", "total", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}
_GENERIC_ANSWER_TYPES = {
    "activity", "activities", "book", "books", "city", "cities", "country",
    "countries", "event", "events", "food", "foods", "game", "games",
    "hobby", "hobbies", "instrument", "instruments", "item", "items",
    "location", "locations", "movie", "movies", "person", "people", "place",
    "places", "plan", "plans", "song", "songs", "subject", "subjects",
    "thing", "things", "time", "times", "type", "types",
}


# Families describe ordinary linguistic relations, never benchmark entities or
# answers.  ``triggers`` are intentionally more conservative than ``forms``:
# auxiliary-like words may match stored predicates without opening a person hub
# by themselves in the question.
_FAMILIES = (
    ("buy", {"buy", "buys", "buying", "bought", "purchase", "purchases", "purchased", "purchasing", "acquire", "acquired", "obtain", "obtained"}),
    ("create", {"create", "creates", "created", "creating", "make", "makes", "made", "paint", "paints", "painted", "painting", "draw", "draws", "drew", "drawing", "build", "builds", "built", "write", "writes", "wrote", "written", "produce", "produced"}),
    ("read", {"read", "reads", "reading", "finish", "finishes", "finished", "study", "studies", "studied", "studying"}),
    ("recommend", {"recommend", "recommends", "recommended", "recommendation", "recommendations", "suggest", "suggests", "suggested", "suggestion", "suggestions", "advise", "advised"}),
    ("plan", {"plan", "plans", "planned", "planning", "intend", "intends", "intended", "research", "researches", "researched", "investigate", "investigated", "apply", "applies", "applied", "prepare", "prepares", "prepared"}),
    ("attend", {"attend", "attends", "attended", "attending", "visit", "visits", "visited", "visiting", "go", "goes", "went", "gone", "going", "join", "joins", "joined", "participate", "participates", "participated"}),
    ("locate", {"live", "lives", "lived", "living", "locate", "locates", "located", "move", "moves", "moved", "moving", "stay", "stays", "stayed", "reside", "resides", "resided"}),
    ("work", {"work", "works", "worked", "working", "employ", "employs", "employed", "job", "jobs"}),
    ("like", {"like", "likes", "liked", "love", "loves", "loved", "prefer", "prefers", "preferred", "favorite", "favourite", "enjoy", "enjoys", "enjoyed"}),
    ("possess", {"have", "has", "had", "own", "owns", "owned", "include", "includes", "included", "contain", "contains", "contained", "consist", "consists", "consisted"}),
    ("communicate", {"meet", "meets", "met", "contact", "contacts", "contacted", "talk", "talks", "talked", "speak", "speaks", "spoke", "spoken", "message", "messages", "messaged"}),
    ("learn", {"learn", "learns", "learned", "learnt", "teach", "teaches", "taught", "inspire", "inspires", "inspired", "influence", "influences", "influenced"}),
    ("play", {"play", "plays", "played", "playing", "practice", "practices", "practiced", "perform", "performs", "performed"}),
    ("interest", {"interest", "interests", "interested", "fascinate", "fascinates", "fascinated"}),
    ("listen", {"listen", "listens", "listened", "listening", "hear", "hears", "heard"}),
    ("adopt", {"adopt", "adopts", "adopted", "adopting"}),
    ("travel", {"travel", "travels", "traveled", "travelled", "traveling", "travelling"}),
    ("win", {"win", "wins", "won", "winning", "victory", "victories"}),
    ("lose", {"lose", "loses", "lost", "losing"}),
    ("start", {"start", "starts", "started", "starting", "begin", "begins", "began", "begun", "launch", "launches", "launched"}),
    ("complete", {"complete", "completes", "completed", "completing", "finish", "finishes", "finished", "finishing"}),
    ("receive", {"receive", "receives", "received", "receiving", "earn", "earns", "earned"}),
    ("suffer", {"suffer", "suffers", "suffered", "suffering", "face", "faces", "faced"}),
    ("use", {"use", "uses", "used", "using"}),
    ("watch", {"watch", "watches", "watched", "watching", "see", "sees", "saw", "seen"}),
    ("give", {"give", "gives", "gave", "given", "donate", "donates", "donated", "send", "sends", "sent"}),
    ("cook", {"cook", "cooks", "cooked", "cooking", "eat", "eats", "ate", "eaten"}),
    ("volunteer", {"volunteer", "volunteers", "volunteered", "volunteering", "mentor", "mentors", "mentored"}),
    ("engage", {"engage", "engages", "engaged", "engaging", "pursue", "pursues", "pursued", "pursuing"}),
    ("born", {"born", "birth", "birthday", "age", "aged"}),
)

_FORM_TO_CANONICAL = {
    form: canonical for canonical, forms in _FAMILIES for form in forms
}
_CANONICAL_TO_FORMS = {canonical: set(forms) for canonical, forms in _FAMILIES}

# Forms that are useful when matching an already-selected graph predicate but
# too ambiguous to open a relation lane on their own.
_WEAK_QUERY_TRIGGERS = {
    "face", "faces", "faced", "get", "got",
    "had", "has", "have", "job", "jobs", "make", "makes", "made", "see",
    "sees", "saw", "seen",
}


@dataclass(frozen=True)
class QueryPlan:
    operator: str
    relations: tuple[str, ...]
    predicate_forms: tuple[str, ...]
    constraint_tokens: tuple[str, ...]
    answer_types: tuple[str, ...]
    temporal_mode: str | None
    years: tuple[int, ...]
    months: tuple[int, ...]
    days: tuple[int, ...]
    fact_limit: int
    base_quota: int
    relation_quota: int

    @property
    def is_relation_query(self) -> bool:
        return bool(self.relations)

    @property
    def preserve_raw_query(self) -> bool:
        """Whether relation parsing is selective enough to replace a tool rewrite.

        Generic ``use`` questions benefit from exact-anchor graph entry but often
        have no matching use(...) fact for the concrete object.  Let the answer
        model issue a narrower follow-up query in that case.
        """

        return bool(self.relations) and set(self.relations) != {"use"}


def is_v722(version: str | None) -> bool:
    return (version or "").strip().lower() == "v7.22"


def _operator(query: str) -> str:
    low = (query or "").lower()
    if re.search(r"\bhow many\b|\bnumber of\b|\bhow much\b|\btotal\b", low):
        return "count"
    if re.search(r"\b(?:both|in common|common to|shared by)\b", low):
        return "intersection"
    if re.search(
        r"\bwhat (?:items|books|activities|things|instruments|events|kinds|types|"
        r"paintings|subjects|places|plans|cities|countries|movies|songs|hobbies)\b",
        low,
    ):
        return "list_union"
    if re.search(
        r"\b(?:when|before|after|latest|most recent|currently|current|first|earliest)\b",
        low,
    ):
        return "temporal"
    return "single"


def _temporal_mode(query: str) -> str | None:
    low = (query or "").lower()
    if re.search(r"\b(?:first|earliest|initially)\b", low):
        return "earliest"
    if re.search(r"\b(?:latest|most recent|currently|current|recently)\b", low):
        return "latest"
    if re.search(r"\bbefore\b", low):
        return "before"
    if re.search(r"\bafter\b", low):
        return "after"
    if re.search(r"\bwhen\b", low):
        return "when"
    return None


def _date_constraints(query: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    low = (query or "").lower()
    years = tuple(sorted({int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", low)}))
    months = tuple(sorted({number for name, number in _MONTHS.items() if re.search(rf"\b{name}\b", low)}))
    days: set[int] = set()
    for match in re.finditer(
        r"\b(?:" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b|"
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:" + "|".join(_MONTHS) + r")\b",
        low,
    ):
        value = match.group(1) or match.group(2)
        if value and 1 <= int(value) <= 31:
            days.add(int(value))
    return years, months, tuple(sorted(days))


def plan_query(query: str) -> QueryPlan:
    raw_tokens = set(_WORD_RE.findall((query or "").lower()))
    relations = {
        canonical
        for token in raw_tokens
        if token not in _WEAK_QUERY_TRIGGERS
        for canonical in [_FORM_TO_CANONICAL.get(token)]
        if canonical
    }
    # Possession is allowed only for an explicit base "have" in a genuine
    # attribute question; inflected auxiliaries must not open a person hub.
    if "have" in raw_tokens and re.search(r"\b(?:what|which)\b", query or "", re.I):
        relations.add("possess")
    # A weak verb may join a plan already opened by a stronger relation, which
    # helps stored predicates without letting generic queries trigger alone.
    if relations:
        relations.update(
            canonical for token in raw_tokens
            if (canonical := _FORM_TO_CANONICAL.get(token))
        )

    forms: set[str] = set()
    for relation in relations:
        forms.update(_CANONICAL_TO_FORMS.get(relation, {relation}))

    answer_types = tuple(sorted(raw_tokens & _GENERIC_ANSWER_TYPES))
    constraint_tokens = set(content_tokens(query))
    constraint_tokens.difference_update(forms)
    constraint_tokens.difference_update(_QUERY_GLUE)
    constraint_tokens.difference_update({str(value) for value in re.findall(r"\d+", query or "")})

    operator = _operator(query)
    budgets = {
        "single": (8, 10, 5),
        "temporal": (12, 9, 6),
        "intersection": (16, 9, 7),
        "list_union": (20, 10, 10),
        "count": (24, 10, 14),
    }
    fact_limit, base_quota, relation_quota = budgets[operator]
    years, months, days = _date_constraints(query)
    return QueryPlan(
        operator=operator,
        relations=tuple(sorted(relations)),
        predicate_forms=tuple(sorted(forms)),
        constraint_tokens=tuple(sorted(constraint_tokens)),
        answer_types=answer_types,
        temporal_mode=_temporal_mode(query),
        years=years,
        months=months,
        days=days,
        fact_limit=fact_limit,
        base_quota=base_quota,
        relation_quota=relation_quota,
    )


def predicate_hints(query: str) -> list[str]:
    """Surface forms used by the bounded indexed-adjacency Cypher query."""

    return list(plan_query(query).predicate_forms)


def canonical_predicate_tokens(predicate: str) -> set[str]:
    raw = set(_WORD_RE.findall((predicate or "").lower().replace("_", " ")))
    return {_FORM_TO_CANONICAL.get(token, token) for token in raw}


def predicate_match_score(query: str, predicate: str, *, planner=plan_query) -> float:
    plan = planner(query)
    if not plan.relations:
        return 0.0
    pred = canonical_predicate_tokens(predicate)
    overlap = len(set(plan.relations) & pred)
    if overlap:
        return min(1.0, 0.72 + 0.14 * (overlap - 1))
    # Preserve a small lexical backstop for a graph predicate not yet covered by
    # the relation ontology; it cannot outrank a canonical family match.
    q = content_tokens(query)
    p = content_tokens(predicate.replace("_", " "))
    return min(0.40, 0.40 * len(q & p) / max(len(p), 1))


T = TypeVar("T")


def rank_anchor_hits(query: str, hits: Sequence[T], limit: int, *, planner=plan_query) -> list[T]:
    """v7.20 anchor precision with explicit query entities protected."""

    relation_query = planner(query).is_relation_query
    scored = []
    for idx, hit in enumerate(hits):
        anchor_type = str(getattr(hit, "anchor_type", "") or "").lower()
        degree = int(getattr(hit, "degree", 0) or 0)
        score = anchor_policy_score(
            query=query,
            name=getattr(hit, "name", ""),
            anchor_type=anchor_type,
            cosine=getattr(hit, "cosine", 0.0),
            degree=degree,
            predicates=getattr(hit, "predicates", ()),
        )
        if relation_query and query_mentions_anchor(query, getattr(hit, "name", "")):
            score += 0.14
            if anchor_type == "person":
                score += min(0.24, 0.040 * math.log1p(max(degree, 0))) + 0.08
        setattr(hit, "policy_score", score)
        scored.append((score, -idx, hit))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:max(0, limit)]]


def _fact_datetime(fact: T) -> datetime | None:
    raw = str(getattr(fact, "timestamp", "") or "").replace("Z", "+00:00")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def rank_relation_facts(
    query: str,
    facts: Sequence[T],
    *,
    exact_anchor_ids: Iterable[str],
    limit: int | None = None,
    planner=plan_query,
    predicate_scorer=None,
) -> list[T]:
    """Rank bounded adjacent facts with generic relation and qualifier coverage."""

    plan = planner(query)
    exact_ids = {str(value) for value in exact_anchor_ids}
    exact_name_tokens: set[str] = set()
    for fact in facts:
        for arg in list(getattr(fact, "args", ()) or ()):
            if str(arg.get("id")) in exact_ids:
                exact_name_tokens.update(content_tokens(str(arg.get("name") or "")))
    non_object_constraints = {
        "ago", "childhood", "current", "currently", "day", "days", "first",
        "got", "last", "latest", "month", "months", "most", "new", "recent",
        "recently", "time", "times", "week", "weeks", "year", "years",
        *_MONTHS.keys(),
    }
    specific_constraints = set(plan.constraint_tokens)
    specific_constraints.difference_update(exact_name_tokens)
    specific_constraints.difference_update(plan.answer_types)
    specific_constraints.difference_update(non_object_constraints)
    dated = [_fact_datetime(fact) for fact in facts]
    valid_dates = [value for value in dated if value is not None]
    earliest = min(valid_dates) if valid_dates else None
    latest = max(valid_dates) if valid_dates else None

    shared_tokens: set[str] = set()
    if plan.operator == "intersection":
        token_owners: dict[str, set[str]] = {}
        for fact in facts:
            args = list(getattr(fact, "args", ()) or ())
            covered = {str(arg.get("id")) for arg in args if str(arg.get("id")) in exact_ids}
            for arg in args:
                if str(arg.get("id")) in exact_ids:
                    continue
                for token in content_tokens(str(arg.get("name") or "")):
                    if token not in {"art", "picture", "thing", "work"}:
                        token_owners.setdefault(token, set()).update(covered)
        shared_tokens = {token for token, owners in token_owners.items() if len(owners) >= 2}

    score_predicate = predicate_scorer or predicate_match_score
    scored = []
    for idx, fact in enumerate(facts):
        if predicate_scorer is None:
            pred_score = score_predicate(
                query, getattr(fact, "predicate", ""), planner=planner
            )
        else:
            pred_score = score_predicate(query, getattr(fact, "predicate", ""))
        if pred_score <= 0:
            continue
        args = list(getattr(fact, "args", ()) or ())
        covered = {str(arg.get("id")) for arg in args if str(arg.get("id")) in exact_ids}
        entity_score = min(0.22, 0.08 + 0.07 * len(covered))

        non_seed_args = [arg for arg in args if str(arg.get("id")) not in exact_ids]
        argument_text = " ".join(str(arg.get("name") or "") for arg in non_seed_args)
        literal_text = " ".join(
            f"{key} {value}" for key, value in zip(
                getattr(fact, "lit_keys", ()) or (), getattr(fact, "lit_vals", ()) or ()
            )
        )
        fact_tokens = content_tokens(" ".join((argument_text, literal_text)))
        constraints = set(plan.constraint_tokens)
        constraint_score = 0.18 * len(constraints & fact_tokens) / max(len(constraints), 1)
        specific_overlap = len(specific_constraints & fact_tokens)

        type_score = 0.0
        if plan.answer_types:
            arg_types = {
                str(arg.get("anchor_type") or "").lower() for arg in non_seed_args
            }
            singular_types = {value[:-1] if value.endswith("s") else value for value in plan.answer_types}
            if arg_types & singular_types:
                type_score = 0.08

        dt = dated[idx]
        temporal_score = 0.0
        if dt is not None:
            if plan.years:
                temporal_score += 0.10 if dt.year in plan.years else -0.06
            if plan.months:
                temporal_score += 0.07 if dt.month in plan.months else -0.04
            if plan.days:
                temporal_score += 0.05 if dt.day in plan.days else -0.03
            if plan.temporal_mode == "earliest" and earliest is not None and dt == earliest:
                temporal_score += 0.12
            elif plan.temporal_mode == "latest" and latest is not None and dt == latest:
                temporal_score += 0.12

        object_tokens = content_tokens(argument_text)
        intersection_score = 0.24 if object_tokens & shared_tokens else 0.0
        specificity = min(0.05, 0.01 * max(len(args) - 1, 0))
        score = (
            pred_score + entity_score + constraint_score + type_score
            + temporal_score + intersection_score + specificity
        )
        setattr(fact, "policy_score", round(score, 6))
        scored.append((score, len(covered), constraint_score, specific_overlap, -idx, fact))

    # A concrete object constraint with zero support means this relation family
    # opened the correct person hub but the wrong fact lane (for example shoes +
    # use(...) returning pottery facts). Surface no graph facts in that case; the
    # exact anchor still supplies its bounded PG citations as the recall fallback.
    if specific_constraints and scored and max(item[3] for item in scored) == 0:
        return []
    scored.sort(
        key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
        reverse=True,
    )
    chosen_limit = plan.fact_limit if limit is None else max(0, limit)
    return [item[5] for item in scored[:chosen_limit]]


def _rank_rows(query: str, rows: Sequence[T]) -> list[T]:
    total = max(len(rows), 1)
    scored = []
    seen: set[str] = set()
    for idx, row in enumerate(rows):
        row_id = str(getattr(row, "id", "") or "")
        if row_id and row_id in seen:
            continue
        if row_id:
            seen.add(row_id)
        extra = getattr(row, "extra", {}) or {}
        lexical = evidence_lexical_score(
            query,
            getattr(row, "summary", ""),
            getattr(row, "details", ""),
            str(extra.get("name") or ""),
        )
        vector_rank = 0.85 + 0.15 * (1.0 - (idx / total))
        score = 0.64 * vector_rank + 0.36 * lexical
        extra["v722_evidence_score"] = round(score, 6)
        setattr(row, "extra", extra)
        scored.append((score, -idx, row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


def merge_memory_rows(
    query: str,
    base_rows: Sequence[T],
    relation_rows: Sequence[T],
    limit: int,
    *,
    planner=plan_query,
) -> list[T]:
    """Merge two graph-owned lanes while protecting base-recall evidence."""

    if limit <= 0:
        return []
    plan = planner(query)
    base = _rank_rows(query, base_rows)
    relation = _rank_rows(query, relation_rows)
    base_quota = min(limit, plan.base_quota)
    relation_quota = min(max(limit - base_quota, 0), plan.relation_quota)

    selected: list[T] = []
    seen: set[str] = set()

    def take(rows: Sequence[T], amount: int) -> None:
        added = 0
        for row in rows:
            row_id = str(getattr(row, "id", "") or "")
            if row_id and row_id in seen:
                continue
            if row_id:
                seen.add(row_id)
            selected.append(row)
            added += 1
            if added >= amount:
                return

    take(base, base_quota)
    take(relation, relation_quota)
    if len(selected) < limit:
        take(base[base_quota:], limit - len(selected))
    if len(selected) < limit:
        take(relation, limit - len(selected))
    return selected[:limit]


__all__ = [
    "QueryPlan",
    "canonical_predicate_tokens",
    "exact_anchor_terms",
    "is_v722",
    "merge_memory_rows",
    "plan_query",
    "predicate_hints",
    "predicate_match_score",
    "query_mentions_anchor",
    "rank_anchor_hits",
    "rank_relation_facts",
]
