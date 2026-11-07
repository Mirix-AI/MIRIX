"""v7.12 n-ary frame extraction — the change that makes the hypergraph a hypergraph.

v7.10 called its V7Fact nodes hyperedges, but the extractor only ever produced
(subject, predicate, object), so every fact connected exactly two anchors —
measured on a real store: 534 of 534 facts had arity 2, zero had more. A reified
binary relation is not a hyperedge; the extra expressive power was never used
because nothing upstream could produce it.

This extracts FRAMES instead of triples. A frame is one predicate plus any number
of typed participants:

    "Caroline flew from Boston to Miami on 6 May with Delta"
      predicate: fly
      args:      agent=Caroline, origin=Boston, destination=Miami, carrier=Delta
      literals:  {"date": "6 May"}

Two deliberate splits:

* **Entities become args, literals become properties.** Dates, amounts, counts and
  durations are recorded ON the frame, not as anchors. In v7.10 they became anchors
  and polluted the entity space so badly that the merge guard needs explicit rules
  to keep "42 campsites" apart from "46 campsites". Values do not have an identity
  to resolve; they are attributes of an assertion.
* **Roles are open vocabulary.** A closed frame inventory (FrameNet-style) is not
  worth the extraction cost here; the roles only need to be consistent enough to
  query, and the prompt anchors the common ones.

A binary fact is just a frame with two args, so this strictly generalises v7.10 —
nothing that used to be extractable stops being extractable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from mirix.log import get_logger
from mirix.services.lightrag_extractor import ExtractedEntity, call_openai_chat
from mirix.services.triple_extractor import ENTITY_TYPES, _NOISE, _norm_type
from mirix.settings import settings

logger = get_logger(__name__)

# Backstop for the "<=4 words" prompt rule; generous enough that a real multi-word
# name ("Hampton Inn & Suites Miami Midtown") survives.
_MAX_ARG_WORDS = 7

# Above this width the pairwise projection degrades to a star — see as_relations.
_CLIQUE_MAX_ARITY = 4

# Versions that opt IN to role canonicalisation. Empty of the shipped line on purpose:
# v7.13 was the experiment, it was never measured end to end, and no later version has
# adopted it. Add a version here only together with the arm that measures it.
_ROLE_CANON_VERSIONS = frozenset({"v7.13"})

# Role canonicalisation. An open role vocabulary keeps extraction cheap, but it lets the
# model label the SAME relation differently in different memories, and frame_identity
# hashes role:anchor pairs — so "attend(event=X, attendee=Y)" and "attend(agent=Y,
# theme=X)" become two nodes for one fact. Measured on conv-26: 71 distinct roles, of
# which 11 covered 87.8% of arms and a 60-role tail covered 12.2%; 7.2% of all facts
# were duplicates differing only in role labelling. Folding the tail onto the head
# collapses those without giving up the distinctions that matter (origin vs destination
# stays a real difference — that is the whole point of the n-ary form).
_ROLE_CANON = {
    # who does it
    "attendee": "agent", "knower": "agent", "mover": "agent", "buyer": "agent",
    "actor": "agent", "speaker": "agent", "author": "agent", "creator": "agent",
    "owner": "agent", "participant": "agent", "member": "agent", "subject": "agent",
    # what it is about / what is affected. The merges here are evidence-driven, not
    # taxonomic: these are the pairs the extractor was measured swapping between two
    # labellings of ONE fact on conv-26 — patient<->theme 12 times, topic<->theme 7,
    # event<->theme 2, activity<->theme 2, stimulus<->theme 2. The distinctions that
    # earn the n-ary form (agent / origin / destination / recipient / instrument /
    # location) are deliberately left alone.
    "patient": "theme", "topic": "theme", "event": "theme", "activity": "theme",
    "stimulus": "theme",
    "known": "theme", "object": "theme", "content": "theme", "product": "theme",
    "possession": "theme", "meaning": "theme", "representation": "theme",
    "depict": "theme", "feeling": "theme", "behavior": "theme", "action": "theme",
    "field": "theme", "career": "theme", "program": "theme", "instance": "theme",
    "identity": "attribute",
    # who experiences it — agent<->experiencer swapped 3 times
    "experiencer": "agent",
    # who receives / benefits
    "beneficiary": "recipient", "addressee": "recipient", "audience": "recipient",
    # where / when
    "place": "location", "venue": "location", "setting": "location",
    "origin": "origin", "destination": "destination",
    # why / how
    "reason": "cause", "trigger": "cause",
    "means": "instrument", "tool": "instrument", "manner": "method",
    # who else
    "partner": "companion", "spouse": "companion", "friend": "companion",
    # structure
    "part": "part", "whole": "whole", "organization": "organization",
}

# Roles that are really MEASUREMENTS wearing a role's clothes. A date is not a
# participant — it has no identity to resolve and becomes an anchor nobody can merge.
# Measured on conv-26, 34 arms were "date"/"time" roles, i.e. 34 date anchors that the
# literal mechanism exists precisely to prevent.
_LITERAL_ROLES = {"date", "time", "when", "duration", "amount", "price", "count",
                  "quantity", "cost", "age", "year"}

# Suggested role vocabulary. Open — the model may coin a role when none fits —
# but naming the common ones keeps them consistent across memories, which is what
# makes them queryable ("every fact where X is the agent").
# "attribute" and "value" are deliberately ABSENT. With them in the list the model
# used "attribute" as a dumping ground for descriptive clauses — measured on a
# 12-memory probe it was the single most common role (17 of 67 arms) and its fillers
# were copied phrases like "fees for carry-on ($30-$45 online) and checked bags",
# which are not entities and can never be resolved, merged or asked about. Properties
# of a thing belong in literals; if the property names an entity, some other role fits.
_ROLE_HINTS = (
    "agent, patient, theme, recipient, origin, destination, location, "
    "instrument, method, purpose, cause, topic, companion, source, beneficiary"
)

FRAME_PROMPT = f"""Extract knowledge FRAMES from the text. A frame is ONE predicate plus every participant it relates.

Rules:
- predicate: a SHORT verb or relation phrase ("fly", "attend", "recommend", "located in").
- args: the ENTITIES taking part, each with a ROLE, a NAME and a TYPE.
  * Include EVERY participant the sentence gives — do not collapse a multi-participant
    event into several two-participant facts. One trip with an origin, a destination,
    a carrier and a companion is ONE frame with four args.
  * ROLE names describe the participant's part in the predicate. Prefer these when they
    fit: {_ROLE_HINTS}. Coin a clearer one only if none applies.
  * NAME is a named thing (person/place/org/product) or a CONCISE CANONICAL concept
    (2-4 words for the general theme, NOT a copied phrase):
      "misses watercooler chats with colleagues while remote" -> "Remote Team Networking"
  * TYPE is exactly one of: {", ".join(ENTITY_TYPES)}
- literals: dates, amounts, counts, durations, measurements, prices — as a flat
  key/value object ON the frame. Do NOT make these args; they are values, not entities.
  Keep the value VERBATIM ("500 Mbps", "$420", "7", "three months").
  * A quantity attached to an entity is a LITERAL, and the arg keeps the bare entity:
    "17 vintage cameras" -> arg name "vintage cameras", literals {{"count": "17"}}.
  * A literal key must name the MEASURE ("date", "count", "price", "duration",
    "speed"), never a purpose or a reason — those belong in a "purpose" arg if they
    name an entity, and are otherwise dropped.
- An arg NAME is one entity, AT MOST 4 WORDS, and must be something a question could
  be asked about. A descriptive clause is never an arg — "fees for carry-on bags are
  $30-$45 online" is not a participant; it is a literal ({{"carry_on_fee": "$30-$45"}})
  or nothing. If you cannot phrase the arg as a noun phrase somebody might search for,
  drop it.
- An arg NAME is one entity. If a phrase bundles an entity with its modifier
  ("dry box with silica gel"), emit the entity ("dry box") and, when the modifier is
  itself an entity, a second arg for it ("silica gel").

A frame with two args is fine when that is all the text supports. Prefer one rich
frame over several thin ones about the same event.

Return JSON only:
{{"frames":[{{"predicate":"...","args":[{{"role":"...","name":"...","type":"..."}}],
             "literals":{{"key":"value"}}}}]}}"""


@dataclass
class Frame:
    predicate: str
    args: List[Tuple[str, str]] = field(default_factory=list)   # (role, entity name)
    literals: dict = field(default_factory=dict)

    @property
    def arity(self) -> int:
        return len(self.args)


@dataclass
class FrameResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)

    def as_relations(self) -> List[Tuple[str, str, str]]:
        """Binary projection of the frames, for the anchor↔anchor relation edges.

        Every pair of args in a frame co-participates in it, so each pair gets an
        edge. For a 2-arg frame this reproduces v7.10's single relation exactly; for
        wider frames it links participants a triple extractor would never have
        connected (Boston↔Delta via the same flight).

        The clique is quadratic, so above ``_CLIQUE_MAX_ARITY`` it degrades to a STAR
        from the first arg. Wide frames in practice are list enumerations ("the card
        includes a sign-up bonus, earning rates, lounge access, ...") — a measured
        arity-11 frame would have projected 55 relation edges on its own, swamping the
        anchor graph with pairs the text never actually related to each other. The
        star keeps every participant reachable in one hop through the frame's head,
        and the frame node itself still records the full n-ary membership.
        """
        out: List[Tuple[str, str, str]] = []
        for fr in self.frames:
            names = [n for _r, n in fr.args]
            if len(names) <= _CLIQUE_MAX_ARITY:
                pairs = ((a, b) for i, a in enumerate(names) for b in names[i + 1:])
            else:
                head = names[0]
                pairs = ((head, b) for b in names[1:])
            for a, b in pairs:
                if a != b:
                    out.append((a, fr.predicate, b))
        return out


def _clean_literals(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        key = re.sub(r"[^a-z0-9_]", "_", str(k).strip().lower())[:40]
        if not key or v is None:
            continue
        val = str(v).strip()[:200]
        if val:
            out[key] = val
    return out


def _parse(raw: str, *, canon_roles: Optional[bool] = None) -> FrameResult:
    """Select the role vocabulary owned by the configured graph revision.

    v7.14 through v7.24 keep v7.12's open role vocabulary. v7.20+ enriches
    LoCoMo input and retrieval, but does not adopt the independent v7.13 role
    canonicalisation experiment.
    """
    if canon_roles is None:
        # OPT-IN, not opt-out. This was an exclusion list that every new version had
        # to remember to extend; it did not list v7.25, so the next version would have
        # silently switched canonicalisation ON and folded an unmeasured variable into
        # whatever else that version changed. Inverting it means a new version
        # inherits v7.24's behaviour by default and enabling canon is a deliberate act.
        #
        # Status: canonicalisation has NEVER run in a shipped version. The live
        # v7.23 store carries 342 distinct role strings, with patient(1,959),
        # topic(1,462) and content(496) sitting unmerged beside theme(5,244).
        canon_roles = settings.graph_version in _ROLE_CANON_VERSIONS
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return FrameResult()
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return FrameResult()

    ents: dict[str, ExtractedEntity] = {}
    frames: List[Frame] = []
    for fr in data.get("frames", []) or []:
        if not isinstance(fr, dict):
            continue
        pred = str(fr.get("predicate") or "").strip()
        if not pred:
            continue
        args: List[Tuple[str, str]] = []
        seen_roles: set = set()
        literal_spill: dict = {}
        for a in fr.get("args", []) or []:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or "").strip()
            role = re.sub(r"[^a-z0-9_]", "_", str(a.get("role") or "arg").strip().lower())[:32]
            if canon_roles:
                # spouse1/spouse2, meaning1/meaning2 — the model numbers symmetric
                # roles. The number carries nothing the arg name does not already have.
                role = re.sub(r"\d+$", "", role) or "arg"
                if role in _LITERAL_ROLES:
                    # Move it where it belongs and drop the arg. Same rule the prompt
                    # states; enforcing it here covers the cases the prompt misses.
                    literal_spill.setdefault(role, name)
                    continue
                role = _ROLE_CANON.get(role, role)
            if not name or name.lower() in _NOISE:
                # A dialogue role as a participant carries no entity identity; drop
                # the arg but keep the frame (the rest of it is still a fact).
                continue
            if len(name.split()) > _MAX_ARG_WORDS:
                # A clause, not an entity — the prompt asks for <=4 words and this is
                # the backstop for when it is ignored. Such names never resolve against
                # another memory's phrasing, so they can only ever bloat the anchor
                # space. Drop the arg; the frame's other args still carry the fact.
                continue
            if name.lower() not in ents:
                ents[name.lower()] = ExtractedEntity(
                    name=name, entity_type=_norm_type(a.get("type")), description="")
            # Same role twice in one frame (two destinations) is legitimate; the
            # storage layer keys on (role, name) so both survive.
            args.append((role, name))
            seen_roles.add(role)
        if len(args) < 2:
            # A frame needs at least two participants to assert a relation. One-arg
            # frames are almost always an entity mention the extractor over-read.
            continue
        lits = _clean_literals(fr.get("literals"))
        for k, v in literal_spill.items():          # extractor-supplied literals win
            lits.setdefault(k, v)
        frames.append(Frame(predicate=pred, args=args, literals=lits))
    return FrameResult(entities=list(ents.values()), frames=frames)


_COVERAGE_QUOTED_RE = re.compile(r"[\"“]([^\"“”]{2,80})[\"”]")
_COVERAGE_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}\s+(?:January|February|March|April|May|"
    r"June|July|August|September|October|November|December)(?:\s*,?\s*(?:19|20)\d{2})?\b",
    re.I,
)
_COVERAGE_ISO_TIMESTAMP_RE = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
    re.I,
)
_COVERAGE_CLOCK_RE = re.compile(
    r"(?<!\d)\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[ap]\.?m\.?)?(?!\d)",
    re.I,
)
_COVERAGE_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")
_VISUAL_SOURCE_RE = re.compile(
    r"\b(?:photo|photograph|image|caption|poster|sign|label|screen|screenshot)\b",
    re.I,
)
_COVERAGE_WORD_RE = re.compile(r"[a-z0-9]+")
_COVERAGE_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "of", "on", "or",
    "she", "that", "the", "their", "they", "this", "to", "was", "were", "with",
    "photo", "photograph", "image", "caption", "shared", "showed", "shows",
}

FRAME_REPAIR_PROMPT = f"""Repair missing coverage in extracted knowledge frames.

Return only JSON in the same schema as the primary extractor:
{{"frames":[{{"predicate":"...","args":[{{"role":"...","name":"...","type":"..."}}],"literals":{{}}}}]}}

Extract only assertions supported by the source. Focus on the listed missing
specifics, including exact titles, names, dates, numbers, signs, labels, colors,
and concrete visual attributes. Preserve dates/counts as literals. Use the same
entity and role rules as the primary extraction. Do not repeat unrelated facts.
Entity TYPE must be one of: {", ".join(ENTITY_TYPES)}.
"""

FRAME_V724_EVENT_SUFFIX = """

V7.24 EVENT/STATE CONTRACT:
- For an occurrence, preserve its event time in literals using `event_time` or
  `event_date`; do not substitute the time when the memory was mentioned.
- Add literal `status=planned` for intended, scheduled, hoped-for, or proposed
  actions, and `status=completed` only for actions asserted to have happened.
- Keep separately timed occurrences as separate frames. Do not infer completion
  from a plan and do not invent an exact event time.
"""


def _frame_result_text(result: FrameResult) -> str:
    parts: list[str] = []
    for frame in result.frames:
        parts.append(frame.predicate)
        parts.extend(name for _role, name in frame.args)
        parts.extend(str(value) for value in frame.literals.values())
    return " ".join(parts).lower()


def _coverage_gaps(text: str, result: FrameResult) -> list[str]:
    """Find deterministic source specifics missing from a frame result.

    This is a retry gate, not a correctness oracle. It intentionally checks only
    details whose absence is mechanically observable.
    """

    source = str(text or "")
    rendered = _frame_result_text(result)
    rendered_words = set(_COVERAGE_WORD_RE.findall(rendered))
    # Session provenance is commonly rendered as an ISO timestamp. Its digits are
    # not independent quantities (``2023-07-17T14:31:00`` must never produce six
    # repair targets). The graph already stores this provenance on the memory and
    # fact, so protect the whole timestamp and any other clock expressions from the
    # exact-number coverage check.
    protected_spans = [
        match.span() for pattern in (_COVERAGE_ISO_TIMESTAMP_RE, _COVERAGE_CLOCK_RE)
        for match in pattern.finditer(source)
    ]

    def overlaps_protected(start: int, end: int) -> bool:
        return any(start < protected_end and end > protected_start
                   for protected_start, protected_end in protected_spans)

    quoted_candidates = [
        match.group(1).strip() for match in _COVERAGE_QUOTED_RE.finditer(source)
    ]

    covered_spans = list(protected_spans)
    date_candidates: list[str] = []
    for match in _COVERAGE_DATE_RE.finditer(source):
        if overlaps_protected(*match.span()):
            continue
        date_candidates.append(match.group(0).strip())
        covered_spans.append(match.span())

    number_candidates: list[str] = []
    for match in _COVERAGE_NUMBER_RE.finditer(source):
        start, end = match.span()
        if any(start < covered_end and end > covered_start
               for covered_start, covered_end in covered_spans):
            continue
        number_candidates.append(match.group(0).strip())

    def missing_candidates(candidates: list[str]) -> list[str]:
        out: list[str] = []
        for candidate in candidates:
            words = [
                word for word in _COVERAGE_WORD_RE.findall(candidate.lower())
                if word not in _COVERAGE_STOP
            ]
            if words and not all(word in rendered_words for word in words):
                out.append(candidate)
        return out

    quoted_missing = missing_candidates(quoted_candidates)
    date_missing = missing_candidates(date_candidates)
    number_missing = missing_candidates(number_candidates)

    visual_missing = False
    if _VISUAL_SOURCE_RE.search(source):
        source_words = {
            word for word in _COVERAGE_WORD_RE.findall(source.lower())
            if word not in _COVERAGE_STOP and len(word) > 2
        }
        if source_words:
            coverage = len(source_words & rendered_words) / len(source_words)
            visual_missing = coverage < 0.45

    # A session/event date is repeated in many generated memories and is already
    # preserved by PG provenance plus the fact timestamp. A missing date alone must
    # not cause a second LLM call for every one of those memories. If a high-signal
    # quote, independent quantity, or visual detail already requires repair, include
    # the missing date in that same bounded call at no additional request cost.
    if not quoted_missing and not number_missing and not visual_missing:
        return []

    missing: list[str] = [*quoted_missing, *date_missing, *number_missing]
    if visual_missing:
        missing.append("concrete visual/source details")

    return list(dict.fromkeys(value for value in missing if value))[:12]


def _merge_frame_results(primary: FrameResult, repair: FrameResult) -> FrameResult:
    entities: dict[str, ExtractedEntity] = {
        entity.name.strip().casefold(): entity
        for entity in primary.entities
        if entity.name and entity.name.strip()
    }
    for entity in repair.entities:
        if entity.name and entity.name.strip():
            entities.setdefault(entity.name.strip().casefold(), entity)

    frames: list[Frame] = []
    seen: set[tuple] = set()
    for frame in [*primary.frames, *repair.frames]:
        key = (
            frame.predicate.strip().casefold(),
            tuple(sorted((role, name.strip().casefold()) for role, name in frame.args)),
            tuple(sorted((str(k), str(v)) for k, v in frame.literals.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        frames.append(frame)
    return FrameResult(entities=list(entities.values()), frames=frames)


async def extract_frames(text: str, *, model: str, api_key: Optional[str] = None,
                         api_base: Optional[str] = None) -> FrameResult:
    """Text -> typed entities + n-ary frames, with one bounded v7.23 repair."""
    if not text or not text.strip():
        return FrameResult()
    try:
        prompt = FRAME_PROMPT + (
            FRAME_V724_EVENT_SUFFIX if settings.graph_version == "v7.24" else ""
        )
        raw = await call_openai_chat(prompt, text, model,
                                     api_key=api_key, api_base=api_base)
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame extraction failed: %s", exc)
        return FrameResult()
    res = _parse(raw)
    if settings.graph_version in ("v7.23", "v7.24"):
        gaps = _coverage_gaps(text, res)
        if gaps:
            try:
                repair_raw = await call_openai_chat(
                    FRAME_REPAIR_PROMPT,
                    "Missing specifics: " + json.dumps(gaps, ensure_ascii=False)
                    + "\n\nSource:\n" + text,
                    model,
                    api_key=api_key,
                    api_base=api_base,
                    temperature=0.0,
                )
                res = _merge_frame_results(res, _parse(repair_raw))
                logger.info("v7.23 frame coverage repair requested for: %s", gaps)
            except Exception as exc:  # noqa: BLE001
                logger.warning("v7.23 frame coverage repair failed: %s", exc)
    if res.frames:
        widths = [f.arity for f in res.frames]
        logger.info("frame extraction: %d entities, %d frames (arity max %d, mean %.1f) "
                    "from %d chars", len(res.entities), len(res.frames),
                    max(widths), sum(widths) / len(widths), len(text))
    return res
