"""Union-coverage gate for consolidation merges.

The auto_dream agent "merges" memories by rewriting N rows into M<N new rows and
hard-deleting the originals — so anything the rewrite omits is destroyed. The
system prompt asks the model to preserve every specific, but a prompt is a soft
constraint fighting the agent's primary remove-redundancy objective; measured on
LoCoMo conv-26, single-occurrence specifics (a purchase date, a festival year, a
band name) kept vanishing.

This module makes the constraint MECHANICAL: before a replace/update executes,
the deterministic specifics of the old rows (numbers, dates, month/weekday
names, word-numbers, multi-word proper names) must all appear in the replacement
text, else the tool call is rejected and the error steers the model to rewrite.

Deliberately NOT gated: single lowercase content words ("sunflowers") — no
deterministic extractor separates them from prose without mass false rejections;
that residue stays prompt-guarded. The gate covers the quantifiable specifics,
which is where the measured QA damage concentrated (temporal + counting).
"""
import os
import re
from typing import Iterable

# Multi-word proper-noun phrase: >=2 capitalised words, optionally joined by a
# lowercase connector ("House of Blues", "Summer Sounds", "Downtown Farmers
# Market"). Same shape the cold-fact extractor uses. Case-sensitive on purpose.
_PROPER = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|the|and|&)?\s*[A-Z][a-zA-Z]+)+\b")

# Digit-bearing token: 420, 3.5, 1,000, 2023, 19:30. Trailing punctuation excluded.
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
# Word-numbers that carry answerable quantities. The very common/ambiguous ones
# (one, first, second, couple, several) are excluded — requiring them would
# reject legitimate rewording far more often than it would save an answer.
_WORD_NUMBERS = {
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "twenty", "thirty", "forty", "fifty", "dozen", "twice",
}
_WORDS = re.compile(r"[a-z]+")


def _normalize(text: str) -> str:
    # lowercase, thousands-commas stripped, whitespace collapsed — so "1,000" in
    # the original matches "1000" in the rewrite and line breaks don't matter.
    t = re.sub(r"(?<=\d),(?=\d)", "", (text or "").lower())
    return re.sub(r"\s+", " ", t)


def extract_specifics(text: str) -> dict:
    """Map of match-key -> display form for every deterministic specific in text."""
    out: dict = {}
    if not text:
        return out
    for m in _PROPER.finditer(text):
        phrase = re.sub(r"\s+", " ", m.group(0))
        out[_normalize(phrase)] = phrase
    for m in _NUMBER.finditer(text):
        tok = m.group(0).rstrip(".,:")
        if tok:
            out[_normalize(tok)] = tok
    for w in _WORDS.findall((text or "").lower()):
        if w in _MONTHS or w in _WEEKDAYS or w in _WORD_NUMBERS:
            out[w] = w
    return out


def merge_item_text(item, fields) -> str:
    """Concatenate an LLM-provided new_item's text fields (dict or pydantic)."""
    vals = []
    for f in fields:
        v = item.get(f, "") if isinstance(item, dict) else getattr(item, f, "")
        vals.append(str(v or ""))
    return "\n".join(vals)


def enforce_merge_coverage(agent, old_texts, new_texts, tool_name: str) -> None:
    """Reject a consolidation merge that would destroy specifics.

    The auto_dream agent "merges" by rewriting N rows into M<N and hard-deleting
    the originals, so anything the rewrite omits is destroyed — and the system
    prompt's preserve-everything instruction is only a soft constraint (measured
    on LoCoMo: single-occurrence dates/names kept vanishing). This makes the
    union requirement MECHANICAL: every deterministic specific of the old rows
    must appear in the replacement text, else the call raises BEFORE any
    deletion and the error steers the model to rewrite (or to keep a superseded
    value as history, or not to merge at all).

    Scoped to the auto_dream agent — ingest-time updates (e.g. knowledge
    updates that legitimately supersede an old value) are not gated. Disable
    with MIRIX_MERGE_COVERAGE_GATE=0.
    """
    if os.environ.get("MIRIX_MERGE_COVERAGE_GATE", "1") == "0":
        return
    try:
        from mirix.schemas.agent import AgentType

        if not agent.agent_state.is_type(AgentType.auto_dream_agent):
            return
    except Exception:  # noqa: BLE001 — never let the gate itself break a tool
        return

    gaps = coverage_gaps(old_texts, "\n".join(new_texts))
    if gaps:
        shown = ", ".join(repr(g) for g in gaps[:12])
        more = f" (+{len(gaps) - 12} more)" if len(gaps) > 12 else ""
        raise ValueError(
            f"{tool_name} REJECTED — nothing was deleted. The replacement text drops these "
            f"specifics that exist in the originals: {shown}{more}. A merge must preserve every "
            f"named entity, date, number and quantity VERBATIM. Rewrite new_items to include all "
            f"of them; if a value is outdated and superseded, keep it in details as history "
            f"(e.g. 'previously ...'). If these entries are not actually redundant, do not merge "
            f"them."
        )


def coverage_gaps(old_texts: Iterable[str], new_text: str,
                  label_texts: Iterable[str] = ()) -> list:
    """As above, but ``label_texts`` (a memory's ``name``/title field) are checked
    at COMPONENT level instead of phrase level.

    A title is a filing label, not an assertion: a consolidated entry gets a new
    title of its own and cannot be expected to restate "Exploring New Genres
    Beyond Thrillers" verbatim. Demanding the whole phrase made the gate reject
    43% of clusters on a real store — and the bigger the cluster (more titles),
    the likelier the rejection, which is exactly backwards.

    Dropping titles from the check entirely is not safe either: measured on the
    same store, 32.6% of the specifics found in ``name`` appear NOWHERE in that
    row's summary/details ("Citi AAdvantage Executive Credit Card", "The
    Mandalorian TV Series"), so ignoring titles would open a real fact-loss hole.

    So titles contribute their COMPONENTS: every capitalised word and every
    number inside a title must still appear in the replacement, while the exact
    title string need not. Generic filing words ("overview", "options",
    "recommendations", ...) are dropped — they classify, they do not assert.
    """
    gaps: dict = {}
    for key, display in _gap_items(old_texts, new_text):
        gaps.setdefault(key, display)
    for key, display in _label_gap_items(label_texts, new_text,
                                         corroborating=old_texts):
        gaps.setdefault(key, display)
    return sorted(gaps.values())


# Words that only ever classify a memory rather than assert something about the
# user. Stripped from title component checks.
_LABEL_STOPWORDS = {
    "overview", "options", "option", "recommendations", "recommendation",
    "tips", "guide", "guides", "guidelines", "ideas", "idea", "suggestions",
    "suggestion", "notes", "note", "list", "lists", "summary", "details",
    "info", "information", "topics", "topic", "misc", "general", "other",
    "beyond", "for", "and", "of", "the", "in", "on", "at", "with", "as",
    "plan", "plans", "preferences", "preference", "interests", "interest",
    "activities", "activity", "experience", "experiences", "discussion",
    "discussions", "challenge", "challenges", "series", "location", "locations",
}


def _label_gap_items(label_texts: Iterable[str], new_text: str,
                     corroborating: Iterable[str] = ()):
    """Component-level check for title fields; yields (key, display) gaps.

    Title-Case makes every word look like a proper noun, so a component is only
    demanded when it is *corroborated* as a real name: it appears capitalised
    mid-sentence somewhere in the bodies (real names are capitalised wherever
    they occur; ordinary words like "Book" or "Exploring" are not), or it is a
    number/date. Everything else in a title is treated as filing vocabulary.
    """
    new_norm = _normalize(new_text)
    new_words = set(_WORDS.findall(new_norm))
    corroborated = set()
    for body in corroborating:
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]+)", body or ""):
            corroborated.add(m.group(1).lower())
    for text in label_texts:
        if not text:
            continue
        for key, display in extract_specifics(text).items():
            if " " in key:
                # Phrase from a title -> require its meaningful components only.
                for comp in key.split():
                    if comp in _LABEL_STOPWORDS or len(comp) < 3:
                        continue
                    if not (comp in corroborated or comp in _MONTHS
                            or comp in _WEEKDAYS or comp in _WORD_NUMBERS):
                        continue  # Title-Case artifact, not a name
                    if comp not in new_words:
                        yield comp, comp
            elif not key.isalpha():
                if key not in new_norm:
                    yield key, display
            else:
                if key in _LABEL_STOPWORDS:
                    continue
                if not (key in corroborated or key in _MONTHS
                        or key in _WEEKDAYS or key in _WORD_NUMBERS):
                    continue
                if key not in new_words:
                    yield key, display


def _gap_items(old_texts: Iterable[str], new_text: str) -> list:
    """(key, display) pairs for specifics in old_texts missing from new_text.

    Containment is checked on normalized text; word-level keys (months,
    weekdays, word-numbers) require a word-boundary match so "may" the month
    does not get satisfied by "maybe".
    """
    new_norm = _normalize(new_text)
    new_words = set(_WORDS.findall(new_norm))
    _connectors = {"of", "the", "and"}
    gaps: dict = {}
    for text in old_texts:
        for key, display in extract_specifics(text).items():
            if key in gaps:
                continue
            if " " in key:
                # Proper phrase: covered by the exact phrase, OR by every
                # non-connector component word appearing somewhere — so
                # "Caroline and Melanie" is satisfied by separate mentions of
                # Caroline and Melanie, without demanding the conjunction.
                if key in new_norm:
                    continue
                comps = [w for w in key.split() if w not in _connectors]
                if comps and all(c in new_words for c in comps):
                    continue
                gaps[key] = display
            elif not key.isalpha():
                if key not in new_norm:
                    gaps[key] = display
            else:
                if key not in new_words:
                    gaps[key] = display
    return list(gaps.items())
