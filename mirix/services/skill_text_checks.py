"""Content checks of the text a skill tool writes (pure CPU, no I/O).

The skill tools (``skill_create`` / ``skill_edit``) are called by a model, and a
weak model can fill a field with a placeholder (``"..."``, ``"string"``), with the
words of the tool's own parameter description (a "schema echo": ``"kebab-case
name"``, ``"detailed plain text"``), or with a stub that says it is one ("This
placeholder skill ... please delete it"). Such text must never become memory,
and a write that carries it must never retire the experiences it cites. This
module finds those values (:func:`placeholder_problem`) and measures what an
edit really changes (:func:`added_words`, :func:`dropped_sentences`,
:func:`numbering_problems`), so the tools can reject the first and report the
rest. They run inline in the tools' coroutines, so each one is linear in its
input, or bounded by a named work cap (:data:`LINE_DIFF_MAX_CELLS`): a weak
model's phrase loop ("check the value. " x 700) must never stall the event
loop (a word-level ``difflib`` diff of such a text took seconds to minutes).

The checks compare a value with the tool's own vocabulary (its parameter
descriptions, the field's name) and with generic placeholder words: nothing
here knows any task, domain or benchmark.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set, Tuple

#: Fewest content words a skill's description / instructions may have: a
#: description is "one or two sentences", and three words is the shortest
#: clause that says anything ("Retry failed uploads").
SKILL_MIN_DESCRIPTION_WORDS = 3
SKILL_MIN_INSTRUCTIONS_WORDS = 3
#: A value of at most this many words made only of the tool's vocabulary is a
#: schema echo (real content that short still names something the tool's
#: description of the field does not).
ECHO_MAX_WORDS = 8
#: ... and of at least this many: one word that happens to occur in the
#: parameter's description ("done" for an example's output described as "what
#: was done") is real content; a lone filler word is a placeholder
#: (:data:`PLACEHOLDER_VALUES`).
ECHO_MIN_WORDS = 2
#: A value of at most this many words that declares itself a placeholder
#: ("to be filled in later") is one; longer text may merely mention the word.
SELF_DECLARED_MAX_WORDS = 20
#: Content words an edit must add for its citations to count (a single word
#: such as "not" or "never", or a number such as "3" -> "5", can change what a
#: skill says; a deletion, a renumbering or a whitespace change adds none).
SKILL_CITATION_MIN_ADDED_WORDS = 1
#: :func:`added_words` aligns the changed lines of two texts with ``difflib``
#: only while (old lines) x (new lines) of the changed region is at most this;
#: a larger region is compared as one block. ``SequenceMatcher`` is
#: super-linear on repeated items: 141 x 282 identical lines take ~30 ms, while
#: a word-level diff of a 12k-character phrase loop took 20 s. A skill's
#: instructions (at most ``SKILL_MAX_INSTRUCTIONS_CHARS``) have ~150 lines, so
#: a whole rewrite still gets a line alignment.
LINE_DIFF_MAX_CELLS = 40_000
#: A sentence of the old text counts as kept by a rewrite when a sentence of the
#: new text shares at least this share of its words (Jaccard of the word sets).
#: Only the advice text of a rewrite depends on it.
REWRITE_KEPT_OVERLAP = 0.6
#: Sentences shorter than this (in words) are ignored by the rewrite report.
REWRITE_MIN_SENTENCE_WORDS = 4
#: Numbering warnings reported at most.
MAX_NUMBERING_PROBLEMS = 10

# Han, kana and hangul have no spaces between words: each character counts as
# one word, so a CJK description is not "one word long".
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿가-힯"
_CJK_CHAR = re.compile(f"[{_CJK}]")
_WORD_CHARS = re.compile(r"[^\W_]+")
# Other scripts written without spaces between words (Thai, Lao, Tibetan,
# Myanmar, Khmer, Tai Le, New Tai Lue, Tai Tham): their letters count one
# word per :data:`SPACELESS_LETTERS_PER_WORD`, their average word length.
_SPACELESS = (
    "\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u1780-\u17ff\u1950-\u19df\u1a20-\u1aaf"
)
_SPACELESS_RUN = re.compile(f"[{_SPACELESS}]+")
SPACELESS_LETTERS_PER_WORD = 4

#: Values that are placeholders, not content (compared after normalization:
#: lowercase, punctuation and quotes stripped, spaces collapsed): JSON-schema
#: type names, generic filler, "not applicable" markers. Words a real name or
#: trigger can be ("test", "json", "list", "number", "key", "field", "type")
#: are not: a skill for running tests may be named "test".
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "string",
        "str",
        "text",
        "value",
        "values",
        "placeholder",
        "todo",
        "tbd",
        "tbc",
        "n a",
        "na",
        "none",
        "null",
        "nil",
        "undefined",
        "empty",
        "lorem ipsum",
        "etc",
        "xxx",
        "foo",
        "bar",
        "example",
        "sample",
        "dummy",
        "object",
        "array",
        "integer",
        "int",
        "float",
        "boolean",
        "bool",
        "dict",
    }
)

#: Words that describe a value's form, or fill a slot, rather than say
#: anything. With the field's own parameter description and its name, a short
#: value made only of them echoes the schema. Type names (json, list, ...) and
#: other fields' names are deliberately NOT here: short real content uses them
#: ("json-output", a trigger "list of values"); as a whole value they are
#: already placeholders (:data:`PLACEHOLDER_VALUES`).
SCHEMA_WORDS = frozenset(
    {
        "kebab",
        "snake",
        "camel",
        "case",
        "lowercase",
        "uppercase",
        "hyphen",
        "hyphenated",
        "identifier",
        "id",
        "slug",
        "format",
        "short",
        "concise",
        "detailed",
        "plain",
        "one",
        "two",
        "sentence",
        "word",
        "your",
        "here",
        "insert",
        "fill",
        "later",
        "placeholder",
        "todo",
        "tbd",
        "example",
        "sample",
        "dummy",
        "foo",
        "bar",
        "xxx",
        "lorem",
        "ipsum",
        "etc",
        "string",
        "text",
        "value",
        "the",
        "a",
        "an",
        "of",
        "or",
        "and",
        "to",
        "in",
        "for",
        "with",
        "is",
        "it",
        "this",
        "that",
        "when",
        "what",
        "how",
        "use",
        "useful",
        "relevant",
        "apply",
        "something",
        "thing",
    }
)

# Text that says of ITSELF that it is a placeholder or unfinished ("This
# placeholder skill ...", "To be filled in later", "TODO: ..."); text that
# merely mentions placeholders ("Replace the placeholder text ...") does not.
_SELF_DECLARED = [
    re.compile(p)
    for p in (
        # "This placeholder skill ...", "This is a placeholder description",
        # or a text that opens with "A placeholder entry" (not "Fill a
        # placeholder description field", which is about something else).
        r"(?i)(\bthis\s+(is\s+(a|an|just\s+a)\s+)?|^\s*(a|an)\s+)placeholder\s+"
        r"(skill|entry|description|instructions?|trigger|example)\b",
        r"(?i)^\s*(this\s+(is|was)\s+)?(a|an|the|just\s+a)?\s*placeholder\s*"
        r"([.,;:!\-\u2014(]|$|for\b|until\b|only\b)",
        # "To be filled in later", "Details to be added later": the text
        # itself, not "Dates to be decided later are marked ...".
        r"(?i)(^\s*|\b(this|it|these|they|details?|steps?|content|instructions?|"
        r"description|section|rest|more)\s+(is\s+|are\s+|will\s+be\s+)?)to\s+be\s+"
        r"(filled|expanded|completed|written|added|updated|defined|determined|"
        r"provided|decided)(\s+(in|out))?\s+later\b",
        # "We will expand this later"; "Fill it in later." as the whole
        # instruction (not "Save a draft and complete it later").
        r"(?i)\b(fill|expand|complete|write|update|add)\s+(this|these)"
        r"(\s+(in|out))?\s+later\b",
        r"(?i)^\s*(please\s+)?(fill|expand|complete|write|update|add)\s+(it|them)"
        r"(\s+(in|out))?\s+later\b",
        r"(?i)\blorem\s+ipsum\b",
        # A marker, not a word of the text: "TODO: write the steps", "TBD",
        # "FIXME - ..." (not "TODO list triage: ...").
        r"^\s*(TODO|TBD|TBC|FIXME)\s*([:\-\u2014.!]|$)",
    )
]

# A line's numbering: a heading mark (with a dotted section number: "### 2.1
# Setup"), a list marker ("3.", "2)", "2.1.", "(1)", "a.", "(b)", "iv.",
# "-") and a "Step N" prefix ("Step 3:", "**Step 3.**", "### Step 3", "- Step 3
# -"; also "Phase", "Stage" and "Part", and "第N步" / "第N阶段"). A list marker
# needs its "." or ")" and a space after it, so a quantity ("3.5 kg", "3
# retries") is no marker. "Step N" counts only with a separator after it or
# alone on its line: "Step 3 of the build" and "Retry step 3" name a step,
# "Step 1.5 kg" is a quantity, and "第1步的输出" is "the output of step 1".
_NUMBERING_PREFIX = re.compile(
    r"^\s*(?:#{1,6}\s+(?:\d{1,4}(?:\.\d{1,4})+[.)]?\s+)?)?"
    r"(?:(?:\d{1,4}(?:\.\d{1,4})*[.)]|\((?:\d{1,4}|[a-z]|[ivx]{1,6})\)"
    r"|(?:[a-z]|[ivx]{1,6})[.)]|[-*•+])\s+)?"
    r"(?:[*_]{0,2}(?:step|phase|stage|part)\s+\d{1,4}[*_]{0,2}\s*"
    r"(?:[:.)\-\u2013\u2014][*_]{0,2}(?:\s+|$)|$)"
    r"|第\s*(?:\d{1,4}|[一二三四五六七八九十百零〇两]{1,4})\s*(?:步|阶段|部分|章|节)"
    r"\s*(?:[:：.．、,，)）\-\u2013\u2014]\s*|$))?",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
_NUMBERED_LINE = re.compile(r"^([ \t]*)(\d{1,4})[.)][ \t]+\S")


def content_words(text: str) -> List[str]:
    """The words of ``text`` that can carry content: whitespace tokens that
    contain a letter, with every CJK character counted as a word of its own
    and a run of another script written without spaces (Thai, Khmer, ...)
    as one word per :data:`SPACELESS_LETTERS_PER_WORD` letters."""
    words: List[str] = []
    for token in str(text or "").split():
        for run in _SPACELESS_RUN.findall(token):
            letters = sum(1 for ch in run if ch.isalpha())
            count = -(-letters // SPACELESS_LETTERS_PER_WORD)  # ceil
            if count:
                size = -(-len(run) // count)
                words.extend(run[i : i + size] for i in range(0, len(run), size))
        for part in _SPACELESS_RUN.sub(" ", token).split():
            cjk = _CJK_CHAR.findall(part)
            if cjk:
                words.extend(cjk)
                rest = _CJK_CHAR.sub(" ", part)
                words.extend(t for t in rest.split() if any(ch.isalpha() for ch in t))
            elif any(ch.isalpha() for ch in part):
                words.append(part)
    return words


def _norm_word(word: str) -> str:
    """A word (letters/digits, see :data:`_WORD_CHARS`) for comparison:
    lowercase, a trailing plural "s" dropped (``sentences`` = ``sentence``)."""
    word = word.lower()
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


_SCHEMA_VOCAB = frozenset(_norm_word(w) for w in SCHEMA_WORDS)


def vocabulary(text: str) -> Set[str]:
    """The comparison words of ``text`` (see :func:`_norm_word`), e.g. of a
    tool's parameter description."""
    return {_norm_word(w) for w in _WORD_CHARS.findall(str(text or ""))}


def normalized_value(text: str) -> str:
    """``text`` lowercased, with punctuation and quotes removed and spaces
    collapsed ("  'N/A'. " -> "n a")."""
    return " ".join(w.lower() for w in _WORD_CHARS.findall(str(text or "")))


def _echo_words(text: str) -> List[str]:
    words = [_norm_word(w) for w in _WORD_CHARS.findall(str(text or ""))]
    return [w for w in words if len(w) > 1 or w.isdigit()]


def placeholder_problem(
    field: str,
    value: str,
    schema_words: Iterable[str] = (),
    *,
    min_words: int = 0,
) -> Optional[str]:
    """Why ``value`` is not real content for ``field`` (a sentence the tool
    shows the model), or None.

    * fewer than ``min_words`` content words (:func:`content_words`);
    * a generic placeholder (:data:`PLACEHOLDER_VALUES`) or the field's name;
    * a short value (<= :data:`ECHO_MAX_WORDS` words) made only of the words of
      ``schema_words`` (the field's parameter description: a text, or its
      words), the field's name and :data:`SCHEMA_WORDS`: an echo of the tool's
      schema;
    * a short value (<= :data:`SELF_DECLARED_MAX_WORDS` words) that declares
      itself a placeholder ("to be filled in later", TODO, TBD).
    """
    text = str(value or "")
    words = content_words(text)
    if min_words and len(words) < min_words:
        return (
            f"'{field}' has {len(words)} word(s) of content; it needs at least "
            f"{min_words}. Write the actual {field} of this skill."
        )
    normalized = normalized_value(text)
    # The field's own words, without an index ("examples[0].input" -> "examples input").
    field_label = re.sub(r"\[\d*\]", " ", field).replace("_", " ").replace(".", " ")
    field_words = normalized_value(field_label)
    if (
        normalized in PLACEHOLDER_VALUES
        or normalized == field_words
        or normalized in field_words.split()
    ):
        return (
            f"'{field}' is a placeholder ({text.strip()[:40]!r}), not content. "
            f"Write the actual {field} of this skill."
        )
    echo = _echo_words(text)
    if ECHO_MIN_WORDS <= len(echo) <= ECHO_MAX_WORDS:
        allowed = (
            _SCHEMA_VOCAB
            | vocabulary(
                schema_words
                if isinstance(schema_words, str)
                else " ".join(schema_words)
            )
            | vocabulary(field_label)
        )
        if all(w in allowed for w in echo):
            return (
                f"'{field}' only repeats the words of the tool's description of "
                f"that parameter ({text.strip()[:60]!r}), not content. Write the "
                f"actual {field} of this skill."
            )
    if len(words) <= SELF_DECLARED_MAX_WORDS and any(
        p.search(text) for p in _SELF_DECLARED
    ):
        return (
            f"'{field}' says it is a placeholder or unfinished ({text.strip()[:60]!r}). "
            f"Write the actual {field} now, or leave the skill unchanged."
        )
    return None


_STRIP_CHARS = "\"'`.,;:!?()[]{}<>*_~"


def _line_tokens(line: str) -> Tuple[str, ...]:
    """The words of one line for a diff: its numbering (a list marker "3." /
    "-", a "Step 3:" prefix, :data:`_NUMBERING_PREFIX`) dropped, then every
    whitespace token that holds a letter or a digit, lowercase and stripped
    of surrounding punctuation ("It." = "it"), with every CJK character a
    word of its own. Numbers count: "3 times" -> "5 times" changes what a
    skill says, and only the line's numbering is a renumbering."""
    out: List[str] = []
    for token in _NUMBERING_PREFIX.sub("", line, count=1).split():
        cjk = _CJK_CHAR.findall(token)
        if cjk:
            out.extend(cjk)
            token = _CJK_CHAR.sub(" ", token)
        for word in token.split():
            stripped = word.strip(_STRIP_CHARS).lower()
            if any(ch.isalnum() for ch in stripped):
                out.append(stripped)
    return tuple(out)


def _diff_lines(text: str) -> List[Tuple[str, ...]]:
    """The non-empty lines of ``text`` as :func:`_line_tokens` tuples."""
    lines = (_line_tokens(line) for line in str(text or "").splitlines())
    return [line for line in lines if line]


def added_words(old: str, new: str) -> Tuple[int, int]:
    """``(added, removed)``: how many words ``new`` inserts (or puts in place
    of others) relative to ``old``, and how many of ``old``'s it removes or
    replaces (words as :func:`_line_tokens`: numbers count, list markers,
    "Step N:" prefixes, punctuation and case do not). A renumbering or a
    reflow adds 0 words; a line moved elsewhere adds its words, and so do
    words swapped within or across lines ("lint before tests" -> "tests
    before lint").

    Linear in the texts' size, apart from a line alignment capped by
    :data:`LINE_DIFF_MAX_CELLS`: the common leading and trailing lines are
    skipped, the changed lines are aligned with ``difflib`` (lines, not words:
    a word diff is super-linear on a phrase loop), and each changed block
    counts the words its new side has more of than its old side (a multiset
    difference, so a reflow within the block adds nothing). A block whose two
    sides hold the same words in another order counts, as added and as
    removed, each word whose position in the block's word sequence changed
    (:func:`_reordered_words`; for a word moved far, an upper bound of what a
    word diff would count)."""
    a, b = _diff_lines(old), _diff_lines(new)
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    end_a, end_b = len(a), len(b)
    while end_a > start and end_b > start and a[end_a - 1] == b[end_b - 1]:
        end_a -= 1
        end_b -= 1
    a, b = a[start:end_a], b[start:end_b]
    if not a and not b:
        return 0, 0
    if a and b and len(a) * len(b) <= LINE_DIFF_MAX_CELLS:
        matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        blocks = [
            (a[i1:i2], b[j1:j2])
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
            if tag != "equal"
        ]
    else:
        blocks = [(a, b)]
    added = removed = 0
    for old_lines, new_lines in blocks:
        old_words = Counter(word for line in old_lines for word in line)
        new_words = Counter(word for line in new_lines for word in line)
        if old_words == new_words:
            moved = _reordered_words(old_lines, new_lines)
            added += moved
            removed += moved
            continue
        added += sum((new_words - old_words).values())
        removed += sum((old_words - new_words).values())
    return added, removed


def _reordered_words(
    old_lines: Sequence[Tuple[str, ...]], new_lines: Sequence[Tuple[str, ...]]
) -> int:
    """How many positions of a changed block's word sequence hold another
    word after the edit, the two sides having the same words (a multiset):
    0 for a reflow (the same words in the same order, broken into other
    lines), 2 for a swap of two words. Linear in the block's size."""
    old_seq = [word for line in old_lines for word in line]
    new_seq = [word for line in new_lines for word in line]
    return sum(1 for x, y in zip(old_seq, new_seq) if x != y)


def _sentences(text: str) -> List[str]:
    return [
        s.strip() for s in _SENTENCE_SPLIT.split(str(text or "")) if s and s.strip()
    ]


def _norm_sentence(sentence: str) -> str:
    """A sentence for comparison: numbering (:data:`_NUMBERING_PREFIX`)
    dropped, normalized, and every CJK character a word of its own (a Chinese
    sentence has no spaces, so it would otherwise be one "word", under
    :data:`REWRITE_MIN_SENTENCE_WORDS`)."""
    text = normalized_value(_NUMBERING_PREFIX.sub("", sentence, count=1))
    return " ".join(_CJK_CHAR.sub(lambda m: f" {m.group(0)} ", text).split())


def dropped_sentences(old: str, new: str) -> Tuple[List[str], int]:
    """``(dropped, total)``: the sentences of ``old`` (at least
    :data:`REWRITE_MIN_SENTENCE_WORDS` words; ``total`` counts them) that
    ``new`` no longer has, not even reworded: a sentence is kept when its
    normalized text occurs in ``new`` or it shares at least
    :data:`REWRITE_KEPT_OVERLAP` of its words with a sentence of ``new``."""
    new_norm = _norm_sentence(new)
    new_sets = [set(_norm_sentence(s).split()) for s in _sentences(new)]
    new_sets = [s for s in new_sets if s]
    dropped: List[str] = []
    total = 0
    for sentence in _sentences(old):
        norm = _norm_sentence(sentence)
        if len(norm.split()) < REWRITE_MIN_SENTENCE_WORDS:
            continue
        total += 1
        if norm and norm in new_norm:
            continue
        words = set(norm.split())
        if any(
            len(words & other) / len(words | other) >= REWRITE_KEPT_OVERLAP
            for other in new_sets
        ):
            continue
        dropped.append(sentence)
    return dropped, total


def numbering_problems(text: str) -> List[str]:
    """Numbered-list defects of ``text``: a number repeated ("6 repeated") or
    a step skipped or out of order ("3→10"). A list restarts at 1; lists at a
    deeper indentation are numbered on their own. At most
    :data:`MAX_NUMBERING_PROBLEMS`."""
    problems: List[str] = []
    last: dict = {}  # indentation -> the last number seen at that level
    for line in str(text or "").splitlines():
        match = _NUMBERED_LINE.match(line)
        if not match:
            continue
        indent = len(match.group(1).expandtabs(4))
        number = int(match.group(2))
        for deeper in [k for k in last if k > indent]:
            del last[deeper]  # a deeper list ended
        previous = last.get(indent)
        if number == 1 or previous is None:
            last[indent] = number
            continue
        if number == previous:
            problems.append(f"{number} repeated")
        elif number != previous + 1:
            problems.append(f"{previous}→{number}")
        last[indent] = number
        if len(problems) >= MAX_NUMBERING_PROBLEMS:
            break
    return problems


def list_text(value: Sequence) -> str:
    """The words of a triggers / examples list as text, for :func:`added_words`
    (an example's input and output; any other item as its string)."""
    parts: List[str] = []
    for item in value or ():
        if isinstance(item, dict):
            parts.extend(str(v) for v in item.values())
        else:
            parts.append(str(item))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Citation quotes                                                               #
# --------------------------------------------------------------------------- #

#: A citation's quote (each part of an elided one counted together with the
#: longest) with fewer letters and digits than this cannot point at the text
#: that applies a lesson: a word or two occurs in any skill. The same length
#: the distiller asks of a quote that grounds an outcome (twelve: "check the
#: lid" is 11, "check the label" 13).
CITATION_QUOTE_MIN_CHARS = 12
#: A CJK character counts as this many letters toward the minimum.
CITATION_QUOTE_CJK_WEIGHT = 3

_QUOTE_NON_WORD = re.compile(r"[\W_]+")
_QUOTE_ELLIPSIS = re.compile(r"\.{3,}|…")

#: Why a citation's quote does not count (:func:`citation_quote_problem`).
QUOTE_MISSING = "no quote given"
QUOTE_TOO_SHORT = "quote too short to show where"
QUOTE_NOT_FOUND = "quote not found in the text this change adds"
QUOTE_UNCHANGED = "quote is text the skill already had"


def squash_letters(text) -> str:
    """``text`` in Unicode NFKC form, casefolded, with everything but
    letters, digits and combining marks removed (a script such as Devanagari
    writes its vowels as marks: "पहले लेबल जांचें" keeps them). Linear in the
    text."""
    import unicodedata

    normal = unicodedata.normalize("NFKC", str(text or "")).casefold()
    if normal.isascii():
        return _QUOTE_NON_WORD.sub("", normal)
    return "".join(
        ch for ch in normal if ch.isalnum() or unicodedata.category(ch).startswith("M")
    )


def squash_quote_text(text: str) -> str:
    """``text`` squashed (:func:`squash_letters`): a quote matches whatever
    its spacing, punctuation, case or line breaks."""
    return squash_letters(text)


def _quote_weight(squashed: str) -> int:
    return len(squashed) + (CITATION_QUOTE_CJK_WEIGHT - 1) * len(
        _CJK_CHAR.findall(squashed)
    )


def quote_weight(text: str) -> int:
    """How much ``text`` says, as a citation quote's minimum counts it: its
    letters and digits (squashed, :func:`squash_quote_text`), a CJK
    character counting :data:`CITATION_QUOTE_CJK_WEIGHT`."""
    return _quote_weight(squash_quote_text(text))


def citation_quote_problem(quote, new_text: str, old_text: str = "") -> Optional[str]:
    """None when ``quote`` shows the text a skill change added or changed —
    every part of it (split at an ellipsis) occurs in ``new_text`` (squashed,
    :func:`squash_quote_text`), its longest part has at least
    :data:`CITATION_QUOTE_MIN_CHARS` letters and digits, and a part that long
    does not occur in ``old_text`` (the text the change replaced, "" for a new
    skill) — else why not (:data:`QUOTE_MISSING`, :data:`QUOTE_TOO_SHORT`,
    :data:`QUOTE_NOT_FOUND`, :data:`QUOTE_UNCHANGED`). Linear in the texts."""
    if not isinstance(quote, str) or not quote.strip():
        return QUOTE_MISSING
    parts = [squash_quote_text(p) for p in _QUOTE_ELLIPSIS.split(quote)]
    parts = [p for p in parts if p]
    if not parts or max(_quote_weight(p) for p in parts) < CITATION_QUOTE_MIN_CHARS:
        return QUOTE_TOO_SHORT
    new = squash_quote_text(new_text)
    if not all(p in new for p in parts):
        return QUOTE_NOT_FOUND
    old = squash_quote_text(old_text) if old_text else ""
    if not any(
        _quote_weight(p) >= CITATION_QUOTE_MIN_CHARS and p not in old for p in parts
    ):
        return QUOTE_UNCHANGED
    return None
