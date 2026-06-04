"""
Substring-match evaluator for RULER-style QA (and any other dataset
whose ``answers`` are short, exact-form needles like ``'France'`` or
``'42'``).

Mirrors the official MemoryAgentBench ``substring_exact_match`` metric:
the response is correct iff it contains (as a case-insensitive substring)
at least one of the gold answers.

Usage from ``organize_results.py``:

    from llm_judge_substring import evaluate_substring_judge
    score = evaluate_substring_judge(predicted_answer, expected_answer)

No LLM call; deterministic; effectively free.
"""

import re
import string
from typing import Iterable, Union


def normalize_answer(text: str) -> str:
    """Port of MemoryAgentBench ``utils/eval_other_utils.normalize_answer``.

    Pipeline (verbatim from official):
      1. lowercase
      2. strip punctuation
      3. drop articles (a / an / the) at word boundaries
      4. collapse whitespace
    """
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _accepted_answers(expected: Union[str, Iterable[str], None]) -> list[str]:
    """Normalize the gold-answer field into a flat list of strings.

    ``expected`` may be:
      * a list of strings (the raw HF shape, e.g. ``['France','France']``)
      * a string built by ``flatten_answer`` (``'France; France'``)
      * None / empty
    """
    if expected is None:
        return []
    if isinstance(expected, str):
        # Flattened ``"a; b; c"`` form.
        return [seg.strip() for seg in expected.split(";") if seg.strip()]
    # Iterable[str] — callers pass a flat list of accepted answers.
    return [str(item).strip() for item in expected if str(item).strip()]


def evaluate_substring_judge(
    predicted: Union[str, None],
    expected: Union[str, Iterable[str], None],
) -> int:
    """Return 1 if ``predicted`` contains any accepted answer, else 0.

    Matches the official ``substring_exact_match`` metric in
    ``MemoryAgentBench/utils/eval_other_utils.py``: both prediction and
    each accepted answer are passed through ``normalize_answer``
    (lowercase, strip punctuation, drop articles, collapse whitespace)
    before substring containment is checked. Score is the max over all
    accepted answers (drqa-style max-over-ground-truths).
    """
    if not predicted:
        return 0
    accepted = _accepted_answers(expected)
    if not accepted:
        return 0
    pred_norm = normalize_answer(predicted)
    for ans in {normalize_answer(a) for a in accepted}:
        if ans and ans in pred_norm:
            return 1
    return 0
