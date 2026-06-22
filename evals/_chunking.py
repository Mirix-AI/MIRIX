"""Sentence-aware token-budgeted chunker, aligned with official MemoryAgentBench.

Port of ``utils/eval_other_utils.chunk_text_into_sentences`` from
https://github.com/HUST-AI-HYZ/MemoryAgentBench. Used by ruler_eval,
lru_eval, and longmem_eval so the ingest chunks all four MAB datasets
(RULER QA1/QA2, LongMemEval-S, infbench_sum, detective_qa) see are
the same shape the leaderboard runs use:

  - **Atom**: sentence (NLTK ``punkt``).
  - **Budget**: 4096 tokens of the ``gpt-4o-mini`` tiktoken encoding —
    the value pinned by every MAB ``data_conf/**/*.yaml`` we've seen.
  - **Bundle rule**: greedy pile sentences into the current chunk
    until adding the next one would push the token count over budget,
    then flush and start a new chunk with that sentence.
  - **No mid-sentence split**: a sentence larger than the budget still
    ships in its own chunk (matches the official behaviour — they have
    no special case for this either).

Why this matters: char-based 4096 (the previous policy) emitted ~4×
more chunks than official because 4096 chars ≈ 1024 tokens. That made
MIRIX's retrieval look worse than apples-to-apples because each
semantic unit was scattered across multiple memories.
"""

from __future__ import annotations

from typing import List

import nltk
import tiktoken

# Match the official MAB chunker exactly.
DEFAULT_TOKEN_MODEL = "gpt-4o-mini"
DEFAULT_CHUNK_TOKENS = 4096


def _ensure_nltk() -> None:
    """Make sure the NLTK punkt tokenizer is available.

    Best-effort — NLTK ships a chatty downloader; quiet=True keeps the
    eval logs clean. If the network is offline and punkt isn't cached
    yet, the eval will surface the LookupError at first sent_tokenize
    call, which is the same failure mode the official MAB script has.
    """
    try:
        nltk.data.find("tokenizers/punkt_tab")
        return
    except LookupError:
        pass
    try:
        nltk.data.find("tokenizers/punkt")
        return
    except LookupError:
        pass
    # punkt_tab is the modern split file; older NLTK versions still ship
    # 'punkt'. Download both so it works on either.
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


_ensure_nltk()
_ENCODING = tiktoken.encoding_for_model(DEFAULT_TOKEN_MODEL)


def chunk_text_into_sentences(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
) -> List[str]:
    """Bundle sentences greedily into <= ``chunk_size``-token chunks.

    Returns a list of plain strings (each chunk's sentences joined by
    a single space, matching the official script). Empty input returns
    an empty list.
    """
    if not text:
        return []

    sentences = nltk.sent_tokenize(text)
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    for sentence in sentences:
        st = len(_ENCODING.encode(sentence, allowed_special={"<|endoftext|>"}))
        if buf and buf_tokens + st > chunk_size:
            chunks.append(" ".join(buf))
            buf = [sentence]
            buf_tokens = st
        else:
            buf.append(sentence)
            buf_tokens += st

    if buf:
        chunks.append(" ".join(buf))
    return chunks
