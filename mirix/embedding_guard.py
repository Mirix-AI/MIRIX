"""Embedding length guard.

Every embedding model MIRIX uses comes from the ``embedding_model()`` factory
(``mirix/embeddings.py``), which wraps the provider's model in
:class:`GuardedEmbedding`. So no call site sends an embedding provider more
than its input limit, and none relies on how a provider treats an overlong
input (silent truncation, an error):

* **Limit** — the model's input token limit, from ``MIRIX_EMBEDDING_TOKEN_LIMITS``
  (default ``{"gemini-embedding-001": 2048}``); a model missing from the table
  is treated as 2,048 tokens, with a warning. A provider prefix is ignored
  (``google/gemini-embedding-001`` is ``gemini-embedding-001``).
  ``EmbeddingConfig.embedding_chunk_size`` (300 by default) is NOT used.
* **Chunks** — at most ``limit * 0.88`` tokens each, rounded down to a whole
  hundred (gemini-embedding-001: 1,800), counted with the ``o200k_base``
  approximation of :mod:`mirix.services.context_budget`. A text within the cap
  is embedded as is. A longer one is cut at paragraph boundaries first
  (whole paragraphs packed together), a paragraph still too long at sentence
  (and line) boundaries, a sentence still too long at a hard token boundary.
  Nothing is dropped: the chunks put together are the text.
* **At most ``MIRIX_EMBEDDING_MAX_CHUNKS`` (16) chunks** — a longer text is
  represented by its first and last chunks (half each) with a warning, so one
  input can never cost thousands of sequential paid calls.
* **Merge** — one embedding per chunk, then, in this order: 1. the mean
  weighted by each chunk's token count; 2. L2 normalization; 3. zero padding
  to ``MAX_EMBEDDING_DIM`` (4096) LAST, so chunk vectors of different sizes
  are an error instead of a silently misaligned mean. A text embedded whole
  keeps the provider's vector, padded to 4096 the same way.
* **Metrics** — every chunked call is counted, once and only once it
  succeeded (a retried call is not counted again), in the enclosing
  :func:`counting_embedding_chunks` block: a learning run counts into its
  ``skill_learning_run.embedding_chunked_calls`` (``SkillLearningRunManager.execute``);
  a retrieval reads its block's counter for its retrieval log.

Splitting is pure CPU work and runs in a worker thread (``asyncio.to_thread``);
a text within the cap is only counted (large texts are counted off the event
loop too).
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from mirix.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "EmbeddingChunkCounter",
    "GuardedEmbedding",
    "counting_embedding_chunks",
    "embedding_chunk_cap",
    "embedding_token_limit",
    "merge_chunk_embeddings",
    "pad_embedding",
    "record_chunked_embedding_call",
    "split_text_for_embedding",
]

TokenCount = Callable[[str], int]


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #

_warned_unknown_models: set = set()


def embedding_token_limit(
    model_name: Optional[str], limits: Optional[Mapping[str, int]] = None
) -> int:
    """The input token limit of ``model_name`` (``MIRIX_EMBEDDING_TOKEN_LIMITS``).

    Looked up as given, then without a provider prefix (the part after the
    last ``/``). A model not in the table gets ``DEFAULT_EMBEDDING_TOKEN_LIMIT``
    (2,048) and one warning per model name.
    """
    from mirix import constants

    table = constants.EMBEDDING_TOKEN_LIMITS if limits is None else limits
    name = str(model_name or "")
    for candidate in (name, name.rsplit("/", 1)[-1]):
        if candidate in table:
            return int(table[candidate])
    default = int(constants.DEFAULT_EMBEDDING_TOKEN_LIMIT)
    if name not in _warned_unknown_models:
        _warned_unknown_models.add(name)
        logger.warning(
            "Embedding model %r has no token limit in MIRIX_EMBEDDING_TOKEN_LIMITS; "
            "using %d tokens",
            name,
            default,
        )
    return default


def embedding_chunk_cap(limit: int, fraction: Optional[float] = None) -> int:
    """Tokens per chunk: ``limit * EMBEDDING_CHUNK_FRACTION`` (0.88), rounded
    down to a whole hundred (2,048 -> 1,800); at least 1."""
    from mirix import constants

    share = constants.EMBEDDING_CHUNK_FRACTION if fraction is None else fraction
    raw = int(int(limit) * float(share))
    if raw >= 100:
        return raw - raw % 100
    return max(1, raw)


# --------------------------------------------------------------------------- #
# Splitting                                                                    #
# --------------------------------------------------------------------------- #

# A blank line (with whatever whitespace around it) ends a paragraph.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r\f\v]*\n\s*")
# A sentence ends at . ! ? … (closing quotes / brackets included) before
# whitespace, right after a CJK full stop, question or exclamation mark, or at
# a line break.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+|(?<=[。！？])\s*|\n\s*")
_LEVELS = (_PARAGRAPH_BREAK, _SENTENCE_BREAK)
# Upper bound for the characters of one hard-split chunk, per token of the
# cap: only the binary search's range, never a limit on the result's validity.
_MAX_CHARS_PER_TOKEN = 16
_WHITESPACE = re.compile(r"\s")


def _cut_after(text: str, pattern: "re.Pattern[str]") -> List[str]:
    """``text`` cut right after every match of ``pattern`` (the separator stays
    with the piece before it): ``"".join(pieces) == text``."""
    pieces: List[str] = []
    start = 0
    for match in pattern.finditer(text):
        end = match.end()
        if end <= start:
            continue
        pieces.append(text[start:end])
        start = end
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _hard_split(text: str, cap: int, count: TokenCount) -> List[str]:
    """``text`` in consecutive pieces of at most ``cap`` tokens: each the
    longest prefix that fits (binary search), moved back to the last
    whitespace within its final fifth when there is one."""
    out: List[str] = []
    rest = text
    while rest:
        if count(rest) <= cap:
            out.append(rest)
            break
        lo, hi = 1, min(len(rest), cap * _MAX_CHARS_PER_TOKEN)
        if count(rest[:hi]) <= cap:
            cut = hi
        else:
            # count(rest[:lo]) fits (one character), count(rest[:hi]) does not.
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if count(rest[:mid]) <= cap:
                    lo = mid
                else:
                    hi = mid
            cut = lo
        floor = max(1, (cut * 4) // 5)
        for position in range(cut - 1, floor - 1, -1):
            if _WHITESPACE.match(rest, position):
                cut = position + 1
                break
        out.append(rest[:cut])
        rest = rest[cut:]
    return out


def _split(text: str, cap: int, count: TokenCount, level: int) -> List[str]:
    """Pieces of ``text`` of at most ``cap`` tokens each, cut at the
    boundaries of ``_LEVELS[level]`` (whole pieces packed together while they
    fit), a piece still too long at the next level's, the last resort a hard
    token boundary."""
    if level >= len(_LEVELS):
        return _hard_split(text, cap, count)
    pieces = _cut_after(text, _LEVELS[level])
    if len(pieces) <= 1:
        return _split(text, cap, count, level + 1)
    out: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        chunk = "".join(current)
        if len(current) > 1 and count(chunk) > cap:
            # Joined, the pieces tokenized longer than their sum (a merge at a
            # boundary): each piece on its own fits.
            out.extend(current)
        else:
            out.append(chunk)
        current.clear()

    for piece in pieces:
        tokens = count(piece)
        if tokens > cap:
            flush()
            current_tokens = 0
            out.extend(_split(piece, cap, count, level + 1))
            continue
        # One token of slack per boundary: pieces are counted one by one.
        if current and current_tokens + tokens + 1 > cap:
            flush()
            current_tokens = 0
        current.append(piece)
        current_tokens += tokens + (1 if len(current) > 1 else 0)
    flush()
    return out


def split_text_for_embedding(
    text: str, cap: int, count: TokenCount
) -> List[Tuple[str, int]]:
    """``text`` as ``[(chunk, tokens), ...]`` of at most ``cap`` tokens each
    (see the module docstring). A text within the cap is one chunk, itself.
    Chunks of whitespace only are left out (unless nothing else is left)."""
    if cap < 1:
        raise ValueError(f"the chunk cap must be at least 1 token, not {cap}")
    total = count(text)
    if total <= cap:
        return [(text, total)]
    chunks = _split(text, cap, count, 0)
    kept = [chunk for chunk in chunks if chunk.strip()] or chunks[:1]
    return [(chunk, max(1, count(chunk))) for chunk in kept]


# --------------------------------------------------------------------------- #
# Merging                                                                      #
# --------------------------------------------------------------------------- #


def pad_embedding(vector: Sequence[float], dim: Optional[int] = None) -> List[float]:
    """``vector`` zero-padded to ``dim`` (``MAX_EMBEDDING_DIM``); longer is an
    error (it cannot be stored)."""
    from mirix.constants import MAX_EMBEDDING_DIM

    size = MAX_EMBEDDING_DIM if dim is None else int(dim)
    values = [float(v) for v in vector]
    if len(values) > size:
        raise ValueError(
            f"the embedding has {len(values)} dimensions, more than the "
            f"{size} MIRIX stores"
        )
    return values + [0.0] * (size - len(values))


def merge_chunk_embeddings(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
    dim: Optional[int] = None,
) -> List[float]:
    """One vector for a chunked text, in this order:

    1. the mean of the chunk ``vectors`` weighted by ``weights`` (their token
       counts);
    2. L2 normalization (a zero mean stays zero);
    3. zero padding to ``dim`` (``MAX_EMBEDDING_DIM``) — LAST: chunk vectors
       of different sizes are refused instead of being averaged misaligned.
    """
    if not vectors:
        raise ValueError("no chunk embeddings to merge")
    if len(vectors) != len(weights):
        raise ValueError(f"{len(vectors)} chunk embeddings but {len(weights)} weights")
    arrays = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    shape = arrays[0].shape
    if len(shape) != 1 or any(array.shape != shape for array in arrays):
        raise ValueError(
            "chunk embeddings of different dimensions: "
            f"{sorted({array.shape for array in arrays})}"
        )
    weight = np.asarray(weights, dtype=np.float64)
    if not np.all(weight > 0):
        raise ValueError(f"chunk weights must be positive, got {list(weights)}")
    mean = (np.stack(arrays) * weight[:, None]).sum(axis=0) / weight.sum()
    norm = float(np.linalg.norm(mean))
    if norm > 0:
        mean = mean / norm
    return pad_embedding(mean.tolist(), dim)


# --------------------------------------------------------------------------- #
# Counting chunked calls                                                       #
# --------------------------------------------------------------------------- #


class EmbeddingChunkCounter:
    """The chunked embedding calls made inside a
    :func:`counting_embedding_chunks` block."""

    def __init__(self) -> None:
        #: Embedding calls whose input had to be chunked.
        self.chunked_calls = 0
        #: Chunks those calls embedded.
        self.chunks = 0

    def record(self, chunks: int) -> None:
        self.chunked_calls += 1
        self.chunks += int(chunks)

    def as_metrics(self) -> Dict[str, int]:
        return {
            "embedding_chunked_calls": self.chunked_calls,
            "embedding_chunks": self.chunks,
        }


_active_counters: contextvars.ContextVar[Tuple[EmbeddingChunkCounter, ...]] = (
    contextvars.ContextVar("mirix_embedding_chunk_counters", default=())
)


@contextmanager
def counting_embedding_chunks(
    counter: Optional[EmbeddingChunkCounter] = None,
) -> Iterator[EmbeddingChunkCounter]:
    """Count every chunked embedding call made inside this block (and in the
    tasks started inside it, which copy the context). Blocks nest: a call
    counts in every enclosing block."""
    counter = counter if counter is not None else EmbeddingChunkCounter()
    token = _active_counters.set(_active_counters.get() + (counter,))
    try:
        yield counter
    finally:
        _active_counters.reset(token)


def record_chunked_embedding_call(chunks: int) -> None:
    """Count one chunked call (of ``chunks`` chunks) in every enclosing
    :func:`counting_embedding_chunks` block; outside any, nothing."""
    for counter in _active_counters.get():
        counter.record(chunks)


# --------------------------------------------------------------------------- #
# The guard                                                                    #
# --------------------------------------------------------------------------- #


class GuardedEmbedding:
    """A provider embedding model behind the length guard.

    ``get_text_embedding(text)`` returns a ``MAX_EMBEDDING_DIM``-wide vector:
    the provider's own for a text within the chunk cap, the merged one (see
    :func:`merge_chunk_embeddings`) for a longer text, whose chunks are
    embedded one after the other. Any other attribute is the wrapped model's.
    """

    def __init__(
        self,
        inner: Any,
        *,
        model_name: Optional[str],
        token_limits: Optional[Mapping[str, int]] = None,
        token_counter: Any = None,
    ):
        self.inner = inner
        self.model_name = model_name
        self.token_limit = embedding_token_limit(model_name, token_limits)
        self.chunk_cap = embedding_chunk_cap(self.token_limit)
        self._token_counter = token_counter

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the guard itself lacks.
        if name.startswith("__"):
            raise AttributeError(name)
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def _counter(self) -> Any:
        if self._token_counter is not None:
            return self._token_counter
        from mirix.services.context_budget import get_token_counter

        return get_token_counter()

    async def plan_chunks(self, text: str) -> List[Tuple[str, int]]:
        """``[(chunk, tokens), ...]`` for ``text``: itself when it fits."""
        if not isinstance(text, str) or not text:
            return [(text, 0)]
        counter = self._counter()
        tokens = await counter.acount_text(text)
        if tokens <= self.chunk_cap:
            return [(text, tokens)]
        return await asyncio.to_thread(
            split_text_for_embedding, text, self.chunk_cap, counter.count_text
        )

    async def get_text_embedding(self, text: str) -> List[float]:
        chunks = await self.plan_chunks(text)
        if len(chunks) == 1:
            only = chunks[0][0]
            vector = await self.inner.get_text_embedding(only)
            if only != text:
                # The text was over the cap and all but one of its chunks were
                # whitespace: that chunk alone is sent, never the whole text,
                # and the call counts as chunked.
                record_chunked_embedding_call(1)
            return pad_embedding(vector)
        from mirix import constants

        cap = max(1, int(constants.EMBEDDING_MAX_CHUNKS))
        planned = len(chunks)
        if planned > cap:
            # The head and the tail of the text stand for it.
            head = cap - cap // 2
            chunks = chunks[:head] + chunks[planned - cap // 2 :]
            logger.warning(
                "Embedding input of %d characters needs %d chunks of %d tokens; "
                "embedding only its first %d and last %d (MIRIX_EMBEDDING_MAX_CHUNKS=%d)",
                len(text),
                planned,
                self.chunk_cap,
                head,
                cap // 2,
                cap,
            )
        logger.info(
            "Embedding input of %d characters is above the %d-token chunk cap of %s "
            "(limit %d): embedding it in %d chunks",
            len(text),
            self.chunk_cap,
            self.model_name,
            self.token_limit,
            len(chunks),
        )
        vectors = []
        for chunk, _ in chunks:
            vectors.append(await self.inner.get_text_embedding(chunk))
        merged = merge_chunk_embeddings(vectors, [tokens for _, tokens in chunks])
        # Counted once the call succeeded: a retried call is one call.
        record_chunked_embedding_call(len(chunks))
        return merged
