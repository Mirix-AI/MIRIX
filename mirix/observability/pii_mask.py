"""Cheap synchronous PII backstop for the Langfuse ``mask=`` callback.

The Langfuse Python SDK accepts a ``mask`` callable on the ``Langfuse``
constructor and invokes it **synchronously, at span start, on whatever
thread opens the span** (OTel captures attributes by value at set-time).
On MIRIX's save/search path that thread is the event-loop thread, so the
callback must be cheap and must NOT do network I/O — a blocking ispy-pii
``httpx.Client.post()`` here starves the loop.

Real PII masking therefore happens **upstream** now: the LLM/embedding
generation sites pre-redact their span input/output via
:func:`mirix.pii.mask_structure` (cooperative async ispy-pii) *before* the
value becomes a span attribute. By the time this callback runs, the data is
already masked, so this is just a **synchronous safety net**.

What this callback does:
- Walks ``dict``/``list``/``tuple``/``str`` (so any attribute that was NOT
  pre-masked — e.g. a future hand-rolled span — still gets covered).
- Applies a **fast local regex scrub** for the obvious high-risk tokens
  (email / SSN / phone). Pure CPU, zero network, no caching needed.
- Never raises: a logging/observability path that re-enters its own
  exception handler is a deadlock waiting to happen.

The Langfuse client wiring (see ``langfuse_client.py``) reads the mask via
:func:`get_langfuse_mask` at construction time, defaulting to the backstop
(:func:`ispy_pii_mask`) when no override has been registered. Downstream
consumers can register a different — but still **synchronous, non-network** —
callable via :func:`set_langfuse_mask` before calling ``initialize_langfuse()``.

The kill switch ``MIRIX_LANGFUSE_MASK_ENABLED=false`` turns the callback
into a passthrough. Defaults to enabled.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Final, Optional

# Re-export the placeholder so the public surface is unchanged for any
# importer that referenced it from this module.
from mirix.pii import REDACTED_PLACEHOLDER  # noqa: F401

# Local, pure-CPU regex scrubs for the obvious high-risk tokens. These are a
# backstop only — primary masking is upstream via ispy-pii. Kept deliberately
# conservative (clear shapes) so the synchronous callback stays cheap and has
# negligible false-positive risk on the loop thread.
_EMAIL_RE: Final = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN_RE: Final = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# US phone shapes: optional +1, separators of space/dot/dash, optional parens.
_PHONE_RE: Final = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)")

_EMAIL_TOKEN: Final = "[EMAIL]"
_SSN_TOKEN: Final = "[SSN]"
_PHONE_TOKEN: Final = "[PHONE]"


def _enabled() -> bool:
    return os.getenv("MIRIX_LANGFUSE_MASK_ENABLED", "true").lower() == "true"


def _scrub_text(text: str) -> str:
    """Local regex scrub of email / SSN / phone. Pure CPU, never raises.

    Order matters: SSN before phone so a ``ddd-dd-dddd`` shape is tokenized
    as an SSN rather than being partially consumed by the phone pattern.
    """
    if not text:
        return text
    try:
        scrubbed = _EMAIL_RE.sub(_EMAIL_TOKEN, text)
        scrubbed = _SSN_RE.sub(_SSN_TOKEN, scrubbed)
        scrubbed = _PHONE_RE.sub(_PHONE_TOKEN, scrubbed)
        return scrubbed
    except Exception:
        # Defensive: a pathological input must not break the span export.
        return text


def build_langfuse_mask() -> Callable[..., Any]:
    """Construct the Langfuse-compatible synchronous backstop mask.

    Returns a function with signature ``mask(data, **kwargs) -> Any`` that
    the Langfuse SDK invokes for every value it would export. Strings are
    run through the local regex scrub; ``dict``/``list``/``tuple`` are walked
    recursively (shapes preserved); other scalars pass through unchanged.

    Pure CPU — holds no client, opens no socket, sleeps for nothing. Real
    masking is done upstream (:func:`mirix.pii.mask_structure`); this is only
    a safety net for attributes that bypassed it.
    """

    def mask(data: Any = None, **_: Any) -> Any:
        if not _enabled():
            return data
        if isinstance(data, str):
            return _scrub_text(data)
        if isinstance(data, dict):
            return {k: mask(data=v) for k, v in data.items()}
        if isinstance(data, list):
            return [mask(data=item) for item in data]
        if isinstance(data, tuple):
            return tuple(mask(data=item) for item in data)
        return data

    return mask


_default_mask_singleton: Optional[Callable[..., Any]] = None


def _default_mask() -> Callable[..., Any]:
    """Build the backstop masker on first use, then cache per process."""
    global _default_mask_singleton
    if _default_mask_singleton is None:
        _default_mask_singleton = build_langfuse_mask()
    return _default_mask_singleton


def ispy_pii_mask(data: Any = None, **kwargs: Any) -> Any:
    """Default synchronous backstop mask.

    Named ``ispy_pii_mask`` for backward compatibility with the public
    surface; the real ispy-pii masking now happens upstream
    (:func:`mirix.pii.mask_structure`). Suitable for direct registration via
    :func:`set_langfuse_mask` or as the default consulted by
    :func:`get_langfuse_mask` when nothing has been registered.
    """
    return _default_mask()(data=data, **kwargs)


# Module-level singleton holding the active mask callable. Exposed via
# set_langfuse_mask / get_langfuse_mask so downstream consumers can register
# their own callable before initialize_langfuse() constructs the Langfuse
# client. Module-level (not pydantic-settings) because Callable isn't an
# env-driven setting type.
_active_mask: Optional[Callable[..., Any]] = None


def set_langfuse_mask(fn: Optional[Callable[..., Any]]) -> None:
    """Register the mask callable that initialize_langfuse() will use.

    Pass ``None`` to clear the registration; the next call to
    :func:`get_langfuse_mask` will return the synchronous backstop
    (:func:`ispy_pii_mask`).
    """
    global _active_mask
    _active_mask = fn


def get_langfuse_mask() -> Callable[..., Any]:
    """Return the active mask callable.

    Falls back to the synchronous backstop (:func:`ispy_pii_mask`) when
    nothing has been registered. The ``Langfuse(...)`` constructor reads from
    this function, so ``set_langfuse_mask`` must be called before
    ``initialize_langfuse()`` to take effect.
    """
    return _active_mask or ispy_pii_mask
